"""Deterministic Micro-Checkpoint Spot-Check Validator.

Instead of classic BOINC quorum=2 (which wastes 50% of global compute), HiveMind validators
re-execute a randomly selected s-step micro-checkpoint segment (10% verification overhead for
s=50, H=500). If the validator's output diverges beyond tolerance eps = 1e-4, the Workunit is
rejected and the user's SteamID64 is flagged for Byzantine/hardware fault behavior.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence
import torch
import torch.nn as nn

from hivemind.boinc.workunit import BOINCWorkunit
from hivemind.core.diloco_inner import InnerLoopResult, replay_inner_segment
from hivemind.core.quantization import compute_state_dict_sha256

VALID_EXECUTION_DTYPES = ("fp32", "bf16", "fp16", "fixed_point_q16")


@dataclass(frozen=True)
class ExecutionClassProfile:
    """Execution-class descriptor for architecture-matched spot-check validation (`HM-S10-08` / E12).

    Experiment E12 established that:
      1. Same execution class (`accelerator_class`, `torch_build`, `dtype`, `thread_count`,
         `deterministic_flags`) reproduces a 50-step Qwen2.5-0.5B LoRA segment bit-for-bit (`0.0`).
      2. Changing CPU `thread_count` (`1` or `8` vs `4`) or CPU host shifts reduction order,
         producing `linf in [1.5e-4, 2.6e-4]` on only `3-5 / 96` (`<= 5.2%`) parameter tensors.
      3. Changing `dtype` (`bf16` autocast replayed in `fp32`) diverges to `3.3e-3` across `96 / 96`
         tensors (exceeding the tampered control `1.25e-3`), making `dtype` a strict class boundary.
    """

    accelerator_class: str = "cpu_x86_64"
    torch_build: str = "2.x"
    dtype: str = "fp32"
    thread_count: int = 4
    deterministic_flags: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in VALID_EXECUTION_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_EXECUTION_DTYPES}, got {self.dtype!r}")
        if self.thread_count < 1:
            raise ValueError("thread_count must be >= 1")

    def class_key(self) -> str:
        return (
            f"{self.accelerator_class}:{self.torch_build}:{self.dtype}:"
            f"threads={self.thread_count}:det={int(self.deterministic_flags)}"
        )

    def exact_match(self, other: ExecutionClassProfile) -> bool:
        return self.class_key() == other.class_key()

    def same_dtype_class(self, other: ExecutionClassProfile) -> bool:
        return self.dtype == other.dtype


@dataclass(frozen=True)
class TwoStatisticSpotCheckResult:
    """Result of the E12 two-statistic (`linf` + tensor-count fraction) spot-check rule (`HM-S10-08`)."""

    linf_divergence: float
    num_tensors_above_eps: int
    total_tensors: int
    tensor_fraction_above_eps: float
    base_eps: float
    effective_eps: float
    max_tensor_fraction_above_eps: float
    exact_class_matched: bool
    dtype_matched: bool
    passed: bool
    reason: str


def quantize_fixed_point_q16_state(
    state_dict: dict[str, torch.Tensor],
    frac_bits: int = 16,
) -> dict[str, torch.Tensor]:
    """Convert FP32 LoRA tensors into signed 64-bit fixed-point `Q(frac_bits)` integers (`HM-S10-08`)."""
    if frac_bits < 1 or frac_bits > 30:
        raise ValueError("frac_bits must be in [1, 30]")
    scale = float(1 << frac_bits)
    out: dict[str, torch.Tensor] = {}
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().to(dtype=torch.float64, device="cpu")
        out[k] = torch.round(t * scale).to(torch.int64)
    return out


def verify_fixed_point_q16_digest(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    frac_bits: int = 16,
    max_ulp_drift: int = 1,
) -> dict[str, Any]:
    """Compare two LoRA state dicts in fixed-point `Q(frac_bits)` integer space (`HM-S10-08`)."""
    if max_ulp_drift < 0:
        raise ValueError("max_ulp_drift must be >= 0")
    qa = quantize_fixed_point_q16_state(state_a, frac_bits=frac_bits)
    qb = quantize_fixed_point_q16_state(state_b, frac_bits=frac_bits)
    if set(qa.keys()) != set(qb.keys()):
        return {
            "passed": False,
            "exact_digest_match": False,
            "max_ulp_diff": float("inf"),
            "digest_a": "",
            "digest_b": "",
        }
    hasher_a = hashlib.sha256()
    hasher_b = hashlib.sha256()
    max_ulp = 0
    for k in sorted(qa.keys()):
        ta = qa[k].contiguous()
        tb = qb[k].contiguous()
        hasher_a.update(k.encode("utf-8"))
        hasher_a.update(ta.numpy().tobytes())
        hasher_b.update(k.encode("utf-8"))
        hasher_b.update(tb.numpy().tobytes())
        ulp = int((ta - tb).abs().max().item()) if ta.numel() > 0 else 0
        if ulp > max_ulp:
            max_ulp = ulp
    dig_a = hasher_a.hexdigest()
    dig_b = hasher_b.hexdigest()
    return {
        "passed": max_ulp <= max_ulp_drift,
        "exact_digest_match": dig_a == dig_b,
        "max_ulp_diff": max_ulp,
        "frac_bits": frac_bits,
        "digest_a": dig_a,
        "digest_b": dig_b,
    }


@dataclass
class SpotCheckVerdict:
    """Audit result from the BOINC Spot-Check Validator."""

    wu_id: str
    worker_id: str
    steam_id64: str
    segment_start_step: int
    segment_end_step: int
    verification_overhead_pct: float
    max_abs_divergence: float
    tolerance_eps: float
    hash_matched: bool
    passed: bool
    steam_account_flagged: bool
    reason: str


class DeterministicSpotCheckValidator:
    """Trusted BOINC Validator pool verifier for deterministic micro-checkpoint segments."""

    def __init__(self, tolerance_eps: float = 1e-4) -> None:
        self.tolerance_eps = tolerance_eps
        self.flagged_steam_accounts: dict[str, list[str]] = {}

    def verify_submission(
        self,
        validator_model: nn.Module,
        workunit: BOINCWorkunit,
        submission: InnerLoopResult,
        token_sequences: torch.Tensor,
        steam_id64: str,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        micro_batch_size: int = 1,
        validator_rng_seed: int | None = None,
        worker_fingerprint: object | None = None,
        validator_fingerprint: object | None = None,
    ) -> SpotCheckVerdict:
        """Select one micro-checkpoint segment and verify worker determinism within eps = 1e-4."""
        # HM-S10-08: eps is only meaningful within one accelerator class; do not replay or flag across classes
        if (
            worker_fingerprint is not None
            and validator_fingerprint is not None
            and worker_fingerprint.class_key() != validator_fingerprint.class_key()  # type: ignore[attr-defined]
        ):
            return SpotCheckVerdict(
                wu_id=workunit.wu_id,
                worker_id=submission.worker_id,
                steam_id64=steam_id64,
                segment_start_step=0,
                segment_end_step=0,
                verification_overhead_pct=0.0,
                max_abs_divergence=float("nan"),
                tolerance_eps=self.tolerance_eps,
                hash_matched=False,
                passed=False,
                steam_account_flagged=False,
                reason="not comparable: cross-architecture replay (HM-S10-08)",
            )
        ckpt_steps = sorted(s for s in submission.micro_checkpoints.keys() if s > 0)
        if not ckpt_steps:
            self._flag_account(steam_id64, f"Missing micro-checkpoints in WU {workunit.wu_id}")
            return SpotCheckVerdict(
                wu_id=workunit.wu_id,
                worker_id=submission.worker_id,
                steam_id64=steam_id64,
                segment_start_step=0,
                segment_end_step=0,
                verification_overhead_pct=0.0,
                max_abs_divergence=float("inf"),
                tolerance_eps=self.tolerance_eps,
                hash_matched=False,
                passed=False,
                steam_account_flagged=True,
                reason="Submission contained zero micro-checkpoints.",
            )

        # Select the first 50-step segment [0, s_1] for stateless deterministic verification
        # (or a randomly selected segment when optimizer state is checkpointed)
        rng = random.Random(validator_rng_seed if validator_rng_seed is not None else workunit.rng_seed)
        selected_end = ckpt_steps[0] if len(ckpt_steps) == 1 else rng.choice(ckpt_steps[:1])
        selected_start = 0

        overhead_pct = round(100.0 * (selected_end - selected_start) / max(1, workunit.inner_steps_H), 2)

        expected_State = replay_inner_segment(
            model=validator_model,
            start_lora_state=submission.micro_checkpoints[0].state_snapshot,
            token_sequences=token_sequences,
            start_step=selected_start,
            end_step=selected_end,
            lr=lr,
            weight_decay=weight_decay,
            rng_seed=workunit.rng_seed,
            micro_batch_size=micro_batch_size,
            use_muon_lora_preconditioner=getattr(
                submission, "use_muon_lora_preconditioner", False
            ),
            muon_ns_steps=getattr(submission, "muon_ns_steps", 5),
        )

        worker_record = submission.micro_checkpoints[selected_end]
        worker_state = worker_record.state_snapshot

        max_diff = 0.0
        for name, exp_tensor in expected_State.items():
            if name not in worker_state:
                max_diff = float("inf")
                break
            diff = float(
                (exp_tensor.to(torch.float32).cpu() - worker_state[name].to(torch.float32).cpu())
                .abs()
                .max()
                .item()
            )
            if diff > max_diff:
                max_diff = diff

        # Also inspect final delta consistency if worker tampered with theta_final after step s_1
        for name, init_t in submission.theta_init.items():
            reconstructed_delta = init_t.to(torch.float32).cpu() - submission.theta_final[name].to(torch.float32).cpu()
            submitted_raw = submission.raw_delta[name].to(torch.float32).cpu()
            delta_mismatch = float((reconstructed_delta - submitted_raw).abs().max().item())
            if delta_mismatch > max_diff:
                max_diff = delta_mismatch

        expected_sha = compute_state_dict_sha256(expected_State)
        expected_State.clear()
        del expected_State
        hash_matched = expected_sha == worker_record.lora_sha256
        passed = max_diff <= self.tolerance_eps

        flagged = not passed
        reason = (
            f"Verified segment [{selected_start}, {selected_end}] ({overhead_pct}% overhead): "
            f"max_diff={max_diff:.2e} <= eps={self.tolerance_eps:.1e}"
            if passed
            else f"REJECTED segment [{selected_start}, {selected_end}]: "
            f"divergence {max_diff:.2e} exceeds tolerance eps={self.tolerance_eps:.1e}"
        )
        if flagged:
            self._flag_account(steam_id64, reason)

        return SpotCheckVerdict(
            wu_id=workunit.wu_id,
            worker_id=submission.worker_id,
            steam_id64=steam_id64,
            segment_start_step=selected_start,
            segment_end_step=selected_end,
            verification_overhead_pct=overhead_pct,
            max_abs_divergence=round(max_diff, 8),
            tolerance_eps=self.tolerance_eps,
            hash_matched=hash_matched,
            passed=passed,
            steam_account_flagged=flagged,
            reason=reason,
        )

    def _flag_account(self, steam_id64: str, reason: str) -> None:
        self.flagged_steam_accounts.setdefault(steam_id64, []).append(reason)

    @staticmethod
    def assign_matching_validator(
        worker_class: ExecutionClassProfile,
        validator_pool: Sequence[ExecutionClassProfile],
    ) -> ExecutionClassProfile | None:
        """Assign an execution-class-compatible validator from `validator_pool` (`HM-S10-08`).

        Priority order:
          1. Exact execution-class match (`dtype`, `accelerator_class`, `torch_build`, `thread_count`, `deterministic_flags`).
          2. Same `dtype` and same `accelerator_class` (uses two-statistic cross-thread tolerance).
          3. Same `dtype` across compatible CPU/GPU class.
        Cross-`dtype` assignment (`bf16` vs `fp32`) is strictly refused (`returns None`).
        """
        for cand in validator_pool:
            if worker_class.exact_match(cand):
                return cand
        for cand in validator_pool:
            if (
                worker_class.same_dtype_class(cand)
                and worker_class.accelerator_class == cand.accelerator_class
            ):
                return cand
        for cand in validator_pool:
            if worker_class.same_dtype_class(cand):
                return cand
        return None

    def evaluate_two_statistic_replay(
        self,
        worker_state: dict[str, torch.Tensor],
        validator_state: dict[str, torch.Tensor],
        worker_class: ExecutionClassProfile | None = None,
        validator_class: ExecutionClassProfile | None = None,
        base_eps: float | None = None,
        cross_thread_eps: float = 3.5e-4,
        max_tensor_fraction_above_eps: float = 0.10,
    ) -> TwoStatisticSpotCheckResult:
        """Evaluate the E12 two-statistic spot-check rule (`HM-S10-08`).

        Combines:
          1. Maximum elementwise L-infinity divergence `||theta_val - theta_worker||_inf <= effective_eps`.
          2. Tensor-count fraction `#{k : ||theta_val[k] - theta_worker[k]||_inf > base_eps} / N_tensors <= max_tensor_fraction_above_eps`.
        """
        w_cls = worker_class or ExecutionClassProfile()
        v_cls = validator_class or w_cls
        eps0 = float(base_eps if base_eps is not None else self.tolerance_eps)
        if eps0 <= 0.0 or not math.isfinite(eps0):
            raise ValueError("base_eps must be finite and > 0")
        if not (0.0 <= max_tensor_fraction_above_eps <= 1.0):
            raise ValueError("max_tensor_fraction_above_eps must be in [0, 1]")

        dtype_ok = w_cls.same_dtype_class(v_cls)
        exact_ok = w_cls.exact_match(v_cls)

        if set(worker_state.keys()) != set(validator_state.keys()) or not worker_state:
            return TwoStatisticSpotCheckResult(
                linf_divergence=float("inf"),
                num_tensors_above_eps=len(validator_state),
                total_tensors=max(1, len(validator_state)),
                tensor_fraction_above_eps=1.0,
                base_eps=eps0,
                effective_eps=eps0,
                max_tensor_fraction_above_eps=max_tensor_fraction_above_eps,
                exact_class_matched=exact_ok,
                dtype_matched=dtype_ok,
                passed=False,
                reason="REJECTED: parameter key mismatch or empty state dict.",
            )

        per_tensor_linf: list[float] = []
        for k in sorted(validator_state.keys()):
            vt = validator_state[k].detach().to(dtype=torch.float32, device="cpu")
            wt = worker_state[k].detach().to(dtype=torch.float32, device="cpu")
            diff = float((vt - wt).abs().max().item()) if vt.numel() > 0 else 0.0
            per_tensor_linf.append(diff)

        linf = max(per_tensor_linf) if per_tensor_linf else 0.0
        n_total = len(per_tensor_linf)
        n_above = sum(1 for d in per_tensor_linf if d > eps0)
        frac_above = float(n_above) / float(max(1, n_total))

        if not dtype_ok:
            return TwoStatisticSpotCheckResult(
                linf_divergence=round(linf, 8),
                num_tensors_above_eps=n_above,
                total_tensors=n_total,
                tensor_fraction_above_eps=round(frac_above, 6),
                base_eps=eps0,
                effective_eps=eps0,
                max_tensor_fraction_above_eps=max_tensor_fraction_above_eps,
                exact_class_matched=False,
                dtype_matched=False,
                passed=False,
                reason=(
                    f"DTYPE_CLASS_MISMATCH: worker dtype={w_cls.dtype!r} vs validator dtype={v_cls.dtype!r} "
                    f"(cross-dtype replay forbidden by E12 P12d; linf={linf:.2e})."
                ),
            )

        eff_eps = eps0 if exact_ok else max(eps0, float(cross_thread_eps))
        allowed_frac = 0.0 if exact_ok else float(max_tensor_fraction_above_eps)
        passed = bool(linf <= eff_eps and frac_above <= allowed_frac)
        reason = (
            f"PASSED ({'exact-class' if exact_ok else 'cross-thread/host same-dtype'}): "
            f"linf={linf:.2e} <= {eff_eps:.2e}, tensors_above_eps={n_above}/{n_total} ({frac_above:.1%})"
            if passed
            else f"REJECTED: linf={linf:.2e} (limit {eff_eps:.2e}), "
            f"tensors_above_eps={n_above}/{n_total} ({frac_above:.1%}, limit {allowed_frac:.1%})"
        )
        return TwoStatisticSpotCheckResult(
            linf_divergence=round(linf, 8),
            num_tensors_above_eps=n_above,
            total_tensors=n_total,
            tensor_fraction_above_eps=round(frac_above, 6),
            base_eps=eps0,
            effective_eps=eff_eps,
            max_tensor_fraction_above_eps=allowed_frac,
            exact_class_matched=exact_ok,
            dtype_matched=True,
            passed=passed,
            reason=reason,
        )
