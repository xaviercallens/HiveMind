# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
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

try:
    import torch
    import torch.nn as nn
    from hivemind.boinc.workunit import BOINCWorkunit
    from hivemind.core.diloco_inner import InnerLoopResult, replay_inner_segment
    from hivemind.core.quantization import compute_state_dict_sha256
except ImportError:  # pragma: no cover - lightweight router container without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    BOINCWorkunit = Any  # type: ignore[assignment,misc]
    InnerLoopResult = Any  # type: ignore[assignment,misc]
    replay_inner_segment = None  # type: ignore[assignment]
    compute_state_dict_sha256 = None  # type: ignore[assignment]

VALID_EXECUTION_DTYPES = ("fp32", "bf16", "fp16", "fixed_point_q16")
NON_FINITE_TENSOR = "NON_FINITE_TENSOR"


def _has_non_finite(state: dict[str, torch.Tensor]) -> bool:
    """Return True when any element of any tensor in `state` is NaN or +/-inf (IP-238)."""
    return any(not bool(torch.isfinite(t.detach().to(device="cpu")).all()) for t in state.values())


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
        if not isinstance(self.accelerator_class, str) or not self.accelerator_class.strip():
            raise ValueError("accelerator_class must be a non-empty string")
        if not isinstance(self.torch_build, str) or not self.torch_build.strip():
            raise ValueError("torch_build must be a non-empty string")
        if self.dtype not in VALID_EXECUTION_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_EXECUTION_DTYPES}, got {self.dtype!r}")
        if isinstance(self.thread_count, bool) or not isinstance(self.thread_count, int) or self.thread_count < 1:
            raise ValueError("thread_count must be >= 1")
        if not isinstance(self.deterministic_flags, bool):
            raise ValueError("deterministic_flags must be a bool")

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
    if isinstance(frac_bits, bool) or not isinstance(frac_bits, int) or frac_bits < 1 or frac_bits > 30:
        raise ValueError("frac_bits must be in [1, 30]")
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict of tensors")
    scale = float(1 << frac_bits)
    out: dict[str, torch.Tensor] = {}
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().to(dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{NON_FINITE_TENSOR}: parameter {k!r} contains NaN or inf")
        out[k] = torch.round(t * scale).to(torch.int64)
    return out


def verify_fixed_point_q16_digest(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    frac_bits: int = 16,
    max_ulp_drift: int = 1,
) -> dict[str, Any]:
    """Compare two LoRA state dicts in fixed-point `Q(frac_bits)` integer space (`HM-S10-08`)."""
    if isinstance(frac_bits, bool) or not isinstance(frac_bits, int) or frac_bits < 1 or frac_bits > 30:
        raise ValueError("frac_bits must be in [1, 30]")
    if isinstance(max_ulp_drift, bool) or not isinstance(max_ulp_drift, int) or max_ulp_drift < 0:
        raise ValueError("max_ulp_drift must be >= 0")
    if _has_non_finite(state_a) or _has_non_finite(state_b):
        return {
            "passed": False,
            "exact_digest_match": False,
            "max_ulp_diff": float("inf"),
            "reason": NON_FINITE_TENSOR,
            "digest_a": "",
            "digest_b": "",
        }
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
        if (
            isinstance(tolerance_eps, bool)
            or not isinstance(tolerance_eps, (int, float))
            or not math.isfinite(float(tolerance_eps))
            or float(tolerance_eps) <= 0.0
        ):
            raise ValueError("tolerance_eps must be a positive finite number")
        self.tolerance_eps = float(tolerance_eps)
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
        expected_init_state: dict[str, torch.Tensor] | None = None,
    ) -> SpotCheckVerdict:
        """Select one micro-checkpoint segment and verify worker determinism within eps = 1e-4."""
        if hasattr(workunit, "shard_sha256") and (
            not str(getattr(workunit, "shard_sha256", "")).strip()
            or not str(getattr(workunit, "shard_hmac_signature", "")).strip()
        ):
            raise ValueError(
                f"workunit {getattr(workunit, 'wu_id', '')!r} requires non-empty shard_sha256 and shard_hmac_signature (IP-218)"
            )
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

        if 0 not in submission.micro_checkpoints:
            self._flag_account(steam_id64, f"Missing step-0 micro-checkpoint in WU {workunit.wu_id}")
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
                reason="Submission missing required step-0 micro-checkpoint.",
            )

        # Select a micro-checkpoint segment [selected_start, selected_end] via validator RNG (IP-76)
        rng = random.Random(validator_rng_seed if validator_rng_seed is not None else workunit.rng_seed)
        seg_idx = 0 if len(ckpt_steps) == 1 else rng.randrange(len(ckpt_steps))
        selected_end = ckpt_steps[seg_idx]
        selected_start = 0 if seg_idx == 0 else ckpt_steps[seg_idx - 1]

        overhead_pct = round(100.0 * (selected_end - selected_start) / max(1, workunit.inner_steps_H), 2)

        step0_record = submission.micro_checkpoints[0]
        step0_sha = compute_state_dict_sha256(step0_record.state_snapshot)
        theta_init_sha = compute_state_dict_sha256(submission.theta_init)
        max_diff = 0.0

        # Verify step-0 micro-checkpoint matches theta_init (and coordinator expected_init_state if supplied)
        if set(step0_record.state_snapshot.keys()) != set(submission.theta_init.keys()):
            max_diff = float("inf")
        else:
            for name, init_t in submission.theta_init.items():
                s0_diff = float(
                    (init_t.to(torch.float32).cpu() - step0_record.state_snapshot[name].to(torch.float32).cpu())
                    .abs()
                    .max()
                    .item()
                )
                if not math.isfinite(s0_diff):
                    max_diff = float("inf")
                    break
                if s0_diff > max_diff:
                    max_diff = s0_diff
        if expected_init_state is not None:
            if set(expected_init_state.keys()) != set(step0_record.state_snapshot.keys()):
                max_diff = float("inf")
            else:
                for name, coord_t in expected_init_state.items():
                    c_diff = float(
                        (coord_t.to(torch.float32).cpu() - step0_record.state_snapshot[name].to(torch.float32).cpu())
                        .abs()
                        .max()
                        .item()
                    )
                    if not math.isfinite(c_diff):
                        max_diff = float("inf")
                        break
                    if c_diff > max_diff:
                        max_diff = c_diff

        steps_to_verify = sorted({ckpt_steps[0], selected_end, ckpt_steps[-1]})
        all_ckpt_hashes_ok = step0_sha == step0_record.lora_sha256 and step0_sha == theta_init_sha
        expected_sha = ""
        worker_record = submission.micro_checkpoints[selected_end]

        for target_end in steps_to_verify:
            expected_state = replay_inner_segment(
                model=validator_model,
                start_lora_state=step0_record.state_snapshot,
                token_sequences=token_sequences,
                start_step=0,
                end_step=target_end,
                lr=lr,
                weight_decay=weight_decay,
                rng_seed=workunit.rng_seed,
                micro_batch_size=micro_batch_size,
                initial_step_state=step0_record.state_snapshot,
                use_muon_lora_preconditioner=getattr(
                    submission, "use_muon_lora_preconditioner", False
                ),
                muon_ns_steps=getattr(submission, "muon_ns_steps", 5),
                use_paged_8bit_adamw=getattr(submission, "use_paged_8bit_adamw", False),
            )
            rec_target = submission.micro_checkpoints[target_end]
            w_state = rec_target.state_snapshot
            for name, exp_tensor in expected_state.items():
                if name not in w_state:
                    max_diff = float("inf")
                    break
                diff = float(
                    (exp_tensor.to(torch.float32).cpu() - w_state[name].to(torch.float32).cpu())
                    .abs()
                    .max()
                    .item()
                )
                # A NaN or infinite tensor must fail, not slip through: NaN compares False against every threshold.
                if not math.isfinite(diff):
                    max_diff = float("inf")
                    break
                if diff > max_diff:
                    max_diff = diff
            target_sha = compute_state_dict_sha256(expected_state)
            if target_end == selected_end:
                expected_sha = target_sha
            if target_sha != rec_target.lora_sha256:
                all_ckpt_hashes_ok = False
            expected_state.clear()
            del expected_state

        # Verify theta_final matches the terminal micro-checkpoint and raw_delta matches theta_init - theta_final
        final_ckpt_state = submission.micro_checkpoints[ckpt_steps[-1]].state_snapshot
        if set(submission.theta_final.keys()) != set(final_ckpt_state.keys()):
            max_diff = float("inf")
        else:
            for name, fin_ckpt_t in final_ckpt_state.items():
                fin_diff = float(
                    (fin_ckpt_t.to(torch.float32).cpu() - submission.theta_final[name].to(torch.float32).cpu())
                    .abs()
                    .max()
                    .item()
                )
                if not math.isfinite(fin_diff):
                    max_diff = float("inf")
                    break
                if fin_diff > max_diff:
                    max_diff = fin_diff

        for name, init_t in submission.theta_init.items():
            reconstructed_delta = init_t.to(torch.float32).cpu() - submission.theta_final[name].to(torch.float32).cpu()
            submitted_raw = submission.raw_delta[name].to(torch.float32).cpu()
            delta_mismatch = float((reconstructed_delta - submitted_raw).abs().max().item())
            if not math.isfinite(delta_mismatch):
                max_diff = float("inf")
                break
            if delta_mismatch > max_diff:
                max_diff = delta_mismatch

        hash_matched = all_ckpt_hashes_ok
        passed = (max_diff <= self.tolerance_eps) and hash_matched

        flagged = not passed
        if passed:
            reason = (
                f"Verified segment [{selected_start}, {selected_end}] ({overhead_pct}% overhead): "
                f"max_diff={max_diff:.2e} <= eps={self.tolerance_eps:.1e}"
            )
        elif not hash_matched and max_diff <= self.tolerance_eps:
            reason = (
                f"REJECTED segment [{selected_start}, {selected_end}]: "
                f"checkpoint lora_sha256 mismatch (expected {expected_sha[:16]}, got {worker_record.lora_sha256[:16]})"
            )
        else:
            reason = (
                f"REJECTED segment [{selected_start}, {selected_end}]: "
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
        if not isinstance(worker_class, ExecutionClassProfile):
            raise TypeError("worker_class must be an ExecutionClassProfile")
        if isinstance(validator_pool, (str, bytes)) or not isinstance(validator_pool, Sequence):
            raise TypeError("validator_pool must be a Sequence of ExecutionClassProfile")
        for cand in validator_pool:
            if not isinstance(cand, ExecutionClassProfile):
                raise TypeError("each validator_pool entry must be an ExecutionClassProfile")
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
        if base_eps is not None and (isinstance(base_eps, bool) or not isinstance(base_eps, (int, float))):
            raise ValueError("base_eps must be finite and > 0")
        eps0 = float(base_eps if base_eps is not None else self.tolerance_eps)
        if eps0 <= 0.0 or not math.isfinite(eps0):
            raise ValueError("base_eps must be finite and > 0")
        if (
            isinstance(cross_thread_eps, bool)
            or not isinstance(cross_thread_eps, (int, float))
            or not math.isfinite(float(cross_thread_eps))
            or float(cross_thread_eps) <= 0.0
        ):
            raise ValueError("cross_thread_eps must be finite and > 0")
        if (
            isinstance(max_tensor_fraction_above_eps, bool)
            or not isinstance(max_tensor_fraction_above_eps, (int, float))
            or not math.isfinite(float(max_tensor_fraction_above_eps))
            or not (0.0 <= float(max_tensor_fraction_above_eps) <= 1.0)
        ):
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

        if _has_non_finite(worker_state) or _has_non_finite(validator_state):
            # Checked over every tensor before any max/threshold comparison: NaN compares False
            # against every bound, so a NaN in a non-first key would otherwise pass (IP-238).
            return TwoStatisticSpotCheckResult(
                linf_divergence=float("inf"),
                num_tensors_above_eps=len(validator_state),
                total_tensors=len(validator_state),
                tensor_fraction_above_eps=1.0,
                base_eps=eps0,
                effective_eps=eps0,
                max_tensor_fraction_above_eps=max_tensor_fraction_above_eps,
                exact_class_matched=exact_ok,
                dtype_matched=dtype_ok,
                passed=False,
                reason=f"{NON_FINITE_TENSOR}: worker or validator state contains NaN or inf.",
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
