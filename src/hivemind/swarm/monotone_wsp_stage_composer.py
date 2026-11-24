# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Monotone Multi-Stage Weight-Space Programming (WSP) Composer with ANSE Ratchet.

Chains heterogeneous training-free open-weight surgery stages — such as self-healing /
FoMoE ghost-sink zeroing, cross-layer anchor protection & nullspace projection,
orthogonalized ``sigma_topk`` multi-gate router surgery, folded LayerNorm temperature
calibration, and 4-bit quantization residual compensation — under a strict per-stage
ANSE thermodynamic ratchet:

1. **Per-Stage Strict Thermodynamic Monotonicity & Guard Ratchet**:
   A stage is committed if and only if:
   - All resulting weights and evaluation domain NLLs are finite,
   - Stage and cumulative edited parameter counts vs pristine input stay within
     ``WSPStageSpec`` and ``max_total_edit_fraction`` budgets,
   - Worst-domain (and general-domain) drift vs pristine base stays within
     ``<= general_guard_rel`` (default ``+1.0%``), and
   - ``Delta E_stage = E_candidate - E_pre < -min_energy_improvement`` (strictly lowers
     ANSE energy).
   Otherwise the stage is rolled back bitwise (``post_sha256 == pre_sha256``), preserving
   all previously committed stage improvements.

2. **Tamper-Evident SHA-256 Stage & Pipeline Provenance**:
   Every ``WSPStageReceipt`` captures stage name, parameter-edit budget and usage,
   pre/candidate/post SHA-256 tensor digests, pre/candidate/post domain NLL vectors,
   ``nll_mean``, ``stage_delta_nats``, and ANSE energy, bound by canonical SHA-256.

3. **Single-Shot Virgin Holdout Vault Verification**:
   ``verify_holdout_vault_single_shot`` enforces strict single-use evaluation against a
   mutable ``vault_lock_state`` (raising ``HoldoutVaultLockedError`` on any second
   attempt) to prevent adaptive hyperparameter or stage-selection overfitting.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from hivemind.swarm.quantization_compensated_wsp import (
    compensate_quantization_error_via_lora_svd,
    compute_payload_sha256,
    symmetric_group_quantize_dequantize,
)

SCHEMA_VERSION: str = "hivemind.monotone_wsp_stage_composer/1"
PIPELINE_RECEIPT_OBJECT: str = "hivemind.monotone_wsp_pipeline_receipt"
STAGE_RECEIPT_OBJECT: str = "hivemind.monotone_wsp_stage_receipt"
VAULT_RECEIPT_OBJECT: str = "hivemind.holdout_vault_single_shot_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_GUARD_REL: float = 0.01
DEFAULT_MAX_TOTAL_EDIT_FRACTION: float = 0.05
DEFAULT_ANSE_SAFE_MARGIN_FRAC: float = 0.85
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_B_KEY_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)\.(w[123]_B)$")
_EXPERT_ANY_KEY_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)\.(w[123]_[AB])$")
_ROUTER_KEY_RE = re.compile(r"^blocks\.(\d+)\.mlp\.router\.weight$")
_VALID_EXPERT_B_FACTORS: frozenset[str] = frozenset({"w1_B", "w2_B", "w3_B"})


class HoldoutVaultLockedError(RuntimeError):
    """Raised when a caller attempts to evaluate a single-shot holdout vault more than once."""


# ---------------------------------------------------------------------------
# 1. Validation, State-Dict Digests & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` is finite and lies within ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _is_tensor_finite(tensor: torch.Tensor) -> bool:
    """Return True iff every element of ``tensor`` is finite."""
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _is_state_dict_finite(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True iff all tensors in ``state_dict`` contain only finite entries."""
    return all(_is_tensor_finite(t) for t in state_dict.values())


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return an isolated detached deep clone of ``state_dict`` preserving order, dtype, and device."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping of str -> torch.Tensor")
    cloned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"state_dict keys must be non-empty strings, got {key!r}")
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        cloned[key] = val.detach().clone()
    return cloned


def _validate_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    *,
    require_finite: bool = True,
) -> dict[str, torch.Tensor]:
    """Validate and clone a non-empty PyTorch ``state_dict`` with total parameter count > 0."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{name} must be a non-empty mapping of str -> torch.Tensor")
    out: dict[str, torch.Tensor] = {}
    total_numel = 0
    for key, val in state_dict.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        if require_finite and not _is_tensor_finite(val):
            raise ValueError(f"{name}[{key!r}] contains non-finite values")
        total_numel += int(val.numel())
        out[key] = val.detach().clone()
    if total_numel <= 0:
        raise ValueError(f"{name} must contain at least one scalar parameter")
    return out


def _validate_candidate_matches_base(
    candidate: Any,
    base: Mapping[str, torch.Tensor],
    *,
    stage_name: str,
) -> dict[str, torch.Tensor]:
    """Validate that a candidate ``state_dict`` matches ``base`` keys and tensor shapes."""
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError(
            f"Stage {stage_name!r} transform_fn must return a non-empty mapping of str -> torch.Tensor"
        )
    if set(candidate.keys()) != set(base.keys()):
        missing = sorted(set(base.keys()) - set(candidate.keys()))
        extra = sorted(set(candidate.keys()) - set(base.keys()))
        raise ValueError(
            f"Stage {stage_name!r} returned mismatched state_dict keys (missing={missing[:3]}, extra={extra[:3]})"
        )
    out: dict[str, torch.Tensor] = {}
    for key in base:
        cand_t = candidate[key]
        base_t = base[key]
        if not isinstance(cand_t, torch.Tensor):
            raise TypeError(
                f"Stage {stage_name!r} tensor {key!r} must be a torch.Tensor, got {type(cand_t).__name__}"
            )
        if cand_t.shape != base_t.shape:
            raise ValueError(
                f"Stage {stage_name!r} changed shape of {key!r}: {tuple(cand_t.shape)} != {tuple(base_t.shape)}"
            )
        out[key] = cand_t.detach().clone()
    return out


def compute_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over ordered tensor keys, dtypes, shapes, and bytes."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping of str -> torch.Tensor")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        cpu_t = val.detach().to(device="cpu").contiguous()
        digest.update(f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        if cpu_t.numel() > 0:
            digest.update(cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def count_edited_parameters(
    reference_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> tuple[int, int]:
    """Return ``(edited_params, total_params)`` counting elementwise changes across matching keys."""
    if not isinstance(reference_state, Mapping) or not reference_state:
        raise ValueError("reference_state must be a non-empty mapping")
    if not isinstance(candidate_state, Mapping) or set(reference_state.keys()) != set(candidate_state.keys()):
        raise ValueError("reference_state and candidate_state must share identical keys")

    edited = 0
    total = 0
    for key in sorted(reference_state.keys()):
        ref_t = reference_state[key]
        cand_t = candidate_state[key]
        if ref_t.shape != cand_t.shape:
            raise ValueError(f"Shape mismatch for {key!r}: {tuple(ref_t.shape)} vs {tuple(cand_t.shape)}")
        total += int(ref_t.numel())
        if ref_t.numel() == 0:
            continue
        r_cpu = ref_t.detach().to(device="cpu")
        c_cpu = cand_t.detach().to(device="cpu")
        if r_cpu.dtype == c_cpu.dtype and torch.equal(r_cpu, c_cpu):
            continue
        if r_cpu.is_floating_point() != c_cpu.is_floating_point():
            edited += int(ref_t.numel())
        else:
            edited += int(torch.ne(r_cpu, c_cpu).sum().item())
    return edited, total


# ---------------------------------------------------------------------------
# 2. Domain NLL Normalization, ANSE Energy & Provenance Verification
# ---------------------------------------------------------------------------


def parse_eval_domain_nll(
    raw_eval: Any,
    *,
    allow_non_finite: bool = False,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
) -> tuple[dict[str, float], bool]:
    """Normalize evaluator output into ``(sorted_domain_nll, all_finite)``."""
    payload: Any = raw_eval
    if isinstance(raw_eval, tuple) and len(raw_eval) == 2 and isinstance(raw_eval[1], Mapping):
        payload = raw_eval[1]
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("eval_fn must return a non-empty mapping of domain -> positive finite NLL")

    domain_map: Mapping[str, Any] = payload
    for container_key in ("domain_nll", "per_domain_nll"):
        if container_key in payload:
            sub = payload[container_key]
            if not isinstance(sub, Mapping) or not sub:
                raise ValueError(f"eval[{container_key!r}] must be a non-empty mapping")
            domain_map = sub
            break

    out: dict[str, float] = {}
    all_finite = True
    for raw_dom in sorted(domain_map.keys(), key=str):
        dom = str(raw_dom).strip()
        if not dom:
            raise ValueError("Domain names returned by eval_fn must be non-empty strings")
        val = domain_map[raw_dom]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"domain_nll[{dom!r}] must be a number, got {val!r}")
        val_f = float(val)
        if not math.isfinite(val_f):
            if not allow_non_finite:
                raise ValueError(f"domain_nll[{dom!r}] must be finite, got {val!r}")
            all_finite = False
            out[dom] = float(maximum_pain)
            continue
        if val_f <= 0.0:
            raise ValueError(f"domain_nll[{dom!r}] must be > 0.0, got {val!r}")
        out[dom] = val_f
    return out, all_finite


def evaluate_anse_domain_metrics(
    base_domain_nll: Mapping[str, float],
    candidate_domain_nll: Mapping[str, float],
    *,
    pre_domain_nll: Mapping[str, float] | None = None,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
    cand_finite: bool = True,
) -> dict[str, Any]:
    """Compute per-domain drift, worst-domain & general guard verdict, and ANSE energy."""
    if not isinstance(base_domain_nll, Mapping) or not base_domain_nll:
        raise ValueError("base_domain_nll must be a non-empty mapping")
    if set(base_domain_nll.keys()) != set(candidate_domain_nll.keys()):
        raise ValueError("base_domain_nll and candidate_domain_nll must share identical domain keys")

    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    soft_pen = _validate_nonneg_finite(soft_margin_penalty, "soft_margin_penalty")
    max_pain = _validate_positive_finite(maximum_pain, "maximum_pain")
    if not isinstance(general_prefix, str) or not general_prefix.strip():
        raise ValueError("general_prefix must be a non-empty string")

    domain_order = tuple(sorted(base_domain_nll.keys()))
    base_vals: list[float] = []
    cand_vals: list[float] = []
    per_domain: dict[str, dict[str, float | bool]] = {}
    per_domain_drift: dict[str, float] = {}
    worst_domain_id = domain_order[0]
    worst_drift = -math.inf

    for dom in domain_order:
        b_v = _validate_positive_finite(base_domain_nll[dom], f"base_domain_nll[{dom!r}]")
        c_v = _validate_positive_finite(candidate_domain_nll[dom], f"candidate_domain_nll[{dom!r}]")
        drift = (c_v / b_v) - 1.0 if cand_finite else float(max_pain)
        base_vals.append(b_v)
        cand_vals.append(c_v)
        per_domain_drift[dom] = drift
        dom_guard_ok = bool(cand_finite and (drift <= guard_rel + 1e-12))
        per_domain[dom] = {
            "base_nll": b_v,
            "cand_nll": c_v,
            "delta_nats": c_v - b_v,
            "drift": drift,
            "drift_pct": 100.0 * drift,
            "guard_ok": dom_guard_ok,
        }
        if drift > worst_drift:
            worst_drift = drift
            worst_domain_id = dom

    general_keys = [
        d for d in domain_order if d.startswith(general_prefix) or d in ("general", "general_nll")
    ]
    general_drift = max(per_domain_drift[k] for k in general_keys) if general_keys else worst_drift
    general_ratio = 1.0 + general_drift
    safe_ratio = 1.0 + DEFAULT_ANSE_SAFE_MARGIN_FRAC * guard_rel

    guard_ok = bool(
        cand_finite
        and (worst_drift <= guard_rel + 1e-12)
        and (general_drift <= guard_rel + 1e-12)
    )

    nll_mean = float(sum(cand_vals) / float(len(cand_vals)))
    base_nll_mean = float(sum(base_vals) / float(len(base_vals)))
    pre_mean = (
        float(sum(float(pre_domain_nll[d]) for d in domain_order) / float(len(domain_order)))
        if pre_domain_nll is not None
        else base_nll_mean
    )
    mean_diff_vs_base = nll_mean - base_nll_mean
    candidate_delta_nats = nll_mean - pre_mean

    if not cand_finite or not guard_ok:
        anse_energy = float(max_pain if not cand_finite else (mean_diff_vs_base + max_pain))
    else:
        soft_cost = soft_pen if general_ratio > safe_ratio + 1e-12 else 0.0
        anse_energy = float(mean_diff_vs_base + soft_cost)

    return {
        "domain_order": domain_order,
        "base_nll_mean": base_nll_mean,
        "nll_mean": nll_mean,
        "mean_diff_vs_base": mean_diff_vs_base,
        "candidate_delta_nats": candidate_delta_nats,
        "worst_domain_id": worst_domain_id,
        "worst_domain_drift": float(worst_drift),
        "worst_domain_drift_pct": float(100.0 * worst_drift),
        "general_domain_drift": float(general_drift),
        "general_domain_drift_pct": float(100.0 * general_drift),
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


# ---------------------------------------------------------------------------
# 3. Core Composer Dataclasses & Provenance Verifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WSPStageSpec:
    """Specification for a single deterministic training-free WSP surgery stage."""

    name: str
    transform_fn: Callable[[dict[str, torch.Tensor]], Mapping[str, torch.Tensor]]
    max_edit_fraction: float | None = None
    max_edited_params: int | None = None
    min_energy_improvement: float = 0.0
    stage_kind: str = "wsp_surgery"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("WSPStageSpec.name must be a non-empty string")
        if not callable(self.transform_fn):
            raise TypeError("WSPStageSpec.transform_fn must be callable")
        if self.max_edit_fraction is not None:
            _validate_unit_interval(self.max_edit_fraction, "max_edit_fraction")
        if self.max_edited_params is not None:
            _validate_nonneg_int(self.max_edited_params, "max_edited_params")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        if not isinstance(self.stage_kind, str) or not self.stage_kind.strip():
            raise ValueError("WSPStageSpec.stage_kind must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("WSPStageSpec.description must be a str")


@dataclass(frozen=True)
class WSPStageReceipt:
    """Tamper-evident thermodynamic receipt for one stage in the monotone WSP pipeline."""

    stage_index: int
    stage_name: str
    stage_kind: str
    status: str
    committed: bool
    rolled_back: bool
    reason: str
    pre_sha256: str
    candidate_sha256: str
    post_sha256: str
    stage_edit_budget_fraction: float
    stage_edited_params: int
    stage_edit_fraction: float
    cumulative_edited_params: int
    cumulative_edit_fraction: float
    total_params: int
    domain_order: tuple[str, ...]
    pre_domain_nll_vector: tuple[float, ...]
    candidate_domain_nll_vector: tuple[float, ...]
    post_domain_nll_vector: tuple[float, ...]
    pre_domain_nll: dict[str, float]
    candidate_domain_nll: dict[str, float]
    post_domain_nll: dict[str, float]
    nll_mean: float
    stage_delta_nats: float
    candidate_delta_nats: float
    pre_anse_energy: float
    candidate_anse_energy: float
    post_anse_energy: float
    delta_anse_energy: float
    candidate_delta_anse_energy: float
    worst_domain_id: str
    worst_domain_drift: float
    worst_domain_drift_pct: float
    general_domain_drift: float
    general_domain_drift_pct: float
    guard_ok: bool
    budget_ok: bool
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MonotoneWSPPipelineResult:
    """Complete output state, per-stage receipts, and SHA-256 provenance of a pipeline run."""

    champion_state: dict[str, torch.Tensor]
    base_sha256: str
    champion_sha256: str
    stage_receipts: tuple[WSPStageReceipt, ...]
    committed_stages: tuple[str, ...]
    rolled_back_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]
    domain_order: tuple[str, ...]
    initial_domain_nll: dict[str, float]
    final_domain_nll: dict[str, float]
    initial_anse_energy: float
    final_anse_energy: float
    total_delta_anse_energy: float
    nll_mean: float
    stage_delta_nats: float
    total_params: int
    cumulative_edited_params: int
    cumulative_edit_fraction: float
    max_total_edit_fraction: float
    general_guard_rel: float
    guard_ok: bool
    aborted_early: bool
    status: str
    receipt: dict[str, Any]
    receipt_sha256: str


@dataclass
class HoldoutVaultLockState:
    """Mutable single-shot lock state protecting a virgin holdout vault from repeat access."""

    vault_id: str = "virgin_holdout_vault"
    locked: bool = False
    access_count: int = 0
    base_sha256: str | None = None
    champion_sha256: str | None = None
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vault_id, str) or not self.vault_id.strip():
            raise ValueError("vault_id must be a non-empty string")
        _validate_bool(self.locked, "locked")
        _validate_nonneg_int(self.access_count, "access_count")


def _build_stage_receipt(**fields: Any) -> WSPStageReceipt:
    """Construct a ``WSPStageReceipt`` with canonical ``receipt_sha256``."""
    core = {k: v for k, v in fields.items() if k != "receipt_sha256"}
    core_json = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in core.items()
    }
    sha = compute_payload_sha256(core_json)
    return WSPStageReceipt(**core, receipt_sha256=sha)


def _verify_payload_sha256_field(payload: Mapping[str, Any]) -> bool:
    """Verify that ``payload['receipt_sha256']`` matches canonical JSON SHA-256 of remaining fields."""
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in payload.items()
        if k != "receipt_sha256"
    }
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def verify_stage_receipt_provenance(receipt: WSPStageReceipt | Mapping[str, Any]) -> bool:
    """Verify canonical ``receipt_sha256`` of a single ``WSPStageReceipt`` or its dict form."""
    if isinstance(receipt, WSPStageReceipt):
        payload: Mapping[str, Any] = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        payload = receipt
    else:
        raise TypeError(f"receipt must be a WSPStageReceipt or Mapping, got {type(receipt).__name__}")
    return _verify_payload_sha256_field(payload)


def verify_pipeline_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify top-level ``receipt_sha256`` and every nested stage receipt digest."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    stages = receipt.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return False
    if any(not verify_stage_receipt_provenance(st) for st in stages):
        return False
    return _verify_payload_sha256_field(receipt)


def verify_vault_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` of a single-shot vault receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    return _verify_payload_sha256_field(receipt)


# ---------------------------------------------------------------------------
# 4. Monotone Multi-Stage WSP Pipeline Orchestrator
# ---------------------------------------------------------------------------


def _evaluate_candidate_or_reject(
    *,
    stage_spec: WSPStageSpec,
    base_state: Mapping[str, torch.Tensor],
    pre_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    base_domain_nll: Mapping[str, float],
    pre_domain_nll: Mapping[str, float],
    pre_metrics: Mapping[str, Any],
    pre_energy: float,
    pre_sha: str,
    candidate_sha: str,
    stage_budget_frac: float,
    max_total_edit_fraction: float,
    total_params: int,
    general_guard_rel: float,
    general_prefix: str,
    soft_margin_penalty: float,
    maximum_pain: float,
) -> tuple[bool, dict[str, Any]]:
    """Assess candidate finiteness, parameter budgets, domain guard, and strict ANSE descent."""
    cand_finite = _is_state_dict_finite(candidate_state)
    stage_edited, _ = count_edited_parameters(pre_state, candidate_state)
    cum_edited_cand, _ = count_edited_parameters(base_state, candidate_state)
    stage_edit_frac = float(stage_edited) / float(total_params)
    cum_edit_frac_cand = float(cum_edited_cand) / float(total_params)

    stage_budget_ok = stage_edit_frac <= stage_budget_frac + 1e-12
    if stage_spec.max_edited_params is not None:
        stage_budget_ok = stage_budget_ok and (stage_edited <= stage_spec.max_edited_params)
    total_budget_ok = cum_edit_frac_cand <= max_total_edit_fraction + 1e-12
    budget_ok = bool(stage_budget_ok and total_budget_ok)

    if candidate_sha == pre_sha:
        return False, {
            "status": "SKIPPED_NOOP",
            "rolled_back": False,
            "reason": "Stage produced bitwise-identical parameters (zero edit delta).",
            "stage_edited_params": 0,
            "stage_edit_fraction": 0.0,
            "cand_cum_edited_params": cum_edited_cand,
            "cand_cum_edit_fraction": cum_edit_frac_cand,
            "candidate_domain_nll": dict(pre_domain_nll),
            "candidate_anse_energy": float(pre_energy),
            "candidate_delta_nats": 0.0,
            "worst_domain_id": str(pre_metrics["worst_domain_id"]),
            "worst_domain_drift": float(pre_metrics["worst_domain_drift"]),
            "worst_domain_drift_pct": float(pre_metrics["worst_domain_drift_pct"]),
            "general_domain_drift": float(pre_metrics["general_domain_drift"]),
            "general_domain_drift_pct": float(pre_metrics["general_domain_drift_pct"]),
            "guard_ok": bool(pre_metrics["guard_ok"]),
            "budget_ok": budget_ok,
        }

    if not cand_finite:
        return False, {
            "status": "ROLLED_BACK_NON_FINITE",
            "rolled_back": True,
            "reason": "Candidate state contains non-finite tensor entries (NaN/Inf).",
            "stage_edited_params": stage_edited,
            "stage_edit_fraction": stage_edit_frac,
            "cand_cum_edited_params": cum_edited_cand,
            "cand_cum_edit_fraction": cum_edit_frac_cand,
            "candidate_domain_nll": {d: float(maximum_pain) for d in base_domain_nll},
            "candidate_anse_energy": float(maximum_pain),
            "candidate_delta_nats": float(maximum_pain),
            "worst_domain_id": str(pre_metrics["worst_domain_id"]),
            "worst_domain_drift": float(maximum_pain),
            "worst_domain_drift_pct": float(100.0 * maximum_pain),
            "general_domain_drift": float(maximum_pain),
            "general_domain_drift_pct": float(100.0 * maximum_pain),
            "guard_ok": False,
            "budget_ok": budget_ok,
        }

    if not budget_ok:
        detail = (
            f"cumulative_edit_fraction={cum_edit_frac_cand:.6f} > max_total_edit_fraction={max_total_edit_fraction:.6f}"
            if not total_budget_ok
            else f"stage_edit_fraction={stage_edit_frac:.6f} exceeded stage budget {stage_budget_frac:.6f}"
        )
        return False, {
            "status": "ROLLED_BACK_EDIT_BUDGET",
            "rolled_back": True,
            "reason": f"Parameter-edit budget exceeded ({detail}).",
            "stage_edited_params": stage_edited,
            "stage_edit_fraction": stage_edit_frac,
            "cand_cum_edited_params": cum_edited_cand,
            "cand_cum_edit_fraction": cum_edit_frac_cand,
            "candidate_domain_nll": dict(pre_domain_nll),
            "candidate_anse_energy": float(maximum_pain),
            "candidate_delta_nats": 0.0,
            "worst_domain_id": str(pre_metrics["worst_domain_id"]),
            "worst_domain_drift": float(pre_metrics["worst_domain_drift"]),
            "worst_domain_drift_pct": float(pre_metrics["worst_domain_drift_pct"]),
            "general_domain_drift": float(pre_metrics["general_domain_drift"]),
            "general_domain_drift_pct": float(pre_metrics["general_domain_drift_pct"]),
            "guard_ok": bool(pre_metrics["guard_ok"]),
            "budget_ok": False,
        }

    raw_eval = eval_fn(clone_state_dict(candidate_state))
    cand_domain_nll, eval_finite = parse_eval_domain_nll(
        raw_eval,
        allow_non_finite=True,
        maximum_pain=maximum_pain,
    )
    if set(cand_domain_nll.keys()) != set(base_domain_nll.keys()):
        raise ValueError(
            f"Stage {stage_spec.name!r} eval_fn returned mismatched domain keys: "
            f"{sorted(cand_domain_nll.keys())} != {sorted(base_domain_nll.keys())}"
        )

    cand_metrics = evaluate_anse_domain_metrics(
        base_domain_nll,
        cand_domain_nll,
        pre_domain_nll=pre_domain_nll,
        general_guard_rel=general_guard_rel,
        general_prefix=general_prefix,
        soft_margin_penalty=soft_margin_penalty,
        maximum_pain=maximum_pain,
        cand_finite=eval_finite,
    )
    cand_energy = float(cand_metrics["anse_energy"])
    cand_delta_e = cand_energy - pre_energy
    strict_descent = cand_delta_e < -float(stage_spec.min_energy_improvement)

    if not eval_finite:
        status = "ROLLED_BACK_NON_FINITE"
        reason = "Candidate evaluation yielded non-finite domain NLL."
        accepted = False
    elif not bool(cand_metrics["guard_ok"]):
        status = "ROLLED_BACK_GUARD_VIOLATION"
        reason = (
            f"Domain guard violated on {cand_metrics['worst_domain_id']}: "
            f"{cand_metrics['worst_domain_drift_pct']:+.3f}% > {100.0 * general_guard_rel:+.3f}%."
        )
        accepted = False
    elif not strict_descent:
        status = "ROLLED_BACK_ENERGY_REGRESSION"
        reason = (
            f"ANSE thermodynamic ratchet rejected Delta E={cand_delta_e:+.6f} "
            f"(required < {-stage_spec.min_energy_improvement:+.6f})."
        )
        accepted = False
    else:
        status = "COMMITTED"
        reason = (
            f"Committed stage with Delta E={cand_delta_e:+.6f} and "
            f"worst_domain_drift={cand_metrics['worst_domain_drift_pct']:+.3f}%."
        )
        accepted = True

    return accepted, {
        "status": status,
        "rolled_back": not accepted,
        "reason": reason,
        "stage_edited_params": stage_edited,
        "stage_edit_fraction": stage_edit_frac,
        "cand_cum_edited_params": cum_edited_cand,
        "cand_cum_edit_fraction": cum_edit_frac_cand,
        "candidate_domain_nll": cand_domain_nll,
        "candidate_anse_energy": cand_energy,
        "candidate_delta_nats": float(cand_metrics["candidate_delta_nats"]),
        "cand_metrics": cand_metrics,
        "worst_domain_id": str(cand_metrics["worst_domain_id"]),
        "worst_domain_drift": float(cand_metrics["worst_domain_drift"]),
        "worst_domain_drift_pct": float(cand_metrics["worst_domain_drift_pct"]),
        "general_domain_drift": float(cand_metrics["general_domain_drift"]),
        "general_domain_drift_pct": float(cand_metrics["general_domain_drift_pct"]),
        "guard_ok": bool(cand_metrics["guard_ok"]),
        "budget_ok": True,
    }


@torch.no_grad()
def run_monotone_wsp_stage_pipeline(
    state_dict: Mapping[str, torch.Tensor],
    stages: Sequence[WSPStageSpec],
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    *,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    max_total_edit_fraction: float = DEFAULT_MAX_TOTAL_EDIT_FRACTION,
    skip_regressing_stage: bool = True,
    base_state: Mapping[str, torch.Tensor] | None = None,
    base_domain_nll: Mapping[str, float] | None = None,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
) -> MonotoneWSPPipelineResult:
    """Execute a multi-stage WSP surgery chain under a strict per-stage ANSE ratchet.

    Each stage transforms an isolated copy of the current committed ``state_dict``.
    A stage is committed iff weights & domain NLLs are finite, stage and cumulative
    parameter edit fractions stay within budget, worst-domain drift vs pristine base
    clears ``general_guard_rel`` (default ``+1.0%``), and ANSE energy strictly
    decreases (``Delta E_stage < -min_energy_improvement``). Any regressing or
    budget-violating stage is atomically rolled back bitwise without losing prior
    committed improvements.
    """
    base_input_state = _validate_state_dict(state_dict, "state_dict", require_finite=True)
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence) or len(stages) == 0:
        raise ValueError("stages must be a non-empty sequence of WSPStageSpec")
    seen_names: set[str] = set()
    for idx, spec in enumerate(stages):
        if not isinstance(spec, WSPStageSpec):
            raise TypeError(f"stages[{idx}] must be a WSPStageSpec, got {type(spec).__name__}")
        if spec.name in seen_names:
            raise ValueError(f"Duplicate stage name {spec.name!r} in pipeline stages")
        seen_names.add(spec.name)

    if not callable(eval_fn):
        raise TypeError("eval_fn must be callable")
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    total_budget_frac = _validate_unit_interval(max_total_edit_fraction, "max_total_edit_fraction")
    _validate_bool(skip_regressing_stage, "skip_regressing_stage")
    soft_pen = _validate_nonneg_finite(soft_margin_penalty, "soft_margin_penalty")
    max_pain = _validate_positive_finite(maximum_pain, "maximum_pain")
    if not isinstance(general_prefix, str) or not general_prefix.strip():
        raise ValueError("general_prefix must be a non-empty string")

    if base_domain_nll is not None:
        ref_base_nll, _ = parse_eval_domain_nll(base_domain_nll, allow_non_finite=False)
    elif base_state is not None:
        ref_base_state = _validate_state_dict(base_state, "base_state", require_finite=True)
        ref_base_nll, _ = parse_eval_domain_nll(eval_fn(clone_state_dict(ref_base_state)), allow_non_finite=False)
    else:
        ref_base_nll = {}

    init_domain_nll, _ = parse_eval_domain_nll(eval_fn(clone_state_dict(base_input_state)), allow_non_finite=False)
    if not ref_base_nll:
        ref_base_nll = dict(init_domain_nll)
    elif set(ref_base_nll.keys()) != set(init_domain_nll.keys()):
        raise ValueError("base_domain_nll keys must match eval_fn(state_dict) domain keys")

    init_metrics = evaluate_anse_domain_metrics(
        ref_base_nll,
        init_domain_nll,
        pre_domain_nll=init_domain_nll,
        general_guard_rel=guard_rel,
        general_prefix=general_prefix,
        soft_margin_penalty=soft_pen,
        maximum_pain=max_pain,
        cand_finite=True,
    )

    domain_order: tuple[str, ...] = tuple(init_metrics["domain_order"])
    base_sha = compute_state_dict_sha256(base_input_state)
    _, total_params = count_edited_parameters(base_input_state, base_input_state)

    current_state = clone_state_dict(base_input_state)
    current_sha = base_sha
    current_domain_nll = dict(init_domain_nll)
    current_metrics = dict(init_metrics)
    current_energy = float(init_metrics["anse_energy"])
    current_nll_mean = float(init_metrics["nll_mean"])
    current_cum_edited = 0
    current_cum_frac = 0.0

    receipts: list[WSPStageReceipt] = []
    committed_names: list[str] = []
    rolled_back_names: list[str] = []
    skipped_names: list[str] = []
    aborted_early = False

    for stage_idx, spec in enumerate(stages):
        pre_sha = current_sha
        pre_nll = dict(current_domain_nll)
        pre_energy = current_energy
        pre_nll_mean = current_nll_mean
        pre_metrics = dict(current_metrics)
        stage_budget = spec.max_edit_fraction if spec.max_edit_fraction is not None else total_budget_frac

        raw_candidate = spec.transform_fn(clone_state_dict(current_state))
        candidate_state = _validate_candidate_matches_base(
            raw_candidate, base_input_state, stage_name=spec.name
        )
        candidate_sha = compute_state_dict_sha256(candidate_state)

        accepted, verdict = _evaluate_candidate_or_reject(
            stage_spec=spec,
            base_state=base_input_state,
            pre_state=current_state,
            candidate_state=candidate_state,
            eval_fn=eval_fn,
            base_domain_nll=ref_base_nll,
            pre_domain_nll=pre_nll,
            pre_metrics=pre_metrics,
            pre_energy=pre_energy,
            pre_sha=pre_sha,
            candidate_sha=candidate_sha,
            stage_budget_frac=stage_budget,
            max_total_edit_fraction=total_budget_frac,
            total_params=total_params,
            general_guard_rel=guard_rel,
            general_prefix=general_prefix,
            soft_margin_penalty=soft_pen,
            maximum_pain=max_pain,
        )

        if accepted:
            current_state = candidate_state
            current_sha = candidate_sha
            current_domain_nll = dict(verdict["candidate_domain_nll"])
            current_metrics = dict(verdict["cand_metrics"])
            current_energy = float(verdict["candidate_anse_energy"])
            current_nll_mean = float(current_metrics["nll_mean"])
            current_cum_edited = int(verdict["cand_cum_edited_params"])
            current_cum_frac = float(verdict["cand_cum_edit_fraction"])
            committed_names.append(spec.name)
        elif verdict["rolled_back"]:
            rolled_back_names.append(spec.name)
            post_check_sha = compute_state_dict_sha256(current_state)
            if post_check_sha != pre_sha:  # pragma: no cover - defensive bitwise invariant
                raise RuntimeError(f"Bitwise rollback invariant failed after {spec.name!r}")
        else:
            skipped_names.append(spec.name)

        cand_nll_map = dict(verdict["candidate_domain_nll"])
        post_nll_map = dict(current_domain_nll)
        stage_receipt = _build_stage_receipt(
            stage_index=stage_idx,
            stage_name=spec.name,
            stage_kind=spec.stage_kind,
            status=str(verdict["status"]),
            committed=accepted,
            rolled_back=bool(verdict["rolled_back"]),
            reason=str(verdict["reason"]),
            pre_sha256=pre_sha,
            candidate_sha256=candidate_sha,
            post_sha256=current_sha,
            stage_edit_budget_fraction=float(stage_budget),
            stage_edited_params=int(verdict["stage_edited_params"]),
            stage_edit_fraction=float(verdict["stage_edit_fraction"]),
            cumulative_edited_params=current_cum_edited,
            cumulative_edit_fraction=current_cum_frac,
            total_params=total_params,
            domain_order=domain_order,
            pre_domain_nll_vector=tuple(pre_nll[d] for d in domain_order),
            candidate_domain_nll_vector=tuple(cand_nll_map[d] for d in domain_order),
            post_domain_nll_vector=tuple(post_nll_map[d] for d in domain_order),
            pre_domain_nll=pre_nll,
            candidate_domain_nll=cand_nll_map,
            post_domain_nll=post_nll_map,
            nll_mean=current_nll_mean,
            stage_delta_nats=current_nll_mean - pre_nll_mean,
            candidate_delta_nats=float(verdict["candidate_delta_nats"]),
            pre_anse_energy=pre_energy,
            candidate_anse_energy=float(verdict["candidate_anse_energy"]),
            post_anse_energy=current_energy,
            delta_anse_energy=current_energy - pre_energy,
            candidate_delta_anse_energy=float(verdict["candidate_anse_energy"]) - pre_energy,
            worst_domain_id=str(verdict["worst_domain_id"]),
            worst_domain_drift=float(verdict["worst_domain_drift"]),
            worst_domain_drift_pct=float(verdict["worst_domain_drift_pct"]),
            general_domain_drift=float(verdict["general_domain_drift"]),
            general_domain_drift_pct=float(verdict["general_domain_drift_pct"]),
            guard_ok=bool(verdict["guard_ok"]),
            budget_ok=bool(verdict["budget_ok"]),
        )
        receipts.append(stage_receipt)

        if verdict["rolled_back"] and not skip_regressing_stage:
            aborted_early = True
            break

    initial_energy = float(init_metrics["anse_energy"])
    total_delta_e = current_energy - initial_energy
    total_delta_nats = current_nll_mean - float(init_metrics["nll_mean"])
    pipeline_guard_ok = bool(current_metrics["guard_ok"])

    if aborted_early:
        pipeline_status = "ABORTED_EARLY_ON_REGRESSION"
    elif committed_names:
        pipeline_status = "COMPLETED_MONOTONE_COMMIT"
    elif rolled_back_names:
        pipeline_status = "COMPLETED_ALL_ROLLED_BACK"
    else:
        pipeline_status = "COMPLETED_NOOP"

    receipt_body: dict[str, Any] = {
        "object": PIPELINE_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "status": pipeline_status,
        "aborted_early": aborted_early,
        "base_sha256": base_sha,
        "champion_sha256": current_sha,
        "total_params": total_params,
        "cumulative_edited_params": current_cum_edited,
        "cumulative_edit_fraction": current_cum_frac,
        "max_total_edit_fraction": total_budget_frac,
        "general_guard_rel": guard_rel,
        "guard_ok": pipeline_guard_ok,
        "domain_order": list(domain_order),
        "initial_domain_nll": init_domain_nll,
        "final_domain_nll": current_domain_nll,
        "initial_anse_energy": initial_energy,
        "anse_energy": current_energy,
        "total_delta_anse_energy": total_delta_e,
        "nll_mean": current_nll_mean,
        "stage_delta_nats": total_delta_nats,
        "committed_stages": committed_names,
        "rolled_back_stages": rolled_back_names,
        "skipped_stages": skipped_names,
        "stages": [r.to_dict() for r in receipts],
    }
    receipt_sha = compute_payload_sha256(receipt_body)
    receipt_body["receipt_sha256"] = receipt_sha

    return MonotoneWSPPipelineResult(
        champion_state=current_state,
        base_sha256=base_sha,
        champion_sha256=current_sha,
        stage_receipts=tuple(receipts),
        committed_stages=tuple(committed_names),
        rolled_back_stages=tuple(rolled_back_names),
        skipped_stages=tuple(skipped_names),
        domain_order=domain_order,
        initial_domain_nll=init_domain_nll,
        final_domain_nll=current_domain_nll,
        initial_anse_energy=initial_energy,
        final_anse_energy=current_energy,
        total_delta_anse_energy=total_delta_e,
        nll_mean=current_nll_mean,
        stage_delta_nats=total_delta_nats,
        total_params=total_params,
        cumulative_edited_params=current_cum_edited,
        cumulative_edit_fraction=current_cum_frac,
        max_total_edit_fraction=total_budget_frac,
        general_guard_rel=guard_rel,
        guard_ok=pipeline_guard_ok,
        aborted_early=aborted_early,
        status=pipeline_status,
        receipt=receipt_body,
        receipt_sha256=receipt_sha,
    )


# ---------------------------------------------------------------------------
# 5. Single-Shot Virgin Holdout Vault Verification
# ---------------------------------------------------------------------------


def _ensure_vault_unlocked(
    vault_lock_state: HoldoutVaultLockState | dict[str, Any],
) -> tuple[str, int]:
    """Verify ``vault_lock_state`` is valid and unconsumed, returning ``(vault_id, prior_count)``."""
    if isinstance(vault_lock_state, HoldoutVaultLockState):
        if vault_lock_state.locked or vault_lock_state.access_count >= 1:
            raise HoldoutVaultLockedError(
                f"Holdout vault {vault_lock_state.vault_id!r} is locked after "
                f"{vault_lock_state.access_count} access(es); second evaluation rejected."
            )
        return vault_lock_state.vault_id, vault_lock_state.access_count

    if isinstance(vault_lock_state, dict):
        raw_id = vault_lock_state.get("vault_id", "virgin_holdout_vault")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("vault_lock_state['vault_id'] must be a non-empty string")
        raw_locked = vault_lock_state.get("locked", False)
        raw_consumed = vault_lock_state.get("consumed", False)
        raw_count = vault_lock_state.get("access_count", 0)
        _validate_bool(raw_locked, "vault_lock_state['locked']")
        _validate_bool(raw_consumed, "vault_lock_state['consumed']")
        count = _validate_nonneg_int(raw_count, "vault_lock_state['access_count']")
        if raw_locked or raw_consumed or count >= 1:
            raise HoldoutVaultLockedError(
                f"Holdout vault {raw_id!r} is locked after {count} access(es); second evaluation rejected."
            )
        return raw_id, count

    raise TypeError(
        "vault_lock_state must be a HoldoutVaultLockState or mutable dict, "
        f"got {type(vault_lock_state).__name__}"
    )


def _trip_vault_lock(
    vault_lock_state: HoldoutVaultLockState | dict[str, Any],
    *,
    vault_id: str,
    prior_count: int,
    base_sha: str,
    champ_sha: str,
) -> int:
    """Persistently lock ``vault_lock_state`` before invoking ``vault_eval_fn``."""
    new_count = prior_count + 1
    if isinstance(vault_lock_state, HoldoutVaultLockState):
        vault_lock_state.locked = True
        vault_lock_state.access_count = new_count
        vault_lock_state.base_sha256 = base_sha
        vault_lock_state.champion_sha256 = champ_sha
    else:
        vault_lock_state["vault_id"] = vault_id
        vault_lock_state["locked"] = True
        vault_lock_state["consumed"] = True
        vault_lock_state["access_count"] = new_count
        vault_lock_state["base_sha256"] = base_sha
        vault_lock_state["champion_sha256"] = champ_sha
    return new_count


@torch.no_grad()
def verify_holdout_vault_single_shot(
    champion_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    vault_eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    *,
    vault_lock_state: HoldoutVaultLockState | dict[str, Any],
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
    base_vault_domain_nll: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate ``champion_state`` vs ``base_state`` on the virgin holdout vault exactly once.

    Consumes and permanently locks ``vault_lock_state`` prior to calling ``vault_eval_fn``;
    any subsequent attempt with the same lock raises ``HoldoutVaultLockedError``.
    """
    vault_id, prior_count = _ensure_vault_unlocked(vault_lock_state)

    if not callable(vault_eval_fn):
        raise TypeError("vault_eval_fn must be callable")
    base_clean = _validate_state_dict(base_state, "base_state", require_finite=True)
    champ_clean = _validate_candidate_matches_base(
        champion_state, base_clean, stage_name="holdout_vault_champion"
    )
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    soft_pen = _validate_nonneg_finite(soft_margin_penalty, "soft_margin_penalty")
    max_pain = _validate_positive_finite(maximum_pain, "maximum_pain")
    if not isinstance(general_prefix, str) or not general_prefix.strip():
        raise ValueError("general_prefix must be a non-empty string")

    base_sha = compute_state_dict_sha256(base_clean)
    champ_sha = compute_state_dict_sha256(champ_clean)
    access_count = _trip_vault_lock(
        vault_lock_state,
        vault_id=vault_id,
        prior_count=prior_count,
        base_sha=base_sha,
        champ_sha=champ_sha,
    )

    if base_vault_domain_nll is not None:
        base_nll, _ = parse_eval_domain_nll(base_vault_domain_nll, allow_non_finite=False)
    else:
        base_nll, _ = parse_eval_domain_nll(
            vault_eval_fn(clone_state_dict(base_clean)), allow_non_finite=False
        )

    domain_order = tuple(sorted(base_nll.keys()))
    champ_weights_finite = _is_state_dict_finite(champ_clean)
    if champ_weights_finite:
        champ_nll, eval_finite = parse_eval_domain_nll(
            vault_eval_fn(clone_state_dict(champ_clean)),
            allow_non_finite=True,
            maximum_pain=max_pain,
        )
        if set(champ_nll.keys()) != set(base_nll.keys()):
            raise ValueError(
                f"vault_eval_fn returned mismatched domain keys: {sorted(champ_nll.keys())} != {sorted(base_nll.keys())}"
            )
    else:
        eval_finite = False
        champ_nll = {d: float(max_pain) for d in domain_order}

    base_metrics = evaluate_anse_domain_metrics(
        base_nll,
        base_nll,
        pre_domain_nll=base_nll,
        general_guard_rel=guard_rel,
        general_prefix=general_prefix,
        soft_margin_penalty=soft_pen,
        maximum_pain=max_pain,
        cand_finite=True,
    )
    champ_metrics = evaluate_anse_domain_metrics(
        base_nll,
        champ_nll,
        pre_domain_nll=base_nll,
        general_guard_rel=guard_rel,
        general_prefix=general_prefix,
        soft_margin_penalty=soft_pen,
        maximum_pain=max_pain,
        cand_finite=eval_finite,
    )

    edited_params, total_params = count_edited_parameters(base_clean, champ_clean)
    delta_e = float(champ_metrics["anse_energy"]) - float(base_metrics["anse_energy"])
    passed = bool(eval_finite and champ_metrics["guard_ok"] and delta_e < 0.0)

    receipt: dict[str, Any] = {
        "object": VAULT_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "vault_id": vault_id,
        "vault_locked": True,
        "vault_access_count": access_count,
        "passed": passed,
        "guard_ok": bool(champ_metrics["guard_ok"]),
        "base_sha256": base_sha,
        "champion_sha256": champ_sha,
        "edited_params_vs_base": edited_params,
        "total_params": total_params,
        "edit_fraction_vs_base": float(edited_params) / float(total_params),
        "domain_order": list(domain_order),
        "base_domain_nll": base_nll,
        "champion_domain_nll": champ_nll,
        "base_domain_nll_vector": [base_nll[d] for d in domain_order],
        "champion_domain_nll_vector": [champ_nll[d] for d in domain_order],
        "base_nll_mean": float(base_metrics["nll_mean"]),
        "nll_mean": float(champ_metrics["nll_mean"]),
        "stage_delta_nats": float(champ_metrics["candidate_delta_nats"]),
        "base_anse_energy": float(base_metrics["anse_energy"]),
        "anse_energy": float(champ_metrics["anse_energy"]),
        "delta_anse_energy": delta_e,
        "worst_domain_id": str(champ_metrics["worst_domain_id"]),
        "worst_domain_drift": float(champ_metrics["worst_domain_drift"]),
        "worst_domain_drift_pct": float(champ_metrics["worst_domain_drift_pct"]),
        "general_domain_drift": float(champ_metrics["general_domain_drift"]),
        "general_domain_drift_pct": float(champ_metrics["general_domain_drift_pct"]),
        "per_domain": champ_metrics["per_domain"],
    }
    receipt_sha = compute_payload_sha256(receipt)
    receipt["receipt_sha256"] = receipt_sha

    if isinstance(vault_lock_state, HoldoutVaultLockState):
        vault_lock_state.receipt_sha256 = receipt_sha
    else:
        vault_lock_state["receipt_sha256"] = receipt_sha
    return receipt


# ---------------------------------------------------------------------------
# 6. Reusable Heterogeneous Open-Weight WSP Stage Builders
# ---------------------------------------------------------------------------


def make_ghost_sink_stage(
    expert_indices: Sequence[int],
    *,
    layer_indices: Sequence[int] | None = None,
    name: str = "stage_1_ghost_sink",
    max_edit_fraction: float | None = None,
    max_edited_params: int | None = None,
    sanitize_non_finite: bool = True,
) -> WSPStageSpec:
    """Create a FoMoE zero-perturbation ghost-sink & self-healing stage (``B = 0 => dy_e(x) == 0``)."""
    if isinstance(expert_indices, (str, bytes)) or not isinstance(expert_indices, Sequence) or not expert_indices:
        raise ValueError("expert_indices must be a non-empty sequence of non-negative integers")
    expert_set = {_validate_nonneg_int(e, "expert_index") for e in expert_indices}
    layer_set: set[int] | None = None
    if layer_indices is not None:
        if isinstance(layer_indices, (str, bytes)) or not isinstance(layer_indices, Sequence) or not layer_indices:
            raise ValueError("layer_indices must be None or a non-empty sequence of non-negative integers")
        layer_set = {_validate_nonneg_int(l_idx, "layer_index") for l_idx in layer_indices}
    _validate_bool(sanitize_non_finite, "sanitize_non_finite")

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        for key, tensor in list(out.items()):
            if sanitize_non_finite and tensor.is_floating_point() and not _is_tensor_finite(tensor):
                if _EXPERT_ANY_KEY_RE.match(key) or _ROUTER_KEY_RE.match(key):
                    out[key] = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
            match = _EXPERT_B_KEY_RE.match(key)
            if match is None:
                continue
            l_idx, e_idx = int(match.group(1)), int(match.group(2))
            if e_idx in expert_set and (layer_set is None or l_idx in layer_set):
                out[key] = torch.zeros_like(out[key])
        return out

    return WSPStageSpec(
        name=name,
        transform_fn=_transform,
        max_edit_fraction=max_edit_fraction,
        max_edited_params=max_edited_params,
        stage_kind="self_healing_ghost_sink",
        description="Zero LoRA output factors (w1_B, w2_B, w3_B) on harmful ghost-sink experts.",
    )


def make_cross_layer_anchor_stage(
    anchor_subspace: torch.Tensor,
    *,
    target_layers: Sequence[int],
    expert_indices: Sequence[int] | None = None,
    factors: Sequence[str] = ("w2_B", "w3_B"),
    strength: float = 1.0,
    orthonormalize: bool = True,
    name: str = "stage_2_cross_layer_anchor",
    max_edit_fraction: float | None = None,
    max_edited_params: int | None = None,
) -> WSPStageSpec:
    """Create a cross-layer residual nullspace projection stage protecting upstream anchor layers."""
    if not isinstance(anchor_subspace, torch.Tensor):
        raise TypeError(f"anchor_subspace must be a torch.Tensor, got {type(anchor_subspace).__name__}")
    if anchor_subspace.ndim not in (1, 2) or anchor_subspace.numel() == 0:
        raise ValueError("anchor_subspace must be a non-empty 1D or 2D tensor")
    if not anchor_subspace.is_floating_point():
        raise TypeError("anchor_subspace must have a floating-point dtype")
    sub_64 = anchor_subspace.detach().to(device="cpu", dtype=torch.float64)
    if sub_64.ndim == 1:
        sub_64 = sub_64.unsqueeze(1)
    if not bool(torch.isfinite(sub_64).all().item()):
        raise ValueError("anchor_subspace must contain only finite values")

    if isinstance(target_layers, (str, bytes)) or not isinstance(target_layers, Sequence) or not target_layers:
        raise ValueError("target_layers must be a non-empty sequence of non-negative integers")
    layer_set = {_validate_nonneg_int(l_idx, "target_layer") for l_idx in target_layers}

    expert_set: set[int] | None = None
    if expert_indices is not None:
        if isinstance(expert_indices, (str, bytes)) or not isinstance(expert_indices, Sequence) or not expert_indices:
            raise ValueError("expert_indices must be None or a non-empty sequence of non-negative integers")
        expert_set = {_validate_nonneg_int(e, "expert_index") for e in expert_indices}

    if isinstance(factors, (str, bytes)) or not isinstance(factors, Sequence) or not factors:
        raise ValueError("factors must be a non-empty sequence of LoRA B factor names")
    factor_set = {str(f) for f in factors}
    if not factor_set.issubset(_VALID_EXPERT_B_FACTORS):
        raise ValueError(f"factors must be a subset of {sorted(_VALID_EXPERT_B_FACTORS)}, got {sorted(factor_set)}")

    strength_f = _validate_unit_interval(strength, "strength")
    if strength_f <= 0.0:
        raise ValueError("strength must be > 0.0")
    _validate_bool(orthonormalize, "orthonormalize")

    hidden_dim = int(sub_64.shape[0])
    if orthonormalize:
        u_svd, s_vals, _ = torch.linalg.svd(sub_64, full_matrices=False)
        keep = s_vals > (1e-10 * max(1.0, float(s_vals[0].item())))
        q_basis = u_svd[:, keep]
    else:
        q_basis = sub_64

    eye_h = torch.eye(hidden_dim, dtype=torch.float64)
    proj_64 = (eye_h - strength_f * (q_basis @ q_basis.T)) if q_basis.shape[1] > 0 else eye_h

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        for key, tensor in list(out.items()):
            match = _EXPERT_B_KEY_RE.match(key)
            if match is None:
                continue
            l_idx, e_idx, factor = int(match.group(1)), int(match.group(2)), match.group(3)
            if l_idx not in layer_set or factor not in factor_set:
                continue
            if expert_set is not None and e_idx not in expert_set:
                continue
            if tensor.ndim != 2 or int(tensor.shape[0]) != hidden_dim:
                continue
            proj = proj_64.to(device=tensor.device)
            updated = (proj @ tensor.detach().to(torch.float64)).to(dtype=tensor.dtype)
            out[key] = updated
        return out

    return WSPStageSpec(
        name=name,
        transform_fn=_transform,
        max_edit_fraction=max_edit_fraction,
        max_edited_params=max_edited_params,
        stage_kind="cross_layer_anchor_protection",
        description="Project downstream LoRA B residual outputs orthogonal to anchor interference subspace.",
    )


def make_orthogonal_router_stage(
    router_directions: Mapping[int, torch.Tensor],
    *,
    target_layers: Sequence[int] | None = None,
    beta: float = 1.0,
    sink_expert: int | None = None,
    sink_boost: float = 0.0,
    sink_direction: torch.Tensor | None = None,
    name: str = "stage_3_orthogonal_router",
    max_edit_fraction: float | None = None,
    max_edited_params: int | None = None,
) -> WSPStageSpec:
    """Create a ``sigma_topk`` orthogonal multi-gate router surgery stage."""
    if not isinstance(router_directions, Mapping) or not router_directions:
        raise ValueError("router_directions must be a non-empty mapping of expert_index -> 1D direction tensor")
    beta_f = _validate_nonneg_finite(beta, "beta")
    boost_f = _validate_nonneg_finite(sink_boost, "sink_boost")

    dir_map: dict[int, torch.Tensor] = {}
    expected_dim: int | None = None
    for raw_idx, raw_vec in router_directions.items():
        e_idx = _validate_nonneg_int(raw_idx, "expert_index")
        if not isinstance(raw_vec, torch.Tensor) or raw_vec.ndim != 1 or raw_vec.numel() == 0:
            raise ValueError(f"router_directions[{raw_idx!r}] must be a non-empty 1D tensor")
        if not raw_vec.is_floating_point():
            raise TypeError(f"router_directions[{raw_idx!r}] must be floating-point")
        vec_64 = raw_vec.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(vec_64).all().item()):
            raise ValueError(f"router_directions[{raw_idx!r}] must be finite")
        if expected_dim is None:
            expected_dim = int(vec_64.shape[0])
        elif int(vec_64.shape[0]) != expected_dim:
            raise ValueError("All router_directions vectors must share identical hidden dimension")
        dir_map[e_idx] = vec_64

    layer_set: set[int] | None = None
    if target_layers is not None:
        if isinstance(target_layers, (str, bytes)) or not isinstance(target_layers, Sequence) or not target_layers:
            raise ValueError("target_layers must be None or a non-empty sequence of non-negative integers")
        layer_set = {_validate_nonneg_int(l_idx, "target_layer") for l_idx in target_layers}

    sink_idx: int | None = None
    sink_vec_64: torch.Tensor | None = None
    if sink_expert is not None:
        sink_idx = _validate_nonneg_int(sink_expert, "sink_expert")
        if boost_f > 0.0:
            if sink_direction is None:
                sink_vec_64 = next(iter(dir_map.values()))
            else:
                if not isinstance(sink_direction, torch.Tensor) or sink_direction.ndim != 1:
                    raise ValueError("sink_direction must be a 1D tensor")
                if int(sink_direction.shape[0]) != expected_dim:
                    raise ValueError(f"sink_direction length must match router width {expected_dim}")
                sink_vec_64 = sink_direction.detach().to(device="cpu", dtype=torch.float64)
                if not bool(torch.isfinite(sink_vec_64).all().item()):
                    raise ValueError("sink_direction must contain only finite values")

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        for key, tensor in list(out.items()):
            match = _ROUTER_KEY_RE.match(key)
            if match is None:
                continue
            l_idx = int(match.group(1))
            if layer_set is not None and l_idx not in layer_set:
                continue
            if tensor.ndim != 2 or int(tensor.shape[1]) != expected_dim:
                continue
            row_mat = tensor.detach().to(torch.float64).clone()
            for e_idx, vec_64 in dir_map.items():
                if e_idx < int(row_mat.shape[0]) and beta_f > 0.0:
                    row_mat[e_idx] = row_mat[e_idx] - beta_f * vec_64.to(device=row_mat.device)
            if sink_idx is not None and sink_vec_64 is not None and sink_idx < int(row_mat.shape[0]):
                row_mat[sink_idx] = row_mat[sink_idx] + boost_f * sink_vec_64.to(device=row_mat.device)
            out[key] = row_mat.to(dtype=tensor.dtype)
        return out

    return WSPStageSpec(
        name=name,
        transform_fn=_transform,
        max_edit_fraction=max_edit_fraction,
        max_edited_params=max_edited_params,
        stage_kind="orthogonal_multigate_router",
        description="Apply orthogonalized sigma_topk router gate updates across target layers.",
    )


def make_layernorm_calibration_stage(
    *,
    target_ln_weight: torch.Tensor | None = None,
    target_ln_bias: torch.Tensor | None = None,
    lambda_ln: float = 1.0,
    temperature_scale: float = 1.0,
    ln_weight_key: str = "ln.weight",
    ln_bias_key: str = "ln.bias",
    name: str = "stage_4_layernorm_calibration",
    max_edit_fraction: float | None = None,
    max_edited_params: int | None = None,
) -> WSPStageSpec:
    """Create a folded final-LayerNorm guard interpolation & temperature calibration stage."""
    lam_f = _validate_finite_float(lambda_ln, "lambda_ln")
    temp_f = _validate_positive_finite(temperature_scale, "temperature_scale")
    if not isinstance(ln_weight_key, str) or not ln_weight_key.strip():
        raise ValueError("ln_weight_key must be a non-empty string")
    if not isinstance(ln_bias_key, str) or not ln_bias_key.strip():
        raise ValueError("ln_bias_key must be a non-empty string")

    target_w_64: torch.Tensor | None = None
    if target_ln_weight is not None:
        if not isinstance(target_ln_weight, torch.Tensor) or target_ln_weight.ndim != 1:
            raise ValueError("target_ln_weight must be a 1D tensor")
        target_w_64 = target_ln_weight.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(target_w_64).all().item()):
            raise ValueError("target_ln_weight must contain only finite values")

    target_b_64: torch.Tensor | None = None
    if target_ln_bias is not None:
        if not isinstance(target_ln_bias, torch.Tensor) or target_ln_bias.ndim != 1:
            raise ValueError("target_ln_bias must be a 1D tensor")
        target_b_64 = target_ln_bias.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(target_b_64).all().item()):
            raise ValueError("target_ln_bias must contain only finite values")

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        if ln_weight_key in out:
            w_orig = out[ln_weight_key]
            w_64 = w_orig.detach().to(torch.float64)
            if target_w_64 is not None:
                if target_w_64.shape != w_64.shape:
                    raise ValueError(f"target_ln_weight shape {tuple(target_w_64.shape)} != {tuple(w_64.shape)}")
                w_64 = w_64 + lam_f * (target_w_64.to(device=w_64.device) - w_64)
            if temp_f != 1.0:
                w_64 = w_64 / temp_f
            out[ln_weight_key] = w_64.to(dtype=w_orig.dtype)

        if ln_bias_key in out and target_b_64 is not None:
            b_orig = out[ln_bias_key]
            b_64 = b_orig.detach().to(torch.float64)
            if target_b_64.shape != b_64.shape:
                raise ValueError(f"target_ln_bias shape {tuple(target_b_64.shape)} != {tuple(b_64.shape)}")
            b_64 = b_64 + lam_f * (target_b_64.to(device=b_64.device) - b_64)
            out[ln_bias_key] = b_64.to(dtype=b_orig.dtype)
        return out

    return WSPStageSpec(
        name=name,
        transform_fn=_transform,
        max_edit_fraction=max_edit_fraction,
        max_edited_params=max_edited_params,
        stage_kind="folded_layernorm_calibration",
        description="Interpolate final LayerNorm parameters and fold output temperature scale.",
    )


def make_quantization_compensation_stage(
    target_keys: Sequence[str],
    *,
    bits: int = 4,
    group_size: int = 32,
    rank: int = 4,
    num_alternating_steps: int = 3,
    m_prose: torch.Tensor | None = None,
    prose_weight: float = 2.0,
    use_lora_compensation: bool = True,
    name: str = "stage_5_quantization_compensation",
    max_edit_fraction: float | None = None,
    max_edited_params: int | None = None,
) -> WSPStageSpec:
    """Create a 4-bit/8-bit quantization + prose-weighted LoftQ SVD residual compensation stage."""
    if isinstance(target_keys, (str, bytes)) or not isinstance(target_keys, Sequence) or not target_keys:
        raise ValueError("target_keys must be a non-empty sequence of state_dict tensor names")
    key_tuple = tuple(str(k) for k in target_keys)
    if any(not k.strip() for k in key_tuple):
        raise ValueError("target_keys must contain only non-empty strings")
    if isinstance(bits, bool) or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    _validate_positive_int(group_size, "group_size")
    _validate_positive_int(rank, "rank")
    _validate_positive_int(num_alternating_steps, "num_alternating_steps")
    _validate_nonneg_finite(prose_weight, "prose_weight")
    _validate_bool(use_lora_compensation, "use_lora_compensation")

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        for key in key_tuple:
            if key not in out:
                raise KeyError(f"Target key {key!r} not found in state_dict")
            w_orig = out[key]
            if use_lora_compensation:
                q_deq, lora_b, lora_a, _ = compensate_quantization_error_via_lora_svd(
                    w_orig,
                    rank=rank,
                    bits=bits,
                    group_size=group_size,
                    num_alternating_steps=num_alternating_steps,
                    m_prose=m_prose,
                    prose_weight=prose_weight,
                )
                reconstructed = (
                    q_deq.detach().to(torch.float64)
                    + (lora_b.detach().to(torch.float64) @ lora_a.detach().to(torch.float64))
                )
                out[key] = reconstructed.to(device=w_orig.device, dtype=w_orig.dtype)
            else:
                q_deq, _ = symmetric_group_quantize_dequantize(
                    w_orig,
                    bits=bits,
                    group_size=group_size,
                )
                out[key] = q_deq
        return out

    return WSPStageSpec(
        name=name,
        transform_fn=_transform,
        max_edit_fraction=max_edit_fraction,
        max_edited_params=max_edited_params,
        stage_kind="quantization_compensation",
        description="Apply groupwise symmetric quantization with optional LoftQ prose-weighted SVD compensation.",
    )


__all__ = [
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_MARGIN_FRAC",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_GENERAL_GUARD_REL",
    "DEFAULT_MAX_TOTAL_EDIT_FRACTION",
    "GENERAL_EXPERT_PREFIX",
    "PIPELINE_RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "STAGE_RECEIPT_OBJECT",
    "VAULT_RECEIPT_OBJECT",
    "HoldoutVaultLockState",
    "HoldoutVaultLockedError",
    "MonotoneWSPPipelineResult",
    "WSPStageReceipt",
    "WSPStageSpec",
    "clone_state_dict",
    "compute_state_dict_sha256",
    "count_edited_parameters",
    "evaluate_anse_domain_metrics",
    "make_cross_layer_anchor_stage",
    "make_ghost_sink_stage",
    "make_layernorm_calibration_stage",
    "make_orthogonal_router_stage",
    "make_quantization_compensation_stage",
    "parse_eval_domain_nll",
    "run_monotone_wsp_stage_pipeline",
    "verify_holdout_vault_single_shot",
    "verify_pipeline_receipt_provenance",
    "verify_stage_receipt_provenance",
    "verify_vault_receipt_provenance",
]
