# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Effective-DOF BIC/MDL Complexity-Penalized Arm Selector & Two-Stage James-Stein Promoter.

Crystallizes the empirical generalization discovery observed across ``PILOT12``
(``Qwen/Qwen2.5-0.5B``) and ``PILOT16`` (``Qwen/Qwen2.5-1.5B``, locked virgin vault
``957932cc4a6fec6d...``), where unregularized ``Full-DOF`` (``1,536`` DOF) narrowly
edged ``CrossFit JS-Rank-4`` (``4`` DOF) on held-out selection by noisy ``0.000287`` nats
(``-0.008264`` vs ``-0.007977``) only to drop ``-25.6%`` on confirm (``-0.006150``, last
place) while ``CrossFit JS-Rank-4`` retained ``95.3%`` (``-0.007602``, ``#1`` NLL and
``#1`` ECE ``-39.21%``):

1. :func:`compute_effective_dof_and_bic_penalty`:
   Computes effective degrees of freedom ``k_eff = sum(c_j)`` for James-Stein shrinkage
   arms vs ``full_dim`` for unregularized Full-DOF vs ``1.0`` for 1-DOF scalar, and the
   exact MDL/BIC complexity penalty ``(k_eff * log(n_sel_tokens)) / (2 * n_sel_tokens)``.
2. :func:`select_variance_guarded_champion`:
   Applies two-stage BIC/MDL complexity penalization and a paired selection-variance
   guard so higher-DOF arms can displace lower-DOF arms if and only if their held-out
   improvement exceeds both the incremental complexity penalty and ``z * paired_se``.
3. :func:`execute_verified_bic_promoted_wsp`:
   Wires the two-stage BIC/variance-guarded promoter into an end-to-end training-free
   WSP norm calibrator gated by ANSE ``Delta E < 0`` and ``D0 <= +1.0%``, enforcing
   atomic bitwise rollback under ``ANSE_MAXIMUM_PAIN`` and SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.conditional_p0_ofc_adapter_calibrator import (
    fit_crossfitted_james_stein_ofc_step,
)
from hivemind.swarm.folded_norm_ece_calibrator import (
    ANSE_MAXIMUM_PAIN,
    compute_token_nll_and_ece,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    compute_module_state_sha256,
    compute_payload_sha256,
    detect_norm_type,
)

SELECTOR_SCHEMA: str = "hivemind.effective_dof_bic_arm_selector/1"
SELECTOR_RECEIPT_OBJECT: str = "hivemind.effective_dof_bic_arm_selector_receipt"

DEFAULT_RANK_K: int = 4
DEFAULT_SECONDARY_RANK_K: int = 8
DEFAULT_FULL_DIM: int = 1536
DEFAULT_N_SEL_TOKENS: int = 5080
DEFAULT_ECE_WEIGHT: float = 0.05
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.010
DEFAULT_BOOTSTRAP_MARGIN_Z: float = 1.0
DEFAULT_MAX_STEP_NORM: float = 0.08
DEFAULT_NULLSPACE_SV_TOL: float = 1e-6
DEFAULT_RIDGE_SCALE: float = 1e-3
DEFAULT_MIN_ENERGY_DROP: float = 1e-6
DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE: float = 1e-3
DEFAULT_NUM_BINS: int = 15
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_TAU_GRID: tuple[float, ...] = (
    0.75,
    0.85,
    0.90,
    0.94,
    0.97,
    1.00,
    1.03,
    1.05,
    1.10,
    1.18,
    1.30,
    1.50,
)
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0, 1.5)
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
VALID_NORM_KINDS: frozenset[str] = frozenset({"auto", "layernorm", "rmsnorm"})

_ARM_KIND_ALIASES: dict[str, str] = {
    "base": "base",
    "base_rollback": "base",
    "identity": "base",
    "unchanged": "base",
    "scalar": "scalar",
    "1dof_scalar": "scalar",
    "scalar_folded_tau": "scalar",
    "scalar_signed_tau": "scalar",
    "folded_scalar_rmsnorm": "scalar",
    "james_stein": "james_stein",
    "james_stein_rank_k": "james_stein",
    "crossfit_js": "james_stein",
    "crossfit_js_rank_k": "james_stein",
    "crossfit_js_signed_tau": "james_stein",
    "js_rank_k": "james_stein",
    "full_dof": "full_dof",
    "unreg_full_dof": "full_dof",
    "full_dof_fisher": "full_dof",
}
VALID_ARM_KINDS: frozenset[str] = frozenset(_ARM_KIND_ALIASES.keys())

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_NORM_ATTR_CANDIDATES: tuple[str, ...] = (
    "ln",
    "norm",
    "final_norm",
    "head.norm",
    "model.norm",
    "model.language_model.norm",
    "transformer.ln_f",
)
_HEAD_ATTR_CANDIDATES: tuple[str, ...] = ("head", "lm_head", "decoder", "output")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_int(value: Any, name: str) -> int:
    """Validate that ``value`` is an integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer."""
    out = _validate_int(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer."""
    out = _validate_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding booleans)."""
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
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _canonicalize_arm_kind(arm_kind: Any) -> str:
    """Normalize and validate an arm complexity family identifier."""
    raw = _validate_nonempty_str(arm_kind, "arm_kind").lower()
    canonical = _ARM_KIND_ALIASES.get(raw)
    if canonical is None:
        raise ValueError(f"Unsupported arm_kind {arm_kind!r}; expected one of {sorted(VALID_ARM_KINDS)}")
    return canonical


def _validate_fp_tensor(
    tensor: Any,
    name: str,
    *,
    exact_ndim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty finite floating-point tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if exact_ndim is not None and tensor.ndim != exact_ndim:
        raise ValueError(f"{name} must have ndim == {exact_ndim}, got shape {tuple(tensor.shape)}")
    if any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be non-empty, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_float_seq(
    seq: Any,
    name: str,
    *,
    strictly_positive: bool = False,
    nonneg: bool = False,
) -> tuple[float, ...]:
    """Validate a non-empty sequence of finite real numbers."""
    if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of real numbers")
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty")
    vals: list[float] = []
    for idx, val in enumerate(seq):
        if strictly_positive:
            vals.append(_validate_positive_finite(val, f"{name}[{idx}]"))
        elif nonneg:
            vals.append(_validate_nonneg_finite(val, f"{name}[{idx}]"))
        else:
            vals.append(_validate_finite_float(val, f"{name}[{idx}]"))
    return tuple(sorted(set(vals)))


def _validate_general_prefixes(prefixes: Any) -> tuple[str, ...]:
    """Validate non-empty sequence of general-prose domain prefixes."""
    if isinstance(prefixes, (str, bytes, bytearray)) or not isinstance(prefixes, Sequence):
        raise TypeError("general_prefixes must be a non-empty sequence of strings")
    if not prefixes:
        raise ValueError("general_prefixes must be non-empty")
    return tuple(_validate_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(prefixes))


def _validate_index_sequence(indices: Any, dim: int, name: str) -> list[int]:
    """Validate and deduplicate integer channel indices in ``[0, dim)``."""
    if isinstance(indices, torch.Tensor):
        if indices.is_floating_point() or indices.dtype == torch.bool or indices.ndim != 1:
            raise ValueError(f"{name} tensor must be a 1D integer tensor")
        raw_list = indices.detach().to(device="cpu", dtype=torch.long).tolist()
    elif isinstance(indices, Sequence) and not isinstance(indices, (str, bytes, bytearray)):
        raw_list = [_validate_int(v, f"{name}[{i}]") for i, v in enumerate(indices)]
    else:
        raise TypeError(f"{name} must be a sequence or 1D integer tensor")

    cleaned: set[int] = set()
    for val in raw_list:
        if val < 0 or val >= dim:
            raise ValueError(f"{name} index {val} out of bounds for dim={dim}")
        cleaned.add(val)
    return sorted(cleaned)


def _validate_shrinkage_factors(raw: Any, full_dim: int) -> tuple[float, ...]:
    """Validate James-Stein shrinkage coefficients ``c_j in [0, 1]``."""
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 1 or not raw.is_floating_point():
            raise ValueError("shrinkage_factors tensor must be a 1D floating-point tensor")
        items: list[Any] = raw.detach().to(device="cpu", dtype=torch.float64).tolist()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(raw)
    else:
        raise TypeError("shrinkage_factors must be a sequence or 1D tensor of floats in [0.0, 1.0]")

    if len(items) > full_dim:
        raise ValueError(f"len(shrinkage_factors)={len(items)} exceeds full_dim={full_dim}")
    cleaned: list[float] = []
    for idx, val in enumerate(items):
        c_j = _validate_nonneg_finite(val, f"shrinkage_factors[{idx}]")
        if c_j > 1.0 + 1e-12:
            raise ValueError(f"shrinkage_factors[{idx}] must lie in [0.0, 1.0], got {c_j}")
        cleaned.append(min(1.0, c_j))
    return tuple(cleaned)


@dataclass(frozen=True)
class EffectiveDOFBICSelectorConfig:
    """Validated configuration for BIC/MDL complexity selection and ANSE-gated promotion."""

    js_rank_primary: int = DEFAULT_RANK_K
    js_rank_secondary: int = DEFAULT_SECONDARY_RANK_K
    include_secondary_js_rank: bool = True
    include_full_dof_control: bool = True
    bootstrap_margin_z: float = DEFAULT_BOOTSTRAP_MARGIN_Z
    ece_energy_weight: float = DEFAULT_ECE_WEIGHT
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    max_step_norm: float = DEFAULT_MAX_STEP_NORM
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL
    ridge_scale: float = DEFAULT_RIDGE_SCALE
    min_energy_drop: float = DEFAULT_MIN_ENERGY_DROP
    min_shrinkage_to_activate: float = DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE
    num_bins: int = DEFAULT_NUM_BINS
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        r1 = _validate_positive_int(self.js_rank_primary, "js_rank_primary")
        r2 = _validate_positive_int(self.js_rank_secondary, "js_rank_secondary")
        if r2 <= r1:
            raise ValueError(f"js_rank_secondary ({r2}) must be > js_rank_primary ({r1})")
        _validate_bool(self.include_secondary_js_rank, "include_secondary_js_rank")
        _validate_bool(self.include_full_dof_control, "include_full_dof_control")
        _validate_nonneg_finite(self.bootstrap_margin_z, "bootstrap_margin_z")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        _validate_nonneg_finite(self.max_general_drift_rel, "max_general_drift_rel")
        _validate_nonneg_finite(self.max_domain_drift_rel, "max_domain_drift_rel")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_positive_finite(self.max_step_norm, "max_step_norm")
        sv_tol = _validate_positive_finite(self.nullspace_sv_tol, "nullspace_sv_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_tol must be < 1.0, got {sv_tol}")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_nonneg_finite(self.min_energy_drop, "min_energy_drop")
        min_sh = _validate_nonneg_finite(self.min_shrinkage_to_activate, "min_shrinkage_to_activate")
        if min_sh > 1.0:
            raise ValueError(f"min_shrinkage_to_activate must be <= 1.0, got {min_sh}")
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_int(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        taus = _validate_float_seq(self.tau_grid, "tau_grid", strictly_positive=True)
        if 1.0 not in taus:
            taus = tuple(sorted({*taus, 1.0}))
        object.__setattr__(self, "tau_grid", taus)
        alphas = _validate_float_seq(self.alpha_grid, "alpha_grid", nonneg=True)
        if 0.0 not in alphas:
            alphas = tuple(sorted({*alphas, 0.0}))
        object.__setattr__(self, "alpha_grid", alphas)
        object.__setattr__(self, "general_prefixes", _validate_general_prefixes(self.general_prefixes))
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


def verify_bic_arm_selector_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != SELECTOR_RECEIPT_OBJECT or receipt.get("schema") != SELECTOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Effective DOF & MDL/BIC Complexity Penalty (PILOT12 & PILOT16)
# ---------------------------------------------------------------------------


def compute_effective_dof_and_bic_penalty(
    arm_kind: str,
    *,
    rank_k: int = DEFAULT_RANK_K,
    full_dim: int = DEFAULT_FULL_DIM,
    n_sel_tokens: int = DEFAULT_N_SEL_TOKENS,
    shrinkage_factors: Sequence[float] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute effective degrees of freedom ``k_eff`` and exact MDL/BIC penalty.

    Implements the ``PILOT12`` / ``PILOT16`` effective complexity law:

    - ``1-DOF scalar`` (``scalar``): ``k_eff = 1.0``
    - ``James-Stein rank-k`` (``james_stein``): ``k_eff = sum(c_j)`` when
      ``shrinkage_factors`` are supplied, else ``float(rank_k)``
    - ``Unregularized Full-DOF`` (``full_dof``): ``k_eff = float(full_dim)``
    - ``Base / identity`` (``base``): ``k_eff = 0.0``

    Returns exact per-token MDL/BIC complexity penalty:
    ``bic_penalty = (k_eff * log(n_sel_tokens)) / (2 * n_sel_tokens)``.
    """
    canonical = _canonicalize_arm_kind(arm_kind)
    f_dim = _validate_positive_int(full_dim, "full_dim")
    r_k = _validate_positive_int(rank_k, "rank_k")
    if r_k > f_dim:
        raise ValueError(f"rank_k ({r_k}) must be <= full_dim ({f_dim})")
    n_tok = _validate_int(n_sel_tokens, "n_sel_tokens")
    if n_tok < 2:
        raise ValueError(f"n_sel_tokens must be >= 2 for log(n) MDL/BIC scaling, got {n_tok}")

    c_tuple: tuple[float, ...] | None = None
    if shrinkage_factors is not None:
        c_tuple = _validate_shrinkage_factors(shrinkage_factors, f_dim)

    if canonical == "base":
        k_eff = 0.0
        zero_snr = False
        mean_c = 0.0
    elif canonical == "scalar":
        k_eff = 1.0
        zero_snr = False
        mean_c = 1.0
    elif canonical == "full_dof":
        k_eff = float(f_dim)
        zero_snr = False
        mean_c = 1.0
    else:  # "james_stein"
        if c_tuple is not None:
            k_eff = float(sum(c_tuple))
            mean_c = (k_eff / float(len(c_tuple))) if c_tuple else 0.0
            zero_snr = k_eff <= 1e-9
        else:
            k_eff = float(r_k)
            mean_c = 1.0
            zero_snr = False

    bic_penalty = (k_eff * math.log(float(n_tok))) / (2.0 * float(n_tok))
    tokens_per_dof = (float(n_tok) / k_eff) if k_eff > 1e-12 else None

    return {
        "arm_kind": canonical,
        "raw_arm_kind": str(arm_kind),
        "effective_dof": float(k_eff),
        "bic_penalty": float(bic_penalty),
        "rank_k": int(len(c_tuple) if (canonical == "james_stein" and c_tuple is not None) else r_k),
        "full_dim": int(f_dim),
        "n_sel_tokens": int(n_tok),
        "dof_per_token_ratio": float(k_eff / float(n_tok)),
        "tokens_per_effective_dof": tokens_per_dof,
        "shrinkage_factors": list(c_tuple) if c_tuple is not None else None,
        "mean_shrinkage_factor": float(mean_c),
        "zero_snr_contracted": bool(zero_snr),
    }


# ---------------------------------------------------------------------------
# 3. Two-Stage Variance-Guarded James-Stein Champion Promoter
# ---------------------------------------------------------------------------


def _extract_chunk_diff_vector(metrics: Mapping[str, Any], arm_name: str) -> torch.Tensor | None:
    """Extract optional per-chunk NLL differences ``cand - base`` in ``float64``."""
    raw_diffs = metrics.get("per_chunk_nll_diffs")
    if raw_diffs is not None:
        if isinstance(raw_diffs, torch.Tensor):
            t = _validate_fp_tensor(raw_diffs, f"{arm_name}.per_chunk_nll_diffs", exact_ndim=1)
            return t.detach().to(device="cpu", dtype=torch.float64)
        if isinstance(raw_diffs, Sequence) and not isinstance(raw_diffs, (str, bytes, bytearray)):
            if len(raw_diffs) == 0:
                raise ValueError(f"{arm_name}.per_chunk_nll_diffs must be non-empty")
            return torch.tensor(
                [_validate_finite_float(v, f"{arm_name}.per_chunk_nll_diffs[{i}]") for i, v in enumerate(raw_diffs)],
                dtype=torch.float64,
            )
        raise TypeError(f"{arm_name}.per_chunk_nll_diffs must be a sequence or 1D float tensor")

    raw_cand = metrics.get("per_chunk_cand_nll")
    raw_base = metrics.get("per_chunk_base_nll")
    if raw_cand is not None or raw_base is not None:
        if raw_cand is None or raw_base is None:
            raise ValueError(f"{arm_name} must provide both per_chunk_cand_nll and per_chunk_base_nll")
        t_c = _extract_chunk_diff_vector({"per_chunk_nll_diffs": raw_cand}, arm_name)
        t_b = _extract_chunk_diff_vector({"per_chunk_nll_diffs": raw_base}, arm_name)
        assert t_c is not None and t_b is not None
        if t_c.shape != t_b.shape:
            raise ValueError(f"{arm_name} per_chunk_cand_nll and per_chunk_base_nll must share length")
        return t_c - t_b
    return None


def _extract_general_drift_ratio(metrics: Mapping[str, Any], arm_name: str, canonical_kind: str) -> float:
    """Extract relative general-domain NLL drift ``(cand - base) / base``."""
    for key in ("general_drift_ratio", "d0_drift_ratio", "d0_rel_drift"):
        if key in metrics and metrics[key] is not None:
            return _validate_finite_float(metrics[key], f"{arm_name}.{key}")
    if "general_pct_vs_base" in metrics and metrics["general_pct_vs_base"] is not None:
        return _validate_finite_float(metrics["general_pct_vs_base"], f"{arm_name}.general_pct_vs_base") / 100.0
    if canonical_kind == "base":
        return 0.0
    raise ValueError(f"{arm_name} must specify general_drift_ratio (or d0_rel_drift / general_pct_vs_base)")


def _compute_paired_diff_se(
    inc_entry: Mapping[str, Any],
    chal_entry: Mapping[str, Any],
) -> float:
    """Compute paired selection standard error between ``inc_entry`` and ``chal_entry``."""
    inc_vec = inc_entry["_chunk_diffs"]
    chal_vec = chal_entry["_chunk_diffs"]
    if inc_vec is not None and chal_vec is not None and inc_vec.shape == chal_vec.shape and int(inc_vec.numel()) >= 2:
        delta = inc_vec - chal_vec
        return float(delta.std(unbiased=True).item() / math.sqrt(float(delta.numel())))
    return math.hypot(float(inc_entry["paired_se"]), float(chal_entry["paired_se"]))


def _parse_arm_selection_entry(
    arm_name: str,
    metrics: Mapping[str, Any],
    *,
    ece_weight: float,
    general_guard_rel: float,
) -> dict[str, Any]:
    """Normalize one arm's held-out selection telemetry and BIC complexity penalty."""
    chunk_diffs = _extract_chunk_diff_vector(metrics, arm_name)
    nll_diff: float | None = None
    for key in ("nll_diff", "delta_nll", "nll_mean_diff", "mean_diff"):
        if key in metrics and metrics[key] is not None:
            nll_diff = _validate_finite_float(metrics[key], f"{arm_name}.{key}")
            break
    if nll_diff is None and chunk_diffs is not None:
        nll_diff = float(chunk_diffs.mean().item())
    if nll_diff is None and metrics.get("nll_mean") is not None and metrics.get("base_nll_mean") is not None:
        nll_diff = _validate_finite_float(metrics["nll_mean"], f"{arm_name}.nll_mean") - _validate_finite_float(
            metrics["base_nll_mean"], f"{arm_name}.base_nll_mean"
        )
    if nll_diff is None:
        raise ValueError(f"{arm_name} must provide nll_diff (or nll_mean + base_nll_mean / per_chunk_nll_diffs)")

    if metrics.get("ece_diff") is not None:
        ece_diff = _validate_finite_float(metrics["ece_diff"], f"{arm_name}.ece_diff")
    elif metrics.get("ece_value") is not None and metrics.get("base_ece_value") is not None:
        ece_diff = _validate_finite_float(metrics["ece_value"], f"{arm_name}.ece_value") - _validate_finite_float(
            metrics["base_ece_value"], f"{arm_name}.base_ece_value"
        )
    else:
        ece_diff = 0.0

    paired_se = 0.0
    if chunk_diffs is not None and int(chunk_diffs.numel()) >= 2:
        paired_se = float(chunk_diffs.std(unbiased=True).item() / math.sqrt(float(chunk_diffs.numel())))
    else:
        se_found = False
        for key in ("paired_se", "selection_se", "nll_se"):
            if key in metrics and metrics[key] is not None:
                paired_se = _validate_nonneg_finite(metrics[key], f"{arm_name}.{key}")
                se_found = True
                break
        if not se_found and metrics.get("ci95_lo") is not None and metrics.get("ci95_hi") is not None:
            lo = _validate_finite_float(metrics["ci95_lo"], f"{arm_name}.ci95_lo")
            hi = _validate_finite_float(metrics["ci95_hi"], f"{arm_name}.ci95_hi")
            if hi < lo:
                raise ValueError(f"{arm_name}.ci95_hi ({hi}) must be >= ci95_lo ({lo})")
            paired_se = (hi - lo) / (2.0 * 1.959963984540054)

    arm_kind_raw = metrics.get("arm_kind")
    if arm_kind_raw is None:
        if "effective_dof" in metrics and metrics["effective_dof"] is not None:
            k_eff = _validate_nonneg_finite(metrics["effective_dof"], f"{arm_name}.effective_dof")
            n_tok = _validate_int(metrics.get("n_sel_tokens", DEFAULT_N_SEL_TOKENS), f"{arm_name}.n_sel_tokens")
            if n_tok < 2:
                raise ValueError(f"{arm_name}.n_sel_tokens must be >= 2, got {n_tok}")
            canonical = "base" if k_eff == 0.0 else ("scalar" if abs(k_eff - 1.0) <= 1e-12 else "james_stein")
            dof_info = {
                "arm_kind": canonical,
                "raw_arm_kind": canonical,
                "effective_dof": float(k_eff),
                "bic_penalty": (float(k_eff) * math.log(float(n_tok))) / (2.0 * float(n_tok)),
                "rank_k": int(metrics.get("rank_k", DEFAULT_RANK_K)),
                "full_dim": int(metrics.get("full_dim", DEFAULT_FULL_DIM)),
                "n_sel_tokens": int(n_tok),
                "dof_per_token_ratio": float(k_eff / float(n_tok)),
                "tokens_per_effective_dof": (float(n_tok) / float(k_eff)) if k_eff > 1e-12 else None,
                "shrinkage_factors": None,
                "mean_shrinkage_factor": 1.0 if k_eff > 0.0 else 0.0,
                "zero_snr_contracted": bool(metrics.get("zero_snr_contracted", False)),
            }
        else:
            raise ValueError(f"{arm_name} must specify arm_kind (or effective_dof)")
    else:
        dof_info = compute_effective_dof_and_bic_penalty(
            arm_kind_raw,
            rank_k=metrics.get("rank_k", DEFAULT_RANK_K),
            full_dim=metrics.get("full_dim", DEFAULT_FULL_DIM),
            n_sel_tokens=metrics.get("n_sel_tokens", DEFAULT_N_SEL_TOKENS),
            shrinkage_factors=metrics.get("shrinkage_factors"),
        )

    canonical_kind = str(dof_info["arm_kind"])
    gen_drift = _extract_general_drift_ratio(metrics, arm_name, canonical_kind)
    worst_drift = (
        _validate_finite_float(metrics["worst_domain_drift_ratio"], f"{arm_name}.worst_domain_drift_ratio")
        if metrics.get("worst_domain_drift_ratio") is not None
        else gen_drift
    )
    explicit_guard = metrics.get("guard_ok")
    if explicit_guard is not None:
        _validate_bool(explicit_guard, f"{arm_name}.guard_ok")
    guard_ok = bool(
        gen_drift <= general_guard_rel + 1e-12
        and worst_drift <= general_guard_rel + 1e-12
        and (True if explicit_guard is None else explicit_guard)
    )

    empirical_risk = float(nll_diff) + float(ece_weight) * float(ece_diff)
    bic_penalty = float(dof_info["bic_penalty"])
    bic_obj = empirical_risk + bic_penalty
    anse_energy = (
        empirical_risk
        if guard_ok
        else ANSE_MAXIMUM_PAIN + max(0.0, float(nll_diff)) + float(ece_weight) * max(0.0, float(ece_diff))
    )

    return {
        "arm_name": arm_name,
        "arm_kind": canonical_kind,
        "nll_mean_diff": float(nll_diff),
        "ece_diff": float(ece_diff),
        "empirical_risk": float(empirical_risk),
        "effective_dof": float(dof_info["effective_dof"]),
        "bic_penalty": bic_penalty,
        "bic_penalized_objective": float(bic_obj),
        "n_sel_tokens": int(dof_info["n_sel_tokens"]),
        "paired_se": float(paired_se),
        "general_drift_ratio": float(gen_drift),
        "worst_domain_drift_ratio": float(worst_drift),
        "guard_ok": guard_ok,
        "zero_snr_contracted": bool(dof_info["zero_snr_contracted"]),
        "mean_shrinkage_factor": float(dof_info["mean_shrinkage_factor"]),
        "shrinkage_factors": dof_info["shrinkage_factors"],
        "anse_energy": float(anse_energy),
        "_chunk_diffs": chunk_diffs,
    }


def select_variance_guarded_champion(
    arm_selection_metrics: Mapping[str, Mapping[str, Any]],
    *,
    ece_weight: float = DEFAULT_ECE_WEIGHT,
    general_guard_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL,
    bootstrap_margin_z: float = DEFAULT_BOOTSTRAP_MARGIN_Z,
) -> dict[str, Any]:
    """Select the winning arm using BIC/MDL complexity penalization + paired variance guard.

    Stage 1 filters infeasible arms breaching ``D0 <= general_guard_rel``, failing
    to reduce empirical ANSE risk (``nll_diff + ece_weight * ece_diff < 0``), or
    exhibiting zero-SNR James-Stein contraction (``sum(c_j) == 0``).

    Stage 2 orders feasible arms by ascending ``effective_dof`` and permits a
    higher-DOF challenger to displace the lower-DOF incumbent **only** when::

        risk(incumbent) - risk(challenger) > (bic(challenger) - bic(incumbent)) + z * paired_se

    and ``bic_penalized_objective(challenger) < 0.0`` (for ``k_eff > 1.0``).
    """
    if not isinstance(arm_selection_metrics, Mapping) or len(arm_selection_metrics) == 0:
        raise ValueError("arm_selection_metrics must be a non-empty mapping of arm_name -> metrics")
    w_ece = _validate_nonneg_finite(ece_weight, "ece_weight")
    g_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    z_margin = _validate_nonneg_finite(bootstrap_margin_z, "bootstrap_margin_z")

    evaluated: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    feasible: list[dict[str, Any]] = []

    for raw_name, raw_metrics in arm_selection_metrics.items():
        arm_name = _validate_nonempty_str(raw_name, "arm_name")
        if not isinstance(raw_metrics, Mapping):
            raise TypeError(f"arm_selection_metrics[{arm_name!r}] must be a Mapping")
        entry = _parse_arm_selection_entry(
            arm_name, raw_metrics, ece_weight=w_ece, general_guard_rel=g_rel
        )
        evaluated[arm_name] = entry

        if entry["arm_kind"] == "base":
            rejected[arm_name] = "baseline_reference"
        elif not entry["guard_ok"]:
            rejected[arm_name] = "general_or_domain_guard_violation"
        elif entry["zero_snr_contracted"]:
            rejected[arm_name] = "zero_snr_contracted_to_scalar"
        elif entry["empirical_risk"] >= 0.0:
            rejected[arm_name] = "non_improving_selection_risk"
        else:
            feasible.append(entry)

    any_guard_breach = any((not e["guard_ok"]) and e["arm_kind"] != "base" for e in evaluated.values())
    has_scalar = any(e["arm_kind"] == "scalar" and e["guard_ok"] and e["empirical_risk"] < 0.0 for e in evaluated.values())
    has_zero_snr_js = any(e["zero_snr_contracted"] for e in evaluated.values())

    def _public_arms() -> dict[str, dict[str, Any]]:
        return {
            k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
            for k, v in evaluated.items()
        }

    def _empty_result(status_str: str, anse_e: float) -> dict[str, Any]:
        return {
            "selected_arm": None,
            "selected_arm_kind": None,
            "status": status_str,
            "fallback_to_scalar": False,
            "effective_dof": 0.0,
            "bic_penalty": 0.0,
            "nll_mean_diff": 0.0,
            "ece_diff": 0.0,
            "empirical_risk": 0.0,
            "bic_penalized_objective": 0.0,
            "general_drift_ratio": 0.0,
            "anse_energy": float(anse_e),
            "evaluated_arms": _public_arms(),
            "promotion_trace": [],
            "rejected_arms": rejected,
        }

    if not feasible:
        return _empty_result(
            "NO_FEASIBLE_ARM" if any_guard_breach else "NO_COMPLEXITY_JUSTIFIED_ARM",
            ANSE_MAXIMUM_PAIN if any_guard_breach else 0.0,
        )

    feasible.sort(key=lambda e: (e["effective_dof"], e["bic_penalized_objective"], e["arm_name"]))
    incumbent = feasible[0]
    promotion_trace: list[dict[str, Any]] = []
    promoted_higher_dof = False

    for challenger in feasible[1:]:
        k_inc = incumbent["effective_dof"]
        k_chal = challenger["effective_dof"]
        same_tier = abs(k_chal - k_inc) <= 1e-12
        raw_gain = incumbent["empirical_risk"] - challenger["empirical_risk"]
        delta_bic = max(0.0, challenger["bic_penalty"] - incumbent["bic_penalty"])
        pair_se = 0.0 if same_tier else _compute_paired_diff_se(incumbent, challenger)
        required_margin = 0.0 if same_tier else (delta_bic + z_margin * pair_se)

        if same_tier:
            promoted = raw_gain > 1e-12
            reason = "lower_risk_within_equal_dof_tier" if promoted else "higher_risk_within_equal_dof_tier"
        else:
            bic_beats_base = k_chal <= 1.0 + 1e-12 or challenger["bic_penalized_objective"] < 0.0
            promoted = bool(raw_gain > required_margin and bic_beats_base)
            if promoted:
                reason = "cleared_bic_and_paired_variance_guard"
            elif raw_gain <= delta_bic or not bic_beats_base:
                reason = "rejected_by_bic_complexity_penalty"
            else:
                reason = "rejected_by_paired_selection_variance_guard"

        promotion_trace.append(
            {
                "incumbent": incumbent["arm_name"],
                "challenger": challenger["arm_name"],
                "incumbent_dof": float(k_inc),
                "challenger_dof": float(k_chal),
                "raw_gain": float(raw_gain),
                "delta_bic": float(delta_bic),
                "paired_se": float(pair_se),
                "required_margin": float(required_margin),
                "promoted": promoted,
                "reason": reason,
            }
        )
        if promoted:
            if not same_tier:
                promoted_higher_dof = True
            incumbent = challenger
        else:
            rejected[challenger["arm_name"]] = reason

    if incumbent["effective_dof"] > 1.0 + 1e-12 and incumbent["bic_penalized_objective"] >= 0.0:
        rejected[incumbent["arm_name"]] = "bic_penalty_exceeds_selection_gain"
        res = _empty_result("NO_COMPLEXITY_JUSTIFIED_ARM", 0.0)
        res["promotion_trace"] = promotion_trace
        return res

    fallback_to_scalar = bool(
        incumbent["arm_kind"] == "scalar"
        and (has_zero_snr_js or any(t["challenger"] != incumbent["arm_name"] and not t["promoted"] for t in promotion_trace))
    )
    if promoted_higher_dof:
        status = "PROMOTED_HIGHER_DOF"
    elif fallback_to_scalar and has_scalar:
        status = "FELL_BACK_TO_SCALAR"
    else:
        status = "RETAINED_LOW_DOF_ANCHOR"

    return {
        "selected_arm": incumbent["arm_name"],
        "selected_arm_kind": incumbent["arm_kind"],
        "status": status,
        "fallback_to_scalar": fallback_to_scalar,
        "effective_dof": float(incumbent["effective_dof"]),
        "bic_penalty": float(incumbent["bic_penalty"]),
        "nll_mean_diff": float(incumbent["nll_mean_diff"]),
        "ece_diff": float(incumbent["ece_diff"]),
        "empirical_risk": float(incumbent["empirical_risk"]),
        "bic_penalized_objective": float(incumbent["bic_penalized_objective"]),
        "general_drift_ratio": float(incumbent["general_drift_ratio"]),
        "anse_energy": float(incumbent["anse_energy"]),
        "evaluated_arms": _public_arms(),
        "promotion_trace": promotion_trace,
        "rejected_arms": rejected,
    }


# ---------------------------------------------------------------------------
# 4. End-to-End BIC-Promoted WSP Executor with ANSE Ratchet & Rollback
# ---------------------------------------------------------------------------


def _resolve_dotted_attr(root: Any, attr_path: str) -> Any:
    """Resolve a dotted attribute path on ``root``, returning ``None`` if missing."""
    cur = root
    for part in attr_path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _resolve_norm_and_head(
    model: nn.Module,
    *,
    norm_kind: str,
    norm_attr: str | None,
    head_attr: str | None,
) -> tuple[nn.Module, nn.Module, str, str, str, int, int, float]:
    """Locate final normalization and unembedding head modules."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if norm_kind not in VALID_NORM_KINDS:
        raise ValueError(f"norm_kind must be one of {sorted(VALID_NORM_KINDS)}, got {norm_kind!r}")

    if norm_attr is not None:
        norm_name = _validate_nonempty_str(norm_attr, "norm_attr")
        norm_mod = _resolve_dotted_attr(model, norm_name)
    else:
        norm_name = ""
        norm_mod = None
        for cand in _NORM_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                norm_name = cand
                norm_mod = found
                break

    if not isinstance(norm_mod, nn.Module):
        raise ValueError("model does not expose a recognizable final normalization module")
    w_param = _validate_fp_tensor(getattr(norm_mod, "weight", None), f"model.{norm_name}.weight", exact_ndim=1)
    hidden_dim = int(w_param.shape[0])
    resolved_norm = detect_norm_type(norm_mod, norm_type=norm_kind)
    b_param = getattr(norm_mod, "bias", None)
    if b_param is not None:
        b_t = _validate_fp_tensor(b_param, f"model.{norm_name}.bias", exact_ndim=1)
        if int(b_t.shape[0]) != hidden_dim:
            raise ValueError(f"model.{norm_name}.bias length {b_t.shape[0]} must match {hidden_dim}")

    raw_eps = getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", None))
    eps_eff = (
        _validate_positive_finite(raw_eps, f"model.{norm_name}.eps")
        if raw_eps is not None
        else (1e-6 if resolved_norm == "rmsnorm" else 1e-5)
    )

    if head_attr is not None:
        head_name = _validate_nonempty_str(head_attr, "head_attr")
        head_mod = _resolve_dotted_attr(model, head_name)
    else:
        head_name = ""
        head_mod = None
        for cand in _HEAD_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                head_name = cand
                head_mod = found
                break

    if not isinstance(head_mod, nn.Module) or not callable(head_mod):
        raise ValueError("model does not expose a callable unembedding head module")
    head_w = _validate_fp_tensor(getattr(head_mod, "weight", None), f"model.{head_name}.weight", exact_ndim=2)
    if int(head_w.shape[1]) != hidden_dim:
        raise ValueError(f"model.{head_name}.weight width {head_w.shape[1]} must match norm dim {hidden_dim}")
    head_b = getattr(head_mod, "bias", None)
    if head_b is not None:
        hb_t = _validate_fp_tensor(head_b, f"model.{head_name}.bias", exact_ndim=1)
        if int(hb_t.shape[0]) != int(head_w.shape[0]):
            raise ValueError(f"model.{head_name}.bias length must match vocab size {head_w.shape[0]}")

    return norm_mod, head_mod, resolved_norm, norm_name, head_name, hidden_dim, int(head_w.shape[0]), eps_eff


def _is_general_domain(domain_label: str, general_prefixes: tuple[str, ...]) -> bool:
    """Return True if ``domain_label`` matches a general-prose prefix."""
    return any(domain_label.startswith(pref) for pref in general_prefixes)


def _to_long_1d(raw: Any, name: str) -> torch.Tensor:
    """Convert a sequence or 1D integer tensor into CPU int64."""
    if isinstance(raw, torch.Tensor):
        if raw.is_floating_point() or raw.dtype == torch.bool:
            raise TypeError(f"{name} tensor must have an integer dtype")
        out = raw.detach().to(device="cpu", dtype=torch.long)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for elem in raw:
            _validate_int(elem, f"{name} element")
        out = torch.tensor(list(raw), dtype=torch.long)
    else:
        raise TypeError(f"{name} must be a sequence or 1D integer tensor")
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(out.shape)}")
    return out


def _prepare_causal_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    vocab_size: int,
    score_from: int,
    general_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Validate causal next-token chunks containing both general and specialist domains."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")

    rows: list[torch.Tensor] = []
    domains: list[str] = []
    seq_len: int | None = None
    start_pos = max(0, score_from - 1)

    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids' and domain label")
        dom = _validate_nonempty_str(ch.get("expert_id", ch.get("domain")), f"{name}[{idx}] domain")
        ids_t = _to_long_1d(ch["ids"], f"{name}[{idx}]['ids']")
        if ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must have length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
            if start_pos >= seq_len - 1:
                raise ValueError(f"{name} sequence length {seq_len} is too short for score_from={score_from}")
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}, got {ids_t.numel()}")
        rows.append(ids_t)
        domains.append(dom)

    ids_2d = torch.stack(rows, dim=0)
    is_gen = torch.tensor([_is_general_domain(d, general_prefixes) for d in domains], dtype=torch.bool)
    if not bool(is_gen.any().item()) or not bool((~is_gen).any().item()):
        raise ValueError(f"{name} must contain both general ({general_prefixes}) and specialist chunks")
    return {
        "ids_2d": ids_2d,
        "domains": domains,
        "is_general": is_gen,
        "targets_2d": ids_2d[:, start_pos + 1 :].contiguous(),
    }


def _split_probe_chunks_fallback(
    chunks: Sequence[Mapping[str, Any]],
    general_prefixes: tuple[str, ...],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministically partition probe chunks into fit and held-out selection folds."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence):
        raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")
    fit_chunks = [ch for i, ch in enumerate(chunks) if i % 2 == 0]
    sel_chunks = [ch for i, ch in enumerate(chunks) if i % 2 == 1]
    for label, sub in (("fit", fit_chunks), ("selection", sel_chunks)):
        if len(sub) < 2:
            raise ValueError(f"Need at least 4 probe chunks for automatic split (got {len(chunks)})")
        doms = [str(ch.get("expert_id", ch.get("domain", ""))) for ch in sub]
        if not any(_is_general_domain(d, general_prefixes) for d in doms) or not any(
            not _is_general_domain(d, general_prefixes) for d in doms
        ):
            raise ValueError(f"Automatic {label} fold lacks both general and specialist chunks")
    return fit_chunks, sel_chunks


def _capture_pre_norm_states(
    model: nn.Module,
    norm_mod: nn.Module,
    ids_2d: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Capture pre-norm hidden states cleanly using a read-only forward pre-hook."""
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    captured: list[torch.Tensor] = []

    def _pre_hook(_mod: nn.Module, inp: tuple[Any, ...]) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            captured.append(inp[0].detach().to(device="cpu"))

    handle = norm_mod.register_forward_pre_hook(_pre_hook)
    parts: list[torch.Tensor] = []
    try:
        for b0 in range(0, int(ids_2d.shape[0]), batch_size):
            batch_ids = ids_2d[b0 : b0 + batch_size].to(dev)
            captured.clear()
            _ = model(batch_ids)
            if not captured:
                raise RuntimeError("Final normalization module was not invoked during model forward pass")
            h_batch = captured[-1]
            if h_batch.shape[:2] != batch_ids.shape:
                raise ValueError(
                    f"Captured pre-norm shape {tuple(h_batch.shape)} incompatible with {tuple(batch_ids.shape)}"
                )
            if not bool(torch.isfinite(h_batch).all().item()):
                raise ValueError("Model trunk produced non-finite pre-norm hidden states")
            parts.append(h_batch)
    finally:
        handle.remove()

    return torch.cat(parts, dim=0)


def _extract_unit_states(
    h_pre_norm: torch.Tensor,
    *,
    norm_kind: str,
    eps: float,
    score_from: int,
) -> torch.Tensor:
    """Slice scored causal positions and normalize to unit pre-scale states in ``float64``."""
    start_pos = max(0, score_from - 1)
    h_scored = h_pre_norm.detach().to(device="cpu", dtype=torch.float64)[:, start_pos:-1, :]
    dim = int(h_scored.shape[-1])
    if norm_kind == "rmsnorm":
        return (h_scored * torch.rsqrt(h_scored.pow(2).mean(dim=-1, keepdim=True) + float(eps))).contiguous()
    return F.layer_norm(h_scored, (dim,), None, None, eps=float(eps)).contiguous()


def _compute_norm_chunk_gradients(
    u_scored: torch.Tensor,
    targets_2d: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-chunk ``d(NLL)/d(norm.weight)`` and diagonal Fisher in ``float64``."""
    c_num, t_len, dim = u_scored.shape
    x_norm = u_scored * w0
    if b0 is not None:
        x_norm = x_norm + b0
    logits = F.linear(x_norm.reshape(-1, dim), head_w, head_b)
    probs = F.softmax(logits, dim=-1)
    t_flat = targets_2d.reshape(-1)
    probs[torch.arange(t_flat.numel()), t_flat] -= 1.0
    tok_grads = (probs @ head_w).reshape(c_num, t_len, dim) * u_scored
    tok_fisher = tok_grads.pow(2).mean(dim=(0, 1)).contiguous()
    return tok_grads.mean(dim=1).contiguous(), tok_fisher


def _eval_logits_summary(
    logits: torch.Tensor,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    *,
    num_bins: int,
    ignore_index: int,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Compute per-chunk NLLs, per-domain NLLs, and global NLL/ECE from logits."""
    flat_logits = logits.reshape(-1, int(logits.shape[-1]))
    flat_targets = targets_2d.reshape(-1)
    tok_nll = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="none")
    seq_nll = tok_nll.reshape(int(targets_2d.shape[0]), int(targets_2d.shape[1])).mean(dim=1)

    global_m = compute_token_nll_and_ece(logits, targets_2d, num_bins=num_bins, ignore_index=ignore_index)
    dom_indices: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        dom_indices.setdefault(dom, []).append(idx)
    dom_nll = {
        dom: float(seq_nll[torch.tensor(ix, dtype=torch.long)].mean().item())
        for dom, ix in sorted(dom_indices.items())
    }
    return seq_nll, dom_nll, {
        "nll_mean": float(global_m["nll_mean"]),
        "ece_value": float(global_m["expected_calibration_error"]),
        "num_tokens": int(global_m["num_tokens"]),
    }


def _compute_anse_verdict(
    ref_dom_nll: Mapping[str, float],
    ref_ece: float,
    cand_dom_nll: Mapping[str, float],
    cand_ece: float,
    *,
    cfg: EffectiveDOFBICSelectorConfig,
) -> dict[str, Any]:
    """Evaluate ANSE thermodynamic energy, ``D0`` drift ratio, and domain guards."""
    ratios: dict[str, float] = {}
    for dom in sorted(cand_dom_nll.keys()):
        base_v = float(ref_dom_nll[dom])
        cand_v = float(cand_dom_nll[dom])
        ratios[dom] = cand_v / max(base_v, 1e-12)

    mean_base = sum(float(ref_dom_nll[d]) for d in cand_dom_nll) / float(len(cand_dom_nll))
    mean_cand = sum(float(cand_dom_nll[d]) for d in cand_dom_nll) / float(len(cand_dom_nll))
    mean_diff = mean_cand - mean_base
    ece_diff = float(cand_ece) - float(ref_ece)

    gen_keys = [d for d in sorted(ratios.keys()) if _is_general_domain(d, cfg.general_prefixes)]
    gen_ratio = max(ratios[d] for d in gen_keys)
    worst_dom = max(sorted(ratios.keys()), key=lambda d: ratios[d])
    worst_ratio = ratios[worst_dom]

    guard_ok = bool(
        gen_ratio <= (1.0 + cfg.max_general_drift_rel) + 1e-12
        and worst_ratio <= (1.0 + cfg.max_domain_drift_rel) + 1e-12
    )
    anse_energy = (
        mean_diff + cfg.ece_energy_weight * ece_diff
        if guard_ok
        else cfg.maximum_pain_energy + max(0.0, mean_diff) + cfg.ece_energy_weight * max(0.0, float(cand_ece))
    )
    return {
        "nll_mean_diff": float(mean_diff),
        "ece_diff": float(ece_diff),
        "ece_rel_delta": float(ece_diff / max(float(ref_ece), 1e-12)),
        "general_drift_ratio": float(gen_ratio - 1.0),
        "worst_domain_id": worst_dom,
        "worst_domain_drift_ratio": float(worst_ratio - 1.0),
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "delta_e": float(anse_energy),
    }


def _fit_unreg_full_dof_nullspace_step(
    spec_grads: torch.Tensor,
    gen_grads: torch.Tensor,
    tok_fisher: torch.Tensor,
    *,
    protected_indices: Sequence[int],
    max_step_norm: float,
    ridge_scale: float,
    sv_tol: float,
) -> tuple[torch.Tensor, float, float]:
    """Fit unregularized full-DOF specialist Fisher step inside ``ker(G_D0) intersect S_val^perp``."""
    dim = int(spec_grads.shape[1])
    g0_mean = gen_grads.mean(dim=0, keepdim=True)
    constraints: list[torch.Tensor] = []
    if float(g0_mean.norm().item()) > 1e-15:
        constraints.append(g0_mean)
    if protected_indices:
        eye_ofc = torch.zeros((len(protected_indices), dim), dtype=torch.float64)
        for ri, ci in enumerate(protected_indices):
            eye_ofc[ri, ci] = 1.0
        constraints.append(eye_ofc)

    if constraints:
        c_mat = torch.cat(constraints, dim=0)
        _, s_c, vh_c = torch.linalg.svd(c_mat, full_matrices=False)
        s_top = float(s_c[0].item()) if s_c.numel() > 0 else 0.0
        keep_c = int((s_c > sv_tol * max(1.0, s_top)).sum().item()) if s_top > 1e-15 else 0
        q_c = vh_c[:keep_c, :].T.contiguous()
    else:
        q_c = torch.empty((dim, 0), dtype=torch.float64)

    idx_t = torch.tensor(protected_indices, dtype=torch.long) if protected_indices else None

    def _proj(v: torch.Tensor) -> torch.Tensor:
        out = v
        if q_c.shape[1] > 0:
            for _ in range(2):
                out = out - (out @ q_c) @ q_c.T
        if idx_t is not None:
            out = out.clone()
            out[idx_t] = 0.0
        return out

    raw = -spec_grads.mean(dim=0) / torch.sqrt(tok_fisher + max(ridge_scale, 1e-12))
    step = _proj(raw)
    nrm = float(step.norm().item())
    if nrm > max_step_norm:
        step = _proj(step * (max_step_norm / nrm))
    d0_err = float(abs((g0_mean @ step).item()))
    ofc_max = float(step[idx_t].abs().max().item()) if idx_t is not None else 0.0
    return step, d0_err, ofc_max


def _search_probe_tau_alpha(
    l_base: torch.Tensor,
    l_dw: torch.Tensor | None,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    base_dom: Mapping[str, float],
    base_ece: float,
    *,
    cfg: EffectiveDOFBICSelectorConfig,
) -> dict[str, Any]:
    """Search ``L(tau, alpha) = tau * L_base + alpha * L_dw`` on probe split."""
    best: dict[str, Any] = {
        "found": False,
        "tau": 1.0,
        "alpha": 0.0,
        "probe_energy": 0.0,
    }
    alpha_vals = (0.0,) if l_dw is None else cfg.alpha_grid
    for tau in cfg.tau_grid:
        for alpha in alpha_vals:
            if abs(float(tau) - 1.0) <= 1e-12 and abs(float(alpha)) <= 1e-12:
                continue
            if l_dw is not None and alpha == 0.0:
                continue
            logits = l_base * float(tau) if l_dw is None else (l_base * float(tau) + l_dw * float(alpha))
            _, dom_m, glob_m = _eval_logits_summary(
                logits, targets_2d, domains, num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
            )
            ver = _compute_anse_verdict(base_dom, base_ece, dom_m, glob_m["ece_value"], cfg=cfg)
            if ver["guard_ok"] and ver["anse_energy"] < best["probe_energy"] - cfg.min_energy_drop:
                best = {
                    "found": True,
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "probe_energy": float(ver["anse_energy"]),
                }
    return best


def _eval_units_with_params(
    u_scored: torch.Tensor,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b0: torch.Tensor | None,
    dw: torch.Tensor | None,
    *,
    tau: float,
    alpha: float,
    cfg: EffectiveDOFBICSelectorConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Evaluate candidate norm/head weights on unit normalized states."""
    w_cand = w0 * float(tau) + (float(alpha) * dw if (dw is not None and alpha != 0.0) else 0.0)
    b_cand = (b0 * float(tau)) if b0 is not None else None
    hb_cand = (head_b0 * float(tau)) if head_b0 is not None else None
    x_norm = u_scored * w_cand
    if b_cand is not None:
        x_norm = x_norm + b_cand
    logits = F.linear(x_norm, head_w, hb_cand)
    return _eval_logits_summary(
        logits, targets_2d, domains, num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
    )


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a detached CPU clone of ``model.state_dict()``."""
    return {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}


def _restore_state_inplace(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``snapshot`` onto ``model`` matching parameter devices and dtypes."""
    cur_sd = model.state_dict()
    model.load_state_dict(
        {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()}
    )


def _apply_norm_candidate_inplace(
    norm_mod: nn.Module,
    head_mod: nn.Module,
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_b0: torch.Tensor | None,
    dw: torch.Tensor | None,
    *,
    tau: float,
    alpha: float,
) -> None:
    """Write ``w = tau * w0 + alpha * dw``, ``b = tau * b0``, ``head_b = tau * head_b0``."""
    w_new = w0 * float(tau) + (float(alpha) * dw if (dw is not None and alpha != 0.0) else 0.0)
    norm_mod.weight.copy_(w_new.to(device=norm_mod.weight.device, dtype=norm_mod.weight.dtype))
    if getattr(norm_mod, "bias", None) is not None and b0 is not None:
        norm_mod.bias.copy_((b0 * float(tau)).to(device=norm_mod.bias.device, dtype=norm_mod.bias.dtype))
    if getattr(head_mod, "bias", None) is not None and head_b0 is not None:
        head_mod.bias.copy_((head_b0 * float(tau)).to(device=head_mod.bias.device, dtype=head_mod.bias.dtype))


@torch.no_grad()
def execute_verified_bic_promoted_wsp(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    protected_indices: Sequence[int] | torch.Tensor = (),
    config: EffectiveDOFBICSelectorConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute BIC/MDL + variance-guarded WSP promotion under ANSE ``Delta E < 0`` and ``D0 <= +1.0%``.

    Fits 1-DOF folded-tau scalar, cross-fitted James-Stein rank-``k`` arms inside
    ``ker(G_D0) intersect S_val^perp``, and an optional unregularized Full-DOF diagnostic
    arm on ``probe_chunks``; runs two-stage BIC/MDL complexity penalization and paired
    selection-variance promotion on ``selection_chunks``; verifies ANSE ``Delta E < 0``
    and ``D0 <= +1.0%`` on ``eval_chunks`` with atomic bitwise rollback and SHA-256 receipts.
    """
    if config is not None and not isinstance(config, EffectiveDOFBICSelectorConfig):
        raise TypeError(f"config must be EffectiveDOFBICSelectorConfig or None, got {type(config).__name__}")
    cfg = config or EffectiveDOFBICSelectorConfig()
    _validate_bool(inplace, "inplace")

    _, _, _, norm_name_chk, head_name_chk, hidden_dim, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
    ofc_clean = _validate_index_sequence(protected_indices, hidden_dim, "protected_indices")

    if selection_chunks is None:
        fit_raw, sel_raw = _split_probe_chunks_fallback(probe_chunks, cfg.general_prefixes)
    else:
        fit_raw, sel_raw = probe_chunks, selection_chunks

    fit_prep = _prepare_causal_chunks(
        fit_raw, name="probe_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
    )
    sel_prep = _prepare_causal_chunks(
        sel_raw, name="selection_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
    )
    ver_prep = (
        sel_prep
        if eval_chunks is None
        else _prepare_causal_chunks(
            eval_chunks, name="eval_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
        )
    )

    work_model = model if inplace else copy.deepcopy(model)
    norm_mod, head_mod, resolved_norm, norm_name, head_name, _, _, eps_eff = _resolve_norm_and_head(
        work_model, norm_kind=norm_kind, norm_attr=norm_name_chk, head_attr=head_name_chk
    )
    n_sel_tokens = max(2, int((sel_prep["targets_2d"] != cfg.ignore_index).sum().item()))

    was_training = work_model.training
    work_model.eval()
    try:
        initial_snapshot = _snapshot_state(work_model)
        initial_sha = compute_module_state_sha256(work_model)

        w0 = norm_mod.weight.detach().to(device="cpu", dtype=torch.float64).clone()
        b0_raw = getattr(norm_mod, "bias", None)
        b0 = b0_raw.detach().to(device="cpu", dtype=torch.float64).clone() if b0_raw is not None else None
        head_w = head_mod.weight.detach().to(device="cpu", dtype=torch.float64)
        hb_raw = getattr(head_mod, "bias", None)
        head_b = hb_raw.detach().to(device="cpu", dtype=torch.float64).clone() if hb_raw is not None else None

        h_fit = _capture_pre_norm_states(work_model, norm_mod, fit_prep["ids_2d"], batch_size=cfg.batch_size)
        h_sel = _capture_pre_norm_states(work_model, norm_mod, sel_prep["ids_2d"], batch_size=cfg.batch_size)
        h_ver = (
            h_sel
            if eval_chunks is None
            else _capture_pre_norm_states(work_model, norm_mod, ver_prep["ids_2d"], batch_size=cfg.batch_size)
        )

        u_fit = _extract_unit_states(h_fit, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        u_sel = _extract_unit_states(h_sel, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        u_ver = (
            u_sel
            if eval_chunks is None
            else _extract_unit_states(h_ver, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        )

        chunk_grads, tok_fisher = _compute_norm_chunk_gradients(
            u_fit, fit_prep["targets_2d"], w0, b0, head_w, head_b
        )
        gen_grads = chunk_grads[fit_prep["is_general"]]
        spec_grads = chunk_grads[~fit_prep["is_general"]]

        # Probe base logits and metrics
        l_base_fit = F.linear(u_fit * w0 + (b0 if b0 is not None else 0.0), head_w, head_b)
        _, fit_base_dom, fit_base_g = _eval_logits_summary(
            l_base_fit, fit_prep["targets_2d"], fit_prep["domains"], num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
        )

        # Build candidate arms on probe_chunks
        arm_specs: dict[str, dict[str, Any]] = {}

        sc_fit = _search_probe_tau_alpha(
            l_base_fit, None, fit_prep["targets_2d"], fit_prep["domains"], fit_base_dom, fit_base_g["ece_value"], cfg=cfg
        )
        arm_specs["scalar_folded_tau"] = {
            "arm_kind": "scalar",
            "rank_k": 1,
            "shrinkage_factors": None,
            "tau": float(sc_fit["tau"]),
            "alpha": 0.0,
            "dw": None,
            "ofc_max": 0.0,
            "d0_ortho_err": 0.0,
            "zero_snr": False,
        }

        js_ranks: list[tuple[str, int]] = [(f"crossfit_js_rank_{cfg.js_rank_primary}", cfg.js_rank_primary)]
        if cfg.include_secondary_js_rank:
            js_ranks.append((f"crossfit_js_rank_{cfg.js_rank_secondary}", cfg.js_rank_secondary))

        for arm_label, r_k in js_ranks:
            js_fit = fit_crossfitted_james_stein_ofc_step(
                spec_grads,
                gen_grads,
                protected_indices=ofc_clean,
                max_rank=r_k,
                max_step_norm=cfg.max_step_norm,
                sv_tol=cfg.nullspace_sv_tol,
            )
            dw_js = None if js_fit["zero_step"] else js_fit["delta_w"].to(dtype=torch.float64)
            zero_snr = bool(
                js_fit["zero_step"] or float(js_fit["mean_shrinkage_factor"]) < cfg.min_shrinkage_to_activate
            )
            if dw_js is not None and not zero_snr:
                l_dw = F.linear(u_fit * dw_js, head_w, None)
                js_srch = _search_probe_tau_alpha(
                    l_base_fit,
                    l_dw,
                    fit_prep["targets_2d"],
                    fit_prep["domains"],
                    fit_base_dom,
                    fit_base_g["ece_value"],
                    cfg=cfg,
                )
            else:
                js_srch = {"tau": 1.0, "alpha": 0.0}
            c_list = list(js_fit["shrinkage_factors"])
            if zero_snr or float(js_srch["alpha"]) == 0.0:
                c_eff: Sequence[float] = [0.0] * max(1, len(c_list))
                zero_snr = True
            else:
                c_eff = c_list
            arm_specs[arm_label] = {
                "arm_kind": "james_stein",
                "rank_k": r_k,
                "shrinkage_factors": c_eff,
                "tau": float(js_srch["tau"]),
                "alpha": float(js_srch["alpha"]),
                "dw": dw_js,
                "ofc_max": float(js_fit["ofc_protected_max_delta"]),
                "d0_ortho_err": float(js_fit["d0_mean_grad_ortho_err"]),
                "zero_snr": zero_snr,
            }

        if cfg.include_full_dof_control:
            dw_full, full_d0_err, full_ofc = _fit_unreg_full_dof_nullspace_step(
                spec_grads,
                gen_grads,
                tok_fisher,
                protected_indices=ofc_clean,
                max_step_norm=cfg.max_step_norm,
                ridge_scale=cfg.ridge_scale,
                sv_tol=cfg.nullspace_sv_tol,
            )
            l_full = F.linear(u_fit * dw_full, head_w, None)
            full_srch = _search_probe_tau_alpha(
                l_base_fit,
                l_full,
                fit_prep["targets_2d"],
                fit_prep["domains"],
                fit_base_dom,
                fit_base_g["ece_value"],
                cfg=cfg,
            )
            arm_specs["unreg_full_dof"] = {
                "arm_kind": "full_dof",
                "rank_k": hidden_dim,
                "shrinkage_factors": None,
                "tau": float(full_srch["tau"]),
                "alpha": float(full_srch["alpha"]),
                "dw": dw_full,
                "ofc_max": full_ofc,
                "d0_ortho_err": full_d0_err,
                "zero_snr": False,
            }

        # Evaluate all candidate arms on held-out selection split
        sel_base_chunks, sel_base_dom, sel_base_g = _eval_units_with_params(
            u_sel, sel_prep["targets_2d"], sel_prep["domains"], w0, b0, head_w, head_b, None, tau=1.0, alpha=0.0, cfg=cfg
        )
        arm_selection_metrics: dict[str, dict[str, Any]] = {}
        for arm_label, spec in arm_specs.items():
            sel_c_chunks, sel_c_dom, sel_c_g = _eval_units_with_params(
                u_sel,
                sel_prep["targets_2d"],
                sel_prep["domains"],
                w0,
                b0,
                head_w,
                head_b,
                spec["dw"],
                tau=spec["tau"],
                alpha=spec["alpha"],
                cfg=cfg,
            )
            sel_anse = _compute_anse_verdict(
                sel_base_dom, sel_base_g["ece_value"], sel_c_dom, sel_c_g["ece_value"], cfg=cfg
            )
            arm_selection_metrics[arm_label] = {
                "arm_kind": spec["arm_kind"],
                "rank_k": spec["rank_k"],
                "full_dim": hidden_dim,
                "n_sel_tokens": n_sel_tokens,
                "shrinkage_factors": spec["shrinkage_factors"],
                "per_chunk_nll_diffs": (sel_c_chunks - sel_base_chunks),
                "nll_mean": float(sel_c_g["nll_mean"]),
                "base_nll_mean": float(sel_base_g["nll_mean"]),
                "ece_value": float(sel_c_g["ece_value"]),
                "base_ece_value": float(sel_base_g["ece_value"]),
                "general_drift_ratio": float(sel_anse["general_drift_ratio"]),
                "worst_domain_drift_ratio": float(sel_anse["worst_domain_drift_ratio"]),
                "guard_ok": bool(sel_anse["guard_ok"]),
            }

        sel_decision = select_variance_guarded_champion(
            arm_selection_metrics,
            ece_weight=cfg.ece_energy_weight,
            general_guard_rel=cfg.max_general_drift_rel,
            bootstrap_margin_z=cfg.bootstrap_margin_z,
        )

        chosen_name = sel_decision["selected_arm"]
        if chosen_name is not None:
            chosen_spec = arm_specs[chosen_name]
            chosen_tau = float(chosen_spec["tau"])
            chosen_alpha = float(chosen_spec["alpha"])
            chosen_dw = chosen_spec["dw"]
            chosen_ofc_max = float(chosen_spec["ofc_max"])
            chosen_d0_err = float(chosen_spec["d0_ortho_err"])
        else:
            chosen_tau = 1.0
            chosen_alpha = 0.0
            chosen_dw = None
            chosen_ofc_max = 0.0
            chosen_d0_err = 0.0

        _apply_norm_candidate_inplace(
            norm_mod, head_mod, w0, b0, head_b, chosen_dw, tau=chosen_tau, alpha=chosen_alpha
        )
        cand_sha = compute_module_state_sha256(work_model)

        _, ver_base_dom, ver_base_g = _eval_units_with_params(
            u_ver, ver_prep["targets_2d"], ver_prep["domains"], w0, b0, head_w, head_b, None, tau=1.0, alpha=0.0, cfg=cfg
        )
        _, ver_cand_dom, ver_cand_g = _eval_units_with_params(
            u_ver,
            ver_prep["targets_2d"],
            ver_prep["domains"],
            w0,
            b0,
            head_w,
            head_b,
            chosen_dw,
            tau=chosen_tau,
            alpha=chosen_alpha,
            cfg=cfg,
        )
        ver_anse = _compute_anse_verdict(
            ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
        )

        # Also evaluate unreg_full_dof on eval_chunks when present (for empirical confirm comparison)
        eval_control_summary: dict[str, dict[str, float]] = {}
        for arm_label, spec in arm_specs.items():
            _, e_dom, e_g = _eval_units_with_params(
                u_ver,
                ver_prep["targets_2d"],
                ver_prep["domains"],
                w0,
                b0,
                head_w,
                head_b,
                spec["dw"],
                tau=spec["tau"],
                alpha=spec["alpha"],
                cfg=cfg,
            )
            e_anse = _compute_anse_verdict(
                ver_base_dom, ver_base_g["ece_value"], e_dom, e_g["ece_value"], cfg=cfg
            )
            eval_control_summary[arm_label] = {
                "nll_mean": float(e_g["nll_mean"]),
                "nll_mean_diff": float(e_anse["nll_mean_diff"]),
                "ece_value": float(e_g["ece_value"]),
                "ece_rel_delta": float(e_anse["ece_rel_delta"]),
                "general_drift_ratio": float(e_anse["general_drift_ratio"]),
                "anse_energy": float(e_anse["anse_energy"]),
            }

        improved = (
            float(ver_anse["delta_e"]) < -cfg.min_energy_drop
            if cfg.require_strict_energy_drop
            else float(ver_anse["delta_e"]) <= 0.0
        )

        if cand_sha == initial_sha:
            accepted = False
            rolled_back = False
            status = (
                "FIXED_POINT_UNCHANGED"
                if (chosen_name is not None and ver_anse["guard_ok"])
                else f"REJECTED_{sel_decision['status']}"
            )
        elif ver_anse["guard_ok"] and improved:
            accepted = True
            rolled_back = False
            status = f"ACCEPTED_{str(chosen_name).upper()}"
        else:
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_GUARD_VIOLATION" if not ver_anse["guard_ok"] else "ROLLED_BACK_ENERGY_NON_IMPROVING"
            _restore_state_inplace(work_model, initial_snapshot)

        final_sha = compute_module_state_sha256(work_model)
        if not accepted and final_sha != initial_sha:
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {initial_sha[:12]}")
    finally:
        work_model.train(was_training)

    active_g = ver_cand_g if accepted else ver_base_g
    active_anse = (
        ver_anse
        if accepted
        else {
            "anse_energy": 0.0 if ver_anse["guard_ok"] else float(ver_anse["anse_energy"]),
            "general_drift_ratio": 0.0,
            "worst_domain_drift_ratio": 0.0,
            "ece_rel_delta": 0.0,
            "delta_e": 0.0,
        }
    )

    receipt: dict[str, Any] = {
        "object": SELECTOR_RECEIPT_OBJECT,
        "schema": SELECTOR_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "selected_arm": chosen_name if accepted else "base_rollback",
        "selected_arm_kind": sel_decision["selected_arm_kind"] if accepted else "base",
        "fallback_to_scalar": bool(sel_decision["fallback_to_scalar"]),
        "effective_dof": float(sel_decision["effective_dof"]) if accepted else 0.0,
        "bic_penalty": float(sel_decision["bic_penalty"]) if accepted else 0.0,
        "n_sel_tokens": n_sel_tokens,
        "norm_kind": resolved_norm,
        "norm_attr": norm_name,
        "head_attr": head_name,
        "tau_temp": chosen_tau if accepted else 1.0,
        "alpha_scale": chosen_alpha if accepted else 0.0,
        "ofc_protected_max_delta": chosen_ofc_max if accepted else 0.0,
        "d0_mean_grad_ortho_err": chosen_d0_err if accepted else 0.0,
        "nll_mean": float(active_g["nll_mean"]),
        "ece_value": float(active_g["ece_value"]),
        "base_nll_mean": float(ver_base_g["nll_mean"]),
        "base_ece_value": float(ver_base_g["ece_value"]),
        "ece_rel_delta": float(active_anse["ece_rel_delta"]),
        "general_drift_ratio": float(active_anse["general_drift_ratio"]),
        "worst_domain_drift_ratio": float(active_anse["worst_domain_drift_ratio"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "cand_anse_energy": float(ver_anse["anse_energy"]),
        "delta_e": float(active_anse["delta_e"]),
        "guard_ok": bool(ver_anse["guard_ok"]),
        "selection_decision": sel_decision,
        "eval_arm_comparison": eval_control_summary,
        "initial_state_sha256": initial_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BOOTSTRAP_MARGIN_Z",
    "DEFAULT_ECE_WEIGHT",
    "DEFAULT_FULL_DIM",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MAX_STEP_NORM",
    "DEFAULT_MIN_ENERGY_DROP",
    "DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE",
    "DEFAULT_N_SEL_TOKENS",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_NUM_BINS",
    "DEFAULT_RANK_K",
    "DEFAULT_RIDGE_SCALE",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SECONDARY_RANK_K",
    "DEFAULT_TAU_GRID",
    "SELECTOR_RECEIPT_OBJECT",
    "SELECTOR_SCHEMA",
    "VALID_ARM_KINDS",
    "VALID_NORM_KINDS",
    "EffectiveDOFBICSelectorConfig",
    "compute_effective_dof_and_bic_penalty",
    "execute_verified_bic_promoted_wsp",
    "select_variance_guarded_champion",
    "verify_bic_arm_selector_receipt_provenance",
]
