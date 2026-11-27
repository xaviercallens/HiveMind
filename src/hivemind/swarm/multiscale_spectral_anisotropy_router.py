# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Multi-Scale Spectral-Anisotropy Profiler and Causal-vs-MLM WSP Operator Regime Router.

Crystallizes the empirical scaling law observed across ``PILOT3A/3B``, ``PILOT9``
(``Qwen/Qwen2.5-3B``: ``top1_sv_energy_frac=0.6874``, 60x self-suppressed outlier
dims ``1819/1874``), ``PILOT12`` (``Qwen/Qwen2.5-0.5B``: ``top1=0.3481``, 22x outlier
dim ``490``, James-Stein rank-4 beating 1-DOF scalar 2.09x on confirm), ``PILOT14``
(``answerdotai/ModernBERT-base`` bidirectional MLM: small masked-token probe overfitting
``D0 +1.025% FAIL``), and ``PILOT15`` (conditional ``P_0`` multi-adapter rescue) into a
deterministic, training-free CPU router:

1. :func:`profile_hidden_spectral_anisotropy`:
   Quantifies pre-norm SVD concentration (``top1_energy_frac``, ``top8_energy_frac``,
   effective spectral rank), massive-activation channel ratio & norm self-suppression
   factor, and effective probe tokens-per-DOF ratio across causal and MLM architectures.
2. :func:`select_wsp_operator_regime`:
   Deterministically routes each model/probe into the mathematically safe operator regime
   (``james_stein_rank_k_fisher``, ``folded_scalar_rmsnorm_only``,
   ``p0_lora_plus_folded_norm``, or ``abstain_insufficient_probe_tokens``).
3. :func:`execute_regime_routed_wsp_plan`:
   Executes the routed operator end-to-end under ``ANSE Delta E < 0`` and ``D0 <= +1.0%``
   with atomic bitwise rollback under ``ANSE_MAXIMUM_PAIN`` and SHA-256 provenance.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.folded_norm_ece_calibrator import (
    ANSE_MAXIMUM_PAIN,
    compute_token_nll_and_ece,
)
from hivemind.swarm.james_stein_spectral_norm_surgeon import (
    fit_james_stein_rank_k_fisher_norm_step,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    compute_module_state_sha256,
    compute_payload_sha256,
    detect_norm_type,
    fold_temperature_into_norm_params,
)

ROUTER_SCHEMA: str = "hivemind.multiscale_spectral_anisotropy_router/1"
ROUTER_RECEIPT_OBJECT: str = "hivemind.multiscale_spectral_anisotropy_router_receipt"

REGIME_JAMES_STEIN_RANK_K_FISHER: str = "james_stein_rank_k_fisher"
REGIME_FOLDED_SCALAR_RMSNORM_ONLY: str = "folded_scalar_rmsnorm_only"
REGIME_P0_LORA_PLUS_FOLDED_NORM: str = "p0_lora_plus_folded_norm"
REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS: str = "abstain_insufficient_probe_tokens"

VALID_WSP_REGIMES: frozenset[str] = frozenset(
    {
        REGIME_JAMES_STEIN_RANK_K_FISHER,
        REGIME_FOLDED_SCALAR_RMSNORM_ONLY,
        REGIME_P0_LORA_PLUS_FOLDED_NORM,
        REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS,
    }
)
VALID_ARCHITECTURE_KINDS: frozenset[str] = frozenset(
    {
        "causal_lm",
        "moe_causal_lm",
        "bidirectional_mlm",
    }
)
VALID_NORM_KINDS: frozenset[str] = frozenset({"auto", "layernorm", "rmsnorm"})

DEFAULT_MIN_CAUSAL_TOKENS_PER_DOF: float = 4.0
DEFAULT_MIN_MLM_MASKED_TOKENS_PER_DOF: float = 12.0
DEFAULT_MIN_CAUSAL_SCALAR_TOKENS_PER_DOF: float = 0.25
DEFAULT_MIN_MLM_SCALAR_TOKENS_PER_DOF: float = 0.50
DEFAULT_HIGH_ANISOTROPY_THRESHOLD: float = 0.55
DEFAULT_EXTREME_ANISOTROPY_THRESHOLD: float = 0.85
DEFAULT_OUTLIER_RATIO_THRESHOLD: float = 5.0
DEFAULT_MASSIVE_OUTLIER_RATIO_THRESHOLD: float = 40.0
DEFAULT_SELF_SUPPRESSION_RATIO_THRESHOLD: float = 4.0
DEFAULT_MAX_RANK: int = 4
DEFAULT_RIDGE_SCALE: float = 1e-3
DEFAULT_MAX_STEP_NORM: float = 0.15
DEFAULT_NULLSPACE_SV_TOL: float = 1e-6
DEFAULT_P0_SV_REL_TOL: float = 5e-2
DEFAULT_MAX_P0_RANK: int = 32
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.01
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.01
DEFAULT_ANSE_SAFE_GENERAL_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_MIN_ENERGY_DROP: float = 1e-6
DEFAULT_MIN_PROBE_SNR: float = 1.0
DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE: float = 1e-3
DEFAULT_NUM_BINS: int = 15
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_TAU_GRID: tuple[float, ...] = (0.90, 0.94, 0.97, 1.00, 1.03, 1.05, 1.08, 1.12, 1.18, 1.25)
DEFAULT_ETA_GRID: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")

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


def _validate_architecture_kind(architecture_kind: Any) -> str:
    """Validate architecture kind against supported causal and MLM families."""
    kind = _validate_nonempty_str(architecture_kind, "architecture_kind")
    if kind not in VALID_ARCHITECTURE_KINDS:
        raise ValueError(
            f"architecture_kind must be one of {sorted(VALID_ARCHITECTURE_KINDS)}, got {architecture_kind!r}"
        )
    return kind


def _validate_fp_tensor(
    tensor: Any,
    name: str,
    *,
    exact_ndim: int | None = None,
    min_ndim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty finite floating-point tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if exact_ndim is not None and tensor.ndim != exact_ndim:
        raise ValueError(f"{name} must have ndim == {exact_ndim}, got shape {tuple(tensor.shape)}")
    if min_ndim is not None and tensor.ndim < min_ndim:
        raise ValueError(f"{name} must have ndim >= {min_ndim}, got shape {tuple(tensor.shape)}")
    if any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be non-empty, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_positive_float_seq(seq: Any, name: str) -> tuple[float, ...]:
    """Validate a non-empty sequence of finite strictly positive floats."""
    if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of positive floats")
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty")
    return tuple(sorted({_validate_positive_finite(val, f"{name}[{idx}]") for idx, val in enumerate(seq)}))


def _validate_general_prefixes(prefixes: Any) -> tuple[str, ...]:
    """Validate non-empty sequence of general-prose domain prefixes."""
    if isinstance(prefixes, (str, bytes, bytearray)) or not isinstance(prefixes, Sequence):
        raise TypeError("general_prefixes must be a non-empty sequence of strings")
    if not prefixes:
        raise ValueError("general_prefixes must be non-empty")
    return tuple(_validate_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(prefixes))


@dataclass(frozen=True)
class MultiscaleSpectralAnisotropyRouterConfig:
    """Validated configuration for multi-scale spectral profiling, regime routing, and execution."""

    min_causal_tokens_per_dof: float = DEFAULT_MIN_CAUSAL_TOKENS_PER_DOF
    min_mlm_masked_tokens_per_dof: float = DEFAULT_MIN_MLM_MASKED_TOKENS_PER_DOF
    min_causal_scalar_tokens_per_dof: float = DEFAULT_MIN_CAUSAL_SCALAR_TOKENS_PER_DOF
    min_mlm_scalar_tokens_per_dof: float = DEFAULT_MIN_MLM_SCALAR_TOKENS_PER_DOF
    high_anisotropy_threshold: float = DEFAULT_HIGH_ANISOTROPY_THRESHOLD
    extreme_anisotropy_threshold: float = DEFAULT_EXTREME_ANISOTROPY_THRESHOLD
    outlier_ratio_threshold: float = DEFAULT_OUTLIER_RATIO_THRESHOLD
    massive_outlier_ratio_threshold: float = DEFAULT_MASSIVE_OUTLIER_RATIO_THRESHOLD
    self_suppression_threshold: float = DEFAULT_SELF_SUPPRESSION_RATIO_THRESHOLD
    allow_mlm_scalar_fallback: bool = False
    max_rank: int = DEFAULT_MAX_RANK
    ridge_scale: float = DEFAULT_RIDGE_SCALE
    max_step_norm: float = DEFAULT_MAX_STEP_NORM
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL
    p0_sv_rel_tol: float = DEFAULT_P0_SV_REL_TOL
    max_p0_rank: int = DEFAULT_MAX_P0_RANK
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_GENERAL_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    min_energy_drop: float = DEFAULT_MIN_ENERGY_DROP
    min_probe_snr: float = DEFAULT_MIN_PROBE_SNR
    min_shrinkage_to_activate: float = DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE
    num_bins: int = DEFAULT_NUM_BINS
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    eta_grid: tuple[float, ...] = DEFAULT_ETA_GRID
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        _validate_positive_finite(self.min_causal_tokens_per_dof, "min_causal_tokens_per_dof")
        _validate_positive_finite(self.min_mlm_masked_tokens_per_dof, "min_mlm_masked_tokens_per_dof")
        _validate_positive_finite(self.min_causal_scalar_tokens_per_dof, "min_causal_scalar_tokens_per_dof")
        _validate_positive_finite(self.min_mlm_scalar_tokens_per_dof, "min_mlm_scalar_tokens_per_dof")
        high_a = _validate_positive_finite(self.high_anisotropy_threshold, "high_anisotropy_threshold")
        ext_a = _validate_positive_finite(self.extreme_anisotropy_threshold, "extreme_anisotropy_threshold")
        if high_a > 1.0 or ext_a > 1.0 or ext_a < high_a:
            raise ValueError(
                "Anisotropy thresholds must satisfy 0 < high_anisotropy_threshold <= extreme_anisotropy_threshold <= 1.0"
            )
        out_th = _validate_positive_finite(self.outlier_ratio_threshold, "outlier_ratio_threshold")
        mass_th = _validate_positive_finite(self.massive_outlier_ratio_threshold, "massive_outlier_ratio_threshold")
        if out_th <= 1.0 or mass_th <= out_th:
            raise ValueError("Must satisfy 1.0 < outlier_ratio_threshold < massive_outlier_ratio_threshold")
        sup_th = _validate_positive_finite(self.self_suppression_threshold, "self_suppression_threshold")
        if sup_th <= 1.0:
            raise ValueError(f"self_suppression_threshold must be > 1.0, got {sup_th}")
        _validate_bool(self.allow_mlm_scalar_fallback, "allow_mlm_scalar_fallback")
        _validate_positive_int(self.max_rank, "max_rank")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_positive_finite(self.max_step_norm, "max_step_norm")
        for tol_val, tol_name in (
            (self.nullspace_sv_tol, "nullspace_sv_tol"),
            (self.p0_sv_rel_tol, "p0_sv_rel_tol"),
        ):
            v_tol = _validate_positive_finite(tol_val, tol_name)
            if v_tol >= 1.0:
                raise ValueError(f"{tol_name} must be < 1.0, got {v_tol}")
        _validate_positive_int(self.max_p0_rank, "max_p0_rank")
        _validate_nonneg_finite(self.max_general_drift_rel, "max_general_drift_rel")
        _validate_nonneg_finite(self.max_domain_drift_rel, "max_domain_drift_rel")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.min_energy_drop, "min_energy_drop")
        _validate_nonneg_finite(self.min_probe_snr, "min_probe_snr")
        _validate_nonneg_finite(self.min_shrinkage_to_activate, "min_shrinkage_to_activate")
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_int(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        object.__setattr__(self, "tau_grid", _validate_positive_float_seq(self.tau_grid, "tau_grid"))
        object.__setattr__(self, "eta_grid", _validate_positive_float_seq(self.eta_grid, "eta_grid"))
        object.__setattr__(self, "general_prefixes", _validate_general_prefixes(self.general_prefixes))
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


def verify_regime_routed_wsp_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != ROUTER_RECEIPT_OBJECT or receipt.get("schema") != ROUTER_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Spectral Anisotropy Profiler & Regime Router
# ---------------------------------------------------------------------------


def _compute_spectral_stats(h_2d: torch.Tensor) -> tuple[float, float, float]:
    """Compute ``(top1_energy_frac, top8_energy_frac, effective_spectral_rank)`` in float64."""
    s_vals = torch.linalg.svdvals(h_2d)
    s2 = s_vals.pow(2)
    tot = float(s2.sum().item())
    if tot <= 1e-30:
        return 0.0, 0.0, 0.0
    top1 = float((s2[0] / tot).item())
    top8 = float((s2[: min(8, int(s2.numel()))].sum() / tot).item())
    eff_rank = (tot * tot) / max(float(s2.pow(2).sum().item()), 1e-30)
    return top1, top8, float(eff_rank)


@torch.no_grad()
def profile_hidden_spectral_anisotropy(
    hidden_states: torch.Tensor,
    norm_weights: torch.Tensor,
    *,
    architecture_kind: str = "causal_lm",
    effective_tokens_per_split: int = 0,
    outlier_ratio_threshold: float = DEFAULT_OUTLIER_RATIO_THRESHOLD,
    self_suppression_threshold: float = DEFAULT_SELF_SUPPRESSION_RATIO_THRESHOLD,
    norm_bias: torch.Tensor | None = None,
    head_bias_dim: int = 0,
    has_lora_adapters: bool = False,
) -> dict[str, Any]:
    """Quantify hidden-state SVD concentration, self-suppressed outliers, and probe tokens per DOF.

    Captures the physical invariants driving multi-scale WSP behavior across ``PILOT9`` (3B),
    ``PILOT12`` (0.5B), and ``PILOT14`` (ModernBERT-base MLM):
    - ``top1_energy_frac`` / ``top8_energy_frac`` & effective spectral participation rank;
    - ``massive_activation_channel_ratio`` (`max_rms / median_rms`) & norm weight
      ``self_suppression_factor`` (`median(|w_norm|) / max(|w_norm[outlier]|, 1e-6)`);
    - ``probe_tokens_per_dof`` (`effective_tokens_per_split / hidden_dim`) and total-head DOF ratio.
    """
    arch = _validate_architecture_kind(architecture_kind)
    eff_tok_arg = _validate_nonneg_int(effective_tokens_per_split, "effective_tokens_per_split")
    out_thresh = _validate_positive_finite(outlier_ratio_threshold, "outlier_ratio_threshold")
    if out_thresh <= 1.0:
        raise ValueError(f"outlier_ratio_threshold must be > 1.0, got {out_thresh}")
    sup_thresh = _validate_positive_finite(self_suppression_threshold, "self_suppression_threshold")
    if sup_thresh <= 1.0:
        raise ValueError(f"self_suppression_threshold must be > 1.0, got {sup_thresh}")
    hb_dim = _validate_nonneg_int(head_bias_dim, "head_bias_dim")
    lora_flag = _validate_bool(has_lora_adapters, "has_lora_adapters")

    h_t = _validate_fp_tensor(hidden_states, "hidden_states", min_ndim=2)
    w_t = _validate_fp_tensor(norm_weights, "norm_weights", exact_ndim=1)
    hidden_dim = int(w_t.shape[0])
    if int(h_t.shape[-1]) != hidden_dim:
        raise ValueError(f"hidden_states last dim {h_t.shape[-1]} must match norm_weights len {hidden_dim}")

    has_norm_bias = norm_bias is not None
    if norm_bias is not None:
        b_t = _validate_fp_tensor(norm_bias, "norm_bias", exact_ndim=1)
        if int(b_t.shape[0]) != hidden_dim:
            raise ValueError(f"norm_bias length {b_t.shape[0]} must match hidden_dim {hidden_dim}")

    h_2d = h_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1, hidden_dim)
    w_abs = w_t.detach().to(device="cpu", dtype=torch.float64).abs()
    observed_tokens = int(h_2d.shape[0])
    eff_tokens = observed_tokens if eff_tok_arg == 0 else eff_tok_arg

    top1_frac, top8_frac, eff_rank = _compute_spectral_stats(h_2d)

    ch_rms = h_2d.pow(2).mean(dim=0).sqrt()
    median_rms = float(ch_rms.median().item())
    max_rms = float(ch_rms.max().item())
    denom_med = max(median_rms, 1e-12)
    ratios = ch_rms / denom_med
    massive_ratio = float(ratios.max().item())

    median_w = float(w_abs.median().item())
    sup_ratios = max(median_w, 1e-12) / w_abs.clamp_min(1e-6)

    outlier_mask = ratios >= out_thresh
    num_outliers = int(outlier_mask.sum().item())
    outlier_frac = float(num_outliers) / float(hidden_dim)
    self_suppr_factor = float(sup_ratios[outlier_mask].max().item()) if num_outliers > 0 else 1.0
    self_suppressed_count = int((outlier_mask & (sup_ratios >= sup_thresh)).sum().item())

    post_norm_rms = ch_rms * w_abs
    post_norm_atten_ratio = float(post_norm_rms.max().item()) / max(float(post_norm_rms.median().item()), 1e-12)

    top_k = min(8, hidden_dim)
    top_vals, top_idxs = torch.topk(ratios, k=top_k)
    top_channels = [
        {
            "dim": int(idx.item()),
            "rms": float(ch_rms[int(idx.item())].item()),
            "ratio_vs_median": float(val.item()),
            "abs_norm_weight": float(w_abs[int(idx.item())].item()),
            "self_suppression_factor": float(sup_ratios[int(idx.item())].item()),
        }
        for val, idx in zip(top_vals, top_idxs, strict=True)
    ]
    outlier_dims = torch.nonzero(outlier_mask, as_tuple=False).squeeze(1)
    order = torch.argsort(ratios[outlier_dims], descending=True) if num_outliers > 0 else outlier_dims
    outlier_indices = [int(outlier_dims[pos].item()) for pos in order]

    if top1_frac >= DEFAULT_EXTREME_ANISOTROPY_THRESHOLD:
        anisotropy_regime = "extreme_anisotropy"
    elif top1_frac >= DEFAULT_HIGH_ANISOTROPY_THRESHOLD:
        anisotropy_regime = "high_anisotropy"
    elif top1_frac >= 0.25:
        anisotropy_regime = "moderate_anisotropy"
    else:
        anisotropy_regime = "low_anisotropy"

    norm_dof = hidden_dim * (2 if has_norm_bias else 1)
    total_head_dof = norm_dof + hb_dim
    tokens_per_dof = float(eff_tokens) / float(hidden_dim)

    return {
        "architecture_kind": arch,
        "hidden_dim": hidden_dim,
        "observed_tokens": observed_tokens,
        "effective_tokens_per_split": eff_tokens,
        "norm_dof": norm_dof,
        "total_head_dof": total_head_dof,
        "probe_tokens_per_dof": tokens_per_dof,
        "probe_tokens_per_norm_dof": float(eff_tokens) / float(norm_dof),
        "probe_tokens_per_total_dof": float(eff_tokens) / float(total_head_dof),
        "top1_energy_frac": top1_frac,
        "top1_sv_energy_frac": top1_frac,
        "top8_energy_frac": top8_frac,
        "top8_sv_energy_frac": top8_frac,
        "effective_spectral_rank": eff_rank,
        "anisotropy_regime": anisotropy_regime,
        "median_channel_rms": median_rms,
        "max_channel_rms": max_rms,
        "massive_activation_channel_ratio": massive_ratio,
        "median_abs_norm_weight": median_w,
        "self_suppression_factor": self_suppr_factor,
        "self_suppressed_outlier_count": self_suppressed_count,
        "num_outlier_channels": num_outliers,
        "outlier_channel_fraction": outlier_frac,
        "outlier_channel_indices": outlier_indices,
        "post_norm_attenuation_ratio": post_norm_atten_ratio,
        "top_channels": top_channels,
        "has_lora_adapters": lora_flag,
    }


def select_wsp_operator_regime(
    profile: Mapping[str, Any],
    *,
    min_causal_tokens_per_dof: float = DEFAULT_MIN_CAUSAL_TOKENS_PER_DOF,
    min_mlm_masked_tokens_per_dof: float = DEFAULT_MIN_MLM_MASKED_TOKENS_PER_DOF,
    min_causal_scalar_tokens_per_dof: float = DEFAULT_MIN_CAUSAL_SCALAR_TOKENS_PER_DOF,
    min_mlm_scalar_tokens_per_dof: float = DEFAULT_MIN_MLM_SCALAR_TOKENS_PER_DOF,
    high_anisotropy_threshold: float = DEFAULT_HIGH_ANISOTROPY_THRESHOLD,
    extreme_anisotropy_threshold: float = DEFAULT_EXTREME_ANISOTROPY_THRESHOLD,
    massive_outlier_ratio_threshold: float = DEFAULT_MASSIVE_OUTLIER_RATIO_THRESHOLD,
    self_suppression_threshold: float = DEFAULT_SELF_SUPPRESSION_RATIO_THRESHOLD,
    max_rank: int = DEFAULT_MAX_RANK,
    allow_mlm_scalar_fallback: bool = False,
) -> dict[str, Any]:
    """Deterministically route a model/probe into the mathematically safe WSP operator regime."""
    if not isinstance(profile, Mapping):
        raise TypeError(f"profile must be a Mapping, got {type(profile).__name__}")

    min_c = _validate_positive_finite(min_causal_tokens_per_dof, "min_causal_tokens_per_dof")
    min_m = _validate_positive_finite(min_mlm_masked_tokens_per_dof, "min_mlm_masked_tokens_per_dof")
    min_c_sc = _validate_positive_finite(min_causal_scalar_tokens_per_dof, "min_causal_scalar_tokens_per_dof")
    min_m_sc = _validate_positive_finite(min_mlm_scalar_tokens_per_dof, "min_mlm_scalar_tokens_per_dof")
    high_a = _validate_positive_finite(high_anisotropy_threshold, "high_anisotropy_threshold")
    ext_a = _validate_positive_finite(extreme_anisotropy_threshold, "extreme_anisotropy_threshold")
    if high_a > 1.0 or ext_a > 1.0 or ext_a < high_a:
        raise ValueError("Anisotropy thresholds must satisfy 0 < high <= extreme <= 1.0")
    mass_th = _validate_positive_finite(massive_outlier_ratio_threshold, "massive_outlier_ratio_threshold")
    sup_th = _validate_positive_finite(self_suppression_threshold, "self_suppression_threshold")
    k_cap = _validate_positive_int(max_rank, "max_rank")
    mlm_fallback = _validate_bool(allow_mlm_scalar_fallback, "allow_mlm_scalar_fallback")

    arch = _validate_architecture_kind(profile.get("architecture_kind"))
    hidden_dim = _validate_positive_int(profile.get("hidden_dim"), "profile['hidden_dim']")
    tokens_per_dof = _validate_nonneg_finite(profile.get("probe_tokens_per_dof"), "profile['probe_tokens_per_dof']")
    top1 = _validate_nonneg_finite(
        profile.get("top1_energy_frac", profile.get("top1_sv_energy_frac")),
        "profile['top1_energy_frac']",
    )
    top8 = _validate_nonneg_finite(
        profile.get("top8_energy_frac", profile.get("top8_sv_energy_frac", top1)),
        "profile['top8_energy_frac']",
    )
    eff_rank = _validate_nonneg_finite(
        profile.get("effective_spectral_rank", 1.0), "profile['effective_spectral_rank']"
    )
    outlier_ratio = _validate_nonneg_finite(
        profile.get("massive_activation_channel_ratio"), "profile['massive_activation_channel_ratio']"
    )
    self_suppr = _validate_nonneg_finite(
        profile.get("self_suppression_factor"), "profile['self_suppression_factor']"
    )
    has_lora = _validate_bool(bool(profile.get("has_lora_adapters", False)), "profile['has_lora_adapters']")

    is_high_aniso = top1 >= high_a
    is_extreme_aniso = top1 >= ext_a
    is_extreme_outlier = (outlier_ratio >= mass_th) and (self_suppr >= sup_th)

    reasons: list[str] = []
    rec_rank = 0
    abstained = False

    if arch == "bidirectional_mlm":
        if tokens_per_dof < min_m:
            reasons.append("mlm_masked_tokens_per_dof_below_threshold")
            if is_extreme_aniso:
                reasons.append("extreme_spectral_anisotropy")
            if mlm_fallback and tokens_per_dof >= min_m_sc:
                regime = REGIME_FOLDED_SCALAR_RMSNORM_ONLY
                reasons.append("mlm_1dof_scalar_guard_fallback")
            else:
                regime = REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS
                abstained = True
        elif has_lora and not is_extreme_outlier:
            regime = REGIME_P0_LORA_PLUS_FOLDED_NORM
            reasons.append("lora_adapters_with_adequate_mlm_tokens")
        elif is_extreme_aniso or is_extreme_outlier:
            regime = REGIME_FOLDED_SCALAR_RMSNORM_ONLY
            if is_extreme_aniso:
                reasons.append("extreme_spectral_anisotropy")
            if is_extreme_outlier:
                reasons.append("extreme_self_suppressed_massive_activations")
        else:
            regime = REGIME_JAMES_STEIN_RANK_K_FISHER
            rank_hint = max(1, int(round(min(max(eff_rank, 1.0), tokens_per_dof / 2.0))))
            rec_rank = min(k_cap, hidden_dim, rank_hint)
            reasons.append("adequate_mlm_masked_tokens_and_bounded_anisotropy")
    else:
        if tokens_per_dof < min_c_sc:
            regime = REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS
            abstained = True
            reasons.append("causal_tokens_per_dof_below_scalar_minimum")
        elif has_lora:
            regime = REGIME_P0_LORA_PLUS_FOLDED_NORM
            reasons.append("lora_adapters_present")
        elif is_extreme_outlier or is_high_aniso or tokens_per_dof < min_c:
            regime = REGIME_FOLDED_SCALAR_RMSNORM_ONLY
            if is_extreme_outlier:
                reasons.append("extreme_self_suppressed_massive_activations")
            if is_high_aniso:
                reasons.append("high_spectral_anisotropy")
            if tokens_per_dof < min_c:
                reasons.append("causal_tokens_per_dof_below_threshold")
        else:
            regime = REGIME_JAMES_STEIN_RANK_K_FISHER
            rank_hint = max(1, int(round(min(max(eff_rank, 1.0), tokens_per_dof))))
            rec_rank = min(k_cap, hidden_dim, rank_hint)
            reasons.append("moderate_anisotropy_and_sufficient_causal_tokens_per_dof")

    return {
        "selected_regime": regime,
        "architecture_kind": arch,
        "anisotropy_regime": str(profile.get("anisotropy_regime", "unknown")),
        "abstained": abstained,
        "safe_for_multi_dof_fisher": regime == REGIME_JAMES_STEIN_RANK_K_FISHER,
        "recommended_max_rank": rec_rank,
        "recommended_dof": (
            0 if abstained else (1 if regime == REGIME_FOLDED_SCALAR_RMSNORM_ONLY else rec_rank)
        ),
        "reasons": reasons,
        "thresholds_applied": {
            "min_causal_tokens_per_dof": min_c,
            "min_mlm_masked_tokens_per_dof": min_m,
            "min_causal_scalar_tokens_per_dof": min_c_sc,
            "min_mlm_scalar_tokens_per_dof": min_m_sc,
            "high_anisotropy_threshold": high_a,
            "extreme_anisotropy_threshold": ext_a,
            "massive_outlier_ratio_threshold": mass_th,
            "self_suppression_threshold": sup_th,
            "allow_mlm_scalar_fallback": mlm_fallback,
        },
        "profile_summary": {
            "top1_energy_frac": top1,
            "top8_energy_frac": top8,
            "massive_activation_channel_ratio": outlier_ratio,
            "self_suppression_factor": self_suppr,
            "probe_tokens_per_dof": tokens_per_dof,
        },
    }


# ---------------------------------------------------------------------------
# 3. Model Resolution, Chunk Preparation, State & LoRA P_0 Primitives
# ---------------------------------------------------------------------------


def _resolve_dotted_attr(root: Any, attr_path: str) -> Any:
    """Resolve a possibly dotted attribute path on ``root``, returning None if absent."""
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
        norm_name, norm_mod = "", None
        for cand in _NORM_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                norm_name, norm_mod = cand, found
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
        head_name, head_mod = "", None
        for cand in _HEAD_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                head_name, head_mod = cand, found
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
    """Return True if ``domain_label`` matches any configured general-prose prefix."""
    return any(domain_label.startswith(pref) for pref in general_prefixes)


def _to_long_1d(raw: Any, name: str) -> torch.Tensor:
    """Validate and convert a sequence or 1D integer tensor to CPU int64."""
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


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    vocab_size: int,
    score_from: int,
    ignore_index: int,
    general_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Validate causal or MLM masked chunks and construct alignment metadata."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")

    rows: list[torch.Tensor] = []
    domains: list[str] = []
    mask_pos_rows: list[torch.Tensor] = []
    mlm_target_rows: list[torch.Tensor] = []
    seq_len: int | None = None
    mask_len: int | None = None
    is_mlm: bool | None = None
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
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}, got {ids_t.numel()}")

        has_mlm = ("mask_positions" in ch) or ("targets" in ch)
        if is_mlm is None:
            is_mlm = has_mlm
        elif has_mlm != is_mlm:
            raise ValueError(f"{name} chunks must consistently use either causal or MLM mode")

        if is_mlm:
            if "mask_positions" not in ch or "targets" not in ch:
                raise ValueError(f"{name}[{idx}] MLM chunk requires both 'mask_positions' and 'targets'")
            mp_t = _to_long_1d(ch["mask_positions"], f"{name}[{idx}]['mask_positions']")
            tg_t = _to_long_1d(ch["targets"], f"{name}[{idx}]['targets']")
            if mp_t.numel() == 0 or mp_t.numel() != tg_t.numel():
                raise ValueError(f"{name}[{idx}] mask_positions and targets must share non-empty length")
            if bool((mp_t < 0).any().item()) or bool((mp_t >= seq_len).any().item()):
                raise ValueError(f"{name}[{idx}]['mask_positions'] indices must lie in [0, {seq_len})")
            valid_tg = tg_t[tg_t != ignore_index]
            if valid_tg.numel() == 0:
                raise ValueError(f"{name}[{idx}]['targets'] must contain non-ignored targets")
            if bool((valid_tg < 0).any().item()) or bool((valid_tg >= vocab_size).any().item()):
                raise ValueError(f"{name}[{idx}]['targets'] must lie in [0, {vocab_size})")
            if mask_len is None:
                mask_len = int(mp_t.numel())
            elif int(mp_t.numel()) != mask_len:
                raise ValueError(f"All {name} MLM chunks must share mask count {mask_len}")
            mask_pos_rows.append(mp_t)
            mlm_target_rows.append(tg_t)
        elif start_pos >= seq_len - 1:
            raise ValueError(f"{name} sequence length {seq_len} is too short for score_from={score_from}")

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
        "is_mlm": bool(is_mlm),
        "mask_positions": torch.stack(mask_pos_rows, dim=0) if is_mlm else None,
        "targets_2d": torch.stack(mlm_target_rows, dim=0) if is_mlm else ids_2d[:, start_pos + 1 :],
    }


def _split_probe_chunks_fallback(
    chunks: Sequence[Mapping[str, Any]],
    general_prefixes: tuple[str, ...],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministically partition probe chunks into fit and held-out selection folds."""
    fit_chunks = [ch for i, ch in enumerate(chunks) if i % 2 == 0]
    sel_chunks = [ch for i, ch in enumerate(chunks) if i % 2 == 1]
    for label, sub in (("fit", fit_chunks), ("selection", sel_chunks)):
        if len(sub) < 2:
            raise ValueError(f"Need at least 4 probe chunks for automatic split (got {len(chunks)})")
        doms = [str(ch.get("expert_id", ch.get("domain", ""))) for ch in sub]
        has_g = any(_is_general_domain(d, general_prefixes) for d in doms)
        has_s = any(not _is_general_domain(d, general_prefixes) for d in doms)
        if not (has_g and has_s):
            raise ValueError(f"Automatic {label} fold lacks both general and specialist chunks")
    return fit_chunks, sel_chunks


def _capture_pre_norm_states(
    model: nn.Module,
    norm_mod: nn.Module,
    ids_2d: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Capture pre-norm hidden states via read-only forward pre-hook on ``norm_mod``."""
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
                    f"Captured pre-norm shape {tuple(h_batch.shape)} incompatible with input {tuple(batch_ids.shape)}"
                )
            if not bool(torch.isfinite(h_batch).all().item()):
                raise ValueError("Model trunk produced non-finite pre-norm hidden states")
            parts.append(h_batch)
    finally:
        handle.remove()

    return torch.cat(parts, dim=0)


def _extract_scored_unit_states(
    h_pre_norm: torch.Tensor,
    prepared: Mapping[str, Any],
    *,
    norm_kind: str,
    eps: float,
    score_from: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice scored token positions and compute unit-normalized states in ``float64``."""
    h64 = h_pre_norm.detach().to(device="cpu", dtype=torch.float64)
    if prepared["is_mlm"]:
        pos = prepared["mask_positions"]
        batch_idx = torch.arange(int(h64.shape[0]), dtype=torch.long).unsqueeze(1)
        h_scored = h64[batch_idx, pos, :]
    else:
        start_pos = max(0, score_from - 1)
        h_scored = h64[:, start_pos:-1, :]

    dim = int(h_scored.shape[-1])
    if norm_kind == "rmsnorm":
        u_scored = h_scored * torch.rsqrt(h_scored.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    else:
        u_scored = F.layer_norm(h_scored, (dim,), None, None, eps=float(eps))
    return h_scored.contiguous(), u_scored.contiguous()


def _compute_chunk_norm_gradients(
    u_scored: torch.Tensor,
    targets_2d: torch.Tensor,
    w_norm: torch.Tensor,
    b_norm: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-chunk gradients wrt ``w_norm`` and diagonal Fisher in ``float64``."""
    c_num, t_len, dim = u_scored.shape
    w_64 = w_norm.detach().to(device="cpu", dtype=torch.float64)
    b_64 = b_norm.detach().to(device="cpu", dtype=torch.float64) if b_norm is not None else None
    hw_64 = head_w.detach().to(device="cpu", dtype=torch.float64)
    hb_64 = head_b.detach().to(device="cpu", dtype=torch.float64) if head_b is not None else None

    x_norm = u_scored * w_64
    if b_64 is not None:
        x_norm = x_norm + b_64
    logits = F.linear(x_norm.reshape(-1, dim), hw_64, hb_64)
    probs = F.softmax(logits, dim=-1)

    t_flat = targets_2d.reshape(-1)
    valid = t_flat != ignore_index
    v_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    probs[v_idx, t_flat[valid]] -= 1.0
    probs[~valid] = 0.0

    tok_grads = (probs @ hw_64).reshape(c_num, t_len, dim) * u_scored
    valid_2d = (targets_2d != ignore_index).to(torch.float64)
    counts = valid_2d.sum(dim=1).clamp_min(1.0).unsqueeze(1)
    chunk_grads = (tok_grads * valid_2d.unsqueeze(2)).sum(dim=1) / counts
    total_valid = float(valid_2d.sum().item())
    tok_fisher = (tok_grads.pow(2) * valid_2d.unsqueeze(2)).sum(dim=(0, 1)) / max(total_valid, 1.0)
    return chunk_grads.contiguous(), tok_fisher.contiguous()


def _eval_norm_metrics(
    u_scored: torch.Tensor,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    w_norm: torch.Tensor,
    b_norm: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    *,
    num_bins: int,
    ignore_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate exact per-domain NLL and global NLL/ECE from unit-normalized states."""
    w_64 = w_norm.detach().to(device="cpu", dtype=torch.float64)
    b_64 = b_norm.detach().to(device="cpu", dtype=torch.float64) if b_norm is not None else None
    hw_64 = head_w.detach().to(device="cpu", dtype=torch.float64)
    hb_64 = head_b.detach().to(device="cpu", dtype=torch.float64) if head_b is not None else None

    x_norm = u_scored * w_64
    if b_64 is not None:
        x_norm = x_norm + b_64
    logits = F.linear(x_norm, hw_64, hb_64)
    global_m = compute_token_nll_and_ece(logits, targets_2d, num_bins=num_bins, ignore_index=ignore_index)

    dom_indices: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        dom_indices.setdefault(dom, []).append(idx)

    dom_nll: dict[str, float] = {}
    for dom in sorted(dom_indices.keys()):
        ix = dom_indices[dom]
        dm = compute_token_nll_and_ece(logits[ix], targets_2d[ix], num_bins=num_bins, ignore_index=ignore_index)
        dom_nll[dom] = float(dm["nll_mean"])

    return dom_nll, {
        "nll_mean": float(global_m["nll_mean"]),
        "ece_value": float(global_m["expected_calibration_error"]),
        "top1_accuracy": float(global_m["top1_accuracy"]),
        "num_tokens": int(global_m["num_tokens"]),
    }


def _compute_anse_verdict(
    ref_dom_nll: Mapping[str, float],
    ref_ece: float,
    cand_dom_nll: Mapping[str, float],
    cand_ece: float,
    *,
    cfg: MultiscaleSpectralAnisotropyRouterConfig,
) -> dict[str, Any]:
    """Compute ANSE energy, ``D0`` drift ratio, worst-domain drift, and guard verdict."""
    ratios: dict[str, float] = {}
    drifts_pct: dict[str, float] = {}
    for dom in sorted(cand_dom_nll.keys()):
        base_v = float(ref_dom_nll[dom])
        cand_v = float(cand_dom_nll[dom])
        ratio = cand_v / max(base_v, 1e-12)
        ratios[dom] = ratio
        drifts_pct[dom] = 100.0 * (ratio - 1.0)

    mean_base = sum(float(ref_dom_nll[d]) for d in cand_dom_nll) / float(len(cand_dom_nll))
    mean_cand = sum(float(cand_dom_nll[d]) for d in cand_dom_nll) / float(len(cand_dom_nll))
    mean_diff = mean_cand - mean_base
    ece_diff = float(cand_ece) - float(ref_ece)

    gen_keys = [d for d in sorted(ratios.keys()) if _is_general_domain(d, cfg.general_prefixes)]
    gen_ratio = max(ratios[d] for d in gen_keys)
    worst_dom = max(sorted(ratios.keys()), key=lambda d: ratios[d])
    worst_ratio = ratios[worst_dom]

    gen_guard_ok = gen_ratio <= (1.0 + cfg.max_general_drift_rel) + 1e-12
    worst_guard_ok = worst_ratio <= (1.0 + cfg.max_domain_drift_rel) + 1e-12
    guard_ok = bool(gen_guard_ok and worst_guard_ok)

    if guard_ok:
        soft_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0
        anse_energy = mean_diff + cfg.ece_energy_weight * ece_diff + soft_pen
    else:
        anse_energy = (
            cfg.maximum_pain_energy + max(0.0, mean_diff) + cfg.ece_energy_weight * float(cand_ece)
        )

    return {
        "nll_mean_diff": float(mean_diff),
        "ece_diff": float(ece_diff),
        "general_drift_ratio": float(gen_ratio - 1.0),
        "general_pct_vs_base": float(100.0 * (gen_ratio - 1.0)),
        "worst_domain_id": worst_dom,
        "worst_domain_pct_vs_base": float(100.0 * (worst_ratio - 1.0)),
        "general_guard_ok": gen_guard_ok,
        "worst_guard_ok": worst_guard_ok,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "delta_e": float(anse_energy),
        "domain_drifts_pct": drifts_pct,
    }


def _search_norm_candidate(
    u_fit: torch.Tensor,
    targets_fit: torch.Tensor,
    domains_fit: Sequence[str],
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    base_dom_nll: Mapping[str, float],
    base_ece: float,
    *,
    delta_w: torch.Tensor | None,
    eta_grid: Sequence[float],
    cfg: MultiscaleSpectralAnisotropyRouterConfig,
    allow_identity: bool = False,
) -> dict[str, Any]:
    """Tune step multiplier ``eta`` and folded temperature ``tau`` strictly on ``fit`` split."""
    etas = (0.0,) if delta_w is None else tuple(eta_grid)
    dw64 = torch.zeros_like(w0, dtype=torch.float64) if delta_w is None else delta_w.to(dtype=torch.float64)

    base_dom_f, base_g_f = _eval_norm_metrics(
        u_fit,
        targets_fit,
        domains_fit,
        w0,
        b0,
        head_w,
        head_b,
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )
    init_ver = _compute_anse_verdict(base_dom_nll, base_ece, base_dom_f, base_g_f["ece_value"], cfg=cfg)
    best: dict[str, Any] = {
        "found": bool(allow_identity and init_ver["guard_ok"]),
        "eta": 0.0,
        "tau": 1.0,
        "w": w0.clone(),
        "b": b0.clone() if b0 is not None else None,
        "head_b": head_b.clone() if head_b is not None else None,
        "probe_energy": float(init_ver["anse_energy"]) if allow_identity else 0.0,
    }

    for eta in etas:
        w_unfolded = w0 + float(eta) * dw64
        for tau in cfg.tau_grid:
            if not allow_identity and abs(float(eta)) <= 1e-15 and abs(float(tau) - 1.0) <= 1e-12:
                continue
            w_try, b_try = fold_temperature_into_norm_params(w_unfolded, b0, float(tau))
            hb_try = (head_b / float(tau)) if head_b is not None else None
            dom_m, glob_m = _eval_norm_metrics(
                u_fit,
                targets_fit,
                domains_fit,
                w_try,
                b_try,
                head_w,
                hb_try,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
            ver = _compute_anse_verdict(base_dom_nll, base_ece, dom_m, glob_m["ece_value"], cfg=cfg)
            if ver["guard_ok"] and ver["anse_energy"] < best["probe_energy"] - cfg.min_energy_drop:
                best = {
                    "found": True,
                    "eta": float(eta),
                    "tau": float(tau),
                    "w": w_try,
                    "b": b_try,
                    "head_b": hb_try,
                    "probe_energy": float(ver["anse_energy"]),
                }
    return best


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a detached CPU clone of ``model.state_dict()``."""
    return {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}


def _restore_state_inplace(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``snapshot`` onto ``model`` matching parameter devices and dtypes."""
    cur_sd = model.state_dict()
    model.load_state_dict({k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()})


def _count_edited_params(ref_state: Mapping[str, torch.Tensor], cand_state: Mapping[str, torch.Tensor]) -> int:
    """Count modified scalar weights avoiding double-counting tied tensor storage."""
    seen_ptrs: set[int] = set()
    edited = 0
    for key in sorted(ref_state.keys()):
        r_t = ref_state[key].detach().cpu()
        c_t = cand_state[key].detach().cpu()
        ptr = r_t.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        edited += int((r_t != c_t).sum().item())
    return edited


def _apply_norm_params(
    norm_mod: nn.Module,
    head_mod: nn.Module,
    w_cand: torch.Tensor,
    b_cand: torch.Tensor | None,
    head_b_cand: torch.Tensor | None,
) -> None:
    """Copy calibrated norm weights (and optional biases) into model modules."""
    norm_mod.weight.copy_(w_cand.to(device=norm_mod.weight.device, dtype=norm_mod.weight.dtype))
    if getattr(norm_mod, "bias", None) is not None and b_cand is not None:
        norm_mod.bias.copy_(b_cand.to(device=norm_mod.bias.device, dtype=norm_mod.bias.dtype))
    if getattr(head_mod, "bias", None) is not None and head_b_cand is not None:
        head_mod.bias.copy_(head_b_cand.to(device=head_mod.bias.device, dtype=head_mod.bias.dtype))


def _validate_lora_adapters(
    lora_adapters: Sequence[Mapping[str, Any]] | None,
    model: nn.Module,
) -> list[dict[str, Any]]:
    """Validate optional LoRA adapter specs against resolved model linear modules."""
    if lora_adapters is None:
        return []
    if isinstance(lora_adapters, (str, bytes, bytearray)) or not isinstance(lora_adapters, Sequence):
        raise TypeError("lora_adapters must be a sequence of adapter mappings or None")

    validated: list[dict[str, Any]] = []
    for a_idx, adapter in enumerate(lora_adapters):
        if not isinstance(adapter, Mapping) or "modules" not in adapter:
            raise ValueError(f"lora_adapters[{a_idx}] must be a mapping containing 'modules'")
        a_name = _validate_nonempty_str(
            adapter.get("adapter_name", f"adapter_{a_idx}"), f"lora_adapters[{a_idx}]['adapter_name']"
        )
        mods = adapter["modules"]
        if not isinstance(mods, Mapping) or len(mods) == 0:
            raise ValueError(f"lora_adapters[{a_idx}]['modules'] must be a non-empty mapping")

        clean_mods: dict[str, dict[str, Any]] = {}
        for mod_path, spec in mods.items():
            m_path = _validate_nonempty_str(mod_path, f"{a_name} module path")
            mod_obj = _resolve_dotted_attr(model, m_path)
            if not isinstance(mod_obj, nn.Module):
                raise ValueError(f"LoRA target module {m_path!r} not found on model")
            w_mod = _validate_fp_tensor(getattr(mod_obj, "weight", None), f"model.{m_path}.weight", exact_ndim=2)
            d_out, d_in = int(w_mod.shape[0]), int(w_mod.shape[1])

            if not isinstance(spec, Mapping):
                raise ValueError(f"{a_name}.modules[{m_path!r}] must be a mapping")
            if "a_mat" in spec and "b_mat" in spec:
                a_t = _validate_fp_tensor(spec["a_mat"], f"{a_name}.{m_path}.a_mat", exact_ndim=2)
                b_t = _validate_fp_tensor(spec["b_mat"], f"{a_name}.{m_path}.b_mat", exact_ndim=2)
                if int(a_t.shape[1]) != d_in or int(b_t.shape[0]) != d_out or int(a_t.shape[0]) != int(b_t.shape[1]):
                    raise ValueError(
                        f"{a_name}.{m_path} LoRA shapes {tuple(b_t.shape)}, {tuple(a_t.shape)} incompatible with ({d_out}, {d_in})"
                    )
                scale = _validate_finite_float(spec.get("scale", 1.0), f"{a_name}.{m_path}.scale")
                clean_mods[m_path] = {
                    "a_mat": a_t.detach().to(device="cpu", dtype=torch.float64),
                    "b_mat": b_t.detach().to(device="cpu", dtype=torch.float64),
                    "scale": scale,
                }
            elif "delta_weight" in spec:
                dw_t = _validate_fp_tensor(spec["delta_weight"], f"{a_name}.{m_path}.delta_weight", exact_ndim=2)
                if tuple(dw_t.shape) != (d_out, d_in):
                    raise ValueError(f"{a_name}.{m_path}.delta_weight must have shape ({d_out}, {d_in})")
                clean_mods[m_path] = {"delta_weight": dw_t.detach().to(device="cpu", dtype=torch.float64)}
            else:
                raise ValueError(f"{a_name}.modules[{m_path!r}] must provide ('a_mat', 'b_mat') or 'delta_weight'")

        validated.append({"adapter_name": a_name, "modules": clean_mods})
    return validated


def _compute_module_prose_bases(
    model: nn.Module,
    mod_paths: Sequence[str],
    gen_ids_2d: torch.Tensor,
    *,
    p0_sv_rel_tol: float,
    max_p0_rank: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Capture ``D0`` pre-layer inputs per targeted module and extract orthonormal prose bases ``V_0``."""
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    buffers: dict[str, list[torch.Tensor]] = {p: [] for p in mod_paths}
    handles: list[Any] = []

    for m_path in mod_paths:
        mod_obj = _resolve_dotted_attr(model, m_path)
        assert isinstance(mod_obj, nn.Module)

        def _make_hook(target_key: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
            def _hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
                if inp and isinstance(inp[0], torch.Tensor):
                    buffers[target_key].append(inp[0].detach().to(device="cpu", dtype=torch.float64))

            return _hook

        handles.append(mod_obj.register_forward_pre_hook(_make_hook(m_path)))

    try:
        for b0 in range(0, int(gen_ids_2d.shape[0]), batch_size):
            _ = model(gen_ids_2d[b0 : b0 + batch_size].to(dev))
    finally:
        for h in handles:
            h.remove()

    bases: dict[str, torch.Tensor] = {}
    for m_path in mod_paths:
        mod_obj = _resolve_dotted_attr(model, m_path)
        assert isinstance(mod_obj, nn.Module)
        d_in = int(mod_obj.weight.shape[1])
        if not buffers[m_path]:
            bases[m_path] = torch.empty((0, d_in), dtype=torch.float64)
            continue
        x_prose = torch.cat([t.reshape(-1, d_in) for t in buffers[m_path]], dim=0)
        _, s_x, vh_x = torch.linalg.svd(x_prose, full_matrices=False)
        if s_x.numel() == 0 or float(s_x[0].item()) <= 1e-15 or d_in <= 1:
            bases[m_path] = torch.empty((0, d_in), dtype=torch.float64)
            continue
        k_keep = int((s_x > p0_sv_rel_tol * float(s_x[0].item())).sum().item())
        k_keep = max(1, min(k_keep, max_p0_rank, d_in - 1))
        bases[m_path] = vh_x[:k_keep, :].contiguous()
    return bases


def _build_lora_module_deltas(
    adapters: Sequence[Mapping[str, Any]],
    prose_bases: Mapping[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Assemble naive or associative ``(scale * B) @ (A @ P_0)`` merged deltas per module."""
    merged: dict[str, torch.Tensor] = {}
    for adapter in adapters:
        for m_path, spec in adapter["modules"].items():
            v0 = prose_bases.get(m_path) if prose_bases is not None else None
            if "a_mat" in spec:
                a_m = spec["a_mat"]
                if v0 is not None and v0.shape[0] > 0:
                    a_eff = a_m - (a_m @ v0.T) @ v0
                else:
                    a_eff = a_m
                delta = (spec["scale"] * spec["b_mat"]) @ a_eff
            else:
                dw = spec["delta_weight"]
                delta = dw - (dw @ v0.T) @ v0 if (v0 is not None and v0.shape[0] > 0) else dw
            merged[m_path] = delta.clone() if m_path not in merged else (merged[m_path] + delta)
    return merged


def _run_external_eval(eval_fn: Callable[[nn.Module], Any], model: nn.Module) -> tuple[dict[str, float], float]:
    """Invoke external verifier and normalize to ``(domain_nll_map, ece_value)``."""
    raw = eval_fn(model)
    ece_val = 0.0
    dom_raw = raw
    if isinstance(raw, tuple) and len(raw) == 2:
        ece_part, dom_raw = raw
        ece_val = 0.0 if ece_part is None else _validate_nonneg_finite(ece_part, "eval_fn ece")
    if not isinstance(dom_raw, Mapping) or len(dom_raw) == 0:
        raise ValueError("eval_fn must return a non-empty domain->NLL mapping or (ece, mapping)")
    dom_map = {
        _validate_nonempty_str(k, "eval_fn domain"): _validate_positive_finite(v, f"eval_fn[{k!r}]")
        for k, v in dom_raw.items()
    }
    return dom_map, ece_val


# ---------------------------------------------------------------------------
# 4. Regime Execution Pipeline & Atomic ANSE Ratchet
# ---------------------------------------------------------------------------


def _is_energy_improving(verdict: Mapping[str, Any], cfg: MultiscaleSpectralAnisotropyRouterConfig) -> bool:
    """Return True when ANSE energy strictly improves under configured ratchet rule."""
    if cfg.require_strict_energy_drop:
        return float(verdict["delta_e"]) < -cfg.min_energy_drop
    return float(verdict["delta_e"]) <= 0.0


def _execute_scalar_or_js_regime(
    *,
    regime: str,
    rec_rank: int,
    u_fit: torch.Tensor,
    fit_prep: Mapping[str, Any],
    u_sel: torch.Tensor,
    sel_prep: Mapping[str, Any],
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    norm_mod: nn.Module,
    head_mod: nn.Module,
    cfg: MultiscaleSpectralAnisotropyRouterConfig,
) -> dict[str, Any]:
    """Execute either 1-DOF folded scalar calibration or Empirical-Bayes James-Stein rank-k surgery."""
    fit_base_dom, fit_base_g = _eval_norm_metrics(
        u_fit,
        fit_prep["targets_2d"],
        fit_prep["domains"],
        w0,
        b0,
        head_w,
        head_b,
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )
    scalar_cand = _search_norm_candidate(
        u_fit,
        fit_prep["targets_2d"],
        fit_prep["domains"],
        w0,
        b0,
        head_w,
        head_b,
        fit_base_dom,
        fit_base_g["ece_value"],
        delta_w=None,
        eta_grid=(0.0,),
        cfg=cfg,
    )
    sel_base_dom, sel_base_g = _eval_norm_metrics(
        u_sel,
        sel_prep["targets_2d"],
        sel_prep["domains"],
        w0,
        b0,
        head_w,
        head_b,
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )
    sel_sc_dom, sel_sc_g = _eval_norm_metrics(
        u_sel,
        sel_prep["targets_2d"],
        sel_prep["domains"],
        scalar_cand["w"],
        scalar_cand["b"],
        head_w,
        scalar_cand["head_b"],
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )
    sel_sc_anse = _compute_anse_verdict(
        sel_base_dom, sel_base_g["ece_value"], sel_sc_dom, sel_sc_g["ece_value"], cfg=cfg
    )
    scalar_ok = bool(scalar_cand["found"] and sel_sc_anse["guard_ok"] and _is_energy_improving(sel_sc_anse, cfg))

    chosen_arm = "folded_scalar_rmsnorm_only" if scalar_ok else "base_rollback"
    chosen_cand: dict[str, Any] = (
        scalar_cand
        if scalar_ok
        else {
            "eta": 0.0,
            "tau": 1.0,
            "w": w0.clone(),
            "b": b0.clone() if b0 is not None else None,
            "head_b": head_b.clone() if head_b is not None else None,
        }
    )
    details: dict[str, Any] = {
        "effective_rank_k": 0,
        "mean_shrinkage_factor": 0.0,
        "nullspace_orthogonality_max_abs": 0.0,
        "ratcheted_to_scalar": False,
    }

    if regime == REGIME_JAMES_STEIN_RANK_K_FISHER:
        chunk_grads, tok_fisher = _compute_chunk_norm_gradients(
            u_fit,
            fit_prep["targets_2d"],
            w0,
            b0,
            head_w,
            head_b,
            ignore_index=cfg.ignore_index,
        )
        gen_mask = fit_prep["is_general"]
        gen_grads = chunk_grads[gen_mask]
        spec_grads = chunk_grads[~gen_mask]

        mu_g = chunk_grads.mean(dim=0)
        sig_e = float(mu_g.pow(2).sum().item())
        n_c = int(chunk_grads.shape[0])
        noise_e = float((chunk_grads.var(dim=0, unbiased=True) / max(n_c, 1)).sum().item()) if n_c >= 2 else 0.0
        probe_snr = sig_e / max(1e-30, noise_e) if (n_c >= 2 and sig_e > 0.0) else 0.0

        js_fit = fit_james_stein_rank_k_fisher_norm_step(
            spec_grads,
            gen_grads,
            max_rank=max(1, rec_rank),
            ridge_scale=cfg.ridge_scale,
            max_step_norm=cfg.max_step_norm,
            tok_grads_fisher_diag=tok_fisher,
            nullspace_sv_tol=cfg.nullspace_sv_tol,
        )
        details["effective_rank_k"] = int(js_fit["effective_rank_k"])
        details["mean_shrinkage_factor"] = float(js_fit["mean_shrinkage_factor"])
        details["nullspace_orthogonality_max_abs"] = float(js_fit["nullspace_orthogonality_max_abs"])
        details["probe_snr"] = float(probe_snr)

        if (
            probe_snr >= cfg.min_probe_snr
            and not js_fit["zero_step"]
            and js_fit["mean_shrinkage_factor"] >= cfg.min_shrinkage_to_activate
        ):
            js_cand = _search_norm_candidate(
                u_fit,
                fit_prep["targets_2d"],
                fit_prep["domains"],
                w0,
                b0,
                head_w,
                head_b,
                fit_base_dom,
                fit_base_g["ece_value"],
                delta_w=js_fit["delta_w"],
                eta_grid=cfg.eta_grid,
                cfg=cfg,
            )
            sel_js_dom, sel_js_g = _eval_norm_metrics(
                u_sel,
                sel_prep["targets_2d"],
                sel_prep["domains"],
                js_cand["w"],
                js_cand["b"],
                head_w,
                js_cand["head_b"],
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
            sel_js_anse = _compute_anse_verdict(
                sel_base_dom, sel_base_g["ece_value"], sel_js_dom, sel_js_g["ece_value"], cfg=cfg
            )
            js_ok = bool(js_cand["found"] and sel_js_anse["guard_ok"] and _is_energy_improving(sel_js_anse, cfg))
            if js_ok and (not scalar_ok or float(sel_js_anse["anse_energy"]) < float(sel_sc_anse["anse_energy"]) - 1e-12):
                chosen_arm = REGIME_JAMES_STEIN_RANK_K_FISHER
                chosen_cand = js_cand
            elif scalar_ok:
                details["ratcheted_to_scalar"] = True
        elif scalar_ok:
            details["ratcheted_to_scalar"] = True

    _apply_norm_params(norm_mod, head_mod, chosen_cand["w"], chosen_cand["b"], chosen_cand["head_b"])
    details["chosen_arm"] = chosen_arm
    details["tau_temp"] = float(chosen_cand["tau"])
    details["eta_scale"] = float(chosen_cand["eta"])
    return details


def _execute_p0_lora_regime(
    *,
    work_model: nn.Module,
    adapters: Sequence[Mapping[str, Any]],
    norm_mod: nn.Module,
    head_mod: nn.Module,
    resolved_norm: str,
    eps_eff: float,
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    fit_prep: Mapping[str, Any],
    sel_prep: Mapping[str, Any],
    u_fit_base: torch.Tensor,
    u_sel_base: torch.Tensor,
    cfg: MultiscaleSpectralAnisotropyRouterConfig,
) -> dict[str, Any]:
    """Execute conditional ``P_0`` LoRA null-space projection + folded norm calibration."""
    if not adapters:
        det = _execute_scalar_or_js_regime(
            regime=REGIME_FOLDED_SCALAR_RMSNORM_ONLY,
            rec_rank=0,
            u_fit=u_fit_base,
            fit_prep=fit_prep,
            u_sel=u_sel_base,
            sel_prep=sel_prep,
            w0=w0,
            b0=b0,
            head_w=head_w,
            head_b=head_b,
            norm_mod=norm_mod,
            head_mod=head_mod,
            cfg=cfg,
        )
        det["p0_gate_mode"] = "no_adapters_folded_norm_only"
        det["requires_trunk_recapture"] = False
        return det

    mod_paths = sorted({m_path for ad in adapters for m_path in ad["modules"]})
    base_mod_weights: dict[str, torch.Tensor] = {}
    for m_path in mod_paths:
        mod_obj = _resolve_dotted_attr(work_model, m_path)
        assert isinstance(mod_obj, nn.Module)
        base_mod_weights[m_path] = mod_obj.weight.detach().to(device="cpu", dtype=torch.float64).clone()

    gen_ids = fit_prep["ids_2d"][fit_prep["is_general"]]
    prose_bases = _compute_module_prose_bases(
        work_model,
        mod_paths,
        gen_ids,
        p0_sv_rel_tol=cfg.p0_sv_rel_tol,
        max_p0_rank=cfg.max_p0_rank,
        batch_size=cfg.batch_size,
    )
    naive_deltas = _build_lora_module_deltas(adapters, None)
    p0_deltas = _build_lora_module_deltas(adapters, prose_bases)

    fit_base_dom, fit_base_g = _eval_norm_metrics(
        u_fit_base,
        fit_prep["targets_2d"],
        fit_prep["domains"],
        w0,
        b0,
        head_w,
        head_b,
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )
    sel_base_dom, sel_base_g = _eval_norm_metrics(
        u_sel_base,
        sel_prep["targets_2d"],
        sel_prep["domains"],
        w0,
        b0,
        head_w,
        head_b,
        num_bins=cfg.num_bins,
        ignore_index=cfg.ignore_index,
    )

    candidates: list[dict[str, Any]] = []
    for mode_label, deltas in (("conditional_bypass_benign_lora", naive_deltas), ("p0_nullspace_projected", p0_deltas)):
        for m_path, base_w in base_mod_weights.items():
            mod_obj = _resolve_dotted_attr(work_model, m_path)
            assert isinstance(mod_obj, nn.Module)
            upd = base_w + deltas[m_path] if m_path in deltas else base_w
            mod_obj.weight.copy_(upd.to(device=mod_obj.weight.device, dtype=mod_obj.weight.dtype))
        h_fit_m = _capture_pre_norm_states(work_model, norm_mod, fit_prep["ids_2d"], batch_size=cfg.batch_size)
        h_sel_m = _capture_pre_norm_states(work_model, norm_mod, sel_prep["ids_2d"], batch_size=cfg.batch_size)
        _, u_fit_m = _extract_scored_unit_states(
            h_fit_m, fit_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
        )
        _, u_sel_m = _extract_scored_unit_states(
            h_sel_m, sel_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
        )
        norm_fit = _search_norm_candidate(
            u_fit_m,
            fit_prep["targets_2d"],
            fit_prep["domains"],
            w0,
            b0,
            head_w,
            head_b,
            fit_base_dom,
            fit_base_g["ece_value"],
            delta_w=None,
            eta_grid=(0.0,),
            cfg=cfg,
            allow_identity=True,
        )
        sel_dom, sel_g = _eval_norm_metrics(
            u_sel_m,
            sel_prep["targets_2d"],
            sel_prep["domains"],
            norm_fit["w"],
            norm_fit["b"],
            head_w,
            norm_fit["head_b"],
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        sel_anse = _compute_anse_verdict(sel_base_dom, sel_base_g["ece_value"], sel_dom, sel_g["ece_value"], cfg=cfg)
        feasible = bool(norm_fit["found"] and sel_anse["guard_ok"] and _is_energy_improving(sel_anse, cfg))
        candidates.append(
            {
                "mode": mode_label,
                "deltas": deltas,
                "norm_fit": norm_fit,
                "sel_energy": float(sel_anse["anse_energy"]),
                "feasible": feasible,
            }
        )

    feasible_list = [c for c in candidates if c["feasible"]]
    if feasible_list:
        feasible_list.sort(key=lambda item: item["sel_energy"])
        winner = feasible_list[0]
        for m_path, base_w in base_mod_weights.items():
            mod_obj = _resolve_dotted_attr(work_model, m_path)
            assert isinstance(mod_obj, nn.Module)
            upd = base_w + winner["deltas"][m_path] if m_path in winner["deltas"] else base_w
            mod_obj.weight.copy_(upd.to(device=mod_obj.weight.device, dtype=mod_obj.weight.dtype))
        nf = winner["norm_fit"]
        _apply_norm_params(norm_mod, head_mod, nf["w"], nf["b"], nf["head_b"])
        return {
            "chosen_arm": REGIME_P0_LORA_PLUS_FOLDED_NORM,
            "p0_gate_mode": winner["mode"],
            "tau_temp": float(nf["tau"]),
            "eta_scale": 0.0,
            "requires_trunk_recapture": True,
            "num_lora_modules_edited": len(mod_paths),
        }

    # Restore base module weights and ratchet to clean scalar folded norm
    for m_path, base_w in base_mod_weights.items():
        mod_obj = _resolve_dotted_attr(work_model, m_path)
        assert isinstance(mod_obj, nn.Module)
        mod_obj.weight.copy_(base_w.to(device=mod_obj.weight.device, dtype=mod_obj.weight.dtype))
    det = _execute_scalar_or_js_regime(
        regime=REGIME_FOLDED_SCALAR_RMSNORM_ONLY,
        rec_rank=0,
        u_fit=u_fit_base,
        fit_prep=fit_prep,
        u_sel=u_sel_base,
        sel_prep=sel_prep,
        w0=w0,
        b0=b0,
        head_w=head_w,
        head_b=head_b,
        norm_mod=norm_mod,
        head_mod=head_mod,
        cfg=cfg,
    )
    det["p0_gate_mode"] = "ratcheted_to_clean_scalar"
    det["requires_trunk_recapture"] = False
    return det


@torch.no_grad()
def execute_regime_routed_wsp_plan(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    lora_adapters: Sequence[Mapping[str, Any]] | None = None,
    config: MultiscaleSpectralAnisotropyRouterConfig | None = None,
    architecture_kind: str | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    eval_fn: Callable[[nn.Module], Any] | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Profile spectral anisotropy, route to the safe WSP operator regime, and verify under ANSE.

    Enforces ``ANSE Delta E < 0`` and ``D0 <= +1.0%`` across ``probe`` / ``selection`` /
    ``eval`` splits, abstaining without weight mutation on under-sampled MLM probes and
    performing atomic bitwise rollback under ``ANSE_MAXIMUM_PAIN`` on any guard breach.
    """
    if config is not None and not isinstance(config, MultiscaleSpectralAnisotropyRouterConfig):
        raise TypeError(
            f"config must be MultiscaleSpectralAnisotropyRouterConfig or None, got {type(config).__name__}"
        )
    cfg = config or MultiscaleSpectralAnisotropyRouterConfig()
    _validate_bool(inplace, "inplace")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    explicit_arch = _validate_architecture_kind(architecture_kind) if architecture_kind is not None else None

    _, _, _, norm_name_chk, head_name_chk, hidden_dim, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
    validated_adapters = _validate_lora_adapters(lora_adapters, model)

    if selection_chunks is None:
        if isinstance(probe_chunks, (str, bytes, bytearray)) or not isinstance(probe_chunks, Sequence):
            raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")
        fit_raw, sel_raw = _split_probe_chunks_fallback(probe_chunks, cfg.general_prefixes)
    else:
        fit_raw, sel_raw = probe_chunks, selection_chunks

    fit_prep = _prepare_chunks(
        fit_raw,
        name="probe_chunks",
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        ignore_index=cfg.ignore_index,
        general_prefixes=cfg.general_prefixes,
    )
    sel_prep = _prepare_chunks(
        sel_raw,
        name="selection_chunks",
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        ignore_index=cfg.ignore_index,
        general_prefixes=cfg.general_prefixes,
    )
    ver_prep = (
        sel_prep
        if eval_chunks is None
        else _prepare_chunks(
            eval_chunks,
            name="eval_chunks",
            vocab_size=vocab_size,
            score_from=cfg.score_from,
            ignore_index=cfg.ignore_index,
            general_prefixes=cfg.general_prefixes,
        )
    )
    resolved_arch = explicit_arch or ("bidirectional_mlm" if fit_prep["is_mlm"] else "causal_lm")

    work_model = model if inplace else copy.deepcopy(model)
    norm_mod, head_mod, resolved_norm, norm_name, head_name, _, _, eps_eff = _resolve_norm_and_head(
        work_model, norm_kind=norm_kind, norm_attr=norm_name_chk, head_attr=head_name_chk
    )
    eff_total_params = sum(p.numel() for p in {p.data_ptr(): p for p in work_model.parameters()}.values())

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

        h_fit_raw = _capture_pre_norm_states(work_model, norm_mod, fit_prep["ids_2d"], batch_size=cfg.batch_size)
        h_sel_raw = _capture_pre_norm_states(work_model, norm_mod, sel_prep["ids_2d"], batch_size=cfg.batch_size)
        h_ver_raw = (
            h_sel_raw
            if eval_chunks is None
            else _capture_pre_norm_states(work_model, norm_mod, ver_prep["ids_2d"], batch_size=cfg.batch_size)
        )

        h_fit_scored, u_fit = _extract_scored_unit_states(
            h_fit_raw, fit_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
        )
        _, u_sel = _extract_scored_unit_states(
            h_sel_raw, sel_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
        )
        _, u_ver_base = (
            (None, u_sel)
            if eval_chunks is None
            else _extract_scored_unit_states(
                h_ver_raw, ver_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
            )
        )

        valid_fit_tokens = int((fit_prep["targets_2d"] != cfg.ignore_index).sum().item())
        profile = profile_hidden_spectral_anisotropy(
            h_fit_scored,
            w0,
            architecture_kind=resolved_arch,
            effective_tokens_per_split=valid_fit_tokens,
            outlier_ratio_threshold=cfg.outlier_ratio_threshold,
            self_suppression_threshold=cfg.self_suppression_threshold,
            norm_bias=b0,
            head_bias_dim=int(head_b.numel()) if head_b is not None else 0,
            has_lora_adapters=bool(validated_adapters),
        )
        routing = select_wsp_operator_regime(
            profile,
            min_causal_tokens_per_dof=cfg.min_causal_tokens_per_dof,
            min_mlm_masked_tokens_per_dof=cfg.min_mlm_masked_tokens_per_dof,
            min_causal_scalar_tokens_per_dof=cfg.min_causal_scalar_tokens_per_dof,
            min_mlm_scalar_tokens_per_dof=cfg.min_mlm_scalar_tokens_per_dof,
            high_anisotropy_threshold=cfg.high_anisotropy_threshold,
            extreme_anisotropy_threshold=cfg.extreme_anisotropy_threshold,
            massive_outlier_ratio_threshold=cfg.massive_outlier_ratio_threshold,
            self_suppression_threshold=cfg.self_suppression_threshold,
            max_rank=cfg.max_rank,
            allow_mlm_scalar_fallback=cfg.allow_mlm_scalar_fallback,
        )
        selected_regime = routing["selected_regime"]

        ver_base_dom, ver_base_g = _eval_norm_metrics(
            u_ver_base,
            ver_prep["targets_2d"],
            ver_prep["domains"],
            w0,
            b0,
            head_w,
            head_b,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        ext_base = _run_external_eval(eval_fn, work_model) if eval_fn is not None else None

        abstained = selected_regime == REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS
        operator_details: dict[str, Any] = {
            "chosen_arm": REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS if abstained else selected_regime,
            "tau_temp": 1.0,
            "eta_scale": 0.0,
            "requires_trunk_recapture": False,
        }
        if not abstained:
            if selected_regime == REGIME_P0_LORA_PLUS_FOLDED_NORM:
                operator_details = _execute_p0_lora_regime(
                    work_model=work_model,
                    adapters=validated_adapters,
                    norm_mod=norm_mod,
                    head_mod=head_mod,
                    resolved_norm=resolved_norm,
                    eps_eff=eps_eff,
                    w0=w0,
                    b0=b0,
                    head_w=head_w,
                    head_b=head_b,
                    fit_prep=fit_prep,
                    sel_prep=sel_prep,
                    u_fit_base=u_fit,
                    u_sel_base=u_sel,
                    cfg=cfg,
                )
            else:
                operator_details = _execute_scalar_or_js_regime(
                    regime=selected_regime,
                    rec_rank=int(routing["recommended_max_rank"]),
                    u_fit=u_fit,
                    fit_prep=fit_prep,
                    u_sel=u_sel,
                    sel_prep=sel_prep,
                    w0=w0,
                    b0=b0,
                    head_w=head_w,
                    head_b=head_b,
                    norm_mod=norm_mod,
                    head_mod=head_mod,
                    cfg=cfg,
                )

        cand_snapshot = _snapshot_state(work_model)
        cand_sha = compute_module_state_sha256(work_model)

        if abstained:
            ver_cand_dom, ver_cand_g = ver_base_dom, ver_base_g
            ver_anse = _compute_anse_verdict(
                ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
            )
            accepted, rolled_back = False, False
            status = "ABSTAINED_INSUFFICIENT_PROBE_TOKENS"
        else:
            if operator_details.get("requires_trunk_recapture"):
                h_ver_cand = _capture_pre_norm_states(
                    work_model, norm_mod, ver_prep["ids_2d"], batch_size=cfg.batch_size
                )
                _, u_ver_cand = _extract_scored_unit_states(
                    h_ver_cand, ver_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
                )
            else:
                u_ver_cand = u_ver_base

            w_cand = norm_mod.weight.detach().to(device="cpu", dtype=torch.float64)
            b_cand_raw = getattr(norm_mod, "bias", None)
            b_cand = b_cand_raw.detach().to(device="cpu", dtype=torch.float64) if b_cand_raw is not None else None
            hb_cand_raw = getattr(head_mod, "bias", None)
            hb_cand = hb_cand_raw.detach().to(device="cpu", dtype=torch.float64) if hb_cand_raw is not None else None
            ver_cand_dom, ver_cand_g = _eval_norm_metrics(
                u_ver_cand,
                ver_prep["targets_2d"],
                ver_prep["domains"],
                w_cand,
                b_cand,
                head_w,
                hb_cand,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )

            eval_failed = False
            if eval_fn is not None and ext_base is not None:
                try:
                    ext_cand = _run_external_eval(eval_fn, work_model)
                    ver_anse = _compute_anse_verdict(
                        ext_base[0], ext_base[1], ext_cand[0], ext_cand[1], cfg=cfg
                    )
                except Exception as exc:  # pragma: no cover - defensive runtime isolation
                    eval_failed = True
                    operator_details["eval_error"] = str(exc)
                    ver_anse = {
                        "general_drift_ratio": 0.0,
                        "general_pct_vs_base": 0.0,
                        "worst_domain_id": str(sorted(ver_base_dom.keys())[0]),
                        "worst_domain_pct_vs_base": 0.0,
                        "guard_ok": False,
                        "anse_energy": float(cfg.maximum_pain_energy),
                        "delta_e": float(cfg.maximum_pain_energy),
                    }
            else:
                ver_anse = _compute_anse_verdict(
                    ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
                )

            if eval_failed:
                accepted, rolled_back = False, True
                status = "ROLLED_BACK_EVAL_FAILURE"
                _restore_state_inplace(work_model, initial_snapshot)
            elif cand_sha == initial_sha:
                accepted, rolled_back = False, False
                status = "FIXED_POINT_UNCHANGED" if ver_anse["guard_ok"] else "REJECTED_GUARD_VIOLATION_UNCHANGED"
            elif ver_anse["guard_ok"] and _is_energy_improving(ver_anse, cfg):
                accepted, rolled_back = True, False
                status = f"ACCEPTED_{selected_regime.upper()}"
            else:
                accepted, rolled_back = False, True
                status = (
                    "ROLLED_BACK_GUARD_VIOLATION" if not ver_anse["guard_ok"] else "ROLLED_BACK_ENERGY_NON_IMPROVING"
                )
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
            "general_pct_vs_base": 0.0,
            "worst_domain_id": str(ver_anse["worst_domain_id"]),
            "worst_domain_pct_vs_base": 0.0,
            "delta_e": 0.0 if ver_anse["guard_ok"] else float(ver_anse["delta_e"]),
        }
    )
    edited_params = _count_edited_params(initial_snapshot, cand_snapshot) if accepted else 0

    receipt: dict[str, Any] = {
        "object": ROUTER_RECEIPT_OBJECT,
        "schema": ROUTER_SCHEMA,
        "accepted": accepted,
        "abstained": abstained,
        "rolled_back": rolled_back,
        "status": status,
        "selected_regime": selected_regime,
        "architecture_kind": resolved_arch,
        "norm_kind": resolved_norm,
        "norm_attr": norm_name,
        "head_attr": head_name,
        "nll_mean": float(active_g["nll_mean"]),
        "ece_value": float(active_g["ece_value"]),
        "top1_energy_frac": float(profile["top1_energy_frac"]),
        "general_drift_ratio": float(active_anse["general_drift_ratio"]),
        "general_pct_vs_base": float(active_anse["general_pct_vs_base"]),
        "worst_domain_id": str(active_anse["worst_domain_id"]),
        "worst_domain_pct_vs_base": float(active_anse["worst_domain_pct_vs_base"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "cand_anse_energy": float(ver_anse["anse_energy"]),
        "delta_e": float(active_anse["delta_e"]),
        "guard_ok": bool(ver_anse["guard_ok"]),
        "base_nll_mean": float(ver_base_g["nll_mean"]),
        "base_ece_value": float(ver_base_g["ece_value"]),
        "cand_nll_mean": float(ver_cand_g["nll_mean"]),
        "cand_ece_value": float(ver_cand_g["ece_value"]),
        "spectral_profile": profile,
        "routing_decision": routing,
        "operator_details": operator_details,
        "edited_params": edited_params,
        "total_model_params": eff_total_params,
        "edited_param_fraction": float(edited_params) / float(eff_total_params),
        "initial_state_sha256": initial_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_GENERAL_RATIO",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ECE_ENERGY_WEIGHT",
    "DEFAULT_ETA_GRID",
    "DEFAULT_EXTREME_ANISOTROPY_THRESHOLD",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_HIGH_ANISOTROPY_THRESHOLD",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_MASSIVE_OUTLIER_RATIO_THRESHOLD",
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MAX_P0_RANK",
    "DEFAULT_MAX_RANK",
    "DEFAULT_MAX_STEP_NORM",
    "DEFAULT_MIN_CAUSAL_SCALAR_TOKENS_PER_DOF",
    "DEFAULT_MIN_CAUSAL_TOKENS_PER_DOF",
    "DEFAULT_MIN_ENERGY_DROP",
    "DEFAULT_MIN_MLM_MASKED_TOKENS_PER_DOF",
    "DEFAULT_MIN_MLM_SCALAR_TOKENS_PER_DOF",
    "DEFAULT_MIN_PROBE_SNR",
    "DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_NUM_BINS",
    "DEFAULT_OUTLIER_RATIO_THRESHOLD",
    "DEFAULT_P0_SV_REL_TOL",
    "DEFAULT_RIDGE_SCALE",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SELF_SUPPRESSION_RATIO_THRESHOLD",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_TAU_GRID",
    "MultiscaleSpectralAnisotropyRouterConfig",
    "REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS",
    "REGIME_FOLDED_SCALAR_RMSNORM_ONLY",
    "REGIME_JAMES_STEIN_RANK_K_FISHER",
    "REGIME_P0_LORA_PLUS_FOLDED_NORM",
    "ROUTER_RECEIPT_OBJECT",
    "ROUTER_SCHEMA",
    "VALID_ARCHITECTURE_KINDS",
    "VALID_NORM_KINDS",
    "VALID_WSP_REGIMES",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "execute_regime_routed_wsp_plan",
    "profile_hidden_spectral_anisotropy",
    "select_wsp_operator_regime",
    "verify_regime_routed_wsp_receipt_provenance",
]
