# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Empirical-Bayes James-Stein Rank-k Spectral-Shrinkage Final-Norm Surgeon.

Implements training-free, zero-VRAM final normalization surgery and small-probe overfit
protection grounded in ``PILOT9`` (``Qwen/Qwen2.5-3B``), ``PILOT12`` (``Qwen/Qwen2.5-0.5B``),
and ``PILOT14`` (``answerdotai/ModernBERT-base``):

1. :func:`compute_effective_probe_dof_and_overfit_risk`:
   Diagnoses probe-to-DOF ratio (flagging underdetermined ``n_chunks / dim < min_chunks_per_dof``
   regimes where full-DOF Fisher steps overfit), SVD spectral anisotropy (``top1_sv_energy_frac``,
   ``top8_sv_energy_frac``, effective participation rank), massive-activation outlier channels,
   and cross-chunk signal-to-noise ratio.
2. :func:`fit_james_stein_rank_k_fisher_norm_step`:
   Projects specialist chunk gradients onto ``ker(G_D0)``, extracts top-``k`` dominant variance
   modes inside ``ker(G_D0)``, applies exact per-mode Empirical-Bayes James-Stein shrinkage
   ``c_j = max(0, 1 - var_noise_j / (mean_signal_j^2 + var_noise_j / n))``, preconditions by
   damped diagonal Fisher, and returns a norm-bounded null-space perturbation ``delta_w`` with
   ``|G_D0 @ delta_w| < 1e-6``.
3. :func:`execute_verified_james_stein_norm_surgery`:
   Compares 1-DOF folded-tau scalar fallback against James-Stein rank-``k`` surgery on held-out
   selection splits under ANSE ``Delta E < 0`` and ``D0 <= +1.0%``, automatically falling back
   to scalar calibration or atomic bitwise rollback when probe SNR is insufficient (preventing
   ``PILOT14`` MLM-style overfit), and emits a canonical SHA-256 provenance receipt.
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

from hivemind.swarm.folded_norm_ece_calibrator import (
    ANSE_MAXIMUM_PAIN,
    compute_token_nll_and_ece,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    compute_module_state_sha256,
    compute_payload_sha256,
    detect_norm_type,
    fold_temperature_into_norm_params,
)

SURGEON_SCHEMA: str = "hivemind.james_stein_spectral_norm_surgeon/1"
SURGEON_RECEIPT_OBJECT: str = "hivemind.james_stein_spectral_norm_surgeon_receipt"

DEFAULT_MIN_CHUNKS_PER_DOF: float = 4.0
DEFAULT_MAX_RANK: int = 4
DEFAULT_RIDGE_SCALE: float = 1e-2
DEFAULT_MAX_STEP_NORM: float = 0.05
DEFAULT_NULLSPACE_SV_TOL: float = 1e-6
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.01
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.01
DEFAULT_ANSE_SAFE_GENERAL_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_MIN_ENERGY_DROP: float = 1e-6
DEFAULT_MIN_PROBE_SNR: float = 1.0
DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE: float = 1e-3
DEFAULT_OUTLIER_RATIO_THRESHOLD: float = 5.0
DEFAULT_HIGH_ANISOTROPY_THRESHOLD: float = 0.60
DEFAULT_EXTREME_ANISOTROPY_THRESHOLD: float = 0.85
DEFAULT_NUM_BINS: int = 15
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_TAU_GRID: tuple[float, ...] = (0.90, 0.94, 0.97, 1.00, 1.03, 1.05, 1.08, 1.12, 1.18, 1.25)
DEFAULT_ETA_GRID: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
VALID_NORM_KINDS: frozenset[str] = frozenset({"auto", "layernorm", "rmsnorm"})

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


def _validate_positive_float_seq(seq: Any, name: str, *, max_val: float | None = None) -> tuple[float, ...]:
    """Validate a non-empty sequence of finite strictly positive floats."""
    if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of positive floats")
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty")
    cleaned: list[float] = []
    for idx, val in enumerate(seq):
        f_val = _validate_positive_finite(val, f"{name}[{idx}]")
        if max_val is not None and f_val > max_val:
            raise ValueError(f"{name}[{idx}] must be <= {max_val}, got {f_val}")
        cleaned.append(f_val)
    return tuple(sorted(set(cleaned)))


def _validate_general_prefixes(prefixes: Any) -> tuple[str, ...]:
    """Validate non-empty sequence of general-prose domain prefixes."""
    if isinstance(prefixes, (str, bytes, bytearray)) or not isinstance(prefixes, Sequence):
        raise TypeError("general_prefixes must be a non-empty sequence of strings")
    if not prefixes:
        raise ValueError("general_prefixes must be non-empty")
    return tuple(_validate_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(prefixes))


@dataclass(frozen=True)
class JamesSteinNormSurgeonConfig:
    """Validated configuration for James-Stein rank-k final-norm surgery and overfit guards."""

    min_chunks_per_dof: float = DEFAULT_MIN_CHUNKS_PER_DOF
    max_rank: int = DEFAULT_MAX_RANK
    ridge_scale: float = DEFAULT_RIDGE_SCALE
    max_step_norm: float = DEFAULT_MAX_STEP_NORM
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_GENERAL_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    min_energy_drop: float = DEFAULT_MIN_ENERGY_DROP
    min_probe_snr: float = DEFAULT_MIN_PROBE_SNR
    min_shrinkage_to_activate: float = DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE
    outlier_ratio_threshold: float = DEFAULT_OUTLIER_RATIO_THRESHOLD
    num_bins: int = DEFAULT_NUM_BINS
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    eta_grid: tuple[float, ...] = DEFAULT_ETA_GRID
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        _validate_positive_finite(self.min_chunks_per_dof, "min_chunks_per_dof")
        _validate_positive_int(self.max_rank, "max_rank")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_positive_finite(self.max_step_norm, "max_step_norm")
        sv_tol = _validate_positive_finite(self.nullspace_sv_tol, "nullspace_sv_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_tol must be < 1.0, got {sv_tol}")
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
        out_th = _validate_positive_finite(self.outlier_ratio_threshold, "outlier_ratio_threshold")
        if out_th <= 1.0:
            raise ValueError(f"outlier_ratio_threshold must be > 1.0, got {out_th}")
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_int(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        object.__setattr__(self, "tau_grid", _validate_positive_float_seq(self.tau_grid, "tau_grid"))
        object.__setattr__(
            self, "eta_grid", _validate_positive_float_seq(self.eta_grid, "eta_grid", max_val=1.0)
        )
        object.__setattr__(self, "general_prefixes", _validate_general_prefixes(self.general_prefixes))
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


def verify_james_stein_norm_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != SURGEON_RECEIPT_OBJECT or receipt.get("schema") != SURGEON_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Spectral Anisotropy, Effective DOF & Empirical-Bayes James-Stein Core
# ---------------------------------------------------------------------------


def _spectral_concentration(mat_2d: torch.Tensor) -> tuple[float, float, float]:
    """Compute ``(top1_sv_energy_frac, top8_sv_energy_frac, participation_rank)`` in float64."""
    s_vals = torch.linalg.svdvals(mat_2d)
    s2 = s_vals.pow(2)
    tot_energy = float(s2.sum().item())
    if tot_energy <= 1e-30:
        return 0.0, 0.0, 0.0
    top1 = float((s2[0] / tot_energy).item())
    k8 = min(8, int(s2.numel()))
    top8 = float((s2[:k8].sum() / tot_energy).item())
    s4_sum = float(s2.pow(2).sum().item())
    eff_rank = (tot_energy * tot_energy) / max(s4_sum, 1e-30)
    return top1, top8, float(eff_rank)


@torch.no_grad()
def compute_effective_probe_dof_and_overfit_risk(
    chunk_grads: torch.Tensor,
    dim: int,
    *,
    min_chunks_per_dof: float = DEFAULT_MIN_CHUNKS_PER_DOF,
    raw_activations: torch.Tensor | None = None,
    outlier_ratio_threshold: float = DEFAULT_OUTLIER_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Diagnose probe-to-DOF ratio, spectral anisotropy, massive-activation channels, and SNR.

    Grounded in ``PILOT9`` (``top1=0.6874``, rogue dims 60.4x), ``PILOT12`` (``top1=0.3481``,
    rogue dim 22.0x), and ``PILOT14`` (``top1=0.9334``, low-SNR masked MLM probe), flags when
    unregularized full-DOF Fisher surgery will overfit and selects safe shrinkage capacity.
    """
    dim_i = _validate_positive_int(dim, "dim")
    min_ratio = _validate_positive_finite(min_chunks_per_dof, "min_chunks_per_dof")
    outlier_thresh = _validate_positive_finite(outlier_ratio_threshold, "outlier_ratio_threshold")
    if outlier_thresh <= 1.0:
        raise ValueError(f"outlier_ratio_threshold must be > 1.0, got {outlier_thresh}")

    grads_t = _validate_fp_tensor(chunk_grads, "chunk_grads", exact_ndim=2)
    if int(grads_t.shape[1]) != dim_i:
        raise ValueError(f"chunk_grads width {grads_t.shape[1]} must match dim={dim_i}")
    grads_64 = grads_t.detach().to(device="cpu", dtype=torch.float64)

    if raw_activations is not None:
        act_t = _validate_fp_tensor(raw_activations, "raw_activations", min_ndim=2)
        if int(act_t.shape[-1]) != dim_i:
            raise ValueError(f"raw_activations last dim {act_t.shape[-1]} must match dim={dim_i}")
        spec_mat = act_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1, dim_i)
    else:
        spec_mat = grads_64

    n_chunks = int(grads_64.shape[0])
    probe_to_full_dof_ratio = float(n_chunks) / float(dim_i)
    safe_max_dof = int(math.floor(float(n_chunks) / min_ratio))
    full_dof_overfit_risk = bool(probe_to_full_dof_ratio < min_ratio)

    top1_sv, top8_sv, eff_spectral_rank = _spectral_concentration(spec_mat)
    grad_top1_sv, grad_top8_sv, grad_eff_rank = (
        _spectral_concentration(grads_64)
        if raw_activations is not None
        else (top1_sv, top8_sv, eff_spectral_rank)
    )

    ch_rms = spec_mat.pow(2).mean(dim=0).sqrt()
    median_rms = float(ch_rms.median().item())
    max_rms = float(ch_rms.max().item())
    denom_med = max(median_rms, 1e-12)
    max_outlier_ratio = max_rms / denom_med

    top_k = min(8, dim_i)
    top_vals, top_idxs = torch.topk(ch_rms, k=top_k)
    top_channels = [
        {
            "dim": int(idx.item()),
            "rms": float(val.item()),
            "ratio_vs_median": float(val.item()) / denom_med,
        }
        for val, idx in zip(top_vals, top_idxs, strict=True)
    ]
    outlier_channels = [row for row in top_channels if row["ratio_vs_median"] >= outlier_thresh]
    outlier_indices = [int(row["dim"]) for row in outlier_channels]

    mu = grads_64.mean(dim=0)
    signal_energy = float(mu.pow(2).sum().item())
    if n_chunks >= 2:
        se2 = grads_64.var(dim=0, unbiased=True) / float(n_chunks)
        noise_energy = float(se2.sum().item())
        probe_snr = signal_energy / max(1e-30, noise_energy) if signal_energy > 0.0 else 0.0
    else:
        noise_energy = 0.0
        probe_snr = 0.0

    if top1_sv >= DEFAULT_EXTREME_ANISOTROPY_THRESHOLD:
        anisotropy_regime = "extreme_anisotropy"
    elif top1_sv >= DEFAULT_HIGH_ANISOTROPY_THRESHOLD:
        anisotropy_regime = "high_anisotropy"
    elif top1_sv >= 0.30:
        anisotropy_regime = "moderate_anisotropy"
    else:
        anisotropy_regime = "low_anisotropy"

    if safe_max_dof < 1:
        recommended_max_rank = 0
    else:
        est_rank = max(1, int(round(eff_spectral_rank))) if eff_spectral_rank > 0.0 else 1
        recommended_max_rank = max(1, min(safe_max_dof, est_rank, dim_i))

    probe_snr_sufficient = bool(n_chunks >= 2 and probe_snr >= DEFAULT_MIN_PROBE_SNR and safe_max_dof >= 1)
    if (
        n_chunks < 2
        or probe_snr < DEFAULT_MIN_PROBE_SNR
        or (full_dof_overfit_risk and top1_sv >= DEFAULT_EXTREME_ANISOTROPY_THRESHOLD)
    ):
        risk_tier = "CRITICAL"
    elif full_dof_overfit_risk:
        risk_tier = "HIGH"
    else:
        risk_tier = "LOW"

    if not full_dof_overfit_risk and probe_snr >= 1.5 and top1_sv < DEFAULT_EXTREME_ANISOTROPY_THRESHOLD:
        recommended_mode = "full_dof_safe"
    elif probe_snr_sufficient:
        recommended_mode = "james_stein_rank_k"
    else:
        recommended_mode = "scalar_fallback_or_abort"

    return {
        "n_chunks": n_chunks,
        "dim": dim_i,
        "min_chunks_per_dof": min_ratio,
        "probe_to_full_dof_ratio": probe_to_full_dof_ratio,
        "safe_max_dof": safe_max_dof,
        "full_dof_overfit_risk": full_dof_overfit_risk,
        "top1_sv_energy_frac": top1_sv,
        "top8_sv_energy_frac": top8_sv,
        "effective_spectral_rank": eff_spectral_rank,
        "grad_top1_sv_energy_frac": grad_top1_sv,
        "grad_top8_sv_energy_frac": grad_top8_sv,
        "grad_effective_spectral_rank": grad_eff_rank,
        "anisotropy_regime": anisotropy_regime,
        "median_channel_rms": median_rms,
        "max_channel_rms": max_rms,
        "max_outlier_ratio": max_outlier_ratio,
        "outlier_ratio_threshold": outlier_thresh,
        "num_outlier_channels": len(outlier_channels),
        "outlier_channel_indices": outlier_indices,
        "top_channels": top_channels,
        "signal_energy": signal_energy,
        "noise_energy": noise_energy,
        "probe_snr": probe_snr,
        "probe_snr_sufficient": probe_snr_sufficient,
        "recommended_max_rank": recommended_max_rank,
        "risk_tier": risk_tier,
        "recommended_mode": recommended_mode,
    }


def _project_onto_ker(tensor: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Apply exact two-pass Gram-Schmidt orthogonal projection onto ``ker(G_D0)``."""
    if v0.shape[0] == 0:
        return tensor
    out = tensor
    for _ in range(2):
        if out.ndim == 1:
            out = out - (v0.T @ (v0 @ out))
        else:
            out = out - ((out @ v0.T) @ v0)
    return out


@torch.no_grad()
def fit_james_stein_rank_k_fisher_norm_step(
    spec_chunk_grads: torch.Tensor,
    general_chunk_grads: torch.Tensor,
    *,
    max_rank: int = DEFAULT_MAX_RANK,
    ridge_scale: float = DEFAULT_RIDGE_SCALE,
    max_step_norm: float = DEFAULT_MAX_STEP_NORM,
    tok_grads_fisher_diag: torch.Tensor | None = None,
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL,
) -> dict[str, Any]:
    """Compute Empirical-Bayes James-Stein rank-``k`` natural-gradient step inside ``ker(G_D0)``.

    Projects specialist chunk gradients strictly inside ``ker(G_D0)``, extracts up to
    ``max_rank`` orthonormal variance modes via SVD, computes exact per-mode Empirical-Bayes
    shrinkage ``c_j = max(0, 1 - var_noise_j / (mean_signal_j^2 + var_noise_j / n))``,
    preconditions with damped diagonal Fisher, and re-projects onto ``ker(G_D0)`` so
    ``max(|G_D0 @ delta_w|) < 1e-6`` and ``||delta_w||_2 <= max_step_norm``.
    """
    spec_t = _validate_fp_tensor(spec_chunk_grads, "spec_chunk_grads", exact_ndim=2)
    gen_t = _validate_fp_tensor(general_chunk_grads, "general_chunk_grads", exact_ndim=2)
    dim = int(spec_t.shape[1])
    if int(gen_t.shape[1]) != dim:
        raise ValueError(
            f"general_chunk_grads width {gen_t.shape[1]} must match spec_chunk_grads width {dim}"
        )

    k_max = _validate_positive_int(max_rank, "max_rank")
    ridge_f = _validate_nonneg_finite(ridge_scale, "ridge_scale")
    step_cap = _validate_positive_finite(max_step_norm, "max_step_norm")
    sv_tol = _validate_positive_finite(nullspace_sv_tol, "nullspace_sv_tol")
    if sv_tol >= 1.0:
        raise ValueError(f"nullspace_sv_tol must be < 1.0, got {sv_tol}")

    if tok_grads_fisher_diag is not None:
        f_in = _validate_fp_tensor(tok_grads_fisher_diag, "tok_grads_fisher_diag", exact_ndim=1)
        if int(f_in.shape[0]) != dim:
            raise ValueError(f"tok_grads_fisher_diag length {f_in.shape[0]} must match dim={dim}")
        f_64 = f_in.detach().to(device="cpu", dtype=torch.float64)
        if bool((f_64 < 0.0).any().item()):
            raise ValueError("tok_grads_fisher_diag must contain only non-negative values")
    else:
        f_64 = None

    spec_64 = spec_t.detach().to(device="cpu", dtype=torch.float64)
    gen_64 = gen_t.detach().to(device="cpu", dtype=torch.float64)
    n_spec = int(spec_64.shape[0])
    n_gen = int(gen_64.shape[0])
    out_dtype = torch.promote_types(spec_t.dtype, gen_t.dtype)

    _, s0, vh0 = torch.linalg.svd(gen_64, full_matrices=False)
    if s0.numel() > 0 and float(s0[0].item()) > 1e-15:
        keep0 = int((s0 > sv_tol * float(s0[0].item())).sum().item())
        v0 = vh0[:keep0, :].contiguous()
    else:
        keep0 = 0
        v0 = torch.empty((0, dim), dtype=torch.float64)

    nullspace_dim = dim - keep0

    def _zero_result(reason: str) -> dict[str, Any]:
        zero_w = torch.zeros(dim, dtype=out_dtype)
        return {
            "delta_w": zero_w,
            "delta_norm": 0.0,
            "dim": dim,
            "n_spec_chunks": n_spec,
            "n_general_chunks": n_gen,
            "general_nullspace_rank_removed": keep0,
            "nullspace_dim": nullspace_dim,
            "effective_rank_k": 0,
            "max_rank": k_max,
            "ridge_scale": ridge_f,
            "max_step_norm": step_cap,
            "shrinkage_factors": [],
            "mean_shrinkage_factor": 0.0,
            "mode_signal_means": [],
            "mode_noise_vars": [],
            "mode_singular_values": [],
            "nullspace_orthogonality_max_abs": 0.0,
            "nullspace_orthogonality_l2": 0.0,
            "nullspace_orthogonal": True,
            "zero_step": True,
            "fallback_reason": reason,
        }

    if nullspace_dim <= 0:
        return _zero_result("empty_general_nullspace")

    spec_null = _project_onto_ker(spec_64, v0)
    _, s_spec, vh_spec = torch.linalg.svd(spec_null, full_matrices=False)
    s_top = float(s_spec[0].item()) if s_spec.numel() > 0 else 0.0
    if s_top <= 1e-15:
        return _zero_result("zero_specialist_nullspace_gradient")

    k_avail = int((s_spec > 1e-12 * max(1.0, s_top)).sum().item())
    k_eff = min(k_max, k_avail, nullspace_dim)
    if k_eff == 0:
        return _zero_result("no_active_nullspace_modes")

    q_raw = _project_onto_ker(vh_spec[:k_eff, :], v0).T
    q_basis, _ = torch.linalg.qr(q_raw, mode="reduced")
    q_basis = _project_onto_ker(q_basis.T, v0).T
    q_basis, _ = torch.linalg.qr(q_basis, mode="reduced")

    alpha = spec_null @ q_basis
    shrinkage_factors: list[float] = []
    mode_signal_means: list[float] = []
    mode_noise_vars: list[float] = []
    mode_singular_values = [float(s_spec[j].item()) for j in range(k_eff)]

    for j in range(k_eff):
        col = alpha[:, j]
        mean_j = float(col.mean().item())
        mode_signal_means.append(mean_j)
        if n_spec < 2:
            mode_noise_vars.append(0.0)
            shrinkage_factors.append(0.0)
            continue
        var_j = float(col.var(unbiased=True).item())
        mode_noise_vars.append(var_j)
        if var_j <= 1e-30:
            c_j = 1.0 if abs(mean_j) > 1e-15 else 0.0
        else:
            denom_j = (mean_j * mean_j) + (var_j / float(n_spec))
            c_j = max(0.0, 1.0 - (var_j / max(denom_j, 1e-30)))
        shrinkage_factors.append(float(c_j))

    mean_shrinkage = sum(shrinkage_factors) / float(len(shrinkage_factors))
    c_vec = torch.tensor(shrinkage_factors, dtype=torch.float64)
    mean_vec = torch.tensor(mode_signal_means, dtype=torch.float64)
    shrunk_modal = c_vec * mean_vec

    if f_64 is None:
        all_64 = torch.cat([spec_64, gen_64], dim=0)
        f_diag = all_64.pow(2).mean(dim=0) + max(ridge_f, 1e-12)
    else:
        f_diag = f_64 + max(ridge_f, 1e-12)

    # Precondition strictly inside span(q_basis) so low-Fisher tail noise outside rank-k never leaks in
    fisher_sub = (q_basis.T * f_diag.unsqueeze(0)) @ q_basis
    fisher_sub = 0.5 * (fisher_sub + fisher_sub.T)
    f_evals, f_evecs = torch.linalg.eigh(fisher_sub)
    inv_sqrt_evals = f_evals.clamp_min(1e-12).rsqrt()
    nat_modal = f_evecs @ (inv_sqrt_evals * (f_evecs.T @ shrunk_modal))
    nat_step = q_basis @ nat_modal
    delta_64 = -_project_onto_ker(nat_step, v0)
    raw_norm = float(delta_64.norm().item())
    if raw_norm > step_cap:
        delta_64 = _project_onto_ker(delta_64 * (step_cap / raw_norm), v0)

    delta_norm = float(delta_64.norm().item())
    zero_step = bool(delta_norm <= 1e-15)
    delta_out = (torch.zeros(dim, dtype=out_dtype) if zero_step else delta_64.to(dtype=out_dtype)).contiguous()

    residual_vec = gen_64 @ delta_out.to(dtype=torch.float64)
    ortho_max = float(residual_vec.abs().max().item())
    ortho_l2 = float(residual_vec.norm().item())

    return {
        "delta_w": delta_out,
        "delta_norm": float(delta_out.to(dtype=torch.float64).norm().item()),
        "dim": dim,
        "n_spec_chunks": n_spec,
        "n_general_chunks": n_gen,
        "general_nullspace_rank_removed": keep0,
        "nullspace_dim": nullspace_dim,
        "effective_rank_k": k_eff,
        "max_rank": k_max,
        "ridge_scale": ridge_f,
        "max_step_norm": step_cap,
        "shrinkage_factors": shrinkage_factors,
        "mean_shrinkage_factor": float(mean_shrinkage),
        "mode_signal_means": mode_signal_means,
        "mode_noise_vars": mode_noise_vars,
        "mode_singular_values": mode_singular_values,
        "nullspace_orthogonality_max_abs": ortho_max,
        "nullspace_orthogonality_l2": ortho_l2,
        "nullspace_orthogonal": bool(ortho_max < 1e-6),
        "zero_step": zero_step,
        "fallback_reason": "insufficient_probe_snr" if zero_step else "none",
    }


# ---------------------------------------------------------------------------
# 3. Model Resolution, Chunk Preparation & Verified Surgery Pipeline
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
    """Locate final norm and unembedding head modules across causal and MLM architectures."""
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
        raise ValueError(
            f"model.{head_name}.weight width {head_w.shape[1]} must match norm dim {hidden_dim}"
        )
    head_b = getattr(head_mod, "bias", None)
    if head_b is not None:
        hb_t = _validate_fp_tensor(head_b, f"model.{head_name}.bias", exact_ndim=1)
        if int(hb_t.shape[0]) != int(head_w.shape[0]):
            raise ValueError(f"model.{head_name}.bias length must match vocab size {head_w.shape[0]}")

    vocab_size = int(head_w.shape[0])
    return norm_mod, head_mod, resolved_norm, norm_name, head_name, hidden_dim, vocab_size, eps_eff


def _is_general_domain(domain_label: str, general_prefixes: tuple[str, ...]) -> bool:
    """Return True if ``domain_label`` starts with any configured general-prose prefix."""
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
    """Run trunk forward pass via read-only pre-hook on ``norm_mod`` to extract pre-norm states."""
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
    if bool(valid.any().item()):
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
        dm = compute_token_nll_and_ece(
            logits[ix], targets_2d[ix], num_bins=num_bins, ignore_index=ignore_index
        )
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
    cfg: JamesSteinNormSurgeonConfig,
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
            cfg.maximum_pain_energy
            + max(0.0, mean_diff)
            + cfg.ece_energy_weight * float(cand_ece)
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


def _search_probe_candidate(
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
    cfg: JamesSteinNormSurgeonConfig,
) -> dict[str, Any]:
    """Tune step multiplier ``eta`` and folded temperature ``tau`` strictly on ``fit`` split."""
    etas = (0.0,) if delta_w is None else tuple(eta_grid)
    dw64 = torch.zeros_like(w0, dtype=torch.float64) if delta_w is None else delta_w.to(dtype=torch.float64)

    best: dict[str, Any] = {
        "found": False,
        "eta": 0.0,
        "tau": 1.0,
        "w": w0.clone(),
        "b": b0.clone() if b0 is not None else None,
        "head_b": head_b.clone() if head_b is not None else None,
        "probe_energy": 0.0,
    }

    for eta in etas:
        w_unfolded = w0 + float(eta) * dw64
        for tau in cfg.tau_grid:
            if abs(float(eta)) <= 1e-15 and abs(float(tau) - 1.0) <= 1e-12:
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
    model.load_state_dict(
        {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()}
    )


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


def _apply_candidate_params(
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


def _resolve_config(
    config: JamesSteinNormSurgeonConfig | None,
    *,
    score_from: int | None,
    max_rank: int | None,
    ridge_scale: float | None,
    max_step_norm: float | None,
    max_general_drift_rel: float | None,
    max_domain_drift_rel: float | None,
) -> JamesSteinNormSurgeonConfig:
    """Merge explicit keyword overrides into a validated configuration."""
    if config is not None and not isinstance(config, JamesSteinNormSurgeonConfig):
        raise TypeError(f"config must be JamesSteinNormSurgeonConfig or None, got {type(config).__name__}")
    base = config or JamesSteinNormSurgeonConfig()
    return JamesSteinNormSurgeonConfig(
        min_chunks_per_dof=base.min_chunks_per_dof,
        max_rank=base.max_rank if max_rank is None else _validate_positive_int(max_rank, "max_rank"),
        ridge_scale=base.ridge_scale if ridge_scale is None else _validate_nonneg_finite(ridge_scale, "ridge_scale"),
        max_step_norm=base.max_step_norm if max_step_norm is None else _validate_positive_finite(max_step_norm, "max_step_norm"),
        nullspace_sv_tol=base.nullspace_sv_tol,
        max_general_drift_rel=(
            base.max_general_drift_rel
            if max_general_drift_rel is None
            else _validate_nonneg_finite(max_general_drift_rel, "max_general_drift_rel")
        ),
        max_domain_drift_rel=(
            base.max_domain_drift_rel
            if max_domain_drift_rel is None
            else _validate_nonneg_finite(max_domain_drift_rel, "max_domain_drift_rel")
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        soft_margin_penalty=base.soft_margin_penalty,
        ece_energy_weight=base.ece_energy_weight,
        maximum_pain_energy=base.maximum_pain_energy,
        min_energy_drop=base.min_energy_drop,
        min_probe_snr=base.min_probe_snr,
        min_shrinkage_to_activate=base.min_shrinkage_to_activate,
        outlier_ratio_threshold=base.outlier_ratio_threshold,
        num_bins=base.num_bins,
        ignore_index=base.ignore_index,
        score_from=base.score_from if score_from is None else _validate_nonneg_int(score_from, "score_from"),
        batch_size=base.batch_size,
        tau_grid=base.tau_grid,
        eta_grid=base.eta_grid,
        general_prefixes=base.general_prefixes,
        require_strict_energy_drop=base.require_strict_energy_drop,
    )


@torch.no_grad()
def execute_verified_james_stein_norm_surgery(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    config: JamesSteinNormSurgeonConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    score_from: int | None = None,
    max_rank: int | None = None,
    ridge_scale: float | None = None,
    max_step_norm: float | None = None,
    max_general_drift_rel: float | None = None,
    max_domain_drift_rel: float | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute verified James-Stein rank-``k`` final-norm surgery with scalar fallback & ANSE rollback.

    Compares 1-DOF folded-tau scalar calibration against Empirical-Bayes James-Stein rank-``k``
    ``ker(G_D0)`` norm surgery on held-out ``selection_chunks`` under ``ANSE Delta E < 0`` and
    ``D0 <= +1.0%``. Automatically falls back to 1-DOF scalar when probe SNR is insufficient or
    shrinkage contracts toward zero, and enforces atomic bitwise rollback on guard/energy failure.
    """
    cfg = _resolve_config(
        config,
        score_from=score_from,
        max_rank=max_rank,
        ridge_scale=ridge_scale,
        max_step_norm=max_step_norm,
        max_general_drift_rel=max_general_drift_rel,
        max_domain_drift_rel=max_domain_drift_rel,
    )
    _validate_bool(inplace, "inplace")

    _, _, _, norm_name_chk, head_name_chk, hidden_dim, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )

    if selection_chunks is None:
        if isinstance(probe_chunks, (str, bytes, bytearray)) or not isinstance(probe_chunks, Sequence):
            raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")
        fit_raw, sel_raw = _split_probe_chunks_fallback(probe_chunks, cfg.general_prefixes)
    else:
        fit_raw = probe_chunks
        sel_raw = selection_chunks

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
        _, u_ver = (
            (None, u_sel)
            if eval_chunks is None
            else _extract_scored_unit_states(
                h_ver_raw, ver_prep, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from
            )
        )

        chunk_grads_fit, tok_fisher = _compute_chunk_norm_gradients(
            u_fit,
            fit_prep["targets_2d"],
            w0,
            b0,
            head_w,
            head_b,
            ignore_index=cfg.ignore_index,
        )
        fit_gen_mask = fit_prep["is_general"]
        gen_grads = chunk_grads_fit[fit_gen_mask]
        spec_grads = chunk_grads_fit[~fit_gen_mask]

        probe_diag = compute_effective_probe_dof_and_overfit_risk(
            chunk_grads_fit,
            hidden_dim,
            min_chunks_per_dof=cfg.min_chunks_per_dof,
            raw_activations=h_fit_scored,
            outlier_ratio_threshold=cfg.outlier_ratio_threshold,
        )
        js_fit = fit_james_stein_rank_k_fisher_norm_step(
            spec_grads,
            gen_grads,
            max_rank=cfg.max_rank,
            ridge_scale=cfg.ridge_scale,
            max_step_norm=cfg.max_step_norm,
            tok_grads_fisher_diag=tok_fisher,
            nullspace_sv_tol=cfg.nullspace_sv_tol,
        )

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
        scalar_cand = _search_probe_candidate(
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

        probe_snr_ok = bool(
            probe_diag["probe_snr"] >= cfg.min_probe_snr
            and not js_fit["zero_step"]
            and js_fit["mean_shrinkage_factor"] >= cfg.min_shrinkage_to_activate
        )
        if probe_snr_ok:
            js_cand = _search_probe_candidate(
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
        else:
            js_cand = {
                "found": False,
                "eta": 0.0,
                "tau": 1.0,
                "w": w0.clone(),
                "b": b0.clone() if b0 is not None else None,
                "head_b": head_b.clone() if head_b is not None else None,
                "probe_energy": 0.0,
            }

        # Held-out selection comparison
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

        sel_scalar_dom, sel_scalar_g = _eval_norm_metrics(
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
        sel_scalar_anse = _compute_anse_verdict(
            sel_base_dom, sel_base_g["ece_value"], sel_scalar_dom, sel_scalar_g["ece_value"], cfg=cfg
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

        # Unregularized full-DOF diagnostic control (PILOT9/PILOT12 Arm 3 replica, never deployed)
        _, s0_g, vh0_g = torch.linalg.svd(gen_grads, full_matrices=False)
        keep0_g = (
            int((s0_g > cfg.nullspace_sv_tol * float(s0_g[0].item())).sum().item())
            if s0_g.numel() > 0 and float(s0_g[0].item()) > 1e-15
            else 0
        )
        v0_g = vh0_g[:keep0_g, :]
        full_nat = _project_onto_ker(
            spec_grads.mean(dim=0) / (tok_fisher + max(cfg.ridge_scale, 1e-12)).sqrt(), v0_g
        )
        full_l2 = float(full_nat.norm().item())
        full_step = (
            _project_onto_ker(full_nat * (cfg.max_step_norm / full_l2), v0_g)
            if full_l2 > cfg.max_step_norm
            else full_nat
        )
        full_cand = _search_probe_candidate(
            u_fit,
            fit_prep["targets_2d"],
            fit_prep["domains"],
            w0,
            b0,
            head_w,
            head_b,
            fit_base_dom,
            fit_base_g["ece_value"],
            delta_w=-full_step,
            eta_grid=cfg.eta_grid,
            cfg=cfg,
        )
        sel_full_dom, sel_full_g = _eval_norm_metrics(
            u_sel,
            sel_prep["targets_2d"],
            sel_prep["domains"],
            full_cand["w"],
            full_cand["b"],
            head_w,
            full_cand["head_b"],
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        sel_full_anse = _compute_anse_verdict(
            sel_base_dom, sel_base_g["ece_value"], sel_full_dom, sel_full_g["ece_value"], cfg=cfg
        )

        def _is_energy_improving(verdict: Mapping[str, Any]) -> bool:
            if cfg.require_strict_energy_drop:
                return float(verdict["delta_e"]) < -cfg.min_energy_drop
            return float(verdict["delta_e"]) <= 0.0

        scalar_sel_ok = bool(scalar_cand["found"] and sel_scalar_anse["guard_ok"] and _is_energy_improving(sel_scalar_anse))
        js_sel_ok = bool(js_cand["found"] and sel_js_anse["guard_ok"] and _is_energy_improving(sel_js_anse))

        if js_sel_ok and (not scalar_sel_ok or float(sel_js_anse["anse_energy"]) < float(sel_scalar_anse["anse_energy"]) - 1e-12):
            chosen_arm = "james_stein_rank_k"
            chosen_cand = js_cand
            fallback_triggered = False
            fallback_reason = "none"
        elif scalar_sel_ok:
            chosen_arm = "scalar_folded_tau"
            chosen_cand = scalar_cand
            fallback_triggered = True
            fallback_reason = (
                "insufficient_probe_snr_or_zero_shrinkage"
                if not probe_snr_ok
                else "scalar_beats_or_replaces_james_stein_on_selection"
            )
        else:
            chosen_arm = "base_rollback"
            chosen_cand = {
                "eta": 0.0,
                "tau": 1.0,
                "w": w0.clone(),
                "b": b0.clone() if b0 is not None else None,
                "head_b": head_b.clone() if head_b is not None else None,
            }
            fallback_triggered = True
            fallback_reason = "no_feasible_selection_candidate"

        _apply_candidate_params(
            norm_mod, head_mod, chosen_cand["w"], chosen_cand["b"], chosen_cand["head_b"]
        )
        cand_snapshot = _snapshot_state(work_model)
        cand_sha = compute_module_state_sha256(work_model)

        ver_base_dom, ver_base_g = _eval_norm_metrics(
            u_ver,
            ver_prep["targets_2d"],
            ver_prep["domains"],
            w0,
            b0,
            head_w,
            head_b,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        ver_cand_dom, ver_cand_g = _eval_norm_metrics(
            u_ver,
            ver_prep["targets_2d"],
            ver_prep["domains"],
            chosen_cand["w"],
            chosen_cand["b"],
            head_w,
            chosen_cand["head_b"],
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        ver_anse = _compute_anse_verdict(
            ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
        )

        ver_improved = _is_energy_improving(ver_anse)
        if cand_sha == initial_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED" if ver_anse["guard_ok"] else "REJECTED_GUARD_VIOLATION_UNCHANGED"
        elif ver_anse["guard_ok"] and ver_improved:
            accepted = True
            rolled_back = False
            status = (
                "ACCEPTED_JAMES_STEIN_RANK_K"
                if chosen_arm == "james_stein_rank_k"
                else "ACCEPTED_SCALAR_FALLBACK"
            )
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
            "general_pct_vs_base": 0.0,
            "worst_domain_id": str(ver_anse["worst_domain_id"]),
            "worst_domain_pct_vs_base": 0.0,
            "delta_e": 0.0,
        }
    )
    edited_params = _count_edited_params(initial_snapshot, cand_snapshot) if accepted else 0
    chosen_delta_norm = float(chosen_cand["eta"]) * float(js_fit["delta_norm"]) if chosen_arm == "james_stein_rank_k" else 0.0

    receipt: dict[str, Any] = {
        "object": SURGEON_RECEIPT_OBJECT,
        "schema": SURGEON_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "selected_arm": chosen_arm if accepted else "base_rollback",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "norm_kind": resolved_norm,
        "norm_attr": norm_name,
        "head_attr": head_name,
        "nll_mean": float(active_g["nll_mean"]),
        "ece_value": float(active_g["ece_value"]),
        "base_nll_mean": float(ver_base_g["nll_mean"]),
        "base_ece_value": float(ver_base_g["ece_value"]),
        "cand_nll_mean": float(ver_cand_g["nll_mean"]),
        "cand_ece_value": float(ver_cand_g["ece_value"]),
        "shrinkage_factor": float(js_fit["mean_shrinkage_factor"]),
        "shrinkage_factors": list(js_fit["shrinkage_factors"]),
        "effective_rank_k": int(js_fit["effective_rank_k"]),
        "tau_temp": float(chosen_cand["tau"]),
        "eta_scale": float(chosen_cand["eta"]),
        "nullspace_delta_norm": chosen_delta_norm,
        "nullspace_orthogonality_max_abs": float(js_fit["nullspace_orthogonality_max_abs"]),
        "general_drift_ratio": float(active_anse["general_drift_ratio"]),
        "general_pct_vs_base": float(active_anse["general_pct_vs_base"]),
        "worst_domain_id": str(active_anse["worst_domain_id"]),
        "worst_domain_pct_vs_base": float(active_anse["worst_domain_pct_vs_base"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "cand_anse_energy": float(ver_anse["anse_energy"]),
        "delta_e": float(active_anse["delta_e"]),
        "guard_ok": bool(ver_anse["guard_ok"]),
        "probe_overfit_diagnostics": probe_diag,
        "james_stein_fit": {k: v for k, v in js_fit.items() if k != "delta_w"},
        "selection_comparison": {
            "base": {"nll_mean": float(sel_base_g["nll_mean"]), "ece_value": float(sel_base_g["ece_value"])},
            "scalar_folded_tau": {
                "nll_mean": float(sel_scalar_g["nll_mean"]),
                "ece_value": float(sel_scalar_g["ece_value"]),
                "anse_energy": float(sel_scalar_anse["anse_energy"]),
                "general_drift_ratio": float(sel_scalar_anse["general_drift_ratio"]),
                "guard_ok": bool(sel_scalar_anse["guard_ok"]),
                "tau_temp": float(scalar_cand["tau"]),
            },
            "james_stein_rank_k": {
                "nll_mean": float(sel_js_g["nll_mean"]),
                "ece_value": float(sel_js_g["ece_value"]),
                "anse_energy": float(sel_js_anse["anse_energy"]),
                "general_drift_ratio": float(sel_js_anse["general_drift_ratio"]),
                "guard_ok": bool(sel_js_anse["guard_ok"]),
                "tau_temp": float(js_cand["tau"]),
                "eta_scale": float(js_cand["eta"]),
            },
            "full_dof_control": {
                "nll_mean": float(sel_full_g["nll_mean"]),
                "ece_value": float(sel_full_g["ece_value"]),
                "anse_energy": float(sel_full_anse["anse_energy"]),
                "general_drift_ratio": float(sel_full_anse["general_drift_ratio"]),
                "guard_ok": bool(sel_full_anse["guard_ok"]),
            },
        },
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
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MAX_RANK",
    "DEFAULT_MAX_STEP_NORM",
    "DEFAULT_MIN_CHUNKS_PER_DOF",
    "DEFAULT_MIN_ENERGY_DROP",
    "DEFAULT_MIN_PROBE_SNR",
    "DEFAULT_MIN_SHRINKAGE_TO_ACTIVATE",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_NUM_BINS",
    "DEFAULT_OUTLIER_RATIO_THRESHOLD",
    "DEFAULT_RIDGE_SCALE",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_TAU_GRID",
    "SURGEON_RECEIPT_OBJECT",
    "SURGEON_SCHEMA",
    "VALID_NORM_KINDS",
    "JamesSteinNormSurgeonConfig",
    "compute_effective_probe_dof_and_overfit_risk",
    "execute_verified_james_stein_norm_surgery",
    "fit_james_stein_rank_k_fisher_norm_step",
    "verify_james_stein_norm_receipt_provenance",
]
