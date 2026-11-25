# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hidden-State Isotropy, RankMe & Anisotropy Cone Guard for Open-Weight WSP Surgery.

Implements a training-free, CPU-exact ``float64`` representation-collapse detector and
closed-form null-space restorer inspired by Ethayarajh (EMNLP 2019) contextual anisotropy
and Garrido et al. (ICML 2023) RankMe effective dimensionality:

1. ``measure_hidden_isotropy_and_rankme(hidden_states, *, eps=1e-8)``:
   Computes spectral RankMe entropy effective rank ``exp(-sum p_i log p_i)``,
   participation ratio ``(sum lambda_i)^2 / sum lambda_i^2``, Ethayarajh mean pairwise
   cosine anisotropy and ``isotropy_index``, mean-to-RMS cone ratio, centered
   diagnostic rank/PR, and explicit collapse warning flags.
2. ``synthesize_anisotropy_centering_and_nullspace_repair(out_weight_or_ln, hidden_states_pre, hidden_states_post, p_null, *, min_rankme_ratio=0.85, max_anisotropy_delta=0.05)``:
   Folds closed-form mean-cone subtraction and singular-floor restoration strictly inside
   the protected prose null-space ``P_0`` into layer output projections (``nn.Linear`` /
   2D weight) or final normalization layers (``nn.LayerNorm`` / ``nn.RMSNorm`` / 1D norm).
3. ``execute_verified_collapse_guarded_wsp_surgery(...)``:
   Detects silent representation collapse on candidate MoE / WSP edits, attempts
   closed-form ``P_0``-preserving repair when configured, and gates commit on
   representation recovery, ANSE ``Delta E < 0``, and general-domain drift ``<= +1.0%``
   with atomic bitwise rollback (``ANSE_MAXIMUM_PAIN`` on unrecoverable rank collapse)
   and canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

GUARD_SCHEMA: str = "hivemind.representation_collapse_wsp_guard/1"
GUARD_RECEIPT_OBJECT: str = "hivemind.representation_collapse_wsp_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_EPS: float = 1e-8
DEFAULT_MIN_RANKME_RATIO: float = 0.85
DEFAULT_MAX_ANISOTROPY_DELTA: float = 0.05
DEFAULT_MAX_ANISOTROPY_ABS: float = 0.85
DEFAULT_MIN_NORMALIZED_RANKME: float = 0.15
DEFAULT_SINGULAR_FLOOR_FRAC: float = 0.25
DEFAULT_MAX_REPAIR_GAIN: float = 25.0
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Provenance & Configuration
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
    """Validate that ``value`` is finite and >= 0.0."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is finite and > 0.0."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = False,
    allow_one: bool = True,
) -> float:
    """Validate a finite real number inside ``[0, 1]`` with configurable endpoints."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def _validate_hidden_states(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty >=2D finite floating-point tensor and flatten to CPU ``float64`` ``(N, d)``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2 or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty tensor with ndim >= 2, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    flat = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, int(tensor.shape[-1]))
        .contiguous()
    )
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat


def _validate_square_projector(proj: Any, expected_dim: int, name: str) -> torch.Tensor:
    """Validate a square symmetric PSD projection/filter matrix in CPU ``float64``."""
    if not isinstance(proj, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(proj).__name__}")
    if proj.ndim != 2 or proj.shape[0] != proj.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {tuple(proj.shape)}")
    if not proj.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {proj.dtype}")
    if int(proj.shape[0]) != expected_dim:
        raise ValueError(
            f"{name} shape {tuple(proj.shape)} must match hidden dimension ({expected_dim}, {expected_dim})"
        )
    mat = proj.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    sym = (0.5 * (mat + mat.T)).contiguous()
    evals = torch.linalg.eigvalsh(sym)
    ev_min = float(evals[0].item())
    ev_max = float(evals[-1].item())
    if ev_min < -1e-4 or ev_max > 1.0 + 1e-4:
        raise ValueError(
            f"{name} eigenvalues must lie inside [0, 1], got range [{ev_min:.3e}, {ev_max:.3e}]"
        )
    return sym


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically in UTF-8 JSON without NaN or Inf."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a tensor mapping."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty Mapping[str, torch.Tensor]")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"State key must be a non-empty string, got {key!r}")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        cpu_t = tensor.detach().to(device="cpu").contiguous()
        if cpu_t.is_floating_point() and not bool(torch.isfinite(cpu_t).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        digest.update(cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_collapse_guard_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != GUARD_RECEIPT_OBJECT or receipt.get("schema") != GUARD_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class CollapseGuardConfig:
    """Validated configuration for representation-collapse detection, repair, and ANSE gating."""

    min_rankme_ratio: float = DEFAULT_MIN_RANKME_RATIO
    max_anisotropy_delta: float = DEFAULT_MAX_ANISOTROPY_DELTA
    max_anisotropy_abs: float = DEFAULT_MAX_ANISOTROPY_ABS
    min_normalized_rankme: float = DEFAULT_MIN_NORMALIZED_RANKME
    eps: float = DEFAULT_EPS
    singular_floor_frac: float = DEFAULT_SINGULAR_FLOOR_FRAC
    max_repair_gain: float = DEFAULT_MAX_REPAIR_GAIN
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    general_guard_max_ratio: float | None = None
    enforce_all_domain_guard: bool = True
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    require_energy_improvement: bool = True
    attempt_closed_form_repair: bool = True
    require_repair_clear_collapse: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.min_rankme_ratio, "min_rankme_ratio", allow_zero=False, allow_one=True)
        _validate_nonneg_finite(self.max_anisotropy_delta, "max_anisotropy_delta")
        _validate_unit_interval(self.max_anisotropy_abs, "max_anisotropy_abs", allow_zero=False, allow_one=True)
        _validate_unit_interval(
            self.min_normalized_rankme, "min_normalized_rankme", allow_zero=False, allow_one=True
        )
        _validate_positive_finite(self.eps, "eps")
        _validate_unit_interval(
            self.singular_floor_frac, "singular_floor_frac", allow_zero=False, allow_one=True
        )
        max_g = _validate_positive_finite(self.max_repair_gain, "max_repair_gain")
        if max_g < 1.0:
            raise ValueError(f"max_repair_gain must be >= 1.0, got {max_g}")
        guard_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_r < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {guard_r}")
        if self.general_guard_max_ratio is not None:
            gen_r = _validate_positive_finite(self.general_guard_max_ratio, "general_guard_max_ratio")
            if gen_r < 1.0:
                raise ValueError(f"general_guard_max_ratio must be >= 1.0, got {gen_r}")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        safe_r = _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_e0_ratio must be >= 1.0, got {safe_r}")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.require_energy_improvement, "require_energy_improvement")
        _validate_bool(self.attempt_closed_form_repair, "attempt_closed_form_repair")
        _validate_bool(self.require_repair_clear_collapse, "require_repair_clear_collapse")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Hidden-State Isotropy, Participation-Ratio & RankMe Diagnostics
# ---------------------------------------------------------------------------


def _spectral_rankme_and_pr(mat_2d: torch.Tensor, eps_f: float) -> dict[str, float]:
    """Compute RankMe effective rank, participation ratio, and singular diagnostics on ``(N, d)``."""
    n_tok = int(mat_2d.shape[0])
    sv = torch.linalg.svdvals(mat_2d)
    sigma_sum = float(sv.sum().item())
    sigma_max = float(sv[0].item())
    sigma_min = float(sv[-1].item())

    if sigma_sum <= eps_f or sigma_max <= eps_f:
        return {
            "rankme": 0.0,
            "participation_ratio": 0.0,
            "spectral_entropy": 0.0,
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "condition_number": 1.0,
        }

    probs = sv / sigma_sum
    pos_probs = probs[probs > 1e-15]
    entropy = max(0.0, float(-(pos_probs * torch.log(pos_probs)).sum().item()))
    rankme = float(math.exp(entropy))

    lambdas = (sv * sv) / float(n_tok)
    lam_sum = float(lambdas.sum().item())
    lam_sq_sum = float((lambdas * lambdas).sum().item())
    participation_ratio = float((lam_sum * lam_sum) / max(lam_sq_sum, eps_f * eps_f))
    condition_number = float(sigma_max / max(sigma_min, eps_f))
    return {
        "rankme": rankme,
        "participation_ratio": participation_ratio,
        "spectral_entropy": entropy,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": condition_number,
    }


@torch.no_grad()
def measure_hidden_isotropy_and_rankme(
    hidden_states: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    min_normalized_rankme: float = DEFAULT_MIN_NORMALIZED_RANKME,
    max_anisotropy: float = DEFAULT_MAX_ANISOTROPY_ABS,
    min_rankme_ratio: float = DEFAULT_MIN_RANKME_RATIO,
    max_anisotropy_delta: float = DEFAULT_MAX_ANISOTROPY_DELTA,
    reference_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure hidden-state RankMe effective rank, participation ratio, and cosine anisotropy.

    Follows Garrido et al. (ICML 2023) for spectral RankMe ``exp(-sum p_i log p_i)`` with
    ``p_i = sigma_i / sum sigma_j``, neural-geometry participation ratio
    ``(sum lambda_i)^2 / sum lambda_i^2``, and Ethayarajh (2019) mean pairwise cosine
    anisotropy across distinct token pairs.
    """
    h2d = _validate_hidden_states(hidden_states, "hidden_states")
    eps_f = _validate_positive_finite(eps, "eps")
    min_norm_rm = _validate_unit_interval(
        min_normalized_rankme, "min_normalized_rankme", allow_zero=False, allow_one=True
    )
    max_aniso = _validate_unit_interval(max_anisotropy, "max_anisotropy", allow_zero=False, allow_one=True)
    min_rm_ratio = _validate_unit_interval(
        min_rankme_ratio, "min_rankme_ratio", allow_zero=False, allow_one=True
    )
    max_aniso_d = _validate_nonneg_finite(max_anisotropy_delta, "max_anisotropy_delta")
    if reference_metrics is not None and not isinstance(reference_metrics, Mapping):
        raise TypeError("reference_metrics must be a Mapping or None")

    n_tok, hidden_dim = int(h2d.shape[0]), int(h2d.shape[1])
    min_dim = min(n_tok, hidden_dim)

    norms = torch.linalg.vector_norm(h2d, dim=1)
    nz_mask = norms > eps_f
    n_nz = int(nz_mask.sum().item())
    nonzero_fraction = float(n_nz) / float(n_tok)

    if n_nz == 0 or n_tok < 2:
        anisotropy = 1.0
        isotropy_index = 0.0
    else:
        unit_h = torch.zeros_like(h2d)
        unit_h[nz_mask] = h2d[nz_mask] / norms[nz_mask].unsqueeze(1)
        u_sum = unit_h.sum(dim=0)
        pair_sum = float(torch.dot(u_sum, u_sum).item()) - float(n_nz)
        anisotropy = max(-1.0, min(1.0, pair_sum / float(n_tok * (n_tok - 1))))
        isotropy_index = max(0.0, min(1.0, 1.0 - max(0.0, anisotropy)))

    raw_spec = _spectral_rankme_and_pr(h2d, eps_f)
    mean_vec = h2d.mean(dim=0)
    h_centered = h2d - mean_vec.unsqueeze(0)
    cen_spec = _spectral_rankme_and_pr(h_centered, eps_f)

    mean_norm = float(torch.linalg.vector_norm(mean_vec).item())
    rms_norm = float(norms.pow(2).mean().sqrt().item())
    mean_to_rms_ratio = float(mean_norm / max(rms_norm, eps_f)) if rms_norm > eps_f else 0.0

    rankme = raw_spec["rankme"]
    pr = raw_spec["participation_ratio"]
    normalized_rankme = rankme / float(min_dim)
    normalized_pr = pr / float(min_dim)
    cen_denom = float(max(1, min(max(1, n_tok - 1), hidden_dim)))

    warnings_list: list[str] = []
    is_zero_deg = raw_spec["sigma_max"] <= eps_f or n_nz == 0
    if is_zero_deg:
        warnings_list.append("zero_or_degenerate_hidden_states")
    else:
        if anisotropy >= max_aniso:
            warnings_list.append("high_anisotropy_cone")
        if reference_metrics is None and min_dim > 1 and normalized_rankme < min_norm_rm:
            warnings_list.append("low_effective_rankme")
        if min_dim > 1 and rankme <= 1.25:
            warnings_list.append("near_rank1_collapse")

    rankme_ratio_vs_ref: float | None = None
    anisotropy_delta_vs_ref: float | None = None
    if reference_metrics is not None:
        if "rankme" not in reference_metrics:
            raise ValueError("reference_metrics must contain 'rankme'")
        ref_rm = _validate_nonneg_finite(reference_metrics["rankme"], "reference_metrics['rankme']")
        ref_aniso_raw = reference_metrics.get(
            "anisotropy", reference_metrics.get("mean_pairwise_cosine_anisotropy")
        )
        if ref_aniso_raw is None:
            raise ValueError(
                "reference_metrics must contain 'anisotropy' or 'mean_pairwise_cosine_anisotropy'"
            )
        ref_aniso = _validate_finite_float(ref_aniso_raw, "reference_metrics['anisotropy']")
        rankme_ratio_vs_ref = float(rankme / max(ref_rm, eps_f))
        anisotropy_delta_vs_ref = float(anisotropy - ref_aniso)
        if rankme_ratio_vs_ref < min_rm_ratio:
            warnings_list.append("rankme_drop_vs_reference")
        if anisotropy_delta_vs_ref > max_aniso_d:
            warnings_list.append("anisotropy_spike_vs_reference")

    dedup_warnings = list(dict.fromkeys(warnings_list))
    collapse_warning = len(dedup_warnings) > 0

    return {
        "n_tokens": n_tok,
        "hidden_dim": hidden_dim,
        "min_dim": min_dim,
        "rankme": float(rankme),
        "normalized_rankme": float(normalized_rankme),
        "centered_rankme": float(cen_spec["rankme"]),
        "centered_normalized_rankme": float(cen_spec["rankme"] / cen_denom),
        "spectral_entropy": float(raw_spec["spectral_entropy"]),
        "participation_ratio": float(pr),
        "normalized_participation_ratio": float(normalized_pr),
        "centered_participation_ratio": float(cen_spec["participation_ratio"]),
        "mean_pairwise_cosine_anisotropy": float(anisotropy),
        "anisotropy": float(anisotropy),
        "isotropy_index": float(isotropy_index),
        "mean_norm": float(mean_norm),
        "rms_norm": float(rms_norm),
        "mean_to_rms_ratio": float(mean_to_rms_ratio),
        "sigma_max": float(raw_spec["sigma_max"]),
        "sigma_min": float(raw_spec["sigma_min"]),
        "condition_number": float(raw_spec["condition_number"]),
        "nonzero_token_fraction": float(nonzero_fraction),
        "rankme_ratio_vs_ref": rankme_ratio_vs_ref,
        "anisotropy_delta_vs_ref": anisotropy_delta_vs_ref,
        "collapse_warnings": dedup_warnings,
        "collapse_warning": collapse_warning,
        "is_collapsed": collapse_warning,
    }


# ---------------------------------------------------------------------------
# 3. Closed-Form Anisotropy Centering & Nullspace Singular-Floor Repair
# ---------------------------------------------------------------------------


def _extract_repair_target(
    out_weight_or_ln: Any,
    hidden_dim: int,
) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    """Classify and extract ``(kind, weight_tensor, bias_tensor_or_none)`` from ``out_weight_or_ln``."""
    if isinstance(out_weight_or_ln, nn.Linear):
        w = out_weight_or_ln.weight
        b = out_weight_or_ln.bias
        kind = "linear"
    elif isinstance(out_weight_or_ln, nn.Module):
        w_attr = getattr(out_weight_or_ln, "weight", None)
        if not isinstance(w_attr, torch.Tensor) or w_attr.ndim != 1:
            raise ValueError(
                f"Unsupported nn.Module {type(out_weight_or_ln).__name__}; expected nn.Linear or 1D norm module"
            )
        w = w_attr
        b_attr = getattr(out_weight_or_ln, "bias", None)
        b = b_attr if isinstance(b_attr, torch.Tensor) else None
        kind = "norm"
    elif isinstance(out_weight_or_ln, torch.Tensor):
        w = out_weight_or_ln
        b = None
        if w.ndim == 2:
            kind = "tensor_2d"
        elif w.ndim == 1:
            kind = "tensor_1d"
        else:
            raise ValueError(f"Tensor out_weight_or_ln must be 1D or 2D, got shape {tuple(w.shape)}")
    elif isinstance(out_weight_or_ln, Mapping):
        if "weight" not in out_weight_or_ln or not isinstance(out_weight_or_ln["weight"], torch.Tensor):
            raise ValueError("Mapping out_weight_or_ln must contain Tensor 'weight'")
        w = out_weight_or_ln["weight"]
        b_raw = out_weight_or_ln.get("bias")
        if b_raw is not None and not isinstance(b_raw, torch.Tensor):
            raise TypeError("Mapping out_weight_or_ln['bias'] must be a torch.Tensor or None")
        b = b_raw
        if w.ndim == 2:
            kind = "mapping_2d"
        elif w.ndim == 1:
            kind = "mapping_1d"
        else:
            raise ValueError(f"weight tensor must be 1D or 2D, got shape {tuple(w.shape)}")
    elif isinstance(out_weight_or_ln, tuple) and len(out_weight_or_ln) == 2:
        w, b = out_weight_or_ln
        if not isinstance(w, torch.Tensor):
            raise TypeError("Tuple out_weight_or_ln[0] must be a torch.Tensor")
        if b is not None and not isinstance(b, torch.Tensor):
            raise TypeError("Tuple out_weight_or_ln[1] must be a torch.Tensor or None")
        if w.ndim == 2:
            kind = "tuple_2d"
        elif w.ndim == 1:
            kind = "tuple_1d"
        else:
            raise ValueError(f"weight tensor must be 1D or 2D, got shape {tuple(w.shape)}")
    else:
        raise TypeError(
            "out_weight_or_ln must be nn.Linear, norm nn.Module, 1D/2D Tensor, Mapping, or (weight, bias) tuple"
        )

    if not w.is_floating_point() or any(d == 0 for d in w.shape):
        raise ValueError("weight tensor must be non-empty and floating-point")
    if not bool(torch.isfinite(w.detach()).all().item()):
        raise ValueError("weight tensor must contain only finite values")
    if int(w.shape[0]) != hidden_dim:
        raise ValueError(
            f"out_weight_or_ln output dimension {int(w.shape[0])} must match hidden_states dim {hidden_dim}"
        )
    if b is not None:
        if not b.is_floating_point() or tuple(b.shape) != (hidden_dim,):
            raise ValueError(f"bias tensor must be floating-point of shape ({hidden_dim},)")
        if not bool(torch.isfinite(b.detach()).all().item()):
            raise ValueError("bias tensor must contain only finite values")
    return kind, w, b


def _package_repaired_target(
    original: Any,
    kind: str,
    w_new: torch.Tensor,
    b_new: torch.Tensor | None,
    *,
    inplace: bool,
) -> Any:
    """Apply or return repaired weights matching ``original`` container type."""
    if kind in ("linear", "norm"):
        target_mod = original if inplace else copy.deepcopy(original)
        target_mod.weight.copy_(w_new.to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))
        if b_new is not None and getattr(target_mod, "bias", None) is not None:
            target_mod.bias.copy_(b_new.to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
        return target_mod
    if kind in ("tensor_2d", "tensor_1d"):
        if inplace:
            original.copy_(w_new.to(device=original.device, dtype=original.dtype))
            return original
        return w_new.to(device=original.device, dtype=original.dtype).clone()
    if kind in ("mapping_2d", "mapping_1d"):
        out = dict(original)
        out["weight"] = w_new.to(device=original["weight"].device, dtype=original["weight"].dtype).clone()
        if b_new is not None and original.get("bias") is not None:
            out["bias"] = b_new.to(device=original["bias"].device, dtype=original["bias"].dtype).clone()
        return out
    orig_w, orig_b = original
    ret_w = w_new.to(device=orig_w.device, dtype=orig_w.dtype).clone()
    ret_b = b_new.to(device=orig_b.device, dtype=orig_b.dtype).clone() if (b_new is not None and orig_b is not None) else None
    return (ret_w, ret_b)


def _solve_2d_nullspace_repair(
    w64: torch.Tensor,
    b64: torch.Tensor | None,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    p0: torch.Tensor,
    *,
    eps_f: float,
    singular_floor_frac: float,
    max_repair_gain: float,
    min_rankme_ratio: float,
    max_anisotropy_delta: float,
    pre_metrics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, bool, float, float]:
    """Compute exact 2D affine repair ``M = (I - P_0) + P_0 R P_0`` and optional bias centering."""
    dim = int(h_post.shape[1])
    eye_d = torch.eye(dim, dtype=torch.float64)
    mu_pre = h_pre.mean(dim=0)
    mu_post = h_post.mean(dim=0)
    delta_mu_0 = p0 @ (mu_post - mu_pre)
    mu_target = (eye_d - p0) @ mu_post + p0 @ mu_pre

    # Step 1: If bias exists, check whether pure mean-cone subtraction in P_0 already cures collapse
    if b64 is not None:
        h_centered_step = h_post - delta_mu_0.unsqueeze(0)
        step_m = measure_hidden_isotropy_and_rankme(
            h_centered_step,
            eps=eps_f,
            min_rankme_ratio=min_rankme_ratio,
            max_anisotropy_delta=max_anisotropy_delta,
            reference_metrics=pre_metrics,
        )
        if not step_m["collapse_warning"]:
            b_new = b64 - delta_mu_0
            return w64.clone(), b_new, h_centered_step, False, 1.0, 1.0

    idem_err = float(torch.linalg.matrix_norm(p0 @ p0 - p0, ord="fro").item())
    p_norm = max(1.0, float(torch.linalg.matrix_norm(p0, ord="fro").item()))
    if idem_err <= 1e-3 * p_norm:
        evals_p, evecs_p = torch.linalg.eigh(p0)
        u_0 = evecs_p[:, evals_p > 0.5]
    else:
        u_0 = eye_d

    r0 = int(u_0.shape[1])
    if r0 == 0:
        return w64.clone(), b64.clone() if b64 is not None else None, h_post.clone(), False, 1.0, 1.0

    # Use centered statistics when a bias absorbs the mean shift; uncentered when biasless
    x_pre_u = ((h_pre - mu_pre.unsqueeze(0)) if b64 is not None else h_pre) @ u_0
    x_post_u = ((h_post - mu_post.unsqueeze(0)) if b64 is not None else h_post) @ u_0

    n_pre = float(x_pre_u.shape[0])
    n_post = float(x_post_u.shape[0])
    cov_pre = 0.5 * ((x_pre_u.T @ x_pre_u) / n_pre + ((x_pre_u.T @ x_pre_u) / n_pre).T)
    cov_post = 0.5 * ((x_post_u.T @ x_post_u) / n_post + ((x_post_u.T @ x_post_u) / n_post).T)

    lam_post, vecs = torch.linalg.eigh(cov_post)
    lam_post = lam_post.clamp_min(0.0)
    lam_pre_dir = torch.diagonal(vecs.T @ cov_pre @ vecs).clamp_min(0.0)

    ev_pre_all = torch.linalg.eigvalsh(cov_pre).clamp_min(0.0)
    dead_tol_pre = max(eps_f * eps_f, 1e-12 * max(1.0, float(ev_pre_all[-1].item())))
    pos_pre = ev_pre_all[ev_pre_all > dead_tol_pre]
    mean_active_pre = float(pos_pre.mean().item()) if pos_pre.numel() > 0 else 0.0
    lam_target = torch.maximum(lam_pre_dir, torch.full_like(lam_pre_dir, singular_floor_frac * mean_active_pre))

    max_post_ev = float(lam_post[-1].item())
    dead_tol_post = max(eps_f * eps_f, 1e-10 * max(max_post_ev, mean_active_pre))
    gains = torch.ones_like(lam_post)
    alive_mask = lam_post > dead_tol_post
    if bool(alive_mask.any().item()):
        raw_g = torch.sqrt(lam_target[alive_mask] / lam_post[alive_mask])
        gains[alive_mask] = raw_g.clamp(1.0 / max_repair_gain, max_repair_gain)

    r_sub = (vecs * gains.unsqueeze(0)) @ vecs.T
    if idem_err <= 1e-3 * p_norm:
        m_op = (eye_d - p0) + u_0 @ r_sub @ u_0.T
    else:
        m_op = (eye_d - p0) + p0 @ r_sub @ p0
    m_op = 0.5 * (m_op + m_op.T)

    w_new = (m_op @ w64).contiguous()
    if b64 is not None:
        b_cand = m_op @ (b64 - mu_post) + mu_target
        b_new: torch.Tensor | None = b64 + p0 @ (b_cand - b64)
        h_rep = (h_post - mu_post.unsqueeze(0)) @ m_op.T + mu_target.unsqueeze(0)
    else:
        b_new = None
        h_rep = h_post @ m_op.T
        # If biasless repair still retains excessive mean-cone anisotropy inside P_0, damp the cone direction
        if float(torch.linalg.vector_norm(delta_mu_0).item()) > eps_f:
            rep_m = measure_hidden_isotropy_and_rankme(
                h_rep,
                eps=eps_f,
                min_rankme_ratio=min_rankme_ratio,
                max_anisotropy_delta=max_anisotropy_delta,
                reference_metrics=pre_metrics,
            )
            if rep_m["anisotropy"] > float(pre_metrics["anisotropy"]) + max_anisotropy_delta:
                mu_rep_0 = p0 @ h_rep.mean(dim=0)
                mu_rep_norm = float(torch.linalg.vector_norm(mu_rep_0).item())
                if mu_rep_norm > eps_f:
                    u_cone = mu_rep_0 / mu_rep_norm
                    rms_rep = float(h_rep.pow(2).sum(dim=1).mean().sqrt().item())
                    cone_scale = max(1.0 / max_repair_gain, min(1.0, (0.25 * rms_rep) / max(mu_rep_norm, eps_f)))
                    cone_op = eye_d - (1.0 - cone_scale) * torch.outer(u_cone, u_cone)
                    m_op = (eye_d - p0) + p0 @ (cone_op @ m_op)
                    m_op = 0.5 * (m_op + m_op.T)
                    w_new = (m_op @ w64).contiguous()
                    h_rep = h_post @ m_op.T
                    gains = gains * cone_scale

    return (
        w_new,
        b_new,
        h_rep,
        True,
        float(gains.min().item()),
        float(gains.max().item()),
    )


def _solve_1d_norm_nullspace_repair(
    w64: torch.Tensor,
    b64: torch.Tensor | None,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    p0: torch.Tensor,
    *,
    eps_f: float,
    singular_floor_frac: float,
    max_repair_gain: float,
    min_rankme_ratio: float,
    max_anisotropy_delta: float,
    pre_metrics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, bool, float, float]:
    """Compute closed-form mean-cone subtraction and channelwise singular floor for LayerNorm/RMSNorm."""
    dim = int(h_post.shape[1])
    eye_d = torch.eye(dim, dtype=torch.float64)
    mu_pre = h_pre.mean(dim=0)
    mu_post = h_post.mean(dim=0)
    delta_mu_0 = p0 @ (mu_post - mu_pre)
    mu_target = (eye_d - p0) @ mu_post + p0 @ mu_pre

    # Step 1: Pure mean-cone subtraction in bias inside P_0 (exact null-space invariance for any P_0)
    if b64 is not None:
        h_centered_step = h_post - delta_mu_0.unsqueeze(0)
        step_m = measure_hidden_isotropy_and_rankme(
            h_centered_step,
            eps=eps_f,
            min_rankme_ratio=min_rankme_ratio,
            max_anisotropy_delta=max_anisotropy_delta,
            reference_metrics=pre_metrics,
        )
        if not step_m["collapse_warning"]:
            b_new = b64 - delta_mu_0
            return w64.clone(), b_new, h_centered_step, False, 1.0, 1.0

    h_pre_c = (h_pre - mu_pre.unsqueeze(0)) if b64 is not None else h_pre
    h_post_c = (h_post - mu_post.unsqueeze(0)) if b64 is not None else h_post
    var_pre = h_pre_c.pow(2).mean(dim=0)
    var_post = h_post_c.pow(2).mean(dim=0)

    dead_tol_pre = max(eps_f * eps_f, 1e-12 * max(1.0, float(var_pre.max().item())))
    pos_pre = var_pre[var_pre > dead_tol_pre]
    mean_active_pre = float(pos_pre.mean().item()) if pos_pre.numel() > 0 else 0.0
    target_var = torch.maximum(var_pre, torch.full_like(var_pre, singular_floor_frac * mean_active_pre))

    dead_tol_post = max(eps_f * eps_f, 1e-10 * max(float(var_post.max().item()), mean_active_pre))
    gains = torch.ones_like(var_post)
    alive_mask = var_post > dead_tol_post
    if bool(alive_mask.any().item()):
        gains[alive_mask] = torch.sqrt(target_var[alive_mask] / var_post[alive_mask]).clamp(
            1.0 / max_repair_gain, max_repair_gain
        )

    diag_p0 = torch.diagonal(p0)
    delta_w_raw = (gains - 1.0) * w64
    # Zero out channels predominantly inside ker(P_0) before projecting onto range(P_0)
    channel_mask = (diag_p0 >= 0.5).to(dtype=torch.float64)
    delta_w = p0 @ (delta_w_raw * channel_mask)
    w_new = (w64 + delta_w).contiguous()

    eff_gain = torch.where(w64.abs() > eps_f, w_new / w64, torch.ones_like(w64))
    if b64 is not None:
        b_cand = eff_gain * (b64 - mu_post) + mu_target
        b_new: torch.Tensor | None = b64 + p0 @ (b_cand - b64)
        h_rep = (h_post - mu_post.unsqueeze(0)) * eff_gain.unsqueeze(0) + mu_target.unsqueeze(0)
    else:
        b_new = None
        h_rep = h_post * eff_gain.unsqueeze(0)

    return (
        w_new,
        b_new,
        h_rep,
        True,
        float(eff_gain.min().item()),
        float(eff_gain.max().item()),
    )


@torch.no_grad()
def synthesize_anisotropy_centering_and_nullspace_repair(
    out_weight_or_ln: Any,
    hidden_states_pre: torch.Tensor,
    hidden_states_post: torch.Tensor,
    p_null: torch.Tensor | None = None,
    *,
    min_rankme_ratio: float = DEFAULT_MIN_RANKME_RATIO,
    max_anisotropy_delta: float = DEFAULT_MAX_ANISOTROPY_DELTA,
    eps: float = DEFAULT_EPS,
    singular_floor_frac: float = DEFAULT_SINGULAR_FLOOR_FRAC,
    max_repair_gain: float = DEFAULT_MAX_REPAIR_GAIN,
    inplace: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Fold closed-form mean-cone subtraction and singular-floor restoration inside ``P_0``.

    Computes hidden-state isotropy & RankMe metrics before and after an edit. When
    ``hidden_states_post`` breaches ``min_rankme_ratio`` or ``max_anisotropy_delta`` (or
    triggers intrinsic collapse flags), synthesizes a closed-form repair inside ``range(P_0)``
    that subtracts the excess anisotropy mean cone and restores collapsed singular directions
    without perturbing any protected prose vector in ``ker(P_0)``.
    """
    h_pre = _validate_hidden_states(hidden_states_pre, "hidden_states_pre")
    h_post = _validate_hidden_states(hidden_states_post, "hidden_states_post")
    if int(h_pre.shape[1]) != int(h_post.shape[1]):
        raise ValueError(
            f"hidden_states_pre dim {int(h_pre.shape[1])} must match hidden_states_post dim {int(h_post.shape[1])}"
        )
    hidden_dim = int(h_pre.shape[1])

    min_rm_ratio = _validate_unit_interval(min_rankme_ratio, "min_rankme_ratio", allow_zero=False, allow_one=True)
    max_aniso_d = _validate_nonneg_finite(max_anisotropy_delta, "max_anisotropy_delta")
    eps_f = _validate_positive_finite(eps, "eps")
    floor_f = _validate_unit_interval(singular_floor_frac, "singular_floor_frac", allow_zero=False, allow_one=True)
    max_g = _validate_positive_finite(max_repair_gain, "max_repair_gain")
    if max_g < 1.0:
        raise ValueError(f"max_repair_gain must be >= 1.0, got {max_g}")
    do_inplace = _validate_bool(inplace, "inplace")

    kind, orig_w, orig_b = _extract_repair_target(out_weight_or_ln, hidden_dim)
    p0 = (
        _validate_square_projector(p_null, hidden_dim, "p_null")
        if p_null is not None
        else torch.eye(hidden_dim, dtype=torch.float64)
    )

    pre_m = measure_hidden_isotropy_and_rankme(
        h_pre,
        eps=eps_f,
        min_rankme_ratio=min_rm_ratio,
        max_anisotropy_delta=max_aniso_d,
    )
    post_m = measure_hidden_isotropy_and_rankme(
        h_post,
        eps=eps_f,
        min_rankme_ratio=min_rm_ratio,
        max_anisotropy_delta=max_aniso_d,
        reference_metrics=pre_m,
    )

    mean_shift = h_post.mean(dim=0) - h_pre.mean(dim=0)
    mean_cone_shift_norm = float(torch.linalg.vector_norm(p0 @ mean_shift).item())
    repair_needed = bool(post_m["collapse_warning"])

    w64 = orig_w.detach().to(device="cpu", dtype=torch.float64)
    b64 = orig_b.detach().to(device="cpu", dtype=torch.float64) if orig_b is not None else None

    if not repair_needed:
        repaired_obj = _package_repaired_target(out_weight_or_ln, kind, w64, b64, inplace=do_inplace)
        return repaired_obj, {
            "target_kind": kind,
            "repair_needed": False,
            "repair_applied": False,
            "singular_floor_applied": False,
            "mean_cone_shift_norm": mean_cone_shift_norm,
            "nullspace_drift_norm": 0.0,
            "min_singular_gain": 1.0,
            "max_singular_gain": 1.0,
            "rankme_before_repair": float(post_m["rankme"]),
            "rankme_after_repair": float(post_m["rankme"]),
            "rankme_recovery_ratio": float(post_m["rankme"] / max(float(pre_m["rankme"]), eps_f)),
            "anisotropy_before_repair": float(post_m["anisotropy"]),
            "anisotropy_after_repair": float(post_m["anisotropy"]),
            "collapse_cleared": True,
            "pre_metrics": pre_m,
            "post_metrics": post_m,
            "repaired_metrics": post_m,
        }

    is_2d = kind in ("linear", "tensor_2d", "mapping_2d", "tuple_2d")
    if is_2d:
        w_new, b_new, h_rep, sf_applied, g_min, g_max = _solve_2d_nullspace_repair(
            w64,
            b64,
            h_pre,
            h_post,
            p0,
            eps_f=eps_f,
            singular_floor_frac=floor_f,
            max_repair_gain=max_g,
            min_rankme_ratio=min_rm_ratio,
            max_anisotropy_delta=max_aniso_d,
            pre_metrics=pre_m,
        )
    else:
        w_new, b_new, h_rep, sf_applied, g_min, g_max = _solve_1d_norm_nullspace_repair(
            w64,
            b64,
            h_pre,
            h_post,
            p0,
            eps_f=eps_f,
            singular_floor_frac=floor_f,
            max_repair_gain=max_g,
            min_rankme_ratio=min_rm_ratio,
            max_anisotropy_delta=max_aniso_d,
            pre_metrics=pre_m,
        )

    repaired_obj = _package_repaired_target(out_weight_or_ln, kind, w_new, b_new, inplace=do_inplace)

    eye_d = torch.eye(hidden_dim, dtype=torch.float64)
    q_proj = eye_d - p0
    dw = w_new - w64
    if is_2d:
        w_drift = float(torch.linalg.matrix_norm(q_proj @ dw, ord="fro").item())
    else:
        w_drift = float(torch.linalg.vector_norm(q_proj @ dw).item())
    b_drift = (
        float(torch.linalg.vector_norm(q_proj @ (b_new - b64)).item())
        if (b_new is not None and b64 is not None)
        else 0.0
    )
    nullspace_drift_norm = max(w_drift, b_drift)

    rep_m = measure_hidden_isotropy_and_rankme(
        h_rep,
        eps=eps_f,
        min_rankme_ratio=min_rm_ratio,
        max_anisotropy_delta=max_aniso_d,
        reference_metrics=pre_m,
    )
    delta_w_norm = float(torch.linalg.norm(dw).item())
    delta_b_norm = float(torch.linalg.norm(b_new - b64).item()) if (b_new is not None and b64 is not None) else 0.0
    repair_applied = (delta_w_norm > 1e-15) or (delta_b_norm > 1e-15)

    return repaired_obj, {
        "target_kind": kind,
        "repair_needed": True,
        "repair_applied": repair_applied,
        "singular_floor_applied": sf_applied,
        "mean_cone_shift_norm": mean_cone_shift_norm,
        "nullspace_drift_norm": float(nullspace_drift_norm),
        "min_singular_gain": g_min,
        "max_singular_gain": g_max,
        "rankme_before_repair": float(post_m["rankme"]),
        "rankme_after_repair": float(rep_m["rankme"]),
        "rankme_recovery_ratio": float(rep_m["rankme"] / max(float(pre_m["rankme"]), eps_f)),
        "anisotropy_before_repair": float(post_m["anisotropy"]),
        "anisotropy_after_repair": float(rep_m["anisotropy"]),
        "collapse_cleared": not bool(rep_m["collapse_warning"]),
        "pre_metrics": pre_m,
        "post_metrics": post_m,
        "repaired_metrics": rep_m,
    }


# ---------------------------------------------------------------------------
# 4. Verified Collapse-Guarded WSP Surgery & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    name: str,
    *,
    general_prefix: str,
    score_from: int,
    require_both_groups: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Validate token chunks and return ``(ids_tensor, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    seq_lens: set[int] = set()
    ids_rows: list[list[int]] = []
    expert_ids: list[str] = []
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch or "expert_id" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids' and 'expert_id'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        raw_ids = ch["ids"]
        if isinstance(raw_ids, torch.Tensor):
            row = [int(t) for t in raw_ids.detach().reshape(-1).tolist()]
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
            row = [int(t) for t in raw_ids]
        else:
            raise TypeError(f"{name}[{idx}]['ids'] must be a sequence or tensor of token IDs")
        if len(row) < 2 or any(t < 0 for t in row):
            raise ValueError(f"{name}[{idx}]['ids'] must contain >= 2 non-negative token IDs")
        seq_lens.add(len(row))
        ids_rows.append(row)
        expert_ids.append(eid)

    if len(seq_lens) != 1:
        raise ValueError(f"All {name} entries must share identical sequence lengths")
    seq_len = next(iter(seq_lens))
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} is too short for score_from={score_from}")

    has_gen = any(eid.startswith(general_prefix) for eid in expert_ids)
    has_spec = any(not eid.startswith(general_prefix) for eid in expert_ids)
    if require_both_groups and (not has_gen or not has_spec):
        raise ValueError(
            f"{name} must include both general ('{general_prefix}*') and specialist chunks"
        )
    return torch.tensor(ids_rows, dtype=torch.long), expert_ids


@torch.no_grad()
def _eval_crowd_model_domain_nlls(
    model: nn.Module,
    ids_eval: torch.Tensor,
    expert_ids: Sequence[str],
    *,
    score_from: int,
    batch_size: int,
) -> dict[str, float]:
    """Compute exact per-domain mean causal NLL on ``CrowdV1WithEngram``."""
    h_pre = forward_model_pre_ln(model, ids_eval, batch_size=batch_size)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_eval,
        score_from=score_from,
    ).tolist()
    grouped: dict[str, list[float]] = {}
    for eid, val in zip(expert_ids, nlls, strict=True):
        grouped.setdefault(str(eid), []).append(float(val))
    return {k: float(sum(v) / float(len(v))) for k, v in sorted(grouped.items())}


def _parse_eval_domain_nll(raw_eval: Any) -> dict[str, float]:
    """Normalize ``eval_fn(model)`` output into a finite ``dict[str, float]``."""
    if isinstance(raw_eval, tuple) and len(raw_eval) == 2 and isinstance(raw_eval[1], Mapping):
        mapping: Mapping[str, Any] = raw_eval[1]
    elif isinstance(raw_eval, Mapping):
        if "per_domain" in raw_eval and isinstance(raw_eval["per_domain"], Mapping):
            mapping = raw_eval["per_domain"]
        else:
            mapping = raw_eval
    else:
        raise TypeError("eval_fn must return a non-empty mapping of domain -> NLL or (nll_mean, domain_map)")

    if not mapping:
        raise ValueError("Domain NLL mapping must be non-empty")
    out: dict[str, float] = {}
    for k in sorted(mapping.keys()):
        val = float(mapping[k])
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"Non-finite or non-positive domain NLL for {k!r}: {val!r}")
        out[str(k)] = val
    return out


def _summarize_anse_metrics(
    ref_nll: Mapping[str, float],
    cand_nll: Mapping[str, float],
    *,
    cfg: CollapseGuardConfig,
) -> dict[str, Any]:
    """Compute general-domain guard clearance, worst-domain drift, and ANSE energy."""
    keys = sorted(ref_nll.keys())
    if set(cand_nll.keys()) != set(keys):
        raise ValueError("Candidate and reference domain NLL mappings must share identical keys")

    gen_keys = [k for k in keys if k.startswith(cfg.general_prefix)]
    if not gen_keys:
        raise ValueError(f"Missing required general-domain key starting with {cfg.general_prefix!r}")

    gen_guard_cap = (
        cfg.general_guard_max_ratio
        if cfg.general_guard_max_ratio is not None
        else cfg.domain_guard_max_ratio
    )
    diffs: list[float] = []
    per_domain: dict[str, dict[str, float]] = {}
    worst_id = ""
    worst_ratio = 0.0
    for k in keys:
        b = float(ref_nll[k])
        c = float(cand_nll[k])
        diff = c - b
        ratio = c / max(b, 1e-8)
        diffs.append(diff)
        per_domain[k] = {
            "base_nll": b,
            "cand_nll": c,
            "diff_vs_base": diff,
            "ratio": ratio,
            "pct_vs_base": 100.0 * (ratio - 1.0),
        }
        if ratio >= worst_ratio:
            worst_ratio = ratio
            worst_id = k

    gen_ratio = max(per_domain[k]["ratio"] for k in gen_keys)
    general_guard_ok = bool(gen_ratio <= gen_guard_cap + 1e-12)
    worst_guard_ok = bool(worst_ratio <= cfg.domain_guard_max_ratio + 1e-12)
    guard_ok = bool(general_guard_ok and (worst_guard_ok if cfg.enforce_all_domain_guard else True))

    mean_diff = float(sum(diffs) / float(len(diffs)))
    nll_mean = float(sum(float(cand_nll[k]) for k in keys) / float(len(keys)))
    base_nll_mean = float(sum(float(ref_nll[k]) for k in keys) / float(len(keys)))

    if not guard_ok:
        anse_energy = float(cfg.maximum_pain_energy)
    else:
        soft_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_e0_ratio else 0.0
        anse_energy = mean_diff + soft_pen

    return {
        "nll_mean": nll_mean,
        "base_nll_mean": base_nll_mean,
        "mean_diff_vs_base": mean_diff,
        "general_ratio": float(gen_ratio),
        "general_pct_vs_base": 100.0 * (float(gen_ratio) - 1.0),
        "worst_domain_id": worst_id,
        "worst_domain_ratio": float(worst_ratio),
        "worst_domain_pct_vs_base": 100.0 * (float(worst_ratio) - 1.0),
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "per_domain": per_domain,
    }


def _resolve_repair_targets(
    work_model: nn.Module,
    repair_target: nn.Module | Sequence[nn.Module | str] | str | None,
) -> list[tuple[str, nn.Module]]:
    """Resolve explicit or default output/norm repair target module(s) on ``work_model``."""
    name_by_id = {id(m): (n if n else "root") for n, m in work_model.named_modules()}

    def _resolve_one(item: nn.Module | str) -> tuple[str, nn.Module]:
        if isinstance(item, str):
            mod = work_model.get_submodule(item)
            return item, mod
        if isinstance(item, nn.Module):
            return name_by_id.get(id(item), type(item).__name__), item
        raise TypeError(f"repair_target items must be nn.Module or str, got {type(item).__name__}")

    if repair_target is not None:
        if isinstance(repair_target, (nn.Module, str)):
            return [_resolve_one(repair_target)]
        if isinstance(repair_target, Sequence):
            if not repair_target:
                raise ValueError("repair_target sequence must not be empty")
            return [_resolve_one(it) for it in repair_target]
        raise TypeError(f"Unsupported repair_target type {type(repair_target).__name__}")

    ln_mod = getattr(work_model, "ln", None)
    if isinstance(ln_mod, nn.Module) and isinstance(getattr(ln_mod, "weight", None), torch.Tensor):
        return [("ln", ln_mod)]
    norm_mod = getattr(getattr(work_model, "model", work_model), "norm", None)
    if isinstance(norm_mod, nn.Module) and isinstance(getattr(norm_mod, "weight", None), torch.Tensor):
        return [(name_by_id.get(id(norm_mod), "norm"), norm_mod)]
    if isinstance(work_model, (nn.Linear, nn.LayerNorm)) or (
        isinstance(getattr(work_model, "weight", None), torch.Tensor)
        and work_model.weight.ndim in (1, 2)
    ):
        return [("root", work_model)]

    candidates = [
        (name, mod)
        for name, mod in work_model.named_modules()
        if isinstance(mod, (nn.LayerNorm, nn.Linear))
        or (
            isinstance(getattr(mod, "weight", None), torch.Tensor)
            and mod.weight.ndim == 1
        )
    ]
    if not candidates:
        raise ValueError("Could not locate any Linear or LayerNorm/RMSNorm repair_target on model")
    return [candidates[-1]]


def _resolve_p_null_for_target(
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    target_name: str,
) -> torch.Tensor | None:
    """Resolve optional ``p_null`` projector for ``target_name``."""
    if p_null is None:
        return None
    if isinstance(p_null, torch.Tensor):
        return p_null
    if isinstance(p_null, Mapping):
        for key in (target_name, "default"):
            if key in p_null:
                return p_null[key]
        return None
    raise TypeError(f"p_null must be Tensor, Mapping, or None, got {type(p_null).__name__}")


def _capture_hidden_states(
    work_model: nn.Module,
    *,
    hidden_states_fn: Callable[[nn.Module], Any] | None,
    ids_cal: torch.Tensor | None,
    targets: Sequence[tuple[str, nn.Module]],
    score_from: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Capture per-target hidden-state tensors via ``hidden_states_fn`` or CrowdV1 forward pass."""
    if hidden_states_fn is not None:
        raw = hidden_states_fn(work_model)
        if isinstance(raw, torch.Tensor):
            return {t_name: raw.detach().to(device="cpu", dtype=torch.float64) for t_name, _ in targets}
        if isinstance(raw, Mapping):
            out: dict[str, torch.Tensor] = {}
            for t_name, _ in targets:
                if t_name in raw:
                    out[t_name] = raw[t_name].detach().to(device="cpu", dtype=torch.float64)
            if not out and len(raw) == 1:
                only = next(iter(raw.values()))
                return {t_name: only.detach().to(device="cpu", dtype=torch.float64) for t_name, _ in targets}
            if len(out) != len(targets):
                raise ValueError("hidden_states_fn mapping missing target keys")
            return out
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != len(targets):
                raise ValueError("hidden_states_fn sequence length must match repair targets")
            return {
                t_name: tens.detach().to(device="cpu", dtype=torch.float64)
                for (t_name, _), tens in zip(targets, raw, strict=True)
            }
        raise TypeError("hidden_states_fn must return Tensor, Mapping[str, Tensor], or Sequence[Tensor]")

    assert ids_cal is not None
    start_pos = max(0, score_from - 1)
    if len(targets) == 1 and targets[0][1] is getattr(work_model, "ln", None):
        h_pre_ln = forward_model_pre_ln(work_model, ids_cal, batch_size=batch_size)
        ln_mod = work_model.ln
        ln_dt = ln_mod.weight.dtype
        h_post_ln = ln_mod(h_pre_ln.to(device=ln_mod.weight.device, dtype=ln_dt)).to(torch.float64)
        sliced = h_post_ln[:, start_pos:-1, :]
        return {targets[0][0]: sliced.reshape(-1, int(sliced.shape[-1])).detach().cpu()}

    slots: dict[str, list[torch.Tensor]] = {t_name: [] for t_name, _ in targets}
    hooks: list[Any] = []
    n_seq = int(ids_cal.shape[1])

    def _make_hook(t_key: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], Any], None]:
        def _hook(_m: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            out_t = output[0] if isinstance(output, tuple) else output
            t_cpu = out_t.detach().to(device="cpu", dtype=torch.float64)
            if t_cpu.ndim == 3:
                t_cpu = t_cpu[:, start_pos:-1, :]
                t_cpu = t_cpu.reshape(-1, int(t_cpu.shape[-1]))
            elif t_cpu.ndim == 2 and n_seq > 2 and int(t_cpu.shape[0]) % n_seq == 0:
                b_cnt = int(t_cpu.shape[0]) // n_seq
                t_cpu = t_cpu.view(b_cnt, n_seq, int(t_cpu.shape[-1]))[:, start_pos:-1, :].reshape(
                    -1, int(t_cpu.shape[-1])
                )
            slots[t_key].append(t_cpu)

        return _hook

    try:
        for t_name, t_mod in targets:
            hooks.append(t_mod.register_forward_hook(_make_hook(t_name)))
        h_pre_ln = forward_model_pre_ln(work_model, ids_cal, batch_size=batch_size)
        if any(t_mod is getattr(work_model, "ln", None) for _, t_mod in targets):
            _ = work_model.ln(h_pre_ln.to(device=work_model.ln.weight.device, dtype=work_model.ln.weight.dtype))
    finally:
        for h in hooks:
            h.remove()

    captured: dict[str, torch.Tensor] = {}
    for t_name, _ in targets:
        if not slots[t_name]:
            raise RuntimeError(f"Repair target {t_name!r} produced no activations during forward pass")
        captured[t_name] = torch.cat(slots[t_name], dim=0).contiguous()
    return captured


def _resolve_guard_config(
    config: CollapseGuardConfig | None,
    *,
    min_rankme_ratio: float | None,
    max_anisotropy_delta: float | None,
    domain_guard_max_ratio: float | None,
    general_guard_max_ratio: float | None,
    enforce_all_domain_guard: bool | None,
    require_energy_improvement: bool | None,
    attempt_closed_form_repair: bool | None,
    require_repair_clear_collapse: bool | None,
    score_from: int | None,
    inplace: bool | None,
) -> CollapseGuardConfig:
    """Merge optional keyword overrides into a validated ``CollapseGuardConfig``."""
    if config is not None and not isinstance(config, CollapseGuardConfig):
        raise TypeError(f"config must be CollapseGuardConfig or None, got {type(config).__name__}")
    base = config or CollapseGuardConfig()
    return CollapseGuardConfig(
        min_rankme_ratio=base.min_rankme_ratio if min_rankme_ratio is None else min_rankme_ratio,
        max_anisotropy_delta=(
            base.max_anisotropy_delta if max_anisotropy_delta is None else max_anisotropy_delta
        ),
        max_anisotropy_abs=base.max_anisotropy_abs,
        min_normalized_rankme=base.min_normalized_rankme,
        eps=base.eps,
        singular_floor_frac=base.singular_floor_frac,
        max_repair_gain=base.max_repair_gain,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio if domain_guard_max_ratio is None else domain_guard_max_ratio
        ),
        general_guard_max_ratio=(
            base.general_guard_max_ratio if general_guard_max_ratio is None else general_guard_max_ratio
        ),
        enforce_all_domain_guard=(
            base.enforce_all_domain_guard if enforce_all_domain_guard is None else enforce_all_domain_guard
        ),
        anse_safe_e0_ratio=base.anse_safe_e0_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        require_energy_improvement=(
            base.require_energy_improvement
            if require_energy_improvement is None
            else require_energy_improvement
        ),
        attempt_closed_form_repair=(
            base.attempt_closed_form_repair
            if attempt_closed_form_repair is None
            else attempt_closed_form_repair
        ),
        require_repair_clear_collapse=(
            base.require_repair_clear_collapse
            if require_repair_clear_collapse is None
            else require_repair_clear_collapse
        ),
        general_prefix=base.general_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


def _apply_candidate_edit(
    work_model: nn.Module,
    *,
    candidate_edit_fn: Callable[[nn.Module], Any] | None,
    candidate_state_dict: Mapping[str, torch.Tensor] | None,
) -> None:
    """Apply candidate WSP edit via callable or state dict."""
    if candidate_edit_fn is not None:
        candidate_edit_fn(work_model)
        return
    assert candidate_state_dict is not None
    cur_sd = work_model.state_dict()
    missing = set(cur_sd.keys()) - set(candidate_state_dict.keys())
    extra = set(candidate_state_dict.keys()) - set(cur_sd.keys())
    if missing or extra:
        raise ValueError(f"candidate_state_dict keys mismatch model keys (missing={sorted(missing)}, extra={sorted(extra)})")
    loaded: dict[str, torch.Tensor] = {}
    for key in cur_sd:
        t = candidate_state_dict[key]
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(cur_sd[key].shape):
            raise ValueError(f"candidate_state_dict[{key!r}] shape mismatch")
        if t.is_floating_point() and not bool(torch.isfinite(t.detach()).all().item()):
            raise ValueError(f"candidate_state_dict[{key!r}] contains non-finite values")
        loaded[key] = t.detach().to(device=cur_sd[key].device, dtype=cur_sd[key].dtype)
    work_model.load_state_dict(loaded)


def _evaluate_candidate_nll(
    work_model: nn.Module,
    *,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None,
    ids_eval: torch.Tensor | None,
    expert_ids_eval: Sequence[str],
    ref_nll: Mapping[str, float],
    parent_eval: Mapping[str, Any],
    cfg: CollapseGuardConfig,
) -> tuple[dict[str, Any], str | None]:
    """Evaluate domain NLLs and ANSE energy, returning Maximum Pain on evaluation failure."""
    try:
        if eval_fn is not None:
            cand_nll = _parse_eval_domain_nll(eval_fn(work_model))
        else:
            assert ids_eval is not None
            cand_nll = _eval_crowd_model_domain_nlls(
                work_model,
                ids_eval,
                expert_ids_eval,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )
        return _summarize_anse_metrics(ref_nll, cand_nll, cfg=cfg), None
    except Exception as exc:
        return (
            {
                "nll_mean": None,
                "base_nll_mean": float(parent_eval["base_nll_mean"]),
                "mean_diff_vs_base": None,
                "general_ratio": None,
                "general_pct_vs_base": None,
                "worst_domain_id": "",
                "worst_domain_ratio": None,
                "worst_domain_pct_vs_base": None,
                "guard_ok": False,
                "anse_energy": float(cfg.maximum_pain_energy),
                "per_domain": {},
            },
            f"{type(exc).__name__}: {exc}",
        )


def _aggregate_hidden_metrics(
    metrics_by_target: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize multi-target hidden isotropy metrics using worst-case collapse semantics."""
    names = sorted(metrics_by_target.keys())
    worst_rm_name = min(names, key=lambda n: float(metrics_by_target[n]["rankme"]))
    worst_iso_name = min(names, key=lambda n: float(metrics_by_target[n]["isotropy_index"]))
    return {
        "rankme": float(metrics_by_target[worst_rm_name]["rankme"]),
        "participation_ratio": float(
            min(float(metrics_by_target[n]["participation_ratio"]) for n in names)
        ),
        "isotropy_index": float(metrics_by_target[worst_iso_name]["isotropy_index"]),
        "anisotropy": float(max(float(metrics_by_target[n]["anisotropy"]) for n in names)),
        "collapse_warning": any(bool(metrics_by_target[n]["collapse_warning"]) for n in names),
        "per_target": {n: dict(metrics_by_target[n]) for n in names},
    }


@torch.no_grad()
def execute_verified_collapse_guarded_wsp_surgery(
    model: nn.Module,
    calibration_chunks: Sequence[Mapping[str, Any]] | None = None,
    *,
    candidate_edit_fn: Callable[[nn.Module], Any] | None = None,
    candidate_state_dict: Mapping[str, torch.Tensor] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    hidden_states_fn: Callable[[nn.Module], Any] | None = None,
    repair_target: nn.Module | Sequence[nn.Module | str] | str | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    config: CollapseGuardConfig | None = None,
    min_rankme_ratio: float | None = None,
    max_anisotropy_delta: float | None = None,
    domain_guard_max_ratio: float | None = None,
    general_guard_max_ratio: float | None = None,
    enforce_all_domain_guard: bool | None = None,
    require_energy_improvement: bool | None = None,
    attempt_closed_form_repair: bool | None = None,
    require_repair_clear_collapse: bool | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> dict[str, Any]:
    """Execute representation-collapse-guarded WSP surgery with closed-form repair and atomic rollback.

    Captures hidden representations before and after the candidate WSP edit, checks RankMe
    retention and anisotropy cone bounds, synthesizes closed-form ``P_0`` centering &
    singular-floor restoration when collapse is detected, and commits if and only if
    collapse is cleared, ANSE ``Delta E < 0``, and general-domain drift ``<= +1.0%``.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if (candidate_edit_fn is None) == (candidate_state_dict is None):
        raise ValueError("Provide exactly one of candidate_edit_fn or candidate_state_dict")
    if candidate_edit_fn is not None and not callable(candidate_edit_fn):
        raise TypeError("candidate_edit_fn must be callable")
    if candidate_state_dict is not None and not isinstance(candidate_state_dict, Mapping):
        raise TypeError("candidate_state_dict must be a Mapping[str, torch.Tensor]")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if hidden_states_fn is not None and not callable(hidden_states_fn):
        raise TypeError("hidden_states_fn must be callable or None")

    cfg = _resolve_guard_config(
        config,
        min_rankme_ratio=min_rankme_ratio,
        max_anisotropy_delta=max_anisotropy_delta,
        domain_guard_max_ratio=domain_guard_max_ratio,
        general_guard_max_ratio=general_guard_max_ratio,
        enforce_all_domain_guard=enforce_all_domain_guard,
        require_energy_improvement=require_energy_improvement,
        attempt_closed_form_repair=attempt_closed_form_repair,
        require_repair_clear_collapse=require_repair_clear_collapse,
        score_from=score_from,
        inplace=inplace,
    )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    targets = _resolve_repair_targets(work_model, repair_target)

    has_crowd_attrs = (
        hasattr(work_model, "blocks")
        and hasattr(work_model, "wte")
        and hasattr(work_model, "ln")
        and hasattr(work_model, "head")
    )

    ids_cal: torch.Tensor | None = None
    ids_eval: torch.Tensor | None = None
    expert_ids_eval: list[str] = []

    if calibration_chunks is not None:
        if not has_crowd_attrs and (hidden_states_fn is None or eval_fn is None):
            raise TypeError("calibration_chunks requires a CrowdV1WithEngram-compatible model")
        ids_cal, expert_ids_cal = _validate_chunks(
            calibration_chunks,
            "calibration_chunks",
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            require_both_groups=True,
        )
        if eval_chunks is not None:
            ids_eval, expert_ids_eval = _validate_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefix=cfg.general_prefix,
                score_from=cfg.score_from,
                require_both_groups=False,
            )
        else:
            ids_eval, expert_ids_eval = ids_cal, expert_ids_cal
    else:
        if hidden_states_fn is None:
            raise ValueError("hidden_states_fn is required when calibration_chunks is None")
        if eval_fn is None:
            if eval_chunks is None:
                raise ValueError("Either calibration_chunks, eval_chunks, or eval_fn must be provided")
            if not has_crowd_attrs:
                raise TypeError("eval_chunks without eval_fn requires a CrowdV1WithEngram-compatible model")
            ids_eval, expert_ids_eval = _validate_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefix=cfg.general_prefix,
                score_from=cfg.score_from,
                require_both_groups=False,
            )

    base_nll_clean = _parse_eval_domain_nll(baseline_domain_nll) if baseline_domain_nll is not None else None
    was_training = work_model.training
    work_model.eval()

    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)

        if eval_fn is not None:
            pre_nll = _parse_eval_domain_nll(eval_fn(work_model))
        else:
            assert ids_eval is not None
            pre_nll = _eval_crowd_model_domain_nlls(
                work_model, ids_eval, expert_ids_eval, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
        ref_nll = base_nll_clean if base_nll_clean is not None else pre_nll
        if set(ref_nll.keys()) != set(pre_nll.keys()):
            raise ValueError("baseline_domain_nll and model domain NLLs must share identical keys")
        parent_eval = _summarize_anse_metrics(ref_nll, pre_nll, cfg=cfg)
        parent_e = float(parent_eval["anse_energy"])

        pre_hidden = _capture_hidden_states(
            work_model,
            hidden_states_fn=hidden_states_fn,
            ids_cal=ids_cal,
            targets=targets,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        pre_by_target = {
            t_name: measure_hidden_isotropy_and_rankme(
                pre_hidden[t_name],
                eps=cfg.eps,
                min_normalized_rankme=cfg.min_normalized_rankme,
                max_anisotropy=cfg.max_anisotropy_abs,
                min_rankme_ratio=cfg.min_rankme_ratio,
                max_anisotropy_delta=cfg.max_anisotropy_delta,
            )
            for t_name, _ in targets
        }
        pre_agg = _aggregate_hidden_metrics(pre_by_target)

        _apply_candidate_edit(
            work_model, candidate_edit_fn=candidate_edit_fn, candidate_state_dict=candidate_state_dict
        )
        cand_sha = compute_state_sha256(work_model.state_dict())

        post_hidden = _capture_hidden_states(
            work_model,
            hidden_states_fn=hidden_states_fn,
            ids_cal=ids_cal,
            targets=targets,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        post_by_target = {
            t_name: measure_hidden_isotropy_and_rankme(
                post_hidden[t_name],
                eps=cfg.eps,
                min_normalized_rankme=cfg.min_normalized_rankme,
                max_anisotropy=cfg.max_anisotropy_abs,
                min_rankme_ratio=cfg.min_rankme_ratio,
                max_anisotropy_delta=cfg.max_anisotropy_delta,
                reference_metrics=pre_by_target[t_name],
            )
            for t_name, _ in targets
        }
        post_agg = _aggregate_hidden_metrics(post_by_target)
        collapse_detected = bool(post_agg["collapse_warning"])

        repair_attempted = False
        repair_applied = False
        repair_reports: list[dict[str, Any]] = []
        final_hidden = post_hidden
        final_by_target = post_by_target
        final_agg = post_agg

        if collapse_detected and cfg.attempt_closed_form_repair:
            repair_attempted = True
            for t_name, t_mod in targets:
                p_unit = _resolve_p_null_for_target(p_null, t_name)
                _, rep_diag = synthesize_anisotropy_centering_and_nullspace_repair(
                    t_mod,
                    pre_hidden[t_name],
                    post_hidden[t_name],
                    p_unit,
                    min_rankme_ratio=cfg.min_rankme_ratio,
                    max_anisotropy_delta=cfg.max_anisotropy_delta,
                    eps=cfg.eps,
                    singular_floor_frac=cfg.singular_floor_frac,
                    max_repair_gain=cfg.max_repair_gain,
                    inplace=True,
                )
                repair_applied = repair_applied or bool(rep_diag["repair_applied"])
                repair_reports.append(
                    {
                        "target": t_name,
                        "target_kind": rep_diag["target_kind"],
                        "repair_applied": bool(rep_diag["repair_applied"]),
                        "singular_floor_applied": bool(rep_diag["singular_floor_applied"]),
                        "mean_cone_shift_norm": float(rep_diag["mean_cone_shift_norm"]),
                        "nullspace_drift_norm": float(rep_diag["nullspace_drift_norm"]),
                        "min_singular_gain": float(rep_diag["min_singular_gain"]),
                        "max_singular_gain": float(rep_diag["max_singular_gain"]),
                        "rankme_before_repair": float(rep_diag["rankme_before_repair"]),
                        "rankme_after_repair": float(rep_diag["rankme_after_repair"]),
                        "rankme_recovery_ratio": float(rep_diag["rankme_recovery_ratio"]),
                    }
                )

            final_hidden = _capture_hidden_states(
                work_model,
                hidden_states_fn=hidden_states_fn,
                ids_cal=ids_cal,
                targets=targets,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )
            final_by_target = {
                t_name: measure_hidden_isotropy_and_rankme(
                    final_hidden[t_name],
                    eps=cfg.eps,
                    min_normalized_rankme=cfg.min_normalized_rankme,
                    max_anisotropy=cfg.max_anisotropy_abs,
                    min_rankme_ratio=cfg.min_rankme_ratio,
                    max_anisotropy_delta=cfg.max_anisotropy_delta,
                    reference_metrics=pre_by_target[t_name],
                )
                for t_name, _ in targets
            }
            final_agg = _aggregate_hidden_metrics(final_by_target)

        collapse_cleared = not bool(final_agg["collapse_warning"])
        collapse_ok = collapse_cleared if cfg.require_repair_clear_collapse else True

        post_repair_sd = work_model.state_dict()
        post_repair_sha = compute_state_sha256(post_repair_sd)
        total_elements = sum(int(v.numel()) for v in parent_snapshot.values())
        edited_elements = sum(
            int(torch.ne(post_repair_sd[k].detach().cpu(), parent_snapshot[k]).sum().item())
            for k in parent_snapshot
        )
        edited_param_ratio = float(edited_elements) / float(max(1, total_elements))

        if collapse_detected and not collapse_ok:
            cand_eval: dict[str, Any] = {
                "nll_mean": None,
                "base_nll_mean": float(parent_eval["base_nll_mean"]),
                "mean_diff_vs_base": None,
                "general_ratio": None,
                "general_pct_vs_base": None,
                "worst_domain_id": "",
                "worst_domain_ratio": None,
                "worst_domain_pct_vs_base": None,
                "guard_ok": False,
                "anse_energy": float(cfg.maximum_pain_energy),
                "per_domain": {},
            }
            eval_error: str | None = None
            cand_e = float(cfg.maximum_pain_energy)
            delta_e = cand_e - parent_e
            status = "ROLLED_BACK_UNRECOVERABLE_COLLAPSE"
            accepted = False
            rolled_back = True
        else:
            cand_eval, eval_error = _evaluate_candidate_nll(
                work_model,
                eval_fn=eval_fn,
                ids_eval=ids_eval,
                expert_ids_eval=expert_ids_eval,
                ref_nll=ref_nll,
                parent_eval=parent_eval,
                cfg=cfg,
            )
            cand_e = float(cand_eval["anse_energy"])
            delta_e = cand_e - parent_e
            energy_ok = (delta_e < 0.0) if cfg.require_energy_improvement else (delta_e <= 0.0)

            if post_repair_sha == parent_sha:
                status = "FIXED_POINT_UNCHANGED"
                accepted = False
                rolled_back = False
            elif eval_error is None and bool(cand_eval["guard_ok"]) and energy_ok:
                status = "ACCEPTED_REPAIRED" if repair_applied else "ACCEPTED"
                accepted = True
                rolled_back = False
            else:
                if eval_error is not None:
                    status = "ROLLED_BACK_EVAL_FAILURE"
                elif not bool(cand_eval["guard_ok"]):
                    status = "ROLLED_BACK_GUARD_VIOLATION"
                else:
                    status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
                accepted = False
                rolled_back = True

        if rolled_back:
            cur_sd = work_model.state_dict()
            work_model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_state_sha256(work_model.state_dict())
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback verification failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
    finally:
        work_model.train(was_training)

    pre_rm = float(pre_agg["rankme"])
    post_rm = float(post_agg["rankme"])
    fin_rm = float(final_agg["rankme"])

    receipt: dict[str, Any] = {
        "object": GUARD_RECEIPT_OBJECT,
        "schema": GUARD_SCHEMA,
        "status": status,
        "accepted": accepted,
        "committed": accepted,
        "rolled_back": rolled_back,
        "collapse_detected": collapse_detected,
        "collapse_cleared": collapse_cleared,
        "repair_attempted": repair_attempted,
        "repair_applied": repair_applied,
        "eval_error": eval_error,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "mean_diff_vs_base": cand_eval["mean_diff_vs_base"],
        "anse_energy": float(cand_e),
        "parent_anse_energy": float(parent_e),
        "final_anse_energy": float(cand_e) if accepted else float(parent_e),
        "delta_e": float(delta_e),
        "general_ratio": cand_eval["general_ratio"],
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "per_domain": cand_eval["per_domain"],
        "rankme_pre": pre_rm,
        "rankme_post": post_rm,
        "rankme_final": fin_rm,
        "rankme_ratio_post": float(post_rm / max(pre_rm, cfg.eps)),
        "rankme_ratio_final": float(fin_rm / max(pre_rm, cfg.eps)),
        "participation_ratio_pre": float(pre_agg["participation_ratio"]),
        "participation_ratio_post": float(post_agg["participation_ratio"]),
        "participation_ratio_final": float(final_agg["participation_ratio"]),
        "isotropy_index_pre": float(pre_agg["isotropy_index"]),
        "isotropy_index_post": float(post_agg["isotropy_index"]),
        "isotropy_index_final": float(final_agg["isotropy_index"]),
        "anisotropy_pre": float(pre_agg["anisotropy"]),
        "anisotropy_post": float(post_agg["anisotropy"]),
        "anisotropy_final": float(final_agg["anisotropy"]),
        "anisotropy_delta_post": float(post_agg["anisotropy"] - pre_agg["anisotropy"]),
        "anisotropy_delta_final": float(final_agg["anisotropy"] - pre_agg["anisotropy"]),
        "pre_isotropy_metrics": pre_agg,
        "post_isotropy_metrics": post_agg,
        "final_isotropy_metrics": final_agg,
        "repair_reports": repair_reports,
        "edited_elements": edited_elements,
        "total_elements": total_elements,
        "edited_param_ratio": edited_param_ratio,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CollapseGuardConfig",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_EPS",
    "DEFAULT_MAX_ANISOTROPY_ABS",
    "DEFAULT_MAX_ANISOTROPY_DELTA",
    "DEFAULT_MAX_REPAIR_GAIN",
    "DEFAULT_MIN_NORMALIZED_RANKME",
    "DEFAULT_MIN_RANKME_RATIO",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SINGULAR_FLOOR_FRAC",
    "GENERAL_EXPERT_PREFIX",
    "GUARD_RECEIPT_OBJECT",
    "GUARD_SCHEMA",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "compute_state_sha256",
    "execute_verified_collapse_guarded_wsp_surgery",
    "measure_hidden_isotropy_and_rankme",
    "synthesize_anisotropy_centering_and_nullspace_repair",
    "verify_collapse_guard_receipt_provenance",
]
