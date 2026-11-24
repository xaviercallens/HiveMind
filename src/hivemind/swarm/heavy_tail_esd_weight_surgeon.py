# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Activation-Free Heavy-Tail ESD Power-Law Alpha & Effective-Rank Weight Surgeon.

Implements Martin & Mahoney (JMLR 2021) Heavy-Tail Self-Regularization / Empirical
Spectral Density (ESD) diagnosis and singular-spectrum surgery for cold-start
open-weight Sparse MoE and Dense checkpoints when calibration activations are limited
or noisy:

1. ``fit_esd_power_law_metrics(weight_matrix, *, tail_quantile=0.25)``:
   Computes the exact CPU ``float64`` singular spectrum, Roy-Vetterli entropy effective
   rank, Frobenius stable rank, Clauset-Shalizi Hill MLE power-law exponent
   ``pl_alpha``, Marchenko-Pastur bulk edge ``bulk_edge``, and correlation-trap /
   rank-collapse diagnostics (``pl_alpha < 2.0`` or ``lambda_max / bulk_edge >> 1``).
2. ``regularize_correlation_trap_spectrum_nullspace(weight_delta, *, p_null=None, ...)``:
   Surgically winsorizes hyper-correlated outlier singular modes (``pl_alpha < 2``
   correlation traps or disconnected bulk-edge spikes) while preserving healthy
   heavy-tail modes (``pl_alpha in [2, 4]``) and bulk spectral energy, optionally
   projecting adjustments into the AlphaEdit prose null-space ``P_0``.
3. ``execute_verified_esd_moe_surgery(...)``:
   Scans all MoE experts and/or dense MLP layers without calibration activations,
   repairs correlation-trapped blocks, and commits edits if and only if ANSE
   ``Delta E < 0`` and general-domain drift ``<= +1.0%``; otherwise executes atomic
   bitwise rollback with a canonical SHA-256 provenance receipt.
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

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

SURGEON_SCHEMA: str = "hivemind.heavy_tail_esd_weight_surgeon/1"
SURGEON_RECEIPT_OBJECT: str = "hivemind.heavy_tail_esd_surgery_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_TAIL_QUANTILE: float = 0.25
DEFAULT_ALPHA_LOW: float = 2.0
DEFAULT_ALPHA_HIGH: float = 4.0
DEFAULT_TARGET_MAX_SPIKE_RATIO: float = 2.5
DEFAULT_RANK_COLLAPSE_RATIO: float = 0.20
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_MAX_ALPHA_CAP: float = 100.0

VALID_TARGET_FACTORS: frozenset[str] = frozenset(
    {"w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"}
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"(?:blocks|layers)\.(\d+)\.")


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
    """Validate a finite float inside ``[0, 1]`` with configurable endpoints."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def _validate_finite_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D finite floating-point tensor and return CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    out = tensor.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_square_projector(proj: Any, name: str) -> torch.Tensor:
    """Validate a non-empty square 2D floating-point nullspace projector in CPU ``float64``."""
    mat = _validate_finite_2d(proj, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {tuple(mat.shape)}")
    return (0.5 * (mat + mat.T)).contiguous()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` into deterministic UTF-8 JSON without NaN or Inf."""
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


def verify_esd_surgery_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
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


@dataclass(frozen=True)
class HeavyTailESDSurgeonConfig:
    """Validated configuration for activation-free heavy-tail ESD weight surgery."""

    tail_quantile: float = DEFAULT_TAIL_QUANTILE
    alpha_low: float = DEFAULT_ALPHA_LOW
    alpha_high: float = DEFAULT_ALPHA_HIGH
    target_max_spike_ratio: float = DEFAULT_TARGET_MAX_SPIKE_RATIO
    rank_collapse_ratio_threshold: float = DEFAULT_RANK_COLLAPSE_RATIO
    preserve_bulk_energy: bool = True
    only_regularize_traps: bool = True
    protect_general_expert: bool = False
    general_expert_index: int = 0
    target_factors: tuple[str, ...] = ("w1", "w2", "w3")
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = False
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    require_energy_improvement: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = 0
    batch_size: int = 24
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.tail_quantile, "tail_quantile", allow_zero=False, allow_one=True)
        a_lo = _validate_positive_finite(self.alpha_low, "alpha_low")
        a_hi = _validate_positive_finite(self.alpha_high, "alpha_high")
        if a_lo >= a_hi:
            raise ValueError(f"alpha_low ({a_lo}) must be < alpha_high ({a_hi})")
        _validate_positive_finite(self.target_max_spike_ratio, "target_max_spike_ratio")
        _validate_unit_interval(
            self.rank_collapse_ratio_threshold,
            "rank_collapse_ratio_threshold",
            allow_zero=True,
            allow_one=False,
        )
        _validate_bool(self.preserve_bulk_energy, "preserve_bulk_energy")
        _validate_bool(self.only_regularize_traps, "only_regularize_traps")
        _validate_bool(self.protect_general_expert, "protect_general_expert")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        if (
            isinstance(self.target_factors, (str, bytes))
            or not isinstance(self.target_factors, Sequence)
            or len(self.target_factors) == 0
        ):
            raise ValueError("target_factors must be a non-empty sequence of factor names")
        unknown = sorted({str(f) for f in self.target_factors} - VALID_TARGET_FACTORS)
        if unknown:
            raise ValueError(f"Unsupported target_factors: {unknown}")
        _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.require_energy_improvement, "require_energy_improvement")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Core Activation-Free ESD Power-Law Diagnosis & Winsorization Primitives
# ---------------------------------------------------------------------------


def _fit_csn_hill_alpha(
    pos_eigs_desc: torch.Tensor,
    tail_q: float,
) -> tuple[float, float, int, float]:
    """Fit Clauset-Shalizi Hill MLE power-law exponent on ``>= 2`` descending eigenvalues."""
    n = int(pos_eigs_desc.numel())
    k_max = max(2, min(n, int(math.ceil(float(n) * tail_q))))
    k_min = min(k_max, max(2, min(4, k_max)))

    best_ks = float("inf")
    best_alpha = DEFAULT_MAX_ALPHA_CAP
    best_xmin = float(pos_eigs_desc[k_max - 1].item())
    best_k = k_max
    found_variation = False

    for k in range(k_min, k_max + 1):
        tail = pos_eigs_desc[:k]
        xmin = tail[-1]
        denom = float(torch.log(tail[:-1] / xmin).sum().item())
        if denom <= 1e-12:
            continue
        found_variation = True
        alpha_cand = 1.0 + float(k - 1) / denom
        emp_sf = torch.arange(1, k + 1, dtype=torch.float64) / float(k)
        theo_sf = (tail / xmin).pow(-(alpha_cand - 1.0))
        ks = float((emp_sf - theo_sf).abs().max().item())
        if ks < best_ks or (abs(ks - best_ks) <= 1e-12 and k > best_k):
            best_ks = ks
            best_alpha = alpha_cand
            best_xmin = float(xmin.item())
            best_k = k

    if not found_variation:
        xmin_all = pos_eigs_desc[-1]
        denom_all = float(torch.log(pos_eigs_desc[:-1] / xmin_all).sum().item())
        if denom_all > 1e-12:
            best_alpha = 1.0 + float(n - 1) / denom_all
            best_xmin = float(xmin_all.item())
            best_k = n
            best_ks = 0.0
        else:
            best_ks = 0.0

    best_alpha = min(DEFAULT_MAX_ALPHA_CAP, max(1.0, float(best_alpha)))
    return best_alpha, best_xmin, best_k, float(best_ks)


@torch.no_grad()
def fit_esd_power_law_metrics(
    weight_matrix: torch.Tensor,
    *,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    alpha_low: float = DEFAULT_ALPHA_LOW,
    alpha_high: float = DEFAULT_ALPHA_HIGH,
    spike_ratio_threshold: float = DEFAULT_TARGET_MAX_SPIKE_RATIO,
    rank_collapse_ratio_threshold: float = DEFAULT_RANK_COLLAPSE_RATIO,
) -> dict[str, Any]:
    """Compute exact singular spectrum, entropy effective rank, stable rank, and Hill MLE ``pl_alpha``.

    Follows Martin & Mahoney (JMLR 2021) WeightWatcher Heavy-Tail Self-Regularization:
    evaluates Roy-Vetterli entropy effective rank ``exp(H(sigma / ||sigma||_1))``,
    Frobenius stable rank ``||W||_F^2 / ||W||_2^2``, Clauset-Shalizi Hill MLE exponent
    ``pl_alpha`` on the empirical spectral density ``lambda_i = sigma_i^2``,
    Marchenko-Pastur trimmed bulk edge ``bulk_edge``, and correlation-trap / rank-collapse flags.
    """
    w64 = _validate_finite_2d(weight_matrix, "weight_matrix")
    tail_q = _validate_unit_interval(tail_quantile, "tail_quantile", allow_zero=False, allow_one=True)
    a_lo = _validate_positive_finite(alpha_low, "alpha_low")
    a_hi = _validate_positive_finite(alpha_high, "alpha_high")
    if a_lo >= a_hi:
        raise ValueError(f"alpha_low ({a_lo}) must be < alpha_high ({a_hi})")
    spike_thr = _validate_positive_finite(spike_ratio_threshold, "spike_ratio_threshold")
    rc_thr = _validate_unit_interval(
        rank_collapse_ratio_threshold,
        "rank_collapse_ratio_threshold",
        allow_zero=True,
        allow_one=False,
    )

    m_dim, n_dim = int(w64.shape[0]), int(w64.shape[1])
    d_min, d_max = min(m_dim, n_dim), max(m_dim, n_dim)
    sv = torch.linalg.svdvals(w64)
    ev = sv.pow(2)

    s_max = float(sv[0].item())
    s_min = float(sv[-1].item())
    frob_norm = float(ev.sum().sqrt().item())

    if s_max <= 1e-12:
        return {
            "shape": [m_dim, n_dim],
            "min_dim": d_min,
            "max_dim": d_max,
            "active_rank": 0,
            "singular_values": [float(x) for x in sv.tolist()],
            "eigenvalues": [float(x) for x in ev.tolist()],
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "lambda_max": 0.0,
            "lambda_min": 0.0,
            "frobenius_norm": 0.0,
            "spectral_norm": 0.0,
            "spectral_entropy": 0.0,
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "normalized_stable_rank": 0.0,
            "pl_alpha": 1.0,
            "tail_quantile": tail_q,
            "tail_count": 0,
            "tail_xmin": 0.0,
            "ks_distance": 0.0,
            "bulk_edge": 0.0,
            "spike_ratio": 0.0,
            "max_tail_step_ratio": 1.0,
            "correlation_trap": True,
            "rank_collapsed": True,
            "is_healthy_heavy_tail": False,
            "needs_surgery": True,
            "diagnostic_regime": "zero_matrix",
        }

    pos_mask = ev > max(1e-18, (s_max * s_max) * 1e-11)
    pos_eigs = ev[pos_mask]
    pos_sv = sv[pos_mask]
    k_pos = int(pos_eigs.numel())

    # Roy-Vetterli (2007) entropy effective rank: exp(-sum p_i log p_i) for p_i = sigma_i / sum(sigma)
    probs = pos_sv / pos_sv.sum().clamp_min(1e-30)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())
    effective_rank = float(math.exp(max(0.0, entropy)))
    stable_rank = float(ev.sum().item() / max(float(ev[0].item()), 1e-30))

    # Effective aspect ratio q_eff handles both full-rank matrices and rank-r LoRA products cleanly
    q_eff = float(k_pos) / float(d_max)
    mp_factor = (1.0 + math.sqrt(q_eff)) ** 2

    if k_pos == 1:
        bulk_edge = float(pos_eigs[0].item())
        spike_ratio = 1.0
        max_tail_step_ratio = 1.0
        pl_alpha, tail_xmin, tail_count, ks_dist = 1.0, bulk_edge, 1, 0.0
    else:
        n_trim = max(1, min(k_pos - 1, int(math.ceil(float(k_pos) * 0.15))))
        bulk_mean = float(pos_eigs[n_trim:].mean().item())
        bulk_edge = max(bulk_mean * mp_factor, float(pos_eigs[-1].item()))
        spike_ratio = float(pos_eigs[0].item()) / max(bulk_edge, 1e-18)
        step_slice = pos_eigs[: min(k_pos, n_trim + 1)]
        step_ratios = step_slice[:-1] / step_slice[1:].clamp_min(1e-18)
        max_tail_step_ratio = float(step_ratios.max().item()) if step_ratios.numel() > 0 else 1.0
        pl_alpha, tail_xmin, tail_count, ks_dist = _fit_csn_hill_alpha(pos_eigs, tail_q)

    rank_collapsed = bool(
        d_min > 1
        and (
            k_pos <= 1
            or stable_rank <= 1.5
            or (effective_rank / float(k_pos) < rc_thr)
        )
    )
    has_alpha_trap = bool(k_pos >= 2 and pl_alpha < a_lo)
    has_spike_outlier = bool(
        k_pos >= 2
        and spike_ratio > spike_thr
        and (pl_alpha < a_lo or max_tail_step_ratio > spike_thr)
    )
    correlation_trap = bool((d_min > 1 and k_pos == 1) or has_alpha_trap or has_spike_outlier)
    needs_surgery = bool(correlation_trap or rank_collapsed)
    is_healthy_heavy_tail = bool((not needs_surgery) and (a_lo <= pl_alpha <= a_hi))

    if rank_collapsed:
        regime = "rank_collapsed"
    elif correlation_trap:
        regime = "correlation_trap"
    elif is_healthy_heavy_tail:
        regime = "healthy_heavy_tail"
    else:
        regime = "bulk_random_like"

    return {
        "shape": [m_dim, n_dim],
        "min_dim": d_min,
        "max_dim": d_max,
        "active_rank": k_pos,
        "singular_values": [float(x) for x in sv.tolist()],
        "eigenvalues": [float(x) for x in ev.tolist()],
        "sigma_max": s_max,
        "sigma_min": s_min,
        "lambda_max": float(ev[0].item()),
        "lambda_min": float(pos_eigs[-1].item()),
        "frobenius_norm": frob_norm,
        "spectral_norm": s_max,
        "spectral_entropy": entropy,
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / float(d_min),
        "stable_rank": stable_rank,
        "normalized_stable_rank": stable_rank / float(d_min),
        "pl_alpha": float(pl_alpha),
        "tail_quantile": tail_q,
        "tail_count": int(tail_count),
        "tail_xmin": float(tail_xmin),
        "ks_distance": float(ks_dist),
        "bulk_edge": float(bulk_edge),
        "spike_ratio": float(spike_ratio),
        "max_tail_step_ratio": float(max_tail_step_ratio),
        "correlation_trap": correlation_trap,
        "rank_collapsed": rank_collapsed,
        "is_healthy_heavy_tail": is_healthy_heavy_tail,
        "needs_surgery": needs_surgery,
        "diagnostic_regime": regime,
    }


def _project_adjustment_with_nullspace(
    adj64: torch.Tensor,
    p_null64: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    """Project a spectral weight adjustment ``adj64`` into prose nullspace ``P_0`` if provided."""
    if p_null64 is None:
        return adj64, "none"
    d_p = int(p_null64.shape[0])
    if int(adj64.shape[1]) == d_p:
        return (adj64 @ p_null64).contiguous(), "right_input_nullspace"
    if int(adj64.shape[0]) == d_p:
        return (p_null64 @ adj64).contiguous(), "left_output_nullspace"
    raise ValueError(
        f"p_null shape {tuple(p_null64.shape)} is incompatible with weight_delta shape {tuple(adj64.shape)}"
    )


@torch.no_grad()
def regularize_correlation_trap_spectrum_nullspace(
    weight_delta: torch.Tensor,
    *,
    p_null: torch.Tensor | None = None,
    target_max_spike_ratio: float = DEFAULT_TARGET_MAX_SPIKE_RATIO,
    preserve_bulk_energy: bool = True,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    alpha_low: float = DEFAULT_ALPHA_LOW,
    alpha_high: float = DEFAULT_ALPHA_HIGH,
    rank_collapse_ratio_threshold: float = DEFAULT_RANK_COLLAPSE_RATIO,
    only_regularize_traps: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Surgically winsorize correlation-trapped singular modes with optional ``P_0`` projection.

    Hyper-correlated outlier modes (``pl_alpha < 2.0`` or disconnected spike ratio
    ``lambda_max / bulk_edge > target_max_spike_ratio``) are winsorized down to
    ``sqrt(target_max_spike_ratio * bulk_edge)`` while preserving healthy heavy-tail
    blocks (``pl_alpha in [2, 4]``) and bulk spectral energy (when
    ``preserve_bulk_energy=True``). When ``p_null`` is provided, only the component of
    the adjustment ``W_win - W`` lying inside ``P_0`` is applied so protected keys
    in ``ker(P_0)`` incur zero drift.
    """
    w64 = _validate_finite_2d(weight_delta, "weight_delta")
    p64 = _validate_square_projector(p_null, "p_null") if p_null is not None else None
    spike_cap = _validate_positive_finite(target_max_spike_ratio, "target_max_spike_ratio")
    _validate_bool(preserve_bulk_energy, "preserve_bulk_energy")
    _validate_bool(only_regularize_traps, "only_regularize_traps")

    if p64 is not None:
        d_p = int(p64.shape[0])
        if int(w64.shape[1]) != d_p and int(w64.shape[0]) != d_p:
            raise ValueError(
                f"p_null shape {tuple(p64.shape)} is incompatible with weight_delta shape {tuple(w64.shape)}"
            )

    pre = fit_esd_power_law_metrics(
        w64,
        tail_quantile=tail_quantile,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        spike_ratio_threshold=spike_cap,
        rank_collapse_ratio_threshold=rank_collapse_ratio_threshold,
    )

    if pre["sigma_max"] <= 1e-12:
        return weight_delta.detach().clone(), {
            "repaired": False,
            "noop_reason": "zero_matrix",
            "clipped_modes": 0,
            "nullspace_mode": "none",
            "bulk_energy_ratio": 1.0,
            "pl_alpha_before": pre["pl_alpha"],
            "pl_alpha_after": pre["pl_alpha"],
            "spike_ratio_before": pre["spike_ratio"],
            "spike_ratio_after": pre["spike_ratio"],
            "effective_rank_before": pre["effective_rank"],
            "effective_rank_after": pre["effective_rank"],
            "stable_rank_before": pre["stable_rank"],
            "stable_rank_after": pre["stable_rank"],
        }

    if only_regularize_traps and not pre["needs_surgery"]:
        return weight_delta.detach().clone(), {
            "repaired": False,
            "noop_reason": "healthy_spectrum_preserved",
            "clipped_modes": 0,
            "nullspace_mode": "none",
            "bulk_energy_ratio": 1.0,
            "pl_alpha_before": pre["pl_alpha"],
            "pl_alpha_after": pre["pl_alpha"],
            "spike_ratio_before": pre["spike_ratio"],
            "spike_ratio_after": pre["spike_ratio"],
            "effective_rank_before": pre["effective_rank"],
            "effective_rank_after": pre["effective_rank"],
            "stable_rank_before": pre["stable_rank"],
            "stable_rank_after": pre["stable_rank"],
        }

    u_mat, sv, vh_mat = torch.linalg.svd(w64, full_matrices=False)
    k_pos = int(pre["active_rank"])
    s_max = float(sv[0].item())

    if k_pos == 1:
        if spike_cap > 1.0:
            s_cap = s_max / math.sqrt(spike_cap)
        else:
            s_cap = s_max * math.sqrt(spike_cap)
    else:
        lam_cap = spike_cap * max(float(pre["bulk_edge"]), 1e-18)
        s_cap = math.sqrt(max(lam_cap, 1e-30))
        if pre["pl_alpha"] < alpha_low and s_max <= s_cap:
            tail_idx = max(1, min(k_pos - 1, int(pre["tail_count"]) - 1))
            sv_anchor = float(sv[tail_idx].item())
            s_alpha_cap = sv_anchor * math.exp(
                float(tail_idx) / max(2.0 * (float(alpha_low) - 1.0), 0.5)
            )
            s_cap = min(s_cap, max(s_alpha_cap, float(sv[1].item())))

    clipped_mask = sv > (s_cap + 1e-12)
    clipped_modes = int(clipped_mask.sum().item())

    if clipped_modes == 0:
        return weight_delta.detach().clone(), {
            "repaired": False,
            "noop_reason": "within_spike_cap",
            "clipped_modes": 0,
            "nullspace_mode": "none",
            "bulk_energy_ratio": 1.0,
            "pl_alpha_before": pre["pl_alpha"],
            "pl_alpha_after": pre["pl_alpha"],
            "spike_ratio_before": pre["spike_ratio"],
            "spike_ratio_after": pre["spike_ratio"],
            "effective_rank_before": pre["effective_rank"],
            "effective_rank_after": pre["effective_rank"],
            "stable_rank_before": pre["stable_rank"],
            "stable_rank_after": pre["stable_rank"],
        }

    if preserve_bulk_energy:
        sv_new = torch.minimum(sv, torch.tensor(s_cap, dtype=torch.float64))
    else:
        scale = min(1.0, s_cap / max(s_max, 1e-18))
        sv_new = sv * scale

    bulk_mask = ~clipped_mask
    bulk_energy_before = float(sv[bulk_mask].pow(2).sum().item())
    bulk_energy_after = float(sv_new[bulk_mask].pow(2).sum().item())
    bulk_energy_ratio = (
        bulk_energy_after / bulk_energy_before if bulk_energy_before > 1e-30 else 1.0
    )

    w_win = (u_mat * sv_new.unsqueeze(0)) @ vh_mat
    adj = w_win - w64
    adj_proj, ns_mode = _project_adjustment_with_nullspace(adj, p64)
    w_reg = (w64 + adj_proj).contiguous()

    post = fit_esd_power_law_metrics(
        w_reg,
        tail_quantile=tail_quantile,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        spike_ratio_threshold=spike_cap,
        rank_collapse_ratio_threshold=rank_collapse_ratio_threshold,
    )

    out_tensor = w_reg.to(device=weight_delta.device, dtype=weight_delta.dtype).contiguous()
    return out_tensor, {
        "repaired": bool(float((w_reg - w64).abs().max().item()) > 1e-12),
        "noop_reason": None,
        "clipped_modes": clipped_modes,
        "singular_cap": float(s_cap),
        "nullspace_mode": ns_mode,
        "bulk_energy_ratio": float(bulk_energy_ratio),
        "pl_alpha_before": pre["pl_alpha"],
        "pl_alpha_after": post["pl_alpha"],
        "spike_ratio_before": pre["spike_ratio"],
        "spike_ratio_after": post["spike_ratio"],
        "effective_rank_before": pre["effective_rank"],
        "effective_rank_after": post["effective_rank"],
        "stable_rank_before": pre["stable_rank"],
        "stable_rank_after": post["stable_rank"],
    }


# ---------------------------------------------------------------------------
# 3. Model Traversal, Evaluation, ANSE Gate & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _resolve_nullspace_for_unit(
    unit_name: str,
    shape: tuple[int, int],
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Resolve optional null-space projector tensor matching a surgical unit."""
    if p_null is None:
        return None
    if isinstance(p_null, torch.Tensor):
        d_p = int(p_null.shape[0])
        if shape[1] == d_p or shape[0] == d_p:
            return p_null
        return None
    if not isinstance(p_null, Mapping):
        raise TypeError("p_null must be a torch.Tensor, Mapping[str, torch.Tensor], or None")

    if unit_name in p_null:
        return p_null[unit_name]

    layer_match = _LAYER_KEY_RE.search(unit_name)
    layer_idx_str = layer_match.group(1) if layer_match is not None else None
    is_act = unit_name.endswith((".w3", ".down_proj", "w3", "down_proj"))
    role = "act" if is_act else "x"

    candidates: list[str] = []
    if layer_idx_str is not None:
        candidates.extend([f"blocks.{layer_idx_str}.{role}", f"layers.{layer_idx_str}.{role}"])
    candidates.extend([role, "default"])

    for cand in candidates:
        if cand in p_null:
            mat = p_null[cand]
            d_p = int(mat.shape[0])
            if shape[1] == d_p or shape[0] == d_p:
                return mat
    return None


def _factor_lora_delta_into_pair(
    delta_w: torch.Tensor,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    scaling: float,
) -> None:
    """Factor regularized ``delta_w / scaling`` into exact rank-``r`` ``(w_b, w_a)`` in-place."""
    rank = int(w_a.shape[0])
    target64 = delta_w.detach().to(device="cpu", dtype=torch.float64) / float(scaling)
    u_mat, sv, vh_mat = torch.linalg.svd(target64, full_matrices=False)
    k_eff = min(rank, int(sv.numel()))
    s_sqrt = sv[:k_eff].clamp_min(0.0).sqrt()

    b_new = torch.zeros((int(w_b.shape[0]), rank), dtype=torch.float64)
    a_new = torch.zeros((rank, int(w_a.shape[1])), dtype=torch.float64)
    b_new[:, :k_eff] = u_mat[:, :k_eff] * s_sqrt.unsqueeze(0)
    a_new[:k_eff, :] = s_sqrt.unsqueeze(1) * vh_mat[:k_eff, :]

    w_b.copy_(b_new.to(device=w_b.device, dtype=w_b.dtype))
    w_a.copy_(a_new.to(device=w_a.device, dtype=w_a.dtype))


def _resolve_base_weight(
    param_key_candidates: Sequence[str],
    shape: tuple[int, int],
    base_state_dict: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Look up matching dense baseline weight from optional ``base_state_dict``."""
    if base_state_dict is None:
        return None
    for key in param_key_candidates:
        if key in base_state_dict:
            t = base_state_dict[key]
            if tuple(t.shape) == shape:
                return t.detach().to(device="cpu", dtype=torch.float64)
    return None


def _surgery_on_dense_weight_tensor(
    unit_name: str,
    weight_param: torch.Tensor,
    base_weight: torch.Tensor | None,
    *,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    cfg: HeavyTailESDSurgeonConfig,
) -> dict[str, Any]:
    """Diagnose and optionally repair a dense 2D parameter (or its delta over ``base_weight``)."""
    w64 = weight_param.detach().to(device="cpu", dtype=torch.float64)
    target_mat = (w64 - base_weight) if base_weight is not None else w64
    proj = _resolve_nullspace_for_unit(
        unit_name, (int(target_mat.shape[0]), int(target_mat.shape[1])), p_null
    )
    reg_mat, diag = regularize_correlation_trap_spectrum_nullspace(
        target_mat,
        p_null=proj,
        target_max_spike_ratio=cfg.target_max_spike_ratio,
        preserve_bulk_energy=cfg.preserve_bulk_energy,
        tail_quantile=cfg.tail_quantile,
        alpha_low=cfg.alpha_low,
        alpha_high=cfg.alpha_high,
        rank_collapse_ratio_threshold=cfg.rank_collapse_ratio_threshold,
        only_regularize_traps=cfg.only_regularize_traps,
    )
    if diag["repaired"]:
        updated64 = (base_weight + reg_mat) if base_weight is not None else reg_mat
        weight_param.copy_(updated64.to(device=weight_param.device, dtype=weight_param.dtype))
    diag["unit"] = unit_name
    return diag


def _surgery_on_lora_pair(
    unit_name: str,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    scaling: float,
    *,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    cfg: HeavyTailESDSurgeonConfig,
) -> dict[str, Any]:
    """Diagnose and optionally repair a rank-``r`` LoRA product ``scaling * (w_b @ w_a)``."""
    b64 = w_b.detach().to(device="cpu", dtype=torch.float64)
    a64 = w_a.detach().to(device="cpu", dtype=torch.float64)
    delta_w = (b64 @ a64) * float(scaling)
    proj = _resolve_nullspace_for_unit(
        unit_name, (int(delta_w.shape[0]), int(delta_w.shape[1])), p_null
    )
    reg_delta, diag = regularize_correlation_trap_spectrum_nullspace(
        delta_w,
        p_null=proj,
        target_max_spike_ratio=cfg.target_max_spike_ratio,
        preserve_bulk_energy=cfg.preserve_bulk_energy,
        tail_quantile=cfg.tail_quantile,
        alpha_low=cfg.alpha_low,
        alpha_high=cfg.alpha_high,
        rank_collapse_ratio_threshold=cfg.rank_collapse_ratio_threshold,
        only_regularize_traps=cfg.only_regularize_traps,
    )
    if diag["repaired"]:
        _factor_lora_delta_into_pair(reg_delta, w_b, w_a, scaling)
    diag["unit"] = unit_name
    return diag


def _is_expert_targeted(
    e_idx: int,
    target_expert_set: set[int] | None,
    cfg: HeavyTailESDSurgeonConfig,
) -> bool:
    """Return True iff expert ``e_idx`` is eligible for surgical inspection."""
    if target_expert_set is not None:
        return e_idx in target_expert_set
    if cfg.protect_general_expert and e_idx == cfg.general_expert_index:
        return False
    return True


def _run_esd_surgery_on_mlp(
    layer_prefix: str,
    mlp_mod: nn.Module,
    *,
    target_expert_set: set[int] | None,
    target_factor_set: set[str],
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    base_state_dict: Mapping[str, torch.Tensor] | None,
    cfg: HeavyTailESDSurgeonConfig,
) -> list[dict[str, Any]]:
    """Scan and repair one MoE or dense MLP module, returning per-unit ESD reports."""
    reports: list[dict[str, Any]] = []
    if isinstance(mlp_mod, CrowdLoRAMoEMLP):
        for e_key in sorted(mlp_mod.experts.keys(), key=int):
            e_idx = int(e_key)
            if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                continue
            exp = mlp_mod.experts[e_key]
            for f_name in ("w1", "w2", "w3"):
                if f_name not in target_factor_set:
                    continue
                w_b = getattr(exp, f"{f_name}_B")
                w_a = getattr(exp, f"{f_name}_A")
                reports.append(
                    _surgery_on_lora_delta_pair_checked(
                        f"{layer_prefix}.experts.{e_key}.{f_name}",
                        w_b,
                        w_a,
                        float(getattr(exp, "scaling", 1.0)),
                        p_null=p_null,
                        cfg=cfg,
                    )
                )
        return reports

    if isinstance(mlp_mod, CrowdFineGrainedMoEMLP):
        for e_idx in range(mlp_mod.num_experts):
            if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                continue
            for f_name, mod_list in (
                ("w1", mlp_mod.experts_w1),
                ("w2", mlp_mod.experts_w2),
                ("w3", mlp_mod.experts_w3),
            ):
                if f_name not in target_factor_set:
                    continue
                unit_name = f"{layer_prefix}.experts.{e_idx}.{f_name}"
                w_param = mod_list[e_idx].weight
                base_w = _resolve_base_weight(
                    (
                        f"{unit_name}.weight",
                        f"{layer_prefix}.experts_{f_name}.{e_idx}.weight",
                        f"experts_{f_name}.{e_idx}.weight",
                    ),
                    (int(w_param.shape[0]), int(w_param.shape[1])),
                    base_state_dict,
                )
                reports.append(
                    _surgery_on_dense_weight_tensor(
                        unit_name, w_param, base_w, p_null=p_null, cfg=cfg
                    )
                )
        return reports

    if hasattr(mlp_mod, "experts") and isinstance(mlp_mod.experts, (nn.ModuleDict, nn.ModuleList)):
        expert_items = (
            [(k, mlp_mod.experts[k]) for k in sorted(mlp_mod.experts.keys(), key=int)]
            if isinstance(mlp_mod.experts, nn.ModuleDict)
            else [(str(i), mod) for i, mod in enumerate(mlp_mod.experts)]
        )
        for e_key, exp in expert_items:
            if not _is_expert_targeted(int(e_key), target_expert_set, cfg):
                continue
            for f_name in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
                if f_name not in target_factor_set:
                    continue
                sub = getattr(exp, f_name, None)
                if isinstance(sub, nn.Linear):
                    unit_name = f"{layer_prefix}.experts.{e_key}.{f_name}"
                    base_w = _resolve_base_weight(
                        (f"{unit_name}.weight",),
                        (int(sub.weight.shape[0]), int(sub.weight.shape[1])),
                        base_state_dict,
                    )
                    reports.append(
                        _surgery_on_dense_weight_tensor(
                            unit_name, sub.weight, base_w, p_null=p_null, cfg=cfg
                        )
                    )
        return reports

    if isinstance(mlp_mod, nn.Linear):
        base_w = _resolve_base_weight(
            (f"{layer_prefix}.weight", "weight"),
            (int(mlp_mod.weight.shape[0]), int(mlp_mod.weight.shape[1])),
            base_state_dict,
        )
        reports.append(
            _surgery_on_dense_weight_tensor(
                f"{layer_prefix}.weight", mlp_mod.weight, base_w, p_null=p_null, cfg=cfg
            )
        )
        return reports

    matched_dense = False
    for f_name in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
        sub = getattr(mlp_mod, f_name, None)
        if isinstance(sub, nn.Linear) and f_name in target_factor_set:
            matched_dense = True
            unit_name = f"{layer_prefix}.{f_name}"
            base_w = _resolve_base_weight(
                (f"{unit_name}.weight", f"{f_name}.weight"),
                (int(sub.weight.shape[0]), int(sub.weight.shape[1])),
                base_state_dict,
            )
            reports.append(
                _surgery_on_dense_weight_tensor(
                    unit_name, sub.weight, base_w, p_null=p_null, cfg=cfg
                )
            )
    if not matched_dense:
        raise ValueError(f"Unsupported MLP module type {type(mlp_mod).__name__} for ESD surgery")
    return reports


def _surgery_on_lora_delta_pair_checked(
    unit_name: str,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    scaling: float,
    *,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    cfg: HeavyTailESDSurgeonConfig,
) -> dict[str, Any]:
    """Skip uninitialized zero-B LoRA adapters cleanly without wasting an SVD."""
    if float(w_b.detach().abs().max().item()) <= 1e-12 or float(w_a.detach().abs().max().item()) <= 1e-12:
        return {
            "unit": unit_name,
            "repaired": False,
            "noop_reason": "zero_adapter",
            "clipped_modes": 0,
            "nullspace_mode": "none",
            "bulk_energy_ratio": 1.0,
            "pl_alpha_before": 1.0,
            "pl_alpha_after": 1.0,
            "spike_ratio_before": 0.0,
            "spike_ratio_after": 0.0,
            "effective_rank_before": 0.0,
            "effective_rank_after": 0.0,
            "stable_rank_before": 0.0,
            "stable_rank_after": 0.0,
        }
    return _surgery_on_lora_pair(unit_name, w_b, w_a, scaling, p_null=p_null, cfg=cfg)


def _validate_eval_chunks(
    eval_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    score_from: int,
) -> tuple[torch.Tensor, list[str]]:
    """Validate evaluation chunks and return ``(ids_tensor, expert_ids)``."""
    if isinstance(eval_chunks, (str, bytes)) or not isinstance(eval_chunks, Sequence) or not eval_chunks:
        raise ValueError("eval_chunks must be a non-empty sequence of chunk mappings")
    seq_lens: set[int] = set()
    ids_rows: list[list[int]] = []
    expert_ids: list[str] = []
    for idx, ch in enumerate(eval_chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch or "expert_id" not in ch:
            raise ValueError(f"eval_chunks[{idx}] must be a mapping containing 'ids' and 'expert_id'")
        row = [int(t) for t in ch["ids"]]
        seq_lens.add(len(row))
        ids_rows.append(row)
        expert_ids.append(str(ch["expert_id"]))
    if len(seq_lens) != 1:
        raise ValueError("All eval_chunks must share identical sequence lengths")
    seq_len = next(iter(seq_lens))
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} is too short for score_from={score_from}")
    if not any(eid.startswith(general_prefix) for eid in expert_ids):
        raise ValueError(f"eval_chunks must include at least one '{general_prefix}*' general chunk")
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
    """Compute per-domain mean causal NLL on ``CrowdV1WithEngram``."""
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
        raise TypeError("eval_fn must return a mapping of domain -> NLL or (nll_mean, domain_map)")

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
    cfg: HeavyTailESDSurgeonConfig,
) -> dict[str, Any]:
    """Compute general-domain guard clearance, worst-domain ratio, and ANSE energy."""
    keys = sorted(ref_nll.keys())
    if set(cand_nll.keys()) != set(keys):
        raise ValueError("Candidate and reference domain NLL mappings must share identical keys")

    gen_keys = [k for k in keys if k.startswith(cfg.general_prefix)]
    if not gen_keys:
        raise ValueError(f"Missing required general-domain key starting with {cfg.general_prefix!r}")

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
    general_guard_ok = bool(gen_ratio <= cfg.domain_guard_max_ratio + 1e-12)
    worst_guard_ok = bool(worst_ratio <= cfg.domain_guard_max_ratio + 1e-12)
    guard_ok = bool(general_guard_ok and (worst_guard_ok if cfg.enforce_all_domain_guard else True))

    mean_diff = float(sum(diffs) / float(len(diffs)))
    nll_mean = float(sum(float(cand_nll[k]) for k in keys) / float(len(keys)))
    base_nll_mean = float(sum(float(ref_nll[k]) for k in keys) / float(len(keys)))

    if not guard_ok:
        anse_energy = float(cfg.maximum_pain_energy)
    else:
        soft_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0
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


def _resolve_surgery_config(
    config: HeavyTailESDSurgeonConfig | None,
    *,
    tail_quantile: float | None,
    target_max_spike_ratio: float | None,
    preserve_bulk_energy: bool | None,
    only_regularize_traps: bool | None,
    protect_general_expert: bool | None,
    domain_guard_max_ratio: float | None,
    enforce_all_domain_guard: bool | None,
    require_energy_improvement: bool | None,
    score_from: int | None,
    inplace: bool | None,
) -> HeavyTailESDSurgeonConfig:
    """Merge optional keyword overrides into a validated ``HeavyTailESDSurgeonConfig``."""
    if config is not None and not isinstance(config, HeavyTailESDSurgeonConfig):
        raise TypeError(
            f"config must be HeavyTailESDSurgeonConfig or None, got {type(config).__name__}"
        )
    base = config or HeavyTailESDSurgeonConfig()
    return HeavyTailESDSurgeonConfig(
        tail_quantile=base.tail_quantile if tail_quantile is None else tail_quantile,
        alpha_low=base.alpha_low,
        alpha_high=base.alpha_high,
        target_max_spike_ratio=(
            base.target_max_spike_ratio
            if target_max_spike_ratio is None
            else target_max_spike_ratio
        ),
        rank_collapse_ratio_threshold=base.rank_collapse_ratio_threshold,
        preserve_bulk_energy=(
            base.preserve_bulk_energy if preserve_bulk_energy is None else preserve_bulk_energy
        ),
        only_regularize_traps=(
            base.only_regularize_traps if only_regularize_traps is None else only_regularize_traps
        ),
        protect_general_expert=(
            base.protect_general_expert
            if protect_general_expert is None
            else protect_general_expert
        ),
        general_expert_index=base.general_expert_index,
        target_factors=base.target_factors,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio
            if domain_guard_max_ratio is None
            else domain_guard_max_ratio
        ),
        enforce_all_domain_guard=(
            base.enforce_all_domain_guard
            if enforce_all_domain_guard is None
            else enforce_all_domain_guard
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        require_energy_improvement=(
            base.require_energy_improvement
            if require_energy_improvement is None
            else require_energy_improvement
        ),
        general_prefix=base.general_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


def _discover_mlp_blocks(model: nn.Module) -> tuple[list[tuple[int, nn.Module]], bool]:
    """Locate transformer MLP blocks on ``CrowdV1WithEngram``, HF models, or standalone modules."""
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        if not isinstance(blocks, (nn.ModuleList, Sequence)) or len(blocks) == 0:
            raise ValueError("model.blocks must be a non-empty sequence of transformer blocks")
        out = []
        for idx, blk in enumerate(blocks):
            mlp = getattr(blk, "mlp", None)
            if not isinstance(mlp, nn.Module):
                raise ValueError(f"model.blocks[{idx}] must expose an nn.Module 'mlp'")
            out.append((idx, mlp))
        return out, True

    hf_layers = getattr(getattr(model, "model", model), "layers", None)
    if hf_layers is not None:
        if not isinstance(hf_layers, (nn.ModuleList, Sequence)) or len(hf_layers) == 0:
            raise ValueError("model.layers must be a non-empty sequence of transformer layers")
        out = []
        for idx, layer in enumerate(hf_layers):
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, nn.Module):
                raise ValueError(f"model.layers[{idx}] must expose an nn.Module 'mlp'")
            out.append((idx, mlp))
        return out, False

    return [(0, model)], False


@torch.no_grad()
def execute_verified_esd_moe_surgery(
    model: nn.Module,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    *,
    config: HeavyTailESDSurgeonConfig | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    base_state_dict: Mapping[str, torch.Tensor] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_factors: Sequence[str] | None = None,
    tail_quantile: float | None = None,
    target_max_spike_ratio: float | None = None,
    preserve_bulk_energy: bool | None = None,
    only_regularize_traps: bool | None = None,
    protect_general_expert: bool | None = None,
    domain_guard_max_ratio: float | None = None,
    enforce_all_domain_guard: bool | None = None,
    require_energy_improvement: bool | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> dict[str, Any]:
    """Scan all MoE experts/layers activation-free, repair correlation traps, and gate on ANSE.

    Commits edits if and only if ``Delta E < 0`` (when ``require_energy_improvement=True``)
    and general-domain drift ``<= +1.0%`` (``general_ratio <= domain_guard_max_ratio``);
    otherwise restores the exact parent checkpoint bit-for-bit and emits a verified
    SHA-256 provenance receipt.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if eval_chunks is None and eval_fn is None:
        raise ValueError("Either eval_chunks or eval_fn must be provided")

    cfg = _resolve_surgery_config(
        config,
        tail_quantile=tail_quantile,
        target_max_spike_ratio=target_max_spike_ratio,
        preserve_bulk_energy=preserve_bulk_energy,
        only_regularize_traps=only_regularize_traps,
        protect_general_expert=protect_general_expert,
        domain_guard_max_ratio=domain_guard_max_ratio,
        enforce_all_domain_guard=enforce_all_domain_guard,
        require_energy_improvement=require_energy_improvement,
        score_from=score_from,
        inplace=inplace,
    )

    factors_seq = cfg.target_factors if target_factors is None else tuple(str(f) for f in target_factors)
    if not factors_seq or set(factors_seq) - VALID_TARGET_FACTORS:
        raise ValueError(f"target_factors must be a non-empty subset of {sorted(VALID_TARGET_FACTORS)}")
    target_factor_set = set(factors_seq)

    target_expert_set = (
        {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
        if expert_indices is not None
        else None
    )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    mlp_blocks, has_crowd_blocks = _discover_mlp_blocks(work_model)
    n_blocks = len(mlp_blocks)

    if layer_indices is None:
        target_layer_set = {idx for idx, _ in mlp_blocks}
    else:
        t_layers = sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if not t_layers or any(idx >= n_blocks for idx in t_layers):
            raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
        target_layer_set = set(t_layers)

    ids_eval: torch.Tensor | None = None
    expert_ids: list[str] = []
    if eval_chunks is not None:
        if eval_fn is None and not (
            has_crowd_blocks
            and hasattr(work_model, "wte")
            and hasattr(work_model, "ln")
            and hasattr(work_model, "head")
        ):
            raise TypeError("eval_chunks without eval_fn requires a CrowdV1WithEngram-compatible model")
        ids_eval, expert_ids = _validate_eval_chunks(
            eval_chunks, general_prefix=cfg.general_prefix, score_from=cfg.score_from
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
                work_model, ids_eval, expert_ids, score_from=cfg.score_from, batch_size=cfg.batch_size
            )

        ref_nll = base_nll_clean if base_nll_clean is not None else pre_nll
        parent_eval = _summarize_anse_metrics(ref_nll, pre_nll, cfg=cfg)

        unit_reports: list[dict[str, Any]] = []
        for l_idx, mlp_mod in mlp_blocks:
            if l_idx not in target_layer_set:
                continue
            prefix = f"blocks.{l_idx}.mlp" if has_crowd_blocks else f"layers.{l_idx}.mlp"
            unit_reports.extend(
                _run_esd_surgery_on_mlp(
                    prefix,
                    mlp_mod,
                    target_expert_set=target_expert_set,
                    target_factor_set=target_factor_set,
                    p_null=p_null,
                    base_state_dict=base_state_dict,
                    cfg=cfg,
                )
            )

        cand_sd = work_model.state_dict()
        candidate_sha = compute_state_sha256(cand_sd)
        total_elements = sum(int(v.numel()) for v in parent_snapshot.values())
        edited_elements = sum(
            int(torch.ne(cand_sd[k].detach().cpu(), parent_snapshot[k]).sum().item())
            for k in parent_snapshot
        )
        edited_param_ratio = float(edited_elements) / float(max(1, total_elements))

        eval_error: str | None = None
        try:
            if eval_fn is not None:
                cand_nll = _parse_eval_domain_nll(eval_fn(work_model))
            else:
                assert ids_eval is not None
                cand_nll = _eval_crowd_model_domain_nlls(
                    work_model, ids_eval, expert_ids, score_from=cfg.score_from, batch_size=cfg.batch_size
                )
            cand_eval = _summarize_anse_metrics(ref_nll, cand_nll, cfg=cfg)
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            cand_eval = {
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

        parent_e = float(parent_eval["anse_energy"])
        cand_e = float(cand_eval["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_energy_improvement else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif eval_error is None and bool(cand_eval["guard_ok"]) and energy_ok:
            status = "ACCEPTED"
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

    active_units = [r for r in unit_reports if r.get("noop_reason") != "zero_adapter"]
    repaired_units = [r for r in active_units if bool(r.get("repaired"))]
    untouched_units = [r for r in active_units if not bool(r.get("repaired"))]

    def _mean_field(items: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(x[key]) for x in items) / float(len(items))) if items else 0.0

    pl_alpha_final = _mean_field(active_units, "pl_alpha_after" if accepted else "pl_alpha_before")
    stable_rank_final = _mean_field(active_units, "stable_rank_after" if accepted else "stable_rank_before")
    effective_rank_final = _mean_field(active_units, "effective_rank_after" if accepted else "effective_rank_before")

    receipt: dict[str, Any] = {
        "object": SURGEON_RECEIPT_OBJECT,
        "schema": SURGEON_SCHEMA,
        "status": status,
        "accepted": accepted,
        "committed": accepted,
        "rolled_back": rolled_back,
        "eval_error": eval_error,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "pl_alpha": pl_alpha_final,
        "pl_alpha_before_mean": _mean_field(active_units, "pl_alpha_before"),
        "pl_alpha_after_mean": _mean_field(active_units, "pl_alpha_after"),
        "stable_rank": stable_rank_final,
        "stable_rank_before_mean": _mean_field(active_units, "stable_rank_before"),
        "stable_rank_after_mean": _mean_field(active_units, "stable_rank_after"),
        "effective_rank": effective_rank_final,
        "anse_energy": cand_e,
        "parent_anse_energy": parent_e,
        "final_anse_energy": cand_e if accepted else parent_e,
        "delta_e": delta_e,
        "general_ratio": cand_eval["general_ratio"],
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "per_domain": cand_eval["per_domain"],
        "units_scanned": len(unit_reports),
        "active_units_scanned": len(active_units),
        "trapped_units_repaired": len(repaired_units),
        "healthy_units_untouched": len(untouched_units),
        "edited_elements": edited_elements,
        "total_elements": total_elements,
        "edited_param_ratio": edited_param_ratio,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "unit_reports": unit_reports,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ALPHA_HIGH",
    "DEFAULT_ALPHA_LOW",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_MAX_ALPHA_CAP",
    "DEFAULT_RANK_COLLAPSE_RATIO",
    "DEFAULT_TAIL_QUANTILE",
    "DEFAULT_TARGET_MAX_SPIKE_RATIO",
    "GENERAL_EXPERT_PREFIX",
    "HeavyTailESDSurgeonConfig",
    "SURGEON_RECEIPT_OBJECT",
    "SURGEON_SCHEMA",
    "VALID_TARGET_FACTORS",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "compute_state_sha256",
    "execute_verified_esd_moe_surgery",
    "fit_esd_power_law_metrics",
    "regularize_correlation_trap_spectrum_nullspace",
    "verify_esd_surgery_receipt_provenance",
]
