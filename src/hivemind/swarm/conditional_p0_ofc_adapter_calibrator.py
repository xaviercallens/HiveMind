# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Conditional Convex P_0 / OFC Multi-Adapter Blender and Signed-Tau Norm Calibrator.

Crystallizes the empirical ``PILOT15`` weight-space discoveries on
``Qwen/Qwen2.5-0.5B-Instruct`` + 3 production PEFT LoRA adapters
(``qwen_lora_ltm_local``, ``qwen-personal-voice-lora``, ``qwen-simpo-reddit-audience``;
locked single-shot virgin vault ``-0.037132`` nats, ECE ``-50.51%``, ``D0 -1.67%``,
Layer-3 ``<1%`` OFC probe AUROC ``0.8643`` with ``IoU = 0.0000`` vs massive-activation
outliers, Instruct ``tau = 0.94 < 1.0`` vs Base ``tau = 1.05 > 1.0``):

1. :func:`diagnose_adapter_d0_alignment_and_ofc_overlap`:
   Audits unguarded multi-adapter ``D0`` prose drift, subspace energy fraction in the
   Ledoit-Wolf prose subspace vs null-space ``P_0``, inter-adapter alignment, and
   early-layer ``<1%`` OFC value-neuron vs massive-activation IoU confound.
2. :func:`compute_conditional_convex_p0_ofc_blend`:
   Applies ``lambda = 0.0`` (zero stripping) when multi-adapter transfer is benign or
   synergistic (``D0 <= trigger_threshold``, e.g., ``+0.5%`` in ``PILOT15``), and
   smoothly ramps convex null-space + OFC ``S_val`` protection
   ``(1 - lambda) * delta_w + lambda * (delta_w @ P_0)_{S_val^perp}`` only when
   unguarded grafting threatens ``D0 > trigger_threshold``.
3. :func:`calibrate_signed_tau_and_crossfit_js_norm`:
   Exploits exact affine logit linearity ``L(tau, alpha) = tau * L_base + alpha * L_dw``
   inside ``ker(G_D0) intersect S_val^perp`` to distinguish under-confident Base regimes
   (``tau > 1.0``) from over-confident RLHF/Instruct regimes (``tau < 1.0``), gated by
   ANSE ``Delta E < 0`` and ``D0 <= +1.0%`` with atomic bitwise rollback and canonical
   SHA-256 provenance receipts.
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
)

CALIBRATOR_SCHEMA: str = "hivemind.conditional_p0_ofc_adapter_calibrator/1"
CALIBRATOR_RECEIPT_OBJECT: str = "hivemind.conditional_p0_ofc_adapter_calibrator_receipt"

DEFAULT_TRIGGER_THRESHOLD: float = 0.005
DEFAULT_FULL_PROTECTION_THRESHOLD: float = 0.010
DEFAULT_P0_ENERGY_KEEP: float = 0.985
DEFAULT_OFC_FRACTION: float = 0.010
DEFAULT_MASSIVE_RATIO_THRESHOLD: float = 5.0
DEFAULT_JS_MAX_RANK: int = 4
DEFAULT_JS_MAX_STEP_NORM: float = 0.08
DEFAULT_NULLSPACE_SV_TOL: float = 1e-6
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.010
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_MIN_ENERGY_DROP: float = 1e-6
DEFAULT_NUM_BINS: int = 15
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEED: int = 20261127
DEFAULT_SIGNED_TAU_GRID: tuple[float, ...] = (
    0.70,
    0.80,
    0.88,
    0.94,
    0.97,
    1.00,
    1.03,
    1.05,
    1.10,
    1.18,
    1.30,
    1.45,
    1.65,
)
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0, 1.5)
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
_EARLY_LAYER_CANDIDATES: tuple[str, ...] = (
    "model.layers.3",
    "model.layers.0",
    "layers.3",
    "layers.0",
    "blocks.0",
    "early_layer",
    "trunk",
)


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
    out: list[float] = []
    for idx, val in enumerate(seq):
        if strictly_positive:
            out.append(_validate_positive_finite(val, f"{name}[{idx}]"))
        elif nonneg:
            out.append(_validate_nonneg_finite(val, f"{name}[{idx}]"))
        else:
            out.append(_validate_finite_float(val, f"{name}[{idx}]"))
    return tuple(sorted(set(out)))


def _validate_general_prefixes(prefixes: Any) -> tuple[str, ...]:
    """Validate non-empty sequence of general-prose domain prefixes."""
    if isinstance(prefixes, (str, bytes, bytearray)) or not isinstance(prefixes, Sequence):
        raise TypeError("general_prefixes must be a non-empty sequence of strings")
    if not prefixes:
        raise ValueError("general_prefixes must be non-empty")
    return tuple(_validate_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(prefixes))


def _validate_index_sequence(indices: Any, dim: int, name: str) -> list[int]:
    """Validate and deduplicate integer channel indices lying in ``[0, dim)``."""
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


@dataclass(frozen=True)
class ConditionalP0OFCCalibratorConfig:
    """Validated configuration for conditional P_0/OFC blending and signed-tau JS calibration."""

    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD
    full_protection_threshold: float = DEFAULT_FULL_PROTECTION_THRESHOLD
    energy_keep: float = DEFAULT_P0_ENERGY_KEEP
    ofc_fraction: float = DEFAULT_OFC_FRACTION
    massive_ratio_threshold: float = DEFAULT_MASSIVE_RATIO_THRESHOLD
    js_max_rank: int = DEFAULT_JS_MAX_RANK
    js_max_step_norm: float = DEFAULT_JS_MAX_STEP_NORM
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    min_energy_drop: float = DEFAULT_MIN_ENERGY_DROP
    num_bins: int = DEFAULT_NUM_BINS
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    tau_grid: tuple[float, ...] = DEFAULT_SIGNED_TAU_GRID
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True
    enable_norm_calibration: bool = True
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        trig = _validate_positive_finite(self.trigger_threshold, "trigger_threshold")
        full_th = _validate_positive_finite(self.full_protection_threshold, "full_protection_threshold")
        if full_th <= trig:
            raise ValueError(
                f"full_protection_threshold ({full_th}) must be > trigger_threshold ({trig})"
            )
        e_keep = _validate_positive_finite(self.energy_keep, "energy_keep")
        if e_keep >= 1.0:
            raise ValueError(f"energy_keep must be < 1.0, got {e_keep}")
        ofc_f = _validate_positive_finite(self.ofc_fraction, "ofc_fraction")
        if ofc_f > 1.0:
            raise ValueError(f"ofc_fraction must be <= 1.0, got {ofc_f}")
        mass_r = _validate_positive_finite(self.massive_ratio_threshold, "massive_ratio_threshold")
        if mass_r <= 1.0:
            raise ValueError(f"massive_ratio_threshold must be > 1.0, got {mass_r}")
        _validate_positive_int(self.js_max_rank, "js_max_rank")
        _validate_positive_finite(self.js_max_step_norm, "js_max_step_norm")
        sv_tol = _validate_positive_finite(self.nullspace_sv_tol, "nullspace_sv_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_tol must be < 1.0, got {sv_tol}")
        _validate_nonneg_finite(self.max_general_drift_rel, "max_general_drift_rel")
        _validate_nonneg_finite(self.max_domain_drift_rel, "max_domain_drift_rel")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.min_energy_drop, "min_energy_drop")
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
        _validate_bool(self.enable_norm_calibration, "enable_norm_calibration")
        _validate_nonneg_int(self.seed, "seed")


def verify_conditional_calibrator_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != CALIBRATOR_RECEIPT_OBJECT or receipt.get("schema") != CALIBRATOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Ledoit-Wolf Null-Space, OFC Discovery & Alignment Diagnostics (PILOT15)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_ledoit_wolf_nullspace_projector(
    acts: torch.Tensor,
    *,
    energy_keep: float = DEFAULT_P0_ENERGY_KEEP,
    center: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute OAS/Ledoit-Wolf shrinkage moment matrix and orthogonal null-space projector ``P_0 = I - V V^T``."""
    acts_t = _validate_fp_tensor(acts, "acts", exact_ndim=2)
    e_keep = _validate_positive_finite(energy_keep, "energy_keep")
    if e_keep >= 1.0:
        raise ValueError(f"energy_keep must be < 1.0, got {e_keep}")
    do_center = _validate_bool(center, "center")

    x = acts_t.detach().to(device="cpu", dtype=torch.float64)
    n, d = int(x.shape[0]), int(x.shape[1])
    x_eff = (x - x.mean(dim=0, keepdim=True)) if do_center else x
    emp_cov = (x_eff.T @ x_eff) / float(max(n, 1))
    tr = float(torch.trace(emp_cov).item())
    mu = tr / float(max(d, 1))
    tr_sq = float((emp_cov * emp_cov).sum().item())
    num = (1.0 - 2.0 / float(d)) * tr_sq + tr * tr
    den = (float(n) + 1.0 - 2.0 / float(d)) * (tr_sq - (tr * tr) / float(max(d, 1)))
    alpha = max(0.0, min(1.0, num / max(den, 1e-12))) if n >= 2 and d >= 2 else 1.0
    cov_shrunk = (1.0 - alpha) * emp_cov + alpha * mu * torch.eye(d, dtype=torch.float64)
    cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)

    evals_asc, evecs_asc = torch.linalg.eigh(cov_shrunk)
    evals = evals_asc.flip(0).clamp_min(0.0)
    evecs = evecs_asc.flip(1)
    iso_floor = alpha * mu
    excess_evals = (evals - iso_floor).clamp_min(0.0)
    total_e = float(excess_evals.sum().item())

    if d <= 1 or total_e <= 1e-30:
        k_preserve = 0
        top1 = 0.0
        top8 = 0.0
    else:
        cum = torch.cumsum(excess_evals, dim=0) / total_e
        k_energy = int(torch.searchsorted(cum, torch.tensor(e_keep, dtype=torch.float64)).item()) + 1
        k_tol = int((excess_evals > 1e-6 * float(excess_evals[0].item())).sum().item())
        k_preserve = min(d - 1, max(k_energy, min(k_tol, max(1, int(0.85 * d)))))
        top1 = float((excess_evals[0] / total_e).item())
        top8 = float((excess_evals[: min(8, d)].sum() / total_e).item())

    v = evecs[:, :k_preserve]
    p0 = torch.eye(d, dtype=torch.float64) - (v @ v.T)
    p0 = 0.5 * (p0 + p0.T)
    return p0, {
        "hidden_dim": d,
        "lw_shrinkage": float(alpha),
        "k_preserve": k_preserve,
        "nullspace_dim": d - k_preserve,
        "top1_sv_energy_frac": top1,
        "top8_sv_energy_frac": top8,
    }


def _compute_auroc(y_true: torch.Tensor, y_preds: torch.Tensor) -> float:
    """Compute exact Mann-Whitney U AUROC without external dependencies."""
    pos = y_preds[y_true == 1]
    neg = y_preds[y_true == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = float((diff > 0).sum().item()) + 0.5 * float((diff == 0).sum().item())
    return wins / float(pos.numel() * neg.numel())


@torch.no_grad()
def discover_ofc_value_neurons(
    hidden_means: torch.Tensor,
    chunk_nlls: torch.Tensor,
    *,
    ofc_fraction: float = DEFAULT_OFC_FRACTION,
    massive_ratio_threshold: float = DEFAULT_MASSIVE_RATIO_THRESHOLD,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Identify ``<1%`` OFC sparse value neurons out-of-fold and audit massive-activation IoU confound."""
    x_t = _validate_fp_tensor(hidden_means, "hidden_means", exact_ndim=2)
    nll_t = _validate_fp_tensor(chunk_nlls, "chunk_nlls", exact_ndim=1)
    n, h = int(x_t.shape[0]), int(x_t.shape[1])
    if int(nll_t.shape[0]) != n:
        raise ValueError(f"chunk_nlls length {nll_t.shape[0]} must match hidden_means rows {n}")
    if n < 4:
        raise ValueError(f"Need at least 4 chunks for 2-fold OOF OFC discovery, got {n}")

    ofc_f = _validate_positive_finite(ofc_fraction, "ofc_fraction")
    if ofc_f > 1.0:
        raise ValueError(f"ofc_fraction must be <= 1.0, got {ofc_f}")
    mass_r = _validate_positive_finite(massive_ratio_threshold, "massive_ratio_threshold")
    if mass_r <= 1.0:
        raise ValueError(f"massive_ratio_threshold must be > 1.0, got {mass_r}")
    seed_i = _validate_nonneg_int(seed, "seed")

    x = x_t.detach().to(device="cpu", dtype=torch.float64)
    nlls = nll_t.detach().to(device="cpu", dtype=torch.float64)
    y = (nlls <= torch.median(nlls)).to(torch.float64)

    gen = torch.Generator(device="cpu").manual_seed(seed_i)
    perm = torch.randperm(n, generator=gen)
    half = n // 2
    tr, va = perm[:half], perm[half:]

    mu = x[tr].mean(dim=0, keepdim=True)
    sd = x[tr].std(dim=0, unbiased=False, keepdim=True) + 1e-6
    x_tr = (x[tr] - mu) / sd
    x_va = (x[va] - mu) / sd

    pos_tr = y[tr] == 1.0
    neg_tr = y[tr] == 0.0
    if bool(pos_tr.any().item()) and bool(neg_tr.any().item()):
        w_contrast = x_tr[pos_tr].mean(dim=0) - x_tr[neg_tr].mean(dim=0)
    else:
        w_contrast = x_tr.abs().mean(dim=0)

    k_ofc = max(1, min(h, int(round(float(h) * ofc_f))))
    _, top_idx = torch.topk(w_contrast.abs(), k=k_ofc, largest=True, sorted=True)

    sub_tr = x_tr[:, top_idx]
    gram = sub_tr.T @ sub_tr + torch.eye(k_ofc, dtype=torch.float64)
    rhs = sub_tr.T @ (2.0 * y[tr] - 1.0)
    w_sub = torch.linalg.solve(gram, rhs)
    oof_preds = x_va[:, top_idx] @ w_sub
    oof_auroc = _compute_auroc(y[va].to(torch.int64), oof_preds)

    chan_rms = x.pow(2).mean(dim=0).sqrt()
    med_rms = float(chan_rms.median().item())
    denom_med = max(med_rms, 1e-12)
    massive_mask = chan_rms > (mass_r * denom_med)
    massive_indices = sorted(
        int(i) for i in torch.nonzero(massive_mask, as_tuple=False).reshape(-1).tolist()
    )
    ofc_indices = sorted(int(i) for i in top_idx.tolist())
    ofc_set = set(ofc_indices)
    massive_set = set(massive_indices)
    union_sz = len(ofc_set | massive_set)
    ofc_iou = float(len(ofc_set & massive_set) / union_sz) if union_sz > 0 else 0.0

    return {
        "hidden_dim": h,
        "k_ofc": k_ofc,
        "ofc_fraction": float(k_ofc) / float(h),
        "ofc_indices": ofc_indices,
        "oof_auroc": float(oof_auroc),
        "massive_activation_indices": massive_indices,
        "massive_activation_count": len(massive_indices),
        "massive_activation_max_ratio": float(chan_rms.max().item()) / denom_med,
        "ofc_iou": ofc_iou,
        "confound_iou_vs_massive_activations": ofc_iou,
        "ofc_disjoint_from_massive": bool(ofc_iou < 0.15),
        "probe_bytes": int(k_ofc * 6),
    }


@torch.no_grad()
def diagnose_adapter_d0_alignment_and_ofc_overlap(
    early_layer_hidden_means: torch.Tensor,
    chunk_nlls: torch.Tensor,
    *,
    base_d0_nll: float,
    merged_d0_nll: float,
    delta_w: torch.Tensor | None = None,
    d0_token_activations: torch.Tensor | None = None,
    adapter_deltas: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    p0: torch.Tensor | None = None,
    ofc_fraction: float = DEFAULT_OFC_FRACTION,
    massive_ratio_threshold: float = DEFAULT_MASSIVE_RATIO_THRESHOLD,
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    energy_keep: float = DEFAULT_P0_ENERGY_KEEP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Audit unguarded multi-adapter ``D0`` drift, ``P_0`` subspace alignment, and OFC vs massive-activation IoU.

    Reproduces the ``PILOT15`` diagnostic finding where synergistic multi-adapter merging yields
    ``D0 < 0`` (safe benign transfer) while early-layer ``<1%`` OFC value neurons remain
    disjoint from massive-activation outliers (``ofc_iou ~ 0.0``).
    """
    base_d0 = _validate_positive_finite(base_d0_nll, "base_d0_nll")
    merged_d0 = _validate_positive_finite(merged_d0_nll, "merged_d0_nll")
    trig = _validate_positive_finite(trigger_threshold, "trigger_threshold")

    ofc_diag = discover_ofc_value_neurons(
        early_layer_hidden_means,
        chunk_nlls,
        ofc_fraction=ofc_fraction,
        massive_ratio_threshold=massive_ratio_threshold,
        seed=seed,
    )

    p0_proj: torch.Tensor | None = None
    spectral_stats: dict[str, Any] | None = None
    if p0 is not None:
        p0_t = _validate_fp_tensor(p0, "p0", exact_ndim=2)
        if p0_t.shape[0] != p0_t.shape[1]:
            raise ValueError(f"p0 must be square, got shape {tuple(p0_t.shape)}")
        p0_proj = p0_t.detach().to(device="cpu", dtype=torch.float64)
    elif d0_token_activations is not None:
        p0_proj, spectral_stats = compute_ledoit_wolf_nullspace_projector(
            d0_token_activations, energy_keep=energy_keep
        )

    delta_alignment: dict[str, Any] | None = None
    if delta_w is not None:
        dw_t = _validate_fp_tensor(delta_w, "delta_w", exact_ndim=2).detach().to(
            device="cpu", dtype=torch.float64
        )
        in_dim = int(dw_t.shape[1])
        total_sq = float((dw_t * dw_t).sum().item())
        frob = math.sqrt(total_sq)
        ofc_cols = [i for i in ofc_diag["ofc_indices"] if i < in_dim]
        if ofc_cols:
            idx_t = torch.tensor(ofc_cols, dtype=torch.long)
            ofc_slice = dw_t[:, idx_t]
            ofc_sq = float((ofc_slice * ofc_slice).sum().item())
            ofc_max_abs = float(ofc_slice.abs().max().item())
        else:
            ofc_sq = 0.0
            ofc_max_abs = 0.0
        delta_alignment = {
            "delta_frob_norm": frob,
            "ofc_column_energy_frac": ofc_sq / max(total_sq, 1e-30) if total_sq > 0.0 else 0.0,
            "ofc_column_max_abs": ofc_max_abs,
        }
        if p0_proj is not None:
            if int(p0_proj.shape[0]) != in_dim:
                raise ValueError(
                    f"p0 dim {p0_proj.shape[0]} must match delta_w input dim {in_dim}"
                )
            dw_null = dw_t @ p0_proj
            null_sq = float((dw_null * dw_null).sum().item())
            null_frac = min(1.0, max(0.0, null_sq / max(total_sq, 1e-30))) if total_sq > 0.0 else 0.0
            delta_alignment["nullspace_energy_frac"] = null_frac
            delta_alignment["prose_subspace_energy_frac"] = max(0.0, 1.0 - null_frac)

    adapter_alignment: dict[str, Any] | None = None
    if adapter_deltas is not None:
        if isinstance(adapter_deltas, Mapping):
            raw_tensors = list(adapter_deltas.values())
        elif isinstance(adapter_deltas, Sequence) and not isinstance(adapter_deltas, (str, bytes, bytearray)):
            raw_tensors = list(adapter_deltas)
        else:
            raise TypeError("adapter_deltas must be a mapping or sequence of 2D tensors")
        if len(raw_tensors) < 2:
            raise ValueError("adapter_deltas must contain at least 2 adapter tensors for alignment audit")
        flat_vecs: list[torch.Tensor] = []
        base_shape: tuple[int, ...] | None = None
        for idx, t in enumerate(raw_tensors):
            vt = _validate_fp_tensor(t, f"adapter_deltas[{idx}]", exact_ndim=2).detach().to(
                device="cpu", dtype=torch.float64
            )
            if base_shape is None:
                base_shape = tuple(vt.shape)
            elif tuple(vt.shape) != base_shape:
                raise ValueError(
                    f"adapter_deltas[{idx}] shape {tuple(vt.shape)} must match {base_shape}"
                )
            flat_vecs.append(vt.reshape(-1))
        cos_vals: list[float] = []
        sign_agrees: list[float] = []
        for i in range(len(flat_vecs)):
            for j in range(i + 1, len(flat_vecs)):
                vi, vj = flat_vecs[i], flat_vecs[j]
                denom = float(vi.norm().item()) * float(vj.norm().item())
                cos_vals.append(float(torch.dot(vi, vj).item()) / denom if denom > 1e-30 else 0.0)
                active = (vi.abs() > 1e-12) & (vj.abs() > 1e-12)
                if bool(active.any().item()):
                    agree = float((torch.sign(vi[active]) == torch.sign(vj[active])).to(torch.float64).mean().item())
                else:
                    agree = 1.0
                sign_agrees.append(agree)
        adapter_alignment = {
            "num_adapters": len(flat_vecs),
            "mean_pairwise_adapter_cosine": sum(cos_vals) / float(len(cos_vals)),
            "min_pairwise_adapter_cosine": min(cos_vals),
            "mean_sign_agreement_frac": sum(sign_agrees) / float(len(sign_agrees)),
        }

    d0_drift_probe = (merged_d0 - base_d0) / base_d0
    if d0_drift_probe <= 0.0:
        transfer_regime = "benign_synergistic"
    elif d0_drift_probe <= trig:
        transfer_regime = "benign_within_trigger"
    else:
        transfer_regime = "hostile_forgetting"

    return {
        "base_d0_nll": base_d0,
        "merged_d0_nll": merged_d0,
        "d0_drift_probe": float(d0_drift_probe),
        "general_drift_ratio": float(d0_drift_probe),
        "trigger_threshold": trig,
        "transfer_regime": transfer_regime,
        "is_synergistic": bool(d0_drift_probe <= 0.0),
        "requires_p0_ofc_protection": bool(d0_drift_probe > trig),
        "ofc_diagnostics": ofc_diag,
        "ofc_indices": ofc_diag["ofc_indices"],
        "oof_auroc": ofc_diag["oof_auroc"],
        "ofc_iou": ofc_diag["ofc_iou"],
        "massive_activation_indices": ofc_diag["massive_activation_indices"],
        "massive_activation_max_ratio": ofc_diag["massive_activation_max_ratio"],
        "ofc_disjoint_from_massive": ofc_diag["ofc_disjoint_from_massive"],
        "probe_bytes": ofc_diag["probe_bytes"],
        "spectral_stats": spectral_stats,
        "delta_alignment": delta_alignment,
        "adapter_alignment": adapter_alignment,
    }


# ---------------------------------------------------------------------------
# 3. Conditional Convex P_0 + OFC S_val Blender (PILOT15 Discovery)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_conditional_convex_p0_ofc_blend(
    delta_w: torch.Tensor,
    p0: torch.Tensor,
    ofc_indices: Sequence[int] | torch.Tensor,
    *,
    d0_drift_probe: float,
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    full_protection_threshold: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply conditional convex ``P_0 + OFC S_val`` blending ONLY when ``D0`` probe drift breaches threshold.

    Computes:
    ``delta_blend = (1 - lambda) * delta_w + lambda * (delta_w @ P_0)_{S_val^perp}``
    where ``lambda = 0.0`` (exact zero stripping, preserving 100% of benign multi-adapter gain
    as in ``PILOT15``) whenever ``d0_drift_probe <= trigger_threshold``, smoothly ramps on
    ``(trigger_threshold, full_protection_threshold)``, and reaches ``lambda = 1.0`` (exact
    ``0.0`` OFC column leak) once ``d0_drift_probe >= full_protection_threshold``.
    """
    dw_t = _validate_fp_tensor(delta_w, "delta_w", exact_ndim=2)
    p0_t = _validate_fp_tensor(p0, "p0", exact_ndim=2)
    in_dim = int(dw_t.shape[1])
    if p0_t.shape != (in_dim, in_dim):
        raise ValueError(f"p0 shape {tuple(p0_t.shape)} must match ({in_dim}, {in_dim})")

    ofc_clean = _validate_index_sequence(ofc_indices, in_dim, "ofc_indices")
    d0_drift = _validate_finite_float(d0_drift_probe, "d0_drift_probe")
    trig = _validate_positive_finite(trigger_threshold, "trigger_threshold")
    full_th = (
        max(2.0 * trig, trig + 0.005)
        if full_protection_threshold is None
        else _validate_positive_finite(full_protection_threshold, "full_protection_threshold")
    )
    if full_th <= trig:
        raise ValueError(
            f"full_protection_threshold ({full_th}) must be > trigger_threshold ({trig})"
        )

    if d0_drift <= trig:
        lam = 0.0
        mode = "unstripped_benign"
    elif d0_drift >= full_th:
        lam = 1.0
        mode = "full_p0_ofc_protected"
    else:
        u = (d0_drift - trig) / (full_th - trig)
        lam = u * u * (3.0 - 2.0 * u)
        mode = "convex_ramp_p0_ofc"

    dw_64 = dw_t.detach().to(device="cpu", dtype=torch.float64)
    p0_64 = p0_t.detach().to(device="cpu", dtype=torch.float64)
    idx_t = torch.tensor(ofc_clean, dtype=torch.long) if ofc_clean else None

    if lam == 0.0:
        blended_64 = dw_64.clone()
    else:
        dw_prot = dw_64 @ p0_64
        if idx_t is not None:
            dw_prot[:, idx_t] = 0.0
        blended_64 = (1.0 - lam) * dw_64 + lam * dw_prot
        if lam == 1.0 and idx_t is not None:
            blended_64[:, idx_t] = 0.0

    raw_frob = float(torch.linalg.norm(dw_64).item())
    blended_frob = float(torch.linalg.norm(blended_64).item())
    retained_ratio = (blended_frob / raw_frob) if raw_frob > 1e-30 else 1.0
    ofc_leak = float(blended_64[:, idx_t].abs().max().item()) if idx_t is not None else 0.0

    v_proj = torch.eye(in_dim, dtype=torch.float64) - p0_64
    p0_leak = (
        float(torch.linalg.norm(blended_64 @ v_proj).item()) / max(blended_frob, 1e-12)
        if blended_frob > 1e-30
        else 0.0
    )

    blended_out = blended_64.to(device=dw_t.device, dtype=dw_t.dtype).contiguous()
    return blended_out, {
        "blend_lambda": float(lam),
        "protection_mode": mode,
        "stripping_applied": bool(lam > 0.0),
        "d0_drift_probe": float(d0_drift),
        "trigger_threshold": trig,
        "full_protection_threshold": full_th,
        "raw_frob_norm": raw_frob,
        "blended_frob_norm": blended_frob,
        "retained_frob_ratio": retained_ratio,
        "ofc_protected_count": len(ofc_clean),
        "ofc_input_leak": ofc_leak,
        "p0_nullspace_leak": p0_leak,
    }


# ---------------------------------------------------------------------------
# 4. Cross-Fitted James-Stein Rank-k Step & Signed-Tau Exact Linearity
# ---------------------------------------------------------------------------


@torch.no_grad()
def fit_crossfitted_james_stein_ofc_step(
    spec_grads: torch.Tensor,
    general_grads: torch.Tensor,
    *,
    protected_indices: Sequence[int] | torch.Tensor = (),
    max_rank: int = DEFAULT_JS_MAX_RANK,
    max_step_norm: float = DEFAULT_JS_MAX_STEP_NORM,
    max_d0_pcs: int = 2,
    sv_tol: float = DEFAULT_NULLSPACE_SV_TOL,
) -> dict[str, Any]:
    """Compute strict 2-fold cross-fitted James-Stein rank-``k`` step inside ``ker(G_D0) intersect S_val^perp``."""
    sp_t = _validate_fp_tensor(spec_grads, "spec_grads", exact_ndim=2)
    g0_t = _validate_fp_tensor(general_grads, "general_grads", exact_ndim=2)
    dim = int(sp_t.shape[1])
    if int(g0_t.shape[1]) != dim:
        raise ValueError(f"general_grads width {g0_t.shape[1]} must match spec_grads width {dim}")

    k_max = _validate_positive_int(max_rank, "max_rank")
    step_cap = _validate_positive_finite(max_step_norm, "max_step_norm")
    pcs_cap = _validate_nonneg_int(max_d0_pcs, "max_d0_pcs")
    tol_f = _validate_positive_finite(sv_tol, "sv_tol")
    if tol_f >= 1.0:
        raise ValueError(f"sv_tol must be < 1.0, got {tol_f}")
    ofc_clean = _validate_index_sequence(protected_indices, dim, "protected_indices")

    sp_64 = sp_t.detach().to(device="cpu", dtype=torch.float64)
    g0_64 = g0_t.detach().to(device="cpu", dtype=torch.float64)
    out_dtype = torch.promote_types(sp_t.dtype, g0_t.dtype)
    n_spec = int(sp_64.shape[0])

    g0_mean = g0_64.mean(dim=0, keepdim=True)
    constraints: list[torch.Tensor] = []
    if float(g0_mean.norm().item()) > 1e-15:
        constraints.append(g0_mean)
    if pcs_cap > 0 and g0_64.shape[0] >= 2:
        _, s_d0, vh_d0 = torch.linalg.svd(g0_64 - g0_mean, full_matrices=False)
        if s_d0.numel() > 0 and float(s_d0[0].item()) > 1e-15:
            keep_pc = min(pcs_cap, int((s_d0 > tol_f * float(s_d0[0].item())).sum().item()))
            if keep_pc > 0:
                constraints.append(vh_d0[:keep_pc, :])
    if ofc_clean:
        eye_ofc = torch.zeros((len(ofc_clean), dim), dtype=torch.float64)
        for ri, ci in enumerate(ofc_clean):
            eye_ofc[ri, ci] = 1.0
        constraints.append(eye_ofc)

    if constraints:
        c_mat = torch.cat(constraints, dim=0)
        _, s_c, vh_c = torch.linalg.svd(c_mat, full_matrices=False)
        s_c_top = float(s_c[0].item()) if s_c.numel() > 0 else 0.0
        keep_c = int((s_c > 1e-12 * max(1.0, s_c_top)).sum().item()) if s_c_top > 1e-15 else 0
        q_c = vh_c[:keep_c, :].T.contiguous()
    else:
        keep_c = 0
        q_c = torch.empty((dim, 0), dtype=torch.float64)

    idx_t = torch.tensor(ofc_clean, dtype=torch.long) if ofc_clean else None

    def _proj_null(vec: torch.Tensor) -> torch.Tensor:
        out = vec
        if q_c.shape[1] > 0:
            for _ in range(2):
                out = out - (out @ q_c) @ q_c.T
        if idx_t is not None:
            if out.ndim == 1:
                out = out.clone()
                out[idx_t] = 0.0
            else:
                out = out.clone()
                out[:, idx_t] = 0.0
        return out

    def _zero_res(reason: str) -> dict[str, Any]:
        return {
            "delta_w": torch.zeros(dim, dtype=out_dtype),
            "rank_k": 0,
            "constraint_rank": keep_c,
            "shrinkage_factors": [],
            "mean_shrinkage_factor": 0.0,
            "cross_signal_sq": [],
            "step_l2_norm": 0.0,
            "d0_mean_grad_ortho_err": 0.0,
            "ofc_protected_max_delta": 0.0,
            "zero_step": True,
            "fallback_reason": reason,
        }

    if dim - keep_c <= 0:
        return _zero_res("empty_nullspace")

    fold_a = _proj_null(sp_64[0::2])
    fold_b = _proj_null(sp_64[1::2])
    if fold_a.shape[0] < 2 or fold_b.shape[0] < 2:
        return _zero_res("insufficient_crossfit_chunks")

    _, s_a, vh_a = torch.linalg.svd(fold_a, full_matrices=False)
    s_top = float(s_a[0].item()) if s_a.numel() > 0 else 0.0
    if s_top <= 1e-15:
        return _zero_res("zero_specialist_nullspace_signal")

    k_avail = int((s_a > 1e-12 * max(1.0, s_top)).sum().item())
    k_eff = min(k_max, k_avail, int(fold_a.shape[0]) - 1, int(fold_b.shape[0]) - 1, dim - keep_c)
    if k_eff <= 0:
        return _zero_res("zero_effective_rank")

    u_k = vh_a[:k_eff, :].T
    proj_a = fold_a @ u_k
    proj_b = fold_b @ u_k
    mu_a = proj_a.mean(dim=0)
    mu_b = proj_b.mean(dim=0)
    mu_all = 0.5 * (mu_a + mu_b)
    var_all = 0.5 * (proj_a.var(dim=0, unbiased=True) + proj_b.var(dim=0, unbiased=True)) + 1e-12
    se_sq = var_all / float(max(n_spec, 1))

    cross_sig_sq = torch.clamp(mu_a * mu_b, min=0.0)
    shrinkage = cross_sig_sq / (cross_sig_sq + se_sq + 1e-12)
    step_coeffs = -(shrinkage * mu_all) / torch.sqrt(var_all + 1e-8)

    dw = _proj_null(u_k @ step_coeffs)
    dw = _proj_null(dw)
    nrm = float(dw.norm().item())
    if nrm > step_cap:
        dw = _proj_null(dw * (step_cap / nrm))

    final_norm = float(dw.norm().item())
    zero_step = bool(final_norm <= 1e-15)
    dw_out = torch.zeros(dim, dtype=out_dtype) if zero_step else dw.to(dtype=out_dtype).contiguous()
    dw_check = dw_out.to(dtype=torch.float64)
    d0_err = float(abs((g0_mean @ dw_check).item()))
    ofc_max = float(dw_check[idx_t].abs().max().item()) if idx_t is not None else 0.0

    return {
        "delta_w": dw_out,
        "rank_k": k_eff,
        "constraint_rank": keep_c,
        "shrinkage_factors": [float(v) for v in shrinkage.tolist()],
        "mean_shrinkage_factor": float(shrinkage.mean().item()),
        "cross_signal_sq": [float(v) for v in cross_sig_sq.tolist()],
        "step_l2_norm": float(dw_check.norm().item()),
        "d0_mean_grad_ortho_err": d0_err,
        "ofc_protected_max_delta": ofc_max,
        "zero_step": zero_step,
        "fallback_reason": "zero_crossfit_signal" if zero_step else "none",
    }


def _resolve_dotted_attr(root: Any, attr_path: str) -> Any:
    """Resolve a dotted attribute path on ``root``, returning ``None`` if absent."""
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

    return norm_mod, head_mod, resolved_norm, norm_name, head_name, hidden_dim, int(head_w.shape[0]), eps_eff


def _is_general_domain(domain_label: str, general_prefixes: tuple[str, ...]) -> bool:
    """Return True if ``domain_label`` starts with any general-prose prefix."""
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
    """Slice scored causal positions and compute unit-normalized states in ``float64``."""
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
) -> torch.Tensor:
    """Compute exact per-chunk gradient ``d(NLL_i)/d(norm.weight)`` in ``float64`` (zero backprop)."""
    c_num, t_len, dim = u_scored.shape
    x_norm = u_scored * w0
    if b0 is not None:
        x_norm = x_norm + b0
    logits = F.linear(x_norm.reshape(-1, dim), head_w, head_b)
    probs = F.softmax(logits, dim=-1)
    t_flat = targets_2d.reshape(-1)
    probs[torch.arange(t_flat.numel()), t_flat] -= 1.0
    tok_grads = (probs @ head_w).reshape(c_num, t_len, dim) * u_scored
    return tok_grads.mean(dim=1).contiguous()


def _eval_logits_metrics(
    logits: torch.Tensor,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    *,
    num_bins: int,
    ignore_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute global NLL/ECE and per-domain mean NLLs from precomputed logits."""
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
        "mean_confidence": float(global_m["mean_confidence"]),
        "overconfidence_gap": float(global_m["overconfidence_gap"]),
        "num_tokens": int(global_m["num_tokens"]),
    }


def _compute_anse_verdict(
    ref_dom_nll: Mapping[str, float],
    ref_ece: float,
    cand_dom_nll: Mapping[str, float],
    cand_ece: float,
    *,
    cfg: ConditionalP0OFCCalibratorConfig,
) -> dict[str, Any]:
    """Compute ANSE thermodynamic energy, ``D0`` drift ratio, and domain guards."""
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
        anse_energy = mean_diff + cfg.ece_energy_weight * ece_diff
    else:
        anse_energy = cfg.maximum_pain_energy + max(0.0, mean_diff) + cfg.ece_energy_weight * float(cand_ece)

    return {
        "nll_mean_diff": float(mean_diff),
        "ece_diff": float(ece_diff),
        "ece_rel_delta": float(ece_diff / max(float(ref_ece), 1e-12)),
        "general_drift_ratio": float(gen_ratio - 1.0),
        "general_pct_vs_base": float(100.0 * (gen_ratio - 1.0)),
        "worst_domain_id": worst_dom,
        "worst_domain_pct_vs_base": float(100.0 * (worst_ratio - 1.0)),
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "delta_e": float(anse_energy),
        "domain_drifts_pct": drifts_pct,
    }


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a detached CPU clone of ``model.state_dict()``."""
    return {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}


def _restore_state_inplace(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``snapshot`` onto ``model`` matching parameter devices and dtypes."""
    cur_sd = model.state_dict()
    model.load_state_dict(
        {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()}
    )


def _fast_logits_nll_ece(
    logits: torch.Tensor,
    targets_2d: torch.Tensor,
    dom_slices: Sequence[tuple[str, torch.Tensor]],
    boundaries_inner: torch.Tensor,
    *,
    num_bins: int,
) -> tuple[dict[str, float], float, float]:
    """Fast vectorized per-domain NLL and global NLL/ECE in ``float64`` for grid search."""
    flat_logits = logits.reshape(-1, int(logits.shape[-1]))
    flat_targets = targets_2d.reshape(-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    tok_nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    probs = log_probs.exp()
    conf, preds = probs.max(dim=-1)
    acc = (preds == flat_targets).to(torch.float64)
    bin_ids = torch.bucketize(conf, boundaries_inner, right=False)
    counts = torch.bincount(bin_ids, minlength=num_bins)
    nz = counts > 0
    counts_f = counts.to(torch.float64)
    sum_acc = torch.bincount(bin_ids, weights=acc, minlength=num_bins)
    sum_conf = torch.bincount(bin_ids, weights=conf, minlength=num_bins)
    abs_gaps = torch.where(nz, (sum_acc - sum_conf).abs() / counts_f.clamp_min(1.0), torch.zeros_like(counts_f))
    ece_val = float(((counts_f / float(flat_targets.numel())) * abs_gaps).sum().item())

    seq_nll = tok_nll.reshape(targets_2d.shape[0], targets_2d.shape[1]).mean(dim=1)
    dom_nll = {dom: float(seq_nll[idx_t].mean().item()) for dom, idx_t in dom_slices}
    return dom_nll, float(tok_nll.mean().item()), ece_val


def _search_linear_tau_alpha(
    l_base: torch.Tensor,
    l_dw: torch.Tensor | None,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    base_dom: Mapping[str, float],
    base_ece: float,
    *,
    taus: Sequence[float],
    alphas: Sequence[float],
    cfg: ConditionalP0OFCCalibratorConfig,
) -> dict[str, Any]:
    """Grid-search ``L(tau, alpha) = tau * L_base + alpha * L_dw`` with exact linearity (zero extra matmuls)."""
    dom_groups: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        dom_groups.setdefault(dom, []).append(idx)
    dom_slices = [(dom, torch.tensor(ix, dtype=torch.long)) for dom, ix in sorted(dom_groups.items())]
    boundaries_inner = torch.linspace(0.0, 1.0, steps=cfg.num_bins + 1, dtype=torch.float64)[1:-1]

    best: dict[str, Any] = {
        "found": False,
        "tau": 1.0,
        "alpha": 0.0,
        "probe_energy": 0.0,
        "probe_nll_mean": float(sum(base_dom.values()) / len(base_dom)),
        "probe_ece_value": float(base_ece),
    }
    alpha_vals = (0.0,) if l_dw is None else tuple(alphas)
    for tau in taus:
        for alpha in alpha_vals:
            if abs(float(tau) - 1.0) <= 1e-12 and abs(float(alpha)) <= 1e-12:
                continue
            logits = l_base * float(tau) if (l_dw is None or alpha == 0.0) else (l_base * float(tau) + l_dw * float(alpha))
            dom_m, nll_m, ece_m = _fast_logits_nll_ece(
                logits, targets_2d, dom_slices, boundaries_inner, num_bins=cfg.num_bins
            )
            ver = _compute_anse_verdict(base_dom, base_ece, dom_m, ece_m, cfg=cfg)
            if ver["guard_ok"] and ver["anse_energy"] < best["probe_energy"] - cfg.min_energy_drop:
                best = {
                    "found": True,
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "probe_energy": float(ver["anse_energy"]),
                    "probe_nll_mean": nll_m,
                    "probe_ece_value": ece_m,
                }
    return best


def _eval_norm_params_on_units(
    u_scored: torch.Tensor,
    targets_2d: torch.Tensor,
    domains: Sequence[str],
    w_cand: torch.Tensor,
    b_cand: torch.Tensor | None,
    head_w: torch.Tensor,
    head_b_cand: torch.Tensor | None,
    *,
    cfg: ConditionalP0OFCCalibratorConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Compute logits and metrics for candidate norm parameters."""
    x_norm = u_scored * w_cand
    if b_cand is not None:
        x_norm = x_norm + b_cand
    logits = F.linear(x_norm, head_w, head_b_cand)
    dom_m, glob_m = _eval_logits_metrics(
        logits, targets_2d, domains, num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
    )
    return logits, dom_m, glob_m


def _apply_signed_tau_norm_params(
    norm_mod: nn.Module,
    head_mod: nn.Module,
    w0: torch.Tensor,
    b0: torch.Tensor | None,
    head_b0: torch.Tensor | None,
    dw: torch.Tensor | None,
    *,
    tau: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute and apply ``w = tau * w0 + alpha * dw``, ``b = tau * b0``, ``head_b = tau * head_b0``."""
    w_new = w0 * float(tau)
    if dw is not None and alpha != 0.0:
        w_new = w_new + float(alpha) * dw
    b_new = (b0 * float(tau)) if b0 is not None else None
    hb_new = (head_b0 * float(tau)) if head_b0 is not None else None

    norm_mod.weight.copy_(w_new.to(device=norm_mod.weight.device, dtype=norm_mod.weight.dtype))
    if getattr(norm_mod, "bias", None) is not None and b_new is not None:
        norm_mod.bias.copy_(b_new.to(device=norm_mod.bias.device, dtype=norm_mod.bias.dtype))
    if getattr(head_mod, "bias", None) is not None and hb_new is not None:
        head_mod.bias.copy_(hb_new.to(device=head_mod.bias.device, dtype=head_mod.bias.dtype))
    return w_new, b_new, hb_new


@torch.no_grad()
def calibrate_signed_tau_and_crossfit_js_norm(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    protected_indices: Sequence[int] | torch.Tensor = (),
    config: ConditionalP0OFCCalibratorConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Calibrate signed-tau folded norm + cross-fitted James-Stein rank-``k`` step inside ``ker(G_D0) intersect S_val^perp``.

    Exploits exact affine logit linearity ``L(tau, alpha) = tau * L_base + alpha * L_dw`` to
    distinguish under-confident Base regimes (``tau > 1.0``) from over-confident RLHF/Instruct
    regimes (``tau < 1.0``) using at most 2 ``lm_head`` projections per chunk batch, verifying
    ANSE ``Delta E < 0`` and ``D0 <= +1.0%`` with atomic bitwise rollback and SHA-256 receipts.
    """
    if config is not None and not isinstance(config, ConditionalP0OFCCalibratorConfig):
        raise TypeError(
            f"config must be ConditionalP0OFCCalibratorConfig or None, got {type(config).__name__}"
        )
    cfg = config or ConditionalP0OFCCalibratorConfig()
    _validate_bool(inplace, "inplace")

    _, _, _, norm_name_chk, head_name_chk, hidden_dim, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
    ofc_clean = _validate_index_sequence(protected_indices, hidden_dim, "protected_indices")

    if selection_chunks is None:
        fit_raw, sel_raw = _split_probe_chunks_fallback(probe_chunks, cfg.general_prefixes)
    else:
        fit_raw = probe_chunks
        sel_raw = selection_chunks

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

        u_fit = _extract_unit_states(h_fit_raw, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        u_sel = _extract_unit_states(h_sel_raw, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        u_ver = (
            u_sel
            if eval_chunks is None
            else _extract_unit_states(h_ver_raw, norm_kind=resolved_norm, eps=eps_eff, score_from=cfg.score_from)
        )

        chunk_grads = _compute_norm_chunk_gradients(u_fit, fit_prep["targets_2d"], w0, b0, head_w, head_b)
        gen_grads = chunk_grads[fit_prep["is_general"]]
        spec_grads = chunk_grads[~fit_prep["is_general"]]

        js_fit = fit_crossfitted_james_stein_ofc_step(
            spec_grads,
            gen_grads,
            protected_indices=ofc_clean,
            max_rank=cfg.js_max_rank,
            max_step_norm=cfg.js_max_step_norm,
            sv_tol=cfg.nullspace_sv_tol,
        )
        dw_js = None if js_fit["zero_step"] else js_fit["delta_w"].to(dtype=torch.float64)

        # Exact affine logit decomposition: L(tau, alpha) = tau * L_base + alpha * L_dw
        x_base_fit = u_fit * w0 + (b0 if b0 is not None else 0.0)
        l_base_fit = F.linear(x_base_fit, head_w, head_b)
        l_dw_fit = F.linear(u_fit * dw_js, head_w, None) if dw_js is not None else None

        fit_base_dom, fit_base_g = _eval_logits_metrics(
            l_base_fit, fit_prep["targets_2d"], fit_prep["domains"], num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
        )
        scalar_cand = _search_linear_tau_alpha(
            l_base_fit,
            None,
            fit_prep["targets_2d"],
            fit_prep["domains"],
            fit_base_dom,
            fit_base_g["ece_value"],
            taus=cfg.tau_grid,
            alphas=(0.0,),
            cfg=cfg,
        )
        js_cand = _search_linear_tau_alpha(
            l_base_fit,
            l_dw_fit,
            fit_prep["targets_2d"],
            fit_prep["domains"],
            fit_base_dom,
            fit_base_g["ece_value"],
            taus=cfg.tau_grid,
            alphas=cfg.alpha_grid,
            cfg=cfg,
        )

        # Compare base, scalar_signed_tau, and crossfit_js_signed_tau on held-out selection split
        _, sel_base_dom, sel_base_g = _eval_norm_params_on_units(
            u_sel, sel_prep["targets_2d"], sel_prep["domains"], w0, b0, head_w, head_b, cfg=cfg
        )
        _, sel_sc_dom, sel_sc_g = _eval_norm_params_on_units(
            u_sel,
            sel_prep["targets_2d"],
            sel_prep["domains"],
            w0 * scalar_cand["tau"],
            b0 * scalar_cand["tau"] if b0 is not None else None,
            head_w,
            head_b * scalar_cand["tau"] if head_b is not None else None,
            cfg=cfg,
        )
        sel_sc_anse = _compute_anse_verdict(
            sel_base_dom, sel_base_g["ece_value"], sel_sc_dom, sel_sc_g["ece_value"], cfg=cfg
        )

        js_w_sel = w0 * js_cand["tau"] + (js_cand["alpha"] * dw_js if dw_js is not None else 0.0)
        _, sel_js_dom, sel_js_g = _eval_norm_params_on_units(
            u_sel,
            sel_prep["targets_2d"],
            sel_prep["domains"],
            js_w_sel,
            b0 * js_cand["tau"] if b0 is not None else None,
            head_w,
            head_b * js_cand["tau"] if head_b is not None else None,
            cfg=cfg,
        )
        sel_js_anse = _compute_anse_verdict(
            sel_base_dom, sel_base_g["ece_value"], sel_js_dom, sel_js_g["ece_value"], cfg=cfg
        )

        def _improving(verdict: Mapping[str, Any]) -> bool:
            return (
                float(verdict["delta_e"]) < -cfg.min_energy_drop
                if cfg.require_strict_energy_drop
                else float(verdict["delta_e"]) <= 0.0
            )

        sc_ok = bool(scalar_cand["found"] and sel_sc_anse["guard_ok"] and _improving(sel_sc_anse))
        js_ok = bool(js_cand["found"] and sel_js_anse["guard_ok"] and _improving(sel_js_anse))

        if js_ok and (not sc_ok or float(sel_js_anse["anse_energy"]) < float(sel_sc_anse["anse_energy"]) - 1e-12):
            chosen_arm = "crossfit_js_signed_tau"
            chosen_tau = float(js_cand["tau"])
            chosen_alpha = float(js_cand["alpha"])
        elif sc_ok:
            chosen_arm = "scalar_signed_tau"
            chosen_tau = float(scalar_cand["tau"])
            chosen_alpha = 0.0
        else:
            chosen_arm = "base_rollback"
            chosen_tau = 1.0
            chosen_alpha = 0.0

        w_cand, b_cand, hb_cand = _apply_signed_tau_norm_params(
            norm_mod, head_mod, w0, b0, head_b, dw_js, tau=chosen_tau, alpha=chosen_alpha
        )
        cand_sha = compute_module_state_sha256(work_model)

        # Verify exact logit linearity L(tau, alpha) = tau * L_base + alpha * L_dw against live forward
        start_pos = max(0, cfg.score_from - 1)
        first_param = next(work_model.parameters(), None)
        dev = first_param.device if first_param is not None else torch.device("cpu")
        live_logits = work_model(fit_prep["ids_2d"][: cfg.batch_size].to(dev))[:, start_pos:-1, :].detach().to(
            device="cpu", dtype=torch.float64
        )
        lin_expected = chosen_tau * l_base_fit[: cfg.batch_size]
        if l_dw_fit is not None and chosen_alpha != 0.0:
            lin_expected = lin_expected + chosen_alpha * l_dw_fit[: cfg.batch_size]
        linearity_err = float((live_logits - lin_expected).abs().max().item())

        _, ver_base_dom, ver_base_g = _eval_norm_params_on_units(
            u_ver, ver_prep["targets_2d"], ver_prep["domains"], w0, b0, head_w, head_b, cfg=cfg
        )
        _, ver_cand_dom, ver_cand_g = _eval_norm_params_on_units(
            u_ver, ver_prep["targets_2d"], ver_prep["domains"], w_cand, b_cand, head_w, hb_cand, cfg=cfg
        )
        ver_anse = _compute_anse_verdict(
            ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
        )

        if cand_sha == initial_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED" if ver_anse["guard_ok"] else "REJECTED_GUARD_VIOLATION_UNCHANGED"
        elif ver_anse["guard_ok"] and _improving(ver_anse):
            accepted = True
            rolled_back = False
            status = (
                "ACCEPTED_CROSSFIT_JS_SIGNED_TAU"
                if chosen_arm == "crossfit_js_signed_tau"
                else "ACCEPTED_SCALAR_SIGNED_TAU"
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

    eff_tau = chosen_tau if accepted else 1.0
    eff_alpha = chosen_alpha if accepted else 0.0
    if accepted and eff_tau < 1.0 - 1e-9:
        cal_regime = "overconfident_rlhf_instruct"
    elif accepted and eff_tau > 1.0 + 1e-9:
        cal_regime = "underconfident_base"
    else:
        cal_regime = "neutral_or_unchanged"

    active_g = ver_cand_g if accepted else ver_base_g
    active_anse = (
        ver_anse
        if accepted
        else {
            "anse_energy": 0.0 if ver_anse["guard_ok"] else float(ver_anse["anse_energy"]),
            "general_drift_ratio": 0.0,
            "ece_rel_delta": 0.0,
            "delta_e": 0.0,
        }
    )

    receipt: dict[str, Any] = {
        "object": CALIBRATOR_RECEIPT_OBJECT,
        "schema": CALIBRATOR_SCHEMA,
        "stage": "signed_tau_crossfit_js_norm",
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "selected_arm": chosen_arm if accepted else "base_rollback",
        "calibration_regime": cal_regime,
        "norm_kind": resolved_norm,
        "norm_attr": norm_name,
        "head_attr": head_name,
        "tau_temp": eff_tau,
        "alpha_scale": eff_alpha,
        "logit_linearity_max_abs_diff": linearity_err,
        "nll_mean": float(active_g["nll_mean"]),
        "ece_value": float(active_g["ece_value"]),
        "base_nll_mean": float(ver_base_g["nll_mean"]),
        "base_ece_value": float(ver_base_g["ece_value"]),
        "ece_rel_delta": float(active_anse["ece_rel_delta"]),
        "general_drift_ratio": float(active_anse["general_drift_ratio"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "cand_anse_energy": float(ver_anse["anse_energy"]),
        "delta_e": float(active_anse["delta_e"]),
        "guard_ok": bool(ver_anse["guard_ok"]),
        "ofc_protected_max_delta": float(js_fit["ofc_protected_max_delta"]),
        "d0_mean_grad_ortho_err": float(js_fit["d0_mean_grad_ortho_err"]),
        "james_stein_fit": {k: v for k, v in js_fit.items() if k != "delta_w"},
        "selection_comparison": {
            "base": {"nll_mean": float(sel_base_g["nll_mean"]), "ece_value": float(sel_base_g["ece_value"])},
            "scalar_signed_tau": {
                "nll_mean": float(sel_sc_g["nll_mean"]),
                "ece_value": float(sel_sc_g["ece_value"]),
                "tau_temp": float(scalar_cand["tau"]),
                "anse_energy": float(sel_sc_anse["anse_energy"]),
                "guard_ok": bool(sel_sc_anse["guard_ok"]),
            },
            "crossfit_js_signed_tau": {
                "nll_mean": float(sel_js_g["nll_mean"]),
                "ece_value": float(sel_js_g["ece_value"]),
                "tau_temp": float(js_cand["tau"]),
                "alpha_scale": float(js_cand["alpha"]),
                "anse_energy": float(sel_js_anse["anse_energy"]),
                "guard_ok": bool(sel_js_anse["guard_ok"]),
            },
        },
        "initial_state_sha256": initial_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


# ---------------------------------------------------------------------------
# 5. End-to-End Conditional P_0/OFC Multi-Adapter & Signed-Tau Calibrator
# ---------------------------------------------------------------------------


def _eval_model_causal_metrics(
    model: nn.Module,
    prep: Mapping[str, Any],
    *,
    cfg: ConditionalP0OFCCalibratorConfig,
) -> tuple[list[float], dict[str, float], dict[str, Any]]:
    """Run read-only forward pass and compute per-chunk NLLs, per-domain NLLs, and global metrics."""
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    start_pos = max(0, cfg.score_from - 1)
    ids_2d = prep["ids_2d"]
    logit_parts: list[torch.Tensor] = []
    for b0 in range(0, int(ids_2d.shape[0]), cfg.batch_size):
        out = model(ids_2d[b0 : b0 + cfg.batch_size].to(dev))
        logit_parts.append(out[:, start_pos:-1, :].detach().to(device="cpu", dtype=torch.float64))
    logits = torch.cat(logit_parts, dim=0)
    targets = prep["targets_2d"]
    chunk_nlls = (
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        .reshape(targets.shape[0], targets.shape[1])
        .mean(dim=1)
        .tolist()
    )
    dom_m, glob_m = _eval_logits_metrics(
        logits, targets, prep["domains"], num_bins=cfg.num_bins, ignore_index=cfg.ignore_index
    )
    return [float(x) for x in chunk_nlls], dom_m, glob_m


def _sum_adapter_deltas(
    raw_entry: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    target_shape: tuple[int, int],
    mod_path: str,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Validate and sum one or more adapter deltas matching ``target_shape``."""
    if isinstance(raw_entry, torch.Tensor):
        t = _validate_fp_tensor(raw_entry, f"adapter_deltas[{mod_path!r}]", exact_ndim=2)
        if tuple(t.shape) != target_shape:
            raise ValueError(f"adapter delta shape {tuple(t.shape)} must match {target_shape} on {mod_path!r}")
        return t.detach().to(device="cpu", dtype=torch.float64), None
    if isinstance(raw_entry, Mapping):
        tensors = list(raw_entry.values())
    elif isinstance(raw_entry, Sequence) and not isinstance(raw_entry, (str, bytes, bytearray)):
        tensors = list(raw_entry)
    else:
        raise TypeError(f"adapter_deltas[{mod_path!r}] must be a tensor, sequence, or mapping")
    if not tensors:
        raise ValueError(f"adapter_deltas[{mod_path!r}] must be non-empty")

    total = torch.zeros(target_shape, dtype=torch.float64)
    checked: list[torch.Tensor] = []
    for idx, item in enumerate(tensors):
        t = _validate_fp_tensor(item, f"adapter_deltas[{mod_path!r}][{idx}]", exact_ndim=2)
        if tuple(t.shape) != target_shape:
            raise ValueError(
                f"adapter delta shape {tuple(t.shape)} must match {target_shape} on {mod_path!r}"
            )
        t64 = t.detach().to(device="cpu", dtype=torch.float64)
        total = total + t64
        checked.append(t64)
    return total, (checked if len(checked) >= 2 else None)


@torch.no_grad()
def execute_verified_conditional_p0_ofc_adapter_calibration(
    model: nn.Module,
    adapter_deltas: Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor],
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    early_layer_attr: str | None = None,
    config: ConditionalP0OFCCalibratorConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run full verified ``PILOT15`` pipeline: (a) align/OFC audit -> (b) conditional blend -> (c) signed-tau JS norm."""
    if config is not None and not isinstance(config, ConditionalP0OFCCalibratorConfig):
        raise TypeError(
            f"config must be ConditionalP0OFCCalibratorConfig or None, got {type(config).__name__}"
        )
    if not isinstance(adapter_deltas, Mapping) or len(adapter_deltas) == 0:
        raise ValueError("adapter_deltas must be a non-empty mapping from module path to adapter delta(s)")
    cfg = config or ConditionalP0OFCCalibratorConfig()
    _validate_bool(inplace, "inplace")

    _, _, _, norm_name_chk, head_name_chk, _, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
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
    _, _, _, norm_name, head_name, norm_dim, _, _ = _resolve_norm_and_head(
        work_model, norm_kind=norm_kind, norm_attr=norm_name_chk, head_attr=head_name_chk
    )

    target_mods: dict[str, nn.Module] = {}
    summed_deltas: dict[str, torch.Tensor] = {}
    first_pair_list: list[torch.Tensor] | None = None
    for mod_path in sorted(adapter_deltas.keys()):
        mod = _resolve_dotted_attr(work_model, mod_path)
        if (
            not isinstance(mod, nn.Module)
            or not isinstance(getattr(mod, "weight", None), torch.Tensor)
            or mod.weight.ndim != 2
        ):
            raise ValueError(f"Target module {mod_path!r} does not exist or has no 2D weight tensor")
        w_shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
        d_sum, pair_list = _sum_adapter_deltas(adapter_deltas[mod_path], w_shape, mod_path)
        target_mods[mod_path] = mod
        summed_deltas[mod_path] = d_sum
        if first_pair_list is None and pair_list is not None:
            first_pair_list = pair_list

    early_mod: nn.Module | None = (
        _resolve_dotted_attr(work_model, _validate_nonempty_str(early_layer_attr, "early_layer_attr"))
        if early_layer_attr is not None
        else None
    )
    if early_layer_attr is not None and not isinstance(early_mod, nn.Module):
        raise ValueError(f"early_layer_attr {early_layer_attr!r} did not resolve to an nn.Module")
    if early_mod is None:
        for cand in _EARLY_LAYER_CANDIDATES:
            found = _resolve_dotted_attr(work_model, cand)
            if isinstance(found, nn.Module):
                early_mod = found
                break
        if early_mod is None:
            early_mod = target_mods[sorted(target_mods.keys())[0]]

    was_training = work_model.training
    work_model.eval()
    try:
        initial_snapshot = _snapshot_state(work_model)
        initial_sha = compute_module_state_sha256(work_model)
        base_target_weights = {mp: mod.weight.detach().clone() for mp, mod in target_mods.items()}

        # 1. Capture base probe D0 activations on each target module + early-layer means
        d0_inputs: dict[str, list[torch.Tensor]] = {mp: [] for mp in target_mods}
        early_means: list[torch.Tensor] = []
        fit_is_gen = fit_prep["is_general"]
        active_slice_is_gen: list[bool] = []
        handles: list[Any] = []

        def _make_target_hook(mp: str):
            def _hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
                x_in = inp[0].detach().to(device="cpu", dtype=torch.float64)
                for bi, flag in enumerate(active_slice_is_gen):
                    if flag:
                        d0_inputs[mp].append(x_in[bi])
            return _hook

        def _early_hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
            h_in = inp[0].detach().to(device="cpu", dtype=torch.float64)
            early_means.append(h_in.mean(dim=1))

        for mp, mod in target_mods.items():
            handles.append(mod.register_forward_pre_hook(_make_target_hook(mp)))
        handles.append(early_mod.register_forward_pre_hook(_early_hook))

        first_param = next(work_model.parameters(), None)
        dev = first_param.device if first_param is not None else torch.device("cpu")
        start_pos = max(0, cfg.score_from - 1)
        base_fit_logits_parts: list[torch.Tensor] = []
        try:
            ids_fit = fit_prep["ids_2d"]
            for b0 in range(0, int(ids_fit.shape[0]), cfg.batch_size):
                b1 = min(int(ids_fit.shape[0]), b0 + cfg.batch_size)
                active_slice_is_gen[:] = fit_is_gen[b0:b1].tolist()
                out = work_model(ids_fit[b0:b1].to(dev))
                base_fit_logits_parts.append(out[:, start_pos:-1, :].detach().to(device="cpu", dtype=torch.float64))
        finally:
            for h in handles:
                h.remove()

        base_fit_logits = torch.cat(base_fit_logits_parts, dim=0)
        fit_targets = fit_prep["targets_2d"]
        base_probe_nlls = (
            F.cross_entropy(
                base_fit_logits.reshape(-1, base_fit_logits.shape[-1]), fit_targets.reshape(-1), reduction="none"
            )
            .reshape(fit_targets.shape[0], fit_targets.shape[1])
            .mean(dim=1)
        )
        base_d0_probe_nll = float(base_probe_nlls[fit_is_gen].mean().item())
        early_means_t = torch.cat(early_means, dim=0)

        p0_by_mod: dict[str, torch.Tensor] = {}
        for mp in target_mods:
            stacked = torch.cat(d0_inputs[mp], dim=0)
            p0_by_mod[mp], _ = compute_ledoit_wolf_nullspace_projector(stacked, energy_keep=cfg.energy_keep)

        # 2. Measure unguarded merged D0 probe drift
        for mp, mod in target_mods.items():
            bw = base_target_weights[mp]
            mod.weight.copy_((bw.to(dtype=torch.float64) + summed_deltas[mp]).to(device=bw.device, dtype=bw.dtype))
        merged_probe_nlls, _, _ = _eval_model_causal_metrics(work_model, fit_prep, cfg=cfg)
        merged_d0_probe_nll = float(
            torch.tensor(merged_probe_nlls, dtype=torch.float64)[fit_is_gen].mean().item()
        )

        # 3. Step (a): Diagnose alignment and OFC vs massive-activation IoU
        first_mp = sorted(target_mods.keys())[0]
        first_d0_acts = torch.cat(d0_inputs[first_mp], dim=0)
        diag = diagnose_adapter_d0_alignment_and_ofc_overlap(
            early_means_t,
            base_probe_nlls,
            base_d0_nll=base_d0_probe_nll,
            merged_d0_nll=merged_d0_probe_nll,
            delta_w=summed_deltas[first_mp],
            d0_token_activations=first_d0_acts,
            adapter_deltas=first_pair_list,
            p0=p0_by_mod[first_mp],
            ofc_fraction=cfg.ofc_fraction,
            massive_ratio_threshold=cfg.massive_ratio_threshold,
            trigger_threshold=cfg.trigger_threshold,
            energy_keep=cfg.energy_keep,
            seed=cfg.seed,
        )

        # 4. Step (b): Conditional convex P_0 + OFC blend across all target modules
        blend_metas: dict[str, dict[str, Any]] = {}
        max_ofc_leak = 0.0
        for mp, mod in target_mods.items():
            bw = base_target_weights[mp]
            in_dim = int(bw.shape[1])
            ofc_for_mod = [idx for idx in diag["ofc_indices"] if idx < in_dim]
            blended_dw, b_meta = compute_conditional_convex_p0_ofc_blend(
                summed_deltas[mp],
                p0_by_mod[mp],
                ofc_for_mod,
                d0_drift_probe=diag["d0_drift_probe"],
                trigger_threshold=cfg.trigger_threshold,
                full_protection_threshold=cfg.full_protection_threshold,
            )
            mod.weight.copy_((bw.to(dtype=torch.float64) + blended_dw).to(device=bw.device, dtype=bw.dtype))
            blend_metas[mp] = b_meta
            max_ofc_leak = max(max_ofc_leak, float(b_meta["ofc_input_leak"]))

        # 5. Step (c): Optional signed-tau + cross-fitted JS norm calibration on top of blended adapters
        norm_receipt: dict[str, Any] | None = None
        if cfg.enable_norm_calibration:
            ofc_for_norm = [idx for idx in diag["ofc_indices"] if idx < norm_dim]
            norm_cfg = ConditionalP0OFCCalibratorConfig(
                trigger_threshold=cfg.trigger_threshold,
                full_protection_threshold=cfg.full_protection_threshold,
                energy_keep=cfg.energy_keep,
                ofc_fraction=cfg.ofc_fraction,
                massive_ratio_threshold=cfg.massive_ratio_threshold,
                js_max_rank=cfg.js_max_rank,
                js_max_step_norm=cfg.js_max_step_norm,
                nullspace_sv_tol=cfg.nullspace_sv_tol,
                max_general_drift_rel=cfg.max_general_drift_rel,
                max_domain_drift_rel=cfg.max_domain_drift_rel,
                ece_energy_weight=cfg.ece_energy_weight,
                maximum_pain_energy=cfg.maximum_pain_energy,
                min_energy_drop=cfg.min_energy_drop,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
                tau_grid=cfg.tau_grid,
                alpha_grid=cfg.alpha_grid,
                general_prefixes=cfg.general_prefixes,
                require_strict_energy_drop=cfg.require_strict_energy_drop,
                enable_norm_calibration=False,
                seed=cfg.seed,
            )
            _, norm_receipt = calibrate_signed_tau_and_crossfit_js_norm(
                work_model,
                fit_raw,
                selection_chunks=sel_raw,
                eval_chunks=eval_chunks,
                protected_indices=ofc_for_norm,
                config=norm_cfg,
                norm_kind=norm_kind,
                norm_attr=norm_name,
                head_attr=head_name,
                inplace=True,
            )

        # 6. Evaluate candidate vs initial BASE model on eval split under ANSE Delta E < 0 and D0 <= +1.0%
        cand_snapshot = _snapshot_state(work_model)
        cand_sha = compute_module_state_sha256(work_model)
        _, ver_cand_dom, ver_cand_g = _eval_model_causal_metrics(work_model, ver_prep, cfg=cfg)
        _restore_state_inplace(work_model, initial_snapshot)
        _, ver_base_dom, ver_base_g = _eval_model_causal_metrics(work_model, ver_prep, cfg=cfg)

        ver_anse = _compute_anse_verdict(
            ver_base_dom, ver_base_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cfg
        )
        improved = (
            float(ver_anse["delta_e"]) < -cfg.min_energy_drop
            if cfg.require_strict_energy_drop
            else float(ver_anse["delta_e"]) <= 0.0
        )

        # Restore candidate snapshot if accepted; otherwise remain atomically rolled back to initial BASE
        if cand_sha == initial_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED" if ver_anse["guard_ok"] else "REJECTED_GUARD_VIOLATION_UNCHANGED"
        elif ver_anse["guard_ok"] and improved:
            _restore_state_inplace(work_model, cand_snapshot)
            accepted = True
            rolled_back = False
            status = f"ACCEPTED_{blend_metas[first_mp]['protection_mode'].upper()}"
        else:
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_GUARD_VIOLATION" if not ver_anse["guard_ok"] else "ROLLED_BACK_ENERGY_NON_IMPROVING"

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
            "ece_rel_delta": 0.0,
            "delta_e": 0.0,
        }
    )

    receipt: dict[str, Any] = {
        "object": CALIBRATOR_RECEIPT_OBJECT,
        "schema": CALIBRATOR_SCHEMA,
        "stage": "conditional_p0_ofc_adapter_calibration",
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "nll_mean": float(active_g["nll_mean"]),
        "ece_value": float(active_g["ece_value"]),
        "base_nll_mean": float(ver_base_g["nll_mean"]),
        "base_ece_value": float(ver_base_g["ece_value"]),
        "ece_rel_delta": float(active_anse["ece_rel_delta"]),
        "general_drift_ratio": float(active_anse["general_drift_ratio"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "cand_anse_energy": float(ver_anse["anse_energy"]),
        "delta_e": float(active_anse["delta_e"]),
        "guard_ok": bool(ver_anse["guard_ok"]),
        "blend_lambda": float(blend_metas[first_mp]["blend_lambda"]),
        "protection_mode": str(blend_metas[first_mp]["protection_mode"]),
        "ofc_input_leak": float(max_ofc_leak),
        "ofc_iou": float(diag["ofc_iou"]),
        "oof_auroc": float(diag["oof_auroc"]),
        "alignment_diagnosis": diag,
        "module_blends": blend_metas,
        "norm_calibration_receipt": norm_receipt,
        "initial_state_sha256": initial_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CALIBRATOR_RECEIPT_OBJECT",
    "CALIBRATOR_SCHEMA",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ECE_ENERGY_WEIGHT",
    "DEFAULT_FULL_PROTECTION_THRESHOLD",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_JS_MAX_RANK",
    "DEFAULT_JS_MAX_STEP_NORM",
    "DEFAULT_MASSIVE_RATIO_THRESHOLD",
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MIN_ENERGY_DROP",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_NUM_BINS",
    "DEFAULT_OFC_FRACTION",
    "DEFAULT_P0_ENERGY_KEEP",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SEED",
    "DEFAULT_SIGNED_TAU_GRID",
    "DEFAULT_TRIGGER_THRESHOLD",
    "VALID_NORM_KINDS",
    "ConditionalP0OFCCalibratorConfig",
    "calibrate_signed_tau_and_crossfit_js_norm",
    "compute_conditional_convex_p0_ofc_blend",
    "compute_ledoit_wolf_nullspace_projector",
    "diagnose_adapter_d0_alignment_and_ofc_overlap",
    "discover_ofc_value_neurons",
    "execute_verified_conditional_p0_ofc_adapter_calibration",
    "fit_crossfitted_james_stein_ofc_step",
    "verify_conditional_calibrator_receipt_provenance",
]
