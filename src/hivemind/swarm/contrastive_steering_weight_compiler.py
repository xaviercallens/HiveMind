# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Overhead Contrastive Steering Vector to AlphaEdit Null-Space Weight Compiler.

Compiles paired positive-vs-negative Contrastive Activation Addition (CAA / RepE)
behavior representations (e.g., concise verified specialist code vs. thinking-loop
or hallucination sinks) directly into static rank-1/rank-r ``w3`` / ``down_proj``
LoRA or dense linear weights inside the AlphaEdit general-prose null-space ``P_0``
with **zero runtime forward/backward hooks**:

1. ``extract_oof_contrastive_steering_subspace(pos_acts, neg_acts, *, rank=2, min_oof_snr=0.25)``:
   Computes disjoint 2-fold out-of-fold (OOF) cross-validated primary and residual
   contrastive steering directions in CPU ``float64``, rejecting pure noise or
   overfitted spurious differences with an exact zero rank-0 subspace.
2. ``compile_steering_vector_into_nullspace_weight(weight_or_lora, trigger_keys, steering_vec, p_null, *, strength_alpha=0.5, ridge_lambda=1e-2)``:
   Solves the closed-form AlphaEdit null-space constrained ridge equation
   ``Delta W = alpha * (1/n S K_proj^T) (P_0 M_trig P_0 + lambda I)^{-1} P_0``
   and folds the update into dense weights or rank-preserving LoRA ``(B, A)``
   factors with exact preserved-key invariance (``Delta W K_0^T == 0``).
3. ``execute_verified_contrastive_weight_compilation(...)``:
   Applies sequential multi-layer contrastive weight compilation across
   ``CrowdV1WithEngram`` (``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``,
   ``CrowdRuntimeMLP``) or generic ``nn.Module`` targets under the ANSE
   thermodynamic gate (``Delta E < 0`` and worst-domain drift ``<= +1.0%``),
   enforcing atomic bitwise parent rollback, zero registered hooks, and
   canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_alphaedit_prose_nullspace
from hivemind.swarm.cross_layer_residual_nullspace import (
    canonical_json_bytes,
    compute_module_state_sha256,
    compute_payload_sha256,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdRuntimeMLP
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.gcp_t4_crowd_diloco import CrowdLoRALinear
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd

SCHEMA_VERSION: str = "hivemind.contrastive_steering_weight_compiler/1"
RECEIPT_OBJECT: str = "hivemind.contrastive_steering_weight_compilation_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_RANK: int = 2
DEFAULT_MIN_OOF_SNR: float = 0.25
DEFAULT_STRENGTH_ALPHA: float = 0.5
DEFAULT_RIDGE_LAMBDA: float = 1e-2
DEFAULT_NULLSPACE_SV_THRESHOLD: float = 1e-2
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_MIN_NULL_RANK: int = 4
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 0
DEFAULT_MIN_EXPERT_TRIGGER_GATE: float = 0.05

VALID_TARGET_PROJECTIONS: frozenset[str] = frozenset({"w3", "w2", "w1", "auto"})
VALID_LORA_FOLD_MODES: frozenset[str] = frozenset({"auto", "zero_slot", "update_a", "qr_svd"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Hook Inspection, Null-Space & Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    out = _validate_nonneg_int(value, name)
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


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    """Validate that ``value`` lies within ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_activation_matrix(
    acts: torch.Tensor | Sequence[torch.Tensor],
    name: str,
    *,
    min_rows: int = 1,
    allow_1d: bool = False,
) -> torch.Tensor:
    """Validate 2D/3D (or optional 1D) activation tensor(s) into CPU ``float64`` ``(N, D)``."""
    if isinstance(acts, torch.Tensor):
        items: list[tuple[str, torch.Tensor]] = [(name, acts)]
    elif isinstance(acts, Sequence) and not isinstance(acts, (str, bytes, bytearray)):
        if not acts:
            raise ValueError(f"{name} sequence must not be empty")
        items = [(f"{name}[{idx}]", t) for idx, t in enumerate(acts)]
    else:
        raise TypeError(
            f"{name} must be a torch.Tensor or Sequence[torch.Tensor], got {type(acts).__name__}"
        )

    parts: list[torch.Tensor] = []
    width: int | None = None
    allowed_ndims = (1, 2, 3) if allow_1d else (2, 3)
    for label, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim not in allowed_ndims or any(dim == 0 for dim in tensor.shape):
            raise ValueError(
                f"{label} must be a non-empty {allowed_ndims}D tensor, got shape {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {tensor.dtype}")
        if tensor.ndim == 1:
            flat = tensor.detach().to(device="cpu", dtype=torch.float64).unsqueeze(0)
        else:
            flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        cur_w = int(flat.shape[1])
        if width is None:
            width = cur_w
        elif cur_w != width:
            raise ValueError(f"{label} width {cur_w} must match expected width {width}")
        parts.append(flat)

    merged = torch.cat(parts, dim=0).contiguous()
    if int(merged.shape[0]) < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} samples, got {merged.shape[0]}")
    return merged


def _validate_square_projector(
    p_null: torch.Tensor,
    name: str = "p_null",
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a finite square 2D projector matrix and return symmetrized CPU ``float64``."""
    if not isinstance(p_null, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(p_null).__name__}")
    if p_null.ndim != 2 or p_null.shape[0] == 0 or p_null.shape[0] != p_null.shape[1]:
        raise ValueError(f"{name} must be a non-empty square 2D tensor, got shape {tuple(p_null.shape)}")
    if not p_null.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {p_null.dtype}")
    dim = int(p_null.shape[0])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"{name} shape {tuple(p_null.shape)} must match input dimension {expected_dim}")
    p64 = p_null.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(p64).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return (0.5 * (p64 + p64.T)).contiguous()


def count_registered_runtime_hooks(model: nn.Module) -> int:
    """Return total count of PyTorch forward/backward hooks registered on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    hook_attrs = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
    )
    total = 0
    for mod in model.modules():
        for attr in hook_attrs:
            total += len(getattr(mod, attr, ()))
    return total


@torch.no_grad()
def build_prose_nullspace_projector(
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    sv_threshold: float = DEFAULT_NULLSPACE_SV_THRESHOLD,
    sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL,
    variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD,
    min_null_rank: int = DEFAULT_MIN_NULL_RANK,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct an AlphaEdit general-prose SVD null-space projector ``P_0`` with free-rank floor."""
    sv_thr = _validate_nonneg_finite(sv_threshold, "sv_threshold")
    rel_tol = _validate_nonneg_finite(sv_rel_tol, "sv_rel_tol")
    if rel_tol >= 1.0:
        raise ValueError(f"sv_rel_tol must be < 1.0, got {sv_rel_tol!r}")
    min_nr = _validate_nonneg_int(min_null_rank, "min_null_rank")

    if isinstance(prose_keys, Mapping):
        if not prose_keys:
            raise ValueError("prose_keys mapping must not be empty")
        keys_64 = _validate_activation_matrix(
            [prose_keys[k] for k in sorted(prose_keys.keys(), key=str)],
            "prose_keys",
        )
    else:
        keys_64 = _validate_activation_matrix(prose_keys, "prose_keys")

    d_key = int(keys_64.shape[1])
    s_vals = torch.linalg.svdvals(keys_64)
    top_sv = float(s_vals[0].item()) if s_vals.numel() > 0 else 0.0
    eff_sv_thr = max(sv_thr, rel_tol * top_sv) if top_sv > 0.0 else sv_thr
    max_active = max(1, d_key - min(min_nr, max(0, d_key - 1))) if min_nr > 0 else None
    return compute_alphaedit_prose_nullspace(
        keys_64,
        sv_threshold=eff_sv_thr,
        variance_threshold=variance_threshold,
        max_active_modes=max_active,
    )


def verify_contrastive_compilation_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema tags and canonical ``receipt_sha256`` integrity."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RECEIPT_OBJECT or receipt.get("schema") != SCHEMA_VERSION:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _restore_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> str:
    """Atomically restore ``model`` state from ``snapshot`` and verify bitwise SHA-256."""
    state = model.state_dict()
    for key, snap_t in snapshot.items():
        if key in state:
            state[key].copy_(snap_t.to(device=state[key].device, dtype=state[key].dtype))
    restored_sha = compute_module_state_sha256(model)
    if restored_sha != expected_sha256:
        raise RuntimeError(
            f"Atomic bitwise rollback failed: expected {expected_sha256}, got {restored_sha}"
        )
    return restored_sha


@dataclass(frozen=True)
class ContrastiveWeightCompilerConfig:
    """Validated configuration governing OOF contrastive compilation and ANSE rollback."""

    rank: int = DEFAULT_RANK
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR
    strength_alpha: float = DEFAULT_STRENGTH_ALPHA
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA
    target_proj: str = "w3"
    lora_fold_mode: str = "auto"
    nullspace_sv_threshold: float = DEFAULT_NULLSPACE_SV_THRESHOLD
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    min_null_rank: int = DEFAULT_MIN_NULL_RANK
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    min_energy_improvement: float = 0.0
    require_strict_energy_drop: bool = True
    enforce_all_domain_guard: bool = True
    loop_energy_weight: float = 0.0
    min_expert_trigger_gate: float = DEFAULT_MIN_EXPERT_TRIGGER_GATE
    max_norm_delta_ratio: float | None = None
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 16
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.rank, "rank")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        _validate_nonneg_finite(self.strength_alpha, "strength_alpha")
        _validate_positive_finite(self.ridge_lambda, "ridge_lambda")
        if self.target_proj not in VALID_TARGET_PROJECTIONS:
            raise ValueError(
                f"target_proj must be one of {sorted(VALID_TARGET_PROJECTIONS)}, got {self.target_proj!r}"
            )
        if self.lora_fold_mode not in VALID_LORA_FOLD_MODES:
            raise ValueError(
                f"lora_fold_mode must be one of {sorted(VALID_LORA_FOLD_MODES)}, got {self.lora_fold_mode!r}"
            )
        _validate_nonneg_finite(self.nullspace_sv_threshold, "nullspace_sv_threshold")
        rel_tol = _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if rel_tol >= 1.0:
            raise ValueError(f"nullspace_sv_rel_tol must be < 1.0, got {rel_tol!r}")
        if self.nullspace_variance_threshold is not None:
            _validate_unit_interval(
                self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
            )
        _validate_nonneg_int(self.min_null_rank, "min_null_rank")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_nonneg_finite(self.loop_energy_weight, "loop_energy_weight")
        _validate_nonneg_finite(self.min_expert_trigger_gate, "min_expert_trigger_gate")
        if self.max_norm_delta_ratio is not None:
            _validate_unit_interval(self.max_norm_delta_ratio, "max_norm_delta_ratio", allow_zero=False)
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. OOF Cross-Validated Contrastive Steering Subspace Extractor
# ---------------------------------------------------------------------------


def _build_contrast_differences(pos_x: torch.Tensor, neg_x: torch.Tensor) -> torch.Tensor:
    """Return paired differences ``pos - neg`` when matched, else centered cross differences."""
    if pos_x.shape[0] == neg_x.shape[0]:
        return pos_x - neg_x
    return torch.cat(
        [
            pos_x - neg_x.mean(dim=0, keepdim=True),
            pos_x.mean(dim=0, keepdim=True) - neg_x,
        ],
        dim=0,
    )


def _compute_directional_snr(pos_x: torch.Tensor, neg_x: torch.Tensor, direction: torch.Tensor) -> float:
    """Compute signed positive-vs-negative separation SNR along unit ``direction``."""
    zp = pos_x @ direction
    zn = neg_x @ direction
    if pos_x.shape[0] == neg_x.shape[0]:
        dz = zp - zn
        diff = max(0.0, float(dz.mean().item()))
        sd = math.sqrt(float(dz.var(unbiased=False).item()) + 1e-10) if dz.numel() > 1 else 1e-5
        return diff / sd
    diff = max(0.0, float(zp.mean().item() - zn.mean().item()))
    vp = float(zp.var(unbiased=False).item()) if zp.numel() > 1 else 0.0
    vn = float(zn.var(unbiased=False).item()) if zn.numel() > 1 else 0.0
    return diff / math.sqrt(vp + vn + 1e-10)


def _compute_primary_oof_snr(pos_2d: torch.Tensor, neg_2d: torch.Tensor) -> tuple[float, float]:
    """Compute disjoint 2-fold OOF SNR and fold cosine alignment for the primary CAA direction."""
    p0, p1 = pos_2d[0::2], pos_2d[1::2]
    n0, n1 = neg_2d[0::2], neg_2d[1::2]

    mu0 = p0.mean(dim=0) - n0.mean(dim=0)
    mu1 = p1.mean(dim=0) - n1.mean(dim=0)
    n_mu0 = float(mu0.norm().item())
    n_mu1 = float(mu1.norm().item())
    if n_mu0 <= 1e-12 or n_mu1 <= 1e-12:
        return 0.0, 0.0

    u0 = mu0 / n_mu0
    u1 = mu1 / n_mu1
    fold_cos = max(0.0, float((u0 @ u1).item()))
    if fold_cos <= 0.0:
        return 0.0, 0.0

    if p0.shape[0] == n0.shape[0] and p1.shape[0] == n1.shape[0]:
        dz_cat = torch.cat([(p1 - n1) @ u0, (p0 - n0) @ u1], dim=0)
        cat_diff = max(0.0, float(dz_cat.mean().item()))
        snr_cat = cat_diff / math.sqrt(float(dz_cat.var(unbiased=False).item()) + 1e-10)
    else:
        zp_cat = torch.cat([p1 @ u0, p0 @ u1], dim=0)
        zn_cat = torch.cat([n1 @ u0, n0 @ u1], dim=0)
        cat_diff = max(0.0, float(zp_cat.mean().item() - zn_cat.mean().item()))
        snr_cat = cat_diff / math.sqrt(
            float(zp_cat.var(unbiased=False).item()) + float(zn_cat.var(unbiased=False).item()) + 1e-10
        )

    if int(p0.shape[0]) >= 2 and int(p1.shape[0]) >= 2 and int(n0.shape[0]) >= 2 and int(n1.shape[0]) >= 2:
        snr_01 = _compute_directional_snr(p1, n1, u0)
        snr_10 = _compute_directional_snr(p0, n0, u1)
        raw_oof = min(snr_cat, snr_01, snr_10)
    else:
        raw_oof = snr_cat
    return raw_oof * fold_cos, fold_cos


def _deflate_rows(mat: torch.Tensor, basis_cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """Project out orthonormal columns ``basis_cols`` from row-major ``mat``."""
    out = mat.clone()
    for col in basis_cols:
        out = out - (out @ col).unsqueeze(1) * col.unsqueeze(0)
    return out


def _residual_mode_oof_snr(
    d0: torch.Tensor,
    d1: torch.Tensor,
    prev0: Sequence[torch.Tensor],
    prev1: Sequence[torch.Tensor],
    remaining_dim: int,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """Evaluate 2-fold OOF residual excess contrast SNR orthogonal to previously accepted modes."""
    if remaining_dim <= 1:
        return 0.0, None, None
    dr0 = _deflate_rows(d0, prev0)
    dr1 = _deflate_rows(d1, prev1)
    _, s0, vh0 = torch.linalg.svd(dr0, full_matrices=False)
    _, s1, vh1 = torch.linalg.svd(dr1, full_matrices=False)
    if s0.numel() == 0 or s1.numel() == 0 or float(s0[0].item()) <= 1e-10 or float(s1[0].item()) <= 1e-10:
        return 0.0, None, None

    v0 = vh0[0].clone()
    v1 = vh1[0].clone()
    for c in prev0:
        v0 = v0 - float((v0 @ c).item()) * c
    for c in prev1:
        v1 = v1 - float((v1 @ c).item()) * c
    nv0, nv1 = float(v0.norm().item()), float(v1.norm().item())
    if nv0 <= 1e-10 or nv1 <= 1e-10:
        return 0.0, None, None
    v0 = v0 / nv0
    v1 = v1 / nv1

    def _excess(d_eval: torch.Tensor, vec: torch.Tensor) -> tuple[float, float]:
        sq = (d_eval @ vec).pow(2)
        e_dir = float(sq.mean().item())
        e_bg = float(d_eval.pow(2).sum(dim=1).mean().item()) / float(remaining_dim)
        ratio = e_dir / max(e_bg, 1e-12)
        snr = max(0.0, e_dir - e_bg) / math.sqrt(float(sq.var(unbiased=False).item()) + 1e-10)
        return ratio, snr

    r01, es01 = _excess(dr1, v0)
    r10, es10 = _excess(dr0, v1)
    cos_res = abs(float((v0 @ v1).item()))
    if min(r01, r10) <= 1.35:
        return 0.0, None, None
    return min(es01, es10) * cos_res, v0, v1


@torch.no_grad()
def extract_oof_contrastive_steering_subspace(
    pos_acts: torch.Tensor | Sequence[torch.Tensor],
    neg_acts: torch.Tensor | Sequence[torch.Tensor],
    *,
    rank: int = DEFAULT_RANK,
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR,
    p_null: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract OOF-validated rank-r contrastive steering subspace rejecting noisy pairs.

    Uses disjoint 2-fold out-of-fold validation:
    - Mode 0 tests the primary CAA mean contrast direction ``bar{x}^+ - bar{x}^-`` via
      held-out signed contrast SNR scaled by fold cosine alignment.
    - Residual modes ``1 .. rank - 1`` iteratively deflate accepted directions and pass
      only if held-out residual variance exceeds the isotropic background floor across
      both folds.
    """
    req_rank = _validate_positive_int(rank, "rank")
    snr_gate = _validate_nonneg_finite(min_oof_snr, "min_oof_snr")
    pos_2d = _validate_activation_matrix(pos_acts, "pos_acts", min_rows=2)
    neg_2d = _validate_activation_matrix(neg_acts, "neg_acts", min_rows=2)

    dim = int(pos_2d.shape[1])
    if int(neg_2d.shape[1]) != dim:
        raise ValueError(f"pos_acts width {dim} must match neg_acts width {neg_2d.shape[1]}")

    if p_null is not None:
        p0 = _validate_square_projector(p_null, "p_null", expected_dim=dim)
        pos_2d = (pos_2d @ p0).contiguous()
        neg_2d = (neg_2d @ p0).contiguous()

    mu_all = pos_2d.mean(dim=0) - neg_2d.mean(dim=0)
    raw_norm = float(mu_all.norm().item())
    primary_oof_snr, fold_cos = _compute_primary_oof_snr(pos_2d, neg_2d)

    if raw_norm <= 1e-12 or primary_oof_snr < snr_gate:
        empty_basis = torch.zeros((dim, 0), dtype=torch.float64)
        zero_vec = torch.zeros(dim, dtype=torch.float64)
        return empty_basis, {
            "steering_vector": zero_vec,
            "unit_steering_vector": zero_vec.clone(),
            "subspace_basis": empty_basis,
            "mode_amplitudes": [],
            "mode_oof_snrs": [float(primary_oof_snr)],
            "singular_values": [],
            "oof_snr": float(primary_oof_snr),
            "steering_snr": float(primary_oof_snr),
            "full_snr": 0.0,
            "fold_cosine_alignment": float(fold_cos),
            "requested_rank": req_rank,
            "effective_rank": 0,
            "min_oof_snr": snr_gate,
            "accepted": False,
            "rejected": True,
            "zeroed_by_oof_gate": True,
            "raw_contrast_norm": raw_norm,
            "denoised_contrast_norm": 0.0,
            "n_pos": int(pos_2d.shape[0]),
            "n_neg": int(neg_2d.shape[0]),
            "dim": dim,
        }

    u_prim = (mu_all / raw_norm).contiguous()
    full_snr = _compute_directional_snr(pos_2d, neg_2d, u_prim)
    cols: list[torch.Tensor] = [u_prim]
    mode_snrs: list[float] = [float(primary_oof_snr)]

    d_all = _build_contrast_differences(pos_2d, neg_2d)
    d0 = _build_contrast_differences(pos_2d[0::2], neg_2d[0::2])
    d1 = _build_contrast_differences(pos_2d[1::2], neg_2d[1::2])
    u_f0 = pos_2d[0::2].mean(dim=0) - neg_2d[0::2].mean(dim=0)
    u_f1 = pos_2d[1::2].mean(dim=0) - neg_2d[1::2].mean(dim=0)
    prev0 = [u_f0 / u_f0.norm().clamp_min(1e-12)]
    prev1 = [u_f1 / u_f1.norm().clamp_min(1e-12)]

    max_modes = min(req_rank, dim, int(d_all.shape[0]))
    for mode_idx in range(1, max_modes):
        snr_j, v0_j, v1_j = _residual_mode_oof_snr(
            d0, d1, prev0, prev1, remaining_dim=dim - mode_idx
        )
        if snr_j < snr_gate or v0_j is None or v1_j is None:
            break
        dr_all = _deflate_rows(d_all, cols)
        _, s_rem, vh_rem = torch.linalg.svd(dr_all, full_matrices=False)
        if s_rem.numel() == 0 or float(s_rem[0].item()) <= 1e-10:
            break
        v_full = vh_rem[0].clone()
        for c in cols:
            v_full = v_full - float((v_full @ c).item()) * c
        nv = float(v_full.norm().item())
        if nv <= 1e-10:
            break
        v_full = v_full / nv
        if float((v_full @ mu_all).item()) < 0.0:
            v_full = -v_full
        cols.append(v_full.contiguous())
        mode_snrs.append(float(snr_j))
        prev0.append(v0_j)
        prev1.append(v1_j)

    subspace_basis = torch.stack(cols, dim=1).contiguous()
    amplitudes = subspace_basis.T @ mu_all
    steering_vector = (subspace_basis @ amplitudes).contiguous()
    steer_norm = float(steering_vector.norm().item())
    unit_steering = (
        (steering_vector / steer_norm).contiguous()
        if steer_norm > 1e-12
        else torch.zeros(dim, dtype=torch.float64)
    )
    proj_svals = torch.linalg.svdvals(d_all @ subspace_basis).tolist()

    return subspace_basis, {
        "steering_vector": steering_vector,
        "unit_steering_vector": unit_steering,
        "subspace_basis": subspace_basis,
        "mode_amplitudes": [float(a) for a in amplitudes.tolist()],
        "mode_oof_snrs": mode_snrs,
        "singular_values": [float(s) for s in proj_svals],
        "oof_snr": float(primary_oof_snr),
        "steering_snr": float(primary_oof_snr),
        "full_snr": float(full_snr),
        "fold_cosine_alignment": float(fold_cos),
        "requested_rank": req_rank,
        "effective_rank": int(subspace_basis.shape[1]),
        "min_oof_snr": snr_gate,
        "accepted": True,
        "rejected": False,
        "zeroed_by_oof_gate": False,
        "raw_contrast_norm": raw_norm,
        "denoised_contrast_norm": steer_norm,
        "n_pos": int(pos_2d.shape[0]),
        "n_neg": int(neg_2d.shape[0]),
        "dim": dim,
    }


# ---------------------------------------------------------------------------
# 3. Null-Space Ridge Compiler & Rank-Preserving LoRA Fold-In
# ---------------------------------------------------------------------------


def _resolve_compile_target(
    weight_or_lora: Any,
    *,
    weight_attr: str,
    expert_index: int | None,
) -> dict[str, Any]:
    """Resolve dense weight or LoRA ``(B, A)`` parameter handles from ``weight_or_lora``."""
    attr = str(weight_attr).strip()
    if not attr:
        raise ValueError("weight_attr must be a non-empty string")

    if isinstance(weight_or_lora, torch.Tensor):
        w = weight_or_lora
        if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
            raise ValueError(f"weight tensor must be non-empty 2D, got shape {tuple(w.shape)}")
        if not w.is_floating_point() or not bool(torch.isfinite(w.detach()).all().item()):
            raise ValueError("weight tensor must be finite and floating-point")
        return {"kind": "dense", "target_kind": "dense_tensor", "weight": w, "d_out": int(w.shape[0]), "d_in": int(w.shape[1])}

    if isinstance(weight_or_lora, tuple) and len(weight_or_lora) == 2:
        return _pack_lora_target(weight_or_lora[0], weight_or_lora[1], scaling=1.0, target_kind="lora_tuple")

    if isinstance(weight_or_lora, Mapping):
        if "B" in weight_or_lora and "A" in weight_or_lora:
            return _pack_lora_target(weight_or_lora["B"], weight_or_lora["A"], scaling=1.0, target_kind="lora_mapping")
        b_k, a_k = f"{attr}_B", f"{attr}_A"
        if b_k in weight_or_lora and a_k in weight_or_lora:
            return _pack_lora_target(weight_or_lora[b_k], weight_or_lora[a_k], scaling=1.0, target_kind=f"lora_{attr}_mapping")
        raise ValueError(f"Mapping target must expose ('B', 'A') or ({b_k!r}, {a_k!r})")

    if not isinstance(weight_or_lora, nn.Module):
        raise TypeError(
            f"weight_or_lora must be Tensor, nn.Module, (B, A) tuple, or Mapping; got {type(weight_or_lora).__name__}"
        )

    if isinstance(weight_or_lora, nn.Linear):
        w = weight_or_lora.weight
        return {"kind": "dense", "target_kind": "dense_linear", "weight": w, "d_out": int(w.shape[0]), "d_in": int(w.shape[1])}

    if isinstance(weight_or_lora, CrowdLoRAMoEMLP):
        if expert_index is None:
            raise ValueError("expert_index must be specified when targeting a CrowdLoRAMoEMLP")
        e_idx = _validate_nonneg_int(expert_index, "expert_index")
        key = str(e_idx)
        if key not in weight_or_lora.experts:
            raise ValueError(f"Expert {e_idx} not found in CrowdLoRAMoEMLP.experts")
        exp = weight_or_lora.experts[key]
        return _pack_lora_target(
            getattr(exp, f"{attr}_B"),
            getattr(exp, f"{attr}_A"),
            scaling=float(exp.scaling),
            target_kind=f"moe_expert_{e_idx}.{attr}",
        )

    if isinstance(weight_or_lora, CrowdFineGrainedMoEMLP):
        if expert_index is None:
            raise ValueError("expert_index must be specified when targeting a CrowdFineGrainedMoEMLP")
        e_idx = _validate_nonneg_int(expert_index, "expert_index")
        if e_idx >= weight_or_lora.num_experts:
            raise ValueError(f"expert_index {e_idx} out of range [0, {weight_or_lora.num_experts})")
        sub_list = getattr(weight_or_lora, f"experts_{attr}", None)
        if sub_list is None:
            raise ValueError(f"CrowdFineGrainedMoEMLP has no experts_{attr}")
        w = sub_list[e_idx].weight
        return {"kind": "dense", "target_kind": f"fine_grained_expert_{e_idx}.{attr}", "weight": w, "d_out": int(w.shape[0]), "d_in": int(w.shape[1])}

    if isinstance(weight_or_lora, CrowdLoRAExpertAdapter) or (
        hasattr(weight_or_lora, f"{attr}_B") and hasattr(weight_or_lora, f"{attr}_A")
    ):
        return _pack_lora_target(
            getattr(weight_or_lora, f"{attr}_B"),
            getattr(weight_or_lora, f"{attr}_A"),
            scaling=float(getattr(weight_or_lora, "scaling", 1.0)),
            target_kind=f"lora_{attr}",
        )

    if hasattr(weight_or_lora, "lora_B") and hasattr(weight_or_lora, "lora_A"):
        return _pack_lora_target(
            weight_or_lora.lora_B,
            weight_or_lora.lora_A,
            scaling=float(getattr(weight_or_lora, "scaling", 1.0)),
            target_kind="crowd_lora_linear" if isinstance(weight_or_lora, CrowdLoRALinear) else "lora_linear",
        )

    sub_mod = getattr(weight_or_lora, attr, None)
    if isinstance(sub_mod, nn.Linear):
        w = sub_mod.weight
        return {"kind": "dense", "target_kind": f"dense_{attr}", "weight": w, "d_out": int(w.shape[0]), "d_in": int(w.shape[1])}
    if isinstance(getattr(weight_or_lora, "weight", None), torch.Tensor):
        w = weight_or_lora.weight
        return {"kind": "dense", "target_kind": "module_weight", "weight": w, "d_out": int(w.shape[0]), "d_in": int(w.shape[1])}
    raise ValueError(f"Unsupported weight_or_lora target {type(weight_or_lora).__name__} for weight_attr={attr!r}")


def _pack_lora_target(
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    *,
    scaling: float,
    target_kind: str,
) -> dict[str, Any]:
    """Validate LoRA ``(B, A)`` tensors and bundle metadata."""
    if not isinstance(w_b, torch.Tensor) or not isinstance(w_a, torch.Tensor):
        raise TypeError("LoRA factors must be torch.Tensor instances")
    if w_b.ndim != 2 or w_a.ndim != 2 or w_b.shape[0] == 0 or w_a.shape[1] == 0:
        raise ValueError(f"LoRA factors must be non-empty 2D tensors, got {tuple(w_b.shape)} and {tuple(w_a.shape)}")
    if w_b.shape[1] != w_a.shape[0] or w_b.shape[1] == 0:
        raise ValueError(f"LoRA rank mismatch: B {tuple(w_b.shape)} vs A {tuple(w_a.shape)}")
    if not w_b.is_floating_point() or not w_a.is_floating_point():
        raise TypeError("LoRA factors must have floating-point dtypes")
    if not bool(torch.isfinite(w_b.detach()).all().item()) or not bool(torch.isfinite(w_a.detach()).all().item()):
        raise ValueError("LoRA factors must contain only finite values")
    return {
        "kind": "lora",
        "target_kind": target_kind,
        "w_b": w_b,
        "w_a": w_a,
        "scaling": _validate_positive_finite(scaling, "scaling"),
        "rank": int(w_b.shape[1]),
        "d_out": int(w_b.shape[0]),
        "d_in": int(w_a.shape[1]),
    }


def _resolve_steering_targets(
    steering_vec: torch.Tensor | Mapping[str, Any] | tuple[torch.Tensor, Mapping[str, Any]],
    k_proj: torch.Tensor,
    d_out: int,
) -> tuple[torch.Tensor, bool]:
    """Normalize ``steering_vec`` into column-major target matrix ``S_col`` of shape ``(d_out, n_trig)``.

    Returns ``(s_col, is_zero_steering)``.
    """
    n_trig = int(k_proj.shape[1])
    if isinstance(steering_vec, tuple) and len(steering_vec) == 2 and isinstance(steering_vec[1], Mapping):
        steering_vec = steering_vec[1]
    if isinstance(steering_vec, Mapping):
        if "steering_vector" not in steering_vec:
            raise ValueError("steering_vec mapping must contain 'steering_vector'")
        steering_vec = steering_vec["steering_vector"]

    if not isinstance(steering_vec, torch.Tensor):
        raise TypeError(
            f"steering_vec must be a torch.Tensor, subspace info dict, or (basis, info) tuple; "
            f"got {type(steering_vec).__name__}"
        )
    if not steering_vec.is_floating_point():
        raise TypeError(f"steering_vec must have a floating-point dtype, got {steering_vec.dtype}")

    s64 = steering_vec.detach().to(device="cpu", dtype=torch.float64)
    if s64.ndim not in (1, 2):
        raise ValueError(f"steering_vec must be 1D or 2D, got shape {tuple(s64.shape)}")
    if not bool(torch.isfinite(s64).all().item()):
        raise ValueError("steering_vec must contain only finite values")

    if s64.ndim == 1:
        if int(s64.shape[0]) != d_out:
            raise ValueError(f"steering_vec length {s64.shape[0]} must match weight d_out={d_out}")
        if float(s64.norm().item()) <= 1e-12:
            return torch.zeros((d_out, n_trig), dtype=torch.float64), True
        return s64.unsqueeze(1).expand(d_out, n_trig).contiguous(), False

    if s64.numel() == 0:
        if int(s64.shape[0]) != d_out and int(s64.shape[1]) != d_out:
            raise ValueError(f"Empty steering_vec shape {tuple(s64.shape)} incompatible with d_out={d_out}")
        return torch.zeros((d_out, n_trig), dtype=torch.float64), True

    if s64.shape == (n_trig, d_out) and (n_trig != d_out or s64.shape[1] == d_out):
        if float(s64.norm().item()) <= 1e-12:
            return torch.zeros((d_out, n_trig), dtype=torch.float64), True
        return s64.T.contiguous(), False

    if int(s64.shape[0]) != d_out:
        raise ValueError(
            f"2D steering_vec shape {tuple(s64.shape)} must be ({n_trig}, {d_out}) or ({d_out}, r)"
        )

    if float(s64.norm().item()) <= 1e-12:
        return torch.zeros((d_out, n_trig), dtype=torch.float64), True

    # Subspace matrix (d_out, r): map mode 0 uniformly and higher modes onto centered trigger variation
    r_sub = int(s64.shape[1])
    s_col = s64[:, 0:1].expand(d_out, n_trig).clone()
    if r_sub > 1 and n_trig >= 2:
        k_rem = k_proj - k_proj.mean(dim=1, keepdim=True)
        u_k, sv_k, _ = torch.linalg.svd(k_rem, full_matrices=False)
        for j in range(1, min(r_sub, int(sv_k.numel()) + 1)):
            sv_j = float(sv_k[j - 1].item())
            if sv_j <= 1e-10:
                break
            coeff = ((u_k[:, j - 1] @ k_rem) / (sv_j / math.sqrt(float(n_trig)))).unsqueeze(0)
            s_col = s_col + (s64[:, j : j + 1] @ coeff)
    return s_col.contiguous(), False


def _fold_nullspace_delta_into_lora(
    b64: torch.Tensor,
    a64: torch.Tensor,
    delta_w: torch.Tensor,
    *,
    scaling: float,
    p0: torch.Tensor,
    has_constraint: bool,
    lora_fold_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Fold null-space update ``delta_w`` into rank-``r`` LoRA ``(B, A)`` preserving ``P_0``."""
    d_out, rank = int(b64.shape[0]), int(b64.shape[1])
    d_in = int(a64.shape[1])
    delta_ba = delta_w / scaling
    delta_norm = float(delta_ba.norm().item())
    if delta_norm <= 1e-18:
        return b64.clone(), a64.clone(), "noop_zero_delta"

    u_d, s_d, vh_d = torch.linalg.svd(delta_ba, full_matrices=False)
    sv_tol = 1e-10 * max(float(s_d[0].item()), 1e-12)
    active_sv = int((s_d > sv_tol).sum().item())

    # 1. Zero-slot fill preserves existing weights bitwise and keeps A_new rows strictly in range(P_0)
    if lora_fold_mode in {"auto", "zero_slot"}:
        col_norms = b64.norm(dim=0)
        zero_slots = [int(i) for i in torch.nonzero(col_norms <= 1e-12, as_tuple=False).squeeze(-1).tolist()]
        b_is_zero = float(b64.norm().item()) <= 1e-12
        if zero_slots and (active_sv <= len(zero_slots) or b_is_zero or lora_fold_mode == "zero_slot"):
            k_fill = min(len(zero_slots), int(s_d.numel()))
            b_new = b64.clone()
            a_new = a64.clone()
            s_sqrt = s_d[:k_fill].clamp_min(0.0).sqrt()
            for idx_k, slot in enumerate(zero_slots[:k_fill]):
                b_new[:, slot] = u_d[:, idx_k] * s_sqrt[idx_k]
                a_new[slot, :] = (s_sqrt[idx_k] * vh_d[idx_k, :]) @ p0
            return b_new.contiguous(), a_new.contiguous(), "zero_slot_exact"

    # 2. Null-space partitioned SVD when preserved prose component uses fewer than rank slots
    w0 = b64 @ a64
    if has_constraint and lora_fold_mode == "auto":
        eye_in = torch.eye(d_in, dtype=torch.float64)
        p_pres = eye_in - p0
        w_pres = w0 @ p_pres
        u_p, s_p, vh_p = torch.linalg.svd(w_pres, full_matrices=False)
        pres_tol = 1e-10 * max(float(s_p[0].item()), 1e-12)
        r_pres = int((s_p > pres_tol).sum().item()) if float(s_p[0].item()) > 1e-12 else 0
        if r_pres < rank:
            b_out = torch.zeros((d_out, rank), dtype=torch.float64)
            a_out = torch.zeros((rank, d_in), dtype=torch.float64)
            if r_pres > 0:
                sp_sqrt = s_p[:r_pres].clamp_min(0.0).sqrt()
                b_out[:, :r_pres] = u_p[:, :r_pres] * sp_sqrt.unsqueeze(0)
                a_out[:r_pres, :] = (sp_sqrt.unsqueeze(1) * vh_p[:r_pres, :]) @ p_pres
            r_free = rank - r_pres
            w_null_new = (w0 @ p0) + delta_ba
            u_n, s_n, vh_n = torch.linalg.svd(w_null_new, full_matrices=False)
            k_n = min(r_free, int(s_n.numel()))
            if k_n > 0:
                sn_sqrt = s_n[:k_n].clamp_min(0.0).sqrt()
                b_out[:, r_pres : r_pres + k_n] = u_n[:, :k_n] * sn_sqrt.unsqueeze(0)
                a_out[r_pres : r_pres + k_n, :] = (sn_sqrt.unsqueeze(1) * vh_n[:k_n, :]) @ p0
            return b_out.contiguous(), a_out.contiguous(), "partitioned_nullspace_svd"

    # 3. Direct A-factor update inside P_0 keeping B fixed (0 leakage on preserved keys!)
    if lora_fold_mode == "update_a" or (has_constraint and lora_fold_mode == "auto"):
        delta_a = torch.linalg.lstsq(b64, delta_ba).solution @ p0
        return b64.clone(), (a64 + delta_a).contiguous(), "update_a_nullspace"

    # 4. Thin QR-SVD compression when unconstrained or explicitly requested
    k_aug = min(rank, int(s_d.numel()))
    s_sqrt = s_d[:k_aug].clamp_min(0.0).sqrt()
    b_aug = torch.cat([b64, u_d[:, :k_aug] * s_sqrt.unsqueeze(0)], dim=1)
    a_aug = torch.cat([a64, (s_sqrt.unsqueeze(1) * vh_d[:k_aug, :]) @ p0], dim=0)
    u_r, s_r, v_r = decompose_lora_qr_svd(b_aug, a_aug)
    k_keep = min(rank, int(s_r.numel()))
    s_top = s_r[:k_keep].clamp_min(0.0).sqrt()
    b_out = torch.zeros((d_out, rank), dtype=torch.float64)
    a_out = torch.zeros((rank, d_in), dtype=torch.float64)
    b_out[:, :k_keep] = u_r[:, :k_keep] * s_top.unsqueeze(0)
    a_out[:k_keep, :] = s_top.unsqueeze(1) * v_r[:, :k_keep].T
    return b_out.contiguous(), a_out.contiguous(), "qr_svd_rank_preserving"


@torch.no_grad()
def compile_steering_vector_into_nullspace_weight(
    weight_or_lora: Any,
    trigger_keys: torch.Tensor | Sequence[torch.Tensor],
    steering_vec: torch.Tensor | Mapping[str, Any] | tuple[torch.Tensor, Mapping[str, Any]],
    p_null: torch.Tensor | None,
    *,
    strength_alpha: float = DEFAULT_STRENGTH_ALPHA,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    weight_attr: str = "w3",
    expert_index: int | None = None,
    lora_fold_mode: str = "auto",
    max_norm_delta_ratio: float | None = None,
    inplace: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Compile a contrastive steering shift into static linear or LoRA weights inside ``P_0``.

    Solves the ridge-regularized AlphaEdit steering system:

        ``Delta W = alpha * (1/n S_col K_proj^T) (P_0 M_trig P_0 + ridge_lambda * I)^{-1} P_0``

    where ``K_proj = P_0 K_trig``. Guarantees ``Delta W K_0^T == 0`` for all keys in the
    kernel of ``P_0``, folding cleanly into dense ``nn.Linear``/tensor weights or rank-r
    LoRA ``(B, A)`` factors without adding runtime hooks.
    """
    alpha_f = _validate_nonneg_finite(strength_alpha, "strength_alpha")
    lam_f = _validate_positive_finite(ridge_lambda, "ridge_lambda")
    if lora_fold_mode not in VALID_LORA_FOLD_MODES:
        raise ValueError(
            f"lora_fold_mode must be one of {sorted(VALID_LORA_FOLD_MODES)}, got {lora_fold_mode!r}"
        )
    if max_norm_delta_ratio is not None:
        norm_cap: float | None = _validate_unit_interval(max_norm_delta_ratio, "max_norm_delta_ratio", allow_zero=False)
    else:
        norm_cap = None
    in_place = _validate_bool(inplace, "inplace")

    work_obj = weight_or_lora if in_place else (
        weight_or_lora.clone()
        if isinstance(weight_or_lora, torch.Tensor)
        else (
            tuple(t.clone() for t in weight_or_lora)
            if isinstance(weight_or_lora, tuple)
            else (
                {k: v.clone() for k, v in weight_or_lora.items()}
                if isinstance(weight_or_lora, Mapping)
                else copy.deepcopy(weight_or_lora)
            )
        )
    )
    target = _resolve_compile_target(work_obj, weight_attr=weight_attr, expert_index=expert_index)
    d_out, d_in = int(target["d_out"]), int(target["d_in"])

    trig_2d = _validate_activation_matrix(trigger_keys, "trigger_keys", min_rows=1, allow_1d=True)
    if int(trig_2d.shape[1]) != d_in:
        raise ValueError(
            f"trigger_keys feature width {trig_2d.shape[1]} must match weight d_in={d_in}"
        )

    if p_null is None:
        p0 = torch.eye(d_in, dtype=torch.float64)
        has_constraint = False
    else:
        p0 = _validate_square_projector(p_null, "p_null", expected_dim=d_in)
        eye_in = torch.eye(d_in, dtype=torch.float64)
        has_constraint = float((eye_in - p0).abs().max().item()) > 1e-9

    k_col = trig_2d.T.contiguous()  # (d_in, n_trig)
    k_proj = (p0 @ k_col).contiguous()
    s_col, is_zero_steer = _resolve_steering_targets(steering_vec, k_proj, d_out)

    if target["kind"] == "dense":
        w_before64 = target["weight"].detach().to(device="cpu", dtype=torch.float64)
    else:
        w_before64 = float(target["scaling"]) * (
            target["w_b"].detach().to(device="cpu", dtype=torch.float64)
            @ target["w_a"].detach().to(device="cpu", dtype=torch.float64)
        )
    base_fro = float(w_before64.norm().item())

    def _noop_report(reason: str) -> dict[str, Any]:
        return {
            "applied": False,
            "target_kind": target["target_kind"],
            "lora_fold_mode_used": reason,
            "delta_w_fro": 0.0,
            "relative_delta_norm": 0.0,
            "effective_delta_rank": 0,
            "trigger_alignment_cosine": 0.0,
            "trigger_shift_norm": 0.0,
            "trigger_fit_rel_err": 1.0,
            "nullspace_leakage_norm": 0.0,
            "actual_weight_nullspace_leakage_norm": 0.0,
            "strength_alpha": alpha_f,
            "ridge_lambda": lam_f,
            "budget_clamped": False,
        }

    if alpha_f == 0.0 or is_zero_steer:
        return work_obj, _noop_report("noop_zero_steering")

    if float(k_proj.norm().item()) <= 1e-12:
        return work_obj, _noop_report("noop_nullspace_collapsed")

    n_trig = float(k_col.shape[1])
    m_proj = (k_proj @ k_proj.T) / n_trig
    m_proj = 0.5 * (m_proj + m_proj.T)
    gram = m_proj + lam_f * torch.eye(d_in, dtype=torch.float64)
    rhs_t = (k_proj @ (alpha_f * s_col).T) / n_trig  # (d_in, d_out)
    sol_t = torch.linalg.solve(gram, rhs_t)
    delta_w = (sol_t.T @ p0).contiguous()  # (d_out, d_in)

    raw_fro = float(delta_w.norm().item())
    if raw_fro <= 1e-15:
        return work_obj, _noop_report("noop_zero_delta")

    budget_clamped = False
    if norm_cap is not None and base_fro > 1e-12 and raw_fro > norm_cap * base_fro:
        delta_w = delta_w * ((norm_cap * base_fro) / raw_fro)
        budget_clamped = True

    if target["kind"] == "dense":
        w_param: torch.Tensor = target["weight"]
        w_new = (w_before64 + delta_w).to(device=w_param.device, dtype=w_param.dtype)
        w_param.copy_(w_new)
        fold_used = "dense_exact"
        w_after64 = w_param.detach().to(device="cpu", dtype=torch.float64)
    else:
        b_param: torch.Tensor = target["w_b"]
        a_param: torch.Tensor = target["w_a"]
        b64 = b_param.detach().to(device="cpu", dtype=torch.float64)
        a64 = a_param.detach().to(device="cpu", dtype=torch.float64)
        b_new, a_new, fold_used = _fold_nullspace_delta_into_lora(
            b64,
            a64,
            delta_w,
            scaling=float(target["scaling"]),
            p0=p0,
            has_constraint=has_constraint,
            lora_fold_mode=lora_fold_mode,
        )
        b_param.copy_(b_new.to(device=b_param.device, dtype=b_param.dtype))
        a_param.copy_(a_new.to(device=a_param.device, dtype=a_param.dtype))
        w_after64 = float(target["scaling"]) * (
            b_param.detach().to(device="cpu", dtype=torch.float64)
            @ a_param.detach().to(device="cpu", dtype=torch.float64)
        )

    actual_delta = (w_after64 - w_before64).contiguous()
    if not bool(torch.isfinite(actual_delta).all().item()):
        return work_obj, {
            **_noop_report(fold_used),
            "applied": True,
            "non_finite_delta": True,
            "delta_w_fro": None,
            "relative_delta_norm": None,
            "nullspace_leakage_norm": None,
            "actual_weight_nullspace_leakage_norm": None,
            "budget_clamped": budget_clamped,
        }
    actual_fro = float(actual_delta.norm().item())
    eye_in = torch.eye(d_in, dtype=torch.float64)
    p_pres = eye_in - p0
    analytic_leak = float((delta_w @ p_pres).norm().item()) if has_constraint else 0.0
    actual_leak = float((actual_delta @ p_pres).norm().item()) if has_constraint else 0.0

    achieved = actual_delta @ k_col
    target_out = alpha_f * s_col
    mean_ach = achieved.mean(dim=1)
    mean_tgt = target_out.mean(dim=1)
    ach_n = float(mean_ach.norm().item())
    tgt_n = float(mean_tgt.norm().item())
    align_cos = (
        float(((mean_ach @ mean_tgt) / (ach_n * tgt_n)).item())
        if ach_n > 1e-12 and tgt_n > 1e-12
        else 0.0
    )
    tgt_fro = float(target_out.norm().item())
    fit_err = float((achieved - target_out).norm().item() / tgt_fro) if tgt_fro > 1e-12 else 0.0
    s_delta = torch.linalg.svdvals(actual_delta)
    sv_cut = 1e-6 * max(float(s_delta[0].item()), 1e-12) if s_delta.numel() > 0 else 1e-12
    eff_rank = int((s_delta > sv_cut).sum().item()) if actual_fro > 1e-12 else 0

    return work_obj, {
        "applied": actual_fro > 1e-12,
        "target_kind": target["target_kind"],
        "lora_fold_mode_used": fold_used,
        "delta_w_fro": actual_fro,
        "relative_delta_norm": (actual_fro / base_fro) if base_fro > 1e-12 else 0.0,
        "effective_delta_rank": eff_rank,
        "trigger_alignment_cosine": align_cos,
        "trigger_shift_norm": ach_n,
        "trigger_fit_rel_err": fit_err,
        "nullspace_leakage_norm": analytic_leak,
        "actual_weight_nullspace_leakage_norm": actual_leak,
        "strength_alpha": alpha_f,
        "ridge_lambda": lam_f,
        "budget_clamped": budget_clamped,
    }


# ---------------------------------------------------------------------------
# 4. End-to-End Verified Multi-Layer Compilation & ANSE Rollback
# ---------------------------------------------------------------------------


def _normalize_token_id_batch(
    raw_ids: torch.Tensor | Sequence[Sequence[int]] | Sequence[Mapping[str, Any]],
    name: str,
    *,
    score_from: int,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Validate 2D integer token-ID batches or chunk mappings into CPU ``int64`` tensor."""
    if isinstance(raw_ids, torch.Tensor):
        if raw_ids.ndim != 2 or raw_ids.shape[0] == 0 or raw_ids.shape[1] < 2:
            raise ValueError(
                f"{name} must be a non-empty 2D tensor with sequence length >= 2, got {tuple(raw_ids.shape)}"
            )
        if raw_ids.is_floating_point() or raw_ids.dtype == torch.bool:
            raise TypeError(f"{name} must have an integer dtype, got {raw_ids.dtype}")
        ids_t = raw_ids.detach().to(device="cpu", dtype=torch.long)
    elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
        if not raw_ids:
            raise ValueError(f"{name} must not be empty")
        if isinstance(raw_ids[0], Mapping):
            rows: list[list[int]] = []
            for idx, item in enumerate(raw_ids):
                if not isinstance(item, Mapping) or "ids" not in item:
                    raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids'")
                row_t = torch.as_tensor(item["ids"], dtype=torch.long)
                if row_t.ndim != 1 or row_t.numel() < 2:
                    raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
                rows.append([int(x) for x in row_t.tolist()])
            ids_t = torch.tensor(rows, dtype=torch.long)
        else:
            rows = []
            seq_len: int | None = None
            for idx, row in enumerate(raw_ids):
                if isinstance(row, torch.Tensor):
                    if row.is_floating_point() or row.dtype == torch.bool:
                        raise TypeError(f"{name}[{idx}] must have an integer dtype")
                    vals = [int(x) for x in row.detach().cpu().tolist()]
                elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                    vals = []
                    for tok in row:
                        if isinstance(tok, bool) or not isinstance(tok, int):
                            raise TypeError(f"{name}[{idx}] entries must be ints")
                        vals.append(int(tok))
                else:
                    raise TypeError(f"{name}[{idx}] must be a sequence of ints")
                if seq_len is None:
                    seq_len = len(vals)
                elif len(vals) != seq_len:
                    raise ValueError(f"All rows in {name} must share sequence length {seq_len}")
                rows.append(vals)
            ids_t = torch.tensor(rows, dtype=torch.long)
    else:
        raise TypeError(f"{name} must be a 2D integer tensor, sequence of token sequences, or chunk mappings")

    if ids_t.ndim != 2 or ids_t.shape[0] < 2 or ids_t.shape[1] < 2:
        raise ValueError(f"{name} must contain at least 2 sequences of length >= 2, got {tuple(ids_t.shape)}")
    if bool((ids_t < 0).any().item()):
        raise ValueError(f"{name} must contain non-negative token IDs")
    if vocab_size is not None and bool((ids_t >= vocab_size).any().item()):
        raise ValueError(f"{name} token IDs must lie in [0, {vocab_size})")
    start_pos = max(0, score_from - 1)
    if start_pos >= int(ids_t.shape[1]) - 1:
        raise ValueError(f"{name} sequence length {ids_t.shape[1]} too short for score_from={score_from}")
    return ids_t.contiguous()


def _validate_guard_chunks(
    guard_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    score_from: int,
    vocab_size: int,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate guard chunks and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(guard_chunks, (str, bytes, bytearray)) or not isinstance(guard_chunks, Sequence) or not guard_chunks:
        raise ValueError("guard_chunks must be a non-empty sequence of chunk mappings")
    expert_ids: list[str] = []
    rows: list[torch.Tensor] = []
    seq_len: int | None = None
    for idx, ch in enumerate(guard_chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"guard_chunks[{idx}] must be a mapping containing 'expert_id' and 'ids'")
        eid = ch["expert_id"]
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"guard_chunks[{idx}]['expert_id'] must be a non-empty string")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"guard_chunks[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"guard_chunks[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All guard_chunks must share sequence length {seq_len}")
        expert_ids.append(eid)
        rows.append(ids_t)

    start_pos = max(0, score_from - 1)
    if seq_len is None or start_pos >= seq_len - 1:
        raise ValueError(f"guard_chunks length {seq_len} too short for score_from={score_from}")
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"guard_chunks must include at least one general chunk starting with {general_prefix!r}")
    return torch.stack(rows, dim=0), is_general, expert_ids


def _validate_repetition_tokens(
    repetition_token_ids: Sequence[int] | torch.Tensor | None,
    vocab_size: int | None,
) -> list[int] | None:
    """Validate optional loop/repetition token IDs."""
    if repetition_token_ids is None:
        return None
    if isinstance(repetition_token_ids, torch.Tensor):
        if repetition_token_ids.ndim != 1 or repetition_token_ids.is_floating_point() or repetition_token_ids.dtype == torch.bool:
            raise ValueError("repetition_token_ids tensor must be a 1D integer tensor")
        raw = [int(t) for t in repetition_token_ids.tolist()]
    elif isinstance(repetition_token_ids, Sequence) and not isinstance(repetition_token_ids, (str, bytes, bytearray)):
        raw = [_validate_nonneg_int(t, "repetition_token_id") for t in repetition_token_ids]
    else:
        raise TypeError("repetition_token_ids must be a sequence of ints or 1D integer tensor")
    unique = sorted(set(raw))
    if not unique:
        raise ValueError("repetition_token_ids must not be empty")
    if vocab_size is not None and any(t < 0 or t >= vocab_size for t in unique):
        raise ValueError(f"repetition_token_ids must lie in [0, {vocab_size})")
    return unique


def _invalid_eval_metrics() -> dict[str, Any]:
    """Return canonical non-finite evaluation sentinel triggering ANSE rollback."""
    return {
        "finite": False,
        "domain_nll": None,
        "nll_mean": None,
        "loop_logit_mean": None,
        "loop_prob_mean": None,
    }


@torch.no_grad()
def _evaluate_model_metrics(
    model: nn.Module,
    *,
    ids_guard: torch.Tensor | None,
    expert_ids: list[str] | None,
    neg_ids: torch.Tensor | None,
    rep_tokens: list[int] | None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None,
    cfg: ContrastiveWeightCompilerConfig,
    expected_domains: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate per-domain finite NLLs and optional loop-token logit/probability metrics."""
    if eval_fn is not None:
        try:
            raw = eval_fn(model)
        except Exception:
            return _invalid_eval_metrics()
        if not isinstance(raw, Mapping) or not raw:
            return _invalid_eval_metrics()
        src = raw.get("domain_nll", raw.get("domain_nlls", raw))
        if not isinstance(src, Mapping) or not src:
            return _invalid_eval_metrics()
        cleaned: dict[str, float] = {}
        loop_logit_val = raw.get("loop_logit_mean")
        loop_prob_val = raw.get("loop_prob_mean")
        for k, v in sorted(src.items()):
            ks = str(k)
            if ks in {"nll_mean", "loop_logit_mean", "loop_prob_mean", "all_finite"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return _invalid_eval_metrics()
            fv = float(v)
            if not math.isfinite(fv) or fv <= 0.0:
                return _invalid_eval_metrics()
            cleaned[ks] = fv
        if not cleaned or (expected_domains is not None and set(cleaned.keys()) != expected_domains):
            return _invalid_eval_metrics()
        if not any(k.startswith(cfg.general_prefix) for k in cleaned):
            return _invalid_eval_metrics()
        return {
            "finite": True,
            "domain_nll": cleaned,
            "nll_mean": float(sum(cleaned.values()) / len(cleaned)),
            "loop_logit_mean": float(loop_logit_val) if isinstance(loop_logit_val, (int, float)) else None,
            "loop_prob_mean": float(loop_prob_val) if isinstance(loop_prob_val, (int, float)) else None,
        }

    assert ids_guard is not None and expert_ids is not None
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)

    def _run_pre_ln(seq_ids: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b0 in range(0, int(seq_ids.shape[0]), cfg.batch_size):
            xb = compute_embeddings_batch(model, seq_ids[b0 : b0 + cfg.batch_size].to(dev))
            for blk in model.blocks:
                xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
                xb = forward_block_post_mlp(blk, xm, mi)
            parts.append(xb)
        return torch.cat(parts, dim=0)

    h_pre = _run_pre_ln(ids_guard)
    if not bool(torch.isfinite(h_pre).all().item()):
        return _invalid_eval_metrics()

    ln_b = getattr(model.ln, "bias", None)
    ln_bias = ln_b if isinstance(ln_b, torch.Tensor) else torch.zeros_like(model.ln.weight)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    losses = per_chunk_causal_midrow_nll(
        h_pre,
        model.ln.weight,
        ln_bias,
        ln_eps,
        model.head.weight,
        ids_guard.to(dev),
        score_from=cfg.score_from,
        seq_batch_size=cfg.batch_size,
    )
    if not bool(torch.isfinite(losses).all().item()):
        return _invalid_eval_metrics()

    grouped: dict[str, list[float]] = {}
    for eid, val in zip(expert_ids, losses.detach().cpu().tolist(), strict=True):
        grouped.setdefault(eid, []).append(float(val))
    dom_nll = {dom: float(sum(vs) / len(vs)) for dom, vs in sorted(grouped.items())}
    nll_mean = float(losses.mean().item())

    loop_logit_mean: float | None = None
    loop_prob_mean: float | None = None
    if rep_tokens is not None and neg_ids is not None:
        h_neg = _run_pre_ln(neg_ids)
        if not bool(torch.isfinite(h_neg).all().item()):
            return _invalid_eval_metrics()
        start_pos = max(0, cfg.score_from - 1)
        sub_h = h_neg[:, start_pos:-1, :].to(torch.float32)
        h_dim = int(sub_h.shape[-1])
        h_norm = F.layer_norm(
            sub_h,
            (h_dim,),
            model.ln.weight.detach().to(device=dev, dtype=torch.float32),
            ln_bias.detach().to(device=dev, dtype=torch.float32),
            eps=ln_eps,
        )
        logits = F.linear(h_norm.to(dtype=model.head.weight.dtype), model.head.weight).to(torch.float32)
        if not bool(torch.isfinite(logits).all().item()):
            return _invalid_eval_metrics()
        probs = F.softmax(logits, dim=-1)
        rep_idx = torch.tensor(rep_tokens, dtype=torch.long, device=dev)
        loop_logit_mean = float(logits.index_select(-1, rep_idx).mean().item())
        loop_prob_mean = float(probs.index_select(-1, rep_idx).sum(dim=-1).mean().item())

    return {
        "finite": True,
        "domain_nll": dom_nll,
        "nll_mean": nll_mean,
        "loop_logit_mean": loop_logit_mean,
        "loop_prob_mean": loop_prob_mean,
    }


def _compute_anse_summary(
    ref_dom_nll: Mapping[str, float],
    cand_eval: Mapping[str, Any],
    *,
    cfg: ContrastiveWeightCompilerConfig,
) -> dict[str, Any]:
    """Compute per-domain drift ratios, guard compliance, and objective ANSE energy."""
    ref_mean = float(sum(ref_dom_nll.values()) / len(ref_dom_nll))
    spec_keys = [k for k in sorted(ref_dom_nll.keys()) if not k.startswith(cfg.general_prefix)]
    if not cand_eval["finite"] or cand_eval["domain_nll"] is None:
        return {
            "finite": False,
            "nll_mean": None,
            "specialist_nll_mean": None,
            "general_drift": None,
            "general_drift_pct": None,
            "worst_domain_id": None,
            "worst_domain_drift": None,
            "worst_domain_drift_pct": None,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain_energy),
            "per_domain": {},
        }

    cand_dom: Mapping[str, float] = cand_eval["domain_nll"]
    per_domain: dict[str, dict[str, Any]] = {}
    worst_dom = ""
    worst_ratio = -math.inf
    for dom in sorted(ref_dom_nll.keys()):
        b_val = float(ref_dom_nll[dom])
        c_val = float(cand_dom[dom])
        ratio = c_val / max(b_val, 1e-12)
        drift = ratio - 1.0
        dom_ok = drift <= cfg.max_domain_drift + 1e-12
        per_domain[dom] = {
            "ref_nll": b_val,
            "cand_nll": c_val,
            "mean_diff": c_val - b_val,
            "ratio": ratio,
            "drift_pct": 100.0 * drift,
            "guard_ok": bool(dom_ok),
        }
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_dom = dom

    gen_keys = [k for k in sorted(per_domain.keys()) if k.startswith(cfg.general_prefix)]
    general_ratio = max(float(per_domain[k]["ratio"]) for k in gen_keys)
    general_drift = general_ratio - 1.0
    general_ok = general_drift <= cfg.max_domain_drift + 1e-12
    all_ok = all(bool(info["guard_ok"]) for info in per_domain.values())
    guard_ok = all_ok if cfg.enforce_all_domain_guard else general_ok

    penalty = (
        float(cfg.maximum_pain_energy)
        if not guard_ok
        else (float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_general_ratio else 0.0)
    )
    loop_term = 0.0
    if cfg.loop_energy_weight > 0.0 and cand_eval["loop_logit_mean"] is not None:
        loop_term = float(cfg.loop_energy_weight) * float(cand_eval["loop_logit_mean"])
    cand_mean = float(cand_eval["nll_mean"])
    spec_mean = (
        float(sum(cand_dom[k] for k in spec_keys) / len(spec_keys))
        if spec_keys
        else cand_mean
    )
    return {
        "finite": True,
        "nll_mean": cand_mean,
        "specialist_nll_mean": spec_mean,
        "general_drift": general_drift,
        "general_drift_pct": 100.0 * general_drift,
        "worst_domain_id": worst_dom,
        "worst_domain_drift": worst_ratio - 1.0,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0),
        "guard_ok": bool(guard_ok),
        "anse_energy": float((cand_mean - ref_mean) + loop_term + penalty),
        "per_domain": per_domain,
    }


def _swiglu_act_for_expert(
    moe_mlp: CrowdLoRAMoEMLP,
    mi_flat: torch.Tensor,
    expert_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token SwiGLU intermediate states and routing gates for one LoRA expert."""
    w_dt = moe_mlp.base_mlp.w1.weight.dtype
    x_base = mi_flat.to(dtype=w_dt)
    h1 = moe_mlp.base_mlp.w1(x_base).to(torch.float32)
    h2 = moe_mlp.base_mlp.w2(x_base).to(torch.float32)
    x_fp32 = mi_flat.to(torch.float32)

    logits = moe_mlp.router(x_fp32)
    probs = torch.sigmoid(logits) if moe_mlp.routing_mode == "sigma_topk" else F.softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)
    if moe_mlp.routing_mode != "sigma_topk":
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    res_mask = moe_mlp.resident_mask.to(device=logits.device)
    gates = torch.where(~res_mask[top_idx], torch.zeros_like(top_vals), top_vals)

    e_idx = int(expert_key)
    gate_tok = (gates * (top_idx == e_idx).to(torch.float32)).sum(dim=-1, keepdim=True)
    exp = moe_mlp.experts[expert_key]
    dh1 = exp.scaling * F.linear(F.linear(x_fp32, exp.w1_A), exp.w1_B)
    dh2 = exp.scaling * F.linear(F.linear(x_fp32, exp.w2_A), exp.w2_B)
    act_full = F.silu(h1 + dh1) * (h2 + dh2)
    return act_full, gate_tok


def _projector_from_cfg(
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    cfg: ContrastiveWeightCompilerConfig,
) -> torch.Tensor:
    """Build null-space projector ``P_0`` from validated compiler configuration."""
    p0, _ = build_prose_nullspace_projector(
        prose_keys,
        sv_threshold=cfg.nullspace_sv_threshold,
        sv_rel_tol=cfg.nullspace_sv_rel_tol,
        variance_threshold=cfg.nullspace_variance_threshold,
        min_null_rank=cfg.min_null_rank,
    )
    return p0


def _compile_with_cfg(
    target_obj: Any,
    trig_keys: torch.Tensor,
    steer_vec: torch.Tensor,
    p_null: torch.Tensor | None,
    cfg: ContrastiveWeightCompilerConfig,
    *,
    weight_attr: str | None = None,
) -> dict[str, Any]:
    """Invoke null-space weight compiler with shared validated hyperparameters."""
    kwargs: dict[str, Any] = {
        "strength_alpha": cfg.strength_alpha,
        "ridge_lambda": cfg.ridge_lambda,
        "lora_fold_mode": cfg.lora_fold_mode,
        "max_norm_delta_ratio": cfg.max_norm_delta_ratio,
        "inplace": True,
    }
    if weight_attr is not None:
        kwargs["weight_attr"] = weight_attr
    _, comp_rep = compile_steering_vector_into_nullspace_weight(
        target_obj,
        trig_keys,
        steer_vec,
        p_null,
        **kwargs,
    )
    return comp_rep


def _build_per_token_steering_targets(
    pos_res: torch.Tensor,
    neg_res: torch.Tensor,
    sub_info: Mapping[str, Any],
    *,
    n_trig: int,
) -> torch.Tensor:
    """Construct rank-r per-trigger steering targets when paired, else rank-1 mean vector."""
    eff_rank = int(sub_info["effective_rank"])
    if eff_rank <= 1 or pos_res.shape[0] != neg_res.shape[0]:
        return sub_info["steering_vector"]
    basis: torch.Tensor = sub_info["subspace_basis"]
    diffs = pos_res.to(torch.float64) - neg_res.to(torch.float64)
    proj = (diffs @ basis) @ basis.T
    if n_trig == 2 * int(proj.shape[0]):
        return torch.cat([proj, proj], dim=0).contiguous()
    return sub_info["steering_vector"]


def _compile_explicit_activation_targets(
    work_model: nn.Module,
    *,
    pos_activations: Mapping[int | str, torch.Tensor],
    neg_activations: Mapping[int | str, torch.Tensor],
    trigger_activations: Mapping[int | str, torch.Tensor] | None,
    prose_keys: Mapping[int | str, torch.Tensor] | torch.Tensor | None,
    p_nulls: Mapping[int | str, torch.Tensor] | torch.Tensor | None,
    cfg: ContrastiveWeightCompilerConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile contrastive steering across explicitly supplied per-key activation mappings."""
    shared_keys = sorted(set(str(k) for k in pos_activations.keys()) & set(str(k) for k in neg_activations.keys()))
    if not shared_keys:
        raise ValueError("pos_activations and neg_activations must share at least one target key")

    pos_str = {str(k): v for k, v in pos_activations.items()}
    neg_str = {str(k): v for k, v in neg_activations.items()}
    trig_str = {str(k): v for k, v in trigger_activations.items()} if trigger_activations is not None else None

    sub_reports: list[dict[str, Any]] = []
    compile_reports: list[dict[str, Any]] = []

    def _resolve_target_module(key: str) -> Any:
        if key in {"default", "self"}:
            return work_model
        cur: Any = work_model
        for part in key.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            elif isinstance(cur, (nn.ModuleDict, Mapping)) and part in cur:
                cur = cur[part]
            elif part.isdigit() and isinstance(cur, (nn.ModuleList, Sequence)):
                cur = cur[int(part)]
            else:
                raise ValueError(f"Target key {key!r} could not be resolved on model")
        return cur

    for key in shared_keys:
        p_t = pos_str[key]
        n_t = neg_str[key]
        sub_basis, sub_info = extract_oof_contrastive_steering_subspace(
            p_t, n_t, rank=cfg.rank, min_oof_snr=cfg.min_oof_snr
        )
        sub_reports.append({"target": key, "oof_snr": sub_info["oof_snr"], "effective_rank": sub_info["effective_rank"], "accepted": sub_info["accepted"]})
        if not sub_info["accepted"]:
            continue

        p_null_k: torch.Tensor | None = None
        if isinstance(p_nulls, torch.Tensor):
            p_null_k = p_nulls
        elif isinstance(p_nulls, Mapping):
            p_null_k = {str(k): v for k, v in p_nulls.items()}.get(key)
        elif isinstance(prose_keys, torch.Tensor):
            p_null_k = _projector_from_cfg(prose_keys, cfg)
        elif isinstance(prose_keys, Mapping):
            pk_map = {str(k): v for k, v in prose_keys.items()}
            if key in pk_map:
                p_null_k = _projector_from_cfg(pk_map[key], cfg)

        trig_k = trig_str[key] if (trig_str is not None and key in trig_str) else torch.cat([n_t, p_t], dim=0)
        target_obj = _resolve_target_module(key)
        comp_rep = _compile_with_cfg(
            target_obj,
            trig_k,
            sub_info["steering_vector"],
            p_null_k,
            cfg,
            weight_attr=cfg.target_proj if cfg.target_proj != "auto" else "w3",
        )
        compile_reports.append({"target": key, "subspace_rank": int(sub_basis.shape[1]), **comp_rep})
    return sub_reports, compile_reports


@torch.no_grad()
def execute_verified_contrastive_weight_compilation(
    model: nn.Module,
    *,
    pos_chunks_or_ids: torch.Tensor | Sequence[Sequence[int]] | Sequence[Mapping[str, Any]] | None = None,
    neg_chunks_or_ids: torch.Tensor | Sequence[Sequence[int]] | Sequence[Mapping[str, Any]] | None = None,
    guard_chunks: Sequence[Mapping[str, Any]] | None = None,
    pos_activations: Mapping[int | str, torch.Tensor] | None = None,
    neg_activations: Mapping[int | str, torch.Tensor] | None = None,
    trigger_activations: Mapping[int | str, torch.Tensor] | None = None,
    prose_keys: Mapping[int | str, torch.Tensor] | torch.Tensor | None = None,
    p_nulls: Mapping[int | str, torch.Tensor] | torch.Tensor | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_proj: str | None = None,
    strength_alpha: float | None = None,
    ridge_lambda: float | None = None,
    rank: int | None = None,
    min_oof_snr: float | None = None,
    max_domain_drift: float | None = None,
    repetition_token_ids: Sequence[int] | torch.Tensor | None = None,
    config: ContrastiveWeightCompilerConfig | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute multi-layer contrastive steering compilation gated by ANSE and atomic rollback.

    1. Captures bitwise parent SHA-256 snapshot and verifies zero runtime hooks.
    2. Extracts per-layer OOF-gated contrastive subspaces (rejecting noisy layers cleanly).
    3. Compiles static ``w3`` / ``down_proj`` LoRA or dense updates inside ``P_0`` with
       sequential hidden-state propagation.
    4. Commits iff ANSE ``Delta E < 0`` and worst-domain drift ``<= +1.0%``; otherwise
       restores the exact parent state bitwise and emits a SHA-256 provenance receipt.
    """
    if not isinstance(model, nn.Module) or not model.state_dict():
        raise ValueError("model must be a non-empty nn.Module")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if config is not None and not isinstance(config, ContrastiveWeightCompilerConfig):
        raise TypeError(f"config must be ContrastiveWeightCompilerConfig or None, got {type(config).__name__}")

    base = config or ContrastiveWeightCompilerConfig()
    cfg = ContrastiveWeightCompilerConfig(
        rank=base.rank if rank is None else rank,
        min_oof_snr=base.min_oof_snr if min_oof_snr is None else min_oof_snr,
        strength_alpha=base.strength_alpha if strength_alpha is None else strength_alpha,
        ridge_lambda=base.ridge_lambda if ridge_lambda is None else ridge_lambda,
        target_proj=base.target_proj if target_proj is None else target_proj,
        lora_fold_mode=base.lora_fold_mode,
        nullspace_sv_threshold=base.nullspace_sv_threshold,
        nullspace_sv_rel_tol=base.nullspace_sv_rel_tol,
        nullspace_variance_threshold=base.nullspace_variance_threshold,
        min_null_rank=base.min_null_rank,
        max_domain_drift=base.max_domain_drift if max_domain_drift is None else max_domain_drift,
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        min_energy_improvement=base.min_energy_improvement,
        require_strict_energy_drop=base.require_strict_energy_drop,
        enforce_all_domain_guard=base.enforce_all_domain_guard,
        loop_energy_weight=base.loop_energy_weight,
        min_expert_trigger_gate=base.min_expert_trigger_gate,
        max_norm_delta_ratio=base.max_norm_delta_ratio,
        general_prefix=base.general_prefix,
        score_from=base.score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )

    use_explicit = pos_activations is not None or neg_activations is not None
    if use_explicit:
        if pos_activations is None or neg_activations is None:
            raise ValueError("Both pos_activations and neg_activations must be provided together")
        if eval_fn is None and guard_chunks is None:
            raise ValueError("eval_fn or guard_chunks is required with explicit activations")
    else:
        if pos_chunks_or_ids is None or neg_chunks_or_ids is None:
            raise ValueError("pos_chunks_or_ids and neg_chunks_or_ids are required")
        if guard_chunks is None and eval_fn is None:
            raise ValueError("guard_chunks or eval_fn is required")

    work_model = model if cfg.inplace else copy.deepcopy(model)
    ids_guard: torch.Tensor | None = None
    is_gen_guard: np.ndarray | None = None
    expert_ids_guard: list[str] | None = None
    pos_ids: torch.Tensor | None = None
    neg_ids: torch.Tensor | None = None
    vocab_sz = int(work_model.head.weight.shape[0]) if hasattr(work_model, "head") else None
    rep_tokens = _validate_repetition_tokens(repetition_token_ids, vocab_sz)

    if guard_chunks is not None:
        if vocab_sz is None or not hasattr(work_model, "blocks"):
            raise ValueError("guard_chunks requires a CrowdV1WithEngram-compatible model")
        ids_guard, is_gen_guard, expert_ids_guard = _validate_guard_chunks(
            guard_chunks,
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            vocab_size=vocab_sz,
        )

    if not use_explicit:
        _validate_model_compatible_model(work_model)
        assert vocab_sz is not None
        pos_ids = _normalize_token_id_batch(
            pos_chunks_or_ids, "pos_chunks_or_ids", score_from=cfg.score_from, vocab_size=vocab_sz  # type: ignore[arg-type]
        )
        neg_ids = _normalize_token_id_batch(
            neg_chunks_or_ids, "neg_chunks_or_ids", score_from=cfg.score_from, vocab_size=vocab_sz  # type: ignore[arg-type]
        )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_sha = compute_module_state_sha256(work_model)
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        pre_hook_count = count_registered_runtime_hooks(work_model)

        parent_eval = _evaluate_model_metrics(
            work_model,
            ids_guard=ids_guard,
            expert_ids=expert_ids_guard,
            neg_ids=neg_ids,
            rep_tokens=rep_tokens,
            eval_fn=eval_fn,
            cfg=cfg,
        )
        if not parent_eval["finite"] or parent_eval["domain_nll"] is None:
            raise ValueError("Parent model evaluation returned invalid or non-finite metrics")
        parent_dom_nll: dict[str, float] = parent_eval["domain_nll"]
        parent_anse = _compute_anse_summary(parent_dom_nll, parent_eval, cfg=cfg)

        subspace_reports: list[dict[str, Any]] = []
        compile_reports: list[dict[str, Any]] = []

        if use_explicit:
            assert pos_activations is not None and neg_activations is not None
            subspace_reports, compile_reports = _compile_explicit_activation_targets(
                work_model,
                pos_activations=pos_activations,
                neg_activations=neg_activations,
                trigger_activations=trigger_activations,
                prose_keys=prose_keys,
                p_nulls=p_nulls,
                cfg=cfg,
            )
        else:
            assert pos_ids is not None and neg_ids is not None
            n_blocks = len(work_model.blocks)
            if layer_indices is None:
                target_layers = set(range(n_blocks))
            else:
                target_layers = {_validate_nonneg_int(idx, "layer_index") for idx in layer_indices}
                if not target_layers or any(idx >= n_blocks for idx in target_layers):
                    raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
            target_experts = (
                {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
                if expert_indices is not None
                else None
            )
            if ids_guard is not None and is_gen_guard is not None:
                prose_ids = ids_guard[torch.as_tensor(is_gen_guard, dtype=torch.bool)]
            else:
                prose_ids = None

            dev = work_model.wte.weight.device
            cos = work_model.cos_cached.to(dev)
            sin = work_model.sin_cached.to(dev)
            start_pos = max(0, cfg.score_from - 1)

            def _embed(seq_t: torch.Tensor | None) -> torch.Tensor | None:
                if seq_t is None:
                    return None
                return torch.cat(
                    [
                        compute_embeddings_batch(work_model, seq_t[b0 : b0 + cfg.batch_size].to(dev))
                        for b0 in range(0, int(seq_t.shape[0]), cfg.batch_size)
                    ],
                    dim=0,
                )

            x_pos = _embed(pos_ids)
            x_neg = _embed(neg_ids)
            x_prose = _embed(prose_ids)
            assert x_pos is not None and x_neg is not None

            for l_idx, blk in enumerate(work_model.blocks):
                xm_pos, mi_pos = forward_block_pre_mlp(blk, x_pos, cos, sin)
                xm_neg, mi_neg = forward_block_pre_mlp(blk, x_neg, cos, sin)
                xm_prose, mi_prose = (
                    forward_block_pre_mlp(blk, x_prose, cos, sin) if x_prose is not None else (None, None)
                )

                if l_idx in target_layers:
                    out_pos = forward_block_post_mlp(blk, xm_pos, mi_pos)[:, start_pos:-1, :]
                    out_neg = forward_block_post_mlp(blk, xm_neg, mi_neg)[:, start_pos:-1, :]
                    pos_flat = out_pos.reshape(-1, int(out_pos.shape[-1]))
                    neg_flat = out_neg.reshape(-1, int(out_neg.shape[-1]))
                    sub_basis, sub_info = extract_oof_contrastive_steering_subspace(
                        pos_flat, neg_flat, rank=cfg.rank, min_oof_snr=cfg.min_oof_snr
                    )
                    subspace_reports.append(
                        {
                            "layer": l_idx,
                            "oof_snr": sub_info["oof_snr"],
                            "effective_rank": sub_info["effective_rank"],
                            "accepted": sub_info["accepted"],
                        }
                    )

                    if sub_info["accepted"]:
                        mi_p_flat = mi_pos[:, start_pos:-1, :].reshape(-1, int(mi_pos.shape[-1]))
                        mi_n_flat = mi_neg[:, start_pos:-1, :].reshape(-1, int(mi_neg.shape[-1]))
                        mi_0_flat = (
                            mi_prose[:, start_pos:-1, :].reshape(-1, int(mi_prose.shape[-1]))
                            if mi_prose is not None
                            else None
                        )
                        steer_target = _build_per_token_steering_targets(
                            pos_flat, neg_flat, sub_info, n_trig=int(mi_p_flat.shape[0]) + int(mi_n_flat.shape[0])
                        )

                        mlp_mod = blk.mlp
                        if isinstance(mlp_mod, CrowdLoRAMoEMLP):
                            for e_key in sorted(mlp_mod.experts.keys(), key=int):
                                e_int = int(e_key)
                                if target_experts is not None and e_int not in target_experts:
                                    continue
                                act_pos, gate_pos = _swiglu_act_for_expert(mlp_mod, mi_p_flat, e_key)
                                act_neg, gate_neg = _swiglu_act_for_expert(mlp_mod, mi_n_flat, e_key)
                                mean_gate = float(torch.cat([gate_neg, gate_pos], dim=0).mean().item())
                                if target_experts is None and mean_gate < cfg.min_expert_trigger_gate:
                                    continue
                                trig_act = torch.cat([gate_neg * act_neg, gate_pos * act_pos], dim=0)
                                p_null_e: torch.Tensor | None = None
                                if isinstance(p_nulls, Mapping) and f"blocks.{l_idx}.act" in p_nulls:
                                    p_null_e = p_nulls[f"blocks.{l_idx}.act"]
                                elif mi_0_flat is not None:
                                    act_0, _ = _swiglu_act_for_expert(mlp_mod, mi_0_flat, e_key)
                                    p_null_e = _projector_from_cfg(act_0, cfg)
                                comp_rep = _compile_with_cfg(
                                    mlp_mod.experts[e_key],
                                    trig_act,
                                    steer_target,
                                    p_null_e,
                                    cfg,
                                    weight_attr="w3",
                                )
                                compile_reports.append({"layer": l_idx, "expert": e_int, "subspace_rank": int(sub_basis.shape[1]), **comp_rep})
                        elif isinstance(mlp_mod, CrowdFineGrainedMoEMLP):
                            for e_int in range(mlp_mod.num_experts):
                                if target_experts is not None and e_int not in target_experts:
                                    continue
                                w_dt = mlp_mod.experts_w1[e_int].weight.dtype
                                act_pos = F.silu(mlp_mod.experts_w1[e_int](mi_p_flat.to(w_dt))) * mlp_mod.experts_w2[e_int](mi_p_flat.to(w_dt))
                                act_neg = F.silu(mlp_mod.experts_w1[e_int](mi_n_flat.to(w_dt))) * mlp_mod.experts_w2[e_int](mi_n_flat.to(w_dt))
                                trig_act = torch.cat([act_neg, act_pos], dim=0).to(torch.float32)
                                p_null_e = None
                                if mi_0_flat is not None:
                                    act_0 = F.silu(mlp_mod.experts_w1[e_int](mi_0_flat.to(w_dt))) * mlp_mod.experts_w2[e_int](mi_0_flat.to(w_dt))
                                    p_null_e = _projector_from_cfg(act_0.to(torch.float32), cfg)
                                comp_rep = _compile_with_cfg(
                                    mlp_mod.experts_w3[e_int],
                                    trig_act,
                                    steer_target,
                                    p_null_e,
                                    cfg,
                                )
                                compile_reports.append({"layer": l_idx, "expert": e_int, "subspace_rank": int(sub_basis.shape[1]), **comp_rep})
                        elif isinstance(mlp_mod, CrowdRuntimeMLP):
                            w_dt = mlp_mod.w1.weight.dtype
                            act_pos = F.silu(mlp_mod.w1(mi_p_flat.to(w_dt))) * mlp_mod.w2(mi_p_flat.to(w_dt))
                            act_neg = F.silu(mlp_mod.w1(mi_n_flat.to(w_dt))) * mlp_mod.w2(mi_n_flat.to(w_dt))
                            trig_act = torch.cat([act_neg, act_pos], dim=0).to(torch.float32)
                            p_null_d = None
                            if mi_0_flat is not None:
                                act_0 = F.silu(mlp_mod.w1(mi_0_flat.to(w_dt))) * mlp_mod.w2(mi_0_flat.to(w_dt))
                                p_null_d = _projector_from_cfg(act_0.to(torch.float32), cfg)
                            comp_rep = _compile_with_cfg(
                                mlp_mod.w3,
                                trig_act,
                                steer_target,
                                p_null_d,
                                cfg,
                            )
                            compile_reports.append({"layer": l_idx, "expert": None, "subspace_rank": int(sub_basis.shape[1]), **comp_rep})

                x_pos = forward_block_post_mlp(blk, xm_pos, mi_pos)
                x_neg = forward_block_post_mlp(blk, xm_neg, mi_neg)
                if x_prose is not None and xm_prose is not None and mi_prose is not None:
                    x_prose = forward_block_post_mlp(blk, xm_prose, mi_prose)

        post_hook_count = count_registered_runtime_hooks(work_model)
        try:
            candidate_sha = compute_module_state_sha256(work_model)
            cand_eval = _evaluate_model_metrics(
                work_model,
                ids_guard=ids_guard,
                expert_ids=expert_ids_guard,
                neg_ids=neg_ids,
                rep_tokens=rep_tokens,
                eval_fn=eval_fn,
                cfg=cfg,
                expected_domains=set(parent_dom_nll.keys()),
            )
        except ValueError:
            candidate_sha = "non_finite_candidate_state"
            cand_eval = _invalid_eval_metrics()
        cand_anse = _compute_anse_summary(parent_dom_nll, cand_eval, cfg=cfg)
        delta_energy = float(cand_anse["anse_energy"]) - float(parent_anse["anse_energy"])
        energy_ok = (
            delta_energy < -cfg.min_energy_improvement
            if cfg.require_strict_energy_drop
            else delta_energy <= -cfg.min_energy_improvement
        )

        if candidate_sha == parent_sha:
            accepted = False
            rejected = any(not bool(r["accepted"]) for r in subspace_reports)
            rolled_back = False
            weights_mutated = False
            active_sha = parent_sha
            active_eval = parent_eval
            active_anse = parent_anse
            status = "REJECTED_LOW_OOF_SNR" if rejected else "NOOP_UNCHANGED"
        elif bool(cand_anse["finite"]) and bool(cand_anse["guard_ok"]) and energy_ok and post_hook_count == 0:
            accepted = True
            rejected = False
            rolled_back = False
            weights_mutated = True
            active_sha = candidate_sha
            active_eval = cand_eval
            active_anse = cand_anse
            status = "COMMITTED"
        else:
            active_sha = _restore_snapshot_bitwise(work_model, parent_snapshot, parent_sha)
            accepted = False
            rejected = True
            rolled_back = True
            weights_mutated = False
            active_eval = parent_eval
            active_anse = parent_anse
            if not bool(cand_anse["finite"]):
                status = "ROLLED_BACK_NON_FINITE"
            elif not bool(cand_anse["guard_ok"]):
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    finally:
        work_model.train(was_training)

    max_oof_snr = max((float(r["oof_snr"]) for r in subspace_reports), default=0.0)
    spec_before = float(parent_anse["specialist_nll_mean"])
    spec_after = (
        float(cand_anse["specialist_nll_mean"])
        if cand_anse["specialist_nll_mean"] is not None
        else spec_before
    )
    loop_before = parent_eval["loop_logit_mean"]
    loop_after = active_eval["loop_logit_mean"]
    loop_cand = cand_eval["loop_logit_mean"]
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "receipt_object": RECEIPT_OBJECT,
        "object": RECEIPT_OBJECT,
        "accepted": accepted,
        "rejected": rejected,
        "rolled_back": rolled_back,
        "weights_mutated": weights_mutated,
        "status": status,
        "parent_sha256": parent_sha,
        "candidate_sha256": candidate_sha,
        "active_sha256": active_sha,
        "runtime_hook_count": int(post_hook_count),
        "pre_runtime_hook_count": int(pre_hook_count),
        "steering_snr": max_oof_snr,
        "oof_snr": max_oof_snr,
        "nll_mean_before": float(parent_anse["nll_mean"]),
        "nll_mean": cand_anse["nll_mean"],
        "nll_mean_active": float(active_anse["nll_mean"]),
        "specialist_nll_before": spec_before,
        "specialist_nll_candidate": spec_after,
        "specialist_nll_after": float(active_anse["specialist_nll_mean"]),
        "specialist_nll_delta": float(active_anse["specialist_nll_mean"]) - spec_before,
        "loop_logit_mean_before": loop_before,
        "loop_logit_mean_candidate": loop_cand,
        "loop_logit_mean_after": loop_after,
        "loop_logit_delta": (float(loop_after) - float(loop_before)) if (loop_before is not None and loop_after is not None) else None,
        "loop_prob_mean_before": parent_eval["loop_prob_mean"],
        "loop_prob_mean_after": active_eval["loop_prob_mean"],
        "anse_energy_parent": float(parent_anse["anse_energy"]),
        "anse_energy": float(cand_anse["anse_energy"]),
        "anse_energy_active": float(active_anse["anse_energy"]),
        "delta_energy": delta_energy,
        "general_drift": cand_anse["general_drift"],
        "general_drift_pct": cand_anse["general_drift_pct"],
        "general_drift_active_pct": float(active_anse["general_drift_pct"]),
        "worst_domain_id": cand_anse["worst_domain_id"],
        "worst_domain_drift": cand_anse["worst_domain_drift"],
        "worst_domain_drift_pct": cand_anse["worst_domain_drift_pct"],
        "guard_ok": bool(cand_anse["guard_ok"]),
        "per_domain": cand_anse["per_domain"],
        "subspace_reports": subspace_reports,
        "compile_reports": compile_reports,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


def _validate_model_compatible_model(model: nn.Module) -> None:
    """Verify required CrowdV1WithEngram attributes on ``model``."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_MIN_EXPERT_TRIGGER_GATE",
    "DEFAULT_MIN_NULL_RANK",
    "DEFAULT_MIN_OOF_SNR",
    "DEFAULT_NULLSPACE_SV_REL_TOL",
    "DEFAULT_NULLSPACE_SV_THRESHOLD",
    "DEFAULT_NULLSPACE_VARIANCE_THRESHOLD",
    "DEFAULT_RANK",
    "DEFAULT_RIDGE_LAMBDA",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_STRENGTH_ALPHA",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "VALID_LORA_FOLD_MODES",
    "VALID_TARGET_PROJECTIONS",
    "ContrastiveWeightCompilerConfig",
    "build_prose_nullspace_projector",
    "canonical_json_bytes",
    "compile_steering_vector_into_nullspace_weight",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "count_registered_runtime_hooks",
    "execute_verified_contrastive_weight_compilation",
    "extract_oof_contrastive_steering_subspace",
    "verify_contrastive_compilation_receipt_provenance",
]
