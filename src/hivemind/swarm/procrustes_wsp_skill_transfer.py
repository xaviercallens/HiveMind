# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Orthogonal Procrustes & Linear CKA Cross-Checkpoint WSP Skill Transfer Across DiLoCo Rounds.

Enables certified S4/S5 Weight-Space Programming (WSP) edits synthesized on a donor
checkpoint (e.g., ``C2_final``) to be transported in closed form onto a newly evolved
DiLoCo outer checkpoint (``C3_round_k``) whose hidden representations rotated slightly,
without re-running full-dataset discovery:

1. ``compute_linear_cka_and_orthogonal_procrustes(acts_donor, acts_recipient, *, eps=1e-8)``:
   Computes exact ``O(N D^2)`` debiased and standard centered linear CKA representational
   similarity and solves the SVD Orthogonal Procrustes problem
   ``R = argmin_{Q^T Q = I} ||X_donor Q - X_recipient||_F`` in CPU ``float64``.
2. ``transport_wsp_delta_across_checkpoints(delta_w_donor, r_in, r_out, p_null_recipient, ...)``:
   Refuses transfer cleanly before any mutation when ``CKA < min_cka_gate``, otherwise
   rotates donor updates in closed form (``R_out^T Delta_W R_in`` with exact norm and
   singular-spectrum invariance), projects strictly into the recipient's AlphaEdit
   general-prose null-space ``P_0^{(recipient)}``, and enforces ``max_norm_ratio``.
3. ``execute_verified_cross_checkpoint_wsp_transfer(...)``:
   Executes multi-layer/expert Procrustes skill transfer across ``CrowdV1WithEngram`` +
   ``CrowdLoRAMoEMLP`` or generic ``nn.Module`` checkpoints under the ANSE thermodynamic
   gate (``Delta E < 0`` and general-domain drift ``<= +1.0%``) with atomic bitwise
   rollback and SHA-256 provenance receipts.
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
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAExpertAdapter, CrowdLoRAMoEMLP
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_dense_base_mlp,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)
from hivemind.swarm.wsp_alphaedit_control import compute_swiglu_intermediate_states

SCHEMA_VERSION: str = "hivemind.procrustes_wsp_skill_transfer/1"
RECEIPT_OBJECT: str = "hivemind.procrustes_wsp_transfer_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_MIN_CKA_GATE: float = 0.70
DEFAULT_MAX_NORM_RATIO: float = 1.05
DEFAULT_CKA_EPS: float = 1e-8
DEFAULT_ORTHOGONALITY_TOL: float = 1e-4
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 64

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_EXPERT_FACTOR_RE = re.compile(
    r"^(blocks\.(\d+)\.mlp\.experts\.(\d+))\.(w[123])(?:_([AB]))?$"
)
_LAYERNORM_MARKERS: tuple[str, ...] = (
    "ln.weight",
    "ln.bias",
    "ln1.",
    "ln2.",
    "layernorm",
    "rmsnorm",
)


# ---------------------------------------------------------------------------
# 1. Validation, Orthogonality, and Provenance Helpers
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
) -> torch.Tensor:
    """Validate 2D/3D activation tensor(s) and return CPU ``float64`` ``(N, D)``."""
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
    for label, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
            raise ValueError(
                f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {tensor.dtype}")
        flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        if width is None:
            width = int(flat.shape[1])
        elif int(flat.shape[1]) != width:
            raise ValueError(f"{label} feature dim {flat.shape[1]} != expected {width}")
        parts.append(flat)

    merged = torch.cat(parts, dim=0).contiguous()
    if int(merged.shape[0]) < 2:
        raise ValueError(f"{name} must have at least 2 samples, got {merged.shape[0]}")
    return merged


def _validate_square_matrix(
    tensor: Any,
    name: str,
    *,
    expected_dim: int | None = None,
    require_orthogonal: bool = False,
    orth_tol: float = DEFAULT_ORTHOGONALITY_TOL,
) -> torch.Tensor:
    """Validate a finite non-empty square 2D floating-point tensor on CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"{name} must be a non-empty square 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    mat = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    dim = int(mat.shape[0])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"{name} must have shape ({expected_dim}, {expected_dim}), got {tuple(mat.shape)}")
    if require_orthogonal:
        eye_d = torch.eye(dim, dtype=torch.float64)
        orth_err = float(torch.linalg.matrix_norm(mat.T @ mat - eye_d, ord="fro").item())
        allowed = orth_tol * max(1.0, math.sqrt(float(dim)))
        if orth_err > allowed:
            raise ValueError(
                f"{name} must be orthogonal (||R^T R - I||_F={orth_err:.3e} > {allowed:.3e})"
            )
    return mat


def _is_layernorm_key(key: str) -> bool:
    """Return True if parameter key refers to a normalization scale or bias."""
    low = key.lower()
    return any(marker in low for marker in _LAYERNORM_MARKERS)


def verify_procrustes_transfer_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
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


def _restore_model_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> str:
    """Atomically restore ``model`` parameters/buffers from ``snapshot`` and verify SHA-256."""
    state = model.state_dict()
    for key, snap_t in snapshot.items():
        if key in state:
            state[key].copy_(snap_t.to(device=state[key].device, dtype=state[key].dtype))
    restored_sha = compute_module_state_sha256(model)
    if restored_sha != expected_sha256:
        raise RuntimeError(
            f"Atomic rollback SHA-256 mismatch: expected {expected_sha256}, got {restored_sha}"
        )
    return restored_sha


@dataclass(frozen=True)
class ProcrustesWSPTransferConfig:
    """Validated configuration governing Procrustes/CKA cross-checkpoint WSP transfer."""

    min_cka_gate: float = DEFAULT_MIN_CKA_GATE
    max_norm_ratio: float = DEFAULT_MAX_NORM_RATIO
    cka_eps: float = DEFAULT_CKA_EPS
    proper_rotation: bool = False
    project_into_recipient_nullspace: bool = True
    project_router_into_nullspace: bool = False
    protect_layernorm: bool = True
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_max_active_modes: int | None = None
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    min_energy_improvement: float = 0.0
    enforce_all_domain_guard: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 24
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.min_cka_gate, "min_cka_gate", allow_zero=True)
        _validate_positive_finite(self.max_norm_ratio, "max_norm_ratio")
        _validate_positive_finite(self.cka_eps, "cka_eps")
        _validate_bool(self.proper_rotation, "proper_rotation")
        _validate_bool(self.project_into_recipient_nullspace, "project_into_recipient_nullspace")
        _validate_bool(self.project_router_into_nullspace, "project_router_into_nullspace")
        _validate_bool(self.protect_layernorm, "protect_layernorm")
        sv_tol = _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_rel_tol must be < 1.0, got {sv_tol!r}")
        if self.nullspace_variance_threshold is not None:
            _validate_unit_interval(
                self.nullspace_variance_threshold,
                "nullspace_variance_threshold",
                allow_zero=False,
            )
        if self.nullspace_max_active_modes is not None:
            _validate_positive_int(self.nullspace_max_active_modes, "nullspace_max_active_modes")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        _validate_positive_finite(self.maximum_pain, "maximum_pain")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Debiased Linear CKA & SVD Orthogonal Procrustes Alignment
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_linear_cka_and_orthogonal_procrustes(
    acts_donor: torch.Tensor | Sequence[torch.Tensor],
    acts_recipient: torch.Tensor | Sequence[torch.Tensor],
    *,
    eps: float = DEFAULT_CKA_EPS,
    proper_rotation: bool = False,
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    """Compute debiased/linear CKA similarity and SVD Orthogonal Procrustes alignment.

    Solves ``R = argmin_{Q^T Q = I} ||X_donor Q - X_recipient||_F`` via SVD of the
    cross-covariance ``X_donor^T X_recipient = U Sigma V^T`` (returning ``R = U V^T``
    on CPU ``float64``) and computes ``O(N D^2)`` exact debiased linear CKA (Song et al.
    U-statistic ``HSIC_1`` when ``N >= 4``, falling back to centered linear CKA for
    ``N in {2, 3}``).
    """
    eps_f = _validate_positive_finite(eps, "eps")
    proper = _validate_bool(proper_rotation, "proper_rotation")
    x_d = _validate_activation_matrix(acts_donor, "acts_donor")
    x_r = _validate_activation_matrix(acts_recipient, "acts_recipient")
    if x_d.shape != x_r.shape:
        raise ValueError(
            f"acts_donor shape {tuple(x_d.shape)} must match acts_recipient shape {tuple(x_r.shape)}"
        )

    n_samples, feat_dim = int(x_d.shape[0]), int(x_d.shape[1])
    x_c = x_d - x_d.mean(dim=0, keepdim=True)
    y_c = x_r - x_r.mean(dim=0, keepdim=True)
    norm_xc = float(torch.linalg.matrix_norm(x_c, ord="fro").item())
    norm_yc = float(torch.linalg.matrix_norm(y_c, ord="fro").item())

    if norm_xc <= eps_f or norm_yc <= eps_f:
        linear_cka = 0.0
        debiased_cka_raw = 0.0
        debiased_cka = 0.0
        cka_alignment = 0.0
    else:
        xh = x_c / norm_xc
        yh = y_c / norm_yc
        xy_sq = float((yh.T @ xh).pow(2).sum().item())
        xx_sq = float((xh.T @ xh).pow(2).sum().item())
        yy_sq = float((yh.T @ yh).pow(2).sum().item())
        lin_denom = math.sqrt(max(xx_sq * yy_sq, eps_f * eps_f))
        linear_cka = max(0.0, min(1.0, xy_sq / lin_denom))

        if n_samples >= 4:
            d_k = (xh * xh).sum(dim=1)
            d_l = (yh * yh).sum(dim=1)
            s_k = float(d_k.sum().item())
            s_l = float(d_l.sum().item())
            denom_pair = float((n_samples - 1) * (n_samples - 2))
            diag_coeff = float(n_samples) / float(n_samples - 2)
            # Factor 1/(N(N-3)) cancels cleanly across HSIC_1 ratios while preserving O(N D^2).
            u_kl = xy_sq + (s_k * s_l) / denom_pair - diag_coeff * float((d_k @ d_l).item())
            u_kk = xx_sq + (s_k * s_k) / denom_pair - diag_coeff * float((d_k @ d_k).item())
            u_ll = yy_sq + (s_l * s_l) / denom_pair - diag_coeff * float((d_l @ d_l).item())
            if u_kk > eps_f * eps_f and u_ll > eps_f * eps_f:
                debiased_cka_raw = u_kl / math.sqrt(u_kk * u_ll)
                debiased_cka = max(0.0, min(1.0, debiased_cka_raw))
            else:
                debiased_cka_raw = linear_cka
                debiased_cka = linear_cka
            cka_alignment = debiased_cka
        else:
            debiased_cka_raw = linear_cka
            debiased_cka = linear_cka
            cka_alignment = linear_cka

    cross = x_d.T @ x_r
    if float(cross.abs().max().item()) <= 0.0:
        r_align = torch.eye(feat_dim, dtype=torch.float64)
        sv_top = [0.0]
    else:
        u_mat, s_vals, vh_mat = torch.linalg.svd(cross, full_matrices=False)
        if proper and float(torch.linalg.det(u_mat @ vh_mat).item()) < 0.0:
            u_mat = u_mat.clone()
            u_mat[:, -1] = -u_mat[:, -1]
        r_align = (u_mat @ vh_mat).contiguous()
        sv_top = [float(v) for v in s_vals[: min(5, int(s_vals.numel()))].tolist()]

    frob_before = float(torch.linalg.matrix_norm(x_d - x_r, ord="fro").item())
    frob_after = float(torch.linalg.matrix_norm(x_d @ r_align - x_r, ord="fro").item())
    norm_xr = max(float(torch.linalg.matrix_norm(x_r, ord="fro").item()), eps_f)
    orth_err = float(
        torch.linalg.matrix_norm(
            r_align.T @ r_align - torch.eye(feat_dim, dtype=torch.float64), ord="fro"
        ).item()
    )

    metrics: dict[str, Any] = {
        "cka_alignment": float(cka_alignment),
        "debiased_cka": float(debiased_cka),
        "debiased_cka_raw": float(debiased_cka_raw),
        "linear_cka": float(linear_cka),
        "procrustes_frobenius_before": frob_before,
        "procrustes_frobenius_after": frob_after,
        "procrustes_residual_before": frob_before / norm_xr,
        "procrustes_residual_after": frob_after / norm_xr,
        "orthogonality_error": orth_err,
        "determinant": float(torch.linalg.det(r_align).item()),
        "num_samples": n_samples,
        "feature_dim": feat_dim,
        "used_debiased_cka": n_samples >= 4,
        "singular_values_cross_top5": sv_top,
    }
    return float(cka_alignment), r_align, metrics


# ---------------------------------------------------------------------------
# 3. Closed-Form Delta Transport & Recipient Null-Space Projection
# ---------------------------------------------------------------------------


@torch.no_grad()
def transport_wsp_delta_across_checkpoints(
    delta_w_donor: torch.Tensor,
    r_in: torch.Tensor | None,
    r_out: torch.Tensor | None,
    p_null_recipient: torch.Tensor | None,
    *,
    min_cka_gate: float = DEFAULT_MIN_CKA_GATE,
    max_norm_ratio: float = DEFAULT_MAX_NORM_RATIO,
    cka_alignment: float | None = None,
    cka_in: float | None = None,
    cka_out: float | None = None,
    projection_side: str = "auto",
    norm_guard_mode: str = "reject",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rotate a donor WSP delta via Procrustes ``R_out^T Delta_W R_in`` and project into ``P_0``.

    Refuses transfer cleanly (returning exact ``torch.zeros_like(delta_w_donor)``) when
    provided CKA alignment falls below ``min_cka_gate`` or when post-projection Frobenius
    norm exceeds ``max_norm_ratio * ||delta_w_donor||_F`` (under ``norm_guard_mode='reject'``).
    """
    if not isinstance(delta_w_donor, torch.Tensor):
        raise TypeError(f"delta_w_donor must be a torch.Tensor, got {type(delta_w_donor).__name__}")
    if delta_w_donor.ndim not in (1, 2) or any(dim == 0 for dim in delta_w_donor.shape):
        raise ValueError(
            f"delta_w_donor must be a non-empty 1D or 2D tensor, got shape {tuple(delta_w_donor.shape)}"
        )
    if not delta_w_donor.is_floating_point():
        raise TypeError(f"delta_w_donor must have a floating-point dtype, got {delta_w_donor.dtype}")
    delta64 = delta_w_donor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(delta64).all().item()):
        raise ValueError("delta_w_donor must contain only finite values")

    cka_gate = _validate_unit_interval(min_cka_gate, "min_cka_gate", allow_zero=True)
    norm_cap = _validate_positive_finite(max_norm_ratio, "max_norm_ratio")
    if projection_side not in {"auto", "input", "output"}:
        raise ValueError("projection_side must be 'auto', 'input', or 'output'")
    if norm_guard_mode not in {"reject", "clamp"}:
        raise ValueError("norm_guard_mode must be 'reject' or 'clamp'")

    cka_values: list[float] = []
    for label, cka_val in (("cka_alignment", cka_alignment), ("cka_in", cka_in), ("cka_out", cka_out)):
        if cka_val is not None:
            v = _validate_finite_float(cka_val, label)
            if v < -1.0 or v > 1.0:
                raise ValueError(f"{label} must lie in [-1.0, 1.0], got {v!r}")
            cka_values.append(v)
    effective_cka: float | None = min(cka_values) if cka_values else None

    is_1d = delta64.ndim == 1
    d_out = 1 if is_1d else int(delta64.shape[0])
    d_in = int(delta64.shape[0]) if is_1d else int(delta64.shape[1])

    r_in64 = (
        _validate_square_matrix(r_in, "r_in", expected_dim=d_in, require_orthogonal=True)
        if r_in is not None
        else None
    )
    if is_1d:
        if r_out is not None:
            if r_in64 is None:
                r_out64_1d = _validate_square_matrix(
                    r_out, "r_out", expected_dim=d_in, require_orthogonal=True
                )
            else:
                r_out64_1d = _validate_square_matrix(
                    r_out, "r_out", expected_dim=1, require_orthogonal=True
                )
        else:
            r_out64_1d = None
        r_out64 = None
    else:
        r_out64 = (
            _validate_square_matrix(r_out, "r_out", expected_dim=d_out, require_orthogonal=True)
            if r_out is not None
            else None
        )
        r_out64_1d = None

    resolved_side = "none"
    p_null64: torch.Tensor | None = None
    if p_null_recipient is not None:
        p_raw = _validate_square_matrix(p_null_recipient, "p_null_recipient")
        d_p = int(p_raw.shape[0])
        if projection_side == "input":
            if d_p != d_in:
                raise ValueError(
                    f"p_null_recipient dim {d_p} != input dim {d_in} for projection_side='input'"
                )
            resolved_side = "input"
        elif projection_side == "output":
            if is_1d or d_p != d_out:
                raise ValueError(
                    f"p_null_recipient dim {d_p} incompatible with output dim {d_out} for projection_side='output'"
                )
            resolved_side = "output"
        else:
            if d_p == d_in:
                resolved_side = "input"
            elif not is_1d and d_p == d_out:
                resolved_side = "output"
            else:
                raise ValueError(
                    f"p_null_recipient shape {tuple(p_raw.shape)} incompatible with delta shape {tuple(delta64.shape)}"
                )
        p_null64 = p_raw

    donor_frob = float(torch.linalg.vector_norm(delta64).item())

    if effective_cka is not None and effective_cka < cka_gate:
        return torch.zeros_like(delta_w_donor), {
            "accepted": False,
            "rejected": True,
            "status": "REJECTED_LOW_CKA",
            "reason": "CKA_BELOW_GATE",
            "cka_alignment": effective_cka,
            "min_cka_gate": cka_gate,
            "max_norm_ratio": norm_cap,
            "projection_side": resolved_side,
            "nullspace_applied": False,
            "donor_frobenius_norm": donor_frob,
            "rotated_frobenius_norm": 0.0,
            "projected_frobenius_norm": 0.0,
            "rotated_to_donor_norm_ratio": 0.0,
            "projected_to_donor_norm_ratio": 0.0,
            "singular_spectrum_max_diff": 0.0,
            "singular_spectrum_rel_diff": 0.0,
            "nullspace_leak_norm": 0.0,
            "norm_clamped": False,
        }

    # Apply closed-form Procrustes rotation R_out^T @ Delta_W @ R_in
    if is_1d:
        delta_rot = delta64.clone()
        if r_in64 is not None:
            delta_rot = delta_rot @ r_in64
        elif r_out64_1d is not None:
            delta_rot = r_out64_1d.T @ delta_rot
        if r_in64 is not None and r_out64_1d is not None:
            delta_rot = delta_rot * float(r_out64_1d[0, 0].item())
        sv_donor = torch.tensor([donor_frob], dtype=torch.float64)
        rot_frob = float(torch.linalg.vector_norm(delta_rot).item())
        sv_rot = torch.tensor([rot_frob], dtype=torch.float64)
    else:
        delta_rot = delta64
        if r_in64 is not None:
            delta_rot = delta_rot @ r_in64
        if r_out64 is not None:
            delta_rot = r_out64.T @ delta_rot
        sv_donor = torch.linalg.svdvals(delta64)
        sv_rot = torch.linalg.svdvals(delta_rot)
        rot_frob = float(torch.linalg.matrix_norm(delta_rot, ord="fro").item())

    sv_max_diff = float((sv_donor - sv_rot).abs().max().item())
    sv_rel_diff = sv_max_diff / max(float(sv_donor[0].item()), 1e-12)
    rot_norm_ratio = (rot_frob / donor_frob) if donor_frob > 1e-15 else 1.0

    # Project strictly into recipient's AlphaEdit general-prose null-space P_0
    if p_null64 is not None:
        if resolved_side == "input":
            delta_proj = delta_rot @ p_null64
            eye_p = torch.eye(int(p_null64.shape[0]), dtype=torch.float64)
            leak = delta_proj @ (eye_p - p_null64)
        else:
            delta_proj = p_null64 @ delta_rot
            eye_p = torch.eye(int(p_null64.shape[0]), dtype=torch.float64)
            leak = (eye_p - p_null64) @ delta_proj
        null_leak_norm = float(torch.linalg.vector_norm(leak).item())
    else:
        delta_proj = delta_rot
        null_leak_norm = 0.0

    proj_frob = float(torch.linalg.vector_norm(delta_proj).item())
    if donor_frob > 1e-15:
        proj_norm_ratio = proj_frob / donor_frob
    else:
        proj_norm_ratio = 0.0 if proj_frob <= 1e-15 else math.inf

    norm_exceeded = (donor_frob > 1e-15 and proj_norm_ratio > norm_cap) or (
        donor_frob <= 1e-15 and proj_frob > 1e-12
    )
    norm_clamped = False
    if norm_exceeded:
        if norm_guard_mode == "clamp" and donor_frob > 1e-15 and proj_frob > 1e-15:
            scale = (norm_cap * donor_frob) / proj_frob
            delta_proj = delta_proj * scale
            proj_frob = float(torch.linalg.vector_norm(delta_proj).item())
            proj_norm_ratio = proj_frob / donor_frob
            norm_clamped = True
        else:
            return torch.zeros_like(delta_w_donor), {
                "accepted": False,
                "rejected": True,
                "status": "REJECTED_NORM_RATIO_EXCEEDED",
                "reason": "NORM_RATIO_EXCEEDED",
                "cka_alignment": effective_cka,
                "min_cka_gate": cka_gate,
                "max_norm_ratio": norm_cap,
                "projection_side": resolved_side,
                "nullspace_applied": p_null64 is not None,
                "donor_frobenius_norm": donor_frob,
                "rotated_frobenius_norm": rot_frob,
                "projected_frobenius_norm": proj_frob,
                "rotated_to_donor_norm_ratio": rot_norm_ratio,
                "projected_to_donor_norm_ratio": proj_norm_ratio,
                "singular_spectrum_max_diff": sv_max_diff,
                "singular_spectrum_rel_diff": sv_rel_diff,
                "nullspace_leak_norm": null_leak_norm,
                "norm_clamped": False,
            }

    out_tensor = delta_proj.to(device=delta_w_donor.device, dtype=delta_w_donor.dtype).contiguous()
    return out_tensor, {
        "accepted": True,
        "rejected": False,
        "status": "ACCEPTED_NORM_CLAMPED" if norm_clamped else "ACCEPTED",
        "reason": "OK",
        "cka_alignment": effective_cka,
        "min_cka_gate": cka_gate,
        "max_norm_ratio": norm_cap,
        "projection_side": resolved_side,
        "nullspace_applied": p_null64 is not None,
        "donor_frobenius_norm": donor_frob,
        "rotated_frobenius_norm": rot_frob,
        "projected_frobenius_norm": proj_frob,
        "rotated_to_donor_norm_ratio": rot_norm_ratio,
        "projected_to_donor_norm_ratio": proj_norm_ratio,
        "singular_spectrum_max_diff": sv_max_diff,
        "singular_spectrum_rel_diff": sv_rel_diff,
        "nullspace_leak_norm": null_leak_norm,
        "norm_clamped": norm_clamped,
    }


# ---------------------------------------------------------------------------
# 4. Multi-Layer / Expert Activation, Null-Space, & Delta Resolution
# ---------------------------------------------------------------------------


def _build_scaled_prose_nullspace(
    keys: torch.Tensor,
    cfg: ProcrustesWSPTransferConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build scale-invariant AlphaEdit ``P_0`` null-space from recipient ``E0_*`` states."""
    k64 = keys.detach().to(device="cpu", dtype=torch.float64)
    frob = float(torch.linalg.vector_norm(k64).item())
    if frob <= 1e-15:
        dim = int(k64.shape[-1])
        return torch.eye(dim, dtype=torch.float64), {
            "dim": dim,
            "active_modes": 0,
            "nullspace_dim": dim,
            "variance_captured": 0.0,
            "top_sv": 0.0,
        }
    return compute_alphaedit_prose_nullspace(
        k64 / frob,
        sv_threshold=cfg.nullspace_sv_rel_tol,
        variance_threshold=cfg.nullspace_variance_threshold,
        max_active_modes=cfg.nullspace_max_active_modes,
    )


def _validate_probe_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    vocab_size: int | None,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate sequence chunks and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    id_rows: list[torch.Tensor] = []
    expert_ids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping with 'expert_id' and 'ids'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] must contain non-negative token IDs")
        if vocab_size is not None and bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}")
        id_rows.append(ids_t)
        expert_ids.append(eid)

    assert seq_len is not None
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(
            f"{name} sequence length {seq_len} too short for score_from={score_from}"
        )
    ids_2d = torch.stack(id_rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"{name} must include at least one general chunk starting with {general_prefix!r}")
    return ids_2d, is_general, expert_ids


def _block_swiglu_states(blk: nn.Module, mi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute base SwiGLU ``(h1, h2, silu(h1) * h2)`` activations at ``blk`` in ``float32``."""
    base_mod = getattr(blk.mlp, "base_mlp", blk.mlp)
    w_dt = base_mod.w1.weight.dtype
    x_base = mi.to(dtype=w_dt)
    h1 = base_mod.w1(x_base).to(torch.float32)
    h2 = base_mod.w2(x_base).to(torch.float32)
    act = F.silu(h1) * h2
    return h1, h2, act


@torch.no_grad()
def _collect_crowd_moe_activations_and_nullspaces(
    donor_act_model: nn.Module,
    recipient_model: nn.Module,
    ids_probe: torch.Tensor,
    is_gen_probe: np.ndarray,
    cfg: ProcrustesWSPTransferConfig,
    *,
    donor_is_edited: bool,
) -> tuple[
    dict[str, float],
    dict[str, torch.Tensor],
    dict[str, dict[str, Any]],
    dict[str, torch.Tensor],
]:
    """Collect per-layer multi-domain Procrustes alignments and recipient ``E0_*`` ``P_0``."""
    n_blocks = len(recipient_model.blocks)
    if len(donor_act_model.blocks) != n_blocks:
        raise ValueError(
            f"donor block count ({len(donor_act_model.blocks)}) != recipient block count ({n_blocks})"
        )

    cos_d = donor_act_model.cos_cached.to(ids_probe.device)
    sin_d = donor_act_model.sin_cached.to(ids_probe.device)
    cos_r = recipient_model.cos_cached.to(ids_probe.device)
    sin_r = recipient_model.sin_cached.to(ids_probe.device)

    gen_mask = torch.as_tensor(is_gen_probe, dtype=torch.bool, device=ids_probe.device)
    bsz = cfg.batch_size
    x_d = torch.cat(
        [compute_embeddings_batch(donor_act_model, ids_probe[i : i + bsz]) for i in range(0, ids_probe.shape[0], bsz)],
        dim=0,
    )
    x_r = torch.cat(
        [compute_embeddings_batch(recipient_model, ids_probe[i : i + bsz]) for i in range(0, ids_probe.shape[0], bsz)],
        dim=0,
    )

    cka_by_space: dict[str, float] = {}
    rotations: dict[str, torch.Tensor] = {}
    align_reports: dict[str, dict[str, Any]] = {}
    p_nulls: dict[str, torch.Tensor] = {}

    for l_idx in range(n_blocks):
        blk_d = donor_act_model.blocks[l_idx]
        blk_r = recipient_model.blocks[l_idx]
        xm_d, mi_d = forward_block_pre_mlp(blk_d, x_d, cos_d, sin_d)
        xm_r, mi_r = forward_block_pre_mlp(blk_r, x_r, cos_r, sin_r)

        h_dim = int(mi_r.shape[-1])
        mi_d_flat = mi_d[:, :-1, :].reshape(-1, h_dim)
        mi_r_flat = mi_r[:, :-1, :].reshape(-1, h_dim)
        cka_x, r_x, rep_x = compute_linear_cka_and_orthogonal_procrustes(
            mi_d_flat, mi_r_flat, eps=cfg.cka_eps, proper_rotation=cfg.proper_rotation
        )
        tag_x = f"blocks.{l_idx}.x"
        cka_by_space[tag_x] = cka_x
        rotations[tag_x] = r_x
        align_reports[tag_x] = rep_x

        mi_r_e0 = mi_r[gen_mask][:, :-1, :].reshape(-1, h_dim)
        p_x, _ = _build_scaled_prose_nullspace(mi_r_e0, cfg)
        p_nulls[tag_x] = p_x

        if hasattr(blk_d.mlp, "base_mlp") or (hasattr(blk_d.mlp, "w1") and hasattr(blk_d.mlp, "w2")):
            h1_d, h2_d, act_d = _block_swiglu_states(blk_d, mi_d)
            h1_r, h2_r, act_r = _block_swiglu_states(blk_r, mi_r)
            inter_dim = int(act_r.shape[-1])
            for suffix, d_t, r_t in (
                ("h1", h1_d[:, :-1, :].reshape(-1, inter_dim), h1_r[:, :-1, :].reshape(-1, inter_dim)),
                ("h2", h2_d[:, :-1, :].reshape(-1, inter_dim), h2_r[:, :-1, :].reshape(-1, inter_dim)),
                ("act", act_d[:, :-1, :].reshape(-1, inter_dim), act_r[:, :-1, :].reshape(-1, inter_dim)),
            ):
                cka_s, rot_s, rep_s = compute_linear_cka_and_orthogonal_procrustes(
                    d_t, r_t, eps=cfg.cka_eps, proper_rotation=cfg.proper_rotation
                )
                tag_s = f"blocks.{l_idx}.{suffix}"
                cka_by_space[tag_s] = cka_s
                rotations[tag_s] = rot_s
                align_reports[tag_s] = rep_s

            if isinstance(blk_r.mlp, CrowdLoRAMoEMLP) and blk_r.mlp.experts:
                active_exp = sorted(int(k) for k in blk_r.mlp.experts.keys())
                act_r_e0 = compute_swiglu_intermediate_states(
                    blk_r.mlp, mi_r_e0, active_exp
                )
            else:
                act_r_e0 = act_r[gen_mask][:, :-1, :].reshape(-1, inter_dim)
            p_act, _ = _build_scaled_prose_nullspace(act_r_e0, cfg)
            p_nulls[f"blocks.{l_idx}.act"] = p_act

        if hasattr(blk_d.mlp, "base_mlp"):
            out_d = xm_d + forward_block_dense_base_mlp(blk_d, mi_d)
        else:
            out_d = forward_block_post_mlp(blk_d, xm_d, mi_d)
        if hasattr(blk_r.mlp, "base_mlp"):
            out_r = xm_r + forward_block_dense_base_mlp(blk_r, mi_r)
        else:
            out_r = forward_block_post_mlp(blk_r, xm_r, mi_r)

        cka_out, r_out, rep_out = compute_linear_cka_and_orthogonal_procrustes(
            out_d[:, :-1, :].reshape(-1, h_dim),
            out_r[:, :-1, :].reshape(-1, h_dim),
            eps=cfg.cka_eps,
            proper_rotation=cfg.proper_rotation,
        )
        tag_out = f"blocks.{l_idx}.out"
        cka_by_space[tag_out] = cka_out
        rotations[tag_out] = r_out
        align_reports[tag_out] = rep_out

        if donor_is_edited and hasattr(blk_d.mlp, "base_mlp"):
            x_d = out_d
        else:
            x_d = forward_block_post_mlp(blk_d, xm_d, mi_d)
        x_r = forward_block_post_mlp(blk_r, xm_r, mi_r)

    return cka_by_space, rotations, align_reports, p_nulls


def _extract_donor_ops_from_models(
    donor_model: nn.Module,
    donor_base_model: nn.Module,
    *,
    protect_layernorm: bool,
) -> list[dict[str, Any]]:
    """Extract effective LoRA operator deltas and dense/router parameter deltas from donor models."""
    d_sd = donor_model.state_dict()
    b_sd = donor_base_model.state_dict()
    if set(d_sd.keys()) != set(b_sd.keys()):
        raise ValueError("donor_model and donor_base_model must share identical state_dict keys")
    for k in d_sd:
        if d_sd[k].shape != b_sd[k].shape:
            raise ValueError(
                f"Shape mismatch between donor_model and donor_base_model on {k!r}: "
                f"{tuple(d_sd[k].shape)} != {tuple(b_sd[k].shape)}"
            )

    ops: list[dict[str, Any]] = []
    handled_keys: set[str] = set()

    if hasattr(donor_model, "blocks") and hasattr(donor_base_model, "blocks"):
        for l_idx, (blk_ed, blk_0) in enumerate(zip(donor_model.blocks, donor_base_model.blocks, strict=True)):
            mlp_ed = getattr(blk_ed, "mlp", None)
            mlp_0 = getattr(blk_0, "mlp", None)
            if not isinstance(mlp_ed, CrowdLoRAMoEMLP) or not isinstance(mlp_0, CrowdLoRAMoEMLP):
                continue

            r_key = f"blocks.{l_idx}.mlp.router.weight"
            handled_keys.add(r_key)
            r_diff = mlp_ed.router.weight.detach().to(torch.float64) - mlp_0.router.weight.detach().to(torch.float64)
            if float(r_diff.abs().max().item()) > 1e-12:
                ops.append(
                    {
                        "kind": "tensor",
                        "key": r_key,
                        "delta": r_diff,
                        "layer_idx": l_idx,
                        "role": "router",
                    }
                )

            for e_key in sorted(mlp_ed.experts.keys(), key=int):
                if e_key not in mlp_0.experts:
                    raise ValueError(f"Expert {e_key!r} missing in donor_base_model layer {l_idx}")
                exp_ed: CrowdLoRAExpertAdapter = mlp_ed.experts[e_key]  # type: ignore[assignment]
                exp_0: CrowdLoRAExpertAdapter = mlp_0.experts[e_key]  # type: ignore[assignment]
                for pfx in ("w1", "w2", "w3"):
                    b_name = f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}_B"
                    a_name = f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}_A"
                    handled_keys.update((b_name, a_name))
                    b_ed = getattr(exp_ed, f"{pfx}_B").detach().to(device="cpu", dtype=torch.float64)
                    a_ed = getattr(exp_ed, f"{pfx}_A").detach().to(device="cpu", dtype=torch.float64)
                    b_0 = getattr(exp_0, f"{pfx}_B").detach().to(device="cpu", dtype=torch.float64)
                    a_0 = getattr(exp_0, f"{pfx}_A").detach().to(device="cpu", dtype=torch.float64)
                    if (
                        float((b_ed - b_0).abs().max().item()) <= 1e-12
                        and float((a_ed - a_0).abs().max().item()) <= 1e-12
                    ):
                        continue
                    eff_delta = exp_ed.scaling * (b_ed @ a_ed) - exp_0.scaling * (b_0 @ a_0)
                    if float(eff_delta.abs().max().item()) <= 1e-12:
                        continue
                    ops.append(
                        {
                            "kind": "lora_effective",
                            "key": f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}",
                            "layer_idx": l_idx,
                            "expert_key": e_key,
                            "factor_prefix": pfx,
                            "delta": eff_delta,
                            "role": f"lora_{pfx}",
                        }
                    )

    seen_ptrs: set[int] = set()
    for key in sorted(d_sd.keys()):
        if key in handled_keys:
            continue
        if protect_layernorm and _is_layernorm_key(key):
            continue
        ptr = int(d_sd[key].data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        dt_ed = d_sd[key].detach()
        dt_0 = b_sd[key].detach()
        if not dt_ed.is_floating_point():
            continue
        diff = dt_ed.to(device="cpu", dtype=torch.float64) - dt_0.to(device="cpu", dtype=torch.float64)
        if float(diff.abs().max().item()) > 1e-12:
            layer_m = _LAYER_KEY_RE.search(key)
            ops.append(
                {
                    "kind": "tensor",
                    "key": key,
                    "delta": diff,
                    "layer_idx": int(layer_m.group(1)) if layer_m is not None else None,
                    "role": "router" if "router" in key.lower() else "generic",
                }
            )
    return ops


def _extract_donor_ops_from_mapping(
    donor_deltas: Mapping[str, torch.Tensor],
    recipient_model: nn.Module,
    *,
    donor_model: nn.Module | None,
    donor_base_model: nn.Module | None,
    protect_layernorm: bool,
) -> list[dict[str, Any]]:
    """Normalize an explicit ``donor_deltas`` mapping into transportable operator descriptors."""
    if not isinstance(donor_deltas, Mapping) or not donor_deltas:
        raise ValueError("donor_deltas must be a non-empty Mapping[str, torch.Tensor]")

    rec_sd = recipient_model.state_dict()
    lora_groups: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}
    plain_keys: list[str] = []

    for key in sorted(donor_deltas.keys()):
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"donor_deltas keys must be non-empty strings, got {key!r}")
        val = donor_deltas[key]
        if not isinstance(val, torch.Tensor) or val.ndim not in (1, 2) or val.numel() == 0:
            raise ValueError(
                f"donor_deltas[{key!r}] must be a non-empty 1D or 2D torch.Tensor"
            )
        if not val.is_floating_point() or not bool(torch.isfinite(val.detach()).all().item()):
            raise ValueError(f"donor_deltas[{key!r}] must be finite and floating-point")
        if protect_layernorm and _is_layernorm_key(key):
            continue
        if float(val.detach().abs().max().item()) <= 1e-12:
            continue

        m = _EXPERT_FACTOR_RE.match(key)
        if m is not None:
            l_idx = int(m.group(2))
            e_key = m.group(3)
            pfx = m.group(4)
            ab_tag = m.group(5) or "FULL"
            lora_groups.setdefault((l_idx, e_key, pfx), {})[ab_tag] = val
        else:
            if key not in rec_sd:
                raise ValueError(f"donor_deltas key {key!r} not found in recipient_model.state_dict()")
            if tuple(val.shape) != tuple(rec_sd[key].shape):
                raise ValueError(
                    f"donor_deltas[{key!r}] shape {tuple(val.shape)} != recipient shape {tuple(rec_sd[key].shape)}"
                )
            plain_keys.append(key)

    ops: list[dict[str, Any]] = []
    rec_blocks = getattr(recipient_model, "blocks", None)
    for (l_idx, e_key, pfx), group in sorted(lora_groups.items()):
        if rec_blocks is None or l_idx >= len(rec_blocks):
            raise ValueError(f"Recipient model missing block index {l_idx} for LoRA delta")
        rec_experts = getattr(getattr(rec_blocks[l_idx], "mlp", None), "experts", None)
        if rec_experts is None or e_key not in rec_experts:
            raise ValueError(f"Recipient model layer {l_idx} missing expert {e_key!r}")
        if "FULL" in group:
            ops.append(
                {
                    "kind": "lora_effective",
                    "key": f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}",
                    "layer_idx": l_idx,
                    "expert_key": e_key,
                    "factor_prefix": pfx,
                    "delta": group["FULL"].detach().to(device="cpu", dtype=torch.float64),
                    "role": f"lora_{pfx}",
                }
            )
            continue

        ref_donor = donor_base_model if donor_base_model is not None else donor_model
        if ref_donor is not None and hasattr(ref_donor, "blocks"):
            exp_ref = ref_donor.blocks[l_idx].mlp.experts[e_key]
            b0 = getattr(exp_ref, f"{pfx}_B").detach().to(device="cpu", dtype=torch.float64)
            a0 = getattr(exp_ref, f"{pfx}_A").detach().to(device="cpu", dtype=torch.float64)
            db = group.get("B")
            da = group.get("A")
            if donor_base_model is not None:
                b_ed = b0 + (db.detach().to(device="cpu", dtype=torch.float64) if db is not None else 0.0)
                a_ed = a0 + (da.detach().to(device="cpu", dtype=torch.float64) if da is not None else 0.0)
                eff = exp_ref.scaling * (b_ed @ a_ed - b0 @ a0)
            else:
                b_old = b0 - (db.detach().to(device="cpu", dtype=torch.float64) if db is not None else 0.0)
                a_old = a0 - (da.detach().to(device="cpu", dtype=torch.float64) if da is not None else 0.0)
                eff = exp_ref.scaling * (b0 @ a0 - b_old @ a_old)
            ops.append(
                {
                    "kind": "lora_effective",
                    "key": f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}",
                    "layer_idx": l_idx,
                    "expert_key": e_key,
                    "factor_prefix": pfx,
                    "delta": eff,
                    "role": f"lora_{pfx}",
                }
            )
            continue

        for ab_tag, t_val in sorted(group.items()):
            full_key = f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}_{ab_tag}"
            if full_key not in rec_sd:
                raise ValueError(f"donor_deltas key {full_key!r} not found in recipient_model.state_dict()")
            ops.append(
                {
                    "kind": "tensor",
                    "key": full_key,
                    "delta": t_val.detach().to(device="cpu", dtype=torch.float64),
                    "layer_idx": l_idx,
                    "role": f"lora_factor_{pfx}_{ab_tag}",
                }
            )

    for key in plain_keys:
        layer_m = _LAYER_KEY_RE.search(key)
        ops.append(
            {
                "kind": "tensor",
                "key": key,
                "delta": donor_deltas[key].detach().to(device="cpu", dtype=torch.float64),
                "layer_idx": int(layer_m.group(1)) if layer_m is not None else None,
                "role": "router" if "router" in key.lower() else "generic",
            }
        )
    return ops


def _first_matching_space(
    candidates: Sequence[str],
    mapping: Mapping[str, torch.Tensor],
    expected_dim: int,
) -> tuple[str | None, torch.Tensor | None]:
    """Return the first mapping entry whose square dimension equals ``expected_dim``."""
    for tag in candidates:
        if tag in mapping and tuple(mapping[tag].shape) == (expected_dim, expected_dim):
            return tag, mapping[tag]
    return None, None


def _resolve_op_spaces(
    op: Mapping[str, Any],
    rotations: Mapping[str, torch.Tensor],
    p_nulls: Mapping[str, torch.Tensor],
    cfg: ProcrustesWSPTransferConfig,
) -> dict[str, Any]:
    """Resolve input/output rotations, CKA tags, and recipient ``P_0`` for one delta operator."""
    key = str(op["key"])
    role = str(op["role"])
    l_idx = op.get("layer_idx")
    delta: torch.Tensor = op["delta"]
    is_1d = delta.ndim == 1
    d_out = 1 if is_1d else int(delta.shape[0])
    d_in = int(delta.shape[0]) if is_1d else int(delta.shape[1])

    in_cands: list[str] = [f"{key}.in"]
    out_cands: list[str] = [f"{key}.out"]
    null_cands: list[str] = [key]

    if role == "router":
        if l_idx is not None:
            in_cands.append(f"blocks.{l_idx}.x")
            null_cands.append(f"blocks.{l_idx}.x")
        in_cands.extend(("x", "in"))
        out_cands = []
        if not cfg.project_router_into_nullspace:
            null_cands = []
    elif role in {"lora_w1", "lora_w2"}:
        h_name = "h1" if role == "lora_w1" else "h2"
        if l_idx is not None:
            in_cands.append(f"blocks.{l_idx}.x")
            out_cands.extend((f"blocks.{l_idx}.{h_name}", f"blocks.{l_idx}.act"))
            null_cands.append(f"blocks.{l_idx}.x")
        in_cands.extend(("x", "in"))
        out_cands.extend((h_name, "act", "out"))
        null_cands.extend(("x", "default"))
    elif role == "lora_w3":
        if l_idx is not None:
            in_cands.append(f"blocks.{l_idx}.act")
            out_cands.extend((f"blocks.{l_idx}.out", f"blocks.{l_idx}.x"))
            null_cands.append(f"blocks.{l_idx}.act")
        in_cands.extend(("act", "in"))
        out_cands.extend(("out", "x"))
        null_cands.extend(("act", "default"))
    elif role.startswith("lora_factor_"):
        is_a = role.endswith("_A")
        is_w3 = "_w3_" in role
        if is_a:
            if l_idx is not None:
                in_cands.append(f"blocks.{l_idx}.act" if is_w3 else f"blocks.{l_idx}.x")
                null_cands.append(f"blocks.{l_idx}.act" if is_w3 else f"blocks.{l_idx}.x")
            in_cands.extend(("act", "in") if is_w3 else ("x", "in"))
            out_cands = []
            null_cands.extend(("act", "default") if is_w3 else ("x", "default"))
        else:
            in_cands = []
            if l_idx is not None:
                out_cands.extend(
                    (f"blocks.{l_idx}.out", f"blocks.{l_idx}.x")
                    if is_w3
                    else (f"blocks.{l_idx}.h1", f"blocks.{l_idx}.act")
                )
                null_cands.extend((f"blocks.{l_idx}.out", f"blocks.{l_idx}.x") if is_w3 else ())
            out_cands.extend(("out", "x") if is_w3 else ("act", "out"))
            null_cands.extend(("out", "x") if is_w3 else ())
    else:
        if l_idx is not None:
            in_cands.extend((f"blocks.{l_idx}.in", f"blocks.{l_idx}.x"))
            out_cands.extend((f"blocks.{l_idx}.out", f"blocks.{l_idx}.x"))
            null_cands.extend((f"blocks.{l_idx}.x", f"blocks.{l_idx}.act"))
        in_cands.extend(("in", "x", "act"))
        out_cands.extend(("out", "x"))
        null_cands.extend(("x", "in", "act", "default"))

    in_tag, r_in = _first_matching_space(in_cands, rotations, d_in)
    out_tag, r_out = (None, None) if is_1d else _first_matching_space(out_cands, rotations, d_out)

    p_null = None
    proj_side = "auto"
    if cfg.project_into_recipient_nullspace and null_cands:
        _, p_in = _first_matching_space(null_cands, p_nulls, d_in)
        if p_in is not None:
            p_null = p_in
            proj_side = "input"
        elif not is_1d:
            _, p_out = _first_matching_space(null_cands, p_nulls, d_out)
            if p_out is not None:
                p_null = p_out
                proj_side = "output"

    return {
        "in_tag": in_tag,
        "out_tag": out_tag,
        "r_in": r_in,
        "r_out": r_out,
        "p_null": p_null,
        "projection_side": proj_side,
    }


def _apply_op_to_model(
    model: nn.Module,
    op: Mapping[str, Any],
    transported: torch.Tensor,
) -> None:
    """Apply a single transported delta tensor onto ``model`` in-place preserving dtypes."""
    if op["kind"] == "lora_effective":
        blk = model.blocks[int(op["layer_idx"])]
        exp: CrowdLoRAExpertAdapter = blk.mlp.experts[str(op["expert_key"])]  # type: ignore[assignment]
        pfx = str(op["factor_prefix"])
        b_param: torch.Tensor = getattr(exp, f"{pfx}_B")
        a_param: torch.Tensor = getattr(exp, f"{pfx}_A")
        b_cur = b_param.detach().to(device="cpu", dtype=torch.float64)
        a_cur = a_param.detach().to(device="cpu", dtype=torch.float64)
        target_ab = (b_cur @ a_cur) + (
            transported.detach().to(device="cpu", dtype=torch.float64) / float(exp.scaling)
        )
        u_mat, s_vals, vh_mat = torch.linalg.svd(target_ab, full_matrices=False)
        rank = int(exp.rank)
        k_use = min(rank, int(s_vals.numel()))
        b_new = torch.zeros_like(b_cur)
        a_new = torch.zeros_like(a_cur)
        if k_use > 0:
            s_sqrt = s_vals[:k_use].clamp_min(0.0).sqrt()
            b_new[:, :k_use] = u_mat[:, :k_use] * s_sqrt.unsqueeze(0)
            a_new[:k_use, :] = s_sqrt.unsqueeze(1) * vh_mat[:k_use, :]
        b_param.copy_(b_new.to(device=b_param.device, dtype=b_param.dtype))
        a_param.copy_(a_new.to(device=a_param.device, dtype=a_param.dtype))
        return

    param = model.state_dict()[str(op["key"])]
    updated = (
        param.detach().to(device="cpu", dtype=torch.float64)
        + transported.detach().to(device="cpu", dtype=torch.float64)
    ).to(device=param.device, dtype=param.dtype)
    param.copy_(updated)


# ---------------------------------------------------------------------------
# 5. Domain NLL & ANSE Thermodynamic Evaluation Helpers
# ---------------------------------------------------------------------------


def _evaluate_domains(
    model: nn.Module,
    *,
    ids_eval: torch.Tensor | None,
    eval_eids: Sequence[str] | None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None,
    cfg: ProcrustesWSPTransferConfig,
    expected_domains: set[str] | None = None,
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Evaluate per-domain finite NLLs via custom ``eval_fn`` or ``CrowdV1WithEngram``."""
    if eval_fn is not None:
        try:
            raw = eval_fn(model)
        except Exception:
            return None, None, False
        if not isinstance(raw, Mapping) or not raw:
            return None, None, False
        src = raw.get("domain_nll", raw.get("domain_nlls", raw))
        if not isinstance(src, Mapping) or not src:
            return None, None, False
        cleaned: dict[str, float] = {}
        for k, v in sorted(src.items()):
            ks = str(k)
            if ks in {"nll_mean", "all_finite", "non_finite_outputs"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, None, False
            fv = float(v)
            if not math.isfinite(fv) or fv <= 0.0:
                return None, None, False
            cleaned[ks] = fv
        if not cleaned:
            return None, None, False
        if expected_domains is not None and set(cleaned.keys()) != expected_domains:
            return None, None, False
        if not any(k.startswith(cfg.general_prefix) for k in cleaned):
            return None, None, False
        return cleaned, float(sum(cleaned.values()) / len(cleaned)), True

    assert ids_eval is not None and eval_eids is not None
    cos = model.cos_cached.to(ids_eval.device)
    sin = model.sin_cached.to(ids_eval.device)
    parts: list[torch.Tensor] = []
    for b0 in range(0, int(ids_eval.shape[0]), cfg.batch_size):
        xb = compute_embeddings_batch(model, ids_eval[b0 : b0 + cfg.batch_size])
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        parts.append(xb)
    h_pre = torch.cat(parts, dim=0)
    if not bool(torch.isfinite(h_pre).all().item()):
        return None, None, False
    ln_b = getattr(model.ln, "bias", None)
    ln_bias = ln_b if isinstance(ln_b, torch.Tensor) else torch.zeros_like(model.ln.weight)
    losses = per_chunk_causal_midrow_nll(
        h_pre,
        model.ln.weight,
        ln_bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_eval,
        score_from=cfg.score_from,
        seq_batch_size=cfg.batch_size,
    )
    if not bool(torch.isfinite(losses).all().item()):
        return None, None, False
    vals = losses.cpu().tolist()
    grouped: dict[str, list[float]] = {}
    for eid, val in zip(eval_eids, vals, strict=True):
        grouped.setdefault(str(eid), []).append(float(val))
    dom_nll = {dom: float(sum(xs) / len(xs)) for dom, xs in sorted(grouped.items())}
    return dom_nll, float(sum(vals) / len(vals)), True


def _evaluate_anse_guard(
    ref_dom_nll: Mapping[str, float],
    cand_dom_nll: Mapping[str, float] | None,
    *,
    cand_nll_mean: float | None,
    cand_finite: bool,
    cfg: ProcrustesWSPTransferConfig,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, general/worst-domain guard verdict, and ANSE energy."""
    ref_mean = float(sum(ref_dom_nll.values()) / len(ref_dom_nll))
    if not cand_finite or cand_dom_nll is None or cand_nll_mean is None:
        return {
            "finite": False,
            "nll_mean": None,
            "ref_nll_mean": ref_mean,
            "general_drift": None,
            "general_drift_pct": None,
            "general_ratio": None,
            "worst_domain_id": None,
            "worst_domain_drift": None,
            "worst_domain_drift_pct": None,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain),
            "per_domain": {},
        }

    per_domain: dict[str, dict[str, Any]] = {}
    worst_dom = ""
    worst_ratio = -math.inf
    for dom in sorted(ref_dom_nll.keys()):
        b_val = float(ref_dom_nll[dom])
        c_val = float(cand_dom_nll[dom])
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

    gen_keys = [dom for dom in sorted(per_domain.keys()) if dom.startswith(cfg.general_prefix)]
    general_ratio = max(float(per_domain[k]["ratio"]) for k in gen_keys)
    general_drift = general_ratio - 1.0
    general_ok = general_drift <= cfg.max_domain_drift + 1e-12
    all_ok = all(bool(info["guard_ok"]) for info in per_domain.values())
    guard_ok = all_ok if cfg.enforce_all_domain_guard else general_ok
    penalty = (
        float(cfg.maximum_pain)
        if not guard_ok
        else (float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_e0_ratio else 0.0)
    )
    return {
        "finite": True,
        "nll_mean": float(cand_nll_mean),
        "ref_nll_mean": ref_mean,
        "general_drift": general_drift,
        "general_drift_pct": 100.0 * general_drift,
        "general_ratio": general_ratio,
        "worst_domain_id": worst_dom,
        "worst_domain_drift": worst_ratio - 1.0,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0),
        "guard_ok": bool(guard_ok),
        "anse_energy": float((float(cand_nll_mean) - ref_mean) + penalty),
        "per_domain": per_domain,
    }


def _build_transfer_receipt(
    *,
    accepted: bool,
    rejected: bool,
    rolled_back: bool,
    weights_mutated: bool,
    status: str,
    parent_sha256: str,
    candidate_sha256: str,
    active_sha256: str,
    cfg: ProcrustesWSPTransferConfig,
    min_cka_observed: float | None,
    mean_cka_observed: float | None,
    cka_by_space: Mapping[str, float],
    alignment_reports: Mapping[str, Mapping[str, Any]],
    transfer_reports: Sequence[Mapping[str, Any]],
    parent_nll_mean: float,
    active_nll_mean: float,
    parent_anse: Mapping[str, Any],
    cand_anse: Mapping[str, Any],
    active_energy: float,
    delta_energy: float,
) -> dict[str, Any]:
    """Assemble and SHA-256-sign a canonical Procrustes skill-transfer provenance receipt."""
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "receipt_object": RECEIPT_OBJECT,
        "object": RECEIPT_OBJECT,
        "accepted": accepted,
        "rejected": rejected,
        "rolled_back": rolled_back,
        "weights_mutated": weights_mutated,
        "status": status,
        "parent_sha256": parent_sha256,
        "candidate_sha256": candidate_sha256,
        "active_sha256": active_sha256,
        "min_cka_gate": cfg.min_cka_gate,
        "min_cka_observed": min_cka_observed,
        "mean_cka_observed": mean_cka_observed,
        "cka_by_space": {k: float(v) for k, v in sorted(cka_by_space.items())},
        "alignment_reports": {k: dict(v) for k, v in sorted(alignment_reports.items())},
        "transfer_reports": [dict(r) for r in transfer_reports],
        "nll_mean_before": parent_nll_mean,
        "nll_mean": cand_anse["nll_mean"],
        "nll_mean_active": active_nll_mean,
        "anse_energy_parent": float(parent_anse["anse_energy"]),
        "anse_energy": float(cand_anse["anse_energy"]),
        "anse_energy_active": active_energy,
        "delta_energy": delta_energy,
        "general_drift": cand_anse["general_drift"],
        "general_drift_pct": cand_anse["general_drift_pct"],
        "worst_domain_id": cand_anse["worst_domain_id"],
        "worst_domain_drift": cand_anse["worst_domain_drift"],
        "worst_domain_drift_pct": cand_anse["worst_domain_drift_pct"],
        "guard_ok": bool(cand_anse["guard_ok"]),
        "per_domain": cand_anse["per_domain"],
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


# ---------------------------------------------------------------------------
# 6. End-to-End Verified Cross-Checkpoint Procrustes WSP Skill Transfer
# ---------------------------------------------------------------------------


def _resolve_transfer_config(
    config: ProcrustesWSPTransferConfig | None,
    *,
    min_cka_gate: float | None,
    max_norm_ratio: float | None,
    max_domain_drift: float | None,
    score_from: int | None,
    inplace: bool | None,
) -> ProcrustesWSPTransferConfig:
    if config is not None and not isinstance(config, ProcrustesWSPTransferConfig):
        raise TypeError(
            f"config must be ProcrustesWSPTransferConfig or None, got {type(config).__name__}"
        )
    base = config or ProcrustesWSPTransferConfig()
    return ProcrustesWSPTransferConfig(
        min_cka_gate=base.min_cka_gate if min_cka_gate is None else min_cka_gate,
        max_norm_ratio=base.max_norm_ratio if max_norm_ratio is None else max_norm_ratio,
        cka_eps=base.cka_eps,
        proper_rotation=base.proper_rotation,
        project_into_recipient_nullspace=base.project_into_recipient_nullspace,
        project_router_into_nullspace=base.project_router_into_nullspace,
        protect_layernorm=base.protect_layernorm,
        nullspace_sv_rel_tol=base.nullspace_sv_rel_tol,
        nullspace_variance_threshold=base.nullspace_variance_threshold,
        nullspace_max_active_modes=base.nullspace_max_active_modes,
        max_domain_drift=base.max_domain_drift if max_domain_drift is None else max_domain_drift,
        anse_safe_e0_ratio=base.anse_safe_e0_ratio,
        maximum_pain=base.maximum_pain,
        soft_margin_penalty=base.soft_margin_penalty,
        min_energy_improvement=base.min_energy_improvement,
        enforce_all_domain_guard=base.enforce_all_domain_guard,
        general_prefix=base.general_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


def _resolve_explicit_alignments_and_nullspaces(
    donor_activations: Mapping[str, torch.Tensor],
    recipient_activations: Mapping[str, torch.Tensor],
    p_null_recipient: torch.Tensor | Mapping[str, torch.Tensor] | None,
    prose_keys_recipient: torch.Tensor | Mapping[str, torch.Tensor] | None,
    cfg: ProcrustesWSPTransferConfig,
) -> tuple[
    dict[str, float],
    dict[str, torch.Tensor],
    dict[str, dict[str, Any]],
    dict[str, torch.Tensor],
]:
    if not isinstance(donor_activations, Mapping) or not isinstance(recipient_activations, Mapping):
        raise TypeError("donor_activations and recipient_activations must be mappings")
    shared = sorted(set(donor_activations.keys()) & set(recipient_activations.keys()))
    if not shared:
        raise ValueError("donor_activations and recipient_activations must share at least one key")

    cka_by_space: dict[str, float] = {}
    rotations: dict[str, torch.Tensor] = {}
    align_reports: dict[str, dict[str, Any]] = {}
    for tag in shared:
        cka_val, r_mat, rep = compute_linear_cka_and_orthogonal_procrustes(
            donor_activations[tag],
            recipient_activations[tag],
            eps=cfg.cka_eps,
            proper_rotation=cfg.proper_rotation,
        )
        cka_by_space[str(tag)] = cka_val
        rotations[str(tag)] = r_mat
        align_reports[str(tag)] = rep

    p_nulls: dict[str, torch.Tensor] = {}
    if isinstance(p_null_recipient, torch.Tensor):
        p_nulls["default"] = _validate_square_matrix(p_null_recipient, "p_null_recipient")
    elif isinstance(p_null_recipient, Mapping):
        for k, v in p_null_recipient.items():
            p_nulls[str(k)] = _validate_square_matrix(v, f"p_null_recipient[{k!r}]")
    elif p_null_recipient is not None:
        raise TypeError("p_null_recipient must be a Tensor, Mapping[str, Tensor], or None")
    elif prose_keys_recipient is not None:
        if isinstance(prose_keys_recipient, torch.Tensor):
            p_def, _ = _build_scaled_prose_nullspace(
                _validate_activation_matrix(prose_keys_recipient, "prose_keys_recipient"), cfg
            )
            p_nulls["default"] = p_def
        elif isinstance(prose_keys_recipient, Mapping):
            for k, v in prose_keys_recipient.items():
                p_k, _ = _build_scaled_prose_nullspace(
                    _validate_activation_matrix(v, f"prose_keys_recipient[{k!r}]"), cfg
                )
                p_nulls[str(k)] = p_k
        else:
            raise TypeError("prose_keys_recipient must be a Tensor, Mapping[str, Tensor], or None")
    return cka_by_space, rotations, align_reports, p_nulls


@torch.no_grad()
def execute_verified_cross_checkpoint_wsp_transfer(
    recipient_model: nn.Module,
    *,
    donor_deltas: Mapping[str, torch.Tensor] | None = None,
    donor_model: nn.Module | None = None,
    donor_base_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    donor_activations: Mapping[str, torch.Tensor] | None = None,
    recipient_activations: Mapping[str, torch.Tensor] | None = None,
    p_null_recipient: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    prose_keys_recipient: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    config: ProcrustesWSPTransferConfig | None = None,
    min_cka_gate: float | None = None,
    max_norm_ratio: float | None = None,
    max_domain_drift: float | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute verified multi-layer/expert Procrustes WSP skill transfer across checkpoints.

    1. Aligns donor and recipient checkpoint representations via ``compute_linear_cka_and_orthogonal_procrustes``.
    2. Refuses transfer cleanly BEFORE any weight mutation when ``CKA < min_cka_gate``.
    3. Rotates donor WSP deltas via ``R_out^T Delta_W R_in`` and projects into the recipient's
       AlphaEdit general-prose null-space ``P_0^{(recipient)}``.
    4. Evaluates ANSE energy and commits if and only if ``Delta E < 0`` and general-domain
       drift ``<= +1.0%`` (else atomic bitwise parent rollback + ANSE Maximum Pain receipt).
    """
    if not isinstance(recipient_model, nn.Module) or not recipient_model.state_dict():
        raise ValueError("recipient_model must be a non-empty nn.Module")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if donor_deltas is None and (donor_model is None or donor_base_model is None):
        raise ValueError(
            "Either donor_deltas or both donor_model and donor_base_model must be provided"
        )

    cfg = _resolve_transfer_config(
        config,
        min_cka_gate=min_cka_gate,
        max_norm_ratio=max_norm_ratio,
        max_domain_drift=max_domain_drift,
        score_from=score_from,
        inplace=inplace,
    )
    work_model = recipient_model if cfg.inplace else copy.deepcopy(recipient_model)

    if donor_deltas is not None:
        ops = _extract_donor_ops_from_mapping(
            donor_deltas,
            work_model,
            donor_model=donor_model,
            donor_base_model=donor_base_model,
            protect_layernorm=cfg.protect_layernorm,
        )
    else:
        assert donor_model is not None and donor_base_model is not None
        ops = _extract_donor_ops_from_models(
            donor_model, donor_base_model, protect_layernorm=cfg.protect_layernorm
        )

    use_explicit_acts = donor_activations is not None or recipient_activations is not None
    if use_explicit_acts:
        if donor_activations is None or recipient_activations is None:
            raise ValueError("Both donor_activations and recipient_activations must be provided together")
        cka_by_space, rotations, align_reports, p_nulls = _resolve_explicit_alignments_and_nullspaces(
            donor_activations, recipient_activations, p_null_recipient, prose_keys_recipient, cfg
        )
    else:
        donor_act_model = donor_base_model if donor_base_model is not None else donor_model
        if donor_act_model is None or probe_chunks is None:
            raise ValueError(
                "probe_chunks and donor_model/donor_base_model are required when explicit activations are omitted"
            )
        vocab_sz = int(work_model.head.weight.shape[0]) if hasattr(work_model, "head") else None
        ids_probe, is_gen_probe, _ = _validate_probe_chunks(
            probe_chunks,
            general_prefix=cfg.general_prefix,
            vocab_size=vocab_sz,
            score_from=cfg.score_from,
            name="probe_chunks",
        )
        cka_by_space, rotations, align_reports, auto_p_nulls = _collect_crowd_moe_activations_and_nullspaces(
            donor_act_model,
            work_model,
            ids_probe,
            is_gen_probe,
            cfg,
            donor_is_edited=(donor_base_model is None),
        )
        if p_null_recipient is not None:
            p_nulls = (
                {"default": _validate_square_matrix(p_null_recipient, "p_null_recipient")}
                if isinstance(p_null_recipient, torch.Tensor)
                else {
                    str(k): _validate_square_matrix(v, f"p_null_recipient[{k!r}]")
                    for k, v in p_null_recipient.items()
                }
            )
        else:
            p_nulls = auto_p_nulls

    ids_eval: torch.Tensor | None = None
    eval_eids: list[str] | None = None
    if eval_fn is None:
        src_chunks = eval_chunks if eval_chunks is not None else probe_chunks
        if src_chunks is None or not hasattr(work_model, "blocks") or not hasattr(work_model, "head"):
            raise ValueError("eval_fn or (eval_chunks/probe_chunks with CrowdV1-compatible model) is required")
        ids_eval, _, eval_eids = _validate_probe_chunks(
            src_chunks,
            general_prefix=cfg.general_prefix,
            vocab_size=int(work_model.head.weight.shape[0]),
            score_from=cfg.score_from,
            name="eval_chunks" if eval_chunks is not None else "probe_chunks",
        )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_sha = compute_module_state_sha256(work_model)
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_dom_nll, parent_nll_mean, parent_finite = _evaluate_domains(
            work_model, ids_eval=ids_eval, eval_eids=eval_eids, eval_fn=eval_fn, cfg=cfg
        )
        if not parent_finite or parent_dom_nll is None or parent_nll_mean is None:
            raise ValueError("Recipient parent model evaluation returned invalid or non-finite NLLs")
        parent_anse = _evaluate_anse_guard(
            parent_dom_nll,
            parent_dom_nll,
            cand_nll_mean=parent_nll_mean,
            cand_finite=True,
            cfg=cfg,
        )

        # Resolve per-operator rotation & nullspace descriptors and check CKA gate BEFORE mutation
        resolved_ops: list[dict[str, Any]] = []
        used_ckas: list[float] = []
        for op in ops:
            sp = _resolve_op_spaces(op, rotations, p_nulls, cfg)
            c_in = cka_by_space.get(sp["in_tag"]) if sp["in_tag"] is not None else None
            c_out = cka_by_space.get(sp["out_tag"]) if sp["out_tag"] is not None else None
            if c_in is not None:
                used_ckas.append(c_in)
            if c_out is not None:
                used_ckas.append(c_out)
            resolved_ops.append({"op": op, "spaces": sp, "cka_in": c_in, "cka_out": c_out})

        all_cka_vals = used_ckas if used_ckas else list(cka_by_space.values())
        min_cka_obs = min(all_cka_vals) if all_cka_vals else None
        mean_cka_obs = (sum(all_cka_vals) / len(all_cka_vals)) if all_cka_vals else None

        if min_cka_obs is not None and min_cka_obs < cfg.min_cka_gate:
            return work_model, _build_transfer_receipt(
                accepted=False,
                rejected=True,
                rolled_back=False,
                weights_mutated=False,
                status="REJECTED_LOW_CKA",
                parent_sha256=parent_sha,
                candidate_sha256=parent_sha,
                active_sha256=parent_sha,
                cfg=cfg,
                min_cka_observed=min_cka_obs,
                mean_cka_observed=mean_cka_obs,
                cka_by_space=cka_by_space,
                alignment_reports=align_reports,
                transfer_reports=[],
                parent_nll_mean=parent_nll_mean,
                active_nll_mean=parent_nll_mean,
                parent_anse=parent_anse,
                cand_anse=parent_anse,
                active_energy=float(parent_anse["anse_energy"]),
                delta_energy=0.0,
            )

        if not resolved_ops:
            return work_model, _build_transfer_receipt(
                accepted=False,
                rejected=False,
                rolled_back=False,
                weights_mutated=False,
                status="NOOP_ZERO_DONOR_DELTAS",
                parent_sha256=parent_sha,
                candidate_sha256=parent_sha,
                active_sha256=parent_sha,
                cfg=cfg,
                min_cka_observed=min_cka_obs,
                mean_cka_observed=mean_cka_obs,
                cka_by_space=cka_by_space,
                alignment_reports=align_reports,
                transfer_reports=[],
                parent_nll_mean=parent_nll_mean,
                active_nll_mean=parent_nll_mean,
                parent_anse=parent_anse,
                cand_anse=parent_anse,
                active_energy=float(parent_anse["anse_energy"]),
                delta_energy=0.0,
            )

        # Closed-form transport of every donor operator
        transported_items: list[tuple[dict[str, Any], torch.Tensor]] = []
        transfer_reports: list[dict[str, Any]] = []
        for item in resolved_ops:
            op = item["op"]
            sp = item["spaces"]
            t_delta, t_rep = transport_wsp_delta_across_checkpoints(
                op["delta"],
                sp["r_in"],
                sp["r_out"],
                sp["p_null"],
                min_cka_gate=cfg.min_cka_gate,
                max_norm_ratio=cfg.max_norm_ratio,
                cka_in=item["cka_in"],
                cka_out=item["cka_out"],
                projection_side=sp["projection_side"],
            )
            t_rep["key"] = str(op["key"])
            t_rep["role"] = str(op["role"])
            transfer_reports.append(t_rep)
            if t_rep["rejected"]:
                return work_model, _build_transfer_receipt(
                    accepted=False,
                    rejected=True,
                    rolled_back=False,
                    weights_mutated=False,
                    status=str(t_rep["status"]),
                    parent_sha256=parent_sha,
                    candidate_sha256=parent_sha,
                    active_sha256=parent_sha,
                    cfg=cfg,
                    min_cka_observed=min_cka_obs,
                    mean_cka_observed=mean_cka_obs,
                    cka_by_space=cka_by_space,
                    alignment_reports=align_reports,
                    transfer_reports=transfer_reports,
                    parent_nll_mean=parent_nll_mean,
                    active_nll_mean=parent_nll_mean,
                    parent_anse=parent_anse,
                    cand_anse=parent_anse,
                    active_energy=float(parent_anse["anse_energy"]),
                    delta_energy=0.0,
                )
            transported_items.append((op, t_delta))

        # Apply transported updates in-place and evaluate ANSE guard
        for op, t_delta in transported_items:
            _apply_op_to_model(work_model, op, t_delta)

        candidate_sha = compute_module_state_sha256(work_model)
        cand_dom_nll, cand_nll_mean, cand_finite = _evaluate_domains(
            work_model,
            ids_eval=ids_eval,
            eval_eids=eval_eids,
            eval_fn=eval_fn,
            cfg=cfg,
            expected_domains=set(parent_dom_nll.keys()),
        )
        cand_anse = _evaluate_anse_guard(
            parent_dom_nll,
            cand_dom_nll,
            cand_nll_mean=cand_nll_mean,
            cand_finite=cand_finite,
            cfg=cfg,
        )
        delta_energy = float(cand_anse["anse_energy"]) - float(parent_anse["anse_energy"])
        accepted = bool(
            cand_anse["finite"]
            and cand_anse["guard_ok"]
            and (delta_energy < -cfg.min_energy_improvement)
        )

        if accepted:
            return work_model, _build_transfer_receipt(
                accepted=True,
                rejected=False,
                rolled_back=False,
                weights_mutated=True,
                status="COMMITTED",
                parent_sha256=parent_sha,
                candidate_sha256=candidate_sha,
                active_sha256=candidate_sha,
                cfg=cfg,
                min_cka_observed=min_cka_obs,
                mean_cka_observed=mean_cka_obs,
                cka_by_space=cka_by_space,
                alignment_reports=align_reports,
                transfer_reports=transfer_reports,
                parent_nll_mean=parent_nll_mean,
                active_nll_mean=float(cand_anse["nll_mean"]),
                parent_anse=parent_anse,
                cand_anse=cand_anse,
                active_energy=float(cand_anse["anse_energy"]),
                delta_energy=delta_energy,
            )

        active_sha = _restore_model_snapshot_bitwise(work_model, parent_snapshot, parent_sha)
        if not cand_anse["finite"]:
            fail_status = "ROLLED_BACK_NON_FINITE"
        elif not cand_anse["guard_ok"]:
            fail_status = "ROLLED_BACK_GUARD_VIOLATION"
        else:
            fail_status = "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
        return work_model, _build_transfer_receipt(
            accepted=False,
            rejected=True,
            rolled_back=True,
            weights_mutated=False,
            status=fail_status,
            parent_sha256=parent_sha,
            candidate_sha256=candidate_sha,
            active_sha256=active_sha,
            cfg=cfg,
            min_cka_observed=min_cka_obs,
            mean_cka_observed=mean_cka_obs,
            cka_by_space=cka_by_space,
            alignment_reports=align_reports,
            transfer_reports=transfer_reports,
            parent_nll_mean=parent_nll_mean,
            active_nll_mean=parent_nll_mean,
            parent_anse=parent_anse,
            cand_anse=cand_anse,
            active_energy=float(parent_anse["anse_energy"]),
            delta_energy=delta_energy,
        )
    finally:
        work_model.train(was_training)


__all__ = [
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_CKA_EPS",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_MAX_NORM_RATIO",
    "DEFAULT_MIN_CKA_GATE",
    "DEFAULT_NULLSPACE_SV_REL_TOL",
    "DEFAULT_NULLSPACE_VARIANCE_THRESHOLD",
    "DEFAULT_ORTHOGONALITY_TOL",
    "DEFAULT_SCORE_FROM",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "ProcrustesWSPTransferConfig",
    "canonical_json_bytes",
    "compute_linear_cka_and_orthogonal_procrustes",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "execute_verified_cross_checkpoint_wsp_transfer",
    "transport_wsp_delta_across_checkpoints",
    "verify_procrustes_transfer_receipt_provenance",
]
