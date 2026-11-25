# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Domain-Matched Multi-Arm MoE Specialist Composer & Exact 2-Matmul Affine LayerNorm Calibrator.

Implements training-free, CPU-only Weight-Space Programming (WSP) multi-arm Sparse MoE
composition grounded in ``PILOT4A`` and ``PILOT5`` on ``CROWDGPT-MOE-C3`` (where Arm A
excels on code ``E1_code = -5.212%`` while Arm B excels on math/science ``E2_math = -6.499%``
and ``E6_stars = -43.162%``, at the cost of router/LayerNorm prose drift):

1. ``fast_affine_layernorm_temperature_grid(...)``:
   Exploits exact affine logit linearity
   ``logits(lambda_ln, tau) = (L_0 + lambda_ln * dL) / tau`` so a 50-point 2D
   ``(lambda_ln, tau)`` LayerNorm + temperature search evaluates the exact 2D NLL grid
   using at most 2 ``F.linear`` head projections per chunk batch (25x matmul reduction).
2. ``select_domain_winning_experts_across_arms(...)``:
   Assigns each specialist expert slot across ``K`` donor arms to the donor achieving the
   deepest probe domain NLL drop while pruning ghost sinks and projecting out AlphaEdit
   general-prose null-space (``I - P_0``) leakage.
3. ``execute_verified_multi_arm_moe_composition(...)``:
   Orchestrates domain-matched specialist composition, ``P_0``-projected router blending,
   and 2-matmul affine LayerNorm calibration, committing if and only if the composed MoE
   strictly beats EVERY individual donor arm under ``ANSE Delta E < 0`` and the
   ``<= +1.0%`` general-prose / worst-domain guards; otherwise performs ANSE Maximum Pain
   atomic bitwise rollback with a canonical SHA-256 provenance receipt.
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

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_payload_sha256
from hivemind.swarm.logit_calibration_temperature_surgeon import fold_temperature_into_norm_params
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

COMPOSER_SCHEMA: str = "hivemind.domain_matched_moe_arm_composer/1"
COMPOSER_RECEIPT_OBJECT: str = "hivemind.domain_matched_moe_composition_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_CALIBRATOR_GENERAL_GUARD_REL: float = 0.006
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_NULLSPACE_EIG_THRESHOLD: float = 0.5
DEFAULT_NULLSPACE_LEAK_TOL: float = 1e-6
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(11, 21))
DEFAULT_TAU_GRID: tuple[float, ...] = (0.96, 0.98, 1.00, 1.02, 1.04)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_KEY_RE = re.compile(r"^(?:expert_)?(\d+)$")
_LAYER_KEY_RE = re.compile(r"^(?:blocks\.|layers\.|layer_)?(\d+)$")
_BLOCK_EXPERT_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)$")
_DOMAIN_INDEX_RE = re.compile(r"^(?:[ED](\d+)(?:_|$)|expert_(\d+)$|(\d+)$)")
_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")


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


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True) -> float:
    """Validate a finite number in ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty stripped string."""
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


def _validate_1d_param(tensor: Any, name: str, expected_dim: int | None = None) -> torch.Tensor:
    """Validate a 1D finite floating-point LayerNorm parameter tensor."""
    t = _validate_fp_tensor(tensor, name, exact_ndim=1)
    if expected_dim is not None and int(t.shape[0]) != expected_dim:
        raise ValueError(f"{name} length {t.shape[0]} must match hidden dim {expected_dim}")
    return t


def _validate_int_tensor(tensor: Any, expected_shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Validate an integer tensor matching ``expected_shape``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} must match {expected_shape}")
    if tensor.is_floating_point() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must have an integer dtype, got {tensor.dtype}")
    return tensor.detach().to(device="cpu", dtype=torch.long).contiguous()


def _validate_grid_values(
    grid: Sequence[float],
    name: str,
    *,
    unit_interval: bool = False,
) -> tuple[float, ...]:
    """Validate a non-empty numeric sequence for ``lambda_grid`` or positive ``tau_grid``."""
    if isinstance(grid, (str, bytes, bytearray)) or not isinstance(grid, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of floats")
    if len(grid) == 0:
        raise ValueError(f"{name} must be non-empty")
    cleaned: list[float] = []
    for idx, val in enumerate(grid):
        if unit_interval:
            cleaned.append(_validate_unit_interval(val, f"{name}[{idx}]", allow_zero=True))
        else:
            cleaned.append(_validate_positive_finite(val, f"{name}[{idx}]"))
    return tuple(dict.fromkeys(cleaned))


def _validate_expert_index_set(indices: Any, name: str) -> set[int]:
    """Validate a sequence or set of non-negative integer expert indices."""
    if isinstance(indices, (str, bytes, bytearray)) or not isinstance(indices, (Sequence, set, frozenset)):
        raise TypeError(f"{name} must be a sequence or set of non-negative integers")
    return {_validate_nonneg_int(item, f"{name} element") for item in indices}


def _validate_square_projector(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a finite symmetric square 2D projector matrix on CPU ``float64``."""
    t = _validate_fp_tensor(proj, name, exact_ndim=2)
    if t.shape[0] != t.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(t.shape)}")
    if expected_dim is not None and int(t.shape[0]) != expected_dim:
        raise ValueError(f"{name} shape {tuple(t.shape)} must match ({expected_dim}, {expected_dim})")
    mat = t.detach().to(device="cpu", dtype=torch.float64).contiguous()
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    return (0.5 * (mat + mat.T)).contiguous()


def verify_composer_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != COMPOSER_RECEIPT_OBJECT or receipt.get("schema") != COMPOSER_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class MultiArmComposerConfig:
    """Validated configuration governing domain-matched multi-arm MoE composition."""

    fold_ln_lambda: float | None = None
    fold_tau_temp: float | None = None
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    calibrator_max_general_drift_rel: float = DEFAULT_CALIBRATOR_GENERAL_GUARD_REL
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    auto_project_p_null: bool = True
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL
    require_beat_all_donors: bool = True
    min_energy_improvement: float = 0.0
    min_domain_gain_nats: float = 0.0
    revert_harmful_to_zero: bool = False
    rollback_target: str = "base"
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = True

    def __post_init__(self) -> None:
        if self.fold_ln_lambda is not None:
            _validate_unit_interval(self.fold_ln_lambda, "fold_ln_lambda", allow_zero=True)
        if self.fold_tau_temp is not None:
            _validate_positive_finite(self.fold_tau_temp, "fold_tau_temp")
        object.__setattr__(
            self, "lambda_grid", _validate_grid_values(self.lambda_grid, "lambda_grid", unit_interval=True)
        )
        object.__setattr__(
            self, "tau_grid", tuple(sorted(_validate_grid_values(self.tau_grid, "tau_grid")))
        )
        _validate_nonneg_finite(self.calibrator_max_general_drift_rel, "calibrator_max_general_drift_rel")
        max_gen = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_worst = _validate_positive_finite(self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio")
        if max_gen < 1.0 or max_worst < 1.0:
            raise ValueError("max_general_drift_ratio and max_worst_domain_drift_ratio must be >= 1.0")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.auto_project_p_null, "auto_project_p_null")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        _validate_unit_interval(
            self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
        )
        eig_t = _validate_unit_interval(
            self.nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True
        )
        if eig_t >= 1.0:
            raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")
        _validate_nonneg_finite(self.nullspace_leak_tol, "nullspace_leak_tol")
        _validate_bool(self.require_beat_all_donors, "require_beat_all_donors")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_nonneg_finite(self.min_domain_gain_nats, "min_domain_gain_nats")
        _validate_bool(self.revert_harmful_to_zero, "revert_harmful_to_zero")
        _validate_nonempty_str(self.rollback_target, "rollback_target")
        _validate_nonempty_str(self.general_prefix, "general_prefix")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Component (a): Exact 2-Matmul Affine LayerNorm & Temperature Grid
# ---------------------------------------------------------------------------


def _resolve_general_token_mask(
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Validate and broadcast ``is_general_mask`` to a boolean tensor matching ``batch_shape``."""
    if isinstance(is_general_mask, torch.Tensor):
        if is_general_mask.dtype != torch.bool:
            raise TypeError(f"is_general_mask tensor must have bool dtype, got {is_general_mask.dtype}")
        mask_t = is_general_mask.detach().to(device="cpu")
    elif isinstance(is_general_mask, np.ndarray):
        if is_general_mask.dtype != np.bool_:
            raise TypeError(f"is_general_mask ndarray must have bool dtype, got {is_general_mask.dtype}")
        mask_t = torch.from_numpy(is_general_mask.copy())
    elif isinstance(is_general_mask, Sequence) and not isinstance(is_general_mask, (str, bytes, bytearray)):
        if len(is_general_mask) == 0:
            raise ValueError("is_general_mask must not be empty")
        if not all(isinstance(v, bool) for v in is_general_mask):
            raise TypeError("is_general_mask sequence elements must be bool")
        mask_t = torch.tensor(list(is_general_mask), dtype=torch.bool)
    else:
        raise TypeError(
            "is_general_mask must be a bool Tensor, ndarray, or Sequence[bool], "
            f"got {type(is_general_mask).__name__}"
        )

    if tuple(mask_t.shape) == batch_shape:
        return mask_t.contiguous()
    if mask_t.ndim == 1 and int(mask_t.shape[0]) == batch_shape[0]:
        expanded = mask_t.view(batch_shape[0], *([1] * (len(batch_shape) - 1)))
        return expanded.expand(batch_shape).contiguous()
    raise ValueError(
        f"is_general_mask shape {tuple(mask_t.shape)} must match {batch_shape} or ({batch_shape[0]},)"
    )


@torch.no_grad()
def fast_affine_layernorm_temperature_grid(
    h_pre_ln: torch.Tensor,
    base_ln_w: torch.Tensor,
    base_ln_b: torch.Tensor | None,
    cand_ln_w: torch.Tensor,
    cand_ln_b: torch.Tensor | None,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    *,
    eps: float = 1e-5,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    max_general_drift_rel: float = DEFAULT_CALIBRATOR_GENERAL_GUARD_REL,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    seq_batch_size: int = DEFAULT_BATCH_SIZE,
    base_general_nll_override: float | None = None,
    base_total_nll_override: float | None = None,
    domain_labels: Sequence[str] | None = None,
    max_domain_drift_rel: float | None = None,
) -> dict[str, Any]:
    """Evaluate exact 2D ``(lambda_ln, tau)`` LayerNorm NLL grid with 2 head matmuls per batch.

    Exploits exact affine logit linearity of final LayerNorm + unembedding:

        ``layer_norm(h; w_0 + lam * dw, b_0 + lam * db) W^T / tau = (L_0 + lam * dL) / tau``

    so evaluating an arbitrary ``|lambda_grid| x |tau_grid|`` (e.g., 50-point) grid requires
    only 2 ``F.linear`` head matmuls per chunk batch (``L_0`` and ``dL``) instead of 50.
    """
    h_t = _validate_fp_tensor(h_pre_ln, "h_pre_ln", min_ndim=2)
    hidden_dim = int(h_t.shape[-1])
    w0_t = _validate_1d_param(base_ln_w, "base_ln_w", expected_dim=hidden_dim)
    w1_t = _validate_1d_param(cand_ln_w, "cand_ln_w", expected_dim=hidden_dim)
    if (base_ln_b is None) != (cand_ln_b is None):
        raise ValueError("base_ln_b and cand_ln_b must either both be provided or both be None")
    b0_t = _validate_1d_param(base_ln_b, "base_ln_b", expected_dim=hidden_dim) if base_ln_b is not None else None
    b1_t = _validate_1d_param(cand_ln_b, "cand_ln_b", expected_dim=hidden_dim) if cand_ln_b is not None else None

    head_t = _validate_fp_tensor(head_weight, "head_weight", exact_ndim=2)
    if int(head_t.shape[1]) != hidden_dim:
        raise ValueError(f"head_weight width {head_t.shape[1]} must match hidden dim {hidden_dim}")
    vocab_size = int(head_t.shape[0])

    batch_shape = tuple(h_t.shape[:-1])
    targets_t = _validate_int_tensor(targets, batch_shape, "targets")
    gen_mask_t = _resolve_general_token_mask(is_general_mask, batch_shape)

    eps_f = _validate_positive_finite(eps, "eps")
    lam_vals = _validate_grid_values(lambda_grid, "lambda_grid", unit_interval=True)
    tau_vals = tuple(sorted(_validate_grid_values(tau_grid, "tau_grid")))
    guard_rel = _validate_nonneg_finite(max_general_drift_rel, "max_general_drift_rel")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    step_b = _validate_positive_int(seq_batch_size, "seq_batch_size")

    valid_all = targets_t != ignore_i
    total_valid = int(valid_all.sum().item())
    if total_valid == 0:
        raise ValueError("targets must contain at least one non-ignored token")
    valid_targets_flat = targets_t[valid_all]
    if bool((valid_targets_flat < 0).any().item()) or bool((valid_targets_flat >= vocab_size).any().item()):
        raise ValueError(
            f"targets token IDs must lie in [0, {vocab_size}) or equal ignore_index={ignore_i}"
        )
    gen_valid_all = gen_mask_t[valid_all]
    total_gen = int(gen_valid_all.sum().item())
    if total_gen == 0:
        raise ValueError("is_general_mask must select at least one non-ignored general token")
    total_spec = total_valid - total_gen

    domain_groups: list[tuple[str, torch.Tensor]] = []
    if domain_labels is not None:
        if len(domain_labels) != batch_shape[0]:
            raise ValueError(
                f"domain_labels length {len(domain_labels)} must match batch dim {batch_shape[0]}"
            )
        if max_domain_drift_rel is not None:
            _validate_nonneg_finite(max_domain_drift_rel, "max_domain_drift_rel")
        idx_by_dom: dict[str, list[int]] = {}
        for row_i, d_lbl in enumerate(domain_labels):
            idx_by_dom.setdefault(str(d_lbl), []).append(row_i)
        for d_lbl in sorted(idx_by_dom.keys()):
            d_mask = torch.zeros(batch_shape, dtype=torch.bool)
            d_mask[idx_by_dom[d_lbl]] = True
            if int((d_mask & valid_all).sum().item()) > 0:
                domain_groups.append((d_lbl, d_mask))

    h64 = h_t.detach().to(device="cpu", dtype=torch.float64)
    w0_64 = w0_t.detach().to(device="cpu", dtype=torch.float64)
    w1_64 = w1_t.detach().to(device="cpu", dtype=torch.float64)
    dw_64 = w1_64 - w0_64
    b0_64 = b0_t.detach().to(device="cpu", dtype=torch.float64) if b0_t is not None else None
    b1_64 = b1_t.detach().to(device="cpu", dtype=torch.float64) if b1_t is not None else None
    db_64 = (b1_64 - b0_64) if (b0_64 is not None and b1_64 is not None) else None
    head_64 = head_t.detach().to(device="cpu", dtype=torch.float64)

    has_delta = bool((dw_64 != 0).any().item()) or (db_64 is not None and bool((db_64 != 0).any().item()))

    n_lam = len(lam_vals)
    n_tau = len(tau_vals)
    nll_sum = np.zeros((n_lam, n_tau), dtype=np.float64)
    gen_nll_sum = np.zeros((n_lam, n_tau), dtype=np.float64)
    spec_nll_sum = np.zeros((n_lam, n_tau), dtype=np.float64)
    dom_nll_sums = {d_lbl: np.zeros((n_lam, n_tau), dtype=np.float64) for d_lbl, _ in domain_groups}
    dom_counts = {d_lbl: 0 for d_lbl, _ in domain_groups}

    base_nll_sum = 0.0
    base_gen_sum = 0.0
    base_spec_sum = 0.0
    base_dom_sums = {d_lbl: 0.0 for d_lbl, _ in domain_groups}

    n_rows = int(h64.shape[0])
    num_batches = 0
    matmuls_used = 0
    for c0 in range(0, n_rows, step_b):
        c1 = min(n_rows, c0 + step_b)
        sub_valid = valid_all[c0:c1].reshape(-1)
        if not bool(sub_valid.any().item()):
            continue
        num_batches += 1
        sub_h = h64[c0:c1]
        z = F.layer_norm(sub_h, (hidden_dim,), None, None, eps=eps_f)
        u0 = z * w0_64 if b0_64 is None else (z * w0_64 + b0_64)
        l0 = F.linear(u0, head_64)
        matmuls_used += 1
        if has_delta:
            du = z * dw_64 if db_64 is None else (z * dw_64 + db_64)
            dl = F.linear(du, head_64)
            matmuls_used += 1
        else:
            dl = torch.zeros_like(l0)

        l0_valid = l0.reshape(-1, vocab_size)[sub_valid]
        dl_valid = dl.reshape(-1, vocab_size)[sub_valid]
        t_valid = targets_t[c0:c1].reshape(-1)[sub_valid]
        g_valid = gen_mask_t[c0:c1].reshape(-1)[sub_valid]
        s_valid = ~g_valid
        dom_valid = {d_lbl: d_mask[c0:c1].reshape(-1)[sub_valid] for d_lbl, d_mask in domain_groups}
        for d_lbl, dv in dom_valid.items():
            dom_counts[d_lbl] += int(dv.sum().item())

        base_tok = F.cross_entropy(l0_valid, t_valid, reduction="none")
        base_nll_sum += float(base_tok.sum().item())
        base_gen_sum += float(base_tok[g_valid].sum().item())
        if bool(s_valid.any().item()):
            base_spec_sum += float(base_tok[s_valid].sum().item())
        for d_lbl, dv in dom_valid.items():
            if bool(dv.any().item()):
                base_dom_sums[d_lbl] += float(base_tok[dv].sum().item())

        for i_lam, lam in enumerate(lam_vals):
            l_lam = l0_valid if (not has_delta or lam == 0.0) else (l0_valid + float(lam) * dl_valid)
            for j_tau, tau in enumerate(tau_vals):
                logits_jt = l_lam if tau == 1.0 else (l_lam * (1.0 / float(tau)))
                tok_nll = F.cross_entropy(logits_jt, t_valid, reduction="none")
                nll_sum[i_lam, j_tau] += float(tok_nll.sum().item())
                gen_nll_sum[i_lam, j_tau] += float(tok_nll[g_valid].sum().item())
                if bool(s_valid.any().item()):
                    spec_nll_sum[i_lam, j_tau] += float(tok_nll[s_valid].sum().item())
                for d_lbl, dv in dom_valid.items():
                    if bool(dv.any().item()):
                        dom_nll_sums[d_lbl][i_lam, j_tau] += float(tok_nll[dv].sum().item())

    nll_grid_arr = nll_sum / float(total_valid)
    gen_grid_arr = gen_nll_sum / float(total_gen)
    spec_grid_arr = (spec_nll_sum / float(total_spec)) if total_spec > 0 else None

    raw_base_total = base_nll_sum / float(total_valid)
    raw_base_gen = base_gen_sum / float(total_gen)
    raw_base_spec = (base_spec_sum / float(total_spec)) if total_spec > 0 else None

    ref_total = (
        raw_base_total
        if base_total_nll_override is None
        else _validate_positive_finite(base_total_nll_override, "base_total_nll_override")
    )
    ref_gen = (
        raw_base_gen
        if base_general_nll_override is None
        else _validate_positive_finite(base_general_nll_override, "base_general_nll_override")
    )
    gen_ceiling = (1.0 + guard_rel) * ref_gen
    gen_drift_grid_arr = (gen_grid_arr / max(ref_gen, 1e-12)) - 1.0

    dom_rel_limit = _validate_nonneg_finite(max_domain_drift_rel, "max_domain_drift_rel") if max_domain_drift_rel is not None else None
    dom_ceilings = (
        {
            d_lbl: (1.0 + dom_rel_limit) * (base_dom_sums[d_lbl] / float(dom_counts[d_lbl]))
            for d_lbl, _ in domain_groups
        }
        if dom_rel_limit is not None
        else {}
    )

    feasible: list[tuple[float, float, float, float, int, int]] = []
    fallback: list[tuple[float, float, float, float, int, int]] = []
    for i_lam, lam in enumerate(lam_vals):
        for j_tau, tau in enumerate(tau_vals):
            mean_nll = float(nll_grid_arr[i_lam, j_tau])
            gen_nll = float(gen_grid_arr[i_lam, j_tau])
            gen_drift = float(gen_drift_grid_arr[i_lam, j_tau])
            mean_diff = mean_nll - ref_total
            ok = gen_nll <= gen_ceiling + 1e-12
            if ok and dom_ceilings:
                for d_lbl, ceil_val in dom_ceilings.items():
                    dom_mean = float(dom_nll_sums[d_lbl][i_lam, j_tau]) / float(dom_counts[d_lbl])
                    if dom_mean > ceil_val + 1e-12:
                        ok = False
                        break
            if ok:
                feasible.append((mean_diff, gen_drift, abs(float(tau) - 1.0), abs(float(lam) - 1.0), i_lam, j_tau))
            fallback.append((gen_drift, mean_diff, abs(float(tau) - 1.0), abs(float(lam) - 1.0), i_lam, j_tau))

    guard_feasible = bool(feasible)
    if feasible:
        _, _, _, _, best_i, best_j = min(feasible)
    else:
        _, _, _, _, best_i, best_j = min(fallback)

    best_lam = float(lam_vals[best_i])
    best_tau = float(tau_vals[best_j])
    best_nll_mean = float(nll_grid_arr[best_i, best_j])
    best_gen_nll = float(gen_grid_arr[best_i, best_j])
    best_spec_nll = float(spec_grid_arr[best_i, best_j]) if spec_grid_arr is not None else None
    best_gen_drift = float(gen_drift_grid_arr[best_i, best_j])

    w_interp = (w0_64 + best_lam * dw_64).to(device=w0_t.device, dtype=w0_t.dtype)
    b_interp = (
        (b0_64 + best_lam * db_64).to(device=b0_t.device, dtype=b0_t.dtype)
        if (b0_64 is not None and db_64 is not None and b0_t is not None)
        else None
    )
    folded_w, folded_b = fold_temperature_into_norm_params(w_interp, b_interp, best_tau)

    num_grid_points = n_lam * n_tau
    naive_head_matmuls = num_grid_points * num_batches
    speedup = float(naive_head_matmuls) / float(max(1, matmuls_used))

    return {
        "best_lambda_ln": best_lam,
        "best_tau_temp": best_tau,
        "folded_ln_weight": folded_w,
        "folded_ln_bias": folded_b,
        "guard_feasible": guard_feasible,
        "best_nll_mean": best_nll_mean,
        "best_general_nll": best_gen_nll,
        "best_specialist_nll": best_spec_nll,
        "best_general_drift_rel": best_gen_drift,
        "base_nll_mean": ref_total,
        "base_general_nll": ref_gen,
        "base_specialist_nll": raw_base_spec,
        "nll_drop_nats": ref_total - best_nll_mean,
        "domain_gain_nats": (raw_base_spec - best_spec_nll) if (raw_base_spec is not None and best_spec_nll is not None) else (ref_total - best_nll_mean),
        "lambda_grid": [float(v) for v in lam_vals],
        "tau_grid": [float(v) for v in tau_vals],
        "nll_grid": nll_grid_arr.tolist(),
        "general_nll_grid": gen_grid_arr.tolist(),
        "general_drift_grid": gen_drift_grid_arr.tolist(),
        "head_matmuls_executed": matmuls_used,
        "naive_head_matmuls": naive_head_matmuls,
        "matmul_speedup_factor": speedup,
        "num_grid_points": num_grid_points,
        "num_batches": num_batches,
        "num_valid_tokens": total_valid,
    }


# ---------------------------------------------------------------------------
# 3. Component (b): Domain-Matched Specialist Selector & P_0 Projection
# ---------------------------------------------------------------------------


def _parse_expert_slot_key(raw_key: Any) -> int | None:
    """Parse expert slot index from ``int``, ``'1'``, ``'expert_1'``, or ``'blocks.0.mlp.experts.1'``."""
    if isinstance(raw_key, int) and not isinstance(raw_key, bool) and raw_key >= 0:
        return int(raw_key)
    if isinstance(raw_key, str):
        text = raw_key.strip()
        m_blk = _BLOCK_EXPERT_RE.match(text)
        if m_blk is not None:
            return int(m_blk.group(2))
        m_exp = _EXPERT_KEY_RE.match(text)
        if m_exp is not None:
            return int(m_exp.group(1))
    return None


def _parse_layer_key(raw_key: Any) -> int | None:
    """Parse layer index from ``int``, ``'0'``, ``'layer_0'``, or ``'blocks.0'``."""
    if isinstance(raw_key, int) and not isinstance(raw_key, bool) and raw_key >= 0:
        return int(raw_key)
    if isinstance(raw_key, str):
        m = _LAYER_KEY_RE.match(raw_key.strip())
        if m is not None:
            return int(m.group(1))
    return None


def _binarize_projector(p_mat: torch.Tensor, eig_threshold: float) -> torch.Tensor:
    """Ensure ``P_0`` is an exact symmetric idempotent orthogonal projector."""
    idem_err = float((p_mat @ p_mat - p_mat).abs().max().item())
    if idem_err <= 1e-10:
        return p_mat
    evals, evecs = torch.linalg.eigh(p_mat)
    keep = evals > eig_threshold
    if not bool(keep.any().item()):
        return torch.zeros_like(p_mat)
    q_keep = evecs[:, keep]
    proj = q_keep @ q_keep.T
    return (0.5 * (proj + proj.T)).contiguous()


def _resolve_expert_nullspace_projectors(
    p_null_by_expert: Any,
    layer_idx: int,
    expert_idx: int,
    *,
    hidden_size: int | None,
    intermediate_size: int | None,
    eig_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve ``(P_x, P_act)`` projectors for ``(layer_idx, expert_idx)``."""
    if p_null_by_expert is None:
        return None, None
    if isinstance(p_null_by_expert, torch.Tensor):
        p_t = _binarize_projector(
            _validate_square_projector(p_null_by_expert, "p_null_by_expert"), eig_threshold
        )
        dim = int(p_t.shape[0])
        if hidden_size is None or dim == hidden_size:
            return p_t, None
        if intermediate_size is not None and dim == intermediate_size:
            return None, p_t
        raise ValueError(
            f"Projector shape {tuple(p_t.shape)} matches neither hidden_size={hidden_size} "
            f"nor intermediate_size={intermediate_size}"
        )
    if not isinstance(p_null_by_expert, Mapping):
        raise TypeError("p_null_by_expert must be a 2D Tensor, Mapping, or None")
    if not p_null_by_expert:
        return None, None

    px_out: torch.Tensor | None = None
    pact_out: torch.Tensor | None = None

    def _consume(entry: Any, label: str) -> None:
        nonlocal px_out, pact_out
        if isinstance(entry, torch.Tensor):
            p_m = _binarize_projector(_validate_square_projector(entry, label), eig_threshold)
            dim = int(p_m.shape[0])
            if (hidden_size is None or dim == hidden_size) and px_out is None:
                px_out = p_m
            elif intermediate_size is not None and dim == intermediate_size and pact_out is None:
                pact_out = p_m
        elif isinstance(entry, Mapping):
            if "x" in entry and isinstance(entry["x"], torch.Tensor) and px_out is None:
                px_out = _binarize_projector(
                    _validate_square_projector(entry["x"], f"{label}['x']", expected_dim=hidden_size),
                    eig_threshold,
                )
            if (
                "act" in entry
                and isinstance(entry["act"], torch.Tensor)
                and intermediate_size is not None
                and pact_out is None
            ):
                pact_out = _binarize_projector(
                    _validate_square_projector(
                        entry["act"], f"{label}['act']", expected_dim=intermediate_size
                    ),
                    eig_threshold,
                )
            for sub_k in (expert_idx, str(expert_idx), f"expert_{expert_idx}", "default"):
                if sub_k in entry:
                    _consume(entry[sub_k], f"{label}[{sub_k!r}]")

    candidates: list[Any] = [
        (layer_idx, expert_idx),
        f"blocks.{layer_idx}.mlp.experts.{expert_idx}",
        f"blocks.{layer_idx}",
        layer_idx,
        str(layer_idx),
        expert_idx,
        str(expert_idx),
        f"expert_{expert_idx}",
        "default",
        "all",
    ]
    for cand in candidates:
        if cand in p_null_by_expert:
            _consume(p_null_by_expert[cand], f"p_null[{cand!r}]")
    if "x" in p_null_by_expert and px_out is None:
        _consume({"x": p_null_by_expert["x"]}, "p_null")
    if "act" in p_null_by_expert and pact_out is None:
        _consume({"act": p_null_by_expert["act"]}, "p_null")
    return px_out, pact_out


def _normalize_single_expert_payload(
    payload: Any,
    label: str,
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Normalize one expert delta/weight entry into ``(tensors_dict, kind, style)``."""
    if isinstance(payload, torch.Tensor):
        t = _validate_fp_tensor(payload, label, min_ndim=1)
        return {"delta": t.detach().clone()}, "dense", "tensor"
    if isinstance(payload, Mapping) and payload:
        t_map: dict[str, torch.Tensor] = {}
        for k, v in payload.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError(f"{label} keys must be non-empty strings")
            if k == "scaling":
                continue
            t = _validate_fp_tensor(v, f"{label}[{k!r}]", min_ndim=1)
            t_map[k] = t.detach().clone()
        if not t_map:
            raise ValueError(f"{label} contains no floating-point tensors")
        is_lora = all(f"{p}_{s}" in t_map for p in _SWIGLU_PROJECTIONS for s in ("A", "B"))
        is_swiglu = all(p in t_map for p in _SWIGLU_PROJECTIONS)
        kind = "lora" if is_lora else ("swiglu" if is_swiglu else "dense")
        return t_map, kind, "mapping"
    raise TypeError(f"{label} must be a non-empty Tensor or Mapping[str, Tensor], got {type(payload).__name__}")


def _normalize_arm_experts_layout(
    arm_data: Any,
    arm_name: str,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Normalize one arm's expert deltas into ``{layer_idx: {expert_idx: spec}}`` + layout metadata."""
    if isinstance(arm_data, nn.Module):
        blocks = extract_checkpoint_blocks(arm_data)
        arm_map: Mapping[str, Any] = {
            k: v for k, v in blocks.items() if _BLOCK_EXPERT_RE.match(k) is not None
        }
        if not arm_map:
            raise ValueError(f"arms_expert_deltas[{arm_name!r}] module exposes no MoE expert blocks")
    elif isinstance(arm_data, Mapping) and arm_data:
        arm_map = arm_data
    else:
        raise TypeError(
            f"arms_expert_deltas[{arm_name!r}] must be a non-empty Mapping or nn.Module, "
            f"got {type(arm_data).__name__}"
        )

    # Detect layout style: checkpoint_blocks | flat_experts | nested_layers
    first_key = next(iter(arm_map.keys()))
    if isinstance(first_key, str) and _BLOCK_EXPERT_RE.match(first_key.strip()):
        norm: dict[int, dict[int, dict[str, Any]]] = {}
        key_map: dict[tuple[int, int], Any] = {}
        for k, val in arm_map.items():
            m = _BLOCK_EXPERT_RE.match(str(k).strip())
            if m is None:
                raise ValueError(f"Mixed block key {k!r} in arms_expert_deltas[{arm_name!r}]")
            l_idx, e_idx = int(m.group(1)), int(m.group(2))
            t_map, kind, entry_style = _normalize_single_expert_payload(val, f"{arm_name}[{k!r}]")
            norm.setdefault(l_idx, {})[e_idx] = {"tensors": t_map, "kind": kind, "style": entry_style}
            key_map[(l_idx, e_idx)] = k
        return norm, {"style": "checkpoint_blocks", "keys": key_map}

    # Check if top-level keys represent expert slots directly (mapping to Tensor or Tensor dict)
    first_val = arm_map[first_key]
    is_flat = isinstance(first_val, torch.Tensor) or (
        isinstance(first_val, Mapping)
        and bool(first_val)
        and isinstance(next(iter(first_val.values())), torch.Tensor)
    )
    if is_flat:
        norm = {0: {}}
        key_map = {}
        for k, val in arm_map.items():
            e_idx = _parse_expert_slot_key(k)
            if e_idx is None:
                raise ValueError(f"Cannot parse expert slot index from {k!r} in arm {arm_name!r}")
            t_map, kind, entry_style = _normalize_single_expert_payload(val, f"{arm_name}[{k!r}]")
            norm[0][e_idx] = {"tensors": t_map, "kind": kind, "style": entry_style}
            key_map[(0, e_idx)] = k
        return norm, {"style": "flat_experts", "keys": key_map}

    # Nested layer -> expert -> payload
    norm = {}
    layer_keys: dict[int, Any] = {}
    key_map = {}
    for lk, exp_map in arm_map.items():
        l_idx = _parse_layer_key(lk)
        if l_idx is None:
            raise ValueError(f"Cannot parse layer index from {lk!r} in arm {arm_name!r}")
        if not isinstance(exp_map, Mapping) or not exp_map:
            raise TypeError(f"{arm_name}[{lk!r}] must be a non-empty Mapping of experts")
        layer_keys[l_idx] = lk
        norm[l_idx] = {}
        for ek, val in exp_map.items():
            e_idx = _parse_expert_slot_key(ek)
            if e_idx is None:
                raise ValueError(f"Cannot parse expert index from {ek!r} in {arm_name}[{lk!r}]")
            t_map, kind, entry_style = _normalize_single_expert_payload(val, f"{arm_name}[{lk!r}][{ek!r}]")
            norm[l_idx][e_idx] = {"tensors": t_map, "kind": kind, "style": entry_style}
            key_map[(l_idx, e_idx)] = ek
    return norm, {"style": "nested_layers", "layer_keys": layer_keys, "keys": key_map}


def _extract_domain_loss_delta(
    entry: Any,
    base_val: float | None,
    label: str,
) -> tuple[float, float]:
    """Return ``(signed_delta_vs_base, raw_arm_loss)`` for one domain probe entry."""
    if isinstance(entry, Mapping):
        if "mean_diff" in entry:
            diff = _validate_finite_float(entry["mean_diff"], f"{label}['mean_diff']")
            raw_v = _validate_finite_float(entry.get("moe_nll", diff), f"{label}['moe_nll']")
            return diff, raw_v
        if "delta_nll" in entry:
            diff = _validate_finite_float(entry["delta_nll"], f"{label}['delta_nll']")
            raw_v = _validate_finite_float(entry.get("arm_nll", diff), f"{label}['arm_nll']")
            return diff, raw_v
        cand_key = next((k for k in ("arm_nll", "cand_nll", "moe_nll", "nll") if k in entry), None)
        if cand_key is None:
            raise ValueError(f"{label} mapping must contain 'mean_diff', 'delta_nll', or 'moe_nll'/'nll'")
        arm_v = _validate_finite_float(entry[cand_key], f"{label}[{cand_key!r}]")
        ref_v = (
            _validate_finite_float(entry["base_nll"], f"{label}['base_nll']")
            if "base_nll" in entry
            else base_val
        )
        return (arm_v - ref_v if ref_v is not None else arm_v), arm_v

    arm_v = _validate_finite_float(entry, label)
    return (arm_v - base_val if base_val is not None else arm_v), arm_v


def _resolve_expert_domain_keys(
    e_idx: int,
    available_domains: Sequence[str],
    expert_domain_map: Mapping[int, str | Sequence[str]] | None,
) -> list[str]:
    """Map expert slot ``e_idx`` to matching domain key(s) in ``available_domains``."""
    avail_set = set(available_domains)
    if expert_domain_map is not None and e_idx in expert_domain_map:
        mapped = expert_domain_map[e_idx]
        keys = [mapped] if isinstance(mapped, str) else list(mapped)
        matched = [k for k in keys if k in avail_set]
        if not matched:
            raise ValueError(
                f"expert_domain_map[{e_idx}] keys {keys!r} not found in probe_domain_losses_by_arm"
            )
        return matched

    matches: list[str] = []
    for dom in available_domains:
        m = _DOMAIN_INDEX_RE.match(dom)
        if m is not None:
            idx_str = m.group(1) or m.group(2) or m.group(3)
            if idx_str is not None and int(idx_str) == e_idx:
                matches.append(dom)
    return matches


def _project_lora_pair_in_nullspace(
    b_base: torch.Tensor | None,
    a_base: torch.Tensor | None,
    b_arm: torch.Tensor,
    a_arm: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
    leak_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, bool]:
    """Project LoRA delta ``B_arm A_arm - B_base A_base`` into ``P_0`` if leakage > ``leak_tol``."""
    b1 = b_arm.detach().to("cpu", dtype=torch.float64)
    a1 = a_arm.detach().to("cpu", dtype=torch.float64)
    q_perp = torch.eye(int(p_mat.shape[0]), dtype=torch.float64) - p_mat
    b0 = b_base.detach().to("cpu", dtype=torch.float64) if b_base is not None else torch.zeros_like(b1)
    a0 = a_base.detach().to("cpu", dtype=torch.float64) if a_base is not None else torch.zeros_like(a1)
    zero_base = int(torch.count_nonzero(b0).item()) == 0

    if zero_base:
        leak_before = float(
            torch.linalg.matrix_norm(b1 @ (a1 @ q_perp)).item()
            if side == "input"
            else torch.linalg.matrix_norm((q_perp @ b1) @ a1).item()
        )
    else:
        d_b_pre = torch.cat([b1, -b0], dim=1)
        d_a_pre = torch.cat([a1, a0], dim=0)
        _, s_pre, _ = decompose_lora_qr_svd(
            d_b_pre if side == "input" else (q_perp @ d_b_pre),
            (d_a_pre @ q_perp) if side == "input" else d_a_pre,
        )
        leak_before = float(s_pre.pow(2).sum().sqrt().item())

    if leak_before <= leak_tol:
        return b_arm.detach().clone(), a_arm.detach().clone(), leak_before, leak_before, False

    if zero_base:
        if side == "input":
            b_new = b1.contiguous()
            a_new = (a1 @ p_mat).contiguous()
            leak_after = float(torch.linalg.matrix_norm(b_new @ (a_new @ q_perp)).item())
        else:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
            leak_after = float(torch.linalg.matrix_norm((q_perp @ b_new) @ a_new).item())
    else:
        base_tol = max(1e-10, leak_tol)
        if side == "input":
            base_perp = float(torch.linalg.matrix_norm(b0 @ (a0 @ q_perp)).item())
            if base_perp <= base_tol:
                b_new = b1.contiguous()
                a_new = (a1 @ p_mat).contiguous()
            else:
                b_new = b0.contiguous()
                a_proj = torch.linalg.lstsq(b0, b1 @ (a1 @ p_mat)).solution
                a_new = (a0 @ q_perp + a_proj @ p_mat).contiguous()
            _, s_post, _ = decompose_lora_qr_svd(
                torch.cat([b_new, -b0], dim=1), torch.cat([a_new @ q_perp, a0 @ q_perp], dim=0)
            )
        else:
            base_perp = float(torch.linalg.matrix_norm((q_perp @ b0) @ a0).item())
            if base_perp <= base_tol:
                b_new = (p_mat @ b1).contiguous()
                a_new = a1.contiguous()
            else:
                a_new = a0.contiguous()
                b_proj_t = torch.linalg.lstsq(a0.T, (p_mat @ b1 @ a1).T).solution
                b_new = (q_perp @ b0 + p_mat @ b_proj_t.T).contiguous()
            _, s_post, _ = decompose_lora_qr_svd(
                torch.cat([q_perp @ b_new, -(q_perp @ b0)], dim=1), torch.cat([a_new, a0], dim=0)
            )
        leak_after = float(s_post.pow(2).sum().sqrt().item())

    return (
        b_new.to(device=b_arm.device, dtype=b_arm.dtype),
        a_new.to(device=a_arm.device, dtype=a_arm.dtype),
        leak_before,
        leak_after,
        True,
    )


def _project_dense_tensor_in_nullspace(
    t_arm: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
    leak_tol: float,
    *,
    t_base: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float, bool]:
    """Project dense tensor (or delta ``t_arm - t_base``) into ``P_0`` when leakage > ``leak_tol``."""
    u64 = t_arm.detach().to("cpu", dtype=torch.float64)
    b64 = t_base.detach().to("cpu", dtype=torch.float64) if t_base is not None else torch.zeros_like(u64)
    delta = u64 - b64
    q_perp = torch.eye(int(p_mat.shape[0]), dtype=torch.float64) - p_mat

    if delta.ndim == 1:
        leak_before = float(torch.linalg.vector_norm(q_perp @ delta).item())
        if leak_before <= leak_tol:
            return t_arm.detach().clone(), leak_before, leak_before, False
        proj = p_mat @ delta
        leak_after = float(torch.linalg.vector_norm(q_perp @ proj).item())
        return (b64 + proj).to(device=t_arm.device, dtype=t_arm.dtype).contiguous(), leak_before, leak_after, True

    leak_before = float(
        torch.linalg.matrix_norm(delta @ q_perp).item()
        if side == "input"
        else torch.linalg.matrix_norm(q_perp @ delta).item()
    )
    if leak_before <= leak_tol:
        return t_arm.detach().clone(), leak_before, leak_before, False
    proj = (delta @ p_mat) if side == "input" else (p_mat @ delta)
    leak_after = float(
        torch.linalg.matrix_norm((proj @ q_perp) if side == "input" else (q_perp @ proj)).item()
    )
    return (b64 + proj).to(device=t_arm.device, dtype=t_arm.dtype).contiguous(), leak_before, leak_after, True


def _repackage_composed_experts(
    composed: Mapping[int, Mapping[int, dict[str, Any]]],
    layout_meta: Mapping[str, Any],
) -> dict[Any, Any]:
    """Repackage normalized composed expert tensors into caller's exact input layout."""
    style = layout_meta["style"]
    key_map = layout_meta["keys"]

    def _emit_entry(entry: Mapping[str, Any]) -> Any:
        if entry["style"] == "tensor":
            return entry["tensors"]["delta"].detach().clone()
        return {k: v.detach().clone() for k, v in entry["tensors"].items()}

    if style == "checkpoint_blocks":
        return {
            key_map[(l_idx, e_idx)]: _emit_entry(entry)
            for l_idx in sorted(composed.keys())
            for e_idx, entry in sorted(composed[l_idx].items())
        }
    if style == "flat_experts":
        return {
            key_map[(0, e_idx)]: _emit_entry(entry)
            for e_idx, entry in sorted(composed[0].items())
        }
    layer_keys = layout_meta["layer_keys"]
    return {
        layer_keys[l_idx]: {
            key_map[(l_idx, e_idx)]: _emit_entry(entry)
            for e_idx, entry in sorted(composed[l_idx].items())
        }
        for l_idx in sorted(composed.keys())
    }


@torch.no_grad()
def select_domain_winning_experts_across_arms(
    arms_expert_deltas: Mapping[str, Any],
    probe_domain_losses_by_arm: Mapping[str, Mapping[str, Any]],
    p_null_by_expert: torch.Tensor | Mapping[Any, Any] | None = None,
    *,
    base_domain_losses: Mapping[str, float] | None = None,
    base_experts: Mapping[str, Any] | None = None,
    expert_domain_map: Mapping[int, str | Sequence[str]] | None = None,
    sink_experts: Sequence[int] | set[int] = (),
    protected_experts: Sequence[int] | set[int] = (),
    general_expert_indices: Sequence[int] | set[int] = (0,),
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD,
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL,
    min_domain_gain_nats: float = 0.0,
    revert_harmful_to_zero: bool = False,
) -> tuple[dict[Any, Any], dict[str, Any]]:
    """Assign each expert slot to the donor arm achieving the deepest domain NLL drop inside ``P_0``.

    For every expert slot ``e``:
    1. Resolves matching probe domains via ``expert_domain_map`` or ``E{e}_*`` / ``D{e}_*``.
    2. Selects the donor arm with the deepest domain NLL drop (smallest ``delta_nll``).
    3. Zero-prunes ``sink_experts``, optionally preserves ``protected_experts`` bitwise,
       and projects surviving deltas into ``p_null_by_expert`` (``P_0``) so ``I - P_0``
       general-prose leakage vanishes.
    """
    if not isinstance(arms_expert_deltas, Mapping) or len(arms_expert_deltas) < 2:
        raise ValueError("arms_expert_deltas must be a mapping with at least 2 donor arms")
    if not isinstance(probe_domain_losses_by_arm, Mapping) or not probe_domain_losses_by_arm:
        raise ValueError("probe_domain_losses_by_arm must be a non-empty mapping")

    sink_set = _validate_expert_index_set(sink_experts, "sink_experts")
    prot_set = _validate_expert_index_set(protected_experts, "protected_experts")
    gen_slots = _validate_expert_index_set(general_expert_indices, "general_expert_indices")
    overlap = sink_set & prot_set
    if overlap:
        raise ValueError(f"Experts cannot be both sink_experts and protected_experts: {sorted(overlap)}")
    eig_thr = _validate_unit_interval(nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
    if eig_thr >= 1.0:
        raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_thr}")
    leak_tol = _validate_nonneg_finite(nullspace_leak_tol, "nullspace_leak_tol")
    min_gain = _validate_nonneg_finite(min_domain_gain_nats, "min_domain_gain_nats")
    _validate_bool(revert_harmful_to_zero, "revert_harmful_to_zero")
    _validate_nonempty_str(general_prefix, "general_prefix")

    arm_names = list(arms_expert_deltas.keys())
    for arm in arm_names:
        _validate_nonempty_str(arm, "arm_name")
        if arm not in probe_domain_losses_by_arm:
            raise ValueError(f"probe_domain_losses_by_arm is missing donor arm {arm!r}")
        if not isinstance(probe_domain_losses_by_arm[arm], Mapping) or not probe_domain_losses_by_arm[arm]:
            raise ValueError(f"probe_domain_losses_by_arm[{arm!r}] must be a non-empty mapping")

    ref_base_losses: dict[str, float] | None = None
    if base_domain_losses is not None:
        if not isinstance(base_domain_losses, Mapping) or not base_domain_losses:
            raise ValueError("base_domain_losses must be a non-empty mapping when provided")
        ref_base_losses = {
            str(k): _validate_finite_float(v, f"base_domain_losses[{k!r}]")
            for k, v in base_domain_losses.items()
        }
    else:
        for base_cand in ("base", "BASE", "base_checkpoint"):
            if base_cand in probe_domain_losses_by_arm and base_cand not in arms_expert_deltas:
                raw_b = probe_domain_losses_by_arm[base_cand]
                ref_base_losses = {
                    str(k): _validate_finite_float(v, f"probe_domain_losses_by_arm[{base_cand!r}][{k!r}]")
                    for k, v in raw_b.items()
                }
                break

    available_domains = sorted(
        set.intersection(*(set(probe_domain_losses_by_arm[arm].keys()) for arm in arm_names))
    )
    if not available_domains:
        raise ValueError("Donor arms share no common domain keys in probe_domain_losses_by_arm")

    arm_deltas: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    ref_meta: dict[str, Any] | None = None
    ref_layers: list[int] = []
    ref_experts: list[int] = []
    for arm in arm_names:
        norm_arm, meta_arm = _normalize_arm_experts_layout(arms_expert_deltas[arm], arm)
        arm_deltas[arm] = norm_arm
        layers_sorted = sorted(norm_arm.keys())
        experts_sorted = sorted(norm_arm[layers_sorted[0]].keys())
        if ref_meta is None:
            ref_meta = meta_arm
            ref_layers = layers_sorted
            ref_experts = experts_sorted
        else:
            if layers_sorted != ref_layers or any(
                sorted(norm_arm[l_idx].keys()) != sorted(arm_deltas[arm_names[0]][l_idx].keys())
                for l_idx in ref_layers
            ):
                raise ValueError("All donor arms must expose identical layer and expert slot layouts")
            for l_idx in ref_layers:
                for e_idx in sorted(norm_arm[l_idx].keys()):
                    ref_t = arm_deltas[arm_names[0]][l_idx][e_idx]["tensors"]
                    cur_t = norm_arm[l_idx][e_idx]["tensors"]
                    if set(ref_t.keys()) != set(cur_t.keys()) or any(
                        ref_t[sk].shape != cur_t[sk].shape for sk in ref_t
                    ):
                        raise ValueError(f"Tensor layout/shape mismatch across arms at layer={l_idx}, expert={e_idx}")

    base_norm: dict[int, dict[int, dict[str, Any]]] | None = None
    if base_experts is not None:
        base_norm, _ = _normalize_arm_experts_layout(base_experts, "base_experts")

    # Compute overall mean delta per arm as deterministic fallback when an expert has no domain key
    arm_overall_delta: dict[str, float] = {}
    arm_general_delta: dict[str, float] = {}
    gen_dom_keys = [d for d in available_domains if d.startswith(general_prefix)]
    for arm in arm_names:
        all_deltas = [
            _extract_domain_loss_delta(
                probe_domain_losses_by_arm[arm][d],
                ref_base_losses.get(d) if ref_base_losses is not None else None,
                f"probe_domain_losses_by_arm[{arm!r}][{d!r}]",
            )[0]
            for d in available_domains
        ]
        arm_overall_delta[arm] = float(np.mean(all_deltas))
        if gen_dom_keys:
            gen_deltas = [
                _extract_domain_loss_delta(
                    probe_domain_losses_by_arm[arm][d],
                    ref_base_losses.get(d) if ref_base_losses is not None else None,
                    f"probe_domain_losses_by_arm[{arm!r}][{d!r}]",
                )[0]
                for d in gen_dom_keys
            ]
            arm_general_delta[arm] = float(np.mean(gen_deltas))
        else:
            arm_general_delta[arm] = arm_overall_delta[arm]

    expert_assignments: dict[str, dict[str, Any]] = {}
    winning_arm_by_slot: dict[int, str] = {}
    arm_win_counts: dict[str, int] = {arm: 0 for arm in arm_names}
    winning_gains: list[float] = []

    for e_idx in ref_experts:
        if e_idx in sink_set:
            expert_assignments[f"expert_{e_idx}"] = {
                "expert_index": e_idx,
                "winning_arm": None,
                "domain_keys": [],
                "domain_gain_nats": 0.0,
                "action": "ghost_sink_pruned",
            }
            continue

        matched_doms = _resolve_expert_domain_keys(e_idx, available_domains, expert_domain_map)
        ranked: list[tuple[float, float, str]] = []
        for arm in arm_names:
            if matched_doms:
                dom_deltas = [
                    _extract_domain_loss_delta(
                        probe_domain_losses_by_arm[arm][d],
                        ref_base_losses.get(d) if ref_base_losses is not None else None,
                        f"probe_domain_losses_by_arm[{arm!r}][{d!r}]",
                    )[0]
                    for d in matched_doms
                ]
                slot_delta = float(np.mean(dom_deltas))
            else:
                slot_delta = arm_general_delta[arm] if e_idx in gen_slots else arm_overall_delta[arm]
            ranked.append((slot_delta, arm_general_delta[arm], arm))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        best_delta, _, best_arm = ranked[0]
        runner_delta, _, runner_arm = ranked[1]

        # If base reference is known and no donor achieves >= min_domain_gain_nats drop, optionally zero-revert
        gain_vs_runner = runner_delta - best_delta
        gain_vs_base = -best_delta if ref_base_losses is not None else gain_vs_runner
        if revert_harmful_to_zero and ref_base_losses is not None and gain_vs_base < min_gain:
            expert_assignments[f"expert_{e_idx}"] = {
                "expert_index": e_idx,
                "winning_arm": None,
                "domain_keys": matched_doms,
                "domain_gain_nats": 0.0,
                "action": "reverted_unhelpful_slot",
            }
            continue

        winning_arm_by_slot[e_idx] = best_arm
        arm_win_counts[best_arm] += 1
        if e_idx not in gen_slots:
            winning_gains.append(float(gain_vs_base))
        expert_assignments[f"expert_{e_idx}"] = {
            "expert_index": e_idx,
            "winning_arm": best_arm,
            "runner_up_arm": runner_arm,
            "domain_keys": matched_doms,
            "winning_arm_delta_nats": float(best_delta),
            "runner_up_delta_nats": float(runner_delta),
            "domain_gain_nats": float(gain_vs_base),
            "gain_over_runner_up_nats": float(gain_vs_runner),
            "action": "assigned",
        }

    composed: dict[int, dict[int, dict[str, Any]]] = {}
    max_leak_before = 0.0
    max_leak_after = 0.0
    projected_experts_set: set[int] = set()
    slot_leak_before: dict[int, float] = {e: 0.0 for e in ref_experts}
    slot_leak_after: dict[int, float] = {e: 0.0 for e in ref_experts}
    slot_projected: dict[int, bool] = {e: False for e in ref_experts}

    for l_idx in ref_layers:
        composed[l_idx] = {}
        for e_idx in ref_experts:
            assign_info = expert_assignments[f"expert_{e_idx}"]
            action = assign_info["action"]
            if action in ("ghost_sink_pruned", "reverted_unhelpful_slot"):
                src_spec = arm_deltas[arm_names[0]][l_idx][e_idx]
                if src_spec["kind"] == "lora":
                    new_t = {
                        k: (torch.zeros_like(v) if k.endswith("_B") else v.detach().clone())
                        for k, v in src_spec["tensors"].items()
                    }
                elif base_norm is not None:
                    new_t = {k: v.detach().clone() for k, v in base_norm[l_idx][e_idx]["tensors"].items()}
                else:
                    new_t = {k: torch.zeros_like(v) for k, v in src_spec["tensors"].items()}
                composed[l_idx][e_idx] = {
                    "tensors": new_t,
                    "kind": src_spec["kind"],
                    "style": src_spec["style"],
                }
                continue

            win_arm = winning_arm_by_slot[e_idx]
            src_spec = arm_deltas[win_arm][l_idx][e_idx]
            t_src = src_spec["tensors"]
            t_base = base_norm[l_idx][e_idx]["tensors"] if base_norm is not None else None
            kind = src_spec["kind"]

            if e_idx in prot_set or p_null_by_expert is None:
                composed[l_idx][e_idx] = {
                    "tensors": {k: v.detach().clone() for k, v in t_src.items()},
                    "kind": kind,
                    "style": src_spec["style"],
                }
                continue

            # Determine hidden & intermediate dims for projector lookup
            hidden_dim: int | None = None
            inter_dim: int | None = None
            if kind == "lora":
                hidden_dim = int(t_src["w1_A"].shape[1])
                inter_dim = int(t_src["w3_A"].shape[1])
            elif kind == "swiglu":
                hidden_dim = int(t_src["w1"].shape[1])
                inter_dim = int(t_src["w3"].shape[1])
            else:
                first_t = next(iter(t_src.values()))
                hidden_dim = int(first_t.shape[1]) if first_t.ndim >= 2 else int(first_t.shape[0])

            px, pact = _resolve_expert_nullspace_projectors(
                p_null_by_expert,
                l_idx,
                e_idx,
                hidden_size=hidden_dim,
                intermediate_size=inter_dim,
                eig_threshold=eig_thr,
            )
            if px is None and pact is None:
                composed[l_idx][e_idx] = {
                    "tensors": {k: v.detach().clone() for k, v in t_src.items()},
                    "kind": kind,
                    "style": src_spec["style"],
                }
                continue

            new_t = {}
            did_proj = False
            if kind == "lora":
                for proj in _SWIGLU_PROJECTIONS:
                    ak, bk = f"{proj}_A", f"{proj}_B"
                    b_b = t_base[bk] if t_base is not None else None
                    a_b = t_base[ak] if t_base is not None else None
                    if proj in ("w1", "w2"):
                        if px is not None:
                            bn, an, lb, la, chg = _project_lora_pair_in_nullspace(
                                b_b, a_b, t_src[bk], t_src[ak], px, "input", leak_tol
                            )
                            new_t[bk], new_t[ak] = bn, an
                            did_proj = did_proj or chg
                            slot_leak_before[e_idx] = max(slot_leak_before[e_idx], lb)
                            slot_leak_after[e_idx] = max(slot_leak_after[e_idx], la)
                        else:
                            new_t[bk], new_t[ak] = t_src[bk].clone(), t_src[ak].clone()
                    else:
                        p_w3, side_w3 = (pact, "input") if pact is not None else (px, "output")
                        assert p_w3 is not None
                        bn, an, lb, la, chg = _project_lora_pair_in_nullspace(
                            b_b, a_b, t_src[bk], t_src[ak], p_w3, side_w3, leak_tol
                        )
                        new_t[bk], new_t[ak] = bn, an
                        did_proj = did_proj or chg
                        slot_leak_before[e_idx] = max(slot_leak_before[e_idx], lb)
                        slot_leak_after[e_idx] = max(slot_leak_after[e_idx], la)
            elif kind == "swiglu":
                for proj in _SWIGLU_PROJECTIONS:
                    b_w = t_base[proj] if t_base is not None else None
                    if proj in ("w1", "w2") and px is not None:
                        wn, lb, la, chg = _project_dense_tensor_in_nullspace(
                            t_src[proj], px, "input", leak_tol, t_base=b_w
                        )
                        new_t[proj] = wn
                        did_proj = did_proj or chg
                        slot_leak_before[e_idx] = max(slot_leak_before[e_idx], lb)
                        slot_leak_after[e_idx] = max(slot_leak_after[e_idx], la)
                    elif proj == "w3" and (pact is not None or px is not None):
                        p_w3, side_w3 = (pact, "input") if pact is not None else (px, "output")
                        assert p_w3 is not None
                        wn, lb, la, chg = _project_dense_tensor_in_nullspace(
                            t_src[proj], p_w3, side_w3, leak_tol, t_base=b_w
                        )
                        new_t[proj] = wn
                        did_proj = did_proj or chg
                        slot_leak_before[e_idx] = max(slot_leak_before[e_idx], lb)
                        slot_leak_after[e_idx] = max(slot_leak_after[e_idx], la)
                    else:
                        new_t[proj] = t_src[proj].clone()
            else:
                for sk, t_val in t_src.items():
                    b_w = t_base[sk] if t_base is not None else None
                    if px is not None:
                        side = (
                            "input"
                            if (t_val.ndim == 1 or int(t_val.shape[1]) == int(px.shape[0]))
                            else "output"
                        )
                        wn, lb, la, chg = _project_dense_tensor_in_nullspace(
                            t_val, px, side, leak_tol, t_base=b_w
                        )
                        new_t[sk] = wn
                        did_proj = did_proj or chg
                        slot_leak_before[e_idx] = max(slot_leak_before[e_idx], lb)
                        slot_leak_after[e_idx] = max(slot_leak_after[e_idx], la)
                    else:
                        new_t[sk] = t_val.clone()

            if did_proj:
                projected_experts_set.add(e_idx)
                slot_projected[e_idx] = True
            max_leak_before = max(max_leak_before, slot_leak_before[e_idx])
            max_leak_after = max(max_leak_after, slot_leak_after[e_idx])
            composed[l_idx][e_idx] = {
                "tensors": new_t,
                "kind": kind,
                "style": src_spec["style"],
            }

    for e_idx in ref_experts:
        info = expert_assignments[f"expert_{e_idx}"]
        info["nullspace_leak_before"] = slot_leak_before[e_idx]
        info["nullspace_leak_after"] = slot_leak_after[e_idx]
        info["projected_into_p_null"] = slot_projected[e_idx]

    assert ref_meta is not None
    composed_out = _repackage_composed_experts(composed, ref_meta)
    report: dict[str, Any] = {
        "schema": COMPOSER_SCHEMA,
        "donor_arms": arm_names,
        "expert_assignments": expert_assignments,
        "arm_win_counts": arm_win_counts,
        "sink_experts": sorted(sink_set),
        "protected_experts": sorted(prot_set),
        "projected_experts": sorted(projected_experts_set),
        "max_nullspace_leak_before": max_leak_before,
        "max_nullspace_leak_after": max_leak_after,
        "mean_winning_domain_gain_nats": float(np.mean(winning_gains)) if winning_gains else 0.0,
    }
    return composed_out, report


# ---------------------------------------------------------------------------
# 4. Component (c): End-to-End Multi-Arm Composition & ANSE Ratchet Gate
# ---------------------------------------------------------------------------


def _normalize_checkpoint_to_blocks(
    ckpt: nn.Module | Mapping[str, Any],
    name: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize an ``nn.Module`` or block mapping into cloned finite block tensors."""
    if isinstance(ckpt, nn.Module):
        blocks = extract_checkpoint_blocks(ckpt)
    elif isinstance(ckpt, Mapping) and ckpt:
        blocks = {}
        for bid, sub in ckpt.items():
            if not isinstance(bid, str) or not bid.strip():
                raise ValueError(f"{name} block keys must be non-empty strings")
            if not isinstance(sub, Mapping) or not sub:
                raise TypeError(f"{name}[{bid!r}] must be a non-empty Mapping[str, Tensor]")
            blocks[bid] = {}
            for sk, t in sub.items():
                if not isinstance(t, torch.Tensor) or t.numel() == 0 or not t.is_floating_point():
                    raise ValueError(f"{name}[{bid!r}][{sk!r}] must be a non-empty floating-point Tensor")
                blocks[bid][sk] = t.detach().clone()
    else:
        raise TypeError(f"{name} must be an nn.Module or non-empty block Mapping")

    for bid, sub in blocks.items():
        for sk, t in sub.items():
            if not bool(torch.isfinite(t).all().item()):
                raise ValueError(f"{name}[{bid!r}][{sk!r}] contains non-finite values")
    return blocks


def _validate_multi_arm_layouts(
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    donor_blocks: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> None:
    """Verify identical block IDs, sub-keys, and tensor shapes across base and all donor arms."""
    ref_keys = set(base_blocks.keys())
    has_router = any(k.endswith(".router") for k in ref_keys)
    has_expert = any(".experts." in k for k in ref_keys)
    if not (has_router and has_expert):
        raise ValueError("Checkpoints must expose MoE '.router' and '.experts.' blocks")

    for arm_name, arm_blk in donor_blocks.items():
        if set(arm_blk.keys()) != ref_keys:
            raise ValueError(f"Donor arm {arm_name!r} block layout does not match base_checkpoint")
        for bid in sorted(ref_keys):
            b_sub, a_sub = base_blocks[bid], arm_blk[bid]
            if set(b_sub.keys()) != set(a_sub.keys()):
                raise ValueError(f"Sub-key mismatch on block {bid!r} for donor arm {arm_name!r}")
            for sk in b_sub:
                if b_sub[sk].shape != a_sub[sk].shape:
                    raise ValueError(f"Tensor shape mismatch on {bid}.{sk} for donor arm {arm_name!r}")


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    general_prefix: str,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunks accepting either ``'expert_id'`` or ``'domain'`` labels."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    eids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids' and 'expert_id'/'domain'")
        raw_eid = ch.get("expert_id", ch.get("domain"))
        if raw_eid is None or not str(raw_eid).strip():
            raise ValueError(f"{name}[{idx}] must contain a non-empty 'expert_id' or 'domain'")
        eid = str(raw_eid).strip()
        raw_ids = ch["ids"]
        if isinstance(raw_ids, (str, bytes)):
            raise TypeError(f"{name}[{idx}]['ids'] must be a sequence or 1D Tensor of ints")
        ids_t = torch.as_tensor(raw_ids, dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} rows must share sequence length {seq_len}")
        rows.append(ids_t)
        eids.append(eid)

    assert seq_len is not None
    if max(0, score_from - 1) >= seq_len - 1:
        raise ValueError(f"{name} sequence length {seq_len} too short for score_from={score_from}")
    ids_2d = torch.stack(rows, dim=0)
    is_gen = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_gen)) or not bool(np.any(~is_gen)):
        raise ValueError(f"{name} must include both general ({general_prefix}*) and specialist chunks")
    return ids_2d, is_gen, eids


@torch.no_grad()
def _forward_pre_ln(
    model: nn.Module,
    ids_2d: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Propagate token IDs through embeddings and all transformer blocks before final LayerNorm."""
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    xb = torch.cat(
        [
            compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size])
            for b0 in range(0, ids_2d.shape[0], batch_size)
        ],
        dim=0,
    )
    for blk in model.blocks:
        nx = []
        for b0 in range(0, xb.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, xb[b0 : b0 + batch_size], cos, sin)
            nx.append(forward_block_post_mlp(blk, xm, mi))
        xb = torch.cat(nx, dim=0)
    return xb


@torch.no_grad()
def _collect_prose_nullspace_by_layer(
    model: nn.Module,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    is_general: np.ndarray,
    cfg: MultiArmComposerConfig,
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect base-model prose hidden/SwiGLU states and synthesize per-layer ``P_0`` projectors."""
    apply_checkpoint_blocks_inplace(model, base_blocks)
    model.eval()
    cos = model.cos_cached.to(probe_ids.device)
    sin = model.sin_cached.to(probe_ids.device)
    xb = torch.cat(
        [
            compute_embeddings_batch(model, probe_ids[b0 : b0 + cfg.batch_size])
            for b0 in range(0, probe_ids.shape[0], cfg.batch_size)
        ],
        dim=0,
    )
    start_pos = max(0, cfg.score_from - 1)
    gen_mask = torch.as_tensor(is_general, dtype=torch.bool)
    p_null_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for l_idx, blk in enumerate(model.blocks):
        mi_l, nx_l = [], []
        for b0 in range(0, xb.shape[0], cfg.batch_size):
            xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + cfg.batch_size], cos, sin)
            mi_l.append(mi_b)
            nx_l.append(forward_block_post_mlp(blk, xm_b, mi_b))
        mlp_in = torch.cat(mi_l, dim=0)
        xb = torch.cat(nx_l, dim=0)

        x_p = mlp_in[:, start_pos:-1, :][gen_mask].reshape(-1, int(mlp_in.shape[-1])).contiguous()
        px, _ = compute_prose_nullspace_projector_scaled(
            x_p,
            sv_rel_tol=cfg.nullspace_sv_rel_tol,
            variance_threshold=cfg.nullspace_variance_threshold,
        )
        layer_p: dict[str, torch.Tensor] = {"x": px}
        base_m = getattr(blk.mlp, "base_mlp", blk.mlp)
        if hasattr(base_m, "w1") and hasattr(base_m, "w2"):
            w_dt = base_m.w1.weight.dtype
            xp_cast = x_p.to(dtype=w_dt)
            act = F.silu(base_m.w1(xp_cast).float()) * base_m.w2(xp_cast).float()
            pact, _ = compute_prose_nullspace_projector_scaled(
                act,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
            )
            layer_p["act"] = pact
        p_null_by_layer[l_idx] = layer_p
    return p_null_by_layer


@torch.no_grad()
def _evaluate_arm_state(
    *,
    model: nn.Module | None,
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
    eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    eval_ids: torch.Tensor | None,
    eval_eids: Sequence[str],
    cfg: MultiArmComposerConfig,
) -> dict[str, Any]:
    """Evaluate one MoE arm via custom ``eval_fn`` or exact causal mid-row NLL."""
    if model is not None:
        apply_checkpoint_blocks_inplace(model, blocks)
        model.eval()
        target: Any = model
    else:
        target = {bid: {sk: t.detach().clone() for sk, t in sub.items()} for bid, sub in blocks.items()}

    if eval_fn is not None:
        raw = eval_fn(target)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("eval_fn must return a non-empty mapping of domain -> NLL")
        if raw.get("non_finite_outputs") is True or raw.get("all_finite") is False:
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        domain_nlls: dict[str, float] = {}
        for k, v in raw.items():
            if k in ("non_finite_outputs", "all_finite"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"eval_fn[{k!r}] must be numeric, got {type(v).__name__}")
            fv = float(v)
            if not math.isfinite(fv):
                return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
            domain_nlls[str(k)] = fv
    else:
        assert model is not None and eval_ids is not None
        hp = _forward_pre_ln(model, eval_ids, cfg.batch_size)
        if not bool(torch.isfinite(hp).all().item()):
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        ln_mod = getattr(model, "ln", getattr(model, "norm", None))
        head_mod = getattr(model, "head", getattr(model, "lm_head", None))
        assert ln_mod is not None and head_mod is not None
        ln_b = getattr(ln_mod, "bias", None)
        if ln_b is None:
            ln_b = torch.zeros_like(ln_mod.weight)
        nlls = per_chunk_nll_from_pre_ln(
            hp,
            ln_mod.weight,
            ln_b,
            float(getattr(ln_mod, "eps", 1e-5)),
            head_mod.weight,
            eval_ids,
            score_from=cfg.score_from,
        )
        if not bool(torch.isfinite(nlls).all().item()):
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        by_dom: dict[str, list[float]] = {}
        for eid, val in zip(eval_eids, nlls.tolist(), strict=True):
            by_dom.setdefault(eid, []).append(float(val))
        domain_nlls = {dom: float(np.mean(vals)) for dom, vals in sorted(by_dom.items())}

    gen_vals = [v for k, v in domain_nlls.items() if k.startswith(cfg.general_prefix)]
    spec_vals = [v for k, v in domain_nlls.items() if not k.startswith(cfg.general_prefix)]
    if not gen_vals or not spec_vals:
        raise ValueError(
            f"Evaluation output must contain both general ({cfg.general_prefix}*) and specialist domains"
        )
    return {
        "all_finite": True,
        "domain_nlls": domain_nlls,
        "general_nll": float(np.mean(gen_vals)),
        "spec_nll": float(np.mean(spec_vals)),
        "nll_mean": float(np.mean(list(domain_nlls.values()))),
    }


def _score_arm_against_base(
    arm_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: MultiArmComposerConfig,
) -> dict[str, Any]:
    """Compute general/worst drift ratios, specialist gain, and ANSE Energy relative to base."""
    if not arm_eval["all_finite"] or set(arm_eval["domain_nlls"].keys()) != set(base_eval["domain_nlls"].keys()):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "general_guard_ok": False,
            "worst_guard_ok": False,
            "guard_ok": False,
            "domain_gain_nats": 0.0,
            "mean_diff_vs_base": float("inf"),
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = arm_eval["domain_nlls"]
    gen_ratio = float(arm_eval["general_nll"]) / max(float(base_eval["general_nll"]), 1e-8)
    worst_ratio = max(float(cand_doms[d]) / max(float(base_doms[d]), 1e-8) for d in base_doms)
    mean_diff = float(arm_eval["nll_mean"]) - float(base_eval["nll_mean"])
    dom_gain = float(base_eval["spec_nll"]) - float(arm_eval["spec_nll"])

    gen_ok = gen_ratio <= cfg.max_general_drift_ratio + 1e-12
    worst_ok = worst_ratio <= cfg.max_worst_domain_drift_ratio + 1e-12
    guard_ok = gen_ok and (worst_ok if cfg.enforce_all_domain_guard else True)
    if not guard_ok:
        anse_e = cfg.maximum_pain_energy + max(0.0, mean_diff)
    else:
        soft_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0
        anse_e = mean_diff + soft_pen

    return {
        "all_finite": True,
        "general_drift_ratio": gen_ratio,
        "worst_domain_ratio": worst_ratio,
        "general_guard_ok": gen_ok,
        "worst_guard_ok": worst_ok,
        "guard_ok": guard_ok,
        "domain_gain_nats": dom_gain,
        "mean_diff_vs_base": mean_diff,
        "anse_energy": float(anse_e),
    }


def _resolve_router_blend_weights(
    arm_names: Sequence[str],
    arm_win_counts: Mapping[str, int],
    explicit_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Compute normalized non-negative blending weights across donor arms."""
    if explicit_weights is not None:
        if not isinstance(explicit_weights, Mapping) or set(explicit_weights.keys()) != set(arm_names):
            raise ValueError("router_blend_weights must provide one weight for every donor arm")
        raw = {
            arm: _validate_nonneg_finite(explicit_weights[arm], f"router_blend_weights[{arm!r}]")
            for arm in arm_names
        }
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("router_blend_weights must sum to > 0.0")
        return {arm: val / total for arm, val in raw.items()}

    total_wins = sum(int(arm_win_counts.get(arm, 0)) for arm in arm_names)
    if total_wins > 0:
        return {arm: float(arm_win_counts.get(arm, 0)) / float(total_wins) for arm in arm_names}
    uni = 1.0 / float(len(arm_names))
    return {arm: uni for arm in arm_names}


def _resolve_composer_config(
    config: MultiArmComposerConfig | None,
    inplace: bool | None,
) -> MultiArmComposerConfig:
    """Resolve optional ``MultiArmComposerConfig`` and ``inplace`` override."""
    if config is not None and not isinstance(config, MultiArmComposerConfig):
        raise TypeError(f"config must be MultiArmComposerConfig or None, got {type(config).__name__}")
    cfg_dict = asdict(config) if config is not None else asdict(MultiArmComposerConfig())
    if inplace is not None:
        cfg_dict["inplace"] = _validate_bool(inplace, "inplace")
    return MultiArmComposerConfig(**cfg_dict)


@torch.no_grad()
def execute_verified_multi_arm_moe_composition(
    base_checkpoint: nn.Module | Mapping[str, Any],
    donor_checkpoints: Mapping[str, nn.Module | Mapping[str, Any]],
    *,
    target_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    probe_domain_losses_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
    base_probe_domain_losses: Mapping[str, float] | None = None,
    p_null_by_expert: torch.Tensor | Mapping[Any, Any] | None = None,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    expert_domain_map: Mapping[int, str | Sequence[str]] | None = None,
    sink_experts: Sequence[int] | set[int] = (),
    protected_experts: Sequence[int] | set[int] = (),
    router_blend_weights: Mapping[str, float] | None = None,
    config: MultiArmComposerConfig | None = None,
    inplace: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compose domain-winning specialists + calibrated LayerNorm across arms under ANSE ratification.

    Commits the composed checkpoint if and only if:
    1. All outputs are finite and clear general-prose + worst-domain guards (``<= +1.0%``),
    2. ``ANSE Delta E < 0`` relative to ``base_checkpoint``, AND
    3. The composed MoE strictly beats EVERY individual donor arm on ANSE energy
       (and beats every guard-feasible donor arm on ``nll_mean``).

    Otherwise triggers ANSE Maximum Pain atomic bitwise rollback to ``rollback_target`` and
    returns a canonical SHA-256 provenance receipt.
    """
    cfg = _resolve_composer_config(config, inplace)
    if not isinstance(donor_checkpoints, Mapping) or len(donor_checkpoints) < 2:
        raise ValueError("donor_checkpoints must be a mapping containing at least 2 donor arms")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    if target_model is not None and not isinstance(target_model, nn.Module):
        raise TypeError(f"target_model must be an nn.Module or None, got {type(target_model).__name__}")

    arm_names = [_validate_nonempty_str(k, "donor_arm_name") for k in donor_checkpoints.keys()]
    if cfg.rollback_target not in ("base", "best_feasible_donor", *arm_names):
        raise ValueError(
            f"rollback_target {cfg.rollback_target!r} must be 'base', 'best_feasible_donor', or a donor arm name"
        )

    base_blocks = _normalize_checkpoint_to_blocks(base_checkpoint, "base_checkpoint")
    donor_blocks = {
        arm: _normalize_checkpoint_to_blocks(donor_checkpoints[arm], f"donor_checkpoints[{arm!r}]")
        for arm in arm_names
    }
    _validate_multi_arm_layouts(base_blocks, donor_blocks)

    work_model: nn.Module | None = None
    if target_model is not None:
        work_model = target_model if cfg.inplace else copy.deepcopy(target_model)
    elif isinstance(base_checkpoint, nn.Module):
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)
    else:
        for arm in arm_names:
            if isinstance(donor_checkpoints[arm], nn.Module):
                work_model = donor_checkpoints[arm] if cfg.inplace else copy.deepcopy(donor_checkpoints[arm])
                break

    if work_model is None and eval_fn is None:
        raise ValueError("Either an nn.Module checkpoint/target_model or eval_fn must be provided")

    head_mod = getattr(work_model, "head", getattr(work_model, "lm_head", None)) if work_model is not None else None
    vocab_size = int(head_mod.weight.shape[0]) if head_mod is not None else 0

    probe_ids: torch.Tensor | None = None
    probe_is_gen: np.ndarray | None = None
    probe_eids: list[str] = []
    if probe_chunks is not None and work_model is not None:
        probe_ids, probe_is_gen, probe_eids = _validate_chunks(
            probe_chunks,
            vocab_size=vocab_size,
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            name="probe_chunks",
        )

    eval_ids: torch.Tensor | None = None
    eval_eids: list[str] = []
    eval_is_gen: np.ndarray | None = None
    if eval_chunks is not None and work_model is not None:
        eval_ids, eval_is_gen, eval_eids = _validate_chunks(
            eval_chunks,
            vocab_size=vocab_size,
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            name="eval_chunks",
        )
    elif probe_ids is not None:
        eval_ids, eval_is_gen, eval_eids = probe_ids, probe_is_gen, probe_eids
    elif eval_fn is None:
        raise ValueError("Either probe_chunks/eval_chunks or eval_fn must be provided")

    base_sha = compute_checkpoint_sha256(base_blocks)
    donor_shas = {arm: compute_checkpoint_sha256(blk) for arm, blk in donor_blocks.items()}
    was_training = work_model.training if work_model is not None else False
    if work_model is not None:
        work_model.eval()

    try:
        # 1. Resolve P_0 prose null-space projectors
        resolved_p_null: Any = p_null_by_expert
        if resolved_p_null is None and cfg.auto_project_p_null:
            if prose_hidden_states is not None:
                if isinstance(prose_hidden_states, torch.Tensor):
                    px_h, _ = compute_prose_nullspace_projector_scaled(
                        prose_hidden_states,
                        sv_rel_tol=cfg.nullspace_sv_rel_tol,
                        variance_threshold=cfg.nullspace_variance_threshold,
                    )
                    resolved_p_null = px_h
                elif isinstance(prose_hidden_states, Mapping):
                    resolved_p_null = {
                        k: compute_prose_nullspace_projector_scaled(
                            v,
                            sv_rel_tol=cfg.nullspace_sv_rel_tol,
                            variance_threshold=cfg.nullspace_variance_threshold,
                        )[0]
                        for k, v in prose_hidden_states.items()
                    }
                elif isinstance(prose_hidden_states, Sequence):
                    resolved_p_null = {
                        idx: compute_prose_nullspace_projector_scaled(
                            v,
                            sv_rel_tol=cfg.nullspace_sv_rel_tol,
                            variance_threshold=cfg.nullspace_variance_threshold,
                        )[0]
                        for idx, v in enumerate(prose_hidden_states)
                    }
            elif work_model is not None and probe_ids is not None and probe_is_gen is not None:
                resolved_p_null = _collect_prose_nullspace_by_layer(
                    work_model, base_blocks, probe_ids, probe_is_gen, cfg
                )

        # 2. Evaluate base & donors on probe split (or eval_fn) when probe_domain_losses_by_arm omitted
        base_probe_eval: dict[str, Any] | None = None
        if probe_domain_losses_by_arm is None:
            p_ids = probe_ids if probe_ids is not None else eval_ids
            p_eids = probe_eids if probe_eids else eval_eids
            base_probe_eval = _evaluate_arm_state(
                model=work_model, blocks=base_blocks, eval_fn=eval_fn, eval_ids=p_ids, eval_eids=p_eids, cfg=cfg
            )
            if not base_probe_eval["all_finite"]:
                raise ValueError("base_checkpoint produced non-finite probe evaluation NLL")
            resolved_base_probe_losses: Mapping[str, float] | None = base_probe_eval["domain_nlls"]
            resolved_probe_losses: dict[str, dict[str, float]] = {}
            for arm in arm_names:
                d_ev = _evaluate_arm_state(
                    model=work_model,
                    blocks=donor_blocks[arm],
                    eval_fn=eval_fn,
                    eval_ids=p_ids,
                    eval_eids=p_eids,
                    cfg=cfg,
                )
                if not d_ev["all_finite"]:
                    raise ValueError(f"Donor arm {arm!r} produced non-finite probe evaluation NLL")
                resolved_probe_losses[arm] = dict(d_ev["domain_nlls"])
        else:
            resolved_probe_losses = {arm: dict(probe_domain_losses_by_arm[arm]) for arm in arm_names}
            resolved_base_probe_losses = base_probe_domain_losses

        # 3. Component (b): Select domain-winning experts across arms & project out P_0 leakage
        arms_expert_blocks = {
            arm: {k: v for k, v in donor_blocks[arm].items() if _BLOCK_EXPERT_RE.match(k) is not None}
            for arm in arm_names
        }
        base_expert_blocks = {
            k: v for k, v in base_blocks.items() if _BLOCK_EXPERT_RE.match(k) is not None
        }
        composed_expert_blocks, selection_report = select_domain_winning_experts_across_arms(
            arms_expert_blocks,
            resolved_probe_losses,
            resolved_p_null,
            base_domain_losses=resolved_base_probe_losses,
            base_experts=base_expert_blocks,
            expert_domain_map=expert_domain_map,
            sink_experts=sink_experts,
            protected_experts=protected_experts,
            general_prefix=cfg.general_prefix,
            nullspace_eig_threshold=cfg.nullspace_eig_threshold,
            nullspace_leak_tol=cfg.nullspace_leak_tol,
            min_domain_gain_nats=cfg.min_domain_gain_nats,
            revert_harmful_to_zero=cfg.revert_harmful_to_zero,
        )

        # 4. Blend router gates (with P_0 projection of router deltas when P_x is available)
        blend_w = _resolve_router_blend_weights(
            arm_names, selection_report["arm_win_counts"], router_blend_weights
        )
        cand_blocks: dict[str, dict[str, torch.Tensor]] = {
            bid: {sk: t.detach().clone() for sk, t in sub.items()}
            for bid, sub in base_blocks.items()
        }
        cand_blocks.update(composed_expert_blocks)

        for bid in sorted(base_blocks.keys()):
            if bid.endswith(".router"):
                l_idx = _parse_layer_key(bid.split(".mlp.router", 1)[0]) or 0
                px_r: torch.Tensor | None = None
                if resolved_p_null is not None:
                    px_r, _ = _resolve_expert_nullspace_projectors(
                        resolved_p_null,
                        l_idx,
                        0,
                        hidden_size=int(base_blocks[bid]["weight"].shape[1]),
                        intermediate_size=None,
                        eig_threshold=cfg.nullspace_eig_threshold,
                    )
                for sk, b_ten in base_blocks[bid].items():
                    b64 = b_ten.detach().to("cpu", dtype=torch.float64)
                    acc = torch.zeros_like(b64)
                    for arm in arm_names:
                        acc = acc + blend_w[arm] * (donor_blocks[arm][bid][sk].detach().to("cpu", dtype=torch.float64) - b64)
                    if px_r is not None and acc.ndim == 2 and int(acc.shape[1]) == int(px_r.shape[0]):
                        acc = acc @ px_r
                    cand_blocks[bid][sk] = (b64 + acc).to(device=b_ten.device, dtype=b_ten.dtype)

        # 5. Component (a): Fast 2-matmul affine LayerNorm + temperature calibration
        ln_cal_summary: dict[str, Any] = {
            "executed": False,
            "matmul_speedup_factor": 1.0,
            "lambda_ln": cfg.fold_ln_lambda if cfg.fold_ln_lambda is not None else 1.0,
            "tau_temp": cfg.fold_tau_temp if cfg.fold_tau_temp is not None else 1.0,
        }
        if "final_ln" in cand_blocks:
            for sk, b_ten in base_blocks["final_ln"].items():
                b64 = b_ten.detach().to("cpu", dtype=torch.float64)
                acc = torch.zeros_like(b64)
                for arm in arm_names:
                    acc = acc + blend_w[arm] * (donor_blocks[arm]["final_ln"][sk].detach().to("cpu", dtype=torch.float64) - b64)
                cand_blocks["final_ln"][sk] = (b64 + acc).to(device=b_ten.device, dtype=b_ten.dtype)

            if cfg.fold_ln_lambda is not None:
                lam_f = float(cfg.fold_ln_lambda)
                tau_f = float(cfg.fold_tau_temp) if cfg.fold_tau_temp is not None else 1.0
                w_base = base_blocks["final_ln"]["weight"]
                w_cand = cand_blocks["final_ln"]["weight"]
                b_base = base_blocks["final_ln"].get("bias")
                b_cand = cand_blocks["final_ln"].get("bias")
                w_int = (w_base.float() + lam_f * (w_cand.float() - w_base.float())).to(w_base.dtype)
                b_int = (
                    (b_base.float() + lam_f * (b_cand.float() - b_base.float())).to(b_base.dtype)
                    if (b_base is not None and b_cand is not None)
                    else None
                )
                fw, fb = fold_temperature_into_norm_params(w_int, b_int, tau_f)
                cand_blocks["final_ln"]["weight"] = fw
                if fb is not None:
                    cand_blocks["final_ln"]["bias"] = fb
            elif work_model is not None and (probe_ids is not None or eval_ids is not None):
                cal_ids = probe_ids if probe_ids is not None else eval_ids
                cal_gen = probe_is_gen if probe_is_gen is not None else eval_is_gen
                assert cal_ids is not None and cal_gen is not None
                start_pos = max(0, cfg.score_from - 1)
                apply_checkpoint_blocks_inplace(work_model, cand_blocks)
                hp_cand = _forward_pre_ln(work_model, cal_ids, cfg.batch_size)
                sub_h = hp_cand[:, start_pos:-1, :]
                sub_t = cal_ids[:, start_pos + 1 :]
                ln_mod = getattr(work_model, "ln", getattr(work_model, "norm", None))
                assert ln_mod is not None and head_mod is not None
                ln_eps = float(getattr(ln_mod, "eps", 1e-5))
                cal_res = fast_affine_layernorm_temperature_grid(
                    sub_h,
                    base_blocks["final_ln"]["weight"],
                    base_blocks["final_ln"].get("bias"),
                    cand_blocks["final_ln"]["weight"],
                    cand_blocks["final_ln"].get("bias"),
                    head_mod.weight,
                    sub_t,
                    cal_gen,
                    eps=ln_eps,
                    lambda_grid=cfg.lambda_grid,
                    tau_grid=cfg.tau_grid,
                    max_general_drift_rel=cfg.calibrator_max_general_drift_rel,
                    seq_batch_size=cfg.batch_size,
                    base_general_nll_override=(
                        base_probe_eval["general_nll"] if base_probe_eval is not None else None
                    ),
                    base_total_nll_override=(
                        base_probe_eval["nll_mean"] if base_probe_eval is not None else None
                    ),
                )
                cand_blocks["final_ln"]["weight"] = cal_res["folded_ln_weight"]
                if cal_res["folded_ln_bias"] is not None:
                    cand_blocks["final_ln"]["bias"] = cal_res["folded_ln_bias"]
                ln_cal_summary = {
                    "executed": True,
                    "lambda_ln": cal_res["best_lambda_ln"],
                    "tau_temp": cal_res["best_tau_temp"],
                    "guard_feasible": cal_res["guard_feasible"],
                    "head_matmuls_executed": cal_res["head_matmuls_executed"],
                    "naive_head_matmuls": cal_res["naive_head_matmuls"],
                    "matmul_speedup_factor": cal_res["matmul_speedup_factor"],
                    "best_general_drift_rel": cal_res["best_general_drift_rel"],
                }

        cand_sha = compute_checkpoint_sha256(cand_blocks)

        # 6. Evaluate base, every donor arm, and candidate on eval split under ANSE ratchet
        base_eval = _evaluate_arm_state(
            model=work_model, blocks=base_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        if not base_eval["all_finite"]:
            raise ValueError("base_checkpoint produced non-finite evaluation NLL")

        donor_evals: dict[str, dict[str, Any]] = {}
        donor_metrics: dict[str, dict[str, Any]] = {}
        for arm in arm_names:
            ev_a = _evaluate_arm_state(
                model=work_model,
                blocks=donor_blocks[arm],
                eval_fn=eval_fn,
                eval_ids=eval_ids,
                eval_eids=eval_eids,
                cfg=cfg,
            )
            donor_evals[arm] = ev_a
            donor_metrics[arm] = _score_arm_against_base(ev_a, base_eval, cfg)

        cand_eval = _evaluate_arm_state(
            model=work_model, blocks=cand_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        cand_metrics = _score_arm_against_base(cand_eval, base_eval, cfg)

        cand_anse_e = float(cand_metrics["anse_energy"])
        delta_e_vs_base = cand_anse_e
        delta_e_vs_donors = {arm: cand_anse_e - float(donor_metrics[arm]["anse_energy"]) for arm in arm_names}

        beats_all_donors = all(
            delta_e_vs_donors[arm] < -cfg.min_energy_improvement
            and (
                not donor_metrics[arm]["guard_ok"]
                or float(cand_eval["nll_mean"]) < float(donor_evals[arm]["nll_mean"])
            )
            for arm in arm_names
        )
        accepted = bool(
            cand_eval["all_finite"]
            and cand_metrics["guard_ok"]
            and (delta_e_vs_base < -cfg.min_energy_improvement)
            and (not cfg.require_beat_all_donors or beats_all_donors)
        )

        # 7. Atomic commit or bitwise rollback
        if accepted:
            status = "MULTI_ARM_COMPOSITION_COMMITTED"
            rolled_back = False
            rollback_verified = False
            final_blocks = cand_blocks
            final_eval = cand_eval
            final_metrics = cand_metrics
            expected_final_sha = cand_sha
            actual_rollback_target = "none"
        else:
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            rolled_back = True
            if cfg.rollback_target == "best_feasible_donor":
                feasible_donors = [arm for arm in arm_names if donor_metrics[arm]["guard_ok"]]
                if feasible_donors:
                    best_d = min(feasible_donors, key=lambda a: float(donor_metrics[a]["anse_energy"]))
                    final_blocks = donor_blocks[best_d]
                    final_eval = donor_evals[best_d]
                    final_metrics = donor_metrics[best_d]
                    expected_final_sha = donor_shas[best_d]
                    actual_rollback_target = best_d
                else:
                    final_blocks = base_blocks
                    final_eval = base_eval
                    final_metrics = _score_arm_against_base(base_eval, base_eval, cfg)
                    expected_final_sha = base_sha
                    actual_rollback_target = "base"
            elif cfg.rollback_target in donor_blocks:
                t_arm = cfg.rollback_target
                final_blocks = donor_blocks[t_arm]
                final_eval = donor_evals[t_arm]
                final_metrics = donor_metrics[t_arm]
                expected_final_sha = donor_shas[t_arm]
                actual_rollback_target = t_arm
            else:
                final_blocks = base_blocks
                final_eval = base_eval
                final_metrics = _score_arm_against_base(base_eval, base_eval, cfg)
                expected_final_sha = base_sha
                actual_rollback_target = "base"

        if work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, final_blocks)
            extracted = extract_checkpoint_blocks(work_model)
            scoped = {bid: extracted[bid] for bid in final_blocks if bid in extracted}
            final_sha = compute_checkpoint_sha256(scoped)
        else:
            final_sha = compute_checkpoint_sha256(final_blocks)

        if rolled_back:
            rollback_verified = final_sha == expected_final_sha
            if not rollback_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {expected_final_sha[:12]}, got {final_sha[:12]}"
                )

        donor_summary = {
            arm: {
                "nll_mean": donor_evals[arm]["nll_mean"],
                "anse_energy": float(donor_metrics[arm]["anse_energy"]),
                "general_drift_ratio": (
                    float(donor_metrics[arm]["general_drift_ratio"])
                    if math.isfinite(float(donor_metrics[arm]["general_drift_ratio"]))
                    else None
                ),
                "worst_domain_ratio": (
                    float(donor_metrics[arm]["worst_domain_ratio"])
                    if math.isfinite(float(donor_metrics[arm]["worst_domain_ratio"]))
                    else None
                ),
                "domain_gain_nats": float(donor_metrics[arm]["domain_gain_nats"]),
                "guard_ok": bool(donor_metrics[arm]["guard_ok"]),
            }
            for arm in arm_names
        }
        cand_gen_r = float(cand_metrics["general_drift_ratio"])
        final_gen_r = float(final_metrics["general_drift_ratio"])

        receipt: dict[str, Any] = {
            "object": COMPOSER_RECEIPT_OBJECT,
            "schema": COMPOSER_SCHEMA,
            "status": status,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "rollback_target": actual_rollback_target,
            "rollback_bitwise_verified": rollback_verified,
            "nll_mean": final_eval["nll_mean"],
            "domain_gain_nats": float(final_metrics["domain_gain_nats"]),
            "matmul_speedup_factor": float(ln_cal_summary["matmul_speedup_factor"]),
            "anse_energy": float(final_metrics["anse_energy"]),
            "base_nll_mean": float(base_eval["nll_mean"]),
            "cand_nll_mean": cand_eval["nll_mean"],
            "cand_anse_energy": cand_anse_e,
            "cand_general_drift_ratio": cand_gen_r if math.isfinite(cand_gen_r) else None,
            "cand_guard_ok": bool(cand_metrics["guard_ok"]),
            "final_general_drift_ratio": final_gen_r if math.isfinite(final_gen_r) else None,
            "delta_e_vs_base": delta_e_vs_base,
            "delta_e_vs_donors": delta_e_vs_donors,
            "beats_all_donors": beats_all_donors,
            "router_blend_weights": blend_w,
            "donor_arm_metrics": donor_summary,
            "selection_summary": {
                "arm_win_counts": selection_report["arm_win_counts"],
                "projected_experts": selection_report["projected_experts"],
                "sink_experts": selection_report["sink_experts"],
                "max_nullspace_leak_before": selection_report["max_nullspace_leak_before"],
                "max_nullspace_leak_after": selection_report["max_nullspace_leak_after"],
                "mean_winning_domain_gain_nats": selection_report["mean_winning_domain_gain_nats"],
            },
            "ln_calibration_summary": ln_cal_summary,
            "base_state_sha256": base_sha,
            "donor_state_sha256": donor_shas,
            "candidate_state_sha256": cand_sha,
            "final_state_sha256": final_sha,
            "config": asdict(cfg),
        }
        receipt["receipt_sha256"] = compute_payload_sha256(receipt)
        return (work_model if work_model is not None else final_blocks), receipt
    finally:
        if work_model is not None:
            work_model.train(was_training)


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "COMPOSER_RECEIPT_OBJECT",
    "COMPOSER_SCHEMA",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_CALIBRATOR_GENERAL_GUARD_REL",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_TAU_GRID",
    "GENERAL_EXPERT_PREFIX",
    "MultiArmComposerConfig",
    "execute_verified_multi_arm_moe_composition",
    "fast_affine_layernorm_temperature_grid",
    "select_domain_winning_experts_across_arms",
    "verify_composer_receipt_provenance",
]
