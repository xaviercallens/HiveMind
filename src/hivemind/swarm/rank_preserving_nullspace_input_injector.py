# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Rank-Preserving Prose-Nullspace LoRA Input Injector & Unfolded Safe-Tau Calibrator.

Implements training-free, CPU-only multi-campaign Weight-Space Programming (WSP)
grounded in ``PILOT6`` on 3B ``CrowdLoRAMoEMLP`` (where preserving the co-adapted
``(router, B_expert)`` pair from ``WSP_S4_multigate_only`` and injecting ``T7`` and
``C3_A`` donor features exclusively on the input factor
``A_new = A_anchor + sum_k alpha_k * (A_donor_k - A_anchor) @ P_0`` preserved exact
LoRA rank ``r`` with ``2.86e-6`` float32 prose invariance and ``-0.5147`` nats on
``confirm_skip760`` at ``E0 = +0.406%``, while un-folding prior ``tau_old < 1.0``
sharpening and recalibrating over ``tau >= 1.0`` eliminated ``E0`` boundary drift):

1. ``inject_donor_lora_via_input_nullspace(a_anchor, b_anchor, donor_a_list, alphas, p_null_input)``:
   Computes ``A_new = A_anchor + sum_k alpha_k * (A_donor_k - A_anchor) @ P_0``
   without modifying ``B_anchor`` or LoRA rank ``r`` (zero SVD truncation error),
   while proving ``||B_anchor @ (A_new - A_anchor) @ x_prose||_inf <= eps``.
2. ``unfold_and_recalibrate_safe_layernorm_temperature(...)``:
   Un-multiplies prior sharpening ``prior_tau`` from folded final LayerNorm
   weights/biases and recalibrates ``(lambda_ln, tau_temp)`` strictly over safe
   ``tau >= 1.0`` under a strict general-prose ratio ceiling (``1.002`` default).
3. ``execute_verified_multicampaign_nullspace_injection(...)``:
   Runs multi-campaign co-adapted input-nullspace injection + unfolded safe-tau
   LayerNorm recalibration under ``ANSE Delta E < 0`` and the ``<= +1.0%`` prose
   guard with atomic bitwise rollback and a canonical SHA-256 provenance receipt.
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
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP
from hivemind.swarm.domain_matched_moe_arm_composer import (
    fast_affine_layernorm_temperature_grid,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    fold_temperature_into_norm_params,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.tangent_task_vector_surgeon import (
    compute_prose_nullspace_projector_scaled,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

INJECTOR_SCHEMA: str = "hivemind.rank_preserving_nullspace_input_injector/1"
INJECTOR_RECEIPT_OBJECT: str = "hivemind.multicampaign_nullspace_injection_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_PROSE_INVARIANCE_EPS: float = 1e-4
DEFAULT_SAFE_TAU_GRID: tuple[float, ...] = (1.0, 1.02, 1.05)
DEFAULT_SAFE_MAX_GENERAL_RATIO: float = 1.002
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_PROJECTOR_EIG_TOL: float = 1e-2
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-5
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float | None = None
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(0, 21))
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_IGNORE_INDEX: int = -100

VALID_INJECT_PROJECTIONS: frozenset[str] = frozenset({"w1", "w2", "w3"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_EXPERT_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)$")
_LAYER_KEY_RE = re.compile(r"^(?:blocks\.|layers\.|layer_)?(\d+)(?:\.|$)")
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


def _validate_safe_tau_grid(
    tau_grid: Sequence[float],
    name: str = "tau_grid",
) -> tuple[float, ...]:
    """Validate a non-empty temperature grid strictly restricted to ``tau >= 1.0``."""
    if isinstance(tau_grid, (str, bytes, bytearray)) or not isinstance(tau_grid, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of floats >= 1.0")
    if len(tau_grid) == 0:
        raise ValueError(f"{name} must not be empty")
    vals: list[float] = []
    for idx, item in enumerate(tau_grid):
        tau = _validate_positive_finite(item, f"{name}[{idx}]")
        if tau < 1.0:
            raise ValueError(f"Safe calibration requires tau >= 1.0, got {name}[{idx}]={tau}")
        vals.append(tau)
    return tuple(sorted(set(vals)))


def _validate_lambda_grid(
    lambda_grid: Sequence[float],
    name: str = "lambda_grid",
) -> tuple[float, ...]:
    """Validate interpolation weights in ``[0.0, 1.0]``."""
    if isinstance(lambda_grid, (str, bytes, bytearray)) or not isinstance(lambda_grid, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of floats in [0.0, 1.0]")
    if len(lambda_grid) == 0:
        raise ValueError(f"{name} must not be empty")
    return tuple(
        dict.fromkeys(
            _validate_unit_interval(item, f"{name}[{idx}]", allow_zero=True)
            for idx, item in enumerate(lambda_grid)
        )
    )


def _validate_projector_matrix(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
    eig_tol: float = DEFAULT_PROJECTOR_EIG_TOL,
    reorthogonalize: bool = False,
) -> tuple[torch.Tensor, float]:
    """Validate a symmetric orthogonal null-space projector ``P_0`` in CPU ``float64``."""
    t = _validate_fp_tensor(proj, name, exact_ndim=2)
    if t.shape[0] != t.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(t.shape)}")
    if expected_dim is not None and int(t.shape[0]) != expected_dim:
        raise ValueError(f"{name} shape {tuple(t.shape)} must match ({expected_dim}, {expected_dim})")
    tol = _validate_nonneg_finite(eig_tol, "projector_eig_tol")
    mat = t.detach().to(device="cpu", dtype=torch.float64).contiguous()
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > max(1e-4, tol) * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    sym = (0.5 * (mat + mat.T)).contiguous()
    evals, evecs = torch.linalg.eigh(sym)
    min_ev = float(evals.min().item())
    max_ev = float(evals.max().item())
    if min_ev < -tol or max_ev > 1.0 + tol:
        raise ValueError(
            f"{name} eigenvalues [{min_ev:.3e}, {max_ev:.3e}] fall outside [-{tol:.2e}, 1+{tol:.2e}]"
        )
    idem_err = float((sym @ sym - sym).abs().max().item())
    if reorthogonalize:
        keep = evals > 0.5
        if bool(keep.any().item()):
            q_keep = evecs[:, keep]
            sym = (0.5 * ((q_keep @ q_keep.T) + (q_keep @ q_keep.T).T)).contiguous()
        else:
            sym = torch.zeros_like(sym)
        idem_err = float((sym @ sym - sym).abs().max().item())
    return sym, idem_err


def _validate_expert_index_set(indices: Any, name: str) -> set[int]:
    """Validate a sequence or set of non-negative integer expert indices."""
    if isinstance(indices, (str, bytes, bytearray)) or not isinstance(indices, (Sequence, set, frozenset)):
        raise TypeError(f"{name} must be a sequence or set of non-negative integers")
    return {_validate_nonneg_int(item, f"{name} element") for item in indices}


def verify_nullspace_injection_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != INJECTOR_RECEIPT_OBJECT or receipt.get("schema") != INJECTOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class NullspaceInputInjectionConfig:
    """Validated configuration governing rank-preserving nullspace input injection."""

    default_alpha: float = 0.15
    prose_invariance_eps: float = DEFAULT_PROSE_INVARIANCE_EPS
    enforce_prose_invariance: bool = True
    reorthogonalize_projector: bool = False
    projector_eig_tol: float = DEFAULT_PROJECTOR_EIG_TOL
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_max_active_modes: int | None = None
    inject_projections: tuple[str, ...] = ("w1", "w2")
    skip_zero_b_sinks: bool = True
    unfold_and_recalibrate_safe_ln: bool = True
    safe_tau_grid: tuple[float, ...] = DEFAULT_SAFE_TAU_GRID
    safe_lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    calibrator_max_general_ratio: float = DEFAULT_SAFE_MAX_GENERAL_RATIO
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    require_beat_anchor: bool = True
    min_energy_improvement: float = 0.0
    rollback_target: str = "anchor"
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_finite_float(self.default_alpha, "default_alpha")
        _validate_nonneg_finite(self.prose_invariance_eps, "prose_invariance_eps")
        _validate_bool(self.enforce_prose_invariance, "enforce_prose_invariance")
        _validate_bool(self.reorthogonalize_projector, "reorthogonalize_projector")
        _validate_nonneg_finite(self.projector_eig_tol, "projector_eig_tol")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if self.nullspace_variance_threshold is not None:
            _validate_unit_interval(
                self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
            )
        if self.nullspace_max_active_modes is not None:
            _validate_positive_int(self.nullspace_max_active_modes, "nullspace_max_active_modes")
        if (
            isinstance(self.inject_projections, (str, bytes))
            or not isinstance(self.inject_projections, Sequence)
            or not self.inject_projections
        ):
            raise ValueError("inject_projections must be a non-empty sequence of projection names")
        cleaned_proj = tuple(dict.fromkeys(str(p).strip() for p in self.inject_projections))
        if any(p not in VALID_INJECT_PROJECTIONS for p in cleaned_proj):
            raise ValueError(
                f"inject_projections entries must lie in {sorted(VALID_INJECT_PROJECTIONS)}, got {cleaned_proj}"
            )
        object.__setattr__(self, "inject_projections", cleaned_proj)
        _validate_bool(self.skip_zero_b_sinks, "skip_zero_b_sinks")
        _validate_bool(self.unfold_and_recalibrate_safe_ln, "unfold_and_recalibrate_safe_ln")
        object.__setattr__(self, "safe_tau_grid", _validate_safe_tau_grid(self.safe_tau_grid, "safe_tau_grid"))
        object.__setattr__(self, "safe_lambda_grid", _validate_lambda_grid(self.safe_lambda_grid, "safe_lambda_grid"))
        cal_r = _validate_positive_finite(self.calibrator_max_general_ratio, "calibrator_max_general_ratio")
        max_gen = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_worst = _validate_positive_finite(
            self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio"
        )
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if cal_r < 1.0 or max_gen < 1.0 or max_worst < 1.0 or safe_r < 1.0:
            raise ValueError("Drift ratio bounds must be >= 1.0")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.require_beat_anchor, "require_beat_anchor")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_nonempty_str(self.rollback_target, "rollback_target")
        _validate_nonempty_str(self.general_prefix, "general_prefix")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Component (a): Rank-Preserving Input Null-Space LoRA Injection
# ---------------------------------------------------------------------------


def _normalize_donor_factors_and_alphas(
    donor_a_list: Sequence[torch.Tensor] | torch.Tensor,
    alphas: Sequence[float] | float,
    expected_shape: tuple[int, int],
) -> tuple[list[torch.Tensor], list[float]]:
    """Validate donor ``A`` factors and scalar weights against ``(r, d_in)``."""
    if isinstance(donor_a_list, torch.Tensor):
        donor_seq: list[torch.Tensor] = [donor_a_list]
    elif isinstance(donor_a_list, Sequence) and not isinstance(donor_a_list, (str, bytes, bytearray)):
        if not donor_a_list:
            raise ValueError("donor_a_list must not be empty")
        donor_seq = list(donor_a_list)
    else:
        raise TypeError(
            f"donor_a_list must be a Sequence[Tensor] or Tensor, got {type(donor_a_list).__name__}"
        )

    for idx, d_ten in enumerate(donor_seq):
        _validate_fp_tensor(d_ten, f"donor_a_list[{idx}]", exact_ndim=2)
        if tuple(d_ten.shape) != expected_shape:
            raise ValueError(
                f"donor_a_list[{idx}] shape {tuple(d_ten.shape)} must match anchor A shape {expected_shape}"
            )

    if isinstance(alphas, (int, float)) and not isinstance(alphas, bool):
        if len(donor_seq) != 1:
            raise ValueError(
                f"Scalar alpha requires exactly 1 donor A factor, got {len(donor_seq)}"
            )
        alpha_seq = [_validate_finite_float(alphas, "alphas")]
    elif isinstance(alphas, Sequence) and not isinstance(alphas, (str, bytes, bytearray)):
        if len(alphas) != len(donor_seq):
            raise ValueError(
                f"alphas length ({len(alphas)}) must match donor_a_list length ({len(donor_seq)})"
            )
        alpha_seq = [_validate_finite_float(val, f"alphas[{idx}]") for idx, val in enumerate(alphas)]
    else:
        raise TypeError(f"alphas must be a Sequence[float] or float, got {type(alphas).__name__}")

    return donor_seq, alpha_seq


@torch.no_grad()
def inject_donor_lora_via_input_nullspace(
    a_anchor: torch.Tensor,
    b_anchor: torch.Tensor,
    donor_a_list: Sequence[torch.Tensor] | torch.Tensor,
    alphas: Sequence[float] | float,
    p_null_input: torch.Tensor,
    *,
    x_prose: torch.Tensor | None = None,
    eps: float = DEFAULT_PROSE_INVARIANCE_EPS,
    enforce_prose_invariance: bool = True,
    reorthogonalize_projector: bool = False,
    projector_eig_tol: float = DEFAULT_PROJECTOR_EIG_TOL,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Inject multi-donor LoRA refinements into ``A`` inside ``P_0`` while preserving rank ``r``.

    Computes in ``float64`` before casting to ``a_anchor``'s dtype/device:

        ``A_new = A_anchor + sum_k alpha_k * (A_donor_k - A_anchor) @ P_0``

    leaving ``B_anchor`` and the LoRA rank ``r = a_anchor.shape[0]`` strictly unchanged
    with zero SVD truncation error, while proving:

        ``||scaling * B_anchor @ (A_new - A_anchor) @ x_prose^T||_inf <= eps``.
    """
    a0 = _validate_fp_tensor(a_anchor, "a_anchor", exact_ndim=2)
    b0 = _validate_fp_tensor(b_anchor, "b_anchor", exact_ndim=2)
    rank_r, d_in = int(a0.shape[0]), int(a0.shape[1])
    if int(b0.shape[1]) != rank_r:
        raise ValueError(
            f"b_anchor inner rank {b0.shape[1]} must match a_anchor rank {rank_r}"
        )

    donor_seq, alpha_seq = _normalize_donor_factors_and_alphas(donor_a_list, alphas, (rank_r, d_in))
    p0_64, idem_err = _validate_projector_matrix(
        p_null_input,
        "p_null_input",
        expected_dim=d_in,
        eig_tol=projector_eig_tol,
        reorthogonalize=reorthogonalize_projector,
    )
    eps_f = _validate_nonneg_finite(eps, "eps")
    enforce_flag = _validate_bool(enforce_prose_invariance, "enforce_prose_invariance")
    scale_f = _validate_positive_finite(scaling, "scaling")

    a0_64 = a0.detach().to(device="cpu", dtype=torch.float64)
    delta_a_64 = torch.zeros_like(a0_64)
    for a_donor, alpha in zip(donor_seq, alpha_seq, strict=True):
        ad_64 = a_donor.detach().to(device="cpu", dtype=torch.float64)
        delta_a_64 = delta_a_64 + float(alpha) * ((ad_64 - a0_64) @ p0_64)

    a_new = (a0_64 + delta_a_64).to(device=a0.device, dtype=a0.dtype).contiguous()
    a_new_64 = a_new.detach().to(device="cpu", dtype=torch.float64)
    delta_actual_64 = a_new_64 - a0_64
    b0_64 = b0.detach().to(device="cpu", dtype=torch.float64)

    q_perp = torch.eye(d_in, dtype=torch.float64) - p0_64
    kernel_leak_mat = scale_f * (b0_64 @ (delta_actual_64 @ q_perp))
    kernel_leak_inf = float(kernel_leak_mat.abs().max().item())
    kernel_leak_fro = float(torch.linalg.matrix_norm(kernel_leak_mat).item())

    if x_prose is not None:
        xp = _validate_fp_tensor(x_prose, "x_prose", min_ndim=2)
        if int(xp.shape[-1]) != d_in:
            raise ValueError(f"x_prose width {xp.shape[-1]} must match d_in={d_in}")
        xp_2d = xp.detach().to(device="cpu").reshape(-1, d_in)
        work_dt = torch.float64 if a0.dtype == torch.float64 else torch.float32
        b_w = b0.detach().to(device="cpu", dtype=work_dt)
        da_w = a_new.detach().to(device="cpu", dtype=work_dt) - a0.detach().to(
            device="cpu", dtype=work_dt
        )
        xp_w = xp_2d.to(dtype=work_dt)
        drift_mat = scale_f * (b_w @ (da_w @ xp_w.T))
        prose_max_diff = float(drift_mat.abs().max().item())
        prose_rms = float(drift_mat.pow(2).mean().sqrt().item())
        exact_f64_diff = float(
            (scale_f * (b0_64 @ (delta_a_64 @ xp_2d.to(torch.float64).T))).abs().max().item()
        )
    else:
        prose_max_diff = kernel_leak_inf
        prose_rms = kernel_leak_fro / math.sqrt(float(max(1, int(b0.shape[0]) * d_in)))
        exact_f64_diff = float((scale_f * (b0_64 @ (delta_a_64 @ q_perp))).abs().max().item())

    if enforce_flag and prose_max_diff > eps_f:
        raise ValueError(
            f"Prose invariance violated: ||B_anchor @ (A_new - A_anchor) @ x_prose||_inf = "
            f"{prose_max_diff:.6e} > eps={eps_f:.6e}"
        )

    _, s_after, _ = decompose_lora_qr_svd(b0, a_new)
    sv_tol = 1e-12 * max(1.0, float(s_after[0].item()))
    eff_rank = int((s_after > sv_tol).sum().item())

    telemetry: dict[str, Any] = {
        "rank_r": rank_r,
        "effective_rank_after": eff_rank,
        "rank_increase": max(0, eff_rank - rank_r),
        "rank_preserved": True,
        "b_anchor_unchanged": True,
        "svd_truncation_error_fro": 0.0,
        "num_donors": len(donor_seq),
        "alphas": [float(a) for a in alpha_seq],
        "prose_invariance_max_diff": prose_max_diff,
        "prose_invariance_rms": prose_rms,
        "prose_invariance_float64_max_diff": exact_f64_diff,
        "kernel_leak_inf": kernel_leak_inf,
        "kernel_leak_fro": kernel_leak_fro,
        "delta_a_fro": float(torch.linalg.matrix_norm(delta_actual_64).item()),
        "projector_idempotence_error": idem_err,
    }
    return a_new, telemetry


@torch.no_grad()
def compare_input_nullspace_vs_full_delta_refactorization(
    a_anchor: torch.Tensor,
    b_anchor: torch.Tensor,
    donor_b_list: Sequence[torch.Tensor],
    donor_a_list: Sequence[torch.Tensor],
    alphas: Sequence[float],
    p_null_input: torch.Tensor,
    *,
    x_prose: torch.Tensor | None = None,
    eps: float = DEFAULT_PROSE_INVARIANCE_EPS,
    rank_sv_rel_tol: float = 1e-9,
) -> dict[str, Any]:
    """Compare input-side ``(A_donor - A_anchor) @ P_0`` against full-delta ``W @ P_0`` SVD re-factorization.

    Demonstrates why injecting strictly on ``A`` preserves exact target rank ``r`` with
    zero SVD truncation error, whereas re-factorizing
    ``W_anchor + sum_k alpha_k * (W_donor_k - W_anchor) @ P_0`` back to rank ``r``
    suffers rank inflation (``> r``) and nonzero SVD truncation error when donor
    output subspaces ``B_donor_k`` span directions outside ``Col(B_anchor)``.
    """
    a_new, inj_meta = inject_donor_lora_via_input_nullspace(
        a_anchor,
        b_anchor,
        donor_a_list,
        alphas,
        p_null_input,
        x_prose=x_prose,
        eps=eps,
    )
    if (
        isinstance(donor_b_list, (str, bytes))
        or not isinstance(donor_b_list, Sequence)
        or len(donor_b_list) != len(donor_a_list)
    ):
        raise ValueError("donor_b_list must match donor_a_list length")

    rank_r = int(a_anchor.shape[0])
    d_out, d_in = int(b_anchor.shape[0]), int(a_anchor.shape[1])
    b0_64 = b_anchor.detach().to("cpu", dtype=torch.float64)
    a0_64 = a_anchor.detach().to("cpu", dtype=torch.float64)
    p0_64, _ = _validate_projector_matrix(p_null_input, "p_null_input", expected_dim=d_in)
    rel_tol = _validate_positive_finite(rank_sv_rel_tol, "rank_sv_rel_tol")

    w_anchor = b0_64 @ a0_64
    w_full = w_anchor.clone()
    for idx, (bd, ad, alpha) in enumerate(zip(donor_b_list, donor_a_list, alphas, strict=True)):
        _validate_fp_tensor(bd, f"donor_b_list[{idx}]", exact_ndim=2)
        if tuple(bd.shape) != (d_out, rank_r):
            raise ValueError(
                f"donor_b_list[{idx}] shape {tuple(bd.shape)} must match b_anchor shape {(d_out, rank_r)}"
            )
        w_donor = bd.detach().to("cpu", dtype=torch.float64) @ ad.detach().to(
            "cpu", dtype=torch.float64
        )
        w_full = w_full + float(alpha) * ((w_donor - w_anchor) @ p0_64)

    u_full, s_full, vh_full = torch.linalg.svd(w_full, full_matrices=False)
    sv_cut = rel_tol * max(1e-12, float(s_full[0].item()))
    full_rank = int((s_full > sv_cut).sum().item())
    k_trunc = min(rank_r, int(s_full.numel()))
    w_trunc_r = (u_full[:, :k_trunc] * s_full[:k_trunc].unsqueeze(0)) @ vh_full[:k_trunc, :]
    trunc_err_fro = float(torch.linalg.matrix_norm(w_full - w_trunc_r).item())
    full_norm_fro = float(torch.linalg.matrix_norm(w_full).item())

    if x_prose is not None:
        xp_2d = _validate_fp_tensor(x_prose, "x_prose", min_ndim=2).detach().to(
            "cpu", dtype=torch.float64
        ).reshape(-1, d_in)
        full_trunc_prose_drift = float(((w_trunc_r - w_anchor) @ xp_2d.T).abs().max().item())
    else:
        q_perp = torch.eye(d_in, dtype=torch.float64) - p0_64
        full_trunc_prose_drift = float(((w_trunc_r - w_anchor) @ q_perp).abs().max().item())

    return {
        "target_rank_r": rank_r,
        "a_new_shape": list(a_new.shape),
        "input_injection_rank_r": inj_meta["rank_r"],
        "input_injection_effective_rank": inj_meta["effective_rank_after"],
        "input_injection_rank_increase": inj_meta["rank_increase"],
        "input_injection_svd_truncation_error_fro": 0.0,
        "input_injection_b_anchor_unchanged": True,
        "input_injection_prose_invariance_max_diff": inj_meta["prose_invariance_max_diff"],
        "full_delta_numerical_rank": full_rank,
        "full_delta_rank_excess": max(0, full_rank - rank_r),
        "full_delta_svd_truncation_error_fro": trunc_err_fro,
        "full_delta_svd_truncation_rel_error": trunc_err_fro / max(full_norm_fro, 1e-12),
        "full_delta_rank_r_prose_drift_inf": full_trunc_prose_drift,
        "singular_values_full_delta": [float(v) for v in s_full.tolist()],
        "input_injection_zero_truncation_verified": (
            inj_meta["svd_truncation_error_fro"] == 0.0 and trunc_err_fro > 0.0
        ),
    }


# ---------------------------------------------------------------------------
# 3. Component (b): Unfolded Safe-Tau (tau >= 1.0) LayerNorm Recalibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate_single_layernorm_nlls(
    h_pre_ln: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    is_general_mask: torch.Tensor,
    *,
    eps: float,
    ignore_index: int,
    seq_batch_size: int,
) -> tuple[float, float, float | None]:
    """Evaluate exact CPU ``float64`` total, general, and specialist NLL for one ``(w, b)``."""
    hidden_dim = int(h_pre_ln.shape[-1])
    vocab_size = int(head_weight.shape[0])
    h64 = h_pre_ln.detach().to(device="cpu", dtype=torch.float64)
    w64 = ln_w.detach().to(device="cpu", dtype=torch.float64)
    b64 = ln_b.detach().to(device="cpu", dtype=torch.float64) if ln_b is not None else None
    head64 = head_weight.detach().to(device="cpu", dtype=torch.float64)

    valid_all = targets != ignore_index
    total_sum = 0.0
    gen_sum = 0.0
    spec_sum = 0.0
    total_cnt = 0
    gen_cnt = 0
    spec_cnt = 0

    n_rows = int(h64.shape[0])
    for c0 in range(0, n_rows, max(1, seq_batch_size)):
        c1 = min(n_rows, c0 + max(1, seq_batch_size))
        sub_valid = valid_all[c0:c1].reshape(-1)
        if not bool(sub_valid.any().item()):
            continue
        z = F.layer_norm(h64[c0:c1], (hidden_dim,), None, None, eps=eps)
        u = z * w64 if b64 is None else (z * w64 + b64)
        logits = F.linear(u, head64).reshape(-1, vocab_size)[sub_valid]
        t_valid = targets[c0:c1].reshape(-1)[sub_valid]
        g_valid = is_general_mask[c0:c1].reshape(-1)[sub_valid]
        s_valid = ~g_valid
        tok_nll = F.cross_entropy(logits, t_valid, reduction="none")

        total_sum += float(tok_nll.sum().item())
        total_cnt += int(t_valid.numel())
        if bool(g_valid.any().item()):
            gen_sum += float(tok_nll[g_valid].sum().item())
            gen_cnt += int(g_valid.sum().item())
        if bool(s_valid.any().item()):
            spec_sum += float(tok_nll[s_valid].sum().item())
            spec_cnt += int(s_valid.sum().item())

    return (
        total_sum / float(max(1, total_cnt)),
        gen_sum / float(max(1, gen_cnt)),
        (spec_sum / float(spec_cnt)) if spec_cnt > 0 else None,
    )


@torch.no_grad()
def unfold_and_recalibrate_safe_layernorm_temperature(
    folded_ln_w: torch.Tensor,
    folded_ln_b: torch.Tensor | None,
    prior_tau: float,
    base_ln_w: torch.Tensor,
    base_ln_b: torch.Tensor | None,
    h_pre_ln: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    *,
    tau_grid: Sequence[float] = DEFAULT_SAFE_TAU_GRID,
    max_general_ratio: float = DEFAULT_SAFE_MAX_GENERAL_RATIO,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    eps: float = 1e-5,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    seq_batch_size: int = 4,
    base_general_nll_override: float | None = None,
    base_total_nll_override: float | None = None,
) -> dict[str, Any]:
    """Un-fold prior ``prior_tau`` sharpening and recalibrate strictly over safe ``tau >= 1.0``.

    When a prior stage folded ``1 / prior_tau`` (e.g., ``prior_tau = 0.98 < 1.0``)
    into the final LayerNorm parameters, multiplying by ``prior_tau`` recovers the
    unscaled candidate LayerNorm weights/biases, then executes an exact 2-matmul
    affine logit-linearity grid search over ``lambda_grid x tau_grid`` where every
    candidate temperature satisfies ``tau >= 1.0`` and general-domain NLL satisfies
    ``general_nll <= max_general_ratio * base_general_nll``.
    """
    h_t = _validate_fp_tensor(h_pre_ln, "h_pre_ln", min_ndim=2)
    hidden_dim = int(h_t.shape[-1])
    fw_t = _validate_1d_param(folded_ln_w, "folded_ln_w", expected_dim=hidden_dim)
    bw_t = _validate_1d_param(base_ln_w, "base_ln_w", expected_dim=hidden_dim)
    if (folded_ln_b is None) != (base_ln_b is None):
        raise ValueError("folded_ln_b and base_ln_b must either both be provided or both be None")
    fb_t = _validate_1d_param(folded_ln_b, "folded_ln_b", expected_dim=hidden_dim) if folded_ln_b is not None else None
    bb_t = _validate_1d_param(base_ln_b, "base_ln_b", expected_dim=hidden_dim) if base_ln_b is not None else None

    prior_tau_f = _validate_positive_finite(prior_tau, "prior_tau")
    safe_taus = _validate_safe_tau_grid(tau_grid)
    max_gen_ratio = _validate_positive_finite(max_general_ratio, "max_general_ratio")
    if max_gen_ratio < 1.0:
        raise ValueError(f"max_general_ratio must be >= 1.0, got {max_gen_ratio}")
    lam_vals = _validate_lambda_grid(lambda_grid)
    eps_f = _validate_positive_finite(eps, "eps")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    step_b = _validate_positive_int(seq_batch_size, "seq_batch_size")

    # Un-multiply prior 1 / prior_tau folding factor cleanly in float64
    unfolded_w = (fw_t.detach().to("cpu", dtype=torch.float64) * prior_tau_f).to(
        device=fw_t.device, dtype=fw_t.dtype
    )
    unfolded_b = (
        (fb_t.detach().to("cpu", dtype=torch.float64) * prior_tau_f).to(
            device=fb_t.device, dtype=fb_t.dtype
        )
        if fb_t is not None
        else None
    )

    grid_res = fast_affine_layernorm_temperature_grid(
        h_t,
        bw_t,
        bb_t,
        unfolded_w,
        unfolded_b,
        head_weight,
        targets,
        is_general_mask,
        eps=eps_f,
        lambda_grid=lam_vals,
        tau_grid=safe_taus,
        max_general_drift_rel=max_gen_ratio - 1.0,
        ignore_index=ignore_i,
        seq_batch_size=step_b,
        base_general_nll_override=base_general_nll_override,
        base_total_nll_override=base_total_nll_override,
    )

    batch_shape = tuple(h_t.shape[:-1])
    targets_t = targets.detach().to(device="cpu", dtype=torch.long)
    if isinstance(is_general_mask, torch.Tensor):
        mask_t = is_general_mask.detach().to(device="cpu", dtype=torch.bool)
    elif isinstance(is_general_mask, np.ndarray):
        mask_t = torch.from_numpy(is_general_mask.astype(bool, copy=True))
    else:
        mask_t = torch.tensor(list(is_general_mask), dtype=torch.bool)
    if mask_t.ndim == 1 and tuple(mask_t.shape) != batch_shape:
        mask_t = mask_t.view(batch_shape[0], *([1] * (len(batch_shape) - 1))).expand(batch_shape)

    nll_before, gen_before, spec_before = _evaluate_single_layernorm_nlls(
        h_t,
        fw_t,
        fb_t,
        head_weight,
        targets_t,
        mask_t,
        eps=eps_f,
        ignore_index=ignore_i,
        seq_batch_size=step_b,
    )
    ref_gen = float(grid_res["base_general_nll"])
    ref_total = float(grid_res["base_nll_mean"])
    gen_ratio_before = gen_before / max(ref_gen, 1e-12)
    gen_ratio_after = float(grid_res["best_general_nll"]) / max(ref_gen, 1e-12)

    return {
        "recalibrated_ln_w": grid_res["folded_ln_weight"],
        "recalibrated_ln_b": grid_res["folded_ln_bias"],
        "unfolded_ln_w": unfolded_w,
        "unfolded_ln_b": unfolded_b,
        "prior_tau": prior_tau_f,
        "best_lambda_ln": float(grid_res["best_lambda_ln"]),
        "best_tau_temp": float(grid_res["best_tau_temp"]),
        "guard_feasible": bool(grid_res["guard_feasible"]),
        "nll_before": nll_before,
        "nll_mean": float(grid_res["best_nll_mean"]),
        "general_nll_before": gen_before,
        "general_nll_after": float(grid_res["best_general_nll"]),
        "base_general_nll": ref_gen,
        "base_nll_mean": ref_total,
        "general_ratio_before": gen_ratio_before,
        "general_ratio_after": gen_ratio_after,
        "general_drift_pct_before": 100.0 * (gen_ratio_before - 1.0),
        "general_drift_pct_after": 100.0 * (gen_ratio_after - 1.0),
        "specialist_nll_before": spec_before,
        "specialist_nll_after": grid_res["best_specialist_nll"],
        "nll_drop_vs_folded_nats": nll_before - float(grid_res["best_nll_mean"]),
        "nll_drop_vs_base_nats": float(grid_res["nll_drop_nats"]),
        "head_matmuls_executed": int(grid_res["head_matmuls_executed"]),
        "naive_head_matmuls": int(grid_res["naive_head_matmuls"]),
        "matmul_speedup_factor": float(grid_res["matmul_speedup_factor"]),
        "tau_grid": list(grid_res["tau_grid"]),
        "lambda_grid": list(grid_res["lambda_grid"]),
    }


# ---------------------------------------------------------------------------
# 4. Component (c): Verified Multi-Campaign Null-Space Injection & ANSE Gate
# ---------------------------------------------------------------------------


def _normalize_checkpoint_to_blocks(
    ckpt: nn.Module | Mapping[str, Any],
    name: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize an ``nn.Module`` or structured block mapping into cloned finite tensors."""
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


def _validate_checkpoint_layouts(
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    anchor_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    donor_blocks: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> None:
    """Verify matching MoE block schemas and tensor shapes across base, anchor, and donors."""
    ref_keys = set(base_blocks.keys())
    if not any(k.endswith(".router") for k in ref_keys) or not any(".experts." in k for k in ref_keys):
        raise ValueError("Checkpoints must expose MoE '.router' and '.experts.' blocks")

    all_arms: list[tuple[str, Mapping[str, Mapping[str, torch.Tensor]]]] = [
        ("coadapted_anchor_checkpoint", anchor_blocks),
        *[(f"donor_checkpoints[{name!r}]", blk) for name, blk in donor_blocks.items()],
    ]
    for label, blk_map in all_arms:
        if set(blk_map.keys()) != ref_keys:
            raise ValueError(f"{label} block layout does not match base_checkpoint")
        for bid in sorted(ref_keys):
            b_sub, a_sub = base_blocks[bid], blk_map[bid]
            if set(b_sub.keys()) != set(a_sub.keys()):
                raise ValueError(f"Sub-key mismatch on block {bid!r} for {label}")
            for sk in b_sub:
                if b_sub[sk].shape != a_sub[sk].shape:
                    raise ValueError(f"Tensor shape mismatch on {bid}.{sk} for {label}")


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    general_prefix: str,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunks accepting either ``'expert_id'`` or ``'domain'`` keys."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    eids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must contain 'ids' and 'expert_id'/'domain'")
        raw_eid = ch.get("expert_id", ch.get("domain"))
        if raw_eid is None or not str(raw_eid).strip():
            raise ValueError(f"{name}[{idx}] must contain a non-empty 'expert_id' or 'domain'")
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
        eids.append(str(raw_eid).strip())

    assert seq_len is not None
    if max(0, score_from - 1) >= seq_len - 1:
        raise ValueError(f"{name} sequence length {seq_len} too short for score_from={score_from}")
    ids_2d = torch.stack(rows, dim=0)
    is_gen = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_gen)) or not bool(np.any(~is_gen)):
        raise ValueError(f"{name} must include both general ({general_prefix}*) and specialist chunks")
    return ids_2d, is_gen, eids


def _resolve_donor_assignments(
    donor_names: Sequence[str],
    donor_alphas: Mapping[str, float] | None,
    expert_donor_map: Mapping[int, str | Sequence[str]] | None,
    default_alpha: float,
) -> tuple[dict[str, float], dict[int, list[str]] | None]:
    """Validate and resolve per-donor alphas and optional expert-to-donor assignments."""
    if donor_alphas is not None:
        if not isinstance(donor_alphas, Mapping) or set(donor_alphas.keys()) != set(donor_names):
            raise ValueError("donor_alphas must provide one finite float weight per donor arm")
        resolved_alphas = {
            arm: _validate_finite_float(donor_alphas[arm], f"donor_alphas[{arm!r}]")
            for arm in donor_names
        }
    else:
        resolved_alphas = {arm: float(default_alpha) for arm in donor_names}

    if expert_donor_map is None:
        return resolved_alphas, None
    if not isinstance(expert_donor_map, Mapping):
        raise TypeError("expert_donor_map must be a Mapping[int, str | Sequence[str]] or None")

    resolved_map: dict[int, list[str]] = {}
    for raw_e, raw_target in expert_donor_map.items():
        e_idx = _validate_nonneg_int(raw_e, "expert_donor_map key")
        if isinstance(raw_target, str):
            targets = [raw_target]
        elif isinstance(raw_target, Sequence) and not isinstance(raw_target, (bytes, bytearray)) and raw_target:
            targets = [str(item) for item in raw_target]
        else:
            raise ValueError(f"expert_donor_map[{e_idx}] must be a donor name or non-empty sequence")
        for d_name in targets:
            if d_name not in donor_names:
                raise ValueError(f"expert_donor_map[{e_idx}] references unknown donor {d_name!r}")
        resolved_map[e_idx] = list(dict.fromkeys(targets))
    return resolved_alphas, resolved_map


def _resolve_layer_projector(
    p_source: torch.Tensor | Mapping[Any, Any] | None,
    layer_idx: int,
    proj_kind: str,
    dim: int,
    cfg: NullspaceInputInjectionConfig,
) -> torch.Tensor | None:
    """Resolve ``P_0`` for ``(layer_idx, 'x'|'act')`` from a tensor or nested mapping."""
    if p_source is None:
        return None
    if isinstance(p_source, torch.Tensor):
        if int(p_source.shape[0]) != dim:
            return None
        p_mat, _ = _validate_projector_matrix(
            p_source,
            "p_null_by_layer",
            expected_dim=dim,
            eig_tol=cfg.projector_eig_tol,
            reorthogonalize=cfg.reorthogonalize_projector,
        )
        return p_mat
    if not isinstance(p_source, Mapping):
        raise TypeError("p_null_by_layer must be a Tensor, Mapping, or None")

    for cand_k in (
        f"blocks.{layer_idx}.{proj_kind}",
        f"blocks.{layer_idx}",
        layer_idx,
        str(layer_idx),
        f"layer_{layer_idx}",
        proj_kind,
        "default",
    ):
        if cand_k not in p_source:
            continue
        val = p_source[cand_k]
        if isinstance(val, torch.Tensor) and int(val.shape[0]) == dim:
            p_mat, _ = _validate_projector_matrix(
                val,
                f"p_null_by_layer[{cand_k!r}]",
                expected_dim=dim,
                eig_tol=cfg.projector_eig_tol,
                reorthogonalize=cfg.reorthogonalize_projector,
            )
            return p_mat
        if isinstance(val, Mapping) and proj_kind in val and isinstance(val[proj_kind], torch.Tensor):
            sub_t = val[proj_kind]
            if int(sub_t.shape[0]) == dim:
                p_mat, _ = _validate_projector_matrix(
                    sub_t,
                    f"p_null_by_layer[{cand_k!r}][{proj_kind!r}]",
                    expected_dim=dim,
                    eig_tol=cfg.projector_eig_tol,
                    reorthogonalize=cfg.reorthogonalize_projector,
                )
                return p_mat
    return None


@torch.no_grad()
def _forward_pre_ln(model: nn.Module, ids_2d: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Propagate token IDs through embeddings and transformer blocks up to pre-final-LayerNorm."""
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
        nx: list[torch.Tensor] = []
        for b0 in range(0, xb.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, xb[b0 : b0 + batch_size], cos, sin)
            nx.append(forward_block_post_mlp(blk, xm, mi))
        xb = torch.cat(nx, dim=0)
    return xb


@torch.no_grad()
def _evaluate_checkpoint_state(
    *,
    model: nn.Module | None,
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
    eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    eval_ids: torch.Tensor | None,
    eval_eids: Sequence[str],
    cfg: NullspaceInputInjectionConfig,
) -> dict[str, Any]:
    """Evaluate one checkpoint via ``eval_fn`` or exact causal mid-row NLL."""
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


def _score_against_base(
    arm_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: NullspaceInputInjectionConfig,
) -> dict[str, Any]:
    """Compute general/worst drift ratios and ANSE energy relative to ``base_eval``."""
    if not arm_eval["all_finite"] or set(arm_eval["domain_nlls"].keys()) != set(base_eval["domain_nlls"].keys()):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "guard_ok": False,
            "specialist_gain_vs_base_nats": 0.0,
            "mean_diff_vs_base": float("inf"),
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = arm_eval["domain_nlls"]
    gen_ratio = float(arm_eval["general_nll"]) / max(float(base_eval["general_nll"]), 1e-8)
    worst_ratio = max(float(cand_doms[d]) / max(float(base_doms[d]), 1e-8) for d in base_doms)
    mean_diff = float(arm_eval["nll_mean"]) - float(base_eval["nll_mean"])
    spec_gain = float(base_eval["spec_nll"]) - float(arm_eval["spec_nll"])

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
        "guard_ok": guard_ok,
        "specialist_gain_vs_base_nats": spec_gain,
        "mean_diff_vs_base": mean_diff,
        "anse_energy": float(anse_e),
    }


def _build_explicit_projectors(
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None,
    p_null_by_layer: torch.Tensor | Mapping[Any, Any] | None,
    cfg: NullspaceInputInjectionConfig,
) -> Any:
    """Resolve user-supplied ``p_null_by_layer`` or construct from ``prose_hidden_states``."""
    if p_null_by_layer is not None:
        return p_null_by_layer
    if prose_hidden_states is None:
        return None
    if isinstance(prose_hidden_states, torch.Tensor):
        p_mat, _ = compute_prose_nullspace_projector_scaled(
            prose_hidden_states,
            sv_rel_tol=cfg.nullspace_sv_rel_tol,
            variance_threshold=cfg.nullspace_variance_threshold,
            max_active_modes=cfg.nullspace_max_active_modes,
        )
        return p_mat
    if isinstance(prose_hidden_states, Mapping):
        return {
            k: compute_prose_nullspace_projector_scaled(
                v,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
                max_active_modes=cfg.nullspace_max_active_modes,
            )[0]
            for k, v in prose_hidden_states.items()
        }
    if isinstance(prose_hidden_states, Sequence) and not isinstance(prose_hidden_states, (str, bytes)):
        return {
            idx: compute_prose_nullspace_projector_scaled(
                v,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
                max_active_modes=cfg.nullspace_max_active_modes,
            )[0]
            for idx, v in enumerate(prose_hidden_states)
        }
    raise TypeError("prose_hidden_states must be a Tensor, Mapping, Sequence, or None")


def _inject_expert_block(
    *,
    layer_idx: int,
    expert_idx: int,
    anchor_exp: Mapping[str, torch.Tensor],
    donor_blocks: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    block_id: str,
    selected_donors: Sequence[str],
    donor_alphas: Mapping[str, float],
    p_x: torch.Tensor | None,
    p_act: torch.Tensor | None,
    x_prose: torch.Tensor | None,
    cfg: NullspaceInputInjectionConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Inject donor ``A`` factors into one LoRA MoE expert slot while keeping ``B`` fixed."""
    out_exp = {k: v.detach().clone() for k, v in anchor_exp.items()}
    is_lora = all(f"{p}_A" in anchor_exp and f"{p}_B" in anchor_exp for p in _SWIGLU_PROJECTIONS)
    if not is_lora:
        return out_exp, {"injected": False, "reason": "non_lora_expert", "max_prose_diff": 0.0}

    if cfg.skip_zero_b_sinks and int(torch.count_nonzero(anchor_exp["w1_B"]).item()) == 0:
        return out_exp, {"injected": False, "reason": "zero_b_ghost_sink", "max_prose_diff": 0.0}

    active_donors = [
        d_name
        for d_name in selected_donors
        if (not cfg.skip_zero_b_sinks or int(torch.count_nonzero(donor_blocks[d_name][block_id]["w1_B"]).item()) > 0)
    ]
    if not active_donors:
        return out_exp, {"injected": False, "reason": "no_active_donors", "max_prose_diff": 0.0}

    max_prose_diff = 0.0
    injected_projs = 0
    for proj in cfg.inject_projections:
        p_mat = p_act if proj == "w3" else p_x
        if p_mat is None:
            continue
        a_key, b_key = f"{proj}_A", f"{proj}_B"
        donor_a_list = [donor_blocks[d_name][block_id][a_key] for d_name in active_donors]
        alphas = [donor_alphas[d_name] for d_name in active_donors]
        a_new, meta = inject_donor_lora_via_input_nullspace(
            anchor_exp[a_key],
            anchor_exp[b_key],
            donor_a_list,
            alphas,
            p_mat,
            x_prose=x_prose if proj != "w3" else None,
            eps=cfg.prose_invariance_eps,
            enforce_prose_invariance=cfg.enforce_prose_invariance,
            reorthogonalize_projector=cfg.reorthogonalize_projector,
            projector_eig_tol=cfg.projector_eig_tol,
        )
        out_exp[a_key] = a_new
        max_prose_diff = max(max_prose_diff, float(meta["prose_invariance_max_diff"]))
        injected_projs += 1

    return out_exp, {
        "injected": injected_projs > 0,
        "reason": "ok" if injected_projs > 0 else "missing_projector",
        "donors_used": active_donors,
        "max_prose_diff": max_prose_diff,
        "layer": layer_idx,
        "expert": expert_idx,
    }


def _resolve_injection_config(
    config: NullspaceInputInjectionConfig | None,
    inplace: bool | None,
) -> NullspaceInputInjectionConfig:
    """Resolve optional ``NullspaceInputInjectionConfig`` with ``inplace`` override."""
    if config is not None and not isinstance(config, NullspaceInputInjectionConfig):
        raise TypeError(
            f"config must be NullspaceInputInjectionConfig or None, got {type(config).__name__}"
        )
    cfg_dict = asdict(config) if config is not None else asdict(NullspaceInputInjectionConfig())
    if inplace is not None:
        cfg_dict["inplace"] = _validate_bool(inplace, "inplace")
    return NullspaceInputInjectionConfig(**cfg_dict)


@torch.no_grad()
def execute_verified_multicampaign_nullspace_injection(
    base_checkpoint: nn.Module | Mapping[str, Any],
    coadapted_anchor_checkpoint: nn.Module | Mapping[str, Any],
    donor_checkpoints: Mapping[str, nn.Module | Mapping[str, Any]],
    *,
    target_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    p_null_by_layer: torch.Tensor | Mapping[Any, Any] | None = None,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    donor_alphas: Mapping[str, float] | None = None,
    expert_donor_map: Mapping[int, str | Sequence[str]] | None = None,
    target_experts: Sequence[int] | set[int] | None = None,
    protected_experts: Sequence[int] | set[int] = (0,),
    prior_tau_folded: float | None = None,
    config: NullspaceInputInjectionConfig | None = None,
    inplace: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute verified multi-campaign rank-preserving input-nullspace injection & safe-tau LN calibration.

    Preserves the co-adapted ``(router, B_expert)`` pair from ``coadapted_anchor_checkpoint``
    bitwise, injects multi-campaign donor ``A`` factors strictly inside prose null-space
    ``P_0``, un-folds ``prior_tau_folded`` if present and recalibrates final LayerNorm
    strictly over safe ``tau >= 1.0``, and enforces ``ANSE Delta E < 0`` and the
    ``<= +1.0%`` general/worst-domain guard with atomic bitwise rollback and SHA-256 receipt.
    """
    cfg = _resolve_injection_config(config, inplace)
    if not isinstance(donor_checkpoints, Mapping) or not donor_checkpoints:
        raise ValueError("donor_checkpoints must be a non-empty Mapping of donor arms")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    if target_model is not None and not isinstance(target_model, nn.Module):
        raise TypeError(f"target_model must be an nn.Module or None, got {type(target_model).__name__}")
    if prior_tau_folded is not None:
        _validate_positive_finite(prior_tau_folded, "prior_tau_folded")

    donor_names = [_validate_nonempty_str(k, "donor_name") for k in donor_checkpoints.keys()]
    if cfg.rollback_target not in ("anchor", "base", *donor_names):
        raise ValueError(
            f"rollback_target {cfg.rollback_target!r} must be 'anchor', 'base', or a donor name"
        )

    prot_set = _validate_expert_index_set(protected_experts, "protected_experts")
    target_exp_set = (
        _validate_expert_index_set(target_experts, "target_experts")
        if target_experts is not None
        else None
    )
    resolved_alphas, resolved_map = _resolve_donor_assignments(
        donor_names, donor_alphas, expert_donor_map, cfg.default_alpha
    )

    base_blocks = _normalize_checkpoint_to_blocks(base_checkpoint, "base_checkpoint")
    anchor_blocks = _normalize_checkpoint_to_blocks(
        coadapted_anchor_checkpoint, "coadapted_anchor_checkpoint"
    )
    donor_blocks = {
        name: _normalize_checkpoint_to_blocks(donor_checkpoints[name], f"donor_checkpoints[{name!r}]")
        for name in donor_names
    }
    _validate_checkpoint_layouts(base_blocks, anchor_blocks, donor_blocks)

    work_model: nn.Module | None = None
    if target_model is not None:
        work_model = target_model if cfg.inplace else copy.deepcopy(target_model)
    elif isinstance(coadapted_anchor_checkpoint, nn.Module):
        work_model = (
            coadapted_anchor_checkpoint
            if cfg.inplace
            else copy.deepcopy(coadapted_anchor_checkpoint)
        )
    elif isinstance(base_checkpoint, nn.Module):
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)

    if work_model is None and eval_fn is None:
        raise ValueError("Either an nn.Module checkpoint/target_model or eval_fn must be provided")

    head_mod = (
        getattr(work_model, "head", getattr(work_model, "lm_head", None))
        if work_model is not None
        else None
    )
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
    eval_is_gen: np.ndarray | None = None
    eval_eids: list[str] = []
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
    anchor_sha = compute_checkpoint_sha256(anchor_blocks)
    donor_shas = {name: compute_checkpoint_sha256(blk) for name, blk in donor_blocks.items()}

    was_training = work_model.training if work_model is not None else False
    if work_model is not None:
        work_model.eval()

    try:
        explicit_p = _build_explicit_projectors(prose_hidden_states, p_null_by_layer, cfg)

        # Base probe reference for safe-tau LayerNorm calibration
        base_probe_eval: dict[str, Any] | None = None
        if work_model is not None and probe_ids is not None:
            base_probe_eval = _evaluate_checkpoint_state(
                model=work_model,
                blocks=base_blocks,
                eval_fn=None,
                eval_ids=probe_ids,
                eval_eids=probe_eids,
                cfg=cfg,
            )

        cand_blocks: dict[str, dict[str, torch.Tensor]] = {
            bid: {sk: t.detach().clone() for sk, t in sub.items()}
            for bid, sub in anchor_blocks.items()
        }

        injected_slots = 0
        max_prose_diff = 0.0
        h_pre_ln_probe: torch.Tensor | None = None

        # Sequential layer-wise injection (refreshing hidden states after each layer as in PILOT6)
        if work_model is not None and probe_ids is not None and probe_is_gen is not None:
            apply_checkpoint_blocks_inplace(work_model, cand_blocks)
            cos = work_model.cos_cached.to(probe_ids.device)
            sin = work_model.sin_cached.to(probe_ids.device)
            start_pos = max(0, cfg.score_from - 1)
            gen_mask = torch.as_tensor(probe_is_gen, dtype=torch.bool)
            xb = torch.cat(
                [
                    compute_embeddings_batch(work_model, probe_ids[b0 : b0 + cfg.batch_size])
                    for b0 in range(0, probe_ids.shape[0], cfg.batch_size)
                ],
                dim=0,
            )

            for l_idx, blk in enumerate(work_model.blocks):
                xm_l: list[torch.Tensor] = []
                mi_l: list[torch.Tensor] = []
                for b0 in range(0, xb.shape[0], cfg.batch_size):
                    xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + cfg.batch_size], cos, sin)
                    xm_l.append(xm_b)
                    mi_l.append(mi_b)
                x_mid = torch.cat(xm_l, dim=0)
                mlp_in = torch.cat(mi_l, dim=0)
                x_prose_l = (
                    mlp_in[:, start_pos:-1, :][gen_mask]
                    .reshape(-1, int(mlp_in.shape[-1]))
                    .contiguous()
                )
                p_x = _resolve_layer_projector(explicit_p, l_idx, "x", int(mlp_in.shape[-1]), cfg)
                if p_x is None:
                    p_x, _ = compute_prose_nullspace_projector_scaled(
                        x_prose_l,
                        sv_rel_tol=cfg.nullspace_sv_rel_tol,
                        variance_threshold=cfg.nullspace_variance_threshold,
                        max_active_modes=cfg.nullspace_max_active_modes,
                    )

                p_act: torch.Tensor | None = None
                if "w3" in cfg.inject_projections:
                    base_m = getattr(blk.mlp, "base_mlp", blk.mlp)
                    inter_dim = int(base_m.w3.weight.shape[1]) if hasattr(base_m, "w3") else None
                    if inter_dim is not None:
                        p_act = _resolve_layer_projector(explicit_p, l_idx, "act", inter_dim, cfg)
                        if p_act is None and hasattr(base_m, "w1") and hasattr(base_m, "w2"):
                            w_dt = base_m.w1.weight.dtype
                            xp_cast = x_prose_l.to(dtype=w_dt)
                            act_p = F.silu(base_m.w1(xp_cast).float()) * base_m.w2(xp_cast).float()
                            p_act, _ = compute_prose_nullspace_projector_scaled(
                                act_p,
                                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                                variance_threshold=cfg.nullspace_variance_threshold,
                                max_active_modes=cfg.nullspace_max_active_modes,
                            )

                if isinstance(blk.mlp, CrowdLoRAMoEMLP):
                    for e_key in sorted(blk.mlp.experts.keys(), key=lambda k: int(k)):
                        e_idx = int(e_key)
                        if e_idx in prot_set or (target_exp_set is not None and e_idx not in target_exp_set):
                            continue
                        bid = f"blocks.{l_idx}.mlp.experts.{e_idx}"
                        sel_donors = (
                            resolved_map.get(e_idx, [])
                            if resolved_map is not None
                            else donor_names
                        )
                        if not sel_donors:
                            continue
                        x_prose_check = (
                            x_prose_l
                            if (
                                cfg.nullspace_variance_threshold is None
                                or cfg.nullspace_variance_threshold >= 0.999999
                            )
                            else (
                                x_prose_l.to(torch.float64)
                                @ (torch.eye(int(p_x.shape[0]), dtype=torch.float64) - p_x)
                            ).to(x_prose_l.dtype)
                        )
                        updated_exp, exp_meta = _inject_expert_block(
                            layer_idx=l_idx,
                            expert_idx=e_idx,
                            anchor_exp=cand_blocks[bid],
                            donor_blocks=donor_blocks,
                            block_id=bid,
                            selected_donors=sel_donors,
                            donor_alphas=resolved_alphas,
                            p_x=p_x,
                            p_act=p_act,
                            x_prose=x_prose_check,
                            cfg=cfg,
                        )
                        cand_blocks[bid] = updated_exp
                        if exp_meta["injected"]:
                            injected_slots += 1
                            max_prose_diff = max(max_prose_diff, float(exp_meta["max_prose_diff"]))
                            exp_mod = blk.mlp.experts[e_key]
                            for p_name in cfg.inject_projections:
                                ak = f"{p_name}_A"
                                getattr(exp_mod, ak).copy_(updated_exp[ak].to(getattr(exp_mod, ak).dtype))

                nx_l = [
                    forward_block_post_mlp(
                        blk,
                        x_mid[b0 : b0 + cfg.batch_size],
                        mlp_in[b0 : b0 + cfg.batch_size],
                    )
                    for b0 in range(0, x_mid.shape[0], cfg.batch_size)
                ]
                xb = torch.cat(nx_l, dim=0)
            h_pre_ln_probe = xb
        else:
            # Dict/projector mode (without sequential forward pass)
            for bid in sorted(cand_blocks.keys()):
                m_exp = _BLOCK_EXPERT_RE.match(bid)
                if m_exp is None:
                    continue
                l_idx, e_idx = int(m_exp.group(1)), int(m_exp.group(2))
                if e_idx in prot_set or (target_exp_set is not None and e_idx not in target_exp_set):
                    continue
                sel_donors = resolved_map.get(e_idx, []) if resolved_map is not None else donor_names
                if not sel_donors or "w1_A" not in cand_blocks[bid]:
                    continue
                d_in = int(cand_blocks[bid]["w1_A"].shape[1])
                inter_dim = int(cand_blocks[bid]["w3_A"].shape[1])
                p_x = _resolve_layer_projector(explicit_p, l_idx, "x", d_in, cfg)
                p_act = (
                    _resolve_layer_projector(explicit_p, l_idx, "act", inter_dim, cfg)
                    if "w3" in cfg.inject_projections
                    else None
                )
                updated_exp, exp_meta = _inject_expert_block(
                    layer_idx=l_idx,
                    expert_idx=e_idx,
                    anchor_exp=cand_blocks[bid],
                    donor_blocks=donor_blocks,
                    block_id=bid,
                    selected_donors=sel_donors,
                    donor_alphas=resolved_alphas,
                    p_x=p_x,
                    p_act=p_act,
                    x_prose=None,
                    cfg=cfg,
                )
                cand_blocks[bid] = updated_exp
                if exp_meta["injected"]:
                    injected_slots += 1
                    max_prose_diff = max(max_prose_diff, float(exp_meta["max_prose_diff"]))

        # Safe-tau LayerNorm unfolding & recalibration (tau >= 1.0)
        ln_summary: dict[str, Any] = {
            "executed": False,
            "prior_tau_unfolded": float(prior_tau_folded) if prior_tau_folded is not None else 1.0,
            "best_lambda_ln": 1.0,
            "best_tau_temp": 1.0,
            "guard_feasible": True,
            "matmul_speedup_factor": 1.0,
        }
        if cfg.unfold_and_recalibrate_safe_ln and "final_ln" in cand_blocks:
            prior_t = float(prior_tau_folded) if prior_tau_folded is not None else 1.0
            if work_model is not None and (probe_ids is not None or eval_ids is not None):
                cal_ids = probe_ids if probe_ids is not None else eval_ids
                cal_gen = probe_is_gen if probe_is_gen is not None else eval_is_gen
                assert cal_ids is not None and cal_gen is not None
                if h_pre_ln_probe is None or cal_ids is not probe_ids:
                    apply_checkpoint_blocks_inplace(work_model, cand_blocks)
                    h_cal_full = _forward_pre_ln(work_model, cal_ids, cfg.batch_size)
                else:
                    h_cal_full = h_pre_ln_probe
                start_pos = max(0, cfg.score_from - 1)
                ln_mod = getattr(work_model, "ln", getattr(work_model, "norm", None))
                assert ln_mod is not None and head_mod is not None
                cal_res = unfold_and_recalibrate_safe_layernorm_temperature(
                    cand_blocks["final_ln"]["weight"],
                    cand_blocks["final_ln"].get("bias"),
                    prior_t,
                    base_blocks["final_ln"]["weight"],
                    base_blocks["final_ln"].get("bias"),
                    h_cal_full[:, start_pos:-1, :],
                    head_mod.weight,
                    cal_ids[:, start_pos + 1 :],
                    cal_gen,
                    tau_grid=cfg.safe_tau_grid,
                    max_general_ratio=cfg.calibrator_max_general_ratio,
                    lambda_grid=cfg.safe_lambda_grid,
                    eps=float(getattr(ln_mod, "eps", 1e-5)),
                    seq_batch_size=cfg.batch_size,
                    base_general_nll_override=(
                        base_probe_eval["general_nll"] if base_probe_eval is not None else None
                    ),
                    base_total_nll_override=(
                        base_probe_eval["nll_mean"] if base_probe_eval is not None else None
                    ),
                )
                cand_blocks["final_ln"]["weight"] = cal_res["recalibrated_ln_w"]
                if cal_res["recalibrated_ln_b"] is not None:
                    cand_blocks["final_ln"]["bias"] = cal_res["recalibrated_ln_b"]
                ln_summary = {
                    "executed": True,
                    "prior_tau_unfolded": prior_t,
                    "best_lambda_ln": cal_res["best_lambda_ln"],
                    "best_tau_temp": cal_res["best_tau_temp"],
                    "guard_feasible": cal_res["guard_feasible"],
                    "general_drift_pct_before": cal_res["general_drift_pct_before"],
                    "general_drift_pct_after": cal_res["general_drift_pct_after"],
                    "matmul_speedup_factor": cal_res["matmul_speedup_factor"],
                }
            elif prior_tau_folded is not None:
                unfolded_w = cand_blocks["final_ln"]["weight"].float() * prior_t
                unfolded_b = (
                    cand_blocks["final_ln"]["bias"].float() * prior_t
                    if "bias" in cand_blocks["final_ln"]
                    else None
                )
                safe_tau = float(cfg.safe_tau_grid[0])
                rw, rb = fold_temperature_into_norm_params(
                    unfolded_w.to(cand_blocks["final_ln"]["weight"].dtype),
                    unfolded_b.to(cand_blocks["final_ln"]["bias"].dtype) if unfolded_b is not None else None,
                    safe_tau,
                )
                cand_blocks["final_ln"]["weight"] = rw
                if rb is not None:
                    cand_blocks["final_ln"]["bias"] = rb
                ln_summary = {
                    "executed": True,
                    "prior_tau_unfolded": prior_t,
                    "best_lambda_ln": 1.0,
                    "best_tau_temp": safe_tau,
                    "guard_feasible": True,
                    "matmul_speedup_factor": 1.0,
                }

        # Verify bitwise co-adaptation of (router, B_expert)
        router_preserved = all(
            torch.equal(cand_blocks[bid][sk].cpu(), anchor_blocks[bid][sk].cpu())
            for bid in anchor_blocks
            if bid.endswith(".router")
            for sk in anchor_blocks[bid]
        )
        b_preserved = all(
            torch.equal(cand_blocks[bid][sk].cpu(), anchor_blocks[bid][sk].cpu())
            for bid in anchor_blocks
            if ".experts." in bid
            for sk in anchor_blocks[bid]
            if sk.endswith("_B")
        )
        cand_sha = compute_checkpoint_sha256(cand_blocks)

        # Evaluate base, anchor, donors, and candidate under ANSE ratchet
        base_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks=base_blocks,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        if not base_eval["all_finite"]:
            raise ValueError("base_checkpoint produced non-finite evaluation NLL")

        anchor_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks=anchor_blocks,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        anchor_metrics = _score_against_base(anchor_eval, base_eval, cfg)

        donor_metrics: dict[str, dict[str, Any]] = {}
        for d_name in donor_names:
            d_ev = _evaluate_checkpoint_state(
                model=work_model,
                blocks=donor_blocks[d_name],
                eval_fn=eval_fn,
                eval_ids=eval_ids,
                eval_eids=eval_eids,
                cfg=cfg,
            )
            d_sc = _score_against_base(d_ev, base_eval, cfg)
            donor_metrics[d_name] = {
                "nll_mean": d_ev["nll_mean"],
                "anse_energy": float(d_sc["anse_energy"]),
                "guard_ok": bool(d_sc["guard_ok"]),
                "general_drift_ratio": (
                    float(d_sc["general_drift_ratio"])
                    if math.isfinite(float(d_sc["general_drift_ratio"]))
                    else None
                ),
            }

        cand_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks=cand_blocks,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        cand_metrics = _score_against_base(cand_eval, base_eval, cfg)

        cand_anse = float(cand_metrics["anse_energy"])
        delta_e_vs_base = cand_anse
        delta_e_vs_anchor = cand_anse - float(anchor_metrics["anse_energy"])
        spec_gain_vs_anchor = (
            float(anchor_eval["spec_nll"]) - float(cand_eval["spec_nll"])
            if (anchor_eval["all_finite"] and cand_eval["all_finite"])
            else 0.0
        )

        accepted = bool(
            cand_eval["all_finite"]
            and cand_metrics["guard_ok"]
            and (delta_e_vs_base < -cfg.min_energy_improvement)
            and (not cfg.require_beat_anchor or delta_e_vs_anchor < -cfg.min_energy_improvement)
            and (not cfg.enforce_prose_invariance or max_prose_diff <= cfg.prose_invariance_eps)
        )

        if accepted:
            status = "MULTICAMPAIGN_NULLSPACE_INJECTION_COMMITTED"
            rolled_back = False
            rollback_verified = False
            final_blocks, final_eval, final_metrics = cand_blocks, cand_eval, cand_metrics
            expected_sha = cand_sha
            actual_rb = "none"
        else:
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            rolled_back = True
            if cfg.rollback_target == "base":
                final_blocks, final_eval = base_blocks, base_eval
                final_metrics = _score_against_base(base_eval, base_eval, cfg)
                expected_sha = base_sha
                actual_rb = "base"
            elif cfg.rollback_target in donor_blocks:
                t_name = cfg.rollback_target
                final_blocks = donor_blocks[t_name]
                final_eval = _evaluate_checkpoint_state(
                    model=work_model,
                    blocks=final_blocks,
                    eval_fn=eval_fn,
                    eval_ids=eval_ids,
                    eval_eids=eval_eids,
                    cfg=cfg,
                )
                final_metrics = _score_against_base(final_eval, base_eval, cfg)
                expected_sha = donor_shas[t_name]
                actual_rb = t_name
            else:
                final_blocks, final_eval, final_metrics = anchor_blocks, anchor_eval, anchor_metrics
                expected_sha = anchor_sha
                actual_rb = "anchor"

        if work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, final_blocks)
            extracted = extract_checkpoint_blocks(work_model)
            scoped = {bid: extracted[bid] for bid in final_blocks if bid in extracted}
            final_sha = compute_checkpoint_sha256(scoped)
        else:
            final_sha = compute_checkpoint_sha256(final_blocks)

        if rolled_back:
            rollback_verified = final_sha == expected_sha
            if not rollback_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {expected_sha[:12]}, got {final_sha[:12]}"
                )

        def _clean_ratio(val: float) -> float | None:
            return float(val) if math.isfinite(float(val)) else None

        receipt: dict[str, Any] = {
            "object": INJECTOR_RECEIPT_OBJECT,
            "schema": INJECTOR_SCHEMA,
            "status": status,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "rollback_target": actual_rb,
            "rollback_bitwise_verified": rollback_verified,
            "nll_mean": final_eval["nll_mean"],
            "anse_energy": float(final_metrics["anse_energy"]),
            "delta_e_vs_base": delta_e_vs_base,
            "delta_e_vs_anchor": delta_e_vs_anchor,
            "specialist_gain_nats": spec_gain_vs_anchor,
            "specialist_gain_vs_base_nats": float(cand_metrics["specialist_gain_vs_base_nats"]),
            "prose_invariance_max_diff": float(max_prose_diff),
            "rank_preserved": True,
            "rank_increase": 0,
            "svd_truncation_error_fro": 0.0,
            "router_bitwise_preserved": router_preserved,
            "b_anchor_bitwise_preserved": b_preserved,
            "injected_layer_expert_slots": injected_slots,
            "final_general_drift_ratio": _clean_ratio(final_metrics["general_drift_ratio"]),
            "final_worst_domain_ratio": _clean_ratio(final_metrics["worst_domain_ratio"]),
            "guard_ok": bool(final_metrics["guard_ok"]),
            "ln_recalibration_summary": ln_summary,
            "anchor_metrics": {
                "nll_mean": anchor_eval["nll_mean"],
                "anse_energy": float(anchor_metrics["anse_energy"]),
                "general_drift_ratio": _clean_ratio(anchor_metrics["general_drift_ratio"]),
                "guard_ok": bool(anchor_metrics["guard_ok"]),
            },
            "candidate_metrics": {
                "nll_mean": cand_eval["nll_mean"],
                "anse_energy": cand_anse,
                "general_drift_ratio": _clean_ratio(cand_metrics["general_drift_ratio"]),
                "worst_domain_ratio": _clean_ratio(cand_metrics["worst_domain_ratio"]),
                "guard_ok": bool(cand_metrics["guard_ok"]),
                "domain_nlls": dict(cand_eval["domain_nlls"]),
            },
            "donor_metrics": donor_metrics,
            "base_state_sha256": base_sha,
            "anchor_state_sha256": anchor_sha,
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
    "DEFAULT_PROSE_INVARIANCE_EPS",
    "DEFAULT_SAFE_MAX_GENERAL_RATIO",
    "DEFAULT_SAFE_TAU_GRID",
    "GENERAL_EXPERT_PREFIX",
    "INJECTOR_RECEIPT_OBJECT",
    "INJECTOR_SCHEMA",
    "NullspaceInputInjectionConfig",
    "compare_input_nullspace_vs_full_delta_refactorization",
    "execute_verified_multicampaign_nullspace_injection",
    "inject_donor_lora_via_input_nullspace",
    "unfold_and_recalibrate_safe_layernorm_temperature",
    "verify_nullspace_injection_receipt_provenance",
]
