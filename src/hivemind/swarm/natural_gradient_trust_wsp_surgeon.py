# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Fisher-Rao Natural-Gradient KL Trust-Region Weight-Space Surgeon for Open-Weight MoEs.

Implements a training-free, CPU-exact ``float64`` Kronecker-Factored (K-FAC)
Natural-Gradient KL Trust-Region Weight-Space Programming (WSP) engine for
open-weight Sparse MoE and FFN/LoRA blocks (``CrowdV1WithEngram``,
``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, ``CrowdRuntimeMLP``, HF-style
``layers``/``mlp`` blocks, and ``nn.Linear``):

1. ``estimate_kronecker_fisher_factors(x_prose, grad_out_prose, *, damping=1e-4)``:
   Computes symmetric positive-definite damped Kronecker empirical Fisher input
   and output-gradient factors ``(A_prose, G_prose)`` in CPU ``float64``.
2. ``solve_kl_trust_region_wsp_step(specialist_target_delta, a_prose, g_prose, *, max_kl_nats=0.005, p_null=None)``:
   Solves the closed-form K-FAC natural-gradient preconditioned step
   ``G^{-1} Delta_spec A^{-1}`` (or the exact Riemannian Schur-complement solution
   inside an AlphaEdit prose null-space ``P_0``) and rescales strictly onto the
   Fisher-Rao KL trust ellipsoid ``1/2 tr(G Delta_W A Delta_W^T) <= max_kl_nats``.
3. ``execute_verified_fisher_trust_moe_surgery(...)``:
   Computes empirical prose Fisher factors and steepest specialist directions in
   one deterministic forward/backward pass (or from explicit activations/deltas),
   applies globally KL-bounded trust-region surgery across MoE layers/experts,
   and gates commit on ANSE ``Delta E < 0`` and general-domain drift ``<= +1.0%``
   with atomic bitwise rollback and canonical SHA-256 provenance receipts.
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
import torch.nn.functional as F

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

SURGEON_SCHEMA: str = "hivemind.natural_gradient_trust_wsp_surgeon/1"
SURGEON_RECEIPT_OBJECT: str = "hivemind.natural_gradient_trust_wsp_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_FISHER_DAMPING: float = 1e-4
DEFAULT_MAX_KL_NATS: float = 0.005
DEFAULT_STEP_DAMPING: float = 0.5
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SCORE_FROM: int = 0

VALID_TARGET_FACTORS: frozenset[str] = frozenset(
    {
        "w1",
        "w2",
        "w3",
        "w1_A",
        "w2_A",
        "w3_A",
        "w1_B",
        "w2_B",
        "w3_B",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
DEFAULT_TARGET_FACTORS: tuple[str, ...] = (
    "w3_B",
    "w1_B",
    "w2_B",
    "w3_A",
    "w1_A",
    "w2_A",
    "w3",
    "w1",
    "w2",
    "down_proj",
    "gate_proj",
    "up_proj",
)
VALID_KL_BUDGET_SCOPES: frozenset[str] = frozenset({"global", "per_unit"})
VALID_NULLSPACE_SIDES: frozenset[str] = frozenset({"auto", "right", "left"})

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
    """Validate that ``value`` is a finite real number >= 0.0."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number > 0.0."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_finite_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D finite floating-point tensor as CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    out = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_sample_matrix(tensor: Any, name: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate a non-empty 2D or 3D finite floating-point sample tensor as CPU ``float64`` 2D."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    batch_dims = tuple(int(d) for d in tensor.shape[:-1])
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1])).contiguous()
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat, batch_dims


def _validate_spd_matrix(
    matrix: Any,
    expected_dim: int,
    name: str,
    *,
    ridge_jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate square symmetric positive-definite matrix and return ``(mat, cholesky_L)``."""
    mat = _validate_finite_2d(matrix, name)
    if tuple(mat.shape) != (expected_dim, expected_dim):
        raise ValueError(f"{name} shape {tuple(mat.shape)} must match ({expected_dim}, {expected_dim})")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    sym = (0.5 * (mat + mat.T)).contiguous()
    if ridge_jitter > 0.0:
        sym = sym + ridge_jitter * torch.eye(expected_dim, dtype=torch.float64)
    l_chol, info = torch.linalg.cholesky_ex(sym)
    if int(info.item()) != 0:
        raise ValueError(f"{name} must be strictly positive definite")
    return sym, l_chol


def _validate_square_projector(proj: Any, name: str) -> torch.Tensor:
    """Validate a square symmetric PSD projection/filter matrix in CPU ``float64``."""
    mat = _validate_finite_2d(proj, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {tuple(mat.shape)}")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    sym = (0.5 * (mat + mat.T)).contiguous()
    evals = torch.linalg.eigvalsh(sym)
    if float(evals[0].item()) < -1e-4 or float(evals[-1].item()) > 1.0 + 1e-4:
        raise ValueError(
            f"{name} eigenvalues must lie inside [0, 1], got range [{float(evals[0].item()):.3e}, {float(evals[-1].item()):.3e}]"
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


def verify_fisher_trust_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
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
class FisherTrustSurgeonConfig:
    """Validated configuration for Fisher-Rao natural-gradient trust-region WSP surgery."""

    max_kl_nats: float = DEFAULT_MAX_KL_NATS
    damping: float = DEFAULT_FISHER_DAMPING
    scale_damping_by_trace: bool = False
    kl_budget_scope: str = "global"
    nullspace_side: str = "auto"
    rescale_only_if_exceeds: bool = False
    use_specialist_ray_damping: bool = True
    step_damping: float = DEFAULT_STEP_DAMPING
    max_frobenius_norm: float | None = None
    protect_general_expert: bool = True
    general_expert_index: int = 0
    target_factors: tuple[str, ...] = DEFAULT_TARGET_FACTORS
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    general_guard_max_ratio: float | None = None
    enforce_all_domain_guard: bool = True
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    require_energy_improvement: bool = True
    enable_trust_backtracking: bool = False
    backtrack_halvings: int = 3
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 16
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_positive_finite(self.max_kl_nats, "max_kl_nats")
        _validate_positive_finite(self.damping, "damping")
        _validate_bool(self.scale_damping_by_trace, "scale_damping_by_trace")
        if self.kl_budget_scope not in VALID_KL_BUDGET_SCOPES:
            raise ValueError(f"kl_budget_scope must be one of {sorted(VALID_KL_BUDGET_SCOPES)}")
        if self.nullspace_side not in VALID_NULLSPACE_SIDES:
            raise ValueError(f"nullspace_side must be one of {sorted(VALID_NULLSPACE_SIDES)}")
        _validate_bool(self.rescale_only_if_exceeds, "rescale_only_if_exceeds")
        _validate_bool(self.use_specialist_ray_damping, "use_specialist_ray_damping")
        _validate_nonneg_finite(self.step_damping, "step_damping")
        if self.max_frobenius_norm is not None:
            _validate_positive_finite(self.max_frobenius_norm, "max_frobenius_norm")
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
        object.__setattr__(self, "target_factors", tuple(dict.fromkeys(str(f) for f in self.target_factors)))

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
        _validate_bool(self.enable_trust_backtracking, "enable_trust_backtracking")
        _validate_positive_int(self.backtrack_halvings, "backtrack_halvings")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Kronecker Empirical Fisher Estimation & Closed-Form KL Trust Solver
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_kronecker_kl_nats(
    delta_w: torch.Tensor,
    a_prose: torch.Tensor,
    g_prose: torch.Tensor,
) -> float:
    """Compute exact second-order Fisher-Rao KL ``1/2 tr(G Delta_W A Delta_W^T)`` in ``float64``."""
    dw = _validate_finite_2d(delta_w, "delta_w")
    d_out, d_in = int(dw.shape[0]), int(dw.shape[1])
    a_mat = _validate_finite_2d(a_prose, "a_prose")
    g_mat = _validate_finite_2d(g_prose, "g_prose")
    if tuple(a_mat.shape) != (d_in, d_in):
        raise ValueError(f"a_prose shape {tuple(a_mat.shape)} must match ({d_in}, {d_in})")
    if tuple(g_mat.shape) != (d_out, d_out):
        raise ValueError(f"g_prose shape {tuple(g_mat.shape)} must match ({d_out}, {d_out})")
    quad = 0.5 * float(torch.sum(dw * (g_mat @ dw @ a_mat)).item())
    return max(0.0, quad)


@torch.no_grad()
def estimate_kronecker_fisher_factors(
    x_prose: torch.Tensor,
    grad_out_prose: torch.Tensor,
    *,
    damping: float = DEFAULT_FISHER_DAMPING,
    ridge_floor: float = 0.0,
    scale_damping_by_trace: bool = False,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Estimate damped Kronecker empirical Fisher factors ``(A_prose, G_prose)`` in CPU ``float64``.

    For a linear or LoRA expert block ``s = W x`` on prose calibration tokens with
    activations ``x_prose`` of shape ``(..., d_in)`` and output score gradients
    ``grad_out_prose`` of shape ``(..., d_out)``:

        ``A_prose = (1 / N) X_prose^T X_prose + lam_A I_{d_in}``
        ``G_prose = (1 / N) Gamma_prose^T Gamma_prose + lam_G I_{d_out}``

    where both ``A_prose`` and ``G_prose`` are guaranteed symmetric positive definite.
    """
    x2d, x_batch = _validate_sample_matrix(x_prose, "x_prose")
    g2d, g_batch = _validate_sample_matrix(grad_out_prose, "grad_out_prose")
    if x_batch != g_batch and int(x2d.shape[0]) != int(g2d.shape[0]):
        raise ValueError(
            f"x_prose sample count {int(x2d.shape[0])} must match grad_out_prose {int(g2d.shape[0])}"
        )
    if len(x_batch) == len(g_batch) and x_batch != g_batch:
        raise ValueError(
            f"x_prose leading shape {x_batch} must match grad_out_prose leading shape {g_batch}"
        )

    damp_f = _validate_positive_finite(damping, "damping")
    floor_f = _validate_nonneg_finite(ridge_floor, "ridge_floor")
    scale_trace = _validate_bool(scale_damping_by_trace, "scale_damping_by_trace")
    ret_diag = _validate_bool(return_diagnostics, "return_diagnostics")

    n_tok = int(x2d.shape[0])
    d_in = int(x2d.shape[1])
    d_out = int(g2d.shape[1])

    a0 = (x2d.T @ x2d) / float(n_tok)
    g0 = (g2d.T @ g2d) / float(n_tok)
    a0 = 0.5 * (a0 + a0.T)
    g0 = 0.5 * (g0 + g0.T)

    if scale_trace:
        tr_a = max(0.0, float(torch.diagonal(a0).mean().item()))
        tr_g = max(0.0, float(torch.diagonal(g0).mean().item()))
        lam_a = max(damp_f * tr_a, max(floor_f, 1e-12))
        lam_g = max(damp_f * tr_g, max(floor_f, 1e-12))
    else:
        lam_a = max(damp_f, floor_f)
        lam_g = max(damp_f, floor_f)

    a_prose = (a0 + lam_a * torch.eye(d_in, dtype=torch.float64)).contiguous()
    g_prose = (g0 + lam_g * torch.eye(d_out, dtype=torch.float64)).contiguous()

    if not ret_diag:
        return a_prose, g_prose

    ev_a = torch.linalg.eigvalsh(a_prose)
    ev_g = torch.linalg.eigvalsh(g_prose)
    return a_prose, g_prose, {
        "n_samples": n_tok,
        "d_in": d_in,
        "d_out": d_out,
        "damping": damp_f,
        "lambda_a": lam_a,
        "lambda_g": lam_g,
        "a_trace_mean": float(torch.diagonal(a_prose).mean().item()),
        "g_trace_mean": float(torch.diagonal(g_prose).mean().item()),
        "a_min_eig": float(ev_a[0].item()),
        "a_max_eig": float(ev_a[-1].item()),
        "g_min_eig": float(ev_g[0].item()),
        "g_max_eig": float(ev_g[-1].item()),
    }


def _resolve_nullspace_side(
    p_null: torch.Tensor,
    d_out: int,
    d_in: int,
    nullspace_side: str,
) -> str:
    """Resolve whether square projector ``p_null`` acts on ``'right'`` (input) or ``'left'`` (output)."""
    if nullspace_side not in VALID_NULLSPACE_SIDES:
        raise ValueError(
            f"nullspace_side must be one of {sorted(VALID_NULLSPACE_SIDES)}, got {nullspace_side!r}"
        )
    d_p = int(p_null.shape[0])
    if nullspace_side == "right":
        if d_p != d_in:
            raise ValueError(f"Right nullspace projector dim {d_p} must match d_in={d_in}")
        return "right"
    if nullspace_side == "left":
        if d_p != d_out:
            raise ValueError(f"Left nullspace projector dim {d_p} must match d_out={d_out}")
        return "left"
    if d_p == d_in:
        return "right"
    if d_p == d_out:
        return "left"
    raise ValueError(
        f"p_null shape {tuple(p_null.shape)} matches neither d_in={d_in} nor d_out={d_out}"
    )


def _solve_nullspace_constrained_natural_step(
    target64: torch.Tensor,
    a64: torch.Tensor,
    l_a: torch.Tensor,
    g64: torch.Tensor,
    l_g: torch.Tensor,
    p64: torch.Tensor | None,
    side: str,
) -> tuple[torch.Tensor, str, float]:
    """Compute unconstrained or exact Riemannian ``P_0``-constrained natural-gradient step."""
    if p64 is None:
        g_inv_delta = torch.cholesky_solve(target64, l_g)
        delta_nat = torch.cholesky_solve(g_inv_delta.T, l_a).T.contiguous()
        return delta_nat, "none", 0.0

    idem_err = float(torch.linalg.matrix_norm(p64 @ p64 - p64, ord="fro").item())
    p_norm = max(1.0, float(torch.linalg.matrix_norm(p64, ord="fro").item()))
    is_projector = idem_err <= 1e-3 * p_norm

    if is_projector:
        evals, evecs = torch.linalg.eigh(p64)
        basis_u = evecs[:, evals > 0.5]
        if int(basis_u.shape[1]) == 0:
            return torch.zeros_like(target64), "zero_nullspace", 0.0

        if side == "right":
            a_sub = 0.5 * (basis_u.T @ a64 @ basis_u + (basis_u.T @ a64 @ basis_u).T)
            l_sub = torch.linalg.cholesky(a_sub)
            g_inv_delta_u = torch.cholesky_solve(target64 @ basis_u, l_g)
            z_sol = torch.cholesky_solve(g_inv_delta_u.T, l_sub).T
            delta_nat = (z_sol @ basis_u.T).contiguous()
            resid = float(torch.linalg.matrix_norm(delta_nat - delta_nat @ p64, ord="fro").item())
            return delta_nat, "right_input_nullspace", resid

        g_sub = 0.5 * (basis_u.T @ g64 @ basis_u + (basis_u.T @ g64 @ basis_u).T)
        l_sub = torch.linalg.cholesky(g_sub)
        u_t_delta_a_inv = torch.cholesky_solve((basis_u.T @ target64).T, l_a).T
        z_sol = torch.cholesky_solve(u_t_delta_a_inv, l_sub)
        delta_nat = (basis_u @ z_sol).contiguous()
        resid = float(torch.linalg.matrix_norm(delta_nat - p64 @ delta_nat, ord="fro").item())
        return delta_nat, "left_output_nullspace", resid

    g_inv_delta = torch.cholesky_solve(target64, l_g)
    delta_unproj = torch.cholesky_solve(g_inv_delta.T, l_a).T
    if side == "right":
        delta_nat = (delta_unproj @ p64).contiguous()
        return delta_nat, "right_soft_filter", 0.0
    delta_nat = (p64 @ delta_unproj).contiguous()
    return delta_nat, "left_soft_filter", 0.0


@torch.no_grad()
def solve_kl_trust_region_wsp_step(
    specialist_target_delta: torch.Tensor,
    a_prose: torch.Tensor,
    g_prose: torch.Tensor,
    *,
    max_kl_nats: float = DEFAULT_MAX_KL_NATS,
    p_null: torch.Tensor | None = None,
    nullspace_side: str = "auto",
    rescale_only_if_exceeds: bool = False,
    max_frobenius_norm: float | None = None,
    ridge_jitter: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve closed-form K-FAC natural-gradient step strictly inside the Fisher-Rao KL ellipsoid.

    Solves ``max_{Delta_W} <Delta_spec, Delta_W>_F`` subject to
    ``1/2 tr(G_prose Delta_W A_prose Delta_W^T) <= max_kl_nats`` (and optionally
    constrained to ``Delta_W in range(P_0)`` when ``p_null`` is supplied).
    Returns ``(delta_w_float64, diagnostics)``.
    """
    target64 = _validate_finite_2d(specialist_target_delta, "specialist_target_delta")
    d_out, d_in = int(target64.shape[0]), int(target64.shape[1])

    kl_cap = _validate_positive_finite(max_kl_nats, "max_kl_nats")
    jit_f = _validate_nonneg_finite(ridge_jitter, "ridge_jitter")
    shrink_only = _validate_bool(rescale_only_if_exceeds, "rescale_only_if_exceeds")
    frob_cap = (
        _validate_positive_finite(max_frobenius_norm, "max_frobenius_norm")
        if max_frobenius_norm is not None
        else None
    )

    a64, l_a = _validate_spd_matrix(a_prose, d_in, "a_prose", ridge_jitter=jit_f)
    g64, l_g = _validate_spd_matrix(g_prose, d_out, "g_prose", ridge_jitter=jit_f)

    p64: torch.Tensor | None = None
    resolved_side = "none"
    if p_null is not None:
        p64 = _validate_square_projector(p_null, "p_null")
        resolved_side = _resolve_nullspace_side(p64, d_out, d_in, nullspace_side)
    elif nullspace_side not in VALID_NULLSPACE_SIDES:
        raise ValueError(f"nullspace_side must be one of {sorted(VALID_NULLSPACE_SIDES)}")

    delta_nat, ns_mode, raw_ns_resid = _solve_nullspace_constrained_natural_step(
        target64, a64, l_a, g64, l_g, p64, resolved_side
    )

    raw_kl = max(0.0, 0.5 * float(torch.sum(delta_nat * (g64 @ delta_nat @ a64)).item()))
    raw_frob = float(torch.linalg.matrix_norm(delta_nat, ord="fro").item())

    if raw_kl <= 1e-30 or raw_frob <= 1e-30:
        delta_out = torch.zeros((d_out, d_in), dtype=torch.float64)
        trust_scale = 0.0
        kl_after = 0.0
        frob_after = 0.0
        ns_resid_after = 0.0
    else:
        trust_scale = min(1.0, math.sqrt(kl_cap / raw_kl)) if shrink_only else math.sqrt(kl_cap / raw_kl)
        if frob_cap is not None and raw_frob * trust_scale > frob_cap:
            trust_scale = frob_cap / max(raw_frob, 1e-30)

        delta_out = (trust_scale * delta_nat).contiguous()
        kl_after = max(0.0, 0.5 * float(torch.sum(delta_out * (g64 @ delta_out @ a64)).item()))
        if kl_after > kl_cap:
            fix = math.sqrt(kl_cap / kl_after) * (1.0 - 1e-12)
            trust_scale *= fix
            delta_out = (trust_scale * delta_nat).contiguous()
            kl_after = max(0.0, 0.5 * float(torch.sum(delta_out * (g64 @ delta_out @ a64)).item()))
        frob_after = float(torch.linalg.matrix_norm(delta_out, ord="fro").item())
        ns_resid_after = trust_scale * raw_ns_resid

    target_frob = float(torch.linalg.matrix_norm(target64, ord="fro").item())
    if target_frob > 1e-30 and frob_after > 0.0:
        delta_euc = (frob_after / target_frob) * target64
        euclidean_kl = max(0.0, 0.5 * float(torch.sum(delta_euc * (g64 @ delta_euc @ a64)).item()))
    else:
        euclidean_kl = 0.0

    spec_alignment = float(torch.sum(target64 * delta_out).item())
    return delta_out, {
        "shape": (d_out, d_in),
        "kl_trust_radius": kl_cap,
        "kl_nats": kl_after,
        "raw_natural_kl_nats": raw_kl,
        "euclidean_kl_nats": euclidean_kl,
        "trust_scale": trust_scale,
        "frobenius_norm": frob_after,
        "raw_frobenius_norm": raw_frob,
        "specialist_alignment": spec_alignment,
        "nullspace_mode": ns_mode,
        "nullspace_residual_norm": ns_resid_after,
        "kl_bound_satisfied": bool(kl_after <= kl_cap + 1e-12),
    }


# ---------------------------------------------------------------------------
# 3. Model Discovery, Empirical Fisher Collection & ANSE MoE Surgery
# ---------------------------------------------------------------------------


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


def _is_expert_targeted(
    e_idx: int,
    target_expert_set: set[int] | None,
    cfg: FisherTrustSurgeonConfig,
) -> bool:
    """Return True iff expert ``e_idx`` is eligible for Fisher trust-region surgery."""
    if target_expert_set is not None:
        return e_idx in target_expert_set
    if cfg.protect_general_expert and e_idx == cfg.general_expert_index:
        return False
    return True


def _enumerate_target_units(
    mlp_blocks: Sequence[tuple[int, nn.Module]],
    *,
    has_crowd_blocks: bool,
    target_layer_set: set[int],
    target_expert_set: set[int] | None,
    target_factor_set: set[str],
    cfg: FisherTrustSurgeonConfig,
) -> list[dict[str, Any]]:
    """Enumerate all targeted Linear or LoRA weight parameter units across selected layers."""
    units: list[dict[str, Any]] = []
    for l_idx, mlp_mod in mlp_blocks:
        if l_idx not in target_layer_set:
            continue
        prefix = f"blocks.{l_idx}.mlp" if has_crowd_blocks else f"layers.{l_idx}.mlp"
        resolved_mlp = getattr(mlp_mod, "moe_mlp", mlp_mod)

        if isinstance(resolved_mlp, CrowdLoRAMoEMLP):
            for e_key in sorted(resolved_mlp.experts.keys(), key=int):
                e_idx = int(e_key)
                if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                    continue
                exp = resolved_mlp.experts[e_key]
                for f_name in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    if f_name not in target_factor_set:
                        continue
                    param = getattr(exp, f_name, None)
                    if isinstance(param, torch.Tensor):
                        units.append(
                            {
                                "name": f"{prefix}.experts.{e_key}.{f_name}",
                                "kind": "lora_factor",
                                "layer_idx": l_idx,
                                "expert_idx": e_idx,
                                "factor": f_name,
                                "module": exp,
                                "mlp_module": resolved_mlp,
                                "param": param,
                            }
                        )
        elif isinstance(resolved_mlp, CrowdFineGrainedMoEMLP):
            for e_idx in range(resolved_mlp.num_experts):
                if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                    continue
                for f_name, mod_list in (
                    ("w1", resolved_mlp.experts_w1),
                    ("w2", resolved_mlp.experts_w2),
                    ("w3", resolved_mlp.experts_w3),
                ):
                    if f_name in target_factor_set:
                        lin = mod_list[e_idx]
                        units.append(
                            {
                                "name": f"{prefix}.experts.{e_idx}.{f_name}",
                                "kind": "linear",
                                "layer_idx": l_idx,
                                "expert_idx": e_idx,
                                "factor": f_name,
                                "module": lin,
                                "mlp_module": resolved_mlp,
                                "param": lin.weight,
                            }
                        )
        elif hasattr(resolved_mlp, "experts"):
            experts_attr = resolved_mlp.experts
            items = (
                [(k, experts_attr[k]) for k in sorted(experts_attr.keys(), key=int)]
                if isinstance(experts_attr, nn.ModuleDict)
                else [(str(i), mod) for i, mod in enumerate(experts_attr)]
            )
            for e_key, exp in items:
                e_idx = int(e_key)
                if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                    continue
                for f_name in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
                    if f_name not in target_factor_set:
                        continue
                    sub = getattr(exp, f_name, None)
                    if isinstance(sub, nn.Linear):
                        units.append(
                            {
                                "name": f"{prefix}.experts.{e_key}.{f_name}",
                                "kind": "linear",
                                "layer_idx": l_idx,
                                "expert_idx": e_idx,
                                "factor": f_name,
                                "module": sub,
                                "mlp_module": resolved_mlp,
                                "param": sub.weight,
                            }
                        )
        elif isinstance(resolved_mlp, nn.Linear):
            units.append(
                {
                    "name": f"{prefix}.weight",
                    "kind": "linear",
                    "layer_idx": l_idx,
                    "expert_idx": None,
                    "factor": "weight",
                    "module": resolved_mlp,
                    "mlp_module": resolved_mlp,
                    "param": resolved_mlp.weight,
                }
            )
        else:
            found_dense = False
            for f_name in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
                if f_name not in target_factor_set:
                    continue
                sub = getattr(resolved_mlp, f_name, None)
                if isinstance(sub, nn.Linear):
                    found_dense = True
                    units.append(
                        {
                            "name": f"{prefix}.{f_name}",
                            "kind": "linear",
                            "layer_idx": l_idx,
                            "expert_idx": None,
                            "factor": f_name,
                            "module": sub,
                            "mlp_module": resolved_mlp,
                            "param": sub.weight,
                        }
                    )
            if not found_dense:
                raise ValueError(
                    f"Unsupported MLP module type {type(resolved_mlp).__name__} for Fisher trust-region surgery"
                )
    return units


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
    cfg: FisherTrustSurgeonConfig,
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


def _forward_crowd_model_causal_sum_loss(
    model: nn.Module,
    ids_batch: torch.Tensor,
    *,
    score_from: int,
) -> tuple[torch.Tensor, int]:
    """Run differentiable forward pass on ``CrowdV1WithEngram`` returning summed causal CE loss."""
    dev = model.wte.weight.device
    ids_dev = ids_batch.to(dev)
    x = model.wte(ids_dev).to(torch.float32)
    if getattr(model.cfg, "engram_buckets", 0) > 0:
        x = x + model.engram(ids_dev).to(device=dev, dtype=torch.float32)
    x = x.detach().requires_grad_(True)

    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    for blk in model.blocks:
        blk_dt = blk.ln1.weight.dtype
        x_in = x.to(dtype=blk_dt)
        h_attn, _ = blk.attn(blk.ln1(x_in), cos, sin, past_kv=None, use_cache=False)
        x_mid = x_in + h_attn
        mlp_out = blk.mlp(blk.ln2(x_mid.to(dtype=blk.ln2.weight.dtype)))
        x = (x_mid.to(torch.float32) + mlp_out.to(torch.float32)).to(dtype=blk_dt)

    h_norm = model.ln(x.to(dtype=model.ln.weight.dtype))
    start_pos = max(0, score_from - 1)
    pred_h = h_norm[:, start_pos:-1, :]
    targets = ids_dev[:, start_pos + 1 :]
    logits = F.linear(pred_h.to(dtype=model.head.weight.dtype), model.head.weight).to(torch.float32)
    vocab = int(logits.shape[-1])
    loss_sum = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="sum")
    return loss_sum, int(targets.numel())


def _capture_linear_pass(
    units: Sequence[dict[str, Any]],
    forward_fn: Callable[[], tuple[torch.Tensor, int]],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor], int]:
    """Capture per-unit ``(x_in, grad_out)`` and parameter gradients in one differentiable pass."""
    hooks: list[Any] = []
    patched_adapters: list[tuple[CrowdLoRAExpertAdapter, Any]] = []
    unit_slots: dict[str, list[list[torch.Tensor | None]]] = {u["name"]: [] for u in units}
    mlp_slots: dict[int, list[list[torch.Tensor | None]]] = {}

    linear_units = [u for u in units if u["kind"] == "linear"]
    lora_by_adapter: dict[int, tuple[CrowdLoRAExpertAdapter, dict[str, dict[str, Any]]]] = {}
    mlp_by_id: dict[int, nn.Module] = {}
    for u in units:
        if u["kind"] == "lora_factor":
            exp = u["module"]
            mlp_by_id[id(u["mlp_module"])] = u["mlp_module"]
            mlp_slots.setdefault(id(u["mlp_module"]), [])
            entry = lora_by_adapter.setdefault(id(exp), (exp, {}))
            entry[1][u["factor"]] = u

    def _register_pair(bucket: list[list[torch.Tensor | None]], x_t: torch.Tensor, y_t: torch.Tensor) -> None:
        x_cap = x_t.detach().reshape(-1, int(x_t.shape[-1])).to(device="cpu", dtype=torch.float64)
        slot: list[torch.Tensor | None] = [x_cap, None]
        bucket.append(slot)
        out_dim = int(y_t.shape[-1])

        def _g_hook(grad: torch.Tensor) -> None:
            slot[1] = grad.detach().reshape(-1, out_dim).to(device="cpu", dtype=torch.float64)

        if y_t.requires_grad:
            y_t.register_hook(_g_hook)

    try:
        for u in linear_units:
            mod: nn.Linear = u["module"]
            u_name = u["name"]

            def _make_lin_hook(target_name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def _lin_hook(_m: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    _register_pair(unit_slots[target_name], inputs[0], output)

                return _lin_hook

            hooks.append(mod.register_forward_hook(_make_lin_hook(u_name)))

        for mlp_id, mlp_mod in mlp_by_id.items():
            def _make_mlp_hook(mid: int) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def _mlp_hook(_m: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    _register_pair(mlp_slots[mid], inputs[0], output)

                return _mlp_hook

            hooks.append(mlp_mod.register_forward_hook(_make_mlp_hook(mlp_id)))

        for _, (exp, factor_map) in lora_by_adapter.items():
            orig_fwd = exp.forward_delta
            patched_adapters.append((exp, orig_fwd))

            def _make_lora_fwd(
                adapter: CrowdLoRAExpertAdapter,
                fmap: dict[str, dict[str, Any]],
            ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
                def _wrapped_forward_delta(
                    x_fp32: torch.Tensor,
                    base_h1: torch.Tensor,
                    base_h2: torch.Tensor,
                    base_w3_weight: torch.Tensor,
                ) -> torch.Tensor:
                    z1 = F.linear(x_fp32, adapter.w1_A)
                    if "w1_A" in fmap:
                        _register_pair(unit_slots[fmap["w1_A"]["name"]], x_fp32, z1)
                    z1_s = z1 * adapter.scaling
                    dh1 = F.linear(z1_s, adapter.w1_B)
                    if "w1_B" in fmap:
                        _register_pair(unit_slots[fmap["w1_B"]["name"]], z1_s, dh1)

                    z2 = F.linear(x_fp32, adapter.w2_A)
                    if "w2_A" in fmap:
                        _register_pair(unit_slots[fmap["w2_A"]["name"]], x_fp32, z2)
                    z2_s = z2 * adapter.scaling
                    dh2 = F.linear(z2_s, adapter.w2_B)
                    if "w2_B" in fmap:
                        _register_pair(unit_slots[fmap["w2_B"]["name"]], z2_s, dh2)

                    act_full = F.silu(base_h1 + dh1) * (base_h2 + dh2)
                    act_base = F.silu(base_h1) * base_h2
                    base_diff = F.linear(
                        (act_full - act_base).to(dtype=base_w3_weight.dtype), base_w3_weight
                    ).to(torch.float32)

                    z3 = F.linear(act_full, adapter.w3_A)
                    if "w3_A" in fmap:
                        _register_pair(unit_slots[fmap["w3_A"]["name"]], act_full, z3)
                    z3_s = z3 * adapter.scaling
                    lora_w3 = F.linear(z3_s, adapter.w3_B)
                    if "w3_B" in fmap:
                        _register_pair(unit_slots[fmap["w3_B"]["name"]], z3_s, lora_w3)

                    return base_diff + lora_w3

                return _wrapped_forward_delta

            exp.forward_delta = _make_lora_fwd(exp, factor_map)  # type: ignore[method-assign]

        for u in units:
            if u["param"].grad is not None:
                u["param"].grad.zero_()

        loss_sum, n_tokens = forward_fn()
        if loss_sum.requires_grad:
            loss_sum.backward()
    finally:
        for h in hooks:
            h.remove()
        for exp, orig_fwd in patched_adapters:
            exp.forward_delta = orig_fwd  # type: ignore[method-assign]

    captures: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    param_grads: dict[str, torch.Tensor] = {}

    mlp_captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for mid, slots in mlp_slots.items():
        valid = [(s[0], s[1]) for s in slots if s[0] is not None and s[1] is not None]
        if valid:
            mlp_captures[mid] = (
                torch.cat([p[0] for p in valid], dim=0),
                torch.cat([p[1] for p in valid], dim=0),
            )

    for u in units:
        u_name = u["name"]
        p_grad = u["param"].grad
        if p_grad is not None:
            param_grads[u_name] = p_grad.detach().to(device="cpu", dtype=torch.float64) / float(max(1, n_tokens))
            u["param"].grad = None
        else:
            param_grads[u_name] = torch.zeros(tuple(u["param"].shape), dtype=torch.float64)

        valid = [(s[0], s[1]) for s in unit_slots[u_name] if s[0] is not None and s[1] is not None]
        if valid:
            x_cat = torch.cat([p[0] for p in valid], dim=0)
            g_cat = torch.cat([p[1] for p in valid], dim=0)
            # Scale gradient second moment by routing frequency sqrt(N_routed / N_total)
            freq_scale = math.sqrt(float(x_cat.shape[0]) / float(max(1, n_tokens)))
            captures[u_name] = (x_cat, g_cat * freq_scale)
        elif id(u["mlp_module"]) in mlp_captures:
            # Counterfactual prose fallback when expert received zero top-k prose routing hits
            mlp_x, mlp_g = mlp_captures[id(u["mlp_module"])]
            d_out, d_in = int(u["param"].shape[0]), int(u["param"].shape[1])
            has_x = int(mlp_x.shape[1]) == d_in
            has_g = int(mlp_g.shape[1]) == d_out
            n_cf = int(mlp_x.shape[0]) if has_x else (int(mlp_g.shape[0]) if has_g else 1)
            cf_x = mlp_x if has_x else torch.zeros((n_cf, d_in), dtype=torch.float64)
            cf_g = 0.05 * mlp_g if has_g and int(mlp_g.shape[0]) == n_cf else torch.zeros((n_cf, d_out), dtype=torch.float64)
            captures[u_name] = (cf_x, cf_g)

    return captures, param_grads, n_tokens


def _collect_crowd_kronecker_statistics(
    model: nn.Module,
    units: Sequence[dict[str, Any]],
    ids_cal: torch.Tensor,
    expert_ids_cal: Sequence[str],
    *,
    cfg: FisherTrustSurgeonConfig,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Run one prose pass and one specialist pass to gather Fisher pairs and target deltas."""
    prose_mask = torch.tensor([eid.startswith(cfg.general_prefix) for eid in expert_ids_cal], dtype=torch.bool)
    ids_prose = ids_cal[prose_mask]
    ids_spec = ids_cal[~prose_mask]

    orig_req_grad = {id(p): p.requires_grad for p in model.parameters()}
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        for u in units:
            u["param"].requires_grad_(True)

        with torch.enable_grad():
            def _run_prose() -> tuple[torch.Tensor, int]:
                total_loss = torch.zeros((), dtype=torch.float32, device=model.wte.weight.device)
                total_tok = 0
                for b0 in range(0, int(ids_prose.shape[0]), cfg.batch_size):
                    ls, nt = _forward_crowd_model_causal_sum_loss(
                        model, ids_prose[b0 : b0 + cfg.batch_size], score_from=cfg.score_from
                    )
                    total_loss = total_loss + ls
                    total_tok += nt
                return total_loss, total_tok

            prose_cap, _, _ = _capture_linear_pass(units, _run_prose)

            def _run_spec() -> tuple[torch.Tensor, int]:
                total_loss = torch.zeros((), dtype=torch.float32, device=model.wte.weight.device)
                total_tok = 0
                for b0 in range(0, int(ids_spec.shape[0]), cfg.batch_size):
                    ls, nt = _forward_crowd_model_causal_sum_loss(
                        model, ids_spec[b0 : b0 + cfg.batch_size], score_from=cfg.score_from
                    )
                    total_loss = total_loss + ls
                    total_tok += nt
                return total_loss, total_tok

            spec_cap, spec_grads, _ = _capture_linear_pass(units, _run_spec)
    finally:
        for p in model.parameters():
            p.requires_grad_(orig_req_grad.get(id(p), False))

    target_deltas = {name: -grad for name, grad in spec_grads.items()}
    return prose_cap, spec_cap, target_deltas


def _lookup_tensor_for_unit(
    source: torch.Tensor | Mapping[str, torch.Tensor] | None,
    unit: Mapping[str, Any],
    *,
    expected_last_dim: int | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """Resolve explicit per-unit or shared tensor from mapping or single-tensor input."""
    if source is None:
        return None
    if isinstance(source, torch.Tensor):
        if expected_shape is not None:
            return source if tuple(source.shape) == expected_shape else None
        if expected_last_dim is not None:
            return source if int(source.shape[-1]) == expected_last_dim else None
        return source
    if isinstance(source, Mapping):
        for key in (
            unit["name"],
            f"layer_{unit['layer_idx']}.expert_{unit['expert_idx']}.{unit['factor']}",
            f"layer_{unit['layer_idx']}.{unit['factor']}",
            str(unit["factor"]),
        ):
            if key in source:
                return source[key]
        return None
    raise TypeError(f"Expected Tensor, Mapping, or None, got {type(source).__name__}")


def _resolve_p_null_for_unit(
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    unit: Mapping[str, Any],
    shape: tuple[int, int],
    nullspace_side: str,
) -> torch.Tensor | None:
    """Resolve optional nullspace projector compatible with ``unit`` and ``shape``."""
    if p_null is None:
        return None
    d_out, d_in = shape
    if isinstance(p_null, Mapping):
        cand = _lookup_tensor_for_unit(p_null, unit)
        if cand is None:
            return None
        d_p = int(cand.shape[0])
        if (nullspace_side == "right" and d_p != d_in) or (nullspace_side == "left" and d_p != d_out):
            raise ValueError(f"p_null for {unit['name']} has incompatible shape {tuple(cand.shape)}")
        return cand
    if isinstance(p_null, torch.Tensor):
        if p_null.ndim != 2 or p_null.shape[0] != p_null.shape[1]:
            raise ValueError(f"p_null must be a square 2D matrix, got shape {tuple(p_null.shape)}")
        d_p = int(p_null.shape[0])
        if nullspace_side == "right":
            return p_null if d_p == d_in else None
        if nullspace_side == "left":
            return p_null if d_p == d_out else None
        return p_null if d_p in (d_in, d_out) else None
    raise TypeError(f"p_null must be Tensor, Mapping, or None, got {type(p_null).__name__}")


def _resolve_surgery_config(
    config: FisherTrustSurgeonConfig | None,
    *,
    max_kl_nats: float | None,
    damping: float | None,
    scale_damping_by_trace: bool | None,
    kl_budget_scope: str | None,
    nullspace_side: str | None,
    rescale_only_if_exceeds: bool | None,
    use_specialist_ray_damping: bool | None,
    step_damping: float | None,
    max_frobenius_norm: float | None,
    protect_general_expert: bool | None,
    domain_guard_max_ratio: float | None,
    general_guard_max_ratio: float | None,
    enforce_all_domain_guard: bool | None,
    require_energy_improvement: bool | None,
    enable_trust_backtracking: bool | None,
    score_from: int | None,
    inplace: bool | None,
    target_factors: Sequence[str] | None,
) -> FisherTrustSurgeonConfig:
    """Merge optional keyword overrides into a validated ``FisherTrustSurgeonConfig``."""
    if config is not None and not isinstance(config, FisherTrustSurgeonConfig):
        raise TypeError(f"config must be FisherTrustSurgeonConfig or None, got {type(config).__name__}")
    base = config or FisherTrustSurgeonConfig()
    factors = tuple(str(f) for f in target_factors) if target_factors is not None else base.target_factors
    return FisherTrustSurgeonConfig(
        max_kl_nats=base.max_kl_nats if max_kl_nats is None else max_kl_nats,
        damping=base.damping if damping is None else damping,
        scale_damping_by_trace=(
            base.scale_damping_by_trace if scale_damping_by_trace is None else scale_damping_by_trace
        ),
        kl_budget_scope=base.kl_budget_scope if kl_budget_scope is None else kl_budget_scope,
        nullspace_side=base.nullspace_side if nullspace_side is None else nullspace_side,
        rescale_only_if_exceeds=(
            base.rescale_only_if_exceeds if rescale_only_if_exceeds is None else rescale_only_if_exceeds
        ),
        use_specialist_ray_damping=(
            base.use_specialist_ray_damping
            if use_specialist_ray_damping is None
            else use_specialist_ray_damping
        ),
        step_damping=base.step_damping if step_damping is None else step_damping,
        max_frobenius_norm=(
            base.max_frobenius_norm if max_frobenius_norm is None else max_frobenius_norm
        ),
        protect_general_expert=(
            base.protect_general_expert if protect_general_expert is None else protect_general_expert
        ),
        general_expert_index=base.general_expert_index,
        target_factors=factors,
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
        enable_trust_backtracking=(
            base.enable_trust_backtracking
            if enable_trust_backtracking is None
            else enable_trust_backtracking
        ),
        backtrack_halvings=base.backtrack_halvings,
        general_prefix=base.general_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


def _compute_unit_proposals(
    units: Sequence[dict[str, Any]],
    prose_cap: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    spec_cap: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    target_deltas_map: Mapping[str, torch.Tensor],
    *,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    cfg: FisherTrustSurgeonConfig,
) -> tuple[list[dict[str, Any]], float, float]:
    """Compute per-unit natural-gradient trust-region proposals and global KL allocation."""
    proposals: list[dict[str, Any]] = []
    any_pnull_used = False

    for unit in units:
        u_name = unit["name"]
        param: torch.Tensor = unit["param"]
        shape = (int(param.shape[0]), int(param.shape[1]))
        if u_name not in target_deltas_map or u_name not in prose_cap:
            continue

        t_delta = target_deltas_map[u_name]
        if tuple(t_delta.shape) != shape:
            raise ValueError(
                f"target_delta for {u_name} has shape {tuple(t_delta.shape)}, expected {shape}"
            )
        if float(t_delta.detach().abs().max().item()) <= 1e-18:
            continue

        xp_u, gp_u = prose_cap[u_name]
        a_prose, g_prose = estimate_kronecker_fisher_factors(
            xp_u,
            gp_u,
            damping=cfg.damping,
            scale_damping_by_trace=cfg.scale_damping_by_trace,
        )
        p_unit = _resolve_p_null_for_unit(p_null, unit, shape, cfg.nullspace_side)
        if p_unit is not None:
            any_pnull_used = True

        dw_unit, step_diag = solve_kl_trust_region_wsp_step(
            t_delta,
            a_prose,
            g_prose,
            max_kl_nats=cfg.max_kl_nats,
            p_null=p_unit,
            nullspace_side=cfg.nullspace_side,
            rescale_only_if_exceeds=cfg.rescale_only_if_exceeds,
            max_frobenius_norm=cfg.max_frobenius_norm,
        )

        a_spec: torch.Tensor | None = None
        g_spec: torch.Tensor | None = None
        if cfg.use_specialist_ray_damping and u_name in spec_cap:
            xs_u, gs_u = spec_cap[u_name]
            a_spec, g_spec = estimate_kronecker_fisher_factors(
                xs_u,
                gs_u,
                damping=cfg.damping,
                scale_damping_by_trace=cfg.scale_damping_by_trace,
            )

        proposals.append(
            {
                "unit": unit,
                "target_delta": t_delta.detach().to(device="cpu", dtype=torch.float64),
                "delta_w": dw_unit,
                "a_prose": a_prose,
                "g_prose": g_prose,
                "a_spec": a_spec,
                "g_spec": g_spec,
                "step_diag": step_diag,
            }
        )

    if isinstance(p_null, torch.Tensor) and units and not any_pnull_used:
        p_dim = int(p_null.shape[0])
        has_dim_match = any(
            (cfg.nullspace_side == "right" and int(u["param"].shape[1]) == p_dim)
            or (cfg.nullspace_side == "left" and int(u["param"].shape[0]) == p_dim)
            or (cfg.nullspace_side == "auto" and p_dim in (int(u["param"].shape[0]), int(u["param"].shape[1])))
            for u in units
        )
        if not has_dim_match:
            raise ValueError(
                f"p_null shape {tuple(p_null.shape)} did not match any targeted unit factor dimension"
            )

    if not proposals:
        return proposals, 0.0, 0.0

    raw_total_kl = sum(float(p["step_diag"]["kl_nats"]) for p in proposals)
    global_scale = 1.0
    if cfg.kl_budget_scope == "global" and raw_total_kl > 0.0:
        if (not cfg.rescale_only_if_exceeds) or (raw_total_kl > cfg.max_kl_nats):
            global_scale = math.sqrt(cfg.max_kl_nats / raw_total_kl)

    total_kl = 0.0
    total_euc_kl = 0.0
    for prop in proposals:
        dw_scaled = (global_scale * prop["delta_w"]).contiguous()
        t_delta = prop["target_delta"]
        ray_beta = 1.0
        if cfg.use_specialist_ray_damping and prop["a_spec"] is not None and prop["g_spec"] is not None:
            align = max(0.0, float(torch.sum(t_delta * dw_scaled).item()))
            spec_quad = max(
                0.0,
                float(torch.sum(dw_scaled * (prop["g_spec"] @ dw_scaled @ prop["a_spec"])).item()),
            )
            frob2 = float(torch.sum(dw_scaled * dw_scaled).item())
            denom = spec_quad + cfg.step_damping * frob2
            if denom > 1e-18 and align > 0.0:
                ray_beta = min(1.0, align / denom)

        dw_final = (ray_beta * dw_scaled).contiguous()
        frob_final = float(torch.linalg.matrix_norm(dw_final, ord="fro").item())
        if cfg.max_frobenius_norm is not None and frob_final > cfg.max_frobenius_norm:
            cap_scale = cfg.max_frobenius_norm / max(frob_final, 1e-30)
            dw_final = (cap_scale * dw_final).contiguous()
            frob_final = float(torch.linalg.matrix_norm(dw_final, ord="fro").item())
        kl_final = compute_kronecker_kl_nats(dw_final, prop["a_prose"], prop["g_prose"])
        t_frob = float(torch.linalg.matrix_norm(t_delta, ord="fro").item())
        if t_frob > 1e-30 and frob_final > 0.0:
            dw_euc = (frob_final / t_frob) * t_delta
            euc_kl = compute_kronecker_kl_nats(dw_euc, prop["a_prose"], prop["g_prose"])
        else:
            euc_kl = 0.0

        prop["final_delta"] = dw_final
        prop["ray_beta"] = ray_beta
        prop["final_kl_nats"] = kl_final
        prop["final_frob"] = frob_final
        prop["euclidean_kl_nats"] = euc_kl
        total_kl += kl_final
        total_euc_kl += euc_kl

    if cfg.kl_budget_scope == "global" and total_kl > cfg.max_kl_nats:
        fix = math.sqrt(cfg.max_kl_nats / total_kl) * (1.0 - 1e-12)
        total_kl = 0.0
        total_euc_kl = 0.0
        for prop in proposals:
            prop["final_delta"] = (fix * prop["final_delta"]).contiguous()
            prop["final_kl_nats"] = compute_kronecker_kl_nats(
                prop["final_delta"], prop["a_prose"], prop["g_prose"]
            )
            prop["final_frob"] = float(torch.linalg.matrix_norm(prop["final_delta"], ord="fro").item())
            prop["euclidean_kl_nats"] = (fix * fix) * prop["euclidean_kl_nats"]
            total_kl += prop["final_kl_nats"]
            total_euc_kl += prop["euclidean_kl_nats"]

    return proposals, total_kl, total_euc_kl


def _apply_proposals_with_scale(
    proposals: Sequence[Mapping[str, Any]],
    parent_snapshot_by_param_name: Mapping[int, torch.Tensor],
    scale: float,
) -> None:
    """Apply ``parent + scale * final_delta`` cleanly to each unit parameter."""
    # Multiple units never alias the same parameter object in CrowdMoE/MLP topologies.
    for prop in proposals:
        param: torch.Tensor = prop["unit"]["param"]
        base_cpu = parent_snapshot_by_param_name[id(param)]
        dw = (scale * prop["final_delta"]).to(device=param.device, dtype=torch.float64)
        updated = base_cpu.to(device=param.device, dtype=torch.float64) + dw
        param.copy_(updated.to(dtype=param.dtype))


def _evaluate_candidate_state(
    work_model: nn.Module,
    *,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None,
    ids_eval: torch.Tensor | None,
    expert_ids_eval: Sequence[str],
    ref_nll: Mapping[str, float],
    parent_eval: Mapping[str, Any],
    cfg: FisherTrustSurgeonConfig,
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


@torch.no_grad()
def execute_verified_fisher_trust_moe_surgery(
    model: nn.Module,
    calibration_chunks: Sequence[Mapping[str, Any]] | None = None,
    *,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    config: FisherTrustSurgeonConfig | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    target_deltas: Mapping[str, torch.Tensor] | None = None,
    prose_activations: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    prose_grad_outs: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    spec_activations: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    spec_grad_outs: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_factors: Sequence[str] | None = None,
    max_kl_nats: float | None = None,
    damping: float | None = None,
    scale_damping_by_trace: bool | None = None,
    kl_budget_scope: str | None = None,
    nullspace_side: str | None = None,
    rescale_only_if_exceeds: bool | None = None,
    use_specialist_ray_damping: bool | None = None,
    step_damping: float | None = None,
    max_frobenius_norm: float | None = None,
    protect_general_expert: bool | None = None,
    domain_guard_max_ratio: float | None = None,
    general_guard_max_ratio: float | None = None,
    enforce_all_domain_guard: bool | None = None,
    require_energy_improvement: bool | None = None,
    enable_trust_backtracking: bool | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> dict[str, Any]:
    """Execute Fisher-Rao KL trust-region MoE surgery gated by ANSE ``Delta E < 0`` and ``<= +1.0%`` drift.

    Supports full ``CrowdV1WithEngram`` Sparse MoE models with ``calibration_chunks``
    (collecting empirical prose Kronecker Fisher factors and specialist steepest-descent
    directions in one deterministic pass) as well as standalone ``nn.Module`` / MoE /
    linear layers with explicit ``(prose_activations, prose_grad_outs, eval_fn)``.
    Restores the exact parent state bit-for-bit on any guard, KL, or energy failure.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if target_deltas is not None and not isinstance(target_deltas, Mapping):
        raise TypeError("target_deltas must be a mapping or None")

    cfg = _resolve_surgery_config(
        config,
        max_kl_nats=max_kl_nats,
        damping=damping,
        scale_damping_by_trace=scale_damping_by_trace,
        kl_budget_scope=kl_budget_scope,
        nullspace_side=nullspace_side,
        rescale_only_if_exceeds=rescale_only_if_exceeds,
        use_specialist_ray_damping=use_specialist_ray_damping,
        step_damping=step_damping,
        max_frobenius_norm=max_frobenius_norm,
        protect_general_expert=protect_general_expert,
        domain_guard_max_ratio=domain_guard_max_ratio,
        general_guard_max_ratio=general_guard_max_ratio,
        enforce_all_domain_guard=enforce_all_domain_guard,
        require_energy_improvement=require_energy_improvement,
        enable_trust_backtracking=enable_trust_backtracking,
        score_from=score_from,
        inplace=inplace,
        target_factors=target_factors,
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

    target_expert_set = (
        {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
        if expert_indices is not None
        else None
    )
    target_factor_set = set(cfg.target_factors)

    units = _enumerate_target_units(
        mlp_blocks,
        has_crowd_blocks=has_crowd_blocks,
        target_layer_set=target_layer_set,
        target_expert_set=target_expert_set,
        target_factor_set=target_factor_set,
        cfg=cfg,
    )

    ids_cal: torch.Tensor | None = None
    expert_ids_cal: list[str] = []
    ids_eval: torch.Tensor | None = None
    expert_ids_eval: list[str] = []

    if calibration_chunks is not None:
        if not (
            has_crowd_blocks
            and hasattr(work_model, "wte")
            and hasattr(work_model, "ln")
            and hasattr(work_model, "head")
        ):
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
        if eval_fn is None:
            if eval_chunks is None:
                raise ValueError(
                    "Either calibration_chunks or (prose_activations, prose_grad_outs, eval_fn) must be provided"
                )
            if not (
                has_crowd_blocks
                and hasattr(work_model, "wte")
                and hasattr(work_model, "ln")
                and hasattr(work_model, "head")
            ):
                raise TypeError("eval_chunks without eval_fn requires a CrowdV1WithEngram-compatible model")
            ids_eval, expert_ids_eval = _validate_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefix=cfg.general_prefix,
                score_from=cfg.score_from,
                require_both_groups=False,
            )
        if prose_activations is None or prose_grad_outs is None:
            raise ValueError(
                "prose_activations and prose_grad_outs are required when calibration_chunks is None"
            )
        if target_deltas is None and (spec_activations is None or spec_grad_outs is None):
            raise ValueError(
                "Either target_deltas or (spec_activations, spec_grad_outs) must be provided when calibration_chunks is None"
            )

    base_nll_clean = _parse_eval_domain_nll(baseline_domain_nll) if baseline_domain_nll is not None else None
    was_training = work_model.training
    work_model.eval()

    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)
        param_base_by_id = {id(u["param"]): u["param"].detach().to(device="cpu").clone() for u in units}

        if eval_fn is not None:
            pre_nll = _parse_eval_domain_nll(eval_fn(work_model))
        else:
            assert ids_eval is not None
            pre_nll = _eval_crowd_model_domain_nlls(
                work_model,
                ids_eval,
                expert_ids_eval,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )

        ref_nll = base_nll_clean if base_nll_clean is not None else pre_nll
        if set(ref_nll.keys()) != set(pre_nll.keys()):
            raise ValueError("baseline_domain_nll and model domain NLLs must share identical keys")
        parent_eval = _summarize_anse_metrics(ref_nll, pre_nll, cfg=cfg)

        if ids_cal is not None:
            prose_cap, spec_cap, auto_targets = _collect_crowd_kronecker_statistics(
                work_model, units, ids_cal, expert_ids_cal, cfg=cfg
            )
            target_deltas_map = dict(auto_targets)
            if target_deltas is not None:
                for u in units:
                    ovr = _lookup_tensor_for_unit(
                        target_deltas, u, expected_shape=tuple(u["param"].shape)
                    )
                    if ovr is not None:
                        target_deltas_map[u["name"]] = ovr.detach().to(device="cpu", dtype=torch.float64)
        else:
            prose_cap = {}
            spec_cap = {}
            target_deltas_map = {}
            for u in units:
                d_out, d_in = int(u["param"].shape[0]), int(u["param"].shape[1])
                xp_u = _lookup_tensor_for_unit(prose_activations, u, expected_last_dim=d_in)
                gp_u = _lookup_tensor_for_unit(prose_grad_outs, u, expected_last_dim=d_out)
                if xp_u is None or gp_u is None:
                    continue
                prose_cap[u["name"]] = (
                    _validate_sample_matrix(xp_u, f"prose_activations[{u['name']}]")[0],
                    _validate_sample_matrix(gp_u, f"prose_grad_outs[{u['name']}]")[0],
                )
                xs_u = _lookup_tensor_for_unit(spec_activations, u, expected_last_dim=d_in)
                gs_u = _lookup_tensor_for_unit(spec_grad_outs, u, expected_last_dim=d_out)
                if xs_u is not None and gs_u is not None:
                    xs_2d, _ = _validate_sample_matrix(xs_u, f"spec_activations[{u['name']}]")
                    gs_2d, _ = _validate_sample_matrix(gs_u, f"spec_grad_outs[{u['name']}]")
                    if int(xs_2d.shape[0]) != int(gs_2d.shape[0]):
                        raise ValueError(
                            f"spec_activations and spec_grad_outs sample counts must match for {u['name']}"
                        )
                    spec_cap[u["name"]] = (xs_2d, gs_2d)
                    auto_t = -(gs_2d.T @ xs_2d) / float(int(xs_2d.shape[0]))
                else:
                    auto_t = None

                ovr_t = _lookup_tensor_for_unit(target_deltas, u, expected_shape=(d_out, d_in))
                if ovr_t is not None:
                    target_deltas_map[u["name"]] = _validate_finite_2d(ovr_t, f"target_deltas[{u['name']}]")
                elif auto_t is not None:
                    target_deltas_map[u["name"]] = auto_t

        proposals, total_kl_nats, total_euc_kl_nats = _compute_unit_proposals(
            units,
            prose_cap,
            spec_cap,
            target_deltas_map,
            p_null=p_null,
            cfg=cfg,
        )

        scales_to_try = (
            [0.5**i for i in range(cfg.backtrack_halvings + 1)]
            if cfg.enable_trust_backtracking and proposals
            else [1.0]
        )

        chosen_scale = 1.0
        parent_e = float(parent_eval["anse_energy"])
        cand_eval: dict[str, Any] = {}
        eval_error: str | None = None
        cand_e = parent_e
        delta_e = 0.0
        energy_ok = False
        kl_ok = True

        for try_scale in scales_to_try:
            _apply_proposals_with_scale(proposals, param_base_by_id, try_scale)
            cand_eval, eval_error = _evaluate_candidate_state(
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
            eff_kl = (try_scale * try_scale) * total_kl_nats
            kl_bound = cfg.max_kl_nats * float(max(1, len(proposals))) if cfg.kl_budget_scope == "per_unit" else cfg.max_kl_nats
            kl_ok = bool(eff_kl <= kl_bound + 1e-7)
            chosen_scale = try_scale
            if eval_error is None and bool(cand_eval["guard_ok"]) and energy_ok and kl_ok:
                break

        eff_total_kl = (chosen_scale * chosen_scale) * total_kl_nats
        eff_total_euc_kl = (chosen_scale * chosen_scale) * total_euc_kl_nats
        cand_sd = work_model.state_dict()
        candidate_sha = compute_state_sha256(cand_sd)
        total_elements = sum(int(v.numel()) for v in parent_snapshot.values())
        edited_elements = sum(
            int(torch.ne(cand_sd[k].detach().cpu(), parent_snapshot[k]).sum().item())
            for k in parent_snapshot
        )
        edited_param_ratio = float(edited_elements) / float(max(1, total_elements))

        if candidate_sha == parent_sha or not proposals:
            status = "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif eval_error is None and bool(cand_eval["guard_ok"]) and energy_ok and kl_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if eval_error is not None:
                status = "ROLLED_BACK_EVAL_FAILURE"
            elif not kl_ok:
                status = "ROLLED_BACK_KL_VIOLATION"
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

    unit_reports = [
        {
            "unit": prop["unit"]["name"],
            "shape": prop["step_diag"]["shape"],
            "nullspace_mode": prop["step_diag"]["nullspace_mode"],
            "nullspace_residual_norm": float(chosen_scale * prop["step_diag"]["nullspace_residual_norm"]),
            "ray_beta": float(prop["ray_beta"]),
            "kl_nats": float((chosen_scale * chosen_scale) * prop["final_kl_nats"]),
            "euclidean_kl_nats": float((chosen_scale * chosen_scale) * prop["euclidean_kl_nats"]),
            "frobenius_norm": float(chosen_scale * prop["final_frob"]),
        }
        for prop in proposals
    ]

    receipt: dict[str, Any] = {
        "object": SURGEON_RECEIPT_OBJECT,
        "schema": SURGEON_SCHEMA,
        "status": status,
        "accepted": accepted,
        "committed": accepted,
        "rolled_back": rolled_back,
        "eval_error": eval_error,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "kl_ok": kl_ok,
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "mean_diff_vs_base": cand_eval["mean_diff_vs_base"],
        "kl_trust_radius": float(cfg.max_kl_nats),
        "total_kl_nats": float(eff_total_kl),
        "total_euclidean_kl_nats": float(eff_total_euc_kl),
        "backtrack_scale_applied": float(chosen_scale),
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
        "units_targeted": len(units),
        "units_edited": len(proposals) if accepted else 0,
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
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_FISHER_DAMPING",
    "DEFAULT_MAX_KL_NATS",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_STEP_DAMPING",
    "DEFAULT_TARGET_FACTORS",
    "FisherTrustSurgeonConfig",
    "GENERAL_EXPERT_PREFIX",
    "SURGEON_RECEIPT_OBJECT",
    "SURGEON_SCHEMA",
    "VALID_KL_BUDGET_SCOPES",
    "VALID_NULLSPACE_SIDES",
    "VALID_TARGET_FACTORS",
    "canonical_json_bytes",
    "compute_kronecker_kl_nats",
    "compute_payload_sha256",
    "compute_state_sha256",
    "estimate_kronecker_fisher_factors",
    "execute_verified_fisher_trust_moe_surgery",
    "solve_kl_trust_region_wsp_step",
    "verify_fisher_trust_receipt_provenance",
]
