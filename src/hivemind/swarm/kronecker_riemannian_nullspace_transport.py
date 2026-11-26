# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Kronecker-Factored Riemannian Null-Space Transport Surgeon for Curved Deep Layers.

Implements **Gen-3 Direction D2** (Zenodo ``10.5281/zenodo.24542441`` Section 6 /
Theorem 3.7 and Lean 4 theorem
``ANSE.ReineCortexDirections.thm_rcd_d2_kronecker_riemannian_nullspace_descent``):

1. **Damped Kronecker Whitened Null-Space Operators**
   (:func:`compute_kronecker_whitened_nullspace_operators`):
   Estimates trace-scaled shrinkage-damped Kronecker empirical Fisher / covariance
   factors ``(A_cov, G_cov)`` in CPU ``float64`` (avoiding undamped ``1e-8``
   ill-conditioning), maps prose singular modes ``V_0`` into whitened metric
   coordinates ``U_A = QR(A_cov^{-1/2} V_0)``, and constructs both:

   - ``R_step = A_cov^{-1/2} (I - U_A U_A^T) A_cov^{-1/2}`` (Riemannian inverse-metric
     null-space operator guaranteeing exact prose annihilation ``Delta W V_0 = 0`` and
     unconditional specialist descent ``<G_spec, -G_cov^{-1} G_spec R_step>_F <= 0``),
   - ``R_proj = A_cov^{1/2} (I - U_A U_A^T) A_cov^{-1/2}`` (idempotent oblique
     null-space transport projector minimizing ``(G_cov (x) A_cov)`` metric distortion).

2. **Geodesic KL Trust-Region Transport & Curved-Layer Parallel Transport**
   (:func:`solve_kronecker_riemannian_transport_step`,
   :func:`transport_delta_between_curved_layers`,
   :func:`transport_factorized_lora_with_kronecker_metric`,
   :func:`diagnose_theorem_3_7_obstruction`):
   Resolves Theorem 3.7's non-commuting uphill obstruction, performs isometric
   layer-to-layer metric transport ``G_dst^{-1/2} G_src^{1/2} dW A_src^{1/2} A_dst^{-1/2}``,
   preserves factorized LoRA rank ``r`` via ``A @ R_proj``, and clamps updates onto the
   geodesic Fisher-Rao KL trust ellipsoid ``1/2 tr(G_prose dW A_prose dW^T) <= max_kl_nats``.

3. **ANSE-Gated MoE & Deep-Layer Surgery**
   (:func:`execute_verified_kronecker_riemannian_surgery`):
   Executes deterministic CPU surgery (``0 MiB`` VRAM, ``0`` gradient steps) across
   ``CrowdV1WithEngram``, HF-style transformer layers, and ``nn.Linear`` blocks under
   ``ANSE Delta E < 0`` and ``D0 / worst_domain <= +1.0%`` with atomic bitwise
   rollback under ``ANSE_MAXIMUM_PAIN`` (``1e6``) and canonical SHA-256 receipts.
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

from hivemind.swarm.natural_gradient_trust_wsp_surgeon import (
    ANSE_MAXIMUM_PAIN,
    _capture_linear_pass,
    _discover_mlp_blocks,
    _enumerate_target_units,
    _eval_crowd_model_domain_nlls,
    _forward_crowd_model_causal_sum_loss,
    _lookup_tensor_for_unit,
    _parse_eval_domain_nll,
    _validate_bool,
    _validate_chunks,
    _validate_finite_2d,
    _validate_nonneg_finite,
    _validate_nonneg_int,
    _validate_positive_finite,
    _validate_positive_int,
    _validate_sample_matrix,
    _validate_spd_matrix,
    canonical_json_bytes,
    compute_kronecker_kl_nats,
    compute_payload_sha256,
    compute_state_sha256,
    estimate_kronecker_fisher_factors,
)

KRONECKER_TRANSPORT_SCHEMA: str = "hivemind.kronecker_riemannian_nullspace_transport/1"
KRONECKER_TRANSPORT_RECEIPT_OBJECT: str = "hivemind.kronecker_riemannian_transport_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
DEFAULT_DAMPING_A: float = 0.05
DEFAULT_DAMPING_G: float = 0.05
DEFAULT_RIDGE_FLOOR: float = 1e-6
DEFAULT_SV_REL_TOL: float = 0.12
DEFAULT_MAX_KL_NATS: float = 0.005
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 0

VALID_TRANSPORT_MODES: frozenset[str] = frozenset({"riemannian_gradient", "whitened_lora_transport"})
VALID_KL_BUDGET_SCOPES: frozenset[str] = frozenset({"global", "per_unit"})
VALID_METRIC_SOURCES: frozenset[str] = frozenset({"prose", "all"})
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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q",
        "k",
        "v",
        "o",
        "head",
        "lm_head",
        "weight",
    }
)
DEFAULT_TARGET_FACTORS: tuple[str, ...] = (
    "w1_A",
    "w2_A",
    "w3_A",
    "w3_B",
    "w3",
    "down_proj",
    "q_proj",
    "v_proj",
    "v",
    "o",
    "head",
    "lm_head",
    "weight",
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTN_FACTORS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "q", "k", "v", "o"})
_MLP_FACTORS = frozenset(
    {"w1", "w2", "w3", "w1_A", "w2_A", "w3_A", "w1_B", "w2_B", "w3_B", "gate_proj", "up_proj", "down_proj", "weight"}
)


# ---------------------------------------------------------------------------
# 1. Configuration, Validation & Receipt Verification
# ---------------------------------------------------------------------------


def verify_kronecker_riemannian_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        receipt.get("object") != KRONECKER_TRANSPORT_RECEIPT_OBJECT
        or receipt.get("schema") != KRONECKER_TRANSPORT_SCHEMA
    ):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class KroneckerRiemannianTransportConfig:
    """Validated configuration for Kronecker-factored Riemannian null-space transport."""

    damping_a: float = DEFAULT_DAMPING_A
    damping_g: float = DEFAULT_DAMPING_G
    ridge_floor: float = DEFAULT_RIDGE_FLOOR
    scale_damping_by_trace: bool = True
    rank_keep: int | None = None
    sv_rel_tol: float = DEFAULT_SV_REL_TOL
    max_kl_nats: float = DEFAULT_MAX_KL_NATS
    kl_budget_scope: str = "global"
    transport_mode: str = "riemannian_gradient"
    metric_source: str = "prose"
    rescale_only_if_exceeds: bool = False
    max_frobenius_norm: float | None = None
    sequential_refresh: bool = False
    protect_general_expert: bool = True
    general_expert_index: int = 0
    target_factors: tuple[str, ...] = DEFAULT_TARGET_FACTORS
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    general_guard_max_ratio: float | None = None
    enforce_all_domain_guard: bool = True
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    require_energy_improvement: bool = True
    enable_trust_backtracking: bool = True
    backtrack_halvings: int = 4
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 16
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_positive_finite(self.damping_a, "damping_a")
        _validate_positive_finite(self.damping_g, "damping_g")
        _validate_nonneg_finite(self.ridge_floor, "ridge_floor")
        _validate_bool(self.scale_damping_by_trace, "scale_damping_by_trace")
        if self.rank_keep is not None:
            _validate_nonneg_int(self.rank_keep, "rank_keep")
        _validate_nonneg_finite(self.sv_rel_tol, "sv_rel_tol")
        _validate_positive_finite(self.max_kl_nats, "max_kl_nats")
        if self.kl_budget_scope not in VALID_KL_BUDGET_SCOPES:
            raise ValueError(f"kl_budget_scope must be one of {sorted(VALID_KL_BUDGET_SCOPES)}")
        if self.transport_mode not in VALID_TRANSPORT_MODES:
            raise ValueError(f"transport_mode must be one of {sorted(VALID_TRANSPORT_MODES)}")
        if self.metric_source not in VALID_METRIC_SOURCES:
            raise ValueError(f"metric_source must be one of {sorted(VALID_METRIC_SOURCES)}")
        _validate_bool(self.rescale_only_if_exceeds, "rescale_only_if_exceeds")
        if self.max_frobenius_norm is not None:
            _validate_positive_finite(self.max_frobenius_norm, "max_frobenius_norm")
        _validate_bool(self.sequential_refresh, "sequential_refresh")
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
        if (
            isinstance(self.general_prefixes, (str, bytes))
            or not isinstance(self.general_prefixes, Sequence)
            or not self.general_prefixes
            or not all(isinstance(p, str) and p.strip() for p in self.general_prefixes)
        ):
            raise ValueError("general_prefixes must be a non-empty sequence of non-empty strings")
        object.__setattr__(self, "general_prefixes", tuple(str(p).strip() for p in self.general_prefixes))
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Kronecker Riemannian Null-Space Operators & Geodesic Transport Primitives
# ---------------------------------------------------------------------------


def _spd_spectral_powers(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute ``(M^{1/2}, M^{-1/2}, M^{-1}, cond)`` for a symmetric PD matrix in ``float64``."""
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = torch.linalg.eigh(sym)
    clamped = evals.clamp_min(1e-12)
    m_sqrt = 0.5 * ((evecs * clamped.sqrt().unsqueeze(0)) @ evecs.T + ((evecs * clamped.sqrt().unsqueeze(0)) @ evecs.T).T)
    m_inv_sqrt = 0.5 * ((evecs * clamped.rsqrt().unsqueeze(0)) @ evecs.T + ((evecs * clamped.rsqrt().unsqueeze(0)) @ evecs.T).T)
    m_inv = 0.5 * ((evecs * (1.0 / clamped).unsqueeze(0)) @ evecs.T + ((evecs * (1.0 / clamped).unsqueeze(0)) @ evecs.T).T)
    cond = float(clamped[-1].item() / clamped[0].item())
    return m_sqrt.contiguous(), m_inv_sqrt.contiguous(), m_inv.contiguous(), cond


@torch.no_grad()
def compute_kronecker_whitened_nullspace_operators(
    x_prose: torch.Tensor,
    *,
    x_metric: torch.Tensor | None = None,
    grad_out_metric: torch.Tensor | None = None,
    rank_keep: int | None = None,
    sv_rel_tol: float = DEFAULT_SV_REL_TOL,
    damping_a: float = DEFAULT_DAMPING_A,
    damping_g: float = DEFAULT_DAMPING_G,
    ridge_floor: float = DEFAULT_RIDGE_FLOOR,
    scale_damping_by_trace: bool = True,
) -> dict[str, Any]:
    """Build damped Kronecker-whitened Riemannian null-space operators in CPU ``float64``.

    Constructs damped empirical Fisher / covariance matrices ``(A_cov, G_cov)``,
    extracts top prose singular directions ``V_0``, orthonormalizes ``A_cov^{-1/2} V_0``
    into ``U_A`` via reduced QR, and returns:
    - ``r_step = A_cov^{-1/2} (I - U_A U_A^T) A_cov^{-1/2}`` (Riemannian descent operator),
    - ``r_proj = A_cov^{1/2} (I - U_A U_A^T) A_cov^{-1/2}`` (idempotent oblique projector).
    """
    xp_2d, _ = _validate_sample_matrix(x_prose, "x_prose")
    d_in = int(xp_2d.shape[1])

    if x_metric is not None:
        xm_2d, _ = _validate_sample_matrix(x_metric, "x_metric")
        if int(xm_2d.shape[1]) != d_in:
            raise ValueError(f"x_metric width {int(xm_2d.shape[1])} != x_prose width {d_in}")
    else:
        xm_2d = xp_2d

    damp_a = _validate_positive_finite(damping_a, "damping_a")
    damp_g = _validate_positive_finite(damping_g, "damping_g")
    floor_f = _validate_nonneg_finite(ridge_floor, "ridge_floor")
    sv_tol = _validate_nonneg_finite(sv_rel_tol, "sv_rel_tol")
    scale_tr = _validate_bool(scale_damping_by_trace, "scale_damping_by_trace")
    r_keep = _validate_nonneg_int(rank_keep, "rank_keep") if rank_keep is not None else None

    a_prose, _ = estimate_kronecker_fisher_factors(
        xp_2d,
        xp_2d[:, :1],
        damping=damp_a,
        ridge_floor=floor_f,
        scale_damping_by_trace=scale_tr,
    )
    if grad_out_metric is not None:
        gm_2d, _ = _validate_sample_matrix(grad_out_metric, "grad_out_metric")
        a_cov, g_tmp = estimate_kronecker_fisher_factors(
            xm_2d,
            gm_2d,
            damping=damp_a,
            ridge_floor=floor_f,
            scale_damping_by_trace=scale_tr,
        )
        if abs(damp_g - damp_a) <= 1e-15:
            g_cov = g_tmp
        else:
            g_cov, _ = estimate_kronecker_fisher_factors(
                gm_2d,
                xm_2d,
                damping=damp_g,
                ridge_floor=floor_f,
                scale_damping_by_trace=scale_tr,
            )
        d_out: int | None = int(g_cov.shape[0])
        g_sqrt, g_inv_sqrt, g_inv, cond_g = _spd_spectral_powers(g_cov)
    else:
        a_cov, _ = estimate_kronecker_fisher_factors(
            xm_2d,
            xm_2d[:, :1],
            damping=damp_a,
            ridge_floor=floor_f,
            scale_damping_by_trace=scale_tr,
        )
        g_cov = None
        d_out = None
        g_sqrt = None
        g_inv_sqrt = None
        g_inv = None
        cond_g = 1.0

    a_sqrt, a_inv_sqrt, a_inv, cond_a = _spd_spectral_powers(a_cov)

    _, s0, vh0 = torch.linalg.svd(xp_2d, full_matrices=False)
    if r_keep is not None:
        k_eff = min(r_keep, int(vh0.shape[0]), d_in)
    elif s0.numel() > 0 and float(s0[0].item()) > 1e-12:
        n_above = int((s0 > sv_tol * float(s0[0].item())).sum().item())
        k_eff = min(max(0, d_in - 1), max(1, n_above)) if d_in > 1 else 0
    else:
        k_eff = 0

    eye_in = torch.eye(d_in, dtype=torch.float64)
    if k_eff == 0:
        v0 = torch.zeros((d_in, 0), dtype=torch.float64)
        u_a = torch.zeros((d_in, 0), dtype=torch.float64)
        p_euc = eye_in
        p_wh = eye_in
        r_step = a_inv
        r_proj = eye_in
    elif k_eff >= d_in:
        v0 = vh0[:d_in, :].T.contiguous()
        u_a, _ = torch.linalg.qr(a_inv_sqrt @ v0, mode="reduced")
        p_euc = torch.zeros((d_in, d_in), dtype=torch.float64)
        p_wh = torch.zeros((d_in, d_in), dtype=torch.float64)
        r_step = torch.zeros((d_in, d_in), dtype=torch.float64)
        r_proj = torch.zeros((d_in, d_in), dtype=torch.float64)
    else:
        v0 = vh0[:k_eff, :].T.contiguous()
        u_a, _ = torch.linalg.qr(a_inv_sqrt @ v0, mode="reduced")
        p_euc = 0.5 * ((eye_in - v0 @ v0.T) + (eye_in - v0 @ v0.T).T)
        p_wh = 0.5 * ((eye_in - u_a @ u_a.T) + (eye_in - u_a @ u_a.T).T)
        r_step_raw = a_inv_sqrt @ p_wh @ a_inv_sqrt
        r_step = 0.5 * (r_step_raw + r_step_raw.T)
        r_proj = (a_sqrt @ p_wh @ a_inv_sqrt).contiguous()

    idem_err = float(torch.linalg.matrix_norm(r_proj @ r_proj - r_proj, ord="fro").item())
    ann_step = float(torch.linalg.matrix_norm(r_step @ v0, ord="fro").item()) if k_eff > 0 else 0.0
    ann_proj = float(torch.linalg.matrix_norm(r_proj @ v0, ord="fro").item()) if k_eff > 0 else 0.0

    return {
        "d_in": d_in,
        "d_out": d_out,
        "rank_used": k_eff,
        "a_cov": a_cov,
        "a_prose": a_prose,
        "a_sqrt": a_sqrt,
        "a_inv_sqrt": a_inv_sqrt,
        "a_inv": a_inv,
        "g_cov": g_cov,
        "g_sqrt": g_sqrt,
        "g_inv_sqrt": g_inv_sqrt,
        "g_inv": g_inv,
        "v0": v0,
        "u_a": u_a,
        "p_euc": p_euc.contiguous(),
        "p_wh": p_wh.contiguous(),
        "r_step": r_step.contiguous(),
        "r_proj": r_proj.contiguous(),
        "diagnostics": {
            "d_in": d_in,
            "d_out": d_out,
            "rank_used": k_eff,
            "cond_a": cond_a,
            "cond_g": cond_g,
            "r_proj_idempotence_error": idem_err,
            "r_step_prose_annihilation": ann_step,
            "r_proj_prose_annihilation": ann_proj,
        },
    }


def _resolve_kl_metrics(
    d_out: int,
    d_in: int,
    operators: Mapping[str, Any],
    a_prose: torch.Tensor | None,
    g_prose: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and resolve output metric inverses alongside prose KL matrices."""
    g_inv_op = operators.get("g_inv")
    g_inv_sqrt_op = operators.get("g_inv_sqrt")
    if g_inv_op is not None and g_inv_sqrt_op is not None:
        if tuple(g_inv_op.shape) != (d_out, d_out):
            raise ValueError(
                f"Operator g_inv shape {tuple(g_inv_op.shape)} != ({d_out}, {d_out})"
            )
        g_inv = g_inv_op
        g_inv_sqrt = g_inv_sqrt_op
    else:
        g_inv = torch.eye(d_out, dtype=torch.float64)
        g_inv_sqrt = g_inv

    if a_prose is not None:
        a_kl, _ = _validate_spd_matrix(a_prose, d_in, "a_prose")
    else:
        a_kl = operators["a_prose"]

    if g_prose is not None:
        g_kl, _ = _validate_spd_matrix(g_prose, d_out, "g_prose")
    elif operators.get("g_cov") is not None:
        g_kl = operators["g_cov"]
    else:
        g_kl = torch.eye(d_out, dtype=torch.float64)

    return g_inv, g_inv_sqrt, a_kl, g_kl


def _clamp_step_to_trust_ellipsoid(
    raw_step: torch.Tensor,
    a_kl: torch.Tensor,
    g_kl: torch.Tensor,
    *,
    kl_cap: float,
    rescale_only_if_exceeds: bool,
    frob_cap: float | None,
) -> tuple[torch.Tensor, float, float, float]:
    """Scale ``raw_step`` onto the Fisher-Rao geodesic KL ellipsoid (and Frobenius cap)."""
    raw_kl = compute_kronecker_kl_nats(raw_step, a_kl, g_kl)
    raw_frob = float(torch.linalg.matrix_norm(raw_step, ord="fro").item())
    if raw_kl <= 1e-30 or raw_frob <= 1e-30:
        return torch.zeros_like(raw_step), 0.0, 0.0, 0.0

    trust_scale = (
        min(1.0, math.sqrt(kl_cap / raw_kl)) if rescale_only_if_exceeds else math.sqrt(kl_cap / raw_kl)
    )
    if frob_cap is not None and raw_frob * trust_scale > frob_cap:
        trust_scale = frob_cap / raw_frob

    delta_out = (trust_scale * raw_step).contiguous()
    kl_after = compute_kronecker_kl_nats(delta_out, a_kl, g_kl)
    if kl_after > kl_cap:
        fix = math.sqrt(kl_cap / kl_after) * (1.0 - 1e-12)
        trust_scale *= fix
        delta_out = (trust_scale * raw_step).contiguous()
        kl_after = compute_kronecker_kl_nats(delta_out, a_kl, g_kl)
    frob_after = float(torch.linalg.matrix_norm(delta_out, ord="fro").item())
    return delta_out, trust_scale, kl_after, frob_after


@torch.no_grad()
def solve_kronecker_riemannian_transport_step(
    target_delta: torch.Tensor,
    operators: Mapping[str, Any],
    *,
    a_prose: torch.Tensor | None = None,
    g_prose: torch.Tensor | None = None,
    max_kl_nats: float = DEFAULT_MAX_KL_NATS,
    transport_mode: str = "riemannian_gradient",
    rescale_only_if_exceeds: bool = False,
    max_frobenius_norm: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve Kronecker-whitened Riemannian null-space transport inside geodesic KL trust radius.

    ``target_delta`` denotes the desired update direction (e.g., ``-G_spec`` or task
    vector ``Delta W``). Returns ``(delta_out_float64, diagnostics)`` satisfying exact
    prose annihilation ``||delta_out V_0||_F == 0`` and ``KL_prose <= max_kl_nats``.
    """
    target64 = _validate_finite_2d(target_delta, "target_delta")
    d_out, d_in = int(target64.shape[0]), int(target64.shape[1])
    if d_in != int(operators["d_in"]):
        raise ValueError(f"target_delta d_in={d_in} != operator d_in={operators['d_in']}")
    if transport_mode not in VALID_TRANSPORT_MODES:
        raise ValueError(f"transport_mode must be one of {sorted(VALID_TRANSPORT_MODES)}")

    kl_cap = _validate_positive_finite(max_kl_nats, "max_kl_nats")
    shrink_only = _validate_bool(rescale_only_if_exceeds, "rescale_only_if_exceeds")
    frob_cap = (
        _validate_positive_finite(max_frobenius_norm, "max_frobenius_norm")
        if max_frobenius_norm is not None
        else None
    )

    g_inv, g_inv_sqrt, a_kl, g_kl = _resolve_kl_metrics(d_out, d_in, operators, a_prose, g_prose)

    if transport_mode == "riemannian_gradient":
        raw_step = (g_inv @ target64 @ operators["r_step"]).contiguous()
    else:
        raw_step = (g_inv_sqrt @ target64 @ operators["r_proj"]).contiguous()

    delta_out, trust_scale, kl_after, frob_after = _clamp_step_to_trust_ellipsoid(
        raw_step,
        a_kl,
        g_kl,
        kl_cap=kl_cap,
        rescale_only_if_exceeds=shrink_only,
        frob_cap=frob_cap,
    )

    euc_raw = target64 @ operators["p_euc"]
    euc_frob = float(torch.linalg.matrix_norm(euc_raw, ord="fro").item())
    if euc_frob > 1e-30 and frob_after > 0.0:
        euc_matched = (frob_after / euc_frob) * euc_raw
        euc_kl = compute_kronecker_kl_nats(euc_matched, a_kl, g_kl)
    else:
        euc_kl = 0.0

    v0 = operators["v0"]
    prose_leak = (
        float(torch.linalg.matrix_norm(delta_out @ v0, ord="fro").item())
        if int(operators["rank_used"]) > 0
        else 0.0
    )
    target_align = float(torch.sum(target64 * delta_out).item())
    kl_ratio = kl_after / max(euc_kl, 1e-30) if euc_kl > 0.0 else 0.0

    return delta_out, {
        "shape": (d_out, d_in),
        "transport_mode": transport_mode,
        "kl_trust_radius": kl_cap,
        "kl_nats": kl_after,
        "euclidean_p0_kl_nats": euc_kl,
        "kl_ratio_vs_euclidean_p0": kl_ratio,
        "trust_scale": trust_scale,
        "frobenius_norm": frob_after,
        "target_alignment": target_align,
        "prose_nullspace_leak": prose_leak,
        "kl_bound_satisfied": bool(kl_after <= kl_cap + 1e-12),
    }


@torch.no_grad()
def transport_delta_between_curved_layers(
    delta_src: torch.Tensor,
    ops_src: Mapping[str, Any],
    ops_dst: Mapping[str, Any],
    *,
    max_kl_nats: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Isometrically parallel-transport a weight update between curved layer metrics.

    Applies ``G_dst^{-1/2} G_src^{1/2} delta_src A_src^{1/2} A_dst^{-1/2}`` (preserving
    Kronecker Fisher-Rao norm prior to null-space projection), projects onto target
    prose null-space via ``r_proj_dst``, and optionally clamps geodesic KL on ``ops_dst``.
    """
    dw64 = _validate_finite_2d(delta_src, "delta_src")
    d_out, d_in = int(dw64.shape[0]), int(dw64.shape[1])
    if d_in != int(ops_src["d_in"]) or d_in != int(ops_dst["d_in"]):
        raise ValueError("Source and destination operators must match delta_src input width")

    a_src_sqrt = ops_src["a_sqrt"]
    a_dst_inv_sqrt = ops_dst["a_inv_sqrt"]
    if ops_src.get("g_sqrt") is not None and ops_dst.get("g_inv_sqrt") is not None:
        if tuple(ops_src["g_sqrt"].shape) != (d_out, d_out) or tuple(ops_dst["g_inv_sqrt"].shape) != (
            d_out,
            d_out,
        ):
            raise ValueError("Output Fisher operators must match delta_src output rows")
        g_left = ops_dst["g_inv_sqrt"] @ ops_src["g_sqrt"]
        g_src_kl = ops_src["g_cov"]
        g_dst_kl = ops_dst["g_cov"]
    else:
        g_left = torch.eye(d_out, dtype=torch.float64)
        g_src_kl = g_left
        g_dst_kl = g_left

    iso = (g_left @ dw64 @ a_src_sqrt @ a_dst_inv_sqrt).contiguous()
    src_kl = compute_kronecker_kl_nats(dw64, ops_src["a_cov"], g_src_kl)
    iso_kl_dst = compute_kronecker_kl_nats(iso, ops_dst["a_cov"], g_dst_kl)

    projected = (iso @ ops_dst["r_proj"]).contiguous()
    if max_kl_nats is not None:
        kl_cap = _validate_positive_finite(max_kl_nats, "max_kl_nats")
        projected, scale_used, final_kl, _ = _clamp_step_to_trust_ellipsoid(
            projected,
            ops_dst["a_prose"],
            g_dst_kl,
            kl_cap=kl_cap,
            rescale_only_if_exceeds=True,
            frob_cap=None,
        )
    else:
        scale_used = 1.0
        final_kl = compute_kronecker_kl_nats(projected, ops_dst["a_prose"], g_dst_kl)

    v0_dst = ops_dst["v0"]
    dst_leak = (
        float(torch.linalg.matrix_norm(projected @ v0_dst, ord="fro").item())
        if int(ops_dst["rank_used"]) > 0
        else 0.0
    )
    return projected, {
        "source_metric_kl": src_kl,
        "isometric_transport_kl_dst": iso_kl_dst,
        "isometry_rel_error": abs(iso_kl_dst - src_kl) / max(src_kl, 1e-12),
        "projected_prose_kl_dst": final_kl,
        "trust_scale_applied": scale_used,
        "dst_prose_nullspace_leak": dst_leak,
    }


@torch.no_grad()
def transport_factorized_lora_with_kronecker_metric(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    operators: Mapping[str, Any],
    *,
    base_weight: torch.Tensor | None = None,
    balance_output_rows: bool = True,
    projection_kind: str = "riemannian_metric",
    match_euclidean_norm: bool = True,
    row_clamp_range: tuple[float, float] = (0.75, 1.35),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply rank-preserving Kronecker null-space transport on factorized LoRA ``(B, A)``.

    Supports ``projection_kind='riemannian_metric'`` (using ``R_step`` + ``G_cov^{-1}``
    matched to Euclidean Frobenius norm so curved prose KL strictly contracts) and
    ``projection_kind='oblique_proj'`` (using ``R_proj`` + balanced row scaling).
    """
    a64 = _validate_finite_2d(lora_a, "lora_a")
    b64 = _validate_finite_2d(lora_b, "lora_b")
    sc_f = _validate_positive_finite(scaling, "scaling")
    balance = _validate_bool(balance_output_rows, "balance_output_rows")
    match_norm = _validate_bool(match_euclidean_norm, "match_euclidean_norm")
    if projection_kind not in {"riemannian_metric", "oblique_proj"}:
        raise ValueError("projection_kind must be 'riemannian_metric' or 'oblique_proj'")
    if a64.shape[0] != b64.shape[1]:
        raise ValueError(f"LoRA rank mismatch: lora_a {tuple(a64.shape)} vs lora_b {tuple(b64.shape)}")
    if int(a64.shape[1]) != int(operators["d_in"]):
        raise ValueError(f"lora_a width {a64.shape[1]} != operator d_in={operators['d_in']}")

    right_op = operators["r_step"] if projection_kind == "riemannian_metric" else operators["r_proj"]
    a_kron = (a64 @ right_op).contiguous()
    d_out = int(b64.shape[0])
    bw64: torch.Tensor | None = None
    if base_weight is not None:
        bw64 = _validate_finite_2d(base_weight, "base_weight")
        if bw64.shape != (d_out, int(a64.shape[1])):
            raise ValueError("base_weight shape must match (d_out, d_in)")
    has_g = operators.get("g_inv") is not None and tuple(operators["g_inv"].shape) == (d_out, d_out)

    if balance:
        if projection_kind == "riemannian_metric" and has_g:
            b_kron = (operators["g_inv"] @ b64).contiguous()
        elif bw64 is not None:
            row_var = bw64.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
            g_scale = (row_var.mean() / row_var).sqrt().clamp(row_clamp_range[0], row_clamp_range[1])
            b_kron = (g_scale * b64).contiguous()
        elif has_g:
            g_diag = torch.diagonal(operators["g_cov"]).clamp_min(1e-8).unsqueeze(1)
            g_scale = (g_diag.mean() / g_diag).sqrt().clamp(row_clamp_range[0], row_clamp_range[1])
            b_kron = (g_scale * b64).contiguous()
        else:
            b_kron = b64.clone()
    else:
        b_kron = b64.clone()

    dw_euc = (sc_f * b64) @ (a64 @ operators["p_euc"])
    dw_kron = (sc_f * b_kron) @ a_kron
    if match_norm:
        euc_nrm = float(torch.linalg.matrix_norm(dw_euc, ord="fro").item())
        kron_nrm = float(torch.linalg.matrix_norm(dw_kron, ord="fro").item())
        if euc_nrm > 1e-30 and kron_nrm > 1e-30:
            factor_s = math.sqrt(euc_nrm / kron_nrm)
            a_kron = (factor_s * a_kron).contiguous()
            b_kron = (factor_s * b_kron).contiguous()
            dw_kron = (sc_f * b_kron) @ a_kron

    g_kl = operators["g_cov"] if has_g else torch.eye(d_out, dtype=torch.float64)
    kron_kl = compute_kronecker_kl_nats(dw_kron, operators["a_prose"], g_kl)
    euc_kl = compute_kronecker_kl_nats(dw_euc, operators["a_prose"], g_kl)
    v0 = operators["v0"]
    leak = float(torch.linalg.matrix_norm(dw_kron @ v0, ord="fro").item()) if int(operators["rank_used"]) > 0 else 0.0

    return a_kron.to(dtype=lora_a.dtype), b_kron.to(dtype=lora_b.dtype), {
        "rank": int(a64.shape[0]),
        "projection_kind": projection_kind,
        "kronecker_prose_kl_nats": kron_kl,
        "euclidean_p0_prose_kl_nats": euc_kl,
        "prose_nullspace_leak": leak,
    }


@torch.no_grad()
def diagnose_theorem_3_7_obstruction(
    g_spec: torch.Tensor,
    x_prose: torch.Tensor,
    *,
    x_metric: torch.Tensor | None = None,
    grad_out_metric: torch.Tensor | None = None,
    rank_keep: int | None = None,
    sv_rel_tol: float = DEFAULT_SV_REL_TOL,
    damping_a: float = DEFAULT_DAMPING_A,
    damping_g: float = DEFAULT_DAMPING_G,
    max_kl_nats: float = DEFAULT_MAX_KL_NATS,
) -> dict[str, Any]:
    """Compare Theorem 3.7 regimes: unwhitened aniso + ``P_0``, flat ``P_0``, undamped K-FAC, and damped Kronecker."""
    g_sp64 = _validate_finite_2d(g_spec, "g_spec")
    ops_damped = compute_kronecker_whitened_nullspace_operators(
        x_prose,
        x_metric=x_metric,
        grad_out_metric=grad_out_metric,
        rank_keep=rank_keep,
        sv_rel_tol=sv_rel_tol,
        damping_a=damping_a,
        damping_g=damping_g,
    )
    ops_undamped = compute_kronecker_whitened_nullspace_operators(
        x_prose,
        x_metric=x_metric,
        grad_out_metric=grad_out_metric,
        rank_keep=rank_keep,
        sv_rel_tol=sv_rel_tol,
        damping_a=1e-8,
        damping_g=1e-8,
        ridge_floor=1e-12,
        scale_damping_by_trace=False,
    )

    d_out = int(g_sp64.shape[0])
    if int(g_sp64.shape[1]) != int(ops_damped["d_in"]):
        raise ValueError(f"g_spec d_in={int(g_sp64.shape[1])} != ops_damped d_in={ops_damped['d_in']}")
    g_kl = (
        ops_damped["g_cov"]
        if ops_damped["g_cov"] is not None
        else torch.eye(d_out, dtype=torch.float64)
    )
    a_kl = ops_damped["a_prose"]
    v0 = ops_damped["v0"]

    # 1. Damped Kronecker Riemannian null-space step (-G_cov^{-1} G_spec R_step)
    dw_kron, diag_kron = solve_kronecker_riemannian_transport_step(
        -g_sp64,
        ops_damped,
        max_kl_nats=max_kl_nats,
        transport_mode="riemannian_gradient",
    )
    frob_ref = max(float(diag_kron["frobenius_norm"]), 1e-12)

    def _summarize(step_mat: torch.Tensor) -> dict[str, float]:
        nrm = float(torch.linalg.matrix_norm(step_mat, ord="fro").item())
        matched = (frob_ref / nrm) * step_mat if nrm > 1e-30 else step_mat
        dir_deriv = float(torch.sum(g_sp64 * matched).item())
        kl_val = compute_kronecker_kl_nats(matched, a_kl, g_kl)
        leak = float(torch.linalg.matrix_norm(matched @ v0, ord="fro").item()) if ops_damped["rank_used"] > 0 else 0.0
        return {
            "spec_directional_deriv": dir_deriv,
            "prose_kl_nats": kl_val,
            "prose_nullspace_leak": leak,
        }

    # 2. Unwhitened diagonal anisotropic preconditioning followed by flat Euclidean P_0
    col_scale = torch.diagonal(ops_damped["a_cov"]).reciprocal()
    col_scale = col_scale / col_scale.mean().clamp_min(1e-12)
    row_scale = torch.diagonal(g_kl).reciprocal()
    row_scale = row_scale / row_scale.mean().clamp_min(1e-12)
    step_aniso_euc = -((row_scale.unsqueeze(1) * g_sp64 * col_scale.unsqueeze(0)) @ ops_damped["p_euc"])

    # 3. Flat Euclidean P_0
    step_flat_euc = -(g_sp64 @ ops_damped["p_euc"])

    # 4. Undamped raw K-FAC (lambda=1e-8)
    g_inv_u = ops_undamped["g_inv"] if ops_undamped["g_inv"] is not None else torch.eye(d_out, dtype=torch.float64)
    step_undamped = -(g_inv_u @ g_sp64 @ ops_undamped["r_step"])

    rep_kron = _summarize(dw_kron)
    return {
        "damped_kronecker_riemannian": {**rep_kron, "cond_a": ops_damped["diagnostics"]["cond_a"]},
        "flat_euclidean_p0": _summarize(step_flat_euc),
        "unwhitened_aniso_euclidean_p0": _summarize(step_aniso_euc),
        "undamped_raw_kfac": {**_summarize(step_undamped), "cond_a": ops_undamped["diagnostics"]["cond_a"]},
        "theorem_3_7_resolved": bool(
            rep_kron["spec_directional_deriv"] <= 1e-12
            and rep_kron["prose_nullspace_leak"] <= 1e-9
            and rep_kron["prose_kl_nats"] <= _summarize(step_flat_euc)["prose_kl_nats"] + 1e-12
        ),
    }


# ---------------------------------------------------------------------------
# 3. Model Discovery, ANSE Evaluation & Verified Surgery Orchestrator
# ---------------------------------------------------------------------------


def _is_general_domain(domain_id: str, prefixes: Sequence[str]) -> bool:
    """Return True if ``domain_id`` matches any configured general-prose prefix."""
    return any(domain_id.startswith(pref) for pref in prefixes)


def _summarize_anse_metrics(
    ref_nll: Mapping[str, float],
    cand_nll: Mapping[str, float],
    *,
    cfg: KroneckerRiemannianTransportConfig,
) -> dict[str, Any]:
    """Compute general-prose guard clearance, worst-domain drift, and ANSE energy."""
    keys = sorted(ref_nll.keys())
    if set(cand_nll.keys()) != set(keys):
        raise ValueError("Candidate and reference domain NLL mappings must share identical keys")
    gen_keys = [k for k in keys if _is_general_domain(k, cfg.general_prefixes)]
    if not gen_keys:
        raise ValueError(f"Missing required general-domain key matching {cfg.general_prefixes}")

    gen_cap = cfg.general_guard_max_ratio if cfg.general_guard_max_ratio is not None else cfg.domain_guard_max_ratio
    diffs: list[float] = []
    per_domain: dict[str, dict[str, float]] = {}
    worst_id = ""
    worst_ratio = 0.0
    for k in keys:
        b = float(ref_nll[k])
        c = float(cand_nll[k])
        ratio = c / max(b, 1e-8)
        diffs.append(c - b)
        per_domain[k] = {
            "base_nll": b,
            "cand_nll": c,
            "diff_vs_base": c - b,
            "ratio": ratio,
            "pct_vs_base": 100.0 * (ratio - 1.0),
        }
        if ratio >= worst_ratio:
            worst_ratio = ratio
            worst_id = k

    gen_ratio = max(per_domain[k]["ratio"] for k in gen_keys)
    gen_ok = bool(gen_ratio <= gen_cap + 1e-12)
    worst_ok = bool(worst_ratio <= cfg.domain_guard_max_ratio + 1e-12)
    guard_ok = bool(gen_ok and (worst_ok if cfg.enforce_all_domain_guard else True))

    mean_diff = float(sum(diffs) / float(len(diffs)))
    nll_mean = float(sum(float(cand_nll[k]) for k in keys) / float(len(keys)))
    base_nll_mean = float(sum(float(ref_nll[k]) for k in keys) / float(len(keys)))
    if not guard_ok:
        anse_energy = float(cfg.maximum_pain_energy)
    else:
        anse_energy = mean_diff + (cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_e0_ratio else 0.0)

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


def _enumerate_kronecker_units(
    model: nn.Module,
    *,
    target_layer_set: set[int] | None,
    target_expert_set: set[int] | None,
    target_factor_set: set[str],
    cfg: KroneckerRiemannianTransportConfig,
) -> tuple[list[dict[str, Any]], bool]:
    """Enumerate MLP/MoE, attention, and optional head units across supported topologies."""
    mlp_blocks, has_crowd = _discover_mlp_blocks(model)
    n_blocks = len(mlp_blocks)
    if target_layer_set is None:
        resolved_layers = {idx for idx, _ in mlp_blocks}
    else:
        if not target_layer_set or any(idx >= n_blocks for idx in target_layer_set):
            raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
        resolved_layers = target_layer_set

    units: list[dict[str, Any]] = []
    mlp_factors = target_factor_set & _MLP_FACTORS
    if mlp_factors:
        units.extend(
            _enumerate_target_units(
                mlp_blocks,
                has_crowd_blocks=has_crowd,
                target_layer_set=resolved_layers,
                target_expert_set=target_expert_set,
                target_factor_set=mlp_factors,
                cfg=cfg,
            )
        )

    attn_factors = target_factor_set & _ATTN_FACTORS
    if attn_factors:
        seq_blocks = getattr(model, "blocks", None) if has_crowd else getattr(getattr(model, "model", model), "layers", None)
        if seq_blocks is not None:
            for l_idx in sorted(resolved_layers):
                blk = seq_blocks[l_idx]
                attn_mod = getattr(blk, "attn", None) if has_crowd else getattr(blk, "self_attn", None)
                if attn_mod is None:
                    continue
                prefix = f"blocks.{l_idx}.attn" if has_crowd else f"layers.{l_idx}.self_attn"
                for f_name in sorted(attn_factors):
                    sub = getattr(attn_mod, f_name, None)
                    if isinstance(sub, nn.Linear):
                        units.append(
                            {
                                "name": f"{prefix}.{f_name}",
                                "kind": "linear",
                                "layer_idx": l_idx,
                                "expert_idx": None,
                                "factor": f_name,
                                "module": sub,
                                "mlp_module": attn_mod,
                                "param": sub.weight,
                            }
                        )

    for h_name in ("head", "lm_head"):
        if h_name in target_factor_set:
            h_mod = getattr(model, h_name, None)
            if isinstance(h_mod, nn.Linear):
                units.append(
                    {
                        "name": h_name,
                        "kind": "linear",
                        "layer_idx": -1,
                        "expert_idx": None,
                        "factor": h_name,
                        "module": h_mod,
                        "mlp_module": h_mod,
                        "param": h_mod.weight,
                    }
                )
    return units, has_crowd


def _collect_crowd_kronecker_pairs(
    model: nn.Module,
    units: Sequence[dict[str, Any]],
    ids_cal: torch.Tensor,
    expert_ids_cal: Sequence[str],
    *,
    cfg: KroneckerRiemannianTransportConfig,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Run deterministic prose and specialist calibration passes on ``CrowdV1WithEngram``."""
    prose_mask = torch.tensor(
        [_is_general_domain(eid, cfg.general_prefixes) for eid in expert_ids_cal], dtype=torch.bool
    )
    ids_prose = ids_cal[prose_mask]
    ids_spec = ids_cal[~prose_mask]
    orig_req = {id(p): p.requires_grad for p in model.parameters()}
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        for u in units:
            u["param"].requires_grad_(True)

        with torch.enable_grad():
            def _pass_for(ids_subset: torch.Tensor) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor], int]:
                def _run() -> tuple[torch.Tensor, int]:
                    total_l = torch.zeros((), dtype=torch.float32, device=model.wte.weight.device)
                    total_t = 0
                    for b0 in range(0, int(ids_subset.shape[0]), cfg.batch_size):
                        ls, nt = _forward_crowd_model_causal_sum_loss(
                            model, ids_subset[b0 : b0 + cfg.batch_size], score_from=cfg.score_from
                        )
                        total_l = total_l + ls
                        total_t += nt
                    return total_l, total_t

                return _capture_linear_pass(units, _run)

            prose_cap, _, _ = _pass_for(ids_prose)
            spec_cap, spec_grads, _ = _pass_for(ids_spec)
    finally:
        for p in model.parameters():
            p.requires_grad_(orig_req.get(id(p), False))

    return prose_cap, spec_cap, {k: -v for k, v in spec_grads.items()}


def _build_explicit_captures(
    units: Sequence[dict[str, Any]],
    prose_activations: torch.Tensor | Mapping[str, torch.Tensor],
    metric_activations: torch.Tensor | Mapping[str, torch.Tensor] | None,
    metric_grad_outs: torch.Tensor | Mapping[str, torch.Tensor] | None,
    target_deltas: Mapping[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor | None],
    dict[str, torch.Tensor],
]:
    """Normalize explicit activation, output-gradient, and target-delta mappings per unit."""
    xp_map: dict[str, torch.Tensor] = {}
    xm_map: dict[str, torch.Tensor] = {}
    gm_map: dict[str, torch.Tensor | None] = {}
    td_map: dict[str, torch.Tensor] = {}

    for u in units:
        u_name = u["name"]
        d_out, d_in = int(u["param"].shape[0]), int(u["param"].shape[1])
        xp_u = _lookup_tensor_for_unit(prose_activations, u, expected_last_dim=d_in)
        td_u = _lookup_tensor_for_unit(target_deltas, u, expected_shape=(d_out, d_in))
        if xp_u is None or td_u is None:
            continue
        xp_map[u_name] = _validate_sample_matrix(xp_u, f"prose_activations[{u_name}]")[0]
        td_map[u_name] = _validate_finite_2d(td_u, f"target_deltas[{u_name}]")

        xm_u = _lookup_tensor_for_unit(metric_activations, u, expected_last_dim=d_in)
        xm_map[u_name] = (
            _validate_sample_matrix(xm_u, f"metric_activations[{u_name}]")[0] if xm_u is not None else xp_map[u_name]
        )
        gm_u = _lookup_tensor_for_unit(metric_grad_outs, u, expected_last_dim=d_out)
        gm_map[u_name] = _validate_sample_matrix(gm_u, f"metric_grad_outs[{u_name}]")[0] if gm_u is not None else None
    return xp_map, xm_map, gm_map, td_map


def _solve_proposals_for_units(
    units: Sequence[dict[str, Any]],
    xp_map: Mapping[str, torch.Tensor],
    xm_map: Mapping[str, torch.Tensor],
    gm_map: Mapping[str, torch.Tensor | None],
    td_map: Mapping[str, torch.Tensor],
    *,
    cfg: KroneckerRiemannianTransportConfig,
    kl_budget_override: float | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Compute Kronecker-whitened Riemannian transport proposals and rescale to KL trust radius."""
    kl_cap = kl_budget_override if kl_budget_override is not None else cfg.max_kl_nats
    proposals: list[dict[str, Any]] = []

    for u in units:
        u_name = u["name"]
        if u_name not in td_map or u_name not in xp_map:
            continue
        t_delta = td_map[u_name]
        if float(t_delta.abs().max().item()) <= 1e-18:
            continue

        ops = compute_kronecker_whitened_nullspace_operators(
            xp_map[u_name],
            x_metric=xm_map[u_name],
            grad_out_metric=gm_map.get(u_name),
            rank_keep=cfg.rank_keep,
            sv_rel_tol=cfg.sv_rel_tol,
            damping_a=cfg.damping_a,
            damping_g=cfg.damping_g,
            ridge_floor=cfg.ridge_floor,
            scale_damping_by_trace=cfg.scale_damping_by_trace,
        )
        dw_u, diag_u = solve_kronecker_riemannian_transport_step(
            t_delta,
            ops,
            max_kl_nats=kl_cap,
            transport_mode=cfg.transport_mode,
            rescale_only_if_exceeds=cfg.rescale_only_if_exceeds,
            max_frobenius_norm=cfg.max_frobenius_norm,
        )
        proposals.append(
            {
                "unit": u,
                "operators": ops,
                "delta_w": dw_u,
                "step_diag": diag_u,
            }
        )

    if not proposals:
        return proposals, 0.0, 0.0

    raw_kl = sum(float(p["step_diag"]["kl_nats"]) for p in proposals)
    g_scale = 1.0
    if cfg.kl_budget_scope == "global" and raw_kl > 0.0 and ((not cfg.rescale_only_if_exceeds) or raw_kl > kl_cap):
        g_scale = math.sqrt(kl_cap / raw_kl)
    if cfg.max_frobenius_norm is not None:
        max_f = float(cfg.max_frobenius_norm)
        for prop in proposals:
            p_frob = float(prop["step_diag"]["frobenius_norm"])
            if p_frob > 1e-30:
                g_scale = min(g_scale, max_f / p_frob)

    total_kl = 0.0
    total_euc_kl = 0.0
    for prop in proposals:
        dw_final = (g_scale * prop["delta_w"]).contiguous()
        d_out = int(dw_final.shape[0])
        g_kl = (
            prop["operators"]["g_cov"]
            if prop["operators"]["g_cov"] is not None
            else torch.eye(d_out, dtype=torch.float64)
        )
        kl_f = compute_kronecker_kl_nats(dw_final, prop["operators"]["a_prose"], g_kl)
        euc_kl_f = (g_scale * g_scale) * float(prop["step_diag"]["euclidean_p0_kl_nats"])
        prop["final_delta"] = dw_final
        prop["final_kl_nats"] = kl_f
        prop["final_euc_kl_nats"] = euc_kl_f
        prop["final_frob"] = float(torch.linalg.matrix_norm(dw_final, ord="fro").item())
        prop["final_leak"] = g_scale * float(prop["step_diag"]["prose_nullspace_leak"])
        total_kl += kl_f
        total_euc_kl += euc_kl_f

    return proposals, total_kl, total_euc_kl


def _apply_proposals(
    proposals: Sequence[Mapping[str, Any]],
    base_params_by_id: Mapping[int, torch.Tensor],
    scale: float,
) -> None:
    """Write ``base + scale * final_delta`` to each targeted parameter tensor."""
    for prop in proposals:
        param: torch.Tensor = prop["unit"]["param"]
        base_cpu = base_params_by_id[id(param)]
        upd = base_cpu.to(device=param.device, dtype=torch.float64) + (scale * prop["final_delta"]).to(
            device=param.device, dtype=torch.float64
        )
        param.copy_(upd.to(dtype=param.dtype))


def _resolve_transport_config(
    config: KroneckerRiemannianTransportConfig | None,
    overrides: Mapping[str, Any],
) -> KroneckerRiemannianTransportConfig:
    """Merge keyword overrides into a validated ``KroneckerRiemannianTransportConfig``."""
    if config is not None and not isinstance(config, KroneckerRiemannianTransportConfig):
        raise TypeError(
            f"config must be KroneckerRiemannianTransportConfig or None, got {type(config).__name__}"
        )
    base = config or KroneckerRiemannianTransportConfig()
    merged = asdict(base)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = tuple(str(x) for x in v) if k == "target_factors" else v
    return KroneckerRiemannianTransportConfig(**merged)


@torch.no_grad()
def execute_verified_kronecker_riemannian_surgery(
    model: nn.Module,
    calibration_chunks: Sequence[Mapping[str, Any]] | None = None,
    *,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    config: KroneckerRiemannianTransportConfig | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    target_deltas: Mapping[str, torch.Tensor] | None = None,
    prose_activations: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    metric_activations: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    metric_grad_outs: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_factors: Sequence[str] | None = None,
    damping_a: float | None = None,
    damping_g: float | None = None,
    scale_damping_by_trace: bool | None = None,
    rank_keep: int | None = None,
    sv_rel_tol: float | None = None,
    max_kl_nats: float | None = None,
    kl_budget_scope: str | None = None,
    transport_mode: str | None = None,
    metric_source: str | None = None,
    rescale_only_if_exceeds: bool | None = None,
    max_frobenius_norm: float | None = None,
    sequential_refresh: bool | None = None,
    protect_general_expert: bool | None = None,
    domain_guard_max_ratio: float | None = None,
    general_guard_max_ratio: float | None = None,
    enforce_all_domain_guard: bool | None = None,
    require_energy_improvement: bool | None = None,
    enable_trust_backtracking: bool | None = None,
    backtrack_halvings: int | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> dict[str, Any]:
    """Execute ANSE-gated Kronecker Riemannian null-space transport surgery with bitwise rollback."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if target_deltas is not None and not isinstance(target_deltas, Mapping):
        raise TypeError("target_deltas must be a mapping or None")

    cfg = _resolve_transport_config(
        config,
        {
            "target_factors": target_factors,
            "damping_a": damping_a,
            "damping_g": damping_g,
            "scale_damping_by_trace": scale_damping_by_trace,
            "rank_keep": rank_keep,
            "sv_rel_tol": sv_rel_tol,
            "max_kl_nats": max_kl_nats,
            "kl_budget_scope": kl_budget_scope,
            "transport_mode": transport_mode,
            "metric_source": metric_source,
            "rescale_only_if_exceeds": rescale_only_if_exceeds,
            "max_frobenius_norm": max_frobenius_norm,
            "sequential_refresh": sequential_refresh,
            "protect_general_expert": protect_general_expert,
            "domain_guard_max_ratio": domain_guard_max_ratio,
            "general_guard_max_ratio": general_guard_max_ratio,
            "enforce_all_domain_guard": enforce_all_domain_guard,
            "require_energy_improvement": require_energy_improvement,
            "enable_trust_backtracking": enable_trust_backtracking,
            "backtrack_halvings": backtrack_halvings,
            "score_from": score_from,
            "inplace": inplace,
        },
    )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    target_layer_set = (
        {_validate_nonneg_int(idx, "layer_index") for idx in layer_indices} if layer_indices is not None else None
    )
    target_expert_set = (
        {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices} if expert_indices is not None else None
    )
    units, has_crowd = _enumerate_kronecker_units(
        work_model,
        target_layer_set=target_layer_set,
        target_expert_set=target_expert_set,
        target_factor_set=set(cfg.target_factors),
        cfg=cfg,
    )

    ids_cal: torch.Tensor | None = None
    expert_ids_cal: list[str] = []
    ids_eval: torch.Tensor | None = None
    expert_ids_eval: list[str] = []

    if calibration_chunks is not None:
        if not (has_crowd and hasattr(work_model, "wte") and hasattr(work_model, "ln") and hasattr(work_model, "head")):
            raise TypeError("calibration_chunks requires a CrowdV1WithEngram-compatible model")
        ids_cal, expert_ids_cal = _validate_chunks(
            calibration_chunks,
            "calibration_chunks",
            general_prefix=cfg.general_prefixes[0],
            score_from=cfg.score_from,
            require_both_groups=True,
        )
        if eval_chunks is not None:
            ids_eval, expert_ids_eval = _validate_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefix=cfg.general_prefixes[0],
                score_from=cfg.score_from,
                require_both_groups=False,
            )
        else:
            ids_eval, expert_ids_eval = ids_cal, expert_ids_cal
    else:
        if eval_fn is None:
            raise ValueError("Either calibration_chunks or (prose_activations, target_deltas, eval_fn) is required")
        if prose_activations is None or target_deltas is None:
            raise ValueError("prose_activations and target_deltas are required when calibration_chunks is None")

    base_nll_clean = _parse_eval_domain_nll(baseline_domain_nll) if baseline_domain_nll is not None else None
    was_training = work_model.training
    work_model.eval()

    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)
        param_base_by_id = {id(u["param"]): u["param"].detach().to(device="cpu").clone() for u in units}

        def _run_eval() -> dict[str, float]:
            if eval_fn is not None:
                return _parse_eval_domain_nll(eval_fn(work_model))
            assert ids_eval is not None
            return _eval_crowd_model_domain_nlls(
                work_model,
                ids_eval,
                expert_ids_eval,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )

        pre_nll = _run_eval()
        ref_nll = base_nll_clean if base_nll_clean is not None else pre_nll
        if set(ref_nll.keys()) != set(pre_nll.keys()):
            raise ValueError("baseline_domain_nll and model domain NLLs must share identical keys")
        parent_eval = _summarize_anse_metrics(ref_nll, pre_nll, cfg=cfg)

        all_proposals: list[dict[str, Any]] = []
        total_kl = 0.0
        total_euc_kl = 0.0

        if ids_cal is not None:
            if cfg.sequential_refresh:
                groups = sorted({int(u["layer_idx"]) for u in units}, key=lambda x: (x < 0, x))
                kl_per_group = cfg.max_kl_nats / float(max(1, len(groups))) if cfg.kl_budget_scope == "global" else cfg.max_kl_nats
                for g_idx in groups:
                    grp_units = [u for u in units if int(u["layer_idx"]) == g_idx]
                    p_cap, s_cap, auto_td = _collect_crowd_kronecker_pairs(
                        work_model, grp_units, ids_cal, expert_ids_cal, cfg=cfg
                    )
                    xp_g, xm_g, gm_g, td_g = _build_group_maps(grp_units, p_cap, s_cap, auto_td, target_deltas, cfg)
                    props_g, kl_g, euc_kl_g = _solve_proposals_for_units(
                        grp_units, xp_g, xm_g, gm_g, td_g, cfg=cfg, kl_budget_override=kl_per_group
                    )
                    _apply_proposals(props_g, param_base_by_id, 1.0)
                    all_proposals.extend(props_g)
                    total_kl += kl_g
                    total_euc_kl += euc_kl_g
            else:
                p_cap, s_cap, auto_td = _collect_crowd_kronecker_pairs(
                    work_model, units, ids_cal, expert_ids_cal, cfg=cfg
                )
                xp_m, xm_m, gm_m, td_m = _build_group_maps(units, p_cap, s_cap, auto_td, target_deltas, cfg)
                all_proposals, total_kl, total_euc_kl = _solve_proposals_for_units(
                    units, xp_m, xm_m, gm_m, td_m, cfg=cfg
                )
        else:
            assert prose_activations is not None and target_deltas is not None
            xp_m, xm_m, gm_m, td_m = _build_explicit_captures(
                units, prose_activations, metric_activations, metric_grad_outs, target_deltas
            )
            all_proposals, total_kl, total_euc_kl = _solve_proposals_for_units(
                units, xp_m, xm_m, gm_m, td_m, cfg=cfg
            )

        scales_to_try = (
            [0.5**i for i in range(cfg.backtrack_halvings + 1)]
            if cfg.enable_trust_backtracking and all_proposals
            else [1.0]
        )
        parent_e = float(parent_eval["anse_energy"])
        chosen_scale = 1.0
        cand_eval: dict[str, Any] = {}
        eval_error: str | None = None
        cand_e = parent_e
        delta_e = 0.0
        energy_ok = False

        for try_scale in scales_to_try:
            _apply_proposals(all_proposals, param_base_by_id, try_scale)
            try:
                cand_eval = _summarize_anse_metrics(ref_nll, _run_eval(), cfg=cfg)
                eval_error = None
            except Exception as exc:
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
                eval_error = f"{type(exc).__name__}: {exc}"
            cand_e = float(cand_eval["anse_energy"])
            delta_e = cand_e - parent_e
            energy_ok = (delta_e < 0.0) if cfg.require_energy_improvement else (delta_e <= 0.0)
            chosen_scale = try_scale
            if eval_error is None and bool(cand_eval["guard_ok"]) and energy_ok:
                break

        eff_kl = (chosen_scale * chosen_scale) * total_kl
        eff_euc_kl = (chosen_scale * chosen_scale) * total_euc_kl
        kl_bound = cfg.max_kl_nats * float(max(1, len(all_proposals))) if cfg.kl_budget_scope == "per_unit" else cfg.max_kl_nats
        kl_ok = bool(eff_kl <= kl_bound + 1e-7)

        cand_sd = work_model.state_dict()
        cand_sha = compute_state_sha256(cand_sd)
        total_elems = sum(int(v.numel()) for v in parent_snapshot.values())
        edited_elems = sum(
            int(torch.ne(cand_sd[k].detach().cpu(), parent_snapshot[k]).sum().item())
            for k in parent_snapshot
        )

        if cand_sha == parent_sha or not all_proposals:
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
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}")
    finally:
        work_model.train(was_training)

    unit_reports = [
        {
            "unit": p["unit"]["name"],
            "shape": p["step_diag"]["shape"],
            "rank_used": int(p["operators"]["rank_used"]),
            "cond_a": float(p["operators"]["diagnostics"]["cond_a"]),
            "cond_g": float(p["operators"]["diagnostics"]["cond_g"]),
            "kl_nats": float((chosen_scale * chosen_scale) * p["final_kl_nats"]),
            "euclidean_p0_kl_nats": float((chosen_scale * chosen_scale) * p["final_euc_kl_nats"]),
            "prose_nullspace_leak": float(chosen_scale * p["final_leak"]),
            "frobenius_norm": float(chosen_scale * p["final_frob"]),
        }
        for p in all_proposals
    ]

    receipt: dict[str, Any] = {
        "object": KRONECKER_TRANSPORT_RECEIPT_OBJECT,
        "schema": KRONECKER_TRANSPORT_SCHEMA,
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
        "total_kl_nats": float(eff_kl),
        "total_euclidean_p0_kl_nats": float(eff_euc_kl),
        "kl_ratio_vs_euclidean_p0": float(eff_kl / max(eff_euc_kl, 1e-30)) if eff_euc_kl > 0.0 else 0.0,
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
        "units_edited": len(all_proposals) if accepted else 0,
        "edited_elements": edited_elems,
        "total_elements": total_elems,
        "edited_param_ratio": float(edited_elems) / float(max(1, total_elems)),
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "unit_reports": unit_reports,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


def _build_group_maps(
    units: Sequence[dict[str, Any]],
    prose_cap: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    spec_cap: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    auto_td: Mapping[str, torch.Tensor],
    override_td: Mapping[str, torch.Tensor] | None,
    cfg: KroneckerRiemannianTransportConfig,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor | None],
    dict[str, torch.Tensor],
]:
    """Build ``(x_prose, x_metric, g_metric, target_deltas)`` dicts from calibration passes."""
    xp_m: dict[str, torch.Tensor] = {}
    xm_m: dict[str, torch.Tensor] = {}
    gm_m: dict[str, torch.Tensor | None] = {}
    td_m: dict[str, torch.Tensor] = {}
    for u in units:
        u_name = u["name"]
        if u_name not in prose_cap:
            continue
        xp_u, gp_u = prose_cap[u_name]
        xp_m[u_name] = xp_u
        if cfg.metric_source == "all" and u_name in spec_cap:
            xs_u, gs_u = spec_cap[u_name]
            xm_m[u_name] = torch.cat([xp_u, xs_u], dim=0)
            gm_m[u_name] = torch.cat([gp_u, gs_u], dim=0)
        else:
            xm_m[u_name] = xp_u
            gm_m[u_name] = gp_u

        ovr = _lookup_tensor_for_unit(override_td, u, expected_shape=tuple(u["param"].shape))
        if ovr is not None:
            td_m[u_name] = _validate_finite_2d(ovr, f"target_deltas[{u_name}]")
        elif u_name in auto_td:
            td_m[u_name] = auto_td[u_name]
    return xp_m, xm_m, gm_m, td_m


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DAMPING_A",
    "DEFAULT_DAMPING_G",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_MAX_KL_NATS",
    "DEFAULT_RIDGE_FLOOR",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SV_REL_TOL",
    "DEFAULT_TARGET_FACTORS",
    "GENERAL_EXPERT_PREFIX",
    "KRONECKER_TRANSPORT_RECEIPT_OBJECT",
    "KRONECKER_TRANSPORT_SCHEMA",
    "KroneckerRiemannianTransportConfig",
    "VALID_KL_BUDGET_SCOPES",
    "VALID_METRIC_SOURCES",
    "VALID_TARGET_FACTORS",
    "VALID_TRANSPORT_MODES",
    "canonical_json_bytes",
    "compute_kronecker_kl_nats",
    "compute_kronecker_whitened_nullspace_operators",
    "compute_payload_sha256",
    "compute_state_sha256",
    "diagnose_theorem_3_7_obstruction",
    "execute_verified_kronecker_riemannian_surgery",
    "solve_kronecker_riemannian_transport_step",
    "transport_delta_between_curved_layers",
    "transport_factorized_lora_with_kronecker_metric",
    "verify_kronecker_riemannian_receipt_provenance",
]
