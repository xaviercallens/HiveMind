# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Ledoit-Wolf Marchenko-Pastur Covariance Shrinkage & Condition-Bounded AlphaEdit Filter.

Solves the ``p / n ~= 1`` Marchenko-Pastur oblique-projection blowup encountered in Pilot 1
``WSP_A2``, where inverting an ill-conditioned empirical specialist second-moment matrix
``M_spec`` estimated from ``N ~= d`` tokens produces an oblique right filter
``G^* = I_d - B^{1/2} Q_p diag(s) Q_p^T B^{-1/2}`` with ``||G^*||_2 >> 1`` and severe
out-of-sample general-domain NLL amplification when propagated through MoE layers.

Implements three training-free ``float64`` numerical-control primitives plus a sequential
SwiGLU MoE layer wrapper:

1. ``compute_oracle_ledoit_wolf_moment(x_2d)``:
   Computes closed-form optimal linear shrinkage
   ``M_shrunk = (1 - rho) * M_emp + rho * (tr(M_emp) / d) * I_d``
   combining empirical distribution-free Ledoit-Wolf (2004) fourth-moment shrinkage and
   Chen-Wiesel-Eldar-Hero (2010) Oracle Approximating Shrinkage (OAS) with analytical
   ``rho in [0, 1]`` and aspect-ratio ``gamma = d / N`` diagnostics.
2. ``compute_condition_bounded_alphaedit_filter(...)``:
   Applies analytical oracle shrinkage to ``(m_prose, m_spec)`` from sample counts
   ``(n_prose, n_spec)`` and guarantees ``||G^*||_2 <= max_operator_norm`` (default ``1.05``)
   via automatic spectral-norm damping / dual-orthogonal Euclidean fallback whenever oblique
   conditioning exceeds safety bounds.
3. ``audit_filter_stability(g_mat, x_holdout_prose, x_holdout_spec)``:
   Certifies spectral norm boundedness, out-of-sample prose activation variance
   non-amplification, and out-of-sample specialist preservation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.wsp_alphaedit_control import compute_swiglu_intermediate_states

DEFAULT_MAX_OPERATOR_NORM: float = 1.05
DEFAULT_REG_SPEC: float = 0.10
DEFAULT_TAU: float = 1.25
DEFAULT_MAX_RANK: int = 96
DEFAULT_MAX_PROSE_ENERGY_RATIO: float = 1.0
DEFAULT_MAX_SPECIALIST_REL_DISTORTION: float = 0.10


def _validate_nonneg_finite(value: float, name: str) -> float:
    """Validate that ``value`` is a finite real number >= 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: float, name: str) -> float:
    """Validate that ``value`` is a finite real number > 0.0."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_nonneg_int(value: int, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: int, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a non-empty 2D tensor and return a detached CPU ``float64`` copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    xd = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(xd).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return xd


def _validate_square_moment(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a non-empty square 2D moment matrix and return a symmetrized CPU ``float64`` copy."""
    mat = _validate_finite_2d(tensor, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {tuple(mat.shape)}")
    return (0.5 * (mat + mat.T)).contiguous()


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _matrix_condition_number(evals_asc: torch.Tensor) -> float:
    """Compute finite condition number from ascending eigenvalues of a PSD matrix."""
    max_ev = max(float(evals_asc[-1].item()), 0.0)
    if max_ev <= 1e-15:
        return 1.0
    min_ev = max(float(evals_asc[0].item()), 0.0)
    return float(max_ev / max(min_ev, 1e-12))


def _compute_oas_shrinkage_rho(m_emp: torch.Tensor, n_samples: int) -> tuple[float, float, float]:
    """Compute analytical Oracle Approximating Shrinkage (OAS) coefficient ``rho in [0, 1]``.

    Implements the Chen-Wiesel-Eldar-Hero (2010) closed-form MMSE oracle shrinkage
    coefficient for second-moment matrix ``m_emp`` estimated from ``n_samples`` vectors,
    returning ``(rho_oas, mu_iso, delta2)``.
    """
    d = int(m_emp.shape[0])
    tr_m = float(torch.trace(m_emp).item())
    mu_iso = tr_m / float(d)
    if d == 1:
        return 0.0, mu_iso, 0.0

    tr_m2 = float((m_emp * m_emp).sum().item())
    delta2 = max(0.0, tr_m2 - (tr_m * tr_m) / float(d))
    denom = (float(n_samples) + 1.0 - 2.0 / float(d)) * delta2
    if denom <= 1e-30:
        return 1.0, mu_iso, delta2
    numer = (1.0 - 2.0 / float(d)) * tr_m2 + tr_m * tr_m
    rho_oas = min(1.0, max(0.0, numer / denom))
    return float(rho_oas), float(mu_iso), float(delta2)


def shrink_moment_oracle_oas(
    m_emp: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shrink an empirical square second-moment matrix using closed-form OAS.

    Computes ``M_shrunk = (1 - rho) * M_emp + rho * (tr(M_emp) / d) * I_d`` directly from
    ``(m_emp, n_samples)`` without requiring raw token activations.
    """
    m64 = _validate_square_moment(m_emp, "m_emp")
    n_int = _validate_positive_int(n_samples, "n_samples")
    d = int(m64.shape[0])

    rho, mu_iso, _ = _compute_oas_shrinkage_rho(m64, n_int)
    eye_d = torch.eye(d, dtype=torch.float64)
    m_shrunk = (1.0 - rho) * m64 + (rho * mu_iso) * eye_d
    m_shrunk = (0.5 * (m_shrunk + m_shrunk.T)).contiguous()

    return m_shrunk, {
        "n_samples": float(n_int),
        "dim": float(d),
        "gamma": float(d) / float(n_int),
        "rho": float(rho),
        "trace_mean": float(mu_iso),
    }


@dataclass(frozen=True)
class MarchenkoPasturGuardConfig:
    """Configuration for Marchenko-Pastur shrinkage and condition-bounded AlphaEdit."""

    eta: float = 2.5
    tau: float = DEFAULT_TAU
    max_rank: int = DEFAULT_MAX_RANK
    reg_spec: float = DEFAULT_REG_SPEC
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM
    shrink_moments: bool = True
    max_prose_energy_ratio: float = DEFAULT_MAX_PROSE_ENERGY_RATIO
    max_specialist_rel_distortion: float = DEFAULT_MAX_SPECIALIST_REL_DISTORTION

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.eta, "eta")
        _validate_nonneg_finite(self.tau, "tau")
        _validate_nonneg_int(self.max_rank, "max_rank")
        _validate_nonneg_finite(self.reg_spec, "reg_spec")
        max_op = _validate_positive_finite(self.max_operator_norm, "max_operator_norm")
        if max_op < 1.0:
            raise ValueError(f"max_operator_norm must be >= 1.0, got {self.max_operator_norm!r}")
        _validate_bool(self.shrink_moments, "shrink_moments")
        _validate_positive_finite(self.max_prose_energy_ratio, "max_prose_energy_ratio")
        _validate_nonneg_finite(self.max_specialist_rel_distortion, "max_specialist_rel_distortion")


@torch.no_grad()
def compute_oracle_ledoit_wolf_moment(
    x_2d: torch.Tensor,
    *,
    center: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute closed-form Ledoit-Wolf / Oracle shrunk second-moment matrix from activations.

    Computes ``M_shrunk = (1 - rho) * M_emp + rho * (tr(M_emp) / d) * I_d`` in CPU
    ``float64``, where ``rho = max(rho_lw, rho_oas) in [0, 1]`` combines:
    - Ledoit-Wolf (2004) distribution-free empirical fourth-moment shrinkage ``rho_lw``
      (robust against heavy-tailed transformer outlier activations), and
    - Chen-Wiesel-Eldar-Hero (2010) Gaussian Wishart Oracle Approximating Shrinkage
      ``rho_oas`` (provably well-behaved down to ``N = 1 << d``).

    Returns ``(m_shrunk, diagnostics)`` including ``gamma = d / N``, ``rho``, and
    condition-number diagnostics before and after shrinkage.
    """
    _validate_bool(center, "center")
    xd = _validate_finite_2d(x_2d, "x_2d")
    n_int, d = int(xd.shape[0]), int(xd.shape[1])
    if center:
        xd = xd - xd.mean(dim=0, keepdim=True)

    m_emp = (xd.T @ xd) / float(n_int)
    m_emp = (0.5 * (m_emp + m_emp.T)).contiguous()

    rho_oas, mu_iso, delta2 = _compute_oas_shrinkage_rho(m_emp, n_int)
    if d == 1:
        rho_lw = 0.0
    elif delta2 <= 1e-30:
        rho_lw = 1.0
    else:
        tr_m2 = float((m_emp * m_emp).sum().item())
        x_norm4 = float(xd.pow(2).sum(dim=1).pow(2).sum().item())
        beta2_lw = max(0.0, x_norm4 / float(n_int * n_int) - tr_m2 / float(n_int))
        rho_lw = min(1.0, max(0.0, beta2_lw / delta2))

    rho = min(1.0, max(0.0, max(rho_lw, rho_oas)))
    eye_d = torch.eye(d, dtype=torch.float64)
    m_shrunk = (1.0 - rho) * m_emp + (rho * mu_iso) * eye_d
    m_shrunk = (0.5 * (m_shrunk + m_shrunk.T)).contiguous()

    ev_emp = torch.linalg.eigvalsh(m_emp)
    ev_shr = torch.linalg.eigvalsh(m_shrunk)
    cond_emp = _matrix_condition_number(ev_emp)
    cond_shr = _matrix_condition_number(ev_shr)
    gamma = float(d) / float(n_int)

    return m_shrunk, {
        "n_samples": n_int,
        "dim": d,
        "gamma": gamma,
        "aspect_ratio": gamma,
        "rho": float(rho),
        "rho_lw": float(rho_lw),
        "rho_oas": float(rho_oas),
        "trace_mean": float(mu_iso),
        "condition_number": cond_shr,
        "condition_number_emp": cond_emp,
        "condition_number_shrunk": cond_shr,
        "min_eig_emp": float(ev_emp[0].item()),
        "max_eig_emp": float(ev_emp[-1].item()),
        "min_eig_shrunk": float(ev_shr[0].item()),
        "max_eig_shrunk": float(ev_shr[-1].item()),
    }


def _build_dual_orthogonal_euclidean_filter(
    mp_s: torch.Tensor,
    b_mat: torch.Tensor,
    b_inv_half: torch.Tensor,
    q_p: torch.Tensor,
    *,
    eta: float,
    tau: float,
) -> torch.Tensor:
    """Construct contraction-safe Euclidean orthogonal fallback ``I - U_p diag(s_euc) U_p^T``.

    Orthonormalizes the dual generalized prose directions ``B^{-1/2} Q_p`` (which naturally
    lie orthogonal to heavy specialist spikes) and evaluates exact Euclidean Rayleigh ratios
    so specialist directions retain zero-or-minimal distortion while ``||G_euc||_2 <= 1.0``.
    """
    d = int(mp_s.shape[0])
    u_p, _ = torch.linalg.qr(b_inv_half @ q_p, mode="reduced")
    p_u = torch.diagonal(u_p.T @ mp_s @ u_p).clamp_min(0.0)
    s_u = torch.diagonal(u_p.T @ b_mat @ u_p).clamp_min(1e-8)
    ex_u = (p_u / s_u - tau).clamp_min(0.0)
    s_euc = (eta * ex_u) / (1.0 + eta * ex_u)
    return (torch.eye(d, dtype=torch.float64) - (u_p * s_euc.unsqueeze(0)) @ u_p.T).contiguous()


@torch.no_grad()
def compute_condition_bounded_alphaedit_filter(
    m_prose: torch.Tensor,
    m_spec: torch.Tensor,
    *,
    n_prose: int | None = None,
    n_spec: int | None = None,
    eta: float = 2.5,
    tau: float = DEFAULT_TAU,
    max_rank: int = DEFAULT_MAX_RANK,
    reg_spec: float = DEFAULT_REG_SPEC,
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM,
    shrink_moments: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute Marchenko-Pastur shrunk and spectral-norm-bounded AlphaEdit filter ``G^*``.

    Guarantees ``||G^*||_2 <= max_operator_norm`` (default ``1.05``) via:
    1. Analytical OAS shrinkage of ``(m_prose, m_spec)`` when ``shrink_moments=True`` and
       ``(n_prose, n_spec)`` are supplied.
    2. Generalized eigendecomposition on ``(M_prose_shrunk, B)`` where
       ``B = M_spec_shrunk + lam_spec * I_d``.
    3. Automatic spectral-norm damping / dual-orthogonal Euclidean fallback whenever the
       unclamped oblique operator norm ``||G_oblique||_2`` exceeds ``max_operator_norm``.
    """
    mp_raw = _validate_square_moment(m_prose, "m_prose")
    ms_raw = _validate_square_moment(m_spec, "m_spec")
    d = int(mp_raw.shape[0])
    if ms_raw.shape != (d, d):
        raise ValueError(
            f"Moment matrices must share shape; got {tuple(mp_raw.shape)} and {tuple(ms_raw.shape)}"
        )

    eta_f = _validate_nonneg_finite(eta, "eta")
    tau_f = _validate_nonneg_finite(tau, "tau")
    max_r = _validate_nonneg_int(max_rank, "max_rank")
    reg_f = _validate_nonneg_finite(reg_spec, "reg_spec")
    max_norm = _validate_positive_finite(max_operator_norm, "max_operator_norm")
    if max_norm < 1.0:
        raise ValueError(f"max_operator_norm must be >= 1.0, got {max_operator_norm!r}")
    _validate_bool(shrink_moments, "shrink_moments")

    np_int = _validate_positive_int(n_prose, "n_prose") if n_prose is not None else None
    ns_int = _validate_positive_int(n_spec, "n_spec") if n_spec is not None else None

    if shrink_moments and np_int is not None:
        mp_s, p_shr = shrink_moment_oracle_oas(mp_raw, np_int)
        rho_p = p_shr["rho"]
    else:
        mp_s = mp_raw
        rho_p = 0.0

    if shrink_moments and ns_int is not None:
        ms_s, s_shr = shrink_moment_oracle_oas(ms_raw, ns_int)
        rho_s = s_shr["rho"]
    else:
        ms_s = ms_raw
        rho_s = 0.0

    gamma_p = (float(d) / float(np_int)) if np_int is not None else 0.0
    gamma_s = (float(d) / float(ns_int)) if ns_int is not None else 0.0

    if eta_f == 0.0 or max_r == 0:
        eye_d = torch.eye(d, dtype=torch.float64)
        return eye_d, {
            "active_modes": 0,
            "operator_norm": 1.0,
            "unbounded_operator_norm": 1.0,
            "condition_number": 1.0,
            "b_condition_number": 1.0,
            "rho_prose": rho_p,
            "rho_spec": rho_s,
            "gamma_prose": gamma_p,
            "gamma_spec": gamma_s,
            "max_ratio": 0.0,
            "variance_ratio": 0.0,
            "mean_shrink": 0.0,
            "damping_alpha": 1.0,
            "fallback_mode": "identity",
            "clamped": False,
            "bounded": True,
        }

    eye_d = torch.eye(d, dtype=torch.float64)
    tr_scale = max(float(torch.diagonal(ms_s).mean().item()), 1e-6)
    lam_spec = max(reg_f * tr_scale, 1e-6)
    b_mat = (ms_s + lam_spec * eye_d).contiguous()

    evals_b, evecs_b = torch.linalg.eigh(b_mat)
    evals_b_clamped = evals_b.clamp_min(1e-8)
    cond_b = _matrix_condition_number(evals_b_clamped)
    b_half = (evecs_b * evals_b_clamped.sqrt().unsqueeze(0)) @ evecs_b.T
    b_inv_half = (evecs_b * evals_b_clamped.rsqrt().unsqueeze(0)) @ evecs_b.T

    whitened = b_inv_half @ mp_s @ b_inv_half
    whitened = 0.5 * (whitened + whitened.T)
    mu_asc, q_asc = torch.linalg.eigh(whitened)
    mu = mu_asc.flip(0)
    q = q_asc.flip(1)

    excess = mu - tau_f
    k_eff = min(max_r, int((excess > 0.0).sum().item()))
    max_ratio = float(mu[0].item())

    if k_eff == 0:
        return eye_d, {
            "active_modes": 0,
            "operator_norm": 1.0,
            "unbounded_operator_norm": 1.0,
            "condition_number": 1.0,
            "b_condition_number": cond_b,
            "rho_prose": rho_p,
            "rho_spec": rho_s,
            "gamma_prose": gamma_p,
            "gamma_spec": gamma_s,
            "max_ratio": max_ratio,
            "variance_ratio": max_ratio,
            "mean_shrink": 0.0,
            "damping_alpha": 1.0,
            "fallback_mode": "identity",
            "clamped": False,
            "bounded": True,
        }

    q_p = q[:, :k_eff]
    shrink = (eta_f * excess[:k_eff]) / (1.0 + eta_f * excess[:k_eff])
    g_ob = (eye_d - (b_half @ q_p) @ ((q_p * shrink.unsqueeze(0)).T @ b_inv_half)).contiguous()
    op_ob = float(torch.linalg.matrix_norm(g_ob, ord=2).item())

    if op_ob <= max_norm:
        g_star = g_ob
        damping_alpha = 1.0
        fallback_mode = "oblique_shrunk"
        clamped = False
    else:
        g_euc = _build_dual_orthogonal_euclidean_filter(
            mp_s, b_mat, b_inv_half, q_p, eta=eta_f, tau=tau_f
        )
        op_euc = float(torch.linalg.matrix_norm(g_euc, ord=2).item())
        if op_euc > max_norm:
            g_euc = (g_euc * (max_norm / op_euc)).contiguous()

        lo = 0.0
        if max_norm > 1.0 + 1e-12:
            hi = 1.0
            for _ in range(36):
                mid = 0.5 * (lo + hi)
                g_cand = mid * g_ob + (1.0 - mid) * g_euc
                if float(torch.linalg.matrix_norm(g_cand, ord=2).item()) <= max_norm:
                    lo = mid
                else:
                    hi = mid

        g_mix = (lo * g_ob + (1.0 - lo) * g_euc).contiguous()
        if lo > 1e-12 and float(torch.linalg.matrix_norm(g_mix, ord=2).item()) <= max_norm:
            g_star = g_mix
            damping_alpha = float(lo)
            fallback_mode = "spectral_damped"
        else:
            g_star = g_euc
            damping_alpha = 0.0
            fallback_mode = "euclidean_fallback"
        clamped = True

    sv_g = torch.linalg.svdvals(g_star)
    op_norm = float(sv_g[0].item())
    cond_g = float(op_norm / max(float(sv_g[-1].item()), 1e-12))

    return g_star, {
        "active_modes": k_eff,
        "operator_norm": op_norm,
        "unbounded_operator_norm": op_ob,
        "condition_number": cond_g,
        "b_condition_number": cond_b,
        "rho_prose": rho_p,
        "rho_spec": rho_s,
        "gamma_prose": gamma_p,
        "gamma_spec": gamma_s,
        "max_ratio": max_ratio,
        "variance_ratio": max_ratio,
        "min_selected_ratio": float(mu[k_eff - 1].item()),
        "mean_shrink": float(shrink.mean().item()),
        "damping_alpha": damping_alpha,
        "fallback_mode": fallback_mode,
        "clamped": clamped,
        "bounded": bool(op_norm <= max_norm + 1e-12),
    }


@torch.no_grad()
def compute_condition_bounded_alphaedit_filter_from_activations(
    x_prose: torch.Tensor,
    x_spec: torch.Tensor,
    *,
    eta: float = 2.5,
    tau: float = DEFAULT_TAU,
    max_rank: int = DEFAULT_MAX_RANK,
    reg_spec: float = DEFAULT_REG_SPEC,
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM,
    center: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute oracle Ledoit-Wolf moments from raw activations and return a condition-bounded filter."""
    xp = _validate_finite_2d(x_prose, "x_prose")
    xs = _validate_finite_2d(x_spec, "x_spec")
    if xp.shape[1] != xs.shape[1]:
        raise ValueError(
            f"x_prose and x_spec must share feature width; got {xp.shape[1]} and {xs.shape[1]}"
        )

    m_prose_s, diag_p = compute_oracle_ledoit_wolf_moment(xp, center=center)
    m_spec_s, diag_s = compute_oracle_ledoit_wolf_moment(xs, center=center)
    g_star, stats = compute_condition_bounded_alphaedit_filter(
        m_prose_s,
        m_spec_s,
        n_prose=int(xp.shape[0]),
        n_spec=int(xs.shape[0]),
        eta=eta,
        tau=tau,
        max_rank=max_rank,
        reg_spec=reg_spec,
        max_operator_norm=max_operator_norm,
        shrink_moments=False,
    )
    stats["rho_prose"] = float(diag_p["rho"])
    stats["rho_spec"] = float(diag_s["rho"])
    stats["rho_lw_prose"] = float(diag_p["rho_lw"])
    stats["rho_lw_spec"] = float(diag_s["rho_lw"])
    stats["rho_oas_prose"] = float(diag_p["rho_oas"])
    stats["rho_oas_spec"] = float(diag_s["rho_oas"])
    return g_star, stats


@torch.no_grad()
def audit_filter_stability(
    g_mat: torch.Tensor,
    x_holdout_prose: torch.Tensor,
    x_holdout_spec: torch.Tensor,
    *,
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM,
    max_prose_energy_ratio: float = DEFAULT_MAX_PROSE_ENERGY_RATIO,
    max_specialist_rel_distortion: float = DEFAULT_MAX_SPECIALIST_REL_DISTORTION,
) -> dict[str, Any]:
    """Certify out-of-sample activation variance non-amplification and specialist preservation."""
    g64 = _validate_finite_2d(g_mat, "g_mat")
    if g64.shape[0] != g64.shape[1]:
        raise ValueError(f"g_mat must be square, got shape {tuple(g64.shape)}")
    d = int(g64.shape[0])

    xp = _validate_finite_2d(x_holdout_prose, "x_holdout_prose")
    xs = _validate_finite_2d(x_holdout_spec, "x_holdout_spec")
    if xp.shape[1] != d or xs.shape[1] != d:
        raise ValueError(
            f"Holdout matrices must have width {d}; got prose={xp.shape[1]}, spec={xs.shape[1]}"
        )

    max_norm = _validate_positive_finite(max_operator_norm, "max_operator_norm")
    if max_norm < 1.0:
        raise ValueError(f"max_operator_norm must be >= 1.0, got {max_operator_norm!r}")
    max_prose_ratio = _validate_positive_finite(max_prose_energy_ratio, "max_prose_energy_ratio")
    max_spec_dist = _validate_nonneg_finite(
        max_specialist_rel_distortion, "max_specialist_rel_distortion"
    )

    sv_g = torch.linalg.svdvals(g64)
    operator_norm = float(sv_g[0].item())
    min_singular_value = float(sv_g[-1].item())
    condition_number = float(operator_norm / max(min_singular_value, 1e-12))

    xp_filt = xp @ g64.T
    prose_before = float(xp.pow(2).sum().item())
    prose_after = float(xp_filt.pow(2).sum().item())
    prose_energy_ratio = (prose_after / prose_before) if prose_before > 1e-30 else 1.0

    xp_norms = xp.norm(dim=1)
    nz_mask = xp_norms > 1e-12
    if bool(nz_mask.any().item()):
        prose_max_token_amplification = float(
            (xp_filt[nz_mask].norm(dim=1) / xp_norms[nz_mask]).max().item()
        )
    else:
        prose_max_token_amplification = 1.0

    xs_filt = xs @ g64.T
    spec_before = float(xs.pow(2).sum().item())
    spec_after = float(xs_filt.pow(2).sum().item())
    spec_delta_energy = float((xs_filt - xs).pow(2).sum().item())
    specialist_rel_distortion = (
        math.sqrt(spec_delta_energy / spec_before) if spec_before > 1e-30 else 0.0
    )
    specialist_energy_ratio = (spec_after / spec_before) if spec_before > 1e-30 else 1.0

    operator_norm_ok = bool(operator_norm <= max_norm + 1e-12)
    prose_variance_non_amplified = bool(prose_energy_ratio <= max_prose_ratio + 1e-12)
    specialist_preserved = bool(specialist_rel_distortion <= max_spec_dist + 1e-12)
    is_stable = bool(operator_norm_ok and prose_variance_non_amplified and specialist_preserved)

    return {
        "operator_norm": operator_norm,
        "min_singular_value": min_singular_value,
        "condition_number": condition_number,
        "prose_energy_ratio": float(prose_energy_ratio),
        "prose_max_token_amplification": prose_max_token_amplification,
        "specialist_rel_distortion": float(specialist_rel_distortion),
        "specialist_energy_ratio": float(specialist_energy_ratio),
        "operator_norm_ok": operator_norm_ok,
        "prose_variance_non_amplified": prose_variance_non_amplified,
        "specialist_preserved": specialist_preserved,
        "is_stable": is_stable,
    }


@torch.no_grad()
def apply_condition_bounded_alphaedit_to_moe_layers_inplace(
    moe_layers: Sequence[nn.Module],
    x_prose: torch.Tensor,
    x_spec: torch.Tensor,
    *,
    eta_x: float = 2.5,
    eta_act: float = 1.5,
    tau: float = DEFAULT_TAU,
    max_rank_x: int = DEFAULT_MAX_RANK,
    max_rank_act: int = 128,
    reg_spec: float = DEFAULT_REG_SPEC,
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM,
    expert_indices: Sequence[int] | None = None,
    residual: bool = True,
    layer_norms: Sequence[nn.Module] | None = None,
) -> dict[str, Any]:
    """Sequentially edit SwiGLU LoRA MoE layers with condition-bounded AlphaEdit filters.

    For each layer in ``moe_layers``, optionally applies ``layer_norms[l]`` pre-MLP,
    computes oracle Ledoit-Wolf shrunk moments from the current hidden activations,
    constructs ``G_x`` (and ``G_act`` when ``eta_act > 0``) with
    ``||G||_2 <= max_operator_norm``, updates ``w1_A, w2_A`` (and ``w3_A``) in-place,
    and propagates activations forward to the next layer.
    """
    if not isinstance(moe_layers, Sequence) or len(moe_layers) == 0:
        raise ValueError("moe_layers must be a non-empty sequence of nn.Module layers")
    for idx, layer in enumerate(moe_layers):
        if not isinstance(layer, nn.Module):
            raise TypeError(f"moe_layers[{idx}] must be an nn.Module, got {type(layer).__name__}")
        if not hasattr(layer, "experts") or not hasattr(layer.experts, "keys"):
            raise ValueError(f"moe_layers[{idx}] must expose an 'experts' mapping")

    if layer_norms is not None:
        if not isinstance(layer_norms, Sequence) or len(layer_norms) != len(moe_layers):
            raise ValueError("layer_norms must match moe_layers length when provided")
        for idx, ln_mod in enumerate(layer_norms):
            if not isinstance(ln_mod, nn.Module):
                raise TypeError(f"layer_norms[{idx}] must be an nn.Module, got {type(ln_mod).__name__}")

    cur_p = _validate_finite_2d(x_prose, "x_prose").to(torch.float32)
    cur_s = _validate_finite_2d(x_spec, "x_spec").to(torch.float32)
    if cur_p.shape[1] != cur_s.shape[1]:
        raise ValueError(
            f"x_prose and x_spec must share feature width; got {cur_p.shape[1]} and {cur_s.shape[1]}"
        )

    _validate_nonneg_finite(eta_x, "eta_x")
    _validate_nonneg_finite(eta_act, "eta_act")
    _validate_nonneg_finite(tau, "tau")
    _validate_nonneg_int(max_rank_x, "max_rank_x")
    _validate_nonneg_int(max_rank_act, "max_rank_act")
    _validate_nonneg_finite(reg_spec, "reg_spec")
    max_norm = _validate_positive_finite(max_operator_norm, "max_operator_norm")
    if max_norm < 1.0:
        raise ValueError(f"max_operator_norm must be >= 1.0, got {max_operator_norm!r}")
    _validate_bool(residual, "residual")

    if expert_indices is not None:
        explicit_ids = [_validate_nonneg_int(e, "expert_index") for e in expert_indices]
    else:
        explicit_ids = None

    init_p_rms = max(float(cur_p.pow(2).mean().sqrt().item()), 1e-12)
    init_s_rms = max(float(cur_s.pow(2).mean().sqrt().item()), 1e-12)
    layer_reports: list[dict[str, Any]] = []
    max_op_seen = 1.0
    max_weight_ratio_seen = 1.0
    max_prose_rms_ratio = 1.0

    for l_idx, layer in enumerate(moe_layers):
        all_ids = sorted(int(k) for k in layer.experts.keys())
        targets = (
            [e for e in explicit_ids if str(e) in layer.experts]
            if explicit_ids is not None
            else all_ids
        )

        ln_mod = layer_norms[l_idx] if layer_norms is not None else None
        mlp_in_p = ln_mod(cur_p).to(torch.float32) if ln_mod is not None else cur_p
        mlp_in_s = ln_mod(cur_s).to(torch.float32) if ln_mod is not None else cur_s

        op_x = 1.0
        op_act = 1.0
        layer_max_w_ratio = 1.0

        if eta_x > 0.0 and targets:
            g_x, gx_stats = compute_condition_bounded_alphaedit_filter_from_activations(
                mlp_in_p,
                mlp_in_s,
                eta=eta_x,
                tau=tau,
                max_rank=max_rank_x,
                reg_spec=reg_spec,
                max_operator_norm=max_norm,
            )
            op_x = float(gx_stats["operator_norm"])
            if gx_stats["active_modes"] > 0:
                gx_f32 = g_x.to(dtype=torch.float32)
                for e_idx in targets:
                    exp = layer.experts[str(e_idx)]
                    for attr in ("w1_A", "w2_A"):
                        param = getattr(exp, attr)
                        before_norm = max(float(param.to(torch.float32).norm().item()), 1e-12)
                        updated = (param.to(torch.float32) @ gx_f32.to(param.device)).to(param.dtype)
                        after_norm = float(updated.to(torch.float32).norm().item())
                        layer_max_w_ratio = max(layer_max_w_ratio, after_norm / before_norm)
                        param.copy_(updated)

        if eta_act > 0.0 and targets:
            act_p = compute_swiglu_intermediate_states(layer, mlp_in_p, targets)
            act_s = compute_swiglu_intermediate_states(layer, mlp_in_s, targets)
            g_a, ga_stats = compute_condition_bounded_alphaedit_filter_from_activations(
                act_p,
                act_s,
                eta=eta_act,
                tau=tau,
                max_rank=max_rank_act,
                reg_spec=reg_spec,
                max_operator_norm=max_norm,
            )
            op_act = float(ga_stats["operator_norm"])
            if ga_stats["active_modes"] > 0:
                ga_f32 = g_a.to(dtype=torch.float32)
                for e_idx in targets:
                    exp = layer.experts[str(e_idx)]
                    param = exp.w3_A
                    before_norm = max(float(param.to(torch.float32).norm().item()), 1e-12)
                    updated = (param.to(torch.float32) @ ga_f32.to(param.device)).to(param.dtype)
                    after_norm = float(updated.to(torch.float32).norm().item())
                    layer_max_w_ratio = max(layer_max_w_ratio, after_norm / before_norm)
                    param.copy_(updated)

        out_p = layer(mlp_in_p).to(torch.float32)
        out_s = layer(mlp_in_s).to(torch.float32)
        cur_p = (cur_p + out_p) if residual else out_p
        cur_s = (cur_s + out_s) if residual else out_s

        p_rms_ratio = float(cur_p.pow(2).mean().sqrt().item()) / init_p_rms
        s_rms_ratio = float(cur_s.pow(2).mean().sqrt().item()) / init_s_rms
        max_op_seen = max(max_op_seen, op_x, op_act)
        max_weight_ratio_seen = max(max_weight_ratio_seen, layer_max_w_ratio)
        max_prose_rms_ratio = max(max_prose_rms_ratio, p_rms_ratio)

        layer_reports.append(
            {
                "layer": l_idx,
                "targets": targets,
                "operator_norm_x": op_x,
                "operator_norm_act": op_act,
                "max_weight_norm_ratio": layer_max_w_ratio,
                "prose_rms_ratio": p_rms_ratio,
                "spec_rms_ratio": s_rms_ratio,
            }
        )

    return {
        "num_layers": len(moe_layers),
        "max_operator_norm_seen": max_op_seen,
        "max_weight_norm_ratio": max_weight_ratio_seen,
        "max_prose_rms_ratio": max_prose_rms_ratio,
        "bounded_across_depth": bool(max_op_seen <= max_norm + 1e-12),
        "layers": layer_reports,
    }


__all__ = [
    "DEFAULT_MAX_OPERATOR_NORM",
    "DEFAULT_MAX_PROSE_ENERGY_RATIO",
    "DEFAULT_MAX_RANK",
    "DEFAULT_MAX_SPECIALIST_REL_DISTORTION",
    "DEFAULT_REG_SPEC",
    "DEFAULT_TAU",
    "MarchenkoPasturGuardConfig",
    "apply_condition_bounded_alphaedit_to_moe_layers_inplace",
    "audit_filter_stability",
    "compute_condition_bounded_alphaedit_filter",
    "compute_condition_bounded_alphaedit_filter_from_activations",
    "compute_oracle_ledoit_wolf_moment",
    "shrink_moment_oracle_oas",
]
