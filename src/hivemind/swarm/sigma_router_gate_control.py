# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Contrastive ``sigma_topk`` multi-gate router controller with 2-fold OOF reliability shutoff.

Solves the non-monotonic ``sigma_topk`` interpolation trap in upcycled Sparse MoE routers
(``CrowdLoRAMoEMLP`` and ``CrowdFineGrainedMoEMLP``), where independent unnormalized sigmoid
gates ``sigma(z)`` satisfy ``sigma(0) = 0.50``. Shrinking router weights toward zero via
uniform WiSE-FT pushes negative off-domain logits up toward ``0.50``, opening inappropriate
specialist gates on general prose while attenuating specialist confidence.

Implements three training-free ``float64`` linear-algebraic primitives:

1. ``fit_uncentered_prose_probe_with_oof``:
   Solves ``(M_0 + omega_spec * M_spec + lam * I) u = m_0`` normalized to
   ``E_prose[u^T x] == 1`` with 2-fold out-of-fold (OOF) SNR and specialist-RMS
   gating that shuts off cleanly (``rel_weight == 0.0``) on inseparable or noisy layers.
2. ``fit_per_domain_orthogonal_gate_vectors``:
   Fits per-expert zero-crosstalk directions ``u_{0 perp e}`` solving
   ``(M_0 + omega_own * M_e + omega_other * M_pool + lam * I) u_raw = m_0`` so
   suppressing expert ``e`` on general prose causes near-zero cross-talk on domain ``e``.
3. ``apply_sigma_router_prose_control_inplace``:
   Applies negative logit suppression on off-domain router rows while adaptively
   protecting helpful general experts (``expert_0``) or steering prose tokens toward
   zeroed FoMoE ghost sinks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

GENERAL_EXPERT_INDEX: int = 0
DEFAULT_OMEGA_SPEC: float = 6.0
DEFAULT_OMEGA_OWN: float = 8.0
DEFAULT_OMEGA_OTHER: float = 2.0
DEFAULT_RIDGE_SCALE: float = 0.75


def _validate_nonneg_finite(value: float, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: float, name: str) -> float:
    """Validate that ``value`` is finite and lies in ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _validate_nonneg_int(value: int, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


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


def _validate_direction_1d(tensor: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    """Validate a 1D finite direction tensor of length ``expected_dim`` in CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1 or int(tensor.shape[0]) != expected_dim:
        raise ValueError(f"{name} must be a 1D tensor of length {expected_dim}, got shape {tuple(tensor.shape)}")
    vec = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(vec).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return vec


@dataclass(frozen=True)
class SigmaRouterControlConfig:
    """Hyperparameters governing contrastive ``sigma_topk`` router gate control."""

    omega_spec: float = DEFAULT_OMEGA_SPEC
    omega_own: float = DEFAULT_OMEGA_OWN
    omega_other: float = DEFAULT_OMEGA_OTHER
    ridge_scale: float = DEFAULT_RIDGE_SCALE
    beta_suppress: float = 2.0
    beta_sink: float = 1.5
    beta_general_boost: float = 0.0
    min_domain_samples: int = 8
    general_expert_index: int = GENERAL_EXPERT_INDEX

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.omega_spec, "omega_spec")
        _validate_nonneg_finite(self.omega_own, "omega_own")
        _validate_nonneg_finite(self.omega_other, "omega_other")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_nonneg_finite(self.beta_suppress, "beta_suppress")
        _validate_nonneg_finite(self.beta_sink, "beta_sink")
        _validate_nonneg_finite(self.beta_general_boost, "beta_general_boost")
        _validate_nonneg_int(self.min_domain_samples, "min_domain_samples")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")


def second_moment_and_mean(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uncentered second moment ``E[x x^T]`` and mean ``E[x]`` in ``float64``."""
    xd = _validate_finite_2d(x_2d, "x_2d")
    n_rows = float(xd.shape[0])
    m2 = (xd.T @ xd) / n_rows
    m2 = 0.5 * (m2 + m2.T)
    mean = xd.mean(dim=0)
    return m2, mean


def _solve_uncentered_probe_direction(
    m_prose: torch.Tensor,
    mu_prose: torch.Tensor,
    x_prose_f64: torch.Tensor,
    penalty_moment: torch.Tensor,
    ridge_scale: float,
) -> tuple[torch.Tensor, float]:
    """Solve ``(M_0 + penalty_moment + lam * I) u = mu_0`` normalized to ``E_0[u^T x] == 1``."""
    d = int(mu_prose.shape[0])
    comb = 0.5 * ((m_prose + penalty_moment) + (m_prose + penalty_moment).T)
    tr_scale = max(float(torch.diagonal(comb).mean().item()), 1e-6)
    lam = max(float(ridge_scale) * tr_scale, 1e-5)
    reg_mat = comb + lam * torch.eye(d, dtype=torch.float64)
    u_raw = torch.linalg.solve(reg_mat, mu_prose)
    denom = float((x_prose_f64 @ u_raw).mean().item())
    if denom <= 1e-6:
        return torch.zeros(d, dtype=torch.float64), lam
    return u_raw / denom, lam


def fit_uncentered_prose_probe_with_oof(
    prose_fold0: torch.Tensor,
    prose_fold1: torch.Tensor,
    spec_fold0: torch.Tensor,
    spec_fold1: torch.Tensor,
    *,
    omega_spec: float = DEFAULT_OMEGA_SPEC,
    ridge_scale: float = DEFAULT_RIDGE_SCALE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit uncentered linear direction ``u`` solving ``(M_0 + omega_spec*M_spec + lam*I) u = m_0``.

    Performs 2-fold out-of-fold validation across ``(fold0, fold1)`` to estimate ``oof_snr``,
    specialist RMS leakage ``oof_spec_rms``, and a reliability gate ``rel_weight in [0.0, 1.0]``.
    When representations are inseparable or noisy, ``rel_weight`` shuts off cleanly to ``0.0``.
    """
    p0 = _validate_finite_2d(prose_fold0, "prose_fold0")
    p1 = _validate_finite_2d(prose_fold1, "prose_fold1")
    s0 = _validate_finite_2d(spec_fold0, "spec_fold0")
    s1 = _validate_finite_2d(spec_fold1, "spec_fold1")
    d = int(p0.shape[1])
    if p1.shape[1] != d or s0.shape[1] != d or s1.shape[1] != d:
        raise ValueError("All probe fold tensors must share the same feature dimension")

    omega_f = _validate_nonneg_finite(omega_spec, "omega_spec")
    ridge_f = _validate_nonneg_finite(ridge_scale, "ridge_scale")

    def _fit_fold(px: torch.Tensor, sx: torch.Tensor) -> tuple[torch.Tensor, float]:
        m0, mu0 = second_moment_and_mean(px)
        ms, _ = second_moment_and_mean(sx)
        return _solve_uncentered_probe_direction(m0, mu0, px, omega_f * ms, ridge_f)

    u0, _ = _fit_fold(p0, s0)
    u1, _ = _fit_fold(p1, s1)

    p_oof_f0 = p1 @ u0
    p_oof_f1 = p0 @ u1
    s_oof_f0 = s1 @ u0
    s_oof_f1 = s0 @ u1

    p_oof = torch.cat([p_oof_f0, p_oof_f1], dim=0)
    s_oof = torch.cat([s_oof_f0, s_oof_f1], dim=0)

    p_mean = float(p_oof.mean().item())
    s_mean = float(s_oof.mean().item())
    s_rms = float(s_oof.pow(2).mean().sqrt().item())
    pooled_var = 0.5 * (float(p_oof.var(unbiased=False).item()) + float(s_oof.var(unbiased=False).item()))
    pooled_std = math.sqrt(pooled_var + 1e-8)
    margin = p_mean - max(0.0, s_mean)
    oof_snr = max(0.0, margin / pooled_std)

    f0_p = float(p_oof_f0.mean().item())
    f1_p = float(p_oof_f1.mean().item())
    f0_s = max(0.0, float(s_oof_f0.mean().item()))
    f1_s = max(0.0, float(s_oof_f1.mean().item()))
    folds_agree = (f0_p > 0.25) and (f1_p > 0.25) and (f0_p > f0_s) and (f1_p > f1_s)
    spec_margin_ok = (margin > 0.50 * s_rms) and (p_mean > 0.75 * s_rms)

    if folds_agree and spec_margin_ok:
        rel_weight = max(0.0, min(1.0, (oof_snr - 0.25) / 0.75))
    else:
        rel_weight = 0.0

    u_full, lam_full = _fit_fold(torch.cat([p0, p1], dim=0), torch.cat([s0, s1], dim=0))
    return u_full, {
        "oof_prose_mean": p_mean,
        "oof_spec_mean": s_mean,
        "oof_spec_rms": s_rms,
        "oof_snr": oof_snr,
        "rel_weight": rel_weight,
        "ridge_lambda": lam_full,
    }


def fit_per_domain_orthogonal_gate_vectors(
    prose_x: torch.Tensor,
    domain_x_map: Mapping[int, torch.Tensor],
    pooled_spec_x: torch.Tensor | None = None,
    *,
    omega_own: float = DEFAULT_OMEGA_OWN,
    omega_other: float = DEFAULT_OMEGA_OTHER,
    ridge_scale: float = DEFAULT_RIDGE_SCALE,
    min_domain_samples: int = 8,
) -> dict[int, torch.Tensor]:
    """Fit per-expert zero-crosstalk control directions ``u_{0 perp e}`` across specialist domains.

    For each expert ``e`` in ``domain_x_map``, solves:
    ``(M_0 + omega_own * M_e + omega_other * M_pool + lam * I) u_raw = m_0``
    and normalizes ``u_{0 perp e} = u_raw / E_prose[u_raw^T x]`` so suppressing expert ``e``
    on general prose induces minimal cross-talk on domain ``e``.
    """
    p_x = _validate_finite_2d(prose_x, "prose_x")
    d = int(p_x.shape[1])
    if not isinstance(domain_x_map, Mapping) or not domain_x_map:
        raise ValueError("domain_x_map must be a non-empty mapping of expert_index -> 2D tensor")

    omega_own_f = _validate_nonneg_finite(omega_own, "omega_own")
    omega_other_f = _validate_nonneg_finite(omega_other, "omega_other")
    ridge_f = _validate_nonneg_finite(ridge_scale, "ridge_scale")
    min_samples = _validate_nonneg_int(min_domain_samples, "min_domain_samples")

    validated_domains: dict[int, torch.Tensor] = {}
    for raw_key, x_dom in domain_x_map.items():
        e_idx = _validate_nonneg_int(raw_key, "domain_x_map key")
        xd = _validate_finite_2d(x_dom, f"domain_x_map[{e_idx}]")
        if int(xd.shape[1]) != d:
            raise ValueError(
                f"domain_x_map[{e_idx}] feature dimension {xd.shape[1]} does not match prose_x width {d}"
            )
        validated_domains[e_idx] = xd

    if pooled_spec_x is None:
        pool_x = torch.cat([validated_domains[k] for k in sorted(validated_domains)], dim=0)
    else:
        pool_x = _validate_finite_2d(pooled_spec_x, "pooled_spec_x")
        if int(pool_x.shape[1]) != d:
            raise ValueError(f"pooled_spec_x feature dimension {pool_x.shape[1]} does not match prose_x width {d}")

    m0, mu0 = second_moment_and_mean(p_x)
    m_pool, _ = second_moment_and_mean(pool_x)

    out: dict[int, torch.Tensor] = {}
    for e_idx in sorted(validated_domains):
        x_dom = validated_domains[e_idx]
        if int(x_dom.shape[0]) >= min_samples:
            me, _ = second_moment_and_mean(x_dom)
            penalty = omega_own_f * me + omega_other_f * m_pool
        else:
            penalty = (omega_own_f + omega_other_f) * m_pool
        u_vec, _ = _solve_uncentered_probe_direction(m0, mu0, p_x, penalty, ridge_f)
        out[e_idx] = u_vec
    return out


def _resolve_direction_for_expert(
    expert_idx: int,
    control_vectors: torch.Tensor | Mapping[int, torch.Tensor],
    fallback_direction: torch.Tensor | None,
) -> torch.Tensor | None:
    """Resolve a validated float64 direction tensor for ``expert_idx``."""
    if isinstance(control_vectors, torch.Tensor):
        return control_vectors
    if expert_idx in control_vectors:
        return control_vectors[expert_idx]
    return fallback_direction


def _resolve_sink_direction(
    target_idx: int,
    control_vectors: torch.Tensor | Mapping[int, torch.Tensor],
    sink_direction: torch.Tensor | None,
    fallback_direction: torch.Tensor | None,
) -> torch.Tensor | None:
    """Resolve the prose attractor direction for a sink or protected general expert."""
    if sink_direction is not None:
        return sink_direction
    if isinstance(control_vectors, torch.Tensor):
        return control_vectors
    if target_idx in control_vectors:
        return control_vectors[target_idx]
    if fallback_direction is not None:
        return fallback_direction
    vecs = list(control_vectors.values())
    return torch.stack(vecs, dim=0).mean(dim=0)


@torch.no_grad()
def apply_sigma_router_prose_control_inplace(
    router_linear: nn.Linear,
    control_vectors: torch.Tensor | Mapping[int, torch.Tensor],
    *,
    rel_weight: float = 1.0,
    beta_suppress: float = 2.0,
    beta_sink: float = 1.5,
    beta_general_boost: float = 0.0,
    sink_experts: Sequence[int] = (),
    expert0_helps_prose: bool = True,
    general_expert_index: int = GENERAL_EXPERT_INDEX,
    sink_direction: torch.Tensor | None = None,
    fallback_direction: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Shift off-domain ``sigma_topk`` router logits negatively on prose while protecting helpful experts.

    When ``rel_weight == 0.0``, exits immediately with zero mutation to ``router_linear``.
    When ``general_expert_index`` is zeroed in ``sink_experts``, applies positive sink attraction
    (``+beta_sink * rel_weight * u``) so ``sigma_topk`` tokens on prose route safely into the
    zero-perturbation sink; when ``expert0_helps_prose=True`` and ``general_expert_index`` is
    not zeroed, protects it from suppression (leaving its row intact unless ``beta_general_boost > 0``).
    """
    if not isinstance(router_linear, nn.Linear):
        raise TypeError(f"router_linear must be an nn.Linear, got {type(router_linear).__name__}")
    if router_linear.weight.ndim != 2 or router_linear.weight.shape[0] == 0 or router_linear.weight.shape[1] == 0:
        raise ValueError(f"router_linear.weight must be non-empty 2D, got {tuple(router_linear.weight.shape)}")
    if not isinstance(expert0_helps_prose, bool):
        raise TypeError(f"expert0_helps_prose must be a bool, got {type(expert0_helps_prose).__name__}")

    num_experts, hidden_size = int(router_linear.weight.shape[0]), int(router_linear.weight.shape[1])
    rel_f = _validate_unit_interval(rel_weight, "rel_weight")
    beta_sup_f = _validate_nonneg_finite(beta_suppress, "beta_suppress")
    beta_sink_f = _validate_nonneg_finite(beta_sink, "beta_sink")
    beta_boost_f = _validate_nonneg_finite(beta_general_boost, "beta_general_boost")

    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")
    if gen_idx >= num_experts:
        raise ValueError(f"general_expert_index={gen_idx} out of range [0, {num_experts})")

    sink_set: set[int] = set()
    for raw_sink in sink_experts:
        s_idx = _validate_nonneg_int(raw_sink, "sink_expert")
        if s_idx >= num_experts:
            raise ValueError(f"sink_expert={s_idx} out of range [0, {num_experts})")
        sink_set.add(s_idx)
    sink_sorted = sorted(sink_set)

    validated_controls: torch.Tensor | dict[int, torch.Tensor]
    if isinstance(control_vectors, torch.Tensor):
        validated_controls = _validate_direction_1d(control_vectors, hidden_size, "control_vectors")
    elif isinstance(control_vectors, Mapping):
        if not control_vectors:
            raise ValueError("control_vectors mapping must be non-empty")
        validated_map: dict[int, torch.Tensor] = {}
        for raw_k, raw_v in control_vectors.items():
            e_idx = _validate_nonneg_int(raw_k, "control_vectors key")
            if e_idx >= num_experts:
                raise ValueError(f"control_vectors key {e_idx} out of range [0, {num_experts})")
            validated_map[e_idx] = _validate_direction_1d(raw_v, hidden_size, f"control_vectors[{e_idx}]")
        validated_controls = validated_map
    else:
        raise TypeError("control_vectors must be a 1D torch.Tensor or Mapping[int, torch.Tensor]")

    val_sink_dir = (
        _validate_direction_1d(sink_direction, hidden_size, "sink_direction") if sink_direction is not None else None
    )
    val_fallback_dir = (
        _validate_direction_1d(fallback_direction, hidden_size, "fallback_direction")
        if fallback_direction is not None
        else None
    )

    surviving = [e for e in range(num_experts) if e not in sink_set]
    general_protected = (gen_idx in surviving) and expert0_helps_prose
    off_domain_experts = [e for e in surviving if not (general_protected and e == gen_idx)]
    sink_target = (gen_idx if gen_idx in sink_set else sink_sorted[0]) if sink_sorted else None

    if rel_f == 0.0 or (beta_sup_f == 0.0 and beta_sink_f == 0.0 and beta_boost_f == 0.0):
        return {
            "applied": False,
            "rel_weight": rel_f,
            "suppressed_experts": [],
            "protected_experts": [gen_idx] if general_protected else [],
            "sink_experts": sink_sorted,
            "sink_target": sink_target,
            "weight_delta_l2": 0.0,
        }

    w_before = router_linear.weight.detach().to(torch.float32).clone()
    dev = router_linear.weight.device
    dt = router_linear.weight.dtype

    suppressed: list[int] = []
    if beta_sup_f > 0.0:
        for e_idx in off_domain_experts:
            vec = _resolve_direction_for_expert(e_idx, validated_controls, val_fallback_dir)
            if vec is None or float(vec.norm().item()) == 0.0:
                continue
            step = (beta_sup_f * rel_f) * vec.to(device=dev, dtype=dt)
            router_linear.weight[e_idx].sub_(step)
            suppressed.append(e_idx)

    if sink_target is not None and beta_sink_f > 0.0:
        u_sink = _resolve_sink_direction(sink_target, validated_controls, val_sink_dir, val_fallback_dir)
        if u_sink is not None and float(u_sink.norm().item()) > 0.0:
            router_linear.weight[sink_target].add_((beta_sink_f * rel_f) * u_sink.to(device=dev, dtype=dt))

    if general_protected and beta_boost_f > 0.0:
        u_gen = _resolve_sink_direction(gen_idx, validated_controls, val_sink_dir, val_fallback_dir)
        if u_gen is not None and float(u_gen.norm().item()) > 0.0:
            router_linear.weight[gen_idx].add_((beta_boost_f * rel_f) * u_gen.to(device=dev, dtype=dt))

    delta_l2 = float((router_linear.weight.detach().to(torch.float32) - w_before).norm().item())
    return {
        "applied": delta_l2 > 0.0,
        "rel_weight": rel_f,
        "suppressed_experts": suppressed,
        "protected_experts": [gen_idx] if general_protected else [],
        "sink_experts": sink_sorted,
        "sink_target": sink_target,
        "weight_delta_l2": delta_l2,
    }


@torch.no_grad()
def interpolate_router_weights(
    start_router: nn.Linear,
    tuned_router: nn.Linear,
    lam: float,
) -> nn.Linear:
    """Return a new ``nn.Linear`` holding ``(1 - lam) * start_router + lam * tuned_router``."""
    if not isinstance(start_router, nn.Linear) or not isinstance(tuned_router, nn.Linear):
        raise TypeError("start_router and tuned_router must be nn.Linear instances")
    if start_router.weight.shape != tuned_router.weight.shape:
        raise ValueError(
            f"Router shape mismatch: {tuple(start_router.weight.shape)} vs {tuple(tuned_router.weight.shape)}"
        )
    if (start_router.bias is None) != (tuned_router.bias is None):
        raise ValueError("start_router and tuned_router must both either have or omit bias")
    lam_f = _validate_unit_interval(lam, "lam")

    has_bias = start_router.bias is not None
    out_features, in_features = int(start_router.weight.shape[0]), int(start_router.weight.shape[1])
    out = nn.Linear(
        in_features,
        out_features,
        bias=has_bias,
        device=tuned_router.weight.device,
        dtype=tuned_router.weight.dtype,
    )
    w_s = start_router.weight.detach().to(torch.float32)
    w_t = tuned_router.weight.detach().to(torch.float32)
    out.weight.copy_((w_s + lam_f * (w_t - w_s)).to(dtype=out.weight.dtype))
    if has_bias and out.bias is not None and start_router.bias is not None and tuned_router.bias is not None:
        b_s = start_router.bias.detach().to(torch.float32)
        b_t = tuned_router.bias.detach().to(torch.float32)
        out.bias.copy_((b_s + lam_f * (b_t - b_s)).to(dtype=out.bias.dtype))
    return out


@torch.no_grad()
def evaluate_sigma_topk_routing_gates(
    router_linear: nn.Linear,
    prose_x: torch.Tensor,
    domain_x_map: Mapping[int, torch.Tensor],
    *,
    top_k: int = 2,
    general_expert_index: int = GENERAL_EXPERT_INDEX,
) -> dict[str, Any]:
    """Measure unnormalized ``sigma_topk`` sigmoid gate openings on prose and specialist domains."""
    if not isinstance(router_linear, nn.Linear):
        raise TypeError(f"router_linear must be an nn.Linear, got {type(router_linear).__name__}")
    num_experts, hidden_size = int(router_linear.weight.shape[0]), int(router_linear.weight.shape[1])
    k_val = _validate_nonneg_int(top_k, "top_k")
    if k_val < 1 or k_val > num_experts:
        raise ValueError(f"top_k must satisfy 1 <= top_k <= {num_experts}, got {k_val}")
    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")
    if gen_idx >= num_experts:
        raise ValueError(f"general_expert_index={gen_idx} out of range [0, {num_experts})")

    p_f64 = _validate_finite_2d(prose_x, "prose_x")
    if int(p_f64.shape[1]) != hidden_size:
        raise ValueError(f"prose_x width {p_f64.shape[1]} does not match router in_features {hidden_size}")
    if not isinstance(domain_x_map, Mapping) or not domain_x_map:
        raise ValueError("domain_x_map must be a non-empty mapping of expert_index -> 2D tensor")

    w_dt = router_linear.weight.dtype
    w_dev = router_linear.weight.device
    p_logits = router_linear(p_f64.to(device=w_dev, dtype=w_dt)).to(torch.float32)
    p_gates = torch.sigmoid(p_logits)
    _, p_top_idx = torch.topk(p_gates, k=k_val, dim=-1)

    prose_gate_by_expert = {e: float(p_gates[:, e].mean().item()) for e in range(num_experts)}
    off_domain_gates = [prose_gate_by_expert[e] for e in range(num_experts) if e != gen_idx]
    prose_offdomain_max_gate = max(off_domain_gates) if off_domain_gates else 0.0
    prose_offdomain_topk_fraction = float((p_top_idx != gen_idx).to(torch.float32).mean().item())

    domain_own_gate: dict[int, float] = {}
    domain_own_topk_hit_rate: dict[int, float] = {}
    for raw_k, x_dom in domain_x_map.items():
        e_idx = _validate_nonneg_int(raw_k, "domain_x_map key")
        if e_idx >= num_experts:
            raise ValueError(f"domain_x_map key {e_idx} out of range [0, {num_experts})")
        xd_f64 = _validate_finite_2d(x_dom, f"domain_x_map[{e_idx}]")
        if int(xd_f64.shape[1]) != hidden_size:
            raise ValueError(
                f"domain_x_map[{e_idx}] width {xd_f64.shape[1]} does not match router in_features {hidden_size}"
            )
        d_logits = router_linear(xd_f64.to(device=w_dev, dtype=w_dt)).to(torch.float32)
        d_gates = torch.sigmoid(d_logits)
        _, d_top_idx = torch.topk(d_gates, k=k_val, dim=-1)
        domain_own_gate[e_idx] = float(d_gates[:, e_idx].mean().item())
        domain_own_topk_hit_rate[e_idx] = float((d_top_idx == e_idx).any(dim=-1).to(torch.float32).mean().item())

    return {
        "prose_gate_by_expert": prose_gate_by_expert,
        "prose_general_gate": prose_gate_by_expert[gen_idx],
        "prose_offdomain_max_gate": prose_offdomain_max_gate,
        "prose_offdomain_topk_fraction": prose_offdomain_topk_fraction,
        "domain_own_gate": domain_own_gate,
        "domain_min_own_gate": min(domain_own_gate.values()),
        "domain_own_topk_hit_rate": domain_own_topk_hit_rate,
    }


__all__ = [
    "DEFAULT_OMEGA_OTHER",
    "DEFAULT_OMEGA_OWN",
    "DEFAULT_OMEGA_SPEC",
    "DEFAULT_RIDGE_SCALE",
    "GENERAL_EXPERT_INDEX",
    "SigmaRouterControlConfig",
    "apply_sigma_router_prose_control_inplace",
    "evaluate_sigma_topk_routing_gates",
    "fit_per_domain_orthogonal_gate_vectors",
    "fit_uncentered_prose_probe_with_oof",
    "interpolate_router_weights",
    "second_moment_and_mean",
]
