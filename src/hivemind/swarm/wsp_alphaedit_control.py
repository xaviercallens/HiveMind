# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""WSP-AlphaEdit-MoE: Discriminative Weight-Space Control for Open-Weight Sparse MoE LoRA.

Implements float64 generalized-eigenvalue minimum-specialist-distortion Weight-Space
Programming (WSP) primitives compatible with ``CrowdV1WithEngram`` and
``CrowdLoRAMoEMLP`` (``sigma_topk`` / ``softmax_topk`` SwiGLU rank-r adapters):

1. Dual-space ``Discriminative-AlphaEdit`` filter (``compute_discriminative_alphaedit_filter``):
   solves ``min_{W'} E_spec[||(W' - W) x||^2] + eta * E_prose[||W' x||^2] + lam ||W' - W||_F^2``
   restricted to generalized modes ``(M_prose, B)`` with variance ratio ``mu_i > tau``,
   guaranteeing exact zero distortion on specialist modes (``mu_i <= tau``).
2. FoMoE Ghost-Sink Zeroing (``zero_expert_lora_b_inplace``):
   enforces the exact bitwise ``w1_B = w2_B = w3_B = 0 => dy_e(x) == 0`` identity.
3. Contrastive ``sigma_topk`` Router Gate Surgery (``fit_uncentered_prose_probe_with_oof``):
   fits an uncentered prose-vs-specialist separator with 2-fold out-of-fold SNR gating.
4. Guard-Optimal Final LayerNorm Calibration (``select_guard_optimal_lambda_ln``):
   interpolates post-surgery final LayerNorm weights subject to a strict general-domain
   drift guard (``<= +0.5%`` vs dense base), with minimum-drift fallback when unfeasible.
5. Sequential Layer-Wise Sparse MoE Synthesis (``calibrate_final_layernorm_and_sink_experts``
   and ``synthesize_wsp_arm_buckets``) and ANSE guard/retention telemetry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
)

GENERAL_EXPERT_PREFIX: str = "E0_"
DOMAIN_GUARD_MAX_RATIO: float = 1.01
LN_GUARD_MAX_RATIO: float = 1.005
ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_LAMBDA_LN_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


def _validate_nonneg_finite(value: float, name: str) -> float:
    """Validate that a numeric parameter is finite and non-negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_nonneg_int(value: int, name: str) -> int:
    """Validate that an integer parameter is non-negative and not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_finite_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a non-empty 2D tensor and return a detached CPU float64 copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    xd = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(xd).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return xd


@dataclass(frozen=True)
class WSPArmConfig:
    """Hyperparameters defining one deterministic Weight-Space Programming candidate."""

    name: str
    description: str = ""
    use_ln_calibration: bool = True
    use_ghost_sink: bool = True
    force_zero_expert0: bool = False
    router_beta_suppress: float = 0.0
    router_beta_sink: float = 0.0
    router_omega_spec: float = 6.0
    router_ridge_scale: float = 0.75
    alphaedit_eta_x: float = 0.0
    alphaedit_eta_act: float = 0.0
    alphaedit_tau: float = 1.25
    alphaedit_max_rank_x: int = 96
    alphaedit_max_rank_act: int = 128
    alphaedit_reg_spec: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("WSPArmConfig.name must be a non-empty string")
        _validate_nonneg_finite(self.router_beta_suppress, "router_beta_suppress")
        _validate_nonneg_finite(self.router_beta_sink, "router_beta_sink")
        _validate_nonneg_finite(self.router_omega_spec, "router_omega_spec")
        _validate_nonneg_finite(self.router_ridge_scale, "router_ridge_scale")
        _validate_nonneg_finite(self.alphaedit_eta_x, "alphaedit_eta_x")
        _validate_nonneg_finite(self.alphaedit_eta_act, "alphaedit_eta_act")
        _validate_nonneg_finite(self.alphaedit_tau, "alphaedit_tau")
        _validate_nonneg_finite(self.alphaedit_reg_spec, "alphaedit_reg_spec")
        _validate_nonneg_int(self.alphaedit_max_rank_x, "alphaedit_max_rank_x")
        _validate_nonneg_int(self.alphaedit_max_rank_act, "alphaedit_max_rank_act")


# ---------------------------------------------------------------------------
# 1. Float64 Linear-Algebraic Weight-Space Primitives
# ---------------------------------------------------------------------------


def second_moment_and_mean(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uncentered second moment ``E[x x^T]`` and mean ``E[x]`` in ``float64``."""
    xd = _validate_finite_2d(x_2d, "x_2d")
    n = float(xd.shape[0])
    m2 = (xd.T @ xd) / n
    m2 = 0.5 * (m2 + m2.T)
    mean = xd.mean(dim=0)
    return m2, mean


def compute_discriminative_alphaedit_filter(
    m_prose: torch.Tensor,
    m_spec: torch.Tensor,
    *,
    eta: float = 2.0,
    tau: float = 1.25,
    max_rank: int = 96,
    reg_spec: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute exact minimum-specialist-distortion right filter ``G^*`` for ``W_A <- W_A @ G^*``.

    Solves ``min_{W'} E_spec[||(W' - W) x||^2] + eta * E_prose[||W' x||^2] + lam_spec ||W' - W||_F^2``
    restricted to generalized modes ``(M_prose, B)`` whose variance ratio ``mu_i > tau``:

        ``G^* = I_d - B^{1/2} Q_p diag(eta * (mu_i - tau) / (1 + eta * (mu_i - tau))) Q_p^T B^{-1/2}``

    where ``B = M_spec + lam_spec * I_d`` and ``B^{-1/2} M_prose B^{-1/2} = Q diag(mu) Q^T``.
    Specialist modes satisfying ``mu_i <= tau`` incur zero distortion.
    """
    mp_raw = _validate_finite_2d(m_prose, "m_prose")
    ms_raw = _validate_finite_2d(m_spec, "m_spec")
    d = int(mp_raw.shape[0])
    if mp_raw.shape != (d, d) or ms_raw.shape != (d, d):
        raise ValueError(
            f"Moment matrices must be square and share shape; got {tuple(mp_raw.shape)} and {tuple(ms_raw.shape)}"
        )

    eta_f = _validate_nonneg_finite(eta, "eta")
    tau_f = _validate_nonneg_finite(tau, "tau")
    reg_f = _validate_nonneg_finite(reg_spec, "reg_spec")
    max_r = _validate_nonneg_int(max_rank, "max_rank")

    if eta_f == 0.0 or max_r == 0:
        return torch.eye(d, dtype=torch.float64), {
            "active_modes": 0,
            "max_ratio": 0.0,
            "variance_ratio": 0.0,
            "mean_shrink": 0.0,
        }

    mp = 0.5 * (mp_raw + mp_raw.T)
    ms = 0.5 * (ms_raw + ms_raw.T)

    tr_scale = max(float(torch.diagonal(ms).mean().item()), 1e-6)
    lam_spec = max(reg_f * tr_scale, 1e-6)
    b_mat = ms + lam_spec * torch.eye(d, dtype=torch.float64)

    evals_b, evecs_b = torch.linalg.eigh(b_mat)
    evals_b_clamped = evals_b.clamp_min(1e-8)
    b_half = (evecs_b * evals_b_clamped.sqrt().unsqueeze(0)) @ evecs_b.T
    b_inv_half = (evecs_b * evals_b_clamped.rsqrt().unsqueeze(0)) @ evecs_b.T

    whitened = b_inv_half @ mp @ b_inv_half
    whitened = 0.5 * (whitened + whitened.T)
    mu_asc, q_asc = torch.linalg.eigh(whitened)

    mu = mu_asc.flip(0)
    q = q_asc.flip(1)

    excess = mu - tau_f
    n_above = int((excess > 0.0).sum().item())
    k_eff = min(max_r, n_above)
    max_ratio = float(mu[0].item())
    if k_eff == 0:
        return torch.eye(d, dtype=torch.float64), {
            "active_modes": 0,
            "max_ratio": max_ratio,
            "variance_ratio": max_ratio,
            "mean_shrink": 0.0,
        }

    q_p = q[:, :k_eff]
    ex_p = excess[:k_eff]
    shrink = (eta_f * ex_p) / (1.0 + eta_f * ex_p)

    left = b_half @ q_p
    right = (q_p * shrink.unsqueeze(0)).T @ b_inv_half
    g_star = torch.eye(d, dtype=torch.float64) - (left @ right)

    return g_star, {
        "active_modes": k_eff,
        "max_ratio": max_ratio,
        "variance_ratio": max_ratio,
        "min_selected_ratio": float(mu[k_eff - 1].item()),
        "mean_shrink": float(shrink.mean().item()),
    }


def fit_uncentered_prose_probe_with_oof(
    prose_fold0: torch.Tensor,
    prose_fold1: torch.Tensor,
    spec_fold0: torch.Tensor,
    spec_fold1: torch.Tensor,
    *,
    omega_spec: float = 6.0,
    ridge_scale: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit uncentered direction ``u`` with ``E_E0[u^T x] ~= 1`` and ``E_spec[(u^T x)^2] ~= 0``.

    Uses 2-fold out-of-fold validation on the probe split to compute ``oof_snr`` and a smooth
    reliability gate ``rel_weight in [0, 1]`` so non-separable layers are left untouched.
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
    eye_d = torch.eye(d, dtype=torch.float64)

    def _solve(p_x: torch.Tensor, s_x: torch.Tensor) -> torch.Tensor:
        m0, mu0 = second_moment_and_mean(p_x)
        ms, _ = second_moment_and_mean(s_x)
        comb = m0 + omega_f * ms
        tr = max(float(torch.diagonal(comb).mean().item()), 1e-6)
        lam = max(ridge_f * tr, 1e-5)
        u_raw = torch.linalg.solve(comb + lam * eye_d, mu0)
        denom = float((p_x @ u_raw).mean().item())
        if denom <= 1e-6:
            return torch.zeros(d, dtype=torch.float64)
        return u_raw / denom

    u0 = _solve(p0, s0)
    u1 = _solve(p1, s1)

    p_oof = torch.cat([p1 @ u0, p0 @ u1], dim=0)
    s_oof = torch.cat([s1 @ u0, s0 @ u1], dim=0)

    p_mean = float(p_oof.mean().item())
    s_mean = float(s_oof.mean().item())
    s_rms = float(s_oof.pow(2).mean().sqrt().item())
    pooled_std = math.sqrt(
        0.5 * (float(p_oof.var(unbiased=False).item()) + float(s_oof.var(unbiased=False).item())) + 1e-8
    )
    oof_snr = max(0.0, (p_mean - max(0.0, s_mean)) / pooled_std)
    if p_mean > 0.25 and p_mean > 0.75 * s_rms:
        rel_weight = max(0.0, min(1.0, (oof_snr - 0.20) / 0.80))
    else:
        rel_weight = max(0.0, min(1.0, (oof_snr - 0.25) / 0.85))

    u_full = _solve(torch.cat([p0, p1], dim=0), torch.cat([s0, s1], dim=0))
    return u_full, {
        "oof_prose_mean": p_mean,
        "oof_spec_mean": s_mean,
        "oof_spec_rms": s_rms,
        "oof_snr": oof_snr,
        "rel_weight": rel_weight,
    }


@torch.no_grad()
def zero_expert_lora_b_inplace(moe_mlp: nn.Module, expert_indices: Sequence[int]) -> int:
    """Set ``w1_B = w2_B = w3_B = 0`` for ``expert_indices`` on a ``CrowdLoRAMoEMLP`` (bitwise ``dy == 0``)."""
    if not isinstance(moe_mlp, nn.Module):
        raise TypeError(f"moe_mlp must be an nn.Module, got {type(moe_mlp).__name__}")
    experts_dict = getattr(moe_mlp, "experts", None)
    if experts_dict is None or not hasattr(experts_dict, "__contains__"):
        raise ValueError("moe_mlp must expose an 'experts' mapping")

    unique_indices = sorted({_validate_nonneg_int(idx, "expert_index") for idx in expert_indices})
    zeroed = 0
    for e_idx in unique_indices:
        key = str(e_idx)
        if key not in experts_dict:
            continue
        exp = experts_dict[key]
        for attr in ("w1_B", "w2_B", "w3_B"):
            param = getattr(exp, attr, None)
            if isinstance(param, torch.Tensor):
                param.zero_()
                zeroed += param.numel()
    return zeroed


@torch.no_grad()
def compute_swiglu_intermediate_states(
    moe_mlp: nn.Module,
    x_2d: torch.Tensor,
    active_expert_indices: Sequence[int],
) -> torch.Tensor:
    """Compute post-LoRA SwiGLU activations ``silu(h1 + dh1) * (h2 + dh2)`` pooled across active experts."""
    if not isinstance(moe_mlp, nn.Module):
        raise TypeError(f"moe_mlp must be an nn.Module, got {type(moe_mlp).__name__}")
    _ = _validate_finite_2d(x_2d, "x_2d")
    hidden_size = int(getattr(moe_mlp, "hidden_size", x_2d.shape[1]))
    if int(x_2d.shape[1]) != hidden_size:
        raise ValueError(f"Expected x_2d with feature width {hidden_size}, got {x_2d.shape[1]}")

    w_dt = moe_mlp.base_mlp.w1.weight.dtype
    x_base = x_2d.to(dtype=w_dt)
    h1 = moe_mlp.base_mlp.w1(x_base).to(torch.float32)
    h2 = moe_mlp.base_mlp.w2(x_base).to(torch.float32)
    x_fp32 = x_2d.to(torch.float32)
    act_base = F.silu(h1) * h2

    if not active_expert_indices:
        return act_base

    logits = moe_mlp.router(x_fp32)
    if getattr(moe_mlp, "routing_mode", "sigma_topk") == "softmax_topk":
        probs = F.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)
    top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)

    act_accum = act_base.clone()
    weight_sum = torch.ones((x_fp32.shape[0], 1), dtype=torch.float32, device=x_fp32.device)
    for e_idx in active_expert_indices:
        e_int = _validate_nonneg_int(e_idx, "active_expert_index")
        key = str(e_int)
        if key not in moe_mlp.experts:
            continue
        exp = moe_mlp.experts[key]
        hit = (top_idx == e_int).to(torch.float32)
        w_tok = (top_vals * hit).sum(dim=-1, keepdim=True)
        if not bool((w_tok > 0).any().item()):
            continue
        dh1 = exp.scaling * ((x_fp32 @ exp.w1_A.T) @ exp.w1_B.T)
        dh2 = exp.scaling * ((x_fp32 @ exp.w2_A.T) @ exp.w2_B.T)
        act_e = F.silu(h1 + dh1) * (h2 + dh2)
        act_accum = act_accum + w_tok * act_e
        weight_sum = weight_sum + w_tok
    return act_accum / weight_sum


@torch.no_grad()
def apply_layer_wsp_edit_inplace(
    moe_mlp: nn.Module,
    prose_fold0: torch.Tensor,
    prose_fold1: torch.Tensor,
    spec_fold0: torch.Tensor,
    spec_fold1: torch.Tensor,
    cfg: WSPArmConfig,
    *,
    sink_experts: Sequence[int] = (),
) -> dict[str, Any]:
    """Edit a single ``CrowdLoRAMoEMLP`` layer in-place using probe activations at that exact layer."""
    layer_meta: dict[str, Any] = {}
    sink_set = {_validate_nonneg_int(e, "sink_expert") for e in sink_experts} if cfg.use_ghost_sink else set()
    if cfg.force_zero_expert0 and cfg.use_ghost_sink:
        sink_set.add(0)

    if sink_set:
        zero_expert_lora_b_inplace(moe_mlp, sorted(sink_set))
    layer_meta["sink_experts"] = sorted(sink_set)

    all_expert_ids = sorted(int(k) for k in moe_mlp.experts.keys())
    surviving_experts = [e for e in all_expert_ids if e not in sink_set]
    raw_sink = {_validate_nonneg_int(e, "sink_expert") for e in sink_experts}
    expert0_helps_e0 = (0 in surviving_experts) and (0 not in raw_sink) and (not cfg.force_zero_expert0)
    off_domain_experts = [e for e in surviving_experts if (e != 0 or not expert0_helps_e0)]

    if cfg.router_beta_suppress > 0.0 or cfg.router_beta_sink > 0.0:
        u_prose, probe_stats = fit_uncentered_prose_probe_with_oof(
            prose_fold0,
            prose_fold1,
            spec_fold0,
            spec_fold1,
            omega_spec=cfg.router_omega_spec,
            ridge_scale=cfg.router_ridge_scale,
        )
        rel = float(probe_stats["rel_weight"])
        layer_meta["router_probe"] = probe_stats
        if rel > 0.0 and float(u_prose.norm().item()) > 0.0:
            u_target = u_prose.to(device=moe_mlp.router.weight.device, dtype=moe_mlp.router.weight.dtype)
            for e_idx in off_domain_experts:
                if cfg.router_beta_suppress > 0.0:
                    moe_mlp.router.weight[e_idx].sub_(float(cfg.router_beta_suppress * rel) * u_target)
            if cfg.router_beta_sink > 0.0 and (0 in all_expert_ids or sink_set):
                primary_target = 0 if 0 in all_expert_ids else min(sink_set)
                moe_mlp.router.weight[primary_target].add_(float(cfg.router_beta_sink * rel) * u_target)

    prose_all = torch.cat([prose_fold0, prose_fold1], dim=0)
    spec_all = torch.cat([spec_fold0, spec_fold1], dim=0)

    if cfg.alphaedit_eta_x > 0.0 and off_domain_experts:
        m_p_x, _ = second_moment_and_mean(prose_all)
        m_s_x, _ = second_moment_and_mean(spec_all)
        g_x, gx_stats = compute_discriminative_alphaedit_filter(
            m_p_x,
            m_s_x,
            eta=cfg.alphaedit_eta_x,
            tau=cfg.alphaedit_tau,
            max_rank=cfg.alphaedit_max_rank_x,
            reg_spec=cfg.alphaedit_reg_spec,
        )
        layer_meta["alphaedit_x"] = gx_stats
        if gx_stats["active_modes"] > 0:
            gx_f32 = g_x.to(dtype=torch.float32)
            for e_idx in off_domain_experts:
                exp = moe_mlp.experts[str(e_idx)]
                exp.w1_A.copy_((exp.w1_A.to(torch.float32) @ gx_f32.to(exp.w1_A.device)).to(exp.w1_A.dtype))
                exp.w2_A.copy_((exp.w2_A.to(torch.float32) @ gx_f32.to(exp.w2_A.device)).to(exp.w2_A.dtype))

    if cfg.alphaedit_eta_act > 0.0 and off_domain_experts:
        act_prose = compute_swiglu_intermediate_states(moe_mlp, prose_all, off_domain_experts)
        act_spec = compute_swiglu_intermediate_states(moe_mlp, spec_all, off_domain_experts)
        m_p_a, _ = second_moment_and_mean(act_prose)
        m_s_a, _ = second_moment_and_mean(act_spec)
        g_a, ga_stats = compute_discriminative_alphaedit_filter(
            m_p_a,
            m_s_a,
            eta=cfg.alphaedit_eta_act,
            tau=cfg.alphaedit_tau,
            max_rank=cfg.alphaedit_max_rank_act,
            reg_spec=cfg.alphaedit_reg_spec,
        )
        layer_meta["alphaedit_act"] = ga_stats
        if ga_stats["active_modes"] > 0:
            ga_f32 = g_a.to(dtype=torch.float32)
            for e_idx in off_domain_experts:
                exp = moe_mlp.experts[str(e_idx)]
                exp.w3_A.copy_((exp.w3_A.to(torch.float32) @ ga_f32.to(exp.w3_A.device)).to(exp.w3_A.dtype))

    return layer_meta


# ---------------------------------------------------------------------------
# 2. Sequential Model Propagation, Final LN Guard Calibration & E2E Synthesis
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_embeddings_batch(model: nn.Module, ids_2d: torch.Tensor) -> torch.Tensor:
    """Compute initial ``wte + engram`` float32 sequence states for ``CrowdV1WithEngram``."""
    x = model.wte(ids_2d).to(torch.float32)
    if getattr(model.cfg, "engram_buckets", 0) > 0:
        x = x + model.engram(ids_2d).to(device=x.device, dtype=torch.float32)
    return x


@torch.no_grad()
def forward_block_pre_mlp(
    blk: nn.Module,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``x = x + attn(ln1(x))`` and return ``(x_post_attn, ln2(x_post_attn))`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    x_in = x.to(dtype=blk_dtype)
    h, _ = blk.attn(blk.ln1(x_in), cos, sin, past_kv=None, use_cache=False)
    x_mid = x_in + h
    mlp_in = blk.ln2(x_mid)
    return x_mid.to(torch.float32), mlp_in.to(torch.float32)


@torch.no_grad()
def forward_block_post_mlp(blk: nn.Module, x_mid: torch.Tensor, mlp_in: torch.Tensor) -> torch.Tensor:
    """Finish transformer block ``x_out = x_mid + mlp(mlp_in)`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    mlp_out = blk.mlp(mlp_in.to(dtype=blk_dtype)).to(torch.float32)
    return (x_mid + mlp_out).to(torch.float32)


@torch.no_grad()
def forward_model_pre_ln(model: nn.Module, ids_2d: torch.Tensor, *, batch_size: int = 24) -> torch.Tensor:
    """Run all transformer blocks and return pre-final-LayerNorm hidden states in ``float32``."""
    if ids_2d.ndim != 2 or ids_2d.shape[0] == 0:
        raise ValueError(f"ids_2d must be a non-empty 2D tensor, got shape {tuple(ids_2d.shape)}")
    bsz = max(1, _validate_nonneg_int(batch_size, "batch_size"))
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    parts: list[torch.Tensor] = []
    for b0 in range(0, ids_2d.shape[0], bsz):
        xb = compute_embeddings_batch(model, ids_2d[b0 : b0 + bsz])
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        parts.append(xb)
    return torch.cat(parts, dim=0)


@torch.no_grad()
def per_chunk_nll_from_pre_ln(
    h_pre_ln: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    head_weight: torch.Tensor,
    ids_2d: torch.Tensor,
    *,
    score_from: int = 64,
) -> torch.Tensor:
    """Compute exact per-chunk causal NLL over ``ids_2d[:, max(1, score_from):]`` from pre-LN states."""
    if h_pre_ln.ndim != 3 or ids_2d.ndim != 2:
        raise ValueError(
            f"Expected 3D h_pre_ln and 2D ids_2d, got {tuple(h_pre_ln.shape)} and {tuple(ids_2d.shape)}"
        )
    n_chunks, seq_len, hidden_dim = h_pre_ln.shape
    if n_chunks == 0 or ids_2d.shape != (n_chunks, seq_len):
        raise ValueError(f"Mismatched h_pre_ln {tuple(h_pre_ln.shape)} and ids_2d {tuple(ids_2d.shape)}")
    if ln_weight.shape != (hidden_dim,) or ln_bias.shape != (hidden_dim,):
        raise ValueError(f"Expected 1D LN params of length {hidden_dim}")
    if head_weight.ndim != 2 or head_weight.shape[1] != hidden_dim:
        raise ValueError(f"Expected 2D head_weight with width {hidden_dim}, got {tuple(head_weight.shape)}")
    eps_f = _validate_nonneg_finite(ln_eps, "ln_eps")
    if eps_f <= 0.0:
        raise ValueError(f"ln_eps must be > 0, got {ln_eps}")
    offset = _validate_nonneg_int(score_from, "score_from")
    start_pos = max(0, offset - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} too short for start_pos={start_pos}")

    h_norm = F.layer_norm(
        h_pre_ln.to(torch.float32),
        (hidden_dim,),
        ln_weight.to(torch.float32),
        ln_bias.to(torch.float32),
        eps=eps_f,
    )
    pred_h = h_norm[:, start_pos:-1, :]
    targets = ids_2d[:, start_pos + 1 :]
    logits = F.linear(pred_h, head_weight.to(torch.float32))
    vocab = logits.shape[-1]
    tok_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    return tok_loss.view(n_chunks, -1).mean(dim=1)


@torch.no_grad()
def select_guard_optimal_lambda_ln(
    h_pre_ln: torch.Tensor,
    start_ln_w: torch.Tensor,
    start_ln_b: torch.Tensor,
    c2_ln_w: torch.Tensor,
    c2_ln_b: torch.Tensor,
    ln_eps: float,
    head_weight: torch.Tensor,
    ids_all: torch.Tensor,
    base_losses: np.ndarray | torch.Tensor | Sequence[float],
    is_e0: np.ndarray | torch.Tensor | Sequence[bool],
    *,
    score_from: int = 64,
    target_e0_max_ratio: float = LN_GUARD_MAX_RATIO,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_LN_GRID,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select ``lambda_ln`` minimizing overall NLL subject to ``E0 <= +0.5%`` vs dense base.

    Falls back to the candidate minimizing general-domain (``E0``) drift ratio when no grid point
    satisfies ``target_e0_max_ratio``.
    """
    n_chunks = int(ids_all.shape[0])
    if isinstance(base_losses, torch.Tensor):
        base_arr = base_losses.detach().cpu().numpy().astype(np.float64)
    else:
        base_arr = np.asarray(base_losses, dtype=np.float64)
    if isinstance(is_e0, torch.Tensor):
        e0_mask = is_e0.detach().cpu().numpy().astype(bool)
    else:
        e0_mask = np.asarray(is_e0, dtype=bool)

    if base_arr.ndim != 1 or base_arr.shape[0] != n_chunks or not bool(np.isfinite(base_arr).all()):
        raise ValueError(f"base_losses must be finite 1D array of length {n_chunks}")
    if e0_mask.ndim != 1 or e0_mask.shape[0] != n_chunks or not bool(np.any(e0_mask)):
        raise ValueError(f"is_e0 must be 1D boolean mask of length {n_chunks} with at least one True")
    if start_ln_w.shape != c2_ln_w.shape or start_ln_b.shape != c2_ln_b.shape:
        raise ValueError("Start and tuned LayerNorm parameter shapes must match")
    ratio_cap = _validate_nonneg_finite(target_e0_max_ratio, "target_e0_max_ratio")
    if ratio_cap <= 0.0:
        raise ValueError("target_e0_max_ratio must be > 0.0")
    if not lambda_grid:
        raise ValueError("lambda_grid must be non-empty")

    base_e0 = float(base_arr[e0_mask].mean())
    base_all = float(base_arr.mean())
    has_spec = bool(np.any(~e0_mask))

    grid: dict[str, dict[str, float]] = {}
    feasible: list[tuple[float, float]] = []
    fallback: list[tuple[float, float, float]] = []

    for raw_l in lambda_grid:
        cand_l = float(raw_l)
        if not math.isfinite(cand_l):
            raise ValueError(f"lambda_grid entries must be finite, got {raw_l!r}")
        wt = start_ln_w + cand_l * (c2_ln_w - start_ln_w)
        bt = start_ln_b + cand_l * (c2_ln_b - start_ln_b)
        ls = (
            per_chunk_nll_from_pre_ln(h_pre_ln, wt, bt, ln_eps, head_weight, ids_all, score_from=score_from)
            .cpu()
            .numpy()
        )
        e0_m = float(ls[e0_mask].mean())
        sp_m = float(ls[~e0_mask].mean()) if has_spec else 0.0
        nll_mean = float(ls.mean())
        mean_diff = nll_mean - base_all
        e0_ratio = e0_m / max(base_e0, 1e-8)

        grid[f"{cand_l:.2f}"] = {
            "nll_mean": nll_mean,
            "e0_nll": e0_m,
            "spec_nll": sp_m,
            "e0_ratio": e0_ratio,
            "e0_pct_vs_base": 100.0 * (e0_ratio - 1.0),
            "mean_diff_vs_base": mean_diff,
        }
        if e0_ratio <= ratio_cap:
            feasible.append((mean_diff, cand_l))
        fallback.append((e0_ratio, mean_diff, cand_l))

    if feasible:
        feasible.sort(key=lambda t: (t[0], t[1]))
        return float(feasible[0][1]), grid
    fallback.sort(key=lambda t: (t[0], t[1], t[2]))
    return float(fallback[0][2]), grid


def _extract_ln_params_from_buckets(buckets: Mapping[str, torch.Tensor], h_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice final ``(ln.weight, ln.bias)`` from the tail of the final ``router_frag_*`` bucket."""
    router_keys = sorted(k for k in buckets if k.startswith("router_frag_"))
    if not router_keys:
        raise ValueError("Buckets mapping must contain at least one 'router_frag_*' entry")
    last_vec = buckets[router_keys[-1]]
    if last_vec.numel() < 2 * h_dim:
        raise ValueError(f"Final router fragment too short ({last_vec.numel()}) for h_dim={h_dim}")
    ln_w = last_vec[-2 * h_dim : -h_dim].detach().to("cpu", dtype=torch.float32).clone()
    ln_b = last_vec[-h_dim:].detach().to("cpu", dtype=torch.float32).clone()
    return ln_w, ln_b


def _resolve_bucket_io(
    buckets: Mapping[str, torch.Tensor],
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None,
) -> tuple[Callable[[nn.Module, Mapping[str, torch.Tensor]], None], Callable[[nn.Module], dict[str, torch.Tensor]]]:
    """Build default OpenDiLoCo bucket loader/extractor bound to the bucket count when omitted."""
    nfrag = max(1, len([k for k in buckets if k.startswith("router_frag_")]))
    loader = load_buckets_fn or (lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag))
    extractor = extract_buckets_fn or (lambda m: extract_moe_trainable_buckets(m, num_router_fragments=nfrag))
    return loader, extractor


def _probe_ids_and_mask(probe_chunks: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, np.ndarray]:
    """Validate probe chunk sequence and extract ``(ids_2d, is_general_mask)``."""
    if not probe_chunks:
        raise ValueError("probe_chunks must be non-empty")
    ids_all = torch.stack([torch.as_tensor(c["ids"], dtype=torch.long) for c in probe_chunks], dim=0)
    is_e0 = np.array([str(c["expert_id"]).startswith(GENERAL_EXPERT_PREFIX) for c in probe_chunks], dtype=bool)
    return ids_all, is_e0


@torch.no_grad()
def calibrate_final_layernorm_and_sink_experts(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    c2_buckets: Mapping[str, torch.Tensor],
    probe_chunks: Sequence[Mapping[str, Any]],
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    *,
    score_from: int = 64,
    loeo_chunks_per_domain: int = 8,
    batch_size: int = 24,
    loeo_e0_harm_threshold: float = -0.0015,
    loeo_max_spec_loss_increase: float = 0.03,
) -> dict[str, Any]:
    """Compute dense base probe NLLs, initial ``lambda_ln``, and E0-harmful sink experts on Probe split."""
    ids_all, is_e0 = _probe_ids_and_mask(probe_chunks)
    loader, _ = _resolve_bucket_io(start_buckets, load_buckets_fn, None)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    h_dim = int(model.ln.weight.numel())

    start_ln_w, start_ln_b = _extract_ln_params_from_buckets(start_buckets, h_dim)
    c2_ln_w, c2_ln_b = _extract_ln_params_from_buckets(c2_buckets, h_dim)

    loader(model, start_buckets)
    model.eval()
    h_base = forward_model_pre_ln(model, ids_all, batch_size=batch_size)
    base_losses = (
        per_chunk_nll_from_pre_ln(h_base, start_ln_w, start_ln_b, ln_eps, model.head.weight, ids_all, score_from=score_from)
        .cpu()
        .numpy()
    )

    loader(model, c2_buckets)
    model.eval()
    h_pre_ln = forward_model_pre_ln(model, ids_all, batch_size=batch_size)
    best_lam_ln, ln_grid = select_guard_optimal_lambda_ln(
        h_pre_ln,
        start_ln_w,
        start_ln_b,
        c2_ln_w,
        c2_ln_b,
        ln_eps,
        model.head.weight,
        ids_all,
        base_losses,
        is_e0,
        score_from=score_from,
    )

    sub_indices: list[int] = []
    seen_per_dom: dict[str, int] = {}
    cap_per_dom = max(1, _validate_nonneg_int(loeo_chunks_per_domain, "loeo_chunks_per_domain"))
    for idx, ch in enumerate(probe_chunks):
        dom = str(ch["expert_id"])
        cap = cap_per_dom * 2 if dom.startswith(GENERAL_EXPERT_PREFIX) else cap_per_dom
        if seen_per_dom.get(dom, 0) < cap:
            sub_indices.append(idx)
            seen_per_dom[dom] = seen_per_dom.get(dom, 0) + 1

    sub_ids = ids_all[sub_indices]
    sub_is_e0 = is_e0[sub_indices]
    sub_is_spec = ~sub_is_e0
    has_sub_spec = bool(np.any(sub_is_spec))
    cal_ln_w = start_ln_w + best_lam_ln * (c2_ln_w - start_ln_w)
    cal_ln_b = start_ln_b + best_lam_ln * (c2_ln_b - start_ln_b)

    def _eval_subset() -> tuple[float, float]:
        hp = forward_model_pre_ln(model, sub_ids, batch_size=batch_size)
        nlls = (
            per_chunk_nll_from_pre_ln(hp, cal_ln_w, cal_ln_b, ln_eps, model.head.weight, sub_ids, score_from=score_from)
            .cpu()
            .numpy()
        )
        return float(nlls[sub_is_e0].mean()), (float(nlls[sub_is_spec].mean()) if has_sub_spec else 0.0)

    loader(model, c2_buckets)
    ref_e0, ref_spec = _eval_subset()
    expert_ids = sorted(int(k.split("_", 1)[1]) for k in c2_buckets if k.startswith("expert_"))
    expert_loeo: dict[str, dict[str, float]] = {}
    harmful_sink_experts: list[int] = []

    for e_idx in expert_ids:
        loader(model, {k: v.clone() for k, v in c2_buckets.items()})
        for blk in model.blocks:
            zero_expert_lora_b_inplace(blk.mlp, [e_idx])
        e0_m, sp_m = _eval_subset()
        d_e0 = e0_m - ref_e0
        d_sp = sp_m - ref_spec
        expert_loeo[f"expert_{e_idx}"] = {
            "e0_nll_without": e0_m,
            "spec_nll_without": sp_m,
            "delta_e0_when_zeroed": d_e0,
            "delta_spec_when_zeroed": d_sp,
        }
        if d_e0 < float(loeo_e0_harm_threshold) and d_sp <= float(loeo_max_spec_loss_increase):
            harmful_sink_experts.append(e_idx)

    loader(model, c2_buckets)
    return {
        "best_lambda_ln": best_lam_ln,
        "base_probe_losses": [float(x) for x in base_losses.tolist()],
        "ln_grid": ln_grid,
        "expert_loeo": expert_loeo,
        "harmful_sink_experts": harmful_sink_experts,
    }


@torch.no_grad()
def synthesize_wsp_arm_buckets(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    c2_buckets: Mapping[str, torch.Tensor],
    probe_chunks: Sequence[Mapping[str, Any]],
    cfg: WSPArmConfig,
    calibration: Mapping[str, Any],
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
    *,
    score_from: int = 64,
    batch_size: int = 24,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Synthesize WSP-AlphaEdit candidate buckets sequentially across all transformer layers."""
    ids_all, is_e0 = _probe_ids_and_mask(probe_chunks)
    loader, extractor = _resolve_bucket_io(start_buckets, load_buckets_fn, extract_buckets_fn)
    loader(model, {k: v.clone() for k, v in c2_buckets.items()})
    model.eval()

    lam_ln = float(calibration["best_lambda_ln"]) if cfg.use_ln_calibration else 1.0
    h_dim = int(model.ln.weight.numel())
    start_ln_w, start_ln_b = _extract_ln_params_from_buckets(start_buckets, h_dim)
    c2_ln_w, c2_ln_b = _extract_ln_params_from_buckets(c2_buckets, h_dim)
    model.ln.weight.copy_((start_ln_w + lam_ln * (c2_ln_w - start_ln_w)).to(model.ln.weight.dtype))
    model.ln.bias.copy_((start_ln_b + lam_ln * (c2_ln_b - start_ln_b)).to(model.ln.bias.dtype))

    sink_experts = [int(e) for e in calibration.get("harmful_sink_experts", ())]
    sink_applied = (
        sorted(set(sink_experts) | ({0} if cfg.force_zero_expert0 else set())) if cfg.use_ghost_sink else []
    )
    needs_sequential_activations = (
        cfg.router_beta_suppress > 0.0
        or cfg.router_beta_sink > 0.0
        or cfg.alphaedit_eta_x > 0.0
        or cfg.alphaedit_eta_act > 0.0
    )

    arm_meta: dict[str, Any] = {
        "config": asdict(cfg),
        "lambda_ln_applied": lam_ln,
        "sink_experts_applied": sink_applied,
        "layers": [],
    }
    bsz = max(1, _validate_nonneg_int(batch_size, "batch_size"))
    ln_eps = float(getattr(model.ln, "eps", 1e-5))

    if not needs_sequential_activations:
        if sink_applied:
            for blk in model.blocks:
                zero_expert_lora_b_inplace(blk.mlp, sink_applied)
            if cfg.use_ln_calibration and "base_probe_losses" in calibration:
                hp_post = forward_model_pre_ln(model, ids_all, batch_size=bsz)
                best_l, _ = select_guard_optimal_lambda_ln(
                    hp_post,
                    start_ln_w,
                    start_ln_b,
                    c2_ln_w,
                    c2_ln_b,
                    ln_eps,
                    model.head.weight,
                    ids_all,
                    np.asarray(calibration["base_probe_losses"], dtype=np.float64),
                    is_e0,
                    score_from=score_from,
                )
                model.ln.weight.copy_((start_ln_w + best_l * (c2_ln_w - start_ln_w)).to(model.ln.weight.dtype))
                model.ln.bias.copy_((start_ln_b + best_l * (c2_ln_b - start_ln_b)).to(model.ln.bias.dtype))
                arm_meta["lambda_ln_applied"] = best_l
        return extractor(model), arm_meta

    prose_f0_idx: list[int] = []
    prose_f1_idx: list[int] = []
    spec_f0_idx: list[int] = []
    spec_f1_idx: list[int] = []
    dom_counters: dict[str, int] = {}
    for idx, ch in enumerate(probe_chunks):
        dom = str(ch["expert_id"])
        cnt = dom_counters.get(dom, 0)
        dom_counters[dom] = cnt + 1
        if dom.startswith(GENERAL_EXPERT_PREFIX):
            (prose_f0_idx if cnt % 2 == 0 else prose_f1_idx).append(idx)
        else:
            (spec_f0_idx if cnt % 2 == 0 else spec_f1_idx).append(idx)

    if not prose_f0_idx or not prose_f1_idx or not spec_f0_idx or not spec_f1_idx:
        raise ValueError("probe_chunks must contain at least 2 general (E0_*) and 2 specialist chunks for 2-fold OOF")

    start_pos = max(0, _validate_nonneg_int(score_from, "score_from") - 1)
    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)

    x_seq = torch.cat([compute_embeddings_batch(model, ids_all[b0 : b0 + bsz]) for b0 in range(0, ids_all.shape[0], bsz)], dim=0)

    for l_idx, blk in enumerate(model.blocks):
        x_mid_parts: list[torch.Tensor] = []
        mlp_in_parts: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], bsz):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + bsz], cos, sin)
            x_mid_parts.append(xm)
            mlp_in_parts.append(mi)
        x_mid = torch.cat(x_mid_parts, dim=0)
        mlp_in = torch.cat(mlp_in_parts, dim=0)

        scored = mlp_in[:, start_pos:-1, :]
        h_dim_blk = scored.shape[-1]
        p0 = scored[prose_f0_idx].reshape(-1, h_dim_blk)
        p1 = scored[prose_f1_idx].reshape(-1, h_dim_blk)
        s0 = scored[spec_f0_idx].reshape(-1, h_dim_blk)
        s1 = scored[spec_f1_idx].reshape(-1, h_dim_blk)

        l_meta = apply_layer_wsp_edit_inplace(
            blk.mlp,
            p0,
            p1,
            s0,
            s1,
            cfg,
            sink_experts=sink_experts,
        )
        l_meta["layer"] = l_idx
        arm_meta["layers"].append(l_meta)

        x_seq = torch.cat(
            [forward_block_post_mlp(blk, x_mid[b0 : b0 + bsz], mlp_in[b0 : b0 + bsz]) for b0 in range(0, x_mid.shape[0], bsz)],
            dim=0,
        )

    if cfg.use_ln_calibration and "base_probe_losses" in calibration:
        best_l, _ = select_guard_optimal_lambda_ln(
            x_seq,
            start_ln_w,
            start_ln_b,
            c2_ln_w,
            c2_ln_b,
            ln_eps,
            model.head.weight,
            ids_all,
            np.asarray(calibration["base_probe_losses"], dtype=np.float64),
            is_e0,
            score_from=score_from,
        )
        model.ln.weight.copy_((start_ln_w + best_l * (c2_ln_w - start_ln_w)).to(model.ln.weight.dtype))
        model.ln.bias.copy_((start_ln_b + best_l * (c2_ln_b - start_ln_b)).to(model.ln.bias.dtype))
        arm_meta["lambda_ln_applied"] = best_l

    return extractor(model), arm_meta


def interpolate_weight_buckets(
    base_buckets: Mapping[str, torch.Tensor],
    tuned_buckets: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Compute uniform WiSE-FT linear interpolation ``(1 - alpha) * base + alpha * tuned``."""
    alpha_f = _validate_nonneg_finite(alpha, "alpha")
    if alpha_f > 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if set(base_buckets.keys()) != set(tuned_buckets.keys()):
        raise ValueError("base_buckets and tuned_buckets must contain identical keys")
    out: dict[str, torch.Tensor] = {}
    for key in sorted(base_buckets.keys()):
        b_t = base_buckets[key].detach().to(torch.float32)
        t_t = tuned_buckets[key].detach().to(torch.float32)
        if b_t.shape != t_t.shape:
            raise ValueError(f"Shape mismatch on bucket {key!r}: {tuple(b_t.shape)} vs {tuple(t_t.shape)}")
        out[key] = (b_t + alpha_f * (t_t - b_t)).contiguous()
    return out


def evaluate_wsp_guard_and_retention(
    base_losses: np.ndarray | torch.Tensor | Sequence[float],
    cand_losses: np.ndarray | torch.Tensor | Sequence[float],
    is_e0: np.ndarray | torch.Tensor | Sequence[bool],
    *,
    tuned_losses: np.ndarray | torch.Tensor | Sequence[float] | None = None,
    domain_guard_max_ratio: float = DOMAIN_GUARD_MAX_RATIO,
    anse_safe_e0_ratio: float = ANSE_SAFE_E0_RATIO,
) -> dict[str, Any]:
    """Summarize general-domain guard clearance, specialist NLL gain retention, and ANSE energy."""
    base_arr = np.asarray(base_losses, dtype=np.float64)
    cand_arr = np.asarray(cand_losses, dtype=np.float64)
    e0_mask = np.asarray(is_e0, dtype=bool)
    if base_arr.ndim != 1 or cand_arr.shape != base_arr.shape or e0_mask.shape != base_arr.shape:
        raise ValueError("base_losses, cand_losses, and is_e0 must share a 1D shape")
    if not bool(np.any(e0_mask)) or not bool(np.any(~e0_mask)):
        raise ValueError("is_e0 must contain both general-domain (True) and specialist (False) chunks")

    base_e0 = float(base_arr[e0_mask].mean())
    cand_e0 = float(cand_arr[e0_mask].mean())
    base_spec = float(base_arr[~e0_mask].mean())
    cand_spec = float(cand_arr[~e0_mask].mean())
    nll_mean = float(cand_arr.mean())
    mean_diff = nll_mean - float(base_arr.mean())

    e0_ratio = cand_e0 / max(base_e0, 1e-8)
    guard_ok = bool(e0_ratio <= float(domain_guard_max_ratio))
    spec_gain_nats = base_spec - cand_spec

    spec_gain_retained_frac: float | None = None
    if tuned_losses is not None:
        tuned_arr = np.asarray(tuned_losses, dtype=np.float64)
        if tuned_arr.shape != base_arr.shape:
            raise ValueError("tuned_losses must match base_losses shape")
        tuned_spec = float(tuned_arr[~e0_mask].mean())
        raw_gain = base_spec - tuned_spec
        spec_gain_retained_frac = spec_gain_nats / raw_gain if abs(raw_gain) > 1e-8 else 0.0

    anse_energy = mean_diff + (1e6 if not guard_ok else (0.05 if e0_ratio > float(anse_safe_e0_ratio) else 0.0))
    return {
        "nll_mean": nll_mean,
        "base_nll_mean": float(base_arr.mean()),
        "mean_diff_vs_base": mean_diff,
        "e0_nll_mean": cand_e0,
        "base_e0_nll_mean": base_e0,
        "e0_ratio": e0_ratio,
        "e0_pct_vs_base": 100.0 * (e0_ratio - 1.0),
        "spec_nll_mean": cand_spec,
        "base_spec_nll_mean": base_spec,
        "spec_gain_nats": spec_gain_nats,
        "spec_gain_retained_frac": spec_gain_retained_frac,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
    }


__all__ = [
    "ANSE_SAFE_E0_RATIO",
    "DEFAULT_LAMBDA_LN_GRID",
    "DOMAIN_GUARD_MAX_RATIO",
    "GENERAL_EXPERT_PREFIX",
    "LN_GUARD_MAX_RATIO",
    "WSPArmConfig",
    "apply_layer_wsp_edit_inplace",
    "calibrate_final_layernorm_and_sink_experts",
    "compute_discriminative_alphaedit_filter",
    "compute_embeddings_batch",
    "compute_swiglu_intermediate_states",
    "evaluate_wsp_guard_and_retention",
    "fit_uncentered_prose_probe_with_oof",
    "forward_block_post_mlp",
    "forward_block_pre_mlp",
    "forward_model_pre_ln",
    "interpolate_weight_buckets",
    "per_chunk_nll_from_pre_ln",
    "second_moment_and_mean",
    "select_guard_optimal_lambda_ln",
    "synthesize_wsp_arm_buckets",
    "zero_expert_lora_b_inplace",
]
