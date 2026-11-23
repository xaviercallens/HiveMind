# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""QR-SVD Rank-Selective LoRA Singular-Mode Surgery (LASER) for Sparse MoE Control.

Implements memory-efficient exact rank-``r`` singular-value decomposition and
selective Rayleigh-ratio mode surgery for LoRA factor pairs ``(B in R^{d_out x r},
A in R^{r x d_in})`` without materializing full ``(d_out x d_in)`` dense matrices:

1. ``decompose_lora_qr_svd(w_b, w_a) -> (U_r, S_r, V_r)``:
   reduced QR factorizations ``B = Q_B R_B``, ``A^T = Q_A R_A`` followed by thin
   SVD on the tiny ``(r x r)`` core ``R_B R_A^T``.
2. ``score_singular_modes_rayleigh_ratio(v_r, m_prose, m_spec, *, reg_spec=0.10)``:
   exact float64 prose-to-specialist generalized Rayleigh variance ratios per
   right singular mode ``k in 0..r-1``.
3. ``apply_lora_svd_mode_surgery_inplace(...)``:
   selectively damps or truncates only the ``1..max_modes_damped`` interfering
   singular modes (``mu_k > tau_svd``) via right-space orthogonal projection of
   ``A``, leaving ``B`` and all orthogonal specialist singular modes bitwise
   preserved up to floating-point precision and reporting exact parameter-fraction
   modified.
4. ``apply_swiglu_expert_svd_surgery_inplace(...)`` and
   ``apply_moe_layer_svd_surgery_inplace(...)``:
   SwiGLU ``(w1, w2, w3)`` and ``CrowdLoRAMoEMLP`` layer orchestration helpers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.wsp_alphaedit_control import (
    compute_swiglu_intermediate_states,
    second_moment_and_mean,
)


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


def _validate_positive_int(value: int, name: str) -> int:
    """Validate that an integer parameter is strictly positive."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


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


def _resolve_lora_factor(exp_mod: nn.Module, attr_name: str) -> torch.Tensor:
    """Extract a LoRA weight tensor from a Parameter/Tensor attribute or nn.Linear."""
    attr = getattr(exp_mod, attr_name, None)
    if isinstance(attr, nn.Linear):
        attr = attr.weight
    if not isinstance(attr, torch.Tensor):
        raise ValueError(f"{type(exp_mod).__name__} must expose tensor or nn.Linear attribute {attr_name!r}")
    return attr


@dataclass(frozen=True)
class SVDSurgeryConfig:
    """Hyperparameters for QR-SVD rank-selective LoRA singular-mode surgery."""

    tau_svd: float = 1.35
    eta_svd: float = 3.0
    max_modes_damped: int = 2
    reg_spec: float = 0.10
    truncate: bool = False
    min_singular_value: float = 1e-10

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.tau_svd, "tau_svd")
        _validate_nonneg_finite(self.eta_svd, "eta_svd")
        _validate_nonneg_int(self.max_modes_damped, "max_modes_damped")
        _validate_nonneg_finite(self.reg_spec, "reg_spec")
        if not isinstance(self.truncate, bool):
            raise TypeError(f"truncate must be a bool, got {type(self.truncate).__name__}")
        _validate_nonneg_finite(self.min_singular_value, "min_singular_value")


@torch.no_grad()
def decompose_lora_qr_svd(
    w_b: torch.Tensor,
    w_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact thin SVD of ``w_b @ w_a`` via reduced QR without forming ``(d_out, d_in)``.

    Args:
        w_b: Left LoRA factor ``B`` of shape ``(d_out, r)``.
        w_a: Right LoRA factor ``A`` of shape ``(r, d_in)``.

    Returns:
        ``(U_r, S_r, V_r)`` in CPU ``float64`` where ``k_eff = min(d_out, d_in, r)``,
        ``U_r`` has shape ``(d_out, k_eff)`` with orthonormal columns, ``S_r`` has shape
        ``(k_eff,)`` in descending order, ``V_r`` has shape ``(d_in, k_eff)`` with
        orthonormal columns, and ``U_r @ diag(S_r) @ V_r.T == w_b @ w_a``.
    """
    b64 = _validate_finite_2d(w_b, "w_b")
    a64 = _validate_finite_2d(w_a, "w_a")
    if b64.shape[1] != a64.shape[0]:
        raise ValueError(
            f"Inner rank mismatch between w_b {tuple(b64.shape)} and w_a {tuple(a64.shape)}"
        )

    qb, rb = torch.linalg.qr(b64, mode="reduced")
    qa, ra = torch.linalg.qr(a64.T, mode="reduced")
    core = rb @ ra.T
    uc, s_r, vct = torch.linalg.svd(core, full_matrices=False)
    u_r = (qb @ uc).contiguous()
    v_r = (qa @ vct.T).contiguous()
    return u_r, s_r.contiguous(), v_r


@torch.no_grad()
def score_singular_modes_rayleigh_ratio(
    v_r: torch.Tensor,
    m_prose: torch.Tensor,
    m_spec: torch.Tensor,
    *,
    reg_spec: float = 0.10,
) -> torch.Tensor:
    """Compute exact float64 prose-to-specialist generalized Rayleigh variance ratios.

    For each right singular direction ``v_k = v_r[:, k]`` (``k in 0..k_eff-1``), evaluates
    ``mu_k = max(0, v_k^T M_prose v_k) / max(1e-8, v_k^T M_spec v_k + lam * ||v_k||_2^2)``
    where ``lam = max(reg_spec * max(mean(diag(M_spec)), 0), 1e-6)``.
    """
    v64 = _validate_finite_2d(v_r, "v_r")
    mp_raw = _validate_finite_2d(m_prose, "m_prose")
    ms_raw = _validate_finite_2d(m_spec, "m_spec")
    reg_f = _validate_nonneg_finite(reg_spec, "reg_spec")

    d_in = int(v64.shape[0])
    if mp_raw.shape != (d_in, d_in) or ms_raw.shape != (d_in, d_in):
        raise ValueError(
            f"m_prose {tuple(mp_raw.shape)} and m_spec {tuple(ms_raw.shape)} must be square "
            f"matrices matching v_r row dimension {d_in}"
        )

    v_norm_sq = (v64 * v64).sum(dim=0)
    if bool((v_norm_sq <= 1e-24).any().item()):
        raise ValueError("v_r columns must have strictly positive Euclidean norm")

    mp = 0.5 * (mp_raw + mp_raw.T)
    ms = 0.5 * (ms_raw + ms_raw.T)
    lam = max(reg_f * max(float(torch.diagonal(ms).mean().item()), 0.0), 1e-6)

    p_quad = (v64 * (mp @ v64)).sum(dim=0).clamp_min(0.0)
    s_quad = ((v64 * (ms @ v64)).sum(dim=0) + lam * v_norm_sq).clamp_min(1e-8)
    return (p_quad / s_quad).contiguous()


def _build_surgery_report(
    *,
    attr_prefix: str,
    rank: int,
    s_vals: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    damped_indices: list[int],
    modified_elements: int,
    adapter_total_elements: int,
    dense_equiv_elements: int,
    denom_params: int,
    noop_reason: str | None,
) -> dict[str, Any]:
    """Assemble standardized telemetry dictionary for single-adapter SVD mode surgery."""
    return {
        "attr_prefix": attr_prefix,
        "rank": rank,
        "effective_rank": int(s_vals.numel()),
        "damped_modes": len(damped_indices),
        "damped_mode_indices": damped_indices,
        "mode_ratios": [float(x) for x in ratios.tolist()],
        "singular_values_before": [float(x) for x in s_vals.tolist()],
        "singular_values_after": [float(x) for x in (s_vals * scales).tolist()],
        "mode_scales": [float(x) for x in scales.tolist()],
        "max_mode_ratio": float(ratios.max().item()),
        "min_scale": float(scales.min().item()),
        "modified_elements": modified_elements,
        "adapter_total_elements": adapter_total_elements,
        "dense_equivalent_elements": dense_equiv_elements,
        "parameter_fraction_modified": float(modified_elements) / float(denom_params),
        "adapter_fraction_modified": float(modified_elements) / float(adapter_total_elements),
        "dense_fraction_modified": float(modified_elements) / float(dense_equiv_elements),
        "noop_reason": noop_reason,
    }


@torch.no_grad()
def apply_lora_svd_mode_surgery_inplace(
    exp_mod: nn.Module,
    attr_prefix: str,
    m_prose: torch.Tensor,
    m_spec: torch.Tensor,
    *,
    tau_svd: float = 1.35,
    eta_svd: float = 3.0,
    max_modes_damped: int = 2,
    reg_spec: float = 0.10,
    truncate: bool = False,
    min_singular_value: float = 1e-10,
    total_model_params: int | None = None,
) -> dict[str, Any]:
    """Selectively damp or truncate interfering LoRA singular modes via QR-SVD right projection.

    For ``B @ A = U_r diag(S_r) V_r^T``, identifies up to ``max_modes_damped`` active singular
    modes (``S_r[k] > min_singular_value``) whose prose-to-specialist Rayleigh ratio satisfies
    ``mu_k > tau_svd``. Computes ``alpha_k = 0.0`` when ``truncate=True`` or
    ``1 / (1 + eta_svd * (mu_k - tau_svd))`` otherwise, and applies the exact rank-``m``
    right-space projector ``A <- A - ((A V_D) diag(1 - alpha_D)) V_D^T``.

    ``B`` and all orthogonal specialist coordinates remain bitwise untouched up to float
    precision, and ``(B @ A_new) @ v_j == (B @ A) @ v_j`` holds for every undamped mode ``j``.
    """
    if not isinstance(exp_mod, nn.Module):
        raise TypeError(f"exp_mod must be an nn.Module, got {type(exp_mod).__name__}")
    if not isinstance(attr_prefix, str):
        raise TypeError(f"attr_prefix must be a str, got {type(attr_prefix).__name__}")
    prefix = attr_prefix.strip()
    if not prefix:
        raise ValueError("attr_prefix must be a non-empty string")

    tau_f = _validate_nonneg_finite(tau_svd, "tau_svd")
    eta_f = _validate_nonneg_finite(eta_svd, "eta_svd")
    max_damped = _validate_nonneg_int(max_modes_damped, "max_modes_damped")
    reg_f = _validate_nonneg_finite(reg_spec, "reg_spec")
    if not isinstance(truncate, bool):
        raise TypeError(f"truncate must be a bool, got {type(truncate).__name__}")
    min_sv = _validate_nonneg_finite(min_singular_value, "min_singular_value")

    w_b = _resolve_lora_factor(exp_mod, f"{prefix}_B")
    w_a = _resolve_lora_factor(exp_mod, f"{prefix}_A")

    _, s_vals, v_r = decompose_lora_qr_svd(w_b, w_a)
    ratios = score_singular_modes_rayleigh_ratio(v_r, m_prose, m_spec, reg_spec=reg_f)

    rank = int(w_a.shape[0])
    k_eff = int(s_vals.numel())
    adapter_total_elements = int(w_b.numel() + w_a.numel())
    dense_equiv_elements = int(w_b.shape[0] * w_a.shape[1])
    denom_params = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else adapter_total_elements
    )
    scales = torch.ones(k_eff, dtype=torch.float64)

    s_max = float(s_vals.max().item())
    if s_max <= min_sv:
        return _build_surgery_report(
            attr_prefix=prefix,
            rank=rank,
            s_vals=s_vals,
            scales=scales,
            ratios=ratios,
            damped_indices=[],
            modified_elements=0,
            adapter_total_elements=adapter_total_elements,
            dense_equiv_elements=dense_equiv_elements,
            denom_params=denom_params,
            noop_reason="zero_singular_norm",
        )

    if max_damped == 0 or (not truncate and eta_f == 0.0):
        return _build_surgery_report(
            attr_prefix=prefix,
            rank=rank,
            s_vals=s_vals,
            scales=scales,
            ratios=ratios,
            damped_indices=[],
            modified_elements=0,
            adapter_total_elements=adapter_total_elements,
            dense_equiv_elements=dense_equiv_elements,
            denom_params=denom_params,
            noop_reason="zero_damping_budget",
        )

    order = torch.argsort(ratios, descending=True)
    damped_indices: list[int] = []
    for idx_t in order:
        k = int(idx_t.item())
        rk = float(ratios[k].item())
        if rk <= tau_f:
            break
        if float(s_vals[k].item()) <= min_sv:
            continue
        scales[k] = 0.0 if truncate else 1.0 / (1.0 + eta_f * (rk - tau_f))
        damped_indices.append(k)
        if len(damped_indices) >= max_damped:
            break

    if not damped_indices:
        return _build_surgery_report(
            attr_prefix=prefix,
            rank=rank,
            s_vals=s_vals,
            scales=scales,
            ratios=ratios,
            damped_indices=[],
            modified_elements=0,
            adapter_total_elements=adapter_total_elements,
            dense_equiv_elements=dense_equiv_elements,
            denom_params=denom_params,
            noop_reason="no_modes_above_threshold",
        )

    idx_tensor = torch.tensor(damped_indices, dtype=torch.long)
    v_d = v_r[:, idx_tensor]
    shrink_d = 1.0 - scales[idx_tensor]
    a64 = w_a.detach().to(device="cpu", dtype=torch.float64)
    delta_a = ((a64 @ v_d) * shrink_d.unsqueeze(0)) @ v_d.T
    a_new = (a64 - delta_a).to(device=w_a.device, dtype=w_a.dtype)

    before_a = w_a.detach().clone()
    w_a.copy_(a_new)
    modified_elements = int(torch.ne(w_a, before_a).sum().item())

    return _build_surgery_report(
        attr_prefix=prefix,
        rank=rank,
        s_vals=s_vals,
        scales=scales,
        ratios=ratios,
        damped_indices=damped_indices,
        modified_elements=modified_elements,
        adapter_total_elements=adapter_total_elements,
        dense_equiv_elements=dense_equiv_elements,
        denom_params=denom_params,
        noop_reason=None,
    )


def _resolve_config(
    config: SVDSurgeryConfig | None,
    *,
    tau_svd: float,
    eta_svd: float,
    max_modes_damped: int,
    reg_spec: float,
    truncate: bool,
    min_singular_value: float,
) -> SVDSurgeryConfig:
    """Return validated SVDSurgeryConfig from explicit config or keyword hyperparameters."""
    if config is not None:
        if not isinstance(config, SVDSurgeryConfig):
            raise TypeError(f"config must be an SVDSurgeryConfig or None, got {type(config).__name__}")
        return config
    return SVDSurgeryConfig(
        tau_svd=tau_svd,
        eta_svd=eta_svd,
        max_modes_damped=max_modes_damped,
        reg_spec=reg_spec,
        truncate=truncate,
        min_singular_value=min_singular_value,
    )


def _aggregate_projection_reports(
    proj_reports: dict[str, dict[str, Any]],
    denom: int,
) -> dict[str, Any]:
    """Aggregate per-projection SVD surgery reports into an expert-level summary."""
    total_damped = sum(int(r["damped_modes"]) for r in proj_reports.values())
    total_modified = sum(int(r["modified_elements"]) for r in proj_reports.values())
    total_adapter = sum(int(r["adapter_total_elements"]) for r in proj_reports.values())
    total_dense = sum(int(r["dense_equivalent_elements"]) for r in proj_reports.values())
    return {
        "projections": proj_reports,
        "damped_modes": total_damped,
        "modified_elements": total_modified,
        "adapter_total_elements": total_adapter,
        "dense_equivalent_elements": total_dense,
        "parameter_fraction_modified": float(total_modified) / float(denom),
        "adapter_fraction_modified": float(total_modified) / float(max(1, total_adapter)),
        "dense_fraction_modified": float(total_modified) / float(max(1, total_dense)),
    }


@torch.no_grad()
def apply_swiglu_expert_svd_surgery_inplace(
    exp_mod: nn.Module,
    m_prose_x: torch.Tensor,
    m_spec_x: torch.Tensor,
    m_prose_act: torch.Tensor | None = None,
    m_spec_act: torch.Tensor | None = None,
    *,
    config: SVDSurgeryConfig | None = None,
    tau_svd: float = 1.35,
    eta_svd: float = 3.0,
    max_modes_damped: int = 2,
    reg_spec: float = 0.10,
    truncate: bool = False,
    min_singular_value: float = 1e-10,
    total_model_params: int | None = None,
) -> dict[str, Any]:
    """Apply QR-SVD singular-mode surgery across a SwiGLU LoRA expert's ``(w1, w2, w3)`` projections."""
    if (m_prose_act is None) != (m_spec_act is None):
        raise ValueError("m_prose_act and m_spec_act must both be provided or both be None")
    cfg = _resolve_config(
        config,
        tau_svd=tau_svd,
        eta_svd=eta_svd,
        max_modes_damped=max_modes_damped,
        reg_spec=reg_spec,
        truncate=truncate,
        min_singular_value=min_singular_value,
    )
    denom_override = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else None
    )

    targets: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("w1", m_prose_x, m_spec_x),
        ("w2", m_prose_x, m_spec_x),
    ]
    if m_prose_act is not None and m_spec_act is not None:
        targets.append(("w3", m_prose_act, m_spec_act))

    proj_reports: dict[str, dict[str, Any]] = {}
    for prefix, mp, ms in targets:
        proj_reports[prefix] = apply_lora_svd_mode_surgery_inplace(
            exp_mod,
            prefix,
            mp,
            ms,
            tau_svd=cfg.tau_svd,
            eta_svd=cfg.eta_svd,
            max_modes_damped=cfg.max_modes_damped,
            reg_spec=cfg.reg_spec,
            truncate=cfg.truncate,
            min_singular_value=cfg.min_singular_value,
            total_model_params=denom_override,
        )

    total_adapter = sum(int(r["adapter_total_elements"]) for r in proj_reports.values())
    denom = denom_override if denom_override is not None else max(1, total_adapter)
    return _aggregate_projection_reports(proj_reports, denom)


@torch.no_grad()
def apply_moe_layer_svd_surgery_inplace(
    moe_mlp: nn.Module,
    prose_x: torch.Tensor,
    spec_x: torch.Tensor,
    *,
    expert_indices: Sequence[int] | None = None,
    config: SVDSurgeryConfig | None = None,
    tau_svd: float = 1.35,
    eta_svd: float = 3.0,
    max_modes_damped: int = 2,
    reg_spec: float = 0.10,
    truncate: bool = False,
    min_singular_value: float = 1e-10,
    include_w3: bool = True,
    total_model_params: int | None = None,
) -> dict[str, Any]:
    """Apply QR-SVD singular-mode surgery across selected experts of a ``CrowdLoRAMoEMLP`` layer."""
    if not isinstance(moe_mlp, nn.Module):
        raise TypeError(f"moe_mlp must be an nn.Module, got {type(moe_mlp).__name__}")
    experts_dict = getattr(moe_mlp, "experts", None)
    if experts_dict is None or not hasattr(experts_dict, "__contains__"):
        raise ValueError("moe_mlp must expose an 'experts' mapping")

    cfg = _resolve_config(
        config,
        tau_svd=tau_svd,
        eta_svd=eta_svd,
        max_modes_damped=max_modes_damped,
        reg_spec=reg_spec,
        truncate=truncate,
        min_singular_value=min_singular_value,
    )
    m_prose_x, _ = second_moment_and_mean(prose_x)
    m_spec_x, _ = second_moment_and_mean(spec_x)

    if expert_indices is None:
        resident_targets = sorted(int(k) for k in experts_dict.keys())
    else:
        requested = sorted({_validate_nonneg_int(idx, "expert_index") for idx in expert_indices})
        resident_targets = [idx for idx in requested if str(idx) in experts_dict]

    layer_total_params = sum(p.numel() for p in moe_mlp.parameters())
    denom = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else max(1, layer_total_params)
    )

    expert_reports: dict[str, dict[str, Any]] = {}
    for e_idx in resident_targets:
        expert_reports[str(e_idx)] = apply_swiglu_expert_svd_surgery_inplace(
            experts_dict[str(e_idx)],
            m_prose_x,
            m_spec_x,
            config=cfg,
            total_model_params=denom,
        )

    if include_w3 and resident_targets:
        act_prose = compute_swiglu_intermediate_states(moe_mlp, prose_x, resident_targets)
        act_spec = compute_swiglu_intermediate_states(moe_mlp, spec_x, resident_targets)
        m_prose_act, _ = second_moment_and_mean(act_prose)
        m_spec_act, _ = second_moment_and_mean(act_spec)
        for e_idx in resident_targets:
            rep = expert_reports[str(e_idx)]
            rep["projections"]["w3"] = apply_lora_svd_mode_surgery_inplace(
                experts_dict[str(e_idx)],
                "w3",
                m_prose_act,
                m_spec_act,
                tau_svd=cfg.tau_svd,
                eta_svd=cfg.eta_svd,
                max_modes_damped=cfg.max_modes_damped,
                reg_spec=cfg.reg_spec,
                truncate=cfg.truncate,
                min_singular_value=cfg.min_singular_value,
                total_model_params=denom,
            )
            expert_reports[str(e_idx)] = _aggregate_projection_reports(rep["projections"], denom)

    total_damped = sum(int(r["damped_modes"]) for r in expert_reports.values())
    total_modified = sum(int(r["modified_elements"]) for r in expert_reports.values())
    total_adapter = sum(int(r["adapter_total_elements"]) for r in expert_reports.values())
    total_dense = sum(int(r["dense_equivalent_elements"]) for r in expert_reports.values())

    return {
        "experts": expert_reports,
        "targeted_experts": resident_targets,
        "damped_modes": total_damped,
        "modified_elements": total_modified,
        "adapter_total_elements": total_adapter,
        "dense_equivalent_elements": total_dense,
        "layer_total_params": layer_total_params,
        "parameter_fraction_modified": float(total_modified) / float(denom),
        "adapter_fraction_modified": float(total_modified) / float(max(1, total_adapter)),
        "dense_fraction_modified": float(total_modified) / float(max(1, total_dense)),
    }


__all__ = [
    "SVDSurgeryConfig",
    "apply_lora_svd_mode_surgery_inplace",
    "apply_moe_layer_svd_surgery_inplace",
    "apply_swiglu_expert_svd_surgery_inplace",
    "decompose_lora_qr_svd",
    "score_singular_modes_rayleigh_ratio",
]
