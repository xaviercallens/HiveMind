# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-04: QR-SVD rank-selective LoRA singular-mode surgery (LASER)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from gwaya.ast_audit import ZeroStubAudit
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdRuntimeMLP, CrowdV1RuntimeConfig
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAExpertAdapter, CrowdLoRAMoEMLP
from hivemind.swarm.lora_svd_mode_surgery import (
    SVDSurgeryConfig,
    apply_lora_svd_mode_surgery_inplace,
    apply_moe_layer_svd_surgery_inplace,
    apply_swiglu_expert_svd_surgery_inplace,
    decompose_lora_qr_svd,
    score_singular_modes_rayleigh_ratio,
)
from hivemind.swarm.wsp_alphaedit_control import second_moment_and_mean
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "lora_svd_mode_surgery.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _build_planted_rank8_factors(
    *,
    d_out: int,
    d_in: int,
    prose_dim: int,
    prose_singular_value: float,
    spec_singular_values: list[float],
    seed: int,
    dtype: torch.dtype = torch.float32,
    out_prose_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct rank-8 factors (B, A) with mode 0 in prose_dim and modes 1..7 in specialist subspace."""
    assert len(spec_singular_values) == 7
    q_p = _make_orthogonal_basis(prose_dim, seed=seed + 1)
    q_s = _make_orthogonal_basis(d_in - prose_dim, seed=seed + 2)

    v_modes = torch.zeros((d_in, 8), dtype=torch.float64)
    v_modes[:prose_dim, 0] = q_p[:, 0]
    v_modes[prose_dim:, 1:] = q_s[:, :7]

    if out_prose_dim is None:
        u_modes = _make_orthogonal_basis(d_out, seed=seed)[:, :8]
    else:
        u_p = _make_orthogonal_basis(out_prose_dim, seed=seed + 3)
        u_s = _make_orthogonal_basis(d_out - out_prose_dim, seed=seed + 4)
        u_modes = torch.zeros((d_out, 8), dtype=torch.float64)
        u_modes[:out_prose_dim, 0] = u_p[:, 0]
        u_modes[out_prose_dim:, 1:] = u_s[:, :7]

    s_vals = torch.tensor([prose_singular_value, *spec_singular_values], dtype=torch.float64)
    s_sqrt = s_vals.sqrt()
    w_b = (u_modes * s_sqrt.unsqueeze(0)).to(dtype=dtype).contiguous()
    w_a = (s_sqrt.unsqueeze(1) * v_modes.T).to(dtype=dtype).contiguous()
    return w_b, w_a, u_modes, v_modes


def _make_subspace_moments(
    *,
    d_in: int,
    prose_dim: int,
    n_samples: int = 128,
    prose_scale: float = 2.5,
    spec_scale: float = 2.5,
    seed: int = 20260410,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (prose_x, spec_x, m_prose, m_spec) separated onto orthogonal subspaces."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    prose_x = torch.zeros((n_samples, d_in), dtype=torch.float32)
    spec_x = torch.zeros((n_samples, d_in), dtype=torch.float32)
    prose_x[:, :prose_dim] = torch.randn((n_samples, prose_dim), generator=gen, dtype=torch.float32) * prose_scale
    spec_x[:, prose_dim:] = torch.randn((n_samples, d_in - prose_dim), generator=gen, dtype=torch.float32) * spec_scale
    m_prose, _ = second_moment_and_mean(prose_x)
    m_spec, _ = second_moment_and_mean(spec_x)
    return prose_x, spec_x, m_prose, m_spec


class _LinearWrappedAdapter(nn.Module):
    """Minimal adapter module storing LoRA factors as nn.Linear attributes."""

    def __init__(self, d_out: int, d_in: int, rank: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.lora_B = nn.Linear(rank, d_out, bias=False, dtype=torch.float32)
        self.lora_A = nn.Linear(d_in, rank, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.lora_B.weight.copy_(torch.randn((d_out, rank), generator=gen, dtype=torch.float32) * 0.2)
            self.lora_A.weight.copy_(torch.randn((rank, d_in), generator=gen, dtype=torch.float32) * 0.2)


# ---------------------------------------------------------------------------
# S1: Exact Equivalence Between QR-SVD and Dense SVD on Rectangular SwiGLU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d_out", "d_in", "rank", "dtype"),
    [
        (128, 64, 8, torch.float64),
        (64, 128, 8, torch.float32),
        (96, 48, 4, torch.bfloat16),
        (6, 8, 10, torch.float32),
    ],
)
def test_s1_qr_svd_exact_equivalence_with_dense_svd_on_rectangular_swiglu(
    d_out: int,
    d_in: int,
    rank: int,
    dtype: torch.dtype,
) -> None:
    """Verify QR-SVD matches dense SVD in singular values, subspaces, reconstruction, and Rayleigh ratios."""
    gen = torch.Generator(device="cpu").manual_seed(20260100 + d_out + d_in + rank)
    w_b = (torch.randn((d_out, rank), generator=gen, dtype=torch.float32) * 0.25).to(dtype)
    w_a = (torch.randn((rank, d_in), generator=gen, dtype=torch.float32) * 0.25).to(dtype)

    u_r, s_r, v_r = decompose_lora_qr_svd(w_b, w_a)
    k_eff = min(d_out, d_in, rank)
    assert u_r.shape == (d_out, k_eff) and u_r.dtype == torch.float64 and u_r.device.type == "cpu"
    assert s_r.shape == (k_eff,) and s_r.dtype == torch.float64
    assert v_r.shape == (d_in, k_eff) and v_r.dtype == torch.float64

    eye_k = torch.eye(k_eff, dtype=torch.float64)
    assert torch.allclose(u_r.T @ u_r, eye_k, atol=1e-12)
    assert torch.allclose(v_r.T @ v_r, eye_k, atol=1e-12)

    dense_prod = w_b.detach().to(torch.float64) @ w_a.detach().to(torch.float64)
    u_d, s_d, vh_d = torch.linalg.svd(dense_prod, full_matrices=False)
    v_d = vh_d.T

    assert torch.allclose(s_r, s_d[:k_eff], atol=1e-11, rtol=1e-11)
    if s_d.numel() > k_eff:
        assert float(s_d[k_eff:].abs().max().item()) < 1e-12

    recon_qr = (u_r * s_r.unsqueeze(0)) @ v_r.T
    recon_dense = (u_d[:, :k_eff] * s_d[:k_eff].unsqueeze(0)) @ v_d[:, :k_eff].T
    assert torch.allclose(recon_qr, dense_prod, atol=1e-11)
    assert torch.allclose(recon_qr, recon_dense, atol=1e-11)

    # Verify per-vector sign-paired equivalence and Rayleigh ratio parity for distinct spectrum
    if rank <= min(d_out, d_in) and dtype != torch.bfloat16:
        q_out = _make_orthogonal_basis(d_out, seed=20260151)
        q_in = _make_orthogonal_basis(d_in, seed=20260152)
        distinct_s = torch.linspace(4.0, 0.8, rank, dtype=torch.float64)
        b_dist = (q_out[:, :rank] * distinct_s.sqrt().unsqueeze(0)).to(dtype)
        a_dist = (distinct_s.sqrt().unsqueeze(1) * q_in[:, :rank].T).to(dtype)

        uq, sq, vq = decompose_lora_qr_svd(b_dist, a_dist)
        ud, sd, vhd = torch.linalg.svd(b_dist.double() @ a_dist.double(), full_matrices=False)
        vd = vhd.T[:, :rank]
        ud = ud[:, :rank]
        assert torch.allclose(sq, sd[:rank], atol=1e-11)
        for k in range(rank):
            u_dot = float((uq[:, k] @ ud[:, k]).item())
            v_dot = float((vq[:, k] @ vd[:, k]).item())
            assert abs(u_dot) > 1.0 - 1e-11
            assert abs(v_dot) > 1.0 - 1e-11
            assert u_dot * v_dot > 1.0 - 1e-11

        _, _, m_prose, m_spec = _make_subspace_moments(d_in=d_in, prose_dim=d_in // 2, seed=20260153)
        ratios_qr = score_singular_modes_rayleigh_ratio(vq, m_prose, m_spec, reg_spec=0.10)
        ratios_dense = score_singular_modes_rayleigh_ratio(vd, m_prose, m_spec, reg_spec=0.10)
        assert torch.allclose(ratios_qr, ratios_dense, atol=1e-11, rtol=1e-11)
        # Positive column scaling invariance of generalized Rayleigh ratio
        ratios_scaled = score_singular_modes_rayleigh_ratio(vq * 3.75, m_prose, m_spec, reg_spec=0.10)
        assert torch.allclose(ratios_qr, ratios_scaled, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# S2: Surgical Removal of Planted Rank-1 Prose Mode Out of Rank 8
# ---------------------------------------------------------------------------


def test_s2_surgical_removal_of_planted_rank1_prose_mode_preserves_7_specialist_modes() -> None:
    """Remove 1 planted prose-corrupting mode out of rank 8 while keeping 7 specialist modes <1e-5 error."""
    d_in, d_out, prose_dim = 64, 128, 16
    _, _, m_prose, m_spec = _make_subspace_moments(
        d_in=d_in, prose_dim=prose_dim, n_samples=192, prose_scale=3.0, spec_scale=2.5, seed=20260201
    )
    spec_svs = [3.6, 3.1, 2.8, 2.1, 1.7, 1.3, 0.9]
    b_init, a_init, _, v_modes = _build_planted_rank8_factors(
        d_out=d_out,
        d_in=d_in,
        prose_dim=prose_dim,
        prose_singular_value=2.4,
        spec_singular_values=spec_svs,
        seed=20260202,
        dtype=torch.float32,
    )

    # Part A: Hard truncation (truncate=True) of the single prose mode out of rank 8
    exp_trunc = CrowdLoRAExpertAdapter(d_in, d_out, rank=8, alpha=16.0, seed=20260203)
    with torch.no_grad():
        exp_trunc.w1_B.copy_(b_init)
        exp_trunc.w1_A.copy_(a_init)
    w_before = exp_trunc.w1_B.double() @ exp_trunc.w1_A.double()
    b_snap = exp_trunc.w1_B.detach().clone()
    a_snap = exp_trunc.w1_A.detach().clone()

    rep_trunc = apply_lora_svd_mode_surgery_inplace(
        exp_trunc,
        "w1",
        m_prose,
        m_spec,
        tau_svd=1.35,
        eta_svd=3.0,
        max_modes_damped=2,
        reg_spec=0.10,
        truncate=True,
        total_model_params=10_000_000,
    )
    assert rep_trunc["damped_modes"] == 1
    assert rep_trunc["min_scale"] == 0.0
    assert rep_trunc["noop_reason"] is None
    assert rep_trunc["max_mode_ratio"] > 5.0
    assert torch.equal(exp_trunc.w1_B, b_snap)
    assert torch.equal(exp_trunc.w1_A[1:, prose_dim:], a_snap[1:, prose_dim:])
    assert torch.allclose(exp_trunc.w1_A[:, prose_dim:], a_snap[:, prose_dim:], atol=1e-6)

    expected_mod = int(torch.ne(exp_trunc.w1_A, a_snap).sum().item())
    assert rep_trunc["modified_elements"] == expected_mod
    assert 0 < rep_trunc["modified_elements"] <= exp_trunc.w1_A.numel()
    assert rep_trunc["parameter_fraction_modified"] == pytest.approx(expected_mod / 10_000_000.0, rel=1e-12)
    assert rep_trunc["parameter_fraction_modified"] < 0.0005

    w_after = exp_trunc.w1_B.double() @ exp_trunc.w1_A.double()
    # Planted prose mode v_0 is annihilated
    assert float((w_after @ v_modes[:, 0]).norm().item()) < 1e-5
    # All 7 specialist modes v_1..v_7 are preserved to <1e-5 error
    for idx in range(1, 8):
        v_j = v_modes[:, idx]
        rel_err = float(((w_after - w_before) @ v_j).norm().item() / (w_before @ v_j).norm().item())
        assert rel_err < 1e-5

    # Part B: Smooth LASER damping (truncate=False) verifies exact attenuation formula
    exp_smooth = CrowdLoRAExpertAdapter(d_in, d_out, rank=8, alpha=16.0, seed=20260204)
    with torch.no_grad():
        exp_smooth.w1_B.copy_(b_init)
        exp_smooth.w1_A.copy_(a_init)
    rep_smooth = apply_lora_svd_mode_surgery_inplace(
        exp_smooth,
        "w1",
        m_prose,
        m_spec,
        tau_svd=1.35,
        eta_svd=4.0,
        max_modes_damped=2,
        reg_spec=0.10,
        truncate=False,
    )
    assert rep_smooth["damped_modes"] == 1
    d_idx = rep_smooth["damped_mode_indices"][0]
    mu_d = rep_smooth["mode_ratios"][d_idx]
    expected_scale = 1.0 / (1.0 + 4.0 * (mu_d - 1.35))
    assert rep_smooth["min_scale"] == pytest.approx(expected_scale, rel=1e-12)
    w_smooth = exp_smooth.w1_B.double() @ exp_smooth.w1_A.double()
    assert torch.allclose(w_smooth @ v_modes[:, 0], expected_scale * (w_before @ v_modes[:, 0]), atol=1e-5)
    for idx in range(1, 8):
        assert float(((w_smooth - w_before) @ v_modes[:, idx]).norm().item()) < 1e-5

    # Part C: 3 interfering modes + 1 zero-singular mode -> max_modes_damped=2 skips zero mode and damps top 2 active
    q_out = _make_orthogonal_basis(d_out, seed=20260205)
    q_p = _make_orthogonal_basis(prose_dim, seed=20260206)
    q_s = _make_orthogonal_basis(d_in - prose_dim, seed=20260207)
    v_multi = torch.zeros((d_in, 8), dtype=torch.float64)
    v_multi[:prose_dim, :4] = q_p[:, :4]
    v_multi[prose_dim:, 4:] = q_s[:, :4]
    s_multi = torch.tensor([3.0, 2.5, 2.0, 0.0, 3.2, 2.7, 2.2, 1.5], dtype=torch.float64)
    # Give prose mode 3 (zero singular value) highest prose variance so it ranks first in ratios
    x_p_skew = torch.zeros((64, d_in), dtype=torch.float32)
    x_p_skew[:, :prose_dim] = (
        torch.randn((64, 4), generator=torch.Generator(device="cpu").manual_seed(20260208))
        * torch.tensor([3.0, 2.5, 2.0, 8.0])
    ) @ q_p[:, :4].T.float()
    mp_skew, _ = second_moment_and_mean(x_p_skew)

    exp_cap = CrowdLoRAExpertAdapter(d_in, d_out, rank=8, alpha=16.0, seed=20260209)
    with torch.no_grad():
        exp_cap.w1_B.copy_((q_out[:, :8] * s_multi.sqrt().unsqueeze(0)).float())
        exp_cap.w1_A.copy_((s_multi.sqrt().unsqueeze(1) * v_multi.T).float())
    w_cap_before = exp_cap.w1_B.double() @ exp_cap.w1_A.double()
    rep_cap = apply_lora_svd_mode_surgery_inplace(
        exp_cap,
        "w1",
        mp_skew,
        m_spec,
        tau_svd=1.20,
        eta_svd=5.0,
        max_modes_damped=2,
        truncate=True,
    )
    assert rep_cap["damped_modes"] == 2
    w_cap_after = exp_cap.w1_B.double() @ exp_cap.w1_A.double()
    for idx in range(4, 8):
        assert float(((w_cap_after - w_cap_before) @ v_multi[:, idx]).norm().item()) < 1e-5


# ---------------------------------------------------------------------------
# S3: Zero-Norm / Freshly Initialized B=0 No-Op Safety
# ---------------------------------------------------------------------------


def test_s3_zero_norm_and_freshly_initialized_b_zero_noop_safety() -> None:
    """Verify freshly initialized B=0, zero A=0, sub-threshold ratios, and zero budgets are bitwise no-ops."""
    d_in, d_out, rank = 32, 64, 8
    _, _, m_prose, m_spec = _make_subspace_moments(d_in=d_in, prose_dim=16, seed=20260301)

    # Fresh CrowdLoRAExpertAdapter has w1_B = w2_B = w3_B = 0 and non-zero w*_A
    exp_fresh = CrowdLoRAExpertAdapter(d_in, d_out, rank=rank, alpha=16.0, seed=20260302)
    snap_fresh = {name: p.detach().clone() for name, p in exp_fresh.named_parameters()}
    _, _, m_act_p, m_act_s = _make_subspace_moments(d_in=d_out, prose_dim=32, seed=20260303)

    rep_w1 = apply_lora_svd_mode_surgery_inplace(
        exp_fresh, "w1", m_prose, m_spec, tau_svd=0.0, eta_svd=10.0, truncate=True
    )
    assert rep_w1["damped_modes"] == 0
    assert rep_w1["modified_elements"] == 0
    assert rep_w1["parameter_fraction_modified"] == 0.0
    assert rep_w1["noop_reason"] == "zero_singular_norm"

    rep_all = apply_swiglu_expert_svd_surgery_inplace(
        exp_fresh, m_prose, m_spec, m_act_p, m_act_s, tau_svd=0.0, eta_svd=10.0, truncate=True
    )
    assert rep_all["damped_modes"] == 0
    assert rep_all["modified_elements"] == 0
    assert rep_all["parameter_fraction_modified"] == 0.0
    for name, p in exp_fresh.named_parameters():
        assert torch.equal(p, snap_fresh[name])

    # Zero A=0 with non-zero B also triggers clean zero_singular_norm no-op
    with torch.no_grad():
        exp_fresh.w1_B.fill_(0.5)
        exp_fresh.w1_A.zero_()
    b_zero_a = exp_fresh.w1_B.detach().clone()
    a_zero_a = exp_fresh.w1_A.detach().clone()
    rep_zero_a = apply_lora_svd_mode_surgery_inplace(exp_fresh, "w1", m_prose, m_spec, tau_svd=0.0, truncate=True)
    assert rep_zero_a["noop_reason"] == "zero_singular_norm"
    assert torch.equal(exp_fresh.w1_B, b_zero_a) and torch.equal(exp_fresh.w1_A, a_zero_a)

    # Non-zero factors with zero budget (max_modes_damped=0 and eta_svd=0.0) or tau_svd above max ratio
    b_nz, a_nz, _, _ = _build_planted_rank8_factors(
        d_out=d_out,
        d_in=d_in,
        prose_dim=16,
        prose_singular_value=2.0,
        spec_singular_values=[3.0, 2.5, 2.2, 1.8, 1.5, 1.2, 0.9],
        seed=20260304,
    )
    with torch.no_grad():
        exp_fresh.w1_B.copy_(b_nz)
        exp_fresh.w1_A.copy_(a_nz)

    rep_budget0 = apply_lora_svd_mode_surgery_inplace(exp_fresh, "w1", m_prose, m_spec, max_modes_damped=0)
    assert rep_budget0["noop_reason"] == "zero_damping_budget"
    assert rep_budget0["modified_elements"] == 0
    assert torch.equal(exp_fresh.w1_A, a_nz)

    rep_eta0 = apply_lora_svd_mode_surgery_inplace(exp_fresh, "w1", m_prose, m_spec, eta_svd=0.0, truncate=False)
    assert rep_eta0["noop_reason"] == "zero_damping_budget"
    assert torch.equal(exp_fresh.w1_A, a_nz)

    rep_high_tau = apply_lora_svd_mode_surgery_inplace(exp_fresh, "w1", m_prose, m_spec, tau_svd=1e6, eta_svd=3.0)
    assert rep_high_tau["noop_reason"] == "no_modes_above_threshold"
    assert rep_high_tau["damped_modes"] == 0
    assert torch.equal(exp_fresh.w1_A, a_nz)


# ---------------------------------------------------------------------------
# S4: bf16 Safetensors Round-Trip Stability Across SwiGLU (w1, w2, w3)
# ---------------------------------------------------------------------------


def test_s4_bf16_safetensors_roundtrip_stability_across_swiglu_adapters(tmp_path: Path) -> None:
    """Verify bf16 safetensors round-trip stability and prose suppression across SwiGLU w1/w2/w3."""
    d_in, d_out, prose_in, prose_act = 48, 96, 16, 32
    prose_x, spec_x, m_prose_x, m_spec_x = _make_subspace_moments(
        d_in=d_in, prose_dim=prose_in, n_samples=160, prose_scale=2.8, spec_scale=2.4, seed=20260401
    )

    gen = torch.Generator(device="cpu").manual_seed(20260402)
    dense_mlp = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=d_in, intermediate_size=d_out))
    with torch.no_grad():
        dense_mlp.w1.weight.zero_()
        dense_mlp.w2.weight.zero_()
        dense_mlp.w3.weight.zero_()
        dense_mlp.w1.weight[:prose_act, :prose_in].copy_(torch.randn((prose_act, prose_in), generator=gen) * 0.25)
        dense_mlp.w2.weight[:prose_act, :prose_in].copy_(torch.randn((prose_act, prose_in), generator=gen) * 0.25)
        dense_mlp.w3.weight[:prose_in, :prose_act].copy_(torch.randn((prose_in, prose_act), generator=gen) * 0.25)
        dense_mlp.w1.weight[prose_act:, prose_in:].copy_(
            torch.randn((d_out - prose_act, d_in - prose_in), generator=gen) * 0.25
        )
        dense_mlp.w2.weight[prose_act:, prose_in:].copy_(
            torch.randn((d_out - prose_act, d_in - prose_in), generator=gen) * 0.25
        )
        dense_mlp.w3.weight[prose_in:, prose_act:].copy_(
            torch.randn((d_in - prose_in, d_out - prose_act), generator=gen) * 0.25
        )

    moe = CrowdLoRAMoEMLP(
        dense_mlp,
        num_experts=4,
        top_k=2,
        rank=8,
        alpha=16.0,
        routing_mode="sigma_topk",
        resident_expert_indices=(0, 1, 2, 3),
        seed=20260403,
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[1:3, :prose_in].fill_(0.8)
        moe.router.weight[1:3, prose_in:].fill_(1.0)
        for e_id in ("1", "2"):
            exp = moe.experts[e_id]
            b1, a1, _, _ = _build_planted_rank8_factors(
                d_out=d_out,
                d_in=d_in,
                prose_dim=prose_in,
                prose_singular_value=2.5,
                spec_singular_values=[3.4, 3.0, 2.6, 2.2, 1.8, 1.4, 1.0],
                seed=20260410 + int(e_id) * 10,
                out_prose_dim=prose_act,
            )
            b2, a2, _, _ = _build_planted_rank8_factors(
                d_out=d_out,
                d_in=d_in,
                prose_dim=prose_in,
                prose_singular_value=2.4,
                spec_singular_values=[3.3, 2.9, 2.5, 2.1, 1.7, 1.3, 0.9],
                seed=20260411 + int(e_id) * 10,
                out_prose_dim=prose_act,
            )
            b3, a3, _, _ = _build_planted_rank8_factors(
                d_out=d_in,
                d_in=d_out,
                prose_dim=prose_act,
                prose_singular_value=2.6,
                spec_singular_values=[3.5, 3.1, 2.7, 2.3, 1.9, 1.5, 1.1],
                seed=20260412 + int(e_id) * 10,
                out_prose_dim=prose_in,
            )
            exp.w1_B.copy_(b1)
            exp.w1_A.copy_(a1)
            exp.w2_B.copy_(b2)
            exp.w2_A.copy_(a2)
            exp.w3_B.copy_(b3)
            exp.w3_A.copy_(a3)

    with torch.no_grad():
        dense_prose = dense_mlp(prose_x)
        dense_spec = dense_mlp(spec_x)
        delta_prose_before = float((moe(prose_x) - dense_prose).norm().item())
        delta_spec_before = moe(spec_x) - dense_spec

    layer_rep = apply_moe_layer_svd_surgery_inplace(
        moe,
        prose_x,
        spec_x,
        expert_indices=[1, 2],
        tau_svd=1.35,
        eta_svd=12.0,
        max_modes_damped=2,
        truncate=True,
        include_w3=True,
    )
    assert layer_rep["targeted_experts"] == [1, 2]
    assert layer_rep["damped_modes"] >= 4
    assert layer_rep["modified_elements"] > 0
    assert 0.0 < layer_rep["parameter_fraction_modified"] < 0.25

    with torch.no_grad():
        delta_prose_after = float((moe(prose_x) - dense_prose).norm().item())
        delta_spec_after = moe(spec_x) - dense_spec
        assert delta_prose_after < 0.05 * delta_prose_before
        spec_rel_drift = float((delta_spec_after - delta_spec_before).norm().item() / delta_spec_before.norm().item())
        assert spec_rel_drift < 1e-4

    # Save edited SwiGLU expert tensors as bf16 safetensors and reload cleanly
    st_path = tmp_path / "swiglu_svd_bf16.safetensors"
    bf16_payload = {
        name: p.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for name, p in moe.experts["1"].named_parameters()
    }
    save_file(bf16_payload, str(st_path))
    reloaded_bf16 = load_file(str(st_path), device="cpu")
    assert set(reloaded_bf16.keys()) == set(bf16_payload.keys())
    for name, tensor in reloaded_bf16.items():
        assert tensor.dtype == torch.bfloat16
        assert bool(torch.isfinite(tensor).all().item())
        assert torch.equal(tensor, bf16_payload[name])

    # Run surgery directly on bf16 SwiGLU adapter weights and round-trip again via safetensors
    exp_bf16 = CrowdLoRAExpertAdapter(d_in, d_out, rank=8, alpha=16.0, seed=20260420)
    b1, a1, _, v1 = _build_planted_rank8_factors(
        d_out=d_out,
        d_in=d_in,
        prose_dim=prose_in,
        prose_singular_value=2.8,
        spec_singular_values=[3.6, 3.2, 2.7, 2.3, 1.9, 1.5, 1.1],
        seed=20260421,
        dtype=torch.bfloat16,
    )
    b2, a2, _, v2 = _build_planted_rank8_factors(
        d_out=d_out,
        d_in=d_in,
        prose_dim=prose_in,
        prose_singular_value=2.7,
        spec_singular_values=[3.5, 3.1, 2.6, 2.2, 1.8, 1.4, 1.0],
        seed=20260422,
        dtype=torch.bfloat16,
    )
    b3, a3, _, v3 = _build_planted_rank8_factors(
        d_out=d_in,
        d_in=d_out,
        prose_dim=prose_act,
        prose_singular_value=2.9,
        spec_singular_values=[3.7, 3.3, 2.8, 2.4, 2.0, 1.6, 1.2],
        seed=20260423,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        exp_bf16.w1_B = nn.Parameter(b1)
        exp_bf16.w1_A = nn.Parameter(a1)
        exp_bf16.w2_B = nn.Parameter(b2)
        exp_bf16.w2_A = nn.Parameter(a2)
        exp_bf16.w3_B = nn.Parameter(b3)
        exp_bf16.w3_A = nn.Parameter(a3)
        b_snaps = {k: getattr(exp_bf16, k).detach().clone() for k in ("w1_B", "w2_B", "w3_B")}
        w1_spec_before = (exp_bf16.w1_B.double() @ exp_bf16.w1_A.double()) @ v1[:, 1:]
        w2_spec_before = (exp_bf16.w2_B.double() @ exp_bf16.w2_A.double()) @ v2[:, 1:]
        w3_spec_before = (exp_bf16.w3_B.double() @ exp_bf16.w3_A.double()) @ v3[:, 1:]

    _, _, m_prose_act, m_spec_act = _make_subspace_moments(
        d_in=d_out, prose_dim=prose_act, n_samples=160, prose_scale=2.8, spec_scale=2.4, seed=20260424
    )
    bf16_rep = apply_swiglu_expert_svd_surgery_inplace(
        exp_bf16,
        m_prose_x,
        m_spec_x,
        m_prose_act,
        m_spec_act,
        config=SVDSurgeryConfig(tau_svd=1.35, eta_svd=6.0, max_modes_damped=2, truncate=True),
        total_model_params=50_000_000,
    )
    assert bf16_rep["damped_modes"] == 3
    assert bf16_rep["parameter_fraction_modified"] < 0.0005
    for k, snap in b_snaps.items():
        assert torch.equal(getattr(exp_bf16, k), snap)

    st_path2 = tmp_path / "exp_bf16_post_surgery.safetensors"
    save_file({n: p.detach().contiguous() for n, p in exp_bf16.named_parameters()}, str(st_path2))
    rel2 = load_file(str(st_path2), device="cpu")
    for n, t in rel2.items():
        assert t.dtype == torch.bfloat16
        assert torch.equal(t, getattr(exp_bf16, n).detach())

    # Specialist singular modes survive bf16 surgery + safetensors round-trip cleanly
    w1_spec_after = (rel2["w1_B"].double() @ rel2["w1_A"].double()) @ v1[:, 1:]
    w2_spec_after = (rel2["w2_B"].double() @ rel2["w2_A"].double()) @ v2[:, 1:]
    w3_spec_after = (rel2["w3_B"].double() @ rel2["w3_A"].double()) @ v3[:, 1:]
    assert float((w1_spec_after - w1_spec_before).norm().item() / w1_spec_before.norm().item()) < 1e-2
    assert float((w2_spec_after - w2_spec_before).norm().item() / w2_spec_before.norm().item()) < 1e-2
    assert float((w3_spec_after - w3_spec_before).norm().item() / w3_spec_before.norm().item()) < 1e-2

    # Exercise include_w3=False, expert_indices=None, and non-resident expert filtering
    rep_no_w3 = apply_moe_layer_svd_surgery_inplace(moe, prose_x, spec_x, expert_indices=None, include_w3=False)
    assert len(rep_no_w3["experts"]) == 4
    assert "w3" not in rep_no_w3["experts"]["0"]["projections"]
    rep_nonres = apply_moe_layer_svd_surgery_inplace(moe, prose_x, spec_x, expert_indices=[99])
    assert rep_nonres["targeted_experts"] == [] and rep_nonres["damped_modes"] == 0


# ---------------------------------------------------------------------------
# S5: Validation, Hyperparameter Boundaries & Security Audits
# ---------------------------------------------------------------------------


def test_s5_validation_hyperparameter_boundaries_and_security_gates() -> None:
    """Test validation error branches, boundary hyperparameters, nn.Linear adapters, and security gates."""
    # 1. decompose_lora_qr_svd validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        decompose_lora_qr_svd([[1.0]], torch.ones((1, 1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D tensor"):
        decompose_lora_qr_svd(torch.ones(4), torch.ones((4, 4)))
    with pytest.raises(ValueError, match="non-empty"):
        decompose_lora_qr_svd(torch.zeros((0, 4)), torch.ones((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        decompose_lora_qr_svd(torch.tensor([[1.0, float("nan")]]), torch.ones((2, 3)))
    with pytest.raises(ValueError, match="Inner rank mismatch"):
        decompose_lora_qr_svd(torch.ones((4, 3)), torch.ones((2, 4)))

    # 2. score_singular_modes_rayleigh_ratio validation & boundary reg_spec=0.0
    v_ok = torch.eye(4, 2, dtype=torch.float64)
    m_id = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="square matrices"):
        score_singular_modes_rayleigh_ratio(v_ok, torch.ones((3, 3)), m_id)
    with pytest.raises(ValueError, match="positive Euclidean norm"):
        score_singular_modes_rayleigh_ratio(torch.zeros((4, 2)), m_id, m_id)
    with pytest.raises(TypeError, match="real number"):
        score_singular_modes_rayleigh_ratio(v_ok, m_id, m_id, reg_spec=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        score_singular_modes_rayleigh_ratio(v_ok, m_id, m_id, reg_spec=-0.1)
    r_zero_reg = score_singular_modes_rayleigh_ratio(v_ok, m_id, torch.zeros((4, 4)), reg_spec=0.0)
    assert torch.allclose(r_zero_reg, torch.full((2,), 1e6, dtype=torch.float64))

    # 3. SVDSurgeryConfig validation
    with pytest.raises(TypeError, match="bool"):
        SVDSurgeryConfig(truncate="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        SVDSurgeryConfig(max_modes_damped=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        SVDSurgeryConfig(max_modes_damped=-1)
    with pytest.raises(ValueError, match=">="):
        SVDSurgeryConfig(tau_svd=float("inf"))

    # 4. apply_lora_svd_mode_surgery_inplace validation & nn.Linear wrapper path
    lin_mod = _LinearWrappedAdapter(d_out=16, d_in=8, rank=4, seed=20260501)
    _, _, mp_8, ms_8 = _make_subspace_moments(d_in=8, prose_dim=4, seed=20260502)
    rep_lin = apply_lora_svd_mode_surgery_inplace(lin_mod, "lora", mp_8, ms_8, tau_svd=0.5, truncate=True)
    assert rep_lin["attr_prefix"] == "lora" and rep_lin["rank"] == 4

    with pytest.raises(TypeError, match="nn.Module"):
        apply_lora_svd_mode_surgery_inplace("bad", "w1", mp_8, ms_8)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str"):
        apply_lora_svd_mode_surgery_inplace(lin_mod, 123, mp_8, ms_8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        apply_lora_svd_mode_surgery_inplace(lin_mod, "  ", mp_8, ms_8)
    with pytest.raises(TypeError, match="bool"):
        apply_lora_svd_mode_surgery_inplace(lin_mod, "lora", mp_8, ms_8, truncate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expose tensor or nn.Linear"):
        apply_lora_svd_mode_surgery_inplace(lin_mod, "missing", mp_8, ms_8)
    with pytest.raises(ValueError, match=">= 1"):
        apply_lora_svd_mode_surgery_inplace(lin_mod, "lora", mp_8, ms_8, total_model_params=0)

    # 5. apply_swiglu_expert_svd_surgery_inplace & apply_moe_layer_svd_surgery_inplace validation
    exp_small = CrowdLoRAExpertAdapter(8, 16, rank=2)
    with pytest.raises(ValueError, match="both be provided"):
        apply_swiglu_expert_svd_surgery_inplace(exp_small, mp_8, ms_8, m_prose_act=mp_8)
    with pytest.raises(TypeError, match="SVDSurgeryConfig"):
        apply_swiglu_expert_svd_surgery_inplace(exp_small, mp_8, ms_8, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        apply_moe_layer_svd_surgery_inplace("bad", torch.ones((4, 8)), torch.ones((4, 8)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="experts"):
        apply_moe_layer_svd_surgery_inplace(nn.Linear(8, 8), torch.ones((4, 8)), torch.ones((4, 8)))
    moe_small = CrowdLoRAMoEMLP(
        CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16)),
        num_experts=2,
        top_k=1,
        rank=2,
    )
    with pytest.raises(ValueError, match=">="):
        apply_moe_layer_svd_surgery_inplace(
            moe_small, torch.ones((4, 8)), torch.ones((4, 8)), expert_indices=[-1]
        )

    # 6. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement & pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl04", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl04", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/lora_svd_mode_surgery.py", TARGET_MODULE_PATH),
        ("tests/test_lora_svd_mode_surgery_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        stub_res = ZeroStubAudit().audit_python_code(src_text)
        assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert stub_res.penalty_energy == 0.0

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {"filename": rel, "status": "added", "additions": len(src_text.splitlines()), "patch": "", "content": src_text}
        )

    telemetry_json = json.dumps({"damped_modes": 1, "max_mode_ratio": 12.5, "parameter_fraction_modified": 0.0002})
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1104,
        pr_title="feat(swarm): QR-SVD rank-selective LoRA singular-mode surgery (LASER) for sub-0.05% weight edits",
        pr_body="Implements NEURAL-CTRL-04 QR-SVD LoRA singular-mode surgery and deterministic scenario suite.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(swarm): QR-SVD rank-selective LoRA singular-mode surgery (LASER) for sub-0.05% weight edits"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
