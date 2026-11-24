# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-03: Ledoit-Wolf Marchenko-Pastur guard & AlphaEdit filter."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdRuntimeMLP, CrowdV1RuntimeConfig
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP
from hivemind.swarm.marchenko_pastur_covariance_guard import (
    DEFAULT_MAX_OPERATOR_NORM,
    MarchenkoPasturGuardConfig,
    apply_condition_bounded_alphaedit_to_moe_layers_inplace,
    audit_filter_stability,
    compute_condition_bounded_alphaedit_filter,
    compute_condition_bounded_alphaedit_filter_from_activations,
    compute_oracle_ledoit_wolf_moment,
    shrink_moment_oracle_oas,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_discriminative_alphaedit_filter,
    compute_swiglu_intermediate_states,
    second_moment_and_mean,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "marchenko_pastur_covariance_guard.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


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


def _build_oblique_marchenko_pastur_problem(
    *,
    d: int = 128,
    n_train: int = 128,
    n_holdout: int = 256,
    seed: int = 20260301,
) -> dict[str, torch.Tensor]:
    """Construct an N=d spiked specialist + obliquely correlated prose benchmark."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_spikes = 4
    spec_std = torch.cat(
        [torch.full((n_spikes,), 40.0), torch.full((d - n_spikes,), 0.15)]
    ).to(torch.float64)

    x_spec = torch.randn((n_train, d), generator=gen, dtype=torch.float64) * spec_std
    z_shared = torch.randn((n_train, n_spikes), generator=gen, dtype=torch.float64)

    x_prose = torch.randn((n_train, d), generator=gen, dtype=torch.float64) * 0.20
    x_prose[:, :n_spikes] = z_shared * 25.0
    x_prose[:, n_spikes:24] += z_shared.repeat(1, 5) * 1.8
    x_prose[:, 24:40] = torch.randn((n_train, 16), generator=gen, dtype=torch.float64) * 2.5

    x_hold_prose = torch.randn((n_holdout, d), generator=gen, dtype=torch.float64) * 0.20
    x_hold_prose[:, :n_spikes] = torch.randn((n_holdout, n_spikes), generator=gen, dtype=torch.float64) * 1.0
    x_hold_prose[:, n_spikes:24] = torch.randn((n_holdout, 20), generator=gen, dtype=torch.float64) * 2.5
    x_hold_prose[:, 24:40] = torch.randn((n_holdout, 16), generator=gen, dtype=torch.float64) * 2.5
    x_hold_spec = torch.randn((n_holdout, d), generator=gen, dtype=torch.float64) * spec_std

    return {
        "x_prose": x_prose,
        "x_spec": x_spec,
        "x_hold_prose": x_hold_prose,
        "x_hold_spec": x_hold_spec,
    }


# ---------------------------------------------------------------------------
# S1: Explicit N=d Marchenko-Pastur Oblique Blowup Reproduction & Monotonic Cure
# ---------------------------------------------------------------------------


def test_s1_marchenko_pastur_oblique_blowup_cured_and_monotonic_nll() -> None:
    """Reproduce N=d oblique blowup (||G_naive||_2 >> 2) and cure it with ||G_safe||_2 <= 1.05."""
    data = _build_oblique_marchenko_pastur_problem(d=128, n_train=128, n_holdout=256, seed=20260301)
    xp, xs = data["x_prose"], data["x_spec"]
    xp_hold, xs_hold = data["x_hold_prose"], data["x_hold_spec"]
    d, n = int(xp.shape[1]), int(xp.shape[0])

    mp_emp, _ = second_moment_and_mean(xp)
    ms_emp, _ = second_moment_and_mean(xs)

    g_naive, _ = compute_discriminative_alphaedit_filter(
        mp_emp, ms_emp, eta=4.0, tau=1.25, max_rank=64, reg_spec=0.10
    )
    op_naive = float(torch.linalg.matrix_norm(g_naive, ord=2).item())
    assert op_naive > 4.0

    naive_audit = audit_filter_stability(g_naive, xp_hold, xs_hold, max_operator_norm=DEFAULT_MAX_OPERATOR_NORM)
    assert naive_audit["operator_norm_ok"] is False
    assert naive_audit["prose_variance_non_amplified"] is False
    assert naive_audit["prose_energy_ratio"] > 1.05
    assert naive_audit["is_stable"] is False

    # Construct synthetic LM head + LoRA adapter where W_A reads specialist spikes & prose dims
    gen = torch.Generator(device="cpu").manual_seed(20260302)
    vocab = 32
    head = torch.randn((vocab, d), generator=gen, dtype=torch.float64) * 0.25
    targets_p = torch.randint(0, vocab, (xp_hold.shape[0],), generator=gen)
    targets_s = torch.randint(0, vocab, (xs_hold.shape[0],), generator=gen)
    h_base_p = 2.0 * head[targets_p] / head[targets_p].norm(dim=1, keepdim=True)
    h_base_s = 2.0 * head[targets_s] / head[targets_s].norm(dim=1, keepdim=True)

    w_a = torch.zeros((16, d), dtype=torch.float64)
    w_a[:4, :4] = torch.eye(4, dtype=torch.float64) * 0.45
    w_a[4:, 24:40] = torch.randn((12, 16), generator=gen, dtype=torch.float64) * 0.15
    w_b = torch.randn((d, 16), generator=gen, dtype=torch.float64) * 0.25

    def _eval_nll(g_mat: torch.Tensor) -> tuple[float, float]:
        w_eff = w_a @ g_mat
        logits_p = (h_base_p + (xp_hold @ w_eff.T) @ w_b.T) @ head.T
        logits_s = (h_base_s + (xs_hold @ w_eff.T) @ w_b.T) @ head.T
        return (
            float(F.cross_entropy(logits_p, targets_p).item()),
            float(F.cross_entropy(logits_s, targets_s).item()),
        )

    tuned_p_nll, tuned_s_nll = _eval_nll(torch.eye(d, dtype=torch.float64))
    naive_p_nll, _ = _eval_nll(g_naive)
    assert naive_p_nll > tuned_p_nll * 2.0

    prose_nlls: list[float] = []
    for eta_val in (0.0, 0.5, 1.5, 4.0, 12.0):
        g_safe, stats = compute_condition_bounded_alphaedit_filter(
            mp_emp,
            ms_emp,
            n_prose=n,
            n_spec=n,
            eta=eta_val,
            tau=1.25,
            max_rank=64,
            reg_spec=0.10,
            max_operator_norm=DEFAULT_MAX_OPERATOR_NORM,
        )
        assert stats["operator_norm"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-12
        assert stats["bounded"] is True
        if eta_val > 0.0:
            assert stats["clamped"] is True
            assert stats["fallback_mode"] == "spectral_damped"
            assert 0.0 < stats["damping_alpha"] < 1.0
        p_nll, s_nll = _eval_nll(g_safe)
        prose_nlls.append(p_nll)
        assert s_nll <= tuned_s_nll + 1e-3

        safe_audit = audit_filter_stability(
            g_safe, xp_hold, xs_hold, max_operator_norm=DEFAULT_MAX_OPERATOR_NORM
        )
        assert safe_audit["is_stable"] is True
        assert safe_audit["operator_norm_ok"] is True
        assert safe_audit["prose_variance_non_amplified"] is True
        assert safe_audit["specialist_preserved"] is True

    # Monotonic holdout prose NLL reduction as eta increases
    assert all(prose_nlls[i] > prose_nlls[i + 1] for i in range(len(prose_nlls) - 1))

    # Tight max_operator_norm=1.0 triggers clean euclidean_fallback (alpha == 0.0)
    g_euc, euc_stats = compute_condition_bounded_alphaedit_filter_from_activations(
        xp, xs, eta=4.0, tau=1.25, max_rank=64, reg_spec=0.10, max_operator_norm=1.0
    )
    assert euc_stats["operator_norm"] <= 1.0 + 1e-12
    assert euc_stats["fallback_mode"] == "euclidean_fallback"
    assert euc_stats["damping_alpha"] == 0.0
    assert float(torch.linalg.matrix_norm(g_euc, ord=2).item()) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# S2: Asymptotic Consistency (rho -> 0 as N >> d, Recovering Unshrunk AlphaEdit)
# ---------------------------------------------------------------------------


def test_s2_asymptotic_consistency_recovers_exact_unshrunk_alphaedit() -> None:
    """Verify rho -> 0 as N >> d and filter converges to exact unshrunk AlphaEdit."""
    d = 16
    q0 = _make_orthogonal_basis(d, seed=20260401)
    p_var = torch.cat([torch.tensor([9.0, 6.0, 4.0]), torch.full((d - 3,), 0.2)], dim=0).to(torch.float64)
    s_var = torch.cat([torch.full((3,), 0.15), torch.linspace(1.0, 5.0, d - 3, dtype=torch.float64)], dim=0)

    rhos: list[float] = []
    for idx, n_samples in enumerate((128, 1024, 16384)):
        gen = torch.Generator(device="cpu").manual_seed(20260410 + idx)
        xp = (torch.randn((n_samples, d), generator=gen, dtype=torch.float64) * p_var.sqrt()) @ q0.T
        xs = (torch.randn((n_samples, d), generator=gen, dtype=torch.float64) * s_var.sqrt()) @ q0.T
        m_p_shr, diag_p = compute_oracle_ledoit_wolf_moment(xp)
        m_s_shr, diag_s = compute_oracle_ledoit_wolf_moment(xs)
        assert diag_p["gamma"] == pytest.approx(d / n_samples)
        assert diag_s["gamma"] == pytest.approx(d / n_samples)
        assert 0.0 <= diag_p["rho"] <= 1.0
        assert 0.0 <= diag_s["rho"] <= 1.0
        rhos.append(diag_p["rho"])
        if n_samples == 16384:
            mp_emp, _ = second_moment_and_mean(xp)
            ms_emp, _ = second_moment_and_mean(xs)
            assert diag_p["rho"] < 0.01
            assert diag_s["rho"] < 0.01
            rel_diff_p = float((m_p_shr - mp_emp).norm().item() / mp_emp.norm().item())
            assert rel_diff_p < 0.01

    assert rhos[0] > rhos[1] > rhos[2]

    # Compare against exact unshrunk AlphaEdit on well-conditioned orthogonal moments with large N
    gen_asym = torch.Generator(device="cpu").manual_seed(20260420)
    n_big = 20000
    xp_big = (torch.randn((n_big, d), generator=gen_asym, dtype=torch.float64) * p_var.sqrt()) @ q0.T
    xs_big = (torch.randn((n_big, d), generator=gen_asym, dtype=torch.float64) * s_var.sqrt()) @ q0.T
    mp_big, _ = second_moment_and_mean(xp_big)
    ms_big, _ = second_moment_and_mean(xs_big)

    g_unshrunk, _ = compute_discriminative_alphaedit_filter(
        mp_big, ms_big, eta=2.5, tau=1.25, max_rank=8, reg_spec=0.10
    )
    g_bounded, st_bounded = compute_condition_bounded_alphaedit_filter(
        mp_big,
        ms_big,
        n_prose=n_big,
        n_spec=n_big,
        eta=2.5,
        tau=1.25,
        max_rank=8,
        reg_spec=0.10,
        max_operator_norm=DEFAULT_MAX_OPERATOR_NORM,
    )
    assert st_bounded["clamped"] is False
    assert st_bounded["fallback_mode"] == "oblique_shrunk"
    assert torch.allclose(g_bounded, g_unshrunk, atol=2e-3)

    # With shrink_moments=False, exact unshrunk AlphaEdit is recovered bitwise-close
    g_exact, _ = compute_condition_bounded_alphaedit_filter(
        mp_big,
        ms_big,
        n_prose=n_big,
        n_spec=n_big,
        eta=2.5,
        tau=1.25,
        max_rank=8,
        reg_spec=0.10,
        shrink_moments=False,
    )
    assert torch.allclose(g_exact, g_unshrunk, atol=1e-11)

    # Verify center=True option computes centered covariance shrinkage cleanly
    shifted = xp_big + 5.0
    m_centered, diag_c = compute_oracle_ledoit_wolf_moment(shifted, center=True)
    x_c = shifted - shifted.mean(dim=0, keepdim=True)
    cov_emp = (x_c.T @ x_c) / float(n_big)
    assert diag_c["rho"] < 0.01
    assert torch.allclose(m_centered, cov_emp, atol=0.05)


# ---------------------------------------------------------------------------
# S3: Extreme Low-Sample Regime N=8 << d=256 & Heavy-Tailed/N=1 Safety
# ---------------------------------------------------------------------------


def test_s3_extreme_low_sample_regime_shrinks_safely_without_nan() -> None:
    """Verify N=8 << d=256 (gamma=32) shrinks safely toward isotropic target with zero NaN/Inf."""
    gen = torch.Generator(device="cpu").manual_seed(20260501)
    d, n = 256, 8

    x_iso_low = torch.randn((n, d), generator=gen, dtype=torch.float64)
    m_shrunk, diag = compute_oracle_ledoit_wolf_moment(x_iso_low)
    assert diag["gamma"] == pytest.approx(32.0)
    assert diag["aspect_ratio"] == pytest.approx(32.0)
    assert 0.90 <= diag["rho"] <= 1.0
    assert diag["min_eig_emp"] < 1e-10
    assert diag["min_eig_shrunk"] > 0.5
    assert diag["condition_number_emp"] > 1e10
    assert diag["condition_number_shrunk"] < 5.0
    assert bool(torch.isfinite(m_shrunk).all().item())

    # Exact linear shrinkage identity check
    m_emp, _ = second_moment_and_mean(x_iso_low)
    mu_iso = float(torch.trace(m_emp).item()) / float(d)
    expected = (1.0 - diag["rho"]) * m_emp + diag["rho"] * mu_iso * torch.eye(d, dtype=torch.float64)
    assert torch.allclose(m_shrunk, expected, atol=1e-12)

    # Collinear spiked low-sample regime + condition-bounded AlphaEdit filter
    v_sig = torch.randn((1, d), generator=gen, dtype=torch.float64)
    x_spec_low = torch.randn((n, 1), generator=gen, dtype=torch.float64) @ v_sig + 0.05 * torch.randn(
        (n, d), generator=gen, dtype=torch.float64
    )
    x_prose_low = torch.randn((n, d), generator=gen, dtype=torch.float64) * 1.5
    g_low, st_low = compute_condition_bounded_alphaedit_filter_from_activations(
        x_prose_low, x_spec_low, eta=3.0, tau=1.25, max_rank=64, max_operator_norm=DEFAULT_MAX_OPERATOR_NORM
    )
    assert bool(torch.isfinite(g_low).all().item())
    assert st_low["operator_norm"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-12
    assert st_low["bounded"] is True
    assert st_low["rho_spec"] > 0.20

    # Heavy-tailed outlier activations trigger distribution-free rho_lw > rho_oas
    z_ht = torch.randn((64, 64), generator=gen, dtype=torch.float64)
    scale_rows = torch.exp(torch.randn((64, 1), generator=gen, dtype=torch.float64) * 1.4)
    _, ht_diag = compute_oracle_ledoit_wolf_moment(z_ht * scale_rows)
    assert ht_diag["rho_lw"] > ht_diag["rho_oas"]
    assert ht_diag["rho"] == pytest.approx(ht_diag["rho_lw"])

    # N=1 single sample, d=1 scalar, exact identity, and all-zero degenerate inputs remain finite
    m_n1, d_n1 = compute_oracle_ledoit_wolf_moment(torch.randn((1, 32), generator=gen))
    assert d_n1["rho"] == 1.0
    assert d_n1["condition_number"] == pytest.approx(1.0)
    assert bool(torch.isfinite(m_n1).all().item())

    m_d1, d_d1 = compute_oracle_ledoit_wolf_moment(torch.randn((8, 1), generator=gen))
    assert d_d1["rho"] == 0.0
    assert m_d1.shape == (1, 1)

    m_eye_shr, d_eye = compute_oracle_ledoit_wolf_moment(torch.eye(8, dtype=torch.float64))
    assert d_eye["rho"] == 1.0
    assert torch.allclose(m_eye_shr, torch.eye(8, dtype=torch.float64) / 8.0, atol=1e-12)

    m_zero, d_zero = compute_oracle_ledoit_wolf_moment(torch.zeros((8, 16), dtype=torch.float64))
    assert d_zero["condition_number"] == 1.0
    assert torch.equal(m_zero, torch.zeros((16, 16), dtype=torch.float64))

    m_oas_d1, oas_d1 = shrink_moment_oracle_oas(torch.tensor([[4.0]]), 5)
    assert oas_d1["rho"] == 0.0
    assert float(m_oas_d1.item()) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# S4: Multi-Layer Sequential Application Across 8 SwiGLU LoRA Layers
# ---------------------------------------------------------------------------


def test_s4_eight_layer_swiglu_lora_sequential_zero_exponential_growth() -> None:
    """Prove zero exponential norm growth across 8 SwiGLU LoRA layers vs naive AlphaEdit blowup."""
    num_layers = 8
    d = 32
    inter = 64
    n_train = 32  # Exact N == d Marchenko-Pastur regime per layer
    n_hold = 64
    gen = torch.Generator(device="cpu").manual_seed(20260601)

    def _build_layer(layer_seed: int) -> CrowdLoRAMoEMLP:
        l_gen = torch.Generator(device="cpu").manual_seed(layer_seed)
        cfg = CrowdV1RuntimeConfig(hidden_size=d, intermediate_size=inter)
        dense = CrowdRuntimeMLP(cfg)
        with torch.no_grad():
            for p in dense.parameters():
                p.copy_(torch.randn(p.shape, generator=l_gen, dtype=torch.float32) * 0.05)
        moe = CrowdLoRAMoEMLP(
            dense,
            num_experts=4,
            top_k=2,
            rank=4,
            alpha=16.0,
            routing_mode="sigma_topk",
            seed=layer_seed + 100,
        )
        with torch.no_grad():
            moe.router.weight.copy_(
                torch.randn(moe.router.weight.shape, generator=l_gen, dtype=torch.float32) * 0.35
            )
            for e_key in moe.experts:
                exp = moe.experts[e_key]
                exp.w1_A[:, :4].copy_(torch.randn((4, 4), generator=l_gen) * 0.45)
                exp.w2_A[:, :4].copy_(torch.randn((4, 4), generator=l_gen) * 0.45)
                exp.w1_A[:, 4:].copy_(torch.randn((4, d - 4), generator=l_gen) * 0.08)
                exp.w2_A[:, 4:].copy_(torch.randn((4, d - 4), generator=l_gen) * 0.08)
                exp.w1_B.copy_(torch.randn(exp.w1_B.shape, generator=l_gen) * 0.14)
                exp.w2_B.copy_(torch.randn(exp.w2_B.shape, generator=l_gen) * 0.14)
                exp.w3_B.copy_(torch.randn(exp.w3_B.shape, generator=l_gen) * 0.14)
        return moe

    layers_safe = [_build_layer(20260610 + i * 17) for i in range(num_layers)]
    layers_naive = copy.deepcopy(layers_safe)
    layer_norms = [nn.LayerNorm(d) for _ in range(num_layers)]

    # Construct N == d training and holdout oblique MP activations
    spec_std = torch.cat([torch.full((4,), 25.0), torch.full((d - 4,), 0.10)]).to(torch.float32)
    x_spec = torch.randn((n_train, d), generator=gen, dtype=torch.float32) * spec_std
    z_tr = torch.randn((n_train, 4), generator=gen, dtype=torch.float32)
    x_prose = torch.randn((n_train, d), generator=gen, dtype=torch.float32) * 0.10
    x_prose[:, :4] = z_tr * 15.0
    x_prose[:, 4:16] += z_tr.repeat(1, 3) * 1.2

    x_hold_prose = torch.randn((n_hold, d), generator=gen, dtype=torch.float32) * 0.10
    x_hold_prose[:, :4] = torch.randn((n_hold, 4), generator=gen, dtype=torch.float32) * 0.8
    x_hold_prose[:, 4:16] = torch.randn((n_hold, 12), generator=gen, dtype=torch.float32) * 1.5

    # 1. Apply condition-bounded AlphaEdit sequentially across all 8 SwiGLU LoRA layers
    safe_report = apply_condition_bounded_alphaedit_to_moe_layers_inplace(
        layers_safe,
        x_prose,
        x_spec,
        eta_x=4.0,
        eta_act=2.0,
        tau=1.25,
        max_rank_x=24,
        max_rank_act=32,
        reg_spec=0.05,
        max_operator_norm=DEFAULT_MAX_OPERATOR_NORM,
        layer_norms=layer_norms,
    )
    assert safe_report["num_layers"] == num_layers
    assert safe_report["bounded_across_depth"] is True
    assert safe_report["max_operator_norm_seen"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-12
    assert safe_report["max_weight_norm_ratio"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-3

    # 2. Apply naive unshrunk AlphaEdit sequentially across the 8 identical layers
    cur_p_naive = x_prose.clone()
    cur_s_naive = x_spec.clone()
    naive_ops: list[float] = []
    naive_max_w_ratio = 1.0
    with torch.no_grad():
        for ln_mod, layer in zip(layer_norms, layers_naive):
            mi_p = ln_mod(cur_p_naive)
            mi_s = ln_mod(cur_s_naive)
            mp_x, _ = second_moment_and_mean(mi_p)
            ms_x, _ = second_moment_and_mean(mi_s)
            g_x, st_x = compute_discriminative_alphaedit_filter(
                mp_x, ms_x, eta=4.0, tau=1.25, max_rank=24, reg_spec=0.05
            )
            naive_ops.append(float(torch.linalg.matrix_norm(g_x, ord=2).item()))
            if st_x["active_modes"] > 0:
                gx32 = g_x.to(torch.float32)
                for e_key in layer.experts:
                    exp = layer.experts[e_key]
                    before_n = max(float(exp.w1_A.norm().item()), 1e-12)
                    exp.w1_A.copy_(exp.w1_A @ gx32)
                    exp.w2_A.copy_(exp.w2_A @ gx32)
                    naive_max_w_ratio = max(naive_max_w_ratio, float(exp.w1_A.norm().item()) / before_n)
            targets = sorted(int(k) for k in layer.experts.keys())
            act_p = compute_swiglu_intermediate_states(layer, mi_p, targets)
            act_s = compute_swiglu_intermediate_states(layer, mi_s, targets)
            mp_a, _ = second_moment_and_mean(act_p)
            ms_a, _ = second_moment_and_mean(act_s)
            g_a, st_a = compute_discriminative_alphaedit_filter(
                mp_a, ms_a, eta=2.0, tau=1.25, max_rank=32, reg_spec=0.05
            )
            naive_ops.append(float(torch.linalg.matrix_norm(g_a, ord=2).item()))
            if st_a["active_modes"] > 0:
                ga32 = g_a.to(torch.float32)
                for e_key in layer.experts:
                    exp = layer.experts[e_key]
                    exp.w3_A.copy_(exp.w3_A @ ga32)
            cur_p_naive = cur_p_naive + layer(mi_p)
            cur_s_naive = cur_s_naive + layer(mi_s)

    assert max(naive_ops) > 3.0
    assert naive_max_w_ratio > 3.0

    # Measure out-of-sample holdout prose activation norm across depth 0..7
    def _trace_holdout_prose_rms(layers: list[CrowdLoRAMoEMLP], x0: torch.Tensor) -> list[float]:
        init_rms = float(x0.pow(2).mean().sqrt().item())
        cur = x0.clone()
        ratios: list[float] = []
        with torch.no_grad():
            for ln_mod, lyr in zip(layer_norms, layers):
                cur = cur + lyr(ln_mod(cur))
                ratios.append(float(cur.pow(2).mean().sqrt().item()) / init_rms)
        return ratios

    safe_ratios = _trace_holdout_prose_rms(layers_safe, x_hold_prose)
    naive_ratios = _trace_holdout_prose_rms(layers_naive, x_hold_prose)
    assert max(safe_ratios) < 5.0
    assert max(naive_ratios) > max(safe_ratios) * 10.0
    assert all(
        lyr["operator_norm_x"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-12
        and lyr["operator_norm_act"] <= DEFAULT_MAX_OPERATOR_NORM + 1e-12
        for lyr in safe_report["layers"]
    )

    # Also test explicit expert_indices, residual=False, and eta_x=0/eta_act=0 paths
    layers_subset = [_build_layer(20260701)]
    sub_rep = apply_condition_bounded_alphaedit_to_moe_layers_inplace(
        layers_subset,
        x_prose,
        x_spec,
        eta_x=0.0,
        eta_act=0.0,
        expert_indices=[1, 2],
        residual=False,
    )
    assert sub_rep["layers"][0]["targets"] == [1, 2]
    assert sub_rep["layers"][0]["operator_norm_x"] == 1.0
    assert sub_rep["layers"][0]["operator_norm_act"] == 1.0


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_edge_cases_and_security_compliance_gates() -> None:
    """Validate error handling, boundary inputs, and HiveMind security/stub compliance gates."""
    eye4 = torch.eye(4, dtype=torch.float64)

    # 1. MarchenkoPasturGuardConfig validation
    cfg = MarchenkoPasturGuardConfig()
    assert cfg.max_operator_norm == DEFAULT_MAX_OPERATOR_NORM
    with pytest.raises(TypeError, match="real number"):
        MarchenkoPasturGuardConfig(eta=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        MarchenkoPasturGuardConfig(eta=-0.5)
    with pytest.raises(ValueError, match="> 0.0"):
        MarchenkoPasturGuardConfig(max_operator_norm=0.0)
    with pytest.raises(ValueError, match=">= 1.0"):
        MarchenkoPasturGuardConfig(max_operator_norm=0.95)
    with pytest.raises(TypeError, match="bool"):
        MarchenkoPasturGuardConfig(shrink_moments="yes")  # type: ignore[arg-type]

    # 2. shrink_moment_oracle_oas and compute_oracle_ledoit_wolf_moment validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_oracle_ledoit_wolf_moment([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D tensor"):
        compute_oracle_ledoit_wolf_moment(torch.zeros((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        compute_oracle_ledoit_wolf_moment(torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="square"):
        shrink_moment_oracle_oas(torch.ones((3, 4)), 10)
    with pytest.raises(TypeError, match="int"):
        shrink_moment_oracle_oas(eye4, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        shrink_moment_oracle_oas(eye4, 0)

    # 3. compute_condition_bounded_alphaedit_filter & _from_activations validation & noop edges
    with pytest.raises(ValueError, match="share shape"):
        compute_condition_bounded_alphaedit_filter(eye4, torch.eye(5, dtype=torch.float64))
    with pytest.raises(ValueError, match=">= 1.0"):
        compute_condition_bounded_alphaedit_filter(eye4, eye4, max_operator_norm=0.9)
    with pytest.raises(ValueError, match=">= 1"):
        compute_condition_bounded_alphaedit_filter(eye4, eye4, n_prose=0, n_spec=4)
    with pytest.raises(ValueError, match=">= 0"):
        compute_condition_bounded_alphaedit_filter(eye4, eye4, max_rank=-1)
    with pytest.raises(ValueError, match="share feature width"):
        compute_condition_bounded_alphaedit_filter_from_activations(
            torch.ones((4, 4)), torch.ones((4, 5))
        )

    g_eta0, st_eta0 = compute_condition_bounded_alphaedit_filter(eye4, eye4, eta=0.0)
    assert torch.equal(g_eta0, eye4)
    assert st_eta0["fallback_mode"] == "identity"
    assert st_eta0["operator_norm"] == 1.0

    g_tau_hi, st_tau_hi = compute_condition_bounded_alphaedit_filter(eye4, eye4, eta=2.0, tau=100.0)
    assert torch.equal(g_tau_hi, eye4)
    assert st_tau_hi["active_modes"] == 0
    assert st_tau_hi["fallback_mode"] == "identity"

    # 4. audit_filter_stability validation and zero holdout rows
    with pytest.raises(ValueError, match="square"):
        audit_filter_stability(torch.ones((3, 4)), torch.ones((2, 4)), torch.ones((2, 4)))
    with pytest.raises(ValueError, match="width 4"):
        audit_filter_stability(eye4, torch.ones((2, 5)), torch.ones((2, 4)))
    with pytest.raises(ValueError, match=">= 1.0"):
        audit_filter_stability(eye4, torch.ones((2, 4)), torch.ones((2, 4)), max_operator_norm=0.9)

    zero_audit = audit_filter_stability(eye4, torch.zeros((4, 4)), torch.zeros((4, 4)))
    assert zero_audit["is_stable"] is True
    assert zero_audit["prose_max_token_amplification"] == 1.0
    assert zero_audit["specialist_rel_distortion"] == 0.0

    # 5. apply_condition_bounded_alphaedit_to_moe_layers_inplace validation
    with pytest.raises(ValueError, match="non-empty"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace([], torch.ones((2, 4)), torch.ones((2, 4)))
    with pytest.raises(TypeError, match="nn.Module"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            ["bad"], torch.ones((2, 4)), torch.ones((2, 4))  # type: ignore[list-item]
        )
    with pytest.raises(ValueError, match="experts"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            [nn.Linear(4, 4)], torch.ones((2, 4)), torch.ones((2, 4))
        )
    moe_tiny = CrowdLoRAMoEMLP(
        CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16)),
        num_experts=2,
        top_k=1,
        rank=2,
    )
    with pytest.raises(ValueError, match="share feature width"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            [moe_tiny], torch.ones((4, 8)), torch.ones((4, 6))
        )
    with pytest.raises(ValueError, match=">= 1.0"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            [moe_tiny], torch.ones((4, 8)), torch.ones((4, 8)), max_operator_norm=0.8
        )
    with pytest.raises(ValueError, match="layer_norms must match"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            [moe_tiny], torch.ones((4, 8)), torch.ones((4, 8)), layer_norms=[]
        )
    with pytest.raises(TypeError, match="nn.Module"):
        apply_condition_bounded_alphaedit_to_moe_layers_inplace(
            [moe_tiny], torch.ones((4, 8)), torch.ones((4, 8)), layer_norms=["bad"]  # type: ignore[list-item]
        )

    # 6. AGPL-3.0 header, ZeroStubAudit (if available), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn03", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn03", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/marchenko_pastur_covariance_guard.py", TARGET_MODULE_PATH),
        ("tests/test_marchenko_pastur_covariance_guard_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        if ZeroStubAudit is not None:
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
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    telemetry_json = json.dumps(
        {"nll_mean": 1.15, "operator_norm": 1.04, "condition_number": 3.42}
    )
    assert (
        gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}")
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1110,
        pr_title="feat(swarm): Ledoit-Wolf Marchenko-Pastur covariance shrinkage and condition-bounded AlphaEdit filter",
        pr_body="Implements NEURAL-CTRL-DYN-03 Marchenko-Pastur covariance guard and deterministic scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): Ledoit-Wolf Marchenko-Pastur covariance shrinkage and condition-bounded AlphaEdit filter"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
