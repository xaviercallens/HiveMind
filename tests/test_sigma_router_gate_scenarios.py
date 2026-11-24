# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-02: contrastive sigma_topk multi-gate controller with OOF shutoff."""

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
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP
from hivemind.swarm.sigma_router_gate_control import (
    DEFAULT_OMEGA_OTHER,
    DEFAULT_OMEGA_OWN,
    DEFAULT_OMEGA_SPEC,
    DEFAULT_RIDGE_SCALE,
    GENERAL_EXPERT_INDEX,
    SigmaRouterControlConfig,
    apply_sigma_router_prose_control_inplace,
    evaluate_sigma_topk_routing_gates,
    fit_per_domain_orthogonal_gate_vectors,
    fit_uncentered_prose_probe_with_oof,
    interpolate_router_weights,
    second_moment_and_mean,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "sigma_router_gate_control.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_subspace_batch(
    n_samples: int,
    hidden_size: int,
    carrier_dim: int,
    *,
    carrier_strength: float = 3.0,
    noise_slice: slice,
    noise_scale: float = 0.25,
    seed: int,
) -> torch.Tensor:
    """Generate deterministic subspace activations with a consistent carrier component."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.zeros((n_samples, hidden_size), dtype=torch.float32)
    x[:, noise_slice] = noise_scale * torch.randn((n_samples, noise_slice.stop - noise_slice.start), generator=gen)
    x[:, carrier_dim] = carrier_strength
    return x


def _build_sigma_moe_fixture(
    *,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_experts: int = 6,
    top_k: int = 2,
    seed: int = 20261123,
) -> CrowdLoRAMoEMLP:
    """Build a deterministic CPU ``CrowdLoRAMoEMLP(routing_mode='sigma_topk')`` fixture."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    dense = CrowdRuntimeMLP(cfg)
    with torch.no_grad():
        for p in dense.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.15)
    moe = CrowdLoRAMoEMLP(
        dense,
        num_experts=num_experts,
        top_k=top_k,
        rank=4,
        alpha=8.0,
        routing_mode="sigma_topk",
        resident_expert_indices=tuple(range(num_experts)),
        seed=seed,
    )
    return moe


# ---------------------------------------------------------------------------
# S1: sigma_topk lambda=0.50 Gate-Opening Regression vs Contrastive Control
# ---------------------------------------------------------------------------


def test_s1_sigma_topk_lambda_050_gate_opening_regression_vs_contrastive_control() -> None:
    """Reproduce the sigma_topk lambda=0.50 interpolation trap and defeat it via contrastive suppression."""
    hidden_size = 32
    num_experts = 6
    moe = _build_sigma_moe_fixture(hidden_size=hidden_size, intermediate_size=64, num_experts=num_experts, top_k=2)

    # Subspaces: dim 0 (noise 0..7) = prose; dims 8, 12, 16, 20, 24 = specialist experts 1..5
    carrier_dims = {1: 8, 2: 12, 3: 16, 4: 20, 5: 24}
    prose_f0 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=101)
    prose_f1 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=102)
    prose_eval = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=103)

    spec_f0_parts: list[torch.Tensor] = []
    spec_f1_parts: list[torch.Tensor] = []
    dom_eval: dict[int, torch.Tensor] = {}
    for e_idx, c_dim in carrier_dims.items():
        dom_f0 = _make_subspace_batch(32, hidden_size, c_dim, carrier_strength=2.5, noise_slice=slice(8, 32), seed=200 + e_idx)
        dom_f1 = _make_subspace_batch(32, hidden_size, c_dim, carrier_strength=2.5, noise_slice=slice(8, 32), seed=300 + e_idx)
        spec_f0_parts.append(dom_f0)
        spec_f1_parts.append(dom_f1)
        dom_eval[e_idx] = _make_subspace_batch(
            48, hidden_size, c_dim, carrier_strength=2.5, noise_slice=slice(8, 32), seed=400 + e_idx
        )
    spec_f0 = torch.cat(spec_f0_parts, dim=0)
    spec_f1 = torch.cat(spec_f1_parts, dim=0)

    # Start router at W_start = 0 (sigmoid(0) = 0.50 across all experts)
    start_router = nn.Linear(hidden_size, num_experts, bias=False)
    nn.init.zeros_(start_router.weight)

    # Tuned C2 router:
    # - Expert 0 helps prose (+1.35 on dim 0 -> sigmoid ~ 0.967)
    # - Expert 1 was strongly closed on prose (-1.32 on dim 0 -> sigmoid(-3.3) = 0.036 < 0.05)
    # - Experts 2..5 leak positively on prose (+0.28 on dim 0 -> sigmoid(+0.7) = 0.668)
    # - Every specialist e in 1..5 strongly activates on its own domain (+1.45 on carrier_dims[e] -> sigmoid > 0.97)
    gen = torch.Generator(device="cpu").manual_seed(555)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.35
        moe.router.weight[1, 0] = -1.32
        for e_idx, c_dim in carrier_dims.items():
            moe.router.weight[e_idx, c_dim] = 1.45
            if e_idx >= 2:
                moe.router.weight[e_idx, 0] = 0.28
        # Give off-domain specialists non-zero LoRA deltas so open gates distort prose output
        for e_idx in range(1, num_experts):
            exp = moe.experts[str(e_idx)]
            for attr in ("w1_B", "w2_B", "w3_B"):
                param = getattr(exp, attr)
                param.copy_(torch.randn(param.shape, generator=gen) * 0.35)

    # Verify start router (lambda = 0.0) opens every gate to exactly 0.50
    start_metrics = evaluate_sigma_topk_routing_gates(start_router, prose_eval, dom_eval, top_k=2)
    assert all(abs(g - 0.50) < 1e-6 for g in start_metrics["prose_gate_by_expert"].values())

    # Verify tuned router baseline: expert_1 closed (<0.05) on prose, specialists > 0.95, but leaky experts 2..5 open
    tuned_metrics = evaluate_sigma_topk_routing_gates(moe.router, prose_eval, dom_eval, top_k=2)
    assert tuned_metrics["prose_gate_by_expert"][1] < 0.05
    assert tuned_metrics["prose_offdomain_max_gate"] > 0.60
    assert tuned_metrics["domain_min_own_gate"] > 0.95

    # Reproduce the non-monotonic sigma_topk interpolation regression at lambda = 0.50:
    # Shrinking W_tuned halfway toward 0 REOPENS expert_1 from <0.05 to >0.15 AND degrades specialist gates <0.90!
    interp_050 = interpolate_router_weights(start_router, moe.router, 0.50)
    w50_metrics = evaluate_sigma_topk_routing_gates(interp_050, prose_eval, dom_eval, top_k=2)
    assert w50_metrics["prose_gate_by_expert"][1] > 0.15
    assert w50_metrics["prose_gate_by_expert"][1] > 3.0 * tuned_metrics["prose_gate_by_expert"][1]
    assert w50_metrics["prose_offdomain_max_gate"] > 0.55
    assert w50_metrics["domain_min_own_gate"] < 0.90

    # Solve contrastive probe with OOF reliability gating and apply negative prose logit shift
    u_prose, probe_stats = fit_uncentered_prose_probe_with_oof(
        prose_f0,
        prose_f1,
        spec_f0,
        spec_f1,
        omega_spec=DEFAULT_OMEGA_SPEC,
        ridge_scale=DEFAULT_RIDGE_SCALE,
    )
    assert probe_stats["rel_weight"] > 0.95
    assert probe_stats["oof_snr"] > 2.0

    with torch.no_grad():
        base_prose_out = moe.base_mlp(prose_eval)
        w50_router_saved = moe.router.weight.detach().clone()
        moe.router.weight.copy_(interp_050.weight)
        w50_prose_drift = float((moe(prose_eval) - base_prose_out).pow(2).mean().item())
        moe.router.weight.copy_(w50_router_saved)
        tuned_prose_drift = float((moe(prose_eval) - base_prose_out).pow(2).mean().item())

    edit_meta = apply_sigma_router_prose_control_inplace(
        moe.router,
        u_prose,
        rel_weight=probe_stats["rel_weight"],
        beta_suppress=4.2,
        beta_sink=0.0,
        expert0_helps_prose=True,
    )
    assert edit_meta["applied"] is True
    assert edit_meta["suppressed_experts"] == [1, 2, 3, 4, 5]
    assert edit_meta["protected_experts"] == [0]

    ctrl_metrics = evaluate_sigma_topk_routing_gates(moe.router, prose_eval, dom_eval, top_k=2)
    assert ctrl_metrics["prose_offdomain_max_gate"] < 0.05
    assert ctrl_metrics["prose_general_gate"] > 0.90
    assert ctrl_metrics["domain_min_own_gate"] > 0.95
    assert all(hit == 1.0 for hit in ctrl_metrics["domain_own_topk_hit_rate"].values())

    with torch.no_grad():
        ctrl_prose_drift = float((moe(prose_eval) - base_prose_out).pow(2).mean().item())
    assert ctrl_prose_drift < 0.02 * tuned_prose_drift
    assert ctrl_prose_drift < 0.02 * w50_prose_drift


# ---------------------------------------------------------------------------
# S2: 2-Fold OOF Shutoff Safety on Pure Gaussian Noise & Identical Distributions
# ---------------------------------------------------------------------------


def test_s2_oof_shutoff_safety_on_pure_gaussian_noise() -> None:
    """Ensure 2-fold OOF shuts off cleanly (rel_weight == 0.0) with zero router mutation on noise."""
    for seed_idx, (n_rows, feat_dim) in enumerate(((40, 16), (80, 32), (120, 64))):
        gen = torch.Generator(device="cpu").manual_seed(7100 + seed_idx)
        p0 = torch.randn((n_rows, feat_dim), generator=gen)
        p1 = torch.randn((n_rows, feat_dim), generator=gen)
        s0 = torch.randn((n_rows, feat_dim), generator=gen)
        s1 = torch.randn((n_rows, feat_dim), generator=gen)

        u_noise, stats_noise = fit_uncentered_prose_probe_with_oof(p0, p1, s0, s1)
        assert stats_noise["rel_weight"] == 0.0

        router = nn.Linear(feat_dim, 6, bias=False)
        with torch.no_grad():
            router.weight.copy_(torch.randn(router.weight.shape, generator=gen))
        w_before = router.weight.detach().clone()

        edit_meta = apply_sigma_router_prose_control_inplace(
            router,
            u_noise,
            rel_weight=stats_noise["rel_weight"],
            beta_suppress=3.5,
            beta_sink=2.0,
            sink_experts=[0],
            expert0_helps_prose=False,
        )
        assert edit_meta["applied"] is False
        assert edit_meta["weight_delta_l2"] == 0.0
        assert torch.equal(router.weight, w_before)

    # Identical shifted non-zero-mean Gaussian (shared token bias across prose & spec) must also shut off
    gen_shift = torch.Generator(device="cpu").manual_seed(7999)
    shared_mu = torch.randn(32, generator=gen_shift) * 2.0
    sp0 = torch.randn((80, 32), generator=gen_shift) + shared_mu
    sp1 = torch.randn((80, 32), generator=gen_shift) + shared_mu
    ss0 = torch.randn((80, 32), generator=gen_shift) + shared_mu
    ss1 = torch.randn((80, 32), generator=gen_shift) + shared_mu
    _, stats_shared = fit_uncentered_prose_probe_with_oof(sp0, sp1, ss0, ss1)
    assert stats_shared["rel_weight"] == 0.0

    # Exact zero-mean folds hit denom <= 1e-6 branch cleanly and return zero direction
    z_half = torch.ones((16, 8), dtype=torch.float64)
    z_exact = torch.cat([z_half, -z_half], dim=0)
    u_zero, stats_zero = fit_uncentered_prose_probe_with_oof(z_exact, z_exact, z_half, -z_half)
    assert torch.equal(u_zero, torch.zeros(8, dtype=torch.float64))
    assert stats_zero["rel_weight"] == 0.0


# ---------------------------------------------------------------------------
# S3: Per-Domain Orthogonal Multi-Gate Zero Cross-Talk Across 6 Domains
# ---------------------------------------------------------------------------


def test_s3_per_domain_orthogonal_multigate_zero_crosstalk_six_domains() -> None:
    """Verify per-domain u_{0 perp e} gives zero cross-talk across code, math, exoplanet, stellar, cyber, prose."""
    hidden_size = 48
    num_experts = 8  # 0=prose, 1=code, 2=math, 3..4=unscored/fallback, 5=exoplanet, 6=stellar, 7=cyber
    domain_specs = {
        1: ("code", 8),
        2: ("math", 14),
        5: ("exoplanet", 20),
        6: ("stellar", 26),
        7: ("cyber", 32),
    }

    gen = torch.Generator(device="cpu").manual_seed(8801)
    # Build 6 heterogeneous domains where prose carries slight non-orthogonal projections onto specialist dims
    prose_train = _make_subspace_batch(96, hidden_size, 0, carrier_strength=3.0, noise_slice=slice(0, 8), seed=8810)
    prose_eval = _make_subspace_batch(64, hidden_size, 0, carrier_strength=3.0, noise_slice=slice(0, 8), seed=8811)
    for _, c_dim in domain_specs.values():
        prose_train[:, c_dim] = 0.55
        prose_eval[:, c_dim] = 0.55

    domain_train: dict[int, torch.Tensor] = {}
    domain_eval: dict[int, torch.Tensor] = {}
    for e_idx, (_, c_dim) in domain_specs.items():
        d_tr = _make_subspace_batch(
            72, hidden_size, c_dim, carrier_strength=3.0, noise_slice=slice(c_dim, c_dim + 6), seed=8900 + e_idx
        )
        d_ev = _make_subspace_batch(
            48, hidden_size, c_dim, carrier_strength=3.0, noise_slice=slice(c_dim, c_dim + 6), seed=9000 + e_idx
        )
        # Inject mild shared prose overlap into specialist tokens to test cross-talk cancellation
        d_tr[:, 0] = 0.25 + 0.10 * torch.randn(72, generator=gen)
        d_ev[:, 0] = 0.25 + 0.10 * torch.randn(48, generator=gen)
        domain_train[e_idx] = d_tr
        domain_eval[e_idx] = d_ev

    pooled_spec = torch.cat(list(domain_train.values()), dim=0)
    u_map = fit_per_domain_orthogonal_gate_vectors(
        prose_train,
        domain_train,
        pooled_spec,
        omega_own=DEFAULT_OMEGA_OWN,
        omega_other=DEFAULT_OMEGA_OTHER,
        ridge_scale=DEFAULT_RIDGE_SCALE,
    )
    assert set(u_map.keys()) == {1, 2, 5, 6, 7}

    # Global pooled probe & un-orthogonalized raw prose mean for cross-talk comparison
    _, raw_mu_prose = second_moment_and_mean(prose_train)
    raw_unit = raw_mu_prose / float((prose_train.double() @ raw_mu_prose).mean().item())
    u_global, probe_stats = fit_uncentered_prose_probe_with_oof(
        prose_train[:48], prose_train[48:], pooled_spec[:180], pooled_spec[180:]
    )
    assert probe_stats["rel_weight"] > 0.9

    for e_idx, x_ev in domain_eval.items():
        u_e = u_map[e_idx]
        prose_proj = float((prose_eval.double() @ u_e).mean().item())
        own_rms = float((x_ev.double() @ u_e).pow(2).mean().sqrt().item())
        raw_rms = float((x_ev.double() @ raw_unit).pow(2).mean().sqrt().item())
        global_rms = float((x_ev.double() @ u_global).pow(2).mean().sqrt().item())
        assert abs(prose_proj - 1.0) < 0.05
        assert own_rms < 0.12
        assert own_rms < 0.35 * raw_rms
        assert own_rms < global_rms

    # Wire into an 8-expert sigma_topk router with leaky prose weights and apply per-domain multi-gate control
    router = nn.Linear(hidden_size, num_experts, bias=False)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0, 0] = 1.10
        for e_idx, (_, c_dim) in domain_specs.items():
            router.weight[e_idx, c_dim] = 1.25
            router.weight[e_idx, 0] = 0.25  # leaky off-domain activation on prose before surgery
        # Rogue unscored experts 3 & 4 also leak on prose
        router.weight[3, 0] = 0.35
        router.weight[4, 0] = 0.35

    meta = apply_sigma_router_prose_control_inplace(
        router,
        u_map,
        rel_weight=probe_stats["rel_weight"],
        beta_suppress=5.0,
        beta_sink=1.2,
        sink_experts=(3,),
        expert0_helps_prose=True,
        sink_direction=u_global,
        fallback_direction=u_global,
    )
    assert meta["applied"] is True
    assert meta["sink_target"] == 3
    assert set(meta["suppressed_experts"]) == {1, 2, 4, 5, 6, 7}

    # Evaluate across all 6 domains: prose off-domain gates < 0.05, own specialist gates > 0.90, 100% top-k recall
    metrics = evaluate_sigma_topk_routing_gates(router, prose_eval, domain_eval, top_k=2)
    for spec_idx in (1, 2, 4, 5, 6, 7):
        assert metrics["prose_gate_by_expert"][spec_idx] < 0.05
    assert metrics["prose_general_gate"] > 0.90
    assert metrics["domain_min_own_gate"] > 0.90
    assert all(hit == 1.0 for hit in metrics["domain_own_topk_hit_rate"].values())

    # Scarce-domain fallback (< min_domain_samples) with pooled_spec_x=None
    scarce_map = {1: domain_train[1][:4], 2: domain_train[2]}
    u_scarce = fit_per_domain_orthogonal_gate_vectors(prose_train, scarce_map, min_domain_samples=8)
    assert set(u_scarce.keys()) == {1, 2}
    assert abs(float((prose_eval.double() @ u_scarce[1]).mean().item()) - 1.0) < 0.1


# ---------------------------------------------------------------------------
# S4: Adaptive Protection of Helpful expert_0 vs Zeroed expert_0 Sink Attraction
# ---------------------------------------------------------------------------


def test_s4_adaptive_protection_vs_zeroed_expert0_sink_attraction() -> None:
    """Verify adaptive protection when expert_0 helps prose vs sink attraction when expert_0 is zeroed."""
    hidden_size = 32
    prose_f0 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=9101)
    prose_f1 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=9102)
    spec_f0 = _make_subspace_batch(64, hidden_size, 10, carrier_strength=2.5, noise_slice=slice(8, 32), seed=9103)
    spec_f1 = _make_subspace_batch(64, hidden_size, 10, carrier_strength=2.5, noise_slice=slice(8, 32), seed=9104)
    u_prose, probe_stats = fit_uncentered_prose_probe_with_oof(prose_f0, prose_f1, spec_f0, spec_f1)
    assert probe_stats["rel_weight"] > 0.95

    dom_eval = {
        1: _make_subspace_batch(32, hidden_size, 10, carrier_strength=2.5, noise_slice=slice(8, 32), seed=9105),
        2: _make_subspace_batch(32, hidden_size, 16, carrier_strength=2.5, noise_slice=slice(8, 32), seed=9106),
        3: _make_subspace_batch(32, hidden_size, 22, carrier_strength=2.5, noise_slice=slice(8, 32), seed=9107),
    }

    # Case 4A: Regime B — expert_0 genuinely helps prose -> strictly protected (unchanged row)
    moe_prot = _build_sigma_moe_fixture(hidden_size=hidden_size, intermediate_size=64, num_experts=4, top_k=2, seed=9110)
    with torch.no_grad():
        moe_prot.router.weight.zero_()
        moe_prot.router.weight[0, 0] = 1.25  # gate ~ 0.958 on prose
        moe_prot.router.weight[1, 0] = 0.20
        moe_prot.router.weight[1, 10] = 1.40
        moe_prot.router.weight[2, 0] = 0.20
        moe_prot.router.weight[2, 16] = 1.40
        moe_prot.router.weight[3, 0] = 0.20
        moe_prot.router.weight[3, 22] = 1.40
    w0_before = moe_prot.router.weight[0].detach().clone()

    meta_prot = apply_sigma_router_prose_control_inplace(
        moe_prot.router,
        u_prose,
        rel_weight=probe_stats["rel_weight"],
        beta_suppress=4.0,
        beta_sink=2.0,
        sink_experts=(),
        expert0_helps_prose=True,
    )
    assert meta_prot["protected_experts"] == [0]
    assert meta_prot["sink_target"] is None
    assert meta_prot["suppressed_experts"] == [1, 2, 3]
    assert torch.equal(moe_prot.router.weight[0], w0_before)

    metrics_prot = evaluate_sigma_topk_routing_gates(moe_prot.router, prose_f0, dom_eval, top_k=2)
    assert metrics_prot["prose_general_gate"] > 0.90
    assert metrics_prot["prose_offdomain_max_gate"] < 0.05
    assert metrics_prot["domain_min_own_gate"] > 0.90

    # Optional general boost when helpful expert_0 started weakly gated
    moe_boost = _build_sigma_moe_fixture(hidden_size=hidden_size, intermediate_size=64, num_experts=4, top_k=2, seed=9111)
    with torch.no_grad():
        moe_boost.router.weight.zero_()
        moe_boost.router.weight[0, 0] = 0.10
    apply_sigma_router_prose_control_inplace(
        moe_boost.router,
        {1: u_prose, 2: u_prose},
        rel_weight=1.0,
        beta_suppress=1.0,
        beta_general_boost=3.0,
        sink_experts=(),
        expert0_helps_prose=True,
    )
    assert float(torch.sigmoid(moe_boost.router(prose_f0)[:, 0]).mean().item()) > 0.90

    # When expert_0 is harmful and NOT zeroed (expert0_helps_prose=False, sink_experts=()), expert_0 is suppressed
    moe_unprot = _build_sigma_moe_fixture(hidden_size=hidden_size, intermediate_size=64, num_experts=4, top_k=2, seed=9112)
    with torch.no_grad():
        moe_unprot.router.weight.zero_()
        moe_unprot.router.weight[0, 0] = 0.25
    meta_unprot = apply_sigma_router_prose_control_inplace(
        moe_unprot.router,
        u_prose,
        rel_weight=1.0,
        beta_suppress=4.0,
        sink_experts=(),
        expert0_helps_prose=False,
    )
    assert meta_unprot["protected_experts"] == []
    assert meta_unprot["suppressed_experts"] == [0, 1, 2, 3]
    assert float(torch.sigmoid(moe_unprot.router(prose_f0)[:, 0]).mean().item()) < 0.05

    # Case 4B: Regime A — expert_0 is zeroed (FoMoE ghost sink: w1_B = w2_B = w3_B = 0 => dy_0 == 0)
    # Sink attraction (+beta_sink * rel * u) pulls prose top-k slot onto zeroed expert_0 while suppressing 1..3
    moe_sink = _build_sigma_moe_fixture(hidden_size=hidden_size, intermediate_size=64, num_experts=4, top_k=2, seed=9120)
    gen_sink = torch.Generator(device="cpu").manual_seed(9121)
    with torch.no_grad():
        for e_idx in range(4):
            exp = moe_sink.experts[str(e_idx)]
            for attr in ("w1_B", "w2_B", "w3_B"):
                param = getattr(exp, attr)
                if e_idx == 0:
                    param.zero_()  # exact FoMoE zeroed ghost sink
                else:
                    param.copy_(torch.randn(param.shape, generator=gen_sink) * 0.35)
        moe_sink.router.weight.zero_()
        moe_sink.router.weight[0, 0] = 0.15  # neutral/weak prior on expert_0
        moe_sink.router.weight[1, 0] = 0.25
        moe_sink.router.weight[1, 10] = 1.40
        moe_sink.router.weight[2, 0] = 0.25
        moe_sink.router.weight[2, 16] = 1.40
        moe_sink.router.weight[3, 0] = 0.25
        moe_sink.router.weight[3, 22] = 1.40

    gate0_before = float(torch.sigmoid(moe_sink.router(prose_f0)[:, 0]).mean().item())
    with torch.no_grad():
        base_out = moe_sink.base_mlp(prose_f0)
        drift_before = float((moe_sink(prose_f0) - base_out).pow(2).mean().item())

    meta_sink = apply_sigma_router_prose_control_inplace(
        moe_sink.router,
        u_prose,
        rel_weight=probe_stats["rel_weight"],
        beta_suppress=4.2,
        beta_sink=3.0,
        sink_experts=(0,),
        expert0_helps_prose=False,
    )
    assert meta_sink["sink_target"] == 0
    assert meta_sink["protected_experts"] == []
    assert meta_sink["suppressed_experts"] == [1, 2, 3]

    metrics_sink = evaluate_sigma_topk_routing_gates(moe_sink.router, prose_f0, dom_eval, top_k=2)
    assert metrics_sink["prose_general_gate"] > gate0_before
    assert metrics_sink["prose_general_gate"] > 0.92
    assert metrics_sink["prose_offdomain_max_gate"] < 0.05
    assert metrics_sink["domain_min_own_gate"] > 0.90

    with torch.no_grad():
        drift_after = float((moe_sink(prose_f0) - base_out).pow(2).mean().item())
    assert drift_after < 0.02 * drift_before


# ---------------------------------------------------------------------------
# S5: Validation Error Branches & Repository Security / Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_error_branches_and_compliance_gates() -> None:
    """Exercise all input validation error branches and verify AGPL/ZeroStub/PRSecurity/Gatekeeper audits."""
    # 1. Config & scalar/tensor validation
    cfg_ok = SigmaRouterControlConfig()
    assert cfg_ok.general_expert_index == GENERAL_EXPERT_INDEX

    with pytest.raises(TypeError, match="real number"):
        SigmaRouterControlConfig(omega_spec=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and >= 0.0"):
        SigmaRouterControlConfig(omega_spec=-0.5)
    with pytest.raises(ValueError, match="finite and >= 0.0"):
        SigmaRouterControlConfig(ridge_scale=float("nan"))
    with pytest.raises(TypeError, match="int"):
        SigmaRouterControlConfig(min_domain_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        SigmaRouterControlConfig(min_domain_samples=-1)

    with pytest.raises(TypeError, match="torch.Tensor"):
        second_moment_and_mean([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        second_moment_and_mean(torch.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="non-empty 2D"):
        second_moment_and_mean(torch.empty((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        second_moment_and_mean(torch.tensor([[1.0, float("nan")]]))

    # 2. fit_uncentered_prose_probe_with_oof & fit_per_domain_orthogonal_gate_vectors validation
    ok2d = torch.ones((4, 6))
    with pytest.raises(ValueError, match="feature dimension"):
        fit_uncentered_prose_probe_with_oof(ok2d, ok2d, ok2d, torch.ones((4, 5)))
    with pytest.raises(ValueError, match="non-empty mapping"):
        fit_per_domain_orthogonal_gate_vectors(ok2d, {})
    with pytest.raises(ValueError, match="does not match prose_x width"):
        fit_per_domain_orthogonal_gate_vectors(ok2d, {1: torch.ones((4, 5))})
    with pytest.raises(ValueError, match="pooled_spec_x feature dimension"):
        fit_per_domain_orthogonal_gate_vectors(ok2d, {1: ok2d}, torch.ones((4, 5)))

    # 3. apply_sigma_router_prose_control_inplace validation & fallback branches
    router = nn.Linear(6, 4, bias=False)
    u_vec = torch.ones(6)
    with pytest.raises(TypeError, match="nn.Linear"):
        apply_sigma_router_prose_control_inplace("not_linear", u_vec)  # type: ignore[arg-type]
    empty_router = nn.Linear(6, 4, bias=False)
    empty_router.weight = nn.Parameter(torch.empty((0, 6)))
    with pytest.raises(ValueError, match="non-empty 2D"):
        apply_sigma_router_prose_control_inplace(empty_router, u_vec)
    with pytest.raises(TypeError, match="bool"):
        apply_sigma_router_prose_control_inplace(router, u_vec, expert0_helps_prose=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        apply_sigma_router_prose_control_inplace(router, u_vec, rel_weight=1.2)
    with pytest.raises(ValueError, match="general_expert_index"):
        apply_sigma_router_prose_control_inplace(router, u_vec, general_expert_index=4)
    with pytest.raises(ValueError, match="sink_expert"):
        apply_sigma_router_prose_control_inplace(router, u_vec, sink_experts=[4])
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_sigma_router_prose_control_inplace(router, "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_sigma_router_prose_control_inplace(router, {1: [1.0, 2.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="1D tensor of length 6"):
        apply_sigma_router_prose_control_inplace(router, torch.ones(5))
    with pytest.raises(ValueError, match="finite"):
        apply_sigma_router_prose_control_inplace(router, torch.tensor([1.0, float("inf"), 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="non-empty"):
        apply_sigma_router_prose_control_inplace(router, {})
    with pytest.raises(ValueError, match="out of range"):
        apply_sigma_router_prose_control_inplace(router, {9: u_vec})
    with pytest.raises(ValueError, match="1D tensor of length 6"):
        apply_sigma_router_prose_control_inplace(router, u_vec, sink_direction=torch.ones(5))
    with pytest.raises(ValueError, match="1D tensor of length 6"):
        apply_sigma_router_prose_control_inplace(router, u_vec, fallback_direction=torch.ones(5))

    # Zero-beta no-op, missing expert skip, fallback_direction, and mean-of-values sink fallback
    meta_noop = apply_sigma_router_prose_control_inplace(
        router, u_vec, rel_weight=1.0, beta_suppress=0.0, beta_sink=0.0, beta_general_boost=0.0
    )
    assert meta_noop["applied"] is False
    meta_zero_vec = apply_sigma_router_prose_control_inplace(
        router, {1: torch.zeros(6)}, rel_weight=1.0, beta_suppress=1.0, beta_sink=1.0, sink_experts=(0,)
    )
    assert meta_zero_vec["applied"] is False
    meta_fb = apply_sigma_router_prose_control_inplace(
        router,
        {1: u_vec},
        rel_weight=1.0,
        beta_suppress=1.0,
        beta_sink=1.0,
        sink_experts=(0,),
        fallback_direction=u_vec,
    )
    assert meta_fb["suppressed_experts"] == [1, 2, 3]
    meta_mean_sink = apply_sigma_router_prose_control_inplace(
        router,
        {1: u_vec},
        rel_weight=1.0,
        beta_suppress=1.0,
        beta_sink=1.0,
        sink_experts=(0,),
    )
    assert meta_mean_sink["sink_target"] == 0
    meta_direct_sink = apply_sigma_router_prose_control_inplace(
        router,
        {0: u_vec, 1: u_vec},
        rel_weight=1.0,
        beta_suppress=1.0,
        beta_sink=1.0,
        sink_experts=(0,),
    )
    assert meta_direct_sink["sink_target"] == 0

    # 4. interpolate_router_weights & evaluate_sigma_topk_routing_gates validation + bias=True support
    r_bias_a = nn.Linear(6, 4, bias=True)
    r_bias_b = nn.Linear(6, 4, bias=True)
    with torch.no_grad():
        r_bias_a.weight.fill_(1.0)
        r_bias_a.bias.fill_(2.0)
        r_bias_b.weight.fill_(3.0)
        r_bias_b.bias.fill_(6.0)
    r_bias_mid = interpolate_router_weights(r_bias_a, r_bias_b, 0.5)
    assert r_bias_mid.bias is not None
    assert torch.allclose(r_bias_mid.weight, torch.full((4, 6), 2.0))
    assert torch.allclose(r_bias_mid.bias, torch.full((4,), 4.0))

    with pytest.raises(TypeError, match="nn.Linear"):
        interpolate_router_weights("a", r_bias_b, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape mismatch"):
        interpolate_router_weights(nn.Linear(5, 4), r_bias_b, 0.5)
    with pytest.raises(ValueError, match="bias"):
        interpolate_router_weights(router, r_bias_b, 0.5)

    with pytest.raises(TypeError, match="nn.Linear"):
        evaluate_sigma_topk_routing_gates("bad", ok2d, {1: ok2d})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        evaluate_sigma_topk_routing_gates(router, ok2d, {1: ok2d}, top_k=0)
    with pytest.raises(ValueError, match="general_expert_index"):
        evaluate_sigma_topk_routing_gates(router, ok2d, {1: ok2d}, general_expert_index=4)
    with pytest.raises(ValueError, match="prose_x width"):
        evaluate_sigma_topk_routing_gates(router, torch.ones((4, 5)), {1: ok2d})
    with pytest.raises(ValueError, match="non-empty mapping"):
        evaluate_sigma_topk_routing_gates(router, ok2d, {})
    with pytest.raises(ValueError, match="out of range"):
        evaluate_sigma_topk_routing_gates(router, ok2d, {9: ok2d})
    with pytest.raises(ValueError, match="does not match router in_features"):
        evaluate_sigma_topk_routing_gates(router, ok2d, {1: torch.ones((4, 5))})

    # 5. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl02", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl02", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/sigma_router_gate_control.py", TARGET_MODULE_PATH),
        ("tests/test_sigma_router_gate_scenarios.py", TEST_FILE_PATH),
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

    telemetry_json = json.dumps({"gate_prob_max": 0.031, "oof_snr": 2.45, "crosstalk_rms": 0.02})
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1102,
        pr_title="feat(swarm): contrastive sigma_topk multi-gate neural controller with OOF reliability shutoff",
        pr_body="Implements NEURAL-CTRL-02 contrastive sigma_topk multi-gate controller and scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(swarm): contrastive sigma_topk multi-gate neural controller with OOF reliability shutoff"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
