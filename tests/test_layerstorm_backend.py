# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and pre-registered control tests for Season 25 LayerStoRm x HiveMind PCIe Bridge (HM-S25-01..05)."""

from __future__ import annotations

import pytest

from hivemind.swarm.layerstorm_backend import (
    DEFAULT_DOMAIN_HOT_EXPERTS,
    FoMoESkipAttentionLookaheadRouter,
    HybridHiSparseE8KVCache,
    LayerStoRmSwarmBridge,
    PCIeWaterlineStreamScheduler,
    RealCUDAPCIeExpertStreamer,
    TwoZoneVRAMBudget,
    TwoZoneVRAMCache,
    build_layerstorm_byob_preset,
)
from hivemind.swarm.swarm_nectar_cache import SwarmNectarCache


def _make_8gb_budget(zone_a_slots: int = 8, zone_b_slots: int = 12) -> TwoZoneVRAMBudget:
    # 2400 MiB dense + 1024 MiB KV + (8 + 12) * 128 MiB = 5984 MiB <= 6656 MiB WDDM ceiling
    return TwoZoneVRAMBudget(
        vram_cap_mib=6656.0,
        dense_attention_mib=2400.0,
        kv_cache_mib=1024.0,
        expert_size_mib=128.0,
        zone_a_slots=zone_a_slots,
        zone_b_slots=zone_b_slots,
    )


def test_p19d_and_negative_control_vram_overflow_guard() -> None:
    """Verify P19d and planted negative control: Two-Zone VRAM ceiling fails closed on overflow."""
    valid_budget = _make_8gb_budget()
    valid_budget.validate()
    assert valid_budget.total_vram_mib == 5984.0
    assert valid_budget.headroom_mib == 672.0

    # Negative control: 2400 + 1024 + (16 + 16) * 128 = 7520 MiB > 6656 MiB -> must raise ValueError
    overflow_budget = TwoZoneVRAMBudget(
        vram_cap_mib=6656.0,
        dense_attention_mib=2400.0,
        kv_cache_mib=1024.0,
        expert_size_mib=128.0,
        zone_a_slots=16,
        zone_b_slots=16,
    )
    with pytest.raises(ValueError, match="Two-Zone VRAM overflow"):
        TwoZoneVRAMCache(overflow_budget)


def test_p19b_waggle_warm_zone_a_vs_cold_and_random_control() -> None:
    """Verify P19b (Waggle-Warm TTFT reduction >= 20%) and random pre-warm negative control."""
    domain = "systems_deeptech"
    domain_trace = [(0, 1, 2, 3, 40, 41)]  # 4 of 6 experts belong to systems_deeptech hot set

    # Condition C0: Cold reactive PCIe streaming
    c0_cache = TwoZoneVRAMCache(_make_8gb_budget())
    c0_sched = PCIeWaterlineStreamScheduler(c0_cache, pcie_bandwidth_gbps=24.0, layer_attention_ms=1.8)
    c0_step = c0_sched.execute_layer_step(domain_trace[0], use_waterline_overlap=False)

    # Condition C1: Waggle-Warm Zone A with domain-matched experts
    c1_cache = TwoZoneVRAMCache(_make_8gb_budget())
    c1_cache.prewarm_zone_a_from_waggle(domain=domain)
    c1_sched = PCIeWaterlineStreamScheduler(c1_cache, pcie_bandwidth_gbps=24.0, layer_attention_ms=1.8)
    c1_step = c1_sched.execute_layer_step(domain_trace[0], use_waterline_overlap=False)

    # Planted Negative Control: Waggle-Warm Zone A with disjoint random experts (100..107)
    ctrl_cache = TwoZoneVRAMCache(_make_8gb_budget())
    ctrl_cache.prewarm_zone_a_from_waggle(domain=domain, explicit_experts=[100, 101, 102, 103, 104, 105, 106, 107])
    ctrl_sched = PCIeWaterlineStreamScheduler(ctrl_cache, pcie_bandwidth_gbps=24.0, layer_attention_ms=1.8)
    ctrl_step = ctrl_sched.execute_layer_step(domain_trace[0], use_waterline_overlap=False)

    ttft_reduction_c1 = (c0_step.total_step_ms - c1_step.total_step_ms) / c0_step.total_step_ms
    ttft_reduction_ctrl = (c0_step.total_step_ms - ctrl_step.total_step_ms) / c0_step.total_step_ms

    assert ttft_reduction_c1 >= 0.20, f"Expected >= 20% TTFT reduction in C1, got {ttft_reduction_c1:.3f}"
    assert ttft_reduction_ctrl == 0.0, "Random expert pre-warm control must yield 0% TTFT reduction"
    assert c1_step.pcie_h2d_mib < c0_step.pcie_h2d_mib


def test_p19a_upstream_swarm_nectar_cache_pcie_relief() -> None:
    """Verify P19a: Upstream SwarmNectarCache cuts PCIe H2D traffic by >= 50% on recurring workloads."""
    domain = "cyber_defense"
    hot_experts = DEFAULT_DOMAIN_HOT_EXPERTS[domain]
    traces = [
        (hot_experts[0], hot_experts[1], 50, 51),
        (hot_experts[2], hot_experts[3], 52, 53),
    ]

    # Condition C0: 10 requests without Nectar Cache or Waggle prewarm (fresh cold cache per request)
    c0_total_h2d = 0.0
    for i in range(10):
        sched = PCIeWaterlineStreamScheduler(TwoZoneVRAMCache(_make_8gb_budget()))
        for t in traces:
            step = sched.execute_layer_step(t, use_waterline_overlap=False)
            c0_total_h2d += step.pcie_h2d_mib

    # Condition C2: Full Season 25 stack across 10 requests (5 unique queries, each repeated once -> 50% cache hit + Zone A/B reuse)
    c2_bridge = LayerStoRmSwarmBridge(
        scheduler=PCIeWaterlineStreamScheduler(TwoZoneVRAMCache(_make_8gb_budget())),
        nectar_cache=SwarmNectarCache(),
    )
    c2_total_h2d = 0.0
    queries = [f"audit smart contract reentrancy vector #{i % 5}" for i in range(10)]
    for q in queries:
        res = c2_bridge.serve_request(domain=domain, query=q, token_expert_traces=traces)
        c2_total_h2d += res["pcie_h2d_mib"]

    pcie_reduction_ratio = (c0_total_h2d - c2_total_h2d) / c0_total_h2d
    assert pcie_reduction_ratio >= 0.50, f"Expected >= 50% PCIe H2D reduction, got {pcie_reduction_ratio:.3f}"
    stats = c2_bridge.nectar_cache.stats()
    assert stats["cache_hits"] == 5
    assert stats["total_hmt_royalties_distributed"] == pytest.approx(5 * 0.005)


def test_p19c_gamer_preemption_circuit_breaker_under_500ms() -> None:
    """Verify P19c: Gamer preemption halts PCIe DMA and clears Zone A/B in < 500 ms."""
    cache = TwoZoneVRAMCache(_make_8gb_budget())
    sched = PCIeWaterlineStreamScheduler(cache)
    bridge = LayerStoRmSwarmBridge(scheduler=sched)

    bridge.serve_request(
        domain="stem_scholar",
        query="solve 3D Navier-Stokes vorticity transport on adaptive mesh",
        token_expert_traces=[(16, 17, 70, 71)],
    )
    assert len(cache.zone_a_experts) > 0
    assert len(cache.zone_b_experts) > 0

    preempt_receipt = bridge.trigger_gamer_preemption()
    assert preempt_receipt["preempted"] is True
    assert preempt_receipt["sla_met_under_500ms"] is True
    assert preempt_receipt["yield_latency_ms"] < 500.0
    assert preempt_receipt["zone_a_remaining"] == 0
    assert preempt_receipt["zone_b_remaining"] == 0

    with pytest.raises(RuntimeError, match="preempted"):
        bridge.serve_request(
            domain="stem_scholar",
            query="blocked query during game",
            token_expert_traces=[(16, 17)],
        )

    bridge.resume_after_game()
    resumed = bridge.serve_request(
        domain="stem_scholar",
        query="post-game query",
        token_expert_traces=[(16, 17)],
    )
    assert resumed["status"] == "layerstorm_pcie_streamed"


def test_real_cuda_pcie_expert_streamer_and_preemption() -> None:
    """Verify RealCUDAPCIeExpertStreamer (HM-S25-01 / E19) on GPU or CPU fallback."""
    import torch

    streamer = RealCUDAPCIeExpertStreamer(
        num_total_experts=16,
        zone_a_slots=4,
        zone_b_slots=4,
        expert_rows=128,
        expert_cols=256,
    )
    warmed = streamer.prewarm_zone_a("systems_deeptech")
    assert warmed == [0, 1, 2, 3]
    x = torch.randn((16, 128), dtype=streamer.dtype, device=streamer.device)
    step = streamer.forward_step(x, [0, 1, 8, 9], overlap_dma_with_attention=True)
    # 0 and 1 hit Zone A; 8 and 9 stream into Zone B
    assert streamer.zone_a_hits == 2
    assert streamer.pcie_misses == 2
    assert step["streamed_experts"] == (8, 9)
    assert step["pcie_h2d_mib"] > 0.0

    # Second step hitting 8 and 9 hits Zone B with 0 PCIe traffic
    step2 = streamer.forward_step(x, [0, 1, 8, 9], overlap_dma_with_attention=False)
    assert step2["pcie_h2d_mib"] == 0.0
    assert streamer.zone_b_hits == 2

    receipt = streamer.preempt_and_release_vram()
    assert receipt["sla_met_under_500ms"] is True
    assert receipt["zone_a_remaining"] == 0
    assert receipt["zone_b_remaining"] == 0


def test_hm_s25_03_fomoe_skip_attention_lookahead_and_car_fallback() -> None:
    """Verify HM-S25-03: FoMoE Skip-Attention lookahead + CAR keeps PCIe stall ratio < 15%."""
    import torch

    router = FoMoESkipAttentionLookaheadRouter(
        hidden_dim=64,
        num_experts=32,
        top_k=4,
        car_margin_threshold=0.45,
        seed=42,
    )
    gen = torch.Generator(device="cpu").manual_seed(101)
    hiddens = [torch.randn((8, 64), generator=gen) for _ in range(8)]
    # Small post-attention residual perturbations so CAR fallback can substitute resident experts
    perturbations = [0.08 * torch.randn((32,), generator=gen) for _ in range(8)]

    # Reactive baseline (no lookahead, no CAR fallback, no waterline overlap)
    reactive_cache = TwoZoneVRAMCache(_make_8gb_budget())
    reactive_cache.prewarm_zone_a_from_waggle("systems_deeptech")
    reactive_sched = PCIeWaterlineStreamScheduler(
        reactive_cache, pcie_bandwidth_gbps=24.0, layer_attention_ms=14.0
    )
    reactive_res = router.run_waterline_schedule(
        reactive_sched,
        hiddens,
        residual_perturbations=perturbations,
        use_skip_attention_lookahead=False,
        use_car_fallback=False,
    )

    # FoMoE Skip-Attention Lookahead + CAR fallback + Waterline overlap
    fomoe_cache = TwoZoneVRAMCache(_make_8gb_budget())
    fomoe_cache.prewarm_zone_a_from_waggle("systems_deeptech")
    fomoe_sched = PCIeWaterlineStreamScheduler(
        fomoe_cache, pcie_bandwidth_gbps=24.0, layer_attention_ms=14.0
    )
    fomoe_res = router.run_waterline_schedule(
        fomoe_sched,
        hiddens,
        residual_perturbations=perturbations,
        use_skip_attention_lookahead=True,
        use_car_fallback=True,
    )

    assert fomoe_res["stall_gate_under_15pct"] is True
    assert fomoe_res["pcie_stall_ratio_pct"] < 15.0
    assert reactive_res["pcie_stall_ratio_pct"] > 35.0
    assert fomoe_res["lookahead_recall"] >= 0.70


def test_hm_s25_04_hybrid_hisparse_e8_kv_cache_and_random_control() -> None:
    """Verify HM-S25-04: Hybrid HiSparse + Gosset E8 KV Cache < 0.05 delta vs random control."""
    import torch

    kv = HybridHiSparseE8KVCache(
        sink_tokens=16,
        sliding_window_tokens=64,
        heavy_hitter_tokens=48,
        e8_scale=0.02,
    )
    est_32k = kv.estimate_vram_kv_mib(32768)
    est_128k = kv.estimate_vram_kv_mib(131072)
    assert est_32k["hisparse_e8_vram_mib"] < 1024.0
    assert est_128k["hisparse_e8_vram_mib"] < 1024.0
    assert est_32k["vram_compression_ratio"] > 50.0

    # Construct a realistic long-context attention distribution (sink + heavy hitters + local window)
    gen = torch.Generator(device="cpu").manual_seed(77)
    seq_len, dim = 512, 32
    q_dir = torch.randn((1, dim), generator=gen)
    q_dir = q_dir / q_dir.norm()
    q = q_dir * 8.0

    # Background cold tokens have orthogonal/negative alignment; sink, heavy-hitters, and window align with q
    k = torch.randn((seq_len, dim), generator=gen) * 0.20 - 2.0 * q_dir
    v = torch.randn((seq_len, dim), generator=gen) * 0.15
    hot_positions = list(range(0, 12)) + [80, 140, 210, 290, 350] + list(range(seq_len - 48, seq_len))
    for pos in hot_positions:
        k[pos] = q_dir * 3.5 + 0.05 * torch.randn((dim,), generator=gen)
        v[pos] = torch.randn((dim,), generator=gen) * 1.2

    fidelity = kv.evaluate_attention_fidelity(q, k, v, random_control=False)
    ctrl_fidelity = kv.evaluate_attention_fidelity(q, k, v, random_control=True)

    assert fidelity["passed_delta_under_0_05"] is True
    assert fidelity["relative_l2_delta"] < 0.05
    assert fidelity["retained_attention_mass"] > 0.95
    assert ctrl_fidelity["relative_l2_delta"] > fidelity["relative_l2_delta"] * 3.0


def test_hm_s25_05_byob_layerstorm_preset_and_catalog_alias() -> None:
    """Verify HM-S25-05: build_layerstorm_byob_preset and router catalog alias normalization."""
    from hivemind.router.catalog import normalize_model_alias, tier_by_id

    preset_8gb = build_layerstorm_byob_preset(vram_gb=8.0, domain="cyber_defense", context_length=32768)
    assert preset_8gb["preset_id"] == "layerstorm-frontier-moe"
    assert preset_8gb["vram_budget"]["total_vram_mib"] <= 6656.0
    assert preset_8gb["gamer_preemption_sla_ms"] == 500.0

    assert normalize_model_alias("layerstorm-frontier-moe") == "moe-general"
    assert normalize_model_alias("hivemind/layerstorm-frontier-moe") == "moe-general"
    tier = tier_by_id("layerstorm-frontier-moe")
    assert tier is not None and tier.tier_id == "moe-general"


def test_e19_layerstorm_pcie_nectar_report_artifact() -> None:
    """Verify HM-S25-01..05 / E19 T4 report artifact passes all P19a..P19f hypotheses."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / "artifacts" / "experiments" / "E19_layerstorm_pcie_nectar_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"] == "E19"
    assert report["all_preregistered_hypotheses_passed"] is True
    assert report["hardware"]["t4_policy_4300mib_floor_respected"] is True
    for hyp_key in (
        "P19a_nectar_cache_pcie_relief_ge_50pct",
        "P19b_waggle_warm_zone_a_ttft_reduction_ge_20pct",
        "P19c_gamer_preemption_yield_under_500ms",
        "P19d_two_zone_vram_ceiling_under_6656mib",
        "P19e_fomoe_waterline_stall_ratio_under_15pct",
        "P19f_hisparse_e8_kv_32k_delta_under_0_05",
    ):
        assert report["hypothesis_verdicts"][hyp_key]["passed"] is True


