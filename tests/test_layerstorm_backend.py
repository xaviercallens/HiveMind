# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and pre-registered control tests for Season 25 LayerStoRm x HiveMind PCIe Bridge (HM-S25-01..05)."""

from __future__ import annotations

import pytest

from hivemind.swarm.layerstorm_backend import (
    DEFAULT_DOMAIN_HOT_EXPERTS,
    LayerStoRmSwarmBridge,
    PCIeWaterlineStreamScheduler,
    TwoZoneVRAMBudget,
    TwoZoneVRAMCache,
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
