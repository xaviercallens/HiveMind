# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for LayerStoRm Backend, Strategy Optimizer,
Federation Agents, and AMCP EDA Traffic Sustainer (Iteration 72).
"""

from __future__ import annotations

from pathlib import Path
import time

import pytest
import torch

from hivemind.swarm.amcp_eda_traffic_sustainer import (
    AMCPEventDrivenTrafficSustainer,
    build_sunday_producer_first_international_plan,
    plan_100_producer_mesh_scaling,
    select_100_node_optimal_producer,
)
from hivemind.swarm.federation_agents import (
    FederationError,
    OfferRefused,
    ResultRejected,
    RoundBook,
    build_abort,
    build_claim,
    build_offer,
    build_result,
    check_delta,
    check_offer,
    check_result,
    code_pin,
    decode_tensors,
    encode_tensors,
    tensors_sha256,
)
from hivemind.swarm.layerstorm_backend import (
    FoMoESkipAttentionLookaheadRouter,
    HybridHiSparseE8KVCache,
    LayerStoRmSwarmBridge,
    PCIeWaterlineStreamScheduler,
    RealCUDAPCIeExpertStreamer,
    TwoZoneVRAMBudget,
    TwoZoneVRAMCache,
    build_layerstorm_byob_preset,
)
from hivemind.swarm.strategy_optimizer import (
    AdaptiveDiLoCoCadenceController,
    DynamicFisherSVDSelector,
    EF21DiLoCoErrorFeedbackBuffer,
    StrategyOptimizerTelemetryEngine,
    UCB1CurriculumShardScheduler,
    reputation_weighted_ties_merge_lora,
)


# ============================================================================
# 1. LayerStoRm Backend Hardening Tests
# ============================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vram_cap_mib": True},
        {"vram_cap_mib": 0.0},
        {"vram_cap_mib": -100.0},
        {"vram_cap_mib": float("nan")},
        {"vram_cap_mib": float("inf")},
        {"dense_attention_mib": -1.0},
        {"dense_attention_mib": False},
        {"kv_cache_mib": -0.5},
        {"expert_size_mib": 0.0},
        {"expert_size_mib": True},
        {"host_numa_ram_cap_mib": 0.0},
        {"host_numa_ram_cap_mib": float("nan")},
        {"zone_a_slots": True},
        {"zone_a_slots": -1},
        {"zone_b_slots": False},
        {"zone_b_slots": -2},
        {"zone_a_slots": 40, "zone_b_slots": 40},  # Overflow
    ],
)
def test_two_zone_vram_budget_rejects_invalid_inputs(kwargs: dict) -> None:
    base = {
        "vram_cap_mib": 6656.0,
        "dense_attention_mib": 2400.0,
        "kv_cache_mib": 512.0,
        "expert_size_mib": 128.0,
        "zone_a_slots": 8,
        "zone_b_slots": 8,
    }
    base.update(kwargs)
    budget = TwoZoneVRAMBudget(**base)
    with pytest.raises(ValueError):
        budget.validate()


def test_two_zone_vram_cache_and_scheduler_reject_invalid_parameters() -> None:
    budget = TwoZoneVRAMBudget(
        vram_cap_mib=6656.0,
        dense_attention_mib=2400.0,
        kv_cache_mib=512.0,
        expert_size_mib=128.0,
        zone_a_slots=4,
        zone_b_slots=4,
    )
    cache = TwoZoneVRAMCache(budget)

    with pytest.raises(ValueError):
        cache.prewarm_zone_a_from_waggle("")
    with pytest.raises(ValueError):
        cache.prewarm_zone_a_from_waggle("systems_deeptech", explicit_experts=[True])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        cache.prewarm_zone_a_from_waggle("systems_deeptech", explicit_experts=[-1])
    with pytest.raises(ValueError):
        cache.access_expert(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cache.access_expert(-1)

    for bad_sched in (
        {"pcie_bandwidth_gbps": 0.0},
        {"pcie_bandwidth_gbps": True},
        {"pcie_bandwidth_gbps": float("nan")},
        {"layer_attention_ms": -0.1},
        {"layer_attention_ms": False},
        {"fomoe_speculative_overlap_ratio": -0.1},
        {"fomoe_speculative_overlap_ratio": 1.1},
        {"fomoe_speculative_overlap_ratio": True},
    ):
        with pytest.raises(ValueError):
            PCIeWaterlineStreamScheduler(cache, **bad_sched)  # type: ignore[arg-type]

    sched = PCIeWaterlineStreamScheduler(cache)
    with pytest.raises(ValueError):
        sched.execute_layer_step([0, 1], use_waterline_overlap=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sched.execute_layer_step([True])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        sched.execute_layer_step([-2])

    bridge = LayerStoRmSwarmBridge(scheduler=sched)
    with pytest.raises(ValueError):
        bridge.serve_request("", "query", [(0, 1)])
    with pytest.raises(ValueError):
        bridge.serve_request("systems_deeptech", "   ", [(0, 1)])
    with pytest.raises(ValueError):
        bridge.serve_request("systems_deeptech", "query", [(0, 1)], discoverer_bee_id="")
    with pytest.raises(ValueError):
        bridge.serve_request("systems_deeptech", "query", [(0, 1)], prewarm_on_waggle=1)  # type: ignore[arg-type]


def test_real_cuda_pcie_streamer_and_fomoe_router_reject_invalid_inputs() -> None:
    for bad_streamer in (
        {"num_total_experts": 0},
        {"num_total_experts": True},
        {"zone_a_slots": -1},
        {"zone_b_slots": True},
        {"expert_rows": 0},
        {"expert_cols": -4},
    ):
        with pytest.raises(ValueError):
            RealCUDAPCIeExpertStreamer(device="cpu", **bad_streamer)  # type: ignore[arg-type]

    streamer = RealCUDAPCIeExpertStreamer(
        num_total_experts=8, zone_a_slots=2, zone_b_slots=2, expert_rows=16, expert_cols=16, device="cpu"
    )
    with pytest.raises(ValueError):
        streamer.prewarm_zone_a("")
    with pytest.raises(ValueError):
        streamer.prewarm_zone_a("systems_deeptech", explicit_experts=[True])  # type: ignore[list-item]
    x = torch.randn(4, 16)
    with pytest.raises(ValueError):
        streamer.forward_step(x, [0, 1], overlap_dma_with_attention=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        streamer.forward_step(x, [-1])

    for bad_router in (
        {"hidden_dim": 0},
        {"hidden_dim": True},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": 100},
        {"top_k": True},
        {"car_margin_threshold": -0.1},
        {"car_margin_threshold": True},
        {"car_margin_threshold": float("nan")},
        {"seed": -1},
        {"seed": False},
    ):
        with pytest.raises(ValueError):
            FoMoESkipAttentionLookaheadRouter(**bad_router)  # type: ignore[arg-type]

    router = FoMoESkipAttentionLookaheadRouter(hidden_dim=16, num_experts=8, top_k=2)
    with pytest.raises(ValueError):
        router.predict_next_layer_logits(torch.tensor([[float("nan")] * 16]))
    with pytest.raises(ValueError):
        router.predict_next_layer_experts(torch.randn(2, 16), top_k=0)
    with pytest.raises(ValueError):
        router.predict_next_layer_experts(torch.randn(2, 16), top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        router.apply_cache_aware_routing(torch.randn(8), {0, 1}, top_k=0)
    with pytest.raises(ValueError):
        router.apply_cache_aware_routing(torch.tensor([float("inf")] * 8), {0, 1})


def test_hisparse_e8_kv_cache_and_preset_reject_invalid_inputs() -> None:
    for bad_kv in (
        {"sink_tokens": -1},
        {"sink_tokens": True},
        {"sink_tokens": 0, "sliding_window_tokens": 0, "heavy_hitter_tokens": 0},
        {"e8_scale": 0.0},
        {"e8_scale": True},
        {"e8_scale": float("nan")},
        {"num_layers": 0},
        {"num_kv_heads": 0},
        {"head_dim": 0},
    ):
        with pytest.raises(ValueError):
            HybridHiSparseE8KVCache(**bad_kv)  # type: ignore[arg-type]

    kv = HybridHiSparseE8KVCache()
    with pytest.raises(ValueError):
        kv.estimate_vram_kv_mib(0)
    with pytest.raises(ValueError):
        kv.estimate_vram_kv_mib(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        kv.estimate_vram_kv_mib(4096, bits_per_dim=0.0)
    with pytest.raises(ValueError):
        kv.estimate_vram_kv_mib(4096, bits_per_dim=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        build_layerstorm_byob_preset(vram_gb=0.0)
    with pytest.raises(ValueError):
        build_layerstorm_byob_preset(vram_gb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_layerstorm_byob_preset(domain="  ")
    with pytest.raises(ValueError):
        build_layerstorm_byob_preset(context_length=0)
    with pytest.raises(ValueError):
        build_layerstorm_byob_preset(context_length=True)  # type: ignore[arg-type]


# ============================================================================
# 2. Strategy Optimizer Hardening Tests
# ============================================================================


def test_dynamic_fisher_and_ucb1_scheduler_reject_invalid_inputs() -> None:
    for bad_fisher in (
        {"avg_rank_budget_r": 2},
        {"avg_rank_budget_r": True},
        {"allowed_ranks": ()},
        {"allowed_ranks": (0, 8)},
        {"allowed_ranks": (True, 8)},
        {"avg_rank_budget_r": 8, "allowed_ranks": (12, 16)},
    ):
        with pytest.raises(ValueError):
            DynamicFisherSVDSelector(**bad_fisher)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        DynamicFisherSVDSelector.retained_energy_at_rank(torch.ones(8), 0)
    with pytest.raises(ValueError):
        DynamicFisherSVDSelector.retained_energy_at_rank(torch.ones(8), True)  # type: ignore[arg-type]

    for bad_ucb in (
        {"domains": []},
        {"domains": [""]},
        {"domains": ["d1"], "exploration_c": -0.1},
        {"domains": ["d1"], "exploration_c": True},
        {"domains": ["d1"], "exploration_c": float("nan")},
    ):
        with pytest.raises(ValueError):
            UCB1CurriculumShardScheduler(**bad_ucb)  # type: ignore[arg-type]

    sched = UCB1CurriculumShardScheduler(["dom_a", "dom_b"])
    with pytest.raises(ValueError):
        sched.record_observation("", 3.0, 2.0)
    with pytest.raises(ValueError):
        sched.record_observation("dom_a", True, 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sched.record_observation("dom_a", 3.0, float("nan"))
    with pytest.raises(ValueError):
        sched.record_observation("dom_a", 3.0, 2.0, gpu_seconds=0.0)
    with pytest.raises(ValueError):
        sched.record_observation("dom_a", 3.0, 2.0, gpu_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sched.select_next_domains(0)
    with pytest.raises(ValueError):
        sched.select_next_domains(True)  # type: ignore[arg-type]


def test_cadence_ties_ef21_and_telemetry_engine_reject_invalid_inputs() -> None:
    for bad_ctrl in (
        {"max_comm_overhead_ratio": 0.0},
        {"max_comm_overhead_ratio": 1.0},
        {"max_comm_overhead_ratio": True},
        {"min_inner_steps": 0},
        {"min_inner_steps": True},
        {"min_inner_steps": 20, "max_inner_steps": 10},
    ):
        with pytest.raises(ValueError):
            AdaptiveDiLoCoCadenceController(**bad_ctrl)  # type: ignore[arg-type]

    ctrl = AdaptiveDiLoCoCadenceController()
    for bad_rec in (
        {"rtt_ms": 0.0, "bbr_pacing_mbps": 100.0},
        {"rtt_ms": True, "bbr_pacing_mbps": 100.0},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 0.0},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "packet_loss_rate": -0.1},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "packet_loss_rate": 1.0},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "turn_relay_active": 1},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "raw_lora_bytes": 0},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "raw_lora_bytes": True},
        {"rtt_ms": 10.0, "bbr_pacing_mbps": 100.0, "compute_step_ms": 0.0},
    ):
        with pytest.raises(ValueError):
            ctrl.recommend_cadence(**bad_rec)  # type: ignore[arg-type]

    base_lora = {"w": torch.zeros(4, 4)}
    workers = [{"w": torch.ones(4, 4)}]
    with pytest.raises(ValueError):
        reputation_weighted_ties_merge_lora(base_lora, workers, trim_density=0.0)
    with pytest.raises(ValueError):
        reputation_weighted_ties_merge_lora(base_lora, workers, trim_density=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        reputation_weighted_ties_merge_lora(base_lora, workers, scaling_lambda=0.0)
    with pytest.raises(ValueError):
        reputation_weighted_ties_merge_lora(base_lora, workers, worker_reputation_weights=[-0.1])
    with pytest.raises(ValueError):
        reputation_weighted_ties_merge_lora(base_lora, workers, worker_reputation_weights=[True])  # type: ignore[list-item]

    for bad_ef21 in (
        {"topk_ratio": True},
        {"stale_decay": True},
        {"max_residual_norm": True},
        {"max_residual_norm": float("nan")},
    ):
        with pytest.raises(ValueError):
            EF21DiLoCoErrorFeedbackBuffer(**bad_ef21)  # type: ignore[arg-type]

    buf = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.25)
    delta = {"lora_A": torch.randn(4, 4)}
    with pytest.raises(ValueError):
        buf.compress(delta, worker_id="", branch_name="b", outer_step=1)
    with pytest.raises(ValueError):
        buf.compress(delta, worker_id="w", branch_name="  ", outer_step=1)
    with pytest.raises(ValueError):
        buf.compress(delta, worker_id="w", branch_name="b", outer_step=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        buf.compress(delta, worker_id="w", branch_name="b", outer_step=1, inner_steps_completed=0)
    with pytest.raises(ValueError):
        buf.compress_with_pulse_sparseloco_2bit(delta, outer_step=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        buf.compress_with_pulse_sparseloco_2bit(delta, outer_step=1, n_bins=1)

    with pytest.raises(ValueError):
        StrategyOptimizerTelemetryEngine(seed=-1)
    with pytest.raises(ValueError):
        StrategyOptimizerTelemetryEngine(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        StrategyOptimizerTelemetryEngine(cache_ttl_seconds=-0.1)

    eng = StrategyOptimizerTelemetryEngine()
    with pytest.raises(ValueError):
        eng.update({"ring_rtt_ms": True})


# ============================================================================
# 3. Federation Agents Hardening Tests
# ============================================================================


def test_federation_agents_reject_invalid_parameters(tmp_path: Path) -> None:
    with pytest.raises(FederationError):
        code_pin([])
    with pytest.raises(FederationError):
        tensors_sha256({})
    with pytest.raises(FederationError):
        encode_tensors({})
    with pytest.raises(ValueError):
        decode_tensors("")
    with pytest.raises(ValueError):
        decode_tensors("   ")

    g = {"lora_A": torch.ones(2, 2)}
    g_sha = tensors_sha256(g)
    pin = "a" * 64
    base_sha = "b" * 64
    dl = time.time() + 120.0

    for bad_offer in (
        {"rnd": -1},
        {"rnd": True},
        {"sender_id": ""},
        {"base_sha256": "  "},
        {"pin": ""},
        {"inner_steps": 0},
        {"inner_steps": True},
        {"quorum": 0},
        {"quorum": False},
        {"deadline_unix": 0.0},
        {"deadline_unix": True},
        {"deadline_unix": float("nan")},
    ):
        kw = dict(
            run_id="run1",
            rnd=1,
            sender_id="coord",
            global_state=g,
            base_sha256=base_sha,
            pin=pin,
            inner_steps=10,
            deadline_unix=dl,
            quorum=1,
        )
        kw.update(bad_offer)
        with pytest.raises(FederationError):
            build_offer(**kw)  # type: ignore[arg-type]

    for bad_claim in (
        {"run_id": "bad id"},
        {"rnd": -1},
        {"rnd": True},
        {"learner_id": ""},
        {"host": "  "},
    ):
        kw = dict(run_id="run1", rnd=1, learner_id="l0", host="h0")
        kw.update(bad_claim)
        with pytest.raises(FederationError):
            build_claim(**kw)  # type: ignore[arg-type]

    for bad_res in (
        {"run_id": "bad id"},
        {"rnd": -1},
        {"rnd": True},
        {"learner_id": ""},
        {"host": ""},
        {"tokens": -1},
        {"tokens": True},
        {"delta_l2": -0.1},
        {"delta_l2": True},
        {"delta_l2": float("nan")},
        {"eval_loss_global": float("inf")},
        {"eval_loss_global": True},
        {"pin": ""},
        {"global_sha256": ""},
        {"vram_peak_mib": -1.0},
        {"vram_peak_mib": True},
    ):
        kw = dict(
            run_id="run1",
            rnd=1,
            learner_id="l0",
            host="h0",
            tokens=100,
            delta=g,
            delta_l2=0.5,
            eval_loss_global=2.0,
            pin=pin,
            global_sha256=g_sha,
        )
        kw.update(bad_res)
        with pytest.raises(FederationError):
            build_result(**kw)  # type: ignore[arg-type]

    with pytest.raises(FederationError):
        build_abort(run_id="run1", rnd=1, sender_id="", reason="timeout")
    with pytest.raises(FederationError):
        build_abort(run_id="run1", rnd=1, sender_id="coord", reason="   ")

    with pytest.raises(ResultRejected):
        check_delta(g, g, max_l2=-1.0)
    with pytest.raises(ResultRejected):
        check_delta(g, g, max_l2=True)  # type: ignore[arg-type]

    with pytest.raises(ResultRejected):
        check_result(
            {
                "run_id": "run1",
                "round": True,
                "learner_id": "l0",
                "global_sha256": g_sha,
                "script_sha256": pin,
                "tokens": 100,
            },
            run_id="run1",
            rnd=1,
            global_sha256=g_sha,
            pin=pin,
            allowed=["l0"],
        )

    with pytest.raises(ResultRejected):
        check_result(
            {
                "run_id": "run1",
                "round": 1,
                "learner_id": "l0",
                "global_sha256": g_sha,
                "script_sha256": pin,
                "tokens": True,
            },
            run_id="run1",
            rnd=1,
            global_sha256=g_sha,
            pin=pin,
            allowed=["l0"],
        )

    for bad_book in (
        {"run_id": "bad id"},
        {"rnd": -1},
        {"allowed": []},
        {"quorum": 0},
        {"quorum": 3},
        {"deadline_unix": -1.0},
        {"max_delta_l2": -1.0},
        {"template": {}},
    ):
        kw = dict(
            run_id="run1",
            rnd=1,
            allowed=["l0", "l1"],
            quorum=1,
            deadline_unix=dl,
            global_sha256=g_sha,
            pin=pin,
            max_delta_l2=10.0,
            template=g,
        )
        kw.update(bad_book)
        with pytest.raises(FederationError):
            RoundBook(**kw)  # type: ignore[arg-type]


# ============================================================================
# 4. AMCP EDA Traffic Sustainer Hardening Tests
# ============================================================================


def test_amcp_eda_traffic_sustainer_rejects_invalid_inputs() -> None:
    for bad_buf in (0, -5, True):
        with pytest.raises(ValueError):
            AMCPEventDrivenTrafficSustainer(max_buffer_events=bad_buf)  # type: ignore[arg-type]

    sustainer = AMCPEventDrivenTrafficSustainer()
    with pytest.raises(ValueError):
        sustainer.publish_cloudevent("bad.namespace", "src", "subj", {})
    with pytest.raises(ValueError):
        sustainer.publish_cloudevent("io.hivemind.amcp.test", "", "subj", {})
    with pytest.raises(ValueError):
        sustainer.publish_cloudevent("io.hivemind.amcp.test", "src", "  ", {})
    with pytest.raises(ValueError):
        sustainer.publish_cloudevent("io.hivemind.amcp.test", "src", "subj", "not_a_dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sustainer.publish_cloudevent("io.hivemind.amcp.test", "src", "subj", {}, correlation_id="  ")

    for bad_plan in (
        {"wave_name": ""},
        {"base_resident_producers_c0": 0},
        {"base_resident_producers_c0": True},
        {"standby_prosumer_pool": -1},
        {"standby_prosumer_pool": False},
        {"peak_arrival_req_per_s": 0.0},
        {"peak_arrival_req_per_s": True},
        {"peak_arrival_req_per_s": float("nan")},
        {"mean_tokens_per_req": 0.0},
        {"mean_producer_tok_per_s": -1.0},
        {"target_rho": 0.0},
        {"target_rho": 1.0},
        {"hot_bee_id": ""},
        {"hot_model_id": "  "},
    ):
        kw = dict(
            wave_name="Wave 1",
            base_resident_producers_c0=10,
            standby_prosumer_pool=20,
            peak_arrival_req_per_s=5.0,
        )
        kw.update(bad_plan)
        with pytest.raises(ValueError):
            sustainer.plan_elastic_bee_mobility(**kw)  # type: ignore[arg-type]

    good_cand = {"account_id": "p1", "capacity_1k": 10, "inflight": 1, "rtt_ms": 15.0, "loaded": True}
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([good_cand], model_id="")
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([good_cand], model_id="hivemind/coder", client_region="")
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([good_cand], model_id="hivemind/coder", seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([{"capacity_1k": True}], model_id="hivemind/coder")
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([{"capacity_1k": 10, "inflight": -1}], model_id="hivemind/coder")
    with pytest.raises(ValueError):
        select_100_node_optimal_producer([{"capacity_1k": 10, "rtt_ms": float("nan")}], model_id="hivemind/coder")

    for bad_mesh in (
        {"active_producers": 0},
        {"active_producers": True},
        {"endpoints_per_producer": 0},
        {"endpoints_per_producer": 9},
        {"endpoints_per_producer": True},
        {"heartbeat_interval_s": 0.0},
        {"heartbeat_interval_s": True},
        {"live_endpoints": -1},
        {"live_endpoints": True},
    ):
        with pytest.raises(ValueError):
            plan_100_producer_mesh_scaling(**bad_mesh)  # type: ignore[arg-type]

    plan = build_sunday_producer_first_international_plan()
    assert plan["plan_id"] == "HM-S19-AMCP-EDA-SUNDAY-LAUNCH"
    assert len(plan["plan_sha256"]) == 64
