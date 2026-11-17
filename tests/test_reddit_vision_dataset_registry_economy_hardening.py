"""Negative-control and invariant tests for reddit_vision_vlm_moe, dataset_registry, compute_coin, and international_launch_model dataclasses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hivemind.core.reddit_vision_vlm_moe import (
    BeeDiLoCoTrainingResult,
    ByzantineConsensusReport,
    NocturnalRedditVisionCampaignReport,
    SparseLoCoDCT2BitPacket,
    ZKOPoITrainingReceipt,
)
from hivemind.data.dataset_registry import PASSED, Shard, SliceResult
from hivemind.economy.compute_coin import (
    KellyEquilibrium,
    Queue,
    RoutingEquilibrium,
    Worker,
    kelly_equilibrium,
    route_supply_equilibrium,
)
from hivemind.economy.international_launch_model import (
    ErlangCQueueDiagnostic,
    ProducerHardwareTierAllocation,
    ProgressiveGrantQuote,
    RegionalProducerShard,
    SocialFunnelProjection,
    SwarmLiveBaseline,
    build_100_producer_international_fleet_plan,
    compute_erlang_c_metrics,
    compute_progressive_starter_grant,
    project_international_launch_wave,
)


def _valid_dct_packet() -> SparseLoCoDCT2BitPacket:
    return SparseLoCoDCT2BitPacket(
        dht_key="hivemind/vmoe/cluster_01/bee_01",
        bee_id="bee_01",
        cluster_id="cluster_01",
        backbone_id="siglip-base-p16",
        shape=(8, 8),
        q_levels=(-1.0, -0.3, 0.3, 1.0),
        scale_row=[1.0] * 8,
        packed_2bit_codes=b"\x01\x02\x03\x04",
        top_k_indices=[0, 1],
        top_k_values=[0.5, -0.4],
        raw_fp32_bytes=256,
        wire_bytes=32,
        compression_ratio=8.0,
        sqnr_db=14.2,
        sha256_digest="a" * 64,
    )


def test_reddit_vision_vlm_moe_dataclasses_validation() -> None:
    pkt = _valid_dct_packet()
    delta = np.ones((8, 8), dtype=np.float32)

    valid_bee = dict(
        bee_id="bee_01",
        cluster_id="cluster_01",
        hf_dataset="laion/aesthetic-visual-dataset",
        backbone_key="siglip-base-p16",
        backbone_model_id="google/siglip-base-patch16-224",
        rank=16,
        alpha=32.0,
        inner_steps=10,
        loss_before=1.8,
        loss_after=1.1,
        loss_reduction_pct=38.89,
        accuracy_before=0.45,
        accuracy_after=0.82,
        peak_vram_mib=1390.0,
        within_8gb_consumer_ceiling=True,
        base_freeze_verified=True,
        dht_packet=pkt,
        effective_delta=delta,
    )
    res = BeeDiLoCoTrainingResult(**valid_bee)
    assert res.within_8gb_consumer_ceiling is True

    with pytest.raises(ValueError, match="bee_id must be a non-empty string"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "bee_id": "  "})
    with pytest.raises(ValueError, match="rank must be an integer >= 1"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "rank": 0})
    with pytest.raises(ValueError, match="alpha must be a finite positive number"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "alpha": -1.0})
    with pytest.raises(ValueError, match="loss_before must be a finite number"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "loss_before": float("nan")})
    with pytest.raises(ValueError, match="accuracy_after must be a finite number in \\[0.0, 1.0\\]"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "accuracy_after": 1.2})
    with pytest.raises(ValueError, match="within_8gb_consumer_ceiling must be a boolean"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "within_8gb_consumer_ceiling": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dht_packet must be a SparseLoCoDCT2BitPacket"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "dht_packet": "not_pkt"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effective_delta must be a non-empty finite numpy ndarray"):
        BeeDiLoCoTrainingResult(**{**valid_bee, "effective_delta": np.array([], dtype=np.float32)})

    valid_byz = dict(
        cluster_id="cluster_01",
        honest_worker_count=4,
        byzantine_worker_count=1,
        selected_honest_indices=[0, 1, 2, 3],
        rejected_byzantine_indices=[4],
        poison_rejection_rate_pct=100.0,
        validation_replay_fraction=0.1,
        validation_replay_samples=8,
        replay_cosine_similarity=0.99,
        replay_l2_drift=0.01,
        consensus_passed=True,
        aggregated_delta=delta,
    )
    byz = ByzantineConsensusReport(**valid_byz)
    assert byz.consensus_passed is True

    with pytest.raises(ValueError, match="cluster_id must be a non-empty string"):
        ByzantineConsensusReport(**{**valid_byz, "cluster_id": ""})
    with pytest.raises(ValueError, match="honest_worker_count must be an integer >= 1"):
        ByzantineConsensusReport(**{**valid_byz, "honest_worker_count": 0})
    with pytest.raises(ValueError, match="byzantine_worker_count must be a non-negative integer"):
        ByzantineConsensusReport(**{**valid_byz, "byzantine_worker_count": -1})
    with pytest.raises(ValueError, match="selected_honest_indices must be a list of non-negative integers"):
        ByzantineConsensusReport(**{**valid_byz, "selected_honest_indices": [-1]})
    with pytest.raises(ValueError, match="poison_rejection_rate_pct must be a finite number in \\[0.0, 100.0\\]"):
        ByzantineConsensusReport(**{**valid_byz, "poison_rejection_rate_pct": 105.0})
    with pytest.raises(ValueError, match="validation_replay_fraction must be a finite number in \\(0.0, 1.0\\]"):
        ByzantineConsensusReport(**{**valid_byz, "validation_replay_fraction": 0.0})
    with pytest.raises(ValueError, match="replay_cosine_similarity must be a finite number in \\[-1.0, 1.0\\]"):
        ByzantineConsensusReport(**{**valid_byz, "replay_cosine_similarity": 1.5})
    with pytest.raises(ValueError, match="replay_l2_drift must be a finite non-negative number"):
        ByzantineConsensusReport(**{**valid_byz, "replay_l2_drift": -0.1})
    with pytest.raises(ValueError, match="consensus_passed must be a boolean"):
        ByzantineConsensusReport(**{**valid_byz, "consensus_passed": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="aggregated_delta must be a non-empty finite numpy ndarray"):
        ByzantineConsensusReport(**{**valid_byz, "aggregated_delta": np.array([float("nan")])})

    valid_camp = dict(
        campaign_id="camp_vmoe_01",
        title="Nocturnal V-MoE",
        timestamp_utc="2026-11-10T00:00:00Z",
        inner_diloco_steps=10,
        lora_rank=16,
        lora_alpha=32.0,
        base_models_catalog_count=4,
        expert_clusters_count=4,
        sub_bees_trained_count=8,
        stage1_all_base_frozen=True,
        stage2_mean_loss_reduction_pct=35.0,
        stage2_max_consumer_vram_mib=1490.0,
        stage2_invariant_i1_satisfied=True,
        stage3_mean_compression_ratio=12.4,
        stage3_mean_sqnr_db=14.8,
        stage3_dht_keys_published=["hivemind/vmoe/cluster_01/bee_01"],
        stage4_poison_rejection_rate_pct=100.0,
        stage4_replay_cosine_similarity=0.99,
        stage4_replay_l2_drift=0.01,
        stage4_all_consensus_passed=True,
        stage5_router_type="FoMoE-Top2",
        stage5_top_k_active=2,
        stage5_total_experts=4,
        stage5_cluster_routing_accuracy=0.92,
        stage5_upcycled_moe_loss=0.95,
        stage5_single_model_baseline_loss=1.35,
        stage5_moe_vs_baseline_gain_pct=29.6,
        bee_training_results=[{"bee_id": "bee_01"}],
        byzantine_consensus_reports=[{"cluster_id": "cluster_01"}],
        overall_gate_passed=True,
    )
    camp = NocturnalRedditVisionCampaignReport(**valid_camp)
    assert camp.to_dict()["overall_gate_passed"] is True

    with pytest.raises(ValueError, match="campaign_id must be a non-empty string"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "campaign_id": ""})
    with pytest.raises(ValueError, match="stage5_top_k_active must be <= stage5_total_experts"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "stage5_top_k_active": 6})
    with pytest.raises(ValueError, match="stage2_max_consumer_vram_mib must be a finite positive number"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "stage2_max_consumer_vram_mib": 0.0})
    with pytest.raises(ValueError, match="stage3_dht_keys_published must be a non-empty list of non-empty strings"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "stage3_dht_keys_published": []})
    with pytest.raises(ValueError, match="bee_training_results must be a non-empty list of dicts"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "bee_training_results": []})
    with pytest.raises(ValueError, match="overall_gate_passed must be a boolean"):
        NocturnalRedditVisionCampaignReport(**{**valid_camp, "overall_gate_passed": "yes"})  # type: ignore[arg-type]

    valid_receipt = dict(
        bee_id="bee_01",
        specialization_id="art_aesthetics",
        inner_steps=10,
        optimizer_name="Muon + 8-bit AdamW",
        seed_commitment="a" * 64,
        initial_weight_sha256="b" * 64,
        final_delta_sha256="c" * 64,
        dct_packet_sha256="d" * 64,
        proof_commitment_hash="e" * 64,
        verified=True,
    )
    rcpt = ZKOPoITrainingReceipt(**valid_receipt)
    assert rcpt.verified is True

    with pytest.raises(ValueError, match="bee_id must be a non-empty string"):
        ZKOPoITrainingReceipt(**{**valid_receipt, "bee_id": " "})
    with pytest.raises(ValueError, match="inner_steps must be an integer >= 1"):
        ZKOPoITrainingReceipt(**{**valid_receipt, "inner_steps": 0})
    with pytest.raises(ValueError, match="verified must be a boolean"):
        ZKOPoITrainingReceipt(**{**valid_receipt, "verified": 1})  # type: ignore[arg-type]


def test_slice_result_and_economy_equilibria_validation() -> None:
    sr = SliceResult(PASSED, [{"q": "1", "a": "2"}], [Shard("https://example.org", "a" * 64, 12, 1)], "")
    assert sr.status == PASSED

    with pytest.raises(ValueError, match="status must be one of"):
        SliceResult("UNKNOWN")
    with pytest.raises(ValueError, match="rows must be a list of mappings"):
        SliceResult(PASSED, rows=["not_dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="shards must be a list of Shard instances"):
        SliceResult(PASSED, shards=["not_shard"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="reason must be a string"):
        SliceResult(PASSED, reason=123)  # type: ignore[arg-type]

    keq = kelly_equilibrium([10.0, 20.0])
    assert keq.efficiency >= 0.75

    with pytest.raises(ValueError, match="valuations must be a tuple of length >= 2"):
        KellyEquilibrium((10.0,), (1.0,), (5.0,), 5.0, 10.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="shares must be a tuple matching valuations length"):
        KellyEquilibrium((10.0, 20.0), (1.0,), (5.0, 5.0), 10.0, 15.0, 20.0, 0.75)
    with pytest.raises(ValueError, match="valuations must be finite positive numbers"):
        KellyEquilibrium((10.0, -1.0), (0.5, 0.5), (5.0, 5.0), 10.0, 15.0, 20.0, 0.75)
    with pytest.raises(ValueError, match="efficiency must be a finite number in \\(0.0, 1.0\\]"):
        KellyEquilibrium((10.0, 20.0), (0.5, 0.5), (5.0, 5.0), 10.0, 15.0, 20.0, 1.5)

    req = route_supply_equilibrium([Worker("w1", 1.0, 10.0), Worker("w2", 1.2, 10.0)], [Queue("q1", 100.0)])
    assert req.converged is True

    with pytest.raises(ValueError, match="allocation must be a non-empty mapping"):
        RoutingEquilibrium({}, req.payoffs, req.queue_supply, req.queue_price, 1, True, 0.0, True)
    with pytest.raises(ValueError, match="iterations must be an integer >= 1"):
        RoutingEquilibrium(req.allocation, req.payoffs, req.queue_supply, req.queue_price, 0, True, 0.0, True)
    with pytest.raises(ValueError, match="converged must be a boolean"):
        RoutingEquilibrium(req.allocation, req.payoffs, req.queue_supply, req.queue_price, 1, 1, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_deviation_gain_rel must be a finite non-negative number"):
        RoutingEquilibrium(req.allocation, req.payoffs, req.queue_supply, req.queue_price, 1, True, -0.1, True)


def test_international_launch_model_dataclasses_validation() -> None:
    base = SwarmLiveBaseline()
    assert base.total_accounts >= 1

    with pytest.raises(ValueError, match="total_accounts must be an integer >= 1"):
        SwarmLiveBaseline(total_accounts=0)
    with pytest.raises(ValueError, match="prosumers must be a non-negative integer"):
        SwarmLiveBaseline(prosumers=-1)
    with pytest.raises(ValueError, match="p95_latency_s must be >= p50_latency_s"):
        SwarmLiveBaseline(p50_latency_s=0.8, p95_latency_s=0.2)

    q_stable = compute_erlang_c_metrics(8, 480.0, 1.2)
    assert q_stable.is_stable is True
    q_unstable = compute_erlang_c_metrics(4, 100.0, 10.0)
    assert q_unstable.is_stable is False
    assert math.isinf(q_unstable.expected_queue_wait_s)

    with pytest.raises(ValueError, match="active_producers_c must be an integer >= 1"):
        ErlangCQueueDiagnostic(0, 480.0, 2.8, 1.0, 0.35, 0.05, 0.01, 0.4, 2.2, 275, True)
    with pytest.raises(ValueError, match="erlang_c_queue_prob must be a finite number in \\[0.0, 1.0\\]"):
        ErlangCQueueDiagnostic(8, 480.0, 2.8, 1.0, 0.35, 1.5, 0.01, 0.4, 2.2, 275, True)
    with pytest.raises(ValueError, match="expected_queue_wait_s must be a non-negative number"):
        ErlangCQueueDiagnostic(8, 480.0, 2.8, 1.0, 0.35, 0.05, float("nan"), 0.4, 2.2, 275, True)

    grant = compute_progressive_starter_grant(200.0, 400.0)
    assert grant.instant_grant_hmt == 15.0
    with pytest.raises(ValueError, match="instant_grant_hmt must be a finite non-negative number"):
        ProgressiveGrantQuote(-1.0, 85.0, 2.5, 0.5, 85.0)
    with pytest.raises(ValueError, match="effective_unbacked_inflation_reduction_pct must be a finite number in \\[0.0, 100.0\\]"):
        ProgressiveGrantQuote(15.0, 85.0, 2.5, 0.5, 120.0)

    proj = project_international_launch_wave("Wave-1", 1000)
    assert proj.new_visitors == 1000
    with pytest.raises(ValueError, match="wave_name must be a non-empty string"):
        SocialFunnelProjection("", 1000, 180, 5, 14, 161, 1.2, 0.05, 0.08, 600.0, 800.0, 0.4, 0.9, True)
    with pytest.raises(ValueError, match="new_visitors must be an integer >= 1"):
        SocialFunnelProjection("W", 0, 180, 5, 14, 161, 1.2, 0.05, 0.08, 600.0, 800.0, 0.4, 0.9, True)
    with pytest.raises(ValueError, match="actual_conversion_rate_kappa must be a finite number in \\[0.0, 1.0\\]"):
        SocialFunnelProjection("W", 1000, 180, 5, 14, 161, 1.2, 0.05, 1.5, 600.0, 800.0, 0.4, 0.9, True)

    fleet = build_100_producer_international_fleet_plan()
    assert fleet["target_active_producers"] == 100

    with pytest.raises(ValueError, match="region_id must be a non-empty string"):
        RegionalProducerShard("", "EU West", 42, 68, 2000.0, 18.0, 0.4, 0.3)
    with pytest.raises(ValueError, match="active_producers must be an integer >= 1"):
        RegionalProducerShard("eu-west", "EU West", 0, 68, 2000.0, 18.0, 0.4, 0.3)

    with pytest.raises(ValueError, match="tier_id must be a non-empty string"):
        ProducerHardwareTierAllocation("", "8GB", 55, 6656, 1536, 48.0, ("m1",), ("ollama",))
    with pytest.raises(ValueError, match="primary_models must be a non-empty tuple of non-empty strings"):
        ProducerHardwareTierAllocation("std_8gb", "8GB", 55, 6656, 1536, 48.0, (), ("ollama",))
