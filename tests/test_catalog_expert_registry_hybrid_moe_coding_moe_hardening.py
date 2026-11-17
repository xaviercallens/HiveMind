# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for Base Model Catalog, Expert Registry RouteDecision, Hybrid/Large-GPU MoE Distributor, and Nocturnal Coding MoE dataclasses (IP-164)."""

from __future__ import annotations

import pytest

from hivemind.core.base_model_catalog import (
    CalibrationRecord,
    FitVerdict,
    ModelGeometry,
    PromotionReceipt,
    VramEstimate,
)
from hivemind.core.expert_registry import RouteDecision
from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    HybridInferenceServiceReport,
    LargeGPUProducerTrainingReport,
    ShardAllocationPlan,
)
from hivemind.core.nocturnal_coding_moe import (
    CodingQualityLoopIteration,
    NocturnalCodingMoEReport,
    run_nocturnal_coding_moe_campaign,
)


def _sample_geometry(**overrides: object) -> ModelGeometry:
    kwargs: dict[str, object] = {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "params_text_linear": 7_000_000_000,
        "params_embedding": 500_000_000,
        "params_text_other": 100_000_000,
        "params_vision": 0,
        "params_mtp": 0,
        "lora_in_plus_out_all_linear": 1_000_000,
        "lora_in_plus_out_mlp": 600_000,
        "max_linear_elements": 50_000_000,
        "hidden": 3584,
        "layers": 28,
        "intermediate": 18944,
        "vocab": 152064,
        "n_kv_heads": 4,
        "head_dim": 128,
        "n_full_attention_layers": 28,
        "n_linear_attention_layers": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
        "linear_key_head_dim": 0,
        "linear_value_head_dim": 0,
        "linear_conv_kernel_dim": 0,
    }
    kwargs.update(overrides)
    return ModelGeometry(**kwargs)  # type: ignore[arg-type]


def test_base_model_catalog_dataclasses_validation() -> None:
    g = _sample_geometry()
    assert g.params_text == 7_600_000_000
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        _sample_geometry(model_id="  ")
    with pytest.raises(ValueError, match="layers must be a positive integer"):
        _sample_geometry(layers=0)
    with pytest.raises(ValueError, match="params_vision must be a non-negative integer"):
        _sample_geometry(params_vision=-1)

    est = VramEstimate(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        mode="qlora_nf4_training",
        config="seq512",
        components_mib={"nf4": 3500.0},
        total_mib=3500.0,
    )
    assert est.total_mib == 3500.0
    with pytest.raises(ValueError, match="components_mib must be a non-empty dict"):
        VramEstimate("m", "qlora_nf4_training", "c", {}, 0.0)
    with pytest.raises(ValueError, match="total_mib must be a finite non-negative number"):
        VramEstimate("m", "qlora_nf4_training", "c", {"nf4": 1.0}, -1.0)

    fv = FitVerdict("FITS", True, 3500.0, 4200.0, 6000.0, 0.2)
    assert fv.fits is True
    with pytest.raises(ValueError, match="verdict must be one of"):
        FitVerdict("UNKNOWN", False, 3500.0, 4200.0, 6000.0, 0.2)
    with pytest.raises(ValueError, match="budget_mib must be positive"):
        FitVerdict("FITS", True, 3500.0, 4200.0, 0.0, 0.2)

    cal = CalibrationRecord(
        model_id="m",
        mode="qlora_nf4_training",
        config="c",
        estimate_mib=3500.0,
        measured_peak_mib=3600.0,
        measured_kind="process_peak",
        ratio_measured_to_estimate=3600.0 / 3500.0,
        within_margin=True,
        margin=0.2,
        gpu="RTX 3070",
        source="nvidia-smi",
        recorded_at="2026-11-10T00:00:00Z",
    )
    assert cal.within_margin is True
    with pytest.raises(ValueError, match="estimate_mib must be a finite positive number"):
        CalibrationRecord(
            model_id="m",
            mode="qlora_nf4_training",
            config="c",
            estimate_mib=0.0,
            measured_peak_mib=3600.0,
            measured_kind="process_peak",
            ratio_measured_to_estimate=1.0,
            within_margin=True,
            margin=0.2,
            gpu="RTX 3070",
            source="nvidia-smi",
            recorded_at="2026-11-10T00:00:00Z",
        )

    pr = PromotionReceipt(
        decision="PROMOTE",
        reasons=["mcnemar_significant"],
        n_items=100,
        wins=30,
        losses=5,
        p_two_sided=0.001,
        guard_items=30,
        guard_wins=5,
        guard_losses=1,
        guard_p_regression=0.95,
    )
    assert pr.promoted is True
    with pytest.raises(ValueError, match="decision must be one of"):
        PromotionReceipt("MAYBE", [], 10, 1, 1, 0.5, 5, 0, 0, 1.0)
    with pytest.raises(ValueError, match="p_two_sided must be a finite probability"):
        PromotionReceipt("HOLD", [], 10, 1, 1, 1.5, 5, 0, 0, 1.0)


def test_route_decision_and_hybrid_moe_distributor_dataclasses_validation() -> None:
    r_exp = RouteDecision(target="expert", expert_id="bee-rust", cluster_id="code", reason="keyword_match")
    assert r_exp.target == "expert"
    r_base = RouteDecision(target="base", expert_id=None, cluster_id=None, reason="no_keyword_match")
    assert r_base.target == "base"

    with pytest.raises(ValueError, match="target must be 'expert' or 'base'"):
        RouteDecision(target="invalid", expert_id=None, cluster_id=None, reason="ok")
    with pytest.raises(ValueError, match="expert_id must be a non-empty string when target is 'expert'"):
        RouteDecision(target="expert", expert_id=None, cluster_id="code", reason="ok")
    with pytest.raises(ValueError, match="expert_id must be None when target is 'base'"):
        RouteDecision(target="base", expert_id="bee-rust", cluster_id=None, reason="ok")

    plan = ShardAllocationPlan(
        division_id="DIV-01",
        division_title="Vision Anchor",
        total_samples=1000,
        total_shards=10,
        large_gpu_shards=7,
        hybrid_micro_shards=3,
        large_gpu_share_pct=70.0,
        assigned_large_producers=("p1",),
        assigned_hybrid_nodes=("h1",),
        primary_large_backbone="Qwen2-VL-7B",
        hybrid_micro_backbone="Qwen2-VL-2B",
    )
    assert plan.total_shards == 10
    with pytest.raises(ValueError, match="large_gpu_shards \\+ hybrid_micro_shards must equal total_shards"):
        ShardAllocationPlan(
            division_id="DIV-01",
            division_title="Vision Anchor",
            total_samples=1000,
            total_shards=10,
            large_gpu_shards=5,
            hybrid_micro_shards=3,
            large_gpu_share_pct=50.0,
            assigned_large_producers=("p1",),
            assigned_hybrid_nodes=("h1",),
            primary_large_backbone="Qwen2-VL-7B",
            hybrid_micro_backbone="Qwen2-VL-2B",
        )

    h_rep = HybridInferenceServiceReport(
        node_id="H-01",
        handle="@hybrid",
        hardware_name="RTX 3060",
        status="ONLINE",
        served_models=("Qwen2.5-Coder-7B",),
        inference_requests_served=120,
        inference_p50_latency_ms=42.0,
        inference_p95_latency_ms=85.0,
        preemption_events_triggered=2,
        mean_preemption_yield_ms=110.0,
        max_preemption_yield_ms=180.0,
        preemption_sla_passed=True,
        inference_vram_mib=3200.0,
        micro_training_vram_mib=2100.0,
        total_active_vram_mib=5300.0,
        free_vram_headroom_mib=2892.0,
        vram_safety_gate_passed=True,
        hot_swapped_vision_adapters=4,
        zero_downtime_hotswap_verified=True,
    )
    assert h_rep.preemption_sla_passed is True
    with pytest.raises(ValueError, match="served_models must be a non-empty tuple"):
        HybridInferenceServiceReport(
            node_id="H-01",
            handle="@hybrid",
            hardware_name="RTX 3060",
            status="ONLINE",
            served_models=(),
            inference_requests_served=120,
            inference_p50_latency_ms=42.0,
            inference_p95_latency_ms=85.0,
            preemption_events_triggered=2,
            mean_preemption_yield_ms=110.0,
            max_preemption_yield_ms=180.0,
            preemption_sla_passed=True,
            inference_vram_mib=3200.0,
            micro_training_vram_mib=2100.0,
            total_active_vram_mib=5300.0,
            free_vram_headroom_mib=2892.0,
            vram_safety_gate_passed=True,
            hot_swapped_vision_adapters=4,
            zero_downtime_hotswap_verified=True,
        )

    l_rep = LargeGPUProducerTrainingReport(
        node_id="L-01",
        handle="@rtx4090",
        hardware_name="RTX 4090",
        assigned_divisions=("DIV-01",),
        assigned_backbones=("Qwen2-VL-7B",),
        shards_assigned=7,
        samples_processed=700,
        lora_rank=16,
        lora_alpha=32.0,
        inner_diloco_steps=50,
        peak_training_vram_mib=18500.0,
        total_vram_mib=24576.0,
        vram_within_budget=True,
        focal_loss_before=2.1,
        focal_loss_after=1.4,
        focal_loss_reduction_pct=33.3,
        wire_compression_ratio=42.0,
        sqnr_db=28.5,
        dht_key="/hivemind/div01",
        zk_opoi_verified=True,
    )
    assert l_rep.zk_opoi_verified is True
    with pytest.raises(ValueError, match="shards_assigned must be a positive integer"):
        LargeGPUProducerTrainingReport(
            node_id="L-01",
            handle="@rtx4090",
            hardware_name="RTX 4090",
            assigned_divisions=("DIV-01",),
            assigned_backbones=("Qwen2-VL-7B",),
            shards_assigned=0,
            samples_processed=700,
            lora_rank=16,
            lora_alpha=32.0,
            inner_diloco_steps=50,
            peak_training_vram_mib=18500.0,
            total_vram_mib=24576.0,
            vram_within_budget=True,
            focal_loss_before=2.1,
            focal_loss_after=1.4,
            focal_loss_reduction_pct=33.3,
            wire_compression_ratio=42.0,
            sqnr_db=28.5,
            dht_key="/hivemind/div01",
            zk_opoi_verified=True,
        )


def test_nocturnal_coding_moe_dataclasses_validation() -> None:
    loop_it = CodingQualityLoopIteration(
        loop_index=1,
        loop_title="Tome 1",
        refinement_strategy="hard_negatives",
        distributed_nodes_active=4,
        inner_steps_executed=16,
        heldout_hard_eval_loss=1.25,
        heldout_hard_routing_acc=0.92,
        heldout_top2_expert_acc=0.98,
        delta_energy_vs_prev_loop=-0.35,
        monotone_ratchet_accepted=True,
        mean_bee_sqnr_db=26.4,
        mean_wire_compression_x=40.0,
        all_zk_opoi_verified=True,
        all_node_vram_within_budget=True,
        polyglot_loop_benchmarks={"Python": {"pass_at_1": 0.88}},
    )
    assert loop_it.monotone_ratchet_accepted is True
    with pytest.raises(ValueError, match="loop_index must be a positive integer"):
        CodingQualityLoopIteration(
            loop_index=0,
            loop_title="Tome 1",
            refinement_strategy="hard_negatives",
            distributed_nodes_active=4,
            inner_steps_executed=16,
            heldout_hard_eval_loss=1.25,
            heldout_hard_routing_acc=0.92,
            heldout_top2_expert_acc=0.98,
            delta_energy_vs_prev_loop=-0.35,
            monotone_ratchet_accepted=True,
            mean_bee_sqnr_db=26.4,
            mean_wire_compression_x=40.0,
            all_zk_opoi_verified=True,
            all_node_vram_within_budget=True,
            polyglot_loop_benchmarks={"Python": {"pass_at_1": 0.88}},
        )

    rep = NocturnalCodingMoEReport(
        campaign_id="HM-S29-NOCTURNAL-POLYGLOT-CODING-MOE-3LOOP-BIG-GPU",
        timestamp_utc=1700000000.0,
        execution_device="cpu",
        inner_diloco_steps=16,
        lora_rank=16,
        lora_alpha=32.0,
        num_quality_loops=3,
        supported_languages=["Python", "Rust"],
        expert_clusters_count=8,
        specialist_bees_count=16,
        big_gpu_nodes_count=4,
        big_gpu_node_allocations=[{"node_id": "N1"}],
        quality_loop_iterations=[{"loop_index": 1}],
        monotone_3loop_ratchet_passed=True,
        stage1_all_base_frozen=True,
        stage2_mean_loss_reduction_pct=85.0,
        stage2_max_consumer_vram_mib=6200.0,
        stage2_invariant_i1_satisfied=True,
        bee_training_results=[{"bee_id": "b1"}],
        stage3_mean_compression_ratio=42.0,
        stage3_mean_sqnr_db=28.0,
        stage3_dht_keys_published=["/hivemind/b1"],
        stage4_poison_rejection_rate_pct=100.0,
        stage4_replay_cosine_similarity=0.99,
        stage4_replay_l2_drift=0.01,
        stage5_cluster_routing_accuracy=0.95,
        stage5_single_model_baseline_loss=1.8,
        stage5_upcycled_moe_loss=1.1,
        stage5_moe_vs_baseline_gain_pct=38.8,
        polyglot_benchmark_scores={"Python": {"pass_at_1": 0.9}},
        sample_epic_dag_plan=[{"subtask_id": "t1"}],
        t4_free_vram_mib=12000.0,
        t4_vram_floor_satisfied=True,
        overall_gate_passed=True,
        report_sha256="a" * 64,
    )
    assert rep.overall_gate_passed is True
    with pytest.raises(ValueError, match="campaign_id must be a non-empty string"):
        NocturnalCodingMoEReport(
            campaign_id="",
            timestamp_utc=1700000000.0,
            execution_device="cpu",
            inner_diloco_steps=16,
            lora_rank=16,
            lora_alpha=32.0,
            num_quality_loops=3,
            supported_languages=["Python", "Rust"],
            expert_clusters_count=8,
            specialist_bees_count=16,
            big_gpu_nodes_count=4,
            big_gpu_node_allocations=[{"node_id": "N1"}],
            quality_loop_iterations=[{"loop_index": 1}],
            monotone_3loop_ratchet_passed=True,
            stage1_all_base_frozen=True,
            stage2_mean_loss_reduction_pct=85.0,
            stage2_max_consumer_vram_mib=6200.0,
            stage2_invariant_i1_satisfied=True,
            bee_training_results=[{"bee_id": "b1"}],
            stage3_mean_compression_ratio=42.0,
            stage3_mean_sqnr_db=28.0,
            stage3_dht_keys_published=["/hivemind/b1"],
            stage4_poison_rejection_rate_pct=100.0,
            stage4_replay_cosine_similarity=0.99,
            stage4_replay_l2_drift=0.01,
            stage5_cluster_routing_accuracy=0.95,
            stage5_single_model_baseline_loss=1.8,
            stage5_upcycled_moe_loss=1.1,
            stage5_moe_vs_baseline_gain_pct=38.8,
            polyglot_benchmark_scores={"Python": {"pass_at_1": 0.9}},
            sample_epic_dag_plan=[{"subtask_id": "t1"}],
            t4_free_vram_mib=12000.0,
            t4_vram_floor_satisfied=True,
            overall_gate_passed=True,
            report_sha256="a" * 64,
        )
