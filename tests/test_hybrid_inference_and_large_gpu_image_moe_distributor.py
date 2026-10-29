# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for Hybrid Inference Server & Large-GPU Producer Image-MoE Distributor."""

from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    GPUProducerNodeSpec,
    MAX_INFERENCE_PREEMPTION_MS,
    PRODUCER_AND_HYBRID_FLEET,
    RTX2070_8GB_GOLDEN_CEILING_MIB,
    TESLA_T4_MIN_FREE_HEADROOM_MIB,
    execute_hybrid_node_inference_and_micro_training,
    execute_large_gpu_producer_training,
    partition_swarm_shards_between_large_and_hybrid,
    run_hybrid_inference_and_large_gpu_image_moe_campaign,
)


def test_partition_swarm_shards_between_large_and_hybrid() -> None:
    plans = partition_swarm_shards_between_large_and_hybrid()
    assert len(plans) == 7
    total_large = sum(p.large_gpu_shards for p in plans)
    total_hybrid = sum(p.hybrid_micro_shards for p in plans)
    assert total_large + total_hybrid == 8574
    # Large-GPU producers (weight=104) take ~80% of shards; Hybrid nodes (weight=26) take ~20%
    assert total_large > total_hybrid * 3
    for p in plans:
        assert p.large_gpu_shards >= 1
        assert p.hybrid_micro_shards >= 1
        assert 75.0 <= p.large_gpu_share_pct <= 85.0


def test_partition_validation_and_edge_clamp() -> None:
    only_large = [n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "LARGE_GPU_DEDICATED_TRAINER"]
    with pytest.raises(ValueError, match="Both LARGE_GPU_DEDICATED_TRAINER"):
        partition_swarm_shards_between_large_and_hybrid(fleet=only_large)

    # Test tiny shard division where rounding would hit total_shards
    tiny_div = {
        "div_tiny": {
            "title": "Tiny Div",
            "primary_backbone": "Qwen/Qwen2.5-VL-7B-Instruct",
            "micro_vram_backbone": "google/vit-base-patch16-224",
            "datasets": [{"total_samples": 10000, "num_shards": 2}],
        }
    }
    plans = partition_swarm_shards_between_large_and_hybrid(divisions=tiny_div)
    assert plans[0].large_gpu_shards == 1
    assert plans[0].hybrid_micro_shards == 1


def test_execute_large_gpu_producer_training_and_validation() -> None:
    large_node = next(n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "LARGE_GPU_DEDICATED_TRAINER")
    hybrid_node = next(n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "HYBRID_INFERENCE_AND_MICRO_TRAINER")

    with pytest.raises(ValueError, match="not a LARGE_GPU_DEDICATED_TRAINER"):
        execute_large_gpu_producer_training(node=hybrid_node, assigned_shards=10, assigned_divisions=("div_1",))

    with pytest.raises(ValueError, match="inner_steps must be >= 1"):
        execute_large_gpu_producer_training(
            node=large_node, assigned_shards=10, assigned_divisions=("div_1",), inner_steps=0
        )

    rep = execute_large_gpu_producer_training(
        node=large_node,
        assigned_shards=120,
        assigned_divisions=("div_1_aesthetics_digital_art", "div_3_technical_infographics"),
        inner_steps=60,
        seed=77,
    )
    assert rep.vram_within_budget is True
    assert rep.zk_opoi_verified is True
    assert rep.focal_loss_after < rep.focal_loss_before
    assert rep.focal_loss_reduction_pct > 50.0
    assert rep.samples_processed == 120 * 5000


def test_execute_hybrid_node_inference_and_micro_training_all_tiers() -> None:
    large_node = next(n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "LARGE_GPU_DEDICATED_TRAINER")
    with pytest.raises(ValueError, match="not a HYBRID_INFERENCE_AND_MICRO_TRAINER"):
        execute_hybrid_node_inference_and_micro_training(node=large_node)

    hybrid_nodes = [n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "HYBRID_INFERENCE_AND_MICRO_TRAINER"]
    assert len(hybrid_nodes) == 3  # 16 GB T4, 8 GB RTX 2070, 12 GB Prosumer
    for node in hybrid_nodes:
        rep = execute_hybrid_node_inference_and_micro_training(node=node, incoming_inference_requests=40)
        assert rep.vram_safety_gate_passed is True
        assert rep.preemption_sla_passed is True
        assert rep.max_preemption_yield_ms <= MAX_INFERENCE_PREEMPTION_MS
        assert rep.zero_downtime_hotswap_verified is True
        if node.total_vram_mib <= 8192.0:
            assert rep.total_active_vram_mib <= RTX2070_8GB_GOLDEN_CEILING_MIB
        elif node.total_vram_mib == 15360.0:
            assert rep.free_vram_headroom_mib >= TESLA_T4_MIN_FREE_HEADROOM_MIB


def test_run_full_hybrid_and_large_gpu_campaign(tmp_path: Path) -> None:
    report_file = tmp_path / "hybrid_report.json"
    registry_file = tmp_path / "producers_registry.json"
    result = run_hybrid_inference_and_large_gpu_image_moe_campaign(
        inner_steps=50,
        save_report_path=report_file,
        update_producer_registry_path=registry_file,
    )
    assert result["overall_distribution_passed"] is True
    assert result["total_shards"] == 8574
    assert result["large_gpu_shards_assigned"] + result["hybrid_micro_shards_assigned"] == 8574
    assert report_file.exists()
    assert registry_file.exists()
