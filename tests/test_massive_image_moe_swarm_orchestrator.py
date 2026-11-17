# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the HiveMind 'La Ruche' Massive 7-Division Image-MoE Swarm Orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MASSIVE_IMAGE_MOE_DIVISIONS,
    SHARD_SAMPLE_SIZE,
    SWARM_HARDWARE_TIERS,
    SwarmDataShardOrchestrator,
    run_massive_swarm_image_moe_campaign,
)
from hivemind.core.reddit_vision_vlm_moe import (
    MAX_CONSUMER_8GB_VRAM_MIB,
    MAX_LA_RUCHE_BACKGROUND_VRAM_MIB,
)


def test_swarm_data_shard_orchestrator_and_hardware_tiers() -> None:
    """Verify the 7 divisions, 14 HF dataset streams, 42.87M samples, 8,574 shards, and 104 Bees."""
    orch = SwarmDataShardOrchestrator()
    assert len(MASSIVE_IMAGE_MOE_DIVISIONS) == 7
    assert orch.total_samples == 42_870_000
    assert orch.total_shards == 8_574
    assert orch.total_allocated_bees == 104
    assert sum(int(t["node_count"]) for t in SWARM_HARDWARE_TIERS.values()) == 104

    # Tier 1 (< 1.5 GB / <= 1,536 MiB)
    a_t1 = orch.assign_shard_to_bee("bee_01", available_vram_mib=1400.0, backend="DirectML")
    assert a_t1.hardware_tier == "tier_1_la_ruche_background_1500mb"
    assert a_t1.allocated_vram_mib <= MAX_LA_RUCHE_BACKGROUND_VRAM_MIB
    assert a_t1.samples_in_shard == SHARD_SAMPLE_SIZE

    # Tier 2 (Consumer 8 GB / <= 6,656 MiB)
    a_t2 = orch.assign_shard_to_bee(
        "bee_02",
        available_vram_mib=6000.0,
        backend="ROCm",
        preferred_division_id="div_2_memes_social_vernacular",
    )
    assert a_t2.hardware_tier == "tier_2_consumer_8gb_gamer"
    assert a_t2.division_id == "div_2_memes_social_vernacular"
    assert a_t2.allocated_vram_mib <= MAX_CONSUMER_8GB_VRAM_MIB

    # Tier 3 (Prosumer / Cloud > 6,656 MiB)
    a_t3 = orch.assign_shard_to_bee("bee_03", available_vram_mib=16000.0, backend="CUDA")
    assert a_t3.hardware_tier == "tier_3_prosumer_cloud_anchor"

    with pytest.raises(ValueError, match="available_vram_mib"):
        orch.assign_shard_to_bee("bee_invalid", available_vram_mib=0.0)


def test_run_massive_swarm_image_moe_campaign_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end 7-division massive Image-MoE swarm campaign execution."""
    out_json = tmp_path / "massive_swarm_image_moe_report.json"
    report = run_massive_swarm_image_moe_campaign(
        inner_steps=45,
        output_json_path=out_json,
    )
    assert out_json.exists()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["campaign_id"] == "MASSIVE-SWARM-IMAGE-MOE-V1"
    assert report["total_divisions"] == 7
    assert report["total_dataset_samples"] == 42_870_000
    assert report["total_streaming_shards"] == 8_574
    assert report["total_swarm_bees"] == 104
    assert report["stage1_all_base_frozen"] is True
    assert report["stage1_strictly_under_1500mb_micro"] is True
    assert report["stage1_invariant_i1_satisfied"] is True
    assert report["stage2_mean_focal_loss_reduction_pct"] > 50.0
    assert report["stage3_dht_keys_published_count"] == 7
    assert report["stage3_all_byzantine_rejected"] is True
    assert report["stage3_all_zk_opoi_verified"] is True
    assert report["stage4_routing_accuracy"] >= 0.90
    assert report["stage4_upcycled_moe_focal_loss"] < report["stage4_single_baseline_focal_loss"]
    assert report["overall_campaign_passed"] is True
    assert len(report["report_sha256"]) == 64
