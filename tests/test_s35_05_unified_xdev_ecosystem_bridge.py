# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S35-05: Unified /home/xavkal/xdev Portfolio Registry & 4-Layer Pipeline."""

from __future__ import annotations

from hivemind.swarm.s35_05_unified_xdev_ecosystem_bridge import (
    build_xdev_portfolio_catalog,
    classify_repo_origin,
    run_unified_4layer_ecosystem_pipeline,
)


def test_classify_repo_origin_distinguishes_x_forks_from_original_ip() -> None:
    assert classify_repo_origin("XSwaT") == "community_fork_bridge"
    assert classify_repo_origin("xLayerStoRm") == "community_fork_bridge"
    assert classify_repo_origin("xensorSharp") == "community_fork_bridge"
    assert classify_repo_origin("AMCP") == "original_core_ip"
    assert classify_repo_origin("HiveMind") == "original_core_ip"
    assert classify_repo_origin("OptimalEnergyAI") == "original_core_ip"
    assert classify_repo_origin("SocrateAI-Lean-Lib") == "original_core_ip"


def test_unified_4layer_ecosystem_pipeline_passes_all_invariants() -> None:
    catalog = build_xdev_portfolio_catalog()
    assert len(catalog) == 19
    receipt = run_unified_4layer_ecosystem_pipeline()
    assert receipt.original_core_count == 11
    assert receipt.fork_bridge_count == 8
    assert receipt.all_layers_passed is True
    assert receipt.layer1_kv_compression_ratio >= 6.0
    assert receipt.layer2_peak_citizen_vram_mib <= 1536.0
    assert receipt.layer3_gwaya_a3_verified is True
    assert receipt.layer4_lean_kernel_clean is True
    assert receipt.layer4_sundials_tier_x_accepted is True
    assert len(receipt.merkle_root_sha256) == 64
