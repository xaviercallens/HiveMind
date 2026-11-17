# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the HiveMind 'La Ruche' Exoplanet & Astrophysics Vision-MoE (<1.5 GB VRAM) PoC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.core.exoplanet_astro_vmoe_poc import (
    EARTH_SUN_TRANSIT_DEPTH_PPM,
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    FrozenAstroVisionBackbone,
    SwarmFFNAdapter,
    UpcycledAstroSparseVisionMoE,
    apply_nesterov_outer_momentum,
    asymmetric_focal_loss,
    build_astrophysics_visual_shard,
    compute_citizen_science_entropy_weights,
    compute_physical_transit_depth_ppm,
    generate_phase_folded_cwt_scalogram,
    run_exoplanet_astro_vmoe_poc,
    synthesize_exoplanet_transit_and_router_response,
)
from hivemind.core.reddit_vision_vlm_moe import MAX_LA_RUCHE_BACKGROUND_VRAM_MIB


def test_exoplanet_astro_branches_and_physical_transit_depth() -> None:
    """Verify the 4 astrophysics branches (< 1.5 GB VRAM) and transit depth equations."""
    assert len(EXOPLANET_ASTRO_EXPERT_BRANCHES) == 4
    for b_id, meta in EXOPLANET_ASTRO_EXPERT_BRANCHES.items():
        assert 900.0 <= meta["peak_vram_mib"] <= MAX_LA_RUCHE_BACKGROUND_VRAM_MIB
        assert meta["lora_rank_ffn"] == 16
        assert meta["lora_alpha_ffn"] == 32.0
        assert meta["lora_rank_attn"] == 4
        assert len(meta["hf_datasets"]) >= 1

    # 1.0 R_earth around 1.0 R_sun -> ~84.0 ppm
    d_earth_sun = compute_physical_transit_depth_ppm(1.0, 1.0)
    assert d_earth_sun == pytest.approx(EARTH_SUN_TRANSIT_DEPTH_PPM, rel=1e-5)

    # 1.0 R_earth around 0.2 R_sun M-dwarf -> 84.0 / 0.04 = 2,100.0 ppm
    d_earth_mdwarf = compute_physical_transit_depth_ppm(1.0, 0.2)
    assert d_earth_mdwarf == pytest.approx(2100.0, rel=1e-5)

    with pytest.raises(ValueError, match="planet_radius_earth"):
        compute_physical_transit_depth_ppm(0.0, 1.0)
    with pytest.raises(ValueError, match="star_radius_solar"):
        compute_physical_transit_depth_ppm(1.0, -0.5)


def test_cwt_scalogram_focal_loss_citizen_entropy_and_nesterov() -> None:
    """Verify 2D CWT scalogram, Focal Loss (gamma=2.0), Citizen Science entropy, and Nesterov momentum."""
    flux, scalogram = generate_phase_folded_cwt_scalogram(
        transit_depth_ppm=2100.0,
        instrumental_noise_ppm=100.0,
        odd_even_asymmetry_ratio=0.1,
        num_phase_bins=32,
        num_scales=16,
    )
    assert flux.shape == (32,)
    assert scalogram.shape == (16, 32)
    assert float(np.max(np.abs(scalogram))) > 100.0

    logits = torch.randn((16, 12), dtype=torch.float32)
    targets = torch.randint(0, 12, size=(16,), dtype=torch.int64)
    targets[0] = 0  # rare transit class
    weights, ambig_idx, mean_ent = compute_citizen_science_entropy_weights(logits, top_ambiguous_fraction=0.25)
    assert weights.shape == (16,)
    assert len(ambig_idx) == 4
    assert mean_ent > 0.0

    fl_unweighted = asymmetric_focal_loss(logits, targets, gamma=2.0)
    fl_weighted = asymmetric_focal_loss(logits, targets, gamma=2.0, sample_weights=weights)
    assert float(fl_unweighted.item()) > 0.0
    assert float(fl_weighted.item()) > 0.0

    master = np.ones((8, 8), dtype=np.float32)
    vel = np.zeros((8, 8), dtype=np.float32)
    grad = np.full((8, 8), 0.1, dtype=np.float32)
    new_master, new_vel = apply_nesterov_outer_momentum(master, vel, grad, momentum=0.9, outer_lr=0.5)
    assert new_master.shape == (8, 8)
    assert new_vel.shape == (8, 8)
    assert float(np.mean(new_master)) < 1.0


def test_interactive_simulator_scenarios_and_end_to_end_poc(tmp_path: Path) -> None:
    """Verify the 3 interactive physical transit scenarios and full 4-stage PoC execution."""
    s_sun = synthesize_exoplanet_transit_and_router_response(1.0, 1.0, instrumental_noise_ppm=100.0)
    assert s_sun["transit_depth_ppm"] == 84.0
    assert s_sun["submerged_in_instrumental_noise"] is True
    assert s_sun["citizen_science_review_triggered"] is True

    s_mdwarf = synthesize_exoplanet_transit_and_router_response(1.0, 0.2, instrumental_noise_ppm=100.0)
    assert s_mdwarf["transit_depth_ppm"] == 2100.0
    assert s_mdwarf["submerged_in_instrumental_noise"] is False
    assert s_mdwarf["top1_expert"] == "expert_1_stellar_transits"
    assert s_mdwarf["top1_probability"] > 0.85

    s_eb = synthesize_exoplanet_transit_and_router_response(
        2.0,
        1.0,
        instrumental_noise_ppm=100.0,
        odd_even_asymmetry_ratio=0.5,
        centroid_shift_arcsec=0.9,
    )
    assert s_eb["top1_expert"] == "expert_2_false_positives_eb"

    out_file = tmp_path / "astro_poc_report.json"
    report = run_exoplanet_astro_vmoe_poc(
        inner_steps=40,
        focal_gamma=2.0,
        output_json_path=out_file,
    )
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["poc_id"] == "LA-RUCHE-EXOPLANET-ASTRO-VMOE-POC-V1"
    assert report["stage1_backbone_frozen"] is True
    assert report["stage1_strictly_under_1500mb_vram"] is True
    assert report["stage2_mean_focal_loss_reduction_pct"] > 50.0
    assert report["stage3_all_byzantine_rejected"] is True
    assert report["stage3_all_zk_opoi_verified"] is True
    assert report["stage4_routing_accuracy"] >= 0.90
    assert report["stage4_upcycled_vmoe_focal_loss"] < report["stage4_single_baseline_focal_loss"]
    assert report["overall_poc_passed"] is True
    assert len(report["report_sha256"]) == 64
