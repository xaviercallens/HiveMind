"""Negative-control and invariant tests for swat_telepathy_bridge, reddit_vision_vlm_moe, exoplanet_astro_vmoe_poc, and image_moe_inference_gallery_benchmark functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hivemind.core.exoplanet_astro_vmoe_poc import project_cwt_scalogram_to_visual_patches
from hivemind.core.image_moe_inference_gallery_benchmark import build_public_image_moe_gallery_scenarios
from hivemind.core.reddit_vision_vlm_moe import (
    DHTDeltaRegistry,
    FrozenVisionLanguageBackbone,
    build_cluster_synthetic_shard,
    dequantize_adamw_state_8bit,
    generate_zk_opoi_training_receipt,
    run_diloco_qlora_worker_training,
    run_la_ruche_1500mb_patch_vmoe_pipeline,
    verify_zk_opoi_training_receipt,
)
from hivemind.swarm.swat_telepathy_bridge import (
    compute_cross_causal_limits,
    cross_attention_with_causal_limits,
    normalized_levenshtein_similarity,
)


def test_swat_telepathy_bridge_functions_validation() -> None:
    limits = compute_cross_causal_limits([(0, 4), (4, 8)], [(0, 5), (5, 8)])
    np.testing.assert_array_equal(limits, [1, 2])

    with pytest.raises(ValueError, match="sender_offsets and receiver_offsets must be sequences"):
        compute_cross_causal_limits("bad", [(0, 4)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sender_offsets must contain non-negative"):
        compute_cross_causal_limits([(-1, 4)], [(0, 4)])
    with pytest.raises(ValueError, match="receiver_offsets must contain non-negative"):
        compute_cross_causal_limits([(0, 4)], [(5, 2)])

    q = np.ones((2, 4), dtype=np.float64)
    k = np.ones((2, 4), dtype=np.float64)
    v = np.ones((2, 4), dtype=np.float64)
    out = cross_attention_with_causal_limits(q, k, v, limits)
    assert out.shape == (2, 4)

    with pytest.raises(ValueError, match="receiver_queries must be a finite 2D numpy ndarray"):
        cross_attention_with_causal_limits(np.ones(4), k, v, limits)
    with pytest.raises(ValueError, match="sender_keys and sender_values must match"):
        cross_attention_with_causal_limits(q, np.ones((2, 3)), v, limits)
    with pytest.raises(ValueError, match="causal_limits must be a 1D numpy ndarray of length seq_r"):
        cross_attention_with_causal_limits(q, k, v, np.array([1], dtype=np.int64))

    assert normalized_levenshtein_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    with pytest.raises(ValueError, match="seq_a must be a str or Sequence"):
        normalized_levenshtein_similarity(b"bytes", "str")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seq_b elements must be integers"):
        normalized_levenshtein_similarity([1, 2], [True, 2])


def test_reddit_vision_vlm_moe_functions_validation() -> None:
    q8 = np.array([10, -20], dtype=np.int8)
    deq = dequantize_adamw_state_8bit(q8, 0.05, torch.device("cpu"))
    assert deq.shape == (2,)

    with pytest.raises(ValueError, match="q8 must be a non-empty numpy ndarray"):
        dequantize_adamw_state_8bit(np.array([], dtype=np.int8), 0.05, torch.device("cpu"))
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        dequantize_adamw_state_8bit(q8, 0.0, torch.device("cpu"))
    with pytest.raises(ValueError, match="device must be a torch.device or non-empty device string"):
        dequantize_adamw_state_8bit(q8, 0.05, "")  # type: ignore[arg-type]

    vis, txt, lbl = build_cluster_synthetic_shard("expert_1_aesthetics_art", num_samples=8)
    assert vis.shape == (8, 4, 48) and txt.shape == (8, 4, 48) and lbl.shape == (8,)

    with pytest.raises(ValueError, match="unknown cluster_id"):
        build_cluster_synthetic_shard("unknown_cluster")
    with pytest.raises(ValueError, match="bee_index must be a non-negative integer"):
        build_cluster_synthetic_shard("expert_1_aesthetics_art", bee_index=-1)
    with pytest.raises(ValueError, match="num_samples must be an integer >= 1"):
        build_cluster_synthetic_shard("expert_1_aesthetics_art", num_samples=0)
    with pytest.raises(ValueError, match="vision_dim must be an integer >= 38"):
        build_cluster_synthetic_shard("expert_1_aesthetics_art", vision_dim=16)

    backbone = FrozenVisionLanguageBackbone()
    dht = DHTDeltaRegistry()
    bee_meta = {"bee_id": "bee_vlm_01", "backbone_key": "qwen25-vl-3b", "hf_dataset": "laion/aesthetics"}

    with pytest.raises(ValueError, match="backbone must be a FrozenVisionLanguageBackbone"):
        run_diloco_qlora_worker_training("not_backbone", bee_meta, "expert_1_aesthetics_art", dht)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bee_meta must be a Mapping containing bee_id and backbone_key"):
        run_diloco_qlora_worker_training(backbone, {}, "expert_1_aesthetics_art", dht)
    with pytest.raises(ValueError, match="dht_registry must be a DHTDeltaRegistry"):
        run_diloco_qlora_worker_training(backbone, bee_meta, "expert_1_aesthetics_art", "not_dht")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps must be an integer >= 1"):
        run_diloco_qlora_worker_training(backbone, bee_meta, "expert_1_aesthetics_art", dht, inner_steps=0)

    d0 = np.ones((4, 4), dtype=np.float32)
    d1 = np.ones((4, 4), dtype=np.float32) * 0.5
    rcpt = generate_zk_opoi_training_receipt("bee_01", "art_aesthetics", 10, "Muon", 42, d0, d1, "a" * 64)
    assert verify_zk_opoi_training_receipt(rcpt, d1) is True

    with pytest.raises(ValueError, match="bee_id must be a non-empty string"):
        generate_zk_opoi_training_receipt("", "art_aesthetics", 10, "Muon", 42, d0, d1, "a" * 64)
    with pytest.raises(ValueError, match="initial_delta must be a non-empty finite numpy ndarray"):
        generate_zk_opoi_training_receipt("bee_01", "art_aesthetics", 10, "Muon", 42, np.array([]), d1, "a" * 64)
    with pytest.raises(ValueError, match="receipt must be a ZKOPoITrainingReceipt"):
        verify_zk_opoi_training_receipt("not_receipt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected_final_delta must be a non-empty finite numpy ndarray"):
        verify_zk_opoi_training_receipt(rcpt, np.array([math.nan]))

    with pytest.raises(ValueError, match="inner_steps must be an integer >= 1"):
        run_la_ruche_1500mb_patch_vmoe_pipeline(inner_steps=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        run_la_ruche_1500mb_patch_vmoe_pipeline(inner_steps=2, seed=True)  # type: ignore[arg-type]


def test_exoplanet_cwt_projection_and_image_gallery_validation() -> None:
    scalo = np.ones((8, 16), dtype=np.float32)
    flux = np.linspace(-200.0, 0.0, 16, dtype=np.float32)
    patches = project_cwt_scalogram_to_visual_patches(scalo, flux, snr_single_transit=8.5)
    assert patches.shape == (1, 4, 48)

    with pytest.raises(ValueError, match="scalogram must be a non-empty finite 2D numpy ndarray"):
        project_cwt_scalogram_to_visual_patches(np.ones(8), flux, 8.5)
    with pytest.raises(ValueError, match="flux_ppm must be a non-empty finite 1D numpy ndarray"):
        project_cwt_scalogram_to_visual_patches(scalo, np.ones((2, 2)), 8.5)
    with pytest.raises(ValueError, match="snr_single_transit must be a finite non-negative number"):
        project_cwt_scalogram_to_visual_patches(scalo, flux, -1.0)
    with pytest.raises(ValueError, match="patch_dim must be an integer >= 40"):
        project_cwt_scalogram_to_visual_patches(scalo, flux, 8.5, patch_dim=16)

    with pytest.raises(ValueError, match="output_gallery_dir must be a non-empty Path or string"):
        build_public_image_moe_gallery_scenarios(output_gallery_dir="   ")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_gallery_dir must be a non-empty Path or string"):
        build_public_image_moe_gallery_scenarios(output_gallery_dir=123)  # type: ignore[arg-type]
