# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary tests for V-MoE astro, massive, hybrid, and reddit modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.core.exoplanet_astro_vmoe_poc import (
    FrozenAstroVisionBackbone,
    SwarmFFNAdapter,
    asymmetric_focal_loss,
    compute_citizen_science_entropy_weights,
    compute_physical_transit_depth_ppm,
    generate_phase_folded_cwt_scalogram,
    run_exoplanet_astro_vmoe_poc,
)
from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    PRODUCER_AND_HYBRID_FLEET,
    execute_hybrid_node_inference_and_micro_training,
    execute_large_gpu_producer_training,
    run_hybrid_inference_and_large_gpu_image_moe_campaign,
)
from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MassiveSwarmSparseImageMoE,
    SwarmDataShardOrchestrator,
    build_massive_division_visual_shard,
    run_massive_swarm_image_moe_campaign,
)
from hivemind.core.reddit_vision_vlm_moe import (
    ByzantineMultiKrumAggregator,
    newton_schulz_orthogonalize_muon,
    quantize_nf4_blockwise,
    quantize_sparseloco_dct_2bit,
    run_nocturnal_reddit_vision_moe_campaign,
)


class TestExoplanetAstroVMoEHardening:
    """Negative-control tests for exoplanet_astro_vmoe_poc.py."""

    @pytest.mark.parametrize("bad_val", [True, False, 0.0, -1.0, float("nan"), float("inf"), "1.0"])
    def test_compute_physical_transit_depth_ppm_rejects_invalid_radii(self, bad_val: object) -> None:
        with pytest.raises(ValueError, match="planet_radius_earth"):
            compute_physical_transit_depth_ppm(bad_val, 1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="star_radius_solar"):
            compute_physical_transit_depth_ppm(1.0, bad_val)  # type: ignore[arg-type]

    def test_generate_phase_folded_cwt_scalogram_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="transit_depth_ppm"):
            generate_phase_folded_cwt_scalogram(transit_depth_ppm=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="instrumental_noise_ppm"):
            generate_phase_folded_cwt_scalogram(transit_depth_ppm=84.0, instrumental_noise_ppm=-1.0)
        with pytest.raises(ValueError, match="odd_even_asymmetry_ratio"):
            generate_phase_folded_cwt_scalogram(
                transit_depth_ppm=84.0, odd_even_asymmetry_ratio=float("nan")
            )
        with pytest.raises(ValueError, match="num_phase_bins"):
            generate_phase_folded_cwt_scalogram(transit_depth_ppm=84.0, num_phase_bins=0)
        with pytest.raises(ValueError, match="num_scales"):
            generate_phase_folded_cwt_scalogram(transit_depth_ppm=84.0, num_scales=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed"):
            generate_phase_folded_cwt_scalogram(transit_depth_ppm=84.0, seed=-1)

    def test_asymmetric_focal_loss_and_citizen_entropy_reject_invalid_params(self) -> None:
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        with pytest.raises(ValueError, match="gamma"):
            asymmetric_focal_loss(logits, targets, gamma=-0.1)
        with pytest.raises(ValueError, match="gamma"):
            asymmetric_focal_loss(logits, targets, gamma=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="rare_transit_weight"):
            asymmetric_focal_loss(logits, targets, rare_transit_weight=0.0)
        with pytest.raises(ValueError, match="top_ambiguous_fraction"):
            compute_citizen_science_entropy_weights(logits, top_ambiguous_fraction=0.0)
        with pytest.raises(ValueError, match="top_ambiguous_fraction"):
            compute_citizen_science_entropy_weights(logits, top_ambiguous_fraction=1.5)
        with pytest.raises(ValueError, match="citizen_boost"):
            compute_citizen_science_entropy_weights(logits, citizen_boost=0.0)

    def test_adapter_backbone_and_poc_runner_reject_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            SwarmFFNAdapter(hidden_dim=0)
        with pytest.raises(ValueError, match="alpha"):
            SwarmFFNAdapter(alpha=0.0)
        with pytest.raises(ValueError, match="seed"):
            SwarmFFNAdapter(seed=-1)
        with pytest.raises(ValueError, match="patch_dim"):
            FrozenAstroVisionBackbone(patch_dim=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="inner_steps"):
            run_exoplanet_astro_vmoe_poc(inner_steps=0)
        with pytest.raises(ValueError, match="focal_gamma"):
            run_exoplanet_astro_vmoe_poc(inner_steps=2, focal_gamma=-1.0)
        with pytest.raises(ValueError, match="seed"):
            run_exoplanet_astro_vmoe_poc(inner_steps=2, seed=True)  # type: ignore[arg-type]


class TestMassiveImageMoESwarmOrchestratorHardening:
    """Negative-control tests for massive_image_moe_swarm_orchestrator.py."""

    def test_assign_shard_to_bee_rejects_invalid_inputs(self) -> None:
        orch = SwarmDataShardOrchestrator()
        with pytest.raises(ValueError, match="bee_id"):
            orch.assign_shard_to_bee(bee_id="  ")
        with pytest.raises(ValueError, match="bee_id"):
            orch.assign_shard_to_bee(bee_id="bee\x00")
        with pytest.raises(ValueError, match="backend"):
            orch.assign_shard_to_bee(bee_id="bee_1", backend="")
        with pytest.raises(ValueError, match="available_vram_mib"):
            orch.assign_shard_to_bee(bee_id="bee_1", available_vram_mib=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="available_vram_mib"):
            orch.assign_shard_to_bee(bee_id="bee_1", available_vram_mib=float("nan"))
        with pytest.raises(ValueError, match="round_idx"):
            orch.assign_shard_to_bee(bee_id="bee_1", round_idx=-1)

    def test_build_massive_division_visual_shard_and_moe_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="Unknown division_id"):
            build_massive_division_visual_shard(division_id="invalid_div")
        with pytest.raises(ValueError, match="num_samples"):
            build_massive_division_visual_shard(
                division_id="div_1_aesthetics_digital_art", num_samples=0
            )
        with pytest.raises(ValueError, match="seed"):
            build_massive_division_visual_shard(
                division_id="div_1_aesthetics_digital_art", seed=-5
            )
        backbone = FrozenAstroVisionBackbone(patch_dim=48, hidden_dim=64, intermediate_dim=128)
        with pytest.raises(ValueError, match="division_deltas"):
            MassiveSwarmSparseImageMoE(backbone=backbone, division_deltas={})
        deltas = {"div_1_aesthetics_digital_art": np.zeros((64, 64), dtype=np.float32)}
        with pytest.raises(ValueError, match="top_k"):
            MassiveSwarmSparseImageMoE(backbone=backbone, division_deltas=deltas, top_k=0)
        with pytest.raises(ValueError, match="load_balance_alpha"):
            MassiveSwarmSparseImageMoE(
                backbone=backbone, division_deltas=deltas, load_balance_alpha=-0.1
            )

    def test_run_massive_swarm_image_moe_campaign_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="inner_steps"):
            run_massive_swarm_image_moe_campaign(inner_steps=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="focal_gamma"):
            run_massive_swarm_image_moe_campaign(inner_steps=2, focal_gamma=float("inf"))
        with pytest.raises(ValueError, match="seed"):
            run_massive_swarm_image_moe_campaign(inner_steps=2, seed=-1)


class TestHybridInferenceAndLargeGPUDistributorHardening:
    """Negative-control tests for hybrid_inference_and_large_gpu_image_moe_distributor.py."""

    def test_execute_large_gpu_and_hybrid_reject_invalid_params(self) -> None:
        large_node = next(
            n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "LARGE_GPU_DEDICATED_TRAINER"
        )
        hybrid_node = next(
            n
            for n in PRODUCER_AND_HYBRID_FLEET
            if n.pool_role == "HYBRID_INFERENCE_AND_MICRO_TRAINER"
        )
        with pytest.raises(ValueError, match="assigned_shards"):
            execute_large_gpu_producer_training(
                node=large_node, assigned_shards=0, assigned_divisions=["div_1_aesthetics_digital_art"]
            )
        with pytest.raises(ValueError, match="inner_steps"):
            execute_large_gpu_producer_training(
                node=large_node,
                assigned_shards=2,
                assigned_divisions=["div_1_aesthetics_digital_art"],
                inner_steps=True,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="focal_gamma"):
            execute_large_gpu_producer_training(
                node=large_node,
                assigned_shards=2,
                assigned_divisions=["div_1_aesthetics_digital_art"],
                inner_steps=2,
                focal_gamma=-0.5,
            )
        with pytest.raises(ValueError, match="incoming_inference_requests"):
            execute_hybrid_node_inference_and_micro_training(
                node=hybrid_node, incoming_inference_requests=-1
            )
        with pytest.raises(ValueError, match="hot_swapped_adapters_count"):
            execute_hybrid_node_inference_and_micro_training(
                node=hybrid_node, hot_swapped_adapters_count=True  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="inner_steps"):
            run_hybrid_inference_and_large_gpu_image_moe_campaign(inner_steps=0)


class TestRedditVisionVLMMoEHardening:
    """Negative-control tests for reddit_vision_vlm_moe.py."""

    def test_quantize_nf4_and_sparseloco_reject_invalid_inputs(self) -> None:
        w = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="block_size"):
            quantize_nf4_blockwise(w, block_size=0)
        with pytest.raises(ValueError, match="block_size"):
            quantize_nf4_blockwise(w, block_size=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="weight"):
            quantize_nf4_blockwise(np.array([[float("nan")]], dtype=np.float32))

        with pytest.raises(ValueError, match="bee_id"):
            quantize_sparseloco_dct_2bit(w, bee_id=" ", cluster_id="c1", backbone_id="b1")
        with pytest.raises(ValueError, match="sparse_topk_ratio"):
            quantize_sparseloco_dct_2bit(
                w, bee_id="b", cluster_id="c", backbone_id="m", sparse_topk_ratio=0.0
            )
        with pytest.raises(ValueError, match="delta_matrix"):
            quantize_sparseloco_dct_2bit(
                np.ones((8,), dtype=np.float32), bee_id="b", cluster_id="c", backbone_id="m"
            )

    def test_aggregator_muon_and_campaign_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="replay_fraction"):
            ByzantineMultiKrumAggregator(replay_fraction=0.0)
        with pytest.raises(ValueError, match="min_cosine_sim"):
            ByzantineMultiKrumAggregator(min_cosine_sim=1.5)
        with pytest.raises(ValueError, match="max_l2_drift"):
            ByzantineMultiKrumAggregator(max_l2_drift=-0.1)

        g = np.eye(4, dtype=np.float32)
        with pytest.raises(ValueError, match="steps"):
            newton_schulz_orthogonalize_muon(g, steps=0)
        with pytest.raises(ValueError, match="eps"):
            newton_schulz_orthogonalize_muon(g, eps=0.0)
        with pytest.raises(ValueError, match="grad"):
            newton_schulz_orthogonalize_muon(np.array([[float("inf")]], dtype=np.float32))

        with pytest.raises(ValueError, match="inner_steps"):
            run_nocturnal_reddit_vision_moe_campaign(inner_steps=0)
        with pytest.raises(ValueError, match="lora_rank"):
            run_nocturnal_reddit_vision_moe_campaign(inner_steps=2, lora_rank=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lora_alpha"):
            run_nocturnal_reddit_vision_moe_campaign(inner_steps=2, lora_alpha=0.0)
        with pytest.raises(ValueError, match="seed"):
            run_nocturnal_reddit_vision_moe_campaign(inner_steps=2, seed=-1)
