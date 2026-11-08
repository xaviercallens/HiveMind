# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary tests for gallery, dev/video MoE benchmark, and swarm_fabric."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.dev_image_video_moe_integration_benchmark import (
    DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS,
    _model_concurrent_modality_stress,
    run_intense_dev_image_video_moe_benchmark,
)
from hivemind.core.image_moe_inference_gallery_benchmark import (
    _svg_to_rgb_tensor,
    build_trained_7division_image_moe,
    run_intensive_image_moe_inference_stress_campaign,
)
from hivemind.swarm.swarm_fabric import (
    HiveLayer2StateChannel,
    OptimisticProofOfInferenceAuditor,
    PinnedRAMExpertDMAStreamer,
    RegionalGeoRingTransport,
    TokenTreeSpeculativeVerifier,
    TwoLevelMacroMicroRouter,
)


class TestDevImageVideoMoEIntegrationBenchmarkHardening:
    """Negative-control tests for dev_image_video_moe_integration_benchmark.py."""

    def test_model_concurrent_modality_stress_rejects_invalid_inputs(self) -> None:
        scenarios = list(DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS[:2])
        with pytest.raises(ValueError, match="modality"):
            _model_concurrent_modality_stress("", scenarios)
        with pytest.raises(ValueError, match="modality"):
            _model_concurrent_modality_stress("mod\x00", scenarios)
        with pytest.raises(ValueError, match="scenarios"):
            _model_concurrent_modality_stress("developer_coding_moe", [])
        with pytest.raises(ValueError, match="num_requests"):
            _model_concurrent_modality_stress("developer_coding_moe", scenarios, num_requests=0)
        with pytest.raises(ValueError, match="num_requests"):
            _model_concurrent_modality_stress(
                "developer_coding_moe", scenarios, num_requests=True  # type: ignore[arg-type]
            )

    def test_run_intense_dev_image_video_moe_benchmark_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="real_requests_executed"):
            run_intense_dev_image_video_moe_benchmark(real_requests_executed=-1)
        with pytest.raises(ValueError, match="real_requests_executed"):
            run_intense_dev_image_video_moe_benchmark(
                real_requests_executed=True  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="scenarios"):
            run_intense_dev_image_video_moe_benchmark(scenarios=[])
        dev_only = [
            s for s in DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS if s.modality == "developer_coding_moe"
        ]
        with pytest.raises(ValueError, match="developer_coding_moe"):
            run_intense_dev_image_video_moe_benchmark(scenarios=dev_only)


class TestImageMoEInferenceGalleryBenchmarkHardening:
    """Negative-control tests for image_moe_inference_gallery_benchmark.py."""

    def test_svg_to_rgb_tensor_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="svg_markup"):
            _svg_to_rgb_tensor("   ", 0)
        with pytest.raises(ValueError, match="division_index"):
            _svg_to_rgb_tensor("<svg></svg>", -1)
        with pytest.raises(ValueError, match="division_index"):
            _svg_to_rgb_tensor("<svg></svg>", True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="height"):
            _svg_to_rgb_tensor("<svg></svg>", 0, height=0)
        with pytest.raises(ValueError, match="width"):
            _svg_to_rgb_tensor("<svg></svg>", 0, width=False)  # type: ignore[arg-type]

    def test_build_trained_moe_and_stress_campaign_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            build_trained_7division_image_moe(seed=-1, inner_steps=2)
        with pytest.raises(ValueError, match="inner_steps"):
            build_trained_7division_image_moe(seed=42, inner_steps=0)
        with pytest.raises(ValueError, match="inner_steps"):
            build_trained_7division_image_moe(seed=42, inner_steps=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="concurrent_requests_per_node"):
            run_intensive_image_moe_inference_stress_campaign(concurrent_requests_per_node=0)
        with pytest.raises(ValueError, match="concurrent_requests_per_node"):
            run_intensive_image_moe_inference_stress_campaign(
                concurrent_requests_per_node=True  # type: ignore[arg-type]
            )


class TestSwarmFabricHardening:
    """Negative-control tests for swarm_fabric.py."""

    def test_regional_geo_ring_transport_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="region"):
            RegionalGeoRingTransport(region="  ")
        with pytest.raises(ValueError, match="max_rtt_ms"):
            RegionalGeoRingTransport(max_rtt_ms=0.0)
        with pytest.raises(ValueError, match="max_rtt_ms"):
            RegionalGeoRingTransport(max_rtt_ms=True)  # type: ignore[arg-type]
        ring = RegionalGeoRingTransport()
        with pytest.raises(ValueError, match="target_key"):
            ring.find_closest_peers("")
        with pytest.raises(ValueError, match="k"):
            ring.find_closest_peers("expert_a", k=0)
        with pytest.raises(ValueError, match="k"):
            ring.find_closest_peers("expert_a", k=True)  # type: ignore[arg-type]

    def test_router_dma_streamer_and_tree_verifier_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            TwoLevelMacroMicroRouter(hidden_size=0, macro_clusters={"c1": ["e1"]})
        with pytest.raises(ValueError, match="top_k_micro"):
            TwoLevelMacroMicroRouter(
                hidden_size=16, macro_clusters={"c1": ["e1"]}, top_k_micro=False  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="macro_clusters"):
            TwoLevelMacroMicroRouter(hidden_size=16, macro_clusters={})
        with pytest.raises(ValueError, match="macro_clusters"):
            TwoLevelMacroMicroRouter(hidden_size=16, macro_clusters={"c1": []})

        with pytest.raises(ValueError, match="max_pinned_ram_bytes"):
            PinnedRAMExpertDMAStreamer(max_pinned_ram_bytes=0)
        with pytest.raises(ValueError, match="max_pinned_ram_bytes"):
            PinnedRAMExpertDMAStreamer(max_pinned_ram_bytes=True)  # type: ignore[arg-type]
        streamer = PinnedRAMExpertDMAStreamer()
        with pytest.raises(ValueError, match="expert_name"):
            streamer.stage_expert_in_pinned_ram("", {"w": torch.ones(2, 2)})
        with pytest.raises(ValueError, match="lora_state_dict"):
            streamer.stage_expert_in_pinned_ram("exp1", {})

        with pytest.raises(ValueError, match="gamma"):
            TokenTreeSpeculativeVerifier(gamma=0)
        with pytest.raises(ValueError, match="branching_factor"):
            TokenTreeSpeculativeVerifier(gamma=6, branching_factor=False)  # type: ignore[arg-type]

    def test_opoi_auditor_and_l2_state_channel_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="audit_probability"):
            OptimisticProofOfInferenceAuditor(audit_probability=-0.1)
        with pytest.raises(ValueError, match="audit_probability"):
            OptimisticProofOfInferenceAuditor(audit_probability=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="min_cosine_sim"):
            OptimisticProofOfInferenceAuditor(min_cosine_sim=1.2)
        with pytest.raises(ValueError, match="max_normalized_linf"):
            OptimisticProofOfInferenceAuditor(max_normalized_linf=-0.01)

        with pytest.raises(ValueError, match="consumer_id"):
            HiveLayer2StateChannel(consumer_id=" ", provider_ids=["p1"])
        with pytest.raises(ValueError, match="provider_ids"):
            HiveLayer2StateChannel(consumer_id="c1", provider_ids=[])
        with pytest.raises(ValueError, match="escrow_hive"):
            HiveLayer2StateChannel(
                consumer_id="c1", provider_ids=["p1"], escrow_hive=True  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="rate_per_token"):
            HiveLayer2StateChannel(
                consumer_id="c1", provider_ids=["p1"], rate_per_token=False  # type: ignore[arg-type]
            )
        chan = HiveLayer2StateChannel(consumer_id="c1", provider_ids=["p1"])
        with pytest.raises(ValueError, match="tokens_generated"):
            chan.sign_micro_voucher(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="tokens_generated"):
            chan.sign_micro_voucher(0)
