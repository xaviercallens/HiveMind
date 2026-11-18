"""Regression tests for IP-186 boundary validation across fomoe_sigma_routing, hybrid_inference_and_large_gpu_image_moe_distributor, image_moe_inference_gallery_benchmark, and kv_page_codecs."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hivemind.core.fomoe_sigma_routing import (
    SigmaMoERoutingResult,
    sigmoid_independent_topk_routing,
    skip_token_flops_per_token,
)
from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    PRODUCER_AND_HYBRID_FLEET,
    execute_large_gpu_producer_training,
    partition_swarm_shards_between_large_and_hybrid,
    run_hybrid_inference_and_large_gpu_image_moe_campaign,
)
from hivemind.core.image_moe_inference_gallery_benchmark import (
    PublicImageScenarioSpec,
    run_intensive_image_moe_inference_stress_campaign,
)
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    LatticePageCodec,
    PluggableKVPageCache,
)


def test_fomoe_sigma_routing_ip186_guards() -> None:
    logits = torch.tensor([[0.5, -0.2, 1.1, 0.0], [-0.4, 0.9, 0.2, -1.2]])
    res = sigmoid_independent_topk_routing(logits, top_k=2)
    assert res.expert_indices.shape == (2, 2)

    with pytest.raises(ValueError, match="router_logits must be"):
        sigmoid_independent_topk_routing(torch.tensor([[True, False]]), top_k=1)

    with pytest.raises(ValueError, match="expert_indices must be a non-empty 2D integer"):
        SigmaMoERoutingResult(
            expert_indices=torch.tensor([[0.5, 1.5], [1.0, 2.0]]),
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.0,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=1.0,
        )

    with pytest.raises(ValueError, match="gate_values must match expert_indices.shape and contain finite"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=torch.tensor([[float("nan"), 0.5], [0.5, 0.5]]),
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.0,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=1.0,
        )

    with pytest.raises(ValueError, match="skipped_slots must match expert_indices.shape and have bool dtype"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=torch.zeros((2, 2), dtype=torch.int64),
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.0,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=1.0,
        )

    with pytest.raises(ValueError, match="FLOP counts must be >= 0"):
        skip_token_flops_per_token(
            dense_flops=10.0,
            ffn_flops_per_layer=[True],  # type: ignore[list-item]
            gate_flops_per_layer=[1.0],
            top_k=1,
            local_experts_per_layer=[2],
            total_experts_per_layer=[4],
        )

    with pytest.raises(ValueError, match="ghost fractions must be in"):
        skip_token_flops_per_token(
            dense_flops=10.0,
            ffn_flops_per_layer=[10.0],
            gate_flops_per_layer=[1.0],
            top_k=1,
            local_experts_per_layer=[2],
            total_experts_per_layer=[4],
            empirical_ghost_fraction_per_layer=[True],  # type: ignore[list-item]
        )


def test_hybrid_distributor_ip186_guards() -> None:
    node = PRODUCER_AND_HYBRID_FLEET[0]
    with pytest.raises(ValueError, match="os_reserved_vram_mib must be a non-negative finite number"):
        replace(node, os_reserved_vram_mib=-1.0)

    with pytest.raises(ValueError, match="inference_reserved_vram_mib must be a non-negative finite number"):
        replace(node, inference_reserved_vram_mib=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="served_inference_models must be a tuple of non-empty strings"):
        replace(node, served_inference_models=("hivemind/coder", "  "))

    with pytest.raises(ValueError, match="assigned_training_backbones must be a tuple of non-empty strings"):
        replace(node, assigned_training_backbones=["bad"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="fleet must be a sequence of GPUProducerNodeSpec"):
        partition_swarm_shards_between_large_and_hybrid(fleet=[node, "not-a-node"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="assigned_divisions must be a sequence"):
        execute_large_gpu_producer_training(node, assigned_shards=1, assigned_divisions=[], inner_steps=1)

    with pytest.raises(ValueError, match="save_report_path must be None or a non-empty Path/str"):
        run_hybrid_inference_and_large_gpu_image_moe_campaign(inner_steps=1, save_report_path="   ")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="update_producer_registry_path must be None or a non-empty Path/str"):
        run_hybrid_inference_and_large_gpu_image_moe_campaign(inner_steps=1, update_producer_registry_path="")  # type: ignore[arg-type]


def test_gallery_benchmark_ip186_guards() -> None:
    scen = PublicImageScenarioSpec(
        scenario_id="IMG-SCEN-01",
        division_id="div_1_aesthetics_digital_art",
        division_title="Division 1",
        category="image_moe_gallery",
        badge_label="AESTHETICS",
        title_fr="Titre FR",
        title_en="Title EN",
        public_source_label="Wikimedia Commons",
        license_spdx="CC-BY-SA-4.0",
        svg_filename="scen01.svg",
        model_id="hivemind/vision-moe",
        primary_expert_bee="bee-aesthetics-1",
        secondary_expert_bee="bee-aesthetics-2",
        backbone_tag="siglip-base",
        served_by_hybrid_node="HYBRID-CLOUD-TESLA-T4-NODE1",
        image_resolution="640x360",
        rgb_mean_channels=(0.5, 0.4, 0.3),
        top1_routed_division="div_1_aesthetics_digital_art",
        top1_gate_weight=0.72,
        top2_routed_division="div_3_technical_infographics",
        top2_gate_weight=0.18,
        warm_latency_ms=132.4,
        tokens_per_sec=44.2,
        active_vram_mib=1380.0,
        gpu_residency="warm",
        grounding_confidence=0.94,
        http_status=200,
        prompt_fr="Prompt FR",
        response_excerpt_fr="Response FR",
        when_to_use_fr="When to use FR",
        how_to_invoke="curl ...",
        svg_markup="<svg></svg>",
    )
    assert scen.to_dict()["scenario_id"] == "IMG-SCEN-01"

    with pytest.raises(ValueError, match="rgb_mean_channels must be a 3-tuple of finite numbers in \\[0\\.0, 1\\.0\\]"):
        replace(scen, rgb_mean_channels=(0.5, 1.5, 0.3))

    with pytest.raises(ValueError, match="top1_gate_weight must be a finite number in \\[0\\.0, 1\\.0\\]"):
        replace(scen, top1_gate_weight=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="primary_expert_bee must be a non-empty string"):
        replace(scen, primary_expert_bee="   ")

    with pytest.raises(ValueError, match="jpg_filename must be a string"):
        replace(scen, jpg_filename=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="save_report_path must be None or a non-empty Path/str"):
        run_intensive_image_moe_inference_stress_campaign(concurrent_requests_per_node=2, save_report_path="  ")  # type: ignore[arg-type]


def test_kv_page_codecs_ip186_guards() -> None:
    bool_page = torch.tensor([[True, False, True, False, True, False, True, False]])
    fp8 = FP8E4M3PageCodec()
    int4 = IntMinMaxPageCodec(4)
    lat = LatticePageCodec("E8", 0.25)

    for codec in (fp8, int4, lat):
        with pytest.raises(ValueError, match="page"):
            codec.encode(bool_page)

    with pytest.raises(ValueError, match="payload must be a 2-tuple"):
        fp8.decode((torch.empty(0), torch.ones(1)))

    with pytest.raises(ValueError, match="payload must be a 3-tuple"):
        int4.decode((torch.empty(0), torch.ones(1), torch.ones(1)))

    with pytest.raises(ValueError, match="payload must be a dict containing 'codewords', 'rms', and 'shape'"):
        lat.decode({})

    with pytest.raises(ValueError, match="payload must be a dict containing 'rate', 'coords', and 'rms'"):
        lat.payload_bits({})

    cache = PluggableKVPageCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    with pytest.raises(ValueError, match="k_tensor and v_tensor must be non-empty tensors"):
        cache.write_kv_page(0, 0, bool_page, bool_page)
