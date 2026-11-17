# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for FoMoE routing, Hybrid Distributor, Gallery Benchmark, and KV Page Codecs (PR-IP104)."""

from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from hivemind.core.fomoe_sigma_routing import (
    sigmoid_independent_topk_routing,
    skip_token_flops_per_token,
)
from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    PRODUCER_AND_HYBRID_FLEET,
    partition_swarm_shards_between_large_and_hybrid,
)
from hivemind.core.image_moe_inference_gallery_benchmark import (
    PublicImageScenarioSpec,
)
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    LatticePageCodec,
    PluggableKVPageCache,
)


def test_fomoe_sigma_routing_rejects_malformed_inputs() -> None:
    logits = torch.randn(4, 8)
    res = sigmoid_independent_topk_routing(logits, top_k=2)
    assert res.expert_indices.shape == (4, 2)

    with pytest.raises(ValueError, match="router_logits"):
        sigmoid_independent_topk_routing([[1.0, 2.0]], top_k=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resident_mask"):
        sigmoid_independent_topk_routing(logits, top_k=2, resident_mask=[True] * 8)  # type: ignore[arg-type]

    flops = skip_token_flops_per_token(
        dense_flops=100.0,
        ffn_flops_per_layer=[50.0, 50.0],
        gate_flops_per_layer=[2.0, 2.0],
        top_k=2,
        local_experts_per_layer=[4, 4],
        total_experts_per_layer=[8, 8],
    )
    assert flops["full_replica_flops_per_token"] > flops["partial_replica_flops_per_token"]

    with pytest.raises(ValueError, match="router_partitioned"):
        skip_token_flops_per_token(
            dense_flops=100.0,
            ffn_flops_per_layer=[50.0],
            gate_flops_per_layer=[2.0],
            top_k=2,
            local_experts_per_layer=[4],
            total_experts_per_layer=[8],
            router_partitioned=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="ffn_flops_per_layer"):
        skip_token_flops_per_token(
            dense_flops=100.0,
            ffn_flops_per_layer="50",  # type: ignore[arg-type]
            gate_flops_per_layer=[2.0, 2.0],
            top_k=2,
            local_experts_per_layer=[4, 4],
            total_experts_per_layer=[8, 8],
        )


def test_hybrid_distributor_spec_and_partition_reject_malformed_inputs() -> None:
    base_node = PRODUCER_AND_HYBRID_FLEET[0]
    with pytest.raises(ValueError, match="node_id"):
        replace(base_node, node_id=" ")
    with pytest.raises(ValueError, match="total_vram_mib"):
        replace(base_node, total_vram_mib=0.0)
    with pytest.raises(ValueError, match="total_vram_mib"):
        replace(base_node, total_vram_mib=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lora_rank"):
        replace(base_node, lora_rank=0)
    with pytest.raises(ValueError, match="throughput_weight"):
        replace(base_node, throughput_weight=True)  # type: ignore[arg-type]

    plans = partition_swarm_shards_between_large_and_hybrid()
    assert len(plans) == 7

    with pytest.raises(ValueError, match="fleet"):
        partition_swarm_shards_between_large_and_hybrid(fleet="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisions"):
        partition_swarm_shards_between_large_and_hybrid(divisions={})


def test_public_image_scenario_spec_rejects_malformed_inputs() -> None:
    base_scen = PublicImageScenarioSpec(
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
    assert base_scen.svg_data_uri().startswith("data:image/svg+xml;base64,")

    with pytest.raises(ValueError, match="scenario_id"):
        replace(base_scen, scenario_id="")
    with pytest.raises(ValueError, match="warm_latency_ms"):
        replace(base_scen, warm_latency_ms=-1.0)
    with pytest.raises(ValueError, match="tokens_per_sec"):
        replace(base_scen, tokens_per_sec=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grounding_confidence"):
        replace(base_scen, grounding_confidence=1.5)
    with pytest.raises(ValueError, match="http_status"):
        replace(base_scen, http_status=True)  # type: ignore[arg-type]


def test_kv_page_codecs_reject_non_tensor_inputs() -> None:
    page = torch.randn(4, 16)
    fp8 = FP8E4M3PageCodec()
    int4 = IntMinMaxPageCodec(bits=4)
    lat = LatticePageCodec(kind="E8", scale=0.25)

    assert fp8.decode(fp8.encode(page)).shape == page.shape
    assert int4.decode(int4.encode(page)).shape == page.shape
    assert lat.decode(lat.encode(page)).shape == page.shape

    for codec in (fp8, int4, lat):
        with pytest.raises(ValueError, match="page"):
            codec.encode([[1.0, 2.0]])  # type: ignore[arg-type]

    cache = PluggableKVPageCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    with pytest.raises(ValueError, match="k_tensor and v_tensor"):
        cache.write_kv_page(0, 0, [[1.0]], page)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_tensor and v_tensor"):
        cache.write_kv_page(0, 0, page, [[1.0]])  # type: ignore[arg-type]
