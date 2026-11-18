# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening tests for Iteration 184:
- src/hivemind/core/btm_moe.py
- src/hivemind/core/btm_upcycle.py
- src/hivemind/core/compression.py
- src/hivemind/core/dev_image_video_moe_integration_benchmark.py
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_moe import (
    FP8PagedKVCache,
    RegisteredDomainExpert,
    hotplug_hierarchical_experts,
    hotplug_three_tier_experts,
)
from hivemind.core.btm_upcycle import (
    LearnedDepthPrior,
    SparseMoEMLPBlock,
    TopKMoERouter,
    UpcycledSparseMoEModel,
    UpcycledSwiGLUExpert,
    _orient_lora_pair,
    svd_project_lora_pair,
)
from hivemind.core.compression import (
    _index_width,
    _pack_2bit_codes,
    _prepare_chunked_dct_matrix,
    _quantize_2bit_centroids,
    compress_tensor_dct_sparse_2bit,
    compress_tensor_topk_fp8,
    compute_pseudo_gradient_delta,
)
from hivemind.core.dev_image_video_moe_integration_benchmark import (
    DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS,
    END_USER_POST_ATTRIBUTIONS,
    REDDIT_POST_COHORT_FUNNELS,
)


def test_btm_moe_registered_domain_expert_kvcache_and_hotplug_validation() -> None:
    valid_expert = RegisteredDomainExpert(
        domain_name="Python_Systems",
        dataset_id="hivemind/python-systems",
        expert_cid="bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        shard_cid="bafybeihkoviema7g3gxyt6la7vd5ho32ictqbilu3wnlo3rs7ewhnp7lly",
        lora_rank_r=16,
        lora_alpha=32,
        initial_loss=2.1,
        final_loss=1.2,
        compression_ratio=16.5,
        compressed_bytes=4096,
        lora_state_dict={"a": torch.ones((4, 4))},
    )
    assert valid_expert.lora_rank_r == 16

    for field_name in ("expert_cid", "shard_cid"):
        with pytest.raises(ValueError, match="expert_cid and shard_cid"):
            dataclasses.replace(valid_expert, **{field_name: "   "})

    for loss_field in ("initial_loss", "final_loss"):
        for bad_loss in (-0.1, math.nan, math.inf, True):
            with pytest.raises(ValueError, match=loss_field):
                dataclasses.replace(valid_expert, **{loss_field: bad_loss})  # type: ignore[arg-type]

    for bad_cr in (0.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="compression_ratio"):
            dataclasses.replace(valid_expert, compression_ratio=bad_cr)  # type: ignore[arg-type]

    for bad_bytes in (-1, True, 3.5):
        with pytest.raises(ValueError, match="compressed_bytes"):
            dataclasses.replace(valid_expert, compressed_bytes=bad_bytes)  # type: ignore[arg-type]

    # FP8PagedKVCache boolean / non-finite tensor rejections
    bool_t = torch.ones((2, 8), dtype=torch.bool)
    nan_t = torch.full((2, 8), float("nan"))
    with pytest.raises(ValueError, match="non-empty torch.Tensor"):
        FP8PagedKVCache._quantize_page_fp8(bool_t)
    with pytest.raises(ValueError, match="finite"):
        FP8PagedKVCache._quantize_page_fp8(nan_t)
    with pytest.raises(ValueError, match="q and scale"):
        FP8PagedKVCache._dequantize_page_fp8(bool_t, torch.ones((2, 1)))
    with pytest.raises(ValueError, match="non-empty torch.Tensor"):
        FP8PagedKVCache._encode_decode_with_codec(bool_t, "fp8_e4m3")

    cache = FP8PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    with pytest.raises(ValueError, match="k_tensor and v_tensor must be non-empty"):
        cache.write_kv_page(0, 0, bool_t, torch.ones((2, 8)))
    with pytest.raises(ValueError, match="k_tensor and v_tensor must be non-empty torch.Tensors"):
        FP8PagedKVCache.benchmark_kv_page_codecs(bool_t, torch.ones((2, 8)))

    # hotplug_hierarchical_experts and hotplug_three_tier_experts parameter validation
    dummy_moe = UpcycledSparseMoEModel(nn.Linear(4, 4), ["Code"])
    for bad_int in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="num_macro_domains"):
            hotplug_hierarchical_experts(dummy_moe, {}, num_macro_domains=bad_int)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="experts_per_domain"):
            hotplug_hierarchical_experts(dummy_moe, {}, experts_per_domain=bad_int)  # type: ignore[arg-type]
        for tier_param in ("num_super_clusters", "macro_per_super", "micro_per_macro"):
            with pytest.raises(ValueError, match=tier_param):
                hotplug_three_tier_experts(dummy_moe, {}, **{tier_param: bad_int})  # type: ignore[arg-type]


def test_btm_upcycle_expert_router_model_and_depth_prior_validation() -> None:
    bool_2d = torch.ones((4, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="LoRA factors"):
        _orient_lora_pair(bool_2d, torch.ones((8, 4)))
    with pytest.raises(ValueError, match="lora_a and lora_b must be non-empty 2-D tensors"):
        svd_project_lora_pair(bool_2d, torch.ones((8, 4)), target_rank_r=2)

    g_w = torch.randn((8, 4))
    u_w = torch.randn((8, 4))
    d_w = torch.randn((4, 8))
    expert = UpcycledSwiGLUExpert(4, 8, g_w, u_w, d_w)
    out = expert(torch.randn((2, 4)))
    assert out.shape == (2, 4)

    with pytest.raises(ValueError, match="hidden_size and intermediate_size"):
        UpcycledSwiGLUExpert(True, 8, g_w, u_w, d_w)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gate_weight"):
        UpcycledSwiGLUExpert(4, 8, torch.ones((8, 4), dtype=torch.bool), u_w, d_w)
    with pytest.raises(ValueError, match="x must be a non-empty numeric"):
        expert(torch.ones((2, 4), dtype=torch.bool))

    router = TopKMoERouter(hidden_size=4, num_experts=3, top_k=2)
    rw, sel, aux = router(torch.randn((5, 4)))
    assert rw.shape == (5, 2) and sel.shape == (5, 2) and bool(torch.isfinite(aux).item())
    with pytest.raises(ValueError, match="hidden_states must be a non-empty 2-D numeric tensor"):
        router(torch.randn((5, 6)))
    with pytest.raises(ValueError, match="hidden_states must be a non-empty 2-D numeric tensor"):
        router(torch.ones((5, 4), dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        router(torch.full((5, 4), float("nan")))

    block = SparseMoEMLPBlock(hidden_size=4, experts=[expert], expert_domains=["Code"], top_k=1)
    with pytest.raises(ValueError, match="enabled must be a bool"):
        block.set_peri_ln(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expert_module must be an nn.Module"):
        block.hotplug_expert("Math", "not-a-module")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="upcycled_backbone must be an nn.Module"):
        UpcycledSparseMoEModel("not-a-module", ["Code"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expert_domains"):
        UpcycledSparseMoEModel(nn.Linear(4, 4), ["   "])
    with pytest.raises(ValueError, match="top_k"):
        UpcycledSparseMoEModel(nn.Linear(4, 4), ["Code"], top_k=True)  # type: ignore[arg-type]
    for bad_coef in (-0.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="router_aux_loss_coef"):
            UpcycledSparseMoEModel(nn.Linear(4, 4), ["Code"], router_aux_loss_coef=bad_coef)  # type: ignore[arg-type]
    moe = UpcycledSparseMoEModel(nn.Linear(4, 4), ["Code"])
    with pytest.raises(ValueError, match="enabled must be a bool"):
        moe.set_peri_ln("yes")  # type: ignore[arg-type]

    prior = LearnedDepthPrior(hidden_size=4, loop_steps=3)
    probs = prior.depth_probabilities(torch.zeros((2, 4)))
    assert probs.shape == (2, 3)
    with pytest.raises(ValueError, match="hidden_states"):
        prior.depth_probabilities(torch.zeros((2, 5)))
    with pytest.raises(ValueError, match="hidden_states"):
        prior.depth_probabilities(torch.ones((2, 4), dtype=torch.bool))
    with pytest.raises(ValueError, match="probs"):
        LearnedDepthPrior.expected_depth(torch.ones((2, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="probs"):
        LearnedDepthPrior.shannon_entropy(torch.ones((2, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="eps"):
        LearnedDepthPrior.shannon_entropy(probs, eps=0.0)
    for bad_budget in (0.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="target_budget_steps"):
            prior.regularization_loss(torch.zeros((2, 4)), target_budget_steps=bad_budget)  # type: ignore[arg-type]


def test_compression_boolean_tensor_and_helper_validation() -> None:
    bool_t = torch.ones((8, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="theta_init keys"):
        compute_pseudo_gradient_delta({"   ": torch.ones((4, 4))}, {"   ": torch.ones((4, 4))})
    with pytest.raises(ValueError, match="non-empty numeric torch.Tensor"):
        compute_pseudo_gradient_delta({"a": bool_t}, {"a": bool_t})
    with pytest.raises(ValueError, match="cannot compress an empty tensor"):
        compress_tensor_topk_fp8(bool_t)
    with pytest.raises(ValueError, match="tensor must be a numeric torch.Tensor"):
        compress_tensor_dct_sparse_2bit(bool_t)
    with pytest.raises(ValueError, match="cannot DCT-compress an empty tensor"):
        _prepare_chunked_dct_matrix(bool_t, chunk_size=8)
    with pytest.raises(ValueError, match="chunk_size"):
        _prepare_chunked_dct_matrix(torch.ones((8, 8)), chunk_size=True)  # type: ignore[arg-type]

    for bad_numel in (-1, True, 1.5):
        with pytest.raises(ValueError, match="numel must be a non-negative integer"):
            _index_width(bad_numel)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="codes must be a numeric torch.Tensor"):
        _pack_2bit_codes(torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="2-bit codes must be in"):
        _pack_2bit_codes(torch.tensor([0, 1, 4, 2], dtype=torch.int32))
    with pytest.raises(ValueError, match="values must be a non-empty finite numeric torch.Tensor"):
        _quantize_2bit_centroids(bool_t.reshape(-1), n_bins=4, range_in_sigmas=4.0)


def test_dev_image_video_moe_integration_benchmark_dataclass_validation() -> None:
    attr = END_USER_POST_ATTRIBUTIONS[0]
    for str_field in ("registered_utc", "hardware_profile", "post_archetype", "observed_behavior", "conversion_lesson"):
        with pytest.raises(ValueError, match=str_field):
            dataclasses.replace(attr, **{str_field: "   "})
    for bad_signals in ((), "not-a-seq", ("ok", "  ")):
        with pytest.raises(ValueError, match="attribution_signals"):
            dataclasses.replace(attr, attribution_signals=bad_signals)  # type: ignore[arg-type]

    funnel = REDDIT_POST_COHORT_FUNNELS[0]
    for str_field in ("top_converting_hook", "primary_friction_point", "remediation_applied"):
        with pytest.raises(ValueError, match=str_field):
            dataclasses.replace(funnel, **{str_field: ""})
    for rate_field in ("visitor_to_signup_rate_pct", "signup_to_active_rate_pct"):
        for bad_rate in (-0.1, 100.1, math.nan, math.inf, True):
            with pytest.raises(ValueError, match=rate_field):
                dataclasses.replace(funnel, **{rate_field: bad_rate})  # type: ignore[arg-type]
    for bad_vram in (-0.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="total_vram_contributed_gb"):
            dataclasses.replace(funnel, total_vram_contributed_gb=bad_vram)  # type: ignore[arg-type]

    scen = DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS[0]
    for str_field in ("target_persona", "baseline_model", "key_artifact_or_output"):
        with pytest.raises(ValueError, match=str_field):
            dataclasses.replace(scen, **{str_field: "  "})
    for bad_experts in ((), "not-a-seq", ("Bee1", "")):
        with pytest.raises(ValueError, match="active_experts"):
            dataclasses.replace(scen, active_experts=bad_experts)  # type: ignore[arg-type]
    for pos_field in (
        "baseline_quality_score",
        "moe_quality_score",
        "quality_gain_pct",
        "baseline_tokens_per_sec",
        "moe_tokens_per_sec",
    ):
        for bad_num in (0.0, -1.0, math.nan, math.inf, True):
            with pytest.raises(ValueError, match=pos_field):
                dataclasses.replace(scen, **{pos_field: bad_num})  # type: ignore[arg-type]
