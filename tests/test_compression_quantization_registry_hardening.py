"""Negative-control tests for compression, quantization, fomoe_sigma_routing, and expert_registry hardening."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from hivemind.core.compression import (
    CompressedTensorPayload,
    compress_lora_delta,
    compress_lora_delta_dct_sparse_2bit,
    compress_tensor_dct_sparse_2bit,
    compress_tensor_topk_fp8,
    compute_pseudo_gradient_delta,
    decompress_tensor_dct_sparse_2bit,
    decompress_tensor_topk_fp8,
)
from hivemind.core.expert_registry import (
    AdapterFootprint,
    ExpertRecord,
    SERVING,
    ServingNode,
    evaluate_wave3_promotion_gates,
)
from hivemind.core.fomoe_sigma_routing import (
    sigmoid_independent_topk_routing,
    skip_token_flops_per_token,
)
from hivemind.core.quantization import (
    create_lora_config,
    load_lora_state_dict,
    quantize_tensor_nf4_double_quant,
)


# ---------------------------------------------------------------------------
# 1. compression.py: compute_pseudo_gradient_delta, topk_fp8, dct_sparse_2bit, compress_lora_delta
# ---------------------------------------------------------------------------


def test_compute_pseudo_gradient_delta_rejects_shape_mismatch_and_non_finite() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_pseudo_gradient_delta({"a": torch.ones(4)}, {"a": torch.ones(5)})
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="NaN or inf"):
            compute_pseudo_gradient_delta({"a": torch.tensor([1.0, bad])}, {"a": torch.ones(2)})
        with pytest.raises(ValueError, match="NaN or inf"):
            compute_pseudo_gradient_delta({"a": torch.ones(2)}, {"a": torch.tensor([1.0, bad])})


def test_topk_fp8_and_dct_sparse_2bit_reject_invalid_payloads_and_ratios() -> None:
    sample = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(8, 8)
    with pytest.raises(ValueError, match="empty"):
        compress_tensor_topk_fp8(torch.empty(0))
    with pytest.raises(ValueError, match="topk_ratio"):
        compress_tensor_topk_fp8(sample, topk_ratio=True)  # type: ignore[arg-type]

    p_fp8 = compress_tensor_topk_fp8(sample, topk_ratio=0.25)
    with pytest.raises(ValueError, match="scale"):
        decompress_tensor_topk_fp8(dataclasses.replace(p_fp8, scale=0.0))
    with pytest.raises(ValueError, match="scale"):
        decompress_tensor_topk_fp8(dataclasses.replace(p_fp8, scale=math.nan))
    with pytest.raises(ValueError, match="out of bounds"):
        decompress_tensor_topk_fp8(
            CompressedTensorPayload(
                original_shape=p_fp8.original_shape,
                original_numel=p_fp8.original_numel,
                indices=torch.tensor([999], dtype=torch.int32),
                fp8_values=p_fp8.fp8_values[:1],
                scale=p_fp8.scale,
            )
        )

    # DCT 2-bit checks
    with pytest.raises(ValueError, match="topk_ratio"):
        compress_tensor_dct_sparse_2bit(sample, topk_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_size"):
        compress_tensor_dct_sparse_2bit(sample, chunk_size=True)  # type: ignore[arg-type]
    for bad_sig in (0.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="range_in_sigmas"):
            compress_tensor_dct_sparse_2bit(sample, range_in_sigmas=bad_sig)  # type: ignore[arg-type]

    p_dct = compress_tensor_dct_sparse_2bit(sample, topk_ratio=0.25)
    with pytest.raises(ValueError, match="shift and scale"):
        decompress_tensor_dct_sparse_2bit(dataclasses.replace(p_dct, scale=0.0))
    with pytest.raises(ValueError, match="shift and scale"):
        decompress_tensor_dct_sparse_2bit(dataclasses.replace(p_dct, shift=math.nan))
    bad_lookup = p_dct.lookup.clone()
    bad_lookup[0] = math.nan
    with pytest.raises(ValueError, match="lookup"):
        decompress_tensor_dct_sparse_2bit(dataclasses.replace(p_dct, lookup=bad_lookup))
    bad_idx = p_dct.indices.clone()
    bad_idx[0] = 9999
    with pytest.raises(ValueError, match="out of bounds"):
        decompress_tensor_dct_sparse_2bit(dataclasses.replace(p_dct, indices=bad_idx))


def test_compress_lora_delta_rejects_empty_ids_and_invalid_steps() -> None:
    d = {"w": torch.ones((8, 8), dtype=torch.float32)}
    for fn in (compress_lora_delta, compress_lora_delta_dct_sparse_2bit):
        with pytest.raises(ValueError, match="worker_id"):
            fn(d, "   ", "Code", 1)
        with pytest.raises(ValueError, match="branch_name"):
            fn(d, "w1", "", 1)
        with pytest.raises(ValueError, match="outer_step"):
            fn(d, "w1", "Code", -1)
        with pytest.raises(ValueError, match="outer_step"):
            fn(d, "w1", "Code", True)
        with pytest.raises(ValueError, match="inner_steps_completed"):
            fn(d, "w1", "Code", 1, inner_steps_completed=-1)
        with pytest.raises(ValueError, match="inner_steps_completed"):
            fn(d, "w1", "Code", 1, inner_steps_completed=True)


# ---------------------------------------------------------------------------
# 2. quantization.py: quantize_tensor_nf4_double_quant, create_lora_config, load_lora_state_dict
# ---------------------------------------------------------------------------


def test_quantization_rejects_non_finite_weights_and_invalid_lora_params() -> None:
    w = torch.randn(16, 16)
    for bad_bs in (0, -4, True):
        with pytest.raises(ValueError, match="block_size"):
            quantize_tensor_nf4_double_quant(w, block_size=bad_bs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        quantize_tensor_nf4_double_quant(torch.empty(0))
    for bad_val in (math.nan, math.inf):
        w_bad = w.clone()
        w_bad[0, 0] = bad_val
        with pytest.raises(ValueError, match="finite"):
            quantize_tensor_nf4_double_quant(w_bad)

    for bad_r in (0, -1, True):
        with pytest.raises(ValueError, match="rank_r"):
            create_lora_config(rank_r=bad_r)  # type: ignore[arg-type]
    for bad_a in (0, -1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="lora_alpha"):
            create_lora_config(lora_alpha=bad_a)  # type: ignore[arg-type]
    for bad_drop in (-0.1, 1.0, math.nan, True):
        with pytest.raises(ValueError, match="lora_dropout"):
            create_lora_config(lora_dropout=bad_drop)  # type: ignore[arg-type]

    model = nn.Linear(4, 4, bias=False)
    with pytest.raises(ValueError, match="finite"):
        load_lora_state_dict(model, {"weight": torch.full((4, 4), math.nan)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_lora_state_dict(model, {"weight": torch.ones((2, 2))})


# ---------------------------------------------------------------------------
# 3. fomoe_sigma_routing.py: sigmoid_independent_topk_routing & skip_token_flops_per_token
# ---------------------------------------------------------------------------


def test_fomoe_sigma_routing_rejects_non_finite_logits_and_flops() -> None:
    logits = torch.zeros((4, 8))
    with pytest.raises(ValueError, match="top_k"):
        sigmoid_independent_topk_routing(logits, top_k=True)
    logits_nan = logits.clone()
    logits_nan[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        sigmoid_independent_topk_routing(logits_nan, top_k=2)

    base_kw = dict(
        dense_flops=10.0,
        ffn_flops_per_layer=[20.0],
        gate_flops_per_layer=[2.0],
        top_k=1,
        local_experts_per_layer=[1],
        total_experts_per_layer=[2],
    )
    for bad in (math.nan, math.inf, True):
        with pytest.raises(ValueError, match="dense_flops"):
            skip_token_flops_per_token(**{**base_kw, "dense_flops": bad})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        skip_token_flops_per_token(**{**base_kw, "top_k": True})
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="FLOP counts"):
            skip_token_flops_per_token(**{**base_kw, "ffn_flops_per_layer": [bad]})
        with pytest.raises(ValueError, match="FLOP counts"):
            skip_token_flops_per_token(**{**base_kw, "gate_flops_per_layer": [bad]})
    with pytest.raises(ValueError, match="local experts"):
        skip_token_flops_per_token(**{**base_kw, "local_experts_per_layer": [True]})


# ---------------------------------------------------------------------------
# 4. expert_registry.py: evaluate_wave3_promotion_gates, ExpertRecord, AdapterFootprint, ServingNode
# ---------------------------------------------------------------------------


def test_expert_registry_wave3_gates_and_whitespace_id_validation() -> None:
    ok = evaluate_wave3_promotion_gates(0.85, 0.70, 0.65, 0.002, 0)
    assert ok["passed"] is True

    for bad_acc in (-0.1, 1.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="in_domain_acc"):
            evaluate_wave3_promotion_gates(bad_acc, 0.70, 0.65, 0.002, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mcnemar_p_value"):
            evaluate_wave3_promotion_gates(0.85, 0.70, 0.65, bad_acc, 0)  # type: ignore[arg-type]

    for bad_lift in (0.0, -0.1, 1.5, math.nan, True):
        with pytest.raises(ValueError, match="min_lift"):
            evaluate_wave3_promotion_gates(0.85, 0.70, 0.65, 0.002, 0, min_lift=bad_lift)  # type: ignore[arg-type]

    for bad_wv in (-1, True, False):
        with pytest.raises(ValueError, match="wrong_verified_count"):
            evaluate_wave3_promotion_gates(0.85, 0.70, 0.65, 0.002, bad_wv)

    with pytest.raises(ValueError, match="non-empty"):
        ExpertRecord("   ", SERVING, "code", "Qwen/Qwen3-4B")
    with pytest.raises(ValueError, match="non-empty"):
        ExpertRecord("exp1", SERVING, "   ", "Qwen/Qwen3-4B")
    with pytest.raises(ValueError, match="non-empty"):
        ExpertRecord("exp1", SERVING, "code", "   ")
    with pytest.raises(ValueError, match="non-empty"):
        AdapterFootprint(size_mib=32, base_model_id="   ")
    with pytest.raises(ValueError, match="non-empty"):
        ServingNode(node_id="   ", vram_mib=8192, base_model_id="Qwen", base_resident_mib=4096, max_lora_slots=4)
