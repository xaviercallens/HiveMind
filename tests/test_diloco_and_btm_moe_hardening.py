"""Negative-control tests for diloco_inner, diloco_outer, and btm_moe hardening."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_moe import (
    FP8PagedKVCache,
    HierarchicalMoERouter,
    SEASON_3_DOMAIN_EXPERT_SPECS,
    ThreeTierHierarchicalMoERouter,
    keyword_route,
)
from hivemind.core.diloco_inner import (
    dequantize_tensor_blockwise_8bit,
    get_deterministic_batch,
    quantize_tensor_blockwise_8bit,
    replay_inner_segment,
    run_diloco_inner_loop,
)
from hivemind.core.diloco_outer import (
    DiLoCoOuterOptimizer,
    filter_byzantine_deltas_multikrum,
    flatten_delta_dict,
)


# ---------------------------------------------------------------------------
# 1. diloco_inner.py: blockwise 8-bit quantization, deterministic batch, inner loop & replay
# ---------------------------------------------------------------------------


def test_quantize_and_dequantize_blockwise_8bit_rejects_invalid_inputs() -> None:
    sample = torch.linspace(-2.0, 2.0, 64, dtype=torch.float32)
    q, absmax, sh = quantize_tensor_blockwise_8bit(sample, block_size=64, signed=True)
    recon = dequantize_tensor_blockwise_8bit(q, absmax, sh, signed=True)
    assert recon.shape == sh

    for bad_bs in (0, -8, True):
        with pytest.raises(ValueError, match="block_size"):
            quantize_tensor_blockwise_8bit(sample, block_size=bad_bs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        quantize_tensor_blockwise_8bit(torch.empty(0))

    for bad_val in (math.nan, math.inf, -math.inf):
        bad_t = sample.clone()
        bad_t[0] = bad_val
        with pytest.raises(ValueError, match="finite"):
            quantize_tensor_blockwise_8bit(bad_t)

    # dequantize shape & absmax validation
    with pytest.raises(ValueError, match="2-D"):
        dequantize_tensor_blockwise_8bit(q.reshape(-1), absmax, sh)
    with pytest.raises(ValueError, match="absmax"):
        dequantize_tensor_blockwise_8bit(q, torch.tensor([0.0]), sh)
    with pytest.raises(ValueError, match="absmax"):
        dequantize_tensor_blockwise_8bit(q, torch.tensor([math.nan]), sh)
    with pytest.raises(ValueError, match="orig_shape requires"):
        dequantize_tensor_blockwise_8bit(q, absmax, torch.Size([128]))


def test_get_deterministic_batch_rejects_invalid_sequences_and_batch_size() -> None:
    seqs = torch.ones((4, 8), dtype=torch.long)
    b, lbl = get_deterministic_batch(seqs, step=1, rng_seed=42, micro_batch_size=2, device=torch.device("cpu"))
    assert b.shape == (2, 8) and torch.equal(b, lbl)

    with pytest.raises(ValueError, match="ndim >= 2"):
        get_deterministic_batch(torch.ones(8, dtype=torch.long), 1, 42, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="non-empty"):
        get_deterministic_batch(torch.empty((0, 8), dtype=torch.long), 1, 42, 1, torch.device("cpu"))
    for bad_mb in (0, -1, True):
        with pytest.raises(ValueError, match="micro_batch_size"):
            get_deterministic_batch(seqs, 1, 42, bad_mb, torch.device("cpu"))  # type: ignore[arg-type]


def test_run_diloco_inner_loop_and_replay_reject_invalid_parameters() -> None:
    model = nn.Linear(4, 4)
    seqs = torch.ones((2, 4), dtype=torch.long)

    with pytest.raises(ValueError, match="worker_id"):
        run_diloco_inner_loop(model, seqs, "   ", "Code")
    with pytest.raises(ValueError, match="branch_name"):
        run_diloco_inner_loop(model, seqs, "w1", "")
    with pytest.raises(ValueError, match="outer_step"):
        run_diloco_inner_loop(model, seqs, "w1", "Code", outer_step=-1)
    with pytest.raises(ValueError, match="outer_step"):
        run_diloco_inner_loop(model, seqs, "w1", "Code", outer_step=True)
    with pytest.raises(ValueError, match="inner_steps_H"):
        run_diloco_inner_loop(model, seqs, "w1", "Code", inner_steps_H=0)
    with pytest.raises(ValueError, match="checkpoint_interval_s"):
        run_diloco_inner_loop(model, seqs, "w1", "Code", checkpoint_interval_s=0)
    for bad_lr in (0.0, -1e-4, math.nan, math.inf):
        with pytest.raises(ValueError, match="lr"):
            run_diloco_inner_loop(model, seqs, "w1", "Code", lr=bad_lr)
    for bad_wd in (-0.01, math.nan, math.inf):
        with pytest.raises(ValueError, match="weight_decay"):
            run_diloco_inner_loop(model, seqs, "w1", "Code", weight_decay=bad_wd)
    for bad_topk in (0.0, -0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="topk_ratio"):
            run_diloco_inner_loop(model, seqs, "w1", "Code", topk_ratio=bad_topk)

    # replay_inner_segment checks
    with pytest.raises(ValueError, match="start_step"):
        replay_inner_segment(model, {}, seqs, start_step=-1, end_step=2)
    with pytest.raises(ValueError, match="end_step"):
        replay_inner_segment(model, {}, seqs, start_step=2, end_step=2)
    with pytest.raises(ValueError, match="lr"):
        replay_inner_segment(model, {}, seqs, start_step=0, end_step=2, lr=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        replay_inner_segment(model, {}, seqs, start_step=0, end_step=2, weight_decay=-0.1)


# ---------------------------------------------------------------------------
# 2. diloco_outer.py: flatten_delta_dict, filter_byzantine_deltas_multikrum, DiLoCoOuterOptimizer
# ---------------------------------------------------------------------------


def test_diloco_outer_rejects_empty_deltas_and_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="empty"):
        flatten_delta_dict({})

    deltas = {"w1": {"p": torch.ones(4)}, "w2": {"p": torch.ones(4)}}
    for bad_f in (-1, True):
        with pytest.raises(ValueError, match="max_byzantine_f"):
            filter_byzantine_deltas_multikrum(deltas, max_byzantine_f=bad_f)  # type: ignore[arg-type]
    for bad_sig in (-0.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="cosine_sigma"):
            filter_byzantine_deltas_multikrum(deltas, cosine_sigma=bad_sig)  # type: ignore[arg-type]

    # DiLoCoOuterOptimizer constructor must validate even in default nesterov mode without cpu flat bucket
    for bad_lr in (0.0, -0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="outer_lr_eta"):
            DiLoCoOuterOptimizer(outer_lr_eta=bad_lr)
    for bad_mu in (-0.1, 1.0, 1.5, math.nan):
        with pytest.raises(ValueError, match="outer_momentum_mu"):
            DiLoCoOuterOptimizer(outer_momentum_mu=bad_mu)
    for bad_f in (-1, True):
        with pytest.raises(ValueError, match="max_byzantine_f"):
            DiLoCoOuterOptimizer(max_byzantine_f=bad_f)  # type: ignore[arg-type]
    for bad_sig in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="cosine_sigma"):
            DiLoCoOuterOptimizer(cosine_sigma=bad_sig)

    opt = DiLoCoOuterOptimizer()
    with pytest.raises(ValueError, match="branch_name"):
        opt.step("   ", 1, {"p": torch.ones(4)}, deltas)
    with pytest.raises(ValueError, match="outer_step"):
        opt.step("Code", -1, {"p": torch.ones(4)}, deltas)
    with pytest.raises(ValueError, match="outer_step"):
        opt.step("Code", True, {"p": torch.ones(4)}, deltas)


# ---------------------------------------------------------------------------
# 3. btm_moe.py: keyword_route, HierarchicalMoERouter, ThreeTierHierarchicalMoERouter, FP8PagedKVCache
# ---------------------------------------------------------------------------


def test_keyword_route_and_hierarchical_routers_reject_invalid_parameters() -> None:
    for bad_k in (0, -1, True):
        with pytest.raises(ValueError, match="top_k"):
            keyword_route("rust borrow checker", (SEASON_3_DOMAIN_EXPERT_SPECS,), top_k=bad_k)  # type: ignore[arg-type]
    for bad_w in (-0.1, 1.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="min_weight"):
            keyword_route("rust borrow checker", (SEASON_3_DOMAIN_EXPERT_SPECS,), min_weight=bad_w)  # type: ignore[arg-type]

    for bad_h in (0, -4, True):
        with pytest.raises(ValueError, match="hidden_size"):
            HierarchicalMoERouter(hidden_size=bad_h)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="hidden_size"):
            ThreeTierHierarchicalMoERouter(hidden_size=bad_h)  # type: ignore[arg-type]

    for bad_k in (0, -2, True):
        with pytest.raises(ValueError, match="top_k"):
            HierarchicalMoERouter(hidden_size=8, top_k=bad_k)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="top_k"):
            ThreeTierHierarchicalMoERouter(hidden_size=8, top_k=bad_k)  # type: ignore[arg-type]


def test_fp8_paged_kv_cache_rejects_non_finite_rates_and_invalid_pages() -> None:
    for bad_bits in (math.nan, math.inf, True):
        with pytest.raises(ValueError, match="target_bits_per_dim"):
            FP8PagedKVCache(target_bits_per_dim=bad_bits)  # type: ignore[arg-type]
    for param in ("num_layers", "num_kv_heads", "head_dim", "page_size_tokens", "max_pages"):
        with pytest.raises(ValueError, match=param):
            FP8PagedKVCache(**{param: 0})
        with pytest.raises(ValueError, match=param):
            FP8PagedKVCache(**{param: True})  # type: ignore[arg-type]

    cache = FP8PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4, max_pages=4)
    k_ok = torch.randn(2, 4, 8)
    v_ok = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="layer_idx"):
        cache.write_kv_page(-1, 0, k_ok, v_ok)
    with pytest.raises(ValueError, match="page_idx"):
        cache.write_kv_page(0, -1, k_ok, v_ok)
    with pytest.raises(ValueError, match="non-empty"):
        cache.write_kv_page(0, 0, torch.empty(0), v_ok)
    k_nan = k_ok.clone()
    k_nan[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        cache.write_kv_page(0, 0, k_nan, v_ok)
