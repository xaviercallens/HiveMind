# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control tests for KV lattice, lattice quantizer, BTM MoE, and BTM upcycle hardening (PR-IP101)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import (
    from_rows,
    integer_coordinates,
    nearest_dn_t,
    nearest_zn_t,
)
from hivemind.compress.lattice_quantizer import (
    factorised_rate_bits_per_dim,
    nearest_dn,
    nearest_zn,
)
from hivemind.core.btm_moe import (
    hotplug_hierarchical_experts,
    hotplug_three_tier_experts,
)
from hivemind.core.btm_upcycle import (
    detect_lora_rank,
    pair_lora_ab_keys,
    train_moe_router_gating,
)


def test_kv_lattice_rejects_invalid_inputs() -> None:
    empty_t = torch.empty((0, 8), dtype=torch.float64)
    nan_t = torch.tensor([[1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    inf_t = torch.tensor([[1.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    for fn in (nearest_zn_t, nearest_dn_t):
        for bad in (None, [1.0, 2.0], empty_t, nan_t, inf_t):
            with pytest.raises(ValueError, match="x must be a non-empty finite torch.Tensor"):
                fn(bad)  # type: ignore[arg-type]

    for bad in (None, [1.0, 2.0], empty_t, nan_t, inf_t):
        with pytest.raises(ValueError, match="c must be a non-empty finite torch.Tensor"):
            integer_coordinates("Z", bad)  # type: ignore[arg-type]

    for bad in (None, [1.0, 2.0], empty_t):
        with pytest.raises(ValueError, match="rows must be a non-empty torch.Tensor"):
            from_rows("Z", bad, torch.Size([0, 8]))  # type: ignore[arg-type]


def test_lattice_quantizer_rejects_invalid_inputs() -> None:
    empty_arr = np.empty((0, 8), dtype=float)
    nan_arr = np.array([[1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    inf_arr = np.array([[1.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)

    for fn in (nearest_zn, nearest_dn):
        for bad in (empty_arr, nan_arr, inf_arr):
            with pytest.raises(ValueError, match="x must be a non-empty finite array"):
                fn(bad)

    for bad in (empty_arr, nan_arr, inf_arr):
        with pytest.raises(ValueError, match="codewords must be a non-empty finite array"):
            factorised_rate_bits_per_dim("Z", bad)


def test_btm_moe_hotplug_rejects_invalid_parameters() -> None:
    for fn in (hotplug_hierarchical_experts, hotplug_three_tier_experts):
        for bad_adapters in (None, [], "invalid", 123):
            with pytest.raises(ValueError, match="community_adapters must be a dict"):
                fn(moe_model=None, community_adapters=bad_adapters)  # type: ignore[arg-type]
        for bad_rank in (True, False, 0, -4, 2.5, "16"):
            with pytest.raises(ValueError, match="canonical_rank_r must be a positive integer"):
                fn(moe_model=None, community_adapters={}, canonical_rank_r=bad_rank)  # type: ignore[arg-type]
        for bad_topk in (True, False, 0, -1, 1.5, "2"):
            with pytest.raises(ValueError, match="top_k must be a positive integer"):
                fn(moe_model=None, community_adapters={}, top_k=bad_topk)  # type: ignore[arg-type]
        for bad_vram in (True, False, 0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="vram_ceiling_gb must be a positive finite float"):
                fn(moe_model=None, community_adapters={}, vram_ceiling_gb=bad_vram)  # type: ignore[arg-type]


def test_btm_upcycle_rejects_invalid_inputs() -> None:
    for fn in (pair_lora_ab_keys, detect_lora_rank):
        for bad in (None, [], "invalid", 42):
            with pytest.raises(ValueError, match="lora_state_dict must be a dict"):
                fn(bad)  # type: ignore[arg-type]

    for bad_batches in (None, {}, [], "invalid"):
        with pytest.raises(ValueError, match="domain_token_batches must be a non-empty dict"):
            train_moe_router_gating(None, bad_batches)  # type: ignore[arg-type]

    valid_batches = {"math": torch.ones((1, 8), dtype=torch.long)}
    for bad_steps in (True, False, 0, -5, 1.5, "40"):
        with pytest.raises(ValueError, match="steps must be a positive integer"):
            train_moe_router_gating(None, valid_batches, steps=bad_steps)  # type: ignore[arg-type]

    for bad_lr in (True, False, 0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lr must be a positive finite float"):
            train_moe_router_gating(None, valid_batches, lr=bad_lr)  # type: ignore[arg-type]
