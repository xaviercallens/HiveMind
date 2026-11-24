# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""router_lr_scale for the MoE inner steps: 1 is unchanged, 0 freezes router gates and final LayerNorm only."""

from __future__ import annotations

import copy

import pytest
import torch

from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    router_and_final_ln_parameters,
    run_crowd_moe_opendiloco_inner_steps,
    upcycle_crowd_v1_to_sparse_moe,
)


def _moe(seed: int = 20261124) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    m = CrowdV1WithEngram(CrowdV1RuntimeConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                               num_key_value_heads=2, head_dim=8, intermediate_size=64,
                                               max_position_embeddings=64, engram_buckets=64))
    upcycle_crowd_v1_to_sparse_moe(m, num_experts=4, top_k=2, lora_rank=2)
    # LoRA B starts at zero: give the experts a non-zero delta so the router receives a gradient
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad and "_B" in n:
                p.add_(0.05)
    return m


def _stream(n: int = 200) -> torch.Tensor:
    return torch.tensor([(i * 5 + 7) % 64 for i in range(n)], dtype=torch.long)


def _run(m: CrowdV1WithEngram, scale: float) -> dict:
    torch.manual_seed(0)
    return run_crowd_moe_opendiloco_inner_steps(m, _stream(), steps=3, seq_len=16, lr=1e-2, num_router_fragments=2,
                                                router_lr_scale=scale)


def test_scale_one_is_the_original_optimizer() -> None:
    a, b = _moe(), _moe()
    out_default = run_crowd_moe_opendiloco_inner_steps(a, _stream(), steps=3, seq_len=16, lr=1e-2, num_router_fragments=2)
    out_one = _run(b, 1.0)
    for k in out_default["pseudo_grads"]:
        assert torch.equal(out_default["pseudo_grads"][k], out_one["pseudo_grads"][k]), k


def test_scale_zero_freezes_router_and_ln_but_keeps_the_layout() -> None:
    m = _moe()
    before = extract_moe_trainable_buckets(m, num_router_fragments=2)
    out = _run(m, 0.0)
    pg = out["pseudo_grads"]
    assert set(pg) == set(before)  # router_frag_* buckets still present
    for k, v in pg.items():
        if k.startswith("router_frag_"):
            assert torch.count_nonzero(v) == 0, k
    # top-2 of 4: an expert the router never picks on this stream gets no update; the experts as a whole do
    assert sum(int(torch.count_nonzero(v)) for k, v in pg.items() if k.startswith("expert_")) > 0
    assert all(p.requires_grad for p in router_and_final_ln_parameters(m))  # restored after the steps
    assert out["steps_completed"] == 3


def test_partial_scale_moves_router_less() -> None:
    full, small = _run(_moe(), 1.0)["pseudo_grads"], _run(_moe(), 0.1)["pseudo_grads"]
    rk = [k for k in full if k.startswith("router_frag_")]
    assert sum(float(full[k].norm()) for k in rk) > sum(float(small[k].norm()) for k in rk) > 0.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_bad_scale_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="router_lr_scale"):
        _run(_moe(), bad)


def test_router_params_are_the_router_frag_contents() -> None:
    m = _moe()
    n_router = sum(p.numel() for p in router_and_final_ln_parameters(m))
    b = extract_moe_trainable_buckets(m, num_router_fragments=2)
    assert n_router == sum(v.numel() for k, v in b.items() if k.startswith("router_frag_"))
    assert copy.deepcopy(m) is not m
