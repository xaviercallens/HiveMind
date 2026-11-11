# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the LLM-JEPA auxiliary objective (arXiv:2509.14252) in hivemind.swarm.llm_jepa."""

from __future__ import annotations

import math
import random

import pytest
import torch

from hivemind.swarm.llm_jepa import (
    JepaConfig,
    block_causal_mask,
    cotuned_lambda,
    embedding_distance,
    llm_jepa_loss,
    pack_two_views,
    relative_epoch_cost,
    view_gap_singular_values,
)


def test_block_causal_mask_isolates_views():
    m = block_causal_mask(6, [(0, 3), (3, 3)])
    # causal inside each block
    assert m[1, 0] == 0 and m[0, 1] == float("-inf")
    assert m[4, 3] == 0 and m[3, 4] == float("-inf")
    # no cross-view attention in either direction
    assert torch.isinf(m[3:, :3]).all() and torch.isinf(m[:3, 3:]).all()


def test_block_causal_mask_padding_rows_are_not_all_masked():
    m = block_causal_mask(5, [(0, 2), (2, 2)])
    assert m[4, 4] == 0.0
    assert not torch.isinf(m).all(dim=-1).any()


def test_block_causal_mask_rejects_bad_blocks():
    with pytest.raises(ValueError):
        block_causal_mask(4, [(0, 3), (2, 2)])
    with pytest.raises(ValueError):
        block_causal_mask(4, [(3, 2)])


def test_pack_two_views_positions():
    ids, mask, tl, cl = pack_two_views([[5, 6], [7]], [[8, 9, 10], [11, 12]], pred_token_id=3, k_pred=2, pad_id=0)
    assert ids.shape == (2, 7) and mask.shape == (2, 1, 7, 7)
    assert ids[0].tolist() == [5, 6, 3, 3, 8, 9, 10]
    assert tl.tolist() == [3, 2] and cl.tolist() == [6, 4]
    assert ids[1, 5:].tolist() == [0, 0]


def test_cosine_distance_zero_for_identical_and_two_for_opposite():
    x = torch.randn(4, 8)
    assert embedding_distance(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert embedding_distance(x, -x).item() == pytest.approx(2.0, abs=1e-6)


def test_cost_and_cotuning_rules_from_paper():
    assert relative_epoch_cost(0.0) == 2.0 and relative_epoch_cost(0.75) == 1.25
    lam = cotuned_lambda(1.0, 0.0, 0.75)
    assert lam * (1 - 0.75) == pytest.approx(1.0)


def test_config_validation():
    with pytest.raises(ValueError):
        JepaConfig(lam=-1)
    with pytest.raises(ValueError):
        JepaConfig(loss_dropout=1.0)
    with pytest.raises(ValueError):
        JepaConfig(metric="infonce")


def _tiny_lm():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    cfg._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(cfg)


def test_views_do_not_leak_through_mask():
    """Changing the Code view must not change the Text-side embedding (block-causal isolation in a real model)."""
    model = _tiny_lm().eval()
    text = [[4, 5, 6]]
    with torch.no_grad():
        outs = []
        for code in ([7, 8, 9], [20, 21, 22]):
            ids, mask, tl, _ = pack_two_views(text, [code], pred_token_id=3, k_pred=1)
            h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            outs.append(h[0, tl[0]])
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_llm_jepa_loss_adds_term_and_dropout_skips_it():
    model = _tiny_lm()
    lm = torch.tensor(1.5, requires_grad=True)
    total, log = llm_jepa_loss(lm, model, [[4, 5]], [[6, 7]], 3, JepaConfig(lam=2.0, k_pred=1))
    assert log["jepa_active"] == 1.0 and total.item() == pytest.approx(1.5 + 2.0 * log["jepa"], rel=1e-5)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    skipped, log2 = llm_jepa_loss(lm, model, [[4, 5]], [[6, 7]], 3, JepaConfig(lam=2.0, loss_dropout=0.999),
                                  rng=random.Random(0))
    assert log2["jepa_active"] == 0.0 and math.isnan(log2["jepa"]) and skipped is lm


def test_view_gap_singular_values_shape():
    s = view_gap_singular_values(torch.randn(10, 6), torch.randn(10, 6), top=3)
    assert len(s) == 3 and s[0] >= s[1] >= s[2] >= 0
