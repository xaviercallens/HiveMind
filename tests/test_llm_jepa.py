# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the LLM-JEPA auxiliary objective (arXiv:2509.14252) in hivemind.swarm.llm_jepa."""

from __future__ import annotations

import math
import random

import pytest
import torch

from hivemind.swarm.llm_jepa import (
    JepaConfig,
    _mask_dtype_for_model,
    block_causal_mask,
    cotuned_lambda,
    embedding_distance,
    evaluate_jepa_b3_gate,
    llm_jepa_loss,
    pack_two_views,
    relative_epoch_cost,
    view_gap_singular_values,
)


def test_block_causal_mask_isolates_views():
    m = block_causal_mask(6, [(0, 3), (3, 3)])
    assert m[1, 0] == 0 and m[0, 1] == float("-inf")
    assert m[4, 3] == 0 and m[3, 4] == float("-inf")
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


def test_pack_two_views_positions_and_validation():
    ids, mask, tl, cl = pack_two_views([[5, 6], [7]], [[8, 9, 10], [11, 12]], pred_token_id=3, k_pred=2, pad_id=0)
    assert ids.shape == (2, 7) and mask.shape == (2, 1, 7, 7)
    assert ids[0].tolist() == [5, 6, 3, 3, 8, 9, 10]
    assert tl.tolist() == [3, 2] and cl.tolist() == [6, 4]
    assert ids[1, 5:].tolist() == [0, 0]
    with pytest.raises(ValueError):
        pack_two_views([], [], pred_token_id=3, k_pred=1)
    with pytest.raises(ValueError):
        pack_two_views([[]], [[1]], pred_token_id=3, k_pred=1)


def test_embedding_distance_metrics_and_validation():
    x = torch.randn(4, 8)
    assert embedding_distance(x, x, "cosine").item() == pytest.approx(0.0, abs=1e-6)
    assert embedding_distance(x, -x, "cosine").item() == pytest.approx(2.0, abs=1e-6)
    assert embedding_distance(x, x, "mse").item() == pytest.approx(0.0, abs=1e-6)
    assert embedding_distance(x, x, "l2").item() == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        embedding_distance(x, torch.randn(4, 7))
    with pytest.raises(ValueError):
        embedding_distance(x, x, "infonce")


def test_cost_and_cotuning_rules_from_paper():
    assert relative_epoch_cost(0.0) == 2.0 and relative_epoch_cost(0.75) == 1.25
    lam = cotuned_lambda(1.0, 0.0, 0.75)
    assert lam * (1 - 0.75) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        relative_epoch_cost(1.5)
    with pytest.raises(ValueError):
        cotuned_lambda(-1.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        cotuned_lambda(1.0, 1.0, 0.5)


def test_config_validation():
    with pytest.raises(ValueError):
        JepaConfig(lam=-1)
    with pytest.raises(ValueError):
        JepaConfig(k_pred=-1)
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
    model = _tiny_lm()
    model.train(False)
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
    # lambda = 0 plumbing control (reproduces plain LM loss identically)
    lam0, log3 = llm_jepa_loss(lm, model, [[4, 5]], [[6, 7]], 3, JepaConfig(lam=0.0, k_pred=1))
    assert log3["jepa_active"] == 0.0 and lam0 is lm


def test_mask_dtype_fallback_for_integer_only_module():
    m = torch.nn.Module()
    m.register_buffer("buf", torch.zeros(2, dtype=torch.long))
    assert _mask_dtype_for_model(m) == torch.float32


def test_view_gap_singular_values_shape():
    s = view_gap_singular_values(torch.randn(10, 6), torch.randn(10, 6), top=3)
    assert len(s) == 3 and s[0] >= s[1] >= s[2] >= 0


def test_evaluate_jepa_b3_gate_diagnostic_never_overrides_nll():
    ok = evaluate_jepa_b3_gate([2.10, 2.08, 2.12, 2.09, 2.11], [1.95, 1.94, 1.97, 1.95, 1.96],
                               view_gap_sv1_ntp=120.0, view_gap_sv1_jepa=2.4)
    assert ok["servable"] is True and ok["bootstrap95"][0] > 0
    assert ok["diagnostic_only_view_gap"]["gating"] is False
    assert ok["diagnostic_only_view_gap"]["sv1_compression_ratio"] == pytest.approx(0.02)

    # Even with huge view-gap compression, a worse or noisy held-out NLL MUST be refused
    bad = evaluate_jepa_b3_gate([2.00, 2.01, 1.99, 2.00], [2.05, 2.04, 2.06, 2.03],
                                view_gap_sv1_ntp=150.0, view_gap_sv1_jepa=1.5)
    assert bad["servable"] is False and bad["mean_nll_gain_nats"] < 0

    with pytest.raises(ValueError):
        evaluate_jepa_b3_gate([1.0], [0.9])
    with pytest.raises(ValueError):
        evaluate_jepa_b3_gate([1.0, float("nan")], [0.9, 0.9])
