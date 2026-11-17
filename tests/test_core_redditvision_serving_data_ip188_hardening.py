# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for Iteration 188:
- `hivemind.core.reddit_vision_vlm_moe`
- `hivemind.core.serving_experts`
- `hivemind.data.dataset_mining_campaign`
- `hivemind.data.dataset_registry`
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from hivemind.core.reddit_vision_vlm_moe import (
    ByzantineMultiKrumAggregator,
    FrozenVisionLanguageBackbone,
    QLoRAVisionAdapter,
    UpcycledRedditVisionSparseMoE,
    quantize_sparseloco_dct_2bit,
)
from hivemind.core.serving_experts import load_gwaya_code_expert
from hivemind.data import dataset_registry as dr
from hivemind.data.dataset_mining_campaign import (
    _compute_max_cross_split_4gram_jaccard,
    _compute_shannon_entropy_bits,
)


def test_reddit_vision_dct_packet_and_low_dim_lora_adapter_hardening() -> None:
    mat = np.ones((8, 8), dtype=np.float32) * 0.25
    pkt = quantize_sparseloco_dct_2bit(
        mat,
        bee_id="bee-vlm-01",
        cluster_id="expert_1_aesthetics_art",
        backbone_id="qwen2.5-vl-3b",
    )

    for bad_shape in ((8,), (0, 8), (8, -1), (True, 8), "8x8"):
        with pytest.raises(ValueError, match="shape must be a 2-tuple of positive integers"):
            replace(pkt, shape=bad_shape)  # type: ignore[arg-type]

    for bad_q in (
        (-1.0, 0.0, 1.0),
        (-1.0, -0.3, 0.3, float("nan")),
        (False, True, 2.0, 3.0),
        "levels",
    ):
        with pytest.raises(
            ValueError,
            match="q_levels must be a 4-tuple of finite numbers",
        ):
            replace(pkt, q_levels=bad_q)  # type: ignore[arg-type]

    for bad_scale in (
        [1.0] * 7,
        [0.0] + [1.0] * 7,
        [-0.1] + [1.0] * 7,
        [float("inf")] + [1.0] * 7,
        [True] + [1.0] * 7,
        "scales",
    ):
        with pytest.raises(
            ValueError,
            match="scale_row must be a list of positive finite numbers matching shape\\[0\\]",
        ):
            replace(pkt, scale_row=bad_scale)  # type: ignore[arg-type]

    # Low hidden_dim < rank SVD truncation positive control
    adapter = QLoRAVisionAdapter(hidden_dim=8, rank=16, alpha=32.0, seed=42)
    rng = np.random.default_rng(7)
    delta_8x8 = rng.normal(0.0, 0.05, size=(8, 8)).astype(np.float32)
    adapter.load_effective_delta(delta_8x8)
    recon = adapter.compute_effective_delta()
    assert recon.shape == (8, 8)
    assert np.allclose(recon, delta_8x8, atol=1e-4)

    # Reject boolean delta matrix
    with pytest.raises(ValueError, match="delta_matrix must be a finite \\(8, 8\\) numpy ndarray"):
        adapter.load_effective_delta(np.ones((8, 8), dtype=bool))


def test_reddit_vision_multi_krum_and_upcycled_moe_hardening() -> None:
    backbone = FrozenVisionLanguageBackbone(vision_dim=48, hidden_dim=8, num_classes=18, seed=11)
    agg = ByzantineMultiKrumAggregator()
    good_delta = np.ones((8, 8), dtype=np.float32) * 0.01

    with pytest.raises(ValueError, match="cluster_id must be a non-empty string"):
        agg.aggregate_cluster_deltas("  ", [good_delta], [], backbone)

    with pytest.raises(TypeError, match="backbone must be a FrozenVisionLanguageBackbone"):
        agg.aggregate_cluster_deltas("expert_1_aesthetics_art", [good_delta], [], "not-backbone")  # type: ignore[arg-type]

    for bad_honest in ([], "deltas", b"deltas"):
        with pytest.raises(ValueError, match="honest_deltas must be a non-empty sequence"):
            agg.aggregate_cluster_deltas("expert_1_aesthetics_art", bad_honest, [], backbone)  # type: ignore[arg-type]

    for bad_byz in ("deltas", b"deltas", 123):
        with pytest.raises(ValueError, match="byzantine_deltas must be a sequence"):
            agg.aggregate_cluster_deltas("expert_1_aesthetics_art", [good_delta], bad_byz, backbone)  # type: ignore[arg-type]

    for bad_arr in (
        np.ones((8, 8), dtype=bool),
        np.ones((4, 4), dtype=np.float32),
        np.full((8, 8), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="honest_deltas\\[0\\] must be a finite numeric numpy ndarray"):
            agg.aggregate_cluster_deltas("expert_1_aesthetics_art", [bad_arr], [], backbone)
        with pytest.raises(ValueError, match="byzantine_deltas\\[0\\] must be a finite numeric numpy ndarray"):
            agg.aggregate_cluster_deltas("expert_1_aesthetics_art", [good_delta], [bad_arr], backbone)

    rep = agg.aggregate_cluster_deltas(
        "expert_1_aesthetics_art",
        [good_delta, good_delta * 1.02, good_delta * 0.98],
        [good_delta * 50.0],
        backbone,
    )
    assert rep.consensus_passed is True

    with pytest.raises(TypeError, match="backbone must be a FrozenVisionLanguageBackbone"):
        UpcycledRedditVisionSparseMoE("not-backbone", {"e1": good_delta})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expert_deltas must be a non-empty mapping"):
        UpcycledRedditVisionSparseMoE(backbone, {})
    with pytest.raises(ValueError, match="expert_deltas keys must be non-empty strings"):
        UpcycledRedditVisionSparseMoE(backbone, {"  ": good_delta})
    for bad_topk in (0, -1, True, False):
        with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
            UpcycledRedditVisionSparseMoE(backbone, {"e1": good_delta}, top_k=bad_topk)  # type: ignore[arg-type]

    moe = UpcycledRedditVisionSparseMoE(
        backbone,
        {"expert_1_aesthetics_art": good_delta, "expert_2_social_meme_culture": good_delta * 0.5},
        top_k=2,
    )
    out = moe(torch.randn(2, 4, 48), torch.randn(2, 4, 48))
    assert out["logits"].shape == (2, 18)


def test_serving_experts_load_gwaya_code_expert_rejects_control_chars_and_bad_schemes() -> None:
    for bad_url in ("ftp://example.org", "javascript:alert(1)", "http://localhost:11434\nX-Injected: 1"):
        with pytest.raises(ValueError, match="large_url must be a non-empty http\\(s\\) URL string"):
            load_gwaya_code_expert(large_url=bad_url)
        with pytest.raises(
            ValueError,
            match="small_url must be a non-empty http\\(s\\) URL string when provided",
        ):
            load_gwaya_code_expert(large_url="https://large.example.org/v1", small_url=bad_url)

    for bad_model in ("qwen\x00model", "qwen\rmodel"):
        with pytest.raises(ValueError, match="large_model must be a non-empty string"):
            load_gwaya_code_expert(large_url="https://large.example.org/v1", large_model=bad_model)
        with pytest.raises(ValueError, match="small_model must be a non-empty string"):
            load_gwaya_code_expert(large_url="https://large.example.org/v1", small_model=bad_model)


def test_dataset_mining_campaign_null_byte_and_blank_split_text_hardening() -> None:
    ent = _compute_shannon_entropy_bits(["alpha beta gamma delta"])
    assert ent > 0.0
    with pytest.raises(ValueError, match="texts must be a sequence of strings"):
        _compute_shannon_entropy_bits(["valid text", "bad\x00text"])

    for bad_train in ([""], ["   "], ["ok one two three", " \t "]):
        with pytest.raises(ValueError, match="train_texts must be a non-empty sequence of strings"):
            _compute_max_cross_split_4gram_jaccard(bad_train, ["alpha beta gamma delta"])

    for bad_hold in ([""], ["   "], ["alpha beta gamma delta", ""]):
        with pytest.raises(ValueError, match="holdout_texts must be a non-empty sequence of strings"):
            _compute_max_cross_split_4gram_jaccard(["alpha beta gamma delta"], bad_hold)


def test_dataset_registry_slice_validate_and_sample_rows_hardening() -> None:
    spec = dr.SourceSpec(
        dataset_id="org/ds",
        domain="math",
        config="default",
        split="train",
        mapping={"kind": "fields", "prompt_fields": ["q"], "completion_field": "a"},
        verified_signal=True,
        signal_basis="gold",
    )

    with pytest.raises(TypeError, match="spec must be a SourceSpec"):
        dr.slice_rows("not-spec", lambda u: b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fetch must be callable"):
        dr.slice_rows(spec, "not-callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="spec must be a SourceSpec"):
        dr.validate_slice("not-spec", [], "sys")  # type: ignore[arg-type]
    for bad_rows in ("not-seq", b"bytes", 123, ["not-mapping"]):
        with pytest.raises(TypeError, match="rows must be a Sequence of row Mappings|row must be a Mapping"):
            dr.validate_slice(spec, bad_rows, "sys")  # type: ignore[arg-type]
    for bad_sys in ("", "   ", 123, None):
        with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
            dr.validate_slice(spec, [], bad_sys)  # type: ignore[arg-type]

    for bad_sample_rows in ("not-seq", b"bytes", 123, ["not-mapping"], [{}], [{"key": "  "}]):
        with pytest.raises(ValueError, match="rows must be a sequence|must be a mapping with a non-empty string 'key'"):
            dr.sample_rows(bad_sample_rows, 1, 42, "src")  # type: ignore[arg-type]

    sampled = dr.sample_rows([{"key": "k1", "v": 1}, {"key": "k2", "v": 2}], 1, 42, "src")
    assert len(sampled) == 1
