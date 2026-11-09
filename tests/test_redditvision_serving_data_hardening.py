# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for Reddit-Vision VLM MoE, Serving Experts, Dataset Mining Campaign, and Dataset Registry (PR-IP106)."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from hivemind.core.reddit_vision_vlm_moe import (
    DHTDeltaRegistry,
    FrozenVisionLanguageBackbone,
    QLoRAVisionAdapter,
    dequantize_nf4_blockwise,
    quantize_nf4_blockwise,
    quantize_sparseloco_dct_2bit,
)
from hivemind.core.serving_experts import route_with_serving_experts
from hivemind.data.dataset_mining_campaign import (
    _compute_max_cross_split_4gram_jaccard,
)
from hivemind.data.dataset_registry import (
    ACCEPTED,
    LicenseDecision,
    Shard,
    retrying_fetch,
)


def test_reddit_vision_vlm_moe_rejects_malformed_inputs() -> None:
    mat = np.ones((8, 8), dtype=np.float32) * 0.25
    idx, absmax = quantize_nf4_blockwise(mat, block_size=16)
    recon = dequantize_nf4_blockwise(idx, absmax, (8, 8))
    assert recon.shape == (8, 8)

    with pytest.raises(ValueError, match="indices"):
        dequantize_nf4_blockwise([1, 2], absmax, (8, 8))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absmax"):
        dequantize_nf4_blockwise(idx, [1.0], (8, 8))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="original_shape"):
        dequantize_nf4_blockwise(idx, absmax, (0, 8))

    pkt = quantize_sparseloco_dct_2bit(
        mat,
        bee_id="bee-vlm-01",
        cluster_id="expert_1_aesthetics_art",
        backbone_id="qwen2.5-vl-3b",
    )
    with pytest.raises(ValueError, match="dht_key"):
        replace(pkt, dht_key="")
    with pytest.raises(ValueError, match="packed_2bit_codes"):
        replace(pkt, packed_2bit_codes=b"")
    with pytest.raises(ValueError, match="raw_fp32_bytes"):
        replace(pkt, raw_fp32_bytes=0)
    with pytest.raises(ValueError, match="compression_ratio"):
        replace(pkt, compression_ratio=0.0)
    with pytest.raises(ValueError, match="sqnr_db"):
        replace(pkt, sqnr_db=True)  # type: ignore[arg-type]

    reg = DHTDeltaRegistry()
    key = reg.publish(pkt)
    assert reg.fetch(key).bee_id == "bee-vlm-01"
    with pytest.raises(ValueError, match="packet"):
        reg.publish("not_a_packet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dht_key"):
        reg.fetch(" ")

    with pytest.raises(ValueError, match="vision_dim"):
        FrozenVisionLanguageBackbone(vision_dim=0)
    with pytest.raises(ValueError, match="seed"):
        FrozenVisionLanguageBackbone(seed=-1)

    with pytest.raises(ValueError, match="hidden_dim"):
        QLoRAVisionAdapter(hidden_dim=0)
    with pytest.raises(ValueError, match="alpha"):
        QLoRAVisionAdapter(alpha=0.0)


def test_serving_experts_route_rejects_malformed_inputs() -> None:
    doms, weights = route_with_serving_experts("def foo(): import pytest", top_k=2)
    assert len(doms) >= 1
    assert sum(weights.values()) > 0.0

    with pytest.raises(ValueError, match="prompt"):
        route_with_serving_experts("   ")
    with pytest.raises(ValueError, match="top_k"):
        route_with_serving_experts("def foo(): pass", top_k=0)
    with pytest.raises(ValueError, match="min_weight"):
        route_with_serving_experts("def foo(): pass", min_weight=1.5)


def test_dataset_mining_campaign_jaccard_rejects_empty_splits() -> None:
    j = _compute_max_cross_split_4gram_jaccard(
        ["alpha beta gamma delta epsilon"],
        ["zeta eta theta iota kappa"],
    )
    assert j == 0.0

    with pytest.raises(ValueError, match="train_texts"):
        _compute_max_cross_split_4gram_jaccard([], ["zeta eta theta iota"])
    with pytest.raises(ValueError, match="holdout_texts"):
        _compute_max_cross_split_4gram_jaccard(["alpha beta gamma delta"], [])


def test_dataset_registry_rejects_malformed_inputs() -> None:
    dec = LicenseDecision(status=ACCEPTED, license="mit", reason="allowlisted")
    assert dec.status == ACCEPTED

    with pytest.raises(ValueError, match="status"):
        LicenseDecision(status="UNKNOWN", license="mit", reason="allowlisted")
    with pytest.raises(ValueError, match="reason"):
        LicenseDecision(status=ACCEPTED, license="mit", reason=" ")

    shard = Shard(uri="file://test.jsonl", sha256="a" * 64, n_bytes=128, n_rows=4)
    assert shard.to_dict()["rows"] == 4
    with pytest.raises(ValueError, match="uri"):
        Shard(uri="", sha256="a" * 64, n_bytes=128, n_rows=4)
    with pytest.raises(ValueError, match="sha256"):
        Shard(uri="file://test.jsonl", sha256="short", n_bytes=128, n_rows=4)
    with pytest.raises(ValueError, match="n_bytes"):
        Shard(uri="file://test.jsonl", sha256="a" * 64, n_bytes=-1, n_rows=4)

    with pytest.raises(ValueError, match="fetch"):
        retrying_fetch("not_callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retries"):
        retrying_fetch(lambda u: b"", retries=-1)
    with pytest.raises(ValueError, match="base_delay_s"):
        retrying_fetch(lambda u: b"", base_delay_s=-0.5)
