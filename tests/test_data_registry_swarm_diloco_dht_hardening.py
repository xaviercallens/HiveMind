# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and negative-control tests for dataset_registry, decoupled_diloco, dht_storage, and diloco_nextgen."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.data.dataset_registry import (
    EvalIndex,
    MixtureConfig,
    ngram_hashes,
    plan_mixture,
    sample_rows,
)
from hivemind.swarm.decoupled_diloco import (
    ContinuousAsyncLearnerCoordinator,
    analyze_campaign_rda_and_delta_cosine,
    direct_weighted_average,
    radial_directional_average,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.diloco_nextgen import (
    CosineGatedAdamDecayOuterOptimizer,
    ReputationWeightedIsoCTSMerger,
)


def test_dataset_registry_input_validation() -> None:
    with pytest.raises(ValueError, match="tokens must be a sequence"):
        ngram_hashes("hello world", n=2)

    with pytest.raises(ValueError, match="n must be an integer >= 1"):
        ngram_hashes(["a", "b"], n=True)  # type: ignore[arg-type]

    assert len(ngram_hashes(["alpha", "beta", "gamma"], n=2)) == 2

    idx = EvalIndex(n=3)
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        idx.add("", ["a b c"])

    with pytest.raises(ValueError, match="texts must be an iterable of strings"):
        idx.add("pack", "a b c")

    with pytest.raises(ValueError, match="keys must be an iterable of strings"):
        idx.add("pack", ["a b c"], keys="K05218.01")

    cfg = MixtureConfig(total_rows=10, domain_weights={"code": 1.0})
    with pytest.raises(ValueError, match="cfg must be a MixtureConfig"):
        plan_mixture({"s1": ("code", True, 5)}, "not_cfg")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid available entry"):
        plan_mixture({"s1": ("code", 1, 5)}, cfg)  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="Invalid available entry"):
        plan_mixture({"s1": ("code", True, True)}, cfg)  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="k must be an integer"):
        sample_rows([{"key": "1"}], k=True, seed=0, source="src")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed must be an integer"):
        sample_rows([{"key": "1"}], k=1, seed=True, source="src")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source must be a non-empty string"):
        sample_rows([{"key": "1"}], k=1, seed=0, source=" ")


def test_decoupled_diloco_input_validation() -> None:
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="all vectors must be torch.Tensor"):
        direct_weighted_average(["not_a_tensor"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="weights must be finite numbers, not bool"):
        radial_directional_average([v1, v2], weights=[True, False])  # type: ignore[list-item]

    coord = ContinuousAsyncLearnerCoordinator(num_learners_m=3, min_quorum_k=2)
    with pytest.raises(ValueError, match="tokens_since_sync and steps_since_sync must be >= 0"):
        coord.submit_learner_delta("w1", 0, v1, tokens_since_sync=True, steps_since_sync=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="arrival_time_s must be finite"):
        coord.submit_learner_delta("w1", 0, v1, tokens_since_sync=100, steps_since_sync=2, arrival_time_s=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="global_param must be a non-empty torch.Tensor"):
        coord.close_quorum_round(torch.tensor([]))

    with pytest.raises(ValueError, match="outer_lr must be finite and > 0"):
        coord.close_quorum_round(v1, outer_lr=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="base_lora_state must be a non-empty dictionary"):
        analyze_campaign_rda_and_delta_cosine({}, {"a": {}, "b": {}})

    with pytest.raises(ValueError, match="low_cosine_threshold"):
        analyze_campaign_rda_and_delta_cosine(
            {"w": v1},
            {"a": {"w": v1}, "b": {"w": v2}},
            low_cosine_threshold=True,  # type: ignore[arg-type]
        )


def test_dht_storage_and_diloco_nextgen_validation(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(tmp_path / "cas")
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        store.store_hf_dataset_shard("", "code", 0, ["valid sample text with sufficient shannon entropy 123456"])

    with pytest.raises(ValueError, match="shard_id must be a non-negative integer"):
        store.store_hf_dataset_shard("node1", "code", True, ["valid sample text with sufficient shannon entropy 123456"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="leaf_hex_digests must be a sequence"):
        ContentAddressedSwarmStore.compute_merkle_root("a" * 64)

    with pytest.raises(ValueError, match="64-character hex string"):
        ContentAddressedSwarmStore.compute_merkle_root(["abcd"])

    assert len(ContentAddressedSwarmStore.compute_merkle_root(["a" * 64, "b" * 64])) == 64

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "adapter.bin").write_bytes(b"hello-hivemind")
    with pytest.raises(ValueError, match="source_node_id must be a non-empty string"):
        store.seed_bundle_chunks(bundle_dir, source_node_id=" ")

    with pytest.raises(ValueError, match="seeders must be a sequence"):
        store.seed_bundle_chunks(bundle_dir, source_node_id="node1", seeders="node2")  # type: ignore[arg-type]

    opt = CosineGatedAdamDecayOuterOptimizer()
    with pytest.raises(ValueError, match="params must be a non-empty dictionary"):
        opt.step_fragments({}, {})

    with pytest.raises(ValueError, match="worker_delays values must be finite numbers, not bool"):
        opt.step_fragments(
            {"w": torch.ones(4)},
            {"w": torch.ones(4) * 0.1},
            worker_delays={"w": True},  # type: ignore[dict-item]
        )

    merger = ReputationWeightedIsoCTSMerger()
    with pytest.raises(ValueError, match="base_lora_state must be a non-empty dictionary"):
        merger.merge_worker_deltas({}, [])

    with pytest.raises(ValueError, match="worker_reputation_weights must be finite numbers, not bool"):
        merger.merge_worker_deltas(
            {"w": torch.ones(2, 2)},
            [{"w": torch.ones(2, 2) * 0.5}],
            worker_reputation_weights=[True],  # type: ignore[list-item]
        )
