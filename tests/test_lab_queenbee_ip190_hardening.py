# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 190 negative-control and positive-control hardening tests for SwarmLab and QueenBee."""

from __future__ import annotations

import hashlib

import pytest
import torch

from hivemind.lab.swarm_lab import (
    _digest,
    carried_weights,
    domain_loss,
    merge_deltas,
)
from hivemind.queenbee.dance import (
    DanceRecord,
    derive_signer_secret,
    recruitment_weights,
)
from hivemind.queenbee.knowledge_market import (
    Admission,
    KnowledgeShare,
    admit_share,
)
from hivemind.queenbee.strategies import (
    Outcome,
    PreferencePair,
    VerifiedSubtask,
    aggregate_lora_product_exact,
    clip_and_noise,
)


def test_swarm_lab_domain_loss_merge_deltas_carried_weights_and_digest_hardening() -> None:
    assert domain_loss([1.0, 2.0], [1.0, 1.0], [1.0, 0.0]) == pytest.approx(2.0)

    for bad_theta, bad_a, bad_c in (
        ([True, 2.0], [1.0, 1.0], [1.0, 0.0]),
        ([float("nan"), 2.0], [1.0, 1.0], [1.0, 0.0]),
        ([1.0, 2.0], [1.0, float("inf")], [1.0, 0.0]),
        ([1.0, 2.0], [-0.1, 1.0], [1.0, 0.0]),
    ):
        with pytest.raises(ValueError, match="finite numbers|non-negative"):
            domain_loss(bad_theta, bad_a, bad_c)  # type: ignore[arg-type]

    assert merge_deltas([([1.0, 3.0], 1.0), ([3.0, 1.0], 1.0)], "direct") == pytest.approx([2.0, 2.0])
    for bad_items in (
        "not-a-seq",
        [([1.0],)],
        [([], 1.0)],
        [([1.0, 2.0], 1.0), ([1.0], 1.0)],
        [([1.0, 2.0], 0.0)],
        [([1.0, 2.0], -1.0)],
        [([1.0, 2.0], True)],
        [([1.0, 2.0], float("nan"))],
    ):
        with pytest.raises(
            ValueError,
            match="items must be non-empty|each item must be|equal-length|item weight w must be",
        ):
            merge_deltas(bad_items, "direct")  # type: ignore[arg-type]

    keep, dropped = carried_weights([{"delta": [1.0], "base": 2, "worker": 0}], 3)
    assert len(keep) == 1 and dropped == 0 and keep[0]["w"] == pytest.approx(0.5)
    for bad_queue in (
        ["not-a-dict"],
        [{"base": 1}],
        [{"base": True, "worker": 0}],
        [{"base": 0, "worker": 0}],
        [{"base": 4, "worker": 0}],
        [{"base": 1, "worker": -1}],
        [{"base": 1, "worker": True}],
    ):
        with pytest.raises(ValueError, match="each queue entry must be a dict|queue entry 'base' must be"):
            carried_weights(bad_queue, 3)  # type: ignore[arg-type]

    assert len(_digest([1.0, 0.5])) == 16
    for bad_curve in ([], "1.0", [True], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="curve must be a non-empty sequence of finite numbers"):
            _digest(bad_curve)  # type: ignore[arg-type]


def test_queenbee_dance_control_chars_from_dict_and_scout_bounds_hardening() -> None:
    secret = b"test-secret-bytes"
    valid = DanceRecord("exp-1", "exp-1", "cluster-a", 10, 8, 0.25, 3, "unit_tests", "signer-1").signed(secret)
    round_trip = DanceRecord.from_dict(valid.to_dict())
    assert round_trip == valid

    for kwargs in (
        {"expert_id": "exp-1\ninjected"},
        {"served_by": "exp-1\x00"},
        {"cluster_id": "cluster\r"},
        {"signer_id": "signer\n2"},
        {"signature_hex": "abcd\n"},
    ):
        base = {
            "expert_id": "exp-1",
            "served_by": "exp-1",
            "cluster_id": "cluster-a",
            "n": 10,
            "successes": 8,
            "cost_per_1k": 0.25,
            "epoch": 3,
            "verifier_kind": "unit_tests",
            "signer_id": "signer-1",
            **kwargs,
        }
        with pytest.raises(ValueError, match="without control characters"):
            DanceRecord(**base)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid DanceRecord schema"):
        DanceRecord.from_dict({**valid.to_dict(), "schema": "wrong-schema"})
    with pytest.raises(ValueError, match="missing required DanceRecord fields"):
        bad_dict = dict(valid.to_dict())
        bad_dict.pop("cluster_id")
        DanceRecord.from_dict(bad_dict)

    with pytest.raises(ValueError, match="without control characters"):
        derive_signer_secret(secret, "signer-1\ninjected")

    for bad_bounds in (
        {("", "exp-1"): 0.8},
        {("cluster-a", "   "): 0.8},
        {(123, "exp-1"): 0.8},
        {("cluster-a", "exp-1"): 1.05},
    ):
        with pytest.raises(
            ValueError,
            match=r"scout_bounds must map \(cluster_id, expert_id\) to finite non-negative numbers in \[0, 1\]",
        ):
            recruitment_weights(
                [valid],
                now_epoch=3,
                horizon_epochs=5,
                cap_per_record=0.5,
                secret=secret,
                scout_bounds=bad_bounds,  # type: ignore[arg-type]
            )


def test_queenbee_knowledge_market_control_chars_and_state_containers_hardening() -> None:
    text = "def add(a, b):\n    return a + b\n"
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    receipt = "a" * 64
    valid_share = KnowledgeShare("acc-1", "verified_code", "MIT", True, 2, h, receipt)

    for bad_kwargs in (
        {"account_id": "acc-1\ninjected"},
        {"licence": "MIT\x00"},
        {"verifier_receipt_sha256": receipt + "\n"},
    ):
        base = {
            "account_id": "acc-1",
            "kind": "verified_code",
            "licence": "MIT",
            "verified": True,
            "epoch": 2,
            "text_sha256": h,
            "verifier_receipt_sha256": receipt,
            **bad_kwargs,
        }
        with pytest.raises(ValueError, match="without control characters"):
            KnowledgeShare(**base)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="without control characters"):
        Admission(True, "admitted\ninjected", valid_share)

    for bad_seen in ({"not-a-tuple"}, {("", h)}, {("acc-1", "   ")}):
        with pytest.raises(ValueError, match="seen_shares must be a set"):
            admit_share(valid_share, text, scanner=lambda _: True, seen_shares=bad_seen)  # type: ignore[arg-type]

    for bad_counts in (
        {"not-a-tuple": 1},
        {(-1, "acc-1"): 1},
        {(True, "acc-1"): 1},
        {(2, ""): 1},
        {(2, "acc-1"): -1},
        {(2, "acc-1"): True},
    ):
        with pytest.raises(ValueError, match="epoch_counts must be a dict"):
            admit_share(valid_share, text, scanner=lambda _: True, epoch_counts=bad_counts)  # type: ignore[arg-type]


def test_queenbee_strategies_types_clip_and_lora_aggregation_hardening() -> None:
    for bad_cluster, bad_expert in (
        (None, "e1"),
        (123, "e1"),
        ("c1", None),
        ("c1\ninjected", "e1"),
        ("c1", "e1\x00"),
    ):
        with pytest.raises(ValueError, match="non-empty strings without control characters"):
            Outcome(bad_cluster, bad_expert, True, True)  # type: ignore[arg-type]

    for bad_pair_kwargs in (
        {"prompt_id": None},
        {"cluster_id": 123},
        {"prompt_id": "p1\n"},
        {"rejected": 123},
        {"chosen": None},
        {"verified": "yes"},
        {"verifier_receipt_sha256": "abc\n"},
    ):
        base = {
            "prompt_id": "p1",
            "cluster_id": "c1",
            "rejected": "bad",
            "chosen": "good",
            "verifier_kind": "unit_tests",
            **bad_pair_kwargs,
        }
        with pytest.raises(ValueError, match="without control characters|must be strings|must be a boolean"):
            PreferencePair(**base)  # type: ignore[arg-type]

    for bad_subtask_id, bad_cluster_id in (
        ("   ", "c1"),
        ("s1", "   "),
        (123, "c1"),
        ("s1\n", "c1"),
    ):
        with pytest.raises(ValueError, match="non-empty strings without control characters"):
            VerifiedSubtask(bad_subtask_id, bad_cluster_id, "unit_tests")  # type: ignore[arg-type]

    gen = torch.Generator().manual_seed(42)
    for bad_delta in (
        {"": torch.ones((2, 2))},
        {"w": torch.empty((0, 2))},
        {"w": torch.ones((2, 2), dtype=torch.bool)},
        {"w": "not-a-tensor"},
    ):
        with pytest.raises(ValueError, match="delta must map non-empty string keys to non-empty numeric torch.Tensor"):
            clip_and_noise(bad_delta, 1.0, 0.1, gen)  # type: ignore[arg-type]

    valid_deltas = {
        "w1": {
            "layer.lora_A": torch.ones((2, 4)),
            "layer.lora_B": torch.ones((4, 2)),
        },
        "w2": {
            "layer.lora_A": torch.full((2, 4), 2.0),
            "layer.lora_B": torch.full((4, 2), 0.5),
        },
    }
    agg = aggregate_lora_product_exact(valid_deltas, ["w1", "w2"])
    assert "layer.lora_product_exact" in agg

    with pytest.raises(ValueError, match="every accepted worker must be a non-empty string key"):
        aggregate_lora_product_exact(valid_deltas, ["w1", "missing"])
    with pytest.raises(ValueError, match="all accepted workers in admitted_deltas must share the same parameter keys"):
        aggregate_lora_product_exact(
            {"w1": {"a": torch.ones((2, 2))}, "w2": {"b": torch.ones((2, 2))}},
            ["w1", "w2"],
        )
    with pytest.raises(ValueError, match="LoRA factor pair .* must be finite 2D numeric tensors"):
        aggregate_lora_product_exact(
            {
                "w1": {
                    "layer.lora_A": torch.ones((3, 4)),
                    "layer.lora_B": torch.ones((4, 2)),
                }
            },
            ["w1"],
        )
