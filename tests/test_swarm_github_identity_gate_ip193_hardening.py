# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 193: Negative-control and invariant tests for amcp_github, identity_bridge, verified_update_gate, and ef_compression_benchmark."""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import torch

from hivemind.swarm.amcp_github import SeenEvents
from hivemind.swarm.ef_compression_benchmark import (
    SyntheticSwarmProblem,
    _local_steps,
    _mse,
    _topk_fp16,
)
from hivemind.swarm.identity_bridge import (
    SHARED_ED25519_VC_TEST_VECTORS,
    hash_hmt_key_commitment,
    issue_producer_verifiable_credential,
    verify_producer_verifiable_credential,
)
from hivemind.swarm.verified_update_gate import (
    ACCEPT,
    HeldOutTask,
    PASS,
    UpdateReceipt,
    decide,
)


def test_seen_events_rejects_invalid_paths_corrupt_ids_and_control_chars(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path"):
        SeenEvents("   ")
    with pytest.raises(ValueError, match="path"):
        SeenEvents("/tmp/bad\x00path.json")
    with pytest.raises(TypeError, match="path"):
        SeenEvents(123)  # type: ignore[arg-type]

    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted SeenEvents"):
        SeenEvents(corrupt_json)

    corrupt_bool = tmp_path / "bool_ids.json"
    corrupt_bool.write_text('{"ids": [true]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted SeenEvents"):
        SeenEvents(corrupt_bool)

    seen = SeenEvents(None)
    seen.add("1001")
    assert seen.ids == ["1001"]
    with pytest.raises(ValueError, match="event_id must be a non-empty string"):
        seen.add("1001\n")


def test_identity_bridge_rejects_control_chars_and_malformed_subject_fields() -> None:
    with pytest.raises(ValueError, match="whitespace or control"):
        hash_hmt_key_commitment("hmt_key with space")

    seed = b"\x11" * 32
    with pytest.raises(ValueError, match="node_id and kwaainet_peer_id"):
        issue_producer_verifiable_credential(
            private_key_seed=seed,
            node_id="node\n01",
            hmt_key="hmt_live_01",
            kwaainet_peer_id="12D3KooWParis01",
            tiers=["hivemind/coder"],
        )
    with pytest.raises(ValueError, match="tiers"):
        issue_producer_verifiable_credential(
            private_key_seed=seed,
            node_id="node-01",
            hmt_key="hmt_live_01",
            kwaainet_peer_id="12D3KooWParis01",
            tiers=["hivemind/coder bad"],
        )

    base_vc = SHARED_ED25519_VC_TEST_VECTORS[0]["credential"]

    bad_peer = copy.deepcopy(base_vc)
    bad_peer["credentialSubject"]["kwaainet_peer_id"] = ""
    ok_p, reason_p = verify_producer_verifiable_credential(bad_peer)
    assert ok_p is False and reason_p == "missing_kwaainet_peer_id"

    bad_tiers = copy.deepcopy(base_vc)
    bad_tiers["credentialSubject"]["tiers"] = []
    ok_t, reason_t = verify_producer_verifiable_credential(bad_tiers)
    assert ok_t is False and reason_t == "invalid_tiers"

    bad_sha = copy.deepcopy(base_vc)
    bad_sha["credentialSubject"]["hmt_key_sha256"] = "z" * 64
    ok_s, reason_s = verify_producer_verifiable_credential(bad_sha)
    assert ok_s is False and reason_s == "invalid_hmt_key_sha256"


def test_verified_update_gate_receipt_accounting_and_key_validation() -> None:
    with pytest.raises(ValueError, match="b \\+ c \\+ both_pass \\+ both_fail must equal n"):
        UpdateReceipt(
            decision=ACCEPT,
            reasons=["ok"],
            n=5,
            b=1,
            c=1,
            both_pass=5,
            both_fail=0,
            baseline_pass=6,
            candidate_pass=6,
            unverified_baseline=0,
            unverified_candidate=0,
            unverified_fraction=0.0,
            p_two_sided=1.0,
            p_regression=1.0,
            p_improvement=1.0,
            policy={},
            receipt_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="baseline_pass must equal b \\+ both_pass"):
        UpdateReceipt(
            decision=ACCEPT,
            reasons=["ok"],
            n=5,
            b=0,
            c=0,
            both_pass=5,
            both_fail=0,
            baseline_pass=4,
            candidate_pass=5,
            unverified_baseline=0,
            unverified_candidate=0,
            unverified_fraction=0.0,
            p_two_sided=1.0,
            p_regression=1.0,
            p_improvement=1.0,
            policy={},
            receipt_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="unverified counts cannot exceed n"):
        UpdateReceipt(
            decision=ACCEPT,
            reasons=["ok"],
            n=5,
            b=0,
            c=0,
            both_pass=5,
            both_fail=0,
            baseline_pass=5,
            candidate_pass=5,
            unverified_baseline=6,
            unverified_candidate=0,
            unverified_fraction=0.0,
            p_two_sided=1.0,
            p_regression=1.0,
            p_improvement=1.0,
            policy={},
            receipt_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="task keys"):
        decide({"   ": PASS}, {"   ": PASS})
    with pytest.raises(ValueError, match="key must be a non-empty string"):
        HeldOutTask(key="task\n1", prompt="def f():", tests=["assert True"])


def test_ef_compression_benchmark_rejects_bool_and_nonfinite_tensors() -> None:
    w_bool = torch.ones((2, 4), dtype=torch.bool)
    x_f = torch.ones((4, 4), dtype=torch.float32)
    y_f = torch.ones((4, 2), dtype=torch.float32)
    w_f = torch.ones((2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="w_true"):
        SyntheticSwarmProblem(w_bool, [x_f], [y_f], x_f, y_f)
    with pytest.raises(ValueError, match="numeric tensors"):
        _mse(w_bool, x_f, y_f)
    with pytest.raises(ValueError, match="numeric tensors"):
        _local_steps(w_bool, x_f, y_f, 2, 0.05)
    with pytest.raises(ValueError, match="finite"):
        _local_steps(torch.full((2, 4), float("nan")), x_f, y_f, 2, 0.05)
    with pytest.raises(ValueError, match="non-empty finite"):
        _topk_fp16(w_bool, 0.5)
