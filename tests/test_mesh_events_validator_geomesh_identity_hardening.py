# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening tests for mesh_events, boinc.validator, geomesh_fuzzy_consensus, and identity_bridge."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.boinc.mesh_events import (
    AMCPMeshEventBus,
    BOINCChainLedger,
    publish_validated_and_assimilated_round,
)
from hivemind.boinc.validator import (
    ExecutionClassProfile,
    SpotCheckVerdict,
    TwoStatisticSpotCheckResult,
)
from hivemind.swarm.geomesh_fuzzy_consensus import (
    FuzzyConsensusResult,
    GeoMeshNode,
    SubSwarmAssignment,
    triangulate_rtt_consistency,
)
from hivemind.swarm.identity_bridge import (
    SHARED_ED25519_VC_TEST_VECTORS,
    _canonical_vc_bytes_without_proof,
    verify_producer_verifiable_credential,
)


def test_mesh_events_dispatch_and_round_publish_validation() -> None:
    ledger = BOINCChainLedger({"host-1": 100.0})
    bus = AMCPMeshEventBus(ledger)

    with pytest.raises(TypeError, match="event must be a BOINCMeshEvent"):
        bus._dispatch("not_an_event")  # type: ignore[arg-type]

    ev_val, ev_assim = publish_validated_and_assimilated_round(
        ledger=ledger,
        bus=bus,
        wu_name="wu-01",
        host_id="host-1",
        chain_stem="latest_adapter",
        round_index=1,
        adapter_rel_path="adapters/r1.safetensors",
        adapter_sha256="a" * 64,
        domain="code",
        timestamp_utc="2026-11-10T00:00:00Z",
    )
    assert ev_val.event_type == "validated" and ev_assim.event_type == "assimilated"

    with pytest.raises(TypeError, match="ledger must be a BOINCChainLedger"):
        publish_validated_and_assimilated_round(
            ledger="bad",  # type: ignore[arg-type]
            bus=bus,
            wu_name="wu-02",
            host_id="host-1",
            chain_stem="latest_adapter",
            round_index=2,
            adapter_rel_path="adapters/r2.safetensors",
            adapter_sha256="b" * 64,
            domain="code",
            timestamp_utc="2026-11-10T00:00:00Z",
        )
    with pytest.raises(TypeError, match="bus must be an AMCPMeshEventBus"):
        publish_validated_and_assimilated_round(
            ledger=ledger,
            bus="bad",  # type: ignore[arg-type]
            wu_name="wu-02",
            host_id="host-1",
            chain_stem="latest_adapter",
            round_index=2,
            adapter_rel_path="adapters/r2.safetensors",
            adapter_sha256="b" * 64,
            domain="code",
            timestamp_utc="2026-11-10T00:00:00Z",
        )
    with pytest.raises(ValueError, match="host_id must be a non-empty string"):
        publish_validated_and_assimilated_round(
            ledger=ledger,
            bus=bus,
            wu_name="wu-02",
            host_id="   ",
            chain_stem="latest_adapter",
            round_index=2,
            adapter_rel_path="adapters/r2.safetensors",
            adapter_sha256="b" * 64,
            domain="code",
            timestamp_utc="2026-11-10T00:00:00Z",
        )


def test_boinc_validator_execution_class_and_result_dataclasses() -> None:
    p1 = ExecutionClassProfile()
    p2 = ExecutionClassProfile()
    assert p1.exact_match(p2) and p1.same_dtype_class(p2)

    with pytest.raises(TypeError, match="other must be an ExecutionClassProfile"):
        p1.exact_match("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="other must be an ExecutionClassProfile"):
        p1.same_dtype_class("bad")  # type: ignore[arg-type]

    res = TwoStatisticSpotCheckResult(
        linf_divergence=1e-5,
        num_tensors_above_eps=0,
        total_tensors=10,
        tensor_fraction_above_eps=0.0,
        base_eps=1e-4,
        effective_eps=1e-4,
        max_tensor_fraction_above_eps=0.1,
        exact_class_matched=True,
        dtype_matched=True,
        passed=True,
        reason="PASSED",
    )
    assert res.passed

    with pytest.raises(ValueError, match="linf_divergence must be a non-negative"):
        TwoStatisticSpotCheckResult(-1.0, 0, 10, 0.0, 1e-4, 1e-4, 0.1, True, True, True, "ok")
    with pytest.raises(ValueError, match="total_tensors must be an integer >= 1"):
        TwoStatisticSpotCheckResult(0.0, 5, 4, 0.0, 1e-4, 1e-4, 0.1, True, True, True, "ok")
    with pytest.raises(ValueError, match="tensor_fraction_above_eps must be a finite"):
        TwoStatisticSpotCheckResult(0.0, 0, 4, 1.5, 1e-4, 1e-4, 0.1, True, True, True, "ok")
    with pytest.raises(ValueError, match="base_eps must be a positive finite"):
        TwoStatisticSpotCheckResult(0.0, 0, 4, 0.0, 0.0, 1e-4, 0.1, True, True, True, "ok")
    with pytest.raises(ValueError, match="passed must be a boolean"):
        TwoStatisticSpotCheckResult(0.0, 0, 4, 0.0, 1e-4, 1e-4, 0.1, True, True, 1, "ok")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        TwoStatisticSpotCheckResult(0.0, 0, 4, 0.0, 1e-4, 1e-4, 0.1, True, True, True, "   ")

    verdict = SpotCheckVerdict(
        wu_id="wu-1",
        worker_id="w-1",
        steam_id64="76561198000000001",
        segment_start_step=0,
        segment_end_step=50,
        verification_overhead_pct=10.0,
        max_abs_divergence=1e-6,
        tolerance_eps=1e-4,
        hash_matched=True,
        passed=True,
        steam_account_flagged=False,
        reason="Verified",
    )
    assert verdict.passed

    with pytest.raises(ValueError, match="wu_id must be a non-empty string"):
        SpotCheckVerdict("", "w-1", "s1", 0, 50, 10.0, 0.0, 1e-4, True, True, False, "ok")
    with pytest.raises(ValueError, match="segment_start_step and segment_end_step"):
        SpotCheckVerdict("wu-1", "w-1", "s1", 60, 50, 10.0, 0.0, 1e-4, True, True, False, "ok")
    with pytest.raises(ValueError, match="verification_overhead_pct"):
        SpotCheckVerdict("wu-1", "w-1", "s1", 0, 50, -1.0, 0.0, 1e-4, True, True, False, "ok")
    with pytest.raises(ValueError, match="tolerance_eps must be a positive finite"):
        SpotCheckVerdict("wu-1", "w-1", "s1", 0, 50, 10.0, 0.0, 0.0, True, True, False, "ok")
    with pytest.raises(ValueError, match="hash_matched must be a boolean"):
        SpotCheckVerdict("wu-1", "w-1", "s1", 0, 50, 10.0, 0.0, 1e-4, "yes", True, False, "ok")  # type: ignore[arg-type]


def test_geomesh_fuzzy_consensus_dataclasses_and_rtt_validation() -> None:
    node = GeoMeshNode("n1", 48.85, 2.35, 12322, {"a1": 12.0})
    sub = SubSwarmAssignment("sub-1", (node,), 0.0, 1)
    assert sub.unique_asn_count == 1

    with pytest.raises(ValueError, match="subswarm_id must be a non-empty"):
        SubSwarmAssignment("", (node,), 0.0, 1)
    with pytest.raises(ValueError, match="nodes must be a non-empty tuple"):
        SubSwarmAssignment("sub-1", (), 0.0, 1)
    with pytest.raises(ValueError, match="mean_pairwise_km must be a finite non-negative"):
        SubSwarmAssignment("sub-1", (node,), -1.0, 1)
    with pytest.raises(ValueError, match="unique_asn_count must be an integer >= 1"):
        SubSwarmAssignment("sub-1", (node,), 0.0, 0)

    fc = FuzzyConsensusResult(
        subswarm_id="sub-1",
        aggregated_gradient=np.array([1.0, 2.0], dtype=np.float32),
        node_weights={"n1": 1.0},
        rejected_node_ids=(),
        honest_cosine_fidelity=1.0,
    )
    assert fc.honest_cosine_fidelity == 1.0

    with pytest.raises(ValueError, match="aggregated_gradient must be a non-empty finite"):
        FuzzyConsensusResult("sub-1", np.array([]), {"n1": 1.0}, (), 1.0)
    with pytest.raises(ValueError, match="node_weights must be a non-empty"):
        FuzzyConsensusResult("sub-1", np.array([1.0]), {}, (), 1.0)
    with pytest.raises(ValueError, match="node_weights must map non-empty"):
        FuzzyConsensusResult("sub-1", np.array([1.0]), {"n1": -0.5}, (), 1.0)
    with pytest.raises(ValueError, match="rejected_node_ids must be a tuple"):
        FuzzyConsensusResult("sub-1", np.array([1.0]), {"n1": 1.0}, ("",), 1.0)
    with pytest.raises(ValueError, match="honest_cosine_fidelity must be a finite"):
        FuzzyConsensusResult("sub-1", np.array([1.0]), {"n1": 1.0}, (), 1.5)

    with pytest.raises(TypeError, match="node must be a GeoMeshNode"):
        triangulate_rtt_consistency("bad", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="anchors must be a dictionary"):
        triangulate_rtt_consistency(node, [])  # type: ignore[arg-type]


def test_identity_bridge_canonical_bytes_and_verifier_validation() -> None:
    vec = SHARED_ED25519_VC_TEST_VECTORS[0]
    vc = vec["credential"]
    ok, reason = verify_producer_verifiable_credential(
        vc,
        expected_node_id=vec["node_id"],
        expected_hmt_key=vec["hmt_key"],
    )
    assert ok and reason == "ok"

    with pytest.raises(ValueError, match="vc must be a non-empty dictionary"):
        _canonical_vc_bytes_without_proof({})
    with pytest.raises(ValueError, match="expected_node_id must be None or a non-empty string"):
        verify_producer_verifiable_credential(vc, expected_node_id="   ")
    with pytest.raises(ValueError, match="expected_hmt_key must be None or a non-empty string"):
        verify_producer_verifiable_credential(vc, expected_hmt_key="")
