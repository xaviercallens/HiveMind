# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-08-D1 (BOINC -> AMCP mesh events read-only on the ledger)
and pre-registration receipts PS-06-D1 (E26 residential compression) & PS-09-D1 (E27 1,000-node testbed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hivemind.boinc.mesh_events import (
    AMCPMeshEventBus,
    AMCPMeshHostView,
    BOINCChainLedger,
    BOINCMeshEvent,
    ReadOnlyLedgerViolationError,
    TOPIC_ASSIMILATED,
    TOPIC_VALIDATED,
    publish_validated_and_assimilated_round,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ps08_d1_paired_mesh_adapter_pointer_equals_ledger_and_preserves_credit() -> None:
    """PS-08-D1: In a paired multi-round test, every subscribed host sees the exact adapter pointer
    recorded in the BOINC chain ledger, and no mesh event changes volunteer credit."""
    initial_credits = {
        "host-paris-01": 1250.0,
        "host-lyon-02": 840.5,
        "host-brussels-03": 3100.25,
    }
    ledger = BOINCChainLedger(initial_credits=initial_credits)
    credit_digest_initial = ledger.credit_digest()

    bus = AMCPMeshEventBus(ledger=ledger)
    host_a = AMCPMeshHostView("host-paris-01")
    host_b = AMCPMeshHostView("host-lyon-02")
    host_c = AMCPMeshHostView("host-brussels-03")
    for view in (host_a, host_b, host_c):
        view.attach(bus)

    domains = ["general", "french", "edu_web", "news", "biomed", "math"]
    for round_idx, domain in enumerate(domains, start=1):
        sha = hashlib.sha256(f"adapter-round-{round_idx}-{domain}".encode("utf-8")).hexdigest()
        rel_path = f"boinc_chain/latest_adapter_r{round_idx:04d}.safetensors"
        wu_name = f"hmq_{domain}_r{round_idx:04d}"

        ev_val, ev_assim = publish_validated_and_assimilated_round(
            ledger=ledger,
            bus=bus,
            wu_name=wu_name,
            host_id="host-paris-01",
            chain_stem="latest_adapter",
            round_index=round_idx,
            adapter_rel_path=rel_path,
            adapter_sha256=sha,
            domain=domain,
            timestamp_utc=f"2026-11-07T0{round_idx}:00:00Z",
        )
        assert ev_val.topic == TOPIC_VALIDATED
        assert ev_assim.topic == TOPIC_ASSIMILATED

        ledger_ptr = ledger.latest_adapter_pointer("latest_adapter")
        assert ledger_ptr is not None
        assert ledger_ptr.round_index == round_idx
        assert ledger_ptr.adapter_sha256 == sha

        # Paired equality across all subscribed AMCP hosts and the BOINC ledger
        for view in (host_a, host_b, host_c):
            mesh_ptr = view.latest_adapter_pointer("latest_adapter")
            assert mesh_ptr == ledger_ptr

        # Credit ledger must remain byte-for-byte identical after every event
        assert ledger.credit_snapshot() == initial_credits
        assert ledger.credit_digest() == credit_digest_initial

    assert len(bus.published_events) == 12
    assert host_a.events_received == 12


def test_ps08_d1_negative_controls_credit_mutation_and_unvalidated_assimilation_rejected() -> None:
    """PS-08-D1 negative controls:
    1) Any mesh event carrying credit/HMT fields raises ReadOnlyLedgerViolationError.
    2) Any subscriber callback that attempts to mutate ledger credit raises ReadOnlyLedgerViolationError.
    3) An assimilated event without a prior validated event is rejected by both ledger and host view.
    """
    ledger = BOINCChainLedger(initial_credits={"host-01": 500.0})
    bus = AMCPMeshEventBus(ledger=ledger)
    host_view = AMCPMeshHostView("host-01")
    host_view.attach(bus)

    sha = hashlib.sha256(b"neg-control-adapter").hexdigest()
    base_payload = {
        "event_type": "validated",
        "wu_name": "hmq_general_r0001",
        "host_id": "host-01",
        "chain_stem": "latest_adapter",
        "round_index": 1,
        "adapter_rel_path": "boinc_chain/r0001.safetensors",
        "adapter_sha256": sha,
        "domain": "general",
        "timestamp_utc": "2026-11-07T01:00:00Z",
    }

    # 1. Payload with forbidden credit fields is rejected immediately
    for forbidden_key in ("credit", "credit_delta", "hmt_delta", "grant_credit", "mint_hmt"):
        bad = dict(base_payload)
        bad[forbidden_key] = 100.0
        with pytest.raises(ReadOnlyLedgerViolationError, match="forbidden credit fields"):
            bus.publish_raw_dict(bad)
    assert ledger.credit_snapshot() == {"host-01": 500.0}

    # 2. Assimilated event without prior validated event is rejected by AMCPMeshHostView
    unvalidated_assim = dict(base_payload)
    unvalidated_assim["event_type"] = "assimilated"
    unvalidated_assim["wu_name"] = "hmq_unvalidated_r0099"
    with pytest.raises(ValueError, match="no matching validated event"):
        bus.publish_raw_dict(unvalidated_assim)

    # 3. Malicious subscriber attempting to mutate ledger._credits during dispatch is caught
    def _rogue_subscriber(_event: BOINCMeshEvent) -> None:
        ledger._credits["host-01"] += 9999.0

    bus.subscribe(TOPIC_VALIDATED, _rogue_subscriber)
    with pytest.raises(ReadOnlyLedgerViolationError, match="mutated the credit ledger"):
        bus.publish_raw_dict(base_payload)


def test_ps06_d1_and_ps09_d1_preregistration_receipts_present_and_complete() -> None:
    """PS-06-D1 & PS-09-D1: E26 and E27 pre-registration JSON files exist and declare all required gates."""
    e26_path = REPO_ROOT / "artifacts" / "experiments" / "E26_residential_compression_prereg.json"
    e27_path = REPO_ROOT / "artifacts" / "experiments" / "E27_testbed_1000_prereg.json"
    assert e26_path.is_file()
    assert e27_path.is_file()

    e26 = json.loads(e26_path.read_text(encoding="utf-8"))
    assert e26["deliverable_id"] == "PS-06-D1"
    assert e26["registered_before_run"] is True
    assert e26["design"]["minimum_residential_links"] >= 5
    assert "C0_uncompressed_control" in e26["design"]["arms"]
    assert "C1_top1pct_dct_ef21" in e26["design"]["arms"]

    e27 = json.loads(e27_path.read_text(encoding="utf-8"))
    assert e27["deliverable_id"] == "PS-09-D1"
    assert e27["registered_before_run"] is True
    assert e27["design"]["node_counts"] == [100, 300, 1000]
    assert e27["design"]["runs_per_node_count"] >= 3
    thresholds = e27["registered_thresholds"]
    for required_key in (
        "time_to_convergence_ceiling_s",
        "bandwidth_per_epoch_ceiling_mb_per_node",
        "prover_or_replay_overhead_ceiling_pct",
        "routing_latency_p95_ceiling_ms",
    ):
        assert required_key in thresholds
    assert e27["constraints"]["monthly_cloud_budget_ceiling_usd"] <= 150.0
