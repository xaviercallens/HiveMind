# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S34-04 Self-Evolving Antigravity + HiveMind 10-PR Autonomous Supervisor."""

from __future__ import annotations

from hivemind.swarm.s34_04_self_evolving_roadmap_supervisor import SelfEvolvingRoadmapSupervisor


def test_self_evolving_supervisor_verifies_all_10_roadmap_prs_and_hash_chain() -> None:
    supervisor = SelfEvolvingRoadmapSupervisor()

    # IP-151 negative control: calling with no request ids must NOT invent req-s32-s34-NN or report passed
    blocked_report = supervisor.verify_all_10_roadmap_modules()
    assert blocked_report["all_invariants_passed"] is False
    assert blocked_report["status"] == "BLOCKED_MISSING_REQUEST_RECEIPTS"
    assert blocked_report["total_prs_verified"] == 0
    assert all(e["openrouter_request_id"] == "" for e in blocked_report["entries"])

    # Positive control: supplying 10 verified request receipt IDs passes all 10 PR checks
    req_ids = [f"or-live-receipt-{i:02d}" for i in range(1, 11)]
    report = supervisor.verify_all_10_roadmap_modules(openrouter_request_ids=req_ids)

    assert report["status"] == "VERIFIED"
    assert report["total_prs_verified"] == 10
    assert report["all_invariants_passed"] is True
    assert len(report["merkle_head_sha256"]) == 64
    tickets = [e["ticket_id"] for e in report["entries"]]
    assert tickets[0] == "HM-S32-01"
    assert tickets[-1] == "HM-S34-04"
