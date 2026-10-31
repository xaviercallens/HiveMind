# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S34-04 Self-Evolving Antigravity + HiveMind 10-PR Autonomous Supervisor."""

from __future__ import annotations

from hivemind.swarm.s34_04_self_evolving_roadmap_supervisor import SelfEvolvingRoadmapSupervisor


def test_self_evolving_supervisor_verifies_all_10_roadmap_prs_and_hash_chain() -> None:
    supervisor = SelfEvolvingRoadmapSupervisor()
    report = supervisor.verify_all_10_roadmap_modules()

    assert report["total_prs_verified"] == 10
    assert report["all_invariants_passed"] is True
    assert len(report["merkle_head_sha256"]) == 64
    tickets = [e["ticket_id"] for e in report["entries"]]
    assert tickets[0] == "HM-S32-01"
    assert tickets[-1] == "HM-S34-04"
