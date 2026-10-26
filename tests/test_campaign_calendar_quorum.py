"""Unit tests for the Campaign Calendar, .ics Exporter & Quorum Coordinator."""

from __future__ import annotations

from pathlib import Path

from hivemind.cli.campaign_calendar_quorum import (
    CAMPAIGN_CALENDAR_EVENTS,
    build_campaign_quorum_schedule_report,
    evaluate_quorum_readiness,
    generate_campaign_ics_calendar,
)
from hivemind.router.points_wording import currency_leftovers


def test_evaluate_quorum_readiness_tiers() -> None:
    # Pre-quorum (only 2 humans, 1 agent)
    q_low = evaluate_quorum_readiness(
        human_participants=["alice", "bob"],
        ai_agent_participants=["agent-1"],
        covered_domains=["Rust_Systems_Kernel", "Lean4_Formal_Mathematics"],
    )
    assert q_low["tier1_reached"] is False
    assert q_low["active_quorum_tier"] == "PRE_QUORUM_GATHERING"
    assert q_low["nodes_needed_for_tier1"] == 2

    # Tier 1 reached (3 humans + 2 AI agents = 5 >= 2(1)+3, 4 domains)
    q_t1 = evaluate_quorum_readiness(
        human_participants=["alice", "bob", "charlie"],
        ai_agent_participants=["agent-1", "agent-2"],
        covered_domains=[
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ],
    )
    assert q_t1["tier1_reached"] is True
    assert q_t1["byzantine_quorum_satisfied_f1"] is True
    assert q_t1["max_byzantine_faults_tolerated_f"] == 1
    assert q_t1["active_quorum_tier"] == "TIER_1_MIN_BYZANTINE_QUORUM"

    # Tier 2 reached (5 humans + 3 AI agents = 8 nodes, all 8 domains)
    all_8_domains = [
        "Rust_Systems_Kernel",
        "Lean4_Formal_Mathematics",
        "ZK_Cryptography_And_Audits",
        "Cyber_Vulnerability_Patching",
        "Triton_CUDA_Kernel_Synth",
        "Distributed_DiLoCo_Protocols",
        "Biomedical_Clinical_Genomics",
        "Astrophysics_JWST_Spectroscopy",
    ]
    q_t2 = evaluate_quorum_readiness(
        human_participants=["h1", "h2", "h3", "h4", "h5"],
        ai_agent_participants=["a1", "a2", "a3"],
        covered_domains=all_8_domains,
    )
    assert q_t2["tier2_reached"] is True
    assert q_t2["max_byzantine_faults_tolerated_f"] == 2
    assert q_t2["active_quorum_tier"] == "TIER_2_FULL_8DOMAIN_MOE_QUORUM"


def test_generate_campaign_ics_and_report(tmp_path: Path) -> None:
    ics_file = tmp_path / "campaign.ics"
    report_file = tmp_path / "quorum_report.json"

    ics_text = generate_campaign_ics_calendar(output_path=ics_file)
    assert ics_file.exists()
    assert "BEGIN:VCALENDAR" in ics_text
    assert "END:VCALENDAR" in ics_text
    assert "RRULE:FREQ=HOURLY;INTERVAL=3;COUNT=240" in ics_text
    for evt in CAMPAIGN_CALENDAR_EVENTS:
        assert evt["event_id"] in ics_text

    report = build_campaign_quorum_schedule_report(
        output_path=report_file,
        ics_output_path=ics_file,
    )
    assert report_file.exists()
    assert len(report["schedule_sha256"]) == 64
    assert "NO FINANCIAL PAYMENT" in report["non_monetary_charter"]
    assert currency_leftovers(report["non_monetary_charter"]) == []
    assert len(report["scheduled_events"]) == 5
