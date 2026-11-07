# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for PR-IP-43: Campaign quorum & Season 9/10 receipt hardening."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hivemind.cli.campaign_calendar_quorum import (
    build_campaign_quorum_schedule_report,
    evaluate_quorum_readiness,
    verify_campaign_quorum_schedule_report,
)
from hivemind.cli.multi_moe_hf_campaign import (
    run_multi_moe_hf_campaign,
    verify_multi_moe_hf_campaign_report,
)
from hivemind.cli.season9_10_gpu_swarm_campaigns import (
    format_season9_10_gpu_campaigns_markdown,
    run_season9_10_gpu_swarm_campaigns,
    verify_season9_10_campaign_report,
)


def _rehash(report: dict[str, Any], digest_key: str) -> dict[str, Any]:
    body = {k: v for k, v in report.items() if k != digest_key}
    canonical = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        **body,
        digest_key: hashlib.sha256(canonical).hexdigest(),
    }


def test_campaign_calendar_quorum_sybil_overlap_and_schedule_verification(tmp_path: Path) -> None:
    # 1. Sybil role overlap (same handle in both human and AI agent lists) is rejected
    with pytest.raises(ValueError, match="double-counted"):
        evaluate_quorum_readiness(
            human_participants=["alice", "bob", "sybil-node"],
            ai_agent_participants=["agent-1", "sybil-node"],
            covered_domains=[
                "Rust_Systems_Kernel",
                "Lean4_Formal_Mathematics",
                "Biomedical_Clinical_Genomics",
                "Astrophysics_JWST_Spectroscopy",
            ],
        )

    # 2. Role-aware nodes_needed_for_tier1 when 5 humans and 0 AI agents are present
    q_imbalanced = evaluate_quorum_readiness(
        human_participants=["h1", "h2", "h3", "h4", "h5"],
        ai_agent_participants=[],
        covered_domains=[
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ],
    )
    assert q_imbalanced["tier1_reached"] is False
    assert q_imbalanced["nodes_needed_for_tier1"] == 2

    # 3. Genuine schedule report passes verification; re-hashed tampered report fails closed
    rep = build_campaign_quorum_schedule_report(
        output_path=tmp_path / "quorum_rep.json",
        ics_output_path=tmp_path / "quorum.ics",
    )
    assert verify_campaign_quorum_schedule_report(rep) is True

    forged_tier = copy.deepcopy(rep)
    forged_tier["current_baseline_quorum_status"]["tier3_reached"] = True
    assert verify_campaign_quorum_schedule_report(_rehash(forged_tier, "schedule_sha256")) is False

    forged_charter = copy.deepcopy(rep)
    forged_charter["non_monetary_charter"] = "PAID CASH BOUNTY"
    assert verify_campaign_quorum_schedule_report(_rehash(forged_charter, "schedule_sha256")) is False


def test_season9_10_gpu_swarm_campaigns_verification_and_markdown_honesty() -> None:
    for bad_kwargs in (
        {"contributor_handle": "   "},
        {"num_domains": True},
        {"num_domains": 1},
        {"num_domains": 9},
        {"inner_steps": True},
        {"inner_steps": 0},
        {"seed": True},
    ):
        with pytest.raises(ValueError):
            run_season9_10_gpu_swarm_campaigns(
                preferred_device="cpu",
                output_path=None,
                **bad_kwargs,
            )

    rep = run_season9_10_gpu_swarm_campaigns(
        contributor_handle="ip43-s9-10-cpu",
        preferred_device="cpu",
        num_domains=2,
        inner_steps=2,
        seed=21,
        output_path=None,
    )
    assert verify_season9_10_campaign_report(rep) is True
    md_ok = format_season9_10_gpu_campaigns_markdown(rep)
    assert "❌ FAIL" not in md_ok

    # 1. Forging GPU points (3500) on a CPU report and rehashing must fail closed
    forged_pts = copy.deepcopy(rep)
    forged_pts["non_monetary_charter"]["symbolic_points_preview_pts"] = 3500
    assert verify_season9_10_campaign_report(_rehash(forged_pts, "campaign_suite_sha256")) is False

    # 2. Corrupting Campaign 4 tampered_replay_rejected and rehashing must fail closed and render ❌ FAIL in Markdown
    forged_c4 = copy.deepcopy(rep)
    forged_c4["campaign_4_s10_spotcheck_leasing_and_gossip"]["tampered_replay_rejected"] = False
    rehashed_c4 = _rehash(forged_c4, "campaign_suite_sha256")
    assert verify_season9_10_campaign_report(rehashed_c4) is False
    md_fail = format_season9_10_gpu_campaigns_markdown(rehashed_c4)
    assert "❌ FAIL" in md_fail


def test_multi_moe_hf_campaign_parameter_and_receipt_verification() -> None:
    for bad_kwargs in (
        {"contributor_handle": "  "},
        {"steps_per_domain": True},
        {"steps_per_domain": 1},
        {"domain_limit": True},
        {"domain_limit": 1},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.0},
        {"sequence_length": True},
        {"sequence_length": 4},
        {"seed": True},
        {"min_quorum_k": True},
        {"min_quorum_k": 0},
        {"domain_limit": 2, "min_quorum_k": 5},
        {"grace_margin_gamma": float("nan")},
        {"grace_margin_gamma": 1.0},
    ):
        with pytest.raises(ValueError):
            run_multi_moe_hf_campaign(
                preferred_device="cpu",
                output_path=None,
                **bad_kwargs,
            )

    rep = run_multi_moe_hf_campaign(
        contributor_handle="ip43-polyglot-cpu",
        preferred_device="cpu",
        steps_per_domain=4,
        domain_limit=2,
        sequence_length=48,
        output_path=None,
    )
    assert verify_multi_moe_hf_campaign_report(rep) is True

    # 1. Forging GPU points (1500) on a CPU report and rehashing fails closed
    forged_pts = copy.deepcopy(rep)
    forged_pts["symbolic_compute_reciprocity_points"]["symbolic_points_preview_pts"] = 1500
    assert verify_multi_moe_hf_campaign_report(_rehash(forged_pts, "campaign_sha256")) is False

    # 2. Tampering with Stage 4 Byzantine TIES defense result and rehashing fails closed
    forged_s4 = copy.deepcopy(rep)
    forged_s4["stage4_byzantine_ties_merge_vs_naive"]["ties_beats_poisoned_naive"] = False
    assert verify_multi_moe_hf_campaign_report(_rehash(forged_s4, "campaign_sha256")) is False

    # 3. Tampering with Stage 5 insurance pool holdings and rehashing fails closed
    forged_s5 = copy.deepcopy(rep)
    forged_s5["stage5_128_expert_moe_and_zk_opoi_v4"]["guild_insurance_pool_holdings_hive"] = 100.0
    assert verify_multi_moe_hf_campaign_report(_rehash(forged_s5, "campaign_sha256")) is False
