"""Synchronized Campaign Calendar, Quorum Coordinator & Non-Technical Onboarding Wizard.

Organizes human end-users (gamers, students, citizen scientists without coding skills),
technical developers, and autonomous AI agents into synchronized time windows so the
decentralized HiveMind Swarm reaches mathematical Byzantine Quorum (`n >= 2f + 3`) and
full 8-Domain / 4-Super-Cluster MoE coverage during Operation Polyglot-Singularity-128.

IMPORTANT NON-MONETARY CHARTER:
No financial payment, fiat currency, cryptocurrency, or wage is ever promised or paid.
Symbolic 'points' (`PTS` / `$HIVE`) confer exclusively reciprocal computing processing
benefits (shared Sparse MoE inference tokens and fine-tuning priority) when the live
production HiveMind Swarm is deployed, available, and used by a large community.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ICS_PATH = (
    REPO_ROOT / "docs" / "calendar" / "hivemind_polyglot_singularity_128_campaign.ics"
)
DEFAULT_QUORUM_REPORT_PATH = (
    REPO_ROOT / "artifacts" / "campaign_quorum_schedule_report.json"
)

NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE = (
    "NO FINANCIAL PAYMENT: Project HiveMind is a voluntary, "
    "non-commercial open-source research project. No money, fiat currency, cryptocurrency, "
    "or wage is ever promised or paid. Accumulated 'points' (PTS and $HIVE PoUW accounting "
    "units) confer exclusively reciprocal computing processing benefits (shared Sparse MoE "
    "inference and training quota) when the live production HiveMind Swarm is deployed, "
    "available, and used by a large community."
)

QUORUM_TIERS: dict[str, dict[str, Any]] = {
    "TIER_1_MIN_BYZANTINE_QUORUM": {
        "tier_id": "TIER_1_MIN_BYZANTINE_QUORUM",
        "title": "Tier 1 — Minimum Byzantine Consensus Quorum (f=1)",
        "min_total_nodes": 5,
        "min_human_gpu_nodes": 3,
        "min_ai_agent_nodes": 2,
        "max_byzantine_faults_tolerated": 1,
        "formula": "n >= 2f + 3 = 5",
        "super_clusters_required": 2,
        "domains_required": 4,
        "symbolic_bonus_pts": 500,
    },
    "TIER_2_FULL_8DOMAIN_MOE_QUORUM": {
        "tier_id": "TIER_2_FULL_8DOMAIN_MOE_QUORUM",
        "title": "Tier 2 — Full 8-Domain Polyglot MoE Quorum (f=2)",
        "min_total_nodes": 8,
        "min_human_gpu_nodes": 5,
        "min_ai_agent_nodes": 3,
        "max_byzantine_faults_tolerated": 2,
        "formula": "n >= 2f + 3 = 7 (8 nodes = 1 per Hugging Face domain)",
        "super_clusters_required": 4,
        "domains_required": 8,
        "symbolic_bonus_pts": 1500,
    },
    "TIER_3_GLOBAL_SUPER_RING_RAID": {
        "tier_id": "TIER_3_GLOBAL_SUPER_RING_RAID",
        "title": "Tier 3 — Continental Super-Ring 128-Expert WAN GameDay (f=5)",
        "min_total_nodes": 16,
        "min_human_gpu_nodes": 10,
        "min_ai_agent_nodes": 6,
        "max_byzantine_faults_tolerated": 5,
        "formula": "n >= 2f + 3 = 13 (16 nodes = 2 replicas per HF domain across 3 continents)",
        "super_clusters_required": 4,
        "domains_required": 8,
        "symbolic_bonus_pts": 3000,
    },
}

CAMPAIGN_CALENDAR_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "event_id": "EVT-S7-01-ONBOARDING-KICKOFF",
        "title": "Week 1 Kickoff: 1-Click Non-Technical GPU Onboarding & Baseline Receipt Sprint",
        "dtstart_utc": "20261024T170000Z",
        "dtend_utc": "20261024T183000Z",
        "human_readable_utc": "Saturday, Oct 24, 2026 · 17:00–18:30 UTC",
        "timezones": "19:00 CEST (Paris) | 13:00 EDT (New York) | 10:00 PDT (SF) | 02:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_1_MIN_BYZANTINE_QUORUM",
        "target_audience": "Non-technical end-users (1-click launcher), GPU testers & AI onboarding agents",
        "focus_domains": (
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ),
        "discussion_url": "https://github.com/xaviercallens/HiveMind/discussions/86",
        "symbolic_points_pts": "750 - 1,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S7-02-APAC-EU-QUORUM-SYNC",
        "title": "Week 1 APAC + Europe Sync: 8-Domain Hugging Face LoRA & CAS Zero-Step Transfer",
        "dtstart_utc": "20261028T120000Z",
        "dtend_utc": "20261028T133000Z",
        "human_readable_utc": "Wednesday, Oct 28, 2026 · 12:00–13:30 UTC",
        "timezones": "13:00 CET (Paris) | 08:00 EDT (New York) | 21:00 JST (Tokyo)",
        "target_quorum_tier": "TIER_2_FULL_8DOMAIN_MOE_QUORUM",
        "target_audience": "Asia-Pacific & European GPU contributors + Autonomous AI Prover Agents",
        "focus_domains": (
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "ZK_Cryptography_And_Audits",
            "Cyber_Vulnerability_Patching",
            "Triton_CUDA_Kernel_Synth",
            "Distributed_DiLoCo_Protocols",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ),
        "discussion_url": "https://github.com/xaviercallens/HiveMind/discussions/86",
        "symbolic_points_pts": "1,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S7-03-BYZANTINE-TIES-SHIELD",
        "title": "Week 2 Red-Team vs. Blue-Team: Byzantine Poisoner & Reputation-Weighted TIES Sprint",
        "dtstart_utc": "20261031T180000Z",
        "dtend_utc": "20261031T193000Z",
        "human_readable_utc": "Saturday, Oct 31, 2026 · 18:00–19:30 UTC",
        "timezones": "19:00 CET (Paris) | 14:00 EDT (New York) | 11:00 PDT (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_2_FULL_8DOMAIN_MOE_QUORUM",
        "target_audience": "All Human GPU Nodes + AI Security/Validator Agents (`ZK-OPoI v4` + `SI-04`)",
        "focus_domains": (
            "ZK_Cryptography_And_Audits",
            "Cyber_Vulnerability_Patching",
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
        ),
        "discussion_url": "https://github.com/xaviercallens/HiveMind/discussions/86",
        "symbolic_points_pts": "1,500 - 2,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S7-04-128MOE-UPCYCLE-RAID",
        "title": "Week 3 Boss Raid: 128-Expert 3-Tier Hierarchical MoE Upcycling & Speculative Prefetch",
        "dtstart_utc": "20261107T180000Z",
        "dtend_utc": "20261107T200000Z",
        "human_readable_utc": "Saturday, Nov 7, 2026 · 18:00–20:00 UTC",
        "timezones": "19:00 CET (Paris) | 13:00 EST (New York) | 10:00 PST (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_3_GLOBAL_SUPER_RING_RAID",
        "target_audience": "All 3 Guilds (`Euro-Singularity`, `Global-ZK-Guardians`, `Biomed-Open-Science`) + AI Agents",
        "focus_domains": (
            "Triton_CUDA_Kernel_Synth",
            "Distributed_DiLoCo_Protocols",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ),
        "discussion_url": "https://github.com/xaviercallens/HiveMind/discussions/86",
        "symbolic_points_pts": "3,000 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S7-05-GLOBAL-WAN-GAMEDAY",
        "title": "Week 4 Finale: Global Multi-Host WAN GameDay (aioquic 0-RTT + TURN ChannelBind + Live HF Stream)",
        "dtstart_utc": "20261114T180000Z",
        "dtend_utc": "20261114T200000Z",
        "human_readable_utc": "Saturday, Nov 14, 2026 · 18:00–20:00 UTC",
        "timezones": "19:00 CET (Paris) | 13:00 EST (New York) | 10:00 PST (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_3_GLOBAL_SUPER_RING_RAID",
        "target_audience": "Global Human Community (Gamers, Devs) + Autonomous AI Swarm Agents",
        "focus_domains": (
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "ZK_Cryptography_And_Audits",
            "Cyber_Vulnerability_Patching",
            "Triton_CUDA_Kernel_Synth",
            "Distributed_DiLoCo_Protocols",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ),
        "discussion_url": "https://github.com/xaviercallens/HiveMind/discussions/86",
        "symbolic_points_pts": "3,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
)

SEASON9_10_GPU_QUORUM_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "event_id": "EVT-S9-01-DECOUPLED-QUORUM-RDA",
        "title": "Season 9 GPU Sprint 1: Decoupled Quorum (K < M), Async Stragglers (HM-S9-13) & RDA Cosine (HM-S9-14)",
        "dtstart_utc": "20261025T170000Z",
        "dtend_utc": "20261025T183000Z",
        "human_readable_utc": "Sunday, Oct 25, 2026 · 17:00–18:30 UTC",
        "timezones": "18:00 CET (Paris) | 13:00 EDT (New York) | 10:00 PDT (SF) | 02:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_1_MIN_BYZANTINE_QUORUM",
        "target_audience": "Community GPU Contributors (@clark719664, @pedromineo, @radialmonster) + AI Agents",
        "command": "uv run python scripts/run_season9_10_gpu_swarm_campaigns.py --contributor-handle <handle>",
        "symbolic_points_pts": "3,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S9-02-SIGMA-MOE-ORTHOINJ",
        "title": "Season 9 GPU Sprint 2: FoMoE Sigma-MoE + Peri-LN (HM-S9-02) & Looped OrthoInj Terminal-KV (HM-S9-03)",
        "dtstart_utc": "20261101T180000Z",
        "dtend_utc": "20261101T193000Z",
        "human_readable_utc": "Sunday, Nov 1, 2026 · 18:00–19:30 UTC",
        "timezones": "19:00 CET (Paris) | 13:00 EST (New York) | 10:00 PST (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_2_FULL_8DOMAIN_MOE_QUORUM",
        "target_audience": "8 GB Consumer GPU Nodes (RTX 20/30/40/50, RX 6000/7000, Apple M1-M4)",
        "command": "uv run python scripts/run_season9_10_gpu_swarm_campaigns.py --contributor-handle <handle>",
        "symbolic_points_pts": "3,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S10-01-E8-A2-LATTICE-KV-SPRINT",
        "title": "Season 10 GPU Sprint 3: Conway-Sloane E8/A2 Hexagonal Lattice Wire (HM-S10-01) & 5-Codec KV-Cache (HM-S10-05)",
        "dtstart_utc": "20261108T180000Z",
        "dtend_utc": "20261108T193000Z",
        "human_readable_utc": "Sunday, Nov 8, 2026 · 18:00–19:30 UTC",
        "timezones": "19:00 CET (Paris) | 13:00 EST (New York) | 10:00 PST (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_2_FULL_8DOMAIN_MOE_QUORUM",
        "target_audience": "GPU Compression & Inference Ring Contributors + AI Prover Agents",
        "command": "uv run python scripts/run_season9_10_gpu_swarm_campaigns.py --contributor-handle <handle>",
        "symbolic_points_pts": "3,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
    {
        "event_id": "EVT-S10-02-CROSS-GPU-SPOTCHECK-GAMEDAY",
        "title": "Season 10 GPU Sprint 4: Cross-Architecture Two-Statistic Spot-Check & Q16 Digest GameDay (HM-S10-08)",
        "dtstart_utc": "20261115T180000Z",
        "dtend_utc": "20261115T200000Z",
        "human_readable_utc": "Sunday, Nov 15, 2026 · 18:00–20:00 UTC",
        "timezones": "19:00 CET (Paris) | 13:00 EST (New York) | 10:00 PST (SF) | 03:00+1 JST (Tokyo)",
        "target_quorum_tier": "TIER_3_GLOBAL_SUPER_RING_RAID",
        "target_audience": "Multi-Host GPU & TPU Validators (NVIDIA T4/RTX, AMD ROCm, Apple MPS, GCP TPU v5e)",
        "command": "uv run python scripts/run_season9_10_gpu_swarm_campaigns.py --contributor-handle <handle>",
        "symbolic_points_pts": "3,500 PTS (Compute Reciprocity Only; HiveMind Reflexion Points (HMT))",
    },
)


def evaluate_quorum_readiness(
    human_participants: Sequence[str],
    ai_agent_participants: Sequence[str],
    covered_domains: Sequence[str],
) -> dict[str, Any]:
    """Compute Byzantine quorum (`n >= 2f + 3`) and 8-domain MoE coverage status."""
    humans = sorted(set(h.strip() for h in human_participants if h and h.strip()))
    agents = sorted(set(a.strip() for a in ai_agent_participants if a and a.strip()))
    domains = sorted(set(d.strip() for d in covered_domains if d and d.strip()))

    total_nodes = len(humans) + len(agents)
    max_byzantine_faults_tolerated = max(0, (total_nodes - 3) // 2) if total_nodes >= 3 else 0

    tier1_reached = (
        total_nodes >= 5 and len(humans) >= 3 and len(agents) >= 2 and len(domains) >= 4
    )
    tier2_reached = (
        total_nodes >= 8 and len(humans) >= 5 and len(agents) >= 3 and len(domains) >= 8
    )
    tier3_reached = (
        total_nodes >= 16 and len(humans) >= 10 and len(agents) >= 6 and len(domains) >= 8
    )

    if tier3_reached:
        active_tier = "TIER_3_GLOBAL_SUPER_RING_RAID"
    elif tier2_reached:
        active_tier = "TIER_2_FULL_8DOMAIN_MOE_QUORUM"
    elif tier1_reached:
        active_tier = "TIER_1_MIN_BYZANTINE_QUORUM"
    else:
        active_tier = "PRE_QUORUM_GATHERING"

    return {
        "human_count": len(humans),
        "ai_agent_count": len(agents),
        "total_nodes": total_nodes,
        "covered_domains_count": len(domains),
        "max_byzantine_faults_tolerated_f": max_byzantine_faults_tolerated,
        "byzantine_quorum_satisfied_f1": total_nodes >= 5,
        "tier1_reached": tier1_reached,
        "tier2_reached": tier2_reached,
        "tier3_reached": tier3_reached,
        "active_quorum_tier": active_tier,
        "nodes_needed_for_tier1": max(0, 5 - total_nodes),
        "nodes_needed_for_tier2": max(0, 8 - total_nodes),
        "nodes_needed_for_tier3": max(0, 16 - total_nodes),
        "humans": humans,
        "ai_agents": agents,
        "covered_domains": domains,
    }


def generate_campaign_ics_calendar(output_path: Path | None = DEFAULT_ICS_PATH) -> str:
    """Generate a standard RFC 5545 `.ics` calendar file for all Campaign Quorum Events."""
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Project HiveMind//Operation Polyglot-Singularity-128 Quorum Calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:HiveMind Polyglot-Singularity-128 GPU & AI Quorum Sprints",
        "X-WR-TIMEZONE:UTC",
    ]
    for evt in CAMPAIGN_CALENDAR_EVENTS:
        desc = (
            f"{evt['title']}\\n"
            f"Target Quorum: {evt['target_quorum_tier']}\\n"
            f"Audience: {evt['target_audience']}\\n"
            f"1-Click Command: uv run python scripts/run_multi_moe_hf_campaign.py --contributor-handle <handle> --tier standard\\n"
            f"Discussion Hub: {evt['discussion_url']}\\n"
            f"NON-MONETARY CHARTER: Points ({evt['symbolic_points_pts']}) confer only reciprocal computing processing benefits when the live production HiveMind Swarm is deployed."
        )
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{evt['event_id']}@hivemind.open-science.org",
                f"DTSTAMP:20261022T110000Z",
                f"DTSTART:{evt['dtstart_utc']}",
                f"DTEND:{evt['dtend_utc']}",
                f"SUMMARY:[HiveMind Quorum] {evt['title']}",
                f"DESCRIPTION:{desc}",
                f"URL:{evt['discussion_url']}",
                "STATUS:CONFIRMED",
                "END:VEVENT",
            ]
        )

    # Also add the standing 3-hour Autonomous Orchestrator & Mini-Quorum Sync rule
    lines.extend(
        [
            "BEGIN:VEVENT",
            "UID:EVT-S7-STANDING-3H-ORCHESTRATOR@hivemind.open-science.org",
            "DTSTAMP:20261022T110000Z",
            "DTSTART:20261022T120000Z",
            "DTEND:20261022T122000Z",
            "RRULE:FREQ=HOURLY;INTERVAL=3;COUNT=240",
            "SUMMARY:[HiveMind 3h Sync] Autonomous PR Review, Merge & Discussion Quorum Pulse",
            "DESCRIPTION:Automated 3-hour PR review, merge, and GitHub Discussion receipt verification window (0 */3 * * * UTC). No financial payment.",
            "URL:https://github.com/xaviercallens/HiveMind/discussions/86",
            "STATUS:CONFIRMED",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )
    ics_text = "\r\n".join(lines) + "\r\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ics_text, encoding="utf-8")
    return ics_text


def build_campaign_quorum_schedule_report(
    output_path: Path | None = DEFAULT_QUORUM_REPORT_PATH,
    ics_output_path: Path | None = DEFAULT_ICS_PATH,
) -> dict[str, Any]:
    """Build the complete Campaign Calendar & Quorum Readiness JSON report and export `.ics`."""
    ics_content = generate_campaign_ics_calendar(output_path=ics_output_path)

    # Current baseline quorum roster (core contributors + active AI agents)
    current_quorum = evaluate_quorum_readiness(
        human_participants=["xaviercallens", "clark719664", "pedromineo", "radialmonster"],
        ai_agent_participants=[
            "antigravity-orchestrator-01",
            "claude-code-prover-02",
            "hivemind-zk-auditor-03",
        ],
        covered_domains=[
            "Rust_Systems_Kernel",
            "Lean4_Formal_Mathematics",
            "ZK_Cryptography_And_Audits",
            "Cyber_Vulnerability_Patching",
            "Triton_CUDA_Kernel_Synth",
            "Distributed_DiLoCo_Protocols",
            "Biomedical_Clinical_Genomics",
            "Astrophysics_JWST_Spectroscopy",
        ],
    )

    report: dict[str, Any] = {
        "campaign_id": "OPERATION-POLYGLOT-SINGULARITY-128-CALENDAR-V1",
        "non_monetary_charter": NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE,
        "ics_calendar_path": str(ics_output_path) if ics_output_path else "",
        "ics_sha256": hashlib.sha256(ics_content.encode("utf-8")).hexdigest(),
        "standing_orchestrator_cron": "0 */3 * * * (Every 3 Hours UTC: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC)",
        "quorum_tiers": QUORUM_TIERS,
        "scheduled_events": list(CAMPAIGN_CALENDAR_EVENTS),
        "season9_10_gpu_quorum_events": list(SEASON9_10_GPU_QUORUM_EVENTS),
        "current_baseline_quorum_status": current_quorum,
    }
    report["schedule_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Campaign Calendar & Quorum Coordinator."""
    parser = argparse.ArgumentParser(
        description="Manage the Project HiveMind Campaign Calendar, .ics export, and Quorum Readiness."
    )
    parser.add_argument(
        "--ics-output",
        type=Path,
        default=DEFAULT_ICS_PATH,
        help="Path to write the RFC 5545 .ics calendar file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QUORUM_REPORT_PATH,
        help="Path to write the JSON quorum schedule report",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_campaign_quorum_schedule_report(
        output_path=args.output,
        ics_output_path=args.ics_output,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
