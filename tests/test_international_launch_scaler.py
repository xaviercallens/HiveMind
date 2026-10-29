# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for Season 29 International Launch Scaler, Human/Agent Calendar & Nocturnal Image-MoE Monitor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.swarm.international_launch_scaler import (
    CLIENT_PRELUDE_MAX_VRAM_MIB,
    GLOBAL_LAUNCH_REGIONS,
    OPENROUTER_MAX_CONCURRENCY,
    TESLA_T4_MIN_FREE_VRAM_MIB,
    build_international_launch_calendar,
    coalesce_ledger_settlement_batch,
    compress_top1pct_dct_ef21,
    compute_hierarchical_fanout_plan,
    decompress_top1pct_dct,
    enroll_launch_participant,
    monitor_nocturnal_image_moe_evolution,
    query_local_tesla_t4_vram_guard,
    route_follow_the_sun_workload,
    select_adaptive_multisplit_cut,
)


def test_hm_s29_01_adaptive_multisplit_cut_respects_1500mb_vram_ceiling() -> None:
    """HM-S29-01: Adaptive Multi-Split ML cut-layer selection stays strictly <= 1,536 MiB and cuts WAN latency."""
    slow_link = select_adaptive_multisplit_cut(
        uplink_mbps=12.0,
        rtt_ms=65.0,
        client_vram_limit_mib=CLIENT_PRELUDE_MAX_VRAM_MIB,
    )
    assert slow_link.within_1500mb_background_ceiling is True
    assert slow_link.client_prelude_vram_mib <= CLIENT_PRELUDE_MAX_VRAM_MIB
    assert slow_link.step_latency_reduction_pct >= 85.0

    fast_link = select_adaptive_multisplit_cut(
        uplink_mbps=500.0,
        rtt_ms=15.0,
        client_vram_limit_mib=CLIENT_PRELUDE_MAX_VRAM_MIB,
    )
    assert fast_link.within_1500mb_background_ceiling is True
    assert fast_link.client_prelude_vram_mib <= CLIENT_PRELUDE_MAX_VRAM_MIB
    # Higher bandwidth allows deeper local cut (cut_index >= slow_link.selected_cut_index)
    assert fast_link.selected_cut_index >= slow_link.selected_cut_index

    # Ultra-constrained <700 MiB ceiling selects cut_index 1
    ultra_low = select_adaptive_multisplit_cut(
        uplink_mbps=100.0,
        rtt_ms=20.0,
        client_vram_limit_mib=700.0,
    )
    assert ultra_low.selected_cut_index == 1
    assert ultra_low.client_prelude_vram_mib <= 700.0


def test_hm_s29_02_top1pct_dct_ef21_exceeds_100x_compression_and_conserves_residual() -> None:
    """HM-S29-02: Top-1% 2D-DCT + 2-bit quantization achieves >= 100x compression and exact EF21 conservation."""
    rng = np.random.default_rng(42)
    shape = (128, 128)
    ef21_res = np.zeros(shape, dtype=np.float32)

    for _ in range(3):
        delta = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
        packet, recon_direct, next_res = compress_top1pct_dct_ef21(
            delta,
            ef21_residual_buffer=ef21_res,
            top_k_fraction=0.01,
        )
        recon_wire = decompress_top1pct_dct(packet)

        assert packet.compression_ratio >= 100.0
        assert packet.residual_conservation_l2_error <= 1e-6
        assert np.max(np.abs(recon_wire - recon_direct)) <= 1e-5

        compensated = delta + ef21_res
        drift = float(np.max(np.abs((recon_wire + next_res) - compensated)))
        assert drift <= 1e-5
        ef21_res = next_res


def test_hm_s31_02_hierarchical_fanout_and_sqlite_coalescing() -> None:
    """HM-S31-02: 6-Region Super-Peer fanout bounds coordinator ingress and SQLite coalescing conserves HMT."""
    plan = compute_hierarchical_fanout_plan(
        total_leaf_nodes=10_000,
        branching_factor=32,
        subswarm_size=18,
        coordinator_wave_stagger_batches=5,
    )
    assert plan.regions_active == len(GLOBAL_LAUNCH_REGIONS) == 6
    assert 15 <= plan.target_subswarm_size <= 20
    assert plan.ingress_reduction_factor >= 30.0
    assert plan.within_openrouter_concurrency_cap is True
    assert sum(r["leaf_nodes"] for r in plan.regional_breakdown) == 10_000

    # Coalesce 200 settlement events across 20 unique contributors
    events = [
        {
            "contributor_handle": f"@bee-{i % 20:02d}",
            "region_id": "eu-west" if i % 2 == 0 else "us-east",
            "campaign_id": "NIGHT-REDDIT-VISION-MOE-V1",
            "hmt_points": (i % 5) + 1,
            "workunits": 1,
        }
        for i in range(200)
    ]
    batch = coalesce_ledger_settlement_batch(events, window_ms=250)
    assert batch.raw_event_count == 200
    assert batch.coalesced_row_count == 20
    assert batch.write_iops_reduction_pct == 90.0
    assert batch.hmt_conservation_exact is True
    assert batch.total_hmt_before == batch.total_hmt_after > 0

    empty_batch = coalesce_ledger_settlement_batch([])
    assert empty_batch.raw_event_count == 0
    assert empty_batch.hmt_conservation_exact is True


def test_follow_the_sun_and_international_launch_calendar_human_and_agent() -> None:
    """Verify Follow-the-Sun routing, multilingual Human announcements, and Agent A2A/MCP enrollment."""
    sun = route_follow_the_sun_workload(utc_hour=23)
    assert len(sun["regional_assignments"]) == 6
    assert len(sun["nocturnal_training_regions"]) + len(sun["diurnal_inference_regions"]) == 6

    cal = build_international_launch_calendar()
    assert cal["event_id"] == "HIVEMIND-INTL-LAUNCH-WEEKEND-2026-10-30"
    assert cal["non_monetary_charter"]["financial_value_usd_eur"] == 0.0
    for lang in ("en", "fr", "es", "zh", "pt", "ja"):
        assert lang in cal["multilingual_human_announcements"]
    assert len(cal["schedule_phases"]) == 4
    assert len(cal["grand_challenges"]) == 4
    assert "/.well-known/hivemind-launch-calendar.json" in (
        cal["agent_a2a_mcp_protocol"]["discovery_endpoints"]["well_known_manifest"]
    )

    # Enroll a Turing (sm_75 -> pre-Ampere FP32) human gamer and an Ada (sm_89 -> BF16) autonomous agent
    human_rec = enroll_launch_participant(
        handle="@CitizenParis",
        participant_type="human",
        region_id="eu-west",
        compute_capability=(7, 5),
        vram_mib=8192.0,
        preferred_lang="fr",
    )
    assert human_rec["participant_type"] == "human"
    assert human_rec["precision_cohort"]["is_pre_ampere_fp32"] is True
    assert human_rec["precision_cohort"]["enforced_dtype"] == "fp32"
    assert human_rec["adaptive_multisplit_plan"]["within_1500mb_background_ceiling"] is True

    agent_rec = enroll_launch_participant(
        handle="agent://ruche/scout-bee-01",
        participant_type="agent",
        region_id="apac-east",
        compute_capability=(8, 9),
        vram_mib=1536.0,
        preferred_lang="ja",
    )
    assert agent_rec["participant_type"] == "agent"
    assert agent_rec["precision_cohort"]["is_pre_ampere_fp32"] is False
    assert agent_rec["precision_cohort"]["enforced_dtype"] == "bf16"


def test_nocturnal_image_moe_evolution_monitor_and_dashboard_routes(tmp_path: Path) -> None:
    """Verify Nocturnal Image-MoE Pareto evolution monitor, Tesla T4 VRAM guard, and FastAPI endpoints."""
    vram = query_local_tesla_t4_vram_guard()
    assert vram["min_required_free_vram_mib"] == TESLA_T4_MIN_FREE_VRAM_MIB
    assert vram["vram_headroom_guard_passed"] is True

    mon = monitor_nocturnal_image_moe_evolution(
        generations=2,
        base_inner_steps=15,
        repo_root=tmp_path,
        save_artifacts=True,
    )
    assert mon["status"] == "ACTIVE_EVOLVING_GREEN"
    assert mon["generations_executed"] == 2
    assert mon["pareto_ratchet_monotone_verified"] is True
    assert (tmp_path / "artifacts" / "nocturnal_image_moe_evolution_monitor.json").exists()

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    r_cal = client.get("/api/v1/launch/calendar")
    assert r_cal.status_code == 200
    assert r_cal.json()["event_id"] == "HIVEMIND-INTL-LAUNCH-WEEKEND-2026-10-30"

    r_wk = client.get("/.well-known/hivemind-launch-calendar.json")
    assert r_wk.status_code == 200
    assert len(r_wk.json()["grand_challenges"]) == 4

    r_enr = client.post(
        "/api/v1/launch/enroll",
        json={
            "handle": "@AutonomousBeeAgent",
            "participant_type": "agent",
            "region_id": "us-east",
            "compute_capability": [8, 6],
            "vram_mib": 1536,
            "uplink_mbps": 100.0,
            "rtt_ms": 20.0,
        },
    )
    assert r_enr.status_code == 200
    assert r_enr.json()["status"] == "ENROLLED_INTERNATIONAL_LAUNCH"

    r_plan = client.post(
        "/api/v1/launch/scalability-plan",
        json={"uplink_mbps": 30.0, "rtt_ms": 40.0, "total_leaf_nodes": 10000},
    )
    assert r_plan.status_code == 200
    assert r_plan.json()["adaptive_multisplit_cut"]["within_1500mb_background_ceiling"] is True
    assert r_plan.json()["hierarchical_fanout_plan"]["ingress_reduction_factor"] >= 30.0

    r_evo = client.get("/api/v1/night-moe/reddit-vision/evolution")
    assert r_evo.status_code == 200
    assert r_evo.json()["pareto_ratchet_monotone_verified"] is True

    # Also verify pre-registered E20 report exists and passed all predictions
    repo_root = Path(__file__).resolve().parents[1]
    e20_report_path = repo_root / "artifacts" / "experiments" / "E20_international_launch_split_dct_report.json"
    assert e20_report_path.exists()
    e20_data = json.loads(e20_report_path.read_text(encoding="utf-8"))
    assert e20_data["status"] == "VERIFIED_PASS"
    assert all(e20_data["predictions_check"].values())
