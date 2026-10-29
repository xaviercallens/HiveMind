"""Unit tests for the AMCP v1.5 + CloudEvents 1.0 EDA Traffic Sustainer & Sunday Nov 1 Producer-First Launch Plan."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.swarm.amcp_eda_traffic_sustainer import (
    LEVERAGED_AMCP_AND_HIVEMIND_PRS,
    AMCPEventDrivenTrafficSustainer,
    build_sunday_producer_first_international_plan,
    plan_100_producer_mesh_scaling,
    select_100_node_optimal_producer,
)


def test_cloudevent_v1_compliance_and_namespace_enforcement() -> None:
    sustainer = AMCPEventDrivenTrafficSustainer()
    ev = sustainer.publish_cloudevent(
        event_type="io.hivemind.amcp.inference.requested",
        source="urn:hivemind:router:eu-west1",
        subject="hivemind/coder",
        data={"prompt_tokens": 128},
    )
    assert ev.specversion == "1.0"
    assert ev.datacontenttype == "application/json"
    assert ev.type == "io.hivemind.amcp.inference.requested"
    assert len(sustainer.recent_events) == 1

    with pytest.raises(ValueError):
        sustainer.publish_cloudevent(
            event_type="invalid.namespace.event",
            source="urn:hivemind:router",
            subject="x",
            data={},
        )


def test_amcp_strong_mobility_sustains_viral_traffic_waves() -> None:
    sustainer = AMCPEventDrivenTrafficSustainer()
    # Without AMCP mobility: 38 producers serving 11.5 req/s -> static_rho ~ 0.925 (near saturation)
    # With AMCP replicate()/federateWith() onto 75 standby prosumers -> elastic_rho <= 0.65
    res = sustainer.plan_elastic_bee_mobility(
        wave_name="r/LocalLLaMA Mix Wave",
        base_resident_producers_c0=38,
        standby_prosumer_pool=75,
        peak_arrival_req_per_s=11.5,
    )
    assert res.static_utilization_rho > 0.85
    assert res.activated_amcp_replicas > 10
    assert res.elastic_eda_utilization_rho <= 0.65
    assert res.cold_start_504_prevented_by_pr249 > 0
    assert res.sustained_without_drops is True
    assert any(cmd["verb"] in {"replicate", "federateWith"} for cmd in res.mobility_commands_emitted)


def test_sunday_nov1_producer_first_plan_and_leveraged_prs() -> None:
    plan = build_sunday_producer_first_international_plan()
    assert plan["sunday_launch_date"] == "2026-11-01"
    assert len(plan["leveraged_prs"]) == len(LEVERAGED_AMCP_AND_HIVEMIND_PRS) >= 7
    assert len(plan["sunday_to_next_week_schedule"]) == 4
    assert all(w["sustained_without_drops"] is True for w in plan["elastic_eda_wave_simulations"])


def test_gateway_exposes_amcp_eda_sunday_launch_plan(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    resp = client.get("/v1/hivemind/international-launch-readiness").json()
    assert resp["sunday_producer_first_launch_date"] == "2026-11-01"
    amcp_plan = resp["amcp_eda_sunday_launch_plan"]
    assert amcp_plan["plan_id"] == "HM-S19-AMCP-EDA-SUNDAY-LAUNCH"
    assert len(amcp_plan["plan_sha256"]) == 64
    assert amcp_plan["hundred_producer_mesh_scaling"]["active_producers"] == 100


def test_select_100_node_optimal_producer_prefers_warm_residency_and_same_region() -> None:
    regions = ("eu-west", "us-east", "us-west", "ap-northeast")
    candidates = [
        {
            "account_id": f"prod-{i:03d}",
            "endpoint_id": "ep-0",
            "region": regions[i % 4],
            "loaded": (i % 5 == 0),
            "inflight": i % 7,
            "capacity_1k": 30,
            "models": ["hivemind/qwen3.5-coder-14b"],
            "paused": (i == 0),  # prod-000 is paused and must be skipped
        }
        for i in range(100)
    ]
    res = select_100_node_optimal_producer(
        candidates,
        model_id="hivemind/qwen3.5-coder-14b",
        client_region="eu-west",
    )
    assert res["total_eligible_candidates"] == 99
    assert res["best_residency_tier"] == 0
    selected = res["selected_producer"]
    assert selected["account_id"] != "prod-000"
    assert selected["loaded"] is True
    assert selected["region"] == "eu-west"
    assert res["warmup_recommended"] is False

    mesh = plan_100_producer_mesh_scaling(active_producers=100, endpoints_per_producer=2, heartbeat_interval_s=30.0)
    assert mesh["total_live_endpoints"] == 200
    assert mesh["sqlite_lock_reduction_factor"] == 50.0

