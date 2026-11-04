# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for HM-S19-CAMPAIGNS: Subject-by-Subject Campaigns, Poll, Joint Calendar & AMCP/Kafka Bus."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.swarm.amcp_kafka_campaign_bus import (
    InMemoryKafkaBridge,
    build_amcp_kafka_event,
    clear_bee_capacity_market,
)

SECRET = b"campaign-amcp-kafka-secret-32b!!"


def test_joint_campaigns_poll_and_calendar_ics() -> None:
    store = RouterStore(":memory:", SECRET)
    app = create_router_app(store, starter_grant_hmt=50.0)
    client = TestClient(app)

    # 1. Inspect public campaigns endpoint
    res = client.get("/v1/hivemind/campaigns")
    assert res.status_code == 200
    data = res.json()
    assert len(data["campaigns"]) == 6
    assert data["calendar_ics_url"] == "/v1/hivemind/campaigns/calendar.ics"
    assert "hivemind.amcp.bee_trading.v1" in data["amcp_kafka_topics"].values()

    # 2. Register human researcher + GPU producer + AI agent in the poll for Subject 2 (Rusty-SUNDIALS)
    u1 = client.post("/v1/hivemind/register", json={"handle": "dr-fluid", "role": "prosumer"}).json()
    u2 = client.post("/v1/hivemind/register", json={"handle": "rtx-miner-1", "role": "producer"}).json()
    u3 = client.post("/v1/hivemind/register", json={"handle": "claude-agent-s20", "role": "prosumer"}).json()

    v1 = client.post(
        "/v1/hivemind/campaigns/poll",
        headers={"Authorization": f"Bearer {u1['api_key']}"},
        json={
            "subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
            "participant_kind": "human_researcher",
            "pledged_vram_gb": 8.0,
            "availability_slot": "eu_evening_cet",
            "comment": "Contributing Navier-Stokes boundary layer benchmark",
        },
    )
    assert v1.status_code == 200

    v2 = client.post(
        "/v1/hivemind/campaigns/poll",
        headers={"Authorization": f"Bearer {u2['api_key']}"},
        json={
            "subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
            "participant_kind": "gpu_producer",
            "pledged_vram_gb": 16.0,
            "availability_slot": "24x7_autonomous",
        },
    )
    assert v2.status_code == 200

    v3 = client.post(
        "/v1/hivemind/campaigns/poll",
        headers={"Authorization": f"Bearer {u3['api_key']}"},
        json={
            "subject_id": "SUBJ-03-ASTRO-SYMPLECTIC-STARS",
            "participant_kind": "ai_agent",
            "pledged_vram_gb": 8.0,
            "availability_slot": "24x7_autonomous",
        },
    )
    assert v3.status_code == 200
    summary = v3.json()["poll_summary"]
    assert summary["total_registrations"] == 3
    assert summary["total_pledged_vram_gb"] == 32.0
    assert summary["leading_subject_id"] == "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE"
    weights = summary["nocturnal_workunit_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    assert weights["SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE"] > weights["SUBJ-06-LEAN4-FORMAL-PROOFS"]

    # 3. Download RFC 5545 .ics calendar feed
    ics_res = client.get("/v1/hivemind/campaigns/calendar.ics")
    assert ics_res.status_code == 200
    assert "BEGIN:VCALENDAR" in ics_res.text
    assert "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE" in ics_res.text
    assert "2 registered participants (24.0 GB VRAM pledged)" in ics_res.text


def test_cross_human_agent_amcp_kafka_and_inter_bee_coordination_and_trading() -> None:
    store = RouterStore(":memory:", SECRET)
    app = create_router_app(store, starter_grant_hmt=50.0)
    client = TestClient(app)

    alice = client.post("/v1/hivemind/register", json={"handle": "alice-human", "role": "prosumer"}).json()
    agent = client.post("/v1/hivemind/register", json={"handle": "agy-agent", "role": "prosumer"}).json()

    # 1. Cross-human -> agent message over AMCP/Kafka (hivemind.amcp.human_agent.v1)
    pub = client.post(
        "/v1/hivemind/amcp/publish",
        headers={"Authorization": f"Bearer {alice['api_key']}"},
        json={
            "topic_key": "human_agent",
            "event_type": "io.hivemind.amcp.problem_proposal",
            "sender_kind": "human_researcher",
            "recipient_id": "agy-agent",
            "subject_id": "SUBJ-04-QUANTUM-BEC-GROSS-PITAEVSKII",
            "payload": {"equation": "i dpsi/dt = (-0.5 lap + g|psi|^2) psi", "target_norm_drift": 1e-9},
        },
    )
    assert pub.status_code == 200
    assert pub.json()["topic"] == "hivemind.amcp.human_agent.v1"

    # 2. Inter-Bee inference coordination (hivemind/coder -> hivemind/bee-rusty-sundials)
    coord = client.post(
        "/v1/hivemind/bees/coordinate",
        headers={"Authorization": f"Bearer {agent['api_key']}"},
        json={
            "primary_bee": "hivemind/coder",
            "prompt": "Solve this stiff ODE Navier-Stokes vorticity equation using SUNDIALS CVODE BDF.",
            "steps": 16,
            "dt": 0.01,
        },
    )
    assert coord.status_code == 200
    c_body = coord.json()
    assert c_body["primary_bee"] == "hivemind/coder"
    assert c_body["specialist_bee"] == "hivemind/bee-rusty-sundials"
    assert c_body["numeric_certificate"]["verdict"] == "PASS"
    assert len(c_body["amcp_kafka_events"]) == 2

    # 3. Inter-Bee & Producer HMT capacity trading (OFFER vs BID double-auction clearing)
    offer_res = client.post(
        "/v1/hivemind/bees/trade",
        headers={"Authorization": f"Bearer {alice['api_key']}"},
        json={
            "bee_id": "hivemind/bee-stars",
            "side": "OFFER",
            "subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
            "capacity_1k": 120.0,
            "price_per_1k": 0.40,
        },
    )
    assert offer_res.status_code == 200

    bid_res = client.post(
        "/v1/hivemind/bees/trade",
        headers={"Authorization": f"Bearer {agent['api_key']}"},
        json={
            "bee_id": "hivemind/bee-rusty-sundials",
            "side": "BID",
            "subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
            "capacity_1k": 80.0,
            "price_per_1k": 0.60,
        },
    )
    assert bid_res.status_code == 200
    mkt = bid_res.json()["market_clearing"]
    assert mkt["matched_trades_count"] == 1
    assert mkt["total_traded_capacity_1k"] == 80.0
    assert mkt["volume_weighted_clearing_price_per_1k"] == 0.50

    # 4. Verify all events are queryable on GET /v1/hivemind/amcp/events
    events_res = client.get("/v1/hivemind/amcp/events")
    assert events_res.status_code == 200
    assert events_res.json()["count"] >= 5


def test_in_memory_kafka_bridge_and_double_auction_edge_cases() -> None:
    bridge = InMemoryKafkaBridge(maxlen_per_topic=16)
    ev = build_amcp_kafka_event(
        topic_key="bee_trading",
        event_type="io.hivemind.amcp.trade.offer",
        sender_id="hivemind/coder",
        sender_kind="bee_specialist",
    )
    ack = bridge.publish(ev)
    assert ack["offset"] == 0
    assert len(bridge.recent("bee_trading")) == 1

    # No match when bid < ask
    no_trade = clear_bee_capacity_market(
        [
            {"order_id": "o1", "bee_id": "b1", "side": "OFFER", "capacity_1k": 50.0, "price_per_1k": 0.8},
            {"order_id": "b1", "bee_id": "b2", "side": "BID", "capacity_1k": 50.0, "price_per_1k": 0.4},
        ]
    )
    assert no_trade["matched_trades_count"] == 0


def test_campaign_poll_votes_are_rate_limited_and_stop_growing_the_event_log() -> None:
    from hivemind.router.gateway import CAMPAIGN_POLL_LIMIT

    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=50.0))
    user = client.post("/v1/hivemind/register", json={"handle": "poll-spammer", "role": "prosumer"}).json()
    body = {"subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE", "participant_kind": "human_researcher", "pledged_vram_gb": 8.0, "availability_slot": "eu_evening_cet"}
    headers = {"Authorization": f"Bearer {user['api_key']}"}
    codes = [client.post("/v1/hivemind/campaigns/poll", headers=headers, json=body).status_code for _ in range(CAMPAIGN_POLL_LIMIT + 5)]
    assert codes[:CAMPAIGN_POLL_LIMIT] == [200] * CAMPAIGN_POLL_LIMIT
    assert codes[CAMPAIGN_POLL_LIMIT:] == [429] * 5
    events = len(store.list_amcp_kafka_events(limit=1000))
    assert client.post("/v1/hivemind/campaigns/poll", headers=headers, json=body).status_code == 429
    assert len(store.list_amcp_kafka_events(limit=1000)) == events


def test_identical_re_vote_writes_no_new_bus_event_but_a_changed_vote_does() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=50.0))
    user = client.post("/v1/hivemind/register", json={"handle": "poll-reposter", "role": "prosumer"}).json()
    headers = {"Authorization": f"Bearer {user['api_key']}"}
    body = {"subject_id": "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE", "participant_kind": "human_researcher", "pledged_vram_gb": 8.0, "availability_slot": "eu_evening_cet"}

    def events() -> int:
        return len(store.list_amcp_kafka_events(limit=1000))

    before = events()
    first = client.post("/v1/hivemind/campaigns/poll", headers=headers, json=body)
    assert first.status_code == 200 and first.json()["amcp_event"] is not None
    assert events() == before + 1
    again = client.post("/v1/hivemind/campaigns/poll", headers=headers, json=body)
    assert again.status_code == 200 and again.json()["amcp_event"] is None
    assert again.json()["registration"]["subject_id"] == body["subject_id"]
    assert events() == before + 1  # an identical vote did not grow the never-pruned log
    changed = client.post("/v1/hivemind/campaigns/poll", headers=headers, json={**body, "pledged_vram_gb": 16.0})
    assert changed.status_code == 200 and changed.json()["amcp_event"] is not None
    assert events() == before + 2
