# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input-validation hardening tests for abc_moe_agent_mesh, adaptive_multi_split_wan, amcp_kafka_campaign_bus, and amcp_kafka_transport."""

from __future__ import annotations

import pytest

from hivemind.swarm.abc_moe_agent_mesh import (
    AgentMeshOrchestrator,
    WaggleDanceScrutinyProtocol,
    parse_scientific_scout_record,
)
from hivemind.swarm.adaptive_multi_split_wan import (
    AdaptiveMultiSplitWANController,
    CitizenHardwareProfile,
    WANLinkTelemetry,
    run_e20_adaptive_multi_split_experiment,
)
from hivemind.swarm.amcp_kafka_campaign_bus import (
    InMemoryKafkaBridge,
    build_amcp_kafka_event,
    render_joint_campaign_ics,
)
from hivemind.swarm.amcp_kafka_transport import InMemoryTransport


def test_abc_moe_agent_mesh_validation() -> None:
    proto = WaggleDanceScrutinyProtocol()
    dance = proto.create_waggle_dance(
        task_id="task-1",
        scout_bee_id="scout-1",
        task_domain="arxiv",
        confidence_score=0.9,
        estimated_tokens=256,
        required_tools=["web_search"],
        affinity_vector={"web_search": 0.9},
        quorum_threshold=2,
    )
    updated_dance, verdict = proto.process_peer_recruitment(
        dance, [{"bee_id": "peer-2", "affinity": 0.8}]
    )
    assert verdict is not None and verdict.status == "LOCKED"
    assert "peer-2" in updated_dance.current_recruits

    with pytest.raises(TypeError, match="dance"):
        proto.process_peer_recruitment("not-a-dance", [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="available_peers"):
        proto.process_peer_recruitment(dance, ("not-a-list",))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="peer"):
        proto.process_peer_recruitment(dance, ["not-a-dict"])  # type: ignore[list-item]

    for bad_aff in (True, -0.1, 1.2, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="affinity"):
            proto.process_peer_recruitment(dance, [{"bee_id": "peer-x", "affinity": bad_aff}])

    parsed = parse_scientific_scout_record({"title": "arXiv", "source": "https://arxiv.org/abs/2604.21428"})
    assert parsed["domain_parser"] == "arxiv"
    for bad_rec in ("not-a-mapping", None, 123, ["list"]):
        with pytest.raises(TypeError, match="record"):
            parse_scientific_scout_record(bad_rec)  # type: ignore[arg-type]

    orch = AgentMeshOrchestrator()
    for bad_schema in ({}, None, "not-a-dict", ["a"]):
        with pytest.raises(ValueError, match="target_schema"):
            orch.execute_mesh_workflow("task-1", "query", bad_schema)  # type: ignore[arg-type]


def test_adaptive_multi_split_wan_validation() -> None:
    ctrl = AdaptiveMultiSplitWANController()
    link = WANLinkTelemetry(uplink_mbps=25.0, downlink_mbps=100.0, rtt_ms=40.0)
    hw = CitizenHardwareProfile(
        node_id="node-1",
        gpu_name="RTX 3060",
        compute_capability_sm=8.6,
    )
    dec = ctrl.select_optimal_cut_layer(link, hw)
    assert dec.cut_layer_index in (1, 2, 3, 4)

    with pytest.raises(TypeError, match="link"):
        ctrl.estimate_cut_layer_cost(2, "bad-link", hw)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hw"):
        ctrl.estimate_cut_layer_cost(2, link, "bad-hw")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="link"):
        ctrl.select_optimal_cut_layer("bad-link", hw)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hw"):
        ctrl.select_optimal_cut_layer(link, "bad-hw")  # type: ignore[arg-type]

    for bad_act in ("123", b"123", None, 42):
        with pytest.raises(TypeError, match="activations"):
            ctrl.quantize_cut_activations_int8(bad_act)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="link"):
        run_e20_adaptive_multi_split_experiment(link="bad")  # type: ignore[arg-type]

    for bad_hws in ("bad", b"bad", ["not-a-profile"], 123):
        with pytest.raises(TypeError, match="hardware_profiles"):
            run_e20_adaptive_multi_split_experiment(hardware_profiles=bad_hws)  # type: ignore[arg-type]


def test_amcp_kafka_campaign_bus_validation() -> None:
    ev = build_amcp_kafka_event(
        topic_key="campaigns",
        event_type="io.hivemind.amcp.test",
        sender_id="alice",
        sender_kind="human_researcher",
        payload={"ok": True},
    )
    assert ev.payload == {"ok": True}

    for bad_payload in ("not-a-dict", ["a"], 123):
        with pytest.raises(TypeError, match="payload"):
            build_amcp_kafka_event(
                topic_key="campaigns",
                event_type="io.hivemind.amcp.test",
                sender_id="alice",
                sender_kind="human_researcher",
                payload=bad_payload,  # type: ignore[arg-type]
            )

    ics = render_joint_campaign_ics(None)
    assert "BEGIN:VCALENDAR" in ics
    for bad_summary in ("not-a-mapping", ["list"], 123):
        with pytest.raises(TypeError, match="poll_summary"):
            render_joint_campaign_ics(bad_summary)  # type: ignore[arg-type]

    bridge = InMemoryKafkaBridge()
    bridge.publish(ev)
    assert len(bridge.recent("campaigns", limit=5)) == 1
    for bad_topic in ("", "   ", 123, True):
        with pytest.raises(ValueError, match="topic"):
            bridge.recent(bad_topic)  # type: ignore[arg-type]


def test_amcp_kafka_transport_validation() -> None:
    tr = InMemoryTransport(num_partitions=2)
    tr.subscribe(["hivemind.amcp.campaigns.v1"])
    ack = tr.publish("hivemind.amcp.campaigns.v1", b"k1", b"v1")
    assert tr.committed_offset("hivemind.amcp.campaigns.v1", ack["partition"]) is None
    assert len(tr.records("hivemind.amcp.campaigns.v1")) == 1

    for bad_topics in ("hivemind.amcp.campaigns.v1", b"bytes", 123, None):
        with pytest.raises(TypeError, match="topics"):
            tr.subscribe(bad_topics)  # type: ignore[arg-type]

    for bad_list in ([""], ["   "], [123], [None]):
        with pytest.raises(ValueError, match="topic"):
            tr.subscribe(bad_list)  # type: ignore[list-item]

    for bad_topic in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="topic"):
            tr.committed_offset(bad_topic, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="topic"):
            tr.records(bad_topic)  # type: ignore[arg-type]

    for bad_part in (True, -1, "0", None):
        with pytest.raises(ValueError, match="partition"):
            tr.committed_offset("hivemind.amcp.campaigns.v1", bad_part)  # type: ignore[arg-type]
