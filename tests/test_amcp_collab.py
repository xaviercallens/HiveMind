# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Who may say what on the AMCP bus: role policy, collaboration payloads, the gateway's publish endpoint, and the development keyring."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.swarm.amcp_collab import (
    AGENT,
    ALL_KINDS,
    BEE,
    COLLAB_EVENTS,
    HUMAN,
    PRODUCER,
    ROLE_KINDS,
    TOPIC_WRITERS,
    CollabError,
    bus_from_env,
    check_publish,
    dev_keyring,
)
from hivemind.swarm.amcp_kafka_transport import KafkaConfigError, KafkaSettings

SECRET = b"s" * 32
PROPOSAL = {"title": "Cache the catalog", "body": "It is static; cache it for a minute.", "area": "code", "refs": ["https://github.com/xaviercallens/HiveMind/pull/1"]}


def refused(code: str, *args, **kw) -> None:  # noqa: ANN002, ANN003
    with pytest.raises(CollabError) as caught:
        check_publish(*args, **kw)
    assert caught.value.code == code, caught.value


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# role -> kind -> topic
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_every_role_has_a_known_set_of_kinds_and_a_prosumer_is_all_of_them() -> None:
    assert set(ROLE_KINDS) == {"consumer", "producer", "prosumer", "admin"}
    assert ROLE_KINDS["prosumer"] == ALL_KINDS == ROLE_KINDS["admin"]
    assert all(kinds <= ALL_KINDS and kinds for kinds in ROLE_KINDS.values())


@pytest.mark.parametrize("role,kind", [("consumer", BEE), ("consumer", PRODUCER), ("producer", AGENT)])
def test_a_role_cannot_publish_as_a_kind_it_is_not(role: str, kind: str) -> None:
    refused("kind_not_allowed_for_role", role, kind, "human_agent", "io.hivemind.amcp.message", {"text": "hi"})


def test_unknown_roles_kinds_and_topics_are_refused() -> None:
    refused("unknown_role", None, HUMAN, "human_agent", "io.hivemind.amcp.message", {})
    refused("unknown_role", "visitor", HUMAN, "human_agent", "io.hivemind.amcp.message", {})
    refused("unknown_kind", "prosumer", "root", "human_agent", "io.hivemind.amcp.message", {})
    refused("unknown_topic", "prosumer", HUMAN, "nowhere", "io.hivemind.amcp.message", {})


def test_a_consumer_cannot_write_on_the_bee_market_or_bee_coordination() -> None:
    for topic in ("bee_trading", "bee_coord"):
        refused("topic_not_allowed_for_kind", "consumer", HUMAN, topic, "io.hivemind.amcp.trade.offer", {})
    check_publish("producer", BEE, "bee_trading", "io.hivemind.amcp.trade.offer", {})  # the topic schema, not this policy, checks trade fields


def test_the_open_topic_accepts_every_kind_and_unlisted_event_types_pass_through_to_the_topic_schema() -> None:
    for kind in ALL_KINDS:
        check_publish("prosumer", kind, "human_agent", "io.hivemind.amcp.problem_proposal", {"equation": "x"})
    assert TOPIC_WRITERS["human_agent"] == ALL_KINDS


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# collaboration events
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_well_formed_proposal_from_a_human_an_agent_or_a_bee_is_accepted() -> None:
    for role, kind in (("consumer", HUMAN), ("consumer", AGENT), ("producer", BEE)):
        check_publish(role, kind, "human_agent", "io.hivemind.amcp.dev.proposal", PROPOSAL)


def test_only_a_human_records_a_decision() -> None:
    decision = {"ref": "https://github.com/xaviercallens/HiveMind/pull/1", "decision": "accept", "reason": "tests pass"}
    check_publish("consumer", HUMAN, "human_agent", "io.hivemind.amcp.dev.decision", decision)
    for role, kind in (("consumer", AGENT), ("producer", BEE), ("producer", PRODUCER), ("prosumer", AGENT)):
        refused("event_type_not_allowed_for_kind", role, kind, "human_agent", "io.hivemind.amcp.dev.decision", decision)
    check_publish(None, HUMAN, "human_agent", "io.hivemind.amcp.dev.decision", decision, via_gateway=False)
    refused("event_type_not_allowed_for_kind", None, AGENT, "human_agent", "io.hivemind.amcp.dev.decision", decision, via_gateway=False)


@pytest.mark.parametrize("payload,why", [
    ({}, "missing"), ({**PROPOSAL, "area": "everything"}, "one of"), ({**PROPOSAL, "title": ""}, "non-empty"), ({**PROPOSAL, "title": "x" * 121}, "longer"),
    ({**PROPOSAL, "body": "bell\x07"}, "control"), ({**PROPOSAL, "refs": ["http://insecure.example"]}, "https"), ({**PROPOSAL, "refs": ["https://a.example"] * 6}, "at most 5"),
    ({**PROPOSAL, "extra": 1}, "unexpected"), ({**PROPOSAL, "title": 5}, "non-empty string"),
])
def test_malformed_collaboration_payloads_are_refused_with_a_reason(payload: dict, why: str) -> None:
    with pytest.raises(CollabError, match=why):
        check_publish("prosumer", AGENT, "human_agent", "io.hivemind.amcp.dev.proposal", payload)


def test_newlines_and_tabs_are_allowed_in_text_but_other_control_characters_are_not() -> None:
    check_publish("consumer", HUMAN, "human_agent", "io.hivemind.amcp.message", {"text": "line one\nline two\tindented"})
    with pytest.raises(CollabError):
        check_publish("consumer", HUMAN, "human_agent", "io.hivemind.amcp.message", {"text": "esc\x1b[31m"})
    check_publish("consumer", HUMAN, "human_agent", "io.hivemind.amcp.message", {})  # the gateway's historical default stays valid


def test_the_github_bridge_event_is_for_trusted_components_only_and_belongs_on_the_open_topic() -> None:
    event = {"repo": "xaviercallens/HiveMind", "kind": "pull_request", "action": "opened", "number": 309, "title": "Join page", "url": "https://github.com/x/y/pull/309", "actor": "someone"}
    refused("direct_only", "prosumer", AGENT, "human_agent", "io.hivemind.amcp.github.event", event)
    check_publish(None, AGENT, "human_agent", "io.hivemind.amcp.github.event", event, via_gateway=False)
    refused("event_type_not_allowed_for_kind", None, HUMAN, "human_agent", "io.hivemind.amcp.github.event", event, via_gateway=False)
    refused("wrong_topic", None, AGENT, "campaigns", "io.hivemind.amcp.github.event", event, via_gateway=False)


def test_bee_reports_need_a_bee_or_a_producer_and_a_bounded_sample_size() -> None:
    report = {"bee_id": "hivemind/bee-stars", "metric": "p50_s", "value": "0.2", "n": 3}
    check_publish("producer", BEE, "human_agent", "io.hivemind.amcp.bee.report", report)
    refused("event_type_not_allowed_for_kind", "consumer", HUMAN, "human_agent", "io.hivemind.amcp.bee.report", report)
    with pytest.raises(CollabError):
        check_publish("producer", BEE, "human_agent", "io.hivemind.amcp.bee.report", {**report, "n": -1})
    with pytest.raises(CollabError):
        check_publish("producer", BEE, "human_agent", "io.hivemind.amcp.bee.report", {**report, "n": True})


def test_the_event_table_covers_the_documented_types() -> None:
    assert set(COLLAB_EVENTS) == {f"io.hivemind.amcp.{t}" for t in ("message", "dev.proposal", "dev.review_request", "dev.review", "dev.decision", "bee.report", "github.event")}


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the gateway endpoint
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def make_client(bus=None) -> tuple[TestClient, dict[str, str]]:  # noqa: ANN001
    client = TestClient(create_router_app(RouterStore(":memory:", SECRET), amcp_bus=bus))
    keys = {role: client.post("/v1/hivemind/register", json={"handle": f"{role}-user", "role": role}).json()["api_key"] for role in ("consumer", "producer", "prosumer")}
    return client, keys


def publish(client: TestClient, key: str, **body):  # noqa: ANN003, ANN201
    return client.post("/v1/hivemind/amcp/publish", headers={"Authorization": f"Bearer {key}"}, json=body)


def test_a_consumer_cannot_pose_as_a_bee_through_the_gateway() -> None:
    client, keys = make_client()
    r = publish(client, keys["consumer"], topic_key="bee_trading", event_type="io.hivemind.amcp.trade.offer", sender_kind="bee_specialist",
                payload={"order_id": "o1", "side": "OFFER", "capacity_1k": 1, "price_per_1k": 0.1})
    assert r.status_code == 403 and r.json()["error"]["code"] == "kind_not_allowed_for_role"
    assert client.get("/v1/hivemind/amcp/events?topic=bee_trading").json()["count"] == 0  # nothing reached the public feed


def test_the_gateway_checks_the_topic_schema_before_storing_anything() -> None:
    client, keys = make_client()
    r = publish(client, keys["prosumer"], topic_key="bee_coord", event_type="io.hivemind.amcp.anything", sender_kind="bee_specialist", payload={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "schema_invalid"
    assert client.get("/v1/hivemind/amcp/events").json()["count"] == 0


def test_a_human_message_is_stored_and_reports_that_no_bus_is_configured() -> None:
    client, keys = make_client()
    r = publish(client, keys["consumer"], event_type="io.hivemind.amcp.dev.proposal", payload=PROPOSAL)
    assert r.status_code == 200 and r.json()["bus"] == {"status": "not_configured"} and r.json()["sender_kind"] == "human_researcher"
    assert client.get("/v1/hivemind/amcp/events").json()["count"] == 1


def test_an_agent_may_not_record_a_decision_through_the_gateway() -> None:
    client, keys = make_client()
    r = publish(client, keys["consumer"], event_type="io.hivemind.amcp.dev.decision", sender_kind="ai_agent",
                payload={"ref": "https://github.com/x/y/pull/1", "decision": "accept", "reason": "looks fine"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "event_type_not_allowed_for_kind"


def test_when_a_bus_is_configured_the_signed_event_is_published_and_a_broker_outage_does_not_lose_the_event() -> None:
    bus = Mock(return_value={"topic": "hivemind.amcp.human_agent.v1", "partition": 2, "offset": 41})
    client, keys = make_client(bus)
    r = publish(client, keys["consumer"], event_type="io.hivemind.amcp.message", payload={"text": "hello"})
    assert r.json()["bus"] == {"status": "published", "topic": "hivemind.amcp.human_agent.v1", "partition": 2, "offset": 41}
    assert bus.call_args.args[0].event_type == "io.hivemind.amcp.message"
    bus.side_effect = RuntimeError("broker down")
    r = publish(client, keys["consumer"], event_type="io.hivemind.amcp.message", payload={"text": "again"})
    assert r.status_code == 200 and r.json()["bus"] == {"status": "unavailable"}
    assert client.get("/v1/hivemind/amcp/events").json()["count"] == 2  # both are in the public feed


def test_publishing_is_rate_limited_per_account() -> None:
    from hivemind.router.gateway import AMCP_PUBLISH_LIMIT

    client, keys = make_client()
    codes = [publish(client, keys["consumer"], event_type="io.hivemind.amcp.message", payload={"text": str(i)}).status_code for i in range(AMCP_PUBLISH_LIMIT + 3)]
    assert codes[:AMCP_PUBLISH_LIMIT] == [200] * AMCP_PUBLISH_LIMIT and codes[AMCP_PUBLISH_LIMIT:] == [429] * 3
    assert publish(client, keys["producer"], event_type="io.hivemind.amcp.message", payload={"text": "other account"}).status_code == 200


def test_an_unauthenticated_publish_is_refused() -> None:
    client, _ = make_client()
    assert client.post("/v1/hivemind/amcp/publish", json={"event_type": "io.hivemind.amcp.message"}).status_code == 401


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# deployment wiring and the development keyring
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_bus_from_env_is_off_with_a_reason_unless_broker_and_keyring_are_both_configured() -> None:
    publish_fn, reason = bus_from_env({})
    assert publish_fn is None and "HIVEMIND_KAFKA_BOOTSTRAP" in reason or "confluent" in reason.lower()
    publish_fn, reason = bus_from_env({"HIVEMIND_KAFKA_BOOTSTRAP": "127.0.0.1:1"})
    assert publish_fn is None and ("HMAC" in reason or "confluent" in reason.lower())


def test_dev_keyring_is_created_once_private_and_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "home" / "dev_keyring"
    with pytest.raises(KafkaConfigError, match="--init-dev-keyring"):
        dev_keyring(path)
    first = dev_keyring(path, create=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and first.active_kid == "dev"
    again = dev_keyring(path, create=True)
    assert again.keys["dev"] == first.keys["dev"]
    os.chmod(path, 0o644)
    with pytest.raises(KafkaConfigError, match="readable by other users"):
        dev_keyring(path)


def test_auto_offset_reset_defaults_to_the_old_behaviour_and_is_validated() -> None:
    assert KafkaSettings.from_env({"HIVEMIND_KAFKA_BOOTSTRAP": "b:1"}).consumer_config()["auto.offset.reset"] == "earliest"
    assert KafkaSettings.from_env({"HIVEMIND_KAFKA_BOOTSTRAP": "b:1", "HIVEMIND_KAFKA_AUTO_OFFSET_RESET": "LATEST"}).consumer_config()["auto.offset.reset"] == "latest"
    with pytest.raises(KafkaConfigError):
        KafkaSettings(bootstrap="b:1", auto_offset_reset="whenever")
