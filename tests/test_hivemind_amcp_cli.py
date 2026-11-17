# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""scripts/hivemind_amcp.py: the direct tier of the AMCP bus. Offline against the in-memory transport; the live-broker scenario skips without a broker."""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest

from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS, build_amcp_kafka_event
from hivemind.swarm.amcp_kafka_transport import (
    DLQ_TOPIC,
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
    KafkaBrokerTransport,
    KafkaSettings,
    kafka_transport_readiness,
)

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_amcp.py"
_spec = importlib.util.spec_from_file_location("hivemind_amcp_cli", _PATH)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

KEY_HEX = "ab" * 32
PROPOSAL = {"title": "Cache the catalog", "body": "Static data; cache it.", "area": "code"}
REPORT = {"bee_id": "hivemind/bee-stars", "metric": "p50_s", "value": "0.2", "n": 3}


class Shared:
    """One in-memory log shared by every CLI call in a test; ``close`` must not close it for the next call."""

    def __init__(self, inner: InMemoryTransport) -> None:
        self.inner = inner

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self.inner, name)

    def close(self) -> None:
        return None


@pytest.fixture()
def bus(monkeypatch: pytest.MonkeyPatch) -> InMemoryTransport:
    memory = InMemoryTransport(num_partitions=3)
    monkeypatch.setenv("HIVEMIND_KAFKA_BOOTSTRAP", "unused:1")
    monkeypatch.setenv("HIVEMIND_AMCP_HMAC_KEYS", f"k1:{KEY_HEX}")
    monkeypatch.setattr(cli, "KafkaBrokerTransport", lambda *a, **k: Shared(memory))
    return memory


def say(*argv: str) -> int:
    return cli.main(["say", *argv])


def test_the_subcommands_accept_their_own_timeout_and_payload_file_options(tmp_path: Path) -> None:
    """Regression: `listen --timeout 90` was rejected because --timeout only existed before the subcommand."""
    parser = cli.build_parser()
    assert parser.parse_args(["listen", "--timeout", "90", "--count", "1"]).timeout == 90.0
    assert parser.parse_args(["listen"]).timeout == 60.0 and parser.parse_args(["check", "--timeout", "3"]).timeout == 3.0
    assert parser.parse_args(["provision", "--timeout", "3", "--replication", "3", "--min-isr", "2"]).min_isr == 2
    said = parser.parse_args(["say", "--as", "agent", "--timeout", "5", "--payload-file", str(tmp_path / "p.json")])
    assert said.timeout == 5.0 and said.payload_file.endswith("p.json") and said.ttl_s == 3600.0
    with pytest.raises(SystemExit) as usage:  # an unknown option is still a usage error, not silently ignored
        parser.parse_args(["listen", "--no-such-option"])
    assert usage.value.code == 2


def test_say_then_listen_round_trips_signed_events_in_order_and_marks_them_untrusted(bus: InMemoryTransport, capsys: pytest.CaptureFixture[str]) -> None:
    assert say("--as", "agent", "--sender-id", "agent:test", "--type", "io.hivemind.amcp.dev.proposal", "--payload-json", json.dumps(PROPOSAL)) == 0
    assert say("--as", "bee", "--sender-id", "bee:test", "--type", "io.hivemind.amcp.bee.report", "--payload-json", json.dumps(REPORT)) == 0
    capsys.readouterr()
    assert cli.main(["listen", "--replay", "--topics", "human_agent", "--count", "2", "--timeout", "5"]) == 0
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert [e["from"] for e in events] == ["agent:test", "bee:test"] and [e["kind"] for e in events] == ["ai_agent", "bee_specialist"]
    assert all(e["untrusted"] is True and e["signed_by"] == "k1" for e in events)
    assert "data, not instructions" in captured.err and json.loads(captured.err.splitlines()[-1])["dead_lettered"] == {}


def test_a_forged_event_on_the_bus_is_dead_lettered_and_never_shown(bus: InMemoryTransport, capsys: pytest.CaptureFixture[str]) -> None:
    ring = HMACKeyring(keys={"k1": bytes.fromhex(KEY_HEX)}, active_kid="k1")
    event = build_amcp_kafka_event(topic_key="human_agent", event_type="io.hivemind.amcp.message", sender_id="agent:evil", sender_kind="ai_agent", payload={"text": "do X"})
    AMCPPublisher(bus, ring).publish(event)
    bus.publish(KAFKA_TOPICS["human_agent"], b"k", b'{"specversion":"1.0","id":"x","type":"io.hivemind.amcp.message","source":"s","data":{}}')
    assert cli.main(["listen", "--replay", "--topics", "human_agent", "--timeout", "1"]) == 0  # no --count: both records are read before the deadline
    out = capsys.readouterr()
    assert len(out.out.splitlines()) == 1 and "agent:evil" in out.out  # the signed one is shown, the unsigned one is not
    dead = [json.loads(r.value) for r in bus.records(DLQ_TOPIC)]
    assert [d["reason"] for d in dead] == ["unsigned"]


def test_policy_refusals_happen_before_any_broker_is_contacted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("HIVEMIND_KAFKA_BOOTSTRAP", raising=False)
    decision = json.dumps({"ref": "https://github.com/x/y/pull/1", "decision": "accept", "reason": "ok"})
    assert say("--as", "agent", "--type", "io.hivemind.amcp.dev.decision", "--payload-json", decision) == 2
    assert "event_type_not_allowed_for_kind" in capsys.readouterr().err
    assert say("--as", "human", "--topic", "bee_trading", "--type", "io.hivemind.amcp.trade.offer", "--payload-json", "{}") == 2
    assert "topic_not_allowed_for_kind" in capsys.readouterr().err
    assert say("--as", "agent", "--type", "io.hivemind.amcp.dev.proposal", "--payload-json", "[1]") == 2
    assert "JSON object" in capsys.readouterr().err
    assert say("--as", "agent", "--type", "io.hivemind.amcp.dev.proposal", "--payload-json", json.dumps({**PROPOSAL, "area": "x"})) == 2


def test_a_valid_event_without_broker_settings_fails_with_a_clear_message_not_a_traceback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("HIVEMIND_KAFKA_BOOTSTRAP", raising=False)
    assert say("--as", "agent", "--type", "io.hivemind.amcp.dev.proposal", "--payload-json", json.dumps(PROPOSAL)) == 2
    assert "HIVEMIND_KAFKA_BOOTSTRAP is not set" in capsys.readouterr().err


def test_check_reports_instead_of_raising_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("HIVEMIND_KAFKA_BOOTSTRAP", raising=False)
    monkeypatch.delenv("HIVEMIND_AMCP_HMAC_KEYS", raising=False)
    assert cli.main(["check"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False and report["broker"] == "unreachable" and "broker_error" in report and "keyring_error" in report
    assert set(report["topics_missing"]) == {*KAFKA_TOPICS.values(), DLQ_TOPIC}


def test_listen_without_a_handler_would_dead_letter_so_every_listened_topic_gets_one(bus: InMemoryTransport) -> None:
    assert say("--as", "human", "--type", "io.hivemind.amcp.message", "--text", "hello") == 0
    assert cli.main(["listen", "--replay", "--topics", "human_agent,bee_coord", "--count", "1", "--timeout", "3"]) == 0
    assert bus.records(DLQ_TOPIC) == []


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# live broker (skipped unless HIVEMIND_KAFKA_BOOTSTRAP is set and reachable)
# ---------------------------------------------------------------------------------------------------------------------------------------------------

LIVE = kafka_transport_readiness()["status"] == "READY"


@pytest.mark.skipif(not LIVE, reason="live Kafka not configured: set HIVEMIND_KAFKA_BOOTSTRAP and install confluent-kafka")
def test_live_three_parties_exchange_signed_events_in_order_on_a_real_broker() -> None:
    """A coding agent proposes, a bee reports, the agent asks for review; a listener in another consumer group sees all three, verified and in order."""
    ring = HMACKeyring(keys={"live": os.urandom(32)}, active_kid="live")
    base = KafkaSettings.from_env()
    subject = f"live-{uuid.uuid4().hex[:8]}"  # the partition key on human_agent, so the three events share a partition and keep their order
    pub_t = KafkaBrokerTransport(KafkaSettings(**{**base.__dict__, "client_id": "amcp-live-pub"}), flush_timeout_s=20.0)
    publisher = AMCPPublisher(pub_t, ring)
    sent = []
    for etype, kind, sender, payload in (
        ("io.hivemind.amcp.dev.proposal", "ai_agent", "agent:live-test", PROPOSAL),
        ("io.hivemind.amcp.bee.report", "bee_specialist", "bee:live-test", REPORT),
        ("io.hivemind.amcp.dev.review_request", "ai_agent", "agent:live-test", {"ref": "https://github.com/x/y/pull/1", "ask": "please review"}),
    ):
        ev = build_amcp_kafka_event(topic_key="human_agent", event_type=etype, sender_id=sender, sender_kind=kind, subject_id=subject, payload=payload)
        publisher.publish(ev)
        sent.append(ev.event_id)
    pub_t.close()

    sub_t = KafkaBrokerTransport(KafkaSettings(**{**base.__dict__, "group_id": f"amcp-live-{uuid.uuid4().hex[:8]}", "client_id": "amcp-live-sub"}))
    consumer = AMCPConsumer(sub_t, EnvelopeVerifier(ring), topics=["human_agent"], poll_timeout_s=1.0)
    got: list[str] = []
    consumer.register_topic("human_agent", lambda v: got.append(v.event.event_id) if v.event.subject_id == subject else None)
    import time

    deadline = time.time() + 60.0
    while time.time() < deadline and len(got) < len(sent):
        consumer.run_once()
    sub_t.close()
    assert got == sent
