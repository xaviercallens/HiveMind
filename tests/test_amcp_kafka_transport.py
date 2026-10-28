# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the AMCP Kafka transport: signing, DLQ, replay window, at-least-once commits, per-key order.

The real broker is external I/O, so ``KafkaBrokerTransport`` is exercised here against a fake
``confluent_kafka`` module that records configs and simulates delivery reports. The live-broker
test lives in ``test_amcp_kafka_broker_integration.py`` and is skipped unless a broker is configured.
"""

from __future__ import annotations

import base64
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS, InMemoryKafkaBridge, build_amcp_kafka_event
from hivemind.swarm.amcp_kafka_transport import (
    DLQ_TOPIC,
    MAX_ENVELOPE_BYTES,
    TOPIC_SCHEMAS,
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeRejected,
    EnvelopeVerifier,
    FieldRule,
    HMACKeyring,
    InMemoryTransport,
    KafkaBrokerTransport,
    KafkaConfigError,
    KafkaSettings,
    ReplayWindow,
    SchemaError,
    TransportError,
    TransportRecord,
    TransportUnavailable,
    VerifiedEvent,
    build_dead_letter,
    canonical_json,
    kafka_transport_readiness,
    load_kafka_client,
    partition_key_for,
    sign_envelope,
    validate_event_dict,
)

NOW = 1_793_000_000.0
KEY_A = bytes(range(32))
KEY_B = bytes(range(100, 132))


def keyring(active: str = "k1") -> HMACKeyring:
    return HMACKeyring(keys={"k1": KEY_A, "k2": KEY_B}, active_kid=active)


def clock() -> float:
    return NOW


def trade_event(bee: str = "hivemind/coder", side: str = "OFFER", order_id: str = "o1", utc: float = NOW) -> Any:
    return build_amcp_kafka_event(
        topic_key="bee_trading",
        event_type=f"io.hivemind.amcp.trade.{side.lower()}",
        sender_id=bee,
        sender_kind="bee_specialist",
        recipient_id="market_book",
        subject_id="SUBJ-01-PYTHON-RUST-CODER",
        payload={"order_id": order_id, "side": side, "capacity_1k": 10.0, "price_per_1k": 0.5},
        now_utc=utc,
    )


def chat_event(text: str = "hello", utc: float = NOW) -> Any:
    return build_amcp_kafka_event(
        topic_key="human_agent",
        event_type="io.hivemind.amcp.message",
        sender_id="alice",
        sender_kind="human_researcher",
        payload={"text": text},
        now_utc=utc,
    )


def make_pair(transport: InMemoryTransport | None = None, **consumer_kw: Any) -> tuple[InMemoryTransport, AMCPPublisher, AMCPConsumer]:
    t = transport or InMemoryTransport(num_partitions=4)
    pub = AMCPPublisher(t, keyring())
    con = AMCPConsumer(t, EnvelopeVerifier(keyring(), clock=clock), clock=clock, **consumer_kw)
    return t, pub, con


def dlq_reasons(t: InMemoryTransport) -> list[str]:
    return [json.loads(r.value)["reason"] for r in t.records(DLQ_TOPIC)]


# --------------------------------------------------------------------------- signing / verification


def test_signed_event_round_trips_and_reaches_handler() -> None:
    t, pub, con = make_pair()
    got: list[VerifiedEvent] = []
    con.register("io.hivemind.amcp.trade.offer", got.append)
    ack = pub.publish(trade_event())
    assert ack["key"] == "hivemind/coder" and ack["key_id"] == "k1"
    assert con.run_once() == 1
    assert len(got) == 1 and got[0].event.payload["order_id"] == "o1"
    assert got[0].key_id == "k1"
    assert t.committed_offset(ack["topic"], ack["partition"]) == ack["offset"] + 1
    assert t.records(DLQ_TOPIC) == []


def test_forged_signature_goes_to_dlq_never_to_handler() -> None:
    t, _pub, con = make_pair()
    got: list[VerifiedEvent] = []
    con.register("io.hivemind.amcp.trade.offer", got.append)
    env = sign_envelope(trade_event(), keyring())
    env["data"]["payload"]["price_per_1k"] = 0.0001  # tamper after signing
    topic = KAFKA_TOPICS["bee_trading"]
    forged = canonical_json(env)
    ack = t.publish(topic, b"hivemind/coder", forged)
    con.run_once()
    assert got == []
    assert dlq_reasons(t) == ["bad_signature"]
    # the forged record is committed (dead-lettered), so it is not redelivered forever
    assert t.committed_offset(topic, ack["partition"]) == ack["offset"] + 1
    dl = json.loads(t.records(DLQ_TOPIC)[0].value)
    assert dl["value_b64"] == base64.b64encode(forged).decode("ascii")  # raw forged bytes kept for forensics
    assert dl["key_b64"] == base64.b64encode(b"hivemind/coder").decode("ascii")


def test_signature_from_key_not_in_keyring_is_rejected() -> None:
    other = HMACKeyring(keys={"k1": bytes(range(50, 82))}, active_kid="k1")
    t, _pub, con = make_pair()
    con.register("io.hivemind.amcp.message", lambda ev: None)
    t.publish(KAFKA_TOPICS["human_agent"], b"x", canonical_json(sign_envelope(chat_event(), other)))
    rogue_kid = HMACKeyring(keys={"rogue": KEY_A}, active_kid="rogue")
    t.publish(KAFKA_TOPICS["human_agent"], b"x", canonical_json(sign_envelope(chat_event(), rogue_kid)))
    con.run_once()
    assert sorted(dlq_reasons(t)) == ["bad_signature", "unknown_key_id"]
    assert con.stats.delivered == 0


def test_rotated_key_still_verifies() -> None:
    t, _pub, con = make_pair()
    got: list[VerifiedEvent] = []
    con.register("io.hivemind.amcp.message", got.append)
    AMCPPublisher(t, keyring(active="k2")).publish(chat_event())
    con.run_once()
    assert [g.key_id for g in got] == ["k2"]
    assert t.records(DLQ_TOPIC) == []


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"\xff\xfe not json", "malformed"),
        (b"[1, 2, 3]", "malformed"),
        (json.dumps({"hello": "world"}).encode(), "not_cloudevent"),
        (b"x" * (MAX_ENVELOPE_BYTES + 1), "oversize"),
    ],
)
def test_garbage_records_are_dead_lettered(raw: bytes, reason: str) -> None:
    t, _pub, con = make_pair()
    con.register_topic("human_agent", lambda ev: None)
    t.publish(KAFKA_TOPICS["human_agent"], b"k", raw)
    con.run_once()
    assert dlq_reasons(t) == [reason]
    assert con.stats.delivered == 0


def test_unsigned_envelope_is_dead_lettered() -> None:
    t, _pub, con = make_pair()
    con.register_topic("human_agent", lambda ev: None)
    env = sign_envelope(chat_event(), keyring())
    del env["hmsig"]
    t.publish(KAFKA_TOPICS["human_agent"], b"k", canonical_json(env))
    con.run_once()
    assert dlq_reasons(t) == ["unsigned"]
    assert con.stats.dead_lettered == {"unsigned": 1}


def _verify(env: dict[str, Any], topic: str, now: float = NOW) -> VerifiedEvent:
    v = EnvelopeVerifier(keyring(), clock=lambda: now)
    return v.verify(TransportRecord(topic, 0, 0, b"k", canonical_json(env)))


def test_expired_and_future_dated_events_are_rejected() -> None:
    env = sign_envelope(chat_event(utc=NOW - 1000.0), keyring(), ttl_s=300.0)
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "expired"
    future = sign_envelope(chat_event(utc=NOW + 600.0), keyring(), ttl_s=300.0)
    with pytest.raises(EnvelopeRejected) as exc2:
        _verify(future, KAFKA_TOPICS["human_agent"])
    assert exc2.value.reason == "future_dated"
    # positive control: same event inside its lifetime verifies
    fresh = sign_envelope(chat_event(utc=NOW - 10.0), keyring(), ttl_s=300.0)
    assert _verify(fresh, KAFKA_TOPICS["human_agent"]).event.payload == {"text": "hello"}


def test_signer_cannot_mint_an_overlong_lifetime() -> None:
    env = sign_envelope(chat_event(), keyring(), ttl_s=300.0)
    env["hmexpiry"] = NOW + 10 * 86400.0
    env.pop("hmsig")
    env["hmsig"] = keyring().sign(canonical_json(env))  # validly signed but lives 10 days
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "ttl_too_long"
    with pytest.raises(ValueError):
        sign_envelope(chat_event(), keyring(), ttl_s=0.0)


def test_event_signed_for_one_topic_cannot_be_moved_to_another() -> None:
    env = sign_envelope(chat_event(), keyring())
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["bee_trading"])
    assert exc.value.reason == "topic_mismatch"
    assert _verify(env, KAFKA_TOPICS["human_agent"]).event.topic == KAFKA_TOPICS["human_agent"]


def test_envelope_attributes_must_match_signed_data() -> None:
    env = sign_envelope(chat_event(), keyring())
    env["id"] = "amcp-somethingelse"
    env.pop("hmsig")
    env["hmsig"] = keyring().sign(canonical_json(env))
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "envelope_mismatch"
    bad_expiry = sign_envelope(chat_event(), keyring())
    bad_expiry["hmexpiry"] = "tomorrow"
    bad_expiry.pop("hmsig")
    bad_expiry["hmsig"] = keyring().sign(canonical_json(bad_expiry))
    with pytest.raises(EnvelopeRejected) as exc2:
        _verify(bad_expiry, KAFKA_TOPICS["human_agent"])
    assert exc2.value.reason == "unsigned"


def test_signature_field_shape_is_checked() -> None:
    env = sign_envelope(chat_event(), keyring())
    env["hmsig"] = env["hmsig"].upper()
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "bad_signature"
    assert "hex" in exc.value.detail


@settings(max_examples=60, deadline=None)
@given(
    payload=st.dictionaries(
        st.text(min_size=1, max_size=12),
        st.one_of(st.integers(-10**6, 10**6), st.text(max_size=20), st.booleans(), st.none()),
        max_size=6,
    ),
    flip=st.text(min_size=1, max_size=8),
)
def test_signature_roundtrip_and_any_tamper_breaks_it(payload: dict[str, Any], flip: str) -> None:
    ev = build_amcp_kafka_event(
        topic_key="human_agent",
        event_type="io.hivemind.amcp.message",
        sender_id="agent-x",
        sender_kind="ai_agent",
        payload=payload,
        now_utc=NOW,
    )
    env = json.loads(canonical_json(sign_envelope(ev, keyring())))
    assert _verify(env, KAFKA_TOPICS["human_agent"]).event.payload == payload
    tampered = json.loads(json.dumps(env))
    tampered["data"]["payload"]["__tamper__" + flip] = 1
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(tampered, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "bad_signature"


@settings(max_examples=60, deadline=None)
@given(obj=st.dictionaries(st.text(max_size=8), st.one_of(st.integers(), st.text(max_size=8)), max_size=8))
def test_canonical_json_is_order_independent(obj: dict[str, Any]) -> None:
    reordered = dict(reversed(list(obj.items())))
    assert canonical_json(obj) == canonical_json(reordered)
    assert json.loads(canonical_json(obj)) == obj


def test_canonical_json_refuses_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    assert canonical_json({"b": 1, "a": [1.5]}) == b'{"a":[1.5],"b":1}'


# --------------------------------------------------------------------------- schema


def test_schema_accepts_existing_gateway_event_shapes() -> None:
    poll = build_amcp_kafka_event(
        topic_key="campaigns",
        event_type="io.hivemind.amcp.campaign.poll_registered",
        sender_id="alice",
        sender_kind="gpu_producer",
        subject_id="SUBJ-01-PYTHON-RUST-CODER",
        payload={"pledged_vram_gb": 8.0, "availability_slot": "eu_evening_cet", "comment": ""},
    )
    handoff = build_amcp_kafka_event(
        topic_key="bee_coord",
        event_type="io.hivemind.amcp.bee.handoff_certified",
        sender_id="hivemind/bee-stars",
        sender_kind="bee_specialist",
        payload={
            "solver_family": "yoshida4",
            "domain": "astro",
            "recommended_crate": "rusty-sundials-cvode",
            "relative_invariant_drift": 1e-9,
            "improvement_factor": 120.0,
            "verdict": "PASS",
        },
    )
    assert validate_event_dict(poll.to_dict(), poll.topic).subject_id == "SUBJ-01-PYTHON-RUST-CODER"
    assert validate_event_dict(handoff.to_dict(), handoff.topic).payload["verdict"] == "PASS"
    assert validate_event_dict(trade_event().to_dict(), KAFKA_TOPICS["bee_trading"]).sender_id == "hivemind/coder"


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d["payload"].update(side="BID"), "not in"),
        (lambda d: d["payload"].update(capacity_1k=0.0), "> 0.0"),
        (lambda d: d["payload"].update(capacity_1k=True), "finite number"),
        (lambda d: d["payload"].update(capacity_1k=1e9), "<= 100000"),
        (lambda d: d["payload"].pop("price_per_1k"), "missing"),
        (lambda d: d["payload"].update(price_per_1k=1_000.5), "<= 1000"),
        (lambda d: d["payload"].update(order_id=7), "must be a string"),
        (lambda d: d["payload"].update(order_id="x" * 65), "longer than"),
        (lambda d: d.update(event_type="io.hivemind.amcp.trade.swap"), "not allowed"),
        (lambda d: d.update(event_type="com.evil.trade"), "must start with"),
        (lambda d: d.update(sender_kind="root"), "sender_kind"),
        (lambda d: d.update(utc=float("inf")), "utc"),
        (lambda d: d.update(sender_id=""), "sender_id"),
        (lambda d: d.update(payload=[1]), "payload must be"),
        (lambda d: d.update(specversion="0.3"), "specversion"),
        (lambda d: d.update(extra="x"), "unexpected"),
        (lambda d: d.update(topic="hivemind.amcp.other.v1"), "does not match"),
    ],
)
def test_schema_rejects_bad_trade_events(mutate: Any, fragment: str) -> None:
    data = trade_event().to_dict()
    mutate(data)
    with pytest.raises(SchemaError) as exc:
        validate_event_dict(data, KAFKA_TOPICS["bee_trading"])
    assert fragment in str(exc.value)


def test_schema_rejects_unknown_campaign_subject_and_unknown_topic() -> None:
    poll = build_amcp_kafka_event(
        topic_key="campaigns",
        event_type="io.hivemind.amcp.campaign.poll_registered",
        sender_id="alice",
        sender_kind="human_researcher",
        subject_id="SUBJ-99-NOPE",
        payload={"pledged_vram_gb": 1.0, "availability_slot": "eu_evening_cet", "comment": ""},
    )
    with pytest.raises(SchemaError, match="unknown campaign subject"):
        validate_event_dict(poll.to_dict(), poll.topic)
    other = poll.to_dict() | {"topic": "hivemind.amcp.unknown.v1"}
    with pytest.raises(SchemaError, match="no schema"):
        validate_event_dict(other, "hivemind.amcp.unknown.v1")
    with pytest.raises(SchemaError, match="JSON object"):
        validate_event_dict("nope", poll.topic)


def test_field_rule_number_bounds() -> None:
    rule = FieldRule("number", ge=0.0, le=1.0)
    rule.check("x", 0.5)
    with pytest.raises(SchemaError):
        rule.check("x", -0.1)
    with pytest.raises(SchemaError):
        rule.check("x", "0.5")


def test_publisher_refuses_invalid_events_and_mirrors_valid_ones() -> None:
    t = InMemoryTransport()
    bridge = InMemoryKafkaBridge(maxlen_per_topic=8)
    pub = AMCPPublisher(t, keyring(), mirror=bridge)
    bad = build_amcp_kafka_event(
        topic_key="bee_trading",
        event_type="io.hivemind.amcp.trade.offer",
        sender_id="b",
        sender_kind="bee_specialist",
        payload={"side": "OFFER"},
    )
    with pytest.raises(SchemaError):
        pub.publish(bad)
    assert t.records(KAFKA_TOPICS["bee_trading"]) == []
    pub.publish(trade_event())
    assert len(bridge.recent("bee_trading")) == 1
    huge = chat_event(text="x" * MAX_ENVELOPE_BYTES)
    with pytest.raises(SchemaError, match="bytes"):
        pub.publish(huge)
    with pytest.raises(ValueError):
        AMCPPublisher(t, keyring(), ttl_s=10_000.0)


def test_partition_keys_follow_campaign_or_bee_identity() -> None:
    coord = build_amcp_kafka_event(
        topic_key="bee_coord",
        event_type="io.hivemind.amcp.bee.handoff_request",
        sender_id="hivemind/coder",
        sender_kind="bee_specialist",
        correlation_id="coord-1",
        payload={},
    )
    assert partition_key_for(coord) == "coord-1"
    assert partition_key_for(trade_event(bee="hivemind/bee-stars")) == "hivemind/bee-stars"
    assert partition_key_for(chat_event()) == chat_event().subject_id


# --------------------------------------------------------------------------- replay & delivery


def test_replay_of_handled_event_is_dropped_exactly_once() -> None:
    t, pub, con = make_pair()
    got: list[str] = []
    con.register("io.hivemind.amcp.trade.offer", lambda ev: got.append(ev.event.event_id))
    ev = trade_event()
    pub.publish(ev)
    con.run_once()
    # an attacker (or a producer retry) re-sends the exact signed bytes
    original = t.records(ev.topic)[0]
    t.publish(ev.topic, original.key or b"", original.value)
    con.run_once()
    assert got == [ev.event_id]
    assert con.stats.duplicates_dropped == 1
    assert t.records(DLQ_TOPIC) == []  # duplicates are expected under at-least-once, not dead-lettered
    assert t.committed_offset(original.topic, original.partition) == 2


def test_handler_failure_leaves_offset_uncommitted_and_redelivers() -> None:
    t, pub, con = make_pair(max_attempts=5)
    calls: list[str] = []

    def flaky(ev: VerifiedEvent) -> None:
        calls.append(ev.event.event_id)
        if len(calls) == 1:
            raise RuntimeError("transient downstream outage")

    con.register("io.hivemind.amcp.trade.offer", flaky)
    ack = pub.publish(trade_event())
    con.run_once()
    assert t.committed_offset(ack["topic"], ack["partition"]) is None
    assert con.stats.handler_failures == 1 and con.stats.rewinds == 1
    con.run_once()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert t.committed_offset(ack["topic"], ack["partition"]) == 1
    assert con.stats.delivered == 1


def test_poison_event_is_dead_lettered_after_max_attempts() -> None:
    t, pub, con = make_pair(max_attempts=2)

    def always_fails(ev: VerifiedEvent) -> None:
        raise ValueError("cannot apply order")

    con.register("io.hivemind.amcp.trade.offer", always_fails)
    ack = pub.publish(trade_event())
    con.run_once()
    assert t.committed_offset(ack["topic"], ack["partition"]) is None
    con.run_once()
    assert dlq_reasons(t) == ["handler_failed"]
    assert "cannot apply order" in json.loads(t.records(DLQ_TOPIC)[0].value)["detail"]
    assert t.committed_offset(ack["topic"], ack["partition"]) == 1


def test_failure_holds_later_records_of_same_key_to_keep_order() -> None:
    t, pub, con = make_pair(max_attempts=5)
    seen: list[str] = []
    fail_once = {"o1": True}

    def handler(ev: VerifiedEvent) -> None:
        oid = ev.event.payload["order_id"]
        if fail_once.pop(oid, False):
            raise RuntimeError("first attempt fails")
        seen.append(oid)

    con.register("io.hivemind.amcp.trade.offer", handler)
    for oid in ("o1", "o2", "o3"):
        pub.publish(trade_event(order_id=oid))
    con.run_once()
    assert seen == []  # o2, o3 were held behind the failed o1
    con.run_once()
    assert seen == ["o1", "o2", "o3"]


def test_ordering_per_key_across_many_bees() -> None:
    t, pub, con = make_pair(transport=InMemoryTransport(num_partitions=3), max_batch=7)
    seen: dict[str, list[int]] = {}
    con.register(
        "io.hivemind.amcp.trade.offer",
        lambda ev: seen.setdefault(ev.event.sender_id, []).append(int(ev.event.payload["order_id"])),
    )
    bees = [f"hivemind/bee-{i}" for i in range(5)]
    for n in range(20):
        for bee in bees:
            pub.publish(trade_event(bee=bee, order_id=str(n)))
    while con.run_once():
        pass
    assert set(seen) == set(bees)
    assert all(seq == list(range(20)) for seq in seen.values())
    partitions = {t.partition_for(b.encode()) for b in bees}
    assert len(partitions) > 1  # the keys really are spread over partitions


def test_no_handler_goes_to_dlq() -> None:
    t, pub, con = make_pair()
    pub.publish(chat_event())
    con.run_once()
    assert dlq_reasons(t) == ["no_handler"]
    assert con.stats.delivered == 0


def test_topic_fallback_handler_and_registration_errors() -> None:
    _t, pub, con = make_pair()
    got: list[str] = []
    con.register_topic("human_agent", lambda ev: got.append(ev.event.event_type))
    pub.publish(chat_event())
    con.run_once()
    assert got == ["io.hivemind.amcp.message"]
    with pytest.raises(ValueError):
        con.register_topic("human_agent", lambda ev: None)
    with pytest.raises(ValueError):
        con.register_topic("hivemind.amcp.unknown.v1", lambda ev: None)
    con.register("io.hivemind.amcp.x", lambda ev: None)
    with pytest.raises(ValueError):
        con.register("io.hivemind.amcp.x", lambda ev: None)
    with pytest.raises(ValueError):
        con.register("bad.type", lambda ev: None)


class FlakyDLQTransport(InMemoryTransport):
    """In-memory transport whose DLQ publish and commit can be made to fail."""

    def __init__(self) -> None:
        super().__init__(num_partitions=1)
        self.fail_dlq = True
        self.fail_commit = False

    def publish(self, topic: str, key: bytes, value: bytes) -> dict[str, Any]:
        if topic == DLQ_TOPIC and self.fail_dlq:
            raise TransportError("dlq broker unavailable")
        return super().publish(topic, key, value)

    def commit(self, record: TransportRecord) -> None:
        if self.fail_commit:
            raise TransportError("commit timed out")
        super().commit(record)


def test_rejected_record_is_not_committed_when_dlq_publish_fails() -> None:
    t = FlakyDLQTransport()
    _t, _pub, con = make_pair(transport=t)
    ack = t.publish(KAFKA_TOPICS["human_agent"], b"k", b"not json")
    con.run_once()
    assert con.stats.dlq_errors == 1
    assert t.committed_offset(ack["topic"], ack["partition"]) is None
    t.fail_dlq = False
    con.run_once()
    assert dlq_reasons(t) == ["malformed"]
    assert t.committed_offset(ack["topic"], ack["partition"]) == 1


def test_no_handler_and_poison_paths_rewind_when_dlq_is_down() -> None:
    t = FlakyDLQTransport()
    _t, pub, con = make_pair(transport=t, max_attempts=1)
    pub.publish(chat_event())
    con.run_once()
    assert con.stats.rewinds == 1 and con.stats.dlq_errors == 1

    def boom(ev: VerifiedEvent) -> None:
        raise RuntimeError("x")

    con.register("io.hivemind.amcp.message", boom)
    con.run_once()
    assert con.stats.rewinds == 2 and con.stats.handler_failures == 1
    assert t.committed_offset(KAFKA_TOPICS["human_agent"], 0) is None


def test_commit_failure_is_counted_and_redelivery_is_deduped() -> None:
    t = FlakyDLQTransport()
    t.fail_dlq = False
    _t, pub, con = make_pair(transport=t)
    got: list[str] = []
    con.register("io.hivemind.amcp.message", lambda ev: got.append(ev.event.event_id))
    pub.publish(chat_event())
    t.fail_commit = True
    con.run_once()
    assert con.stats.commit_errors == 1
    t.fail_commit = False
    t.seek(KAFKA_TOPICS["human_agent"], 0, 0)  # what a group rebalance does after a lost commit
    con.run_once()
    assert len(got) == 1
    assert con.stats.duplicates_dropped == 1


def test_backpressure_bounds_each_batch() -> None:
    t, pub, con = make_pair(max_batch=4)
    con.register("io.hivemind.amcp.message", lambda ev: None)
    for i in range(10):
        pub.publish(chat_event(text=str(i)))
    sizes = []
    while (n := con.run_once()) > 0:
        sizes.append(n)
    assert sizes == [4, 4, 2]
    assert con.stats.delivered == 10


def test_stop_flag_halts_run_and_rewinds_unprocessed() -> None:
    t, pub, con = make_pair(max_batch=10)
    handled: list[str] = []

    def stop_after_first(ev: VerifiedEvent) -> None:
        handled.append(ev.event.event_id)
        con.stop()

    con.register("io.hivemind.amcp.message", stop_after_first)
    for i in range(3):
        pub.publish(chat_event(text=str(i)))
    stats = con.run(max_batches=100)
    assert con.stopped and len(handled) == 1
    assert stats.rewinds == 1
    assert t.committed_offset(KAFKA_TOPICS["human_agent"], t.partition_for(chat_event().subject_id.encode())) == 1


def test_run_in_thread_stops_on_flag() -> None:
    t, pub, con = make_pair()
    done = threading.Event()
    con.register("io.hivemind.amcp.message", lambda ev: done.set())
    worker = threading.Thread(target=con.run, kwargs={"idle_sleep_s": 0.01})
    worker.start()
    pub.publish(chat_event())
    assert done.wait(5.0)
    con.stop()
    worker.join(5.0)
    assert not worker.is_alive()
    assert con.stats.delivered == 1


def test_consumer_constructor_guards() -> None:
    t = InMemoryTransport()
    v = EnvelopeVerifier(keyring())
    with pytest.raises(ValueError, match="TTL"):
        AMCPConsumer(t, v, replay_window=ReplayWindow(ttl_s=10.0))
    with pytest.raises(ValueError, match="max_batch"):
        AMCPConsumer(t, v, max_batch=0)
    with pytest.raises(ValueError, match="max_attempts"):
        AMCPConsumer(t, v, max_attempts=0)
    with pytest.raises(ValueError, match="unknown"):
        AMCPConsumer(t, v, topics=["hivemind.amcp.unknown.v1"])
    with pytest.raises(ValueError, match="unknown"):
        AMCPConsumer(t, v, topics=[])
    with pytest.raises(ValueError, match="dead-letter"):
        AMCPConsumer(t, v, topics=["human_agent"], dlq_topic=KAFKA_TOPICS["human_agent"])
    with pytest.raises(ValueError):
        EnvelopeVerifier(keyring(), max_ttl_s=0.0)
    assert AMCPConsumer(t, v, topics=["bee_trading"]).topics == (KAFKA_TOPICS["bee_trading"],)


def test_replay_window_ttl_and_capacity() -> None:
    w = ReplayWindow(capacity=2, ttl_s=100.0)
    w.mark("a", 0.0)
    w.mark("b", 1.0)
    w.mark("c", 2.0)
    assert w.evictions == 1 and not w.seen("a", 2.0)
    assert w.seen("b", 50.0) and len(w) == 2
    assert not w.seen("b", 200.0)  # expired out of the window
    with pytest.raises(ValueError):
        ReplayWindow(capacity=0)


# --------------------------------------------------------------------------- in-memory transport


def test_in_memory_transport_retention_and_close() -> None:
    t = InMemoryTransport(num_partitions=1, retention_per_partition=3)
    for i in range(5):
        t.publish("topic-a", b"k", str(i).encode())
    t.subscribe(["topic-a"])
    recs = t.poll(10, 0.0)
    assert [r.offset for r in recs] == [2, 3, 4]
    assert [r.value for r in recs] == [b"2", b"3", b"4"]
    t.close()
    with pytest.raises(TransportError):
        t.publish("topic-a", b"k", b"x")
    with pytest.raises(TransportError):
        t.poll(1, 0.0)
    with pytest.raises(ValueError):
        InMemoryTransport(num_partitions=0)
    with pytest.raises(ValueError):
        InMemoryTransport(retention_per_partition=0)


def test_dead_letter_record_shape() -> None:
    rec = TransportRecord("t", 2, 9, None, b"z" * (70 * 1024))
    dl = json.loads(build_dead_letter(rec, "oversize", "d" * 900, NOW))
    assert dl["reason"] == "oversize" and len(dl["detail"]) == 512
    assert dl["value_truncated"] is True and dl["key_b64"] == ""
    assert (dl["source_topic"], dl["source_partition"], dl["source_offset"]) == ("t", 2, 9)


# --------------------------------------------------------------------------- config & keyring


def test_keyring_from_env_and_guards() -> None:
    env = {"HIVEMIND_AMCP_HMAC_KEYS": f"k1:{KEY_A.hex()}, k2:{KEY_B.hex()}", "HIVEMIND_AMCP_HMAC_ACTIVE_KID": "k2"}
    ring = HMACKeyring.from_env(env)
    assert ring.active_kid == "k2" and set(ring.keys) == {"k1", "k2"}
    assert "keys=" not in repr(ring) and KEY_A.hex() not in repr(ring)
    for bad in (
        {},
        {"HIVEMIND_AMCP_HMAC_KEYS": "k1"},
        {"HIVEMIND_AMCP_HMAC_KEYS": "k1:zz"},
        {"HIVEMIND_AMCP_HMAC_KEYS": "k1:" + "ab" * 8},
        {"HIVEMIND_AMCP_HMAC_KEYS": f"bad id:{KEY_A.hex()}"},
        {"HIVEMIND_AMCP_HMAC_KEYS": f"k1:{KEY_A.hex()}", "HIVEMIND_AMCP_HMAC_ACTIVE_KID": "k9"},
    ):
        with pytest.raises(KafkaConfigError):
            HMACKeyring.from_env(bad)
    with pytest.raises(KafkaConfigError):
        HMACKeyring(keys={}, active_kid="k1")


def test_kafka_settings_from_env() -> None:
    plain = KafkaSettings.from_env({"HIVEMIND_KAFKA_BOOTSTRAP": "127.0.0.1:9092"})
    assert plain.security_protocol == "PLAINTEXT" and plain.sasl_mechanism is None
    pc = plain.producer_config()
    assert pc["acks"] == "all" and pc["enable.idempotence"] is True
    assert plain.consumer_config()["enable.auto.commit"] is False
    sasl = KafkaSettings.from_env(
        {
            "HIVEMIND_KAFKA_BOOTSTRAP": "broker:9093",
            "HIVEMIND_KAFKA_SASL_USERNAME": "bee",
            "HIVEMIND_KAFKA_SASL_PASSWORD": "s3cret-value",
            "HIVEMIND_KAFKA_SSL_CA_LOCATION": "/etc/ssl/ca.pem",
        }
    )
    assert sasl.security_protocol == "SASL_SSL" and sasl.sasl_mechanism == "SCRAM-SHA-512"
    assert sasl.producer_config()["sasl.username"] == "bee"
    assert sasl.consumer_config()["ssl.ca.location"] == "/etc/ssl/ca.pem"
    assert "s3cret-value" not in repr(sasl)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"HIVEMIND_KAFKA_BOOTSTRAP": "   "},
        {"HIVEMIND_KAFKA_BOOTSTRAP": "b:1", "HIVEMIND_KAFKA_SECURITY_PROTOCOL": "HTTP"},
        {"HIVEMIND_KAFKA_BOOTSTRAP": "b:1", "HIVEMIND_KAFKA_SECURITY_PROTOCOL": "SASL_SSL"},
        {
            "HIVEMIND_KAFKA_BOOTSTRAP": "b:1",
            "HIVEMIND_KAFKA_SASL_USERNAME": "u",
            "HIVEMIND_KAFKA_SASL_PASSWORD": "p",
            "HIVEMIND_KAFKA_SASL_MECHANISM": "GSSAPI",
        },
        {
            "HIVEMIND_KAFKA_BOOTSTRAP": "b:1",
            "HIVEMIND_KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
            "HIVEMIND_KAFKA_SASL_USERNAME": "u",
            "HIVEMIND_KAFKA_SASL_PASSWORD": "p",
            "HIVEMIND_KAFKA_SASL_MECHANISM": "PLAIN",
        },
        {"HIVEMIND_KAFKA_BOOTSTRAP": "b:1", "HIVEMIND_KAFKA_SECURITY_PROTOCOL": "SSL", "HIVEMIND_KAFKA_SASL_USERNAME": "u"},
    ],
)
def test_kafka_settings_reject_missing_or_unsafe_config(env: dict[str, str]) -> None:
    with pytest.raises(KafkaConfigError):
        KafkaSettings.from_env(env)
    assert kafka_transport_readiness(env)["status"] == "BLOCKED"


def test_readiness_reports_blocked_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import = importlib.import_module

    def no_kafka(name: str, package: str | None = None) -> Any:
        if name == "confluent_kafka":
            raise ImportError("not installed")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", no_kafka)
    with pytest.raises(TransportUnavailable, match="project-hivemind\\[kafka\\]"):
        load_kafka_client()
    res = kafka_transport_readiness({"HIVEMIND_KAFKA_BOOTSTRAP": "127.0.0.1:9092"})
    assert res["status"] == "BLOCKED" and "confluent-kafka" in res["reason"]


def test_readiness_reports_ready_with_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    fake = FakeKafkaModule()
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda name, package=None: fake if name == "confluent_kafka" else real_import(name, package)
    )
    res = kafka_transport_readiness({"HIVEMIND_KAFKA_BOOTSTRAP": "127.0.0.1:9092"})
    assert res["status"] == "READY" and res["client_version"] == "fake-2.0"
    assert KafkaBrokerTransport.from_env({"HIVEMIND_KAFKA_BOOTSTRAP": "127.0.0.1:9092"}).settings.bootstrap == "127.0.0.1:9092"


# --------------------------------------------------------------------------- broker transport vs fake client


class FakeKafkaException(Exception):
    pass


class FakeErr:
    def __init__(self, code: int) -> None:
        self._code = code

    def code(self) -> int:
        return self._code

    def __str__(self) -> str:
        return f"err-{self._code}"


class FakeMsg:
    def __init__(self, topic: str, partition: int, offset: int, key: bytes | None, value: bytes | None, err: FakeErr | None = None) -> None:
        self._t, self._p, self._o, self._k, self._v, self._e = topic, partition, offset, key, value, err

    def topic(self) -> str:
        return self._t

    def partition(self) -> int:
        return self._p

    def offset(self) -> int:
        return self._o

    def key(self) -> bytes | None:
        return self._k

    def value(self) -> bytes | None:
        return self._v

    def error(self) -> FakeErr | None:
        return self._e


class FakeProducer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = "ok"
        self.pending: list[tuple[Any, FakeMsg]] = []
        self.offset = 0

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any) -> None:
        if self.mode == "buffer_full":
            raise BufferError("queue full")
        msg = FakeMsg(topic, 1, self.offset, key, value)
        self.offset += 1
        self.pending.append((on_delivery, msg))

    def flush(self, timeout: float) -> int:
        if self.mode == "timeout":
            return len(self.pending)
        for cb, msg in self.pending:
            cb(FakeErr(7) if self.mode == "nack" else None, msg)
        self.pending.clear()
        return 0


class FakeConsumer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.subscribed: list[str] = []
        self.queue: list[FakeMsg] = []
        self.commits: list[Any] = []
        self.seeks: list[Any] = []
        self.closed = False
        self.raise_on: set[str] = set()

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    def consume(self, num_messages: int, timeout: float) -> list[FakeMsg]:
        if "consume" in self.raise_on:
            raise FakeKafkaException("broker down")
        out, self.queue = self.queue[:num_messages], self.queue[num_messages:]
        return out

    def commit(self, offsets: list[Any], asynchronous: bool) -> None:
        if "commit" in self.raise_on:
            raise FakeKafkaException("rebalance in progress")
        assert asynchronous is False
        self.commits.extend(offsets)

    def seek(self, tp: Any) -> None:
        if "seek" in self.raise_on:
            raise FakeKafkaException("not assigned")
        self.seeks.append(tp)

    def close(self) -> None:
        self.closed = True


class FakeKafkaModule(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            Producer=FakeProducer,
            Consumer=FakeConsumer,
            TopicPartition=lambda topic, partition, offset: (topic, partition, offset),
            KafkaException=FakeKafkaException,
            KafkaError=SimpleNamespace(_PARTITION_EOF=-191),
            version=lambda: ("fake-2.0", 0x020000),
        )


def broker(mod: FakeKafkaModule | None = None) -> KafkaBrokerTransport:
    return KafkaBrokerTransport(
        KafkaSettings(bootstrap="127.0.0.1:19092"), client_module=mod or FakeKafkaModule(), flush_timeout_s=0.1
    )


def test_broker_transport_publish_uses_idempotent_acks_all_and_reports_offsets() -> None:
    b = broker()
    producer = b._producer
    assert producer.config["acks"] == "all" and producer.config["enable.idempotence"] is True
    ack = b.publish("hivemind.amcp.bee_trading.v1", b"hivemind/coder", b"{}")
    assert ack == {"topic": "hivemind.amcp.bee_trading.v1", "partition": 1, "offset": 0}
    assert b.publish("t", b"k", b"v")["offset"] == 1


@pytest.mark.parametrize(("mode", "fragment"), [("nack", "failed: err-7"), ("timeout", "not confirmed"), ("buffer_full", "produce to")])
def test_broker_transport_publish_raises_on_unconfirmed_delivery(mode: str, fragment: str) -> None:
    b = broker()
    b._producer.mode = mode
    with pytest.raises(TransportError) as exc:
        b.publish("t", b"k", b"v")
    assert fragment in str(exc.value)


def test_broker_transport_consume_commit_seek_close() -> None:
    b = broker()
    with pytest.raises(TransportError, match="subscribe"):
        b.poll(1, 0.0)
    b.subscribe(["a", "b"])
    c = b._consumer
    assert c.subscribed == ["a", "b"] and c.config["enable.auto.commit"] is False
    c.queue = [
        FakeMsg("a", 0, 5, b"k", b"v1"),
        FakeMsg("a", 0, 6, None, None, err=FakeErr(-191)),
        FakeMsg("a", 0, 6, b"k", None),
    ]
    recs = b.poll(10, 0.1)
    assert [(r.offset, r.value) for r in recs] == [(5, b"v1"), (6, b"")]
    b.commit(recs[0])
    assert c.commits == [("a", 0, 6)]
    b.seek("a", 0, 5)
    assert c.seeks == [("a", 0, 5)]
    b.close()
    assert c.closed and b._consumer is None


def test_broker_transport_wraps_client_errors() -> None:
    b = broker()
    b.subscribe(["a"])
    c = b._consumer
    c.queue = [FakeMsg("a", 0, 1, b"k", b"v", err=FakeErr(3))]
    with pytest.raises(TransportError, match="consume error"):
        b.poll(1, 0.0)
    c.raise_on = {"consume", "commit", "seek"}
    with pytest.raises(TransportError, match="consume failed"):
        b.poll(1, 0.0)
    with pytest.raises(TransportError, match="commit failed"):
        b.commit(TransportRecord("a", 0, 1, b"k", b"v"))
    with pytest.raises(TransportError, match="seek failed"):
        b.seek("a", 0, 1)


def test_consumer_end_to_end_over_broker_transport_with_fake_client() -> None:
    b = broker()
    pub = AMCPPublisher(b, keyring())
    ev = trade_event()
    pub.publish(ev)
    sent_value = b._producer.offset  # one message produced
    assert sent_value == 1
    con = AMCPConsumer(b, EnvelopeVerifier(keyring(), clock=clock), topics=["bee_trading"], clock=clock)
    got: list[str] = []
    con.register("io.hivemind.amcp.trade.offer", lambda v: got.append(v.event.event_id))
    con.run_once()  # subscribes
    signed = canonical_json(sign_envelope(ev, keyring()))
    b._consumer.queue = [FakeMsg(ev.topic, 0, 0, b"hivemind/coder", signed), FakeMsg(ev.topic, 0, 1, b"x", b"garbage")]
    con.run_once()
    assert got == [ev.event_id]
    assert b._consumer.commits == [(ev.topic, 0, 1), (ev.topic, 0, 2)]
    assert con.stats.dead_lettered == {"malformed": 1}


def test_validly_signed_but_schema_invalid_event_is_dead_lettered() -> None:
    t, _pub, con = make_pair()
    got: list[VerifiedEvent] = []
    con.register("io.hivemind.amcp.trade.offer", got.append)
    bad = build_amcp_kafka_event(
        topic_key="bee_trading",
        event_type="io.hivemind.amcp.trade.offer",
        sender_id="hivemind/coder",
        sender_kind="bee_specialist",
        payload={"order_id": "o1", "side": "OFFER", "capacity_1k": -5.0, "price_per_1k": 0.5},
        now_utc=NOW,
    )
    # a key holder signs it directly, bypassing the publisher's schema check
    t.publish(bad.topic, b"hivemind/coder", canonical_json(sign_envelope(bad, keyring())))
    con.run_once()
    assert got == []
    assert dlq_reasons(t) == ["schema_invalid"]
    assert "capacity_1k" in json.loads(t.records(DLQ_TOPIC)[0].value)["detail"]


def test_nan_smuggled_into_json_is_rejected_as_malformed() -> None:
    env = sign_envelope(chat_event(), keyring())
    raw = canonical_json(env).replace(b'"text":"hello"', b'"text":NaN')
    with pytest.raises(EnvelopeRejected) as exc:
        EnvelopeVerifier(keyring(), clock=clock).verify(TransportRecord(KAFKA_TOPICS["human_agent"], 0, 0, b"k", raw))
    assert exc.value.reason == "malformed"
    assert "canonicalisable" in exc.value.detail


def test_stats_snapshot_and_direct_settings_guard() -> None:
    _t, pub, con = make_pair()
    con.register("io.hivemind.amcp.message", lambda ev: None)
    pub.publish(chat_event())
    con.run_once()
    snap = con.stats.to_dict()
    assert snap["delivered"] == 1 and snap["committed"] == 1 and snap["polled"] == 1
    snap["dead_lettered"]["x"] = 1
    assert con.stats.dead_lettered == {}  # snapshot is a copy
    with pytest.raises(KafkaConfigError, match="empty"):
        KafkaSettings(bootstrap=" ")


def test_attempt_counter_memory_is_bounded() -> None:
    _t, _pub, con = make_pair()
    for i in range(10_005):
        con._bump_attempts(("t", 0, i))
    assert len(con._attempts) == 10_000
    assert ("t", 0, 0) not in con._attempts and con._bump_attempts(("t", 0, 10_004)) == 2


def test_stop_rewinds_every_partition_so_no_fetched_record_is_lost() -> None:
    t, pub, con = make_pair(max_batch=50)
    seen: list[str] = []

    def stop_after_first(ev: VerifiedEvent) -> None:
        seen.append(ev.event.event_id)
        con.stop()

    con.register("io.hivemind.amcp.trade.offer", stop_after_first)
    sent = []
    for i in range(8):
        ev = trade_event(bee=f"hivemind/bee{i}", order_id=f"o{i}")
        pub.publish(ev)
        sent.append(ev.event_id)
    partitions = {t.partition_for(f"hivemind/bee{i}".encode()) for i in range(8)}
    assert len(partitions) > 1  # the case under test needs records on several partitions
    con.run(max_batches=5)
    assert len(seen) == 1 and con.stats.rewinds == len(partitions)
    resumed = AMCPConsumer(t, EnvelopeVerifier(keyring(), clock=clock), clock=clock)
    resumed.register("io.hivemind.amcp.trade.offer", lambda ev: seen.append(ev.event.event_id))
    resumed.run(max_batches=5)
    assert sorted(seen) == sorted(sent)


def test_event_signed_with_the_maximum_lifetime_verifies() -> None:
    # hmexpiry is rounded to 6 decimals; at these epochs that can exceed utc + MAX_EVENT_TTL_S by < 1 us.
    rejected = []
    for i in range(200):
        utc = NOW + i * 0.1234567891  # sub-microsecond fractions, as time.time() produces
        env = sign_envelope(chat_event(utc=utc), keyring(), ttl_s=3600.0)
        try:
            _verify(env, KAFKA_TOPICS["human_agent"], now=utc)
        except EnvelopeRejected as exc:
            rejected.append(exc.reason)
    assert rejected == []
    # negative control: one second over the cap is still refused
    env = sign_envelope(chat_event(), keyring(), ttl_s=300.0)
    env["hmexpiry"] = NOW + 3601.0
    env.pop("hmsig")
    env["hmsig"] = keyring().sign(canonical_json(env))
    with pytest.raises(EnvelopeRejected) as exc:
        _verify(env, KAFKA_TOPICS["human_agent"])
    assert exc.value.reason == "ttl_too_long"


def test_trade_price_cap_matches_the_gateway_ceiling() -> None:
    from hivemind.router.gateway import MAX_PRICE_CEILING

    rule = TOPIC_SCHEMAS[KAFKA_TOPICS["bee_trading"]].payloads["io.hivemind.amcp.trade.offer"]["price_per_1k"]
    assert rule.le == MAX_PRICE_CEILING
    rule.check("price_per_1k", MAX_PRICE_CEILING)
    with pytest.raises(SchemaError):
        rule.check("price_per_1k", MAX_PRICE_CEILING * 1.001)
