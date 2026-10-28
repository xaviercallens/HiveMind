# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live-broker integration test for the AMCP Kafka transport.

Skipped unless ``HIVEMIND_KAFKA_BOOTSTRAP`` points at a reachable Kafka-compatible broker and the optional
``confluent-kafka`` client is installed (``pip install 'project-hivemind[kafka]'``). The broker must allow topic
auto-creation (or have the AMCP topics and ``hivemind.amcp.dlq.v1`` created beforehand).
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS, build_amcp_kafka_event
from hivemind.swarm.amcp_kafka_transport import (
    DLQ_TOPIC,
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    KafkaBrokerTransport,
    KafkaSettings,
    VerifiedEvent,
    canonical_json,
    kafka_transport_readiness,
    sign_envelope,
)

READINESS = kafka_transport_readiness()
pytestmark = pytest.mark.skipif(READINESS["status"] != "READY", reason=f"live Kafka not configured: {READINESS.get('reason')}")


def test_live_broker_round_trip_dlq_and_per_key_order() -> None:
    ring = HMACKeyring(keys={"it": os.urandom(32)}, active_kid="it")
    base = KafkaSettings.from_env()
    group = f"hivemind-amcp-it-{uuid.uuid4().hex[:8]}"
    settings = KafkaSettings(**{**base.__dict__, "group_id": group})
    producer_side = KafkaBrokerTransport(settings, flush_timeout_s=20.0)
    publisher = AMCPPublisher(producer_side, ring)
    run_tag = uuid.uuid4().hex[:8]
    bee = f"hivemind/bee-it-{run_tag}"
    sent: list[str] = []
    for n in range(5):
        ev = build_amcp_kafka_event(
            topic_key="bee_trading",
            event_type="io.hivemind.amcp.trade.offer",
            sender_id=bee,
            sender_kind="bee_specialist",
            payload={"order_id": f"{run_tag}-{n}", "side": "OFFER", "capacity_1k": 1.0, "price_per_1k": 0.5},
        )
        ack = publisher.publish(ev)
        assert ack["offset"] >= 0
        sent.append(ev.event_id)
    forged = sign_envelope(ev, ring)
    forged["data"]["payload"]["price_per_1k"] = 99.0
    producer_side.publish(KAFKA_TOPICS["bee_trading"], bee.encode(), canonical_json(forged))
    producer_side.close()

    consumer_side = KafkaBrokerTransport(settings, flush_timeout_s=20.0)
    consumer = AMCPConsumer(consumer_side, EnvelopeVerifier(ring), topics=["bee_trading"], poll_timeout_s=1.0)
    got: list[str] = []

    def on_offer(v: VerifiedEvent) -> None:
        if v.event.sender_id == bee:
            got.append(v.event.event_id)

    consumer.register("io.hivemind.amcp.trade.offer", on_offer)
    deadline = time.time() + 60.0
    while time.time() < deadline and (len(got) < len(sent) or not consumer.stats.dead_lettered.get("bad_signature")):
        consumer.run_once()
    consumer_side.close()
    assert got == sent
    assert consumer.stats.dead_lettered.get("bad_signature", 0) >= 1

    dlq_settings = KafkaSettings(**{**base.__dict__, "group_id": group + "-dlq"})
    dlq_reader = KafkaBrokerTransport(dlq_settings)
    dlq_reader.subscribe([DLQ_TOPIC])
    reasons: list[str] = []
    deadline = time.time() + 30.0
    while time.time() < deadline and "bad_signature" not in reasons:
        reasons += [json.loads(r.value)["reason"] for r in dlq_reader.poll(50, 1.0)]
    dlq_reader.close()
    assert "bad_signature" in reasons
