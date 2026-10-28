#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Failure-injection proof for the five-node cluster (3 voters + 2 broker-only, topics replication 5 / min in-sync 3) started by ``scripts/amcp_kafka_cluster.py local up``.

Publishes AMCP events with the project's own acknowledged-write transport (idempotent producer, acks=all) while stopping and restarting nodes, then reads everything back.
The invariants checked:

See PHASES: two simultaneous failures are tolerated; a third failure (replicas below min in-sync) or loss of the controller quorum refuses writes; everything
recovers; every acknowledged event is read back in order exactly once and no refused event is present.

  python scripts/amcp_kafka_cluster.py local up --dir /tmp/k5 --total 5 --voters 3 --replication 5 --min-isr 3
  python scripts/amcp_kafka_quorum_proof.py --dir /tmp/k5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm.amcp_kafka_campaign_bus import build_amcp_kafka_event  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import (  # noqa: E402
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    KafkaBrokerTransport,
    KafkaSettings,
)

BOOTSTRAP = ",".join(f"127.0.0.1:{29092 + 100 * i}" for i in range(5))  # extra, absent brokers in the list are harmless


def harness(args: argparse.Namespace, *words: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "amcp_kafka_cluster.py"), "local", *words, "--dir", args.dir, "--total", "5"], check=True, capture_output=True, text=True)


def publish(ring: HMACKeyring, run: str, tag: str, count: int, acked: list[str], failed: list[str], timeout_s: float) -> float:
    settings = KafkaSettings(bootstrap=BOOTSTRAP, client_id="quorum-proof")
    transport = KafkaBrokerTransport(settings, flush_timeout_s=timeout_s)
    publisher = AMCPPublisher(transport, ring)
    started = time.time()
    for i in range(count):
        event = build_amcp_kafka_event(topic_key="human_agent", event_type="io.hivemind.amcp.message", sender_id="quorum-proof", sender_kind="ai_agent", subject_id=run,
                                       payload={"text": f"{tag}-{i}"})
        try:
            publisher.publish(event)
            acked.append(event.event_id)
        except Exception as exc:  # noqa: BLE001 - a refused write is the expected outcome in phase B
            failed.append(event.event_id)
            print(f"   refused: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            break
    transport.close()
    return time.time() - started


def read_back(ring: HMACKeyring, run: str, expect: int, timeout_s: float = 60.0) -> list[str]:
    settings = KafkaSettings(bootstrap=BOOTSTRAP, group_id=f"quorum-proof-{uuid.uuid4().hex[:8]}", client_id="quorum-proof-reader")
    transport = KafkaBrokerTransport(settings)
    consumer = AMCPConsumer(transport, EnvelopeVerifier(ring), topics=["human_agent"], poll_timeout_s=1.0)
    got: list[str] = []
    consumer.register_topic("human_agent", lambda v: got.append(v.event.event_id) if v.event.subject_id == run else None)
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(got) < expect:
        consumer.run_once()
    transport.close()
    return got


PHASES = [
    # (label, nodes to stop, nodes to start before it, writes must be acknowledged: True / False / None = observe and report only)
    ("all five up", [], [], True),
    ("voter 3 and broker 5 down: 2 of 3 voters, 3 of 5 replicas (two failures)", [3, 5], [], True),
    ("broker 4 down as well: controller quorum still 2 of 3 but only 2 replicas, below min in-sync 3 -> refused", [4], [], False),
    ("broker 4 back: 3 replicas again", [], [4], True),
    ("brokers 4, 5 up and voter 1 only: controller quorum LOST, replicas still 3 (observed, not asserted)", [2], [5], None),
    ("everything back", [], [2, 3], True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    ring = HMACKeyring(keys={"proof": os.urandom(32)}, active_kid="proof")
    run = f"quorum-{uuid.uuid4().hex[:8]}"
    acked: list[str] = []
    failed: list[str] = []
    checks: dict[str, bool] = {}
    observed: dict[str, str] = {}
    for label, stop, start, must_ack in PHASES:
        print(f"- {label}", flush=True)
        for n in start:  # first: a broker cannot finish starting while the controller quorum is gone
            harness(args, "start-node", str(n))
        for n in stop:
            harness(args, "stop-node", str(n))
        time.sleep(10)
        before = len(acked)
        seconds = publish(ring, run, label[:8], args.count if must_ack else 3, acked, failed, 45 if must_ack else 30)
        got_n = len(acked) - before
        print(f"   acknowledged {got_n}, {seconds:.1f}s", flush=True)
        if must_ack is None:
            observed[label] = f"{got_n} acknowledged"
            continue
        checks[f"{label}: " + ("all acknowledged" if must_ack else "refused, none acknowledged")] = (got_n == args.count) if must_ack else (got_n == 0)
    harness(args, "start-node", "2")
    got = read_back(ring, run, len(acked))
    checks["no acknowledged event lost"] = set(acked) <= set(got)
    checks["no duplicates"] = len(got) == len(set(got))
    checks["order kept"] = [e for e in got if e in set(acked)] == acked
    checks["no refused event present"] = not (set(failed) & set(got))
    print(json.dumps({"run": run, "acknowledged": len(acked), "read_back": len(got), "observed": observed, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
