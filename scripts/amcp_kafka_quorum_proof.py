#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Failure-injection proof for the three-node cluster started by ``scripts/amcp_kafka_cluster.py local up``.

Publishes AMCP events with the project's own acknowledged-write transport (idempotent producer, acks=all) while stopping and restarting nodes, then reads everything back.
The invariants checked:

  A. with 3 of 3 nodes up, and with 2 of 3 up (one failure), publishing succeeds;
  B. with only 1 of 3 up (quorum lost) publishing FAILS: an event is never acknowledged without a majority;
  C. after the nodes return, publishing succeeds again;
  D. every event the broker acknowledged is read back, in order, exactly once, and no event that failed is present.

  python scripts/amcp_kafka_cluster.py local up --dir /tmp/k3
  python scripts/amcp_kafka_quorum_proof.py --dir /tmp/k3
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

BOOTSTRAP = "127.0.0.1:29092,127.0.0.1:29192,127.0.0.1:29292"


def harness(args: argparse.Namespace, *words: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "amcp_kafka_cluster.py"), "local", *words, "--dir", args.dir], check=True, capture_output=True, text=True)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()
    ring = HMACKeyring(keys={"proof": os.urandom(32)}, active_kid="proof")
    run = f"quorum-{uuid.uuid4().hex[:8]}"
    acked: list[str] = []
    failed: list[str] = []
    result: dict[str, object] = {"run": run}

    print("A1. three of three up")
    before = len(acked)
    t = publish(ring, run, "3of3", args.count, acked, failed, 20)
    result["A1_3of3_acked"], result["A1_seconds"] = len(acked) - before, round(t, 2)

    print("A2. node 3 stopped: two of three up (one failure tolerated)")
    harness(args, "stop-node", "3")
    time.sleep(8)
    before = len(acked)
    t = publish(ring, run, "2of3", args.count, acked, failed, 40)
    result["A2_2of3_acked"], result["A2_seconds"] = len(acked) - before, round(t, 2)

    print("B. node 2 stopped as well: one of three up (quorum lost), the write must be refused")
    harness(args, "stop-node", "2")
    time.sleep(8)
    before, refused_before = len(acked), len(failed)
    t = publish(ring, run, "1of3", 3, acked, failed, 40)
    result["B_1of3_acked"], result["B_1of3_refused"], result["B_seconds"] = len(acked) - before, len(failed) - refused_before, round(t, 2)

    print("C. nodes 2 and 3 back")
    started = time.time()
    harness(args, "start-node", "2")
    harness(args, "start-node", "3")
    time.sleep(10)
    result["C_restart_seconds"] = round(time.time() - started, 1)
    before = len(acked)
    t = publish(ring, run, "back", args.count, acked, failed, 40)
    result["C_after_restore_acked"], result["C_seconds"] = len(acked) - before, round(t, 2)

    print("D. read everything back")
    got = read_back(ring, run, len(acked))
    result["D_acked_total"], result["D_read_back"] = len(acked), len(got)
    result["D_none_lost"] = set(acked) <= set(got)
    result["D_no_duplicates"] = len(got) == len(set(got))
    result["D_in_order"] = [e for e in got if e in set(acked)] == acked
    result["D_refused_event_absent"] = not (set(failed) & set(got))
    checks = {
        "A both 3of3 and 2of3 publish": result["A1_3of3_acked"] == args.count and result["A2_2of3_acked"] == args.count,
        "B quorum lost: write refused, never acknowledged": result["B_1of3_acked"] == 0 and result["B_1of3_refused"] >= 1,
        "C recovers after the nodes return": result["C_after_restore_acked"] == args.count,
        "D acknowledged events: none lost, no duplicates, in order, refused one absent": all(result[k] for k in ("D_none_lost", "D_no_duplicates", "D_in_order", "D_refused_event_absent")),
    }
    result["checks"] = checks
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
