#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Failure-injection proof on the REAL cluster: stops and starts the actual Spot VMs with gcloud while publishing acknowledged AMCP events over the WireGuard overlay.

Run it from a host on the overlay (the T4). It stops ONLY the VMs named below, and restarts every VM it stopped, even on error.

  python scripts/amcp_kafka_real_proof.py --project PROJECT --voters hm-kafka-1:us-east4-a hm-kafka-2:us-east4-b hm-kafka-3:us-east4-c

Phases: all up; one voter VM stopped (must still acknowledge); a second voter VM stopped (quorum lost, must be refused); both restarted (must recover, takes minutes: the VMs boot,
the units start and the controller re-elects). Finally every acknowledged event is read back in order, once.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amcp_kafka_quorum_proof as proof  # noqa: E402


def gcloud(project: str, *words: str) -> None:
    subprocess.run(["gcloud", "compute", "instances", *words, "--project", project, "--quiet"], check=True, capture_output=True, text=True, timeout=300)


def wait_acknowledged(ring: object, run: str, acked: list[str], failed: list[str], deadline_s: float) -> float | None:
    """Keep trying one event at a time until one is acknowledged; returns the seconds it took, or None."""
    start = time.time()
    while time.time() - start < deadline_s:
        before = len(acked)
        proof.publish(ring, run, "recover", 1, acked, failed, 20)  # type: ignore[arg-type]
        if len(acked) > before:
            return time.time() - start
        time.sleep(5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--voters", nargs=3, required=True, metavar="NAME:ZONE")
    ap.add_argument("--bootstrap", default="10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092")
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    proof.BOOTSTRAP = args.bootstrap
    vms = [v.split(":", 1) for v in args.voters]
    ring = proof.HMACKeyring(keys={"proof": os.urandom(32)}, active_kid="proof")
    run = f"real-{uuid.uuid4().hex[:8]}"
    acked: list[str] = []
    failed: list[str] = []
    report: dict[str, object] = {"run": run, "bootstrap": args.bootstrap}
    stopped: list[list[str]] = []
    try:
        print("1. all three voters up", flush=True)
        proof.publish(ring, run, "all3", args.count, acked, failed, 30)
        report["all3_acked"] = len(acked)

        print(f"2. stopping {vms[2][0]} (one voter down)", flush=True)
        gcloud(args.project, "stop", vms[2][0], "--zone", vms[2][1])
        stopped.append(vms[2])
        time.sleep(20)
        before = len(acked)
        proof.publish(ring, run, "two", args.count, acked, failed, 60)
        report["one_down_acked"] = len(acked) - before

        print(f"3. stopping {vms[1][0]} too (quorum lost)", flush=True)
        gcloud(args.project, "stop", vms[1][0], "--zone", vms[1][1])
        stopped.append(vms[1])
        time.sleep(20)
        before = len(acked)
        proof.publish(ring, run, "one", 3, acked, failed, 40)
        report["two_down_acked"] = len(acked) - before
        report["two_down_refused"] = len(failed)

        print("4. restarting both", flush=True)
        t0 = time.time()
        for name, zone in list(stopped):
            gcloud(args.project, "start", name, "--zone", zone)
        stopped.clear()
        took = wait_acknowledged(ring, run, acked, failed, 600)
        report["recovery_seconds"] = round(time.time() - t0, 1) if took is not None else None
    finally:
        for name, zone in stopped:  # never leave a VM stopped because this script died
            try:
                gcloud(args.project, "start", name, "--zone", zone)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not restart {name}: {exc}", file=sys.stderr)
    print("5. read back", flush=True)
    got = proof.read_back(ring, run, len(acked), timeout_s=120)
    report.update(acknowledged=len(acked), read_back=len(got))
    checks = {
        "all voters up: acknowledged": report.get("all3_acked") == args.count,
        "one voter VM stopped: still acknowledged": report.get("one_down_acked") == args.count,
        "two voter VMs stopped: refused, none acknowledged": report.get("two_down_acked") == 0 and int(report.get("two_down_refused", 0)) >= 1,
        "recovered after the VMs came back": report.get("recovery_seconds") is not None,
        "none lost, no duplicates, in order": set(acked) <= set(got) and len(got) == len(set(got)) and [e for e in got if e in set(acked)] == acked,
        "refused event absent": not (set(failed) & set(got)) if failed else True,
    }
    report["checks"] = checks
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
