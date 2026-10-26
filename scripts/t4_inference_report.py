#!/usr/bin/env python3
"""Summarise what the T4 node's telemetry says about the inference service (read-only; standard library only).

Reads ``t4_node.jsonl`` (one snapshot per ~15 s), ``events.jsonl`` and ``governor.jsonl`` from the telemetry directory and prints
only what they support: request volume, in-flight concurrency, GPU memory/utilisation/temperature while serving, gateway-visible tiers,
unit health, restarts, preemption notices and governor yields. It does not infer latency (nginx stub_status has none).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry"))
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.dir / "t4_node.jsonl").read_text().splitlines() if x.strip()]
    ev = [json.loads(x) for x in (a.dir / "events.jsonl").read_text().splitlines() if x.strip()] if (a.dir / "events.jsonl").exists() else []
    gov = [json.loads(x) for x in (a.dir / "governor.jsonl").read_text().splitlines() if x.strip()] if (a.dir / "governor.jsonl").exists() else []
    span_min = (rows[-1]["t"] - rows[0]["t"]) / 60
    print(f"window: {time.strftime('%H:%M', time.gmtime(rows[0]['t']))}-{time.strftime('%H:%M', time.gmtime(rows[-1]['t']))} UTC, {len(rows)} snapshots over {span_min:.0f} min")

    ng = [r["nginx"] for r in rows if r["nginx"].get("ok")]
    total = ng[-1]["requests_total"] - ng[0]["requests_total"]
    print(f"nginx: {total} requests handled ({total / (span_min * 60):.2f}/s average). These include the gateway's endpoint checks and the monitors' polls, "
          "so this is NOT a count of user inferences.")
    infl = [n["in_flight"] for n in ng]
    busy = [x for x in infl if x > 0]
    print(f"in-flight: busy in {len(busy)}/{len(infl)} samples ({100 * len(busy) / len(infl):.1f}%), max {max(infl)}; "
          "a sample is a 15 s snapshot, so requests shorter than ~15 s can fall between samples")

    g = [r["gpu"] for r in rows if "used_mib" in r["gpu"]]
    print(f"GPU memory {min(x['used_mib'] for x in g)}-{max(x['used_mib'] for x in g)} MiB of {g[0]['total_mib']}; utilisation p50 {pct([x['util_pct'] for x in g], .5):.0f}% "
          f"p95 {pct([x['util_pct'] for x in g], .95):.0f}% max {max(x['util_pct'] for x in g)}%; temperature {min(x['temp_c'] for x in g)}-{max(x['temp_c'] for x in g)} C; "
          f"power p95 {pct([x['power_w'] for x in g], .95):.0f} W")
    load = [r["loadavg"][0] for r in rows]
    print(f"host load (1 min) median {st.median(load):.1f}, max {max(load):.1f} on 8 vCPUs")

    tiers = Counter(tuple(r["gateway"]["live_tiers"]) for r in rows)
    print("gateway-visible tiers:", {", ".join(k): v for k, v in tiers.items()})
    bad = Counter()
    for r in rows:
        for u, s in r["units"].items():
            if s["enabled"] == "enabled" and s["active"] != "active" and not u.startswith(("hivemind-live-bee", "hivemind-live-shield-bee", "hivemind-live-agent-bee", "hivemind-live-shield-coder", "hivemind-live-agent-coder")):
                bad[u] += 1
    print("enabled-but-inactive samples (legacy units excluded, they are inactive by design until the cutover):", dict(bad) or "none")
    print("events:", dict(Counter(e["event"] for e in ev)), "| preemption notices:", sum(e["event"] == "PREEMPTION_NOTICE" for e in ev))
    y = [e for e in gov if e["event"] == "yield"]
    real = [e for e in y if e.get("had_boinc_gpu_task")]
    print(f"governor yields: {len(y)} logged, {len(real)} with a real BOINC GPU task (only these measure yield latency)")
    for e in real:  # always list them with time and trigger, so a stale "none yet" can never be carried forward
        print(f"  {time.strftime('%H:%M:%S', time.gmtime(e['t']))}Z trigger={e.get('trigger')} in_flight={e.get('in_flight')} ended_by_governor={e.get('ended_by_governor')} gpu_freed_ms={e.get('gpu_freed_ms')}")
    print("governor states seen:", dict(Counter(json.loads(x)["state"] for x in [json.dumps({'state': e.get('state')}) for e in gov if e["event"] == "snapshot"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
