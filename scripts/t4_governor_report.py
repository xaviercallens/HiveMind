#!/usr/bin/env python3
"""Report on what the T4 governor actually did: is the nightly QLoRA training happening, and how fast does it yield?

Reads ``governor.jsonl`` (any schema: events from before ``schema`` 2 lack ``trigger``, ``ended_by_governor`` and ``other_procs`` and are
read with defaults). Standard library only, read-only.

A "burst" is a ``resume`` (GPU mode set to ``always``) up to the next ``yield``. A burst that ends with ``had_boinc_gpu_task`` false means no
training task was running when the governor yielded (the round had finished or never started); it is reported as "empty", because a high share
of empty bursts at night means the training duty cycle is set by the quiet-period rule and not by the work available.

Usage: t4_governor_report.py [governor.jsonl] [--json] [--since-epoch T]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT = Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl")
P99_MIN_SAMPLES = 20


def load(path: Path, since: float = 0.0) -> list[dict]:
    out = []
    for ln in path.read_text().splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if isinstance(e, dict) and "event" in e and e.get("t", 0) >= since:
            out.append(e)
    return out


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


def summarise(events: list[dict]) -> dict:
    ev = [e for e in events if e["event"] != "snapshot"]
    resumes = [e for e in ev if e["event"] == "resume"]
    yields = [e for e in ev if e["event"] == "yield"]
    bursts, open_resume = [], None
    for e in ev:
        if e["event"] == "resume":
            open_resume = e
        elif e["event"] == "yield" and open_resume is not None:
            bursts.append({"start": open_resume["t"], "seconds": e["t"] - open_resume["t"], "night": open_resume.get("night"),
                           "task": bool(e.get("had_boinc_gpu_task")), "trigger": e.get("trigger", "unknown"), "procs": e.get("other_procs", [])})
            open_resume = None
    real = [y for y in yields if y.get("had_boinc_gpu_task") and y.get("gpu_freed_ms") is not None]
    lat_new = [y["gpu_freed_ms"] for y in real if y.get("ended_by_governor")]
    lat_old = [y["gpu_freed_ms"] for y in real if not y.get("ended_by_governor")]
    per_day: dict[str, float] = defaultdict(float)
    for b in bursts:
        per_day[datetime.fromtimestamp(b["start"], timezone.utc).strftime("%Y-%m-%d")] += b["seconds"] / 60
    culprits = Counter()
    for b in bursts:
        for p in b["procs"]:
            culprits[(p.get("name") or "?") + " " + (p.get("cmd") or "")[:40]] += 1
    n = len(bursts)
    return {
        "events": dict(Counter(e["event"] for e in events)),
        "resumes": len(resumes), "yields": len(yields), "yield_triggers": dict(Counter(y.get("trigger", "unknown") for y in yields)),
        "bursts": n, "bursts_empty": sum(1 for b in bursts if not b["task"]),
        "burst_seconds": {"median": round(statistics.median(b["seconds"] for b in bursts), 1), "max": round(max(b["seconds"] for b in bursts), 1)} if n else None,
        "mode_always_min_per_day": {d: round(m, 1) for d, m in sorted(per_day.items())},
        "yield_with_task": {"n": len(real), "ended_by_governor": len(lat_new), "latency_ms_new": sorted(lat_new), "latency_ms_old": sorted(lat_old),
                            "p50_new": percentile(lat_new, 0.5) if lat_new else None,
                            "p99_new": percentile(lat_new, 0.99) if len(lat_new) >= P99_MIN_SAMPLES else f"n={len(lat_new)} < {P99_MIN_SAMPLES}: too few for a p99"},
        "gpu_trigger_culprits": dict(culprits.most_common(8)),
        "schema2_events": sum(1 for e in events if e.get("schema", 1) >= 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--since-epoch", type=float, default=0.0)
    a = ap.parse_args()
    s = summarise(load(a.path, a.since_epoch))
    if a.json:
        print(json.dumps(s, indent=1))
        return 0
    print(f"resumes {s['resumes']}, yields {s['yields']} (by trigger {s['yield_triggers']}), bursts {s['bursts']} of which empty {s['bursts_empty']}")
    print(f"burst length (s): {s['burst_seconds']}")
    print(f"minutes with GPU mode 'always' per UTC day: {s['mode_always_min_per_day']}")
    y = s["yield_with_task"]
    print(f"yields that stopped a real task: {y['n']} ({y['ended_by_governor']} ended by the governor); latency ms new {y['latency_ms_new']}, "
          f"old {y['latency_ms_old']}; p50(new) {y['p50_new']}; p99(new) {y['p99_new']}")
    print(f"processes seen at gpu-triggered yields (schema >= 2 only; {s['schema2_events']} schema-2 events): {s['gpu_trigger_culprits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
