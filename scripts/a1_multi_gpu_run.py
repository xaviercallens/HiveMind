#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion A1: one tier served by two physical GPUs through the production gateway, one producer killed mid-run.

    HIVEMIND_API_KEY=... python scripts/a1_multi_gpu_run.py --model hivemind/granite-moe --producer t4=86cd37a1 --producer rtx2070=ef1f1643 \
        --kill-cmd "sudo systemctl stop hivemind-live-shield-user" --restore-cmd "sudo systemctl start hivemind-live-shield-user" --out artifacts/mvp/A1_multi_gpu_serving.json

``passed`` needs: both GPUs answered before the kill, no failed request after it, nothing answered by the killed GPU afterwards, and the balance change equal to the sum of the charges.
The restore command always runs. Requests are short (public text only) and cost the key's own points.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
PROMPTS = ["The capital of France is", "Water boils at", "The largest planet is", "A prime number is", "Light travels at", "The first moon landing was in"]


def label(producer_id: str | None, producers: dict[str, str]) -> str:
    for name, prefix in producers.items():
        if producer_id and producer_id.startswith(prefix):
            return name
    return f"other:{(producer_id or 'none')[:8]}"


def summarize(rows: list[dict[str, Any]], producers: dict[str, str]) -> dict[str, Any]:
    by: dict[str, int] = {}
    for r in rows:
        if r["status"] == 200:
            name = label(r.get("producer_id"), producers)
            by[name] = by.get(name, 0) + 1
    latencies = sorted(r["seconds"] for r in rows if r["status"] == 200)
    codes: dict[str, int] = {}
    for r in rows:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
    return {"requests": len(rows), "status_codes": codes, "non_200": sum(r["status"] != 200 for r in rows), "answered_by": by,
            "p50_s": round(latencies[len(latencies) // 2], 2) if latencies else None, "max_s": round(latencies[-1], 2) if latencies else None,
            "charged_hmt": round(sum(r.get("charged", 0.0) for r in rows), 6)}


def decide(before: dict[str, Any], after: dict[str, Any], killed: str, survivor: str, balance_delta: float, min_requests: int = 20, tol: float = 1e-3) -> tuple[bool, list[str]]:
    problems = []
    if before["requests"] < min_requests or after["requests"] < min_requests:
        problems.append(f"fewer than {min_requests} requests in a phase")
    if before["answered_by"].get(killed, 0) < 1 or before["answered_by"].get(survivor, 0) < 1:
        problems.append(f"both GPUs did not serve before the kill: {before['answered_by']}")
    if after["non_200"]:
        problems.append(f"{after['non_200']} failed requests after the kill: {after['status_codes']}")
    if after["answered_by"].get(killed, 0):
        problems.append(f"the killed GPU still answered {after['answered_by'][killed]} requests")
    if after["answered_by"].get(survivor, 0) < 1:
        problems.append("the surviving GPU answered nothing after the kill")
    charged = before["charged_hmt"] + after["charged_hmt"]
    if abs(balance_delta - charged) > tol:
        problems.append(f"balance changed by {balance_delta:.6f} but charges sum to {charged:.6f}")
    return not problems, problems


def burst(client: httpx.Client, model: str, n: int, workers: int, max_tokens: int) -> list[dict[str, Any]]:
    def one(i: int) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            r = client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": max_tokens, "temperature": 0}, timeout=300)
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            h = d.get("hivemind") or {}
            return {"status": r.status_code, "seconds": time.monotonic() - t0, "producer_id": h.get("producer_id"), "charged": float(h.get("charged_hmt") or 0.0), "error": (d.get("error") or {}).get("code")}
        except httpx.HTTPError as exc:
            return {"status": 0, "seconds": time.monotonic() - t0, "error": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, range(n)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--producer", action="append", required=True, help="name=account-id-prefix; the first is the one that gets killed")
    ap.add_argument("--kill-cmd", required=True)
    ap.add_argument("--restore-cmd", required=True)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts/mvp/A1_multi_gpu_serving.json")
    a = ap.parse_args(argv)
    producers = dict(p.split("=", 1) for p in a.producer)
    killed, survivor = list(producers)[0], list(producers)[1]
    client = httpx.Client(base_url=GATEWAY, headers={"Authorization": f"Bearer {os.environ['HIVEMIND_API_KEY']}"})
    start_balance = client.get("/v1/hivemind/me", timeout=60).json()["balance"]
    before_rows = burst(client, a.model, a.requests, a.workers, a.max_tokens)
    kill = subprocess.run(a.kill_cmd, shell=True, capture_output=True, text=True)  # noqa: S602 - the operator's own command line
    time.sleep(a.settle_s)
    try:
        after_rows = burst(client, a.model, a.requests, a.workers, a.max_tokens)
    finally:
        restore = subprocess.run(a.restore_cmd, shell=True, capture_output=True, text=True)  # noqa: S602
    time.sleep(3)
    end_balance = client.get("/v1/hivemind/me", timeout=60).json()["balance"]
    before, after = summarize(before_rows, producers), summarize(after_rows, producers)
    passed, problems = decide(before, after, killed, survivor, start_balance - end_balance)
    if kill.returncode:
        passed, problems = False, [f"kill command failed: {kill.stderr[:100]}", *problems]
    result = {"criterion": "A1", "model": a.model, "gateway_revision_note": "production gateway", "producers": producers, "killed": killed, "kill_cmd": a.kill_cmd,
              "distinct_gpus": sum(1 for p in producers if before["answered_by"].get(p)), "before_kill": before, "after_kill": after,
              "balance_delta_hmt": round(start_balance - end_balance, 6), "restore_exit": restore.returncode, "passed": passed, "reason": "; ".join(problems)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
