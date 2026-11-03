#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP end-to-end smoke test with a REAL model: no mocks, no canned answers.

    real GGUF model (llama.cpp server, CPU)  <-  real producer shield (HTTP, endpoint token)
        <-  real gateway (escrow hold, dispatch, metering, settlement, hash-chained ledger)  <-  a consumer request

It starts a llama.cpp server for the GGUF file you give it, puts the shield in front of it, registers a producer and a
consumer on a fresh in-memory gateway, sends two different prompts and checks, from the responses and the ledger, that:
  * the answers are non-empty, differ from each other and came from the model (not a constant);
  * the consumer was debited, the producer credited, the fee paid, and no HMT was created or lost;
  * the hash-chained ledger verifies; and a request for a model nobody serves is refused without charge.
It runs on CPU on purpose: it needs no GPU memory, so it never disturbs a live producer (the T4 keeps >= 4,300 MiB free).

Usage:
    python scripts/mvp_e2e_smoke.py --gguf /path/to/model.gguf [--llama-server /path/to/llama-server] [--report out.json]
Exit code 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

from hivemind.router import RouterStore, create_router_app  # noqa: E402
from hivemind.router.hivemind_producer_shield import create_shield_app  # noqa: E402

PROMPTS = [
    "Write a Python function is_palindrome(s) that ignores case and spaces. Reply with code only.",
    "In one sentence, what does the HTTP status code 404 mean?",
]
TIER = "coder-lite"  # public model id served by this smoke test (catalog tier)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def wait_port(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="path to a GGUF model file (for example Qwen2.5-Coder 1.5B Q4)")
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server", help="path to the llama.cpp server binary")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--startup-timeout", type=float, default=240.0)
    ap.add_argument("--report", help="write the JSON report here")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {"model_file": Path(args.gguf).name, "checks": {}, "requests": []}
    checks: dict[str, bool] = report["checks"]
    llama_port, shield_port = free_port(), free_port()
    local_name = "mvp-smoke-model"
    endpoint_token = secrets.token_urlsafe(32)
    llama = None
    server = None
    try:
        # 1. a real model server on CPU (-ngl 0: no GPU memory is touched)
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"llama-smoke-{llama_port}.log"
        llama = subprocess.Popen(
            [args.llama_server, "-m", args.gguf, "--host", "127.0.0.1", "--port", str(llama_port), "-c", "2048", "-ngl", "0",
             "-t", str(args.threads), "--alias", local_name],
            stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        )
        checks["model_server_started"] = wait_http(f"http://127.0.0.1:{llama_port}/health", args.startup_timeout)
        if not checks["model_server_started"]:
            report["error"] = f"llama-server did not become healthy; see {log_path}"
            return _finish(report, args)

        # 2. the real producer shield in front of it, served over real HTTP
        import uvicorn

        shield = create_shield_app(f"http://127.0.0.1:{llama_port}", endpoint_token, allowed_models=[local_name])
        server = uvicorn.Server(uvicorn.Config(shield, host="127.0.0.1", port=shield_port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        checks["shield_started"] = wait_port(shield_port, 30.0)
        try:
            no_token = httpx.get(f"http://127.0.0.1:{shield_port}/v1/models", timeout=5.0)
            with_token = httpx.get(f"http://127.0.0.1:{shield_port}/v1/models", headers={"Authorization": f"Bearer {endpoint_token}"}, timeout=30.0)
            checks["shield_refuses_requests_without_the_token"] = no_token.status_code in (401, 403)
            checks["shield_serves_the_real_model_with_the_token"] = with_token.status_code == 200 and local_name in with_token.text
        except httpx.HTTPError:
            checks["shield_refuses_requests_without_the_token"] = False
            checks["shield_serves_the_real_model_with_the_token"] = False

        # 3. the real gateway with a real ledger
        store = RouterStore(":memory:", secrets.token_bytes(32))
        client = TestClient(create_router_app(store, allow_private_producers=True, starter_grant_hmt=0.0))
        prod = client.post("/v1/hivemind/register", json={"handle": "smoke-producer", "role": "producer"}).json()
        cons = client.post("/v1/hivemind/register", json={"handle": "smoke-consumer", "role": "consumer"}).json()
        store.grant(cons["account_id"], 50.0, "smoke-test-funds")
        beat = client.post(
            "/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {prod['api_key']}"},
            json={"endpoint_url": f"http://127.0.0.1:{shield_port}", "models": [{"id": TIER, "local_name": local_name}],
                  "cost_per_1k": 0.2, "capacity_1k": 40.0, "execution_class": "cpu-smoke", "endpoint_token": endpoint_token},
        )
        checks["producer_registered"] = beat.status_code == 200
        minted_before = store.total_minted()

        # 4. real requests, real answers
        answers: list[str] = []
        for prompt in PROMPTS:
            t0 = time.monotonic()
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {cons['api_key']}"},
                json={"model": f"hivemind/{TIER}", "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "temperature": 0},
            )
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            text = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content", "") if r.status_code == 200 else ""
            answers.append(text)
            report["requests"].append({"prompt": prompt, "status": r.status_code, "seconds": round(time.monotonic() - t0, 2),
                                       "answer_chars": len(text), "usage": body.get("usage"), "answer_preview": text[:160]})
        checks["both_requests_ok"] = all(q["status"] == 200 for q in report["requests"])
        checks["answers_non_empty"] = all(len(a.strip()) > 10 for a in answers)
        checks["answers_differ_so_not_a_constant"] = len(set(answers)) == len(answers)
        checks["code_prompt_answered_with_code"] = "def " in answers[0]

        # 5. the money: debit, credit, fee, conservation, hash chain
        spent = 50.0 - store.balance(cons["account_id"])
        earned = store.balance(prod["account_id"])
        report["ledger"] = {"consumer_spent_hmt": round(spent, 6), "producer_earned_hmt": round(earned, 6)}
        checks["consumer_was_charged"] = spent > 0
        checks["producer_was_paid_less_than_charged"] = 0 < earned < spent
        checks["no_hmt_created_or_lost"] = abs(store.total_minted() - minted_before) < 1e-9 and abs(store.total_balances() - store.total_minted()) < 1e-6
        checks["no_hold_left_open"] = store.active_hold_count(cons["account_id"]) == 0
        checks["ledger_hash_chain_verifies"] = bool(store.verify_chain())

        # 6. a model nobody serves is refused and costs nothing
        before = store.balance(cons["account_id"])
        miss = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {cons['api_key']}"},
                           json={"model": "hivemind/moe-coder", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        checks["unserved_model_refused_without_charge"] = miss.status_code >= 400 and abs(store.balance(cons["account_id"]) - before) < 1e-9
        return _finish(report, args)
    finally:
        if server is not None:
            server.should_exit = True
        if llama is not None:
            llama.terminate()
            try:
                llama.wait(timeout=15)
            except subprocess.TimeoutExpired:
                llama.kill()


def _finish(report: dict[str, Any], args: argparse.Namespace) -> int:
    ok = bool(report["checks"]) and all(report["checks"].values())
    report["passed"] = ok
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
