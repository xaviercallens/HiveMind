#!/usr/bin/env python3
"""Live end-to-end run of the HiveMind OpenRouter against a real local Ollama server.

Starts the gateway in-process (uvicorn, loopback, temp SQLite, random secret), registers a producer
and two consumers, advertises Ollama through scripts/hivemind_producer_agent.py --once, grants HMT,
then calls the gateway with the official openai client and checks HMT conservation.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from openai import OpenAI

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
PUBLIC_MODEL = "hivemind/qwen2.5-coder-1.5b"
TIMEOUT_S = float(os.environ.get("ROUTER_E2E_TIMEOUT_S", "400"))


def free_port() -> int:
    """Return an unused loopback TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    secret = secrets.token_bytes(32)
    admin_key = secrets.token_urlsafe(24)
    tmp = tempfile.TemporaryDirectory()
    store = RouterStore(Path(tmp.name) / "router.db", secret)
    app = create_router_app(
        store,
        http_client=httpx.Client(timeout=TIMEOUT_S),
        allow_private_producers=True,
        admin_key_hash=store.key_digest(admin_key),
    )
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        print("gateway did not start", file=sys.stderr)
        return 1

    api = httpx.Client(base_url=base, timeout=TIMEOUT_S)

    def register(handle: str, role: str) -> tuple[str, str]:
        r = api.post("/v1/hivemind/register", json={"handle": handle, "role": role})
        r.raise_for_status()
        body = r.json()
        return body["account_id"], body["api_key"]

    prod_id, prod_key = register("live-producer", "producer")
    c1_id, c1_key = register("live-consumer-1", "consumer")
    c2_id, c2_key = register("live-consumer-2", "consumer")
    for key, price in ((c1_key, 4.0), (c2_key, 3.0)):
        api.post(
            "/v1/hivemind/consumer/max-price",
            json={"max_price_per_1k": price},
            headers={"Authorization": f"Bearer {key}"},
        ).raise_for_status()
    # only funded accounts bid in the market (zero-balance sybils cannot move the price), so fund both consumers
    for account_id in (c1_id, c2_id):
        api.post(
            "/v1/hivemind/admin/grant",
            json={"account_id": account_id, "amount": 10.0, "ref": "live-e2e"},
            headers={"X-Admin-Key": admin_key},
        ).raise_for_status()

    agent = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "hivemind_producer_agent.py"),
            "--gateway", base, "--endpoint", OLLAMA,
            "--model", f"{PUBLIC_MODEL.removeprefix('hivemind/')}={LOCAL_MODEL}",
            "--cost-per-1k", "0.5", "--capacity-1k", "1000", "--once",
        ],
        env={**os.environ, "HIVEMIND_API_KEY": prod_key},
        capture_output=True, text=True, timeout=60, check=False,
    )
    print("producer agent rc:", agent.returncode, agent.stdout.strip().replace("\n", " "), agent.stderr.strip())
    if agent.returncode != 0:
        return 1

    client = OpenAI(base_url=base + "/v1", api_key=c1_key, timeout=TIMEOUT_S, max_retries=0)
    print("models:", [m.id for m in client.models.list().data])
    completion = client.chat.completions.create(
        model=PUBLIC_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
        temperature=0,
        max_tokens=16,
    )
    print("reply:", repr(completion.choices[0].message.content))
    extra = completion.model_dump().get("hivemind") or getattr(completion, "hivemind", None)
    print("hivemind billing:", json.dumps(extra, sort_keys=True))
    print("usage:", completion.usage)

    bal = {name: store.balance(acct) for name, acct in
           (("consumer1", c1_id), ("consumer2", c2_id), ("producer", prod_id),
            ("validators", "sys:validators"), ("burn", "sys:burn"))}
    print("balances:", json.dumps(bal, sort_keys=True))
    market = api.get("/v1/hivemind/market", params={"model": PUBLIC_MODEL}, headers={"Authorization": f"Bearer {c1_key}"}).json()
    print("market certificate:", json.dumps(market["certificate"], sort_keys=True))
    print("market mechanism/price:", market["equilibrium"]["mechanism"], market["equilibrium"]["price_per_1k"])

    total = sum(bal.values())
    assert abs(total - store.total_minted()) < 1e-9, (total, store.total_minted())
    assert abs(total - 20.0) < 1e-9
    assert bal["consumer1"] < 10.0 and bal["consumer2"] == 10.0 and bal["producer"] > 0.0
    assert store.verify_chain()
    assert market["certificate"]["win_win"] is True
    assert extra is not None and abs(extra["charged_hmt"] - (10.0 - bal["consumer1"])) < 1e-9
    print(f"CONSERVATION OK: sum balances {total:.12f} == minted {store.total_minted():.12f}; chain verified")
    server.should_exit = True
    thread.join(timeout=5)
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
