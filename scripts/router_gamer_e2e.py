#!/usr/bin/env python3
"""Live end-to-end run of the gamer path: connector -> gateway -> producer shield -> real local Ollama.

Everything runs on loopback with a temp database. The producer registers through scripts/hivemind_connect.py,
advertises a catalog tier with scripts/hivemind_producer_agent.py (preflight goes through the shield with the endpoint
token), and a consumer calls the gateway with the official openai client. Negative controls: the shield refuses a
request without the token and refuses Ollama's management API. Run: PYTHONPATH=src python scripts/router_gamer_e2e.py
"""

from __future__ import annotations

import importlib.util
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
from types import ModuleType

import httpx
import uvicorn
from openai import OpenAI

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
TIER = os.environ.get("ROUTER_E2E_TIER", "coder-lite")
TIMEOUT_S = float(os.environ.get("ROUTER_E2E_TIMEOUT_S", "400"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(app: object, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))  # type: ignore[arg-type]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("server did not start")
    return server, thread


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env={**os.environ, **(env or {})},
                          capture_output=True, text=True, timeout=120, check=False)


def main() -> int:
    work = tempfile.TemporaryDirectory()
    secret, admin_key = secrets.token_bytes(32), secrets.token_urlsafe(24)
    store = RouterStore(Path(work.name) / "router.db", secret)
    gw_port, shield_port = free_port(), free_port()
    gateway = f"http://127.0.0.1:{gw_port}"
    shield_url = f"http://127.0.0.1:{shield_port}"
    gw_server, gw_thread = serve(create_router_app(store, http_client=httpx.Client(timeout=TIMEOUT_S), allow_private_producers=True,
                                                   admin_key_hash=store.key_digest(admin_key)), gw_port)
    endpoint_token = secrets.token_urlsafe(32)
    shield = load("hivemind_producer_shield")
    sh_server, sh_thread = serve(shield.create_shield_app(OLLAMA, endpoint_token, allowed_models=[LOCAL_MODEL], max_concurrent=2), shield_port)

    # negative controls on the shield itself
    assert httpx.post(shield_url + "/v1/chat/completions", json={"model": LOCAL_MODEL, "messages": []}).status_code == 401
    assert httpx.post(shield_url + "/api/pull", json={"name": "x"}, headers={"Authorization": f"Bearer {endpoint_token}"}).status_code in (404, 405)
    print("shield: no token -> 401, Ollama management API -> 404/405 (both refused)")

    keys: dict[str, str] = {}
    for handle, role in (("gamer-producer", "producer"), ("gamer-consumer", "consumer")):
        keyfile = Path(work.name) / f"{handle}.key"
        done = run_script("hivemind_connect.py", "--gateway", gateway, "register", "--handle", handle, "--role", role, "--save-key", str(keyfile))
        assert done.returncode == 0, done.stderr
        keys[handle] = keyfile.read_text()
    consumer_id = httpx.get(gateway + "/v1/hivemind/me", headers={"Authorization": f"Bearer {keys['gamer-consumer']}"}).json()["account_id"]
    producer_id = httpx.get(gateway + "/v1/hivemind/me", headers={"Authorization": f"Bearer {keys['gamer-producer']}"}).json()["account_id"]
    httpx.post(gateway + "/v1/hivemind/admin/grant", headers={"X-Admin-Key": admin_key},
               json={"account_id": consumer_id, "amount": 10.0, "ref": "gamer-e2e"}).raise_for_status()

    suggest = run_script("hivemind_producer_agent.py", "--gateway", gateway, "--suggest", "--vram-gb", "8")
    print("suggest for 8 GB:", json.loads(suggest.stdout)["recommended"]["tier_id"])
    agent = run_script("hivemind_producer_agent.py", "--gateway", gateway, "--endpoint", shield_url, "--preset", TIER, "--local-name", LOCAL_MODEL,
                       "--once", env={"HIVEMIND_API_KEY": keys["gamer-producer"], "HIVEMIND_ENDPOINT_TOKEN": endpoint_token})
    print("producer agent rc:", agent.returncode, agent.stdout.strip().replace("\n", " "), agent.stderr.strip())
    assert agent.returncode == 0 and endpoint_token not in agent.stdout + agent.stderr

    check = run_script("hivemind_connect.py", "--gateway", gateway, "check", env={"HIVEMIND_API_KEY": keys["gamer-consumer"]})
    print(check.stdout.strip())

    client = OpenAI(base_url=gateway + "/v1", api_key=keys["gamer-consumer"], timeout=TIMEOUT_S, max_retries=0)
    completion = client.chat.completions.create(
        model=f"hivemind/{TIER}", temperature=0, max_tokens=40,
        messages=[{"role": "user", "content": "In Python, write one line that prints Game Over."}],
    )
    reply = completion.choices[0].message.content
    print("reply:", repr(reply))
    billing = completion.model_dump().get("hivemind") or {}
    print("billing:", json.dumps(billing, sort_keys=True))

    bal = {n: store.balance(a) for n, a in (("consumer", consumer_id), ("producer", producer_id), ("validators", "sys:validators"), ("burn", "sys:burn"))}
    print("balances:", json.dumps(bal, sort_keys=True))
    total = sum(bal.values())
    assert reply and bal["consumer"] < 10.0 and bal["producer"] > 0.0
    assert abs(total - store.total_minted()) < 1e-9 and store.verify_chain()
    assert abs(billing["charged_hmt"] - (10.0 - bal["consumer"])) < 1e-9
    print(f"GAMER PATH OK: sum of balances {total:.9f} == minted {store.total_minted():.9f}; chain verified; answer came through the shield")
    for server, thread in ((gw_server, gw_thread), (sh_server, sh_thread)):
        server.should_exit = True
        thread.join(timeout=5)
    work.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
