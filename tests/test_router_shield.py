"""The producer shield: only the gateway (holding the endpoint token) can reach only the two chat routes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_producer_shield.py"
_spec = importlib.util.spec_from_file_location("hivemind_producer_shield", _PATH)
assert _spec is not None and _spec.loader is not None
shield: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shield)

TOKEN = "shield-token-" + "z" * 20
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CHAT = {"model": "qwen2.5-coder:7b", "messages": [{"role": "user", "content": "hi"}]}


def _stream(status: int, payload: bytes, content_type: str = "application/json") -> httpx.Response:
    """A response that behaves like a real server's: an unread byte stream (MockTransport's json= responses are pre-read)."""
    return httpx.Response(status, headers={"content-type": content_type}, stream=httpx.ByteStream(payload))


def _app(handler=None, **kwargs):  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def default(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen2.5-coder:7b"}]})  # read in full by the shield
        body = json.loads(request.content)
        if body.get("stream"):
            return _stream(200, b"data: {}\n\ndata: [DONE]\n\n", "text/event-stream")
        return _stream(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    transport = httpx.MockTransport(handler or default)
    app = shield.create_shield_app("http://upstream.local:11434", TOKEN, transport=transport, **kwargs)
    return TestClient(app), seen


def test_requests_without_the_exact_token_are_refused_and_never_reach_the_model_server() -> None:
    client, seen = _app()
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}, {"Authorization": f"bearer {TOKEN}"}):
        assert client.post("/v1/chat/completions", json=CHAT, headers=headers).status_code == 401
        assert client.get("/v1/models", headers=headers).status_code == 401
    assert seen == []


def test_only_the_two_chat_routes_exist_so_ollama_management_is_not_exposed() -> None:
    client, seen = _app()
    for method, path in (("POST", "/api/pull"), ("DELETE", "/api/delete"), ("POST", "/api/generate"), ("GET", "/api/tags"), ("GET", "/"), ("POST", "/v1/completions")):
        assert client.request(method, path, headers=AUTH, json={}).status_code in (404, 405), (method, path)
    assert seen == []


def test_authorised_chat_and_models_pass_through_including_streams() -> None:
    client, seen = _app()
    assert client.get("/v1/models", headers=AUTH).json()["data"][0]["id"] == "qwen2.5-coder:7b"
    assert client.post("/v1/chat/completions", json=CHAT, headers=AUTH).json()["choices"][0]["message"]["content"] == "ok"
    streamed = client.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH)
    assert streamed.status_code == 200 and streamed.text.endswith("data: [DONE]\n\n")
    assert all("authorization" not in r.headers for r in seen)  # the gateway's token is not forwarded to the model server


def test_oversized_bodies_bad_json_and_unlisted_models_are_refused() -> None:
    client, seen = _app(max_body_bytes=200, allowed_models=["qwen2.5-coder:7b"])
    assert client.post("/v1/chat/completions", content=b"x" * 500, headers={**AUTH, "content-type": "application/json"}).status_code == 413
    assert client.post("/v1/chat/completions", content=b"not json", headers={**AUTH, "content-type": "application/json"}).status_code == 400
    assert client.post("/v1/chat/completions", json={**CHAT, "model": "llama3:70b"}, headers=AUTH).status_code == 403
    assert seen == [] and client.post("/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 200


def test_a_busy_producer_answers_429_at_once_and_recovers() -> None:
    import asyncio

    async def scenario() -> tuple[int, int, int, str]:
        gate = asyncio.Event()

        async def held_open(request: httpx.Request) -> httpx.Response:
            await gate.wait()  # the model server is "busy" until released
            return _stream(200, b'{"ok": true}')

        app = shield.create_shield_app("http://upstream.local", TOKEN, transport=httpx.MockTransport(held_open), max_concurrent=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shield") as client:
            first = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT, headers=AUTH))
            await asyncio.sleep(0.1)
            busy = await client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
            gate.set()
            released = await first
            again = await client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
            return busy.status_code, released.status_code, again.status_code, busy.json()["error"]["code"]

    assert asyncio.run(scenario()) == (429, 200, 200, "busy")


def test_unreachable_model_server_is_a_502_and_does_not_leak_the_slot() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no ollama", request=request)

    client, _ = _app(down, max_concurrent=1)
    for _ in range(3):
        assert client.post("/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 502  # never 429: the slot is freed


def test_weak_tokens_and_bad_limits_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        shield.create_shield_app("http://x", "short")
    with pytest.raises(ValueError):
        shield.create_shield_app("http://x", TOKEN, max_concurrent=0)


def test_models_falls_back_to_the_native_tag_list_for_older_ollama_and_hides_everything_else() -> None:
    def old_ollama(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(404, json={"error": "page not found"})
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:1.5b", "size": 1}, {"name": "all-minilm:latest"}]})

    client, _ = _app(old_ollama)
    listed = client.get("/v1/models", headers=AUTH).json()
    assert [m["id"] for m in listed["data"]] == ["qwen2.5-coder:1.5b", "all-minilm:latest"] and listed["object"] == "list"
    assert "size" not in json.dumps(listed)  # only names leave the machine
    client2, _ = _app(lambda r: httpx.Response(500, json={}))
    assert client2.get("/v1/models", headers=AUTH).status_code == 502


def test_routes_send_a_model_to_its_own_server_list_it_only_while_up_and_allow_it_implicitly() -> None:
    hosts: list[str] = []
    bee_up = {"v": True}

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "bee.local" and not bee_up["v"]:
            raise httpx.ConnectError("down")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "coder"}]})
        return _stream(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    c, _ = _app(handler, allowed_models=["coder"], routes={"bee-stars": "http://bee.local:8294/"})
    assert [m["id"] for m in c.get("/v1/models", headers=AUTH).json()["data"]] == ["coder", "bee-stars"]
    assert c.post("/v1/chat/completions", headers=AUTH, json={**CHAT, "model": "bee-stars"}).status_code == 200  # allowed without --allow-model
    assert c.post("/v1/chat/completions", headers=AUTH, json={**CHAT, "model": "coder"}).status_code == 200
    assert hosts[-2:] == ["bee.local", "upstream.local"]  # each model reached ITS server
    assert c.post("/v1/chat/completions", headers=AUTH, json={**CHAT, "model": "other"}).status_code == 403
    bee_up["v"] = False
    assert [m["id"] for m in c.get("/v1/models", headers=AUTH).json()["data"]] == ["coder"]  # a down server's model is not advertised
