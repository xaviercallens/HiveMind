"""IP-201: /api/scientific-game/la-ruche-copilot forwards only the caller's own router key.

Anonymous callers get 401 and nothing is sent upstream; with a caller key, the La Ruche request
carries that key and never the server-held cached session key.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import hivemind.game.scientific_discovery_game as sdg
from hivemind.game.steam_edition import create_hack_miner_fastapi_app

CALLER_KEY = "caller-router-key-test"
SERVER_KEY = "server-held-session-key-test"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_hack_miner_fastapi_app(), raise_server_exceptions=False)


def _install_stub_router(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Route every httpx.Client made by the copilot module to an in-memory stub router."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/v1/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "stub reply"}}],
                    "hivemind": {"charged_hmt": 0.0},
                },
            )
        return httpx.Response(200, json={"chain_verified": True})

    real_client = httpx.Client

    def make_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sdg.httpx, "Client", make_client)
    return seen


def _forbid_server_key_load(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_loader(**kwargs: object) -> dict[str, str]:
        calls.append("loaded")
        return {"api_key": SERVER_KEY, "handle": "server", "account_id": "server"}

    monkeypatch.setattr(sdg, "_load_or_register_la_ruche_gamer_session", fake_loader)
    return calls


def test_anonymous_copilot_post_is_401_and_reaches_no_router(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _install_stub_router(monkeypatch)
    server_loads = _forbid_server_key_load(monkeypatch)

    response = client.post("/api/scientific-game/la-ruche-copilot", json={"prompt": "hello"})

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert seen == []
    assert server_loads == []


def test_copilot_forwards_only_the_callers_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stub_router(monkeypatch)
    server_loads = _forbid_server_key_load(monkeypatch)

    response = client.post(
        "/api/scientific-game/la-ruche-copilot",
        json={"prompt": "hello"},
        headers={"Authorization": f"Bearer {CALLER_KEY}"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "stub reply"
    auth_headers = [request.headers.get("authorization") for request in seen]
    assert auth_headers, "the stub router must have received the caller's request"
    assert all(header == f"Bearer {CALLER_KEY}" for header in auth_headers)
    assert server_loads == []


def test_copilot_rejects_malformed_authorization_without_upstream_call(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _install_stub_router(monkeypatch)

    response = client.post(
        "/api/scientific-game/la-ruche-copilot",
        json={"prompt": "hello"},
        headers={"Authorization": f"Basic {CALLER_KEY}"},
    )

    assert response.status_code == 401
    assert seen == []


def test_ask_function_refuses_empty_router_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stub_router(monkeypatch)
    server_loads = _forbid_server_key_load(monkeypatch)

    with pytest.raises(ValueError, match="router_api_key is required"):
        sdg.ask_la_ruche_bee_copilot_live(prompt="hello", router_api_key="   ")
    assert seen == []
    assert server_loads == []
