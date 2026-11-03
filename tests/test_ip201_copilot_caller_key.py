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


def test_strategy_optimizer_post_needs_a_worker_token_and_anonymous_posts_change_nothing(monkeypatch) -> None:
    """IP-198: an anonymous POST used to rewrite the process-wide strategy engine every viewer reads."""
    from fastapi.testclient import TestClient

    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    # the token is read from the environment so this control runs unchanged on a main that has no worker_token kwarg
    monkeypatch.setenv("HIVEMIND_STRATEGY_WORKER_TOKEN", "test-worker-token-not-a-secret")
    client = TestClient(create_hack_miner_fastapi_app(), raise_server_exceptions=False)
    before = client.get("/api/strategy-optimizer").json()["si03_adaptive_diloco_cadence"]
    anonymous = client.post("/api/strategy-optimizer", json={"ring_rtt_ms": 900.0, "bbr_pacing_mbps": 0.5, "packet_loss_rate": 0.9, "turn_relay_active": True})
    assert anonymous.status_code == 401, anonymous.status_code
    assert client.get("/api/strategy-optimizer").json()["si03_adaptive_diloco_cadence"] == before
    wrong = client.post("/api/strategy-optimizer", headers={"Authorization": "Bearer not-the-token"}, json={"ring_rtt_ms": 900.0})
    assert wrong.status_code == 401
    ok = client.post("/api/strategy-optimizer", headers={"Authorization": "Bearer test-worker-token-not-a-secret"}, json={"ring_rtt_ms": 145.0, "bbr_pacing_mbps": 18.0, "packet_loss_rate": 0.03, "turn_relay_active": False})
    assert ok.status_code == 200
    # no token configured on the server: the telemetry is read-only, fail closed
    monkeypatch.delenv("HIVEMIND_STRATEGY_WORKER_TOKEN", raising=False)
    closed = TestClient(create_hack_miner_fastapi_app(), raise_server_exceptions=False)
    assert closed.post("/api/strategy-optimizer", headers={"Authorization": "Bearer anything"}, json={"ring_rtt_ms": 1.0}).status_code == 401
    assert closed.get("/api/strategy-optimizer").status_code == 200
