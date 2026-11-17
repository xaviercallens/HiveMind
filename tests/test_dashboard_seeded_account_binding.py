# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A seeded dashboard account (owner, MasterMinders) is opened only by its token, its own password, or a password the
gateway verifies for the SAME handle. A stranger can neither choose its password on first sign-in nor borrow a gateway
account registered under one of its (unverified) e-mail addresses."""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

SEEDED = ["xavier_callens", "callensxavier@gmail.com", "vxtzq", "clark719664"]


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_gateway(monkeypatch: pytest.MonkeyPatch, answer_handle: str | None, seen: list) -> None:
    """Stand-in for the gateway's /api/v1/auth/login: says ok for any password, as the account ``answer_handle``."""
    monkeypatch.setenv("HIVEMIND_OPENROUTER_URL", "http://gateway.invalid")

    def urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        seen.append(body.get("handle_or_email"))
        if answer_handle is None:
            raise OSError("gateway down")
        return _Resp(json.dumps({"ok": True, "handle": answer_handle, "balance": 1.0}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize("ident", SEEDED)
def test_a_stranger_cannot_choose_a_seeded_accounts_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ident: str) -> None:
    _fake_gateway(monkeypatch, None, [])  # gateway unreachable: no one vouches for the password
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r.status_code == 401 and "token" not in r.json()
    # and the attempt did not bind the password: the same password still fails
    r2 = client.post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r2.status_code in (401, 429) and "token" not in r2.json()


@pytest.mark.parametrize("ident", ["vxtzq", "xavier_callens"])
def test_a_gateway_account_with_another_handle_does_not_open_a_seeded_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ident: str) -> None:
    seen: list = []
    _fake_gateway(monkeypatch, "attacker_account", seen)  # the gateway vouches, but for a different account
    r = _client(tmp_path).post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r.status_code == 401 and "token" not in r.json()
    assert all("@" not in str(s) for s in seen), f"the gateway was asked by e-mail: {seen}"


def test_the_gateway_password_of_the_same_handle_still_opens_the_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    _fake_gateway(monkeypatch, "vxtzq", seen)
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/login", json={"handle_or_email": "vxtzq", "password": "the-real-gateway-pw"})
    assert r.status_code == 200 and r.json()["token"].startswith("hm_live_")
    assert seen == ["vxtzq"]


def test_jblam62122_gateway_canonical_handle_opens_seeded_jblam_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    _fake_gateway(monkeypatch, "jblam62122", seen)
    client = _client(tmp_path)
    r = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "jblam62122", "password": "Jblam-WebUI-Password!"},
    )
    assert r.status_code == 200 and r.json()["token"].startswith("hm_live_")
    assert r.json()["handle"] == "@jblam-O"
    assert seen == ["jblam_o"]


def test_auto_model_prefers_loaded_resident_model_over_unloaded_cold_model() -> None:
    import httpx

    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(":memory:", b"x" * 32)
    prod_id, prod_key = store.register("rtx2070_prod", "prosumer")
    cons_id, cons_key = store.register("vxtzq_cons", "prosumer")
    store.grant(cons_id, 100.0, "test-grant")

    store.upsert_producer(
        prod_id,
        endpoint_url="https://93.184.216.34/rtx2070",
        models=[{"id": "coder-lite"}, {"id": "crowd-v1"}, {"id": "granite-moe"}],
        cost_per_1k=0.35,
        capacity_1k=60.0,
        execution_class="rtx2070-8gb-turing",
        now_utc=1000.0,
        residency={
            "models": {
                "coder-lite": {"loaded": False, "cold_start_s": 60.0},
                "crowd-v1": {"loaded": True, "cold_start_s": 0.0},
                "granite-moe": {"loaded": False, "cold_start_s": 60.0},
            }
        },
    )

    routed_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        routed_models.append(str(payload.get("model")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    transport = httpx.MockTransport(handler)
    app = create_router_app(
        store,
        http_client=httpx.Client(transport=transport),
        clock=lambda: 1010.0,
    )
    gw_client = TestClient(app)

    r_reset = gw_client.get("/auth/reset")
    assert r_reset.status_code == 200
    assert "passwordResetCard" in r_reset.text

    r_chat = gw_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16},
    )
    assert r_chat.status_code == 200, r_chat.text
    assert routed_models == ["crowd-v1"]

