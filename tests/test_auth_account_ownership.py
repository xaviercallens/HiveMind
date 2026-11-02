# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Account ownership: an account's key is only ever returned to a caller who proves ownership (GHSA-gjc6-74f9-wmrr)."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from hivemind.router import accounts as accounts_mod
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

SECRET = b"hivemind-test-secret-at-least-32-bytes-long!"


def _client(store: RouterStore) -> TestClient:
    return TestClient(create_router_app(store, starter_grant_hmt=0.0))


def test_register_never_returns_an_existing_account_without_its_password() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    first = client.post("/v1/hivemind/auth/register", json={"handle": "victim", "password": "correct horse battery"})
    assert first.status_code == 200
    victim_id = first.json()["account_id"]

    for body in ({"handle": "victim"}, {"handle": "victim", "password": "guess"}):
        r = client.post("/v1/hivemind/auth/register", json=body)
        assert r.status_code == 409, body
        assert "api_key" not in r.text

    ok = client.post("/v1/hivemind/auth/register", json={"handle": "victim", "password": "correct horse battery"})
    assert ok.status_code == 200
    assert ok.json()["account_id"] == victim_id


def test_register_and_login_never_bind_a_password_to_a_passwordless_account() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    _aid, key = store.register("handle-only", "prosumer")  # like a handle-only or Google/OIDC account

    r = client.post("/v1/hivemind/auth/register", json={"handle": "handle-only", "password": "attacker-chosen"})
    assert r.status_code == 409
    for _ in range(2):  # a second try must still fail: nothing was bound by the first
        r = client.post("/v1/hivemind/auth/login", json={"identifier": "handle-only", "password": "attacker-chosen"})
        assert r.status_code == 401
        assert "api_key" not in r.text
    r = client.post("/v1/hivemind/auth/login", json={"identifier": "handle-only"})
    assert r.status_code == 401

    # the owner still gets in with the account's own key, and that key keeps working
    r = client.post("/v1/hivemind/auth/login", json={"identifier": key})
    assert r.status_code == 200
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"}).status_code == 200


def test_login_does_not_create_accounts_or_grant_credit() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    for ident in ("brand-new-handle", "vxtzq", "@xaviercallens"):
        r = client.post("/v1/hivemind/auth/login", json={"identifier": ident, "password": "anything"})
        assert r.status_code == 401, ident
        assert store.get_account(store.account_id_for(ident.lstrip("@"))) is None


def test_keyless_chat_request_is_not_authenticated_whatever_the_user_agent() -> None:
    client = _client(RouterStore(":memory:", SECRET))
    r = client.post(
        "/v1/chat/completions",
        headers={"User-Agent": "Go-http-client/1.1"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_a_burned_key_is_refused_even_when_a_stored_row_still_matches_it(monkeypatch: pytest.MonkeyPatch) -> None:
    store = RouterStore(":memory:", SECRET)
    _aid, key = store.register("owner", "prosumer")
    assert store.authenticate(key) is not None
    monkeypatch.setattr(accounts_mod, "BURNED_KEY_SHA256", frozenset({hashlib.sha256(key.encode()).hexdigest()}))
    assert store.authenticate(key) is None
    assert _client(store).get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"}).status_code == 401


def test_no_source_file_carries_a_burned_key() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    token = re.compile(r"hm-[A-Za-z0-9_-]{20,}")
    for path in list((root / "src").rglob("*.py")) + list((root / "artifacts").glob("*.json")):
        for tok in set(token.findall(path.read_text(encoding="utf-8", errors="ignore"))):
            assert hashlib.sha256(tok.encode()).hexdigest() not in accounts_mod.BURNED_KEY_SHA256, path


def test_contribution_routes_need_a_key_and_mint_nothing() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    for n in range(3):
        r = client.post("/v1/hivemind/steam/contribute-boinc", json={"handle": "minter", "receipt_sha256": str(n)})
        assert r.status_code == 401
    assert store.get_account(store.account_id_for("minter")) is None

    aid, key = store.register("worker", "prosumer")
    before = store.balance(aid)
    for n in range(3):
        r = client.post(
            "/v1/hivemind/steam/contribute-boinc", headers={"Authorization": f"Bearer {key}"}, json={"receipt_sha256": f"r{n}"}
        )
        assert r.status_code == 200
        assert r.json()["hmt_awarded"] == 0.0
    assert store.balance(aid) == before


def test_steam_and_webgpu_sign_in_need_the_accounts_key_for_an_existing_handle() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    for route in ("/v1/hivemind/steam/register-login", "/v1/hivemind/webgpu/register-login"):
        handle = "player-" + route.split("/")[3]
        first = client.post(route, json={"handle": handle})
        assert first.status_code == 200
        again = client.post(route, json={"handle": handle})  # same client address, no key
        assert again.status_code == 409
        assert "api_key" not in again.text


def test_password_login_is_rate_limited() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _client(store)
    client.post("/v1/hivemind/auth/register", json={"handle": "target", "password": "right password"})
    codes = [client.post("/v1/hivemind/auth/login", json={"identifier": "target", "password": f"wrong{i}"}).status_code for i in range(12)]
    assert 429 in codes
    assert codes[0] == 401


def test_every_gateway_response_carries_the_proof_of_concept_header() -> None:
    client = _client(RouterStore(":memory:", SECRET))
    for resp in (
        client.get("/health"),
        client.get("/v1/models"),
        client.get("/definitely-not-a-route"),  # 404
        client.post("/v1/chat/completions", json={"model": "hivemind/auto", "messages": []}),  # 401
    ):
        assert "proof-of-concept" in resp.headers.get("x-hivemind-status", ""), resp.status_code
        assert "public" in resp.headers["x-hivemind-status"]
    big = client.post("/v1/chat/completions", content=b"x" * (5 * 1024 * 1024))  # refused by the body limit before routing
    assert big.status_code == 413
    assert "proof-of-concept" in big.headers.get("x-hivemind-status", "")
    assert "Proof of concept" in client.get("/openapi.json").json()["info"]["description"]


def test_an_unexpected_error_is_a_generic_500_with_the_notice_and_no_internals() -> None:
    app = create_router_app(RouterStore(":memory:", SECRET), starter_grant_hmt=0.0)

    @app.get("/boom-for-test")
    def boom() -> None:
        raise RuntimeError("secret internal detail /home/user/x.py")

    r = TestClient(app, raise_server_exceptions=False).get("/boom-for-test")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text and "Traceback" not in r.text and "/home/" not in r.text
    assert "proof-of-concept" in r.headers.get("x-hivemind-status", "")


@pytest.mark.parametrize("path", ["/signup", "/join"])
def test_public_pages_carry_the_proof_of_concept_notice(path: str) -> None:
    r = _client(RouterStore(":memory:", SECRET)).get(path)
    assert r.status_code == 200
    assert 'id="poc-notice"' in r.text and "Proof of concept" in r.text and "personal" in r.text


def test_admin_can_issue_a_fresh_key_for_an_account_and_the_old_one_dies() -> None:
    store = RouterStore(":memory:", SECRET)
    admin_key = "admin-secret-key-for-hivemaster"
    client = TestClient(create_router_app(store, admin_key_hash=store.key_digest(admin_key), starter_grant_hmt=0.0))
    aid, old = store.register("owner-account", "prosumer")
    assert client.post("/v1/hivemind/admin/accounts/rotate-key", json={"handle": "owner-account"}).status_code in (401, 403)
    r = client.post("/v1/hivemind/admin/accounts/rotate-key", headers={"x-admin-key": admin_key}, json={"handle": "owner-account"})
    assert r.status_code == 200
    new = r.json()["api_key"]
    assert r.json()["account_id"] == aid and new != old
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {new}"}).status_code == 200
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {old}"}).status_code == 401
    missing = client.post("/v1/hivemind/admin/accounts/rotate-key", headers={"x-admin-key": admin_key}, json={"handle": "nobody"})
    assert missing.status_code == 404
