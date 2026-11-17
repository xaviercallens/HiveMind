"""v3.2 end to end: the web portal wired to the REAL gateway (in process), plus the signed client-IP forwarding.

This is the test that catches what a fake gateway cannot: field names, the shared rate-limit bucket, and previous users
whose e-mail is only in the legacy credentials table.
"""
from __future__ import annotations

import hashlib
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.router import forwarded_client
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.web_v32 import GatewayClient, mount_web_v32

SECRET = b"0123456789abcdef0123456789abcdef"
ORIGIN = "https://hivemind.example.org"


class _Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t


def _stack(tmp_path):
    clock = _Clock()
    store = RouterStore(tmp_path / "gw.sqlite3", SECRET)
    sent: list[tuple[str, str]] = []
    gateway_app = create_router_app(
        store, clock=clock, starter_grant_hmt=0.0,
        reset_email_sender=lambda to, token: sent.append((to, token)) or True,
        reset_email_enabled=lambda: True,
    )
    gw_http = TestClient(gateway_app, base_url="http://gateway")
    gw = GatewayClient("http://gateway", client=gw_http, forward_secret=SECRET, clock=clock)
    web = FastAPI()
    mount_web_v32(web, env={"HIVEMIND_WEB_SESSION_SECRET": "w" * 48, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN}, gateway=gw, clock=clock)
    return store, TestClient(web, base_url=ORIGIN), sent, clock


def _csrf(client: TestClient) -> dict[str, str]:
    client.get("/v32")
    return {"X-CSRF-Token": client.cookies.get("hm_csrf"), "Origin": ORIGIN}


def test_register_login_change_reset_and_login_again_against_the_real_gateway(tmp_path) -> None:
    store, web, sent, _ = _stack(tmp_path)
    h = _csrf(web)
    reg = web.post("/v32/api/register", json={"email": "carol@example.org", "handle": "carol", "password": "CarolPassword#2026"}, headers=h)
    assert reg.status_code == 200 and reg.json()["api_key_once"].startswith("hm-")
    web.post("/v32/api/logout", json={}, headers=h)
    login = web.post("/v32/api/login", json={"identifier": "carol@example.org", "password": "CarolPassword#2026"}, headers=h)
    assert login.status_code == 200 and login.json()["handle"] == "carol"
    assert web.post("/v32/api/change-password", json={"new_password": "CarolChanged#2026"}, headers=h).status_code == 200
    web.post("/v32/api/logout", json={}, headers=h)
    assert web.post("/v32/api/login", json={"identifier": "carol", "password": "CarolPassword#2026"}, headers=h).status_code == 401
    assert web.post("/v32/api/password-reset/request", json={"email": "carol@example.org"}, headers=h).status_code == 202
    token = sent[-1][1]
    ok = web.post("/v32/api/password-reset/confirm", json={"token": token, "new_password": "CarolAfterReset#1"}, headers=h)
    assert ok.status_code == 200
    assert web.post("/v32/api/login", json={"identifier": "carol@example.org", "password": "CarolAfterReset#1"}, headers=h).status_code == 200


def test_previous_user_with_email_only_in_legacy_credentials_can_reset(tmp_path) -> None:
    store, web, sent, _ = _stack(tmp_path)
    email = "pioneer_oct25@gmail.com"
    ehash = store._hmac_hex(f"email:{email}")
    acc_id, _ = store.register(f"user.{ehash[:16]}", "prosumer")
    salt = "db75c3761836010dfa4bf470418b689d"
    with sqlite3.connect(tmp_path / "gw.sqlite3") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS credentials (email_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, salt_hex TEXT NOT NULL, password_hash TEXT NOT NULL)")
        conn.execute("INSERT INTO credentials VALUES (?,?,?,?)", (ehash, acc_id, salt, hashlib.pbkdf2_hmac("sha256", b"Oct25Old#Pass", bytes.fromhex(salt), 120_000).hex()))
        conn.commit()
    h = _csrf(web)
    assert web.post("/v32/api/login", json={"identifier": email, "password": "Oct25Old#Pass"}, headers=h).status_code == 200
    web.post("/v32/api/logout", json={}, headers=h)
    web.post("/v32/api/password-reset/request", json={"email": email}, headers=h)
    assert sent and sent[-1][0] == email
    assert web.post("/v32/api/password-reset/confirm", json={"token": sent[-1][1], "new_password": "Oct25NewPass#2026"}, headers=h).status_code == 200
    r = web.post("/v32/api/login", json={"identifier": email, "password": "Oct25NewPass#2026"}, headers=h)
    assert r.status_code == 200


def test_each_visitor_gets_their_own_rate_limit_bucket_at_the_gateway(tmp_path) -> None:
    _store, web, _sent, _ = _stack(tmp_path)
    h = _csrf(web)
    for _ in range(30):
        web.post("/v32/api/login", json={"identifier": "nobody@example.org", "password": "wrong-password-x"}, headers={**h, "X-Forwarded-For": "203.0.113.7"})
    blocked = web.post("/v32/api/login", json={"identifier": "nobody@example.org", "password": "wrong-password-x"}, headers={**h, "X-Forwarded-For": "203.0.113.7"})
    other = web.post("/v32/api/login", json={"identifier": "nobody@example.org", "password": "wrong-password-x"}, headers={**h, "X-Forwarded-For": "198.51.100.9"})
    assert blocked.status_code == 429, "the attacker's own bucket fills up"
    assert other.status_code == 401, "another visitor is not locked out"


def test_forwarded_client_header_is_ignored_when_forged_or_stale() -> None:
    now = 1_800_000_000.0
    good = forwarded_client.sign(SECRET, "203.0.113.7", now)
    assert forwarded_client.verify(SECRET, good, now) == "203.0.113.7"
    assert forwarded_client.verify(b"x" * 32, good, now) is None
    assert forwarded_client.verify(SECRET, good, now + 120) is None
    assert forwarded_client.verify(SECRET, good.replace("203.0.113.7", "198.51.100.9"), now) is None
    assert forwarded_client.verify(SECRET, "not-an-ip|1|00", now) is None


def test_non_canonical_host_is_redirected_and_posts_are_refused(tmp_path) -> None:
    _store, _web, _sent, _ = _stack(tmp_path)
    other = TestClient(_web.app, base_url="https://other-host.example.net")
    r = other.get("/v32?x=1", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == f"{ORIGIN}/v32?x=1"
    assert other.post("/v32/api/login", json={}).status_code == 421
