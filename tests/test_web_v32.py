"""HiveMind web portal v3.2: security properties and honest content.

A fake gateway stands in for the real one (no network): it implements login, register, change-password, rotate-key,
password reset, capabilities, Google sign-in and linking with in-memory accounts.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.web_v32 import mount_web_v32

ORIGIN = "https://hivemind.example.org"
SECRET = "s" * 48


class FakeGateway:
    base_url = "https://gateway.example.org"

    def __init__(self) -> None:
        self.accounts = {"alice@example.org": {"handle": "alice", "password": "AlicePassword#2026", "key": "hm-alice-key-0000000000000000", "role": "prosumer"}}
        self.reset_email = True
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.telemetry: dict[str, Any] = {
            "accounts": {"total_registered_accounts": 78, "active_accounts_24h": 2},
            "producers": {"distinct_physical_gpus": 1},
            "models": {"live_models_count": 3},
            "t4_boinc": {"held_out_nll_current": 2.138},
        }

    def _by_key(self, key: str | None):
        return next((a for a in self.accounts.values() if a["key"] == key), None)

    def get(self, path: str):
        if path == "/v1/hivemind/live-telemetry":
            return (200, self.telemetry) if self.telemetry is not None else (502, {})
        if path == "/api/v1/auth/capabilities":
            return 200, {"password_reset_email": self.reset_email}
        return 404, {}

    def post(self, path: str, payload: dict[str, Any], bearer: str | None = None, client_ip: str | None = None):
        self.calls.append((path, dict(payload), bearer))
        if path == "/api/v1/auth/login":
            ident = payload["handle_or_email"]
            acct = self.accounts.get(ident) or next((a for a in self.accounts.values() if a["handle"] == ident), None)
            if acct and acct["password"] == payload["password"]:
                return 200, {"api_key": acct["key"], "handle": acct["handle"], "role": acct["role"], "created": False}
            return 401, {"error": {"message": "Account not found."}}
        if path == "/api/v1/auth/register":
            if payload["email"] in self.accounts:
                return 409, {}
            key = "hm-new-" + payload["handle"] + "-key-000000000000"
            self.accounts[payload["email"]] = {"handle": payload["handle"], "password": payload["password"], "key": key, "role": "prosumer"}
            return 200, {"api_key": key, "handle": payload["handle"], "role": "prosumer", "created": True}
        if path == "/api/v1/auth/change-password":
            acct = self._by_key(bearer)
            if acct is None:
                return 401, {}
            if len(payload["new_password"]) < 12:
                return 400, {}
            acct["password"] = payload["new_password"]
            return 200, {"status": "PASSWORD_SET"}
        if path == "/api/v1/auth/rotate-key":
            acct = self._by_key(bearer)
            if acct is None:
                return 200, {"old_token_revoked": False, "token": "hm_live_ephemeral"}
            acct["key"] = acct["key"] + "x"
            return 200, {"old_token_revoked": True, "api_key": acct["key"]}
        if path == "/api/v1/auth/password-reset/request":
            return (202, {"status": "RESET_REQUESTED"}) if self.reset_email else (503, {})
        if path == "/api/v1/auth/password-reset/confirm":
            return (200, {"status": "PASSWORD_RESET"}) if payload["token"] == "t" * 43 else (400, {"error": {"message": "This reset link is invalid or has expired."}})
        if path == "/v1/hivemind/register/google":
            return 200, {"api_key": "hm-google-key-0000000000000000", "role": "prosumer", "created": True}
        if path == "/api/v1/auth/link/google":
            return (200, {}) if self._by_key(bearer) else (401, {})
        return 404, {}


def _id_token(nonce: str) -> str:
    def enc(d: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{enc({'alg': 'RS256'})}.{enc({'nonce': nonce, 'sub': 'g-1'})}.sig"


class FakeVerifier:
    def __init__(self) -> None:
        self.fail = False

    def verify(self, token: str):
        if self.fail:
            raise ValueError("bad token")
        return object()


def _client(env_extra: dict[str, str] | None = None, *, gateway: FakeGateway | None = None, verifier: FakeVerifier | None = None,
            exchanged: dict[str, Any] | None = None):
    env = {"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN}
    env.update(env_extra or {})
    gw = gateway or FakeGateway()
    state: dict[str, Any] = {"nonce": ""}

    def exchanger(form):
        state["form"] = dict(form)
        return exchanged if exchanged is not None else {"id_token": _id_token(state["nonce"])}

    app = FastAPI()
    mount_web_v32(app, env=env, gateway=gw, google_verifier=verifier or FakeVerifier(), token_exchanger=exchanger)
    client = TestClient(app, base_url=ORIGIN)
    return client, gw, state


def _csrf(client: TestClient) -> dict[str, str]:
    client.get("/v32")
    return {"X-CSRF-Token": client.cookies.get("hm_csrf"), "Origin": ORIGIN}


# ---- headers and page --------------------------------------------------------------------------------------------------

def test_page_has_a_nonce_csp_without_unsafe_inline_and_no_inline_script() -> None:
    client, _, _ = _client()
    r = client.get("/v32")
    csp = r.headers["content-security-policy"]
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "frame-ancestors 'none'" in csp
    scripts = re.findall(r"<script[^>]*>", r.text)
    assert scripts and all(f'nonce="{nonce}"' in s and "src=" in s for s in scripts)
    bodies = re.findall(r"<script[^>]*>(.*?)</script>", r.text, flags=re.S)
    assert bodies and all(not b.strip() for b in bodies), "no inline script body"
    assert " style=" not in r.text and "<style" not in r.text


def test_dashboard_middleware_keeps_the_strict_policy_on_v32_pages() -> None:
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    client = TestClient(create_hivemind_at_home_app())
    r = client.get("/v32")
    assert r.status_code == 200 and "unsafe-inline" not in r.headers["content-security-policy"]


def test_page_shows_no_demo_or_invented_content() -> None:
    client, _, _ = _client()
    text = client.get("/v32").text.lower()
    for banned in ("demo", "elena", "berkeley", "llama_king", "gamer_pro", "pflops", "zk-snark", "steam", "reddit", "github sign"):
        assert banned not in text, banned


def test_facts_come_with_sources_and_never_invent_numbers() -> None:
    client, gw, _ = _client()
    facts = client.get("/v32/api/facts").json()["facts"]
    assert {f["label"] for f in facts} >= {"Registered accounts", "Models served now"}
    assert all(f["source"].startswith(gw.base_url) for f in facts)
    assert "2.138" not in json.dumps(facts), "unverified training telemetry is not shown"
    gw.telemetry = None
    down = client.get("/v32/api/facts").json()["facts"]
    assert all(f["available"] is False and f["value"] is None for f in down)


# ---- sessions, CSRF, login --------------------------------------------------------------------------------------------

def test_login_sets_a_hardened_session_cookie_and_never_returns_the_key() -> None:
    client, _, _ = _client()
    r = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=_csrf(client))
    assert r.status_code == 200 and "hm-" not in r.text
    cookie = r.headers["set-cookie"]
    assert "hm_s=" in cookie and "HttpOnly" in cookie and "Secure" in cookie and "samesite=lax" in cookie.lower()
    assert "hm-alice" not in cookie, "the key is encrypted inside the cookie"
    s = client.get("/v32/api/session").json()
    assert s == {"authenticated": True, "handle": "alice", "role": "prosumer"}


def test_wrong_password_and_unknown_account_get_the_same_answer() -> None:
    client, _, _ = _client()
    h = _csrf(client)
    a = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "nope-nope-nope"}, headers=h)
    b = client.post("/v32/api/login", json={"identifier": "ghost@example.org", "password": "nope-nope-nope"}, headers=h)
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


@pytest.mark.parametrize("headers", [{}, {"X-CSRF-Token": "wrong", "Origin": ORIGIN}, "bad-origin"])
def test_state_changing_requests_need_csrf_token_and_same_origin(headers) -> None:
    client, gw, _ = _client()
    good = _csrf(client)
    if headers == "bad-origin":
        headers = {"X-CSRF-Token": good["X-CSRF-Token"], "Origin": "https://evil.example.com"}
    r = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=headers)
    assert r.status_code == 403 and gw.calls == []


def test_without_a_session_secret_sign_in_fails_closed() -> None:
    client, gw, _ = _client({"HIVEMIND_WEB_SESSION_SECRET": ""})
    r = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=_csrf(client))
    assert r.status_code == 503 and gw.calls == []
    assert client.get("/v32/api/capabilities").json()["password_login"] is False


def test_register_shows_the_key_once_and_logout_ends_the_session() -> None:
    client, _, _ = _client()
    h = _csrf(client)
    weak = client.post("/v32/api/register", json={"email": "bob@example.org", "password": "short"}, headers=h)
    assert weak.status_code == 400
    r = client.post("/v32/api/register", json={"email": "bob@example.org", "handle": "bob", "password": "BobPassword#2026"}, headers=h)
    assert r.status_code == 200 and r.json()["api_key_once"].startswith("hm-")
    assert client.get("/v32/api/session").json()["authenticated"] is True
    client.post("/v32/api/logout", json={}, headers=h)
    assert client.get("/v32/api/session").json()["authenticated"] is False


def test_change_password_and_rotate_key_use_the_session_key_server_side() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    assert client.post("/v32/api/change-password", json={"new_password": "AliceNewPassword#1"}, headers=h).status_code == 200
    assert gw.calls[-1][2] == "hm-alice-key-0000000000000000"
    rot = client.post("/v32/api/rotate-key", json={}, headers=h)
    assert rot.status_code == 200 and rot.json()["api_key_once"].endswith("x")
    assert client.post("/v32/api/change-password", json={"new_password": "AnotherNewPass#2"}, headers=h).status_code == 200
    assert gw.calls[-1][2].endswith("x"), "the session follows the rotated key"


def test_tampered_session_cookie_is_ignored() -> None:
    client, _, _ = _client()
    client.cookies.set("hm_s", "gAAAAAforged", domain="hivemind.example.org")
    assert client.get("/v32/api/session").json()["authenticated"] is False


# ---- password reset ----------------------------------------------------------------------------------------------------

def test_reset_request_is_generic_and_unavailable_is_said_plainly() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    r = client.post("/v32/api/password-reset/request", json={"email": "anyone@example.org"}, headers=h)
    assert r.status_code == 202 and "If an account uses this e-mail" in r.json()["message"]
    gw.reset_email = False
    off = client.post("/v32/api/password-reset/request", json={"email": "anyone@example.org"}, headers=h)
    assert off.status_code == 503 and "not available" in off.json()["error"]
    assert client.get("/v32/api/capabilities").json()["password_reset_email"] is False


def test_reset_confirm_forwards_the_token_and_reports_failures() -> None:
    client, _, _ = _client()
    h = _csrf(client)
    assert client.post("/v32/api/password-reset/confirm", json={"token": "t" * 43, "new_password": "BrandNewPassword#1"}, headers=h).status_code == 200
    bad = client.post("/v32/api/password-reset/confirm", json={"token": "u" * 43, "new_password": "BrandNewPassword#1"}, headers=h)
    assert bad.status_code == 400 and "expired" in bad.json()["error"]
    assert client.get("/auth/reset").status_code == 200, "the link in the e-mail opens the page"


# ---- Google ------------------------------------------------------------------------------------------------------------

GOOGLE_ENV = {"HIVEMIND_GOOGLE_CLIENT_ID": "cid.apps.googleusercontent.com", "HIVEMIND_GOOGLE_CLIENT_SECRET": "csecret"}


def test_google_is_hidden_and_closed_when_not_configured() -> None:
    client, _, _ = _client()
    assert client.get("/v32/api/capabilities").json()["google"] is False
    assert client.get("/auth/google/start", follow_redirects=False).status_code == 404


def _start(client, state_holder, link: bool = False):
    r = client.get("/auth/google/start" + ("?link=1" if link else ""), follow_redirects=False)
    assert r.status_code == 302
    q = parse_qs(urlsplit(r.headers["location"]).query)
    state_holder["nonce"] = q["nonce"][0]
    return r, q


def test_google_start_uses_pkce_state_and_nonce_and_keeps_secrets_server_side() -> None:
    client, _, holder = _client(GOOGLE_ENV)
    r, q = _start(client, holder)
    assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert q["code_challenge_method"] == ["S256"] and q["response_type"] == ["code"] and q["client_id"] == [GOOGLE_ENV["HIVEMIND_GOOGLE_CLIENT_ID"]]
    assert q["redirect_uri"] == [f"{ORIGIN}/auth/google/callback"]
    assert "code_verifier" not in r.headers["location"] and "csecret" not in r.headers["location"]
    cookie = r.headers["set-cookie"]
    assert "hm_oidc=" in cookie and "HttpOnly" in cookie and "Path=/auth/google" in cookie
    assert q["state"][0] not in cookie and q["nonce"][0] not in cookie, "state and nonce are sealed in the cookie"


def test_google_callback_signs_in_after_verifying_state_token_and_nonce() -> None:
    client, gw, holder = _client(GOOGLE_ENV)
    _r, q = _start(client, holder)
    cb = client.get(f"/auth/google/callback?state={q['state'][0]}&code=abc", follow_redirects=False)
    assert cb.status_code == 302 and cb.headers["location"].endswith("#account")
    assert holder["form"]["code_verifier"] and holder["form"]["client_secret"] == "csecret"
    assert gw.calls[-1][0] == "/v1/hivemind/register/google"
    assert client.get("/v32/api/session").json()["authenticated"] is True


def test_google_callback_rejects_wrong_state() -> None:
    client, gw, holder = _client(GOOGLE_ENV)
    _start(client, holder)
    cb = client.get("/auth/google/callback?state=forged&code=abc", follow_redirects=False)
    assert "signin-error=state" in cb.headers["location"] and gw.calls == []


def test_google_callback_rejects_a_token_with_the_wrong_nonce() -> None:
    client, gw, holder = _client(GOOGLE_ENV, exchanged={"id_token": _id_token("someone-elses-nonce")})
    _r, q = _start(client, holder)
    cb = client.get(f"/auth/google/callback?state={q['state'][0]}&code=abc", follow_redirects=False)
    assert "signin-error=nonce" in cb.headers["location"] and gw.calls == []


def test_google_callback_rejects_an_unverified_token() -> None:
    verifier = FakeVerifier()
    verifier.fail = True
    client, gw, holder = _client(GOOGLE_ENV, verifier=verifier)
    _r, q = _start(client, holder)
    cb = client.get(f"/auth/google/callback?state={q['state'][0]}&code=abc", follow_redirects=False)
    assert "signin-error=token" in cb.headers["location"] and gw.calls == []


def test_google_link_needs_a_session_and_uses_its_key() -> None:
    client, gw, holder = _client(GOOGLE_ENV)
    h = _csrf(client)
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    _r, q = _start(client, holder, link=True)
    cb = client.get(f"/auth/google/callback?state={q['state'][0]}&code=abc", follow_redirects=False)
    assert cb.headers["location"].endswith("#linked")
    assert gw.calls[-1][0] == "/api/v1/auth/link/google" and gw.calls[-1][2] == "hm-alice-key-0000000000000000"


# ---- legacy sandbox SSO is retired ---------------------------------------------------------------------------------

def test_legacy_sandbox_sso_endpoints_are_retired(monkeypatch) -> None:
    monkeypatch.delenv("HIVEMIND_LEGACY_SANDBOX_SSO", raising=False)
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    client = TestClient(create_hivemind_at_home_app())
    a = client.get("/api/v1/auth/sso/google/authorize")
    b = client.post("/api/v1/auth/sso/google", json={})
    assert a.status_code == 410 and "code_verifier" not in a.text
    assert b.status_code in (410, 422)


def test_root_is_taken_over_only_with_the_flag() -> None:
    client, _, _ = _client()
    assert client.get("/").status_code == 404
    on, _, _ = _client({"HIVEMIND_WEB_V32": "1"})
    assert on.get("/").headers.get("x-hivemind-ui") == "v3.2"
    assert on.get("/auth").headers.get("x-hivemind-ui") == "v3.2"
