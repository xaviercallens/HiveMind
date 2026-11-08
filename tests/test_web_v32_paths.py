"""v3.2 failure paths: static assets, gateway outages and garbage, session expiry, Google-flow failures, reset mailer."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.router import password_reset_email as mailer
from hivemind.web_v32 import GatewayClient, mount_web_v32
from hivemind.web_v32.app import SESSION_TTL_S, _jwt_claims_unverified
from tests.test_web_v32 import GOOGLE_ENV, ORIGIN, SECRET, FakeGateway, FakeVerifier, _client, _csrf, _id_token, _start


class _Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t


def test_static_assets_are_served_and_nothing_else_is() -> None:
    client, _, _ = _client()
    js = client.get("/v32/static/app.js")
    css = client.get("/v32/static/app.css")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert js.headers["x-content-type-options"] == "nosniff"
    for name in ("index.html", "..%2Fapp.py", "secrets.json"):
        assert client.get(f"/v32/static/{name}").status_code == 404


def test_gateway_client_turns_outages_and_garbage_into_safe_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/down":
            raise httpx.ConnectError("down", request=request)
        if request.url.path == "/text":
            return httpx.Response(200, text="not json")
        if request.url.path == "/list":
            return httpx.Response(200, json=[1, 2])
        return httpx.Response(200, json={"ok": True, "headers": dict(request.headers)})

    gw = GatewayClient("http://gw", client=httpx.Client(transport=httpx.MockTransport(handler)), forward_secret=b"k" * 32, clock=lambda: 1.0)
    for verb in (gw.get, lambda p: gw.post(p, {})):
        assert verb("/down") == (502, {})
        assert verb("/text") == (200, {})
        assert verb("/list") == (200, {})
    status, body = gw.post("/ok", {"a": 1}, bearer="hm-x", client_ip="203.0.113.7")
    assert status == 200 and body["headers"]["authorization"] == "Bearer hm-x"
    assert body["headers"]["x-hivemind-forwarded-client"].startswith("203.0.113.7|1|")


def test_unverified_claims_helper_never_raises() -> None:
    assert _jwt_claims_unverified("garbage") == {}
    assert _jwt_claims_unverified("a.!!!.c") == {}
    assert _jwt_claims_unverified(_id_token("n1"))["nonce"] == "n1"


def test_session_expires_after_its_lifetime() -> None:
    clock = _Clock()
    gw = FakeGateway()
    app = FastAPI()
    mount_web_v32(app, env={"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN}, gateway=gw, clock=clock)
    client = TestClient(app, base_url=ORIGIN)
    h = _csrf(client)
    assert client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h).status_code == 200
    clock.t += SESSION_TTL_S - 60
    assert client.get("/v32/api/session").json()["authenticated"] is True
    clock.t += 120
    assert client.get("/v32/api/session").json()["authenticated"] is False


def test_csrf_accepts_a_same_origin_referer_when_origin_is_absent_and_refuses_a_foreign_one() -> None:
    client, _, _ = _client()
    token = _csrf(client)["X-CSRF-Token"]
    ok = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"},
                     headers={"X-CSRF-Token": token, "Referer": f"{ORIGIN}/v32"})
    bad = client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"},
                      headers={"X-CSRF-Token": token, "Referer": "https://evil.example.com/x"})
    assert ok.status_code == 200 and bad.status_code == 403


def test_origin_falls_back_to_the_request_when_no_public_url_is_set() -> None:
    gw = FakeGateway()
    app = FastAPI()
    mount_web_v32(app, env={"HIVEMIND_WEB_SESSION_SECRET": SECRET}, gateway=gw)
    client = TestClient(app, base_url="https://local.example.org")
    client.get("/v32")
    h = {"X-CSRF-Token": client.cookies.get("hm_csrf"), "Origin": "https://local.example.org"}
    assert client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h).status_code == 200


def test_malformed_bodies_and_empty_fields_are_refused_cleanly() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    assert client.post("/v32/api/login", content=b"not json", headers={**h, "Content-Type": "application/json"}).status_code == 401
    assert client.post("/v32/api/login", json=[1, 2], headers=h).status_code == 401
    assert client.post("/v32/api/login", json={"identifier": "", "password": ""}, headers=h).status_code == 401
    assert gw.calls == []


class _Scripted(FakeGateway):
    """A gateway that answers chosen status codes for chosen paths."""

    def __init__(self, answers: dict[str, tuple[int, dict]]) -> None:
        super().__init__()
        self.answers = answers

    def post(self, path, payload, bearer=None, client_ip=None):
        if path in self.answers:
            self.calls.append((path, dict(payload), bearer))
            return self.answers[path]
        return super().post(path, payload, bearer, client_ip)


def test_gateway_errors_map_to_clear_answers() -> None:
    gw = _Scripted({"/api/v1/auth/login": (503, {}), "/api/v1/auth/register": (429, {}), "/api/v1/auth/password-reset/request": (429, {})})
    client, _, _ = _client(gateway=gw)
    h = _csrf(client)
    assert client.post("/v32/api/login", json={"identifier": "a@example.org", "password": "x" * 12}, headers=h).status_code == 503
    assert client.post("/v32/api/register", json={"email": "z@example.org", "password": "ZPassword#2026"}, headers=h).status_code == 429
    assert client.post("/v32/api/password-reset/request", json={"email": "z@example.org"}, headers=h).status_code == 429
    gw2 = _Scripted({"/api/v1/auth/login": (429, {})})
    c2, _, _ = _client(gateway=gw2)
    assert c2.post("/v32/api/login", json={"identifier": "a@example.org", "password": "x" * 12}, headers=_csrf(c2)).status_code == 429


def test_register_conflict_and_restored_account_do_not_show_a_key() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    assert client.post("/v32/api/register", json={"email": "alice@example.org", "password": "AlicePassword#2026"}, headers=h).status_code == 409
    gw2 = _Scripted({"/api/v1/auth/register": (200, {"api_key": "hm-restored-key-000000000000", "handle": "old", "role": "prosumer", "created": False})})
    c2, _, _ = _client(gateway=gw2)
    r = c2.post("/v32/api/register", json={"email": "old@example.org", "password": "OldPassword#2026"}, headers=_csrf(c2))
    assert r.status_code == 200 and "api_key_once" not in r.json() and "hm-" not in r.text


def test_account_actions_need_a_session_and_report_failures() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    assert client.post("/v32/api/change-password", json={"new_password": "x" * 12}, headers=h).status_code == 401
    assert client.post("/v32/api/rotate-key", json={}, headers=h).status_code == 401
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    assert client.post("/v32/api/change-password", json={"new_password": "short"}, headers=h).status_code == 400
    gw.accounts["alice@example.org"]["key"] = "hm-revoked-elsewhere-0000000"
    assert client.post("/v32/api/change-password", json={"new_password": "LongEnough#2026"}, headers=h).status_code == 401
    assert client.post("/v32/api/rotate-key", json={}, headers=h).status_code == 401


@pytest.mark.parametrize("path,body", [
    ("/v32/api/logout", {}), ("/v32/api/change-password", {"new_password": "x" * 12}), ("/v32/api/rotate-key", {}),
    ("/v32/api/password-reset/request", {"email": "a@example.org"}),
    ("/v32/api/password-reset/confirm", {"token": "t" * 43, "new_password": "x" * 12}),
    ("/v32/api/register", {"email": "a@example.org", "password": "x" * 12}),
])
def test_every_state_changing_route_refuses_without_csrf(path, body) -> None:
    client, gw, _ = _client()
    client.get("/v32")
    assert client.post(path, json=body, headers={"Origin": ORIGIN}).status_code == 403
    assert gw.calls == []


def test_google_flow_failures_end_on_the_page_with_a_reason() -> None:
    client, gw, holder = _client(GOOGLE_ENV, exchanged={})
    _r, q = _start(client, holder)
    assert "signin-error=exchange" in client.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]
    assert "signin-error=expired" in client.get("/auth/google/callback?state=s&code=c", follow_redirects=False).headers["location"]
    _r, q = _start(client, holder)
    assert "signin-error=state" in client.get(f"/auth/google/callback?state={q['state'][0]}&error=access_denied", follow_redirects=False).headers["location"]

    gw2 = _Scripted({"/v1/hivemind/register/google": (401, {}), "/api/v1/auth/link/google": (409, {})})
    c2, _, h2 = _client(GOOGLE_ENV, gateway=gw2)
    _r, q = _start(c2, h2)
    assert "signin-error=gateway" in c2.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]
    c2.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=_csrf(c2))
    _r, q = _start(c2, h2, link=True)
    assert "signin-error=link" in c2.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]


def test_google_callback_is_closed_when_google_is_not_configured() -> None:
    client, _, _ = _client()
    assert client.get("/auth/google/callback?state=s&code=c", follow_redirects=False).status_code == 404


def test_google_oauth_cookie_expires_after_ten_minutes() -> None:
    clock = _Clock()
    gw = FakeGateway()
    holder: dict = {}
    app = FastAPI()
    env = {"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, **GOOGLE_ENV}
    mount_web_v32(app, env=env, gateway=gw, google_verifier=FakeVerifier(), token_exchanger=lambda f: {"id_token": _id_token(holder["nonce"])}, clock=clock)
    client = TestClient(app, base_url=ORIGIN)
    r = client.get("/auth/google/start", follow_redirects=False)
    q = parse_qs(urlsplit(r.headers["location"]).query)
    holder["nonce"] = q["nonce"][0]
    clock.t += 601
    assert "signin-error=expired" in client.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]


def test_default_google_verifier_and_token_exchange_are_used_when_not_injected(monkeypatch) -> None:
    import hivemind.web_v32.app as appmod

    calls = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"id_token": ""}

    def fake_post(url, data, timeout):
        calls["url"] = url
        return _Resp()

    monkeypatch.setattr(appmod.httpx, "post", fake_post)
    app = FastAPI()
    real_id = {"HIVEMIND_GOOGLE_CLIENT_ID": "123456789-abc.apps.googleusercontent.com", "HIVEMIND_GOOGLE_CLIENT_SECRET": "csecret"}
    mount_web_v32(app, env={"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, **real_id}, gateway=FakeGateway())
    client = TestClient(app, base_url=ORIGIN)
    r = client.get("/auth/google/start", follow_redirects=False)
    q = parse_qs(urlsplit(r.headers["location"]).query)
    cb = client.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False)
    assert calls["url"] == appmod.GOOGLE_TOKEN_URL and "signin-error=exchange" in cb.headers["location"]

    def failing_post(url, data, timeout):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(appmod.httpx, "post", failing_post)
    r = client.get("/auth/google/start", follow_redirects=False)
    q = parse_qs(urlsplit(r.headers["location"]).query)
    assert "signin-error=exchange" in client.get(f"/auth/google/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]


# ---- reset mailer -------------------------------------------------------------------------------------------------------

ENV_OK = {"HIVEMIND_SMTP_HOST": "smtp.example.org", "HIVEMIND_SMTP_FROM": "noreply@example.org", "HIVEMIND_PUBLIC_WEB_URL": "https://hivemind.example.org"}


class _FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.sent, self.tls, self.logged = host, port, [], False, None
        self.offers_starttls = True
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self):
        pass

    def has_extn(self, name):
        return self.offers_starttls

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        self.logged = user

    def send_message(self, msg):
        self.sent.append(msg)


def test_mailer_is_off_until_fully_configured() -> None:
    assert mailer.reset_email_configured({}) is False
    assert mailer.reset_email_configured({**ENV_OK, "HIVEMIND_PUBLIC_WEB_URL": "http://insecure.example.org"}) is False
    assert mailer.reset_email_configured(ENV_OK) is True
    assert mailer.send_reset_email("a@example.org", "tok", {}) is False
    assert mailer.send_reset_email("not-an-email", "tok", ENV_OK) is False


def test_reset_link_puts_the_token_in_the_fragment_and_the_message_has_no_secret_but_the_link() -> None:
    link = mailer.reset_link("TOKEN123", ENV_OK)
    assert link == "https://hivemind.example.org/auth/reset#token=TOKEN123"
    msg = mailer.build_reset_message("a@example.org", link, "noreply@example.org")
    assert msg["To"] == "a@example.org" and link in msg.get_content() and "30 minutes" in msg.get_content()


def test_mailer_uses_starttls_and_refuses_clear_text(monkeypatch) -> None:
    _FakeSMTP.instances = []
    monkeypatch.setattr(mailer.smtplib, "SMTP", _FakeSMTP)
    env = {**ENV_OK, "HIVEMIND_SMTP_USER": "u", "HIVEMIND_SMTP_PASSWORD": "p"}
    assert mailer.send_reset_email("a@example.org", "tok", env) is True
    s = _FakeSMTP.instances[-1]
    assert s.tls and s.logged == "u" and len(s.sent) == 1

    class _NoTLS(_FakeSMTP):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.offers_starttls = False

    monkeypatch.setattr(mailer.smtplib, "SMTP", _NoTLS)
    assert mailer.send_reset_email("a@example.org", "tok", env) is False
    assert _FakeSMTP.instances[-1].sent == [], "never sent in clear text"


def test_mailer_uses_implicit_tls_on_465_and_never_raises(monkeypatch) -> None:
    _FakeSMTP.instances = []
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", _FakeSMTP)
    env = {**ENV_OK, "HIVEMIND_SMTP_PORT": "465", "HIVEMIND_SMTP_USER": "u", "HIVEMIND_SMTP_PASSWORD": "p"}
    assert mailer.send_reset_email("a@example.org", "tok", env) is True and _FakeSMTP.instances[-1].port == 465

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", boom)
    assert mailer.send_reset_email("a@example.org", "tok", env) is False


def test_a_malformed_google_client_id_disables_google_but_not_the_portal() -> None:
    app = FastAPI()
    env = {"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, "HIVEMIND_GOOGLE_CLIENT_ID": "typo", "HIVEMIND_GOOGLE_CLIENT_SECRET": "s"}
    mount_web_v32(app, env=env, gateway=FakeGateway())
    client = TestClient(app, base_url=ORIGIN)
    assert client.get("/v32").status_code == 200
    assert client.get("/v32/api/capabilities").json()["google"] is False
    assert client.get("/auth/google/start", follow_redirects=False).status_code == 404
