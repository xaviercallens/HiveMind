"""Intensive security tests of the web portal (v3.3), run by the required CI job (deterministic profile) and, with
HYPOTHESIS_PROFILE=hivemind-intense, by the web-auth intense workflow and nightly.

Routes are discovered from ``app.routes`` (not listed by hand), so a new route cannot escape these invariants.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hivemind.web_v32 import mount_web_v32
from tests.test_web_v32 import GOOGLE_ENV, ORIGIN, SECRET, FakeGateway, FakeVerifier, _client, _csrf, _start

settings.register_profile("hivemind-ci", derandomize=True, deadline=None, max_examples=30,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
settings.register_profile("hivemind-intense", deadline=None, max_examples=1500,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
_name = os.environ.get("HYPOTHESIS_PROFILE", "")
PROFILE = settings.get_profile(_name if _name.startswith("hivemind-") else "hivemind-ci")
STATIC = Path(__file__).resolve().parents[1] / "src" / "hivemind" / "web_v32" / "static"
KEY_RE = re.compile(r"hm-[A-Za-z0-9_\-]{8,}")


def _v3_routes(app: FastAPI) -> list[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute) and (r.path.startswith("/v32") or r.path.startswith("/auth/"))]


def _post_paths() -> list[str]:
    app = FastAPI()
    mount_web_v32(app, env={"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, **GOOGLE_ENV}, gateway=FakeGateway(), google_verifier=FakeVerifier())
    return sorted({r.path for r in _v3_routes(app) if "POST" in r.methods})


POST_PATHS = _post_paths()


def test_route_discovery_finds_every_state_changing_route() -> None:
    assert {"/v32/api/login", "/v32/api/register", "/v32/api/logout", "/v32/api/change-password", "/v32/api/rotate-key",
            "/v32/api/password-reset/request", "/v32/api/password-reset/confirm"} <= set(POST_PATHS)


# ---- headers on every response, including the framework's own errors ------------------------------------------------

def _assert_strict(resp, where: str) -> None:
    csp = resp.headers.get("content-security-policy", "")
    assert csp, where
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp, (where, csp)
    assert "frame-ancestors 'none'" in csp, (where, csp)
    assert resp.headers.get("x-content-type-options") == "nosniff", where


@pytest.fixture(scope="module")
def dashboard_client() -> TestClient:
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    return TestClient(create_hivemind_at_home_app())


@pytest.mark.parametrize("method,path", [
    ("GET", "/v32"), ("GET", "/v32/api/facts"), ("GET", "/v32/api/session"), ("GET", "/v32/api/capabilities"),
    ("GET", "/v32/static/app.js"), ("GET", "/v32/api/does-not-exist"), ("GET", "/v32/static/%2e%2e%2fapp.py"),
    ("GET", "/v32/api/login"), ("POST", "/v32/api/session"), ("GET", "/auth/google/start?link=not-a-number"),
    ("GET", "/auth/reset"), ("POST", "/v32/api/login"),
])
def test_every_v3_response_in_the_full_dashboard_keeps_the_strict_policy(dashboard_client: TestClient, method: str, path: str) -> None:
    resp = dashboard_client.request(method, path, json={} if method == "POST" else None)
    # 503 is the intended fail-closed answer of sign-in routes when no session secret is configured
    assert resp.status_code < 500 or resp.status_code == 503, (method, path, resp.status_code)
    _assert_strict(resp, f"{method} {path} -> {resp.status_code}")


def test_every_discovered_get_route_answers_with_strict_headers() -> None:
    client, _, _ = _client(GOOGLE_ENV)
    app = client.app
    for route in _v3_routes(app):
        if "GET" not in route.methods or "{" in route.path:
            continue
        resp = client.get(route.path, follow_redirects=False)
        assert resp.status_code < 500, route.path
        _assert_strict(resp, route.path)
        if not route.path.startswith("/v32/static"):
            assert "no-store" in resp.headers.get("cache-control", ""), route.path


# ---- CSRF and Origin matrix over every state-changing route ------------------------------------------------------------

@pytest.mark.parametrize("path", POST_PATHS)
@pytest.mark.parametrize("case", ["no-token", "wrong-token", "foreign-origin", "no-origin-no-referer", "token-in-body-only"])
def test_csrf_matrix(path: str, case: str) -> None:
    client, gw, _ = _client()
    token = _csrf(client)["X-CSRF-Token"]
    headers = {
        "no-token": {"Origin": ORIGIN},
        "wrong-token": {"X-CSRF-Token": "wrong-" + token, "Origin": ORIGIN},
        "foreign-origin": {"X-CSRF-Token": token, "Origin": "https://evil.example.com"},
        "no-origin-no-referer": {"X-CSRF-Token": token},
        "token-in-body-only": {"Origin": ORIGIN},
    }[case]
    body = {"csrf": token, "identifier": "alice@example.org", "password": "AlicePassword#2026"}
    resp = client.post(path, json=body, headers=headers)
    assert resp.status_code == 403, (path, case, resp.status_code)
    assert gw.calls == [], "the gateway is never reached without a valid CSRF check"


# ---- fuzzing every POST with arbitrary JSON -----------------------------------------------------------------------------

JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=300),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=6),
    max_leaves=25,
)
FIELDS = st.fixed_dictionaries({}, optional={
    "identifier": st.text(max_size=400), "password": st.text(max_size=400), "email": st.text(max_size=400),
    "handle": st.text(max_size=100), "new_password": st.text(max_size=400), "token": st.text(max_size=200),
})


@PROFILE
@given(path=st.sampled_from(POST_PATHS), body=st.one_of(JSON, FIELDS))
def test_no_post_ever_fails_with_a_server_error_or_leaks_a_key(path: str, body) -> None:
    client, _gw, _ = _client()
    h = _csrf(client)
    resp = client.post(path, json=body, headers=h)
    assert resp.status_code < 500, (path, body, resp.status_code)
    if KEY_RE.search(resp.text):
        assert "api_key_once" in resp.json(), (path, resp.text[:200])


@PROFILE
@given(raw=st.binary(max_size=2000))
def test_non_json_bodies_are_refused_cleanly(raw: bytes) -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    resp = client.post("/v32/api/login", content=raw, headers={**h, "Content-Type": "application/json"})
    assert resp.status_code in (401, 400) and gw.calls == []


# ---- session cookie tampering ------------------------------------------------------------------------------------------

def _signed_in() -> tuple[TestClient, str]:
    client, _, _ = _client()
    h = _csrf(client)
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    return client, client.cookies.get("hm_s")


@PROFILE
@given(pos=st.integers(min_value=0, max_value=10_000), char=st.sampled_from("ABCDEFabcdef0123456789-_"))
def test_any_change_to_the_session_cookie_signs_out(pos: int, char: str) -> None:
    client, cookie = _signed_in()
    i = pos % len(cookie)
    forged = cookie[:i] + char + cookie[i + 1:]
    client.cookies.set("hm_s", forged, domain="hivemind.example.org")
    authenticated = client.get("/v32/api/session").json()["authenticated"]
    assert authenticated is (forged == cookie)


def test_a_cookie_sealed_with_another_secret_is_refused() -> None:
    other = FastAPI()
    mount_web_v32(other, env={"HIVEMIND_WEB_SESSION_SECRET": "o" * 48, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN}, gateway=FakeGateway())
    oc = TestClient(other, base_url=ORIGIN)
    h = _csrf(oc)
    oc.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    client, _, _ = _client()
    client.cookies.set("hm_s", oc.cookies.get("hm_s"), domain="hivemind.example.org")
    assert client.get("/v32/api/session").json()["authenticated"] is False


# ---- redirects never leave the site ----------------------------------------------------------------------------------

@PROFILE
@given(state=st.text(max_size=80), code=st.text(max_size=80), error=st.text(max_size=40))
def test_google_callback_only_redirects_to_the_portal(state: str, code: str, error: str) -> None:
    client, _, holder = _client(GOOGLE_ENV)
    _start(client, holder)
    resp = client.get("/auth/google/callback", params={"state": state, "code": code, "error": error}, follow_redirects=False)
    assert resp.status_code in (302, 404)
    if resp.status_code == 302:
        assert resp.headers["location"].startswith("/v32#")


@PROFILE
@given(host=st.from_regex(r"[a-z0-9]{1,12}\.(example|evil)\.(com|net)", fullmatch=True), path=st.sampled_from(["/v32", "/auth/reset", "/v32/api/facts"]))
def test_host_header_never_controls_the_redirect_target(host: str, path: str) -> None:
    client, _, _ = _client()
    resp = client.get(path, headers={"Host": host}, follow_redirects=False)
    if host != "hivemind.example.org":
        assert resp.status_code == 301 and resp.headers["location"].startswith(ORIGIN + "/")
        assert host not in resp.headers["location"]


# ---- static analysis of the page ---------------------------------------------------------------------------------------

def test_page_script_uses_no_dangerous_sinks() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "setTimeout(\"", "setInterval(\""):
        assert sink not in js, sink
    assert "localStorage" not in js and "sessionStorage" not in js, "nothing secret is stored in the browser"


def test_page_markup_has_no_inline_code_and_every_input_is_labelled() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"\son[a-z]+\s*=", html), "no inline event handlers"
    assert "javascript:" not in html.lower()
    for m in re.finditer(r"<input\b[^>]*>", html):
        before = html[: m.start()]
        assert before.rfind("<label") > before.rfind("</label>"), f"input without an enclosing label: {m.group(0)[:80]}"
    for src in re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html):
        assert src.startswith("/"), f"scripts and styles are same-origin only: {src}"


@pytest.mark.parametrize("header", ["\u00e9t\u00e9", "\x80", "token\u2028"])
def test_non_ascii_csrf_headers_and_cookies_are_refused_not_crashed(header: str) -> None:
    client, gw, _ = _client()
    client.get("/v32")
    resp = client.post("/v32/api/login", json={}, headers={"X-CSRF-Token": header.encode("utf-8"), "Origin": ORIGIN})
    assert resp.status_code == 403 and gw.calls == []
