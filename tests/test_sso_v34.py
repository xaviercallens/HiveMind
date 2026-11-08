"""v3.4 single sign-on: generic OpenID Connect (Entra, Okta, GitLab...) and GitHub, tested against the REAL verifiers.

OpenID Connect tokens are signed here with a test RSA key and served through an injected JWKS, so the production
verifier runs unmodified; the attacks are the ones that break naive implementations (wrong issuer or audience, expiry, alg
"none", HS256 signed with the public key, unknown key id, wrong nonce, replay). GitHub runs through httpx.MockTransport.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.github_auth import GitHubAuthError, GitHubCodeVerifier
from hivemind.router.oidc_auth import OidcIdTokenVerifier, OidcProvider, providers_from_json
from hivemind.web_v32 import GatewayClient, mount_web_v32

SECRET = b"0123456789abcdef0123456789abcdef"
ORIGIN = "https://hivemind.example.org"
ISSUER = "https://login.acme.example/tenant-123/v2.0"
CLIENT = "acme-client-id"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwk(key, kid: str) -> dict:
    nums = key.public_key().public_numbers()
    to_b = lambda i: i.to_bytes((i.bit_length() + 7) // 8, "big")  # noqa: E731
    return {"kty": "RSA", "kid": kid, "alg": "RS256", "use": "sig", "n": _b64(to_b(nums.n)), "e": _b64(to_b(nums.e))}


JWKS = {"keys": [_jwk(KEY, "k1")]}


def _token(claims: dict, *, key=KEY, kid: str = "k1", alg: str = "RS256") -> str:
    header = {"alg": alg, "kid": kid, "typ": "JWT"}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}".encode()
    if alg == "RS256":
        sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    elif alg == "HS256":  # algorithm confusion: HMAC keyed with the public key
        pub = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        sig = hmac.new(pub, signing_input, hashlib.sha256).digest()
    else:
        sig = b""
    return signing_input.decode() + "." + _b64(sig)


def _claims(**over) -> dict:
    now = int(time.time())
    c = {"iss": ISSUER, "aud": CLIENT, "sub": "user-42", "iat": now, "exp": now + 600, "nonce": "N", "email": "u@acme.example", "email_verified": True}
    c.update(over)
    return {k: v for k, v in c.items() if v is not None}


def _verifier(require_verified_email: bool = True) -> OidcIdTokenVerifier:
    provider = OidcProvider("acme", frozenset({ISSUER}), "https://login.acme.example/keys", CLIENT, require_verified_email)
    return OidcIdTokenVerifier(provider, jwks_fetcher=lambda: (JWKS, time.time() + 3600))


# ---- the real verifier under attack -------------------------------------------------------------------------------------

def test_a_correctly_signed_token_is_accepted_once() -> None:
    v = _verifier()
    tok = _token(_claims())
    assert v.verify(tok).subject == "user-42"
    with pytest.raises(Exception):
        v.verify(tok)  # replay


@pytest.mark.parametrize("tok", [
    _token(_claims(iss="https://evil.example/")),
    _token(_claims(aud="someone-else")),
    _token(_claims(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200)),
    _token(_claims(), alg="none"),
    _token(_claims(), alg="HS256"),
    _token(_claims(), kid="unknown-kid"),
    _token(_claims(), key=OTHER_KEY),
    _token(_claims(email_verified=False)),
    "not.a.jwt",
])
def test_forged_or_invalid_tokens_are_refused(tok: str) -> None:
    with pytest.raises(Exception):
        _verifier().verify(tok)


def test_entra_style_token_without_email_verified_needs_the_provider_flag_off() -> None:
    tok = _token(_claims(email_verified=None))
    with pytest.raises(Exception):
        _verifier(require_verified_email=True).verify(tok)
    assert _verifier(require_verified_email=False).verify(_token(_claims(email_verified=None))).subject == "user-42"


@pytest.mark.parametrize("issuer", [
    "https://login.microsoftonline.com/common/v2.0",
    "https://login.microsoftonline.com/organizations/v2.0",
    "https://login.microsoftonline.com/{tenantid}/v2.0",
])
def test_tenant_wildcard_issuers_are_refused_at_configuration(issuer: str) -> None:
    entry = {"name": "entra", "issuer": issuer, "jwks_url": "https://login.microsoftonline.com/k", "client_id": "c"}
    with pytest.raises(ValueError):
        providers_from_json(json.dumps([entry]))


def test_provider_json_accepts_the_portal_keys() -> None:
    entry = {"name": "entra", "issuer": "https://login.microsoftonline.com/1111-2222/v2.0", "jwks_url": "https://login.microsoftonline.com/k",
             "client_id": "c", "require_verified_email": False, "authorization_endpoint": "https://login.microsoftonline.com/a",
             "token_endpoint": "https://login.microsoftonline.com/t", "label": "Microsoft"}
    assert set(providers_from_json(json.dumps([entry]))) == {"entra"}


# ---- end to end: portal + real gateway + real verifiers (separate instances, as in production) ---------------------------

def _stack(tmp_path, *, github: GitHubCodeVerifier | None = None, exchanged: dict | None = None, require_verified_email=True):
    store = RouterStore(tmp_path / "gw.sqlite3", SECRET)
    gw_app = create_router_app(store, oidc_verifiers={"acme": _verifier(require_verified_email)}, github_verifier=github)
    gw = GatewayClient("http://gateway", client=TestClient(gw_app, base_url="http://gateway"), forward_secret=SECRET)
    holder: dict = {}
    providers_json = json.dumps([{"name": "acme", "issuer": ISSUER, "jwks_url": "https://login.acme.example/keys", "client_id": CLIENT,
                                  "authorization_endpoint": "https://login.acme.example/authorize", "token_endpoint": "https://login.acme.example/token",
                                  "label": "Acme SSO", "require_verified_email": require_verified_email}])
    env = {"HIVEMIND_WEB_SESSION_SECRET": "w" * 48, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, "HIVEMIND_OIDC_PROVIDERS": providers_json,
           "HIVEMIND_OIDC_ACME_CLIENT_SECRET": "acme-secret"}
    if github is not None:
        env["HIVEMIND_GITHUB_CLIENT_ID"] = "gh-client"

    def exchanger(form):
        holder["form"] = dict(form)
        if exchanged is not None:
            return exchanged
        return {"id_token": _token(_claims(nonce=holder["nonce"]))}

    web = FastAPI()
    mount_web_v32(web, env=env, gateway=gw, token_exchanger=exchanger, oidc_verifiers={"acme": _verifier(require_verified_email)})
    return store, TestClient(web, base_url=ORIGIN), holder


def _start(client, holder, provider="acme", link=False):
    r = client.get(f"/auth/{provider}/start" + ("?link=1" if link else ""), follow_redirects=False)
    assert r.status_code == 302
    q = parse_qs(urlsplit(r.headers["location"]).query)
    holder["nonce"] = q.get("nonce", [""])[0]
    return r, q


def test_capabilities_list_the_configured_providers(tmp_path) -> None:
    _s, client, _ = _stack(tmp_path)
    caps = client.get("/v32/api/capabilities").json()
    assert {"name": "acme", "label": "Acme SSO"} in caps["providers"]


def test_oidc_start_uses_pkce_state_and_nonce(tmp_path) -> None:
    _s, client, holder = _stack(tmp_path)
    r, q = _start(client, holder)
    assert r.headers["location"].startswith("https://login.acme.example/authorize?")
    assert q["client_id"] == [CLIENT] and q["code_challenge_method"] == ["S256"] and q["nonce"] and q["state"]
    assert q["redirect_uri"] == [f"{ORIGIN}/auth/acme/callback"]
    assert "acme-secret" not in r.headers["location"]


def test_oidc_sign_in_end_to_end_creates_a_session(tmp_path) -> None:
    store, client, holder = _stack(tmp_path)
    _r, q = _start(client, holder)
    cb = client.get(f"/auth/acme/callback?state={q['state'][0]}&code=c", follow_redirects=False)
    assert cb.headers["location"].endswith("#account"), cb.headers["location"]
    assert holder["form"]["client_secret"] == "acme-secret" and holder["form"]["code_verifier"]
    s = client.get("/v32/api/session").json()
    assert s["authenticated"] is True and s["handle"] == "Acme SSO account"


def test_oidc_sign_in_refuses_a_token_with_the_wrong_nonce(tmp_path) -> None:
    _s, client, holder = _stack(tmp_path, exchanged={"id_token": _token(_claims(nonce="someone-elses"))})
    _r, q = _start(client, holder)
    cb = client.get(f"/auth/acme/callback?state={q['state'][0]}&code=c", follow_redirects=False)
    assert "signin-error=nonce" in cb.headers["location"]


def test_oidc_sign_in_refuses_a_forged_token(tmp_path) -> None:
    _s, client, holder = _stack(tmp_path, exchanged={"id_token": _token(_claims(nonce="N"), key=OTHER_KEY)})
    _r, q = _start(client, holder)
    assert "signin-error=token" in client.get(f"/auth/acme/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"]


def test_oidc_link_needs_a_session_and_then_opens_the_password_account(tmp_path) -> None:
    store, client, holder = _stack(tmp_path)
    acc_id, _key, _ = store.register_with_credentials("linker", "LinkerPassword#2026", email="linker@example.org")
    client.get("/v32")
    h = {"X-CSRF-Token": client.cookies.get("hm_csrf"), "Origin": ORIGIN}
    client.post("/v32/api/login", json={"identifier": "linker@example.org", "password": "LinkerPassword#2026"}, headers=h)
    _r, q = _start(client, holder, link=True)
    assert client.get(f"/auth/acme/callback?state={q['state'][0]}&code=c", follow_redirects=False).headers["location"].endswith("#linked")
    client.post("/v32/api/logout", json={}, headers=h)
    _r, q = _start(client, holder)
    client.get(f"/auth/acme/callback?state={q['state'][0]}&code=c", follow_redirects=False)
    assert client.get("/v32/api/account").json()["handle"] == "linker", "signing in with the linked identity opens the password account"


def test_a_state_cookie_from_one_provider_does_not_work_for_another(tmp_path) -> None:
    _s, client, holder = _stack(tmp_path, github=_github_verifier(_github_transport()))
    _r, q = _start(client, holder, "acme")
    cookie = client.cookies.get("hm_oidc")
    client.cookies.set("hm_oidc", cookie, domain="hivemind.example.org", path="/auth/github")
    r = client.get(f"/auth/github/callback?state={q['state'][0]}&code=c", follow_redirects=False)
    assert "signin-error" in r.headers["location"]


# ---- GitHub ---------------------------------------------------------------------------------------------------------------

def _github_transport(*, token_ok=True, verified=True, uid=777, used: set | None = None):
    used = used if used is not None else set()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            code = form.get("code")
            if not token_ok or code in used or form.get("client_secret") != "gh-secret":
                return httpx.Response(200, json={"error": "bad_verification_code"})
            used.add(code)
            return httpx.Response(200, json={"access_token": "gho_test", "token_type": "bearer"})
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": uid, "login": "octo"})
        if request.url.path == "/user/emails":
            return httpx.Response(200, json=[{"email": "o@example.org", "verified": verified, "primary": True}])
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _github_verifier(transport) -> GitHubCodeVerifier:
    return GitHubCodeVerifier("gh-client", "gh-secret", f"{ORIGIN}/auth/github/callback", http=httpx.Client(transport=transport))


VERIFIER_43 = "v" * 43


def test_github_code_exchange_identifies_by_numeric_id() -> None:
    gh = _github_verifier(_github_transport())
    ident = gh.verify_code("code-1", VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    assert ident.subject == "777" and ident.login == "octo"


@pytest.mark.parametrize("kwargs,code,why", [
    ({"token_ok": False}, "c", "exchange_failed"),
    ({"verified": False}, "c", "no_verified_email"),
    ({"uid": True}, "c", "no_user"),
])
def test_github_refusals(kwargs, code, why) -> None:
    gh = _github_verifier(_github_transport(**kwargs))
    with pytest.raises(GitHubAuthError) as e:
        gh.verify_code(code, VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    assert e.value.code == why


def test_github_replayed_code_and_foreign_redirect_are_refused() -> None:
    used: set = set()
    gh = _github_verifier(_github_transport(used=used))
    gh.verify_code("once", VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    with pytest.raises(GitHubAuthError):
        gh.verify_code("once", VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    with pytest.raises(GitHubAuthError) as e:
        gh.verify_code("other", VERIFIER_43, "https://evil.example/cb")
    assert e.value.code == "redirect_mismatch"


def test_gateway_never_accepts_a_bare_github_access_token(tmp_path) -> None:
    store = RouterStore(tmp_path / "g.sqlite3", SECRET)
    client = TestClient(create_router_app(store, github_verifier=_github_verifier(_github_transport())))
    r = client.post("/v1/hivemind/register/github", json={"access_token": "gho_stolen_from_another_app"})
    assert r.status_code == 401 and "hm-" not in r.text


def test_github_sign_in_end_to_end(tmp_path) -> None:
    store, client, holder = _stack(tmp_path, github=_github_verifier(_github_transport()))
    r, q = _start(client, holder, "github")
    assert r.headers["location"].startswith("https://github.com/login/oauth/authorize?")
    assert q["scope"] == ["read:user user:email"] and q["code_challenge_method"] == ["S256"] and "nonce" not in q
    cb = client.get(f"/auth/github/callback?state={q['state'][0]}&code=gh-code", follow_redirects=False)
    assert cb.headers["location"].endswith("#account"), cb.headers["location"]
    assert client.get("/v32/api/session").json()["handle"] == "GitHub account"
    caps = client.get("/v32/api/capabilities").json()
    assert {"name": "github", "label": "GitHub"} in caps["providers"]


def test_unconfigured_providers_are_not_offered_and_their_routes_are_closed(tmp_path) -> None:
    _s, client, _ = _stack(tmp_path)
    assert all(p["name"] != "github" for p in client.get("/v32/api/capabilities").json()["providers"])
    assert client.get("/auth/github/start", follow_redirects=False).status_code == 404
    assert client.get("/auth/nosuch/callback?state=s&code=c", follow_redirects=False).status_code == 404


# ---- failure modes of configuration and network ----------------------------------------------------------------------

def test_github_verifier_refuses_incomplete_or_insecure_configuration() -> None:
    with pytest.raises(ValueError):
        GitHubCodeVerifier("", "s", f"{ORIGIN}/auth/github/callback")
    with pytest.raises(ValueError):
        GitHubCodeVerifier("c", "s", "http://insecure.example/auth/github/callback")


def _down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("down", request=request)


def test_github_network_failures_are_reported_as_unreachable() -> None:
    gh = _github_verifier(httpx.MockTransport(_down))
    with pytest.raises(GitHubAuthError) as e:
        gh.verify_code("c", VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    assert e.value.code == "github_unreachable"

    def token_then_down(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gho_x"})
        raise httpx.ConnectError("down", request=request)

    gh = _github_verifier(httpx.MockTransport(token_then_down))
    with pytest.raises(GitHubAuthError) as e:
        gh.verify_code("c", VERIFIER_43, f"{ORIGIN}/auth/github/callback")
    assert e.value.code == "github_unreachable"


def test_gateway_reports_github_unreachable_as_503(tmp_path) -> None:
    store = RouterStore(tmp_path / "g.sqlite3", SECRET)
    client = TestClient(create_router_app(store, github_verifier=_github_verifier(httpx.MockTransport(_down))))
    r = client.post("/v1/hivemind/register/github", json={"code": "c", "code_verifier": VERIFIER_43, "redirect_uri": f"{ORIGIN}/auth/github/callback"})
    assert r.status_code == 503


@pytest.mark.parametrize("bad", [
    "not json",
    json.dumps({"name": "x"}),
    json.dumps([{"name": "acme", "issuer": "https://login.microsoftonline.com/common/v2.0", "jwks_url": "https://k.example/k", "client_id": "c",
                 "authorization_endpoint": "https://a.example/a", "token_endpoint": "https://a.example/t"}]),
])
def test_a_broken_provider_list_disables_those_providers_but_not_the_portal(bad: str) -> None:
    app = FastAPI()
    env = {"HIVEMIND_WEB_SESSION_SECRET": "w" * 48, "HIVEMIND_PUBLIC_WEB_URL": ORIGIN, "HIVEMIND_OIDC_PROVIDERS": bad,
           "HIVEMIND_OIDC_ACME_CLIENT_SECRET": "s"}
    gw = GatewayClient("http://gw", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))))
    mount_web_v32(app, env=env, gateway=gw)
    client = TestClient(app, base_url=ORIGIN)
    assert client.get("/v32").status_code == 200
    assert client.get("/v32/api/capabilities").json()["providers"] == []


def test_gateway_link_route_refuses_unknown_providers_and_missing_keys(tmp_path) -> None:
    store = RouterStore(tmp_path / "g.sqlite3", SECRET)
    client = TestClient(create_router_app(store, oidc_verifiers={"acme": _verifier()}))
    assert client.post("/api/v1/auth/link/nosuch", json={}).status_code == 404
    assert client.post("/api/v1/auth/link/github", json={}).status_code == 503
    assert client.post("/api/v1/auth/link/acme", json={"id_token": _token(_claims())}).status_code == 401
    _a, key, _c = store.register_with_credentials("someone", "SomeonePassword#2026", email="s@example.org")
    bad = client.post("/api/v1/auth/link/acme", json={"id_token": "x.y.z"}, headers={"Authorization": f"Bearer {key}"})
    assert bad.status_code == 401
