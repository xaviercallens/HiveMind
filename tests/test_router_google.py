"""Sign in with Google: the ID-token verifier (one negative control per rule), the store, the endpoints and the page."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.router.google_auth import GoogleAuthError, GoogleIdTokenVerifier, fetch_google_jwks
from hivemind.router.signup_page import content_security_policy, render_signup_page

CLIENT_ID = "123456789012-abcdefghijklmnop.apps.googleusercontent.com"
NOW = 1_800_000_000.0
SECRET = b"g" * 32


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_key(bits: int = 2048) -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=bits)


def jwk(key: rsa.RSAPrivateKey, kid: str) -> dict[str, str]:
    numbers = key.public_key().public_numbers()
    return {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid,
            "n": b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")), "e": b64(numbers.e.to_bytes(3, "big"))}


def claims(**override: Any) -> dict[str, Any]:
    base = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "110169484474386276334", "iat": NOW - 5, "exp": NOW + 3000,
            "email": "someone@example.com", "email_verified": True}
    base.update(override)
    return {k: v for k, v in base.items() if v is not None}


def sign(key: rsa.RSAPrivateKey, payload: dict[str, Any], *, kid: str = "k1", alg: str = "RS256") -> str:
    head = b64(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    signature = key.sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{head}.{body}.{b64(signature)}"


class Keys:
    """A fake Google key set whose contents and fetch count the tests control."""

    def __init__(self, key: rsa.RSAPrivateKey, kid: str = "k1") -> None:
        self.keys = {kid: jwk(key, kid)}
        self.fetches = 0
        self.fail = False

    def __call__(self) -> tuple[dict[str, Any], float]:
        self.fetches += 1
        if self.fail:
            raise httpx.ConnectError("offline")
        return {"keys": list(self.keys.values())}, 3600.0


@pytest.fixture(scope="module")
def key() -> rsa.RSAPrivateKey:
    return make_key()


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


def verifier(key: rsa.RSAPrivateKey, clock: Clock | None = None) -> tuple[GoogleIdTokenVerifier, Keys, Clock]:
    keys, clk = Keys(key), clock or Clock()
    return GoogleIdTokenVerifier(CLIENT_ID, jwks_fetcher=keys, clock=clk), keys, clk


def refused(v: GoogleIdTokenVerifier, token: str) -> str:
    with pytest.raises(GoogleAuthError) as caught:
        v.verify(token)
    return caught.value.code


def test_a_valid_token_yields_the_google_subject(key: rsa.RSAPrivateKey) -> None:
    v, _, _ = verifier(key)
    identity = v.verify(sign(key, claims()))
    assert identity.subject == "110169484474386276334" and identity.expires_at == NOW + 3000


def test_both_google_issuer_spellings_are_accepted(key: rsa.RSAPrivateKey) -> None:
    v, _, _ = verifier(key)
    assert v.verify(sign(key, claims(iss="accounts.google.com", sub="a"))).subject == "a"


@pytest.mark.parametrize(("override", "code"), [
    ({"aud": "999-other.apps.googleusercontent.com"}, "bad_audience"),
    ({"aud": [CLIENT_ID]}, "bad_audience"),
    ({"iss": "https://evil.example.com"}, "bad_issuer"),
    ({"exp": NOW - 100}, "expired"),
    ({"iat": NOW - 3000}, "not_fresh"),
    ({"iat": NOW + 3000}, "not_fresh"),
    ({"email_verified": False}, "email_not_verified"),
    ({"email_verified": "true"}, "email_not_verified"),
    ({"email_verified": None}, "email_not_verified"),
    ({"sub": None}, "malformed_token"),
    ({"sub": ""}, "malformed_token"),
    ({"sub": "x" * 256}, "malformed_token"),
    ({"exp": True}, "malformed_token"),
])
def test_each_claim_rule_refuses_its_violation(key: rsa.RSAPrivateKey, override: dict[str, Any], code: str) -> None:
    v, _, _ = verifier(key)
    assert refused(v, sign(key, claims(**override))) == code


def test_a_tampered_payload_or_foreign_key_fails_the_signature(key: rsa.RSAPrivateKey) -> None:
    v, _, _ = verifier(key)
    good = sign(key, claims())
    head, _, sig = good.split(".")
    forged_body = b64(json.dumps(claims(sub="someone-else")).encode())
    assert refused(v, f"{head}.{forged_body}.{sig}") == "bad_signature"
    assert refused(v, sign(make_key(), claims())) == "bad_signature"  # signed by a key Google does not publish under k1


def test_algorithm_confusion_attacks_are_refused(key: rsa.RSAPrivateKey) -> None:
    v, _, _ = verifier(key)
    public_pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    head = b64(json.dumps({"alg": "HS256", "kid": "k1"}).encode())
    body = b64(json.dumps(claims()).encode())
    mac = b64(hmac.new(public_pem, f"{head}.{body}".encode(), hashlib.sha256).digest())
    assert refused(v, f"{head}.{body}.{mac}") == "bad_algorithm"  # HS256 keyed with the public key
    none_head = b64(json.dumps({"alg": "none", "kid": "k1"}).encode())
    assert refused(v, f"{none_head}.{body}.{mac}") == "bad_algorithm"
    assert refused(v, f"{head}.{body}.") == "malformed_token"  # empty signature


@pytest.mark.parametrize("token", ["", "abc", "a.b", "a.b.c.d", "!!!.???.***", "x" * 5000, "é.é.é"])
def test_malformed_tokens_are_refused_without_crashing(key: rsa.RSAPrivateKey, token: str) -> None:
    v, _, _ = verifier(key)
    assert refused(v, token) in {"malformed_token"}


def test_a_token_works_once_and_the_memory_is_bounded_by_expiry(key: rsa.RSAPrivateKey) -> None:
    v, _, clock = verifier(key)
    token = sign(key, claims())
    v.verify(token)
    assert refused(v, token) == "replayed"
    clock.now = NOW + 3100  # past exp + leeway: the token is expired anyway and its memory entry is dropped
    assert refused(v, token) == "expired"
    v.verify(sign(key, claims(sub="other", iat=NOW + 3095, exp=NOW + 6000)))


def test_signing_keys_are_cached_refetched_once_on_rotation_and_never_hammered(key: rsa.RSAPrivateKey) -> None:
    v, keys, clock = verifier(key)
    v.verify(sign(key, claims(sub="a")))
    v.verify(sign(key, claims(sub="b")))
    assert keys.fetches == 1  # cached
    new_key = make_key()
    keys.keys["k2"] = jwk(new_key, "k2")  # Google rotates
    clock.now = NOW + 120
    assert v.verify(sign(new_key, claims(sub="c", iat=NOW + 115, exp=NOW + 3000), kid="k2")).subject == "c"
    assert keys.fetches == 2
    before = keys.fetches
    for _ in range(5):  # an attacker spamming unknown kids cannot make us refetch more than once a minute
        assert refused(v, sign(key, claims(sub="d"), kid="nope")) == "unknown_key"
    assert keys.fetches - before <= 1


def test_weak_keys_and_unreachable_google_fail_closed(key: rsa.RSAPrivateKey) -> None:
    weak = make_key(1024)
    keys = Keys(weak)
    v = GoogleIdTokenVerifier(CLIENT_ID, jwks_fetcher=keys, clock=Clock())
    assert refused(v, sign(weak, claims())) == "unusable_key"
    offline_keys = Keys(key)
    offline_keys.fail = True
    v2 = GoogleIdTokenVerifier(CLIENT_ID, jwks_fetcher=offline_keys, clock=Clock())
    assert refused(v2, sign(key, claims())) == "keys_unavailable"


def test_construction_rejects_bad_client_ids_and_limits() -> None:
    for bad in ("", "not-a-client-id", "123.apps.googleusercontent.com", CLIENT_ID + "\"><script>"):
        with pytest.raises(ValueError):
            GoogleIdTokenVerifier(bad)
    with pytest.raises(ValueError):
        GoogleIdTokenVerifier(CLIENT_ID, max_age_s=0)


def test_default_fetcher_reads_google_keys_and_clamps_cache_lifetime() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.googleapis.com"
        return httpx.Response(200, json={"keys": [{"kid": "k"}]}, headers={"cache-control": "public, max-age=5, must-revalidate"})

    body, ttl = fetch_google_jwks(httpx.Client(transport=httpx.MockTransport(handler)))
    assert body["keys"][0]["kid"] == "k" and ttl == 300.0  # 5 s asked, clamped up to the 5-minute floor


# ------------------------------------------------------------------------------------------------ store


def test_sign_in_creates_then_rotates_keeping_account_balance_and_role() -> None:
    store = RouterStore(":memory:", SECRET)
    account_id, key1, created = store.sign_in_with_identity("google", "sub-1", "producer")
    store.grant(account_id, 7.0, "test")
    same_id, key2, created2 = store.sign_in_with_identity("google", "sub-1", "consumer")  # role on return is ignored
    assert (same_id, created, created2) == (account_id, True, False)
    assert key1 != key2 and store.authenticate(key1) is None  # the old key is dead
    account = store.authenticate(key2)
    assert account is not None and account.role == "producer" and store.balance(account_id) == 7.0
    assert account.handle.startswith("google.") and "sub-1" not in account.handle
    other_id, _, _ = store.sign_in_with_identity("google", "sub-2", "consumer")
    assert other_id != account_id


def test_provider_handles_cannot_be_squatted_and_inputs_are_validated() -> None:
    store = RouterStore(":memory:", SECRET)
    handle = store.authenticate(store.sign_in_with_identity("google", "sub-1")[1]).handle  # type: ignore[union-attr]
    from hivemind.router.accounts import ReservedHandle

    with pytest.raises(ReservedHandle):
        store.register(handle, "consumer")
    with pytest.raises(ReservedHandle):
        store.register("google.anything", "consumer")
    for bad in (("facebook", "s", "consumer"), ("google", "", "consumer"), ("google", "x" * 256, "consumer"), ("google", "s", "admin")):
        with pytest.raises(ValueError):
            store.sign_in_with_identity(*bad)
    assert store.verify_chain()


# ------------------------------------------------------------------------------------------------ gateway


def make_app(key: rsa.RSAPrivateKey, **kwargs: Any) -> tuple[TestClient, RouterStore, Clock]:
    store, clock = RouterStore(":memory:", SECRET), Clock()
    keys = Keys(key)
    verifier_ = GoogleIdTokenVerifier(CLIENT_ID, jwks_fetcher=keys, clock=clock)
    app = create_router_app(store, clock=clock, google_verifier=verifier_, **kwargs)
    return TestClient(app), store, clock


def google_signin(client: TestClient, token: str, role: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"id_token": token}
    if role:
        body["role"] = role
    return client.post("/v1/hivemind/register/google", json=body)


def test_google_signin_creates_an_account_whose_key_works(key: rsa.RSAPrivateKey) -> None:
    client, store, _ = make_app(key)
    reply = google_signin(client, sign(key, claims()), "producer")
    assert reply.status_code == 200 and reply.headers["cache-control"] == "no-store"
    body = reply.json()
    assert body["created"] is True and body["role"] == "producer" and body["api_key"].startswith("hm-")
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {body['api_key']}"})
    assert me.status_code == 200 and me.json()["account_id"] == body["account_id"] and me.json()["balance"] == 0.0
    assert "someone@example.com" not in reply.text + me.text  # the e-mail address is never returned or stored


def test_signing_in_again_rotates_the_key_and_a_replayed_token_is_refused(key: rsa.RSAPrivateKey) -> None:
    client, _, clock = make_app(key)
    first = google_signin(client, sign(key, claims())).json()
    clock.now += 60
    second_token = sign(key, claims(iat=NOW + 55, exp=NOW + 3000))
    second = google_signin(client, second_token).json()
    assert second["created"] is False and second["account_id"] == first["account_id"]
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {first['api_key']}"}).status_code == 401
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {second['api_key']}"}).status_code == 200
    replay = google_signin(client, second_token)
    assert replay.status_code == 401 and replay.json()["error"]["code"] == "google_replayed"


@pytest.mark.parametrize(("override", "code"), [({"aud": "9-x.apps.googleusercontent.com"}, "google_bad_audience"),
                                                ({"exp": NOW - 99}, "google_expired"), ({"email_verified": False}, "google_email_not_verified")])
def test_bad_tokens_are_401_with_a_stable_code_and_no_account_is_created(key: rsa.RSAPrivateKey, override: dict[str, Any], code: str) -> None:
    client, store, _ = make_app(key)
    reply = google_signin(client, sign(key, claims(**override)))
    assert reply.status_code == 401 and reply.json()["error"]["code"] == code
    assert "eyJ" not in reply.text  # the token is never echoed
    assert store.consumer_max_prices() == {}


def test_google_signin_is_disabled_without_a_client_id_and_the_page_says_so() -> None:
    client = TestClient(create_router_app(RouterStore(":memory:", SECRET)))
    reply = client.post("/v1/hivemind/register/google", json={"id_token": "x"})
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "google_signin_disabled"
    page = client.get("/signup")
    assert page.status_code == 200 and "not enabled" in page.text and "<script" not in page.text


def test_unreachable_google_is_a_503_not_a_login(key: rsa.RSAPrivateKey) -> None:
    store, clock = RouterStore(":memory:", SECRET), Clock()
    keys = Keys(key)
    keys.fail = True
    client = TestClient(create_router_app(store, clock=clock, google_verifier=GoogleIdTokenVerifier(CLIENT_ID, jwks_fetcher=keys, clock=clock)))
    reply = google_signin(client, sign(key, claims()))
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "google_unreachable"


def test_closing_open_registration_leaves_google_as_the_only_way_in(key: rsa.RSAPrivateKey) -> None:
    client, _, _ = make_app(key, open_registration=False)
    closed = client.post("/v1/hivemind/register", json={"handle": "someone", "role": "consumer"})
    assert closed.status_code == 403 and closed.json()["error"]["code"] == "registration_closed"
    assert google_signin(client, sign(key, claims())).status_code == 200


def test_open_registration_cannot_claim_a_provider_handle(key: rsa.RSAPrivateKey) -> None:
    client, _, _ = make_app(key)
    reply = client.post("/v1/hivemind/register", json={"handle": "google.0123456789abcdef", "role": "consumer"})
    assert reply.status_code == 400 and reply.json()["error"]["code"] == "invalid_handle"
    assert client.post("/v1/hivemind/register", json={"handle": "normal-user", "role": "consumer"}).status_code == 200


def test_google_signin_shares_the_registration_rate_limit(key: rsa.RSAPrivateKey) -> None:
    client, _, _ = make_app(key)
    codes = [google_signin(client, "not.a.token").status_code for _ in range(12)]
    assert 401 in codes and 429 in codes  # bad tokens are throttled like any other registration attempt


def test_signup_page_carries_a_strict_policy_a_matching_nonce_and_only_the_validated_client_id(key: rsa.RSAPrivateKey) -> None:
    client, _, _ = make_app(key)
    page = client.get("/signup")
    policy = page.headers["content-security-policy"]
    nonce = policy.split("'nonce-")[1].split("'")[0]
    assert page.status_code == 200 and CLIENT_ID in page.text and page.headers["cache-control"] == "no-store"
    assert page.text.count(f'nonce="{nonce}"') >= 3  # style + inline script + library script all carry this request's nonce
    assert "default-src 'none'" in policy and "'unsafe-inline'" not in policy and "'unsafe-eval'" not in policy
    assert "frame-ancestors 'none'" in policy and "https://accounts.google.com/gsi/client" in policy
    other = client.get("/signup").headers["content-security-policy"]
    assert other != policy  # a fresh nonce per request
    assert "innerHTML" not in page.text  # the key and messages are written with textContent only
    with pytest.raises(ValueError):
        render_signup_page('1-a.apps.googleusercontent.com"; alert(1); "', "n")
    assert "nonce-abc" in content_security_policy("abc")


def _load_server(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("HIVEMIND_ROUTER_SECRET", "s" * 40)
    monkeypatch.setenv("HIVEMIND_ROUTER_DB", ":memory:")
    for name in ("HIVEMIND_GOOGLE_CLIENT_ID", "HIVEMIND_OPEN_REGISTRATION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("deploy_server_under_test", Path(__file__).resolve().parents[1] / "deploy/gcp_openrouter/server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_wires_the_google_client_id_and_refuses_to_close_registration_without_it(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_server(monkeypatch)
    assert "not enabled" in TestClient(module.app).get("/signup").text  # default: Google off, handle registration open
    on = _load_server(monkeypatch, HIVEMIND_GOOGLE_CLIENT_ID=CLIENT_ID)
    assert CLIENT_ID in TestClient(on.app).get("/signup").text
    closed = _load_server(monkeypatch, HIVEMIND_GOOGLE_CLIENT_ID=CLIENT_ID, HIVEMIND_OPEN_REGISTRATION="0")
    assert TestClient(closed.app).post("/v1/hivemind/register", json={"handle": "x", "role": "consumer"}).status_code == 403
    with pytest.raises(RuntimeError, match="needs HIVEMIND_GOOGLE_CLIENT_ID"):
        _load_server(monkeypatch, HIVEMIND_OPEN_REGISTRATION="0")
    with pytest.raises(ValueError):
        _load_server(monkeypatch, HIVEMIND_GOOGLE_CLIENT_ID="not-a-client-id")
