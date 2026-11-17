"""General OpenID Connect sign-in: the provider-agnostic verifier, per-store provider prefixes, the endpoints and the config."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from hivemind.router import RouterStore, create_router_app
from hivemind.router.accounts import ReservedHandle
from hivemind.router.oidc_auth import (
    OidcAuthError,
    OidcIdTokenVerifier,
    OidcProvider,
    providers_from_json,
)

NOW = 1_800_000_000.0
ISSUER = "https://login.acme.test/"
AUD = "acme-client-1"
SECRET = b"o" * 32


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwk(key: rsa.RSAPrivateKey, kid: str) -> dict[str, str]:
    numbers = key.public_key().public_numbers()
    return {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid,
            "n": b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")), "e": b64(numbers.e.to_bytes(3, "big"))}


def sign(key: rsa.RSAPrivateKey, **override: Any) -> str:
    payload = {"iss": ISSUER, "aud": AUD, "sub": "user-42", "iat": NOW - 5, "exp": NOW + 3000, "email_verified": True}
    payload.update(override)
    payload = {k: v for k, v in payload.items() if v is not None}
    head = b64(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    body = b64(json.dumps(payload).encode())
    return f"{head}.{body}.{b64(key.sign(f'{head}.{body}'.encode(), padding.PKCS1v15(), hashes.SHA256()))}"


@pytest.fixture(scope="module")
def key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def acme(key: rsa.RSAPrivateKey, *, verified: bool = True, name: str = "acme", audience: str = AUD) -> OidcIdTokenVerifier:
    provider = OidcProvider(name, frozenset({ISSUER}), "https://login.acme.test/jwks.json", audience, verified)
    return OidcIdTokenVerifier(provider, jwks_fetcher=lambda: ({"keys": [jwk(key, "k1")]}, 3600.0), clock=lambda: NOW)


def code_of(verifier: OidcIdTokenVerifier, token: str) -> str:
    with pytest.raises(OidcAuthError) as caught:
        verifier.verify(token)
    return caught.value.code


# ---------------------------------------------------------------- verifier

def test_a_non_google_provider_token_verifies_and_each_rule_still_refuses(key: rsa.RSAPrivateKey) -> None:
    assert acme(key).verify(sign(key)).subject == "user-42"
    assert code_of(acme(key), sign(key, iss="https://accounts.google.com")) == "bad_issuer"  # Google's issuer is not Acme's
    assert code_of(acme(key), sign(key, aud="someone-elses-client")) == "bad_audience"
    assert code_of(acme(key), sign(key, exp=NOW - 100)) == "expired"
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert code_of(acme(key), sign(other)) == "bad_signature"


def test_verified_email_is_required_unless_the_provider_is_configured_without_it(key: rsa.RSAPrivateKey) -> None:
    unverified = sign(key, email_verified=None, sub="no-email-claim")
    assert code_of(acme(key, verified=True), unverified) == "email_not_verified"
    assert acme(key, verified=False).verify(unverified).subject == "no-email-claim"


def test_a_token_is_single_use_per_verifier(key: rsa.RSAPrivateKey) -> None:
    v, token = acme(key), sign(key)
    v.verify(token)
    assert code_of(v, token) == "replayed"
    # a re-encoded spelling of the same signed token is not a new token: padding or extra characters are refused, and the
    # replay cache is keyed on the signed bytes (it used to hash the raw string, so tok + "==" was accepted again)
    for variant in (token + "==", token + "=", token.replace(".", ".=", 1)):
        assert code_of(v, variant) in ("malformed_token", "replayed"), variant
    h, c, s = token.split(".")
    last = s[-1]
    flipped = s[:-1] + ("A" if last != "A" else "B")  # same signature bytes only if the unused trailing bits change
    assert code_of(v, f"{h}.{c}.{flipped}") in ("malformed_token", "replayed", "bad_signature")


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=30))
def test_provider_names_outside_the_safe_pattern_are_rejected(name: str) -> None:
    import re

    valid = re.fullmatch(r"[a-z][a-z0-9]{1,15}", name) is not None
    if valid:
        assert OidcProvider(name, frozenset({ISSUER}), "https://x.test/jwks", "c").name == name
    else:
        with pytest.raises(ValueError):
            OidcProvider(name, frozenset({ISSUER}), "https://x.test/jwks", "c")


def test_provider_needs_https_keys_an_issuer_and_a_client_id() -> None:
    for kwargs in ({"jwks_url": "http://x.test/jwks"}, {"issuers": frozenset()}, {"client_id": ""}):
        args = {"name": "acme", "issuers": frozenset({ISSUER}), "jwks_url": "https://x.test/jwks", "client_id": "c", **kwargs}
        with pytest.raises(ValueError):
            OidcProvider(**args)


# ---------------------------------------------------------------- store

def test_providers_get_separate_accounts_for_the_same_subject_and_reserve_their_prefix() -> None:
    store = RouterStore(":memory:", SECRET)
    with pytest.raises(ValueError):
        store.sign_in_with_identity("acme", "user-42")  # not enabled yet
    store.allow_identity_provider("acme")
    store.allow_identity_provider("globex")
    a, _, created_a = store.sign_in_with_identity("acme", "user-42")
    g, _, created_g = store.sign_in_with_identity("globex", "user-42")
    assert created_a and created_g and a != g  # the provider name is part of the hashed identity
    with pytest.raises(ReservedHandle):
        store.register("acme.anything", "consumer")
    assert store.register("acmeish", "consumer")[0]  # a different name is not caught by the prefix


def test_enabling_a_provider_is_refused_when_a_handle_already_uses_its_prefix() -> None:
    store = RouterStore(":memory:", SECRET)
    store.register("acme.early", "consumer")
    with pytest.raises(ReservedHandle):
        store.allow_identity_provider("acme")
    with pytest.raises(ValueError):
        store.allow_identity_provider("Bad Name")


# ---------------------------------------------------------------- endpoints

def app_with(key: rsa.RSAPrivateKey, *, open_registration: bool = True) -> tuple[TestClient, RouterStore]:
    store = RouterStore(":memory:", SECRET)
    app = create_router_app(store, oidc_verifiers={"acme": acme(key)}, open_registration=open_registration, clock=lambda: NOW)
    return TestClient(app), store


def test_oidc_sign_in_creates_an_account_then_rotates_the_key(key: rsa.RSAPrivateKey) -> None:
    client, _ = app_with(key)
    first = client.post("/v1/hivemind/register/oidc/acme", json={"id_token": sign(key), "role": "consumer"})
    assert first.status_code == 200 and first.json()["created"] is True
    assert first.headers["cache-control"] == "no-store"
    k1 = first.json()["api_key"]
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {k1}"}).status_code == 200
    second = client.post("/v1/hivemind/register/oidc/acme", json={"id_token": sign(key, jti="again")})
    assert second.json()["created"] is False and second.json()["account_id"] == first.json()["account_id"]
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {k1}"}).status_code == 401  # old key retired
    assert client.post("/v1/hivemind/register/oidc/acme", json={"id_token": sign(key)}).status_code == 401  # replayed token


def test_bad_unknown_or_foreign_tokens_create_nothing(key: rsa.RSAPrivateKey) -> None:
    client, store = app_with(key)
    assert client.post("/v1/hivemind/register/oidc/nobody", json={"id_token": sign(key)}).status_code == 404
    bad = client.post("/v1/hivemind/register/oidc/acme", json={"id_token": sign(key, aud="other")})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "acme_bad_audience"
    assert store.get_account(store.account_id_for("acme.x")) is None
    assert client.post("/v1/hivemind/register/oidc/google", json={"id_token": sign(key)}).status_code == 404  # Google is not an extra provider


def test_provider_listing_is_public_and_closing_handles_leaves_oidc_as_the_way_in(key: rsa.RSAPrivateKey) -> None:
    client, _ = app_with(key, open_registration=False)
    listing = client.get("/v1/hivemind/signin/providers").json()
    assert [p["name"] for p in listing["providers"]] == ["acme"] and listing["handle_registration"] is False
    assert listing["providers"][0]["register"] == "/v1/hivemind/register/oidc/acme"
    assert client.post("/v1/hivemind/register", json={"handle": "x", "role": "consumer"}).status_code == 403
    assert client.post("/v1/hivemind/register/oidc/acme", json={"id_token": sign(key)}).status_code == 200


def test_google_cannot_be_configured_as_an_extra_provider(key: rsa.RSAPrivateKey) -> None:
    with pytest.raises(ValueError, match="google"):
        create_router_app(RouterStore(":memory:", SECRET), oidc_verifiers={"google": acme(key, name="google")})


# ---------------------------------------------------------------- operator config

GOOD = {"name": "acme", "issuer": ISSUER, "jwks_url": "https://login.acme.test/jwks.json", "client_id": AUD}


def test_json_config_builds_verifiers_and_rejects_every_malformed_form() -> None:
    built = providers_from_json(json.dumps([GOOD, {**GOOD, "name": "globex", "issuers": [ISSUER, "https://b.test/"]}]))
    assert set(built) == {"acme", "globex"} and built["globex"].provider.issuers == {ISSUER, "https://b.test/"}
    bad = ["not json", "{}", json.dumps([{**GOOD, "name": "google"}]), json.dumps([GOOD, GOOD]), json.dumps([{**GOOD, "extra": 1}]),
           json.dumps([{**GOOD, "jwks_url": "http://insecure.test/k"}]), json.dumps([{**GOOD, "require_verified_email": "no"}]),
           json.dumps(["acme"]), json.dumps([{**GOOD, "issuer": None}])]
    for text in bad:
        with pytest.raises((ValueError, TypeError)):
            providers_from_json(text)


def test_server_reads_the_provider_list_and_lets_it_replace_google_for_closing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    def load(**env: str) -> Any:
        monkeypatch.setenv("HIVEMIND_ROUTER_SECRET", "s" * 40)
        monkeypatch.setenv("HIVEMIND_ROUTER_DB", ":memory:")
        for name in ("HIVEMIND_GOOGLE_CLIENT_ID", "HIVEMIND_OPEN_REGISTRATION", "HIVEMIND_OIDC_PROVIDERS"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        spec = importlib.util.spec_from_file_location("deploy_server_oidc", Path(__file__).resolve().parents[1] / "deploy/gcp_openrouter/server.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    on = load(HIVEMIND_OIDC_PROVIDERS=json.dumps([GOOD]))
    assert [p["name"] for p in TestClient(on.app).get("/v1/hivemind/signin/providers").json()["providers"]] == ["acme"]
    closed = load(HIVEMIND_OIDC_PROVIDERS=json.dumps([GOOD]), HIVEMIND_OPEN_REGISTRATION="0")
    assert TestClient(closed.app).post("/v1/hivemind/register", json={"handle": "x", "role": "consumer"}).status_code == 403
    with pytest.raises(RuntimeError, match="nobody could register"):
        load(HIVEMIND_OPEN_REGISTRATION="0")
    with pytest.raises(ValueError):
        load(HIVEMIND_OIDC_PROVIDERS="[{\"name\": \"BAD\"}]")
