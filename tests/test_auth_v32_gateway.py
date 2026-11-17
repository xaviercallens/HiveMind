"""v3.2 gateway auth: password reset by e-mail, capabilities, linking a Google identity to an existing account."""
from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.oidc_auth import OidcIdentity

SECRET = b"0123456789abcdef0123456789abcdef"


class _Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _app(tmp_path, *, email_enabled=True, google=None, clock=None):
    store = RouterStore(tmp_path / "r.sqlite3", SECRET)
    sent: list[tuple[str, str]] = []
    app = create_router_app(
        store,
        starter_grant_hmt=0.0,
        clock=clock or _Clock(),
        google_verifier=google,
        reset_email_sender=lambda to, token: sent.append((to, token)) or True,
        reset_email_enabled=lambda: email_enabled,
    )
    return store, TestClient(app), sent


def _user(store, handle="alice_v32", email="alice@example.org", password="AlicePassword#2026"):
    acc_id, key, created = store.register_with_credentials(handle, password, email=email)
    assert created
    return acc_id, key


def test_capabilities_report_only_what_is_configured(tmp_path) -> None:
    _s, client, _ = _app(tmp_path, email_enabled=False)
    caps = client.get("/api/v1/auth/capabilities").json()
    assert caps["password_login"] is True and caps["password_reset_email"] is False and caps["google"] is False
    (tmp_path / "b").mkdir()
    _s2, client2, _ = _app(tmp_path / "b", email_enabled=True)
    assert client2.get("/api/v1/auth/capabilities").json()["password_reset_email"] is True


def test_reset_request_is_identical_for_known_and_unknown_emails(tmp_path) -> None:
    store, client, sent = _app(tmp_path)
    _user(store)
    known = client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    unknown = client.post("/api/v1/auth/password-reset/request", json={"email": "nobody@example.org"})
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert [to for to, _tok in sent] == ["alice@example.org"]
    assert sent[0][1] not in known.text, "the token is only ever sent by e-mail"


def test_reset_request_refuses_honestly_when_email_is_not_configured(tmp_path) -> None:
    store, client, sent = _app(tmp_path, email_enabled=False)
    _user(store)
    r = client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    assert r.status_code == 503 and sent == []


def test_reset_token_sets_password_once_and_revokes_the_old_key(tmp_path) -> None:
    store, client, sent = _app(tmp_path)
    acc_id, old_key = _user(store)
    client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    token = sent[-1][1]
    weak = client.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "short"})
    assert weak.status_code == 400
    ok = client.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "BrandNewPassword#1"})
    assert ok.status_code == 200 and "hm-" not in ok.text
    again = client.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "AnotherPassword#2"})
    assert again.status_code == 400, "a token works once"
    assert store.authenticate(old_key) is None, "the previous key stops working"
    login = client.post("/api/v1/auth/login", json={"handle_or_email": "alice@example.org", "password": "BrandNewPassword#1"})
    assert login.status_code == 200 and login.json()["account_id"] == acc_id
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "alice@example.org", "password": "AlicePassword#2026"}).status_code == 401


def test_reset_token_expires_after_30_minutes(tmp_path) -> None:
    clock = _Clock()
    store, client, sent = _app(tmp_path, clock=clock)
    _user(store)
    client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    clock.t += 1801
    r = client.post("/api/v1/auth/password-reset/confirm", json={"token": sent[-1][1], "new_password": "BrandNewPassword#1"})
    assert r.status_code == 400


def test_new_reset_request_revokes_the_previous_token_and_tokens_are_stored_hashed(tmp_path) -> None:
    store, client, sent = _app(tmp_path)
    _user(store)
    client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    first = sent[-1][1]
    client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    second = sent[-1][1]
    assert first != second
    with sqlite3.connect(tmp_path / "r.sqlite3") as conn:
        stored = [r[0] for r in conn.execute("SELECT token_hmac FROM password_resets")]
    assert first not in stored and second not in stored, "tokens are never stored in clear"
    assert client.post("/api/v1/auth/password-reset/confirm", json={"token": first, "new_password": "BrandNewPassword#1"}).status_code == 400
    assert client.post("/api/v1/auth/password-reset/confirm", json={"token": second, "new_password": "BrandNewPassword#1"}).status_code == 200


def test_reset_requests_are_rate_limited_per_email(tmp_path) -> None:
    store, client, sent = _app(tmp_path)
    _user(store)
    for _ in range(6):
        client.post("/api/v1/auth/password-reset/request", json={"email": "alice@example.org"})
    assert len(sent) <= 3


class _FakeGoogle:
    client_id = "test-client.apps.googleusercontent.com"

    def __init__(self, subject: str = "google-subject-1") -> None:
        self.subject = subject

    def verify(self, token: str) -> OidcIdentity:
        if token != "valid-id-token":
            from hivemind.router.google_auth import GoogleAuthError

            raise GoogleAuthError("bad_signature")
        return OidcIdentity(subject=self.subject, issued_at=0.0, expires_at=1e12)


def test_google_link_requires_the_accounts_key_then_google_opens_that_account(tmp_path) -> None:
    store, client, _ = _app(tmp_path, google=_FakeGoogle())
    acc_id, key = _user(store)
    assert client.post("/api/v1/auth/link/google", json={"id_token": "valid-id-token"}).status_code == 401
    assert client.post("/api/v1/auth/link/google", json={"id_token": "forged"}, headers={"Authorization": f"Bearer {key}"}).status_code == 401
    ok = client.post("/api/v1/auth/link/google", json={"id_token": "valid-id-token"}, headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200
    signin = client.post("/v1/hivemind/register/google", json={"id_token": "valid-id-token"})
    assert signin.status_code == 200 and signin.json()["account_id"] == acc_id and signin.json()["created"] is False


def test_a_google_identity_cannot_be_linked_to_a_second_account(tmp_path) -> None:
    store, client, _ = _app(tmp_path, google=_FakeGoogle())
    _a, key_a = _user(store, "alice_v32", "alice@example.org")
    _b, key_b = _user(store, "bob_v32", "bob@example.org")
    assert client.post("/api/v1/auth/link/google", json={"id_token": "valid-id-token"}, headers={"Authorization": f"Bearer {key_a}"}).status_code == 200
    assert client.post("/api/v1/auth/link/google", json={"id_token": "valid-id-token"}, headers={"Authorization": f"Bearer {key_b}"}).status_code == 409
