"""The gateway's password check accepts only evidence bound to the account being opened.

Regression tests for three account-takeover paths and for the legacy formats that previous users must keep:
- no account accepts an arbitrary password, whatever its stored hash (placeholder, empty or unknown), and a failed
  check never rewrites the stored hash;
- a legacy user's password never opens another legacy account (no table-wide credentials fallback);
- a caller-chosen e-mail never selects the credentials used to open an account (register and login);
- genuine passwords in every stored format still work: salted PBKDF2 at 100k, 120k and 200k iterations, the
  secret-salted handle hash, and legacy ``credentials`` rows tied to the account id or to the account's own e-mail;
- ``/api/v1/auth/change-password`` sets a real password only for the holder of the account's hm-... key.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

SECRET = b"0123456789abcdef0123456789abcdef"
PLACEHOLDER_HASHES = (
    "e3746644bb071aff7215888e5d4c6832c6a7035d43b286a921c2e75255699747",
    "ad8f4691fca5a1bd3760219751ef208fde7599e24a01db30ba4b1834df08009a",
    "",
)
CREDENTIALS_DDL = (
    "CREATE TABLE IF NOT EXISTS credentials (email_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, "
    "salt_hex TEXT NOT NULL, password_hash TEXT NOT NULL)"
)


def _setup(tmp_path):
    db = tmp_path / "router.sqlite3"
    store = RouterStore(db, SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))
    return db, store, client


def _stored_hash(db, account_id: str) -> str:
    with sqlite3.connect(db) as conn:
        return str(conn.execute("SELECT password_hash FROM accounts WHERE account_id=?", (account_id,)).fetchone()[0] or "")


def _login(client, ident: str, password: str):
    return client.post("/api/v1/auth/login", json={"handle_or_email": ident, "password": password})


@pytest.mark.parametrize("placeholder", PLACEHOLDER_HASHES)
@pytest.mark.parametrize("ident", ["xavier_callens", "callensxavier@gmail.com"])
def test_owner_account_never_accepts_an_arbitrary_password(tmp_path, placeholder: str, ident: str) -> None:
    db, store, client = _setup(tmp_path)
    acc_id, _ = store.register("xavier_callens", "prosumer")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=?, password_hash=? WHERE account_id=?", ("callensxavier@gmail.com", placeholder or None, acc_id))
        conn.commit()
    before = _stored_hash(db, acc_id)
    for guess in ("AnyPassword1", "attacker-chosen-pass", "12345678"):
        r = _login(client, ident, guess)
        assert r.status_code == 401, (ident, guess, r.status_code)
        body = r.json()
        assert "api_key" not in body and "token" not in body and "account_id" not in body
        assert not re.search(r"hm-[A-Za-z0-9_-]{20,}", r.text), "no key value may be returned"
    assert _stored_hash(db, acc_id) == before, "a failed check must not rewrite the stored hash"


def _legacy_credentials(db, store, *pairs) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(CREDENTIALS_DDL)
        for email, account_id, password, iters in pairs:
            salt = "db75c3761836010dfa4bf470418b689d"
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iters).hex()
            conn.execute(
                "INSERT INTO credentials(email_hash, account_id, salt_hex, password_hash) VALUES (?,?,?,?)",
                (store._hmac_hex(f"email:{email}"), account_id, salt, digest),
            )
        conn.commit()


def test_one_legacy_users_password_never_opens_another_legacy_account(tmp_path) -> None:
    db, store, client = _setup(tmp_path)
    owner_id, _ = store.register("xavier_callens", "prosumer")
    other_id, _ = store.register("vxtzq", "prosumer")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=? WHERE account_id=?", ("callensxavier@gmail.com", owner_id))
        conn.execute("UPDATE accounts SET email=? WHERE account_id=?", ("vxtzq@github.com", other_id))
        conn.commit()
    _legacy_credentials(db, store, ("vxtzq@github.com", other_id, "VxtzqOwnSecret#1", 120_000))
    # the other legacy user can open their own account ...
    assert _login(client, "vxtzq@github.com", "VxtzqOwnSecret#1").status_code == 200
    # ... but not the owner's, by handle or by e-mail
    assert _login(client, "xavier_callens", "VxtzqOwnSecret#1").status_code == 401
    assert _login(client, "callensxavier@gmail.com", "VxtzqOwnSecret#1").status_code == 401


def test_caller_chosen_email_never_selects_the_credentials_of_another_account(tmp_path) -> None:
    db, store, client = _setup(tmp_path)
    owner_id, _ = store.register("xavier_callens", "prosumer")
    attacker_id, _ = store.register("attacker", "prosumer")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=? WHERE account_id=?", ("callensxavier@gmail.com", owner_id))
        conn.execute("UPDATE accounts SET email=? WHERE account_id=?", ("attacker@example.org", attacker_id))
        conn.commit()
    _legacy_credentials(db, store, ("attacker@example.org", attacker_id, "AttackerKnows#123", 200_000))
    r = client.post(
        "/api/v1/auth/register",
        json={"handle": "xavier_callens", "email": "attacker@example.org", "password": "AttackerKnows#123"},
    )
    assert r.status_code != 200 or r.json().get("account_id") != owner_id
    assert owner_id not in r.text


@pytest.mark.parametrize("iters", [100_000, 120_000, 200_000])
def test_previous_users_keep_their_salted_pbkdf2_password(tmp_path, iters: int) -> None:
    db, store, client = _setup(tmp_path)
    acc_id, _ = store.register("previous_user", "prosumer")
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    digest = hashlib.pbkdf2_hmac("sha256", b"MyPreviousPass#2025", salt, iters).hex()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=?, password_hash=? WHERE account_id=?", ("prev@example.org", f"{salt.hex()}:{digest}", acc_id))
        conn.commit()
    for ident in ("previous_user", "prev@example.org"):
        r = _login(client, ident, "MyPreviousPass#2025")
        assert r.status_code == 200 and r.json()["account_id"] == acc_id
    assert _login(client, "previous_user", "WrongPass#2025").status_code == 401


def test_previous_users_keep_their_secret_salted_handle_password(tmp_path) -> None:
    _db, store, client = _setup(tmp_path)
    acc_id, _key, created = store.register_with_credentials("handle_user", "HandleUserPass#77", email="handle@example.org")
    assert created
    r = _login(client, "handle@example.org", "HandleUserPass#77")
    assert r.status_code == 200 and r.json()["account_id"] == acc_id


def test_previous_users_keep_their_legacy_credentials_row(tmp_path) -> None:
    db, store, client = _setup(tmp_path)
    by_id, _ = store.register("legacy_by_id", "prosumer")
    by_email, _ = store.register("legacy_by_email", "prosumer")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=? WHERE account_id=?", ("own@example.org", by_email))
        conn.commit()
    _legacy_credentials(
        db, store,
        ("unrelated-key@example.org", by_id, "ById#Password1", 200_000),
        ("own@example.org", by_email, "ByEmail#Password1", 120_000),
    )
    assert _login(client, "legacy_by_id", "ById#Password1").status_code == 200
    assert _login(client, "own@example.org", "ByEmail#Password1").status_code == 200


def test_change_password_requires_the_accounts_own_key_and_then_works(tmp_path) -> None:
    db, store, client = _setup(tmp_path)
    acc_id, key = store.register("xavier_callens", "prosumer")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=?, password_hash=? WHERE account_id=?", ("callensxavier@gmail.com", PLACEHOLDER_HASHES[0], acc_id))
        conn.commit()
    assert client.post("/api/v1/auth/change-password", json={"new_password": "ANewStrongPassword#1"}).status_code == 401
    assert client.post(
        "/api/v1/auth/change-password",
        json={"new_password": "ANewStrongPassword#1"},
        headers={"Authorization": "Bearer hm-not-a-real-key"},
    ).status_code == 401
    weak = client.post("/api/v1/auth/change-password", json={"new_password": "short"}, headers={"Authorization": f"Bearer {key}"})
    assert weak.status_code == 400
    ok = client.post("/api/v1/auth/change-password", json={"new_password": "ANewStrongPassword#1"}, headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200 and ok.json()["status"] == "PASSWORD_SET"
    assert _login(client, "callensxavier@gmail.com", "ANewStrongPassword#1").status_code == 200
    assert _login(client, "callensxavier@gmail.com", "AnyPassword1").status_code == 401
