"""Property tests ("never again") for the gateway password check, the class of bug that commit 316c106 shipped.

For random accounts stored in every supported format and random wrong passwords:
- the right password opens the account; any other password never does;
- a failed check never changes anything stored (password hash, legacy credentials);
- a successful sign-in with a legacy format upgrades the stored hash to the modern format, and the account still opens;
- another account's password never opens this account, whatever the two storage formats.

Profiles: ``hivemind-ci`` (deterministic, used by the required CI job) and ``hivemind-intense`` (``HYPOTHESIS_PROFILE``
environment variable; many random examples, run by the web-auth intense workflow and nightly).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import string
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from hivemind.router.accounts import MODERN_HASH_PREFIX, InvalidCredentials, RouterStore, modern_password_hash


@pytest.fixture(autouse=True)
def _fast_modern_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The properties are about which evidence is accepted, not the work factor: hash at 120k (still >= the 100k floor)
    so the deterministic profile fits the required CI job. Production keeps MODERN_HASH_ITERATIONS = 600,000."""
    import hivemind.router.accounts as accounts

    monkeypatch.setattr(accounts, "MODERN_HASH_ITERATIONS", 120_000)

settings.register_profile("hivemind-ci", derandomize=True, deadline=None, max_examples=6,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
settings.register_profile("hivemind-intense", deadline=None, max_examples=300,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
PROFILE = settings.get_profile(os.environ.get("HYPOTHESIS_PROFILE", "hivemind-ci") if os.environ.get("HYPOTHESIS_PROFILE", "").startswith("hivemind-") else "hivemind-ci")

SECRET = b"0123456789abcdef0123456789abcdef"
FORMATS = ("modern", "salted_100k", "salted_120k", "salted_200k", "handle_hash", "cred_by_id", "cred_by_own_email")
PASSWORDS = st.text(alphabet=string.ascii_letters + string.digits + "#!@%&*-_+=", min_size=8, max_size=24)
CRED_DDL = ("CREATE TABLE IF NOT EXISTS credentials (email_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, "
            "salt_hex TEXT NOT NULL, password_hash TEXT NOT NULL)")


def _store(dirpath: str) -> tuple[RouterStore, Path]:
    db = Path(dirpath) / "r.sqlite3"
    return RouterStore(db, SECRET), db


def _make(store: RouterStore, db: Path, handle: str, email: str, password: str, fmt: str) -> str:
    acc_id, _ = store.register(handle, "prosumer")
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    stored: str | None
    if fmt == "modern":
        stored = modern_password_hash(password)
    elif fmt.startswith("salted_"):
        iters = int(fmt.split("_")[1][:-1]) * 1000
        stored = f"{salt.hex()}:{hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iters).hex()}"
    elif fmt == "handle_hash":
        stored = store._hash_password(handle, password)
    else:
        stored = None
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE accounts SET email=?, password_hash=? WHERE account_id=?", (email, stored, acc_id))
        if fmt.startswith("cred_"):
            conn.execute(CRED_DDL)
            key = store._hmac_hex(f"email:{email}") if fmt == "cred_by_own_email" else store._hmac_hex(f"email:unrelated-{handle}@x.org")
            conn.execute("INSERT INTO credentials VALUES (?,?,?,?)",
                         (key, acc_id, salt.hex(), hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()))
        conn.commit()
    return acc_id


def _snapshot(db: Path) -> tuple:
    with sqlite3.connect(db) as conn:
        accounts = tuple(conn.execute("SELECT account_id, password_hash, key_hash FROM accounts ORDER BY account_id"))
        try:
            creds = tuple(conn.execute("SELECT * FROM credentials ORDER BY email_hash"))
        except sqlite3.OperationalError:
            creds = ()
    return accounts, creds


def _opens(store: RouterStore, ident: str, password: str) -> str | None:
    try:
        acct, _key, _ = store.login_with_credentials(ident, password)
        return acct.account_id
    except (InvalidCredentials, ValueError):
        return None


@PROFILE
@given(right=PASSWORDS, wrong=PASSWORDS, fmt=st.sampled_from(FORMATS))
def test_only_the_right_password_opens_and_failures_change_nothing(right: str, wrong: str, fmt: str) -> None:
    assume(right != wrong)
    with tempfile.TemporaryDirectory() as d:
        store, db = _store(d)
        acc_id = _make(store, db, "prop_user", "prop@example.org", right, fmt)
        before = _snapshot(db)
        for ident in ("prop_user", "prop@example.org"):
            assert _opens(store, ident, wrong) is None
        assert _snapshot(db) == before, "a failed check must not change anything stored"
        assert _opens(store, "prop@example.org", right) == acc_id


@PROFILE
@given(right=PASSWORDS, fmt=st.sampled_from(FORMATS))
def test_a_legacy_sign_in_upgrades_the_hash_and_the_account_still_opens(right: str, fmt: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        store, db = _store(d)
        acc_id = _make(store, db, "legacy_user", "legacy@example.org", right, fmt)
        assert _opens(store, "legacy@example.org", right) == acc_id
        with sqlite3.connect(db) as conn:
            stored = conn.execute("SELECT password_hash FROM accounts WHERE account_id=?", (acc_id,)).fetchone()[0]
        assert str(stored).startswith(MODERN_HASH_PREFIX)
        assert _opens(store, "legacy_user", right) == acc_id
        assert _opens(store, "legacy_user", right + "x") is None


@PROFILE
@given(pa=PASSWORDS, pb=PASSWORDS, fa=st.sampled_from(FORMATS), fb=st.sampled_from(FORMATS),
       a_handle=st.sampled_from(["xavier_callens", "plain_a"]), b_handle=st.sampled_from(["vxtzq", "plain_b"]))
def test_another_accounts_password_never_opens_this_account(pa: str, pb: str, fa: str, fb: str, a_handle: str, b_handle: str) -> None:
    assume(pa != pb)
    with tempfile.TemporaryDirectory() as d:
        store, db = _store(d)
        a_email = "callensxavier@gmail.com" if a_handle == "xavier_callens" else "a@example.org"
        b_email = "vxtzq@github.com" if b_handle == "vxtzq" else "b@example.org"
        a_id = _make(store, db, a_handle, a_email, pa, fa)
        b_id = _make(store, db, b_handle, b_email, pb, fb)
        assert _opens(store, a_email, pb) is None and _opens(store, a_handle, pb) is None
        assert _opens(store, b_email, pa) is None
        assert _opens(store, a_email, pa) == a_id and _opens(store, b_email, pb) == b_id


@pytest.mark.parametrize("stored", ["pbkdf2_sha256$1$00$aa", "pbkdf2_sha256$99999999$0011$aa", "pbkdf2_sha256$x$y$z", "pbkdf2_sha256$600000$$"])
def test_malformed_or_abusive_modern_hashes_never_match(stored: str) -> None:
    from hivemind.router.accounts import verify_modern_password_hash

    assert verify_modern_password_hash(stored, "anything") is False


def test_change_password_rotates_the_key(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "r.sqlite3", SECRET)
    client = TestClient(create_router_app(store))
    _acc, key = store.register("rotate_me", "prosumer")
    r = client.post("/api/v1/auth/change-password", json={"new_password": "RotateMePassword#1"}, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200 and r.json()["old_key_revoked"] is True
    new_key = r.json()["api_key"]
    assert store.authenticate(key) is None and store.authenticate(new_key) is not None
    with sqlite3.connect(tmp_path / "r.sqlite3") as conn:
        assert str(conn.execute("SELECT password_hash FROM accounts WHERE handle='rotate_me'").fetchone()[0]).startswith(MODERN_HASH_PREFIX)


def test_production_work_factor_is_600k() -> None:
    """Read from the source (the autouse fixture lowers the module value for speed; reloading the module would replace its
    classes under other modules' feet)."""
    src = (Path(__file__).resolve().parents[1] / "src" / "hivemind" / "router" / "accounts.py").read_text(encoding="utf-8")
    assert "\nMODERN_HASH_ITERATIONS = 600_000" in src
