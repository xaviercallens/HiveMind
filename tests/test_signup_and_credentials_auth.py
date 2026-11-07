"""Tests for unified /signup & /login page and returning user password/credential authentication."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


SECRET = b"0123456789abcdef0123456789abcdef"


def test_signup_and_login_pages_render_both_login_and_register_forms() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    for route in ("/signup", "/login", "/register", "/connexion"):
        resp = client.get(route)
        assert resp.status_code == 200
        assert 'id="loginForm"' in resp.text
        assert 'id="registerForm"' in resp.text
        assert 'id="loginIdentifier"' in resp.text
        assert 'id="loginPassword"' in resp.text
        assert 'id="regHandle"' in resp.text
        assert 'id="regPassword"' in resp.text


def test_register_and_login_with_username_and_password_preserves_key_and_balance() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    # 1. Register new account with handle + password + email
    r_reg = client.post(
        "/v1/hivemind/auth/register",
        json={
            "handle": "alice_coder",
            "email": "alice@hivemind.network",
            "password": "MyStrongPassword!42",
            "role": "prosumer",
        },
    )
    assert r_reg.status_code == 200
    reg_body = r_reg.json()
    assert reg_body["created"] is True
    assert reg_body["handle"] == "alice_coder"
    assert reg_body["api_key"].startswith("hm-")
    assert reg_body["balance_hmt"] >= 100.0

    # 2. Log in with username + password -> returns same active key and balance
    r_login = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "alice_coder", "password": "MyStrongPassword!42"},
    )
    assert r_login.status_code == 200
    login_body = r_login.json()
    assert login_body["account_id"] == reg_body["account_id"]
    assert login_body["api_key"] == reg_body["api_key"]
    assert login_body["balance_hmt"] == reg_body["balance_hmt"]
    assert "v1_migrated" not in login_body
    assert "v2_access_granted" not in login_body

    # 3. Log in with email + password
    r_email = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "alice@hivemind.network", "password": "MyStrongPassword!42"},
    )
    assert r_email.status_code == 200
    assert r_email.json()["account_id"] == reg_body["account_id"]

    # 4. Wrong password is rejected with HTTP 401
    r_bad = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "alice_coder", "password": "WrongPassword999"},
    )
    assert r_bad.status_code == 401

    # 5. Direct login with existing hm-... API key works
    r_key = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": reg_body["api_key"]},
    )
    assert r_key.status_code == 200
    assert r_key.json()["account_id"] == reg_body["account_id"]


def test_legacy_handle_and_previous_v1_user_migration_login() -> None:
    store = RouterStore(":memory:", SECRET)
    # Pre-existing legacy handle registered before password_hash existed
    legacy_id, _ = store.register("quebec-tester-2147", "consumer")
    store.grant(legacy_id, 250.0, "legacy-balance")

    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    # GHSA-gjc6-74f9-wmrr: a handle-only account has no password to check, so a password login on its handle is refused
    # and binds nothing (it used to bind the caller's password and return the account's key to whoever asked first).
    r1 = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "quebec-tester-2147", "password": "QuebecSecretPassword"},
    )
    assert r1.status_code == 401
    assert "api_key" not in r1.text
    r_again = client.post("/v1/hivemind/auth/login", json={"identifier": "quebec-tester-2147", "password": "QuebecSecretPassword"})
    assert r_again.status_code == 401  # still refused: the first attempt did not bind the password
    assert store.balance(legacy_id) == 250.0

    # The owner of that account still signs in with the account's own API key.
    _legacy_id2, legacy_key = store.register("quebec-tester-key", "consumer")
    r_key = client.post("/v1/hivemind/auth/login", json={"identifier": legacy_key})
    assert r_key.status_code == 200
    assert r_key.json()["handle"] == "quebec-tester-key"

    # A seeded legacy handle that is not in the store is not created on login, and no starter HMT is granted.
    r_v1 = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "@Vxtzq", "password": "vxtzq_password"},
    )
    assert r_v1.status_code == 401
    assert store.get_account(store.account_id_for("vxtzq")) is None


def test_returning_user_restore_by_email_and_password_and_bearer_header_on_router() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    # 1. Register alice_coder with custom handle + email + password on /api/v1/auth/register
    r_reg = client.post(
        "/api/v1/auth/register",
        json={
            "handle": "alice_coder",
            "email": "alice@hivemind.network",
            "password": "MyStrongPassword!42",
            "role": "HiveMinder Contributor",
        },
    )
    assert r_reg.status_code == 200
    reg = r_reg.json()
    assert reg["status"] == "REGISTERED"
    assert reg["handle"] == "alice_coder"
    assert reg["token"].startswith("hm-")

    # 2. Check /api/v1/auth/session with Bearer token
    s_ok = client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {reg['token']}"})
    assert s_ok.status_code == 200
    assert s_ok.json()["authenticated"] is True
    assert s_ok.json()["user"]["handle"] == "alice_coder"

    # 3. Returning user submits email-only registration form (handle derived as 'alice' from email) + right password
    r_restore = client.post(
        "/api/v1/auth/register",
        json={
            "handle": "alice",
            "email": "alice@hivemind.network",
            "password": "MyStrongPassword!42",
        },
    )
    assert r_restore.status_code == 200
    restored = r_restore.json()
    assert restored["status"] == "EXISTING_USER_RESTORED"
    assert restored["created"] is False
    assert restored["account_id"] == reg["account_id"]
    assert restored["handle"] == "alice_coder"
    assert restored["token"] == reg["token"]

    # 4. Wrong password with existing email is rejected with 409
    r_conflict = client.post(
        "/api/v1/auth/register",
        json={
            "handle": "alice",
            "email": "alice@hivemind.network",
            "password": "WrongPassword999",
        },
    )
    assert r_conflict.status_code == 409

    # 5. Handle-only previous user can log in on /api/v1/auth/login using Authorization: Bearer hm-...
    _lid, legacy_key = store.register("quebec-tester-key", "prosumer")
    r_bearer_login = client.post(
        "/api/v1/auth/login",
        headers={"Authorization": f"Bearer {legacy_key}"},
        json={"handle_or_email": "quebec-tester-key"},
    )
    assert r_bearer_login.status_code == 200
    assert r_bearer_login.json()["handle"] == "quebec-tester-key"

    # 6. Another user's Bearer header cannot log into alice_coder without alice_coder's password
    r_cross = client.post(
        "/api/v1/auth/login",
        headers={"Authorization": f"Bearer {legacy_key}"},
        json={"handle_or_email": "alice_coder", "password": "WrongPassword"},
    )
    assert r_cross.status_code == 401

    # 7. E2E v3.1 write routes work with the authenticated router token
    hdr = {"Authorization": f"Bearer {reg['token']}"}
    pub = client.post("/api/v1/amcp/kafka/publish", headers=hdr, json={"message": "E2E router test"})
    assert pub.status_code == 200
    assert pub.json()["published_event"]["actor_handle"] == "@alice_coder"


def test_dashboard_returning_user_login_and_register_after_logout_and_rotation(tmp_path) -> None:
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    client = TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))

    # 1. Register bob_veteran with password
    r_reg = client.post(
        "/api/v1/auth/register",
        json={
            "handle": "bob_veteran",
            "email": "bob@hivemind.network",
            "password": "BobStrongPass#2026",
        },
    )
    assert r_reg.status_code == 200
    tok = r_reg.json()["token"]
    assert tok.startswith("hm_live_")

    # 2. Logout removes token from active sessions
    assert client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {tok}"}).status_code == 200
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {tok}"}).json()["authenticated"] is False
    assert client.post("/api/v1/amcp/kafka/publish", headers={"Authorization": f"Bearer {tok}"}, json={"message": "hi"}).status_code == 401

    # 3. Returning user can sign back in after logout using their hm_live_... token in handle_or_email
    r_tok_login = client.post("/api/v1/auth/login", json={"handle_or_email": tok})
    assert r_tok_login.status_code == 200
    assert r_tok_login.json()["handle"] == "bob_veteran"
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {tok}"}).json()["authenticated"] is True

    # 4. Logout again, then sign back in using handle + hm_live_... token in the password field
    client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {tok}"})
    r_pwd_tok = client.post("/api/v1/auth/login", json={"handle_or_email": "bob_veteran", "password": tok})
    assert r_pwd_tok.status_code == 200
    assert r_pwd_tok.json()["token"] == tok

    # 5. Logout again, then sign back in using email + password
    client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {tok}"})
    r_email_pwd = client.post("/api/v1/auth/login", json={"handle_or_email": "bob@hivemind.network", "password": "BobStrongPass#2026"})
    assert r_email_pwd.status_code == 200
    assert r_email_pwd.json()["handle"] == "bob_veteran"

    # 6. Rotate key -> old token is permanently revoked even for login, new token works
    r_rot = client.post("/api/v1/auth/rotate-key", headers={"Authorization": f"Bearer {tok}"})
    assert r_rot.status_code == 200
    new_tok = r_rot.json()["token"]
    assert new_tok != tok
    assert client.post("/api/v1/auth/login", json={"handle_or_email": tok}).status_code == 404
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "bob_veteran", "password": tok}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"handle_or_email": new_tok}).status_code == 200

    # 7. Recreating the app with the same vault_dir preserves bob_veteran's previous password and rotated token
    client_restarted = TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))
    r_restart_login = client_restarted.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "bob@hivemind.network", "password": "  BobStrongPass#2026  "},
    )
    assert r_restart_login.status_code == 200
    assert r_restart_login.json()["handle"] == "bob_veteran"
    assert r_restart_login.json()["token"] == new_tok


def test_old_user_previous_password_with_email_handle_collision_and_legacy_pbkdf2_and_credentials_table(tmp_path) -> None:
    import hashlib
    import sqlite3

    db_path = tmp_path / "router_legacy.db"
    store = RouterStore(db_path, SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    # User 1 has handle 'alice' with a different email and password
    r_u1 = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "alice", "email": "other_alice@domain.org", "password": "OtherAlicePassword!1"},
    )
    assert r_u1.status_code == 200

    # User 2 (old user) has handle 'alice_legacy_2025' and email 'alice@hivemind.network' (prefix 'alice' collides with User 1!)
    r_u2 = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "alice_legacy_2025", "email": "alice@hivemind.network", "password": "MyOld2025Password#99"},
    )
    assert r_u2.status_code == 200
    u2 = r_u2.json()

    # Old user logs in with their email + previous password (even with incidental whitespace) -> must resolve to User 2, not fail on User 1!
    r_login = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "alice@hivemind.network", "password": " MyOld2025Password#99 "},
    )
    assert r_login.status_code == 200
    assert r_login.json()["account_id"] == u2["account_id"]
    assert r_login.json()["handle"] == "alice_legacy_2025"

    # Also works via /api/v1/auth/login and /api/v1/auth/register restore
    r_v1_login = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "alice@hivemind.network", "password": "MyOld2025Password#99"},
    )
    assert r_v1_login.status_code == 200
    assert r_v1_login.json()["account_id"] == u2["account_id"]

    r_v1_reg = client.post(
        "/api/v1/auth/register",
        json={"handle": "alice", "email": "alice@hivemind.network", "password": "MyOld2025Password#99"},
    )
    assert r_v1_reg.status_code == 200
    assert r_v1_reg.json()["account_id"] == u2["account_id"]
    assert r_v1_reg.json()["status"] == "EXISTING_USER_RESTORED"

    # Test legacy SQLite `credentials` table (salt_hex + 200_000 PBKDF2-HMAC-SHA256) when accounts.password_hash is empty
    legacy_acc_id, _legacy_key = store.register("veteran_pbkdf2", "prosumer")
    salt_hex = "aabbccddeeff00112233445566778899"
    pwd_hash = hashlib.pbkdf2_hmac("sha256", b"VeteranPass2024!", bytes.fromhex(salt_hex), 200_000).hex()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS credentials (email_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, salt_hex TEXT NOT NULL, password_hash TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO credentials(email_hash, account_id, salt_hex, password_hash) VALUES (?, ?, ?, ?)",
            ("ehash1", legacy_acc_id, salt_hex, pwd_hash),
        )
        conn.commit()

    r_cred_login = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "veteran_pbkdf2", "password": "VeteranPass2024!"},
    )
    assert r_cred_login.status_code == 200
    assert r_cred_login.json()["account_id"] == legacy_acc_id


def test_dashboard_cross_service_openrouter_previous_password_login(monkeypatch, tmp_path) -> None:
    import io
    import json
    import urllib.request
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    monkeypatch.setenv("HIVEMIND_OPENROUTER_URL", "https://hivemind-openrouter.example.test")

    class _FakeResp:
        status = 200

        def __init__(self, payload: dict) -> None:
            self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=4.0):
        body = json.loads(req.data.decode("utf-8"))
        if body.get("handle_or_email") == "upstream_vet@hivemind.network" and body.get("password") == "UpstreamPass!2025":
            return _FakeResp(
                {
                    "ok": True,
                    "status": "AUTHENTICATED",
                    "handle": "upstream_vet",
                    "email": "upstream_vet@hivemind.network",
                    "role": "prosumer",
                    "token": "hm_live_upstream1234567890",
                    "balance": 1500,
                }
            )
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    client = TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))

    r_ok = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "upstream_vet@hivemind.network", "password": "UpstreamPass!2025"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["handle"] == "upstream_vet"
    assert r_ok.json()["token"] == "hm_live_upstream1234567890"

    r_bad = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "upstream_vet@hivemind.network", "password": "WrongPassword"},
    )
    assert r_bad.status_code == 401


def test_oct25_credentials_120k_pbkdf2_email_hash_and_https_api_base(monkeypatch, tmp_path) -> None:
    import hashlib
    import io
    import json
    import sqlite3
    import urllib.request
    from hivemind.boinc.hivemind_at_home_dashboard import (
        build_hivemind_at_home_state,
        create_hivemind_at_home_app,
        render_hivemind_at_home_html,
    )
    from hivemind.boinc.stitch_portal import render_stitch_enhanced_portal

    db_path = tmp_path / "router_oct25.db"
    store = RouterStore(db_path, SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    # 1. Oct 25 user registered with email_hash = hmac("email:<email>"), handle = "user.<email_hash[:16]>", 120_000 PBKDF2 iterations, and NULL accounts.email
    oct25_email = "pioneer_oct25@gmail.com"
    ehash = store._hmac_hex(f"email:{oct25_email}")
    oct25_acc_id, _ = store.register(f"user.{ehash[:16]}", "prosumer")
    salt_hex = "db75c3761836010dfa4bf470418b689d"
    pwd_120k = hashlib.pbkdf2_hmac("sha256", b"MyOct25PreviousPass!99", bytes.fromhex(salt_hex), 120_000).hex()

    # 2. Seeded/legacy user also has a newer accounts row with a different password_hash, plus an Oct 25 credentials row for their email
    r_seeded = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "xavier_callens", "email": "callensxavier@gmail.com", "password": "LaterE2ETestPassword!1"},
    )
    assert r_seeded.status_code == 200
    xav_ehash = store._hmac_hex("email:callensxavier@gmail.com")
    xav_oct25_acc_id, _ = store.register(f"user.{xav_ehash[:16]}", "prosumer")
    xav_pwd_120k = hashlib.pbkdf2_hmac("sha256", b"XavierOct25RealPass#42", bytes.fromhex(salt_hex), 120_000).hex()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS credentials (email_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, salt_hex TEXT NOT NULL, password_hash TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO credentials(email_hash, account_id, salt_hex, password_hash) VALUES (?, ?, ?, ?)",
            (ehash, oct25_acc_id, salt_hex, pwd_120k),
        )
        conn.execute(
            "INSERT INTO credentials(email_hash, account_id, salt_hex, password_hash) VALUES (?, ?, ?, ?)",
            (xav_ehash, xav_oct25_acc_id, salt_hex, xav_pwd_120k),
        )
        conn.commit()

    # Both Oct 25 email login and xavier_callens Oct 25 password login succeed!
    r_oct25_login = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": oct25_email, "password": "MyOct25PreviousPass!99"},
    )
    assert r_oct25_login.status_code == 200
    assert r_oct25_login.json()["account_id"] == oct25_acc_id

    r_xav_login = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "callensxavier@gmail.com", "password": "XavierOct25RealPass#42"},
    )
    assert r_xav_login.status_code == 200
    assert r_xav_login.json()["handle"] == "xavier_callens"

    r_xav_reg = client.post(
        "/api/v1/auth/register",
        json={"handle": "xavier_callens", "email": "callensxavier@gmail.com", "password": "XavierOct25RealPass#42"},
    )
    assert r_xav_reg.status_code == 200
    assert r_xav_reg.json()["status"] == "EXISTING_USER_RESTORED"

    # 3. Verify Cloud Run reverse-proxy http:// base_url is upgraded to https:// for non-local hosts to prevent Mixed Content blocking
    state = build_hivemind_at_home_state()
    stitch_html = render_stitch_enhanced_portal(
        state,
        api_base_url="http://hivemind-at-home-1003063861791.us-central1.run.app",
    )
    assert '"https://hivemind-at-home-1003063861791.us-central1.run.app"' in stitch_html
    assert '"http://hivemind-at-home-1003063861791.us-central1.run.app"' not in stitch_html

    legacy_html = render_hivemind_at_home_html(
        state,
        api_base_url="http://hivemind-at-home-1003063861791.us-central1.run.app",
    )
    assert '"https://hivemind-at-home-1003063861791.us-central1.run.app"' in legacy_html
    assert '"http://hivemind-at-home-1003063861791.us-central1.run.app"' not in legacy_html




