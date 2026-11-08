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


def test_comprehensive_sso_pkce_s256_ownership_and_persistence_dashboard(tmp_path: Path, monkeypatch) -> None:
    """Comprehensive E2E test for SSO PKCE S256, lookalike protection, token restoration, and persistence on Dashboard."""
    monkeypatch.setenv("HIVEMIND_LEGACY_SANDBOX_SSO", "1")  # exercises the retired sandbox SSO, off by default since v3.2
    import base64
    import hashlib
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    users_file = tmp_path / "dashboard_sso_users.json"
    monkeypatch.setenv("HIVEMIND_DASHBOARD_USERS_FILE", str(users_file))

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. PKCE S256 with client-supplied code_challenge
    verifier = "my_high_entropy_pkce_verifier_secret_1234567890"
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    r_az = client.get(f"/api/v1/auth/sso/google/authorize?code_challenge={challenge}")
    assert r_az.status_code == 200
    az_data = r_az.json()
    assert az_data["code_challenge"] == challenge
    assert az_data["federation_mode"] == "LOCAL_PKCE_SANDBOX_FEDERATION"

    # Cross-provider state replay (google state on github) must be rejected with 400
    r_cross = client.post(
        "/api/v1/auth/sso/github",
        json={"state": az_data["state"], "code_verifier": verifier, "handle_hint": "pkce_tester"},
    )
    assert r_cross.status_code == 400

    # Wrong code_verifier must be rejected with 400
    r_bad_verifier = client.post(
        "/api/v1/auth/sso/google",
        json={"state": az_data["state"], "code_verifier": "wrong_verifier_value", "handle_hint": "pkce_tester"},
    )
    assert r_bad_verifier.status_code == 400

    # Correct code_verifier succeeds and creates account
    r_sso_ok = client.post(
        "/api/v1/auth/sso/google",
        json={"state": az_data["state"], "code_verifier": verifier, "handle_hint": "pkce_tester"},
    )
    assert r_sso_ok.status_code == 200
    sso_user = r_sso_ok.json()
    assert sso_user["status"] == "AUTHENTICATED_VIA_SSO"
    assert sso_user["handle"] == "pkce_tester"
    assert sso_user["provider"] == "GOOGLE_SSO"
    sso_tok = sso_user["token"]

    # 2. Lookalike handles/emails on SSO cannot hijack xavier_callens or pkce_tester
    for lookalike in ("@xavier_callens", "Xavier_Callens", "xaviercallens", "callensxavier@gmail.com", "@pkce_tester", "PKCE_TESTER"):
        r_hijack = client.post(
            "/api/v1/auth/sso/reddit",
            json={"handle_hint": lookalike, "user_mode": "auto"},
        )
        assert r_hijack.status_code == 401, f"Expected 401 for lookalike {lookalike}, got {r_hijack.status_code}"
        r_force_new = client.post(
            "/api/v1/auth/sso/reddit",
            json={"handle_hint": lookalike, "user_mode": "force_new"},
        )
        assert r_force_new.status_code == 409, f"Expected 409 for force_new lookalike {lookalike}, got {r_force_new.status_code}"

    # 3. Returning user with valid Bearer token can log back in via SSO even after logout and without typing handle_hint
    client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {sso_tok}"})
    r_sso_return = client.post(
        "/api/v1/auth/sso/github",
        headers={"Authorization": f"Bearer {sso_tok}"},
        json={"user_mode": "auto"},
    )
    assert r_sso_return.status_code == 200
    ret_body = r_sso_return.json()
    assert ret_body["handle"] == "pkce_tester"
    assert ret_body["user_state"] == "EXISTING_USER"
    assert "GITHUB_SSO" in ret_body["linked_providers"]
    assert "GOOGLE_SSO" in ret_body["linked_providers"]

    # 4. Restart dashboard app and verify SSO user persisted to disk
    app2 = create_hivemind_at_home_app()
    client2 = TestClient(app2)
    r_sess2 = client2.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {sso_tok}"})
    assert r_sess2.status_code == 200
    assert r_sess2.json()["authenticated"] is True
    assert r_sess2.json()["user"]["handle"] == "pkce_tester"

    # 5. Guest upgrade lookalike protection and 0 unverified bonus
    r_guest = client2.post("/api/v1/auth/guest", json={"target_profile": "developer"})
    assert r_guest.status_code == 200
    g_tok = r_guest.json()["token"]
    assert r_guest.json()["compute_points"] == 0

    r_up_hijack = client2.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {g_tok}"},
        json={"handle": "@xavier_callens", "email": "callensxavier@gmail.com", "provider": "google"},
    )
    assert r_up_hijack.status_code == 409

    r_up_ok = client2.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {g_tok}"},
        json={"handle": "honest_upgraded_guest", "email": "honest@example.com", "provider": "github"},
    )
    assert r_up_ok.status_code == 200
    assert r_up_ok.json()["compute_points"] == 0
    assert r_up_ok.json()["handle"] == "honest_upgraded_guest"


def test_comprehensive_sso_guest_link_and_key_rotation_on_router_gateway(tmp_path: Path) -> None:
    """Verify full parity of /api/v1/auth/sso/*, /guest, /upgrade-guest, /link-provider, and /rotate-key on Router Gateway."""
    import base64
    import hashlib

    store = RouterStore(tmp_path / "router_sso_parity.sqlite3", SECRET)
    app = create_router_app(store, open_registration=True)
    client = TestClient(app)

    # 1. Test all 4 core MVP SSO providers on Router Gateway
    for prov in ("google", "reddit", "steam", "github"):
        r_az = client.get(f"/api/v1/auth/sso/{prov}/authorize")
        assert r_az.status_code == 200
        az = r_az.json()
        assert az["status"] == "SSO_AUTHORIZATION_READY"
        assert az["is_core_mvp_provider"] is True

        r_sso = client.post(
            f"/api/v1/auth/sso/{prov}",
            json={"state": az["state"], "code_verifier": az["code_verifier"], "handle_hint": f"router_{prov}_user"},
        )
        assert r_sso.status_code == 200
        body = r_sso.json()
        assert body["status"] == "AUTHENTICATED_VIA_SSO"
        assert body["handle"] == f"router_{prov}_user"
        assert body["token"].startswith("hm-")

    # 2. Enforce PKCE S256 challenge and cross-provider state check on Router Gateway
    verifier = "router_pkce_secret_verifier_987654321"
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    az_custom = client.get(f"/api/v1/auth/sso/google/authorize?code_challenge={challenge}").json()
    assert (
        client.post(
            "/api/v1/auth/sso/steam",
            json={"state": az_custom["state"], "code_verifier": verifier},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/sso/google",
            json={"state": az_custom["state"], "code_verifier": "bad_verifier"},
        ).status_code
        == 400
    )
    r_pkce_ok = client.post(
        "/api/v1/auth/sso/google",
        json={"state": az_custom["state"], "code_verifier": verifier, "handle_hint": "router_pkce_valid"},
    )
    assert r_pkce_ok.status_code == 200
    pkce_tok = r_pkce_ok.json()["token"]

    # 3. Lookalike handle protection on Router Gateway SSO
    for lookalike in ("@xavier_callens", "callensxavier@gmail.com", "router_pkce_valid", "@router_pkce_valid"):
        assert (
            client.post("/api/v1/auth/sso/google", json={"handle_hint": lookalike, "user_mode": "auto"}).status_code
            == 401
        )
        assert (
            client.post("/api/v1/auth/sso/google", json={"handle_hint": lookalike, "user_mode": "force_new"}).status_code
            == 409
        )

    # 4. Link provider & Rotate key on Router Gateway
    r_link = client.post(
        "/api/v1/auth/link-provider",
        headers={"Authorization": f"Bearer {pkce_tok}"},
        json={"provider": "github"},
    )
    assert r_link.status_code == 200
    assert "GITHUB_SSO" in r_link.json()["linked_providers"]

    r_rot = client.post("/api/v1/auth/rotate-key", headers={"Authorization": f"Bearer {pkce_tok}"})
    assert r_rot.status_code == 200
    rot_data = r_rot.json()
    assert rot_data["old_token_revoked"] is True
    new_tok = rot_data["token"]
    assert new_tok != pkce_tok

    # Old token is revoked, new token authenticates
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {pkce_tok}"}).json()["authenticated"] is False
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {new_tok}"}).json()["authenticated"] is True

    # 5. Guest sandbox and upgrade on Router Gateway
    r_guest = client.post("/api/v1/auth/guest", json={"target_profile": "gamer"})
    assert r_guest.status_code == 200
    g_tok = r_guest.json()["token"]
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {g_tok}"}).json()["user"]["is_guest"] is True

    r_up = client.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {g_tok}"},
        json={"handle": "router_guest_upgraded", "email": "rg@example.com", "provider": "steam"},
    )
    assert r_up.status_code == 200
    assert r_up.json()["compute_points"] == 0
    assert r_up.json()["token"].startswith("hm-")


def test_web_ui_honest_claims_and_simplified_auth_drawer(tmp_path: Path) -> None:
    """Verify Web UI pages contain only real claims and simplified auth layout."""
    store = RouterStore(tmp_path / "ui_honesty.sqlite3", SECRET)
    app = create_router_app(store, open_registration=True)
    client = TestClient(app)

    for path in ("/signup", "/login"):
        html = client.get(path).text
        assert "sets password on first login" not in html
        assert "optional if using API key or legacy handle" not in html
        assert "required for password accounts" in html

    portal_html = client.get("/v3.1").text
    assert "ENTER AS GUEST (2,500 HMT)" not in portal_html
    assert "(+2,500 HMT BONUS)" not in portal_html
    assert "PASSWORDLESS QUANTUM MAGIC LINK" not in portal_html
    assert "ENTER AS GUEST (0 HMT SANDBOX)" in portal_html
    assert "(0 UNVERIFIED BONUS)" in portal_html
    assert "LOCAL EMAIL TOKEN HANDSHAKE (NO OUTBOUND EMAIL SENT)" in portal_html
    assert 'data-testid="advanced-legacy-auth-drawer"' in portal_html







def test_webui_v3_1_2_unified_session_bridge_producer_strip_and_honest_badges(tmp_path: Path) -> None:
    """Verify v3.1.2 cross-page session storage bridge, live producer pool strip, quick session bar, and non-monetary badges."""
    store = RouterStore(tmp_path / "ui_v312.sqlite3", SECRET)
    app = create_router_app(store, open_registration=True)
    client = TestClient(app)

    for path in ("/signup", "/login"):
        html = client.get(path).text
        assert 'data-testid="signup-quick-nav"' in html
        assert 'data-testid="existing-browser-session-banner"' in html
        assert 'data-testid="btn-open-v31-authenticated"' in html
        assert 'data-testid="signup-rotate-key-btn"' in html
        assert 'data-testid="signup-signout-btn"' in html
        assert "persistUnifiedSession" in html
        assert "checkActiveBrowserSession" in html

    for portal_path in ("/v3", "/v3.1"):
        portal_html = client.get(portal_path).text
        # Honest non-monetary reciprocity labels
        assert "+$50 BONUS" not in portal_html
        assert "500k Free Token Allotment" not in portal_html
        assert "UNLIMITED OPEN" not in portal_html
        assert "OPENROUTER + AMCP API" in portal_html
        assert "OpenAI-Compatible /v1/chat/completions" in portal_html
        assert "RECIPROCITY HMT LEDGER" in portal_html
        # Live Volunteer GPU Producer Pool strip on home screen
        assert 'data-testid="live-producer-pool-strip"' in portal_html
        assert 'data-testid="strip-producer-count"' in portal_html
        assert 'data-testid="strip-active-models"' in portal_html
        assert 'data-testid="btn-refresh-producer-strip"' in portal_html
        assert 'data-testid="btn-strip-connect-byob"' in portal_html
        # Quick Active Session Summary bar on auth screen
        assert 'data-testid="quick-session-summary-bar"' in portal_html
        assert 'data-testid="quick-copy-key-btn"' in portal_html
        assert 'data-testid="quick-rotate-key-btn"' in portal_html
        assert "hydrateSessionFromApiKey" in portal_html
        assert "fetchLiveProducerPoolStrip" in portal_html
