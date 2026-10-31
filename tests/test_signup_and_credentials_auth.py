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

    # First login with legacy handle + password binds password and preserves 250.0 HMT balance
    r1 = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "quebec-tester-2147", "password": "QuebecSecretPassword"},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["account_id"] == legacy_id
    assert b1["balance_hmt"] == 250.0
    assert b1["api_key"].startswith("hm-")

    # Subsequent login with wrong password fails
    r_wrong = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "quebec-tester-2147", "password": "wrong"},
    )
    assert r_wrong.status_code == 401

    # Previous V1/V2 seeded user (e.g. @Vxtzq or callensxavier@gmail.com) restores seamlessly
    r_v1 = client.post(
        "/v1/hivemind/auth/login",
        json={"identifier": "@Vxtzq", "password": "vxtzq_password"},
    )
    assert r_v1.status_code == 200
    bv1 = r_v1.json()
    assert bv1["handle"] == "vxtzq"
    assert bv1["balance_hmt"] >= 30000.0
    assert bv1["api_key"].startswith("hm-")
