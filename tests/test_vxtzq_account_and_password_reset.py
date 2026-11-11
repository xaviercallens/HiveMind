# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for Vxtzq account provisioning, email aliases, HMT quota restoration, and functional password reset."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.signup_page import render_signup_page

SECRET = b"hivemind-test-secret-at-least-32-bytes-long!"
ADMIN_KEY = "admin-secret-key-for-vxtzq-reset-test"


def _router_client(store: RouterStore) -> TestClient:
    app = create_router_app(
        store,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        starter_grant_hmt=0.0,
    )
    return TestClient(app)


@pytest.mark.parametrize(
    "ident",
    [
        "vxtzq",
        "@vxtzq",
        "vxtzq1",
        "@vxtzq1",
        "vxtzq@github.com",
        "116555319+vxtzq@users.noreply.github.com",
        "vxtzq@crowdgpt.net",
        "contact@crowdgpt.net",
        "vxtzq@hivemind.network",
    ],
)
def test_vxtzq_aliases_and_emails_resolve_to_vxtzq(ident: str) -> None:
    store = RouterStore(":memory:", SECRET)
    resolved = store._resolve_legacy_user(ident)
    assert resolved is not None
    assert resolved["handle"] == "vxtzq"
    assert resolved["starter_hmt"] == 30000.0


def test_ensure_legacy_hiveminders_tops_up_existing_depleted_vxtzq_account() -> None:
    store = RouterStore(":memory:", SECRET)
    aid, _key = store.register("vxtzq", "prosumer")
    assert store.balance(aid) == 0.0

    store.ensure_legacy_hiveminders_registered()
    assert store.balance(aid) == pytest.approx(30000.0)
    # reset mail goes only to the primary address; aliases are sign-in names, never reset destinations
    assert store.account_id_for_email("vxtzq@github.com") == aid
    for email_alias in (
        "116555319+vxtzq@users.noreply.github.com",
        "vxtzq@crowdgpt.net",
        "contact@crowdgpt.net",
        "vxtzq@hivemind.network",
    ):
        assert store.account_id_for_email(email_alias) is None


def test_admin_reset_password_provisions_vxtzq_and_enables_login_and_self_service_reset() -> None:
    store = RouterStore(":memory:", SECRET)
    client = _router_client(store)

    # Without admin key -> 403
    unauth = client.post(
        "/v1/hivemind/admin/accounts/reset-password",
        json={"handle": "vxtzq", "new_password": "VxtzqInitialPassword!2026"},
    )
    assert unauth.status_code == 403

    # Weak password -> 400
    weak = client.post(
        "/v1/hivemind/admin/accounts/reset-password",
        headers={"x-admin-key": ADMIN_KEY},
        json={"handle": "vxtzq", "new_password": "short"},
    )
    assert weak.status_code == 400

    # Valid admin reset & provision for vxtzq
    res = client.post(
        "/v1/hivemind/admin/accounts/reset-password",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "handle": "vxtzq",
            "email": "vxtzq@github.com",
            "new_password": "VxtzqInitialPassword!2026",
            "min_balance_hmt": 30000.0,
            "issue_reset_token": True,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["handle"] == "vxtzq"
    assert body["email"] == "vxtzq@github.com"
    assert "116555319+vxtzq@users.noreply.github.com" in body["email_aliases"]
    assert "vxtzq@crowdgpt.net" in body["email_aliases"]
    assert body["balance"] == pytest.approx(30000.0)
    assert body["api_key"].startswith("hm-")
    assert body["reset_token"]
    assert "/auth/reset#token=" in body["reset_url"]

    initial_key = body["api_key"]
    reset_token = body["reset_token"]

    # Direct API key check on /v1/hivemind/me works with 30,000 HMT
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {initial_key}"})
    assert me.status_code == 200
    assert me.json()["handle"] == "vxtzq"
    assert me.json()["balance"] == pytest.approx(30000.0)

    # Login with handle and email aliases works with the initial password
    for ident in (
        "vxtzq",
        "vxtzq@github.com",
        "116555319+vxtzq@users.noreply.github.com",
        "vxtzq@crowdgpt.net",
    ):
        login_res = client.post(
            "/api/v1/auth/login",
            json={"handle_or_email": ident, "password": "VxtzqInitialPassword!2026"},
        )
        assert login_res.status_code == 200, (ident, login_res.text)
        assert login_res.json()["handle"] == "vxtzq"
        assert login_res.json()["balance"] == pytest.approx(30000.0)

    # Self-service password reset using the single-use reset_token works and rotates the API key
    confirm = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": reset_token, "new_password": "VxtzqChosenPassword#2026"},
    )
    assert confirm.status_code == 200
    assert confirm.json()["status"] == "PASSWORD_RESET"

    # Old password and old key are now rejected
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"handle_or_email": "vxtzq", "password": "VxtzqInitialPassword!2026"},
        ).status_code
        == 401
    )
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {initial_key}"}).status_code == 401

    # New password logs in and returns the rotated hm-... key
    relogin = client.post(
        "/api/v1/auth/login",
        json={
            "handle_or_email": "116555319+vxtzq@users.noreply.github.com",
            "password": "VxtzqChosenPassword#2026",
        },
    )
    assert relogin.status_code == 200
    rotated_key = relogin.json()["api_key"]
    assert rotated_key.startswith("hm-")
    assert rotated_key != initial_key

    # Authenticated change-password with the hm-... key also works and returns a fresh key
    chg = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {rotated_key}"},
        json={"new_password": "VxtzqThirdPassword$2026"},
    )
    assert chg.status_code == 200
    third_key = chg.json()["api_key"]
    assert third_key.startswith("hm-")
    assert third_key != rotated_key


def test_signup_page_includes_password_reset_card() -> None:
    page = render_signup_page(None, "nonce-vxtzq-test")
    assert 'id="passwordResetCard"' in page
    assert 'id="changePasswordForm"' in page
    assert "/api/v1/auth/change-password" in page
    assert "/api/v1/auth/password-reset/confirm" in page


def test_dashboard_vxtzq_emails_and_change_password(tmp_path: Path) -> None:
    app = create_hivemind_at_home_app(vault_dir=tmp_path / "vault")
    client = TestClient(app)

    # Unauthenticated change-password fails with 401
    assert client.post("/api/v1/auth/change-password", json={"new_password": "ValidPassword123!"}).status_code == 401

    vxtzq_user = app.state.auth_state["users"]["vxtzq"]
    assert "116555319+vxtzq@users.noreply.github.com" in vxtzq_user["secondary_emails"]
    assert "vxtzq@crowdgpt.net" in vxtzq_user["secondary_emails"]
    seed_token = vxtzq_user["token"]

    # Authenticated with vxtzq's token, change-password binds a new password and rotates token
    chg = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {seed_token}"},
        json={"new_password": "VxtzqDashboardPassword!2026"},
    )
    assert chg.status_code == 200
    new_tok = chg.json()["token"]
    assert new_tok != seed_token

    # Vxtzq can now sign in with any of his emails and the new password
    for ident in ("@Vxtzq", "vxtzq1", "vxtzq@github.com", "116555319+vxtzq@users.noreply.github.com", "vxtzq@crowdgpt.net"):
        login = client.post(
            "/api/v1/auth/login",
            json={"handle_or_email": ident, "password": "VxtzqDashboardPassword!2026"},
        )
        assert login.status_code == 200, (ident, login.text)
        assert login.json()["token"] == new_tok


def test_a_reset_request_for_an_alias_sends_no_mail() -> None:
    """Regression: a reset link must never be sent to a legacy alias address (it would hand over the account)."""
    from fastapi.testclient import TestClient

    from hivemind.router.gateway import create_router_app

    store = RouterStore(":memory:", SECRET)
    store.ensure_legacy_hiveminders_registered()
    sent: list[tuple[str, str]] = []
    client = TestClient(create_router_app(store, reset_email_sender=lambda to, token: sent.append((to, token)),
                                          reset_email_enabled=lambda: True))
    for alias in ("contact@crowdgpt.net", "vxtzq@crowdgpt.net", "vxtzq@hivemind.network"):
        r = client.post("/api/v1/auth/password-reset/request", json={"email": alias})
        assert r.status_code == 202  # same answer as for any address
    assert sent == [], f"reset mail sent to an alias: {sent}"
    client.post("/api/v1/auth/password-reset/request", json={"email": "vxtzq@github.com"})
    assert [to for to, _ in sent] == ["vxtzq@github.com"]
