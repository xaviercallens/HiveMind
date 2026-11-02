# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""La Ruche dashboard: an existing account's token is only returned to its holder; admin routes need the admin key."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))


@pytest.mark.parametrize("ident", ["xavier_callens", "callensxavier@gmail.com", "xaviercallens"])
def test_owner_account_cannot_be_opened_by_handle_or_email(tmp_path: Path, ident: str) -> None:
    client = _client(tmp_path)
    for route, body in (
        ("/api/v1/auth/login", {"handle_or_email": ident}),
        ("/api/v1/auth/register", {"handle": ident, "email": ident if "@" in ident else "x@example.org"}),
        ("/api/v1/auth/magic-link", {"email": "callensxavier@gmail.com"}),
        ("/api/v1/auth/sso/google", {"handle_hint": "xavier_callens"}),
        ("/api/v1/auth/v1-migrate", {"handle_or_email": ident}),
    ):
        r = client.post(route, json=body)
        assert r.status_code in (401, 404, 422), (route, r.status_code)
        assert "MasterCommander" not in r.text and "admin:all" not in r.text, route


def test_empty_v1_migrate_no_longer_returns_the_owner(tmp_path: Path) -> None:
    r = _client(tmp_path).post("/api/v1/auth/v1-migrate")
    assert "callensxavier@gmail.com" not in r.text


def test_admin_routes_and_user_list_need_the_admin_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path)  # no admin key configured: closed
    assert client.post("/api/v1/admin/campaigns/create", json={"name": "x", "domain": "math"}).status_code == 403
    assert client.get("/api/v1/admin/dashboard").status_code == 403
    assert client.get("/api/v1/auth/users").status_code == 403

    monkeypatch.setenv("HIVEMIND_DASHBOARD_ADMIN_KEY", "k" * 32)
    client = _client(tmp_path)
    assert client.get("/api/v1/admin/dashboard", headers={"x-admin-key": "wrong"}).status_code == 403
    assert client.get("/api/v1/admin/dashboard", headers={"x-admin-key": "k" * 32}).status_code == 200


def test_seeded_tokens_are_fresh_at_every_start(tmp_path: Path) -> None:
    import re

    src = (Path(__file__).resolve().parents[1] / "src/hivemind/boinc/hivemind_at_home_dashboard.py").read_text(encoding="utf-8")
    published = set(re.findall(r'"token": "(hm_live_[0-9a-f]{12,})"', src))
    client = _client(tmp_path)
    for tok in published:
        r = client.post("/api/v1/auth/login", json={"handle_or_email": tok})
        assert "admin:all" not in r.text and "MasterCommander" not in r.text


def test_a_new_user_keeps_working_with_its_own_token(tmp_path: Path) -> None:
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/register", json={"handle": "new-bee", "email": "new-bee@example.org"})
    assert r.status_code == 200
    tok = r.json()["token"]
    assert client.post("/api/v1/auth/login", json={"handle_or_email": tok}).status_code == 200
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "new-bee"}).status_code == 401
    assert client.post("/api/v1/auth/login", headers={"Authorization": f"Bearer {tok}"}, json={"handle_or_email": "new-bee"}).status_code == 200
