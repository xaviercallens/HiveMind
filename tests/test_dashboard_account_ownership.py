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


def test_upgrade_guest_cannot_take_an_existing_handle_or_mint_points(tmp_path: Path) -> None:
    client = _client(tmp_path)
    victim = client.post("/api/v1/auth/register", json={"handle": "victim01", "email": "v1@example.org"}).json()["token"]
    body = {"handle": "victim01", "email": "x@example.org", "provider": "google"}
    assert client.post("/api/v1/auth/upgrade-guest", json=body).status_code == 401  # no guest session
    guest = client.post("/api/v1/auth/guest", json={}).json()["token"]
    assert client.post("/api/v1/auth/upgrade-guest", headers={"Authorization": f"Bearer {guest}"}, json=body).status_code == 409
    for seeded in ("xavier_callens", "clark719664"):
        r = client.post("/api/v1/auth/upgrade-guest", headers={"Authorization": f"Bearer {guest}"}, json=dict(body, handle=seeded))
        assert r.status_code == 409, seeded
    # the victim still signs in with its own token
    assert client.post("/api/v1/auth/login", headers={"Authorization": f"Bearer {victim}"}, json={"handle_or_email": "victim01"}).status_code == 200
    # one upgrade per guest session: the upgraded token is not a guest any more, so chaining mints nothing
    up = client.post("/api/v1/auth/upgrade-guest", headers={"Authorization": f"Bearer {guest}"}, json=dict(body, handle="fresh-bee"))
    assert up.status_code == 200
    again = client.post(
        "/api/v1/auth/upgrade-guest", headers={"Authorization": f"Bearer {up.json()['token']}"}, json=dict(body, handle="fresh-bee-2")
    )
    assert again.status_code == 401


def test_sso_force_new_cannot_overwrite_an_existing_handle(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/api/v1/auth/register", json={"handle": "victim02", "email": "v2@example.org"})
    for handle in ("victim02", "clark719664"):
        r = client.post("/api/v1/auth/sso/google", json={"handle_hint": handle, "user_mode": "force_new"})
        assert r.status_code == 409, handle


def test_migration_status_is_admin_only(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/v1/auth/migration/status")
    assert r.status_code == 403
    assert "callensxavier@gmail.com" not in r.text


def test_public_forecast_inputs_are_clamped() -> None:
    import time as _time

    from hivemind.economy import international_launch_model as m

    t0 = _time.monotonic()
    m.forecast_200plus_node_international_launch(target_total_nodes=10**9, visitor_count=10**12)
    assert _time.monotonic() - t0 < 10.0  # 10^7 nodes used to take 44 s; 10^9 is clamped to MAX_FORECAST_NODES
    with pytest.raises(ValueError):
        m.compute_erlang_c_metrics(m.MAX_FORECAST_NODES + 1, 100.0, 1.0)


def test_federated_submit_refuses_oversized_lora_factors(tmp_path: Path) -> None:
    client = _client(tmp_path)
    lease = client.post("/api/v1/federated/workunit/request", json={"contributor_handle": "bee-1"})
    assert lease.status_code == 200, lease.text[:200]
    wu = lease.json()
    n = 5000  # a rank-1 1xN / Nx1 pair used to force an NxN matmul + SVD
    r = client.post(
        "/api/v1/federated/workunit/submit",
        json={
            "workunit_id": wu["workunit_id"],
            "lease_id": wu["lease_id"],
            "contributor_handle": "bee-1",
            "domain_name": wu["domain_name"],
            "train_nll_before": 2.0,
            "train_nll_after": 1.0,
            "holdout_nll_before": 2.0,
            "holdout_nll_after": 1.0,
            "lora_a_matrix": [[0.001] * n],
            "lora_b_matrix": [[0.001] for _ in range(n)],
        },
    )
    assert r.status_code == 422
    assert "r <= 64" in r.text  # refused by the shape bound, before any matmul or SVD


def test_rate_limiter_keys_on_the_rightmost_forwarded_address(tmp_path: Path) -> None:
    client = _client(tmp_path)
    codes = [
        client.post("/api/v1/auth/guest", headers={"X-Forwarded-For": f"10.0.{i // 250}.{i % 250}, 203.0.113.9"}, json={}).status_code
        for i in range(320)
    ]
    assert 429 in codes  # a rotating first X-Forwarded-For entry no longer gets a fresh bucket per request


def test_a_new_user_keeps_working_with_its_own_token(tmp_path: Path) -> None:
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/register", json={"handle": "new-bee", "email": "new-bee@example.org"})
    assert r.status_code == 200
    tok = r.json()["token"]
    assert client.post("/api/v1/auth/login", json={"handle_or_email": tok}).status_code == 200
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "new-bee"}).status_code == 401
    assert client.post("/api/v1/auth/login", headers={"Authorization": f"Bearer {tok}"}, json={"handle_or_email": "new-bee"}).status_code == 200
