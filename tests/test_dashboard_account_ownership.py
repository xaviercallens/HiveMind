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
        assert r.status_code in (401, 404, 410, 422), (route, r.status_code)  # 410: the sandbox SSO is retired in v3.2
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


def test_migration_status_is_admin_only(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/v1/auth/migration/status")
    assert r.status_code == 403
    assert "callensxavier@gmail.com" not in r.text


def test_dashboard_responses_and_volunteer_tools_carry_the_proof_of_concept_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    client = _client(tmp_path)
    for resp in (client.get("/health"), client.get("/api/v1/auth/migration/status")):
        assert "proof-of-concept" in resp.headers.get("x-hivemind-status", "")
    assert "Proof of concept" in client.get("/openapi.json").json()["info"]["description"]

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import hivemind_volunteer_preflight as pf

    pf.main(["--explain"])
    err = capsys.readouterr().err
    assert "Proof of concept" in err and "public, non-sensitive" in err


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


def test_rate_limiter_keys_on_the_rightmost_forwarded_address(tmp_path: Path, monkeypatch) -> None:
    # pin the POST limit to the historic 300/min so the bypass check stays meaningful with the campaign default
    monkeypatch.setenv("HIVEMIND_RATE_POST_PER_MIN", "300")
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


def test_user_facing_texts_carry_the_notice_and_no_longer_claim_post_quantum_or_100_percent_safety() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("wiki/Home.md", "docs/USER_GUIDE_AND_CAMPAIGN_CALENDAR.md", "README.md"):
        assert "Proof of concept" in (root / rel).read_text(encoding="utf-8")[:2500], rel
    for rel in ("src/hivemind/router/welcome_email.py", "src/hivemind/router/manifesto_page.py"):
        assert "Proof of concept" in (root / rel).read_text(encoding="utf-8"), rel
    for path in list((root / "src").rglob("*.py")) + list((root / "docs").glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "Kyber-1024" not in text, path  # no post-quantum cryptography is implemented anywhere
        assert "100% Safe for Your" not in text and "**100% Private**" not in text, path


def test_guest_key_is_issued_fresh_per_start_and_no_published_value_authenticates(tmp_path, monkeypatch) -> None:
    """IP-240: the guest api_keys entry kept a literal token after the seeded users were reseeded."""
    import re

    from fastapi.testclient import TestClient

    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    monkeypatch.chdir(tmp_path)
    client = TestClient(create_hivemind_at_home_app())
    # a syntactically valid guest-looking key that was never issued by this process must not open the pre-seeded
    # guest account (login-by-unknown-string registers a new account of that name, which is a different matter)
    r = client.post("/api/v1/auth/login", json={"handle_or_email": "hm_live_" + "0" * 32})
    assert r.json().get("handle") != "guest_hiveminder"
    source = open(__import__("hivemind.boinc.hivemind_at_home_dashboard", fromlist=["x"]).__file__, encoding="utf-8").read()
    assert re.search(r"hm_live_[0-9a-f]{20,}", source) is None
