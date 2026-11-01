# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""La Ruche must not credit points from numbers the client made up, and the page must not add points on its own."""

from __future__ import annotations

import re

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

_BODY = {"workunit_id": "wu-x", "lease_id": "lease-invented", "lease_version_tag": 1, "contributor_handle": "@t",
         "domain_name": "code_python", "train_nll_before": 5.5, "train_nll_after": 5.1,
         "holdout_nll_before": 5.5, "holdout_nll_after": 5.3}


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))


def test_invented_lease_is_refused_and_credits_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/v1/federated/workunit/submit", json=_BODY)
    assert resp.status_code == 422
    assert client.get("/api/v1/federated/ledger").json()["completed_receipts_count"] == 0


def test_page_script_never_adds_a_fixed_250_points(tmp_path: Path) -> None:
    html = _client(tmp_path).get("/").text
    assert not re.search(r"computePointsBalance\s*\+\s*250", html)
    assert "|| 250" not in html


def test_page_makes_no_fixed_point_reward_claim(tmp_path: Path) -> None:
    html = _client(tmp_path).get("/").text
    assert not re.search(r"\+\s?250\s*(pts|points|compute|hivemind|pts/wu)", html, re.I)
    assert "Earning Rate" not in html
    assert "credited only after server-side verification" in html


def test_boinc_status_promises_no_fixed_reward(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/boinc/status").text
    assert not re.search(r"\+\s?250\s*HMT", body)
    assert "no fixed amount per workunit is promised" in body


def test_leaderboard_and_wallet_show_no_invented_figures(tmp_path: Path) -> None:
    """Fresh vault: no invented leaderboard rows (35,500 pts for a founder) and no 1,250-point starting wallet."""
    client = _client(tmp_path)
    html = client.get("/").text
    table = html.split("<th>HiveMind Reflexion Points</th>")[1].split("</table>")[0]
    assert "No verified ledger rows yet" in table
    assert "pts</strong>" not in table  # no row with a points total
    assert "1,250" not in html
    assert "computePointsBalance = 0;" in html


def test_steam_daemon_reports_nothing_it_did_not_measure_and_credits_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    status = client.get("/api/v1/steam/daemon/status").json()
    assert status["workunits_completed"] == 0 and status["points_retributed"] == 0
    assert status["preemption_latency_ms"] is None and status["measured"] is False
    claim = client.post("/api/v1/steam/claim-points", json={"contributor_handle": "sso-user-01", "amount": 250}).json()
    assert claim["status"] == "NOT_CREDITED_UNVERIFIED" and claim["claimed_hmt"] == 0
    assert client.get("/api/v1/steam/daemon/status").json()["points_retributed"] == 0
    path = next(r.path for r in client.app.routes if getattr(r, "name", "") == "api_cockpit_steam_contribute_boinc")
    cockpit = client.post(path, json={"handle": "sso-user-01"}).json()
    assert cockpit["hmt_awarded"] == 0 and cockpit["cobblestones_awarded"] == 0 and cockpit["status"] == "RECEIVED_UNVERIFIED"
    assert "0.42s" not in client.get("/").text
