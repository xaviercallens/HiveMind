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
