# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The signup page, the join page and the README state the starter grant the gateway is configured with, not a fixed 100."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

ROOT = Path(__file__).resolve().parents[1]


def _page(tmp_path: Path, grant: float) -> str:
    store = RouterStore(tmp_path / "router.db", secret=b"starter-grant-consistency-test-32b!!")
    client = TestClient(create_router_app(store, starter_grant_hmt=grant))
    resp = client.get("/signup")
    assert resp.status_code == 200
    return resp.text


def test_signup_page_with_no_grant_says_zero_and_never_shows_100(tmp_path: Path) -> None:
    html = _page(tmp_path, 0.0)
    assert "starts at 0.00 HMT" in html
    assert "100.00" not in html
    assert "minting starter HMT" not in html


def test_signup_page_shows_the_configured_grant(tmp_path: Path) -> None:
    html = _page(tmp_path, 25.0)
    assert "Starter grant on this gateway: 25.00 HMT" in html
    assert "100.00" not in html


def test_readme_and_join_page_make_no_fixed_100_claim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"\+\s?100(\.0)?\s*HMT", readme)
    join = (ROOT / "src" / "hivemind" / "router" / "join_page.py").read_text(encoding="utf-8")
    assert "0 points unless the operator grants some" in join
