# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Maintenance mode (HIVEMIND_MAINTENANCE_NOTICE): a header on every response and a visible banner on HTML pages; off means no change at all."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import MAINTENANCE_ENV, create_router_app

SECRET = b"hivemind-test-secret-at-least-32-bytes-long!"
NOTICE = "MVP work in progress: some models may be unavailable for a few hours."


def _client(**kw) -> TestClient:
    return TestClient(create_router_app(RouterStore(":memory:", SECRET), **kw))


def test_off_by_default_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAINTENANCE_ENV, raising=False)
    c = _client()
    page, api = c.get("/benchmarks"), c.get("/v1/models")
    assert "x-hivemind-maintenance" not in page.headers and "x-hivemind-maintenance" not in api.headers
    assert "hivemind-maintenance-banner" not in page.text and "x-hivemind-status" in page.headers  # the PoC header is still there


def test_on_adds_a_header_to_every_response_and_a_banner_to_html() -> None:
    c = _client(maintenance_notice=NOTICE)
    page, api, missing = c.get("/benchmarks"), c.get("/v1/models"), c.get("/no-such-page")
    for r in (page, api, missing):
        assert r.headers["x-hivemind-maintenance"] == NOTICE and "x-hivemind-status" in r.headers
    assert 'id="hivemind-maintenance-banner"' in page.text and NOTICE in page.text
    assert page.text.index("hivemind-maintenance-banner") < page.text.index("<h1")  # right after <body>, above the content
    assert "hivemind-maintenance-banner" not in api.text  # JSON bodies are never modified
    assert int(page.headers["content-length"]) == len(page.content)  # the length is recomputed after the injection


def test_the_notice_is_escaped_in_the_page_and_safe_in_the_header() -> None:
    evil = '<script>alert(1)</script>\r\nSet-Cookie: x=1'
    c = _client(maintenance_notice=evil)
    page = c.get("/benchmarks")
    assert "<script>alert(1)</script>" not in page.text and "&lt;script&gt;" in page.text
    value = page.headers["x-hivemind-maintenance"]
    assert "\r" not in value and "\n" not in value and "set-cookie" not in page.headers


def test_head_and_post_requests_still_work_with_the_banner_on() -> None:
    c = _client(maintenance_notice=NOTICE)
    assert c.head("/benchmarks").status_code == 200
    assert c.post("/v1/hivemind/register", json={"handle": "maint-user", "role": "consumer"}).status_code == 200


def test_the_environment_variable_turns_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAINTENANCE_ENV, NOTICE)
    assert _client().get("/v1/models").headers["x-hivemind-maintenance"] == NOTICE
    assert "x-hivemind-maintenance" not in _client(maintenance_notice="").get("/v1/models").headers  # an explicit empty notice switches it off


def test_a_very_long_notice_is_cut() -> None:
    assert len(_client(maintenance_notice="x" * 5000).get("/v1/models").headers["x-hivemind-maintenance"]) <= 300
