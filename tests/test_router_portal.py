"""Tests for the OpenRouter portal, playground and landing page at ``/`` and ``/playground``."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.portal_page import portal_content_security_policy, render_portal_page


def test_portal_content_security_policy_structure() -> None:
    csp = portal_content_security_policy("test-nonce-12345")
    assert "default-src 'none'" in csp
    assert "script-src 'nonce-test-nonce-12345'" in csp
    assert "style-src 'nonce-test-nonce-12345'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_render_portal_page_without_client_id() -> None:
    html = render_portal_page(None, "ephemeral-nonce-abc")
    assert "HiveMind OpenRouter" in html
    assert "Live AI Completion Console" in html
    assert "5 Community Capability Tiers" in html
    assert "qwen2.5-coder:7b" in html
    assert "ephemeral-nonce-abc" in html


def test_portal_and_playground_endpoints(tmp_path: pytest.TempPathFactory) -> None:
    store = RouterStore(tmp_path / "router.db", b"x" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    # GET /
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "HiveMind OpenRouter" in res_root.text
    assert "script-src 'nonce-" in res_root.headers.get("content-security-policy", "")
    assert res_root.headers.get("x-content-type-options") == "nosniff"

    # HEAD /
    res_head = client.head("/")
    assert res_head.status_code == 200
    assert "text/html" in res_head.headers.get("content-type", "")

    # GET /playground
    res_play = client.get("/playground")
    assert res_play.status_code == 200
    assert "text/html" in res_play.headers.get("content-type", "")
    assert "Live AI Completion Console" in res_play.text

    # HEAD /signup
    res_signup_head = client.head("/signup")
    assert res_signup_head.status_code == 200

    # HEAD /health and /healthz
    assert client.head("/health").status_code == 200
    assert client.head("/healthz").status_code == 200
