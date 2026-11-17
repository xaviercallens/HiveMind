"""Defensive Security & OWASP Top-10 Verification Suite for the HiveMind Portal & OpenRouter Gateway.

Verifies the portal, authentication, ledger, SSRF guards, rate limiters, CSP nonces, security headers,
and Zero-Trust Producer Shield against the defensive categories from `antigravity-awesome-skills`
(`top-web-vulnerabilities`, `broken-authentication`, `pentest-checklist`).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import (
    create_hivemind_at_home_app,
    sanitize_handle,
    sanitize_prompt_text,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app, producer_url_problem
from hivemind.router.hivemind_producer_shield import create_shield_app
from hivemind.router.welcome_email import build_welcome_email_content


def _make_gateway_client(tmp_path: Path) -> tuple[TestClient, RouterStore]:
    store = RouterStore(tmp_path / "router.db", secret=b"k" * 32)
    admin_hash = store.key_digest("top-secret-admin-key")
    app = create_router_app(
        store,
        admin_key_hash=admin_hash,
        bid_activity_window_s=900.0,
    )
    return TestClient(app), store


def test_security_headers_and_csp_nonce_isolation(tmp_path: Path) -> None:
    """Every portal page sets strict security headers and unique per-request CSP nonces."""
    client, _ = _make_gateway_client(tmp_path)
    nonces: set[str] = set()
    for path in ("/", "/playground", "/signup"):
        r1 = client.get(path)
        r2 = client.get(path)
        assert r1.status_code == 200 and r2.status_code == 200
        for resp in (r1, r2):
            assert resp.headers.get("x-content-type-options") == "nosniff"
            assert resp.headers.get("referrer-policy") in ("no-referrer", "strict-origin-when-cross-origin")
            if path == "/signup":
                assert resp.headers.get("cache-control") == "no-store"
            csp = resp.headers.get("content-security-policy", "")
            assert "default-src 'none'" in csp
            assert "frame-ancestors 'none'" in csp
            m = re.search(r"'nonce-([A-Za-z0-9_\-]+)'", csp)
            assert m is not None, f"Missing CSP nonce on {path}: {csp}"
            nonces.add(m.group(1))
    assert len(nonces) == 6, "Every HTML response must generate a fresh cryptographic CSP nonce"


def test_xss_html_escaping_and_csp_on_welcome_email_preview(tmp_path: Path) -> None:
    """Reflected XSS payloads in welcome email parameters and prompt sanitizers are strictly escaped."""
    xss_payload = "<script>alert('xss')</script><img src=x onerror=alert(1)>"
    content = build_welcome_email_content(
        account_id=xss_payload,
        handle=xss_payload,
        role=xss_payload,
        balance_hmt=100.0,
    )
    html_body = content["html_body"]
    assert "<script>alert" not in html_body
    assert "&lt;script&gt;alert" in html_body

    sanitized_prompt = sanitize_prompt_text(xss_payload)
    assert "<script>" not in sanitized_prompt
    assert "[SANITIZED_INJECTION_PAYLOAD]" in sanitized_prompt


def test_sql_injection_and_handle_validation_defenses(tmp_path: Path) -> None:
    """SQL injection vectors in handle fields are rejected without table corruption."""
    client, store = _make_gateway_client(tmp_path)
    sqli_vectors = [
        "' OR '1'='1",
        "admin'--",
        "user\"; DROP TABLE accounts;--",
        "google.spoofed_identity",
    ]
    for vec in sqli_vectors:
        r_handle = client.post("/v1/hivemind/register", json={"handle": vec, "role": "consumer"})
        assert r_handle.status_code == 400
    assert store.verify_chain() is True


def test_authentication_rate_limiting_and_key_rotation(tmp_path: Path) -> None:
    """Identity sign-in rotates the API key (revoking the prior key), and registration bursts trigger 429 rate limiting."""
    client, store = _make_gateway_client(tmp_path)
    acc1, old_key, created1 = store.sign_in_with_identity("google", "sub-12345", "prosumer")
    assert created1 is True
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {old_key}"}).status_code == 200

    # Re-signing in rotates the key: old key must be rejected (401), new key must succeed (200)
    acc2, new_key, created2 = store.sign_in_with_identity("google", "sub-12345", "prosumer")
    assert acc1 == acc2 and created2 is False and new_key != old_key
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {old_key}"}).status_code == 401
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {new_key}"}).status_code == 200

    # Registration burst exceeds SlidingWindowLimiter (10/min) -> 429
    statuses = [
        client.post("/v1/hivemind/register", json={"handle": f"burst_user_{i}", "role": "consumer"}).status_code
        for i in range(12)
    ]
    assert 200 in statuses
    assert 429 in statuses


def test_ssrf_guard_blocks_private_loopback_metadata_and_nat64() -> None:
    """SSRF guard rejects non-HTTPS, loopback, Cloud Run metadata (169.254.169.254), RFC1918, and NAT64-smuggled IPs."""
    blocked_urls = [
        ("http://example.com:8080", ["93.184.216.34"]),
        ("https://localhost:8080", ["127.0.0.1"]),
        ("https://127.0.0.1:11434", ["127.0.0.1"]),
        ("https://169.254.169.254/computeMetadata/v1/", ["169.254.169.254"]),
        ("https://10.0.0.12:8443", ["10.0.0.12"]),
        ("https://[::1]:8443", ["::1"]),
        ("https://[64:ff9b::a9fe:a9fe]:8443", ["64:ff9b::a9fe:a9fe"]),
        ("https://user:pass@example.com:8443", ["93.184.216.34"]),
    ]
    for url, ips in blocked_urls:
        reason = producer_url_problem(url, allow_private=False, resolver=lambda _: ips)
        assert reason is not None, f"Expected SSRF guard to block {url} ({ips})"

    ok_reason = producer_url_problem("https://gpu.example.com:8443", allow_private=False, resolver=lambda _: ["93.184.216.34"])
    assert ok_reason is None


def test_access_control_and_body_size_limit(tmp_path: Path) -> None:
    """Consumer accounts cannot advertise as producers (403), admin grant requires HMAC key (403), and >4MB bodies get 413."""
    client, _ = _make_gateway_client(tmp_path)
    reg = client.post("/v1/hivemind/register", json={"handle": "consumer_only", "role": "consumer"})
    assert reg.status_code == 200
    ckey = reg.json()["api_key"]

    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {ckey}"},
        json={"endpoint_url": "https://gpu.example.com", "models": [{"id": "coder", "local_name": "qwen"}], "cost_per_1k": 0.4, "capacity_1k": 20.0},
    )
    assert hb.status_code == 403

    # Admin grant without or with wrong x-admin-key -> 403
    assert client.post("/v1/hivemind/admin/grant", json={"handle": "consumer_only", "amount": 500.0, "ref": "hack"}).status_code == 403
    assert client.post(
        "/v1/hivemind/admin/grant",
        headers={"x-admin-key": "wrong-admin-key"},
        json={"handle": "consumer_only", "amount": 500.0, "ref": "hack"},
    ).status_code == 403

    # Oversized body (>4MB) rejected with 413 before JSON parsing
    huge = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {ckey}", "Content-Type": "application/json"},
        content=b'{"model":"hivemind/auto","messages":[{"role":"user","content":"' + (b"A" * 4_100_000) + b'"}]}',
    )
    assert huge.status_code == 413


def test_zero_trust_producer_shield_blocks_unauthenticated_and_admin_routes() -> None:
    """The Producer Shield enforces constant-time bearer token auth, model allow-lists, and blocks Ollama management routes."""
    shield_app = create_shield_app(
        upstream="http://127.0.0.1:65534",
        token="secret-shield-token-2026-long",
        max_concurrent=1,
        allowed_models=("qwen2.5-coder:7b",),
    )
    shield_client = TestClient(shield_app)

    # Docs and OpenAPI schema are disabled on the shield (404)
    assert shield_client.get("/docs").status_code == 404
    assert shield_client.get("/openapi.json").status_code == 404

    # Unauthenticated or wrong-token requests to /v1/models and /v1/chat/completions -> 401
    assert shield_client.get("/v1/models").status_code == 401
    assert shield_client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"}).status_code == 401
    assert shield_client.post("/v1/chat/completions", json={"model": "qwen2.5-coder:7b", "messages": []}).status_code == 401

    # Non-allowlisted model with valid token -> 403
    r_disallowed = shield_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-shield-token-2026-long"},
        json={"model": "unauthorized-model", "messages": []},
    )
    assert r_disallowed.status_code == 403

    # Ollama management endpoints (/api/pull, /api/delete, /api/tags) are not exposed (404)
    for blocked_path in ("/api/pull", "/api/delete", "/api/create", "/api/copy"):
        r_mgmt = shield_client.post(blocked_path, headers={"Authorization": "Bearer secret-shield-token-2026-long"}, json={"name": "llama3"})
        assert r_mgmt.status_code == 404


def test_ruche_dashboard_security_headers_and_ssrf_guard(tmp_path: Path) -> None:
    """HiveMind@home Ruche dashboard enforces security headers and gateway blocks SSRF producer URLs."""
    with tempfile.TemporaryDirectory() as tmp:
        app = create_hivemind_at_home_app(vault_dir=Path(tmp) / "vault")
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert "frame-ancestors 'none'" in r.headers.get("content-security-policy", "")
        assert "camera=()" in r.headers.get("permissions-policy", "")

    gw_client, _ = _make_gateway_client(tmp_path)
    reg = gw_client.post("/v1/hivemind/register", json={"handle": "ruche_prosumer", "role": "prosumer"})
    assert reg.status_code == 200
    key = reg.json()["api_key"]

    # SSRF attempt to Cloud Run metadata IP on heartbeat -> 400
    ssrf_hb = gw_client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "endpoint_url": "http://169.254.169.254/computeMetadata/v1/",
            "models": [{"id": "coder", "local_name": "qwen"}],
            "cost_per_1k": 0.4,
            "capacity_1k": 10.0,
        },
    )
    assert ssrf_hb.status_code == 400


def test_t4_sole_gpu_inference_and_european_night_qlora_moe_visibility(tmp_path: Path) -> None:
    """Verify T4 sole-GPU inference priority, European night multi-MoE QLoRA visibility, and /night-moe."""
    from hivemind.router.points_wording import currency_leftovers
    from hivemind.router.t4_visibility import build_t4_visibility_payload

    # 1. Verify daytime vs European night schedule calculation (Europe/Paris)
    day_payload = build_t4_visibility_payload(now_utc=1792936800.0, active_producers_count=1)
    assert isinstance(day_payload["sole_active_gpu"], bool)
    assert day_payload["european_schedule"]["is_european_night_window"] is False
    assert day_payload["operational_mode"] == "SOLE_T4_INFERENCE_PRIORITY_ACTIVE"
    assert len(day_payload["qlora_moe_curriculum"]) >= 4
    assert day_payload["qlora_progress"]["assimilated_workunits"] >= 4

    # 01:00 UTC in October = 02:00 CET (European Night Window)
    night_payload = build_t4_visibility_payload(now_utc=1792890000.0, active_producers_count=1)
    assert night_payload["european_schedule"]["is_european_night_window"] is True
    assert "EUROPEAN_NIGHT" in night_payload["operational_mode"]

    # 2. Verify OpenRouter HTML portal currency cleanliness and partner models endpoint
    client, _ = _make_gateway_client(tmp_path)
    r_portal = client.get("/")
    assert r_portal.status_code == 200
    assert currency_leftovers(r_portal.text) == []

    r_partners = client.get("/v1/hivemind/partner-models")
    assert r_partners.status_code == 200
    assert r_partners.json()["object"] == "hivemind.partner_models"

    # 3. Verify HiveMind@home Ruche dashboard and /night-moe visibility
    with tempfile.TemporaryDirectory() as tmp:
        ruche_app = create_hivemind_at_home_app(vault_dir=Path(tmp) / "vault")
        ruche_client = TestClient(ruche_app)
        r_dash = ruche_client.get("/")
        assert r_dash.status_code == 200

        r_night = ruche_client.get("/night-moe")
        assert r_night.status_code == 200
        assert "NIGHT-MOE-1" in r_night.text
