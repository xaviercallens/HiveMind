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


def test_portal_snippets_cover_the_documented_editors_and_keep_the_output_limit() -> None:
    html = render_portal_page(None, "n")
    for tab in ("opencode", "continue", "cline", "zed", "jetbrains", "aider", "env", "python", "curl"):
        assert f'data-tab="{tab}"' in html, tab  # every button has a snippet below
        assert f"  {tab}: `" in html, tab
    # the OpenCode snippet must carry the output limit, or a small balance is refused (observed on the live gateway)
    opencode = html.split("  opencode: `", 1)[1].split("`,", 1)[0]
    assert '"limit": { "context": 8192, "output": 700 }' in opencode
    # Continue now shows the YAML format the docs use, not the retired JSON "models" list
    assert "roles: [chat, edit, apply]" in html and '"title": "HiveMind Coder (7B)"' not in html
    # only tiers that can be served today are shown as ready snippets
    assert "hivemind/coder-plus" not in html.split("// Model ID:", 1)[1].split("\n", 1)[0]


def test_portal_page_renders_overnight_boinc_qlora_telemetry_and_caveats() -> None:
    html = render_portal_page(None, "nonce-night")
    assert 'id="night-telemetry"' in html
    assert "100 Assimilated WUs (98 hmq_*)" in html
    assert "5,632.11 Cobblestones" in html
    assert "2.2705 &rarr; 2.1380 (-5.8%)" in html
    assert "hmq_00092" in html
    assert "Plateau after ~30 rounds" in html
    assert "scripts/t4_training_report.py" in html



# IP-208 / IP-209 / IP-210: the OpenTrainer heartbeat and the live-telemetry script must not
# invent hardware, status or counter values. These tests read the rendered page source.

_HEARTBEAT_START = "$('opentrainerHeartbeatBtn').addEventListener"
_OPENTRAINER_STATUS_START = "fetch('/v1/hivemind/opentrainer/status')"
_LIVE_TELEMETRY_START = "async function refreshLivePortalTelemetry()"
_LIVE_TELEMETRY_END = "refreshLivePortalTelemetry();"


def _slice_between(html: str, start: str, end: str) -> str:
    """Return the source between the first ``start`` marker and the next ``end`` marker."""
    begin = html.index(start)
    return html[begin : html.index(end, begin)]


def test_ip208_heartbeat_sends_only_fields_the_user_typed() -> None:
    html = render_portal_page(None, "nonce-ip208")
    heartbeat = _slice_between(html, _HEARTBEAT_START, "// Contributive & Community Help Hub")
    for literal in ("XAVGAMSRV-RTX2070", "NVIDIA GeForce RTX 2070", "8192", "6400"):
        assert literal not in heartbeat, f"heartbeat JS still carries literal {literal!r}"
    assert 'value="XAVGAMSRV-RTX2070"' not in html, "host input is prefilled with a literal"
    assert 'id="otGpuInput" type="text" value=' not in html, "OpenTrainer GPU input is prefilled with a value"
    assert "if (hostLabel) payload.host_label = hostLabel;" in heartbeat
    assert "if (gpuName) payload.gpu_name = gpuName;" in heartbeat
    assert 'id="otVramTotalInput"' in html and 'id="otVramFreeInput"' in html
    assert "payload.vram_total_mib = Number(vramTotal)" in heartbeat


def test_ip209_status_prefixes_come_from_status_fields_not_string_literals() -> None:
    html = render_portal_page(None, "nonce-ip209")
    assert "'OPENTRAINER ONLINE" not in html, "status prefix is a hard-coded string"
    status = _slice_between(html, _OPENTRAINER_STATUS_START, "Network error querying OpenTrainer coordinator")
    assert "data.status" in status
    live = _slice_between(html, _LIVE_TELEMETRY_START, _LIVE_TELEMETRY_END)
    assert "'LIVE OPENTRAINER" not in live, "live status prefix is a hard-coded string"
    assert "v(ot.status)" in live
    for fallback in ("'1.0.0-s36'", "'hivemind.amcp.federation.v1'", "'IDLE_COMPUTE_GREEN'", "'just now'"):
        assert fallback not in live, f"live script still falls back to {fallback}"
    assert "|| '" not in live, "live script still has a string fallback"


def test_ip210_live_telemetry_script_has_no_numeric_or_constant_fallbacks() -> None:
    import re

    html = render_portal_page(None, "nonce-ip210")
    live = _slice_between(html, _LIVE_TELEMETRY_START, _LIVE_TELEMETRY_END)
    assert re.search(r"\|\|\s*\d", live) is None, "live script has a numeric || fallback"
    assert re.search(r"\|\|\s*'[A-Z_]{4,}'", live) is None, "live script has a constant || fallback"
    assert "v(byRole.trainer)" in live and "(byRole.trainer || 0)" not in live
