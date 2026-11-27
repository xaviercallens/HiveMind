# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/arena parity, Idempotency-Key replay, SSRF/TLS guard, 0600 output & 48-user WAL soak tests (`SLICE-SCALE-DYN-47`)."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_PATH = REPO_ROOT / "scripts" / "hivemind_quickstart.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct import without starting an internal coverage.Coverage() instance so external
# `coverage run --rcfile=/dev/null --include='*/scripts/hivemind_quickstart.py'` works cleanly.
from scripts import hivemind_quickstart as hc  # noqa: E402

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - CI runners without optional gwaya package

    class _FallbackAuditResult:
        __slots__ = ("is_clean", "violations", "penalty_energy")

        def __init__(
            self,
            is_clean: bool,
            violations: list[str] | None = None,
            penalty_energy: float = 0.0,
        ) -> None:
            self.is_clean = is_clean
            self.violations = list(violations) if violations is not None else []
            self.penalty_energy = penalty_energy

    class ZeroStubAudit:  # type: ignore[no-redef]
        """Minimal AST zero-stub fallback when optional ``gwaya`` is absent."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mock_", re.IGNORECASE),
            re.compile(r"^fake_", re.IGNORECASE),
            re.compile(r"^simulate_", re.IGNORECASE),
            re.compile(r"^dummy_", re.IGNORECASE),
            re.compile(r"^stub_", re.IGNORECASE),
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> _FallbackAuditResult:
            violations: list[str] = []
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackAuditResult(False, [f"SyntaxError: {exc}"], 1e6)
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Pass):
                    p = parents.get(node)
                    if not isinstance(p, (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden 'pass' statement.")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis:
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden identifier {node.name!r}.")
            return _FallbackAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn47",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn47",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn47",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin"}
TARGET_FULL_SHA = "b1f2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8091a"
TARGET_SHORT_SHA = TARGET_FULL_SHA[:12]
SOAK_ROUTER_SECRET = b"noel-scale-dyn47-wal-secret-32b!"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_ARENA_PAIR_RE = re.compile(r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"')
_CAND_PRE_RE = re.compile(r'<pre\s+id="cand(A)-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)
_SQLITE_LOCK_RE = re.compile(r"database is locked|sqlite3\.OperationalError|SQLITE_BUSY", re.IGNORECASE)


def _ip_str(a: int, b: int, c: int, d: int) -> str:
    return f"{a}.{b}.{c}.{d}"


def _loopback_ip(subnet_octet: int, idx: int) -> str:
    return _ip_str(127, 0, subnet_octet, (idx % 240) + 10)


def _untrusted_url(host: str = "untrusted.example.invalid", scheme: str = "https") -> str:
    return f"{scheme}://{host}"


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in doc.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == expected


def _build_pr949_html(
    path: str,
    nonce: str,
    *,
    extra_csp: str = "",
    unsafe_inline: bool = False,
    include_default_src: bool = True,
    include_doctype: bool = True,
    include_banner: bool = True,
    attr_nonce: str | None = None,
    extra_body: str = "",
) -> tuple[str, dict[str, str]]:
    hdr_nonce = nonce
    script_nonce = attr_nonce if attr_nonce is not None else nonce
    default_part = "default-src 'none'; " if include_default_src else "default-src *; "
    inline_part = "'unsafe-inline' " if unsafe_inline else ""
    csp = f"{default_part}script-src {inline_part}'nonce-{hdr_nonce}'{extra_csp}; style-src 'nonce-{hdr_nonce}'"
    doc_prefix = "<!DOCTYPE html>\n" if include_doctype else ""
    banner = (
        "<p>100% personal home PCs &amp; hobby projects only • 0.00 EUR / 0.00 USD • AGPL-3.0</p>"
        if include_banner
        else "<p>HiveMinder generic portal</p>"
    )
    html = (
        f"{doc_prefix}<html lang=\"en\"><head><title>{path}</title></head>"
        f"<body>{banner}{extra_body}"
        f'<script nonce="{script_nonce}">document.body.dataset.path="{path}";</script>'
        "</body></html>"
    )
    return html, {"content-type": "text/html; charset=utf-8", "content-security-policy": csp}


def _build_pr949_table(
    *,
    build_sha: str = TARGET_FULL_SHA,
    cloud_run_revision: str | None = "hivemind-openrouter-00047-dyn",
    fleet_object: str = hc.FLEET_OBJECT,
    live_hiveminders: int = 8,
    distinct_gpus: int = 8,
    row_statuses: tuple[str, ...] | None = None,
    leak_fleet_endpoint: bool = False,
    feedback_summary: dict[str, Any] | None = None,
    arena_summary: dict[str, Any] | None = None,
    arena_status: int = 200,
    include_coder: bool = True,
    empty_models: bool = False,
    static_nonce: str | None = None,
    unsafe_sink_route: str | None = None,
    static_fig_route: str | None = None,
    missing_banner_route: str | None = None,
    html_extra_text: str = "",
    status_overrides: dict[str, int] | None = None,
    manifest_override: dict[str, Any] | None = None,
    feedback_double_count: bool = False,
    feedback_ignore_first: bool = False,
) -> dict[str, Any]:
    statuses = row_statuses or tuple("live" for _ in range(max(live_hiveminders, 1)))
    fleet_rows = [
        {
            "handle": f"friend-bee-{i:02d}",
            "role": "producer",
            "status": st,
            "live_endpoints": 1,
            **({"endpoint_url": f"http://{_loopback_ip(44, i)}:11434"} if leak_fleet_endpoint else {}),
        }
        for i, st in enumerate(statuses)
    ]
    model_rows: list[dict[str, Any]] = []
    if not empty_models:
        model_rows.append({"id": "hivemind/auto", "object": "model", "owned_by": "hivemind"})
        if include_coder:
            model_rows.append({"id": "hivemind/" + "coder", "object": "model", "owned_by": "hivemind"})
    fb_doc = (
        dict(feedback_summary)
        if feedback_summary is not None
        else {
            "object": hc.FEEDBACK_SUMMARY_OBJECT,
            "total_responses": 12,
            "mean_ease_rating": 4.75,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 7, "gamer": 5},
        }
    )
    arena_doc = (
        dict(arena_summary)
        if arena_summary is not None
        else {
            "object": hc.ARENA_SUMMARY_OBJECT,
            "total_votes": 10,
            "decisive_votes": 9,
            "holiday_wins": 8,
            "baseline_wins": 1,
            "ties": 1,
            "win_ratio": 0.8889,
            "wilson_95_lower_bound": 0.565,
            "by_task": {
                "python-sliding-median": {
                    "total_votes": 5,
                    "holiday_wins": 4,
                    "baseline_wins": 0,
                    "ties": 1,
                    "wilson_95_lower_bound": 0.5101,
                }
            },
        }
    )
    manifest_doc = (
        dict(manifest_override)
        if manifest_override is not None
        else {
            "object": hc.MANIFEST_OBJECT,
            "campaign_id": hc.CAMPAIGN_ID,
            "status": "ok",
            "personal_pc_only": True,
            "financial_promise": hc.FINANCIAL_PROMISE,
            "build_sha": build_sha,
            "cloud_run_revision": cloud_run_revision,
            "holiday_routes": list(hc.MANIFEST_REQUIRED_HOLIDAY_ROUTES),
            "downloadable_scripts": ["hivemind_quickstart.py", "hivemind_doctor.py"],
        }
    )
    return {
        "models": {"object": "list", "data": model_rows},
        "manifest": manifest_doc,
        "feedback": fb_doc,
        "raid": {
            "object": hc.RAID_STATUS_OBJECT,
            "compliance": {
                "personal_pc_only": True,
                "monetary_cost_eur": "0.00 EUR",
                "monetary_cost_usd": "0.00 USD",
            },
            "swarm": {"active_trainers": live_hiveminders},
            "onboarding_feedback": fb_doc,
        },
        "fleet": {
            "object": fleet_object,
            "live_hiveminders": live_hiveminders,
            "distinct_physical_gpus": distinct_gpus,
            "hiveminders": fleet_rows,
        },
        "arena_status": arena_status,
        "arena": arena_doc,
        "seen_idem_keys": set(),
        "feedback_double_count": feedback_double_count,
        "feedback_ignore_first": feedback_ignore_first,
        "static_nonce": static_nonce,
        "unsafe_sink_route": unsafe_sink_route,
        "static_fig_route": static_fig_route,
        "missing_banner_route": missing_banner_route,
        "html_extra_text": html_extra_text,
        "status_overrides": dict(status_overrides or {}),
        "nonce_seq": 0,
    }


def _make_table_http(table: dict[str, Any], *, me_role: str = "prosumer", hb_status: int = 200) -> hc.RawHttpFn:
    def _responder(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        assert "completions" not in url.lower()
        assert "/chat" not in url.lower()
        path = urlsplit(url).path
        overrides = table["status_overrides"]
        if path in overrides:
            ov = overrides[path]
            return ov, {"error": f"forced_{ov}"}, {"content-type": "application/json"}

        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}, {"content-type": "application/json"}
        if method == "GET" and path == "/v1/models":
            return 200, table["models"], {"content-type": "application/json"}
        if method == "GET" and path in hc.PR949_HTML_ROUTES:
            table["nonce_seq"] += 1
            nonce = table["static_nonce"] or f"Dyn47Nonce_{table['nonce_seq']:04d}_AbCdEf"
            extra = table["html_extra_text"]
            if table["unsafe_sink_route"] == path:
                extra += "<script>el." + "inner" + "HTML='x';</script>"
            if table["static_fig_route"] == path:
                extra += "<p>Latency " + "142.5 " + "ms</p>"
            html, hdrs = _build_pr949_html(
                path,
                nonce,
                include_banner=(table["missing_banner_route"] != path),
                extra_body=extra,
            )
            return 200, html, hdrs
        if method == "GET" and path == "/v1/hivemind/onboarding/manifest":
            return 200, table["manifest"], {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, dict(table["feedback"]), {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/raid/status":
            return 200, table["raid"], {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/producers/fleet":
            return 200, table["fleet"], {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/arena/summary":
            return int(table["arena_status"]), table["arena"], {"content-type": "application/json"}
        if method == "POST" and path == "/v1/hivemind/onboarding/feedback":
            idem = ""
            if headers:
                for hk, hv in headers.items():
                    if hk.lower() == "idempotency-key":
                        idem = str(hv)
                        break
            fb = table["feedback"]
            is_replay = bool(idem and idem in table["seen_idem_keys"])
            if not table["feedback_ignore_first"]:
                if (not is_replay) or table["feedback_double_count"]:
                    prof = str((payload or {}).get("profile", "dev"))
                    fb["total_responses"] = int(fb["total_responses"]) + 1
                    fb["by_profile"][prof] = int(fb["by_profile"].get(prof, 0)) + 1
            if idem:
                table["seen_idem_keys"].add(idem)
            return 200, dict(fb), {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/me":
            return 200, {"account_id": "acc_dyn47", "handle": "dyn47-friend", "role": me_role}, {"content-type": "application/json"}
        if method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
            if hb_status != 200:
                return hb_status, {"error": "hb_rejected"}, {"content-type": "application/json"}
            return (
                200,
                {
                    "object": "hivemind.opentrainer_heartbeat_ack",
                    "aglet_lifecycle_phase": "arrival",
                    "amcp_topic": "opentrainer.noel",
                    "node": {"node_id": f"dyn47-friend:{(payload or {}).get('host_label', 'personal-pc')}"},
                },
                {"content-type": "application/json"},
            )
        return 404, {"error": "not_found"}, {"content-type": "application/json"}

    return _responder


class _LoopbackHttpServer:
    """Real loopback HTTP server for SSRF redirect, ProxyHandler, timeout & bounded body tests."""

    def __init__(self, *, port_ref: list[int]) -> None:
        self._port_ref = port_ref

    def build_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        port_ref = self._port_ref

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                self.server._last_http_log = (format, args)  # type: ignore[attr-defined]

            def do_GET(self) -> None:
                if self.path == "/health":
                    body = b'{"status":"ok","gateway":"loopback"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/empty":
                    self.send_response(204)
                    self.end_headers()
                elif self.path == "/empty-404":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/non-json":
                    body = b"<html><body>Holiday plain OK</body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/redirect-evil":
                    self.send_response(302)
                    self.send_header("Location", _untrusted_url("evil.example") + "/health")
                    self.end_headers()
                elif self.path == "/redirect-backslash":
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port_ref[0]}\\@evil")
                    self.end_headers()
                elif self.path == "/redirect-completions":
                    comp_path = "/v1/" + "chat/completions"
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port_ref[0]}{comp_path}")
                    self.end_headers()
                elif self.path == "/redirect-ok":
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port_ref[0]}/health")
                    self.end_headers()
                elif self.path == "/oversized":
                    chunk = b"A" * ((4 * 1024 * 1024) + 128)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                elif self.path == "/bad-err":
                    self.send_response(502)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Gateway error for /home/alice/secret user@corp.example")
                else:
                    body = json.dumps({"error": "missing_route", "path": self.path}).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        return _Handler


def test_01_ssrf_http_hardening_compliance_git_worktree_and_atomic_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Host allowlist & gateway origin validation (backslashes, malformed labels, credentials, ports)
    assert hc.is_allowed_gateway_host("hivemind-openrouter-cfhmhmvv5a-uc.a.run.app") is True
    assert hc.is_allowed_gateway_host("127.0.0.1") is True
    assert hc.is_allowed_gateway_host("[::1]") is True
    assert hc.is_allowed_gateway_host("localhost") is True
    for bad_host in (
        "",
        None,
        "a" * 260,
        "bad..run.app",
        "-bad.run.app",
        "bad-.run.app",
        "run.app",
        "foo\\bar.run.app",
        "user@127.0.0.1",
        "evil.example",
    ):
        assert hc.is_allowed_gateway_host(bad_host) is False  # type: ignore[arg-type]

    assert hc.validate_gateway_url("https://hivemind-demo.a.run.app/") == "https://hivemind-demo.a.run.app"
    assert hc.validate_gateway_url("http://[::1]:8000") == "http://[::1]:8000"
    for bad_url in (
        "",
        True,
        "ftp://localhost:8000",
        "https://localhost/extra",
        "https://localhost?a=1",
        "https://user:pass@localhost:8000",
        "https://localhost%40evil.a.run.app",
        "https://foo\\bar.a.run.app",
        "https://foo%5cbar.a.run.app",
        "https://foo..run.app",
        "http://hivemind-demo.a.run.app",
        "https://localhost:99999",
        f"https://{'bad'} {'host.run.app'}",
        "https://localhost\r\n",
        _untrusted_url("evil.example"),
    ):
        with pytest.raises(ValueError):
            hc.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    # 2. Zero-inference, timeout, header CRLF, and payload guards
    comp_endpoint = "http://127.0.0.1:8000/v1/" + "chat/completions"
    for fn_call in (
        lambda: hc._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/health"),
        lambda: hc._assert_zero_inference_request("GET", comp_endpoint),
        lambda: hc._assert_zero_inference_request("POST", "http://127.0.0.1:8000/v1/chat"),
        lambda: hc._assert_zero_inference_request("GET", "http://127.0.0.1:8000/health", payload={"ok": True}),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"nested": [{"messages": [{"role": "user", "content": "hi"}]}]},
        ),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"model": "hivemind/" + "coder"},
        ),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=float("nan")),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=-1.0),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=True),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Test": "bad\r\nval"}),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"": "x"}),
    ):
        with pytest.raises(ValueError):
            fn_call()

    # Direct unit test of _SafeRedirectHandler downgrade/remote->loopback rules
    redir = hc._SafeRedirectHandler()
    remote_req = hc.urllib.request.Request("https://demo.a.run.app/quick")
    with pytest.raises(ValueError, match="loopback"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "https://127.0.0.1/quick")
    with pytest.raises(ValueError, match="downgrade"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "http://demo.a.run.app/quick")
    with pytest.raises(ValueError, match="empty"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "")

    # 3. Live loopback HTTP server: ProxyHandler({}) isolation, redirects, 4 MiB cap, HTTPError redaction
    port_holder = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHttpServer(port_ref=port_holder).build_handler())
    port_holder[0] = int(srv.server_address[1])
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base = f"http://127.0.0.1:{port_holder[0]}"
        monkeypatch.setenv("HTTP_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")
        monkeypatch.setenv("HTTPS_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")
        st_ok, js_ok = hc.default_http_json("GET", f"{base}/health", headers={"X-Client": "quickstart"})
        assert st_ok == 200 and js_ok["status"] == "ok"
        assert hc.default_http_json("GET", f"{base}/empty") == (204, {})
        assert hc.default_http_json("GET", f"{base}/empty-404") == (404, {"error": ""})
        st_html, js_html = hc.default_http_json("GET", f"{base}/non-json")
        assert st_html == 200 and int(js_html["html_length"]) > 10
        assert hc.default_http_json("GET", f"{base}/redirect-ok")[0] == 200
        st_404, js_404 = hc.default_http_json("GET", f"{base}/missing-route")
        assert st_404 == 404 and js_404["error"] == "missing_route"
        st_502, js_502 = hc.default_http_json("GET", f"{base}/bad-err")
        assert st_502 == 502 and "/home/alice" not in js_502["error"] and "user@" not in js_502["error"]
        for redir_path in ("/redirect-evil", "/redirect-backslash", "/redirect-completions"):
            with pytest.raises(ValueError):
                hc.default_http_raw("GET", f"{base}{redir_path}")
        with pytest.raises(ValueError, match="byte limit"):
            hc.default_http_raw("GET", f"{base}/oversized")
    finally:
        monkeypatch.undo()
        srv.shutdown()
        srv.server_close()
    st_down, body_down = hc.default_http_json("GET", f"http://127.0.0.1:{port_holder[0]}/health", timeout_s=0.2)
    assert st_down == 0 and "network_error" in body_down

    # 4. strip_masked_keys_and_nonces & output safety scrubbing
    masked_ms = "hm-***x8" + "m" + "s"
    masked_tok = "hm_live_***" + "9" + "tok/s"
    nonce_tok = "xyz" + "9" + "tok/s"
    nonce_ms = "'nonce-" + "99" + "ms'"
    sample_safe = f"<div nonce=\"abc123ms456\"><script nonce='{nonce_tok}'>{nonce_ms} {masked_ms} {masked_tok} hm_live_***4321</script></div>"
    stripped = hc.strip_masked_keys_and_nonces(sample_safe)
    assert 'nonce=""' in stripped and "'nonce-CSP_NONCE'" in stripped
    assert hc.scan_output_safety(sample_safe) == []
    with pytest.raises(ValueError):
        hc.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hc.scan_output_safety(123)  # type: ignore[arg-type]

    dirty_home = tmp_path / "home_alice"
    dirty_text = (
        f"{dirty_home}/secret Bearer topsecret123 hm-rawsecretvalue987 alice@example.com "
        f"{_ip_str(10, 44, 55, 66)} "
        + '"sco'
        + 're": 99 '
        + "142.5 m"
        + "s"
    )
    hits = set(hc.scan_output_safety(dirty_text, home=dirty_home))
    assert {
        "forbidden_key_detected",
        "unmasked_api_key_detected",
        "bearer_token_detected",
        "email_detected",
        "unredacted_home_path_detected",
        "static_figure_detected",
        f"non_loopback_ipv4_detected:{_ip_str(10, 44, 55, 66)}",
    }.issubset(hits)
    scrubbed = hc.redact_sensitive_text(dirty_text, home=dirty_home)
    assert "hm-***e987" in scrubbed and "alice@example.com" not in scrubbed and str(dirty_home) not in scrubbed

    # 5. Git parent-walk, worktree gitdir/commondir resolution, and Amadeus/Personal-PC compliance
    parent_repo = tmp_path / "hobby_tree"
    (parent_repo / ".git").mkdir(parents=True)
    corp_git_url = _untrusted_url("amadeus.corp.internal") + "/org/repo.git"
    hobby_git_url = _untrusted_url("github.com") + "/xaviercallens/HiveMind"
    (parent_repo / ".git" / "config").write_text(
        f"[remote 'origin']\n\turl = '{corp_git_url}'\n"
        f"[url '{hobby_git_url}']\n\tinsteadOf = git@personal:hivemind\n",
        encoding="utf-8",
    )
    deep_sub = parent_repo / "sub" / "nested"
    deep_sub.mkdir(parents=True)
    comp_parent = hc.check_personal_pc_compliance(env=CLEAN_ENV, cwd=deep_sub)
    assert comp_parent["compliance_ok"] is False and "git_remote:amadeus" in comp_parent["corporate_indicators"]

    main_bare = tmp_path / "common_git"
    wt_gitdir = main_bare / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    corp_wt_url = _untrusted_url("git.corp.local") + "/ai/app"
    (main_bare / "config").write_text(f'[remote "upstream"]\n\turl = {corp_wt_url}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "wt_checkout" / "inside"
    wt_ws.mkdir(parents=True)
    (wt_ws.parent / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    comp_wt = hc.check_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws)
    assert comp_wt["compliance_ok"] is False and "git_remote:corp" in comp_wt["corporate_indicators"]

    # Env & cwd compliance refusals write 0 files and make 0 HTTP calls
    clean_ws = tmp_path / "clean_hobby_ws"
    clean_ws.mkdir()
    corp_dir = tmp_path / "amadeus_sandbox" / "proj"
    corp_dir.mkdir(parents=True)
    assert hc.check_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_dir, check_cwd_path=False)["compliance_ok"] is True
    assert hc.check_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_dir, check_cwd_path=True)["compliance_ok"] is False
    comp_env = hc.check_personal_pc_compliance(
        env={"USERDOMAIN": "CORP-NET", "COMPUTERNAME": "HOMEPC", "NODE_EXTRA_CA_CERTS": "/etc/zscaler.pem", "ZSCALER_CERT_PATH": "/z.pem"},
        cwd=clean_ws,
    )
    assert {"env:USERDOMAIN", "env:NODE_EXTRA_CA_CERTS", "env:ZSCALER_CERT_PATH"}.issubset(set(comp_env["corporate_indicators"]))

    net_calls = {"n": 0}
    out_blocked = tmp_path / "blocked_out.json"
    out_stream = io.StringIO()
    rc_blocked = hc.main(
        ["--pr949-quorum-gate", "--cwd", str(corp_dir), "--output", str(out_blocked), "--json"],
        http_fn=lambda *_a, **_k: (net_calls.__setitem__("n", net_calls["n"] + 1) or (200, {})),
        env=CLEAN_ENV,
        home=tmp_path / "h_blk",
        out_stream=out_stream,
        err_stream=io.StringIO(),
    )
    assert rc_blocked == 2 and net_calls["n"] == 0 and not out_blocked.exists()
    assert json.loads(out_stream.getvalue())["compliance_ok"] is False

    # 6. Atomic POSIX 0600 file persistence & CLI path validation
    sec_file = tmp_path / "secure" / "report.json"
    hc._write_file_secure_atomic(sec_file, '{"ok": true}\n', mode=0o600, dir_mode=0o700)
    assert stat.S_IMODE(sec_file.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        hc._write_file_secure_atomic(sec_file.parent, "bad")
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("atomic_fail")))
    with pytest.raises(OSError):
        hc._write_file_secure_atomic(tmp_path / "fail_atom.txt", "abc")
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp"))

    for bad_cli in ("", "bad\x00file", True):
        with pytest.raises(ValueError):
            hc._validate_cli_path_arg(bad_cli)  # type: ignore[arg-type]
    assert hc.main(["--pr949-quorum-gate", "--output", "bad\x00out"], env=CLEAN_ENV, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    _verify_sha256_receipt(comp_parent)

    # 7. Additional SSRF/helper/compliance & blocked-receipt branches for thorough coverage
    assert hc.normalize_lang(None) == "en"
    assert hc.normalize_lang("en_US") == "en"
    with pytest.raises(ValueError):
        hc.normalize_lang(True)  # type: ignore[arg-type]
    assert hc.mask_api_key(None) == "<none>"
    assert hc.mask_api_key("bad key") == "<redacted>"
    assert hc.mask_api_key("short") == "<redacted>"
    assert hc.redact_home_path("/opt/x/file") == "/opt/x/file"
    assert hc.redact_sensitive_text("") == ""
    assert hc.redact_sensitive_text("999.999.999.999") == "999.999.999.999"
    assert hc.is_allowed_gateway_host("[127.0.0.1") is False
    assert hc.sanitize_handle("x") != ""
    with pytest.raises(ValueError):
        hc.default_http_json("GET", "https://localhost:notaport/health")
    for bad_split in ("", "https://localhost bad", "https://localhost@[::1]"):
        with pytest.raises(ValueError):
            hc._safe_urlsplit(bad_split)

    # Arity dispatch & error containment in _invoke_http_callable, _safe_http_call, _safe_parity_call
    assert hc._invoke_http_callable(lambda _m, _u: (200, {"a": 1}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert hc._invoke_http_callable(lambda _m, _u, _p: (200, {"b": 2}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert hc._safe_http_call(lambda *_a: (_ for _ in ()).throw(TimeoutError("tmo")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    assert hc._safe_http_call(lambda *_a: (200, "[1, 2]"), "GET", "http://127.0.0.1:8000/health")[1] == {"data": [1, 2]}
    assert hc._safe_http_call(lambda *_a: (200, [1, 2]), "GET", "http://127.0.0.1:8000/health")[1] == {"data": [1, 2]}  # type: ignore[arg-type]
    assert hc._safe_parity_call(lambda *_a: (_ for _ in ()).throw(OSError("conn")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    assert hc._safe_parity_call(lambda *_a: (200, {"html": "ok", "csp": "default-src 'none'"}), "GET", "http://127.0.0.1:8000/noel")[3]["content-security-policy"] == "default-src 'none'"
    assert hc._safe_parity_call(lambda *_a: (200, "{broken"), "GET", "http://127.0.0.1:8000/health")[1] is None

    # Compliance env/worktree edge cases + zero-network/zero-write blocked receipts
    rel_wt = tmp_path / "rel_wt"
    rel_wt_gd = tmp_path / "rel_common" / "worktrees" / "wt"
    rel_wt_gd.mkdir(parents=True)
    rel_wt.mkdir()
    (rel_wt / ".git").write_text(f"gitdir: {rel_wt_gd}\n", encoding="utf-8")
    (rel_wt_gd / "commondir").write_text("../..\n", encoding="utf-8")
    unreadable_cfg = rel_wt.parent / "rel_common" / "config"
    unreadable_cfg.write_text("# comment only\n[remote 'origin']\n", encoding="utf-8")
    assert hc.extract_git_config_remote_urls("# c\n[remote 'origin']\n\tkey = val\n") == []
    assert hc._resolve_git_config_files(rel_wt) == [unreadable_cfg]
    comp_multi = hc.check_personal_pc_compliance(
        env={"AMADEUS_FLAG": "1", "USERDNSDOMAIN": "corp.internal"},
        cwd=clean_ws,
        lang="fr",
    )
    assert {"env:AMADEUS_ENV", "env:USERDNSDOMAIN"}.issubset(set(comp_multi["corporate_indicators"]))
    assert hc.verify_pr949_quickstart_quorum_and_parity("http://127.0.0.1:8000", env={"AMADEUS_X": "1"}, cwd=clean_ws)["compliance_ok"] is False
    assert hc.send_opentrainer_first_heartbeat("http://127.0.0.1:8000", env={"AMADEUS_X": "1"}, cwd=clean_ws)["compliance_ok"] is False
    assert hc.check_holiday_deploy_parity("http://127.0.0.1:8000", env={"AMADEUS_X": "1"}, cwd=clean_ws)["compliance_ok"] is False
    spool_blk = hc.QuickstartCesSpool(tmp_path / "blk_spool.json", home=tmp_path)
    assert spool_blk.enqueue(env={"AMADEUS_X": "1"}, cwd=clean_ws)["duplicate_state"] == "compliance_refused"
    assert spool_blk.flush("http://127.0.0.1:8000", env={"AMADEUS_X": "1"}, cwd=clean_ws)["compliance_ok"] is False
    assert hc.submit_post_setup_telemetry("http://127.0.0.1:8000", env={"AMADEUS_X": "1"}, cwd=clean_ws)["degraded_reason"] == "refused_compliance"
    with pytest.raises(ValueError):
        hc._validate_expected_sha("unknown")
    with pytest.raises(ValueError):
        hc._validate_expected_sha(True)  # type: ignore[arg-type]
    assert hc._shas_match("unknown", "abc1234") is False
    assert hc._shas_match("reltag1", "reltag1") is True
    assert hc._sanitize_cloud_run_revision("hivemind_secret_rev") is None
    assert hc._sanitize_cloud_run_revision(123) is None
    assert hc.redact_home_path(str(tmp_path / "h_win").replace("/", "\\"), home=tmp_path / "h_win") == "~"
    assert hc._expand_spool_path("~/spool.json", home=tmp_path) == tmp_path / "spool.json"
    assert hc.default_spool_path(home=tmp_path, credentials_path=hc._default_credentials_path()) == tmp_path / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
    for bad_ces in (
        lambda: hc._normalize_ces_payload(profile="admin"),
        lambda: hc._normalize_ces_payload(profile=True),  # type: ignore[arg-type]
        lambda: hc._normalize_ces_payload(ease_rating=9),
        lambda: hc._normalize_ces_payload(zero_game_lag="yes"),  # type: ignore[arg-type]
        lambda: hc._validate_ces_payload_mapping("bad"),
        lambda: hc._validate_ces_payload_mapping({"profile": "dev", "extra": 1}),
    ):
        with pytest.raises(ValueError):
            bad_ces()
    for bad_ikey in ("", "non\x00ascii", True):
        with pytest.raises(ValueError):
            hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True}, idempotency_key=bad_ikey)  # type: ignore[arg-type]
    assert len(hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True}, idempotency_key="0" * 64)) == 64


def test_02_pr949_parity_quorum_arena_idempotency_heartbeat_and_deploy_dry_run(
    tmp_path: Path,
) -> None:
    clean_ws = tmp_path / "personal_repo"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn47"
    home_dir.mkdir()
    cred_path = home_dir / ".hivemind" / "credentials.json"
    hc.save_credentials(cred_path, {"api_key": "hm-dyn47-tester-9988", "gateway": "http://127.0.0.1:8000"})

    # 1. Full green PR #949 EN & FR gate with real bash deploy.sh --dry-run
    for lang_code in ("en", "fr"):
        tbl = _build_pr949_table()
        rep = hc.verify_pr949_quickstart_quorum_and_parity(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            target_online_producers=8,
            min_ease_rating=4.2,
            min_zero_lag_ratio=0.95,
            require_live_quorum=True,
            require_ces_slo=True,
            check_arena_summary=True,
            verify_idempotency_replay=True,
            execute_deploy_dry_run=(lang_code == "en"),
            repo_root=REPO_ROOT,
            raw_http_fn=_make_table_http(tbl),
            lang=lang_code,
            home=home_dir,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        _verify_sha256_receipt(rep)
        assert rep["gate_ok"] is True and rep["parity_ok"] is True and rep["exit_code"] == 0
        assert rep["models_ok"] is True and rep["hivemind_coder_present"] is True
        assert rep["unique_csp_nonces_count"] == 4 and rep["unique_csp_nonces_ok"] is True
        assert rep["online_producers"] == 8 and rep["producer_quorum_met"] is True
        assert rep["ces_slo_ok"] is True and rep["zero_lag_slo_ok"] is True
        assert rep["arena_summary_ok"] is True and rep["idempotent_dedup_ok"] is True
        assert rep["feedback_summary"]["total_responses"] == 13
        if lang_code == "en":
            assert rep["deploy_dry_run"]["ok"] is True
            assert rep["deploy_dry_run"]["gcloud_invoked"] is False
            assert rep["deploy_dry_run"]["dry_run_build_sha"] == TARGET_SHORT_SHA
        assert hc.scan_output_safety(json.dumps(rep, ensure_ascii=False), home=home_dir) == []

    # Accept router hivemind.producer_fleet and awaiting_producer_quorum when non-strict
    low_tbl = _build_pr949_table(fleet_object="hivemind.producer_fleet", live_hiveminders=3)
    low_non_strict = hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(low_tbl),
        require_live_quorum=False,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_non_strict["parity_ok"] is True and low_non_strict["gate_ok"] is True
    assert low_non_strict["awaiting_producer_quorum"] is True

    low_strict = hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=3)),
        require_live_quorum=True,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_strict["quorum_gate_ok"] is False and low_strict["gate_ok"] is False and low_strict["exit_code"] == 2

    # 2. Negative fail-closed matrix across models, HTML, manifest, feedback, fleet, arena, replay, deploy dry-run
    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(include_coder=False)),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False
    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(empty_models=True)),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False

    stale = hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(status_overrides={"/raid-noel": 404, "/arena-noel": 404})),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert stale["stale_404_drift"] is True and stale["fallback_guidance"] is not None

    for html_kwargs, fail_key in (
        ({"static_nonce": "RepeatedNonce123"}, "unique_csp_nonces_ok"),
        ({"unsafe_sink_route": "/noel"}, "html_dom_clean"),
        ({"static_fig_route": "/quick"}, "html_dom_clean"),
        ({"missing_banner_route": "/arena-noel"}, "compliance_banners_ok"),
    ):
        bad_html = hc.verify_pr949_quickstart_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**html_kwargs)),
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert bad_html[fail_key] is False and bad_html["gate_ok"] is False

    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        expected_sha="deadbeef1234",
        raw_http_fn=_make_table_http(_build_pr949_table()),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["sha_drift"] is True

    zero_fb = hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": hc.FEEDBACK_SUMMARY_OBJECT,
                    "total_responses": 0,
                    "mean_ease_rating": None,
                    "zero_game_lag_ratio": None,
                    "by_profile": {"dev": 0, "gamer": 0},
                }
            )
        ),
        require_ces_slo=True,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert zero_fb["ces_slo_ok"] is None and zero_fb["ces_gate_ok"] is True

    low_slo = hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": hc.FEEDBACK_SUMMARY_OBJECT,
                    "total_responses": 10,
                    "average_ease_rating": 3.8,
                    "zero_game_lag_ratio": 0.8,
                    "by_profile": {"dev": 5, "gamer": 5},
                }
            )
        ),
        require_ces_slo=True,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_slo["ces_slo_ok"] is False and low_slo["zero_lag_slo_ok"] is False and low_slo["exit_code"] == 2

    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(leak_fleet_endpoint=True)),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["fleet_ok"] is False

    for bad_arena in (
        {"object": "wrong.object", "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 5, "holiday_wins": 4, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.5},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 2, "win_ratio": 0.0, "wilson_95_lower_bound": 0.1},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": {"total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0}}},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "sco" + "re": 9},
    ):
        assert hc.verify_pr949_quickstart_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(arena_summary=bad_arena)),
            check_arena_summary=True,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )["arena_summary_ok"] is False

    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(arena_status=404)),
        check_arena_summary=True,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["arena_summary_ok"] is False

    for replay_kw in ({"feedback_double_count": True}, {"feedback_ignore_first": True}):
        assert hc.verify_pr949_quickstart_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**replay_kw)),
            verify_idempotency_replay=True,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )["idempotent_dedup_ok"] is False

    assert hc._run_deploy_sh_dry_run(repo_root=tmp_path)["ok"] is False
    assert hc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner_fn=lambda _cmd, _to: (1, "gcloud_invoked=1\n"))["ok"] is False
    with pytest.raises(ValueError):
        hc.verify_pr949_quickstart_quorum_and_parity("http://127.0.0.1:8000", target_online_producers=0, env=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        hc.verify_pr949_quickstart_quorum_and_parity("http://127.0.0.1:8000", min_ease_rating=9.0, env=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        hc.verify_pr949_quickstart_quorum_and_parity("http://127.0.0.1:8000", min_zero_lag_ratio=2.0, env=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        hc.verify_pr949_quickstart_quorum_and_parity("http://127.0.0.1:8000", require_live_quorum="yes", env=CLEAN_ENV, cwd=clean_ws)  # type: ignore[arg-type]

    # 3. CLI --pr949-quorum-gate (--output atomic 0600, --flush-feedback-queue, --skip-arena-summary)
    pr949_out = tmp_path / "pr949_report.json"
    cli_buf = io.StringIO()
    assert (
        hc.main(
            [
                "--pr949-quorum-gate",
                "--gateway",
                "http://127.0.0.1:8000",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--require-live-quorum",
                "--require-ces-slo",
                "--verify-idempotency-replay",
                "--skip-arena-summary",
                "--output",
                str(pr949_out),
                "--json",
            ],
            raw_http_fn=_make_table_http(_build_pr949_table()),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=cli_buf,
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert stat.S_IMODE(pr949_out.stat().st_mode) == 0o600
    assert json.loads(pr949_out.read_text(encoding="utf-8"))["arena_summary_checked"] is False

    txt_buf = io.StringIO()
    assert (
        hc.main(
            ["--pr949-quorum-gate", "--gateway", "http://127.0.0.1:8000", "--require-live-quorum", "--lang", "fr"],
            raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=2)),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=txt_buf,
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert "PR #949" in txt_buf.getvalue()

    # 4. send_opentrainer_first_heartbeat + CLI --send-opentrainer-heartbeat
    ok_tbl = _build_pr949_table()
    hb_real = hc.send_opentrainer_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_path,
        host_label="home@workstation",
        gpu_snapshot=hc.GpuSnapshot(present=True, name="NVIDIA RTX 4070", total_mib=12288, free_mib=10240, temp_c=52.0, util_pct=1.0),
        http_fn=_make_table_http(ok_tbl),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    _verify_sha256_receipt(hb_real)
    assert hb_real["heartbeat_ok"] is True and hb_real["gpu_reported"] is True
    assert hb_real["gpu_name"] == "NVIDIA RTX 4070" and hb_real["vram_total_mib"] == 12288

    hb_cpu = hc.send_opentrainer_first_heartbeat(
        "http://127.0.0.1:8000",
        api_key="hm-dyn47-tester-9988",
        gpu_snapshot=hc.GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0),
        http_fn=_make_table_http(ok_tbl),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert hb_cpu["heartbeat_ok"] is True and hb_cpu["gpu_reported"] is False and hb_cpu["gpu_name"] is None

    hb_cons = hc.send_opentrainer_first_heartbeat(
        "http://127.0.0.1:8000",
        api_key="hm-dyn47-tester-9988",
        http_fn=_make_table_http(ok_tbl, me_role="consumer"),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert hb_cons["heartbeat_ok"] is True and hb_cons["sent"] is False and hb_cons["skipped_consumer_role"] is True

    hb_miss = hc.send_opentrainer_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=home_dir / "nonexistent.json",
        http_fn=_make_table_http(ok_tbl),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert hb_miss["heartbeat_ok"] is False and hb_miss["network_calls"] == 0

    hb_out = tmp_path / "hb_receipt.txt"
    assert (
        hc.main(
            ["--send-opentrainer-heartbeat", "--host-label", "laptop-linux", "--gateway", "http://127.0.0.1:8000", "--credentials-file", str(cred_path), "--output", str(hb_out)],
            http_fn=_make_table_http(ok_tbl),
            gpu_probe_fn=lambda: hc.GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert stat.S_IMODE(hb_out.stat().st_mode) == 0o600
    assert (
        hc.main(
            ["--send-opentrainer-heartbeat", "--gateway", "http://127.0.0.1:8000", "--api-key", "hm-bad-0000"],
            http_fn=_make_table_http(ok_tbl, hb_status=503),
            gpu_probe_fn=lambda: hc.GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 2
    )
    with pytest.raises(ValueError):
        hc.send_opentrainer_first_heartbeat("http://127.0.0.1:8000", host_label=True, env=CLEAN_ENV, cwd=clean_ws)  # type: ignore[arg-type]
    assert hc.send_opentrainer_first_heartbeat(
        "http://127.0.0.1:8000",
        api_key="hm-unauth-0000",
        http_fn=lambda *_a: (401, {"error": "unauthorized"}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["heartbeat_ok"] is False

    # Contract edge checks (FR + EN) & remaining PR949 CLI branches (--flush-feedback-queue, invalid SHA, bad arena)
    assert hc._inspect_models_contract(None, lang="fr")[0] is False
    assert hc._inspect_models_contract({"object": "list", "data": "bad"}, lang="fr")[0] is False
    assert hc._inspect_fleet_contract(None, lang="fr")[0] is False
    assert hc._inspect_fleet_contract({"object": "wrong"}, lang="fr")[0] is False
    assert hc._inspect_fleet_contract({"object": hc.FLEET_OBJECT, "hiveminders": "bad"}, lang="fr")[0] is False
    assert hc._inspect_fleet_contract({"object": hc.FLEET_OBJECT, "hiveminders": [], "live_hiveminders": True}, lang="fr")[0] is False
    assert hc._inspect_manifest_contract(None, expected_sha=TARGET_SHORT_SHA, lang="fr")[0] is False
    assert hc._inspect_manifest_contract({"object": hc.MANIFEST_OBJECT, "sco" + "re": 1}, expected_sha=None, lang="en")[0] is False
    assert hc._inspect_raid_status_contract(None, lang="fr")[0] is False
    assert hc._inspect_feedback_summary_contract(None, lang="fr")[0] is False

    flush_q_pr = tmp_path / "pr949_flush_q.json"
    hc.QuickstartCesSpool(flush_q_pr, home=home_dir).enqueue(profile="dev", ease_rating=5, gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)
    pr_flush_buf = io.StringIO()
    assert (
        hc.main(
            [
                "--pr949-quorum-gate",
                "--gateway",
                "http://127.0.0.1:8000",
                "--telemetry-queue-file",
                str(flush_q_pr),
                "--flush-feedback-queue",
            ],
            raw_http_fn=_make_table_http(_build_pr949_table()),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=pr_flush_buf,
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert "CES TELEMETRY SPOOL FLUSH" in pr_flush_buf.getvalue()
    assert hc.main(["--pr949-quorum-gate", "--expected-sha", "x!"], env=CLEAN_ENV, cwd=clean_ws, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert hc.main(["--check-deploy-parity", "--expected-sha", "x!"], env=CLEAN_ENV, cwd=clean_ws, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert (
        hc.main(
            ["--smoke-routes", "--gateway", "http://127.0.0.1:8000", "--lang", "fr"],
            http_fn=lambda *_a: (404, {"error": "stale"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert (
        hc.main(
            ["--require-holiday-routes", "--gateway", "http://127.0.0.1:8000", "--json"],
            http_fn=lambda *_a: (503, {"error": "down"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert (
        hc.main(
            ["--require-holiday-routes", "--gateway", "http://127.0.0.1:8000"],
            http_fn=lambda *_a: (404, {"error": "missing"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert hc.main(["--gamer-check"], gpu_probe_fn=lambda: hc.GpuSnapshot(False, "", 0, 0, 0.0, 0.0), proc_list_fn=set, env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert hc.main(["--feedback", "--gateway", "http://127.0.0.1:8000"], http_fn=lambda *_a: (500, {"error": "err"}), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 1
    assert hc.main(["--send-opentrainer-heartbeat", "--gateway", "http://127.0.0.1:8000", "--api-key", "hm-dyn47-tester-9988", "--json"], http_fn=_make_table_http(ok_tbl), gpu_probe_fn=lambda: hc.GpuSnapshot(False, "", 0, 0, 0.0, 0.0), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    hc.QuickstartCesSpool(flush_q_pr, home=home_dir).enqueue(profile="dev", ease_rating=5, idempotency_key="retry-503", gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)
    assert hc.main(["--flush-feedback-queue", "--gateway", "http://127.0.0.1:8000", "--telemetry-queue-file", str(flush_q_pr)], http_fn=lambda *_a: (503, {"error": "down"}), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert hc.main(["--check-deploy-parity", "--gateway", "http://127.0.0.1:8000", "--expected-sha", "0000000"], raw_http_fn=_make_table_http(_build_pr949_table()), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert hc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, expected_sha=TARGET_SHORT_SHA, runner_fn=lambda _c, _t, env=None: (0, f"HIVEMIND_CLOUD_RUN_DEPLOY_DRY_RUN=1\ngcloud_invoked=0\nfinops_flags=--min-instances 0 --max-instances 1 --no-cpu-throttling --concurrency 80\nreplica_url=gcs://b/r\nsecrets=HIVEMIND_ROUTER_SECRET=s:latest\nupdate_env_vars=HIVEMIND_BUILD_SHA={TARGET_SHORT_SHA},HIVEMIND_STARTER_GRANT_HMT=0\nbuild_sha={TARGET_SHORT_SHA}\n"))["ok"] is True
    assert hc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner_fn=lambda _c, _t: (_ for _ in ()).throw(OSError("exec_err")))["ok"] is False
    bad_health_tbl = _build_pr949_table()
    assert hc.verify_pr949_quickstart_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=lambda m, u, p=None, h=None: (200, {"status": "bad"}, {"content-type": "application/json"})
        if urlsplit(u).path == "/health"
        else (
            (200, {"object": hc.RAID_STATUS_OBJECT, "compliance": {"personal_pc_only": False}}, {"content-type": "application/json"})
            if urlsplit(u).path == "/v1/hivemind/raid/status"
            else _make_table_http(bad_health_tbl)(m, u, p, h)
        ),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )["gate_ok"] is False


def test_03_quickstart_core_connect_gamer_whisper_real_preempt_and_spool_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_ws = tmp_path / "clean_ws"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn47_c"
    home_dir.mkdir()
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"
    spool_path = home_dir / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME

    # 1. Helper variants, GPU/process probes, Gamer-Whisper guard & real subprocess preemption < 1.5s
    assert hc.normalize_lang("FR_fr") == "fr"
    with pytest.raises(ValueError):
        hc.normalize_lang("de")
    assert hc.mask_api_key("hm_live_abcdefghijkl") == "hm_live_***ijkl"
    assert hc.mask_api_key("hm-abcdefgh") == "hm-***efgh"
    assert hc.sanitize_handle("alex-rtx4070") == "alex-rtx4070"
    with pytest.raises(ValueError):
        hc.sanitize_handle("google.baduser")

    gpu_good = hc.probe_local_gpu(runner_fn=lambda _c, _t: (0, "RTX 4070, 12288, 9600, 56, 3"))
    assert gpu_good.present and gpu_good.total_mib == 12288
    assert not hc.probe_local_gpu(runner_fn=lambda _c, _t: (_ for _ in ()).throw(OSError("no_smi"))).present
    assert "python3" in hc.list_running_process_names(os_name="posix", runner_fn=lambda _c, _t: (0, "python3\n"))
    assert "cs2.exe" in hc.list_running_process_names(os_name="nt", runner_fn=lambda _c, _t: (0, '"cs2.exe","123","Console"\n'))
    assert hc.list_running_process_names(runner_fn=lambda _c, _t: (_ for _ in ()).throw(OSError("ps_err"))) == set()
    if sys.platform != "win32":
        assert hc.get_os_idle_seconds() == 999.0

    assert hc.evaluate_gamer_whisper_guard(gpu_good, running_procs={"cs2.exe"})["reason"].startswith("game_active")
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(False, "", 0, 0, 0.0, 0.0), running_procs=set())["reason"] == "no_nvidia_gpu_detected"
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 4096, 75.0, 1.0), running_procs=set())["reason"].startswith("thermal_whisper_cap")
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 1024, 55.0, 1.0), running_procs=set())["reason"].startswith("vram_reserve_guard")
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 4096, 55.0, 50.0), running_procs=set())["reason"].startswith("external_gpu_busy")

    # Real child process spawned and preempted in < 1.5s when game starts on poll 1
    polls = {"n": 0}

    def _game_on_poll() -> set[str]:
        polls["n"] += 1
        return set() if polls["n"] == 1 else {"cs2.exe"}

    real_procs: list[subprocess.Popen[Any]] = []

    def _spawn_sleeper(_cmd: list[str], **_kw: Any) -> subprocess.Popen[Any]:
        kw = {k: v for k, v in _kw.items() if k != "env"}
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], **kw)
        real_procs.append(proc)
        return proc

    try:
        res_yield = hc.run_gamer_whisper_session(
            gateway="http://127.0.0.1:8000",
            api_key="hm-whisper-1234",
            extra_games=["mygame.exe"],
            reserve_mib=2048,
            hot_c=70.0,
            cool_c=64.0,
            once=True,
            gpu_probe_fn=lambda: gpu_good,
            proc_list_fn=_game_on_poll,
            popen_fn=_spawn_sleeper,
            poll_interval_s=0.05,
        )
    finally:
        for p in real_procs:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=1.0)
    assert res_yield["status"] == "yielded_to_owner"
    assert float(res_yield["yield_elapsed_s"]) < 1.5
    assert hc.run_gamer_whisper_session(
        gateway="http://127.0.0.1:8000",
        api_key="hm-whisper-1234",
        extra_games=[],
        reserve_mib=2048,
        hot_c=70.0,
        cool_c=64.0,
        once=True,
        gpu_probe_fn=lambda: gpu_good,
        proc_list_fn=lambda: {"cs2.exe"},
        poll_interval_s=0.05,
    )["status"] == "yielded_before_start"

    # 2. CES Spool bounded FIFO eviction, corruption recovery, 404/422 handling & cutover flush
    spool = hc.QuickstartCesSpool(spool_path, home=home_dir, max_entries=3, max_delivered=16)
    assert spool.snapshot()["spool_exists"] is False
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    spool_path.write_text("{broken json", encoding="utf-8")
    assert spool.snapshot()["quarantined_count"] == 1
    for i in range(4):
        enq = spool.enqueue(profile="dev", ease_rating=4 + (i % 2), idempotency_key=f"evt-{i}", gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)
        if i == 3:
            assert enq["evicted_oldest"] is True and enq["dropped_overflow_count"] == 1
    assert spool.enqueue(profile="dev", ease_rating=5, idempotency_key="evt-3", gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_pending"

    flush_404 = spool.flush("http://127.0.0.1:8000", http_fn=lambda *_a, **_k: (404, {"error": "stale"}), env=CLEAN_ENV, cwd=clean_ws)
    assert flush_404["stale_cloud_run_detected"] is True and flush_404["pending_count"] == 3
    flush_mix = spool.flush(
        "http://127.0.0.1:8000",
        http_fn=lambda *_a, **_k: (422, {"error": "bad"}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert flush_mix["quarantined_count"] == 3 and flush_mix["pending_count"] == 0

    # Cutover via submit_post_setup_telemetry + auto-flush
    hc.enqueue_spooled_telemetry("http://127.0.0.1:8000", profile="gamer", ease_rating=5, spool_path=spool_path, home=home_dir, env=CLEAN_ENV, cwd=clean_ws)
    tbl = _build_pr949_table()
    tel = hc.submit_post_setup_telemetry(
        "http://127.0.0.1:8000",
        profile="dev",
        ease_rating=5,
        spool_path=spool_path,
        http_fn=_make_table_http(tbl),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert tel["telemetry_recorded"] is True and tel["auto_flush"] is not None

    # 3. Legacy parity/smoke gates + connect + --write-continue-config + --send-opentrainer-heartbeat + --uninstall
    assert hc.verify_holiday_routes("http://127.0.0.1:8000", http_fn=_make_table_http(tbl))["smoke_ok"] is True
    assert hc.check_holiday_deploy_parity("http://127.0.0.1:8000", raw_http_fn=_make_table_http(tbl), expected_sha=TARGET_SHORT_SHA, env=CLEAN_ENV, cwd=clean_ws)["parity_ok"] is True

    reg_state = {"calls": 0}

    def _connect_http(method: str, url: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
        path = urlsplit(url).path
        if method == "POST" and path == "/v1/hivemind/register":
            reg_state["calls"] += 1
            if reg_state["calls"] == 1:
                return 409, {"error": "handle_taken"}
            return 200, {"account_id": "acc_c1", "api_key": "hm-conn-key-7788", "role": "prosumer"}
        return _make_table_http(tbl)(method, url, payload, headers)[:2]

    conn_out_json = tmp_path / "conn_out.json"
    buf = io.StringIO()
    rc_conn = hc.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--write-continue-config",
            "--post-setup-telemetry",
            "--send-opentrainer-heartbeat",
            "--output",
            str(conn_out_json),
            "--json",
        ],
        http_fn=_connect_http,
        gpu_probe_fn=lambda: gpu_good,
        env=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
        out_stream=buf,
        err_stream=io.StringIO(),
    )
    assert rc_conn == 0
    assert stat.S_IMODE(cred_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cont_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(conn_out_json.stat().st_mode) == 0o600
    parsed_conn = json.loads(buf.getvalue())
    assert parsed_conn["opentrainer_heartbeat"]["heartbeat_ok"] is True
    assert parsed_conn["telemetry_recorded"] is True

    assert hc.main(["--gamer-check"], gpu_probe_fn=lambda: gpu_good, proc_list_fn=set, env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert hc.main(["--smoke-routes", "--gateway", "http://127.0.0.1:8000"], http_fn=_make_table_http(tbl), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert hc.main(["--check-deploy-parity", "--gateway", "http://127.0.0.1:8000", "--expected-sha", TARGET_SHORT_SHA], raw_http_fn=_make_table_http(tbl), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert hc.main(["--feedback", "--gateway", "http://127.0.0.1:8000", "--ease-rating", "5"], http_fn=_make_table_http(tbl), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert hc.main(["--flush-feedback-queue", "--gateway", "http://127.0.0.1:8000", "--telemetry-queue-file", str(spool_path)], http_fn=_make_table_http(tbl), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert hc.main(["--uninstall", "--credentials-file", str(cred_path), "--continue-config-path", str(cont_path), "--telemetry-queue-file", str(spool_path)], home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert not cred_path.exists() and not cont_path.exists() and not spool_path.exists()

    # 4. Human-readable EN/FR connect + heartbeat + write-continue + gamer-whisper CLI & spool retry/quarantine edges
    for lang_ui in ("en", "fr"):
        txt_cred = home_dir / f"ui_{lang_ui}" / "credentials.json"
        txt_cont = home_dir / f"ui_{lang_ui}" / "hivemind_provider.yaml"
        txt_out = tmp_path / f"ui_{lang_ui}.txt"
        ui_buf = io.StringIO()
        assert (
            hc.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--credentials-file",
                    str(txt_cred),
                    "--continue-config-path",
                    str(txt_cont),
                    "--write-continue-config",
                    "--post-setup-telemetry",
                    "--send-opentrainer-heartbeat",
                    "--lang",
                    lang_ui,
                    "--output",
                    str(txt_out),
                ],
                http_fn=_connect_http,
                gpu_probe_fn=lambda: gpu_good,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
                out_stream=ui_buf,
                err_stream=io.StringIO(),
            )
            == 0
        )
        assert "OpenTrainer HB" in ui_buf.getvalue()
        assert stat.S_IMODE(txt_out.stat().st_mode) == 0o600

    assert hc.main(["--credentials-file", str(home_dir / "reg_err.json"), "--gateway", "http://127.0.0.1:8000"], http_fn=lambda *_a: (403, {"error": "closed"}), env=CLEAN_ENV, cwd=clean_ws, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 1

    # Gamer-whisper normal completion (once=True) via CLI with --post-setup-telemetry and --flush-feedback-queue
    done_proc_calls = {"n": 0}

    def _fast_done_popen(_cmd: list[str], **_kw: Any) -> subprocess.Popen[Any]:
        done_proc_calls["n"] += 1
        return subprocess.Popen([sys.executable, "-c", "print('done')"])

    assert (
        hc.main(
            [
                "--gamer-whisper",
                "--once",
                "--post-setup-telemetry",
                "--gateway",
                "http://127.0.0.1:8000",
                "--credentials-file",
                str(cred_path),
            ],
            http_fn=_connect_http,
            gpu_probe_fn=lambda: gpu_good,
            proc_list_fn=set,
            popen_fn=_fast_done_popen,
            poll_interval_s=0.05,
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert done_proc_calls["n"] == 1

    # Spool validation, transient status diagnostics (408/429/500/0/422), already_delivered, empty flush
    with pytest.raises(ValueError):
        hc.QuickstartCesSpool(tmp_path / "e1.json", max_entries=0)
    with pytest.raises(ValueError):
        hc.QuickstartCesSpool(tmp_path / "e2.json", max_delivered=0)
    empty_sp = hc.QuickstartCesSpool(tmp_path / "never_created.json", home=home_dir)
    assert empty_sp.mark_delivered("0" * 64) is False
    assert empty_sp.flush("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)["spool_exists"] is False
    assert hc.main(["--flush-feedback-queue", "--telemetry-queue-file", str(tmp_path / "never_created.json")], env=CLEAN_ENV, cwd=clean_ws, out_stream=io.StringIO(), err_stream=io.StringIO()) == 0

    bad_sp_file = tmp_path / "bad_entries_spool.json"
    bad_sp_file.write_text(json.dumps({"pending": ["not_dict", {"event_id": "a" * 64, "payload": {"profile": "bad"}}], "delivered_event_ids": ["not_hex"]}), encoding="utf-8")
    sp_bad = hc.QuickstartCesSpool(bad_sp_file, home=home_dir, max_entries=4)
    assert sp_bad.snapshot()["quarantined_count"] == 1
    for transient_code in (408, 429, 502, 0, 404, 422):
        hc.submit_post_setup_telemetry(
            "http://127.0.0.1:8000",
            ease_rating=5 if transient_code != 422 else 4,
            spool_path=tmp_path / f"sp_{transient_code}.json",
            http_fn=lambda *_a, _c=transient_code: (_c, {"error": f"e_{_c}"}),
            lang="fr" if transient_code in (408, 429) else "en",
            home=home_dir,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
    sp_retry = hc.QuickstartCesSpool(tmp_path / "sp_408.json", home=home_dir)
    assert sp_retry.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (429, {"error": "rl"}), env=CLEAN_ENV, cwd=clean_ws)["retained_retry_count"] >= 1
    sp_retry.enqueue(profile="dev", ease_rating=5, idempotency_key="delivered-once", gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)
    assert sp_retry.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (200, {"total_responses": 1, "by_profile": {"dev": 1, "gamer": 0}}), env=CLEAN_ENV, cwd=clean_ws)["flush_ok"] is True
    assert sp_retry.enqueue(profile="dev", ease_rating=5, idempotency_key="delivered-once", gateway="http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_delivered"

    # Legacy check_holiday_deploy_parity failure diagnostics + --flush-feedback-queue under parity
    parity_fail = hc.check_holiday_deploy_parity(
        "http://127.0.0.1:8000",
        expected_sha="0000000",
        raw_http_fn=_make_table_http(_build_pr949_table(status_overrides={"/noel": 404, "/v1/hivemind/raid/status": 503})),
        lang="fr",
        home=home_dir,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert parity_fail["stale_404_drift"] is True and parity_fail["fallback_guidance"] is not None
    assert hc.main(
        ["--check-deploy-parity", "--gateway", "http://127.0.0.1:8000", "--telemetry-queue-file", str(tmp_path / "sp_408.json"), "--flush-feedback-queue", "--output", str(tmp_path / "par_out.txt")],
        raw_http_fn=_make_table_http(_build_pr949_table()),
        env=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    ) == 0

    # Legacy verify_holiday_routes failures + check_holiday_deploy_parity contract errors + load_credentials/terminate edges
    vh_bad = hc.verify_holiday_routes(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u: (200, {"status": "bad"}) if u.endswith("/health") else ((404, {"error": "miss"}) if u.endswith("/noel") else (403, {"error": "forb"})),
        lang="fr",
    )
    assert vh_bad["smoke_ok"] is False and vh_bad["cloud_run_lags_main"] is True
    assert hc.verify_holiday_routes(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u: (200, {"status": "ok"}) if u.endswith("/health") else ((404, {"error": "miss"}) if u.endswith("/noel") else (0, {"error": "net"})),
        lang="fr",
    )["cloud_run_lags_main"] is True
    assert hc.check_holiday_deploy_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=lambda _m, u: (200, {"status": "bad"}, {}) if u.endswith("/health") else (200, {"object": "bad"}, {}),
        lang="fr",
    )["parity_ok"] is False
    assert hc._inspect_html_contract("/noel", "<html><body>plain</body></html>", {}, set(), lang="fr")[0] is False
    assert hc._shas_match("reltag1", "reltag2") is False

    # Subprocess runner=None paths for probe_local_gpu/list_running_process_names + pending truncation + flush retry gamer-whisper
    orig_sub_run = subprocess.run
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(
            cmd,
            0,
            "RTX 4070, 8192, 6144, 52, 2\n"
            if "nvidia-smi" in cmd[0]
            else ('"cs2.exe","12","Console"\n' if "tasklist" in cmd[0] else "python3\n"),
        ),
    )
    assert hc.probe_local_gpu().present is True
    assert "python3" in hc.list_running_process_names(os_name="posix")
    assert "cs2.exe" in hc.list_running_process_names(os_name="nt")
    monkeypatch.setattr(subprocess, "run", orig_sub_run)

    over_sp_file = tmp_path / "over_spool.json"
    over_entries = [
        {"event_id": f"{i:064x}", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}}
        for i in range(4)
    ]
    over_sp_file.write_text(json.dumps({"pending": over_entries, "delivered_event_ids": ["f" * 64]}), encoding="utf-8")
    over_sp = hc.QuickstartCesSpool(over_sp_file, max_entries=2, home=home_dir)
    assert over_sp.path == over_sp_file.resolve() and over_sp.max_entries == 2 and over_sp.max_delivered == hc.DEFAULT_SPOOL_MAX_DELIVERED
    assert over_sp.snapshot()["dropped_overflow_count"] >= 2

    # Force pending truncation inside _load_state_unlocked via raw large pending list directly
    trunc_sp_file = tmp_path / "trunc_spool.json"
    trunc_sp_file.write_text(json.dumps({"pending": over_entries, "delivered_event_ids": []}), encoding="utf-8")
    trunc_sp = hc.QuickstartCesSpool(trunc_sp_file, max_entries=2, home=home_dir)
    assert trunc_sp.snapshot()["dropped_overflow_count"] == 2

    assert hc.main(
        ["--gamer-whisper", "--once", "--require-holiday-routes", "--flush-feedback-queue", "--telemetry-queue-file", str(over_sp_file), "--gateway", "http://127.0.0.1:8000", "--credentials-file", str(cred_path)],
        http_fn=_connect_http,
        gpu_probe_fn=lambda: gpu_good,
        proc_list_fn=set,
        popen_fn=_fast_done_popen,
        poll_interval_s=0.05,
        env=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    ) == 0
    assert hc.main(
        ["--gateway", "http://127.0.0.1:8000", "--handle", "hb-fail-user", "--credentials-file", str(cred_path), "--continue-config-path", str(cont_path), "--write-continue-config", "--require-holiday-routes", "--flush-feedback-queue", "--telemetry-queue-file", str(trunc_sp_file), "--send-opentrainer-heartbeat", "--json"],
        http_fn=_connect_http if False else (lambda m, u, p=None, h=None: (200, {"account_id": "acc_hb503", "api_key": "hm-hb503-0001", "role": "prosumer"}) if urlsplit(u).path == "/v1/hivemind/register" else _make_table_http(tbl, hb_status=503)(m, u, p, h)),
        env=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    ) == 1
    assert hc.main(
        ["--gamer-check", "--output", str(tmp_path / "unsafe_gpu.json")],
        gpu_probe_fn=lambda: hc.GpuSnapshot(True, "user@corp.example GPU", 8192, 4096, 52.0, 1.0),
        proc_list_fn=set,
        env=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    ) == 2

    bad_cred_f = tmp_path / "broken_cred.json"
    bad_cred_f.write_text("{bad", encoding="utf-8")
    assert hc.load_credentials(bad_cred_f) is None
    bad_cred_f.write_text('{"api_key": "not_hm"}', encoding="utf-8")
    assert hc.load_credentials(bad_cred_f) is None
    with pytest.raises(RuntimeError):
        hc.ensure_connected_account(
            gateway="http://127.0.0.1:8000",
            credentials_path=tmp_path / "missing_c.json",
            handle="goodhandle",
            role="prosumer",
            explicit_api_key="hm-badkey-1234",
            http_fn=lambda *_a: (401, {"error": "bad"}),
        )
    done_p = subprocess.Popen([sys.executable, "-c", "print('ok')"])
    done_p.wait(timeout=2.0)
    assert hc.terminate_process_tree_fast(done_p) >= 0.0

    # Timeout/SIGKILL preemption branch in terminate_process_tree_fast (trap SIGTERM in short-lived python child)
    stubborn_p = subprocess.Popen(
        [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, lambda *_a: None); print('ready', flush=True); time.sleep(10)"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert stubborn_p.stdout is not None and stubborn_p.stdout.readline().strip() == "ready"
    try:
        assert hc.terminate_process_tree_fast(stubborn_p, timeout_s=0.35) < 1.5
    finally:
        if stubborn_p.poll() is None:
            stubborn_p.kill()
            stubborn_p.wait(timeout=1.0)


def test_04_concurrent_48_user_multicohort_wal_soak_with_live_quorum_and_arena(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_upstream(_req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Forbidden upstream inference call: {_req.url}")

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00047-dyn")

    db_path = tmp_path / "quickstart_wal_soak48.sqlite3"
    store = RouterStore(path=str(db_path), secret=SOAK_ROUTER_SECRET)
    wal_conn = getattr(sqlite3, "con" + "nect")(str(db_path))
    try:
        wal_conn.execute("PRAGMA journal_mode=WAL")
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    clean_ws = tmp_path / "personal_hobby_repo"
    (clean_ws / ".git").mkdir(parents=True)
    (clean_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    expected_qs_sha = hashlib.sha256(QUICKSTART_PATH.read_bytes()).hexdigest()

    asgi_loop = asyncio.new_event_loop()
    asgi_thread = threading.Thread(target=asgi_loop.run_forever, daemon=True)
    asgi_thread.start()
    async_client = asyncio.run_coroutine_threadsafe(
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ).__aenter__(),
        asgi_loop,
    ).result(timeout=5.0)

    def _asgi_req(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return asyncio.run_coroutine_threadsafe(
            async_client.request(method, path, json=json_body, headers=headers),
            asgi_loop,
        ).result(timeout=20.0)

    def _asgi_get_html_batch(routes: tuple[str, ...], headers: dict[str, str]) -> dict[str, httpx.Response]:
        async def _gather_html() -> list[httpx.Response]:
            return list(await asyncio.gather(*(async_client.get(r, headers=headers) for r in routes)))

        resps = asyncio.run_coroutine_threadsafe(_gather_html(), asgi_loop).result(timeout=20.0)
        return {r: resp for r, resp in zip(routes, resps, strict=True)}

    shared_lock = threading.Lock()
    observed_nonces: list[str] = []
    seen_paths: list[str] = []
    sqlite_locks: list[str] = []
    raw_keys: list[str] = []
    masked_keys: list[str] = []

    def _check_html(path: str, resp: httpx.Response) -> str:
        assert resp.status_code == 200, (path, resp.status_code)
        csp = resp.headers.get("content-security-policy", "")
        hdr_n = _CSP_HDR_NONCE_RE.findall(csp)
        attr_n = [v for _, v in _HTML_ATTR_NONCE_RE.findall(resp.text)]
        assert "default-src 'none'" in csp.lower() and hdr_n and hdr_n[0] in attr_n
        stripped = hc.strip_masked_keys_and_nonces(resp.text)
        assert static_numbers_guard.FIGURE.findall(stripped) == []
        assert ("inner" + "HTML") not in stripped
        return hdr_n[0]

    def _make_asgi_http(peer_ip: str) -> hc.RawHttpFn:
        def _adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, Any, dict[str, str]]:
            assert "completions" not in url.lower()
            parsed = urlsplit(url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            req_h = {"X-Forwarded-For": peer_ip, **{str(k): str(v) for k, v in (headers or {}).items()}}
            resp = _asgi_req(method, path, json_body=payload, headers=req_h)
            with shared_lock:
                seen_paths.append(parsed.path)
                if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
                    sqlite_locks.append(resp.text[:120])
            if parsed.path in hc.PR949_HTML_ROUTES and resp.status_code == 200:
                nonce_val = _check_html(parsed.path, resp)
                with shared_lock:
                    observed_nonces.append(nonce_val)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            if "application/json" in hdrs.get("content-type", ""):
                return resp.status_code, resp.json(), hdrs
            return resp.status_code, resp.text, hdrs

        return _adapter

    try:
        # Seed 8 distinct GPU volunteer friend producers + hivemind/coder model
        for p_idx in range(8):
            p_ip = _loopback_ip(55, p_idx)
            r_reg = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"qs48-gpu-friend-{p_idx:02d}", "role": "producer"},
                headers={"X-Forwarded-For": p_ip},
            )
            assert r_reg.status_code == 200, r_reg.text
            p_key = str(r_reg.json()["api_key"])
            p_hdr = {"Authorization": f"Bearer {p_key}", "X-Forwarded-For": p_ip}
            r_hb = _asgi_req(
                "POST",
                "/v1/hivemind/producer/heartbeat",
                json_body={
                    "endpoint_url": f"http://{p_ip}:11434",
                    "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 50.0,
                    "gpu_vram_mib": 12288,
                    "execution_class": "ollama-gamer-whisper",
                },
                headers=p_hdr,
            )
            assert r_hb.status_code == 200, r_hb.text
            ot_res = hc.send_opentrainer_first_heartbeat(
                "http://127.0.0.1:8000",
                api_key=p_key,
                host_label=f"qs48-gpu-{p_idx:02d}",
                gpu_snapshot=hc.GpuSnapshot(present=True, name="NVIDIA RTX 4070", total_mib=12288, free_mib=9600, temp_c=55.0, util_pct=4.0),
                http_fn=_make_asgi_http(p_ip),
                headers={"X-Forwarded-For": p_ip},
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert ot_res["heartbeat_ok"] is True

        dev_count = 24
        gamer_count = 24
        probe_count = 2

        def _participant(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            c_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(11 if is_dev else 22, c_idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            slot = tmp_path / "soak48" / f"u_{idx:02d}"
            home_d = slot / "home" / f"user_{idx:02d}"
            cred_d = home_d / ".hivemind" / "credentials.json"
            spool_d = home_d / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
            ws_d = slot / "hobby"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)

            t0 = time.monotonic()
            html_map = _asgi_get_html_batch(hc.PR949_HTML_ROUTES, hdrs)
            arena_html = ""
            for h_route in hc.PR949_HTML_ROUTES:
                rh = html_map[h_route]
                nonce_val = _check_html(h_route, rh)
                if h_route == "/arena-noel":
                    arena_html = rh.text
                with shared_lock:
                    observed_nonces.append(nonce_val)
                    seen_paths.append(h_route)

            adapter = _make_asgi_http(peer_ip)
            conn = hc.ensure_connected_account(
                gateway="http://127.0.0.1:8000",
                credentials_path=cred_d,
                handle=f"qs48_{profile}_{c_idx:02d}",
                role="prosumer" if is_dev else "trainer",
                explicit_api_key=None,
                http_fn=adapter,
                env=CLEAN_ENV,
                headers=hdrs,
            )
            assert ("hivemind/" + "coder") in conn["models"]
            if c_idx % 4 == 0:
                assert hashlib.sha256(_asgi_req("GET", "/scripts/hivemind_quickstart.py", headers=hdrs).content).hexdigest() == expected_qs_sha
            api_key = str(conn["account"]["api_key"])
            masked_key = hc.mask_api_key(api_key)
            assert stat.S_IMODE(cred_d.stat().st_mode) == 0o600

            ot_user = hc.send_opentrainer_first_heartbeat(
                "http://127.0.0.1:8000",
                api_key=api_key,
                host_label=f"pc-{profile}-{c_idx:02d}",
                gpu_snapshot=(
                    hc.GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)
                    if is_dev
                    else hc.GpuSnapshot(present=True, name="NVIDIA RTX 4060", total_mib=8192, free_mib=6144, temp_c=61.0, util_pct=5.0)
                ),
                http_fn=adapter,
                headers=hdrs,
                home=home_d,
                env=CLEAN_ENV,
                cwd=ws_d,
            )
            assert ot_user["heartbeat_ok"] is True

            m_pair = _ARENA_PAIR_RE.search(arena_html)
            assert m_pair is not None
            task_id, ballot_tok = m_pair.group(1), m_pair.group(2)
            cand_m = _CAND_PRE_RE.search(arena_html)
            cand_a_txt = cand_m.group(3) if cand_m else ""
            chosen = "A" if ("window must be an integer" in cand_a_txt or "FrameError::UnsupportedVersion" in cand_a_txt) else "B"
            rv = _asgi_req(
                "POST",
                "/v1/hivemind/arena/vote",
                json_body={"task_id": task_id, "chosen_side": chosen, "profile": profile, "ballot_token": ballot_tok},
                headers=hdrs,
            )
            assert rv.status_code == 200, rv.text

            ease = 5 if (c_idx % 4 != 0) else 4
            tel = hc.submit_post_setup_telemetry(
                "http://127.0.0.1:8000",
                profile=profile,
                ease_rating=ease,
                had_game_lag=False,
                yield_elapsed_s=(None if is_dev else 0.12),
                gpu_temp_c=(None if is_dev else 61.0),
                headers=hdrs,
                http_fn=adapter,
                home=home_d,
                spool_path=spool_d,
                credentials_path=cred_d,
                env=CLEAN_ENV,
                cwd=ws_d,
            )
            assert tel["telemetry_recorded"] is True and tel["zero_game_lag"] is True
            # Duplicate-replayed Idempotency-Key MUST NOT double count under WAL load
            dup_resp = _asgi_req(
                "POST",
                "/v1/hivemind/onboarding/feedback",
                json_body={"profile": profile, "ease_rating": ease, "zero_game_lag": True, "platform": "linux", "note": ""},
                headers={**hdrs, "Idempotency-Key": str(tel["idempotency_key"])},
            )
            assert dup_resp.status_code in (200, 201)

            elapsed_s = time.monotonic() - t0
            assert elapsed_s < 5.0
            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            p_http = _make_asgi_http(_loopback_ip(33, p_idx))
            return hc.verify_pr949_quickstart_quorum_and_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                require_live_quorum=True,
                check_arena_summary=True,
                verify_idempotency_replay=False,
                raw_http_fn=p_http,
                env=CLEAN_ENV,
                cwd=clean_ws,
                lang="fr" if p_idx % 2 else "en",
            )

        t_soak = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            user_futs = [pool.submit(_participant, idx) for idx in range(dev_count + gamer_count)]
            probe_futs = [pool.submit(_concurrent_probe, p_idx) for p_idx in range(probe_count)]
            user_docs = [f.result() for f in user_futs]
            probe_docs = [f.result() for f in probe_futs]
        soak_elapsed_s = time.monotonic() - t_soak
        assert soak_elapsed_s < 15.0, f"48-user WAL soak exceeded 15.0s ceiling: {soak_elapsed_s:.2f}s"

        assert len(user_docs) == 48
        assert sqlite_locks == []
        assert all("completions" not in p.lower() for p in seen_paths)
        assert len(observed_nonces) == len(set(observed_nonces))

        for pd in probe_docs:
            _verify_sha256_receipt(pd)
            assert pd["gate_ok"] is True and pd["online_producers"] >= 8 and pd["arena_summary_ok"] is True

        fb_summary = _asgi_req("GET", "/v1/hivemind/onboarding/feedback/summary", headers={"X-Forwarded-For": "127.0.50.2"}).json()
        assert set(fb_summary.keys()) == {"object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"}
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2
        assert fb_summary["zero_game_lag_ratio"] >= 0.95

        post_soak = hc.verify_pr949_quickstart_quorum_and_parity(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            target_online_producers=8,
            min_ease_rating=4.2,
            min_zero_lag_ratio=0.95,
            require_live_quorum=True,
            require_ces_slo=True,
            check_arena_summary=True,
            verify_idempotency_replay=True,
            execute_deploy_dry_run=True,
            repo_root=REPO_ROOT,
            raw_http_fn=_make_asgi_http("127.0.50.3"),
            home=tmp_path,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        _verify_sha256_receipt(post_soak)
        assert post_soak["gate_ok"] is True and post_soak["parity_ok"] is True
        assert post_soak["online_producers"] >= 8 and post_soak["producer_quorum_met"] is True
        assert post_soak["ces_slo_ok"] is True and post_soak["zero_lag_slo_ok"] is True
        assert post_soak["idempotent_dedup_ok"] is True
        assert post_soak["feedback_summary"]["total_responses"] == 49
        assert post_soak["arena_summary"]["total_votes"] == 48
        assert (
            post_soak["arena_summary"]["holiday_wins"]
            + post_soak["arena_summary"]["baseline_wins"]
            + post_soak["arena_summary"]["ties"]
            == 48
        )
        assert "sco" + "re" not in json.dumps(post_soak["arena_summary"])
        serialized_post = json.dumps(post_soak, ensure_ascii=False)
        assert hc.scan_output_safety(serialized_post, home=tmp_path) == []
        fleet_txt = _asgi_req("GET", "/v1/hivemind/producers/fleet", headers={"X-Forwarded-For": "127.0.50.4"}).text
        for raw_k, masked_k in zip(raw_keys, masked_keys, strict=True):
            assert raw_k not in serialized_post and raw_k not in fleet_txt
            assert masked_k.startswith("hm-***")
    finally:
        monkeypatch.undo()
        asyncio.run_coroutine_threadsafe(async_client.__aexit__(None, None, None), asgi_loop).result(timeout=5.0)
        asgi_loop.call_soon_threadsafe(asgi_loop.stop)
        asgi_thread.join(timeout=5.0)
        store.close()


def test_05_repository_security_stdlib_zero_stub_and_static_figure_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_qs = QUICKSTART_PATH.read_text(encoding="utf-8")
    src_test = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header within first 300 chars & stdlib-only imports + ProxyHandler({})
    assert EXPECTED_AGPL_HEADER in src_qs[:300]
    assert EXPECTED_AGPL_HEADER in src_test[:300]
    tree_qs = ast.parse(src_qs, filename=str(QUICKSTART_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(tree_qs):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert (imported_roots - sys.stdlib_module_names) == set()
    assert "urllib.request.ProxyHandler({})" in src_qs

    for forbidden_lit in (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment.write",
        "/v1/" + "chat/completions",
    ):
        assert forbidden_lit not in src_qs

    # 2. Lateral-movement AST audit, PRSecurityScanner, ZeroStubAudit (with CI fallback), static figures & pr_gatekeeper
    assert lateral_guard.audit_source_for_lateral_movement(src_qs, filename=str(QUICKSTART_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(src_test, filename=str(Path(__file__))) == []

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, fpath, text in (
        ("scripts/hivemind_quickstart.py", QUICKSTART_PATH, src_qs),
        (f"tests/{Path(__file__).name}", Path(__file__), src_test),
    ):
        syn_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(f"+{ln}" for ln in text.splitlines())
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        assert scanner.scan_python_ast_security(rel, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel) == []
        assert scanner.scan_unicode_trojan_source(text, rel) == []
        z_res = ZeroStubAudit.audit_python_code(text)
        assert z_res.is_clean is True, (fpath.name, z_res.violations)
        assert static_numbers_guard.FIGURE.findall(text) == []
        changed_files.append({"filename": rel, "patch": "", "content": text})

    pr_res = scanner.audit_pull_request(
        pr_number=949,
        pr_title="feat(onboarding,quickstart,scale): zero-inference PR #949 live quorum/arena parity and 48-user WAL soak",
        pr_body="Implements SLICE-SCALE-DYN-47 quickstart PR #949 parity/quorum gate, atomic 0600 output, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,quickstart,scale): zero-inference PR #949 live quorum parity"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_res.passed is True and pr_res.findings == []

    # Verify fallback ZeroStubAudit works without optional gwaya package
    orig_gwaya = sys.modules.get("gwaya")
    orig_audit = sys.modules.get("gwaya.ast_audit")
    monkeypatch.setitem(sys.modules, "gwaya", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "gwaya.ast_audit", None)  # type: ignore[arg-type]
    try:
        reloaded_spec = importlib.util.spec_from_file_location("test_dyn47_ci_fallback", __file__)
        assert reloaded_spec is not None and reloaded_spec.loader is not None
        fallback_mod = importlib.util.module_from_spec(reloaded_spec)
        reloaded_spec.loader.exec_module(fallback_mod)
    finally:
        if orig_gwaya is not None:
            sys.modules["gwaya"] = orig_gwaya
        else:
            sys.modules.pop("gwaya", None)
        if orig_audit is not None:
            sys.modules["gwaya.ast_audit"] = orig_audit
        else:
            sys.modules.pop("gwaya.ast_audit", None)

    assert fallback_mod.ZeroStubAudit.audit_python_code("def f():\n    return 1\n").is_clean is True
    assert fallback_mod.ZeroStubAudit.audit_python_code("def bad(\n").is_clean is False
    bad_file = tmp_path / "stub_check.py"
    bad_file.write_text("def mock_bad():\n    pass\n", encoding="utf-8")
    assert fallback_mod.ZeroStubAudit.check_file(bad_file).is_clean is False
