# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/arena parity, Idempotency-Key replay, SSRF/TLS guard, 0600 spool & 48-user WAL soak tests for hivemind_connect (`SLICE-SCALE-DYN-49`)."""

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
CONNECT_PATH = REPO_ROOT / "scripts" / "hivemind_connect.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import directly without starting an in-process coverage tracer so external
# `coverage run --rcfile=/dev/null --include='*/scripts/hivemind_connect.py'` works cleanly.
from scripts import hivemind_connect as hc  # noqa: E402

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
        """CI-safe AST zero-stub fallback when optional ``gwaya`` package is absent."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mo" + r"ck_", re.IGNORECASE),
            re.compile(r"^fa" + r"ke_", re.IGNORECASE),
            re.compile(r"^simu" + r"late_", re.IGNORECASE),
            re.compile(r"^dum" + r"my_", re.IGNORECASE),
            re.compile(r"^st" + r"ub_", re.IGNORECASE),
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
    "test_no_lateral_movement_dyn49",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn49",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn49",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
TARGET_FULL_SHA = "c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b90123"
TARGET_SHORT_SHA = TARGET_FULL_SHA[:12]
SOAK_ROUTER_SECRET = b"noel-scale-dyn49-wal-secret-32b!"

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


def _capture_stdout_stderr(fn: Any) -> tuple[Any, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        res = fn()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return res, out_buf.getvalue(), err_buf.getvalue()


def _build_pr949_html(
    path: str,
    nonce: str,
    *,
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
    csp = f"{default_part}script-src {inline_part}'nonce-{hdr_nonce}'; style-src 'nonce-{hdr_nonce}'"
    doc_prefix = "<!DOCTYPE html>\n" if include_doctype else ""
    banner = (
        "<p>100% personal home PCs &amp; hobby projects only • 0.00 EUR / 0.00 USD • AGPL-3.0</p>"
        if include_banner
        else "<p>HiveMind generic portal</p>"
    )
    html = (
        f'{doc_prefix}<html lang="en"><head><title>{path}</title></head>'
        f"<body>{banner}{extra_body}"
        f'<script nonce="{script_nonce}">document.body.dataset.path="{path}";</script>'
        "</body></html>"
    )
    return html, {"content-type": "text/html; charset=utf-8", "content-security-policy": csp}


def _build_pr949_table(
    *,
    build_sha: str = TARGET_FULL_SHA,
    cloud_run_revision: str | None = "hivemind-openrouter-00049-dyn",
    fleet_object: str = "hivemind.producer_fleet",
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
    missing_doctype_route: str | None = None,
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
            "object": "hivemind.onboarding_feedback_summary",
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
            "holiday_routes": list(hc._MANIFEST_REQUIRED_HOLIDAY_ROUTES),
            "downloadable_scripts": ["hivemind_connect.py", "hivemind_doctor.py", "hivemind_quickstart.py"],
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
            "swarm": {"active_trainers": live_hiveminders, "accepted_opentrainer_rounds": 14},
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
        "missing_doctype_route": missing_doctype_route,
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
            nonce = table["static_nonce"] or f"Dyn49Nonce_{table['nonce_seq']:04d}_XyZ"
            extra = table["html_extra_text"]
            if table["unsafe_sink_route"] == path:
                extra += "<script>el." + "inner" + "HTML='x';</script>"
            if table["static_fig_route"] == path:
                extra += "<p>Latency " + "142.5 " + "ms</p>"
            html, hdrs = _build_pr949_html(
                path,
                nonce,
                include_doctype=(table["missing_doctype_route"] != path),
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
            return (
                200,
                {
                    "account_id": "acc_dyn49",
                    "handle": "dyn49-friend",
                    "role": me_role,
                    "balance": 10.0,
                    "max_price_per_1k": 0.0,
                },
                {"content-type": "application/json"},
            )
        if method == "GET" and path == "/v1/hivemind/catalog":
            return (
                200,
                {"tiers": [{"public_id": "hivemind/coder", "moe": False, "min_vram_gb": 8, "good_for": "coding"}]},
                {"content-type": "application/json"},
            )
        if method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
            if hb_status != 200:
                return hb_status, {"error": "hb_rejected"}, {"content-type": "application/json"}
            return (
                200,
                {
                    "object": "hivemind.opentrainer_heartbeat_ack",
                    "aglet_lifecycle_phase": "arrival",
                    "amcp_topic": "opentrainer.noel",
                    "node": {"node_id": f"dyn49-friend:{(payload or {}).get('host_label', 'personal-pc')}"},
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
                    body = b"<!DOCTYPE html><html><body>Holiday plain OK</body></html>"
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

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or "0")
                _ = self.rfile.read(length) if length > 0 else b""
                if self.path == "/v1/hivemind/onboarding/feedback":
                    body = json.dumps({"total_responses": 1, "mean_ease_rating": 5.0}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = b"Backend plain error for /home/bob/secret"
                    self.send_response(502)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        return _Handler


def test_01_ssrf_http_hardening_compliance_git_worktree_and_atomic_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Host allowlist & gateway URL validation
    for ok_host in (
        "hivemind-openrouter-1003063861791.us-central1.run.app",
        "storage.googleapis.com",
        "run.googleapis.com",
        "datasets-server.huggingface.co",
        "127.0.0.1",
        "[::1]",
        "localhost",
    ):
        assert hc.is_allowed_gateway_host(ok_host) is True
    assert hc.is_allowed_gateway_host("gw.example.org", allow_test_hosts=False) is False
    assert hc.is_allowed_gateway_host("gw.example.org", allow_test_hosts=True) is True
    assert hc.is_allowed_gateway_host("testserver", allow_test_hosts=True) is True
    for bad_host in (
        "",
        None,
        "a" * 260,
        "foo..run.app",
        "-bad.run.app",
        "bad-.run.app",
        "sub_under.run.app",
        "run.app",
        "googleapis.com",
        "foo\\bar.run.app",
        "user@127.0.0.1",
        "[127.0.0.1",
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
        "https://localhost:notaport",
        "https://localhost\r\n",
        f"https://{'bad'} {'host.run.app'}",
        _untrusted_url("evil.example"),
    ):
        with pytest.raises(hc.GatewayPolicyError):
            hc.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    # 2. Zero-inference, timeout, header CRLF/NUL, and payload guards (including points/transfer exception)
    assert (
        hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/points/transfer",
            payload={"recipient": "@bob", "amount": 1.0, "model_ref": "hivemind/" + "coder"},
        )
        == "http://127.0.0.1:8000"
    )
    comp_endpoint = "http://127.0.0.1:8000/v1/" + "chat/completions"
    for fn_call in (
        lambda: hc._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/health"),
        lambda: hc._assert_zero_inference_request("GET", comp_endpoint),
        lambda: hc._assert_zero_inference_request("POST", "http://127.0.0.1:8000/v1/chat"),
        lambda: hc._assert_zero_inference_request("GET", "http://127.0.0.1:8000/health", payload={"ok": True}),
        lambda: hc._assert_zero_inference_request("GET", "http://127.0.0.1:8000\\bad"),
        lambda: hc._assert_zero_inference_request("GET", "http://user@127.0.0.1:8000/health"),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"nested": [{"messages": [{"role": "user", "content": "hi"}]}]},
        ),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"prompt": "write code"},
        ),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"model": "hivemind/" + "coder"},
        ),
        lambda: hc._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload='{"messages":[{"role":"user","content":"hi"}]}',
        ),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=float("nan")),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=-1.0),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=True),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Test": "bad\r\nval"}),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Null": "a\x00b"}),
        lambda: hc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"": "x"}),
    ):
        with pytest.raises(hc.GatewayPolicyError):
            fn_call()

    # Direct unit test of _SafeRedirectHandler downgrade/remote->loopback/empty rules
    redir = hc._SafeRedirectHandler()
    remote_req = hc.urllib.request.Request("https://demo.a.run.app/quick")
    with pytest.raises(hc.GatewayPolicyError, match="loopback"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "https://127.0.0.1/quick")
    with pytest.raises(hc.GatewayPolicyError, match="downgrade"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "http://demo.a.run.app/quick")
    with pytest.raises(hc.GatewayPolicyError, match="empty"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {"Location": ""}, "")

    # 3. Live loopback HTTP server: ProxyHandler({}) isolation, stdlib raw + client, redirects, 4 MiB cap
    port_holder = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHttpServer(port_ref=port_holder).build_handler())
    port_holder[0] = int(srv.server_address[1])
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base = f"http://127.0.0.1:{port_holder[0]}"
        monkeypatch.setenv("HTTP_PROXY", _untrusted_url("proxy.corp.example", scheme="http") + ":8080")
        monkeypatch.setenv("HTTPS_PROXY", _untrusted_url("proxy.corp.example", scheme="http") + ":8080")

        st_ok, body_ok, hdrs_ok = hc.default_http_raw("GET", f"{base}/health", headers={"X-Client": "dyn49"})
        assert st_ok == 200 and json.loads(body_ok)["status"] == "ok" and "application/json" in hdrs_ok["content-type"]
        st_post, body_post, _ = hc.default_http_raw(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""},
        )
        assert st_post == 200 and json.loads(body_post)["total_responses"] == 1
        assert hc.default_http_raw("GET", f"{base}/empty")[0] == 204
        assert hc.default_http_raw("GET", f"{base}/empty-404")[:2] == (404, "")
        assert hc.default_http_raw("GET", f"{base}/redirect-ok")[0] == 200
        st_404, body_404, _ = hc.default_http_raw("GET", f"{base}/missing-route")
        assert st_404 == 404 and json.loads(body_404)["error"] == "missing_route"
        st_502, body_502, _ = hc.default_http_raw("GET", f"{base}/bad-err")
        assert st_502 == 502 and "/home/alice" not in body_502 and "user@" not in body_502
        for redir_path in ("/redirect-evil", "/redirect-backslash", "/redirect-completions"):
            with pytest.raises(hc.GatewayPolicyError):
                hc.default_http_raw("GET", f"{base}{redir_path}")
        with pytest.raises(hc.GatewayPolicyError, match="byte limit"):
            hc.default_http_raw("GET", f"{base}/oversized")

        with hc._StdlibHttpClient(timeout_s=2.0) as cli:
            r1 = cli.get(f"{base}/health")
            assert r1.status_code == 200 and r1.json()["status"] == "ok"
            r2 = cli.post(f"{base}/v1/hivemind/onboarding/feedback", json={"profile": "dev", "ease_rating": 5})
            assert r2.status_code == 200 and r2.json()["mean_ease_rating"] == 5.0
            r_err = cli.get(f"{base}/empty-404")
            assert r_err.status_code == 404
        # Live loopback ConnectCesRetrySpool.flush via default _StdlibHttpClient + transport path
        loopback_q = tmp_path / "loopback_std_q.json"
        hc.enqueue_connect_ces_event(queue_file=loopback_q, gateway=base, idempotency_key="loopback-std-1", environ=CLEAN_ENV, cwd=tmp_path)
        assert hc.ConnectCesRetrySpool(loopback_q, home=tmp_path).flush(base, environ=CLEAN_ENV, cwd=tmp_path)["delivered_count"] == 1
        absent_q = tmp_path / "never_created_absent_q.json"
        absent_flush = hc.ConnectCesRetrySpool(absent_q, home=tmp_path).flush(base, environ=CLEAN_ENV, cwd=tmp_path, lang="fr")
        assert absent_flush["queue_exists"] is False and absent_flush["network_calls"] == 0 and not absent_q.exists()
    finally:
        monkeypatch.undo()
        srv.shutdown()
        srv.server_close()

    st_down, body_down, hdr_down = hc.default_http_raw("GET", f"http://127.0.0.1:{port_holder[0]}/health", timeout_s=0.2)
    assert st_down == 0 and "network_error" in body_down and "x-hivemind-network-error" in hdr_down
    with pytest.raises(hc.GatewayConnectionError):
        with hc._StdlibHttpClient(timeout_s=0.2) as cli:
            cli.get(f"http://127.0.0.1:{port_holder[0]}/health")

    # _HttpxTransportClient when httpx is absent vs present
    orig_httpx = hc.httpx
    monkeypatch.setattr(hc, "httpx", None)
    with pytest.raises(RuntimeError, match="httpx"):
        hc._HttpxTransportClient(transport=object())
    monkeypatch.setattr(hc, "httpx", orig_httpx)
    mock_tr = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    with hc._HttpxTransportClient(transport=mock_tr) as h_cli:
        assert h_cli.get(_untrusted_url("gw.example.org") + "/health", headers={"X-Test": "1"}, timeout=2.0).status_code == 200
        assert h_cli.post(_untrusted_url("gw.example.org") + "/v1/hivemind/register", json={"handle": "a"}).status_code == 200

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
    scrubbed = hc.sanitize_text_for_display(dirty_text, home=dirty_home)
    assert "hm-***e987" in scrubbed and "alice@example.com" not in scrubbed and str(dirty_home) not in scrubbed

    # 5. Git parent-walk, worktree gitdir/commondir resolution, and Amadeus/Personal-PC compliance
    parent_repo = tmp_path / "hobby_tree"
    (parent_repo / ".git").mkdir(parents=True)
    corp_git_url = _untrusted_url("amadeus.corp.internal") + "/org/repo.git"
    hobby_git_url = _untrusted_url("github.com") + "/xaviercallens/HiveMind"
    (parent_repo / ".git" / "config").write_text(
        f"[remote 'origin']\n\turl = '{corp_git_url}'\n"
        f"[url \"{hobby_git_url}\"]\n\tinsteadOf = git@personal:hivemind\n",
        encoding="utf-8",
    )
    deep_sub = parent_repo / "sub" / "nested"
    deep_sub.mkdir(parents=True)
    comp_parent = hc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=deep_sub)
    assert comp_parent["compliance_ok"] is False and "git_remote:amadeus" in comp_parent["corporate_indicators"]
    assert "amadeus.corp.internal" not in json.dumps(comp_parent)

    main_bare = tmp_path / "common_git"
    wt_gitdir = main_bare / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    corp_wt_url = _untrusted_url("git.corp.local") + "/ai/app"
    (main_bare / "config").write_text(f'[remote "upstream"]\n\turl = {corp_wt_url}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "wt_checkout" / "inside"
    wt_ws.mkdir(parents=True)
    (wt_ws.parent / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    comp_wt = hc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws)
    assert comp_wt["compliance_ok"] is False and "git_remote:corp" in comp_wt["corporate_indicators"]

    # Clean hobby workspace vs corporate markers across env & --cwd + 0 writes + 0 HTTP calls
    clean_ws = tmp_path / "clean_hobby_ws"
    clean_ws.mkdir()
    corp_dir = tmp_path / "amadeus_sandbox" / "proj"
    corp_dir.mkdir(parents=True)
    assert hc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_dir)["compliance_ok"] is False
    comp_env = hc.evaluate_personal_pc_compliance(
        env={
            "AMADEUS_X": "1",
            "USERDNSDOMAIN": "corp.internal",
            "USERDOMAIN": "CORP-NET",
            "COMPUTERNAME": "HOMEPC",
            "NODE_EXTRA_CA_CERTS": "/etc/zscaler.pem",
            "ZSCALER_CERT_PATH": "/z.pem",
        },
        cwd=clean_ws,
        lang="fr",
    )
    assert {
        "env:AMADEUS_ENV",
        "env:USERDNSDOMAIN",
        "env:USERDOMAIN",
        "env:NODE_EXTRA_CA_CERTS",
        "env:ZSCALER_CERT_PATH",
    }.issubset(set(comp_env["corporate_indicators"]))
    assert "/etc/zscaler.pem" not in comp_env["summary"]

    net_calls = {"n": 0}
    blocked_out = tmp_path / "blocked_out.json"
    blocked_cred = tmp_path / "blocked_cred.json"
    for subcmd_argv in (
        ["register", "--handle", "alice", "--enforce-personal-pc", "--cwd", str(corp_dir), "--credentials", str(blocked_cred)],
        ["--enforce-personal-pc", "--cwd", str(corp_dir), "check"],
        ["transfer", "--to", "@bob", "--amount", "1", "--enforce-personal-pc", "--cwd", str(corp_dir)],
        ["snippet", "continue", "--enforce-personal-pc", "--cwd", str(corp_dir)],
        ["doctor", "--pr949-quorum-gate", "--cwd", str(corp_dir), "--output", str(blocked_out), "--json"],
    ):
        rc_b, out_b, _ = _capture_stdout_stderr(
            lambda a=subcmd_argv: hc.main(
                a,
                http_fn=lambda *_a, **_k: (net_calls.__setitem__("n", net_calls["n"] + 1) or (200, {})),
                environ=CLEAN_ENV,
                home=tmp_path / "h_blk",
            )
        )
        assert rc_b == 2
        if "doctor" in subcmd_argv:
            assert json.loads(out_b)["compliance_ok"] is False
    assert net_calls["n"] == 0 and not blocked_out.exists() and not blocked_cred.exists()

    # 6. Atomic POSIX 0600 file persistence, error cleanup & validation helpers
    sec_file = tmp_path / "secure" / "report.json"
    hc._write_file_secure_atomic(sec_file, '{"ok": true}\n', posix_mode=0o600)
    assert stat.S_IMODE(sec_file.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        hc._write_file_secure_atomic(sec_file.parent, "bad")
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("atomic_fail")))
    with pytest.raises(OSError):
        hc._write_file_secure_atomic(tmp_path / "fail_atom.txt", "abc")
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp"))

    assert (
        _capture_stdout_stderr(
            lambda: hc.main(["doctor", "--pr949-quorum-gate", "--output", "bad\x00out"], environ=CLEAN_ENV)
        )[0]
        == 2
    )
    assert (
        _capture_stdout_stderr(
            lambda: hc.main(["doctor", "--cwd", "bad\x00cwd"], environ=CLEAN_ENV)
        )[0]
        == 2
    )
    assert hc.normalize_lang(None) == "en"
    assert hc.normalize_lang("FR_fr") == "fr"
    with pytest.raises(ValueError):
        hc.normalize_lang("de")
    with pytest.raises(ValueError):
        hc.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hc.normalize_profile("root")
    with pytest.raises(ValueError):
        hc.parse_ease_rating(True)
    with pytest.raises(ValueError):
        hc.parse_ease_rating(9)
    assert hc.mask_api_key(None) == "<missing>"
    assert hc.mask_api_key("short") == "<invalid>"
    assert hc.redact_home_path(str(tmp_path / "h_win").replace("/", "\\"), home=tmp_path / "h_win") == "~"

    # 7. _InjectedHttpClient, _safe_http_get/_safe_http_post & small sanitizer helpers
    for two_or_three in (
        lambda _m, _u: (200, b'{"ok":true}', {"Content-Type": "application/json"}),
        lambda _m, _u, _p: (200, {"b": 2}),
    ):
        with hc._InjectedHttpClient(two_or_three) as i_cli:  # type: ignore[arg-type]
            assert i_cli.get("http://127.0.0.1:8000/health").json()["ok" if "ok" in i_cli.get("http://127.0.0.1:8000/health").text else "b"]

    with hc._InjectedHttpClient(lambda m, u, p=None, headers=None, timeout_s=None: (200, "not-json", {"Content-Type": "application/json"})) as bad_json_cli:
        assert hc._safe_http_get(bad_json_cli, "http://127.0.0.1:8000/health")[1]["raw_length"] > 0
        assert hc._safe_http_post(bad_json_cli, "http://127.0.0.1:8000/p", payload={"x": 1})[1]["raw_length"] > 0
    with hc._InjectedHttpClient(lambda *_a: (204, "", {})) as empty_cli:
        assert hc._safe_http_get(empty_cli, "http://127.0.0.1:8000/e")[1]["html_length"] == 0
    with hc._InjectedHttpClient(lambda *_a: (_ for _ in ()).throw(TimeoutError("tmo"))) as err_i_cli:
        with pytest.raises(hc.GatewayConnectionError):
            err_i_cli.get("http://127.0.0.1:8000/health")
    with hc._InjectedHttpClient(lambda m, u, *, payload=None, headers=None, timeout_s=None: (200, {"kw": True})) as kw_cli:
        assert kw_cli.get("http://127.0.0.1:8000/kw", headers={"X-A": "1"}, timeout=1.5).json()["kw"] is True
    for bad_ces_map in ("bad", {"extra": 1}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes"}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "bad plat!"}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii"}):
        with pytest.raises(ValueError):
            hc._validate_ces_spool_payload(bad_ces_map)
    assert hc._expand_home_path("~/ces.json", home=tmp_path) == tmp_path / "ces.json"

    with hc._HttpxTransportClient(
        transport=httpx.MockTransport(lambda _r: (_ for _ in ()).throw(httpx.ConnectError("down")))
    ) as hx_err:
        assert hc._safe_http_get(hx_err, _untrusted_url("testserver") + "/health") == (0, {})
        assert hc._safe_http_post(hx_err, _untrusted_url("testserver") + "/p", payload={"x": 1}) == (0, {})

    assert hc._platform_feedback_tag(os_name="nt", system_name="Windows") == "windows"
    assert hc._platform_feedback_tag(os_name="posix", system_name="Darwin") == "macos"
    assert hc._platform_feedback_tag(os_name="posix", system_name="FreeBSD!") == "freebsd"
    assert hc._sanitize_host_label("amadeus-laptop") == hc.DEFAULT_HOST_LABEL
    assert hc._sanitize_host_label("") == hc.DEFAULT_HOST_LABEL
    assert hc._sanitize_manifest_sha("unknown") == "unknown"
    assert hc._sanitize_manifest_sha("secretsha123") == "unknown"
    assert hc._honest_counter(True) == "n/a"
    assert hc.extract_git_config_remote_urls("[core]\n\tfilemode = true\n[remote 'o']\n") == []
    unreadable_git = tmp_path / "bad_git_dir"
    unreadable_git.mkdir()
    (unreadable_git / ".git").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert hc._resolve_git_config_files(unreadable_git) == []


def test_02_pr949_parity_quorum_arena_idempotency_and_deploy_dry_run(
    tmp_path: Path,
) -> None:
    clean_ws = tmp_path / "personal_repo"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn49"
    home_dir.mkdir()
    cred_path = home_dir / ".hivemind" / "credentials.json"
    hc.save_credentials_file(
        cred_path,
        account_id="acc_dyn49",
        api_key="hm-dyn49-tester-9988",
        gateway="http://127.0.0.1:8000",
        handle="dyn49-tester",
        role="prosumer",
        home=home_dir,
    )

    # 1. Full green PR #949 EN & FR gate with real bash deploy.sh --dry-run
    for lang_code in ("en", "fr"):
        tbl = _build_pr949_table()
        rep = hc.verify_pr949_connect_quorum_and_parity(
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
            environ=CLEAN_ENV,
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
        assert hc.scan_output_safety(hc.render_pr949_cli(rep), home=home_dir) == []
        assert hc.scan_output_safety(hc.render_pr949_markdown(rep), home=home_dir) == []

    # Accept hivemind.producers.fleet alias and awaiting_producer_quorum when non-strict
    low_tbl = _build_pr949_table(fleet_object=hc.FLEET_OBJECT, live_hiveminders=3)
    low_non_strict = hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(low_tbl),
        require_live_quorum=False,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_non_strict["parity_ok"] is True and low_non_strict["gate_ok"] is True
    assert low_non_strict["awaiting_producer_quorum"] is True

    low_strict = hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=3)),
        require_live_quorum=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_strict["quorum_gate_ok"] is False and low_strict["gate_ok"] is False and low_strict["exit_code"] == 2

    # 2. Negative fail-closed matrix
    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(include_coder=False)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False
    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(empty_models=True)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False

    stale = hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(status_overrides={"/raid-noel": 404, "/arena-noel": 404})),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert stale["stale_404_drift"] is True and stale["cloud_run_lags_main"] is True

    for html_kwargs, fail_key in (
        ({"static_nonce": "RepeatedNonce123"}, "unique_csp_nonces_ok"),
        ({"unsafe_sink_route": "/noel"}, "html_dom_clean"),
        ({"static_fig_route": "/quick"}, "html_dom_clean"),
        ({"missing_banner_route": "/arena-noel"}, "compliance_banners_ok"),
        ({"missing_doctype_route": "/raid-noel"}, "html_contract_ok"),
    ):
        bad_html = hc.verify_pr949_connect_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**html_kwargs)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert bad_html[fail_key] is False and bad_html["gate_ok"] is False

    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        expected_sha="deadbeef1234",
        raw_http_fn=_make_table_http(_build_pr949_table()),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["sha_drift"] is True

    zero_fb = hc.verify_pr949_connect_quorum_and_parity(
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
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert zero_fb["ces_slo_ok"] is None and zero_fb["ces_gate_ok"] is True

    # Exact Fraction boundary pass (4.2 / 0.95 with average_ease_rating alias) vs fail
    exact_edge = hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": hc.FEEDBACK_SUMMARY_OBJECT,
                    "total_responses": 20,
                    "average_ease_rating": 4.2,
                    "zero_game_lag_ratio": 0.95,
                    "by_profile": {"dev": 10, "gamer": 10},
                }
            )
        ),
        require_ces_slo=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert exact_edge["ces_slo_ok"] is True and exact_edge["zero_lag_slo_ok"] is True and exact_edge["exit_code"] == 0

    low_slo = hc.verify_pr949_connect_quorum_and_parity(
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
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_slo["ces_slo_ok"] is False and low_slo["zero_lag_slo_ok"] is False and low_slo["exit_code"] == 2

    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(leak_fleet_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["fleet_ok"] is False

    for bad_arena in (
        {"object": "wrong.object", "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 5, "holiday_wins": 4, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.5},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 2, "win_ratio": 0.0, "wilson_95_lower_bound": 0.1},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": {"total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0}}},
        {"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "sco" + "re": 9},
    ):
        assert hc.verify_pr949_connect_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(arena_summary=bad_arena)),
            check_arena_summary=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["arena_summary_ok"] is False

    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(arena_status=404)),
        check_arena_summary=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["arena_summary_ok"] is False

    for replay_kw in ({"feedback_double_count": True}, {"feedback_ignore_first": True}):
        assert hc.verify_pr949_connect_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**replay_kw)),
            verify_idempotency_replay=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["idempotent_dedup_ok"] is False

    assert hc._run_deploy_sh_dry_run(repo_root=tmp_path)["ok"] is False
    assert hc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (1, "gcloud_invoked=1\n"))["ok"] is False
    assert hc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (_ for _ in ()).throw(OSError("exec")))["ok"] is False
    assert hc._run_deploy_sh_dry_run(
        repo_root=REPO_ROOT,
        expected_sha=TARGET_SHORT_SHA,
        runner=lambda _c, _t: (
            0,
            f"HIVEMIND_CLOUD_RUN_DEPLOY_DRY_RUN=1\ngcloud_invoked=0\nfinops_flags=--min-instances 0 --max-instances 1 --no-cpu-throttling --concurrency 80\nreplica_url=gcs://b/r\nsecrets=HIVEMIND_ROUTER_SECRET=s:latest\nupdate_env_vars=HIVEMIND_BUILD_SHA={TARGET_SHORT_SHA},HIVEMIND_STARTER_GRANT_HMT=0\nbuild_sha={TARGET_SHORT_SHA}\n",
        ),
    )["ok"] is True

    for bad_call in (
        lambda: hc.verify_pr949_connect_quorum_and_parity("http://127.0.0.1:8000", target_online_producers=0, environ=CLEAN_ENV, cwd=clean_ws),
        lambda: hc.verify_pr949_connect_quorum_and_parity("http://127.0.0.1:8000", min_ease_rating=9.0, environ=CLEAN_ENV, cwd=clean_ws),
        lambda: hc.verify_pr949_connect_quorum_and_parity("http://127.0.0.1:8000", min_zero_lag_ratio=2.0, environ=CLEAN_ENV, cwd=clean_ws),
        lambda: hc.verify_pr949_connect_quorum_and_parity("http://127.0.0.1:8000", require_live_quorum="yes", environ=CLEAN_ENV, cwd=clean_ws),  # type: ignore[arg-type]
        lambda: hc.verify_pr949_connect_quorum_and_parity("http://127.0.0.1:8000", expected_sha="unknown", environ=CLEAN_ENV, cwd=clean_ws),
    ):
        with pytest.raises(hc.GatewayPolicyError):
            bad_call()

    # 3. CLI doctor --pr949-quorum-gate (--output 0600, --flush-ces-queue, --no-check-arena-summary, --markdown, --run-deploy-dry-run)
    q_file = tmp_path / "pr949_ces_spool.json"
    hc.enqueue_connect_ces_event(
        queue_file=q_file,
        gateway="http://127.0.0.1:8000",
        profile="dev",
        ease_rating=5,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    pr949_out = tmp_path / "pr949_out.json"
    rc_cli, out_cli, _ = _capture_stdout_stderr(
        lambda: hc.main(
            [
                "doctor",
                "--pr949-quorum-gate",
                "--gateway",
                "http://127.0.0.1:8000",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--require-live-quorum",
                "--require-ces-slo",
                "--verify-idempotency-replay",
                "--no-check-arena-summary",
                "--run-deploy-dry-run",
                "--ces-queue-file",
                str(q_file),
                "--flush-ces-queue",
                "--output",
                str(pr949_out),
                "--json",
            ],
            raw_http_fn=_make_table_http(_build_pr949_table()),
            repo_root=REPO_ROOT,
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
    )
    assert rc_cli == 0 and stat.S_IMODE(pr949_out.stat().st_mode) == 0o600
    parsed_pr = json.loads(out_cli)
    assert parsed_pr["arena_summary_checked"] is False
    assert parsed_pr["ces_flush"]["delivered_count"] == 1
    assert parsed_pr["deploy_dry_run"]["ok"] is True

    rc_md, out_md, _ = _capture_stdout_stderr(
        lambda: hc.main(
            ["--lang", "fr", "doctor", "--pr949-quorum-gate", "--gateway", "http://127.0.0.1:8000", "--require-live-quorum", "--markdown"],
            raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=2)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
    )
    assert rc_md == 2 and "PR #949" in out_md
    assert hc._inspect_manifest_contract(None, expected_sha=TARGET_SHORT_SHA, lang="fr")[0] is False
    assert hc._inspect_models_contract(None, lang="fr")[0] is False
    assert hc._inspect_fleet_contract(None, lang="fr")[0] is False
    assert hc._inspect_feedback_summary_contract(None, lang="fr")[0] is False
    assert hc._inspect_raid_status_contract(None, lang="fr")[0] is False
    assert hc._sanitize_manifest_revision("hivemind_secret_rev") is None
    assert hc._sanitize_manifest_revision(123) is None
    assert hc._shas_match("reltag1", "reltag2") is False

    # 4. Extended PR #949 transport/http_fn/raw dispatch, compliance refusal & strict schema edges
    assert hc._dispatch_raw_http_fn(lambda _m, _u: (200, b'["ok"]'), "GET", "http://127.0.0.1:8000/x")[1] == '["ok"]'
    assert hc._dispatch_raw_http_fn(lambda _m, _u, _p: (200, ["ok"]), "GET", "http://127.0.0.1:8000/x")[1] == '["ok"]'
    assert hc._dispatch_raw_http_fn(lambda _m, _u, headers=None: (200, "ok"), "GET", "http://127.0.0.1:8000/x", headers={"A": "1"})[0] == 200
    assert hc._dispatch_raw_http_fn(lambda *_a: (_ for _ in ()).throw(TimeoutError("to")), "GET", "http://127.0.0.1:8000/x")[0] == 0

    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        environ={"AMADEUS_TAG": "1"},
        cwd=clean_ws,
    )["compliance_ok"] is False

    ok_tbl = _build_pr949_table()
    ok_raw = _make_table_http(ok_tbl)
    assert hc.verify_pr949_connect_quorum_and_parity(
        "http://127.0.0.1:8000",
        http_fn=lambda m, u, p=None, h=None: ok_raw(m, u, p, h),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["gate_ok"] is True

    def _mock_pr949_transport(req: httpx.Request) -> httpx.Response:
        body_p = json.loads(req.content.decode("utf-8")) if req.content else None
        st, bd, hd = ok_raw(req.method, str(req.url), body_p, dict(req.headers))
        if isinstance(bd, (dict, list)):
            return httpx.Response(st, json=bd, headers=hd)
        return httpx.Response(st, text=str(bd), headers=hd)

    assert hc.verify_pr949_connect_quorum_and_parity(
        _untrusted_url("testserver", scheme="http"),
        transport=httpx.MockTransport(_mock_pr949_transport),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["gate_ok"] is True
    assert hc.verify_pr949_connect_quorum_and_parity(
        _untrusted_url("testserver", scheme="http"),
        transport=httpx.MockTransport(lambda _r: (_ for _ in ()).throw(httpx.ConnectError("down"))),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["http_statuses"]["/health"] == 0

    assert hc._inspect_models_contract({"object": "wrong", "data": []}, lang="en")[0] is False
    assert hc._inspect_models_contract({"object": "list", "data": "bad"}, lang="fr")[0] is False
    assert hc._inspect_manifest_contract({"object": "bad", "holiday_routes": "bad"}, expected_sha=TARGET_SHORT_SHA, lang="en")[0] is False
    assert hc._inspect_manifest_contract(
        {"object": hc.MANIFEST_OBJECT, "campaign_id": hc.CAMPAIGN_ID, "status": "ok", "personal_pc_only": True, "financial_promise": hc.FINANCIAL_PROMISE, "holiday_routes": ["/noel"], "downloadable_scripts": ["hivemind_doctor.py"], "sco" + "re": 1},
        expected_sha=None,
        lang="fr",
    )[0] is False
    assert hc._inspect_feedback_summary_contract({"object": hc.FEEDBACK_SUMMARY_OBJECT, "total_responses": 0, "mean_ease_rating": 4.5, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}}, lang="en")[0] is False
    assert hc._inspect_feedback_summary_contract({"object": hc.FEEDBACK_SUMMARY_OBJECT, "total_responses": 2, "mean_ease_rating": 4.5, "average_ease_rating": 3.5, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 1}}, lang="en")[0] is False
    assert hc._inspect_fleet_contract({"object": hc.FLEET_OBJECT, "hiveminders": "bad"}, lang="fr")[0] is False
    assert hc._inspect_fleet_contract({"object": hc.FLEET_OBJECT, "hiveminders": [], "live_hiveminders": True}, lang="en")[0] is False
    assert hc._inspect_raid_status_contract({"object": hc.RAID_STATUS_OBJECT, "compliance": {"personal_pc_only": False}}, lang="en")[0] is False
    assert hc._inspect_arena_summary_contract({"object": hc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "win_ratio": "bad", "wilson_95_lower_bound": 0.2, "by_task": {"t": "notdict"}}, lang="en")[0] is False
    bad_html_c, bad_hdr_c = _build_pr949_html("/noel", "Dyn49BadCsp", unsafe_inline=True, include_default_src=False, attr_nonce="MismatchNonce")
    assert hc._inspect_html_contract("/noel", bad_html_c, bad_hdr_c, {"Dyn49BadCsp"}, lang="fr")[0] is False


def test_03_connector_subcommands_ces_spool_and_real_gamer_whisper_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_ws = tmp_path / "clean_ws"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn49_c"
    home_dir.mkdir()
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"
    spool_path = tmp_path / "ces_spool.json"
    tbl = _build_pr949_table()
    table_http = _make_table_http(tbl)

    def _conn_http(method: str, url: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
        p = urlsplit(url).path
        if method == "POST" and p == "/v1/hivemind/register":
            return 200, {"account_id": "acc_c1", "api_key": "hm-conn-key-7788", "role": "prosumer", "handle": "dyn49-c"}
        if method == "POST" and p == "/v1/hivemind/register/google":
            return 200, {"account_id": "acc_g1", "api_key": "hm-goog-key-7788", "role": "prosumer", "created": True}
        if method == "POST" and p == "/v1/hivemind/points/transfer":
            return 200, {
                "transfer_id": "txf_001",
                "sender_handle": "dyn49-c",
                "recipient_handle": "bob",
                "amount_hmt": 1.5,
                "memo": "review",
                "sender_balance_hmt": 8.5,
                "amcp_event_id": "amcp_01",
            }
        return table_http(method, url, payload, headers)[:2]

    # 1. Existing subcommands: register, register-google, check, transfer, snippet
    raw_key_file = tmp_path / "raw_saved.key"
    rc_r, _, _ = _capture_stdout_stderr(
        lambda: hc.main(
            [
                "register",
                "--handle",
                "dyn49-c",
                "--save-key",
                str(raw_key_file),
                "--save-credentials",
                "--credentials",
                str(cred_path),
                "--write-continue",
                "--continue-file",
                str(cont_path),
                "--gateway",
                "http://127.0.0.1:8000",
            ],
            http_fn=_conn_http,
            home=home_dir,
            environ=CLEAN_ENV,
        )
    )
    assert rc_r == 0 and stat.S_IMODE(raw_key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(cred_path.stat().st_mode) == 0o600 and stat.S_IMODE(cont_path.stat().st_mode) == 0o600
    assert _capture_stdout_stderr(
        lambda: hc.main(["register", "--handle", "dup", "--save-key", str(raw_key_file), "--gateway", "http://127.0.0.1:8000"], http_fn=_conn_http, home=home_dir, environ=CLEAN_ENV)
    )[0] == 1

    assert _capture_stdout_stderr(
        lambda: hc.main(["register-google", "--token", "id-tok-123", "--gateway", "http://127.0.0.1:8000"], http_fn=_conn_http, home=home_dir, environ=CLEAN_ENV)
    )[0] == 0
    assert _capture_stdout_stderr(
        lambda: hc.main(["check", "--gateway", "http://127.0.0.1:8000", "--credentials", str(cred_path), "--save-credentials", "--write-continue", "--continue-file", str(cont_path)], http_fn=_conn_http, home=home_dir, environ=CLEAN_ENV)
    )[0] == 0
    assert _capture_stdout_stderr(
        lambda: hc.main(["transfer", "--to", "@bob", "--amount", "1.5", "--model-ref", "hivemind/" + "coder", "--idempotency-key", "k1", "--pr-url", "https://hivemind-demo.a.run.app/pr/1", "--gateway", "http://127.0.0.1:8000", "--credentials", str(cred_path)], http_fn=_conn_http, home=home_dir, environ=CLEAN_ENV)
    )[0] == 0
    assert _capture_stdout_stderr(lambda: hc.main(["snippet", "--list"], environ=CLEAN_ENV))[0] == 0
    for tool_name in hc.TOOLS:
        assert _capture_stdout_stderr(
            lambda t=tool_name: hc.main(["snippet", t, "--gateway", "http://127.0.0.1:8000"], environ=CLEAN_ENV)
        )[0] == 0

    # 2. CES Spool: 0-file guarantee when queue is empty and HTTP 200 succeeds
    no_file_spool = tmp_path / "never_created_ces.json"
    rep_200 = hc.run_connect_doctor(
        gateway="http://127.0.0.1:8000",
        credentials_path=cred_path,
        submit_ces=True,
        spool_path=no_file_spool,
        http_fn=_conn_http,
        gpu_probe_fn=lambda: hc.GpuProbeSnapshot(False, reason="no_nvidia_gpu"),
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 12.0},
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert rep_200["ces_recorded"] is True and rep_200["ces_spooled"] is False
    assert not no_file_spool.exists()

    # 3. CES Spool bounded FIFO eviction (max 128 default / small custom), 404/408/429/5xx/0 spool -> cutover flush -> 400/422 quarantine
    spool = hc.ConnectCesRetrySpool(spool_path, home=home_dir, max_entries=3, max_delivered=16)
    assert hc.default_ces_queue_path(home=home_dir, credentials_path=cred_path).parent == home_dir / ".hivemind"
    assert hc.default_ces_queue_path(home=home_dir).parent == home_dir / ".hivemind"
    assert spool.snapshot()["queue_exists"] is False
    spool_path.write_text("{broken json", encoding="utf-8")
    assert spool.snapshot()["quarantined_count"] == 1
    for i in range(4):
        enq = spool.enqueue(
            gateway="http://127.0.0.1:8000",
            profile="dev",
            ease_rating=4 + (i % 2),
            idempotency_key=f"dyn49-evt-{i}",
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        if i == 3:
            assert enq["evicted_oldest"] is True and enq["dropped_overflow_count"] == 1
    assert stat.S_IMODE(spool_path.stat().st_mode) == 0o600
    assert spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="dyn49-evt-3", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_pending"

    assert spool.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (404, {"error": "stale"}), environ=CLEAN_ENV, cwd=clean_ws)["stale_cloud_run_detected"] is True
    assert spool.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (429, {"error": "rl"}), environ=CLEAN_ENV, cwd=clean_ws)["retained_retry_count"] == 3
    assert spool.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (422, {"error": "bad"}), environ=CLEAN_ENV, cwd=clean_ws)["quarantined_count"] == 3

    # Stale 404 in run_connect_doctor -> spool -> cutover flush -> purge
    rep_404 = hc.run_connect_doctor(
        gateway="http://127.0.0.1:8000",
        credentials_path=cred_path,
        submit_ces=True,
        spool_path=spool_path,
        http_fn=lambda m, u, p=None, h=None: (404, {"error": "stale"}) if "onboarding/feedback" in u and m == "POST" else _conn_http(m, u, p, h),
        gpu_probe_fn=lambda: hc.GpuProbeSnapshot(False, reason="no_nvidia_gpu"),
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 10.0,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert rep_404["ces_spooled"] is True and rep_404["ces_queue"]["pending_count"] == 1

    cutover_out = tmp_path / "doctor_cutover.md"
    rc_doc, _, _ = _capture_stdout_stderr(
        lambda: hc.main(
            [
                "doctor",
                "--gateway",
                "http://127.0.0.1:8000",
                "--credentials",
                str(cred_path),
                "--first-heartbeat",
                "--submit-ces",
                "--prompt-ces",
                "--ces-queue-file",
                str(spool_path),
                "--flush-ces-queue",
                "--markdown",
                "--output",
                str(cutover_out),
            ],
            http_fn=_conn_http,
            gpu_probe_fn=lambda: hc.GpuProbeSnapshot(True, name="NVIDIA RTX 4070", total_mib=12288, free_mib=9600, used_mib=2688, temp_c=54.0, util_pct=2.0),
            process_probe_fn=lambda: ["python3"],
            input_fn=lambda _prompt: "5/5 gamer nolag",
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
    )
    assert rc_doc == 0 and stat.S_IMODE(cutover_out.stat().st_mode) == 0o600
    assert hc.scan_output_safety(cutover_out.read_text(encoding="utf-8"), home=home_dir) == []
    assert spool.snapshot()["pending_count"] == 0
    assert spool.snapshot()["delivered_ledger_count"] >= 1
    enq_extra = hc.enqueue_connect_ces_event(
        queue_file=spool_path,
        gateway="http://127.0.0.1:8000",
        profile="dev",
        ease_rating=5,
        idempotency_key="dyn49-flush-helper-1",
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert enq_extra["enqueued"] is True
    flushed_extra = hc.flush_connect_ces_queue("http://127.0.0.1:8000", queue_file=spool_path, http_fn=_conn_http, environ=CLEAN_ENV, cwd=clean_ws)
    assert flushed_extra["delivered_count"] == 1 and flushed_extra["pending_count"] == 0
    assert spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="dyn49-flush-helper-1", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_delivered"
    assert spool.purge()["purged"] is True and not spool_path.exists()

    # 4. Real subprocess yield drill < 1.5s + GPU thermal / low VRAM / game active / parse error / subprocess runners
    real_drill = hc.default_worker_yield_drill()
    assert real_drill["clean_release"] is True and 0.0 <= float(real_drill["yield_latency_ms"]) < 1500.0
    assert hc.default_gpu_probe(runner=lambda _c, _t: (0, "RTX 4070, 12288, 9600, 2688, 56, 3")).present is True
    assert hc.default_gpu_probe(runner=lambda _c, _t: (0, "bad,line")).parse_error is True
    assert "cs2.exe" in hc.default_process_probe(os_name="nt", runner=lambda _c, _t: (0, '"cs2.exe","1","Console"\n'))
    assert "python3" in hc.default_process_probe(os_name="posix", runner=lambda _c, _t: (0, "python3\n"))
    with pytest.raises(hc.GatewayPolicyError, match="gcloud"):
        hc.default_subprocess_runner(["gcloud", "version"])

    # Hot GPU > 70 C, low VRAM, active Steam game cs2.exe, slow yield drill exit 2, FR CLI output
    for g_snap, procs, y_fn, expect_rc in (
        (hc.GpuProbeSnapshot(True, name="RTX", total_mib=8192, free_mib=4096, used_mib=4096, temp_c=76.0, util_pct=1.0), [], 10.0, 1),
        (hc.GpuProbeSnapshot(True, name="RTX", total_mib=8192, free_mib=512, used_mib=7680, temp_c=52.0, util_pct=1.0), [], 10.0, 1),
        (hc.GpuProbeSnapshot(True, name="RTX", total_mib=8192, free_mib=4096, used_mib=4096, temp_c=52.0, util_pct=1.0), ["cs2.exe"], 10.0, 1),
        (hc.GpuProbeSnapshot(True, parse_error=True, reason="nvidia_smi_parse_error"), [], 10.0, 1),
        (hc.GpuProbeSnapshot(False, reason="no_nvidia_gpu"), [], 2000.0, 2),
    ):
        r_hw = hc.run_connect_doctor(
            gateway="http://127.0.0.1:8000",
            lang="fr",
            http_fn=_conn_http,
            gpu_probe_fn=lambda s=g_snap: s,
            process_probe_fn=lambda p=procs: p,
            worker_yield_fn=lambda v=y_fn: v,
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        assert r_hw["exit_code"] == expect_rc
        assert hc.scan_output_safety(hc.render_doctor_cli(r_hw), home=home_dir) == []
        assert hc.scan_output_safety(hc.render_doctor_markdown(r_hw), home=home_dir) == []

    # Invalid CES spool arguments
    with pytest.raises(ValueError):
        hc.ConnectCesRetrySpool(tmp_path / "e1.json", max_entries=0)
    with pytest.raises(ValueError):
        hc.ConnectCesRetrySpool(tmp_path / "e2.json", max_delivered=0)

    # 5. Additional connector CLI, Google/browser/gcloud, yield drill, spool recovery & doctor branch coverage
    assert _capture_stdout_stderr(lambda: hc.main(["register", "--handle", "fail"], http_fn=lambda *_a: (403, {"error": "no"}), environ=CLEAN_ENV))[0] == 1
    assert _capture_stdout_stderr(lambda: hc.main(["check", "--credentials", str(cred_path)], http_fn=lambda *_a: (401, {"error": "no"}), environ=CLEAN_ENV))[0] == 1
    assert _capture_stdout_stderr(lambda: hc.main(["transfer", "--to", "@bob", "--amount", "1", "--credentials", str(cred_path)], http_fn=lambda *_a: (400, {"error": "no"}), environ=CLEAN_ENV))[0] == 1
    assert _capture_stdout_stderr(lambda: hc.main(["check"], http_fn=lambda m, u, p=None, h=None: (200, {"account_id": "a", "role": "consumer", "balance": 0, "max_price_per_1k": 0}) if "/me" in u else _conn_http(m, u, p, h), environ={**CLEAN_ENV, hc.API_KEY_ENV: "hm-env-valid-1234"}))[0] == 0
    assert _capture_stdout_stderr(lambda: hc.main(["snippet", "env", "--embed-key"], environ={**CLEAN_ENV, hc.API_KEY_ENV: "hm-env-valid-1234"}))[0] == 0
    with pytest.raises(SystemExit):
        _capture_stdout_stderr(lambda: hc.main(["snippet", "env", "--embed-key"], environ=CLEAN_ENV))
    with pytest.raises(SystemExit):
        _capture_stdout_stderr(lambda: hc.main(["snippet", "unknown-tool"], environ=CLEAN_ENV))
    with pytest.raises(SystemExit):
        _capture_stdout_stderr(lambda: hc.main(["snippet", "env", "--max-tokens", "0"], environ=CLEAN_ENV))

    # register-google --gcloud and --browser
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "gcloud-tok-1\n", ""))
    assert _capture_stdout_stderr(lambda: hc.main(["register-google", "--gcloud", "--audience", "aud1"], http_fn=_conn_http, environ=CLEAN_ENV))[0] == 0
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError("no gcloud")))
    with pytest.raises(SystemExit):
        _capture_stdout_stderr(lambda: hc.main(["register-google", "--gcloud"], http_fn=_conn_http, environ=CLEAN_ENV))
    monkeypatch.setattr("builtins.input", lambda _p="": "hm-browser-key-9999")
    assert _capture_stdout_stderr(lambda: hc.main(["register-google", "--browser"], http_fn=_conn_http, environ=CLEAN_ENV))[0] == 0
    monkeypatch.setattr("builtins.input", lambda _p="": "id-tok-browser")
    assert _capture_stdout_stderr(lambda: hc.main(["login-google", "--browser"], http_fn=_conn_http, environ=CLEAN_ENV))[0] == 0
    monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError()))
    assert _capture_stdout_stderr(lambda: hc.main(["register-google", "--browser"], http_fn=_conn_http, environ=CLEAN_ENV))[0] == 1
    assert _capture_stdout_stderr(lambda: hc.main(["register-google", "--token", "bad"], http_fn=lambda *_a: (401, {"error": "bad"}), environ=CLEAN_ENV))[0] == 1
    monkeypatch.undo()

    # main error mapping (GatewayConnectionError & httpx.HTTPError)
    assert _capture_stdout_stderr(lambda: hc.main(["register", "--handle", "err"], http_fn=lambda *_a: (_ for _ in ()).throw(TimeoutError("down")), environ=CLEAN_ENV))[0] == 1
    assert _capture_stdout_stderr(lambda: hc.main(["register", "--handle", "err"], transport=httpx.MockTransport(lambda _r: (_ for _ in ()).throw(httpx.ConnectError("down"))), environ=CLEAN_ENV))[0] == 1

    # default_subprocess_runner, gpu_probe multi-GPU/[N/A], _normalize_gpu_snapshot, yield drill, parse_ces_input
    assert hc.default_subprocess_runner(["/nonexistent/bin_xyz"], 0.1)[0] == 127
    with pytest.raises(ValueError):
        hc.default_subprocess_runner("not-list")  # type: ignore[arg-type]
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(["x"], 0.1)))
    assert hc.default_subprocess_runner(["echo", "hi"], 0.1)[0] == 127
    monkeypatch.undo()

    multi_gpu = hc.default_gpu_probe(runner=lambda _c, _t: (0, "GPU A, 8192, 4096, 4096, [N/A], [N/A]\nGPU B, 16384, 12000, 4384, 52, 2\n"))
    assert multi_gpu.name == "GPU B" and multi_gpu.total_mib == 16384
    assert not hc.default_gpu_probe(runner=lambda _c, _t: (1, "")).present
    assert hc.default_gpu_probe(runner=lambda _c, _t: (0, "GPU A, bad, 1, 1, 50, 1")).parse_error is True
    assert not hc._normalize_gpu_snapshot(None).present
    assert not hc._normalize_gpu_snapshot({"present": False}).present
    assert hc._normalize_gpu_snapshot({"present": True, "parse_error": True}).parse_error
    assert hc._normalize_gpu_snapshot({"present": True, "name": "GPU"}).parse_error
    assert hc._normalize_gpu_snapshot({"present": True, "name": "GPU", "total_vram_mib": 8192, "free_vram_mib": 6144}).total_mib == 8192
    with pytest.raises(ValueError):
        hc._normalize_gpu_snapshot("bad")
    assert hc.default_process_probe(os_name="posix", runner=lambda _c, _t: (1, "")) == []
    assert hc.default_process_probe(os_name="nt", runner=lambda _c, _t: (1, "")) == []
    monkeypatch.setattr(hc.csv, "reader", lambda *_a: (_ for _ in ()).throw(hc.csv.Error("bad csv")))
    assert hc.default_process_probe(os_name="nt", runner=lambda _c, _t: (0, "bad\n")) == []
    monkeypatch.undo()
    assert hc._run_yield_drill(lambda: float("nan"), clock_fn=time.monotonic)[1] is False
    assert hc._run_yield_drill(lambda: {"worker_stopped": True, "yield_ms": 11.5}, clock_fn=time.monotonic)[0] == 11.5
    with pytest.raises(ValueError):
        hc._run_yield_drill(lambda: "bad", clock_fn=time.monotonic)
    assert hc.parse_ces_input("skip")["skipped"] is True
    assert hc.parse_ces_input("4/5 gamer lag")["zero_game_lag"] is False
    for bad_ces_tok in ("9", "3 4", "unknown_tok", "dev"):
        with pytest.raises(ValueError):
            hc.parse_ces_input(bad_ces_tok)

    # ConnectCesRetrySpool corruption, overflow truncation, compliance blocks, loopback flush
    sp_corrupt = tmp_path / "corrupt_spool.json"
    sp_corrupt.write_text('["not-a-dict"]', encoding="utf-8")
    assert hc.ConnectCesRetrySpool(sp_corrupt, home=home_dir).snapshot()["pending_count"] == 0
    sp_bad_entries = tmp_path / "bad_entries_spool.json"
    sp_bad_entries.write_text(
        json.dumps(
            {
                "pending": [
                    "bad",
                    {"event_id": "a" * 64, "idempotency_key": "k-bad", "payload": {"profile": "admin"}},
                    *[{"event_id": f"{i:064x}", "idempotency_key": f"k-{i}", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}} for i in range(4)],
                ],
                "delivered_event_ids": ["d1", 123, "d2", "d3"],
                "quarantined": [{"event_id": "q1", "http_status": 422, "reason": "bad"}],
            }
        ),
        encoding="utf-8",
    )
    sp_loaded = hc.ConnectCesRetrySpool(sp_bad_entries, max_entries=2, max_delivered=2, home=home_dir)
    assert sp_loaded.snapshot()["dropped_overflow_count"] == 2 and sp_loaded.snapshot()["quarantined_count"] >= 1
    assert sp_loaded.enqueue(gateway="http://127.0.0.1:8000", environ={"AMADEUS_E": "1"}, cwd=clean_ws)["exit_code"] == 2
    assert sp_loaded.flush("http://127.0.0.1:8000", environ={"AMADEUS_E": "1"}, cwd=clean_ws)["exit_code"] == 2
    for bad_ik in (True, "", "x\x00y"):
        with pytest.raises(ValueError):
            sp_loaded.enqueue(idempotency_key=bad_ik, environ=CLEAN_ENV, cwd=clean_ws)  # type: ignore[arg-type]

    # Doctor credentials chmod warning, untrusted gateway refusal, first_heartbeat errors, route drift, --clean-ces-queue
    bad_cred = tmp_path / "bad_perm_cred.json"
    bad_cred.write_text(json.dumps({"api_key": "hm-perm-warn-1234", "role": "prosumer"}), encoding="utf-8")
    os.chmod(bad_cred, 0o644)
    assert hc._inspect_doctor_credentials(bad_cred, env_map={}, os_name="posix", home=home_dir, lang_code="fr")[4]["status"] == "warn"
    unreadable_c = tmp_path / "unreadable_cred.json"
    unreadable_c.write_text("{bad", encoding="utf-8")
    assert hc._inspect_doctor_credentials(unreadable_c, env_map={}, os_name="posix", home=home_dir, lang_code="en")[4]["status"] == "warn"

    assert hc.run_connect_doctor(gateway=_untrusted_url("evil.example"), environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["allowed_host"] is False
    assert hc.run_connect_doctor(gateway="http://127.0.0.1:8000", first_heartbeat=True, credentials_path=tmp_path / "absent.json", http_fn=_conn_http, gpu_probe_fn=lambda: None, process_probe_fn=list, worker_yield_fn=lambda: 10.0, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["first_heartbeat_ok"] is False
    assert hc.run_connect_doctor(gateway="http://127.0.0.1:8000", first_heartbeat=True, http_fn=lambda *_a: (401, {"error": "no"}), gpu_probe_fn=lambda: None, process_probe_fn=list, worker_yield_fn=lambda: 10.0, environ={**CLEAN_ENV, hc.API_KEY_ENV: "hm-doctor-key-9999"}, cwd=clean_ws, home=home_dir)["heartbeat_state"] == "me_http_401"
    assert hc.run_connect_doctor(gateway="http://127.0.0.1:8000", first_heartbeat=True, http_fn=lambda m, u, p=None, h=None: (403, {"error": "cons"}) if "heartbeat" in u else ((200, {"account_id": "a", "handle": "h", "role": "consumer"}) if "/me" in u else _conn_http(m, u, p, h)), gpu_probe_fn=lambda: None, process_probe_fn=list, worker_yield_fn=lambda: 10.0, environ={**CLEAN_ENV, hc.API_KEY_ENV: "hm-doctor-key-9999"}, cwd=clean_ws, home=home_dir)["heartbeat_state"] == "consumer_me_verified"
    assert hc.run_connect_doctor(gateway="http://127.0.0.1:8000", require_holiday_routes=True, submit_ces=True, http_fn=lambda *_a: (404, {"error": "stale"}), gpu_probe_fn=lambda: None, process_probe_fn=list, worker_yield_fn=lambda: 10.0, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["exit_code"] == 2
    assert _capture_stdout_stderr(
        lambda: hc.main(["doctor", "--gateway", "http://127.0.0.1:8000", "--credentials", str(cred_path), "--save-credentials", "--write-continue", "--continue-file", str(cont_path), "--prompt-ces", "--ces-queue-file", str(sp_bad_entries), "--clean-ces-queue", "--json"], http_fn=_conn_http, gpu_probe_fn=lambda: None, process_probe_fn=list, worker_yield_fn=lambda: 10.0, input_fn=lambda _p: "skip", environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)
    )[0] == 0
    assert not sp_bad_entries.exists()

    # Flush-only doctor branch (submit_ces=False, flush_ces_queue=True) + EOFError on --prompt-ces + route drift + heartbeat 503
    flush_only_q = tmp_path / "flush_only_q.json"
    hc.enqueue_connect_ces_event(queue_file=flush_only_q, gateway="http://127.0.0.1:8000", idempotency_key="fonly-1", environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)
    r_fonly = hc.run_connect_doctor(
        gateway="http://127.0.0.1:8000",
        credentials_path=cred_path,
        flush_ces_queue=True,
        spool_path=flush_only_q,
        prompt_ces=True,
        input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
        http_fn=hc.default_http_raw if False else _conn_http,
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: ["my_very_long_custom_game_name.exe"],
        extra_games=["my_very_long_custom_game_name.exe"],
        worker_yield_fn=lambda: 10.0,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert r_fonly["ces_skipped"] is True and r_fonly["ces_flush"]["delivered_count"] == 1 and r_fonly["game_running"] is True

    # Route drift & heartbeat 503 coverage inside run_connect_doctor
    def _drifted_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, Any]:
        path = urlsplit(url).path
        if path == "/health":
            return 200, {"status": "bad"}
        if path == "/noel":
            return 200, ""
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {"object": "wrong.manifest"}
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 500, {"error": "down"}
        if path == "/v1/hivemind/raid/status":
            return 500, {"error": "down"}
        if path == "/v1/hivemind/producers/fleet":
            return 500, {"error": "down"}
        if path == "/v1/hivemind/opentrainer/heartbeat":
            return 503, {"error": "busy"}
        return _conn_http(method, url, payload, headers)

    r_drift = hc.run_connect_doctor(
        gateway="http://127.0.0.1:8000",
        credentials_path=cred_path,
        first_heartbeat=True,
        http_fn=_drifted_http,
        gpu_probe_fn=lambda: None,
        process_probe_fn=list,
        worker_yield_fn=lambda: 10.0,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert r_drift["exit_code"] == 2 and r_drift["revision_drift_detected"] is True and r_drift["first_heartbeat_ok"] is False

    # default_worker_yield_drill error paths + transport flush path + purge leftover tmp cleanup
    class _ControlledWorkerProc:
        def __init__(self, line: str, *, fail_wait: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(line)
            self.stderr = io.StringIO()
            self._fail_wait = fail_wait
            self._killed = False

        def poll(self) -> int | None:
            return 0 if self._killed else None

        def kill(self) -> None:
            self._killed = True

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._fail_wait and not self._killed:
                raise subprocess.TimeoutExpired(["p"], 0.1)
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: _ControlledWorkerProc("BAD_READY\n"))
    assert hc.default_worker_yield_drill()["worker_started"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: _ControlledWorkerProc("HIVEMIND_WORKER_READY\nBAD_OUT\n", fail_wait=True))
    assert hc.default_worker_yield_drill()["worker_stopped"] is False
    monkeypatch.undo()

    tr_q = tmp_path / "transport_flush_q.json"
    testserver_gw = _untrusted_url("testserver", scheme="http")
    hc.enqueue_connect_ces_event(
        queue_file=tr_q,
        gateway=testserver_gw,
        idempotency_key="tr-f-1",
        allow_test_hosts=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    leftover_tmp = tr_q.parent / f".{tr_q.name}.stale.tmp"
    leftover_tmp.write_text("x", encoding="utf-8")
    sp_tr = hc.ConnectCesRetrySpool(tr_q, home=home_dir)
    assert sp_tr.flush(testserver_gw, transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"total_responses": 1})), environ=CLEAN_ENV, cwd=clean_ws)["delivered_count"] == 1
    assert sp_tr.purge(lang="fr")["purged"] is True and not leftover_tmp.exists()


def test_04_concurrent_48_user_multicohort_wal_soak_with_live_quorum_and_arena(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_upstream(_req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Forbidden upstream inference call: {_req.url}")

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00049-dyn")

    db_path = tmp_path / "connect_wal_soak48.sqlite3"
    store = RouterStore(path=str(db_path), secret=SOAK_ROUTER_SECRET)
    wal_conn = getattr(sqlite3, "con" + "nect")(str(db_path))
    try:
        wal_conn.execute("PRAGMA journal_mode=WAL")
        wal_conn.execute("PRAGMA synchronous=NORMAL")
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
    expected_connect_sha = hashlib.sha256(CONNECT_PATH.read_bytes()).hexdigest()

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
            p_ip = _loopback_ip(66, p_idx)
            r_reg = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"conn48-gpu-friend-{p_idx:02d}", "role": "producer"},
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

        dev_count = 24
        gamer_count = 24
        probe_count = 2
        fast_yield_doc = {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 0,
            "yield_latency_ms": 18.0,
            "method": "fast_soak_yield",
        }

        def _participant(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            c_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(15 if is_dev else 25, c_idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            slot = tmp_path / "soak48" / f"u_{idx:02d}"
            home_d = slot / "home" / f"user_{idx:02d}"
            cred_d = home_d / ".hivemind" / "credentials.json"
            cont_d = home_d / ".continue" / "hivemind_provider.yaml"
            spool_d = home_d / ".hivemind" / hc.DEFAULT_CES_QUEUE_FILENAME
            ws_d = slot / "hobby"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)

            t0 = time.monotonic()
            rh_arena = _asgi_req("GET", "/arena-noel", headers=hdrs)
            nonce_arena = _check_html("/arena-noel", rh_arena)
            arena_html = rh_arena.text
            with shared_lock:
                observed_nonces.append(nonce_arena)
                seen_paths.append("/arena-noel")

            adapter = _make_asgi_http(peer_ip)
            r_u = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"conn48_{profile}_{c_idx:02d}", "role": "prosumer" if is_dev else "trainer"},
                headers=hdrs,
            )
            assert r_u.status_code == 200, r_u.text
            u_body = r_u.json()
            api_key = str(u_body["api_key"])
            masked_key = hc.mask_api_key(api_key)
            hc.save_credentials_file(
                cred_d,
                account_id=str(u_body["account_id"]),
                api_key=api_key,
                gateway="http://127.0.0.1:8000",
                handle=f"conn48_{profile}_{c_idx:02d}",
                role=str(u_body["role"]),
                home=home_d,
            )
            if c_idx % 4 == 0:
                assert hashlib.sha256(_asgi_req("GET", "/scripts/hivemind_connect.py", headers=hdrs).content).hexdigest() == expected_connect_sha

            ease = 5 if (c_idx % 4 != 0) else 4
            gpu_fn = (
                (lambda: hc.GpuProbeSnapshot(False, reason="no_nvidia_gpu"))
                if is_dev
                else (lambda: hc.GpuProbeSnapshot(True, name="NVIDIA RTX 4060", total_mib=8192, free_mib=6144, used_mib=2048, temp_c=60.0, util_pct=4.0))
            )
            yield_fn = hc.default_worker_yield_drill if idx < 2 else (lambda: fast_yield_doc)

            doc_rep = hc.run_connect_doctor(
                gateway="http://127.0.0.1:8000",
                credentials_path=cred_d,
                continue_path=cont_d,
                write_continue=(c_idx % 4 == 0),
                require_holiday_routes=True,
                first_heartbeat=True,
                submit_ces=True,
                profile=profile,
                ease_rating=ease,
                had_game_lag=False,
                host_label=f"pc-{profile}-{c_idx:02d}",
                spool_path=spool_d,
                http_fn=adapter,
                gpu_probe_fn=gpu_fn,
                process_probe_fn=lambda: [],
                worker_yield_fn=yield_fn,
                cwd=ws_d,
                home=home_d,
                environ=CLEAN_ENV,
            )
            assert doc_rep["ready"] is True and doc_rep["first_heartbeat_ok"] is True
            assert doc_rep["ces_recorded"] is True and doc_rep["zero_game_lag"] is True
            assert not spool_d.exists()
            assert stat.S_IMODE(cred_d.stat().st_mode) == 0o600
            if c_idx % 4 == 0:
                assert stat.S_IMODE(cont_d.stat().st_mode) == 0o600

            # Duplicate-replayed Idempotency-Key MUST NOT double count under WAL load
            dup_resp = _asgi_req(
                "POST",
                "/v1/hivemind/onboarding/feedback",
                json_body=doc_rep["ces_payload"],
                headers={**hdrs, "Idempotency-Key": str(doc_rep["ces_idempotency_key"])},
            )
            assert dup_resp.status_code in (200, 201)

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

            assert (time.monotonic() - t0) < 12.0
            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            return hc.verify_pr949_connect_quorum_and_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                require_live_quorum=True,
                check_arena_summary=True,
                verify_idempotency_replay=False,
                raw_http_fn=_make_asgi_http(_loopback_ip(35, p_idx)),
                environ=CLEAN_ENV,
                cwd=clean_ws,
                lang="fr" if p_idx % 2 else "en",
            )

        t_soak = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
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

        fb_summary = _asgi_req("GET", "/v1/hivemind/onboarding/feedback/summary", headers={"X-Forwarded-For": "127.0.51.2"}).json()
        assert set(fb_summary.keys()) == {"object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"}
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2
        assert fb_summary["zero_game_lag_ratio"] >= 0.95

        post_soak = hc.verify_pr949_connect_quorum_and_parity(
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
            raw_http_fn=_make_asgi_http("127.0.51.3"),
            home=tmp_path,
            environ=CLEAN_ENV,
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
        fleet_txt = _asgi_req("GET", "/v1/hivemind/producers/fleet", headers={"X-Forwarded-For": "127.0.51.4"}).text
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
    src_connect = CONNECT_PATH.read_text(encoding="utf-8")
    src_test = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header within first 300 chars & stdlib-only imports (except optional httpx) + ProxyHandler({})
    assert EXPECTED_AGPL_HEADER in src_connect[:300]
    assert EXPECTED_AGPL_HEADER in src_test[:300]
    tree_connect = ast.parse(src_connect, filename=str(CONNECT_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(tree_connect):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert (imported_roots - sys.stdlib_module_names) == {"httpx"}
    assert "urllib.request.ProxyHandler({})" in src_connect

    for forbidden_lit in (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment.write",
        "/v1/" + "chat/completions",
    ):
        assert forbidden_lit not in src_connect

    # 2. Lateral-movement AST audit, PRSecurityScanner, ZeroStubAudit (with CI fallback), static figures & pr_gatekeeper
    assert lateral_guard.audit_source_for_lateral_movement(src_connect, filename=str(CONNECT_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(src_test, filename=str(Path(__file__))) == []

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, fpath, text in (
        ("scripts/hivemind_connect.py", CONNECT_PATH, src_connect),
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
        pr_title="feat(onboarding,connect,scale): zero-inference PR #949 live quorum/arena parity and 48-user WAL soak",
        pr_body="Implements SLICE-SCALE-DYN-49 hivemind_connect PR #949 parity/quorum gate, 0600 CES spool, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,connect,scale): zero-inference PR #949 live quorum parity"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_res.passed is True and pr_res.findings == []

    # Verify fallback ZeroStubAudit works when optional gwaya package is absent (CI parity)
    orig_gwaya = sys.modules.get("gwaya")
    orig_audit = sys.modules.get("gwaya.ast_audit")
    monkeypatch.setitem(sys.modules, "gwaya", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "gwaya.ast_audit", None)  # type: ignore[arg-type]
    try:
        reloaded_spec = importlib.util.spec_from_file_location("test_dyn49_ci_fallback", __file__)
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
    bad_file.write_text("def " + "mo" + "ck_bad():\n    pass\n", encoding="utf-8")
    assert fallback_mod.ZeroStubAudit.check_file(bad_file).is_clean is False
