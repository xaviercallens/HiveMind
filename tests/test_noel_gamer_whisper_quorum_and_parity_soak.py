# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/arena parity, Idempotency-Key replay, SSRF/redirect & worktree Amadeus guard, 0600 CES spool & 48-user WAL soak tests (`SLICE-SCALE-DYN-50`)."""

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
SELFTEST_PATH = REPO_ROOT / "scripts" / "noel_gamer_whisper_selftest.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import directly without starting any internal coverage.Coverage() instance so external
# `coverage run --rcfile=/dev/null --include='*/scripts/noel_gamer_whisper_selftest.py'` works cleanly.
from scripts import noel_gamer_whisper_selftest as selftest  # noqa: E402

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
        """Minimal AST zero-stub fallback when optional `gwaya` is absent."""

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
    "test_no_lateral_movement_dyn50",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn50",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn50",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-gaming-rig"}
TARGET_FULL_SHA = "e5f6a7b8901234567890abcdef1234567890abcdef"
TARGET_SHORT_SHA = TARGET_FULL_SHA[:12]
SOAK_ROUTER_SECRET = b"noel-scale-dyn50-wal-secret-32b!"
ALPHA_KEY = "hm-dyn50-gamer-alpha-9876"
LIVE_KEY = "hm_live_dyn50_gamer_4321"

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


def _verify_sha256_receipt(payload: dict[str, Any]) -> None:
    digest = payload.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == expected


def _quiet_rtx4070_probe(**_: Any) -> selftest.GpuTelemetry:
    return selftest.GpuTelemetry(
        present=True,
        parse_error=False,
        name="NVIDIA GeForce RTX 4070",
        total_vram_mib=12288,
        free_vram_mib=9600,
        used_vram_mib=2688,
        temp_c=53.0,
        util_pct=3.0,
    )


def _fast_worker_drill(**_: Any) -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 34.5,
        "exit_code": 0,
        "method": "injected_fast_drill",
    }


def _idle_processes(**_: Any) -> list[str]:
    return ["systemd", "python3", "explorer.exe", "steam.exe"]


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
    script_nonce = attr_nonce if attr_nonce is not None else nonce
    default_part = "default-src 'none'; " if include_default_src else "default-src *; "
    inline_part = "'unsafe-inline' " if unsafe_inline else ""
    csp = f"{default_part}script-src {inline_part}'nonce-{nonce}'; style-src 'nonce-{nonce}'"
    doc_prefix = "<!DOCTYPE html>\n" if include_doctype else ""
    banner = (
        "<p>100% personal home PCs &amp; hobby projects only • 0.00 EUR / 0.00 USD • AGPL-3.0</p>"
        if include_banner
        else "<p>HiveMind generic page</p>"
    )
    html = (
        f'{doc_prefix}<html lang="en"><head><title>{path}</title></head>'
        f"<body>{banner}{extra_body}"
        f'<script nonce="{script_nonce}">document.body.dataset.route="{path}";</script>'
        "</body></html>"
    )
    return html, {"content-type": "text/html; charset=utf-8", "content-security-policy": csp}


def _build_pr949_table(
    *,
    build_sha: str = TARGET_FULL_SHA,
    cloud_run_revision: str | None = "hivemind-openrouter-00050-dyn",
    fleet_object: str = selftest.FLEET_OBJECT,
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
    status_overrides: dict[str, int] | None = None,
    manifest_override: dict[str, Any] | None = None,
    feedback_double_count: bool = False,
    feedback_ignore_first: bool = False,
) -> dict[str, Any]:
    statuses = row_statuses or tuple("live" for _ in range(max(live_hiveminders, 1)))
    fleet_rows = [
        {
            "handle": f"gamer-friend-{i:02d}",
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
            "object": selftest.FEEDBACK_SUMMARY_OBJECT,
            "total_responses": 12,
            "mean_ease_rating": 4.8,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 4, "gamer": 8},
        }
    )
    arena_doc = (
        dict(arena_summary)
        if arena_summary is not None
        else {
            "object": selftest.ARENA_SUMMARY_OBJECT,
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
            "object": selftest.MANIFEST_OBJECT,
            "campaign_id": selftest.CAMPAIGN_ID,
            "status": "ok",
            "personal_pc_only": True,
            "financial_promise": selftest.FINANCIAL_PROMISE,
            "build_sha": build_sha,
            "cloud_run_revision": cloud_run_revision,
            "holiday_routes": list(selftest.PR949_MANIFEST_HOLIDAY_ROUTES),
            "downloadable_scripts": ["noel_gamer_whisper_selftest.py", "hivemind_quickstart.py"],
        }
    )
    return {
        "models": {"object": "list", "data": model_rows},
        "manifest": manifest_doc,
        "feedback": fb_doc,
        "raid": {
            "object": selftest.RAID_STATUS_OBJECT,
            "compliance": {
                "personal_pc_only": True,
                "monetary_cost_eur": "0.00 EUR",
                "monetary_cost_usd": "0.00 USD",
            },
            "swarm": {"active_trainers": live_hiveminders, "live_producers": live_hiveminders},
            "onboarding_feedback": fb_doc,
        },
        "fleet": {
            "object": fleet_object,
            "live_hiveminders": live_hiveminders,
            "total_hiveminders": max(live_hiveminders, len(fleet_rows)),
            "distinct_physical_gpus": distinct_gpus,
            "live_endpoints": live_hiveminders,
            "live_models": ["hivemind/" + "coder"],
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
        "status_overrides": dict(status_overrides or {}),
        "nonce_seq": 0,
    }


def _make_table_http(table: dict[str, Any], *, me_role: str = "prosumer", hb_status: int = 200) -> selftest.RawHttpFn:
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
        if method == "GET" and path in selftest.PR949_HTML_ROUTES:
            table["nonce_seq"] += 1
            nonce = table["static_nonce"] or f"Dyn50Nonce_{table['nonce_seq']:04d}_AbCd"
            extra = ""
            if table["unsafe_sink_route"] == path:
                extra += "<script>el." + "inner" + "HTML='x';</script>"
            if table["static_fig_route"] == path:
                extra += "<p>Latency " + "142.5 " + "m" + "s</p>"
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
                    prof = str((payload or {}).get("profile", "gamer"))
                    fb["total_responses"] = int(fb["total_responses"]) + 1
                    fb["by_profile"][prof] = int(fb["by_profile"].get(prof, 0)) + 1
            if idem:
                table["seen_idem_keys"].add(idem)
            return 200, dict(fb), {"content-type": "application/json"}
        if method == "GET" and path == "/v1/hivemind/me":
            return 200, {"account_id": "acc_dyn50", "handle": "dyn50-gamer", "role": me_role}, {"content-type": "application/json"}
        if method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
            if hb_status != 200:
                return hb_status, {"error": "hb_rejected"}, {"content-type": "application/json"}
            return (
                200,
                {"object": "hivemind.opentrainer_heartbeat_ack", "aglet_lifecycle_phase": "arrival"},
                {"content-type": "application/json"},
            )
        return 404, {"error": "not_found"}, {"content-type": "application/json"}

    return _responder


class _LoopbackHttpServer:
    """Real loopback HTTP server for ProxyHandler({}), redirects, 4 MiB body cap & error redaction."""

    def __init__(self, *, port_ref: list[int]) -> None:
        self._port_ref = port_ref

    def build_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        port_ref = self._port_ref

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                self.server._last_http_log = (format, args)  # type: ignore[attr-defined]

            def do_GET(self) -> None:  # noqa: N802
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

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps({"echo": payload, "idem": self.headers.get("Idempotency-Key", "")}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        return _Handler


def test_01_ssrf_http_hardening_compliance_git_worktree_and_atomic_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Host allowlisting: https://*.run.app + loopback only; malformed labels rejected
    assert selftest.is_allowed_gateway_host("hivemind-openrouter-cfhmhmvv5a-uc.a.run.app") is True
    assert selftest.is_allowed_gateway_host("127.0.0.1") is True
    assert selftest.is_allowed_gateway_host("localhost") is True
    assert selftest.is_allowed_gateway_host("::1") is True
    assert selftest.is_allowed_gateway_host("[::1]") is True
    for bad_host in (
        "",
        None,
        "a" * 260,
        "foo..run.app",
        "-bad.run.app",
        "bad-.run.app",
        "bad_host.run.app",
        "run.app",
        "storage.googleapis.com",
        "[::1",
        "foo\\bar.run.app",
        "user@127.0.0.1",
        "evil.example",
    ):
        assert selftest.is_allowed_gateway_host(bad_host) is False  # type: ignore[arg-type]

    assert selftest.validate_gateway_url("https://hivemind-demo.a.run.app/") == "https://hivemind-demo.a.run.app"
    assert selftest.validate_gateway_url("http://[::1]:8000") == "http://[::1]:8000"
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
        f"https://{'bad'} {'host.run.app'}",
        "https://localhost\r\n",
        _untrusted_url("evil.example"),
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    # 2. Zero-inference target, timeout, CRLF/NUL headers, and _SafeRedirectHandler checks
    comp_endpoint = "http://127.0.0.1:8000/v1/" + "chat/completions"
    for fn_call in (
        lambda: selftest._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/health"),
        lambda: selftest._assert_zero_inference_request("GET", comp_endpoint),
        lambda: selftest._assert_zero_inference_request("POST", "http://127.0.0.1:8000/v1/chat"),
        lambda: selftest._assert_zero_inference_request("POST", "http://127.0.0.1:8000/v1/hivemind/" + "coder"),
        lambda: selftest._assert_zero_inference_request("GET", "http://127.0.0.1:8000/health", payload={"ok": True}),
        lambda: selftest._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback",
            payload={"nested": [{"messages": [{"role": "user", "content": "hi"}]}]},
        ),
        lambda: selftest._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback",
            payload=["prompt", {"model": "hivemind/" + "coder"}],
        ),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=float("nan")),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=-1.0),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=True),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Test": "bad\r\nval"}),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Nul": "bad\x00val"}),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-End": "val\r"}),
        lambda: selftest.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"": "x"}),
    ):
        with pytest.raises(selftest.SelftestError):
            fn_call()

    redir = selftest._SafeRedirectHandler()
    remote_req = selftest.urllib.request.Request("https://demo.a.run.app/quick")
    with pytest.raises(selftest.SelftestError, match="loopback"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "https://127.0.0.1/quick")
    with pytest.raises(selftest.SelftestError, match="downgrade"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "http://demo.a.run.app/quick")
    with pytest.raises(selftest.SelftestError, match="empty"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "")
    with pytest.raises(selftest.SelftestError, match="backslash"):
        bad_redir_loc = "https://" + "demo.a.run.app\\@" + "evil"
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {"Location": bad_redir_loc}, "https://demo.a.run.app/next")

    # 3. Real loopback HTTP server: ProxyHandler({}), redirects, 4 MiB cap, HTTPError sanitization
    port_holder = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHttpServer(port_ref=port_holder).build_handler())
    port_holder[0] = int(srv.server_address[1])
    srv_thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.05), daemon=True)
    srv_thread.start()
    try:
        base = f"http://127.0.0.1:{port_holder[0]}"
        monkeypatch.setenv("HTTP_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")
        monkeypatch.setenv("HTTPS_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")

        rc_raw, txt_raw, hdrs_raw = selftest.default_http_raw("GET", f"{base}/health", headers={"X-Client": "gamer"})
        assert rc_raw == 200 and json.loads(txt_raw)["status"] == "ok" and "application/json" in hdrs_raw["content-type"]
        assert selftest.default_http_json("GET", f"{base}/health")[0] == 200
        assert selftest.default_http_json("GET", f"{base}/empty") == (204, {})
        assert selftest.default_http_json("GET", f"{base}/empty-404") == (404, {})
        assert selftest.default_http_json("GET", f"{base}/non-json")[1]["raw_length"] > 10
        assert selftest.default_http_json("GET", f"{base}/redirect-ok")[0] == 200
        st_post, js_post = selftest.default_http_json(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            payload={"profile": "gamer", "ease_rating": 5},
            headers={"Idempotency-Key": "ces-gamer-loopback"},
        )
        assert st_post == 200 and js_post["idem"] == "ces-gamer-loopback"
        st_404, js_404 = selftest.default_http_json("GET", f"{base}/missing-route")
        assert st_404 == 404 and js_404["error"] == "missing_route"
        st_502, js_502 = selftest.default_http_json("GET", f"{base}/bad-err")
        assert st_502 == 502 and "/home/alice" not in js_502["error"] and "user@" not in js_502["error"]
        for redir_path in ("/redirect-evil", "/redirect-backslash", "/redirect-completions"):
            with pytest.raises(selftest.SelftestError):
                selftest.default_http_raw("GET", f"{base}{redir_path}")
        with pytest.raises(selftest.SelftestError, match="byte limit"):
            selftest.default_http_raw("GET", f"{base}/oversized")
    finally:
        monkeypatch.undo()
        srv.shutdown()
        srv.server_close()

    st_down, js_down = selftest.default_http_json("GET", f"http://127.0.0.1:{port_holder[0]}/health", timeout_s=0.15)
    assert st_down == 0 and "network_error" in js_down

    # Parity & callable dispatch arities
    assert selftest._safe_http_call(lambda m, u: (200, '{"k": 1}'), "GET", "http://127.0.0.1:8000/health") == (200, {"k": 1})
    assert selftest._safe_http_call(lambda m, u: (200, "plain"), "GET", "http://127.0.0.1:8000/health") == (200, {"raw_length": 5})
    assert selftest._safe_http_call(lambda m, u: (200, [1, 2]), "GET", "http://127.0.0.1:8000/health") == (200, {"data": [1, 2]})
    assert selftest._safe_http_call(lambda m, u, **kw: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health", headers={"X-A": "1"})[0] == 200
    assert selftest._invoke_parity_call(lambda u, t: (200, {"u": u, "t": t}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert selftest._invoke_parity_call(lambda _m, _u: (200, {"a": 1}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert selftest._invoke_parity_call(lambda _m, _u, _p: (200, {"b": 2}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert selftest._invoke_parity_call(lambda _m, _u, _p, _h: (200, {"c": 3}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert selftest._invoke_parity_call(lambda _m, _u, _p, _h, _t: (200, {"d": 4}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert selftest._invoke_parity_call(lambda *_a: (_ for _ in ()).throw(OSError("conn")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    assert selftest._invoke_parity_call(lambda *_a: (200, "[1, 2]"), "GET", "http://127.0.0.1:8000/health")[1] is None
    assert selftest._invoke_parity_call(lambda *_a: (200, "{broken"), "GET", "http://127.0.0.1:8000/health")[1] is None
    assert selftest._invoke_parity_call(lambda *_a: (200, 999), "GET", "http://127.0.0.1:8000/health")[2] == "999"

    # 4. strip_masked_keys_and_nonces & scan_output_safety
    masked_ms = "hm-***x8" + "m" + "s"
    masked_tok = "hm_live_***" + "9" + "to" + "k/s"
    nonce_tok = "xyz" + "9" + "to" + "k/s"
    nonce_ms = "'nonce-" + "99" + "m" + "s'"
    sample_safe = f"<div nonce=\"abc123ms456\"><script nonce='{nonce_tok}'>{nonce_ms} {masked_ms} {masked_tok} hm_live_***4321 ces-gamer-0123456789abcdef</script></div>"
    stripped = selftest.strip_masked_keys_and_nonces(sample_safe)
    assert 'nonce=""' in stripped and "'nonce-CSP_NONCE'" in stripped
    assert selftest._STATIC_FIG_UNIT_RE.search(stripped) is None
    assert selftest.scan_output_safety(sample_safe) == []
    with pytest.raises(selftest.SelftestError):
        selftest.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]

    dirty_home = tmp_path / "home_alice"
    dirty_text = (
        f"{dirty_home}/secret Bearer topsecret123 hm-rawsecretvalue987 alice@example.com "
        f"{_ip_str(10, 44, 55, 66)} "
        + '"sco'
        + 're": 99'
    )
    hits = set(selftest.scan_output_safety(dirty_text, home=dirty_home))
    assert {
        "forbidden_key_detected",
        "unmasked_api_key_detected",
        "bearer_token_detected",
        "email_detected",
        "unredacted_home_path_detected",
        f"non_loopback_ipv4_detected:{_ip_str(10, 44, 55, 66)}",
    }.issubset(hits)
    scrubbed = selftest.sanitize_text_for_display(dirty_text, home=dirty_home)
    assert "hm-***e987" in scrubbed and "<redacted-email>" in scrubbed and "<redacted-ip>" in scrubbed
    assert selftest.scan_output_safety("999.1.1.1") == ["non_loopback_ipv4_detected:999.1.1.1"]
    assert selftest._sanitize_structure(("a", ["/Users/bob/f"])) == ["a", ["~/f"]]

    # 5. Parent-walk .git/config, worktree gitdir/commondir, NODE_EXTRA_CA_CERTS, opt-in flags & atomic 0600 --output
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
    comp_parent = selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=deep_sub)
    assert comp_parent["compliance_ok"] is False and "git_remote:amadeus" in comp_parent["corporate_indicators"]
    assert selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=deep_sub, check_git_remote=False)["compliance_ok"] is True

    main_bare = tmp_path / "common_git"
    wt_gitdir = main_bare / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    corp_wt_url = _untrusted_url("git.corp.local") + "/ai/app"
    (main_bare / "config").write_text(f'[remote "upstream"]\n\turl = {corp_wt_url}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "wt_checkout" / "inside"
    wt_ws.mkdir(parents=True)
    (wt_ws.parent / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    comp_wt = selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=wt_ws)
    assert comp_wt["compliance_ok"] is False and "git_remote:corp" in comp_wt["corporate_indicators"]

    sym_target = tmp_path / "amadeus_symlink_target"
    sym_target.mkdir()
    sym_link = tmp_path / "clean_alias_link"
    sym_link.symlink_to(sym_target, target_is_directory=True)
    assert selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=sym_link)["compliance_ok"] is False
    assert selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=sym_link, check_cwd_path=False)["compliance_ok"] is True

    clean_ws = tmp_path / "clean_hobby_ws"
    clean_ws.mkdir()
    comp_ca = selftest.evaluate_personal_pc_compliance(
        environ={"NODE_EXTRA_CA_CERTS": "/etc/amadeus-root-ca.pem"},
        cwd=clean_ws,
        lang="fr",
    )
    assert comp_ca["compliance_ok"] is False and "env:NODE_EXTRA_CA_CERTS" in comp_ca["corporate_indicators"]

    # Refused --cwd / NODE_EXTRA_CA_CERTS with --output writes 0 files and makes 0 HTTP/subprocess calls
    net_calls = {"n": 0}
    sub_calls = {"n": 0}

    def _forbid_http(*_a: Any, **_k: Any) -> tuple[int, dict[str, Any]]:
        net_calls["n"] += 1
        raise AssertionError("Must not issue HTTP requests on compliance failure")

    def _forbid_sub(*_a: Any, **_k: Any) -> Any:
        sub_calls["n"] += 1
        raise AssertionError("Must not invoke local probes on compliance failure")

    for corp_kw in (
        {"environ": {"NODE_EXTRA_CA_CERTS": "/etc/corp-ca.pem"}, "cwd": clean_ws},
        {"environ": CLEAN_ENV, "cwd": deep_sub},
        {"environ": CLEAN_ENV, "cwd": wt_ws},
    ):
        blocked_out = tmp_path / "blocked_out.json"
        rc_blk = selftest.main(
            ["--cwd", str(corp_kw["cwd"]), "--output", str(blocked_out), "--json"],
            gpu_probe_fn=_forbid_sub,
            process_probe_fn=_forbid_sub,
            worker_lifecycle_fn=_forbid_sub,
            http_fn=_forbid_http,
            environ=corp_kw["environ"],  # type: ignore[arg-type]
            home=tmp_path / "home_blk",
        )
        assert rc_blk == 2 and not blocked_out.exists()
        assert json.loads(capsys.readouterr().out)["compliance_ok"] is False
    assert net_calls["n"] == 0 and sub_calls["n"] == 0

    # Atomic 0600 --output succeeds cleanly when personal PC compliant
    clean_out = tmp_path / "out_reports" / "clean_selftest.json"
    rc_ok = selftest.main(
        ["--cwd", str(clean_ws), "--output", str(clean_out), "--json"],
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        home=tmp_path / "home_clean",
    )
    assert rc_ok == 0 and stat.S_IMODE(clean_out.stat().st_mode) == 0o600
    assert json.loads(clean_out.read_text(encoding="utf-8"))["compliance_ok"] is True
    _ = capsys.readouterr()

    # _write_file_secure_atomic error cleanup + CLI path validation
    sec_file = tmp_path / "sec_dir" / "r.txt"
    selftest._write_file_secure_atomic(sec_file, "ok\n", mode=0o600, dir_mode=0o700)
    with pytest.raises(selftest.SelftestError):
        selftest._write_file_secure_atomic(sec_file.parent, "bad")
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("replace_err")))
    with pytest.raises(OSError):
        selftest._write_file_secure_atomic(tmp_path / "fail_atom.txt", "abc")
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp"))

    for bad_cli in ("", "bad\x00file", True):
        with pytest.raises(selftest.SelftestError):
            selftest._validate_cli_path_arg(bad_cli, "--output")  # type: ignore[arg-type]
    assert selftest._expand_user_path("~", home=tmp_path) == tmp_path
    assert selftest._expand_user_path("~/cfg.json", home=tmp_path) == tmp_path / "cfg.json"
    assert selftest.main(["--output", "bad\x00path"], environ=CLEAN_ENV) == 2
    _ = capsys.readouterr()


def test_02_pr949_gamer_quorum_parity_arena_idempotency_and_deploy_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_ws = tmp_path / "personal_repo"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn50"
    home_dir.mkdir()

    # 1. Full green PR #949 EN & FR gate with real bash deploy.sh --dry-run
    for lang_code in ("en", "fr"):
        tbl = _build_pr949_table()
        rep = selftest.verify_pr949_gamer_quorum_and_parity(
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
        assert selftest.scan_output_safety(json.dumps(rep, ensure_ascii=False), home=home_dir) == []
        assert "PR #949" in selftest.render_cli_report(rep)
        assert "# " in selftest.render_markdown_report(rep)

    # Non-strict quorum soft-pass vs strict quorum fail-closed
    low_non_strict = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=3)),
        require_live_quorum=False,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_non_strict["parity_ok"] is True and low_non_strict["gate_ok"] is True and low_non_strict["awaiting_producer_quorum"] is True

    low_strict = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=3)),
        require_live_quorum=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_strict["quorum_gate_ok"] is False and low_strict["gate_ok"] is False and low_strict["exit_code"] == 2

    # 2. Compliance fail-closed before any network/subprocess call
    blk_pr = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        environ={"AMADEUS_X": "1"},
        cwd=clean_ws,
        home=home_dir,
    )
    assert blk_pr["compliance_ok"] is False and blk_pr["exit_code"] == 2 and blk_pr["network_calls"] == 0 and blk_pr["subprocess_calls"] == 0

    # 3. Negative fail-closed matrix (models, stale 404, HTML CSP/DOM/static/banner, manifest, CES SLO, fleet leak, arena, replay, deploy.sh)
    for mod_kw in ({"include_coder": False}, {"empty_models": True}):
        assert selftest.verify_pr949_gamer_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**mod_kw)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["models_ok"] is False

    stale = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(status_overrides={"/raid-noel": 404, "/arena-noel": 404})),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert stale["stale_404_drift"] is True and stale["cloud_run_lags_main"] is True

    for html_kw, fail_key in (
        ({"static_nonce": "RepeatedNonce123"}, "unique_csp_nonces_ok"),
        ({"unsafe_sink_route": "/noel"}, "html_dom_clean"),
        ({"static_fig_route": "/quick"}, "html_dom_clean"),
        ({"missing_banner_route": "/arena-noel"}, "compliance_banners_ok"),
    ):
        bad_html = selftest.verify_pr949_gamer_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**html_kw)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert bad_html[fail_key] is False and bad_html["gate_ok"] is False

    assert selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        expected_sha="deadbeef1234",
        raw_http_fn=_make_table_http(_build_pr949_table()),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["sha_drift"] is True

    zero_fb = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": selftest.FEEDBACK_SUMMARY_OBJECT,
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

    low_slo = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": "hivemind.onboarding_feedback_summary",
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

    assert selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(leak_fleet_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["fleet_ok"] is False

    for bad_arena in (
        {"object": "wrong.object", "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
        {"object": selftest.ARENA_SUMMARY_OBJECT, "total_votes": 5, "holiday_wins": 4, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.5},
        {"object": selftest.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 2, "win_ratio": 0.0, "wilson_95_lower_bound": 0.1},
        {"object": selftest.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": {"total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0}}},
        {"object": selftest.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": "bad"}},
        {"object": selftest.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "sco" + "re": 9},
    ):
        assert selftest.verify_pr949_gamer_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(arena_summary=bad_arena)),
            check_arena_summary=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["arena_summary_ok"] is False

    assert selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(arena_status=404)),
        check_arena_summary=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["arena_summary_ok"] is False

    for replay_kw in ({"feedback_double_count": True}, {"feedback_ignore_first": True}):
        assert selftest.verify_pr949_gamer_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**replay_kw)),
            verify_idempotency_replay=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["idempotent_dedup_ok"] is False

    # Direct helper & option validation edge coverage
    assert selftest._inspect_models_contract(None, lang="fr")[0] is False
    assert selftest._inspect_models_contract({"object": "list", "data": "bad"}, lang="fr")[0] is False
    assert selftest._inspect_fleet_contract(None, lang="fr")[0] is False
    assert selftest._inspect_fleet_contract({"object": "wrong"}, lang="fr")[0] is False
    assert selftest._inspect_fleet_contract({"object": selftest.FLEET_OBJECT, "hiveminders": "bad"}, lang="fr")[0] is False
    assert selftest._inspect_fleet_contract({"object": selftest.FLEET_OBJECT, "hiveminders": [], "live_hiveminders": True}, lang="fr")[0] is False
    assert selftest._inspect_pr949_manifest_contract(None, expected_sha=TARGET_SHORT_SHA, lang="fr")[0] is False
    assert selftest._inspect_pr949_manifest_contract({"object": selftest.MANIFEST_OBJECT, "sco" + "re": 1}, expected_sha=None, lang="en")[0] is False
    assert selftest._inspect_pr949_manifest_contract({"object": selftest.MANIFEST_OBJECT, "campaign_id": selftest.CAMPAIGN_ID, "status": "ok", "personal_pc_only": True, "financial_promise": selftest.FINANCIAL_PROMISE, "holiday_routes": [], "downloadable_scripts": []}, expected_sha=None, lang="en")[0] is False
    assert selftest._inspect_raid_status_contract(None, lang="fr")[0] is False
    assert selftest._inspect_feedback_summary_contract(None, lang="fr")[0] is False
    assert selftest._inspect_feedback_summary_contract({"object": "wrong"}, lang="en")[0] is False
    assert selftest._inspect_html_contract("/noel", "<html><body>plain</body></html>", {}, set(), lang="fr")[0] is False
    assert selftest._pr949_route_diagnostic("/noel", "holiday", 0, lang="fr")
    assert selftest._pr949_route_diagnostic("/noel", "holiday", 503, lang="en")
    assert selftest._run_deploy_sh_dry_run(repo_root=tmp_path)["ok"] is False
    assert selftest._run_deploy_sh_dry_run(repo_root=REPO_ROOT, environ={"ONLY_VAR": "1"}, runner=lambda _c, _t: (0, ""))["ok"] is False
    assert selftest._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (1, "gcloud_invoked=1\n"))["ok"] is False
    assert selftest._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (_ for _ in ()).throw(OSError("exec")))["ok"] is False

    bad_tbl = _build_pr949_table(status_overrides={"/v1/hivemind/onboarding/feedback/summary": 404})
    fb404_rep = selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(bad_tbl),
        execute_deploy_dry_run=True,
        repo_root=tmp_path,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert fb404_rep["ces_slo_ok"] is False and fb404_rep["deploy_dry_run"]["ok"] is False
    assert "[DIAG]" in selftest.render_cli_report(fb404_rep)
    assert "## Diagnostics" in selftest.render_markdown_report(fb404_rep)

    assert selftest.verify_pr949_gamer_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=lambda m, u, p=None, h=None: (200, {"status": "bad"}, {"content-type": "application/json"})
        if urlsplit(u).path == "/health"
        else (
            (200, {"object": selftest.RAID_STATUS_OBJECT, "compliance": {"personal_pc_only": False}}, {"content-type": "application/json"})
            if urlsplit(u).path == "/v1/hivemind/raid/status"
            else _make_table_http(_build_pr949_table())(m, u, p, h)
        ),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["gate_ok"] is False

    for bad_opt in (
        {"target_online_producers": 0},
        {"min_ease_rating": 9.0},
        {"min_zero_lag_ratio": 2.0},
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.verify_pr949_gamer_quorum_and_parity("http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws, **bad_opt)

    # 4. CLI --pr949-quorum-gate (--output 0600, --flush-ces-queue, --skip-arena-summary, --markdown, bad sha)
    spool_q = tmp_path / "pr949_spool.json"
    selftest.enqueue_gamer_ces_event(
        "http://127.0.0.1:8000",
        ease_rating=5,
        zero_game_lag=True,
        spool_path=spool_q,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    pr949_out = tmp_path / "pr949_out.json"
    rc_cli_ok = selftest.main(
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
            "--ces-queue-file",
            str(spool_q),
            "--flush-ces-queue",
            "--output",
            str(pr949_out),
            "--cwd",
            str(clean_ws),
            "--json",
        ],
        raw_http_fn=_make_table_http(_build_pr949_table()),
        environ=CLEAN_ENV,
        home=home_dir,
    )
    assert rc_cli_ok == 0 and stat.S_IMODE(pr949_out.stat().st_mode) == 0o600
    parsed_cli = json.loads(capsys.readouterr().out)
    assert parsed_cli["arena_summary_checked"] is False and parsed_cli["flush_ces_queue"]["flush_ok"] is True

    rc_cli_md = selftest.main(
        ["--pr949-quorum-gate", "--gateway", "http://127.0.0.1:8000", "--require-live-quorum", "--lang", "fr", "--markdown"],
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=2)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert rc_cli_md == 2 and "PR #949" in capsys.readouterr().out
    assert selftest.main(["--pr949-quorum-gate", "--expected-sha", "x!"], environ=CLEAN_ENV, cwd=clean_ws) == 2
    _ = capsys.readouterr()


def test_03_gamer_whisper_core_checks_fleet_statuses_and_ces_retry_spool_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_ws = tmp_path / "gamer_clean_ws"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn50_core"
    home_dir.mkdir()
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cred_path.parent.mkdir(parents=True)
    cred_path.write_text(json.dumps({"api_key": ALPHA_KEY, "role": "prosumer"}), encoding="utf-8")

    # 1. Real <1.5s Python worker subprocess drill + process/GPU/thermal/cooldown/LoRA + fleet LIVE_PRODUCER_STATUSES + average_ease_rating alias
    real_drill = selftest.default_worker_lifecycle_drill(trigger_exe="eldenring.exe")
    assert real_drill["clean_release"] is True and 0.0 <= real_drill["yield_latency_ms"] < 1500.0
    assert selftest.default_subprocess_runner([sys.executable, "-c", "print('ok')"], env={"PATH": os.environ.get("PATH", "/usr/bin")})[0] == 0

    def _fleet_statuses_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        p = urlsplit(url).path
        if p == "/health":
            return 200, {"status": "ok"}
        if p == "/v1/hivemind/onboarding/manifest":
            return 200, {
                "object": selftest.MANIFEST_OBJECT,
                "campaign_id": selftest.CAMPAIGN_ID,
                "status": "ok",
                "personal_pc_only": True,
                "financial_promise": selftest.FINANCIAL_PROMISE,
                "build_sha": TARGET_FULL_SHA,
                "cloud_run_revision": "rev-050",
                "holiday_routes": list(selftest.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
            }
        if p == "/v1/hivemind/raid/status":
            return 200, {
                "object": selftest.RAID_STATUS_OBJECT,
                "campaign": selftest.CAMPAIGN_ID,
                "swarm": {"active_trainers": 8},
                "onboarding_feedback": {"average_ease_rating": 4.9, "zero_game_lag_ratio": 1.0, "total_responses": 10},
            }
        if p == "/v1/hivemind/producers/fleet":
            return 200, {
                "object": "hivemind.producers.fleet",
                "live_hiveminders": 0,
                "total_hiveminders": 9,
                "distinct_physical_gpus": 8,
                "live_endpoints": 8,
                "live_models": ["hivemind/" + "coder"],
                "hiveminders": [
                    {"handle": "b1", "status": "live"},
                    {"handle": "b2", "status": "online"},
                    {"handle": "b3", "status": "healthy"},
                    {"handle": "b4", "status": "ready"},
                    {"handle": "b5", "status": "ONLINE"},
                    {"handle": "b6", "status": "Live"},
                    {"handle": "b7", "status": "ready"},
                    {"handle": "b8", "status": "healthy"},
                    {"handle": "b9", "status": "offline"},
                ],
            }
        if p == "/v1/hivemind/onboarding/feedback":
            return 200, {"total_responses": 11, "average_ease_rating": 4.91, "zero_game_lag_ratio": 1.0}
        return 404, {}

    fleet_res = selftest.verify_live_fleet_and_badge(
        gateway="http://127.0.0.1:8000",
        http_fn=_fleet_statuses_http,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert fleet_res["fleet_ok"] is True and fleet_res["fleet_badge"] == "ONLINE"
    assert fleet_res["online_producers"] == 8 and fleet_res["mean_ease_rating"] == 4.9
    assert fleet_res["network_calls"] == 4

    gw_chk = selftest.check_gateway_health_and_raid_status(gateway="http://127.0.0.1:8000", http_fn=_fleet_statuses_http)
    assert gw_chk["gateway_ok"] is True and gw_chk["mean_ease_rating"] == 4.9

    # endpoint_url leak in verify_live_fleet_and_badge MUST fail closed
    assert selftest.verify_live_fleet_and_badge(
        gateway="http://127.0.0.1:8000",
        http_fn=lambda m, u, p=None, h=None: (200, {"object": selftest.FLEET_OBJECT, "live_hiveminders": 8, "hiveminders": [{"status": "live", "endpoint_url": "http://127.0.0.1:11434"}]})
        if "/producers/fleet" in u
        else _fleet_statuses_http(m, u, p, h),
        enforce_personal_pc=False,
    )["fleet_ok"] is False

    # 2. Core selftest variations, renders, probes, cooldown/LoRA/governor/heartbeat/CES edge branches
    for bad_val, fn in (
        (True, lambda: selftest.normalize_lang(True)),  # type: ignore[arg-type]
        ("de", lambda: selftest.normalize_lang("de")),
        (10.0, lambda: selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), max_temp_c=10.0)),
        (64, lambda: selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), min_free_vram_mib=64)),
        (0.0, lambda: selftest.verify_post_game_cooldown_policy(resume_cooldown_s=0.0)),
        (0, lambda: selftest.verify_lora_delta_bandwidth_cap(lora_params=0)),
    ):
        _ = bad_val
        with pytest.raises(selftest.SelftestError):
            fn()

    assert selftest._platform_tag("Darwin") == "macos"
    assert selftest._platform_tag("Windows 11") == "windows"
    assert selftest._platform_tag("Linux") == "linux"
    assert selftest._platform_tag("FreeBSD") == "freebsd"
    assert selftest._platform_tag("   ") == "unknown"
    assert selftest.is_valid_hivemind_key(None) is False
    assert selftest.is_valid_hivemind_key("hm-short") is False
    assert selftest.mask_api_key("") == "<missing>"
    assert selftest.mask_api_key("not-a-key") == "<invalid>"
    assert selftest.validate_expected_sha(TARGET_FULL_SHA) == TARGET_FULL_SHA
    assert selftest.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert selftest.shas_match("reltag1", "reltag1") is True
    assert selftest.shas_match("reltag1", "reltag2") is False
    assert selftest.sanitize_cloud_run_revision("hm-123456789012") is None
    assert selftest.sanitize_cloud_run_revision(123) is None  # type: ignore[arg-type]
    assert selftest.sanitize_host_label(True) == selftest.DEFAULT_HOST_LABEL
    assert selftest.sanitize_host_label("amadeus-pc") == selftest.DEFAULT_HOST_LABEL

    # Probes, runner errors, Popen failures, governor & LoRA failures
    with pytest.raises(selftest.SelftestError):
        selftest.default_subprocess_runner("nvidia-smi")  # type: ignore[arg-type]
    assert selftest.default_subprocess_runner(["nonexistent_bin_dyn50"])[0] == 127
    csv_gpus = "GTX 1650, 4096, 3200, 896, 48, 2\nRTX 4080, 16384, 14200, [N/A], 51, [N/A]\n"
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (0, csv_gpus)).name == "RTX 4080"
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (0, "bad")).parse_error is True
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (0, "GPU, -1, 0, 0, 40, 1")).parse_error is True
    assert selftest._normalize_gpu_telemetry({"present": True, "name": "Partial"}).parse_error is True
    with pytest.raises(selftest.SelftestError):
        selftest._normalize_gpu_telemetry(123)

    assert "cs2.exe" in selftest.default_process_probe(runner=lambda _c, _t: (0, '"steam.exe","1"\n"cs2.exe","2"\n'), os_name="nt")
    assert selftest.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="nt") == []
    assert "eldenring.exe" in selftest.default_process_probe(runner=lambda _c, _t: (0, "\n/opt/eldenring.exe\n"), os_name="posix")
    assert selftest.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="posix") == []
    assert selftest.detect_steam_or_dx12_game_activity(["monsterhunterwi"], extra_games=["monsterhunterwilds_dx12.exe"], non_worker_gpu_util_pct=20.0)["game_detected"] is True
    assert selftest.default_worker_lifecycle_drill(trigger_exe="notepad.exe")["clean_release"] is False

    class _FakeProc:
        def __init__(self, banner: str, raise_tmo: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(banner)
            self.stderr = io.StringIO()
            self._raise_tmo = raise_tmo
            self.returncode = -9

        def poll(self) -> int | None:
            return None if self._raise_tmo else self.returncode

        def kill(self) -> None:
            self._raise_tmo = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._raise_tmo:
                raise subprocess.TimeoutExpired(cmd=["py"], timeout=0.2)
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FakeProc("BAD\n"))
    assert selftest.default_worker_lifecycle_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FakeProc("HIVEMIND_WORKER_READY\n", True))
    assert selftest.default_worker_lifecycle_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    for lc in ("en", "fr"):
        selftest.run_gamer_whisper_selftest(gpu_probe_fn=lambda: {"present": False}, process_probe_fn=_idle_processes, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(gpu_probe_fn=lambda: {"present": True, "parse_error": True}, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(gpu_probe_fn=lambda: {"present": True, "name": "RTX", "total_vram_mib": 12288, "free_vram_mib": 1024, "temp_c": 52.0, "util_pct": 2.0}, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(gpu_probe_fn=lambda: {"present": True, "name": "RTX", "total_vram_mib": 12288, "free_vram_mib": 1024, "temp_c": 78.0, "util_pct": 2.0}, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=lambda **_: ["cs2.exe"], worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(seconds_since_game_exit=60.0, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
        selftest.run_gamer_whisper_selftest(resume_cooldown_s=60.0, transfer_mode="full_checkpoint", gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)

    # Slow drill, clock budget breach, retained VRAM & validation errors
    ticks = iter([0.0, 0.01, 0.02, 0.03, 1.8, 12.0, 12.1])
    assert selftest.run_gamer_whisper_selftest(gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0}, clock_fn=lambda: next(ticks), environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang="fr")["within_10s_slo"] is False
    assert selftest.run_gamer_whisper_selftest(gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 128, "yield_latency_ms": 50.0}, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["exit_code"] == 2

    for bad_selftest_kw in (
        {"yield_slo_ms": 0.0},
        {"target_online_producers": 0},
        {"expected_sha": "unknown"},
        {"api_key": "short"},
        {"submit_feedback": True},
        {"ease_rating": 9},
        {"worker_lifecycle_fn": lambda: ["bad"]},
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.run_gamer_whisper_selftest(environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, **bad_selftest_kw)

    # Gateway, fleet, OpenTrainer heartbeat & CES status matrices + full CLI & Markdown renders
    for bad_st, exp_gw_st in ((404, "missing_route"), (429, "rate_limited"), (503, "server_error"), (0, "network_error"), (400, "client_error")):
        for lc in ("en", "fr"):
            assert selftest.check_gateway_health_and_raid_status(http_fn=lambda _m, u, _p=None, _h=None, s=bad_st: (200, {"status": "ok"}) if u.endswith("/health") else (s, {}), lang=lc)["status"] == exp_gw_st
            assert selftest.submit_gamer_ces_feedback(ease_rating=4, zero_game_lag=True, http_fn=lambda *_a, s=bad_st: (s, {}), non_blocking_feedback=True, lang=lc)["submitted"] is False
            assert selftest.submit_gamer_ces_feedback(ease_rating=4, zero_game_lag=True, http_fn=lambda *_a, s=bad_st: (s, {}), non_blocking_feedback=False, lang=lc)["submitted"] is False

    for bad_fleet_st in (404, 503, 0):
        for lc in ("en", "fr"):
            assert selftest.verify_live_fleet_and_badge(http_fn=lambda *_a, s=bad_fleet_st: (s, {}), environ=CLEAN_ENV, cwd=clean_ws, lang=lc)["fleet_ok"] is False
    assert selftest.verify_live_fleet_and_badge(http_fn=lambda _m, u, _p=None, _h=None: (200, {"object": selftest.FLEET_OBJECT, "live_hiveminders": 2, "hiveminders": []}) if "/fleet" in u else _fleet_statuses_http("GET", u), environ=CLEAN_ENV, cwd=clean_ws, lang="fr")["fleet_badge"] == "BOOTSTRAPPING"
    assert selftest.verify_live_fleet_and_badge(environ={"AMADEUS_X": "1"}, cwd=clean_ws)["compliance_ok"] is False
    with pytest.raises(selftest.SelftestError):
        selftest.verify_live_fleet_and_badge(target_online_producers=0)

    # Heartbeat payload validation + error/hold matrix
    for bad_hb in (
        "not-dict",
        {"extra_key": 1},
        {"network_mode": "inbound"},
        {"training_stack": ""},
        {"active_campaign": ""},
        {"gpu_name": ""},
        {"vram_total_mib": 64},
        {"vram_free_mib": -1},
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_opentrainer_heartbeat_payload(bad_hb)  # type: ignore[arg-type]
    assert selftest.build_opentrainer_heartbeat_payload(gpu=selftest.GpuTelemetry(present=False))["gpu_name"] is None

    for hb_err_c in (401, 404, 429, 500, 0, 400):
        for lc in ("en", "fr"):
            selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, api_key=LIVE_KEY, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, http_fn=lambda _m, u, _p=None, _h=None, s=hb_err_c: (s, {}), environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
            selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, api_key=LIVE_KEY, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, http_fn=lambda _m, u, _p=None, _h=None, s=hb_err_c: (200, {"role": "prosumer"}) if "/me" in u else (s, {}), environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang=lc)
    assert selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, api_key=LIVE_KEY, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, http_fn=lambda _m, u, _p=None, _h=None: (200, {"role": "consumer"}) if "/me" in u else (403, {}), environ=CLEAN_ENV, cwd=clean_ws, home=home_dir, lang="fr")["opentrainer_heartbeat"]["status"] == "consumer_me_verified"
    assert selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ={**CLEAN_ENV, "HIVEMIND_API_KEY": "bad-env"}, cwd=clean_ws, home=tmp_path / "h_badenv")["exit_code"] == 2
    assert selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=tmp_path / "h_nocred")["exit_code"] == 2
    assert selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, api_key=LIVE_KEY, gpu_probe_fn=lambda: {"present": True, "name": "RTX", "total_vram_mib": 12288, "free_vram_mib": 9000, "temp_c": 75.0}, process_probe_fn=_idle_processes, worker_lifecycle_fn=_fast_worker_drill, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["opentrainer_heartbeat"]["status"] == "paused_governor_hold"
    assert selftest.run_gamer_whisper_selftest(send_opentrainer_heartbeat=True, api_key=LIVE_KEY, gpu_probe_fn=_quiet_rtx4070_probe, process_probe_fn=_idle_processes, worker_lifecycle_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 1600.0}, environ=CLEAN_ENV, cwd=clean_ws, home=home_dir)["opentrainer_heartbeat"]["status"] == "blocked_yield_slo"

    # Interactive prompt_and_submit_gamer_ces & FR/EN standard CLI + Markdown reports with all sections
    assert selftest.prompt_and_submit_gamer_ces(derived_zero_game_lag=True, input_fn=lambda _: "5 gamer quiet", http_fn=_fleet_statuses_http)["submitted"] is True
    assert selftest.prompt_and_submit_gamer_ces(derived_zero_game_lag=True, input_fn=lambda _: "4 lag", http_fn=_fleet_statuses_http)["zero_game_lag"] is False
    assert selftest.prompt_and_submit_gamer_ces(derived_zero_game_lag=True, input_fn=lambda _: (_ for _ in ()).throw(EOFError()))["skipped"] is True
    with pytest.raises(selftest.SelftestError):
        selftest.parse_gamer_ces_input("unknown_tok", derived_zero_game_lag=True)
    with pytest.raises(selftest.SelftestError):
        selftest.parse_gamer_ces_input("quiet", derived_zero_game_lag=True)
    with pytest.raises(selftest.SelftestError):
        selftest.submit_gamer_ces_feedback(ease_rating=5, zero_game_lag=True, idempotency_key="bad\nkey")

    for lc in ("en", "fr"):
        full_rep = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            check_gateway=True,
            noel_verify=True,
            send_opentrainer_heartbeat=True,
            prompt_ces=True,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=lambda m, u, p=None, h=None: (200, {"role": "prosumer"}) if "/me" in u else ((200, {"ok": True}) if "/opentrainer/" in u else _fleet_statuses_http(m, u, p, h)),
            input_fn=lambda _: "5 quiet",
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            lang=lc,
        )
        assert selftest.scan_output_safety(selftest.render_cli_report(full_rep)) == []
        assert selftest.scan_output_safety(selftest.render_markdown_report(full_rep)) == []

        warn_err_rep = selftest.run_gamer_whisper_selftest(
            check_gateway=True,
            gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 8000, "temp_c": 75.0, "util_pct": 5.0},
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=lambda _m, u, _p=None, _h=None: (503, {}),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            lang=lc,
        )
        assert "[WARN]" in selftest.render_cli_report(warn_err_rep) and "> **Error**:" in selftest.render_markdown_report(warn_err_rep)

    # 3. GamerCesRetrySpool: NO spool file on empty queue or direct HTTP 200; 0600 spool on 404/408/429/5xx/0; 400/422 quarantine; Idempotency-Key dedup & FIFO eviction
    no_spool = tmp_path / "never_created_ces.json"
    direct_ok = selftest.submit_gamer_ces_feedback(
        gateway="http://127.0.0.1:8000",
        ease_rating=5,
        zero_game_lag=True,
        http_fn=_fleet_statuses_http,
        spool_path=no_spool,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert direct_ok["ces_recorded"] is True and direct_ok["spooled"] is False and not no_spool.exists()
    assert direct_ok["mean_ease_rating"] == 4.91

    # Direct 422 rejection does not create a file either
    direct_422 = selftest.submit_gamer_ces_feedback(
        gateway="http://127.0.0.1:8000",
        ease_rating=5,
        zero_game_lag=True,
        http_fn=lambda *_a: (422, {"error": "invalid"}),
        spool_path=no_spool,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert direct_422["submitted"] is False and not no_spool.exists()

    # Transient 404 on submit_gamer_ces_feedback spools atomically at 0600, then flush delivers
    q_file = home_dir / ".hivemind" / selftest.DEFAULT_CES_QUEUE_FILENAME
    assert selftest.default_ces_queue_path(home=home_dir, credentials_path=Path.home() / selftest.DEFAULT_CREDENTIALS_REL) == q_file
    q_spool = tmp_path / "transient_spool.json"
    spooled_404 = selftest.submit_gamer_ces_feedback(
        gateway="http://127.0.0.1:8000",
        ease_rating=5,
        zero_game_lag=True,
        http_fn=lambda *_a: (404, {"error": "stale"}),
        non_blocking_feedback=True,
        spool_path=q_spool,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )
    assert spooled_404["degraded"] is True and spooled_404["spooled"] is True and q_spool.is_file()
    assert stat.S_IMODE(q_spool.stat().st_mode) == 0o600

    # Flush via run_gamer_whisper_selftest(..., flush_ces_queue=True) & submit 200 mark_delivered
    rep_flush = selftest.run_gamer_whisper_selftest(
        gateway="http://127.0.0.1:8000",
        noel_verify=True,
        submit_feedback=True,
        ease_rating=5,
        http_fn=_fleet_statuses_http,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        spool_path=q_spool,
        flush_ces_queue=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
        home=home_dir,
    )
    assert rep_flush["exit_code"] == 0 and rep_flush["ces_spool_flush"]["delivered_count"] == 1

    # Spool unit edge cases: empty snapshot/flush/clean, corrupted JSON, overflow FIFO eviction, dedup, quarantine on 400/422, unsafe serialization guard
    sp = selftest.GamerCesRetrySpool(tmp_path / "unit_spool.json", home=home_dir, max_entries=3, max_delivered=4)
    assert sp.max_entries == 3 and sp.max_delivered == 4 and sp.snapshot()["spool_exists"] is False
    assert sp.mark_delivered("0" * 16) is False and sp.mark_delivered("bad") is False
    assert sp.clean()["removed"] is False
    assert sp.flush("http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["spool_exists"] is False

    sp.path.parent.mkdir(parents=True, exist_ok=True)
    sp.path.write_text("[]", encoding="utf-8")
    assert sp.snapshot()["pending_count"] == 0
    sp.path.write_text("{broken", encoding="utf-8")
    assert sp.snapshot()["quarantined_count"] == 1
    assert sp.enqueue(ease_rating=5, idempotency_key="ces-gamer-u1", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["enqueued"] is True
    assert sp.enqueue(ease_rating=5, idempotency_key="ces-gamer-u1", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_pending"
    assert sp.flush("http://127.0.0.1:8000", http_fn=_fleet_statuses_http, environ=CLEAN_ENV, cwd=clean_ws)["delivered_count"] == 1
    assert sp.enqueue(ease_rating=5, idempotency_key="ces-gamer-u1", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_delivered"

    # Inject corrupted entries on disk + overflow > max_entries + unsafe serialization check
    sp.path.write_text(
        json.dumps(
            {
                "delivered_event_ids": ["bad", "000001", "000002", "000003", "000004", "ces-gamer-u1"],
                "pending": [
                    "not-dict",
                    {"event_id": "ces-gamer-u1", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "bad-p1", "payload": {"profile": "invalid_role"}},
                    {"event_id": "ces-gamer-p1", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "ces-gamer-p2", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "ces-gamer-p3", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "ces-gamer-p4", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    snap_ov = sp.snapshot()
    assert snap_ov["pending_count"] == 3 and snap_ov["dropped_overflow_count"] >= 1
    assert sp.mark_delivered("ces-gamer-p3") is True
    with pytest.raises(selftest.SelftestError):
        sp._atomic_save_unlocked({"pending": [{"event_id": "unsafe-01", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "secret@corp.example"}}]})

    for i in range(4):
        enq = sp.enqueue(ease_rating=5, idempotency_key=f"ces-gamer-fifo-{i}", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)
        if i == 3:
            assert enq["evicted_oldest"] is True
    for t_code in (404, 408, 429, 503, 0):
        ret = selftest.flush_gamer_ces_queue(
            "http://127.0.0.1:8000",
            spool_path=sp.path,
            http_fn=(lambda *_a: (_ for _ in ()).throw(TimeoutError("tmo"))) if t_code == 0 else (lambda *_a, c=t_code: (c, {})),
            lang="fr",
            home=home_dir,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert ret["flush_ok"] is False and ret["retained_retry_count"] == 3
    quar_res = sp.flush("http://127.0.0.1:8000", http_fn=lambda *_a: (400, {"error": "bad"}), environ=CLEAN_ENV, cwd=clean_ws)
    assert quar_res["quarantined_count"] == 3 and quar_res["pending_count"] == 0

    # CLI --ces-queue-file / --flush-ces-queue / --clean-ces-queue (--json, --markdown, CLI, compliance fail)
    cli_q = tmp_path / "cli_ces_queue.json"
    rc_sub_q = selftest.main(
        ["--gateway", "http://127.0.0.1:8000", "--submit-feedback", "--ease-rating", "5", "--non-blocking-feedback", "--ces-queue-file", str(cli_q), "--cwd", str(clean_ws), "--json"],
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        http_fn=lambda *_a: (503, {}),
        environ=CLEAN_ENV,
        home=home_dir,
    )
    assert rc_sub_q == 0 and cli_q.is_file()
    _ = capsys.readouterr()

    assert selftest.main(
        ["--flush-ces-queue", "--gateway", "http://127.0.0.1:8000", "--ces-queue-file", str(cli_q), "--cwd", str(clean_ws), "--markdown"],
        http_fn=_fleet_statuses_http,
        environ=CLEAN_ENV,
        home=home_dir,
    ) == 0
    assert "CES" in capsys.readouterr().out

    assert selftest.main(
        ["--clean-ces-queue", "--ces-queue-file", str(cli_q), "--cwd", str(clean_ws)],
        environ=CLEAN_ENV,
        home=home_dir,
    ) == 0
    assert not cli_q.exists() and "Removed" in capsys.readouterr().out

    # Compliance failure refuses flush and clean without touching disk/net
    refused_q = tmp_path / "refused_q.json"
    assert selftest.main(["--clean-ces-queue", "--ces-queue-file", str(refused_q), "--output", str(tmp_path / "no_write.json"), "--json"], environ={"AMADEUS_X": "1"}, cwd=clean_ws, home=home_dir) == 2
    assert not (tmp_path / "no_write.json").exists()
    _ = capsys.readouterr()
    assert selftest.flush_gamer_ces_queue("http://127.0.0.1:8000", spool_path=refused_q, environ={"AMADEUS_X": "1"}, cwd=clean_ws, home=home_dir)["compliance_ok"] is False
    assert selftest.enqueue_gamer_ces_event("http://127.0.0.1:8000", spool_path=refused_q, environ={"AMADEUS_X": "1"}, cwd=clean_ws, home=home_dir)["compliance_ok"] is False

    for bad_q_arg in (
        lambda: selftest.GamerCesRetrySpool(tmp_path / "b1.json", max_entries=0),
        lambda: selftest.GamerCesRetrySpool(tmp_path / "b2.json", max_delivered=0),
        lambda: selftest._validate_spooled_ces_payload("bad"),
        lambda: selftest._validate_spooled_ces_payload({"profile": "gamer", "extra": 1}),
        lambda: selftest._normalize_ces_spool_payload(profile="admin"),
        lambda: selftest._normalize_ces_spool_payload(ease_rating=9),
        lambda: selftest._normalize_ces_spool_payload(zero_game_lag="no"),  # type: ignore[arg-type]
        lambda: selftest._normalize_ces_event_id(gateway="http://127.0.0.1:8000", ease_rating=5, zero_game_lag=True, platform_tag="linux", profile="gamer", idempotency_key=True),  # type: ignore[arg-type]
        lambda: selftest._normalize_ces_event_id(gateway="http://127.0.0.1:8000", ease_rating=5, zero_game_lag=True, platform_tag="linux", profile="gamer", idempotency_key="   "),
    ):
        with pytest.raises(selftest.SelftestError):
            bad_q_arg()


def test_04_concurrent_48_participant_gamer_whisper_wal_soak_and_pr949_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_upstream(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Forbidden upstream inference call: {request.url}")

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00050-dyn")

    db_path = tmp_path / "gamer_wal_soak48.sqlite3"
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
    clean_git_remote = _untrusted_url("github.com") + "/xaviercallens/HiveMind"
    (clean_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {clean_git_remote}\n', encoding="utf-8")

    tls = threading.local()
    tls_clients: list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = []
    tls_lock = threading.Lock()

    def _get_thread_client() -> tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]:
        pair = getattr(tls, "asgi_pair", None)
        if pair is None:
            loop = asyncio.new_event_loop()
            client = loop.run_until_complete(
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://127.0.0.1:8000",
                ).__aenter__()
            )
            pair = (loop, client)
            tls.asgi_pair = pair
            with tls_lock:
                tls_clients.append(pair)
        return pair

    def _asgi_req(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        loop, client = _get_thread_client()
        return loop.run_until_complete(client.request(method, path, json=json_body, headers=headers))

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
        stripped = selftest.strip_masked_keys_and_nonces(resp.text)
        assert static_numbers_guard.FIGURE.findall(stripped) == []
        assert ("inner" + "HTML") not in stripped
        return hdr_n[0]

    def _make_asgi_raw(peer_ip: str, *, recorded_ces: list[tuple[str, dict[str, Any]]] | None = None) -> selftest.RawHttpFn:
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
            if recorded_ces is not None and method == "POST" and parsed.path == "/v1/hivemind/onboarding/feedback" and payload is not None:
                idem = req_h.get("Idempotency-Key", "")
                if idem:
                    recorded_ces.append((idem, dict(payload)))
            resp = _asgi_req(method, path, json_body=payload, headers=req_h)
            with shared_lock:
                seen_paths.append(parsed.path)
                if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
                    sqlite_locks.append(resp.text[:120])
            if parsed.path in selftest.PR949_HTML_ROUTES and resp.status_code == 200:
                nonce_val = _check_html(parsed.path, resp)
                with shared_lock:
                    observed_nonces.append(nonce_val)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            if "application/json" in hdrs.get("content-type", ""):
                return resp.status_code, resp.json(), hdrs
            return resp.status_code, resp.text, hdrs

        return _adapter

    try:
        # Seed 8 distinct volunteer GPU producers
        for p_idx in range(8):
            p_ip = _loopback_ip(55, p_idx)
            r_reg = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"gw50-gpu-{p_idx:02d}", "role": "producer"},
                headers={"X-Forwarded-For": p_ip},
            )
            assert r_reg.status_code == 200, r_reg.text
            p_key = str(r_reg.json()["api_key"])
            r_hb = _asgi_req(
                "POST",
                "/v1/hivemind/producer/heartbeat",
                json_body={
                    "endpoint_url": f"http://{p_ip}:11434",
                    "models": [{"id": "hivemind/" + "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 50.0,
                    "gpu_vram_mib": 12288,
                    "execution_class": "ollama-gamer-whisper",
                },
                headers={"Authorization": f"Bearer {p_key}", "X-Forwarded-For": p_ip},
            )
            assert r_hb.status_code == 200, r_hb.text

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
            home_d = slot / "home" / f"gamer_user_{idx:02d}"
            cred_d = home_d / ".hivemind" / "credentials.json"
            ws_d = slot / "hobby"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)

            # Inspect all 4 PR #949 HTML routes for CSP nonces + zero static figures
            arena_html = ""
            for html_path in selftest.PR949_HTML_ROUTES:
                h_resp = _asgi_req("GET", html_path, headers=hdrs)
                nonce_v = _check_html(html_path, h_resp)
                with shared_lock:
                    observed_nonces.append(nonce_v)
                    seen_paths.append(html_path)
                if html_path == "/arena-noel":
                    arena_html = h_resp.text

            reg_resp = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"gw50_{profile}_{c_idx:02d}", "role": "prosumer" if is_dev else "producer"},
                headers=hdrs,
            )
            assert reg_resp.status_code == 200, reg_resp.text
            api_key = str(reg_resp.json()["api_key"])
            masked_key = selftest.mask_api_key(api_key)
            selftest._write_file_secure_atomic(
                cred_d,
                json.dumps({"api_key": api_key, "role": "prosumer" if is_dev else "producer", "gateway": "http://127.0.0.1:8000"}),
                mode=0o600,
                dir_mode=0o700,
            )

            recorded_ces: list[tuple[str, dict[str, Any]]] = []
            raw_adapter = _make_asgi_raw(peer_ip, recorded_ces=recorded_ces)

            def _json_adapter(
                m: str,
                u: str,
                p: dict[str, Any] | None = None,
                h: dict[str, str] | None = None,
            ) -> tuple[int, dict[str, Any]]:
                sc, bd, _ = raw_adapter(m, u, p, h)
                return sc, (bd if isinstance(bd, dict) else {"raw_length": len(str(bd))})

            ease = 5 if (c_idx % 4 != 0) else 4
            if is_dev:
                # Dev cohort posts profile='dev' + Idempotency-Key with 0 GPU claims
                selftest.verify_live_fleet_and_badge(gateway="http://127.0.0.1:8000", http_fn=_json_adapter, environ=CLEAN_ENV, cwd=ws_d)
                dev_idem = f"ces-dev-{c_idx:02d}-{hashlib.sha256(peer_ip.encode()).hexdigest()[:16]}"
                dev_payload = {"profile": "dev", "ease_rating": ease, "zero_game_lag": True, "platform": "linux"}
                r1 = _asgi_req("POST", "/v1/hivemind/onboarding/feedback", json_body=dev_payload, headers={**hdrs, "Idempotency-Key": dev_idem})
                assert r1.status_code in (200, 201)
                recorded_ces.append((dev_idem, dev_payload))
                combined = json.dumps({"ok": True, "masked": masked_key})
            else:
                rep = selftest.run_gamer_whisper_selftest(
                    gateway="http://127.0.0.1:8000",
                    noel_verify=True,
                    send_opentrainer_heartbeat=True,
                    api_key=api_key,
                    host_label=f"pc-gamer-{c_idx:02d}",
                    submit_feedback=True,
                    ease_rating=ease,
                    gpu_probe_fn=_quiet_rtx4070_probe,
                    process_probe_fn=_idle_processes,
                    worker_lifecycle_fn=_fast_worker_drill,
                    http_fn=_json_adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_d,
                    home=home_d,
                    lang="fr" if (idx % 2 == 0) else "en",
                )
                _verify_sha256_receipt(rep)
                assert rep["exit_code"] == 0 and rep["ces_recorded"] is True and rep["zero_game_lag"] is True
                combined = json.dumps(rep, ensure_ascii=False)

            assert len(recorded_ces) == 1
            idem_k, ces_pl = recorded_ces[0]

            # Duplicate-replayed Idempotency-Key MUST NOT double count under WAL load
            dup_resp = _asgi_req(
                "POST",
                "/v1/hivemind/onboarding/feedback",
                json_body=ces_pl,
                headers={**hdrs, "Idempotency-Key": idem_k},
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

            assert api_key not in combined and f"gamer_user_{idx:02d}" not in combined
            assert selftest.scan_output_safety(combined, home=home_d) == []

            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            return selftest.verify_pr949_gamer_quorum_and_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                require_live_quorum=True,
                check_arena_summary=True,
                verify_idempotency_replay=False,
                raw_http_fn=_make_asgi_raw(_loopback_ip(33, p_idx)),
                environ=CLEAN_ENV,
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

        post_soak = selftest.verify_pr949_gamer_quorum_and_parity(
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
            raw_http_fn=_make_asgi_raw("127.0.50.3"),
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
        assert selftest.scan_output_safety(serialized_post, home=tmp_path) == []
        fleet_txt = _asgi_req("GET", "/v1/hivemind/producers/fleet", headers={"X-Forwarded-For": "127.0.50.4"}).text
        for raw_k, masked_k in zip(raw_keys, masked_keys, strict=True):
            assert raw_k not in serialized_post and raw_k not in fleet_txt
            assert masked_k.startswith(("hm-***", "hm_live_***"))
    finally:
        monkeypatch.undo()
        for loop, client in tls_clients:
            loop.run_until_complete(client.__aexit__(None, None, None))
            loop.close()
        store.close()


def test_05_repository_security_stdlib_zero_stub_and_static_figure_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_script = SELFTEST_PATH.read_text(encoding="utf-8")
    src_test = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header within first 300 chars & stdlib-only imports + ProxyHandler({}) + zero forbidden literals
    assert EXPECTED_AGPL_HEADER in src_script[:300]
    assert EXPECTED_AGPL_HEADER in src_test[:300]
    tree_script = ast.parse(src_script, filename=str(SELFTEST_PATH))
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    for node in ast.walk(tree_script):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "ProxyHandler"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Dict)
                and len(node.args[0].keys) == 0
            ):
                saw_empty_proxy_handler = True
    assert (imported_roots - sys.stdlib_module_names) == set()
    assert saw_empty_proxy_handler is True

    test_tree = ast.parse(src_test, filename=str(Path(__file__)))
    test_imports = {
        a.name.split(".", 1)[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.Import)
        for a in n.names
    } | {
        n.module.split(".", 1)[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "coverage" not in test_imports

    for forbidden_lit in (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment.write",
        "/v1/" + "chat/completions",
    ):
        assert forbidden_lit not in src_script
        assert forbidden_lit not in src_test

    # 2. Lateral-movement AST audit (+ negative control), PRSecurityScanner, ZeroStubAudit, static figures & gatekeeper
    assert lateral_guard.audit_source_for_lateral_movement(src_script, filename=str(SELFTEST_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(src_test, filename=str(Path(__file__))) == []
    assert lateral_guard.audit_source_for_lateral_movement(
        f'URL = "{_untrusted_url("evil.example")}"\n',
        filename="neg_control.py",
    )

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, fpath, text in (
        ("scripts/noel_gamer_whisper_selftest.py", SELFTEST_PATH, src_script),
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
        pr_title="feat(onboarding,gamer,scale): zero-inference PR #949 live quorum/arena parity and 48-user WAL soak",
        pr_body="Implements SLICE-SCALE-DYN-50 Gamer-Whisper PR #949 parity/quorum gate, 0600 spool, and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,gamer,scale): zero-inference PR #949 live quorum parity"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_res.passed is True and pr_res.findings == []

    # Verify fallback ZeroStubAudit works without optional gwaya package
    orig_gwaya = sys.modules.get("gwaya")
    orig_audit = sys.modules.get("gwaya.ast_audit")
    monkeypatch.setitem(sys.modules, "gwaya", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "gwaya.ast_audit", None)  # type: ignore[arg-type]
    try:
        reloaded_spec = importlib.util.spec_from_file_location("test_dyn50_ci_fallback", __file__)
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
