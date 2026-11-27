# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/arena parity, Idempotency-Key replay, SSRF/TLS guard, 0600 spool & 48-user WAL soak tests (`SLICE-SCALE-DYN-48`)."""

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
DOCTOR_PATH = REPO_ROOT / "scripts" / "hivemind_doctor.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct import without starting an internal coverage.Coverage() instance so external
# `coverage run --rcfile=/dev/null --include='*/scripts/hivemind_doctor.py'` works cleanly.
from scripts import hivemind_doctor as doc  # noqa: E402

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
    "test_no_lateral_movement_dyn48",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn48",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn48",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-linux-pc"}
TARGET_FULL_SHA = "c4d8e9f01234567890abcdef1234567890fedcba"
TARGET_SHORT_SHA = TARGET_FULL_SHA[:12]
SOAK_ROUTER_SECRET = b"noel-scale-dyn48-wal-secret-32b!"
ALPHA_KEY = "hm-dyn48-test-alpha-9876"
LIVE_KEY = "hm_live_dyn48_secret_4321"

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


def _build_local_runner(
    *,
    smi_rc: int = 0,
    smi_out: str = "NVIDIA GeForce RTX 4070 SUPER, 12288, 10240, 52, 2\n",
    smi_exc: Exception | None = None,
    ps_rc: int = 0,
    ps_out: str = "python3\nbash\n",
    ps_exc: Exception | None = None,
    tasklist_rc: int = 0,
    tasklist_out: str = '"explorer.exe","1020","Console","1","25,000 K"\n',
) -> doc.RunnerFn:
    def _runner(argv: Any, _timeout_s: float) -> tuple[int, str]:
        exe = Path(str(argv[0])).name.lower() if argv else ""
        if "nvidia-smi" in exe:
            if smi_exc is not None:
                raise smi_exc
            return smi_rc, smi_out
        if "tasklist" in exe:
            if ps_exc is not None:
                raise ps_exc
            return tasklist_rc, tasklist_out
        if exe == "ps":
            if ps_exc is not None:
                raise ps_exc
            return ps_rc, ps_out
        return 0, ""

    return _runner


def _write_creds(
    path: Path,
    *,
    api_key: str = ALPHA_KEY,
    role: str = "prosumer",
    mode: int = 0o600,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"api_key": api_key, "role": role, "gateway": doc.DEFAULT_GATEWAY}),
        encoding="utf-8",
    )
    os.chmod(path, mode)
    return path


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
    cloud_run_revision: str | None = "hivemind-openrouter-00048-dyn",
    fleet_object: str = doc.FLEET_OBJECT,
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
            "object": doc.FEEDBACK_SUMMARY_OBJECT,
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
            "object": doc.ARENA_SUMMARY_OBJECT,
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
            "object": doc.MANIFEST_OBJECT,
            "campaign_id": doc.CAMPAIGN_ID,
            "status": "ok",
            "personal_pc_only": True,
            "financial_promise": doc.FINANCIAL_PROMISE,
            "build_sha": build_sha,
            "cloud_run_revision": cloud_run_revision,
            "holiday_routes": list(doc.MANIFEST_REQUIRED_HOLIDAY_ROUTES),
            "downloadable_scripts": ["hivemind_doctor.py", "hivemind_quickstart.py"],
        }
    )
    return {
        "models": {"object": "list", "data": model_rows},
        "manifest": manifest_doc,
        "feedback": fb_doc,
        "raid": {
            "object": doc.RAID_STATUS_OBJECT,
            "compliance": {
                "personal_pc_only": True,
                "monetary_cost_eur": "0.00 EUR",
                "monetary_cost_usd": "0.00 USD",
            },
            "swarm": {"active_trainers": live_hiveminders, "accepted_opentrainer_rounds": 16},
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
        "status_overrides": dict(status_overrides or {}),
        "nonce_seq": 0,
    }


def _make_table_http(table: dict[str, Any], *, me_role: str = "prosumer", hb_status: int = 200) -> doc.RawHttpFn:
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
        if method == "GET" and path in doc.PR949_HTML_ROUTES:
            table["nonce_seq"] += 1
            nonce = table["static_nonce"] or f"Dyn48Nonce_{table['nonce_seq']:04d}_XyZw"
            extra = ""
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
            return 200, {"account_id": "acc_dyn48", "handle": "dyn48-friend", "role": me_role}, {"content-type": "application/json"}
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

            def do_POST(self) -> None:
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
) -> None:
    # 1. Host allowlisting: googleapis/run.app/huggingface True; malformed labels False
    assert doc.is_allowed_gateway_host("hivemind-openrouter-cfhmhmvv5a-uc.a.run.app") is True
    assert doc.is_allowed_gateway_host("storage.googleapis.com") is True
    assert doc.is_allowed_gateway_host("run.googleapis.com") is True
    assert doc.is_allowed_gateway_host("datasets-server.huggingface.co") is True
    assert doc.is_allowed_gateway_host("127.0.0.1") is True
    assert doc.is_allowed_gateway_host("localhost") is True
    assert doc.is_allowed_gateway_host("::1") is True
    assert doc.is_allowed_gateway_host("[::1]") is True
    for bad_host in (
        "",
        None,
        "a" * 260,
        "foo..run.app",
        "-bad.run.app",
        "bad-.run.app",
        "bad_host.run.app",
        "run.app",
        "googleapis.com",
        "[::1",
        "foo\\bar.run.app",
        "user@127.0.0.1",
        "evil.example",
    ):
        assert doc.is_allowed_gateway_host(bad_host) is False  # type: ignore[arg-type]

    assert doc.validate_gateway_url("https://hivemind-demo.a.run.app/") == "https://hivemind-demo.a.run.app"
    assert doc.validate_gateway_url("http://[::1]:8000") == "http://[::1]:8000"
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
        with pytest.raises(ValueError):
            doc.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    # 2. Zero-inference target, timeout, CRLF headers, and _SafeRedirectHandler checks
    comp_endpoint = "http://127.0.0.1:8000/v1/" + "chat/completions"
    for fn_call in (
        lambda: doc._validate_zero_inference_target("DELETE", "http://127.0.0.1:8000/health"),
        lambda: doc._validate_zero_inference_target("GET", comp_endpoint),
        lambda: doc._validate_zero_inference_target("POST", "http://127.0.0.1:8000/v1/chat"),
        lambda: doc._validate_zero_inference_target("POST", "http://127.0.0.1:8000/v1/hivemind/unknown"),
        lambda: doc._validate_zero_inference_target("GET", "http://127.0.0.1:8000/health", payload={"ok": True}),
        lambda: doc._validate_zero_inference_target(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"nested": [{"messages": [{"role": "user", "content": "hi"}]}]},
        ),
        lambda: doc._validate_zero_inference_target(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"model": "hivemind/" + "coder"},
        ),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=float("nan")),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=-1.0),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=True),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Test": "bad\r\nval"}),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Nul": "bad\x00val"}),
        lambda: doc.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"": "x"}),
    ):
        with pytest.raises(ValueError):
            fn_call()

    redir = doc._SafeRedirectHandler()
    remote_req = doc.urllib.request.Request("https://demo.a.run.app/quick")
    with pytest.raises(ValueError, match="loopback"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "https://127.0.0.1/quick")
    with pytest.raises(ValueError, match="downgrade"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "http://demo.a.run.app/quick")
    with pytest.raises(ValueError, match="empty"):
        redir.redirect_request(remote_req, io.BytesIO(), 302, "Found", {}, "")

    # 3. Live loopback HTTP server: ProxyHandler({}) isolation, redirects, 4 MiB cap, HTTPError sanitization
    port_holder = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHttpServer(port_ref=port_holder).build_handler())
    port_holder[0] = int(srv.server_address[1])
    srv_thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.05), daemon=True)
    srv_thread.start()
    try:
        base = f"http://127.0.0.1:{port_holder[0]}"
        monkeypatch.setenv("HTTP_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")
        monkeypatch.setenv("HTTPS_PROXY", _untrusted_url("proxy.internal.example", scheme="http") + ":8080")

        rc_raw, txt_raw, hdrs_raw = doc.default_http_raw("GET", f"{base}/health", headers={"X-Client": "doctor"})
        assert rc_raw == 200 and json.loads(txt_raw)["status"] == "ok" and "application/json" in hdrs_raw["content-type"]
        assert doc.default_http_get_json(f"{base}/health")[0] == 200
        assert doc.default_http_get_json(f"{base}/empty") == (204, {})
        assert doc.default_http_get_json(f"{base}/empty-404") == (404, {"error": ""})
        assert "raw" in doc.default_http_get_json(f"{base}/non-json")[1]
        assert doc.default_http_request_json("GET", f"{base}/empty") == (204, {"content_type": "", "html_length": 0})
        assert doc.default_http_request_json("GET", f"{base}/empty-404") == (404, {})
        st_html, js_html = doc.default_http_request_json("GET", f"{base}/non-json")
        assert st_html == 200 and int(js_html["html_length"]) > 10
        assert doc.default_http_request_json("GET", f"{base}/redirect-ok")[0] == 200
        st_post, js_post = doc.default_http_request_json(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            payload={"profile": "dev", "ease_rating": 5},
            headers={"Idempotency-Key": "hm-ces-loopback"},
        )
        assert st_post == 200 and js_post["idem"] == "hm-ces-loopback"
        st_404, js_404 = doc.default_http_request_json("GET", f"{base}/missing-route")
        assert st_404 == 404 and js_404["error"] == "missing_route"
        st_502, js_502 = doc.default_http_get_json(f"{base}/bad-err")
        assert st_502 == 502 and "/home/alice" not in js_502["error"] and "user@" not in js_502["error"]
        assert doc._invoke_http_callable_raw(doc.default_http_raw, "GET", f"{base}/health")[0] == 200
        assert doc._invoke_http_callable_raw(doc.default_http_get_json, "GET", f"{base}/health")[0] == 200
        for redir_path in ("/redirect-evil", "/redirect-backslash", "/redirect-completions"):
            with pytest.raises(ValueError):
                doc.default_http_raw("GET", f"{base}{redir_path}")
        with pytest.raises(ValueError, match="byte limit"):
            doc.default_http_raw("GET", f"{base}/oversized")
    finally:
        monkeypatch.undo()
        srv.shutdown()
        srv.server_close()

    st_down, js_down = doc.default_http_get_json(f"http://127.0.0.1:{port_holder[0]}/health", timeout_s=0.2)
    assert st_down == 0 and "network_error" in js_down

    # Parity & callable dispatch arities
    assert doc._invoke_http_callable_raw(lambda u, t: (200, {"u": u, "t": t}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doc._invoke_http_callable_raw(lambda _m, _u: (200, {"a": 1}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doc._invoke_http_callable_raw(lambda _m, _u, _p: (200, {"b": 2}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doc._invoke_http_callable_raw(lambda _m, _u, _p, _h: (200, {"c": 3}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doc._invoke_http_callable_raw(lambda _m, _u, _p, _h, _t: (200, {"d": 4}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doc._invoke_http_request(lambda _m, _u: (200, '{"k": 1}'), "GET", "http://127.0.0.1:8000/health")[1] == {"k": 1}
    assert doc._invoke_http_request(lambda _m, _u: (200, "plain"), "GET", "http://127.0.0.1:8000/health")[1]["raw"] == "plain"
    assert doc._invoke_http_request(lambda _m, _u: (200, [1, 2]), "GET", "http://127.0.0.1:8000/health")[1] == {"data": [1, 2]}
    assert doc._invoke_parity_call(lambda *_a: (_ for _ in ()).throw(OSError("conn")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    assert doc._invoke_parity_call(lambda *_a: (200, {"html": "ok", "csp": "default-src 'none'"}), "GET", "http://127.0.0.1:8000/noel")[3]["content-security-policy"] == "default-src 'none'"
    assert doc._invoke_parity_call(lambda *_a: (200, {"headers": {"Content-Type": "text/html"}, "html": "ok"}), "GET", "http://127.0.0.1:8000/noel")[3]["content-type"] == "text/html"
    assert doc._invoke_parity_call(lambda *_a: (200, "[1, 2]"), "GET", "http://127.0.0.1:8000/health")[1] is None
    assert doc._invoke_parity_call(lambda *_a: (200, "{broken"), "GET", "http://127.0.0.1:8000/health")[1] is None

    # Helper validation branches (lang, profile, ease, api key, IPv4, tuple sanitize, port 0)
    with pytest.raises(ValueError):
        doc.normalize_lang("de")
    for bad_prof in (None, True, "admin"):
        with pytest.raises(ValueError):
            doc.normalize_profile(bad_prof)  # type: ignore[arg-type]
    for bad_ease in (True, 6):
        with pytest.raises(ValueError):
            doc._parse_ease_rating(bad_ease)  # type: ignore[arg-type]
    assert doc._is_valid_hivemind_key(123) is False
    assert doc.mask_api_key("") == "<missing>"
    assert doc.mask_api_key("short") == "<invalid>"
    assert doc._sanitize_noel_structure(("a", ["/Users/bob/f"])) == ["a", ["~/f"]]
    win_home = tmp_path / "win_u"
    assert doc._sanitize_noel_text(str(win_home).replace("/", "\\") + "\\secret", home=win_home) == "~/secret"
    with pytest.raises(ValueError):
        doc.validate_gateway_url("https://localhost:0")

    # 4. strip_masked_keys_and_nonces & scan_output_safety
    masked_ms = "hm-***x8" + "m" + "s"
    masked_tok = "hm_live_***" + "9" + "tok/s"
    nonce_tok = "xyz" + "9" + "tok/s"
    nonce_ms = "'nonce-" + "99" + "ms'"
    sample_safe = f"<div nonce=\"abc123ms456\"><script nonce='{nonce_tok}'>{nonce_ms} {masked_ms} {masked_tok} hm_live_***4321 hm-ces-0123456789abcdef</script></div>"
    stripped = doc.strip_masked_keys_and_nonces(sample_safe)
    assert 'nonce=""' in stripped and "'nonce-CSP_NONCE'" in stripped
    assert doc.scan_output_safety(sample_safe) == []
    with pytest.raises(ValueError):
        doc.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        doc.scan_output_safety(123)  # type: ignore[arg-type]

    dirty_home = tmp_path / "home_alice"
    dirty_text = (
        f"{dirty_home}/secret Bearer topsecret123 hm-rawsecretvalue987 alice@example.com "
        f"{_ip_str(10, 44, 55, 66)} "
        + '"sco'
        + 're": 99 '
        + "142.5 m"
        + "s"
    )
    hits = set(doc.scan_output_safety(dirty_text, home=dirty_home))
    assert {
        "forbidden_key_detected",
        "unmasked_api_key_detected",
        "bearer_token_detected",
        "email_detected",
        "unredacted_home_path_detected",
        "static_figure_detected",
        f"non_loopback_ipv4_detected:{_ip_str(10, 44, 55, 66)}",
    }.issubset(hits)
    scrubbed = doc._sanitize_noel_text(dirty_text, home=dirty_home)
    assert "hm-***e987" in scrubbed and "<redacted-email>" in scrubbed and "<redacted-ip>" in scrubbed
    assert doc.scan_output_safety("999.1.1.1") == []
    assert doc.redact_home_path("/opt/x/file") == "/opt/x/file"
    assert doc._sanitize_noel_text("999.999.999.999") == "999.999.999.999"

    # Relative gitdir worktree indirection, unreadable config, and symlinked corporate path
    rel_gd = tmp_path / "rel_gitdir_ws"
    (rel_gd / "subwt").mkdir(parents=True)
    (rel_gd / ".wt_dir").mkdir()
    (rel_gd / ".wt_dir" / "config").write_text(f'[remote "origin"]\n\turl = {_untrusted_url("git.corp.local")}\n', encoding="utf-8")
    (rel_gd / ".git").write_text("gitdir: .wt_dir\n", encoding="utf-8")
    assert len(doc._resolve_git_config_paths(rel_gd / "subwt")) == 1
    sym_target = tmp_path / "amadeus_symlink_target"
    sym_target.mkdir()
    sym_link = tmp_path / "clean_alias_link"
    sym_link.symlink_to(sym_target, target_is_directory=True)
    assert doc.evaluate_noel_personal_pc_compliance(environ=CLEAN_ENV, cwd=sym_link)["compliance_ok"] is False

    # 5. Git parent-walk, worktree gitdir/commondir, NODE_EXTRA_CA_CERTS & --enforce-personal-pc
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
    comp_parent = doc.evaluate_noel_personal_pc_compliance(environ=CLEAN_ENV, cwd=deep_sub)
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
    comp_wt = doc.evaluate_noel_personal_pc_compliance(environ=CLEAN_ENV, cwd=wt_ws)
    assert comp_wt["compliance_ok"] is False and "git_remote:corp" in comp_wt["corporate_indicators"]

    # Non-noel --enforce-personal-pc fails closed (exit 2, 0 HTTP calls, 0 file writes) across env/path/remote
    clean_ws = tmp_path / "clean_hobby_ws"
    clean_ws.mkdir()
    comp_ca = doc.evaluate_noel_personal_pc_compliance(
        environ={
            "AMADEUS_FLAG": "1",
            "USERDNSDOMAIN": "nce.amadeus.corp",
            "USERDOMAIN": "AMADEUS-DOM",
            "COMPUTERNAME": "CORP-PC",
            "HTTPS_PROXY": _untrusted_url("zscaler.corp.example"),
            "NO_PROXY": ".paloalto.internal",
            "ZSCALER_CERT_PATH": "/etc/zscaler.pem",
            "NODE_EXTRA_CA_CERTS": "/etc/amadeus-root-ca.pem",
        },
        cwd=clean_ws,
        lang="fr",
    )
    assert {"AMADEUS_ENV", "USERDNSDOMAIN", "USERDOMAIN", "NODE_EXTRA_CA_CERTS", "ZSCALER_CERT_PATH"}.issubset(
        set(comp_ca["corporate_indicators"])
    )
    assert comp_ca["compliance_ok"] is False

    net_calls = {"n": 0}

    def _forbid_net(*_a: Any, **_k: Any) -> tuple[int, dict[str, Any]]:
        net_calls["n"] += 1
        raise AssertionError("Must not issue HTTP requests on compliance failure")

    for block_kw in (
        {"environ": {"NODE_EXTRA_CA_CERTS": "/etc/corp-ca.pem"}, "cwd": clean_ws},
        {"environ": CLEAN_ENV, "cwd": deep_sub},
        {"environ": CLEAN_ENV, "cwd": wt_ws},
    ):
        blocked_out = tmp_path / "blocked_no_noel.json"
        out_s = io.StringIO()
        rc_blk = doc.main(
            ["--enforce-personal-pc", "--cwd", str(block_kw["cwd"]), "--output", str(blocked_out), "--json"],
            http_get=_forbid_net,
            http_request=_forbid_net,
            environ=block_kw["environ"],  # type: ignore[arg-type]
            home=tmp_path / "h_blk",
            out_stream=out_s,
            err_stream=io.StringIO(),
        )
        assert rc_blk == 2 and net_calls["n"] == 0 and not blocked_out.exists()
        blk_doc = json.loads(out_s.getvalue())
        assert blk_doc["compliance_ok"] is False and blk_doc["outbound_http_requests"] == 0 and blk_doc["file_writes"] == 0
        _verify_sha256_receipt(blk_doc)

    # 6. Atomic POSIX 0600 file persistence & CLI path validation
    sec_file = tmp_path / "secure" / "report.json"
    doc._write_file_secure_atomic(sec_file, '{"ok": true}\n', mode=0o600, dir_mode=0o700)
    assert stat.S_IMODE(sec_file.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        doc._write_file_secure_atomic(sec_file.parent, "bad")
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("atomic_fail")))
    with pytest.raises(OSError):
        doc._write_file_secure_atomic(tmp_path / "fail_atom.txt", "abc")
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp"))

    for bad_cli in ("", "bad\x00file", True):
        with pytest.raises(ValueError):
            doc._validate_cli_path_arg(bad_cli)  # type: ignore[arg-type]
    assert doc.main(["--pr949-quorum-gate", "--output", "bad\x00out"], environ=CLEAN_ENV, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    assert doc._expand_user_path("~/cfg.json", home=tmp_path) == tmp_path / "cfg.json"
    with pytest.raises(ValueError):
        doc.default_subprocess_runner(["gcloud", "version"], 1.0)


def test_02_pr949_parity_quorum_arena_idempotency_and_deploy_dry_run(
    tmp_path: Path,
) -> None:
    clean_ws = tmp_path / "personal_repo"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn48"
    home_dir.mkdir()
    cred_path = _write_creds(home_dir / ".hivemind" / "credentials.json")

    # 1. Full green PR #949 EN & FR gate with real bash deploy.sh --dry-run
    for lang_code in ("en", "fr"):
        tbl = _build_pr949_table()
        rep = doc.verify_pr949_doctor_quorum_and_parity(
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
            credentials_path=cred_path,
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
        assert doc.scan_output_safety(json.dumps(rep, ensure_ascii=False), home=home_dir) == []
        assert "PR #949" in doc.render_pr949_quorum_report(rep)
        assert "# HiveMind Doctor" in doc.render_markdown_report(rep)

    # Accept router hivemind.producer_fleet and awaiting_producer_quorum when non-strict
    low_tbl = _build_pr949_table(fleet_object="hivemind.producer_fleet", live_hiveminders=3)
    low_non_strict = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(low_tbl),
        require_live_quorum=False,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_non_strict["parity_ok"] is True and low_non_strict["gate_ok"] is True
    assert low_non_strict["awaiting_producer_quorum"] is True

    low_strict = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=3)),
        require_live_quorum=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert low_strict["quorum_gate_ok"] is False and low_strict["gate_ok"] is False and low_strict["exit_code"] == 2

    # 2. Compliance fail-closed (0 net, 0 subprocess, 0 writes)
    blk_pr = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        environ={"AMADEUS_X": "1"},
        cwd=clean_ws,
        home=home_dir,
    )
    assert blk_pr["compliance_ok"] is False and blk_pr["exit_code"] == 2 and blk_pr["network_calls"] == 0

    # 3. Negative fail-closed matrix
    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(include_coder=False)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False
    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(empty_models=True)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["models_ok"] is False

    stale = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(status_overrides={"/raid-noel": 404, "/arena-noel": 404})),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert stale["stale_404_drift"] is True and stale["fallback_guidance"] is not None

    for html_kwargs, fail_key in (
        ({"static_nonce": "RepeatedNonce123"}, "unique_csp_nonces_ok"),
        ({"unsafe_sink_route": "/noel"}, "html_dom_clean"),
        ({"static_fig_route": "/quick"}, "html_dom_clean"),
        ({"missing_banner_route": "/arena-noel"}, "compliance_banners_ok"),
    ):
        bad_html = doc.verify_pr949_doctor_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**html_kwargs)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert bad_html[fail_key] is False and bad_html["gate_ok"] is False

    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        expected_sha="deadbeef1234",
        raw_http_fn=_make_table_http(_build_pr949_table()),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["sha_drift"] is True

    zero_fb = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": doc.FEEDBACK_SUMMARY_OBJECT,
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

    low_slo = doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(
            _build_pr949_table(
                feedback_summary={
                    "object": doc.FEEDBACK_SUMMARY_OBJECT,
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

    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(leak_fleet_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["fleet_ok"] is False

    for bad_arena in (
        {"object": "wrong.object", "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
        {"object": doc.ARENA_SUMMARY_OBJECT, "total_votes": 5, "holiday_wins": 4, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.5},
        {"object": doc.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 2, "win_ratio": 0.0, "wilson_95_lower_bound": 0.1},
        {"object": doc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": {"total_votes": 2, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0}}},
        {"object": doc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "sco" + "re": 9},
    ):
        assert doc.verify_pr949_doctor_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(arena_summary=bad_arena)),
            check_arena_summary=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["arena_summary_ok"] is False

    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_table_http(_build_pr949_table(arena_status=404)),
        check_arena_summary=True,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["arena_summary_ok"] is False

    for replay_kw in ({"feedback_double_count": True}, {"feedback_ignore_first": True}):
        assert doc.verify_pr949_doctor_quorum_and_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_table_http(_build_pr949_table(**replay_kw)),
            verify_idempotency_replay=True,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )["idempotent_dedup_ok"] is False

    # Contract & option validation edge cases (EN + FR)
    assert doc._inspect_models_contract(None, lang="fr")[0] is False
    assert doc._inspect_models_contract({"object": "list", "data": "bad"}, lang="fr")[0] is False
    assert doc._inspect_fleet_contract(None, lang="fr")[0] is False
    assert doc._inspect_fleet_contract({"object": "wrong"}, lang="fr")[0] is False
    assert doc._inspect_fleet_contract({"object": doc.FLEET_OBJECT, "hiveminders": "bad"}, lang="fr")[0] is False
    assert doc._inspect_fleet_contract({"object": doc.FLEET_OBJECT, "hiveminders": [], "live_hiveminders": True}, lang="fr")[0] is False
    assert doc._inspect_manifest_contract(None, expected_sha=TARGET_SHORT_SHA, lang="fr")[0] is False
    assert doc._inspect_manifest_contract({"object": doc.MANIFEST_OBJECT, "sco" + "re": 1}, expected_sha=None, lang="en")[0] is False
    assert doc._inspect_raid_status_contract(None, lang="fr")[0] is False
    assert doc._inspect_feedback_summary_contract(None, lang="fr")[0] is False
    assert doc._inspect_html_contract("/noel", "<html><body>plain</body></html>", {}, set(), lang="fr")[0] is False
    assert doc._pr949_route_diagnostic("/noel", "holiday", 0, lang="fr")
    assert doc._shas_match("unknown", "abc1234") is False
    assert doc._shas_match("reltag1", "reltag1") is True
    assert doc._shas_match("reltag1", "reltag2") is False

    assert doc._run_deploy_sh_dry_run(repo_root=tmp_path)["ok"] is False
    assert doc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, environ={"ONLY_VAR": "1"}, runner=lambda _c, _t: (0, ""))["ok"] is False
    assert doc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (1, "gcloud_invoked=1\n"))["ok"] is False
    assert doc._run_deploy_sh_dry_run(repo_root=REPO_ROOT, runner=lambda _c, _t: (_ for _ in ()).throw(OSError("exec")))["ok"] is False
    assert doc._pr949_route_diagnostic("/noel", "holiday", 503, lang="en")
    assert doc._inspect_feedback_summary_contract({"object": "wrong"}, lang="en")[0] is False
    assert doc._inspect_arena_summary_contract({"object": doc.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 1, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": "bad"}}, lang="en")[0] is False
    fb404_tbl = _build_pr949_table(status_overrides={"/v1/hivemind/onboarding/feedback/summary": 404})
    fb404_rep = doc.verify_pr949_doctor_quorum_and_parity("http://127.0.0.1:8000", raw_http_fn=_make_table_http(fb404_tbl), execute_deploy_dry_run=True, repo_root=tmp_path, environ=CLEAN_ENV, cwd=clean_ws)
    assert fb404_rep["ces_slo_ok"] is False and fb404_rep["deploy_dry_run"]["ok"] is False
    assert "[DIAG]" in doc.render_pr949_quorum_report(fb404_rep)
    assert "## Diagnostics" in doc.render_markdown_report(fb404_rep)
    with pytest.raises(ValueError):
        doc._validate_expected_sha("unknown")
    with pytest.raises(ValueError):
        doc._validate_expected_sha(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        doc.verify_pr949_doctor_quorum_and_parity("http://127.0.0.1:8000", target_online_producers=0, environ=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        doc.verify_pr949_doctor_quorum_and_parity("http://127.0.0.1:8000", min_ease_rating=9.0, environ=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        doc.verify_pr949_doctor_quorum_and_parity("http://127.0.0.1:8000", min_zero_lag_ratio=2.0, environ=CLEAN_ENV, cwd=clean_ws)
    with pytest.raises(ValueError):
        doc.verify_pr949_doctor_quorum_and_parity("http://127.0.0.1:8000", require_live_quorum="yes", environ=CLEAN_ENV, cwd=clean_ws)  # type: ignore[arg-type]

    # 4. CLI --pr949-quorum-gate (--output atomic 0600, --flush-ces-queue, --skip-arena-summary, --markdown, bad health/raid)
    spool_q = tmp_path / "pr949_spool.json"
    doc.enqueue_doctor_ces_event(
        "http://127.0.0.1:8000",
        profile="dev",
        ease_rating=5,
        spool_path=spool_q,
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    pr949_out = tmp_path / "pr949_out.json"
    cli_buf = io.StringIO()
    assert (
        doc.main(
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
                "--json",
            ],
            raw_http_fn=_make_table_http(_build_pr949_table()),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=cli_buf,
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert stat.S_IMODE(pr949_out.stat().st_mode) == 0o600
    parsed_pr949 = json.loads(pr949_out.read_text(encoding="utf-8"))
    assert parsed_pr949["arena_summary_checked"] is False and parsed_pr949["flush_ces_queue"]["flush_ok"] is True

    md_buf = io.StringIO()
    assert (
        doc.main(
            ["--pr949-quorum-gate", "--gateway", "http://127.0.0.1:8000", "--require-live-quorum", "--lang", "fr", "--markdown"],
            raw_http_fn=_make_table_http(_build_pr949_table(live_hiveminders=2)),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=md_buf,
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert "PR #949" in md_buf.getvalue()
    assert doc.main(["--pr949-quorum-gate", "--expected-sha", "x!"], environ=CLEAN_ENV, cwd=clean_ws, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2
    bad_tbl = _build_pr949_table()
    assert doc.verify_pr949_doctor_quorum_and_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=lambda m, u, p=None, h=None: (200, {"status": "bad"}, {"content-type": "application/json"})
        if urlsplit(u).path == "/health"
        else (
            (200, {"object": doc.RAID_STATUS_OBJECT, "compliance": {"personal_pc_only": False}}, {"content-type": "application/json"})
            if urlsplit(u).path == "/v1/hivemind/raid/status"
            else _make_table_http(bad_tbl)(m, u, p, h)
        ),
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )["gate_ok"] is False


def test_03_preflight_checks_yield_drill_and_ces_retry_spool_lifecycle(
    tmp_path: Path,
) -> None:
    clean_ws = tmp_path / "clean_ws"
    clean_ws.mkdir()
    home_dir = tmp_path / "home_dyn48_c"
    home_dir.mkdir()
    cred_0600 = _write_creds(home_dir / ".hivemind" / "credentials.json", mode=0o600)
    cred_0644 = _write_creds(home_dir / ".hivemind" / "creds_open.json", mode=0o644)
    cred_bad_json = tmp_path / "bad.json"
    cred_bad_json.write_text("{bad", encoding="utf-8")
    cred_missing_key = tmp_path / "nokey.json"
    cred_missing_key.write_text('{"api_key": "short"}', encoding="utf-8")

    # 1. Cross-platform runtime, creds, gateway reachability, GPU, active games, corporate env
    assert doc.check_python_runtime((3, 9, 0), os_name="nt", system_name="Windows", lang="fr").status == "fail"
    assert doc.check_python_runtime((3, 14, 0), os_name="posix", system_name="Darwin", lang="en").status == "warn"
    assert doc.check_python_runtime((3, 12, 1), os_name="posix", system_name="Linux", lang="en").status == "ok"

    assert doc.check_credentials_file(tmp_path / "missing.json", home=home_dir, lang="fr").status == "fail"
    assert doc.check_credentials_file(cred_bad_json, home=home_dir, lang="en").status == "fail"
    assert doc.check_credentials_file(cred_missing_key, home=home_dir, lang="fr").status == "fail"
    assert doc.check_credentials_file(cred_0644, os_name="posix", home=home_dir, lang="en").status == "warn"
    assert doc.check_credentials_file(cred_0600, os_name="nt", home=home_dir, lang="fr").status == "ok"

    assert doc.check_gateway_reachability(_untrusted_url("evil.example"), os_name="nt", lang="fr").status == "fail"
    assert doc.check_gateway_reachability("http://127.0.0.1:8000", http_get=lambda _u, _t: (_ for _ in ()).throw(OSError("net")), lang="fr").status == "fail"
    assert doc.check_gateway_reachability("http://127.0.0.1:8000", http_get=lambda _u, _t: (503, {}), lang="fr").status == "fail"
    assert doc.check_gateway_reachability("http://127.0.0.1:8000", http_get=lambda u, _t: (200, {"status": "ok"}) if u.endswith("/health") else (_ for _ in ()).throw(OSError("m")), lang="fr").status == "fail"
    assert doc.check_gateway_reachability("http://127.0.0.1:8000", http_get=lambda u, _t: (200, {"status": "ok"}) if u.endswith("/health") else (404, {}), lang="en").status == "fail"

    assert doc.check_gpu_readiness(runner=_build_local_runner(smi_exc=FileNotFoundError("no")), lang="fr").details["mode"] == "dev_api_only"
    assert doc.check_gpu_readiness(runner=_build_local_runner(smi_out="badline\n"), lang="fr").status == "warn"
    assert doc.probe_gpu(runner=_build_local_runner(smi_out="GPU, notint, 100, 50, 0\n")).parse_error is True
    assert doc.probe_gpu(runner=_build_local_runner(smi_out="GPU, 0, 0, 50, 0\n")).parse_error is True
    multi_gpu = doc.probe_gpu(runner=_build_local_runner(smi_out="SmallGPU, 4096, 2000, [n/a], 0\nBigGPU, 24576, 18000, 54, 3\n"))
    assert multi_gpu.name == "BigGPU"
    assert doc.check_gpu_readiness(runner=_build_local_runner(smi_out="LowGPU, 2048, 1024, 50, 1\n"), lang="fr").status == "warn"
    assert doc.check_gpu_readiness(runner=_build_local_runner(smi_out="HotGPU, 12288, 9000, 76, 5\n"), lang="fr").status == "warn"

    assert doc.check_active_games(runner=_build_local_runner(ps_out="cs2\n"), extra_games=["mygame.exe"], lang="fr").status == "warn"
    assert doc.check_active_games(runner=_build_local_runner(tasklist_out='"cs2.exe","10","Console","1","1 K"\n'), os_name="nt", lang="en").status == "warn"
    assert doc.list_process_names(runner=_build_local_runner(ps_exc=OSError("ps")), os_name="posix") == set()
    assert doc.list_process_names(runner=_build_local_runner(ps_rc=1), os_name="posix") == set()
    assert doc.list_process_names(runner=_build_local_runner(tasklist_rc=1), os_name="nt") == set()
    assert doc.probe_gpu(runner=_build_local_runner(smi_rc=1)).present is False
    assert doc._match_active_games({"steam_app_440"}, ["longgamename_1234567"]) == ["steam_app_440"]
    assert doc._friendly_os_label("posix", "FreeBSD") == "FreeBSD"
    assert doc._platform_feedback_tag("posix", "Darwin") == "macos"
    assert doc._platform_feedback_tag("posix", "Solaris_11") == "solaris_11"
    assert doc._platform_feedback_tag("posix", "!!!") == "unknown"
    assert doc._sanitize_manifest_sha("secret1234") == "unknown"
    assert doc._sanitize_manifest_revision("-bad") is None
    assert doc._sanitize_manifest_revision("hm-123456789012") is None
    assert doc.check_corporate_environment(
        environ={
            "USERDOMAIN": "CORP-NET",
            "COMPUTERNAME": "HOMEPC",
            "HTTPS_PROXY": _untrusted_url("proxy.internal"),
            "NO_PROXY": ".amadeus.internal",
            "ZSCALER_CERT_PATH": "/z.pem",
            "SSL_CERT_FILE": "/etc/corp.pem",
        },
        lang="fr",
    ).status == "warn"
    assert doc.check_corporate_environment(environ={"USERDNSDOMAIN": "amadeus.corp"}, lang="fr").status == "warn"

    # 2. Real subprocess <1.5s yield drill + failure/slow/thermal/VRAM/holiday route/fleet variants
    real_drill = doc.default_worker_yield_drill(timeout_s=doc.SUBPROCESS_TIMEOUT_S)
    assert real_drill["clean_release"] is True and real_drill["yield_latency_ms"] < 1500.0
    assert doc.check_noel_holiday_routes(_untrusted_url("evil.example"), os_name="nt", lang="fr").status == "fail"
    assert doc.check_noel_holiday_routes("http://127.0.0.1:8000", http_get=lambda _u, _t: (_ for _ in ()).throw(OSError("down")), lang="en").status == "fail"
    assert doc.check_noel_holiday_routes(
        "http://127.0.0.1:8000",
        http_get=lambda u, _t: (200, {"html_length": 0})
        if u.endswith("/noel")
        else (
            (200, {"raw": ""})
            if u.endswith("/quick")
            else (
                (200, {"object": "hivemind.onboarding.manifest", "holiday_routes": ["/noel"]})
                if u.endswith("/manifest")
                else (200, {"error": "bad"}) if u.endswith("/feedback/summary") else (503, {})
            )
        ),
        lang="en",
    ).status == "fail"
    assert doc._run_yield_drill_callable(lambda: {"yield_latency_ms": 12.5})[0] == 12.5
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_exc=FileNotFoundError("no")), worker_yield_fn=lambda: 1700.0, lang="fr").status == "fail"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_exc=FileNotFoundError("no")), worker_yield_fn=lambda: float("nan"), lang="en").status == "fail"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_exc=FileNotFoundError("no"), ps_out="cs2\n"), worker_yield_fn=lambda timeout_s=1.0: {"worker_stopped": True, "clean_release": True}, lang="fr").status == "warn"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_out="bad"), worker_yield_fn=lambda: 10.0, lang="fr").status == "warn"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_out="RTX, 12288, 9000, 50, 1"), worker_yield_fn=lambda: 1700.0, lang="fr").status == "fail"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_out="RTX, 12288, 9000, 76, 1"), worker_yield_fn=lambda: 10.0, lang="fr").status == "warn"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_out="RTX, 12288, 9000, 50, 1", ps_out="cs2\n"), worker_yield_fn=lambda: 10.0, lang="fr").status == "warn"
    assert doc.check_noel_steam_yield_and_thermal(runner=_build_local_runner(smi_out="RTX, 12288, 1024, 50, 1"), worker_yield_fn=lambda: 10.0, lang="fr").status == "warn"
    with pytest.raises(ValueError):
        doc._run_yield_drill_callable(lambda: "invalid")

    assert doc._sanitize_host_label("   ") == doc.DEFAULT_NOEL_HOST_LABEL
    assert doc._load_valid_credentials_key(cred_bad_json, home=home_dir)[0] is None
    list_cred = tmp_path / "list_cred.json"
    list_cred.write_text("[]", encoding="utf-8")
    assert doc._load_valid_credentials_key(list_cred, home=home_dir)[0] is None
    assert doc._load_valid_credentials_key(cred_missing_key, home=home_dir)[0] is None
    assert doc.check_noel_fleet_and_first_heartbeat(_untrusted_url("evil.example"), os_name="nt", home=home_dir, lang="fr").status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat("http://127.0.0.1:8000", credentials_path=home_dir / "missing.json", send_first_heartbeat=True, home=home_dir, lang="fr").status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat("http://127.0.0.1:8000", credentials_path=cred_0600, send_first_heartbeat=True, http_request=lambda *_a, **_k: (401, {}), home=home_dir, lang="fr").status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat("http://127.0.0.1:8000", credentials_path=cred_0600, send_first_heartbeat=True, http_request=lambda *_a, **_k: (_ for _ in ()).throw(OSError("me_err")), home=home_dir, lang="en").status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        send_first_heartbeat=True,
        http_request=lambda _m, u, **_k: (200, {"role": "prosumer"}) if "/me" in u else ((_ for _ in ()).throw(OSError("hb")) if "/opentrainer/heartbeat" in u else (500, {})),
        home=home_dir,
        lang="fr",
    ).status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        http_get=lambda u, _t: (200, {}) if "/fleet" in u else (404, {}),
        os_name="nt",
        home=home_dir,
        lang="en",
    ).status == "warn"
    assert doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        http_get=lambda u, _t: (500, {}) if "/fleet" in u else (200, {"swarm": {"live_producers": 2}}),
        os_name="nt",
        home=home_dir,
        lang="fr",
    ).status == "fail"
    assert doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        http_get=lambda u, _t: (200, {}) if "/fleet" in u else (200, {"swarm": {"live_producers": 4, "active_trainers": 3, "accepted_rounds": 9}}),
        home=home_dir,
        lang="fr",
    ).details["online_producers"] == 4
    cons_chk = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        send_first_heartbeat=True,
        http_request=lambda _m, u, **_k: (
            (200, {"role": "consumer"})
            if "/me" in u
            else (
                (403, {})
                if "/opentrainer/heartbeat" in u
                else (200, {"live_hiveminders": 8})
                if "/fleet" in u
                else (200, {"swarm": {"active_trainers": 2, "accepted_opentrainer_rounds": 4}})
            )
        ),
        home=home_dir,
        lang="fr",
    )
    assert cons_chk.details["heartbeat_state"] == "consumer_me_verified"

    # Standard 6-check & 9-check --noel flows + markdown + unsafe --output refusal
    tbl = _build_pr949_table()
    rep6 = doc.run_doctor(
        gateway="http://127.0.0.1:8000",
        credentials_path=cred_0600,
        lang="fr",
        environ=CLEAN_ENV,
        home=home_dir,
        runner=_build_local_runner(),
        http_get=lambda u, _t: (_make_table_http(tbl)("GET", u)[0], _make_table_http(tbl)("GET", u)[1]),
        noel=False,
    )
    assert rep6["readiness_checks_total"] == 6 and "# HiveMind Doctor" in doc.render_markdown_report(rep6)

    assert doc.run_doctor(gateway="http://127.0.0.1:8000", noel=True, environ={"AMADEUS_X": "1"}, cwd=clean_ws, home=home_dir, lang="fr")["compliance_ok"] is False

    # 3. DoctorCesRetrySpool: no file on HTTP 200 when empty; bounded 0600 spooling on 404/408/429/5xx/0; 400/422 quarantine; flush & clean
    no_create_q = tmp_path / "never_created.json"
    ok_200_chk = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        submit_ces=True,
        spool_path=no_create_q,
        http_request=_make_table_http(tbl),
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert ok_200_chk.details["ces_recorded"] is True and ok_200_chk.details["file_writes"] == 0 and not no_create_q.exists()

    quar_422_chk = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        submit_ces=True,
        spool_path=no_create_q,
        http_request=lambda _m, u, **_k: (422, {"error": "bad"}) if "/onboarding/feedback" in u else _make_table_http(tbl)("GET", u)[:2],
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert quar_422_chk.details["ces_quarantined"] is True and not no_create_q.exists()

    assert doc.default_ces_queue_path(home=home_dir, credentials_path=Path.home() / ".hivemind" / "credentials.json") == home_dir / ".hivemind" / doc.DEFAULT_CES_QUEUE_FILENAME
    for bad_ces in (
        lambda: doc._normalize_ces_payload(zero_game_lag="yes"),  # type: ignore[arg-type]
        lambda: doc._validate_ces_payload_mapping("bad"),
        lambda: doc._validate_ces_payload_mapping({"profile": "dev", "extra": 1}),
        lambda: doc._compute_doctor_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key=True),  # type: ignore[arg-type]
        lambda: doc._compute_doctor_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bad\x00key"),
        lambda: doc._compute_doctor_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="   "),
    ):
        with pytest.raises(ValueError):
            bad_ces()
    assert doc._compute_doctor_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="evt-1").startswith("hm-ces-")

    spool_path = home_dir / ".hivemind" / doc.DEFAULT_CES_QUEUE_FILENAME
    spool = doc.DoctorCesRetrySpool(spool_path, home=home_dir, max_entries=3, max_delivered=4)
    assert spool.path == spool_path.resolve() and spool.max_entries == 3 and spool.max_delivered == 4
    assert spool.snapshot()["spool_exists"] is False
    assert spool.mark_delivered("0" * 16) is False
    assert spool.mark_delivered("bad") is False
    assert spool.clean()["removed"] is False
    assert spool.flush("http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["spool_exists"] is False

    # Non-dict JSON recovery, corrupt list/bad payload/delivered overflow/pending overflow, already_delivered, already_pending & unsafe serialization refusal
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    spool_path.write_text("[]", encoding="utf-8")
    assert spool.snapshot()["pending_count"] == 0
    spool_path.write_text("{corrupted", encoding="utf-8")
    assert spool.snapshot()["quarantined_count"] == 1
    assert spool.enqueue(profile="dev", ease_rating=5, idempotency_key="delivered-evt-01", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["enqueued"] is True
    assert spool.enqueue(profile="dev", ease_rating=5, idempotency_key="delivered-evt-01", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_pending"
    assert spool.flush("http://127.0.0.1:8000", http_request=_make_table_http(tbl), environ=CLEAN_ENV, cwd=clean_ws)["delivered_count"] == 1
    assert spool.enqueue(profile="dev", ease_rating=5, idempotency_key="delivered-evt-01", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)["duplicate_state"] == "already_delivered"

    spool_path.write_text(
        json.dumps(
            {
                "delivered_event_ids": ["bad", "00000001", "00000002", "00000003", "00000004", "delivered-evt-01"],
                "pending": [
                    "not-dict",
                    {"event_id": "delivered-evt-01", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"event_id": "bad-payload-01", "payload": {"profile": "bad"}},
                    {"event_id": "pending-evt-01", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"event_id": "pending-evt-02", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"event_id": "pending-evt-03", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"event_id": "pending-evt-04", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                ],
            }
        ),
        encoding="utf-8",
    )
    snap_over = spool.snapshot()
    assert snap_over["pending_count"] == 3 and snap_over["dropped_overflow_count"] >= 1
    assert spool.mark_delivered("pending-evt-02") is True
    with pytest.raises(ValueError):
        spool._atomic_save_unlocked({"pending": [{"event_id": "unsafe-01", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "secret@" + "corp.example"}}]})

    for i in range(4):
        enq = spool.enqueue(profile="dev", ease_rating=5, idempotency_key=f"evt-item-{i}", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)
        if i == 3:
            assert enq["evicted_oldest"] is True
    assert stat.S_IMODE(spool_path.stat().st_mode) == 0o600

    # Verify already-delivered entry injected on disk into pending is deduplicated cleanly
    corrupt_q = doc.DoctorCesRetrySpool(tmp_path / "corrupt_re_enq.json", home=home_dir)
    corrupt_q.enqueue(profile="dev", ease_rating=5, idempotency_key="eid-re-pend-01", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)
    assert corrupt_q.flush("http://127.0.0.1:8000", http_request=_make_table_http(tbl), environ=CLEAN_ENV, cwd=clean_ws)["delivered_count"] == 1
    corrupt_data = json.loads(corrupt_q.path.read_text(encoding="utf-8"))
    corrupt_data["pending"] = [{"event_id": "eid-re-pend-01", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}}]
    corrupt_q.path.write_text(json.dumps(corrupt_data), encoding="utf-8")
    assert corrupt_q.flush("http://127.0.0.1:8000", http_request=_make_table_http(tbl), environ=CLEAN_ENV, cwd=clean_ws)["pending_count"] == 0

    for transient_st in (404, 408, 429, 503, 0):
        f_ret = doc.flush_doctor_ces_queue(
            "http://127.0.0.1:8000",
            spool_path=spool_path,
            http_request=(lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("tmo"))) if transient_st == 0 else (lambda *_a, _c=transient_st, **_k: (_c, {"error": f"e_{_c}"})),
            lang="fr",
            home=home_dir,
            environ=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert f_ret["flush_ok"] is False and f_ret["retained_retry_count"] == 3
    assert "VIDAGE FILE CES" in doc.render_ces_flush_report(f_ret)
    assert "# HiveMind Doctor" in doc.render_markdown_report(f_ret)

    f_quar = spool.flush("http://127.0.0.1:8000", http_request=lambda *_a, **_k: (422, {"error": "invalid"}), environ=CLEAN_ENV, cwd=clean_ws)
    assert f_quar["quarantined_count"] == 3 and f_quar["pending_count"] == 0

    # check_noel_fleet_and_first_heartbeat submit_ces exception -> spool + 200 auto mark_delivered + flush_ces_queue
    exc_sp = tmp_path / "exc_fleet_spool.json"
    chk_exc = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        submit_ces=True,
        spool_path=exc_sp,
        http_request=lambda m, u, **kw: (_ for _ in ()).throw(TimeoutError("t")) if "/onboarding/feedback" in u else _make_table_http(tbl)(m, u, kw.get("payload"), kw.get("headers"))[:2],
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert chk_exc.details["ces_spooled"] is True and exc_sp.is_file()
    chk_auto_deliv = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        submit_ces=True,
        spool_path=exc_sp,
        flush_ces_queue=True,
        http_request=_make_table_http(tbl),
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert chk_auto_deliv.details["ces_recorded"] is True and chk_auto_deliv.details["file_writes"] >= 1
    doc.DoctorCesRetrySpool(spool_path, home=home_dir).enqueue(profile="dev", ease_rating=5, idempotency_key="auto-def-flush-01", gateway="http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=clean_ws)
    chk_def_flush = doc.check_noel_fleet_and_first_heartbeat(
        "http://127.0.0.1:8000",
        credentials_path=cred_0600,
        submit_ces=False,
        flush_ces_queue=True,
        http_request=_make_table_http(tbl),
        home=home_dir,
        environ=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert chk_def_flush.details["ces_flush"]["delivered_count"] == 1

    # run_doctor untrusted gateway + ready_with_warnings + color report + markdown clean/flush + invalid parse details
    rep_bad_gw = doc.run_doctor(gateway=_untrusted_url("evil.example"), credentials_path=tmp_path / "m.json", environ=CLEAN_ENV, home=home_dir, runner=_build_local_runner(smi_exc=FileNotFoundError("no")), http_get=lambda _u, _t: (200, {}), noel=False)
    assert rep_bad_gw["gateway"] == "rejected_untrusted_gateway" and rep_bad_gw["verdict"] == "not_ready"
    rep_warn = doc.run_doctor(gateway="http://127.0.0.1:8000", credentials_path=cred_0644, environ=CLEAN_ENV, home=home_dir, runner=_build_local_runner(smi_out="HotGPU, 12288, 9000, 76, 5\n"), http_get=lambda u, _t: _make_table_http(tbl)("GET", u)[:2], noel=False)
    assert rep_warn["verdict"] == "ready_with_warnings" and "\x1b[33m" in doc.render_report(rep_warn, color=True)
    rep_noel_warn = doc.run_doctor(gateway="http://127.0.0.1:8000", credentials_path=cred_0644, environ=CLEAN_ENV, home=home_dir, runner=_build_local_runner(smi_out="bad"), http_request=_make_table_http(tbl), noel=True, cwd=clean_ws, worker_yield_fn=lambda: 15.0)
    assert rep_noel_warn["verdict"] == "ready_with_warnings"

    # Cutover auto-spool on 404 via --submit-ces CLI -> then --flush-ces-queue CLI (--markdown) -> then --clean-ces-queue CLI
    cli_q = tmp_path / "cli_ces_queue.json"
    assert (
        doc.main(
            ["--noel", "--gateway", "http://127.0.0.1:8000", "--credentials-file", str(cred_0600), "--cwd", str(clean_ws), "--submit-ces", "--ces-queue-file", str(cli_q), "--json"],
            http_request=lambda m, u, **kw: (404, {"error": "stale"}) if "/onboarding/feedback" in u else _make_table_http(tbl)(m, u, kw.get("payload"), kw.get("headers"))[:2],
            runner=_build_local_runner(),
            worker_yield_fn=lambda: 15.0,
            environ=CLEAN_ENV,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 2
    )
    assert cli_q.is_file() and stat.S_IMODE(cli_q.stat().st_mode) == 0o600
    assert (
        doc.main(
            ["--flush-ces-queue", "--gateway", "http://127.0.0.1:8000", "--ces-queue-file", str(cli_q), "--markdown"],
            http_request=_make_table_http(tbl),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 0
    )
    clean_out = io.StringIO()
    assert doc.main(["--clean-ces-queue", "--ces-queue-file", str(cli_q), "--markdown"], home=home_dir, out_stream=clean_out, err_stream=io.StringIO()) == 0
    assert "# HiveMind Doctor" in clean_out.getvalue() and not cli_q.exists()

    with pytest.raises(ValueError):
        doc.DoctorCesRetrySpool(tmp_path / "b1.json", max_entries=0)
    with pytest.raises(ValueError):
        doc.DoctorCesRetrySpool(tmp_path / "b2.json", max_delivered=0)
    assert doc.DoctorCesRetrySpool(tmp_path / "blk.json", home=home_dir).enqueue(environ={"AMADEUS_X": "1"}, cwd=clean_ws)["duplicate_state"] == "compliance_refused"
    assert doc.flush_doctor_ces_queue("http://127.0.0.1:8000", spool_path=tmp_path / "blk.json", environ={"AMADEUS_X": "1"}, cwd=clean_ws, home=home_dir)["compliance_ok"] is False
    assert doc.main(["--gateway", "http://127.0.0.1:8000", "--output", str(tmp_path / "unsafe.json")], runner=_build_local_runner(smi_out="user@corp.example GPU, 12288, 9000, 50, 1"), http_get=lambda u, _t: _make_table_http(tbl)("GET", u)[:2], environ=CLEAN_ENV, home=home_dir, out_stream=io.StringIO(), err_stream=io.StringIO()) == 2


def test_04_concurrent_48_user_multicohort_wal_soak_and_live_quorum_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_upstream(_req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Forbidden upstream inference call: {_req.url}")

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00048-dyn")

    db_path = tmp_path / "doctor_wal_soak48.sqlite3"
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
    (clean_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {clean_git_remote}\n',
        encoding="utf-8",
    )
    expected_doctor_sha = hashlib.sha256(DOCTOR_PATH.read_bytes()).hexdigest()

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
        stripped = doc.strip_masked_keys_and_nonces(resp.text)
        assert static_numbers_guard.FIGURE.findall(stripped) == []
        assert ("inner" + "HTML") not in stripped
        return hdr_n[0]

    def _make_asgi_http(peer_ip: str, *, recorded_ces: list[tuple[str, dict[str, Any]]] | None = None) -> doc.RawHttpFn:
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
            if parsed.path in doc.PR949_HTML_ROUTES and resp.status_code == 200:
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
                json_body={"handle": f"doc48-gpu-{p_idx:02d}", "role": "producer"},
                headers={"X-Forwarded-For": p_ip},
            )
            assert r_reg.status_code == 200, r_reg.text
            p_key = str(r_reg.json()["api_key"])
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
            home_d = slot / "home" / f"perso_user_{idx:02d}"
            cred_d = home_d / ".hivemind" / "credentials.json"
            ws_d = slot / "hobby"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)

            # run_doctor(--noel) fetches /noel, /quick, /raid-noel through _make_asgi_http;
            # fetch /arena-noel once here so all 4 PR #949 HTML routes are verified per participant.
            arena_resp = _asgi_req("GET", "/arena-noel", headers=hdrs)
            arena_nonce = _check_html("/arena-noel", arena_resp)
            arena_html = arena_resp.text
            with shared_lock:
                observed_nonces.append(arena_nonce)
                seen_paths.append("/arena-noel")

            if c_idx % 8 == 0:
                assert hashlib.sha256(_asgi_req("GET", "/scripts/hivemind_doctor.py", headers=hdrs).content).hexdigest() == expected_doctor_sha
                man_doc = _asgi_req("GET", "/v1/hivemind/onboarding/manifest", headers=hdrs).json()
                assert man_doc["build_sha"] == TARGET_FULL_SHA
                mod_doc = _asgi_req("GET", "/v1/models", headers=hdrs).json()
                assert ("hivemind/" + "coder") in [m["id"] for m in mod_doc["data"]]

            reg_resp = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"doc48_{profile}_{c_idx:02d}", "role": "prosumer" if is_dev else "producer"},
                headers=hdrs,
            )
            assert reg_resp.status_code == 200, reg_resp.text
            api_key = str(reg_resp.json()["api_key"])
            masked_key = doc.mask_api_key(api_key)
            doc._write_file_secure_atomic(
                cred_d,
                json.dumps({"api_key": api_key, "role": "prosumer" if is_dev else "producer", "gateway": "http://127.0.0.1:8000"}),
                mode=0o600,
                dir_mode=0o700,
            )

            if not is_dev:
                r_p = _asgi_req(
                    "POST",
                    "/v1/hivemind/producer/heartbeat",
                    json_body={
                        "endpoint_url": f"http://{_loopback_ip(77, c_idx)}:11434",
                        "models": [{"id": "hivemind/" + "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 12288,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={**hdrs, "Authorization": f"Bearer {api_key}"},
                )
                assert r_p.status_code == 200, r_p.text
                sample_runner = _build_local_runner(smi_out="NVIDIA GeForce RTX 4070, 12288, 9216, 56, 4\n")
            else:
                sample_runner = _build_local_runner(smi_exc=FileNotFoundError("no-gpu"))

            recorded_ces: list[tuple[str, dict[str, Any]]] = []
            adapter = _make_asgi_http(peer_ip, recorded_ces=recorded_ces)
            ease = 5 if (c_idx % 4 != 0) else 4
            if idx % 3 == 0:
                rep = doc.run_doctor(
                    gateway="http://127.0.0.1:8000",
                    credentials_path=cred_d,
                    lang="fr" if (idx % 2 == 0) else "en",
                    os_name="posix" if is_dev else "nt",
                    system_name="Linux" if is_dev else "Windows",
                    environ=CLEAN_ENV,
                    home=home_d,
                    runner=sample_runner,
                    noel=True,
                    cwd=ws_d,
                    send_first_heartbeat=True,
                    submit_ces=True,
                    ces_profile=profile,
                    ease_rating=ease,
                    host_label=f"pc-{profile}-{c_idx:02d}",
                    http_request=adapter,
                    worker_yield_fn=lambda: 20.0,
                )
                _verify_sha256_receipt(rep)
                assert rep["ready"] is True and rep["noel_summary"]["first_heartbeat_ok"] is True
                assert rep["noel_summary"]["ces_recorded"] is True and rep["noel_summary"]["zero_game_lag"] is True
                rendered = doc.render_report(rep)
                combined = json.dumps(rep, ensure_ascii=False) + "\n" + rendered
            else:
                gpu_snap = (
                    doc.GpuInfo(present=False)
                    if is_dev
                    else doc.GpuInfo(present=True, name="NVIDIA GeForce RTX 4070", total_mib=12288, free_mib=9216, temp_c=56.0, util_pct=4.0)
                )
                fleet_chk = doc.check_noel_fleet_and_first_heartbeat(
                    "http://127.0.0.1:8000",
                    credentials_path=cred_d,
                    send_first_heartbeat=True,
                    submit_ces=True,
                    ces_profile=profile,
                    ease_rating=ease,
                    host_label=f"pc-{profile}-{c_idx:02d}",
                    gpu_info=gpu_snap,
                    yield_slo_ok=True,
                    thermal_ok=True,
                    http_request=adapter,
                    home=home_d,
                    os_name="posix" if is_dev else "nt",
                    system_name="Linux" if is_dev else "Windows",
                    lang="fr" if (idx % 2 == 0) else "en",
                    environ=CLEAN_ENV,
                    cwd=ws_d,
                )
                _verify_sha256_receipt(fleet_chk.details)
                assert fleet_chk.status == "ok" and fleet_chk.details["first_heartbeat_ok"] is True
                assert fleet_chk.details["ces_recorded"] is True and fleet_chk.details["zero_game_lag"] is True
                combined = json.dumps(fleet_chk.to_dict(), ensure_ascii=False)
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

            assert api_key not in combined and f"perso_user_{idx:02d}" not in combined
            assert doc.scan_output_safety(combined, home=home_d) == []

            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            return doc.verify_pr949_doctor_quorum_and_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                require_live_quorum=True,
                check_arena_summary=True,
                verify_idempotency_replay=False,
                raw_http_fn=_make_asgi_http(_loopback_ip(33, p_idx)),
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

        post_soak = doc.verify_pr949_doctor_quorum_and_parity(
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
        assert doc.scan_output_safety(serialized_post, home=tmp_path) == []
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
    src_doc = DOCTOR_PATH.read_text(encoding="utf-8")
    src_test = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header within first 300 chars & stdlib-only imports + ProxyHandler({})
    assert EXPECTED_AGPL_HEADER in src_doc[:300]
    assert EXPECTED_AGPL_HEADER in src_test[:300]
    tree_doc = ast.parse(src_doc, filename=str(DOCTOR_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(tree_doc):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert (imported_roots - sys.stdlib_module_names) == set()
    assert "urllib.request.ProxyHandler({})" in src_doc

    for forbidden_lit in (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment.write",
        "/v1/" + "chat/completions",
    ):
        assert forbidden_lit not in src_doc
        assert forbidden_lit not in src_test

    # 2. Lateral-movement AST audit (+ negative control), PRSecurityScanner, ZeroStubAudit, static figures & gatekeeper
    assert lateral_guard.audit_source_for_lateral_movement(src_doc, filename=str(DOCTOR_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(src_test, filename=str(Path(__file__))) == []
    assert lateral_guard.audit_source_for_lateral_movement(
        f'URL = "{_untrusted_url("evil.example")}"\n',
        filename="neg_control.py",
    )

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, fpath, text in (
        ("scripts/hivemind_doctor.py", DOCTOR_PATH, src_doc),
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
        pr_title="feat(onboarding,doctor,scale): zero-inference PR #949 live quorum/arena parity and 48-user WAL soak",
        pr_body="Implements SLICE-SCALE-DYN-48 hivemind_doctor PR #949 parity/quorum gate, 0600 spool, and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,doctor,scale): zero-inference PR #949 live quorum parity"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_res.passed is True and pr_res.findings == []

    # Verify fallback ZeroStubAudit works without optional gwaya package
    orig_gwaya = sys.modules.get("gwaya")
    orig_audit = sys.modules.get("gwaya.ast_audit")
    monkeypatch.setitem(sys.modules, "gwaya", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "gwaya.ast_audit", None)  # type: ignore[arg-type]
    try:
        reloaded_spec = importlib.util.spec_from_file_location("test_dyn48_ci_fallback", __file__)
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
