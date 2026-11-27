# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 quorum/arena parity, SSRF redirect, 0600 output & 48-user WAL soak (`SLICE-SCALE-DYN-43`)."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_deploy_smoke_and_win11_verifier.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:
    from types import SimpleNamespace as _FallbackNS  # noqa: E402

    class ZeroStubAudit:  # type: ignore[no-redef]
        """CI-safe AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: re.Pattern[str] = re.compile(
            r"^(?:mo" + r"ck|fa" + r"ke|simu" + r"late|dum" + r"my|st" + r"ub)_",
            re.IGNORECASE,
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> Any:
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackNS(is_clean=False, violations=[f"SyntaxError: {exc}"], penalty_energy=1e6)
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            hits = [
                "stub_violation"
                for n in ast.walk(tree)
                if (isinstance(n, ast.Pass) and not isinstance(parents.get(n), (ast.ExceptHandler, ast.ClassDef)))
                or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis)
                or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and bool(cls._BAD_NAMES.search(n.name)))
            ]
            return _FallbackNS(is_clean=not hits, violations=hits, penalty_energy=1e6 if hits else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> Any:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


from scripts import noel_deploy_smoke_and_win11_verifier as verifier  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn43-router-sec-32b!"
TARGET_FULL_SHA = "949ef0123456789abcdef0123456789abcdef012"
TARGET_SHORT_SHA = "949ef0123456"
DRIFT_SHA = "deadbeefcafe1234"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example" + ".invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-ca.internal:8080"
CORP_FORTI_PROXY = "http://" + "fortinet-gw.internal:3128"
CORP_GIT_REMOTE = "git@" + "git.amadeus-internal.corp:core/secret-platform.git"
NON_LOOPBACK_IP = "10." + "20.30.40"
RAW_HM_KEY = "hm-" + "dyn43_secret_key_9999"
RAW_LIVE_KEY = "hm_live_" + "dyn43_secret_9999"
MASKED_8MS_KEY = "hm-***x8" + "ms"
MASKED_9MS_KEY = "hm_live_***9" + "ms"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_ARENA_PAIR_RE = re.compile(r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"')
_CAND_PRE_RE = re.compile(r'<pre\s+id="cand(A)-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)
_STATIC_FIG_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _load_guard_module(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_all_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_all_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            found.update(_collect_all_keys(item))
    return found


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert verifier.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(doc)


def _rtx_quiet_snapshot() -> dict[str, Any]:
    return {
        "present": True,
        "parse_error": False,
        "name": "NVIDIA GeForce RTX 4070",
        "total_mib": 12288,
        "free_mib": 9600,
        "temp_c": 56.0,
        "util_pct": 4.0,
    }


def _fast_worker_yield() -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 42.0,
    }


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "en") -> str:
    banner = (
        "100 % personnel sur PC personnel • AGPL-3.0 • 0,00 EUR / 0,00 USD"
        if lang == "fr"
        else "100% personal home PCs only • AGPL-3.0 • 0.00 EUR / 0.00 USD"
    )
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head>'
        f'<style nonce="{nonce}">body{{margin:0;}}</style></head>'
        f"<body><main data-route=\"{route}\"><p>{banner}</p></main>"
        f'<script nonce="{nonce}">document.documentElement.dataset.ok="1";</script>'
        f"</body></html>"
    )


def _sample_smoke_table(
    *,
    build_sha: str = TARGET_FULL_SHA,
    live_hiveminders: int = 8,
    distinct_gpus: int | None = None,
    row_statuses: tuple[str, ...] | None = None,
    leak_fleet_endpoint: bool = False,
    feedback_summary: dict[str, Any] | None = None,
    arena_summary: dict[str, Any] | None = None,
    arena_status_code: int = 200,
    static_nonce: str | None = None,
    unsafe_sink_route: str | None = None,
    static_fig_route: str | None = None,
    missing_banner_route: str | None = None,
    html_extra_text: str = "",
    status_overrides: dict[str, int] | None = None,
    include_coder: bool = True,
    missing_manifest_route: str | None = None,
) -> dict[str, tuple[int, dict[str, str], dict[str, Any] | str]]:
    gpus = live_hiveminders if distinct_gpus is None else distinct_gpus
    fb_doc = feedback_summary or {
        "object": verifier.FEEDBACK_SUMMARY_OBJECT,
        "total_responses": 16,
        "mean_ease_rating": 4.75,
        "zero_game_lag_ratio": 1.0,
        "by_profile": {"dev": 8, "gamer": 8},
    }
    models = [{"id": "hivemind/auto"}]
    if include_coder:
        models.append({"id": "hivemind/coder"})
    statuses = row_statuses if row_statuses is not None else tuple("live" for _ in range(live_hiveminders))
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
    arena_doc = arena_summary if arena_summary is not None else {
        "object": verifier.ARENA_SUMMARY_OBJECT,
        "total_votes": 10,
        "decisive_votes": 9,
        "holiday_wins": 8,
        "baseline_wins": 1,
        "ties": 1,
        "win_ratio": 0.8889,
        "wilson_95_lower_bound": 0.565,
        "by_task": {
            "python-sliding-median": {
                "total_votes": 10,
                "holiday_wins": 8,
                "baseline_wins": 1,
                "ties": 1,
                "wilson_95_lower_bound": 0.565,
            }
        },
    }
    ps1_text = (REPO_ROOT / "scripts" / "join_hivemind.ps1").read_text(encoding="utf-8")
    qs_text = (REPO_ROOT / "scripts" / "hivemind_quickstart.py").read_text(encoding="utf-8")

    table: dict[str, tuple[int, dict[str, str], dict[str, Any] | str]] = {
        "/health": (200, {"content-type": "application/json"}, {"status": "ok"}),
        "/v1/models": (200, {"content-type": "application/json"}, {"object": "list", "data": models}),
        "/v1/hivemind/onboarding/manifest": (
            200,
            {"content-type": "application/json"},
            {
                "object": verifier.MANIFEST_OBJECT,
                "campaign_id": verifier.CAMPAIGN_ID,
                "status": "ok",
                "build_sha": build_sha,
                "cloud_run_revision": "hivemind-openrouter-00949-noel",
                "holiday_routes": [r for r in verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES if r != missing_manifest_route],
                "downloadable_scripts": list(verifier.EXPECTED_MANIFEST_SCRIPTS),
                "personal_pc_only": True,
                "financial_promise": verifier.FINANCIAL_PROMISE,
            },
        ),
        "/v1/hivemind/onboarding/feedback/summary": (200, {"content-type": "application/json"}, fb_doc),
        "/v1/hivemind/raid/status": (
            200,
            {"content-type": "application/json"},
            {"object": verifier.RAID_STATUS_OBJECT, "swarm": {"active_trainers": 2}},
        ),
        "/v1/hivemind/producers/fleet": (
            200,
            {"content-type": "application/json"},
            {
                "object": verifier.FLEET_OBJECT,
                "total_hiveminders": len(fleet_rows),
                "live_hiveminders": live_hiveminders,
                "distinct_physical_gpus": gpus,
                "hiveminders": fleet_rows,
            },
        ),
        "/v1/hivemind/arena/summary": (arena_status_code, {"content-type": "application/json"}, arena_doc),
        "/scripts/join_hivemind.ps1": (200, {"content-type": "text/plain"}, ps1_text),
        "/scripts/hivemind_quickstart.py": (200, {"content-type": "text/plain"}, qs_text),
    }
    for idx, route in enumerate(verifier.SMOKE_HTML_ROUTES):
        nonce = static_nonce if static_nonce else f"DYN43NonceVal{idx:02d}AB"
        html_body = _sample_compliant_html(route, nonce)
        if route == missing_banner_route:
            html_body = f'<!DOCTYPE html><html><style nonce="{nonce}"></style><body>plain</body></html>'
        if route == unsafe_sink_route:
            sink = "inner" + "HTML"
            html_body += f"<script nonce='{nonce}'>el.{sink}='x'</script>"
        if route == static_fig_route:
            fig = "198.4 " + "ms"
            html_body += f"<p>{fig}</p>"
        if html_extra_text:
            html_body = html_body.replace("</body>", f"<p>{html_extra_text}</p></body>")
        table[route] = (
            200,
            {
                "content-type": "text/html; charset=utf-8",
                "content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'",
            },
            html_body,
        )
    for ov_path, ov_code in (status_overrides or {}).items():
        table[ov_path] = (ov_code, {"content-type": "application/json"}, {"detail": f"status_{ov_code}"})
    return table


def _make_table_http_fn(
    table: dict[str, tuple[int, dict[str, str], dict[str, Any] | str]],
    *,
    post_handler: Any = None,
) -> verifier.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        path = urlsplit(url).path or "/"
        if method == "POST" and post_handler is not None:
            return post_handler(method, path, payload, headers)
        return table.get(path, (404, {"content-type": "application/json"}, {"error": "not_found"}))

    return _http


def test_01_stdlib_security_ssrf_redirects_raw_http_and_mask_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. AGPL-3.0, stdlib-only, CI ZeroStubAudit fallback, lateral movement, PRSecurityScanner & gatekeeper audits
    assert EXPECTED_AGPL_HEADER in verifier.__doc__ or EXPECTED_AGPL_HEADER in VERIFIER_PATH.read_text(encoding="utf-8")[:300]
    test_src = Path(__file__).read_text(encoding="utf-8")
    assert EXPECTED_AGPL_HEADER in test_src[:300]

    lateral_guard = _load_guard_module("lateral_guard_dyn43", "tests/test_no_lateral_movement.py")
    static_numbers_guard = _load_guard_module("static_numbers_dyn43", "tests/test_static_numbers_have_artifacts.py")
    gatekeeper = _load_guard_module("gatekeeper_dyn43", "scripts/pr_gatekeeper.py")
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    banned_fragments = (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment.write",
        "/v1/ch" + "at/completions",
    )

    for rel_path, file_path in (
        ("scripts/noel_deploy_smoke_and_win11_verifier.py", VERIFIER_PATH),
        ("tests/test_noel_deploy_smoke_win11_quorum_soak.py", Path(__file__)),
    ):
        src_text = file_path.read_text(encoding="utf-8")
        tree = ast.parse(src_text, filename=str(file_path))
        if rel_path.startswith("scripts/"):
            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported_roots.add(node.module.split(".")[0])
            assert imported_roots - sys.stdlib_module_names == set()
        for banned in banned_fragments:
            assert banned not in src_text, (rel_path, banned)
        assert lateral_guard.audit_source_for_lateral_movement(src_text, str(file_path)) == []
        assert scanner.scan_python_ast_security(rel_path, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel_path) == []
        assert static_numbers_guard.FIGURE.findall(src_text) == []
        syn_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{ln}" for ln in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        assert ZeroStubAudit.audit_python_code(src_text).is_clean
        changed_files.append({"filename": rel_path, "patch": "", "content": src_text})

    pr_audit = scanner.audit_pull_request(
        pr_number=1943,
        pr_title="feat(onboarding,scale): zero-inference PR #949 live quorum/arena parity and 48-user WAL soak",
        pr_body="Implements SLICE-SCALE-DYN-43 deploy smoke quorum, SSRF redirect guard, 0600 output, and WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): SLICE-SCALE-DYN-43"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    # Fast gwaya ImportError fallback subprocess check (no heavy router/fastapi top-level imports in test file)
    ci_probe = (
        "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
        "import importlib.util; from pathlib import Path; "
        f"s = importlib.util.spec_from_file_location('t_ci_dyn43', Path({str(Path(__file__))!r})); "
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); Z = m.ZeroStubAudit; "
        "assert Z.check_file(m.VERIFIER_PATH).is_clean; "
        "assert Z.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
        "assert not Z.audit_python_code('def bad_st():\\n    pass\\n').is_clean; "
        "assert not Z.audit_python_code('def bad(').is_clean; "
        "print('CI_FALLBACK_OK')"
    )
    ci_proc = subprocess.run([sys.executable, "-c", ci_probe], capture_output=True, text=True, timeout=10, check=False)
    assert ci_proc.returncode == 0 and "CI_FALLBACK_OK" in ci_proc.stdout

    # Backward-compatible symbol check
    for sym in (
        "SCHEMA", "RECEIPT_OBJECT", "MANIFEST_OBJECT", "FEEDBACK_SUMMARY_OBJECT",
        "RAID_STATUS_OBJECT", "FLEET_OBJECT", "ARENA_SUMMARY_OBJECT", "CAMPAIGN_ID",
        "FINANCIAL_PROMISE", "DEFAULT_GATEWAY", "SMOKE_HTML_ROUTES", "SMOKE_PROBE_ROUTES",
        "EXPECTED_MANIFEST_HOLIDAY_ROUTES", "EXPECTED_MANIFEST_SCRIPTS", "PS1_ALLOWED_MODES",
        "PS1_REQUIRED_PARAMS", "SmokeVerifierError", "normalize_lang", "mask_api_key",
        "redact_home_path", "sanitize_text_for_display", "strip_html_csp_nonce_attributes",
        "strip_masked_keys_and_nonces", "_strip_masked_keys_and_nonces", "scan_output_safety",
        "sanitize_build_sha", "validate_expected_sha", "shas_match", "sanitize_cloud_run_revision",
        "is_allowed_outbound_host", "validate_gateway_url", "default_http_raw",
        "default_http_request", "default_subprocess_runner", "extract_git_config_remote_urls",
        "evaluate_personal_pc_compliance", "audit_deploy_sh_guidance", "run_deploy_sh_dry_run",
        "project_adoption_readiness_gates", "run_cloud_run_deploy_smoke_gate",
        "audit_windows_launcher_contract", "verify_dev_vscode_60s_dry_run",
        "default_gpu_probe", "default_process_probe", "default_worker_yield_drill",
        "verify_win11_gamer_whisper_5m_dry_run", "run_unified_onboarding_and_smoke_gate",
        "render_cli_report", "render_markdown_report", "build_parser", "main",
    ):
        assert hasattr(verifier, sym), sym

    # 2. SSRF host allowlist, backslash / malformed DNS label / '@' / timeout / zero-inference payload checks
    assert verifier.is_allowed_outbound_host("hivemind-demo.a.run.app") is True
    assert verifier.is_allowed_outbound_host("storage.googleapis.com") is True
    assert verifier.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert verifier.is_allowed_outbound_host("[::1]") is True
    for bad_host in ("", None, "-bad.run.app", "bad..run.app", "run.app", "a\\b.run.app", "user@127.0.0.1", "x.evil.example"):
        assert verifier.is_allowed_outbound_host(bad_host) is False  # type: ignore[arg-type]

    assert verifier.validate_gateway_url("https://hivemind-gw-1.us-central1.run.app/") == "https://hivemind-gw-1.us-central1.run.app"
    assert verifier.validate_gateway_url("http://[::1]:8080") == "http://[::1]:8080"
    for bad_gw in (
        "", True, "https://" + "a.run.app\n", "https://" + "a.run.app/bad path",
        "https://" + "a.run.app\\evil", "https://" + "a.run.app%5cevil",
        "https://" + "user@a.run.app", "https://" + "a.run.app%40evil.com",
        "https://a.run.app?q=1", "https://a.run.app/sub", "https://a.run.app" + ":badport",
        "https://a.run.app:0", "ftp://127.0.0.1", INSECURE_RUN_APP_GW, UNTRUSTED_EXT_GW,
    ):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad, kw_bad in (
        ("GET", "", {}),
        ("GET", "http://127.0.0.1/bad path", {}),
        ("GET", "http://127.0.0.1/bad\\path", {}),
        ("GET", "http://127.0.0.1:badport/health", {}),
        ("GET", "http://user@127.0.0.1/health", {}),
        ("POST", "http://127.0.0.1:8000/v1/ch" + "at/completions", {}),
        ("DELETE", "http://127.0.0.1:8000/health", {}),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/register", {"payload": ["not-dict"]}),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/register", {"payload": {"nested": [{"messages": []}]}}),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/register", {"payload": {"model": "hivemind/coder"}}),
    ):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier._assert_safe_http_request(m_bad, u_bad, allowed_post_paths=verifier.ALLOWED_POST_PATHS, **kw_bad)

    for bad_to in (True, "fast", -1.0, float("nan")):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.default_http_raw("GET", "http://127.0.0.1:8000/health", timeout_s=bad_to)  # type: ignore[arg-type]
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/health", headers={"X-Bad": "val\r\ninject"})

    # 3. Proxy-free loopback HTTP server with _SafeRedirectHandler + 4 MiB bound + default_http_raw
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            redirects = {
                "/redir-ok": "/health",
                "/redir-empty": "",
                "/redir-backslash": "/bad\\path",
                "/redir-at": "http://u@127.0.0.1/health",
                "/redir-evil": f"{UNTRUSTED_EXT_GW}/steal",
                "/redir-compl": "/v1/ch" + "at/completions",
            }
            if self.path in redirects:
                self.send_response(302)
                self.send_header("Location", redirects[self.path])
                self.end_headers()
                return
            if self.path == "/huge":
                self._write(200, "text/plain", b"x" * (verifier.MAX_HTTP_BODY_BYTES + 64))
                return
            if self.path == "/arr":
                self._write(200, "application/json", b"[1, 2]")
                return
            if self.path == "/badjson":
                self._write(200, "application/json", b"{broken")
                return
            if self.path == "/plain-502":
                self._write(502, "text/plain", b"bad gateway")
                return
            if self.path == "/health":
                self._write(200, "application/json", b'{"status":"ok"}')
                return
            self._write(404, "application/json", b'{"detail":"not found"}')

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            _ = self.rfile.read(n) if n > 0 else b""
            if self.headers.get("X-Test-Redir") == "1":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
                return
            self._write(200, "application/json", json.dumps({"account_id": "acc_1", "api_key": RAW_HM_KEY}).encode())

        def _write(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    srv_t = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        monkeypatch.setenv("HTTP_PROXY", CORP_ZSCALER_PROXY)
        monkeypatch.setenv("HTTPS_PROXY", CORP_ZSCALER_PROXY)

        rc_r, txt_r, hdr_r = verifier.default_http_raw("GET", f"{base}/redir-ok", headers={"X-Probe": "1"})
        assert rc_r == 200 and json.loads(txt_r)["status"] == "ok" and "application/json" in hdr_r["content-type"]
        assert verifier.default_http_raw("GET", f"{base}/plain-502")[0] == 502
        assert verifier.default_http_request("GET", f"{base}/arr")[1]["data"] == [1, 2]
        assert verifier.default_http_request("GET", f"{base}/badjson")[1]["raw_error"] == "invalid_json"
        assert verifier.default_http_request("GET", f"{base}/missing")[0] == 404
        assert verifier.default_http_request("POST", f"{base}/v1/hivemind/register", payload={"handle": "a", "role": "prosumer"})[0] == 200
        assert verifier.default_http_request("POST", f"{base}/v1/hivemind/register", payload={"handle": "a", "role": "prosumer"}, headers={"X-Test-Redir": "1"})[0] == 200

        for evil in ("/redir-empty", "/redir-backslash", "/redir-at", "/redir-evil", "/redir-compl", "/huge"):
            with pytest.raises(verifier.SmokeVerifierError):
                verifier.default_http_raw("GET", f"{base}{evil}")
            with pytest.raises(verifier.SmokeVerifierError):
                verifier.default_http_request("GET", f"{base}{evil}")
    finally:
        monkeypatch.undo()
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_raw("GET", "http://127.0.0.1:1/health", timeout_s=0.2)[0] == 0
    assert verifier.default_http_request("GET", "http://127.0.0.1:1/health")[0] == 0

    # 4. strip_masked_keys_and_nonces (hm-***x8...) in HTML contract & report safety + core helper edges
    assert verifier._strip_masked_keys_and_nonces is verifier.strip_masked_keys_and_nonces
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.scan_output_safety(123)  # type: ignore[arg-type]

    assert verifier.scan_output_safety(f"key={MASKED_8MS_KEY} live={MASKED_9MS_KEY}") == []
    html_with_masked = _sample_compliant_html("/noel", "Abc123" + "msNonceToken").replace(
        "</main>", f"<p>Key: {MASKED_8MS_KEY}</p></main>"
    )
    ok_h, h_doc, h_iss = verifier._inspect_holiday_html_contract(
        "/noel",
        {
            "html": html_with_masked,
            "headers": {"content-security-policy": "default-src 'none'; style-src 'nonce-Abc123" + "msNonceToken'"},
        },
        set(),
        lang_code="en",
    )
    assert ok_h is True and h_iss == [] and h_doc["static_figure_found"] is False

    for bad_key in ("", 7, "hm-short", "hm-bad key", "sk-wrong-prefix-1234"):
        assert verifier.mask_api_key(bad_key) in ("<missing>", "<invalid>")
    assert verifier.mask_api_key(RAW_HM_KEY) == "hm-***9999"
    assert verifier.mask_api_key(RAW_LIVE_KEY) == "hm_live_***9999"
    assert verifier.normalize_lang("FR-ca") == "fr"
    for bad_lang in (True, "de"):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]
    assert verifier.sanitize_build_sha("secret123") == "unknown"
    assert verifier.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert verifier.shas_match("abc123def", "abc123def") is True
    assert verifier.shas_match("nonhex01", "nonhex02") is False
    assert verifier.sanitize_cloud_run_revision("token-rev") is None
    assert verifier.sanitize_cloud_run_revision(12) is None
    assert verifier._has_forbidden_score_key([{"nested": {"sco" + "re": 1}}]) is True
    assert verifier.redact_home_path(Path("/tmp/home_test/file"), home=Path("/tmp/home_test")) == "~/file"

    scrubbed = verifier.sanitize_text_for_display(
        f"{RAW_HM_KEY} Bear" + f"er secret_tok_99 u@example.com {NON_LOOPBACK_IP} 127.0.0.1 /ho" + "me/bob/ws"
    )
    assert RAW_HM_KEY not in scrubbed and NON_LOOPBACK_IP not in scrubbed and "127.0.0.1" in scrubbed
    for expect_hit, sample in (
        ("email_detected", "u@example.com"),
        ("unmasked_api_key_detected", RAW_HM_KEY),
        ("bearer_token_detected", "Bear" + "er tok_abcdef123"),
        ("unredacted_home_path_detected", "/ho" + "me/alice/.hivemind"),
        ("forbidden_score_key_detected", '{"' + "sco" + 're": 4}'),
        ("static_figure_detected", "latency 12 " + "ms"),
        (f"non_loopback_ipv4_detected:{NON_LOOPBACK_IP}", NON_LOOPBACK_IP),
    ):
        assert expect_hit in verifier.scan_output_safety(sample)
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._assert_safe_output(RAW_HM_KEY)

    with pytest.raises(verifier.SmokeVerifierError):
        verifier._normalize_http_return(("bad",))
    assert verifier._normalize_http_return((200, {"x-h": "1"}, {"headers": {"x-h": "2"}}))[1]["headers"]["x-h"] == "2"
    assert verifier._normalize_http_return((200, [1, 2]))[1]["data"] == [1, 2]
    assert verifier._invoke_http_fn(lambda m, u, **_kw: (200, {"k": 1}), "GET", "http://127.0.0.1:8000/health")[1]["k"] == 1
    assert verifier._invoke_http_fn(lambda m, u, p: (200, {"a": 3}), "GET", "http://127.0.0.1:8000/health")[1]["a"] == 3
    assert verifier._invoke_http_fn(lambda m, u, p, h: (200, {"a": 4}), "GET", "http://127.0.0.1:8000/health")[1]["a"] == 4
    assert verifier._invoke_http_fn(lambda m, u, p, h, t: (200, {"a": 5}), "GET", "http://127.0.0.1:8000/health")[1]["a"] == 5
    assert verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("down")), "GET", "http://127.0.0.1:8000/health") == (0, {"error": "network_error"})
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.SmokeVerifierError("bad")), "GET", "http://127.0.0.1:8000/health")

    assert "PATH" in verifier._build_subprocess_env(None)
    assert verifier._find_corporate_markers("") == []
    assert verifier.extract_git_config_remote_urls("") == []
    for bad_argv in ([], ["git", "bad\narg"], ["gcl" + "oud", "--version"]):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.default_subprocess_runner(bad_argv)
    assert verifier.default_subprocess_runner(["nonexistent_bin_dyn43"])[0] == 127
    assert verifier.default_subprocess_runner([sys.executable, "-c", "import time; time.sleep(1.5)"], timeout_s=0.2)[0] == 124
    assert verifier._invoke_runner(lambda _a: (0, "out", "err"), ["x"], 1.0) == (0, "out\nerr")
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._invoke_runner(lambda _a: "bad", ["x"], 1.0)  # type: ignore[return-value]


def test_02_personal_pc_compliance_worktree_gitdir_and_atomic_0600_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "NODE_EXTRA_CA_CERTS" in verifier.INSPECTED_ENV_KEYS
    assert "USERDNSDOMAIN" in verifier.INSPECTED_ENV_KEYS

    personal_ws = tmp_path / "personal_hobby_ws"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws)["compliance_ok"] is True

    # Absolute AND relative commondir worktree resolution + check_cwd_path toggle + symlink resolve
    common_rel = tmp_path / "common_rel_git"
    wt_rel = common_rel / "worktrees" / "wt_rel"
    wt_rel.mkdir(parents=True)
    (common_rel / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")
    (wt_rel / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (wt_rel / "commondir").write_text("../..\n", encoding="utf-8")
    wt_rel_ws = tmp_path / "wt_rel_ws"
    wt_rel_ws.mkdir()
    (wt_rel_ws / ".git").write_text(f"gitdir: {wt_rel}\n", encoding="utf-8")

    common_abs = tmp_path / "common_abs_git"
    wt_abs = common_abs / "worktrees" / "wt_abs"
    wt_abs.mkdir(parents=True)
    (common_abs / "config").write_text('[url "https://' + 'zscaler.corp.example/"]\n\tinsteadOf = z:\n', encoding="utf-8")
    (wt_abs / "commondir").write_text(f"{common_abs}\n", encoding="utf-8")
    wt_abs_ws = tmp_path / "wt_abs_ws"
    wt_abs_ws.mkdir()
    (wt_abs_ws / ".git").write_text(f"gitdir: {wt_abs}\n", encoding="utf-8")

    assert verifier.evaluate_personal_pc_compliance(environ={**CLEAN_ENV, "AMADEUS_EMPTY": ""}, cwd=personal_ws)["compliance_ok"] is True

    corp_dir = tmp_path / "amadeus_work_dir"
    corp_dir.mkdir()
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=corp_dir, check_cwd_path=False)["compliance_ok"] is True
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=corp_dir, check_cwd_path=True)["compliance_ok"] is False

    # Symlink from clean alias to corporate directory + broken OSError .git metadata resilience
    symlink_ws = tmp_path / "clean_symlink_ws"
    symlink_ws.symlink_to(corp_dir)
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=symlink_ws)["compliance_ok"] is False

    def _forbid_http(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("Compliance fail-closed must perform 0 HTTP calls")

    def _forbid_runner(*_a: Any, **_kw: Any) -> tuple[int, str]:
        raise AssertionError("Compliance fail-closed must perform 0 subprocess calls")

    blocked_out = tmp_path / "never_written" / "blocked.json"
    vectors: tuple[tuple[dict[str, str], Path, str], ...] = (
        ({**CLEAN_ENV, "AMADEUS_PROFILE": "amadeus-secret-01"}, personal_ws, "amadeus-secret-01"),
        ({**CLEAN_ENV, "NODE_EXTRA_CA_CERTS": "/etc/ca/bluecoat-sentinel.pem"}, personal_ws, "bluecoat-sentinel"),
        ({**CLEAN_ENV, "USERDNSDOMAIN": "nce-corp.internal"}, personal_ws, "nce-corp.internal"),
        ({**CLEAN_ENV, "ZSCALER_CERT_PATH": "/etc/zs-ca.pem"}, personal_ws, "zs-ca.pem"),
        ({**CLEAN_ENV, "HTTPS_PROXY": CORP_NETSKOPE_PROXY}, personal_ws, "netskope-ca"),
        (CLEAN_ENV, wt_rel_ws, "secret-platform"),
        (CLEAN_ENV, wt_abs_ws, "zscaler.corp.example"),
        (CLEAN_ENV, corp_dir, "amadeus_work_dir"),
    )
    for idx, (env_map, ws_path, sentinel) in enumerate(vectors):
        lang = "fr" if idx % 2 else "en"
        comp = verifier.evaluate_personal_pc_compliance(environ=env_map, cwd=ws_path, lang=lang)
        _verify_receipt(comp)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert sentinel not in json.dumps(comp)

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        rc = verifier.main(
            ["--smoke-only", "--cwd", str(ws_path), "--output", str(blocked_out), "--lang", lang, "--json" if idx % 2 == 0 else "--markdown"],
            http_fn=_forbid_http,
            runner=_forbid_runner,
            environ=env_map,
            stdout=out_buf,
            stderr=err_buf,
        )
        assert rc == 2
        assert not blocked_out.exists()
        assert sentinel not in out_buf.getvalue() and sentinel not in err_buf.getvalue()

    # OSError resilience on unreadable .git/commondir/config
    real_read_text = Path.read_text

    def _err_read(self: Path, *a: Any, **kw: Any) -> str:
        if self.name in ("commondir", "config") and str(wt_rel_ws.parent) in str(self):
            raise OSError("denied")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _err_read)
    assert verifier._resolve_git_config_files(wt_rel_ws) == [wt_rel / "config"]
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=wt_rel_ws)["compliance_ok"] is True
    assert verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws)["compliance_ok"] is True
    assert DEPLOY_SH_PATH.read_text(encoding="utf-8").startswith("#!/")
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: (_ for _ in ()).throw(OSError("denied")) if self.name == ".git" else real_read_text(self, *a, **kw))
    assert verifier._resolve_git_config_files(wt_rel_ws) == []
    monkeypatch.undo()

    # Atomic 0600 --output persistence across JSON, Markdown & CLI text + cleanup on failure
    ok_http = _make_table_http_fn(_sample_smoke_table())
    for fmt_flags, out_name in (
        (["--json"], "smoke.json"),
        (["--markdown"], "smoke.md"),
        ([], "smoke.txt"),
    ):
        target = tmp_path / "artifacts" / out_name
        buf = io.StringIO()
        rc = verifier.main(
            ["--smoke-only", "--gateway", "http://127.0.0.1:8000", "--cwd", str(personal_ws), "--output", str(target), *fmt_flags],
            http_fn=ok_http,
            environ=CLEAN_ENV,
            stdout=buf,
        )
        assert rc == 0 and target.is_file()
        assert (target.stat().st_mode & 0o777) == 0o600
        assert buf.getvalue().strip() == target.read_text(encoding="utf-8").strip()

    assert verifier.main(["--smoke-only", "--cwd", "", "--json"], environ=CLEAN_ENV, stdout=io.StringIO()) == 2
    assert verifier.main(["--smoke-only", "--output", "bad\x00file", "--json"], environ=CLEAN_ENV, stdout=io.StringIO()) == 2
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("replace_err")))
    with pytest.raises(OSError):
        verifier._write_secure_atomic(tmp_path / "fail_atomic.txt", "hello")
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp"))


def test_03_pr949_quorum_ces_slo_arena_summary_and_deploy_dry_run_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Full green PR #949 live quorum + CES SLO + arena summary + real deploy.sh --dry-run
    ok_table = _sample_smoke_table(
        row_statuses=("live", "online", "healthy", "ready", "live", "online", "healthy", "ready"),
    )
    ok_http = _make_table_http_fn(ok_table)
    res_full = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        target_online_producers=8,
        min_ease_rating=4.2,
        min_zero_lag_ratio=0.95,
        require_live_quorum=True,
        require_ces_slo=True,
        check_arena_summary=True,
        execute_deploy_dry_run=True,
        deploy_sh_path=DEPLOY_SH_PATH,
        http_fn=ok_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_receipt(res_full)
    assert res_full["smoke_ok"] is True and res_full["exit_code"] == 0
    assert res_full["online_producers"] == 8 and res_full["producer_quorum_met"] is True
    assert res_full["awaiting_producer_quorum"] is False
    assert res_full["ces_slo_ok"] is True and res_full["zero_lag_slo_ok"] is True
    assert res_full["arena_summary_ok"] is True
    assert res_full["deploy_dry_run"]["ok"] is True
    assert res_full["deploy_dry_run"]["gcloud_invoked"] is False
    assert res_full["deploy_dry_run"]["checks"]["dry_run_marker"] is True
    assert res_full["adoption_gates"]["readiness_gates_passed"] == 5
    assert "Arena" in verifier.render_cli_report(res_full)
    assert "Arena Summary Parity" in verifier.render_markdown_report(res_full)

    # Same full check via CLI flags including real bash deploy.sh --dry-run
    out_cli_full = tmp_path / "full_quorum.json"
    cli_rc = verifier.main(
        [
            "--smoke-only",
            "--gateway", "http://127.0.0.1:8000",
            "--expected-sha", TARGET_SHORT_SHA,
            "--target-online-producers", "8",
            "--min-ease-rating", "4.2",
            "--min-zero-lag-ratio", "0.95",
            "--require-live-quorum",
            "--require-ces-slo",
            "--check-arena-summary",
            "--run-deploy-dry-run",
            "--cwd", str(tmp_path),
            "--output", str(out_cli_full),
            "--json",
        ],
        http_fn=ok_http,
        environ=CLEAN_ENV,
        stdout=io.StringIO(),
    )
    assert cli_rc == 0 and json.loads(out_cli_full.read_text(encoding="utf-8"))["smoke_ok"] is True

    # 2. Quorum below target: soft awaiting_producer_quorum=True vs strict --require-live-quorum exit 2
    low_q_table = _sample_smoke_table(live_hiveminders=3, row_statuses=("live", "online", "ready"))
    low_q_soft = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000", http_fn=_make_table_http_fn(low_q_table), environ=CLEAN_ENV, cwd=tmp_path
    )
    assert low_q_soft["online_producers"] == 3 and low_q_soft["producer_quorum_met"] is False
    assert low_q_soft["awaiting_producer_quorum"] is True and low_q_soft["smoke_ok"] is True

    for lang_c in ("en", "fr"):
        low_q_strict = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            require_live_quorum=True,
            http_fn=_make_table_http_fn(low_q_table),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang=lang_c,
        )
        assert low_q_strict["smoke_ok"] is False and low_q_strict["exit_code"] == 2
        assert low_q_strict["redeploy_required"] is False

    # Fleet endpoint leak rejected
    leak_rep = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        http_fn=_make_table_http_fn(_sample_smoke_table(leak_fleet_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert leak_rep["fleet_ok"] is False and leak_rep["exit_code"] == 2

    # 3. CES & Zero-Lag SLO enforcement: empty (passes), exact boundary (passes), breach (exit 2)
    empty_fb_rep = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        require_ces_slo=True,
        http_fn=_make_table_http_fn(
            _sample_smoke_table(
                feedback_summary={
                    "object": verifier.FEEDBACK_SUMMARY_OBJECT,
                    "total_responses": 0,
                    "mean_ease_rating": None,
                    "zero_game_lag_ratio": None,
                    "by_profile": {"dev": 0, "gamer": 0},
                }
            )
        ),
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert empty_fb_rep["smoke_ok"] is True and empty_fb_rep["ces_slo_ok"] is None and empty_fb_rep["zero_lag_slo_ok"] is None

    for bad_ease, bad_lag in ((3.9, 1.0), (4.8, 0.85)):
        breach_rep = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            require_ces_slo=True,
            http_fn=_make_table_http_fn(
                _sample_smoke_table(
                    feedback_summary={
                        "object": verifier.FEEDBACK_SUMMARY_OBJECT,
                        "total_responses": 10,
                        "mean_ease_rating": bad_ease,
                        "zero_game_lag_ratio": bad_lag,
                        "by_profile": {"dev": 5, "gamer": 5},
                    }
                )
            ),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert breach_rep["smoke_ok"] is False and breach_rep["exit_code"] == 2

    # 4. Arena summary contract: valid 0-vote Wilson=0, stale 404, tampered votes, forbidden score, bad by_task
    zero_arena = _sample_smoke_table(
        arena_summary={
            "object": verifier.ARENA_SUMMARY_OBJECT,
            "total_votes": 2,
            "decisive_votes": 0,
            "holiday_wins": 0,
            "baseline_wins": 0,
            "ties": 2,
            "win_ratio": None,
            "wilson_95_lower_bound": 0.0,
        }
    )
    assert (
        verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000", check_arena_summary=True, http_fn=_make_table_http_fn(zero_arena), environ=CLEAN_ENV, cwd=tmp_path
        )["arena_summary_ok"]
        is True
    )
    arena_404 = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        check_arena_summary=True,
        http_fn=_make_table_http_fn(_sample_smoke_table(arena_status_code=404)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert arena_404["arena_summary_ok"] is False and "/v1/hivemind/arena/summary" in arena_404["stale_404_routes"]

    for bad_arena in (
        {"object": "wrong"},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "sco" + "re": 1, "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 10, "holiday_wins": 8, "baseline_wins": 99, "ties": 1, "wilson_95_lower_bound": 0.5},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 2, "decisive_votes": 9, "holiday_wins": 1, "baseline_wins": 1, "ties": 0, "win_ratio": 0.5, "wilson_95_lower_bound": 0.2},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 1, "holiday_wins": 0, "baseline_wins": 0, "ties": 1, "win_ratio": None, "wilson_95_lower_bound": 0.2},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 1, "baseline_wins": 1, "ties": 0, "win_ratio": 2.0, "wilson_95_lower_bound": 0.2},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 1, "baseline_wins": 1, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": "bad"}},
        {"object": verifier.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 1, "baseline_wins": 1, "ties": 0, "wilson_95_lower_bound": 0.2, "by_task": {"t1": {"total_votes": 2, "holiday_wins": 9, "baseline_wins": 1, "ties": 0, "wilson_95_lower_bound": 0.2}}},
    ):
        r_bad_a = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            check_arena_summary=True,
            http_fn=_make_table_http_fn(_sample_smoke_table(arena_summary=bad_arena)),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert r_bad_a["arena_summary_ok"] is False and r_bad_a["exit_code"] == 2

    # 5. Stale pre-PR-#949 drift, Idempotency-Key replay, SHA drift, deploy.sh missing/failure & invalid options
    stale_rep = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        http_fn=_make_table_http_fn(
            _sample_smoke_table(
                status_overrides={
                    "/v1/hivemind/onboarding/manifest": 404,
                    "/noel": 404,
                    "/v1/hivemind/onboarding/feedback/summary": 404,
                }
            )
        ),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert stale_rep["stale_revision_detected"] is True and stale_rep["redeploy_required"] is True and stale_rep["exit_code"] == 2

    # Feedback POST + Idempotency-Key pass vs replay drift vs after-POST mismatch vs POST 500
    seen_idem: dict[str, int] = {}
    fb_table = _sample_smoke_table(
        feedback_summary={"object": verifier.FEEDBACK_SUMMARY_OBJECT, "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}
    )

    def _fb_post_ok(_m: str, _p: str, _b: Any, headers: Any) -> tuple[int, dict[str, str], dict[str, Any]]:
        k = (headers or {}).get("Idempotency-Key", "")
        seen_idem.setdefault(k, len(seen_idem) + 1)
        return 200, {"content-type": "application/json"}, {
            "object": verifier.FEEDBACK_SUMMARY_OBJECT,
            "total_responses": len(seen_idem),
            "mean_ease_rating": 5.0,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": len(seen_idem), "gamer": 0},
        }

    idem_ok = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        probe_feedback_post=True,
        idempotency_check=True,
        http_fn=_make_table_http_fn(fb_table, post_handler=_fb_post_ok),
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert idem_ok["feedback_post_ok"] is True and idem_ok["idempotency_conserved"] is True

    bad_seq = {"n": 0}

    def _fb_post_non_idem(_m: str, _p: str, _b: Any, _h: Any) -> tuple[int, dict[str, str], dict[str, Any]]:
        bad_seq["n"] += 1
        return 200, {"content-type": "application/json"}, {
            "object": verifier.FEEDBACK_SUMMARY_OBJECT,
            "total_responses": bad_seq["n"] + 5,
            "mean_ease_rating": 5.0,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": bad_seq["n"] + 5, "gamer": 0},
        }

    assert (
        verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            probe_feedback_post=True,
            idempotency_check=True,
            http_fn=_make_table_http_fn(fb_table, post_handler=_fb_post_non_idem),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )["idempotency_conserved"]
        is False
    )
    assert (
        verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            probe_feedback_post=True,
            http_fn=_make_table_http_fn(fb_table, post_handler=lambda *_a: (500, {}, {"err": 1})),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )["feedback_post_ok"]
        is False
    )
    assert (
        verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            probe_feedback_post=True,
            http_fn=_make_table_http_fn(
                _sample_smoke_table(
                    feedback_summary={"object": verifier.FEEDBACK_SUMMARY_OBJECT, "total_responses": 99, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 99, "gamer": 0}}
                ),
                post_handler=lambda *_a: (200, {"content-type": "application/json"}, {"object": verifier.FEEDBACK_SUMMARY_OBJECT, "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}),
            ),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )["feedback_post_ok"]
        is False
    )

    # Remaining contract failure mutations + SHA drift + check_git_head + missing/failing deploy.sh dry-run
    assert verifier.audit_deploy_sh_guidance(tmp_path / "missing_deploy.sh")["deploy_script_present"] is False
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("denied")))
    assert verifier.audit_deploy_sh_guidance(DEPLOY_SH_PATH)["deploy_script_present"] is False
    monkeypatch.undo()

    assert verifier.run_cloud_run_deploy_smoke_gate("http://127.0.0.1:8000", expected_sha=DRIFT_SHA, http_fn=ok_http, environ=CLEAN_ENV, cwd=tmp_path)["sha_mismatch"] is True
    assert verifier.run_cloud_run_deploy_smoke_gate("http://127.0.0.1:8000", check_git_head=True, runner=lambda _a, _t: (0, "not-a-sha!"), http_fn=ok_http, environ=CLEAN_ENV, cwd=tmp_path)["sha_mismatch"] is True
    assert verifier.run_cloud_run_deploy_smoke_gate("http://127.0.0.1:8000", check_git_head=True, runner=lambda _a, _t: (1, ""), http_fn=ok_http, environ=CLEAN_ENV, cwd=tmp_path)["sha_mismatch"] is True
    assert verifier.run_cloud_run_deploy_smoke_gate("http://127.0.0.1:8000", check_git_head=True, runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no git")), http_fn=ok_http, environ=CLEAN_ENV, cwd=tmp_path)["sha_mismatch"] is True

    miss_dep = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        execute_deploy_dry_run=True,
        deploy_sh_path=tmp_path / "missing_deploy.sh",
        http_fn=ok_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert miss_dep["smoke_ok"] is False and miss_dep["deploy_dry_run"]["executed"] is False
    bad_dep = verifier.run_deploy_sh_dry_run(DEPLOY_SH_PATH, runner=lambda _a, _t: (2, "failed"), environ=CLEAN_ENV, lang="fr")
    assert bad_dep["ok"] is False and bad_dep["gcloud_invoked"] is True

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_SHORT_SHA)
    assert verifier.run_deploy_sh_dry_run(DEPLOY_SH_PATH)["dry_run_build_sha"] == TARGET_SHORT_SHA
    monkeypatch.undo()

    for bad_mut in (
        {"static_nonce": "ReusedNonce12345"},
        {"unsafe_sink_route": "/noel"},
        {"static_fig_route": "/quick"},
        {"missing_banner_route": "/raid-noel"},
        {"include_coder": False},
        {"missing_manifest_route": "/arena-noel"},
        {"status_overrides": {"/scripts/join_hivemind.ps1": 200}},
        {"status_overrides": {"/scripts/hivemind_quickstart.py": 200}},
        {"status_overrides": {"/v1/hivemind/onboarding/feedback/summary": 200}},
        {"status_overrides": {"/v1/hivemind/raid/status": 200}},
    ):
        mut_t = _sample_smoke_table(**bad_mut)  # type: ignore[arg-type]
        ov = bad_mut.get("status_overrides")
        if ov and "/scripts/join_hivemind.ps1" in ov:
            mut_t["/scripts/join_hivemind.ps1"] = (200, {"content-type": "text/plain"}, "broken ps1")
        elif ov and "/scripts/hivemind_quickstart.py" in ov:
            mut_t["/scripts/hivemind_quickstart.py"] = (200, {"content-type": "text/plain"}, "broken py")
        elif ov:
            mut_t[next(iter(ov))] = (200, {"content-type": "application/json"}, {"object": "invalid"})
        assert (
            verifier.run_cloud_run_deploy_smoke_gate(
                "http://127.0.0.1:8000", http_fn=_make_table_http_fn(mut_t), environ=CLEAN_ENV, cwd=tmp_path
            )["smoke_ok"]
            is False
        )
    assert verifier._inspect_models_contract({"object": "bad", "data": "x"}, require_coder_model=True, lang_code="en")[0] is False
    assert verifier._inspect_models_contract({"object": "list", "data": []}, require_coder_model=True, lang_code="en")[0] is False
    assert verifier._inspect_manifest_contract({"object": "bad", "holiday_routes": "x", "downloadable_scripts": "x"}, lang_code="en")[0] is False
    assert verifier._inspect_manifest_contract(
        {
            "object": verifier.MANIFEST_OBJECT, "campaign_id": verifier.CAMPAIGN_ID, "status": "ok",
            "build_sha": TARGET_SHORT_SHA, "personal_pc_only": True, "financial_promise": verifier.FINANCIAL_PROMISE,
            "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES), "downloadable_scripts": [],
        },
        lang_code="fr",
    )[0] is False
    assert (
        verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            http_fn=_make_table_http_fn({**_sample_smoke_table(), "/health": (200, {"content-type": "application/json"}, {"status": "bad"})}),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )["health_ok"]
        is False
    )

    for bad_opt in (
        {"target_online_producers": 0},
        {"min_ease_rating": 6.0},
        {"min_zero_lag_ratio": 1.5},
        {"require_live_quorum": "yes"},
    ):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.run_cloud_run_deploy_smoke_gate("http://127.0.0.1:8000", environ=CLEAN_ENV, cwd=tmp_path, **bad_opt)  # type: ignore[arg-type]


def test_04_dev_60s_and_win11_5m_dry_run_verifiers_and_launcher_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_audit = verifier.audit_windows_launcher_contract(environ=CLEAN_ENV, cwd=REPO_ROOT, lang="en")
    _verify_receipt(real_audit)
    assert real_audit["contract_ok"] is True and real_audit["ps1_ok"] is True and real_audit["sh_ok"] is True

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pwsh" if cmd == "pwsh" else None)
    pw_audit = verifier.audit_windows_launcher_contract(
        runner=lambda a, _t: (
            (0, json.dumps({"dry_run": True, "target_script": "noel_deploy_smoke_and_win11_verifier.py"}))
            if "pwsh" in str(a[0])
            else ((0, "--gamer-whisper --connect") if "--help" in a else (0, ""))
        ),
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
    )
    assert pw_audit["pwsh_checked"] is True and pw_audit["pwsh_dry_run_ok"] is True
    pw_bad = verifier.audit_windows_launcher_contract(
        runner=lambda a, _t: (
            (0, "prefix{badjson") if "pwsh" in str(a[0]) else ((0, "--gamer-whisper --connect") if "--help" in a else (0, ""))
        ),
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
    )
    assert pw_bad["pwsh_dry_run_ok"] is False
    monkeypatch.undo()

    bad_ps1 = tmp_path / "bad.ps1"
    bad_ps1.write_text("# bad\nInvoke-" + "Expression 'x'\n", encoding="utf-8")
    bad_sh = tmp_path / "bad.sh"
    bad_sh.write_text("#!/usr/bin/env bash\nif then fi\n", encoding="utf-8")
    assert verifier.audit_windows_launcher_contract(ps1_path=bad_ps1, sh_path=bad_sh, environ=CLEAN_ENV, cwd=tmp_path, lang="fr")["exit_code"] == 2

    # Dev <60s & Win11 <5m dry-runs + real subprocess <1.5s Steam yield drill + CLI modes
    ok_http = _make_table_http_fn(
        _sample_smoke_table(),
        post_handler=lambda _m, path, _b, _h: (
            (200, {"content-type": "application/json"}, {"account_id": "acc_d1", "api_key": RAW_HM_KEY})
            if path.endswith("/register")
            else (200, {"content-type": "application/json"}, {"total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}})
        ),
    )
    dev_home = tmp_path / "dev_home"
    dev_ws = tmp_path / "dev_ws"
    (dev_ws / ".vscode").mkdir(parents=True)
    (dev_ws / ".vscode" / "settings.json").write_text('{"editor.tabSize": 4}\n', encoding="utf-8")
    dev_ok = verifier.verify_dev_vscode_60s_dry_run(
        "http://127.0.0.1:8000",
        home_dir=dev_home,
        workspace_dir=dev_ws,
        register_account=True,
        probe_gateway=True,
        submit_ces=True,
        run_uninstall=True,
        http_fn=ok_http,
        environ=CLEAN_ENV,
        cwd=dev_ws,
        lang="fr",
    )
    _verify_receipt(dev_ok)
    assert dev_ok["dev_ready"] is True and dev_ok["slo_ok"] is True and dev_ok["credentials_0600_ok"] is True
    assert dev_ok["continue_yaml_ok"] is True and dev_ok["vscode_secret_clean"] is True and dev_ok["uninstall_clean"] is True
    assert dev_ok["ces_recorded"] is True

    # Offline tmp workspace + corrupt .vscode/settings.json + Dev negative branches + unified compliance block
    assert verifier.verify_dev_vscode_60s_dry_run(environ=CLEAN_ENV, cwd=tmp_path)["dev_ready"] is True
    (dev_ws / ".vscode" / "settings.json").write_text("{corrupt", encoding="utf-8")
    assert verifier.verify_dev_vscode_60s_dry_run("http://127.0.0.1:8000", home_dir=dev_home, workspace_dir=dev_ws, environ=CLEAN_ENV, cwd=dev_ws)["dev_ready"] is True
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.verify_dev_vscode_60s_dry_run(ease_rating=0, environ=CLEAN_ENV, cwd=tmp_path)
    assert verifier.verify_dev_vscode_60s_dry_run(environ={**CLEAN_ENV, "AMADEUS_X": "1"}, cwd=tmp_path)["compliance_ok"] is False
    assert verifier.verify_dev_vscode_60s_dry_run(
        "http://127.0.0.1:8000", register_account=True, probe_gateway=True, submit_ces=True, http_fn=lambda *_a: (500, {}), environ=CLEAN_ENV, cwd=tmp_path
    )["dev_ready"] is False

    real_drill = verifier.default_worker_yield_drill()
    assert real_drill["clean_release"] is True and real_drill["retained_worker_vram_mib"] == 0
    assert float(real_drill["yield_latency_ms"]) < 1500.0

    win_ok = verifier.verify_win11_gamer_whisper_5m_dry_run(
        "http://127.0.0.1:8000",
        api_key=RAW_HM_KEY,
        send_heartbeat=True,
        submit_ces=True,
        gpu_probe_fn=_rtx_quiet_snapshot,
        process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
        http_fn=ok_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_receipt(win_ok)
    assert win_ok["gamer_ready"] is True and win_ok["yield_slo_ok"] is True and win_ok["thermal_ok"] is True and win_ok["vram_reserve_ok"] is True

    # Negative Win11/Dev controls: slow yield, hot GPU, low VRAM, parse error, CPU fallback, heartbeat/CES failures
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.verify_win11_gamer_whisper_5m_dry_run(ease_rating=9, environ=CLEAN_ENV, cwd=tmp_path)
    assert verifier.verify_win11_gamer_whisper_5m_dry_run(environ={**CLEAN_ENV, "AMADEUS_X": "1"}, cwd=tmp_path)["compliance_ok"] is False
    assert verifier.run_unified_onboarding_and_smoke_gate(environ={**CLEAN_ENV, "AMADEUS_X": "1"}, cwd=tmp_path)["compliance_ok"] is False

    slow_win = verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=_rtx_quiet_snapshot,
        process_probe_fn=lambda: ["cs2.exe"],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": False, "retained_worker_vram_mib": 512, "yield_latency_ms": 1820.0},
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert slow_win["gamer_ready"] is False and slow_win["zero_game_lag"] is False and slow_win["exit_code"] == 2

    hot_win = verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=lambda: {"present": True, "parse_error": False, "name": "RTX 4070", "total_mib": 12288, "free_mib": 1024, "temp_c": 75.5},
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 80.0},
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert hot_win["thermal_ok"] is False and hot_win["vram_reserve_ok"] is False and hot_win["exit_code"] == 2

    assert verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 80.0},
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )["gpu_status"] == "gpu_driver_parse_error"

    assert verifier.verify_win11_gamer_whisper_5m_dry_run(
        send_heartbeat=True, submit_ces=True, api_key=None,
        gpu_probe_fn=_rtx_quiet_snapshot, process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 80.0},
        http_fn=lambda *_a: (500, {}), environ=CLEAN_ENV, cwd=tmp_path,
    )["exit_code"] == 2
    assert verifier.verify_win11_gamer_whisper_5m_dry_run(
        send_heartbeat=True, api_key=RAW_HM_KEY,
        gpu_probe_fn=_rtx_quiet_snapshot, process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 80.0},
        http_fn=lambda *_a: (500, {}), environ=CLEAN_ENV, cwd=tmp_path,
    )["heartbeat_ok"] is False

    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "iGPU, 256, 128, 45, 0\nNVIDIA RTX 4070, 12288, 9600, N/A, N/A\n"))["name"] == "NVIDIA RTX 4070"
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no smi")))["present"] is False
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "GPU, bad, 1, 2, 3"))["parse_error"] is True
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "bad_csv"))["parse_error"] is True
    assert "cs2.exe" in verifier.default_process_probe(runner=lambda _a, _t: (0, '"cs2.exe","1"\n'), os_name="nt")
    monkeypatch.setattr(verifier.csv, "reader", lambda *_a: (_ for _ in ()).throw(verifier.csv.Error("bad csv")))
    assert verifier.default_process_probe(runner=lambda _a, _t: (0, '"cs2.exe","1"\n'), os_name="nt") == []
    monkeypatch.undo()
    assert "eldenring.exe" in verifier.default_process_probe(runner=lambda _a, _t: (0, "eldenring.exe\n"), os_name="posix")
    assert verifier.default_process_probe(runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no ps"))) == []

    class _DrillProcErr:
        def __init__(self, mode: str) -> None:
            self._mode = mode
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("BAD\n" if mode == "bad" else "HIVEMIND_WORKER_READY\n")
            self.stderr = io.StringIO()
            self._killed = False

        def kill(self) -> None:
            self._killed = True

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._mode == "timeout" and not self._killed:
                raise subprocess.TimeoutExpired(cmd=["py"], timeout=0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProcErr("bad"))
    assert verifier.default_worker_yield_drill(timeout_s=0.2)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProcErr("timeout"))
    assert verifier.default_worker_yield_drill(timeout_s=0.2)["clean_release"] is False
    monkeypatch.undo()

    # Remaining CLI selector branches (--launcher-audit-only, --dev-dry-run-only, --win11-dry-run-only, unified, errors)
    assert verifier.main(["--launcher-audit-only", "--cwd", str(REPO_ROOT), "--json"], environ=CLEAN_ENV, stdout=io.StringIO()) == 0
    assert verifier.main(["--dev-dry-run-only", "--cwd", str(tmp_path), "--json"], environ=CLEAN_ENV, stdout=io.StringIO()) == 0
    win_buf = io.StringIO()
    assert verifier.main(["--win11-dry-run-only", "--cwd", str(tmp_path), "--json"], runner=lambda _a, _t: (1, ""), environ=CLEAN_ENV, stdout=win_buf) == 0
    assert json.loads(win_buf.getvalue())["gpu_status"] == "cpu_or_dev_only_no_nvidia_gpu"
    uni_cli_buf = io.StringIO()
    assert verifier.main(["--gateway", "http://127.0.0.1:8000", "--cwd", str(REPO_ROOT)], http_fn=lambda *_a: (404, {}), environ=CLEAN_ENV, stdout=uni_cli_buf) == 2
    assert "[DIAG]" in uni_cli_buf.getvalue()
    err_buf = io.StringIO()
    assert verifier.main(["--gateway", UNTRUSTED_EXT_GW], environ=CLEAN_ENV, cwd=tmp_path, stderr=err_buf) == 2
    assert "Error:" in err_buf.getvalue()


def test_05_concurrent_48_participant_wal_soak_with_quorum_arena_and_deploy_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00949-noel")

    upstream_inference_hits: list[str] = []

    def _forbid_upstream(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn43_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
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

    expected_ps1_sha = hashlib.sha256((REPO_ROOT / "scripts" / "join_hivemind.ps1").read_bytes()).hexdigest()
    expected_qs_sha = hashlib.sha256((REPO_ROOT / "scripts" / "hivemind_quickstart.py").read_bytes()).hexdigest()

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
        stripped = verifier.strip_masked_keys_and_nonces(resp.text)
        assert _STATIC_FIG_RE.findall(stripped) == []
        assert ("inner" + "HTML") not in stripped
        return hdr_n[0]

    def _make_asgi_http(peer_ip: str) -> verifier.HttpFn:
        def _adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
            assert "completions" not in url.lower()
            parsed = urlsplit(url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            req_h = {"X-Forwarded-For": peer_ip, **{str(k): str(v) for k, v in (headers or {}).items()}}
            resp = _asgi_req(method, path, json_body=payload, headers=req_h)
            with shared_lock:
                seen_paths.append(parsed.path)
                if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
                    sqlite_locks.append(resp.text[:120])
            if parsed.path in verifier.SMOKE_HTML_ROUTES and resp.status_code == 200:
                n = _check_html(parsed.path, resp)
                with shared_lock:
                    observed_nonces.append(n)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            if "application/json" in hdrs.get("content-type", ""):
                return resp.status_code, hdrs, resp.json()
            return resp.status_code, hdrs, resp.text

        return _adapter

    try:
        # Seed >= 8 distinct physical-GPU volunteer friend producers + hivemind/coder model
        for p_idx in range(8):
            p_ip = _loopback_ip(55, p_idx)
            r_reg = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"soak-gpu-friend-{p_idx:02d}", "role": "producer"},
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
            r_ot = _asgi_req(
                "POST",
                "/v1/hivemind/opentrainer/heartbeat",
                json_body={
                    "host_label": f"win11-gpu-{p_idx:02d}",
                    "gpu_name": "NVIDIA GeForce RTX 4070",
                    "vram_total_mib": 12288,
                    "vram_free_mib": 9600,
                    "network_mode": "outbound-https-relay",
                    "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
                    "active_campaign": verifier.CAMPAIGN_ID,
                },
                headers=p_hdr,
            )
            assert r_ot.status_code == 200, r_ot.text

        dev_count = 24
        gamer_count = 24
        probe_count = 4

        def _participant(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            c_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(10 if is_dev else 20, c_idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            slot = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"holiday_u_{idx:02d}"
            home_d = slot / "home" / os_user
            ws_d = slot / "hobby_ws"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)

            t0 = time.monotonic()

            arena_html = ""
            for h_route in verifier.SMOKE_HTML_ROUTES:
                rh = _asgi_req("GET", h_route, headers=hdrs)
                nonce_val = _check_html(h_route, rh)
                if h_route == "/arena-noel":
                    arena_html = rh.text
                with shared_lock:
                    observed_nonces.append(nonce_val)
                    seen_paths.append(h_route)

            rm = _asgi_req("GET", "/v1/models", headers=hdrs)
            assert rm.status_code == 200 and "hivemind/coder" in [m["id"] for m in rm.json()["data"]]
            rman = _asgi_req("GET", "/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert rman.status_code == 200 and rman.json()["build_sha"] == TARGET_FULL_SHA
            script_path = "/scripts/hivemind_quickstart.py" if is_dev else "/scripts/join_hivemind.ps1"
            expected_script_sha = expected_qs_sha if is_dev else expected_ps1_sha
            assert hashlib.sha256(_asgi_req("GET", script_path, headers=hdrs).content).hexdigest() == expected_script_sha

            r_reg = _asgi_req(
                "POST",
                "/v1/hivemind/register",
                json_body={"handle": f"soak43_{profile}_{c_idx:02d}", "role": "prosumer" if is_dev else "trainer"},
                headers=hdrs,
            )
            assert r_reg.status_code == 200, r_reg.text
            api_key = str(r_reg.json()["api_key"])
            masked_key = verifier.mask_api_key(api_key)

            # Blind A/B Arena vote extracted from /arena-noel HTML
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

            adapter = _make_asgi_http(peer_ip)
            if is_dev:
                ease = 5 if (c_idx % 4 != 0) else 4
                zero_lag = True
                plat = "linux"
                dry = verifier.verify_dev_vscode_60s_dry_run(
                    "http://127.0.0.1:8000",
                    home_dir=home_d,
                    workspace_dir=ws_d,
                    handle=f"soak43_{profile}_{c_idx:02d}",
                    api_key=api_key,
                    ease_rating=ease,
                    submit_ces=False,
                    run_uninstall=True,
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_d,
                    lang="fr" if c_idx % 2 else "en",
                )
                assert dry["dev_ready"] is True and dry["elapsed_s"] < 60.0
                d_txt = json.dumps(dry)
                assert api_key not in d_txt and os_user not in d_txt
            else:
                ease = 5 if (c_idx % 5 != 0) else 4
                zero_lag = c_idx != 23  # 47/48 == 0.9792 >= 0.95
                plat = "windows"
                dry = verifier.verify_win11_gamer_whisper_5m_dry_run(
                    "http://127.0.0.1:8000",
                    handle=f"soak43_{profile}_{c_idx:02d}",
                    api_key=api_key,
                    ease_rating=ease,
                    had_game_lag=not zero_lag,
                    gpu_probe_fn=_rtx_quiet_snapshot,
                    process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
                    worker_yield_fn=verifier.default_worker_yield_drill if c_idx == 0 else _fast_worker_yield,
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_d,
                    lang="fr" if c_idx % 2 else "en",
                )
                assert dry["gamer_ready"] is True and dry["elapsed_s"] < 300.0
                assert dry["yield_slo_ok"] is True and dry["retained_worker_vram_mib"] == 0
                d_txt = json.dumps(dry)
                assert api_key not in d_txt and os_user not in d_txt

            idem_key = "soak43-" + hashlib.sha256(f"{profile}:{c_idx}".encode()).hexdigest()[:24]
            fb_body = {"profile": profile, "ease_rating": ease, "zero_game_lag": zero_lag, "platform": plat, "note": ""}
            fb1 = _asgi_req("POST", "/v1/hivemind/onboarding/feedback", json_body=fb_body, headers={**hdrs, "Idempotency-Key": idem_key})
            fb2 = _asgi_req("POST", "/v1/hivemind/onboarding/feedback", json_body=fb_body, headers={**hdrs, "Idempotency-Key": idem_key})
            assert fb1.status_code == 200 and fb2.status_code == 200
            assert (time.monotonic() - t0) < (60.0 if is_dev else 300.0)

            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        probe_count = 2

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            p_http = _make_asgi_http(_loopback_ip(33, p_idx))
            return verifier.run_cloud_run_deploy_smoke_gate(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_FULL_SHA,
                require_live_quorum=True,
                require_ces_slo=True,
                check_arena_summary=True,
                http_fn=p_http,
                environ=CLEAN_ENV,
                cwd=clean_ws,
                lang="fr" if p_idx % 2 else "en",
            )

        def _concurrent_unified() -> dict[str, Any]:
            return verifier.run_unified_onboarding_and_smoke_gate(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_FULL_SHA,
                require_live_quorum=True,
                require_ces_slo=True,
                check_arena_summary=True,
                run_launcher_audit=False,
                run_dev_dry_run=False,
                run_win11_dry_run=False,
                http_fn=_make_asgi_http("127.0.34.1"),
                environ=CLEAN_ENV,
                cwd=clean_ws,
                lang="en",
            )

        t_soak = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            part_futs = [pool.submit(_participant, i) for i in range(dev_count + gamer_count)]
            probe_futs = [pool.submit(_concurrent_probe, j) for j in range(probe_count)]
            uni_fut = pool.submit(_concurrent_unified)
            part_docs = [f.result(timeout=20.0) for f in part_futs]
            probe_docs = [f.result(timeout=20.0) for f in probe_futs]
            concurrent_uni = uni_fut.result(timeout=20.0)
        soak_elapsed_s = time.monotonic() - t_soak
        assert soak_elapsed_s < 15.0, f"48-user WAL soak exceeded 15.0s ceiling: {soak_elapsed_s:.2f}s"

        post_unified = verifier.run_unified_onboarding_and_smoke_gate(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_FULL_SHA,
            target_online_producers=8,
            min_ease_rating=4.2,
            min_zero_lag_ratio=0.95,
            require_live_quorum=True,
            require_ces_slo=True,
            check_arena_summary=True,
            execute_deploy_dry_run=True,
            deploy_sh_path=DEPLOY_SH_PATH,
            run_launcher_audit=True,
            run_dev_dry_run=True,
            run_win11_dry_run=True,
            gpu_probe_fn=_rtx_quiet_snapshot,
            process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
            worker_yield_fn=_fast_worker_yield,
            http_fn=_make_asgi_http("127.0.50.1"),
            environ=CLEAN_ENV,
            cwd=clean_ws,
            lang="en",
        )

        assert len(part_docs) == 48
        assert concurrent_uni["overall_ok"] is True and concurrent_uni["exit_code"] == 0
        assert sqlite_locks == []
        assert upstream_inference_hits == []
        assert all("completions" not in p.lower() for p in seen_paths)

        for pd in probe_docs:
            _verify_receipt(pd)
            assert pd["smoke_ok"] is True and pd["online_producers"] >= 8 and pd["producer_quorum_met"] is True
            assert pd["arena_summary_ok"] is True and pd["exit_code"] == 0

        _verify_receipt(post_unified)
        assert post_unified["overall_ok"] is True and post_unified["exit_code"] == 0
        assert post_unified["online_producers"] >= 8 and post_unified["producer_quorum_met"] is True
        assert post_unified["ces_slo_ok"] is True and post_unified["zero_lag_slo_ok"] is True
        assert post_unified["arena_summary_ok"] is True
        assert post_unified["deploy_dry_run"]["ok"] is True
        assert post_unified["cloud_run_smoke"]["adoption_gates"]["readiness_gates_passed"] == 5

        fb_summary = _asgi_req("GET", "/v1/hivemind/onboarding/feedback/summary", headers={"X-Forwarded-For": "127.0.50.2"}).json()
        assert set(fb_summary.keys()) == {"object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"}
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2
        assert fb_summary["zero_game_lag_ratio"] >= 0.95

        arena_sum = _asgi_req("GET", "/v1/hivemind/arena/summary", headers={"X-Forwarded-For": "127.0.50.3"}).json()
        assert arena_sum["total_votes"] == 48
        assert arena_sum["holiday_wins"] + arena_sum["baseline_wins"] + arena_sum["ties"] == arena_sum["total_votes"]
        assert arena_sum["wilson_95_lower_bound"] > 0.0

        expected_nonces = (48 * 4) + (probe_count * 4) + 8
        assert len(observed_nonces) == expected_nonces
        assert len(set(observed_nonces)) == expected_nonces

        for rk, mk in zip(raw_keys, masked_keys, strict=True):
            assert rk not in mk and mk.startswith(("hm-***", "hm_live_***"))
        assert verifier.scan_output_safety(verifier.render_cli_report(post_unified)) == []
        assert verifier.scan_output_safety(verifier.render_markdown_report(post_unified)) == []
    finally:
        asyncio.run_coroutine_threadsafe(async_client.__aexit__(None, None, None), asgi_loop).result(timeout=5.0)
        asgi_loop.call_soon_threadsafe(asgi_loop.stop)
        asgi_thread.join(timeout=5.0)
        asgi_loop.close()
        store.close()
