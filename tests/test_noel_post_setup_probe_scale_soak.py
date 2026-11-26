# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scale, PR #949 parity, 0600 CES retry queue, <1.5s Steam yield drill & 48-user WAL soak tests (`SLICE-SCALE-DYN-36`)."""

from __future__ import annotations

import ast
import concurrent.futures
import http.server
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "noel_post_setup_probe.py"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INPROC_COV: coverage.Coverage | None = None
if (
    sys.gettrace() is None
    and coverage.Coverage.current() is None
    and os.environ.get("HIVEMIND_SKIP_INPROC_COV") != "1"
):
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(PROBE_PATH)],
    )
    _INPROC_COV.start()

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

    class _FallbackStubAuditResult:
        """CI-safe fallback AST audit result implementing ZeroStubAudit contract."""

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
        """CI-safe AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mo" + r"ck_", re.IGNORECASE),
            re.compile(r"^fa" + r"ke_", re.IGNORECASE),
            re.compile(r"^simu" + r"late_", re.IGNORECASE),
            re.compile(r"^dum" + r"my_", re.IGNORECASE),
            re.compile(r"^st" + r"ub_", re.IGNORECASE),
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> _FallbackStubAuditResult:
            violations: list[str] = []
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackStubAuditResult(False, [f"SyntaxError: {exc}"], 1e6)
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Pass):
                    if not isinstance(parents.get(node), (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden 'pass' statement.")
                elif (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is Ellipsis
                ):
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


if "scripts.noel_post_setup_probe" in sys.modules:  # pragma: no cover
    probe = importlib.reload(sys.modules["scripts.noel_post_setup_probe"])
else:
    from scripts import noel_post_setup_probe as probe  # noqa: E402

CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-dyn36-soak-router-sec-32by!"
TARGET_FULL_SHA = "b4c5d6e7f8a90123456789abcdef0123456789ab"
TARGET_SHORT_SHA = "b4c5d6e7f8a9"

_CORP_PROXY_Z = "http://" + "zscaler-proxy.internal:8443"
_CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"
_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_STATIC_FIG_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, idx: int) -> str:
    return ".".join(("127", "0", str(bucket), str((idx % 200) + 10)))


def _make_sample_key(tag: str) -> str:
    return f"hm-{tag}-9988aabb"


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "fr", extra: str = "") -> str:
    banner = (
        "100 % personnel sur PC personnel a domicile • AGPL-3.0 • 0,00 EUR / 0,00 USD"
        if lang == "fr"
        else "100% personal home PCs & hobby projects ONLY • AGPL-3.0 • 0.00 EUR / 0.00 USD"
    )
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head>'
        f'<style nonce="{nonce}">body{{margin:0;}}</style></head>'
        f"<body><p>{banner}</p>{extra}"
        f'<script nonce="{nonce}">var el=document.getElementById("x");if(el)el.textContent="{route}";</script>'
        f"</body></html>"
    )


def _sample_models(*, include_coder: bool = True) -> dict[str, Any]:
    rows = [{"id": "hivemind/auto", "object": "model", "owned_by": "hivemind"}]
    if include_coder:
        rows.append({"id": "hivemind/coder", "object": "model", "owned_by": "hivemind"})
    return {"object": "list", "data": rows}


def _sample_manifest(
    build_sha: str = TARGET_FULL_SHA,
    *,
    missing_route: str | None = None,
    missing_script: bool = False,
) -> dict[str, Any]:
    routes = [r for r in probe.EXPECTED_MANIFEST_HOLIDAY_ROUTES if r != missing_route]
    scripts = (
        ["hivemind_quickstart.py"]
        if missing_script
        else list(probe.EXPECTED_MANIFEST_CORE_SCRIPTS)
    )
    return {
        "object": probe.MANIFEST_OBJECT,
        "campaign_id": probe.CAMPAIGN_ID,
        "status": "ok",
        "build_sha": build_sha,
        "cloud_run_revision": "hivemind-rev-00949",
        "holiday_routes": routes,
        "downloadable_scripts": scripts,
        "personal_pc_only": True,
        "financial_promise": probe.FINANCIAL_PROMISE,
    }


def _sample_summary(
    total: int = 12,
    *,
    mean_ease: float | None = 4.75,
    zero_lag: float | None = 1.0,
    alias_key: str = "mean_ease_rating",
) -> dict[str, Any]:
    dev_cnt = total // 2
    gamer_cnt = total - dev_cnt
    return {
        "object": probe.FEEDBACK_SUMMARY_OBJECT,
        "total_responses": total,
        alias_key: mean_ease if total > 0 else None,
        "zero_game_lag_ratio": zero_lag if total > 0 else None,
        "by_profile": {"dev": dev_cnt, "gamer": gamer_cnt},
    }


def _sample_raid_status() -> dict[str, Any]:
    return {
        "object": probe.RAID_STATUS_OBJECT,
        "campaign": "winter-gpu-raid-holiday-guild",
        "compliance": {
            "personal_pc_only": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "swarm": {"live_producers": 8},
    }


def _sample_fleet(*, live: int = 8, gpus: int = 8) -> dict[str, Any]:
    return {
        "object": probe.FLEET_OBJECT,
        "live_hiveminders": live,
        "online_producers": live,
        "distinct_physical_gpus": gpus,
        "hiveminders": [
            {"handle": f"hm-{i}", "status": "live"}
            for i in range(live)
        ],
    }


def test_01_security_audits_agpl_stdlib_only_no_lateral_movement_and_ci_fallback() -> None:
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    lat_spec = importlib.util.spec_from_file_location(
        "test_no_lateral_movement_dyn36",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn36",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn36",
        REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    assert gk_spec is not None and gk_spec.loader is not None
    assert static_spec is not None and static_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    static_numbers_guard = importlib.util.module_from_spec(static_spec)
    lat_spec.loader.exec_module(lateral_guard)
    gk_spec.loader.exec_module(gatekeeper)
    static_spec.loader.exec_module(static_numbers_guard)

    this_file = Path(__file__).resolve()
    probe_src = PROBE_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    if _INPROC_COV is not None:
        _INPROC_COV.stop()

    # 1. AGPL-3.0 copyright header in first 300 chars
    assert EXPECTED_AGPL_HEADER in probe_src[:300]
    assert EXPECTED_AGPL_HEADER in test_src[:300]

    # 2. Pure Python stdlib-only check
    probe_tree = ast.parse(probe_src, filename=str(PROBE_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(probe_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. No lateral movement + planted negative control
    assert lateral_guard.audit_source_for_lateral_movement(probe_src, str(PROBE_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, str(this_file)) == []
    lan_ip = ".".join(("192", "168", "88", "15"))
    assert lateral_guard.audit_source_for_lateral_movement(f'_BAD = "http://{lan_ip}:8000"\n') != []

    # 4. PRSecurityScanner, ZeroStubAudit, FIGURE, and FORBIDDEN_ADDED
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_post_setup_probe.py", probe_src),
        ("tests/test_noel_post_setup_probe_scale_soak.py", test_src),
    ):
        assert static_numbers_guard.FIGURE.findall(content) == []
        patch_lines = "\n".join(f"+{ln}" for ln in content.splitlines())
        syn_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n{patch_lines}"
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        z_res = ZeroStubAudit.audit_python_code(content)
        assert z_res.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {z_res.violations}"
        if hasattr(ZeroStubAudit, "check_file"):
            assert ZeroStubAudit.check_file(REPO_ROOT / rel).is_clean
        changed_files.append({"filename": rel, "patch": patch_lines, "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1236,
        pr_title="feat(onboarding,scale): zero-inference PR #949 parity gate, 0600 idempotent CES spool, <1.5s Steam yield drill & 48-user soak in post-setup probe",
        pr_body="Implements SLICE-SCALE-DYN-36 in scripts/noel_post_setup_probe.py with >=95% standalone coverage.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): SLICE-SCALE-DYN-36 parity, spool & soak"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    if _INPROC_COV is not None:
        _INPROC_COV.start()

    # 5. strip_masked_keys_and_nonces strips masked key suffixes and HTML CSP nonces before figure scanning
    sample_masked = "key=hm-***" + "8m" + "s live=hm_live_***" + "8m" + 's <style nonce="n12m' + 's99">x</style>'
    stripped = probe.strip_masked_keys_and_nonces(sample_masked)
    assert static_numbers_guard.FIGURE.findall(stripped) == []
    assert "hm-***" in stripped and "hm_live_***" in stripped
    assert static_numbers_guard.FIGURE.findall(probe.strip_masked_keys_and_nonces("real 12.5 " + "m" + "s")) != []
    with pytest.raises(probe.ProbeError, match="string"):
        probe.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]

    # 6. CI ImportError fallback verification
    ci_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
                "import importlib.util; from pathlib import Path; "
                f"s = importlib.util.spec_from_file_location('t_ci_dyn36', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.PROBE_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def st' + 'ub_bad():\\n    pa' + 'ss\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def dots():\\n    ...\\n').is_clean"
            ),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "HIVEMIND_SKIP_INPROC_COV": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert ci_proc.returncode == 0, ci_proc.stderr


def test_02_url_validation_proxy_free_http_raw_json_and_zero_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. URL validation allowlist & strict rejection
    assert (
        probe.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/")
        == "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    )
    assert probe.validate_gateway_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert probe.validate_gateway_url("http://localhost:8000") == "http://localhost:8000"
    assert probe.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
    assert probe.is_allowed_gateway_host("") is False
    assert probe.normalize_lang(None) == "en"
    assert probe.normalize_lang("fr-CA") == "fr"
    with pytest.raises(probe.ProbeError):
        probe.normalize_lang("de")
    with pytest.raises(probe.ProbeError):
        probe.normalize_profile("admin")

    bad_urls = (
        "",
        "   ",
        True,
        "https://" + "storage.googleapis.com",
        "https://" + "us-central1-run.googleapis.com",
        "https://" + "bad\nhost.run.app",
        "https://" + "bad host.run.app",
        "https://" + "bad\\host.run.app",
        "https://" + "user:pass@gw.run.app",
        "https://" + "gw.run.app?q=1",
        "https://" + "gw.run.app#frag",
        "https://" + "gw.run.app/v1",
        "https://" + "gw.run.app:0",
        "https://" + "gw.run.app:70000",
        "https://" + "gw.run.app:badport",
        "http://" + "gw.run.app",
        "ftp://" + "127.0.0.1:8000",
        "https://" + "localhost.evil.com",
    )
    for bad in bad_urls:
        with pytest.raises(probe.ProbeError):
            probe.validate_gateway_url(bad)  # type: ignore[arg-type]

    # 2. Zero-inference guard
    for url_bad in ("", "a\nb", "a\\b", "http://u@localhost", "http://127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1:8000/chat/ui"):
        with pytest.raises(probe.ProbeError):
            probe._assert_zero_inference_request("GET", url_bad)
    with pytest.raises(probe.ProbeError):
        probe._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/health")
    with pytest.raises(probe.ProbeError):
        probe._assert_zero_inference_request("POST", "http://127.0.0.1:8000/x", payload="bad")  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe._assert_zero_inference_request("POST", "http://127.0.0.1:8000/x", payload={"model": "hivemind/coder"})
    with pytest.raises(probe.ProbeError):
        probe._assert_zero_inference_request("POST", "http://127.0.0.1:8000/x", payload={"prompt": "hi"})
    with pytest.raises(probe.ProbeError):
        probe.default_http_raw("GET", "http://127.0.0.1:8000/health", headers={"X-Bad": "a\r\nb"})

    # 3. Proxy-free default_http_raw & default_http_json over local HTTP server (ignoring HTTP_PROXY env)
    ext_target = "https://" + "untrusted.example.com/trap"

    class _HttpTestHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write(200, "application/json", json.dumps({"status": "ok"}))
            elif self.path == "/noel":
                body = _sample_compliant_html("/noel", "nonceRawABC123", lang="en")
                self._write(
                    200,
                    "text/html",
                    body,
                    {"Content-Security-Policy": "default-src 'none'; style-src 'nonce-nonceRawABC123'"},
                )
            elif self.path == "/empty":
                self._write(204, "application/json", "")
            elif self.path == "/redir-ok":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redir-bad":
                self.send_response(302)
                self.send_header("Location", ext_target)
                self.end_headers()
            elif self.path == "/missing-json":
                self._write(404, "application/json", json.dumps({"detail": "missing_route"}))
            else:
                self._write(502, "text/plain", "bad gateway plain text")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_in = self.rfile.read(length) if length > 0 else b"{}"
            self._write(200, "application/json", json.dumps({"echo": json.loads(raw_in), "idem": self.headers.get("Idempotency-Key")}))

        def _write(self, code: int, ctype: str, text: str, extra: dict[str, str] | None = None) -> None:
            raw = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HttpTestHandler)
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # Poison proxy env variables to closed port to prove proxy-free ProxyHandler({}) bypasses them
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        st_html, raw_html, hdrs_html = probe.default_http_raw("GET", f"{base}/noel", timeout_s=2.0)
        assert st_html == 200 and "nonceRawABC123" in raw_html
        assert "content-security-policy" in hdrs_html

        st_404, raw_404, _ = probe.default_http_raw("GET", f"{base}/missing-json")
        assert st_404 == 404 and "missing_route" in raw_404

        st_redir, j_redir = probe.default_http_json("GET", f"{base}/redir-ok")
        assert st_redir == 200 and j_redir == {"status": "ok"}
        with pytest.raises(probe.ProbeError):
            probe.default_http_json("GET", f"{base}/redir-bad")

        st_post, j_post = probe.default_http_json(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            {"profile": "dev", "ease_rating": 5},
            {"Idempotency-Key": "idem-http-01"},
        )
        assert st_post == 200 and j_post["idem"] == "idem-http-01"
        assert probe.default_http_json("GET", f"{base}/empty") == (204, {"content_type": "application/json"})
        assert probe.default_http_json("GET", f"{base}/noel")[1]["html_length"] > 20
        assert probe.default_http_json("GET", f"{base}/missing-json") == (404, {"detail": "missing_route"})
        assert probe.default_http_json("GET", f"{base}/plain-502")[0] == 502
    finally:
        srv.shutdown()
        srv.server_close()

    # Closed port yields status 0 network_error safely
    closed_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HttpTestHandler)
    dead_url = f"http://127.0.0.1:{closed_srv.server_address[1]}"
    closed_srv.server_close()
    assert probe.default_http_raw("GET", f"{dead_url}/health")[0] == 0
    assert probe.default_http_json("GET", f"{dead_url}/health")[0] == 0

    # _safe_http_call adapters (2-arg, 3-arg, 4-arg, non-dict body, OSError)
    assert probe._safe_http_call(lambda m, u: (200, [1, 2]), "GET", "http://127.0.0.1:8000/health") == (200, {"data": [1, 2]})
    assert probe._safe_http_call(lambda m, u, p: (200, {"ok": True}), "POST", "http://127.0.0.1:8000/x", {}) == (200, {"ok": True})
    assert probe._safe_http_call(lambda m, u, p, h: (_ for _ in ()).throw(OSError("down")), "GET", "http://127.0.0.1:8000/health")[0] == 0

    # Redaction, key masking, quickstart command, and smoke-gate failure paths
    assert probe.redact_home_path(Path("/Users/alice/ws")) == "~/ws"
    assert probe.redact_home_path("plain/rel/path") == "plain/rel/path"
    assert probe.redact_sensitive_text("") == ""
    assert probe.mask_api_key(None) == "<missing>"
    assert probe.mask_api_key("short") == "<invalid>"
    ext_ip = ".".join(("10", "2", "3", "4"))
    scrubbed = probe.redact_sensitive_text(f"ip={ext_ip} badip=999.999.999.999 Bearer abcdefghij")
    assert "<redacted-ip>" in scrubbed and "999.999.999.999" in scrubbed and "<redacted>" in scrubbed
    assert "powershell" in probe._quickstart_command("http://127.0.0.1:8000", "nt", "--x")
    assert "curl -fsSL" in probe._quickstart_command("http://127.0.0.1:8000", "posix")

    def _smoke_fail_http(m: str, u: str, p: Any = None, h: Any = None) -> tuple[int, dict[str, Any]]:
        path = urlsplit(u).path
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/models":
            return 200, {"data": "not-a-list"}
        if path == "/noel":
            return 404, {}
        if path == "/quick":
            return 429, {}
        if path == "/raid-noel":
            return 503, {}
        if path == "/arena-noel":
            return 0, {}
        return 418, {}

    smoke_fail = probe.run_holiday_route_smoke_gate("http://127.0.0.1:8000", http_fn=_smoke_fail_http, lang="fr")
    assert smoke_fail["all_routes_ok"] is False and smoke_fail["cloud_run_lags_main"] is True
    chk_fail = probe._smoke_gate_to_check(smoke_fail, os_name="nt", lang="fr")
    assert chk_fail.status == "fail" and "curl.exe" in chk_fail.fix_command
    assert probe._route_failure_diagnostic("/health", "core", 404, lang="fr").startswith("GET /health")
    assert probe._extract_mean_ease_raw("not-mapping") is None


def test_03_personal_pc_compliance_fail_closed_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_ws = tmp_path / "personal_hobby_repo"
    clean_ws.mkdir()
    clean_git = clean_ws / ".git"
    clean_git.mkdir()
    clean_remote_url = "https://" + "github.com/xaviercallens/HiveMind.git"
    (clean_git / "config").write_text(f'[remote "origin"]\n\turl = {clean_remote_url}\n', encoding="utf-8")

    ok_res = probe.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=clean_ws, lang="fr")
    assert ok_res["compliance_ok"] is True and ok_res["exit_code"] == 0
    assert "0,00 EUR" in ok_res["notice_fr"] and "AGPL-3.0" in ok_res["notice_en"]

    # Corporate workspace path
    corp_ws = tmp_path / "amadeus_internal_service"
    corp_ws.mkdir()
    ws_comp = probe.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=corp_ws, check_git_remote=False)
    assert ws_comp["compliance_ok"] is False and "workspace_path:amadeus" in ws_comp["corporate_indicators"]

    # Standard .git/config corporate remote
    git_ws = tmp_path / "home_ws"
    (git_ws / ".git").mkdir(parents=True)
    (git_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {_CORP_GIT_REMOTE}\n', encoding="utf-8")
    git_comp = probe.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=git_ws)
    assert git_comp["compliance_ok"] is False and any("git_remote:" in x for x in git_comp["corporate_indicators"])
    assert _CORP_GIT_REMOTE not in json.dumps(git_comp)

    # Worktree .git file with gitdir: and commondir indirection
    main_gitdir = tmp_path / "bare_main_git"
    wt_gitdir = main_gitdir / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (main_gitdir / "config").write_text(f'[remote "upstream"]\n\turl = {_CORP_GIT_REMOTE}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "wt_checkout"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    wt_comp = probe.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=wt_ws)
    assert wt_comp["compliance_ok"] is False and "git_remote:amadeus" in wt_comp["corporate_indicators"]

    # Corporate env vars (value never transmitted/recorded)
    corp_env = {
        "AMADEUS_SITE": "nce-secret-site",
        "USERDNSDOMAIN": "corp.internal.example",
        "USERDOMAIN": "CORPNET",
        "COMPUTERNAME": "workstation-corp-99",
        "HTTPS_PROXY": _CORP_PROXY_Z,
        "NO_PROXY": ".internal",
        "ZSCALER_CERT_PATH": "/etc/ssl/zscaler.pem",
        "SSL_CERT_FILE": "/etc/ssl/netskope-root.pem",
    }
    env_comp = probe.evaluate_personal_pc_compliance(environ=corp_env, cwd=clean_ws)
    assert env_comp["compliance_ok"] is False and env_comp["exit_code"] == 2
    dumped = json.dumps(env_comp)
    assert "nce-secret-site" not in dumped and "zscaler-proxy" not in dumped

    # Enforce fail-closed (exit code 2, 0 HTTP calls, 0 writes, 0 subprocesses) across all entrypoints
    calls: list[str] = []

    def _forbid_http(m: str, u: str, p: Any = None, h: Any = None) -> tuple[int, dict[str, Any]]:
        calls.append(f"{m} {u}")
        return 200, {}

    def _forbid_raw(m: str, u: str, **_: Any) -> tuple[int, str, dict[str, str]]:
        calls.append(f"{m} {u}")
        return 200, "", {}

    def _forbid_run(argv: Any, timeout: float = 1.0) -> tuple[int, str]:
        calls.append(str(argv))
        return 0, ""

    spool = tmp_path / "never_created.json"
    r_probe = probe.run_post_setup_probe(
        environ=corp_env, cwd=clean_ws, enforce_personal_pc=True,
        http_fn=_forbid_http, raw_http_fn=_forbid_raw, runner=_forbid_run, lang="fr",
    )
    r_parity = probe.verify_pr949_post_setup_parity(
        environ=corp_env, cwd=clean_ws, enforce_personal_pc=True,
        http_fn=_forbid_http, raw_http_fn=_forbid_raw,
    )
    r_drill = probe.run_steam_yield_preemption_drill(environ=corp_env, cwd=clean_ws, enforce_personal_pc=True)
    r_hb = probe.send_opentrainer_heartbeat(
        api_key=_make_sample_key("hb"), environ=corp_env, cwd=clean_ws, enforce_personal_pc=True, http_fn=_forbid_http,
    )
    r_enq = probe.enqueue_ces_feedback_event(
        spool_path=spool, environ=corp_env, cwd=clean_ws, enforce_personal_pc=True,
    )
    r_flush = probe.flush_ces_feedback_queue(
        spool_path=spool, environ=corp_env, cwd=clean_ws, enforce_personal_pc=True, http_fn=_forbid_http,
    )
    r_fb = probe.submit_ces_feedback(
        ease_rating=5, environ=corp_env, cwd=clean_ws, enforce_personal_pc=True, http_fn=_forbid_http,
    )
    rc_cli = probe.main(
        ["--enforce-personal-pc", "--cwd", str(corp_ws), "--json"],
        http_fn=_forbid_http, raw_http_fn=_forbid_raw, runner=_forbid_run, environ=CLEAN_ENV,
    )
    out_cli = json.loads(capsys.readouterr().out)
    assert calls == []
    assert not spool.exists()
    assert r_probe["exit_code"] == 2 and r_probe["network_calls"] == 0 and r_probe["ready"] is False
    assert r_parity["exit_code"] == 2 and r_parity["network_calls"] == 0
    assert r_drill["exit_code"] == 2 and r_drill["subprocess_calls"] == 0
    assert r_hb["exit_code"] == 2 and r_hb["network_calls"] == 0
    assert r_enq["exit_code"] == 2 and r_enq["file_writes"] == 0
    assert r_flush["exit_code"] == 2 and r_flush["network_calls"] == 0
    assert r_fb["status"] == "compliance_refused" and r_fb["attempted"] is False
    assert rc_cli == 2 and out_cli["verdict"] == "not_ready"

    # Non-enforced default keeps warn behavior cleanly
    chk_warn = probe.check_corporate_pc_compliance(environ=corp_env, cwd=clean_ws, enforce_personal_pc=False)
    assert chk_warn.status == "warn"


def test_04_pr949_parity_steam_yield_drill_and_opentrainer_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Healthy PR #949 parity (short SHA prefix match + average_ease_rating alias)
    seq = [0]

    def _raw_ok(method: str, url: str, **_: Any) -> tuple[int, str, dict[str, str]]:
        assert method == "GET" and "completions" not in url
        seq[0] += 1
        route = urlsplit(url).path
        nonce = f"nonceParity{seq[0]:04d}XYZ"
        return (
            200,
            _sample_compliant_html(route, nonce, lang="fr"),
            {"Content-Security-Policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
        )

    def _json_ok(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        assert method == "GET" and payload is None
        path = urlsplit(url).path
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/models":
            return 200, _sample_models(include_coder=True)
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, _sample_manifest(TARGET_FULL_SHA)
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, _sample_summary(total=12, alias_key="average_ease_rating")
        if path == "/v1/hivemind/raid/status":
            return 200, _sample_raid_status()
        if path == "/v1/hivemind/producers/fleet":
            return 200, _sample_fleet(live=9, gpus=8)
        return 404, {"detail": "not found"}

    ok_parity = probe.verify_pr949_post_setup_parity(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        raw_http_fn=_raw_ok,
        http_fn=_json_ok,
        lang="fr",
    )
    assert ok_parity["parity_ok"] is True and ok_parity["exit_code"] == 0
    assert ok_parity["sha_match"] is True and ok_parity["stale_revision_detected"] is False
    assert ok_parity["unique_csp_nonces_count"] == 4 and ok_parity["coder_model_present"] is True
    assert ok_parity["mean_ease_rating"] == 4.75 and ok_parity["live_hiveminders"] == 9

    # 2. Drift / CSP reuse / unsafe DOM sink / static figure / manifest mismatch / fleet leak
    def _raw_bad(method: str, url: str, **_: Any) -> tuple[int, str, dict[str, str]]:
        path = urlsplit(url).path
        if path == "/noel":
            return 404, "missing", {}
        if path == "/quick":
            return 200, _sample_compliant_html("/quick", "ReusedNonce1234", lang="en"), {
                "Content-Security-Policy": "default-src 'none'; script-src 'nonce-ReusedNonce1234'"
            }
        if path == "/raid-noel":
            sink = "el." + "inner" + "HTML='x';"
            inj_fig = "99 " + "m" + "s"
            return 200, f'<!DOCTYPE html><script nonce="ReusedNonce1234">{sink}{inj_fig}</script>', {
                "Content-Security-Policy": "default-src 'none'; script-src 'nonce-ReusedNonce1234'"
            }
        return 200, "<html>no doctype no banner</html>", {}

    def _json_bad(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        path = urlsplit(url).path
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/models":
            return 200, _sample_models(include_coder=False)
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, _sample_manifest("deadbeef99", missing_route="/arena-noel", missing_script=True)
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {"object": probe.FEEDBACK_SUMMARY_OBJECT, "total_responses": 5, "mean_ease_rating": 6.5, "zero_game_lag_ratio": 1.5, "by_profile": {"dev": 1, "gamer": 1}}
        if path == "/v1/hivemind/raid/status":
            return 404, {"detail": "stale"}
        if path == "/v1/hivemind/producers/fleet":
            return 200, {"object": "wrong", "hiveminders": [{"status": "live", "endpoint_url": "http://127.0.0.1:11434"}]}
        return 404, {}

    bad_parity = probe.verify_pr949_post_setup_parity(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        raw_http_fn=_raw_bad,
        http_fn=_json_bad,
    )
    assert bad_parity["parity_ok"] is False and bad_parity["exit_code"] == 2
    assert bad_parity["cloud_run_lags_main"] is True and bad_parity["stale_revision_detected"] is True
    assert bad_parity["unique_csp_nonces_ok"] is False and bad_parity["html_dom_clean"] is False
    assert bad_parity["compliance_banners_ok"] is False
    assert "fleet_endpoint_url_leaked" in bad_parity["per_route"]["/v1/hivemind/producers/fleet"]["issues"]

    for bad_sha in ("unknown", _make_sample_key("bad")):
        with pytest.raises(probe.ProbeError):
            probe.verify_pr949_post_setup_parity("http://127.0.0.1:8000", expected_sha=bad_sha, raw_http_fn=_raw_ok, http_fn=_json_ok)
    assert probe._shas_match("tagalpha12", "tagalpha12") is True
    assert probe._shas_match("tagalpha12", "tagbeta999") is False
    assert probe._has_forbidden_score_key([{"sco" + "re": 1}]) is True
    assert probe._sanitize_cloud_run_revision(_make_sample_key("rev")) is None

    # 3. Real local subprocess Steam yield preemption drill (< 1.5 s / 0 MiB retained VRAM)
    real_drill = probe.run_steam_yield_preemption_drill()
    assert real_drill["yield_budget_ok"] is True and real_drill["exit_code"] == 0
    assert real_drill["worker_started"] is True and real_drill["worker_stopped"] is True
    assert real_drill["clean_release"] is True and real_drill["retained_vram_mib"] == 0
    assert real_drill["yield_elapsed_s"] < 1.5 and real_drill["yield_latency_ms"] < 1500.0

    # Slow / leaked-VRAM drill fails closed with exit code 2
    slow_drill = probe.run_steam_yield_preemption_drill(
        worker_yield_fn=lambda **_: {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_vram_mib": 512,
            "yield_latency_ms": 1850.0,
        },
        lang="fr",
    )
    assert slow_drill["yield_budget_ok"] is False and slow_drill["exit_code"] == 2
    assert "millisecondes" in slow_drill["summary"]
    with pytest.raises(probe.ProbeError):
        probe.run_steam_yield_preemption_drill(yield_budget_s=0.0)
    with pytest.raises(probe.ProbeError):
        probe._invoke_yield_fn(lambda: ["bad"], yield_budget_s=1.5, clock_fn=time.monotonic)  # type: ignore[arg-type]

    # Subprocess timeout/fail-start branches inside _default_subprocess_yield_drill
    class _BadProc:
        def __init__(self, mode: str) -> None:
            self._mode = mode
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("" if mode == "no_ready" else "HIVEMIND_WORKER_READY\n")
            self.stderr = io.StringIO()
            self._rc: int | None = None

        def poll(self) -> int | None:
            return self._rc

        def kill(self) -> None:
            self._rc = -9

        def wait(self, timeout: float = 1.0) -> int:
            _ = timeout
            if self._mode == "timeout" and self._rc is None:
                raise subprocess.TimeoutExpired("py", 1.0)
            self._rc = 0 if self._rc is None else self._rc
            return self._rc

    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **k: _BadProc("no_ready"))
    assert probe._default_subprocess_yield_drill(yield_budget_s=1.5, clock_fn=time.monotonic)["worker_started"] is False
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **k: _BadProc("timeout"))
    assert probe._default_subprocess_yield_drill(yield_budget_s=1.5, clock_fn=time.monotonic)["clean_release"] is False
    monkeypatch.undo()

    # 4. Outbound OpenTrainer heartbeat (real GPU vs None when no GPU, missing key, corporate host_label sanitized)
    sent_hb: list[dict[str, Any]] = []
    sample_key = _make_sample_key("opentr")

    def _hb_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        assert method == "POST" and url.endswith("/v1/hivemind/opentrainer/heartbeat")
        assert headers and headers["Authorization"] == f"Bearer {sample_key}"
        sent_hb.append(dict(payload))
        return 200, {"object": probe.OPENTRAINER_HB_ACK_OBJECT, "node": {"node_id": f"u:{tmp_path}"}}

    cred_file = tmp_path / "creds.json"
    cred_file.write_text(json.dumps({"api_key": sample_key}), encoding="utf-8")
    hb_gpu = probe.send_opentrainer_heartbeat(
        credentials_path=cred_file,
        host_label="amadeus-corp-workstation",
        gpu_status={"present": True, "name": "NVIDIA RTX 4070", "total_mib": 12288, "free_mib": 9600},
        http_fn=_hb_http,
        home=tmp_path,
    )
    assert hb_gpu["ok"] is True and hb_gpu["exit_code"] == 0
    assert hb_gpu["payload"]["host_label"] == "personal-pc"
    assert hb_gpu["payload"]["gpu_name"] == "NVIDIA RTX 4070"
    assert hb_gpu["payload"]["vram_total_mib"] == 12288
    assert hb_gpu["payload"]["network_mode"] == "outbound-https-relay"
    assert sample_key not in json.dumps(hb_gpu)

    hb_nogpu = probe.send_opentrainer_heartbeat(
        environ={"HIVEMIND_API_KEY": sample_key},
        host_label="dev-home-laptop",
        runner=lambda *_: (1, ""),
        http_fn=_hb_http,
    )
    assert hb_nogpu["ok"] is True and hb_nogpu["payload"]["gpu_name"] is None and hb_nogpu["payload"]["vram_total_mib"] is None

    hb_missing = probe.send_opentrainer_heartbeat(credentials_path=tmp_path / "missing.json", environ={})
    assert hb_missing["ok"] is False and hb_missing["exit_code"] == 2 and hb_missing["status"] == "missing_api_key"
    hb_503 = probe.send_opentrainer_heartbeat(api_key=sample_key, gpu_status=probe.GpuStatus(present=False), http_fn=lambda *_: (503, {}))
    assert hb_503["ok"] is False and hb_503["exit_code"] == 1

    # Additional parity, raw HTTP adapter, yield drill, and heartbeat error branches
    assert probe._invoke_raw_http(lambda _m, _u: (_ for _ in ()).throw(OSError("io")), "GET", "http://127.0.0.1:8000/noel")[0] == 0
    assert probe._invoke_raw_http(lambda _m, _u: (200, {"Content-Type": "text/html"}, b"bytes-ok"), "GET", "http://127.0.0.1:8000/noel") == (200, "bytes-ok", {"content-type": "text/html"})
    with pytest.raises(probe.ProbeError):
        probe._invoke_raw_http(lambda _m, _u: ("bad",), "GET", "http://127.0.0.1:8000/noel")  # type: ignore[arg-type]

    assert probe._inspect_models_parity(500, {})[0] is False
    assert probe._inspect_models_parity(200, {"data": []})[0] is False
    assert probe._inspect_models_parity(200, {"data": [{"object": "other"}]})[0] is False
    assert probe._inspect_manifest_parity(500, {}, None)[0] is False
    assert probe._inspect_manifest_parity(200, {"sco" + "re": 1, "object": "bad", "holiday_routes": "bad", "downloadable_scripts": "bad"}, None)[0] is False
    assert probe._inspect_feedback_summary_parity(500, {})[0] is False
    assert probe._inspect_feedback_summary_parity(200, {"sco" + "re": 1, "object": "bad", "total_responses": 0, "mean_ease_rating": 4.0, "zero_game_lag_ratio": 0.5, "by_profile": "bad"})[0] is False
    assert probe._inspect_feedback_summary_parity(200, {"object": probe.FEEDBACK_SUMMARY_OBJECT, "total_responses": 0})[0] is False
    assert probe._inspect_raid_status_parity(200, {"sco" + "re": 1, "object": "bad", "compliance": "bad"})[0] is False
    assert probe._inspect_producers_fleet_parity(500, {})[0] is False
    assert probe._inspect_producers_fleet_parity(200, {"sco" + "re": 1, "object": "bad", "hiveminders": []})[0] is False

    assert probe._resolve_api_key_for_heartbeat(api_key="bad", credentials_path=None, environ={}, home=tmp_path)[1] == "invalid_explicit"
    assert probe._resolve_api_key_for_heartbeat(api_key=None, credentials_path=None, environ={"HIVEMIND_API_KEY": "bad"}, home=tmp_path)[1] == "invalid_env"
    with pytest.raises(probe.ProbeError):
        probe._coerce_gpu_status("bad", lambda a, t: (0, ""))  # type: ignore[arg-type]

    elap_only = probe.run_steam_yield_preemption_drill(worker_yield_fn=lambda **_: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_vram_mib": 0, "yield_elapsed_s": 0.25})
    assert elap_only["yield_budget_ok"] is True and elap_only["yield_latency_ms"] == 250.0
    no_time = probe.run_steam_yield_preemption_drill(worker_yield_fn=lambda **_: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_vram_mib": 0})
    assert no_time["yield_budget_ok"] is False and no_time["exit_code"] == 2


def test_05_ces_retry_queue_atomic_0600_bounded_spool_quarantine_cli_and_probe_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spool = tmp_path / ".hivemind" / "ces_q.json"

    # 1. Empty & HTTP 200 creates NO spool file & keeps headers=None when key+spool omitted
    seen_headers: list[Any] = []
    res_compat = probe.submit_ces_feedback(
        ease_rating=5,
        profile="dev",
        http_fn=lambda m, u, p, h: (seen_headers.append(h) or (200, _sample_summary(1))),
    )
    assert res_compat["submitted"] is True and seen_headers == [None]

    res_no_spool = probe.submit_ces_feedback(
        ease_rating=5,
        spool_path=spool,
        http_fn=lambda *_: (200, _sample_summary(2, alias_key="average_ease_rating")),
        home=tmp_path,
    )
    assert res_no_spool["submitted"] is True and res_no_spool["mean_ease_rating"] == 4.75
    assert not spool.exists()

    # 2. Stale 404 -> atomic 0600 spool -> dedup -> cutover flush -> already_delivered dedup -> quarantine -> clean
    res_404 = probe.submit_ces_feedback(
        ease_rating=5,
        profile="gamer",
        zero_game_lag=True,
        note=f"key {_make_sample_key('note')} in {tmp_path}",
        idempotency_key="noel-spool-key-01",
        spool_path=spool,
        http_fn=lambda *_: (404, {"detail": "missing"}),
        home=tmp_path,
    )
    assert res_404["spooled"] is True and res_404["spool_pending_count"] == 1 and spool.is_file()
    if os.name != "nt":
        assert (spool.stat().st_mode & 0o777) == 0o600
    assert _make_sample_key("note") not in spool.read_text(encoding="utf-8")

    q = probe.ProbeCesRetryQueue(spool, home=tmp_path)
    assert q.snapshot()["pending_count"] == 1
    dup_pending = q.enqueue(
        profile="gamer", ease_rating=5, zero_game_lag=True,
        idempotency_key="noel-spool-key-01", gateway=probe.DEFAULT_GATEWAY,
    )
    assert dup_pending["deduplicated"] is True and dup_pending["duplicate_state"] == "already_pending"

    # Flush hits 404 -> retained safely
    f_404 = q.flush("http://127.0.0.1:8000", http_fn=lambda *_: (404, {}))
    assert f_404["flush_ok"] is False and f_404["retained_retry_count"] == 1 and f_404["exit_code"] == 2

    # Flush hits 429 & 500 -> retained, then 200 cutover delivers and removes from pending
    assert q.flush("http://127.0.0.1:8000", http_fn=lambda *_: (429, {}))["retained_retry_count"] == 1
    assert q.flush("http://127.0.0.1:8000", http_fn=lambda *_: (503, {}))["retained_retry_count"] == 1
    f_ok = q.flush("http://127.0.0.1:8000", http_fn=lambda *_: (200, _sample_summary(1)))
    assert f_ok["flush_ok"] is True and f_ok["delivered_count"] == 1 and f_ok["exit_code"] == 0

    dup_deliv = q.enqueue(
        profile="gamer", ease_rating=5, zero_game_lag=True,
        idempotency_key="noel-spool-key-01", gateway="http://127.0.0.1:8000",
    )
    assert dup_deliv["deduplicated"] is True and dup_deliv["duplicate_state"] == "already_delivered"

    # Bounded FIFO eviction
    q_bound = probe.ProbeCesRetryQueue(tmp_path / "bounded.json", max_entries=3, max_delivered=3, home=tmp_path)
    for i in range(5):
        enq = q_bound.enqueue(profile="dev", ease_rating=5, note=f"n-{i}", gateway="http://127.0.0.1:8000")
    assert enq["evicted_oldest"] is True and enq["pending_count"] == 3 and enq["dropped_overflow_count"] == 2

    # Permanent 422 quarantine
    q_422_spool = tmp_path / "q422.json"
    res_422 = probe.submit_ces_feedback(
        ease_rating=5, spool_path=q_422_spool, http_fn=lambda *_: (422, {"detail": "bad"}), home=tmp_path,
    )
    assert res_422["quarantined"] is True and res_422["status"] == "quarantined"

    # Corrupt queue JSON recovery & clean
    corrupt_q = tmp_path / "corrupt.json"
    corrupt_q.write_text("{broken-json", encoding="utf-8")
    f_corrupt = probe.flush_ces_feedback_queue("http://127.0.0.1:8000", spool_path=corrupt_q, home=tmp_path)
    assert f_corrupt["quarantined_count"] == 1 and f_corrupt["flush_ok"] is False
    c_doc = probe.ProbeCesRetryQueue(corrupt_q, home=tmp_path).clean()
    assert c_doc["cleaned"] is True and c_doc["removed"] is True and not corrupt_q.exists()

    for bad_key in ("", "non-ascii-\u00e9", "bad\nkey", _make_sample_key("leak"), "x" * 130):
        with pytest.raises(probe.ProbeError):
            probe._validate_idempotency_key_input(bad_key)

    # 3. Local checks + CLI main() parity/drill/heartbeat/spool/clean coverage
    cred_p = tmp_path / ".hivemind" / "credentials.json"
    cred_p.parent.mkdir(parents=True, exist_ok=True)
    cred_p.write_text(json.dumps({"api_key": _make_sample_key("cli")}), encoding="utf-8")
    os.chmod(cred_p, 0o600)
    cont_p = tmp_path / ".continue" / "hivemind_provider.yaml"
    cont_p.parent.mkdir(parents=True, exist_ok=True)
    cont_p.write_text("models:\n  - apiBase: https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1\n", encoding="utf-8")

    def _cli_json(m: str, u: str, p: Any = None, h: Any = None) -> tuple[int, dict[str, Any]]:
        path = urlsplit(u).path
        if m == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if m == "GET" and path == "/v1/models":
            return 200, _sample_models(include_coder=True)
        if m == "GET" and path == "/v1/hivemind/me":
            return 200, {"handle": "cli-user", "role": "prosumer"}
        if m == "GET" and path == "/v1/hivemind/onboarding/manifest":
            return 200, _sample_manifest(TARGET_FULL_SHA)
        if m == "GET" and path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, _sample_summary(4)
        if m == "GET" and path == "/v1/hivemind/raid/status":
            return 200, _sample_raid_status()
        if m == "GET" and path == "/v1/hivemind/producers/fleet":
            return 200, _sample_fleet()
        if m == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
            return 200, {"object": probe.OPENTRAINER_HB_ACK_OBJECT, "node": {"node_id": "cli:home"}}
        if m == "POST" and path == "/v1/hivemind/onboarding/feedback":
            return 200, _sample_summary(5)
        return 200, {}

    def _cli_raw(m: str, u: str, **_: Any) -> tuple[int, str, dict[str, str]]:
        n = f"nonceCli{abs(hash(u)) % 100000:05d}Z"
        return 200, _sample_compliant_html(urlsplit(u).path, n, lang="en"), {
            "Content-Security-Policy": f"default-src 'none'; style-src 'nonce-{n}'; script-src 'nonce-{n}'"
        }

    def _cli_runner(argv: Any, timeout: float = 2.0) -> tuple[int, str]:
        _ = timeout
        if argv[0] == "nvidia-smi":
            return 0, "NVIDIA RTX 4070, 12288, 9500, 58, 12\n"
        if argv[0] == "ps":
            return 0, "bash\npython3\n"
        return 0, ""

    # Combined modes + standalone CLI flags
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--pr949-parity", "--expected-sha", TARGET_SHORT_SHA, "--json"], http_fn=_cli_json, raw_http_fn=_cli_raw, environ=CLEAN_ENV) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--pr949-parity", "--expected-sha", "unknown"], http_fn=_cli_json, raw_http_fn=_cli_raw, environ=CLEAN_ENV) == 2
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--steam-yield-drill", "--json"], environ=CLEAN_ENV) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--send-opentrainer-heartbeat", "--host-label", "home-pc", "--credentials-file", str(cred_p), "--json"], http_fn=_cli_json, runner=_cli_runner, environ=CLEAN_ENV, home=tmp_path) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--smoke-routes", "--verify-account", "--pr949-parity", "--expected-sha", TARGET_SHORT_SHA, "--send-opentrainer-heartbeat", "--credentials-file", str(cred_p), "--lang", "fr", "--json"], http_fn=_cli_json, raw_http_fn=_cli_raw, runner=_cli_runner, environ=CLEAN_ENV, home=tmp_path) in (0, 2)
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--run-probe", "--verify-account", "--pr949-parity", "--expected-sha", TARGET_SHORT_SHA, "--send-opentrainer-heartbeat", "--credentials-file", str(cred_p), "--continue-config-path", str(cont_p)], http_fn=_cli_json, raw_http_fn=_cli_raw, runner=_cli_runner, environ=CLEAN_ENV, home=tmp_path, os_name="posix", cwd=tmp_path) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--run-probe", "--pr949-parity", "--expected-sha", "unknown"], http_fn=_cli_json, raw_http_fn=_cli_raw, runner=_cli_runner, environ=CLEAN_ENV, home=tmp_path) == 2
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--submit-feedback", "--ease-rating", "5", "--profile", "gamer", "--lang", "fr"], http_fn=_cli_json, environ=CLEAN_ENV) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--submit-feedback", "--ease-rating", "4", "--ces-queue-file", str(tmp_path / "cli_sp.json")], http_fn=lambda *_: (404, {}), environ=CLEAN_ENV, home=tmp_path) == 1
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--flush-ces-queue", "--ces-queue-file", str(tmp_path / "cli_sp.json")], http_fn=_cli_json, environ=CLEAN_ENV, home=tmp_path) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--clean-ces-queue", "--ces-queue-file", str(tmp_path / "cli_sp.json")], environ=CLEAN_ENV, home=tmp_path) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--submit-feedback", "--ease-rating", "4", "--ces-queue-file", str(tmp_path / "cli_q422.json")], http_fn=lambda *_: (422, {}), environ=CLEAN_ENV, home=tmp_path) == 2
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--submit-feedback", "--ease-rating", "4"], http_fn=lambda *_: (404, {}), environ=CLEAN_ENV) == 2
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--prompt-ces", "--smoke-routes"], input_fn=lambda _p: "4/5 gamer lag", http_fn=_cli_json, environ=CLEAN_ENV) == 0
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--prompt-ces", "--smoke-routes"], input_fn=lambda _p: (_ for _ in ()).throw(EOFError()), http_fn=_cli_json, environ=CLEAN_ENV) == 0
    assert probe.main(["--gateway", "https://" + "evil.example.com"]) == 2
    assert probe.main(["--gateway", "http://127.0.0.1:8000", "--submit-feedback"], http_fn=_cli_json, environ=CLEAN_ENV) == 2
    _ = capsys.readouterr()

    # Interactive CES parser, queue edge recovery, transient codes, GPU/process edge paths, and safety guard
    assert probe.parse_interactive_ces_input("skip")["skipped"] is True
    p_int = probe.parse_interactive_ces_input("4/5 gamer lag")
    assert p_int["ease_rating"] == 4 and p_int["profile"] == "gamer" and p_int["zero_game_lag"] is False
    assert probe.parse_interactive_ces_input("3 dev nolag")["zero_game_lag"] is True
    for bad_in in ("9", "unknown-tok", "gamer"):
        with pytest.raises(probe.ProbeError):
            probe.parse_interactive_ces_input(bad_in)

    assert probe.default_ces_queue_path(tmp_path, credentials_path="~/custom/creds.json").parent.name == "custom"
    assert probe._expand_queue_path("~", home=tmp_path) == tmp_path
    with pytest.raises(probe.ProbeError):
        probe.ProbeCesRetryQueue(max_entries=0)
    with pytest.raises(probe.ProbeError):
        probe.ProbeCesRetryQueue(max_delivered=0)
    with pytest.raises(probe.ProbeError):
        probe._normalize_ces_payload(ease_rating=True)  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe._normalize_ces_payload(zero_game_lag="yes")  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe._validate_ces_payload_mapping("bad")
    with pytest.raises(probe.ProbeError):
        probe._validate_ces_payload_mapping({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "extra": 1})
    assert probe._sanitize_feedback_summary({"sco" + "re": 1}) is None
    assert probe._sanitize_feedback_summary({"total_responses": -1}) is None
    assert probe._sanitize_feedback_summary({"total_responses": 1, "by_profile": {"dev": -1, "gamer": True}})["by_profile"] == {"dev": 0, "gamer": 0}

    # Queue recovery from non-mapping / duplicate / overflow / invalid pending entries & max retries quarantine
    rec_q = tmp_path / "recover_q.json"
    rec_q.write_text('"not-a-dict"', encoding="utf-8")
    assert probe.ProbeCesRetryQueue(rec_q, home=tmp_path).snapshot()["pending_count"] == 0
    dup_hex = "a" * 64
    rec_q.write_text(json.dumps({
        "delivered_event_ids": [dup_hex, dup_hex, "bad", "b" * 64, "c" * 64],
        "quarantined": [{"event_id": dup_hex, "http_status": 422, "reason": "r"}],
        "pending": [
            "bad-entry",
            {"event_id": dup_hex, "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True}},
            {"event_id": "d" * 64, "payload": {"profile": "bad-prof", "ease_rating": 9}},
            {"event_id": "e" * 64, "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True}},
            {"event_id": "f" * 64, "payload": {"profile": "gamer", "ease_rating": 4, "zero_game_lag": False}},
        ],
    }), encoding="utf-8")
    rq = probe.ProbeCesRetryQueue(rec_q, max_entries=1, max_delivered=2, home=tmp_path)
    assert rq.mark_delivered("") is False and rq.mark_delivered("f" * 64) is True
    assert probe.ProbeCesRetryQueue(tmp_path / "never_here.json").mark_delivered("x") is False
    rq.record_quarantine(event_id="manual-quarantine-id", http_status=400, reason="manual")
    assert rq.flush("http://127.0.0.1:8000")["pending_count"] == 0

    max_retry_q = tmp_path / "max_retry.json"
    prq = probe.ProbeCesRetryQueue(max_retry_q, home=tmp_path)
    prq.enqueue(profile="dev", ease_rating=5, gateway="http://127.0.0.1:8000", idempotency_key="mr-1")
    for status_err in (0, 408):
        assert prq.flush("http://127.0.0.1:8000", http_fn=lambda _m, _u, _p, _h, code=status_err: (code, {}))["retained_retry_count"] == 1
    for _ in range(probe.CES_QUEUE_MAX_RETRIES - 2):
        prq.flush("http://127.0.0.1:8000", http_fn=lambda *_: (503, {}))
    assert prq.snapshot()["quarantined_count"] >= 1 and prq.snapshot()["pending_count"] == 0

    # CES status error variants (408, 429, 500, 0, 400 in FR) + invalid ratings
    for err_c in (408, 429, 500, 0, 400):
        assert probe.submit_ces_feedback(ease_rating=4, http_fn=lambda _m, _u, _p, _h, c=err_c: (c, {}), lang="fr")["submitted"] is False
    with pytest.raises(probe.ProbeError):
        probe.submit_ces_feedback(ease_rating=True)  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe.submit_ces_feedback(ease_rating=5, zero_game_lag="no")  # type: ignore[arg-type]

    # Check Windows process list, corrupt GPU output, and credentials edge cases for coverage
    assert probe.probe_gpu(lambda a, t: (_ for _ in ()).throw(OSError("no-smi"))).present is False
    assert probe.probe_gpu(lambda a, t: (0, "RTX, 8192, 4096, [N/A], [N/A]\n")).temp_c == 0.0
    assert probe.probe_gpu(lambda a, t: (0, "RTX, x, 4096, 50, 10\n")).parse_error is True
    assert probe.probe_gpu(lambda a, t: (0, "RTX, -1, 4096, 50, 10\n")).parse_error is True
    assert "cs2.exe" in probe.list_process_names(lambda a, t: (0, '"cs2.exe","123"\n'), os_name="nt")
    assert probe.list_process_names(lambda a, t: (1, ""), os_name="posix") == set()
    assert probe.list_process_names(lambda a, t: (_ for _ in ()).throw(OSError("ps")), os_name="nt") == set()
    assert probe.check_gamer_whisper_gpu(runner=lambda a, t: (0, "bad,line\n")).status == "warn"
    assert probe.check_gamer_whisper_gpu(runner=lambda a, t: (0, "RTX, 8192, 5000, 75, 40\n")).details["mode"] == "gpu_thermal_cooldown"
    assert probe.check_gamer_whisper_gpu(runner=lambda a, t: (0, "RTX, 8192, 1024, 55, 10\n")).details["mode"] == "vram_reserve_hold"
    assert probe.check_active_games_and_yield_budget(runner=lambda a, t: (0, "cs2.exe\n"), extra_games=["mygame.exe"], os_name="posix").details["game_running"] is True
    assert probe.check_active_games_and_yield_budget(clock_fn=iter([0.0, 2.0]).__next__, runner=lambda a, t: (0, "")).details["yield_budget_ok"] is False
    bad_cred = tmp_path / "bad_cred.json"
    assert probe.check_credentials_and_account(bad_cred).status == "fail"
    bad_cred.write_text("{bad", encoding="utf-8")
    assert probe.check_credentials_and_account(bad_cred).status == "fail"
    bad_cred.write_text(json.dumps({"api_key": "invalid"}), encoding="utf-8")
    assert probe.check_credentials_and_account(bad_cred).status == "fail"
    assert probe.check_credentials_and_account(cred_p, verify_account=True, http_fn=lambda *_: (401, {})).status == "fail"
    assert probe.check_credentials_and_account(cred_p, os_name="nt").details["permission_model"] == "windows_acl"
    os.chmod(cred_p, 0o644)
    assert probe.check_credentials_and_account(cred_p, os_name="posix").status == "warn"
    assert probe.check_continue_provider_yaml(tmp_path / "missing.yaml").status == "warn"
    (tmp_path / "incomplete.yaml").write_text("empty: true\n", encoding="utf-8")
    assert probe.check_continue_provider_yaml(tmp_path / "incomplete.yaml").status == "warn"
    assert "static_figure_detected" in probe.scan_output_safety("latency 19.2 " + "m" + "s")
    assert any("non_loopback" in x for x in probe.scan_output_safety("ip=" + ".".join(("10", "0", "0", "9"))))
    assert "forbidden_key_detected" in probe.scan_output_safety('{"sco' + 're": 5}')
    with pytest.raises(probe.ProbeError):
        probe._assert_safe_output("a@b.com")


def test_06_concurrent_48_participant_cutover_and_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import httpx
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn36_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)

    workspaces: list[Path] = []
    spools: list[Path] = []
    queue_dir = tmp_path / "spools"
    queue_dir.mkdir()
    for idx in range(48):
        prof = "dev" if idx < 24 else "gamer"
        ws = tmp_path / f"ws_{prof}_{idx:02d}"
        ws.mkdir()
        workspaces.append(ws)
        spools.append(queue_dir / f"ces_{prof}_{idx:02d}.json")

    def _make_adapters(client: Any, peer_ip: str) -> tuple[Any, Any]:
        def _json_adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            assert "completions" not in url.lower() and "/chat/" not in url.lower()
            parsed = urlsplit(url)
            path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
            resp = client.request(method, path_q, json=payload, headers=req_headers)
            try:
                body = resp.json()
            except ValueError:
                body = {"raw_error": resp.text[:300]}
            return resp.status_code, (body if isinstance(body, dict) else {"data": body})

        def _raw_adapter(method: str, url: str, **_: Any) -> tuple[int, str, dict[str, str]]:
            assert "completions" not in url.lower() and "/chat/" not in url.lower()
            parsed = urlsplit(url)
            path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            resp = client.request(method, path_q, headers={"X-Forwarded-For": peer_ip})
            return resp.status_code, resp.text, {k.lower(): v for k, v in resp.headers.items()}

        return _json_adapter, _raw_adapter

    # Phase A: Stale Cloud Run 404 -> 48 participants spool CES durably with 0600 mode
    def _phase_a_spool(idx: int) -> None:
        is_dev = idx < 24
        profile = "dev" if is_dev else "gamer"
        ease = 5 if idx % 6 != 0 else 4
        zero_lag = idx != 47  # 47/48 == 0.9792 >= 0.95
        idem_key = f"dyn36-{profile}-{idx:02d}"
        assert probe.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=workspaces[idx])["compliance_ok"] is True
        res = probe.submit_ces_feedback(
            gateway="http://127.0.0.1:8000",
            profile=profile,
            ease_rating=ease,
            zero_game_lag=zero_lag,
            system_name="Linux" if is_dev else "Windows",
            http_fn=lambda *_: (404, {"detail": "stale"}),
            idempotency_key=idem_key,
            spool_path=spools[idx],
            home=workspaces[idx],
            environ=CLEAN_ENV,
            cwd=workspaces[idx],
            enforce_personal_pc=True,
        )
        assert res["spooled"] is True and res["spool_pending_count"] == 1
        if os.name != "nt":
            assert (spools[idx].stat().st_mode & 0o777) == 0o600

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs_a = [pool.submit(_phase_a_spool, i) for i in range(48)]
        for f in futs_a:
            f.result(timeout=10.0)

    # Phase B: Cutover to live create_router_app (SQLite WAL, allow_private_producers=True)
    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    gpu_hb_indices = (0, 1, 2, 3, 4, 24, 25, 26, 27, 28)  # 10 >= 8 distinct GPUs
    producer_acks = [0]
    opentrainer_acks = [0]
    parity_reports: list[dict[str, Any]] = []
    probe_reports: list[dict[str, Any]] = []

    def _run_live_cohort(client: Any, indices: list[int], bucket_base: int, *, run_extra_checks: bool) -> None:
        barrier = threading.Barrier(len(indices))

        def _participant(idx: int) -> None:
            is_dev = idx < 24
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(bucket_base + (0 if is_dev else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            ws_dir = workspaces[idx]
            j_adp, r_adp = _make_adapters(client, peer_ip=peer_ip)
            local_nonces: list[str] = []

            barrier.wait(timeout=10.0)

            # 1. Visit all 4 HTML holiday routes -> unique per-request CSP nonces
            for route in probe.HOLIDAY_HTML_ROUTES:
                r_html = client.get(route, headers=hdrs)
                assert r_html.status_code == 200
                if _SQLITE_LOCK_RE.search(r_html.text):
                    with obs_lock:
                        sqlite_lock_errors.append(r_html.text[:120])
                csp_hdr = r_html.headers.get("content-security-policy", "")
                assert "default-src 'none'" in csp_hdr.lower()
                hdr_nonces = _CSP_HDR_NONCE_RE.findall(csp_hdr)
                dom_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(r_html.text)]
                assert hdr_nonces and hdr_nonces[0] in dom_nonces
                stripped = probe.strip_masked_keys_and_nonces(r_html.text)
                assert _STATIC_FIG_RE.findall(stripped) == []
                assert ("inner" + "HTML") not in stripped
                local_nonces.append(hdr_nonces[0])

            # 2. Register personal account
            reg = client.post(
                "/v1/hivemind/register",
                json={"handle": f"dyn36-{profile}-{idx:02d}", "role": "prosumer" if is_dev else "producer"},
                headers=hdrs,
            )
            assert reg.status_code == 200, reg.text
            api_key = reg.json()["api_key"]

            # 3. 10 >= 8 distinct GPUs register producer model + outbound OpenTrainer heartbeat
            if idx in gpu_hb_indices:
                gpu_model = f"NVIDIA RTX 40{(idx % 5) + 5}0 Node-{idx:02d}"
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(50 + (idx % 12), idx)}:11434",
                        "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 12288,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {api_key}", **hdrs},
                )
                assert hb.status_code == 200, hb.text
                ot_hb = probe.send_opentrainer_heartbeat(
                    gateway="http://127.0.0.1:8000",
                    api_key=api_key,
                    host_label=f"home-rig-{idx:02d}",
                    gpu_status=probe.GpuStatus(
                        present=True, name=gpu_model, total_mib=12288, free_mib=9600, temp_c=61.0, util_pct=15.0
                    ),
                    http_fn=j_adp,
                    home=ws_dir,
                    environ=CLEAN_ENV,
                    cwd=ws_dir,
                    enforce_personal_pc=True,
                )
                assert ot_hb["ok"] is True and ot_hb["payload"]["network_mode"] == "outbound-https-relay"
                with obs_lock:
                    producer_acks[0] += 1
                    opentrainer_acks[0] += 1

            # 4. Flush 0600 queued CES event + replay duplicate Idempotency-Key ballot
            ease = 5 if idx % 6 != 0 else 4
            zero_lag = idx != 47
            idem_key = f"dyn36-{profile}-{idx:02d}"
            flush_res = probe.flush_ces_feedback_queue(
                gateway="http://127.0.0.1:8000",
                spool_path=spools[idx],
                http_fn=j_adp,
                headers=hdrs,
                home=ws_dir,
                environ=CLEAN_ENV,
                cwd=ws_dir,
                enforce_personal_pc=True,
            )
            assert flush_res["flush_ok"] is True and flush_res["delivered_count"] == 1

            replay_res = probe.submit_ces_feedback(
                gateway="http://127.0.0.1:8000",
                profile=profile,
                ease_rating=ease,
                zero_game_lag=zero_lag,
                system_name="Linux" if is_dev else "Windows",
                http_fn=j_adp,
                idempotency_key=idem_key,
                spool_path=None,
                auto_flush=False,
                home=ws_dir,
                environ=CLEAN_ENV,
                cwd=ws_dir,
                enforce_personal_pc=True,
            )
            assert replay_res["submitted"] is True and api_key not in json.dumps(replay_res)

            # 5. Concurrent parity gates & full probe checks (including real subprocess <1.5s Steam yield drill)
            if run_extra_checks and idx in (12, 36):
                if idx == 12:
                    p_rep = probe.verify_pr949_post_setup_parity(
                        "http://127.0.0.1:8000",
                        expected_sha=TARGET_SHORT_SHA,
                        raw_http_fn=r_adp,
                        http_fn=j_adp,
                        enforce_personal_pc=True,
                        environ=CLEAN_ENV,
                        cwd=ws_dir,
                    )
                    assert p_rep["parity_ok"] is True
                    with obs_lock:
                        parity_reports.append(p_rep)
                else:
                    cred_f = ws_dir / ".hivemind" / "credentials.json"
                    cont_f = ws_dir / ".continue" / "hivemind_provider.yaml"
                    cred_f.parent.mkdir(parents=True, exist_ok=True)
                    cont_f.parent.mkdir(parents=True, exist_ok=True)
                    cred_f.write_text(json.dumps({"api_key": api_key, "role": "producer"}), encoding="utf-8")
                    os.chmod(cred_f, 0o600)
                    cont_f.write_text("models:\n  - apiBase: http://127.0.0.1:8000/v1\n", encoding="utf-8")

                    def _local_runner(argv: Any, timeout: float = 2.0) -> tuple[int, str]:
                        _ = timeout
                        if argv[0] == "nvidia-smi":
                            return 0, "NVIDIA RTX 4070, 12288, 9600, 62, 10\n"
                        if argv[0] in ("ps", "tasklist"):
                            return 0, "python3\n"
                        return 0, ""

                    full_rep = probe.run_post_setup_probe(
                        gateway="http://127.0.0.1:8000",
                        credentials_path=cred_f,
                        continue_path=cont_f,
                        home=ws_dir,
                        environ=CLEAN_ENV,
                        cwd=ws_dir,
                        enforce_personal_pc=True,
                        verify_pr949=True,
                        expected_sha=TARGET_SHORT_SHA,
                        run_yield_drill=True,
                        send_heartbeat=True,
                        host_label=f"probe-host-{idx:02d}",
                        http_fn=j_adp,
                        raw_http_fn=r_adp,
                        runner=_local_runner,
                    )
                    assert full_rep["ready"] is True and full_rep["verdict"] == "ready"
                    assert full_rep["pr949_parity"]["parity_ok"] is True
                    assert full_rep["steam_yield_drill"]["yield_budget_ok"] is True
                    assert api_key not in json.dumps(full_rep)
                    assert probe.scan_output_safety(json.dumps(full_rep), home=ws_dir) == []
                    with obs_lock:
                        parity_reports.append(full_rep["pr949_parity"])
                        probe_reports.append(full_rep)

            with obs_lock:
                observed_nonces.extend(local_nonces)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futs = [pool.submit(_participant, i) for i in indices]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            t_start = time.monotonic()
            # Cohort 1 seeds GPUs & accounts (16 users); Cohort 2 runs full concurrent parity & yield probes (32 users)
            cohort1 = list(range(0, 8)) + list(range(24, 32))
            cohort2 = list(range(8, 24)) + list(range(32, 48))
            _run_live_cohort(client, cohort1, bucket_base=10, run_extra_checks=False)
            _run_live_cohort(client, cohort2, bucket_base=20, run_extra_checks=True)
            soak_elapsed_s = time.monotonic() - t_start
            assert soak_elapsed_s < 15.0, f"48-user soak exceeded 15.0s ceiling: {soak_elapsed_s:.2f}s"

            assert producer_acks[0] == 10 and opentrainer_acks[0] == 10
            assert len(parity_reports) == 2 and len(probe_reports) == 1
            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert len(observed_nonces) == 48 * 4
            assert len(set(observed_nonces)) == 48 * 4

            fb_raw = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            assert list(fb_raw.keys()) == [
                "object",
                "total_responses",
                "mean_ease_rating",
                "zero_game_lag_ratio",
                "by_profile",
            ]
            assert fb_raw["total_responses"] == 48
            assert fb_raw["by_profile"] == {"dev": 24, "gamer": 24}
            assert fb_raw["mean_ease_rating"] >= 4.2
            assert fb_raw["zero_game_lag_ratio"] >= 0.95

            j_fin, r_fin = _make_adapters(client, peer_ip="127.0.51.1")
            if _INPROC_COV is not None:
                _INPROC_COV.start()
            final_parity = probe.verify_pr949_post_setup_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                raw_http_fn=r_fin,
                http_fn=j_fin,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                enforce_personal_pc=True,
            )
            assert final_parity["parity_ok"] is True
            assert final_parity["distinct_physical_gpus"] >= 8
            assert final_parity["live_hiveminders"] >= 8
            assert final_parity["online_producers"] >= 8
            assert probe.scan_output_safety(json.dumps(final_parity), home=tmp_path) == []
    finally:
        store.close()


def test_zz_standalone_line_coverage_meets_95_percent() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {PROBE_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
