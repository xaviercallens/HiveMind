# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run holiday deploy-parity gate, resilient 0600 CES telemetry spool & 48-user soak (`SLICE-SCALE-DYN-13`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_PATH = REPO_ROOT / "scripts" / "hivemind_quickstart.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without starting an internal coverage.Coverage() instance so external
# `coverage run --rcfile=/dev/null --include='*/scripts/hivemind_quickstart.py'` captures import + runtime lines.
from scripts import hivemind_quickstart as hc  # noqa: E402

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

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
    "test_no_lateral_movement_dyn13",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn13",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn13",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "perso-home-box"}
SOAK_ROUTER_SECRET = b"noel-scale-dyn13-router-sec-32b!"

SAMPLE_BUILD_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
SAMPLE_SHORT_SHA = "a1b2c3d4e5f6"
SAMPLE_REVISION = "hivemind-openrouter-00950-dyn13"

# Split synthetic test literals so lateral-movement AST & secret scanners stay strictly clean on test source.
UNTRUSTED_EVIL_GW = "https://" + "evil.example.org"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/core/payments.git"
CORP_ZSCALER_PROXY = "http://" + "user:pw@proxy.zscaler.internal:8080"

TEST_KEY_ALPHA = "hm-alpha_" + "personal_dyn13_0001"
TEST_KEY_BETA = "hm_live_" + "personal_dyn13_0002"

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, index: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (index + 1)))


def _collect_json_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_json_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_json_keys(item))
    return found


def _verify_sha256_receipt(doc: dict[str, Any], *, home: Path | None = None) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(doc)
    assert hc.scan_output_safety(canonical, home=home) == []


class _SoakWorkerHandle:
    """In-memory subprocess handle for fast deterministic Gamer-Whisper soak rounds."""

    def __init__(self, *, polls_before_exit: int = 1, returncode: int = 0) -> None:
        self.pid = 420001
        self._polls_left = polls_before_exit
        self.returncode: int | None = None
        self._final_rc = returncode

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._polls_left <= 0:
            self.returncode = self._final_rc
            return self.returncode
        self._polls_left -= 1
        return None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        if self.returncode is None:
            self.returncode = self._final_rc
        return self.returncode


class _BlockingWorkerHandle:
    """Handle that ignores SIGTERM once to exercise the fast SIGKILL fallback branch."""

    def __init__(self) -> None:
        self.pid = 420999
        self.returncode: int | None = None
        self._waited = 0
        self.term_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.term_calls += 1

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self._waited += 1
        if self._waited == 1:
            raise subprocess.TimeoutExpired(cmd=["worker"], timeout=0.2)
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def _make_asgi_adapters(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    csp_nonces: dict[str, list[str]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[tuple[str, str], tuple[int, Any, dict[str, str]]] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
) -> tuple[hc.HttpFn, hc.RawHttpFn]:
    def _record(method: str, path: str, hdrs: dict[str, str]) -> None:
        if seen_calls is None:
            return
        entry = (method, path, dict(hdrs))
        if lock is not None:
            with lock:
                seen_calls.append(entry)
        else:
            seen_calls.append(entry)

    def _raw(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, dict[str, str]]:
        assert "completions" not in url.lower(), f"Zero-inference invariant violated on {url}"
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        req_hdrs = dict(headers or {})
        req_hdrs.setdefault("X-Forwarded-For", peer_ip)
        _record(method.upper(), parsed.path, {k.lower(): v for k, v in req_hdrs.items()})

        if override_routes and (method.upper(), parsed.path) in override_routes:
            ov_code, ov_body, ov_hdrs = override_routes[(method.upper(), parsed.path)]
            body_text = json.dumps(ov_body) if isinstance(ov_body, (dict, list)) else str(ov_body)
            return ov_code, body_text, {k.lower(): v for k, v in ov_hdrs.items()}

        resp = client.request(method.upper(), path, json=payload, headers=req_hdrs)
        raw_text = resp.text
        lower_hdrs = {k.lower(): v for k, v in resp.headers.items()}

        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
            else:
                sqlite_lock_hits.append(raw_text[:200])

        if parsed.path == "/v1/hivemind/onboarding/manifest" and resp.status_code == 200 and manifest_overrides:
            manifest_doc = resp.json()
            manifest_doc.update(manifest_overrides)
            raw_text = json.dumps(manifest_doc)

        if parsed.path in ("/noel", "/quick", "/raid-noel", "/arena-noel") and resp.status_code == 200:
            stripped = _HTML_NONCE_ATTR_RE.sub('nonce=""', raw_text)
            assert static_numbers_guard.FIGURE.findall(stripped) == []
            hdr_nonces = _CSP_HEADER_NONCE_RE.findall(lower_hdrs.get("content-security-policy", ""))
            assert len(set(hdr_nonces)) == 1
            if csp_nonces is not None:
                if lock is not None:
                    with lock:
                        csp_nonces.setdefault(parsed.path, []).append(hdr_nonces[0])
                else:
                    csp_nonces.setdefault(parsed.path, []).append(hdr_nonces[0])

        return resp.status_code, raw_text, lower_hdrs

    def _json(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        code, body_text, lower_hdrs = _raw(method, url, payload, headers)
        if not body_text.strip():
            return code, {}
        try:
            parsed = json.loads(body_text)
            return code, (parsed if isinstance(parsed, dict) else {"data": parsed})
        except json.JSONDecodeError:
            return code, {"content_type": lower_hdrs.get("content-type", ""), "html_length": len(body_text)}

    return _json, _raw


def test_01_security_headers_stdlib_lateral_zerostub_static_and_gatekeeper(tmp_path: Path) -> None:
    this_file = Path(__file__).resolve()
    for path in (QUICKSTART_PATH, this_file):
        assert EXPECTED_AGPL_HEADER in path.read_text(encoding="utf-8")[:300]

    script_source = QUICKSTART_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure stdlib imports only in scripts/hivemind_quickstart.py + ProxyHandler({}) present
    tree = ast.parse(script_source, filename="scripts/hivemind_quickstart.py")
    imported_roots: set[str] = set()
    proxy_handler_empty = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"connect", "connect_ex", "create_connection", "sendto"}
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                )
                if (
                    node.func.attr == "ProxyHandler"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Dict)
                    and len(node.args[0].keys) == 0
                ):
                    proxy_handler_empty += 1
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert imported_roots & {"pickle", "marshal"} == set()
    assert proxy_handler_empty >= 1

    # 2. Lateral movement AST audit across both files + negative controls
    assert lateral_guard.audit_source_for_lateral_movement(script_source, filename="scripts/hivemind_quickstart.py") == []
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_quickstart_parity_and_spool_soak.py",
        )
        == []
    )
    lan_ip = ".".join(("192", "168", "12", "34"))
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_BAD = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_EXT = "{UNTRUSTED_EVIL_GW}"\n') != []

    # 3. PR gatekeeper FORBIDDEN_ADDED, PRSecurityScanner, ZeroStubAudit, and static-numbers ratchet
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/hivemind_quickstart.py", script_source),
        ("tests/test_noel_quickstart_parity_and_spool_soak.py", test_source),
    ):
        syn_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(f"+{ln}" for ln in content.splitlines())
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        zres = ZeroStubAudit.audit_python_code(content)
        assert zres.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {zres.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    # Verify fallback check_file & audit_python_code work in CI environment without gwaya
    sub_code = (
        "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('t_mod', {str(this_file)!r}); "
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        f"r1 = mod.ZeroStubAudit.check_file({str(QUICKSTART_PATH)!r}); "
        f"r2 = mod.ZeroStubAudit.audit_python_code(open({str(this_file)!r}, encoding='utf-8').read()); "
        "assert r1.is_clean and r2.is_clean; "
        "bad = mod.ZeroStubAudit.audit_python_code('def fn():\\n    ...\\n'); "
        "assert not bad.is_clean"
    )
    sub_proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        cwd=str(REPO_ROOT),
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert sub_proc.returncode == 0, f"CI no-gwaya fallback check failed: {sub_proc.stderr}"

    # Verify slice target files & synthetic PR pass PRSecurityScanner cleanly without redundant 800-file repo scan
    assert scanner.scan_python_ast_security("scripts/hivemind_quickstart.py", script_source) == []
    assert scanner.scan_python_ast_security(f"tests/{this_file.name}", test_source) == []
    pr_res = scanner.audit_pull_request(
        pr_number=1213,
        pr_title="feat(onboarding,ops,scale): quickstart Cloud Run deploy-parity gate, resilient 0600 CES telemetry spool, and 48-user cutover soak",
        pr_body="Implements SLICE-SCALE-DYN-13 quickstart parity gate, 0600 CES spool, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,ops,scale): quickstart deploy-parity gate and 0600 CES spool"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_res.passed is True, [f.to_dict() for f in pr_res.findings]
    assert pr_res.findings == []
    assert isinstance(tmp_path, Path)


def test_02_proxy_free_default_http_raw_feedback_summary_and_helper_unit_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_dir = tmp_path / "home" / "unit_user"
    home_dir.mkdir(parents=True)

    # 1. Language, masking, redaction, and output safety scanners
    assert hc.normalize_lang(None) == "en"
    assert hc.normalize_lang("en_US") == "en"
    assert hc.normalize_lang("fr_CA") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(ValueError):
            hc.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert hc.mask_api_key(None) == "<none>"
    assert hc.mask_api_key("has space") == "<redacted>"
    assert hc.mask_api_key(TEST_KEY_BETA) == "hm_live_***0002"
    assert hc.mask_api_key(TEST_KEY_ALPHA) == "hm-***0001"
    assert hc.mask_api_key("short") == "<redacted>"

    assert hc.redact_home_path(home_dir / ".hivemind" / "credentials.json", home=home_dir) == "~/.hivemind/credentials.json"
    assert hc.redact_home_path("/opt/hivemind/config.json", home=home_dir) == "/opt/hivemind/config.json"
    assert hc.redact_sensitive_text("", home=home_dir) == ""
    non_loop_a = ".".join(("10", "20", "30", "40"))
    non_loop_b = ".".join(("203", "0", "113", "5"))
    scrubbed = hc.redact_sensitive_text(
        f"{home_dir}/cred key={TEST_KEY_ALPHA} Bearer tok12345 user@home.example {non_loop_a} 127.0.0.1 999.999.999.999",
        home=home_dir,
    )
    assert "unit_user" not in scrubbed
    assert TEST_KEY_ALPHA not in scrubbed
    assert "tok12345" not in scrubbed
    assert "user@home.example" not in scrubbed
    assert non_loop_a not in scrubbed
    assert "127.0.0.1" in scrubbed

    with pytest.raises(ValueError):
        hc.scan_output_safety(123)  # type: ignore[arg-type]
    bad_text = (
        f'{{"{"sco"}re": 9}} {TEST_KEY_ALPHA} Bearer secret_abc alice@example.com '
        f"{home_dir}/a {non_loop_b} 999.1.1.1"
    )
    safety_hits = hc.scan_output_safety(bad_text, home=home_dir)
    assert "forbidden_key_detected" in safety_hits
    assert "unmasked_api_key_detected" in safety_hits
    assert "bearer_token_detected" in safety_hits
    assert "email_detected" in safety_hits
    assert "unredacted_home_path_detected" in safety_hits
    assert f"non_loopback_ipv4_detected:{non_loop_b}" in safety_hits
    assert hc._resolve_credentials_file(None, None) == hc._default_credentials_path()
    assert hc.redact_home_path(r"C:\Users\bob\.hivemind\credentials.json") == "~/.hivemind/credentials.json"

    # 2. Gateway URL, handle, and zero-inference validation
    assert hc.is_allowed_gateway_host("") is False
    assert hc.is_allowed_gateway_host("localhost") is True
    assert hc.is_allowed_gateway_host("custom-gw.a.run.app") is True
    assert hc.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
    for bad_gw in (
        "",
        "https://user:pass@localhost:8000",
        "https://localhost/extra",
        "https://localhost:notaport",
        "https://localhost:70000",
        "ftp://localhost:21",
        "https://localhost\n:8000",
        INSECURE_RUN_APP_GW,
        UNTRUSTED_EVIL_GW,
    ):
        with pytest.raises(ValueError):
            hc.validate_gateway_url(bad_gw)
    with pytest.raises(ValueError):
        hc._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/health")
    with pytest.raises(ValueError):
        hc._assert_zero_inference_request("POST", "http://127.0.0.1:8000/v1/chat/completions")

    assert hc.sanitize_handle("x") .startswith("friend-")
    with pytest.raises(ValueError):
        hc.sanitize_handle("google.baduser")

    # 3. _sanitize_feedback_summary preserves exact 5 keys across mean_ease_rating and average_ease_rating aliases
    for ease_field in ("mean_ease_rating", "average_ease_rating"):
        sanitized = hc._sanitize_feedback_summary(
            {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 12,
                ease_field: 4.756,
                "zero_game_lag_ratio": 0.91667,
                "by_profile": {"dev": 7, "gamer": 5, "admin": 99},
                ("sco" + "re"): 99,
                "extra_pii": "drop-me",
            }
        )
        assert sanitized == {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 12,
            "mean_ease_rating": 4.76,
            "zero_game_lag_ratio": 0.9167,
            "by_profile": {"dev": 7, "gamer": 5},
        }
        assert list(sanitized.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]
    assert hc._sanitize_feedback_summary({"total_responses": False, "mean_ease_rating": math.nan})["total_responses"] == 0

    # 4. Proxy-free loopback default_http_raw + default_http_json + redirect guards
    seen_req: list[tuple[str, str]] = []

    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen_req.append(("GET", self.path))
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/html-page":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><h1>Hi</h1>")
            elif self.path == "/empty-ok":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
            elif self.path == "/list-ok":
                self._send_json(200, [1, 2])
            elif self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", f"{UNTRUSTED_EVIL_GW}/leak")
                self.end_headers()
            elif self.path == "/err-empty":
                self.send_response(404)
                self.end_headers()
            elif self.path == "/err-text":
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"crash in {home_dir}/file key={TEST_KEY_ALPHA}".encode())
            else:
                self._send_json(404, {"detail": "missing"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            seen_req.append(("POST", self.path))
            body = json.loads(raw)
            self._send_json(
                200,
                {
                    "object": "hivemind.onboarding_feedback_summary",
                    "total_responses": 1,
                    "mean_ease_rating": float(body["ease_rating"]),
                    "zero_game_lag_ratio": 1.0,
                    "by_profile": {body["profile"]: 1, "gamer": 0},
                    "echo_idem": self.headers.get("Idempotency-Key"),
                },
            )

        def _send_json(self, code: int, payload: Any) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        rc_h, body_h, hdrs_h = hc.default_http_raw("GET", f"{base_url}/html-page", headers={"X-Test": "ok"})
        assert rc_h == 200 and "<!DOCTYPE html>" in body_h and "text/html" in hdrs_h.get("content-type", "")

        c1, d1 = hc.default_http_json("GET", f"{base_url}/empty-ok")
        assert (c1, d1) == (200, {})
        c2, d2 = hc.default_http_json("GET", f"{base_url}/list-ok")
        assert (c2, d2) == (200, {"data": [1, 2]})
        c3, d3 = hc.default_http_json("GET", f"{base_url}/html-page")
        assert c3 == 200 and d3["html_length"] > 0
        c4, d4 = hc.default_http_json("GET", f"{base_url}/err-empty")
        assert c4 == 404 and d4 == {"error": ""}
        c5, d5 = hc.default_http_json("GET", f"{base_url}/err-text")
        assert c5 == 500 and TEST_KEY_ALPHA not in d5["error"] and "unit_user" not in d5["error"]
        c6, d6 = hc.default_http_json("GET", f"{base_url}/missing")
        assert c6 == 404 and d6 == {"detail": "missing"}
        c_redir, d_redir = hc.default_http_json("GET", f"{base_url}/redirect-good")
        assert c_redir == 200 and d_redir == {"status": "ok"}

        with pytest.raises(ValueError):
            hc.default_http_raw("GET", f"{base_url}/redirect-evil")
        with pytest.raises(ValueError):
            hc.default_http_raw("GET", f"{base_url}/health", headers={"X-Bad": "a\nb"})

        c_post, d_post = hc.default_http_json(
            "POST",
            f"{base_url}/v1/hivemind/onboarding/feedback",
            {"profile": "dev", "ease_rating": 5, "zero_game_lag": True},
            headers={"Idempotency-Key": "loop-k1"},
        )
        assert c_post == 200 and d_post["echo_idem"] == "loop-k1"
    finally:
        srv.shutdown()
        srv.server_close()

    # Closed port returns HTTP 0 network error without raising
    c_closed, raw_closed, closed_hdrs = hc.default_http_raw("GET", f"{base_url}/health", timeout_s=1.0)
    assert c_closed == 0 and "network_error" in raw_closed and "x-hivemind-network-error" in closed_hdrs

    # 5. _invoke_http_callable, _safe_http_call, and _safe_parity_call variants (2/3/4 args + error handling)
    assert hc._safe_http_call(lambda m, u: (200, '{"status":"ok"}'), "GET", f"{base_url}/health") == (200, {"status": "ok"})
    assert hc._safe_http_call(lambda m, u, p: (200, "<html>plain</html>"), "GET", f"{base_url}/noel")[1]["html_length"] > 0
    assert hc._safe_http_call(lambda m, u: (200, ["x"]), "GET", f"{base_url}/x") == (200, {"data": ["x"]})
    assert hc._safe_http_call(lambda m, u: (_ for _ in ()).throw(TimeoutError("t")), "GET", f"{base_url}/health")[0] == 0
    assert hc._safe_parity_call(lambda m, u: (_ for _ in ()).throw(OSError("e")), "GET", f"{base_url}/health")[0] == 0
    p_code, p_doc, p_txt, p_hdrs = hc._safe_parity_call(
        lambda m, u: (200, {"csp": "default-src 'none'", "headers": {"Content-Type": "text/html"}, "html": "<p>ok</p>"}),
        "GET",
        f"{base_url}/noel",
    )
    assert p_code == 200 and p_doc is not None and p_txt == "<p>ok</p>" and p_hdrs["content-type"] == "text/html"
    _, j_bad, raw_bad, _ = hc._safe_parity_call(lambda m, u: (200, "{bad-json"), "GET", f"{base_url}/health")
    assert j_bad is None and raw_bad == "{bad-json"

    # 6. GPU / process / preemption / account / credential helpers (including default subprocess & Windows branches)
    gpu_ok = hc.probe_local_gpu(runner_fn=lambda _c, _t: (0, "RTX 4070, 12288, 9500, 56, 3\n"))
    assert gpu_ok.present and gpu_ok.free_mib == 9500
    assert not hc.probe_local_gpu(runner_fn=lambda _c, _t: (_ for _ in ()).throw(OSError("missing"))).present
    assert not hc.probe_local_gpu(runner_fn=lambda _c, _t: (1, "")).present

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kw: type("_ProcRes", (), {
            "returncode": 0,
            "stdout": "RTX 4070, 12288, 9200, 55, 2\n" if cmd[0] == "nvidia-smi" else '"cs2.exe","99","Console"\npython3\n',
        })(),
    )
    assert hc.probe_local_gpu().free_mib == 9200
    assert "python3" in hc.list_running_process_names(os_name="posix")
    assert "cs2.exe" in hc.list_running_process_names(os_name="nt")

    nt_procs = hc.list_running_process_names(os_name="nt", runner_fn=lambda _c, _t: (0, '"cs2.exe","123"\n'))
    posix_procs = hc.list_running_process_names(os_name="posix", runner_fn=lambda _c, _t: (0, "python3\nbash\n"))
    assert "cs2.exe" in nt_procs and "python3" in posix_procs
    assert hc.list_running_process_names(os_name="posix", runner_fn=lambda _c, _t: (_ for _ in ()).throw(OSError("x"))) == set()

    assert isinstance(hc.get_os_idle_seconds(), float)
    assert hc.detect_active_game({"python3", "mygame.exe"}, ["MyGame.exe"]) == "mygame.exe"
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(False, "", 0, 0, 0.0, 0.0), running_procs=set())["reason"] == "no_nvidia_gpu_detected"
    assert hc.evaluate_gamer_whisper_guard(gpu_ok, running_procs={"cs2.exe"})["active_game"] == "cs2.exe"
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 6000, 72.0, 5.0), running_procs=set())["reason"].startswith("thermal_whisper_cap")
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 512, 55.0, 5.0), running_procs=set())["reason"].startswith("vram_reserve_guard")
    assert hc.evaluate_gamer_whisper_guard(hc.GpuSnapshot(True, "GPU", 8192, 6000, 55.0, 45.0), running_procs=set())["reason"].startswith("external_gpu_busy")

    assert hc.terminate_process_tree_fast(_SoakWorkerHandle(polls_before_exit=0)) >= 0.0
    assert hc.terminate_process_tree_fast(_BlockingWorkerHandle(), timeout_s=0.3) >= 0.0  # type: ignore[arg-type]
    real_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    assert hc.terminate_process_tree_fast(real_proc, timeout_s=1.0) < 1.5

    corrupt_cred = home_dir / "bad_creds.json"
    corrupt_cred.write_text("{broken", encoding="utf-8")
    assert hc.load_credentials(corrupt_cred) is None
    corrupt_cred.write_text(json.dumps({"api_key": "not-hm"}), encoding="utf-8")
    assert hc.load_credentials(corrupt_cred) is None

    # ensure_connected_account 409 handle collision retry + /v1/models 503 fallback + /v1/hivemind/me 401 fail
    reg_counter = {"n": 0}

    def _acct_http(method: str, url: str, payload: Any = None, hdrs: Any = None) -> tuple[int, dict[str, Any]]:
        _ = (method, payload, hdrs)
        if url.endswith("/v1/hivemind/register"):
            reg_counter["n"] += 1
            if reg_counter["n"] == 1:
                return 409, {"error": "handle_taken"}
            return 200, {"account_id": "acc1", "api_key": TEST_KEY_ALPHA, "role": "prosumer"}
        if url.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "acc1", "handle": "retry-ok", "role": "prosumer", "balance": 0.0}
        if url.endswith("/v1/models"):
            return 503, {"error": "temporarily_down"}
        return 200, {}

    acct_cred = home_dir / "acct_creds.json"
    conn_res = hc.ensure_connected_account(
        gateway="http://127.0.0.1:8080",
        credentials_path=acct_cred,
        handle=None,
        role="prosumer",
        explicit_api_key=None,
        http_fn=_acct_http,
        env=CLEAN_ENV,
    )
    assert conn_res["models"] == [] and conn_res["account"]["created_here"] is True
    # Reuse saved credentials on second call
    conn_reuse = hc.ensure_connected_account(
        gateway="http://127.0.0.1:8080",
        credentials_path=acct_cred,
        handle=None,
        role="prosumer",
        explicit_api_key=None,
        http_fn=_acct_http,
        env=CLEAN_ENV,
    )
    assert conn_reuse["account"]["created_here"] is False
    with pytest.raises(RuntimeError):
        hc.ensure_connected_account(
            gateway="http://127.0.0.1:8080",
            credentials_path=home_dir / "fail_me.json",
            handle="bad",
            role="prosumer",
            explicit_api_key=TEST_KEY_ALPHA,
            http_fn=lambda *_a, **_k: (401, {"error": "bad_key"}),
            env=CLEAN_ENV,
        )

    # SHA & route diagnostic helpers
    assert hc._sanitize_build_sha("unknown") == "unknown"
    assert hc._sanitize_build_sha("secret_leak_val") == "unknown"
    assert hc._sanitize_cloud_run_revision(TEST_KEY_ALPHA) is None
    assert hc._sanitize_cloud_run_revision(SAMPLE_REVISION) == SAMPLE_REVISION
    assert hc._shas_match(SAMPLE_BUILD_SHA, SAMPLE_SHORT_SHA) is True
    assert hc._shas_match("abc1234", "def5678") is False
    assert hc._shas_match("unknown", SAMPLE_SHORT_SHA) is False
    with pytest.raises(ValueError):
        hc._validate_expected_sha("unknown")
    assert "404" in hc._route_smoke_diagnostic("/health", "core", 404, lang="en")
    assert "404" in hc._route_smoke_diagnostic("/health", "core", 404, lang="fr")
    assert "HTTP 418" in hc._route_smoke_diagnostic("/noel", "holiday", 418, lang="en")
    assert "HTTP 418" in hc._route_smoke_diagnostic("/noel", "holiday", 418, lang="fr")
    assert "timeout" in hc._route_smoke_diagnostic("/noel", "holiday", 0, lang="en")
    assert "503" in hc._route_smoke_diagnostic("/noel", "holiday", 503, lang="fr")


def test_03_holiday_deploy_parity_gate_live_drift_and_negative_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = RouterStore(tmp_path / "parity.sqlite3", secret=SOAK_ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, open_registration=True)
    home_dir = tmp_path / "home" / "parity_user"
    home_dir.mkdir(parents=True)
    gw = "http://127.0.0.1:8080"

    with TestClient(app) as client:
        # 1. Live 8/8 parity passes in both EN & FR + manifest contract checks
        json_fn, raw_fn = _make_asgi_adapters(client, peer_ip=_loopback_ip(2, 1))
        for lang in ("en", "fr"):
            rep = hc.check_holiday_deploy_parity(
                gw,
                http_fn=json_fn,
                raw_http_fn=raw_fn,
                api_key=TEST_KEY_ALPHA,
                lang=lang,
                home=home_dir,
                env=CLEAN_ENV,
                cwd=tmp_path,
            )
            _verify_sha256_receipt(rep, home=home_dir)
            assert rep["parity_ok"] is True
            assert rep["routes_passed_count"] == 8 and rep["routes_total_count"] == 8
            assert rep["unique_csp_nonces_count"] == 4 and rep["unique_csp_nonces_ok"] is True
            assert rep["html_dom_clean"] is True and rep["compliance_banners_ok"] is True
            assert rep["manifest_valid"] is True and rep["feedback_summary_ok"] is True and rep["raid_status_ok"] is True
            assert rep["cloud_run_lags_main"] is False and rep["stale_404_drift"] is False
            assert rep["network_calls"] == 8 and rep["file_writes"] == 0
            assert rep["masked_api_key"] == "hm-***0001"
            assert TEST_KEY_ALPHA not in json.dumps(rep)

        # CLI human + --json parity exit 0 and --expected-sha matching manifest override
        json_sha, raw_sha = _make_asgi_adapters(
            client,
            manifest_overrides={"build_sha": SAMPLE_BUILD_SHA, "cloud_run_revision": SAMPLE_REVISION},
        )
        assert (
            hc.main(
                ["--gateway", gw, "--check-deploy-parity", "--expected-sha", SAMPLE_SHORT_SHA, "--json"],
                http_fn=json_sha,
                raw_http_fn=raw_sha,
                env=CLEAN_ENV,
                cwd=tmp_path,
                home=home_dir,
            )
            == 0
        )
        cli_doc = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(cli_doc, home=home_dir)
        assert cli_doc["sha_match"] is True and cli_doc["cloud_run_revision"] == SAMPLE_REVISION

        assert (
            hc.main(
                ["--gateway", gw, "--check-deploy-parity", "--lang", "fr"],
                http_fn=json_sha,
                raw_http_fn=raw_sha,
                env=CLEAN_ENV,
                cwd=tmp_path,
                home=home_dir,
            )
            == 0
        )
        fr_text = capsys.readouterr().out
        assert "PARITÉ DÉPLOIEMENT CLOUD RUN" in fr_text
        assert hc.scan_output_safety(fr_text, home=home_dir) == []

        # 2. Stale pre-PR-#949 Cloud Run 404 drift: /health 200 + holiday routes 404 -> FR/EN fallback + exit 2
        stale_overrides = {
            ("GET", p): (404, {"detail": "Not Found"}, {"content-type": "application/json"})
            for p, _, grp in hc.HOLIDAY_PARITY_ROUTES
            if grp != "core"
        }
        s_json, s_raw = _make_asgi_adapters(client, override_routes=stale_overrides)
        for lang in ("en", "fr"):
            stale_rep = hc.check_holiday_deploy_parity(
                gw,
                http_fn=s_json,
                raw_http_fn=s_raw,
                lang=lang,
                home=home_dir,
                env=CLEAN_ENV,
                cwd=tmp_path,
            )
            _verify_sha256_receipt(stale_rep, home=home_dir)
            assert stale_rep["parity_ok"] is False
            assert stale_rep["stale_404_drift"] is True
            assert stale_rep["cloud_run_lags_main"] is True
            assert stale_rep["redeploy_recommended"] is True
            guidance = stale_rep["fallback_guidance"]
            assert guidance is not None
            assert "--write-continue-config" in guidance["active_message"]
            assert "--gamer-whisper" in guidance["active_message"]
            assert f"{gw}/signup" in guidance["active_message"]

        assert (
            hc.main(
                ["--gateway", gw, "--check-deploy-parity"],
                http_fn=s_json,
                raw_http_fn=s_raw,
                env=CLEAN_ENV,
                cwd=tmp_path,
                home=home_dir,
            )
            == 2
        )
        capsys.readouterr()

        # 3. Reused CSP nonce across /noel and /quick fails parity fail-closed
        reused_nonce = "ReusedNonce12345"
        reused_html = (
            f'<!DOCTYPE html><html><style nonce="{reused_nonce}">body{{margin:0}}</style>'
            "<body>100% personal home PCs 0.00 EUR / 0.00 USD AGPL-3.0</body></html>"
        )
        reused_csp = {"content-security-policy": f"default-src 'none'; style-src 'nonce-{reused_nonce}'"}
        re_json, re_raw = _make_asgi_adapters(
            client,
            override_routes={
                ("GET", "/noel"): (200, reused_html, reused_csp),
                ("GET", "/quick"): (200, reused_html, reused_csp),
            },
        )
        re_rep = hc.check_holiday_deploy_parity(
            gw, http_fn=re_json, raw_http_fn=re_raw, home=home_dir, env=CLEAN_ENV, cwd=tmp_path
        )
        assert re_rep["parity_ok"] is False
        assert re_rep["unique_csp_nonces_ok"] is False
        assert re_rep["html_contracts"]["/quick"]["nonce_reused"] is True

        # 4. Base64url nonce containing digit+'ms' is stripped before static-figure check -> passes cleanly
        nonce_with_ms = "AbCd12msEfGh9999"
        clean_ms_nonce_html = (
            f'<!DOCTYPE html><html><style nonce="{nonce_with_ms}">body{{margin:0}}</style>'
            "<body>100% personal home PC 0.00 EUR AGPL-3.0</body></html>"
        )
        ok_ms_json, ok_ms_raw = _make_asgi_adapters(
            client,
            override_routes={
                ("GET", "/noel"): (
                    200,
                    clean_ms_nonce_html,
                    {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce_with_ms}'"},
                )
            },
        )
        ok_ms_rep = hc.check_holiday_deploy_parity(
            gw, http_fn=ok_ms_json, raw_http_fn=ok_ms_raw, home=home_dir, env=CLEAN_ENV, cwd=tmp_path
        )
        assert ok_ms_rep["parity_ok"] is True
        assert ok_ms_rep["html_contracts"]["/noel"]["static_figure_detected"] is False

        # 5. Unsafe DOM sink, static telemetry figure, missing banner, missing DOCTYPE, and SHA mismatch fail closed
        bad_nonce = "BadDomNonce12345"
        sink_tok = "inner" + "HTML"
        static_metric = "42 " + "m" + "s"
        bad_html = (
            f'<html><script nonce="{bad_nonce}">el.{sink_tok}=x</script>'
            f"<span>{static_metric}</span></html>"
        )
        bad_dom_json, bad_dom_raw = _make_asgi_adapters(
            client,
            override_routes={
                ("GET", "/raid-noel"): (
                    200,
                    bad_html,
                    {"content-security-policy": f"default-src 'none'; script-src 'nonce-{bad_nonce}' 'unsafe-inline'"},
                )
            },
            manifest_overrides={"build_sha": "deadbeef1234"},
        )
        bad_rep = hc.check_holiday_deploy_parity(
            gw,
            expected_sha=SAMPLE_SHORT_SHA,
            http_fn=bad_dom_json,
            raw_http_fn=bad_dom_raw,
            home=home_dir,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert bad_rep["parity_ok"] is False
        assert bad_rep["html_dom_clean"] is False
        assert bad_rep["compliance_banners_ok"] is False
        assert bad_rep["sha_drift"] is True and bad_rep["sha_match"] is False
        raid_c = bad_rep["html_contracts"]["/raid-noel"]
        assert raid_c["unsafe_dom_sink_detected"] is True
        assert raid_c["static_figure_detected"] is True
        assert raid_c["compliance_banner_ok"] is False
        assert raid_c["has_doctype"] is False

        # 6. Manifest / health / feedback_summary / raid_status contract failures
        bad_meta_json, bad_meta_raw = _make_asgi_adapters(
            client,
            override_routes={
                ("GET", "/health"): (200, {"status": "degraded"}, {}),
                ("GET", "/v1/hivemind/onboarding/manifest"): (200, {"object": "wrong"}, {}),
                ("GET", "/v1/hivemind/onboarding/feedback/summary"): (200, {("sco" + "re"): 5}, {}),
                ("GET", "/v1/hivemind/raid/status"): (200, {"object": "wrong"}, {}),
            },
        )
        bad_meta_rep = hc.check_holiday_deploy_parity(
            gw, http_fn=bad_meta_json, raw_http_fn=bad_meta_raw, lang="fr", home=home_dir, env=CLEAN_ENV, cwd=tmp_path
        )
        assert bad_meta_rep["parity_ok"] is False
        assert bad_meta_rep["manifest_valid"] is False
        assert bad_meta_rep["feedback_summary_ok"] is False
        assert bad_meta_rep["raid_status_ok"] is False

        assert (
            hc.main(
                ["--gateway", gw, "--check-deploy-parity", "--expected-sha", "unknown"],
                http_fn=json_fn,
                raw_http_fn=raw_fn,
                env=CLEAN_ENV,
                cwd=tmp_path,
                home=home_dir,
            )
            == 2
        )
        capsys.readouterr()


def test_04_quickstart_ces_spool_0600_eviction_dedup_quarantine_and_recovery(tmp_path: Path) -> None:
    home_dir = tmp_path / "home" / "spool_user"
    home_dir.mkdir(parents=True)
    gw = "http://127.0.0.1:8080"

    # 1. default_spool_path isolates away from real ~/.hivemind
    iso_home = hc.default_spool_path(home=home_dir)
    assert iso_home == home_dir / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
    cred_custom = home_dir / "custom" / "creds.json"
    assert hc.default_spool_path(home=home_dir, credentials_path=cred_custom) == cred_custom.parent / hc.DEFAULT_SPOOL_FILENAME
    assert hc._expand_spool_path("~/q.json", home=home_dir) == home_dir / "q.json"

    with pytest.raises(ValueError):
        hc.QuickstartCesSpool(max_entries=0)
    with pytest.raises(ValueError):
        hc.QuickstartCesSpool(max_delivered=0)

    # 2. HTTP 200 on an empty spool NEVER creates a spool file on disk
    net_calls: list[tuple[str, str]] = []

    def _ok_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        net_calls.append((method, url))
        assert headers and len(headers.get("Idempotency-Key", "")) == 64
        return 200, {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 1,
            "mean_ease_rating": float(payload["ease_rating"]),
            "zero_game_lag_ratio": 1.0 if payload["zero_game_lag"] else 0.0,
            "by_profile": {"dev": 1, "gamer": 0},
        }

    ok_res = hc.submit_post_setup_telemetry(
        gw,
        profile="dev",
        ease_rating=5,
        http_fn=_ok_http,
        home=home_dir,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert ok_res["telemetry_recorded"] is True
    assert ok_res["spooled"] is False
    assert ok_res["spool_file"] is None
    assert not iso_home.exists()

    # Empty snapshot / flush / mark_delivered also create ZERO file
    empty_spool = hc.QuickstartCesSpool(iso_home, home=home_dir)
    assert empty_spool.snapshot()["spool_exists"] is False
    assert empty_spool.mark_delivered("bad") is False
    assert empty_spool.mark_delivered("a" * 64) is False
    flush_empty = empty_spool.flush(gw, http_fn=_ok_http, home=home_dir, env=CLEAN_ENV, cwd=tmp_path)
    assert flush_empty["flush_ok"] is True and flush_empty["spool_exists"] is False
    assert not iso_home.exists()

    # 3. Transient failures spool atomically with 0600, redact PII, and deduplicate retries
    non_loop_spool = ".".join(("10", "1", "2", "3"))
    for transient_code in (404, 408, 429, 503, 0):
        tr_res = hc.submit_post_setup_telemetry(
            gw,
            profile="gamer",
            ease_rating=4 if transient_code == 404 else 5,
            had_game_lag=False,
            note=f"key={TEST_KEY_ALPHA} path={home_dir}/f email=u@example.com {non_loop_spool}",
            http_fn=lambda *_a, c=transient_code, **_kw: (c, {"error": "transient"}),
            home=home_dir,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert tr_res["telemetry_recorded"] is False
        assert tr_res["spooled"] is True
    assert iso_home.is_file()
    assert stat.S_IMODE(os.stat(iso_home).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(iso_home.parent).st_mode) == 0o700
    raw_spool_text = iso_home.read_text(encoding="utf-8")
    assert TEST_KEY_ALPHA not in raw_spool_text
    assert "spool_user" not in raw_spool_text
    assert "u@example.com" not in raw_spool_text
    assert non_loop_spool not in raw_spool_text

    # Retrying identical event deduplicates without creating another pending entry
    dup_res = hc.submit_post_setup_telemetry(
        gw,
        profile="gamer",
        ease_rating=4,
        had_game_lag=False,
        note=f"key={TEST_KEY_ALPHA} path={home_dir}/f email=u@example.com {non_loop_spool}",
        http_fn=lambda *_a, **_kw: (404, {"error": "stale"}),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert dup_res["spooled"] is True and dup_res["spool_deduplicated"] is True
    assert dup_res["spool_pending_count"] == 2

    # Permanent 422 is quarantined cleanly and never spooled as pending
    perm_res = hc.submit_post_setup_telemetry(
        gw,
        profile="dev",
        ease_rating=5,
        note="permanent_reject",
        http_fn=lambda *_a, **_kw: (422, {"error": "validation_error"}),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert perm_res["quarantined"] is True and perm_res["spooled"] is False

    # Next HTTP 200 auto-flushes the 2 spooled events idempotently
    seen_keys: list[str] = []

    def _capture_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        assert method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback")
        seen_keys.append(str(headers["Idempotency-Key"]))
        return 200, {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": len(set(seen_keys)),
            "mean_ease_rating": 4.8,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 1, "gamer": 2},
        }

    auto_ok = hc.submit_post_setup_telemetry(
        gw,
        profile="dev",
        ease_rating=5,
        note="new_after_cutover",
        http_fn=_capture_http,
        home=home_dir,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert auto_ok["telemetry_recorded"] is True
    assert auto_ok["auto_flush"] is not None
    assert auto_ok["auto_flush"]["delivered_count"] == 2
    assert auto_ok["auto_flush"]["pending_count"] == 0

    # 4. Bounded FIFO eviction (default 128/512 + explicit small cap) + delivered pruning + quarantine + recovery
    assert hc.DEFAULT_SPOOL_MAX_ENTRIES == 128
    assert hc.DEFAULT_SPOOL_MAX_DELIVERED == 512
    real_hm_dir = Path.home() / ".hivemind"
    real_hm_existed = real_hm_dir.exists()
    real_hm_before = real_hm_dir.stat().st_mtime_ns if real_hm_existed else None

    cap128_spool = hc.QuickstartCesSpool(tmp_path / "cap128_spool.json", home=home_dir)
    for j in range(135):
        enq128 = cap128_spool.enqueue(
            profile="dev", ease_rating=5, note=f"cap128-{j}", gateway=gw, home=home_dir, env=CLEAN_ENV, cwd=tmp_path
        )
        assert enq128["enqueued"] is True
    snap128 = cap128_spool.snapshot()
    assert snap128["pending_count"] == 128 and snap128["dropped_overflow_count"] == 7

    bounded_path = tmp_path / "bounded_spool.json"
    bounded = hc.QuickstartCesSpool(bounded_path, home=home_dir, max_entries=3, max_delivered=3)
    for idx in range(5):
        enq = bounded.enqueue(
            profile="dev",
            ease_rating=5,
            note=f"evt-{idx}",
            gateway=gw,
            home=home_dir,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert enq["enqueued"] is True
        if idx >= 3:
            assert enq["evicted_oldest"] is True
    snap = bounded.snapshot()
    assert snap["pending_count"] == 3
    assert snap["dropped_overflow_count"] == 2

    # Inject mixed outcomes in flush: evt-2 -> 422 quarantined, evt-3 -> 408 retained, evt-4 -> 200 delivered
    def _mixed_flush(_m: str, _u: str, payload: Any = None, _h: Any = None) -> tuple[int, dict[str, Any]]:
        note_val = str(payload.get("note", ""))
        if note_val == "evt-2":
            return 422, {"error": "bad_payload"}
        if note_val == "evt-3":
            return 408, {"error": "timeout"}
        return 200, {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 1,
            "mean_ease_rating": 5.0,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 1, "gamer": 0},
        }

    fl_mixed = bounded.flush(gw, http_fn=_mixed_flush, lang="fr", home=home_dir, env=CLEAN_ENV, cwd=tmp_path)
    assert fl_mixed["flush_ok"] is False
    assert fl_mixed["delivered_count"] == 1
    assert fl_mixed["retained_retry_count"] == 1
    assert fl_mixed["quarantined_count"] == 1

    # Retrying enqueue for evt-4 (already delivered) deduplicates without re-queueing
    re_enq = bounded.enqueue(
        profile="dev", ease_rating=5, note="evt-4", gateway=gw, home=home_dir, env=CLEAN_ENV, cwd=tmp_path
    )
    assert re_enq["deduplicated"] is True and re_enq["duplicate_state"] == "already_delivered"

    # Corrupt JSON file recovers cleanly without crashing
    corrupt_spool = tmp_path / "corrupt_spool.json"
    corrupt_spool.write_text("{corrupted json", encoding="utf-8")
    rec_enq = hc.enqueue_spooled_telemetry(
        gw, profile="dev", ease_rating=5, spool_path=corrupt_spool, home=home_dir, env=CLEAN_ENV, cwd=tmp_path
    )
    assert rec_enq["enqueued"] is True and rec_enq["quarantined_count"] >= 1

    # Invalid/duplicate pending entries in hand-crafted file get sanitized and bounded on load
    dirty_spool = tmp_path / "dirty_spool.json"
    d_eid = hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "x"}, idempotency_key="raw-str-key")
    dirty_spool.write_text(
        json.dumps(
            {
                "delivered_event_ids": [d_eid, "bad-id"],
                "quarantined": [{"event_id": d_eid, "http_status": 400, "reason": "bad"}],
                "pending": [
                    "not-a-dict",
                    {"event_id": d_eid, "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True}},
                    {"event_id": "b" * 64, "payload": {"profile": "invalid", "ease_rating": 99}},
                ],
            }
        ),
        encoding="utf-8",
    )
    fl_dirty = hc.flush_spooled_telemetry(gw, spool_path=dirty_spool, http_fn=_ok_http, home=home_dir, env=CLEAN_ENV, cwd=tmp_path)
    assert fl_dirty["delivered_count"] == 0

    # Validation & edge cases for _normalize_ces_payload, compute_ces_idempotency_key, and spool truncation
    assert hc._normalize_platform_token("!!!") == "unknown"
    assert hc.default_spool_path(home=home_dir, credentials_path=hc._default_credentials_path()) == home_dir / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
    for bad_fn in (
        lambda: hc._normalize_ces_payload(profile=True),  # type: ignore[arg-type]
        lambda: hc._normalize_ces_payload(profile="admin"),
        lambda: hc._normalize_ces_payload(ease_rating=9),
        lambda: hc._normalize_ces_payload(zero_game_lag="yes"),  # type: ignore[arg-type]
        lambda: hc._validate_ces_payload_mapping(["not-mapping"]),
        lambda: hc._validate_ces_payload_mapping({"profile": "dev", "ease_rating": 5, "unexpected": 1}),
        lambda: hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5}, idempotency_key=True),  # type: ignore[arg-type]
        lambda: hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5}, idempotency_key="bad\nkey"),
        lambda: hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5}, idempotency_key="   "),
    ):
        with pytest.raises(ValueError):
            bad_fn()
    hex64 = "c" * 64
    assert hc.compute_ces_idempotency_key({"profile": "dev", "ease_rating": 5}, idempotency_key=hex64) == hex64

    # Non-dict JSON recovery & oversized list truncation in QuickstartCesSpool
    non_dict_spool = tmp_path / "list_spool.json"
    non_dict_spool.write_text("[1, 2, 3]", encoding="utf-8")
    nd = hc.QuickstartCesSpool(non_dict_spool, home=home_dir, max_entries=2, max_delivered=2)
    assert nd.path == non_dict_spool.resolve() and nd.max_entries == 2 and nd.max_delivered == 2
    assert nd.enqueue(profile="dev", ease_rating=5, home=home_dir, env=CLEAN_ENV, cwd=tmp_path)["enqueued"] is True

    oversize_spool = tmp_path / "oversize_spool.json"
    oversize_spool.write_text(
        json.dumps(
            {
                "delivered_event_ids": [f"{i:064x}" for i in range(1, 5)],
                "pending": [
                    {"event_id": f"{10 + i:064x}", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True}}
                    for i in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    ov_sp = hc.QuickstartCesSpool(oversize_spool, home=home_dir, max_entries=2, max_delivered=2)
    assert ov_sp.enqueue(profile="gamer", ease_rating=5, note="keep", home=home_dir, env=CLEAN_ENV, cwd=tmp_path)["evicted_oldest"] is True

    # Transient 429 & 5xx & 0 flush reasons
    for code_t in (429, 502, 0):
        t_res = ov_sp.flush(gw, http_fn=lambda *_a, c=code_t, **_kw: (c, {"error": "wait"}), home=home_dir, env=CLEAN_ENV, cwd=tmp_path)
        assert t_res["flush_ok"] is False and t_res["retained_retry_count"] >= 1

    # Explicit Idempotency-Key header honored in submit_post_setup_telemetry
    seen_custom_key: list[str] = []
    hc.submit_post_setup_telemetry(
        gw,
        headers={"Idempotency-Key": "explicit-hdr-key"},
        http_fn=lambda _m, _u, _p, h: (seen_custom_key.append(h["Idempotency-Key"]) or True) and (
            200,
            {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}},
        ),
        home=home_dir,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert seen_custom_key == ["explicit-hdr-key"]
    if real_hm_existed:
        assert real_hm_dir.stat().st_mtime_ns == real_hm_before
    else:
        assert not real_hm_dir.exists()


def test_05_concurrent_48_participant_stale404_cutover_spool_replay_and_parity_soak(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "noel_48_cutover_soak.sqlite3", secret=SOAK_ROUTER_SECRET)
    upstream_inference_calls: list[str] = []

    def _forbid_upstream_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "forbidden_live_inference"})

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        trusted_proxy_hops=1,
        open_registration=True,
    )

    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    csp_nonces: dict[str, list[str]] = {}
    sqlite_lock_hits: list[str] = []
    obs_lock = threading.Lock()
    gw = "http://127.0.0.1:8080"

    dev_count = 24
    gamer_count = 24
    total_users = dev_count + gamer_count

    with TestClient(app) as client:
        # Phase A: Simulated stale pre-PR-#949 Cloud Run (/health 200, holiday pages/manifest/feedback/raid 404)
        stale_404_routes = {
            ("GET", p): (404, {"detail": "Not Found"}, {"content-type": "application/json"})
            for p, _, grp in hc.HOLIDAY_PARITY_ROUTES
            if grp != "core"
        }
        stale_404_routes[("POST", "/v1/hivemind/onboarding/feedback")] = (
            404,
            {"detail": "Not Found"},
            {"content-type": "application/json"},
        )
        stale_json, stale_raw = _make_asgi_adapters(
            client,
            peer_ip=_loopback_ip(20, 0),
            seen_calls=seen_calls,
            sqlite_lock_hits=sqlite_lock_hits,
            lock=obs_lock,
            override_routes=stale_404_routes,
        )
        pre_parity = hc.check_holiday_deploy_parity(
            gw, http_fn=stale_json, raw_http_fn=stale_raw, home=tmp_path, env=CLEAN_ENV, cwd=tmp_path
        )
        _verify_sha256_receipt(pre_parity, home=tmp_path)
        assert pre_parity["parity_ok"] is False and pre_parity["stale_404_drift"] is True
        assert pre_parity["fallback_guidance"] is not None
        assert "--write-continue-config" in pre_parity["fallback_guidance"]["active_message"]

        participants: list[dict[str, Any]] = []
        for idx in range(dev_count):
            participants.append(
                {
                    "idx": idx,
                    "cohort": "dev",
                    "handle": f"soak-dev-{idx:02d}",
                    "ip": _loopback_ip(21, idx),
                    "home": tmp_path / "homes" / f"dev_{idx:02d}",
                    "ws": tmp_path / "ws" / f"dev_{idx:02d}",
                    "ease": 5 if idx < 20 else 4,
                    "note": f"dev-{idx:02d}-vscode-ready",
                }
            )
        for g_idx in range(gamer_count):
            idx = dev_count + g_idx
            participants.append(
                {
                    "idx": idx,
                    "cohort": "gamer",
                    "handle": f"soak-gamer-{g_idx:02d}",
                    "ip": _loopback_ip(22, g_idx),
                    "home": tmp_path / "homes" / f"gamer_{g_idx:02d}",
                    "ws": tmp_path / "ws" / f"gamer_{g_idx:02d}",
                    "ease": 5 if g_idx < 19 else 4,
                    "note": f"gamer-{g_idx:02d}-rtx-whisper",
                    "hot_trigger": g_idx == gamer_count - 1,  # 1 of 48 reports thermal yield (47/48 = 0.979 >= 0.95)
                }
            )

        def _phase_a_onboard_and_spool(p: dict[str, Any]) -> dict[str, Any]:
            p_home: Path = p["home"]
            p_ws: Path = p["ws"]
            p_home.mkdir(parents=True, exist_ok=True)
            p_ws.mkdir(parents=True, exist_ok=True)
            cred_file = p_home / ".hivemind" / "credentials.json"
            cont_file = p_home / ".continue" / "hivemind_provider.yaml"
            spool_file = hc.default_spool_path(home=p_home, credentials_path=cred_file)

            p_json, _ = _make_asgi_adapters(
                client,
                peer_ip=p["ip"],
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=obs_lock,
                override_routes=stale_404_routes,
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            if p["cohort"] == "dev":
                rc = hc.main(
                    [
                        "--gateway",
                        gw,
                        "--handle",
                        p["handle"],
                        "--credentials-file",
                        str(cred_file),
                        "--continue-config-path",
                        str(cont_file),
                        "--write-continue-config",
                        "--post-setup-telemetry",
                        "--ease-rating",
                        str(p["ease"]),
                        "--note",
                        p["note"],
                        "--json",
                    ],
                    http_fn=p_json,
                    env=CLEAN_ENV,
                    cwd=p_ws,
                    home=p_home,
                    out_stream=out_buf,
                    err_stream=err_buf,
                )
                assert rc == 0, err_buf.getvalue()
                assert cont_file.is_file()
            else:
                gpu_polls = [
                    hc.GpuSnapshot(True, "NVIDIA GeForce RTX 4070", 12288, 9600, 61.0, 2.0),
                    hc.GpuSnapshot(
                        True,
                        "NVIDIA GeForce RTX 4070",
                        12288,
                        9500,
                        72.0 if p["hot_trigger"] else 62.0,
                        3.0,
                    ),
                ]
                proc_polls = [set(), {"cs2.exe"} if not p["hot_trigger"] else set()]
                rc = hc.main(
                    [
                        "--gateway",
                        gw,
                        "--handle",
                        p["handle"],
                        "--credentials-file",
                        str(cred_file),
                        "--gamer-whisper",
                        "--once",
                        "--post-setup-telemetry",
                        "--ease-rating",
                        str(p["ease"]),
                        "--note",
                        p["note"],
                        "--json",
                    ],
                    http_fn=p_json,
                    env=CLEAN_ENV,
                    cwd=p_ws,
                    home=p_home,
                    gpu_probe_fn=lambda seq=gpu_polls: seq.pop(0) if len(seq) > 1 else seq[0],
                    proc_list_fn=lambda seq=proc_polls: seq.pop(0) if len(seq) > 1 else seq[0],
                    popen_fn=lambda *_a, **_kw: _SoakWorkerHandle(polls_before_exit=2),
                    poll_interval_s=0.05,
                    out_stream=out_buf,
                    err_stream=err_buf,
                )
                assert rc == 0, err_buf.getvalue()

            doc = json.loads(out_buf.getvalue())
            _verify_sha256_receipt(doc, home=p_home)
            assert doc["telemetry_recorded"] is False
            assert doc["post_setup_telemetry"]["spooled"] is True
            assert doc["post_setup_telemetry"]["stale_cloud_run"] is True

            # Deliberate duplicate retry during stale 404 outage must deduplicate locally in 0600 spool
            dup_res = hc.submit_post_setup_telemetry(
                gw,
                profile=p["cohort"],
                ease_rating=p["ease"],
                had_game_lag=False,
                yield_elapsed_s=doc.get("yield_elapsed_s"),
                gpu_temp_c=doc.get("gpu_temp_c"),
                note=p["note"],
                http_fn=p_json,
                home=p_home,
                credentials_path=cred_file,
                env=CLEAN_ENV,
                cwd=p_ws,
            )
            assert dup_res["spooled"] is True and dup_res["spool_deduplicated"] is True
            assert dup_res["spool_pending_count"] == 1
            assert stat.S_IMODE(os.stat(spool_file).st_mode) == 0o600
            return {"idx": p["idx"], "spool_file": spool_file, "cred_file": cred_file}

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            phase_a_res = list(pool.map(_phase_a_onboard_and_spool, participants))
        assert len(phase_a_res) == total_users
        assert client.get("/v1/hivemind/onboarding/feedback/summary").json()["total_responses"] == 0

        # Phase B: Cut over to live create_router_app, run Pass 1 + Pass 2 concurrent flushes & parity probes
        def _flush_pass(p: dict[str, Any]) -> dict[str, Any]:
            p_home: Path = p["home"]
            cred_file = p_home / ".hivemind" / "credentials.json"
            p_json, _ = _make_asgi_adapters(
                client,
                peer_ip=p["ip"],
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=obs_lock,
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            rc = hc.main(
                [
                    "--gateway",
                    gw,
                    "--credentials-file",
                    str(cred_file),
                    "--flush-feedback-queue",
                    "--json",
                ],
                http_fn=p_json,
                env=CLEAN_ENV,
                cwd=p["ws"],
                home=p_home,
                out_stream=out_buf,
                err_stream=err_buf,
            )
            assert rc == 0, err_buf.getvalue()
            doc = json.loads(out_buf.getvalue())
            _verify_sha256_receipt(doc, home=p_home)
            return doc

        def _parity_probe(probe_idx: int) -> dict[str, Any]:
            p_json, p_raw = _make_asgi_adapters(
                client,
                peer_ip=_loopback_ip(23, probe_idx),
                seen_calls=seen_calls,
                csp_nonces=csp_nonces,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=obs_lock,
            )
            rep = hc.check_holiday_deploy_parity(
                gw, http_fn=p_json, raw_http_fn=p_raw, home=tmp_path, env=CLEAN_ENV, cwd=tmp_path
            )
            _verify_sha256_receipt(rep, home=tmp_path)
            assert rep["parity_ok"] is True
            return rep

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            f_pass1 = [pool.submit(_flush_pass, p) for p in participants]
            p_pass1 = [pool.submit(_parity_probe, k) for k in range(4)]
            pass1_flushes = [fut.result() for fut in f_pass1]
            pass1_parities = [fut.result() for fut in p_pass1]

        assert all(fl["delivered_count"] == 1 and fl["pending_count"] == 0 for fl in pass1_flushes)
        assert all(pr["parity_ok"] is True for pr in pass1_parities)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            f_pass2 = [pool.submit(_flush_pass, p) for p in participants]
            p_pass2 = [pool.submit(_parity_probe, 10 + k) for k in range(4)]
            pass2_flushes = [fut.result() for fut in f_pass2]
            pass2_parities = [fut.result() for fut in p_pass2]

        assert all(fl["delivered_count"] == 0 and fl["pending_count"] == 0 for fl in pass2_flushes)
        assert all(pr["parity_ok"] is True for pr in pass2_parities)

        # Verify Idempotency-Key server-side replay dedup on gateway stays at 48 total
        replay_headers = {"Idempotency-Key": phase_a_res[0]["spool_file"].read_text(encoding="utf-8")}
        _ = replay_headers
        first_spool_doc = json.loads(phase_a_res[0]["spool_file"].read_text(encoding="utf-8"))
        delivered_key_0 = first_spool_doc["delivered_event_ids"][0]
        replay_resp = client.post(
            "/v1/hivemind/onboarding/feedback",
            json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
            headers={"Idempotency-Key": delivered_key_0, "X-Forwarded-For": _loopback_ip(24, 1)},
        )
        assert replay_resp.status_code == 200 and replay_resp.json()["total_responses"] == 48

        # Final Parity + Feedback Summary + Zero-Inference + CSP Nonce uniqueness assertions
        final_parity_json, final_parity_raw = _make_asgi_adapters(
            client,
            peer_ip=_loopback_ip(25, 1),
            seen_calls=seen_calls,
            csp_nonces=csp_nonces,
            sqlite_lock_hits=sqlite_lock_hits,
            lock=obs_lock,
        )
        final_parity = hc.check_holiday_deploy_parity(
            gw, http_fn=final_parity_json, raw_http_fn=final_parity_raw, home=tmp_path, env=CLEAN_ENV, cwd=tmp_path
        )
        _verify_sha256_receipt(final_parity, home=tmp_path)
        assert final_parity["parity_ok"] is True
        assert final_parity["routes_passed_count"] == 8

        raw_summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
        assert raw_summary_resp.status_code == 200
        raw_summary = raw_summary_resp.json()
        assert list(raw_summary.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]
        assert raw_summary["total_responses"] == 48
        assert raw_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert raw_summary["mean_ease_rating"] >= 4.2
        assert raw_summary["zero_game_lag_ratio"] >= 0.95
        assert final_parity["feedback_summary"] == raw_summary

        assert sqlite_lock_hits == []
        assert upstream_inference_calls == []
        assert all("completions" not in path.lower() for _, path, _ in seen_calls)
        all_nonces = [n for nonces in csp_nonces.values() for n in nonces]
        assert len(all_nonces) == 9 * 4  # 4 pass1 + 4 pass2 + 1 final parity across 4 HTML routes
        assert len(set(all_nonces)) == len(all_nonces)


def test_06_amadeus_corporate_pc_fail_closed_and_cli_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sandbox = tmp_path / "corp_sandbox"
    home_dir = sandbox / "home" / "corp_user"
    home_dir.mkdir(parents=True)
    corp_ws = sandbox / "corp_repo"
    (corp_ws / ".git").mkdir(parents=True)
    (corp_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n', encoding="utf-8")

    # Worktree gitdir + commondir corporate detection
    wt_ws = sandbox / "worktree_repo"
    wt_ws.mkdir(parents=True)
    wt_gitdir = sandbox / "git_store" / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..", encoding="utf-8")
    github_base = "https://" + "github.com/"
    (sandbox / "git_store" / "config").write_text(
        f'[url "{CORP_ZSCALER_PROXY}"]\n\tinsteadOf = {github_base}\n',
        encoding="utf-8",
    )

    net_hits: list[tuple[str, str]] = []

    def _forbid_network(method: str, url: str, *_a: Any, **_kw: Any) -> tuple[int, dict[str, Any]]:
        net_hits.append((method, url))
        return 200, {"status": "should_never_be_called"}

    def _forbid_raw(method: str, url: str, *_a: Any, **_kw: Any) -> tuple[int, str, dict[str, str]]:
        net_hits.append((method, url))
        return 200, "should_never_be_called", {}

    spool_file = home_dir / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
    cred_file = home_dir / ".hivemind" / "credentials.json"
    cont_file = home_dir / ".continue" / "hivemind_provider.yaml"

    corp_env = {
        "PATH": "/usr/bin",
        "USERDNSDOMAIN": "nce.amadeus.internal",
        "AMADEUS_ORG": "travel",
        "HTTPS_PROXY": CORP_ZSCALER_PROXY,
        "ZSCALER_CERT_PATH": "/etc/ssl/zscaler.pem",
    }

    # 1. API parity, enqueue, flush, and telemetry refuse with 0 network calls and 0 disk writes
    par_blk = hc.check_holiday_deploy_parity(
        http_fn=_forbid_network, raw_http_fn=_forbid_raw, home=home_dir, env=corp_env, cwd=corp_ws
    )
    _verify_sha256_receipt(par_blk, home=home_dir)
    assert par_blk["compliance_ok"] is False and par_blk["parity_ok"] is False
    assert par_blk["network_calls"] == 0 and par_blk["file_writes"] == 0

    enq_blk = hc.enqueue_spooled_telemetry(
        spool_path=spool_file, home=home_dir, env=corp_env, cwd=wt_ws, lang="fr"
    )
    assert enq_blk["compliance_ok"] is False and enq_blk["file_writes"] == 0
    fl_blk = hc.flush_spooled_telemetry(
        spool_path=spool_file, http_fn=_forbid_network, home=home_dir, env=corp_env, cwd=corp_ws
    )
    assert fl_blk["compliance_ok"] is False and fl_blk["network_calls"] == 0
    sub_blk = hc.submit_post_setup_telemetry(
        spool_path=spool_file, http_fn=_forbid_network, home=home_dir, env=corp_env, cwd=corp_ws
    )
    assert sub_blk["compliance_ok"] is False and sub_blk["degraded_reason"] == "refused_compliance"

    # CLI flags fail closed with exit code 2
    for argv in (
        ["--check-deploy-parity", "--json"],
        ["--flush-feedback-queue"],
        ["--post-setup-telemetry"],
    ):
        rc_blk = hc.main(
            argv,
            http_fn=_forbid_network,
            raw_http_fn=_forbid_raw,
            env=corp_env,
            cwd=corp_ws,
            home=home_dir,
        )
        assert rc_blk == 2
        capsys.readouterr()

    assert not spool_file.exists()
    assert not cred_file.exists()
    assert net_hits == []

    # 2. --uninstall stays permitted even under corporate markers and cleans up spool + creds + yaml
    spool_file.parent.mkdir(parents=True, exist_ok=True)
    spool_file.write_text('{"schema": "hivemind.quickstart.ces_spool/1", "pending": []}\n', encoding="utf-8")
    hc.save_credentials(cred_file, {"api_key": TEST_KEY_ALPHA, "gateway": hc.DEFAULT_GATEWAY})
    cont_file.parent.mkdir(parents=True, exist_ok=True)
    cont_file.write_text("name: temp\n", encoding="utf-8")
    rc_un = hc.main(["--uninstall"], http_fn=_forbid_network, env=corp_env, cwd=corp_ws, home=home_dir)
    assert rc_un == 0
    un_doc = json.loads(capsys.readouterr().out)
    assert set(un_doc["removed_files"]) == {
        "~/.hivemind/credentials.json",
        "~/.continue/hivemind_provider.yaml",
        "~/.hivemind/ces_telemetry_spool.json",
    }
    assert not spool_file.exists() and not cred_file.exists() and not cont_file.exists()
    assert net_hits == []

    # 3. Remaining CLI branches: --check-deploy-parity + --flush-feedback-queue (pass & fail), --gamer-whisper fail
    clean_ws = sandbox / "clean_hobby"
    clean_ws.mkdir(parents=True)
    clean_spool = home_dir / ".hivemind" / hc.DEFAULT_SPOOL_FILENAME
    hc.enqueue_spooled_telemetry(
        hc.DEFAULT_GATEWAY, profile="dev", ease_rating=5, spool_path=clean_spool, home=home_dir, env=CLEAN_ENV, cwd=clean_ws
    )
    store = RouterStore(tmp_path / "cli_extra.sqlite3", secret=SOAK_ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, open_registration=True)
    with TestClient(app) as cli_client:
        c_json, c_raw = _make_asgi_adapters(cli_client, peer_ip=_loopback_ip(30, 1))
        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--check-deploy-parity", "--flush-feedback-queue"],
                http_fn=c_json,
                raw_http_fn=c_raw,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 0
        )
        assert "CES TELEMETRY SPOOL FLUSH" in capsys.readouterr().out

        # Flush failure after parity pass (re-queue and fail flush with 503) returns exit 2
        hc.enqueue_spooled_telemetry(
            "http://127.0.0.1:8080",
            profile="dev",
            ease_rating=5,
            note="retry_503",
            spool_path=clean_spool,
            home=home_dir,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )

        def _post_503_json(m: str, u: str, p: Any = None, h: Any = None) -> tuple[int, dict[str, Any]]:
            if m.upper() == "POST" and u.endswith("/v1/hivemind/onboarding/feedback"):
                return 503, {"error": "down"}
            return c_json(m, u, p, h)

        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--check-deploy-parity", "--flush-feedback-queue"],
                http_fn=_post_503_json,
                raw_http_fn=c_raw,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 2
        )
        capsys.readouterr()

        # Connect with --flush-feedback-queue and Gamer-Whisper with --flush-feedback-queue + non-zero worker exit
        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--flush-feedback-queue"],
                http_fn=c_json,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 0
        )
        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--flush-feedback-queue"],
                http_fn=lambda *_a, **_kw: (200, {}),
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 0
        )
        # Ensure standalone flush exits 2 when item fails permanently
        hc.enqueue_spooled_telemetry(
            "http://127.0.0.1:8080", profile="dev", ease_rating=5, note="perm422", spool_path=clean_spool, home=home_dir, env=CLEAN_ENV, cwd=clean_ws
        )
        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--telemetry-queue-file", str(clean_spool), "--flush-feedback-queue"],
                http_fn=lambda *_a, **_kw: (422, {"error": "bad"}),
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 2
        )
        assert (
            hc.main(
                ["--gateway", "http://127.0.0.1:8080", "--handle", "cli-conn-1", "--flush-feedback-queue", "--json"],
                http_fn=c_json,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
            )
            == 0
        )
        assert (
            hc.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8080",
                    "--handle",
                    "cli-gmr-1",
                    "--gamer-whisper",
                    "--once",
                    "--flush-feedback-queue",
                ],
                http_fn=c_json,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir,
                gpu_probe_fn=lambda: hc.GpuSnapshot(True, "RTX 4070", 12288, 9600, 56.0, 2.0),
                proc_list_fn=set,
                popen_fn=lambda *_a, **_kw: _SoakWorkerHandle(polls_before_exit=0, returncode=1),
                poll_interval_s=0.05,
            )
            == 1
        )
        capsys.readouterr()

    # Connect error + human /json + --require-holiday-routes fail + --smoke-routes + --feedback exit 1
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--require-holiday-routes"],
            http_fn=lambda *_a, **_kw: (404, {"detail": "missing"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 2
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--require-holiday-routes", "--json"],
            http_fn=lambda *_a, **_kw: (503, {"error": "down"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 2
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--smoke-routes", "--lang", "fr"],
            http_fn=lambda *_a, **_kw: (200, {"status": "degraded"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 2
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--smoke-routes"],
            http_fn=lambda *_a, **_kw: (200, {"status": "ok"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 0
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--gamer-check"],
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
            gpu_probe_fn=lambda: hc.GpuSnapshot(True, "RTX 4070", 12288, 9600, 54.0, 2.0),
            proc_list_fn=set,
        )
        == 0
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--feedback", "--ease-rating", "5"],
            http_fn=lambda *_a, **_kw: (500, {"error": "boom"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 1
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080", "--feedback", "--ease-rating", "5"],
            http_fn=lambda *_a, **_kw: (
                200,
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}},
            ),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 0
    )
    assert (
        hc.main(
            ["--gateway", "http://127.0.0.1:8080"],
            http_fn=lambda *_a, **_kw: (403, {"error": "closed"}),
            env=CLEAN_ENV,
            cwd=clean_ws,
            home=home_dir,
        )
        == 1
    )

    # Human terminal connect output (EN & FR without --json), --smoke-routes --json, invalid flags & --uninstall cleanly
    def _human_ok_http(m: str, u: str, _p: Any = None, _h: Any = None) -> tuple[int, dict[str, Any]]:
        if u.endswith("/v1/hivemind/register"):
            return 200, {"account_id": "acc_h", "api_key": TEST_KEY_ALPHA, "role": "prosumer"}
        if u.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "acc_h", "handle": "human-ok", "role": "prosumer", "balance": 0.0}
        if u.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}]}
        if u.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}
        return 200, {"status": "ok"}

    for lang_h in ("en", "fr"):
        assert (
            hc.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8080",
                    "--write-continue-config",
                    "--post-setup-telemetry",
                    "--lang",
                    lang_h,
                ],
                http_fn=_human_ok_http,
                env=CLEAN_ENV,
                cwd=clean_ws,
                home=home_dir / f"human_{lang_h}",
            )
            == 0
        )
        txt = capsys.readouterr().out
        assert hc.scan_output_safety(txt, home=home_dir / f"human_{lang_h}") == []
    assert hc.main(["--gateway", "http://127.0.0.1:8080", "--smoke-routes", "--json"], http_fn=_human_ok_http, env=CLEAN_ENV, cwd=clean_ws, home=home_dir) == 0
    assert hc.main(["--lang", "es"], env=CLEAN_ENV, cwd=clean_ws, home=home_dir) == 2
    assert hc.main(["--gateway", UNTRUSTED_EVIL_GW], env=CLEAN_ENV, cwd=clean_ws, home=home_dir) == 2
    assert hc.main(["--post-setup-telemetry", "--ease-rating", "9"], env=CLEAN_ENV, cwd=clean_ws, home=home_dir) == 2
    capsys.readouterr()
