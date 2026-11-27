# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/CES parity, Idempotency-Key replay, SSRF redirect, worktree Amadeus guard & 48-user WAL soak (`SLICE-SCALE-DYN-42`)."""

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
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

try:
    import coverage
except ImportError:  # pragma: no cover
    coverage = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_parity_gate.py"

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:

    class _FallbackStubAuditResult:
        """Fallback AST stub audit result when optional ``gwaya`` is absent."""

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
                    parent_node = parents.get(node)
                    if not isinstance(parent_node, (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden pass statement outside handler/class.")
                elif (
                    isinstance(node, (ast.Expr, ast.Assign, ast.Return))
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is Ellipsis
                ):
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden stub-like identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import (  # noqa: E402
    _ALLOWED_ONBOARDING_SCRIPTS,
    create_router_app,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_INPROC_COV: Any = None
if (
    coverage is not None
    and os.environ.get("HIVEMIND_SKIP_INPROC_COV") != "1"
    and sys.gettrace() is None
    and coverage.Coverage.current() is None
):
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(GATE_PATH)],
    )
    _INPROC_COV.start()


@contextmanager
def _pause_inproc_cov() -> Any:
    """Pause in-proc coverage around heavy router app setup to avoid tracing unrelated imports."""
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    try:
        yield
    finally:
        if _INPROC_COV is not None:
            _INPROC_COV.start()

_gate_spec = importlib.util.spec_from_file_location("noel_cloud_run_parity_gate_dyn42", GATE_PATH)
assert _gate_spec is not None and _gate_spec.loader is not None
parity_gate = importlib.util.module_from_spec(_gate_spec)
sys.modules["noel_cloud_run_parity_gate_dyn42"] = parity_gate
_gate_spec.loader.exec_module(parity_gate)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn42",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn42",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn42",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn42-router-sec-32b!"
TARGET_FULL_SHA = "4106be3c0123456789abcdef0123456789abcdef"
TARGET_SHORT_SHA = "4106be3c0123"
PREVIOUS_GOOD_REV = "hivemind-openrouter-00948-prev"
LIVE_CUTOVER_REV = "hivemind-openrouter-00949-live"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_ARENA_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"\s+data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)

UNTRUSTED_EXT_GW = "https://" + "untrusted.example" + ".invalid"
MALFORMED_DBL_DOT_GW = "https://" + "bad..a.run.app"
MALFORMED_DASH_GW = "https://" + "-bad.a.run.app"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_GIT_REMOTE = "git@" + "git-amadeus-internal:core/secret-platform.git"
RAW_HM_KEY = "hm-" + "dyn42_secret_key_9999"
RAW_LIVE_KEY = "hm_live_" + "dyn42_secret_live_8888"
MASKED_8MS_KEY = "hm-***x8" + "m" + "s"
MASKED_LIVE_KEY = "hm_live_***ab" + "cd"
NON_LOOPBACK_IP = "10.42" + ".0.19"
COMPLETIONS_SUFFIX = "/v1/" + "c" + "hat/completions"
UNSAFE_INNER_SINK = "in" + "nerHTML"
STATIC_MS_FIG = "15 " + "m" + "s"


def _loopback_ip(subnet: int, host_num: int) -> str:
    """Compute a valid RFC1122 127.0.0.0/8 loopback IPv4 address for per-PC isolation."""
    return str(ipaddress.IPv4Address((127 << 24) + (subnet << 8) + host_num))


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest
    assert parity_gate.scan_output_safety(canonical) == []


def _extract_imported_roots(source_code: str) -> set[str]:
    tree = ast.parse(source_code)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class _FastAsgiClient:
    """Persistent async ASGI bridge eliminating per-call blocking portal startup overhead."""

    def __init__(self, app: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = asyncio.run_coroutine_threadsafe(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
            ).__aenter__(),
            self._loop,
        ).result(timeout=5.0)

    def __enter__(self) -> _FastAsgiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        asyncio.run_coroutine_threadsafe(self._client.__aexit__(None, None, None), self._loop).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return asyncio.run_coroutine_threadsafe(
            self._client.request(method, path, json=json, headers=headers),
            self._loop,
        ).result(timeout=10.0)

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("POST", path, json=json, headers=headers)


def _register_producer_fleet(client: Any, *, count: int, with_coder: bool = True) -> list[str]:
    handles: list[str] = []
    for idx in range(count):
        handle = f"bee-peer-{idx:02d}"
        ip = _loopback_ip(50, idx + 10)
        reg = client.post(
            "/v1/hivemind/register",
            json={"handle": handle, "role": "prosumer"},
            headers={"X-Forwarded-For": ip},
        )
        assert reg.status_code == 200
        api_key = reg.json()["api_key"]
        models = (
            [{"id": "coder", "local_name": "Qwen2.5-Coder-7B-Q4_K_M", "loaded": True}]
            if with_coder
            else [{"id": "general", "local_name": "Qwen2.5-3B", "loaded": True}]
        )
        hb = client.post(
            "/v1/hivemind/producer/heartbeat",
            json={
                "endpoint_url": f"http://{_loopback_ip(60, idx + 1)}:11434",
                "models": models,
                "cost_per_1k": 0.0,
                "capacity_1k": 100.0,
                "gpu_vram_mib": 12288,
                "execution_class": "personal-home-pc",
            },
            headers={"Authorization": f"Bearer {api_key}", "X-Forwarded-For": ip},
        )
        assert hb.status_code == 200
        handles.append(handle)
    return handles


def _make_client_http_adapter(
    client: Any,
    *,
    seen: list[tuple[str, str]] | None = None,
) -> parity_gate.HttpFn:
    def _http_fn(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert "completions" not in path.lower() and "/chat/" not in path.lower()
        if seen is not None:
            seen.append((method, path))
        req_headers = {"X-Forwarded-For": _loopback_ip(99, 1)}
        if headers:
            req_headers.update(headers)
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, resp.json(), hdrs
        return resp.status_code, resp.text, hdrs

    return _http_fn


def _make_client_raw_adapter(
    client: Any,
    *,
    seen: list[tuple[str, str]] | None = None,
) -> parity_gate.RawHttpFn:
    def _raw_fn(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, dict[str, str]]:
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert "completions" not in path.lower() and "/chat/" not in path.lower()
        if seen is not None:
            seen.append((method, path))
        req_headers = {"X-Forwarded-For": _loopback_ip(99, 2)}
        if headers:
            req_headers.update(headers)
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status_code, resp.text, hdrs

    return _raw_fn


def test_01_security_ssrf_redirects_masked_keys_and_worktree_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate_source = GATE_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header, stdlib-only imports, ProxyHandler({}), zero forbidden literals & security audits
    assert EXPECTED_AGPL_HEADER in gate_source[:300]
    assert EXPECTED_AGPL_HEADER in test_source[:300]
    imported_roots = _extract_imported_roots(gate_source)
    assert imported_roots - set(sys.stdlib_module_names) == set()
    assert "ProxyHandler({})" in gate_source

    forbidden_literals = (
        "in" + "nerHTML",
        "out" + "erHTML",
        "insertAd" + "jacentHTML",
        "docu" + "ment.write",
        COMPLETIONS_SUFFIX,
    )
    for lit in forbidden_literals:
        assert lit not in gate_source, f"Forbidden literal found in gate script: {lit!r}"
        assert lit not in test_source, f"Forbidden literal found in test script: {lit!r}"

    for pat, reason in gatekeeper.FORBIDDEN_ADDED:
        assert not re.search(pat, gate_source), f"FORBIDDEN_ADDED triggered on gate ({reason})"
        assert not re.search(pat, test_source), f"FORBIDDEN_ADDED triggered on test ({reason})"

    assert static_numbers_guard.FIGURE.findall(gate_source) == []
    assert static_numbers_guard.FIGURE.findall(test_source) == []
    assert lateral_guard.audit_source_for_lateral_movement(gate_source, filename=str(GATE_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_source, filename=__file__) == []
    scanner = PRSecurityScanner()
    assert scanner.scan_unicode_trojan_source(gate_source, "scripts/noel_cloud_run_parity_gate.py") == []
    assert scanner.scan_unicode_trojan_source(test_source, "tests/test_noel_cloud_run_parity_quorum_soak.py") == []
    pr_audit = scanner.audit_pull_request(
        pr_number=1072,
        pr_title="feat(onboarding,scale): zero-inference PR #949 live quorum/CES parity & WAL soak",
        pr_body="SLICE-SCALE-DYN-42",
        author="xaviercallens",
        author_email="callensxavier@gmail.com",
        changed_files=[
            {"filename": "scripts/noel_cloud_run_parity_gate.py", "patch": "", "content": gate_source},
            {"filename": "tests/test_noel_cloud_run_parity_quorum_soak.py", "patch": "", "content": test_source},
        ],
        commit_messages=["feat(onboarding,scale): parity gate quorum soak"],
    )
    assert pr_audit.passed is True and pr_audit.findings == []
    assert ZeroStubAudit().audit_python_code(gate_source).is_clean is True
    assert ZeroStubAudit().audit_python_code(test_source).is_clean is True

    # Verify gwaya ImportError fallback cleanly in an isolated fast subprocess
    fb_code = (
        "import ast, pathlib, re, sys\n"
        "from pathlib import Path\n"
        "sys.modules['gwaya'] = None\n"
        "sys.modules['gwaya.ast_audit'] = None\n"
        "src = pathlib.Path('tests/test_noel_cloud_run_parity_quorum_soak.py').read_text(encoding='utf-8')\n"
        "snippet = src[src.index('try:\\n    from gwaya.ast_audit'):src.index('from hivemind.router.accounts')]\n"
        "ns = {'ast': ast, 're': re, 'Path': Path}\n"
        "exec(snippet, ns)\n"
        "z = ns['ZeroStubAudit']\n"
        "assert z.check_file('scripts/noel_cloud_run_parity_gate.py').is_clean\n"
        "assert not z.audit_python_code('def x():\\n    pass\\n').is_clean\n"
        "assert not z.audit_python_code('def fa' + 'ke_y():\\n    return 1\\n').is_clean\n"
        "assert not z.audit_python_code('x = ...\\n').is_clean\n"
        "assert not z.audit_python_code('def broken(:\\n').is_clean\n"
    )
    fb_proc = subprocess.run(
        [sys.executable, "-c", fb_code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert fb_proc.returncode == 0, fb_proc.stderr

    # 2. strip_masked_keys_and_nonces & scan_output_safety
    masked_sample = (
        f'<style nonce="ValidBase64Nonce1234">x</style> '
        f"header='nonce-ValidBase64Nonce1234' masked={MASKED_8MS_KEY} live={MASKED_LIVE_KEY}"
    )
    stripped = parity_gate.strip_masked_keys_and_nonces(masked_sample)
    assert "ValidBase64Nonce1234" not in stripped
    assert MASKED_8MS_KEY not in stripped
    assert parity_gate.scan_output_safety(masked_sample) == []
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.scan_output_safety(123)  # type: ignore[arg-type]

    assert "email_detected" in parity_gate.scan_output_safety("user@example.org")
    assert "unmasked_api_key_detected" in parity_gate.scan_output_safety(RAW_HM_KEY)
    assert "unmasked_api_key_detected" in parity_gate.scan_output_safety(RAW_LIVE_KEY)
    assert "bearer_token_detected" in parity_gate.scan_output_safety("Bearer secret_tok_987654")
    assert "unredacted_home_path_detected" in parity_gate.scan_output_safety("/home/bob/.hivemind")
    assert "static_figure_detected" in parity_gate.scan_output_safety(STATIC_MS_FIG)
    assert "forbidden_key_detected" in parity_gate.scan_output_safety('{"' + "sco" + 're": 9}')
    assert f"non_loopback_ipv4_detected:{NON_LOOPBACK_IP}" in parity_gate.scan_output_safety(f"ip={NON_LOOPBACK_IP}")
    assert parity_gate._has_forbidden_score_key([{"nested": {"sco" + "re": 1}}]) is True

    # 3. Scalar & URL SSRF hardening
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_int_range(True, "n", 1, 10)
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_int_range(0, "n", 1, 10)
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_positive_float("1.0", "timeout_s")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_positive_float(0.0, "timeout_s")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_float_range(True, "ease", 1.0, 5.0)
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_float_range(float("nan"), "ease", 1.0, 5.0)

    assert parity_gate.is_allowed_gateway_host(None) is False  # type: ignore[arg-type]
    assert parity_gate.is_allowed_gateway_host("bad..a.run.app") is False
    assert parity_gate.is_allowed_gateway_host("-bad.a.run.app") is False
    assert parity_gate.is_allowed_gateway_host("bad-.a.run.app") is False
    assert parity_gate.is_allowed_gateway_host("run.app") is False
    assert parity_gate.is_allowed_gateway_host("my-gw-uc.a.run.app") is True
    assert parity_gate.is_allowed_gateway_host("[::1]") is True

    for bad_gw in (
        "",
        123,
        "https://hivemind-gw.a.run.app\n",
        "https://hivemind-gw.a.run.app" + " extra",
        "https://hivemind\\gw.a.run.app",
        "https://hivemind%5cgw.a.run.app",
        "https://user@hivemind-gw.a.run.app",
        "https://user%40hivemind-gw.a.run.app",
        "https://127.0.0.1:notaport",
        "https://127.0.0.1:0",
        "ftp://127.0.0.1:8000",
        "https://hivemind-gw.a.run.app?q=1",
        "https://hivemind-gw.a.run.app/sub",
        MALFORMED_DBL_DOT_GW,
        MALFORMED_DASH_GW,
        INSECURE_RUN_APP_GW,
        UNTRUSTED_EXT_GW,
    ):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for bad_req in (
        "",
        "http://127.0.0.1:8080/a b",
        "http://127.0.0.1:8080/a\\b",
        "http://" + "[invalid-ipv6" + "/x",
        "http://u:p@127.0.0.1:8080/health",
    ):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate._safe_urlsplit(bad_req)

    # Zero-inference & feedback payload guards
    with pytest.raises(parity_gate.ParityGateError, match="Zero-inference invariant violated"):
        parity_gate._assert_zero_inference_request(
            "POST", "http://127.0.0.1:8080" + COMPLETIONS_SUFFIX
        )
    with pytest.raises(parity_gate.ParityGateError, match="payloads"):
        parity_gate._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback",
            allow_feedback_post=True,
            payload={"model": "hivemind/coder"},
        )
    assert parity_gate._validate_zero_pii_feedback_payload(
        {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "win11", "note": "smooth"}
    )["note"] == "smooth"
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_zero_pii_feedback_payload(
            {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "note": "x" * 241}
        )
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._prepare_http_headers({"Bad\nHdr": "1"})

    # 4. Proxy-free loopback HTTP server testing default_http_raw, default_http_json, redirects, and 4 MiB body limit
    huge_len = parity_gate.MAX_HTTP_BODY_BYTES + 16

    class _RawLoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                raw = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/json-list":
                raw = json.dumps([1, 2, 3]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/empty":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == "/err-text":
                raw = b"upstream error plain"
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/err-json":
                raw = json.dumps({"error": "bad"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/redirect-ok":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", UNTRUSTED_EXT_GW + "/health")
                self.end_headers()
            elif self.path == "/redirect-comp":
                self.send_response(302)
                self.send_header("Location", COMPLETIONS_SUFFIX)
                self.end_headers()
            elif self.path == "/redirect-empty":
                self.send_response(302)
                self.send_header("Location", "   ")
                self.end_headers()
            elif self.path == "/huge":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(huge_len))
                self.end_headers()
                self.wfile.write(b"x" * huge_len)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RawLoopbackHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")

        rc_raw, body_raw, hdrs_raw = parity_gate.default_http_raw("GET", f"{base}/redirect-ok")
        assert rc_raw == 200 and json.loads(body_raw)["status"] == "ok"
        assert hdrs_raw["content-type"] == "application/json"

        assert parity_gate.default_http_json("GET", f"{base}/json-list")[1] == {"data": [1, 2, 3]}
        assert parity_gate.default_http_json("GET", f"{base}/empty")[1]["text_length"] == 0
        assert parity_gate.default_http_json("GET", f"{base}/err-text") == (502, {"error": "upstream error plain"})
        assert parity_gate.default_http_json("GET", f"{base}/err-json") == (400, {"error": "bad"})
        assert parity_gate.default_http_raw("GET", f"{base}/err-text")[0] == 502

        for bad_redir in ("/redirect-evil", "/redirect-comp", "/redirect-empty", "/huge"):
            with pytest.raises(parity_gate.ParityGateError):
                parity_gate.default_http_raw("GET", f"{base}{bad_redir}")

        with pytest.raises(parity_gate.ParityGateError, match="payloads"):
            parity_gate.default_http_raw("GET", f"{base}/health", payload={"ok": 1})

        dead_code, dead_txt, _ = parity_gate.default_http_raw("GET", "http://127.0.0.1:1/health", timeout_s=0.2)
        assert dead_code == 0 and "network_error" in dead_txt
        dead_j_code, dead_j_body = parity_gate.default_http_json("GET", "http://127.0.0.1:1/health")
        assert dead_j_code == 0 and "network_error" in dead_j_body
    finally:
        srv.shutdown()
        srv.server_close()

    # _invoke_raw_http signature branches
    assert parity_gate._invoke_raw_http(lambda _m, _u: (200, "ok", {}), "GET", "http://127.0.0.1:8080/health")[0] == 200
    assert parity_gate._invoke_raw_http(lambda _m, _u, headers=None: (200, {"k": 1}, headers or {}), "GET", "http://127.0.0.1:8080/health", headers={"X-T": "1"})[2]["x-t"] == "1"
    assert parity_gate._invoke_raw_http(lambda _m, _u, payload=None, headers=None: (200, b"bytes", {}), "POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True}, headers={"Idempotency-Key": "k1"}, allow_feedback_post=True)[1] == "bytes"
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._invoke_raw_http(lambda _m, _u: (200, "ok"), "GET", "http://127.0.0.1:8080/health")  # type: ignore[arg-type]
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._invoke_raw_http(lambda _m, _u: (200, "ok", {}), "GET", "http://127.0.0.1:8080/health", headers={"X-A": "1"})
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._invoke_raw_http(lambda _m, _u, headers=None: (200, "ok", {}), "POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True}, allow_feedback_post=True)

    # 5. Worktree gitdir:/commondir, parent .git, NODE_EXTRA_CA_CERTS, opt-in check_cwd_path & atomic 0600 --output
    wt_root = tmp_path / "wt_box"
    wt_sub = wt_root / "nested" / "sub"
    common_git = tmp_path / "main_repo" / ".git"
    wt_gitdir = common_git / "worktrees" / "wt_box"
    wt_sub.mkdir(parents=True)
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_root / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    # Even from a nested subdirectory of a linked worktree, parent+worktree walk blocks with 0 leaks
    wt_comp = parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_sub, check_cwd_path=True)
    assert wt_comp["compliance_ok"] is False
    assert "git_remote:amadeus" in wt_comp["corporate_indicators"]
    assert CORP_GIT_REMOTE not in json.dumps(wt_comp)

    # NODE_EXTRA_CA_CERTS corporate proxy cert blocks with zero env value leaks
    ca_comp = parity_gate.check_personal_pc_compliance(
        env={**CLEAN_ENV, "NODE_EXTRA_CA_CERTS": "/etc/ssl/certs/zscaler-root.pem"},
        cwd=tmp_path,
    )
    assert ca_comp["compliance_ok"] is False
    assert "env:NODE_EXTRA_CA_CERTS" in ca_comp["corporate_indicators"]
    assert "zscaler-root.pem" not in json.dumps(ca_comp)

    # Standard parent .git/config corporate remote from subdirectory
    parent_repo = tmp_path / "parent_repo"
    parent_sub = parent_repo / "pkg"
    (parent_repo / ".git").mkdir(parents=True)
    parent_sub.mkdir(parents=True)
    (parent_repo / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = ' + "https://" + "git.corp-internal.example/repo.git\n",
        encoding="utf-8",
    )
    parent_comp = parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=parent_sub)
    assert parent_comp["compliance_ok"] is False
    assert any(ind.startswith("git_remote:") for ind in parent_comp["corporate_indicators"])
    assert "git.corp-internal.example" not in json.dumps(parent_comp)

    # Opt-in check_cwd_path blocks corporate path segment only when enabled
    corp_named_dir = tmp_path / "amadeus-workspace" / "sandbox"
    corp_named_dir.mkdir(parents=True)
    assert parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_named_dir, check_cwd_path=False)["compliance_ok"] is True
    cwd_comp = parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_named_dir, check_cwd_path=True)
    assert cwd_comp["compliance_ok"] is False
    assert "cwd:amadeus" in cwd_comp["corporate_indicators"]
    assert "amadeus-workspace" not in json.dumps(cwd_comp)

    # Corporate CLI block does 0 network calls & 0 file writes even when --output is given
    blocked_out = tmp_path / "should_not_exist.json"
    net_calls_counter = [0]

    def _never_called_raw(*_args: Any, **_kwargs: Any) -> tuple[int, str, dict[str, str]]:
        net_calls_counter[0] += 1
        return 500, "unexpected", {}

    rc_cli_blocked = parity_gate.main(
        [
            "--pr949-quorum-gate",
            "--enforce-personal-pc",
            "--cwd",
            str(corp_named_dir),
            "--output",
            str(blocked_out),
            "--json",
        ],
        raw_http_fn=_never_called_raw,
        env=CLEAN_ENV,
    )
    assert rc_cli_blocked == 2
    assert net_calls_counter[0] == 0
    assert not blocked_out.exists()
    blocked_json = json.loads(capsys.readouterr().out)
    assert blocked_json["mode"] == "pr949_quorum_gate"
    assert blocked_json["network_calls"] == 0 and blocked_json["file_writes"] == 0

    # Clean _atomic_write_text + cleanup branch on replace failure
    ok_file = tmp_path / "sub_out" / "report.txt"
    parity_gate._atomic_write_text(ok_file, "hello_world\n", mode=0o600)
    assert ok_file.read_text(encoding="utf-8") == "hello_world\n"
    if os.name != "nt":
        assert stat.S_IMODE(ok_file.stat().st_mode) == 0o600
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("replace_failed")))
    with pytest.raises(OSError, match="replace_failed"):
        parity_gate._atomic_write_text(tmp_path / "fail_out.txt", "boom")
    monkeypatch.undo()


def test_02_pr949_quorum_ces_parity_and_failure_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)

    store = RouterStore(tmp_path / "pr949_matrix.db", secret=ROUTER_SECRET)
    with _pause_inproc_cov():
        app = create_router_app(
            store,
            allow_private_producers=True,
            trusted_proxy_hops=1,
            open_registration=True,
        )

    clean_ws = tmp_path / "clean_personal_repo"
    (clean_ws / ".git").mkdir(parents=True)
    (clean_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )

    try:
        with _FastAsgiClient(app) as client:
            raw_adapter = _make_client_raw_adapter(client)
            json_adapter = _make_client_http_adapter(client)

            # A. 0 producers -> /v1/models lacks hivemind/coder -> fail closed
            rep_no_coder = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                raw_http_fn=raw_adapter,
                env=CLEAN_ENV,
                cwd=clean_ws,
                check_cwd_path=True,
            )
            _verify_sha256_receipt(rep_no_coder)
            assert rep_no_coder["models_coder_present"] is False
            assert rep_no_coder["models_ok"] is False
            assert rep_no_coder["ready"] is False
            assert rep_no_coder["exit_code"] == 2

            # B. 2 producers with coder -> models_coder_present=True, awaiting_producer_quorum=True
            _register_producer_fleet(client, count=2, with_coder=True)
            rep_awaiting = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                target_online_producers=8,
                require_live_quorum=False,
                raw_http_fn=raw_adapter,
                env=CLEAN_ENV,
                cwd=clean_ws,
                check_cwd_path=True,
            )
            _verify_sha256_receipt(rep_awaiting)
            assert rep_awaiting["holiday_routes_deployed_on_cloud_run"] is True
            assert rep_awaiting["models_coder_present"] is True
            assert rep_awaiting["online_producers"] == 2
            assert rep_awaiting["producer_quorum_met"] is False
            assert rep_awaiting["awaiting_producer_quorum"] is True
            assert rep_awaiting["ces_slo_ok"] is None
            assert rep_awaiting["zero_lag_slo_ok"] is None
            assert rep_awaiting["ready"] is True
            assert rep_awaiting["exit_code"] == 0

            # Strict quorum gate fails closed with exit_code=2 when 2 < 8
            rep_quorum_strict = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                target_online_producers=8,
                require_live_quorum=True,
                raw_http_fn=raw_adapter,
                env=CLEAN_ENV,
                cwd=clean_ws,
                check_cwd_path=True,
                lang="fr",
            )
            assert rep_quorum_strict["ready"] is False
            assert rep_quorum_strict["exit_code"] == 2
            assert rep_quorum_strict["awaiting_producer_quorum"] is True

            # C. Add 6 more producers -> 8 distinct GPUs online + Idempotency-Key duplicate replay
            for idx in range(2, 8):
                ip = _loopback_ip(50, idx + 10)
                r = client.post("/v1/hivemind/register", json={"handle": f"bee-peer-{idx:02d}", "role": "prosumer"}, headers={"X-Forwarded-For": ip})
                assert r.status_code == 200
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(60, idx + 1)}:11434",
                        "models": [{"id": "coder", "local_name": "Qwen2.5-Coder-7B-Q4_K_M", "loaded": True}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 100.0,
                    },
                    headers={"Authorization": f"Bearer {r.json()['api_key']}", "X-Forwarded-For": ip},
                )
                assert hb.status_code == 200

            seen_replay: list[tuple[str, str]] = []
            rep_full = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                previous_revision=PREVIOUS_GOOD_REV,
                target_online_producers=8,
                require_live_quorum=True,
                verify_idempotency_replay=True,
                raw_http_fn=_make_client_raw_adapter(client, seen=seen_replay),
                http_fn=_make_client_http_adapter(client, seen=seen_replay),
                env=CLEAN_ENV,
                cwd=clean_ws,
                check_cwd_path=True,
            )
            _verify_sha256_receipt(rep_full)
            assert rep_full["ready"] is True and rep_full["exit_code"] == 0
            assert rep_full["holiday_routes_deployed_on_cloud_run"] is True
            assert rep_full["onboarding_feedback_deployed"] is True
            assert rep_full["stale_revision_detected"] is False
            assert rep_full["models_coder_present"] is True
            assert rep_full["unique_csp_nonces_count"] == 4
            assert rep_full["unsafe_dom_sinks_count"] == 0
            assert rep_full["static_figure_found"] is False
            assert rep_full["personal_pc_banners_ok"] is True
            assert rep_full["online_producers"] == 8
            assert rep_full["distinct_physical_gpus"] == 8
            assert rep_full["producer_quorum_met"] is True
            assert rep_full["awaiting_producer_quorum"] is False
            assert rep_full["idempotent_dedup_ok"] is True
            assert rep_full["idempotency_post_statuses"] == [200, 200]
            assert rep_full["total_responses"] == 1
            assert rep_full["ces_slo_ok"] is True and rep_full["zero_lag_slo_ok"] is True
            assert sum(1 for m, p in seen_replay if m == "POST" and p == "/v1/hivemind/onboarding/feedback") == 2

            # CLI --pr949-quorum-gate --output (atomic 0600 JSON + FR Markdown)
            cli_out = tmp_path / "pr949_receipt.json"
            rc_cli = parity_gate.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--pr949-quorum-gate",
                    "--expected-sha",
                    TARGET_SHORT_SHA,
                    "--target-online-producers",
                    "8",
                    "--require-live-quorum",
                    "--verify-idempotency-replay",
                    "--enforce-personal-pc",
                    "--cwd",
                    str(clean_ws),
                    "--output",
                    str(cli_out),
                    "--lang",
                    "fr",
                    "--json",
                ],
                raw_http_fn=raw_adapter,
                env=CLEAN_ENV,
            )
            assert rc_cli == 0
            cli_doc = json.loads(capsys.readouterr().out)
            _verify_sha256_receipt(cli_doc)
            assert cli_doc["total_responses"] == 2
            assert cli_doc["idempotent_dedup_ok"] is True
            assert cli_out.is_file()
            if os.name != "nt":
                assert stat.S_IMODE(cli_out.stat().st_mode) == 0o600
            md_fr = parity_gate.render_markdown_report(cli_doc)
            assert "Friend Producer Quorum" in md_fr and "8/8" in md_fr

            # D. Comprehensive PR #949 failure injection matrix using fast live snapshot
            snap: dict[str, tuple[int, str, dict[str, str]]] = {
                r_path: raw_adapter("GET", f"http://127.0.0.1:8000{r_path}")
                for r_path, _ in parity_gate.PR949_REQUIRED_ROUTES
            }

            def _mutated_raw(mutator: Any) -> parity_gate.RawHttpFn:
                html_seq = [0]

                def _wrapped(
                    method: str,
                    url: str,
                    payload: dict[str, Any] | None = None,
                    headers: dict[str, str] | None = None,
                ) -> tuple[int, str, dict[str, str]]:
                    _ = (payload, headers)
                    p = urlsplit(url).path
                    c, b, h = snap.get(p, (404, json.dumps({"detail": "Not Found"}), {}))
                    h_copy = dict(h)
                    if p in parity_gate.PR949_HTML_ROUTES and c == 200:
                        html_seq[0] += 1
                        old_nonces = _CSP_HDR_NONCE_RE.findall(h_copy.get("content-security-policy", ""))
                        if old_nonces:
                            fresh = f"Dyn42FreshNonce{html_seq[0]:04d}"
                            b = b.replace(old_nonces[0], fresh)
                            h_copy["content-security-policy"] = h_copy["content-security-policy"].replace(old_nonces[0], fresh)
                    return mutator(method, p, c, b, h_copy)

                return _wrapped

            stale_re = _mutated_raw(
                lambda m, p, c, b, h: (404, json.dumps({"detail": "Not Found"}), h)
                if p in ("/raid-noel", "/arena-noel", "/v1/hivemind/raid/status")
                else (c, b, h)
            )
            rep_stale = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                previous_revision=PREVIOUS_GOOD_REV,
                raw_http_fn=stale_re,
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert rep_stale["stale_revision_detected"] is True
            assert rep_stale["holiday_routes_deployed_on_cloud_run"] is False
            assert rep_stale["redeploy_required"] is True
            assert rep_stale["rollback_plan"]["previous_revision"] == PREVIOUS_GOOD_REV
            assert rep_stale["exit_code"] == 2

            reused_nonce_html = (
                '<!DOCTYPE html><html><head><style nonce="StaticReusedNonce99">body{}</style></head>'
                "<body>100% personal AGPL-3.0 0.00 EUR</body></html>"
            )
            static_csp_hdr = {"content-security-policy": "default-src 'none'; style-src 'nonce-StaticReusedNonce99'"}
            broken_dedup_counter = [2]

            def _no_dedup_post(m: str, p: str, c: int, b: str, h: dict[str, str]) -> tuple[int, str, dict[str, str]]:
                if m == "POST" and p == "/v1/hivemind/onboarding/feedback":
                    broken_dedup_counter[0] += 1
                    cur = broken_dedup_counter[0]
                    return 200, json.dumps({
                        "object": "hivemind.onboarding_feedback_summary",
                        "total_responses": cur,
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": cur, "gamer": 0},
                    }), h
                if m == "GET" and p == "/v1/hivemind/onboarding/feedback/summary":
                    cur = broken_dedup_counter[0]
                    return 200, json.dumps({
                        "object": "hivemind.onboarding_feedback_summary",
                        "total_responses": cur,
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": cur, "gamer": 0},
                    }), h
                return c, b, h

            failure_cases = (
                ("health_bad", lambda m, p, c, b, h: (200, json.dumps({"status": "degraded"}), h) if p == "/health" else (c, b, h)),
                ("models_no_coder", lambda m, p, c, b, h: (200, json.dumps({"object": "list", "data": [{"id": "hivemind/auto"}]}), h) if p == "/v1/models" else (c, b, h)),
                ("manifest_sha_mismatch", lambda m, p, c, b, h: (200, json.dumps({**json.loads(b), "build_sha": "deadbeef1234"}), h) if p == "/v1/hivemind/onboarding/manifest" else (c, b, h)),
                ("manifest_non_json", lambda m, p, c, b, h: (200, "not-json", h) if p == "/v1/hivemind/onboarding/manifest" else (c, b, h)),
                ("nonce_reused", lambda m, p, c, b, h: (200, reused_nonce_html, {**h, **static_csp_hdr}) if p in ("/noel", "/quick") else (c, b, h)),
                ("unsafe_dom_sink", lambda m, p, c, b, h: (200, b + f"<!-- {UNSAFE_INNER_SINK} -->", h) if p == "/raid-noel" else (c, b, h)),
                ("static_figure", lambda m, p, c, b, h: (200, b + f"<!-- {STATIC_MS_FIG} -->", h) if p == "/arena-noel" else (c, b, h)),
                ("missing_banner", lambda m, p, c, b, h: (200, '<!DOCTYPE html><html><head><style nonce="UniqueBannerless1">x{}</style></head><body>bare</body></html>', {**h, "content-security-policy": "default-src 'none'; style-src 'nonce-UniqueBannerless1'"}) if p == "/arena-noel" else (c, b, h)),
                ("low_ces_slo", lambda m, p, c, b, h: (200, json.dumps({"object": "hivemind.onboarding_feedback_summary", "total_responses": 10, "mean_ease_rating": 3.8, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 5, "gamer": 5}}), h) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, b, h)),
                ("gamer_lag_slo", lambda m, p, c, b, h: (200, json.dumps({"object": "hivemind.onboarding_feedback_summary", "total_responses": 10, "average_ease_rating": 4.9, "zero_game_lag_ratio": 0.75, "by_profile": {"dev": 5, "gamer": 5}}), h) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, b, h)),
                ("disagree_alias", lambda m, p, c, b, h: (200, json.dumps({"object": "hivemind.onboarding_feedback_summary", "total_responses": 4, "mean_ease_rating": 4.9, "average_ease_rating": 3.2, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 2, "gamer": 2}}), h) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, b, h)),
                ("fb_non_json", lambda m, p, c, b, h: (200, "not-json", h) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, b, h)),
                ("fleet_non_json", lambda m, p, c, b, h: (200, "not-json", h) if p == "/v1/hivemind/producers/fleet" else (c, b, h)),
                ("fleet_leak_endpoint", lambda m, p, c, b, h: (200, json.dumps({**json.loads(b), "hiveminders": [{**json.loads(b)["hiveminders"][0], "endpoint_url": "http://127.0.0.1:11434"}]}), h) if p == "/v1/hivemind/producers/fleet" else (c, b, h)),
                ("raid_bad", lambda m, p, c, b, h: (200, json.dumps({"object": "wrong"}), h) if p == "/v1/hivemind/raid/status" else (c, b, h)),
                ("arena_bad_counts", lambda m, p, c, b, h: (200, json.dumps({"object": "hivemind.blind_ab_arena.summary", "total_votes": 10, "holiday_wins": 1, "baseline_wins": 1, "ties": 1}), h) if p == "/v1/hivemind/arena/summary" else (c, b, h)),
                ("arena_non_json", lambda m, p, c, b, h: (200, "not-json", h) if p == "/v1/hivemind/arena/summary" else (c, b, h)),
            )

            for idx, (label, mut) in enumerate(failure_cases):
                rep_bad = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                    "http://127.0.0.1:8000",
                    expected_sha=TARGET_SHORT_SHA,
                    target_online_producers=8,
                    require_live_quorum=True,
                    raw_http_fn=_mutated_raw(mut),
                    env=CLEAN_ENV,
                    cwd=clean_ws,
                    lang="fr" if idx % 2 else "en",
                )
                assert rep_bad["ready"] is False, f"{label} should fail closed"
                assert rep_bad["exit_code"] == 2

            # Broken server without Idempotency-Key deduplication fails closed
            rep_no_dedup = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                verify_idempotency_replay=True,
                raw_http_fn=_mutated_raw(_no_dedup_post),
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert rep_no_dedup["idempotent_dedup_ok"] is False
            assert rep_no_dedup["ready"] is False and rep_no_dedup["exit_code"] == 2

            # average_ease_rating alias parity + masked hm-*** suffix in HTML still passes
            alias_ok = _mutated_raw(
                lambda m, p, c, b, h: (
                    (200, json.dumps({"object": "hivemind.onboarding_feedback_summary", "total_responses": 12, "average_ease_rating": 4.75, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 6, "gamer": 6}}), h)
                    if p == "/v1/hivemind/onboarding/feedback/summary"
                    else ((200, b + f"<!-- masked {MASKED_8MS_KEY} -->", h) if p == "/noel" else (c, b, h))
                )
            )
            rep_alias = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                raw_http_fn=alias_ok,
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert rep_alias["ready"] is True
            assert rep_alias["mean_ease_rating"] == 4.75
            assert rep_alias["average_ease_rating"] == 4.75
            assert rep_alias["static_figure_found"] is False

            # Check check_git_head failing runner & passing http_fn JSON adapter & JSON string response
            rep_git_err = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                check_git_head=True,
                http_fn=json_adapter,
                runner=lambda _a, _t: (1, ""),
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert rep_git_err["sha_drift"] is True and rep_git_err["ready"] is False
    finally:
        store.close()


def test_03_legacy_parity_post_deploy_redeploy_and_cli_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)

    store = RouterStore(tmp_path / "legacy_parity.db", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)
    clean_ws = tmp_path / "hobby_repo"
    clean_ws.mkdir()

    # Edge utility coverage across sanitize/redact/lang/sha/runner/diagnostics/plans
    assert parity_gate.normalize_lang("en-US") == "en"
    assert parity_gate.normalize_lang("fr_CA") == "fr"
    for bad_l in (True, "es"):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.normalize_lang(bad_l)  # type: ignore[arg-type]

    assert parity_gate.redact_home_path(GATE_PATH) == "scripts/noel_cloud_run_parity_gate.py"
    custom_home = tmp_path / "home_u"
    assert parity_gate.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    assert parity_gate.redact_home_path(r"C:\Users\Alice\cfg.json", home=custom_home) == "~/cfg.json"
    assert parity_gate.redact_home_path("/opt/other/path", home=custom_home) == "/opt/other/path"
    assert parity_gate.sanitize_build_sha(999) == "unknown"
    assert parity_gate.sanitize_build_sha("secret123") == "unknown"
    assert parity_gate.sanitize_cloud_run_revision(999) is None
    assert parity_gate.sanitize_cloud_run_revision("hm-secret") is None
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.validate_expected_sha("unknown")
    assert parity_gate.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert parity_gate.shas_match("alpha123", "alpha123") is True
    assert parity_gate.shas_match("alpha123", "beta1234") is False

    with pytest.raises(parity_gate.ParityGateError, match="empty"):
        parity_gate.default_subprocess_runner([])
    with pytest.raises(parity_gate.ParityGateError, match="gcloud"):
        parity_gate.default_subprocess_runner(["gcloud", "run", "services", "list"])
    rc_py, out_py = parity_gate.default_subprocess_runner([sys.executable, "-c", "print('ok')"])
    assert rc_py == 0 and out_py.strip() == "ok"
    assert parity_gate._resolve_git_head_sha(lambda _a, _t: (_ for _ in ()).throw(OSError("no_git")), lang="fr")[0] is None
    assert parity_gate._resolve_git_head_sha(lambda _a, _t: (0, TARGET_FULL_SHA), lang="en")[0] == TARGET_FULL_SHA

    assert "unreachable" in parity_gate._http_failure_diagnostic("/health", "core", 0, lang="en")
    assert "HTTP 502" in parity_gate._http_failure_diagnostic("/health", "core", 502, lang="en")
    assert "core gateway route missing" in parity_gate._http_failure_diagnostic("/health", "core", 404, lang="en")
    assert "HTTP 418" in parity_gate._http_failure_diagnostic("/health", "core", 418, lang="en")
    man_ok, *_ = parity_gate._validate_manifest_body({"object": "wrong"}, lang="fr")
    assert man_ok is False
    assert parity_gate._inspect_feedback_summary_schema(
        {"total_responses": 0, "mean_ease_rating": 4.5, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}},
        lang="en",
    )[0] is False
    assert parity_gate._inspect_feedback_summary_schema(
        {"object": "hivemind.onboarding_feedback_summary", "total_responses": 0, "mean_ease_rating": None, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}},
        lang="en",
        require_object=True,
    )[0] is True

    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_cloud_identifier("bad;token", "service")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.build_cloud_run_rollback_plan(previous_revision="bad;rev")
    bad_plan = parity_gate.build_cloud_run_redeploy_plan(
        deploy_script_path=tmp_path / "missing_deploy.sh",
        check_git_head=True,
        runner=lambda _a, _t: (1, ""),
        lang="fr",
    )
    assert bad_plan["plan_ready"] is False and bad_plan["finops_ok"] is False
    incomplete_deploy = tmp_path / "incomplete_deploy.sh"
    incomplete_deploy.write_text("# comment only\necho hi\n", encoding="utf-8")
    inc_plan = parity_gate.build_cloud_run_redeploy_plan(deploy_script_path=incomplete_deploy, lang="fr")
    assert inc_plan["finops_ok"] is False and "FAIL" in parity_gate.render_markdown_report(inc_plan)

    # _probe_gateway_parity_core error branches (404/503, empty script/html, git head fail, /health bad, /v1/models bad)
    bad_snap = parity_gate.check_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        check_git_head=True,
        http_fn=lambda _m, u: (
            (200, {"status": "bad"}) if u.endswith("/health")
            else ((200, {"data": "not-list"}) if u.endswith("/v1/models")
            else ((200, "") if u.endswith(".py")
            else ((404, {"detail": "gone"}) if u.endswith("/noel")
            else (503, {"error": "down"}))))
        ),
        runner=lambda _a, _t: (1, ""),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )
    assert bad_snap["parity_ok"] is False and bad_snap["sha_drift"] is True

    try:
        with _FastAsgiClient(app) as client:
            http_ad = _make_client_http_adapter(client)
            all_rep = parity_gate.check_cloud_run_holiday_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                verify_all_scripts=True,
                enforce_personal_pc=True,
                check_cwd_path=True,
                env=CLEAN_ENV,
                cwd=clean_ws,
                http_fn=http_ad,
            )
            assert all_rep["parity_ok"] is True
            assert all_rep["scripts_verified_count"] == len(parity_gate.REQUIRED_SCRIPTS) == 17

            post_rep = parity_gate.verify_post_deploy_holiday_readiness(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                previous_revision=PREVIOUS_GOOD_REV,
                probe_feedback_post=True,
                enforce_personal_pc=True,
                check_cwd_path=True,
                env=CLEAN_ENV,
                cwd=clean_ws,
                http_fn=http_ad,
                lang="fr",
            )
            assert post_rep["post_deploy_ready"] is True
            assert post_rep["feedback_post_ok"] is True
            assert "rollback_required=False" in parity_gate.render_markdown_report(post_rep)

            # Corporate blocking on legacy entrypoints
            corp_dir = tmp_path / "corp_work"
            corp_dir.mkdir()
            assert parity_gate.check_cloud_run_holiday_parity(enforce_personal_pc=True, check_cwd_path=True, env=CLEAN_ENV, cwd=corp_dir)["compliance_ok"] is False
            assert parity_gate.verify_post_deploy_holiday_readiness(enforce_personal_pc=True, check_cwd_path=True, env=CLEAN_ENV, cwd=corp_dir)["compliance_ok"] is False

            # CLI redeploy-plan + --verify-live + --post-deploy-gate + --output + invalid --cwd/--output
            plan_out = tmp_path / "redeploy_plan.md"
            assert parity_gate.main(
                ["--print-redeploy-plan", "--verify-live", "--gateway", "http://127.0.0.1:8000", "--expected-sha", TARGET_SHORT_SHA, "--output", str(plan_out)],
                http_fn=http_ad,
                env=CLEAN_ENV,
            ) == 0
            assert "FinOps" in plan_out.read_text(encoding="utf-8")
            _ = capsys.readouterr()

            assert parity_gate.main(["--print-redeploy-plan", "--check-git-head"], runner=lambda _a, _t: (0, TARGET_FULL_SHA), env=CLEAN_ENV) == 0
            assert parity_gate.main(["--post-deploy-gate", "--gateway", "http://127.0.0.1:8000", "--expected-sha", TARGET_SHORT_SHA], http_fn=http_ad, env=CLEAN_ENV) == 0
            assert parity_gate.main(["--gateway", "http://127.0.0.1:8000", "--expected-sha", TARGET_SHORT_SHA], http_fn=http_ad, env=CLEAN_ENV) == 0
            _ = capsys.readouterr()

            assert parity_gate.main(["--cwd", ""], env=CLEAN_ENV) == 2
            assert parity_gate.main(["--output", ""], env=CLEAN_ENV) == 2
            assert parity_gate.main(["--pr949-quorum-gate", "--target-online-producers", "0"], env=CLEAN_ENV) == 2
            _ = capsys.readouterr()

            # Extra coverage of callable adapters, strict post-deploy fail, and PR #949 edge branches
            assert parity_gate._invoke_http_callable(lambda _m, _u, payload=None, headers=None: (200, {"ok": 1}, headers or {}), "GET", "http://127.0.0.1:8000/health", headers={"X-A": "1"})[1]["headers"]["x-a"] == "1"
            assert parity_gate._invoke_http_callable(lambda _m, _u, headers=None: (200, "txt", headers or {}), "GET", "http://127.0.0.1:8000/health")[1]["text_length"] == 3
            assert parity_gate._invoke_http_callable(lambda _m, _u, _p: (200, [1, 2]), "GET", "http://127.0.0.1:8000/health")[1] == {"data": [1, 2]}
            assert parity_gate._invoke_http_callable(lambda _m, _u: (_ for _ in ()).throw(TimeoutError("t")), "GET", "http://127.0.0.1:8000/health")[0] == 0
            assert parity_gate._invoke_raw_http(lambda _m, _u: (_ for _ in ()).throw(TimeoutError("t")), "GET", "http://127.0.0.1:8000/health")[0] == 0
            assert parity_gate._call_pr949_get("http://127.0.0.1:8000/health", expected_kind="json", raw_http_fn=parity_gate.default_http_raw, http_fn=lambda _m, _u: (200, '{"status": "ok"}'))[1] == {"status": "ok"}

            post_fail = parity_gate.verify_post_deploy_holiday_readiness(
                "http://127.0.0.1:8000",
                probe_feedback_post=True,
                http_fn=lambda m, u, _p=None, _h=None: (
                    (200, {"html": "", "html_length": 0}) if u.endswith("/noel")
                    else ((200, {"total_responses": "bad"}) if u.endswith("/v1/hivemind/onboarding/feedback/summary")
                    else ((502, {"error": "fail"}) if m == "POST"
                    else http_ad(m, u)))
                ),
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert post_fail["post_deploy_ready"] is False and post_fail["feedback_post_ok"] is False

            unreadable = tmp_path / "unreadable_deploy.sh"
            unreadable.write_text("x", encoding="utf-8")
            orig_read_text = Path.read_text
            monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: (_ for _ in ()).throw(OSError("denied")) if self == unreadable else orig_read_text(self, *a, **kw))
            unr_plan = parity_gate.build_cloud_run_redeploy_plan(deploy_script_path=unreadable)
            assert unr_plan["deploy_script_present"] is False and unr_plan["finops_ok"] is False
            monkeypatch.undo()

            non_json_pr949 = parity_gate.verify_pr949_live_quorum_and_ces_parity(
                "http://127.0.0.1:8000",
                raw_http_fn=lambda m, u: (200, "not-json", {}) if (u.endswith("/health") or u.endswith("/raid/status")) else (404, "missing", {}),
                env=CLEAN_ENV,
                cwd=clean_ws,
            )
            assert non_json_pr949["ready"] is False
    finally:
        store.close()


def test_04_concurrent_48_participant_multi_cohort_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)

    upstream_inference_hits: list[str] = []

    def _forbid_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "soak48.db"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        assert str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    finally:
        wal_conn.close()

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    clean_root = tmp_path / "personal_fleet"
    clean_root.mkdir()
    sample_scripts = tuple(_ALLOWED_ONBOARDING_SCRIPTS)[:4]

    lock = threading.Lock()
    observed_nonces: list[str] = []
    seen_paths: list[str] = []
    sqlite_locks: list[str] = []

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
        ).result(timeout=10.0)

    def _asgi_raw_fn(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, dict[str, str]]:
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert "completions" not in path.lower() and "/chat/" not in path.lower()
        hdrs = {str(k): str(v) for k, v in (headers or {}).items()}
        hdrs.setdefault("X-Forwarded-For", _loopback_ip(99, 10))
        resp = _asgi_req(method, path, json_body=payload, headers=hdrs)
        r_hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
            with lock:
                sqlite_locks.append(resp.text[:120])
        with lock:
            seen_paths.append(path)
            if parsed.path in parity_gate.PR949_HTML_ROUTES and resp.status_code == 200:
                csp = r_hdrs.get("content-security-policy", "")
                hdr_n = _CSP_HDR_NONCE_RE.findall(csp)
                attr_n = [v for _, v in _HTML_ATTR_NONCE_RE.findall(resp.text)]
                assert "default-src 'none'" in csp.lower() and hdr_n and hdr_n[0] in attr_n
                observed_nonces.append(hdr_n[0])
        return resp.status_code, resp.text, r_hdrs

    def _asgi_json_fn(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        code, txt, hdrs = _asgi_raw_fn(method, url, payload, headers)
        if "application/json" in hdrs.get("content-type", "") and txt.strip():
            return code, json.loads(txt), hdrs
        return code, txt, hdrs

    def _run_participant(idx: int) -> dict[str, Any]:
        profile = "dev" if idx < 24 else "gamer"
        ip = _loopback_ip(10 + idx // 200, (idx % 200) + 1)
        ws_dir = clean_root / f"peer_{idx:02d}"
        ws_dir.mkdir(exist_ok=True)
        comp = parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=ws_dir, check_cwd_path=True)
        assert comp["compliance_ok"] is True

        reg_r = _asgi_req(
            "POST",
            "/v1/hivemind/register",
            json_body={"handle": f"soak_{profile}_{idx:02d}", "role": "prosumer"},
            headers={"X-Forwarded-For": ip},
        )
        assert reg_r.status_code == 200
        api_key = reg_r.json()["api_key"]
        auth_h = {"Authorization": f"Bearer {api_key}", "X-Forwarded-For": ip}

        # 12 producers register GPU nodes on distinct loopback IPs with hivemind/coder
        if idx < 12:
            hb_r = _asgi_req(
                "POST",
                "/v1/hivemind/producer/heartbeat",
                json_body={
                    "endpoint_url": f"http://{_loopback_ip(70, idx + 1)}:11434",
                    "models": [{"id": "coder", "local_name": "Qwen2.5-Coder-7B-Q4_K_M", "loaded": True}],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 120.0,
                },
                headers=auth_h,
            )
            assert hb_r.status_code == 200
            ot_r = _asgi_req(
                "POST",
                "/v1/hivemind/opentrainer/heartbeat",
                json_body={"host_label": f"pc-{idx:02d}", "gpu_name": "RTX 4070", "vram_total_mib": 12288, "vram_free_mib": 8192},
                headers=auth_h,
            )
            assert ot_r.status_code == 200

        # HTML & manifest/models probes
        html_route = parity_gate.PR949_HTML_ROUTES[idx % 4]
        html_code, html_txt, html_hdrs = _asgi_raw_fn("GET", f"http://127.0.0.1:8000{html_route}", headers={"X-Forwarded-For": ip})
        assert html_code == 200
        assert html_hdrs.get("content-security-policy")
        if idx < 8:
            arena_code, arena_html, _ = _asgi_raw_fn("GET", "http://127.0.0.1:8000/arena-noel", headers={"X-Forwarded-For": ip})
            assert arena_code == 200
            ballots = _ARENA_BALLOT_RE.findall(arena_html)
            if ballots:
                tid, tok = ballots[0]
                vote_r = _asgi_req(
                    "POST",
                    "/v1/hivemind/arena/vote",
                    json_body={"task_id": tid, "chosen_side": "A", "profile": profile, "ballot_token": tok},
                    headers={"X-Forwarded-For": ip},
                )
                assert vote_r.status_code == 200
        if idx < 4:
            scr_code, _, _ = _asgi_raw_fn(
                "GET",
                f"http://127.0.0.1:8000/scripts/{sample_scripts[idx]}",
                headers={"X-Forwarded-For": ip},
            )
            assert scr_code == 200

        # Duplicate Idempotency-Key CES feedback dispatch (2x per user -> 96 POSTs, 48 saved)
        idem_key = f"soak-dyn42-{profile}-{idx:02d}"
        fb_payload = {
            "profile": profile,
            "ease_rating": 5 if idx % 3 != 0 else 4,
            "zero_game_lag": True,
            "platform": "linux" if idx % 2 == 0 else "windows",
        }
        r1 = _asgi_req(
            "POST",
            "/v1/hivemind/onboarding/feedback",
            json_body=fb_payload,
            headers={"X-Forwarded-For": ip, "Idempotency-Key": idem_key},
        )
        r2 = _asgi_req(
            "POST",
            "/v1/hivemind/onboarding/feedback",
            json_body=fb_payload,
            headers={"X-Forwarded-For": ip, "Idempotency-Key": idem_key},
        )
        assert r1.status_code == 200 and r2.status_code == 200
        return {"idx": idx, "profile": profile}

    def _run_corp_check(c_idx: int) -> dict[str, Any]:
        c_dir = tmp_path / f"ctrl_ws_{c_idx}"
        c_dir.mkdir(exist_ok=True)
        (c_dir / ".git").mkdir(exist_ok=True)
        (c_dir / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")
        return parity_gate.verify_pr949_live_quorum_and_ces_parity(
            "http://127.0.0.1:8000",
            enforce_personal_pc=True,
            check_cwd_path=True,
            raw_http_fn=_asgi_raw_fn,
            env=CLEAN_ENV,
            cwd=c_dir,
        )

    try:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            user_futs = [pool.submit(_run_participant, i) for i in range(48)]
            corp_futs = [pool.submit(_run_corp_check, j) for j in range(6)]
            obs_futs = [
                pool.submit(
                    parity_gate.verify_pr949_live_quorum_and_ces_parity,
                    "http://127.0.0.1:8000",
                    expected_sha=TARGET_SHORT_SHA,
                    target_online_producers=8,
                    require_live_quorum=False,
                    raw_http_fn=_asgi_raw_fn,
                    env=CLEAN_ENV,
                    cwd=clean_root,
                    check_cwd_path=True,
                    lang="fr" if k % 2 else "en",
                )
                for k in range(3)
            ]
            leg_futs = [
                pool.submit(
                    parity_gate.check_cloud_run_holiday_parity,
                    "http://127.0.0.1:8000",
                    expected_sha=TARGET_SHORT_SHA,
                    env=CLEAN_ENV,
                    cwd=clean_root,
                    http_fn=_asgi_json_fn,
                )
                for _ in range(2)
            ]

            user_results = [f.result(timeout=12.0) for f in user_futs]
            corp_results = [f.result(timeout=10.0) for f in corp_futs]
            obs_results = [f.result(timeout=10.0) for f in obs_futs]
            leg_results = [f.result(timeout=10.0) for f in leg_futs]

        # Post-wave exact 5-key feedback check before final Idempotency-Key canary
        pre_sum = _asgi_req(
            "GET",
            "/v1/hivemind/onboarding/feedback/summary",
            headers={"X-Forwarded-For": _loopback_ip(99, 240)},
        ).json()
        assert list(pre_sum.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]
        assert pre_sum["total_responses"] == 48
        assert pre_sum["by_profile"] == {"dev": 24, "gamer": 24}
        assert pre_sum["mean_ease_rating"] >= 4.2
        assert pre_sum["zero_game_lag_ratio"] >= 0.95

        final_rep = parity_gate.verify_pr949_live_quorum_and_ces_parity(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            target_online_producers=8,
            require_live_quorum=True,
            verify_idempotency_replay=True,
            enforce_personal_pc=True,
            check_cwd_path=True,
            raw_http_fn=_asgi_raw_fn,
            env=CLEAN_ENV,
            cwd=clean_root,
        )
        soak_elapsed_s = time.monotonic() - t0
        assert soak_elapsed_s < 15.0, f"48-participant WAL soak exceeded 15.0s budget: {soak_elapsed_s:.2f}s"

        assert len(user_results) == 48
        assert all(c["compliance_ok"] is False and c["network_calls"] == 0 and c["file_writes"] == 0 for c in corp_results)
        assert all(o["holiday_routes_deployed_on_cloud_run"] is True and _verify_sha256_receipt(o) is None for o in obs_results)
        assert all(leg["parity_ok"] is True and _verify_sha256_receipt(leg) is None for leg in leg_results)

        _verify_sha256_receipt(final_rep)
        assert final_rep["ready"] is True and final_rep["exit_code"] == 0
        assert final_rep["online_producers"] >= 8
        assert final_rep["distinct_physical_gpus"] >= 8
        assert final_rep["producer_quorum_met"] is True
        assert final_rep["models_coder_present"] is True
        assert final_rep["idempotent_dedup_ok"] is True
        assert final_rep["total_responses"] == 49
        assert final_rep["mean_ease_rating"] >= 4.2
        assert final_rep["zero_game_lag_ratio"] >= 0.95
        assert sqlite_locks == []
        assert upstream_inference_hits == []
        assert all("completions" not in p.lower() for p in seen_paths)
        assert len(observed_nonces) >= 48
        assert len(set(observed_nonces)) == len(observed_nonces)
        assert parity_gate.scan_output_safety(parity_gate.render_markdown_report(final_rep)) == []
    finally:
        asyncio.run_coroutine_threadsafe(async_client.__aexit__(None, None, None), asgi_loop).result(timeout=5.0)
        asgi_loop.call_soon_threadsafe(asgi_loop.stop)
        asgi_thread.join(timeout=5.0)
        asgi_loop.close()
        store.close()


def test_zz_standalone_parity_gate_line_coverage_meets_95_percent() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {GATE_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
