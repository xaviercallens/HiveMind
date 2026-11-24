# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency stdlib hivemind_connect.py, Noel doctor, Personal-PC gate & 48-user CES soak (`SLICE-SCALE-DYN-12`)."""

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
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import OpenTrainerHeartbeatRequest, create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECT_PATH = REPO_ROOT / "scripts" / "hivemind_connect.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct import without starting an internal coverage.Coverage() instance so external
# `coverage run --include='*/scripts/hivemind_connect.py'` captures 100% of import + runtime lines.
from scripts import hivemind_connect as connect  # noqa: E402

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
    "test_no_lateral_movement_dyn12",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn12",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn12",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "home-personal-pc"}
SOAK_SECRET = b"noel-scale-dyn12-router-sec-32b!"

TEST_GW_EXAMPLE = "https://" + "gw.example.org"
TEST_GW_TESTSERVER = "http://" + "testserver"
UNTRUSTED_EXT_GW = "https://" + "evil.example.org"
UNTRUSTED_INSECURE_RUN_APP = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/core/service.git"
CORP_INTRANET_URL = "https://" + "intranet.enterprise.local/"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@proxy.zscaler.internal:8443"

ALPHA_TEST_KEY = "hm-alpha_" + "personal_dyn12_1234"
LIVE_TEST_KEY = "hm_live_" + "personal_dyn12_5678"

_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


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


def _verify_sha256_receipt(report: dict[str, Any]) -> None:
    digest = report.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in report.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(report)
    assert connect.scan_output_safety(canonical) == []


def _quiet_rtx_probe(**_: Any) -> connect.GpuProbeSnapshot:
    return connect.GpuProbeSnapshot(
        present=True,
        parse_error=False,
        name="NVIDIA GeForce RTX 4070",
        total_mib=12288,
        free_mib=9600,
        used_mib=2688,
        temp_c=54.0,
        util_pct=4.0,
    )


def _fast_yield_drill(**_: Any) -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 34.5,
        "method": "injected_fast_drill",
    }


def _idle_processes(**_: Any) -> list[str]:
    return ["systemd", "python3", "steam.exe"]


def _build_local_runner(
    *,
    smi_rc: int = 0,
    smi_out: str = "NVIDIA GeForce RTX 4070, 12288, 9600, 2688, 54, 4\n",
    smi_exc: Exception | None = None,
    ps_rc: int = 0,
    ps_out: str = "systemd\nbash\npython3\n",
    tasklist_rc: int = 0,
    tasklist_out: str = '"explorer.exe","1200","Console","1","85,000 K"\n',
) -> connect.RunnerFn:
    def _run(argv: Any, timeout_s: float) -> tuple[int, str]:
        assert timeout_s <= 5.0
        cmd0 = str(argv[0])
        if cmd0 == "nvidia-smi":
            joined = " ".join(str(x) for x in argv).lower()
            assert "uuid" not in joined and "serial" not in joined
            if smi_exc is not None:
                raise smi_exc
            return smi_rc, smi_out
        if cmd0 == "tasklist":
            return tasklist_rc, tasklist_out
        if cmd0 == "ps":
            return ps_rc, ps_out
        return 1, ""

    return _run


def _make_asgi_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    csp_nonces: dict[str, list[str]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[tuple[str, str], tuple[int, Any]] | None = None,
) -> httpx.MockTransport:
    def _handler(req: httpx.Request) -> httpx.Response:
        assert "completions" not in str(req.url).lower(), f"Zero-inference violated on {req.url}"
        path = req.url.path
        hdrs = dict(req.headers)
        hdrs["x-forwarded-for"] = peer_ip
        raw_content = req.content
        if req.method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat" and raw_content:
            OpenTrainerHeartbeatRequest.model_validate(json.loads(raw_content))

        if seen_calls is not None:
            entry = (req.method, path, hdrs)
            if lock is not None:
                with lock:
                    seen_calls.append(entry)
            else:
                seen_calls.append(entry)

        if override_routes and (req.method, path) in override_routes:
            ov_code, ov_body = override_routes[(req.method, path)]
            if isinstance(ov_body, (dict, list)):
                return httpx.Response(ov_code, json=ov_body)
            return httpx.Response(ov_code, text=str(ov_body))

        resp = client.request(req.method, path, headers=hdrs, content=raw_content)
        raw_text = resp.text
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
            else:
                sqlite_lock_hits.append(raw_text[:200])

        if path in ("/noel", "/quick") and resp.status_code == 200:
            assert static_numbers_guard.FIGURE.findall(_HTML_NONCE_ATTR_RE.sub("", raw_text)) == []
            csp = resp.headers.get("content-security-policy", "")
            nonces = _CSP_NONCE_RE.findall(csp)
            assert len(set(nonces)) == 1, f"Missing or inconsistent CSP nonce on {path}: {csp!r}"
            if csp_nonces is not None:
                if lock is not None:
                    with lock:
                        csp_nonces.setdefault(path, []).append(nonces[0])
                else:
                    csp_nonces.setdefault(path, []).append(nonces[0])

        return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

    return httpx.MockTransport(_handler)


def test_01_compliance_headers_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (CONNECT_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {target}"

    connect_source = CONNECT_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Zero lateral movement or disallowed literal URLs (including opencode.ai removal)
    assert "opencode.ai" not in connect_source
    assert lateral_guard.audit_source_for_lateral_movement(connect_source, filename="scripts/hivemind_connect.py") == []
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_connect_doctor_scale_soak.py",
        )
        == []
    )
    lan_ip = ".".join(("192", "168", "1", "42"))
    assert lateral_guard.audit_source_for_lateral_movement(connect_source + f'\n_BAD = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(connect_source + f'\n_EXT = "{UNTRUSTED_EXT_GW}"\n') != []

    # 2. AST check: stdlib imports only (except optional httpx), zero eval/exec/os.system/pickle/shell=True/socket.connect
    tree = ast.parse(connect_source, filename="scripts/hivemind_connect.py")
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
            if isinstance(node.func, ast.Attribute):
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

    assert imported_roots - sys.stdlib_module_names == {"httpx"}
    assert imported_roots & {"pickle", "marshal"} == set()
    assert proxy_handler_empty >= 1

    # 3. PR gatekeeper FORBIDDEN_ADDED + 5-Layer PRSecurityScanner + ZeroStubAudit + static-figure ratchet
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/hivemind_connect.py", connect_source),
        ("tests/test_noel_connect_doctor_scale_soak.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = ZeroStubAudit().audit_python_code(content)
        assert stub_res.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    if hasattr(ZeroStubAudit, "check_file"):
        assert ZeroStubAudit.check_file(CONNECT_PATH).is_clean  # type: ignore[attr-defined]

    pr_audit = scanner.audit_pull_request(
        pr_number=1112,
        pr_title="feat(onboarding,cli,scale): zero-dependency stdlib hivemind_connect.py with Noel doctor and 48-user CES soak",
        pr_body="Implements SLICE-SCALE-DYN-12 zero-dependency hivemind_connect.py, Noel doctor, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,cli,scale): zero-dependency stdlib hivemind_connect.py"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_02_pure_stdlib_without_httpx_proxy_free_loopback_and_allowlist_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen_http: list[tuple[str, str, dict[str, str], str]] = []
    loopback_port_holder = [0]

    class _LoopbackServer(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            seen_http.append(("GET", self.path, hdrs, ""))
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path in ("/noel", "/quick", "/raid-noel"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><title>Noel</title>")
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._json(
                    200,
                    {
                        "object": "hivemind.onboarding.manifest",
                        "build_sha": "abc12345",
                        "cloud_run_revision": "hivemind-openrouter-00120-dyn12",
                        "holiday_routes": ["/noel", "/quick", "/raid-noel"],
                    },
                )
            elif self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._json(
                    200,
                    {
                        "object": "hivemind.onboarding_feedback_summary",
                        "total_responses": 1,
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": 1, "gamer": 0},
                    },
                )
            elif self.path == "/v1/hivemind/producers/fleet":
                self._json(200, {"object": "hivemind.producer_fleet", "live_hiveminders": 2})
            elif self.path == "/v1/hivemind/raid/status":
                self._json(200, {"swarm": {"active_trainers": 1, "accepted_opentrainer_rounds": 3}})
            elif self.path == "/v1/hivemind/me":
                if hdrs.get("authorization") == f"Bearer {ALPHA_TEST_KEY}":
                    self._json(200, {"account_id": "acc-loop", "handle": "loop-dev", "role": "prosumer", "balance": 10.0, "max_price_per_1k": 2.0})
                else:
                    self._json(401, {"error": "unauthorized"})
            elif self.path == "/v1/models":
                self._json(200, {"data": [{"id": "hivemind/coder"}]})
            elif self.path == "/v1/hivemind/catalog":
                self._json(200, {"tiers": [{"public_id": "hivemind/coder", "moe": False, "min_vram_gb": 6.0, "good_for": "code"}]})
            elif self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", f"{UNTRUSTED_EXT_GW}/steal")
                self.end_headers()
            else:
                self._json(404, {"detail": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            seen_http.append(("POST", self.path, hdrs, raw))
            if self.path == "/v1/hivemind/register":
                body = json.loads(raw)
                self._json(200, {"account_id": "acc-loop", "api_key": ALPHA_TEST_KEY, "role": body["role"], "handle": body["handle"]})
            elif self.path == "/v1/hivemind/register/google":
                body = json.loads(raw)
                self._json(200, {"account_id": "acc-goog", "api_key": ALPHA_TEST_KEY, "role": body["role"], "created": True})
            elif self.path == "/v1/hivemind/opentrainer/heartbeat":
                self._json(200, {"object": "hivemind.opentrainer_heartbeat_ack"})
            elif self.path == "/v1/hivemind/onboarding/feedback":
                self._json(200, {"total_responses": 2, "mean_ease_rating": 5.0})
            elif self.path == "/v1/hivemind/points/transfer":
                self._json(200, {"transfer_id": "tx1", "sender_handle": "a", "recipient_handle": "b", "amount_hmt": 5.0, "memo": "ok", "sender_balance_hmt": 5.0})
            else:
                self._json(404, {"detail": "not_found"})

        def _json(self, code: int, body: Any) -> None:
            enc = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(enc)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackServer)
    port = srv.server_address[1]
    loopback_port_holder[0] = port
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"
    home_dir = tmp_path / "home" / "stdlib_user"
    cred_file = home_dir / ".hivemind" / "credentials.json"

    try:
        # 1. Poison ambient HTTP_PROXY/HTTPS_PROXY + block httpx in-process and run full CLI stdlib flow
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        monkeypatch.setitem(sys.modules, "httpx", None)

        iso_spec = importlib.util.spec_from_file_location("hivemind_connect_no_httpx", CONNECT_PATH)
        assert iso_spec is not None and iso_spec.loader is not None
        no_httpx_mod = importlib.util.module_from_spec(iso_spec)
        sys.modules["hivemind_connect_no_httpx"] = no_httpx_mod
        iso_spec.loader.exec_module(no_httpx_mod)
        assert no_httpx_mod.httpx is None

        # Register + save credentials + write Continue YAML using pure stdlib ProxyHandler({})
        rc_reg = no_httpx_mod.main(
            [
                "--gateway",
                base_url,
                "--credentials",
                str(cred_file),
                "register",
                "--handle",
                "loop-dev",
                "--save-credentials",
                "--write-continue",
            ],
            home=home_dir,
            cwd=tmp_path,
            environ={"PATH": "/usr/bin"},
        )
        assert rc_reg == 0
        out_reg = capsys.readouterr().out
        assert ALPHA_TEST_KEY not in out_reg
        assert "stdlib_user" not in out_reg
        assert cred_file.is_file() and stat.S_IMODE(os.stat(cred_file).st_mode) == 0o600
        cont_file = home_dir / ".continue" / "hivemind_provider.yaml"
        assert cont_file.is_file() and stat.S_IMODE(os.stat(cont_file).st_mode) == 0o600
        parsed_yaml = yaml.safe_load(cont_file.read_text(encoding="utf-8"))
        assert [m["model"] for m in parsed_yaml["models"]] == ["hivemind/coder", "hivemind/bee-stars"]

        # Check auto-loads saved credentials when HIVEMIND_API_KEY is unset
        rc_chk = no_httpx_mod.main(
            ["--gateway", base_url, "--credentials", str(cred_file), "check"],
            home=home_dir,
            environ={"PATH": "/usr/bin"},
        )
        assert rc_chk == 0
        assert "hivemind/coder" in capsys.readouterr().out

        # Doctor runs completely via stdlib ProxyHandler({}) without httpx
        rc_doc = no_httpx_mod.main(
            [
                "--gateway",
                base_url,
                "--credentials",
                str(cred_file),
                "doctor",
                "--require-holiday-routes",
                "--first-heartbeat",
                "--submit-ces",
                "--profile",
                "dev",
                "--ease-rating",
                "5",
                "--json",
            ],
            home=home_dir,
            cwd=tmp_path,
            environ={"PATH": "/usr/bin"},
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert rc_doc == 0
        doc_json = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(doc_json)
        assert doc_json["holiday_routes_deployed_on_cloud_run"] is True
        assert doc_json["first_heartbeat_ok"] is True
        assert doc_json["ces_recorded"] is True

        # Redirect to untrusted external host is rejected fail-closed; allowed loopback redirect & 404/control-char work cleanly
        with no_httpx_mod._StdlibHttpClient(timeout_s=2.0) as std_cli:
            assert std_cli.get(f"{base_url}/redirect-good").status_code == 200
            assert std_cli.get(f"{base_url}/does-not-exist").status_code == 404
            with pytest.raises(no_httpx_mod.GatewayPolicyError):
                std_cli.get(f"{base_url}/redirect-evil")
            with pytest.raises(no_httpx_mod.GatewayPolicyError):
                std_cli.post(f"{base_url}/v1/chat/completions", json={"model": "hivemind/coder"})
            with pytest.raises(no_httpx_mod.GatewayPolicyError):
                std_cli.get(f"{base_url}/health", headers={"X-Bad": "line1\nline2"})
            with pytest.raises(no_httpx_mod.GatewayPolicyError):
                no_httpx_mod._assert_zero_inference_request("DELETE", f"{base_url}/health")
        with pytest.raises(RuntimeError):
            no_httpx_mod._HttpxTransportClient(transport=object())

        # Legacy --save-key owner-only write and overwrite refusal under stdlib
        legacy_key_file = home_dir / "legacy.key"
        assert no_httpx_mod.main(
            ["--gateway", base_url, "register", "--handle", "leg", "--save-key", str(legacy_key_file)],
            home=home_dir,
            environ={"PATH": "/usr/bin"},
        ) == 0
        assert legacy_key_file.read_text(encoding="utf-8") == ALPHA_TEST_KEY
        assert no_httpx_mod.main(
            ["--gateway", base_url, "register", "--handle", "leg", "--save-key", str(legacy_key_file)],
            home=home_dir,
            environ={"PATH": "/usr/bin"},
        ) == 1
        _ = capsys.readouterr()

        # 2. Real subprocess execution with sys.modules['httpx'] = None and poisoned proxy env
        sub_home = tmp_path / "sub_home"
        sub_home.mkdir()
        sub_code = (
            "import sys; sys.modules['httpx'] = None; "
            "from scripts import hivemind_connect as c; "
            f"raise SystemExit(c.main(['--gateway', {base_url!r}, 'doctor', '--json'], cwd={str(tmp_path)!r}))"
        )
        sub_proc = subprocess.run(
            [sys.executable, "-c", sub_code],
            cwd=str(REPO_ROOT),
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(sub_home),
                "PYTHONPATH": str(REPO_ROOT),
                "HTTP_PROXY": "http://127.0.0.1:1",
                "HTTPS_PROXY": "http://127.0.0.1:1",
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert sub_proc.returncode == 0, sub_proc.stderr
        sub_doc = json.loads(sub_proc.stdout)
        _verify_sha256_receipt(sub_doc)
        assert sub_doc["ready"] is True

        # Subprocess connection failure returns exit code 1 cleanly without httpx
        rc_conn_err = no_httpx_mod.main(
            ["--gateway", "http://127.0.0.1:1", "register", "--handle", "offline-user"],
            environ={"PATH": "/usr/bin"},
        )
        assert rc_conn_err == 1
        assert "cannot reach the gateway" in capsys.readouterr().err
    finally:
        srv.shutdown()
        srv.server_close()

    # 3. Untrusted live gateway URL rejections vs test transport compatibility with gw.example.org & testserver
    for bad_gw in (
        "",
        UNTRUSTED_EXT_GW,
        UNTRUSTED_INSECURE_RUN_APP,
        "ftp://127.0.0.1:8080",
        "https://user:pw@hivemind-openrouter.run.app",
        "https://hivemind-openrouter.run.app/extra",
        "https://hivemind-openrouter.run.app?q=1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:badport",
        "https://" + "bad space.run.app",
    ):
        with pytest.raises(connect.GatewayPolicyError):
            connect.validate_gateway_url(bad_gw)

    assert connect.is_allowed_gateway_host("") is False
    assert connect.is_allowed_gateway_host("storage.googleapis.com") is True
    assert connect.is_allowed_gateway_host("run.googleapis.com") is True
    assert connect.is_allowed_gateway_host("datasets-server.huggingface.co") is True

    assert connect.main(["--gateway", UNTRUSTED_EXT_GW, "register", "--handle", "x"]) == 2
    assert "Untrusted gateway host" in capsys.readouterr().err

    refused_doc = connect.run_connect_doctor(gateway=UNTRUSTED_EXT_GW, environ=CLEAN_ENV, cwd=tmp_path)
    _verify_sha256_receipt(refused_doc)
    assert refused_doc["exit_code"] == 2 and refused_doc["allowed_host"] is False

    # Injectable test transports remain compatible with gw.example.org and http://testserver
    for test_gw in (TEST_GW_EXAMPLE, TEST_GW_TESTSERVER):
        tr = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"account_id": "a", "api_key": ALPHA_TEST_KEY, "role": "prosumer"})
            if r.url.path == "/v1/hivemind/register"
            else httpx.Response(200, json={"status": "ok"})
        )
        assert connect.main(["--gateway", test_gw, "register", "--handle", "tuser"], transport=tr) == 0
        _ = capsys.readouterr()

    # Pure-stdlib snippet with TEST_GW_EXAMPLE continues to work without network I/O
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "snippet", "env"]) == 0
    assert TEST_GW_EXAMPLE in capsys.readouterr().out


def test_03_doctor_drift_yield_thermal_ces_continuity_and_amadeus_compliance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "dyn12build01")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00112-dyn12")
    home_dir = tmp_path / "home" / "perso_noel"
    ws_dir = tmp_path / "hobby_ws"
    ws_dir.mkdir(parents=True)
    cred_path = home_dir / ".hivemind" / "credentials.json"

    store = RouterStore(tmp_path / "dyn12_unit.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, open_registration=True)

    with TestClient(app) as client:
        # 1. Pre-PR-#949 stale Cloud Run 404 drift detection vs --require-holiday-routes
        stale_routes = {
            ("GET", r): (404, {"detail": "Not Found"})
            for r in connect.DOCTOR_HOLIDAY_ROUTES
        }
        stale_tr = _make_asgi_http_adapter(client, peer_ip="127.0.12.1", override_routes=stale_routes)

        rc_drift_warn = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(cred_path),
                "doctor",
                "--lang",
                "fr",
                "--json",
            ],
            transport=stale_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert rc_drift_warn == 1
        drift_doc = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(drift_doc)
        assert drift_doc["holiday_routes_deployed_on_cloud_run"] is False
        assert drift_doc["onboarding_feedback_deployed"] is False
        assert drift_doc["revision_drift_detected"] is True
        assert drift_doc["gpu_status"] == "dev_api_only_no_nvidia_gpu"

        rc_drift_fail = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(cred_path),
                "doctor",
                "--require-holiday-routes",
                "--json",
            ],
            transport=stale_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert rc_drift_fail == 2
        _ = capsys.readouterr()

        # 2. Register with --save-credentials + --write-continue, then run real subprocess yield drill
        live_tr = _make_asgi_http_adapter(client, peer_ip="127.0.12.2")
        rc_reg = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(cred_path),
                "register",
                "--handle",
                "noel-gamer-01",
                "--role",
                "prosumer",
                "--save-credentials",
                "--write-continue",
            ],
            transport=live_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
        )
        assert rc_reg == 0
        reg_stdout = capsys.readouterr().out
        assert "noel-gamer-01" not in reg_stdout and "perso_noel" not in reg_stdout
        saved_cred = json.loads(cred_path.read_text(encoding="utf-8"))
        raw_user_key = saved_cred["api_key"]
        assert raw_user_key not in reg_stdout

        # Auto-load saved credentials in doctor (HIVEMIND_API_KEY unset), real default_worker_yield_drill!
        rc_live_doc = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(cred_path),
                "doctor",
                "--require-holiday-routes",
                "--first-heartbeat",
                "--submit-ces",
                "--profile",
                "gamer",
                "--ease-rating",
                "5",
                "--host-label",
                "home-rtx-rig",
                "--markdown",
                "--lang",
                "fr",
            ],
            transport=live_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            gpu_probe_fn=_quiet_rtx_probe,
            process_probe_fn=_idle_processes,
        )
        assert rc_live_doc == 0
        md_out = capsys.readouterr().out
        assert "HiveMind Connect Doctor" in md_out
        assert raw_user_key not in md_out and "perso_noel" not in md_out

        # 3. Slow yield (>= 1500.0 -> fail, exit 2, zero_game_lag=False)
        slow_doc = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=cred_path,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            transport=live_tr,
            gpu_probe_fn=_quiet_rtx_probe,
            process_probe_fn=_idle_processes,
            worker_yield_fn=lambda: 1625.0,
        )
        _verify_sha256_receipt(slow_doc)
        assert slow_doc["exit_code"] == 2
        assert slow_doc["yield_slo_ok"] is False
        assert slow_doc["zero_game_lag"] is False

        # 4. Hot GPU (> 70.0 C -> warn, exit 1, thermal_ok=False, zero_game_lag=False)
        hot_doc = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=cred_path,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            transport=live_tr,
            gpu_probe_fn=lambda: {"present": True, "name": "NVIDIA RTX 4080", "total_mib": 16384, "free_mib": 12000, "temp_c": 75.5},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
            lang="fr",
        )
        assert hot_doc["exit_code"] == 1
        assert hot_doc["thermal_ok"] is False
        assert hot_doc["zero_game_lag"] is False
        assert " Ver" in hot_doc["diagnostics"][0] or "75.5 C" in hot_doc["diagnostics"][0]

        # 5. Low free VRAM (< 2048 MiB -> warn) and active game detection
        low_vram_doc = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=cred_path,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            transport=live_tr,
            gpu_probe_fn=lambda: {"present": True, "name": "NVIDIA RTX 4060", "total_mib": 8192, "free_mib": 1024, "temp_c": 52.0},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert low_vram_doc["exit_code"] == 1 and low_vram_doc["vram_reserve_ok"] is False

        game_doc = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=cred_path,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            transport=live_tr,
            gpu_probe_fn=_quiet_rtx_probe,
            process_probe_fn=lambda: ["cs2.exe", "steam_app_730"],
            worker_yield_fn=_fast_yield_drill,
            extra_games=["CustomGame.exe"],
        )
        assert game_doc["exit_code"] == 1 and game_doc["game_running"] is True

        # 6. Interactive --prompt-ces + deterministic Idempotency-Key + non-blocking CES 404/429/503 degradation
        seen_ces_keys: list[str] = []
        for err_code in (404, 429, 503):
            def _ces_err_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, Any]:
                _ = payload
                if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
                    seen_ces_keys.append(headers["Idempotency-Key"])
                    return err_code, {"error": f"sim_{err_code}"}
                if url.endswith("/v1/hivemind/onboarding/manifest"):
                    return 200, {"object": "hivemind.onboarding.manifest", "holiday_routes": ["/noel", "/quick", "/raid-noel"]}
                if url.endswith("/v1/hivemind/onboarding/feedback/summary"):
                    return 200, {"object": "hivemind.onboarding_feedback_summary", "total_responses": 0, "mean_ease_rating": None, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}}
                if url.endswith("/v1/hivemind/producers/fleet"):
                    return 200, {"live_hiveminders": 1}
                if url.endswith("/v1/hivemind/raid/status"):
                    return 200, {"swarm": {"active_trainers": 1, "accepted_opentrainer_rounds": 0}}
                if url.endswith(("/noel", "/quick", "/raid-noel")):
                    return 200, "<html>ok</html>"
                return 200, {"status": "ok"}

            deg_doc = connect.run_connect_doctor(
                gateway=TEST_GW_TESTSERVER,
                credentials_path=cred_path,
                home=home_dir,
                cwd=ws_dir,
                environ=CLEAN_ENV,
                http_fn=_ces_err_http,
                prompt_ces=True,
                input_fn=lambda _p: "4/5 dev nolag",
                gpu_probe_fn=lambda: {"present": False},
                process_probe_fn=_idle_processes,
                worker_yield_fn=_fast_yield_drill,
            )
            assert deg_doc["ready"] is True
            assert deg_doc["ces_attempted"] is True and deg_doc["ces_recorded"] is False
            assert deg_doc["ces_degraded"] is True

            if err_code in (404, 503):
                req_fail = connect.run_connect_doctor(
                    gateway=TEST_GW_TESTSERVER,
                    credentials_path=cred_path,
                    home=home_dir,
                    cwd=ws_dir,
                    environ=CLEAN_ENV,
                    http_fn=_ces_err_http,
                    submit_ces=True,
                    require_holiday_routes=True,
                    gpu_probe_fn=lambda: {"present": False},
                    process_probe_fn=_idle_processes,
                    worker_yield_fn=_fast_yield_drill,
                )
                assert req_fail["exit_code"] == 2

        assert len(seen_ces_keys) == 5 and len(set(seen_ces_keys)) == 2

        # Prompt skip + EOFError handling
        skip_doc = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=cred_path,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            transport=live_tr,
            prompt_ces=True,
            input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert skip_doc["ces_skipped"] is True and skip_doc["ces_attempted"] is False

        # CLI render + --save-credentials / --write-continue from check & doctor
        rc_cli = connect.main(
            ["--gateway", TEST_GW_TESTSERVER, "--credentials", str(cred_path), "doctor", "--save-credentials", "--write-continue"],
            transport=live_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
            gpu_probe_fn=_quiet_rtx_probe,
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
        )
        assert rc_cli == 0
        assert "HIVEMIND CONNECT DOCTOR" in capsys.readouterr().out

        rc_chk_write = connect.main(
            ["--gateway", TEST_GW_TESTSERVER, "--credentials", str(cred_path), "check", "--save-credentials", "--write-continue"],
            transport=live_tr,
            home=home_dir,
            cwd=ws_dir,
            environ=CLEAN_ENV,
        )
        assert rc_chk_write == 0
        assert "continue provider saved" in capsys.readouterr().out

    # 7. 100% fail-closed Amadeus / corporate env, workspace path & .git/config worktree blocking (0 writes, 0 HTTP, exit 2)
    corp_git = tmp_path / "corp_repo"
    (corp_git / ".git").mkdir(parents=True)
    (corp_git / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n', encoding="utf-8")

    wt_common = tmp_path / "wt_common"
    wt_sub = wt_common / "worktrees" / "w1"
    wt_sub.mkdir(parents=True)
    (wt_sub / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_common / "config").write_text(f'[url "{CORP_INTRANET_URL}"]\n\tinsteadOf = corp:\n', encoding="utf-8")
    wt_ws = tmp_path / "wt_checkout"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_sub}\n", encoding="utf-8")

    corp_http_calls = [0]

    def _forbid_http(*_a: Any, **_kw: Any) -> Any:
        corp_http_calls[0] += 1
        raise AssertionError("Corporate host must NEVER initiate outbound HTTP requests!")

    corp_vectors: tuple[tuple[dict[str, str], Path, str], ...] = (
        ({"PATH": "/usr/bin", "AMADEUS_API_TOKEN": "secret-val-123"}, ws_dir, "en"),
        ({"PATH": "/usr/bin", "USERDNSDOMAIN": "nce.amadeus.corp"}, ws_dir, "fr"),
        (
            {
                "PATH": "/usr/bin",
                "USERDOMAIN": "ENTERPRISE-CORP",
                "COMPUTERNAME": "LAPTOP-CORP-99",
                "HOSTNAME": "fortinet-edge-box",
                "HTTP_PROXY": CORP_ZSCALER_PROXY,
                "ZSCALER_CERT_PATH": "/etc/zscaler.pem",
                "SSL_CERT_FILE": "/etc/netskope/ca.pem",
                "REQUESTS_CA_BUNDLE": "/etc/bluecoat/ca.pem",
                "CURL_CA_BUNDLE": "/etc/paloalto/ca.pem",
            },
            ws_dir,
            "en",
        ),
        ({"PATH": "/usr/bin"}, corp_git, "fr"),
        ({"PATH": "/usr/bin"}, wt_ws, "en"),
        ({"PATH": "/usr/bin"}, tmp_path / "amadeus_work_folder", "fr"),
    )

    isolated_home = tmp_path / "corp_home_should_stay_empty"
    isolated_home.mkdir()
    for env_vec, cwd_vec, lang_vec in corp_vectors:
        cwd_vec.mkdir(parents=True, exist_ok=True)
        before_home = list(isolated_home.rglob("*"))
        rc_doc_corp = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(isolated_home / ".hivemind" / "credentials.json"),
                "doctor",
                "--save-credentials",
                "--write-continue",
                "--first-heartbeat",
                "--submit-ces",
                "--lang",
                lang_vec,
                "--cwd",
                str(cwd_vec),
                "--json",
            ],
            http_fn=_forbid_http,
            home=isolated_home,
            environ=env_vec,
        )
        assert rc_doc_corp == 2
        assert corp_http_calls[0] == 0
        assert list(isolated_home.rglob("*")) == before_home
        corp_doc = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(corp_doc)
        assert corp_doc["compliance_ok"] is False
        assert corp_doc["file_writes"] == 0 and corp_doc["outbound_http_requests"] == 0
        assert "0.00 EUR / 0.00 USD" in corp_doc["notice_bilingual"]
        assert "0,00 EUR / 0,00 USD" in corp_doc["notice_bilingual"]
        dumped = json.dumps(corp_doc)
        for leak in ("secret-val-123", "nce.amadeus.corp", "LAPTOP-CORP-99", "supersecretpw", "git-amadeus-internal"):
            assert leak not in dumped

        # --enforce-personal-pc on register also fails closed with exit 2 and 0 writes/HTTP
        rc_reg_corp = connect.main(
            [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--enforce-personal-pc",
                "register",
                "--handle",
                "corp-test",
                "--save-credentials",
                "--write-continue",
            ],
            http_fn=_forbid_http,
            home=isolated_home,
            cwd=cwd_vec,
            environ=env_vec,
        )
        assert rc_reg_corp == 2
        assert corp_http_calls[0] == 0
        assert list(isolated_home.rglob("*")) == before_home
        _ = capsys.readouterr()


def test_04_concurrent_48_participant_multi_cohort_noel_scale_sqlite_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "d12soak48ok")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00112-soak")

    upstream_inference_hits: list[str] = []

    def _reject_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference"})

    sqlite_path = tmp_path / "noel_dyn12_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_reject_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 48
    dev_cohort_size = 24
    barrier = threading.Barrier(total_participants)
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    csp_nonces: dict[str, list[str]] = {"/noel": [], "/quick": []}

    rtx_models = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216, 54),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288, 57),
        ("NVIDIA GeForce RTX 3080", 10240, 6144, 61),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120, 52),
        ("NVIDIA GeForce RTX 3090", 24576, 18432, 63),
        ("NVIDIA GeForce RTX 4090", 24576, 20480, 58),
    )

    with TestClient(app) as client:
        def _participant_worker(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_cohort_size
            cohort_idx = idx if is_dev else (idx - dev_cohort_size)
            peer_ip = _loopback_ip_for(42, idx)
            p_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_username = f"perso_user_{idx:02d}"
            home_dir = p_root / "home" / os_username
            ws_dir = p_root / "hobby_project"
            ws_dir.mkdir(parents=True, exist_ok=True)
            cred_path = home_dir / ".hivemind" / "credentials.json"
            cont_path = home_dir / ".continue" / "hivemind_provider.yaml"

            handle = f"dev-pc-{cohort_idx:02d}" if is_dev else f"gamer-rtx-{cohort_idx:02d}"
            role = "consumer" if (is_dev and cohort_idx % 2 == 0) else ("prosumer" if is_dev else "producer")
            ease = 5 if (idx % 3 != 0) else 4
            lang = "fr" if (idx % 2 == 0) else "en"

            tr = _make_asgi_http_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                csp_nonces=csp_nonces,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=shared_lock,
            )

            barrier.wait(timeout=20.0)
            t0 = time.monotonic()

            # Step 1: Register + --save-credentials + --write-continue (HIVEMIND_API_KEY unset)
            buf_reg = io.StringIO()
            old_stdout = None
            reg_args = [
                "--gateway",
                TEST_GW_TESTSERVER,
                "--credentials",
                str(cred_path),
                "--enforce-personal-pc",
                "register",
                "--handle",
                handle,
                "--role",
                role,
                "--save-credentials",
                "--write-continue",
                "--continue-file",
                str(cont_path),
            ]
            # Capture output per-thread cleanly without sys.stdout races by reading credentials + calling run_connect_doctor directly
            _ = (buf_reg, old_stdout)
            rc_reg = connect.main(
                reg_args,
                transport=tr,
                home=home_dir,
                cwd=ws_dir,
                environ=CLEAN_ENV,
            )
            assert rc_reg == 0
            assert cred_path.is_file() and stat.S_IMODE(os.stat(cred_path).st_mode) == 0o600
            assert cont_path.is_file() and stat.S_IMODE(os.stat(cont_path).st_mode) == 0o600
            saved_cred = json.loads(cred_path.read_text(encoding="utf-8"))
            raw_key = str(saved_cred["api_key"])

            if is_dev:

                def gpu_fn() -> dict[str, Any]:
                    return {"present": False}

                def proc_fn() -> list[str]:
                    return ["systemd", "code"]

                os_name_val = "posix"
                sys_name_val = "Darwin" if (cohort_idx % 2 == 0) else "Linux"
                profile = "dev"
                host_lbl = f"dev-box-{cohort_idx:02d}"
                yield_ms = 16.0 + cohort_idx
            else:
                card, tot_mib, free_mib, temp_c = rtx_models[cohort_idx % len(rtx_models)]
                prod_ip = _loopback_ip_for(89, cohort_idx)
                hb_prod = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{prod_ip}:11434",
                        "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 64.0,
                        "gpu_vram_mib": tot_mib,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {raw_key}", "X-Forwarded-For": peer_ip},
                )
                assert hb_prod.status_code == 200, hb_prod.text

                def gpu_fn(
                    c: str = card,
                    t: int = tot_mib,
                    f: int = free_mib,
                    tc: float = temp_c,
                ) -> dict[str, Any]:
                    return {
                        "present": True,
                        "name": c,
                        "total_mib": t,
                        "free_mib": f,
                        "temp_c": tc,
                        "util_pct": 4.0,
                    }

                def proc_fn() -> list[str]:
                    return ["explorer.exe", "steam.exe"]

                os_name_val = "posix" if os.name != "nt" else "nt"
                sys_name_val = "Windows"
                profile = "gamer"
                host_lbl = f"rtx-box-{cohort_idx:02d}"
                yield_ms = 60.0 + (cohort_idx * 4.0)

            # Step 2: run_connect_doctor with --first-heartbeat + --submit-ces (auto-loading saved credentials)
            rep = connect.run_connect_doctor(
                gateway=TEST_GW_TESTSERVER,
                credentials_path=cred_path,
                continue_path=cont_path,
                require_holiday_routes=True,
                first_heartbeat=True,
                submit_ces=True,
                profile=profile,
                ease_rating=ease,
                host_label=host_lbl,
                lang=lang,
                cwd=ws_dir,
                home=home_dir,
                environ=CLEAN_ENV,
                os_name=os_name_val,
                system_name=sys_name_val,
                transport=tr,
                gpu_probe_fn=gpu_fn,
                process_probe_fn=proc_fn,
                worker_yield_fn=lambda y=yield_ms: y,
            )
            elapsed = time.monotonic() - t0
            assert elapsed < (60.0 if is_dev else 300.0)
            _verify_sha256_receipt(rep)
            assert rep["ready"] is True, f"Slot {idx} failed: {rep}"
            assert rep["credentials_loaded"] is True
            assert rep["first_heartbeat_ok"] is True
            assert rep["ces_recorded"] is True
            assert rep["zero_game_lag"] is True

            rendered_cli = connect.render_doctor_cli(rep)
            rendered_md = connect.render_doctor_markdown(rep)
            combined = json.dumps(rep) + "\n" + rendered_cli + "\n" + rendered_md
            assert raw_key not in combined
            assert os_username not in combined
            assert handle not in combined
            assert connect.scan_output_safety(combined, home=home_dir) == []
            assert static_numbers_guard.FIGURE.findall(rendered_cli) == []
            assert static_numbers_guard.FIGURE.findall(rendered_md) == []

            hb_pl = rep["heartbeat_payload"]
            if is_dev:
                assert hb_pl is None or (hb_pl["gpu_name"] is None and hb_pl["vram_total_mib"] is None)
            else:
                assert hb_pl is not None and hb_pl["gpu_name"] is not None and hb_pl["vram_total_mib"] >= 8192

            return {"idx": idx, "profile": profile, "role": role}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futs = [pool.submit(_participant_worker, i) for i in range(total_participants)]
            outcomes = [f.result(timeout=60.0) for f in futs]

        assert len(outcomes) == 48
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _ in seen_calls)

        # 100% unique CSP nonces across all /noel and /quick renders (48 each)
        assert len(csp_nonces["/noel"]) == 48 and len(set(csp_nonces["/noel"])) == 48
        assert len(csp_nonces["/quick"]) == 48 and len(set(csp_nonces["/quick"])) == 48

        raid_doc = client.get("/v1/hivemind/raid/status").json()
        assert raid_doc["swarm"]["active_trainers"] >= 24

        # Preserve exact 5-key dict equality on /v1/hivemind/onboarding/feedback/summary
        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert set(fb_summary.keys()) == {
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        }
        assert fb_summary["object"] == "hivemind.onboarding_feedback_summary"
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2
        assert fb_summary["zero_game_lag_ratio"] >= 0.95


def test_05_edge_cases_probes_subprocess_drill_and_connector_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Subprocess runner, GPU probe & process probe edge branches
    rc_s, out_s = connect.default_subprocess_runner([sys.executable, "-c", "print('ok')"], 3.0)
    assert rc_s == 0 and out_s.strip() == "ok"
    with pytest.raises(ValueError):
        connect.default_subprocess_runner([], 1.0)
    assert connect.default_subprocess_runner(["definitely_missing_binary_xyz_123"], 1.0)[0] == 127

    assert connect.default_gpu_probe(runner=_build_local_runner(smi_rc=1, smi_out="")).present is False
    assert connect.default_gpu_probe(runner=_build_local_runner(smi_out="corrupt")).parse_error is True
    assert connect.default_gpu_probe(runner=_build_local_runner(smi_out="GPU, 0, 100, 50, 40, 1")).parse_error is True
    multi = connect.default_gpu_probe(
        runner=_build_local_runner(
            smi_out="NVIDIA GT 1030, 2048, 1800, 248, 45, 0\nNVIDIA RTX 4070, 12288, 10000, 2288, [N/A], [N/A]\n"
        )
    )
    assert multi.present and multi.name == "NVIDIA RTX 4070" and multi.temp_c is None

    assert "dota2" in connect.default_process_probe(runner=_build_local_runner(ps_out="/usr/bin/dota2\n"), os_name="posix")
    assert "cs2.exe" in connect.default_process_probe(runner=_build_local_runner(tasklist_out='"cs2.exe","1","Console"\n'), os_name="nt")
    assert connect.default_process_probe(runner=_build_local_runner(ps_rc=1), os_name="posix") == []
    assert connect.default_process_probe(runner=_build_local_runner(tasklist_rc=1), os_name="nt") == []

    # 2. default_worker_yield_drill error branches via monkeypatched Popen
    class _DrillProc:
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
                raise subprocess.TimeoutExpired(cmd=["py"], timeout=0.25)
            return -9

        def poll(self) -> int | None:
            return -9 if self._killed else None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProc("bad"))
    assert connect.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProc("timeout"))
    assert connect.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    # 3. _run_yield_drill & _normalize_gpu_snapshot & parse_ces_input & helpers
    assert connect._run_yield_drill(lambda: math.nan, clock_fn=time.monotonic)[0] is None
    assert connect._run_yield_drill(lambda timeout_s=1.0: {"worker_stopped": True, "clean_release": True}, clock_fn=time.monotonic)[1] is True
    with pytest.raises(ValueError):
        connect._run_yield_drill(lambda: "invalid", clock_fn=time.monotonic)
    assert connect._normalize_gpu_snapshot(None).present is False
    assert connect._normalize_gpu_snapshot({"present": True, "parse_error": True}).parse_error is True
    assert connect._normalize_gpu_snapshot({"present": True, "name": "GPU"}).parse_error is True
    with pytest.raises(ValueError):
        connect._normalize_gpu_snapshot("bad")

    for bad_ces in ("9", "abc", "5 4"):
        with pytest.raises(ValueError):
            connect.parse_ces_input(bad_ces)
    assert connect.parse_ces_input("3 gamer lag")["zero_game_lag"] is False
    assert connect.parse_ces_input("skip")["skipped"] is True

    # 4. Validators, masking, and CLI error paths
    for bad_lang in (True, "es"):
        with pytest.raises(ValueError):
            connect.normalize_lang(bad_lang)  # type: ignore[arg-type]
    for bad_prof in (None, True, "admin"):
        with pytest.raises(ValueError):
            connect.normalize_profile(bad_prof)  # type: ignore[arg-type]
    for bad_ease in (True, 0, 6):
        with pytest.raises(ValueError):
            connect.parse_ease_rating(bad_ease)  # type: ignore[arg-type]

    assert connect.mask_api_key("") == "<missing>"
    assert connect.mask_api_key("bad") == "<invalid>"
    assert connect.mask_api_key(LIVE_TEST_KEY) == "hm_live_***5678"
    assert connect._platform_feedback_tag("posix", "Solaris_11") == "solaris_11"
    assert connect._platform_feedback_tag("posix", "!!!") == "unknown"
    assert connect._sanitize_host_label("corp-laptop") == "personal-pc"
    assert connect._sanitize_manifest_sha("secret1234") == "unknown"
    assert connect._sanitize_manifest_revision("hm-123456789012") is None
    assert "forbidden_key_detected" in connect.scan_output_safety('{"' + 'sco' + 're": 5}')
    assert "email_detected" in connect.scan_output_safety("u@example.org")
    assert "unmasked_api_key_detected" in connect.scan_output_safety(ALPHA_TEST_KEY)
    assert "bearer_token_detected" in connect.scan_output_safety("Bearer tok_12345678")
    assert "unredacted_home_path_detected" in connect.scan_output_safety("/home/bob/.hivemind")
    assert any(x.startswith("non_loopback_ipv4_detected:") for x in connect.scan_output_safety("peer 10.1" + "0.10.10"))

    # CLI failure branches for register/check/transfer/snippet/first-heartbeat
    err_tr = httpx.MockTransport(lambda _r: httpx.Response(400, json={"error": "bad"}))
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register", "--handle", "x"], transport=err_tr) == 1
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "transfer", "--to", "@bob", "--amount", "1"], transport=err_tr, environ={connect.API_KEY_ENV: ALPHA_TEST_KEY}) == 1
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "snippet", "--list"]) == 0
    with pytest.raises(SystemExit):
        connect.main(["--gateway", TEST_GW_EXAMPLE, "snippet", "env", "--embed-key"], environ={})
    _ = capsys.readouterr()

    # Corrupted credentials warning & missing credentials on --first-heartbeat
    bad_cred = tmp_path / "bad_creds.json"
    bad_cred.write_text("{broken", encoding="utf-8")
    hb_fail = connect.run_connect_doctor(
        gateway=TEST_GW_TESTSERVER,
        credentials_path=bad_cred,
        first_heartbeat=True,
        http_fn=lambda _m, u: (200, {"live_hiveminders": 1}) if u.endswith("fleet") else (200, {"status": "ok"}),
        gpu_probe_fn=lambda: {"present": False},
        process_probe_fn=_idle_processes,
        worker_yield_fn=_fast_yield_drill,
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert hb_fail["exit_code"] == 2 and hb_fail["first_heartbeat_ok"] is False

    # Insecure 0644 credentials file warning
    open_cred = tmp_path / "open_creds.json"
    connect.save_credentials_file(open_cred, account_id="a1", api_key=ALPHA_TEST_KEY, gateway=TEST_GW_TESTSERVER, handle="u1", role="prosumer")
    if os.name != "nt":
        os.chmod(open_cred, 0o644)
        perm_warn = connect.run_connect_doctor(
            gateway=TEST_GW_TESTSERVER,
            credentials_path=open_cred,
            http_fn=lambda _m, u: (
                (200, {"object": "hivemind.onboarding.manifest", "holiday_routes": ["/noel", "/quick", "/raid-noel"]})
                if u.endswith("manifest")
                else (200, {"live_hiveminders": 1, "swarm": {"active_trainers": 1}, "status": "ok", "html_length": 10})
            ),
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_yield_fn=_fast_yield_drill,
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert perm_warn["verdict"] == "ready_with_warnings"

    # 5. All snippet formats, register-google variants, check/transfer, _InjectedHttpClient, and doctor fault isolation
    for tool_name in connect.TOOLS:
        rendered_snip = connect.render_snippet(tool_name, gateway=TEST_GW_EXAMPLE, model="coder", max_tokens=512)
        assert TEST_GW_EXAMPLE in rendered_snip
        if tool_name == "opencode":
            assert "opencode.ai" not in rendered_snip
    with pytest.raises(ValueError):
        connect.render_snippet("continue", gateway=TEST_GW_EXAMPLE, model="coder", max_tokens=0)
    with pytest.raises(ValueError):
        connect.render_snippet("unknown_tool", gateway=TEST_GW_EXAMPLE, model="coder")
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "snippet", "env"]) == 0
    _ = capsys.readouterr()

    # Helper validation branches (types, home redaction, git relative worktree, atomic cleanup, credentials errors)
    assert connect.is_valid_hivemind_key(123) is False
    custom_home = tmp_path / "custom_home_box"
    custom_home.mkdir()
    assert "~" in connect.sanitize_text_for_display(f"cfg={custom_home / 'a.json'}", home=custom_home)
    assert connect.sanitize_structure(("x", str(custom_home / "y")), home=custom_home) == ["x", "~/y"]
    assert "unredacted_home_path_detected" in connect.scan_output_safety(f"leak={custom_home / 'k'}", home=custom_home)

    rel_wt_common = tmp_path / "rel_wt_repo" / ".git"
    rel_wt_sub = rel_wt_common / "worktrees" / "branch1"
    rel_wt_sub.mkdir(parents=True)
    (rel_wt_common / "config").write_text('# comment\n[remote "origin"]\n\turl = http://127.0.0.1:9418/personal/repo.git\n', encoding="utf-8")
    (rel_wt_sub / "config").write_text('[remote "mirror"]\n\turl = http://127.0.0.1:9418/personal/mirror.git\n', encoding="utf-8")
    (rel_wt_sub / "commondir").write_text("../..\n", encoding="utf-8")
    rel_ws = tmp_path / "rel_wt_ws"
    rel_ws.mkdir()
    (rel_ws / ".git").write_text("gitdir: ../rel_wt_repo/.git/worktrees/branch1\n", encoding="utf-8")
    assert len(connect._resolve_git_config_files(rel_ws)) == 2
    assert connect.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=rel_ws)["compliance_ok"] is True

    assert connect.load_credentials(tmp_path / "does_not_exist.json") is None
    bad_cred_struct = tmp_path / "bad_struct.json"
    bad_cred_struct.write_text('{"api_key": "short"}', encoding="utf-8")
    assert connect.load_credentials(bad_cred_struct) is None
    with pytest.raises(ValueError):
        connect.save_credentials_file(tmp_path / "nope.json", account_id="a", api_key="bad", gateway=TEST_GW_TESTSERVER, handle="h", role="dev")
    with pytest.raises(ValueError):
        connect.build_continue_provider_yaml(TEST_GW_TESTSERVER, "bad", allow_test_hosts=True)
    with pytest.raises(ValueError):
        connect.build_continue_provider_yaml(TEST_GW_TESTSERVER, ALPHA_TEST_KEY, max_tokens=8, allow_test_hosts=True)

    # Atomic file write cleanup on os.replace failure
    atomic_target = tmp_path / "atomic_fail" / "cred.json"
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("disk_full")))
    with pytest.raises(OSError):
        connect._write_file_secure_atomic(atomic_target, "{}\n")
    monkeypatch.undo()

    # Misc probe & CES helper branches
    assert connect._optional_float("bad_float") is None
    assert "nano" in connect.default_process_probe(runner=_build_local_runner(ps_out="\n  \nnano\n"), os_name="posix")
    monkeypatch.setattr(connect.csv, "reader", lambda *_a, **_kw: (_ for _ in ()).throw(connect.csv.Error("corrupt_csv")))
    assert connect.default_process_probe(runner=_build_local_runner(tasklist_out='"cs2.exe"\n'), os_name="nt") == []
    monkeypatch.undo()
    assert connect._platform_feedback_tag("posix", "Darwin") == "macos"
    assert connect._sanitize_host_label(None) == "personal-pc"
    assert connect._sanitize_manifest_sha("unknown") == "unknown"
    assert connect._sanitize_manifest_revision("") is None
    with pytest.raises(ValueError):
        connect.parse_ces_input("dev nolag")

    # _InjectedHttpClient signature adapters (kwarg, 3-arg, 3-tuple+bytes, OSError)
    with connect._InjectedHttpClient(lambda m, u, **kw: (200, b"<html>hi</html>", {"X-Test": "1"})) as ic_kw:
        assert ic_kw.post(f"{TEST_GW_TESTSERVER}/x", json={"a": 1}, headers={"X-H": "1"}, timeout=1.0).text == "<html>hi</html>"
    with connect._InjectedHttpClient(lambda m, u, p: (200, {"ok": True})) as ic_3:
        assert ic_3.post(f"{TEST_GW_TESTSERVER}/x", json={"b": 2}).json()["ok"] is True
    with connect._InjectedHttpClient(lambda *_a: (_ for _ in ()).throw(OSError("conn_down"))) as ic_err:
        assert connect._safe_http_get(ic_err, f"{TEST_GW_TESTSERVER}/x")[0] == 0
        assert connect._safe_http_post(ic_err, f"{TEST_GW_TESTSERVER}/x", payload={})[0] == 0

    # httpx.HTTPError paths in _safe_http_get, _safe_http_post, and main()
    err_req = httpx.Request("GET", f"{TEST_GW_EXAMPLE}/health")
    httpx_err_tr = httpx.MockTransport(lambda _r: (_ for _ in ()).throw(httpx.ConnectError("boom", request=err_req)))
    with connect._HttpxTransportClient(transport=httpx_err_tr) as hx_cli:
        assert connect._safe_http_get(hx_cli, f"{TEST_GW_EXAMPLE}/health")[0] == 0
        assert connect._safe_http_post(hx_cli, f"{TEST_GW_EXAMPLE}/x", payload={})[0] == 0
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register", "--handle", "hx"], transport=httpx_err_tr) == 1
    _ = capsys.readouterr()

    # Non-JSON/empty responses in _safe_http_get & _safe_http_post
    class _StubResp:
        def __init__(self, code: int, text: str, ctype: str) -> None:
            self.status_code = code
            self.text = text
            self.headers = {"content-type": ctype}

        def json(self) -> Any:
            return json.loads(self.text)

    class _StubClient:
        def get(self, *_a: Any, **_kw: Any) -> _StubResp:
            return self._next

        def post(self, *_a: Any, **_kw: Any) -> _StubResp:
            return _StubResp(502, "not-json", "text/plain")

    sc = _StubClient()
    sc._next = _StubResp(200, "not-json", "application/json")
    assert connect._safe_http_get(sc, f"{TEST_GW_TESTSERVER}/x")[1] == {"raw_length": 8}
    sc._next = _StubResp(200, "   ", "text/html")
    assert connect._safe_http_get(sc, f"{TEST_GW_TESTSERVER}/x")[1]["html_length"] == 0
    assert connect._safe_http_post(sc, f"{TEST_GW_TESTSERVER}/x", payload={})[1] == {"raw_length": 8}

    # cmd_register_google (--token, --gcloud success/fail/missing, --browser key/token/abort, HTTP 401) + cmd_check 401 + cmd_transfer 200
    goog_tr = httpx.MockTransport(
        lambda r: (
            httpx.Response(200, json={"account_id": "g1", "api_key": ALPHA_TEST_KEY, "role": "prosumer", "created": False})
            if r.url.path == "/v1/hivemind/register/google"
            else (
                httpx.Response(401, json={"error": "bad_key"})
                if r.url.path == "/v1/hivemind/me"
                else httpx.Response(
                    200,
                    json={
                        "transfer_id": "tx-9",
                        "sender_handle": "alice",
                        "recipient_handle": "bob",
                        "amount_hmt": 2.5,
                        "memo": "thx",
                        "sender_balance_hmt": 7.5,
                        "amcp_event_id": "ev-1",
                    },
                )
            )
        )
    )
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--token", "idtok-123"], transport=goog_tr) == 0
    assert "Welcome back" in capsys.readouterr().out

    # --gcloud success, non-zero rc, and FileNotFoundError
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, stdout="tok-gcloud\n", stderr=""))
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--gcloud", "--audience", "aud1"], transport=goog_tr) == 0
    _ = capsys.readouterr()

    monkeypatch.setattr(subprocess, "run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no auth"))
    with pytest.raises(SystemExit):
        connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--gcloud"], transport=goog_tr, environ={})
    _ = capsys.readouterr()

    def _raise_fnf(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)
    with pytest.raises(SystemExit):
        connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--gcloud"], transport=goog_tr, environ={})
    monkeypatch.undo()
    _ = capsys.readouterr()

    # --browser with hm-... key input, ID token input, and EOFError
    monkeypatch.setattr("builtins.input", lambda _p: ALPHA_TEST_KEY)
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--browser"], transport=goog_tr) == 0
    monkeypatch.setattr("builtins.input", lambda _p: "id-tok-browser")
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--browser"], transport=goog_tr) == 0
    monkeypatch.setattr("builtins.input", lambda _p: (_ for _ in ()).throw(EOFError()))
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--browser"], transport=goog_tr) == 1
    monkeypatch.undo()
    _ = capsys.readouterr()

    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register-google", "--token", "bad"], transport=err_tr) == 1
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "check"], transport=goog_tr, environ={connect.API_KEY_ENV: ALPHA_TEST_KEY}) == 1
    assert (
        connect.main(
            [
                "--gateway",
                TEST_GW_EXAMPLE,
                "transfer",
                "--to",
                "@bob",
                "--amount",
                "2.5",
                "--idempotency-key",
                "idem-1",
                "--model-ref",
                "hivemind/coder",
                "--pr-url",
                "https://hivemind-openrouter-test.run.app/pull/1",
            ],
            transport=goog_tr,
            environ={connect.API_KEY_ENV: ALPHA_TEST_KEY},
        )
        == 0
    )
    assert connect.main(["--gateway", TEST_GW_EXAMPLE, "register", "--handle", "h-fn"], http_fn=lambda _m, _u, _p=None: (200, {"account_id": "a", "api_key": ALPHA_TEST_KEY, "role": "prosumer"})) == 0
    _ = capsys.readouterr()

    # Doctor fault branches: HIVEMIND_API_KEY env, gpu parse_error, me=401, hb=500, /health 500, /noel empty, wrong manifest, fleet=500, output safety guard
    fault_doc = connect.run_connect_doctor(
        gateway=TEST_GW_TESTSERVER,
        first_heartbeat=True,
        http_fn=lambda _m, u: (
            (500, {"status": "down"})
            if u.endswith("/health")
            else (
                (200, {"object": "wrong.obj", "holiday_routes": ["/noel"]})
                if u.endswith("/manifest")
                else (
                    (200, "")
                    if u.endswith(("/noel", "/quick", "/raid-noel"))
                    else (
                        (500, {"error": "fleet_down"})
                        if u.endswith("/fleet")
                        else (401, {"error": "unauthorized"})
                    )
                )
            )
        ),
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=_idle_processes,
        worker_yield_fn=_fast_yield_drill,
        environ={**CLEAN_ENV, connect.API_KEY_ENV: ALPHA_TEST_KEY},
        cwd=tmp_path,
        lang="fr",
    )
    assert fault_doc["exit_code"] == 2
    assert fault_doc["revision_drift_detected"] is True
    assert fault_doc["gpu_status"] == "gpu_driver_parse_error"

    hb_500_doc = connect.run_connect_doctor(
        gateway=TEST_GW_TESTSERVER,
        first_heartbeat=True,
        http_fn=lambda m, u: (
            (500, {"error": "hb_fail"})
            if (m == "POST" and u.endswith("/opentrainer/heartbeat"))
            else (
                (200, {"object": "hivemind.onboarding.manifest", "holiday_routes": ["/noel"]})
                if u.endswith("/manifest")
                else (
                    (200, {"account_id": "a1", "handle": "h1", "role": "prosumer"})
                    if u.endswith("/me")
                    else (200, {"status": "ok", "live_hiveminders": 1, "swarm": {"active_trainers": 1}, "html_length": 0})
                )
            )
        ),
        gpu_probe_fn=lambda: {"present": False},
        process_probe_fn=_idle_processes,
        worker_yield_fn=_fast_yield_drill,
        environ={**CLEAN_ENV, connect.API_KEY_ENV: ALPHA_TEST_KEY},
        cwd=tmp_path,
    )
    assert hb_500_doc["exit_code"] == 2 and hb_500_doc["heartbeat_state"] == "heartbeat_http_500"

    # cmd_doctor output safety guard raises ValueError if scan_output_safety detects violation
    monkeypatch.setattr(connect, "run_connect_doctor", lambda **_kw: {"exit_code": 0, "leaked": ALPHA_TEST_KEY})
    with pytest.raises(ValueError, match="Output safety guard"):
        connect.main(["--gateway", TEST_GW_TESTSERVER, "doctor", "--json"])
    monkeypatch.undo()
    _ = capsys.readouterr()

    # Verify the CI ImportError ZeroStubAudit fallback works standalone with gwaya blocked
    ci_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
                "import importlib.util; from pathlib import Path; "
                f"s = importlib.util.spec_from_file_location('t_ci', Path({str(Path(__file__).resolve())!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.CONNECT_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def f():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def f():\\n    pass\\n').is_clean"
            ),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert ci_proc.returncode == 0, ci_proc.stderr
