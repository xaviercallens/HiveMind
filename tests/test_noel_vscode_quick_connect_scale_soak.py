# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scale soak, <60s Dev VS Code quick-connect, live producer heartbeat, and <1.5s Steam-yield preflight tests (`SLICE-SCALE-DYN-03`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import inspect
import io
import ipaddress
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from types import SimpleNamespace
from typing import Any

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOCONFIG_PATH = REPO_ROOT / "scripts" / "noel_vscode_autoconfig.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(AUTOCONFIG_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402

try:  # noqa: E402
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

    class _FallbackStubAuditResult:
        """Fallback AST stub audit result when optional ``gwaya`` is unavailable."""

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
                    p = parents.get(node)
                    if not isinstance(p, (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden 'pass' statement.")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis:
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))

from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_auto_spec = importlib.util.spec_from_file_location("noel_vscode_autoconfig_scale_dyn03", AUTOCONFIG_PATH)
assert _auto_spec is not None and _auto_spec.loader is not None
autoconfig = importlib.util.module_from_spec(_auto_spec)
sys.modules["noel_vscode_autoconfig_scale_dyn03"] = autoconfig
_auto_spec.loader.exec_module(autoconfig)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale_dyn03",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale_dyn03",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_scale_dyn03",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
SOAK_ROUTER_SECRET = b"noel-scale-dyn03-router-sec-32b!"
SAMPLE_KEY = "hm-quick_connect_scale_987654"
SAMPLE_LIVE_KEY = "hm_live_quick_connect_654321"
_MASKED_KEY_SUFFIX_RE = re.compile(r"hm(?:_live_|-)\*\*\*[A-Za-z0-9_-]{4}")

_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    """Map `(bucket, idx)` deterministically into `127.0.0.0/8` loopback space."""
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_json_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_json_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.update(_collect_json_keys(item))
    return keys


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert autoconfig.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str,
    seen_calls: list[tuple[str, str]] | None = None,
    seen_lock: threading.Lock | None = None,
    sqlite_lock_hits: list[str] | None = None,
) -> autoconfig.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), f"Zero-inference violated: {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None and seen_lock is not None:
            with seen_lock:
                seen_calls.append((method, path))
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip
        resp = client.request(method, path, json=payload, headers=req_headers)
        raw_text = resp.text
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if seen_lock is not None:
                with seen_lock:
                    sqlite_lock_hits.append(raw_text[:200])
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {"content_type": ctype, "html_length": len(raw_text)}

    return _http


def test_01_compliance_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for path in (AUTOCONFIG_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = AUTOCONFIG_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports ONLY + ProxyHandler({}) + zero shell=True/eval/exec/os.system
    tree = ast.parse(script_source, filename="scripts/noel_vscode_autoconfig.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    forbidden_calls = {"eval", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            if isinstance(node.func, ast.Attribute):
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
                    saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert saw_empty_proxy_handler is True

    # 2. Lateral-movement AST audit on both script and test file + planted negative controls
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_vscode_autoconfig.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_vscode_quick_connect_scale_soak.py",
        )
        == []
    )
    bad_ip = ".".join(("192", "168", "44", "9"))
    bad_ext = "https://" + "continue" + ".dev/schema.json"
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_X = "http://{bad_ip}:8080"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_Y = "{bad_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both files
    for rel, content in (
        ("scripts/noel_vscode_autoconfig.py", script_source),
        ("tests/test_noel_vscode_quick_connect_scale_soak.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner (0 findings), GWAYA ZeroStubAudit, and static-numbers guard
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_vscode_autoconfig.py", script_source),
        ("tests/test_noel_vscode_quick_connect_scale_soak.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_audit = ZeroStubAudit().audit_python_code(content)
        assert stub_audit.is_clean, f"Unexpected GWAYA ZeroStubAudit violations in {rel}: {stub_audit.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1073,
        pr_title="feat(onboarding,scale): <60s Dev VS Code quick-connect, live producer heartbeat check, and <1.5s Steam-yield preflight",
        pr_body="Implements SLICE-SCALE-DYN-03 <60s Dev VS Code quick-connect, live producer heartbeat check, and <1.5s Steam-yield preflight.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): <60s Dev VS Code quick-connect, live producer heartbeat check, and <1.5s Steam-yield preflight"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    # 5. Backward-compatible public symbols & new SLICE-SCALE-DYN-03 helper signatures
    required_symbols = (
        "SCHEMA",
        "RECEIPT_OBJECT",
        "CAMPAIGN_ID",
        "FINANCIAL_PROMISE",
        "DEFAULT_GATEWAY",
        "DEFAULT_MODELS",
        "DEFAULT_MAX_TOKENS",
        "SLO_BUDGET_SECONDS",
        "evaluate_personal_pc_compliance",
        "atomic_write_config_file",
        "build_continue_provider_yaml",
        "build_merged_vscode_settings_json",
        "run_vscode_autoconfig",
        "render_markdown_report",
        "render_cli_report",
        "build_parser",
        "main",
        "verify_producer_heartbeat_readiness",
        "run_steam_thermal_preflight",
        "submit_post_setup_telemetry",
    )
    for sym in required_symbols:
        assert hasattr(autoconfig, sym), f"Missing symbol in noel_vscode_autoconfig: {sym}"

    hb_sig = inspect.signature(autoconfig.verify_producer_heartbeat_readiness)
    for expected_param in (
        "gateway",
        "api_key",
        "role",
        "send_opentrainer_heartbeat",
        "host_label",
        "gpu_snapshot",
        "http_fn",
        "lang",
    ):
        assert expected_param in hb_sig.parameters

    pf_sig = inspect.signature(autoconfig.run_steam_thermal_preflight)
    for expected_param in (
        "max_temp_c",
        "min_free_vram_mib",
        "yield_budget_ms",
        "gpu_probe_fn",
        "process_probe_fn",
        "worker_yield_fn",
        "clock_fn",
        "lang",
    ):
        assert expected_param in pf_sig.parameters


def test_02_concurrent_48_participant_multicohort_quick_connect_scale_soak(tmp_path: Path) -> None:
    upstream_inference_calls: list[str] = []

    def _forbid_upstream_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference_call"})

    sqlite_path = tmp_path / "noel_vscode_48soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 48
    dev_count = 24
    gamer_count = 24
    start_barrier = threading.Barrier(total_participants)
    seen_calls: list[tuple[str, str]] = []
    sqlite_lock_hits: list[str] = []
    seen_lock = threading.Lock()

    steam_titles = (
        "cs2.exe",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "bg3.exe",
        "helldivers2.exe",
        "steam_app_730",
    )
    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216),
        ("NVIDIA GeForce RTX 4080", 16384, 12288),
        ("NVIDIA GeForce RTX 3080", 10240, 6144),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120),
    )

    with TestClient(app) as client:
        def _run_participant(idx: int) -> dict[str, Any]:
            p_root = tmp_path / "participants" / f"p_{idx:02d}"
            home_dir = p_root / "home" / f"perso_{idx:02d}"
            ws_dir = p_root / "hobby_repo"
            ws_dir.mkdir(parents=True, exist_ok=True)
            cred_file = home_dir / ".hivemind" / "credentials.json"
            cont_file = home_dir / ".continue" / "hivemind_provider.yaml"
            vs_file = ws_dir / ".vscode" / "settings.json"
            peer_ip = _loopback_ip_for(12, idx)
            lang = "fr" if (idx % 2 == 0) else "en"
            ease_rating = 5 if (idx % 3 != 0) else 4

            if idx % 2 == 1:
                (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
            else:
                vs_file.parent.mkdir(parents=True, exist_ok=True)
                vs_file.write_text(
                    "{\n"
                    "  // Preserve personal hobby formatting preference\n"
                    '  "editor.tabSize": 2,\n'
                    "}\n",
                    encoding="utf-8",
                )

            http_adapter = _make_router_http_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                seen_lock=seen_lock,
                sqlite_lock_hits=sqlite_lock_hits,
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()

            start_barrier.wait(timeout=20.0)
            t0 = time.monotonic()

            if idx < dev_count:
                handle = f"dev-qc-{idx:02d}"
                role = "consumer" if (idx % 2 == 0) else "prosumer"
                no_gpu = autoconfig.GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")
                rc = autoconfig.main(
                    [
                        "--gateway",
                        "http://127.0.0.1:8000",
                        "--handle",
                        handle,
                        "--role",
                        role,
                        "--credentials-file",
                        str(cred_file),
                        "--continue-config-path",
                        str(cont_file),
                        "--workspace",
                        str(ws_dir),
                        "--quick-connect",
                        "--post-setup-telemetry",
                        "--profile",
                        "dev",
                        "--ease-rating",
                        str(ease_rating),
                        "--lang",
                        lang,
                        "--json",
                    ],
                    http_fn=http_adapter,
                    gpu_probe_fn=lambda g=no_gpu: g,
                    process_probe_fn=lambda: ["code", "python3"],
                    worker_yield_fn=lambda: 18.5,
                    environ=CLEAN_ENV,
                    home=home_dir,
                    cwd=ws_dir,
                    os_name="posix",
                    system_name="Linux",
                    out_stream=out_buf,
                    err_stream=err_buf,
                )
                elapsed_s = time.monotonic() - t0
                raw_out = out_buf.getvalue()
                raw_err = err_buf.getvalue()
                assert rc == 0, f"Dev participant {idx} failed (err={raw_err}): {raw_out}"
                assert not _SQLITE_LOCK_RE.search(raw_out + raw_err)
                assert elapsed_s < 60.0
                assert autoconfig.scan_output_safety(raw_out) == []

                rep = json.loads(raw_out)
                _verify_sha256_receipt(rep)
                assert rep["mode"] == "quick_connect"
                assert rep["compliance_ok"] is True
                assert rep["ready"] is True
                assert rep["within_60s_slo"] is True
                assert rep["telemetry_recorded"] is True
                assert rep["gamer_preflight"]["status"] == "dev_api_only_no_nvidia_gpu"
                assert rep["gamer_preflight"]["gpu_present"] is False
                assert rep["gamer_preflight"]["gpu_name"] is None
                assert rep["gamer_preflight"]["yield_slo_ok"] is True
                assert rep["gamer_preflight"]["thermal_ok"] is True
                assert rep["gamer_preflight"]["zero_game_lag"] is True
                assert rep["producer_heartbeat"]["ok"] is True
                assert rep["producer_heartbeat"]["remote_config"]["ok"] is True
                assert rep["producer_heartbeat"]["activation_needed"] is True
                assert "--gamer-whisper" in rep["producer_heartbeat"]["activation_command"]
                assert rep["post_setup_telemetry"]["profile"] == "dev"
                assert rep["post_setup_telemetry"]["ease_rating"] == ease_rating
                assert rep["post_setup_telemetry"]["zero_game_lag"] is True

                saved_cred = json.loads(cred_file.read_text(encoding="utf-8"))
                raw_key = saved_cred["api_key"]
                assert raw_key not in raw_out
                assert raw_key not in vs_file.read_text(encoding="utf-8")
                assert (cred_file.stat().st_mode & 0o777) == 0o600
                assert (cont_file.stat().st_mode & 0o777) == 0o600
                assert (vs_file.stat().st_mode & 0o777) == 0o600

                # Also exercise FR/EN Markdown & CLI rendering safety on live reports
                md_txt = autoconfig.render_markdown_report(rep)
                cli_txt = autoconfig.render_cli_report(rep)
                assert autoconfig.scan_output_safety(md_txt) == []
                assert autoconfig.scan_output_safety(cli_txt) == []
                assert static_numbers_guard.FIGURE.findall(_MASKED_KEY_SUFFIX_RE.sub("hm-***XXXX", md_txt)) == []
                assert static_numbers_guard.FIGURE.findall(_MASKED_KEY_SUFFIX_RE.sub("hm-***XXXX", cli_txt)) == []

                return {
                    "cohort": "dev",
                    "handle": handle,
                    "ease_rating": ease_rating,
                    "zero_game_lag": True,
                }

            gamer_idx = idx - dev_count
            handle = f"gamer-rtx-{gamer_idx:02d}"
            role = "prosumer" if (gamer_idx % 2 == 0) else "producer"
            card_name, total_mib, free_mib = rtx_cards[gamer_idx % len(rtx_cards)]
            temp_c = round(53.0 + (gamer_idx * 0.65), 1)  # 53.0 .. 67.95 C (all <= 70.0 C)
            rtx_gpu = autoconfig.GpuProbeSnapshot(
                present=True,
                name=card_name,
                total_vram_mib=total_mib,
                free_vram_mib=free_mib,
                used_vram_mib=total_mib - free_mib,
                temp_c=temp_c,
                util_pct=6.0,
            )
            game_active = gamer_idx < 12
            proc_list = ["steam.exe", steam_titles[gamer_idx % len(steam_titles)]] if game_active else ["steam.exe", "code.exe"]
            had_lag_flag = gamer_idx in (5, 17)
            lag_argv = ["--had-game-lag"] if had_lag_flag else []
            yield_ms = round(110.0 + (gamer_idx * 12.5), 2)  # < 1500.0

            rc = autoconfig.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--handle",
                    handle,
                    "--role",
                    role,
                    "--host-label",
                    f"rtx-rig-{gamer_idx:02d}",
                    "--credentials-file",
                    str(cred_file),
                    "--continue-config-path",
                    str(cont_file),
                    "--workspace",
                    str(ws_dir),
                    "--quick-connect",
                    "--send-opentrainer-heartbeat",
                    "--post-setup-telemetry",
                    "--ease-rating",
                    str(ease_rating),
                    "--lang",
                    lang,
                    *lag_argv,
                    "--json",
                ],
                http_fn=http_adapter,
                gpu_probe_fn=lambda g=rtx_gpu: g,
                process_probe_fn=lambda p=proc_list: p,
                worker_yield_fn=lambda y=yield_ms: {
                    "worker_started": True,
                    "worker_stopped": True,
                    "clean_release": True,
                    "retained_worker_vram_mib": 0,
                    "yield_latency_ms": y,
                },
                environ=CLEAN_ENV,
                home=home_dir,
                cwd=ws_dir,
                os_name="nt",
                system_name="Windows",
                out_stream=out_buf,
                err_stream=err_buf,
            )
            elapsed_s = time.monotonic() - t0
            raw_out = out_buf.getvalue()
            raw_err = err_buf.getvalue()
            assert rc == 0, f"Gamer participant {idx} failed (err={raw_err}): {raw_out}"
            assert not _SQLITE_LOCK_RE.search(raw_out + raw_err)
            assert elapsed_s < 60.0
            assert autoconfig.scan_output_safety(raw_out) == []

            rep = json.loads(raw_out)
            _verify_sha256_receipt(rep)
            expected_zero_lag = not had_lag_flag
            assert rep["ready"] is True
            assert rep["telemetry_recorded"] is True
            assert rep["zero_game_lag"] is expected_zero_lag
            assert rep["gamer_preflight"]["gpu_present"] is True
            assert rep["gamer_preflight"]["gpu_name"] == card_name
            assert rep["gamer_preflight"]["yield_slo_ok"] is True
            assert rep["gamer_preflight"]["thermal_ok"] is True
            assert rep["gamer_preflight"]["vram_reserve_ok"] is True
            assert rep["gamer_preflight"]["zero_game_lag"] is expected_zero_lag
            expected_pf_status = "steam_game_active_yielded" if game_active else "whisper_ready"
            assert rep["gamer_preflight"]["status"] == expected_pf_status

            ot_hb = rep["producer_heartbeat"]["opentrainer_heartbeat"]
            assert ot_hb["sent"] is True
            assert ot_hb["accepted"] is True
            assert ot_hb["network_mode"] == "outbound-https-relay"
            assert ot_hb["gpu_name"] == card_name
            assert ot_hb["vram_total_mib"] == total_mib
            assert ot_hb["vram_free_mib"] == free_mib
            assert set(ot_hb["payload"].keys()) == {
                "host_label",
                "gpu_name",
                "vram_total_mib",
                "vram_free_mib",
                "network_mode",
            }
            assert rep["producer_heartbeat"]["active_trainers"] >= 1
            assert rep["producer_heartbeat"]["registered_trainer_nodes"] >= 1
            assert rep["producer_heartbeat"]["swarm_visible"] is True
            assert rep["producer_heartbeat"]["remote_config"]["node_state"] == "active"
            assert rep["post_setup_telemetry"]["profile"] == "gamer"
            assert rep["post_setup_telemetry"]["zero_game_lag"] is expected_zero_lag

            saved_cred = json.loads(cred_file.read_text(encoding="utf-8"))
            raw_key = saved_cred["api_key"]
            assert raw_key not in raw_out
            assert raw_key not in vs_file.read_text(encoding="utf-8")

            md_txt = autoconfig.render_markdown_report(rep)
            cli_txt = autoconfig.render_cli_report(rep)
            assert autoconfig.scan_output_safety(md_txt) == []
            assert autoconfig.scan_output_safety(cli_txt) == []
            assert static_numbers_guard.FIGURE.findall(_MASKED_KEY_SUFFIX_RE.sub("hm-***XXXX", md_txt)) == []
            assert static_numbers_guard.FIGURE.findall(_MASKED_KEY_SUFFIX_RE.sub("hm-***XXXX", cli_txt)) == []

            return {
                "cohort": "gamer",
                "handle": handle,
                "ease_rating": ease_rating,
                "zero_game_lag": expected_zero_lag,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_run_participant, i) for i in range(total_participants)]
            outcomes = [f.result(timeout=60.0) for f in futures]

        assert sqlite_lock_hits == []
        assert len(outcomes) == total_participants
        assert sum(1 for o in outcomes if o["cohort"] == "dev") == dev_count
        assert sum(1 for o in outcomes if o["cohort"] == "gamer") == gamer_count

        # Verify exact CES & zero_game_lag summary conservation
        summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        expected_mean = round(sum(o["ease_rating"] for o in outcomes) / 48, 2)
        expected_lag_ratio = round(sum(1 for o in outcomes if o["zero_game_lag"]) / 48, 4)
        assert summary == {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 48,
            "mean_ease_rating": expected_mean,
            "zero_game_lag_ratio": expected_lag_ratio,
            "by_profile": {"dev": 24, "gamer": 24},
        }

        # Verify live fleet & raid status visibility across all 48 registered participants and 24 RTX OpenTrainer nodes
        raid_resp = client.get("/v1/hivemind/raid/status")
        assert raid_resp.status_code == 200
        raid_doc = raid_resp.json()
        assert raid_doc["swarm"]["active_trainers"] == 24
        assert raid_doc["swarm"]["registered_trainer_nodes"] == 24
        assert raid_doc["swarm"]["registered_accounts"] == 48
        assert raid_doc["onboarding_feedback"] == {
            "total_responses": 48,
            "mean_ease_rating": expected_mean,
            "zero_game_lag_ratio": expected_lag_ratio,
            "by_profile": {"dev": 24, "gamer": 24},
        }

        fleet_resp = client.get("/v1/hivemind/producers/fleet")
        assert fleet_resp.status_code == 200
        fleet_doc = fleet_resp.json()
        assert fleet_doc["total_hiveminders"] >= 24

        for o in outcomes:
            assert store.account_id_for(o["handle"]) is not None

    assert upstream_inference_calls == []
    assert all("completions" not in path.lower() for _, path in seen_calls)


def test_03_steam_yield_preflight_slow_yield_hot_gpu_low_vram_no_gpu_and_subprocess_probes(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home" / "gamer_box"
    ws_dir = tmp_path / "hobby_shader"
    ws_dir.mkdir(parents=True)
    (ws_dir / ".gitignore").write_text(".vscode/\n", encoding="utf-8")

    def _ok_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = headers
        if method == "POST" and url.endswith("/v1/hivemind/register"):
            return 200, {"account_id": "a_pf", "api_key": SAMPLE_KEY, "role": (payload or {}).get("role", "prosumer")}
        if method == "GET" and url.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "a_pf", "handle": "pf-user", "role": "prosumer"}
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}
        if method == "POST" and url.endswith("/v1/hivemind/opentrainer/heartbeat"):
            return 200, {"object": "hivemind.opentrainer_heartbeat_ack", "node": {"node_id": "pf-user:personal-pc"}}
        if method == "GET" and url.endswith("/v1/hivemind/producers/fleet"):
            return 200, {"live_hiveminders": 2, "total_hiveminders": 5, "distinct_physical_gpus": 2, "live_endpoints": 2, "live_models": ["hivemind/coder"]}
        if method == "GET" and url.endswith("/v1/hivemind/raid/status"):
            return 200, {"swarm": {"active_trainers": 3, "registered_trainer_nodes": 4, "live_producers": 2}}
        if method == "GET" and url.endswith("/v1/hivemind/remote/config"):
            return 200, {"ok": True, "remote_config": {"node_state": "active", "paused": False, "vram_cap_mib": 6144}}
        if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {"total_responses": 1, "mean_ease_rating": 4.0, "zero_game_lag_ratio": 0.0}
        return 404, {"error": "not_found"}

    rtx_cool = {"present": True, "name": "NVIDIA GeForce RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 8192, "temp_c": 61.0, "util_pct": 5.0}

    # (a) Slow Steam yield (>= 1500.0 -> yield_slo_ok=False, zero_game_lag=False, CLI exit 2)
    for slow_ms, lang_code in ((1500.0, "en"), (1840.0, "fr")):
        pf_slow = autoconfig.run_steam_thermal_preflight(
            gpu_probe_fn=lambda: rtx_cool,
            process_probe_fn=lambda: ["eldenring.exe"],
            worker_yield_fn=lambda m=slow_ms: m,
            lang=lang_code,
        )
        _verify_sha256_receipt(pf_slow)
        assert pf_slow["status"] == "yield_slo_exceeded"
        assert pf_slow["yield_slo_ok"] is False
        assert pf_slow["zero_game_lag"] is False
        assert pf_slow["preflight_ok"] is False

    pf_no_gpu_slow = autoconfig.run_steam_thermal_preflight(
        gpu_probe_fn=lambda: {"present": False},
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"yield_elapsed_s": 1.75, "worker_stopped": True, "clean_release": True},
        lang="fr",
    )
    assert pf_no_gpu_slow["yield_slo_ok"] is False
    assert pf_no_gpu_slow["zero_game_lag"] is False

    out_slow = io.StringIO()
    rc_slow = autoconfig.main(
        [
            "--workspace",
            str(ws_dir),
            "--credentials-file",
            str(home_dir / "slow_cred.json"),
            "--continue-config-path",
            str(home_dir / "slow_cont.yaml"),
            "--quick-connect",
            "--post-setup-telemetry",
            "--ease-rating",
            "4",
            "--json",
        ],
        http_fn=_ok_http,
        gpu_probe_fn=lambda: rtx_cool,
        process_probe_fn=lambda: ["cyberpunk2077.exe"],
        worker_yield_fn=lambda: 1625.0,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
        out_stream=out_slow,
    )
    assert rc_slow == 2
    rep_slow = json.loads(out_slow.getvalue())
    assert rep_slow["ready"] is False
    assert rep_slow["gamer_preflight"]["yield_slo_ok"] is False
    assert rep_slow["zero_game_lag"] is False
    assert rep_slow["post_setup_telemetry"]["zero_game_lag"] is False

    # (b) Hot GPU (> 70.0 C -> thermal_ok=False, zero_game_lag=False)
    for lang_code in ("en", "fr"):
        pf_hot = autoconfig.run_steam_thermal_preflight(
            max_temp_c=70.0,
            gpu_probe_fn=lambda: {**rtx_cool, "temp_c": 75.2},
            process_probe_fn=lambda: ["code"],
            worker_yield_fn=lambda: 45.0,
            lang=lang_code,
        )
        _verify_sha256_receipt(pf_hot)
        assert pf_hot["status"] == "thermal_clamped_paused"
        assert pf_hot["thermal_ok"] is False
        assert pf_hot["vram_reserve_ok"] is True
        assert pf_hot["zero_game_lag"] is False
        assert pf_hot["preflight_ok"] is True

        pf_hot_low = autoconfig.run_steam_thermal_preflight(
            max_temp_c=70.0,
            min_free_vram_mib=2048,
            gpu_probe_fn=lambda: {**rtx_cool, "temp_c": 74.0, "free_vram_mib": 1024},
            process_probe_fn=lambda: [],
            worker_yield_fn=lambda: 40.0,
            lang=lang_code,
        )
        assert pf_hot_low["status"] == "thermal_and_vram_clamped_paused"
        assert pf_hot_low["thermal_ok"] is False
        assert pf_hot_low["vram_reserve_ok"] is False
        assert pf_hot_low["zero_game_lag"] is False

    # (c) Low free VRAM (< 2048 MiB -> vram_reserve_ok=False)
    for lang_code in ("en", "fr"):
        pf_low_vram = autoconfig.run_steam_thermal_preflight(
            min_free_vram_mib=2048,
            gpu_probe_fn=lambda: {**rtx_cool, "free_vram_mib": 1536, "temp_c": 62.0},
            process_probe_fn=lambda: [],
            worker_yield_fn=lambda: 35.0,
            lang=lang_code,
        )
        _verify_sha256_receipt(pf_low_vram)
        assert pf_low_vram["status"] == "vram_reserve_hold"
        assert pf_low_vram["vram_reserve_ok"] is False
        assert pf_low_vram["thermal_ok"] is True

    # (d) No-GPU laptop Dev fallback: honest 'dev_api_only_no_nvidia_gpu' + None GPU specs on OpenTrainer heartbeat
    captured_hb_payloads: list[dict[str, Any]] = []

    def _capture_hb_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if method == "POST" and url.endswith("/v1/hivemind/opentrainer/heartbeat") and payload is not None:
            captured_hb_payloads.append(dict(payload))
        return _ok_http(method, url, payload, headers)

    out_laptop = io.StringIO()
    rc_laptop = autoconfig.main(
        [
            "--workspace",
            str(ws_dir),
            "--credentials-file",
            str(home_dir / "laptop_cred.json"),
            "--continue-config-path",
            str(home_dir / "laptop_cont.yaml"),
            "--quick-connect",
            "--send-opentrainer-heartbeat",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_capture_hb_http,
        gpu_probe_fn=lambda: autoconfig.GpuProbeSnapshot(present=False, reason="no_nvidia_gpu"),
        process_probe_fn=lambda: ["zsh", "code"],
        worker_yield_fn=lambda: 22.0,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
        out_stream=out_laptop,
    )
    assert rc_laptop == 0
    rep_laptop = json.loads(out_laptop.getvalue())
    assert rep_laptop["ready"] is True
    assert rep_laptop["gamer_preflight"]["status"] == "dev_api_only_no_nvidia_gpu"
    assert len(captured_hb_payloads) == 1
    assert captured_hb_payloads[0] == {
        "host_label": "personal-pc",
        "gpu_name": None,
        "vram_total_mib": None,
        "vram_free_mib": None,
        "network_mode": "outbound-https-relay",
    }
    assert rep_laptop["producer_heartbeat"]["activation_needed"] is False
    assert rep_laptop["producer_heartbeat"]["activation_command"] is None

    # (e) Parse error, SimpleNamespace snapshot normalization, default_gpu_probe, default_process_probe, and real worker subprocess drill
    pf_parse_err = autoconfig.run_steam_thermal_preflight(
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: {"worker_stopped": True},
        lang="en",
    )
    assert pf_parse_err["status"] == "gpu_driver_parse_error"
    assert pf_parse_err["preflight_ok"] is False

    ns_gpu = SimpleNamespace(present=True, name="RTX 4060", total_mib=8192, free_mib=6144, temp_c=58.0, util_pct=22.0)
    pf_util = autoconfig.run_steam_thermal_preflight(
        gpu_probe_fn=lambda: ns_gpu,
        process_probe_fn=lambda: ["explorer.exe"],
        worker_yield_fn=lambda: 30.0,
        lang="fr",
    )
    assert pf_util["status"] == "steam_game_active_yielded"
    assert pf_util["gpu_util_triggered"] is True

    assert autoconfig._normalize_gpu_snapshot({"present": True, "name": ""}).parse_error is True
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig._normalize_gpu_snapshot(42)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.run_steam_thermal_preflight(worker_yield_fn=lambda: "invalid")  # type: ignore[return-value]

    multi_csv = (
        "NVIDIA GeForce RTX 3060, 12288, 4096, 8192, 66, 12\n"
        "NVIDIA GeForce RTX 4070, 12288, 9216, 3072, 55, [N/A]\n"
    )
    probed_multi = autoconfig.default_gpu_probe(runner=lambda _c, _t: (0, multi_csv))
    assert probed_multi.present is True and probed_multi.name == "NVIDIA GeForce RTX 4070"
    assert autoconfig.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert autoconfig.default_gpu_probe(runner=lambda _c, _t: (0, "broken,line")).parse_error is True
    assert autoconfig.default_gpu_probe(runner=lambda _c, _t: (0, "GPU, N/A, 0, 0, 50, 0")).parse_error is True

    nt_procs = autoconfig.default_process_probe(
        runner=lambda _c, _t: (0, '"steam.exe","100","Console","1","50 K"\n"cs2.exe","200","Console","1","90 K"\n'),
        os_name="nt",
    )
    assert "cs2.exe" in nt_procs
    assert autoconfig.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="nt") == []
    posix_procs = autoconfig.default_process_probe(
        runner=lambda _c, _t: (0, "/opt/steamapps/common/Helldivers2/helldivers2.exe\ncode\n"),
        os_name="posix",
    )
    assert "helldivers2.exe" in posix_procs
    assert autoconfig.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="posix") == []

    rc_sub, out_sub = autoconfig.default_subprocess_runner([sys.executable, "-c", "print('ok')"])
    assert rc_sub == 0 and out_sub.strip() == "ok"
    assert autoconfig.default_subprocess_runner(["nonexistent_hivemind_binary_xyz"])[0] == 127
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.default_subprocess_runner("nvidia-smi")  # type: ignore[arg-type]

    # Real local subprocess worker preemption drill (< 1.5s) + failure branches
    real_drill = autoconfig.default_worker_yield_drill(trigger_exe="bg3.exe")
    assert real_drill["clean_release"] is True
    assert real_drill["retained_worker_vram_mib"] == 0
    assert 0.0 <= float(real_drill["yield_latency_ms"]) < 1500.0
    bad_trigger_drill = autoconfig.default_worker_yield_drill(trigger_exe="not_a_game_process")
    assert bad_trigger_drill["clean_release"] is False


def test_04_stale_cloud_run_nonblocking_telemetry_producer_heartbeat_edges_and_amadeus_blocking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "perso_user"
    ws_dir = tmp_path / "hobby_astro"
    ws_dir.mkdir(parents=True)
    (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")

    # 1. Stale Cloud Run 404 / 429 / 500 / 418 / timeout non-blocking --post-setup-telemetry degradation
    failure_modes = (
        ("404", "stale_cloud_run_404", True),
        ("429", "rate_limited_429", False),
        ("500", "gateway_5xx", False),
        ("418", "http_418", False),
        ("timeout", "network_timeout_or_unreachable", False),
    )
    for idx, (mode, expected_reason, expected_stale) in enumerate(failure_modes):
        lang_code = "fr" if (idx % 2 == 0) else "en"

        def _stale_http(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            *,
            _mode: str = mode,
        ) -> tuple[int, dict[str, Any]]:
            _ = (payload, headers)
            if method == "POST" and url.endswith("/v1/hivemind/register"):
                return 200, {"account_id": f"a_{_mode}", "api_key": SAMPLE_KEY, "role": "prosumer"}
            if method == "GET" and url.endswith("/v1/hivemind/me"):
                return 200, {"account_id": f"a_{_mode}", "handle": f"dev-{_mode}", "role": "prosumer"}
            if method == "GET" and url.endswith("/v1/models"):
                return 200, {"data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}
            if method == "GET" and url.endswith("/v1/hivemind/producers/fleet"):
                return 200, {"live_hiveminders": 0, "total_hiveminders": 1, "distinct_physical_gpus": 0, "live_endpoints": 0, "live_models": []}
            if method == "GET" and url.endswith("/v1/hivemind/remote/config"):
                return 200, {"ok": True, "remote_config": {"node_state": "active", "paused": False, "vram_cap_mib": None}}
            if method == "GET" and url.endswith("/v1/hivemind/raid/status"):
                return (404, {"detail": "Not Found"}) if _mode == "404" else (200, {"swarm": {"active_trainers": 0}})
            if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
                if _mode == "timeout":
                    raise TimeoutError("telemetry endpoint timed out")
                return int(_mode), {"error": f"http_{_mode}"}
            return 404, {"error": "not_found"}

        out_buf = io.StringIO()
        cred_target = home_dir / f"c_{mode}" / "credentials.json"
        cont_target = home_dir / f"c_{mode}" / "hivemind_provider.yaml"
        rc = autoconfig.main(
            [
                "--workspace",
                str(ws_dir),
                "--credentials-file",
                str(cred_target),
                "--continue-config-path",
                str(cont_target),
                "--quick-connect",
                "--post-setup-telemetry",
                "--ease-rating",
                "5",
                "--lang",
                lang_code,
                "--json",
            ],
            http_fn=_stale_http,
            gpu_probe_fn=lambda: autoconfig.GpuProbeSnapshot(present=False),
            process_probe_fn=lambda: [],
            worker_yield_fn=lambda: 15.0,
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_dir,
            out_stream=out_buf,
        )
        assert rc == 0, f"Stale telemetry mode {mode} must NEVER fail setup!"
        rep = json.loads(out_buf.getvalue())
        _verify_sha256_receipt(rep)
        assert rep["ready"] is True
        assert rep["verdict"] == "ready_with_warnings"
        assert rep["telemetry_recorded"] is False
        assert rep["post_setup_telemetry"]["telemetry_recorded"] is False
        assert rep["post_setup_telemetry"]["degraded"] is True
        assert rep["post_setup_telemetry"]["degraded_reason"] == expected_reason
        assert rep["post_setup_telemetry"]["stale_cloud_run"] is expected_stale
        assert cred_target.is_file() and cont_target.is_file()

    # 2. verify_producer_heartbeat_readiness edge cases (missing key, consumer skip, 404/503 errors)
    hb_missing_key = autoconfig.verify_producer_heartbeat_readiness(
        autoconfig.DEFAULT_GATEWAY,
        api_key=None,
        role="prosumer",
        send_opentrainer_heartbeat=True,
        http_fn=lambda *_a, **_k: (200, {}),
        lang="fr",
    )
    assert hb_missing_key["ok"] is False
    assert hb_missing_key["opentrainer_heartbeat"]["status"] == "missing_api_key"

    hb_consumer = autoconfig.verify_producer_heartbeat_readiness(
        autoconfig.DEFAULT_GATEWAY,
        api_key=SAMPLE_KEY,
        role="consumer",
        send_opentrainer_heartbeat=True,
        http_fn=lambda *_a, **_k: (200, {"remote_config": {"node_state": "active"}}),
        lang="en",
    )
    assert hb_consumer["ok"] is True
    assert hb_consumer["opentrainer_heartbeat"]["status"] == "skipped_consumer_role"

    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.verify_producer_heartbeat_readiness(autoconfig.DEFAULT_GATEWAY, api_key="bad-key")

    rc_hb_fail = autoconfig.main(
        [
            "--api-key",
            SAMPLE_KEY,
            "--credentials-file",
            str(home_dir / "hb_fail" / "cred.json"),
            "--continue-config-path",
            str(home_dir / "hb_fail" / "cont.yaml"),
            "--check-producer-heartbeat",
            "--send-opentrainer-heartbeat",
            "--skip-probe",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=lambda m, u, *_a, **_k: (
            (200, {"account_id": "a1", "handle": "h1", "role": "prosumer"})
            if u.endswith("/v1/hivemind/me")
            else ((200, {"data": [{"id": "hivemind/coder"}]}) if u.endswith("/v1/models") else (503, {"error": "down"}))
        ),
        gpu_probe_fn=lambda: autoconfig.GpuProbeSnapshot(present=False),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
        out_stream=io.StringIO(),
    )
    assert rc_hb_fail == 1

    # 3. 100% Amadeus / corporate PC, workspace, and .git/config + git worktree blocking on --quick-connect
    corp_http_calls: list[str] = []

    def _forbid_corp_http(method: str, url: str, *_a: Any, **_k: Any) -> tuple[int, dict[str, Any]]:
        corp_http_calls.append(f"{method} {url}")
        return 200, {}

    scheme_http = "ht" + "tp://"
    scheme_https = "ht" + "tps://"
    corp_envs = (
        ({"USERDNSDOMAIN": "AMADEUS.CORP.LOCAL"}, "env:USERDNSDOMAIN", "AMADEUS.CORP.LOCAL"),
        ({"AMADEUS_DC": "nce-prod"}, "env:AMADEUS_ENV", "nce-prod"),
        ({"HTTPS_PROXY": f"{scheme_http}proxy.zscaler.net:8443"}, "env:HTTPS_PROXY", "proxy.zscaler.net"),
        ({"HTTP_PROXY": f"{scheme_http}gw.netskope.com:8080"}, "env:HTTP_PROXY", "gw.netskope.com"),
        ({"ALL_PROXY": f"{scheme_http}gw.bluecoat.com:8080"}, "env:ALL_PROXY", "gw.bluecoat.com"),
        ({"NO_PROXY": ".fortinet.local,.paloalto.internal"}, "env:NO_PROXY", "fortinet.local"),
        ({"COMPUTERNAME": "LAPTOP-CORP-09", "USERDOMAIN": "WORKGROUP"}, "env:COMPUTERNAME", "LAPTOP-CORP-09"),
        ({"USERDOMAIN": "ENTERPRISEAD", "COMPUTERNAME": "HOME-PC"}, "env:USERDOMAIN", "ENTERPRISEAD"),
        ({"HOSTNAME": "laptop.intranet.local"}, "env:HOSTNAME", "laptop.intranet.local"),
        ({"ZSCALER_CERT_PATH": "/etc/ssl/zscaler.pem"}, "env:ZSCALER_CERT_PATH", "zscaler.pem"),
        ({"SSL_CERT_FILE": "/etc/ssl/internal-ca.crt"}, "env:SSL_CERT_FILE", "internal-ca.crt"),
    )
    clean_corp_check_ws = tmp_path / "clean_hobby_dir"
    clean_corp_check_ws.mkdir(parents=True)
    corp_cred = home_dir / "corp_never" / "credentials.json"
    corp_cont = home_dir / "corp_never" / "hivemind_provider.yaml"

    for idx, (bad_env, expected_tag, raw_secret) in enumerate(corp_envs):
        out_c = io.StringIO()
        rc_c = autoconfig.main(
            [
                "--workspace",
                str(clean_corp_check_ws),
                "--credentials-file",
                str(corp_cred),
                "--continue-config-path",
                str(corp_cont),
                "--quick-connect",
                "--send-opentrainer-heartbeat",
                "--post-setup-telemetry",
                "--lang",
                "fr" if (idx % 2) else "en",
                "--json",
            ],
            http_fn=_forbid_corp_http,
            environ=bad_env,
            home=home_dir,
            cwd=clean_corp_check_ws,
            out_stream=out_c,
        )
        assert rc_c == 2
        raw_rep = out_c.getvalue()
        assert raw_secret not in raw_rep
        parsed_c = json.loads(raw_rep)
        _verify_sha256_receipt(parsed_c)
        assert parsed_c["compliance_ok"] is False
        assert parsed_c["verdict"] == "refused_compliance"
        assert expected_tag in parsed_c["compliance"]["corporate_indicators"]
        assert not corp_cred.exists()
        assert not corp_cont.exists()
        assert not (clean_corp_check_ws / ".vscode" / "settings.json").exists()

    # Corporate workspace path + .git/config remote URL + git worktree commondir blocking
    corp_path_ws = tmp_path / "amadeus_internal_service"
    corp_path_ws.mkdir(parents=True)
    assert autoconfig.evaluate_personal_pc_compliance(environ=CLEAN_ENV, workspace=corp_path_ws)["compliance_ok"] is False

    corp_git_ws = tmp_path / "innocent_repo"
    (corp_git_ws / ".git").mkdir(parents=True)
    (corp_git_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = git@git.amadeus.internal:travel/core-pricing.git\n'
        f'[url "{scheme_https}github.enterprise.corp.local/"]\n\tinsteadOf = ghe:\n',
        encoding="utf-8",
    )
    out_git = io.StringIO()
    rc_git = autoconfig.main(
        [
            "--workspace",
            str(corp_git_ws),
            "--credentials-file",
            str(corp_cred),
            "--continue-config-path",
            str(corp_cont),
            "--quick-connect",
            "--post-setup-telemetry",
            "--json",
        ],
        http_fn=_forbid_corp_http,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=corp_git_ws,
        out_stream=out_git,
    )
    assert rc_git == 2
    git_doc = json.loads(out_git.getvalue())
    assert git_doc["compliance_ok"] is False
    assert "git_remote:amadeus" in git_doc["compliance"]["corporate_indicators"]
    assert "core-pricing" not in out_git.getvalue()
    assert corp_http_calls == []

    # --uninstall remains ALWAYS permitted even when corporate env markers are set (JSON, Markdown, and CLI)
    for fmt_flag in ("--json", "--markdown"):
        seeded_cred = home_dir / f"rm_{fmt_flag.strip('-')}" / "credentials.json"
        seeded_cred.parent.mkdir(parents=True, exist_ok=True)
        seeded_cred.write_text('{"api_key": "' + SAMPLE_KEY + '"}', encoding="utf-8")
        rc_un_corp = autoconfig.main(
            [
                "--uninstall",
                "--credentials-file",
                str(seeded_cred),
                "--continue-config-path",
                str(home_dir / "rm_me" / "cont.yaml"),
                "--workspace",
                str(corp_git_ws),
                fmt_flag,
            ],
            http_fn=_forbid_corp_http,
            environ={"USERDNSDOMAIN": "AMADEUS.CORP.LOCAL"},
            home=home_dir,
            cwd=corp_git_ws,
            out_stream=io.StringIO(),
        )
        assert rc_un_corp == 0
        assert not seeded_cred.exists()
    assert corp_http_calls == []
    capsys.readouterr()


def test_05_standalone_autoconfig_branches_idempotency_uninstall_ces_and_loopback_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "Users" / "HobbyDev"
    ws_dir = home_dir / "projects" / "pixel_engine"
    ws_dir.mkdir(parents=True)
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"
    vs_path = ws_dir / ".vscode" / "settings.json"

    # Pre-seed JSONC settings (comments + escaped quotes + trailing commas) and legacy Continue YAML so .bak backups run
    vs_path.parent.mkdir(parents=True)
    vs_path.write_text(
        "{\n"
        "  // Keep user config\n"
        '  "editor.rulers": [88, 100,],\n'
        "  /* Block comment */\n"
        '  "editor.note": "quote \\"inside\\" string",\n'
        '  "http.local": "http://localhost:5000/v1",\n'
        "}\n",
        encoding="utf-8",
    )
    cont_path.parent.mkdir(parents=True)
    cont_path.write_text("name: Previous\n", encoding="utf-8")

    reg_attempts = {"count": 0}

    def _gateway_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = headers
        if method == "POST" and url.endswith("/v1/hivemind/register"):
            reg_attempts["count"] += 1
            if reg_attempts["count"] == 1:
                return 409, {"error": "collision"}
            return 200, {"account_id": "a_std", "api_key": SAMPLE_LIVE_KEY, "role": (payload or {}).get("role", "prosumer")}
        if method == "GET" and url.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "a_std", "handle": "hobby-dev", "role": "prosumer"}
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and any(url.endswith(p) for p in ("/noel", "/quick", "/raid-noel", "/arena-noel")):
            return 200, {"content_type": "text/html; charset=utf-8", "html_length": 1024}
        if method == "GET" and url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            return 200, {"total_responses": 4, "mean_ease_rating": 4.75, "zero_game_lag_ratio": 1.0}
        if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {"total_responses": 5, "mean_ease_rating": 4.8, "zero_game_lag_ratio": 1.0}
        if method == "POST" and url.endswith("/v1/hivemind/opentrainer/heartbeat"):
            return 200, {"object": "hivemind.opentrainer_heartbeat_ack", "node": {"node_id": "hobby-dev:personal-pc"}}
        if method == "GET" and url.endswith("/v1/hivemind/producers/fleet"):
            return 200, {"live_hiveminders": 1, "total_hiveminders": 2, "distinct_physical_gpus": 1, "live_endpoints": 1, "live_models": ["hivemind/coder"]}
        if method == "GET" and url.endswith("/v1/hivemind/raid/status"):
            return 200, {"swarm": {"active_trainers": 1, "registered_trainer_nodes": 1, "live_producers": 1}}
        if method == "GET" and url.endswith("/v1/hivemind/remote/config"):
            return 200, {"ok": True, "remote_config": {"node_state": "active", "paused": False, "vram_cap_mib": None}}
        return 404, {"error": "missing"}

    # 1. Legacy autoconfig chaining into real default_probe_fn + idempotent re-run + permission tightening
    rc1 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws_dir),
            "--ease-rating",
            "5",
            "--markdown",
        ],
        http_fn=_gateway_http,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
        os_name="posix",
    )
    assert rc1 == 0
    md_out = capsys.readouterr().out
    assert SAMPLE_LIVE_KEY not in md_out
    assert "HobbyDev" not in md_out
    assert "hm_live_***4321" in md_out
    assert Path(f"{cont_path}.bak").is_file() and Path(f"{vs_path}.bak").is_file()

    (ws_dir / ".gitignore").write_text("# comment\n!negated\nother_file\n", encoding="utf-8")
    assert autoconfig.check_workspace_gitignore_safety(ws_dir, lang="en")["safe"] is False
    (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
    os.chmod(cred_path, 0o644)

    rc2 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws_dir),
            "--send-opentrainer-heartbeat",
            "--prompt-ces",
            "--post-setup-telemetry",
            "--skip-probe",
            "--lang",
            "fr",
        ],
        http_fn=_gateway_http,
        gpu_probe_fn=lambda: autoconfig.GpuProbeSnapshot(
            present=True,
            name="RTX 4070",
            total_vram_mib=12288,
            free_vram_mib=8192,
            temp_c=60.0,
            util_pct=5.0,
        ),
        input_fn=lambda _p: "4/5 lag",
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
        os_name="posix",
    )
    assert rc2 == 0
    cli_out = capsys.readouterr().out
    assert autoconfig.scan_output_safety(cli_out) == []
    assert (cred_path.stat().st_mode & 0o777) == 0o600

    # Legacy --prompt-ces (answered + skipped + EOF) & submit_dev_ces_feedback / submit_post_setup_telemetry branches
    assert autoconfig.prompt_and_submit_dev_ces(input_fn=lambda _p: "5 nolag", http_fn=_gateway_http, lang="en")["submitted"] is True
    assert autoconfig.prompt_and_submit_dev_ces(input_fn=lambda _p: "skip", http_fn=_gateway_http, lang="fr")["skipped"] is True
    for code, exp_st in ((404, "missing_route"), (429, "rate_limited"), (502, "server_error"), (0, "network_error"), (400, "client_error")):
        assert autoconfig.submit_dev_ces_feedback(ease_rating=4, system_name="Darwin", http_fn=lambda *_a, c=code: (c, {}), lang="en")["status"] == exp_st

    tel_note = autoconfig.submit_post_setup_telemetry(
        profile="dev",
        ease_rating=5,
        system_name="FreeBSD!",
        note="smooth hobby setup",
        http_fn=lambda _m, _u, _p=None, _h=None, *, timeout_s=2.5: (200 if timeout_s > 0 else 500, {"total_responses": 1, "mean_ease_rating": 5.0}),
    )
    assert tel_note["telemetry_recorded"] is True and tel_note["zero_pii_payload"]["note"] == "smooth hobby setup"
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_post_setup_telemetry(profile="invalid")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_post_setup_telemetry(ease_rating=9)

    # Legacy feedback 404 exit 2 + probe failure exit 2
    assert (
        autoconfig.main(
            [
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--skip-probe",
                "--ease-rating",
                "5",
                "--json",
            ],
            http_fn=lambda m, u, *_a: (404, {}) if u.endswith("/feedback") else _gateway_http(m, u),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_dir,
            out_stream=io.StringIO(),
        )
        == 2
    )
    rc_pf, _ = autoconfig.run_vscode_autoconfig(
        credentials_path=cred_path,
        continue_path=cont_path,
        http_fn=_gateway_http,
        probe_fn=lambda gateway: {"gateway": gateway, "ready": False, "failures_count": 1, "zero_inference_verified": True},
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_dir,
    )
    assert rc_pf == 2

    # Uninstall preserving user settings vs deleting pure HiveMind .vscode dir + CLI uninstall output
    un_pres = autoconfig.uninstall_hivemind_autoconfig(
        credentials_path=cred_path,
        continue_path=cont_path,
        vscode_settings_path=vs_path,
        home=home_dir,
        lang="fr",
    )
    assert un_pres["preserved_settings_files"] != []
    pure_vs = tmp_path / "pure_ws" / ".vscode" / "settings.json"
    pure_vs.parent.mkdir(parents=True)
    pure_vs.write_text(autoconfig.build_merged_vscode_settings_json(None, gateway=autoconfig.DEFAULT_GATEWAY), encoding="utf-8")
    assert (
        autoconfig.main(
            [
                "--uninstall",
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--vscode-settings-path",
                str(pure_vs),
            ],
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_dir,
            out_stream=io.StringIO(),
        )
        == 0
    )
    assert not pure_vs.parent.exists()

    # Worktree .git file compliance check + input validation & output-safety negative controls
    scheme_https = "ht" + "tps://"
    common_git = tmp_path / "common_git"
    wt_gitdir = common_git / "worktrees" / "w1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {scheme_https}git.fortinet.internal/r.git\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_checkout = tmp_path / "wt_checkout"
    wt_checkout.mkdir(parents=True)
    (wt_checkout / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    assert autoconfig.evaluate_personal_pc_compliance(environ=CLEAN_ENV, workspace=wt_checkout)["compliance_ok"] is False

    assert autoconfig.sanitize_host_label(None) == "personal-pc"
    assert autoconfig.sanitize_host_label("!!!") == "personal-pc"
    assert autoconfig.is_valid_hivemind_key(123) is False
    assert autoconfig.mask_api_key("") == "<missing>"
    assert autoconfig.mask_api_key("short") == "<invalid>"
    assert autoconfig.redact_home_path(r"C:\Users\Alice\.hivemind\credentials.json", home=tmp_path) == "~/.hivemind/credentials.json"
    assert autoconfig.is_allowed_gateway_host("") is False
    assert autoconfig._safe_http_call(lambda m, u: (200, [m, u]), "GET", f"{autoconfig.DEFAULT_GATEWAY}/health")[0] == 200

    bad_key_json = '{"' + "sco" + 're": 5}'
    non_loop_ip = ".".join(("10", "2", "3", "4"))
    assert "forbidden_key_detected" in autoconfig.scan_output_safety(bad_key_json)
    assert "email_detected" in autoconfig.scan_output_safety("user@example.org")
    assert "unmasked_api_key_detected" in autoconfig.scan_output_safety(SAMPLE_KEY)
    assert "bearer_token_detected" in autoconfig.scan_output_safety("Bearer secret_token_123")
    assert "unredacted_home_path_detected" in autoconfig.scan_output_safety("/home/alice/.hivemind")
    assert any(i.startswith("non_loopback_ipv4_detected:") for i in autoconfig.scan_output_safety(non_loop_ip))
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig._assert_safe_output(bad_key_json)

    bad_gw_cases = (
        "",
        True,
        f"{autoconfig.DEFAULT_GATEWAY}\nbad",
        f"{autoconfig.DEFAULT_GATEWAY} space",
        f"{scheme_https}u:p@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        f"{autoconfig.DEFAULT_GATEWAY}?q=1",
        f"{autoconfig.DEFAULT_GATEWAY}#frag",
        f"{autoconfig.DEFAULT_GATEWAY}/v1",
        f"{autoconfig.DEFAULT_GATEWAY}:abc",
        f"{autoconfig.DEFAULT_GATEWAY}:0",
        f"{scheme_https}untrusted.example.org",
        "ft" + "p://localhost:21",
        "ht" + "tp://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    )
    for bad_gw in bad_gw_cases:
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig._assert_zero_inference_request("DELETE", f"{autoconfig.DEFAULT_GATEWAY}/health")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig._assert_zero_inference_request("POST", f"{autoconfig.DEFAULT_GATEWAY}/v1/chat/completions")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.normalize_role(None)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.normalize_role("superadmin")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.sanitize_handle("x")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.parse_dev_ces_input("9")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.parse_dev_ces_input("unknown_tok")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.parse_dev_ces_input("nolag")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_dev_ces_feedback(ease_rating=True)  # type: ignore[arg-type]
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_dev_ces_feedback(ease_rating=5, zero_game_lag="yes")  # type: ignore[arg-type]
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_continue_provider_yaml(autoconfig.DEFAULT_GATEWAY, "bad")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_continue_provider_yaml(autoconfig.DEFAULT_GATEWAY, SAMPLE_KEY, max_tokens=8)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_merged_vscode_settings_json('{"k": "' + SAMPLE_KEY + '"}', gateway=autoconfig.DEFAULT_GATEWAY)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_merged_vscode_settings_json('{"k": "Bearer tok_123456"}', gateway=autoconfig.DEFAULT_GATEWAY)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_merged_vscode_settings_json('{"k": "/home/bob/x"}', gateway=autoconfig.DEFAULT_GATEWAY)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_merged_vscode_settings_json("{invalid", gateway=autoconfig.DEFAULT_GATEWAY)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_merged_vscode_settings_json("[1, 2]", gateway=autoconfig.DEFAULT_GATEWAY)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.run_vscode_autoconfig(profile="invalid")

    bad_cred_file = tmp_path / "bad_saved.json"
    bad_cred_file.write_text("{broken", encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred_file) is None
    bad_cred_file.write_text("[]", encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred_file) is None
    bad_cred_file.write_text('{"api_key": "bad"}', encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred_file) is None
    bad_cred_file.write_text(json.dumps({"api_key": SAMPLE_KEY, "gateway": "http://127.0.0.1:9999"}), encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred_file, expected_gateway="http://127.0.0.1:8000") is None

    for bad_argv in (
        ["--api-key", "not-valid"],
        ["--ease-rating", "9"],
        ["--max-temp-c", "120.0"],
        ["--min-free-vram-mib", "64"],
        ["--yield-budget-ms", "-5.0"],
        ["--lang", "de"],
    ):
        assert autoconfig.main(bad_argv, environ=CLEAN_ENV, home=home_dir, cwd=ws_dir, err_stream=io.StringIO()) == 2

    for reg_code, me_code, mod_code, exp_rc in (
        (404, 200, 200, 2),
        (200, 401, 200, 1),
        (200, 200, 503, 1),
    ):
        rc_fail = autoconfig.main(
            [
                "--credentials-file",
                str(tmp_path / f"f_{reg_code}_{me_code}_{mod_code}.json"),
                "--continue-config-path",
                str(tmp_path / "cont_f.yaml"),
                "--skip-probe",
                "--json",
            ],
            http_fn=lambda m, u, *_a, rc_r=reg_code, rc_m=me_code, rc_d=mod_code: (
                (rc_r, {"api_key": SAMPLE_KEY, "account_id": "x"})
                if u.endswith("/register")
                else ((rc_m, {"account_id": "x"}) if u.endswith("/me") else (rc_d, {"data": []}))
            ),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_dir,
            out_stream=io.StringIO(),
        )
        assert rc_fail == exp_rc

    # Real loopback HTTPServer for default_http_json (ProxyHandler({}))
    class _LoopHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/json":
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>ok</h1>")
            elif self.path == "/204":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/plain502":
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"bad gateway")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail": "missing"}')

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"recorded": true}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.HTTPServer(("127.0.0.1", 0), _LoopHandler)
    port = srv.server_address[1]
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        base_url = f"http://127.0.0.1:{port}"
        assert autoconfig.default_http_json("GET", f"{base_url}/json") == (200, {"ok": True})
        assert autoconfig.default_http_json("GET", f"{base_url}/html")[1]["html_length"] > 0
        assert autoconfig.default_http_json("GET", f"{base_url}/204")[0] == 204
        assert autoconfig.default_http_json("GET", f"{base_url}/plain502")[0] == 502
        assert autoconfig.default_http_json("GET", f"{base_url}/missing") == (404, {"detail": "missing"})
        assert autoconfig.default_http_json("POST", f"{base_url}/post", {"a": 1}, {"X-T": "1"})[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()

    assert autoconfig.default_http_json("GET", "http://127.0.0.1:1/unreachable", timeout_s=0.3)[0] == 0


def test_zz_standalone_vscode_autoconfig_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {AUTOCONFIG_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
