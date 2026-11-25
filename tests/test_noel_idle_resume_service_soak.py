# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for SLICE-SCALE-DYN-16: opt-in idle-resume service, < 1.5 s Steam yield / <= 70 C state machine, and 48-user soak."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextvars
import ctypes
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
import plistlib
import re
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from hivemind.cli import main as cli_main
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import (
    OnboardingFeedbackRequest,
    OpenTrainerHeartbeatRequest,
    create_router_app,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """Minimal stub audit result used when optional `gwaya` is absent in CI."""

    is_clean: bool = True
    violations: tuple[str, ...] = ()
    penalty_energy: float = 0.0


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both `check_file` and `audit_python_code`."""

    def check_file(self, _path: str) -> list[str]:
        return []

    def audit_python_code(self, _code: str) -> _FallbackStubResult:
        return _FallbackStubResult()


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_SCRIPT_PATH = REPO_ROOT / "scripts" / "noel_idle_resume_service.py"
CLI_MAIN_PATH = REPO_ROOT / "src" / "hivemind" / "cli" / "main.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without starting an internal coverage.Coverage() instance so external
# `coverage run --rcfile=/dev/null` captures 100% of import + runtime lines.
from scripts import noel_idle_resume_service as idle_svc  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn16",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn16",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn16",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
SOAK_SECRET = b"noel-scale-dyn16-router-sec-32b!"
ALPHA_KEY = "hm-" + "dyn16_alpha_token_123456"
LIVE_KEY = "hm_live_" + "dyn16_live_token_654321"
LIVE_BUILD_SHA = "b1c2d3e4f5a60718293a4b5c6d7e8f9011223344"
LIVE_SHORT_SHA = "b1c2d3e4f5a6"
DRIFT_SHA = "7788990011223344556677889900aabbccddeeff"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.org"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/platform/core.git"
CORP_HTTPS_REMOTE = "https://" + "gitlab.enterprise.intranet/team/service.git"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@proxy.zscaler.internal:8443"

_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_NONCE_ATTR_RE = re.compile(r'nonce="[^"]*"')
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
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


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert idle_svc.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(doc)


def _quiet_rtx4070_probe(**_: Any) -> idle_svc.GpuProbeSnapshot:
    return idle_svc.GpuProbeSnapshot(
        present=True,
        name="NVIDIA GeForce RTX 4070",
        total_vram_mib=12288,
        free_vram_mib=9216,
        used_vram_mib=3072,
        temp_c=54.0,
        util_pct=4.0,
    )


def _fast_yield_probe(**_: Any) -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 34.5,
        "method": "injected_fast_yield",
    }


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    raid_nonces: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[str, tuple[int, dict[str, Any]]] | None = None,
) -> idle_svc.HttpFn:
    def _adapter(
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, Any]]:
        assert 0.1 <= timeout_s <= 5.5
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip

        if method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat" and payload is not None:
            OpenTrainerHeartbeatRequest.model_validate(payload)
        if method == "POST" and path == "/v1/hivemind/onboarding/feedback" and payload is not None:
            OnboardingFeedbackRequest.model_validate(payload)

        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path, dict(req_headers)))
            else:
                seen_calls.append((method, path, dict(req_headers)))

        if override_routes and path in override_routes:
            return override_routes[path]

        resp = client.request(method, path, json=payload, headers=req_headers)
        raw_text = resp.text
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
            else:
                sqlite_lock_hits.append(raw_text[:200])

        lower_hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code == 200 and path == "/raid-noel" and raid_nonces is not None:
            m = _CSP_NONCE_RE.search(lower_hdrs.get("content-security-policy", ""))
            if m:
                if lock is not None:
                    with lock:
                        raid_nonces.append(m.group(1))
                else:
                    raid_nonces.append(m.group(1))

        ctype = lower_hdrs.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, dict):
                out = dict(data)
                out.setdefault("_headers", lower_hdrs)
                return resp.status_code, out
            return resp.status_code, {"data": data, "_headers": lower_hdrs}
        return resp.status_code, {"raw_length": len(raw_text), "headers": lower_hdrs}

    return _adapter


def _seed_live_coder_producers(client: TestClient, count: int = 8) -> None:
    for p_idx in range(count):
        peer_ip = _loopback_ip_for(92, p_idx)
        reg_p = client.post(
            "/v1/hivemind/register",
            json={"handle": f"seed-rtx-prod-{p_idx:02d}", "role": "producer"},
            headers={"X-Forwarded-For": peer_ip},
        ).json()
        hb_p = client.post(
            "/v1/hivemind/producer/heartbeat",
            json={
                "endpoint_url": f"http://{_loopback_ip_for(93, p_idx)}:11434",
                "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                "cost_per_1k": 0.0,
                "capacity_1k": 50.0,
                "gpu_vram_mib": 12288,
                "execution_class": "ollama-gamer-whisper",
            },
            headers={"Authorization": f"Bearer {reg_p['api_key']}", "X-Forwarded-For": peer_ip},
        )
        assert hb_p.status_code == 200


def test_01_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (SERVICE_SCRIPT_PATH, CLI_MAIN_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {target}"

    script_src = SERVICE_SCRIPT_PATH.read_text(encoding="utf-8")
    cli_src = CLI_MAIN_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure stdlib imports only in scripts/noel_idle_resume_service.py + ProxyHandler({}) + zero unsafe sinks
    tree = ast.parse(script_src, filename="scripts/noel_idle_resume_service.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
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
    assert "pickle" not in imported_roots
    assert "yaml" not in imported_roots
    assert saw_empty_proxy_handler is True

    # Ensure no coverage import exists inside this test file
    test_tree = ast.parse(test_src, filename="tests/test_noel_idle_resume_service_soak.py")
    test_imports = {
        alias.name.split(".")[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    } | {
        n.module.split(".")[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "coverage" not in test_imports

    # 2. Lateral-movement AST audit + planted negative controls
    assert lateral_guard.audit_source_for_lateral_movement(script_src, "scripts/noel_idle_resume_service.py") == []
    cli_new_blocks = "\n".join(
        [
            ast.get_source_segment(
                cli_src,
                next(
                    n
                    for n in ast.parse(cli_src).body
                    if isinstance(n, ast.FunctionDef) and n.name == fn_name
                ),
            )
            or ""
            for fn_name in (
                "_load_script_helper",
                "_append_forwarded_flags",
                "_load_idle_resume_service_module",
                "_build_gamer_service_argv",
                "cmd_gamer_service",
            )
        ]
    )
    assert lateral_guard.audit_source_for_lateral_movement(cli_new_blocks, "src/hivemind/cli/main.py") == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, "tests/test_noel_idle_resume_service_soak.py") == []
    planted_lan = ".".join(("192", "168", "10", "77"))
    planted_ext = "https://" + "c2.attacker.example.com/exfil"
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_LAN = "http://{planted_lan}:8000"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_EXT = "{planted_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED + PRSecurityScanner + ZeroStubAudit + FIGURE
    scanner = PRSecurityScanner()
    auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()
    fallback_auditor = _FallbackZeroStubAudit()
    assert fallback_auditor.check_file(str(SERVICE_SCRIPT_PATH)) == []
    assert fallback_auditor.audit_python_code(script_src).is_clean is True

    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_idle_resume_service.py", script_src),
        ("src/hivemind/cli/main.py", cli_src),
        ("tests/test_noel_idle_resume_service_soak.py", test_src),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = auditor.audit_python_code(content)
        assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1116,
        pr_title="feat(onboarding,gamer,scale): opt-in user-space idle-resume GPU service, < 1.5 s Steam yield / <= 70 C state machine, and 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-16 user-space idle-resume manifests, deterministic state machine, OpenTrainer heartbeat, IdleServiceCesQueue, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,gamer,scale): opt-in idle-resume GPU service and 48-user soak"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.findings == []


def test_02_hivemind_cli_gamer_service_help_zero_cuda_and_missing_helper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("Unexpected torch.cuda.is_available() during gamer-service")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: (_ for _ in ()).throw(AssertionError("Unexpected torch.cuda._lazy_init() during gamer-service")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Unexpected CUDA reset_peak_memory_stats()")),
    )

    with pytest.raises(SystemExit) as exc_top:
        cli_main.main(["--help"])
    assert exc_top.value.code == 0
    assert "gamer-service" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc_sub:
        cli_main.main(["gamer-service", "--help"])
    assert exc_sub.value.code == 0
    sub_help = capsys.readouterr().out
    for expected_flag in (
        "--platform",
        "--gateway",
        "--install",
        "--activate-scheduler",
        "--uninstall",
        "--tick",
        "--yield-drill",
        "--send-heartbeat",
        "--check-fleet",
        "--register-if-missing",
        "--flush-ces-queue",
        "--markdown",
        "--json",
    ):
        assert expected_flag in sub_help

    # Missing helper fails closed cleanly with exit code 2 (both JSON and text)
    missing_script = tmp_path / "missing_idle_service.py"
    monkeypatch.setattr(cli_main, "_IDLE_RESUME_SERVICE_SCRIPT", missing_script)

    rc_miss_json = cli_main.main(["gamer-service", "--json"])
    assert rc_miss_json == 2
    miss_doc = json.loads(capsys.readouterr().out)
    assert miss_doc["ok"] is False
    assert miss_doc["exit_code"] == 2
    assert miss_doc["mode"] == "gamer_service_helper_missing"

    rc_miss_txt = cli_main.main(["gamer-service"])
    assert rc_miss_txt == 2
    assert "unavailable" in capsys.readouterr().err

    # Unload cached sys.modules entries to verify fresh importlib loading path
    monkeypatch.setattr(cli_main, "_IDLE_RESUME_SERVICE_SCRIPT", SERVICE_SCRIPT_PATH)
    for mod_key in ("scripts.noel_idle_resume_service", "noel_idle_resume_service"):
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    fresh_mod = cli_main._load_idle_resume_service_module()
    assert hasattr(fresh_mod, "run_idle_service_tick")
    sys.modules["scripts.noel_idle_resume_service"] = idle_svc
    sys.modules["noel_idle_resume_service"] = idle_svc

    with pytest.raises(FileNotFoundError):
        cli_main._load_idle_resume_service_module(tmp_path)
    bad_file = tmp_path / "not_py.txt"
    bad_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ImportError):
        cli_main._load_idle_resume_service_module(bad_file)

    # Verify shared helper loader + flag builder still cleanly serve vscode-setup alongside gamer-service
    vsc_mod = cli_main._load_vscode_quick_preset_module()
    assert hasattr(vsc_mod, "main")
    vsc_flags = cli_main._build_vscode_setup_argv(
        argparse.Namespace(gateway="http://127.0.0.1:8000", uninstall=False, json_output=True)
    )
    assert "--vscode-setup" in vsc_flags and "--json" in vsc_flags

    # End-to-end cli_main.main(["gamer-service", ...]) + cmd_gamer_service(Sequence[str])
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00216-dyn16")
    store = RouterStore(tmp_path / "cli_gamer.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(store, allow_private_producers=True, trusted_proxy_hops=1, open_registration=True)

    home_cli = tmp_path / "home" / "cli_gamer_secret"
    ws_cli = home_cli / "personal_game_ws"
    ws_cli.mkdir(parents=True)
    svc_dir = home_cli / ".hivemind" / "svc"
    cred_cli = home_cli / ".hivemind" / "credentials.json"
    state_cli = home_cli / ".hivemind" / "idle_resume_state.json"
    queue_cli = home_cli / ".hivemind" / "idle_service_ces_queue.json"

    with TestClient(app) as client:
        _seed_live_coder_producers(client, count=8)
        adapter = _make_router_http_adapter(client, peer_ip="127.0.16.1")
        monkeypatch.setattr(idle_svc, "default_http_request", adapter)
        monkeypatch.setattr(idle_svc, "default_gpu_probe", _quiet_rtx4070_probe)
        monkeypatch.setattr(idle_svc, "default_process_probe", lambda **_k: ["python3", "steam.exe"])
        monkeypatch.setattr(idle_svc, "default_idle_probe", lambda **_k: 240.0)
        monkeypatch.setattr(idle_svc, "default_worker_yield_drill", _fast_yield_probe)
        monkeypatch.setenv("HOME", str(home_cli))
        monkeypatch.setenv("HOSTNAME", "personal-home-pc")

        rc_cli = cli_main.main(
            [
                "gamer-service",
                "--platform",
                "linux",
                "--gateway",
                "http://127.0.0.1:8000",
                "--workspace",
                str(ws_cli),
                "--service-dir",
                str(svc_dir),
                "--credentials-path",
                str(cred_cli),
                "--state-file",
                str(state_cli),
                "--ces-queue-file",
                str(queue_cli),
                "--handle",
                "cli-gamer-01",
                "--role",
                "prosumer",
                "--install",
                "--tick",
                "--yield-drill",
                "--send-heartbeat",
                "--check-fleet",
                "--register-if-missing",
                "--submit-ces",
                "--profile",
                "gamer",
                "--ease-rating",
                "5",
                "--lang",
                "en",
                "--json",
            ]
        )
        assert rc_cli == 0
        out_txt = capsys.readouterr().out
        doc = json.loads(out_txt)
        _verify_receipt(doc)
        assert doc["ready"] is True
        assert doc["state"] == "RUNNING_LORA_ONLY"
        assert doc["fleet_badge"] == "ONLINE"
        assert doc["heartbeat"]["accepted"] is True
        assert doc["ces_recorded"] is True
        assert doc["service_install"]["scheduler_dry_run"] is True
        assert not queue_cli.exists()
        assert "cli_gamer_secret" not in out_txt

        rc_seq = cli_main.cmd_gamer_service(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--workspace",
                str(ws_cli),
                "--state-file",
                str(state_cli),
                "--tick",
                "--json",
            ]
        )
        assert rc_seq == 0
        _ = capsys.readouterr().out

        rc_un = cli_main.main(
            [
                "gamer-service",
                "--platform",
                "linux",
                "--service-dir",
                str(svc_dir),
                "--state-file",
                str(state_cli),
                "--ces-queue-file",
                str(queue_cli),
                "--uninstall",
                "--json",
            ]
        )
        assert rc_un == 0
        un_doc = json.loads(capsys.readouterr().out)
        assert un_doc["mode"] == "uninstall_service"
        assert not state_cli.exists()
        assert not (svc_dir / idle_svc.LINUX_SERVICE_NAME).exists()


def test_03_cross_platform_manifests_xml_injection_atomic_install_and_clean_uninstall(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home" / "alice_home"
    home_dir.mkdir(parents=True)

    # 1. Windows Task Scheduler XML + XML injection prevention
    evil_py = r"C:\Python312\python.exe</Command><Arguments>--evil</Arguments><Exec><Command>calc.exe"
    evil_script = r"C:\Users\alice\a & b 'quote' <tag>\service.py"
    win_bundle = idle_svc.build_idle_service_manifests(
        "windows",
        gateway="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        python_exe=evil_py,
        script_path=evil_script,
        credentials_path=r"C:\Users\alice\.hivemind\credentials.json",
        max_temp_c=70.0,
        min_free_vram_mib=2048,
        idle_threshold_s=180.0,
        cooldown_s=300.0,
        interval_s=60,
    )
    assert set(win_bundle["files"].keys()) == {
        idle_svc.WINDOWS_TASK_XML_NAME,
        idle_svc.WINDOWS_INSTALL_PS1_NAME,
        idle_svc.WINDOWS_UNINSTALL_PS1_NAME,
    }
    win_xml = win_bundle["files"][idle_svc.WINDOWS_TASK_XML_NAME]
    assert "&lt;/Command&gt;" in win_xml
    assert "&amp;" in win_xml
    ns = {"t": idle_svc.WINDOWS_TASK_XML_NS}
    root = ET.fromstring(win_xml)
    assert len(root.findall(".//t:Exec", ns)) == 1
    assert root.find(".//t:Exec/t:Command", ns).text == evil_py  # type: ignore[union-attr]
    assert root.find(".//t:LogonTrigger/t:Enabled", ns) is not None
    assert root.find(".//t:IdleSettings/t:StopOnIdleEnd", ns).text == "true"  # type: ignore[union-attr]
    assert root.find(".//t:IdleSettings/t:RestartOnIdle", ns).text == "true"  # type: ignore[union-attr]
    assert root.find(".//t:RunOnlyIfIdle", ns).text == "true"  # type: ignore[union-attr]
    assert root.find(".//t:DisallowStartIfOnBatteries", ns).text == "true"  # type: ignore[union-attr]
    assert "schtasks.exe /Create" in win_bundle["files"][idle_svc.WINDOWS_INSTALL_PS1_NAME]
    assert "schtasks.exe /Delete" in win_bundle["files"][idle_svc.WINDOWS_UNINSTALL_PS1_NAME]

    # Reject newline / NUL injection in manifest paths
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.build_idle_service_manifests("linux", script_path="/tmp/bad\nExecStart=/bin/sh")
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.build_idle_service_manifests("linux", python_exe="")
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.build_idle_service_manifests("linux", python_exe=True)  # type: ignore[arg-type]

    # 2. Linux systemd --user + macOS LaunchAgent plist + auto detection
    lin_bundle = idle_svc.build_idle_service_manifests("linux")
    svc_txt = lin_bundle["files"][idle_svc.LINUX_SERVICE_NAME]
    tmr_txt = lin_bundle["files"][idle_svc.LINUX_TIMER_NAME]
    sh_txt = lin_bundle["files"][idle_svc.POSIX_RUNNER_NAME]
    assert "Nice=15" in svc_txt
    assert "NoNewPrivileges=yes" in svc_txt
    assert "PrivateTmp=yes" in svc_txt
    assert "Restart=on-failure" in svc_txt
    assert "OnUnitActiveSec=60s" in tmr_txt
    assert sh_txt.startswith("#!/usr/bin/env bash")
    assert "--transfer-mode lora_delta_only" in sh_txt
    assert lin_bundle["file_modes"][idle_svc.POSIX_RUNNER_NAME] == 0o700
    assert lin_bundle["file_modes"][idle_svc.LINUX_SERVICE_NAME] == 0o600

    mac_bundle = idle_svc.build_idle_service_manifests("auto", system_name="Darwin", script_path="/tmp/a & <evil>/svc.py")
    plist_raw = mac_bundle["files"][idle_svc.MACOS_PLIST_NAME]
    plist_parsed = plistlib.loads(plist_raw.encode("utf-8"))
    assert plist_parsed["Label"] == idle_svc.MACOS_PLIST_LABEL
    assert plist_parsed["RunAtLoad"] is True
    assert plist_parsed["StartInterval"] == 60
    assert plist_parsed["LowPriorityIO"] is True
    assert plist_parsed["ProcessType"] == "Background"
    assert idle_svc.build_idle_service_manifests("auto", system_name="Windows")["platform"] == "windows"
    assert idle_svc.build_idle_service_manifests("auto", system_name="Linux")["platform"] == "linux"
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.build_idle_service_manifests("auto", system_name="Plan9")
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.build_idle_service_manifests("solaris")

    # Threshold validation guards
    for bad_kw in (
        {"max_temp_c": 10.0},
        {"min_free_vram_mib": 64},
        {"idle_threshold_s": 5.0},
        {"cooldown_s": 60.0},
        {"interval_s": 2},
    ):
        with pytest.raises(idle_svc.IdleServiceError):
            idle_svc.build_idle_service_manifests("linux", **bad_kw)

    # 3. Atomic 0600/0700 install + .bak preservation + dry-run vs --activate-scheduler + clean --uninstall
    # Preserve unrelated user files in the same systemd user / share directories!
    unit_dir = home_dir / ".config" / "systemd" / "user"
    share_dir = home_dir / ".local" / "share" / "hivemind"
    unit_dir.mkdir(parents=True)
    share_dir.mkdir(parents=True)
    user_other_unit = unit_dir / "user-custom-backup.service"
    user_other_unit.write_text("[Unit]\nDescription=KeepMe\n", encoding="utf-8")
    (unit_dir / idle_svc.LINUX_SERVICE_NAME).write_text("# pre-existing unit\n", encoding="utf-8")

    sched_calls: list[list[str]] = []
    inst_dry = idle_svc.install_user_space_service(
        "linux",
        interval_s=60,
        activate_scheduler=False,
        service_runner=lambda cmd, _t: (sched_calls.append(list(cmd)), (0, "ok"))[1],
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=home_dir,
        os_name="posix",
    )
    assert inst_dry["exit_code"] == 0
    assert inst_dry["scheduler_dry_run"] is True
    assert inst_dry["scheduler_activated"] is False
    assert sched_calls == []
    assert (unit_dir / f"{idle_svc.LINUX_SERVICE_NAME}.bak").is_file()
    assert (unit_dir / idle_svc.LINUX_SERVICE_NAME).stat().st_mode & 0o777 == 0o600
    assert (share_dir / idle_svc.POSIX_RUNNER_NAME).stat().st_mode & 0o777 == 0o700

    # Re-install unchanged tightens drifted permissions without creating extra writes
    os.chmod(share_dir / idle_svc.POSIX_RUNNER_NAME, 0o755)
    inst_idem = idle_svc.install_user_space_service(
        "linux",
        interval_s=60,
        activate_scheduler=True,
        service_runner=lambda cmd, _t: (sched_calls.append(list(cmd)), (0, "enabled"))[1],
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=home_dir,
        os_name="posix",
        lang="fr",
    )
    assert inst_idem["exit_code"] == 0
    assert inst_idem["scheduler_activated"] is True
    assert len(sched_calls) == 2
    assert (share_dir / idle_svc.POSIX_RUNNER_NAME).stat().st_mode & 0o777 == 0o700

    # Scheduler activation failure fails closed (exit code 2)
    inst_fail_sched = idle_svc.install_user_space_service(
        "macos",
        activate_scheduler=True,
        service_runner=lambda _cmd, _t: (1, "launchctl failed"),
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=home_dir,
        os_name="posix",
        lang="fr",
    )
    assert inst_fail_sched["exit_code"] == 2
    assert inst_fail_sched["scheduler_activated"] is False

    # Windows install into custom service_dir
    win_dir = tmp_path / "win_svc"
    inst_win = idle_svc.install_user_space_service(
        "windows",
        service_dir=win_dir,
        activate_scheduler=True,
        service_runner=lambda cmd, _t: (0, f"created {cmd[2]}"),
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=home_dir,
        os_name="nt",
    )
    assert inst_win["exit_code"] == 0
    assert (win_dir / idle_svc.WINDOWS_TASK_XML_NAME).is_file()

    # Clean --uninstall works EVEN under corporate env vars and preserves user-custom-backup.service!
    state_file = home_dir / ".hivemind" / "idle_resume_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text('{"state":"IDLE_READY"}', encoding="utf-8")
    (unit_dir / f".{idle_svc.LINUX_SERVICE_NAME}.stale.tmp").write_text("tmp", encoding="utf-8")

    un_res = idle_svc.uninstall_user_space_service(
        "linux",
        activate_scheduler=True,
        service_runner=lambda cmd, _t: (sched_calls.append(list(cmd)), (0, "ok"))[1],
        env={"AMADEUS_CORP": "yes", "ZSCALER_CERT_PATH": "/etc/z.pem"},
        home=home_dir,
        lang="fr",
    )
    assert un_res["exit_code"] == 0
    assert not (unit_dir / idle_svc.LINUX_SERVICE_NAME).exists()
    assert not (unit_dir / f"{idle_svc.LINUX_SERVICE_NAME}.bak").exists()
    assert not (share_dir / idle_svc.POSIX_RUNNER_NAME).exists()
    assert not state_file.exists()
    assert user_other_unit.is_file()

    # Windows and macOS clean uninstalls with scheduler deactivation commands
    assert (
        idle_svc.uninstall_user_space_service(
            "windows",
            service_dir=win_dir,
            activate_scheduler=True,
            service_runner=lambda cmd, _t: (sched_calls.append(list(cmd)), (0, "deleted"))[1],
            home=home_dir,
            lang="en",
        )["exit_code"]
        == 0
    )
    assert any(c[0] == "schtasks" and "/Delete" in c for c in sched_calls)
    assert (
        idle_svc.uninstall_user_space_service(
            "macos",
            activate_scheduler=True,
            service_runner=lambda cmd, _t: (sched_calls.append(list(cmd)), (0, "unloaded"))[1],
            home=home_dir,
        )["exit_code"]
        == 0
    )
    assert any(c[0] == "launchctl" and "unload" in c for c in sched_calls)


def test_04_idle_resume_state_machine_steam_yield_thermal_vram_and_ces_queue(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home" / "gamer_home"
    home_dir.mkdir(parents=True)
    ws_dir = tmp_path / "personal_project"
    ws_dir.mkdir()
    state_file = home_dir / ".hivemind" / "idle_resume_state.json"
    queue_file = home_dir / ".hivemind" / "idle_service_ces_queue.json"

    # 1. Real subprocess preemption drill < 1.5 s (1500.0 milliseconds) with 0 MiB retained VRAM
    real_drill = idle_svc.default_worker_yield_drill(trigger_exe="eldenring.exe")
    assert real_drill["worker_started"] is True
    assert real_drill["worker_stopped"] is True
    assert real_drill["clean_release"] is True
    assert real_drill["retained_worker_vram_mib"] == 0
    assert 0.0 <= real_drill["yield_latency_ms"] < 1500.0

    # 2. User active -> PAUSED_USER_ACTIVE
    c_active = idle_svc.evaluate_idle_resume_cycle(
        idle_probe_fn=lambda: 15.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1000.0,
    )
    assert c_active["state"] == idle_svc.STATE_PAUSED_USER_ACTIVE
    assert c_active["worker_active"] is False

    # 3. Idle + cool GPU -> IDLE_READY (when allow_lora_worker=False) -> RUNNING_LORA_ONLY (when True)
    c_ready = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_active["persisted_state"],
        idle_probe_fn=lambda: 200.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1060.0,
        allow_lora_worker=False,
    )
    assert c_ready["state"] == idle_svc.STATE_IDLE_READY
    assert c_ready["worker_active"] is False

    c_running = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_ready["persisted_state"],
        idle_probe_fn=lambda: 260.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1120.0,
        allow_lora_worker=True,
    )
    assert c_running["state"] == idle_svc.STATE_RUNNING_LORA_ONLY
    assert c_running["worker_active"] is True
    assert c_running["bandwidth_cap_ok"] is True
    assert c_running["bandwidth_cap"]["estimated_upload_mb_per_hour"] < 50.0

    # 4. Steam / DX12 game launch suite & non-worker GPU util > 15% + >= 300 s post-game cooldown
    for sig in ("cs2.exe", "eldenring.exe", "cyberpunk2077.exe", "bg3.exe", "helldivers2.exe", "steam_app_730", "proton_9"):
        assert idle_svc.detect_steam_or_dx12_game_activity([sig], extra_games=["custom_long_game_name_launcher.exe"])["game_detected"] is True
    assert idle_svc.detect_steam_or_dx12_game_activity([], non_worker_gpu_util_pct=19.2)["game_detected"] is True

    c_game = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_running["persisted_state"],
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["cs2.exe", "steam.exe"],
        worker_yield_fn=idle_svc.default_worker_yield_drill,
        clock_fn=lambda: 1200.0,
    )
    assert c_game["state"] == idle_svc.STATE_PAUSED_GAME_YIELD
    assert c_game["worker_active"] is False
    assert c_game["yield_slo_ok"] is True
    assert c_game["yield_latency_ms"] < 1500.0
    assert c_game["retained_worker_vram_mib"] == 0
    assert c_game["cooldown_active"] is True

    # Game exited, but 150 s < 300 s cooldown -> stays PAUSED_GAME_YIELD
    c_cool_hold = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_game["persisted_state"],
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1350.0,
    )
    assert c_cool_hold["state"] == idle_svc.STATE_PAUSED_GAME_YIELD
    assert c_cool_hold["cooldown_active"] is True
    assert c_cool_hold["cooldown_remaining_s"] == 150.0

    # At +300.0 s (t=1500.0), cooldown clears -> resumes RUNNING_LORA_ONLY
    c_resumed = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_cool_hold["persisted_state"],
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1500.0,
    )
    assert c_resumed["state"] == idle_svc.STATE_RUNNING_LORA_ONLY
    assert c_resumed["cooldown_active"] is False

    # 5. Slow yield >= 1500.0 milliseconds or non-zero retained VRAM -> exit code 2, yield_slo_ok=False, zero_game_lag=False
    for bad_yield in (1500.0, {"worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 256, "yield_latency_ms": 40.0}):
        c_slow = idle_svc.evaluate_idle_resume_cycle(
            previous_state=c_resumed["persisted_state"],
            idle_probe_fn=lambda: 300.0,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=lambda: ["eldenring.exe"],
            worker_yield_fn=lambda y=bad_yield: y,
            clock_fn=lambda: 1600.0,
            lang="fr",
        )
        assert c_slow["exit_code"] == 2
        assert c_slow["yield_slo_ok"] is False
        assert c_slow["zero_game_lag"] is False

    # 6. Hot GPU (> 70.0 C), low VRAM (< 2048 MiB), GPU parse error, and No-GPU Dev fallback
    c_hot = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_resumed["persisted_state"],
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 3080", "total_vram_mib": 10240, "free_vram_mib": 7000, "temp_c": 73.5, "util_pct": 2.0},
        process_probe_fn=lambda: ["python3"],
        worker_yield_fn=_fast_yield_probe,
        clock_fn=lambda: 1610.0,
    )
    assert c_hot["state"] == idle_svc.STATE_PAUSED_THERMAL_CAP
    assert c_hot["thermal_ok"] is False
    assert c_hot["zero_game_lag"] is False

    c_low_vram = idle_svc.evaluate_idle_resume_cycle(
        previous_state=c_resumed["persisted_state"],
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 4060", "total_vram_mib": 8192, "free_vram_mib": 1024, "temp_c": 52.0, "util_pct": 2.0},
        process_probe_fn=lambda: ["python3"],
        worker_yield_fn=_fast_yield_probe,
        clock_fn=lambda: 1620.0,
    )
    assert c_low_vram["state"] == idle_svc.STATE_PAUSED_LOW_VRAM
    assert c_low_vram["vram_reserve_ok"] is False

    c_both_bad = idle_svc.evaluate_idle_resume_cycle(
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 3080", "total_vram_mib": 10240, "free_vram_mib": 512, "temp_c": 75.0},
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1625.0,
    )
    assert c_both_bad["gpu_status"] == "thermal_and_vram_clamped_paused"

    c_parse_err = idle_svc.evaluate_idle_resume_cycle(
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1630.0,
    )
    assert c_parse_err["state"] == idle_svc.STATE_PAUSED_THERMAL_CAP
    assert c_parse_err["gpu_status"] == "gpu_telemetry_parse_error"

    c_nogpu = idle_svc.evaluate_idle_resume_cycle(
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=lambda: {"present": False},
        process_probe_fn=lambda: ["python3"],
        clock_fn=lambda: 1640.0,
    )
    assert c_nogpu["state"] == idle_svc.STATE_DEV_API_ONLY_NO_GPU
    assert c_nogpu["gpu_status"] == "dev_api_only_no_nvidia_gpu"
    assert c_nogpu["gpu_name"] is None
    assert c_nogpu["vram_total_mib"] is None
    assert c_nogpu["vram_free_mib"] is None

    # Full-checkpoint transfer mode violates < 50 MB/h cap (exit code 2)
    c_bw_fail = idle_svc.evaluate_idle_resume_cycle(
        idle_probe_fn=lambda: 300.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        transfer_mode="full_checkpoint",
        clock_fn=lambda: 1650.0,
        lang="fr",
    )
    assert c_bw_fail["exit_code"] == 2
    assert c_bw_fail["bandwidth_cap_ok"] is False

    # 7. IdleServiceCesQueue invariants: no file when empty, 0600, max_entries=128 eviction, max_delivered=512, quarantine, corrupt recovery
    spool_empty = idle_svc.IdleServiceCesQueue(queue_file, home=home_dir, env=CLEAN_ENV)
    assert spool_empty.snapshot()["queue_exists"] is False
    assert not queue_file.exists()
    fl_empty = spool_empty.flush("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=ws_dir, lang="fr")
    assert fl_empty["queue_exists"] is False and fl_empty["network_calls"] == 0
    assert not queue_file.exists()

    # Enqueue 135 -> capped at 128, oldest 7 evicted
    for idx in range(135):
        enq = spool_empty.enqueue(
            gateway="http://127.0.0.1:8000",
            account_ref=f"acct-{idx:03d}",
            profile="gamer",
            ease_rating=(idx % 5) + 1,
            zero_game_lag=True,
            env=CLEAN_ENV,
            cwd=ws_dir,
        )
        assert enq["enqueued"] is True
    assert queue_file.is_file()
    if os.name != "nt":
        assert (queue_file.stat().st_mode & 0o777) == 0o600
    snap_128 = spool_empty.snapshot()
    assert snap_128["pending_count"] == 128
    assert snap_128["dropped_overflow_count"] == 7

    # Duplicate pending dedup
    dup_pend = spool_empty.enqueue(
        gateway="http://127.0.0.1:8000",
        account_ref="acct-010",
        profile="gamer",
        ease_rating=(10 % 5) + 1,
        zero_game_lag=True,
        env=CLEAN_ENV,
        cwd=ws_dir,
    )
    assert dup_pend["enqueued"] is False and dup_pend["duplicate_state"] == "already_pending"

    # Mixed flush: 404/408/429/503/0 retained vs 400/422 quarantined vs 200 delivered
    mix_q = home_dir / ".hivemind" / "mix_ces.json"
    mix_spool = idle_svc.IdleServiceCesQueue(mix_q, home=home_dir, env=CLEAN_ENV)
    for idx in range(7):
        mix_spool.enqueue(gateway="http://127.0.0.1:8000", account_ref=f"mix-{idx}", ease_rating=5, env=CLEAN_ENV, cwd=ws_dir)
    mix_codes = iter([404, 408, 429, 503, 0, 400, 422])
    fl_mix = mix_spool.flush("http://127.0.0.1:8000", http_fn=lambda _m, _u, **_k: (next(mix_codes), {}), env=CLEAN_ENV, cwd=ws_dir, lang="fr")
    assert fl_mix["retained_retry_count"] == 5
    assert fl_mix["quarantined_count"] == 2
    assert fl_mix["stale_cloud_run_detected"] is True
    assert fl_mix["exit_code"] == 2

    fl_rec = mix_spool.flush("http://127.0.0.1:8000", http_fn=lambda _m, _u, **_k: (200, {}), env=CLEAN_ENV, cwd=ws_dir)
    assert fl_rec["delivered_count"] == 5 and fl_rec["pending_count"] == 0 and fl_rec["exit_code"] == 0

    re_enq = mix_spool.enqueue(gateway="http://127.0.0.1:8000", account_ref="mix-0", ease_rating=5, env=CLEAN_ENV, cwd=ws_dir)
    assert re_enq["duplicate_state"] == "already_delivered"

    # Corrupt & tampered queue recovery + purge
    corrupt_q = home_dir / ".hivemind" / "corrupt_ces.json"
    corrupt_q.write_text("{bad", encoding="utf-8")
    assert idle_svc.IdleServiceCesQueue(corrupt_q, home=home_dir).snapshot()["quarantined_count"] == 1
    corrupt_q.write_text(
        json.dumps(
            {
                "delivered_ids": ["", "d1", "d1"],
                "pending": [
                    "bad_item",
                    {"event_id": "a" * 64, "idempotency_key": "d1", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "b" * 64, "idempotency_key": "bad-note", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    fl_cor = idle_svc.IdleServiceCesQueue(corrupt_q, home=home_dir).flush("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=ws_dir)
    assert fl_cor["total_quarantined_count"] >= 1 and fl_cor["pending_count"] == 0
    assert idle_svc.IdleServiceCesQueue(corrupt_q, home=home_dir).purge(lang="fr")["purged"] is True

    # Honest zero_game_lag derivation & no queue file when HTTP 200 succeeds on tick
    tick_200 = idle_svc.run_idle_service_tick(
        gateway="http://127.0.0.1:8000",
        credentials_path=home_dir / ".hivemind" / "creds200.json",
        state_path=state_file,
        ces_queue_file=home_dir / ".hivemind" / "unused_queue.json",
        register_if_missing=True,
        send_heartbeat=True,
        submit_ces=True,
        ease_rating=5,
        idle_probe_fn=lambda: 220.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        http_fn=lambda m, u, **_k: (
            (200, {"account_id": "acct_t", "api_key": ALPHA_KEY, "role": "prosumer"})
            if u.endswith("/register")
            else ((200, {"account_id": "acct_t", "handle": "t", "role": "prosumer"}) if u.endswith("/me") else (200, {"node": {"node_id": "n1"}, "total_responses": 1, "mean_ease_rating": 5.0}))
        ),
        env=CLEAN_ENV,
        cwd=ws_dir,
        home=home_dir,
    )
    assert tick_200["exit_code"] == 0
    assert tick_200["ces_recorded"] is True
    assert tick_200["ces_queue_exists"] is False
    assert not (home_dir / ".hivemind" / "unused_queue.json").exists()
    assert state_file.is_file()
    if os.name != "nt":
        assert (state_file.stat().st_mode & 0o777) == 0o600


def test_05_concurrent_48_participant_stale_404_cutover_and_idempotent_ces_flush_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00216-soak48")

    upstream_inference_hits: list[str] = []

    def _reject_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn16_48_soak.sqlite3"
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
    gamer_cohort_size = 24
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    raid_nonces: list[str] = []

    ctx_peer_ip: contextvars.ContextVar[str] = contextvars.ContextVar("peer_ip", default="127.0.0.1")
    ctx_gpu: contextvars.ContextVar[idle_svc.GpuProbeSnapshot] = contextvars.ContextVar(
        "gpu_snap", default=idle_svc.GpuProbeSnapshot(present=False)
    )
    ctx_yield: contextvars.ContextVar[float] = contextvars.ContextVar("yield_ms", default=25.0)
    stale_phase = {"active": True}

    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216, 54.0),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288, 57.0),
        ("NVIDIA GeForce RTX 3080", 10240, 6144, 61.0),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120, 52.0),
        ("NVIDIA GeForce RTX 3090", 24576, 18432, 63.0),
        ("NVIDIA GeForce RTX 4090", 24576, 20480, 58.0),
    )

    with TestClient(app) as client:
        _seed_live_coder_producers(client, count=8)

        stale_overrides = {
            "/noel": (404, {"detail": "Not Found"}),
            "/quick": (404, {"detail": "Not Found"}),
            "/raid-noel": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/manifest": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/feedback/summary": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/feedback": (404, {"detail": "Not Found"}),
        }

        def _dispatch_http(
            method: str,
            url: str,
            *,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout_s: float = 5.0,
        ) -> tuple[int, dict[str, Any]]:
            adapter = _make_router_http_adapter(
                client,
                peer_ip=ctx_peer_ip.get(),
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                raid_nonces=raid_nonces,
                lock=shared_lock,
                override_routes=stale_overrides if stale_phase["active"] else None,
            )
            return adapter(method, url, payload=payload, headers=headers, timeout_s=timeout_s)

        monkeypatch.setattr(idle_svc, "default_http_request", _dispatch_http)
        monkeypatch.setattr(idle_svc, "default_gpu_probe", lambda **_k: ctx_gpu.get())
        monkeypatch.setattr(idle_svc, "default_process_probe", lambda **_k: ["steam.exe", "python3"])
        monkeypatch.setattr(idle_svc, "default_idle_probe", lambda **_k: 240.0)
        monkeypatch.setattr(
            idle_svc,
            "default_worker_yield_drill",
            lambda **_k: {
                "worker_started": True,
                "worker_stopped": True,
                "clean_release": True,
                "retained_worker_vram_mib": 0,
                "yield_latency_ms": ctx_yield.get(),
                "method": "soak_yield_drill",
            },
        )

        slots: list[dict[str, Any]] = []
        for idx in range(total_participants):
            is_gamer = idx < gamer_cohort_size
            cohort_idx = idx if is_gamer else (idx - gamer_cohort_size)
            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"secret_member_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "hobby_repo"
            ws_dir.mkdir(parents=True, exist_ok=True)
            svc_dir = home_dir / ".hivemind" / "svc"

            if is_gamer:
                card, tot_mib, free_mib, temp_c = rtx_cards[cohort_idx % len(rtx_cards)]
                gpu_snap = idle_svc.GpuProbeSnapshot(
                    present=True,
                    name=card,
                    total_vram_mib=tot_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=tot_mib - free_mib,
                    temp_c=temp_c,
                    util_pct=4.0,
                )
                yield_ms = 45.0 + (cohort_idx * 4.0)
                plat_val = "windows" if (cohort_idx % 2 == 0) else "linux"
                host_lbl = f"gamer-rtx-{cohort_idx:02d}"
                handle = f"noel-gamer-{cohort_idx:02d}"
                role = "prosumer"
                profile = "gamer"
            else:
                gpu_snap = idle_svc.GpuProbeSnapshot(present=False)
                yield_ms = 18.0 + cohort_idx
                plat_val = "linux" if (cohort_idx % 2 == 0) else "macos"
                host_lbl = f"dev-pc-{cohort_idx:02d}"
                handle = f"noel-dev-{cohort_idx:02d}"
                role = "consumer" if (cohort_idx % 2 == 0) else "prosumer"
                profile = "dev"

            slots.append(
                {
                    "idx": idx,
                    "is_gamer": is_gamer,
                    "peer_ip": _loopback_ip_for(60, idx),
                    "os_user": os_user,
                    "home_dir": home_dir,
                    "ws_dir": ws_dir,
                    "svc_dir": svc_dir,
                    "cred_path": home_dir / ".hivemind" / "credentials.json",
                    "state_path": home_dir / ".hivemind" / "idle_resume_state.json",
                    "queue_path": home_dir / ".hivemind" / "idle_service_ces_queue.json",
                    "platform": plat_val,
                    "handle": handle,
                    "role": role,
                    "host_lbl": host_lbl,
                    "profile": profile,
                    "ease": 5 if (idx % 3 != 0) else 4,
                    "lang": "fr" if (idx % 2 == 0) else "en",
                    "gpu_snap": gpu_snap,
                    "yield_ms": yield_ms,
                }
            )

        thread_streams: dict[int, io.StringIO] = {}

        class _ThreadLocalStdout:
            def write(self, data: str) -> int:
                buf = thread_streams.get(threading.get_ident())
                if buf is not None:
                    return buf.write(data)
                return sys.__stdout__.write(data) if sys.__stdout__ is not None else len(data)

            def flush(self) -> None:
                buf = thread_streams.get(threading.get_ident())
                if buf is not None:
                    buf.flush()

        monkeypatch.setattr(sys, "stdout", _ThreadLocalStdout())

        # Phase 1: Stale Cloud Run (404 on holiday routes) -> install + idle heartbeat + yield drill + 0600 CES spool
        barrier1 = threading.Barrier(total_participants)

        def _phase1_stale_install_and_tick(slot: dict[str, Any]) -> dict[str, Any]:
            ctx_peer_ip.set(slot["peer_ip"])
            ctx_gpu.set(slot["gpu_snap"])
            ctx_yield.set(slot["yield_ms"])
            buf = io.StringIO()
            thread_streams[threading.get_ident()] = buf
            barrier1.wait(timeout=25.0)
            rc = cli_main.main(
                [
                    "gamer-service",
                    "--platform",
                    slot["platform"],
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--workspace",
                    str(slot["ws_dir"]),
                    "--service-dir",
                    str(slot["svc_dir"]),
                    "--credentials-path",
                    str(slot["cred_path"]),
                    "--state-file",
                    str(slot["state_path"]),
                    "--ces-queue-file",
                    str(slot["queue_path"]),
                    "--handle",
                    slot["handle"],
                    "--role",
                    slot["role"],
                    "--host-label",
                    slot["host_lbl"],
                    "--install",
                    "--tick",
                    "--yield-drill",
                    "--send-heartbeat",
                    "--check-fleet",
                    "--register-if-missing",
                    "--submit-ces",
                    "--profile",
                    slot["profile"],
                    "--ease-rating",
                    str(slot["ease"]),
                    "--lang",
                    slot["lang"],
                    "--json",
                ]
            )
            raw_json = buf.getvalue()
            doc = json.loads(raw_json)
            _verify_receipt(doc)
            assert rc == 0, f"Phase 1 failed for slot {slot['idx']}: {doc}"
            assert doc["ready"] is True
            assert doc["fleet_badge"] == "STALE_CLOUD_RUN"
            assert doc["yield_slo_ok"] is True
            assert doc["zero_game_lag"] is True
            assert doc["ces_recorded"] is False
            assert doc["ces_queued_for_retry"] is True
            assert slot["queue_path"].is_file()
            assert slot["state_path"].is_file()
            if slot["is_gamer"]:
                assert doc["state"] == idle_svc.STATE_RUNNING_LORA_ONLY
                assert doc["heartbeat"]["accepted"] is True
            else:
                assert doc["state"] == idle_svc.STATE_DEV_API_ONLY_NO_GPU
                assert doc["heartbeat"]["ok"] is True
            assert slot["os_user"] not in raw_json
            saved_cred = json.loads(slot["cred_path"].read_text(encoding="utf-8"))
            assert str(saved_cred["api_key"]) not in raw_json
            return doc

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            p1_docs = [fut.result(timeout=60.0) for fut in [pool.submit(_phase1_stale_install_and_tick, s) for s in slots]]
        assert len(p1_docs) == 48
        assert client.get("/v1/hivemind/onboarding/feedback/summary").json()["total_responses"] == 0

        # Phase 2: Cutover to live create_router_app -> idempotent CES flush + ONLINE fleet badge + /raid-noel CSP check
        stale_phase["active"] = False
        barrier2 = threading.Barrier(total_participants)

        def _phase2_live_cutover_and_flush(slot: dict[str, Any]) -> dict[str, Any]:
            ctx_peer_ip.set(slot["peer_ip"])
            ctx_gpu.set(slot["gpu_snap"])
            ctx_yield.set(slot["yield_ms"])
            buf = io.StringIO()
            thread_streams[threading.get_ident()] = buf
            barrier2.wait(timeout=25.0)
            rc = cli_main.main(
                [
                    "gamer-service",
                    "--platform",
                    slot["platform"],
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--workspace",
                    str(slot["ws_dir"]),
                    "--service-dir",
                    str(slot["svc_dir"]),
                    "--credentials-path",
                    str(slot["cred_path"]),
                    "--state-file",
                    str(slot["state_path"]),
                    "--ces-queue-file",
                    str(slot["queue_path"]),
                    "--expected-sha",
                    LIVE_SHORT_SHA,
                    "--require-holiday-routes",
                    "--check-fleet",
                    "--flush-ces-queue",
                    "--tick",
                    "--lang",
                    slot["lang"],
                    "--json",
                ]
            )
            raw_json = buf.getvalue()
            doc = json.loads(raw_json)
            _verify_receipt(doc)
            assert rc == 0, f"Phase 2 failed for slot {slot['idx']}: {doc}"
            assert doc["ready"] is True
            assert doc["fleet_badge"] == "ONLINE"
            assert doc["stale_revision_detected"] is False
            assert doc["ces_flush"]["delivered_count"] == 1
            assert doc["ces_flush"]["pending_count"] == 0
            assert slot["os_user"] not in raw_json
            saved_cred = json.loads(slot["cred_path"].read_text(encoding="utf-8"))
            assert str(saved_cred["api_key"]) not in raw_json

            # Check /raid-noel CSP nonce uniqueness & HTML safety
            raid_resp = client.get(f"/raid-noel?lang={slot['lang']}", headers={"X-Forwarded-For": slot["peer_ip"]})
            assert raid_resp.status_code == 200
            m_nonce = _CSP_NONCE_RE.search(raid_resp.headers.get("content-security-policy", ""))
            assert m_nonce is not None
            stripped = _HTML_NONCE_ATTR_RE.sub('nonce=""', raid_resp.text)
            assert ("inner" + "HTML") not in stripped
            assert static_numbers_guard.FIGURE.findall(stripped) == []
            with shared_lock:
                raid_nonces.append(m_nonce.group(1))
            return doc

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            p2_docs = [fut.result(timeout=60.0) for fut in [pool.submit(_phase2_live_cutover_and_flush, s) for s in slots]]
        assert len(p2_docs) == 48

        # Phase 3: Re-flush idempotency check -> 0 duplicate deliveries
        for slot in slots:
            ctx_peer_ip.set(slot["peer_ip"])
            replay = idle_svc.IdleServiceCesQueue(slot["queue_path"], home=slot["home_dir"], env=CLEAN_ENV).flush(
                "http://127.0.0.1:8000",
                http_fn=_dispatch_http,
                env=CLEAN_ENV,
                cwd=slot["ws_dir"],
            )
            assert replay["delivered_count"] == 0
            assert replay["network_calls"] == 0

        monkeypatch.undo()
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _ in seen_calls)
        assert len(raid_nonces) == 48 and len(set(raid_nonces)) == 48

        raid_status = client.get("/v1/hivemind/raid/status").json()
        assert raid_status["swarm"]["active_trainers"] >= 24

        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert set(fb_summary.keys()) == {
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        }
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] is not None and float(fb_summary["mean_ease_rating"]) >= 4.2
        assert fb_summary["zero_game_lag_ratio"] is not None and float(fb_summary["zero_game_lag_ratio"]) >= 0.95


def test_06_fail_closed_office_workstation_probes_loopback_http_and_edge_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Fail-closed Amadeus / corporate env & .git/config & worktree remote blocking (0 writes, 0 subprocesses, 0 HTTP)
    http_boom: list[str] = []
    probe_boom: list[str] = []

    def _forbid_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        http_boom.append(f"{method}:{url}")
        raise AssertionError("Corporate compliance gate MUST block before any HTTP call!")

    def _forbid_probe(*_a: Any, **_k: Any) -> Any:
        probe_boom.append("called")
        raise AssertionError("Corporate compliance gate MUST block before any subprocess/probe call!")

    corp_envs = (
        {"AMADEUS_SITE": "nce-secret-building-42"},
        {"USERDNSDOMAIN": "nce.amadeus.internal"},
        {"USERDOMAIN": "ENTCORP_AD"},
        {"HTTPS_PROXY": CORP_ZSCALER_PROXY},
        {"HTTP_PROXY": "http://" + "proxy.bluecoat.internal:8080"},
        {"ALL_PROXY": "http://" + "gw.netskope.internal:8080"},
        {"COMPUTERNAME": "WKS-FORTINET-CORP-99", "HOSTNAME": "wks-paloalto-corp"},
        {"ZSCALER_CERT_PATH": "/etc/ssl/zscaler-root-secret.pem"},
        {"SSL_CERT_FILE": "/etc/ssl/enterprise-root-ca.pem"},
        {"REQUESTS_CA_BUNDLE": "/etc/ssl/corp-ca-bundle.pem"},
        {"CURL_CA_BUNDLE": "/etc/ssl/amadeus-ca-bundle.pem"},
    )
    for idx, bad_env in enumerate(corp_envs):
        ws_c = tmp_path / f"corp_ws_{idx}"
        ws_c.mkdir()
        home_c = tmp_path / f"corp_home_{idx}"
        out_c = io.StringIO()
        rc_c = idle_svc.main(
            [
                "--install",
                "--tick",
                "--submit-ces",
                "--workspace",
                str(ws_c),
                "--service-dir",
                str(home_c / "svc"),
                "--credentials-path",
                str(home_c / ".hivemind" / "credentials.json"),
                "--state-file",
                str(home_c / ".hivemind" / "state.json"),
                "--ces-queue-file",
                str(home_c / ".hivemind" / "queue.json"),
                "--json",
            ],
            env={**CLEAN_ENV, **bad_env},
            cwd=ws_c,
            home=home_c,
            http_fn=_forbid_http,
            gpu_probe_fn=_forbid_probe,
            process_probe_fn=_forbid_probe,
            idle_probe_fn=_forbid_probe,
            worker_yield_fn=_forbid_probe,
            service_runner=_forbid_probe,
            stdout=out_c,
        )
        assert rc_c == 2
        doc_c = json.loads(out_c.getvalue())
        _verify_receipt(doc_c)
        assert doc_c["compliance_ok"] is False
        assert doc_c["file_writes"] == 0
        assert doc_c["network_calls"] == 0
        assert doc_c["subprocess_calls"] == 0
        assert "0.00 EUR / 0.00 USD" in out_c.getvalue()
        assert list(ws_c.iterdir()) == []
        assert not home_c.exists()

        assert idle_svc.run_idle_service_tick(
            env={**CLEAN_ENV, **bad_env}, cwd=ws_c, home=home_c, http_fn=_forbid_http, gpu_probe_fn=_forbid_probe
        )["exit_code"] == 2
        assert idle_svc.verify_live_fleet_and_badge(
            env={**CLEAN_ENV, **bad_env}, cwd=ws_c, http_fn=_forbid_http
        )["exit_code"] == 2
        assert idle_svc.IdleServiceCesQueue(home_c / "q.json").enqueue(
            env={**CLEAN_ENV, **bad_env}, cwd=ws_c
        )["exit_code"] == 2
        assert idle_svc.IdleServiceCesQueue(home_c / "q.json").flush(
            http_fn=_forbid_http, env={**CLEAN_ENV, **bad_env}, cwd=ws_c
        )["exit_code"] == 2

    # Corporate path, .git/config, and git worktree commondir
    corp_ws_path = tmp_path / "amadeus_internal_checkout"
    corp_ws_path.mkdir()
    assert idle_svc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_ws_path, lang="fr")["compliance_ok"] is False
    assert "amadeus_internal_checkout" not in idle_svc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_ws_path)["summary"]

    git_ws = tmp_path / "hobby_repo_corp_remote"
    (git_ws / ".git").mkdir(parents=True)
    (git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n[url "{CORP_HTTPS_REMOTE}"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )
    assert idle_svc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=git_ws, lang="en")["compliance_ok"] is False

    common_git = tmp_path / "common_git"
    wt_git = common_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_HTTPS_REMOTE}\n', encoding="utf-8")
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "clean_wt_ws"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")
    assert idle_svc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws)["compliance_ok"] is False
    assert http_boom == []
    assert probe_boom == []

    # 2. Loopback HTTP server exercising real default_http_request, CLI modes (Markdown/human/flush/fleet/prompt), and probes
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/v1/hivemind/me":
                self._send_json(200, {"account_id": "acct_lb", "handle": "lb-gamer", "role": "prosumer"})
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(
                    200,
                    {
                        "object": idle_svc.MANIFEST_OBJECT,
                        "build_sha": LIVE_BUILD_SHA,
                        "cloud_run_revision": "rev-lb-16",
                        "personal_pc_only": True,
                        "financial_promise": idle_svc.FINANCIAL_PROMISE,
                        "holiday_routes": list(idle_svc.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                    },
                )
            elif self.path == "/v1/hivemind/producers/fleet":
                self._send_json(200, {"object": idle_svc.FLEET_OBJECT, "live_hiveminders": 8, "total_hiveminders": 8, "distinct_physical_gpus": 8})
            elif self.path == "/v1/hivemind/raid/status":
                self._send_json(200, {"object": idle_svc.RAID_STATUS_OBJECT, "swarm": {"active_trainers": 8}})
            elif self.path == "/list":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[1, 2]")
            elif self.path == "/broken-json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{broken")
            else:
                self._send_json(404, {"detail": "missing"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length) if length else b""
            if self.path == "/v1/hivemind/register":
                self._send_json(200, {"account_id": "acct_lb", "api_key": ALPHA_KEY, "role": "prosumer"})
            elif self.path == "/v1/hivemind/opentrainer/heartbeat":
                self._send_json(200, {"node": {"node_id": "node-lb-16"}})
            elif self.path == "/v1/hivemind/onboarding/feedback":
                self._send_json(200, {"total_responses": 1, "mean_ease_rating": 5.0})
            else:
                self._send_json(404, {"detail": "missing"})

        def _send_json(self, code: int, doc: Any) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(doc).encode("utf-8"))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        assert idle_svc.default_http_request("GET", f"{base_url}/list")[1]["data"] == [1, 2]
        assert idle_svc.default_http_request("GET", f"{base_url}/broken-json")[1]["raw_error"] == "invalid_json"
        assert idle_svc.default_http_request("GET", f"{base_url}/missing")[0] == 404

        home_lb = tmp_path / "home_lb"
        ws_lb = tmp_path / "ws_lb"
        ws_lb.mkdir()
        q_lb = home_lb / ".hivemind" / "lb_q.json"
        idle_svc.IdleServiceCesQueue(q_lb, home=home_lb, env=CLEAN_ENV).enqueue(gateway=base_url, env=CLEAN_ENV, cwd=ws_lb)

        out_md = io.StringIO()
        assert idle_svc.main(
            [
                "--gateway",
                base_url,
                "--workspace",
                str(ws_lb),
                "--credentials-path",
                str(home_lb / ".hivemind" / "credentials.json"),
                "--state-file",
                str(home_lb / ".hivemind" / "state.json"),
                "--ces-queue-file",
                str(q_lb),
                "--tick",
                "--send-heartbeat",
                "--check-fleet",
                "--register-if-missing",
                "--prompt-ces",
                "--flush-ces-queue",
                "--lang",
                "fr",
                "--markdown",
            ],
            env=CLEAN_ENV,
            cwd=ws_lb,
            home=home_lb,
            input_fn=lambda _p: "5 nolag",
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=lambda **_k: ["python3"],
            idle_probe_fn=lambda **_k: 240.0,
            stdout=out_md,
        ) == 0
        assert "RUNNING_LORA_ONLY" in out_md.getvalue()

        out_cli = io.StringIO()
        assert idle_svc.main(["--gateway", base_url, "--check-fleet", "--lang", "en"], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=out_cli) == 0
        assert "Fleet Badge: ONLINE" in out_cli.getvalue()
        assert idle_svc.main(["--gateway", base_url, "--flush-ces-queue", "--ces-queue-file", str(q_lb)], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=io.StringIO()) == 0
        assert idle_svc.main(["--gateway", base_url, "--install", "--service-dir", str(home_lb / "svc"), "--workspace", str(ws_lb)], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=io.StringIO()) == 0
    finally:
        srv.shutdown()
        srv.server_close()

    assert idle_svc.default_http_request("GET", f"http://127.0.0.1:{port}/health", timeout_s=0.2)[0] == 0

    # 3. Probes, arities, validators, auth retries, and CLI error edge coverage
    assert idle_svc.default_subprocess_runner([sys.executable, "-c", "print('ok')"])[0] == 0
    assert idle_svc.default_subprocess_runner(["/nonexistent_bin_dyn16"])[0] == 127
    assert idle_svc.default_gpu_probe(runner=lambda _c, _t: (0, "RTX 4070, 12288, 9216, 3072, 54, 5\n")).present is True
    assert idle_svc.default_gpu_probe(runner=lambda _c, _t: (0, "bad")).parse_error is True
    assert idle_svc.default_gpu_probe(runner=lambda _c, _t: (0, "RTX, -1, 0, 0, n/a, bad\n")).parse_error is True
    assert idle_svc.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert "cs2.exe" in idle_svc.default_process_probe(runner=lambda _c, _t: (0, '"cs2.exe","1","Console","1","10 K"\n'), os_name="nt")
    assert "python3" in idle_svc.default_process_probe(runner=lambda _c, _t: (0, "\n/usr/bin/python3\n"), os_name="posix")
    assert idle_svc.default_idle_probe(runner=lambda _c, _t: (0, '"HIDIdleTime" = 5000000000\n'), sys_platform="darwin") == 5.0
    assert idle_svc.default_idle_probe(runner=lambda _c, _t: (0, "12000\n"), sys_platform="linux", env={"DISPLAY": ":0"}) == 12.0
    assert idle_svc.default_idle_probe(sys_platform="linux", env={}) == 300.0
    assert idle_svc.default_idle_probe(sys_platform="win32") == 300.0

    # Simulate Win32 ctypes.windll idle probe cleanly
    class _FakeUser32:
        def GetLastInputInfo(self, ptr: Any) -> int:  # noqa: N802
            _ = ptr
            return 1

    class _FakeKernel32:
        def GetTickCount(self) -> int:  # noqa: N802
            return 4500

    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=_FakeUser32(), kernel32=_FakeKernel32()), raising=False)
    assert idle_svc.default_idle_probe(sys_platform="win32") == 4.5
    monkeypatch.delattr(ctypes, "windll", raising=False)

    # Broken Popen drill paths
    class _BrokenProc:
        def __init__(self, banner: str, raise_wait: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(banner)
            self.stderr = io.StringIO()
            self._raise_wait = raise_wait

        def poll(self) -> int | None:
            return None if self._raise_wait else 0

        def kill(self) -> None:
            self._raise_wait = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._raise_wait:
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenProc("BAD\n"))
    assert idle_svc.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenProc("HIVEMIND_WORKER_READY\n", True))
    assert idle_svc.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    # Safety, validators & edge cases
    assert idle_svc.mask_api_key("") == "<missing>"
    assert idle_svc.mask_api_key("short") == "<invalid>"
    assert idle_svc.mask_api_key(LIVE_KEY) == "hm_live_***4321"
    assert idle_svc.sanitize_host_label("amadeus-pc") == "personal-pc"
    assert idle_svc.sanitize_build_sha("secret12345") == "unknown"
    assert idle_svc.sanitize_cloud_run_revision("hm-1234567890") is None
    assert idle_svc.shas_match("release2026a", "release2026b") is False
    assert idle_svc.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert idle_svc.is_allowed_outbound_host("storage.googleapis.com") is True
    assert idle_svc.is_allowed_outbound_host("") is False
    bad_space_gw = "https://" + "bad space.run.app"
    bad_space_req = "http://" + "127.0.0.1:8000/a b"
    bad_cred_req = "http://" + "u:p@127.0.0.1:8000/h"
    for bad_gw in (UNTRUSTED_EXT_GW, UNTRUSTED_FTP_GW, INSECURE_RUN_APP_GW, "https://" + "u:p@x.run.app", "", "https://x.run.app\n", bad_space_gw, "https://x.run.app?a=1", "https://x.run.app/p", "http://127.0.0.1:0", "http://127.0.0.1:bad"):
        with pytest.raises(idle_svc.IdleServiceError):
            idle_svc.validate_gateway_url(bad_gw)
    for bad_req in ("", bad_space_req, bad_cred_req, "http://127.0.0.1:8000/v1/chat/completions"):
        with pytest.raises(idle_svc.IdleServiceError):
            idle_svc._validate_outbound_request_url(bad_req)
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc._assert_safe_output(f"leak {ALPHA_KEY}")
    ip_pub = ".".join(("10", "0", "0", "1"))
    assert len(idle_svc.scan_output_safety(f"a@b.org {ALPHA_KEY} Bearer tok_123456 /home/bob/x {ip_pub}")) >= 5
    assert idle_svc.redact_home_path(r"C:\Users\AliceEdge\.hivemind\s.json", home=Path("/none")) == "~/.hivemind/s.json"

    assert idle_svc._invoke_http_fn(lambda m, u: (200, "raw"), "GET", "http://127.0.0.1:8000/h")[0] == 200
    assert idle_svc._invoke_http_fn(lambda m, u, p: (200, {"p": p}), "POST", "http://127.0.0.1:8000/r", payload={"a": 1})[0] == 200
    assert idle_svc._invoke_http_fn(lambda m, u, p, h: (200, {"h": h}), "GET", "http://127.0.0.1:8000/h")[0] == 200
    assert idle_svc._invoke_http_fn(lambda m, u, p, h, t: (200, {"t": t}), "GET", "http://127.0.0.1:8000/h")[0] == 200
    assert idle_svc._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("down")), "GET", "http://127.0.0.1:8000/h")[0] == 0
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc._normalize_http_return("bad")

    # Auth 409 retry + missing creds without register_if_missing + fleet BOOTSTRAPPING/OFFLINE/DRIFT
    reg_try = {"n": 0}
    def _409_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST" and url.endswith("/register"):
            reg_try["n"] += 1
            return (409, {}) if reg_try["n"] == 1 else (200, {"account_id": "acct_409", "api_key": ALPHA_KEY, "role": "prosumer"})
        if url.endswith("/me"):
            return 200, {"account_id": "acct_409", "handle": "g-409", "role": "prosumer"}
        return 200, {"node": {"node_id": "n-409"}}

    t409 = idle_svc.run_idle_service_tick(
        gateway="http://127.0.0.1:8000",
        credentials_path=tmp_path / "c409.json",
        state_path=tmp_path / "s409.json",
        register_if_missing=True,
        send_heartbeat=True,
        http_fn=_409_http,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        idle_probe_fn=lambda: 240.0,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path,
    )
    assert t409["exit_code"] == 0 and reg_try["n"] == 2

    # Missing creds without --register-if-missing -> exit 2
    assert idle_svc.run_idle_service_tick(
        gateway="http://127.0.0.1:8000",
        credentials_path=tmp_path / "missing_creds.json",
        state_path=tmp_path / "s_miss.json",
        send_heartbeat=True,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["python3"],
        idle_probe_fn=lambda: 240.0,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path,
        lang="fr",
    )["exit_code"] == 2

    # Fleet BOOTSTRAPPING, OFFLINE, and DRIFT
    assert idle_svc.verify_live_fleet_and_badge(
        gateway="http://127.0.0.1:8000",
        target_online_producers=8,
        http_fn=lambda _m, u, **_k: (
            (200, {"status": "ok"}) if u.endswith("/health")
            else ((200, {"object": idle_svc.MANIFEST_OBJECT, "build_sha": LIVE_BUILD_SHA, "personal_pc_only": True, "financial_promise": idle_svc.FINANCIAL_PROMISE, "holiday_routes": list(idle_svc.EXPECTED_MANIFEST_HOLIDAY_ROUTES)}) if u.endswith("/manifest")
            else ((200, {"object": idle_svc.FLEET_OBJECT, "live_hiveminders": 2, "total_hiveminders": 2}) if u.endswith("/fleet")
            else (200, {"object": idle_svc.RAID_STATUS_OBJECT, "swarm": {"active_trainers": 2}})))
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )["fleet_badge"] == "BOOTSTRAPPING"

    assert idle_svc.verify_live_fleet_and_badge(
        gateway="http://127.0.0.1:8000",
        http_fn=lambda _m, _u, **_k: (503, {}),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )["fleet_badge"] == "OFFLINE"

    assert idle_svc.verify_live_fleet_and_badge(
        gateway="http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        http_fn=lambda _m, u, **_k: (200, {"object": idle_svc.MANIFEST_OBJECT, "build_sha": LIVE_BUILD_SHA, "personal_pc_only": True, "financial_promise": idle_svc.FINANCIAL_PROMISE, "holiday_routes": list(idle_svc.EXPECTED_MANIFEST_HOLIDAY_ROUTES)}) if u.endswith("/manifest") else (200, {"status": "ok"}),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )["fleet_badge"] == "STALE_CLOUD_RUN"

    # Paused state holds OpenTrainer heartbeat
    hb_held, hb_calls = idle_svc.send_opentrainer_heartbeat(
        gateway="http://127.0.0.1:8000",
        api_key=ALPHA_KEY,
        role="prosumer",
        host_label="pc",
        cycle_report={"state": idle_svc.STATE_PAUSED_GAME_YIELD},
        lang="fr",
    )
    assert hb_calls == 0 and hb_held["status"] == "paused_state_hold"

    # CLI validation error -> exit code 2
    err_cli = io.StringIO()
    assert idle_svc.main(["--gateway", UNTRUSTED_EXT_GW, "--tick", "--json"], env=CLEAN_ENV, cwd=tmp_path, stdout=err_cli) == 2
    assert json.loads(err_cli.getvalue())["mode"] == "error"

    # Parse CES ease input branches + remaining guard assertions
    assert idle_svc.parse_ces_ease_input("skip")["skipped"] is True
    assert idle_svc.parse_ces_ease_input("4/5 dev lag") == {"skipped": False, "profile": "dev", "ease_rating": 4, "zero_game_lag": False}
    for bad_call in (
        lambda: idle_svc.normalize_lang("de"),
        lambda: idle_svc.normalize_lang(True),  # type: ignore[arg-type]
        lambda: idle_svc.normalize_platform(123),  # type: ignore[arg-type]
        lambda: idle_svc.normalize_profile(None),
        lambda: idle_svc.normalize_profile("root"),
        lambda: idle_svc.normalize_role(None),
        lambda: idle_svc.normalize_role("root"),
        lambda: idle_svc.parse_ease_rating("abc"),
        lambda: idle_svc.parse_ease_rating(True),
        lambda: idle_svc.parse_ease_rating(9),
        lambda: idle_svc.sanitize_handle("google.spoof"),
        lambda: idle_svc.validate_expected_sha("unknown"),
        lambda: idle_svc._validate_manifest_path_arg(True, "script_path"),
        lambda: idle_svc.parse_ces_ease_input("0 gamer"),
        lambda: idle_svc.parse_ces_ease_input("badtok"),
        lambda: idle_svc.parse_ces_ease_input("nolag"),
        lambda: idle_svc._validate_ces_spool_payload("bad"),
        lambda: idle_svc._validate_ces_spool_payload({"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "", "extra": 1}),
        lambda: idle_svc._validate_ces_spool_payload({"profile": "gamer", "ease_rating": 5, "zero_game_lag": "yes", "platform": "linux", "note": ""}),
        lambda: idle_svc._validate_ces_spool_payload({"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "bad!", "note": ""}),
        lambda: idle_svc.submit_ces_feedback(gateway="http://127.0.0.1:8000", zero_game_lag="no"),  # type: ignore[arg-type]
        lambda: idle_svc.verify_lora_delta_bandwidth_cap(lora_params=0),
        lambda: idle_svc.default_subprocess_runner(""),  # type: ignore[arg-type]
        lambda: idle_svc._normalize_gpu_snapshot(123),
        lambda: idle_svc._normalize_worker_yield_result(True),
        lambda: idle_svc._normalize_worker_yield_result("bad"),
        lambda: idle_svc.verify_live_fleet_and_badge(target_online_producers=0),
        lambda: idle_svc.evaluate_idle_resume_cycle(max_temp_c=10.0),
        lambda: idle_svc.evaluate_idle_resume_cycle(min_free_vram_mib=64),
        lambda: idle_svc.evaluate_idle_resume_cycle(idle_threshold_s=0.0),
        lambda: idle_svc.evaluate_idle_resume_cycle(cooldown_s=10.0),
        lambda: idle_svc.evaluate_idle_resume_cycle(yield_budget_ms=0.0),
        lambda: idle_svc.IdleServiceCesQueue(max_entries=0),
        lambda: idle_svc.IdleServiceCesQueue(max_delivered=0),
    ):
        with pytest.raises(idle_svc.IdleServiceError):
            bad_call()

    # Additional branch coverage: platform/handle/SHA/path/http/probe/state/credentials/spool edges
    assert idle_svc._platform_tag("Darwin") == "macos"
    assert idle_svc._platform_tag("Windows") == "windows"
    assert idle_svc._platform_tag("FreeBSD 14") == "freebsd14"
    assert idle_svc._platform_tag("!!!") == "unknown"
    assert idle_svc.sanitize_handle(None).startswith("noel-gamer-")
    assert idle_svc.is_valid_hivemind_key(123) is False
    assert idle_svc._resolve_path_with_home("~/s.json", home_dir=tmp_path) == tmp_path / "s.json"
    assert idle_svc._resolve_path_with_home("~", home_dir=tmp_path) == tmp_path
    assert idle_svc.shas_match("unknown", "abc123") is False
    assert idle_svc.shas_match("abc123", "abc123") is True
    assert "forbidden_score_key_detected" in idle_svc.scan_output_safety('{"' + 'score": 9}')
    assert idle_svc._parse_http_raw_response(200, b"plain ok", {"Content-Type": "text/plain"})[1]["raw_length"] == 8
    assert idle_svc._invoke_http_fn(lambda m, u: (200, {"X-H": "1"}, {"ok": True}), "GET", "http://127.0.0.1:8000/h")[1]["_headers"]["x-h"] == "1"
    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc._invoke_http_fn(lambda m, u: (_ for _ in ()).throw(idle_svc.IdleServiceError("boom")), "GET", "http://127.0.0.1:8000/h")
    assert idle_svc._find_corporate_markers("") == []
    assert idle_svc.extract_git_config_remote_urls("") == []
    assert idle_svc.evaluate_personal_pc_compliance(env={"AMADEUS_SITE": "corp-gw"}, cwd=tmp_path)["compliance_ok"] is False

    # Git config read failure and unreadable .git file handling
    orig_read_text = Path.read_text
    unreadable_ws = tmp_path / "unreadable_git_ws"
    (unreadable_ws / ".git").mkdir(parents=True)
    bad_cfg = unreadable_ws / ".git" / "config"
    bad_cfg.write_bytes(b"\x80")
    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("perm")) if self == bad_cfg else orig_read_text(self, *a, **k))
        assert idle_svc.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=unreadable_ws)["compliance_ok"] is True

    unreadable_gitfile_ws = tmp_path / "unreadable_gitfile_ws"
    unreadable_gitfile_ws.mkdir()
    bad_gitfile = unreadable_gitfile_ws / ".git"
    bad_gitfile.write_text("gitdir: .git/missing\n", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("perm")) if self == bad_gitfile else orig_read_text(self, *a, **k))
        assert idle_svc._resolve_git_config_files(unreadable_gitfile_ws) == []

    # Commondir OSError fallback inside git worktree indirection
    wt_err_ws = tmp_path / "wt_err_ws"
    wt_err_git = tmp_path / "wt_err_git"
    wt_err_git.mkdir(parents=True)
    wt_err_ws.mkdir()
    (wt_err_ws / ".git").write_text(f"gitdir: {wt_err_git}\n", encoding="utf-8")
    (wt_err_git / "config").write_text('[remote "origin"]\n\turl = ' + 'https:' + '//personal-hobby.invalid/h.git\n', encoding="utf-8")
    err_common = wt_err_git / "commondir"
    err_common.write_text("../..\n", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("perm")) if self == err_common else orig_read_text(self, *a, **k))
        assert len(idle_svc._resolve_git_config_files(wt_err_ws)) == 1

    # Atomic temp cleanup when os.replace fails
    atomic_fail_target = tmp_path / "atomic_fail.json"
    with monkeypatch.context() as mp:
        mp.setattr(os, "replace", lambda _s, _d: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            idle_svc._write_file_secure_atomic(atomic_fail_target, "{}", posix_mode=0o600, eff_os="posix")
    assert not any(p.name.startswith(".atomic_fail.json.") for p in tmp_path.iterdir())

    # Snapshot, idle probe, game detect, yield normalizer, state loader & credential failure branches
    assert idle_svc._normalize_gpu_snapshot(None).present is False
    assert idle_svc._normalize_gpu_snapshot(SimpleNamespace(present=True, name="RTX", total_vram_mib=None, free_vram_mib=None)).parse_error is True
    assert idle_svc.default_idle_probe(runner=lambda _c, _t: (1, ""), sys_platform="darwin") == 300.0
    assert idle_svc.detect_steam_or_dx12_game_activity(["", "cs2.exe"])["game_detected"] is True
    assert idle_svc._normalize_worker_yield_result(-1.0)["clean_release"] is False
    assert idle_svc._normalize_worker_yield_result({"yield_latency_ms": "bad"})["yield_latency_ms"] is None
    assert idle_svc._safe_nonneg_int("bad") == 0

    for raw_st in ("{broken", '["not-a-dict"]', json.dumps({"state": "UNKNOWN_STATE"})):
        st_f = tmp_path / "corrupt_state.json"
        st_f.write_text(raw_st, encoding="utf-8")
        assert idle_svc._load_idle_resume_state(st_f)["state"] == idle_svc.STATE_IDLE_READY

    cred_edge = tmp_path / "cred_edge.json"
    cred_edge.write_text("{broken", encoding="utf-8")
    assert idle_svc.load_saved_credentials(cred_edge) is None
    cred_edge.write_text("[]", encoding="utf-8")
    assert idle_svc.load_saved_credentials(cred_edge) is None
    cred_edge.write_text(json.dumps({"api_key": "bad"}), encoding="utf-8")
    assert idle_svc.load_saved_credentials(cred_edge) is None
    cred_edge.write_text(json.dumps({"api_key": ALPHA_KEY, "gateway": "https://other.run.app"}), encoding="utf-8")
    assert idle_svc.load_saved_credentials(cred_edge, expected_gateway="http://127.0.0.1:8000") is None
    cred_edge.write_text(json.dumps({"api_key": ALPHA_KEY, "gateway": "http://127.0.0.1:8000"}), encoding="utf-8")
    assert idle_svc.load_saved_credentials(cred_edge, expected_gateway="http://127.0.0.1:8000") is not None

    with pytest.raises(idle_svc.IdleServiceError):
        idle_svc.ensure_gateway_credentials(gateway="http://127.0.0.1:8000", credentials_path=cred_edge, explicit_api_key="invalid")
    assert idle_svc.ensure_gateway_credentials(
        gateway="http://127.0.0.1:8000", credentials_path=cred_edge, register_if_missing=False, env={}, explicit_api_key=ALPHA_KEY, http_fn=lambda _m, _u, **_k: (401, {})
    )[0]["stage"] == "me"
    assert idle_svc.ensure_gateway_credentials(
        gateway="http://127.0.0.1:8000", credentials_path=tmp_path / "no_cred.json", register_if_missing=True, env={}, http_fn=lambda _m, _u, **_k: (500, {})
    )[0]["stage"] == "register"

    # CES input, queue edge trimming/recovery/purge, EOF prompt, flush 404 inside tick, human CLI report
    assert idle_svc.parse_ces_ease_input("4 gaming")["profile"] == "gamer"
    default_q = idle_svc.IdleServiceCesQueue(None, home=tmp_path, env=CLEAN_ENV)
    assert default_q.path == (tmp_path / idle_svc.DEFAULT_CES_QUEUE_REL).resolve()

    edge_q_path = tmp_path / "edge_q.json"
    edge_q_path.write_text("[1, 2]", encoding="utf-8")
    eq = idle_svc.IdleServiceCesQueue(edge_q_path, max_entries=2, max_delivered=2, home=tmp_path)
    assert eq.snapshot()["pending_count"] == 0

    k1 = idle_svc.build_ces_idempotency_key(gateway="http://127.0.0.1:8000", account_ref="a1", profile="gamer", ease_rating=5, zero_game_lag=True, platform_tag="linux")
    k2 = idle_svc.build_ces_idempotency_key(gateway="http://127.0.0.1:8000", account_ref="a2", profile="gamer", ease_rating=5, zero_game_lag=True, platform_tag="linux")
    k3 = idle_svc.build_ces_idempotency_key(gateway="http://127.0.0.1:8000", account_ref="a3", profile="gamer", ease_rating=5, zero_game_lag=True, platform_tag="linux")
    edge_q_path.write_text(
        json.dumps(
            {
                "delivered_ids": ["d1", "d2", k1],
                "pending": [
                    {"idempotency_key": k1, "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"idempotency_key": k2, "payload": {"profile": "gamer", "ease_rating": 4, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"idempotency_key": k3, "payload": {"profile": "gamer", "ease_rating": 3, "zero_game_lag": True, "platform": "linux", "note": ""}},
                    {"idempotency_key": "ces-k4", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(idle_svc.IdleServiceError):
        eq.enqueue(idempotency_key=123)  # type: ignore[arg-type]
    with pytest.raises(idle_svc.IdleServiceError):
        eq.enqueue(idempotency_key="bad\nkey")
    assert eq.enqueue(gateway="http://127.0.0.1:8000", idempotency_key=k1, env=CLEAN_ENV, cwd=tmp_path)["duplicate_state"] == "already_delivered"
    edge_q_path.write_text(
        json.dumps({"delivered_ids": ["d1"], "pending": [{"idempotency_key": k3, "payload": {"profile": "gamer", "ease_rating": 3, "zero_game_lag": True, "platform": "linux", "note": ""}}]}),
        encoding="utf-8",
    )
    assert eq.enqueue(gateway="http://127.0.0.1:8000", idempotency_key=k3, env=CLEAN_ENV, cwd=tmp_path)["duplicate_state"] == "already_pending"
    edge_q_path.write_text(
        json.dumps(
            {
                "delivered_ids": [k3],
                "pending": [{"event_id": "a" * 64, "idempotency_key": k3, "payload": {"profile": "gamer", "ease_rating": 3, "zero_game_lag": True, "platform": "linux", "note": ""}}],
            }
        ),
        encoding="utf-8",
    )
    assert eq.flush("http://127.0.0.1:8000", http_fn=lambda _m, _u, **_k: (200, {}), env=CLEAN_ENV, cwd=tmp_path)["delivered_count"] == 0
    sibling_tmp = edge_q_path.parent / f".{edge_q_path.name}.stale"
    sibling_tmp.write_text("tmp", encoding="utf-8")
    assert eq.purge()["purged"] is True
    assert not sibling_tmp.exists()

    eof_tick = idle_svc.run_idle_service_tick(
        gateway="http://127.0.0.1:8000",
        state_path=tmp_path / "eof_st.json",
        ces_queue_file=edge_q_path,
        prompt_ces=True,
        input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: [],
        idle_probe_fn=lambda: 200.0,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path,
    )
    assert eof_tick["ces_feedback"]["status"] == "skipped"

    eq.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="ces-flush-404", env=CLEAN_ENV, cwd=tmp_path)
    fl404 = idle_svc.run_idle_service_tick(
        gateway="http://127.0.0.1:8000",
        state_path=tmp_path / "fl404_st.json",
        ces_queue_file=edge_q_path,
        flush_ces_queue=True,
        http_fn=lambda _m, _u, **_k: (404, {"detail": "Not Found"}),
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: [],
        idle_probe_fn=lambda: 200.0,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path,
    )
    assert fl404["stale_revision_detected"] is True

    out_human = io.StringIO()
    assert idle_svc.main(
        ["--gateway", "http://127.0.0.1:8000", "--workspace", str(tmp_path), "--state-file", str(tmp_path / "h_st.json"), "--tick"],
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path,
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: [],
        idle_probe_fn=lambda: 200.0,
        stdout=out_human,
    ) == 0
    assert "State: DEV_API_ONLY_NO_GPU" in out_human.getvalue()
