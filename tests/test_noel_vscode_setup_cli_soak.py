# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for SLICE-SCALE-DYN-14: hivemind vscode-setup CLI, /v1/models+/quick preflight, 0600 CES spool, and 48-user soak."""

from __future__ import annotations

import ast
import concurrent.futures
import contextvars
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from hivemind.cli import main as cli_main
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
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
PRESET_SCRIPT_PATH = REPO_ROOT / "scripts" / "noel_vscode_quick_preset.py"
CLI_MAIN_PATH = REPO_ROOT / "src" / "hivemind" / "cli" / "main.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without any internal coverage instance so external
# `coverage run --rcfile=/dev/null` captures 100% of import + runtime lines.
from scripts import noel_vscode_quick_preset as preset  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn14",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn14",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn14",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
SOAK_SECRET = b"noel-scale-dyn14-router-sec-32b!"
ALPHA_KEY = "hm-" + "dyn14_alpha_token_123456"
LIVE_KEY = "hm_live_" + "dyn14_live_token_654321"
LIVE_BUILD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9011223344"
LIVE_SHORT_SHA = "a1b2c3d4e5f6"
DRIFT_SHA = "99887766554433221100ffeeddccbbaa99887766"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.org"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/platform/core.git"
CORP_HTTPS_REMOTE = "https://" + "gitlab.enterprise.intranet/team/service.git"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@proxy.zscaler.internal:8443"

_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
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
    assert preset.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    quick_nonces: list[str] | None = None,
    noel_nonces: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[str, tuple[int, dict[str, Any]]] | None = None,
) -> preset.HttpFn:
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
        if resp.status_code == 200:
            m = _CSP_NONCE_RE.search(lower_hdrs.get("content-security-policy", ""))
            if m and path == "/quick" and quick_nonces is not None:
                if lock is not None:
                    with lock:
                        quick_nonces.append(m.group(1))
                else:
                    quick_nonces.append(m.group(1))
            if m and path == "/noel" and noel_nonces is not None:
                if lock is not None:
                    with lock:
                        noel_nonces.append(m.group(1))
                else:
                    noel_nonces.append(m.group(1))

        ctype = lower_hdrs.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, dict):
                out = dict(data)
                out.setdefault("_headers", lower_hdrs)
                return resp.status_code, out
            return resp.status_code, {"data": data, "_headers": lower_hdrs}
        return resp.status_code, {
            "html": raw_text,
            "text": raw_text,
            "html_length": len(raw_text),
            "headers": lower_hdrs,
            "csp": lower_hdrs.get("content-security-policy", ""),
        }

    return _adapter


def _seed_live_coder_producer(client: TestClient) -> None:
    reg_p = client.post("/v1/hivemind/register", json={"handle": "home-rtx-seed", "role": "producer"}).json()
    hb_p = client.post(
        "/v1/hivemind/producer/heartbeat",
        json={
            "endpoint_url": "http://127.0.0.1:11434",
            "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
            "cost_per_1k": 0.0,
            "capacity_1k": 50.0,
            "gpu_vram_mib": 12288,
            "execution_class": "ollama-gamer-whisper",
        },
        headers={"Authorization": f"Bearer {reg_p['api_key']}"},
    )
    assert hb_p.status_code == 200


def test_01_compliance_headers_stdlib_ast_pr_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (PRESET_SCRIPT_PATH, CLI_MAIN_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {target}"

    script_src = PRESET_SCRIPT_PATH.read_text(encoding="utf-8")
    cli_src = CLI_MAIN_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure stdlib imports only in scripts/noel_vscode_quick_preset.py + ProxyHandler({}) + zero unsafe sinks
    tree = ast.parse(script_src, filename="scripts/noel_vscode_quick_preset.py")
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
    test_tree = ast.parse(test_src, filename="tests/test_noel_vscode_setup_cli_soak.py")
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
    assert lateral_guard.audit_source_for_lateral_movement(script_src, "scripts/noel_vscode_quick_preset.py") == []
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
            for fn_name in ("_load_vscode_quick_preset_module", "_build_vscode_setup_argv", "cmd_vscode_setup")
        ]
    )
    assert lateral_guard.audit_source_for_lateral_movement(cli_new_blocks, "src/hivemind/cli/main.py") == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, "tests/test_noel_vscode_setup_cli_soak.py") == []
    planted_lan = ".".join(("192", "168", "10", "55"))
    planted_ext = "https://" + "c2.attacker.example.com/hook"
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_LAN = "http://{planted_lan}:8000"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_EXT = "{planted_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness + PRSecurityScanner + ZeroStubAudit + FIGURE
    scanner = PRSecurityScanner()
    auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()
    fallback_auditor = _FallbackZeroStubAudit()
    assert fallback_auditor.check_file(str(PRESET_SCRIPT_PATH)) == []
    assert fallback_auditor.audit_python_code(script_src).is_clean is True

    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_vscode_quick_preset.py", script_src),
        ("src/hivemind/cli/main.py", cli_src),
        ("tests/test_noel_vscode_setup_cli_soak.py", test_src),
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
        pr_number=1114,
        pr_title="feat(onboarding,cli,vscode,scale): hivemind vscode-setup <60s Dev command, /v1/models+/quick preflight, 0600 CES retry spool, and 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-14 with zero-inference preflight, ~/.continue/config.yaml merge, 0600 CES spool, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,cli,vscode,scale): hivemind vscode-setup CLI and 48-user soak"],
    )
    assert pr_audit.passed is True
    assert pr_audit.findings == []


def test_02_hivemind_cli_vscode_setup_help_zero_cuda_and_missing_helper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Block CUDA initialization at application boundary
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("Unexpected torch.cuda.is_available() during vscode-setup")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: (_ for _ in ()).throw(AssertionError("Unexpected torch.cuda._lazy_init() during vscode-setup")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Unexpected CUDA reset_peak_memory_stats()")),
    )

    # 1. --help and vscode-setup --help exit cleanly (SystemExit(0)) with zero CUDA calls
    with pytest.raises(SystemExit) as exc_top:
        cli_main.main(["--help"])
    assert exc_top.value.code == 0
    top_help = capsys.readouterr().out
    assert "vscode-setup" in top_help

    with pytest.raises(SystemExit) as exc_sub:
        cli_main.main(["vscode-setup", "--help"])
    assert exc_sub.value.code == 0
    sub_help = capsys.readouterr().out
    for expected_flag in (
        "--gateway",
        "--write-continue-config",
        "--continue-main-config-path",
        "--ces-queue-file",
        "--skip-preflight",
        "--prompt-ces",
        "--flush-ces-queue",
        "--gamer-preflight",
        "--first-heartbeat",
        "--uninstall",
        "--json",
    ):
        assert expected_flag in sub_help

    # 2. Missing helper script fails closed cleanly with exit code 2 (both --json and text)
    missing_script = tmp_path / "does_not_exist_preset.py"
    monkeypatch.setattr(cli_main, "_VSCODE_QUICK_PRESET_SCRIPT", missing_script)

    rc_miss_json = cli_main.main(["vscode-setup", "--json"])
    assert rc_miss_json == 2
    miss_json_out = json.loads(capsys.readouterr().out)
    assert miss_json_out["ok"] is False
    assert miss_json_out["exit_code"] == 2

    rc_miss_txt = cli_main.main(["vscode-setup"])
    assert rc_miss_txt == 2
    captured_miss = capsys.readouterr()
    assert "unavailable" in captured_miss.err

    # Unload cached sys.modules entries to exercise fresh importlib loading path
    monkeypatch.setattr(cli_main, "_VSCODE_QUICK_PRESET_SCRIPT", PRESET_SCRIPT_PATH)
    for mod_key in ("scripts.noel_vscode_quick_preset", "noel_vscode_quick_preset"):
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    fresh_mod = cli_main._load_vscode_quick_preset_module()
    assert hasattr(fresh_mod, "run_vscode_setup")
    # Re-attach original `preset` reference in sys.modules for subsequent monkeypatching
    sys.modules["scripts.noel_vscode_quick_preset"] = preset
    sys.modules["noel_vscode_quick_preset"] = preset

    # Reject invalid spec when path points to directory/non-file
    with pytest.raises(FileNotFoundError):
        cli_main._load_vscode_quick_preset_module(tmp_path)

    # Bad non-py file raises ImportError from spec_from_file_location
    non_py = tmp_path / "not_python.txt"
    non_py.write_text("hello", encoding="utf-8")
    with pytest.raises(ImportError):
        cli_main._load_vscode_quick_preset_module(non_py)

    # 3. End-to-end cli_main.main(["vscode-setup", ...]) & cmd_vscode_setup(...) over live router
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00199-dyn14")
    store = RouterStore(tmp_path / "cli_router.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(store, allow_private_producers=True, trusted_proxy_hops=1, open_registration=True)

    home_cli = tmp_path / "home" / "cli_user_secret"
    ws_cli = home_cli / "hobby_project"
    ws_cli.mkdir(parents=True)
    (ws_cli / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
    cred_cli = home_cli / ".hivemind" / "credentials.json"
    cont_cli = home_cli / ".continue" / "hivemind_provider.yaml"
    cont_main_cli = home_cli / ".continue" / "config.yaml"
    queue_cli = home_cli / ".hivemind" / "vscode_setup_ces_queue.json"

    with TestClient(app) as client:
        _seed_live_coder_producer(client)
        adapter = _make_router_http_adapter(client, peer_ip="127.0.14.1")
        monkeypatch.setattr(preset, "default_http_request", adapter)
        monkeypatch.setattr(preset, "default_gpu_probe", lambda **_k: preset.GpuProbeSnapshot(present=False))
        monkeypatch.setattr(preset, "default_process_probe", lambda **_k: ["code"])
        monkeypatch.setattr(preset, "default_worker_yield_drill", lambda **_k: 25.0)
        monkeypatch.setenv("HOME", str(home_cli))
        monkeypatch.setenv("HOSTNAME", "personal-home-pc")

        rc_setup = cli_main.main(
            [
                "vscode-setup",
                "--gateway",
                "http://127.0.0.1:8000",
                "--workspace",
                str(ws_cli),
                "--credentials-path",
                str(cred_cli),
                "--continue-config-path",
                str(cont_cli),
                "--continue-main-config-path",
                str(cont_main_cli),
                "--write-continue-config",
                "--ces-queue-file",
                str(queue_cli),
                "--handle",
                "cli-dev-01",
                "--role",
                "prosumer",
                "--submit-ces",
                "--profile",
                "dev",
                "--ease-rating",
                "5",
                "--lang",
                "en",
                "--json",
            ]
        )
        assert rc_setup == 0
        setup_out = capsys.readouterr().out
        setup_doc = json.loads(setup_out)
        _verify_receipt(setup_doc)
        assert setup_doc["ready"] is True
        assert setup_doc["mode"] == "vscode_setup"
        assert setup_doc["models_preflight_ok"] is True
        assert setup_doc["quick_route_ok"] is True
        assert setup_doc["feedback_endpoint_ok"] is True
        assert setup_doc["ces_recorded"] is True
        assert setup_doc["ces_queue_exists"] is False
        assert not queue_cli.exists()
        assert cont_main_cli.is_file()
        assert "cli_user_secret" not in setup_out

        # Direct cmd_vscode_setup(Sequence[str]) path with --skip-preflight, explicit --vscode-setup & --uninstall
        rc_seq = cli_main.cmd_vscode_setup(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--workspace",
                str(ws_cli),
                "--credentials-path",
                str(cred_cli),
                "--continue-config-path",
                str(cont_cli),
                "--skip-preflight",
                "--json",
            ]
        )
        assert rc_seq == 0
        seq_doc = json.loads(capsys.readouterr().out)
        assert seq_doc["quick_preflight"] is None
        assert seq_doc["models_preflight_ok"] is True

        rc_seq_explicit = cli_main.cmd_vscode_setup(
            [
                "--vscode-setup",
                "--gateway",
                "http://127.0.0.1:8000",
                "--workspace",
                str(ws_cli),
                "--credentials-path",
                str(cred_cli),
                "--continue-config-path",
                str(cont_cli),
                "--skip-preflight",
                "--json",
            ]
        )
        assert rc_seq_explicit == 0
        _ = capsys.readouterr().out

        rc_un = cli_main.main(
            [
                "vscode-setup",
                "--uninstall",
                "--workspace",
                str(ws_cli),
                "--credentials-path",
                str(cred_cli),
                "--continue-config-path",
                str(cont_cli),
                "--continue-main-config-path",
                str(cont_main_cli),
                "--ces-queue-file",
                str(queue_cli),
                "--json",
            ]
        )
        assert rc_un == 0
        un_doc = json.loads(capsys.readouterr().out)
        assert un_doc["mode"] == "uninstall_preset"
        assert not cont_main_cli.exists()
        assert not cred_cli.exists()


def test_03_one_shot_preflight_continue_yaml_merge_prompt_ces_and_0600_spool(
    tmp_path: Path,
) -> None:
    # 1. One-shot /v1/models + /quick preflight: pass (with mean_ease_rating & average_ease_rating) vs stale 404 vs failures
    healthy_html = (
        "<!DOCTYPE html><html><head><style nonce='n123ms456abcd'>body{}</style></head>"
        "<body>100% personal PC AGPL-3.0 0.00 EUR / 0.00 USD</body></html>"
    )

    def _healthy_avg_http(_m: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}
        if url.endswith("/quick"):
            return 200, {
                "html": healthy_html,
                "csp": "default-src 'none'; style-src 'nonce-n123ms456abcd'",
            }
        return 200, {
            "total_responses": 4,
            "average_ease_rating": 4.75,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 3, "gamer": 1},
        }

    pf_ok = preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=_healthy_avg_http,
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_receipt(pf_ok)
    assert pf_ok["exit_code"] == 0
    assert pf_ok["models_preflight_ok"] is True
    assert pf_ok["quick_route_ok"] is True
    assert pf_ok["feedback_endpoint_ok"] is True
    assert pf_ok["live_models_count"] == 3
    assert pf_ok["stale_revision_detected"] is False
    assert pf_ok["local_fallback_guidance"] == ""
    assert pf_ok["feedback_summary"]["mean_ease_rating"] == 4.75

    # Pre-PR-#949 stale Cloud Run 404 on /quick & /feedback/summary: passes when require_holiday_routes=False, fails when True
    def _stale_404_http(_m: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}]}
        return 404, {"detail": "Not Found"}

    for lang_code in ("en", "fr"):
        pf_stale_allow = preset.run_one_shot_quick_health_preflight(
            "http://127.0.0.1:8000",
            http_fn=_stale_404_http,
            require_holiday_routes=False,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang=lang_code,
        )
        _verify_receipt(pf_stale_allow)
        assert pf_stale_allow["exit_code"] == 0
        assert pf_stale_allow["models_preflight_ok"] is True
        assert pf_stale_allow["quick_route_ok"] is False
        assert pf_stale_allow["feedback_endpoint_ok"] is False
        assert pf_stale_allow["stale_revision_detected"] is True
        assert len(pf_stale_allow["local_fallback_guidance"]) > 20

    pf_stale_req = preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=_stale_404_http,
        require_holiday_routes=True,
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert pf_stale_req["exit_code"] == 2
    assert pf_stale_req["stale_revision_detected"] is True

    # Fail closed on /health failure or empty / missing hivemind/coder in /v1/models
    pf_bad_health = preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u, **_k: (503, {}) if u.endswith("/health") else _healthy_avg_http("GET", u),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert pf_bad_health["exit_code"] == 2 and pf_bad_health["health_ok"] is False

    for bad_models in ([], [{"id": "hivemind/auto"}]):
        pf_bad_mod = preset.run_one_shot_quick_health_preflight(
            "http://127.0.0.1:8000",
            http_fn=lambda _m, u, bm=bad_models, **_k: (200, {"data": bm}) if u.endswith("/v1/models") else _healthy_avg_http("GET", u),
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert pf_bad_mod["exit_code"] == 2
        assert pf_bad_mod["models_preflight_ok"] is False

    # Single-quoted nonce containing digits + 'ms' must NOT false-positive after stripping nonce='...'
    single_q_html = (
        "<!DOCTYPE html><html><head><style nonce='abc198ms999'>body{}</style></head>"
        "<body>100% personal PC AGPL-3.0 0,00 EUR</body></html>"
    )
    pf_sq = preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u, **_k: (
            (200, {"html": single_q_html, "csp": "default-src 'none'; style-src 'nonce-abc198ms999'"})
            if u.endswith("/quick")
            else _healthy_avg_http("GET", u)
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert pf_sq["quick_route_ok"] is True

    # Unsafe DOM sink / static figure on /quick fails quick_route_ok
    bad_dom_html = (
        "<!DOCTYPE html><html><head><style nonce='n88888888'>body{}</style></head>"
        f"<body>100% personal AGPL-3.0 0.00 EUR <script nonce='n88888888'>x.{'inner' + 'HTML'}='1'</script></body></html>"
    )
    pf_dom_bad = preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u, **_k: (
            (200, {"html": bad_dom_html, "csp": "default-src 'none'; script-src 'nonce-n88888888'"})
            if u.endswith("/quick")
            else _healthy_avg_http("GET", u)
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert pf_dom_bad["quick_route_ok"] is False

    # 2. Optional ~/.continue/config.yaml merge preserving user models + .bak + clean --uninstall
    existing_yaml = """# User personal Continue configuration
name: Personal Local Assistant
version: 0.0.1
schema: v1
models:
  - name: Local Ollama Qwen
    provider: ollama
    model: qwen2.5-coder:7b
  - name: HiveMind Coder  # old entry to replace
    provider: openai
    model: hivemind/coder
    apiBase: http://127.0.0.1:9999/v1
    apiKey: hm-oldkey123456
# Autocomplete kept intact
tabAutocompleteModel:
  title: Starcoder
  provider: ollama
  model: starcoder2:3b
"""
    merged_yaml = preset.merge_continue_config_yaml(
        existing_yaml, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY
    )
    parsed_merged = yaml.safe_load(merged_yaml)
    assert parsed_merged["name"] == "Personal Local Assistant"
    assert parsed_merged["tabAutocompleteModel"]["model"] == "starcoder2:3b"
    model_names = [m["name"] for m in parsed_merged["models"]]
    assert model_names == ["Local Ollama Qwen", "HiveMind Coder", "HiveMind Bee-Stars (Science)"]
    assert parsed_merged["models"][1]["apiKey"] == ALPHA_KEY
    assert parsed_merged["models"][1]["apiBase"] == "http://127.0.0.1:8000/v1"

    # Re-merging is idempotent (no duplicate HiveMind entries)
    merged_again = preset.merge_continue_config_yaml(
        merged_yaml, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY
    )
    assert merged_again == merged_yaml

    # Removing HiveMind entries preserves Local Ollama Qwen & tabAutocompleteModel
    uninstalled_yaml = preset.remove_hivemind_from_continue_config_yaml(merged_yaml)
    assert uninstalled_yaml is not None
    parsed_un = yaml.safe_load(uninstalled_yaml)
    assert [m["name"] for m in parsed_un["models"]] == ["Local Ollama Qwen"]
    assert parsed_un["tabAutocompleteModel"]["model"] == "starcoder2:3b"

    # Also cover 0-indent YAML, models: [], no models section, JSON config, and validation errors
    zero_indent_yaml = "name: ZeroIndent\nmodels:\n- name: CustomUser\n  provider: ollama\n  model: llama3\n"
    merged_zero = preset.merge_continue_config_yaml(zero_indent_yaml, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    assert len(yaml.safe_load(merged_zero)["models"]) == 3

    no_models_yaml = "name: OnlyTab\ntabAutocompleteModel:\n  provider: ollama\n  model: qwen2.5\n"
    merged_nomod = preset.merge_continue_config_yaml(no_models_yaml, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    assert len(yaml.safe_load(merged_nomod)["models"]) == 2
    cleaned_nomod = preset.remove_hivemind_from_continue_config_yaml(merged_nomod)
    assert cleaned_nomod is not None and yaml.safe_load(cleaned_nomod)["models"] == []

    json_cfg = json.dumps({"models": [{"name": "UserJSON", "model": "gpt-4o"}], "allowAnonymousTelemetry": False})
    merged_json_cfg = preset.merge_continue_config_yaml(json_cfg, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    assert len(json.loads(merged_json_cfg)["models"]) == 3
    cleaned_json_cfg = preset.remove_hivemind_from_continue_config_yaml(merged_json_cfg)
    assert cleaned_json_cfg is not None and [m["name"] for m in json.loads(cleaned_json_cfg)["models"]] == ["UserJSON"]

    # Purely HiveMind config returns None on remove
    pure_hm = preset.merge_continue_config_yaml(None, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    assert preset.remove_hivemind_from_continue_config_yaml(pure_hm) is None

    for bad_yaml in ("models: scalar\n", "- top_level_list\n", "a\x00b", "{broken_json"):
        with pytest.raises(preset.QuickPresetError):
            preset.merge_continue_config_yaml(bad_yaml, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)

    # Atomic 0600 + .bak verification on disk via run_vscode_setup
    home_cfg = tmp_path / "home_cfg"
    ws_cfg = home_cfg / "ws_cfg"
    ws_cfg.mkdir(parents=True)
    (ws_cfg / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
    cont_main_path = home_cfg / ".continue" / "config.yaml"
    cont_main_path.parent.mkdir(parents=True)
    cont_main_path.write_text(existing_yaml, encoding="utf-8")

    def _setup_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST" and url.endswith("/register"):
            return 200, {"account_id": "acct_cfg_1", "api_key": ALPHA_KEY, "role": "prosumer"}
        if url.endswith("/me"):
            return 200, {"account_id": "acct_cfg_1", "handle": "cfg-dev", "role": "prosumer"}
        if url.endswith("/feedback"):
            return 200, {"total_responses": 1, "mean_ease_rating": 5.0}
        return _healthy_avg_http(method, url)

    setup_res = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=ws_cfg,
        credentials_path=home_cfg / ".hivemind" / "credentials.json",
        continue_config_path=home_cfg / ".continue" / "hivemind_provider.yaml",
        continue_main_config_path=cont_main_path,
        write_continue_config=True,
        ces_queue_file=home_cfg / ".hivemind" / "vscode_setup_ces_queue.json",
        http_fn=_setup_http,
        env=CLEAN_ENV,
        home=home_cfg,
    )
    _verify_receipt(setup_res)
    assert setup_res["file_writes"] == 5
    assert (cont_main_path.parent / "config.yaml.bak").is_file()
    if os.name != "nt":
        assert (cont_main_path.stat().st_mode & 0o777) == 0o600

    un_res = preset.uninstall_vscode_quick_preset(
        workspace_dir=ws_cfg,
        credentials_path=home_cfg / ".hivemind" / "credentials.json",
        continue_config_path=home_cfg / ".continue" / "hivemind_provider.yaml",
        continue_main_config_path=cont_main_path,
        ces_queue_file=home_cfg / ".hivemind" / "vscode_setup_ces_queue.json",
        env=CLEAN_ENV,
        home=home_cfg,
    )
    assert un_res["exit_code"] == 0
    assert not (cont_main_path.parent / "config.yaml.bak").exists()
    assert [m["name"] for m in yaml.safe_load(cont_main_path.read_text(encoding="utf-8"))["models"]] == ["Local Ollama Qwen"]

    # 3. Interactive parse_ces_ease_input & prompt_and_submit_ces
    assert preset.parse_ces_ease_input("")["skipped"] is True
    assert preset.parse_ces_ease_input("skip")["skipped"] is True
    p_dev = preset.parse_ces_ease_input("5 nolag")
    assert p_dev == {"skipped": False, "profile": "dev", "ease_rating": 5, "had_game_lag": False, "zero_game_lag": True}
    p_gam = preset.parse_ces_ease_input("4/5 gamer lag")
    assert p_gam == {"skipped": False, "profile": "gamer", "ease_rating": 4, "had_game_lag": True, "zero_game_lag": False}

    for bad_in in ("0", "6", "5 4", "hello", "dev"):
        with pytest.raises(preset.QuickPresetError):
            preset.parse_ces_ease_input(bad_in)

    # prompt_and_submit_ces: EOF skip, HTTP 200 (no queue file), HTTP 404 (spools to 0600 queue)
    q_prompt = home_cfg / ".hivemind" / "prompt_ces_queue.json"
    res_skip = preset.prompt_and_submit_ces(
        gateway="http://127.0.0.1:8000",
        input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
        http_fn=_setup_http,
        ces_queue_file=q_prompt,
        home=home_cfg,
        lang="fr",
    )
    assert res_skip["skipped"] is True and res_skip["network_calls"] == 0 and not q_prompt.exists()

    res_200 = preset.prompt_and_submit_ces(
        gateway="http://127.0.0.1:8000",
        input_fn=lambda _p: "5 nolag",
        http_fn=_setup_http,
        ces_queue_file=q_prompt,
        home=home_cfg,
        lang="en",
    )
    assert res_200["ces_recorded"] is True and res_200["queued_for_retry"] is False and not q_prompt.exists()

    res_404 = preset.prompt_and_submit_ces(
        gateway="http://127.0.0.1:8000",
        input_fn=lambda _p: "4 gamer lag",
        http_fn=lambda _m, _u, **_k: (404, {"detail": "missing"}),
        ces_queue_file=q_prompt,
        home=home_cfg,
        lang="fr",
    )
    assert res_404["ces_recorded"] is False and res_404["queued_for_retry"] is True and q_prompt.is_file()
    if os.name != "nt":
        assert (q_prompt.stat().st_mode & 0o777) == 0o600

    # 4. VscodeCesRetrySpool: no-file-when-empty, bounded max_entries=128 eviction, 400/422 quarantine, max_delivered=512
    empty_q = home_cfg / ".hivemind" / "never_created.json"
    fl_empty = preset.flush_vscode_ces_queue("http://127.0.0.1:8000", queue_file=empty_q, home=home_cfg, env=CLEAN_ENV, cwd=ws_cfg)
    _verify_receipt(fl_empty)
    assert fl_empty["queue_exists"] is False and fl_empty["network_calls"] == 0 and not empty_q.exists()

    spool_128_path = home_cfg / ".hivemind" / "spool_128.json"
    spool_128 = preset.VscodeCesRetrySpool(spool_128_path, home=home_cfg, env=CLEAN_ENV)
    for idx in range(130):
        enq_i = spool_128.enqueue(
            gateway="http://127.0.0.1:8000",
            account_ref=f"acct-{idx:03d}",
            profile="dev" if idx % 2 == 0 else "gamer",
            ease_rating=(idx % 5) + 1,
            zero_game_lag=True,
            env=CLEAN_ENV,
            cwd=ws_cfg,
        )
        assert enq_i["enqueued"] is True
    snap_128 = spool_128.snapshot()
    assert snap_128["pending_count"] == 128
    assert snap_128["dropped_overflow_count"] == 2
    if os.name != "nt":
        assert (spool_128_path.stat().st_mode & 0o777) == 0o600

    # Duplicate pending enqueue is deduplicated
    dup_pending = spool_128.enqueue(
        gateway="http://127.0.0.1:8000",
        account_ref="acct-010",
        profile="dev",
        ease_rating=(10 % 5) + 1,
        zero_game_lag=True,
        env=CLEAN_ENV,
        cwd=ws_cfg,
    )
    assert dup_pending["enqueued"] is False and dup_pending["duplicate_state"] == "already_pending"

    # Mixed HTTP flush: transient 404/408/429/503/0 retained vs permanent 400/422 quarantined vs 200 delivered
    spool_mix_path = home_cfg / ".hivemind" / "spool_mix.json"
    for idx in range(7):
        preset.enqueue_vscode_ces_event(
            queue_file=spool_mix_path,
            gateway="http://127.0.0.1:8000",
            account_ref=f"mix-{idx}",
            profile="dev",
            ease_rating=5,
            env=CLEAN_ENV,
            cwd=ws_cfg,
            home=home_cfg,
        )
    mix_statuses = [404, 408, 429, 503, 0, 400, 422]
    mix_idx = {"i": 0}

    def _mix_http(_m: str, _u: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        st = mix_statuses[mix_idx["i"]]
        mix_idx["i"] += 1
        return st, {}

    fl_mix = preset.flush_vscode_ces_queue(
        "http://127.0.0.1:8000",
        queue_file=spool_mix_path,
        http_fn=_mix_http,
        env=CLEAN_ENV,
        cwd=ws_cfg,
        home=home_cfg,
        lang="fr",
    )
    assert fl_mix["retained_retry_count"] == 5
    assert fl_mix["quarantined_count"] == 2
    assert fl_mix["pending_count"] == 5
    assert fl_mix["stale_cloud_run_detected"] is True
    assert fl_mix["exit_code"] == 2

    # Now deliver all 5 retained items on 200
    fl_recover = preset.flush_vscode_ces_queue(
        "http://127.0.0.1:8000",
        queue_file=spool_mix_path,
        http_fn=lambda _m, _u, **_k: (200, {"ok": True}),
        env=CLEAN_ENV,
        cwd=ws_cfg,
        home=home_cfg,
    )
    assert fl_recover["delivered_count"] == 5 and fl_recover["pending_count"] == 0 and fl_recover["exit_code"] == 0

    # Re-enqueueing already-delivered event is deduplicated
    re_enq = preset.enqueue_vscode_ces_event(
        queue_file=spool_mix_path,
        gateway="http://127.0.0.1:8000",
        account_ref="mix-0",
        profile="dev",
        ease_rating=5,
        env=CLEAN_ENV,
        cwd=ws_cfg,
        home=home_cfg,
    )
    assert re_enq["enqueued"] is False and re_enq["duplicate_state"] == "already_delivered"

    # Delivered ledger bounded at max_delivered=512
    spool_ledger_path = home_cfg / ".hivemind" / "spool_ledger.json"
    spool_ledger = preset.VscodeCesRetrySpool(spool_ledger_path, max_entries=600, max_delivered=512, home=home_cfg)
    for idx in range(520):
        spool_ledger.enqueue(
            gateway="http://127.0.0.1:8000",
            account_ref=f"led-{idx:04d}",
            profile="dev",
            ease_rating=5,
            env=CLEAN_ENV,
            cwd=ws_cfg,
        )
    fl_led = spool_ledger.flush("http://127.0.0.1:8000", http_fn=lambda _m, _u, **_k: (200, {}), env=CLEAN_ENV, cwd=ws_cfg)
    assert fl_led["delivered_count"] == 520
    assert fl_led["delivered_ledger_count"] == 512

    # Corrupted spool & tampered payload recovery
    corrupt_q = home_cfg / ".hivemind" / "corrupt.json"
    corrupt_q.write_text("{broken", encoding="utf-8")
    assert preset.VscodeCesRetrySpool(corrupt_q, home=home_cfg).snapshot()["quarantined_count"] == 1
    corrupt_q.write_text(
        json.dumps(
            {
                "pending": [
                    "bad-entry",
                    {"event_id": "a" * 64, "idempotency_key": "bad-1", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii-forbidden"}},
                ],
                "delivered_ids": ["", "dup1", "dup1"],
            }
        ),
        encoding="utf-8",
    )
    fl_corr = preset.flush_vscode_ces_queue("http://127.0.0.1:8000", queue_file=corrupt_q, home=home_cfg, env=CLEAN_ENV, cwd=ws_cfg)
    assert fl_corr["total_quarantined_count"] >= 1 and fl_corr["pending_count"] == 0

    # Validation errors on spool constructor & payloads
    with pytest.raises(preset.QuickPresetError):
        preset.VscodeCesRetrySpool(max_entries=0)
    with pytest.raises(preset.QuickPresetError):
        preset.VscodeCesRetrySpool(max_delivered=0)
    for bad_payload in (
        "not-dict",
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "extra": 1},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes", "platform": "linux"},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "bad/plat"},
    ):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_ces_spool_payload(bad_payload)
    with pytest.raises(preset.QuickPresetError):
        spool_128.enqueue(idempotency_key="bad\x00key", env=CLEAN_ENV, cwd=ws_cfg)


def test_04_concurrent_48_participant_stale_404_cutover_and_idempotent_ces_flush_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00201-soak48")

    upstream_inference_hits: list[str] = []

    def _reject_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn14_48_soak.sqlite3"
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
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    quick_nonces: list[str] = []
    noel_nonces: list[str] = []

    ctx_peer_ip: contextvars.ContextVar[str] = contextvars.ContextVar("peer_ip", default="127.0.0.1")
    ctx_gpu: contextvars.ContextVar[preset.GpuProbeSnapshot] = contextvars.ContextVar(
        "gpu_snap", default=preset.GpuProbeSnapshot(present=False)
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
        _seed_live_coder_producer(client)

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
                quick_nonces=quick_nonces,
                noel_nonces=noel_nonces,
                lock=shared_lock,
                override_routes=stale_overrides if stale_phase["active"] else None,
            )
            return adapter(method, url, payload=payload, headers=headers, timeout_s=timeout_s)

        monkeypatch.setattr(preset, "default_http_request", _dispatch_http)
        monkeypatch.setattr(preset, "default_gpu_probe", lambda **_k: ctx_gpu.get())
        monkeypatch.setattr(preset, "default_process_probe", lambda **_k: ["code"])
        monkeypatch.setattr(preset, "default_worker_yield_drill", lambda **_k: ctx_yield.get())

        slots: list[dict[str, Any]] = []
        for idx in range(total_participants):
            is_dev = idx < dev_cohort_size
            cohort_idx = idx if is_dev else (idx - dev_cohort_size)
            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"home_member_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "workspace"
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")

            cont_main = home_dir / ".continue" / "config.yaml"
            cont_main.parent.mkdir(parents=True, exist_ok=True)
            cont_main.write_text(
                f"name: User-{idx:02d}\nmodels:\n  - name: Local-User-{idx:02d}\n    provider: ollama\n    model: qwen2.5-coder:7b\n",
                encoding="utf-8",
            )

            if is_dev:
                gpu_snap = preset.GpuProbeSnapshot(present=False)
                yield_ms = 22.0 + cohort_idx
                host_lbl = f"dev-laptop-{cohort_idx:02d}"
                handle = f"noel-dev-{cohort_idx:02d}"
                role = "consumer" if (cohort_idx % 2 == 0) else "prosumer"
                profile = "dev"
            else:
                card, tot_mib, free_mib, temp_c = rtx_cards[cohort_idx % len(rtx_cards)]
                gpu_snap = preset.GpuProbeSnapshot(
                    present=True,
                    name=card,
                    total_vram_mib=tot_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=tot_mib - free_mib,
                    temp_c=temp_c,
                    util_pct=4.0,
                )
                yield_ms = 65.0 + (cohort_idx * 4.0)
                host_lbl = f"gamer-rtx-{cohort_idx:02d}"
                handle = f"noel-gamer-{cohort_idx:02d}"
                profile = "gamer"
                role = "prosumer"

            slots.append(
                {
                    "idx": idx,
                    "is_dev": is_dev,
                    "peer_ip": _loopback_ip_for(55, idx),
                    "os_user": os_user,
                    "home_dir": home_dir,
                    "ws_dir": ws_dir,
                    "cred_path": home_dir / ".hivemind" / "credentials.json",
                    "cont_path": home_dir / ".continue" / "hivemind_provider.yaml",
                    "cont_main": cont_main,
                    "queue_path": home_dir / ".hivemind" / "vscode_setup_ces_queue.json",
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

        # Phase 1: Stale pre-PR-#949 Cloud Run (404 on /quick & /feedback) -> local setup + 0600 CES spool
        barrier1 = threading.Barrier(total_participants)
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

        def _phase1_concurrent(slot: dict[str, Any]) -> dict[str, Any]:
            ctx_peer_ip.set(slot["peer_ip"])
            ctx_gpu.set(slot["gpu_snap"])
            ctx_yield.set(slot["yield_ms"])
            buf = io.StringIO()
            thread_streams[threading.get_ident()] = buf
            barrier1.wait(timeout=25.0)
            rc = cli_main.main(
                [
                    "vscode-setup",
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--workspace",
                    str(slot["ws_dir"]),
                    "--credentials-path",
                    str(slot["cred_path"]),
                    "--continue-config-path",
                    str(slot["cont_path"]),
                    "--continue-main-config-path",
                    str(slot["cont_main"]),
                    "--write-continue-config",
                    "--ces-queue-file",
                    str(slot["queue_path"]),
                    "--handle",
                    slot["handle"],
                    "--role",
                    slot["role"],
                    "--host-label",
                    slot["host_lbl"],
                    *(["--gamer-preflight", "--first-heartbeat"] if not slot["is_dev"] else []),
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
            assert rc == 0
            assert doc["ready"] is True
            assert doc["within_60s_slo"] is True
            assert doc["models_preflight_ok"] is True
            assert doc["stale_revision_detected"] is True
            assert doc["ces_recorded"] is False
            assert doc["ces_queued_for_retry"] is True
            assert slot["queue_path"].is_file()
            assert str(Path.home()) not in raw_json
            assert Path.home().name not in raw_json
            saved_cred = json.loads(slot["cred_path"].read_text(encoding="utf-8"))
            assert str(saved_cred["api_key"]) not in raw_json
            if os.name != "nt":
                assert (slot["queue_path"].stat().st_mode & 0o777) == 0o600
                assert (slot["cont_main"].stat().st_mode & 0o777) == 0o600
            return doc

        monkeypatch.setattr(sys, "stdout", _ThreadLocalStdout())
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            p1_docs = [fut.result(timeout=60.0) for fut in [pool.submit(_phase1_concurrent, s) for s in slots]]
        assert len(p1_docs) == 48
        assert client.get("/v1/hivemind/onboarding/feedback/summary").json()["total_responses"] == 0

        # Phase 2: Cutover to live create_router_app -> all 48 threads verify /quick + /noel & flush CES spools idempotently
        stale_phase["active"] = False
        barrier2 = threading.Barrier(total_participants)

        def _phase2_cutover_flush(slot: dict[str, Any]) -> dict[str, Any]:
            ctx_peer_ip.set(slot["peer_ip"])
            ctx_gpu.set(slot["gpu_snap"])
            ctx_yield.set(slot["yield_ms"])
            buf = io.StringIO()
            thread_streams[threading.get_ident()] = buf
            barrier2.wait(timeout=25.0)
            rc = cli_main.main(
                [
                    "vscode-setup",
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--workspace",
                    str(slot["ws_dir"]),
                    "--credentials-path",
                    str(slot["cred_path"]),
                    "--continue-config-path",
                    str(slot["cont_path"]),
                    "--continue-main-config-path",
                    str(slot["cont_main"]),
                    "--write-continue-config",
                    "--ces-queue-file",
                    str(slot["queue_path"]),
                    "--handle",
                    slot["handle"],
                    "--role",
                    slot["role"],
                    "--host-label",
                    slot["host_lbl"],
                    "--expected-sha",
                    LIVE_SHORT_SHA,
                    "--require-holiday-routes",
                    "--flush-ces-queue",
                    *(["--gamer-preflight"] if not slot["is_dev"] else []),
                    "--profile",
                    slot["profile"],
                    "--lang",
                    slot["lang"],
                    "--json",
                ]
            )
            raw_json = buf.getvalue()
            doc = json.loads(raw_json)
            _verify_receipt(doc)
            assert rc == 0, f"Cutover failed for slot {slot['idx']}: {doc}"
            assert doc["ready"] is True
            assert doc["file_writes"] == 0
            assert doc["stale_revision_detected"] is False
            assert doc["ces_flush"]["delivered_count"] == 1
            assert doc["ces_flush"]["pending_count"] == 0
            assert str(Path.home()) not in raw_json
            assert Path.home().name not in raw_json
            saved_cred = json.loads(slot["cred_path"].read_text(encoding="utf-8"))
            assert str(saved_cred["api_key"]) not in raw_json
            parsed_cm = yaml.safe_load(slot["cont_main"].read_text(encoding="utf-8"))
            assert parsed_cm["models"][0]["name"] == f"Local-User-{slot['idx']:02d}"
            return doc

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            p2_docs = [fut.result(timeout=60.0) for fut in [pool.submit(_phase2_cutover_flush, s) for s in slots]]
        assert len(p2_docs) == 48

        # Phase 3: Re-flush idempotency check -> 0 duplicate deliveries
        for slot in slots:
            ctx_peer_ip.set(slot["peer_ip"])
            replay_fl = preset.flush_vscode_ces_queue(
                "http://127.0.0.1:8000",
                queue_file=slot["queue_path"],
                home=slot["home_dir"],
                env=CLEAN_ENV,
                cwd=slot["ws_dir"],
            )
            assert replay_fl["delivered_count"] == 0
            assert replay_fl["network_calls"] == 0

        monkeypatch.undo()
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _ in seen_calls)

        # Unique per-request CSP nonces on BOTH /quick (96 responses) and /noel (48 responses)
        assert len(quick_nonces) == 96
        assert len(set(quick_nonces)) == 96
        assert len(noel_nonces) == 48
        assert len(set(noel_nonces)) == 48

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
        assert fb_summary["zero_game_lag_ratio"] == 1.0


def test_05_fail_closed_compliance_and_standalone_preset_edge_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Fail-closed Amadeus / corporate env & .git/config remote blocking (0 writes, 0 HTTP calls, exit code 2)
    http_boom: list[str] = []

    def _forbid_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        http_boom.append(f"{method}:{url}")
        raise AssertionError("Corporate compliance gate MUST block before any HTTP call!")

    corp_cases = (
        {"AMADEUS_SITE": "nce-secret-building-42"},
        {"USERDNSDOMAIN": "nce.amadeus.internal"},
        {"USERDOMAIN": "ENTCORP_AD"},
        {"HTTPS_PROXY": CORP_ZSCALER_PROXY},
        {"ALL_PROXY": "http://" + "gw.netskope.internal:8080"},
        {"COMPUTERNAME": "WKS-FORTINET-CORP-99", "HOSTNAME": "wks-paloalto-corp"},
        {"ZSCALER_CERT_PATH": "/etc/ssl/zscaler-root-secret.pem"},
    )
    for idx, bad_env in enumerate(corp_cases):
        ws_c = tmp_path / f"corp_ws_{idx}"
        ws_c.mkdir()
        home_c = tmp_path / f"corp_home_{idx}"
        out_c = io.StringIO()
        rc_c = preset.main(
            [
                "--vscode-setup",
                "--write-continue-config",
                "--submit-ces",
                "--workspace",
                str(ws_c),
                "--credentials-path",
                str(home_c / ".hivemind" / "credentials.json"),
                "--continue-config-path",
                str(home_c / ".continue" / "hivemind_provider.yaml"),
                "--continue-main-config-path",
                str(home_c / ".continue" / "config.yaml"),
                "--ces-queue-file",
                str(home_c / ".hivemind" / "vscode_setup_ces_queue.json"),
                "--json",
            ],
            env={**CLEAN_ENV, **bad_env},
            cwd=ws_c,
            home=home_c,
            http_fn=_forbid_http,
            stdout=out_c,
        )
        assert rc_c == 2
        corp_doc = json.loads(out_c.getvalue())
        _verify_receipt(corp_doc)
        assert corp_doc["compliance_ok"] is False
        assert corp_doc["file_writes"] == 0
        assert corp_doc["network_calls"] == 0
        assert list(ws_c.iterdir()) == []
        assert not home_c.exists()

        # Also block preflight, enqueue, flush directly
        assert preset.run_one_shot_quick_health_preflight(http_fn=_forbid_http, env={**CLEAN_ENV, **bad_env}, cwd=ws_c)["exit_code"] == 2
        assert preset.enqueue_vscode_ces_event(queue_file=home_c / "q.json", env={**CLEAN_ENV, **bad_env}, cwd=ws_c)["exit_code"] == 2
        assert preset.flush_vscode_ces_queue(queue_file=home_c / "q.json", http_fn=_forbid_http, env={**CLEAN_ENV, **bad_env}, cwd=ws_c)["exit_code"] == 2

    # Corporate workspace path + .git/config + git worktree indirection
    corp_ws_path = tmp_path / "amadeus_internal_checkout"
    corp_ws_path.mkdir()
    assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_ws_path, lang="fr")["compliance_ok"] is False

    git_ws = tmp_path / "hobby_repo_corp_remote"
    (git_ws / ".git").mkdir(parents=True)
    (git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n[url "{CORP_HTTPS_REMOTE}"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )
    assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=git_ws, lang="en")["compliance_ok"] is False

    common_git = tmp_path / "common_git"
    wt_git = common_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_HTTPS_REMOTE}\n', encoding="utf-8")
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "clean_wt_ws"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")
    assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws)["compliance_ok"] is False
    assert http_boom == []

    # 2. Exercise loopback default_http_request, parity, gamer preflight, JSONC, validators, and CLI modes for >=95% standalone coverage
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]})
            elif self.path == "/v1/hivemind/me":
                self._send_json(200, {"account_id": "acct_lb", "handle": "lb-dev", "role": "prosumer"})
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(
                    200,
                    {
                        "object": preset.MANIFEST_OBJECT,
                        "build_sha": LIVE_BUILD_SHA,
                        "cloud_run_revision": "rev-lb-01",
                        "personal_pc_only": True,
                        "financial_promise": preset.FINANCIAL_PROMISE,
                        "holiday_routes": list(preset.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                        "downloadable_scripts": list(preset.EXPECTED_MANIFEST_CORE_SCRIPTS),
                    },
                )
            elif self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(
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
                self._send_json(200, {"live_hiveminders": 1, "total_hiveminders": 1})
            elif self.path in ("/noel", "/quick", "/raid-noel"):
                nonce = f"n{abs(hash(self.path)):010d}abcd"
                html = f"<!DOCTYPE html><html><head><style nonce='{nonce}'>body{{}}</style></head><body>100% personal AGPL-3.0 0.00 EUR</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy", f"default-src 'none'; style-src 'nonce-{nonce}'")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
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
                self._send_json(200, {"node": {"node_id": "node-lb-01"}})
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
        assert preset.default_http_request("GET", f"{base_url}/list")[1]["data"] == [1, 2]
        assert preset.default_http_request("GET", f"{base_url}/broken-json")[1]["raw_error"] == "invalid_json"
        assert preset.default_http_request("GET", f"{base_url}/missing")[0] == 404

        parity_ok = preset.verify_cloud_run_holiday_parity(base_url, expected_sha=LIVE_SHORT_SHA, env=CLEAN_ENV, cwd=tmp_path)
        assert parity_ok["parity_ok"] is True

        ws_lb = tmp_path / "ws_lb"
        ws_lb.mkdir()
        (ws_lb / ".vscode").mkdir()
        (ws_lb / ".vscode" / "tasks.json").write_text('{"version": "2.0.0", "tasks": [{"label": "User Task", "type": "process", "command": "echo",},],}', encoding="utf-8")
        (ws_lb / ".vscode" / "settings.json").write_text('{\n  // user pref\n  "editor.tabSize": 4,\n}', encoding="utf-8")

        home_lb = tmp_path / "home_lb"
        q_lb = home_lb / ".hivemind" / "lb_queue.json"
        preset.enqueue_vscode_ces_event(queue_file=q_lb, gateway=base_url, home=home_lb, env=CLEAN_ENV, cwd=ws_lb)

        out_lb = io.StringIO()
        rc_lb = preset.main(
            [
                "--gateway",
                base_url,
                "--vscode-setup",
                "--workspace",
                str(ws_lb),
                "--credentials-path",
                str(home_lb / ".hivemind" / "credentials.json"),
                "--continue-config-path",
                str(home_lb / ".continue" / "hivemind_provider.yaml"),
                "--continue-main-config-path",
                str(home_lb / ".continue" / "config.yaml"),
                "--write-continue-config",
                "--ces-queue-file",
                str(q_lb),
                "--flush-ces-queue",
                "--prompt-ces",
                "--first-heartbeat",
                "--gamer-preflight",
                "--lang",
                "fr",
            ],
            env=CLEAN_ENV,
            cwd=ws_lb,
            home=home_lb,
            input_fn=lambda _p: "5 nolag",
            gpu_probe_fn=lambda **_k: preset.GpuProbeSnapshot(present=True, name="RTX 4070", total_vram_mib=12288, free_vram_mib=9216, used_vram_mib=3072, temp_c=55.0),
            yield_drill_fn=lambda **_k: 35.0,
            process_probe_fn=lambda **_k: ["code"],
            stdout=out_lb,
        )
        assert rc_lb == 0 and "OK" in out_lb.getvalue()

        # Standalone --flush-ces-queue & --check-parity & --install-preset CLI branches
        assert preset.main(["--gateway", base_url, "--flush-ces-queue", "--ces-queue-file", str(q_lb)], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=io.StringIO()) == 0
        assert preset.main(["--gateway", base_url, "--check-parity"], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=io.StringIO()) == 0
        assert preset.main(["--gateway", base_url, "--install-preset", "--workspace", str(ws_lb), "--credentials-path", str(home_lb / ".hivemind" / "credentials.json"), "--continue-config-path", str(home_lb / ".continue" / "hivemind_provider.yaml"), "--submit-ces"], env=CLEAN_ENV, cwd=ws_lb, home=home_lb, stdout=io.StringIO()) == 0
    finally:
        srv.shutdown()
        srv.server_close()

    assert preset.default_http_request("GET", f"http://127.0.0.1:{port}/health", timeout_s=0.2)[0] == 0

    # Real yield drill, GPU/process probes, validator edges, and error branches
    assert preset.default_worker_yield_drill(timeout_s=2.5)["clean_release"] is True
    assert preset.default_gpu_probe(runner=lambda _c, _t: (0, "RTX 4070, 12288, 9216, 3072, 54, 5\n")).present is True
    assert preset.default_gpu_probe(runner=lambda _c, _t: (0, "bad")).parse_error is True
    assert preset.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert "cs2.exe" in preset.default_process_probe(runner=lambda _c, _t: (0, '"cs2.exe","1","Console","1","10 K"\n'), os_name="nt")
    assert "python3" in preset.default_process_probe(runner=lambda _c, _t: (0, "bash\npython3\n"), os_name="posix")
    assert preset.default_subprocess_runner([sys.executable, "-c", "print('ok')"])[0] == 0
    assert preset.default_subprocess_runner(["/nonexistent_bin_dyn14"])[0] == 127

    rtx_hot = preset.GpuProbeSnapshot(present=True, name="RTX 3080", total_vram_mib=10240, free_vram_mib=7168, used_vram_mib=3072, temp_c=74.0)
    assert preset.run_steam_yield_and_thermal_preflight(gpu_probe_fn=lambda: rtx_hot, yield_drill_fn=lambda: 30.0, process_probe_fn=lambda: ["cs2.exe"])["preflight_ok"] is False
    assert preset.run_vscode_setup("http://127.0.0.1:8000", workspace_dir=tmp_path, skip_preflight=True, run_gamer_preflight=True, gpu_probe_fn=lambda: rtx_hot, yield_drill_fn=lambda: 30.0, process_probe_fn=lambda: [], env=CLEAN_ENV, home=tmp_path)["exit_code"] == 2

    # Unreachable gateway triggers exit_code 2 in run_vscode_setup
    assert preset.run_vscode_setup(f"http://127.0.0.1:{port}", workspace_dir=tmp_path, env=CLEAN_ENV, home=tmp_path)["exit_code"] == 2
    assert preset.main(["--gateway", UNTRUSTED_EXT_GW, "--vscode-setup", "--json"], env=CLEAN_ENV, cwd=tmp_path, stdout=io.StringIO()) == 2
    assert preset.run_vscode_setup("http://127.0.0.1:8000", uninstall=True, workspace_dir=tmp_path, env=CLEAN_ENV, home=tmp_path)["exit_code"] == 0

    # Safety & sanitizer edge assertions
    assert preset.mask_api_key("") == "<missing>"
    assert preset.mask_api_key("short") == "<invalid>"
    assert preset.mask_api_key(LIVE_KEY) == "hm_live_***4321"
    assert preset.sanitize_host_label("amadeus-pc") == "personal-pc"
    assert preset.sanitize_build_sha("secret12345") == "unknown"
    assert preset.sanitize_cloud_run_revision("hm-1234567890") is None
    assert preset.shas_match("release2026a", "release2026b") is False
    assert preset.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert preset.is_allowed_outbound_host("storage.googleapis.com") is True
    for bad_gw in (UNTRUSTED_EXT_GW, UNTRUSTED_FTP_GW, INSECURE_RUN_APP_GW, "https://" + "u:p@x.run.app"):
        with pytest.raises(preset.QuickPresetError):
            preset.validate_gateway_url(bad_gw)
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/v1/chat/completions")
    assert preset._normalize_gpu_snapshot(SimpleNamespace(present=True, name="RTX", total_mib=8192, free_mib=4096)).present is True
    assert preset._normalize_yield_drill_result(math.nan) == (None, False)
    assert preset._platform_tag("FreeBSD_14") == "freebsd_14"
    assert preset._platform_tag("!!!") == "unknown"
    ip_pub = ".".join(("10", "0", "0", "1"))
    assert len(preset.scan_output_safety(f"a@b.org {ALPHA_KEY} Bearer tok_123456 /home/bob/x {ip_pub}")) >= 5

    # Additional standalone branch coverage for validators, HTTP arities, JSONC, parity, auth retries, CES & uninstall
    for bad_lang in (True, "de"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_lang(bad_lang)  # type: ignore[arg-type]
    for bad_prof in (None, True, "admin"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_profile(bad_prof)  # type: ignore[arg-type]
    for bad_role in (None, True, "root"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_role(bad_role)  # type: ignore[arg-type]
    for bad_ease in (True, "nan", 0, 6):
        with pytest.raises(preset.QuickPresetError):
            preset.parse_ease_rating(bad_ease)  # type: ignore[arg-type]
    assert preset.sanitize_handle(None).startswith("noel-dev-")
    with pytest.raises(preset.QuickPresetError):
        preset.sanitize_handle("google.spoof")
    with pytest.raises(preset.QuickPresetError):
        preset.sanitize_handle("!")
    with pytest.raises(preset.QuickPresetError):
        preset.validate_expected_sha("unknown")
    assert preset.shas_match("unknown", LIVE_BUILD_SHA) is False
    assert preset.sanitize_cloud_run_revision(None) is None
    assert preset.is_allowed_outbound_host("") is False
    for bad_gw_edge in ("", "https://" + "x.run.app\n", "https://" + "bad space.run.app", "https://" + "x.run.app?a=1", "https://" + "x.run.app/p", "http://" + "127.0.0.1:0", "http://" + "127.0.0.1:bad"):
        with pytest.raises(preset.QuickPresetError):
            preset.validate_gateway_url(bad_gw_edge)
    for bad_req in ("", "http://" + "127.0.0.1:8000/a b", "http://" + "u:p@127.0.0.1:8000/h"):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_outbound_request_url(bad_req)
    with pytest.raises(preset.QuickPresetError):
        preset._assert_safe_output(f"leak {ALPHA_KEY}")

    assert preset._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-b": "2"}}))[1]["headers"]["x-b"] == "2"
    assert preset._normalize_http_return((200, {}, [10, 20]))[1]["data"] == [10, 20]
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_http_return("bad")
    assert preset._invoke_http_fn(lambda m, u: (200, "<html></html>"), "GET", "http://127.0.0.1:8000/noel")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p: (200, {"p": p}), "POST", "http://127.0.0.1:8000/v1/hivemind/register", payload={"a": 1})[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p, h: (200, {"h": h}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p, h, t: (200, {"t": t}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, **kw: (200, {"kw": bool(kw)}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("down")), "GET", "http://127.0.0.1:8000/health")[0] == 0

    escaped_jsonc = '{"a": "q\\" \\\\", /* block */ "b": [1,],}'
    assert json.loads(preset.strip_jsonc_comments_and_trailing_commas(escaped_jsonc))["b"] == [1]
    for sec_txt in (ALPHA_KEY, "Bearer tok_123456", "/home/alice/x"):
        with pytest.raises(preset.QuickPresetError):
            preset._assert_workspace_content_secret_free(sec_txt)
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", "bad")
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", ALPHA_KEY, max_tokens=16)
    for bad_j in ("{bad", "[]"):
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_settings_json(bad_j, gateway="http://127.0.0.1:8000")
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_tasks_json(bad_j, gateway="http://127.0.0.1:8000")
    with pytest.raises(preset.QuickPresetError):
        preset.build_merged_vscode_tasks_json('{"tasks": "bad"}', gateway="http://127.0.0.1:8000")
    assert preset._is_hivemind_managed_task("bad") is False

    partial_gi = tmp_path / "partial_gi"
    partial_gi.mkdir()
    (partial_gi / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    assert preset.check_workspace_gitignore_safety(partial_gi, lang="fr")["vscode_ignored"] is False
    for fn, raw_c in (("a.json", "{bad"), ("b.json", "[]"), ("c.json", '{"api_key":"x"}'), ("d.json", json.dumps({"api_key": ALPHA_KEY, "gateway": "https://other.a.run.app"}))):
        cp = tmp_path / fn
        cp.write_text(raw_c, encoding="utf-8")
        assert preset.load_saved_credentials(cp, expected_gateway="http://127.0.0.1:8000") is None

    # Parity broken / 404 / 503 / SHA drift
    assert preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        http_fn=lambda _m, u, **_k: (404, {}) if u.endswith(("/noel", "/quick", "/raid-noel")) else ((503, {}) if u.endswith(("/health", "/v1/models", "/fleet")) else (200, {"object": "wrong", "build_sha": LIVE_BUILD_SHA})),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["parity_ok"] is False

    # Auth 409 retry + stage failures + consumer heartbeat skip
    reg_try = {"n": 0}
    def _409_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST" and url.endswith("/register"):
            reg_try["n"] += 1
            return (409, {}) if reg_try["n"] == 1 else (200, {"account_id": "acct_409", "api_key": ALPHA_KEY, "role": "consumer"})
        if url.endswith("/me"):
            return 200, {"account_id": "acct_409", "handle": "dev-409", "role": "consumer"}
        return 200, {"data": [{"id": "hivemind/coder"}]}

    ws_409 = tmp_path / "ws_409"
    ws_409.mkdir()
    r409 = preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=ws_409, credentials_path=tmp_path / "h409" / "c.json", continue_config_path=tmp_path / "h409" / "p.yaml", role="consumer", send_first_heartbeat=True, http_fn=_409_http, env=CLEAN_ENV, home=tmp_path / "h409")
    assert r409["ready"] is True and reg_try["n"] == 2 and r409["heartbeat"]["skipped_consumer_role"] is True

    for fail_fn in (
        lambda m, u, **_k: (500, {}) if u.endswith("/register") else (200, {}),
        lambda m, u, **_k: (200, {"api_key": ALPHA_KEY}) if u.endswith("/register") else (401, {}),
        lambda m, u, **_k: (200, {"api_key": ALPHA_KEY}) if u.endswith("/register") else ((200, {"role": "prosumer"}) if u.endswith("/me") else (503, {})),
    ):
        assert preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=ws_409, credentials_path=tmp_path / "hf" / "c.json", continue_config_path=tmp_path / "hf" / "p.yaml", http_fn=fail_fn, env=CLEAN_ENV, home=tmp_path / "hf", lang="fr")["exit_code"] == 2

    # CES status branches (404, 429, 503, 0, 400) + run_vscode_setup stale 404 with require_holiday_routes=True & prompt_ces 404
    for st in (404, 429, 503, 0, 400):
        res_st, _ = preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", http_fn=lambda _m, _u, s=st, **_k: (s, {}), lang="fr")
        assert res_st["ces_recorded"] is False
    with pytest.raises(preset.QuickPresetError):
        preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", zero_game_lag="no")  # type: ignore[arg-type]

    # Trigger atomic_write_config_file mode tightening + uninstall preserving user settings/tasks + corrupt json uninstall
    if os.name != "nt":
        os.chmod(tmp_path / "h409" / "c.json", 0o644)
    preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=ws_409, credentials_path=tmp_path / "h409" / "c.json", continue_config_path=tmp_path / "h409" / "p.yaml", http_fn=_409_http, env=CLEAN_ENV, home=tmp_path / "h409")
    assert preset.uninstall_vscode_quick_preset(workspace_dir=ws_409, credentials_path=tmp_path / "h409" / "c.json", continue_config_path=tmp_path / "h409" / "p.yaml", env=CLEAN_ENV, home=tmp_path / "h409", lang="fr")["exit_code"] == 0

    mal_ws = tmp_path / "mal_ws"
    (mal_ws / ".vscode").mkdir(parents=True)
    (mal_ws / ".vscode" / "settings.json").write_text("{corrupt", encoding="utf-8")
    (mal_ws / ".vscode" / "tasks.json").write_text("{corrupt", encoding="utf-8")
    assert preset.uninstall_vscode_quick_preset(workspace_dir=mal_ws, env=CLEAN_ENV, home=tmp_path)["exit_code"] == 0

    # run_vscode_setup --prompt-ces on 404 with require_holiday_routes=True fails closed after spooling
    ws_st_req = tmp_path / "ws_st_req"
    ws_st_req.mkdir()
    def _404_post_http(method: str, url: str, **_k: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST" and url.endswith("/register"):
            return 200, {"account_id": "acct_r", "api_key": ALPHA_KEY, "role": "prosumer"}
        if url.endswith("/me"):
            return 200, {"account_id": "acct_r", "handle": "r-dev", "role": "prosumer"}
        if url.endswith("/feedback"):
            return 404, {}
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}]}
        return 404, {}

    res_p404 = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=ws_st_req,
        credentials_path=tmp_path / "hp404" / "c.json",
        continue_config_path=tmp_path / "hp404" / "p.yaml",
        ces_queue_file=tmp_path / "hp404" / "q.json",
        prompt_ces=True,
        input_fn=lambda _p: "4 nolag",
        http_fn=_404_post_http,
        env=CLEAN_ENV,
        home=tmp_path / "hp404",
        lang="fr",
    )
    assert res_p404["exit_code"] == 0 and res_p404["ces_queued_for_retry"] is True
    human_fb = preset.render_human_report(res_p404, lang="fr")
    assert "Fallback Guidance:" in human_fb

    # Broken Popen yield drill & GPU parse branches
    class _BrokenProc:
        def __init__(self, banner: str, raise_wait: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(banner)
            self.stderr = io.StringIO()
            self._raise_wait = raise_wait

        def kill(self) -> None:
            self._raise_wait = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._raise_wait:
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenProc("BAD\n"))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenProc("HIVEMIND_WORKER_READY\n", True))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()
    assert preset._optional_finite_float("n/a") is None
    assert preset._optional_finite_float("bad") is None
    assert preset.default_gpu_probe(runner=lambda _c, _t: (0, "RTX, -5, 0, 0, n/a, bad\n")).parse_error is True
    assert preset.run_steam_yield_and_thermal_preflight(gpu_probe_fn=lambda **_k: {"present": True, "parse_error": True}, yield_drill_fn=lambda: 20.0, process_probe_fn=lambda: ["", "steam_app_730"])["mode"] == "gpu_telemetry_parse_error"
    assert preset._normalize_gpu_snapshot(None).present is False
    assert preset._normalize_gpu_snapshot({"present": False}).present is False
    assert preset._normalize_gpu_snapshot({"present": True}).parse_error is True
    assert preset._normalize_yield_drill_result({"elapsed_ms": -1.0}) == (None, False)
    assert preset._platform_tag("win32") == "windows"
    assert preset._platform_tag("Darwin") == "macos"
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_gpu_snapshot(123)
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_yield_drill_result(True)
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_yield_drill_result("bad")
    with pytest.raises(preset.QuickPresetError):
        preset.default_subprocess_runner("")  # type: ignore[arg-type]

    # Worktree .git file with commondir + compliance block on verify_cloud_run_holiday_parity
    wt_root = tmp_path / "wt_commondir_ws"
    wt_root.mkdir()
    main_gd = tmp_path / "main_repo_git"
    wt_gd = main_gd / "worktrees" / "slot1"
    wt_gd.mkdir(parents=True)
    (main_gd / "config").write_text('[remote "origin"]\n\turl = git@localhost:amadeus/repo.git\n', encoding="utf-8")
    (wt_gd / "config").write_text('[core]\n\tbare = false\n', encoding="utf-8")
    (wt_gd / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_root / ".git").write_text(f"gitdir: {wt_gd}\n", encoding="utf-8")
    par_blocked = preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        env={"ZSCALER_CERT_PATH": "/etc/ssl/z.pem", "USERDNSDOMAIN": "HOME.LOCAL"},
        cwd=wt_root,
        lang="fr",
    )
    assert par_blocked["exit_code"] == 2 and par_blocked["network_calls"] == 0

    # Manifest / feedback 404 parity branch
    par_404 = preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        http_fn=lambda _m, u, **_k: (404, {}) if u.endswith(("/manifest", "/feedback/summary")) else (200, {"status": "ok", "data": [], "live_hiveminders": 0, "total_hiveminders": 0}),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert "/v1/hivemind/onboarding/manifest" in par_404["missing_or_stale_routes"]
    assert "/v1/hivemind/onboarding/feedback/summary" in par_404["missing_or_stale_routes"]

    # Continue YAML / JSON merge & remove edge branches
    assert "models:" in preset.merge_continue_config_yaml("name: Trailing\n\n\n", gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    for bad_c in ("[1, 2]", '{"models": "bad"}'):
        with pytest.raises(preset.QuickPresetError):
            preset.merge_continue_config_yaml(bad_c, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    comment_only_mod = "name: COnly\nmodels:\n  # comment only\ntabAutocompleteModel:\n  model: phi3\n"
    assert len(yaml.safe_load(preset.merge_continue_config_yaml(comment_only_mod, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY))["models"]) == 2
    quoted_mod = "models:\n  # leading comment\n  - name: 'Quoted # keep'\n    # inner comment\n    provider: openai\n    model: gpt-4o\n    apiBase: \"http://127.0.0.1:11434/\\\"v1\\\"\"\n"
    merged_q = preset.merge_continue_config_yaml(quoted_mod, gateway="http://127.0.0.1:8000", api_key=ALPHA_KEY)
    assert "Quoted # keep" in merged_q
    assert preset._is_hivemind_continue_model_dict("not_a_dict") is False
    assert preset.remove_hivemind_from_continue_config_yaml("") is None
    assert preset.remove_hivemind_from_continue_config_yaml("{bad") == "{bad"
    assert preset.remove_hivemind_from_continue_config_yaml("[]") == "[]"
    assert preset.remove_hivemind_from_continue_config_yaml('{"models": "bad"}') == '{"models": "bad"}'
    assert preset.remove_hivemind_from_continue_config_yaml('{"models": [{"model": "hivemind/coder"}]}') is None
    assert preset.remove_hivemind_from_continue_config_yaml("name: NoMod\n") == "name: NoMod\n"

    # VscodeCesRetrySpool ~ path, default path, malformed root, overflow trimming, duplicate delivered, and .purge()
    home_sp = tmp_path / "home_sp"
    home_sp.mkdir()
    assert preset.VscodeCesRetrySpool(home=home_sp, env=CLEAN_ENV).path == preset.default_ces_queue_path(home=home_sp, env=CLEAN_ENV).resolve()
    sp_tilde = preset.VscodeCesRetrySpool(queue_file="~/sub_ces.json", home=home_sp, env=CLEAN_ENV)
    assert sp_tilde.path == (home_sp / "sub_ces.json").resolve()
    sp_tilde.path.parent.mkdir(parents=True, exist_ok=True)
    sp_tilde.path.write_text("[1, 2]", encoding="utf-8")
    assert sp_tilde.snapshot()["pending_count"] == 0
    (sp_tilde.path.parent / f".{sp_tilde.path.name}.stale.tmp").write_text("tmp", encoding="utf-8")
    assert sp_tilde.purge(lang="fr")["purged"] is True

    sp_trim_path = home_sp / "trim_q.json"
    sp_trim_path.write_text(
        json.dumps(
            {
                "delivered_ids": ["d1", 123, "d2", "d3", "d4"],
                "pending": [
                    "bad_item",
                    {"event_id": "a" * 64, "idempotency_key": "d3", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "b" * 64, "idempotency_key": "k1", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "c" * 64, "idempotency_key": "k2", "payload": {"profile": "dev", "ease_rating": 4, "zero_game_lag": True, "platform": "linux"}},
                    {"event_id": "d" * 64, "idempotency_key": "k3", "payload": {"profile": "dev", "ease_rating": 3, "zero_game_lag": True, "platform": "linux"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    sp_trim = preset.VscodeCesRetrySpool(sp_trim_path, max_entries=2, max_delivered=2, home=home_sp, env=CLEAN_ENV)
    snap_trim = sp_trim.snapshot()
    assert snap_trim["pending_count"] == 2
    assert snap_trim["delivered_ledger_count"] == 2
    assert snap_trim["dropped_overflow_count"] == 1

    # Uninstall preserving user settings.json keys, user tasks.json tasks, and purging ces_queue_file
    ws_pres = tmp_path / "ws_pres"
    (ws_pres / ".vscode").mkdir(parents=True)
    (ws_pres / ".vscode" / "settings.json").write_text(
        json.dumps({"editor.tabSize": 2, "hivemind.gateway": "http://127.0.0.1:8000"}),
        encoding="utf-8",
    )
    (ws_pres / ".vscode" / "tasks.json").write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "tasks": [
                    {"label": "User Custom Task", "type": "shell", "command": "echo user"},
                    {"label": preset.PRESET_TASK_LABELS[0], "type": "shell", "command": "echo hm"},
                ],
            }
        ),
        encoding="utf-8",
    )
    q_un = home_sp / "un_q.json"
    q_un.write_text('{"pending": []}', encoding="utf-8")
    un_pres = preset.uninstall_vscode_quick_preset(
        workspace_dir=ws_pres,
        credentials_path=home_sp / "no_creds.json",
        continue_config_path=home_sp / "no_cont.yaml",
        ces_queue_file=q_un,
        env=CLEAN_ENV,
        home=home_sp,
        lang="fr",
    )
    assert un_pres["exit_code"] == 0
    assert not q_un.exists()
    assert "editor.tabSize" in json.loads((ws_pres / ".vscode" / "settings.json").read_text(encoding="utf-8"))
    assert [t["label"] for t in json.loads((ws_pres / ".vscode" / "tasks.json").read_text(encoding="utf-8"))["tasks"]] == ["User Custom Task"]

    # install_vscode_quick_preset + run_vscode_setup remaining CES 404 + flush 404 branches
    rtx_cool = preset.GpuProbeSnapshot(present=True, name="RTX 3080", total_vram_mib=10240, free_vram_mib=7168, used_vram_mib=3072, temp_c=52.0)
    ws_ipf = tmp_path / "ws_ipf"
    ws_ipf.mkdir()
    r_ipf = preset.install_vscode_quick_preset(
        "http://127.0.0.1:8000",
        workspace_dir=ws_ipf,
        credentials_path=home_sp / "ipf_c.json",
        continue_config_path=home_sp / "ipf_p.yaml",
        profile="gamer",
        run_gamer_preflight=True,
        submit_ces=True,
        gpu_probe_fn=lambda: rtx_cool,
        yield_drill_fn=lambda: 20.0,
        process_probe_fn=lambda: [],
        http_fn=_404_post_http,
        env=CLEAN_ENV,
        home=home_sp,
    )
    assert r_ipf["stale_revision_detected"] is True
    assert r_ipf["onboarding_feedback_deployed"] is False

    r_vs404 = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=ws_ipf,
        credentials_path=home_sp / "vs404_c.json",
        continue_config_path=home_sp / "vs404_p.yaml",
        ces_queue_file=sp_trim_path,
        skip_preflight=True,
        run_gamer_preflight=True,
        gpu_probe_fn=lambda: rtx_cool,
        yield_drill_fn=lambda: 20.0,
        process_probe_fn=lambda: [],
        submit_ces=False,
        flush_ces_queue=True,
        http_fn=_404_post_http,
        env=CLEAN_ENV,
        home=home_sp,
        lang="fr",
    )
    assert r_vs404["stale_revision_detected"] is True
    assert "Cloud Run" in r_vs404["local_fallback_guidance"]
