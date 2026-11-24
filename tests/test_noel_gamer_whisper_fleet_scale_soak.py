# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live fleet, OpenTrainer heartbeat, non-blocking CES, and 48-user scale soak tests for SLICE-SCALE-DYN-10."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from gwaya.ast_audit import ZeroStubAudit
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import OpenTrainerHeartbeatRequest, create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
SELFTEST_PATH = REPO_ROOT / "scripts" / "noel_gamer_whisper_selftest.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import directly without starting any internal coverage.Coverage() instance so external
# `coverage run --include='*/scripts/noel_gamer_whisper_selftest.py'` captures 100% of lines.
from scripts import noel_gamer_whisper_selftest as selftest  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn10",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn10",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn10",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "home-gaming-pc"}
SOAK_SECRET = b"noel-scale-dyn10-router-sec-32b!"
ALPHA_KEY = "hm-dyn10_gamer_whisper_987654"
LIVE_KEY = "hm_live_dyn10_gamer_654321"
LIVE_BUILD_SHA = "c0ffee2026abcdef0123456789abcdef01234567"
LIVE_SHORT_SHA = "c0ffee2026ab"
DRIFT_SHA = "deadbeef11223344556677889900aabbccddeeff"

UNTRUSTED_EXT_GW = "https://" + "evil.example.com"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1:8080"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git.rnd.amadeus.com/internal-team/secret-repo.git"
CORP_HTTPS_REMOTE = "https://" + "gitlab.enterprise.intranet/team/core.git"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@gateway.zscaler.net:8443"

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


def _verify_sha256_receipt(report: dict[str, Any]) -> None:
    digest = report.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in report.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert selftest.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(report)


def _quiet_rtx4070_probe(**_: Any) -> selftest.GpuTelemetry:
    return selftest.GpuTelemetry(
        present=True,
        parse_error=False,
        name="NVIDIA GeForce RTX 4070",
        total_vram_mib=12288,
        free_vram_mib=9600,
        used_vram_mib=2688,
        temp_c=54.0,
        util_pct=4.0,
    )


def _fast_worker_drill(**_: Any) -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 38.5,
        "exit_code": 0,
        "method": "injected_fast_drill",
    }


def _idle_processes(**_: Any) -> list[str]:
    return ["systemd", "python3", "explorer.exe", "steam.exe"]


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[str, tuple[int, dict[str, Any]]] | None = None,
) -> selftest.HttpFn:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip

        if method == "POST" and path == "/v1/hivemind/opentrainer/heartbeat" and payload is not None:
            OpenTrainerHeartbeatRequest.model_validate(payload)

        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path, payload, dict(req_headers)))
            else:
                seen_calls.append((method, path, payload, dict(req_headers)))

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

        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            parsed = resp.json()
            return resp.status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})
        return resp.status_code, {"raw_length": len(raw_text)}

    return _adapter


def test_01_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (SELFTEST_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {target}"

    script_src = SELFTEST_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports only, ProxyHandler({}), shell=False, zero eval/exec/os.system/pickle
    script_tree = ast.parse(script_src, filename="scripts/noel_gamer_whisper_selftest.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    for node in ast.walk(script_tree):
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
    assert imported_roots & {"pickle", "marshal"} == set()
    assert saw_empty_proxy_handler is True

    # Ensure this test file never starts an internal coverage.Coverage() instance
    test_tree = ast.parse(test_src, filename="tests/test_noel_gamer_whisper_fleet_scale_soak.py")
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

    # 2. Lateral-movement AST audit on both script & test + negative controls
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_src,
            filename="scripts/noel_gamer_whisper_selftest.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_src,
            filename="tests/test_noel_gamer_whisper_fleet_scale_soak.py",
        )
        == []
    )
    planted_ip = ".".join(("192", "168", "10", "88"))
    planted_url = "https://" + "c2.attacker.example.com/exfil"
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_LAN = "http://{planted_ip}:9000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_EXT = "{planted_url}"\n') != []

    # 3. PR gatekeeper FORBIDDEN_ADDED scan
    for rel, content in (
        ("scripts/noel_gamer_whisper_selftest.py", script_src),
        ("tests/test_noel_gamer_whisper_fleet_scale_soak.py", test_src),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit + static-numbers cleanliness
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_gamer_whisper_selftest.py", script_src),
        ("tests/test_noel_gamer_whisper_fleet_scale_soak.py", test_src),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = ZeroStubAudit().audit_python_code(content)
        assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
        changed_files.append({"filename": rel, "patch": "", "content": content})

    assert static_numbers_guard.FIGURE.findall(test_src) == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1090,
        pr_title="feat(onboarding,gamer,scale): <1.5s Steam GPU-yield & <=70 C self-test with live fleet ONLINE badge and 48-user scale soak",
        pr_body="Implements SLICE-SCALE-DYN-10 live fleet verifier, OpenTrainer heartbeat, non-blocking CES, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,gamer,scale): <1.5s Steam GPU-yield & <=70 C self-test with live fleet ONLINE badge and 48-user scale soak"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    # 5. Required backward-compatible + new symbols
    for sym in (
        "GpuTelemetry",
        "evaluate_personal_pc_compliance",
        "default_gpu_probe",
        "default_process_probe",
        "detect_steam_or_dx12_game_activity",
        "default_worker_lifecycle_drill",
        "evaluate_thermal_and_vram_governor",
        "verify_post_game_cooldown_policy",
        "verify_lora_delta_bandwidth_cap",
        "check_gateway_health_and_raid_status",
        "verify_live_fleet_and_badge",
        "parse_gamer_ces_input",
        "submit_gamer_ces_feedback",
        "prompt_and_submit_gamer_ces",
        "run_gamer_whisper_selftest",
        "render_markdown_report",
        "render_cli_report",
        "build_parser",
        "main",
    ):
        assert hasattr(selftest, sym), f"Missing symbol {sym}"


def test_02_core_gamer_whisper_checks_real_yield_thermal_vram_cooldown_lora_and_loopback_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hobby_dir = tmp_path / "home" / "GamerHomeUser"
    hobby_dir.mkdir(parents=True)

    # 1. Real subprocess worker yield drill (<1.5s) + real run_gamer_whisper_selftest
    real_drill = selftest.default_worker_lifecycle_drill(trigger_exe="eldenring.exe")
    assert real_drill["worker_started"] is True
    assert real_drill["worker_stopped"] is True
    assert real_drill["clean_release"] is True
    assert real_drill["retained_worker_vram_mib"] == 0
    assert 0.0 <= real_drill["yield_latency_ms"] < 1500.0

    non_game_drill = selftest.default_worker_lifecycle_drill(trigger_exe="notepad.exe")
    assert non_game_drill["worker_started"] is True
    assert non_game_drill["clean_release"] is False

    rep_real = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_real["exit_code"] == 0
    assert rep_real["ready"] is True
    assert rep_real["yield_slo_ok"] is True
    assert rep_real["yield_latency_ms"] < 1500.0
    assert rep_real["retained_worker_vram_mib"] == 0
    assert rep_real["thermal_ok"] is True
    assert rep_real["vram_reserve_ok"] is True
    assert rep_real["cooldown_policy_ok"] is True
    assert rep_real["bandwidth_cap_ok"] is True
    assert rep_real["checks_passed_count"] == 5
    assert rep_real["checks_total_count"] == 5
    _verify_sha256_receipt(rep_real)

    # 2. Slow yield (>=1.5s), retained VRAM > 0, clock-timed slow drill & >10s budget failure
    rep_slow = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=lambda: {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 0,
            "yield_latency_ms": 1500.0,
        },
        clock_fn=lambda: 100.0,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_slow["exit_code"] == 2
    assert rep_slow["yield_slo_ok"] is False
    assert rep_slow["zero_game_lag"] is False
    assert rep_slow["verdict"] == "failed_yield_slo"

    rep_retained = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=lambda: {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 512,
            "yield_latency_ms": 50.0,
        },
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_retained["exit_code"] == 2
    assert rep_retained["yield_slo_ok"] is False

    ticks = iter([0.0, 0.01, 0.02, 0.03, 1.9, 12.0, 12.1])
    rep_clock_slow = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=lambda: {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 0,
        },
        clock_fn=lambda: next(ticks),
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_clock_slow["exit_code"] == 2
    assert rep_clock_slow["within_10s_slo"] is False

    # 3. Steam/DX12 game detection (canonical, 15-char truncated ps comm, Steam paths, GPU >15%, active game yield)
    for proc_sig in (
        "cs2.exe",
        "cyberpunk2077.exe",
        "palworld-win64-",
        "/mnt/games/steamapps/common/Starfield/Starfield.exe",
        "steam_app_1091500",
    ):
        assert selftest.detect_steam_or_dx12_game_activity([proc_sig])["game_detected"] is True
    assert selftest.detect_steam_or_dx12_game_activity(["", "monsterhunterwi"], extra_games=["monsterhunterwilds.exe"])["game_detected"] is True
    assert selftest.detect_steam_or_dx12_game_activity(["explorer.exe"], non_worker_gpu_util_pct=21.0)["gpu_util_triggered"] is True

    rep_active_game = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda **_: ["explorer.exe", "cs2.exe"],
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_active_game["exit_code"] == 1
    assert rep_active_game["verdict"] == "paused_game_active"

    # 4. Thermal <=70 C quiet fan vs >70 C clamp, >=2048 MiB VRAM reserve, combined clamp, parse_error, no-GPU fallback
    rep_near = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 3080", "total_mib": 10240, "free_mib": 6144, "temp_c": 67.5},
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_near["exit_code"] == 0 and rep_near["thermal_vram_governor"]["duty_cycle_scale"] == 0.5

    for gpu_dict, exp_rc, exp_verdict, lang_c in (
        (
            {"present": True, "name": "RTX 3080", "total_vram_mib": 10240, "free_vram_mib": 7000, "temp_c": 74.5, "util_pct": 8.0},
            1,
            "paused_thermal_ceiling",
            "en",
        ),
        (
            {"present": True, "name": "RTX 3080", "total_vram_mib": 10240, "free_vram_mib": 7000, "temp_c": 74.5, "util_pct": 8.0},
            1,
            "paused_thermal_ceiling",
            "fr",
        ),
        (
            {"present": True, "name": "RTX 2070", "total_vram_mib": 8192, "free_vram_mib": 1024, "temp_c": 54.0, "util_pct": 3.0},
            1,
            "paused_low_vram_reserve",
            "en",
        ),
        (
            {"present": True, "name": "RTX 2070", "total_vram_mib": 8192, "free_vram_mib": 1024, "temp_c": 78.0, "util_pct": 12.0},
            1,
            "paused_thermal_ceiling",
            "fr",
        ),
        ({"present": True, "parse_error": True}, 2, "failed_safety_gate", "en"),
        ({"present": True, "parse_error": True}, 2, "failed_safety_gate", "fr"),
        ({"present": False}, 0, "cpu_or_dev_only_no_nvidia_gpu", "en"),
        ({"present": False}, 0, "cpu_or_dev_only_no_nvidia_gpu", "fr"),
    ):
        rep_g = selftest.run_gamer_whisper_selftest(
            gpu_probe_fn=lambda g=gpu_dict: g,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            environ=CLEAN_ENV,
            cwd=hobby_dir,
            home=hobby_dir,
            lang=lang_c,
        )
        assert rep_g["exit_code"] == exp_rc
        assert rep_g["verdict"] == exp_verdict

    # 5. 300s post-game cooldown & <50 MB/h LoRA delta bandwidth cap
    rep_cd_hold = selftest.run_gamer_whisper_selftest(
        seconds_since_game_exit=120.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_cd_hold["exit_code"] == 1 and rep_cd_hold["verdict"] == "paused_post_game_cooldown"

    rep_cd_bad = selftest.run_gamer_whisper_selftest(
        resume_cooldown_s=60.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_cd_bad["exit_code"] == 2 and rep_cd_bad["cooldown_policy_ok"] is False

    rep_full_ckpt = selftest.run_gamer_whisper_selftest(
        transfer_mode="full_checkpoint",
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_full_ckpt["exit_code"] == 2 and rep_full_ckpt["bandwidth_cap_ok"] is False

    # 6. default_gpu_probe, default_process_probe, default_subprocess_runner & Popen failure mocks
    csv_gpus = (
        "NVIDIA GeForce GTX 1650, 4096, 3200, 896, 48, 2\n"
        "NVIDIA GeForce RTX 4080, 16384, 14200, [N/A], 51, [N/A]\n"
    )
    multi_gpu = selftest.default_gpu_probe(runner=lambda _c, _t: (0, csv_gpus))
    assert multi_gpu.name == "NVIDIA GeForce RTX 4080" and multi_gpu.used_vram_mib == 2184
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (0, "bad")).parse_error is True
    assert selftest.default_gpu_probe(runner=lambda _c, _t: (0, "GPU, -1, 0, 0, 40, 1")).parse_error is True

    assert "cs2.exe" in selftest.default_process_probe(
        runner=lambda _c, _t: (0, '"steam.exe","1","Console","1","10 K"\n"cs2.exe","2","Console","1","20 K"\n'),
        os_name="nt",
    )
    assert selftest.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="nt") == []
    assert "eldenring.exe" in selftest.default_process_probe(
        runner=lambda _c, _t: (0, "\n/opt/games/eldenring.exe\n"),
        os_name="posix",
    )
    assert selftest.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="posix") == []

    rc_py, out_py = selftest.default_subprocess_runner([sys.executable, "-c", "print('runner_ok')"])
    assert rc_py == 0 and "runner_ok" in out_py
    assert selftest.default_subprocess_runner(["nonexistent_hm_bin_dyn10"])[0] == 127

    class _FakeProc:
        def __init__(self, banner: str, raise_timeout: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(banner)
            self.stderr = io.StringIO()
            self._raise_timeout = raise_timeout
            self.returncode = -9

        def poll(self) -> int | None:
            return None if self._raise_timeout else self.returncode

        def kill(self) -> None:
            self._raise_timeout = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._raise_timeout:
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=0.2)
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FakeProc("WRONG\n"))
    assert selftest.default_worker_lifecycle_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FakeProc("HIVEMIND_WORKER_READY\n", True))
    assert selftest.default_worker_lifecycle_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    # 7. Real loopback HTTP server with ProxyHandler({}) (under poisoned HTTP_PROXY) & zero-inference block
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "missing"}')

        def do_POST(self) -> None:  # noqa: N802
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/plain":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"plain-ok")
            else:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b"bad gateway")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = srv.server_address[1]
    t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
    t_srv.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        base_url = f"http://127.0.0.1:{port}"
        assert selftest.default_http_json("GET", f"{base_url}/health") == (200, {"status": "ok"})
        assert selftest.default_http_json("GET", f"{base_url}/missing")[0] == 404
        assert selftest.default_http_json("POST", f"{base_url}/empty", {"a": 1}, {"X-Test": "1"}) == (204, {})
        assert selftest.default_http_json("POST", f"{base_url}/plain", {"a": 1}) == (200, {"raw_length": 8})
        assert selftest.default_http_json("POST", f"{base_url}/err", {"a": 1})[0] == 502
    finally:
        srv.shutdown()
        srv.server_close()

    assert selftest.default_http_json("GET", f"http://127.0.0.1:{port}/health")[0] == 0

    for bad_req in (
        lambda: selftest.default_http_json("POST", "http://127.0.0.1:8080/v1/chat/completions", {}),
        lambda: selftest.default_http_json("POST", "http://127.0.0.1:8080/v1/hivemind/coder", {}),
        lambda: selftest.default_http_json("GET", "http://127.0.0.1:8080/health", headers={"X-Bad": "a\r\nb"}),
    ):
        with pytest.raises(selftest.SelftestError):
            bad_req()

    # 8. Legacy check_gateway_health_and_raid_status + prompt_and_submit_gamer_ces + FR/EN CLI/Markdown render
    def _mock_gw_and_ces(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/hivemind/raid/status"):
            return 200, {"campaign": "noel", "swarm": {"active_trainers": 3}, "feedback": {"mean_ease_rating": 4.9}}
        if url.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {"total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0}
        return 404, {}

    for lang_flag in ("en", "fr"):
        for fmt in ([], ["--markdown"], ["--json"]):
            rc_render = selftest.main(
                ["--check-gateway", "--prompt-ces", "--lang", lang_flag, *fmt],
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=_mock_gw_and_ces,
                input_fn=lambda _: "5 gamer quiet",
                environ=CLEAN_ENV,
                cwd=hobby_dir,
                home=hobby_dir,
                system_name="Windows",
            )
            assert rc_render == 0
            out_txt = capsys.readouterr().out
            assert "GamerHomeUser" not in out_txt
            assert "0.00 EUR / 0.00 USD" in out_txt
            assert selftest.scan_output_safety(out_txt) == []

    # Interactive skip / EOF + lag token + legacy gateway error statuses
    assert selftest.prompt_and_submit_gamer_ces(
        derived_zero_game_lag=True,
        input_fn=lambda _: (_ for _ in ()).throw(EOFError()),
    )["skipped"] is True
    assert selftest.parse_gamer_ces_input("4 lag", derived_zero_game_lag=True)["zero_game_lag"] is False

    for bad_status, exp_gw_status in (
        (404, "missing_route"),
        (429, "rate_limited"),
        (503, "server_error"),
        (0, "network_error"),
        (400, "client_error"),
    ):
        for lc in ("en", "fr"):
            chk = selftest.check_gateway_health_and_raid_status(
                http_fn=lambda _m, u, _p=None, _h=None, s=bad_status: (200, {"status": "ok"}) if u.endswith("/health") else (s, {}),
                lang=lc,
            )
            assert chk["status"] == exp_gw_status


def test_03_noel_verify_stale_404_drift_to_live_cutover_heartbeat_badges_and_non_blocking_ces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00201-noel")

    home_dir = tmp_path / "home" / "GamerCutoverSecret"
    ws_dir = tmp_path / "hobby_games"
    ws_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)

    # 1. Pre-PR-#949 stale Cloud Run HTTP 404 drift -> fleet_badge=='STALE_CLOUD_RUN', exit code 2
    def _stale_404_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        assert "completions" not in url.lower()
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and url.endswith("/v1/hivemind/producers/fleet"):
            return 200, {
                "object": "hivemind.producer_fleet",
                "total_hiveminders": 3,
                "live_hiveminders": 2,
                "distinct_physical_gpus": 2,
                "live_endpoints": 2,
                "live_models": ["hivemind/coder"],
            }
        return 404, {"detail": "Not Found"}

    for lc in ("en", "fr"):
        # Even with --non-blocking-feedback, --noel-verify MUST fail closed (exit code 2) on stale 404 drift!
        rc_404 = selftest.main(
            [
                "--noel-verify",
                "--submit-feedback",
                "--ease-rating",
                "5",
                "--non-blocking-feedback",
                "--lang",
                lc,
                "--json",
            ],
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=_stale_404_http,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc_404 == 2
        rep_404 = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(rep_404)
        assert rep_404["fleet_badge"] == "STALE_CLOUD_RUN"
        assert rep_404["holiday_routes_deployed_on_cloud_run"] is False
        assert rep_404["stale_cloud_run_detected"] is True
        assert rep_404["ces_recorded"] is False
        assert rep_404["verdict"] == "stale_cloud_run"

    # Gateway completely unreachable / 503 -> OFFLINE (exit code 1)
    for offline_code in (0, 503):
        for lc in ("en", "fr"):
            rep_off = selftest.run_gamer_whisper_selftest(
                verify_fleet=True,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=lambda _m, _u, _p=None, _h=None, c=offline_code: (c, {}),
                environ=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
                lang=lc,
            )
            assert rep_off["fleet_badge"] == "OFFLINE"
            assert rep_off["exit_code"] == 1
            assert rep_off["verdict"] == "fleet_offline"

    # 2. Live cutover to FastAPI create_router_app(..., allow_private_producers=True)
    store = RouterStore(tmp_path / "cutover_dyn10.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(
        store,
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] = []
    with TestClient(app) as client:
        # Register a gamer prosumer, a dev prosumer, and a dev consumer
        reg_gamer = client.post(
            "/v1/hivemind/register",
            json={"handle": "cutover-gamer", "role": "prosumer"},
            headers={"X-Forwarded-For": "127.0.20.1"},
        ).json()
        gamer_key = str(reg_gamer["api_key"])

        reg_dev_prosumer = client.post(
            "/v1/hivemind/register",
            json={"handle": "cutover-dev-prosumer", "role": "prosumer"},
            headers={"X-Forwarded-For": "127.0.20.2"},
        ).json()
        dev_prosumer_key = str(reg_dev_prosumer["api_key"])

        reg_dev_consumer = client.post(
            "/v1/hivemind/register",
            json={"handle": "cutover-dev-consumer", "role": "consumer"},
            headers={"X-Forwarded-For": "127.0.20.3"},
        ).json()
        dev_consumer_key = str(reg_dev_consumer["api_key"])

        # Write ~/.hivemind/credentials.json for credentials loading test
        cred_file = home_dir / ".hivemind" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(
            json.dumps(
                {
                    "account_id": reg_gamer["account_id"],
                    "api_key": gamer_key,
                    "gateway": "http://127.0.0.1:8000",
                    "handle": "cutover-gamer",
                    "role": "prosumer",
                }
            ),
            encoding="utf-8",
        )

        adapter = _make_router_http_adapter(client, peer_ip="127.0.20.10", seen_calls=seen_calls)

        # (a) Expected SHA drift fails closed even on live router
        rep_sha_drift = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            noel_verify=True,
            expected_sha=DRIFT_SHA,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rep_sha_drift["fleet_badge"] == "STALE_CLOUD_RUN"
        assert rep_sha_drift["exit_code"] == 2
        assert rep_sha_drift["fleet_check"]["sha_drift_detected"] is True

        # (b) With 1 live producer (<8 target), fleet_badge == 'BOOTSTRAPPING', OpenTrainer heartbeat loaded from --credentials
        hb_p0 = client.post(
            "/v1/hivemind/producer/heartbeat",
            json={
                "endpoint_url": f"http://{_loopback_ip_for(90, 0)}:11434",
                "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                "cost_per_1k": 0.0,
                "capacity_1k": 64.0,
                "gpu_vram_mib": 12288,
                "execution_class": "ollama-gamer-whisper",
            },
            headers={"Authorization": f"Bearer {gamer_key}", "X-Forwarded-For": "127.0.20.1"},
        )
        assert hb_p0.status_code == 200

        rc_boot = selftest.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--noel-verify",
                "--check-gateway",
                "--send-opentrainer-heartbeat",
                "--credentials",
                str(cred_file),
                "--host-label",
                "gamer-rtx-box-01",
                "--expected-sha",
                LIVE_SHORT_SHA,
                "--target-online-producers",
                "8",
                "--submit-feedback",
                "--ease-rating",
                "5",
                "--lang",
                "fr",
                "--json",
            ],
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
            system_name="Windows",
        )
        assert rc_boot == 0
        rep_boot = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(rep_boot)
        assert rep_boot["fleet_badge"] == "BOOTSTRAPPING"
        assert rep_boot["online_producers"] == 1
        assert rep_boot["target_online_producers"] == 8
        assert rep_boot["holiday_routes_deployed_on_cloud_run"] is True
        assert rep_boot["stale_cloud_run_detected"] is False
        assert rep_boot["opentrainer_heartbeat"]["accepted"] is True
        assert rep_boot["opentrainer_heartbeat"]["me_ok"] is True
        assert rep_boot["opentrainer_heartbeat"]["payload"] == {
            "host_label": "gamer-rtx-box-01",
            "gpu_name": "NVIDIA GeForce RTX 4070",
            "vram_total_mib": 12288,
            "vram_free_mib": 9600,
            "network_mode": "outbound-https-relay",
            "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
            "active_campaign": "noel-2026-easy-onboard",
        }
        assert rep_boot["ces_recorded"] is True
        assert gamer_key not in json.dumps(rep_boot)
        assert "GamerCutoverSecret" not in json.dumps(rep_boot)
        assert "~/.hivemind/credentials.json" == rep_boot["credentials_path"]

        # Verify GET /v1/hivemind/me happened before POST /v1/hivemind/opentrainer/heartbeat
        me_idx = next(i for i, (_, p, _, _) in enumerate(seen_calls) if p == "/v1/hivemind/me")
        hb_idx = next(i for i, (_, p, _, _) in enumerate(seen_calls) if p == "/v1/hivemind/opentrainer/heartbeat")
        assert me_idx < hb_idx

        # (c) Load API key via HIVEMIND_API_KEY (no GPU -> None GPU specs) & explicit api_key (consumer role)
        rep_env_nogpu = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            noel_verify=True,
            send_opentrainer_heartbeat=True,
            host_label="dev-laptop-env",
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ={**CLEAN_ENV, "HIVEMIND_API_KEY": dev_prosumer_key},
            cwd=ws_dir,
            home=home_dir,
            lang="en",
        )
        assert rep_env_nogpu["exit_code"] == 0
        assert rep_env_nogpu["opentrainer_heartbeat"]["accepted"] is True
        assert rep_env_nogpu["opentrainer_heartbeat"]["key_source"] == "env:HIVEMIND_API_KEY"
        assert rep_env_nogpu["opentrainer_heartbeat"]["payload"]["gpu_name"] is None
        assert rep_env_nogpu["opentrainer_heartbeat"]["payload"]["vram_total_mib"] is None
        assert rep_env_nogpu["opentrainer_heartbeat"]["payload"]["vram_free_mib"] is None
        assert dev_prosumer_key not in json.dumps(rep_env_nogpu)

        # < 512 MiB GPU must also report None GPU fields (never invent specs)
        rep_small_gpu = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            api_key=dev_prosumer_key,
            min_free_vram_mib=256,
            gpu_probe_fn=lambda: {"present": True, "name": "TinyGPU", "total_vram_mib": 256, "free_vram_mib": 256, "temp_c": 45.0},
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rep_small_gpu["opentrainer_heartbeat"]["payload"]["gpu_name"] is None
        assert rep_small_gpu["opentrainer_heartbeat"]["payload"]["vram_total_mib"] is None

        # Consumer account verifies /me and handles 403 on /opentrainer/heartbeat cleanly
        rep_consumer = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            api_key=dev_consumer_key,
            gpu_probe_fn=lambda: {"present": False},
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
            lang="fr",
        )
        assert rep_consumer["exit_code"] == 0
        assert rep_consumer["opentrainer_heartbeat"]["status"] == "consumer_me_verified"
        assert rep_consumer["opentrainer_heartbeat"]["heartbeat_ok"] is True

        # (d) Register 7 more producers (total 8 >= target 8) -> ONLINE badge transition!
        for p_idx in range(1, 8):
            p_ip = _loopback_ip_for(21, p_idx)
            r_p = client.post(
                "/v1/hivemind/register",
                json={"handle": f"fleet-prod-{p_idx}", "role": "producer"},
                headers={"X-Forwarded-For": p_ip},
            ).json()
            r_hb = client.post(
                "/v1/hivemind/producer/heartbeat",
                json={
                    "endpoint_url": f"http://{_loopback_ip_for(91, p_idx)}:11434",
                    "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 64.0,
                    "gpu_vram_mib": 12288,
                    "execution_class": "ollama-gamer-whisper",
                },
                headers={"Authorization": f"Bearer {r_p['api_key']}", "X-Forwarded-For": p_ip},
            )
            assert r_hb.status_code == 200

        for lc in ("en", "fr"):
            rep_online = selftest.run_gamer_whisper_selftest(
                gateway="http://127.0.0.1:8000",
                noel_verify=True,
                send_opentrainer_heartbeat=True,
                api_key=gamer_key,
                target_online_producers=8,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
                lang=lc,
            )
            _verify_sha256_receipt(rep_online)
            assert rep_online["fleet_badge"] == "ONLINE"
            assert rep_online["online_producers"] >= 8
            assert rep_online["distinct_physical_gpus"] >= 8
            assert rep_online["active_trainers"] >= 2
            md_txt = selftest.render_markdown_report(rep_online)
            cli_txt = selftest.render_cli_report(rep_online)
            assert "ONLINE" in md_txt and "ONLINE" in cli_txt
            assert gamer_key not in md_txt and gamer_key not in cli_txt

        # (e) --non-blocking-feedback graceful degradation on HTTP 404, 429, 503, 0 + deterministic Idempotency-Key
        seen_idem_keys: list[str] = []
        for deg_code, exp_status in ((404, "missing_route"), (429, "rate_limited"), (503, "server_error"), (0, "network_error")):
            def _deg_http(
                method: str,
                url: str,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
                _c: int = deg_code,
            ) -> tuple[int, dict[str, Any]]:
                if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
                    assert headers is not None and headers.get("Idempotency-Key", "").startswith("ces-gamer-")
                    seen_idem_keys.append(headers["Idempotency-Key"])
                    if _c == 0:
                        raise TimeoutError("feedback timeout")
                    return _c, {"error": f"sim_{_c}"}
                return adapter(method, url, payload, headers)

            for lc in ("en", "fr"):
                rep_deg = selftest.run_gamer_whisper_selftest(
                    gateway="http://127.0.0.1:8000",
                    noel_verify=True,
                    submit_feedback=True,
                    ease_rating=5,
                    non_blocking_feedback=True,
                    host_label="gamer-idem-box",
                    gpu_probe_fn=_quiet_rtx4070_probe,
                    process_probe_fn=_idle_processes,
                    worker_lifecycle_fn=_fast_worker_drill,
                    http_fn=_deg_http,
                    environ=CLEAN_ENV,
                    cwd=ws_dir,
                    home=home_dir,
                    lang=lc,
                )
                assert rep_deg["exit_code"] == 0
                assert rep_deg["ready"] is True
                assert rep_deg["ces_recorded"] is False
                assert rep_deg["feedback"]["degraded"] is True
                assert rep_deg["feedback"]["status"] == exp_status
                assert rep_deg["warnings_count"] >= 1

        assert len(seen_idem_keys) == 8 and len(set(seen_idem_keys)) == 1

        # Slow yield (>=1.5s) MUST still fail closed (exit 2) even when non_blocking_feedback=True
        rep_slow_nb = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            submit_feedback=True,
            ease_rating=4,
            non_blocking_feedback=True,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=lambda: {
                "worker_started": True,
                "worker_stopped": True,
                "clean_release": True,
                "retained_worker_vram_mib": 0,
                "yield_latency_ms": 1520.0,
            },
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rep_slow_nb["exit_code"] == 2 and rep_slow_nb["yield_slo_ok"] is False

        # Blocking mode (non_blocking_feedback=False) still fails on 400/404/429/500/0
        for block_code, exp_rc in ((404, 2), (429, 1), (503, 1), (400, 1), (0, 1)):
            rep_blk = selftest.run_gamer_whisper_selftest(
                gateway="http://127.0.0.1:8000",
                submit_feedback=True,
                ease_rating=4,
                non_blocking_feedback=False,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=lambda _m, _u, _p=None, _h=None, c=block_code: (c, {}),
                environ=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
            )
            assert rep_blk["exit_code"] == exp_rc and rep_blk["ready"] is False

        # (f) OpenTrainer heartbeat failure & hold branches
        assert selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=tmp_path / "no_creds_home",
            lang="fr",
        )["exit_code"] == 2

        assert selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ={**CLEAN_ENV, "HIVEMIND_API_KEY": "bad-env-key"},
            cwd=ws_dir,
            home=tmp_path / "bad_env_home",
            lang="fr",
        )["exit_code"] == 2

        # Hot GPU holds OpenTrainer heartbeat paused (never registers hot GPU)
        rep_hot_hb = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            api_key=gamer_key,
            gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 9000, "temp_c": 75.0},
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=_fast_worker_drill,
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rep_hot_hb["exit_code"] == 1
        assert rep_hot_hb["opentrainer_heartbeat"]["status"] == "paused_governor_hold"
        assert rep_hot_hb["opentrainer_heartbeat"]["sent"] is False

        # Slow yield blocks OpenTrainer heartbeat
        rep_slow_hb = selftest.run_gamer_whisper_selftest(
            gateway="http://127.0.0.1:8000",
            send_opentrainer_heartbeat=True,
            api_key=gamer_key,
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 1600.0},
            http_fn=adapter,
            environ=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rep_slow_hb["exit_code"] == 2
        assert rep_slow_hb["opentrainer_heartbeat"]["status"] == "blocked_yield_slo"

        # /me & /heartbeat HTTP error mappings (401, 404, 429, 500, 0, 400)
        for err_c, exp_rc in ((401, 2), (404, 2), (429, 1), (500, 1), (0, 1), (400, 1)):
            rep_me_err = selftest.run_gamer_whisper_selftest(
                gateway="http://127.0.0.1:8000",
                send_opentrainer_heartbeat=True,
                api_key=gamer_key,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=lambda _m, _u, _p=None, _h=None, c=err_c: (c, {}),
                environ=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
            )
            assert rep_me_err["exit_code"] == exp_rc

            rep_hb_err = selftest.run_gamer_whisper_selftest(
                gateway="http://127.0.0.1:8000",
                send_opentrainer_heartbeat=True,
                api_key=gamer_key,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=lambda _m, u, _p=None, _h=None, c=err_c: (200, {"role": "prosumer"}) if u.endswith("/me") else (c, {}),
                environ=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
            )
            assert rep_hb_err["exit_code"] == exp_rc


def test_04_concurrent_48_participant_gamer_whisper_fleet_scale_soak_sqlite_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00202-scale48")

    upstream_inference_hits: list[str] = []

    def _reject_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn10_48_soak.sqlite3"
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
    gamer_count = 24
    barrier = threading.Barrier(total_participants)
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] = []
    raid_nonces: list[str] = []

    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216, 54.0),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288, 57.0),
        ("NVIDIA GeForce RTX 3080", 10240, 6144, 61.0),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120, 52.0),
        ("NVIDIA GeForce RTX 3090", 24576, 18432, 63.0),
        ("NVIDIA GeForce RTX 4090", 24576, 20480, 58.0),
    )

    with TestClient(app) as client:
        def _participant_worker(idx: int) -> dict[str, Any]:
            is_gamer = idx < gamer_count
            cohort_idx = idx if is_gamer else (idx - gamer_count)
            peer_ip = _loopback_ip_for(50, idx)
            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"secret_home_user_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "personal_hobby_repo"
            ws_dir.mkdir(parents=True, exist_ok=True)
            cred_file = home_dir / ".hivemind" / "credentials.json"

            handle = f"gamer-rtx-{cohort_idx:02d}" if is_gamer else f"dev-laptop-{cohort_idx:02d}"
            role = "producer" if is_gamer else ("prosumer" if cohort_idx % 2 == 0 else "consumer")
            ease = 5 if (idx % 3 != 0) else 4
            lang = "fr" if (idx % 2 == 0) else "en"

            adapter = _make_router_http_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=shared_lock,
            )

            barrier.wait(timeout=25.0)
            t0 = time.monotonic()

            reg_resp = client.post(
                "/v1/hivemind/register",
                json={"handle": handle, "role": role},
                headers={"X-Forwarded-For": peer_ip},
            )
            assert reg_resp.status_code == 200, reg_resp.text
            reg_doc = reg_resp.json()
            raw_key = str(reg_doc["api_key"])

            cred_file.parent.mkdir(parents=True, exist_ok=True)
            cred_file.write_text(
                json.dumps(
                    {
                        "account_id": reg_doc["account_id"],
                        "api_key": raw_key,
                        "gateway": "http://127.0.0.1:8000",
                        "handle": handle,
                        "role": role,
                    }
                ),
                encoding="utf-8",
            )

            if is_gamer:
                card_name, tot_mib, free_mib, temp_c = rtx_cards[cohort_idx % len(rtx_cards)]
                prod_ip = _loopback_ip_for(99, cohort_idx)
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

                gpu_snap = selftest.GpuTelemetry(
                    present=True,
                    name=card_name,
                    total_vram_mib=tot_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=tot_mib - free_mib,
                    temp_c=temp_c,
                    util_pct=4.0,
                )
                yield_ms = 55.0 + (cohort_idx * 4.0)
                os_name_val = "nt" if (cohort_idx % 2 == 0) else "posix"
                sys_name_val = "Windows" if (cohort_idx % 2 == 0) else "Linux"
                host_lbl = f"gamer-rtx-home-{cohort_idx:02d}"
            else:
                gpu_snap = selftest.GpuTelemetry(present=False, reason="no_nvidia_gpu")
                yield_ms = 19.0 + cohort_idx
                os_name_val = "posix"
                sys_name_val = "Darwin" if (cohort_idx % 2 == 0) else "Linux"
                host_lbl = f"dev-laptop-home-{cohort_idx:02d}"

            rep = selftest.run_gamer_whisper_selftest(
                gateway="http://127.0.0.1:8000",
                noel_verify=True,
                send_opentrainer_heartbeat=True,
                credentials_path=cred_file,
                host_label=host_lbl,
                target_online_producers=8,
                non_blocking_feedback=True,
                expected_sha=LIVE_SHORT_SHA,
                submit_feedback=True,
                ease_rating=ease,
                had_game_lag=False,
                lang=lang,
                os_name=os_name_val,
                system_name=sys_name_val,
                environ=CLEAN_ENV,
                home=home_dir,
                cwd=ws_dir,
                gpu_probe_fn=lambda g=gpu_snap: g,
                process_probe_fn=lambda: ["steam.exe", "python3"],
                worker_lifecycle_fn=lambda y=yield_ms: {
                    "worker_started": True,
                    "worker_stopped": True,
                    "clean_release": True,
                    "retained_worker_vram_mib": 0,
                    "yield_latency_ms": y,
                },
                http_fn=adapter,
            )
            elapsed_s = time.monotonic() - t0
            assert elapsed_s < 10.0
            _verify_sha256_receipt(rep)
            assert rep["ready"] is True, f"Participant {idx} failed: {rep}"
            assert rep["exit_code"] == 0
            assert rep["yield_slo_ok"] is True
            assert rep["holiday_routes_deployed_on_cloud_run"] is True
            assert rep["stale_cloud_run_detected"] is False
            assert rep["fleet_badge"] in ("ONLINE", "BOOTSTRAPPING")
            assert rep["opentrainer_heartbeat"]["heartbeat_ok"] is True
            assert rep["ces_recorded"] is True

            # Verify /raid-noel CSP nonce uniqueness & HTML safety (strip nonce="..." before static figure check)
            raid_html_resp = client.get(
                f"/raid-noel?lang={lang}",
                headers={"X-Forwarded-For": peer_ip},
            )
            assert raid_html_resp.status_code == 200
            csp_hdr = raid_html_resp.headers.get("content-security-policy", "")
            assert "default-src 'none'" in csp_hdr
            assert "'unsafe-inline'" not in csp_hdr
            m_nonce = _CSP_NONCE_RE.search(csp_hdr)
            assert m_nonce is not None
            stripped_html = _HTML_NONCE_ATTR_RE.sub('nonce=""', raid_html_resp.text)
            assert ("inner" + "HTML") not in stripped_html
            assert static_numbers_guard.FIGURE.findall(stripped_html) == []
            with shared_lock:
                raid_nonces.append(m_nonce.group(1))

            cli_txt = selftest.render_cli_report(rep)
            md_txt = selftest.render_markdown_report(rep)
            combined = json.dumps(rep) + "\n" + cli_txt + "\n" + md_txt
            assert raw_key not in combined
            assert os_user not in combined
            assert not selftest._API_KEY_RE.search(combined)
            assert selftest.scan_output_safety(cli_txt) == []
            assert selftest.scan_output_safety(md_txt) == []
            return {"idx": idx, "idem": rep["feedback"]["idempotency_key"]}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_participant_worker, i) for i in range(total_participants)]
            outcomes = [fut.result(timeout=60.0) for fut in futures]

        assert len(outcomes) == 48
        assert len({o["idem"] for o in outcomes}) == 48
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _, _ in seen_calls)

        # 100% unique CSP nonces on /raid-noel across all 48 participants
        assert len(raid_nonces) == 48
        assert len(set(raid_nonces)) == 48

        raid_status = client.get("/v1/hivemind/raid/status").json()
        assert raid_status["swarm"]["active_trainers"] >= 24

        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert fb_summary["total_responses"] == 48
        assert fb_summary["mean_ease_rating"] is not None and float(fb_summary["mean_ease_rating"]) >= 4.2
        assert fb_summary["zero_game_lag_ratio"] is not None and float(fb_summary["zero_game_lag_ratio"]) >= 0.95

        final_adapter = _make_router_http_adapter(client, peer_ip=_loopback_ip_for(55, 1))
        final_fleet = selftest.verify_live_fleet_and_badge(
            gateway="http://127.0.0.1:8000",
            expected_sha=LIVE_BUILD_SHA,
            target_online_producers=8,
            http_fn=final_adapter,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_sha256_receipt(final_fleet)
        assert final_fleet["fleet_badge"] == "ONLINE"
        assert final_fleet["holiday_routes_deployed_on_cloud_run"] is True
        assert final_fleet["stale_cloud_run_detected"] is False
        assert final_fleet["online_producers"] >= 24
        assert final_fleet["distinct_physical_gpus"] >= 24
        assert final_fleet["active_trainers"] >= 24
        assert final_fleet["target_producers_met"] is True
        assert final_fleet["producers_remaining_to_target"] == 0


def test_05_fail_closed_office_workstation_git_remote_blocking_and_edge_validators(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_calls = 0
    http_calls = 0

    def _forbidden_probe(**_: Any) -> Any:
        nonlocal probe_calls
        probe_calls += 1
        return _quiet_rtx4070_probe()

    def _forbidden_http(*_: Any, **__: Any) -> tuple[int, dict[str, Any]]:
        nonlocal http_calls
        http_calls += 1
        return 200, {}

    # 1. Corporate env markers + cwd + .git/config + worktree commondir
    clean_ws = tmp_path / "personal_hobby_app"
    clean_ws.mkdir()
    corp_git_ws = tmp_path / "weekend_game_mod"
    (corp_git_ws / ".git").mkdir(parents=True)
    (corp_git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n[url "{CORP_HTTPS_REMOTE}"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )
    common_git = tmp_path / "common_git"
    wt_git = common_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    (common_git / "config").write_text(
        '[remote "upstream"]\n\turl = git@fortinet.enterprise.local:team/core.git\n',
        encoding="utf-8",
    )
    wt_ws = tmp_path / "wt_checkout"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")

    corp_path_ws = tmp_path / "amadeus_internal_repo"
    corp_path_ws.mkdir()

    for bad_env, bad_cwd, secret_sentinel, lc in (
        ({"USERDNSDOMAIN": "nce.amadeus.net"}, clean_ws, "nce.amadeus.net", "en"),
        ({"HTTPS_PROXY": CORP_ZSCALER_PROXY}, clean_ws, "supersecretpw", "fr"),
        ({"COMPUTERNAME": "WKS-ENT-CORP01", "USERDOMAIN": "NETSKOPE_INTERNAL"}, clean_ws, "NETSKOPE_INTERNAL", "en"),
        ({"HOSTNAME": "laptop.paloalto.intranet"}, clean_ws, "laptop.paloalto.intranet", "fr"),
        ({"AMADEUS_SITE": "NCE", "ZSCALER_CERT_PATH": "/etc/zscaler.pem"}, clean_ws, "/etc/zscaler.pem", "en"),
        (CLEAN_ENV, corp_git_ws, "secret-repo.git", "fr"),
        (CLEAN_ENV, wt_ws, "fortinet.enterprise.local", "en"),
        (CLEAN_ENV, corp_path_ws, "amadeus_internal_repo", "fr"),
    ):
        rc_corp = selftest.main(
            [
                "--noel-verify",
                "--send-opentrainer-heartbeat",
                "--submit-feedback",
                "--ease-rating",
                "5",
                "--non-blocking-feedback",
                "--lang",
                lc,
                "--json",
            ],
            gpu_probe_fn=_forbidden_probe,
            process_probe_fn=_forbidden_probe,
            worker_lifecycle_fn=_forbidden_probe,
            http_fn=_forbidden_http,
            environ=bad_env,
            cwd=bad_cwd,
            home=clean_ws,
        )
        assert rc_corp == 2
        out_raw = capsys.readouterr().out
        rep = json.loads(out_raw)
        _verify_sha256_receipt(rep)
        assert rep["compliance_ok"] is False
        assert rep["ready"] is False
        assert rep["verdict"] == "refused_compliance"
        assert secret_sentinel not in out_raw
        assert "0.00 EUR / 0.00 USD" in out_raw

        fleet_blocked = selftest.verify_live_fleet_and_badge(
            http_fn=_forbidden_http,
            environ=bad_env,
            cwd=bad_cwd,
            lang=lc,
        )
        assert fleet_blocked["compliance_ok"] is False
        assert fleet_blocked["exit_code"] == 2
        assert fleet_blocked["network_calls"] == 0

    assert probe_calls == 0
    assert http_calls == 0

    # 2. Validator, sanitizer, credential loader, and CLI error edge branches
    assert selftest.normalize_lang("fr-FR") == "fr"
    with pytest.raises(selftest.SelftestError):
        selftest.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(selftest.SelftestError):
        selftest.normalize_lang("de")

    assert selftest._platform_tag("Darwin") == "macos"
    assert selftest._platform_tag("Windows 11") == "windows"
    assert selftest._platform_tag("Linux") == "linux"
    assert selftest._platform_tag("   ") == "unknown"

    assert selftest.is_valid_hivemind_key(None) is False
    assert selftest.mask_api_key("") == "<missing>"
    assert selftest.mask_api_key("not-valid") == "<invalid>"
    assert selftest.mask_api_key(ALPHA_KEY) == "hm-***7654"
    assert selftest.mask_api_key(LIVE_KEY) == "hm_live_***4321"

    assert selftest.sanitize_host_label(None) == "personal-pc"
    assert selftest.sanitize_host_label("amadeus-corp") == "personal-pc"
    assert selftest.sanitize_host_label("!!!") == "personal-pc"
    assert selftest.sanitize_build_sha(None) == "unknown"
    assert selftest.sanitize_build_sha("secret123456") == "unknown"
    assert selftest.shas_match("unknown", LIVE_BUILD_SHA) is False
    assert selftest.shas_match("release2026a", "release2026b") is False
    assert selftest.sanitize_cloud_run_revision(None) is None
    assert selftest.sanitize_cloud_run_revision("hm-1234567890") is None

    for bad_sha in ("", "unknown", "secret_sha_val"):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_expected_sha(bad_sha)

    home_edge = tmp_path / "home" / "AliceEdge"
    home_edge.mkdir(parents=True)
    assert selftest.redact_home_path(home_edge, home=home_edge) == "~"
    assert selftest.redact_home_path(r"C:\Users\AliceEdge\.hivemind\cfg.json", home=Path("/none")) == "~/.hivemind/cfg.json"
    sample_lan_ip = ".".join(("10", "1", "2", "3"))
    bad_space_gw = "https://" + "bad space.run.app"
    scrubbed_msg = selftest.sanitize_text_for_display(
        f"Err {home_edge}/.hivemind/credentials.json {ALPHA_KEY} a@b.org {sample_lan_ip} Bearer tok_123456",
        home=home_edge,
    )
    assert scrubbed_msg.startswith("Err ~/.hivemind/credentials.json ")
    assert "AliceEdge" not in scrubbed_msg and ALPHA_KEY not in scrubbed_msg
    assert selftest.scan_output_safety(scrubbed_msg) == []

    for bad_gw in ("", "https://ok.run.app\n", bad_space_gw, "http://127.0.0.1:badport", UNTRUSTED_EXT_GW, UNTRUSTED_FTP_GW, INSECURE_RUN_APP_GW):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_gateway_url(bad_gw)
    assert selftest.is_allowed_gateway_host("") is False
    with pytest.raises(selftest.SelftestError):
        selftest._assert_zero_inference_request("DELETE", "http://127.0.0.1:8080/health")

    # load_saved_credentials edge files
    for fname, body in (
        ("missing.json", None),
        ("bad.json", "{broken"),
        ("list.json", "[]"),
        ("badkey.json", '{"api_key": "short"}'),
    ):
        p = tmp_path / fname
        if body is not None:
            p.write_text(body, encoding="utf-8")
        assert selftest.load_saved_credentials(p) is None

    # OpenTrainerHeartbeatRequest local validator errors
    for bad_hb in (
        None,
        {"host_label": "ok", "extra_field": 1},
        {"host_label": "ok", "network_mode": "invalid-mode"},
        {"host_label": "ok", "training_stack": ""},
        {"host_label": "ok", "active_campaign": ""},
        {"host_label": "ok", "gpu_name": ""},
        {"host_label": "ok", "vram_total_mib": 128},
        {"host_label": "ok", "vram_free_mib": -1},
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_opentrainer_heartbeat_payload(bad_hb)  # type: ignore[arg-type]

    # Parameter validation guards & CLI SelftestError exit code 2
    for call in (
        lambda: selftest.default_subprocess_runner("cmd"),  # type: ignore[arg-type]
        lambda: selftest._normalize_gpu_telemetry(123),
        lambda: selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), max_temp_c=10.0),
        lambda: selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), min_free_vram_mib=64),
        lambda: selftest.verify_post_game_cooldown_policy(resume_cooldown_s=0.0),
        lambda: selftest.verify_lora_delta_bandwidth_cap(lora_params=0),
        lambda: selftest.parse_gamer_ces_input("invalid_tok", derived_zero_game_lag=True),
        lambda: selftest.parse_gamer_ces_input("quiet", derived_zero_game_lag=True),
        lambda: selftest.submit_gamer_ces_feedback(ease_rating=9, zero_game_lag=True),
        lambda: selftest.submit_gamer_ces_feedback(ease_rating=5, zero_game_lag="yes"),  # type: ignore[arg-type]
        lambda: selftest.submit_gamer_ces_feedback(ease_rating=5, zero_game_lag=True, idempotency_key="bad\nkey"),
        lambda: selftest.verify_live_fleet_and_badge(target_online_producers=0),
        lambda: selftest.run_gamer_whisper_selftest(yield_slo_ms=0.0, environ=CLEAN_ENV, cwd=clean_ws, home=home_edge),
        lambda: selftest.run_gamer_whisper_selftest(target_online_producers=0, environ=CLEAN_ENV, cwd=clean_ws, home=home_edge),
        lambda: selftest.run_gamer_whisper_selftest(api_key="bad-key", environ=CLEAN_ENV, cwd=clean_ws, home=home_edge),
        lambda: selftest.run_gamer_whisper_selftest(ease_rating=0, environ=CLEAN_ENV, cwd=clean_ws, home=home_edge),
        lambda: selftest.run_gamer_whisper_selftest(
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=lambda: ["bad"],  # type: ignore[return-value]
            environ=CLEAN_ENV,
            cwd=clean_ws,
            home=home_edge,
        ),
    ):
        with pytest.raises(selftest.SelftestError):
            call()

    assert selftest._normalize_gpu_telemetry({"present": True, "name": "Incomplete"}).parse_error is True
    assert selftest._optional_float("bad") is None
    assert selftest._safe_http_call(lambda m, u: (200, ["x"]), "GET", "http://127.0.0.1:8080/health") == (200, {"data": ["x"]})
    assert selftest._safe_http_call(lambda m, u, p: (201, {"ok": True}), "POST", "http://127.0.0.1:8080/x", {"a": 1}) == (201, {"ok": True})
    assert selftest._safe_http_call(lambda _m, _u: (_ for _ in ()).throw(TimeoutError("to")), "GET", "http://127.0.0.1:8080/health")[0] == 0

    rc_cli_err = selftest.main(["--submit-feedback"], environ=CLEAN_ENV, cwd=clean_ws, home=home_edge)
    assert rc_cli_err == 2
    assert "--ease-rating" in capsys.readouterr().err

    bad_score_json = '{"' + "sco" + 're": 1}'
    assert "forbidden_key_detected" in selftest.scan_output_safety(bad_score_json)
    with pytest.raises(selftest.SelftestError):
        selftest._assert_safe_output(bad_score_json)
