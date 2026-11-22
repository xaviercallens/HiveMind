# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, compliance, and gateway integration tests for SLICE-DYN-04 (`scripts/noel_gamer_whisper_selftest.py`)."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
SELFTEST_PATH = REPO_ROOT / "scripts" / "noel_gamer_whisper_selftest.py"

_spec = importlib.util.spec_from_file_location("noel_gamer_whisper_selftest", SELFTEST_PATH)
assert _spec is not None and _spec.loader is not None
selftest = importlib.util.module_from_spec(_spec)
sys.modules["noel_gamer_whisper_selftest"] = selftest
_spec.loader.exec_module(selftest)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn04",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn04",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

TEST_KEY = "hm-test_gamer_whisper_9876"
TEST_LIVE_KEY = "hm_live_gamer_whisper_4321"
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin"}


def _quiet_rtx4070_probe(**_: Any) -> selftest.GpuTelemetry:
    return selftest.GpuTelemetry(
        present=True,
        parse_error=False,
        name="NVIDIA GeForce RTX 4070",
        total_vram_mib=12288,
        free_vram_mib=9600,
        used_vram_mib=2688,
        temp_c=52.0,
        util_pct=4.0,
    )


def _fast_worker_drill(**_: Any) -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 42.5,
        "exit_code": 0,
        "method": "injected_fast_drill",
    }


def _idle_processes(**_: Any) -> list[str]:
    return ["systemd", "python3", "explorer.exe", "steam.exe"]


def _verify_sha256_receipt(report: dict[str, Any]) -> None:
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in report.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert report["receipt_sha256"] == expected


def _make_gateway_http(
    seen: list[tuple[str, str, dict[str, Any] | None]] | None = None,
    *,
    health_status: int = 200,
    raid_status: int = 200,
    feedback_status: int = 200,
) -> selftest.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = headers
        assert "completions" not in url.lower(), "Must NEVER call /v1/chat/completions"
        if seen is not None:
            seen.append((method, url, payload))
        if method == "GET" and url.endswith("/health"):
            if health_status != 200:
                return health_status, {"status": "error"}
            return 200, {"status": "ok"}
        if method == "GET" and url.endswith("/v1/hivemind/raid/status"):
            if raid_status != 200:
                return raid_status, {"error": f"http_{raid_status}"}
            return 200, {
                "object": "hivemind.raid_noel.status",
                "campaign": "winter-gpu-raid-holiday-guild",
                "swarm": {"active_trainers": 3, "live_producers": 5},
                "feedback": {"mean_ease_rating": 4.8, "total_responses": 12},
            }
        if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
            if feedback_status not in (200, 201):
                return feedback_status, {"error": f"http_{feedback_status}"}
            return feedback_status, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 13,
                "mean_ease_rating": 4.85,
                "zero_game_lag_ratio": 1.0,
            }
        return 404, {"error": "not_found"}

    return _http


def test_headers_stdlib_lateral_movement_security_scanner_zero_stub_and_forbidden_added() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (SELFTEST_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = SELFTEST_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit (zero unauthorized URLs, zero raw sockets, zero SSH/LAN binaries)
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_gamer_whisper_selftest.py",
        )
        == []
    )
    poisoned = script_source + '\n_BAD_URL = "https://telemetry.steampowered.com/ping"\n'
    assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    # 2. Pure Python standard library only in scripts/noel_gamer_whisper_selftest.py
    tree = ast.parse(script_source, filename="scripts/noel_gamer_whisper_selftest.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both target and test files
    for rel, content in (
        ("scripts/noel_gamer_whisper_selftest.py", script_source),
        ("tests/test_noel_gamer_whisper_selftest.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner (0 findings) + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_gamer_whisper_selftest.py", script_source),
        ("tests/test_noel_gamer_whisper_selftest.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1004,
        pr_title="feat(onboarding): 10s Gamer-Whisper pre-flight self-test",
        pr_body="Implements SLICE-DYN-04 sub-1.5s Steam GPU yield and 70 C quiet-fan self-test.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding): add noel_gamer_whisper_selftest"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    from gwaya.ast_audit import ZeroStubAudit

    stub_audit = ZeroStubAudit().audit_python_code(script_source)
    assert stub_audit.is_clean, f"Unexpected GWAYA stub violations: {stub_audit.violations}"


def test_real_subprocess_worker_yield_drill_and_steam_dx12_detection(tmp_path: Path) -> None:
    # Execute the real local Python worker subprocess lifecycle drill (list args, shell=False)
    drill = selftest.default_worker_lifecycle_drill(trigger_exe="eldenring.exe")
    assert drill["worker_started"] is True
    assert drill["worker_stopped"] is True
    assert drill["clean_release"] is True
    assert drill["retained_worker_vram_mib"] == 0
    assert 0.0 <= drill["yield_latency_ms"] < 1500.0
    assert drill["exit_code"] == 0

    # End-to-end run_gamer_whisper_selftest using the real subprocess drill
    hobby_dir = tmp_path / "home_pc_gaming"
    hobby_dir.mkdir()
    report = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert report["exit_code"] == 0
    assert report["compliance_ok"] is True
    assert report["ready"] is True
    assert report["verdict"] == "ready"
    assert report["yield_slo_ok"] is True
    assert report["yield_latency_ms"] < 1500.0
    assert report["retained_worker_vram_mib"] == 0
    assert report["thermal_ok"] is True
    assert report["vram_reserve_ok"] is True
    assert report["cooldown_policy_ok"] is True
    assert report["bandwidth_cap_ok"] is True
    assert report["zero_game_lag"] is True
    assert report["within_10s_slo"] is True
    assert report["checks_passed_count"] == 5
    assert report["checks_total_count"] == 5
    _verify_sha256_receipt(report)

    # Verify Steam / DX12 game-launch detection across canonical titles, truncated POSIX comm, Steam paths, and GPU >15%
    for game_proc in (
        "cs2.exe",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "bg3.exe",
        "helldivers2.exe",
        "palworld-win64-",  # 15-char Linux ps comm truncation
        "/mnt/nvme/SteamLibrary/steamapps/common/Starfield/game_bin",
        "steam_app_1245620",
    ):
        det = selftest.detect_steam_or_dx12_game_activity([game_proc])
        assert det["game_detected"] is True
        assert len(det["matched_processes"]) == 1

    gpu_det = selftest.detect_steam_or_dx12_game_activity(["explorer.exe"], non_worker_gpu_util_pct=22.0)
    assert gpu_det["game_detected"] is True
    assert gpu_det["gpu_util_triggered"] is True

    idle_det = selftest.detect_steam_or_dx12_game_activity(["explorer.exe", "steam.exe"], non_worker_gpu_util_pct=8.0)
    assert idle_det["game_detected"] is False

    # Live Steam game running right now triggers immediate worker yield + paused_game_active state
    live_game_rep = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda **_: ["explorer.exe", "cs2.exe"],
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert live_game_rep["exit_code"] == 1
    assert live_game_rep["ready"] is False
    assert live_game_rep["verdict"] == "paused_game_active"
    assert live_game_rep["yield_slo_ok"] is True
    assert live_game_rep["preemption_drill"]["live_game_detected"] is True
    assert live_game_rep["cooldown_policy"]["current_state"] == "paused_game_active"
    assert live_game_rep["cooldown_policy"]["cooldown_remaining_s"] == 300.0


def test_slow_yield_and_unclean_release_fail_closed_exit_code_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hobby_dir = tmp_path / "gamer_rig"
    hobby_dir.mkdir()

    # 1. Exact boundary 1500.0 ms fails closed (>= 1500.0 ms -> exit code 2, yield_slo_ok=False, zero_game_lag=False)
    rc = selftest.main(
        ["--json"],
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=lambda: ["systemd"],
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
    )
    assert rc == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["exit_code"] == 2
    assert rep["yield_slo_ok"] is False
    assert rep["ready"] is False
    assert rep["verdict"] == "failed_yield_slo"
    assert rep["zero_game_lag"] is False
    assert rep["failures_count"] >= 1
    _verify_sha256_receipt(rep)

    # 2. Retained worker VRAM > 0 MiB or unclean worker stop also fails closed with exit code 2
    rep_retained = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=lambda: {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 256,
            "yield_latency_ms": 80.0,
        },
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_retained["exit_code"] == 2
    assert rep_retained["yield_slo_ok"] is False
    assert rep_retained["zero_game_lag"] is False
    assert rep_retained["retained_worker_vram_mib"] == 256

    # 3. Clock-measured slow drill (when yield_latency_ms is omitted by worker_lifecycle_fn) & >10s overall SLO breach
    ticks = iter([0.0, 0.01, 0.02, 0.03, 1.85, 11.5, 11.6])
    rep_clock = selftest.run_gamer_whisper_selftest(
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
    )
    assert rep_clock["exit_code"] == 2
    assert rep_clock["yield_slo_ok"] is False
    assert rep_clock["within_10s_slo"] is False
    assert rep_clock["zero_game_lag"] is False


def test_quiet_vs_hot_gpu_vs_low_vram_vs_no_nvidia_gpu_and_parse_error(tmp_path: Path) -> None:
    hobby_dir = tmp_path / "gpu_matrix"
    hobby_dir.mkdir()

    # 1. Quiet GPU near ceiling (67.5 C <= 70.0 C): gentle whisper throttle (0.5), ready=True, zero_game_lag=True
    rep_near = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 3080",
            "total_vram_mib": 10240,
            "free_vram_mib": 6144,
            "temp_c": 67.5,
            "util_pct": 5.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_near["exit_code"] == 0
    assert rep_near["ready"] is True
    assert rep_near["thermal_ok"] is True
    assert rep_near["zero_game_lag"] is True
    assert rep_near["thermal_vram_governor"]["governor_action"] == "whisper_gentle_throttle"
    assert rep_near["thermal_vram_governor"]["duty_cycle_scale"] == 0.5

    # 2. Hot GPU (> 70.0 C): thermal clamp/pause, thermal_ok=False, zero_game_lag=False
    rep_hot = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 3080",
            "total_vram_mib": 10240,
            "free_vram_mib": 7000,
            "temp_c": 74.2,
            "util_pct": 6.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_hot["exit_code"] == 1
    assert rep_hot["ready"] is False
    assert rep_hot["verdict"] == "paused_thermal_ceiling"
    assert rep_hot["gpu_status"] == "thermal_clamped_paused"
    assert rep_hot["thermal_ok"] is False
    assert rep_hot["vram_reserve_ok"] is True
    assert rep_hot["zero_game_lag"] is False
    assert rep_hot["thermal_vram_governor"]["duty_cycle_scale"] == 0.0

    # Custom --max-temp-c (e.g. 65.0 C) clamps 67.0 C GPU in French
    rep_custom_cap = selftest.run_gamer_whisper_selftest(
        max_temp_c=65.0,
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 4060",
            "total_vram_mib": 8192,
            "free_vram_mib": 5000,
            "temp_c": 67.0,
            "util_pct": 2.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_custom_cap["thermal_ok"] is False
    assert rep_custom_cap["verdict"] == "paused_thermal_ceiling"

    # 3. Low free VRAM (< 2048 MiB): vram_reserve_hold
    rep_low_vram = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 2070",
            "total_vram_mib": 8192,
            "free_vram_mib": 1536,
            "temp_c": 55.0,
            "util_pct": 4.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_low_vram["exit_code"] == 1
    assert rep_low_vram["ready"] is False
    assert rep_low_vram["verdict"] == "paused_low_vram_reserve"
    assert rep_low_vram["gpu_status"] == "vram_reserve_hold"
    assert rep_low_vram["vram_reserve_ok"] is False
    assert rep_low_vram["thermal_ok"] is True

    # Combined hot GPU + low VRAM
    rep_both = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 2070",
            "total_vram_mib": 8192,
            "free_vram_mib": 1024,
            "temp_c": 78.0,
            "util_pct": 10.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_both["gpu_status"] == "thermal_and_vram_clamped_paused"
    assert rep_both["thermal_ok"] is False
    assert rep_both["vram_reserve_ok"] is False

    # 4. No NVIDIA GPU present: honest 'cpu_or_dev_only_no_nvidia_gpu' without fake GPU numbers
    rep_no_gpu = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {"present": False},
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_no_gpu["exit_code"] == 0
    assert rep_no_gpu["ready"] is True
    assert rep_no_gpu["verdict"] == "cpu_or_dev_only_no_nvidia_gpu"
    assert rep_no_gpu["gpu_status"] == "cpu_or_dev_only_no_nvidia_gpu"
    gov_no_gpu = rep_no_gpu["thermal_vram_governor"]
    assert gov_no_gpu["gpu_present"] is False
    assert gov_no_gpu["gpu_name"] is None
    assert gov_no_gpu["temp_c"] is None
    assert gov_no_gpu["total_vram_mib"] is None
    assert gov_no_gpu["free_vram_mib"] is None

    # 5. Unreadable / corrupted nvidia-smi output fails closed (exit code 2)
    rep_parse_err = selftest.run_gamer_whisper_selftest(
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_parse_err["exit_code"] == 2
    assert rep_parse_err["ready"] is False
    assert rep_parse_err["gpu_status"] == "gpu_driver_parse_error"


def test_300s_cooldown_and_lora_delta_bandwidth_cap_policies(tmp_path: Path) -> None:
    hobby_dir = tmp_path / "policy_checks"
    hobby_dir.mkdir()

    # Active cooldown (< 300s since game exit) keeps worker paused
    rep_holding = selftest.run_gamer_whisper_selftest(
        seconds_since_game_exit=120.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_holding["exit_code"] == 1
    assert rep_holding["ready"] is False
    assert rep_holding["verdict"] == "paused_post_game_cooldown"
    assert rep_holding["cooldown_policy_ok"] is True
    assert rep_holding["cooldown_policy"]["cooldown_remaining_s"] == 180.0
    assert rep_holding["cooldown_policy"]["resume_allowed_now"] is False

    # Cooldown elapsed (>= 300s since game exit) allows quiet resume
    rep_clear = selftest.run_gamer_whisper_selftest(
        seconds_since_game_exit=305.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_clear["exit_code"] == 0
    assert rep_clear["ready"] is True
    assert rep_clear["cooldown_policy"]["resume_allowed_now"] is True

    # Misconfigured cooldown < 300s fails closed (exit code 2)
    rep_bad_cd = selftest.run_gamer_whisper_selftest(
        resume_cooldown_s=60.0,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        lang="fr",
    )
    assert rep_bad_cd["exit_code"] == 2
    assert rep_bad_cd["cooldown_policy_ok"] is False

    # Legacy full-checkpoint transfer mode violates <50 MB/h cap and fails closed (exit code 2)
    rep_full_ckpt = selftest.run_gamer_whisper_selftest(
        transfer_mode="full_checkpoint",
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_full_ckpt["exit_code"] == 2
    assert rep_full_ckpt["bandwidth_cap_ok"] is False
    assert rep_full_ckpt["bandwidth_cap"]["estimated_upload_mb_per_hour"] == 28000.0


def test_corporate_pc_and_amadeus_git_remote_refusal_without_network_or_probe_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_called = False
    http_called = False

    def _forbidden_probe(**_: Any) -> Any:
        nonlocal probe_called
        probe_called = True
        return _quiet_rtx4070_probe()

    def _forbidden_http(*_: Any, **__: Any) -> tuple[int, dict[str, Any]]:
        nonlocal http_called
        http_called = True
        return 200, {}

    # 1. Corporate env markers (USERDNSDOMAIN, HTTPS_PROXY zscaler, COMPUTERNAME, AMADEUS_ENV, USERDOMAIN)
    clean_dir = tmp_path / "personal_folder"
    clean_dir.mkdir()
    for corp_env in (
        {"USERDNSDOMAIN": "nce.amadeus.net"},
        {"HTTPS_PROXY": "http://gateway.zscaler.net:8080"},
        {"COMPUTERNAME": "WKS-ENT-CORP01", "USERDOMAIN": "NETSKOPE_INTERNAL"},
        {"HOSTNAME": "laptop.paloalto.intranet"},
        {"AMADEUS_SITE": "NCE", "ZSCALER_CERT_PATH": "/etc/zscaler.pem"},
    ):
        rc = selftest.main(
            ["--check-gateway", "--submit-feedback", "--ease-rating", "5", "--json"],
            gpu_probe_fn=_forbidden_probe,
            process_probe_fn=_forbidden_probe,
            worker_lifecycle_fn=_forbidden_probe,
            http_fn=_forbidden_http,
            environ=corp_env,
            cwd=clean_dir,
            home=clean_dir,
        )
        assert rc == 2
        out_raw = capsys.readouterr().out
        rep = json.loads(out_raw)
        assert rep["compliance_ok"] is False
        assert rep["ready"] is False
        assert rep["verdict"] == "refused_compliance"
        assert rep["zero_game_lag"] is False
        # Never leak actual env values
        for secret_val in corp_env.values():
            assert secret_val not in out_raw

    # 2. Standard .git/config with corporate Amadeus remote URL
    repo_dir = tmp_path / "weekend_game_mod"
    (repo_dir / ".git").mkdir(parents=True)
    secret_corp_remote = "https://git.rnd.amadeus.com/internal-team/secret-repo.git"
    (repo_dir / ".git" / "config").write_text(
        f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {secret_corp_remote}\n'
        '[url "ssh://git@bluecoat.corp.example/"]\n\tinsteadOf = https://mirror.example/\n',
        encoding="utf-8",
    )
    rc_git = selftest.main(
        ["--lang", "fr", "--markdown"],
        gpu_probe_fn=_forbidden_probe,
        http_fn=_forbidden_http,
        environ=CLEAN_ENV,
        cwd=repo_dir,
        home=tmp_path,
    )
    assert rc_git == 2
    md_out = capsys.readouterr().out
    assert "amadeus" in md_out.lower()
    assert secret_corp_remote not in md_out
    assert "0,00 EUR / 0,00 USD" in md_out

    # 3. Git worktree (.git file -> gitdir + commondir) corporate remote detection
    main_git = tmp_path / "common_git_dir"
    wt_git = main_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (main_git / "config").write_text(
        '[remote "upstream"]\n\turl = git@fortinet.enterprise.local:team/proj.git\n',
        encoding="utf-8",
    )
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    wt_workspace = tmp_path / "wt_checkout"
    wt_workspace.mkdir()
    (wt_workspace / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")

    comp_wt = selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=wt_workspace)
    assert comp_wt["compliance_ok"] is False
    assert "git_remote:fortinet" in comp_wt["corporate_indicators"]
    assert "git_remote:enterprise" in comp_wt["corporate_indicators"]

    # Zero probes and zero HTTP requests ever executed across all corporate rejections!
    assert probe_called is False
    assert http_called is False


def test_gateway_health_ces_feedback_statuses_prompt_and_live_router_integration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hobby_dir = tmp_path / "Users" / "GamerNoel"
    hobby_dir.mkdir(parents=True)

    # 1. Real FastAPI router integration (GET /health, GET /v1/hivemind/raid/status, POST /v1/hivemind/onboarding/feedback)
    app = create_router_app(RouterStore(tmp_path / "router.sqlite", secret=b"slice-dyn-04-router-secret-32b!!"))
    client = TestClient(app)

    def _asgi_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower()
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        resp = client.request(method, path, json=payload, headers=headers)
        return resp.status_code, resp.json()

    rc_live = selftest.main(
        [
            "--gateway",
            "http://127.0.0.1:8080",
            "--check-gateway",
            "--submit-feedback",
            "--ease-rating",
            "5",
            "--json",
        ],
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        http_fn=_asgi_http,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        system_name="Windows",
    )
    assert rc_live == 0
    rep_live = json.loads(capsys.readouterr().out)
    assert rep_live["ready"] is True
    assert rep_live["gateway_check"]["gateway_ok"] is True
    assert rep_live["feedback"]["submitted"] is True
    assert rep_live["feedback"]["zero_pii_payload"] == {
        "profile": "gamer",
        "ease_rating": 5,
        "zero_game_lag": True,
        "platform": "windows",
    }

    # --had-game-lag overrides zero_game_lag to False even when yield_slo_ok and thermal_ok are True
    rep_lag_override = selftest.run_gamer_whisper_selftest(
        gateway="http://127.0.0.1:8080",
        submit_feedback=True,
        ease_rating=4,
        had_game_lag=True,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        http_fn=_asgi_http,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        system_name="Linux",
    )
    assert rep_lag_override["zero_game_lag"] is False
    assert rep_lag_override["feedback"]["zero_pii_payload"]["zero_game_lag"] is False

    # Hot GPU (>70 C) derives zero_game_lag=False honestly in CES telemetry
    rep_hot_fb = selftest.run_gamer_whisper_selftest(
        gateway="http://127.0.0.1:8080",
        submit_feedback=True,
        ease_rating=3,
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 4070",
            "total_vram_mib": 12288,
            "free_vram_mib": 8000,
            "temp_c": 76.0,
            "util_pct": 5.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        http_fn=_asgi_http,
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_hot_fb["zero_game_lag"] is False
    assert rep_hot_fb["feedback"]["zero_pii_payload"]["zero_game_lag"] is False

    # 2. HTTP 404 / 429 / 500 / 400 / 0 handling across gateway check and CES feedback in EN & FR
    for code, expected_rc, expected_status in (
        (404, 2, "missing_route"),
        (429, 1, "rate_limited"),
        (500, 1, "server_error"),
        (400, 1, "client_error"),
        (0, 1, "network_error"),
    ):
        for lang_code in ("en", "fr"):
            rep_fb_err = selftest.run_gamer_whisper_selftest(
                submit_feedback=True,
                ease_rating=4,
                lang=lang_code,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=_make_gateway_http(feedback_status=code),
                environ=CLEAN_ENV,
                cwd=hobby_dir,
                home=hobby_dir,
            )
            assert rep_fb_err["exit_code"] == expected_rc
            assert rep_fb_err["feedback"]["status"] == expected_status

            rep_gw_err = selftest.run_gamer_whisper_selftest(
                check_gateway=True,
                lang=lang_code,
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=_make_gateway_http(raid_status=code),
                environ=CLEAN_ENV,
                cwd=hobby_dir,
                home=hobby_dir,
            )
            assert rep_gw_err["exit_code"] == expected_rc
            assert rep_gw_err["gateway_check"]["status"] == expected_status

    # 3. Interactive --prompt-ces (answer '5 gamer quiet', '4 lag', skip on empty, skip on EOF)
    seen_posts: list[tuple[str, str, dict[str, Any] | None]] = []
    rep_prompt_ok = selftest.run_gamer_whisper_selftest(
        prompt_ces=True,
        lang="fr",
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        input_fn=lambda _: "5 gamer quiet",
        http_fn=_make_gateway_http(seen_posts),
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
        system_name="Darwin",
    )
    assert rep_prompt_ok["exit_code"] == 0
    assert rep_prompt_ok["ease_rating"] == 5
    assert rep_prompt_ok["zero_game_lag"] is True
    assert seen_posts[-1][2] == {
        "profile": "gamer",
        "ease_rating": 5,
        "zero_game_lag": True,
        "platform": "macos",
    }

    rep_prompt_lag = selftest.run_gamer_whisper_selftest(
        prompt_ces=True,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        input_fn=lambda _: "4 stutter",
        http_fn=_make_gateway_http(),
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_prompt_lag["zero_game_lag"] is False

    def _eof_input(_: str) -> str:
        raise EOFError

    rep_prompt_skip = selftest.run_gamer_whisper_selftest(
        prompt_ces=True,
        gpu_probe_fn=_quiet_rtx4070_probe,
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        input_fn=_eof_input,
        http_fn=_make_gateway_http(),
        environ=CLEAN_ENV,
        cwd=hobby_dir,
        home=hobby_dir,
    )
    assert rep_prompt_skip["exit_code"] == 0
    assert rep_prompt_skip["feedback"]["skipped"] is True


def test_default_probes_http_loopback_redaction_and_cli_markdown_rendering(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "SecretGamerName"
    home_dir.mkdir(parents=True)

    # 1. default_gpu_probe & default_process_probe with mocked runner
    csv_gpus = (
        "NVIDIA GeForce GTX 1650, 4096, 3200, 896, 48, 2\n"
        "NVIDIA GeForce RTX 4080, 16384, 14200, [N/A], 51, [N/A]\n"
    )
    probed_gpu = selftest.default_gpu_probe(runner=lambda cmd, t: (0, csv_gpus))
    assert probed_gpu.present is True
    assert probed_gpu.name == "NVIDIA GeForce RTX 4080"
    assert probed_gpu.free_vram_mib == 14200
    assert probed_gpu.used_vram_mib == 2184
    assert probed_gpu.util_pct == 0.0

    assert selftest.default_gpu_probe(runner=lambda cmd, t: (127, "")).present is False
    assert selftest.default_gpu_probe(runner=lambda cmd, t: (0, "bad,row")).parse_error is True
    assert selftest.default_gpu_probe(runner=lambda cmd, t: (0, "GPU, -1, 0, 0, 40, 1")).parse_error is True

    win_procs = selftest.default_process_probe(
        runner=lambda cmd, t: (0, '"Steam.exe","100","Console","1","90 K"\n"cs2.exe","200","Console","1","4 G"\n'),
        os_name="nt",
    )
    assert "cs2.exe" in win_procs
    posix_procs = selftest.default_process_probe(
        runner=lambda cmd, t: (0, "systemd\n/opt/games/eldenring.exe\n"),
        os_name="posix",
    )
    assert "eldenring.exe" in posix_procs

    # Real read-only default_subprocess_runner smoke check
    rc_py, out_py = selftest.default_subprocess_runner([sys.executable, "-c", "print('ok_runner')"])
    assert rc_py == 0 and "ok_runner" in out_py
    rc_missing, _ = selftest.default_subprocess_runner(["nonexistent_hivemind_bin_xyz"])
    assert rc_missing == 127

    # 2. Real loopback HTTP server exercising default_http_json (200 JSON, 404 JSON error, zero-inference block)
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                body = json.dumps({"error": "missing"}).encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        c200, b200 = selftest.default_http_json("GET", f"http://127.0.0.1:{port}/health")
        assert c200 == 200 and b200["status"] == "ok"
        c404, _ = selftest.default_http_json("GET", f"http://127.0.0.1:{port}/v1/hivemind/raid/status")
        assert c404 == 404
    finally:
        server.shutdown()
        server.server_close()

    with pytest.raises(selftest.SelftestError, match="Zero-inference invariant violated"):
        selftest.default_http_json("POST", "http://127.0.0.1:8080/v1/chat/completions", {})

    # 3. Render CLI and Markdown in both EN & FR + verify redaction & zero forbidden keys
    for lang_flag in ("en", "fr"):
        for fmt_flags in ([], ["--markdown"], ["--json"]):
            rc = selftest.main(
                [
                    "--check-gateway",
                    "--submit-feedback",
                    "--ease-rating",
                    "5",
                    "--lang",
                    lang_flag,
                    *fmt_flags,
                ],
                gpu_probe_fn=_quiet_rtx4070_probe,
                process_probe_fn=_idle_processes,
                worker_lifecycle_fn=_fast_worker_drill,
                http_fn=_make_gateway_http(),
                environ=CLEAN_ENV,
                cwd=home_dir,
                home=home_dir,
            )
            assert rc == 0
            out = capsys.readouterr().out
            assert "SecretGamerName" not in out
            assert "0.00 EUR / 0.00 USD" in out
            assert selftest.scan_output_safety(out) == []

    # 4. Redaction, key masking, gateway validation, and CLI error handling negative controls
    scrubbed = selftest.sanitize_text_for_display(
        f"path=/home/SecretGamerName/.hivemind key={TEST_KEY} live={TEST_LIVE_KEY} "
        "auth='Bearer tok_secret_123456' mail=gamer@example.org ip=192.168.1.50 loop=127.0.0.1",
        home=home_dir,
    )
    assert "SecretGamerName" not in scrubbed
    assert TEST_KEY not in scrubbed and "hm-***9876" in scrubbed
    assert TEST_LIVE_KEY not in scrubbed and "hm_live_***4321" in scrubbed
    assert "tok_secret_123456" not in scrubbed
    assert "gamer@example.org" not in scrubbed
    assert "192.168.1.50" not in scrubbed and "127.0.0.1" in scrubbed
    assert selftest.scan_output_safety(scrubbed) == []

    bad_key_fragment = '{"' + "sco" + 're": 5}'
    assert "forbidden_key_detected" in selftest.scan_output_safety(bad_key_fragment)
    with pytest.raises(selftest.SelftestError, match="Output safety guard failed"):
        selftest._assert_safe_output(bad_key_fragment)

    for bad_gw in (
        "",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://evil.example.com",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1",
        "https://user:pass@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?x=1",
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_gateway_url(bad_gw)

    # CLI validation error returns exit code 2 with redacted stderr
    rc_bad_cli = selftest.main(
        ["--submit-feedback"],
        environ=CLEAN_ENV,
        cwd=home_dir,
        home=home_dir,
    )
    assert rc_bad_cli == 2
    assert "--ease-rating" in capsys.readouterr().err

    # Subprocess CLI smoke execution
    proc = subprocess.run(
        [sys.executable, str(SELFTEST_PATH), "--json"],
        cwd=str(home_dir),
        env={"PATH": "/usr/bin:/bin", "HOME": str(home_dir)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode in (0, 1)
    cli_json = json.loads(proc.stdout)
    assert cli_json["compliance_ok"] is True
    assert cli_json["yield_slo_ok"] is True
    _verify_sha256_receipt(cli_json)


def test_validation_guards_edge_branches_and_render_warnings_errors(tmp_path: Path) -> None:
    # Language, platform tag, key masking, home path, and output safety edge branches
    with pytest.raises(selftest.SelftestError):
        selftest.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(selftest.SelftestError):
        selftest.normalize_lang("de")

    assert selftest._platform_tag("Darwin") == "macos"
    assert selftest._platform_tag("Windows 11") == "windows"
    assert selftest._platform_tag("Linux") == "linux"
    assert selftest._platform_tag("FreeBSD 14") == "freebsd-14"
    assert selftest._platform_tag("   ") == "unknown"

    assert selftest.is_valid_hivemind_key(None) is False
    assert selftest.is_valid_hivemind_key("hm-short") is False
    assert selftest.mask_api_key("") == "<missing>"
    assert selftest.mask_api_key("not-a-valid-key") == "<invalid>"

    home_dir = tmp_path / "home" / "AliceGamer"
    home_dir.mkdir(parents=True)
    assert selftest.redact_home_path(home_dir, home=home_dir) == "~"
    assert selftest.redact_home_path(r"C:\Users\AliceGamer\.hivemind\cfg.json", home=Path("/nonexistent")) == "~/.hivemind/cfg.json"

    safety_hits = selftest.scan_output_safety(
        f"a@b.org {TEST_KEY} Bearer secret_tok_99 /home/bob/x 10.1.2.3"
    )
    assert "email_detected" in safety_hits
    assert "unmasked_api_key_detected" in safety_hits
    assert "bearer_token_detected" in safety_hits
    assert "unredacted_home_path_detected" in safety_hits
    assert "non_loopback_ipv4_detected:10.1.2.3" in safety_hits

    # Gateway URL & zero-inference method guards
    assert selftest.is_allowed_gateway_host("") is False
    for invalid_url in (
        "https://ok.run.app\n",
        "https://ok.run.app/ space",
        "http://127.0.0.1:badport",
        "ftp://127.0.0.1:8080",
    ):
        with pytest.raises(selftest.SelftestError):
            selftest.validate_gateway_url(invalid_url)

    with pytest.raises(selftest.SelftestError, match="Unsupported HTTP method"):
        selftest._assert_zero_inference_request("DELETE", "http://127.0.0.1:8080/health")

    # default_http_json POST, empty body, non-JSON body, non-JSON HTTPError, and unreachable port
    class _EdgeHandler(http.server.BaseHTTPRequestHandler):
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
                self.wfile.write(b"bad gateway upstream")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.HTTPServer(("127.0.0.1", 0), _EdgeHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c204, b204 = selftest.default_http_json(
            "POST",
            f"http://127.0.0.1:{port}/empty",
            {"ping": 1},
            {"X-Test": "1"},
        )
        assert c204 == 204 and b204 == {}
        c_plain, b_plain = selftest.default_http_json("POST", f"http://127.0.0.1:{port}/plain", {"a": 1})
        assert c_plain == 200 and b_plain["raw_length"] == 8
        c502, b502 = selftest.default_http_json("POST", f"http://127.0.0.1:{port}/err", {"a": 1})
        assert c502 == 502 and "bad gateway" in b502["error"]
    finally:
        srv.shutdown()
        srv.server_close()

    # Closed port triggers URLError -> (0, network_error)
    c_closed, b_closed = selftest.default_http_json("GET", f"http://127.0.0.1:{port}/health")
    assert c_closed == 0 and "network_error" in b_closed

    # _safe_http_call arity variants & transport exception catch
    assert selftest._safe_http_call(lambda m, u: (200, ["item"]), "GET", "http://127.0.0.1:8080/health") == (
        200,
        {"data": ["item"]},
    )
    assert selftest._safe_http_call(
        lambda m, u, p: (201, {"ok": True}),
        "POST",
        "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback",
        {"x": 1},
    ) == (201, {"ok": True})

    def _timeout_http(m: str, u: str) -> tuple[int, dict[str, Any]]:
        _ = (m, u)
        raise TimeoutError("timed out")

    c_to, _ = selftest._safe_http_call(_timeout_http, "GET", "http://127.0.0.1:8080/health")
    assert c_to == 0

    # Relative gitdir worktree config + cwd corporate path detection
    rel_wt = tmp_path / "amadeus_work_clone"
    rel_gitdir = tmp_path / "rel_git_meta"
    rel_wt.mkdir()
    rel_gitdir.mkdir()
    (rel_gitdir / "config").write_text('[remote "origin"]\n\turl = https://github.com/user/hobby.git\n', encoding="utf-8")
    (rel_wt / ".git").write_text("gitdir: ../rel_git_meta\n", encoding="utf-8")
    comp_cwd = selftest.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=rel_wt)
    assert comp_cwd["compliance_ok"] is False
    assert "cwd_path:amadeus" in comp_cwd["corporate_indicators"]

    # Subprocess runner, GPU normalization, process probe, and drill edge guards
    with pytest.raises(selftest.SelftestError):
        selftest.default_subprocess_runner("nvidia-smi")  # type: ignore[arg-type]
    assert selftest._optional_float("not-a-float") is None
    with pytest.raises(selftest.SelftestError):
        selftest._normalize_gpu_telemetry(123)
    assert selftest._normalize_gpu_telemetry({"present": True, "name": "IncompleteGPU"}).parse_error is True

    assert selftest.default_process_probe(runner=lambda c, t: (1, ""), os_name="nt") == []
    assert selftest.default_process_probe(runner=lambda c, t: (1, ""), os_name="posix") == []
    assert selftest.default_process_probe(runner=lambda c, t: (0, "\n  \nsystemd\n"), os_name="posix") == ["systemd"]
    det_extra = selftest.detect_steam_or_dx12_game_activity(
        ["", "monsterhunterwi"],
        extra_games=["monsterhunterwilds_dx12.exe"],
    )
    assert det_extra["game_detected"] is True

    non_game_drill = selftest.default_worker_lifecycle_drill(trigger_exe="notepad.exe")
    assert non_game_drill["worker_started"] is True
    assert non_game_drill["clean_release"] is False

    # Parameter validation guards across governor, cooldown, bandwidth, CES, and selftest
    with pytest.raises(selftest.SelftestError):
        selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), max_temp_c=10.0)
    with pytest.raises(selftest.SelftestError):
        selftest.evaluate_thermal_and_vram_governor(_quiet_rtx4070_probe(), min_free_vram_mib=64)
    with pytest.raises(selftest.SelftestError):
        selftest.verify_post_game_cooldown_policy(resume_cooldown_s=0.0)
    with pytest.raises(selftest.SelftestError):
        selftest.verify_lora_delta_bandwidth_cap(lora_params=0)
    with pytest.raises(selftest.SelftestError):
        selftest.parse_gamer_ces_input("unknown_token", derived_zero_game_lag=True)
    with pytest.raises(selftest.SelftestError):
        selftest.parse_gamer_ces_input("quiet", derived_zero_game_lag=True)
    with pytest.raises(selftest.SelftestError):
        selftest.submit_gamer_ces_feedback(ease_rating=9, zero_game_lag=True)
    with pytest.raises(selftest.SelftestError):
        selftest.submit_gamer_ces_feedback(ease_rating=5, zero_game_lag="yes")  # type: ignore[arg-type]
    with pytest.raises(selftest.SelftestError):
        selftest.run_gamer_whisper_selftest(yield_slo_ms=0.0, environ=CLEAN_ENV, cwd=home_dir, home=home_dir)
    with pytest.raises(selftest.SelftestError):
        selftest.run_gamer_whisper_selftest(ease_rating=0, environ=CLEAN_ENV, cwd=home_dir, home=home_dir)
    with pytest.raises(selftest.SelftestError):
        selftest.run_gamer_whisper_selftest(
            gpu_probe_fn=_quiet_rtx4070_probe,
            process_probe_fn=_idle_processes,
            worker_lifecycle_fn=lambda: ["bad"],  # type: ignore[return-value]
            environ=CLEAN_ENV,
            cwd=home_dir,
            home=home_dir,
        )

    # Render Markdown and CLI when both warnings and errors are populated
    warn_err_rep = selftest.run_gamer_whisper_selftest(
        check_gateway=True,
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 4070",
            "total_vram_mib": 12288,
            "free_vram_mib": 8000,
            "temp_c": 75.0,
            "util_pct": 5.0,
        },
        process_probe_fn=_idle_processes,
        worker_lifecycle_fn=_fast_worker_drill,
        http_fn=_make_gateway_http(raid_status=503),
        environ=CLEAN_ENV,
        cwd=home_dir,
        home=home_dir,
    )
    md_text = selftest.render_markdown_report(warn_err_rep)
    cli_text = selftest.render_cli_report(warn_err_rep)
    assert "> **Warning**:" in md_text and "> **Error**:" in md_text
    assert "[WARN]" in cli_text and "[ERROR]" in cli_text

