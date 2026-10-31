# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The idle-GPU worker must fail on the side of the player: unknown means busy, nothing is invented, and what it reports is what it measured."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker_safety", ROOT / "scripts" / "hivemind_boinc_idle_worker.py")
assert _spec is not None and _spec.loader is not None
worker = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_boinc_idle_worker_safety"] = worker
_spec.loader.exec_module(worker)


@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """No network, and every request body the worker tries to send is recorded."""
    sent: list[dict] = []

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202
        try:
            sent.append(json.loads(req.data.decode()))
        except Exception:  # noqa: BLE001
            pass
        raise OSError("offline test")

    monkeypatch.setattr(worker.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_unmeasurable_idle_time_means_the_user_is_active_not_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux without xprintidle (Wayland, headless, minimal install): it used to answer 10 s, so the worker kept running during a game."""
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    assert worker.get_user_idle_seconds() == 0.0
    assert worker.get_user_idle_seconds(assume_idle_when_unknown=True) == 10.0  # a headless, never-played machine can opt in


def test_a_failing_xprintidle_is_also_treated_as_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/xprintidle")

    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise subprocess.TimeoutExpired("xprintidle", 1.5)

    monkeypatch.setattr(worker.subprocess, "run", boom)
    assert worker.get_user_idle_seconds() == 0.0


def test_a_gpu_that_cannot_be_read_is_busy_and_unavailable_never_an_invented_card(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(worker.subprocess, "run", boom)
    gpu = worker.query_nvidia_gpu_status()
    assert gpu["available"] is False and gpu["gpu_model"] == "unknown"
    assert gpu["utilization_pct"] >= 45.0 and gpu["temp_c"] >= 78.0  # on the wrong side of both thresholds in the worker
    assert "RTX 2070" not in json.dumps(gpu)  # the old fallback invented this card


def test_na_fields_from_nvidia_smi_are_not_filled_with_made_up_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    out = SimpleNamespace(returncode=0, stdout="Tesla T4, 3, 41, 15360, 700, [N/A], 580.99\n")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: out)
    gpu = worker.query_nvidia_gpu_status()
    assert gpu["available"] is True and gpu["power_w"] is None and gpu["gpu_model"] == "Tesla T4"


def test_no_boinc_client_is_reported_as_nothing_done_not_as_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    assert worker.set_boinc_client_gpu_mode(False) == "boinccmd_not_found"
    assert worker.set_boinc_client_gpu_mode(True) == "boinccmd_not_found"


def test_boinccmd_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/boinccmd")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert worker.set_boinc_client_gpu_mode(False) == "boinccmd_failed_rc1"
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert worker.set_boinc_client_gpu_mode(False) == "boinccmd_gpu_never"
    assert worker.set_boinc_client_gpu_mode(True) == "boinccmd_gpu_auto"


def test_when_the_gpu_cannot_be_read_the_worker_does_not_start_even_if_the_user_is_idle(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 600.0)
    monkeypatch.setattr(worker, "query_nvidia_gpu_status", lambda: {"available": False, "gpu_model": "unknown", "utilization_pct": 100.0, "temp_c": 99.0,
                                                                   "total_mib": 0.0, "used_mib": 0.0, "power_w": None, "driver_version": "unknown"})
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "boinccmd_gpu_auto" if enable else "boinccmd_gpu_never")
    out = worker.run_one_idle_gpu_boinc_cycle()
    assert out["boinc_active"] is False and out["boinc_driver"] == "boinccmd_gpu_never" and out["cuda_fold"]["n_points_folded"] == 0


def test_a_player_is_never_disturbed_and_an_idle_pc_is_used(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    gpu = {"available": True, "gpu_model": "Test GPU", "utilization_pct": 3.0, "temp_c": 40.0, "total_mib": 8192.0, "used_mib": 500.0, "power_w": 20.0, "driver_version": "1"}
    monkeypatch.setattr(worker, "query_nvidia_gpu_status", lambda: gpu)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto" if enable else "never")
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 1.0)  # typing a second ago
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is False
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 30.0)
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is True
    busy = dict(gpu, utilization_pct=80.0)  # a game rendering with the player not touching anything (a cutscene)
    monkeypatch.setattr(worker, "query_nvidia_gpu_status", lambda: busy)
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is False


def test_the_idle_time_sent_to_the_hub_is_the_measured_one(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    """It used to be raised to threshold + 0.5 s whenever the worker ran, so the hub was told a PC had been idle longer than anyone measured."""
    gpu = {"available": True, "gpu_model": "Test GPU", "utilization_pct": 3.0, "temp_c": 40.0, "total_mib": 8192.0, "used_mib": 500.0, "power_w": 20.0, "driver_version": "1"}
    monkeypatch.setattr(worker, "query_nvidia_gpu_status", lambda: gpu)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto")
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 4.2)
    worker.run_one_idle_gpu_boinc_cycle(idle_threshold_seconds=4.0)
    assert offline and all(body.get("gpu_idle_seconds") == 4.2 for body in offline if "gpu_idle_seconds" in body)


def test_the_report_says_the_qlora_numbers_are_a_placeholder(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 60.0)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto")
    out = worker.run_one_idle_gpu_boinc_cycle(force_fold=True)
    assert out["qlora_outer_delta_summary"]["placeholder_not_training"] is True
    a = worker.compute_deterministic_qlora_outer_delta("WU-1", "physics_sim", "d")
    assert a == worker.compute_deterministic_qlora_outer_delta("WU-1", "physics_sim", "d")  # a pure function of its inputs: nothing was trained
    assert a["holdout_nll_after"] == 1.798  # the same "result" for every workunit
