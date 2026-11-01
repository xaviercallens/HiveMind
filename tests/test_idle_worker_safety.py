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
    out = SimpleNamespace(returncode=0, stdout="Tesla T4, 3, 41, 15360, 700, [N/A], 580.99\n", stderr="")
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/nvidia-smi")  # the probe checks the binary exists before running it
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


_T4 = "Tesla T4, 3, 41, 15360, 500, 20.0, 1, 00000000:00:04.0\n"
_REAL_PROBE_GPU = worker.probe_gpu  # captured before any monkeypatch replaces worker.probe_gpu


def _probe_of(csv: str, returncode: int = 0, stderr: str = ""):  # noqa: ANN202
    """A probe built through the injected runner (the only faked part is the external nvidia-smi process)."""
    return _REAL_PROBE_GPU(
        run=lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=csv, stderr=stderr), which=lambda name: "/usr/bin/nvidia-smi"
    )


def _cuda_fold(target_planet: str = "K2-18 b", n_pts: int = 16384, gpu=None, cuda_lib=None) -> dict:  # noqa: ANN001
    return {"backend": worker.CUDA_BACKEND, "cuda_device": gpu.name, "cuda_total_vram_mib": int(gpu.total_mib), "n_points_folded": n_pts, "vram_allocated_bytes": n_pts * 8,
            "kernel_exec_ms": 0.3, "recovered_transit_depth_ppm": 2789.0, "orbital_period_days": 32.9, "result_sha256": "d" * 64}


def test_when_the_gpu_cannot_be_read_the_worker_does_not_start_even_if_the_user_is_idle(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 600.0)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe_of("", returncode=1, stderr="no driver"))
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "boinccmd_gpu_auto" if enable else "boinccmd_gpu_never")
    out = worker.run_one_idle_gpu_boinc_cycle()
    assert out["boinc_active"] is False and out["boinc_driver"] == "boinccmd_gpu_never" and "cuda_fold" not in out and out["status"] == "BLOCKED"
    assert offline == []  # nothing was sent anywhere


def test_a_player_is_never_disturbed_and_an_idle_pc_is_used(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe_of(_T4))
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _cuda_fold)  # the CUDA driver is external hardware
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto" if enable else "never")
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 1.0)  # typing a second ago
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is False
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 30.0)
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is True
    # a game rendering with the player not touching anything (a cutscene)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe_of("Tesla T4, 80, 41, 15360, 500, 20.0, 1, 00000000:00:04.0\n"))
    assert worker.run_one_idle_gpu_boinc_cycle()["boinc_active"] is False


def test_the_idle_time_sent_to_the_hub_is_the_measured_one(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    """It used to be raised to threshold + 0.5 s whenever the worker ran, so the hub was told a PC had been idle longer than anyone measured."""
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe_of(_T4))
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _cuda_fold)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto")
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 4.2)
    worker.run_one_idle_gpu_boinc_cycle(idle_threshold_seconds=4.0)
    assert offline and all(body.get("gpu_idle_seconds") == 4.2 for body in offline if "gpu_idle_seconds" in body)
    assert any("gpu_idle_seconds" in body for body in offline)


def test_the_report_carries_no_invented_qlora_numbers(monkeypatch: pytest.MonkeyPatch, offline: list[dict]) -> None:
    """The deterministic hash-derived QLoRA "delta" (fixed loss 1.798 for every workunit) is gone; the report says training is BLOCKED."""
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe_of(_T4))
    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", _cuda_fold)
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda assume=False: 60.0)
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: "auto")
    out = worker.run_one_idle_gpu_boinc_cycle(force_fold=True)
    assert out["qlora"] == {"status": "BLOCKED", "reason": "no_trainer_wired"}
    assert "qlora_outer_delta_summary" not in out and "1.798" not in json.dumps(out)
    assert not hasattr(worker, "compute_deterministic_qlora_outer_delta")
    assert out["lease_attempted"] is False and out["submit_attempted"] is False
