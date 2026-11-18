# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A READY idle-worker cycle must mean a fold really ran on the CUDA device the idle gate measured (`scripts/hivemind_boinc_idle_worker.py`).

External I/O only is faked: nvidia-smi output, the CUDA driver library and boinccmd. The fold, the gate and the cycle are the real code.
"""

from __future__ import annotations

import ctypes
import importlib.util
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hivemind_boinc_idle_worker.py"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # deliberately not registered in sys.modules, like the gateway
    return mod


worker = _load("hivemind_boinc_idle_worker_truthful_tests")
_REAL_PROBE_GPU = worker.probe_gpu

SMI_ONE = "Tesla T4, 5, 52, 15360, 612, 28.4, 550.54.15, 00000000:00:04.0\n"
# GPU 0: small and busy (a game). GPU 1: large and idle. best() (most VRAM) is GPU 1.
SMI_TWO_BIG_IDLE = "GeForce GTX 1650, 92, 81, 4096, 3500, 70.0, 550.54.15, 00000000:01:00.0\nTesla T4, 3, 41, 15360, 400, 20.0, 550.54.15, 00000000:02:00.0\n"
# GPU 0: small and idle. GPU 1: large and busy.
SMI_TWO_BIG_BUSY = "GeForce GTX 1650, 2, 40, 4096, 300, 20.0, 550.54.15, 00000000:01:00.0\nTesla T4, 95, 82, 15360, 9000, 70.0, 550.54.15, 00000000:02:00.0\n"


def _probe(csv: str) -> Any:
    return _REAL_PROBE_GPU(
        run=lambda *a, **k: subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=csv, stderr=""),
        which=lambda _n: "/usr/bin/nvidia-smi",
    )


class _Boinc:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def __call__(self, enable: bool) -> str:
        self.calls.append(enable)
        return "boinccmd_gpu_auto" if enable else "boinccmd_gpu_never"


@pytest.fixture()
def boinc(monkeypatch: pytest.MonkeyPatch) -> _Boinc:
    rec = _Boinc()
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", rec)
    return rec


def _no_cuda_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise OSError("libcuda not loadable")

    monkeypatch.setattr(worker.ctypes, "CDLL", refuse)
    if hasattr(worker.ctypes, "WinDLL"):
        monkeypatch.setattr(worker.ctypes, "WinDLL", refuse)


# ------------------------------------------------------------------ 1. a CPU fallback is never READY


def test_cpu_fallback_fold_makes_the_cycle_blocked_and_never_touches_boinccmd(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    _no_cuda_driver(monkeypatch)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_ONE))
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 600.0)
    rep = worker.run_one_idle_gpu_boinc_cycle()
    assert rep["status"] == "BLOCKED" and rep["reason"] == "cuda_fold_unavailable" and rep["stage"] == "cuda_fold"
    assert rep["boinc_active"] is False and rep["boinc_touched"] is False
    assert rep["cuda_fold"]["backend"] == "cpu_fallback" and rep["cuda_fold"]["cuda_device"] == "CPU"
    assert boinc.calls == []  # neither pause nor resume


def test_dry_run_with_cpu_fallback_does_not_pass(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    _no_cuda_driver(monkeypatch)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_ONE))
    rep = worker.run_one_idle_gpu_boinc_cycle(dry_run=True)
    assert rep["status"] == "BLOCKED" and rep.get("dry_run_passed") is not True
    assert boinc.calls == []


def test_cuda_backed_fold_gives_ready_and_reports_backend_and_device(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    """The driver call is the external I/O faked here; the cycle must carry the backend and the device through and resume BOINC."""
    seen: dict[str, Any] = {}

    def cuda_fold(target_planet: str = "K2-18 b", n_pts: int = 16384, gpu: Any = None, cuda_lib: Any = None) -> dict[str, Any]:
        seen["gpu"] = gpu
        return {"backend": worker.CUDA_BACKEND, "cuda_device": gpu.name, "cuda_total_vram_mib": 15360, "n_points_folded": n_pts, "vram_allocated_bytes": 131072,
                "kernel_exec_ms": 0.3, "recovered_transit_depth_ppm": 2789.0, "orbital_period_days": 32.9, "result_sha256": "a" * 64}

    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", cuda_fold)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_ONE))
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 600.0)
    rep = worker.run_one_idle_gpu_boinc_cycle(dry_run=True)
    assert rep["status"] == "READY" and rep["boinc_active"] is True and rep["dry_run_passed"] is True
    assert rep["cuda_fold"]["backend"] == worker.CUDA_BACKEND and rep["cuda_fold"]["cuda_device"] == "Tesla T4"
    assert boinc.calls == [True] and seen["gpu"].name == "Tesla T4"


# ------------------------------------------------------------------ 2. the gate and the fold use the same device


def test_two_gpus_the_fold_runs_on_the_card_the_gate_measured(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    folded_on: list[Any] = []

    def cuda_fold(target_planet: str = "K2-18 b", n_pts: int = 16384, gpu: Any = None, cuda_lib: Any = None) -> dict[str, Any]:
        folded_on.append(gpu)
        return {"backend": worker.CUDA_BACKEND, "cuda_device": gpu.name, "cuda_total_vram_mib": 15360, "n_points_folded": n_pts, "vram_allocated_bytes": 131072,
                "kernel_exec_ms": 0.3, "recovered_transit_depth_ppm": 2789.0, "orbital_period_days": 32.9, "result_sha256": "b" * 64}

    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", cuda_fold)
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 600.0)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_TWO_BIG_IDLE))
    rep = worker.run_one_idle_gpu_boinc_cycle()
    assert rep["boinc_active"] is True and rep["gpu"]["gpu_model"] == "Tesla T4"
    assert len(folded_on) == 1 and folded_on[0].pci_bus_id == "00000000:02:00.0" and folded_on[0].name == "Tesla T4"  # not ordinal 0 (the busy GTX 1650)


def test_two_gpus_busy_measured_card_pauses_and_runs_no_fold(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    def must_not_run(*a: Any, **k: Any) -> dict[str, Any]:
        raise AssertionError("fold ran on a busy GPU")

    monkeypatch.setattr(worker, "execute_cuda_ptx_bls_fold", must_not_run)
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 600.0)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_TWO_BIG_BUSY))
    rep = worker.run_one_idle_gpu_boinc_cycle()
    assert rep["boinc_active"] is False and rep["cuda_fold"]["backend"] == "paused_for_active_gameplay" and rep["cuda_fold"]["n_points_folded"] == 0
    assert boinc.calls == [False]


class _FakeCuda:
    """Driver API stand-in for two devices: CUDA ordinal 0 is the T4, ordinal 1 is the GTX 1650 (the reverse of nvidia-smi's order)."""

    NAMES = {0: b"Tesla T4", 1: b"GeForce GTX 1650"}
    BUS = {b"00000000:02:00.0": 0, b"00000000:01:00.0": 1}

    def cuDeviceGetByPCIBusId(self, ref: Any, bus: bytes) -> int:
        if bus not in self.BUS:
            return 101
        ref._obj.value = self.BUS[bus]
        return 0

    def cuDeviceGet(self, ref: Any, ordinal: int) -> int:
        ref._obj.value = ordinal
        return 0

    def cuDeviceGetCount(self, ref: Any) -> int:
        ref._obj.value = 2
        return 0

    def cuDeviceGetName(self, buf: Any, n: int, dev: Any) -> int:
        buf.value = self.NAMES[dev.value]
        return 0


def test_device_is_matched_by_pci_bus_id_not_by_position() -> None:
    gpus = {g.name: g for g in _probe(SMI_TWO_BIG_IDLE).gpus}
    lib = _FakeCuda()
    t4 = worker._select_cuda_device(lib, gpus["Tesla T4"])
    gtx = worker._select_cuda_device(lib, gpus["GeForce GTX 1650"])
    assert t4 is not None and t4.value == 0  # nvidia-smi lists the T4 second, CUDA lists it first
    assert gtx is not None and gtx.value == 1


def test_device_selection_refuses_a_mismatch_and_an_ambiguous_machine() -> None:
    lib = _FakeCuda()
    t4 = {g.name: g for g in _probe(SMI_TWO_BIG_IDLE).gpus}["Tesla T4"]
    assert worker._select_cuda_device(lib, t4._replace(pci_bus_id="00000000:09:00.0")) is None  # unknown bus id
    assert worker._select_cuda_device(lib, t4._replace(name="Quadro X")) is None  # same bus id, different card
    assert worker._select_cuda_device(lib, t4._replace(pci_bus_id=None)) is None  # two CUDA devices and no way to tell which


# ------------------------------------------------------------------ 3. importing the script is thread-safe


def test_concurrent_exec_of_the_script_never_raises_and_leaves_sys_modules_alone() -> None:
    errors: list[BaseException] = []
    before = set(sys.modules)
    for round_no in range(6):
        barrier = threading.Barrier(16)

        def run() -> None:
            try:
                barrier.wait(timeout=10)
                mod = _load("hivemind_boinc_idle_worker")  # same name in every thread, as the gateway does
                assert mod.GpuProbe(status="BLOCKED", reason="x", gpus=(), detail="").best() is None
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert errors == []
    assert "hivemind_boinc_idle_worker" not in set(sys.modules) - before


def test_script_has_no_dataclass_and_no_sys_modules_write() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s*@dataclass|import dataclass", src) is None and "sys.modules[" not in src


# ------------------------------------------------------------------ 4. unknown idle time is not idle


def test_default_cycle_with_no_xprintidle_is_not_idle(monkeypatch: pytest.MonkeyPatch, boinc: _Boinc) -> None:
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _probe(SMI_ONE))
    assert worker.get_user_idle_seconds() == 0.0
    rep = worker.run_one_idle_gpu_boinc_cycle()
    assert rep["status"] == "READY" and rep["boinc_active"] is False and rep["idle_seconds"] == 0.0
    assert rep["cuda_fold"]["backend"] == "paused_for_active_gameplay" and boinc.calls == [False]


def test_cli_flag_opts_a_headless_box_in(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def cycle(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"status": "READY"}

    monkeypatch.setattr(worker, "run_one_idle_gpu_boinc_cycle", cycle)
    assert worker.main(["--once"]) == 0 and seen["assume_idle_when_unknown"] is False
    assert worker.main(["--once", "--assume-idle-when-unmeasurable"]) == 0 and seen["assume_idle_when_unknown"] is True
