# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-02-D1 — VRAM Floor Guard tests across night_probe_queue, e8_amcp_round, and hivemind_opentrainer_worker.

Verifies that every probe and trainer launch checks ``free_vram - estimated_peak >= 4,300 MiB``,
computes/applies ``torch.cuda.set_per_process_memory_fraction``, and refuses a simulated
4,000 MiB-free GPU (negative control).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script(name: str, rel_path: str) -> Any:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_night_probe_queue_refuses_4000_mib_free_gpu_and_caps_above_floor() -> None:
    """Negative + positive controls on scripts/night_probe_queue.py for the 4,300 MiB floor."""
    npq = _load_script("night_probe_queue_vram_guard", "scripts/night_probe_queue.py")
    arms = npq.load_arms()
    arm_a0 = arms[0]

    # Negative control 1: simulated 4,000 MiB-free GPU (< 4,300 MiB floor) is refused by check_vram_floor_guard
    guard_4000 = npq.check_vram_floor_guard(free_mib=4000, estimated_peak_mib=1800, floor_mib=4300, total_mib=15360)
    assert guard_4000["ok"] is False
    assert guard_4000["status"] == "BLOCKED"
    assert "4300 MiB" in guard_4000["reason"]

    # Negative control 2: preflight with vram_floor_mib=4300 refuses a 4,000 MiB-free GPU
    def _gpu_4000() -> dict[str, Any]:
        return {"used_mib": 11360, "total_mib": 15360, "temp_c": 52, "util_pct": 0, "power_w": 28.0, "procs": []}

    def _http_idle(url: str) -> str:
        if "api/ps" in url:
            return '{"models": []}'
        return "Active connections: 1\nserver accepts handled requests\n 1 1 1\nReading: 0 Writing: 1 Waiting: 0\n"

    env_4000 = npq.Env(
        now=lambda: dt.datetime(2026, 11, 7, 23, 30, tzinfo=dt.timezone.utc),
        gpu=_gpu_4000,
        http_get=_http_idle,
        boinc_pids=lambda procs: [],
        estimate=lambda _arm: {"available": True, "alloc_peak_mib": 1900, "process_peak_mib": 2250},
    )
    opts_floor = npq.Options(vram_floor_mib=4300)
    pre_4000 = npq.preflight(env_4000, opts_floor, arm_a0)
    assert pre_4000["ok"] is False
    assert pre_4000["status"] == "BLOCKED"
    assert "below floor 4300 MiB" in pre_4000["reason"]

    # Negative control 3: free=6,000 MiB (> 4,300) where free - estimated_peak (2,250 MiB) = 3,750 MiB < 4,300 MiB
    env_6000 = npq.Env(
        now=lambda: dt.datetime(2026, 11, 7, 23, 30, tzinfo=dt.timezone.utc),
        gpu=lambda: {"used_mib": 9360, "total_mib": 15360, "temp_c": 52, "util_pct": 0, "power_w": 28.0, "procs": []},
        http_get=_http_idle,
        boinc_pids=lambda procs: [],
        estimate=lambda _arm: {"available": True, "alloc_peak_mib": 1900, "process_peak_mib": 2250},
    )
    pre_6000 = npq.preflight(env_6000, opts_floor, arm_a0)
    assert pre_6000["ok"] is False
    assert pre_6000["status"] == "BLOCKED"
    assert "below estimated peak 2250 MiB" in pre_6000["reason"]

    # Positive control: free=9,500 MiB, estimated_peak=2,250 MiB -> cap = 9500 - 4300 = 5200 MiB >= 2250 MiB
    guard_9500 = npq.check_vram_floor_guard(free_mib=9500, estimated_peak_mib=2250, floor_mib=4300, total_mib=15360)
    assert guard_9500["ok"] is True
    assert guard_9500["cap_mib"] == 5200
    assert guard_9500["memory_fraction"] == pytest.approx(5200 / 15360, rel=1e-4)

    env_9500 = npq.Env(
        now=lambda: dt.datetime(2026, 11, 7, 23, 30, tzinfo=dt.timezone.utc),
        gpu=lambda: {"used_mib": 5860, "total_mib": 15360, "temp_c": 52, "util_pct": 0, "power_w": 28.0, "procs": []},
        http_get=_http_idle,
        boinc_pids=lambda procs: [],
        estimate=lambda _arm: {"available": True, "alloc_peak_mib": 1900, "process_peak_mib": 2250},
    )
    pre_9500 = npq.preflight(env_9500, opts_floor, arm_a0)
    assert pre_9500["ok"] is True
    assert pre_9500["status"] == "READY"
    assert pre_9500["cap_mib"] == 5200


def test_e8_amcp_round_gate_refuses_4000_mib_free_gpu_and_applies_memory_fraction() -> None:
    """Negative + positive controls on scripts/e8_amcp_round.py for the 4,300 MiB floor and CUDA fraction."""
    amcp = _load_script("e8_amcp_round_vram_guard", "scripts/e8_amcp_round.py")
    args = amcp.build_parser().parse_args(
        ["learner", "--run-id", "test-vram-floor", "--learner-id", "l0", "--device", "cuda"]
    )
    assert args.min_free_mib == 4300.0

    # Negative control 1: simulated 4,000 MiB-free GPU is refused
    ok_4000, reason_4000 = amcp.gate(args, "cuda", free_fn=lambda: 4000.0)
    assert ok_4000 is False
    assert "4000 MiB free" in reason_4000
    assert "4300 MiB must stay free" in reason_4000

    # Negative control 2: simulated 4,700 MiB-free GPU with need_mib=600 leaves 4,100 MiB < 4,300 MiB -> refused
    ok_4700, reason_4700 = amcp.gate(args, "cuda", free_fn=lambda: 4700.0)
    assert ok_4700 is False
    assert "4300 MiB must stay free" in reason_4700

    # Positive control: 9,300 MiB free -> accepted, and apply_vram_floor_fraction sets (9300 - 4300) / 15360
    recorded_fractions: list[float] = []
    frac = amcp.apply_vram_floor_fraction(
        9300.0,
        min_free_mib=4300.0,
        total_mib=15360.0,
        set_fraction_fn=recorded_fractions.append,
    )
    assert frac == pytest.approx(5000.0 / 15360.0, rel=1e-4)
    assert len(recorded_fractions) == 1
    assert recorded_fractions[0] == pytest.approx(5000.0 / 15360.0, rel=1e-4)

    ok_9300, reason_9300 = amcp.gate(args, "cuda", free_fn=lambda: 9300.0)
    assert ok_9300 is True
    assert reason_9300 == "ok"


def test_opentrainer_worker_refuses_4000_mib_free_gpu_and_caps_fraction_above_floor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Negative + positive controls on scripts/hivemind_opentrainer_worker.py for the 4,300 MiB floor."""
    worker = _load_script("hivemind_opentrainer_worker_vram_guard", "scripts/hivemind_opentrainer_worker.py")
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-vram-guard-key")

    fractions_applied: list[float] = []
    guard_4000 = worker.check_vram_floor_guard(
        free_vram_mib=4000,
        estimated_peak_mib=600,
        floor_mib=4300,
        total_vram_mib=15360,
        set_fraction_fn=fractions_applied.append,
    )
    assert guard_4000["ok"] is False
    assert "below floor 4300 MiB" in guard_4000["reason"]
    assert fractions_applied == []

    # CLI negative control: --vram-free-mib 4000 exits with code 2 before any network request
    rc = worker.main(
        [
            "--server-url",
            "http://127.0.0.1:9",
            "--gpu-name",
            "Tesla T4",
            "--vram-total-mib",
            "15360",
            "--vram-free-mib",
            "4000",
            "--once",
        ]
    )
    assert rc == 2
    assert "VRAM floor guard refused launch" in capsys.readouterr().err

    # Positive control: 9,100 MiB free with 3,200 MiB estimated peak leaves 5,900 MiB >= 4,300 MiB
    guard_9100 = worker.check_vram_floor_guard(
        free_vram_mib=9100,
        estimated_peak_mib=3200,
        floor_mib=4300,
        total_vram_mib=15360,
        set_fraction_fn=fractions_applied.append,
    )
    assert guard_9100["ok"] is True
    assert guard_9100["cap_mib"] == 4800
    assert len(fractions_applied) == 1
    assert fractions_applied[0] == pytest.approx(4800.0 / 15360.0, rel=1e-4)
