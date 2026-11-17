# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""IP-296: the cross-project T4 GPU lease (admission against the floor, mutual exclusion, floor watchdog)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lease():
    spec = importlib.util.spec_from_file_location("t4_gpu_lease", ROOT / "scripts" / "t4_gpu_lease.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_admission_keeps_the_floor_and_refuses_blind() -> None:
    m = _lease()
    assert m.admit(10_091, 5_000)[0] is True
    assert m.admit(10_091, 7_000)[0] is False  # the 2026-11-16 dream job next to a 3 GB run
    assert m.admit(None, 100)[0] is False


def test_a_second_job_waits_then_gives_up_with_tempfail(tmp_path) -> None:
    m = _lease()
    lock = tmp_path / "t4.lock"
    first = m.Lease(lock)
    assert first.try_acquire({"pid": 1})
    clk = _clock = _Clock()
    logs: list[str] = []
    rc = m.run_leased(["true"], need_mib=1000, wait_s=60, lock=lock, free_fn=lambda: 12_000,
                      sleep=_clock.sleep, clock=clk, log=logs.append)
    assert rc == m.EX_TEMPFAIL and "held by another GPU job" in logs[-1]
    first.release()
    assert m.status(lock)["held"] is False


def test_job_runs_when_admitted_and_is_stopped_when_the_floor_breaks(tmp_path) -> None:
    m = _lease()
    clk = _Clock()
    rc = m.run_leased([sys.executable, "-c", "pass"], need_mib=1000, wait_s=0, lock=tmp_path / "a.lock",
                      free_fn=lambda: 12_000, sleep=lambda s: None, clock=clk, log=lambda _m: None)
    assert rc == 0
    readings = iter([12_000, 3_000, 2_900, 2_900, 2_900])
    logs: list[str] = []
    rc = m.run_leased([sys.executable, "-c", "import time; time.sleep(30)"], need_mib=1000, wait_s=0,
                      lock=tmp_path / "b.lock", free_fn=lambda: next(readings, 2_900), sample_s=0.05,
                      sleep=__import__("time").sleep, clock=__import__("time").monotonic, log=logs.append, grace_s=5)
    assert rc != 0 and any("stopping the job" in x for x in logs)
    assert m.status(tmp_path / "b.lock")["held"] is False  # released after the job ends


def test_not_enough_memory_means_not_started(tmp_path) -> None:
    m = _lease()
    started: list[list[str]] = []
    rc = m.run_leased(["true"], need_mib=7_000, wait_s=0, lock=tmp_path / "c.lock", free_fn=lambda: 10_091,
                      sleep=lambda s: None, clock=_Clock(), log=lambda _m: None, popen=lambda c: started.append(c))
    assert rc == m.EX_TEMPFAIL and started == []


def test_cli_refuses_a_lower_floor(tmp_path) -> None:
    m = _lease()
    assert m.main(["run", "--need-mib", "100", "--floor-mib", "1000", "--lock", str(tmp_path / "d.lock"), "--", "true"]) == 2
