"""Tests of the governor's telemetry additions and of the governor report (stdlib only: no torch/pyarrow needed)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from t4_governor import SCHEMA, other_gpu_procs  # noqa: E402
from t4_governor_report import load, percentile, summarise  # noqa: E402


def test_other_gpu_procs_excludes_the_boinc_task_and_survives_a_vanished_process() -> None:
    me = str(os.getpid())
    got = other_gpu_procs([(me, "python", "123"), ("999999", "ghost", "5"), ("424242", "boinc-task", "4300")], mine={"424242"})
    assert [p["pid"] for p in got] == [me, "999999"]  # the BOINC task is not a culprit
    assert got[0]["cmd"] != "" and got[0]["mib"] == "123"
    assert got[1]["cmd"] == ""  # no /proc entry: empty command line, not an exception
    assert len(other_gpu_procs([(me, "p", "1")] * 20, mine=set())) == 6  # bounded


def test_schema_is_additive_and_the_policy_guard_still_holds() -> None:
    assert SCHEMA >= 2
    src = (Path(__file__).resolve().parents[1] / "scripts" / "t4_governor.py").read_text()
    assert "environ.get(" not in src and "getenv(" not in src and "INTENSE_MOE" not in src  # same guard as test_boinc_node_layer


def test_yield_grace_default_meets_the_roadmap_latency_target_and_is_not_a_policy_argument() -> None:
    import t4_governor

    # HM-S19-Q35-03 asks for a preemption latency < 1.0 s; the measured latency is the grace plus ~0.12 s, so the grace must leave room for that
    assert t4_governor.YIELD_GRACE_S + 0.12 < 1.0
    src = (Path(__file__).resolve().parents[1] / "scripts" / "t4_governor.py").read_text()
    assert "default=YIELD_GRACE_S" in src  # the unit file passes no --yield-grace-s, and the owner's policy arguments are untouched


def _jl(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "governor.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")  # a corrupt line must be skipped
    return p


def test_report_counts_bursts_empty_bursts_and_separates_old_and_new_latencies(tmp_path: Path) -> None:
    ev = [
        {"event": "snapshot", "t": 1.0},
        # an old-schema yield with a real task (no trigger / ended_by_governor fields)
        {"event": "resume", "t": 100.0, "night": True}, {"event": "yield", "t": 160.0, "had_boinc_gpu_task": True, "gpu_freed_ms": 7129},
        # a new-schema yield that ended the task itself
        {"event": "resume", "t": 1000.0, "night": True},
        {"event": "yield", "t": 1100.0, "trigger": "nginx", "had_boinc_gpu_task": True, "ended_by_governor": True, "gpu_freed_ms": 1135},
        # an empty burst triggered by another GPU user
        {"event": "resume", "t": 3000.0, "night": True},
        {"event": "yield", "t": 3120.0, "trigger": "gpu", "had_boinc_gpu_task": False, "gpu_freed_ms": 55, "schema": 2,
         "other_procs": [{"pid": "7", "name": "llama-server", "mib": "2086", "cmd": "/usr/local/lib/ollama/llama-server --model x"}]},
    ]
    s = summarise(load(_jl(tmp_path, ev)))
    assert (s["resumes"], s["yields"], s["bursts"], s["bursts_empty"]) == (3, 3, 3, 1)
    assert s["yield_triggers"] == {"unknown": 1, "nginx": 1, "gpu": 1}
    y = s["yield_with_task"]
    assert y["n"] == 2 and y["latency_ms_new"] == [1135] and y["latency_ms_old"] == [7129]
    assert "too few" in y["p99_new"]  # one sample is not a p99
    cmd = "/usr/local/lib/ollama/llama-server --model x"
    assert s["gpu_trigger_culprits"] == {"llama-server " + cmd[:40]: 1}  # name plus the first 40 characters of the command line
    assert s["mode_always_min_per_day"] == {"1970-01-01": round((60 + 100 + 120) / 60, 1)}


def test_report_on_an_empty_log_does_not_divide_by_zero(tmp_path: Path) -> None:
    s = summarise(load(_jl(tmp_path, [])))
    assert s["bursts"] == 0 and s["burst_seconds"] is None and s["yield_with_task"]["n"] == 0


def test_percentile_is_the_nearest_rank_and_bounded() -> None:
    assert percentile([5.0], 0.99) == 5.0
    assert percentile(list(map(float, range(100))), 0.5) == 50.0
    assert percentile(list(map(float, range(100))), 0.99) == 99.0
