"""CPU-only tests of the T4 node layer: multi-domain workunit data, per-domain evaluation, the governor's night window, idle wait."""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hivemind.boinc.real_client import moe_work  # noqa: E402
from hivemind.boinc.real_client.boinc_server_scripts import wait_for_idle  # noqa: E402
from hivemind.boinc.real_client.moe_app import eval_by_domain  # noqa: E402


def test_domain_registry_is_consistent_and_excludes_benchmarks() -> None:
    assert set(moe_work.DOMAIN_ORDER) == set(moe_work._SOURCES)
    joined = " ".join(s["ds"] for s in moe_work._SOURCES.values()).lower()
    assert "mbpp" not in joined and "humaneval" not in joined  # the repository's evaluation benchmarks are never training data


def test_rows_slices_across_row_groups_and_from_the_end(tmp_path: Path) -> None:
    f = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": [f"row{i}" for i in range(2500)]}), f, row_group_size=1000)
    assert [r["text"] for r in moe_work._rows(f, ["text"], 995, 10)] == [f"row{i}" for i in range(995, 1005)]  # crosses a group boundary
    assert [r["text"] for r in moe_work._rows(f, ["text"], -3, 3)] == ["row2497", "row2498", "row2499"]
    assert len(moe_work._rows(f, ["text"], 2495, 50)) == 5  # asking past the end returns what exists


def test_ids_stops_when_enough_tokens_and_skips_blank_docs() -> None:
    tok = lambda s: SimpleNamespace(input_ids=list(range(len(s))))  # noqa: E731 - one token per character
    ids = moe_work._ids(tok, ["", "  ", "a" * 10, "b" * 10, "c" * 10], need=15)
    assert 15 <= len(ids) < 40  # stopped after the second real document, did not read the third


class _Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):  # a deterministic "loss": mean token id
        return SimpleNamespace(loss=input_ids.float().mean())


def test_eval_by_domain_per_domain_and_single_set_fallback() -> None:
    m = _Stub()
    per = eval_by_domain(m, {"eval_sets": {"a": [[1, 1]], "b": [[3, 3]], "empty": []}})
    assert per == {"a": 1.0, "b": 3.0}  # an empty domain is skipped, not divided by zero
    assert eval_by_domain(m, {"eval_tokens": [[2, 2]]}) == {"general": 2.0}  # workunits created before multi-domain evaluation


def test_governor_night_window_wraps_midnight() -> None:
    from t4_governor import in_night

    def at(h: int) -> float:  # 2026-10-25 is after the end of daylight saving: Paris = UTC+1
        return dt.datetime(2026, 10, 25, h, 0, tzinfo=dt.timezone.utc).timestamp()

    assert [in_night(at(h), "Europe/Paris", 22, 7) for h in (12, 20, 21, 23, 2, 5, 6)] == [False, False, True, True, True, True, False]
    assert in_night(at(10), "Europe/Paris", 8, 18) is True  # a non-wrapping window


def test_parse_pmon_ignores_header_and_unsampled_processes() -> None:
    from t4_governor import parse_pmon

    txt = ("# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command \n# Idx           #    C/G      %      %      %      %      %      %    name \n"
           "    0     742539     C      -      -      -      -      -      -    llama-server   \n    0     946827     C     37     12      -      -      -      -    llama-server   \n")
    assert parse_pmon(txt) == {"946827": 37}  # '-' means "no sample", which must not be read as idle (0)


def test_other_gpu_busy_without_a_boinc_task_uses_whole_gpu_utilisation() -> None:
    from t4_governor import other_gpu_busy

    assert other_gpu_busy({"procs": [], "util_pct": 80}) == (True, 80)  # e.g. a local Ollama client that never touched nginx
    assert other_gpu_busy({"procs": [], "util_pct": 3}) == (False, 3)


def test_chatlog_counts_new_completions_by_route_and_survives_rotation(tmp_path: Path) -> None:
    from t4_monitor import ChatLog

    log = tmp_path / "access.log"
    log.write_text('1.2.3.4 - - [25/Oct/2026:10:00:00 +0000] "POST /v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
    c = ChatLog(log)
    assert c.poll() == {"ok": True, "n": 0, "by": {}}  # the first poll starts "now": history is not recounted
    with open(log, "a") as f:
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:01 +0000] "POST /v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:02 +0000] "POST /bee/v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:03 +0000] "POST /u/v1/chat/completions HTTP/1.1" 499 0 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:04 +0000] "GET /v1/models HTTP/1.1" 401 5 "-" "x"\n')  # a poll, not an inference
    assert c.poll() == {"ok": True, "n": 3, "by": {"coder:200": 1, "bee:200": 1, "u:499": 1}}
    log.rename(tmp_path / "access.log.1")  # logrotate: a new file with a new inode
    log.write_text('1.2.3.4 - - [25/Oct/2026:10:05:00 +0000] "POST /v1/chat/completions HTTP/1.1" 502 0 "-" "x"\n')
    assert c.poll()["by"] == {"coder:502": 1}


def test_terminate_tree_ends_a_process_and_its_children_quickly() -> None:
    import subprocess as sp

    from t4_governor import descendants, terminate_tree

    # a parent that spawns a child, and both ignore SIGTERM to prove the SIGKILL fallback works
    code = ("import signal, subprocess, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\ntime.sleep(60)\n")
    p = sp.Popen([sys.executable, "-c", code])
    time.sleep(1.0)
    kids = descendants(p.pid)
    assert len(kids) >= 1
    t = time.time()
    assert terminate_tree(p.pid, grace_s=0.5) is True
    assert time.time() - t < 4.0
    p.wait(timeout=5)
    from t4_governor import pid_alive

    # "gone" means not running: an orphaned grandchild is a zombie until init reaps it (a few ms), so /proc/<pid> may still exist
    assert not any(pid_alive(k) for k in kids)
    assert pid_alive(os.getpid()) is True and pid_alive(p.pid) is False


def test_policy_drift_flags_changed_arguments_and_dropins_but_not_the_expected_state() -> None:
    from t4_monitor import policy_drift

    exp = "--idle-start-s 1800 --day-idle-s 7200 --min-free-mib 4300"
    assert policy_drift(exp, exp, [])["drift"] is False
    changed = policy_drift(exp, "--idle-start-s 5 --day-idle-s 5 --min-free-mib 3800", [])
    assert changed["drift"] is True and changed["args_match"] is False
    # the 22:08 incident: arguments may even match, but a drop-in override exists and silently wins at the next restart
    assert policy_drift(exp, exp, ["/etc/systemd/system/hivemind-t4-governor.service.d/intense-moe.conf"])["drift"] is True
    assert policy_drift(None, exp, [])["drift"] is False  # no recorded expectation (installer never ran): nothing to compare, not an alarm


def test_code_drift_flags_edited_and_missing_files_only() -> None:
    from t4_monitor import code_drift

    manifest = "aaa  scripts/t4_governor.py\nbbb  scripts/t4_monitor.py\nccc  bin/startup.sh\n"
    now = {"scripts/t4_governor.py": "aaa", "scripts/t4_monitor.py": "bbb", "bin/startup.sh": "ccc"}
    assert code_drift(manifest, now.get) == []
    now["scripts/t4_governor.py"] = "edited-by-hand"  # the 02:08Z incident: same arguments, different code
    del now["bin/startup.sh"]
    assert code_drift(manifest, now.get) == ["bin/startup.sh", "scripts/t4_governor.py"]


def test_governor_policy_comes_only_from_its_arguments() -> None:
    """Regression guard for the 02:08Z bypass: the resume rule once did min(os.environ.get("HIVEMIND_INTENSE_MOE_IDLE_S", 10), idle), which silently
    capped the owner's 1800 s / 7200 s to 10 s. The governor's policy must come from its arguments only, never from environment variables."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "t4_governor.py").read_text()
    assert "environ.get(" not in src and "getenv(" not in src and "INTENSE_MOE" not in src


def test_wait_for_idle_does_not_block_when_there_is_no_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIVEMIND_IDLE_URL", "http://127.0.0.1:9/nginx_status")  # nothing listens here
    t = time.time()
    wait_for_idle(quiet_s=5, max_wait_s=30)
    assert time.time() - t < 5  # a box without nginx must not stall validation
