# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Offline tests for the OpenTrainer CrowdGPT contributor mode (no GPU, no network, no real client)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.opentrainer import crowdgpt_contributor as cc  # noqa: E402
from hivemind.opentrainer.crowdgpt_contributor import (  # noqa: E402
    Action,
    ComputeApp,
    GpuSnapshot,
    GuardConfig,
    GuardState,
    Phase,
    decide,
)

AGENT_PATH = REPO_ROOT / "scripts" / "hivemind_opentrainer_agent.py"
SECRET = "tok-SECRET-7f3a9c"
PASSWORD = "pw-SECRET-b81e"


def _load_agent() -> Any:
    spec = importlib.util.spec_from_file_location("hivemind_opentrainer_agent_crowd", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(free: int = 7000, util: float | None = 0.0, temp: float | None = 60.0, total: int = 8192,
         apps: tuple[ComputeApp, ...] = (), name: str = "NVIDIA GeForce RTX 2070") -> GpuSnapshot:
    return GpuSnapshot(total_mib=total, used_mib=total - free, free_mib=free, util_pct=util, temp_c=temp,
                       name=name, apps=apps)


# --------------------------------------------------------------------------- log parsing


@pytest.mark.parametrize(
    ("line", "kind", "data"),
    [
        ("12:00:01 [INFO] [train] step 26/250 loss=4.0170 tps=345 left=44m", "step",
         {"step": 26, "target": 250, "loss": 4.017, "tps": 345, "left_min": 44}),
        ("12:40:00 [INFO] Done: 250 micro-steps, loss 4.2369 ± 0.8566 (mean of 250, last 4.4138)", "round_done",
         {"steps": 250, "mean_loss": 4.2369, "std": 0.8566, "mean_of": 250, "last": 4.4138}),
        ("12:41:00 [INFO] ✅ Submitted successfully!", "submitted", {}),
        ("12:41:10 [INFO] Round 2", "round_start", {"round": 2, "of": None}),
        ("12:41:10 [INFO] Round 1/1", "round_start", {"round": 1, "of": 1}),
        ("12:50:00 [INFO] Stopped by user. Skipping upload.", "stopped_by_user", {}),
    ],
)
def test_parse_real_client_lines(line: str, kind: str, data: dict[str, Any]) -> None:
    ev = cc.parse_client_line(line)
    assert ev is not None and ev.kind == kind
    for key, value in data.items():
        assert ev.data[key] == pytest.approx(value) if isinstance(value, float) else ev.data[key] == value


def test_parse_windows_mangled_glyphs_and_na_last() -> None:
    ev = cc.parse_client_line("[INFO] Done: 12 micro-steps, loss 5.0000 Â± 0.0000 (mean of 0, last n/a)\r\n")
    assert ev is not None and ev.kind == "round_done" and ev.data["last"] is None and ev.data["steps"] == 12
    assert cc.parse_client_line("[INFO] â\x9c\x85 Submitted successfully!").kind == "submitted"


def test_parse_failures_and_errors() -> None:
    lost = cc.parse_client_line("[INFO] ❌ Round lost — coordinator rejected the upload (HTTP 403): forbidden")
    assert lost is not None and lost.kind == "round_lost" and "403" in lost.data["detail"]
    assert cc.parse_client_line("[INFO] Upload failed: timeout").kind == "upload_failed"
    assert cc.parse_client_line("[INFO] ⏭️ Not counted by server (duplicate)").kind == "not_counted"
    assert cc.parse_client_line("Traceback (most recent call last):").kind == "error"
    assert cc.parse_client_line("[ERROR] Round failed: CUDA out of memory").kind == "error"
    assert cc.parse_client_line("[ERROR] [auth] login failed: bad").kind == "error"
    assert cc.parse_client_line("[INFO] Direct submit failed: x; trying chunked upload") is None
    assert cc.parse_client_line("[INFO] [status] uploading") is None
    assert cc.parse_client_line("") is None


# --------------------------------------------------------------------------- nvidia-smi and budget


def _fake_smi(gpu_out: str, apps_out: str = "", rc: int = 0):
    def run(argv: list[str], **_: Any) -> Any:
        if argv[0] != "nvidia-smi":
            raise AssertionError(argv)
        out = gpu_out if argv[1].startswith("--query-gpu") else apps_out
        return SimpleNamespace(returncode=rc, stdout=out, stderr="")
    return run


def test_query_nvidia_gpu_parses_csv_and_apps() -> None:
    s = cc.query_nvidia_gpu(_fake_smi("Tesla T4, 15360, 11000, 4360, 97, 71\n", "1234, python3, 10800\n5678, [N/A], [N/A]\n"))
    assert s == GpuSnapshot(15360, 11000, 4360, 97.0, 71.0, "Tesla T4",
                            (ComputeApp(1234, "python3", 10800), ComputeApp(5678, "[N/A]", None)))


def test_query_nvidia_gpu_na_temperature_and_failures() -> None:
    s = cc.query_nvidia_gpu(_fake_smi("RTX 2070, 8192, 1000, 7192, [N/A], [N/A]\n"))
    assert s is not None and s.temp_c is None and s.util_pct is None
    assert cc.query_nvidia_gpu(_fake_smi("", rc=9)) is None

    def missing(argv: list[str], **_: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")
    assert cc.query_nvidia_gpu(missing) is None


def test_budget_maths() -> None:
    assert cc.plan_vram_budget_gb(snap(free=7000), 1500) == 4.5  # (7000-1500-700)/1024 = 4.69 -> 4.5
    assert cc.plan_vram_budget_gb(snap(free=15000), 4300) == 9.5  # 9.76 -> 9.5
    assert cc.plan_vram_budget_gb(snap(free=3200), 1500) is None  # 0.98 GB < 1.0
    assert cc.plan_vram_budget_gb(snap(free=3250), 1500) == 1.0
    assert cc.plan_vram_budget_gb(snap(free=1000), 1500) is None
    assert cc.plan_vram_budget_gb(None, 1500) is None


# --------------------------------------------------------------------------- decide()

CFG = GuardConfig(reserve_mib=1500, breach_s=10, hot_c=82, cool_c=74, busy_util=15, quiet_s=60, relaunch_idle_s=300)


def _state(phase: Phase, now: float = 0.0, **kw: Any) -> GuardState:
    st = GuardState(phase=phase, now=now, client_pids=frozenset({100}), launched_at=0.0)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def test_breach_stops_only_after_breach_s() -> None:
    st = _state(Phase.RUNNING, now=1000)
    assert decide(st, snap(free=900), 4000, CFG) == Action.RUN
    st.now = 1005
    assert decide(st, snap(free=900), 4000, CFG) == Action.RUN
    st.now = 1010
    assert decide(st, snap(free=900), 4000, CFG) == Action.STOP and st.phase == Phase.STOPPED
    # a transient dip that recovers resets the timer
    st2 = _state(Phase.RUNNING, now=0)
    decide(st2, snap(free=900), 4000, CFG)
    st2.now = 5
    decide(st2, snap(free=5000), 4000, CFG)
    st2.now = 12
    assert decide(st2, snap(free=900), 4000, CFG) == Action.RUN


def test_paused_client_still_stopped_on_breach() -> None:
    st = _state(Phase.PAUSED, now=0)
    assert decide(st, snap(free=500, util=90), 4000, CFG) == Action.PAUSE
    st.now = 10
    assert decide(st, snap(free=500, util=90), 4000, CFG) == Action.STOP


def test_hot_pauses_and_hysteresis_requires_cool_and_quiet() -> None:
    st = _state(Phase.RUNNING, now=0)
    assert decide(st, snap(temp=83), 4000, CFG) == Action.PAUSE and st.phase == Phase.PAUSED
    st.now = 100
    assert decide(st, snap(temp=78), 4000, CFG) == Action.PAUSE  # below hot, above cool: stays paused
    st.now = 200
    assert decide(st, snap(temp=74), 4000, CFG) == Action.PAUSE  # cool: quiet timer starts
    st.now = 259
    assert decide(st, snap(temp=74), 4000, CFG) == Action.PAUSE
    st.now = 260
    assert decide(st, snap(temp=70), 4000, CFG) == Action.RUN and st.phase == Phase.RUNNING


def test_none_temperature_never_pauses() -> None:
    st = _state(Phase.RUNNING)
    assert decide(st, snap(temp=None, util=99), 4000, CFG) == Action.RUN


def test_busy_gpu_keeps_paused_client_paused_and_resets_quiet() -> None:
    st = _state(Phase.PAUSED, now=0)
    assert decide(st, snap(util=5), 4000, CFG) == Action.PAUSE
    st.now = 50
    assert decide(st, snap(util=60), 4000, CFG) == Action.PAUSE and st.quiet_since is None
    st.now = 100
    assert decide(st, snap(util=5), 4000, CFG) == Action.PAUSE
    st.now = 160
    assert decide(st, snap(util=5), 4000, CFG) == Action.RUN


def test_running_client_own_utilization_is_not_busy() -> None:
    assert decide(_state(Phase.RUNNING), snap(util=100), 4000, CFG) == Action.RUN


def test_other_compute_app_pauses_but_client_pids_do_not() -> None:
    mine = (ComputeApp(100, "python", 4000),)
    assert decide(_state(Phase.RUNNING), snap(apps=mine), 4000, CFG) == Action.RUN
    st = _state(Phase.RUNNING)
    assert decide(st, snap(apps=mine + (ComputeApp(999, "ollama", 2000),)), 4000, CFG) == Action.PAUSE
    assert "ollama" in st.reason


def test_pause_for_process_pauses_and_blocks_resume() -> None:
    st = _state(Phase.RUNNING, pause_for_running=("eldenring",))
    assert decide(st, snap(), 4000, CFG) == Action.PAUSE
    st.now = 1000
    assert decide(st, snap(), 4000, CFG) == Action.PAUSE
    st.pause_for_running = ()
    st.now = 1001
    decide(st, snap(), 4000, CFG)
    st.now = 1061
    assert decide(st, snap(), 4000, CFG) == Action.RUN


def test_pause_for_works_without_nvidia_snapshot() -> None:
    st = _state(Phase.RUNNING, pause_for_running=("game",))
    assert decide(st, None, 0, CFG) == Action.PAUSE


def test_idle_relaunch_after_stop_with_backoff() -> None:
    st = _state(Phase.STOPPED, now=0, quick_stops=1, client_pids=frozenset())
    assert decide(st, snap(free=7000, util=40), 0, CFG) == Action.WAIT  # busy
    st.now = 10
    assert decide(st, snap(free=7000, util=0), 0, CFG) == Action.WAIT  # idle timer starts
    st.now = 309
    assert decide(st, snap(free=7000, util=0), 0, CFG) == Action.WAIT
    st.now = 310
    assert decide(st, snap(free=7000, util=0), 0, CFG) == Action.RUN and st.phase == Phase.RUNNING
    # second quick stop doubles the idle requirement
    st2 = _state(Phase.STOPPED, now=0, quick_stops=2, client_pids=frozenset())
    decide(st2, snap(), 0, CFG)
    st2.now = 300
    assert decide(st2, snap(), 0, CFG) == Action.WAIT
    st2.now = 600
    assert decide(st2, snap(), 0, CFG) == Action.RUN


def test_relaunch_waits_for_budget_memory_release_and_cool() -> None:
    for s, mem in ((snap(free=2000), 0), (snap(), 3000), (snap(temp=80), 0),
                   (snap(apps=(ComputeApp(5, "game", 100),)), 0)):
        st = _state(Phase.STOPPED, now=0, client_pids=frozenset())
        decide(st, s, mem, CFG)
        st.now = 10_000
        assert decide(st, s, mem, CFG) == Action.WAIT, st.reason


def test_init_launches_immediately_when_idle_else_waits() -> None:
    st = GuardState()
    assert decide(st, snap(free=2000), 0, CFG) == Action.WAIT and st.phase == Phase.INIT
    assert decide(st, snap(), 0, CFG) == Action.RUN and st.phase == Phase.RUNNING


def test_quick_stop_counter() -> None:
    st = _state(Phase.RUNNING, now=0, launched_at=0.0)
    decide(st, snap(free=100), 1, CFG)
    st.now = 10
    assert decide(st, snap(free=100), 1, CFG) == Action.STOP and st.quick_stops == 1
    st2 = _state(Phase.RUNNING, now=5000, launched_at=0.0, quick_stops=3)
    decide(st2, snap(free=100), 1, CFG)
    st2.now = 5010
    decide(st2, snap(free=100), 1, CFG)
    assert st2.quick_stops == 1  # a long run resets the backoff


# --------------------------------------------------------------------------- preflight


def _doctor_run(report: dict[str, Any] | None, rc: int = 0, prefix: str = ""):
    calls: list[list[str]] = []

    def run(argv: list[str], **_: Any) -> Any:
        calls.append(argv)
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 2070, 8192, 1192, 7000, 0, 55\n", stderr="")
        stdout = prefix + (json.dumps(report) if report is not None else "doctor crashed")
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")
    run.calls = calls  # type: ignore[attr-defined]
    return run


def _good_report(**update: Any) -> dict[str, Any]:
    base = {"checked": True, "signature": True, "client_matches": True, "remote_version": "3.2", "error": None}
    base.update(update)
    return {"schema": "crowdgpt.doctor.v1", "client": {"version": "3.2"}, "update": base}


@pytest.fixture()
def client_dir(tmp_path: Path) -> Path:
    d = tmp_path / "CrowdGPT"
    d.mkdir()
    (d / "client.py").write_text("# placeholder, never executed in tests\n")
    return d


def test_preflight_accepts_signed_matching_release_even_when_doctor_exits_1(client_dir: Path) -> None:
    run = _doctor_run(_good_report(), rc=1, prefix="noise before json\n")
    res = cc.preflight_client(client_dir, "python3", run=run)
    assert res.ok and res.local_version == "3.2"
    assert run.calls[0][1:] == [str(client_dir / "client.py"), "--doctor", "--json", "--no-update"]


@pytest.mark.parametrize(
    ("report", "needle"),
    [
        (_good_report(client_matches=False), "differs from the signed release"),
        (_good_report(signature=False), "signature"),
        (_good_report(remote_version="3.3"), "version mismatch"),
        (_good_report(checked=False), "not checked"),
        (_good_report(error="ConnectionError: offline", checked=False), "offline"),
        (None, "no JSON"),
    ],
)
def test_preflight_refusals(client_dir: Path, report: dict[str, Any] | None, needle: str) -> None:
    res = cc.preflight_client(client_dir, "python3", run=_doctor_run(report))
    assert not res.ok and needle in res.reason


def test_preflight_missing_client(tmp_path: Path) -> None:
    res = cc.preflight_client(tmp_path, "python3", run=_doctor_run(_good_report()))
    assert not res.ok and "not found" in res.reason


# --------------------------------------------------------------------------- command and secrets


def test_command_anon_by_default_and_flags() -> None:
    argv = cc.build_client_command("py", "/c", vram_gb=4.5, backend="cuda", env={})
    assert argv[:2] == ["py", str(Path("/c") / "client.py")]
    for flag in ("--headless", "--no-update", "--no-tui", "--anon"):
        assert flag in argv
    assert argv[argv.index("--vram-gb") + 1] == "4.5" and argv[argv.index("--backend") + 1] == "cuda"
    assert "--rounds" not in argv
    assert cc.build_client_command("py", "/c", vram_gb=2, env={}, rounds=1)[-2:] == ["--rounds", "1"]


def test_command_never_contains_secrets() -> None:
    tok = cc.build_client_command("py", "/c", vram_gb=3, env={"CROWDGPT_TOKEN": SECRET})
    assert "--anon" not in tok and SECRET not in " ".join(tok)
    pw = cc.build_client_command("py", "/c", vram_gb=3, env={"CROWDGPT_USER": "alice", "CROWDGPT_PASSWORD": PASSWORD})
    assert "--anon" not in pw and pw[pw.index("--user") + 1] == "alice" and PASSWORD not in " ".join(pw)
    assert "--password" not in pw and "--token" not in tok


def test_anon_child_env_drops_secrets_and_sets_hf_home(tmp_path: Path) -> None:
    env = cc.build_client_env({"CROWDGPT_PASSWORD": PASSWORD, "PATH": "/bin"}, tmp_path)
    assert "CROWDGPT_PASSWORD" not in env and env["HF_HOME"] == str(tmp_path / "hf")
    assert env["PYTHONIOENCODING"] == "utf-8"
    kept = cc.build_client_env({"CROWDGPT_TOKEN": SECRET}, tmp_path)
    assert kept["CROWDGPT_TOKEN"] == SECRET  # opted-in account: passed by env only


def test_non_nvidia_requires_explicit_vram_gb() -> None:
    gb, msg = cc.resolve_vram_gb(0.0, None, CFG)
    assert gb is None and "--vram-gb" in msg and "unavailable" in msg
    gb, msg = cc.resolve_vram_gb(6.0, None, CFG)
    assert gb == 6.0 and "unavailable" in msg
    gb, _ = cc.resolve_vram_gb(8.0, snap(free=7000), CFG)
    assert gb == 4.5  # capped to the budget


def test_next_stop_at() -> None:
    now = dt.datetime(2026, 11, 14, 22, 30)
    assert cc.next_stop_at("23:15", now) == dt.datetime(2026, 11, 14, 23, 15)
    assert cc.next_stop_at("07:00", now) == dt.datetime(2026, 11, 15, 7, 0)
    with pytest.raises(ValueError):
        cc.next_stop_at("25:00", now)


# --------------------------------------------------------------------------- fakes for the supervisor


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeClientProcess:
    def __init__(self, pid: int, clock: FakeClock, script: list[tuple[float, str]]) -> None:
        self.pid, self.clock = pid, clock
        self.start = clock.t
        self.script = list(script)
        self.rc: int | None = None
        self.suspended = False

    def poll(self) -> int | None:
        return self.rc

    def drain_lines(self) -> list[str]:
        if self.suspended:
            return []
        due = [line for (t, line) in self.script if self.start + t <= self.clock.t]
        self.script = [(t, line) for (t, line) in self.script if self.start + t > self.clock.t]
        return due


class FakePsutil:
    def __init__(self) -> None:
        self.procs: dict[int, FakeClientProcess] = {}
        self.calls: list[tuple[str, int]] = []
        self.running_names: list[str] = []

    def Process(self, pid: int) -> Any:  # noqa: N802 - mirrors psutil API
        owner = self

        class P:
            def children(self, recursive: bool = False) -> list[Any]:
                return []

            def _rec(self, what: str) -> None:
                owner.calls.append((what, pid))

            def suspend(self) -> None:
                self._rec("suspend")
                owner.procs[pid].suspended = True

            def resume(self) -> None:
                self._rec("resume")
                owner.procs[pid].suspended = False

            def terminate(self) -> None:
                self._rec("terminate")
                owner.procs[pid].rc = 0  # client.py handles SIGTERM and exits 0

            def kill(self) -> None:
                self._rec("kill")
                owner.procs[pid].rc = -9
        return P()

    def process_iter(self, attrs: list[str]) -> list[Any]:
        return [SimpleNamespace(info={"name": n}) for n in self.running_names]


ROUND = [
    (5, "10:00:00 [INFO] Round 1"),
    (6, "10:00:01 [INFO] [train] step 26/250 loss=4.0170 tps=345 left=44m"),
    (10, "10:00:05 [INFO] Done: 250 micro-steps, loss 4.2369 ± 0.8566 (mean of 250, last 4.4138)"),
    (12, "10:00:07 [INFO] ✅ Submitted successfully!"),
]


def _supervisor(tmp_path: Path, snapshots, scripts, *, once: bool = False, max_ticks: int = 40, env=None,
                stop_at: float | None = None, out=None):
    clock = FakeClock()
    ps = FakePsutil()
    launched: list[list[str]] = []
    pids = iter(range(100, 1000, 100))
    script_iter = iter(scripts)

    def factory(argv: list[str], cwd: str, child_env: Any) -> FakeClientProcess:
        launched.append(argv)
        p = FakeClientProcess(next(pids), clock, next(script_iter))
        ps.procs[p.pid] = p
        return p

    def snapshot_fn() -> GpuSnapshot | None:
        return snapshots(clock.t, ps)

    lines: list[str] = []
    cfg = GuardConfig(reserve_mib=1500, breach_s=10, quiet_s=30, relaunch_idle_s=30)
    sup = cc.Supervisor(
        cc.SupervisorOptions(client_dir="/c", python="py", data_dir=str(tmp_path / "data"), once=once,
                             stop_at_epoch=stop_at, poll_s=5.0),
        cfg, snapshot_fn=snapshot_fn, process_factory=factory, psutil_mod=ps, clock=clock, sleep=clock.sleep,
        env=env or {}, out=out or lines.append, utc=lambda: "2026-11-14T00:00:00+00:00", max_ticks=max_ticks,
    )
    return sup, ps, launched, lines, clock


def _receipts(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "data" / cc.RECEIPTS_FILENAME
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_full_supervisor_run_with_stop_and_relaunch(tmp_path: Path) -> None:
    def snapshots(t: float, ps: FakePsutil) -> GpuSnapshot:
        alive = [p for p in ps.procs.values() if p.rc is None]
        apps = tuple(ComputeApp(p.pid, "python", 4000) for p in alive)
        util = 95.0 if any(not p.suspended for p in alive) else 0.0
        free = 7000 - 4000 * len(alive)
        if 40 <= t < 70:  # an inference server loads a model: another compute app + memory pressure
            apps += (ComputeApp(4242, "llama-server", 2600),)
            free -= 2600
            util = 100.0
        elif 70 <= t < 75:  # it is unloading
            util = max(util, 30.0)
        return snap(free=free, util=util, apps=apps)

    second_round = [(5, "11:00:00 [INFO] Round 1"),
                    (10, "11:00:05 [INFO] Done: 250 micro-steps, loss 4.1000 ± 0.5000 (mean of 250, last 4.0000)"),
                    (12, "11:00:07 [INFO] ✅ Submitted successfully!")]
    first = ROUND + [(20, "10:00:20 [INFO] Round 2"), (21, "10:00:21 [INFO] [train] step 3/250 loss=4.5 tps=300 left=50m")]
    sup, ps, launched, lines, clock = _supervisor(tmp_path, snapshots, [first, second_round], max_ticks=40)
    assert sup.run() == 0
    assert len(launched) == 2 and sup.stops == 1
    assert ("suspend", 100) in ps.calls and ("terminate", 100) in ps.calls
    assert ps.calls.index(("suspend", 100)) < ps.calls.index(("terminate", 100))
    assert ("kill", 100) not in ps.calls  # graceful SIGTERM was enough
    receipts = _receipts(tmp_path)
    assert [(r["round"], r["submitted"], r["outcome"]) for r in receipts] == [
        (1, True, "submitted"),
        (2, False, "round_lost_on_stop"),
        (3, True, "submitted"),
    ]
    assert receipts[0]["steps"] == 250 and receipts[0]["mean_loss"] == pytest.approx(4.2369)
    assert receipts[0]["evidence"] == "client_reported" and receipts[1]["evidence"] == "supervisor_observed"
    assert any("STOP" in ln for ln in lines) and any("PAUSE" in ln for ln in lines)
    # the relaunch recomputed the budget from a fresh snapshot (7000 free -> 4.5 GB)
    assert launched[1][launched[1].index("--vram-gb") + 1] == "4.5"


def test_upload_after_next_round_started_is_attributed_to_the_finished_round(tmp_path: Path) -> None:
    script = [(5, "[INFO] Round 1"),
              (10, "[INFO] Done: 250 micro-steps, loss 4.0 ± 0.1 (mean of 250, last 4.0)"),
              (15, "[INFO] Round 2"),
              (16, "[INFO] ✅ Submitted successfully!")]
    sup, *_ = _supervisor(tmp_path, lambda t, ps: snap(), [script], max_ticks=8)
    sup.run()
    receipts = _receipts(tmp_path)
    assert receipts[0]["round"] == 1 and receipts[0]["submitted"] is True
    assert receipts[1] == {**receipts[1], "round": 2, "submitted": False}


def test_once_passes_rounds_1_and_exits_when_client_exits(tmp_path: Path) -> None:
    sup, ps, launched, _, clock = _supervisor(tmp_path, lambda t, ps: snap(), [ROUND], once=True, max_ticks=50)
    orig_drain = FakeClientProcess.drain_lines

    def drain_then_exit(self: FakeClientProcess) -> list[str]:
        out = orig_drain(self)
        if not self.script:
            self.rc = 0
        return out
    FakeClientProcess.drain_lines = drain_then_exit  # type: ignore[method-assign]
    try:
        assert sup.run() == 0
    finally:
        FakeClientProcess.drain_lines = orig_drain  # type: ignore[method-assign]
    assert launched[0][-2:] == ["--rounds", "1"] and ("terminate", 100) not in ps.calls
    assert [r["submitted"] for r in _receipts(tmp_path)] == [True]


def test_fatal_client_exit_is_not_relaunched(tmp_path: Path) -> None:
    sup, ps, launched, lines, _ = _supervisor(tmp_path, lambda t, ps: snap(), [[]], max_ticks=10)
    sup.tick()
    ps.procs[100].rc = 2
    assert sup.tick() == 2 and len(launched) == 1
    assert any("authentication" in ln for ln in lines)


def test_pause_for_process_suspends_and_resumes(tmp_path: Path) -> None:
    sup, ps, *_ = _supervisor(tmp_path, lambda t, ps: snap(), [[]], max_ticks=10)
    sup.cfg.pause_for = ("eldenring.exe",)
    sup.tick()
    ps.running_names = ["EldenRing.exe"]
    sup.tick()
    assert ps.calls[-1] == ("suspend", 100)
    ps.running_names = []
    for _ in range(8):
        sup.clock.sleep(5)
        sup.tick()
    assert ps.calls[-1] == ("resume", 100) and sup.state.phase == Phase.RUNNING


def test_stop_at_terminates_and_exits(tmp_path: Path) -> None:
    sup, ps, *_ = _supervisor(tmp_path, lambda t, ps: snap(), [[(5, "[INFO] Round 1")]], stop_at=20.0, max_ticks=100)
    assert sup.run() == 0
    assert ("terminate", 100) in ps.calls
    assert _receipts(tmp_path)[0]["outcome"] == "round_lost_on_stop_at"


def test_supervisor_output_and_logs_redact_secrets(tmp_path: Path) -> None:
    env = {"CROWDGPT_TOKEN": SECRET}
    script = [(5, f"[INFO] debug token={SECRET}")]
    sup, ps, launched, lines, _ = _supervisor(tmp_path, lambda t, ps: snap(), [script], env=env, max_ticks=4)
    sup.run()
    assert SECRET not in " ".join(launched[0])
    assert all(SECRET not in ln for ln in lines)
    log = (tmp_path / "data" / cc.CLIENT_LOG_FILENAME).read_text()
    assert SECRET not in log and "***" in log


# --------------------------------------------------------------------------- run_contributor / agent wiring


def test_run_contributor_non_nvidia_requires_vram_gb(client_dir: Path, tmp_path: Path) -> None:
    def no_smi(argv: list[str], **_: Any) -> Any:
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError
        return SimpleNamespace(returncode=0, stdout=json.dumps(_good_report()), stderr="")

    out: list[str] = []
    args = cc.ContributorArgs(crowdgpt_dir=str(client_dir), data_dir=str(tmp_path / "d"), dry_run=True, backend="rocm")
    assert cc.run_contributor(args, run=no_smi, out=out.append, env={}) == 2
    text = "\n".join(out)
    assert "--vram-gb" in text and "UNAVAILABLE" in text
    out.clear()
    args.vram_gb = 6.0
    assert cc.run_contributor(args, run=no_smi, out=out.append, env={}) == 0
    assert "--vram-gb 6" in "\n".join(out) and "NVIDIA-only" in "\n".join(out)


def test_run_contributor_refuses_t4_without_4300_reserve(client_dir: Path, tmp_path: Path) -> None:
    def t4(argv: list[str], **_: Any) -> Any:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="Tesla T4, 15360, 2000, 13360, 0, 50\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(_good_report()), stderr="")
    out: list[str] = []
    args = cc.ContributorArgs(crowdgpt_dir=str(client_dir), data_dir=str(tmp_path), dry_run=True)
    assert cc.run_contributor(args, run=t4, out=out.append, env={}) == 2
    args.reserve_mib = 4300
    assert cc.run_contributor(args, run=t4, out=out.append, env={}) == 0


def test_run_contributor_preflight_refusal_starts_nothing(client_dir: Path, tmp_path: Path) -> None:
    def factory(*a: Any) -> Any:
        raise AssertionError("must not start the client")
    out: list[str] = []
    args = cc.ContributorArgs(crowdgpt_dir=str(client_dir), data_dir=str(tmp_path))
    rc = cc.run_contributor(args, run=_doctor_run(_good_report(client_matches=False)), process_factory=factory,
                            out=out.append, env={})
    assert rc == 3 and any("REFUSED" in ln for ln in out)


@pytest.fixture()
def agent() -> Any:
    return _load_agent()


def test_agent_dry_run_starts_no_process(agent: Any, client_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    run = _doctor_run(_good_report())
    started: list[Any] = []
    monkeypatch.setattr(cc, "_default_run", run)
    monkeypatch.setattr(cc, "_default_process_factory", lambda *a: started.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    monkeypatch.setenv("CROWDGPT_TOKEN", SECRET)
    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    data = tmp_path / "data"
    rc = agent.main(["--crowdgpt", "--crowdgpt-dir", str(client_dir), "--dry-run", "--data-dir", str(data),
                     "--pause-for", "game.exe"])
    out = capsys.readouterr().out
    assert rc == 0 and started == []
    assert any(c[0] != "nvidia-smi" and "--doctor" in c for c in run.calls)
    assert "Dry run, nothing started" in out and "0.9 GB" in out and "safetensors" in out
    assert "NVIDIA-only" in out and SECRET not in out
    assert not data.exists()


def test_agent_missing_crowdgpt_dir_prints_install_steps(agent: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cc, "_default_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run")))
    rc = agent.main(["--crowdgpt", "--crowdgpt-dir", str(tmp_path / "nope"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "git clone https://github.com/Vxtzq/CrowdGPT" in out and "pip install -r requirements_cuda.txt" in out


def test_agent_without_crowdgpt_keeps_heartbeat_requirement(agent: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    monkeypatch.setattr(agent, "detect_local_gpu", lambda: ("", 0, 0))
    assert agent.main(["--once"]) == 2


def test_real_popen_wrapper_merges_stderr_and_decodes_utf8(tmp_path: Path) -> None:
    """Offline: a tiny Python child writes real client lines to stderr (as client.py's logging does)."""
    code = (
        "import sys\n"
        "sys.stderr.write('12:00:00 [INFO] Done: 250 micro-steps, loss 4.2369 \u00b1 0.8566 (mean of 250, last 4.4138)\\n')\n"
        "sys.stderr.write('12:00:01 [INFO] \u2705 Submitted successfully!\\n')\n"
    )
    env = cc.build_client_env(dict(__import__("os").environ), tmp_path)
    proc = cc.PopenClientProcess([sys.executable, "-c", code], cwd=str(tmp_path), env=env)
    lines: list[str] = []
    import time as _time
    deadline = _time.monotonic() + 20
    while _time.monotonic() < deadline:
        lines += proc.drain_lines()
        if proc.poll() is not None and len(lines) >= 2:
            break
        _time.sleep(0.05)
    lines += proc.drain_lines()
    kinds = [ev.kind for ev in map(cc.parse_client_line, lines) if ev is not None]
    assert proc.poll() == 0 and kinds == ["round_done", "submitted"]


def test_non_finite_loss_written_as_null(tmp_path: Path) -> None:
    script = [(5, "[INFO] Round 1"), (6, "[INFO] Done: 3 micro-steps, loss nan ± nan (mean of 3, last inf)"),
              (7, "[INFO] ✅ Submitted successfully!")]
    sup, *_ = _supervisor(tmp_path, lambda t, ps: snap(), [script], max_ticks=4)
    sup.run()
    raw = (tmp_path / "data" / cc.RECEIPTS_FILENAME).read_text().splitlines()[0]
    assert "NaN" not in raw and "Infinity" not in raw and json.loads(raw)["mean_loss"] is None



def test_ignored_inference_servers_do_not_block_but_the_reserve_still_stops() -> None:
    """An always-on llama-server on the same GPU must not make the contributor wait forever."""
    from hivemind.opentrainer import crowdgpt_contributor as cc

    apps = (cc.ComputeApp(pid=1996, name="/opt/llama.cpp/build/bin/llama-server", used_mib=4818),
            cc.ComputeApp(pid=42, name="python client.py", used_mib=2000))
    snap = cc.GpuSnapshot(total_mib=15360, used_mib=6818, free_mib=8542, util_pct=0.0, temp_c=55.0, name="Tesla T4", apps=apps)
    assert [a.pid for a in cc.other_compute_apps(snap, {42})] == [1996]
    assert cc.other_compute_apps(snap, {42}, ("LLAMA-SERVER",)) == []
    assert [a.pid for a in cc.other_compute_apps(snap, {42}, ("ollama",))] == [1996]


def test_client_env_turns_on_mid_round_checkpoints_without_overriding_the_user(tmp_path) -> None:
    """CrowdGPT >= 3.3 resumes a stopped round from a local checkpoint every N steps."""
    from hivemind.opentrainer import crowdgpt_contributor as cc

    assert cc.build_client_env({}, tmp_path)["CROWDGPT_UNIT_CHECKPOINT_EVERY"] == "25"
    assert cc.build_client_env({"CROWDGPT_UNIT_CHECKPOINT_EVERY": "50"}, tmp_path)["CROWDGPT_UNIT_CHECKPOINT_EVERY"] == "50"
