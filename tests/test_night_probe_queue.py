"""CPU-only tests for scripts/night_probe_queue.py. Every external effect (clock, HTTP, nvidia-smi, subprocess, paths) is faked; no GPU, no network, no torch."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


q = _load("night_probe_queue_under_test", "scripts/night_probe_queue.py")
UTC = dt.timezone.utc


def utc(*a) -> dt.datetime:
    return dt.datetime(*a, tzinfo=UTC)


# --- fakes -------------------------------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self, start: dt.datetime) -> None:
        self.t, self.m = start, 0.0

    def now(self) -> dt.datetime:
        return self.t

    def mono(self) -> float:
        return self.m

    def sleep(self, s: float) -> None:
        self.t += dt.timedelta(seconds=s)
        self.m += s
        time.sleep(0.002)


class FakeProc:
    def __init__(self, pid: int, ignore_term: bool = False) -> None:
        self.pid, self.rc, self.ignore_term, self.signals, self.stdout = pid, None, ignore_term, [], None
        self._ev = threading.Event()

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        if not self._ev.wait(timeout):
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.rc

    def finish(self, rc: int, out: dict | None = None) -> None:
        if out is not None and self.stdout is not None:
            self.stdout.write((json.dumps(out) + "\n").encode())
            self.stdout.flush()
        self.rc = rc
        self._ev.set()

    def killpg(self, sig: int) -> None:
        if sig == 0:
            if self.rc is not None:
                raise ProcessLookupError
            return
        self.signals.append(sig)
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and not self.ignore_term):
            self.finish(-sig)


OK_JSON = {"status": "OK", "ok": True, "torch_peak_allocated_mib": 2900, "torch_peak_reserved_mib": 3100, "after_load_mib": 2444, "load_seconds": 10.0,
           "seconds_per_step": 1.9, "tokens_per_second": 134, "trainable_params": 1000, "total_params_b": 0.8}


class World:
    """Everything outside the process: GPU, nginx, Ollama, subprocess."""

    def __init__(self, tmp: Path, start: dt.datetime) -> None:
        self.clock = Clock(start)
        self.inflight, self.models, self.procs, self.temp, self.util, self.used, self.total, self.power = 0, [], [(777, "/x/llama-server", 4800)], 50, 0, 5000, 15360, 60.0
        self.mode, self.ignore_term, self.spawned, self.cmds, self.gpu_calls, self.on_gpu = "ok", False, [], [], 0, None
        self.ollama_error: Exception | None = None
        self.status_error: Exception | None = None
        self.hold, self.lock, self.out = tmp / "HOLD", tmp / "queue.lock", tmp / "out"
        self.next_pid = 5000
        self.boinc: set[int] = set()  # pids that count as BOINC QLoRA tasks
        self.sm: dict[str, int] = {}  # nvidia-smi pmon SM% by pid

    def http_get(self, url: str) -> str:
        if url.endswith("/api/ps"):
            if self.ollama_error:
                raise self.ollama_error
            return json.dumps({"models": [{"name": m} for m in self.models]})
        if self.status_error:
            raise self.status_error
        return f"Active connections: 3\nserver accepts handled requests\n 10 10 20\nReading: 0 Writing: {self.inflight + 1 if self.inflight >= 0 else 1} Waiting: 2\n"

    def gpu(self) -> dict:
        # numbering used by the triggers: 1 = the preflight sample (any earlier BOINC-wait samples do not count), 2.. = monitor ticks
        self.gpu_calls = self.gpu_calls + 1 if self.spawned else 1
        if self.on_gpu:
            self.on_gpu(self)
        return {"used_mib": self.used, "total_mib": self.total, "temp_c": self.temp, "util_pct": self.util, "power_w": self.power, "procs": list(self.procs)}

    def spawn(self, cmd, so, se, env):
        self.cmds.append(cmd)
        p = FakeProc(self.next_pid, self.ignore_term)
        self.next_pid += 1
        p.stdout = so
        self.spawned.append(p)
        if self.mode == "ok":
            p.finish(0, OK_JSON)
        elif self.mode == "oom":
            p.finish(1, {"status": "OOM", "ok": False, "error": "OutOfMemoryError: CUDA out of memory"})
        elif self.mode == "ckpt":
            p.finish(1, {"status": "CHECKPOINTING_UNSUPPORTED", "ok": False, "error": "x"})
        elif self.mode == "crash":
            se.write(b"Traceback ... RuntimeError: CUDA out of memory\n")
            se.flush()
            p.finish(1)
        return p

    def killpg(self, pgid: int, sig: int) -> None:
        for p in self.spawned:
            if p.pid == pgid:
                return p.killpg(sig)
        raise ProcessLookupError

    def pmon(self) -> str:
        return "# gpu pid type sm mem enc dec command\n" + "".join(f"    0 {pid} C {sm} 0 0 0 proc\n" for pid, sm in self.sm.items())

    def env(self, estimate=None):
        return q.Env(now=self.clock.now, mono=self.clock.mono, sleep=self.clock.sleep, http_get=self.http_get, gpu=self.gpu, spawn=self.spawn, killpg=self.killpg,
                     descendants=lambda pid: [], estimate=estimate, environ={}, pmon=self.pmon, parse_pmon=q.governor().parse_pmon,
                     boinc_pids=lambda procs: [str(p[0]) for p in procs if p[0] in self.boinc])

    def options(self, **kw) -> "q.Options":
        d = dict(out_dir=self.out, hold_file=self.hold, lock_file=self.lock, poll_s=1.0, kill_grace_s=0.3, arms=["A0"])
        d.update(kw)
        return q.Options(**d)


NIGHT = utc(2026, 10, 27, 22, 30)  # 00:30 Paris (CET/CEST: Oct 27 is after the DST end of Oct 25, so UTC+1)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path, NIGHT)


ARMS = q.load_arms()
FAKE_EST = lambda arm: {"available": True, "alloc_peak_mib": 2843, "process_peak_mib": 3150, "process_peak_with_20pct_margin_mib": 3780}  # noqa: E731


def run(world: World, **kw):
    o = world.options(**kw)
    rc = q.run_queue(world.env(), o, ARMS, FAKE_EST, install_signals=False)
    return rc, o


def arm_json(world: World, arm="A0") -> dict:
    return json.loads((world.out / f"arm_{arm}.json").read_text())


# --- window -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("when,expected", [
    (utc(2026, 10, 20, 20, 59), False),  # 22:59 Paris (CEST, UTC+2)
    (utc(2026, 10, 20, 21, 0), True),    # 23:00
    (utc(2026, 10, 21, 4, 29), True),    # 06:29
    (utc(2026, 10, 21, 4, 30), False),   # 06:30
    (utc(2026, 12, 1, 21, 59), False),   # 22:59 Paris (CET, UTC+1)
    (utc(2026, 12, 1, 22, 0), True),
    (utc(2026, 12, 2, 5, 29), True),
    (utc(2026, 12, 2, 5, 30), False),
    (utc(2026, 10, 24, 21, 0), True),    # 23:00 CEST the evening before the DST end
    (utc(2026, 10, 24, 22, 0), True),    # the transition night (02:00 CET after the clocks go back) stays inside the window
    (utc(2026, 10, 25, 4, 29), True),    # 05:29 CET after the clocks went back at 01:00Z
    (utc(2026, 10, 25, 5, 30), False),   # 06:30 CET
    (utc(2026, 3, 28, 21, 59), False),   # 22:59 CET the evening before the DST start
    (utc(2026, 3, 28, 22, 0), True),     # 23:00 CET
    (utc(2026, 3, 29, 4, 29), True),     # 06:29 CEST
    (utc(2026, 3, 29, 4, 30), False),    # 06:30 CEST is 04:30Z
    (utc(2026, 10, 27, 12, 0), False),
])
def test_window_boundaries(when, expected):
    assert q.in_window(when) is expected


def test_window_requires_aware_datetime():
    with pytest.raises(ValueError):
        q.in_window(dt.datetime(2026, 10, 27, 23, 30))


def test_outside_window_refuses_and_never_touches_hold(world):
    world.clock.t = utc(2026, 10, 27, 12, 0)
    rc, o = run(world)
    assert rc == 2 and not world.hold.exists() and not world.spawned and not world.out.exists()


def test_ignore_window_runs_and_records_it(world):
    world.clock.t = utc(2026, 10, 27, 12, 0)
    rc, _ = run(world, ignore_window=True)
    assert rc == 0 and arm_json(world)["window_ignored"] is True
    assert json.loads((world.out / "summary.json").read_text())["ignore_window"] is True


def test_window_checked_again_before_each_arm(world):
    world.clock.t = utc(2026, 10, 27, 5, 29, 59)  # 06:29:59 Paris (CET): the first arm starts inside
    seen = []
    orig = world.spawn

    def spawn(cmd, so, se, env):
        seen.append(cmd[2])
        world.clock.t = utc(2026, 10, 27, 5, 30, 1)  # 06:30:01 Paris
        return orig(cmd, so, se, env)

    world.spawn = spawn
    rc, _ = run(world, arms=["A0", "A1"])
    assert len(seen) == 1
    assert arm_json(world, "A1")["status"] == "OUTSIDE_WINDOW" and rc == 1


def test_window_end_stops_a_running_arm(world):
    world.clock.t = utc(2026, 10, 27, 5, 29, 57)  # 06:29:57 Paris
    world.mode = "hang"
    rc, _ = run(world)
    r = arm_json(world)
    assert r["status"] == "OUTSIDE_WINDOW" and rc == 1
    assert world.spawned[0].signals[0] == signal.SIGTERM and world.spawned[0].rc is not None
    assert not world.hold.exists()


# --- HOLD ---------------------------------------------------------------------------------------------------------------------------------
def test_hold_exists_during_run_and_is_removed_after(world):
    seen = {}
    orig = world.spawn

    def spawn(cmd, so, se, env):
        seen["hold"] = world.hold.exists()
        return orig(cmd, so, se, env)

    world.spawn = spawn
    rc, _ = run(world)
    s = json.loads((world.out / "summary.json").read_text())
    assert seen["hold"] is True and not world.hold.exists() and rc == 0
    assert s["hold_preexisting"] is False and s["hold_created_by_queue"] is True and s["hold_removed_by_queue"] is True


def test_preexisting_hold_is_kept_and_recorded(world):
    world.hold.write_text("owner\n")
    rc, _ = run(world)
    s = json.loads((world.out / "summary.json").read_text())
    assert world.hold.read_text() == "owner\n"
    assert s["hold_preexisting"] is True and s["hold_created_by_queue"] is False and s["hold_removed_by_queue"] is False
    assert arm_json(world)["hold_preexisting"] is True


def test_hold_removed_when_an_exception_escapes(world):
    def boom(cmd, so, se, env):
        assert world.hold.exists()
        raise RuntimeError("spawn failed")

    world.spawn = boom
    with pytest.raises(RuntimeError):
        run(world)
    assert not world.hold.exists()
    assert not q.QueueLock.is_busy(world.lock)  # the lock is released too


def test_hold_removed_on_sigterm_and_current_arm_stopped(world):
    world.mode = "hang"
    orig = world.spawn

    def spawn(cmd, so, se, env):
        p = orig(cmd, so, se, env)
        os.kill(os.getpid(), signal.SIGTERM)  # delivered to the handler installed by run_queue
        return p

    world.spawn = spawn
    previous = signal.getsignal(signal.SIGTERM)
    o = world.options(arms=["A0", "A1"])
    rc = q.run_queue(world.env(), o, ARMS, FAKE_EST, install_signals=True)
    assert rc == 130 and not world.hold.exists()
    assert arm_json(world)["status"] == "INTERRUPTED" and not (world.out / "arm_A1.json").exists()
    assert world.spawned[0].rc is not None and len(world.spawned) == 1
    assert signal.getsignal(signal.SIGTERM) is previous  # handler restored
    assert json.loads((world.out / "summary.json").read_text())["interrupted"] is True


def test_hold_acquire_release_unit(tmp_path):
    h = q.Hold(tmp_path / "sub" / "HOLD")
    h.acquire()
    assert (tmp_path / "sub" / "HOLD").exists() and h.created
    h.release()
    h.release()  # idempotent
    assert not (tmp_path / "sub" / "HOLD").exists()


# --- never evict / blocked ------------------------------------------------------------------------------------------------------------------
def test_resident_ollama_model_blocks_the_arm_and_nothing_is_started(world):
    world.models = ["qwen2.5-coder:7b-instruct"]
    rc, _ = run(world)
    r = arm_json(world)
    assert r["status"] == "BLOCKED" and "resident" in r["reason"] and not world.spawned and rc == 1


def test_foreign_gpu_process_blocks_but_allowlisted_bee_server_does_not(world):
    world.procs = [(777, "/x/llama-server", 4800), (888, "/usr/bin/python3", 500)]
    run(world)
    r = arm_json(world)
    assert r["status"] == "BLOCKED" and "python3" in r["reason"] and not world.spawned


def test_allowlist_is_configurable(world):
    world.procs = [(888, "/usr/bin/other-server", 500)]
    run(world, allow_gpu=["other-server"])
    assert arm_json(world)["status"] == "OK"


def test_ollama_unreachable_fails_closed_but_refused_means_nothing_resident(world):
    world.ollama_error = urllib.error.URLError(TimeoutError("timed out"))
    run(world)
    assert arm_json(world)["status"] == "BLOCKED" and not world.spawned
    world.ollama_error = urllib.error.URLError(ConnectionRefusedError())
    run(world)
    assert arm_json(world)["status"] == "OK"


def test_queue_stops_after_a_block(world):
    world.models = ["m"]
    run(world, arms=["A0", "A1"])
    assert not (world.out / "arm_A1.json").exists()


def test_requests_in_flight_before_start_block(world):
    world.inflight = 1
    run(world)
    assert arm_json(world)["status"] == "BLOCKED" and "in flight" in arm_json(world)["reason"]


def test_busy_gpu_utilisation_blocks(world):
    world.util = 35
    run(world)
    assert arm_json(world)["status"] == "BLOCKED" and "utilisation" in arm_json(world)["reason"]


def test_nginx_unreadable_fails_closed(world):
    world.status_error = OSError("refused")
    run(world)
    assert arm_json(world)["status"] == "BLOCKED"


def test_cap_is_min_of_max_cap_and_free_minus_margin_by_default(world):
    world.used = 5000  # free 10360
    run(world, arms=["A2"], max_cap_mib=9500)
    r = arm_json(world, "A2")
    assert r["cap_mib"] == 9500 and r["cap_basis"]["free_vram_mib"] == 10360 and r["cap_basis"]["tier_cap_applied"] is False
    world.used = 12000  # free 3360 - 600 = 2760
    run(world, arms=["A2"], out_dir=world.out / "b")
    assert json.loads((world.out / "b" / "arm_A2.json").read_text())["cap_mib"] == 2760
    world.used = 14500  # free 860 - 600 = 260 < min cap
    run(world, arms=["A2"], out_dir=world.out / "c")
    assert json.loads((world.out / "c" / "arm_A2.json").read_text())["status"] == "BLOCKED"


def test_n0_arm_uses_the_lead_formula_by_default_and_the_tier_cap_only_when_asked(world):
    run(world, arms=["A0"])
    assert arm_json(world)["cap_mib"] == 9500  # min(9500, 10360 - 600): no silent tier override
    i = world.cmds[0].index("--cap-mib")
    assert world.cmds[0][i + 1] == "9500"
    run(world, arms=["A0"], tier_cap=True, out_dir=world.out / "t")
    assert json.loads((world.out / "t" / "arm_A0.json").read_text())["cap_mib"] == 3950


def test_fits_tier_budget_from_the_probes_own_nvidia_smi_memory(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 2:
            w.procs = w.procs + [(w.spawned[0].pid, "python", 4600)]  # above the N0 budget of 4300
        if w.gpu_calls == 4:
            w.spawned[0].finish(0, OK_JSON)

    world.on_gpu = trigger
    run(world)
    r = arm_json(world)
    assert r["tier_budget_mib"] == 4300 and r["measured"]["nvidia_smi_probe_peak_used_mib"] == 4600 and r["fits_tier_budget"] is False


def test_oom_under_a_cap_that_allowed_the_whole_budget_means_does_not_fit(world):
    world.mode = "oom"
    run(world)  # cap 9500 >= 4300
    assert arm_json(world)["fits_tier_budget"] is False
    run(world, tier_cap=True, out_dir=world.out / "t")  # cap 3950 < 4300: not determined
    assert json.loads((world.out / "t" / "arm_A0.json").read_text())["fits_tier_budget"] is None


# --- BOINC task still on the GPU when HOLD appears ------------------------------------------------------------------------------------------------
def test_waits_for_a_running_boinc_task_to_leave_then_runs(world):
    world.boinc = {4242}
    world.procs = world.procs + [(4242, "/venv/bin/python", 3700)]

    def trigger(w):
        if w.clock.m >= 20 and 4242 in w.boinc:  # the governor sees HOLD and BOINC stops the task about 20 s later
            w.boinc, w.procs = set(), [p for p in w.procs if p[0] != 4242]

    world.on_gpu = trigger
    rc, _ = run(world)
    r = arm_json(world)
    assert r["status"] == "OK" and rc == 0
    assert 20 <= r["preflight"]["boinc_wait_s"] < 25 and len(world.spawned) == 1


def test_boinc_task_that_never_leaves_blocks_naming_boinc_and_is_not_killed(world):
    world.boinc = {4242}
    world.procs = world.procs + [(4242, "/venv/bin/python", 3700)]
    run(world, boinc_wait_s=90)
    r = arm_json(world)
    assert r["status"] == "BLOCKED" and "BOINC" in r["reason"] and not world.spawned
    assert r["preflight"]["boinc_wait_s"] >= 90
    assert all(not p.signals for p in world.spawned)  # nothing was signalled
    assert not world.hold.exists()


# --- traffic guard --------------------------------------------------------------------------------------------------------------------------
def test_nginx_request_aborts_the_arm(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 4:
            w.inflight = 1

    world.on_gpu = trigger
    rc, _ = run(world)
    r = arm_json(world)
    assert r["status"] == "ABORTED_BY_TRAFFIC" and "in flight" in r["reason"] and rc == 1
    assert world.spawned[0].signals[0] == signal.SIGTERM and world.spawned[0].rc is not None
    assert not world.hold.exists()


def test_new_foreign_gpu_process_aborts_the_arm(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 3:
            w.procs = w.procs + [(9999, "/usr/bin/ollama", 6000)]

    world.on_gpu = trigger
    run(world)
    r = arm_json(world)
    assert r["status"] == "ABORTED_BY_TRAFFIC" and "ollama" in r["reason"]


def test_probe_own_gpu_process_is_not_foreign(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 2:
            w.procs = w.procs + [(w.spawned[0].pid, "python", 3000)]  # the probe itself, then a child of the probe
        if w.gpu_calls == 3:
            w.procs = w.procs + [(w.spawned[0].pid + 1, "python", 100)]
        if w.gpu_calls == 6:
            w.spawned[0].finish(0, OK_JSON)

    world.on_gpu = trigger
    env = world.env()
    env.descendants = lambda pid: [pid + 1]
    o = world.options()
    assert q.run_queue(env, o, ARMS, FAKE_EST, install_signals=False) == 0
    assert arm_json(world)["status"] == "OK"


def test_direct_use_of_the_allowlisted_bee_server_aborts_via_pmon(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 3:
            w.sm = {"5000": 90, "777": 30}  # the probe is busy (ours), the bee server (777, allow-listed, already resident) starts computing

    world.on_gpu = trigger
    run(world)
    r = arm_json(world)
    assert r["status"] == "ABORTED_BY_TRAFFIC" and "pid 777" in r["reason"]


def test_probe_own_sm_utilisation_is_not_traffic(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 2:
            w.sm = {"5000": 95, "5001": 80}
        if w.gpu_calls == 5:
            w.spawned[0].finish(0, OK_JSON)

    world.on_gpu = trigger
    env = world.env()
    env.descendants = lambda pid: [pid + 1]
    assert q.run_queue(env, world.options(), ARMS, FAKE_EST, install_signals=False) == 0


def test_ollama_loading_a_model_during_the_arm_aborts(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 3:
            w.models = ["qwen2.5-coder:7b-instruct"]

    world.on_gpu = trigger
    run(world)
    assert arm_json(world)["status"] == "ABORTED_BY_TRAFFIC" and "Ollama" in arm_json(world)["reason"]


def test_sigkill_follows_sigterm_when_the_probe_ignores_it(world):
    world.mode, world.ignore_term = "hang", True

    def trigger(w):
        if w.gpu_calls == 2:
            w.inflight = 2

    world.on_gpu = trigger
    t0 = time.monotonic()
    run(world, kill_grace_s=0.3)
    assert world.spawned[0].signals == [signal.SIGTERM, signal.SIGKILL]
    assert time.monotonic() - t0 < 3


def test_stub_status_parser_matches_the_governors():
    gov = _load("t4_governor_under_test", "scripts/t4_governor.py")
    txt = "Active connections: 5\nserver accepts handled requests\n 100 100 250\nReading: 1 Writing: 3 Waiting: 1\n"

    class R:
        def read(self):
            return txt.encode()

    orig = gov.urllib.request.urlopen
    gov.urllib.request.urlopen = lambda *a, **k: R()
    try:
        assert gov.stub_status("http://x") == q.parse_stub_status(txt)
    finally:
        gov.urllib.request.urlopen = orig
    assert q.in_flight(q.parse_stub_status(txt)) == 2  # writing 3 minus the poll itself, as the governor computes it
    assert q.in_flight({"writing": 0}) == 0


CHILD = textwrap.dedent("""
    import os, signal, subprocess, sys, time
    pidfile = sys.argv[sys.argv.index("--pidfile") + 1] if "--pidfile" in sys.argv else None
    if os.environ.get("IGNORE_TERM"): signal.signal(signal.SIGTERM, signal.SIG_IGN)
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    open(os.environ["PIDFILE"], "w").write(str(os.getpid()) + " " + str(g.pid))
    time.sleep(120)
""")


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


@pytest.mark.parametrize("ignore_term,limit", [(False, 3.5), (True, 3.5)])
def test_real_process_group_is_killed_within_the_polling_interval(world, tmp_path, monkeypatch, ignore_term, limit):
    probe = tmp_path / "fake_probe.py"
    probe.write_text(CHILD)
    pidfile = tmp_path / "pids"
    monkeypatch.setenv("PIDFILE", str(pidfile))
    if ignore_term:
        monkeypatch.setenv("IGNORE_TERM", "1")
    env = world.env()
    env.spawn, env.killpg, env.environ = q.real_spawn, q.real_killpg, dict(os.environ)
    env.sleep = time.sleep
    env.now = lambda: dt.datetime.now(UTC)
    env.mono = time.monotonic
    o = world.options(probe=probe, poll_s=0.05, kill_grace_s=2.0, ignore_window=True, arm_timeout_s=60)
    res = {}
    th = threading.Thread(target=lambda: res.update(rc=q.run_queue(env, o, ARMS, FAKE_EST, install_signals=False)))
    th.start()
    deadline = time.time() + 15
    while not pidfile.exists() or " " not in pidfile.read_text():
        assert time.time() < deadline, "fake probe did not start"
        time.sleep(0.02)
    child, grand = map(int, pidfile.read_text().split())
    assert _alive(child) and _alive(grand)
    t0 = time.monotonic()
    world.inflight = 1  # a user request arrives
    th.join(20)
    elapsed = time.monotonic() - t0
    assert not th.is_alive()
    r = arm_json(world)
    assert r["status"] == "ABORTED_BY_TRAFFIC"
    assert not _alive(child) and not _alive(grand), "the whole process group must be dead"
    assert elapsed < limit + 0.5 and r["stop_latency_s"] < 2.6
    assert not world.hold.exists()


# --- sampler / temperature / timeout ------------------------------------------------------------------------------------------------------------
def test_temperature_above_limit_stops_the_arm(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 4:
            w.temp = 84

    world.on_gpu = trigger
    run(world)
    r = arm_json(world)
    assert r["status"] == "TOO_HOT" and r["measured"]["max_temp_c"] == 84
    assert world.spawned[0].rc is not None


def test_sampler_records_max_temperature_power_and_memory(world):
    world.mode = "hang"

    def trigger(w):
        if w.gpu_calls == 3:
            w.temp, w.power, w.used = 71, 68.5, 9000
        if w.gpu_calls == 4:
            w.temp, w.power, w.used = 66, 40.0, 6000
        if w.gpu_calls == 6:
            w.spawned[0].finish(0, OK_JSON)

    world.on_gpu = trigger
    run(world)
    m = arm_json(world)["measured"]
    assert (m["max_temp_c"], m["max_power_w"], m["nvidia_smi_peak_used_mib"]) == (71, 68.5, 9000) and m["samples"] >= 3


def test_arm_timeout(world):
    world.mode = "hang"
    run(world, arm_timeout_s=5)
    assert arm_json(world)["status"] == "TIMEOUT"


def test_temperature_already_high_before_start_blocks_as_too_hot(world):
    world.temp = 90
    run(world)
    assert arm_json(world)["status"] == "TOO_HOT" and not world.spawned


# --- results ------------------------------------------------------------------------------------------------------------------------------
def test_result_schema_and_estimate_comparison(world):
    rc, _ = run(world)
    r = arm_json(world)
    for k in ("schema", "arm", "status", "model_id", "recipe", "cap_mib", "cap_basis", "estimate", "measured", "estimate_vs_measured", "probe", "cmd", "started_utc", "ended_utc",
              "window_ignored", "hold_preexisting", "reason", "returncode", "duration_s"):
        assert k in r, k
    assert r["status"] == "OK" and r["recipe"] == {"rank": 8, "alpha": 16, "seq_len": 128, "micro_batch": 2, "checkpointing": False,
                                                   "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    assert r["estimate_vs_measured"] == {"available": True, "estimated_alloc_peak_mib": 2843, "measured_alloc_peak_mib": 2900, "measured_minus_estimated_mib": 57,
                                         "measured_over_estimated": 1.02}
    assert r["measured"]["tokens_per_second"] == 134 and r["measured"]["torch_peak_reserved_mib"] == 3100
    s = json.loads((world.out / "summary.json").read_text())
    assert s["schema"] == "night_probe_summary/1" and s["arms"][0]["arm"] == "A0" and s["arms"][0]["measured_torch_peak_allocated_mib"] == 2900
    assert (world.out / "arm_A0.probe.stdout.log").exists()


def test_estimator_unavailable_is_stated_not_invented(world):
    o = world.options()
    q.run_queue(world.env(), o, ARMS, lambda a: {"available": False, "reason": "no config"}, install_signals=False)
    r = arm_json(world)
    assert r["estimate"] == {"available": False, "reason": "no config"} and r["estimate_vs_measured"] == {"available": False}


def test_estimator_reports_missing_hub_honestly(tmp_path):
    est = q.Estimator(hub=str(tmp_path / "nohub"))
    e = est(ARMS[0])
    assert e["available"] is False and "config.json" in e["reason"]


@pytest.mark.parametrize("mode,status", [("oom", "OOM"), ("ckpt", "CHECKPOINTING_UNSUPPORTED"), ("crash", "OOM")])
def test_probe_statuses_are_mapped(world, mode, status):
    world.mode = mode
    rc, _ = run(world)
    assert arm_json(world)["status"] == status and rc == 0  # conclusive results are not failures of the queue


def test_resume_skips_finished_arms_and_retries_others(world):
    run(world, arms=["A0", "A1"])
    assert len(world.spawned) == 2
    (world.out / "arm_A1.json").write_text(json.dumps({"arm": "A1", "status": "ABORTED_BY_TRAFFIC"}))
    rc, _ = run(world, arms=["A0", "A1"])
    assert len(world.spawned) == 3  # only A1 re-ran
    assert arm_json(world, "A1")["status"] == "OK" and rc == 0
    s = json.loads((world.out / "summary.json").read_text())
    assert [a["skipped_resume"] for a in s["arms"]] == [True, None]


def test_rc_one_when_an_arm_is_not_conclusive(world):
    world.mode = "hang"
    world.on_gpu = lambda w: setattr(w, "inflight", 1) if w.gpu_calls == 3 else None
    rc, _ = run(world)
    assert rc == 1


# --- lock ---------------------------------------------------------------------------------------------------------------------------------
def test_second_queue_is_refused_while_the_lock_is_held(world):
    first = q.QueueLock(world.lock)
    first.acquire()
    try:
        rc, _ = run(world)
        assert rc == 4 and not world.spawned and not world.hold.exists()
        assert q.QueueLock.is_busy(world.lock) is True
    finally:
        first.release()
    assert q.QueueLock.is_busy(world.lock) is False
    rc, _ = run(world)
    assert rc == 0


# --- dry run / list / arms -----------------------------------------------------------------------------------------------------------------
def _tree(root: Path) -> set:
    return {(str(p.relative_to(root)), p.stat().st_mtime_ns) for p in root.rglob("*")}


def test_dry_run_writes_nothing_and_never_creates_hold(world, tmp_path, capsys):
    before = _tree(tmp_path)
    o = world.options(arms=["A0", "A2"])
    rc = q.dry_run(world.env(), o, q.select_arms(ARMS, o.arms), FAKE_EST)
    out = capsys.readouterr().out
    assert rc == 0 and "would RUN" in out and "DRY RUN" in out
    assert _tree(tmp_path) == before and not world.hold.exists() and not world.out.exists() and not world.lock.exists() and not world.spawned


def test_dry_run_reports_blocking_preconditions(world, capsys):
    world.models = ["coder"]
    world.clock.t = utc(2026, 10, 27, 12, 0)
    o = world.options()
    rc = q.dry_run(world.env(), o, q.select_arms(ARMS, o.arms), FAKE_EST)
    out = capsys.readouterr().out
    assert rc == 3 and "OUTSIDE" in out and "would be BLOCKED" in out and not world.hold.exists()


def test_arms_follow_the_preregistration():
    assert [a.arm_id for a in ARMS] == ["A0", "A1", "A2", "A3"]
    a0, a1, a2, a3 = ARMS
    assert (a0.model_id, a0.rank, a0.alpha, a0.seq_len, a0.micro_batch, a0.checkpointing, a0.targets, a0.tier) == ("ibm-granite/granite-3.1-1b-a400m-base", 8, 16, 128, 2, False, q.QKVO, "N0")
    assert (a1.rank, a1.alpha, a1.seq_len, a1.micro_batch, a1.checkpointing) == (32, 64, 256, 2, False)
    assert (a2.model_id, a2.rank, a2.alpha, a2.seq_len, a2.micro_batch, a2.checkpointing, a2.tier) == ("ibm-granite/granite-3.1-3b-a800m-base", 16, 32, 256, 2, True, "N1")
    assert (a3.model_id, a3.rank, a3.alpha, a3.seq_len, a3.micro_batch, a3.checkpointing, a3.targets, a3.tier) == ("Qwen/Qwen3-4B-Instruct-2507", 16, 32, 256, 1, True, q.ALL_LINEAR, "N1")


def test_select_arms_and_unknown_arm():
    assert [a.arm_id for a in q.select_arms(ARMS, ["A3", "A0"])] == ["A0", "A3"]
    with pytest.raises(SystemExit):
        q.select_arms(ARMS, ["A9"])


def test_probe_command_maps_each_arm_to_the_new_flags():
    o = q.Options(python="py", probe=Path("probe.py"))
    c0 = q.probe_cmd(o, ARMS[0], 3950)
    assert c0 == ["py", "probe.py", "ibm-granite/granite-3.1-1b-a400m-base", "--cap-mib", "3950", "--rank", "8", "--alpha", "16", "--seq-len", "128", "--micro-batch", "2", "--max-gpu-temp-c", "83"]
    c2 = q.probe_cmd(o, ARMS[2], 9500)
    assert "--checkpointing" in c2 and "--targets" not in c2 and c2[c2.index("--rank") + 1] == "16"
    c3 = q.probe_cmd(o, ARMS[3], 9500)
    assert c3[c3.index("--targets") + 1] == "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" and "--checkpointing" in c3


def test_probe_command_is_accepted_by_the_probes_own_parser():
    probe = _load("t4_probe_for_queue_test", "scripts/t4_model_probe.py")
    for arm in ARMS:
        cmd = q.probe_cmd(q.Options(), arm, 5000)
        a = probe.build_parser().parse_args(cmd[2:])
        assert probe.build_spec(a)["rank"] == arm.rank and probe.build_spec(a)["alpha"] == arm.alpha and a.checkpointing == arm.checkpointing
        assert a.seq_len == arm.seq_len and a.micro_batch == arm.micro_batch and a.cap_mib == 5000


def test_list_needs_no_gpu_or_network(capsys, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("external effect in --list")

    env = q.Env(gpu=forbidden, http_get=forbidden, spawn=forbidden, descendants=lambda p: [], estimate=FAKE_EST)
    assert q.main(["--list", "--arms", "A0,A1"], env) == 0
    out = capsys.readouterr().out
    assert "A0" in out and "A1" in out and "A2" not in out


def test_source_has_no_dynamic_code_execution():
    src = (ROOT / "scripts" / "night_probe_queue.py").read_text()
    for bad in ("ev" + "al(", "ex" + "ec(", "shell" + "=True", "pick" + "le", "os." + "system"):
        assert bad not in src, bad
