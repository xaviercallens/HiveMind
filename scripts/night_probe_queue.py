#!/usr/bin/env python3
"""Guarded queue of NIGHT-INTENSE-1 memory/speed feasibility probes (arms A0..A3) on the shared Tesla T4.

A thin wrapper around ``scripts/t4_model_probe.py`` (one subprocess per arm, one arm at a time). It is a FEASIBILITY probe on random token ids: it measures peak
VRAM and speed, never held-out quality. It does not train anything. Python 3.10, standard library only.

Safety contract (each point is enforced in code and covered by tests/test_night_probe_queue.py):
  WINDOW   runs only inside 23:00-06:30 Europe/Paris (checked at start, before every arm, and every second while an arm runs; an arm still running at the end is stopped).
  HOLD     creates the governor's HOLD file for the whole run and removes it on every exit path (finally, atexit, SIGTERM, SIGINT); a HOLD that existed before is never removed.
  NO EVICT reads Ollama ``/api/ps`` (GET only); a resident model, or a GPU compute process whose name is not allow-listed, makes the arm BLOCKED. Nothing is ever unloaded.
  TRAFFIC  a monitor thread polls nginx stub_status (the governor's in-flight signal) and the GPU process list every second; any user request or any new GPU process
           sends SIGTERM, then SIGKILL after 2 s, to the probe's whole process group (status ABORTED_BY_TRAFFIC).
  SAMPLER  nvidia-smi once per second: memory, temperature, power. Above the temperature limit the arm is stopped (TOO_HOT).
  LOCK     one queue at a time (flock on a lock file).

Examples:
  python3 scripts/night_probe_queue.py --list
  python3 scripts/night_probe_queue.py --dry-run --arms A0
  python3 scripts/night_probe_queue.py --arms A0,A1
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import fcntl
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "artifacts" / "experiments" / "NIGHT_INTENSE_1_prereg.json"
PROBE = REPO / "scripts" / "t4_model_probe.py"
GOVERNOR = REPO / "scripts" / "t4_governor.py"
ESTIMATOR = REPO / "scripts" / "qlora_vram_estimate.py"
DATA = Path("/mnt/disks/disk-socrateai-local-1")
DEFAULT_HOLD = DATA / "hivemind_telemetry" / "HOLD"  # the governor's --hold-file default
DEFAULT_OUT_ROOT = DATA / "night_probe"
DEFAULT_LOCK = DEFAULT_OUT_ROOT / "queue.lock"
STATUS_URL = "http://127.0.0.1:8091/nginx_status"  # the governor's --status-url default
OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
TZ = "Europe/Paris"
WINDOW_START, WINDOW_END = dt.time(23, 0), dt.time(6, 30)
UTIL_BUSY_PCT = 10  # t4_governor.other_gpu_busy default threshold: whole-GPU utilisation >= 10 % with no task of ours means someone else is using the card
SAFETY_MARGIN_MIB = 600
TIER_CAP_MIB = {"N0": 3950, "N1": 9650}  # N0: governor's 4300 MiB rule minus ~350 MiB context (probe default); N1: 10000 MiB budget minus context (plan section 6, step 3)
TIER_BUDGET_MIB = {"N0": 4300, "N1": 10000}  # prereg R1: process peak budgets by tier
BOINC_WAIT_S = 90.0  # after our HOLD exists the governor only sets BOINC to "never"; an already-running task can take 0.8-60 s to leave (t4_governor.terminate_tree docstring)
FINISHED = {"OK", "OOM", "CHECKPOINTING_UNSUPPORTED"}  # conclusive results: skipped on resume (everything else is retried)
STATUSES = ("OK", "OOM", "ABORTED_BY_TRAFFIC", "BLOCKED", "TIMEOUT", "TOO_HOT", "ERROR", "CHECKPOINTING_UNSUPPORTED", "INTERRUPTED", "OUTSIDE_WINDOW")
DEFAULT_QUEUE_STATUS_STOP = {"ABORTED_BY_TRAFFIC", "TOO_HOT", "INTERRUPTED", "OUTSIDE_WINDOW", "BLOCKED"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_GOV: list = []


def governor():
    """scripts/t4_governor.py imported by path (its functions are reused, not re-defined). Cached."""
    if not _GOV:
        _GOV.append(load_module(GOVERNOR, "t4_governor_for_queue"))
    return _GOV[0]


def real_pmon() -> str:
    """Same call as t4_governor.other_gpu_busy: per-process SM utilisation, one sample."""
    return subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=8).stdout


# --- window ------------------------------------------------------------------------------------------------------------------------------
def in_window(now: dt.datetime, tz: str = TZ) -> bool:
    """True from 23:00 (inclusive) to 06:30 (exclusive) on the Paris wall clock. ``now`` must be timezone-aware; DST is handled by zoneinfo."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    t = now.astimezone(ZoneInfo(tz)).time()
    return t >= WINDOW_START or t < WINDOW_END


# --- arms --------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    arm_id: str
    name: str
    model_id: str
    rank: int
    alpha: int
    seq_len: int
    micro_batch: int
    checkpointing: bool
    targets: tuple[str, ...]
    tier: str

    @property
    def tier_cap_mib(self) -> int:
        return TIER_CAP_MIB[self.tier]

    def recipe(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha, "seq_len": self.seq_len, "micro_batch": self.micro_batch,
                "checkpointing": self.checkpointing, "targets": list(self.targets)}


QKVO = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_LINEAR = QKVO + ("gate_proj", "up_proj", "down_proj")


def _one(rx: str, text: str, what: str, name: str) -> int:
    m = re.search(rx, text)
    if not m:
        raise ValueError(f"prereg arm {name}: cannot read {what} from recipe {text!r}")
    return int(m.group(1))


def load_arms(prereg: Path = PREREG) -> list[Arm]:
    """The arms exactly as pre-registered (``design.arms`` / ``design.order``); the recipe prose is parsed and a parse failure is an error, never a guess."""
    d = json.loads(prereg.read_text(encoding="utf-8"))["design"]
    out = []
    for key in d["order"]:
        a = d["arms"][key]
        r = a["recipe"]
        if "no checkpointing" in r:
            ck = False
        elif "gradient checkpointing on" in r:
            ck = True
        else:
            raise ValueError(f"prereg arm {key}: cannot read checkpointing from recipe {r!r}")
        targets = ALL_LINEAR if "q/k/v/o/gate/up/down" in r else QKVO if "q/k/v/o" in r else None
        if targets is None:
            raise ValueError(f"prereg arm {key}: cannot read target modules from recipe {r!r}")
        tier = a["tier"][:2]
        if tier not in TIER_CAP_MIB:
            raise ValueError(f"prereg arm {key}: unknown tier {a['tier']!r}")
        out.append(Arm(key.split("_")[0], key, a["base"], _one(r"rank (\d+)", r, "rank", key), _one(r"alpha (\d+)", r, "alpha", key), _one(r"seq (\d+)", r, "seq", key),
                       _one(r"micro-batch (\d+)", r, "micro-batch", key), ck, targets, tier))
    return out


# --- external effects (all injectable) ----------------------------------------------------------------------------------------------------
def http_get_text(url: str, timeout: float = 2.0) -> str:
    """GET only. The only URLs this script ever requests are the nginx stub_status and Ollama /api/ps on 127.0.0.1."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback URLs from constants
        return r.read().decode()


def gpu_snapshot() -> dict:
    """Read-only nvidia-smi queries (same fields as t4_governor.gpu_state, plus power): {used_mib,total_mib,temp_c,util_pct,power_w,procs:[(pid,name,mib)]}."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0].split(",")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    procs = []
    for ln in apps.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 3 and p[0].isdigit():
            procs.append((int(p[0]), p[1], int(float(p[2])) if p[2].replace(".", "", 1).isdigit() else 0))
    try:
        power = float(q[4])
    except (ValueError, IndexError):
        power = 0.0
    return {"used_mib": int(float(q[0])), "total_mib": int(float(q[1])), "temp_c": int(float(q[2])), "util_pct": int(float(q[3])), "power_w": power, "procs": procs}


def parse_stub_status(txt: str) -> dict[str, int]:
    """Mirror of t4_governor.stub_status's parser (kept identical on purpose; tests compare the two). The governor reads the URL itself, here the text is injected."""
    out = {"active": 0, "writing": 0, "reading": 0, "waiting": 0, "requests": 0}
    for ln in txt.splitlines():
        if ln.startswith("Active connections:"):
            out["active"] = int(ln.split(":")[1])
        elif ln.startswith("Reading:"):
            p = ln.replace(":", " ").split()
            out.update(reading=int(p[1]), writing=int(p[3]), waiting=int(p[5]))
        elif ln.strip() and ln.split()[0].isdigit() and len(ln.split()) == 3:
            out["requests"] = int(ln.split()[2])
    return out


def in_flight(st: dict[str, int]) -> int:
    """The governor's definition: writing connections minus this poll request itself."""
    return max(st["writing"] - 1, 0)


def real_killpg(pgid: int, sig: int) -> None:
    os.killpg(pgid, sig)


def real_spawn(cmd: list[str], stdout, stderr, env: dict[str, str]):
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, env=env, start_new_session=True)  # own process group: the whole tree can be killed


@dataclass
class Env:
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
    mono: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    http_get: Callable[[str], str] = http_get_text
    gpu: Callable[[], dict] = gpu_snapshot
    spawn: Callable[..., Any] = real_spawn
    killpg: Callable[[int, int], None] = real_killpg
    descendants: Callable[[int], list[int]] = field(default_factory=lambda: governor().descendants)
    pmon: Callable[[], str] = real_pmon
    parse_pmon: Callable[[str], dict] = field(default_factory=lambda: governor().parse_pmon)
    boinc_pids: Callable[[list], list] = field(default_factory=lambda: governor().boinc_gpu_pids)  # BOINC QLoRA tasks (python running real_client.moe_app) among the GPU processes
    estimate: Callable[[Arm], dict] | None = None
    environ: dict[str, str] = field(default_factory=lambda: dict(os.environ))


# --- estimator ---------------------------------------------------------------------------------------------------------------------------
class Estimator:
    """scripts/qlora_vram_estimate.py imported by path. If it cannot be imported or the model config is not in the local cache, the arm says so instead of inventing a number."""

    def __init__(self, hub: str | None = None) -> None:
        self.hub, self._mod, self._specs, self._scale, self._err = hub, None, None, None, None

    def _load(self) -> None:
        if self._mod is not None or self._err is not None:
            return
        try:
            self._mod = load_module(ESTIMATOR, "qlora_vram_estimate_for_queue")
            specs = self._mod.load_presets(self.hub) if self.hub else self._mod.load_presets()
            self._specs, self._scale = specs, self._mod.calibrate(specs)["act_scale"]
        except Exception as e:  # noqa: BLE001
            self._err = f"{type(e).__name__}: {str(e)[:200]}"

    def __call__(self, arm: Arm) -> dict:
        self._load()
        if self._err:
            return {"available": False, "reason": f"estimator unavailable: {self._err}"}
        m = self._specs.get(arm.model_id.split("/")[-1])
        if m is None:
            return {"available": False, "reason": f"no local config.json for {arm.model_id} in the Hugging Face cache, so the estimator has no geometry"}
        try:
            r = self._mod.Recipe(rank=arm.rank, targets=arm.targets, seq_len=arm.seq_len, micro_batch=arm.micro_batch, checkpointing=arm.checkpointing)
            b = self._mod.breakdown(m, r, self._scale)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        return {"available": True, "alloc_peak_mib": round(b["alloc_peak"]), "process_peak_mib": round(b["process_peak"]),
                "process_peak_with_20pct_margin_mib": round(b["process_peak"] * 1.2), "note": "estimate with calibration error up to ~20 %; Not tested for this recipe"}


# --- HOLD, lock --------------------------------------------------------------------------------------------------------------------------
class Hold:
    """The governor's HOLD file for the length of the run. A HOLD that already existed belongs to the owner: never created again, never removed."""

    def __init__(self, path: Path) -> None:
        self.path, self.preexisting, self.created, self.removed = path, False, False, False
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            if self.path.exists():
                self.preexisting = True
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                self.preexisting = True
                return
            with os.fdopen(fd, "w") as f:
                f.write(f"night_probe_queue pid {os.getpid()}\n")
            self.created = True

    def release(self) -> None:
        with self._lock:
            if self.created and not self.removed and not self.preexisting:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                self.removed = True


class LockBusy(RuntimeError):
    pass


class QueueLock:
    def __init__(self, path: Path) -> None:
        self.path, self.fd = path, None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            raise LockBusy(f"another night_probe_queue holds {self.path}") from e
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self.fd = fd

    def release(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    @staticmethod
    def is_busy(path: Path) -> bool:
        """Read-only check for --dry-run: never creates the file."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError:
            return True
        finally:
            os.close(fd)


# --- preconditions -----------------------------------------------------------------------------------------------------------------------
def allowed_name(name: str, allow: list[str]) -> bool:
    base = os.path.basename(name)
    return any(base == a or base.startswith(a) for a in allow)


def resident_models(env: Env, url: str) -> tuple[list[str] | None, str | None]:
    """Models Ollama has in memory (GET /api/ps). A refused connection means Ollama is not running: nothing resident. Any other failure cannot be verified: (None, why)."""
    try:
        data = json.loads(env.http_get(url))
        return [str(m.get("name") or m.get("model") or "?") for m in data.get("models") or []], None
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError):
            return [], None
        return None, f"{type(e).__name__}: {e.reason}"
    except ConnectionRefusedError:
        return [], None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def preflight(env: Env, o: "Options", arm: Arm) -> dict:
    """Everything that must hold before an arm starts. Returns {ok, status, reason, cap_mib, cap_basis, snapshot, checks}. Read-only."""
    checks: dict[str, Any] = {}
    snap = env.gpu()
    free = snap["total_mib"] - snap["used_mib"]
    cap = min(o.max_cap_mib, free - SAFETY_MARGIN_MIB)
    if o.tier_cap:
        cap = min(cap, arm.tier_cap_mib)
    basis = {"max_cap_mib": o.max_cap_mib, "free_vram_mib": free, "safety_margin_mib": SAFETY_MARGIN_MIB, "tier_cap_applied": o.tier_cap, "tier_cap_mib": arm.tier_cap_mib}
    out = {"ok": False, "status": "BLOCKED", "reason": "", "cap_mib": cap, "cap_basis": basis, "snapshot": snap, "checks": checks}
    models, why = resident_models(env, o.ollama_url)
    checks["ollama_resident_models"] = models if models is not None else f"unreadable: {why}"
    if models is None:
        out["reason"] = f"cannot verify that no Ollama model is resident ({why}); failing closed"
        return out
    if models:
        out["reason"] = f"Ollama model(s) resident: {', '.join(models)}; this queue never unloads anything"
        return out
    foreign = [p for p in snap["procs"] if not allowed_name(p[1], o.allow_gpu)]
    checks["foreign_gpu_processes"] = [list(p) for p in foreign]
    if foreign:
        boinc = set(map(str, env.boinc_pids(foreign)))
        if boinc:
            out["reason"] = f"a BOINC QLoRA task is still on the GPU (pid {', '.join(sorted(boinc))}); it was not killed, the governor stops it once HOLD is seen"
            return out
        out["reason"] = "GPU compute process(es) not on the allow-list: " + ", ".join(f"{p[1]} (pid {p[0]}, {p[2]} MiB)" for p in foreign)
        return out
    checks["gpu_util_pct"] = snap["util_pct"]
    if snap["util_pct"] >= UTIL_BUSY_PCT:
        out["reason"] = f"GPU utilisation {snap['util_pct']} % >= {UTIL_BUSY_PCT} % with no foreign process listed: someone is using the card (governor's busy threshold)"
        return out
    try:
        n = in_flight(parse_stub_status(env.http_get(o.status_url)))
    except Exception as e:  # noqa: BLE001
        checks["nginx_in_flight"] = f"unreadable: {type(e).__name__}"
        out["reason"] = f"cannot read nginx stub_status at {o.status_url} ({type(e).__name__}); failing closed"
        return out
    checks["nginx_in_flight"] = n
    if n >= 1:
        out["reason"] = f"{n} request(s) in flight on nginx"
        return out
    if snap["temp_c"] > o.max_gpu_temp_c:
        out["status"], out["reason"] = "TOO_HOT", f"GPU {snap['temp_c']} C above limit {o.max_gpu_temp_c} C before start"
        return out
    if cap < o.min_cap_mib:
        out["reason"] = f"cap {cap} MiB below the minimum {o.min_cap_mib} MiB (free {free} MiB - margin {SAFETY_MARGIN_MIB} MiB)"
        return out
    out.update(ok=True, status="READY", reason="")
    return out


# --- monitor -----------------------------------------------------------------------------------------------------------------------------
def kill_group(env: Env, proc, grace_s: float = 2.0) -> None:
    """SIGTERM to the probe's whole process group, SIGKILL after ``grace_s`` (2 s) if any member is still alive, then reap."""
    pgid = proc.pid
    try:
        env.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    end, gone = time.monotonic() + grace_s, False
    while time.monotonic() < end:
        if proc.poll() is not None and _group_gone(env, pgid):
            gone = True
            break
        time.sleep(0.02)
    if not gone:
        try:
            env.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def _group_gone(env: Env, pgid: int) -> bool:
    try:
        env.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


class Monitor(threading.Thread):
    """Once per ``poll_s``: sample the GPU, read the in-flight signal and Ollama, apply the stop rules. ``tick()`` is public so tests can step it."""

    def __init__(self, env: Env, o: "Options", proc, baseline_pids: set[int], deadline_mono: float, interrupted: threading.Event | None = None) -> None:
        super().__init__(daemon=True)
        self.env, self.o, self.proc, self.baseline, self.deadline = env, o, proc, baseline_pids, deadline_mono
        self.interrupted = interrupted or threading.Event()
        self.reason: str | None = None
        self.status: str | None = None
        self.stop_flag = threading.Event()
        self.max_temp_c: int | None = None
        self.max_power_w: float | None = None
        self.peak_used_mib: int | None = None
        self.peak_probe_mib: int | None = None
        self.samples = 0
        self.signal_at: float | None = None
        self.dead_at: float | None = None
        self.errors: list[str] = []

    def _fire(self, status: str, reason: str) -> None:
        self.status, self.reason = status, reason
        self.signal_at = time.monotonic()
        kill_group(self.env, self.proc, self.o.kill_grace_s)
        self.dead_at = time.monotonic()
        self.stop_flag.set()

    def tick(self) -> bool:
        """One poll. Returns True if the probe was stopped."""
        env, o = self.env, self.o
        if self.proc.poll() is not None:
            return False
        if self.interrupted.is_set():
            self._fire("INTERRUPTED", "stopped by SIGTERM/SIGINT")
            return True
        if not o.ignore_window and not in_window(env.now()):
            self._fire("OUTSIDE_WINDOW", "the 23:00-06:30 Europe/Paris window ended while the arm was running")
            return True
        if env.mono() >= self.deadline:
            self._fire("TIMEOUT", f"arm exceeded {o.arm_timeout_s:.0f} s")
            return True
        ours = {self.proc.pid, *self.env.descendants(self.proc.pid)}
        try:
            snap = env.gpu()
        except Exception as e:  # noqa: BLE001 - a failed sample is recorded, the other signals still run
            snap = None
            self.errors.append(f"nvidia-smi: {type(e).__name__}")
        if snap is not None:
            self.samples += 1
            self.max_temp_c = snap["temp_c"] if self.max_temp_c is None else max(self.max_temp_c, snap["temp_c"])
            self.max_power_w = snap["power_w"] if self.max_power_w is None else max(self.max_power_w, snap["power_w"])
            self.peak_used_mib = snap["used_mib"] if self.peak_used_mib is None else max(self.peak_used_mib, snap["used_mib"])
            if snap["temp_c"] > o.max_gpu_temp_c:
                self._fire("TOO_HOT", f"GPU {snap['temp_c']} C above limit {o.max_gpu_temp_c} C")
                return True
            mine_mib = sum(p[2] for p in snap["procs"] if p[0] in ours)
            self.peak_probe_mib = mine_mib if self.peak_probe_mib is None else max(self.peak_probe_mib, mine_mib)
            new = [p for p in snap["procs"] if p[0] not in ours and p[0] not in self.baseline and not allowed_name(p[1], o.allow_gpu)]
            if new:
                self._fire("ABORTED_BY_TRAFFIC", "new GPU process: " + ", ".join(f"{p[1]} (pid {p[0]})" for p in new))
                return True
        try:
            n = in_flight(parse_stub_status(env.http_get(o.status_url)))
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"stub_status: {type(e).__name__}")
            n = 0
        if n >= 1:
            self._fire("ABORTED_BY_TRAFFIC", f"{n} request(s) in flight on nginx")
            return True
        models, _ = resident_models(env, o.ollama_url)
        if models:
            self._fire("ABORTED_BY_TRAFFIC", "Ollama loaded a model: " + ", ".join(models))
            return True
        # the governor's running-task branch of other_gpu_busy: SM% of every process that is not ours. This is what sees a direct call to the (allow-listed) bee server
        try:
            pm = env.parse_pmon(env.pmon())
            top_pid, top = max(((pid, sm) for pid, sm in pm.items() if int(pid) not in ours), key=lambda x: x[1], default=("", 0))
        except Exception as e:  # noqa: BLE001 - like the governor, an unreadable pmon is "not busy"
            self.errors.append(f"pmon: {type(e).__name__}")
            top_pid, top = "", 0
        if top >= max(UTIL_BUSY_PCT // 2, 5):
            self._fire("ABORTED_BY_TRAFFIC", f"GPU activity by another process (pid {top_pid}, SM {top} %)")
            return True
        return False

    def run(self) -> None:
        while not self.stop_flag.is_set() and self.proc.poll() is None:
            if self.tick():
                return
            self.env.sleep(self.o.poll_s)


# --- options / queue ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Options:
    arms: list[str] | None = None
    out_dir: Path | None = None
    hold_file: Path = DEFAULT_HOLD
    lock_file: Path = DEFAULT_LOCK
    status_url: str = STATUS_URL
    ollama_url: str = OLLAMA_PS_URL
    allow_gpu: list[str] = field(default_factory=lambda: ["llama-server"])
    max_cap_mib: int = 9500
    min_cap_mib: int = 1500
    max_gpu_temp_c: int = 83
    arm_timeout_s: float = 1800.0
    poll_s: float = 1.0
    kill_grace_s: float = 2.0
    ignore_window: bool = False
    tier_cap: bool = False
    boinc_wait_s: float = BOINC_WAIT_S
    python: str = sys.executable
    probe: Path = PROBE
    prereg: Path = PREREG
    hub: str | None = None


def probe_cmd(o: Options, arm: Arm, cap: int) -> list[str]:
    cmd = [o.python, str(o.probe), arm.model_id, "--cap-mib", str(cap), "--rank", str(arm.rank), "--alpha", str(arm.alpha), "--seq-len", str(arm.seq_len),
           "--micro-batch", str(arm.micro_batch), "--max-gpu-temp-c", str(o.max_gpu_temp_c)]
    if arm.checkpointing:
        cmd.append("--checkpointing")
    if arm.targets != QKVO:
        cmd += ["--targets", ",".join(arm.targets)]
    return cmd


def select_arms(arms: list[Arm], wanted: list[str] | None) -> list[Arm]:
    if not wanted:
        return list(arms)
    ids = {a.arm_id for a in arms}
    bad = [w for w in wanted if w not in ids]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {sorted(ids)}")
    return [a for a in arms if a.arm_id in wanted]


def utc_iso(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_probe_json(path: Path) -> dict | None:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for ln in reversed(lines):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                v = json.loads(ln)
                return v if isinstance(v, dict) else None
            except ValueError:
                continue
    return None


def arm_file(out_dir: Path, arm: Arm) -> Path:
    return out_dir / f"arm_{arm.arm_id}.json"


def already_finished(out_dir: Path, arm: Arm) -> bool:
    try:
        return json.loads(arm_file(out_dir, arm).read_text()).get("status") in FINISHED
    except (OSError, ValueError):
        return False


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, path)


def compare(est: dict, probe: dict | None) -> dict:
    if not est.get("available") or not probe or probe.get("torch_peak_allocated_mib") is None:
        return {"available": False}
    pa = probe["torch_peak_allocated_mib"]
    return {"available": True, "estimated_alloc_peak_mib": est["alloc_peak_mib"], "measured_alloc_peak_mib": pa, "measured_minus_estimated_mib": pa - est["alloc_peak_mib"],
            "measured_over_estimated": round(pa / est["alloc_peak_mib"], 3) if est["alloc_peak_mib"] else None}


def wait_boinc_clear(env: Env, o: "Options") -> float:
    """Our HOLD is in place; the governor then sets BOINC to "never" but does not end a task that is already running (it can take up to ~60 s to leave). Wait up to
    ``o.boinc_wait_s`` for BOINC GPU tasks to go. Never kills anything: if one is still there the arm is BLOCKED by the preflight, naming BOINC."""
    start = env.mono()
    while True:
        pids = env.boinc_pids(env.gpu()["procs"])
        waited = env.mono() - start
        if not pids or waited >= o.boinc_wait_s:
            return round(waited, 1)
        env.sleep(o.poll_s)


class Queue:
    def __init__(self, env: Env, o: Options, estimator: Callable[[Arm], dict]) -> None:
        self.env, self.o, self.estimator = env, o, estimator
        self.interrupted = threading.Event()
        self.current: Monitor | None = None
        self.current_proc = None

    def request_stop(self) -> None:
        """Signal-handler safe: sets the flag and sends SIGTERM (never waits: the handler runs inside Popen.wait on the main thread). The monitor escalates to SIGKILL."""
        self.interrupted.set()
        proc = self.current_proc
        if proc is not None:
            try:
                self.env.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def run_arm(self, arm: Arm, pre: dict, hold_preexisting: bool) -> dict:
        env, o = self.env, self.o
        assert o.out_dir is not None
        est = self.estimator(arm)
        rec: dict[str, Any] = {"schema": "night_probe_arm/1", "arm": arm.arm_id, "name": arm.name, "model_id": arm.model_id, "recipe": arm.recipe(), "tier": arm.tier,
                               "cap_mib": pre["cap_mib"], "cap_basis": pre["cap_basis"], "tier_budget_mib": TIER_BUDGET_MIB[arm.tier], "window_ignored": o.ignore_window, "hold_preexisting": hold_preexisting,
                               "estimate": est, "started_utc": utc_iso(env.now()), "preflight": pre["checks"]}
        if not pre["ok"]:
            rec.update(fits_tier_budget=None, status=pre["status"], reason=pre["reason"], ended_utc=utc_iso(env.now()), probe=None, measured=None, estimate_vs_measured={"available": False})
            return rec
        cmd = probe_cmd(o, arm, pre["cap_mib"])
        rec["cmd"] = cmd
        out_p, err_p = o.out_dir / f"arm_{arm.arm_id}.probe.stdout.log", o.out_dir / f"arm_{arm.arm_id}.probe.stderr.log"
        child_env = dict(env.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
        t0 = env.mono()
        with open(out_p, "wb") as so, open(err_p, "wb") as se:
            proc = env.spawn(cmd, so, se, child_env)
            self.current_proc = proc
            baseline = {p[0] for p in pre["snapshot"]["procs"]}
            mon = Monitor(env, o, proc, baseline, t0 + o.arm_timeout_s, self.interrupted)
            self.current = mon
            mon.start()
            if self.interrupted.is_set():  # a signal arrived between the preflight and the spawn
                kill_group(env, proc, o.kill_grace_s)
            rc = proc.wait()
            mon.stop_flag.set()
            mon.join(timeout=10)
        self.current = self.current_proc = None
        probe = parse_probe_json(out_p)
        status, reason = None, ""
        if self.interrupted.is_set():
            status, reason = "INTERRUPTED", "stopped by SIGTERM/SIGINT"
        elif mon.status:
            status, reason = mon.status, mon.reason or ""
        elif rc == 0 and probe and probe.get("status") == "OK":
            status = "OK"
        else:
            status = (probe or {}).get("status") if (probe or {}).get("status") in STATUSES else None
            if status is None:
                tail = err_p.read_text(errors="replace")[-400:] if err_p.exists() else ""
                status = "OOM" if "out of memory" in tail.lower() else "ERROR"
            reason = (probe or {}).get("error") or f"probe exited {rc}"
        measured = None
        if probe:
            keys = ("torch_peak_allocated_mib", "torch_peak_reserved_mib", "after_load_mib", "load_seconds", "seconds_per_step", "tokens_per_second", "trainable_params", "total_params_b")
            measured = {k: probe.get(k) for k in keys}
        measured = dict(measured or {}, nvidia_smi_peak_used_mib=mon.peak_used_mib, nvidia_smi_probe_peak_used_mib=mon.peak_probe_mib, max_temp_c=mon.max_temp_c,
                        max_power_w=mon.max_power_w, samples=mon.samples)
        budget = TIER_BUDGET_MIB[arm.tier]
        fits: bool | None = None  # does the arm fit its tier budget? nvidia-smi on the probe's own processes; None = not determined
        if status == "OK" and mon.peak_probe_mib is not None:
            fits = mon.peak_probe_mib <= budget
        elif status == "OOM" and pre["cap_mib"] >= budget:
            fits = False  # ran out of memory although the cap allowed the whole tier budget
        rec["fits_tier_budget"] = fits
        rec.update(status=status, reason=reason, returncode=rc, ended_utc=utc_iso(env.now()), duration_s=round(env.mono() - t0, 1), probe=probe, measured=measured,
                   estimate_vs_measured=compare(est, probe), monitor_errors=mon.errors[:5],
                   stop_latency_s=round(mon.dead_at - mon.signal_at, 2) if mon.dead_at and mon.signal_at else None)
        return rec

    def run(self, arms: list[Arm], hold_preexisting: bool) -> list[dict]:
        env, o = self.env, self.o
        assert o.out_dir is not None
        results: list[dict] = []
        for arm in arms:
            if self.interrupted.is_set():
                break
            if already_finished(o.out_dir, arm):
                results.append({"arm": arm.arm_id, "status": json.loads(arm_file(o.out_dir, arm).read_text())["status"], "skipped_resume": True})
                continue
            if not o.ignore_window and not in_window(env.now()):
                rec = {"schema": "night_probe_arm/1", "arm": arm.arm_id, "status": "OUTSIDE_WINDOW", "reason": "outside 23:00-06:30 Europe/Paris before this arm", "started_utc": utc_iso(env.now())}
                write_json(arm_file(o.out_dir, arm), rec)
                results.append(rec)
                break
            try:
                waited = wait_boinc_clear(env, o)
                pre = preflight(env, o, arm)
                pre["checks"]["boinc_wait_s"] = waited
            except Exception as e:  # noqa: BLE001 - nvidia-smi unreadable etc.: fail closed
                pre = {"ok": False, "status": "ERROR", "reason": f"preflight failed: {type(e).__name__}: {str(e)[:160]}", "cap_mib": None, "cap_basis": {}, "snapshot": {"procs": []}, "checks": {}}
            rec = self.run_arm(arm, pre, hold_preexisting)
            write_json(arm_file(o.out_dir, arm), rec)
            results.append(rec)
            if rec["status"] in DEFAULT_QUEUE_STATUS_STOP:
                break
        return results


def run_queue(env: Env, o: Options, all_arms: list[Arm], estimator: Callable[[Arm], dict], install_signals: bool = True) -> int:
    """The real run. Exit codes: 0 all selected arms finished OK/OOM/CHECKPOINTING_UNSUPPORTED, 1 some other status, 2 outside the window, 4 lock busy, 130 interrupted."""
    arms = select_arms(all_arms, o.arms)
    lock = QueueLock(o.lock_file)
    try:
        lock.acquire()
    except LockBusy as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 4
    if not o.ignore_window and not in_window(env.now()):
        print(f"REFUSED: now is outside the 23:00-06:30 {TZ} window (use --ignore-window only for a supervised test)", file=sys.stderr)
        lock.release()
        return 2
    hold = Hold(o.hold_file)
    q = Queue(env, o, estimator)
    started = env.now()
    results: list[dict] = []
    old: dict[int, Any] = {}
    atexit.register(hold.release)

    def handler(signum, _frame) -> None:
        q.request_stop()

    try:
        if install_signals and threading.current_thread() is threading.main_thread():
            for s in (signal.SIGTERM, signal.SIGINT):
                old[s] = signal.signal(s, handler)
        hold.acquire()
        o.out_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        results = q.run(arms, hold.preexisting)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        hold.release()
        try:
            atexit.unregister(hold.release)
        except Exception:  # noqa: BLE001
            pass
        lock.release()
        if o.out_dir is not None and o.out_dir.exists():
            summary = {"schema": "night_probe_summary/1", "started_utc": utc_iso(started), "ended_utc": utc_iso(env.now()), "ignore_window": o.ignore_window,
                       "window": f"23:00-06:30 {TZ}", "hold_file": str(o.hold_file), "hold_preexisting": hold.preexisting, "hold_created_by_queue": hold.created,
                       "hold_removed_by_queue": hold.removed, "interrupted": q.interrupted.is_set(), "max_cap_mib": o.max_cap_mib, "allow_gpu_names": o.allow_gpu,
                       "arms_requested": [a.arm_id for a in arms], "arms": [{k: r.get(k) for k in ("arm", "status", "reason", "cap_mib", "skipped_resume")} | {
                           "measured_torch_peak_allocated_mib": (r.get("measured") or {}).get("torch_peak_allocated_mib"),
                           "estimated_alloc_peak_mib": (r.get("estimate") or {}).get("alloc_peak_mib"),
                           "max_temp_c": (r.get("measured") or {}).get("max_temp_c"), "max_power_w": (r.get("measured") or {}).get("max_power_w")} for r in results]}
            write_json(o.out_dir / "summary.json", summary)
    if q.interrupted.is_set():
        return 130
    bad = [r for r in results if r["status"] not in FINISHED]
    return 1 if bad else 0


# --- dry run / list ----------------------------------------------------------------------------------------------------------------------
def print_arm_table(arms: list[Arm], estimator: Callable[[Arm], dict], o: Options) -> None:
    print(f"{'arm':4} {'model':42} {'rank':>4} {'alpha':>5} {'seq':>4} {'mb':>3} {'ckpt':5} {'tier':4} {'tier_cap':>8}  estimate (alloc / process, MiB)")
    for a in arms:
        e = estimator(a)
        es = f"{e['alloc_peak_mib']} / {e['process_peak_mib']}" if e.get("available") else f"n/a ({e.get('reason')})"
        print(f"{a.arm_id:4} {a.model_id:42} {a.rank:>4} {a.alpha:>5} {a.seq_len:>4} {a.micro_batch:>3} {str(a.checkpointing):5} {a.tier:4} {a.tier_cap_mib:>8}  {es}")


def dry_run(env: Env, o: Options, arms: list[Arm], estimator: Callable[[Arm], dict]) -> int:
    """Prints the plan and every precondition. Only read-only GETs and nvidia-smi queries; creates no HOLD, lock, directory or file."""
    now = env.now()
    ok_all = True
    print(f"DRY RUN (nothing is written, no probe is started, no HOLD is created)\nnow: {now.astimezone(ZoneInfo(TZ)).isoformat()}")
    w = in_window(now)
    print(f"window 23:00-06:30 {TZ}: {'inside' if w else 'OUTSIDE'}" + (" (ignored by --ignore-window)" if o.ignore_window and not w else ""))
    ok_all &= w or o.ignore_window
    print(f"HOLD file {o.hold_file}: {'EXISTS (owner switch; will not be touched, recorded as hold_preexisting)' if o.hold_file.exists() else 'absent (the real run would create it and remove it at the end)'}")
    busy = QueueLock.is_busy(o.lock_file)
    print(f"lock file {o.lock_file}: {'HELD by another queue' if busy else 'free'}")
    ok_all &= not busy
    print(f"out dir: {o.out_dir} ({'exists' if o.out_dir and o.out_dir.exists() else 'would be created'})")
    print()
    print_arm_table(arms, estimator, o)
    print()
    for a in arms:
        if o.out_dir and already_finished(o.out_dir, a):
            print(f"[{a.arm_id}] would be SKIPPED (finished result already in out dir)")
            continue
        try:
            pre = preflight(env, o, a)
        except Exception as e:  # noqa: BLE001
            print(f"[{a.arm_id}] preflight could not run: {type(e).__name__}: {str(e)[:160]} -> would be BLOCKED")
            ok_all = False
            continue
        print(f"[{a.arm_id}] cap {pre['cap_mib']} MiB = min(max-cap {o.max_cap_mib}, free {pre['cap_basis'].get('free_vram_mib')} - {SAFETY_MARGIN_MIB}" + (f", tier {a.tier_cap_mib}" if o.tier_cap else "") + ")")
        for k, v in pre["checks"].items():
            print(f"      check {k}: {v}")
        if pre["ok"]:
            print("      verdict: would RUN: " + " ".join(probe_cmd(o, a, pre["cap_mib"])))
        else:
            ok_all = False
            print(f"      verdict: would be {pre['status']}: {pre['reason']}")
    print("\nall preconditions satisfied" if ok_all else "\nsome precondition would stop the run")
    return 0 if ok_all else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", help="comma-separated arm ids, e.g. A0,A1 (default: all, in pre-registered order)")
    ap.add_argument("--list", action="store_true", help="print the arm table and exit (no GPU, no network)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and every precondition; writes nothing")
    ap.add_argument("--out-dir", type=Path, help="results directory (default: <data disk>/night_probe/<UTC timestamp>/); an existing one is resumed")
    ap.add_argument("--max-cap-mib", type=int, default=9500)
    ap.add_argument("--min-cap-mib", type=int, default=1500)
    ap.add_argument("--max-gpu-temp-c", type=int, default=83)
    ap.add_argument("--arm-timeout-s", type=float, default=1800.0)
    ap.add_argument("--allow-gpu", action="append", help="GPU process name allowed to hold memory (repeatable; default llama-server, the bee server)")
    ap.add_argument("--tier-cap", action="store_true", help="also cap each arm at its tier's allocator cap (N0 3950, N1 9650 MiB); default: cap = min(--max-cap-mib, free - 600)")
    ap.add_argument("--boinc-wait-s", type=float, default=BOINC_WAIT_S, help="seconds to wait after HOLD for a running BOINC GPU task to leave (it is never killed)")
    ap.add_argument("--ignore-window", action="store_true", help="run outside 23:00-06:30 (recorded in the results); for a supervised test only")
    ap.add_argument("--hold-file", type=Path, default=DEFAULT_HOLD)
    ap.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--status-url", default=STATUS_URL)
    ap.add_argument("--ollama-url", default=OLLAMA_PS_URL)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the probe (needs torch, transformers, peft, bitsandbytes)")
    ap.add_argument("--hub", help="Hugging Face hub cache for the estimator's config.json files")
    return ap


def options_from_args(a: argparse.Namespace, env: Env) -> Options:
    out = a.out_dir or DEFAULT_OUT_ROOT / env.now().astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Options(arms=[x.strip() for x in a.arms.split(",") if x.strip()] if a.arms else None, out_dir=out, hold_file=a.hold_file, lock_file=a.lock_file,
                   status_url=a.status_url, ollama_url=a.ollama_url, allow_gpu=a.allow_gpu or ["llama-server"], max_cap_mib=a.max_cap_mib, min_cap_mib=a.min_cap_mib,
                   max_gpu_temp_c=a.max_gpu_temp_c, arm_timeout_s=a.arm_timeout_s, ignore_window=a.ignore_window, tier_cap=a.tier_cap, boinc_wait_s=a.boinc_wait_s, python=a.python, hub=a.hub)


def main(argv: list[str] | None = None, env: Env | None = None) -> int:
    a = build_parser().parse_args(argv)
    env = env or Env()
    o = options_from_args(a, env)
    arms = select_arms(load_arms(o.prereg), o.arms)
    estimator = env.estimate or Estimator(o.hub)
    if a.list:
        print_arm_table(arms, estimator, o)
        return 0
    if a.dry_run:
        return dry_run(env, o, arms, estimator)
    return run_queue(env, o, arms, estimator)


if __name__ == "__main__":
    raise SystemExit(main())
