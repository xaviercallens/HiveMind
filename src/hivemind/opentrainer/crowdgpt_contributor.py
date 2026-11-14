# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CrowdGPT contributor mode for the HiveMind OpenTrainer agent.

A HiveMinder lends their GPU to CrowdGPT's shared dense model (Crowd-v1) by running
CrowdGPT's own signed ``client.py`` under a small supervisor that protects the machine:

* it pauses the client (``psutil`` suspend) when the GPU is hot or another GPU workload starts,
* it terminates the client when another program needs GPU memory (a suspended client keeps
  all its VRAM, so suspending is not enough), and relaunches it once the GPU has been idle,
* it writes one local JSONL receipt per round from what the client itself printed.

Nothing here trains, aggregates or uploads weights itself: the CrowdGPT client does the
training and sends its round to CrowdGPT's coordinator. No HMT is credited and no round is
submitted to the HiveMind gateway. The memory, temperature and utilization guards need
``nvidia-smi`` (NVIDIA only); ``--pause-for`` process names work on every OS.

All side effects (``subprocess.run``, the clock, ``sleep``, the process factory, ``psutil``)
are injected so the logic is testable without a GPU.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

CROWDGPT_REPO_URL = "https://github.com/Vxtzq/CrowdGPT"
UPLOAD_GB_PER_ROUND = 0.9
RECEIPTS_FILENAME = "crowdgpt_receipts.jsonl"
CLIENT_LOG_FILENAME = "crowdgpt_client.log"
SECRET_ENV_KEYS = ("CROWDGPT_TOKEN", "CROWDGPT_PASSWORD")
# Client exit codes that a relaunch cannot fix (see client.py _HEADLESS_EXIT_MESSAGES).
FATAL_CLIENT_EXIT_CODES = {
    2: "authentication failed or no saved session",
    3: "backend setup failed",
    4: "GUI unavailable (headless flag missing?)",
    5: "could not fetch task metadata from the coordinator",
    6: "the coordinator rejected this client's architecture",
    7: "the coordinator rejected this client",
}


def install_instructions() -> str:
    """Return the manual install steps printed when no CrowdGPT checkout is found (never auto-installed)."""
    return (
        "CrowdGPT client not found. Install it yourself, then pass --crowdgpt-dir:\n"
        f"  git clone {CROWDGPT_REPO_URL}\n"
        "  cd CrowdGPT\n"
        "  pip install -r requirements_cuda.txt   # NVIDIA; use requirements_rocm.txt / requirements_directml.txt / requirements.txt otherwise\n"
        "  python client.py --doctor --no-update   # check that client.py matches the signed release\n"
        "then: python scripts/hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir <path to CrowdGPT>"
    )


def startup_notice(*, nvidia: bool) -> str:
    """Return the disclosure printed when contributor mode starts."""
    guard = (
        "NVIDIA GPU detected: automatic pause/stop on memory pressure, heat and other GPU work is active "
        "(these guards are NVIDIA-only; --pause-for process names work on every OS)."
        if nvidia
        else "No nvidia-smi: automatic memory release, heat and utilization guards are UNAVAILABLE on this GPU "
        "(they are NVIDIA-only). Only --pause-for process names and --stop-at apply."
    )
    return "\n".join(
        [
            "CrowdGPT contributor mode (HiveMind OpenTrainer)",
            f"- Each finished round uploads about {UPLOAD_GB_PER_ROUND} GB to CrowdGPT's coordinator (CrowdGPT's server, not HiveMind's).",
            "- The client runs no server-supplied code; model weights are loaded as safetensors.",
            "- Anonymous by default (--anon) unless you opt into an account through environment variables.",
            "- A STOP (GPU memory needed by another program) terminates the client: the round in progress is lost.",
            f"- {guard}",
            "- No HMT is credited and nothing is submitted to the HiveMind gateway; receipts stay on this machine.",
        ]
    )


# --------------------------------------------------------------------------- log parsing

_STEP_RE = re.compile(r"\[train\] step (\d+)/(\d+) loss=(\S+) tps=(\d+) left=(\d+)m")
_DONE_RE = re.compile(r"Done: (\d+) micro-steps, loss (\S+) \S+ (\S+) \(mean of (\d+), last (\S+)\)")
_ROUND_RE = re.compile(r"(?:^|\] )Round (\d+)(?:/(\d+))?\s*$")
_ROUND_LOST_RE = re.compile(r"Round lost\s*\S*\s*(.*)$")


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class ClientEvent:
    """One meaningful line printed by the CrowdGPT client."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    line: str = ""


def parse_client_line(line: str) -> ClientEvent | None:
    """Parse one client log line into an event, or ``None`` when the line carries nothing we track.

    Kinds: ``step``, ``round_done``, ``submitted``, ``not_counted``, ``round_lost``,
    ``upload_failed``, ``round_start``, ``stopped_by_user``, ``error``.
    """
    text = line.rstrip("\r\n")
    if not text.strip():
        return None
    m = _STEP_RE.search(text)
    if m:
        return ClientEvent(
            "step",
            {"step": int(m.group(1)), "target": int(m.group(2)), "loss": _num(m.group(3)),
             "tps": int(m.group(4)), "left_min": int(m.group(5))},
            text,
        )
    m = _DONE_RE.search(text)
    if m:
        return ClientEvent(
            "round_done",
            {"steps": int(m.group(1)), "mean_loss": _num(m.group(2)), "std": _num(m.group(3)),
             "mean_of": int(m.group(4)), "last": _num(m.group(5))},
            text,
        )
    if "Submitted successfully" in text:
        return ClientEvent("submitted", {}, text)
    if "Not counted by server" in text:
        return ClientEvent("not_counted", {"detail": text.split("Not counted by server", 1)[1].strip(" ()")}, text)
    m = _ROUND_LOST_RE.search(text)
    if m:
        return ClientEvent("round_lost", {"detail": m.group(1).strip()}, text)
    if "Upload failed:" in text:
        return ClientEvent("upload_failed", {"detail": text.split("Upload failed:", 1)[1].strip()}, text)
    if "Stopped by user. Skipping upload." in text:
        return ClientEvent("stopped_by_user", {}, text)
    if "Traceback (most recent call last)" in text:
        return ClientEvent("error", {"detail": "traceback"}, text)
    if "Round failed:" in text:
        return ClientEvent("error", {"detail": text.split("Round failed:", 1)[1].strip()}, text)
    if "[ERROR]" in text or "[CRITICAL]" in text:
        return ClientEvent("error", {"detail": re.split(r"\[(?:ERROR|CRITICAL)\]", text, maxsplit=1)[1].strip()}, text)
    m = _ROUND_RE.search(text)
    if m:
        return ClientEvent("round_start", {"round": int(m.group(1)), "of": int(m.group(2)) if m.group(2) else None}, text)
    return None


# --------------------------------------------------------------------------- GPU telemetry


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    name: str
    used_mib: int | None


@dataclass(frozen=True)
class GpuSnapshot:
    total_mib: int
    used_mib: int
    free_mib: int
    util_pct: float | None
    temp_c: float | None
    name: str = ""
    apps: tuple[ComputeApp, ...] = ()


def _na(text: str) -> float | None:
    value = text.strip()
    if not value or value.strip("[]").upper() in {"N/A", "NOT SUPPORTED", "UNKNOWN ERROR"}:
        return None
    return _num(value)


def _default_run(*args: Any, **kwargs: Any) -> Any:
    return subprocess.run(*args, **kwargs)


def query_nvidia_gpu(run: Callable[..., Any] | None = None, gpu_index: int = 0) -> GpuSnapshot | None:
    """Read memory, utilization and temperature of one NVIDIA GPU; ``None`` when nvidia-smi is missing or fails."""
    runner = run or _default_run
    try:
        proc = runner(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0 or not (proc.stdout or "").strip():
        return None
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if gpu_index >= len(lines):
        return None
    parts = [p.strip() for p in lines[gpu_index].split(",")]
    if len(parts) < 6:
        return None
    total, used, free = _na(parts[1]), _na(parts[2]), _na(parts[3])
    if total is None or used is None or free is None or total <= 0:
        return None
    apps: list[ComputeApp] = []
    try:
        aproc = runner(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if getattr(aproc, "returncode", 1) == 0:
            for ln in (aproc.stdout or "").splitlines():
                cols = [c.strip() for c in ln.split(",")]
                if len(cols) >= 3 and cols[0].isdigit():
                    mem = _na(cols[2])
                    apps.append(ComputeApp(int(cols[0]), cols[1], int(mem) if mem is not None else None))
    except (OSError, subprocess.SubprocessError):
        apps = []
    return GpuSnapshot(
        total_mib=int(total), used_mib=int(used), free_mib=int(free),
        util_pct=_na(parts[4]), temp_c=_na(parts[5]), name=parts[0], apps=tuple(apps),
    )


def plan_vram_budget_gb(
    snapshot: GpuSnapshot | None, reserve_mib: int, context_mib: int = 700, min_gb: float = 1.0
) -> float | None:
    """Budget for the client = (free - reserve - CUDA context) / 1024, rounded down to 0.5 GB; ``None`` below ``min_gb``."""
    if snapshot is None:
        return None
    raw_gb = (snapshot.free_mib - reserve_mib - context_mib) / 1024.0
    budget = int(raw_gb * 2) / 2.0 if raw_gb > 0 else 0.0
    return budget if budget >= min_gb else None


# --------------------------------------------------------------------------- guard state machine


class Action(str, Enum):
    RUN = "RUN"
    PAUSE = "PAUSE"
    STOP = "STOP"
    WAIT = "WAIT"


class Phase(str, Enum):
    INIT = "INIT"          # nothing launched yet
    RUNNING = "RUNNING"    # client running
    PAUSED = "PAUSED"      # client suspended (still holds its VRAM)
    STOPPED = "STOPPED"    # client terminated, waiting to relaunch


@dataclass
class GuardConfig:
    reserve_mib: int = 1500
    context_mib: int = 700
    min_gb: float = 1.0
    breach_s: float = 10.0
    hot_c: float = 82.0
    cool_c: float = 74.0
    busy_util: float = 15.0
    quiet_s: float = 60.0
    relaunch_idle_s: float = 300.0
    short_run_s: float = 900.0
    max_backoff_factor: int = 8
    pause_for: tuple[str, ...] = ()


@dataclass
class GuardState:
    """Mutable guard state; ``decide`` reads the observations and updates the timers and the phase."""

    phase: Phase = Phase.INIT
    now: float = 0.0
    client_pids: frozenset[int] = frozenset()
    pause_for_running: tuple[str, ...] = ()
    breach_since: float | None = None
    quiet_since: float | None = None
    idle_since: float | None = None
    launched_at: float | None = None
    stopped_at: float | None = None
    quick_stops: int = 0
    reason: str = ""


def other_compute_apps(snapshot: GpuSnapshot | None, client_pids: Iterable[int]) -> list[ComputeApp]:
    if snapshot is None:
        return []
    mine = set(client_pids)
    return [a for a in snapshot.apps if a.pid not in mine]


def required_idle_s(state: GuardState, cfg: GuardConfig) -> float:
    """Idle time needed before a relaunch; doubles after each STOP that followed a short run (anti-flapping)."""
    factor = min(2 ** max(state.quick_stops - 1, 0), cfg.max_backoff_factor)
    return cfg.relaunch_idle_s * factor


def _enter(state: GuardState, phase: Phase, reason: str) -> None:
    state.phase = phase
    state.reason = reason
    state.breach_since = None
    state.quiet_since = None
    state.idle_since = None


def decide(state: GuardState, snapshot: GpuSnapshot | None, client_mem_mib: int, cfg: GuardConfig) -> Action:
    """One guard step. Updates ``state`` (phase, timers, reason) and returns what the supervisor must do.

    ======== =============================================================== ========== ==========
    phase    condition (checked in this order)                               action     next phase
    ======== =============================================================== ========== ==========
    RUNNING  free < reserve for >= breach_s                                  STOP       STOPPED
    PAUSED   free < reserve for >= breach_s                                  STOP       STOPPED
    RUNNING  temp >= hot_c, a --pause-for process, or another compute app    PAUSE      PAUSED
    RUNNING  otherwise                                                       RUN        RUNNING
    PAUSED   busy (util > busy_util, --pause-for, other app) or temp > cool  PAUSE      PAUSED
    PAUSED   quiet and temp <= cool_c for >= quiet_s                         RUN        RUNNING
    INIT     idle now (see below)                                            RUN        RUNNING
    STOPPED  idle for >= relaunch_idle_s (x backoff)                         RUN        RUNNING
    INIT/STP not idle / not yet long enough                                  WAIT       same
    ======== =============================================================== ========== ==========

    "idle" = no --pause-for process, no other compute app, util <= busy_util, temp <= cool_c,
    the previous client's memory released (``client_mem_mib == 0``) and a budget >= min_gb.
    Without a snapshot (no nvidia-smi) only the --pause-for rule applies.
    """
    now = state.now
    others = other_compute_apps(snapshot, state.client_pids)
    temp = snapshot.temp_c if snapshot is not None else None
    util = snapshot.util_pct if snapshot is not None else None
    hot = temp is not None and temp >= cfg.hot_c
    cool = temp is None or temp <= cfg.cool_c
    pause_names = list(state.pause_for_running)

    if state.phase in (Phase.RUNNING, Phase.PAUSED) and snapshot is not None:
        if snapshot.free_mib < cfg.reserve_mib:
            if state.breach_since is None:
                state.breach_since = now
            if now - state.breach_since >= cfg.breach_s:
                ran = now - state.launched_at if state.launched_at is not None else 0.0
                state.quick_stops = state.quick_stops + 1 if ran < cfg.short_run_s else 1
                _enter(state, Phase.STOPPED,
                       f"free {snapshot.free_mib} MiB < reserve {cfg.reserve_mib} MiB for {now - (state.breach_since or now):.0f}s"
                       f" (client holds {client_mem_mib} MiB)")
                state.stopped_at = now
                return Action.STOP
        else:
            state.breach_since = None

    if state.phase == Phase.RUNNING:
        if hot:
            _enter(state, Phase.PAUSED, f"GPU at {temp:.0f} C >= {cfg.hot_c:.0f} C")
            return Action.PAUSE
        if pause_names:
            _enter(state, Phase.PAUSED, f"--pause-for process running: {', '.join(pause_names)}")
            return Action.PAUSE
        if others:
            _enter(state, Phase.PAUSED, "other GPU compute apps: " + ", ".join(f"{a.name}({a.pid})" for a in others))
            return Action.PAUSE
        return Action.RUN

    if state.phase == Phase.PAUSED:
        busy = bool(pause_names) or bool(others) or (util is not None and util > cfg.busy_util)
        if busy or not cool:
            state.quiet_since = None
            state.reason = "paused: " + ("GPU busy" if busy else f"cooling ({temp:.0f} C > {cfg.cool_c:.0f} C)")
            return Action.PAUSE
        if state.quiet_since is None:
            state.quiet_since = now
        if now - state.quiet_since >= cfg.quiet_s:
            _enter(state, Phase.RUNNING, f"quiet for {cfg.quiet_s:.0f}s and cool; resuming")
            return Action.RUN
        state.reason = "paused: waiting for quiet period"
        return Action.PAUSE

    # INIT or STOPPED: decide whether to (re)launch.
    if snapshot is None:
        idle = not pause_names
        why = "a --pause-for process is running" if pause_names else ""
    else:
        budget = plan_vram_budget_gb(snapshot, cfg.reserve_mib, cfg.context_mib, cfg.min_gb)
        checks = [
            (not pause_names, "a --pause-for process is running"),
            (not others, "other GPU compute apps are running"),
            (util is None or util <= cfg.busy_util, f"GPU utilization {util}% > {cfg.busy_util}%"),
            (cool, f"GPU at {temp} C > {cfg.cool_c} C"),
            (client_mem_mib == 0, f"previous client still holds {client_mem_mib} MiB"),
            (budget is not None, f"budget below {cfg.min_gb} GB (free {snapshot.free_mib} MiB, reserve {cfg.reserve_mib} MiB)"),
        ]
        failed = [msg for ok, msg in checks if not ok]
        idle = not failed
        why = "; ".join(failed)
    if not idle:
        state.idle_since = None
        state.reason = "waiting: " + why
        return Action.WAIT
    if state.phase == Phase.INIT:
        _enter(state, Phase.RUNNING, "GPU idle; launching")
        state.launched_at = now
        return Action.RUN
    if state.idle_since is None:
        state.idle_since = now
    need = required_idle_s(state, cfg)
    if now - state.idle_since >= need:
        _enter(state, Phase.RUNNING, f"GPU idle for {need:.0f}s; relaunching")
        state.launched_at = now
        return Action.RUN
    state.reason = f"waiting: idle {now - state.idle_since:.0f}/{need:.0f}s before relaunch"
    return Action.WAIT


# --------------------------------------------------------------------------- preflight and command


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    reason: str
    local_version: str | None = None
    remote_version: str | None = None
    report: dict[str, Any] | None = None


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(text[start:])
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def preflight_client(client_dir: str | os.PathLike[str], python: str, run: Callable[..., Any] | None = None,
                     env: Mapping[str, str] | None = None, timeout_s: float = 300.0) -> PreflightResult:
    """Run ``client.py --doctor --json --no-update`` and refuse unless client.py matches the signed release.

    Requires ``update.signature`` True, ``update.client_matches`` True and local version == remote version.
    The doctor exits 1 when any unrelated check fails, so the JSON is parsed whatever the exit code.
    """
    runner = run or _default_run
    client_py = Path(client_dir) / "client.py"
    if not client_py.is_file():
        return PreflightResult(False, f"client.py not found in {client_dir}")
    try:
        proc = runner([python, str(client_py), "--doctor", "--json", "--no-update"], cwd=str(client_dir),
                      capture_output=True, text=True, timeout=timeout_s, check=False,
                      env=dict(env) if env is not None else None)
    except (OSError, subprocess.SubprocessError) as exc:
        return PreflightResult(False, f"could not run client.py --doctor: {type(exc).__name__}: {exc}")
    report = _extract_json(getattr(proc, "stdout", "") or "")
    if report is None:
        return PreflightResult(False, f"client.py --doctor --json printed no JSON report (exit {getattr(proc, 'returncode', '?')})")
    update = report.get("update") or {}
    local = str((report.get("client") or {}).get("version") or "") or None
    remote = str(update.get("remote_version") or "") or None
    if update.get("error"):
        return PreflightResult(False, f"signed release manifest could not be checked: {update['error']}", local, remote, report)
    if update.get("checked") is not True:
        return PreflightResult(False, "signed release manifest was not checked", local, remote, report)
    if update.get("signature") is not True:
        return PreflightResult(False, "release manifest signature did not verify", local, remote, report)
    if update.get("client_matches") is not True:
        return PreflightResult(False, "client.py differs from the signed release manifest (modified or outdated copy)", local, remote, report)
    if not local or not remote or local != remote:
        return PreflightResult(False, f"client version mismatch: local {local}, remote {remote}; update the checkout", local, remote, report)
    return PreflightResult(True, f"client.py {local} matches the signed release manifest", local, remote, report)


def account_mode(env: Mapping[str, str]) -> str:
    """``token``, ``password``, ``saved`` or ``anon`` depending on what the user opted into via env."""
    if env.get("CROWDGPT_TOKEN", "").strip():
        return "token"
    if env.get("CROWDGPT_USER", "").strip() and env.get("CROWDGPT_PASSWORD", "").strip():
        return "password"
    if env.get("CROWDGPT_USE_SAVED_SESSION", "").strip() in {"1", "true", "yes"}:
        return "saved"
    return "anon"


def build_client_command(python: str, client_dir: str | os.PathLike[str], *, vram_gb: float, backend: str = "auto",
                         env: Mapping[str, str] | None = None, rounds: int = 0) -> list[str]:
    """Argv for the CrowdGPT client. Secrets stay in the environment; only a username may appear in argv."""
    env = env if env is not None else os.environ
    argv = [python, str(Path(client_dir) / "client.py"), "--headless", "--no-update", "--no-tui"]
    mode = account_mode(env)
    if mode == "anon":
        argv.append("--anon")
    elif mode == "password":
        argv += ["--user", env["CROWDGPT_USER"].strip()]
    argv += ["--backend", backend or "auto", "--vram-gb", f"{float(vram_gb):g}"]
    if rounds > 0:
        argv += ["--rounds", str(int(rounds))]
    return argv


def build_client_env(base_env: Mapping[str, str], data_dir: str | os.PathLike[str]) -> dict[str, str]:
    env = dict(base_env)
    env["HF_HOME"] = str(Path(data_dir) / "hf")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if account_mode(base_env) == "anon":
        for key in SECRET_ENV_KEYS:
            env.pop(key, None)
    return env


def redact(text: str, env: Mapping[str, str]) -> str:
    for key in SECRET_ENV_KEYS:
        secret = env.get(key, "")
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


def resolve_vram_gb(user_vram_gb: float, snapshot: GpuSnapshot | None, cfg: GuardConfig) -> tuple[float | None, str]:
    """Pick the ``--vram-gb`` value. Without nvidia-smi an explicit value is required."""
    if snapshot is None:
        if user_vram_gb and user_vram_gb > 0:
            return float(user_vram_gb), "no nvidia-smi: using the explicit --vram-gb; automatic memory release is unavailable"
        return None, "no nvidia-smi on this GPU: pass --vram-gb explicitly (automatic memory release is unavailable)"
    budget = plan_vram_budget_gb(snapshot, cfg.reserve_mib, cfg.context_mib, cfg.min_gb)
    if budget is None:
        return None, f"only {snapshot.free_mib} MiB free; below reserve {cfg.reserve_mib} + context {cfg.context_mib} + {cfg.min_gb} GB"
    if user_vram_gb and user_vram_gb > 0:
        if user_vram_gb > budget:
            return budget, f"--vram-gb {user_vram_gb:g} capped to the {budget:g} GB budget"
        return float(user_vram_gb), f"--vram-gb {user_vram_gb:g} (budget {budget:g} GB)"
    return budget, f"auto budget {budget:g} GB"


def next_stop_at(hhmm: str, now: _dt.datetime) -> _dt.datetime:
    """Next local occurrence of HH:MM after ``now`` (tomorrow if it already passed today)."""
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", hhmm.strip())
    if not m:
        raise ValueError(f"--stop-at must be HH:MM, got {hhmm!r}")
    target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    if target <= now:
        target += _dt.timedelta(days=1)
    return target


def find_running_processes(names: Iterable[str], psutil_mod: Any) -> tuple[str, ...]:
    """Names from ``names`` that match a running process (case-insensitive, with or without ``.exe``)."""
    wanted = {n.strip().lower().removesuffix(".exe") for n in names if n.strip()}
    if not wanted or psutil_mod is None:
        return ()
    found: set[str] = set()
    for proc in psutil_mod.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower().removesuffix(".exe")
        except Exception:  # noqa: BLE001 - a process may vanish while iterating
            continue
        if name in wanted:
            found.add(name)
    return tuple(sorted(found))


# --------------------------------------------------------------------------- client process


class PopenClientProcess:
    """Real client process: stdout and stderr merged, read on a thread into a queue."""

    def __init__(self, argv: list[str], *, cwd: str, env: Mapping[str, str]) -> None:
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv, cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        stream = self._proc.stdout
        if stream is None:
            return
        for line in stream:
            self._lines.put(line)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def drain_lines(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self._lines.get_nowait())
            except queue.Empty:
                return out


def _default_process_factory(argv: list[str], cwd: str, env: Mapping[str, str]) -> PopenClientProcess:
    return PopenClientProcess(argv, cwd=cwd, env=env)


# --------------------------------------------------------------------------- supervisor


@dataclass
class SupervisorOptions:
    client_dir: str
    python: str
    data_dir: str
    backend: str = "auto"
    user_vram_gb: float = 0.0
    once: bool = False
    stop_at_epoch: float | None = None
    poll_s: float = 5.0
    term_grace_s: float = 30.0
    max_crashes: int = 3
    nvidia: bool = True


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Supervisor:
    """Runs the CrowdGPT client under the guard. Every side effect is injected."""

    def __init__(
        self,
        opts: SupervisorOptions,
        cfg: GuardConfig,
        *,
        snapshot_fn: Callable[[], GpuSnapshot | None],
        process_factory: Callable[[list[str], str, Mapping[str, str]], Any],
        psutil_mod: Any,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        env: Mapping[str, str] | None = None,
        out: Callable[[str], None] = print,
        utc: Callable[[], str] = _utc,
        max_ticks: int | None = None,
    ) -> None:
        self.opts, self.cfg = opts, cfg
        self.snapshot_fn, self.process_factory, self.psutil = snapshot_fn, process_factory, psutil_mod
        self.clock, self.sleep, self.out, self.utc = clock, sleep, out, utc
        self.env = dict(env if env is not None else os.environ)
        self.max_ticks = max_ticks
        self.state = GuardState()
        self.proc: Any = None
        self.round_no = 0
        self.current_round: dict[str, Any] | None = None
        self.pending: deque[dict[str, Any]] = deque()
        self.launches = 0
        self.stops = 0
        self.crashes = 0
        self.receipts_written = 0
        self.data_dir = Path(opts.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_path = self.data_dir / RECEIPTS_FILENAME
        self.client_log_path = self.data_dir / CLIENT_LOG_FILENAME

    # ---- output and receipts
    def _say(self, msg: str) -> None:
        self.out(redact(f"[crowdgpt-contributor] {msg}", self.env))

    def _write_receipt(self, entry: dict[str, Any], *, submitted: bool, outcome: str, evidence: str) -> None:
        receipt = {
            "round": entry["round"],
            "client_round": entry.get("client_round"),
            "steps": entry.get("steps"),
            "mean_loss": entry.get("mean_loss"),
            "std": entry.get("std"),
            "last": entry.get("last"),
            "submitted": submitted,
            "outcome": outcome,
            "evidence": evidence,
            "utc": self.utc(),
        }
        with self.receipts_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, sort_keys=True) + "\n")
        self.receipts_written += 1
        self._say(f"round {entry['round']}: {outcome}")

    def _resolve_pending(self, *, submitted: bool, outcome: str, evidence: str = "client_reported") -> None:
        if self.pending:
            self._write_receipt(self.pending.popleft(), submitted=submitted, outcome=outcome, evidence=evidence)

    def handle_line(self, line: str) -> ClientEvent | None:
        clean = redact(line.rstrip("\r\n"), self.env)
        with self.client_log_path.open("a", encoding="utf-8") as fh:
            fh.write(clean + "\n")
        ev = parse_client_line(clean)
        if ev is None:
            return None
        if ev.kind == "round_start":
            self.round_no += 1
            self.current_round = {"round": self.round_no, "client_round": ev.data["round"]}
        elif ev.kind == "step" and self.current_round is not None:
            self.current_round["last_step"] = ev.data
        elif ev.kind == "round_done":
            entry = self.current_round or {"round": self.round_no or 1, "client_round": None}
            entry.update(steps=ev.data["steps"], mean_loss=ev.data["mean_loss"], std=ev.data["std"], last=ev.data["last"])
            self.pending.append(entry)
            self.current_round = None
        elif ev.kind == "submitted":
            self._resolve_pending(submitted=True, outcome="submitted")
            self.crashes = 0
        elif ev.kind == "not_counted":
            self._resolve_pending(submitted=False, outcome="not_counted_by_server")
        elif ev.kind in ("round_lost", "upload_failed"):
            self._resolve_pending(submitted=False, outcome=ev.kind)
        elif ev.kind == "stopped_by_user" and self.current_round is not None:
            self._write_receipt(self.current_round, submitted=False, outcome="stopped_before_upload", evidence="client_reported")
            self.current_round = None
        elif ev.kind == "error":
            self._say(f"client error: {ev.data.get('detail', '')}"[:300])
        return ev

    def _drain(self) -> None:
        if self.proc is not None:
            for line in self.proc.drain_lines():
                self.handle_line(line)

    def _flush_unresolved(self, why: str) -> None:
        while self.pending:
            self._write_receipt(self.pending.popleft(), submitted=False, outcome=f"no_submit_line_before_{why}",
                                evidence="no_submit_line_seen")
        if self.current_round is not None:
            self._write_receipt(self.current_round, submitted=False, outcome=f"round_lost_on_{why}",
                                evidence="supervisor_observed")
            self.current_round = None

    # ---- process control
    def _pid_tree(self) -> frozenset[int]:
        if self.proc is None:
            return frozenset()
        pids = {self.proc.pid}
        try:
            for child in self.psutil.Process(self.proc.pid).children(recursive=True):
                pids.add(child.pid)
        except Exception:  # noqa: BLE001 - process already gone
            return frozenset(pids)
        return frozenset(pids)

    def _each_process(self, method: str) -> None:
        for pid in sorted(self._pid_tree()):
            try:
                getattr(self.psutil.Process(pid), method)()
            except Exception as exc:  # noqa: BLE001 - process may have exited meanwhile
                self._say(f"{method}({pid}) failed: {type(exc).__name__}")

    def _launch(self, snapshot: GpuSnapshot | None) -> bool:
        vram_gb, note = resolve_vram_gb(self.opts.user_vram_gb, snapshot, self.cfg)
        if vram_gb is None:
            self._say(f"not launching: {note}")
            return False
        argv = build_client_command(self.opts.python, self.opts.client_dir, vram_gb=vram_gb, backend=self.opts.backend,
                                    env=self.env, rounds=1 if self.opts.once else 0)
        self.proc = self.process_factory(argv, self.opts.client_dir, build_client_env(self.env, self.data_dir))
        self.launches += 1
        self._say(f"launched client pid {self.proc.pid} ({note}): {redact(' '.join(argv), self.env)}")
        return True

    def _terminate(self, why: str) -> None:
        """One graceful signal, wait up to ``term_grace_s``, then kill. Never a second early signal (it force-exits)."""
        if self.proc is None:
            return
        self._each_process("resume")  # a suspended process cannot handle SIGTERM
        try:
            self.psutil.Process(self.proc.pid).terminate()
        except Exception as exc:  # noqa: BLE001
            self._say(f"terminate failed: {type(exc).__name__}")
        start = self.clock()
        while self.proc.poll() is None and self.clock() - start < self.opts.term_grace_s:
            self._drain()
            self.sleep(1.0)
        if self.proc.poll() is None:
            self._say("client did not exit in time; killing")
            self._each_process("kill")
        self._drain()
        self._flush_unresolved(why)
        self.proc = None

    # ---- main loop
    def tick(self) -> int | None:
        """One supervision step. Returns an exit code when the supervisor must end, else ``None``."""
        now = self.clock()
        if self.opts.stop_at_epoch is not None and now >= self.opts.stop_at_epoch:
            self._say("--stop-at reached; stopping the client (a round in progress is lost)")
            self._terminate("stop_at")
            return 0
        self._drain()
        if self.proc is not None:
            rc = self.proc.poll()
            if rc is not None:
                self._drain()
                self._flush_unresolved("client_exit")
                self.proc = None
                if rc == 0:
                    self._say("client finished")
                    return 0
                if rc in FATAL_CLIENT_EXIT_CODES:
                    self._say(f"client exited {rc}: {FATAL_CLIENT_EXIT_CODES[rc]}; not relaunching")
                    return rc
                self.crashes += 1
                if self.crashes >= self.opts.max_crashes:
                    self._say(f"client crashed {self.crashes} times (last exit {rc}); giving up")
                    return 1
                self._say(f"client exited {rc}; will relaunch after the idle period")
                _enter(self.state, Phase.STOPPED, f"client exited {rc}")
                self.state.stopped_at = now
                self.state.quick_stops += 1
        snapshot = self.snapshot_fn()
        pids = self._pid_tree()
        client_mem = 0
        if snapshot is not None:
            client_mem = sum(a.used_mib or 0 for a in snapshot.apps if a.pid in pids)
        self.state.now = now
        self.state.client_pids = pids
        self.state.pause_for_running = find_running_processes(self.cfg.pause_for, self.psutil)
        before = self.state.phase
        action = decide(self.state, snapshot, client_mem, self.cfg)
        if action == Action.STOP:
            self.stops += 1
            self._say(f"STOP: {self.state.reason}; terminating the client to free GPU memory")
            self._terminate("stop")
        elif action == Action.PAUSE and before == Phase.RUNNING:
            self._say(f"PAUSE: {self.state.reason}")
            self._each_process("suspend")
        elif action == Action.RUN and before == Phase.PAUSED:
            self._say(f"RESUME: {self.state.reason}")
            self._each_process("resume")
        elif action == Action.RUN and before in (Phase.INIT, Phase.STOPPED):
            if not self._launch(snapshot):
                _enter(self.state, before, "launch refused")
        return None

    def run(self) -> int:
        ticks = 0
        try:
            while True:
                rc = self.tick()
                if rc is not None:
                    return rc
                ticks += 1
                if self.max_ticks is not None and ticks >= self.max_ticks:
                    self._terminate("supervisor_exit")
                    return 0
                self.sleep(self.opts.poll_s)
        except KeyboardInterrupt:
            self._say("interrupted; stopping the client (a round in progress is lost)")
            self._terminate("interrupt")
            return 130


# --------------------------------------------------------------------------- entry point used by the agent


@dataclass
class ContributorArgs:
    crowdgpt_dir: str
    data_dir: str
    reserve_mib: int = 1500
    vram_gb: float = 0.0
    backend: str = "auto"
    pause_for: tuple[str, ...] = ()
    hot_c: float = 82.0
    cool_c: float = 74.0
    stop_at: str = ""
    once: bool = False
    dry_run: bool = False
    python: str = sys.executable


T4_MIN_RESERVE_MIB = 4300


def run_contributor(
    a: ContributorArgs,
    *,
    run: Callable[..., Any] | None = None,
    process_factory: Callable[..., Any] | None = None,
    psutil_mod: Any = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    now_local: Callable[[], _dt.datetime] = _dt.datetime.now,
    env: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
    max_ticks: int | None = None,
) -> int:
    """Preflight, plan and (unless ``dry_run``) supervise the CrowdGPT client. Returns a process exit code."""
    env = dict(env if env is not None else os.environ)
    runner = run or _default_run
    if not a.crowdgpt_dir or not (Path(a.crowdgpt_dir) / "client.py").is_file():
        out(install_instructions())
        return 2
    if a.cool_c >= a.hot_c:
        out(f"Error: --cool-c ({a.cool_c}) must be below --hot-c ({a.hot_c}).")
        return 2
    stop_at_epoch = None
    if a.stop_at:
        try:
            target = next_stop_at(a.stop_at, now_local())
        except ValueError as exc:
            out(f"Error: {exc}")
            return 2
        stop_at_epoch = clock() + (target - now_local()).total_seconds()
    cfg = GuardConfig(reserve_mib=int(a.reserve_mib), hot_c=float(a.hot_c), cool_c=float(a.cool_c),
                      pause_for=tuple(p for p in a.pause_for if p))
    snapshot = query_nvidia_gpu(runner)
    out(startup_notice(nvidia=snapshot is not None))
    if snapshot is not None and "T4" in snapshot.name and cfg.reserve_mib < T4_MIN_RESERVE_MIB:
        out(f"Error: on a Tesla T4 --reserve-mib must be >= {T4_MIN_RESERVE_MIB} (HiveMind inference headroom rule).")
        return 2
    vram_gb, note = resolve_vram_gb(a.vram_gb, snapshot, cfg)
    if snapshot is None and vram_gb is None:
        out(f"Error: {note}")
        return 2
    pre = preflight_client(a.crowdgpt_dir, a.python, run=runner, env=build_client_env(env, a.data_dir))
    out(f"Preflight: {'OK' if pre.ok else 'REFUSED'}: {pre.reason}")
    if not pre.ok:
        out("Run `python client.py --doctor --no-update` in the CrowdGPT folder for details, then `git pull` to the signed release.")
        return 3
    if snapshot is not None:
        out(f"GPU: {snapshot.name or 'NVIDIA'} {snapshot.free_mib}/{snapshot.total_mib} MiB free, "
            f"reserve {cfg.reserve_mib} MiB, context {cfg.context_mib} MiB -> {note}")
    else:
        out(f"GPU: not NVIDIA / no nvidia-smi -> {note}")
    out(f"Account: {account_mode(env)}; data dir: {a.data_dir}; receipts: {Path(a.data_dir) / RECEIPTS_FILENAME}")
    if stop_at_epoch is not None:
        out(f"Stop at: {a.stop_at} local time")
    if a.dry_run:
        if vram_gb is None:
            out(f"Dry run, nothing started. Not enough free GPU memory now ({note}); the supervisor would wait.")
        else:
            argv = build_client_command(a.python, a.crowdgpt_dir, vram_gb=vram_gb, backend=a.backend, env=env,
                                        rounds=1 if a.once else 0)
            out("Dry run, nothing started. Would run: " + redact(" ".join(argv), env))
        return 0
    if psutil_mod is None:
        try:
            import psutil as psutil_mod  # noqa: PLC0415
        except ImportError:
            out("Error: psutil is required for contributor mode (pip install psutil).")
            return 2
    sup = Supervisor(
        SupervisorOptions(client_dir=a.crowdgpt_dir, python=a.python, data_dir=a.data_dir, backend=a.backend,
                          user_vram_gb=a.vram_gb, once=a.once, stop_at_epoch=stop_at_epoch, nvidia=snapshot is not None),
        cfg,
        snapshot_fn=lambda: query_nvidia_gpu(runner),
        process_factory=process_factory or _default_process_factory,
        psutil_mod=psutil_mod, clock=clock, sleep=sleep, env=env, out=out, max_ticks=max_ticks,
    )
    return sup.run()
