"""Shared-T4 guard for CrowdGPT experiments: inference first, VRAM headroom always.

- Hard rule (AGENTS.md): free VRAM on the T4 must stay >= 4300 MiB. `check()` raises GuardAbort below that, so the caller
  checkpoints and exits instead of squeezing the live inference services.
- `cap_process()` caps this process with set_per_process_memory_fraction so an OOM lands on us, not on llama.cpp/ollama.
- `wait_idle()` pauses between steps while nginx shows in-flight inference requests (writing - 1 for our own poll).
- A lock file keeps a single GPU experiment at a time.
"""
from __future__ import annotations

import fcntl
import os
import time
import urllib.request

MIN_FREE_MIB = 4300
STATUS_URL = "http://127.0.0.1:8091/nginx_status"
LOCK_PATH = "/mnt/disks/disk-socrateai-local-1/crowdgpt_exp/gpu.lock"


class GuardAbort(RuntimeError):
    pass


def free_mib() -> float:
    import torch
    f, _ = torch.cuda.mem_get_info()
    return f / 2**20


def cap_process(margin_mib: int = 300, max_cap_mib: float | None = None) -> float:
    """Allow this process at most min(free - MIN_FREE_MIB - margin, max_cap_mib) MiB. Returns the cap in MiB.
    max_cap_mib sizes the process for the worst case (other tenants reloading their models), not for the current free."""
    import torch
    f, t = torch.cuda.mem_get_info()
    cap = f / 2**20 - MIN_FREE_MIB - margin_mib
    if max_cap_mib is not None:
        cap = min(cap, max_cap_mib)
    if cap < 1024:
        raise GuardAbort(f"only {f / 2**20:.0f} MiB free; cap {cap:.0f} MiB too small")
    torch.cuda.set_per_process_memory_fraction(cap * 2**20 / t)
    return cap


def wait_for_room(need_mib: float, max_wait_s: float = 1800, poll_s: float = 15.0) -> float:
    """Wait until free VRAM >= MIN_FREE_MIB + need_mib (other jobs, e.g. an ollama model, may hold memory for a while)."""
    t0 = time.time()
    while free_mib() < MIN_FREE_MIB + need_mib:
        if time.time() - t0 > max_wait_s:
            raise GuardAbort(f"no room: need {need_mib:.0f} MiB above the {MIN_FREE_MIB} MiB guard, free {free_mib():.0f} MiB")
        time.sleep(poll_s)
    return time.time() - t0


def check() -> None:
    fm = free_mib()
    if fm < MIN_FREE_MIB:
        raise GuardAbort(f"free VRAM {fm:.0f} MiB < {MIN_FREE_MIB} MiB")


def in_flight() -> int:
    try:
        txt = urllib.request.urlopen(STATUS_URL, timeout=2).read().decode()
    except Exception:
        return 0
    for ln in txt.splitlines():
        if ln.startswith("Reading:"):
            p = ln.replace(":", " ").split()
            return max(0, int(p[1]) + int(p[3]) - 1)
    return 0


DEADLINE: float | None = None  # unix time; set by the caller. Waiting past it raises DeadlineReached.


class DeadlineReached(RuntimeError):
    pass


def wait_idle(max_wait_s: float = 3600, poll_s: float = 1.0) -> float:
    """Block while inference requests are in flight (nginx gateway only: direct local calls to llama-server/ollama are
    not seen). Raises DeadlineReached if DEADLINE passes while waiting. Returns seconds waited."""
    t0 = time.time()
    while in_flight() > 0 and time.time() - t0 < max_wait_s:
        if DEADLINE and time.time() > DEADLINE:
            raise DeadlineReached("deadline while waiting for inference to go idle")
        time.sleep(poll_s)
    if DEADLINE and time.time() > DEADLINE:
        raise DeadlineReached("deadline")
    check()
    return time.time() - t0


class GpuLock:
    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
        self.fh = open(LOCK_PATH, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GuardAbort("another CrowdGPT GPU experiment holds the lock")
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()
