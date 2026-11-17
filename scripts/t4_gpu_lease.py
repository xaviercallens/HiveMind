#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cross-project GPU lease for the shared Tesla T4 (IP-296).

Every GPU job on the T4, whatever its project (HiveMind training, AutoevolveAI nightly jobs, experiment queues),
runs through this wrapper so the 4,300 MiB free-VRAM floor of AGENTS.md holds:

    python scripts/t4_gpu_lease.py run --need-mib 7000 --wait-s 600 -- ./run_cron_dream.sh
    python scripts/t4_gpu_lease.py status

1. Lease: an exclusive ``flock`` on a lock file shared by all projects, so two GPU jobs never start at once.
2. Admission: the job starts only if free VRAM >= floor + ``--need-mib`` (its measured peak). Otherwise the wrapper
   waits up to ``--wait-s`` and then exits 75 (EX_TEMPFAIL, "try later") without running it.
3. Watchdog: while the job runs, two samples below the floor in a row SIGTERM it (then SIGKILL after a grace period).

Why: on 2026-11-16 at 03:02 Paris a nightly job of another project loaded a model next to a capped HiveMind run
and free VRAM fell to 365 MiB. Each job had its own guard; none knew about the other.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_LOCK = Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/t4_gpu.lock")
FLOOR_MIB = 4300
EX_TEMPFAIL = 75


def nvidia_free_mib() -> int | None:
    """Free memory of GPU 0 in MiB, or None when nvidia-smi is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def admit(free_mib: int | None, need_mib: int, floor_mib: int = FLOOR_MIB) -> tuple[bool, str]:
    """May a job needing ``need_mib`` start now without pushing free VRAM below the floor?"""
    if free_mib is None:
        return False, "free VRAM unknown (nvidia-smi failed): not starting a GPU job blind"
    if free_mib - int(need_mib) < int(floor_mib):
        return False, f"{free_mib} MiB free - {need_mib} MiB needed < {floor_mib} MiB floor"
    return True, f"{free_mib} MiB free - {need_mib} MiB needed >= {floor_mib} MiB floor"


class Lease:
    """Exclusive flock on the shared lock file; records the holder in the file for `status`."""

    def __init__(self, path: Path = DEFAULT_LOCK):
        self.path = Path(path)
        self.fh = None

    def try_acquire(self, holder: dict) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(holder) + "\n")
        fh.flush()
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


def run_leased(cmd: Sequence[str], *, need_mib: int, wait_s: float, floor_mib: int = FLOOR_MIB,
               lock: Path = DEFAULT_LOCK, sample_s: float = 5.0, grace_s: float = 30.0,
               free_fn: Callable[[], int | None] = nvidia_free_mib, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
               popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> int:
    """Wait for the lease and enough free VRAM, run ``cmd``, watch the floor; return the job's exit code (or 75)."""
    lease = Lease(lock)
    holder = {"pid": os.getpid(), "cmd": " ".join(cmd)[:200], "need_mib": int(need_mib), "since": time.strftime("%FT%TZ", time.gmtime())}
    deadline = clock() + max(0.0, float(wait_s))
    while True:
        if lease.try_acquire(holder):
            ok, why = admit(free_fn(), need_mib, floor_mib)
            if ok:
                break
            lease.release()
        else:
            why = f"lease held by another GPU job ({lock})"
        if clock() >= deadline:
            log(f"t4_gpu_lease: not starting: {why}")
            return EX_TEMPFAIL
        sleep(min(30.0, max(1.0, sample_s)))
    log(f"t4_gpu_lease: lease acquired ({why}); starting: {holder['cmd']}")
    try:
        proc = popen(list(cmd))
        low = 0
        while proc.poll() is None:
            sleep(sample_s)
            free = free_fn()
            low = low + 1 if (free is not None and free < floor_mib) else 0
            if low >= 2:
                log(f"t4_gpu_lease: free VRAM {free} MiB < {floor_mib} MiB for 2 samples: stopping the job")
                proc.send_signal(signal.SIGTERM)
                end = clock() + grace_s
                while proc.poll() is None and clock() < end:
                    sleep(1.0)
                if proc.poll() is None:
                    proc.kill()
                break
        return int(proc.wait())
    finally:
        lease.release()


def status(lock: Path = DEFAULT_LOCK) -> dict:
    lease = Lease(lock)
    if lease.try_acquire({"probe": True}):
        lease.release()
        return {"held": False, "free_mib": nvidia_free_mib()}
    try:
        holder = json.loads(Path(lock).read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError):
        holder = None
    return {"held": True, "holder": holder, "free_mib": nvidia_free_mib()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run", help="run a GPU job under the lease")
    r.add_argument("--need-mib", type=int, required=True, help="the job's measured peak GPU memory, MiB")
    r.add_argument("--wait-s", type=float, default=0.0, help="wait this long for the lease and free memory")
    r.add_argument("--floor-mib", type=int, default=FLOOR_MIB)
    r.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sub.add_parser("status", help="who holds the lease, and free VRAM")
    s.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    a = ap.parse_args(argv)
    if a.action == "status":
        print(json.dumps(status(a.lock)))
        return 0
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give the command after --")
    if a.floor_mib < FLOOR_MIB:
        print(f"refused: --floor-mib {a.floor_mib} is below the {FLOOR_MIB} MiB T4 floor (AGENTS.md)", file=sys.stderr)
        return 2
    return run_leased(cmd, need_mib=a.need_mib, wait_s=a.wait_s, floor_mib=a.floor_mib, lock=a.lock)


if __name__ == "__main__":
    raise SystemExit(main())
