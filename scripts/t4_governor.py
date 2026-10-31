#!/usr/bin/env python3
"""Idle-time governor for the shared T4: QLoRA training only when nobody is asking for inference.

Inference always has priority. The governor polls nginx's loopback ``stub_status`` once a second (live in-flight
requests, including ones that are still generating, which an access log would only show afterwards). While requests
are in flight it keeps the BOINC client's GPU mode at ``never`` (the client stops the GPU task; the QLoRA app resumes
from its segment checkpoint later). After ``--idle-start-s`` seconds without requests, and only if the card has room
and is cool, it sets the GPU mode to ``always`` so the training workunits run again.

It measures the yield latency (request seen -> BOINC GPU process gone) and writes state + events as JSON lines.
Standard library only. Run with --dry-run to log decisions without calling boinccmd.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def stub_status(url: str) -> dict[str, int]:
    txt = urllib.request.urlopen(url, timeout=2).read().decode()
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


def gpu_state() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(",")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    procs = [tuple(x.strip() for x in ln.split(",")) for ln in apps.splitlines() if ln.strip()]
    return {"used_mib": int(q[0]), "total_mib": int(q[1]), "temp_c": int(q[2]), "util_pct": int(q[3]), "procs": procs}


YIELD_GRACE_S = 0.1  # was 1.0. Why it can be short: the QLoRA app does not use the BOINC API, so it never reads the <quit/> BOINC posts and only ends by itself
#                      after 0.8-23 s; all four governor-ended yields measured 1122-1148 ms = this grace (1.0) + ~0.12 s to SIGTERM/SIGKILL the tree. The roadmap
#                      (HM-S19-Q35-03) asks for < 1.0 s. A kill loses at most the current segment either way (atomic checkpoints), so a shorter grace costs no more work.
SCHEMA = 2  # governor.jsonl event schema: 2 adds `schema` to every event and `other_procs` to a yield (fields are only ever added)


def other_gpu_procs(procs: list[tuple[str, ...]], mine: set[str], limit: int = 6) -> list[dict]:
    """The GPU processes that are not the BOINC task, as seen when the governor decided someone else was busy (telemetry only).

    Without this a yield with trigger "gpu" says only that something used >= 10 % of the card; it cannot say whether that was a user's Ollama
    client, the validator's replay or a research job. The command line is cut to 100 characters."""
    out = []
    for p in procs:
        if p[0] in mine:
            continue
        try:  # interpreter path reduced to its basename so the module / script name still fits: "python -W ignore -m hivemind...moe_app"
            argv = Path(f"/proc/{p[0]}/cmdline").read_bytes().decode(errors="replace").split("\0")
            cmd = " ".join([os.path.basename(argv[0]), *argv[1:]]).strip()[:100]
        except OSError:
            cmd = ""
        out.append({"pid": p[0], "name": p[1] if len(p) > 1 else "", "mib": p[2] if len(p) > 2 else "", "cmd": cmd})
    return out[:limit]


def boinc_gpu_pids(procs: list[tuple[str, ...]]) -> list[str]:
    """GPU processes that belong to the BOINC QLoRA app (python from the project venv running moe_app)."""
    out = []
    for pid, *_ in procs:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "real_client.moe_app" in cmd:
            out.append(pid)
    return out


def descendants(pid: int) -> list[int]:
    """All child/grandchild processes of ``pid`` (from /proc), the same set the BOINC client enumerates."""
    kids: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(p.name))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def pid_alive(pid: int) -> bool:
    """Running or sleeping; a zombie (dead, waiting to be reaped by its parent) and a vanished process both count as gone."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def terminate_tree(pid: int, grace_s: float = 1.0) -> bool:
    """End a BOINC task process and its descendants: SIGTERM, then SIGKILL for whatever is still alive after ``grace_s``.

    Why the governor does this instead of waiting for BOINC: for GPU jobs the client always "removes from memory" on a suspend by posting
    a <quit/> message into shared memory, which an app that does not use the BOINC API never reads, so the task only ends when it ends by
    itself (measured 0.8-23 s) or after the client's 60 s quit timeout. The QLoRA app checkpoints every segment, so ending it loses at most
    one segment, and the client treats the exit exactly like a requested quit (task_state QUIT_PENDING -> UNINITIALIZED, restart later)."""
    targets = [pid, *descendants(pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for t in targets:
            try:
                os.kill(t, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
        end = time.time() + (grace_s if sig == signal.SIGTERM else 2.0)
        while time.time() < end:
            if not any(pid_alive(t) for t in targets):
                return True
            time.sleep(0.05)
    return not any(pid_alive(t) for t in targets)


def parse_pmon(text: str) -> dict[str, int]:
    """``nvidia-smi pmon -s u`` -> {pid: sm%}. A '-' (no sample for that process) is omitted, never read as 0."""
    out: dict[str, int] = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= 4 and not ln.lstrip().startswith("#") and p[1].isdigit() and p[3].isdigit():
            out[p[1]] = int(p[3])
    return out


def other_gpu_busy(g: dict, threshold_pct: int = 10) -> tuple[bool, int]:
    """Is something other than the BOINC task using the GPU? This catches callers that never pass through nginx (local Ollama or
    llama-server clients) and requests shorter than the in-flight poll interval.

    No BOINC task running: whole-GPU utilisation >= threshold. BOINC task running: the highest SM% among the other processes."""
    mine = set(boinc_gpu_pids(g["procs"]))
    if not mine:
        return g["util_pct"] >= threshold_pct, g["util_pct"]
    try:
        pm = parse_pmon(subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=8).stdout)
    except Exception:
        return False, 0
    top = max((sm for pid, sm in pm.items() if pid not in mine), default=0)
    return top >= max(threshold_pct // 2, 5), top


def in_night(now: float, tz: str, start_h: int, end_h: int) -> bool:
    """True inside the owner's night window (wraps past midnight when start_h > end_h)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    h = datetime.fromtimestamp(now, ZoneInfo(tz)).hour
    return (start_h <= h < end_h) if start_h < end_h else (h >= start_h or h < end_h)


class Boinc:
    def __init__(self, boinccmd: str, lib: str, port: int, datadir: Path, dry: bool) -> None:
        self.cmd, self.lib, self.port, self.dir, self.dry = boinccmd, lib, port, datadir, dry
        self.mode = "unknown"

    def set_gpu(self, mode: str) -> bool:
        if self.dry:
            self.mode = mode
            return True
        pw = (self.dir / "gui_rpc_auth.cfg").read_text().strip()
        r = subprocess.run([self.cmd, "--host", f"127.0.0.1:{self.port}", "--passwd", pw, "--set_gpu_mode", mode, "0"],
                           capture_output=True, text=True, timeout=15, env=dict(os.environ, LD_LIBRARY_PATH=self.lib))
        ok = r.returncode == 0
        if ok:
            self.mode = mode
        return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-url", default="http://127.0.0.1:8091/nginx_status")
    ap.add_argument("--boinccmd", default="/mnt/disks/disk-socrateai-local-1/boinc/root/usr/bin/boinccmd")
    ap.add_argument("--lib", default="/mnt/disks/disk-socrateai-local-1/boinc/root/usr/lib/x86_64-linux-gnu")
    ap.add_argument("--boinc-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/node/client"))
    ap.add_argument("--rpc-port", type=int, default=31420)
    ap.add_argument("--telemetry", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl"))
    ap.add_argument("--state-file", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor_state.json"))
    ap.add_argument("--idle-start-s", type=float, default=1800.0, help="quiet seconds before training may run INSIDE the night window")
    ap.add_argument("--day-idle-s", type=float, default=7200.0, help="quiet seconds required OUTSIDE the night window (training is meant to be mostly nocturnal)")
    ap.add_argument("--night-tz", default="Europe/Paris", help="timezone of the owner's night (an assumption: change it if wrong)")
    ap.add_argument("--night-start-h", type=int, default=22)
    ap.add_argument("--night-end-h", type=int, default=7)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--min-free-mib", type=int, default=4300, help="free VRAM needed (beyond what BOINC itself holds) to start training")
    ap.add_argument("--max-temp-c", type=int, default=82)
    ap.add_argument("--hold-file", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD"),
                    help="operator switch: while this file exists the governor keeps training OFF however quiet the node is "
                         "(`touch` to pause training, `rm` to allow it again)")
    ap.add_argument("--yield-grace-s", type=float, default=YIELD_GRACE_S, help="seconds BOINC gets to stop the task by itself before the governor ends it")
    ap.add_argument("--no-kill", action="store_true", help="never end the task process directly; rely on BOINC's own (slow) stop")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.telemetry.parent.mkdir(parents=True, exist_ok=True)
    boinc = Boinc(a.boinccmd, a.lib, a.rpc_port, a.boinc_dir, a.dry_run)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    def log(ev: dict) -> None:
        ev["t"] = round(time.time(), 2)
        ev["schema"] = SCHEMA
        with open(a.telemetry, "a") as f:
            f.write(json.dumps(ev) + "\n")

    # a fresh start counts as "just busy": training waits out a full quiet period first
    state, last_busy, yields, last_yield_ms, last_snapshot = "init", time.time(), 0, None, 0.0
    # every (re)start begins with training OFF; it can only be turned on by a full quiet period
    boinc.set_gpu("never")
    state = "yielded"
    log({"event": "start", "dry_run": a.dry_run, "idle_start_s": a.idle_start_s, "day_idle_s": a.day_idle_s,
         "night": f"{a.night_start_h:02d}:00-{a.night_end_h:02d}:00 {a.night_tz}"})
    while not stop["now"]:
        t0 = time.time()
        try:
            st = stub_status(a.status_url)
            in_flight = max(st["writing"] - 1, 0)  # minus this poll itself
            http_ok = True
        except Exception as exc:
            in_flight, http_ok = 0, False
            st = {"error": str(exc)[:80]}
        # two independent "someone is asking" signals: nginx in-flight (router traffic) and GPU activity by anything but BOINC
        g_seen = None
        try:
            g_seen = gpu_state()
            other_busy, other_util = other_gpu_busy(g_seen)
        except Exception:
            other_busy, other_util = False, 0
        busy = in_flight >= 1 or other_busy
        trigger = "nginx" if in_flight >= 1 else ("gpu" if other_busy else None)
        if busy:
            last_busy = t0
        if a.hold_file.exists():
            if state != "held":
                boinc.set_gpu("never")
                state = "held"
                log({"event": "hold", "reason": f"{a.hold_file} exists: training is paused by the operator"})
            last_busy = t0  # leaving the hold starts a fresh quiet period
        elif state == "held":
            state = "yielded"
            log({"event": "hold_released"})
        elif busy and state != "yielded":
            task_pids = boinc_gpu_pids(gpu_state()["procs"]) if not a.dry_run else []
            had_task = bool(task_pids)
            ok = boinc.set_gpu("never") if boinc.mode != "never" or state != "yielded" else True
            t_req = time.time()
            freed_ms = None
            ended_by_governor = False
            if ok and task_pids and not a.no_kill:
                # BOINC only posts a <quit/> the non-API app never reads (measured stops: 0.8-23 s): give it a short grace, then end the task ourselves
                grace_end = time.time() + a.yield_grace_s
                while time.time() < grace_end and any(pid_alive(int(p)) for p in task_pids):
                    time.sleep(0.1)
                for p in [p for p in task_pids if pid_alive(int(p))]:
                    ended_by_governor = True
                    terminate_tree(int(p), grace_s=1.0)
            if ok and not a.dry_run:
                deadline = t_req + 30
                while time.time() < deadline:
                    if not boinc_gpu_pids(gpu_state()["procs"]):
                        freed_ms = int((time.time() - t_req) * 1000)
                        break
                    time.sleep(0.25)
            elif ok:
                freed_ms = 0
            yields += 1
            last_yield_ms = freed_ms
            state = "yielded"
            # gpu_freed_ms is a real measurement only when a BOINC GPU task was running (had_boinc_gpu_task); otherwise there was nothing to free
            others = other_gpu_procs(g_seen["procs"], set(boinc_gpu_pids(g_seen["procs"]))) if (g_seen and trigger == "gpu") else []
            log({"event": "yield", "trigger": trigger, "in_flight": in_flight, "other_gpu_util": other_util, "set_never_ok": ok,
                 "had_boinc_gpu_task": had_task, "ended_by_governor": ended_by_governor, "gpu_freed_ms": freed_ms, "yields": yields,
                 "other_procs": others})
        elif not busy and state in ("init", "yielded") and (t0 - last_busy) >= (a.idle_start_s if in_night(t0, a.night_tz, a.night_start_h, a.night_end_h) else a.day_idle_s):
            g = gpu_state()
            mine = sum(int(p[2]) for p in g["procs"] if p[0] in boinc_gpu_pids(g["procs"]))
            free = g["total_mib"] - g["used_mib"] + mine
            if free >= a.min_free_mib and g["temp_c"] < a.max_temp_c and http_ok:
                ok = boinc.set_gpu("always")
                if ok:
                    state = "training"
                    log({"event": "resume", "idle_s": round(t0 - last_busy, 1), "free_mib": free, "temp_c": g["temp_c"],
                         "night": in_night(t0, a.night_tz, a.night_start_h, a.night_end_h)})
            elif state == "init":
                boinc.set_gpu("never")
                state = "yielded"
        if t0 - last_snapshot >= 10:
            g = gpu_state()
            snap = {"state": state, "boinc_gpu_mode": boinc.mode, "in_flight": in_flight, "http_ok": http_ok, "yields": yields,
                    "last_yield_ms": last_yield_ms, "other_gpu_util": other_util, "busy_trigger": trigger, "quiet_s": round(t0 - last_busy, 1) if last_busy else None,
                    "gpu_used_mib": g["used_mib"], "gpu_temp_c": g["temp_c"], "gpu_util_pct": g["util_pct"], "boinc_procs": boinc_gpu_pids(g["procs"]), "t": round(t0, 1)}
            a.state_file.write_text(json.dumps(snap))
            log({"event": "snapshot", **{k: snap[k] for k in ("state", "in_flight", "gpu_used_mib", "gpu_util_pct", "yields", "last_yield_ms")}})
            last_snapshot = t0
        time.sleep(max(a.poll_s - (time.time() - t0), 0.05))
    boinc.set_gpu("never")
    log({"event": "stop"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
