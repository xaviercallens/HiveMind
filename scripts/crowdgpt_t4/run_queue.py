"""Night queue for P1 (see ../GOAL.md): runs the arms in order, one at a time, deadline-aware, with the floor gate.

State: ../results/queue_state.json (one line per event in ../logs/queue.jsonl). Re-running resumes: finished runs are skipped.
Stop file: ../STOP (checked between runs) -> clean stop.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
RES, LOGS = ROOT / "results/p1", ROOT / "logs"
PARIS = ZoneInfo("Europe/Paris")
END = datetime(2026, 11, 14, 6, 30, tzinfo=PARIS).timestamp()
EXTRA_CUTOFF = datetime(2026, 11, 14, 3, 30, tzinfo=PARIS).timestamp()
QUEUE = [("sft", 0), ("jepa", 0), ("sft", 1), ("jepa", 1), ("sft", 2), ("jepa", 2)]
EST = {"sft": 45 * 60, "jepa": 60 * 60}  # initial estimates (train + eval), replaced by measurements
COMMON = ["--epochs", "3", "--bs", "8", "--accum", "2", "--lr", "2e-4", "--rank", "64", "--n-test", "1000", "--need-mib", "2800"]
MAX_ATTEMPTS = 12  # guard aborts resume from the checkpoint; this only bounds a crash loop


def event(**kw):
    kw["t"] = datetime.now(PARIS).isoformat(timespec="seconds")
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "queue.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def result(arm, seed, lam=1.0):
    tag = f"{arm}_s{seed}" + ("" if arm == "sft" or lam == 1.0 else f"_lam{lam:g}")
    return RES / f"{tag}.json", tag


def run(arm, seed, lam=1.0):
    out, tag = result(arm, seed, lam)
    if out.exists() and json.loads(out.read_text()).get("status") == "done":
        return json.loads(out.read_text())
    if (ROOT / "STOP").exists():
        event(kind="stop_file"); return None
    est = EST[arm]
    if time.time() + est > END:
        event(kind="skip_deadline", run=tag, est_min=round(est / 60)); return None
    event(kind="start", run=tag, est_min=round(est / 60))
    t0 = time.time()
    cmd = [sys.executable, "-u", str(ROOT / "scripts/night_jepa.py"), "--arm", arm, "--seed", str(seed), "--lam", str(lam),
           "--deadline", str(END), "--out", str(out), *COMMON]
    with open(LOGS / f"{tag}.log", "a") as lf:
        try:
            rc = subprocess.call(["nice", "-n", "10", *cmd], stdout=lf, stderr=subprocess.STDOUT,
                                 env={**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                                 timeout=max(60.0, END + 120 - time.time()))
        except subprocess.TimeoutExpired:
            rc = "timeout_killed"
    dur = time.time() - t0
    r = json.loads(out.read_text()) if out.exists() else {}
    if r.get("status") == "done":
        EST[arm] = dur * 1.1
    event(kind="end", run=tag, rc=rc, status=r.get("status"), acc=r.get("accuracy"), far=r.get("far_loss"), min=round(dur / 60, 1))
    return r if r.get("status") == "done" else None


def run_until_done(arm, seed, lam=1.0):
    """Attempt a run until it is done, the deadline makes it impossible, or a STOP file appears. Resumes from checkpoints."""
    for attempt in range(MAX_ATTEMPTS):
        r = run(arm, seed, lam)
        if r is not None:
            return r
        out, tag = result(arm, seed, lam)
        st = json.loads(out.read_text()).get("status") if out.exists() else None
        if (ROOT / "STOP").exists() or st in ("deadline_train", "deadline_eval") or time.time() > END - 300:
            return None
        if not out.exists() and time.time() + 60 > END:
            return None
        event(kind="retry", run=tag, attempt=attempt + 1, last_status=st)
        time.sleep(60)
    return None


def main():
    import fcntl
    lock = open(ROOT / "queue.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run_queue.py is running", flush=True); return
    RES.mkdir(parents=True, exist_ok=True)
    event(kind="queue_start", end="06:30 CET")
    for arm, seed in QUEUE:
        if (ROOT / "STOP").exists():
            event(kind="stop_file"); break
        r = run_until_done(arm, seed)
        if (arm, seed) == ("sft", 0):
            if r is None:
                event(kind="sft0_not_done_stop"); return
            if r["accuracy"] < 5.0:
                event(kind="floor_fail", acc=r["accuracy"]); (ROOT / "FLOOR_FAIL").write_text(json.dumps(r["accuracy"])); return
    done = all(result(a, s)[0].exists() and json.loads(result(a, s)[0].read_text()).get("status") == "done" for a, s in QUEUE)
    if done and time.time() < EXTRA_CUTOFF:
        run_until_done("jepa", 0, lam=4.0)
    event(kind="queue_end")


if __name__ == "__main__":
    main()
