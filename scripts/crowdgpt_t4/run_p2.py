"""P2 queue (see ../P2_GOAL.md): the paper's LoRA recipe on Qwen3-0.6B and Crowd-v1, SFT vs JEPA, paired seeds.

Launch only when the T4 is free for experiments (governor back to "held"):
    setsid nohup ~/venv/bin/python -u run_p2.py [--end-hour 23] >> ../logs/p2_queue.out 2>&1 &
Yields (no retry) when the weekend-training signals fire; resumes guard aborts from checkpoints; skips finished runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
RES, LOGS = ROOT / "results/p2", ROOT / "logs"
PARIS = ZoneInfo("Europe/Paris")
RECIPE = ["--epochs", "4", "--lr", "2e-5", "--bs", "8", "--accum", "2", "--rank", "64", "--n-test", "1000",
          "--need-mib", "2800", "--cap-mib", "2700"]
ORDER = [(m, arm, s) for s in (0, 1, 2) for m in ("qwen", "crowd") for arm in ("sft", "jepa")]


def event(**kw):
    kw["t"] = datetime.now(PARIS).isoformat(timespec="seconds")
    with open(LOGS / "p2_queue.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-hour", type=int, default=23, help="hard stop, Paris time, today (or tomorrow if past)")
    a = ap.parse_args()
    now = datetime.now(PARIS)
    end = now.replace(hour=a.end_hour, minute=0, second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    END = end.timestamp()
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    event(kind="p2_start", end=end.isoformat(timespec="minutes"))
    for model, arm, seed in ORDER:
        tag = f"{model}_{arm}_s{seed}"
        out = RES / f"{tag}.json"
        for attempt in range(8):
            if out.exists() and json.loads(out.read_text()).get("status") == "done":
                break
            if (ROOT / "STOP_P2").exists() or time.time() > END - 1800:
                event(kind="stop", run=tag); return
            extra = ["--k-pred", "1", "--lam", "1.0"] if arm == "jepa" else []
            cmd = [sys.executable, "-u", str(ROOT / "scripts/night_jepa.py"), "--model", model, "--arm", arm, "--seed", str(seed),
                   "--deadline", str(END), "--out", str(out), *RECIPE, *extra]
            event(kind="start", run=tag, attempt=attempt)
            with open(LOGS / f"p2_{tag}.log", "a") as lf:
                rc = subprocess.call(["nice", "-n", "10", *cmd], stdout=lf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                                     timeout=max(60.0, END + 120 - time.time()))
            st = json.loads(out.read_text()).get("status") if out.exists() else None
            r = json.loads(out.read_text()) if out.exists() else {}
            event(kind="end", run=tag, rc=rc, status=st, acc=r.get("accuracy"), far=r.get("far_loss"))
            if st in ("yield", "yield_eval"):
                event(kind="yield_stop", why=r.get("guard")); return  # the owner's training has priority: no retry
            if st == "done":
                break
            time.sleep(60)
    event(kind="p2_end")


if __name__ == "__main__":
    main()
