"""E-STP queue (docs/HIVEMIND_JEPA_LLM_PROPOSAL.md §4): Semantic Tube Prediction vs SFT, pre-registered.

E-STP-1: Crowd-v1 (rev 5ac3916 + engram), LoRA r64, NL-RX-SYNTH, data fractions 1, 1/4, 1/16 (equal steps), seeds 0-2.
E-STP-2: the same on Qwen3-0.6B.
Recipe: lr 2e-5, 4 epochs-equivalent of steps, effective batch 16, STP lam 0.02 (paper Table 2), 1 triple per example.

T4 courtesy rules, enforced here AND in t4guard inside each run:
- never start before NOT_BEFORE (owner: no GPU use during the weekend of 14-15 Nov 2026);
- run only while the governor is "held" with its HOLD file present (the owner's signal that the T4 is free for
  experiments) and no foreign GPU process holds > 1 GB; otherwise wait (polling), and resume from checkpoints;
- STOP_ESTP file stops the queue.
Launch: setsid nohup ~/venv/bin/python -u run_estp.py >> <logs>/estp_queue.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t4guard  # noqa: E402

EXP = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
RES, LOGS = EXP / "results/estp", EXP / "logs"
PARIS = ZoneInfo("Europe/Paris")
NOT_BEFORE = datetime(2026, 11, 16, 7, 0, tzinfo=PARIS).timestamp()
RECIPE = ["--epochs", "4", "--lr", "2e-5", "--bs", "8", "--accum", "2", "--rank", "64", "--n-test", "1000",
          "--need-mib", "2800", "--cap-mib", "2700", "--stp-lam", "0.02"]
FRACS = (1.0, 0.25, 0.0625)
QUEUE = ([("crowd", f, arm, s) for s in (0, 1, 2) for f in FRACS for arm in ("sft", "stp")]
         + [("qwen", f, arm, s) for s in (0, 1, 2) for f in FRACS for arm in ("sft", "stp")])


def event(**kw):
    kw["t"] = datetime.now(PARIS).isoformat(timespec="seconds")
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "estp_queue.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def allowed() -> tuple[bool, str]:
    if time.time() < NOT_BEFORE:
        return False, "before NOT_BEFORE (weekend)"
    try:
        t4guard.yield_check()
    except t4guard.YieldToTraining as e:
        return False, str(e)
    return True, "ok"


def tag(model, frac, arm, seed):
    return f"{model}_f{frac:g}_{arm}_s{seed}"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    event(kind="estp_start", n=len(QUEUE), not_before=datetime.fromtimestamp(NOT_BEFORE, PARIS).isoformat())
    last_reason = None
    for model, frac, arm, seed in QUEUE:
        t = tag(model, frac, arm, seed)
        out = RES / f"{t}.json"
        while True:
            if (EXP / "STOP_ESTP").exists():
                event(kind="stop_file"); return
            if out.exists() and json.loads(out.read_text()).get("status") == "done":
                break
            ok, why = allowed()
            if not ok:
                if why != last_reason:
                    event(kind="waiting", reason=why); last_reason = why
                time.sleep(600)
                continue
            last_reason = None
            cmd = [sys.executable, "-u", str(HERE / "night_jepa.py"), "--model", model, "--arm", arm, "--seed", str(seed),
                   "--frac", str(frac), "--out", str(out), *RECIPE]
            event(kind="start", run=t)
            with open(LOGS / f"estp_{t}.log", "a") as lf:
                rc = subprocess.call(["nice", "-n", "10", *cmd], stdout=lf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                          "PYTHONPATH": str(HERE.parents[1] / "src")})
            r = json.loads(out.read_text()) if out.exists() else {}
            event(kind="end", run=t, rc=rc, status=r.get("status"), acc=r.get("accuracy"), far=r.get("far_loss"))
            if r.get("status") != "done":
                time.sleep(300)  # yield / guard abort / crash: wait, then retry (the run resumes from its checkpoint)
    event(kind="estp_end")


if __name__ == "__main__":
    main()
