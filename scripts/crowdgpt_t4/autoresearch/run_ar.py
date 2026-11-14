"""autoresearch loop driver: one git commit per experiment, run, log results.tsv (untracked), keep or reset."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

AR = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp/autoresearch")
PY = "/home/callensxavier_gmail_com/venv/bin/python"
END = datetime(2026, 11, 14, 11, 0, tzinfo=ZoneInfo("Europe/Paris")).timestamp()
ORDER = ["sft", "jepa", "h1_low_lam", "h2_stopgrad", "h3_center", "h4_infonce", "h6_midlayer"]
RUN_TIMEOUT = 1800  # 600 s budget + model load + eval; > 2x budget = failure (autoresearch rule)
TSV = AR / "results.tsv"


def git(*a):
    return subprocess.run(["git", "-C", str(AR), *a], capture_output=True, text=True, check=True).stdout.strip()


def done():
    if not TSV.exists():
        return {}
    rows = [l.split("\t") for l in TSV.read_text().splitlines()[1:]]
    return {r[4].split(":")[0]: r for r in rows}


def run(name):
    if name in done():
        return
    if time.time() + 900 > END or (AR / "STOP").exists():
        print(f"skip {name}: deadline/stop", flush=True)
        return
    (AR / "EXPERIMENT").write_text(name + "\n")
    git("add", "EXPERIMENT"); git("commit", "-q", "--allow-empty", "-m", f"experiment: {name}")
    commit = git("rev-parse", "--short", "HEAD")
    log = AR / f"run_{name}.log"
    t0 = time.time()
    try:
        with open(log, "w") as f:
            rc = subprocess.call(["nice", "-n", "10", PY, "-u", str(AR / "train.py"), name, str(END)], stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    res = None
    for line in log.read_text().splitlines():
        if line.startswith("{\"experiment\""):
            res = json.loads(line)
    base = done().get("sft")
    if res is None:
        status, ce, mem, desc = "crash", 0.0, 0.0, f"{name}: rc={rc}"
    else:
        ce, mem = res["val_ce"], round(res["peak_vram_mb"] / 1024, 1)
        if name == "sft":
            status = "keep"
        else:
            bce, bfar = (float(base[1]), float(base[5])) if base else (9e9, 9e9)
            status = "keep" if (ce < bce and res["far_loss"] <= bfar + 0.05) else "discard"
        desc = f"{name}: far={res['far_loss']} dev_acc300={res['dev_acc300']} steps={res['steps']} jepa_last={res['jepa_last']}"
    new = not TSV.exists()
    with open(TSV, "a") as f:
        if new:
            f.write("commit\tval_ce\tmemory_gb\tstatus\tdescription\tfar_loss\n")
        f.write(f"{commit}\t{ce:.6f}\t{mem}\t{status}\t{desc}\t{res['far_loss'] if res else 0}\n")
    if status != "keep" and name not in ("jepa",):
        git("reset", "-q", "--hard", "HEAD~1")
    print(json.dumps({"name": name, "status": status, "val_ce": ce, "min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or ORDER):
        run(n)
    print("AR_DONE", flush=True)
