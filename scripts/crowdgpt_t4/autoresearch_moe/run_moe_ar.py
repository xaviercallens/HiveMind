# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Run the pre-registered MoE autoresearch screen (program.md) on the T4. Restartable: state in OUT/state.json.

    setsid nohup python3 -u run_moe_ar.py >> OUT/runner.out 2>&1 &        # stop: touch OUT/STOP
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ar_decide import decide  # noqa: E402

PARIS = ZoneInfo("Europe/Paris")
W = pathlib.Path("/mnt/disks/disk-socrateai-local-1")
HW = W / "hivemind_work"
OUT = HW / "autoresearch_moe"
LAUNCHER = HW / "launch_crowdgpt_moe_interleaved.sh"
JUDGE_REPO = HW / "wt-ip300"  # feat/ip300-paired-moe-eval at 9bebbfd1 (frozen judge)
JUDGE_SPEC = JUDGE_REPO / "docs/E_B3_MOE_MIDROW_PREREGISTRATION.json"
JUDGE_SPEC_SHA = "dce3b43e3cab608693d2d063261c0331c1a7df471083f4db83374ef215cd5369"
PY_EVAL = HW / "venv311/bin/python3.11"
PRODUCER = "hivemind-live-granite-moe.service"
INNER, TOKENS_PER_LEARNER = 16, 30 * 16 * 64
R49 = W / "gcpcrowd_moe/run_20261116T104052Z_day_full/crowd_moe_opendiloco_round_49.safetensors"
COMMON = ("--outer-lr 0.2 --outer-lr-decay 0.5 --outer-lr-growth 1.25 --outer-lr-min 0.02 --local-data "
          "--max-rows-per-source 5000 --train-tokens-per-expert 262144 --heldout-tokens-per-expert 8192 "
          "--eval-windows 224 --stop-after-rejections 0")
BASE = {"ilv": 0, "seq": 64, "lr": None, "init": None, "off": 0}
SLICE2 = TOKENS_PER_LEARNER  # second data slice: the next 30,720 tokens per learner
TRIALS = [  # (id, on_champion, delta, description) -- program.md order, Amendment A1
    ("T1", False, {}, "baseline A: concatenated layout"),
    ("T2b", False, {"off": SLICE2}, "noise control: baseline on data slice 2"),
    ("T3", True, {"ilv": 1024}, "H1 interleaved expert shards (1024)"),
    ("T4", True, {"seq": 128}, "H3 seq_len 128 (same tokens)"),
    ("T5", True, {"init": str(R49)}, "H2 start from run 2 round 49 (on unseen data)"),
    ("T6", True, {"lr": 4e-4}, "H4 inner lr 4e-4"),
    ("T7", True, {}, "confirmation: champion on data slice 2 vs baseline on slice 2 (T2b)"),
]


def log(msg: str) -> None:
    line = f"{datetime.now(PARIS).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(OUT / "runner.log", "a") as f:
        f.write(line + "\n")


def sh(argv: list[str], env: dict | None = None, cwd: str | None = None, log_to: pathlib.Path | None = None
       ) -> subprocess.CompletedProcess:
    """Run argv without a shell. With log_to, stdout and stderr go to that file instead of being captured."""
    full_env = {**os.environ, **(env or {})}
    if log_to is None:
        return subprocess.run(argv, capture_output=True, text=True, env=full_env, cwd=cwd, check=False)
    with open(log_to, "w", encoding="utf-8") as fh:
        return subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, text=True, env=full_env, cwd=cwd, check=False)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def in_window(now: datetime) -> bool:
    hm = now.hour * 60 + now.minute
    return not (2 * 60 + 30 <= hm < 6 * 60 + 20)  # stop launching 10 min before the 02:40 STOP_AT


def stopped() -> bool:
    return (OUT / "STOP").exists()


def producer(active: bool) -> None:
    sh(["sudo", "-n", "systemctl", "start" if active else "stop", PRODUCER])
    log(f"producer {'started' if active else 'stopped'}: {sh(['systemctl', 'is-active', PRODUCER]).stdout.strip()}")


def worker_alive() -> bool:
    return sh(["pgrep", "-f", "[g]cp_t4_crowd_diloco_worker[.]py"]).returncode == 0


def load_state() -> dict:
    p = OUT / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"done": {}, "champion": None}


def save_state(s: dict) -> None:
    (OUT / "state.json").write_text(json.dumps(s, indent=1))


def flags(cfg: dict, offset: int) -> tuple[dict, str, int]:
    seq = int(cfg["seq"])
    extra = f"{COMMON} --seq-len {seq} --start-data-offset {offset}"
    if cfg.get("lr"):
        extra += f" --lr {cfg['lr']}"
    env = {"ILV": str(int(cfg["ilv"])), "RUN_TAG": "ar", "CAMPAIGN_ID": "CROWDGPT-MOE-AUTORESEARCH-2026-11",
           "STOP_AT": "02:40", "EXTRA_ARGS": extra}
    return env, extra, seq


def train(tid: str, cfg: dict) -> dict:
    """Train one trial to its token budget, in segments if a yield or the window cuts it. Returns the final state."""
    seq = int(cfg["seq"])
    target = TOKENS_PER_LEARNER // (INNER * seq)
    done, offset, init, segs, fails, last_ckpt, accepted = 0, int(cfg.get("off", 0)), cfg.get("init"), [], 0, None, 0
    while done < target:
        if stopped():
            raise SystemExit("STOP found")
        if not in_window(datetime.now(PARIS)) or worker_alive():
            time.sleep(300)
            continue
        producer(False)
        time.sleep(5)
        env, _extra, _ = flags(cfg, offset)
        if init:
            env["INIT"] = init
        out = sh(["bash", str(LAUNCHER), "full", str(target - done), str(INNER)], env=env).stdout.strip()
        m = re.search(r"logs in (\S+)", out)
        if "launched" not in out or not m:
            fails += 1
            log(f"{tid}: launch refused ({out.splitlines()[-1] if out else 'no output'}); retry {fails}")
            producer(True)
            if fails >= 6:
                raise RuntimeError("launch refused 6 times")
            time.sleep(600)
            continue
        run_dir = pathlib.Path(m.group(1))
        log(f"{tid}: segment {len(segs) + 1} in {run_dir.name} (rounds {target - done}, offset {offset}, init {init})")
        while worker_alive():
            if stopped():
                sh(["pkill", "-TERM", "-f", "[g]cp_t4_crowd_diloco_worker[.]py"])
            time.sleep(60)
        producer(True)
        rows = []
        if (run_dir / "rounds.jsonl").exists():
            rows = [json.loads(l) for l in (run_dir / "rounds.jsonl").read_text().splitlines() if l.strip()]
            rows = [r for r in rows if "accepted_by_loss_ratchet" in r and "round_index" in r]
        if not rows:
            fails += 1
            log(f"{tid}: segment produced no rounds; retry {fails}")
            if fails >= 2:
                raise RuntimeError("two segments without a round")
            continue
        done += len(rows)
        accepted += sum(1 for r in rows if r["accepted_by_loss_ratchet"])
        offset = int(rows[-1]["data_offsets"]["learner0"]) + INNER * seq
        ck = run_dir / f"crowd_moe_opendiloco_round_{rows[-1]['round_index']}.safetensors"
        if ck.exists():  # the saved state after a round is the ratchet's master = best accepted so far
            last_ckpt, init = str(ck), str(ck)
        evals = [r.get("eval_loss_after") for r in rows if r.get("eval_loss_after") is not None]
        segs.append({"run": run_dir.name, "rounds": len(rows), "train_best": min(evals) if evals else None})
        log(f"{tid}: {done}/{target} rounds, {accepted} accepted, next offset {offset}")
    if not last_ckpt:
        raise RuntimeError("no checkpoint written")
    return {"checkpoint": last_ckpt, "rounds": done, "accepted": accepted, "segments": segs}


def judge(tid: str, ckpt: str) -> dict:
    assert sha256(JUDGE_SPEC) == JUDGE_SPEC_SHA, "judge spec changed"
    dst = OUT / f"{tid}.safetensors"
    shutil.copyfile(ckpt, dst)
    s = json.loads(JUDGE_SPEC.read_text())
    s["experiment_id"] = f"AUTORESEARCH-MOE-2026-11-{tid}"
    s["arms"].pop("moe_secondary", None)
    s["arms"]["moe"].update({"checkpoint": str(dst), "checkpoint_source": ckpt + " (byte copy)",
                             "checkpoint_sha256": sha256(dst)})
    spec = OUT / f"{tid}_spec.json"
    spec.write_text(json.dumps(s, indent=1))
    rec = OUT / f"{tid}_judge.json"
    argv = ["nice", "-n", "19", str(PY_EVAL), "scripts/crowdgpt_moe_paired_eval.py",
            "--spec", str(spec), "--rows-cache-dir", f"{HW}/ip300_scratch/midrow/rows",
            "--train-rows-cache-dir", f"{W}/gcpcrowd_moe/rows", "--tokenizer-json", f"{W}/crowd_v1_cache/tokenizer.json",
            "--base-cache-dir", f"{W}/crowd_v1_cache", "--threads", "4", "--out", str(rec)]
    r = sh(argv, env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "src:."}, cwd=str(JUDGE_REPO),
           log_to=OUT / f"{tid}_judge.log")
    if r.returncode != 0 or not rec.exists():
        raise RuntimeError(f"judge failed (see {tid}_judge.log)")
    return json.loads(rec.read_text())


def tsv(row: dict) -> None:
    p = OUT / "results.tsv"
    cols = ["trial", "status", "J_midrow_nll", "mean_diff_vs_champion", "ci95_lo", "ci95_hi", "guard_ok", "rounds",
            "accepted", "per_expert_pct", "config", "description"]
    if not p.exists():
        p.write_text("\t".join(cols) + "\n")
    with open(p, "a") as f:
        f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    log(f"runner start; done={list(st['done'])}")
    for tid, on_champ, delta, desc in TRIALS:
        if tid in st["done"]:
            continue
        base = {**BASE, **(st["champion"]["config"] if (on_champ and st["champion"]) else {})}
        cfg = {**base, **delta}
        if "init" in delta:  # A1: r49 already read 76,800 tokens per learner of the concatenated layout: skip seen data
            cfg["off"] = 307200 if int(base["ilv"]) else 76800
        if tid == "T7":
            if st["champion"]["trial"] == "T1":
                log("T7 skipped: no trial beat the baseline, nothing to confirm")
                st["done"]["T7"] = {"status": "skipped (champion is the baseline)"}
                st["confirmed"] = False
                save_state(st)
                continue
            cfg["off"] = int(base.get("off", 0)) + SLICE2
        log(f"{tid}: {desc}; config {cfg}")
        res = None
        for attempt in (1, 2):
            try:
                tr = train(tid, cfg)
                rec = judge(tid, tr["checkpoint"])
                res = (tr, rec)
                break
            except SystemExit:
                log("STOP: runner exits; restart to resume")
                producer(True)
                return 0
            except Exception as exc:  # noqa: BLE001 - one retry, then logged as a crash
                log(f"{tid}: attempt {attempt} failed: {type(exc).__name__}: {exc}")
        if res is None:
            st["done"][tid] = {"status": "crash"}
            tsv({"trial": tid, "status": "crash", "config": json.dumps(cfg), "description": desc})
            save_state(st)
            continue
        tr, rec = res
        per = {e["expert_id"].split("_")[0]: round(float(e["relative_change_pct"]), 2) for e in rec["per_expert"]}
        row = {"trial": tid, "J_midrow_nll": round(float(rec["mean_moe_nll"]), 6), "rounds": tr["rounds"],
               "accepted": tr["accepted"], "per_expert_pct": json.dumps(per), "config": json.dumps(cfg),
               "description": desc}
        if tid == "T1":
            row.update(status="champion (baseline)", guard_ok=rec["domain_guard_ok"])
            st["champion"] = {"trial": "T1", "config": cfg, "receipt": str(OUT / "T1_judge.json")}
        else:
            ref = OUT / "T2b_judge.json" if tid == "T7" else (OUT / "T1_judge.json" if tid == "T2b" else pathlib.Path(st["champion"]["receipt"]))
            d = decide(rec, json.loads(pathlib.Path(ref).read_text()))
            row.update(mean_diff_vs_champion=round(d["mean_diff_vs_champion"], 6), ci95_lo=round(d["ci95_vs_champion"][0], 6),
                       ci95_hi=round(d["ci95_vs_champion"][1], 6), guard_ok=d["guard_ok"])
            row["per_expert_pct"] = json.dumps({"vs_base": per, "vs_ref": d["per_expert_pct_vs_champion"]})
            if tid == "T2b":
                row["status"] = "noise control (slice 2 vs T1)"
            elif tid == "T7":
                st["confirmed"] = d["status"] == "keep"
                row["status"] = ("CONFIRMED" if st["confirmed"] else "NOT CONFIRMED") + f" ({st['champion']['trial']} vs T2b on slice 2)"
            else:
                row["status"] = d["status"]
                if d["status"] == "keep":
                    st["champion"] = {"trial": tid, "config": cfg, "receipt": str(OUT / f"{tid}_judge.json")}
        tsv(row)
        st["done"][tid] = {**row, "segments": tr["segments"], "checkpoint": tr["checkpoint"]}
        save_state(st)
        log(f"{tid}: {row['status']} J={row['J_midrow_nll']} diff={row.get('mean_diff_vs_champion')} "
            f"ci=[{row.get('ci95_lo')}, {row.get('ci95_hi')}] guard={row.get('guard_ok')}")
    producer(True)
    log(f"screen finished; champion {st['champion']['trial']} confirmed={st.get('confirmed')} config {st['champion']['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
