"""E-MOE-C3 driver (pre-registered: wt-c3 docs/E_MOE_C3_PREREGISTRATION.md). Trains arms D, B, C, A on the T4 through the
screen's train() (launcher, window, segments, producer stop/start), judges each on the skip-40 split on CPU from the
pinned judge checkout, applies the keep rule, and scores the winner once on skip 120.

    cd autoresearch_moe_code/c3 && setsid nohup python3 -u campaign3.py >> ../../autoresearch_moe/c3/runner.out 2>&1 &
    stop: touch autoresearch_moe/c3/STOP
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_moe_ar as ar  # noqa: E402  (the #948 driver with the SEC-01 fix)

PARIS = ar.PARIS
HW = ar.HW
OUT = HW / "autoresearch_moe" / "c3"
OUT.mkdir(parents=True, exist_ok=True)
ar.OUT = OUT
ar.TOKENS_PER_LEARNER = 30 * 16 * 128  # 30 rounds at seq 128
REPO_C3 = HW / "wt-c3"
C3_COMMIT = os.environ.get("C3_COMMIT", "")
JUDGE = HW / "wt-judge-e5f6faf4"
JUDGE_COMMIT = "e5f6faf4"
SPEC = JUDGE / "docs/E_B3_MOE_MIDROW_PREREGISTRATION.json"
SPEC_SHA = "dce3b43e3cab608693d2d063261c0331c1a7df471083f4db83374ef215cd5369"
E50 = HW / "autoresearch_moe/correction_a2/interp_e50_r00.safetensors"
E50_SHA = "80dccfa8246058ea71d56f924b679cad7376d99790f36816cbe998c80fcb22d3"
A2 = HW / "autoresearch_moe/correction_a2"
HARD_STOP = datetime(2026, 11, 25, 21, 0, tzinfo=PARIS)
ARMS = [("D", 8192, 0.0), ("B", 8192, 1.0), ("C", 0, 0.0), ("A", 0, 1.0)]
MIN_EFFECT, GUARD, SEED, MIN_N = 0.025, 1.0, 20261118, 20
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "src:."}
log = ar.log
_orig_flags = ar.flags


def flags(cfg: dict, offset: int):
    env, extra, seq = _orig_flags(cfg, offset)
    extra += f" --router-lr-scale {cfg['router']}"
    env.update({"EXTRA_ARGS": extra, "REPO": str(REPO_C3), "RUN_TAG": f"c3{cfg['arm']}",
                "CAMPAIGN_ID": "CROWDGPT-MOE-C3-2026-11"})
    return env, extra, seq


def stopped() -> bool:
    return (OUT / "STOP").exists() or datetime.now(PARIS) >= HARD_STOP


ar.flags = flags
ar.stopped = stopped


def git_head(repo: pathlib.Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def multi_eval(cands: dict[str, str], skip: int, out: pathlib.Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cf = out / f"cands_skip{skip}.json"
    cf.write_text(json.dumps(cands))
    cmd = ["nice", "-n", "19", str(ar.PY_EVAL), "scripts/crowdgpt_moe_multi_eval.py", "--spec", str(SPEC),
           "--skip-eligible", str(skip), "--candidates", str(cf), "--out-dir", str(out),
           "--rows-cache-dir", str(HW / "ip300_scratch/midrow/rows"), "--train-rows-cache-dir", str(ar.W / "gcpcrowd_moe/rows"),
           "--tokenizer-json", str(ar.W / "crowd_v1_cache/tokenizer.json"), "--base-cache-dir", str(ar.W / "crowd_v1_cache"),
           "--threads", "4", "--min-avail-gb", "12"]
    with open(out / f"multi_eval_skip{skip}.log", "a") as lf:
        return subprocess.run(cmd, cwd=str(JUDGE), env=ENV, stdout=lf, stderr=subprocess.STDOUT).returncode


def boot_ci(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(arm_rec: dict, ref_rec: dict) -> dict:
    """Per-expert % vs base, guard, and the paired (arm - reference) mean with CI over experts with >= 20 chunks."""
    ref = {(c["dataset_id"], int(c["row"])): float(c["moe_nll"]) for c in ref_rec["chunks"]}
    by = defaultdict(lambda: [[], [], []])
    for c in arm_rec["chunks"]:
        k = (c["dataset_id"], int(c["row"]))
        if k not in ref:
            raise RuntimeError(f"chunk {k} missing from the reference")
        by[c["expert_id"]][0].append(float(c["base_nll"]))
        by[c["expert_id"]][1].append(float(c["moe_nll"]))
        by[c["expert_id"]][2].append(ref[k])
    scored = {k: v for k, v in by.items() if len(v[0]) >= MIN_N}
    pct = {k: round(100.0 * (np.mean(v[1]) / np.mean(v[0]) - 1.0), 3) for k, v in scored.items()}
    d = np.array([m - r for v in scored.values() for m, r in zip(v[1], v[2])])
    lo, hi = boot_ci(d)
    guard = all(p <= GUARD for p in pct.values())
    return {"J": float(np.mean([m for v in scored.values() for m in v[1]])), "mean_diff_vs_e50": float(d.mean()),
            "ci95_vs_e50": [lo, hi], "n": int(len(d)), "per_expert_pct_vs_base": pct, "guard_ok": guard,
            "keep": bool(guard and d.mean() <= -MIN_EFFECT and hi < 0)}


def main() -> int:
    assert ar.sha256(SPEC) == SPEC_SHA, "judge spec changed"
    assert git_head(JUDGE).startswith(JUDGE_COMMIT), "judge checkout moved"
    assert ar.sha256(E50) == E50_SHA, "e50_r00 changed"
    assert C3_COMMIT and git_head(REPO_C3) == C3_COMMIT, "wt-c3 is not at the pre-registered commit"
    sel_dir = OUT / "judge_skip40"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for f in ("base_skip40.json", "e50_r00_skip40.json"):  # reused only if multi_eval finds the signature matches
        if not (sel_dir / f).exists():
            shutil.copyfile(A2 / f, sel_dir / f)
    st = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {"done": {}}
    log(f"E-MOE-C3 start: worker commit {C3_COMMIT[:8]}, judge {JUDGE_COMMIT}, arms {[a for a, _, _ in ARMS]}")
    judges: list[threading.Thread] = []
    for arm, ilv, router in ARMS:
        if arm in st["done"]:
            continue
        if stopped():
            log(f"{arm}: not started (STOP or hard stop {HARD_STOP.isoformat()}); dropped")
            break
        cfg = {"arm": arm, "ilv": ilv, "seq": 128, "lr": 4e-4, "init": str(E50), "off": 0, "router": router}
        log(f"{arm}: config {cfg}")
        try:
            tr = ar.train(f"c3{arm}", cfg)
        except SystemExit as e:
            log(f"{arm}: stopped during training ({e}); dropped")
            break
        ck = OUT / f"{arm}.safetensors"
        shutil.copyfile(tr["checkpoint"], ck)
        st["done"][arm] = {"config": cfg, "train": tr, "checkpoint": str(ck), "checkpoint_sha256": ar.sha256(ck)}
        (OUT / "state.json").write_text(json.dumps(st, indent=1))
        log(f"{arm}: trained {tr['rounds']} rounds, {tr['accepted']} accepted; judging on skip 40 in the background")
        t = threading.Thread(target=multi_eval, args=({arm: str(ck)}, 40, sel_dir), daemon=False)
        for prev in judges:  # one judge at a time (RAM): wait for the previous one before starting the next
            prev.join()
        t.start()
        judges.append(t)
    for t in judges:
        t.join()
    ref = json.loads((sel_dir / "e50_r00_skip40.json").read_text())
    sel = {}
    for arm in st["done"]:
        p = sel_dir / f"{arm}_skip40.json"
        if p.exists():
            sel[arm] = score(json.loads(p.read_text()), ref)
            log(f"{arm} skip40: {json.dumps(sel[arm])}")
    gen = {a: v["per_expert_pct_vs_base"].get("E0_general_fineweb") for a, v in sel.items()}
    contrasts = {}
    if all(a in sel for a in "ABCD"):
        m = {a: sel[a]["mean_diff_vs_e50"] for a in "ABCD"}
        contrasts = {"layout_mean": (m["B"] + m["D"]) / 2 - (m["A"] + m["C"]) / 2,
                     "router_mean": (m["C"] + m["D"]) / 2 - (m["A"] + m["B"]) / 2,
                     "layout_general_pct": (gen["B"] + gen["D"]) / 2 - (gen["A"] + gen["C"]) / 2,
                     "router_general_pct": (gen["C"] + gen["D"]) / 2 - (gen["A"] + gen["B"]) / 2}
    kept = {a: v for a, v in sel.items() if v["keep"]}
    winner = min(kept, key=lambda a: kept[a]["J"]) if kept else None
    decision = {"selection_skip40": sel, "contrasts": contrasts, "kept": sorted(kept), "winner": winner,
                "min_effect_nats": MIN_EFFECT, "worker_commit": C3_COMMIT, "judge_commit": JUDGE_COMMIT}
    log(f"kept {sorted(kept)}; winner {winner}; contrasts {contrasts}")
    if winner:
        conf_dir = OUT / "judge_skip120"
        rc = multi_eval({winner: st["done"][winner]["checkpoint"], "e50_r00": str(E50)}, 120, conf_dir)
        if rc == 0:
            c = score(json.loads((conf_dir / f"{winner}_skip120.json").read_text()),
                      json.loads((conf_dir / "e50_r00_skip120.json").read_text()))
            e50_vs_base = score(json.loads((conf_dir / "e50_r00_skip120.json").read_text()),
                                json.loads((conf_dir / "e50_r00_skip120.json").read_text()))["per_expert_pct_vs_base"]
            hi = c["ci95_vs_e50"][1]
            verdict = ("confirmed" if c["guard_ok"] and c["mean_diff_vs_e50"] <= -MIN_EFFECT and hi < 0 else
                       "partly confirmed" if c["guard_ok"] and hi < 0 else "not confirmed")
            decision["confirmation_skip120"] = {**c, "verdict": verdict, "e50_r00_pct_vs_base": e50_vs_base}
            log(f"skip120 {winner}: {verdict} {json.dumps(c)}")
        else:
            decision["confirmation_skip120"] = {"error": f"multi_eval rc {rc}"}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1))
    log("E-MOE-C3 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
