#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Experiment S-3 analysis: acceptance rules A1-A4 and paired exact McNemar tests, from the arm files (stdlib only).

    s3_analyze.py --baseline artifacts/experiments/S3_formulation_baseline_baseline_cpu.json \
                  --model-cpu artifacts/experiments/S3_formulation_model_<name>_cpu.json \
                  [--model-gpu artifacts/experiments/S3_formulation_model_<name>_gpu.json] [--json OUT]

The rules are the ones frozen in artifacts/experiments/S3_formulation_prereg.json; nothing is adjusted here. The GPU arm is descriptive and does not enter A1-A4.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LIMITS_SENTENCE_NOTE = "A4 uses the per-record flag answer_contains_limits_sentence written by the runner (the exact sentence is searched in every answer text)."


def load(path: str) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != "hivemind-s3-arm/1":
        raise SystemExit(f"{path}: not an S-3 arm file")
    return doc


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b = baseline-only correct, c = model-only correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def by_id(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in doc["records"]}


def paired(base: Dict[str, Any], model: Dict[str, Any], tier: Optional[str]) -> Dict[str, Any]:
    bi, mi = by_id(base), by_id(model)
    ids = sorted(set(bi) & set(mi))
    if tier:
        ids = [i for i in ids if bi[i]["tier"] == tier]
    b = sum(1 for i in ids if bi[i]["correct"] and not mi[i]["correct"])  # regressions
    c = sum(1 for i in ids if mi[i]["correct"] and not bi[i]["correct"])
    both = sum(1 for i in ids if bi[i]["correct"] and mi[i]["correct"])
    return {"n": len(ids), "baseline_correct": sum(bi[i]["correct"] for i in ids), "model_correct": sum(mi[i]["correct"] for i in ids),
            "both_correct": both, "baseline_only_b": b, "model_only_c": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "significant_at_0.05": mcnemar_exact(b, c) < 0.05, "regression_ids": [i for i in ids if bi[i]["correct"] and not mi[i]["correct"]]}


def rules(base: Dict[str, Any], cpu: Dict[str, Any]) -> Dict[str, Any]:
    pa = paired(base, cpu, "A")
    a1 = pa["model_correct"] >= pa["baseline_correct"] and pa["baseline_only_b"] == 0
    a3_bad = [(d["arm"], r["id"]) for d in (base, cpu) for r in d["records"] if r.get("verified_but_wrong")]
    a4_bad = [(d["arm"], r["id"]) for d in (base, cpu) for r in d["records"] if not r.get("answer_contains_limits_sentence")]
    return {
        "A1": {"verdict": "PASS" if a1 else "FAIL", "rule": "model CPU >= baseline on tier A with zero regressions",
               "tier_A": pa},
        "A2": {"verdict": "REPORTED", "tier_B": paired(base, cpu, "B"), "tier_C": paired(base, cpu, "C")},
        "A3": {"verdict": "PASS" if not a3_bad else "HARD FAILURE", "verified_but_wrong": a3_bad,
               "audited_but_wrong": {d["arm"]: sum(1 for r in d["records"] if r.get("audited_but_wrong")) for d in (base, cpu)},
               "note": "VERIFIED needs a library hit, which ad-hoc S-3 specs never get, so A3 can hold vacuously; audited_but_wrong is the descriptive count of undetectable formulation errors."},
        "A4": {"verdict": "PASS" if not a4_bad else "FAIL", "missing_limits_sentence": a4_bad, "note": LIMITS_SENTENCE_NOTE},
        "overall_paired": paired(base, cpu, None),
    }


def arm_line(doc: Dict[str, Any]) -> str:
    s = doc["summary"]
    tiers = "  ".join(f"{t}: {b['correct']}/{b['n']}" for t, b in s["by_tier"].items())
    return (f"{doc['arm']:<8} {doc['model']:<24} {doc['device']}  correct {s['overall']['correct']}/{s['overall']['n']}  {tiers}  exact-match {s['overall']['exact_match']}  "
            f"could-not-formulate {s['overall']['could_not_formulate']}  retries {s['overall']['retries_total']}  verified_but_wrong {s['overall']['verified_but_wrong']}  "
            f"audited_but_wrong {s['overall']['audited_but_wrong']}" + ("  [DRY RUN, scripted fake backend]" if doc.get("is_dry_run_with_scripted_fake_backend") else ""))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--model-cpu", required=True)
    ap.add_argument("--model-gpu", default="")
    ap.add_argument("--json", default="", help="write the analysis here")
    args = ap.parse_args(argv)
    base, cpu = load(args.baseline), load(args.model_cpu)
    gpu = load(args.model_gpu) if args.model_gpu else None
    out: Dict[str, Any] = {"baseline": arm_line(base), "model_cpu": arm_line(cpu), "rules": rules(base, cpu), "warnings": []}
    for d in [cpu] + ([gpu] if gpu else []):
        if d["binary"]["sha256"] != base["binary"]["sha256"]:
            out["warnings"].append(f"binary differs between baseline and {d['device']} arm: the comparison is not like for like")
        if d.get("reduced_set") or base.get("reduced_set"):
            out["warnings"].append("a reduced question set was used; this is not the registered 40-question run")
        if d.get("is_dry_run_with_scripted_fake_backend"):
            out["warnings"].append("a model arm file comes from the scripted fake backend dry run, not a real model")
    if gpu:
        out["model_gpu"] = arm_line(gpu)
        out["gpu_vs_baseline_descriptive"] = {t or "overall": paired(base, gpu, t) for t in (None, "A", "B", "C")}
        out["gpu_vs_cpu_descriptive"] = {t or "overall": paired(cpu, gpu, t) for t in (None, "A", "B", "C")}
    print(out["baseline"])
    print(out["model_cpu"])
    if gpu:
        print(out["model_gpu"])
    r = out["rules"]
    print(f"\nA1 {r['A1']['verdict']}: tier A model {r['A1']['tier_A']['model_correct']} vs baseline {r['A1']['tier_A']['baseline_correct']}, "
          f"regressions {r['A1']['tier_A']['baseline_only_b']} {r['A1']['tier_A']['regression_ids']}")
    for t in ("tier_B", "tier_C"):
        p = r["A2"][t]
        print(f"A2 {t}: model {p['model_correct']} vs baseline {p['baseline_correct']} (n={p['n']}); b={p['baseline_only_b']} c={p['model_only_c']} McNemar exact p={p['mcnemar_exact_p']:.4f}")
    pa = r["A1"]["tier_A"]
    print(f"   tier_A McNemar: b={pa['baseline_only_b']} c={pa['model_only_c']} p={pa['mcnemar_exact_p']:.4f}")
    po = r["overall_paired"]
    print(f"   overall McNemar: model {po['model_correct']} vs baseline {po['baseline_correct']}; b={po['baseline_only_b']} c={po['model_only_c']} p={po['mcnemar_exact_p']:.4f}")
    print(f"A3 {r['A3']['verdict']}: verified-but-wrong {r['A3']['verified_but_wrong']}; audited-but-wrong {r['A3']['audited_but_wrong']}")
    print(f"A4 {r['A4']['verdict']}: {len(r['A4']['missing_limits_sentence'])} answer(s) without the limits sentence")
    for w in out["warnings"]:
        print("WARNING:", w)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if r["A3"]["verdict"] == "HARD FAILURE" else 0


if __name__ == "__main__":
    sys.exit(main())
