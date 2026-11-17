# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""S-3 paired report: one model arm against the keyword baseline, per tier (stdlib only).

Used when the registered CPU arm file required by ``s3_analyze.py`` does not exist (amendment 1 records why for 2026-10-27). It reads two arm files,
computes for three metrics (the registered primary ``correct``, the secondary ``exact_match`` and ``agreement``) the counts per tier, the paired
exact McNemar p-value, the hard-failure counters (VERIFIED but wrong; AUDITED but wrong), the verdict counts, the could-not-formulate map by
problem type and the wall-clock figures. It changes no threshold and no rule; it only prints.

    python3 scripts/s3_paired_report.py --baseline artifacts/experiments/S3_formulation_baseline_baseline_cpu_10families.json \
        --model artifacts/experiments/S3_formulation_model_qwen2.5-coder-7b-instruct_gpu.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments"


def load_records(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "id" in value[0]:
            return data, {r["id"]: r for r in value}
    raise SystemExit(f"no per-question records in {path}")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b = model only correct, c = baseline only correct)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def report(baseline: Path, model: Path) -> None:
    _, base = load_records(baseline)
    _, mod = load_records(model)
    ids = sorted(set(base) & set(mod))
    print("questions in both arms:", len(ids))

    def table(metric: str, label: str) -> None:
        print(f"\n== {label}: paired exact test, model versus keyword baseline")
        for tier in ("A", "B", "C", "ALL"):
            sel = [i for i in ids if tier == "ALL" or mod[i]["tier"] == tier]
            m = sum(1 for i in sel if mod[i][metric])
            bl = sum(1 for i in sel if base[i][metric])
            only_m = sum(1 for i in sel if mod[i][metric] and not base[i][metric])
            only_b = sum(1 for i in sel if base[i][metric] and not mod[i][metric])
            print(f"  tier {tier:<3} n={len(sel):2d}  model {m:2d}  baseline {bl:2d}  model-only {only_m:2d}  baseline-only {only_b:2d}  McNemar p={mcnemar_exact(only_m, only_b):.4f}")

    table("correct", "PRIMARY (verdict VERIFIED or AUDITED, numbers agree)")
    table("exact_match", "SECONDARY (final spec equals the expected spec)")
    table("agreement", "SECONDARY (numbers agree with the expected answer, any verdict)")

    print("\n== hard-failure counters (model arm)")
    print("  VERIFIED but wrong (registered hard failure A3):", sum(1 for r in mod.values() if r.get("verified_but_wrong")))
    audited = [r for r in mod.values() if r["verdict"] == "AUDITED"]
    aw = [r["id"] + "/" + r["tier"] for r in mod.values() if r.get("audited_but_wrong")]
    print(f"  AUDITED but wrong (not part of A3, same kind of risk): {len(aw)} of {len(audited)} AUDITED answers -> {sorted(aw)}")
    print("  responses carrying the limits sentence (A4):", sum(1 for r in mod.values() if r.get("answer_contains_limits_sentence")), "of", len(mod))
    print("\n== verdicts (model):   ", dict(collections.Counter(r["verdict"] for r in mod.values())))
    print("== verdicts (baseline):", dict(collections.Counter(r["verdict"] for r in base.values())))
    right_failed = sorted(r["id"] for r in mod.values() if r["verdict"] == "FAILED" and r["agreement"])
    print(f"== numbers RIGHT but verdict FAILED (tolerance policy v1): {len(right_failed)} -> {right_failed}")

    print("\n== could not formulate, by problem type (model arm)")
    cnf = collections.Counter(r["type"] for r in mod.values() if r["verdict"] == "COULD_NOT_FORMULATE")
    total = collections.Counter(r["type"] for r in mod.values())
    for t in sorted(total):
        print(f"  {t:<11} {cnf.get(t, 0)} of {total[t]}")

    walls = [float(r["wall_seconds"]) for r in mod.values()]
    print(f"\n== wall seconds per question (model arm): median {statistics.median(walls):.1f}, max {max(walls):.1f}, total {sum(walls):.0f} (the first question includes a cold model load)")
    print("== retries (model arm):", dict(sorted(collections.Counter(r["retries"] for r in mod.values()).items())))


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path, default=EXP / "S3_formulation_baseline_baseline_cpu_10families.json")
    ap.add_argument("--model", type=Path, default=EXP / "S3_formulation_model_qwen2.5-coder-7b-instruct_gpu.json")
    a = ap.parse_args(argv)
    report(a.baseline, a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
