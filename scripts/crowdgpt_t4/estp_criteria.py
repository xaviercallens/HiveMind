# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E-STP-1 / E-STP-2 verdicts, computed exactly as pre-registered (docs/HIVEMIND_JEPA_LLM_PROPOSAL.md §4, Amendment A1).

Written 2026-11-16, BEFORE any E-STP result exists. Reads results/estp/<model>_f<frac>_<arm>_s<seed>.json
(written by night_jepa.py through run_estp.py) and writes results/estp/criteria_<model>.json.

A1 rules, per model:
1. Floor: a fraction is interpretable only if mean SFT exact match >= 5 %; otherwise "uninformative".
2. Success: mean(STP) - mean(SFT) > 2 x pooled sd at the primary fraction 1/16, OR at two interpretable fractions.
   And at every interpretable fraction: general-loss guard rail STP <= SFT + 0.05 nat (far_loss), and no STP run
   flagged by health().
3. The executed code is frozen (run_estp.py records the commit).

health(): the executed (frozen) code defines no health() function, so that clause cannot be evaluated. It is reported
as "not_evaluable" and the verdict states that it is conditional on it; no definition is invented after the fact.

    python scripts/crowdgpt_t4/estp_criteria.py [--results DIR] [--model crowd|qwen|all]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

FRACS = (1.0, 0.25, 0.0625)
PRIMARY = 0.0625
SEEDS = (0, 1, 2)
FLOOR_PCT = 5.0
GUARD_NAT = 0.05
DEFAULT_RESULTS = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp/results/estp")


def _tag(model: str, frac: float, arm: str, seed: int) -> str:
    return f"{model}_f{frac:g}_{arm}_s{seed}"


def _load(results: Path, model: str, frac: float, arm: str) -> list[dict[str, Any]]:
    out = []
    for s in SEEDS:
        p = results / f"{_tag(model, frac, arm, s)}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "done":
                out.append(r)
    return out


def pooled_sd(a: list[float], b: list[float]) -> float:
    return ((st.variance(a) + st.variance(b)) / 2) ** 0.5


def evaluate_model(results: Path, model: str) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for frac in FRACS:
        sft, stp = _load(results, model, frac, "sft"), _load(results, model, frac, "stp")
        row: dict[str, Any] = {"n_sft": len(sft), "n_stp": len(stp)}
        if len(sft) < len(SEEDS) or len(stp) < len(SEEDS):
            row["state"] = "incomplete"
            per[f"{frac:g}"] = row
            continue
        sa, ja = [float(r["accuracy"]) for r in sft], [float(r["accuracy"]) for r in stp]
        sf = [r.get("far_loss") for r in sft]
        jf = [r.get("far_loss") for r in stp]
        row.update(sft_acc=sa, stp_acc=ja, sft_mean=round(st.mean(sa), 3), stp_mean=round(st.mean(ja), 3),
                   diff=round(st.mean(ja) - st.mean(sa), 3), pooled_sd=round(pooled_sd(sa, ja), 3))
        row["interpretable"] = st.mean(sa) >= FLOOR_PCT
        row["state"] = "interpretable" if row["interpretable"] else "uninformative"
        row["gain_over_2sd"] = bool(row["diff"] > 2 * row["pooled_sd"])
        if None in sf or None in jf:
            row["guard_rail"] = "not_evaluable"
        else:
            row.update(sft_far_mean=round(st.mean(sf), 4), stp_far_mean=round(st.mean(jf), 4))
            row["guard_rail"] = "pass" if st.mean(jf) <= st.mean(sf) + GUARD_NAT else "fail"
        row["health"] = "not_evaluable"
        per[f"{frac:g}"] = row
    complete = all(per[f"{f:g}"]["state"] != "incomplete" for f in FRACS)
    interp = [f"{f:g}" for f in FRACS if per[f"{f:g}"].get("interpretable")]
    gains = [k for k in interp if per[k]["gain_over_2sd"]]
    primary_gain = f"{PRIMARY:g}" in gains
    guard_fail = [k for k in interp if per[k].get("guard_rail") == "fail"]
    guard_unknown = [k for k in interp if per[k].get("guard_rail") == "not_evaluable"]
    if not complete:
        verdict = "incomplete"
    elif not interp:
        verdict = "uninformative"
    elif guard_fail:
        verdict = "fail_guard_rail"
    elif primary_gain or len(gains) >= 2:
        verdict = "success_conditional_on_health" if not guard_unknown else "success_conditional_on_health_and_guard"
    else:
        verdict = "no_gain"
    return {"model": model, "rule": "A1 (2026-11-15)", "fractions": per, "interpretable_fractions": interp,
            "fractions_with_gain": gains, "primary_fraction_gain": primary_gain, "verdict": verdict,
            "notes": ["health() is not defined in the executed code: that clause is not evaluable",
                      "a negative verdict ends this branch of the program (CITIZEN_AI_JEPA_PROGRAM.md)"]}


def m4_gate(e1: dict[str, Any], e2: dict[str, Any]) -> str:
    """M4 (OpenTrainer stp_lora) is allowed only if E-STP-1 or E-STP-2 passed (M3 must also show no harm)."""
    ok = {"success_conditional_on_health", "success_conditional_on_health_and_guard"}
    if e1["verdict"] in ok or e2["verdict"] in ok:
        return "open_pending_M3"
    if "incomplete" in (e1["verdict"], e2["verdict"]):
        return "undecided"
    return "closed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    a = ap.parse_args(argv)
    e1, e2 = evaluate_model(a.results, "crowd"), evaluate_model(a.results, "qwen")
    out = {"E-STP-1": e1, "E-STP-2": e2, "M4_gate": m4_gate(e1, e2)}
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / "criteria_estp.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"E-STP-1": e1["verdict"], "E-STP-2": e2["verdict"], "M4_gate": out["M4_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
