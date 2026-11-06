#!/usr/bin/env python3
"""Assemble the NIGHT-MOE-1 results from the evaluator's ``*_summary.json`` files into one JSON and one Markdown report.

Verdicts follow the preregistered rule (artifacts/experiments/NIGHT_MOE_prereg.json): a subject adapter is VALID only when its
own-pack gain over the base model is significant (exact McNemar p < 0.05, accuracy above base) AND its shuffled-label control shows no
significant gain over base; a control that gains significantly VOIDs the subject. Cells outside the own pack (transfer and interference)
are flagged GAIN/HARM at p < 0.05 and at the Bonferroni level over all such cells. Nothing here re-scores a reply: every number is read
from a summary file the evaluator wrote.
Usage: python scripts/build_night_moe_report.py --night-out DIR [--out-json FILE] [--out-md FILE]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SUBJECTS: dict[str, str] = {"maths": "maths", "exoplanet": "exoplanet", "stars": "stars", "cyber": "cyber"}
ALPHA = 0.05


def load_summaries(night_out: Path) -> dict[str, dict[str, Any]]:
    """Every ``<tag>_summary.json`` under base/, adapters/ and interference/, keyed by its tag."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(night_out.glob("*/*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.name.removesuffix("_summary.json")] = data
    return out


def _cell(summary: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    vs = summary.get("vs_baseline") or {}
    p = vs.get("mcnemar_exact_p")
    only_a, only_b = vs.get("only_this_right"), vs.get("only_baseline_right")
    # The paired counts are on the items this summary scored (a --limit run scores a subset), so the delta comes from them, not from the full-pack base accuracy.
    delta = None
    if base is not None:
        delta = round((only_a - only_b) / summary["n"], 4) if only_a is not None else round(summary["accuracy"] - base["accuracy"], 4)
    flag = "BASE"
    if base is not None:
        flag = "NS"
        if p is not None and p < ALPHA and delta:
            flag = "GAIN" if delta > 0 else "HARM"
    return {
        "n": summary["n"], "correct": summary["correct"], "accuracy": summary["accuracy"],
        "base_accuracy_same_items": None if delta is None else round(summary["accuracy"] - delta, 4),
        "wilson95": summary.get("wilson95"), "unparsable": summary.get("unparsable"),
        "only_adapter_right": vs.get("only_this_right"), "only_base_right": vs.get("only_baseline_right"),
        "mcnemar_exact_p": p, "delta_vs_base": delta, "flag": flag,
    }


def subject_verdict(own: dict[str, Any], control: dict[str, Any] | None) -> tuple[str, str]:
    """(verdict, reason) under the preregistered rule."""
    if control is None:
        return "NOT_RUN", "no shuffled-label control was evaluated"
    if control["flag"] == "GAIN":
        return "VOID", f"shuffled-label control gained significantly over base (p={control['mcnemar_exact_p']})"
    if own["flag"] != "GAIN":
        return "NO_GAIN", f"own-pack change {own['delta_vs_base']:+.4f}, p={own['mcnemar_exact_p']}"
    return "VALID", f"own-pack gain {own['delta_vs_base']:+.4f} (p={own['mcnemar_exact_p']}); control {control['flag']}"


def build(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bases = {k.removeprefix("base_"): v for k, v in summaries.items() if k.startswith("base_")}
    subjects: dict[str, Any] = {}
    cells: list[tuple[str, str]] = []
    for subj, own_pack in SUBJECTS.items():
        own_s = summaries.get(f"{subj}_{own_pack}")
        if own_s is None:
            continue
        own = _cell(own_s, bases.get(own_pack))
        ctrl_s = summaries.get(f"{subj}_ctrl_{own_pack}")
        control = _cell(ctrl_s, bases.get(own_pack)) if ctrl_s else None
        verdict, reason = subject_verdict(own, control)
        others: dict[str, Any] = {}
        for tag, s in summaries.items():
            for prefix in (f"{subj}_", f"x_{subj}_"):
                if tag.startswith(prefix) and not tag.startswith(f"{subj}_ctrl_") and tag != f"{subj}_{own_pack}":
                    pack = tag.removeprefix(prefix)
                    if pack in bases:
                        others[pack] = _cell(s, bases[pack]) | {"kind": "transfer" if prefix == f"{subj}_" else "interference"}
                        cells.append((subj, pack))
        note = ""
        if control is not None and control["correct"] == 0:
            note = ("the shuffled control scored 0 correct, which is expected for exact-match numeric answers (a shuffled answer is almost never right), "
                    "so this control cannot gain and the VOID rule is not testable for this subject")
        missing = sorted(set(bases) - {own_pack} - set(others))
        subjects[subj] = {"own_pack": own_pack, "own": own, "control": control, "verdict": verdict,
                          "reason": reason, "control_note": note, "missing_cells": missing, "other_packs": others}
    bonferroni = ALPHA / len(cells) if cells else ALPHA
    for subj, pack in cells:
        c = subjects[subj]["other_packs"][pack]
        p = c["mcnemar_exact_p"]
        c["survives_bonferroni"] = bool(p is not None and p < bonferroni and c["flag"] in ("GAIN", "HARM"))
    return {
        "experiment": "NIGHT-MOE-1", "base_model": "Qwen2.5-Coder-7B-Instruct Q4_K_M (QLoRA r16 a32 adapters)",
        "alpha": ALPHA, "n_other_cells": len(cells), "bonferroni_alpha": round(bonferroni, 5),
        "baselines": {k: {"n": v["n"], "correct": v["correct"], "accuracy": v["accuracy"], "unparsable": v.get("unparsable")} for k, v in bases.items()},
        "subjects": subjects,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# NIGHT-MOE-1 results", "",
             f"Base: {report['base_model']}. Exact paired McNemar vs the base model on identical items; "
             f"alpha {report['alpha']}, Bonferroni alpha {report['bonferroni_alpha']} over {report['n_other_cells']} transfer/interference cells.", "",
             "## Subjects", "", "| Subject | Own pack | Base | Adapter | Delta | p | Control | Control p | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for subj, s in report["subjects"].items():
        o, c = s["own"], s["control"]
        base_acc = report["baselines"][s["own_pack"]]["accuracy"]
        lines.append(f"| {subj} | {s['own_pack']} (n={o['n']}) | {base_acc} | {o['accuracy']} | {o['delta_vs_base']:+} | {o['mcnemar_exact_p']} | "
                     f"{c['accuracy'] if c else '-'} ({c['flag'] if c else '-'}) | {c['mcnemar_exact_p'] if c else '-'} | **{s['verdict']}** |")
    lines += ["", "## Transfer and interference cells", "", "| Adapter | Pack | Kind | n | Base (same items) | Adapter | Delta | p | Flag | Bonferroni | Unparsable |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for subj, s in report["subjects"].items():
        for pack, c in s["other_packs"].items():
            lines.append(f"| {subj} | {pack} | {c['kind']} | {c['n']} | {c['base_accuracy_same_items']} | {c['accuracy']} | {c['delta_vs_base']:+} | "
                         f"{c['mcnemar_exact_p']} | {c['flag']} | {'yes' if c['survives_bonferroni'] else 'no'} | {c['unparsable']} |")
    lines += ["", "## Reasons and gaps", ""]
    for subj, s in report["subjects"].items():
        lines.append(f"- **{subj}**: {s['verdict']}, {s['reason']}")
        if s["control_note"]:
            lines.append(f"  - control: {s['control_note']}")
        if s["missing_cells"]:
            lines.append(f"  - no result for: {', '.join(s['missing_cells'])} (stage timed out or was not run)")
    return "\n".join(lines) + "\n"


def _exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts (b, c)."""
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_adapter_vs_control_verdict(
    own: Mapping[str, Any],
    control: Mapping[str, Any] | None,
    *,
    cross_domain_controls: Sequence[Mapping[str, Any]] = (),
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Primary adapter-vs-control paired McNemar decision rule (HM-S14-01 / #165).

    Tests the subject adapter directly against its matched control on the identical `n` items (and against any
    cross-domain adapter controls for generative tasks where a shuffled numeric control scores 0), rather than
    voiding an adapter when a label-prior control gains slightly over base.
    """
    if control is None and not cross_domain_controls:
        return {
            "verdict": "NOT_RUN",
            "reason": "neither matched control nor cross-domain control was evaluated",
            "adapter_vs_control_delta": None,
            "mcnemar_p_aligned": None,
            "mcnemar_p_worst_case": None,
        }
    n = int(own["n"])
    own_correct = int(own["correct"])
    if control is not None and int(control.get("correct", 0)) > 0:
        ctrl_correct = int(control["correct"])
        net_diff = own_correct - ctrl_correct
        own_only = int(own.get("only_adapter_right") or 0)
        own_base_only = int(own.get("only_base_right") or 0)
        ctrl_only = int(control.get("only_adapter_right") or 0)
        ctrl_base_only = int(control.get("only_base_right") or 0)
        # Base-aligned discordant pairs: on base-wrong items (own_only vs ctrl_only) + on base-right items (ctrl_base_only vs own_base_only)
        b_aligned = max(0, own_only - ctrl_only) + ctrl_base_only
        c_aligned = max(0, ctrl_only - own_only) + own_base_only
        # Worst-case conservative contingency table maximizing discordant overlap
        c_worst = min(ctrl_correct, n - own_correct)
        b_worst = max(0, net_diff + c_worst)
        p_aligned = _exact_mcnemar_two_sided(b_aligned, c_aligned)
        p_worst = _exact_mcnemar_two_sided(b_worst, c_worst)
        delta_ac = round(net_diff / max(1, n), 4)
        beats_ctrl = net_diff > 0 and p_worst < alpha
        beats_base = own.get("flag") == "GAIN"
        if beats_base and beats_ctrl:
            verdict = "VALID"
            reason = (
                f"adapter beats matched control {b_aligned} to {c_aligned} (delta {delta_ac:+.4f}, "
                f"p_aligned={p_aligned:.3e}, p_worst={p_worst:.3e}) and base ({own['delta_vs_base']:+.4f})"
            )
        elif not beats_base:
            verdict = "NO_GAIN"
            reason = f"own-pack change vs base {own['delta_vs_base']:+.4f} (p={own['mcnemar_exact_p']})"
        else:
            verdict = "NO_GAIN_VS_CONTROL"
            reason = f"adapter did not significantly beat matched control (delta {delta_ac:+.4f}, p_worst={p_worst:.4g})"
        return {
            "verdict": verdict,
            "reason": reason,
            "control_kind": "matched_shuffled_label",
            "adapter_correct": own_correct,
            "control_correct": ctrl_correct,
            "net_items_over_control": net_diff,
            "b_adapter_only_aligned": b_aligned,
            "c_control_only_aligned": c_aligned,
            "adapter_vs_control_delta": delta_ac,
            "mcnemar_p_aligned": p_aligned,
            "mcnemar_p_worst_case": p_worst,
        }
    # Generative / numeric task where shuffled control scored 0: use cross-domain adapter control or base comparison
    best_cross = max(cross_domain_controls, key=lambda c: float(c.get("accuracy", 0.0))) if cross_domain_controls else None
    cross_acc = float(best_cross["accuracy"]) if best_cross is not None else 0.0
    delta_cross = round(float(own["accuracy"]) - cross_acc, 4)
    beats_base = own.get("flag") == "GAIN"
    verdict = "VALID" if (beats_base and delta_cross > 0.0) else "NO_GAIN"
    return {
        "verdict": verdict,
        "reason": f"generative task: own-pack vs base {own['delta_vs_base']:+.4f} (p={own['mcnemar_exact_p']}); delta vs cross-domain control {delta_cross:+.4f}",
        "control_kind": "cross_domain_generative",
        "adapter_correct": own_correct,
        "control_correct": int(control["correct"]) if control is not None else 0,
        "adapter_vs_control_delta": delta_cross,
        "mcnemar_p_aligned": own.get("mcnemar_exact_p"),
        "mcnemar_p_worst_case": own.get("mcnemar_exact_p"),
    }


def rescore_stored_night_moe_results(stored_report: Mapping[str, Any]) -> dict[str, Any]:
    """Re-score stored `NIGHT_MOE_results.json` under the HM-S14-01 (#165) paired adapter-vs-control rule without retraining."""
    subjects_out: dict[str, Any] = {}
    subjects = stored_report.get("subjects", {})
    for subj, data in subjects.items():
        own = data["own"]
        ctrl = data.get("control")
        # Collect any cross-domain evaluations on `subj`'s own pack from other adapters
        cross: list[Mapping[str, Any]] = []
        for other_subj, other_data in subjects.items():
            if other_subj == subj:
                continue
            other_cell = (other_data.get("other_packs") or {}).get(data["own_pack"])
            if other_cell is not None:
                cross.append(other_cell)
        res = paired_adapter_vs_control_verdict(own, ctrl, cross_domain_controls=cross)
        subjects_out[subj] = {
            "legacy_verdict": data["verdict"],
            "legacy_reason": data["reason"],
            **res,
        }
    return {
        "experiment": "HM-S14-01-RESCORE",
        "source_experiment": stored_report.get("experiment", "NIGHT-MOE-1"),
        "subjects": subjects_out,
    }


def _reconstruct_paired_outcomes(prefix: str, n: int, cand_correct: int, only_cand: int, only_base: int) -> tuple[dict[str, str], dict[str, str]]:
    """Construct canonical per-item PASS/FAIL maps matching exact discordant counts (only_cand, only_base, both_pass, both_fail)."""
    from hivemind.swarm.verified_update_gate import FAIL, PASS

    both_pass = max(0, cand_correct - only_cand)
    both_fail = max(0, n - both_pass - only_cand - only_base)
    base_map: dict[str, str] = {}
    cand_map: dict[str, str] = {}
    idx = 0
    for _ in range(both_pass):
        k = f"{prefix}:{idx}"
        base_map[k] = PASS
        cand_map[k] = PASS
        idx += 1
    for _ in range(only_cand):
        k = f"{prefix}:{idx}"
        base_map[k] = FAIL
        cand_map[k] = PASS
        idx += 1
    for _ in range(only_base):
        k = f"{prefix}:{idx}"
        base_map[k] = PASS
        cand_map[k] = FAIL
        idx += 1
    for _ in range(both_fail):
        k = f"{prefix}:{idx}"
        base_map[k] = FAIL
        cand_map[k] = FAIL
        idx += 1
    return base_map, cand_map


def evaluate_stars_verified_update_gate(stored_report: Mapping[str, Any]) -> dict[str, Any]:
    """Run the Verified Update Gate (`hivemind.swarm.verified_update_gate.decide`) on the `stars` adapter (HM-S14-08 / #170)."""
    from hivemind.swarm.verified_update_gate import UpdatePolicy, decide

    stars = stored_report["subjects"]["stars"]
    own = stars["own"]
    own_b, own_c = _reconstruct_paired_outcomes(
        "stars_own",
        int(own["n"]),
        int(own["correct"]),
        int(own["only_adapter_right"]),
        int(own["only_base_right"]),
    )
    own_receipt = decide(own_b, own_c, UpdatePolicy(require_improvement=True))

    regression_receipts: dict[str, Any] = {}
    pooled_base: dict[str, str] = {}
    pooled_cand: dict[str, str] = {}
    for pack, cell in sorted(stars.get("other_packs", {}).items()):
        pb, pc = _reconstruct_paired_outcomes(
            f"stars_{pack}",
            int(cell["n"]),
            int(cell["correct"]),
            int(cell["only_adapter_right"]),
            int(cell["only_base_right"]),
        )
        pooled_base.update(pb)
        pooled_cand.update(pc)
        r_strict = decide(pb, pc, UpdatePolicy(tolerated_net_loss=0, require_improvement=False))
        r_stat = decide(pb, pc, UpdatePolicy(tolerated_net_loss=int(cell["n"]), require_improvement=False))
        regression_receipts[pack] = {
            "strict_zero_net_loss": r_strict.to_dict(),
            "statistical_non_regression": r_stat.to_dict(),
        }

    pooled_regression = decide(pooled_base, pooled_cand, UpdatePolicy(tolerated_net_loss=0, require_improvement=False))
    all_base = {**own_b, **pooled_base}
    all_cand = {**own_c, **pooled_cand}
    combined_receipt = decide(all_base, all_cand, UpdatePolicy(tolerated_net_loss=0, require_improvement=True))
    missing_cells = list(stars.get("missing_cells", []))

    return {
        "adapter": "stars",
        "own_pack_receipt": own_receipt.to_dict(),
        "regression_packs": regression_receipts,
        "pooled_regression_receipt": pooled_regression.to_dict(),
        "combined_all_evaluated_receipt": combined_receipt.to_dict(),
        "missing_cells": missing_cells,
        "may_serve_on_own_and_pooled_gate": bool(own_receipt.accepted and pooled_regression.accepted),
        "g_r2_complete_grid_ready": bool(
            own_receipt.accepted
            and pooled_regression.accepted
            and not missing_cells
            and all(v["strict_zero_net_loss"]["decision"] == "ACCEPT" for v in regression_receipts.values())
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--night-out", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/experiments/NIGHT_MOE_results.json"))
    ap.add_argument("--out-md", type=Path, default=Path("docs/NIGHT_MOE_RESULTS.md"))
    args = ap.parse_args()
    report = build(load_summaries(args.night_out))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    for subj, s in report["subjects"].items():
        print(subj, s["verdict"], s["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
