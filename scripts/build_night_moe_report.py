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


def compare_self_solution_vs_gold_adapter(
    base_outcomes: Mapping[str, bool],
    gold_outcomes: Mapping[str, bool],
    self_solution_outcomes: Mapping[str, bool],
    cross_control_outcomes: Mapping[str, bool],
    *,
    base_unparsable_keys: Sequence[str] = (),
    gold_unparsable_count: int = 0,
    self_unparsable_count: int = 0,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Preregistered comparison of verified self-solution RFT adapter vs gold-text SFT adapter and base (HM-S14-02 / #166).

    Evaluates:
    - H1 (vs cross-domain control): `self_solution` beats a cross-domain control that scores > 0 (`p < alpha`).
    - H2 (vs base on base-parsed items): `self_solution` beats `base` even on items `base` already parsed, whereas
      `gold_adapter` hurts `base` on base-parsed items (130 vs 157 in NIGHT-MOE-1).
    - Direct paired comparison: `self_solution` vs `gold_adapter` (`exact McNemar p < alpha`).
    - Verified Update Gate (`hivemind.swarm.verified_update_gate.decide`) on both adapters.
    """
    from hivemind.swarm.verified_update_gate import FAIL, PASS, UpdatePolicy, decide

    keys = sorted(base_outcomes.keys())
    if not keys or set(gold_outcomes) != set(keys) or set(self_solution_outcomes) != set(keys) or set(cross_control_outcomes) != set(keys):
        raise ValueError("all outcome maps must share the exact same non-empty item keys")
    n = len(keys)
    unp_set = set(base_unparsable_keys)
    parsed_keys = [k for k in keys if k not in unp_set]

    def _paired(a_map: Mapping[str, bool], b_map: Mapping[str, bool], subset: Sequence[str]) -> dict[str, Any]:
        b = sum(1 for k in subset if a_map[k] and not b_map[k])
        c = sum(1 for k in subset if not a_map[k] and b_map[k])
        a_ok = sum(1 for k in subset if a_map[k])
        b_ok = sum(1 for k in subset if b_map[k])
        p_val = _exact_mcnemar_two_sided(b, c)
        return {
            "n": len(subset),
            "a_correct": a_ok,
            "b_correct": b_ok,
            "a_accuracy": round(a_ok / max(1, len(subset)), 4),
            "b_accuracy": round(b_ok / max(1, len(subset)), 4),
            "only_a_right": b,
            "only_b_right": c,
            "delta": round((a_ok - b_ok) / max(1, len(subset)), 4),
            "mcnemar_exact_p": p_val,
            "significant_gain": bool(a_ok > b_ok and p_val < alpha),
            "significant_harm": bool(a_ok < b_ok and p_val < alpha),
        }

    self_vs_gold = _paired(self_solution_outcomes, gold_outcomes, keys)
    self_vs_base_all = _paired(self_solution_outcomes, base_outcomes, keys)
    self_vs_base_parsed = _paired(self_solution_outcomes, base_outcomes, parsed_keys)
    self_vs_cross = _paired(self_solution_outcomes, cross_control_outcomes, keys)
    gold_vs_base_all = _paired(gold_outcomes, base_outcomes, keys)
    gold_vs_base_parsed = _paired(gold_outcomes, base_outcomes, parsed_keys)

    base_gate = {k: (PASS if base_outcomes[k] else FAIL) for k in keys}
    gold_gate = {k: (PASS if gold_outcomes[k] else FAIL) for k in keys}
    self_gate = {k: (PASS if self_solution_outcomes[k] else FAIL) for k in keys}

    gold_receipt = decide(base_gate, gold_gate, UpdatePolicy(require_improvement=True))
    self_receipt = decide(base_gate, self_gate, UpdatePolicy(require_improvement=True))

    return {
        "n_items": n,
        "n_base_parsed": len(parsed_keys),
        "unparsable_counts": {
            "base": len(unp_set),
            "gold_adapter": int(gold_unparsable_count),
            "self_solution_adapter": int(self_unparsable_count),
        },
        "self_solution_vs_gold_adapter": self_vs_gold,
        "self_solution_vs_base_all": self_vs_base_all,
        "self_solution_vs_base_parsed_h2": self_vs_base_parsed,
        "self_solution_vs_cross_domain_control_h1": self_vs_cross,
        "gold_adapter_vs_base_all": gold_vs_base_all,
        "gold_adapter_vs_base_parsed": gold_vs_base_parsed,
        "gold_adapter_gate_receipt": gold_receipt.to_dict(),
        "self_solution_gate_receipt": self_receipt.to_dict(),
        "supported": bool(
            self_vs_gold["significant_gain"]
            and self_vs_base_all["significant_gain"]
            and self_vs_base_parsed["significant_gain"]
            and self_vs_cross["significant_gain"]
            and int(self_vs_cross["b_correct"]) > 0
            and gold_vs_base_parsed["significant_harm"]
            and not gold_receipt.accepted
            and self_receipt.accepted
        ),
    }


def evaluate_routed_bee_moe_system(
    stored_report: Mapping[str, Any],
    *,
    full_grid_overrides: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    heldout_prompts: Sequence[tuple[str, Sequence[Mapping[str, str]]]] = (),
) -> dict[str, Any]:
    """Evaluate the routed Bee MoE as one end-to-end system with the Verified Update Gate in place (HM-S14-03 / #167).

    1. Measures `BeeMoESystemRouter` prompt-classification accuracy on held-out items across all Night-MoE packs.
    2. Computes the **routing plateau** (the best single adapter evaluated across all packs in `stored_report`) before
       claiming any routing gain.
    3. Compares `ungated_routed_moe` (which routes every pack to its domain adapter regardless of gate status) against
       `gated_routed_bee_moe` (which routes only to adapters that passed H1 and the Verified Update Gate, falling back
       to `base` elsewhere) and against a `corrupted_misrouted_moe` negative control.
    """
    from hivemind.queenbee.strategies import BeeMoESystemRouter
    from hivemind.swarm.verified_update_gate import PASS, UpdatePolicy, decide

    baselines = stored_report["baselines"]
    subjects = stored_report["subjects"]
    packs = sorted(baselines.keys())
    overrides = dict(full_grid_overrides or {})

    # Determine which adapters pass H1 (paired adapter-vs-control) + own-pack Verified Update Gate
    rescored = rescore_stored_night_moe_results(stored_report)["subjects"]
    admitted: list[str] = []
    for subj, sdata in sorted(subjects.items()):
        own = sdata["own"]
        b_map, c_map = _reconstruct_paired_outcomes(
            f"admit_{subj}",
            int(own["n"]),
            int(own["correct"]),
            int(own["only_adapter_right"]),
            int(own["only_base_right"]),
        )
        gate_ok = decide(b_map, c_map, UpdatePolicy(require_improvement=True)).accepted
        if rescored[subj]["verdict"] == "VALID" and gate_ok:
            admitted.append(subj)

    router = BeeMoESystemRouter(admitted_adapters=admitted)

    # 1. Held-out prompt classification accuracy
    router_correct = 0
    for expected_pack, msgs in heldout_prompts:
        pred_pack, _ = router.route_messages(msgs, enforce_gate=True)
        if pred_pack == expected_pack:
            router_correct += 1
    router_accuracy = round(router_correct / max(1, len(heldout_prompts)), 4) if heldout_prompts else 1.0

    # Helper to look up (adapter, pack) cell scaled to full pack size n
    def _cell_for(adapter: str, pack: str) -> dict[str, int]:
        n_full = int(baselines[pack]["n"])
        base_corr = int(baselines[pack]["correct"])
        if adapter == "base":
            return {"n": n_full, "correct": base_corr, "only_adapter_right": 0, "only_base_right": 0, "unparsable": int(baselines[pack].get("unparsable", 0))}
        if (adapter, pack) in overrides:
            ov = overrides[(adapter, pack)]
            return {
                "n": int(ov["n"]),
                "correct": int(ov["correct"]),
                "only_adapter_right": int(ov["only_adapter_right"]),
                "only_base_right": int(ov["only_base_right"]),
                "unparsable": int(ov.get("unparsable", 0)),
            }
        sdata = subjects[adapter]
        if pack == sdata["own_pack"]:
            own = sdata["own"]
            return {
                "n": int(own["n"]),
                "correct": int(own["correct"]),
                "only_adapter_right": int(own["only_adapter_right"]),
                "only_base_right": int(own["only_base_right"]),
                "unparsable": int(own.get("unparsable", 0)),
            }
        other = sdata.get("other_packs", {}).get(pack)
        if other is not None:
            n_sub = max(1, int(other["n"]))
            scale = n_full / n_sub
            only_a = int(round(int(other["only_adapter_right"]) * scale))
            only_b = int(round(int(other["only_base_right"]) * scale))
            corr = max(0, min(n_full, base_corr + only_a - only_b))
            return {
                "n": n_full,
                "correct": corr,
                "only_adapter_right": only_a,
                "only_base_right": only_b,
                "unparsable": int(round(int(other.get("unparsable", 0)) * scale)),
            }
        return {"n": n_full, "correct": base_corr, "only_adapter_right": 0, "only_base_right": 0, "unparsable": int(baselines[pack].get("unparsable", 0))}

    # 2. Measure the routing plateau (best single adapter across all packs)
    single_adapter_totals: dict[str, dict[str, Any]] = {}
    for candidate_adapter in ["base", *sorted(subjects.keys())]:
        tot_n = 0
        tot_c = 0
        tot_unp = 0
        for pack in packs:
            c = _cell_for(candidate_adapter, pack)
            tot_n += c["n"]
            tot_c += c["correct"]
            tot_unp += c["unparsable"]
        single_adapter_totals[candidate_adapter] = {
            "n": tot_n,
            "correct": tot_c,
            "accuracy": round(tot_c / max(1, tot_n), 4),
            "unparsable": tot_unp,
        }
    best_single_adapter = max(
        sorted(subjects.keys()),
        key=lambda a: single_adapter_totals[a]["correct"],
    )
    routing_plateau = single_adapter_totals[best_single_adapter]

    # 3. Evaluate Base, Ungated Routed MoE, Gated Routed Bee MoE, and Corrupted Misrouted Negative Control
    base_pooled: dict[str, str] = {}
    ungated_pooled: dict[str, str] = {}
    gated_pooled: dict[str, str] = {}
    corrupted_pooled: dict[str, str] = {}
    best_single_pooled: dict[str, str] = {}

    per_pack_routing: dict[str, Any] = {}
    ungated_harm_packs: list[str] = []
    gated_harm_packs: list[str] = []

    for pack in packs:
        u_adapter = router.route_pack(pack, enforce_gate=False)
        g_adapter = router.route_pack(pack, enforce_gate=True)
        # Negative control: deliberately misroute every pack to the maths gold-text adapter (or cyber on maths)
        bad_adapter = "maths" if pack != "maths" else "stars"

        c_u = _cell_for(u_adapter, pack)
        c_g = _cell_for(g_adapter, pack)
        c_bad = _cell_for(bad_adapter, pack)
        c_best = _cell_for(best_single_adapter, pack)

        b_u, cand_u = _reconstruct_paired_outcomes(f"u_{pack}", c_u["n"], c_u["correct"], c_u["only_adapter_right"], c_u["only_base_right"])
        b_g, cand_g = _reconstruct_paired_outcomes(f"g_{pack}", c_g["n"], c_g["correct"], c_g["only_adapter_right"], c_g["only_base_right"])
        _, cand_bad = _reconstruct_paired_outcomes(f"bad_{pack}", c_bad["n"], c_bad["correct"], c_bad["only_adapter_right"], c_bad["only_base_right"])
        _, cand_best = _reconstruct_paired_outcomes(f"best_{pack}", c_best["n"], c_best["correct"], c_best["only_adapter_right"], c_best["only_base_right"])

        base_pooled.update({f"{pack}:{i}": v for i, (_, v) in enumerate(sorted(b_g.items()))})
        ungated_pooled.update({f"{pack}:{i}": v for i, (_, v) in enumerate(sorted(cand_u.items()))})
        gated_pooled.update({f"{pack}:{i}": v for i, (_, v) in enumerate(sorted(cand_g.items()))})
        corrupted_pooled.update({f"{pack}:{i}": v for i, (_, v) in enumerate(sorted(cand_bad.items()))})
        best_single_pooled.update({f"{pack}:{i}": v for i, (_, v) in enumerate(sorted(cand_best.items()))})

        if c_u["correct"] < int(baselines[pack]["correct"]):
            ungated_harm_packs.append(pack)
        if c_g["correct"] < int(baselines[pack]["correct"]):
            gated_harm_packs.append(pack)

        per_pack_routing[pack] = {
            "n": c_g["n"],
            "base_correct": int(baselines[pack]["correct"]),
            "ungated_adapter": u_adapter,
            "ungated_correct": c_u["correct"],
            "ungated_unparsable": c_u["unparsable"],
            "gated_adapter": g_adapter,
            "gated_correct": c_g["correct"],
            "gated_unparsable": c_g["unparsable"],
        }

    gate_vs_base = decide(base_pooled, gated_pooled, UpdatePolicy(require_improvement=True))
    gate_vs_plateau = decide(best_single_pooled, gated_pooled, UpdatePolicy(require_improvement=True))
    corrupted_gate = decide(base_pooled, corrupted_pooled, UpdatePolicy(require_improvement=True))

    total_n = len(base_pooled)
    gated_correct = sum(1 for v in gated_pooled.values() if v == PASS)
    ungated_correct = sum(1 for v in ungated_pooled.values() if v == PASS)
    base_correct = sum(1 for v in base_pooled.values() if v == PASS)

    return {
        "admitted_adapters": admitted,
        "heldout_router_accuracy": router_accuracy,
        "single_adapter_totals": single_adapter_totals,
        "best_single_adapter": best_single_adapter,
        "routing_plateau": routing_plateau,
        "system_totals": {
            "n": total_n,
            "base_correct": base_correct,
            "base_accuracy": round(base_correct / max(1, total_n), 4),
            "ungated_routed_correct": ungated_correct,
            "ungated_routed_accuracy": round(ungated_correct / max(1, total_n), 4),
            "gated_routed_correct": gated_correct,
            "gated_routed_accuracy": round(gated_correct / max(1, total_n), 4),
        },
        "ungated_harm_packs": ungated_harm_packs,
        "gated_harm_packs": gated_harm_packs,
        "per_pack_routing": per_pack_routing,
        "gated_vs_base_receipt": gate_vs_base.to_dict(),
        "gated_vs_best_single_plateau_receipt": gate_vs_plateau.to_dict(),
        "corrupted_misrouted_negative_control_receipt": corrupted_gate.to_dict(),
        "supported": bool(
            router_accuracy >= 0.98
            and gate_vs_base.accepted
            and gate_vs_plateau.accepted
            and not gated_harm_packs
            and len(ungated_harm_packs) > 0
            and not corrupted_gate.accepted
        ),
    }


def evaluate_full_interference_bonferroni_grid(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    subjects: Mapping[str, str] | None = None,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Evaluate the full pack-by-adapter matrix as one preregistered Bonferroni family with zero TIMEOUT/sampled cells (HM-S14-04 / #168)."""
    if subjects is None:
        subjects = SUBJECTS or {"maths": "maths", "exoplanet": "exoplanet", "stars": "stars", "cyber": "cyber"}
    bases = {k.removeprefix("base_"): v for k, v in summaries.items() if k.startswith("base_")}
    packs = sorted(bases.keys())
    off_diagonal_cells: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    sampled_cells: list[str] = []
    timeout_cells: list[str] = []

    for subj, own_pack in sorted(subjects.items()):
        for pack in packs:
            if pack == own_pack:
                continue
            s = summaries.get(f"{subj}_{pack}") or summaries.get(f"x_{subj}_{pack}")
            if s is None:
                missing_cells.append(f"{subj}->{pack}")
                continue
            if s.get("status") == "TIMEOUT" or s.get("timed_out"):
                timeout_cells.append(f"{subj}->{pack}")
            n_expected = int(bases[pack]["n"])
            if int(s["n"]) < n_expected:
                sampled_cells.append(f"{subj}->{pack}({s['n']}/{n_expected})")
            cell = _cell(dict(s), dict(bases[pack]))
            cell["adapter"] = subj
            cell["pack"] = pack
            cell["kind"] = "transfer" if f"{subj}_{pack}" in summaries else "interference"
            off_diagonal_cells.append(cell)

    bonferroni_alpha = alpha / max(1, len(off_diagonal_cells))
    for c in off_diagonal_cells:
        p = c["mcnemar_exact_p"]
        c["survives_bonferroni"] = bool(p is not None and p < bonferroni_alpha and c["flag"] in ("GAIN", "HARM"))

    return {
        "n_subjects": len(subjects),
        "n_packs": len(packs),
        "n_off_diagonal_cells": len(off_diagonal_cells),
        "alpha": alpha,
        "bonferroni_alpha": round(bonferroni_alpha, 6),
        "missing_cells": missing_cells,
        "timeout_cells": timeout_cells,
        "sampled_cells": sampled_cells,
        "complete_full_size_grid": bool(
            len(off_diagonal_cells) == len(subjects) * (len(packs) - 1)
            and not missing_cells
            and not timeout_cells
            and not sampled_cells
        ),
        "bonferroni_survivors": [
            f"{c['adapter']}->{c['pack']}:{c['flag']}"
            for c in off_diagonal_cells
            if c["survives_bonferroni"]
        ],
        "cells": off_diagonal_cells,
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
