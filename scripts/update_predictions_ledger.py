"""Write ``artifacts/predictions_ledger.json`` from the experiment artifacts that exist.

The ledger lists every prediction of ``docs/RESEARCH_PROGRAM.md`` with its status
(``untested`` / ``supported`` / ``refuted`` / ``inconclusive`` / ``inapplicable``; E13-E16 add ``mixed`` = decided per model / tier and
the per-group statuses disagree, and ``descriptive`` = registered without a status), the artifact
and receipt that decided it, and the scale at which it was decided. Statuses are copied from
the artifacts, never typed here.

Run: ``PYTHONPATH=src python scripts/update_predictions_ledger.py``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import time
from typing import Any, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = PROJECT_ROOT / "artifacts" / "predictions_ledger.json"
E5_TOY = PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_report.json"
E5_HET = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_report.json"
E10 = PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_report.json"
E12_PILOT = PROJECT_ROOT / "artifacts" / "experiments" / "E12_pilot_report.json"

PREDICTIONS: Dict[str, Dict[str, str]] = {
    "P1": {"rq": "RQ1", "claim": "WAN loss with K<M quorum, grace and RDA <= single-host loss with M(1-churn) workers", "decided_by": "E8"},
    "P2": {"rq": "RQ2", "claim": "best outer LR grows with M under direct averaging and is constant under RDA (conditional on near-orthogonal learner deltas)", "decided_by": "E5"},
    "P3": {"rq": "RQ3", "claim": "adaptive grace admission beats matched-random admission at equal loss on wall-clock", "decided_by": "E5"},
    "P4": {"rq": "RQ4", "claim": "PA-CGAD gate helps under quorum with stale deltas and never hurts; zero effect without stale deltas", "decided_by": "E5"},
    "P5": {"rq": "RQ5", "claim": "Skip-Token quality loss is governed by the measured ghost fraction", "decided_by": "E7"},
    "P6": {"rq": "RQ6", "claim": "whitening inner optimizer flattens loss-vs-M slope", "decided_by": "E6"},
    "P7": {"rq": "RQ7", "claim": "learned depth prior makes terminal-KV sharing safe", "decided_by": "E7b"},
    "P8": {"rq": "RQ8", "claim": "Multi-Krum + 2sigma does not stop ALIE at f/n >= 0.2; spot-check replay does", "decided_by": "E9"},
    "P9": {"rq": "RQ2'", "claim": "RDA acts as an outer-LR multiplier: log(best_lr_direct / best_lr_rda) within one grid step of log(mean ||RDA||/||direct||)", "decided_by": "E5-het"},
    "P10a": {"rq": "RQ9", "claim": "E8 blockwise quantisation of outer deltas has lower distortion than Z8 at matched measured rate", "decided_by": "E10-lattice"},
    "P10b": {"rq": "RQ9", "claim": "E8 reaches a lower held-out loss than Z8 at matched rate (headline: 1 bit/dim with error feedback)", "decided_by": "E10-lattice"},
    "P10c": {"rq": "RQ9", "claim": "error feedback recovers the unquantised loss at 1 bit/dim with E8", "decided_by": "E10-lattice"},
    "P11a": {"rq": "RQ10", "claim": "A2 per RoPE plane lowers KV-cache dNLL vs scalar quantisation at matched measured bits", "decided_by": "E11-kv-lattice"},
    "P11b": {"rq": "RQ10", "claim": "E8 lowers KV-cache dNLL vs scalar only after a per-head KLT", "decided_by": "E11-kv-lattice"},
    "P12-same": {"rq": "RQ11", "claim": "same-class spot-check replay divergence <= eps = 1e-4", "decided_by": "E12-replay-divergence"},
    "P12-cross": {"rq": "RQ11", "claim": "cross-class spot-check replay divergence > eps = 1e-4", "decided_by": "E12-replay-divergence"},
    "P12-fixed-point": {"rq": "RQ11", "claim": "fixed-point replay of the checked segment is exact across classes", "decided_by": "E12 (needs a fixed-point kernel)"},
    # E13-E16 (2026-10-24): registered in artifacts/experiments/E1[3-6]_*prereg*.json; E14 is exploratory and E13d a replication
    "P13a": {"rq": "E13", "claim": "protocol correctness: the final global LoRA of S1 (swarm identity) equals S0 within max|diff| <= 1e-6", "decided_by": "E13"},
    "P13b": {"rq": "E13", "claim": "swarm viability: the all-on swarm (K = ceil(M/2), persistent straggler, churn) keeps >= 0.5 of the synchronous improvement (lower end of the df = 2 interval)", "decided_by": "E13"},
    "P13c": {"rq": "E13", "claim": "wire cost: measured upload bytes of S2 are <= 0.30 of the raw fp32 upload", "decided_by": "E13"},
    "P14a": {"rq": "E14 (exploratory)", "claim": "fair quorum (S6, bounded exclusion) keeps more of the synchronous improvement than the plain K<M swarm S2: df = 2 interval of loss(S2) - loss(S6) above 0", "decided_by": "E14"},
    "P14b": {"rq": "E14 (exploratory)", "claim": "deferred carry (S7) keeps more of the synchronous improvement than S2: df = 2 interval of loss(S2) - loss(S7) above 0", "decided_by": "E14"},
    "P15a": {"rq": "E15", "claim": "GWAYA-gated RFT improves greedy pass@1 on MBPP-sanitized: C2 - C0 >= +3.0 points with exact McNemar p < 0.05", "decided_by": "E15"},
    "P15b": {"rq": "E15", "claim": "the GWAYA inference loop improves the base model: C1 - C0 >= +5.0 points with McNemar p < 0.05", "decided_by": "E15"},
    "P15c": {"rq": "E15", "claim": "stacking (descriptive): C3 - C1 and C3 - C0 with McNemar p, no status", "decided_by": "E15"},
    "P15d": {"rq": "E15", "claim": "training and every inference stage run to completion with the 7.5 GiB cap enforced (peak reserved <= 8.0 GiB)", "decided_by": "E15"},
    "P15e": {"rq": "E15", "claim": "generalisation (descriptive): HumanEval C2 - C0 with McNemar p, no status", "decided_by": "E15"},
    "P16a": {"rq": "E16 (GWAYA PR #21 P1)", "claim": "the 7B Q4_K_M tier fits within 6656 MiB of GPU memory at num_ctx 4096 (and 8192)", "decided_by": "E16"},
    "P16b": {"rq": "E16 (GWAYA PR #21 P1)", "claim": "estimator adequacy: measured/estimated VRAM ratio within [0.80, 1.25] for every tier x ctx", "decided_by": "E16"},
    "P16c": {"rq": "E16 (GWAYA PR #21 P1)", "claim": "the GWAYA loop does not lower pass@1 (refuted if L - R < 0 with exact McNemar p < 0.05), per tier", "decided_by": "E16"},
    "P16d": {"rq": "E16", "claim": "the GWAYA loop improves pass@1 by >= +5.0 points with McNemar p < 0.05, per tier", "decided_by": "E16"},
    "P16e": {"rq": "E16 (GWAYA PR #21 P2)", "claim": "precision of VERIFIED verdicts against the hidden tests >= 0.80, per tier", "decided_by": "E16"},
    "P16f": {"rq": "E16", "claim": "decode tok/s of the 7B tier at num_ctx 4096 >= 20 (interactive-use bar, descriptive)", "decided_by": "E16"},
    "P12a": {"rq": "RQ11", "claim": "replaying the same 50-step LoRA segment in a second process on the same host and thread count reproduces the parameters bit for bit", "decided_by": "E12-pilot"},
    "P12b": {"rq": "RQ11", "claim": "changing only the CPU thread count gives a non-zero divergence that stays within eps = 1e-4", "decided_by": "E12-pilot"},
    "P12c": {"rq": "RQ11", "claim": "a different host (CPU, torch build, library versions) diverges by more than the thread-count cells and more than 1e-6; whether it stays within eps decides the matching rule", "decided_by": "E12-pilot"},
    "P12d": {"rq": "RQ11", "claim": "bf16 autocast (the consumer-GPU precision class) exceeds eps", "decided_by": "E12-pilot"},
}
E10B = PROJECT_ROOT / "artifacts" / "experiments" / "E10b_lattice_05b_report.json"
E11 = PROJECT_ROOT / "artifacts" / "experiments" / "E11_kv_lattice_report.json"
E12 = PROJECT_ROOT / "artifacts" / "experiments" / "E12_replay_divergence_report.json"
E13_PRED = PROJECT_ROOT / "artifacts" / "experiments" / "E13_swarm_local_predictions.json"
E13_PRED_M6 = PROJECT_ROOT / "artifacts" / "experiments" / "E13_swarm_local_predictions_m6.json"
E14_HYP = PROJECT_ROOT / "artifacts" / "experiments" / "E14_hypotheses.json"
E15_HYP = PROJECT_ROOT / "artifacts" / "experiments" / "E15_hypotheses.json"
E16_REPORT = PROJECT_ROOT / "artifacts" / "experiments" / "E16_gwaya_lite_t4_report.json"
SCALE_E13 = "Qwen2.5-0.5B (M = 4) and Granite-3.1-1B-A400M (M = 3) LoRA, peers over real localhost HTTP on one Tesla T4, 3 seeds; Granite-3B 1 seed (E13); no WAN claim"
SCALE_E14 = ("Qwen2.5-0.5B (4 workers, 4 domains) and Granite-3.1-1B-A400M MoE (3 workers, 3 domains), 6 rounds x 20 inner steps, thread-hosted peers over localhost HTTP on "
             "one Tesla T4, 3 seeds (E14, exploratory, no multiple-comparison correction); no WAN claim")
SCALE_E15 = ("Qwen2.5-Coder-1.5B-Instruct + rank-16 LoRA RFT on MBPP (train 462, eval n = 257) with HumanEval n = 164 descriptive, one training seed, Tesla T4 with a 7.5 GiB cap; "
             "gate = vendored GWAYA zero-stub auditor + namespace sandbox (not GWAYA's own oracle) (E15)")
SCALE_E16 = "Qwen2.5-Coder 1.5B / 3B / 7B-Instruct Q4_K_M served by Ollama 0.34.1 on a Tesla T4, MBPP-sanitized n = 257, num_ctx 4096 (VRAM also at 8192), same gate as E15 (E16)"


def _combine(per_group: Dict[str, str]) -> str:
    """One word for a prediction decided per model / tier / context: the common status when all agree, else 'mixed' (the detail is stored next to it)."""
    vals = set(per_group.values())
    return next(iter(vals)) if len(vals) == 1 else "mixed"


def _detail(per_group: Dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in per_group.items())


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _e13_entry(pid: str, pred: Dict[str, Any], m6: Dict[str, Any] | None) -> Dict[str, Any]:
    per_model = {m: v[pid]["status"] for m, v in pred["models"].items() if pid in v}
    decided = {m: s for m, s in per_model.items() if not s.startswith("descriptive")}
    ev = {m: {k: x for k, x in v[pid].items() if k != "rows"} for m, v in pred["models"].items() if pid in v}
    entry: Dict[str, Any] = {"status": _combine(decided) if decided else "untested", "status_detail": _detail(per_model), "per_model": per_model,
                             "scale": SCALE_E13, "artifact": "artifacts/experiments/E13_swarm_local_predictions.json", "receipt_sha256": pred["receipt_sha256"],
                             "prereg_sha256": pred["prereg_sha256"], "controls_all_passed": all(not v.get("controls_failed") and v.get("valid") for v in pred["models"].values()),
                             "evidence": ev,
                             "note": "decided per model by the registered rules (scripts/analyze_E13_predictions.py); a model with fewer than 3 seeds is descriptive and does not enter the headline; no WAN claim"}
    decisions = [{"experiment": f"E13-{m}", "status": s, "artifact": entry["artifact"], "receipt_sha256": pred["receipt_sha256"], "evidence": ev[m]} for m, s in per_model.items()]
    mm = ((m6 or {}).get("models") or {}).get("qwen0.5b", {})
    if pid in mm:
        decisions.append({"experiment": "E13d-qwen0.5b-M6-replication", "status": mm[pid]["status"], "scale": "Qwen2.5-0.5B, 6 workers on 6 Hugging Face domains, thread-hosted, 3 seeds (E13d, amendment 6: exploratory replication, changes no registered result)",
                          "artifact": "artifacts/experiments/E13_swarm_local_predictions_m6.json", "receipt_sha256": m6["receipt_sha256"], "evidence": {k: x for k, x in mm[pid].items() if k != "rows"}})
    entry["decisions"] = decisions
    return entry


def _e14_entry(pid: str, hyp: Dict[str, Any]) -> Dict[str, Any]:
    h = {"P14a": "H14a", "P14b": "H14b"}[pid]
    per_model = {m: v[h]["status"] for m, v in hyp["models"].items()}
    ev = {m: {k: v[h].get(k) for k in ("cell", "S2_minus_cell_per_seed", "mean", "half_width", "interval")} | {"retention_vs_S0": v["retention_vs_S0"]} for m, v in hyp["models"].items()}
    entry = {"status": _combine(per_model), "status_detail": _detail(per_model), "per_model": per_model, "scale": SCALE_E14,
             "artifact": "artifacts/experiments/E14_hypotheses.json", "receipt_sha256": hyp["receipt_sha256"], "prereg_sha256": hyp["prereg_sha256"],
             "controls_all_passed": all(v["controls"]["valid"] for v in hyp["models"].values()), "evidence": ev,
             "note": "exploratory follow-up of E13 (decides no registered E13 prediction); two hypotheses per model, no multiple-comparison correction; the report's own controls_all_passed is false only because the S1 identity and raw-wire controls are not evaluated in the E14 grid"}
    entry["decisions"] = [{"experiment": f"E14-{m}", "status": s, "scale": SCALE_E14, "artifact": f"artifacts/experiments/E14_report_{m}.json",
                           "receipt_sha256": hyp["sources"][m]["receipt_sha256"], "evidence": ev[m]} for m, s in per_model.items()]
    return entry


def _e15_entry(pid: str, hyp: Dict[str, Any]) -> Dict[str, Any]:
    h = {"P15a": "H15a", "P15b": "H15b", "P15c": "H15c_descriptive", "P15d": "H15d", "P15e": "H15e_descriptive"}[pid]
    row = hyp["hypotheses"][h]
    descriptive = pid in ("P15c", "P15e")
    return {"status": "descriptive" if descriptive else row["status"], "status_detail": row["status"], "scale": SCALE_E15,
            "artifact": "artifacts/experiments/E15_hypotheses.json", "receipt_sha256": hyp["receipt_sha256"], "prereg_sha256": hyp["prereg_sha256"],
            "controls_all_passed": hyp["hypotheses"]["valid"], "evidence": row,
            "source_report": hyp["source"],
            "note": "decided by the registered rules from artifacts/experiments/E15_gwaya_rft_report.json (scripts/analyze_E14_E15_hypotheses.py); one training seed, one model size"}


def _e16_entry(pid: str, rep: Dict[str, Any]) -> Dict[str, Any]:
    hyp = rep["analysis"]["H16"]
    key = pid.replace("P", "H")
    base = {"scale": SCALE_E16, "artifact": "artifacts/experiments/E16_gwaya_lite_t4_report.json", "receipt_sha256": rep["receipt_sha256"],
            "controls_all_passed": bool(rep.get("run_valid"))}
    if pid == "P16a":
        per = {f"num_ctx {k}": v["status"] for k, v in hyp["H16a"].items()}
        return {**base, "status": _combine(per), "status_detail": _detail(per), "per_ctx": per, "evidence": hyp["H16a"],
                "note": "threshold 6656 MiB is a Windows WDDM reserve; here it is tested as a threshold on Linux, not as a physical limit"}
    if pid == "P16b":
        b = hyp["H16b"]
        return {**base, "status": b["status"], "status_detail": b["status"], "evidence": b,
                "note": "the E16 pre-registration defines this prediction's outcomes as 'adequate' (all six ratios in the registered band) or a reported discrepancy; the report's word is stored verbatim"}
    if pid == "P16f":
        f = hyp["H16f_descriptive"]
        return {**base, "status": "descriptive", "status_detail": f"{f['status']} ({f['decode_tps_median']:.1f} tok/s vs bar {f['bar_tps']:.0f})", "evidence": f}
    per = dict(hyp[key])
    return {**base, "status": _combine(per), "status_detail": _detail(per), "per_tier": per, "evidence": {t: rep["analysis"]["tiers"][t].get(
        {"H16c": "paired_L_minus_R", "H16d": "paired_L_minus_R", "H16e": "p2_gate_calibration"}[key]) for t in per},
        "note": "decided per tier; the headline word is the common status when all tiers agree, else 'mixed' (see per_tier)"}


def _norm(status: str) -> str:
    s = status.split(";")[0].split(" ")[0].strip("(:")
    return s if s in ("supported", "refuted", "inconclusive", "invalid", "inapplicable") else "inconclusive" if s.startswith("mixed") else s


def _e10b_decision(rep: Dict[str, Any], pid: str) -> Dict[str, Any]:
    """E10b at 0.5B; the registered rule 'run_is_valid_iff all controls pass' is applied here even when the
    receipted report predates the runner applying it."""
    a = rep["analysis"][pid]
    computed = a.get("computed_status", a["status"])
    ok = rep["controls_all_passed"]
    failed = [k for k, v in rep["controls"].items() if isinstance(v, dict) and v.get("passed") is False]
    status = computed if ok else f"invalid (controls failed: {', '.join(failed)}); computed: {computed}"
    return {"experiment": "E10b-lattice-05b", "status": _norm(status), "status_detail": status,
            "scale": "Qwen2.5-0.5B rank-8 LoRA deltas (540,672 coords), M = 4, R = 6, H = 20, fineweb-edu, 3 seeds (E10b, amendment 2)",
            "artifact": "artifacts/experiments/E10b_lattice_05b_report.json", "receipt_sha256": rep["receipt_sha256"],
            "evidence": {k: v for k, v in a.items() if k not in ("cells",)} | ({"cells": a["cells"]} if "cells" in a else {})}


def _status_from_e5(report: Dict[str, Any], key: str) -> Dict[str, Any]:
    a = report["analysis"][key]
    status = a["status"]
    norm = "inapplicable" if status.startswith("inapplicable") else status.split(" ")[0]
    return {
        "status": norm,
        "status_detail": status,
        "scale": "toy synthetic LM, single host, 3 seeds (E5-toy)",
        "artifact": "artifacts/experiments/E5_toy_report.json",
        "receipt_sha256": report["receipt_sha256"],
        "prereg_sha256": report["prereg_sha256"],
        "controls_all_passed": report["controls_all_passed"],
        "evidence": {k: v for k, v in a.items() if k not in ("rows",)},
    }


def _status_from_e5_het(report: Dict[str, Any], key: str) -> Dict[str, Any]:
    a = report["analysis"][key]
    status = a["status"]
    norm = "inapplicable" if status.startswith("inapplicable") else status.split(" ")[0]
    return {
        "status": norm,
        "status_detail": status,
        "scale": "toy synthetic LM with per-learner teachers (rho = 4, delta cosine < 0.3), single host, 3 seeds (E5-het)",
        "artifact": "artifacts/experiments/E5_het_report.json",
        "receipt_sha256": report["receipt_sha256"],
        "prereg_sha256": report["prereg_sha256"],
        "controls_all_passed": report["controls_all_passed"],
        "manipulation_check": {k: v for k, v in report["manipulation_check"].items() if k != "rda_over_direct_norm_ratio"},
        "evidence": {k: v for k, v in a.items() if k not in ("rows",)},
    }


def main() -> int:
    ledger: Dict[str, Any] = {"schema_version": "hivemind-predictions-ledger/v1.1", "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "predictions": {}}
    e5 = json.loads(E5_TOY.read_text(encoding="utf-8")) if E5_TOY.exists() else None
    e5h = json.loads(E5_HET.read_text(encoding="utf-8")) if E5_HET.exists() else None
    e10 = json.loads(E10.read_text(encoding="utf-8")) if E10.exists() else None
    e12 = json.loads(E12_PILOT.read_text(encoding="utf-8")) if E12_PILOT.exists() else None
    for pid, meta in PREDICTIONS.items():
        entry: Dict[str, Any] = dict(meta)
        if e5 is not None and pid in ("P2", "P3", "P4"):
            entry.update(_status_from_e5(e5, pid))
            entry["note"] = "decided at toy scale only; the full-scale E5 cell on Qwen2.5-0.5B is still open"
            if e5h is not None:
                het = _status_from_e5_het(e5h, pid)
                entry["decisions"] = [{"experiment": "E5-toy", **{k: entry[k] for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "evidence")}}, {"experiment": "E5-het", **{k: het[k] for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "evidence")}}]
                # the heterogeneity-controlled run supersedes the IID one whenever it is applicable
                if het["status"] != "inapplicable":
                    for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "prereg_sha256", "controls_all_passed"):
                        entry[k] = het[k]
                    entry["evidence"] = het["evidence"]
                    entry["manipulation_check"] = het["manipulation_check"]
                    entry["note"] = "decided at toy scale (E5-het supersedes E5-toy); the full-scale E5 cell on Qwen2.5-0.5B is still open"
        elif e12 is not None and pid in ("P12a", "P12b", "P12c", "P12d"):
            a = e12["analysis"][pid]
            status = a["status"]
            entry.update({"status": "untested" if status.startswith("not run") else status.split(" ")[0], "status_detail": status,
                          "scale": "Qwen2.5-0.5B LoRA, one 50-step segment, CPU classes only (E12-pilot)", "artifact": "artifacts/experiments/E12_pilot_report.json",
                          "receipt_sha256": e12["receipt_sha256"], "prereg_sha256": e12["prereg_sha256"], "controls_all_passed": e12["controls_all_passed"],
                          "evidence": a, "note": "CPU pilot; GPU classes remain HM-S10-08"})
        elif e10 is not None and pid.startswith("P10"):
            a = e10["analysis"][pid]
            status = a["status"]
            entry.update({"status": status.split(";")[0].split(" ")[0], "status_detail": status,
                          "scale": "toy LoRA deltas (2048 coords) on the E5-het rho = 4, M = 8 testbed, 3 seeds (E10-lattice)",
                          "artifact": "artifacts/experiments/E10_lattice_report.json", "receipt_sha256": e10["receipt_sha256"], "prereg_sha256": e10["prereg_sha256"],
                          "controls_all_passed": e10["controls_all_passed"], "evidence": {k: v for k, v in a.items() if k != "cells"} | ({"cells": {ck: {kk: vv for kk, vv in cv.items() if kk != "per_seed"} for ck, cv in a["cells"].items()}} if "cells" in a else {}),
                          "note": "decided at toy scale only"})
            if E10B.exists():
                rep = json.loads(E10B.read_text(encoding="utf-8"))
                entry["decisions"] = [{"experiment": "E10-lattice", **{k: entry[k] for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256")}},
                                      _e10b_decision(rep, pid)]
                entry["note"] = "headline status from E10 (toy scale); the 0.5B run E10b is listed under 'decisions' and is invalid unless all its controls passed"
        elif pid.startswith("P11") and E11.exists():
            e11 = json.loads(E11.read_text(encoding="utf-8"))
            a = e11["analysis"][pid]
            # Review condition (scientific lead, 2026-10-24): the Qwen2.5-Coder-7B row is INVALID (negative control
            # negative_A2at8_equals_fp8 failed by -0.47 nats on a float16 model, so the FP8 arm is suspect); its verdicts are
            # withdrawn and never enter the status. The 0.5B P11a verdict rests on ONE rate-matched cell (R = 4).
            per_model = {m: ("withdrawn (negative control failed)" if m == "qwen7b" else v) for m, v in a["per_model"].items()}
            kept = {m: v for m, v in per_model.items() if not v.startswith("withdrawn")}
            if pid == "P11a":
                detail = ("refuted on qwen0.5b (one rate-matched cell, R = 4; R = 2 and 3 not rate-matched) and gemma4-e4b; "
                          "qwen7b withdrawn: negative control failed")
            else:
                detail = ("refuted on gemma4-e4b (E8 beats Z without the transform); qwen0.5b inconclusive; "
                          "qwen7b withdrawn: negative control failed")
            entry.update({"status": _norm(a["status"]), "status_detail": detail, "per_model": per_model,
                          "withdrawn": {"qwen7b": {"reason": "negative control negative_A2at8_equals_fp8 failed: -0.47 +- 0.20 nats (an 8-bit lattice cannot be half a nat better than FP8 unless the FP8 arm is broken on this float16 model)"}},
                          "scale": "real KV tensors of Qwen2.5-0.5B-Instruct and Gemma-4-E4B on WikiText-2, 1024-token windows, 3 seeds (E11, amendment 1); the Qwen2.5-Coder-7B row is INVALID and withdrawn",
                          "artifact": "artifacts/experiments/E11_kv_lattice_report.json", "receipt_sha256": e11["receipt_sha256"], "prereg_sha256": e11["prereg_sha256"],
                          "controls_all_passed": e11["controls_all_passed"],
                          "note": "receipt controls_all_passed is false; qwen7b (control off by -0.47 nats) is withdrawn; on qwen0.5b the same control reads -0.04 nats, explained in amendment 1 by FP8 e4m3 being coarser than an 8-bit lattice, and that failure is reported; only gemma4-e4b passed every control"})
        elif pid.startswith("P12") and E12.exists():
            e12g = json.loads(E12.read_text(encoding="utf-8"))
            a = e12g["analysis"][pid]
            entry.update({"status": "untested" if a["status"].startswith("not decided") else _norm(a["status"]), "status_detail": a["status"],
                          "scale": "Qwen2.5-0.5B rank-8 LoRA, 50-step segments, 9 segments per cell; Tesla T4 vs the same T4 (separate process) vs this host's x86 CPU (E12)",
                          "artifact": "artifacts/experiments/E12_replay_divergence_report.json", "receipt_sha256": e12g["receipt_sha256"], "prereg_sha256": e12g["prereg_sha256"],
                          "controls_all_passed": e12g["controls_all_passed"], "evidence": e12g["summary"], "basis": a.get("basis"),
                          "note": "RTX 2070, TPU-VM CPU and a second T4 were not run; the LoRA init is ~18x PEFT's default, which may inflate cross-class magnitudes"})
        elif e5h is not None and pid == "P9":
            entry.update(_status_from_e5_het(e5h, "P9"))
            entry["note"] = "decided at toy scale only"
        elif pid in ("P13a", "P13b", "P13c") and E13_PRED.exists():
            entry.update(_e13_entry(pid, _load(E13_PRED), _load(E13_PRED_M6) if E13_PRED_M6.exists() else None))
        elif pid in ("P14a", "P14b") and E14_HYP.exists():
            entry.update(_e14_entry(pid, _load(E14_HYP)))
        elif pid.startswith("P15") and E15_HYP.exists():
            entry.update(_e15_entry(pid, _load(E15_HYP)))
        elif pid.startswith("P16") and E16_REPORT.exists():
            entry.update(_e16_entry(pid, _load(E16_REPORT)))
        else:
            entry.update({"status": "untested", "scale": None, "artifact": None})
        ledger["predictions"][pid] = entry
    counts: Dict[str, int] = {}
    for e in ledger["predictions"].values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    ledger["status_counts"] = counts
    payload = json.dumps({k: v for k, v in ledger.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    ledger["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(json.dumps({"status_counts": counts, "receipt": ledger["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
