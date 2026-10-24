"""Benchmark report of the T4 campaign (2026-10-23/24), built only from committed artifacts.

Reads the queue logs and the E11 / E12 / E10b / E11b / E13 (incl. E13d) / E14 / E15 / E16 reports, the live-check receipt of the GWAYA-Lite
deployment and the measured table of docs/community/GWAYA_T4_DEPLOYMENT.md (any may be missing) and writes
``docs/benchmarks/T4_10h_campaign_report.md``. Every number in the report is copied from an artifact or computed by the registered rules
(``scripts/analyze_E14_E15_hypotheses.py``); the text adds no result. Run: python3 scripts/write_t4_benchmark_report.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
EXP = ROOT / "artifacts" / "experiments"
OUT = ROOT / "docs" / "benchmarks" / "T4_10h_campaign_report.md"


def load(name: str) -> Optional[Dict[str, Any]]:
    p = EXP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sha(name: str) -> str:
    p = EXP / name
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "-"


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if abs(x) < 100 else f"{x:.3g}"
    return str(x)


def pm(m: Optional[float], h: Optional[float], nd: int = 4) -> str:
    return "-" if m is None else f"{m:+.{nd}f} ± {h:.{nd}f}"


JOB_NOTES = {
    "E11_qwen0.5b": "crashed at the end (a code bug: KLT table is a dict); fixed and retried",
    "E11_qwen7b": "same crash; retried", "E11_gemma4-e4b": "same crash; retried",
    "E11_qwen0.5b_retry": "registered E11 run (amendment 1)", "E11_gemma4-e4b_retry": "registered E11 run", "E11_qwen7b_retry": "registered E11 run",
    "E11_aggregate": "report written; exit 1 = a negative control failed on the Qwen models (reported)",
    "E12_replay": "crashed in 20 s (wrong torch API path); fixed and retried", "E12_replay_retry": "registered E12 run",
    "E10b_lattice_05b": "first execution, stopped by the lead after one cell (not a registered rule)",
    "E10b_amended": "amendment-2 run; exit 1 = positive control failed, run INVALID by its registered rule",
    "E13_qwen0.5b": "registered E13 (Qwen2.5-0.5B, 18 cells); all controls passed",
    "E13_granite1b": "stopped by the registered time fail-fast (2.69 s/step with eager experts + checkpointing)",
    "E13_granite3b": "registered E13 feasibility run (Granite-3B, 1 seed)",
    "E13_granite_step_bench": "Granite step-time microbenchmark (decided amendment 4)",
    "E13_qwen0.5b_m2proc": "stopped by the learning gate (M=2 learns less); waiver registered (amendment 5)",
    "E13c_qwen0.5b_rotstraggle": "exploratory diagnostic; exit 1 only because the identity control is absent from its S0+S2 grid",
    "E13_qwen0.5b_m2proc_v2": "two real OS processes: S1 == S0 bit-identical; exit 1 only because the codec-rate control is n/a",
    "E13_granite1b_v2": "Granite-1B with grouped_mm (amendment 4); cut by the deadline, resumed in the extension queue",
    "E11b_qwen0.5b_fineweb-edu_w1024": "E11b exploratory", "E11b_qwen7b_fineweb-edu_w1024": "E11b exploratory", "E11b_gemma4-e4b_fineweb-edu_w1024": "E11b exploratory",
}


def queue_section(q: Optional[Dict[str, Any]]) -> List[str]:
    if not q:
        return ["Queue log not found."]
    start = dt.datetime.fromtimestamp(q["started_unix"], dt.timezone.utc)
    lines = [f"- Started {start:%Y-%m-%d %H:%M} UTC, budget {q['hours']} h, wall time {q.get('wall_hours', 'running')} h.",
             f"- GPU utilisation (NVML, 30 s samples, {q.get('gpu_samples')} samples): mean {q.get('gpu_util_mean_pct')} %, "
             f"peak memory {q.get('gpu_mem_peak_gib')} GiB.", "",
             "| job | status | wall (min) | mean GPU util % | what it was |", "| :-- | :-- | --: | --: | :-- |"]
    util = q.get("gpu_util_by_job", {})
    for j in q["jobs"]:
        mins = f"{j['seconds'] / 60:.1f}" if "seconds" in j else "-"
        lines.append(f"| `{j['name']}` | {j['status']} | {mins} | {util.get(j['name'], '-')} | {JOB_NOTES.get(j['name'], '')} |")
    gpu_h = sum(j.get("seconds", 0) for j in q["jobs"] if j.get("status") == "ok") / 3600
    lines += ["", f"Successful jobs used {gpu_h:.2f} GPU-hours. Failed attempts are listed and explained below; they are not hidden."]
    return lines


def e11_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E11 report not found."]
    a = r["analysis"]
    # Scientific lead's review (2026-10-24): the Qwen2.5-Coder-7B row is INVALID (its negative control failed by -0.47 nats on a
    # float16 model, so the FP8 arm is suspect); its verdicts are withdrawn. P11a on Qwen2.5-0.5B rests on ONE rate-matched cell.
    withdrawn = {"qwen7b"}
    q7 = r["models"].get("qwen7b", {}).get("controls", {}).get("negative_A2at8_equals_fp8", {})
    lines = ["**Decision (pre-registered, amendment 1), valid models only:** P11a **refuted on qwen0.5b (one rate-matched cell, R = 4; R = 2 and 3 are not rate-matched) "
             "and on gemma4-e4b**; P11b **refuted on gemma4-e4b** (inconclusive on qwen0.5b). "
             f"**The qwen7b row is INVALID (negative control `negative_A2at8_equals_fp8` failed: {pm(q7.get('diff_mean', float('nan')), q7.get('ci95', float('nan')))} nats) and its verdicts are withdrawn; no 7B number is quoted.** "
             f"Controls all passed: {r['controls_all_passed']}. Receipt `{r['receipt_sha256'][:16]}`.", "",
             "| model | dtype | peak GiB | baseline NLL | ΔNLL FP8 | ΔNLL INT4 | P11a | P11b | controls |",
             "| :-- | :-- | --: | --: | --: | --: | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if k in withdrawn:
            lines.append(f"| {k} | {v.get('dtype', '-')} | {v.get('peak_gpu_gib', '-')} | - | - | - | INVALID, withdrawn | INVALID, withdrawn | failed: negative_A2at8_equals_fp8 |")
            continue
        if "cells" not in v:
            lines.append(f"| {k} | - | - | - | - | - | {v.get('status')} | | |")
            continue
        c = v["cells"]
        failed = [n for n, x in v["controls"].items() if isinstance(x, dict) and x.get("passed") is False]
        lines.append(f"| {k} | {v['dtype']} | {v['peak_gpu_gib']} | {v['baseline_nll']:.3f} | {pm(c['fp8']['dnll_mean'], c['fp8']['dnll_ci95'])} | "
                     f"{pm(c['int@4']['dnll_mean'], c['int@4']['dnll_ci95'], 3)} | {v['P11a']['status']} | {v['P11b']['status']} | "
                     f"{'all pass' if not failed else 'failed: ' + ', '.join(failed)} |")
    lines += ["", "Paired differences at matched measured rate (ΔNLL(Z) − ΔNLL(lattice), nats/token, positive = lattice better):", "",
              "| model | R | Z − A₂ (none) | Z − E₈ (none) | Z − E₈ (KLT) |", "| :-- | --: | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if "cells" not in v or k in withdrawn:
            continue
        for R in (2, 3, 4):
            a2 = v["P11a"]["cells"][f"R={R}"]
            e0 = v["P11b"]["cells"][f"R={R}|none"]
            e1 = v["P11b"]["cells"][f"R={R}|klt"]
            cell = lambda d, key: f"{pm(d[key], d['ci95'])} ({d['status']})"  # noqa: E731
            lines.append(f"| {k} | {R} | {cell(a2, 'z_minus_a2_mean')} | {cell(e0, 'z_minus_e8_mean')} | {cell(e1, 'z_minus_e8_mean')} |")
    lines += ["", "Descriptive only (decides nothing): does a 4-bit codec reach FP8 (paired ΔNLL difference CI includes 0) at about half the bytes?", "",
              "| model | A₂@4 | E₈@4 + KLT | Z@4 | INT4 |", "| :-- | :-- | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if "descriptive_frontier_vs_fp8" in v and k not in withdrawn:
            fr = v["descriptive_frontier_vs_fp8"]
            lines.append(f"| {k} | " + " | ".join(f"{pm(fr[c]['minus_fp8_mean'], fr[c]['ci95'])}{' ✓' if fr[c]['reaches_fp8'] else ''}"
                                                   for c in ("A2@4|none", "E8@4|klt", "Z@4|none", "int@4")) + " |")
    return lines


def e12_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E12 report not found (not run or not finished)."]
    lines = [f"**Decision (pre-registered):** P12-same **{r['analysis']['P12-same']['status']}** ({r['analysis']['P12-same']['basis']}); "
             f"P12-cross **{r['analysis']['P12-cross']['status']}** ({r['analysis']['P12-cross']['basis']}); "
             f"fixed-point clause: {r['analysis']['P12-fixed-point']['status']}. Controls all passed: {r['controls_all_passed']}. "
             f"ε = {r['eps']}.", "", "| cell | max ‖Δθ‖∞ | min ‖Δθ‖∞ | segments above ε |", "| :-- | --: | --: | --: |"]
    for k, s in r["summary"].items():
        lines.append(f"| {k} | {s['max_linf']:.3e} | {s['min_linf']:.3e} | {s['n_above_eps']}/{s['n']} |")
    lines += ["", f"Not run on this host: {', '.join(r['not_run'])}."]
    w = {k: len(v) for k, v in r.get("determinism_warnings", {}).items()}
    lines.append(f"Determinism warnings captured per configuration (count): {w}.")
    return lines


def e10b_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E10b report not found (not run, not finished, or cut by the deadline; a partial file may exist)."]
    an = r.get("analysis", {})
    ok = r.get("controls_all_passed")
    failed = [k for k, v in r.get("controls", {}).items() if isinstance(v, dict) and v.get("passed") is False]

    def st(v: Dict[str, Any]) -> str:
        s = v.get("computed_status", v.get("status"))
        # the registered rule 'run_is_valid_iff all controls pass' (the receipted report predates the runner applying it)
        return s if ok else f"invalid (controls failed: {', '.join(failed)}); computed: {s}"

    lines = [f"**Decision (pre-registered, amendment 2):** " + "; ".join(f"{k} **{st(v)}**" for k, v in an.items() if isinstance(v, dict)) +
             f". Controls all passed: {ok}.", ""]
    p10a = an.get("P10a", {}).get("cells", {}).get("R=1.0|EF=on")
    if p10a:
        lines += [f"Distortion on real 0.5B LoRA deltas (descriptive whatever the validity): relMSE(E₈)/relMSE(ℤ⁸) = "
                  f"{p10a['ratio_mean']:.3f} ± {p10a['ci95']:.3f} per seed {[round(x, 4) for x in p10a['per_seed']]} "
                  f"(Gaussian prediction 0.86; E10 toy 0.81).", ""]
    cells = r.get("cells", {})
    if cells:
        lines += ["| arm | final held-out loss | rel. MSE | rate (bits/dim) |", "| :-- | :-- | --: | --: |"]
        for k, c in cells.items():
            lines.append(f"| {k} | {pm(c.get('final_loss_mean'), c.get('final_loss_ci95'))} | {f(c.get('mean_relative_mse'))} | {f(c.get('mean_rate_bits_per_dim'), 3)} |")
    return lines


def e11b_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E11b report not found (sweep not aggregated)."]
    lines = ["Exploratory: this sweep decides nothing. It checks whether E11's codec ranking holds on other Hugging Face data.", ""]
    entries = r.get("entries", r.get("configurations", {}))
    if isinstance(entries, dict):
        lines += ["| configuration | baseline NLL | ΔNLL FP8 | P11a-style | P11b-style | ranking agreement with E11 |", "| :-- | --: | --: | :-- | :-- | :-- |"]
        for k, v in entries.items():
            if k.startswith("qwen7b") or "_qwen7b" in k:
                lines.append(f"| {k} | INVALID model (E11 negative control failed): withdrawn | | | | |")
                continue
            dm = v.get("decide_model", v)
            if not isinstance(dm, dict) or "cells" not in dm:
                lines.append(f"| {k} | error: {str(v.get('error', '?'))[:60]} | | | | |")
                continue
            ag = v.get("ranking_agreement", {})
            tot = ag.get("totals", ag) if isinstance(ag, dict) else {}
            lines.append(f"| {k} | {dm['baseline_nll']:.3f} | {pm(dm['cells']['fp8']['dnll_mean'], dm['cells']['fp8']['dnll_ci95'])} | "
                         f"{dm['P11a']['status']} | {dm['P11b']['status']} | {json.dumps(tot)[:90]} |")
    return lines


def routing_table(model: str) -> List[str]:
    """Descriptive MoE routing statistics of the final global model (from the raw E13 file; only MoE models have them)."""
    p = EXP / f"E13_swarm_local_raw_{model}.json"
    if not p.exists():
        return []
    runs = [x for x in json.loads(p.read_text(encoding="utf-8")).get("runs", []) if x.get("routing", {}).get("available")]
    if not runs:
        return []
    by: Dict[Any, List[Any]] = {}
    for x in runs:
        for d, v in x["routing"]["per_domain"].items():
            by.setdefault((x["cell"], d), []).append((v["mean_norm_entropy"], v["mean_load_imbalance"], v["mean_top25pct_expert_share"]))
    avg = lambda vs, i: sum(z[i] for z in vs) / len(vs)  # noqa: E731
    lines = ["MoE routing of the final global model on each domain's held-out tokens (descriptive; mean over seeds; "
             "entropy = normalised expert-usage entropy per layer, imbalance = max/mean expert load, top-25% = share of routed assignments on the 25% most-used experts):", "",
             "| cell | domain | seeds | entropy | load imbalance | top-25% share |", "| :-- | :-- | --: | --: | --: | --: |"]
    for (c, d), v in sorted(by.items()):
        lines.append(f"| {c} | {d} | {len(v)} | {avg(v, 0):.3f} | {avg(v, 1):.2f} | {avg(v, 2):.3f} |")
    return lines + [""]


def e13_section(rep: Optional[Dict[str, Any]], pred: Optional[Dict[str, Any]], rot: Optional[Dict[str, Any]], m2: Optional[Dict[str, Any]]) -> List[str]:
    if not rep:
        return ["E13 report not found (not run or not finished)."]
    lines = ["Single-GPU swarm over real localhost HTTP (E8's localhost negative control): **no WAN claim**; latencies are GPU time-sharing artefacts. "
             "Peers are separate OS processes only where the memory preflight allowed it (see the hosting-fallback count).", ""]
    for name, m in rep["models"].items():
        pr = (pred or {}).get("models", {}).get(name, {})
        lines += [f"### {name}", "",
                  f"Controls all passed: {m['controls_all_passed']}; failed controls: {pr.get('controls_failed')}; cells failed: {m['failed_cells']}. "
                  f"Hosting fallback cells: {len(m.get('hosting_fallback_cells', []))}.", "",
                  "| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |", "| :-- | --: | :-- | --: | --: | :-- | --: |"]
        for k, c in m["cells"].items():
            w = c["wire"]
            ratio = "-" if not w.get("compression_ratio_actual") else f"{w['compression_ratio_actual']:.2f}x / {w['compression_ratio_ideal']:.2f}x"
            lines.append(f"| {k} {c['name']} | {len(c['seeds'])} | {', '.join(f'{x:.4f}' for x in c['final_loss_per_seed'])} | {c['admitted_fraction_mean']:.2f} | {c['late_discards_total']} | {ratio} | {c['round_latency_s']['p50']:.1f} |")
        lines.append("")
        b, cw = pr.get("P13b", {}), pr.get("P13c", {})
        lines.append(f"Registered predictions: P13a **{pr.get('P13a', {}).get('status')}**; P13b **{b.get('status')}**"
                     + (f" (retention {b['mean']:+.3f} ± {b['half_width']:.3f}, bar 0.5)" if "mean" in b else f" (per seed {b.get('per_seed')})")
                     + f"; P13c **{cw.get('status')}**" + (f" (upload {cw['upload_ratio_actual']:.4f} of raw, ideal {cw['upload_ratio_ideal']:.4f}, bar 0.30)" if "upload_ratio_actual" in cw else "") + ".")
        d = pr.get("P13d")
        if d:
            lines += ["", "Each mechanism off (descriptive; retention = share of the synchronous improvement kept):", "", "| cell | mean retention | admitted | late |", "| :-- | --: | --: | --: |"]
            for c, v in d.items():
                lines.append(f"| {c} {v['name']} | {v['mean_retention']:.3f} | {v['admitted_fraction_mean']:.2f} | {v['late_discards_total']} |")
        lines += routing_table(name)
        lines.append("")
    if rot:
        try:
            import analyze_E13_predictions as an  # noqa: PLC0415

            r = an.analyse_model(rot["models"]["qwen0.5b"])
            b = r.get("P13b", {})
            lines += ["### E13c — rotating straggler (exploratory diagnostic, decides nothing)", "",
                      "Same as S2 but the delayed worker rotates (3, 2, 1) instead of always being the GSM8K worker. Retention of the synchronous improvement: "
                      + (f"{b['mean']:+.3f} ± {b['half_width']:.3f}" if "mean" in b else str(b.get("per_seed"))) + f" (per seed {b.get('per_seed')}), "
                      "against the registered persistent-straggler value in the table above.", ""]
        except Exception as exc:  # an optional section must not break report generation
            lines += [f"E13c report present but not analysable: {exc!r}", ""]
    if m2:
        c = m2["models"]["qwen0.5b"]["controls"]["s1_equals_s0"]
        lines += ["### E13 with M=2 separate OS processes (identity control only)", "", f"S1 (2 OS processes over HTTP) equals its S0: passed = {c.get('passed')}, rows = {c.get('rows')}.", ""]
    return lines


EXT_NOTES = {
    "E13_granite1b_v2": "resumed (--resume) from the first queue's partial run: Granite-3.1-1B, cells S0 S1 S2, 3 seeds, grouped_mm (amendment 4); completes the registered E13 Granite-1B result",
    "E14_qwen0.5b": "E14 exploratory (Qwen2.5-0.5B, S0 S2 S6 S7, 3 seeds); report written. exit 1 = the runner's rule controls_all_passed requires every control to be true and two (S1 identity, raw wire bytes) are not evaluated (null) in this grid; every evaluated control passed",
    "E13d_qwen0.5b_m6": "E13d exploratory replication (amendment 6): 6 workers on 6 domains, cells S0 S1 S2 S5, all controls passed",
    "E15_gwaya_rft": "E15: GWAYA-gated RFT of Qwen2.5-Coder-1.5B (stages baseline, loop_base, rft, train, tuned, controls), all controls passed",
    "E16_gwaya_lite_t4": "E16: GWAYA-Lite tiers 1.5B/3B/7B Q4_K_M via Ollama (PR #21 P1/P2), run valid",
    "E14_granite1b": "E14 exploratory (Granite-3.1-1B MoE, S0 S2 S6 S7, 3 seeds); exit 1 for the same reason as E14_qwen0.5b (null controls), every evaluated control passed",
}
EXT_SKIPPED = "skipped: output exists (this fineweb-edu configuration ran in the first queue)"


def ext_queue_section(q: Optional[Dict[str, Any]]) -> List[str]:
    if not q:
        return ["Extension queue log not found."]
    start = dt.datetime.fromtimestamp(q["started_unix"], dt.timezone.utc)
    ran = [j for j in q["jobs"] if "seconds" in j]
    end = max(j["start_unix"] + j["seconds"] for j in ran) if ran else q["started_unix"]
    util = q.get("gpu_util_by_job", {})
    lines = [f"- Extension queue started {start:%Y-%m-%d %H:%M} UTC (right after the first queue's deadline), budget {q['hours']} h; last job ended "
             f"{dt.datetime.fromtimestamp(end, dt.timezone.utc):%Y-%m-%d %H:%M} UTC, i.e. {(end - q['started_unix']) / 3600:.2f} h of wall time.",
             f"- GPU utilisation (NVML, {q.get('gpu_samples')} samples): mean {q.get('gpu_util_mean_pct')} %, peak memory {q.get('gpu_mem_peak_gib')} GiB.", "",
             "| job | status | start (UTC) | wall (min) | mean GPU util % | what it was |", "| :-- | :-- | :-- | --: | --: | :-- |"]
    for j in q["jobs"]:
        st = f"{dt.datetime.fromtimestamp(j['start_unix'], dt.timezone.utc):%H:%M}" if "start_unix" in j else "-"
        mins = f"{j['seconds'] / 60:.1f}" if "seconds" in j else "-"
        note = EXT_NOTES.get(j["name"]) or (EXT_SKIPPED if str(j["status"]).startswith("skipped") else "")
        lines.append(f"| `{j['name']}` | {j['status']} | {st} | {mins} | {util.get(j['name'], '-')} | {note} |")
    gpu_h = sum(j.get("seconds", 0) for j in ran if j.get("status") in ("ok", "exit 1")) / 3600
    lines += ["", f"The six jobs above used {gpu_h:.2f} GPU-hours in total ({sum(1 for j in ran if j['status'] == 'ok')} ended ok, "
              f"{sum(1 for j in ran if j['status'] == 'exit 1')} with exit 1 for the null-control reason explained in the table).",
              "The remaining E11b sweep jobs of the first queue (code, GSM8K, news, fiction, PubMed, French Wikipedia, 4096-token contexts) were not re-queued here: "
              "they were stopped on purpose, so E11b stays partial (the three fineweb-edu 1024-token configurations) and exploratory, and its aggregate report does not exist.",
              "Explanations of the non-ok rows of the first queue: `E13_granite1b` was aborted by the registered time fail-fast (2.69 s/step projected 3.09 h > 3.0 h; "
              "amendment 4 switched to grouped_mm, 0.78 s/step); `E13_qwen0.5b_m2proc` was aborted by the registered learning gate (M = 2 learns only 0.0253 nats by round 3 < 0.05) "
              "and re-run as `E13_qwen0.5b_m2proc_v2` under the waiver of amendment 5; `E13_granite1b_v2` was cut by the deadline and resumed above."]
    return lines


def _lat(c: Dict[str, Any], k: str = "p50") -> str:
    v = (c.get("round_latency_s") or {}).get(k)
    return "-" if v is None else f"{v:.1f}"


def e13d_section(rep: Optional[Dict[str, Any]], pred: Optional[Dict[str, Any]]) -> List[str]:
    if not rep or not pred:
        return ["E13d report not found."]
    m = rep["models"]["qwen0.5b"]
    p = pred["models"]["qwen0.5b"]
    doms = list(m["cells"]["S0"]["per_domain_final_mean"])
    lines = ["Exploratory replication registered in amendment 6 (changes no registered E13 result): Qwen2.5-0.5B, **6 workers on 6 Hugging Face domains** "
             "(code, wiki-fr, pubmed, gsm8k, news, fiction), K = 3 of 6 for S2, K = M = 6 for S5, the registered straggler (worker 5, fiction) and churn, thread-hosted peers "
             f"(memory preflight), 3 seeds. Controls all passed: {m['controls_all_passed']}; cells failed: {m['failed_cells']}. Receipt `{rep['receipt_sha256'][:16]}`.", "",
             "| cell | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |", "| :-- | :-- | --: | --: | :-- | --: |"]
    for k, c in m["cells"].items():
        w = c["wire"]
        ratio = "-" if not w.get("compression_ratio_actual") else f"{w['compression_ratio_actual']:.2f}x / {w['compression_ratio_ideal']:.2f}x"
        lines.append(f"| {k} {c['name']} | {', '.join(f'{x:.4f}' for x in c['final_loss_per_seed'])} | {c['admitted_fraction_mean']:.2f} | {c['late_discards_total']} | {ratio} | {_lat(c)} |")
    lines += ["", "Per-domain final held-out loss (mean over seeds):", "", "| cell | " + " | ".join(doms) + " |", "| :-- |" + " --: |" * len(doms)]
    for k, c in m["cells"].items():
        lines.append(f"| {k} | " + " | ".join(f"{c['per_domain_final_mean'][d]:.3f}" for d in doms) + " |")
    b = p["P13b"]
    lines += ["", f"Replication statuses by the registered rules: P13a **{p['P13a']['status']}** (S1 equals S0 bit-identically); P13b **{b['status']}** (retention {b['mean']:+.3f} ± {b['half_width']:.3f}, "
              f"interval [{b['interval'][0]:+.3f}, {b['interval'][1]:+.3f}] includes the 0.5 bar; per seed {[round(x, 3) for x in b['per_seed'].values()]}); "
              f"P13c **{p['P13c']['status']}** (upload {p['P13c']['upload_ratio_actual']:.4f} of raw, ideal {p['P13c']['upload_ratio_ideal']:.4f}). "
              f"Without the quorum (S5, K = M) retention is {p['P13d']['S5']['mean_retention']:.3f}.",
              "Reading: the registered E13 result (P13b refuted at M = 4) is **not replicated as a refutation** at M = 6 — it is inconclusive — and the point estimate is positive; "
              "the K = M cell again keeps almost all of the synchronous improvement. Three seeds give a wide interval; nothing here supports a claim for WAN or volunteer swarms.", ""]
    return lines


def e14_section(art_dir: Path) -> List[str]:
    try:
        import analyze_E14_E15_hypotheses as hyp  # noqa: PLC0415

        doc = hyp.build_e14(art_dir)
    except Exception as exc:  # an optional section must not break report generation
        return [f"E14 reports not analysable: {exc!r}"]
    if not doc["models"]:
        return ["E14 reports not found."]
    e13 = load("E13_swarm_local_report.json") or {}
    lines = ["**Exploratory follow-up of E13** (pre-registered in `E14_swarm_fairness_prereg.json` before any S6/S7 run; decides no registered E13 prediction). Two admission policies against the plain "
             "K = ceil(M/2) swarm S2: **S6 fair quorum** (a worker excluded from one closed round must be in the next quorum; the round waits for it, capped) and **S7 deferred carry** "
             "(a late delta is queued and merged into the next round with weight token_weight / (1 + staleness), staleness <= 2). S2 is re-run in the same invocation so that S2, S6, S7 are paired "
             "seed by seed. Hypotheses by the registered rule: H14a (S6) / H14b (S7) are **supported** if the df = 2 95 % t-interval (t = 4.302652729911275) of the per-seed difference "
             "loss(S2) − loss(cell) is above 0, **refuted** if below 0, **inconclusive** if it includes 0. Two hypotheses per model, no multiple-comparison correction. One GPU, thread-hosted peers over "
             "localhost HTTP, 6 rounds × 20 inner steps, 3 seeds: **no WAN claim**.", ""]
    for name, d in doc["models"].items():
        rep = load(f"E14_report_{name}.json") or {}
        m = rep["models"][name]
        cells = m["cells"]
        lines += [f"### {name}", "", f"Report receipt `{rep.get('receipt_sha256', '')[:16]}`; runner's controls_all_passed = {m['controls_all_passed']} "
                  f"(not evaluated in this grid: {', '.join(d['controls']['not_evaluated'])}; failed: {d['controls']['failed'] or 'none'}; S0 twin bit-identical: {d['controls']['s0_twin_bit_identical']}); "
                  f"E14 controls valid: {d['controls']['valid']}.", "",
                  "| cell | final held-out loss (per seed) | mean | admitted | late | upload ratio (actual / ideal) | round latency p50 / p95 (s) |", "| :-- | :-- | --: | --: | --: | :-- | :-- |"]
        for k in ("S0", "S2", "S6", "S7"):
            c = cells[k]
            w = c["wire"]
            ratio = "-" if not w.get("compression_ratio_actual") else f"{w['compression_ratio_actual']:.2f}x / {w['compression_ratio_ideal']:.2f}x"
            fl = c["final_loss_per_seed"]
            lines.append(f"| {k} {c['name']} | {', '.join(f'{x:.4f}' for x in fl)} | {sum(fl) / len(fl):.4f} | {c['admitted_fraction_mean']:.2f} | {c['late_discards_total']} | {ratio} | {_lat(c)} / {_lat(c, 'p95')} |")
        lines += ["", "Paired differences (S2 − cell, nats; positive = the cell is better than S2) and the registered statuses:", "",
                  "| hypothesis | cell | per seed | mean ± half-width | interval | status |", "| :-- | :-- | :-- | :-- | :-- | :-- |"]
        for h in ("H14a", "H14b"):
            r = d[h]
            lines.append(f"| {h} | {r['cell']} | {', '.join(f'{x:+.4f}' for x in r['S2_minus_cell_per_seed'].values())} | {pm(r['mean'], r['half_width'])} | "
                         f"[{r['interval'][0]:+.4f}, {r['interval'][1]:+.4f}] | **{r['status']}** |")
        lines += ["", "Retention of the synchronous improvement ρ = (L_round0 − L_cell)/(L_round0 − L_S0) (viability bar 0.5 of P13b; the K = M value on Qwen2.5-0.5B in E13 was 0.94):", "",
                  "| cell | per seed | mean ± half-width |", "| :-- | :-- | :-- |"]
        for c, r in d["retention_vs_S0"].items():
            lines.append(f"| {c} | {', '.join(f'{x:+.3f}' for x in r['per_seed'].values())} | {pm(r['mean'], r['half_width'], 3)} |")
        cost = d["descriptive_costs"]
        s6, s7 = cost["S6_fair"], cost["S7_carry"]
        lines += ["", "Costs (descriptive): round latency p50 " + ", ".join(f"{c} {v:.1f} s" for c, v in cost["round_latency_p50_s"].items() if v is not None)
                  + f" — S6 / S2 = {cost['S6_over_S2_latency']:.2f}, S7 / S2 = {cost['S7_over_S2_latency']:.2f} (one time-shared GPU, so these are scheduling artefacts, not WAN latencies). "
                  f"S6 waited an extra {s6['extra_wait_s_total']:.1f} s in total ({s6['rounds_with_extra_wait']} rounds with extra wait, {s6['forced_admissions_total']} forced admissions, "
                  f"{s6['fair_wait_timeouts_total']} wait timeouts); S7 carried {s7['carried_count_total']} deltas ({s7['carried_tokens_total']} tokens, staleness histogram {s7['staleness_hist']}, "
                  f"{s7['too_stale_discarded_total']} discarded as too stale). Late discards S2 / S6 / S7: {cost['late_discards_total']['S2']} / {cost['late_discards_total']['S6']} / {cost['late_discards_total']['S7']}.", ""]
        e13m = (((e13.get("models") or {}).get(name) or {}).get("cells") or {}).get("S2")
        if e13m:
            same = [round(x, 4) for x in e13m["final_loss_per_seed"]] == [round(x, 4) for x in cells["S2"]["final_loss_per_seed"]]
            lines += [f"Replication check: the E13 registered S2 on this model had final losses {[round(x, 4) for x in e13m['final_loss_per_seed']]}; the S2 re-run inside E14 gave "
                      f"{[round(x, 4) for x in cells['S2']['final_loss_per_seed']]} ("
                      + ("identical to 4 decimals" if same else "per-seed values differ: thread scheduling is not deterministic between runs") + ").", ""]
    parts = []
    for name, d in doc["models"].items():
        rt = d["retention_vs_S0"]
        parts.append(f"{name}: H14a **{d['H14a']['status']}**, H14b **{d['H14b']['status']}** (retention S2 {rt['S2']['mean']:+.2f}, S6 {rt['S6']['mean']:+.2f}, S7 {rt['S7']['mean']:+.2f}; "
                     f"S6 round latency {d['descriptive_costs']['S6_over_S2_latency']:.2f}x S2)")
    lines += ["**Reading.** " + "; ".join(parts) + ". Where a hypothesis is inconclusive its interval includes 0: that is absence of a detected effect with 3 seeds, not evidence of none. "
              "Whether the dense-model support transfers to the pretrained MoE is therefore not established at this size. The fixed constants (MAX_EXCLUSION 1, MAX_STALENESS 2) were not tuned.", ""]
    return lines


def _wilson(c: Dict[str, Any]) -> str:
    lo, hi = c["wilson95"]
    return f"{100 * c['pass_at_1']:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}]"


def e15_section(r: Optional[Dict[str, Any]], art_dir: Path) -> List[str]:
    if not r:
        return ["E15 report not found."]
    try:
        import analyze_E14_E15_hypotheses as hyp  # noqa: PLC0415

        h = hyp.build_e15(art_dir)["hypotheses"]
    except Exception as exc:
        return [f"E15 report present but not analysable: {exc!r}"]
    st, cd, ctl = r["stages"], r["conditions"], r["controls"]
    lines = [f"**GWAYA-gated rejection-sampling fine-tuning (RFT) of {r['model']}** (rank-16 LoRA, 7.5 GiB per-process cap on the T4), pre-registered with one amendment. "
             "GWAYA here is a fail-closed gate: a candidate is accepted only if the vendored zero-stub AST audit is clean **and** the sandboxed task tests pass. "
             f"Controls all passed: {r['controls_all_passed']}. Receipt `{r['receipt_sha256'][:16]}`; design hash `{r['design_hash'][:16]}`.", "",
             "Stages (NVML power × time for energy; whole-GPU, descriptive):", "", "| stage | status | wall (min) | peak reserved (GiB) | peak allocated (GiB) | energy (Wh) |", "| :-- | :-- | --: | --: | --: | --: |"]
    for k, s in st.items():
        mt = s.get("meter")
        lines.append(f"| {k} | {s['status']} | " + (f"{mt['wall_s'] / 60:.1f} | {mt['peak_reserved_gib']:.2f} | {mt['peak_allocated_gib']:.2f} | {mt['energy_wh']:.1f} |" if mt else f"{s['elapsed_s'] / 60:.2f} | - | - | - |"))
    lines += ["", "Conditions (pass@1 = all tests pass; Wilson 95 % interval; MBPP-sanitized n = 257, HumanEval n = 164 for the *h* rows):", "",
              "| condition | what | n | passes | pass@1 [Wilson 95 %] | accepted by the gate | generated tokens | GPU s | correct per GPU-min |", "| :-- | :-- | --: | --: | :-- | --: | --: | --: | --: |"]
    what = {"C0": "base, greedy", "C1": "base + GWAYA loop", "C2": "RFT-tuned, greedy", "C3": "tuned + GWAYA loop", "C0h": "HumanEval base", "C2h": "HumanEval tuned"}
    for k in ("C0", "C1", "C2", "C3", "C0h", "C2h"):
        c = cd[k]
        lines.append(f"| {k} | {what[k]} | {c['n']} | {c['passes']} | {_wilson(c)} | {c['accepted_gwaya']} | {c['generated_tokens']} | {c['gen_gpu_seconds']:.0f} | {c['correct_per_gpu_minute']:.1f} |")
    lines += ["", "Paired comparisons (exact two-sided McNemar from the discordant counts; the p recomputed here equals the report's):", "",
              "| comparison | difference (points) | b only (second wins) | a only (first wins) | McNemar p | registered status |", "| :-- | --: | --: | --: | --: | :-- |"]
    for label, key in (("C2 − C0 (H15a)", "H15a"), ("C1 − C0 (H15b)", "H15b")):
        x = h[key]
        lines.append(f"| {label} | {x['points']:+.2f} | {x['b_only']} | {x['a_only']} | {x['mcnemar_p']:.3g} | **{x['status']}** |")
    for label, x in (("C3 − C1 (H15c)", h["H15c_descriptive"]["C3_minus_C1"]), ("C3 − C0 (H15c)", h["H15c_descriptive"]["C3_minus_C0"]), ("HumanEval C2h − C0h (H15e)", h["H15e_descriptive"])):
        lines.append(f"| {label} | {x['points']:+.2f} | {x['b_only']} | {x['a_only']} | {x['mcnemar_p']:.3g} | descriptive |")
    lines += ["", f"H15d (peak reserved <= 8.0 GiB in every stage): **{h['H15d']['status']}** (maximum {h['H15d']['max_peak_reserved_gib']:.2f} GiB).",
              f"Reading: the GWAYA inference loop moves the base by {h['H15b']['points']:+.1f} points ({h['H15b']['b_only']} problems gained, {h['H15b']['a_only']} lost; H15b {h['H15b']['status']}); "
              f"RFT-tuned greedy moves it by {h['H15a']['points']:+.1f} points (p = {h['H15a']['mcnemar_p']:.2f}; H15a {h['H15a']['status']}, i.e. no detected gain from the fine-tuning) and by "
              f"{h['H15e_descriptive']['points']:+.1f} points on HumanEval (p = {h['H15e_descriptive']['mcnemar_p']:.2f}); "
              f"stacking the loop on the tuned model vs on the base: {h['H15c_descriptive']['C3_minus_C1']['points']:+.1f} points (p = {h['H15c_descriptive']['C3_minus_C1']['mcnemar_p']:.2f}). The loop costs "
              f"{cd['C1']['gen_gpu_seconds'] / cd['C0']['gen_gpu_seconds']:.1f}x the GPU seconds of greedy decoding (C1 {cd['C1']['gen_gpu_seconds']:.0f} s vs C0 {cd['C0']['gen_gpu_seconds']:.0f} s; the winner fell back to the first greedy candidate on "
              f"{cd['C1']['winner_source']['fallback_first_greedy']} of {cd['C1']['n']} problems). One training seed, one model size: no training-seed interval.", "",
              "RFT data (rejection sampling with the gate, all tests of the train problem):", "",
              f"- {st['rft']['problems']} train problems after decontamination (464 raw, 2 removed for prompt-text overlap with eval); **{st['rft']['problems_with_accepted']} / {st['rft']['problems']} problems yielded at least one accepted sample** "
              f"(fail-fast threshold 150); mean yield {st['rft']['mean_yield']:.4f} accepted per sample; **{st['rft']['kept_pairs']} kept (problem, solution) pairs**; unverified samples {st['rft']['unverified_samples']}; "
              f"{st['rft']['tokens']} generated tokens in {st['rft']['gen_gpu_seconds']:.0f} GPU s ({st['rft']['meter']['wall_s'] / 60:.1f} min wall, {st['rft']['meter']['energy_wh']:.1f} Wh).",
              f"- Training: {st['train']['steps']} steps over {st['train']['pairs']} pairs (3 epochs), {st['train']['lora_params']} LoRA parameters on {st['train']['wrapped_linears']} linears, "
              f"epoch mean CE {[round(x, 4) for x in st['train']['epoch_mean_ce']]}, skipped non-finite steps {st['train']['skipped_nonfinite_steps']}, adapter sha256 `{st['train']['lora_sha256'][:16]}`.", "",
              "Controls (the run is valid iff all pass): "
              f"gate accepts {ctl['i_mbpp_reference_accept']['accepted']}/{ctl['i_mbpp_reference_accept']['n']} MBPP reference solutions and {ctl['i_humaneval_canonical_accept']['accepted']}/{ctl['i_humaneval_canonical_accept']['n']} HumanEval canonical solutions; "
              f"{ctl['ii_planted_stubs_rejected']['rejected_as_stub']}/{ctl['ii_planted_stubs_rejected']['n']} planted stubs rejected; hostile-candidate containment probes all contained = {ctl['iii_containment_probes']['all_contained']}; "
              f"no leakage (id overlap {ctl['iv_no_leakage']['id_overlap']}, text overlap {ctl['iv_no_leakage']['text_overlap']}, {ctl['iv_no_leakage']['reference_solution_overlap']} identical reference solutions under different prompts reported); "
              f"peak reserved <= 8 GiB = {ctl['vi_peak_reserved_le_8gib']['pass']}.",
              "", "**Sandbox caveat.** The gate is the vendored GWAYA zero-stub auditor plus our own user/mount/net/pid-namespace sandbox with rlimits (no seccomp, no cgroups, shared kernel), not GWAYA's Bubblewrap oracle. "
              "It is a defence for a 1.5B model generating Python functions, not for adversarial third-party code, and results are labelled GWAYA-style. The vendored auditor can reject legitimate code (it flags any `pass` outside an except handler or class body), which lowers the RFT yield.", ""]
    return lines


def _extract_marked(path: Path, begin: str, end: str) -> List[str]:
    try:
        t = path.read_text(encoding="utf-8")
    except OSError:
        return []
    i = t.find(begin)
    if i < 0 or t.find(end) < 0:
        return []
    a = t.index("-->", i) + 3
    return t[a:t.index(end)].strip("\n").splitlines()


def e16_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E16 report not found."]
    a = r["analysis"]
    H = a["H16"]
    ctl = r["controls"]
    lad = a["escalation_ladder"]
    lines = [f"**GWAYA-Lite on a Tesla T4** (GWAYA draft PR #21 criteria P1 and P2 measured as a specification): Qwen2.5-Coder 1.5B / 3B / 7B-Instruct, Q4_K_M, served by Ollama {r['environment']['ollama_version']}, "
             f"MBPP-sanitized n = {r['n_problems']}, same gate and sandbox as E15. `run_valid` = {r['run_valid']} (idle GPU, gate and unload controls). Receipt `{r['receipt_sha256'][:16]}`.", ""]
    block = _extract_marked(ROOT / "docs" / "community" / "GWAYA_T4_DEPLOYMENT.md", "<!-- E16-MEASURED-TABLE-BEGIN", "<!-- E16-MEASURED-TABLE-END -->")
    if block:
        lines += ["Measured table (generated from the receipt into `docs/community/GWAYA_T4_DEPLOYMENT.md` by `scripts/fill_deployment_guide.py`, reproduced here unchanged):", "", *block, ""]
    lines += ["Registered hypotheses (E16 pre-registration), statuses copied from the report:", "", "| hypothesis | claim | status |", "| :-- | :-- | :-- |",
              "| H16a | 7B Q4 fits within 6656 MiB (nvidia-smi delta incl. CUDA context) | " + ", ".join(f"num_ctx {k}: **{v['status']}** ({v['measured_mib']} MiB)" for k, v in H["H16a"].items()) + " |",
              f"| H16b | measured/estimated VRAM ratio in [0.80, 1.25] for every tier × ctx | **{H['H16b']['status']}** (ratios " + ", ".join(f"{x['tier']}@{x['ctx']}: {x['ratio']:.2f}" for x in H["H16b"]["rows"]) + ") |",
              "| H16c | the loop does not lower pass@1 | " + ", ".join(f"{t}: **{s}**" for t, s in H["H16c"].items()) + " |",
              "| H16d | loop gain >= +5.0 points with McNemar p < 0.05 | " + ", ".join(f"{t}: **{s}**" for t, s in H["H16d"].items()) + " |",
              "| H16e | precision of VERIFIED >= 0.80 vs hidden tests | " + ", ".join(f"{t}: **{s}** ({100 * a['tiers'][t]['p2_gate_calibration']['precision']:.1f} %)" for t, s in H["H16e"].items()) + " |",
              f"| H16f (descriptive) | 7B decode >= 20 tok/s at 4096 | **{H['H16f_descriptive']['status']}** ({H['H16f_descriptive']['decode_tps_median']:.1f} tok/s) |", "",
              "Loop effect per tier (paired by problem; exact McNemar):", "", "| tier | raw R | loop L | L − R (points) | gained / lost | McNemar p | loop extra model time (s) | determinism (10 repeats identical) |", "| :-- | :-- | :-- | --: | --: | --: | --: | --: |"]
    for t, v in a["tiers"].items():
        p = v["paired_L_minus_R"]
        lines.append(f"| {t} | {_wilson(v['R4096'])} | {_wilson(v['L4096'])} | {100 * p['diff']:+.1f} | {p['b_only']} / {p['a_only']} | {p['mcnemar_p']:.2g} | {v['loop_extra_model_seconds']:.0f} | {v['determinism']['fraction_identical']:.1f} |")
    r8 = a["tiers"]["7b"]
    lines += ["", f"Context invariance (7B raw): num_ctx 8192 gives {r8['R8192']['passes']}/{r8['R8192']['n']} against {r8['R4096']['passes']}/{r8['R4096']['n']} at 4096 (paired difference {r8['paired_R8192_minus_R4096']['diff']:+.1f}).",
              "", f"Escalation ladder (descriptive, no status): 1.5B loop → 3B loop on FAILED → 7B loop on FAILED reaches {lad['ladder_passes']}/{lad['n']} ({100 * lad['ladder_pass_at_1']:.1f} %) against {lad['seven_b_only_passes']}/{lad['n']} "
              f"({100 * lad['seven_b_only_pass_at_1']:.1f} %) for the 7B loop alone (paired difference {100 * lad['paired_ladder_vs_7b_only']['diff']:+.1f} points, McNemar p = {lad['paired_ladder_vs_7b_only']['mcnemar_p']:.2f}); "
              f"it escalates {lad['escalated_to_3b']} problems ({100 * lad['escalation_rate_3b']:.0f} %) to the 3B and {lad['escalated_to_7b']} ({100 * lad['escalation_rate_7b']:.0f} %) to the 7B and uses "
              f"{lad['ladder_model_seconds']:.0f} s of model time against {lad['seven_b_only_model_seconds']:.0f} s ({100 * lad['ladder_over_7b_only_time']:.0f} %).", "",
              f"Controls: idle GPU at start {ctl['idle_gpu']['memory_used_mib']} MiB (limit {ctl['idle_gpu']['threshold_mib']}); gate accepts {ctl['gate']['reference_control']['accepted']}/{ctl['gate']['reference_control']['n']} references and containment = {ctl['gate']['containment']}; "
              f"unload returned to {ctl['unload']['memory_used_mib']} MiB (baseline {ctl['unload']['baseline_start_mib']}); every probe 100 % on the GPU = {ctl['gpu_offload']['pass']}; probe contamination clean = {ctl['probe_contamination']['clean_all']}.",
              "", "**Caveats.** One T4 and one runtime (Ollama, Q4_K_M only); the 6656 MiB PR ceiling is a Windows WDDM reserve, tested here as a threshold on Linux, not as a physical limit; the gate is the E15 gate "
              "(vendored zero-stub auditor + namespace sandbox, not GWAYA's Bubblewrap oracle); VERIFIED means audit clean and the *visible* test passed, and 5-7 % of those solutions still fail the hidden tests; "
              "Ollama greedy decoding is not guaranteed bit-stable (10 repeated problems per tier were identical here). Publishing anything on the GWAYA repository or its PR is left to the owner.", ""]
    return lines


def live_section(c: Optional[Dict[str, Any]]) -> List[str]:
    if not c:
        return ["Live-check receipt not found."]
    lines = [f"`scripts/gwaya_lite.py` (small-first ladder Qwen2.5-Coder 1.5B → 3B → 7B Q4_K_M behind the fail-closed gate, guide `docs/community/GWAYA_LITE_ASSISTANT.md`) was deployed as a local "
             f"OpenAI-compatible service on the T4 at `{c['url']}` (loopback only). Live check `artifacts/gwaya_lite_live_check.json` (receipt `{c['receipt_sha256'][:16]}`, written {c['written_utc']}), "
             f"`all_ok` = {c['all_ok']}:", "",
             f"- health: HTTP {c['health'][0]} {c['health'][1]}; models: HTTP {c['models'][0]}; listening on loopback: {c['listening']}.", "",
             "| task | tests | verdict | tier that answered | candidates | repair rounds | generated tokens | wall (s) | gate verdict |", "| :-- | --: | :-- | :-- | --: | --: | --: | --: | :-- |"]
    for t in c["tasks"]:
        lines.append(f"| {t['task']} | {t['tests']} | **{t['verdict']}** | {t['tier']} | {t['candidates']} | {t['repair_rounds']} | {t['generated_tokens']} | {t['wall_s']} | {t['gate_verdict']} |")
    u, s = c["untested_request"], c["sse"]
    lines += ["", f"- A request without tests was answered **{u['verdict']}** (never VERIFIED without tests = {u['never_verified_without_tests']}).",
              f"- Streaming: {s['data_chunks']} SSE data chunks, `[DONE]` terminator = {s['done_terminator']}.",
              f"- Foreign `Host` header → HTTP {c['foreign_host_status']}; foreign `Origin` header → HTTP {c['foreign_origin_status']}.",
              f"- After the requests: GPU memory used {c['gpu_memory_used_mib_after']} MiB; resident Ollama models {c['ollama_resident_after']} (name, MiB).", "",
              "Scope: a smoke check of three small tasks, not a benchmark (the benchmark is E16 above). The service shares the E15 sandbox limits (not GWAYA's Bubblewrap oracle); the editor-integration examples in the guide are untested in real editors; "
              "it must stay on loopback unless an API key is configured.", ""]
    return lines


def main() -> int:
    q = load("T4_benchmark_queue_log.json")
    e11, e12, e10b, e11b = load("E11_kv_lattice_report.json"), load("E12_replay_divergence_report.json"), load("E10b_lattice_05b_report.json"), load("E11b_kv_lattice_sweep_report.json")
    e13, e13p, e13r, e13m2 = load("E13_swarm_local_report.json"), load("E13_swarm_local_predictions.json"), load("E13_swarm_local_report_rotstraggle.json"), load("E13_swarm_local_report_m2proc.json")
    hw = (e11 or {}).get("hardware", {})
    qx, e13m, e13mp = load("T4_benchmark_queue_ext_log.json"), load("E13_swarm_local_report_m6.json"), load("E13_swarm_local_predictions_m6.json")
    e14s, e15, e16 = load("E14_swarm_fairness_prereg.json"), load("E15_gwaya_rft_report.json"), load("E16_gwaya_lite_t4_report.json")
    live = json.loads((ROOT / "artifacts" / "gwaya_lite_live_check.json").read_text(encoding="utf-8")) if (ROOT / "artifacts" / "gwaya_lite_live_check.json").exists() else None
    md = [
        "# HiveMind T4 benchmark campaign (2026-10-23/24)",
        "",
        f"Hardware: {hw.get('gpu', 'Tesla T4')} (15 GB, sm_75), 8 vCPU, 31.5 GB RAM; torch {hw.get('torch', '?')}, transformers {hw.get('transformers', '?')}. "
        "Data: Hugging Face WikiText-2, fineweb-edu, codeparrot-clean-valid, GSM8K, French Wikipedia, PubMed, CNN/DailyMail and LAMBADA (E13/E13d/E14 domains), MBPP and HumanEval (local cache). Models: Qwen2.5-0.5B-Instruct (fp16), "
        "Qwen2.5-Coder-7B-Instruct (fp16, embeddings on CPU), Gemma-4-E4B (bf16, per-layer embeddings on CPU), Granite-3.1-1B-A400M and 3B MoE (fp16), "
        "Qwen2.5-Coder-1.5B-Instruct (fp16, E15) and Qwen2.5-Coder 1.5B / 3B / 7B Q4_K_M via Ollama (E16).",
        "",
        "Every experiment below was pre-registered before it ran according to the file modification times (files in `artifacts/experiments/*prereg*.json`; note that the `registered_utc` stamps of several of them "
        "post-date the stage timestamps of their runs, an unresolved inconsistency recorded in `docs/RESEARCH_PROGRAM.md`; deviations and amendments are recorded in the files, "
        "E14 and the E13c/E13d diagnostics are labelled exploratory). Numbers are copied from the receipts listed at the end; 95 % intervals are df = 2 t-intervals over "
        "three seeds. A refuted or inconclusive prediction is reported as such. No peer review of record exists for any of this; PTS / $HIVE are non-monetary and no reward is attached to any experiment. "
        "The first queue ran 2026-10-23 20:50 to 2026-10-24 06:50 UTC (sections 1-6); the extension queue (sections 1b and 6-10) ran after it.",
        "",
        "## 1. Campaign timeline", "", *queue_section(q), "",
        "## 1b. Extension queue timeline", "", *ext_queue_section(qx), "",
        "## 2. E11 — lattice KV-cache codecs vs FP8 / scalar (RQ10, HM-S10-05)", "", *e11_section(e11), "",
        "## 3. E12 — spot-check replay divergence across hardware (RQ11, HM-S10-08)", "", *e12_section(e12), "",
        "## 4. E10b — lattice outer-delta codec at 0.5B (RQ9, HM-S10-01)", "", *e10b_section(e10b), "",
        "## 5. E11b — exploratory sweep over Hugging Face domains and context length", "", *e11b_section(e11b), "",
        "## 6. E13 — single-GPU swarm simulation (Qwen2.5-0.5B dense; Granite-3.1 MoE 1B and 3B)", "", *e13_section(e13, e13p, e13r, e13m2), "",
        "### E13d — six-domain, six-worker replication (exploratory, amendment 6)", "", *e13d_section(e13m, e13mp), "",
        "## 7. E14 — fair quorum and deferred carry for the swarm (exploratory; Qwen2.5-0.5B and Granite-3.1-1B)", "", *(e14_section(EXP) if e14s else ["E14 pre-registration not found."]), "",
        "## 8. E15 — GWAYA-gated RFT of a coding assistant under 8 GB of VRAM", "", *e15_section(e15, EXP), "",
        "## 9. E16 — GWAYA-Lite tiers on the T4 (PR #21 P1 / P2)", "", *e16_section(e16), "",
        "## 10. GWAYA-Lite deployment on the T4", "", *live_section(live), "",
        "## 11. Receipts", "",
        "| artifact | sha256 (first 12) |", "| :-- | :-- |",
    ]
    for n in ("E11_kv_lattice_prereg.json", "E11_kv_lattice_prereg_amendment_1.json", "E11_kv_lattice_report.json", "E12_replay_divergence_prereg.json",
              "E12_replay_divergence_report.json", "E10b_lattice_05b_prereg.json", "E10b_lattice_05b_report.json", "E11b_kv_lattice_sweep_prereg.json",
              "E11b_kv_lattice_sweep_report.json", "E13_swarm_local_prereg.json", "E13_swarm_local_prereg_amendment_1.json", "E13_swarm_local_prereg_amendment_2.json", "E13_swarm_local_prereg_amendment_3.json",
              "E13_swarm_local_prereg_amendment_4.json", "E13_swarm_local_prereg_amendment_5.json", "E13_swarm_local_prereg_amendment_6.json",
              "E13_swarm_local_report.json", "E13_swarm_local_predictions.json", "E13_swarm_local_report_m6.json", "E13_swarm_local_predictions_m6.json",
              "E14_swarm_fairness_prereg.json", "E14_report_qwen0.5b.json", "E14_report_granite1b.json", "E14_hypotheses.json",
              "E15_gwaya_rft_prereg.json", "E15_gwaya_rft_prereg_amendment_1.json", "E15_gwaya_rft_report.json", "E15_hypotheses.json",
              "E16_gwaya_lite_t4_prereg.json", "E16_gwaya_lite_t4_report.json", "../gwaya_lite_live_check.json",
              "T4_benchmark_queue_log.json", "T4_benchmark_queue_ext_log.json"):
        md.append(f"| `{n.replace('../', 'artifacts/')}` | `{sha(n)}` |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
