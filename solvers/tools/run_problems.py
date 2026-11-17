# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Run the hivemind-solver binary on every problem in solvers/problems and report the error table (stdlib only).

    python3 solvers/tools/run_problems.py --bin PATH/TO/hivemind-solver [--markdown] [--dir DIR]

For each problem: max over all elements of |y - y_ref| / (rtol*|y_ref| + atol) ("ratio"; the tolerance
policy accepts ratio <= max_rel_err_factor), the max invariant drift as a fraction of each invariant's tol,
solver statistics. Exit status 1 if any problem is outside its policy or errors. Nothing is loosened here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_problems import NON_PROBLEM_FILES, eval_invariant  # noqa: E402


def run_solver(binary: str, spec_bytes: bytes, timeout: float = 600.0) -> dict:
    p = subprocess.run([binary, "run"], input=spec_bytes, capture_output=True, timeout=timeout, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"solver exited with {p.returncode}: {p.stderr.decode(errors='replace')[:200]}")
    return json.loads(p.stdout)


def tightened_spec(spec: dict, factor: float = 100.0) -> dict:
    """The spec with rtol and atol divided by `factor` (same y0, t_out, method, max_steps)."""
    out = dict(spec)
    out["rtol"] = spec.get("rtol", 1e-8) / factor
    atol = spec.get("atol", 1e-10)
    out["atol"] = [a / factor for a in atol] if isinstance(atol, list) else atol / factor
    return out


def ratio_of(doc: dict, result: dict, rtol: float | None = None, atol: float | None = None) -> float:
    """max |y - y_ref| / (rtol*|y_ref| + atol); by default with the problem's own spec tolerances."""
    spec, ref = doc["spec"], doc["reference"]
    rtol = spec["rtol"] if rtol is None else rtol
    atol = spec["atol"] if atol is None else atol
    return max(abs(b - a) / (rtol * abs(a) + atol) for yr, ys in zip(ref["y"], result["y"]) for a, b in zip(yr, ys))


def evaluate(doc: dict, result: dict) -> dict:
    spec, ref = doc["spec"], doc["reference"]
    out = {"id": doc["id"], "status": result.get("status")}
    if result.get("status") != "ok":
        out["error"] = result.get("error")
        out["ok"] = False
        return out
    ratio = 0.0
    for yr, ys in zip(ref["y"], result["y"]):
        for a, b in zip(yr, ys):
            ratio = max(ratio, abs(b - a) / (spec["rtol"] * abs(a) + spec["atol"]))
    drift = 0.0
    worst_inv = ""
    for inv in doc.get("invariants", []):
        for row in result["y"]:
            d = abs(eval_invariant(inv, row) - inv["value"]) / inv["tol"]
            if d > drift:
                drift, worst_inv = d, inv["name"]
    factor = doc["tolerance_policy"]["max_rel_err_factor"]
    out.update(
        ratio=ratio,
        factor=factor,
        inv_drift=drift,
        inv_name=worst_inv,
        steps=result["stats"]["steps"],
        rhs=result["stats"]["rhs_evals"],
        method=result["solver"]["method"],
        ok=ratio <= factor and drift <= 1.0,
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent / "problems")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--tighten", type=float, default=None,
                    help="also run each problem with rtol and atol divided by this factor (e.g. 100) and report the "
                         "absolute error in the tightened run (as max |y - y_ref| / (spec rtol*|y_ref|+atol)) and the "
                         "reference's own error bound")
    ap.add_argument("--json", type=Path, default=None, help="write the rows to this JSON file")
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(args.dir.glob("*.json")):
        if f.name in NON_PROBLEM_FILES:
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        spec_bytes = (json.dumps(doc["spec"], separators=(",", ":")) + "\n").encode()
        row = evaluate(doc, run_solver(args.bin, spec_bytes))
        if args.tighten and row["status"] == "ok":
            spec_t = tightened_spec(doc["spec"], args.tighten)
            res_t = run_solver(args.bin, (json.dumps(spec_t, separators=(",", ":")) + "\n").encode())
            if res_t.get("status") == "ok":
                abs_err = max(abs(b - a) for yr, ys in zip(doc["reference"]["y"], res_t["y"]) for a, b in zip(yr, ys))
                abs_err_1 = max(abs(b - a) for yr, ys in zip(doc["reference"]["y"], run_solver(args.bin, spec_bytes)["y"])
                                for a, b in zip(yr, ys))
                row.update(tight_status="ok", tight_ratio=ratio_of(doc, res_t), tight_abs_err=abs_err,
                           abs_err=abs_err_1, tight_steps=res_t["stats"]["steps"],
                           ref_bound=doc["reference"]["max_abs_error_bound"])
            else:
                row.update(tight_status="error", tight_error=res_t.get("error"))
        rows.append(row)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    if args.markdown:
        print("| problem | method | steps | rhs evals | max error / (rtol*|y|+atol) | allowed | invariant drift / tol | verdict |")
        print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] != "ok":
            line = f"{r['id']}: solver status {r['status']}: {r.get('error')}"
            print(f"| {r['id']} | - | - | - | - | - | - | ERROR: {r.get('error')} |" if args.markdown else line)
            continue
        verdict = "within policy" if r["ok"] else "OUTSIDE POLICY"
        if args.markdown:
            print(f"| {r['id']} | {r['method']} | {r['steps']} | {r['rhs']} | {r['ratio']:.3g} | {r['factor']:g} | "
                  f"{r['inv_drift']:.3g} | {verdict} |")
        else:
            print(f"{r['id']:22s} ratio={r['ratio']:10.4g} (allowed {r['factor']:g}) inv_drift/tol={r['inv_drift']:.3g} "
                  f"steps={r['steps']} {verdict}")
    if args.tighten:
        print()
        if args.markdown:
            print(f"| problem | max abs error at spec tolerance | max abs error at tolerance/{args.tighten:g} | error scaling (x) | ratio at tightened run vs spec tolerance | reference error bound |")
            print("|---|---|---|---|---|---|")
        for r in rows:
            if r.get("tight_status") != "ok":
                print(f"| {r['id']} | - | - | - | - | tightened run: {r.get('tight_error', 'not run')} |" if args.markdown
                      else f"{r['id']}: tightened run {r.get('tight_status', 'not run')} {r.get('tight_error', '')}")
                continue
            scale = r["abs_err"] / r["tight_abs_err"] if r["tight_abs_err"] > 0 else float("inf")
            if args.markdown:
                print(f"| {r['id']} | {r['abs_err']:.3g} | {r['tight_abs_err']:.3g} | {scale:.3g} | {r['tight_ratio']:.3g} | {r['ref_bound']:.3g} |")
            else:
                print(f"{r['id']:26s} abs_err={r['abs_err']:.3g} -> {r['tight_abs_err']:.3g} (x{scale:.3g}) "
                      f"tight_ratio={r['tight_ratio']:.3g} ref_bound={r['ref_bound']:.3g}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
