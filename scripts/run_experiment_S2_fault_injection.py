#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Experiment S-2: does the Solver Bee verifier catch wrong answers?

Pre-registered in artifacts/experiments/S2_fault_injection_prereg.json (thresholds
fixed there; this script only measures). Stdlib only, CPU only, no network.

Default is self-contained: an in-script RK4 solver, closed-form references.
With --binary and --problems the real hivemind-solver CLI and problem library
are used instead (not part of the registered self-contained thresholds).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess  # noqa: S404 - fixed argv list, never a shell
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.solverbee import verifier  # noqa: E402

SEED = 20261027
FAULTS = ("F1_wrong_matrix_entry", "F2_perturbed_trajectory", "F3_loosened_tolerance",
          "F4_nan_injected", "F5_truncated_output")
FAMILIES = ("harmonic", "chain", "affine")
SOLVER_INFO = {"name": "s2-inscript-rk4", "rev": "0" * 40, "method": "bdf", "crate_version": "0.0.0"}


# ------------------------------------------------------------ in-script solver

def rk4_solve(spec_bytes: bytes) -> dict:
    """Fixed-step RK4 for x' = A x + b; substeps scale with rtol^(-1/4) (error ~ rtol)."""
    spec = json.loads(spec_bytes)
    A, b, x = spec["problem"]["A"], spec["problem"]["b"], [float(v) for v in spec["y0"]]
    n = len(x)
    norm = max(1.0, max(sum(abs(v) for v in row) for row in A))
    scale = norm * spec["rtol"] ** -0.25

    def f(v):
        return [sum(A[i][j] * v[j] for j in range(n)) + b[i] for i in range(n)]

    t, ys = spec["t0"], []
    for t_next in spec["t_out"]:
        m = max(1, math.ceil(1.0 * (t_next - t) * scale))
        h = (t_next - t) / m
        for _ in range(m):
            k1 = f(x)
            k2 = f([x[i] + 0.5 * h * k1[i] for i in range(n)])
            k3 = f([x[i] + 0.5 * h * k2[i] for i in range(n)])
            k4 = f([x[i] + h * k3[i] for i in range(n)])
            x = [x[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        t = t_next
        ys.append(list(x))
    return make_result(spec_bytes, spec, ys)


def make_result(spec_bytes: bytes, spec: dict, ys: list) -> dict:
    return {"schema": "hivemind-solver-result/1", "id": spec["id"], "status": "ok",
            "t": list(spec["t_out"]), "y": ys,
            "stats": {"steps": 0, "rhs_evals": 0, "jac_evals": None, "last_order": None},
            "solver": dict(SOLVER_INFO),
            "input_sha256": hashlib.sha256(spec_bytes).hexdigest()}


# ----------------------------------------------------------------- case library

class Case:
    def __init__(self, spec: dict, problem: dict | None, solve):
        self.spec = spec
        self.problem = problem
        self.solve = solve  # bytes -> result dict
        self.bytes = json.dumps(spec).encode("utf-8")
        self.honest = solve(self.bytes)


def _times(rng: random.Random, horizon: float) -> list[float]:
    k = 8
    return [round(horizon * (i + 1) / k, 6) for i in range(k)]


def _spec(sid, A, b, y0, t_out) -> dict:
    return {"schema": "hivemind-solver-spec/1", "id": sid,
            "problem": {"kind": "linear", "A": A, "b": b}, "y0": y0, "t0": 0.0,
            "t_out": t_out, "rtol": 1e-8, "atol": 1e-10, "method": "bdf", "max_steps": 100000}


def _problem(spec, ref_y, invariants, how) -> dict:
    return {"schema": "hivemind-solver-problem/1", "id": spec["id"], "spec": spec,
            "reference": {"t": list(spec["t_out"]), "y": ref_y, "max_abs_error_bound": 1e-12,
                          "method": how, "tool": "scripts/run_experiment_S2_fault_injection.py",
                          "tool_versions": {"python": "stdlib math"}},
            "invariants": invariants, "tolerance_policy": {"max_rel_err_factor": 100.0}}


def make_case(family: str, idx: int, rng: random.Random) -> Case:
    sid = f"s2-{family}-{idx}"
    if family == "harmonic":
        w, horizon = rng.uniform(0.5, 3.0), rng.uniform(2.0, 6.0)
        t = _times(rng, horizon)
        spec = _spec(sid, [[0.0, 1.0], [-w * w, 0.0]], [0.0, 0.0], [1.0, 0.0], t)
        ref = [[math.cos(w * s), -w * math.sin(w * s)] for s in t]
        inv = [{"name": "energy", "kind": "energy", "matrix": [[w * w, 0.0], [0.0, 1.0]],
                "value": w * w / 2.0, "tol": 1e-6}]
        how = "closed form cos/sin"
    elif family == "chain":
        k1, k2, horizon = rng.uniform(0.5, 2.0), rng.uniform(2.5, 5.0), rng.uniform(2.0, 6.0)
        t = _times(rng, horizon)
        spec = _spec(sid, [[-k1, 0.0, 0.0], [k1, -k2, 0.0], [0.0, k2, 0.0]], [0.0] * 3, [1.0, 0.0, 0.0], t)
        ref = []
        for s in t:
            y1 = math.exp(-k1 * s)
            y2 = k1 / (k2 - k1) * (math.exp(-k1 * s) - math.exp(-k2 * s))
            ref.append([y1, y2, 1.0 - y1 - y2])
        inv = [{"name": "mass_sum", "kind": "linear", "weights": [1.0, 1.0, 1.0], "value": 1.0, "tol": 1e-9}]
        how = "closed form two-step chain"
    else:
        k = [rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0)]
        bb = [rng.uniform(0.1, 1.0), rng.uniform(0.1, 1.0)]
        y0 = [rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)]
        t = _times(rng, rng.uniform(2.0, 6.0))
        spec = _spec(sid, [[-k[0], 0.0], [0.0, -k[1]]], bb, y0, t)
        ref = [[bb[i] / k[i] + (y0[i] - bb[i] / k[i]) * math.exp(-k[i] * s) for i in range(2)] for s in t]
        inv, how = [], "closed form affine decay"
    return Case(spec, _problem(spec, ref, inv, how), rk4_solve)


# --------------------------------------------------------------- fault injection

def inject(fault: str, case: Case, rng: random.Random) -> dict:
    """Return a faulty result for the TRUE spec bytes of ``case``."""
    res = copy.deepcopy(case.honest)
    if fault == "F1_wrong_matrix_entry":
        spec2 = copy.deepcopy(case.spec)
        A = spec2["problem"]["A"]
        i, j = rng.randrange(len(A)), rng.randrange(len(A))
        A[i][j] += 0.01
        res["y"] = case.solve(json.dumps(spec2).encode("utf-8"))["y"]
    elif fault == "F2_perturbed_trajectory":
        cells = [(k, j) for k, row in enumerate(res["y"]) for j, v in enumerate(row) if abs(v) >= 1e-2]
        k, j = rng.choice(cells)
        res["y"][k][j] *= 1.0 + 1e-3
    elif fault == "F3_loosened_tolerance":
        spec2 = copy.deepcopy(case.spec)
        spec2["rtol"] = 1e-3
        res["y"] = case.solve(json.dumps(spec2).encode("utf-8"))["y"]
    elif fault == "F4_nan_injected":
        k, j = rng.randrange(len(res["y"])), rng.randrange(len(res["y"][0]))
        res["y"][k][j] = float("nan")
    elif fault == "F5_truncated_output":
        res["t"], res["y"] = res["t"][:-1], res["y"][:-1]
    else:
        raise ValueError(fault)
    return res


def _same(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # NaN != NaN is irrelevant here


# --------------------------------------------------------------------- the run

def presentations(case: Case, result: dict) -> dict:
    """Both presentation modes of one result -> certificates."""
    rerun = case.solve
    return {
        "LIBRARY": verifier.verify(case.bytes, result, problem=case.problem, rerun=rerun),
        "ADHOC": verifier.verify(case.bytes, result, problem=None, rerun=rerun),
    }


def run(seed: int, clean_n: int, n_per_fault: int, case_factory=None) -> dict:
    factory = case_factory or make_case
    caveat_ok = True
    all_certs = 0

    def count(certs):
        nonlocal caveat_ok, all_certs
        for c in certs.values():
            all_certs += 1
            caveat_ok &= c.get("caveat") == verifier.CAVEAT

    # clean set
    rng = random.Random(f"{seed}-clean")
    clean_cases = [factory(FAMILIES[i % 3], i, rng) for i in range(clean_n)]
    clean = {m: Counter() for m in ("LIBRARY", "ADHOC")}
    first_library = None
    for c in clean_cases:
        certs = presentations(c, c.honest)
        count(certs)
        for m, cert in certs.items():
            clean[m][cert["verdict"]] += 1
        first_library = first_library or certs["LIBRARY"]

    # faults
    faults = {}
    for fault in FAULTS:
        rng = random.Random(f"{seed}-{fault}")
        per_mode = {m: {"verdicts": Counter(), "failed_checks": Counter(), "misses": []}
                    for m in ("LIBRARY", "ADHOC")}
        redraws = 0
        for i in range(n_per_fault):
            while True:
                case = factory(FAMILIES[i % 3], 1000 + i + 100 * redraws, rng)
                faulty = inject(fault, case, rng)
                if not _same(faulty, case.honest):
                    break
                redraws += 1
            certs = presentations(case, faulty)
            count(certs)
            for m, cert in certs.items():
                per_mode[m]["verdicts"][cert["verdict"]] += 1
                for ck in cert["checks"]:
                    if ck["status"] == "fail":
                        per_mode[m]["failed_checks"][ck["name"]] += 1
                if cert["verdict"] in ("VERIFIED", "AUDITED"):
                    per_mode[m]["misses"].append(case.spec["id"])
        faults[fault] = {"N": n_per_fault, "ineffective_redraws": redraws, "modes": {
            m: {"verdicts": dict(v["verdicts"]),
                "detected": v["verdicts"]["FAILED"] + v["verdicts"]["UNVERIFIED"],
                "failed_by_check": dict(v["failed_checks"]), "missed_case_ids": v["misses"]}
            for m, v in per_mode.items()}}

    # controls
    pos_ok = first_library is not None and first_library["verdict"] == "VERIFIED"
    base = clean_cases[0]
    spec_b = copy.deepcopy(base.spec)
    spec_b["y0"] = [v * 1.001 if v else 0.001 for v in spec_b["y0"]]
    bytes_b = json.dumps(spec_b).encode("utf-8")
    res_b = base.solve(bytes_b)  # carries sha256 of spec B's own bytes
    neg = verifier.verify(base.bytes, res_b, problem=None, rerun=base.solve)
    count({"neg": neg})
    hashes_chk = next(c for c in neg["checks"] if c["name"] == "hashes")
    neg_ok = hashes_chk["status"] == "fail" and neg["verdict"] == "FAILED"

    a1 = all(f["modes"][m]["detected"] == f["N"] for f in faults.values() for m in f["modes"])
    a2 = all(clean[m]["FAILED"] == 0 for m in clean)
    report = {
        "experiment_id": "S2-solver-verifier-fault-injection",
        "prereg": "artifacts/experiments/S2_fault_injection_prereg.json",
        "mode": "self_contained" if case_factory is None else "custom",
        "seed": seed,
        "verifier_version": verifier.VERIFIER_VERSION,
        "clean": {"n": clean_n, "verdicts_by_mode": {m: dict(v) for m, v in clean.items()}},
        "faults": faults,
        "controls": {"positive_library_verified": pos_ok,
                     "negative_hashes_flagged": neg_ok,
                     "negative_verdict": neg["verdict"], "negative_hashes_detail": hashes_chk["detail"]},
        "run_valid": pos_ok and neg_ok,
        "acceptance": {"A1_all_injected_failed_or_unverified": "PASS" if a1 else "FAIL",
                       "A2_no_clean_failed": "PASS" if a2 else "FAIL",
                       "A3_caveat_on_every_certificate": "PASS" if caveat_ok else "FAIL",
                       "certificates_checked_for_caveat": all_certs},
    }
    report["receipt_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return report


# ---------------------------------------------------------------- real CLI mode

def real_cases(binary: str, problems_dir: str):  # pragma: no cover - needs the real CLI
    def solve(spec_bytes: bytes) -> dict:
        p = subprocess.run([binary, "run"], input=spec_bytes, capture_output=True, timeout=120, check=False)
        return json.loads(p.stdout)

    cases = []
    for path in sorted(Path(problems_dir).glob("*.json")):
        prob = json.loads(path.read_text(encoding="utf-8"))
        if prob.get("schema") == "hivemind-solver-problem/1":
            cases.append(Case(prob["spec"], prob, solve))
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--clean-n", type=int, default=60)
    ap.add_argument("--n-per-fault", type=int, default=30)
    ap.add_argument("--binary", help="path to the real hivemind-solver (needs --problems)")
    ap.add_argument("--problems", help="problem library directory (needs --binary)")
    ap.add_argument("--out", help="report path")
    args = ap.parse_args(argv)
    if bool(args.binary) != bool(args.problems):
        ap.error("--binary and --problems go together")
    if args.binary:  # pragma: no cover
        lib = real_cases(args.binary, args.problems)
        it = iter(range(10**9))
        report = run(args.seed, len(lib), args.n_per_fault,
                     case_factory=lambda fam, idx, rng: lib[next(it) % len(lib)])
        report["mode"] = "real_cli"
        report.pop("receipt_sha256")
        report["binary_sha256"] = hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()
        report["receipt_sha256"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        default_out = "S2_fault_injection_report_real_cli.json"
    else:
        report = run(args.seed, args.clean_n, args.n_per_fault)
        default_out = "S2_fault_injection_report.json"
    out = Path(args.out) if args.out else ROOT / "artifacts" / "experiments" / default_out
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": report["acceptance"], "run_valid": report["run_valid"],
                      "out": str(out)}, indent=2))
    return 0 if report["run_valid"] else 3


if __name__ == "__main__":
    sys.exit(main())
