# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Validate solvers/problems/*.json against the hivemind-solver-problem/1 contract (stdlib only).

    python3 solvers/tools/check_problems.py [DIR] [--min N]

Exit status 0 when every file is valid and at least N (default 12) problems exist.
Besides the schema it checks that the stored reference satisfies each declared invariant
within the invariant's tolerance and that the reference is at least 100x tighter than the
spec tolerance (reference.max_abs_error_bound).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

SPEC_SCHEMA = "hivemind-solver-spec/1"
PROBLEM_SCHEMA = "hivemind-solver-problem/1"
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SPECIES_RE = re.compile(r"^[A-Za-z0-9_+-]{1,32}$")
FAMILIES = {
    "robertson": ("k1", "k2", "k3"),
    "van_der_pol": ("mu",),
    "brusselator": ("a", "b"),
    "lotka_volterra": ("alpha", "beta", "delta", "gamma"),
    "harmonic": ("omega", "zeta"),
    "stiff_chain": ("n", "k"),
    "pendulum": ("g_over_l", "damping"),
    "projectile_drag": ("g", "k"),
    "kepler": ("mu",),
    "logistic": ("r", "K"),
}
FIXED_DIM = {"robertson": 3, "van_der_pol": 2, "brusselator": 2, "lotka_volterra": 2, "harmonic": 2,
             "pendulum": 2, "projectile_drag": 4, "kepler": 4, "logistic": 1}
ENERGY_MODELS = {"pendulum": ("g_over_l",), "kepler": ("mu",), "projectile": ("g",)}
# Files in a problem directory that are not problems (the held-out directory carries its freeze manifest).
NON_PROBLEM_FILES = {"FROZEN.json"}
MAX_DIM, MAX_REACTIONS, MAX_T_OUT, MAX_STOICH = 20, 50, 1000, 6
SPEC_KEYS = {"schema", "id", "problem", "y0", "t0", "t_out", "rtol", "atol", "method", "max_steps"}
REQUIRED_SPEC = {"schema", "id", "problem", "y0", "t_out"}
INVARIANT_KINDS = {"linear", "quadratic_form", "lv_first_integral", "energy"}
REFERENCE_MARGIN = 100.0


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec(x, n=None) -> bool:
    return isinstance(x, list) and (n is None or len(x) == n) and all(_num(v) for v in x)


def spec_dim(problem: dict, errs: list) -> int | None:
    kind = problem.get("kind")
    if kind == "named":
        extra = set(problem) - {"kind", "family", "params"}
        if extra:
            errs.append(f"spec.problem: unknown keys {sorted(extra)}")
        fam = problem.get("family")
        if fam not in FAMILIES:
            errs.append(f"spec.problem.family {fam!r} unknown")
            return None
        params = problem.get("params", {})
        if not isinstance(params, dict) or not all(_num(v) for v in params.values()):
            errs.append("spec.problem.params must map names to finite numbers")
            return None
        for k in params:
            if k not in FAMILIES[fam]:
                errs.append(f"spec.problem.params: unknown parameter {k!r} for {fam}")
        if fam == "stiff_chain":
            n = params.get("n", 10)
            if not (_num(n) and float(n).is_integer() and 2 <= n <= MAX_DIM):
                errs.append("stiff_chain n must be an integer in 2..20")
                return None
            return int(n)
        return FIXED_DIM[fam]
    if kind == "linear":
        extra = set(problem) - {"kind", "A", "b"}
        if extra:
            errs.append(f"spec.problem: unknown keys {sorted(extra)}")
        A, b = problem.get("A"), problem.get("b")
        if not (isinstance(A, list) and 1 <= len(A) <= MAX_DIM and all(_vec(r, len(A)) for r in A)):
            errs.append("linear A must be a square matrix of finite numbers, 1..20 rows")
            return None
        if not _vec(b, len(A)):
            errs.append("linear b must be a finite vector matching A")
            return None
        return len(A)
    if kind == "mass_action":
        extra = set(problem) - {"kind", "species", "reactions"}
        if extra:
            errs.append(f"spec.problem: unknown keys {sorted(extra)}")
        sp, rxs = problem.get("species"), problem.get("reactions")
        if not (isinstance(sp, list) and 1 <= len(sp) <= MAX_DIM and len(set(sp)) == len(sp)
                and all(isinstance(s, str) and SPECIES_RE.match(s) for s in sp)):
            errs.append("mass_action species: 1..20 unique names matching [A-Za-z0-9_+-]{1,32}")
            return None
        if not (isinstance(rxs, list) and 1 <= len(rxs) <= MAX_REACTIONS):
            errs.append("mass_action reactions: 1..50 entries")
            return None
        for i, r in enumerate(rxs):
            if not isinstance(r, dict) or set(r) - {"reactants", "products", "k"} or "k" not in r:
                errs.append(f"reaction {i}: needs k and only reactants/products/k")
                continue
            if not (_num(r["k"]) and r["k"] >= 0):
                errs.append(f"reaction {i}: k must be a finite number >= 0")
            for side in ("reactants", "products"):
                d = r.get(side, {})
                if not isinstance(d, dict):
                    errs.append(f"reaction {i}: {side} must be an object")
                    continue
                for s, nu in d.items():
                    if s not in sp:
                        errs.append(f"reaction {i}: unknown species {s!r}")
                    if not (isinstance(nu, int) and not isinstance(nu, bool) and 1 <= nu <= MAX_STOICH):
                        errs.append(f"reaction {i}: coefficient of {s!r} must be an integer 1..{MAX_STOICH}")
            if not r.get("reactants") and not r.get("products"):
                errs.append(f"reaction {i}: empty")
        return len(sp)
    errs.append(f"spec.problem.kind {kind!r} unknown")
    return None


def validate_spec(spec, errs: list) -> int | None:
    if not isinstance(spec, dict):
        errs.append("spec must be an object")
        return None
    for k in sorted(set(spec) - SPEC_KEYS):
        errs.append(f"spec: unknown key {k!r}")
    for k in sorted(REQUIRED_SPEC - set(spec)):
        errs.append(f"spec: missing key {k!r}")
    if spec.get("schema") != SPEC_SCHEMA:
        errs.append(f"spec.schema must be {SPEC_SCHEMA!r}")
    if not (isinstance(spec.get("id"), str) and ID_RE.match(spec["id"])):
        errs.append("spec.id invalid")
    n = spec_dim(spec.get("problem", {}), errs) if isinstance(spec.get("problem"), dict) else None
    if n is None and isinstance(spec.get("problem"), dict) is False:
        errs.append("spec.problem must be an object")
    if n is not None and not _vec(spec.get("y0"), n):
        errs.append(f"spec.y0 must be {n} finite numbers")
    t0 = spec.get("t0", 0.0)
    if not _num(t0):
        errs.append("spec.t0 must be finite")
        t0 = 0.0
    t = spec.get("t_out")
    if not (_vec(t) and 1 <= len(t) <= MAX_T_OUT):
        errs.append("spec.t_out: 1..1000 finite numbers")
    else:
        prev = t0
        for i, v in enumerate(t):
            if v <= prev:
                errs.append(f"spec.t_out not strictly increasing above t0 at index {i}")
                break
            prev = v
    rtol = spec.get("rtol", 1e-8)
    if not (_num(rtol) and 0 < rtol < 1):
        errs.append("spec.rtol must satisfy 0 < rtol < 1")
    atol = spec.get("atol", 1e-10)
    if isinstance(atol, list):
        if not (n is not None and _vec(atol, n) and all(a > 0 for a in atol)):
            errs.append("spec.atol array must be n positive numbers")
    elif not (_num(atol) and atol > 0):
        errs.append("spec.atol must be a positive number or an array")
    if spec.get("method", "bdf") not in ("bdf", "adams"):
        errs.append("spec.method must be bdf or adams")
    ms = spec.get("max_steps", 100000)
    if not (isinstance(ms, int) and not isinstance(ms, bool) and 1 <= ms <= 1_000_000):
        errs.append("spec.max_steps must be an integer in 1..1000000")
    return n


def eval_invariant(inv: dict, y: list) -> float:
    kind = inv["kind"]
    if kind == "linear":
        return sum(w * v for w, v in zip(inv["weights"], y))
    if kind == "quadratic_form":
        m = inv["matrix"]
        return sum(m[i][j] * y[i] * y[j] for i in range(len(y)) for j in range(len(y)))
    if kind == "lv_first_integral":
        p = inv["params"]
        return p["delta"] * y[0] - p["gamma"] * math.log(y[0]) + p["beta"] * y[1] - p["alpha"] * math.log(y[1])
    if kind == "energy":
        # model-specific conserved energy (held-out set): pendulum (damping 0), kepler, projectile (k = 0)
        m, p = inv["model"], inv["params"]
        if m == "pendulum":
            return y[1] * y[1] / 2 + p["g_over_l"] * (1 - math.cos(y[0]))
        if m == "kepler":
            return (y[2] * y[2] + y[3] * y[3]) / 2 - p["mu"] / math.sqrt(y[0] * y[0] + y[1] * y[1])
        if m == "projectile":
            return (y[2] * y[2] + y[3] * y[3]) / 2 + p["g"] * y[1]
    raise ValueError(kind)


def validate_invariants(invs, n, ref_y, errs: list):
    if not isinstance(invs, list):
        errs.append("invariants must be a list")
        return
    for i, inv in enumerate(invs):
        tag = f"invariants[{i}]"
        if not isinstance(inv, dict) or inv.get("kind") not in INVARIANT_KINDS:
            errs.append(f"{tag}: unknown kind")
            continue
        if not (isinstance(inv.get("name"), str) and inv["name"]):
            errs.append(f"{tag}: name required")
        if not _num(inv.get("value")):
            errs.append(f"{tag}: value must be finite")
        if not (_num(inv.get("tol")) and inv["tol"] > 0):
            errs.append(f"{tag}: tol must be > 0")
        kind = inv["kind"]
        ok = True
        if kind == "linear":
            ok = _vec(inv.get("weights"), n)
        elif kind == "quadratic_form":
            m = inv.get("matrix")
            ok = isinstance(m, list) and len(m) == n and all(_vec(r, n) for r in m)
        elif kind == "lv_first_integral":
            p = inv.get("params")
            ok = n == 2 and isinstance(p, dict) and all(_num(p.get(k)) and p[k] > 0 for k in ("alpha", "beta", "delta", "gamma"))
        elif kind == "energy":
            p = inv.get("params")
            need = ENERGY_MODELS.get(inv.get("model"))
            ok = need is not None and n == (2 if inv.get("model") == "pendulum" else 4) and isinstance(p, dict) \
                and set(p) == set(need) and all(_num(p[k]) for k in need)
            if ok and inv["model"] == "kepler":
                ok = p["mu"] > 0
        if not ok:
            errs.append(f"{tag}: malformed fields for kind {kind}")
            continue
        if _num(inv.get("value")) and _num(inv.get("tol")):
            for row in ref_y:
                if not all(v > 0 for v in row) and kind == "lv_first_integral":
                    errs.append(f"{tag}: reference leaves the positive quadrant")
                    break
                drift = abs(eval_invariant(inv, row) - inv["value"])
                if drift > inv["tol"]:
                    errs.append(f"{tag}: reference violates the invariant by {drift:.3g} > tol {inv['tol']:.3g}")
                    break


def validate_problem(doc, filename: str | None = None) -> list:
    errs: list = []
    if not isinstance(doc, dict):
        return ["problem must be an object"]
    allowed = {"schema", "id", "description", "spec", "reference", "invariants", "tolerance_policy"}
    for k in sorted(set(doc) - allowed):
        errs.append(f"unknown top-level key {k!r}")
    for k in ("schema", "id", "spec", "reference", "tolerance_policy"):
        if k not in doc:
            errs.append(f"missing key {k!r}")
    if doc.get("schema") != PROBLEM_SCHEMA:
        errs.append(f"schema must be {PROBLEM_SCHEMA!r}")
    if not (isinstance(doc.get("id"), str) and ID_RE.match(doc["id"])):
        errs.append("id invalid")
    if filename is not None and doc.get("id") != filename:
        errs.append(f"id {doc.get('id')!r} does not match file name {filename!r}")
    spec = doc.get("spec")
    n = validate_spec(spec, errs)
    if isinstance(spec, dict) and spec.get("id") != doc.get("id"):
        errs.append("spec.id must equal the problem id")
    ref = doc.get("reference")
    ref_y = []
    if not isinstance(ref, dict):
        errs.append("reference must be an object")
    else:
        for k in ("t", "y", "max_abs_error_bound", "method", "tool", "tool_versions"):
            if k not in ref:
                errs.append(f"reference: missing {k!r}")
        if isinstance(spec, dict) and ref.get("t") != spec.get("t_out"):
            errs.append("reference.t must equal spec.t_out")
        y = ref.get("y")
        if not (isinstance(y, list) and isinstance(spec, dict) and len(y) == len(spec.get("t_out", [])) and n is not None
                and all(_vec(r, n) for r in y)):
            errs.append("reference.y must be a finite matrix with one row per t_out and n columns")
        else:
            ref_y = y
        b = ref.get("max_abs_error_bound")
        if not (_num(b) and b > 0):
            errs.append("reference.max_abs_error_bound must be > 0")
        elif ref_y and isinstance(spec, dict):
            rtol, atol = spec.get("rtol", 1e-8), spec.get("atol", 1e-10)
            if not isinstance(atol, list) and _num(rtol) and _num(atol):
                tol_min = min(rtol * abs(v) + atol for r in ref_y for v in r)
                if b * REFERENCE_MARGIN > tol_min:
                    errs.append(f"reference bound {b:.3g} is not 100x tighter than the spec tolerance {tol_min:.3g}")
        tv = ref.get("tool_versions")
        if not (isinstance(tv, dict) and tv and all(isinstance(v, str) for v in tv.values())):
            errs.append("reference.tool_versions must be a non-empty string map")
        for k in ("method", "tool"):
            if not (isinstance(ref.get(k), str) and ref[k]):
                errs.append(f"reference.{k} must be a non-empty string")
        cc = ref.get("cross_check")
        if not (isinstance(cc, dict) and isinstance(cc.get("method"), str) and _num(cc.get("max_abs_diff"))):
            errs.append("reference.cross_check (method, max_abs_diff) is required: every reference needs a second method")
    validate_invariants(doc.get("invariants", []), n, ref_y, errs) if n is not None else None
    tp = doc.get("tolerance_policy")
    if not (isinstance(tp, dict) and _num(tp.get("max_rel_err_factor")) and tp["max_rel_err_factor"] > 0):
        errs.append("tolerance_policy.max_rel_err_factor must be > 0")
    return errs


def check_dir(directory: Path, minimum: int = 12) -> tuple[int, list]:
    problems = []
    files = sorted(f for f in directory.glob("*.json") if f.name not in NON_PROBLEM_FILES)
    if len(files) < minimum:
        problems.append(f"{directory}: {len(files)} problem files, need at least {minimum}")
    seen = set()
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{f.name}: unreadable: {exc}")
            continue
        for e in validate_problem(doc, f.stem):
            problems.append(f"{f.name}: {e}")
        if doc.get("id") in seen:
            problems.append(f"{f.name}: duplicate id")
        seen.add(doc.get("id"))
    return len(files), problems


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    minimum = 12
    if "--min" in argv:
        i = argv.index("--min")
        minimum = int(argv[i + 1])
        del argv[i : i + 2]
    directory = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / "problems"
    n, problems = check_dir(directory, minimum)
    for p in problems:
        print(p)
    print(f"{n} problem files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
