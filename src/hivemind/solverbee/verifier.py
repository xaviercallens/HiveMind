# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Independent verifier for Solver Bee results (contract hivemind-solver-cert/1).

Pure standard library. ``verify`` never raises: every check is wrapped so an
exception becomes ``status = "fail"`` with the message, and a missing or
unusable result gives the verdict UNVERIFIED (fail closed).

Checks: shape, hashes, reference, invariants, tightening. The verdict is
computed by the single function :func:`decide`.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

VERIFIER_VERSION = "1"
CERT_SCHEMA = "hivemind-solver-cert/1"
SPEC_SCHEMA = "hivemind-solver-spec/1"
RESULT_SCHEMA = "hivemind-solver-result/1"

CAVEAT = (
    "VERIFIED means the listed checks passed. In the E16 measurement 5-7% of "
    "VERIFIED code answers still failed hidden tests; it never means 'correct'."
)

MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_SPEC_BYTES = 64 * 1024

# Tightening: the loose result may differ from the 100x-tighter rerun by at most
# TIGHTENING_FACTOR * (rtol*|y_tight| + atol) with the LOOSE tolerances. The tight
# run is ~100x more accurate, so the difference is essentially the loose run's own
# error; solvers control local (not global) error, so a factor of 100 allows for
# error accumulation, matching the library's default max_rel_err_factor.
TIGHTENING_FACTOR = 100.0
DEFAULT_REF_FACTOR = 100.0
# Times are matched with this relative tolerance (the CLI echoes t_out exactly).
_T_MATCH_RTOL = 1e-12

_CHECK_NAMES = ("shape", "hashes", "reference", "invariants", "tightening")


class _CheckError(Exception):
    """Raised inside a check to report a failure with a message."""


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON used for result_sha256 (sort_keys, compact, non-ASCII kept)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chk(name: str, status: str, detail: str, value=None, bound=None) -> dict:
    return {"name": name, "status": status, "value": value, "bound": bound, "detail": detail}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x: Any) -> bool:
    return _is_num(x) and math.isfinite(x)


def _safe(name: str, fn: Callable[[], dict]) -> dict:
    """Run one check; any exception becomes a failing check (fail closed)."""
    try:
        return fn()
    except _CheckError as exc:
        return _chk(name, "fail", str(exc))
    except Exception as exc:  # noqa: BLE001 - fail closed on anything
        return _chk(name, "fail", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- the decision

def decide(checks: dict[str, dict], *, usable: bool, has_problem: bool) -> tuple[str, list[str]]:
    """The one verdict decision function (contract rules, exactly).

    ``checks`` maps check name -> check dict. ``usable`` is False when the result
    could not be obtained or checked at all (missing/oversized/non-ok result,
    unparseable spec). Rules, in order:

    1. not usable                         -> UNVERIFIED
    2. any check has status "fail"        -> FAILED
    3. required checks all "pass"         -> VERIFIED if a problem (reference) was
       given, else AUDITED. Required: shape, hashes, tightening; reference when a
       problem is given; invariants whenever they are defined (a skipped
       invariants check means none are defined and is allowed).
    4. otherwise (a required check skipped, e.g. no rerun) -> UNVERIFIED
    """
    reasons: list[str] = []
    if not usable:
        return "UNVERIFIED", ["result could not be obtained or checked (fail closed)"]
    failed = [n for n in _CHECK_NAMES if checks[n]["status"] == "fail"]
    if failed:
        return "FAILED", [f"{n}: {checks[n]['detail']}" for n in failed]

    required = ["shape", "hashes", "tightening"]
    if has_problem:
        required.append("reference")
    missing = [n for n in required if checks[n]["status"] != "pass"]
    if missing:
        return "UNVERIFIED", [f"required check not passed: {n} ({checks[n]['detail']})" for n in missing]
    inv = checks["invariants"]
    if inv["status"] == "skip" and not inv["detail"].startswith("no invariants"):
        return "UNVERIFIED", [f"invariants not checked: {inv['detail']}"]
    if has_problem:
        return "VERIFIED", ["shape, hashes, reference, invariants, tightening passed"]
    return "AUDITED", ["no reference: shape, hashes, invariants-if-any, tightening passed"]


# ------------------------------------------------------------------- the checks

def _parse_spec(spec_bytes: bytes) -> dict:
    if not isinstance(spec_bytes, (bytes, bytearray)):
        raise _CheckError("spec_bytes must be bytes")
    if len(spec_bytes) > MAX_SPEC_BYTES:
        raise _CheckError(f"spec larger than {MAX_SPEC_BYTES} bytes")
    spec = json.loads(bytes(spec_bytes).decode("utf-8"))
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise _CheckError("spec schema is not hivemind-solver-spec/1")
    return spec


def _dimension(spec: dict) -> int:
    n = len(spec["y0"])
    if n < 1:
        raise _CheckError("empty y0")
    return n


def _validate_trajectory(res: dict, spec: dict, what: str) -> None:
    """Shared structural validation of a result against its spec."""
    t_out = spec["t_out"]
    n = _dimension(spec)
    if res.get("schema") != RESULT_SCHEMA:
        raise _CheckError(f"{what}: schema is not {RESULT_SCHEMA}")
    if res.get("status") != "ok":
        raise _CheckError(f"{what}: status is {res.get('status')!r}")
    if res.get("id") != spec.get("id"):
        raise _CheckError(f"{what}: id {res.get('id')!r} != spec id {spec.get('id')!r}")
    t, y = res.get("t"), res.get("y")
    if not isinstance(t, list) or not isinstance(y, list):
        raise _CheckError(f"{what}: t and y must be lists")
    if len(t) != len(t_out) or len(y) != len(t_out):
        raise _CheckError(f"{what}: length t={len(t)} y={len(y)} expected {len(t_out)}")
    for i, (a, b) in enumerate(zip(t, t_out)):
        if not _finite(a) or a != b:
            raise _CheckError(f"{what}: t[{i}]={a!r} differs from t_out[{i}]={b!r}")
    for k, row in enumerate(y):
        if not isinstance(row, list) or len(row) != n:
            raise _CheckError(f"{what}: y[{k}] is not a list of {n} numbers")
        for j, v in enumerate(row):
            if not _finite(v):
                raise _CheckError(f"{what}: y[{k}][{j}] is not a finite number ({v!r})")


def _check_shape(spec: dict, result: dict) -> dict:
    _validate_trajectory(result, spec, "result")
    return _chk("shape", "pass", "schema, id, status, finiteness, lengths and t == t_out ok")


def _check_hashes(spec_bytes: bytes, result: dict) -> tuple[dict, str | None]:
    got = result.get("input_sha256")
    want = sha256_hex(bytes(spec_bytes))
    try:
        result_sha = sha256_hex(canonical_json(result))
    except (ValueError, TypeError):
        result_sha = None
    if got != want:
        return _chk("hashes", "fail", f"input_sha256 {got!r} != sha256 of the spec bytes {want}"), result_sha
    if result_sha is None:
        return _chk("hashes", "fail", "result is not canonically serialisable"), None
    return _chk("hashes", "pass", "input_sha256 equals sha256 of the exact spec bytes"), result_sha


# CLI defaults (solvers/hivemind-solver/src/lib.rs): rtol and atol are optional in a spec.
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10


def _tolerances(spec: dict, n: int) -> tuple[float, list[float]]:
    rtol = spec.get("rtol", DEFAULT_RTOL)
    atol = spec.get("atol", DEFAULT_ATOL)
    if not _finite(rtol) or rtol <= 0:
        raise _CheckError("rtol must be a positive finite number")
    if _is_num(atol):
        atols = [float(atol)] * n
    elif isinstance(atol, list) and len(atol) == n and all(_finite(a) for a in atol):
        atols = [float(a) for a in atol]
    else:
        raise _CheckError("atol must be a number or a list of n numbers")
    if any(a < 0 for a in atols):
        raise _CheckError("atol must be non-negative")
    return float(rtol), atols


def _worst_ratio(y, y_ref, rtol, atols, factor) -> float:
    worst = 0.0
    for row, ref in zip(y, y_ref):
        for j, (a, b) in enumerate(zip(row, ref)):
            bound = factor * (rtol * abs(b) + atols[j])
            err = abs(a - b)
            ratio = err / bound if bound > 0 else (0.0 if err == 0 else math.inf)
            worst = max(worst, ratio)
    return worst


def _check_reference(spec: dict, result: dict, problem: dict) -> dict:
    if not isinstance(problem, dict) or problem.get("schema") != "hivemind-solver-problem/1":
        raise _CheckError("problem schema is not hivemind-solver-problem/1")
    pspec = problem.get("spec")
    if not isinstance(pspec, dict):
        raise _CheckError("problem has no spec")
    for key in ("problem", "y0", "t0"):
        if pspec.get(key) != spec.get(key):
            raise _CheckError(f"problem library entry does not describe this spec (field {key} differs)")
    ref = problem.get("reference")
    if not isinstance(ref, dict) or not isinstance(ref.get("t"), list) or not isinstance(ref.get("y"), list):
        raise _CheckError("problem reference missing t or y")
    n = _dimension(spec)
    rtol, atols = _tolerances(spec, n)
    factor = (problem.get("tolerance_policy") or {}).get("max_rel_err_factor", DEFAULT_REF_FACTOR)
    if not _finite(factor) or factor <= 0:
        raise _CheckError("tolerance_policy.max_rel_err_factor must be positive")
    ys, yr = [], []
    for tr, row in zip(ref["t"], ref["y"]):
        for i, tv in enumerate(spec["t_out"]):
            if abs(tr - tv) <= _T_MATCH_RTOL * max(1.0, abs(tv)):
                if len(row) != n or not all(_finite(v) for v in row):
                    raise _CheckError("reference row malformed")
                ys.append(result["y"][i])
                yr.append(row)
                break
    if not ys:
        raise _CheckError("no reference time coincides with t_out; reference check impossible")
    worst = _worst_ratio(ys, yr, rtol, atols, float(factor))
    detail = f"worst error/(factor*(rtol*|y_ref|+atol)) over {len(ys)} reference times, factor {factor}"
    if worst <= 1.0:
        return _chk("reference", "pass", detail, worst, 1.0)
    return _chk("reference", "fail", detail + " exceeds 1", worst, 1.0)


# --- invariants.  To add a kind: write _inv_<kind>(inv, spec, y) -> float returning the
# --- conserved quantity, register it in _INVARIANT_KINDS (see docs/solver_bee/VERIFIER.md).

def _vec(inv: dict, key: str, n: int) -> list[float]:
    v = inv.get(key)
    if not isinstance(v, list) or len(v) != n or not all(_finite(x) for x in v):
        raise _CheckError(f"invariant {inv.get('name')!r}: {key} must be a list of {n} finite numbers")
    return [float(x) for x in v]


def _mat(inv: dict, n: int) -> list[list[float]]:
    m = inv.get("matrix", inv.get("Q"))
    if (not isinstance(m, list) or len(m) != n
            or not all(isinstance(r, list) and len(r) == n and all(_finite(x) for x in r) for r in m)):
        raise _CheckError(f"invariant {inv.get('name')!r}: matrix must be {n}x{n} finite numbers")
    return m


def _quad(m, x) -> float:
    return sum(x[i] * m[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))


def _inv_linear(inv, spec, x):
    w = _vec(inv, "weights", len(x))
    return sum(a * b for a, b in zip(w, x))


def _inv_quadratic_form(inv, spec, x):
    return _quad(_mat(inv, len(x)), x)


def _inv_energy(inv, spec, x):
    if "model" in inv:
        model = inv.get("model")
        inv_params = inv.get("params")
        if inv_params is None:
            inv_params = {}
        if not isinstance(inv_params, dict):
            raise _CheckError(f"invariant {inv.get('name')!r}: params must be an object")
        spec_params = (spec.get("problem") or {}).get("params") or {}
        if model == "pendulum":
            if len(x) != 2:
                raise _CheckError("pendulum energy needs 2 state components [theta, omega]")
            g_over_l = inv_params.get("g_over_l", inv.get("g_over_l", spec_params.get("g_over_l")))
            if not _finite(g_over_l):
                raise _CheckError("pendulum energy needs finite g_over_l")
            return 0.5 * x[1] * x[1] + float(g_over_l) * (1.0 - math.cos(x[0]))
        if model == "kepler":
            if len(x) != 4:
                raise _CheckError("kepler energy needs 4 state components [x, y, vx, vy]")
            mu = inv_params.get("mu", inv.get("mu", spec_params.get("mu")))
            if not _finite(mu) or float(mu) <= 0:
                raise _CheckError("kepler energy needs positive finite mu")
            r = math.hypot(x[0], x[1])
            if r <= 0:
                raise _CheckError("kepler energy needs positive radius r > 0")
            return 0.5 * (x[2] * x[2] + x[3] * x[3]) - float(mu) / r
        if model == "projectile":
            if len(x) != 4:
                raise _CheckError("projectile energy needs 4 state components [x, y, vx, vy]")
            g = inv_params.get("g", inv.get("g", spec_params.get("g")))
            if not _finite(g):
                raise _CheckError("projectile energy needs finite g")
            return 0.5 * (x[2] * x[2] + x[3] * x[3]) + float(g) * x[1]
        raise _CheckError(f"unknown energy model {model!r}")
    return 0.5 * _quad(_mat(inv, len(x)), x)


def _inv_lv_first_integral(inv, spec, x):
    # x' = alpha x - beta x y ; y' = delta x y - gamma y ; V = delta x - gamma ln x + beta y - alpha ln y
    if len(x) != 2:
        raise _CheckError("lv_first_integral needs 2 species")
    # Canonical layout (what solvers/problems writes): inv["params"] = {alpha,beta,gamma,delta}.
    # Documented aliases, in order: the same four names at the top level of the invariant,
    # then spec.problem.params.
    spec_params = (spec.get("problem") or {}).get("params") or {}
    inv_params = inv.get("params")
    if inv_params is None:
        inv_params = {}
    if not isinstance(inv_params, dict):
        raise _CheckError(f"invariant {inv.get('name')!r}: params must be an object")
    p = {}
    for k in ("alpha", "beta", "gamma", "delta"):
        if k in inv_params:
            p[k] = inv_params[k]
        elif k in inv:
            p[k] = inv[k]
        else:
            p[k] = spec_params.get(k)
    if not all(_finite(v) for v in p.values()):
        raise _CheckError("lv_first_integral needs finite alpha, beta, gamma, delta")
    if x[0] <= 0 or x[1] <= 0:
        raise _CheckError("lv_first_integral needs positive populations")
    return (p["delta"] * x[0] - p["gamma"] * math.log(x[0])
            + p["beta"] * x[1] - p["alpha"] * math.log(x[1]))


_INVARIANT_KINDS: dict[str, Callable[[dict, dict, list], float]] = {
    "linear": _inv_linear,
    "quadratic_form": _inv_quadratic_form,
    "energy": _inv_energy,
    "lv_first_integral": _inv_lv_first_integral,
}


def _check_invariants(spec: dict, result: dict, problem: dict | None) -> dict:
    invs = (problem or {}).get("invariants") or []
    if not invs:
        return _chk("invariants", "skip", "no invariants defined")
    worst, worst_name = 0.0, ""
    for inv in invs:
        kind = inv.get("kind")
        fn = _INVARIANT_KINDS.get(kind)
        if fn is None:
            raise _CheckError(f"unknown invariant kind {kind!r}")
        tol = inv.get("tol")
        if not _finite(tol) or tol < 0:
            raise _CheckError(f"invariant {inv.get('name')!r}: tol must be a non-negative number")
        value = inv.get("value")
        if value is None and kind in ("lv_first_integral", "energy"):
            value = fn(inv, spec, [float(v) for v in spec["y0"]])
        if not _finite(value):
            raise _CheckError(f"invariant {inv.get('name')!r}: value must be a finite number")
        for row in result["y"]:
            drift = abs(fn(inv, spec, [float(v) for v in row]) - value) / tol if tol > 0 else (
                0.0 if fn(inv, spec, [float(v) for v in row]) == value else math.inf)
            if drift > worst:
                worst, worst_name = drift, str(inv.get("name"))
    detail = f"worst drift/tol over {len(invs)} invariant(s) (worst: {worst_name or 'n/a'})"
    if worst <= 1.0:
        return _chk("invariants", "pass", detail, worst, 1.0)
    return _chk("invariants", "fail", detail + " exceeds 1", worst, 1.0)


def _check_tightening(spec_bytes: bytes, spec: dict, result: dict, rerun) -> dict:
    if rerun is None:
        return _chk("tightening", "skip", "no rerun callable supplied")
    n = _dimension(spec)
    rtol, atols = _tolerances(spec, n)
    tight = dict(spec)
    tight["rtol"] = rtol / 100.0
    tight["atol"] = [a / 100.0 for a in atols] if isinstance(spec.get("atol"), list) else atols[0] / 100.0
    tight_bytes = json.dumps(tight, ensure_ascii=False).encode("utf-8")
    res2 = rerun(tight_bytes)
    if not isinstance(res2, dict):
        raise _CheckError("rerun did not return a result dict")
    _validate_trajectory(res2, spec, "rerun result")
    worst = _worst_ratio(result["y"], res2["y"], rtol, atols, TIGHTENING_FACTOR)
    detail = f"worst |y_loose-y_tight|/({TIGHTENING_FACTOR:g}*(rtol*|y_tight|+atol)) at rtol/100, atol/100"
    if worst <= 1.0:
        return _chk("tightening", "pass", detail, worst, 1.0)
    return _chk("tightening", "fail", detail + " exceeds 1", worst, 1.0)


# ------------------------------------------------------------------------ verify

def _result_size_ok(result: Any) -> bool:
    try:
        return len(json.dumps(result, allow_nan=True)) <= MAX_RESULT_BYTES
    except (TypeError, ValueError, RecursionError):
        return False


def verify(spec_bytes: bytes, result: dict, *, problem: dict | None = None,
           rerun: Callable[[bytes], dict] | None = None,
           binary_sha256: str | None = None) -> dict:
    """Return a hivemind-solver-cert/1 certificate. Never raises."""
    checks: dict[str, dict] = {}
    spec: dict | None = None
    usable = True
    unusable_why = ""

    try:
        spec = _parse_spec(spec_bytes)
        _dimension(spec)
    except Exception as exc:  # noqa: BLE001
        spec, usable, unusable_why = None, False, f"spec unusable: {exc}"

    if usable and (not isinstance(result, dict)):
        usable, unusable_why = False, "no result dict"
    if usable and not _result_size_ok(result):
        usable, unusable_why = False, f"result missing or larger than {MAX_RESULT_BYTES} bytes"
    if usable and result.get("status") != "ok":
        usable, unusable_why = False, f"solver status {result.get('status')!r}: {str(result.get('error'))[:200]}"
    if usable and result.get("schema") != RESULT_SCHEMA:
        usable, unusable_why = False, "result schema is not hivemind-solver-result/1"

    result_sha = None
    if not usable:
        for name in _CHECK_NAMES:
            checks[name] = _chk(name, "skip", unusable_why)
        checks["shape"] = _chk("shape", "fail", unusable_why)
    else:
        checks["shape"] = _safe("shape", lambda: _check_shape(spec, result))
        h = _safe("hashes", lambda: _check_hashes(spec_bytes, result)[0])
        checks["hashes"] = h
        try:
            result_sha = sha256_hex(canonical_json(result))
        except Exception:  # noqa: BLE001
            result_sha = None
        shape_ok = checks["shape"]["status"] == "pass"
        if problem is not None:
            checks["reference"] = (_safe("reference", lambda: _check_reference(spec, result, problem))
                                   if shape_ok else _chk("reference", "skip", "shape failed"))
        else:
            checks["reference"] = _chk("reference", "skip", "no problem library entry (ad-hoc spec)")
        checks["invariants"] = (_safe("invariants", lambda: _check_invariants(spec, result, problem))
                                if shape_ok else _chk("invariants", "skip", "shape failed"))
        checks["tightening"] = (_safe("tightening", lambda: _check_tightening(spec_bytes, spec, result, rerun))
                                if shape_ok else _chk("tightening", "skip", "shape failed"))

    verdict, reasons = decide(checks, usable=usable, has_problem=problem is not None)
    if not usable:
        reasons = reasons + [unusable_why]

    solver = result.get("solver") if isinstance(result, dict) else None
    rev = solver.get("rev") if isinstance(solver, dict) else None
    rid = (result.get("id") if isinstance(result, dict) else None) or (spec or {}).get("id")
    return {
        "schema": CERT_SCHEMA,
        "id": rid if isinstance(rid, str) else None,
        "verdict": verdict,
        "checks": [checks[n] for n in _CHECK_NAMES],
        "reasons": reasons,
        "input_sha256": sha256_hex(bytes(spec_bytes)) if isinstance(spec_bytes, (bytes, bytearray)) else None,
        "result_sha256": result_sha,
        "binary_sha256": binary_sha256,
        "solver_rev": rev if isinstance(rev, str) else None,
        "verifier_version": VERIFIER_VERSION,
        "caveat": CAVEAT,
    }


def certify_achieved_accuracy(
    spec_bytes: bytes,
    loose_result: dict,
    tight_result: dict,
    *,
    problem: dict | None = None,
    tightening_factor: float = 100.0,
) -> dict:
    """Return a Richardson empirical accuracy certificate (hivemind-accuracy-cert/1).

    In Issue #273, long-horizon orbits (e.g. kepler_eccentric_long over 100 orbits)
    accumulate global phase error at local rtol=1e-8 even though first integrals
    (energy, angular momentum) are conserved and tightening tolerances by 100x cuts
    global error 30x-70x (empirical order p in [0.73, 0.93]). This function reports
    the achieved global accuracy without altering or loosening the contract rules
    in verify() / decide().
    """
    spec = _parse_spec(spec_bytes)
    n = _dimension(spec)
    rtol, atols = _tolerances(spec, n)
    _validate_trajectory(loose_result, spec, "loose result")
    _validate_trajectory(tight_result, spec, "tight result")
    if tightening_factor <= 1.0 or not _finite(tightening_factor):
        raise ValueError("tightening_factor must be a finite number > 1")

    max_abs_diff = 0.0
    max_rel_ratio = 0.0
    for r_loose, r_tight in zip(loose_result["y"], tight_result["y"]):
        for j, (yl, yt) in enumerate(zip(r_loose, r_tight)):
            diff = abs(float(yl) - float(yt))
            if diff > max_abs_diff:
                max_abs_diff = diff
            scale = TIGHTENING_FACTOR * (rtol * abs(float(yt)) + atols[j])
            ratio = diff / scale if scale > 0 else (0.0 if diff == 0.0 else math.inf)
            if ratio > max_rel_ratio:
                max_rel_ratio = ratio

    ref_err_loose = None
    ref_err_tight = None
    empirical_order_p = None
    ref = (problem or {}).get("reference")
    if isinstance(ref, dict) and isinstance(ref.get("y"), list) and isinstance(ref.get("t"), list):
        ref_l = 0.0
        ref_t = 0.0
        matched_any = False
        for tr, ref_row in zip(ref["t"], ref["y"]):
            for i, tv in enumerate(spec["t_out"]):
                if abs(float(tr) - float(tv)) <= _T_MATCH_RTOL * max(1.0, abs(float(tv))):
                    matched_any = True
                    yl_row = loose_result["y"][i]
                    yt_row = tight_result["y"][i]
                    for yl, yt, yr in zip(yl_row, yt_row, ref_row):
                        ref_l = max(ref_l, abs(float(yl) - float(yr)))
                        ref_t = max(ref_t, abs(float(yt) - float(yr)))
                    break
        if matched_any:
            ref_err_loose = ref_l
            ref_err_tight = ref_t
            if ref_l > 0.0 and ref_t > 0.0:
                empirical_order_p = math.log(ref_l / ref_t) / math.log(tightening_factor)

    inv_loose = _safe("invariants", lambda: _check_invariants(spec, loose_result, problem))
    inv_tight = _safe("invariants", lambda: _check_invariants(spec, tight_result, problem))
    t0 = float(spec.get("t0", 0.0))
    t_span = float(spec["t_out"][-1]) - t0 if spec.get("t_out") else 0.0
    invariants_conserved = inv_loose["status"] == "pass" and inv_tight["status"] == "pass"
    long_horizon_phase_warning = bool(
        t_span >= 100.0 and max_rel_ratio > 1.0 and (invariants_conserved or inv_loose["status"] == "skip")
    )

    return {
        "schema": "hivemind-accuracy-cert/1",
        "id": spec.get("id"),
        "tightening_factor": float(tightening_factor),
        "achieved_abs_accuracy": max_abs_diff,
        "achieved_rel_ratio": max_rel_ratio,
        "reference_max_abs_err_loose": ref_err_loose,
        "reference_max_abs_err_tight": ref_err_tight,
        "empirical_order_p": empirical_order_p,
        "invariant_drift_ratio_loose": inv_loose.get("value"),
        "invariant_drift_ratio_tight": inv_tight.get("value"),
        "invariants_conserved": invariants_conserved,
        "t_span": t_span,
        "long_horizon_phase_warning": long_horizon_phase_warning,
        "caveat": CAVEAT,
    }
