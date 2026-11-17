# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Solver Bee verifier, scoring and the S-2 runner (stdlib, deterministic)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from hivemind.solverbee import scoring, verifier
from hivemind.solverbee.verifier import CAVEAT, decide, verify

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "solverbee"

_spec = importlib.util.spec_from_file_location("s2_runner", ROOT / "scripts" / "run_experiment_S2_fault_injection.py")
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture(params=["harmonic_problem.json", "chain_problem.json"])
def setup(request):
    problem = load(request.param)
    spec_bytes = json.dumps(problem["spec"]).encode("utf-8")
    result = s2.rk4_solve(spec_bytes)
    return problem, spec_bytes, result


def check(cert, name):
    return next(c for c in cert["checks"] if c["name"] == name)


# ------------------------------------------------------------------ verdict paths

def test_verified_with_problem_and_rerun(setup):
    problem, sb, res = setup
    cert = verify(sb, res, problem=problem, rerun=s2.rk4_solve, binary_sha256="ab" * 32)
    assert cert["verdict"] == "VERIFIED", cert
    assert [c["status"] for c in cert["checks"]] == ["pass"] * 5
    assert cert["schema"] == "hivemind-solver-cert/1"
    assert cert["binary_sha256"] == "ab" * 32
    assert cert["input_sha256"] == hashlib.sha256(sb).hexdigest()
    assert cert["verifier_version"] == "1"
    assert cert["solver_rev"] == "0" * 40


def test_audited_without_problem(setup):
    _, sb, res = setup
    cert = verify(sb, res, rerun=s2.rk4_solve)
    assert cert["verdict"] == "AUDITED"
    assert check(cert, "reference")["status"] == "skip"
    assert check(cert, "invariants")["status"] == "skip"


def test_unverified_without_rerun(setup):
    problem, sb, res = setup
    assert verify(sb, res, problem=problem)["verdict"] == "UNVERIFIED"
    assert verify(sb, res)["verdict"] == "UNVERIFIED"
    assert check(verify(sb, res), "tightening")["status"] == "skip"


def test_reference_failure():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)
    res["y"][2][0] *= 1.001
    cert = verify(sb, res, problem=problem, rerun=s2.rk4_solve)
    assert cert["verdict"] == "FAILED"
    ref = check(cert, "reference")
    assert ref["status"] == "fail" and ref["value"] > 1.0 and ref["bound"] == 1.0


def test_reference_requires_matching_problem():
    problem = load("harmonic_problem.json")
    other = load("chain_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    cert = verify(sb, s2.rk4_solve(sb), problem=other, rerun=s2.rk4_solve)
    assert check(cert, "reference")["status"] == "fail"
    assert cert["verdict"] == "FAILED"


def test_reference_no_coincident_time_fails():
    problem = load("harmonic_problem.json")
    problem["reference"]["t"] = [0.123, 0.456]
    problem["reference"]["y"] = [[0, 0], [0, 0]]
    sb = json.dumps(problem["spec"]).encode()
    cert = verify(sb, s2.rk4_solve(sb), problem=problem, rerun=s2.rk4_solve)
    assert check(cert, "reference")["status"] == "fail"


def test_invariant_failure_alone():
    problem = load("chain_problem.json")
    problem["invariants"][0]["value"] = 1.001  # wrong conserved value, reference still fine
    sb = json.dumps(problem["spec"]).encode()
    cert = verify(sb, s2.rk4_solve(sb), problem=problem, rerun=s2.rk4_solve)
    assert check(cert, "reference")["status"] == "pass"
    assert check(cert, "invariants")["status"] == "fail"
    assert cert["verdict"] == "FAILED"


def test_tightening_failure_alone():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)
    res["y"][1][0] += 1e-5
    cert = verify(sb, res, rerun=s2.rk4_solve)  # ad hoc: only tightening can see it
    assert check(cert, "tightening")["status"] == "fail"
    assert cert["verdict"] == "FAILED"


def test_loose_solver_caught_by_tightening():
    problem = load("harmonic_problem.json")
    spec = problem["spec"]
    loose = dict(spec, rtol=1e-3)
    res = s2.rk4_solve(json.dumps(loose).encode())
    res["input_sha256"] = hashlib.sha256(json.dumps(spec).encode()).hexdigest()
    cert = verify(json.dumps(spec).encode(), res, rerun=s2.rk4_solve)
    assert cert["verdict"] == "FAILED"


# -------------------------------------------------------------- shape and hashes

@pytest.mark.parametrize("mutate", [
    lambda r: r["y"].__setitem__(0, [float("nan"), 0.0]),
    lambda r: r["y"].__setitem__(0, [float("inf"), 0.0]),
    lambda r: r["y"].__setitem__(0, [True, 0.0]),
    lambda r: r.update(t=r["t"][:-1], y=r["y"][:-1]),
    lambda r: r["y"].__setitem__(0, [1.0]),
    lambda r: r["t"].__setitem__(1, r["t"][1] + 1e-9),
    lambda r: r.update(id="other"),
    lambda r: r.update(y="nope"),
])
def test_shape_failures(mutate):
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)
    mutate(res)
    cert = verify(sb, res, problem=problem, rerun=s2.rk4_solve)
    assert check(cert, "shape")["status"] == "fail"
    assert cert["verdict"] == "FAILED"


def test_hash_mismatch_wrong_spec_bytes():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    spec_b = dict(problem["spec"], y0=[1.0, 0.001])
    res_b = s2.rk4_solve(json.dumps(spec_b).encode())
    cert = verify(sb, res_b, rerun=s2.rk4_solve)
    assert check(cert, "hashes")["status"] == "fail"
    assert check(cert, "shape")["status"] == "pass"
    assert cert["verdict"] == "FAILED"


def test_input_hash_is_over_exact_bytes():
    problem = load("harmonic_problem.json")
    pretty = json.dumps(problem["spec"], indent=2).encode()
    compact = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(compact)
    assert check(verify(compact, res, rerun=s2.rk4_solve), "hashes")["status"] == "pass"
    assert check(verify(pretty, res, rerun=s2.rk4_solve), "hashes")["status"] == "fail"


def test_canonical_hashing_stable():
    a = {"b": 1, "a": [1.5, "é"], "c": {"y": 2, "x": 1}}
    b = {"c": {"x": 1, "y": 2}, "a": [1.5, "é"], "b": 1}
    assert verifier.canonical_json(a) == verifier.canonical_json(b)
    assert verifier.canonical_json(a) == '{"a":[1.5,"é"],"b":1,"c":{"x":1,"y":2}}'.encode("utf-8")
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)
    c1 = verify(sb, res, rerun=s2.rk4_solve)
    c2 = verify(sb, json.loads(json.dumps(res)), rerun=s2.rk4_solve)
    assert c1["result_sha256"] == c2["result_sha256"] == hashlib.sha256(verifier.canonical_json(res)).hexdigest()


# ------------------------------------------------------------------ fail closed

def test_unusable_results_are_unverified():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    good = s2.rk4_solve(sb)
    cases = [None, "text", [], {}, dict(good, status="error", error="boom"),
             dict(good, schema="other/1"), dict(good, y=[[0.0, 0.0]] * 900000)]
    for bad in cases:
        cert = verify(sb, bad, problem=problem, rerun=s2.rk4_solve)
        assert cert["verdict"] == "UNVERIFIED", bad if not isinstance(bad, dict) else list(bad)
        assert cert["caveat"] == CAVEAT


def test_oversize_guard_boundary():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    big = dict(s2.rk4_solve(sb), pad="x" * (verifier.MAX_RESULT_BYTES + 1))
    assert verify(sb, big)["verdict"] == "UNVERIFIED"


@pytest.mark.parametrize("bad", [b"", b"not json", b"\xff\xfe", b"[]", b'{"schema":"x"}', None, "str"])
def test_bad_spec_bytes_unverified(bad):
    problem = load("harmonic_problem.json")
    res = s2.rk4_solve(json.dumps(problem["spec"]).encode())
    cert = verify(bad, res)
    assert cert["verdict"] == "UNVERIFIED"
    assert cert["caveat"] == CAVEAT


def test_oversize_spec_unverified():
    problem = load("harmonic_problem.json")
    res = s2.rk4_solve(json.dumps(problem["spec"]).encode())
    assert verify(b" " * (verifier.MAX_SPEC_BYTES + 1), res)["verdict"] == "UNVERIFIED"


def test_rerun_exceptions_and_garbage_fail_closed():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)

    def boom(_):
        raise RuntimeError("solver crashed")

    for rr in (boom, lambda b: None, lambda b: {"status": "error"}, lambda b: dict(res, y=res["y"][:-1])):
        cert = verify(sb, res, problem=problem, rerun=rr)
        assert check(cert, "tightening")["status"] == "fail"
        assert cert["verdict"] == "FAILED"


def test_weird_problem_and_invariants_do_not_crash():
    problem = load("harmonic_problem.json")
    sb = json.dumps(problem["spec"]).encode()
    res = s2.rk4_solve(sb)
    assert verify(sb, res, problem="junk", rerun=s2.rk4_solve)["verdict"] == "FAILED"
    bad = copy.deepcopy(problem)
    bad["invariants"] = [{"name": "x", "kind": "mystery", "tol": 1}]
    assert check(verify(sb, res, problem=bad, rerun=s2.rk4_solve), "invariants")["status"] == "fail"
    bad["invariants"] = [{"name": "x", "kind": "linear", "weights": [1.0], "value": 0, "tol": 1}]
    assert check(verify(sb, res, problem=bad, rerun=s2.rk4_solve), "invariants")["status"] == "fail"
    bad["invariants"] = [{"name": "x", "kind": "linear", "weights": [1, 1], "value": 0, "tol": "a"}]
    assert check(verify(sb, res, problem=bad, rerun=s2.rk4_solve), "invariants")["status"] == "fail"


def test_caveat_in_every_certificate(setup):
    problem, sb, res = setup
    bad = copy.deepcopy(res)
    bad["y"][0][0] = float("nan")
    for cert in (verify(sb, res, problem=problem, rerun=s2.rk4_solve), verify(sb, bad), verify(sb, None),
                 verify(b"x", res)):
        assert cert["caveat"] == CAVEAT
        assert "never means 'correct'" in cert["caveat"]
        assert json.dumps(cert, allow_nan=False)  # certificate is strict JSON


# ------------------------------------------------------------------- invariants

def _inv_cert(kind_fields, y, spec_extra=None):
    spec = copy.deepcopy(load("harmonic_problem.json")["spec"])
    spec["y0"] = list(y[0]) if isinstance(y[0], list) else spec["y0"]
    problem = {"schema": "hivemind-solver-problem/1", "spec": spec, "invariants": [kind_fields]}
    return verifier._check_invariants(spec, {"y": y}, problem)


def test_invariant_kinds():
    assert _inv_cert({"name": "n", "kind": "linear", "weights": [1, 1], "value": 3.0, "tol": 1e-9},
                     [[1.0, 2.0], [2.0, 1.0]])["status"] == "pass"
    assert _inv_cert({"name": "n", "kind": "linear", "weights": [1, 1], "value": 3.0, "tol": 1e-9},
                     [[1.0, 2.0], [2.0, 1.1]])["status"] == "fail"
    q = {"name": "q", "kind": "quadratic_form", "matrix": [[1, 0], [0, 4]], "value": 5.0, "tol": 1e-9}
    assert _inv_cert(q, [[1.0, 1.0], [math.sqrt(5), 0.0]])["status"] == "pass"
    assert _inv_cert(dict(q, kind="energy", value=2.5), [[1.0, 1.0]])["status"] == "pass"
    lv = {"name": "v", "kind": "lv_first_integral", "alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 0.5,
          "tol": 1e-9}
    x, y = 2.0, 2.0
    assert _inv_cert(lv, [[x, y], [x, y]])["status"] == "pass"
    assert _inv_cert(lv, [[x, y], [x, y * 1.01]])["status"] == "fail"


def test_lv_canonical_params_layout_and_aliases():
    # canonical (library) layout: params object
    canon = {"name": "v", "kind": "lv_first_integral",
             "params": {"alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 0.5}, "tol": 1e-9}
    assert _inv_cert(canon, [[2.0, 2.0], [2.0, 2.0]])["status"] == "pass"
    assert _inv_cert(canon, [[2.0, 2.0], [2.0, 2.02]])["status"] == "fail"
    # params take precedence over a conflicting top-level alias
    both = dict(canon, alpha=99.0)
    assert _inv_cert(both, [[2.0, 2.0], [2.0, 2.0]])["status"] == "pass"
    # a malformed params object fails closed
    with pytest.raises(verifier._CheckError):
        _inv_cert(dict(canon, params=[1, 2]), [[2.0, 2.0]])
    # missing parameter fails closed
    with pytest.raises(verifier._CheckError):
        _inv_cert(dict(canon, params={"alpha": 1.0}), [[2.0, 2.0]])


def test_library_invariant_layouts_are_understood():
    """Every invariant written by solvers/problems is evaluated (not rejected) at its own y0."""
    import glob
    import os
    root = os.path.join(os.path.dirname(__file__), "..", "solvers", "problems")
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        pytest.skip("problem library not present")
    n_checked = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            prob = json.load(fh)
        spec = prob["spec"]
        for inv in prob.get("invariants", []):
            fn = verifier._INVARIANT_KINDS[inv["kind"]]
            got = fn(inv, spec, [float(v) for v in spec["y0"]])
            assert abs(got - inv["value"]) <= inv["tol"], (prob["id"], inv["name"], got)
            n_checked += 1
    assert n_checked >= 9


def test_lv_nonpositive_population_fails_closed():
    lv = {"name": "v", "kind": "lv_first_integral", "alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 0.5,
          "tol": 1e-9, "value": 0.0}
    with pytest.raises(verifier._CheckError):
        _inv_cert(lv, [[-1.0, 1.0]])


# -------------------------------------------------------------- decision function

def _checks(**kw):
    base = {n: {"name": n, "status": "pass", "detail": "d", "value": None, "bound": None}
            for n in verifier._CHECK_NAMES}
    for k, v in kw.items():
        base[k] = dict(base[k], status=v, detail="no invariants defined" if (k == "invariants" and v == "skip") else "d")
    return base


@pytest.mark.parametrize("kw,usable,has_problem,expected", [
    ({}, False, True, "UNVERIFIED"),
    ({}, True, True, "VERIFIED"),
    ({"reference": "skip"}, True, False, "AUDITED"),
    ({"reference": "skip", "invariants": "skip"}, True, False, "AUDITED"),
    ({"invariants": "skip"}, True, True, "VERIFIED"),
    ({"tightening": "skip"}, True, True, "UNVERIFIED"),
    ({"tightening": "skip", "reference": "skip"}, True, False, "UNVERIFIED"),
    ({"reference": "skip"}, True, True, "UNVERIFIED"),
    ({"reference": "fail"}, True, True, "FAILED"),
    ({"hashes": "fail", "tightening": "skip"}, True, False, "FAILED"),
    ({"shape": "fail"}, True, True, "FAILED"),
    ({"invariants": "fail", "reference": "skip"}, True, False, "FAILED"),
    ({"tightening": "fail"}, True, False, "FAILED"),
])
def test_decision_function(kw, usable, has_problem, expected):
    assert decide(_checks(**kw), usable=usable, has_problem=has_problem)[0] == expected


def test_decision_invariants_skipped_for_other_reason_is_unverified():
    c = _checks()
    c["invariants"] = dict(c["invariants"], status="skip", detail="shape failed")
    assert decide(c, usable=True, has_problem=True)[0] == "UNVERIFIED"


# ------------------------------------------------------------------------ scoring

def test_wilson_hand_values():
    # 8/10 at 95%: textbook interval [0.4902, 0.9433]
    assert scoring.wilson_lower(8, 10) == pytest.approx(0.4902, abs=5e-4)
    # 10/10: lower = n/(n+z^2) = 10/13.8416
    assert scoring.wilson_lower(10, 10) == pytest.approx(10 / (10 + 1.959963984540054 ** 2), abs=1e-9)
    assert scoring.wilson_lower(0, 10) == 0.0
    assert scoring.wilson_lower(0, 0) == 0.0
    with pytest.raises(ValueError):
        scoring.wilson_lower(11, 10)


def test_wilson_matches_queenbee_if_available():
    dance = pytest.importorskip("hivemind.queenbee.dance")
    for s, n in [(8, 10), (3, 7), (50, 60), (0, 5)]:
        assert scoring.wilson_lower(s, n) == pytest.approx(dance.wilson_lower(s, n), abs=1e-6)


def _without_tols(**extra):
    spec = {k: v for k, v in load("harmonic_problem.json")["spec"].items() if k not in ("rtol", "atol")}
    spec.update(extra)
    return json.dumps(spec).encode()


def cli_like_solve(spec_bytes):
    """Stand-in for the CLI: applies the CLI defaults, hashes the exact submitted bytes."""
    spec = json.loads(spec_bytes)
    spec.setdefault("rtol", 1e-8)
    spec.setdefault("atol", 1e-10)
    if isinstance(spec["atol"], list):
        spec["atol"] = min(spec["atol"])
    res = s2.rk4_solve(json.dumps(spec).encode())
    res["input_sha256"] = hashlib.sha256(spec_bytes).hexdigest()
    return res


def test_spec_without_rtol_atol_uses_cli_defaults():
    # rtol/atol are optional in a spec; the CLI defaults are 1e-8 / 1e-10.
    sb = _without_tols()
    seen = []

    def rerun(b):
        seen.append(json.loads(b))
        return cli_like_solve(b)

    cert = verify(sb, cli_like_solve(sb), rerun=rerun)
    assert check(cert, "tightening")["status"] != "fail", cert
    assert cert["verdict"] == "AUDITED", cert
    assert seen[0]["rtol"] == pytest.approx(1e-10)
    assert seen[0]["atol"] == pytest.approx(1e-12)


def test_spec_with_only_one_tolerance_given():
    n = len(load("harmonic_problem.json")["spec"]["y0"])
    for extra in ({"rtol": 1e-6}, {"atol": [1e-9] * n}, {"atol": 1e-9}):
        sb = _without_tols(**extra)
        cert = verify(sb, cli_like_solve(sb), rerun=cli_like_solve)
        assert check(cert, "tightening")["status"] != "fail", (extra, cert)


def test_recruitment_cold_start_and_record():
    rec = scoring.track_record(["VERIFIED"] * 4)
    assert rec["recruitment_weight"] == 0.0 and "cold start" in rec["recruitment_note"]
    assert rec["wilson_lower_95"] > 0.0  # the bound is reported even though the weight is 0
    rec = scoring.track_record(["VERIFIED"] * 8 + ["FAILED", "AUDITED"])
    assert rec["counts"] == {"VERIFIED": 8, "AUDITED": 1, "FAILED": 1, "UNVERIFIED": 0}
    assert rec["n"] == 10 and rec["verified_rate"] == 0.8
    assert rec["recruitment_weight"] == pytest.approx(0.4902, abs=5e-4)
    assert scoring.track_record([])["recruitment_weight"] == 0.0
    with pytest.raises(ValueError):
        scoring.track_record(["MAYBE"])


# --------------------------------------------------------------------- S-2 runner

def test_runner_reduced_end_to_end(tmp_path):
    rep = s2.run(s2.SEED, 6, 3)
    assert rep["run_valid"] is True
    assert rep["controls"]["positive_library_verified"] and rep["controls"]["negative_hashes_flagged"]
    assert rep["acceptance"]["A2_no_clean_failed"] == "PASS"
    assert rep["acceptance"]["A3_caveat_on_every_certificate"] == "PASS"
    assert rep["clean"]["verdicts_by_mode"] == {"LIBRARY": {"VERIFIED": 6}, "ADHOC": {"AUDITED": 6}}
    for f in ("F1_wrong_matrix_entry", "F2_perturbed_trajectory", "F4_nan_injected", "F5_truncated_output"):
        for m in ("LIBRARY", "ADHOC"):
            assert rep["faults"][f]["modes"][m]["detected"] == 3
    assert rep["acceptance"]["A1_all_injected_failed_or_unverified"] in ("PASS", "FAIL")
    assert s2.run(s2.SEED, 6, 3)["receipt_sha256"] == rep["receipt_sha256"]  # deterministic
    out = tmp_path / "r.json"
    assert s2.main(["--clean-n", "3", "--n-per-fault", "2", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["faults"]["F5_truncated_output"]["N"] == 2


def test_committed_report_receipt_is_consistent():
    path = ROOT / "artifacts" / "experiments" / "S2_fault_injection_report.json"
    if not path.exists():
        pytest.skip("report not generated")
    rep = json.loads(path.read_text())
    receipt = rep.pop("receipt_sha256")
    body = json.dumps(rep, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(body).hexdigest() == receipt
    assert all(f["N"] >= 30 for f in rep["faults"].values())
