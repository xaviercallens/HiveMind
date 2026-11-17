# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Solver Bee problem library and its tools (stdlib only; regeneration tests skip without mpmath/scipy)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "solvers" / "tools"
PROBLEMS = ROOT / "solvers" / "problems"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = _load("check_problems")


def _docs():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PROBLEMS.glob("*.json"))}


def test_library_has_at_least_twelve_valid_problems():
    n, problems = check.check_dir(PROBLEMS, 12)
    assert n >= 12
    assert problems == []


def test_library_covers_required_families():
    docs = _docs()
    need = {"harmonic_undamped", "linear_2x2", "linear_5x5", "decay_chain", "robertson", "van_der_pol_mu1",
            "van_der_pol_mu100", "brusselator", "lotka_volterra", "stiff_chain"}
    assert need <= set(docs)
    kinds = {d["spec"]["problem"]["kind"] for d in docs.values()}
    assert kinds == {"named", "linear", "mass_action"}


def test_every_reference_records_method_tool_versions_and_cross_check():
    for pid, d in _docs().items():
        r = d["reference"]
        assert r["tool"] == "solvers/tools/make_references.py", pid
        assert {"python", "mpmath", "scipy", "numpy"} <= set(r["tool_versions"]), pid
        assert r["cross_check"]["method"] and r["cross_check"]["max_abs_diff"] >= 0, pid
        assert r["method"] != r["cross_check"]["method"], pid


def test_reference_is_100x_tighter_than_spec_tolerance():
    for pid, d in _docs().items():
        s, r = d["spec"], d["reference"]
        tol_min = min(s["rtol"] * abs(v) + s["atol"] for row in r["y"] for v in row)
        assert r["max_abs_error_bound"] * 100 <= tol_min, pid


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda d: d.__setitem__("schema", "nope"), "schema"),
        (lambda d: d["spec"].__setitem__("bogus", 1), "unknown key"),
        (lambda d: d["spec"].__setitem__("t_out", [2.0, 1.0]), "t_out"),
        (lambda d: d["spec"].__setitem__("y0", [1.0]), "y0"),
        (lambda d: d["spec"].__setitem__("rtol", 0.0), "rtol"),
        (lambda d: d["spec"].__setitem__("method", "rk4"), "method"),
        (lambda d: d["reference"].__setitem__("t", [9.0]), "reference.t"),
        (lambda d: d["reference"].__setitem__("y", [[float("nan")] * 2] * 5), "reference.y"),
        (lambda d: d["reference"].__setitem__("max_abs_error_bound", 1e-3), "100x"),
        (lambda d: d["reference"].pop("cross_check"), "cross_check"),
        (lambda d: d["tolerance_policy"].__setitem__("max_rel_err_factor", -1), "max_rel_err_factor"),
        (lambda d: d.__setitem__("extra", 1), "unknown top-level"),
        (lambda d: d["invariants"][0].__setitem__("value", 5.0), "invariant"),
    ],
)
def test_checker_rejects_corrupted_problem(mutate, fragment):
    d = copy.deepcopy(_docs()["harmonic_undamped"])
    mutate(d)
    errs = check.validate_problem(d, "harmonic_undamped")
    assert any(fragment in e for e in errs), errs


def test_checker_rejects_mass_action_problems():
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"][0]["reactants"] = {"Z": 1}
    assert any("unknown species" in e for e in check.validate_problem(d, "decay_chain"))
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"][0]["reactants"] = {"A": 0}
    assert any("coefficient" in e for e in check.validate_problem(d, "decay_chain"))
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"] = d["spec"]["problem"]["reactions"] * 20
    assert any("reactions" in e for e in check.validate_problem(d, "decay_chain"))


def test_checker_main_reports_too_few_problems(tmp_path):
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    n, problems = check.check_dir(tmp_path, 12)
    assert n == 1 and problems


def test_run_problems_evaluate_flags_out_of_policy():
    run = _load("run_problems")
    d = _docs()["harmonic_undamped"]
    good = {"status": "ok", "y": d["reference"]["y"], "stats": {"steps": 1, "rhs_evals": 1},
            "solver": {"method": "bdf"}}
    assert run.evaluate(d, good)["ok"]
    bad = copy.deepcopy(good)
    bad["y"][0][0] += 1e-3
    assert not run.evaluate(d, bad)["ok"]
    assert not run.evaluate(d, {"status": "error", "error": "x"})["ok"]


def test_tools_contain_no_dynamic_code_execution():
    for p in TOOLS.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for bad in ("eval(", "exec(", "pickle", "shell=True", "os.system"):
            assert bad not in text.replace("evaluate(", "").replace("eval_invariant(", ""), (p.name, bad)


# ---------------------------------------------------------------- regeneration (needs mpmath + scipy + numpy)

FAST_IDS = ["harmonic_undamped", "harmonic_damped", "linear_2x2", "linear_5x5", "decay_chain",
            "reversible_network", "stiff_chain"]


def _strip_versions(doc):
    # Drop the fields that legitimately depend on the installed numpy/scipy versions (the cross-check
    # disagreement is float noise of the second method); the exact reference values must still match.
    doc = copy.deepcopy(doc)
    for k in ("tool_versions", "cross_check", "max_abs_error_bound"):
        doc["reference"].pop(k)
    return doc


def test_regenerating_exact_references_reproduces_committed_files(tmp_path):
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    assert mk.main(["--ids", *FAST_IDS, "--out", str(tmp_path)]) == 0
    docs = _docs()
    for pid in FAST_IDS:
        new = json.loads((tmp_path / f"{pid}.json").read_text(encoding="utf-8"))
        assert _strip_versions(new) == _strip_versions(docs[pid]), pid
    # and the regenerated files pass the checker
    # (only these ids exist in tmp_path, so use the per-document validator)
    for pid in FAST_IDS:
        new = json.loads((tmp_path / f"{pid}.json").read_text(encoding="utf-8"))
        assert check.validate_problem(new, pid) == []


def test_make_references_lists_every_committed_problem():
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    assert sorted(e["id"] for e in mk.problems()) == sorted(_docs())
    for e in mk.problems():
        assert e["spec"] == _docs()[e["id"]]["spec"], e["id"]


# ---------------------------------------------------------------- held-out set (frozen before any solver run)

HELDOUT = PROBLEMS / "heldout"
freeze = _load("freeze_heldout")


def _heldout_docs():
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(HELDOUT.glob("*.json")) if p.name != freeze.MANIFEST}


def test_heldout_set_has_at_least_fourteen_valid_new_problems():
    n, problems = check.check_dir(HELDOUT, 14)
    assert n >= 14  # FROZEN.json is not counted as a problem by the checker
    assert problems == []
    assert not (set(_heldout_docs()) & set(_docs())), "held-out ids must be new"


def test_heldout_set_covers_the_required_physics_and_horizons():
    docs = _heldout_docs()
    need = {"pendulum_small_angle", "pendulum_large_angle", "pendulum_damped", "projectile_no_drag", "projectile_drag",
            "kepler_circular", "kepler_eccentric", "kepler_eccentric_long", "logistic_growth", "sir_epidemic",
            "seir_epidemic", "series_rlc", "spring_mass_damper", "stiff_reaction", "harmonic_long",
            "coupled_oscillators_long"}
    assert need <= set(docs)
    kinds = {d["spec"]["problem"]["kind"] for d in docs.values()}
    assert kinds == {"named", "linear", "mass_action"}
    fams = {d["spec"]["problem"].get("family") for d in docs.values()} - {None}
    assert {"pendulum", "projectile_drag", "kepler", "logistic", "harmonic"} <= fams
    horizons = sorted(d["spec"]["t_out"][-1] for d in docs.values())
    assert horizons[0] <= 10 and horizons[-1] >= 1000, horizons  # short and long horizons are both present
    assert sum(h >= 100 for h in horizons) >= 6


def test_heldout_policy_is_the_library_policy_and_references_are_cross_checked():
    for pid, d in _heldout_docs().items():
        assert d["tolerance_policy"] == {"max_rel_err_factor": 100.0}, pid
        r = d["reference"]
        assert r["method"] != r["cross_check"]["method"], pid
        assert r["tool"] == "solvers/tools/make_references.py", pid
        tol_min = min(d["spec"]["rtol"] * abs(v) + d["spec"]["atol"] for row in r["y"] for v in row)
        assert r["max_abs_error_bound"] * 100 <= tol_min, pid


def test_heldout_files_match_their_frozen_sha256():
    """Fails if any frozen file changes, goes missing, or an unlisted problem file appears."""
    assert freeze.verify(HELDOUT) == []
    manifest = json.loads((HELDOUT / freeze.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema"] == "hivemind-solver-heldout-frozen/1"
    assert len(manifest["files"]) >= 14
    assert set(manifest["files"]) == {p.name for p in HELDOUT.glob("*.json") if p.name != freeze.MANIFEST}


def test_freeze_verification_detects_edits_removals_and_additions(tmp_path):
    import shutil

    work = tmp_path / "heldout"
    shutil.copytree(HELDOUT, work)
    assert freeze.verify(work) == []
    victim = work / "kepler_circular.json"
    victim.write_text(victim.read_text(encoding="utf-8").replace('"rtol": 1e-08', '"rtol": 1e-07'), encoding="utf-8")
    assert any("kepler_circular.json" in e and "differs" in e for e in freeze.verify(work))
    shutil.copy(HELDOUT / "kepler_circular.json", victim)
    assert freeze.verify(work) == []
    (work / "pendulum_damped.json").unlink()
    assert any("missing" in e for e in freeze.verify(work))
    shutil.copy(HELDOUT / "pendulum_damped.json", work / "pendulum_damped.json")
    (work / "extra.json").write_text("{}", encoding="utf-8")
    assert any("not listed" in e for e in freeze.verify(work))


def test_energy_invariant_kind_is_checked_and_rejects_corruption():
    d = copy.deepcopy(_heldout_docs()["kepler_eccentric"])
    assert any(i["kind"] == "energy" for i in d["invariants"])
    assert check.validate_problem(d, "kepler_eccentric") == []
    bad = copy.deepcopy(d)
    next(i for i in bad["invariants"] if i["kind"] == "energy")["value"] = -0.1
    assert any("invariant" in e for e in check.validate_problem(bad, "kepler_eccentric"))
    bad = copy.deepcopy(d)
    next(i for i in bad["invariants"] if i["kind"] == "energy")["model"] = "nope"
    assert any("malformed" in e for e in check.validate_problem(bad, "kepler_eccentric"))


def test_checker_knows_the_science_families_and_their_dimensions():
    base = copy.deepcopy(_heldout_docs()["logistic_growth"])
    bad = copy.deepcopy(base)
    bad["spec"]["problem"]["params"] = {"r": 0.8, "bogus": 1.0}
    assert any("unknown parameter" in e for e in check.validate_problem(bad, "logistic_growth"))
    bad = copy.deepcopy(base)
    bad["spec"]["y0"] = [2.0, 1.0]
    assert any("y0" in e for e in check.validate_problem(bad, "logistic_growth"))
    assert check.FIXED_DIM["kepler"] == check.FIXED_DIM["projectile_drag"] == 4


def test_run_problems_tightened_spec_divides_both_tolerances():
    run = _load("run_problems")
    s = {"rtol": 1e-8, "atol": 1e-10, "y0": [1.0], "t_out": [1.0]}
    t = run.tightened_spec(s, 100.0)
    assert t["rtol"] == pytest.approx(1e-10) and t["atol"] == pytest.approx(1e-12)
    assert t["y0"] == s["y0"] and s["rtol"] == 1e-8  # the original is untouched
    t = run.tightened_spec({"rtol": 1e-6, "atol": [1e-9, 2e-9]}, 10.0)
    assert t["atol"] == pytest.approx([1e-10, 2e-10])


def test_run_problems_skips_the_freeze_manifest():
    run = _load("run_problems")
    assert freeze.MANIFEST in check.NON_PROBLEM_FILES
    assert hasattr(run, "NON_PROBLEM_FILES")


def test_regenerating_heldout_exact_references_reproduces_committed_files(tmp_path):
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    ids = ["projectile_no_drag", "logistic_growth", "series_rlc", "spring_mass_damper", "coupled_oscillators_long",
           "harmonic_long", "pendulum_small_angle"]
    assert mk.main(["--heldout", "--ids", *ids, "--out", str(tmp_path)]) == 0
    docs = _heldout_docs()
    for pid in ids:
        new = json.loads((tmp_path / f"{pid}.json").read_text(encoding="utf-8"))
        assert _strip_versions(new) == _strip_versions(docs[pid]), pid


def test_make_references_heldout_specs_equal_the_committed_specs():
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    entries = mk.heldout_problems()
    assert sorted(e["id"] for e in entries) == sorted(_heldout_docs())
    for e in entries:
        assert e["spec"] == _heldout_docs()[e["id"]]["spec"], e["id"]
