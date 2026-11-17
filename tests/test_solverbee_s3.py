# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for experiment S-3: frozen question set, scoring rule, baseline arm on a reduced set, analysis. CPU only, deterministic, no model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.solverbee import science_bee as sb  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("s3_runner", "scripts/run_experiment_S3_formulation.py")
analyze = _load("s3_analyze", "scripts/s3_analyze.py")
BINARY = os.environ.get("HIVEMIND_SOLVER_BIN", runner.DEFAULT_BINARY)
QDOC = json.loads((ROOT / "artifacts/experiments/S3_questions.json").read_text(encoding="utf-8"))
Q = {q["id"]: q for q in QDOC["questions"]}


class PerfectPipeline:
    """Stand-in solver+verifier that returns the expected answers with verdict AUDITED (tests the scoring and the runner logic only)."""

    def __init__(self, ans):
        self.ans = ans

    def run(self, spec_bytes):
        ans = self.ans
        return {"verdict": "AUDITED", "result": {"status": "ok", "t": ans["t"], "y": ans["y"]}, "certificate": {"checks": [{"name": "shape", "status": "pass"}], "reasons": []}, "wall_ms": 0}


def test_question_set_is_frozen_and_well_formed():
    raw = (ROOT / "artifacts/experiments/S3_questions.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (ROOT / "artifacts/experiments/S3_questions.sha256").read_text().split()[0]
    qs = QDOC["questions"]
    assert len(qs) >= 40 and QDOC["by_lang"] == {"en": 24, "fr": 16}
    assert set(QDOC["by_tier"]) == {"A", "B", "C"} and all(v > 0 for v in QDOC["by_tier"].values())
    assert {q["type"] for q in qs} == {"pendulum", "projectile", "kepler", "logistic", "sir", "seir", "rlc", "msd", "chain", "stiff"}
    assert len({q["id"] for q in qs}) == len(qs) and len({q["text"] for q in qs}) == len(qs)
    for q in qs:
        sb.validate_spec(q["expected_spec"])
        a = q["expected_answers"]
        assert a["t"] == q["expected_spec"]["t_out"] and len(a["y"]) == len(a["t"]) and all(len(r) == len(q["expected_spec"]["y0"]) for r in a["y"])
    assert runner.load_questions()["n_questions"] == len(qs)


def test_closed_form_references_match_the_stored_answers():
    pytest.importorskip("mpmath")
    ref = _load("s3_reference", "artifacts/experiments/s3_reference.py")
    for qid in ("q06", "q10", "q12", "q13", "q03"):
        q = Q[qid]
        y0, ts = q["expected_spec"]["y0"], q["expected_answers"]["t"]
        p = {"q06": dict(r=0.5, K=100.0), "q10": dict(R=10.0, L=0.5, C=0.01, V=5.0), "q12": dict(rates=[1.0, 0.5]), "q13": dict(rates=[1000.0, 1.0]), "q03": dict(g=9.81, k=0.0)}[qid]
        got = ref.REFERENCE[q["type"]](p, y0, ts)
        assert got == q["expected_answers"]["y"]


def test_agreement_rule_is_the_fixed_policy():
    exp = {"t": [1.0, 2.0], "y": [[1.0, 0.0], [2.0, 1e-3]]}
    ok = {"status": "ok", "t": [1.0, 2.0], "y": [[1.0, 0.0], [2.0, 1e-3]]}
    assert runner.agreement(ok, exp)
    near = {"status": "ok", "t": [1.0, 2.0], "y": [[1.0 + 5e-7, 5e-11], [2.0, 1e-3]]}   # inside 100*(1e-8*1+1e-10) = 1.01e-6
    assert runner.agreement(near, exp)
    far = {"status": "ok", "t": [1.0, 2.0], "y": [[1.0 + 5e-6, 0.0], [2.0, 1e-3]]}
    assert not runner.agreement(far, exp)
    assert not runner.agreement(dict(ok, status="error"), exp) and not runner.agreement(None, exp)
    assert not runner.agreement(dict(ok, t=[1.0, 3.0]), exp)                            # other time axis (e.g. wrong unit)
    assert not runner.agreement(dict(ok, y=[[1.0, 0.0]]), exp)
    assert not runner.agreement(dict(ok, y=[[1.0, 0.0], [2.0, float("nan")]]), exp)
    assert not runner.agreement(dict(ok, y=[[1.0], [2.0]]), exp)


def test_baseline_arm_on_a_reduced_set_with_a_perfect_pipeline():
    ids = ["q10", "q12", "q15", "q08"]
    recs = [runner.run_question(Q[i], sb.ScienceBee(sb.BaselineBackend(), PerfectPipeline(Q[i]["expected_answers"])), ["linear"]) for i in ids]
    by = {r["id"]: r for r in recs}
    assert by["q10"]["correct"] and by["q10"]["exact_match"] and by["q12"]["correct"]
    assert by["q15"]["verdict"] == "COULD_NOT_FORMULATE" and not by["q15"]["correct"]     # paraphrased tier B: the keyword baseline gives up
    assert by["q08"]["verdict"] == "COULD_NOT_FORMULATE"
    assert all(r["answer_contains_limits_sentence"] for r in recs)
    s = runner.summarise(recs)
    assert s["overall"]["correct"] == 2 and s["by_tier"]["A"]["n"] == 3 and s["by_tier"]["B"]["n"] == 1 and s["all_answers_contain_limits_sentence"]


def test_wrong_formulation_is_not_correct_even_when_audited():
    q = Q["q10"]
    wrong = json.loads(json.dumps(q["expected_spec"]))
    wrong["problem"]["A"][1][1] = -50.0
    be = sb.FakeBackend([json.dumps(wrong)])

    class Echo:
        def run(self, spec_bytes):   # pretend the solver returned something that disagrees, with an AUDITED verdict
            return {"verdict": "AUDITED", "result": {"status": "ok", "t": q["expected_answers"]["t"], "y": [[v + 1.0 for v in r] for r in q["expected_answers"]["y"]]},
                    "certificate": {"checks": [], "reasons": []}, "wall_ms": 0}
    rec = runner.run_question(q, sb.ScienceBee(be, Echo()), [])
    assert rec["verdict"] == "AUDITED" and not rec["agreement"] and not rec["correct"] and rec["audited_but_wrong"] and not rec["verified_but_wrong"] and not rec["exact_match"]


@pytest.mark.skipif(not Path(BINARY).exists(), reason="solver binary not built")
def test_baseline_and_fake_dry_run_end_to_end_with_the_real_binary(tmp_path):
    ids = "q09,q10,q12,q13,q14,q15"
    base_out, fake_out = tmp_path / "base.json", tmp_path / "fake.json"
    assert runner.main(["--arm", "baseline", "--binary", BINARY, "--ids", ids, "--out", str(base_out)]) == 0
    assert runner.main(["--arm", "model", "--model", "fake", "--binary", BINARY, "--ids", ids, "--out", str(fake_out)]) == 0
    base, fake = json.loads(base_out.read_text()), json.loads(fake_out.read_text())
    assert base["n_questions_run"] == 6 and base["reduced_set"] and base["binary"]["sha256"] and "harmonic" in base["binary"]["families"]
    got = {r["id"]: r for r in base["records"]}
    for qid in ("q09", "q10", "q12", "q13"):
        assert got[qid]["verdict"] in ("AUDITED", "VERIFIED") and got[qid]["correct"], qid
    assert got["q15"]["verdict"] == "COULD_NOT_FORMULATE"
    assert fake["is_dry_run_with_scripted_fake_backend"] and fake["summary"]["overall"]["exact_match"] == 6
    assert base["summary"]["all_answers_contain_limits_sentence"] and len(base["receipt_sha256"]) == 64
    assert analyze.main(["--baseline", str(base_out), "--model-cpu", str(fake_out)]) == 0


def test_mcnemar_exact_and_acceptance_rules():
    assert analyze.mcnemar_exact(0, 0) == 1.0 and analyze.mcnemar_exact(3, 3) == 1.0
    assert analyze.mcnemar_exact(0, 8) == pytest.approx(2 / 256)
    assert analyze.mcnemar_exact(1, 5) == pytest.approx(2 * (1 + 6) / 64)

    def doc(arm, correct, verified_wrong=False, limits=True):
        recs = [{"id": f"q{i}", "tier": "A", "correct": c, "verified_but_wrong": verified_wrong and i == 0, "audited_but_wrong": False, "answer_contains_limits_sentence": limits}
                for i, c in enumerate(correct)]
        return {"arm": arm, "model": "m", "device": "cpu", "records": recs, "binary": {"sha256": "x"}}
    base = doc("baseline", [True, True, False, False])
    assert analyze.rules(base, doc("model", [True, True, True, False]))["A1"]["verdict"] == "PASS"
    assert analyze.rules(base, doc("model", [False, True, True, True]))["A1"]["verdict"] == "FAIL"        # one regression
    assert analyze.rules(base, doc("model", [True, True, True, False], verified_wrong=True))["A3"]["verdict"] == "HARD FAILURE"
    assert analyze.rules(base, doc("model", [True, True, True, False], limits=False))["A4"]["verdict"] == "FAIL"
