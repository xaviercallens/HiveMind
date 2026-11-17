# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Science Bee: scripted fake backend, stub pipeline, deterministic, CPU only, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hivemind.solverbee import families as fam  # noqa: E402
from hivemind.solverbee import science_bee as sb  # noqa: E402

LIMITS = "The numbers solve the equations that were set up; whether those equations describe your real situation is not checked."
T = [1.0, 2.0, 5.0]
HOSTILE = "__imp" + "ort__('os')"  # built by concatenation so the repository scanner does not flag the fixture
GOOD = fam.pendulum_spec(0.5, 0.0, T, damping=0.1)


def j(spec) -> str:
    return json.dumps(spec)


class StubPipeline:
    """Returns a canned report; records the spec bytes it was given."""

    def __init__(self, verdict="AUDITED"):
        self.verdict, self.seen = verdict, []

    def run(self, spec_bytes):
        self.seen.append(spec_bytes)
        spec = json.loads(spec_bytes)
        n = len(spec["y0"])
        ys = [[0.1 * (i + 1) + 0.01 * k for k in range(n)] for i in range(len(spec["t_out"]))]
        cert = {"verdict": self.verdict, "checks": [{"name": "shape", "status": "pass"}, {"name": "hashes", "status": "pass"}, {"name": "reference", "status": "skip"},
                                                    {"name": "invariants", "status": "skip"}, {"name": "tightening", "status": "pass"}], "reasons": []}
        return {"verdict": self.verdict, "result": {"status": "ok", "t": spec["t_out"], "y": ys}, "certificate": cert, "wall_ms": 1}


def test_valid_first_time():
    b = sb.ScienceBee(sb.FakeBackend([j(GOOD)]), StubPipeline())
    a = b.ask("A pendulum with g_over_l = 9.81 and damping 0.1, theta0 = 0.5 rad. theta at t = 1, 2, 5 s?")
    assert a["formulated"] and a["attempts"] == 1 and a["retries"] == 0 and a["verdict"] == "AUDITED"
    assert a["spec_sha256"] == sb.spec_sha256(sb.with_house_tolerances(GOOD)) and a["spec"]["rtol"] == 1e-10 and a["spec"]["atol"] == 1e-12


def test_invalid_then_valid_feeds_back_the_exact_error():
    bad = dict(GOOD, bogus=1)
    be = sb.FakeBackend([j(bad), j(GOOD)])
    a = sb.ScienceBee(be, StubPipeline()).ask("pendulum question")
    assert a["formulated"] and a["retries"] == 1 and len(a["errors"]) == 1
    last = be.calls[1][-1]["content"]
    assert a["errors"][0] in last and "unknown field" in last
    assert be.calls[1][-2]["role"] == "assistant"


def test_three_retries_then_could_not_formulate_without_guessing():
    be = sb.FakeBackend([j(dict(GOOD, bogus=1))] * 4 + [j(GOOD)])
    pipe = StubPipeline()
    a = sb.ScienceBee(be, pipe).ask("pendulum question")
    assert not a["formulated"] and a["spec"] is None and a["attempts"] == 4 and a["retries"] == 3 and len(be.calls) == 4
    assert pipe.seen == [] and a["verdict"] is None
    assert "could not" in a["text"].lower()
    assert LIMITS in a["text"]


def test_hostile_output_with_code_in_strings_is_rejected_not_evaluated():
    evil = {"schema": "hivemind-solver-spec/1", "id": HOSTILE + ".system('echo hi')", "problem": {"kind": "named", "family": "pendulum", "params": {"g_over_l": "9.81*2"}},
            "y0": ["0.5", 0], "t_out": [1]}
    be = sb.FakeBackend([j(evil)] * 4)
    a = sb.ScienceBee(be, StubPipeline()).ask("x")
    assert not a["formulated"]
    for bad in ({"problem": {"kind": "named", "family": "pendulum", "params": {"g_over_l": HOSTILE}}},
                {"problem": {"kind": "named", "family": "pendulum", "params": {"exec": 1}}}, {"id": "a b"}, {"y0": [0.5, "1+1"]}, {"extra": "ev" + "al(1)"}):
        with pytest.raises(sb.SpecInvalid):
            sb.validate_spec(dict(GOOD, **bad))


def test_json_in_prose_and_fences_and_multiple_objects():
    assert sb.extract_json("Sure! Here is the spec:\n```json\n" + j(GOOD) + "\n```\nHope this helps {not json}.") == GOOD
    assert sb.extract_json('text with a } brace and "{" in prose ' + j(GOOD)) == GOOD
    with pytest.raises(sb.SpecInvalid, match="more than one"):
        sb.extract_json(j(GOOD) + " and also " + j(GOOD))
    for bad in ("", "no json here", '{"a": NaN}', '{"a": 1, "a": 2}', "{unbalanced"):
        with pytest.raises(sb.SpecInvalid):
            sb.extract_json(bad)


def test_validator_limits_and_finiteness():
    for mutate in (lambda s: s.update(t_out=[2.0, 1.0]), lambda s: s.update(t_out=[0.0]), lambda s: s.update(t_out=list(range(1, 1002))),
                   lambda s: s.update(y0=[0.5]), lambda s: s.update(y0=[0.5, float("inf")]), lambda s: s.update(rtol=2.0), lambda s: s.update(method="rk4"),
                   lambda s: s.update(max_steps=0), lambda s: s.update(schema="other/1"), lambda s: s.update(t0="0"), lambda s: s.update(y0=[True, 0])):
        s = json.loads(j(GOOD))
        mutate(s)
        with pytest.raises(sb.SpecInvalid):
            sb.validate_spec(s)
    bad_ma = dict(fam.sir_spec(0.5, 0.2, 0.99, 0.01, T), problem={"kind": "mass_action", "species": ["S"], "reactions": [{"reactants": {"S": 1}, "products": {"X": 1}, "k": 1}]}, y0=[1.0])
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec(bad_ma)
    big = fam.mass_action_spec(["A"], [{"reactants": {"A": 1}, "products": {}, "k": 1.0}], [1.0], [1.0], spec_id="x" * 64)
    big["t_out"] = [float(i + 1) for i in range(1000)]
    sb.validate_spec(big)
    big["atol"] = [1e-10] * 5000
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec(big)


def test_cannot_formulate_reply_is_respected():
    be = sb.FakeBackend([json.dumps({"cannot_formulate": "the pressure is not given"}), j(GOOD)])
    a = sb.ScienceBee(be, StubPipeline()).ask("weird")
    assert not a["formulated"] and a["attempts"] == 1 and "pressure" in a["failure_reason"]


def test_backend_error_gives_could_not_formulate():
    def boom(_m):
        raise sb.BackendError("down")
    a = sb.ScienceBee(sb.FakeBackend(boom), StubPipeline()).ask("q")
    assert not a["formulated"] and "backend" in a["errors"][0]


def test_french_question_gets_french_explanation_with_both_limits_sentences():
    spec = fam.sir_spec(0.5, 0.2, 0.99, 0.01, [10.0, 30.0])
    a = sb.ScienceBee(sb.FakeBackend([j(spec)]), StubPipeline()).ask("Dans une épidémie SIR avec beta = 0,5 et gamma = 0,2, quelle est la fraction infectée après 10 et 30 jours ?")
    assert a["language"] == "fr" and "Résultats" in a["text"] and "Contrôles" in a["text"]
    assert LIMITS in a["text"] and sb.LIMITS_SENTENCE_FR in a["text"]
    assert "Votre question mentionne (temps : jours)" in a["text"] and "did not convert" not in a["text"]
    assert sb.units_mentioned("10 jours")["time"] == ["days"]


def test_follow_up_modifies_previous_validated_spec_through_the_loop():
    doubled = fam.pendulum_spec(0.5, 0.0, T, damping=0.2)
    be = sb.FakeBackend([j(GOOD), j(dict(doubled, bogus=1)), j(doubled)])
    b = sb.ScienceBee(be, StubPipeline())
    first = b.ask("Pendulum with damping 0.1, theta0 0.5. theta at 1, 2, 5 s?")
    second = b.ask("and if the damping doubles?")
    assert second["follow_up"] and second["retries"] == 1 and second["formulated"]
    prompt = be.calls[1][-1]["content"]
    assert json.dumps(first["spec"], separators=(",", ":"), ensure_ascii=False) in prompt and "VALIDATED" in prompt
    assert b.last_spec["problem"]["params"]["damping"] == 0.2
    # an unrelated new question is not treated as a follow-up
    be2 = sb.FakeBackend([j(GOOD)])
    c = sb.ScienceBee(be2, StubPipeline())
    c.last_spec = GOOD
    c.ask("Kepler orbit question")
    assert "VALIDATED" not in be2.calls[0][-1]["content"]
    # French follow-up marker
    assert sb._FOLLOWUP_RE.match("Et si l'amortissement double ?")


def test_explainer_wording_units_checks_verdict_and_limits():
    q = "A 0.5 m pendulum, damping 0.1, released at 0.5 rad. theta after 2 minutes?"
    rep = StubPipeline("AUDITED").run(sb.canonical_bytes(GOOD))
    txt = sb.explain(q, GOOD, rep, language="en")
    assert "pendulum" in txt and "g_over_l = 9.81" in txt and "damping = 0.1" in txt and "theta = 0.1" in txt
    assert "Your question mentions (time: minutes; length: metres; angle: radians)" in txt
    assert "did not convert any unit silently" in txt
    assert "AUDITED" in txt and "agreement with a stored independent reference solution: not applicable" in txt
    assert txt.rstrip().endswith(LIMITS)
    default_used = sb.explain("q", fam.named_spec("pendulum", {}, [0.5, 0.0], T), rep, language="en")
    assert "default value" in default_used
    assert "VERIFIED never means" in sb.explain("q", GOOD, dict(rep, verdict="VERIFIED"), language="en")
    bad = sb.explain("q", GOOD, dict(rep, verdict="FAILED", certificate={"checks": [{"name": "tightening", "status": "fail"}], "reasons": ["too loose"]}), language="en")
    assert "Do not use these numbers" in bad and "Key numbers" not in bad and LIMITS in bad


def test_explainer_covers_every_problem_kind():
    for spec in (fam.sir_spec(0.5, 0.2, 0.99, 0.01, T), fam.series_rlc_spec(10, 0.5, 0.01, 5, 0, 0, T), fam.kepler_spec(1, 0, 0, 1, T)):
        rep = StubPipeline().run(sb.canonical_bytes(spec))
        txt = sb.explain("q", spec, rep, language="en")
        assert LIMITS in txt and "Initial state" in txt


def test_ollama_backend_loopback_only():
    for ok in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.2:5000/"):
        assert sb.OllamaBackend("m", base_url=ok).base_url.startswith("http://")
    for bad in ("http://10.0.0.5:11434", "http://example.com", "http://127.0.0.1@evil.example/", "http://127.0.0.1.evil.example/", "http://evil.example#@127.0.0.1",
                "ftp://127.0.0.1", "http://user:pw@127.0.0.1", "http://127.0.0.1:11434/api", "file:///etc/passwd", "http://0.0.0.0:11434", "http://169.254.169.254"):
        with pytest.raises(sb.BackendError):
            sb.OllamaBackend("m", base_url=bad)
    with pytest.raises(sb.BackendError):
        sb.OllamaBackend("bad model;rm", base_url="http://127.0.0.1:11434")


def test_ollama_payload_is_cpu_only_non_streaming_by_default():
    o = sb.OllamaBackend("qwen:7b")
    p = o.payload([{"role": "user", "content": "x"}])
    assert p["stream"] is False and p["options"]["num_gpu"] == 0 and p["model"] == "qwen:7b" and "think" not in p
    assert sb.OllamaBackend("m", num_gpu=99).payload([])["options"]["num_gpu"] == 99
    with pytest.raises(sb.BackendError):
        sb.OllamaBackend("m", num_gpu=-1)


def test_system_prompt_is_generated_from_the_catalogue():
    p = sb.build_system_prompt()
    for name in fam.FAMILIES:
        assert name in p
    assert "ONE JSON object" in p and "g_over_l" in p and "mass_action" in p


def test_baseline_formulator_handles_direct_questions_and_refuses_the_rest():
    s = sb.baseline_formulate("Pendulum with g_over_l = 9.81 and damping = 0.1, theta0 = 0.5, omega0 = 0. What are theta and omega at t = 1, 2, 5 s?")
    assert s is not None and sb.specs_match(s, GOOD)
    s = sb.baseline_formulate("SIR epidemic with beta = 0.5, gamma = 0.2, i0 = 0.01. Give S, I, R at t = 10, 30 days.")
    assert s is not None and s["problem"]["species"] == ["S", "I", "R"]
    s = sb.baseline_formulate("Épidémie SIR avec beta = 0,5, gamma = 0,2, i0 = 0,01. Donne S, I, R à t = 10, 30.")
    assert s is not None and s["problem"]["reactions"][0]["k"] == 0.5
    assert sb.baseline_formulate("What happens to a swing after a while?") is None
    assert sb.baseline_formulate("Pendulum theta0 = 0.5. theta at t = 5, 1?") is None


def test_baseline_backend_adapter_runs_through_the_same_loop():
    q = "Logistic growth with r = 0.5, K = 2, N0 = 0.1. N at t = 1, 2, 5."
    a = sb.ScienceBee(sb.BaselineBackend(), StubPipeline()).ask(q)
    assert a["formulated"] and a["spec"]["problem"]["family"] == "logistic"
    assert not sb.ScienceBee(sb.BaselineBackend(), StubPipeline()).ask("tell me a story")["formulated"]


def test_pipeline_fails_closed():
    rep = sb.SolverPipeline("/nonexistent/binary").run(sb.canonical_bytes(GOOD))
    assert rep["verdict"] == "UNVERIFIED" and rep["certificate"]["reasons"]
    assert sb.SolverPipeline("/nonexistent").run(b"not json")["verdict"] == "UNVERIFIED"


def test_specs_match_semantics():
    a = fam.sir_spec(0.5, 0.2, 0.99, 0.01, T)
    b = json.loads(json.dumps(a))
    b["id"], b["rtol"] = "other", 1e-6
    assert sb.specs_match(a, b)
    b["problem"]["reactions"].reverse()
    assert sb.specs_match(a, b)
    b["y0"][1] = 0.02
    assert not sb.specs_match(a, b)
    assert not sb.specs_match(a, {"bad": "spec"})
    assert sb._canon_num(3) == 3.0


def test_additional_coverage_for_validator_ollama_and_baseline(monkeypatch):
    # FakeBackend exhaustion
    fb = sb.FakeBackend(["one"])
    assert fb.complete([]) == "one"
    with pytest.raises(sb.BackendError):
        fb.complete([])

    # check_loopback_url edge cases & _NoRedirect
    with pytest.raises(sb.BackendError):
        sb.check_loopback_url("http://" + "a" * 210)
    assert sb._NoRedirect().redirect_request(None, None, 302, "Moved", {}, "http://127.0.0.1") is None

    # OllamaBackend with think=True and mocked opener responses
    ob = sb.OllamaBackend("qwen:7b", think=True, max_response_bytes=64)
    assert ob.payload([])["think"] is True

    class _DummyResp:
        def __init__(self, payload_bytes: bytes):
            self._data = payload_bytes

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n: int = -1) -> bytes:
            return self._data[:n] if n >= 0 else self._data

    monkeypatch.setattr(ob._opener, "open", lambda req, timeout=None: _DummyResp(b'{"message":{"content":"ok"}}'))
    assert ob.complete([{"role": "user", "content": "hi"}]) == "ok"

    monkeypatch.setattr(ob._opener, "open", lambda req, timeout=None: _DummyResp(b"x" * 100))
    with pytest.raises(sb.BackendError, match="too large"):
        ob.complete([])

    monkeypatch.setattr(ob._opener, "open", lambda req, timeout=None: _DummyResp(b'{"bad":1}'))
    with pytest.raises(sb.BackendError, match="not an Ollama chat reply"):
        ob.complete([])

    monkeypatch.setattr(ob._opener, "open", lambda req, timeout=None: _DummyResp(b'{"message":{"content":123}}'))
    with pytest.raises(sb.BackendError, match="not text"):
        ob.complete([])

    def _raise_oserror(req, timeout=None):
        raise OSError("conn refused")

    monkeypatch.setattr(ob._opener, "open", _raise_oserror)
    with pytest.raises(sb.BackendError, match="failed"):
        ob.complete([])

    # _balanced_objects escaped characters and extract_json max length
    escaped_json = r'prefix {"a": "hello \"world\" \\ end"} suffix'
    objs = sb._balanced_objects(escaped_json)
    assert len(objs) == 1
    with pytest.raises(sb.SpecInvalid, match="too long"):
        sb.extract_json("x" * (sb.MAX_RAW_CHARS + 10))

    # validate_spec error branches
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec([1, 2, 3])
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "bad"}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "named", "extra": 1}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "named", "family": "unknown"}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "named", "family": "pendulum", "params": []}, "y0": [1.0, 0.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "linear", "extra": 1}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "linear", "A": [], "b": []}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "linear", "A": [[1.0, 2.0]], "b": [1.0]}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "linear", "A": [[1.0]], "b": [1.0, 2.0]}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "extra": 1}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "species": [], "reactions": []}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "species": ["A"], "reactions": []}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "species": ["A"], "reactions": [{"bad": 1}]}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "species": ["A"], "reactions": [{"reactants": [], "products": {}, "k": 1.0}]}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        sb.validate_spec({"schema": sb.SPEC_SCHEMA, "id": "x", "problem": {"kind": "mass_action", "species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "k": 1.0}]}, "y0": [1.0], "t_out": [1.0]})
    with pytest.raises(sb.SpecInvalid):
        bad_atol = dict(GOOD, atol=[1e-8])
        sb.validate_spec(bad_atol)
    with pytest.raises(sb.SpecInvalid):
        bad_atol2 = dict(GOOD, atol=-1.0)
        sb.validate_spec(bad_atol2)

    # explain with > 8 time points
    long_spec = fam.pendulum_spec(0.5, 0.0, [float(i) for i in range(1, 12)])
    long_rep = {"verdict": "AUDITED", "result": {"t": [float(i) for i in range(1, 12)], "y": [[0.1, 0.0]] * 11}, "certificate": {"checks": []}}
    assert "intermediate times omitted" in sb.explain("q", long_spec, long_rep, language="en")

    # ScienceBee empty question error
    with pytest.raises(ValueError):
        sb.ScienceBee(sb.BaselineBackend(), StubPipeline()).ask("   ")

    # baseline_formulate across all families
    assert sb.baseline_formulate("SEIR epidemic with beta = 0.5, sigma = 0.3, gamma = 0.1, i0 = 0.01 at t = 1, 2, 5") is not None
    assert sb.baseline_formulate("SEIR epidemic with beta = 0.5 at t = 1, 2") is None
    assert sb.baseline_formulate("SIR epidemic with beta = 0.5 at t = 1, 2") is None
    assert sb.baseline_formulate("RLC circuit with R = 10, L = 0.5, C = 0.01, V = 5 at t = 1, 2") is not None
    assert sb.baseline_formulate("RLC circuit with R = 10 at t = 1, 2") is None
    assert sb.baseline_formulate("Spring with m = 1, c = 0.1, k = 4, x0 = 0.5 at t = 1, 2") is not None
    assert sb.baseline_formulate("Spring with m = 1 at t = 1, 2") is None
    assert sb.baseline_formulate("Pendulum without angle at t = 1, 2") is None
    assert sb.baseline_formulate("Projectile with vx0 = 10, vy0 = 20, k = 0.01 at t = 1, 2") is not None
    assert sb.baseline_formulate("Projectile with vx0 = 10 at t = 1, 2") is None
    assert sb.baseline_formulate("Kepler orbit with x0 = 1, y0 = 0, vx0 = 0, vy0 = 1 at t = 1, 2") is not None
    assert sb.baseline_formulate("Kepler orbit with x0 = 1 at t = 1, 2") is None
    assert sb.baseline_formulate("Logistic growth with r = 0.5 at t = 1, 2") is None
    assert sb.baseline_formulate("Decay chain with k1 = 1.0, k2 = 0.2, a0 = 1.0 at t = 1, 2") is not None
    assert sb.baseline_formulate("Decay chain with k1 = 1.0 at t = 1, 2") is None

    # SolverPipeline._library_problem and .run mocked execution
    pipe = sb.SolverPipeline("/bin/true", problems_dir="/tmp")

    class _FakeSvc:
        @staticmethod
        def load_problem_library(_d):
            return {"pendulum_1": {"spec": GOOD}}

        @staticmethod
        def canonical_sha256(obj):
            return json.dumps(obj, sort_keys=True)

    class _FakeOut:
        ok = True
        limit = None
        rc = 0
        error = None
        stderr_tail = ""
        stdout = b'{"status":"ok","t":[1.0,2.0,5.0],"y":[[0.1,0.0],[0.1,0.0],[0.1,0.0]]}'

    class _FakeSbx:
        @staticmethod
        def file_sha256(_b):
            return "abc123"

        class Limits:
            def __init__(self, **kw):
                pass

        @staticmethod
        def run_binary(cmd, raw, limits=None, expected_sha256=None):
            return _FakeOut()

    class _FakeVer:
        @staticmethod
        def verify(spec_bytes, result, problem=None, rerun=None, binary_sha256=None):
            if rerun is not None:
                rerun(spec_bytes)
            return {"verdict": "AUDITED", "checks": [], "reasons": []}

    monkeypatch.setattr(
        pipe,
        "_mod",
        lambda name: {"service": _FakeSvc, "sandbox": _FakeSbx, "verifier": _FakeVer}[name],
    )
    spec_with_lib = dict(GOOD, id="pendulum_1")
    out_ok = pipe.run(sb.canonical_bytes(spec_with_lib))
    assert out_ok["verdict"] == "AUDITED"
    spec_no_lib = dict(GOOD, id="missing_id")
    assert pipe._library_problem(spec_no_lib) is None

    # Solver error branch
    _FakeOut.stdout = b'{"status":"error","error":"diverged"}'
    out_err = pipe.run(sb.canonical_bytes(GOOD))
    assert out_err["verdict"] == "UNVERIFIED"


