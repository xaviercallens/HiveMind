# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S35-02: GWAYA (Gwen-Laya) x ANSE Fail-Closed Gate & Arm A3 Repair Loop."""

from __future__ import annotations

from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import (
    MAXIMUM_PAIN_ENERGY,
    audit_candidate_gwaya_anse,
    inspect_python_ast_anti_stub,
    run_gwaya_a3_sample_and_repair,
)


def test_gwaya_ast_anti_stub_rejects_placeholders() -> None:
    stub_pass = "def solve(x: float) -> float:\n    pass\n"
    stub_ellipsis = "def solve(x: float) -> float:\n    ...\n"
    stub_mock = "def mock_solver(x: float) -> float:\n    return 1.0\n"
    assert len(inspect_python_ast_anti_stub(stub_pass)) >= 1
    assert len(inspect_python_ast_anti_stub(stub_ellipsis)) >= 1
    assert len(inspect_python_ast_anti_stub(stub_mock)) >= 1

    audit = audit_candidate_gwaya_anse(stub_pass, lambda _: True, lambda _: True)
    assert audit.is_ast_clean is False
    assert audit.anse_energy == MAXIMUM_PAIN_ENERGY
    assert audit.verdict == "UNVERIFIED_MAX_PAIN"


def test_gwaya_a3_sample_and_repair_recovers_from_initial_failure() -> None:
    candidates = [
        "def f(x: int) -> int:\n    pass\n",
        "def f(x: int) -> int:\n    return x + 1\n",
        "def f(x: int) -> int:\n    return x * x\n",
    ]
    report = run_gwaya_a3_sample_and_repair(
        candidates=candidates,
        public_test_fn=lambda code: "x * x" in code,
        hidden_test_fn=lambda code: "x * x" in code and "pass" not in code,
    )
    assert report.initial_a0_passed is False
    assert report.final_a3_passed is True
    assert report.rounds_used == 3
    assert report.final_energy < 100.0


def test_a_function_whose_body_is_only_a_docstring_is_a_stub() -> None:
    from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import audit_candidate_gwaya_anse, inspect_python_ast_anti_stub

    doc_only = 'def f(x):\n    "doc"\n'
    assert any("docstring" in v for v in inspect_python_ast_anti_stub(doc_only))
    verdict = audit_candidate_gwaya_anse(doc_only, lambda s: True, lambda s: True).verdict
    assert verdict != "VERIFIED"
    # positive control: a docstring plus a real statement is not a stub
    real = 'def f(x):\n    "doc"\n    return x + 1\n'
    assert inspect_python_ast_anti_stub(real) == []
    assert audit_candidate_gwaya_anse(real, lambda s: True, lambda s: True).verdict == "VERIFIED"


def test_confidence_must_be_a_probability_not_nan_or_out_of_range() -> None:
    import pytest

    good_src = "def f(x: int) -> int:\n    return x + 1\n"
    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        with pytest.raises(ValueError):
            audit_candidate_gwaya_anse(good_src, lambda s: True, lambda s: True, raw_logprob_confidence=bad)
    ok = audit_candidate_gwaya_anse(good_src, lambda s: True, lambda s: True, raw_logprob_confidence=0.93)
    assert ok.verdict == "VERIFIED"


def test_gwaya_ast_anti_stub_catches_all_six_stub_shapes_ip272() -> None:
    stub_shapes = [
        "def f(x): return None",
        'def f(x):\n    x = 1\n    raise NotImplementedError("later")',
        "class A: pass",
        "def f(x): return NotImplemented",
        'def f(x):\n    "doc"\n    return ...',
        "g = lambda x: 1",
        "async def af(x):\n    return None",
    ]
    for snippet in stub_shapes:
        violations = inspect_python_ast_anti_stub(snippet)
        assert violations, f"Expected non-empty violations for {snippet!r}, got {violations!r}"

    # Positive controls: non-stub class, function, and lambda pass
    clean_code = (
        "class Accumulator:\n"
        "    def __init__(self, start: int) -> None:\n"
        "        self.total = start\n"
        "    def add(self, x: int) -> int:\n"
        "        self.total += x\n"
        "        return self.total\n"
        "scale = lambda x: x * 2\n"
    )
    assert inspect_python_ast_anti_stub(clean_code) == []


def test_clean_candidate_with_low_confidence_triggers_abstain_repair_ip185() -> None:
    clean_candidate = "def f(x: int) -> int:\n    return x * 2\n"
    audit = audit_candidate_gwaya_anse(
        clean_candidate,
        lambda _s: True,
        lambda _s: True,
        raw_logprob_confidence=0.3,
    )
    assert audit.is_ast_clean is True
    assert audit.passed_public_test is True
    assert audit.passed_hidden_tests is True
    assert audit.laya_calibrated_score == 0.685
    assert audit.verdict == "ABSTAIN_REPAIR"
    assert audit.anse_energy == MAXIMUM_PAIN_ENERGY


