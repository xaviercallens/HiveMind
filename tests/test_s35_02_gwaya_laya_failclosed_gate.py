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
