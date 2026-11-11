# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 192: Negative-control and invariant tests for canary_probe, bee_energy, solverbee.scoring, and solverbee.sandbox."""

from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.solverbee.sandbox import Limits, file_sha256, run_binary
from hivemind.solverbee.scoring import track_record
from hivemind.verification.bee_energy import (
    AutopoieticPromotionVerdict,
    BeeEnergyEvaluation,
    evaluate_bee_candidate_energy,
    harvest_energy_gap_dpo_pair_d6,
    select_best_so_far_d3,
)
from hivemind.verification.canary_probe import CanaryPrompt, NodeCanaryState


def test_canary_probe_rejects_control_chars_and_invalid_regex() -> None:
    cp = CanaryPrompt(
        canary_id="canary-1",
        prompt="What is 3 * 5?",
        expected_pattern=r"\b15\b",
        category="math",
    )
    assert cp.verify_answer("15") is True
    assert cp.to_chat_request("  hivemind/coder  ")["model"] == "hivemind/coder"

    with pytest.raises(ValueError, match="canary_id"):
        CanaryPrompt(
            canary_id="canary\x001",
            prompt="What is 3 * 5?",
            expected_pattern=r"\b15\b",
            category="math",
        )
    with pytest.raises(ValueError, match="category"):
        CanaryPrompt(
            canary_id="canary-1",
            prompt="What is 3 * 5?",
            expected_pattern=r"\b15\b",
            category="math\n",
        )
    with pytest.raises(ValueError, match="expected_pattern"):
        CanaryPrompt(
            canary_id="canary-1",
            prompt="What is 3 * 5?",
            expected_pattern="[unclosed",
            category="math",
        )
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        cp.to_chat_request("hivemind/coder bad")

    with pytest.raises(ValueError, match="node_id"):
        NodeCanaryState(node_id="node\n1", planted_role="honest")
    with pytest.raises(ValueError, match="planted_role"):
        NodeCanaryState(node_id="node-1", planted_role="honest\x00")
    with pytest.raises(ValueError, match="wilson_lb_threshold"):
        NodeCanaryState(node_id="node-1", planted_role="honest", wilson_lb_threshold=float("nan"))


def test_bee_energy_evaluation_and_verdict_hardening() -> None:
    valid_ev = BeeEnergyEvaluation(
        candidate_id="cand-192",
        response_text="def f(x):\n    return x + 1\n",
        tests_passed=2,
        tests_total=2,
        latency_ms=4.0,
        memory_mb=12.0,
        invariant_residual=0.0,
        is_stub_or_invalid=False,
        failure_reason=None,
        total_energy=0.05,
        receipt_sha256="a1" * 32,
    )
    assert valid_ev.passed_all_tests is True

    with pytest.raises(ValueError, match="tests_passed cannot exceed tests_total"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=3,
            tests_total=2,
            latency_ms=4.0,
            memory_mb=12.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=0.05,
            receipt_sha256="a1" * 32,
        )
    with pytest.raises(ValueError, match="latency_ms"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=-1.0,
            memory_mb=12.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=0.05,
            receipt_sha256="a1" * 32,
        )
    with pytest.raises(ValueError, match="memory_mb"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=4.0,
            memory_mb=float("nan"),
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=0.05,
            receipt_sha256="a1" * 32,
        )
    with pytest.raises(ValueError, match="failure_reason"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=1,
            tests_total=2,
            latency_ms=4.0,
            memory_mb=12.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason="   ",
            total_energy=50.0,
            receipt_sha256="a1" * 32,
        )
    with pytest.raises(ValueError, match="total_energy"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=4.0,
            memory_mb=12.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=float("inf"),
            receipt_sha256="a1" * 32,
        )
    with pytest.raises(ValueError, match="receipt_sha256"):
        BeeEnergyEvaluation(
            candidate_id="cand-192",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=4.0,
            memory_mb=12.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=0.05,
            receipt_sha256="z" * 64,
        )

    # Stub candidate with tests_passed > tests_total clamps cleanly
    stub_ev = evaluate_bee_candidate_energy(
        candidate_id="cand-stub",
        response_text="def f(x):\n    pass\n",
        tests_passed=2,
        tests_total=0,
    )
    assert stub_ev.tests_passed == 0 and stub_ev.tests_total == 0

    with pytest.raises(ValueError, match="attempts"):
        select_best_so_far_d3("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BeeEnergyEvaluation"):
        select_best_so_far_d3(["not-an-eval"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="attempts"):
        harvest_energy_gap_dpo_pair_d6("prompt", "not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BeeEnergyEvaluation"):
        harvest_energy_gap_dpo_pair_d6("prompt", [valid_ev, "not-an-eval"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="mcnemar_p"):
        AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=0.0,
            parent_regressions_failed=0,
            shuffled_control_gained=False,
            reason="REJECTED",
            mcnemar_p=1.5,
        )


def test_solverbee_scoring_and_sandbox_hardening(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterable"):
        track_record(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="all verdicts must be strings"):
        track_record([{"bad": 1}])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="path"):
        file_sha256("/tmp/bad\x00path")

    out_non_str_arg = run_binary(["/bin/true", 123], b"")  # type: ignore[list-item]
    assert out_non_str_arg.refused is True
    assert out_non_str_arg.error == "empty command"

    out_nul_arg = run_binary(["/bin/true", "arg\x00bad"], b"")
    assert out_nul_arg.refused is True
    assert out_nul_arg.error == "empty command"

    dummy_bin = tmp_path / "dummy.sh"
    dummy_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dummy_bin.chmod(0o755)
    sha = file_sha256(str(dummy_bin))

    out_bad_sha = run_binary([str(dummy_bin)], b"", limits=Limits(allow_unsandboxed=True), expected_sha256="not-hex")
    assert out_bad_sha.refused is True
    assert "64-character hex" in out_bad_sha.error

    out_mismatch_sha = run_binary(
        [str(dummy_bin)],
        b"",
        limits=Limits(allow_unsandboxed=True),
        expected_sha256=("0" if sha[0] != "0" else "1") + sha[1:],
    )
    assert out_mismatch_sha.refused is True
    assert "does not match" in out_mismatch_sha.error
