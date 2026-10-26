"""Tests for the Season 18 Autopoietic Energy-Bees module (`src/hivemind/verification/bee_energy.py`)."""

from __future__ import annotations

import pytest

from hivemind.verification.bee_energy import (
    MAX_PAIN_ENERGY,
    calibrate_noise_margin,
    compress_error_trace_d1,
    detect_boinc_loss_plateau,
    evaluate_bee_candidate_energy,
    evaluate_child_bee_promotion,
    exact_mcnemar_p,
    harvest_energy_gap_dpo_pair_d6,
    select_best_so_far_d3,
)


def test_bee_energy_assigns_max_pain_to_stubs_and_grades_real_code() -> None:
    stub = evaluate_bee_candidate_energy(
        candidate_id="c_stub",
        response_text="def solve(x):\n    pass\n",
        tests_passed=4,
        tests_total=4,
        latency_ms=5.0,
        memory_mb=10.0,
    )
    assert stub.is_stub_or_invalid is True
    assert stub.total_energy == MAX_PAIN_ENERGY
    assert stub.passed_all_tests is False

    partial = evaluate_bee_candidate_energy(
        candidate_id="c_partial",
        response_text="def solve(x):\n    return x + 1\n",
        tests_passed=2,
        tests_total=4,
        latency_ms=20.0,
        memory_mb=16.0,
    )
    assert partial.is_stub_or_invalid is False
    assert partial.passed_all_tests is False
    assert 50.0 <= partial.total_energy < 75.0  # two failed tests, plus a cost below one test's weight

    verified = evaluate_bee_candidate_energy(
        candidate_id="c_ok",
        response_text="def solve(x):\n    return x * 2\n",
        tests_passed=4,
        tests_total=4,
        latency_ms=15.0,
        memory_mb=16.0,
    )
    assert verified.passed_all_tests is True
    assert verified.total_energy < 25.0  # strictly below the energy of one failed test (100 / 4)


def test_correctness_dominates_cost_regression_from_review() -> None:
    """Review finding (2026-10-26): with linear cost terms a correct 6 s answer scored 60.5 and a half-wrong 5 ms
    answer 50.1, so the fast-but-wrong answer ranked first. Cost must never outweigh one failed test."""
    slow_ok = evaluate_bee_candidate_energy(
        candidate_id="slow_ok", response_text="def f(x):\n    return 2 * x\n",
        tests_passed=4, tests_total=4, latency_ms=60_000.0, memory_mb=4096.0,
    )
    fast_one_wrong = evaluate_bee_candidate_energy(
        candidate_id="fast_one_wrong", response_text="def f(x):\n    return x + x if x else 1\n",
        tests_passed=3, tests_total=4, latency_ms=1.0, memory_mb=1.0,
    )
    assert slow_ok.total_energy < fast_one_wrong.total_energy
    # among equally correct answers, cost still orders them
    fast_ok = evaluate_bee_candidate_energy(
        candidate_id="fast_ok", response_text="def f(x):\n    return 2 * x\n",
        tests_passed=4, tests_total=4, latency_ms=5.0, memory_mb=8.0,
    )
    assert fast_ok.total_energy < slow_ok.total_energy


def test_directives_d1_d3_and_d6_behavior() -> None:
    msg = compress_error_trace_d1(
        "def very_long_function_name(a, b, c): return a + b - c * 999999999999999",
        "AssertionError: expected 42 but got -999999999999999 in hidden test 3 of 5",
        max_chars=120,
    )
    assert len(msg) <= 120
    assert msg.startswith("ERR:")

    a1 = evaluate_bee_candidate_energy(
        candidate_id="try1", response_text="def f(x):\n    return x + 1\n",
        tests_passed=2, tests_total=4, latency_ms=20.0,
    )
    a2_worse = evaluate_bee_candidate_energy(
        candidate_id="try2", response_text="def f(x):\n    return x - 1\n",
        tests_passed=1, tests_total=4, latency_ms=25.0,
    )
    best, stop = select_best_so_far_d3([a1, a2_worse])
    assert best.candidate_id == "try1"
    assert stop is True

    a3_solved = evaluate_bee_candidate_energy(
        candidate_id="try3", response_text="def f(x):\n    return x * 2\n",
        tests_passed=4, tests_total=4, latency_ms=12.0,
    )
    pair = harvest_energy_gap_dpo_pair_d6("Double x", [a1, a2_worse, a3_solved], min_delta_energy=10.0)
    assert pair is not None
    assert pair["chosen_response"] == a3_solved.response_text
    assert pair["rejected_response"] == a2_worse.response_text
    assert pair["delta_energy"] >= 50.0


def test_boinc_loss_plateau_detector_on_synthetic_chain() -> None:
    losses = [2.27 - 0.006 * i for i in range(20)] + [2.142 + (0.001 if i % 2 == 0 else -0.001) for i in range(20)]
    plat = detect_boinc_loss_plateau(losses, block_size=10)
    assert plat["plateau_detected"] is True
    assert plat["recommended_action"] == "ROTATE_CURRICULUM_OR_COSINE_LR_RESTART"
    assert detect_boinc_loss_plateau(losses[:15])["reason"] == "INSUFFICIENT_ROUNDS"


def test_exact_mcnemar_and_noise_calibration() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(193, 7) < 1e-4  # NIGHT-MOE-1 stars adapter vs shuffled control
    assert exact_mcnemar_p(5, 5) == 1.0
    assert calibrate_noise_margin([0.1, -0.3, 0.2]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        calibrate_noise_margin([0.1, 0.2])


def test_autopoietic_promotion_gate() -> None:
    margin = calibrate_noise_margin([0.12, -0.2, 0.08, 0.15])

    wrong = evaluate_child_bee_promotion(
        parent_energies=[10.0, 12.0, 11.0, 9.0, 10.5],
        child_energies=[1.0, 1.2, 1.1, 0.9, 1.0],
        noise_margin=margin,
        parent_regressions_failed=1,
    )
    assert wrong.promoted is False and "REJECTED_FAST_BUT_WRONG" in wrong.reason

    noise = evaluate_child_bee_promotion(
        parent_energies=[10.0, 10.2, 9.9, 10.1, 10.0],
        child_energies=[9.95, 10.15, 9.88, 10.05, 9.98],
        noise_margin=margin,
    )
    assert noise.promoted is False and "REJECTED_WITHIN_NOISE_MARGIN" in noise.reason

    control = evaluate_child_bee_promotion(
        parent_energies=[25.0, 22.0, 24.0], child_energies=[2.0, 1.5, 2.2],
        noise_margin=margin, shuffled_control_gained=True,
    )
    assert control.promoted is False and control.reason == "VOID_SHUFFLED_CONTROL_GAINED"

    not_significant = evaluate_child_bee_promotion(
        parent_energies=[25.0, 22.0, 24.0], child_energies=[2.0, 1.5, 2.2],
        noise_margin=margin, paired_discordant=(3, 1),
    )
    assert not_significant.promoted is False and "REJECTED_NOT_SIGNIFICANT" in not_significant.reason

    good = evaluate_child_bee_promotion(
        parent_energies=[25.0, 22.0, 24.0, 20.0, 23.0],
        child_energies=[2.0, 1.5, 2.2, 1.8, 1.9],
        noise_margin=margin,
        paired_discordant=(40, 4),
    )
    assert good.promoted is True
    assert good.median_delta_energy == pytest.approx(-21.1)
    assert good.mcnemar_p is not None and good.mcnemar_p < 0.05
