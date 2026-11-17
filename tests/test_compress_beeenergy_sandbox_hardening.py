# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for KV lattice, lattice quantizer, Bee energy, and SolverBee sandbox hardening."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import is_a2_point, lattice_quantize
from hivemind.compress.lattice_quantizer import (
    LatticeDeltaQuantizer,
    SignDeltaQuantizer,
    brute_force_decoder_check,
    calibrate_scale_for_rate,
    is_lattice_point,
)
from hivemind.solverbee.sandbox import Limits, file_sha256, namespace_available
from hivemind.verification.bee_energy import (
    calibrate_noise_margin,
    compress_error_trace_d1,
    detect_boinc_loss_plateau,
    evaluate_bee_candidate_energy,
    evaluate_child_bee_promotion,
    exact_mcnemar_p,
    harvest_energy_gap_dpo_pair_d6,
    select_best_so_far_d3,
)


def test_kv_lattice_rejects_boolean_or_invalid_inputs() -> None:
    pts = torch.zeros((4, 2), dtype=torch.float64)
    assert bool(is_a2_point(pts).all())
    with pytest.raises(ValueError, match="tol"):
        is_a2_point(pts, tol=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        is_a2_point(pts, tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        is_a2_point(pts, tol=float("nan"))
    with pytest.raises(ValueError, match="A2 points"):
        is_a2_point(torch.zeros((4, 3), dtype=torch.float64))

    x = torch.randn((2, 8), dtype=torch.float32)
    deq, _ = lattice_quantize("E8", x, 0.5, measure_rate=False)
    assert deq.shape == x.shape
    with pytest.raises(ValueError, match="scale"):
        lattice_quantize("E8", x, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="measure_rate"):
        lattice_quantize("E8", x, 0.5, measure_rate=1)  # type: ignore[arg-type]


def test_lattice_quantizer_rejects_boolean_or_invalid_inputs() -> None:
    arr = np.zeros((2, 8), dtype=float)
    assert bool(is_lattice_point("E8", arr).all())
    with pytest.raises(ValueError, match="tol"):
        is_lattice_point("E8", arr, tol=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        is_lattice_point("E8", arr, tol=0.0)

    with pytest.raises(ValueError, match="scale"):
        LatticeDeltaQuantizer("E8", scale=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block"):
        LatticeDeltaQuantizer("Z", scale=0.5, block=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_feedback"):
        LatticeDeltaQuantizer("E8", scale=0.5, error_feedback=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_feedback"):
        SignDeltaQuantizer(error_feedback=0)  # type: ignore[arg-type]

    sample = torch.randn(64, dtype=torch.float32)
    with pytest.raises(ValueError, match="target_bits_per_dim"):
        calibrate_scale_for_rate("E8", sample, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        calibrate_scale_for_rate("E8", sample, 2.0, tol=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="iters"):
        calibrate_scale_for_rate("E8", sample, 2.0, iters=True)  # type: ignore[arg-type]

    with pytest.raises(KeyError):
        brute_force_decoder_check("UNKNOWN", 8, 0)
    with pytest.raises(ValueError, match="n_queries"):
        brute_force_decoder_check("A2", True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_queries"):
        brute_force_decoder_check("A2", 0, 0)
    with pytest.raises(ValueError, match="seed"):
        brute_force_decoder_check("A2", 8, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="radius"):
        brute_force_decoder_check("A2", 8, 0, radius=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="radius"):
        brute_force_decoder_check("A2", 8, 0, radius=0.0)


def test_bee_energy_rejects_boolean_and_blank_parameters() -> None:
    code = "def solve(x):\n    return x * 2\n"
    with pytest.raises(ValueError, match="candidate_id"):
        evaluate_bee_candidate_energy(candidate_id="   ", response_text=code, tests_passed=1, tests_total=1)
    with pytest.raises(ValueError, match="timed_out_or_crashed"):
        evaluate_bee_candidate_energy(
            candidate_id="c1", response_text=code, tests_passed=1, tests_total=1, timed_out_or_crashed=1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="tests_passed"):
        evaluate_bee_candidate_energy(
            candidate_id="c1", response_text=code, tests_passed=True, tests_total=1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="latency_ms"):
        evaluate_bee_candidate_energy(
            candidate_id="c1", response_text=code, tests_passed=1, tests_total=1, latency_ms=False  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="energy weights"):
        evaluate_bee_candidate_energy(
            candidate_id="c1", response_text=code, tests_passed=1, tests_total=1, w_fail=True  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="max_chars"):
        compress_error_trace_d1(code, "err", max_chars=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_chars"):
        compress_error_trace_d1(code, "err", max_chars=0)

    ev = evaluate_bee_candidate_energy(candidate_id="c1", response_text=code, tests_passed=1, tests_total=1)
    with pytest.raises(ValueError, match="early_stop_energy_ceiling"):
        select_best_so_far_d3([ev], early_stop_energy_ceiling=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prompt"):
        harvest_energy_gap_dpo_pair_d6("   ", [ev, ev])
    with pytest.raises(ValueError, match="min_delta_energy"):
        harvest_energy_gap_dpo_pair_d6("p", [ev, ev], min_delta_energy=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="block_size"):
        detect_boinc_loss_plateau([1.0, 0.9], block_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_block_gain"):
        detect_boinc_loss_plateau([1.0, 0.9], block_size=1, min_block_gain=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="round_losses"):
        detect_boinc_loss_plateau([1.0, True], block_size=1)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="discordant"):
        exact_mcnemar_p(True, 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="aa_median_deltas"):
        calibrate_noise_margin([0.1, True, 0.2])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="parent_regressions_failed"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, 12.0],
            child_energies=[2.0, 3.0],
            noise_margin=0.5,
            parent_regressions_failed=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="shuffled_control_gained"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, 12.0],
            child_energies=[2.0, 3.0],
            noise_margin=0.5,
            shuffled_control_gained=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="noise_margin"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, 12.0],
            child_energies=[2.0, 3.0],
            noise_margin=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="parent_energies"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, True],  # type: ignore[list-item]
            child_energies=[2.0, 3.0],
            noise_margin=0.5,
        )


def test_solverbee_sandbox_limits_reject_boolean_and_invalid_bounds() -> None:
    lim = Limits(timeout_s=5.0, mem_mb=512, stdin_limit=0)
    assert lim.timeout_s == 5.0 and lim.stdin_limit == 0
    with pytest.raises(ValueError, match="timeout_s"):
        Limits(timeout_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout_s"):
        Limits(timeout_s=0.0)
    with pytest.raises(ValueError, match="mem_mb"):
        Limits(mem_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mem_mb"):
        Limits(mem_mb=0)
    with pytest.raises(ValueError, match="stdin_limit"):
        Limits(stdin_limit=-1)
    with pytest.raises(ValueError, match="allow_unsandboxed"):
        Limits(allow_unsandboxed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refresh"):
        namespace_available(refresh=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path"):
        file_sha256("   ")
