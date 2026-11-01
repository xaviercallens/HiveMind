# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S34-02 Dynamic < 1.5s Game Preemption Circuit-Breaker with EF21 Flush."""

from __future__ import annotations

import pytest

from hivemind.swarm.s34_02_game_preemption_ef21_flush import GamePreemptionEF21CircuitBreaker


def test_game_preemption_flushes_6gb_vram_under_1500ms_and_conserves_ef21_state() -> None:
    breaker = GamePreemptionEF21CircuitBreaker(pcie_dma_bandwidth_gbps=12.8, top_k_ratio=0.25)
    grad = [0.12, -0.85, 0.03, 0.91, -0.04, 0.02, -0.15, 0.44]
    err = [0.01, -0.05, 0.02, 0.04, 0.00, -0.01, 0.03, -0.02]

    receipt = breaker.trigger_game_preemption(
        trigger_process="uss_hivemind_steam_3133700.exe",
        allocated_vram_mib=6144,
        inflight_gradient=grad,
        prev_error_buffer=err,
        kernel_drain_ms=38.0,
    )
    assert receipt.invariant_4_satisfied is True
    assert receipt.eviction_latency_ms < 1500.0
    assert receipt.residual_vram_after_mib == 0
    assert receipt.residual_vram_measurement_kind == "model_only"
    assert receipt.reconstruction_l2_error == pytest.approx(0.0, abs=1e-12)
    # Top 25% of 8 elements = 2 non-zero transmitted entries
    non_zero = [x for x in receipt.ef21_top_k_transmitted if abs(x) > 0.0]
    assert len(non_zero) == 2

    # IP-47 negative control: non-zero measured residual VRAM must fail invariant_4_satisfied
    failed_receipt = breaker.trigger_game_preemption(
        trigger_process="uss_hivemind_steam_3133700.exe",
        allocated_vram_mib=6144,
        inflight_gradient=grad,
        prev_error_buffer=err,
        kernel_drain_ms=38.0,
        measured_residual_vram_mib=256,
    )
    assert failed_receipt.residual_vram_measurement_kind == "measured"
    assert failed_receipt.residual_vram_after_mib == 256
    assert failed_receipt.invariant_4_satisfied is False


def test_ip47_h5_copy_time_above_attention_window_fails() -> None:
    """IP-47 negative control: H5 with DMA copy time above the attention window must report passed=False."""
    from hivemind.swarm.architecture_hypothesis_verifier import ArchitectureHypothesisVerifier

    verifier = ArchitectureHypothesisVerifier(seed=2026)
    rec_pass = verifier.verify_h5_skip_attention_car_and_async_dma(
        simulated_attention_compute_ms=5.0,
        simulated_dma_copy_ms=2.0,
    )
    assert rec_pass.passed is True

    rec_fail = verifier.verify_h5_skip_attention_car_and_async_dma(
        simulated_attention_compute_ms=5.0,
        simulated_dma_copy_ms=9.5,
    )
    assert rec_fail.passed is False
