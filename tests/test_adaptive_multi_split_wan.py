"""Unit tests for HM-S29-01 Adaptive Multi-Split ML over Constrained WAN Links (Experiment E20)."""

from __future__ import annotations

import pytest

from hivemind.swarm.adaptive_multi_split_wan import (
    MAX_GAMER_PREEMPTION_SLA_MS,
    MAX_PRELUDE_CLIENT_VRAM_MIB,
    AdaptiveMultiSplitWANController,
    CitizenHardwareProfile,
    WANLinkTelemetry,
    run_e20_adaptive_multi_split_experiment,
)


def test_wan_link_telemetry_validation() -> None:
    link = WANLinkTelemetry(uplink_mbps=20.0, downlink_mbps=50.0, rtt_ms=60.0, jitter_ms=4.0, packet_loss_rate=0.01)
    assert link.uplink_mbps == 20.0
    with pytest.raises(ValueError):
        WANLinkTelemetry(uplink_mbps=0.0, downlink_mbps=50.0, rtt_ms=60.0)
    with pytest.raises(ValueError):
        WANLinkTelemetry(uplink_mbps=20.0, downlink_mbps=50.0, rtt_ms=-1.0)
    with pytest.raises(ValueError):
        WANLinkTelemetry(uplink_mbps=20.0, downlink_mbps=50.0, rtt_ms=20.0, packet_loss_rate=1.0)


def test_precision_cohort_isolation_pre_ampere_vs_ampere() -> None:
    pascal = CitizenHardwareProfile("n1", "GTX 1080 Ti", compute_capability_sm=6.1)
    turing = CitizenHardwareProfile("n2", "RTX 2070", compute_capability_sm=7.5)
    ampere = CitizenHardwareProfile("n3", "RTX 3060", compute_capability_sm=8.6)
    blackwell = CitizenHardwareProfile("n4", "RTX 5090", compute_capability_sm=10.0)

    assert pascal.precision_cohort == "cohort_fp32_pre_ampere"
    assert turing.precision_cohort == "cohort_fp32_pre_ampere"
    assert pascal.dtype_bits == 32
    assert ampere.precision_cohort == "cohort_bf16_ampere_plus"
    assert blackwell.precision_cohort == "cohort_bf16_ampere_plus"
    assert ampere.dtype_bits == 16


def test_adaptive_cut_layer_selection_and_gamer_preemption() -> None:
    ctrl = AdaptiveMultiSplitWANController()
    slow_wan = WANLinkTelemetry(uplink_mbps=20.0, downlink_mbps=40.0, rtt_ms=60.0, jitter_ms=5.0)
    fast_wan = WANLinkTelemetry(uplink_mbps=800.0, downlink_mbps=1000.0, rtt_ms=10.0, jitter_ms=0.5)
    hw = CitizenHardwareProfile("node-1", "RTX 3060", compute_capability_sm=8.6, vram_budget_mib=1500.0)

    dec_slow = ctrl.select_optimal_cut_layer(slow_wan, hw)
    dec_fast = ctrl.select_optimal_cut_layer(fast_wan, hw)

    assert dec_slow.vram_compliant is True
    assert dec_slow.client_prelude_vram_mib <= MAX_PRELUDE_CLIENT_VRAM_MIB
    assert dec_slow.preemption_compliant is True
    assert dec_slow.preemption_latency_ms <= MAX_GAMER_PREEMPTION_SLA_MS
    assert dec_slow.activation_bits == 8
    assert dec_fast.activation_bits == 16
    assert dec_slow.step_time_reduction_pct >= 55.0

    # Gamer preemption forces cut layer 1 (minimal VRAM footprint)
    gaming_hw = CitizenHardwareProfile(
        "node-game", "RTX 2070", compute_capability_sm=7.5, vram_budget_mib=1500.0, foreground_game_active=True
    )
    dec_game = ctrl.select_optimal_cut_layer(slow_wan, gaming_hw)
    assert dec_game.cut_layer_index == 1
    assert dec_game.preemption_latency_ms <= MAX_GAMER_PREEMPTION_SLA_MS


def test_int8_cut_boundary_quantization_and_e20_gate() -> None:
    ctrl = AdaptiveMultiSplitWANController()
    q = ctrl.quantize_cut_activations_int8([-1.5, -0.5, 0.0, 0.5, 1.5])
    assert q["compression_vs_fp32"] == 4.0
    assert q["relative_l2_error"] < 0.02
    assert len(q["sha256"]) == 64

    report = run_e20_adaptive_multi_split_experiment()
    assert report["experiment_id"] == "E20"
    assert report["task_id"] == "HM-S29-01"
    assert report["e20_gate_passed"] is True
    assert report["mean_step_time_reduction_pct_at_20mbps_60ms"] >= 55.0
