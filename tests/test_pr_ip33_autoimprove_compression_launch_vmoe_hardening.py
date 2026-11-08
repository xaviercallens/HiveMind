# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and negative-control tests for Iteration 33 (IP-18, IP-22, IP-36, IP-37, IP-50, IP-63).

- IP-18: RNG-synthesized AutoImprovementExecutionResult is marked simulated=True and awards
  0 HMT in PointsRetrofitManager; synthetic refinement pairs carry solution_source='synthetic_template'.
- IP-22: project_international_launch_wave parameterizes uncontrolled_scraping_factor and
  uncontrolled_conversion_kappa so equal assumptions yield identical controlled vs uncontrolled rho.
- IP-36: run_exoplanet_astro_vmoe_poc evaluates stage4_routing_accuracy on a held-out shard
  generated with an independent seed (seed + 99991).
- IP-37: quantize_tensor_nf4_double_quant does not mutate the input weight tensor in place;
  compress_tensor_dct_sparse_2bit uses symmetric 2-bit centroids and computes index_width
  from the padded DCT coefficient count.
- IP-50: t4_governor.governor_step retries set_gpu('never') when a call returns False,
  fails closed when nginx status is unreachable during training, and other_gpu_busy fails closed on pmon failure.
- IP-63: test_quic_edge_coverage uses 10.0s wait_for timeouts instead of 1s.
"""

from __future__ import annotations

from pathlib import Path
import secrets
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hivemind.core.auto_improvement import (
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
)
from hivemind.core.compression import (
    compress_tensor_dct_sparse_2bit,
    decompress_tensor_dct_sparse_2bit,
)
from hivemind.core.exoplanet_astro_vmoe_poc import run_exoplanet_astro_vmoe_poc
from hivemind.core.quantization import quantize_tensor_nf4_double_quant
from hivemind.economy.international_launch_model import project_international_launch_wave
from hivemind.router.accounts import RouterStore
import t4_governor


def test_ip18_simulated_workunit_execution_mints_zero_hmt_and_labels_synthetic_pairs(
    tmp_path: Path,
) -> None:
    """IP-18 negative control: simulated RNG workunit execution earns 0 HMT; observed telemetry earns full reward."""
    scheduler = NocturnalAutoImprovementScheduler()
    wu = scheduler.create_domain_workunit(domain="maths_formal", num_samples=4, inner_steps=20, seed=42)
    assert all(p.get("solution_source") == "synthetic_template" for p in wu.sample_pairs)

    store = RouterStore(tmp_path / "router_ip18.sqlite", secrets.token_bytes(32))
    acct_id, _ = store.register("worker_ip18", "producer")
    retrofit_mgr = PointsRetrofitManager()

    # Negative control: without observed_telemetry, result is marked simulated=True and earns 0 HMT
    sim_res = scheduler.verify_workunit_execution(wu, worker_id="w-sim", account_id=acct_id)
    assert sim_res.simulated is True
    assert sim_res.telemetry_source == "rng_synthetic"
    assert retrofit_mgr.calculate_work_reward(sim_res) == 0.0

    rep_sim = retrofit_mgr.retrofit_work_points(store, [sim_res])
    assert rep_sim.total_hmt_granted == 0.0
    assert rep_sim.newly_credited_count == 0
    assert store.balance(acct_id) == 0.0
    assert any("SKIPPED_SIMULATED" in entry for entry in rep_sim.audit_log)

    # Positive control: with real observed worker telemetry, simulated is False and reward is credited
    obs_res = scheduler.verify_workunit_execution(
        wu,
        worker_id="w-obs",
        account_id=acct_id,
        observed_telemetry={
            "total_eval_samples": 200,
            "base_correct": 100,
            "adapted_correct": 168,
            "base_only_correct": 4,
            "adapted_only_correct": 72,
            "loss_initial": 2.40,
            "loss_final": 1.85,
            "peak_vram_mb": 4100.0,
            "duration_seconds": 88.0,
        },
    )
    assert obs_res.simulated is False
    assert obs_res.telemetry_source == "observed_worker_telemetry"
    assert retrofit_mgr.calculate_work_reward(obs_res) > 0.0

    rep_obs = retrofit_mgr.retrofit_work_points(store, [obs_res])
    assert rep_obs.newly_credited_count == 1
    assert rep_obs.total_hmt_granted > 0.0
    assert store.balance(acct_id) == pytest.approx(rep_obs.total_hmt_granted)


def test_ip22_international_launch_wave_equal_assumptions_yield_equal_rho() -> None:
    """IP-22 negative control: under equal scraping and conversion assumptions, controlled and uncontrolled rho match."""
    equal_proj = project_international_launch_wave(
        wave_name="EqualAssumptions",
        new_visitors=5000,
        consumer_to_prosumer_conversion_kappa=0.08,
        uncontrolled_scraping_factor=1.0,
        uncontrolled_conversion_kappa=0.08,
    )
    assert equal_proj.utilization_rho_without_controls == pytest.approx(
        equal_proj.utilization_rho_with_controls, abs=1e-6
    )

    default_proj = project_international_launch_wave(
        wave_name="DefaultAssumptions",
        new_visitors=5000,
    )
    assert default_proj.utilization_rho_without_controls > default_proj.utilization_rho_with_controls

    with pytest.raises(ValueError, match="uncontrolled_scraping_factor"):
        project_international_launch_wave("BadFactor", 1000, uncontrolled_scraping_factor=0.0)


def test_ip36_exoplanet_vmoe_poc_evaluates_routing_on_held_out_shard() -> None:
    """IP-36: Stage 4 routing accuracy is evaluated on an independent held-out shard (seed + 99991)."""
    report = run_exoplanet_astro_vmoe_poc(inner_steps=6, seed=42)
    assert report["stage4_holdout_seed_base"] == 42 + 99991
    assert report["stage4_calibration_seed_base"] == 9200
    assert report["stage4_holdout_seed_base"] != report["stage4_calibration_seed_base"]
    assert report["stage4_routing_accuracy"] == report["stage4_holdout_routing_accuracy"]
    assert report["stage4_routing_accuracy"] >= 0.90
    assert report["overall_poc_passed"] is True


def test_ip37_nf4_does_not_mutate_input_and_dct_2bit_is_symmetric_with_padded_index_width() -> None:
    """IP-37 negative controls: NF4 preserves input tensor; 2-bit DCT has symmetric relMSE and padded index width."""
    torch.manual_seed(1234)
    weight = torch.randn(64, 64, dtype=torch.float32)
    before = weight.clone()
    _state, reconstructed = quantize_tensor_nf4_double_quant(weight, block_size=64)
    assert torch.equal(weight, before), "quantize_tensor_nf4_double_quant mutated input weight tensor in place!"
    assert reconstructed.shape == weight.shape

    # Symmetric 2-bit centroids: positive vs negative tensor must have symmetric relative MSE
    t_pos = torch.linspace(0.1, 2.5, steps=256, dtype=torch.float32).reshape(16, 16)
    t_neg = -t_pos.clone()
    p_pos = compress_tensor_dct_sparse_2bit(t_pos, topk_ratio=0.25, chunk_size=8)
    p_neg = compress_tensor_dct_sparse_2bit(t_neg, topk_ratio=0.25, chunk_size=8)
    rec_pos = decompress_tensor_dct_sparse_2bit(p_pos)
    rec_neg = decompress_tensor_dct_sparse_2bit(p_neg)
    rel_mse_pos = float((rec_pos - t_pos).pow(2).sum().item() / t_pos.pow(2).sum().item())
    rel_mse_neg = float((rec_neg - t_neg).pow(2).sum().item() / t_neg.pow(2).sum().item())
    assert rel_mse_pos == pytest.approx(rel_mse_neg, rel=1e-3, abs=1e-4)

    # Padded index_width: 1D tensor of 65531 elements (<= 65536) pads to (8, 65536) = 524288 (> 65536) -> index_width == 4
    borderline = torch.ones(65531, dtype=torch.float32)
    payload = compress_tensor_dct_sparse_2bit(borderline, topk_ratio=0.001, chunk_size=8)
    assert payload.original_numel == 65531
    assert payload.padded_shape[0] * payload.padded_shape[1] > 0x10000
    assert payload.index_width == 4


def test_ip50_t4_governor_step_retries_failed_set_gpu_and_fails_closed() -> None:
    """IP-50 negative control: failed set_gpu('never') does not transition to 'yielded' and retries next tick."""
    calls: list[str] = []
    outcomes = [False, True]

    def fake_set_gpu(mode: str) -> bool:
        calls.append(mode)
        return outcomes.pop(0)

    # Tick 1: busy while training, set_gpu("never") fails -> state stays "training"
    state1, action1, ok1 = t4_governor.governor_step(
        "training",
        in_flight=1,
        http_ok=True,
        other_busy=False,
        hold_exists=False,
        quiet_s=0.0,
        in_night_window=True,
        set_gpu_fn=fake_set_gpu,
    )
    assert ok1 is False
    assert action1 == "yield"
    assert state1 == "training"

    # Tick 2: next tick retries set_gpu("never") and succeeds -> state becomes "yielded"
    state2, action2, ok2 = t4_governor.governor_step(
        state1,
        in_flight=1,
        http_ok=True,
        other_busy=False,
        hold_exists=False,
        quiet_s=0.0,
        in_night_window=True,
        set_gpu_fn=fake_set_gpu,
    )
    assert ok2 is True
    assert action2 == "yield"
    assert state2 == "yielded"
    assert calls == ["never", "never"]

    # Losing nginx status while training must fail closed and yield to "never"
    state3, action3, ok3 = t4_governor.governor_step(
        "training",
        in_flight=0,
        http_ok=False,
        other_busy=False,
        hold_exists=False,
        quiet_s=3600.0,
        in_night_window=True,
        set_gpu_fn=lambda _m: True,
    )
    assert ok3 is True
    assert action3 == "yield"
    assert state3 == "yielded"

    # other_gpu_busy fails closed when nvidia-smi pmon raises
    fake_g = {
        "util_pct": 45,
        "procs": [(1234, "/mnt/disks/boinc/projects/hivemind/wrapper", 1200)],
    }
    with (
        patch.object(t4_governor, "pipeline_gpu_pids", return_value=[1234]),
        patch.object(subprocess, "run", side_effect=OSError("pmon failed")),
    ):
        busy_flag, util_val = t4_governor.other_gpu_busy(fake_g, threshold_pct=10)
    assert busy_flag is True
    assert util_val >= 10
