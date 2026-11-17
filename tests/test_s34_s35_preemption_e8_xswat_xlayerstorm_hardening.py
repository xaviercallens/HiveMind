# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening tests for s34_02, s34_03, s35_01, and s35_04 dataclasses and entrypoints."""

from __future__ import annotations

import pytest

from hivemind.swarm.s34_02_game_preemption_ef21_flush import (
    GamePreemptionEF21CircuitBreaker,
    GamePreemptionFlushReceipt,
)
from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import (
    E8QuantizedBlock,
    GossetE8LatticeQuantizer,
)
from hivemind.swarm.s35_01_optimal_energy_amcp_xswat_codec import (
    A2RoPEQuantizationReport,
    AMCPLatentKVCacheEnvelope,
    benchmark_a2_rope_plane_quantizer,
    pack_amcp_xswat_latent_kv_envelope,
)
from hivemind.swarm.s35_04_xlayerstorm_xensorsharp_bridge import (
    NativeCrossPlatformWorkerSpec,
    PCIeMoEStreamingPlan,
    build_native_worker_fleet_manifests,
    compute_xlayerstorm_turboquant_schedule,
)


def test_s34_02_game_preemption_flush_receipt_validation() -> None:
    cb = GamePreemptionEF21CircuitBreaker()
    rec = cb.trigger_game_preemption(
        trigger_process="cyberpunk2077.exe",
        allocated_vram_mib=1024,
        inflight_gradient=[0.5, -0.2, 0.1, 0.8],
        prev_error_buffer=[0.0, 0.0, 0.0, 0.0],
        cuda_device_available=True,
    )
    assert rec.invariant_4_satisfied and rec.residual_vram_after_mib == 0

    with pytest.raises(ValueError, match="trigger_process must be a non-empty"):
        GamePreemptionFlushReceipt(
            "",
            True,
            50.0,
            1024,
            0,
            (0.5,),
            (0.0,),
            0.0,
            True,
            "a" * 64,
        )
    with pytest.raises(ValueError, match="preempted must be a boolean"):
        GamePreemptionFlushReceipt(
            "game.exe",
            1,  # type: ignore[arg-type]
            50.0,
            1024,
            0,
            (0.5,),
            (0.0,),
            0.0,
            True,
            "a" * 64,
        )
    with pytest.raises(ValueError, match="eviction_latency_ms must be a finite non-negative"):
        GamePreemptionFlushReceipt(
            "game.exe",
            True,
            -1.0,
            1024,
            0,
            (0.5,),
            (0.0,),
            0.0,
            True,
            "a" * 64,
        )
    with pytest.raises(ValueError, match="vram_before_mib must be a non-negative integer"):
        GamePreemptionFlushReceipt(
            "game.exe",
            True,
            50.0,
            -1,
            0,
            (0.5,),
            (0.0,),
            0.0,
            True,
            "a" * 64,
        )
    with pytest.raises(ValueError, match="ef21_top_k_transmitted and ef21_host_ram_error_buffer"):
        GamePreemptionFlushReceipt(
            "game.exe",
            True,
            50.0,
            1024,
            0,
            (),
            (),
            0.0,
            True,
            "a" * 64,
        )
    with pytest.raises(ValueError, match="state_digest_sha256 must be a 64-character hex"):
        GamePreemptionFlushReceipt(
            "game.exe",
            True,
            50.0,
            1024,
            0,
            (0.5,),
            (0.0,),
            0.0,
            True,
            "bad_sha",
        )


def test_s34_03_e8_quantized_block_validation() -> None:
    q = GossetE8LatticeQuantizer()
    blk = q.quantize_8d([0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.2])
    assert blk.is_valid_e8_point

    z8 = (0.0,) * 8
    with pytest.raises(ValueError, match="lattice_point must be an 8-element tuple"):
        E8QuantizedBlock((0.0, 1.0), 1.0, z8, 0.0, True, "D8_integer")
    with pytest.raises(ValueError, match="scale must be a finite non-negative"):
        E8QuantizedBlock(z8, -0.5, z8, 0.0, True, "D8_integer")
    with pytest.raises(ValueError, match="is_valid_e8_point must be a boolean"):
        E8QuantizedBlock(z8, 1.0, z8, 0.0, 1, "D8_integer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coset must be"):
        E8QuantizedBlock(z8, 1.0, z8, 0.0, True, "bad_coset")


def test_s35_01_a2_rope_and_amcp_latent_envelope_validation() -> None:
    rep = benchmark_a2_rope_plane_quantizer([(0.12, -0.34), (0.56, 0.78)])
    assert rep.num_2d_planes == 2

    with pytest.raises(ValueError, match="num_2d_planes must be an integer >= 1"):
        A2RoPEQuantizationReport(0, 0.1, 0.1, 0.0, 3.77, True)
    with pytest.raises(ValueError, match="mse_z2 must be a finite non-negative"):
        A2RoPEQuantizationReport(2, -0.1, 0.1, 0.0, 3.77, True)
    with pytest.raises(ValueError, match="empirical_mse_reduction_pct must be a finite"):
        A2RoPEQuantizationReport(2, 0.1, 0.1, float("nan"), 3.77, True)
    with pytest.raises(ValueError, match="beats_z2_in_mse must be a boolean"):
        A2RoPEQuantizationReport(2, 0.1, 0.1, 0.0, 3.77, "yes")  # type: ignore[arg-type]

    env = pack_amcp_xswat_latent_kv_envelope(
        "agent-a",
        "agent-b",
        [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]],
    )
    assert env.num_tokens == 1

    with pytest.raises(ValueError, match="num_tokens must be an integer >= 1"):
        AMCPLatentKVCacheEnvelope("a", "b", 0, 8, 2.5, 16, 3, 5.3, 0.01, 0.02, 50.0, "a" * 64)
    with pytest.raises(ValueError, match="mse_e8_klt must be >= 0.0"):
        AMCPLatentKVCacheEnvelope("a", "b", 1, 8, 2.5, 16, 3, 5.3, -0.01, 0.02, 50.0, "a" * 64)
    with pytest.raises(ValueError, match="payload_sha256 must be a 64-character hex"):
        AMCPLatentKVCacheEnvelope("a", "b", 1, 8, 2.5, 16, 3, 5.3, 0.01, 0.02, 50.0, "short")


def test_s35_04_xlayerstorm_plan_and_worker_spec_validation() -> None:
    plan = compute_xlayerstorm_turboquant_schedule()
    assert plan.within_citizen_vram_cap

    with pytest.raises(ValueError, match="total_experts must be an integer >= 1"):
        PCIeMoEStreamingPlan(0, 1, 80.0, 600.0, 300.0, 160.0, 1200.0, 336.0, 14.0, 10.0, 10.0, 0.0, 100.0, True)
    with pytest.raises(ValueError, match="active_top_k_experts must satisfy"):
        PCIeMoEStreamingPlan(8, 9, 80.0, 600.0, 300.0, 160.0, 1200.0, 336.0, 14.0, 10.0, 10.0, 0.0, 100.0, True)
    with pytest.raises(ValueError, match="vram_headroom_mib must be a finite"):
        PCIeMoEStreamingPlan(8, 2, 80.0, 600.0, 300.0, 160.0, 1200.0, float("nan"), 14.0, 10.0, 10.0, 0.0, 100.0, True)
    with pytest.raises(ValueError, match="within_citizen_vram_cap must be a boolean"):
        PCIeMoEStreamingPlan(8, 2, 80.0, 600.0, 300.0, 160.0, 1200.0, 336.0, 14.0, 10.0, 10.0, 0.0, 100.0, 1)  # type: ignore[arg-type]

    specs = build_native_worker_fleet_manifests()
    assert len(specs) == 4
    with pytest.raises(ValueError, match="os_family must be a non-empty string"):
        NativeCrossPlatformWorkerSpec("", "eng", False, 1024.0, 400.0, "http://127.0.0.1:8787/v1")
    with pytest.raises(ValueError, match="openai_compatible_endpoint must start with"):
        NativeCrossPlatformWorkerSpec("Linux", "eng", False, 1024.0, 400.0, "ftp://127.0.0.1")
    with pytest.raises(ValueError, match="requires_python_or_wsl must be a boolean"):
        NativeCrossPlatformWorkerSpec("Linux", "eng", "no", 1024.0, 400.0, "http://127.0.0.1:8787/v1")  # type: ignore[arg-type]
