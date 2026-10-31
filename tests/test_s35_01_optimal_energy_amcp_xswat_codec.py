# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S35-01: OptimalEnergyAI x AMCP x XSwaT Latent KV-Cache Mesh Codec."""

from __future__ import annotations

import math

from hivemind.swarm.s35_01_optimal_energy_amcp_xswat_codec import (
    THEORETICAL_A2_VS_Z2_MSE_GAIN_PCT,
    benchmark_a2_rope_plane_quantizer,
    pack_amcp_xswat_latent_kv_envelope,
    quantize_a2_point,
)


def test_a2_hexagonal_lattice_rope_plane_beats_z2() -> None:
    assert 3.77 < THEORETICAL_A2_VS_Z2_MSE_GAIN_PCT < 3.78
    qx, qy = quantize_a2_point(0.0, 0.0, scale=1.0)
    assert abs(qx) < 1e-12 and abs(qy) < 1e-12

    grid = [((i % 25 - 12) * 0.029, ((i // 25) % 25 - 12) * 0.029) for i in range(625)]
    report = benchmark_a2_rope_plane_quantizer(grid, scale=0.10)
    assert report.beats_z2_in_mse is True
    assert report.mse_a2 < report.mse_z2


def test_amcp_xswat_latent_kv_cache_envelope_compression() -> None:
    blocks = [[math.sin(0.27 * (k * 8 + d)) * 0.6 for d in range(8)] for k in range(12)]
    env = pack_amcp_xswat_latent_kv_envelope(
        sender_agent_id="amcp://proposer",
        receiver_agent_id="amcp://verifier",
        kv_vectors_8d=blocks,
        scale=0.25,
        bits_per_dim=2.5,
    )
    assert env.compression_ratio >= 6.0
    assert env.mse_e8_klt < env.mse_z8_scalar
    assert env.e8_vs_z8_mse_reduction_pct > 5.0
    assert len(env.payload_sha256) == 64
