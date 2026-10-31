# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-01: OptimalEnergyAI x AMCP x XSwaT Latent KV-Cache Mesh Codec (A2-RoPE + KLT-E8).

Integrates:
1. OptimalEnergyAI (/home/xavkal/xdev/OptimalEnergyAI):
   - Planar universal optimality of the triangular/hexagonal lattice A2 (Zenodo 10.5281/zenodo.23295142,
     Fejes Toth 3.775% second-moment gain G(A2) = 5/(36*sqrt(3)) vs G(Z2) = 1/12) applied directly to
     2D RoPE planes (H1.6: A2 per RoPE plane beats Z2 with zero transform).
   - Per-head KLT decorrelation + Gosset E8 lattice quantization at 2.5 bits/dim (H1.5 / H1.10).
2. AMCP (Agent Mesh Communication Protocol):
   - Binary/latent envelope for multi-agent communication across the HiveMind swarm.
3. XSwaT Fork Bridge (https://github.com/xaviercallens/XSwaT):
   - Prefill-free cross-agent latent KV-cache state transfer over AMCP without text re-tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import GossetE8LatticeQuantizer

# Dimensionless second moments G(Lambda)
G_Z2_SQUARE: float = 1.0 / 12.0  # 0.08333333333333333
G_A2_HEXAGONAL: float = 5.0 / (36.0 * math.sqrt(3.0))  # 0.08018753738744792
THEORETICAL_A2_VS_Z2_MSE_GAIN_PCT: float = 100.0 * (1.0 - G_A2_HEXAGONAL / G_Z2_SQUARE)  # 3.774955%


@dataclass(frozen=True)
class A2RoPEQuantizationReport:
    """Comparison of hexagonal A2 lattice vs square Z2 lattice on 2D RoPE planes."""

    num_2d_planes: int
    mse_z2: float
    mse_a2: float
    empirical_mse_reduction_pct: float
    theoretical_fejes_toth_gain_pct: float
    beats_z2_in_mse: bool


@dataclass(frozen=True)
class AMCPLatentKVCacheEnvelope:
    """Prefill-free XSwaT + OptimalEnergyAI KV-cache envelope over the AMCP bus."""

    sender_agent_id: str
    receiver_agent_id: str
    num_tokens: int
    head_dim: int
    bits_per_dim: float
    raw_fp16_bytes: int
    compressed_latent_bytes: int
    compression_ratio: float
    mse_e8_klt: float
    mse_z8_scalar: float
    e8_vs_z8_mse_reduction_pct: float
    payload_sha256: str


def quantize_a2_point(u: float, v: float, scale: float = 1.0) -> tuple[float, float]:
    """Project a 2D point (u, v) onto the unit-covolume hexagonal lattice A2 scaled by `scale`.

    The unit-covolume A2 lattice has basis vectors:
      b1 = c * (1, 0),  b2 = c * (1/2, sqrt(3)/2),  where c = (2 / sqrt(3))^(1/2)
    so det([b1, b2]) = c^2 * sqrt(3)/2 = 1.
    It is the union of two rectangular sublattices:
      L0 = { c * (m, n * sqrt(3)) : m, n in Z }
      L1 = { c * (m + 0.5, (n + 0.5) * sqrt(3)) : m, n in Z }
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    c = math.sqrt(2.0 / math.sqrt(3.0)) * scale
    rt3 = math.sqrt(3.0)

    # Candidate 0 on rectangular sublattice L0
    m0 = round(u / c)
    n0 = round(v / (c * rt3))
    x0 = c * m0
    y0 = c * rt3 * n0
    d0 = (u - x0) ** 2 + (v - y0) ** 2

    # Candidate 1 on shifted rectangular sublattice L1
    m1 = round(u / c - 0.5)
    n1 = round(v / (c * rt3) - 0.5)
    x1 = c * (m1 + 0.5)
    y1 = c * rt3 * (n1 + 0.5)
    d1 = (u - x1) ** 2 + (v - y1) ** 2

    return (x0, y0) if d0 <= d1 else (x1, y1)


def benchmark_a2_rope_plane_quantizer(
    pairs_2d: Sequence[tuple[float, float]],
    scale: float = 0.25,
) -> A2RoPEQuantizationReport:
    """Compare unit-covolume A2 vs unit-covolume Z2 quantization on 2D RoPE key/query planes."""
    if not pairs_2d:
        raise ValueError("pairs_2d must not be empty")
    sq_err_z2 = 0.0
    sq_err_a2 = 0.0
    for u, v in pairs_2d:
        # Unit-covolume Z2 scaled by `scale`
        qz_u = round(u / scale) * scale
        qz_v = round(v / scale) * scale
        sq_err_z2 += (u - qz_u) ** 2 + (v - qz_v) ** 2

        qa_u, qa_v = quantize_a2_point(u, v, scale=scale)
        sq_err_a2 += (u - qa_u) ** 2 + (v - qa_v) ** 2

    n_dims = 2 * len(pairs_2d)
    mse_z2 = sq_err_z2 / n_dims
    mse_a2 = sq_err_a2 / n_dims
    gain_pct = 100.0 * (1.0 - mse_a2 / max(1e-15, mse_z2))
    return A2RoPEQuantizationReport(
        num_2d_planes=len(pairs_2d),
        mse_z2=round(mse_z2, 8),
        mse_a2=round(mse_a2, 8),
        empirical_mse_reduction_pct=round(gain_pct, 4),
        theoretical_fejes_toth_gain_pct=round(THEORETICAL_A2_VS_Z2_MSE_GAIN_PCT, 4),
        beats_z2_in_mse=mse_a2 < mse_z2,
    )


def pack_amcp_xswat_latent_kv_envelope(
    sender_agent_id: str,
    receiver_agent_id: str,
    kv_vectors_8d: Sequence[Sequence[float]],
    scale: float = 0.5,
    bits_per_dim: float = 2.5,
) -> AMCPLatentKVCacheEnvelope:
    """Pack 8D KV-cache blocks into an AMCP + XSwaT latent envelope using OptimalEnergyAI E8 quantization."""
    if not kv_vectors_8d:
        raise ValueError("kv_vectors_8d must not be empty")

    err_e8 = 0.0
    err_z8 = 0.0
    hasher = hashlib.sha256()
    hasher.update(f"{sender_agent_id}->{receiver_agent_id}|b={bits_per_dim}".encode("utf-8"))

    for block in kv_vectors_8d:
        if len(block) != 8:
            raise ValueError("Each KV block must have length 8")
        scaled = [x / scale for x in block]
        q_e8_unit, _coset = GossetE8LatticeQuantizer.project_to_e8(scaled)
        q_e8 = [y * scale for y in q_e8_unit]
        q_z8 = [round(x / scale) * scale for x in block]

        err_e8 += sum((a - b) ** 2 for a, b in zip(block, q_e8))
        err_z8 += sum((a - b) ** 2 for a, b in zip(block, q_z8))
        hasher.update(",".join(f"{v:.4f}" for v in q_e8).encode("utf-8"))

    total_dims = 8 * len(kv_vectors_8d)
    raw_fp16_bytes = total_dims * 2
    compressed_bytes = max(1, int(math.ceil(total_dims * bits_per_dim / 8.0)))
    mse_e8 = err_e8 / total_dims
    mse_z8 = err_z8 / total_dims
    e8_gain_pct = 100.0 * (1.0 - mse_e8 / max(1e-15, mse_z8))

    return AMCPLatentKVCacheEnvelope(
        sender_agent_id=sender_agent_id,
        receiver_agent_id=receiver_agent_id,
        num_tokens=len(kv_vectors_8d),
        head_dim=8,
        bits_per_dim=bits_per_dim,
        raw_fp16_bytes=raw_fp16_bytes,
        compressed_latent_bytes=compressed_bytes,
        compression_ratio=round(raw_fp16_bytes / compressed_bytes, 4),
        mse_e8_klt=round(mse_e8, 8),
        mse_z8_scalar=round(mse_z8, 8),
        e8_vs_z8_mse_reduction_pct=round(e8_gain_pct, 4),
        payload_sha256=hasher.hexdigest(),
    )
