# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S34-03 — Gosset E8 8D Lattice Vector Quantizer (30.89x Adapter Compression for La Ruche Hexagonal Alvéoles).

Implements exact nearest-lattice-point projection onto the 8-dimensional Gosset lattice ``E_8``
(Conway & Sloane, 1982, O(8 log 8) algorithm):
- ``E_8 = D_8 U (D_8 + (1/2, ..., 1/2))``
- ``D_8 = {z in Z^8 : sum(z_i) is even}``
Used in Project HiveMind to quantize LoRA / pseudo-gradient blocks into optimal 8D sphere-packing codewords,
achieving up to ``30.89x`` bandwidth reduction over raw FP32/FP16 when combined with Top-K sparsification.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from hivemind.compress.lattice_quantizer import nearest_dn, nearest_e8


@dataclass(frozen=True)
class E8QuantizedBlock:
    """Quantized 8D Gosset lattice block with scale factor and mathematical validity flags."""

    lattice_point: tuple[float, ...]
    scale: float
    reconstructed: tuple[float, ...]
    l2_distortion: float
    is_valid_e8_point: bool
    coset: str  # "D8_integer" or "D8_half_integer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_finite_8d(x: Sequence[float], label: str) -> None:
    if not isinstance(x, (list, tuple)) or len(x) != 8:
        raise ValueError(f"{label} requires an 8D vector")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in x):
        raise ValueError(f"{label} must contain only finite numbers")


def _nearest_d8(x: Sequence[float]) -> tuple[float, ...]:
    """Project an 8D vector onto the checkerboard lattice ``D_8 = {z in Z^8 : sum(z_i) even}`` via canonical ``nearest_dn``."""
    _require_finite_8d(x, "D8 projection")
    arr = np.asarray(x, dtype=np.float64).reshape(1, 8)
    q = nearest_dn(arr)[0]
    return tuple(float(r) for r in q.tolist())


def is_valid_e8_lattice_point(pt: Sequence[float], tol: float = 1e-9) -> bool:
    """Verify whether an 8D point lies on ``E_8 = D_8 U (D_8 + 1/2)``."""
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    if not isinstance(pt, (list, tuple)) or len(pt) != 8:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pt):
        return False
    # Check integer coset D_8
    if all(abs(v - round(v)) <= tol for v in pt):
        s = sum(int(round(v)) for v in pt)
        return (s % 2) == 0
    # Check half-integer coset D_8 + 0.5
    shifted = [v - 0.5 for v in pt]
    if all(abs(v - round(v)) <= tol for v in shifted):
        s = sum(int(round(v)) for v in shifted)
        return (s % 2) == 0
    return False


class GossetE8LatticeQuantizer:
    """Conway-Sloane O(8 log 8) Gosset E8 lattice vector quantizer for LoRA adapters (`artifacts/experiments/E10_lattice_diloco.json`)."""

    dense_e8_compression_ratio: float = 16.0
    reference_artifact: str = "artifacts/experiments/E10_lattice_diloco.json"

    def __init__(self, *, codebook_radius: float = 4.0) -> None:
        if (
            isinstance(codebook_radius, bool)
            or not isinstance(codebook_radius, (int, float))
            or not math.isfinite(codebook_radius)
            or codebook_radius <= 0.0
        ):
            raise ValueError("codebook_radius must be positive")
        self.codebook_radius = float(codebook_radius)

    @staticmethod
    def project_to_e8(x: Sequence[float]) -> tuple[tuple[float, ...], str]:
        """Exact nearest point in ``E_8 = D_8 U (D_8 + 0.5)`` via canonical ``nearest_e8``."""
        _require_finite_8d(x, "E8 projection")
        arr = np.asarray(x, dtype=np.float64).reshape(1, 8)
        winner_arr = nearest_e8(arr)[0]
        winner = tuple(float(v) for v in winner_arr.tolist())
        is_half = abs(winner[0] - round(winner[0])) > 1e-6
        return winner, ("D8_half_integer" if is_half else "D8_integer")

    def quantize_8d(self, vec: Sequence[float]) -> E8QuantizedBlock:
        """Scale and quantize an 8D parameter block onto the Gosset E8 lattice."""
        _require_finite_8d(vec, "vec")
        norm = math.sqrt(sum(float(v) * float(v) for v in vec))
        if norm < 1e-12:
            zero8 = (0.0,) * 8
            return E8QuantizedBlock(zero8, 0.0, zero8, 0.0, True, "D8_integer")

        scale = norm / self.codebook_radius
        scaled = [float(v) / scale for v in vec]
        lattice_pt, coset = self.project_to_e8(scaled)
        recon = tuple(c * scale for c in lattice_pt)
        distortion = math.sqrt(sum((float(vec[i]) - recon[i]) ** 2 for i in range(8)))
        return E8QuantizedBlock(
            lattice_point=lattice_pt,
            scale=scale,
            reconstructed=recon,
            l2_distortion=distortion,
            is_valid_e8_point=is_valid_e8_lattice_point(lattice_pt),
            coset=coset,
        )
