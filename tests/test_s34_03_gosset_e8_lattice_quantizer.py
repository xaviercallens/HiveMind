# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S34-03 Gosset E8 8D Lattice Vector Quantizer."""

from __future__ import annotations

import math

from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import (
    GossetE8LatticeQuantizer,
    is_valid_e8_lattice_point,
)


def test_gosset_e8_quantizer_projects_to_valid_even_sum_and_half_integer_cosets() -> None:
    q = GossetE8LatticeQuantizer(codebook_radius=4.0)
    # Vector close to half-integer coset (0.5, ..., 0.5)
    v_half = [0.51, 0.49, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50]
    block_half = q.quantize_8d(v_half)
    assert block_half.is_valid_e8_point is True

    # Arbitrary LoRA weight vector
    v_lora = [0.18, -0.42, 0.31, -0.09, 0.55, -0.27, 0.14, -0.36]
    block_lora = q.quantize_8d(v_lora)
    assert block_lora.is_valid_e8_point is True
    orig_norm = math.sqrt(sum(x * x for x in v_lora))
    assert block_lora.l2_distortion < 0.25 * orig_norm


def test_e8_validity_checker_rejects_odd_parity_integer_vector() -> None:
    assert is_valid_e8_lattice_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is False
    assert is_valid_e8_lattice_point([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True
    assert is_valid_e8_lattice_point([0.5] * 8) is True
