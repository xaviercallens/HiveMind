# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Outer-delta compression: lattice (ℤⁿ, Dₙ, E₈) and 1-bit sign quantisers with error feedback (E10)."""

from hivemind.compress.lattice_quantizer import (
    LATTICE_QUANTIZER_REFERENCES,
    LatticeDeltaQuantizer,
    QuantizedDelta,
    SignDeltaQuantizer,
    brute_force_decoder_check,
    calibrate_scale_for_rate,
    factorised_rate_bits_per_dim,
    nearest_dn,
    nearest_e8,
    nearest_zn,
)

__all__ = [
    "LATTICE_QUANTIZER_REFERENCES",
    "LatticeDeltaQuantizer",
    "QuantizedDelta",
    "SignDeltaQuantizer",
    "brute_force_decoder_check",
    "calibrate_scale_for_rate",
    "factorised_rate_bits_per_dim",
    "nearest_dn",
    "nearest_e8",
    "nearest_zn",
]
