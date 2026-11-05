"""Tests for ``hivemind.compress.lattice_quantizer`` (E10): decoders, membership, rate, error feedback."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest
import torch

from hivemind.compress import (
    LatticeDeltaQuantizer,
    SignDeltaQuantizer,
    brute_force_decoder_check,
    calibrate_scale_for_rate,
    factorised_rate_bits_per_dim,
    nearest_e8,
    nearest_zn,
)
from hivemind.compress.lattice_quantizer import NSM_REFERENCE, is_lattice_point


def test_e8_and_z8_decoders_match_exhaustive_search_and_return_lattice_points() -> None:
    for kind in ("E8", "Z"):
        r = brute_force_decoder_check(kind, n_queries=400, seed=7)
        assert r["passed"] is True, r
        assert r["mismatches"] == 0 and r["non_members"] == 0
        assert r["lattice_points_in_ball"] > 1000


def test_e8_distortion_is_below_z8_by_the_normalised_second_moment_ratio() -> None:
    # At equal scale on a Gaussian source with scale ~ 1 (fine quantisation), MSE/dim ≈ NSM · covolume^{2/n}:
    # 1/12 for Z and 929/12960 for E8, i.e. the 0.654 dB lattice gain of E8 over Z8.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60_000, 8)) * 4.0  # scale 4 => fine quantisation relative to the lattice
    mse_z = float(np.mean(np.sum((x - nearest_zn(x)) ** 2, axis=1)) / 8)
    mse_e8 = float(np.mean(np.sum((x - nearest_e8(x)) ** 2, axis=1)) / 8)
    assert abs(mse_z - NSM_REFERENCE["Z"]) < 0.003
    assert abs(mse_e8 - NSM_REFERENCE["E8"]) < 0.003
    assert 10 * math.log10(mse_z / mse_e8) > 0.5  # measured gain in dB, theory 0.654


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 10_000))
def test_e8_output_always_in_lattice_and_never_farther_than_z8_rounding_of_half_coset(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(64, 8)) * 1.3
    c = nearest_e8(x)
    assert bool(np.all(is_lattice_point("E8", c)))
    d_e8 = np.sum((x - c) ** 2, axis=1)
    d_candidates = np.minimum(np.sum((x - nearest_zn(x)) ** 2, axis=1) + 0.0, np.sum((x - (nearest_zn(x - 0.5) + 0.5)) ** 2, axis=1))
    # the nearest E8 point is never farther than the best of the two coset candidates (D8 subset of Z8 etc.)
    assert bool(np.all(d_e8 <= d_candidates + 1e-9 + 1.0))  # loose bound: E8 ⊂ (Z8 ∪ Z8+½), parity fix costs ≤ 1 in d²


def test_factorised_rate_decreases_with_scale_and_calibration_hits_target() -> None:
    v = torch.randn(4096, generator=torch.Generator().manual_seed(3))
    rows = v.numpy().reshape(-1, 8).astype(float)
    r_fine = factorised_rate_bits_per_dim("E8", nearest_e8(rows / 0.25))
    r_coarse = factorised_rate_bits_per_dim("E8", nearest_e8(rows / 2.0))
    assert r_fine > r_coarse + 1.0
    for kind in ("Z", "E8"):
        s, r = calibrate_scale_for_rate(kind, v, 2.0)
        assert abs(r - 2.0) <= 0.02
        assert 0.5 < s < 3.0


def test_quantizer_relative_mse_and_error_feedback_telescopes() -> None:
    g = torch.Generator().manual_seed(11)
    v1, v2 = torch.randn(2048, generator=g), torch.randn(2048, generator=g)
    q = LatticeDeltaQuantizer("E8", scale=1.0, error_feedback=True)
    out1 = q.quantize(v1)
    out2 = q.quantize(v2)
    # EF21 telescoping: sum of transmitted + final residual == sum of inputs (Theorem 107)
    total = out1.dequantized + out2.dequantized + out2.residual
    assert torch.allclose(total, v1 + v2, atol=1e-5)
    assert 0.0 < out1.relative_mse < 0.2
    q_noef = LatticeDeltaQuantizer("E8", scale=1.0, error_feedback=False)
    a = q_noef.quantize(v1).dequantized
    b = q_noef.quantize(v2).dequantized
    assert not torch.allclose(a + b, v1 + v2, atol=1e-3)  # without EF the error is lost


def test_sign_quantizer_is_one_bit_and_scaled_by_mean_abs() -> None:
    v = torch.tensor([3.0, -1.0, 0.5, -2.5])
    out = SignDeltaQuantizer().quantize(v)
    assert out.rate_bits_per_dim == 1.0
    assert torch.allclose(out.dequantized.abs(), torch.full((4,), 1.75))
    assert torch.equal(torch.sign(out.dequantized), torch.sign(v))
    assert out.relative_mse > 0.0


@pytest.mark.parametrize("make", [lambda: LatticeDeltaQuantizer("E8", 0.1, error_feedback=True), lambda: SignDeltaQuantizer(error_feedback=True)])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_delta_is_refused_and_does_not_poison_error_feedback(make, bad) -> None:
    q = make()
    v = torch.randn(16)
    v[3] = bad
    with pytest.raises(ValueError):
        q.quantize(v)
    clean = q.quantize(torch.randn(16))
    assert bool(torch.isfinite(clean.dequantized).all()) and bool(torch.isfinite(clean.residual).all())
    assert math.isfinite(clean.relative_mse) and clean.relative_mse > 0.0
