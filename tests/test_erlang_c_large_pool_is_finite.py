"""Erlang-C must stay finite and in [0, 1] for large producer pools (the direct a^k/k! sum overflowed near c = 925-1450)."""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from hivemind.economy.international_launch_model import compute_erlang_c_metrics


def test_thousand_producers_at_ninety_percent_load_is_finite() -> None:
    mu = 50000.0 / 168.3
    d = compute_erlang_c_metrics(1000, 50000.0, 0.9 * mu * 1.0)
    assert math.isfinite(d.erlang_c_queue_prob) and 0.0 <= d.erlang_c_queue_prob <= 1.0
    assert math.isfinite(d.expected_queue_wait_s) and math.isfinite(d.p95_total_latency_s)


def test_single_server_matches_closed_form_rho() -> None:
    d = compute_erlang_c_metrics(1, 168.3, 0.5)  # mu = 1 req/s, rho = 0.5: M/M/1 waiting probability equals rho
    assert d.erlang_c_queue_prob == 0.5


def test_non_finite_inputs_are_rejected() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            compute_erlang_c_metrics(10, 1000.0, bad)


@settings(max_examples=40, deadline=None)
@given(c=st.integers(1, 10000), rho=st.floats(0.01, 0.99))
def test_probability_is_finite_and_in_unit_interval(c: int, rho: float) -> None:
    agg = 100.0 * c
    d = compute_erlang_c_metrics(c, agg, rho * agg / 168.3)
    assert math.isfinite(d.erlang_c_queue_prob) and 0.0 <= d.erlang_c_queue_prob <= 1.0
