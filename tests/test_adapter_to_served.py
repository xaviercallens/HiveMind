# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the train-to-serve gate (scripts/adapter_to_served.py): pure logic, no model, no llama.cpp."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adapter_to_served as gate  # noqa: E402


def _chunks(n: int, mean: float, spread: float = 0.2, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    return [mean + rng.uniform(-spread, spread) for _ in range(n)]


def _per(shift_by_domain: dict[str, float], n: int = 64) -> dict:
    """Base NLL per chunk, adapter = base + shift (negative shift = adapter better), paired by chunk."""
    out = {}
    for i, (d, shift) in enumerate(shift_by_domain.items()):
        base = _chunks(n, 2.0, seed=i + 1)
        out[d] = {"base": base, "adapter": [b + shift for b in base]}
    return out


def test_nll_is_recovered_from_running_perplexity() -> None:
    true = [2.0, 2.4, 1.6, 2.2, 1.8]
    running, total = [], 0.0
    for i, x in enumerate(true, start=1):
        total += x
        running.append(round(math.exp(total / i), 4))  # llama-perplexity prints 4 decimals
    got = gate.nll_from_running(running)
    assert all(abs(g - t) < 0.01 for g, t in zip(got, true))
    assert abs(sum(got) / len(got) - sum(true) / len(true)) < 1e-3


def test_a_clear_gain_everywhere_is_servable() -> None:
    v = gate.decide(_per({"a": -0.05, "b": -0.05, "c": -0.05}))
    assert v["servable"] is True and v["pooled"]["mean_nll_gain_nats"] == pytest.approx(0.05, abs=1e-4) and v["pooled"]["bootstrap95"][0] > 0


def test_no_gain_is_not_servable() -> None:
    v = gate.decide(_per({"a": 0.0, "b": 0.0}))
    assert v["servable"] is False and v["pooled"]["mean_nll_gain_nats"] == 0


def test_a_gain_whose_interval_includes_zero_is_not_servable() -> None:
    # deterministic: chunk differences alternate +0.30 / -0.299, so the mean gain is tiny and positive and the spread is huge
    per = {"a": {"base": [2.0] * 64, "adapter": [2.0 - (0.30 if i % 2 == 0 else -0.299) for i in range(64)]}}
    v = gate.decide(per)
    assert v["pooled"]["mean_nll_gain_nats"] > 0 and v["pooled"]["bootstrap95"][0] < 0 and v["servable"] is False


def test_one_domain_more_than_one_percent_worse_blocks_even_a_good_pooled_gain() -> None:
    # big gains on two domains, a bit over +1% on the third (constant base 2.0, so +0.0210 nats is +1.05%)
    per = _per({"a": -0.2, "b": -0.2})
    per["c"] = {"base": [2.0] * 64, "adapter": [2.0210] * 64}
    v = gate.decide(per)
    assert v["pooled"]["mean_nll_gain_nats"] > 0 and v["pooled"]["bootstrap95"][0] > 0
    assert v["worst_domain"] == "c" and v["worst_domain_regression"] > 0.01 and v["servable"] is False


def test_a_domain_exactly_within_the_limit_is_allowed() -> None:
    per = _per({"a": -0.2, "b": -0.2})
    per["c"] = {"base": [2.0] * 64, "adapter": [2.0190] * 64}  # +0.95%
    v = gate.decide(per)
    assert v["worst_domain_regression"] <= 0.01 and v["servable"] is True


def test_nothing_to_compare_is_an_error_not_a_pass() -> None:
    with pytest.raises(ValueError):
        gate.decide({"a": {"base": [], "adapter": []}})


def test_the_selection_chunks_are_skipped_and_the_rule_is_recorded() -> None:
    assert gate.SKIP_CHUNKS >= 8  # the chain evaluates its best-of-N pick on the first 8 chunks of each stream
    assert "SKIP_CHUNKS" in Path(gate.__file__).read_text(encoding="utf-8")


def test_the_gate_is_deterministic() -> None:
    per = _per({"a": -0.01, "b": 0.005})
    assert gate.decide(per) == gate.decide(per)
