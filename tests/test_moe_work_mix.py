# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Stratified training rounds: every round draws equally from the six evaluated domains."""

from __future__ import annotations

import pytest

from hivemind.boinc.real_client import moe_work as mw


def test_mix_domains_are_the_six_the_chain_evaluates() -> None:
    assert mw.MIX_DOMAINS == ("general", "french", "edu_web", "news", "biomed", "math")


@pytest.mark.parametrize(("n", "k"), [(64, 6), (66, 6), (6, 6), (7, 6), (128, 6)])
def test_mix_counts_are_even_and_sum_to_n(n: int, k: int) -> None:
    c = mw.mix_counts(n, k)
    assert sum(c) == n and len(c) == k and max(c) - min(c) <= 1 and c == sorted(c, reverse=True)


def test_mix_counts_refuse_to_drop_a_domain() -> None:
    with pytest.raises(ValueError):
        mw.mix_counts(5, 6)
    with pytest.raises(ValueError):
        mw.mix_counts(10, 0)


def test_interleave_alternates_domains_and_loses_nothing() -> None:
    a, b, c = [[1], [2], [3]], [[10], [20]], [[100], [200], [300], [400]]
    out = mw.interleave([a, b, c])
    assert out == [[1], [10], [100], [2], [20], [200], [3], [300], [400]]
    assert sorted(map(tuple, out)) == sorted(map(tuple, a + b + c))
    assert mw.interleave([]) == []


def test_the_unknown_domain_error_mentions_mix() -> None:
    import inspect

    assert "MIX" in inspect.getsource(mw.build_train_spec) and mw.MIX == "mix"


def test_the_work_generator_exposes_the_flag() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts/t4_workgen.py").read_text(encoding="utf-8")
    assert '"--mix"' in src and "MIX" in src and "a.mix" in src
    assert "DOMAIN_ORDER[(rnd - 1) % len(DOMAIN_ORDER)]" in src  # the default stays one domain per round
