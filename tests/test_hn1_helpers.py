# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pure helpers of the HN1 boundary-signal experiment: AUROC, Spearman, adjacent cosine, labels."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_HN1_boundary_signal.py"
spec = importlib.util.spec_from_file_location("hn1_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
hn1 = importlib.util.module_from_spec(spec)
sys.modules["hn1_under_test"] = hn1
spec.loader.exec_module(hn1)


def test_auroc_perfect_reversed_and_chance() -> None:
    assert hn1.auroc([0.1, 0.2, 0.9, 0.8], [False, False, True, True]) == 1.0
    assert hn1.auroc([0.9, 0.8, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert hn1.auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert hn1.auroc([1.0, 2.0], [True, True]) == 0.5


def test_auroc_matches_pairwise_definition() -> None:
    scores = [0.3, 0.7, 0.7, 0.1, 0.9, 0.4]
    labels = [False, True, False, False, True, True]
    pos = [s for s, b in zip(scores, labels) if b]
    neg = [s for s, b in zip(scores, labels) if not b]
    pairwise = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))
    assert hn1.auroc(scores, labels) == pytest.approx(pairwise)


def test_spearman_monotone_and_constant() -> None:
    assert hn1.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert hn1.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert hn1.spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0
    with pytest.raises(ValueError):
        hn1.spearman([1.0], [2.0])


def test_adjacent_cosine_identical_orthogonal_and_zero() -> None:
    h = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    cos = hn1.adjacent_cosine(h)
    assert len(cos) == 3
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(0.0, abs=1e-7)
    assert cos[2] == 0.0
    assert all(math.isfinite(c) for c in cos)


def test_word_start_and_newline_labels() -> None:
    toks = ["Ġhello", "world", "Ċ", "def", " x", "\n"]
    assert hn1.word_start_labels(toks) == [True, False, True, False, True, True]
    assert hn1.newline_labels(toks) == [False, False, True, False, False, True]


def test_windows_of_drops_the_tail_and_caps() -> None:
    ids = list(range(1000))
    wins = hn1.windows_of(ids, 256, 8)
    assert len(wins) == 3
    assert wins[0] == list(range(256))
    assert hn1.windows_of(ids, 256, 2) == wins[:2]
    assert hn1.windows_of(list(range(100)), 256, 8) == []
