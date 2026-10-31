# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S33-01 Straggler-Free Decoupled DiLoCo with PA-CGAD Radial-Directional Averaging."""

from __future__ import annotations

import math
import pytest

from hivemind.swarm.s33_01_decoupled_diloco_pacgad import (
    StragglerFreeDecoupledDiLoCoAggregator,
    WorkerPseudoGradient,
)


def test_decoupled_diloco_pacgad_preserves_radial_norm_over_naive_cancellation() -> None:
    agg = StragglerFreeDecoupledDiLoCoAggregator(staleness_alpha=0.5, outer_lr=0.5, max_staleness_steps=8)
    # Orthogonal / out-of-phase consumer GPU pseudo-gradients of equal L2 norm = 2.0
    w1 = WorkerPseudoGradient("rtx2070-fr", staleness_steps=0, inner_steps=50, pseudo_gradient=(2.0, 0.0, 0.0, 0.0))
    w2 = WorkerPseudoGradient("rtx3060-de", staleness_steps=2, inner_steps=50, pseudo_gradient=(0.0, 2.0, 0.0, 0.0))
    res = agg.aggregate_and_step([1.0, 1.0, 1.0, 1.0], [w1, w2])

    agg_norm = math.sqrt(sum(x * x for x in res.aggregated_vector))
    assert agg_norm == pytest.approx(2.0, rel=1e-6)
    assert res.radial_norm == pytest.approx(2.0, rel=1e-6)
    assert res.naive_mean_norm < 1.50
    assert res.norm_preservation_ratio > 1.30


def test_decoupled_diloco_filters_excessive_staleness() -> None:
    agg = StragglerFreeDecoupledDiLoCoAggregator(max_staleness_steps=4)
    fresh = WorkerPseudoGradient("fresh-node", staleness_steps=1, inner_steps=40, pseudo_gradient=(1.0, 0.0))
    stale = WorkerPseudoGradient("stale-node", staleness_steps=12, inner_steps=40, pseudo_gradient=(-100.0, 0.0))
    res = agg.aggregate_and_step([0.0, 0.0], [fresh, stale])
    assert res.effective_worker_count == 1.0
    assert res.aggregated_vector[0] == pytest.approx(1.0)
