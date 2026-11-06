# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for IP-20 (#385): Multi-Krum reference schema derived from the round, not the first worker."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.diloco_outer import DiLoCoOuterOptimizer, filter_byzantine_deltas_multikrum


def test_multikrum_attacker_first_and_attacker_last_quarantines_only_attacker() -> None:
    honest = {
        f"honest-{i}": {
            "a": torch.full((2, 2), 0.1 + 0.01 * i),
            "b": torch.full((2,), 0.2 + 0.01 * i),
        }
        for i in range(5)
    }
    attacker = {"attacker": {"a": torch.full((2, 2), 0.1)}}

    report_first = filter_byzantine_deltas_multikrum({**attacker, **honest})
    report_last = filter_byzantine_deltas_multikrum({**honest, **attacker})

    assert list(report_first.quarantined_pre_krum.keys()) == ["attacker"]
    assert list(report_last.quarantined_pre_krum.keys()) == ["attacker"]
    assert "attacker" not in report_first.accepted_workers
    assert "attacker" not in report_last.accepted_workers
    assert set(report_first.accepted_workers) == set(report_last.accepted_workers)
    assert len(report_first.accepted_workers) >= 4


def test_diloco_outer_step_attacker_first_and_subset_keys_never_raises_keyerror() -> None:
    theta_prev = {"a": torch.full((2, 2), 2.0), "b": torch.full((2,), 4.0)}
    honest = {
        f"honest-{i}": {
            "a": torch.full((2, 2), 0.1),
            "b": torch.full((2,), 0.2),
        }
        for i in range(5)
    }
    attacker = {"attacker": {"a": torch.full((2, 2), 0.1)}}

    opt = DiLoCoOuterOptimizer()
    res = opt.step("STEM", 1, theta_prev, {**attacker, **honest})
    assert list(res.filter_report.quarantined_pre_krum.keys()) == ["attacker"]
    assert "attacker" not in res.filter_report.accepted_workers
    assert set(res.theta_updated.keys()) == {"a", "b"}

    # Subset-keys-only attacker through step() must raise ValueError (all quarantined), never KeyError
    with pytest.raises(ValueError, match="quarantined"):
        opt.step("STEM", 2, theta_prev, attacker)


def test_diloco_outer_step_k2_early_accept_enforces_round_reference_schema() -> None:
    theta_prev = {"a": torch.full((2, 2), 1.0), "b": torch.full((2,), 1.0)}
    attacker_first = {
        "attacker": {"a": torch.full((2, 2), 0.5)},
        "honest": {"a": torch.full((2, 2), 0.1), "b": torch.full((2,), 0.2)},
    }
    opt = DiLoCoOuterOptimizer()
    res = opt.step("STEM", 1, theta_prev, attacker_first)
    assert res.filter_report.accepted_workers == ["honest"]
    assert list(res.filter_report.quarantined_pre_krum.keys()) == ["attacker"]
