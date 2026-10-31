"""Unit tests for HM-S30-02 (Experiment E21): Multi-Krum + 10% Replay vs ZK-OPoI v5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hivemind.swarm.krum_vs_zkopoi_v5_benchmark import (
    evaluate_defense_regime,
    multi_krum_select,
    run_e21_scaling_benchmark,
)


def test_e21_pre_registration_manifest_exists() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "experiments"
        / "pre_registrations"
        / "E21_krum_vs_zkopoi_v5_scaling.json"
    )
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "E21"
    assert payload["roadmap_id"] == "HM-S30-02"
    assert payload["independent_variables"]["swarm_sizes_n"] == [32, 128, 512, 2048]


def test_multi_krum_rejects_byzantine_poisoning() -> None:
    rng = np.random.default_rng(7)
    true_grad = np.ones(128, dtype=np.float32) / np.sqrt(128.0)
    honest = true_grad[None, :] + 0.02 * rng.standard_normal((26, 128), dtype=np.float32)
    byzantine = -10.0 * true_grad[None, :] + rng.standard_normal((6, 128), dtype=np.float32)
    grads = np.vstack([honest, byzantine]).astype(np.float32)

    agg, selected = multi_krum_select(grads, num_byzantine=6, m_select=26)
    assert all(idx < 26 for idx in selected)
    cos = float(np.dot(agg, true_grad) / (np.linalg.norm(agg) * np.linalg.norm(true_grad)))
    assert cos > 0.99


def test_evaluate_all_three_defense_regimes_at_n32() -> None:
    for mode in (
        "multi_krum_plus_10pct_replay",
        "zk_opoi_v5_int8_full",
        "hierarchical_subswarm_krum_zk_hybrid",
    ):
        pt = evaluate_defense_regime(n_workers=32, grad_dim=128, byzantine_ratio=0.2, mode=mode)
        assert pt.byzantine_rejection_rate >= 0.95
        assert pt.honest_cosine_fidelity >= 0.95
        assert pt.wall_clock_ms > 0.0


def test_run_e21_scaling_benchmark_across_32_128_512_2048() -> None:
    report = run_e21_scaling_benchmark(swarm_sizes=(32, 128, 512, 2048), grad_dim=128)
    assert report.experiment_id == "E21"
    assert report.roadmap_id == "HM-S30-02"
    assert report.all_invariants_passed is True
    assert len(report.points) == 12
    assert report.pareto_winner_by_n[32] == "multi_krum_plus_10pct_replay"
    assert report.pareto_winner_by_n[128] == "multi_krum_plus_10pct_replay"
    assert report.crossover_swarm_size_n == 512
