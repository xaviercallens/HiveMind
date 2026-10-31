"""Unit tests for HM-S36-02 E21 Multi-Krum + 10% Replay vs ZK-OPoI v5 Crossover Benchmark."""

from __future__ import annotations

from hivemind.swarm.s36_02_e21_multikrum_vs_zk_crossover import run_e21_multikrum_vs_zk_crossover


def test_e21_crossover_benchmark_proves_zk_dominance_and_energy_reduction() -> None:
    report = run_e21_multikrum_vs_zk_crossover()
    assert report.gate_passed is True
    assert report.measured_single_zk_verify_ms < 5.0
    assert report.measured_proof_bytes < 32_768
    assert report.energy_reduction_at_10k_pct >= 99.0
    assert len(report.rows) == 7
    assert report.rows[-1].swarm_nodes_n == 10_000
    assert report.rows[-1].zk_dominates is True
    assert report.rows[-1].speedup_ratio > 100.0
    as_dict = report.to_dict()
    assert as_dict["experiment_id"] == "E21-MULTIKRUM-VS-ZK-OPOI-V5"
