# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening tests for s35_02, s35_03, s36_02, and s36_04 dataclasses and entrypoints."""

from __future__ import annotations

import pytest

from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import (
    GWAYAA3RepairReport,
    GWAYACandidateAudit,
    run_gwaya_a3_sample_and_repair,
)
from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    LeanCommentStrippedAudit,
    TierXR64InstrumentReceipt,
    audit_lean4_source_elenchus,
    verify_robertson_sundials_tier_x,
)
from hivemind.swarm.s36_02_e21_multikrum_vs_zk_crossover import (
    E21CrossoverBenchmarkReport,
    E21CrossoverRow,
    run_e21_multikrum_vs_zk_crossover,
)
from hivemind.swarm.s36_04_fuzzy_subswarm_consensus import (
    FuzzySubSwarmConsensusReport,
    LocalizedFuzzySubSwarmConsensus,
    SubSwarmVote,
)


def test_s35_02_gwaya_audit_and_repair_report_validation() -> None:
    rep = run_gwaya_a3_sample_and_repair(
        ["def add(a, b):\n    return a + b\n"],
        lambda _s: True,
        lambda _s: True,
    )
    assert rep.final_a3_passed and len(rep.audit_trail) == 1
    aud = rep.audit_trail[0]

    with pytest.raises(ValueError, match="is_ast_clean must be a boolean"):
        GWAYACandidateAudit(1, True, True, 0.9, 0.9, 1.0, "VERIFIED", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="laya_calibrated_score must be a finite number"):
        GWAYACandidateAudit(True, True, True, 1.5, 0.9, 1.0, "VERIFIED", ())
    with pytest.raises(ValueError, match="anse_energy must be a finite non-negative"):
        GWAYACandidateAudit(True, True, True, 0.9, 0.9, -1.0, "VERIFIED", ())
    with pytest.raises(ValueError, match="verdict must be"):
        GWAYACandidateAudit(True, True, True, 0.9, 0.9, 1.0, "BAD", ())
    with pytest.raises(ValueError, match="violations must be a tuple of non-empty strings"):
        GWAYACandidateAudit(False, False, False, 0.0, 0.9, 1e6, "UNVERIFIED_MAX_PAIN", ("",))

    with pytest.raises(ValueError, match="initial_a0_passed must be a boolean"):
        GWAYAA3RepairReport("yes", True, 1, 1.0, "x=1", (aud,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rounds_used must be an integer >= 1"):
        GWAYAA3RepairReport(True, True, 0, 1.0, "x=1", (aud,))
    with pytest.raises(ValueError, match="final_energy must be a finite non-negative"):
        GWAYAA3RepairReport(True, True, 1, -1.0, "x=1", (aud,))
    with pytest.raises(ValueError, match="selected_code must be a string"):
        GWAYAA3RepairReport(True, True, 1, 1.0, 123, (aud,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="audit_trail must be a non-empty tuple"):
        GWAYAA3RepairReport(True, True, 1, 1.0, "x=1", ())


def test_s35_03_elenchus_lean_audit_and_tier_x_receipt_validation() -> None:
    la = audit_lean4_source_elenchus("-- no sorry here\ntheorem t1 : 1 = 1 := rfl\n")
    assert la.is_kernel_clean and la.classification == "PROVEN_INVARIANT"

    with pytest.raises(ValueError, match="raw_sorry_substring_hits must be a non-negative integer"):
        LeanCommentStrippedAudit(-1, 0, 0, 0, 0, 1, "PROVEN_INVARIANT", True)
    with pytest.raises(ValueError, match="classification must be"):
        LeanCommentStrippedAudit(0, 0, 0, 0, 0, 1, "BAD", True)
    with pytest.raises(ValueError, match="is_kernel_clean must be a boolean"):
        LeanCommentStrippedAudit(0, 0, 0, 0, 0, 1, "PROVEN_INVARIANT", 1)  # type: ignore[arg-type]

    rec = verify_robertson_sundials_tier_x(
        {"t_end": 10.0},
        {"t_end": 10.0},
        y_initial=(1.0, 0.0, 0.0),
        y_final=(0.8, 0.0001, 0.1999),
    )
    assert rec.accepted_by_elenchus_bridge

    with pytest.raises(ValueError, match="tier must be a non-empty string"):
        TierXR64InstrumentReceipt("", False, True, False, "tier_x_measured", True, 0.0, "a" * 64)
    with pytest.raises(ValueError, match="ran must be a boolean"):
        TierXR64InstrumentReceipt("X", False, 1, False, "tier_x_measured", True, 0.0, "a" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evidentiary_value must be"):
        TierXR64InstrumentReceipt("X", False, True, False, "bad", True, 0.0, "a" * 64)
    with pytest.raises(ValueError, match="mass_conservation_drift must be a non-negative"):
        TierXR64InstrumentReceipt("X", False, True, False, "tier_x_measured", True, -1.0, "a" * 64)
    with pytest.raises(ValueError, match="provenance_sha256 must be a 64-character hex"):
        TierXR64InstrumentReceipt("X", False, True, False, "tier_x_measured", True, 0.0, "short")


def test_s36_02_e21_crossover_row_and_report_validation() -> None:
    rep = run_e21_multikrum_vs_zk_crossover(swarm_sizes=(100, 10000))
    assert rep.gate_passed and len(rep.rows) == 2
    row = rep.rows[0]

    with pytest.raises(ValueError, match="swarm_nodes_n must be an integer >= 2"):
        E21CrossoverRow(1, 1.0, 1.0, 2.0, 0.5, 4.0, 50.0, True)
    with pytest.raises(ValueError, match="multikrum_pairwise_ms must be a finite non-negative"):
        E21CrossoverRow(10, -1.0, 1.0, 2.0, 0.5, 4.0, 50.0, True)
    with pytest.raises(ValueError, match="energy_reduction_pct must be a finite number"):
        E21CrossoverRow(10, 1.0, 1.0, 2.0, 0.5, 4.0, 105.0, True)
    with pytest.raises(ValueError, match="zk_dominates must be a boolean"):
        E21CrossoverRow(10, 1.0, 1.0, 2.0, 0.5, 4.0, 50.0, "yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="experiment_id must be a non-empty string"):
        E21CrossoverBenchmarkReport("", "est", 1e-8, 0.5, 512, 100, 99.5, (row,), True)
    with pytest.raises(ValueError, match="measured_flop_ms_coeff must be a positive finite"):
        E21CrossoverBenchmarkReport("E21", "est", 0.0, 0.5, 512, 100, 99.5, (row,), True)
    with pytest.raises(ValueError, match="measured_proof_bytes must be an integer >= 1"):
        E21CrossoverBenchmarkReport("E21", "est", 1e-8, 0.5, 0, 100, 99.5, (row,), True)
    with pytest.raises(ValueError, match="rows must be a non-empty tuple"):
        E21CrossoverBenchmarkReport("E21", "est", 1e-8, 0.5, 512, 100, 99.5, (), True)
    with pytest.raises(ValueError, match="gate_passed must be a boolean"):
        E21CrossoverBenchmarkReport("E21", "est", 1e-8, 0.5, 512, 100, 99.5, (row,), 1)  # type: ignore[arg-type]


def test_s36_04_fuzzy_subswarm_consensus_report_validation() -> None:
    eng = LocalizedFuzzySubSwarmConsensus()
    votes = [SubSwarmVote(f"node-{i}", "dig-A", 0.95, 0.9) for i in range(16)]
    rep = eng.evaluate_subswarm(votes)
    assert rep.quorum_reached and rep.subswarm_size == 16

    with pytest.raises(ValueError, match="subswarm_size must be an integer"):
        FuzzySubSwarmConsensusReport(10, "dig-A", 0.9, True, 1.0, True, 1e-5, 14)
    with pytest.raises(ValueError, match="winning_digest must be a string"):
        FuzzySubSwarmConsensusReport(16, 123, 0.9, True, 1.0, True, 1e-5, 14)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fuzzy_agreement_score must be a finite"):
        FuzzySubSwarmConsensusReport(16, "dig-A", 1.5, True, 1.0, True, 1e-5, 14)
    with pytest.raises(ValueError, match="consensus_latency_ms must be a finite non-negative"):
        FuzzySubSwarmConsensusReport(16, "dig-A", 0.9, True, -1.0, True, 1e-5, 14)
    with pytest.raises(ValueError, match="quorum_reached must be a boolean"):
        FuzzySubSwarmConsensusReport(16, "dig-A", 0.9, "yes", 1.0, True, 1e-5, 14)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="complexity_hops_log2_n must be an integer >= 1"):
        FuzzySubSwarmConsensusReport(16, "dig-A", 0.9, True, 1.0, True, 1e-5, 0)
