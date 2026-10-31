# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S33-02 Multi-Bee Debate & Cross-Verification Arena over AMCP Bus."""

from __future__ import annotations

from hivemind.swarm.s33_02_multibee_amcp_debate import (
    BeeDebateTurn,
    MultiBeeAMCPDebateArena,
)


def test_multibee_debate_accepts_when_all_three_bees_verify_invariants() -> None:
    arena = MultiBeeAMCPDebateArena(residual_ceiling=1e-4, min_posterior=0.85)
    turns = [
        BeeDebateTurn("hivemind/bee-stars", "hypothesis", True, 0.92, 1.2e-6, "Transit depth 2789 ppm matches K2-18b"),
        BeeDebateTurn("hivemind/bee-rusty-sundials", "numeric_solver", True, 0.96, 4.5e-7, "Symplectic BDF drift < 1e-6"),
        BeeDebateTurn("hivemind/bee-lean-prover", "formal_prover", True, 0.98, 0.0, "Lean 4 conservation theorem verified"),
    ]
    verdict = arena.adjudicate_debate("kepler-442b-orbital-resonance", turns)
    assert verdict.accepted is True
    assert verdict.vetoed_by == ()
    assert verdict.bayesian_posterior_confidence > 0.99


def test_multibee_debate_fail_closed_veto_when_solver_detects_energy_drift() -> None:
    arena = MultiBeeAMCPDebateArena(residual_ceiling=1e-4, min_posterior=0.85)
    turns = [
        BeeDebateTurn("hivemind/bee-stars", "hypothesis", True, 0.95, 1e-5, "Plausible narrative"),
        BeeDebateTurn("hivemind/bee-rusty-sundials", "numeric_solver", False, 0.10, 2.5e-2, "Energy drift 2.5% > 1e-4"),
        BeeDebateTurn("hivemind/bee-lean-prover", "formal_prover", True, 0.90, 0.0, "Syntax valid"),
    ]
    verdict = arena.adjudicate_debate("unstable-orbit-proposal", turns)
    assert verdict.accepted is False
    assert "hivemind/bee-rusty-sundials" in verdict.vetoed_by
