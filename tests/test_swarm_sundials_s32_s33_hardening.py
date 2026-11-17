# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and negative-control tests for rusty_sundials_scientific_bee, s32_01, s32_02, and s33_02 hardening."""

from __future__ import annotations

import copy
import pytest

from hivemind.swarm.rusty_sundials_scientific_bee import (
    NextGenMoEScienceBee,
    run_wave2_parameter_sweep_quorum,
    verify_wave2_quorum_replicas,
)
from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    AntigravityHiveMindCoPilot,
    SymplecticVelocityVerletIntegrator,
)
from hivemind.swarm.s32_02_lifelong_decentralized_ratchet import (
    AnchorEvalSample,
    DailyCorpusIngestionBatch,
    LifelongDecentralizedRatchet,
)
from hivemind.swarm.s33_02_multibee_amcp_debate import MultiBeeAMCPDebateArena


def test_rusty_sundials_quorum_nan_rejection_and_ask_validation() -> None:
    with pytest.raises(TypeError, match="must be Mappings"):
        verify_wave2_quorum_replicas("not-a-dict", {}, {})  # type: ignore[arg-type]

    quorum = run_wave2_parameter_sweep_quorum(steps=10, dt=0.05)
    assert verify_wave2_quorum_replicas(quorum["spec"], quorum["replica_a"], quorum["replica_b"]) is True

    bad_drift = copy.deepcopy(quorum["replica_a"])
    bad_drift["sweep_results"][0]["invariant_drift"] = float("nan")
    assert verify_wave2_quorum_replicas(quorum["spec"], bad_drift, quorum["replica_b"]) is False

    bad_y = copy.deepcopy(quorum["replica_b"])
    bad_y["sweep_results"][0]["y"][0][0] = float("nan")
    assert verify_wave2_quorum_replicas(quorum["spec"], quorum["replica_a"], bad_y) is False

    bee = NextGenMoEScienceBee()
    with pytest.raises(ValueError, match="question must be a non-empty string"):
        bee.ask("   ")


def test_s32_01_acceleration_and_query_openrouter_validation() -> None:
    integrator = SymplecticVelocityVerletIntegrator()
    with pytest.raises(ValueError, match="x_au and y_au must be finite"):
        integrator.acceleration(float("nan"), 1.0)
    with pytest.raises(ValueError, match="x_au and y_au must be finite"):
        integrator.acceleration(True, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="orbital radius must be positive"):
        integrator.acceleration(0.0, 0.0)
    ax, ay = integrator.acceleration(1.0, 0.0)
    assert ax < 0.0 and ay == 0.0

    copilot = AntigravityHiveMindCoPilot(api_key="hm-test-key-123456")
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        copilot.query_openrouter_direct("")
    with pytest.raises(ValueError, match="max_tokens"):
        copilot.query_openrouter_direct("Hello", max_tokens=0)
    with pytest.raises(ValueError, match="timeout_s"):
        copilot.query_openrouter_direct("Hello", timeout_s=-1.0)


def test_s32_02_ratchet_nan_weight_and_ewc_validation() -> None:
    ratchet = LifelongDecentralizedRatchet()
    with pytest.raises(ValueError, match="samples must be non-empty"):
        ratchet.weighted_perplexity("not-samples", use_candidate=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="each sample must be an AnchorEvalSample"):
        ratchet.weighted_perplexity(["bad"], use_candidate=False)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-finite weight"):
        ratchet.weighted_perplexity(
            [AnchorEvalSample("s1", "astro", 1.0, 1.0, weight=float("nan"))],
            use_candidate=False,
        )
    with pytest.raises(ValueError, match="candidate_params must contain only finite numbers"):
        ratchet.compute_ewc_penalty([1.0, 1.0], [0.5, 0.5], [0.5, float("nan")])
    with pytest.raises(ValueError, match="fisher_diag must contain only finite numbers"):
        ratchet.evaluate_checkpoint(
            "ckpt-1",
            [AnchorEvalSample("s1", "astro", 1.1, 1.1)],
            DailyCorpusIngestionBatch("b1", "arxiv", 10, 1.4, 1.2),
            fisher_diag=[True, 1.0],  # type: ignore[list-item]
            anchor_params=[0.1, 0.2],
            candidate_params=[0.1, 0.2],
        )


def test_s33_02_compute_bayesian_posterior_validation() -> None:
    with pytest.raises(ValueError, match="confidences must be a non-empty sequence"):
        MultiBeeAMCPDebateArena.compute_bayesian_posterior([])
    with pytest.raises(ValueError, match="confidences must be a non-empty sequence"):
        MultiBeeAMCPDebateArena.compute_bayesian_posterior("0.9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidences must contain only numbers"):
        MultiBeeAMCPDebateArena.compute_bayesian_posterior([True, 0.9])  # type: ignore[list-item]
    assert MultiBeeAMCPDebateArena.compute_bayesian_posterior([0.9, 0.9]) > 0.95
