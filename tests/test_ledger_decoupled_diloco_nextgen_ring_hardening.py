# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 66: Negative-control and boundary tests for ledger, decoupled DiLoCo, next-gen DiLoCo, and inference ring."""

from __future__ import annotations

import pytest
import torch

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm.decoupled_diloco import (
    AdaptiveGraceWindowController,
    ContinuousAsyncLearnerCoordinator,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    compute_cosine_damped_outer_hyperparams,
    select_quorum,
    token_weighted_quality_weight,
)
from hivemind.swarm.diloco_nextgen import (
    AutoLoCoIntervalCorrector,
    CosineGatedAdamDecayOuterOptimizer,
    FoMoEPartialReplicaPlanner,
    HeLoCoDirectionCorrector,
    ReputationWeightedIsoCTSMerger,
)
from hivemind.swarm.inference_ring import (
    RegionalDHTSwarmRing,
    prefix_trie_branch_probabilities,
)
from hivemind.swarm.ledger import GuildSlashingInsurancePool, HiveSwarmLedger


def test_ledger_slash_and_reward_and_settlement_validation() -> None:
    ledger = HiveSwarmLedger()
    ledger.credit_reward("node-a", 10.0)
    assert ledger.get_balance("node-a") == pytest.approx(10.0)

    # Boolean and non-finite reward rejections
    with pytest.raises(SwarmSecurityViolationError):
        ledger.credit_reward("node-a", True)  # type: ignore[arg-type]
    with pytest.raises(SwarmSecurityViolationError):
        ledger.credit_reward("node-a", False)  # type: ignore[arg-type]

    # Slash fraction validation (reject NaN, Inf, bool, negative, > 1.0)
    for bad_frac in (True, False, -0.1, 1.5, float("nan"), float("inf"), "0.5"):
        with pytest.raises(ValueError, match="fraction"):
            ledger.slash("node-a", fraction=bad_frac)  # type: ignore[arg-type]

    # Settlement input validation
    with pytest.raises(SwarmSecurityViolationError):
        ledger.settle_swarm_inference("   ", ["prov-1"], "math", 10, 2.0)
    with pytest.raises(SwarmSecurityViolationError):
        ledger.settle_swarm_inference("node-a", ["prov-1"], "math", True, 2.0)  # type: ignore[arg-type]
    with pytest.raises(SwarmSecurityViolationError):
        ledger.settle_swarm_inference("node-a", ["prov-1"], "math", 10.5, 2.0)  # type: ignore[arg-type]
    with pytest.raises(SwarmSecurityViolationError):
        ledger.settle_swarm_inference("node-a", ["prov-1"], "math", 10, 2.0, rate_per_token=True)  # type: ignore[arg-type]
    with pytest.raises(SwarmSecurityViolationError):
        ledger.settle_swarm_inference("node-a", ["   "], "math", 10, 2.0)

    receipt = ledger.settle_swarm_inference("node-a", ["prov-1"], "math", 10, 2.0, rate_per_token=0.01)
    assert receipt.fee_paid_hive == pytest.approx(0.1)

    # Insurance pool boolean amount rejection
    with pytest.raises(SwarmSecurityViolationError):
        GuildSlashingInsurancePool(ledger, max_claim_hive=True)  # type: ignore[arg-type]
    pool = GuildSlashingInsurancePool(ledger, max_claim_hive=100.0)
    with pytest.raises(SwarmSecurityViolationError):
        pool.bond_stake("guild-1", "node-a", True)  # type: ignore[arg-type]


def test_decoupled_diloco_boundary_validation() -> None:
    with pytest.raises(ValueError, match="learner_id"):
        LearnerArrival(learner_id="   ", arrival_time_s=1.0, steps_since_sync=1, tokens_since_sync=16)
    with pytest.raises(ValueError, match="arrival_time_s"):
        LearnerArrival(learner_id="l1", arrival_time_s=True, steps_since_sync=1, tokens_since_sync=16)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps_since_sync"):
        LearnerArrival(learner_id="l1", arrival_time_s=1.0, steps_since_sync=True, tokens_since_sync=16)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps_since_sync"):
        LearnerArrival(learner_id="l1", arrival_time_s=1.0, steps_since_sync=1, tokens_since_sync=16.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tokens_since_sync"):
        token_weighted_quality_weight(True, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_total_tokens"):
        token_weighted_quality_weight(16, 1, max_total_tokens=0.0)
    with pytest.raises(ValueError, match="max_total_tokens"):
        token_weighted_quality_weight(16, 1, max_tokens_per_step=float("nan"))

    a1 = LearnerArrival("l1", 1.0, 2, 32)
    with pytest.raises(ValueError, match="min_quorum_k"):
        select_quorum([a1], True, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grace_window_s"):
        select_quorum([a1], 1, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="overlap_steps_tau"):
        AdaptiveGraceWindowController(overlap_steps_tau=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="safety_margin_gamma"):
        AdaptiveGraceWindowController(safety_margin_gamma=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ema_beta"):
        AdaptiveGraceWindowController(ema_beta=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sync_duration_s"):
        AdaptiveGraceWindowController(sync_duration_s=False)  # type: ignore[arg-type]

    ctrl = AdaptiveGraceWindowController()
    with pytest.raises(ValueError, match="durations"):
        ctrl.observe_step_duration(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_learners_m"):
        DecoupledDiLoCoQuorumSyncer(num_learners_m=True, min_quorum_k=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_quorum_k"):
        DecoupledDiLoCoQuorumSyncer(num_learners_m=2, min_quorum_k=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="staleness_decay"):
        ContinuousAsyncLearnerCoordinator(num_learners_m=2, min_quorum_k=1, staleness_decay=True)  # type: ignore[arg-type]

    d1 = torch.tensor([1.0, 0.0])
    d2 = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="learning rates"):
        compute_cosine_damped_outer_hyperparams([d1, d2], base_lr=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_momentum"):
        compute_cosine_damped_outer_hyperparams([d1, d2], base_momentum=False)  # type: ignore[arg-type]


def test_diloco_nextgen_boundary_validation() -> None:
    with pytest.raises(ValueError, match="lr"):
        CosineGatedAdamDecayOuterOptimizer(lr=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lr"):
        CosineGatedAdamDecayOuterOptimizer(lr=float("nan"))
    with pytest.raises(ValueError, match="alpha"):
        CosineGatedAdamDecayOuterOptimizer(alpha=float("inf"))
    with pytest.raises(ValueError, match="tau_cut"):
        CosineGatedAdamDecayOuterOptimizer(tau_cut=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eps"):
        CosineGatedAdamDecayOuterOptimizer(eps=0.0)
    with pytest.raises(ValueError, match="betas"):
        CosineGatedAdamDecayOuterOptimizer(betas=(True, 0.95))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="momentum_restart_period"):
        CosineGatedAdamDecayOuterOptimizer(momentum_restart_period=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="common_space_fraction"):
        ReputationWeightedIsoCTSMerger(common_space_fraction=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trim_density"):
        ReputationWeightedIsoCTSMerger(trim_density=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scaling_lambda"):
        ReputationWeightedIsoCTSMerger(scaling_lambda=0.0)

    with pytest.raises(ValueError, match="conflict_damping"):
        HeLoCoDirectionCorrector(conflict_damping=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="reference_inner_steps_h"):
        AutoLoCoIntervalCorrector(reference_inner_steps_h=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_outer_lr"):
        AutoLoCoIntervalCorrector(base_outer_lr=0.0)
    with pytest.raises(ValueError, match="base_outer_momentum"):
        AutoLoCoIntervalCorrector(base_outer_momentum=1.0)
    corrector = AutoLoCoIntervalCorrector()
    with pytest.raises(ValueError, match="actual_inner_steps_h"):
        corrector.correct_outer_hyperparams(True, 1e-3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="total_experts"):
        FoMoEPartialReplicaPlanner(total_experts=4, active_top_k=5)
    planner = FoMoEPartialReplicaPlanner()
    with pytest.raises(ValueError, match="num_workers_m"):
        planner.evaluate_overlap_configuration(True, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reshuffle_frequency_gamma"):
        planner.evaluate_overlap_configuration(4, 2, reshuffle_frequency_gamma=True)  # type: ignore[arg-type]


def test_inference_ring_boundary_validation() -> None:
    with pytest.raises(ValueError, match="draft step probabilities"):
        prefix_trie_branch_probabilities([[1, 2]], [[0.8, -0.2]])
    with pytest.raises(ValueError, match="draft step probabilities"):
        prefix_trie_branch_probabilities([[1, 2]], [[True, 0.5]])  # type: ignore[list-item]

    nodes, probs = prefix_trie_branch_probabilities([[1, 2]], [[0.8, 0.5]])
    assert len(nodes) == 3
    assert sum(probs) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="region"):
        RegionalDHTSwarmRing(region="   ")
    with pytest.raises(ValueError, match="max_ping_ms"):
        RegionalDHTSwarmRing(max_ping_ms=0.0)
    with pytest.raises(ValueError, match="max_ping_ms"):
        RegionalDHTSwarmRing(max_ping_ms=True)  # type: ignore[arg-type]
