# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and negative-control tests for BTM upcycle and QueenBee input hardening."""

from __future__ import annotations

import hashlib
import random

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import (
    SparseMoEMLPBlock,
    hotplug_rank_adaptive_experts,
    initialize_domain_branches,
)
from hivemind.router.market import FeePolicy, MarketEquilibrium
from hivemind.queenbee.dance import (
    DanceRecord,
    RecruitmentReport,
    dual_scout_bounds,
    rotate_scouts,
)
from hivemind.queenbee.knowledge_market import (
    Admission,
    KnowledgeSettlement,
    KnowledgeShare,
    admit_share,
    certify_knowledge_bonus,
    knowledge_points,
)
from hivemind.queenbee.strategies import (
    BanditStrategy,
    FederatedShareStrategy,
    PrivacyAccountant,
    ServedExpertTask,
    VerifiedIDECorrectionFlow,
    evaluate_served_expert_bandit_routing,
    verify_python_candidate_isolated,
)


def test_btm_upcycle_input_validation() -> None:
    with pytest.raises(ValueError, match="initial_lora_state"):
        initialize_domain_branches({})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expert_domains"):
        initialize_domain_branches({"w": torch.ones(2, 2)}, expert_domains="code")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="torch.Tensor"):
        initialize_domain_branches({"w": "not_a_tensor"})  # type: ignore[dict-item]

    branches = initialize_domain_branches({"w": torch.ones(2, 2)}, expert_domains=["code", "maths"])
    assert set(branches.keys()) == {"code", "maths"}

    with pytest.raises(ValueError, match="experts must be non-empty"):
        SparseMoEMLPBlock(hidden_size=8, experts=[], expert_domains=[])

    with pytest.raises(ValueError, match="expert_domains length must match experts"):
        SparseMoEMLPBlock(hidden_size=8, experts=[nn.Linear(8, 8)], expert_domains=["a", "b"])

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        SparseMoEMLPBlock(hidden_size=8, experts=[nn.Linear(8, 8)], expert_domains=["a"], top_k=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target_rank_r"):
        hotplug_rank_adaptive_experts(None, {}, target_rank_r=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="vram_budget_gb"):
        hotplug_rank_adaptive_experts(None, {}, target_rank_r=16, vram_budget_gb=True)  # type: ignore[arg-type]


def test_queenbee_dance_input_validation() -> None:
    report = RecruitmentReport(weights={"c1": {"e1": 0.8}}, accepted=1, rejected={})
    with pytest.raises(ValueError, match="floor must be finite and >= 0"):
        report.probabilities("c1", ["e1"], floor=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidates must be an iterable of expert ID strings"):
        report.probabilities("c1", "e1")

    with pytest.raises(ValueError, match="pool must be a sequence"):
        rotate_scouts("scout1", epoch=0, k=1)

    with pytest.raises(ValueError, match="secret must be non-empty bytes"):
        rotate_scouts(["s1", "s2"], epoch=0, k=1, secret=b"")

    scouts = rotate_scouts(["s1", "s2", "s3"], epoch=2, k=2)
    assert len(scouts) == 2

    with pytest.raises(ValueError, match="min_independent_scouts must be >= 1"):
        dual_scout_bounds([], ["s1", "s2"], min_independent_scouts=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scout_ids must be an iterable"):
        dual_scout_bounds([], "s1")

    with pytest.raises(ValueError, match="z must be finite and > 0"):
        dual_scout_bounds([], ["s1"], z=True)  # type: ignore[arg-type]


def test_queenbee_knowledge_market_validation() -> None:
    text = "Clean verified engineering solution for distributed LoRA aggregation."
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    receipt = "a" * 64
    share = KnowledgeShare(
        account_id="acc_1",
        kind="verified_code",
        text_sha256=sha,
        licence="CC-BY-4.0",
        verified=True,
        epoch=1,
        verifier_receipt_sha256=receipt,
    )

    with pytest.raises(ValueError, match="share must be a KnowledgeShare"):
        admit_share("not_a_share", text)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="text must be a string"):
        admit_share(share, 12345)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected_epoch must be a non-negative integer"):
        admit_share(share, text, expected_epoch=True)  # type: ignore[arg-type]

    adm = admit_share(share, text, scanner=lambda _: True, expected_epoch=1)
    assert adm.admitted is True

    with pytest.raises(ValueError, match="cap_per_identity_per_epoch must be >= 1"):
        knowledge_points([adm], cap_per_identity_per_epoch=True)  # type: ignore[arg-type]

    fee = FeePolicy()
    eq = MarketEquilibrium(
        model_id="qwen2.5-coder-7b",
        mechanism="kelly",
        price_per_1k=1.0,
        supply_1k=100.0,
        consumer_shares={"c1": 1.0},
        consumer_surplus={"c1": 10.0},
        active_producers=("p1",),
        producer_profit={"p1": 50.0},
        producer_revenue_per_1k=0.85,
        fee=fee,
    )
    settlement = KnowledgeSettlement(
        epoch=1,
        validator_fee_hmt=5.0,
        bonus_share=0.2,
        pool_hmt=1.0,
        payouts_hmt={"acc_1": 1.0},
        points={"acc_1": 1},
    )

    with pytest.raises(ValueError, match="epoch_payment_hmt must be finite and >= 0"):
        certify_knowledge_bonus(eq, eq, settlement, fee, True, True, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="win_win_before and win_win_after must be booleans"):
        certify_knowledge_bonus(eq, eq, settlement, fee, 100.0, 1, True)  # type: ignore[arg-type]


def test_queenbee_strategies_validation() -> None:
    report = RecruitmentReport(weights={"c1": {"e1": 0.8}}, accepted=1, rejected={})
    b = BanditStrategy()
    with pytest.raises(ValueError, match="report must be a RecruitmentReport"):
        b.warm_start("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pseudo_trials must be finite and > 0"):
        b.warm_start(report, pseudo_trials=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="use_trial_budget must be a boolean"):
        b.warm_start(report, pseudo_trials=10.0, use_trial_budget=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidates must be non-empty"):
        b.choose("c1", "e1", random.Random(0))

    with pytest.raises(ValueError, match="delta must be in"):
        PrivacyAccountant(delta=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_epsilon must be finite and > 0"):
        PrivacyAccountant(max_epsilon=True)  # type: ignore[arg-type]

    acc = PrivacyAccountant()
    with pytest.raises(ValueError, match="clip_norm must be finite and > 0"):
        acc.would_exceed_budget(True, 1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="enforce_budget must be a boolean"):
        acc.record_step(1.0, 2.0, enforce_budget=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clip_norm"):
        FederatedShareStrategy(clip_norm=True, sigma=1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="opt_in must be a boolean"):
        VerifiedIDECorrectionFlow(opt_in=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="beta must be finite and > 0"):
        VerifiedIDECorrectionFlow(beta=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="code must be a string"):
        verify_python_candidate_isolated(123, ["assert True"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tests must be a sequence of strings"):
        verify_python_candidate_isolated("def f(): return 1", "assert f() == 1")

    with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
        verify_python_candidate_isolated("def f(): return 1", ["assert f() == 1"], timeout_s=True)  # type: ignore[arg-type]

    task = ServedExpertTask(
        task_id="t1",
        cluster_id="c1",
        prompt="p",
        tests=("assert f() == 1",),
        expert_candidates={"e1": "def f(): return 1"},
    )
    with pytest.raises(ValueError, match="rounds must be >= 2"):
        evaluate_served_expert_bandit_routing([task], rounds=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_trials_per_signer must be finite and > 0"):
        evaluate_served_expert_bandit_routing([task], max_trials_per_signer=True)  # type: ignore[arg-type]
