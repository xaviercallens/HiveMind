# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and regression tests for QueenBee Strategies and Router Market hardening."""

from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from hivemind.queenbee.strategies import (
    BanditStrategy,
    FederatedShareStrategy,
    Outcome,
    PreferencePair,
    PrivacyAccountant,
    ServedExpertTask,
    clip_and_noise,
    evaluate_served_expert_bandit_routing,
)
from hivemind.router.market import (
    ConsumerBid,
    FeePolicy,
    ProducerOffer,
    certify,
    clear_market,
    split_payment,
)


def test_strategies_outcome_bandit_and_preference_pair_guards() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Outcome("   ", "e1", True, True)

    with pytest.raises(ValueError, match="cost_tokens"):
        Outcome("c1", "e1", True, True, cost_tokens=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Beta prior"):
        BanditStrategy(prior_alpha=float("nan"), prior_beta=1.0)

    with pytest.raises(ValueError, match="Beta prior"):
        BanditStrategy(prior_alpha=1.0, prior_beta=float("inf"))

    with pytest.raises(ValueError, match="non-empty"):
        PreferencePair("   ", "c1", "bad", "good", "unit_tests")

    with pytest.raises(ValueError, match="verifier_kind"):
        PreferencePair("p1", "c1", "bad", "good", "unknown_kind")


def test_strategies_clip_and_noise_privacy_accountant_and_served_expert_guards() -> None:
    g = torch.Generator().manual_seed(7)
    bad_delta = {"w": torch.tensor([1.0, float("nan")])}
    with pytest.raises(ValueError, match="NaN or inf"):
        clip_and_noise(bad_delta, clip_norm=1.0, sigma=0.1, generator=g)

    inf_delta = {"w": torch.tensor([1.0, float("inf")])}
    with pytest.raises(ValueError, match="NaN or inf"):
        clip_and_noise(inf_delta, clip_norm=1.0, sigma=0.1, generator=g)

    with pytest.raises(ValueError, match="RDP orders"):
        PrivacyAccountant(orders=(2.0, float("inf")))

    acc = PrivacyAccountant()
    with pytest.raises(ValueError, match="clip_norm"):
        acc.would_exceed_budget(clip_norm=0.0, sigma=0.5)

    with pytest.raises(ValueError, match="sigma"):
        acc.would_exceed_budget(clip_norm=1.0, sigma=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        FederatedShareStrategy(clip_norm=float("nan"), sigma=0.1)

    t1 = ServedExpertTask(
        task_id="t1",
        cluster_id="c1",
        prompt="p",
        tests=("assert f(1) == 2",),
        expert_candidates={"e1": "def f(x):\n    return x + 1\n"},
    )
    t2_mismatch = ServedExpertTask(
        task_id="t2",
        cluster_id="c1",
        prompt="p",
        tests=("assert f(1) == 2",),
        expert_candidates={"e2": "def f(x):\n    return x + 1\n"},
    )
    with pytest.raises(ValueError, match="seeds"):
        evaluate_served_expert_bandit_routing([t1], seeds=())

    with pytest.raises(ValueError, match="rounds"):
        evaluate_served_expert_bandit_routing([t1], seeds=(1,), rounds=1)

    with pytest.raises(ValueError, match="expert_candidates"):
        evaluate_served_expert_bandit_routing([t1, t2_mismatch], seeds=(1,), rounds=4)


def test_router_market_rejects_nan_forged_equilibria_and_invalid_inputs() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    offers = [ProducerOffer("p", 1.0, 10.0)]
    eq = clear_market("m", bids, offers)
    assert certify(eq, bids, offers).win_win is True

    # Negative control: forging NaN consumer_surplus or NaN producer_profit must NOT pass certification!
    nan_surplus_eq = replace(eq, consumer_surplus={"a": float("nan"), "b": eq.consumer_surplus["b"]})
    cert_nan_surplus = certify(nan_surplus_eq, bids, offers)
    assert cert_nan_surplus.individually_rational_consumers is False
    assert cert_nan_surplus.win_win is False

    nan_profit_eq = replace(eq, producer_profit={"p": float("nan")})
    cert_nan_profit = certify(nan_profit_eq, bids, offers)
    assert cert_nan_profit.individually_rational_producers is False
    assert cert_nan_profit.win_win is False

    nan_price_eq = replace(eq, price_per_1k=float("nan"))
    cert_nan_price = certify(nan_price_eq, bids, offers)
    assert cert_nan_price.win_win is False

    with pytest.raises(ValueError, match="model_id"):
        clear_market("   ", bids, offers)

    with pytest.raises(ValueError, match="measured_load_1k"):
        clear_market("m", bids, offers, measured_load_1k=-1.0)

    with pytest.raises(ValueError, match="measured_load_1k"):
        clear_market("m", bids, offers, measured_load_1k=float("nan"))

    with pytest.raises(ValueError, match="consumer account ids"):
        clear_market("m", [ConsumerBid("   ", 2.0)], offers)

    with pytest.raises(ValueError, match="producer account ids"):
        clear_market("m", bids, [ProducerOffer("   ", 1.0, 5.0)])

    with pytest.raises(ValueError, match="tokens"):
        split_payment(1.0, True, FeePolicy())  # type: ignore[arg-type]
