"""Tests for `hivemind.router.market`: the Kelly consumer equilibrium, producer dominant-strategy supply, the reserve
fallback, the win-win certificate (with negative controls that must make it fail) and the fee split."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.router import (
    ConsumerBid,
    FeePolicy,
    ProducerOffer,
    certify,
    clear_market,
    split_payment,
)

KEEP = FeePolicy().producer_share  # 0.85

log_uniform = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False).map(lambda e: 10.0**e)
capacity_st = st.floats(min_value=-1.0, max_value=3.0, allow_nan=False).map(lambda e: 10.0**e)


@st.composite
def instances(draw: st.DrawFn) -> tuple[list[ConsumerBid], list[ProducerOffer]]:
    values = draw(st.lists(log_uniform, min_size=2, max_size=10))
    producers = draw(st.lists(st.tuples(log_uniform, capacity_st), min_size=1, max_size=8))
    bids = [ConsumerBid(f"c{j}", v) for j, v in enumerate(values)]
    offers = [ProducerOffer(f"p{i}", c, k) for i, (c, k) in enumerate(producers)]
    return bids, offers


def _omega_closed_form(values: list[float]) -> float:
    """Independent solution of sum_j max(0, 1 - w/v_j) = 1: for the top-m values, w = (m-1) / sum(1/v)."""
    ordered = sorted(values, reverse=True)
    for m in range(len(ordered), 1, -1):
        w = (m - 1) / sum(1.0 / v for v in ordered[:m])
        if w < ordered[m - 1]:
            return w
    raise AssertionError("no active set found")


# ---------------------------------------------------------------- closed forms


def test_two_equal_consumers_give_omega_two_and_a_certified_win_win() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    offers = [ProducerOffer("p", 1.0, 10.0)]
    eq = clear_market("m", bids, offers)
    assert eq.mechanism == "kelly"
    assert eq.price_per_1k == pytest.approx(2.0, abs=1e-9)
    assert eq.consumer_shares["a"] == pytest.approx(0.5) and eq.consumer_shares["b"] == pytest.approx(0.5)
    assert eq.supply_1k == 10.0 and eq.active_producers == ("p",)
    assert eq.consumer_surplus["a"] == pytest.approx((4.0 - 2.0) * 0.5 * 10.0, abs=1e-8)
    assert eq.producer_profit["p"] == pytest.approx((KEEP * 2.0 - 1.0) * 10.0, abs=1e-8)
    cert = certify(eq, bids, offers)
    assert cert.win_win and cert.consumer_nash is True and cert.producer_nash and cert.conserved
    assert cert.max_deviation_gain <= 1e-8


def test_unequal_consumers_closed_form_values_two_and_one() -> None:
    eq = clear_market("m", [ConsumerBid("hi", 2.0), ConsumerBid("lo", 1.0)], [ProducerOffer("p", 0.1, 3.0)])
    assert eq.price_per_1k == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert eq.consumer_shares["hi"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert eq.consumer_shares["lo"] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_low_value_consumer_is_priced_out_of_the_kelly_game() -> None:
    bids = [ConsumerBid("a", 10.0), ConsumerBid("b", 10.0), ConsumerBid("tiny", 0.2)]
    eq = clear_market("m", bids, [ProducerOffer("p", 0.5, 5.0)])
    assert eq.mechanism == "kelly"
    assert eq.consumer_shares["tiny"] == 0.0 and eq.consumer_surplus["tiny"] == 0.0
    assert sum(eq.consumer_shares.values()) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- dominant strategy


def test_price_does_not_depend_on_supply_and_profit_is_linear_in_supply() -> None:
    bids = [ConsumerBid("a", 3.0), ConsumerBid("b", 5.0), ConsumerBid("c", 1.5)]
    prices, profits = [], []
    for k in (1.0, 10.0, 100.0, 1000.0):
        eq = clear_market("m", bids, [ProducerOffer("p", 0.4, k), ProducerOffer("q", 0.6, 7.0)])
        prices.append(eq.price_per_1k)
        profits.append(eq.producer_profit["p"] / k)
    assert max(prices) - min(prices) < 1e-12
    assert max(profits) - min(profits) < 1e-9 and profits[0] > 0.0


def test_producer_above_break_even_is_inactive_and_below_is_active() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    eq = clear_market("m", bids, [ProducerOffer("cheap", 1.0, 5.0), ProducerOffer("dear", KEEP * 2.0 + 0.01, 5.0)])
    assert eq.mechanism == "kelly" and eq.active_producers == ("cheap",)
    assert eq.producer_profit["dear"] == 0.0 and eq.supply_1k == 5.0
    assert certify(eq, bids, [ProducerOffer("cheap", 1.0, 5.0), ProducerOffer("dear", KEEP * 2.0 + 0.01, 5.0)]).win_win


def test_indifferent_producer_at_break_even_does_not_supply_under_kelly() -> None:
    fee = FeePolicy(validator_share=0.0, burn_share=0.0)
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    eq = clear_market("m", bids, [ProducerOffer("p", 2.0, 4.0)], fee)  # c == (1-f) omega exactly
    assert eq.mechanism == "posted_price"  # Kelly leaves nobody willing, so the reserve fallback opens the market
    assert eq.price_per_1k == pytest.approx(2.0 * 1.05)
    assert eq.producer_profit["p"] == pytest.approx((2.1 - 2.0) * 4.0)


# ---------------------------------------------------------------- reserve fallback / no trade


def test_single_consumer_gets_posted_price_at_reserve_plus_margin() -> None:
    bids = [ConsumerBid("solo", 5.0)]
    offers = [ProducerOffer("p", 1.0, 10.0)]
    eq = clear_market("m", bids, offers)
    posted = (1.0 / KEEP) * 1.05
    assert eq.mechanism == "posted_price"
    assert eq.price_per_1k == pytest.approx(posted)
    assert eq.consumer_shares["solo"] == 1.0 and eq.supply_1k == 10.0
    assert eq.consumer_surplus["solo"] == pytest.approx((5.0 - posted) * 10.0)
    cert = certify(eq, bids, offers)
    assert cert.win_win and cert.consumer_nash is None


def test_posted_price_splits_capacity_equally_among_served_consumers() -> None:
    unaffordable = clear_market("m", [ConsumerBid("a", 9.0), ConsumerBid("b", 9.0)], [ProducerOffer("p", 20.0, 6.0)])
    assert unaffordable.mechanism == "no_trade" and unaffordable.supply_1k == 0.0
    bids2 = [ConsumerBid("a", 30.0), ConsumerBid("b", 30.0), ConsumerBid("c", 0.1)]
    eq2 = clear_market("m", bids2, [ProducerOffer("p", 20.0, 6.0)])
    assert eq2.mechanism == "posted_price"
    assert eq2.consumer_shares["a"] == eq2.consumer_shares["b"] == 0.5 and eq2.consumer_shares["c"] == 0.0


def test_no_trade_when_every_value_is_below_reserve_with_margin() -> None:
    bids = [ConsumerBid("a", 1.0), ConsumerBid("b", 1.0)]
    offers = [ProducerOffer("p", 2.0, 10.0)]
    eq = clear_market("m", bids, offers)
    assert eq.mechanism == "no_trade" and eq.supply_1k == 0.0 and eq.active_producers == ()
    assert all(x == 0.0 for x in eq.consumer_shares.values())
    cert = certify(eq, bids, offers)
    assert cert.win_win and cert.conserved and cert.consumer_nash is None


def test_value_between_reserve_and_reserve_with_margin_is_not_served() -> None:
    reserve = 1.0 / KEEP
    bids = [ConsumerBid("solo", reserve * 1.02)]
    eq = clear_market("m", bids, [ProducerOffer("p", 1.0, 3.0)])
    assert eq.mechanism == "no_trade"
    served = clear_market("m", [ConsumerBid("solo", reserve * 1.06)], [ProducerOffer("p", 1.0, 3.0)])
    assert served.mechanism == "posted_price"


def test_no_offers_or_zero_capacity_means_no_trade() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    assert clear_market("m", bids, []).mechanism == "no_trade"
    zero = clear_market("m", bids, [ProducerOffer("p", 1.0, 0.0)])
    assert zero.mechanism == "no_trade" and zero.producer_profit == {"p": 0.0}


# ---------------------------------------------------------------- certificate negative controls


def test_certificate_rejects_a_tampered_price() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0), ConsumerBid("c", 2.0)]
    offers = [ProducerOffer("p", 0.5, 8.0)]
    eq = clear_market("m", bids, offers)
    assert certify(eq, bids, offers).win_win
    bad = replace(eq, price_per_1k=eq.price_per_1k * 1.5)
    cert = certify(bad, bids, offers)
    assert not cert.win_win and cert.consumer_nash is False and cert.max_deviation_gain > 1e-6


def test_certificate_rejects_loss_making_active_producer_and_leaked_fees() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    offers = [ProducerOffer("p", 1.0, 5.0), ProducerOffer("dear", 3.9, 5.0)]
    eq = clear_market("m", bids, offers)
    forced = replace(eq, active_producers=("p", "dear"), supply_1k=10.0)
    assert not certify(forced, bids, offers).individually_rational_producers
    leaked = replace(eq, producer_revenue_per_1k=eq.price_per_1k)  # producers paid the burn share too
    cert = certify(leaked, bids, offers)
    assert not cert.conserved and not cert.win_win


def test_certificate_rejects_negative_consumer_surplus() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    offers = [ProducerOffer("p", 1.0, 5.0)]
    eq = clear_market("m", bids, offers)
    bad = replace(eq, consumer_surplus={"a": -1.0, "b": eq.consumer_surplus["b"]})
    assert not certify(bad, bids, offers).individually_rational_consumers
    assert not certify(replace(eq, mechanism="auction"), bids, offers).win_win


# ---------------------------------------------------------------- properties


@settings(max_examples=40, deadline=None)
@given(instances())
def test_certificate_is_win_win_for_random_instances(inst: tuple[list[ConsumerBid], list[ProducerOffer]]) -> None:
    bids, offers = inst
    eq = clear_market("m", bids, offers)
    cert = certify(eq, bids, offers)
    assert cert.win_win, (eq.mechanism, cert)
    assert cert.max_deviation_gain <= 1e-6
    assert eq.mechanism in ("kelly", "posted_price", "no_trade")


def test_random_sweep_exercises_all_three_mechanisms() -> None:
    rng = random.Random(20261024)
    seen: dict[str, int] = {}
    for _ in range(250):
        n, m = rng.randint(2, 10), rng.randint(1, 8)
        bids = [ConsumerBid(f"c{j}", 10 ** rng.uniform(-2, 2)) for j in range(n)]
        offers = [ProducerOffer(f"p{i}", 10 ** rng.uniform(-2, 2), 10 ** rng.uniform(-1, 3)) for i in range(m)]
        eq = clear_market("m", bids, offers)
        assert certify(eq, bids, offers).win_win
        seen[eq.mechanism] = seen.get(eq.mechanism, 0) + 1
    assert set(seen) == {"kelly", "posted_price", "no_trade"}, seen
    assert seen["kelly"] >= 20


@settings(max_examples=60, deadline=None)
@given(st.lists(log_uniform, min_size=2, max_size=10), capacity_st, st.floats(min_value=0.5, max_value=50.0))
def test_kelly_price_matches_independent_closed_form_and_ignores_capacity(values: list[float], k: float, scale: float) -> None:
    omega = _omega_closed_form(values)
    bids = [ConsumerBid(f"c{j}", v) for j, v in enumerate(values)]
    cheap = 0.5 * KEEP * omega  # always strictly profitable, so the Kelly mechanism is used
    eq1 = clear_market("m", bids, [ProducerOffer("p", cheap, k)])
    eq2 = clear_market("m", bids, [ProducerOffer("p", cheap, k * scale)])
    assert eq1.mechanism == "kelly" and eq2.mechanism == "kelly"
    assert eq1.price_per_1k == pytest.approx(omega, rel=1e-9)
    assert eq1.price_per_1k == eq2.price_per_1k
    assert sum(eq1.consumer_shares.values()) == pytest.approx(1.0, abs=1e-12)


@settings(max_examples=80, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
    st.integers(min_value=0, max_value=10**7),
    st.floats(min_value=0.0, max_value=0.5),
    st.floats(min_value=0.0, max_value=0.45),
)
def test_split_payment_conserves_and_is_nonnegative(price: float, tokens: int, v_share: float, b_share: float) -> None:
    fee = FeePolicy(validator_share=v_share, burn_share=b_share)
    charge, producer, validators, burn = split_payment(price, tokens, fee)
    assert charge == pytest.approx(price * tokens / 1000.0)
    assert abs(charge - (producer + validators + burn)) <= 1e-9 * max(1.0, charge)
    assert producer >= -1e-9 * max(1.0, charge) and validators >= 0.0 and burn >= 0.0
    assert validators == pytest.approx(charge * v_share) and burn == pytest.approx(charge * b_share)


def test_split_payment_default_fee_gives_85_10_5() -> None:
    charge, producer, validators, burn = split_payment(2.0, 1000, FeePolicy())
    assert (charge, validators, burn) == (2.0, pytest.approx(0.2), pytest.approx(0.1))
    assert producer == pytest.approx(1.7)


# ---------------------------------------------------------------- validation and serialisation


def test_invalid_inputs_are_rejected() -> None:
    ok_bids = [ConsumerBid("a", 1.0), ConsumerBid("b", 1.0)]
    ok_offers = [ProducerOffer("p", 1.0, 1.0)]
    for bad_bids in ([ConsumerBid("a", 1.0), ConsumerBid("a", 2.0)], [ConsumerBid("a", math.nan)], [ConsumerBid("a", -1.0)]):
        with pytest.raises(ValueError):
            clear_market("m", bad_bids, ok_offers)
    for bad_offers in (
        [ProducerOffer("p", 1.0, 1.0), ProducerOffer("p", 1.0, 1.0)],
        [ProducerOffer("p", -0.1, 1.0)],
        [ProducerOffer("p", 1.0, math.inf)],
    ):
        with pytest.raises(ValueError):
            clear_market("m", ok_bids, bad_offers)


def test_fee_policy_validation_and_split_arguments() -> None:
    for kwargs in ({"validator_share": -0.1}, {"burn_share": math.nan}, {"validator_share": 0.6, "burn_share": 0.4}):
        with pytest.raises(ValueError):
            FeePolicy(**kwargs)
    assert FeePolicy().fee_rate == pytest.approx(0.15) and FeePolicy().producer_share == pytest.approx(0.85)
    with pytest.raises(ValueError):
        split_payment(-1.0, 10, FeePolicy())
    with pytest.raises(ValueError):
        split_payment(1.0, -10, FeePolicy())


def test_to_dict_is_json_serialisable_and_round_trips_key_fields() -> None:
    bids = [ConsumerBid("a", 4.0), ConsumerBid("b", 4.0)]
    offers = [ProducerOffer("p", 1.0, 10.0)]
    eq = clear_market("m", bids, offers)
    payload = json.loads(json.dumps(eq.to_dict()))
    assert payload["mechanism"] == "kelly" and payload["active_producers"] == ["p"]
    assert payload["fee"] == {"validator_share": 0.10, "burn_share": 0.05, "posted_margin": 0.05}
    cert = json.loads(json.dumps(certify(eq, bids, offers).to_dict()))
    assert cert["win_win"] is True and cert["consumer_nash"] is True
