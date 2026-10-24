"""Tests for `hivemind.economy.compute_coin`: BOINC credit rules, cap-aware issuance, and the closed-form Nash equilibria
(Kelly consumers, Tullock workers, multi-queue routing) with their certificates. Pure functions get Hypothesis properties."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

import hivemind.economy as e

pos = st.floats(min_value=0.05, max_value=50.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------- credit

def test_cobblestone_is_432_gflop_and_rejects_bad_input() -> None:
    assert e.COBBLESTONE_FLOP == 432e9
    assert e.cobblestones_from_flops(432e9) == 1.0
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            e.cobblestones_from_flops(bad)


def test_credit_is_zero_unless_verified_and_capped_by_class_peak_and_canonical_replica() -> None:
    ok = e.CreditClaim("n1", "wu", 864e9, 100.0, 10.0, True, True)  # 2 cobblestones claimed; peak allows 1000 GFLOP = 2.31
    assert e.verified_credit(ok).cobblestones == pytest.approx(2.0) and e.verified_credit(ok).capped_by == "verified"
    failed = e.CreditClaim("n1", "wu", 864e9, 100.0, 10.0, False, True)
    assert e.verified_credit(failed).cobblestones == 0.0 and e.verified_credit(failed).capped_by == "rejected"
    not_admitted = e.CreditClaim("n1", "wu", 864e9, 100.0, 10.0, True, False)
    assert e.verified_credit(not_admitted).cobblestones == 0.0
    over = e.CreditClaim("n1", "wu", 5000e9, 100.0, 10.0, True, True)  # claims 11.6 but the class can only do 2.31
    g = e.verified_credit(over)
    assert g.capped_by == "class_peak" and g.cobblestones == pytest.approx(1000e9 / 432e9)
    replica = e.CreditClaim("n2", "wu", 432e9, 100.0, 10.0, True, True)  # verified replica claims 1.0
    g2 = e.verified_credit(ok, [replica])
    assert g2.capped_by == "canonical_replica" and g2.cobblestones == pytest.approx(1.0)
    unverified_replica = e.CreditClaim("n3", "wu", 1e9, 100.0, 10.0, False, True)  # a failed replica cannot drag the canonical down
    assert e.verified_credit(ok, [unverified_replica]).cobblestones == pytest.approx(2.0)


@settings(max_examples=100, deadline=None)
@given(st.floats(min_value=0.0, max_value=1e15, allow_nan=False), st.floats(min_value=0.1, max_value=1e4), st.floats(min_value=0.1, max_value=200.0))
def test_credit_never_exceeds_claim_or_class_peak(flops: float, wall: float, peak_gflops: float) -> None:
    g = e.verified_credit(e.CreditClaim("n", "wu", flops, wall, peak_gflops, True, True))
    assert g.cobblestones <= e.cobblestones_from_flops(flops) + 1e-9
    assert g.cobblestones <= peak_gflops * 1e9 * wall / e.COBBLESTONE_FLOP + 1e-9
    assert g.cobblestones >= 0.0


# ---------------------------------------------------------------- issuance

def test_settlement_mints_per_credit_respects_cap_and_splits_fees() -> None:
    grants = [e.CreditGrant("a", "wu1", 3.0, "verified", ""), e.CreditGrant("b", "wu2", 1.0, "verified", ""), e.CreditGrant("z", "wu3", 0.0, "rejected", "")]
    s = e.settle_epoch(0, grants, consumer_payment_hive=100.0, circulating_supply_hive=0.0)
    assert s.minted_hive == pytest.approx(4.0 * 0.01) and s.cap_binding is False
    assert s.minted_by_node["a"] == pytest.approx(0.03) and "z" not in s.minted_by_node
    assert s.fees_to_workers["a"] == pytest.approx(60.0) and s.fees_to_workers["b"] == pytest.approx(20.0)
    assert s.fees_to_validators == pytest.approx(15.0) and s.fees_burned == pytest.approx(5.0)
    near_cap = e.settle_epoch(0, grants, 0.0, circulating_supply_hive=21_000_000.0 - 0.01)
    assert near_cap.cap_binding is True and near_cap.minted_hive == pytest.approx(0.01)
    assert sum(near_cap.minted_by_node.values()) == pytest.approx(0.01)
    later = e.EmissionSchedule().emission_per_cobblestone(1000)
    assert later < 0.01 and later == pytest.approx(0.01 * 0.999 ** 1000)
    with pytest.raises(ValueError):
        e.EmissionSchedule(worker_share=0.5, validator_share=0.5, burn_share=0.5)


def test_apply_settlement_pushes_balances_through_the_ledger_callable() -> None:
    balances: dict[str, float] = {}

    def credit(node: str, amount: float) -> float:
        balances[node] = balances.get(node, 0.0) + amount
        return balances[node]

    s = e.settle_epoch(1, [e.CreditGrant("a", "wu", 2.0, "verified", "")], 10.0, 0.0)
    out = e.apply_settlement(s, credit)
    assert out["a"] == pytest.approx(2.0 * 0.01 * 0.999 + 8.0) and out["validators"] == pytest.approx(1.5)
    assert sum(balances.values()) == pytest.approx(s.minted_hive + 10.0 - s.fees_burned)  # the burn never reaches a balance


# ---------------------------------------------------------------- consumers: Kelly

@settings(max_examples=200, deadline=None)
@given(st.lists(pos, min_size=2, max_size=12))
def test_kelly_equilibrium_is_feasible_and_captures_at_least_three_quarters_of_optimal_welfare(vals: list[float]) -> None:
    k = e.kelly_equilibrium(vals)
    assert sum(k.shares) == pytest.approx(1.0, abs=1e-9) and all(x >= 0.0 for x in k.shares)
    assert k.efficiency >= 0.75 - 1e-9  # Johari & Tsitsiklis: price of anarchy <= 4/3
    assert k.efficiency <= 1.0 + 1e-9
    # higher value never gets a smaller share; payments equal W x_j
    order = sorted(range(len(vals)), key=lambda j: vals[j])
    for a, b in zip(order, order[1:]):
        assert k.shares[a] <= k.shares[b] + 1e-9
    assert sum(k.payments) == pytest.approx(k.total_payment, rel=1e-9)


def test_kelly_two_equal_consumers_split_evenly_and_a_lone_consumer_is_not_a_game() -> None:
    k = e.kelly_equilibrium([4.0, 4.0])
    assert k.shares == pytest.approx((0.5, 0.5)) and k.total_payment == pytest.approx(2.0) and k.efficiency == pytest.approx(1.0)
    with pytest.raises(ValueError):
        e.kelly_equilibrium([4.0])
    # the price-of-anarchy worst case is approached by one consumer of value 1 against many of value 1/2 (W -> 1/2, welfare -> 3/4)
    worst = e.kelly_equilibrium([1.0] + [0.5] * 50)
    assert 0.75 <= worst.efficiency < 0.77
    tiny = e.kelly_equilibrium([1.0] + [1e-3] * 11)  # tiny rivals are priced out: the high-value consumer gets almost everything
    assert tiny.efficiency > 0.99 and tiny.shares[0] > 0.99


# ---------------------------------------------------------------- workers: Tullock

@settings(max_examples=150, deadline=None)
@given(st.floats(min_value=1.0, max_value=1000.0), st.lists(pos, min_size=2, max_size=8))
def test_tullock_closed_form_is_a_nash_equilibrium_with_positive_payoffs(pool: float, costs: list[float]) -> None:
    t = e.tullock_equilibrium(pool, costs)
    ok, worst = e.is_epsilon_nash(t, eps=1e-6)
    assert ok, worst
    assert all(p >= -1e-9 for p in t.payoffs)  # nobody active loses money at equilibrium
    assert t.total_supply == pytest.approx(sum(t.supplies))
    assert len(t.active) >= 2 and t.price_per_cobblestone > min(costs)  # the pool pays more than the cheapest cost: rents exist


def test_tullock_symmetric_split_and_expensive_worker_drops_out() -> None:
    t = e.tullock_equilibrium(100.0, [1.0, 1.0, 1.0])
    assert t.supplies == pytest.approx((200 / 9,) * 3) and t.price_per_cobblestone == pytest.approx(1.5)
    t2 = e.tullock_equilibrium(100.0, [1.0, 2.0, 10.0])
    assert t2.active == (0, 1) and t2.supplies[2] == 0.0 and t2.supplies[0] > t2.supplies[1] > 0.0
    with pytest.raises(ValueError):
        e.tullock_equilibrium(100.0, [1.0])


# ---------------------------------------------------------------- routing across queues

def test_routing_equalises_prices_matches_tullock_and_certifies_nash() -> None:
    qs = [e.Queue("train", 100.0), e.Queue("infer", 50.0)]
    ws = [e.Worker("a", 1.0, 1e6), e.Worker("b", 1.0, 1e6), e.Worker("c", 2.0, 1e6)]
    r = e.route_supply_equilibrium(ws, qs)
    assert r.converged and r.is_epsilon_nash
    assert r.allocation["c"]["train"] == 0.0 and r.allocation["c"]["infer"] == 0.0  # too expensive against two cost-1 rivals
    assert r.queue_price["train"] == pytest.approx(r.queue_price["infer"], rel=5e-3)  # marginal returns equalise
    assert r.queue_supply["train"] == pytest.approx(50.0, rel=5e-3)  # two-player contest: S = P / sum c, reserve kappa aside
    assert sum(r.allocation["a"].values()) == pytest.approx(sum(r.allocation["b"].values()), rel=1e-6)


def test_routing_with_binding_capacity_uses_all_capacity_and_zero_allocation_is_never_certified() -> None:
    qs = [e.Queue("train", 100.0), e.Queue("infer", 50.0)]
    ws = [e.Worker("a", 1.0, 5.0), e.Worker("b", 1.0, 5.0), e.Worker("c", 2.0, 5.0)]
    r = e.route_supply_equilibrium(ws, qs)
    assert r.converged and r.is_epsilon_nash
    for w in ws:
        assert sum(r.allocation[w.node_id].values()) == pytest.approx(5.0, rel=1e-6)
    # a reserve of zero is refused: with kappa = 0 an epsilon of lone supply would capture a whole pool
    with pytest.raises(ValueError):
        e.Queue("x", 1.0, reserve_cobblestones=0.0)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(pos, st.floats(min_value=1.0, max_value=100.0)), min_size=2, max_size=5), st.lists(st.floats(min_value=5.0, max_value=500.0), min_size=1, max_size=3))
def test_routing_always_returns_a_certificate_and_a_feasible_allocation(wspec: list[tuple[float, float]], pools: list[float]) -> None:
    ws = [e.Worker(f"w{i}", c, cap) for i, (c, cap) in enumerate(wspec)]
    qs = [e.Queue(f"q{i}", p) for i, p in enumerate(pools)]
    r = e.route_supply_equilibrium(ws, qs)
    for w in ws:
        assert -1e-9 <= sum(r.allocation[w.node_id].values()) <= w.capacity_cobblestones + 1e-6
    assert r.max_deviation_gain_rel >= 0.0
    if r.converged:
        assert r.is_epsilon_nash, r.max_deviation_gain_rel
