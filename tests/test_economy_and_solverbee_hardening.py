"""Negative-control tests for compute_coin, international_launch_model, and solverbee.scoring hardening."""

from __future__ import annotations

import math

import pytest

import hivemind.economy.compute_coin as cc
from hivemind.economy.international_launch_model import (
    compute_congestion_price,
    compute_erlang_c_metrics,
    compute_progressive_starter_grant,
    project_international_launch_wave,
)
from hivemind.solverbee import scoring


# ---------------------------------------------------------------------------
# 1. compute_coin.py: verified_credit, EmissionSchedule, settle_epoch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_verified_credit_rejects_empty_identifiers(bad_id: str) -> None:
    with pytest.raises(ValueError, match="node_id"):
        cc.verified_credit(cc.CreditClaim(bad_id, "wu1", 432e9, 100.0, 10.0, True, True))
    with pytest.raises(ValueError, match="workunit_id"):
        cc.verified_credit(cc.CreditClaim("n1", bad_id, 432e9, 100.0, 10.0, True, True))


def test_verified_credit_rejects_non_bool_verification_flags() -> None:
    with pytest.raises(ValueError, match="booleans"):
        cc.verified_credit(cc.CreditClaim("n1", "wu1", 432e9, 100.0, 10.0, 1, True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="booleans"):
        cc.verified_credit(cc.CreditClaim("n1", "wu1", 432e9, 100.0, 10.0, True, "yes"))  # type: ignore[arg-type]


def test_emission_schedule_rejects_negative_shares_and_nonpositive_max_supply() -> None:
    # Negative validator share that still sums to 1.0 must be rejected
    with pytest.raises(ValueError, match="fee shares must be >= 0"):
        cc.EmissionSchedule(worker_share=1.5, validator_share=-0.5, burn_share=0.0)
    with pytest.raises(ValueError, match="max_supply_hive must be > 0"):
        cc.EmissionSchedule(max_supply_hive=0.0)
    with pytest.raises(ValueError, match="max_supply_hive must be > 0"):
        cc.EmissionSchedule(max_supply_hive=-100.0)
    sched = cc.EmissionSchedule()
    with pytest.raises(ValueError, match="epoch"):
        sched.emission_per_cobblestone(True)
    with pytest.raises(ValueError, match="epoch"):
        sched.emission_per_cobblestone(-1)


def test_settle_epoch_rejects_bool_epoch_empty_grant_ids_and_negative_cobblestones() -> None:
    ok_grant = cc.CreditGrant("n1", "wu1", 2.0, "verified", "ok")
    with pytest.raises(ValueError, match="epoch"):
        cc.settle_epoch(True, [ok_grant], 10.0, 0.0)
    with pytest.raises(ValueError, match="node_id"):
        cc.settle_epoch(0, [cc.CreditGrant("  ", "wu1", 2.0, "verified", "ok")], 10.0, 0.0)
    with pytest.raises(ValueError, match="workunit_id"):
        cc.settle_epoch(0, [cc.CreditGrant("n1", "", 2.0, "verified", "ok")], 10.0, 0.0)
    with pytest.raises(ValueError, match="negative cobblestone"):
        cc.settle_epoch(0, [cc.CreditGrant("n1", "wu1", -1.0, "verified", "bad")], 10.0, 0.0)


# ---------------------------------------------------------------------------
# 2. compute_coin.py: kelly_equilibrium, is_epsilon_nash, route_supply_equilibrium, apply_settlement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_tol", [0.0, -1e-9, math.nan, math.inf])
def test_kelly_equilibrium_rejects_bad_tol(bad_tol: float) -> None:
    with pytest.raises(ValueError, match="tol"):
        cc.kelly_equilibrium([2.0, 3.0], tol=bad_tol)


def test_is_epsilon_nash_rejects_forged_or_invalid_inputs() -> None:
    eq = cc.tullock_equilibrium(100.0, [1.0, 2.0])
    ok, _ = cc.is_epsilon_nash(eq)
    assert ok is True

    for bad_eps in (-1e-9, math.nan, math.inf):
        with pytest.raises(ValueError, match="eps"):
            cc.is_epsilon_nash(eq, eps=bad_eps)
    for bad_grid in (0, -5, True):
        with pytest.raises(ValueError, match="grid"):
            cc.is_epsilon_nash(eq, grid=bad_grid)  # type: ignore[arg-type]

    # Forged equilibrium with NaN or Inf payoffs must raise ValueError instead of certifying True
    forged_nan = cc.TullockEquilibrium(
        pool_hive=100.0,
        costs=(1.0, 2.0),
        supplies=eq.supplies,
        payoffs=(math.nan, math.nan),
        total_supply=eq.total_supply,
        active=eq.active,
        price_per_cobblestone=eq.price_per_cobblestone,
    )
    with pytest.raises(ValueError, match="payoffs"):
        cc.is_epsilon_nash(forged_nan)

    forged_inf = cc.TullockEquilibrium(
        pool_hive=100.0,
        costs=(1.0, 2.0),
        supplies=eq.supplies,
        payoffs=(math.inf, math.inf),
        total_supply=eq.total_supply,
        active=eq.active,
        price_per_cobblestone=eq.price_per_cobblestone,
    )
    with pytest.raises(ValueError, match="payoffs"):
        cc.is_epsilon_nash(forged_inf)

    forged_pool = cc.TullockEquilibrium(
        pool_hive=0.0,
        costs=(1.0, 2.0),
        supplies=eq.supplies,
        payoffs=eq.payoffs,
        total_supply=eq.total_supply,
        active=eq.active,
        price_per_cobblestone=eq.price_per_cobblestone,
    )
    with pytest.raises(ValueError, match="pool_hive"):
        cc.is_epsilon_nash(forged_pool)


def test_route_supply_equilibrium_rejects_bad_parameters_and_empty_ids() -> None:
    ws = [cc.Worker("w1", 1.0, 10.0), cc.Worker("w2", 1.0, 10.0)]
    qs = [cc.Queue("q1", 100.0)]
    for bad_rel in (0.0, -0.5, 1.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="relaxation"):
            cc.route_supply_equilibrium(ws, qs, relaxation=bad_rel)
    for bad_iter in (0, -1, True):
        with pytest.raises(ValueError, match="max_iter"):
            cc.route_supply_equilibrium(ws, qs, max_iter=bad_iter)  # type: ignore[arg-type]
    for bad_grid in (0, -1, True):
        with pytest.raises(ValueError, match="grid"):
            cc.route_supply_equilibrium(ws, qs, grid=bad_grid)  # type: ignore[arg-type]
    for bad_tol in (-1e-9, math.nan, math.inf):
        with pytest.raises(ValueError, match="tol"):
            cc.route_supply_equilibrium(ws, qs, tol=bad_tol)
    with pytest.raises(ValueError, match="node_id"):
        cc.route_supply_equilibrium([cc.Worker("  ", 1.0, 10.0), cc.Worker("w2", 1.0, 10.0)], qs)
    with pytest.raises(ValueError, match="queue_id"):
        cc.route_supply_equilibrium(ws, [cc.Queue("  ", 100.0)])


def test_apply_settlement_rejects_invalid_amounts_and_node_ids() -> None:
    base = cc.settle_epoch(0, [cc.CreditGrant("n1", "wu1", 2.0, "verified", "ok")], 10.0, 0.0)
    with pytest.raises(ValueError, match="validators_node_id"):
        cc.apply_settlement(base, lambda n, a: a, validators_node_id="  ")

    bad_val = cc.EpochSettlement(0, 1.0, {"n1": 1.0}, {"n1": 1.0}, -1.0, 0.0, False, 1.0)
    with pytest.raises(ValueError, match="fees_to_validators"):
        cc.apply_settlement(bad_val, lambda n, a: a)

    bad_mint = cc.EpochSettlement(0, 1.0, {"n1": math.nan}, {}, 0.0, 0.0, False, 1.0)
    with pytest.raises(ValueError, match="minted_by_node"):
        cc.apply_settlement(bad_mint, lambda n, a: a)

    bad_worker_fee = cc.EpochSettlement(0, 1.0, {"n1": 1.0}, {"n1": -5.0}, 0.0, 0.0, False, 1.0)
    with pytest.raises(ValueError, match="fees_to_workers"):
        cc.apply_settlement(bad_worker_fee, lambda n, a: a)


# ---------------------------------------------------------------------------
# 3. international_launch_model.py: Erlang-C, starter grant, congestion price, launch wave
# ---------------------------------------------------------------------------


def test_compute_erlang_c_metrics_rejects_bool_overhead_and_safe_rho_bounds() -> None:
    with pytest.raises(ValueError, match="active_producers_c"):
        compute_erlang_c_metrics(True, 380.0, 0.4)
    for bad_ovh in (-0.01, math.nan, math.inf):
        with pytest.raises(ValueError):
            compute_erlang_c_metrics(8, 380.0, 0.4, base_network_overhead_s=bad_ovh)
    for bad_ceil in (0.0, 1.0, -0.5, 1.2, math.nan):
        with pytest.raises(ValueError):
            compute_erlang_c_metrics(8, 380.0, 0.4, safe_rho_ceiling=bad_ceil)


def test_compute_progressive_starter_grant_rejects_invalid_inputs() -> None:
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            compute_progressive_starter_grant(bad, 400.0)
        with pytest.raises(ValueError):
            compute_progressive_starter_grant(200.0, bad)
    with pytest.raises(ValueError, match="legacy_lump_sum_hmt"):
        compute_progressive_starter_grant(200.0, 400.0, legacy_lump_sum_hmt=0.0)
    with pytest.raises(ValueError, match="legacy_lump_sum_hmt"):
        compute_progressive_starter_grant(200.0, 400.0, legacy_lump_sum_hmt=10.0, instant_grant_hmt=15.0)


def test_compute_congestion_price_rejects_invalid_inputs() -> None:
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            compute_congestion_price(bad, 0.5)
        with pytest.raises(ValueError, match="finite"):
            compute_congestion_price(0.4, bad)
    with pytest.raises(ValueError, match="utilization_rho"):
        compute_congestion_price(0.4, -0.1)
    with pytest.raises(ValueError, match="rho_threshold"):
        compute_congestion_price(0.4, 0.5, rho_threshold=0.95, rho_cap=0.60)
    with pytest.raises(ValueError, match="surge_gamma"):
        compute_congestion_price(0.4, 0.5, surge_gamma=-1.0)
    for bad_fee in (-0.1, 1.0, 1.2):
        with pytest.raises(ValueError, match="validator_plus_burn_fee"):
            compute_congestion_price(0.4, 0.5, validator_plus_burn_fee=bad_fee)


def test_project_international_launch_wave_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="wave_name"):
        project_international_launch_wave("   ", 1000)
    for bad_vis in (0, -10, True, 1_000_001):
        with pytest.raises(ValueError, match="new_visitors"):
            project_international_launch_wave("Wave", bad_vis)  # type: ignore[arg-type]
    for bad_rate in (0.0, -0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="signup_rate"):
            project_international_launch_wave("Wave", 1000, signup_rate=bad_rate)
    for bad_alpha in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError, match="direct_producer_fraction_alpha_p"):
            project_international_launch_wave("Wave", 1000, direct_producer_fraction_alpha_p=bad_alpha)
    for bad_kappa in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="consumer_to_prosumer_conversion_kappa"):
            project_international_launch_wave("Wave", 1000, consumer_to_prosumer_conversion_kappa=bad_kappa)


# ---------------------------------------------------------------------------
# 4. solverbee/scoring.py: wilson_lower & recruitment_weight
# ---------------------------------------------------------------------------


def test_solverbee_scoring_rejects_negative_n_bool_and_impossible_counts() -> None:
    with pytest.raises(ValueError, match="n must be >= 0"):
        scoring.wilson_lower(0, -1)
    with pytest.raises(ValueError, match="within 0..n"):
        scoring.wilson_lower(1, 0)
    with pytest.raises(ValueError, match="integers"):
        scoring.wilson_lower(True, 10)
    with pytest.raises(ValueError, match="integers"):
        scoring.wilson_lower(5, True)
    for bad_z in (0.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="z must be finite and > 0"):
            scoring.wilson_lower(5, 10, z=bad_z)  # type: ignore[arg-type]

    # recruitment_weight must reject impossible counts even below MIN_VERDICTS (cold start)
    with pytest.raises(ValueError, match="within 0..n"):
        scoring.recruitment_weight(10, 3)
    with pytest.raises(ValueError, match="within 0..n"):
        scoring.recruitment_weight(-1, 3)
    with pytest.raises(ValueError, match="within 0..n"):
        scoring.recruitment_weight(0, -5)
    with pytest.raises(ValueError, match="integers"):
        scoring.recruitment_weight(True, 10)
