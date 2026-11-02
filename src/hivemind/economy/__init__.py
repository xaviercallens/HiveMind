# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind compute economy: BOINC-style verified credit, work-backed $HIVE issuance and Nash-equilibrium markets."""

from hivemind.economy.compute_coin import (
    COBBLESTONE_FLOP,
    SPOT_CHECK_FRACTION,
    CreditClaim,
    CreditGrant,
    EmissionSchedule,
    EpochSettlement,
    KellyEquilibrium,
    Queue,
    RoutingEquilibrium,
    TullockEquilibrium,
    Worker,
    apply_settlement,
    cobblestones_from_flops,
    is_epsilon_nash,
    kelly_equilibrium,
    route_supply_equilibrium,
    settle_epoch,
    tullock_equilibrium,
    claim_figures_are_finite_and_positive,
    verified_credit,
)

__all__ = [
    "COBBLESTONE_FLOP", "SPOT_CHECK_FRACTION", "CreditClaim", "CreditGrant", "EmissionSchedule", "EpochSettlement", "KellyEquilibrium", "Queue",
    "RoutingEquilibrium", "TullockEquilibrium", "Worker", "apply_settlement", "cobblestones_from_flops", "is_epsilon_nash", "kelly_equilibrium",
    "route_supply_equilibrium", "settle_epoch", "tullock_equilibrium", "verified_credit",
    "claim_figures_are_finite_and_positive",
]
