"""HiveMind computing coin: BOINC-style verified credit, work-backed issuance, and a Nash-equilibrium compute market
(`hivemind.economy.compute_coin`).

Owner's brief (2026-10-24): propose the BOINC computing-coin model for HiveMind and a Nash-equilibrium model that
distributes demand (consumers of compute) and supply (volunteer GPUs) well.

Three layers, all deterministic and torch-free:

1. **Credit** (`cobblestones_from_flops`, `verified_credit`). BOINC's unit: one cobblestone is 1/200 of a day of one
   GFLOPS, i.e. 432 GFLOP. HiveMind grants credit only for *verified* work: the receipt's spot-check must pass and the
   delta must have been admitted by Multi-Krum; when a workunit was replicated, every replica is capped at the
   canonical (smallest) claim, which is BOINC's canonical-result rule. Claimed FLOPs are capped by the class's
   measured throughput so that an over-claim earns nothing extra.
2. **Issuance** (`EmissionSchedule`, `settle_epoch`). Coins are minted *per verified cobblestone*, not per block, with
   a geometric decay per epoch and the ledger's 21,000,000 $HIVE cap as a hard ceiling. Consumers pay in $HIVE; a fee
   split sends most of the payment to the queue's workers, a share to validators and a share is burned, so supply
   tracks useful work and demand.
3. **Market** with closed-form Nash equilibria.
   - Consumers bid for a queue's capacity under the Kelly proportional-allocation mechanism. With price-anticipating
     consumers of linear value `v_j` per unit, the unique Nash equilibrium satisfies `x_j = max(0, 1 - W / v_j)` with
     `sum x_j = 1`, where `W` is the total payment (Johari & Tsitsiklis 2004, "Efficiency loss in a network resource
     allocation game", Math. Oper. Res. 29(3)). Their theorem bounds the price of anarchy by 4/3, so the equilibrium
     captures at least 75 % of the optimal welfare; `kelly_equilibrium` reports the realised efficiency.
   - Workers choose how much verified credit to supply to a queue whose reward pool is shared in proportion to
     credit: a Tullock contest with linear costs `c_i`. Its unique Nash equilibrium (Szidarovszky & Okuguchi 1997,
     Games Econ. Behav. 18) is `S = (n-1) P / sum c` over the active set and `s_i = S (1 - (n-1) c_i / sum c)`;
     workers whose cost is too high are inactive. `tullock_equilibrium` computes it and `is_epsilon_nash` certifies it
     by checking every unilateral deviation on a grid.
   - Across several queues, `route_supply_equilibrium` runs deterministic best-response dynamics with per-worker
     capacity and returns the allocation together with the largest profitable deviation found, so a caller never has
     to trust convergence: the certificate is in the result. At equilibrium marginal returns equalise across queues,
     which is what "well distributed demand" means operationally: no worker can earn more by moving compute, and no
     consumer can buy more value by re-bidding.

Nothing here calls a model, a clock or the network; everything is a pure function of its arguments.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Final, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------------------------------------
# 1. Credit
# ---------------------------------------------------------------------------------------------------------

GFLOP: Final[float] = 1e9
COBBLESTONE_FLOP: Final[float] = GFLOP * 86_400.0 / 200.0  # 432 GFLOP: 1/200 of a day of one GFLOPS (BOINC)
SPOT_CHECK_FRACTION: Final[float] = 0.10  # HiveMind replays 10 % of segments instead of BOINC's quorum = 2


def cobblestones_from_flops(flops: float) -> float:
    """BOINC credit for a FLOP count; negative or non-finite input is an error, never silently zero."""
    if not math.isfinite(flops) or flops < 0.0:
        raise ValueError(f"flops must be finite and >= 0, got {flops!r}")
    return flops / COBBLESTONE_FLOP


@dataclass(frozen=True)
class CreditClaim:
    """What a worker claims for one workunit, with the verification bits the validator attached."""

    node_id: str
    workunit_id: str
    flops_claimed: float
    wall_seconds: float
    class_peak_gflops: float  # measured throughput of the worker's execution class (from the T4/RTX benchmarks)
    spot_check_passed: bool
    multikrum_accepted: bool


@dataclass(frozen=True)
class CreditGrant:
    node_id: str
    workunit_id: str
    cobblestones: float
    capped_by: str  # "verified" | "class_peak" | "canonical_replica" | "rejected"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verified_credit(claim: CreditClaim, replica_claims: Sequence[CreditClaim] = ()) -> CreditGrant:
    """Credit granted for one claim: zero unless verified; capped by the class peak and by the canonical replica."""
    if not claim.spot_check_passed or not claim.multikrum_accepted:
        return CreditGrant(claim.node_id, claim.workunit_id, 0.0, "rejected", "spot-check failed or delta not admitted by Multi-Krum")
    if claim.wall_seconds <= 0.0 or claim.class_peak_gflops <= 0.0:
        return CreditGrant(claim.node_id, claim.workunit_id, 0.0, "rejected", "non-positive wall time or class peak")
    peak_flops = claim.class_peak_gflops * GFLOP * claim.wall_seconds
    flops = cobblestones_from_flops(min(claim.flops_claimed, peak_flops))
    capped_by = "class_peak" if claim.flops_claimed > peak_flops else "verified"
    verified_replicas = [r for r in replica_claims if r.workunit_id == claim.workunit_id and r.spot_check_passed and r.multikrum_accepted and r.node_id != claim.node_id]
    if verified_replicas:
        canonical = min(cobblestones_from_flops(min(r.flops_claimed, r.class_peak_gflops * GFLOP * r.wall_seconds)) for r in verified_replicas)
        if canonical < flops:
            return CreditGrant(claim.node_id, claim.workunit_id, canonical, "canonical_replica", f"capped at the smallest verified replica claim ({len(verified_replicas)} replicas)")
    return CreditGrant(claim.node_id, claim.workunit_id, flops, capped_by, "verified" if capped_by == "verified" else "claim above the class's measured peak throughput; capped")


# ---------------------------------------------------------------------------------------------------------
# 2. Issuance
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EmissionSchedule:
    """$HIVE minted per verified cobblestone, decaying geometrically per epoch, under the ledger's hard cap."""

    initial_hive_per_cobblestone: float = 0.01
    decay_per_epoch: float = 0.999
    max_supply_hive: float = 21_000_000.0
    worker_share: float = 0.80  # of consumer payments
    validator_share: float = 0.15
    burn_share: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.decay_per_epoch <= 1.0) or self.initial_hive_per_cobblestone <= 0.0:
            raise ValueError("decay must be in (0, 1] and the initial emission positive")
        if abs(self.worker_share + self.validator_share + self.burn_share - 1.0) > 1e-12:
            raise ValueError("fee shares must sum to 1")

    def emission_per_cobblestone(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        return self.initial_hive_per_cobblestone * (self.decay_per_epoch ** epoch)


@dataclass(frozen=True)
class EpochSettlement:
    epoch: int
    minted_hive: float
    minted_by_node: Mapping[str, float]
    fees_to_workers: Mapping[str, float]
    fees_to_validators: float
    fees_burned: float
    cap_binding: bool
    total_verified_cobblestones: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["minted_by_node"] = dict(self.minted_by_node)
        d["fees_to_workers"] = dict(self.fees_to_workers)
        return d


def settle_epoch(epoch: int, grants: Sequence[CreditGrant], consumer_payment_hive: float, circulating_supply_hive: float,
                 schedule: EmissionSchedule = EmissionSchedule()) -> EpochSettlement:
    """Mint for verified credit (cap-aware) and split consumer payments pro rata to verified credit."""
    if consumer_payment_hive < 0.0 or circulating_supply_hive < 0.0:
        raise ValueError("payments and supply must be >= 0")
    by_node: dict[str, float] = {}
    for g in grants:
        if g.cobblestones > 0.0:
            by_node[g.node_id] = by_node.get(g.node_id, 0.0) + g.cobblestones
    total = sum(by_node.values())
    rate = schedule.emission_per_cobblestone(epoch)
    wanted = total * rate
    remaining = max(0.0, schedule.max_supply_hive - circulating_supply_hive)
    minted = min(wanted, remaining)
    cap_binding = wanted > remaining
    scale = (minted / wanted) if wanted > 0.0 else 0.0
    minted_by_node = {n: c * rate * scale for n, c in by_node.items()}
    worker_pool = consumer_payment_hive * schedule.worker_share
    fees_to_workers = {n: worker_pool * c / total for n, c in by_node.items()} if total > 0.0 else {}
    burned = consumer_payment_hive * schedule.burn_share
    to_validators = consumer_payment_hive * schedule.validator_share + (worker_pool if total == 0.0 else 0.0)  # nobody worked: validators keep the pool
    return EpochSettlement(epoch, minted, minted_by_node, fees_to_workers, to_validators, burned, cap_binding, total)


# ---------------------------------------------------------------------------------------------------------
# 3a. Consumers: Kelly proportional allocation (Johari & Tsitsiklis 2004)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KellyEquilibrium:
    valuations: tuple[float, ...]
    shares: tuple[float, ...]  # x_j, sum to 1 over the queue's capacity
    payments: tuple[float, ...]  # w_j in $HIVE
    total_payment: float  # W = the market price of the whole capacity
    welfare: float  # sum v_j x_j
    optimal_welfare: float  # max_j v_j (give everything to the highest value)
    efficiency: float  # welfare / optimal_welfare, >= 0.75 by the 4/3 price-of-anarchy bound

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kelly_equilibrium(valuations: Sequence[float], capacity: float = 1.0, tol: float = 1e-12) -> KellyEquilibrium:
    """Unique Nash equilibrium of the price-anticipating Kelly game with linear valuations (needs >= 2 consumers).

    Solves `sum_j max(0, 1 - W / v_j) = 1` for W by bisection (the left side is strictly decreasing in W), then
    `x_j = max(0, 1 - W / v_j)` and `w_j = W x_j`. Shares are reported for a queue of `capacity` units."""
    v = [float(x) for x in valuations]
    if len(v) < 2:
        raise ValueError("the Kelly game needs at least two consumers; a single consumer faces a reserve price, not a game")
    if any((not math.isfinite(x)) or x <= 0.0 for x in v) or capacity <= 0.0:
        raise ValueError("valuations must be finite and > 0; capacity > 0")
    lo, hi = 0.0, max(v)  # at W = max v every share is 0; at W = 0 the sum is n >= 2 > 1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(max(0.0, 1.0 - mid / vj) for vj in v) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    W = 0.5 * (lo + hi)
    x = [max(0.0, 1.0 - W / vj) for vj in v]
    s = sum(x)
    x = [xi / s for xi in x]  # remove the bisection residual so the shares sum to exactly 1
    w = [W * xi for xi in x]
    welfare = sum(vj * xi for vj, xi in zip(v, x))
    opt = max(v)
    return KellyEquilibrium(tuple(v), tuple(xi * capacity for xi in x), tuple(w), W, welfare * capacity, opt * capacity, welfare / opt)


# ---------------------------------------------------------------------------------------------------------
# 3b. Workers: Tullock contest with linear costs (Szidarovszky & Okuguchi 1997)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TullockEquilibrium:
    pool_hive: float
    costs: tuple[float, ...]  # $HIVE per cobblestone supplied
    supplies: tuple[float, ...]  # s_i in cobblestones (0 for inactive workers)
    payoffs: tuple[float, ...]  # P s_i / S - c_i s_i
    total_supply: float
    active: tuple[int, ...]
    price_per_cobblestone: float  # P / S

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tullock_equilibrium(pool_hive: float, costs: Sequence[float]) -> TullockEquilibrium:
    """Unique pure Nash equilibrium of the proportional-share contest with linear costs.

    FOC for an active worker: `P (S - s_i) / S^2 = c_i`. Summing over the n active workers gives
    `S = (n - 1) P / sum c` and `s_i = S (1 - (n - 1) c_i / sum c)`; a worker whose implied supply is not positive
    is inactive, and the active set is found by dropping the most expensive worker until all implied supplies are
    positive (the standard iterative argument; the fixed point is unique)."""
    c = [float(x) for x in costs]
    if len(c) < 2:
        raise ValueError("a contest needs at least two workers; with one worker the pool must be priced by reserve, not by competition")
    if any((not math.isfinite(x)) or x <= 0.0 for x in c) or not math.isfinite(pool_hive) or pool_hive <= 0.0:
        raise ValueError("costs must be finite and > 0; pool > 0")
    order = sorted(range(len(c)), key=lambda i: c[i])
    active = list(order)
    while len(active) >= 2:
        n = len(active)
        csum = sum(c[i] for i in active)
        S = (n - 1) * pool_hive / csum
        s = {i: S * (1.0 - (n - 1) * c[i] / csum) for i in active}
        if all(s[i] > 0.0 for i in active):
            break
        active.pop()  # the most expensive active worker drops out
    else:  # pragma: no cover - unreachable: with the two cheapest workers both supplies are positive
        raise RuntimeError("no active set found")
    supplies = [s.get(i, 0.0) for i in range(len(c))]
    S = sum(supplies)
    payoffs = [(pool_hive * si / S - ci * si) if si > 0.0 else 0.0 for si, ci in zip(supplies, c)]
    return TullockEquilibrium(pool_hive, tuple(c), tuple(supplies), tuple(payoffs), S, tuple(sorted(active)), pool_hive / S)


def is_epsilon_nash(eq: TullockEquilibrium, eps: float = 1e-9, grid: int = 400) -> tuple[bool, float]:
    """Certificate: no worker gains more than `eps` (relative to the pool) by any unilateral deviation on a grid."""
    worst = 0.0
    P, c, s = eq.pool_hive, eq.costs, list(eq.supplies)
    S = sum(s)
    for i in range(len(c)):
        others = S - s[i]
        base = eq.payoffs[i]
        hi = max(2.0 * S, 2.0 * P / c[i])
        for k in range(grid + 1):
            d = hi * k / grid
            tot = others + d
            pay = (P * d / tot - c[i] * d) if tot > 0.0 else 0.0
            worst = max(worst, pay - base)
    return worst <= eps * P, worst / P


# ---------------------------------------------------------------------------------------------------------
# 3c. Several queues: best-response routing with an epsilon-Nash certificate
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Worker:
    node_id: str
    cost_per_cobblestone: float
    capacity_cobblestones: float


@dataclass(frozen=True)
class Queue:
    queue_id: str
    pool_hive: float  # consumer payments routed to this queue (e.g. from kelly_equilibrium) plus emission
    reserve_cobblestones: float = 1.0  # kappa: the share is a / (supply + kappa); the pool is not fully paid out when supply is tiny

    def __post_init__(self) -> None:
        if not math.isfinite(self.reserve_cobblestones) or self.reserve_cobblestones <= 0.0:
            raise ValueError("reserve_cobblestones must be > 0: without it an epsilon of lone supply would capture the whole pool")


@dataclass(frozen=True)
class RoutingEquilibrium:
    allocation: Mapping[str, Mapping[str, float]]  # node_id -> queue_id -> cobblestones
    payoffs: Mapping[str, float]
    queue_supply: Mapping[str, float]
    queue_price: Mapping[str, float]  # pool / supply
    iterations: int
    converged: bool
    max_deviation_gain_rel: float  # largest unilateral improvement found, relative to the largest pool
    is_epsilon_nash: bool

    def to_dict(self) -> dict[str, Any]:
        return {"allocation": {n: dict(q) for n, q in self.allocation.items()}, "payoffs": dict(self.payoffs), "queue_supply": dict(self.queue_supply),
                "queue_price": dict(self.queue_price), "iterations": self.iterations, "converged": self.converged,
                "max_deviation_gain_rel": self.max_deviation_gain_rel, "is_epsilon_nash": self.is_epsilon_nash}


def _payoff(alloc: Mapping[str, float], others: Mapping[str, float], pools: Mapping[str, float], cost: float, reserve: Mapping[str, float]) -> float:
    """Worker payoff: sum over queues of P_q a_q / (O_q + a_q + kappa_q) - c a_q (kappa_q > 0 keeps it continuous at zero supply)."""
    total = 0.0
    for q, a in alloc.items():
        if a > 0.0:
            total += pools[q] * a / (others[q] + a + reserve[q]) - cost * a
    return total


def _best_response(others: Mapping[str, float], pools: Mapping[str, float], w: Worker, reserve: Mapping[str, float]) -> dict[str, float]:
    """Exact best response of one worker: water-filling on marginal returns, then the capacity cap.

    With B_q = O_q + kappa_q > 0 the marginal return in queue q is P_q B_q / (B_q + a)^2 - c, strictly decreasing in
    a, so the optimum equalises marginal returns at a common level lambda >= 0 over the queues it uses:
    a_q(lambda) = sqrt(P_q B_q / (c + lambda)) - B_q, clipped at 0. lambda = 0 if the unconstrained total fits the
    capacity, else lambda is found by bisection so that sum a_q = capacity."""
    qids = list(pools)

    def alloc_for(lam: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for q in qids:
            b = others[q] + reserve[q]
            out[q] = max(0.0, math.sqrt(pools[q] * b / (w.cost_per_cobblestone + lam)) - b)
        return out

    free = alloc_for(0.0)
    if sum(free.values()) <= w.capacity_cobblestones:
        return free
    lo, hi = 0.0, 1.0
    while sum(alloc_for(hi).values()) > w.capacity_cobblestones:
        hi *= 2.0
        if hi > 1e18:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(alloc_for(mid).values()) > w.capacity_cobblestones:
            lo = mid
        else:
            hi = mid
    return alloc_for(hi)


def route_supply_equilibrium(workers: Sequence[Worker], queues: Sequence[Queue], max_iter: int = 500, tol: float = 1e-9, grid: int = 60, relaxation: float = 0.5) -> RoutingEquilibrium:
    """Deterministic damped Gauss-Seidel best-response dynamics over workers; returns the allocation with its certificate.

    `relaxation` in (0, 1] moves each worker a fraction of the way to its best response; 1.0 is the undamped iteration, which
    cycled in 5 of 60 random instances with binding capacities in the first simulation run (artifacts/economy/
    compute_market_report_run1_undamped.json). A damped fixed point is the same equilibrium; the certificate is what counts."""
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    if len(workers) < 2 or not queues:
        raise ValueError("need at least two workers and one queue")
    if len({w.node_id for w in workers}) != len(workers) or len({q.queue_id for q in queues}) != len(queues):
        raise ValueError("duplicate worker or queue ids")
    for w in workers:
        if w.cost_per_cobblestone <= 0.0 or w.capacity_cobblestones <= 0.0:
            raise ValueError(f"worker {w.node_id}: cost and capacity must be > 0")
    pools = {q.queue_id: float(q.pool_hive) for q in queues}
    reserve = {q.queue_id: float(q.reserve_cobblestones) for q in queues}
    if any(p <= 0.0 for p in pools.values()):
        raise ValueError("every queue pool must be > 0")
    # start: each worker spreads a tenth of its capacity evenly (a strictly positive, symmetric start)
    alloc: dict[str, dict[str, float]] = {w.node_id: {q: 0.1 * w.capacity_cobblestones / len(pools) for q in pools} for w in workers}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        delta = 0.0
        for w in workers:
            others = {q: sum(alloc[o.node_id][q] for o in workers if o.node_id != w.node_id) for q in pools}
            br = _best_response(others, pools, w, reserve)
            new = {q: (1.0 - relaxation) * alloc[w.node_id][q] + relaxation * br[q] for q in pools}
            delta = max(delta, max(abs(br[q] - alloc[w.node_id][q]) for q in pools))  # distance to the *best response*, not to the damped step
            alloc[w.node_id] = new
        if delta < tol * max(w.capacity_cobblestones for w in workers):
            converged = True
            break
    # one final undamped sweep: the damped iterate carries a geometric residual (an inactive worker at 1e-4 instead of 0);
    # each worker is set to its exact best response against the others, which is where a converged run already is
    for w in workers:
        others = {q: sum(alloc[o.node_id][q] for o in workers if o.node_id != w.node_id) for q in pools}
        alloc[w.node_id] = _best_response(others, pools, w, reserve)
    supply = {q: sum(alloc[w.node_id][q] for w in workers) for q in pools}
    payoffs = {}
    worst = 0.0
    big = max(pools.values())
    for w in workers:
        others = {q: supply[q] - alloc[w.node_id][q] for q in pools}
        base = _payoff(alloc[w.node_id], others, pools, w.cost_per_cobblestone, reserve)
        payoffs[w.node_id] = base
        br = _best_response(others, pools, w, reserve)
        worst = max(worst, _payoff(br, others, pools, w.cost_per_cobblestone, reserve) - base)
        # second, independent certificate: single-queue deviations on a log grid from 1e-9 x capacity to capacity
        for q in pools:
            for k in range(grid + 1):
                a = w.capacity_cobblestones * 10.0 ** (-9.0 + 9.0 * k / grid)
                cand = {qq: (a if qq == q else 0.0) for qq in pools}
                worst = max(worst, _payoff(cand, others, pools, w.cost_per_cobblestone, reserve) - base)
    price = {q: (pools[q] / supply[q] if supply[q] > 0.0 else math.inf) for q in pools}
    return RoutingEquilibrium({n: dict(a) for n, a in alloc.items()}, payoffs, supply, price, it, converged, worst / big, worst / big <= 1e-6)


# ---------------------------------------------------------------------------------------------------------
# Adapter to the ledger (kept as a callable so this module stays free of the ledger's imports)
# ---------------------------------------------------------------------------------------------------------


def apply_settlement(settlement: EpochSettlement, credit: Callable[[str, float], float], validators_node_id: str = "validators") -> dict[str, float]:
    """Push a settlement into a ledger through its `credit(node_id, amount)` function; returns new balances."""
    out: dict[str, float] = {}
    totals: dict[str, float] = {}
    for n, a in settlement.minted_by_node.items():
        totals[n] = totals.get(n, 0.0) + a
    for n, a in settlement.fees_to_workers.items():
        totals[n] = totals.get(n, 0.0) + a
    if settlement.fees_to_validators > 0.0:
        totals[validators_node_id] = totals.get(validators_node_id, 0.0) + settlement.fees_to_validators
    for n, a in totals.items():
        if a > 0.0:
            out[n] = credit(n, a)
    return out


__all__ = [
    "COBBLESTONE_FLOP", "CreditClaim", "CreditGrant", "EmissionSchedule", "EpochSettlement", "KellyEquilibrium", "Queue", "RoutingEquilibrium",
    "SPOT_CHECK_FRACTION", "TullockEquilibrium", "Worker", "apply_settlement", "cobblestones_from_flops", "is_epsilon_nash", "kelly_equilibrium",
    "route_supply_equilibrium", "settle_epoch", "tullock_equilibrium", "verified_credit",
]
