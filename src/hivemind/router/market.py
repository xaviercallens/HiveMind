"""Win-win two-sided market for the HiveMind OpenRouter (pure functions, stdlib + `hivemind.economy.compute_coin`).

Prices are in HMT per 1,000 tokens; producers receive `1 - f` of every payment, `f = validator_share + burn_share`.

Why producer supply is a dominant strategy (Kelly game, linear values). With capacity `S` supplied, consumer j bids
`w` and gets share `w / (w + W_-j)`, so `u_j(w) = v_j S w / (w + W_-j) - w`. The first-order condition
`v_j S W_-j / W^2 = 1` with `W = omega S` and `W_-j = (1 - x_j) W` reads `v_j (1 - x_j) / omega = 1`, i.e.
`x_j = max(0, 1 - omega / v_j)`; `S` has cancelled, so `sum_j x_j = 1` fixes the per-token price `omega` from the values
alone (existence/uniqueness: Johari & Tsitsiklis 2004, "Efficiency loss in a network resource allocation game", Math.
Oper. Res. 29(3)). A producer's payoff `((1 - f) omega - c_i) s` is therefore linear in its own supply `s` with a slope
that depends on neither `s` nor rivals' supply: `s = k_i` if `c_i < (1 - f) omega`, else `s = 0`, whatever others do.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from hivemind.economy.compute_coin import KellyEquilibrium, kelly_equilibrium

MECHANISM_KELLY = "kelly"
MECHANISM_POSTED = "posted_price"
MECHANISM_NO_TRADE = "no_trade"
_MECHANISMS = (MECHANISM_KELLY, MECHANISM_POSTED, MECHANISM_NO_TRADE)

_REL_TOL = 1e-9
_SUPPLY_GRID_POINTS = 41


@dataclass(frozen=True)
class ConsumerBid:
    account_id: str
    max_price_per_1k: float


@dataclass(frozen=True)
class ProducerOffer:
    account_id: str
    cost_per_1k: float
    capacity_1k: float


@dataclass(frozen=True)
class FeePolicy:
    validator_share: float = 0.10
    burn_share: float = 0.05
    posted_margin: float = 0.05

    def __post_init__(self) -> None:
        for name in ("validator_share", "burn_share", "posted_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if self.validator_share + self.burn_share >= 1.0:
            raise ValueError("validator_share + burn_share must be < 1 so producers keep a positive share")

    @property
    def fee_rate(self) -> float:
        return self.validator_share + self.burn_share

    @property
    def producer_share(self) -> float:
        return 1.0 - self.fee_rate

    def to_dict(self) -> dict[str, float]:
        return {"validator_share": self.validator_share, "burn_share": self.burn_share, "posted_margin": self.posted_margin}


@dataclass(frozen=True)
class MarketEquilibrium:
    model_id: str
    mechanism: str
    price_per_1k: float
    supply_1k: float
    consumer_shares: Mapping[str, float]
    consumer_surplus: Mapping[str, float]
    active_producers: tuple[str, ...]
    producer_profit: Mapping[str, float]
    producer_revenue_per_1k: float
    fee: FeePolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_shares", MappingProxyType(dict(self.consumer_shares)))
        object.__setattr__(self, "consumer_surplus", MappingProxyType(dict(self.consumer_surplus)))
        object.__setattr__(self, "producer_profit", MappingProxyType(dict(self.producer_profit)))
        object.__setattr__(self, "active_producers", tuple(self.active_producers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "mechanism": self.mechanism,
            "price_per_1k": self.price_per_1k,
            "supply_1k": self.supply_1k,
            "consumer_shares": dict(self.consumer_shares),
            "consumer_surplus": dict(self.consumer_surplus),
            "active_producers": list(self.active_producers),
            "producer_profit": dict(self.producer_profit),
            "producer_revenue_per_1k": self.producer_revenue_per_1k,
            "fee": self.fee.to_dict(),
        }


@dataclass(frozen=True)
class Certificate:
    individually_rational_consumers: bool
    individually_rational_producers: bool
    producer_nash: bool
    consumer_nash: bool | None  # None for posted_price / no_trade: no bidding game was played
    conserved: bool
    max_deviation_gain: float  # largest unilateral gain found on the deviation grids, in HMT (>= 0)
    win_win: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "individually_rational_consumers": self.individually_rational_consumers,
            "individually_rational_producers": self.individually_rational_producers,
            "producer_nash": self.producer_nash,
            "consumer_nash": self.consumer_nash,
            "conserved": self.conserved,
            "max_deviation_gain": self.max_deviation_gain,
            "win_win": self.win_win,
        }


# ---------------------------------------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------------------------------------


def _clean_bids(bids: Sequence[ConsumerBid]) -> list[tuple[str, float]]:
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for b in bids:
        if not b.account_id or b.account_id in seen:
            raise ValueError(f"consumer account ids must be non-empty and unique, got {b.account_id!r}")
        if not math.isfinite(b.max_price_per_1k) or b.max_price_per_1k < 0.0:
            raise ValueError(f"max_price_per_1k must be finite and >= 0 for {b.account_id!r}")
        seen.add(b.account_id)
        out.append((b.account_id, float(b.max_price_per_1k)))
    return out


def _clean_offers(offers: Sequence[ProducerOffer]) -> list[tuple[str, float, float]]:
    seen: set[str] = set()
    out: list[tuple[str, float, float]] = []
    for o in offers:
        if not o.account_id or o.account_id in seen:
            raise ValueError(f"producer account ids must be non-empty and unique, got {o.account_id!r}")
        if not math.isfinite(o.cost_per_1k) or o.cost_per_1k < 0.0:
            raise ValueError(f"cost_per_1k must be finite and >= 0 for {o.account_id!r}")
        if not math.isfinite(o.capacity_1k) or o.capacity_1k < 0.0:
            raise ValueError(f"capacity_1k must be finite and >= 0 for {o.account_id!r}")
        seen.add(o.account_id)
        out.append((o.account_id, float(o.cost_per_1k), float(o.capacity_1k)))
    return out


@lru_cache(maxsize=128)
def _kelly(values: tuple[float, ...]) -> KellyEquilibrium:
    return kelly_equilibrium(values)


# ---------------------------------------------------------------------------------------------------------
# clearing
# ---------------------------------------------------------------------------------------------------------


def _build(
    model_id: str,
    mechanism: str,
    price: float,
    shares: Mapping[str, float],
    consumers: list[tuple[str, float]],
    producers: list[tuple[str, float, float]],
    fee: FeePolicy,
) -> MarketEquilibrium:
    """Assemble an equilibrium at `price`: producers with c_i < (1-f) price and capacity > 0 supply everything."""
    keep = fee.producer_share
    active = [(a, c, k) for a, c, k in producers if k > 0.0 and c < keep * price]
    supply = math.fsum(k for _, _, k in active)
    share_map = {a: float(shares.get(a, 0.0)) for a, _ in consumers}
    surplus = {a: ((v - price) * share_map[a] * supply if share_map[a] > 0.0 else 0.0) for a, v in consumers}
    active_ids = {a for a, _, _ in active}
    profit = {a: ((keep * price - c) * k if a in active_ids else 0.0) for a, c, k in producers}
    return MarketEquilibrium(
        model_id, mechanism, price, supply, share_map, surplus, tuple(a for a, _, _ in active), profit, keep * price, fee
    )


def _no_trade(
    model_id: str, price: float, consumers: list[tuple[str, float]], producers: list[tuple[str, float, float]], fee: FeePolicy
) -> MarketEquilibrium:
    return MarketEquilibrium(
        model_id,
        MECHANISM_NO_TRADE,
        price,
        0.0,
        {a: 0.0 for a, _ in consumers},
        {a: 0.0 for a, _ in consumers},
        (),
        {a: 0.0 for a, _, _ in producers},
        fee.producer_share * price,
        fee,
    )


def clear_market(model_id: str, bids: Sequence[ConsumerBid], offers: Sequence[ProducerOffer], fee: FeePolicy = FeePolicy()) -> MarketEquilibrium:
    """Clear one model's market for one epoch: Kelly equilibrium if it leaves a producer willing to supply, else a
    posted reserve price at the cheapest producer's break-even price plus `fee.posted_margin`, else no trade."""
    consumers = _clean_bids(bids)
    producers = _clean_offers(offers)
    live = [(a, c, k) for a, c, k in producers if k > 0.0]
    if not live:
        return _no_trade(model_id, 0.0, consumers, producers, fee)
    keep = fee.producer_share
    cheapest = min(c for _, c, _ in live)
    positive = [(a, v) for a, v in consumers if v > 0.0]
    if len(positive) >= 2:
        kel = _kelly(tuple(v for _, v in positive))
        omega = kel.total_payment  # capacity-1 Kelly payment W *is* the per-token price omega
        if cheapest < keep * omega:
            shares = {a: x for (a, _), x in zip(positive, kel.shares)}
            return _build(model_id, MECHANISM_KELLY, omega, shares, consumers, producers, fee)
    reserve = cheapest / keep
    posted = reserve * (1.0 + fee.posted_margin)
    served = [a for a, v in positive if v >= posted]
    has_producer = any(c < keep * posted for _, c, _ in live)
    if not served or not has_producer:
        return _no_trade(model_id, posted, consumers, producers, fee)
    return _build(model_id, MECHANISM_POSTED, posted, {a: 1.0 / len(served) for a in served}, consumers, producers, fee)


def split_payment(price_per_1k: float, tokens: int, fee: FeePolicy) -> tuple[float, float, float, float]:
    """Split one request's charge: returns (charge, producer, validators, burn); charge = price * tokens / 1000."""
    if not math.isfinite(price_per_1k) or price_per_1k < 0.0:
        raise ValueError("price_per_1k must be finite and >= 0")
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    charge = price_per_1k * tokens / 1000.0
    validators = charge * fee.validator_share
    burn = charge * fee.burn_share
    producer = charge - validators - burn
    return charge, producer, validators, burn


# ---------------------------------------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------------------------------------


def _tol(*scales: float) -> float:
    return _REL_TOL * max(1.0, *(abs(s) for s in scales))


def _consumers_rational(eq: MarketEquilibrium, values: Mapping[str, float]) -> bool:
    p, supply = eq.price_per_1k, eq.supply_1k
    if set(eq.consumer_shares) != set(values) or set(eq.consumer_surplus) != set(values):
        return False
    for a, v in values.items():
        x = eq.consumer_shares[a]
        if x < 0.0 or x > 1.0 + _REL_TOL:
            return False
        expected = (v - p) * x * supply if x > 0.0 else 0.0
        tol = _tol(v, p * supply)
        if x > 0.0 and (v < p - tol or expected < -tol):
            return False
        if abs(expected - eq.consumer_surplus[a]) > tol:
            return False
    return True


def _producers_rational(eq: MarketEquilibrium, costs: Mapping[str, tuple[float, float]]) -> bool:
    keep, p = eq.fee.producer_share, eq.price_per_1k
    if set(eq.producer_profit) != set(costs) or not set(eq.active_producers) <= set(costs):
        return False
    active = set(eq.active_producers)
    if len(active) != len(eq.active_producers):
        return False
    if eq.mechanism == MECHANISM_NO_TRADE:
        return not active and all(pr == 0.0 for pr in eq.producer_profit.values())
    for a, (c, k) in costs.items():
        should_supply = k > 0.0 and c < keep * p
        if should_supply != (a in active):
            return False
        expected = (keep * p - c) * k if should_supply else 0.0
        if should_supply and not expected > 0.0:
            return False
        if abs(expected - eq.producer_profit[a]) > _tol(expected, p * k):
            return False
    return True


def _producer_nash(
    eq: MarketEquilibrium, bids: Sequence[ConsumerBid], offers: Sequence[ProducerOffer]
) -> tuple[bool, float]:
    """Re-clear the market with each producer's capacity replaced by s on a 41-point grid in [0, k_i]; the producer's
    profit in the re-cleared market must not beat its profit at the equilibrium (price held by the mechanism, not assumed)."""
    keep, p = eq.fee.producer_share, eq.price_per_1k
    active = set(eq.active_producers)
    worst = 0.0
    ok = True
    for o in offers:
        base = (keep * p - o.cost_per_1k) * o.capacity_1k if o.account_id in active else 0.0
        for i in range(_SUPPLY_GRID_POINTS):
            s = o.capacity_1k * i / (_SUPPLY_GRID_POINTS - 1)
            deviated = [ProducerOffer(q.account_id, q.cost_per_1k, s) if q.account_id == o.account_id else q for q in offers]
            alt = clear_market(eq.model_id, bids, deviated, eq.fee)
            gain = alt.producer_profit.get(o.account_id, 0.0) - base
            worst = max(worst, gain)
            if gain > _tol(base, p * o.capacity_1k):
                ok = False
    return ok, worst


def _consumer_nash(eq: MarketEquilibrium, values: Mapping[str, float]) -> tuple[bool, float]:
    """Best-response check on a log grid of bids around the others' total and around the equilibrium bid."""
    p, supply = eq.price_per_1k, eq.supply_1k
    bid = {a: p * eq.consumer_shares.get(a, 0.0) * supply for a in values}
    total = math.fsum(bid.values())
    worst = 0.0
    ok = True
    for a, v in values.items():
        w_star = bid[a]
        others = math.fsum(w for b, w in bid.items() if b != a)
        if others <= 0.0:
            return False, math.inf
        u_star = v * supply * w_star / (w_star + others) - w_star
        grid = [0.0] + [others * 10.0 ** (k / 10.0) for k in range(-40, 41)]
        if w_star > 0.0:
            grid += [w_star * 10.0 ** (k / 20.0) for k in range(-20, 21)]
        tol = _tol(v * supply, total)
        for w in grid:
            gain = v * supply * w / (w + others) - w - u_star
            worst = max(worst, gain)
            if gain > tol:
                ok = False
    return ok, worst


def _conserved(eq: MarketEquilibrium, costs: Mapping[str, tuple[float, float]]) -> bool:
    p, supply, fee = eq.price_per_1k, eq.supply_1k, eq.fee
    if eq.mechanism == MECHANISM_NO_TRADE:
        return supply == 0.0 and all(x == 0.0 for x in eq.consumer_shares.values()) and not eq.active_producers
    if any(a not in costs for a in eq.active_producers):
        return False
    capacity = math.fsum(costs[a][1] for a in eq.active_producers)
    payments = math.fsum(p * x * supply for x in eq.consumer_shares.values())
    producer_revenue = eq.producer_revenue_per_1k * capacity
    validators = fee.validator_share * p * supply
    burn = fee.burn_share * p * supply
    tol = _tol(payments)
    return (
        abs(math.fsum(eq.consumer_shares.values()) - 1.0) <= _REL_TOL
        and abs(supply - capacity) <= _tol(capacity)
        and abs(eq.producer_revenue_per_1k - fee.producer_share * p) <= _tol(p)
        and abs(payments - (producer_revenue + validators + burn)) <= tol
    )


def certify(eq: MarketEquilibrium, bids: Sequence[ConsumerBid], offers: Sequence[ProducerOffer]) -> Certificate:
    """Recompute the win-win properties of `eq` from the raw bids and offers (never trusting the stored numbers)."""
    values = dict(_clean_bids(bids))
    costs = {a: (c, k) for a, c, k in _clean_offers(offers)}
    if eq.mechanism not in _MECHANISMS:
        return Certificate(False, False, False, None, False, math.inf, False)
    rational_consumers = _consumers_rational(eq, values)
    rational_producers = _producers_rational(eq, costs)
    producer_ok, producer_gain = _producer_nash(eq, bids, offers)
    consumer_ok: bool | None = None
    consumer_gain = 0.0
    if eq.mechanism == MECHANISM_KELLY:
        consumer_ok, consumer_gain = _consumer_nash(eq, values)
    conserved = _conserved(eq, costs)
    win_win = rational_consumers and rational_producers and producer_ok and consumer_ok is not False and conserved
    return Certificate(rational_consumers, rational_producers, producer_ok, consumer_ok, conserved, max(producer_gain, consumer_gain), win_win)
