"""End users share verified knowledge as public data and receive a lump-sum HMT bonus.

A HiveMinder who opts in publishes a verified triple (task, accepted solution, the tests that accepted it) under an
open licence. Admission is strict: the share must be verified, carry a permitted licence, pass the prompt-injection
scan of the firewall, and come from an identity that has not exceeded its per-epoch cap. Each admitted share is one
knowledge point.

The bonus is paid from a fraction of the validator share of the epoch's fees, split in proportion to points. It is a
lump sum: it does not depend on how much capacity a producer supplies or how a consumer bids, so the Kelly price and
every best response of `hivemind.router.market` are unchanged (Theorems 137-138), and the payouts never exceed the pool
(Theorem 139). The certificate below recomputes these facts; it never trusts the stored numbers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from hivemind.router.market import FeePolicy, MarketEquilibrium

PERMITTED_LICENCES: frozenset[str] = frozenset({"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0"})
SHARE_KINDS: frozenset[str] = frozenset({"verified_code", "verified_numeric", "verified_mcq", "verified_label"})


@dataclass(frozen=True)
class KnowledgeShare:
    account_id: str
    kind: str
    licence: str
    verified: bool
    epoch: int
    text_sha256: str

    def __post_init__(self) -> None:
        if not self.account_id:
            raise ValueError("account_id must be non-empty")
        if self.kind not in SHARE_KINDS:
            raise ValueError(f"kind must be one of {sorted(SHARE_KINDS)}")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if len(self.text_sha256) != 64:
            raise ValueError("text_sha256 must be a hex sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    share: KnowledgeShare


Scanner = Callable[[str], bool]  # True when the text is clean


def firewall_scanner(text: str) -> bool:
    """The repository's prompt-injection scan; shared text is data and must not carry instructions."""
    from hivemind.security.hivemind_firewall import scan_text_for_injection

    return scan_text_for_injection(text).verdict.value == "CLEAN"


def admit_share(share: KnowledgeShare, text: str, *, scanner: Scanner = firewall_scanner, prior_points_this_epoch: int = 0,
                cap_per_identity_per_epoch: int = 20) -> Admission:
    """Admit or refuse a share. The hash in the share must match the text, so the record cannot be swapped later."""
    if cap_per_identity_per_epoch < 1:
        raise ValueError("cap_per_identity_per_epoch must be >= 1")
    if not share.verified:
        return Admission(False, "not_verified", share)
    if share.licence not in PERMITTED_LICENCES:
        return Admission(False, "licence_not_permitted", share)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != share.text_sha256:
        return Admission(False, "hash_mismatch", share)
    if prior_points_this_epoch >= cap_per_identity_per_epoch:
        return Admission(False, "identity_cap_reached", share)
    if not scanner(text):
        return Admission(False, "injection_scan_failed", share)
    return Admission(True, "admitted", share)


def knowledge_points(admissions: Iterable[Admission]) -> dict[str, int]:
    """One point per admitted share, per account."""
    out: dict[str, int] = {}
    for a in admissions:
        if a.admitted:
            out[a.share.account_id] = out.get(a.share.account_id, 0) + 1
    return out


@dataclass(frozen=True)
class KnowledgeSettlement:
    epoch: int
    validator_fee_hmt: float
    bonus_share: float
    pool_hmt: float
    payouts_hmt: Mapping[str, float]
    points: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "validator_fee_hmt": self.validator_fee_hmt, "bonus_share": self.bonus_share,
                "pool_hmt": self.pool_hmt, "payouts_hmt": dict(self.payouts_hmt), "points": dict(self.points)}


def settle_knowledge_bonus(epoch: int, validator_fee_hmt: float, bonus_share: float, points: Mapping[str, int]) -> KnowledgeSettlement:
    """Split `bonus_share` of the validator fee in proportion to points. Accounts with zero points get nothing; with no
    points at all the pool is not paid (it stays with the validators)."""
    if not math.isfinite(validator_fee_hmt) or validator_fee_hmt < 0.0:
        raise ValueError("validator_fee_hmt must be finite and >= 0")
    if not (0.0 <= bonus_share <= 1.0):
        raise ValueError("bonus_share must be in [0, 1]")
    if any(p < 0 for p in points.values()):
        raise ValueError("points must be >= 0")
    pool = validator_fee_hmt * bonus_share
    total = sum(points.values())
    payouts = {a: pool * p / total for a, p in points.items() if p > 0} if total > 0 else {}
    return KnowledgeSettlement(epoch, validator_fee_hmt, bonus_share, pool, payouts, dict(points))


@dataclass(frozen=True)
class KnowledgeCertificate:
    price_unchanged: bool
    supply_unchanged: bool
    payouts_within_pool: bool
    pool_within_validator_fee: bool
    nobody_paid_without_points: bool
    win_win_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return all(asdict(self).values())


def certify_knowledge_bonus(before: MarketEquilibrium, after: MarketEquilibrium, settlement: KnowledgeSettlement,
                            fee: FeePolicy, epoch_payment_hmt: float, win_win_before: bool, win_win_after: bool) -> KnowledgeCertificate:
    """Recompute, from the two equilibria and the settlement, that the bonus changed nothing it must not change.

    `before` is the market cleared without the bonus and `after` with it; a correct implementation produces identical
    equilibria because the bonus is a lump sum paid outside the bidding game. `epoch_payment_hmt` is the total paid by
    consumers in the epoch, from which the validator fee is `fee.validator_share * epoch_payment_hmt`.
    """
    tol = 1e-9 * max(1.0, abs(before.price_per_1k), abs(before.supply_1k))
    price_same = abs(before.price_per_1k - after.price_per_1k) <= tol and before.mechanism == after.mechanism
    supply_same = abs(before.supply_1k - after.supply_1k) <= tol and before.active_producers == after.active_producers
    paid = sum(settlement.payouts_hmt.values())
    within_pool = paid <= settlement.pool_hmt + 1e-9 * max(1.0, settlement.pool_hmt)
    validator_fee = fee.validator_share * epoch_payment_hmt
    pool_ok = settlement.pool_hmt <= validator_fee + 1e-9 * max(1.0, validator_fee)
    nobody = all(settlement.points.get(a, 0) > 0 for a in settlement.payouts_hmt)
    return KnowledgeCertificate(price_same, supply_same, within_pool, pool_ok, nobody, win_win_before == win_win_after)
