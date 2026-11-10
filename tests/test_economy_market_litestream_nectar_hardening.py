"""Negative-control boundary-validation tests for compute_coin, router/market,
router/litestream_lock, and swarm/swarm_nectar_cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.economy.compute_coin import (
    CreditClaim,
    CreditGrant,
    EmissionSchedule,
    Queue,
    Worker,
    apply_settlement,
    claim_figures_are_finite_and_positive,
    is_epsilon_nash,
    route_supply_equilibrium,
    settle_epoch,
    tullock_equilibrium,
    verified_credit,
)
from hivemind.router.litestream_lock import (
    FileGenerationStore,
    InMemoryGenerationStore,
    LitestreamWriterLock,
)
from hivemind.router.market import (
    ConsumerBid,
    FeePolicy,
    ProducerOffer,
    certify,
    clear_market,
)
from hivemind.swarm.swarm_nectar_cache import (
    NectarEntry,
    SwarmNectarCache,
    canonical_query_hash,
)


def test_compute_coin_boundary_validation() -> None:
    with pytest.raises(ValueError, match="CreditClaim"):
        claim_figures_are_finite_and_positive("not-a-claim")  # type: ignore[arg-type]

    bool_claim = CreditClaim(
        node_id="n1",
        workunit_id="w1",
        flops_claimed=True,  # type: ignore[arg-type]
        wall_seconds=1.0,
        class_peak_gflops=10.0,
        spot_check_passed=True,
        multikrum_accepted=True,
    )
    assert claim_figures_are_finite_and_positive(bool_claim) is False

    valid_claim = CreditClaim(
        node_id="n1",
        workunit_id="w1",
        flops_claimed=1e12,
        wall_seconds=10.0,
        class_peak_gflops=100.0,
        spot_check_passed=True,
        multikrum_accepted=True,
    )
    assert claim_figures_are_finite_and_positive(valid_claim) is True

    with pytest.raises(ValueError, match="CreditClaim"):
        verified_credit("not-a-claim")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="replica_claims"):
        verified_credit(valid_claim, "not-a-seq")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CreditClaim"):
        verified_credit(valid_claim, ["bad"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="finite"):
        EmissionSchedule(decay_per_epoch=True)  # type: ignore[arg-type]

    grant = verified_credit(valid_claim)
    assert grant.cobblestones > 0.0

    with pytest.raises(ValueError, match="schedule"):
        settle_epoch(0, [grant], 10.0, 100.0, schedule="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grants"):
        settle_epoch(0, "not-a-seq", 10.0, 100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CreditGrant"):
        settle_epoch(0, ["not-a-grant"], 10.0, 100.0)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        settle_epoch(0, [grant], True, 100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        settle_epoch(
            0,
            [CreditGrant("n1", "w1", True, "verified", "ok")],  # type: ignore[arg-type]
            10.0,
            100.0,
        )

    eq = tullock_equilibrium(100.0, [1.0, 2.0])
    with pytest.raises(ValueError, match="TullockEquilibrium"):
        is_epsilon_nash("not-eq")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eps"):
        is_epsilon_nash(eq, eps=True)  # type: ignore[arg-type]
    assert is_epsilon_nash(eq)[0] is True

    with pytest.raises(ValueError, match="reserve_cobblestones"):
        Queue("q1", 10.0, reserve_cobblestones=True)  # type: ignore[arg-type]

    workers = [Worker("w1", 0.5, 10.0), Worker("w2", 0.8, 10.0)]
    queues = [Queue("q1", 50.0)]
    with pytest.raises(ValueError, match="relaxation"):
        route_supply_equilibrium(workers, queues, relaxation=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        route_supply_equilibrium(workers, queues, tol=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workers and queues"):
        route_supply_equilibrium("bad", queues)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cost and capacity"):
        route_supply_equilibrium([Worker("w1", True, 10.0), Worker("w2", 0.8, 10.0)], queues)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="queue pool"):
        route_supply_equilibrium(workers, [Queue("q1", True)])  # type: ignore[arg-type]

    settlement = settle_epoch(0, [grant], 10.0, 100.0)
    with pytest.raises(ValueError, match="EpochSettlement"):
        apply_settlement("bad", lambda _n, a: a)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="callable"):
        apply_settlement(settlement, "not-callable")  # type: ignore[arg-type]


def test_router_market_and_litestream_lock_boundary_validation(tmp_path: Path) -> None:
    bids = [ConsumerBid("c1", 2.0), ConsumerBid("c2", 3.0)]
    offers = [ProducerOffer("p1", 0.5, 10.0)]
    with pytest.raises(ValueError, match="bids"):
        clear_market("m1", "not-bids", offers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ConsumerBid"):
        clear_market("m1", ["bad"], offers)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="offers"):
        clear_market("m1", bids, "not-offers")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ProducerOffer"):
        clear_market("m1", bids, ["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="FeePolicy"):
        clear_market("m1", bids, offers, fee="bad")  # type: ignore[arg-type]

    eq = clear_market("m1", bids, offers, FeePolicy())
    with pytest.raises(ValueError, match="MarketEquilibrium"):
        certify("not-eq", bids, offers)  # type: ignore[arg-type]
    assert certify(eq, bids, offers).win_win is True

    with pytest.raises(ValueError, match="root_dir"):
        FileGenerationStore("")
    fstore = FileGenerationStore(tmp_path / "locks")
    with pytest.raises(ValueError, match="key"):
        fstore._path_for("")

    with pytest.raises(ValueError, match="backend"):
        LitestreamWriterLock("gcs://bucket/db", "owner-1", object())  # type: ignore[arg-type]
    lock = LitestreamWriterLock("gcs://bucket/db", "owner-1", InMemoryGenerationStore())
    assert lock.acquire(now_utc=1000.0) is True


def test_swarm_nectar_cache_boundary_validation() -> None:
    with pytest.raises(ValueError, match="query"):
        canonical_query_hash("search", 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content"):
        NectarEntry(
            nectar_id="n1",
            tool_name="search",
            query_hash="a" * 64,
            content=[],  # type: ignore[arg-type]
            nectar_richness=0.9,
            discovered_by_bee="bee-1",
        )

    cache = SwarmNectarCache()
    with pytest.raises(ValueError, match="content"):
        cache.put("search", "q", [], "bee-1")  # type: ignore[arg-type]
    entry = cache.put("search", "q", {"ans": 42}, "bee-1")
    assert entry.content["ans"] == 42
    with pytest.raises(ValueError, match="requesting_bee"):
        cache.get("search", "q", requesting_bee=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sender_model_id"):
        cache.put_telepathy_kv(
            sender_model_id="",
            receiver_model_id="m2",
            context_key="ctx",
            compacted_kv_manifest={},
            discovered_by_bee="bee-1",
        )
    with pytest.raises(ValueError, match="compacted_kv_manifest"):
        cache.put_telepathy_kv(
            sender_model_id="m1",
            receiver_model_id="m2",
            context_key="ctx",
            compacted_kv_manifest=[],  # type: ignore[arg-type]
            discovered_by_bee="bee-1",
        )
    with pytest.raises(ValueError, match="context_key"):
        cache.get_telepathy_kv(
            sender_model_id="m1",
            receiver_model_id="m2",
            context_key="  ",
        )
