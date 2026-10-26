"""Tests for hivemind.router.accounts (accounts, API keys, hash-chained HMT ledger)."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hivemind.router.accounts import (
    DuplicateRequest,
    HandleTaken,
    InsufficientBalance,
    RouterStore,
    TooManyHolds,
    UnknownAccount,
)

SECRET = b"s" * 32


@dataclass(frozen=True)
class _Fee:
    validator_share: float = 0.10
    burn_share: float = 0.05


def _store() -> RouterStore:
    return RouterStore(":memory:", SECRET)


def _setup(store: RouterStore) -> tuple[str, str]:
    consumer, _ = store.register("alice", "consumer")
    producer, _ = store.register("bob", "producer")
    return consumer, producer


def test_short_secret_rejected() -> None:
    with pytest.raises(ValueError):
        RouterStore(":memory:", b"x" * 31)
    assert RouterStore(":memory:", b"x" * 32) is not None


def test_register_authenticate_and_key_not_stored() -> None:
    store = _store()
    account_id, key = store.register("Alice", "consumer")
    assert key.startswith("hm-") and len(key) > 40
    acct = store.authenticate(key)
    assert acct is not None and acct.account_id == account_id and acct.role == "consumer"
    assert store.authenticate(key + "x") is None
    assert store.authenticate("hm-nonsense") is None
    dump = "\n".join(store._conn.iterdump())
    assert key not in dump
    # same handle after normalisation is a duplicate; ids are deterministic per secret
    with pytest.raises(HandleTaken):
        store.register("ALICE", "producer")
    assert store.account_id_for("alice") == account_id
    assert RouterStore(":memory:", b"t" * 32).account_id_for("alice") != account_id


def test_invalid_role_rejected() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.register("carol", "wizard")
    assert store.authenticate("hm-x") is None


def test_grant_and_balance() -> None:
    store = _store()
    consumer, _ = _setup(store)
    store.grant(consumer, 10.0, "seed")
    assert store.balance(consumer) == 10.0
    assert store.total_minted() == 10.0
    with pytest.raises(ValueError):
        store.grant(consumer, -1.0, "bad")
    assert store.balance(consumer) == 10.0


def test_settle_splits_and_conserves() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 100.0, "seed")
    s = store.settle("r1", consumer, producer, "m", 400, 600, 1000, 2.0, _Fee())
    assert s.charge == pytest.approx(2.0)
    assert store.balance(consumer) == pytest.approx(98.0)
    assert store.balance(producer) == pytest.approx(1.7)
    assert store.balance("sys:validators") == pytest.approx(0.2)
    assert store.balance("sys:burn") == pytest.approx(0.1)
    assert store.total_balances() == pytest.approx(store.total_minted(), abs=1e-9)
    assert store.verify_chain()
    assert store.recent_usage(consumer)[0]["request_id"] == "r1"


def test_settle_accepts_plain_float_shares_and_default() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 10.0, "seed")
    s = store.settle("a", consumer, producer, "m", 0, 1000, 1000, 1.0, 0.2, 0.1)
    assert s.validator_fee == pytest.approx(0.2) and s.burn == pytest.approx(0.1)
    d = store.settle("b", consumer, producer, "m", 0, 1000, 1000, 1.0)
    assert d.validator_fee == pytest.approx(0.10) and d.burn == pytest.approx(0.05)
    with pytest.raises(ValueError):
        store.settle("c", consumer, producer, "m", 0, 1, 1, 1.0, 0.7, 0.3)


def test_insufficient_balance_writes_nothing() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 1.0, "seed")
    with pytest.raises(InsufficientBalance):
        store.settle("r1", consumer, producer, "m", 0, 2000, 2000, 1.0, _Fee())
    assert store.balance(consumer) == 1.0
    assert store.balance(producer) == 0.0
    assert store.recent_usage(consumer) == []
    # the failed request id is not burned: a smaller retry succeeds
    store.settle("r1", consumer, producer, "m", 0, 500, 500, 1.0, _Fee())
    assert store.balance(consumer) == pytest.approx(0.5)


def test_duplicate_request_rejected() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 5.0, "seed")
    store.settle("r1", consumer, producer, "m", 0, 1000, 1000, 1.0, _Fee())
    with pytest.raises(DuplicateRequest):
        store.settle("r1", consumer, producer, "m", 0, 1000, 1000, 1.0, _Fee())
    assert store.balance(consumer) == pytest.approx(4.0)


def test_tamper_breaks_chain() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 5.0, "seed")
    store.settle("r1", consumer, producer, "m", 0, 1000, 1000, 1.0, _Fee())
    assert store.verify_chain()
    store._conn.execute("UPDATE ledger SET delta = delta + 1000 WHERE seq = 1")
    assert not store.verify_chain()


def test_deleting_a_row_breaks_chain() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 5.0, "seed")
    store.settle("r1", consumer, producer, "m", 0, 1000, 1000, 1.0, _Fee())
    assert store.verify_chain()
    store._conn.execute("DELETE FROM ledger WHERE seq = 2")
    assert not store.verify_chain()


def test_producers_ttl_and_model_filter() -> None:
    store = _store()
    _, producer = _setup(store)
    store.upsert_producer(producer, "https://p.example", [{"id": "hm-moe", "local_name": "qwen3:8b"}], 0.4, 50.0, "gpu", now_utc=1000.0)
    live = store.active_producers("hm-moe", now_utc=1060.0)
    assert [p.account_id for p in live] == [producer]
    assert live[0].local_name("hm-moe") == "qwen3:8b"
    assert store.active_producers("other", now_utc=1060.0) == []
    assert store.active_producers("hm-moe", now_utc=1121.0) == []
    consumer = store.register("dave", "consumer")[0]
    with pytest.raises(ValueError):
        store.upsert_producer(consumer, "https://x", [{"id": "m"}], 1.0, 1.0)


def test_set_max_price_and_listing() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.set_max_price(consumer, 3.5)
    prices = store.consumer_max_prices()
    assert prices[consumer] == 3.5
    assert producer not in prices
    with pytest.raises(ValueError):
        store.set_max_price(consumer, 0.0)


def test_file_database_persists(tmp_path: Path) -> None:
    db = tmp_path / "router.db"
    first = RouterStore(db, SECRET)
    consumer, key = first.register("erin", "consumer")[0], None
    first.grant(consumer, 7.0, "seed")
    first.close()
    second = RouterStore(db, SECRET)
    assert second.balance(consumer) == 7.0
    assert second.verify_chain()
    assert key is None


def test_concurrent_settlements_never_overdraw() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 10.0, "seed")
    results: list[bool] = []

    def worker(i: int) -> None:
        try:
            store.settle(f"r{i}", consumer, producer, "m", 0, 1000, 1000, 1.0, _Fee())
            results.append(True)
        except InsufficientBalance:
            results.append(False)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 10
    assert store.balance(consumer) == pytest.approx(0.0, abs=1e-9)
    assert store.verify_chain()


_op = st.one_of(
    st.tuples(st.just("grant"), st.integers(0, 2), st.floats(0.01, 500.0)),
    st.tuples(st.just("settle"), st.integers(0, 2), st.integers(0, 20000), st.floats(0.0, 20.0)),
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(_op, max_size=25),
    vs=st.floats(0.0, 0.5),
    bs=st.floats(0.0, 0.4),
)
def test_conservation_invariant(ops: list[tuple[object, ...]], vs: float, bs: float) -> None:
    store = _store()
    ids = [store.register(f"user{i}", "prosumer")[0] for i in range(3)]
    for n, op in enumerate(ops):
        if op[0] == "grant":
            store.grant(ids[int(op[1])], float(op[2]), f"g{n}")  # type: ignore[arg-type]
        else:
            consumer = ids[int(op[1])]  # type: ignore[arg-type]
            producer = ids[(int(op[1]) + 1) % 3]  # type: ignore[arg-type]
            try:
                store.settle(f"r{n}", consumer, producer, "m", 0, int(op[2]), int(op[2]), float(op[3]), _Fee(vs, bs))  # type: ignore[arg-type]
            except InsufficientBalance:
                pass
    assert abs(store.total_balances() - store.total_minted()) <= 1e-9 * max(1.0, store.total_minted())
    assert store.min_balance() >= -1e-12
    assert store.balance("sys:validators") >= 0.0 and store.balance("sys:burn") >= 0.0
    assert store.verify_chain()


# ---------------------------------------------------------------------------------------------------------
# security regressions: escrow holds and funded bids
# ---------------------------------------------------------------------------------------------------------


def test_hold_moves_funds_to_escrow_and_a_second_hold_cannot_double_spend() -> None:
    store = _store()
    consumer, _ = _setup(store)
    store.grant(consumer, 10.0, "seed")

    store.hold("r1", consumer, 7.0)

    assert store.balance(consumer) == pytest.approx(3.0)
    assert store.balance("sys:escrow") == pytest.approx(7.0)
    with pytest.raises(InsufficientBalance):
        store.hold("r2", consumer, 7.0)
    assert store.total_balances() == pytest.approx(store.total_minted(), abs=1e-9)
    assert store.release_hold("r1") is True
    assert store.release_hold("r1") is False  # idempotent
    assert store.balance(consumer) == pytest.approx(10.0)
    assert store.balance("sys:escrow") == pytest.approx(0.0)
    assert store.verify_chain()


def test_settle_against_a_hold_is_capped_at_the_escrow_and_never_free() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 10.0, "seed")
    store.hold("r1", consumer, 4.0)

    receipt = store.settle("r1", consumer, producer, "m", 100, 900_000, 1_000_000, 1.0, _Fee())  # asks for 1000 HMT

    assert receipt.charge == pytest.approx(4.0)
    assert receipt.billed_tokens == 4000
    assert store.balance(consumer) == pytest.approx(6.0)
    assert store.balance(producer) == pytest.approx(4.0 * 0.85)
    assert store.active_hold_count(consumer) == 0
    assert store.balance("sys:escrow") == pytest.approx(0.0, abs=1e-12)
    assert store.total_balances() == pytest.approx(store.total_minted(), abs=1e-9)
    assert store.verify_chain()


def test_failed_settlement_keeps_the_hold_and_duplicate_request_is_refused() -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 10.0, "seed")
    store.hold("r1", consumer, 2.0)
    store.settle("r1", consumer, producer, "m", 1, 1, 2, 1.0, _Fee())

    with pytest.raises(DuplicateRequest):
        store.settle("r1", consumer, producer, "m", 1, 1, 2, 1.0, _Fee())
    with pytest.raises(DuplicateRequest):
        store.hold("r1", consumer, 1.0)
    assert store.total_balances() == pytest.approx(store.total_minted(), abs=1e-9)


def test_inflight_holds_are_bounded_and_stale_holds_expire() -> None:
    store = _store()
    consumer, _ = _setup(store)
    store.grant(consumer, 100.0, "seed")
    for i in range(3):
        store.hold(f"r{i}", consumer, 1.0, max_inflight=3, ttl_s=60.0, now_utc=1000.0)

    with pytest.raises(TooManyHolds):
        store.hold("r3", consumer, 1.0, max_inflight=3, ttl_s=60.0, now_utc=1010.0)
    store.hold("r3", consumer, 1.0, max_inflight=3, ttl_s=60.0, now_utc=1100.0)  # the first three expired

    assert store.active_hold_count(consumer) == 1
    assert store.balance(consumer) == pytest.approx(99.0)
    assert store.total_balances() == pytest.approx(store.total_minted(), abs=1e-9)


def test_only_funded_accounts_bid_and_bids_are_capped_by_balance() -> None:
    store = _store()
    rich, _ = store.register("rich", "consumer")
    poor, _ = store.register("poor", "consumer")
    broke, _ = store.register("broke", "prosumer")
    seller, _ = store.register("seller", "producer")
    for account in (rich, poor, broke, seller):
        store.set_max_price(account, 500.0)
    store.grant(rich, 1000.0, "seed")
    store.grant(poor, 0.5, "seed")
    store.grant(seller, 5.0, "seed")

    bids = store.funded_consumer_bids(1000)

    assert set(bids) == {rich, poor}
    assert bids[rich] == pytest.approx(500.0)
    assert bids[poor] == pytest.approx(0.5)



def test_activity_window_drops_idle_high_bidders_but_keeps_recent_ones() -> None:
    """Idle funded accounts that raised their max price must not set a new user's price; active ones still bid."""
    store = _store()
    idle, _ = store.register("idle-whale", "consumer")
    recent, _ = store.register("recent-whale", "consumer")
    old, _ = store.register("old-whale", "consumer")
    newbie, _ = store.register("newbie", "consumer")
    for account in (idle, recent, old):
        store.grant(account, 100.0, "seed")
        store.set_max_price(account, 10.0)
    store.grant(newbie, 100.0, "seed")
    store.hold("req-recent", recent, 1.0, now_utc=10_000.0)
    store.hold("req-old", old, 1.0, now_utc=8_000.0)

    windowed = store.funded_consumer_bids(256, active_since=9_100.0)
    unwindowed = store.funded_consumer_bids(256)

    assert set(windowed) == {recent}
    assert windowed[recent] == pytest.approx(10.0)
    assert set(unwindowed) == {idle, recent, old, newbie}  # control: without the window every funded account bids
    assert store.funded_consumer_bids(256, active_since=8_000.0).keys() == {recent, old}  # boundary is inclusive


def test_activity_window_rejects_a_non_finite_time() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.funded_consumer_bids(256, active_since=float("nan"))
    assert store.funded_consumer_bids(256, active_since=0.0) == {}


def test_test_accounts_never_bid_and_unflagging_restores_the_bid() -> None:
    store = _store()
    real, _ = store.register("real-user", "consumer")
    probe, _ = store.register("probe-user", "consumer", is_test=True)
    for account in (real, probe):
        store.grant(account, 100.0, "seed")
        store.set_max_price(account, 100.0)

    assert store.get_account(probe).is_test is True and store.get_account(real).is_test is False  # type: ignore[union-attr]
    assert set(store.funded_consumer_bids(256)) == {real}
    assert set(store.consumer_max_prices()) == {real}

    store.set_test_flag(real, True)
    assert store.funded_consumer_bids(256) == {}
    store.set_test_flag(probe, False)
    assert set(store.funded_consumer_bids(256)) == {probe}
    assert store.funded_consumer_bids(256)[probe] == pytest.approx(100.0)


def test_set_test_flag_rejects_unknown_and_system_accounts() -> None:
    store = _store()
    with pytest.raises(UnknownAccount):
        store.set_test_flag("f" * 32, True)
    with pytest.raises(UnknownAccount):
        store.set_test_flag("sys:burn", True)
    assert store.get_account("sys:burn").is_test is False  # type: ignore[union-attr]


def test_a_database_made_before_the_test_flag_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE accounts(account_id TEXT PRIMARY KEY, handle TEXT NOT NULL UNIQUE, "
        "role TEXT NOT NULL CHECK (role IN ('consumer','producer','prosumer','admin')), key_hash TEXT NOT NULL UNIQUE, "
        "created_utc REAL NOT NULL, max_price_per_1k REAL NOT NULL)"
    )
    legacy.execute("INSERT INTO accounts VALUES ('old-account','old-user','consumer','hash-1',1.0,2.0)")
    legacy.commit()
    legacy.close()

    store = RouterStore(path, SECRET)
    migrated = store.get_account("old-account")
    assert migrated is not None and migrated.is_test is False and migrated.handle == "old-user"
    store.grant("old-account", 5.0, "seed")
    assert set(store.funded_consumer_bids(256)) == {"old-account"}  # an existing account stays a real bidder
    store.set_test_flag("old-account", True)
    assert store.funded_consumer_bids(256) == {}
    _, key = store.register("new-user", "consumer", is_test=True)
    assert store.authenticate(key).is_test is True  # type: ignore[union-attr]
    store.close()

    reopened = RouterStore(path, SECRET)  # the upgrade is idempotent
    assert reopened.get_account("old-account").is_test is True  # type: ignore[union-attr]
    reopened.close()


def test_user_counts_keep_real_and_test_accounts_apart() -> None:
    store = _store()
    store.register("producer-only", "producer")
    real_a, _ = store.register("real-a", "consumer")
    real_b, _ = store.register("real-b", "consumer")
    probe, _ = store.register("probe", "consumer", is_test=True)
    for account in (real_a, real_b, probe):
        store.grant(account, 10.0, "seed")
    store.hold("req-a", real_a, 1.0, now_utc=time.time() + 100.0)
    store.hold("req-probe", probe, 1.0, now_utc=time.time() + 100.0)
    store.hold("req-b-old", real_b, 1.0, now_utc=1.0)

    everything = store.user_counts(0.0)
    recent = store.user_counts(time.time() + 50.0)

    assert everything == {"real": {"total": 3, "created": 3, "active": 2}, "test": {"total": 1, "created": 1, "active": 1}}
    assert recent == {"real": {"total": 3, "created": 0, "active": 1}, "test": {"total": 1, "created": 0, "active": 1}}
    with pytest.raises(ValueError):
        store.user_counts(float("nan"))

_hold_op = st.one_of(
    st.tuples(st.just("hold"), st.floats(0.0, 30.0)),
    st.tuples(st.just("settle"), st.integers(0, 40_000), st.floats(0.0, 5.0)),
    st.tuples(st.just("release"), st.integers(0, 5)),
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=st.lists(_hold_op, max_size=30))
def test_escrow_preserves_conservation_and_never_overdraws(ops: list[tuple[object, ...]]) -> None:
    store = _store()
    consumer, producer = _setup(store)
    store.grant(consumer, 50.0, "seed")
    open_ids: list[str] = []
    for n, op in enumerate(ops):
        rid = f"r{n}"
        if op[0] == "hold":
            try:
                store.hold(rid, consumer, float(op[1]), max_inflight=6)  # type: ignore[arg-type]
                open_ids.append(rid)
            except (InsufficientBalance, TooManyHolds):
                pass
        elif op[0] == "settle" and open_ids:
            held_id = open_ids.pop(0)
            store.settle(held_id, consumer, producer, "m", 0, int(op[1]), int(op[1]), float(op[2]), _Fee())  # type: ignore[arg-type]
        elif op[0] == "release" and open_ids:
            store.release_hold(open_ids.pop(int(op[1]) % len(open_ids)))  # type: ignore[arg-type]
        assert store.min_balance() >= -1e-9
    assert abs(store.total_balances() - store.total_minted()) <= 1e-9 * max(1.0, store.total_minted())
    assert store.balance("sys:escrow") >= -1e-9
    assert store.active_hold_count(consumer) == len(open_ids)
    assert store.verify_chain()
