"""HM-S16-07: test and probe accounts stay off the market and out of the user counts; starter grants are capped per client address."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore, UnknownAccount
from tests.test_router_gateway import ADMIN_HEADERS, SECRET, FakeProducer, World, _auth, _build_world

DAY = 86_400.0
PROXY = "10.0.0.1"


def _market(world: World, key: str) -> dict[str, Any]:
    reply = world.app_client.get("/v1/hivemind/market?model=hivemind/qwen", headers=_auth(key))
    assert reply.status_code == 200, reply.text
    return dict(reply.json()["equilibrium"])


def _bidder(world: World, kind: str, index: int) -> str:
    """A funded account at max price 100 of the given kind ('user' registers openly, others need the admin key)."""
    headers = {} if kind == "user" else ADMIN_HEADERS
    reply = world.app_client.post("/v1/hivemind/register", headers=headers, json={"handle": f"{kind}-{index}", "role": "consumer", "kind": kind})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    world.store.grant(body["account_id"], 100.0, "seed")
    assert world.app_client.post("/v1/hivemind/consumer/max-price", headers=_auth(body["api_key"]), json={"max_price_per_1k": 100.0}).status_code == 200
    return str(body["api_key"])


def _world_with_market() -> tuple[World, str]:
    world = _build_world()
    world.producer("prod-a.test", {"qwen": "q"})
    _, key = world.consumer(100.0)
    return world, key


# ------------------------------------------------------------------ market: kinds off the bids


@pytest.mark.parametrize("kind", ["test", "probe"])
def test_test_and_probe_bidders_at_max_price_leave_price_and_bidder_count_unchanged(kind: str) -> None:
    world, key = _world_with_market()
    before = _market(world, key)
    for i in range(5):
        _bidder(world, kind, i)
    world.clock.now += 15.0  # past the 10 s market cache epoch, inside the producer TTL
    after = _market(world, key)

    assert after["price_per_1k"] == pytest.approx(before["price_per_1k"])
    assert after["num_consumers"] == before["num_consumers"]


def test_the_same_number_of_user_bidders_does_move_price_and_count() -> None:
    world, key = _world_with_market()
    before = _market(world, key)
    for i in range(5):
        _bidder(world, "user", i)
    world.clock.now += 15.0
    after = _market(world, key)

    assert after["num_consumers"] == before["num_consumers"] + 5
    assert after["price_per_1k"] != pytest.approx(before["price_per_1k"])


def test_a_probe_account_is_served_at_the_price_computed_without_it() -> None:
    world, key = _world_with_market()
    probe_key = _bidder(world, "probe", 0)
    user_price = world.chat(key).json()["hivemind"]["price_per_1k"]
    probe_reply = world.chat(probe_key)

    assert probe_reply.status_code == 200, probe_reply.text
    assert probe_reply.json()["hivemind"]["price_per_1k"] == pytest.approx(user_price)


def test_test_and_probe_producers_are_not_supply_for_pricing() -> None:
    world = _build_world()
    reply = world.app_client.post("/v1/hivemind/register", headers=ADMIN_HEADERS, json={"handle": "probe-gpu", "role": "producer", "kind": "probe"})
    key = reply.json()["api_key"]
    world.producers["gpu.test"] = FakeProducer()
    assert world.beat(key, "gpu.test", {"qwen": "q"}, 0.01, 100.0).status_code == 200
    _, consumer_key = world.consumer(100.0)

    refused = world.chat(consumer_key)
    market = _market(world, consumer_key)

    assert refused.status_code == 503  # the only offer is a probe: no supply counts
    assert market["supply_1k"] == 0.0 and market["active_producers"] == []


# ------------------------------------------------------------------ who may set the kind


@pytest.mark.parametrize("body", [{"kind": "test"}, {"kind": "probe"}, {"test": True}])
def test_only_an_admin_can_register_a_non_user_kind(body: dict[str, Any]) -> None:
    world = _build_world()
    anonymous = world.app_client.post("/v1/hivemind/register", json={"handle": "sneaky", "role": "consumer", **body})
    wrong = world.app_client.post("/v1/hivemind/register", headers={"X-Admin-Key": "nope"}, json={"handle": "sneaky", "role": "consumer", **body})

    assert anonymous.status_code == 403 and wrong.status_code == 403
    assert world.store.user_counts(0.0)["real"]["total"] == 0


def test_a_user_cannot_change_its_own_kind_and_the_admin_endpoint_checks_the_key() -> None:
    world = _build_world()
    user_id, key = world.register("consumer")
    url = "/v1/hivemind/admin/test-flag"

    assert world.app_client.post(url, json={"account_id": user_id, "kind": "probe"}).status_code == 403
    assert world.app_client.post(url, headers=_auth(key), json={"account_id": user_id, "kind": "probe"}).status_code == 403
    assert world.app_client.post(url, headers={"X-Admin-Key": "wrong"}, json={"account_id": user_id, "kind": "probe"}).status_code == 403
    assert world.store.get_account(user_id).kind == "user"  # type: ignore[union-attr]

    ok = world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": user_id, "kind": "probe"})
    assert ok.status_code == 200 and ok.json() == {"account_id": user_id, "test": True, "kind": "probe"}
    assert world.app_client.get("/v1/hivemind/me", headers=_auth(key)).json()["kind"] == "probe"
    back = world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": user_id, "kind": "user"})
    assert back.json()["kind"] == "user"


def test_invalid_kinds_and_contradictions_are_rejected() -> None:
    world = _build_world()
    user_id, _ = world.register("consumer")
    url = "/v1/hivemind/admin/test-flag"

    assert world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": user_id, "kind": "admin"}).status_code == 400
    assert world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": user_id}).status_code == 400
    assert world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": user_id, "kind": "user", "test": True}).status_code == 400
    both = world.app_client.post("/v1/hivemind/register", headers=ADMIN_HEADERS, json={"handle": "x1", "role": "consumer", "kind": "user", "test": True})
    assert both.status_code == 400
    assert world.app_client.post(url, headers=ADMIN_HEADERS, json={"account_id": "f" * 32, "kind": "probe"}).status_code == 404


# ------------------------------------------------------------------ metrics


def test_user_counts_report_test_and_probe_accounts_as_separate_fields() -> None:
    world = _build_world()
    world.register("consumer")
    _bidder(world, "test", 0)
    _bidder(world, "probe", 0)
    _bidder(world, "probe", 1)

    report = world.app_client.get("/v1/hivemind/admin/users?window_s=3600", headers=ADMIN_HEADERS)
    body = report.json()

    assert report.status_code == 200
    assert (body["real"]["total"], body["test"]["total"], body["probe"]["total"]) == (1, 1, 2)
    assert world.app_client.get("/v1/hivemind/admin/users").status_code == 403


def test_store_kind_validation_and_system_accounts() -> None:
    store = RouterStore(":memory:", SECRET)
    with pytest.raises(ValueError):
        store.register("bad-kind", "consumer", kind="vip")
    with pytest.raises(UnknownAccount):
        store.set_kind("sys:burn", "probe")
    account, _ = store.register("pro", "consumer", kind="probe")
    assert store.non_user_account_ids() == {account}


def test_a_database_with_flagged_accounts_gets_kinds_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE accounts(account_id TEXT PRIMARY KEY, handle TEXT NOT NULL UNIQUE, "
        "role TEXT NOT NULL CHECK (role IN ('consumer','producer','prosumer','admin')), key_hash TEXT NOT NULL UNIQUE, "
        "created_utc REAL NOT NULL, max_price_per_1k REAL NOT NULL, is_test INTEGER NOT NULL DEFAULT 0)"
    )
    legacy.execute("INSERT INTO accounts VALUES ('a-real','real','consumer','h1',1.0,2.0,0)")
    legacy.execute("INSERT INTO accounts VALUES ('a-test','tst','consumer','h2',1.0,2.0,1)")
    legacy.commit()
    legacy.close()

    store = RouterStore(path, SECRET)
    assert store.get_account("a-real").kind == "user" and store.get_account("a-test").kind == "test"  # type: ignore[union-attr]
    store.close()
    again = RouterStore(path, SECRET)  # idempotent
    assert again.get_account("a-test").kind == "test"  # type: ignore[union-attr]
    again.close()


# ------------------------------------------------------------------ starter grant cap per client address


def _signup(client: TestClient, handle: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    reply = client.post("/v1/hivemind/register", headers=headers or {}, json={"handle": handle, "role": "consumer"})
    assert reply.status_code == 200, reply.text
    return dict(reply.json())


def _client_at(world: World, host: str) -> TestClient:
    return TestClient(world.app_client.app, client=(host, 50000))


def test_no_starter_grant_by_default() -> None:
    world = _build_world()
    body = _signup(world.app_client, "plain")
    assert "starter_grant" not in body and world.store.balance(body["account_id"]) == 0.0


def test_the_fourth_signup_from_one_address_gets_no_grant_but_is_registered() -> None:
    world = _build_world(starter_grant_hmt=100.0)
    client = _client_at(world, "203.0.113.7")
    bodies = [_signup(client, f"user-{i}") for i in range(4)]

    assert [b["starter_grant"] for b in bodies] == [100.0, 100.0, 100.0, 0.0]
    assert "starter_grant_note" in bodies[3] and not any(ch.isdigit() for ch in bodies[3]["starter_grant_note"])  # no counts leaked
    assert world.store.balance(bodies[3]["account_id"]) == 0.0
    assert world.store.balance(bodies[0]["account_id"]) == 100.0
    assert world.store.authenticate(bodies[3]["api_key"]) is not None  # still a working account
    assert _signup(_client_at(world, "203.0.113.8"), "other")["starter_grant"] == 100.0  # another address is unaffected


def test_the_cap_is_configurable_and_the_window_expires() -> None:
    world = _build_world(starter_grant_hmt=50.0, starter_grants_per_ip_24h=1)
    client = _client_at(world, "203.0.113.7")
    assert _signup(client, "acct-1")["starter_grant"] == 50.0
    assert _signup(client, "acct-2")["starter_grant"] == 0.0
    world.clock.now += DAY - 60.0
    assert _signup(client, "acct-3")["starter_grant"] == 0.0
    world.clock.now += 120.0  # more than 24 h after the first grant
    assert _signup(client, "acct-4")["starter_grant"] == 50.0


def test_admin_made_test_and_probe_accounts_take_no_starter_grant_and_no_slot() -> None:
    world = _build_world(starter_grant_hmt=100.0, starter_grants_per_ip_24h=1)
    client = _client_at(world, "203.0.113.7")
    probe = client.post("/v1/hivemind/register", headers=ADMIN_HEADERS, json={"handle": "prb", "role": "consumer", "kind": "probe"}).json()

    assert "starter_grant" not in probe and world.store.balance(probe["account_id"]) == 0.0
    assert _signup(client, "real")["starter_grant"] == 100.0


def test_a_spoofed_forwarded_for_header_is_ignored_without_a_trusted_proxy() -> None:
    world = _build_world(starter_grant_hmt=100.0, starter_grants_per_ip_24h=1)
    client = _client_at(world, "203.0.113.7")
    first = _signup(client, "spoof-1", {"X-Forwarded-For": "198.51.100.1"})
    second = _signup(client, "spoof-2", {"X-Forwarded-For": "198.51.100.2"})

    assert first["starter_grant"] == 100.0 and second["starter_grant"] == 0.0


def test_a_trusted_proxy_forwarded_for_is_honoured_and_a_forged_left_entry_is_not() -> None:
    world = _build_world(starter_grant_hmt=100.0, starter_grants_per_ip_24h=1, trusted_proxies=[PROXY])
    proxy = _client_at(world, PROXY)
    assert _signup(proxy, "trust-1", {"X-Forwarded-For": "198.51.100.1"})["starter_grant"] == 100.0
    assert _signup(proxy, "trust-2", {"X-Forwarded-For": "198.51.100.2"})["starter_grant"] == 100.0  # a different real client behind the proxy
    assert _signup(proxy, "trust-3", {"X-Forwarded-For": "198.51.100.1"})["starter_grant"] == 0.0
    # a client prepends a fake entry: the right-most non-proxy entry is the one the proxy saw
    assert _signup(proxy, "trust-4", {"X-Forwarded-For": "198.51.100.99, 198.51.100.1"})["starter_grant"] == 0.0
    # the header from a peer that is not the proxy is still ignored
    stranger = _client_at(world, "203.0.113.50")
    assert _signup(stranger, "trust-5", {"X-Forwarded-For": "198.51.100.77"})["starter_grant"] == 100.0
    assert _signup(stranger, "trust-6", {"X-Forwarded-For": "198.51.100.78"})["starter_grant"] == 0.0


def test_the_raw_address_is_never_stored(tmp_path: Path) -> None:
    path = tmp_path / "router.db"
    store = RouterStore(path, SECRET)
    assert store.claim_starter_slot("203.0.113.7", 1000.0, 3)
    store.close()

    raw = path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        raw += wal.read_bytes()
    assert b"203.0.113.7" not in raw
    rows = sqlite3.connect(path).execute("SELECT ip_hash, utc FROM starter_grant_ips").fetchall()
    assert len(rows) == 1 and len(rows[0][0]) == 64 and rows[0][1] == 1000.0


def test_old_entries_are_purged_on_the_next_claim() -> None:
    store = RouterStore(":memory:", SECRET)
    assert store.claim_starter_slot("203.0.113.7", 0.0, 1)
    assert not store.claim_starter_slot("203.0.113.7", 10.0, 1)
    assert store.claim_starter_slot("203.0.113.9", DAY + 5.0, 1)  # any claim after the window drops the old rows
    assert store._conn.execute("SELECT COUNT(*) FROM starter_grant_ips").fetchone()[0] == 1


def test_a_bad_starter_configuration_is_refused() -> None:
    for kwargs in ({"starter_grant_hmt": -1.0}, {"starter_grants_per_ip_24h": -1}, {"trusted_proxies": ["not-an-ip"]}):
        with pytest.raises(ValueError):
            _build_world(**kwargs)
