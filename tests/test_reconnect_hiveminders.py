# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HiveMinder fleet reconnection, admin heartbeats, and replace_endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


def _make_app_and_store() -> tuple[TestClient, RouterStore, str]:
    secret = b"hivemind-test-secret-at-least-32-bytes-long!"
    store = RouterStore(":memory:", secret)
    admin_key = "admin-secret-key-for-hivemaster"
    admin_hash = store.key_digest(admin_key)
    app = create_router_app(
        store,
        allow_private_producers=True,
        admin_key_hash=admin_hash,
        resolver=lambda _host: ["34.48.48.154"],
    )
    return TestClient(app), store, admin_key


def test_admin_heartbeat_keeps_user_api_key_intact() -> None:
    client, store, admin_key = _make_app_and_store()
    account_id, user_key = store.register("pioneer-miner-01", "prosumer")
    store.grant(account_id, 500.0, "init")

    # HiveMaster reconnects pioneer-miner-01 via x-admin-key
    resp = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"x-admin-key": admin_key},
        json={
            "handle": "pioneer-miner-01",
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}],
            "cost_per_1k": 0.35,
            "capacity_1k": 80.0,
            "execution_class": "t4-16gb-moe",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 4841}, {"id": "bee-stars", "local_name": "hivemind-bee-stars", "loaded": True, "vram_mib": 4841}],
            "replace_endpoints": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["account_id"] == account_id
    assert body["handle"] == "pioneer-miner-01"
    assert body["status"] == "live"

    # User's original API key still authenticates without rotation
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {user_key}"})
    assert me.status_code == 200
    assert me.json()["account_id"] == account_id


def test_non_admin_cannot_heartbeat_another_account() -> None:
    client, store, _ = _make_app_and_store()
    _id1, key1 = store.register("miner-a", "producer")
    id2, _key2 = store.register("miner-b", "producer")

    resp = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {key1}"},
        json={
            "account_id": id2,
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}],
            "cost_per_1k": 0.35,
            "capacity_1k": 80.0,
        },
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "heartbeat_forbidden"


def test_admin_reconnect_all_hiveminders_and_replace_stale_endpoints() -> None:
    client, store, admin_key = _make_app_and_store()
    aid1, key1 = store.register("tibrib", "producer")
    aid2, _key2 = store.register("cybervalkyrie-fr-5ecb", "prosumer")

    # Seed a stale dead tunnel endpoint on tibrib and pause cybervalkyrie
    store.upsert_producer(
        aid1,
        "https://dead-tunnel.trycloudflare.com",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        0.4,
        60.0,
        "rtx4070",
    )
    store.set_producer_control(aid2, paused=True, vram_cap_mib=None)

    resp = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "bee_endpoint_url": "https://34-48-48-154.sslip.io/bee",
            "coder_endpoint_url": "https://34-48-48-154.sslip.io",
            "rtx2070_endpoint_url": "https://34-48-48-154.sslip.io/rtx2070",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 4841}, {"id": "bee-stars", "local_name": "hivemind-bee-stars", "loaded": True, "vram_mib": 4841}],
            "replace_endpoints": True,
            "include_legacy_known_users": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["live_hiveminders"] == data["total_hiveminders"]
    assert data["live_hiveminders"] >= 12
    assert "coder" in data["live_models"]
    assert "bee-stars" in data["live_models"]

    # Verify tibrib's dead tunnel was replaced with the live endpoint and its API key still works
    fleet = client.get("/v1/hivemind/admin/producers", headers={"x-admin-key": admin_key}).json()
    by_handle = {r["handle"]: r for r in fleet["hiveminders"]}
    assert by_handle["tibrib"]["status"] == "live"
    assert by_handle["tibrib"]["total_endpoints"] == 1
    assert by_handle["tibrib"]["endpoint_url"] == "https://34-48-48-154.sslip.io/u"
    assert by_handle["cybervalkyrie-fr-5ecb"]["status"] == "live"

    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key1}"})
    assert me.status_code == 200

    pub = client.get("/v1/hivemind/producers/fleet")
    assert pub.status_code == 200
    assert pub.json()["live_hiveminders"] == data["live_hiveminders"]


def test_reconnect_without_any_endpoint_token_is_refused_and_no_token_is_built_in() -> None:
    from hivemind.router.gateway import ReconnectProducersRequest

    assert ReconnectProducersRequest.model_fields["endpoint_token"].default is None
    client, store, admin_key = _make_app_and_store()
    store.register("fresh-miner", "producer")
    resp = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={"endpoint_url": "https://34-48-48-154.sslip.io/u", "include_legacy_known_users": False},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "endpoint_token_required"


def test_reconnect_without_a_model_list_is_refused() -> None:
    """The handler must not fill in a model table on the caller's behalf."""
    client, store, admin_key = _make_app_and_store()
    store.register("list-less-miner", "producer")
    resp = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "include_legacy_known_users": False,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "models_required"


def test_keepalive_refresh_after_a_producer_heartbeat_keeps_the_account_id_and_the_new_endpoint() -> None:
    """Regression (2026-11-02): a producer's own heartbeat rewrote its keepalive entry without "account_id", and the next
    35 s refresh read item["account_id"] and raised KeyError on every route that refreshes the fleet."""
    secret = b"hivemind-test-secret-at-least-32-bytes-long!"
    store = RouterStore(":memory:", secret)
    admin_key = "admin-secret-key-for-hivemaster"
    now = [1_800_000_000.0]
    client = TestClient(
        create_router_app(
            store,
            allow_private_producers=True,
            admin_key_hash=store.key_digest(admin_key),
            resolver=lambda _h: ["34.48.48.154"],
            clock=lambda: now[0],
        )
    )
    _aid, key = store.register("tibrib", "producer")
    models = [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}]
    resp = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": models,
            "replace_endpoints": True,
            "include_legacy_known_users": False,
        },
    )
    assert resp.status_code == 200

    own = "https://34-48-48-154.sslip.io/own"
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "endpoint_url": own,
            "models": models,
            "cost_per_1k": 0.3,
            "capacity_1k": 50.0,
            "execution_class": "t4-16gb-moe",
            "endpoint_token": "own-endpoint-token-0123456789abcdefghij",
            "replace_endpoints": True,
        },
    )
    assert hb.status_code == 200

    now[0] += 40.0  # past the 35 s refresh interval
    assert client.get("/v1/hivemind/producers/fleet").status_code == 200
    rows = {r["handle"]: r for r in client.get("/v1/hivemind/admin/producers", headers={"x-admin-key": admin_key}).json()["hiveminders"]}
    assert rows["tibrib"]["endpoint_url"] == own
    assert rows["tibrib"]["status"] == "live"
