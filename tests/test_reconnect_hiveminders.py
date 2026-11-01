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
            "endpoint_token": "sDkyBiiDsjX3KL48lUZxdfZu1LK0c8QZXoS7RnJ7D2Q",
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
            "endpoint_token": "sDkyBiiDsjX3KL48lUZxdfZu1LK0c8QZXoS7RnJ7D2Q",
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
