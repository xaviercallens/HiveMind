# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Security regression suite for PR #2:
- Blocks unauthenticated account takeover on /v1/hivemind/steam/register-login & /v1/hivemind/webgpu/register-login
- Enforces receipt_sha256 deduplication on BOINC/WebGPU contribution endpoints
- Verifies /v1/hivemind/t4/telemetry authentication and live snapshot update
- Verifies Ruche overlay /api/v1/community/boinc-sync clamping, receipt deduplication, and optional sync key
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


def _make_client(tmp_path: Path, admin_key: str = "adm-secret-test-key") -> TestClient:
    store = RouterStore(tmp_path / "router.sqlite3", secret=b"test-hmac-secret-32-bytes-long!!")
    app = create_router_app(store, admin_key_hash=store.key_digest(admin_key))
    return TestClient(app)


def test_steam_and_webgpu_register_login_block_unauthenticated_account_takeover(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Victim registers a standard account via /v1/hivemind/register
    reg = client.post("/v1/hivemind/register", json={"handle": "victim_producer", "role": "prosumer"})
    assert reg.status_code == 200
    victim_key = reg.json()["api_key"]
    victim_id = reg.json()["account_id"]

    # 2. Attacker attempts to take over victim_producer via /v1/hivemind/steam/register-login without API key
    hijack_steam = client.post("/v1/hivemind/steam/register-login", json={"handle": "victim_producer"})
    assert hijack_steam.status_code == 409
    assert hijack_steam.json()["error"]["code"] == "handle_taken"

    # 3. Attacker attempts to take over victim_producer via /v1/hivemind/webgpu/register-login without API key
    hijack_webgpu = client.post("/v1/hivemind/webgpu/register-login", json={"handle": "victim_producer"})
    assert hijack_webgpu.status_code == 409
    assert hijack_webgpu.json()["error"]["code"] == "handle_taken"

    # 4. Victim's original API key remains valid and unrotated
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {victim_key}"})
    assert me.status_code == 200
    assert me.json()["account_id"] == victim_id

    # 5. Victim can legitimately sign in to Steam/WebGPU by presenting their Bearer API key
    legit_steam = client.post(
        "/v1/hivemind/steam/register-login",
        json={"handle": "victim_producer"},
        headers={"Authorization": f"Bearer {victim_key}"},
    )
    assert legit_steam.status_code == 200
    assert legit_steam.json()["api_key"] == victim_key
    assert legit_steam.json()["created"] is False


def test_boinc_and_webgpu_contribute_deduplicate_replayed_receipts(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    reg = client.post("/v1/hivemind/webgpu/register-login", json={"handle": "Cmdr_Dedup_Test"})
    assert reg.status_code == 200
    api_key = reg.json()["api_key"]
    headers = {"Authorization": f"Bearer {api_key}"}

    # First submission with explicit receipt_sha256 awards 250 HMT
    r1 = client.post(
        "/v1/hivemind/webgpu/contribute",
        json={"target_planet": "K2-18 b", "receipt_sha256": "a" * 64},
        headers=headers,
    )
    assert r1.status_code == 200
    assert r1.json()["hmt_awarded"] == 250.0
    assert r1.json()["duplicate_receipt"] is False
    bal_after_first = r1.json()["balance_hmt"]

    # Replay with identical receipt_sha256 awards 0 HMT
    r2 = client.post(
        "/v1/hivemind/webgpu/contribute",
        json={"target_planet": "K2-18 b", "receipt_sha256": "a" * 64},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json()["hmt_awarded"] == 0.0
    assert r2.json()["duplicate_receipt"] is True
    assert r2.json()["balance_hmt"] == bal_after_first


def test_t4_telemetry_endpoint_requires_auth_and_updates_status(tmp_path: Path) -> None:
    admin_key = "adm-t4-sync-secret"
    client = _make_client(tmp_path, admin_key=admin_key)

    # Unauthenticated call is rejected with 403
    unauth = client.post("/v1/hivemind/t4/telemetry", json={"gpu": {"used_mib": 5120}})
    assert unauth.status_code == 403

    # Authenticated via Authorization: Bearer <admin_key> (as used by t4_cloudrun_sync.py) succeeds
    auth_res = client.post(
        "/v1/hivemind/t4/telemetry",
        json={"gpu": {"used_mib": 4912, "total_mib": 15360}, "governor": {"yields_count": 142}},
        headers={"Authorization": f"Bearer {admin_key}"},
    )
    assert auth_res.status_code == 200
    assert auth_res.json()["ok"] is True

    status = client.get("/v1/hivemind/t4/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["hardware"]["vram_used_mib"] == 4912
    assert payload["hardware"]["vram_free_mib"] == 15360 - 4912
    assert payload["governor"]["yields_count"] == 142


def test_ruche_overlay_boinc_sync_clamping_and_dedup(tmp_path: Path, monkeypatch) -> None:
    overlay_path = Path(__file__).resolve().parents[1] / "deploy" / "ruche_router_overlay" / "server.py"
    spec = importlib.util.spec_from_file_location("ruche_overlay_server_test", overlay_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("HIVEMIND_COMMUNITY_STATE_PATH", str(tmp_path / "community_state.json"))
    monkeypatch.setenv("HIVEMIND_BOINC_SYNC_KEY", "sync-secret-123")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    overlay_client = TestClient(mod.app)

    # Missing sync key when HIVEMIND_BOINC_SYNC_KEY is configured -> 401
    denied = overlay_client.post(
        "/api/v1/community/boinc-sync",
        json={"handle": "Cmdr_Overlay", "cobblestones_delta": 50.0, "hmt_delta": 100},
    )
    assert denied.status_code == 401

    # Valid sync key with exaggerated deltas clamps to <= 100.0 Cobblestones and <= 250 HMT
    ok = overlay_client.post(
        "/api/v1/community/boinc-sync",
        json={
            "handle": "Cmdr_Overlay",
            "cobblestones_delta": 999999.0,
            "hmt_delta": 999999,
            "receipt_sha256": "b" * 64,
        },
        headers={"X-Boinc-Sync-Key": "sync-secret-123"},
    )
    assert ok.status_code == 200
    data = ok.json()
    assert data["duplicate_receipt"] is False
    assert data["cobblestones_earned"] == 100.0
    assert data["hmt_points_earned"] == 250

    # Replaying the same receipt_sha256 does not increment cobblestones or HMT
    dup = overlay_client.post(
        "/api/v1/community/boinc-sync",
        json={
            "handle": "Cmdr_Overlay",
            "cobblestones_delta": 50.0,
            "hmt_delta": 100,
            "receipt_sha256": "b" * 64,
        },
        headers={"X-Boinc-Sync-Key": "sync-secret-123"},
    )
    assert dup.status_code == 200
    dup_data = dup.json()
    assert dup_data["duplicate_receipt"] is True
    assert dup_data["cobblestones_earned"] == 0.0
    assert dup_data["hmt_points_earned"] == 0
