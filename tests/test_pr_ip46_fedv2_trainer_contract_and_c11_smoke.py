# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 46 tests: Audit C11 CI E2E smoke test & Federation v2 Section 9 trainer registration contract."""

from __future__ import annotations

import base64
import json
import secrets
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mvp_e2e_smoke as smoke  # noqa: E402
from hivemind.router import RouterStore, create_router_app  # noqa: E402
from hivemind.router.opentrainer import (  # noqa: E402
    NETWORK_MODES,
    validate_federation_v2_trainer_registration,
)


def _valid_wg_pubkey() -> str:
    raw = bytes(range(1, 33))
    return base64.b64encode(raw).decode("ascii")


def test_federation_v2_trainer_registration_positive_and_negative_controls() -> None:
    """Section 9 of docs/FEDERATION_V2_AMCP_AGLETS.md: positive manifest and negative controls."""
    pubkey = _valid_wg_pubkey()
    manifest = validate_federation_v2_trainer_registration(
        learner_id="trainer_rtx2070_01",
        gpu_model="NVIDIA GeForce RTX 2070",
        vram_free_mib=6400,
        vram_total_mib=8192,
        night_only_consent=True,
        shard_index=1,
        run_id="night-001",
        pinned_commit="6089aee",
        network_mode="wireguard-t4-hub",
        wireguard_public_key=pubkey,
        overlay_slot_ip="10.77.0.8/32",
    )
    assert manifest["accepted"] is True
    assert manifest["learner_id"] == "trainer_rtx2070_01"
    assert manifest["shard_index"] == 1
    assert manifest["run_id"] == "night-001"
    assert manifest["pinned_commit"] == "6089aee"
    assert manifest["vram_floor_required_mib"] == 1536
    assert manifest["night_only_consent"] is True
    assert manifest["points_or_payment_promised"] is False
    assert manifest["shared_signing_key_included"] is False
    assert manifest["auto_starts_training"] is False
    assert f"hm-wg-hub add-member 10.77.0.8/32 {pubkey}" == manifest["operator_wg_hub_command"]
    assert "--learner-id trainer_rtx2070_01" in manifest["cli_join_command"]
    assert "--night-only" in manifest["cli_join_command"]

    # Negative control 1: WireGuard private key must be rejected
    with pytest.raises(ValueError, match="wireguard_private_key"):
        validate_federation_v2_trainer_registration(
            learner_id="trainer_rtx2070_01",
            gpu_model="NVIDIA GeForce RTX 2070",
            vram_free_mib=6400,
            night_only_consent=True,
            wireguard_private_key=pubkey,
        )

    # Negative control 2: Shared HMAC signing key must be rejected
    with pytest.raises(ValueError, match="HIVEMIND_AMCP_HMAC_KEYS"):
        validate_federation_v2_trainer_registration(
            learner_id="trainer_rtx2070_01",
            gpu_model="NVIDIA GeForce RTX 2070",
            vram_free_mib=6400,
            night_only_consent=True,
            amcp_hmac_keys="fed1:" + "a" * 64,
        )

    # Negative control 3: Missing night_only_consent must be rejected
    with pytest.raises(ValueError, match="night_only_consent"):
        validate_federation_v2_trainer_registration(
            learner_id="trainer_rtx2070_01",
            gpu_model="NVIDIA GeForce RTX 2070",
            vram_free_mib=6400,
            night_only_consent=False,
        )

    # Negative control 4: Free VRAM below T4 4,300 MiB floor must be rejected
    with pytest.raises(ValueError, match="safety floor"):
        validate_federation_v2_trainer_registration(
            learner_id="trainer_t4_01",
            gpu_model="NVIDIA Tesla T4",
            vram_free_mib=4100,
            vram_total_mib=15360,
            night_only_consent=True,
        )

    # Negative control 5: wireguard-t4-hub without wireguard_public_key must be rejected
    with pytest.raises(ValueError, match="wireguard_public_key"):
        validate_federation_v2_trainer_registration(
            learner_id="trainer_rtx2070_01",
            gpu_model="NVIDIA GeForce RTX 2070",
            vram_free_mib=6400,
            night_only_consent=True,
            network_mode="wireguard-t4-hub",
        )


def test_gateway_register_trainer_and_all_network_modes() -> None:
    """Gateway /v1/hivemind/opentrainer/register-trainer and heartbeat accept all 8 NETWORK_MODES and enforce §9."""
    store = RouterStore(":memory:", secrets.token_bytes(32))
    client = TestClient(create_router_app(store, allow_private_producers=True, starter_grant_hmt=0.0))

    trainer = client.post("/v1/hivemind/register", json={"handle": "fedv2-trainer", "role": "trainer"}).json()
    consumer = client.post("/v1/hivemind/register", json={"handle": "plain-consumer", "role": "consumer"}).json()
    headers = {"Authorization": f"Bearer {trainer['api_key']}"}

    # All 8 NETWORK_MODES from opentrainer.py are accepted by heartbeat
    for mode in NETWORK_MODES:
        hb = client.post(
            "/v1/hivemind/opentrainer/heartbeat",
            headers=headers,
            json={
                "host_label": "node-primary",
                "gpu_name": "NVIDIA GeForce RTX 3060",
                "vram_total_mib": 12288,
                "vram_free_mib": 9000,
                "network_mode": mode,
            },
        )
        assert hb.status_code == 200, (mode, hb.text)

    # Positive registration via /v1/hivemind/opentrainer/register-trainer
    reg = client.post(
        "/v1/hivemind/opentrainer/register-trainer",
        headers=headers,
        json={
            "learner_id": "learner_02",
            "gpu_model": "NVIDIA GeForce RTX 3060",
            "vram_free_mib": 9000,
            "vram_total_mib": 12288,
            "night_only_consent": True,
            "shard_index": 2,
            "run_id": "night-002",
            "network_mode": "headscale-tailscale",
        },
    )
    assert reg.status_code == 200, reg.text
    body = reg.json()
    assert body["accepted"] is True
    assert body["points_or_payment_promised"] is False
    assert body["shared_signing_key_included"] is False
    assert body["node_id"] == "fedv2-trainer:learner_02"

    # Negative control: submitting a WireGuard private key returns HTTP 400
    bad_priv = client.post(
        "/v1/hivemind/opentrainer/register-trainer",
        headers=headers,
        json={
            "learner_id": "learner_02",
            "gpu_model": "NVIDIA GeForce RTX 3060",
            "vram_free_mib": 9000,
            "night_only_consent": True,
            "wireguard_private_key": _valid_wg_pubkey(),
        },
    )
    assert bad_priv.status_code == 400
    assert "wireguard_private_key" in bad_priv.text

    # Negative control: consumer account is refused with HTTP 403
    forbidden = client.post(
        "/v1/hivemind/opentrainer/register-trainer",
        headers={"Authorization": f"Bearer {consumer['api_key']}"},
        json={
            "learner_id": "learner_03",
            "gpu_model": "NVIDIA GeForce RTX 3060",
            "vram_free_mib": 9000,
            "night_only_consent": True,
        },
    )
    assert forbidden.status_code == 403


def test_mvp_e2e_smoke_negative_control_missing_gguf(tmp_path: Path) -> None:
    """Negative control: missing GGUF file fails model_server_started and exits with code 1."""
    report = tmp_path / "missing_gguf_report.json"
    code = smoke.main(["--gguf", str(tmp_path / "nonexistent.gguf"), "--report", str(report)])
    assert code == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert data["checks"]["model_server_started"] is False
