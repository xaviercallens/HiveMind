# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""La Ruche dashboard: minting is refused, unperformed actions say so, and missing campaign flags are not passes."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import build_hivemind_at_home_state, create_hivemind_at_home_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))


def test_two_guest_upgrade_cycles_mint_zero_points(tmp_path: Path) -> None:
    client = _client(tmp_path)
    minted = 0
    for i in range(2):
        guest = client.post("/api/v1/auth/guest", json={}).json()
        minted += int(guest["compute_points"])
        up = client.post(
            "/api/v1/auth/upgrade-guest",
            headers={"Authorization": f"Bearer {guest['token']}"},
            json={"handle": f"nomint-{i}", "email": f"n{i}@example.org", "provider": "google"},
        )
        assert up.status_code == 200
        minted += int(up.json()["compute_points"])
    assert minted == 0, f"guest/upgrade cycles minted {minted} points"


def test_upgrade_role_is_fixed_server_side(tmp_path: Path) -> None:
    client = _client(tmp_path)
    guest = client.post("/api/v1/auth/guest", json={}).json()["token"]
    up = client.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {guest}"},
        json={"handle": "roleprobe01", "email": "r@example.org", "role": "admin:all"},
    )
    assert up.status_code == 200
    assert up.json()["role"] != "admin:all", up.json()["role"]
    assert "admin:all" not in up.text


def test_night_moe_start_does_not_claim_dispatch(tmp_path: Path) -> None:
    body = _client(tmp_path).post("/api/v1/night-moe/start", json={}).json()
    assert body["status"] == "NOT_PERFORMED", body
    assert body["measured"] is False
    assert body["job_id"] is None
    assert "NIGHT_TRAINING_DISPATCHED" not in json.dumps(body)


def test_shield_lock_without_cuda_reports_no_cuda_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    body = _client(tmp_path).post("/api/v1/hardware/rtx2070/shield/lock").json()
    assert body["status"] == "NOT_PERFORMED", body
    assert body["device"] is None, body["device"]
    assert body["training_preemption_lock"] is False
    assert "cuda:0" not in json.dumps(body)


def test_astro_discover_unknown_target_is_refused_and_unawarded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    r = client.post("/api/astro/discover", json={"target_id": "no-such-target-xyz", "player_classification": "PLANET"})
    assert 400 <= r.status_code < 500, r.status_code
    assert "hmt_awarded" not in r.text


def test_astro_discover_known_target_awards_nothing_unperformed(tmp_path: Path) -> None:
    body = _client(tmp_path).post("/api/astro/discover", json={"target_id": "k218b", "player_classification": "PLANET"}).json()
    assert body["status"] == "NOT_PERFORMED", body
    assert body["measured"] is False
    assert body["hmt_awarded"] == 0
    assert body["receipt_sha256"] is None


def test_missing_campaign_keys_are_not_measured(tmp_path: Path) -> None:
    report = tmp_path / "gpu.json"
    report.write_text(
        json.dumps(
            {
                "campaign_1_s9_quorum_and_rda": {},
                "campaign_2_s9_sigma_and_looped_moe": {},
                "campaign_3_s10_lattice_wire_and_kv": {},
                "campaign_4_s10_spotcheck_leasing_and_gossip": {},
            }
        ),
        encoding="utf-8",
    )
    state = build_hivemind_at_home_state(gpu_campaigns_report_path=report)
    campaigns = state["season9_10_gpu_campaigns"]["campaigns"]
    assert len(campaigns) == 4
    assert [c["passed"] for c in campaigns] == [None, None, None, None]
    assert state["season9_10_gpu_campaigns"]["all_campaigns_passed"] is None
    metrics = [v for c in campaigns for v in c["metrics"].values()]
    assert 1.86 not in metrics and 0.42 not in metrics and 9216 not in metrics
