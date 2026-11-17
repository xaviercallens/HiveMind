"""Season 8 Track 6 (`HM-S8-10`): live `/api/diloco-nextgen` Command HUD telemetry."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.game.steam_edition import STEAM_APP_ID, create_hack_miner_fastapi_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_hack_miner_fastapi_app(worker_token="test-worker-token-not-a-secret"), raise_server_exceptions=False)


def test_diloco_nextgen_endpoint_exposes_season8_telemetry(client: TestClient) -> None:
    response = client.get("/api/diloco-nextgen")
    assert response.status_code == 200
    data = response.json()

    assert data["schema_version"] == "hm-s8-10/diloco-nextgen/v1"
    assert data["milestone"] == "HM-S8-10"
    assert data["season"] == 8
    assert data["steam_app_id"] == STEAM_APP_ID
    assert data["telemetry_endpoint"] == "/api/diloco-nextgen"
    assert data["generated_at"] > 0.0

    autoloco = data["autoloco"]
    assert autoloco["rho_t"] > 0.0
    assert autoloco["recommended_inner_steps_h"] >= 1
    assert autoloco["reference_inner_steps_h"] == 32
    assert autoloco["corrected_outer_lr"] > 0.0
    assert 0.0 < autoloco["corrected_outer_momentum"] < 1.0

    heloco = data["heloco"]
    assert heloco["heterogeneity_drift"] >= 0.0
    assert -1.0 <= heloco["cosine_before"] <= 1.0
    assert -1.0 <= heloco["cosine_after"] <= 1.0
    assert isinstance(heloco["conflict_detected"], bool)

    car = data["skip_attention_car"]
    assert 0.0 <= car["car_hit_rate"] <= 1.0
    assert 0.0 <= car["speculative_topk_accuracy"] <= 1.0
    assert car["car_substitutions"] >= 0
    assert car["synchronous_dma_misses"] >= 0

    fomoe = data["fomoe"]
    assert fomoe["total_experts"] == 128
    assert fomoe["active_experts_per_worker"] + fomoe["ghost_experts_per_worker"] == 128
    assert fomoe["active_experts_per_worker"] > 0
    assert fomoe["ghost_experts_per_worker"] > 0

    repositories = data["leveraged_github_repositories"]
    assert len(repositories) == 8
    names = {entry["name"] for entry in repositories}
    assert {"SparseLoCo", "DeMo", "fomoe", "isoloco"}.issubset(names)
    for entry in repositories:
        assert entry["repo_url"].startswith("https://github.com/")
        assert entry["authors"]

    assert data["honesty"]["modeled"]
    assert data["honesty"]["derived_from_live_state"]


def test_diloco_nextgen_telemetry_is_deterministic_and_tracks_strategy_updates(
    client: TestClient,
) -> None:
    first = client.get("/api/diloco-nextgen").json()
    second = client.get("/api/diloco-nextgen").json()

    # The strategy snapshot and seeded probes are stable between requests at the same revision.
    assert first["autoloco"] == second["autoloco"]
    assert first["heloco"] == second["heloco"]
    assert first["skip_attention_car"] == second["skip_attention_car"]
    assert first["fomoe"] == second["fomoe"]

    before_sequence = first["sequence"]
    updated = client.post(
        "/api/strategy-optimizer",
        headers={"Authorization": "Bearer test-worker-token-not-a-secret"},
        json={
            "ring_rtt_ms": 250.0,
            "bbr_pacing_mbps": 8.0,
            "packet_loss_rate": 0.05,
            "turn_relay_active": True,
        },
    )
    assert updated.status_code == 200
    cadence = updated.json()["si03_adaptive_diloco_cadence"]["current_link_profile"]
    expected_steps = int(cadence["recommended_inner_steps_H"])

    after = client.get("/api/diloco-nextgen").json()
    assert after["sequence"] == before_sequence + 1
    assert after["autoloco"]["recommended_inner_steps_h"] == expected_steps
    assert after["autoloco"]["rho_t"] == pytest.approx(round(expected_steps / 32.0, 6))
