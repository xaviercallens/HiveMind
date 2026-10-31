# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for scripts/run_2h_roadmap_pr_and_night_moe_cron.py."""

from __future__ import annotations

from pathlib import Path

from scripts.run_2h_roadmap_pr_and_night_moe_cron import (
    probe_local_8095_producer_server,
    run_night_moe_training_cycle,
    scan_roadmap_status,
)


def test_scan_roadmap_status_detects_season_33() -> None:
    res = scan_roadmap_status()
    assert res["exists"] is True
    assert res["max_season"] >= 33
    assert 25 in res["seasons_detected"]
    assert 33 in res["seasons_detected"]
    assert res["total_task_ids"] >= 190


def test_local_cockpit_5lang_portal_and_evolution_cron_endpoint() -> None:
    from fastapi.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    client = TestClient(create_exoplanet_hunter_fastapi_app())
    for route, expected_lang in (("/portal?lang=es", "es"), ("/de", "de"), ("/zh", "zh"), ("/inscription", "fr")):
        resp = client.get(route)
        assert resp.status_code == 200
        assert resp.headers.get("content-language") == expected_lang

    i18n_resp = client.get("/v1/hivemind/i18n?lang=de")
    assert i18n_resp.status_code == 200
    assert i18n_resp.json()["negotiated_language"] == "de"

    cron_resp = client.get("/api/v1/monitor/evolution-cron")
    assert cron_resp.status_code == 200
    cron_body = cron_resp.json()
    assert cron_body["object"] == "hivemind.evolution_cron_telemetry"
    assert cron_body["schedules"]["roadmap_pr_evolution_2h"] == "0 */2 * * *"


def test_probe_local_8095_producer_server_returns_dict() -> None:
    res = probe_local_8095_producer_server()
    assert isinstance(res.get("online"), bool)
    assert "pool_status" in res


def test_run_night_moe_training_cycle_enforces_vram_guards(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_2h_roadmap_pr_and_night_moe_cron as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    summary = run_night_moe_training_cycle(
        {
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "memory_used_mib": 1458.0,
            "memory_free_mib": 6549.0,
            "effective_sparse_moe_nf4_tflops": 65.24,
        }
    )
    assert summary["status"] == "ACTIVE_OVERNIGHT_EXECUTION"
    assert summary["rtx2070_live_guard"]["wddm_ceiling_6656_mib_ok"] is True
    assert summary["rtx2070_live_guard"]["headroom_2048_mib_ok"] is True
    assert summary["nocturnal_workunits_executed"] == 5
    assert (tmp_path / "artifacts" / "expert_registry_s21.json").exists()
