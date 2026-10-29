# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Comprehensive coverage tests for game cockpit, fat builder, engine, and router producer agent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from hivemind.game import exoplanet_hunter_cockpit as cockpit_mod
from hivemind.game import exoplanet_hunter_engine as engine_mod
from hivemind.game import fat_steam_astro_builder as builder_mod
from hivemind.router import hivemind_producer_agent as agent_mod


def test_router_producer_agent_full_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover src/hivemind/router/hivemind_producer_agent.py functions and CLI branches."""
    assert agent_mod.parse_model_spec("coder=qwen2.5-coder:7b") == {
        "id": "coder",
        "local_name": "qwen2.5-coder:7b",
    }
    with pytest.raises(Exception):
        agent_mod.parse_model_spec("invalid_spec")

    payload = agent_mod.build_heartbeat_payload(
        endpoint="https://node.example.com",
        models=[{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        cost_per_1k=0.4,
        capacity_1k=60.0,
        execution_class="gpu",
        endpoint_token="tok-123",
    )
    assert payload["endpoint_token"] == "tok-123"

    # detect_vram_gb success & failure branches
    def fake_run_ok(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, stdout="8192\ninvalid\n16384\n", stderr="")

    def fake_run_fail(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="err")

    assert agent_mod.detect_vram_gb(fake_run_ok) == 16.0
    assert agent_mod.detect_vram_gb(fake_run_fail) is None

    tiers = [
        {"tier_id": "coder-lite", "min_vram_gb": 4.0, "ollama_tag": "qwen2.5-coder:3b", "suggested_cost_per_1k": 0.2, "suggested_capacity_1k": 40.0},
        {"tier_id": "coder", "min_vram_gb": 6.0, "ollama_tag": "qwen2.5-coder:7b", "suggested_cost_per_1k": 0.4, "suggested_capacity_1k": 60.0},
    ]
    assert agent_mod.recommend(tiers, 8.0)["tier_id"] == "coder"
    assert agent_mod.recommend(tiers, 2.0) is None

    state = {"node_state": "active", "min_price_per_1k": 0.55}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/hivemind/catalog":
            return httpx.Response(200, json={"tiers": tiers})
        if path == "/v1/hivemind/remote/config":
            return httpx.Response(200, json={"remote_config": state})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen2.5-coder:7b"}]})
        if path == "/v1/hivemind/producer/heartbeat":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    # 1. --suggest with --vram-gb
    rc1 = agent_mod.main(
        ["--gateway", "https://gw.example.com", "--suggest", "--vram-gb", "8"],
        transport=transport,
    )
    assert rc1 == 0

    # 2. --suggest with --detect-vram and tiny vram
    rc2 = agent_mod.main(
        ["--gateway", "https://gw.example.com", "--suggest", "--vram-gb", "1"],
        transport=transport,
    )
    assert rc2 == 0

    rc3 = agent_mod.main(
        ["--gateway", "https://gw.example.com", "--suggest", "--detect-vram"],
        transport=transport,
        runner=fake_run_ok,
    )
    assert rc3 == 0

    # 3. --once with --preset and --sync-remote-config
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-key")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "ep-tok")
    rc4 = agent_mod.main(
        [
            "--gateway",
            "https://gw.example.com",
            "--endpoint",
            "https://node.example.com",
            "--preset",
            "coder",
            "--once",
            "--sync-remote-config",
        ],
        transport=transport,
    )
    assert rc4 == 0

    # 4. --once when remote config is paused
    state["node_state"] = "paused"
    rc5 = agent_mod.main(
        [
            "--gateway",
            "https://gw.example.com",
            "--endpoint",
            "https://node.example.com",
            "--preset",
            "coder",
            "--once",
            "--sync-remote-config",
        ],
        transport=transport,
    )
    assert rc5 == 0

    # 5. Missing model in preflight returns 1
    rc6 = agent_mod.main(
        [
            "--gateway",
            "https://gw.example.com",
            "--endpoint",
            "https://node.example.com",
            "--model",
            "coder=missing-model:7b",
            "--cost-per-1k",
            "0.4",
            "--capacity-1k",
            "60",
            "--once",
        ],
        transport=transport,
    )
    assert rc6 == 1


def test_exoplanet_engine_helpers_and_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover src/hivemind/game/exoplanet_hunter_engine.py TTS, NASA TAP, and pool methods."""
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestFakeKey1234567890")
    assert engine_mod._load_gemini_api_key_from_env() == "AIzaSyTestFakeKey1234567890"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    wav = engine_mod._wrap_pcm16_to_wav(b"\x00\x01" * 100, sample_rate=24000)
    assert wav[:4] == b"RIFF" and len(wav) == 44 + 200

    chunks = engine_mod._split_text_for_gcp_tts(
        "Sentence one is short. Sentence two is also short! " + (" word" * 60),
        max_chars=80,
    )
    assert len(chunks) >= 3

    # Cached WAV and MP3 paths in synthesize_gemini_or_gcp_free_tts
    cache_dir = tmp_path / "voice_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text = "Hello K2-18 b explorer"
    import hashlib
    key_hash = hashlib.sha256(f"en|Puck|{text}".encode("utf-8")).hexdigest()[:16]
    mp3_cache = cache_dir / f"gcp_en_{key_hash}.mp3"
    mp3_cache.write_bytes(b"ID3" + b"\x00" * 250)
    res_mp3 = engine_mod.synthesize_gemini_or_gcp_free_tts(text, lang="en", voice_name="Puck", cache_dir=cache_dir)
    assert res_mp3["ok"] is True and res_mp3["cached"] is True

    wav_cache = cache_dir / f"gemini_en_puck_{key_hash}.wav"
    wav_cache.write_bytes(wav + b"\x00" * 100)
    res_wav = engine_mod.synthesize_gemini_or_gcp_free_tts(text, lang="en", voice_name="Puck", cache_dir=cache_dir)
    assert res_wav["ok"] is True and res_wav["mime_type"] == "audio/wav"

    # NASA TAP fallback and cache
    tap_fallback = engine_mod.query_nasa_exoplanet_archive_tap(live_network=False)
    assert tap_fallback["row_count"] >= 8
    tap_cache_file = tmp_path / "nasa_tap_pscomppars_cache.json"
    tap_cache_file.write_text(json.dumps([{"pl_name": "K2-18 b"}]), encoding="utf-8")
    tap_cached = engine_mod.query_nasa_exoplanet_archive_tap(live_network=False)
    assert tap_cached["source"] == "CACHED_NASA_TAP_ADQL"

    # Service and multi-producer pool + parquet enrichment path
    fake_pq = tmp_path / "train-00000-of-00001.parquet"
    fake_pq.write_bytes(b"PAR1")

    class _FakeTable:
        def to_pylist(self) -> list[dict[str, object]]:
            return [
                {
                    "id": 101,
                    "period_days": 12.5,
                    "planet_radius_rearth": 1.8,
                    "transit_depth_ppm": 950.0,
                    "model_snr": 18.0,
                    "stellar_teff_k": 5400.0,
                    "stellar_radius_rsun": 0.92,
                    "duration_hours": 3.4,
                    "insolation_earth": 1.1,
                    "target": "CANDIDATE",
                },
                {
                    "id": 102,
                    "period_days": 4.2,
                    "planet_radius_rearth": 9.5,
                    "transit_depth_ppm": 18000.0,
                    "model_snr": 45.0,
                    "stellar_teff_k": 6100.0,
                    "stellar_radius_rsun": 1.15,
                    "duration_hours": 2.8,
                    "insolation_earth": 85.0,
                    "target": "FALSE POSITIVE",
                },
                {
                    "id": 103,
                    "period_days": 28.0,
                    "planet_radius_rearth": 2.4,
                    "transit_depth_ppm": 1200.0,
                    "model_snr": 22.0,
                    "stellar_teff_k": 4900.0,
                    "stellar_radius_rsun": 0.81,
                    "duration_hours": 4.1,
                    "insolation_earth": 0.75,
                    "target": "CONFIRMED",
                },
                {"id": 104, "period_days": None, "target": "CANDIDATE"},
            ]

    with patch.object(engine_mod, "_find_cached_parquet", return_value=fake_pq):
        with patch.object(engine_mod.pq, "read_table", return_value=_FakeTable()):
            cat_enriched = engine_mod.load_real_hf_exoplanet_catalog(max_extra_per_category=2)
            assert len(cat_enriched) >= 10

    svc = engine_mod.AstroBeeMoEGPUService()
    gen = svc.run_general_hivemind_moe_inference("Test exoplanet transit query", max_new_tokens=16, preferred_producer="t4")
    assert gen["status"] == "ok" and gen["t4_failover_active"] is True

    pool = engine_mod.HiveMindMultiProducerInferencePool(svc)
    reg = pool.register_or_update_producer(
        producer_id="PROD-EXT-1",
        name="External Node",
        hardware="RTX 4090",
        status="ONLINE_SERVING",
        endpoint_url="http://127.0.0.1:9999",
        priority=1,
    )
    assert reg["producer_id"] == "PROD-EXT-1"
    inf1 = pool.serve_inference("Check transit", max_new_tokens=16, preferred_producer="t4")
    assert inf1["status"] == "ok"
    inf2 = pool.serve_inference("Analyze planet", target_id="SECTOR-02-K2-18B")
    assert "producer_routing" in inf2

    pool.record_traffic_event("GET", "/api/v1/monitor/telemetry")
    pool.record_traffic_event("POST", "/v1/chat/completions", user_agent="python-httpx")
    pool.record_traffic_event("POST", "/api/astro/discover", user_agent="curl/8.0", gamer_tag="Tester")
    pool.record_traffic_event("GET", "/", user_agent="UnityPlayer/2023")
    rep = pool.get_live_traffic_and_tflops_report()
    assert rep["status"] == "LIVE_MONITORING_ACTIVE"


def test_fat_steam_astro_builder_assets_and_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover src/hivemind/game/fat_steam_astro_builder.py asset and package builders."""
    monkeypatch.setattr(builder_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder_mod, "STEAM_LIBRARY_COMMON", tmp_path / "SteamLibrary" / "steamapps" / "common")
    monkeypatch.setattr(builder_mod, "STEAM_USERDATA_CONFIG", tmp_path / "Steam" / "userdata" / "1" / "config")
    (tmp_path / "SteamLibrary" / "steamapps" / "common").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Steam" / "userdata" / "1" / "config").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        builder_mod,
        "synthesize_gemini_or_gcp_free_tts",
        lambda **kw: {"ok": True, "cached_path": str(tmp_path / "dummy.mp3")},
    )
    res = builder_mod.build_fat_steam_exoplanet_client(install_to_local_steam_library=True)
    assert res["steam_app_id"] == builder_mod.STEAM_APP_ID
    assert Path(res["bundled_catalog_json"]).exists()
    assert Path(res["boinc_idle_gpu_worker_path"]).exists()


def test_exoplanet_cockpit_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover src/hivemind/game/exoplanet_hunter_cockpit.py FastAPI routes."""
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cockpit_mod,
        "synthesize_gemini_or_gcp_free_tts",
        lambda **kw: {"ok": True, "audio_bytes": b"FAKE_MP3", "mime_type": "audio/mpeg", "engine": "test-tts"},
    )
    monkeypatch.setattr(
        cockpit_mod,
        "query_nasa_exoplanet_archive_tap",
        lambda **kw: {"source": "MOCK_TAP", "row_count": 1, "rows": []},
    )

    stitch_v2 = tmp_path / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "stitch_sync_v2"
    stitch_v2.mkdir(parents=True, exist_ok=True)
    (stitch_v2 / "02_uss_hivemind_ncc-31337_v4_6_-_complete_tactical.html").write_text("<!DOCTYPE html><html>Bridge</html>", encoding="utf-8")
    (stitch_v2 / "04_uss_hivemind_-_fat_client_unity_unreal_vs_web.html").write_text("Plain text architecture doc", encoding="utf-8")
    (stitch_v2 / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    app = cockpit_mod.create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/monitor").status_code == 200
    assert client.get("/game").status_code == 200
    assert client.get("/community").status_code == 200
    assert client.get("/api/v1/monitor/telemetry").status_code == 200
    assert client.get("/api/astro/nasa-tap-sync").status_code == 200
    assert client.get("/api/astro/stitch_project").status_code == 200
    assert client.get("/stitch-assets/test.png").status_code == 200
    assert client.get("/stitch-assets/missing.png").status_code == 404
    assert client.get("/stitch/v46-complete").status_code == 200
    assert client.get("/stitch/architecture").status_code == 200
    assert client.get("/stitch/stage4").status_code == 404
    assert client.get("/api/steam/fat-client-manifest").status_code == 200
    assert client.get("/api/astro/gemini-tts").status_code == 200
    assert client.get("/api/v1/producers/status").status_code == 200
    assert client.post("/api/v1/producers/register", json={"producer_id": "P1", "name": "N1", "hardware": "RTX"}).status_code == 200
    assert client.post("/api/v1/producers/infer", json={"prompt": "hi", "max_new_tokens": 8}).status_code == 200
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).status_code == 200
    assert client.get("/api/astro/catalog").status_code == 200
    assert client.post("/api/astro/bee-moe-scan", json={"target_id": "SECTOR-02-K2-18B"}).status_code == 200
    assert client.post("/api/astro/discover", json={"target_id": "SECTOR-02-K2-18B", "gamer_tag": "G1", "player_classification": "CONFIRMED"}).status_code == 200
    assert client.get("/api/astro/random-hf-target").status_code == 200
    assert client.post("/api/v1/son/feedback", json={"gamer_tag": "Son", "overall_rating": 5, "review_text": "Great"}).status_code == 200
    assert client.get("/api/v1/community/hub").status_code == 200
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert client.post("/api/v1/community/boinc-sync", json={"handle": "@NewGamer", "idle_seconds": 10.0}).status_code == 200
    assert client.post("/api/v1/community/post-note", json={"handle": "@Sci", "note": "Good transit"}).status_code == 200
    assert client.get("/join?lang=en").status_code == 200
    assert client.get("/rejoindre").status_code == 200
    assert client.get("/api/v1/mcp/exoplanet-catalog").status_code == 200
    assert client.get("/portal").status_code == 200
    assert client.get("/v1/hivemind/steam/status").status_code == 200

    reg_steam = client.post("/v1/hivemind/steam/register-login", json={"handle": "steam_pioneer_1", "role": "prosumer"})
    assert reg_steam.status_code == 200
    steam_key = reg_steam.json()["api_key"]
    # Second call exercises HandleTaken branch
    assert client.post("/v1/hivemind/steam/register-login", json={"handle": "steam_pioneer_1"}).status_code == 200
    assert client.get("/v1/hivemind/me").status_code == 401
    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {steam_key}"}).status_code == 200
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert client.post(
            "/v1/hivemind/steam/contribute-boinc",
            json={"target_planet": "K2-18 b"},
            headers={"Authorization": f"Bearer {steam_key}"},
        ).status_code == 200

    assert client.get("/webgpu?lang=en").status_code == 200
    assert client.get("/v1/hivemind/webgpu/status").status_code == 200
    reg_wgpu = client.post("/v1/hivemind/webgpu/register-login", json={"handle": "webgpu_pioneer_1"})
    assert reg_wgpu.status_code == 200
    wgpu_key = reg_wgpu.json()["api_key"]
    assert client.post("/v1/hivemind/webgpu/register-login", json={"handle": "webgpu_pioneer_1"}).status_code == 200
    assert client.post("/v1/hivemind/webgpu/contribute", json={}).status_code == 401
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert client.post(
            "/v1/hivemind/webgpu/contribute",
            json={"target_planet": "K2-18 b"},
            headers={"Authorization": f"Bearer {wgpu_key}"},
        ).status_code == 200
