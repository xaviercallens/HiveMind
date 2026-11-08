# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""IP-200 / IP-202 / IP-204 contract tests for the exoplanet hunter cockpit and engine.

GPU work is never run here: the TFLOPS benchmark is replaced at the class level, and the NASA TAP
network call is replaced at the cockpit module level.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from hivemind.game import exoplanet_hunter_cockpit as cockpit_mod
from hivemind.game import exoplanet_hunter_engine as engine_mod


def _fake_tap(**kwargs: Any) -> Dict[str, Any]:
    return {"source": "LIVE_NASA_TAP_ADQL", "row_count": 1, "rows": [{"pl_name": "K2-18 b"}]}


@pytest.fixture
def cockpit_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "query_nasa_exoplanet_archive_tap", _fake_tap)
    return tmp_path


def test_son_feedback_sanitises_tag_and_claims_no_hmt(cockpit_env: Path) -> None:
    """Defect: tag was written raw (newline + heading injection) and the reply claimed +500 HMT."""
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    resp = client.post(
        "/api/v1/son/feedback",
        json={"gamer_tag": "Evil\n## INJECTED *x*", "overall_rating": 4, "review_text": "ok"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "HMT" not in body["message"] and "+500" not in body["message"]
    assert body["hmt_credited"] is False and body["hmt_awarded"] == 0
    log_text = (cockpit_env / "SON_FEEDBACK.md").read_text(encoding="utf-8")
    assert "\n## INJECTED" not in log_text
    assert "INJECTED" in log_text  # letters survive, only markup and line breaks are removed


def test_son_feedback_rejects_oversize_fields(cockpit_env: Path) -> None:
    """Defect: review_text above 2048 chars was silently truncated and accepted (200)."""
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    resp = client.post("/api/v1/son/feedback", json={"gamer_tag": "Big", "review_text": "x" * 3000})
    assert resp.status_code == 413
    assert resp.json()["status"] == "rejected"
    ok = client.post("/api/v1/son/feedback", json={"gamer_tag": "Small", "review_text": "x" * 2048})
    assert ok.status_code == 200


def test_son_feedback_rate_capped_per_client(cockpit_env: Path) -> None:
    """Defect: unlimited submissions were accepted; the sixth within the window must be refused."""
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    codes = [
        client.post("/api/v1/son/feedback", json={"gamer_tag": f"G{i}", "review_text": "ok"}).status_code
        for i in range(cockpit_mod.FEEDBACK_RATE_LIMIT_MAX + 1)
    ]
    assert codes[:-1] == [200] * cockpit_mod.FEEDBACK_RATE_LIMIT_MAX
    assert codes[-1] == 429


def test_discover_awards_once_per_account_and_target(cockpit_env: Path) -> None:
    """Defect: every replay of the same (account, target) earned points and was appended to the ledger."""
    svc = engine_mod.AstroBeeMoEGPUService()
    svc.discoveries_path = cockpit_env / "ledger.json"
    target = "SECTOR-02-K2-18B"
    first = svc.record_planet_discovery(target, "Alice", "PLANET")
    assert first["award_granted"] is True and first["points_earned"] > 0
    replay = svc.record_planet_discovery(target, " alice ", "PLANET")
    assert replay["award_granted"] is False and replay["points_earned"] == 0
    assert replay["discovery_entry"] is None and replay["status"] == "duplicate_no_award"
    other = svc.record_planet_discovery("SECTOR-06-LHS-1140B", "Alice", "PLANET")
    assert other["award_granted"] is True
    ledger = svc.discoveries_path.read_text(encoding="utf-8")
    assert ledger.count('"target_id": "SECTOR-02-K2-18B"') == 1


def test_benchmark_tflops_cooldown_after_run(cockpit_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: every POST re-ran the GPU benchmark; a second call inside the cooldown must be refused."""
    calls: list[int] = []

    def counted_bench(self: Any, matrix_dim: int = 3072) -> Dict[str, Any]:
        if matrix_dim != 3072:  # startup warm-up in the pool constructor, not the endpoint
            return {"measured_fp32_tflops": 0.0}
        calls.append(matrix_dim)
        return {"measured_fp32_tflops": 1.0}

    monkeypatch.setattr(engine_mod.HiveMindMultiProducerInferencePool, "benchmark_live_tflops", counted_bench)
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    first = client.post("/api/v1/monitor/benchmark-tflops")
    assert first.status_code == 200 and first.json()["benchmark_status"] == "MEASURED_NOW"
    second = client.post("/api/v1/monitor/benchmark-tflops")
    assert second.status_code == 429
    assert second.json()["status"] == "COOLDOWN"
    assert len(calls) == 1


def test_benchmark_tflops_single_flight(cockpit_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: a second benchmark started while one was running ran concurrently on the GPU."""
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow_bench(self: Any, matrix_dim: int = 3072) -> Dict[str, Any]:
        if matrix_dim != 3072:  # startup warm-up in the pool constructor, not the endpoint
            return {"measured_fp32_tflops": 0.0}
        calls.append(matrix_dim)
        started.set()
        release.wait(timeout=5.0)
        return {"measured_fp32_tflops": 1.0}

    monkeypatch.setattr(engine_mod.HiveMindMultiProducerInferencePool, "benchmark_live_tflops", slow_bench)
    app = cockpit_mod.create_exoplanet_hunter_fastapi_app()
    holder: Dict[str, Any] = {}

    def run_first() -> None:
        holder["first"] = TestClient(app).post("/api/v1/monitor/benchmark-tflops")

    worker = threading.Thread(target=run_first)
    worker.start()
    assert started.wait(timeout=5.0)
    second = TestClient(app).post("/api/v1/monitor/benchmark-tflops")
    release.set()
    worker.join(timeout=10.0)
    assert second.status_code == 409 and second.json()["status"] == "BUSY"
    assert holder["first"].status_code == 200
    assert len(calls) == 1


def test_nasa_tap_sync_served_from_cache_within_interval(cockpit_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: every GET went to the NASA TAP network (4.5 s timeout each time)."""
    upstream: list[int] = []

    def counting_tap(**kwargs: Any) -> Dict[str, Any]:
        upstream.append(1)
        return _fake_tap()

    monkeypatch.setattr(cockpit_mod, "query_nasa_exoplanet_archive_tap", counting_tap)
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    first = client.get("/api/astro/nasa-tap-sync").json()
    second = client.get("/api/astro/nasa-tap-sync").json()
    assert first["served_from"] == "UPSTREAM_REFRESH"
    assert second["served_from"] == "MEMORY_CACHE"
    assert second["refresh_interval_s"] == cockpit_mod.TAP_REFRESH_INTERVAL_S
    assert len(upstream) == 1


def test_discover_dedup_survives_200_entry_eviction_and_caps_tag_length_ip286(cockpit_env: Path) -> None:
    """IP-286: dedupe must persist across 200-entry display ledger eviction, and gamer_tag/notes lengths are capped."""
    svc = engine_mod.AstroBeeMoEGPUService()
    svc.discoveries_path = cockpit_env / "ledger_ip286.json"
    target = "SECTOR-02-K2-18B"

    first = svc.record_planet_discovery(target, "Alice", "PLANET")
    assert first["award_granted"] is True and first["points_earned"] == 600

    for i in range(205):
        svc.record_planet_discovery(target, f"junk-{i}", "PLANET")

    # Even after 205 intervening entries evict Alice from the 200-row display ledger, Alice earns 0 on replay
    second = svc.record_planet_discovery(target, "Alice", "PLANET")
    assert second["award_granted"] is False
    assert second["points_earned"] == 0
    assert second["status"] == "duplicate_no_award"

    # Oversized gamer_tag and notes are capped
    long_res = svc.record_planet_discovery("SECTOR-06-LHS-1140B", "A" * 5000, "PLANET", notes="N" * 5000)
    assert long_res["award_granted"] is True
    assert len(long_res["discovery_entry"]["gamer_tag"]) <= 64
    assert len(long_res["discovery_entry"]["notes"]) <= 512


def test_serve_inference_counts_generated_tokens_and_producer_register_is_capped_ip290(
    cockpit_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-290: serve_inference counts generated tokens (clamping max_new_tokens), and producer registration is capped & auth-checked."""
    monkeypatch.setattr(
        engine_mod.HiveMindMultiProducerInferencePool,
        "benchmark_live_tflops",
        lambda self, matrix_dim=3072: {"measured_fp32_tflops": 0.0},
    )
    svc = engine_mod.AstroBeeMoEGPUService()
    pool = engine_mod.HiveMindMultiProducerInferencePool(svc)
    pool.serve_inference("Reply with OK.", max_new_tokens=10**9)
    assert pool.total_tokens_processed < 10_000

    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    # Primary anchor producer cannot be set offline
    bad_anchor = client.post(
        "/api/v1/producers/register",
        json={"producer_id": "PRODUCER-LOCAL-RTX2070-CUDA0", "name": "Anchor", "hardware": "RTX", "status": "OFFLINE"},
    )
    assert bad_anchor.status_code == 400

    # Flooding registrations hits the 64-producer cap (413)
    statuses = [
        client.post(
            "/api/v1/producers/register",
            json={"producer_id": f"FLOOD-{i}", "name": f"N-{i}", "hardware": "GPU"},
        ).status_code
        for i in range(80)
    ]
    assert 413 in statuses

    # When HIVEMIND_PRODUCER_REGISTER_TOKEN is set, unauthenticated requests return 401
    monkeypatch.setenv("HIVEMIND_PRODUCER_REGISTER_TOKEN", "secret-reg-token")
    unauth = client.post("/api/v1/producers/register", json={"producer_id": "FLOOD-0", "name": "N", "hardware": "GPU"})
    assert unauth.status_code == 401


def test_ip265_ip266_end_users_lru_cap_and_non_inference_served_by_none(
    cockpit_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-265 & IP-266: end_users is capped at MAX_TRACKED_END_USERS with an 'other' bucket, and /health has served_by=None."""
    monkeypatch.setattr(
        engine_mod.HiveMindMultiProducerInferencePool,
        "benchmark_live_tflops",
        lambda self, matrix_dim=3072: {"measured_fp32_tflops": 0.0},
    )
    pool = engine_mod.HiveMindMultiProducerInferencePool(engine_mod.AstroBeeMoEGPUService())
    pool.monitor_path = cockpit_env / "monitor_ip265.json"

    for i in range(5000):
        pool.record_traffic_event("GET", "/health", client_ip=f"198.51.{i // 256}.{i % 256}")

    assert len(pool.end_users) <= pool.MAX_TRACKED_END_USERS
    assert "other" in pool.end_users
    assert pool.end_users["other"]["aggregated_users"] >= 5000 - pool.MAX_TRACKED_END_USERS
    assert pool.recent_traffic[0]["path"] == "/health"
    assert pool.recent_traffic[0]["served_by"] is None

    report = pool.get_live_traffic_and_tflops_report()
    assert len(report["end_users_summary"]["active_end_users"]) <= pool.MAX_TRACKED_END_USERS
    assert pool.monitor_path.stat().st_size < 150_000


def test_ip267_ip268_feedback_markdown_sanitisation_and_steam_hud_script_escape(cockpit_env: Path) -> None:
    """IP-267 & IP-268: feedback comments/ideas cannot inject ## headings, and _generate_cyberpunk_hud_html escapes </script>."""
    from hivemind.game.steam_edition import _generate_cyberpunk_hud_html

    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    resp = client.post(
        "/api/v1/son/feedback",
        json={
            "gamer_tag": "Tester",
            "overall_rating": 5,
            "review_text": "x\n## Avis de Admin",
            "feature_suggestions": "y\n## Fake Entry",
        },
    )
    assert resp.status_code == 200
    log_text = (cockpit_env / "SON_FEEDBACK.md").read_text(encoding="utf-8")
    heading_lines = [ln for ln in log_text.splitlines() if ln.startswith("## ")]
    assert len(heading_lines) == 1

    html = _generate_cyberpunk_hud_html(
        {"leaderboard": [{"gamer_tag": "</script><img src=x onerror=alert(1)>", "rank_title": "R"}]}
    )
    assert "</script><img" not in html
    assert "<\\/script>" in html
    assert "escHtml(g.gamer_tag)" in html


