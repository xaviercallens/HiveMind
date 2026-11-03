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

    def fake_bench(self: Any, matrix_dim: int = 3072) -> Dict[str, Any]:
        if matrix_dim != 3072:  # startup warm-up in the pool constructor, not the endpoint
            return {"measured_fp32_tflops": 0.0}
        calls.append(matrix_dim)
        return {"measured_fp32_tflops": 1.0}

    monkeypatch.setattr(engine_mod.HiveMindMultiProducerInferencePool, "benchmark_live_tflops", fake_bench)
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
