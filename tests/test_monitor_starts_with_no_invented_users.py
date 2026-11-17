# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A fresh exoplanet pool has served nobody: no sample end-users and no pre-set request, inference or token counters."""

from __future__ import annotations

from pathlib import Path

import pytest

import hivemind.game.exoplanet_hunter_engine as engine_mod


def _fresh_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> engine_mod.HiveMindMultiProducerInferencePool:
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    pool = engine_mod.HiveMindMultiProducerInferencePool(engine_mod.AstroBeeMoEGPUService())
    pool.monitor_path = tmp_path / "monitor.json"
    return pool


def test_a_fresh_pool_reports_no_users_and_zero_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _fresh_pool(tmp_path, monkeypatch).get_live_traffic_and_tflops_report()
    assert report["end_users_summary"]["total_unique_end_users"] == 0
    assert report["end_users_summary"]["active_end_users"] == []
    assert report["traffic_summary"]["total_http_requests"] == 0
    assert report["traffic_summary"]["total_gpu_inferences_served"] == 0
    assert report["traffic_summary"]["total_tokens_processed"] == 0


def test_only_real_requests_create_users_and_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _fresh_pool(tmp_path, monkeypatch)
    pool.record_traffic_event("GET", "/", client_ip="198.51.100.4")
    pool.record_traffic_event("POST", "/api/astro/discover", client_ip="198.51.100.5", gamer_tag="Real_Player")
    report = pool.get_live_traffic_and_tflops_report()
    assert report["end_users_summary"]["total_unique_end_users"] == 2
    assert report["traffic_summary"]["total_http_requests"] == 2
    assert {u["gamer_tag"] for u in report["end_users_summary"]["active_end_users"]} >= {"Real_Player"}
