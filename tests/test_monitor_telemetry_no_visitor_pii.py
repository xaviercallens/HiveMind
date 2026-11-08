# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public monitor telemetry must not carry per-visitor addresses, and nothing may persist them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hivemind.game.exoplanet_hunter_engine as engine_mod

VISITOR_IP = "203.0.113.77"


def _pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> engine_mod.HiveMindMultiProducerInferencePool:
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    pool = engine_mod.HiveMindMultiProducerInferencePool(engine_mod.AstroBeeMoEGPUService())
    pool.monitor_path = tmp_path / "monitor.json"
    return pool


def test_report_and_persisted_file_carry_no_visitor_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _pool(tmp_path, monkeypatch)
    pool.record_traffic_event("GET", "/", client_ip=VISITOR_IP, user_agent="Mozilla/5.0 (X11)")
    pool.record_traffic_event("POST", "/v1/chat/completions", client_ip=VISITOR_IP, user_agent="python-httpx")
    report = pool.get_live_traffic_and_tflops_report()
    wire = json.dumps(report)
    assert VISITOR_IP not in wire
    assert "ip_address" not in wire and "client_ip" not in wire
    assert VISITOR_IP not in pool.monitor_path.read_text(encoding="utf-8")


def test_report_still_counts_visitors_and_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _pool(tmp_path, monkeypatch)
    before = pool.get_live_traffic_and_tflops_report()
    pool.record_traffic_event("GET", "/", client_ip=VISITOR_IP)
    pool.record_traffic_event("GET", "/", client_ip="198.51.100.9")
    report = pool.get_live_traffic_and_tflops_report()
    assert report["end_users_summary"]["total_unique_end_users"] == before["end_users_summary"]["total_unique_end_users"] + 2
    assert report["traffic_summary"]["total_http_requests"] == before["traffic_summary"]["total_http_requests"] + 2
    assert len(report["recent_traffic_stream"]) == 2
