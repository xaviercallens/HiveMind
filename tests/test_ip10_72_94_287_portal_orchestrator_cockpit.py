# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for IP-10 (#380), IP-94 part 2 (#481), and IP-287 (#642)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from hivemind.economy.international_launch_model import (
    build_foundation_paper_v42_portal_payload,
    forecast_200plus_node_international_launch,
    render_foundation_paper_portal_html,
)
from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app


def test_ip94_portal_payload_sources_e20_artifact_and_marks_missing_not_sourced(
    tmp_path: Path,
) -> None:
    payload = build_foundation_paper_v42_portal_payload()
    exp = payload["live_experimentation_v42"]
    e20 = exp["experiment_e20_international_launch"]
    assert e20["status"] == "VERIFIED_PASS"
    assert "E20_international_launch_split_dct_report.json" in e20["data_source"]
    assert e20["executed_at_utc"] == "2026-11-01T09:15:25Z"
    for sub_key in (
        "P20a_adaptive_multisplit_guard",
        "P20b_top1pct_2d_dct_ef21",
        "P20c_super_peer_fanout_and_sqlite_coalescer",
    ):
        sub = e20[sub_key]
        assert sub["artifact_path"] == "artifacts/experiments/E20_international_launch_split_dct_report.json"
        assert sub["executed_at_utc"] == "2026-11-01T09:15:25Z"
        assert "E20_international_launch_split_dct_report.json" in sub["data_source"]

    vmoe = exp["nocturnal_vmoe_pareto_ratchet"]
    assert vmoe["status"] == "NOT_SOURCED"
    assert vmoe["data_source"].startswith("NOT_SOURCED")

    missing_payload = build_foundation_paper_v42_portal_payload(
        e20_artifact_path=tmp_path / "does_not_exist.json"
    )
    missing_e20 = missing_payload["live_experimentation_v42"]["experiment_e20_international_launch"]
    assert missing_e20["status"] == "NOT_SOURCED"
    assert missing_e20["data_source"] == "NOT_SOURCED"
    assert missing_e20["P20a_adaptive_multisplit_guard"]["data_source"] == "NOT_SOURCED"

    corrupt_path = tmp_path / "corrupt_e20.json"
    corrupt_path.write_text("{not valid json", encoding="utf-8")
    corrupt_payload = build_foundation_paper_v42_portal_payload(e20_artifact_path=corrupt_path)
    assert (
        corrupt_payload["live_experimentation_v42"]["experiment_e20_international_launch"]["status"]
        == "NOT_SOURCED"
    )


def test_ip94_render_foundation_paper_portal_html_uses_payload_figures_and_labels(
    tmp_path: Path,
) -> None:
    default_html = render_foundation_paper_portal_html()
    assert "SNAPSHOT_2026-10-31_NOT_LIVE" in default_html
    assert "59-Node WAN Baseline" in default_html
    assert "218.0 tok/s" in default_html
    assert "480.0 tok/s" in default_html
    assert "10,519.0 MiB free VRAM" in default_html
    assert "E20_international_launch_split_dct_report.json" in default_html

    custom = build_foundation_paper_v42_portal_payload(
        e20_artifact_path=tmp_path / "missing_e20.json"
    )
    custom["live_experimentation_v42"]["live_47_node_wan_baseline"]["active_nodes"] = 63
    custom["live_experimentation_v42"]["live_47_node_wan_baseline"]["data_source"] = "CUSTOM_SNAPSHOT_TEST"
    custom_html = render_foundation_paper_portal_html(custom)
    assert "63-Node WAN Baseline (CUSTOM_SNAPSHOT_TEST)" in custom_html
    assert "NOT_SOURCED" in custom_html


@pytest.mark.parametrize("bad_tokens", [0.0, -1.0, float("nan"), float("inf"), 1e308, True])
def test_ip287_forecast_direct_call_rejects_invalid_mean_tokens(bad_tokens: object) -> None:
    with pytest.raises(ValueError, match="mean_tokens_per_req"):
        forecast_200plus_node_international_launch(mean_tokens_per_req=bad_tokens)  # type: ignore[arg-type]


def test_ip287_forecast_direct_call_rejects_non_finite_cv() -> None:
    with pytest.raises(ValueError, match="token_length_cv and arrival_cv must be finite"):
        forecast_200plus_node_international_launch(token_length_cv=float("nan"))


def test_ip10_cockpit_cors_and_host_allowlist() -> None:
    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    # Foreign Host header must be rejected with 400 (DNS rebinding guard)
    bad_host_resp = client.get("/api/v1/monitor/telemetry", headers={"Host": "attacker.example"})
    assert bad_host_resp.status_code == 400

    # Local loopback and IPv6 loopback Host headers are accepted
    for valid_host in ("127.0.0.1:8788", "localhost:8788", "[::1]:8788"):
        ok_resp = client.get("/api/v1/monitor/telemetry", headers={"Host": valid_host})
        assert ok_resp.status_code == 200

    # Foreign Origin must not receive wildcard or reflected Access-Control-Allow-Origin or credentials
    evil_cors = client.get(
        "/api/v1/monitor/telemetry",
        headers={"Host": "127.0.0.1:8788", "Origin": "https://evil.example"},
    )
    assert evil_cors.headers.get("access-control-allow-origin") is None
    assert evil_cors.headers.get("access-control-allow-credentials") != "true"

    # Local Origin receives explicit allow-origin without credentials
    local_cors = client.get(
        "/api/v1/monitor/telemetry",
        headers={"Host": "127.0.0.1:8788", "Origin": "http://127.0.0.1:8788"},
    )
    assert local_cors.headers.get("access-control-allow-origin") == "http://127.0.0.1:8788"
    assert local_cors.headers.get("access-control-allow-credentials") is None
