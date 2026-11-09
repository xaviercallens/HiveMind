# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and validation tests for astrophysics 24h campaign, RTX BOINC deployer, campaign calendar quorum, and multi-MoE HF campaign hardening."""

from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.boinc.astrophysics_24h_campaign import (
    render_astrophysics_24h_dashboard_png,
    render_cosmology_deep_field_discovery_png,
)
from hivemind.boinc.rtx_boinc_deployer import (
    PLATFORM_LINUX,
    PLATFORM_MACOS,
    PLATFORM_WINDOWS,
    build_rtx_boinc_fastapi_router,
    current_boinc_platform,
    deploy_rtx_boinc_client_files,
    execute_rtx_boinc_workunit,
)
from hivemind.cli.campaign_calendar_quorum import (
    build_campaign_quorum_schedule_report,
    generate_campaign_ics_calendar,
    verify_campaign_quorum_schedule_report,
)
from hivemind.cli.multi_moe_hf_campaign import (
    format_campaign_markdown_summary,
    load_polyglot_8domain_fixture,
)


def test_astrophysics_24h_campaign_png_renderers_reject_invalid_inputs(tmp_path: Path) -> None:
    out = tmp_path / "dash.png"
    for fn in (render_astrophysics_24h_dashboard_png, render_cosmology_deep_field_discovery_png):
        for bad_rep in ({}, [], "bad", None, True):
            with pytest.raises(ValueError, match="report must be a non-empty dict"):
                fn(bad_rep, out)  # type: ignore[arg-type]
        for bad_path in ("not-a-path", True, 123, None):
            with pytest.raises(ValueError, match="output_path must be a Path"):
                fn({"ok": True}, bad_path)  # type: ignore[arg-type]


def test_rtx_boinc_deployer_rejects_invalid_inputs(tmp_path: Path) -> None:
    assert current_boinc_platform("win32") == PLATFORM_WINDOWS
    assert current_boinc_platform("darwin") == PLATFORM_MACOS
    assert current_boinc_platform("linux") == PLATFORM_LINUX

    for bad_plat in ("", "   ", True, 123):
        with pytest.raises(ValueError, match="sys_platform must be a non-empty string or None"):
            current_boinc_platform(bad_plat)  # type: ignore[arg-type]

    for bad_srv in (None, "not-a-server", {}, True):
        with pytest.raises(ValueError, match="server must be an RTXBoincServer instance"):
            build_rtx_boinc_fastapi_router(bad_srv)  # type: ignore[arg-type]

    for bad_spec in ({}, {"wu_id": ""}, {"wu_id": "   "}, "not-a-dict", None):
        with pytest.raises(ValueError, match="spec must be a dict with a non-empty wu_id"):
            execute_rtx_boinc_workunit(bad_spec, device="cpu")  # type: ignore[arg-type]

    for bad_dev in ("tpu", "", True, 123):
        with pytest.raises(ValueError, match="device must be one of"):
            execute_rtx_boinc_workunit({"wu_id": "wu-01"}, device=bad_dev)  # type: ignore[arg-type]

    for bad_dir in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="install_dir must be a non-empty Path or str"):
            deploy_rtx_boinc_client_files(bad_dir)  # type: ignore[arg-type]

    for bad_str in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="server_url must be a non-empty string"):
            deploy_rtx_boinc_client_files(tmp_path, server_url=bad_str)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="handle must be a non-empty string"):
            deploy_rtx_boinc_client_files(tmp_path, handle=bad_str)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="authenticator must be a non-empty string"):
            deploy_rtx_boinc_client_files(tmp_path, authenticator=bad_str)  # type: ignore[arg-type]


def test_campaign_calendar_quorum_rejects_invalid_paths(tmp_path: Path) -> None:
    report = build_campaign_quorum_schedule_report(output_path=None, ics_output_path=None)
    assert verify_campaign_quorum_schedule_report(report) is True

    for bad_path in ("not-a-path", True, 123):
        with pytest.raises(ValueError, match="output_path must be a Path or None"):
            generate_campaign_ics_calendar(output_path=bad_path)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="output_path must be a Path or None"):
            build_campaign_quorum_schedule_report(output_path=bad_path, ics_output_path=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ics_output_path must be a Path or None"):
            build_campaign_quorum_schedule_report(output_path=tmp_path / "r.json", ics_output_path=bad_path)  # type: ignore[arg-type]


def test_multi_moe_hf_campaign_helpers_reject_invalid_inputs() -> None:
    payload, digest = load_polyglot_8domain_fixture()
    assert len(digest) == 64 and "domains" in payload

    for bad_path in ("not-a-path", True, 123, None):
        with pytest.raises(ValueError, match="fixture_path must be a Path"):
            load_polyglot_8domain_fixture(bad_path)  # type: ignore[arg-type]

    for bad_rep in ({}, [], "bad", None, True):
        with pytest.raises(ValueError, match="report must be a non-empty dict"):
            format_campaign_markdown_summary(bad_rep)  # type: ignore[arg-type]
