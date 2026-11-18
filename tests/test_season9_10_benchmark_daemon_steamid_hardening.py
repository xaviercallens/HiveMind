# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and validation tests for Season 9/10 GPU swarm campaigns, community GPU benchmark, ZeroIntrusionDaemon, and Steam identity hardening."""

from __future__ import annotations

import pytest

from hivemind.cli.community_gpu_benchmark import run_community_gpu_benchmark
from hivemind.cli.season9_10_gpu_swarm_campaigns import (
    format_season9_10_gpu_campaigns_markdown,
    run_season9_10_gpu_swarm_campaigns,
)
from hivemind.client.daemon import DaemonTelemetry, detect_running_game_process
from hivemind.client.steam_identity import AccountFacts, default_http


def test_season9_10_gpu_swarm_campaigns_rejects_invalid_inputs() -> None:
    for bad_seed in (-1, True, 1.5, "42"):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            run_season9_10_gpu_swarm_campaigns(seed=bad_seed)  # type: ignore[arg-type]

    for bad_path in ("not-a-path", True, 123):
        with pytest.raises(ValueError, match="output_path must be a Path or None"):
            run_season9_10_gpu_swarm_campaigns(output_path=bad_path)  # type: ignore[arg-type]

    for bad_rep in ({}, [], "bad", None, True):
        with pytest.raises(ValueError, match="report must be a non-empty dict"):
            format_season9_10_gpu_campaigns_markdown(bad_rep)  # type: ignore[arg-type]


def test_community_gpu_benchmark_rejects_invalid_seed_and_path() -> None:
    for bad_seed in (-1, True, 1.5, "42"):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            run_community_gpu_benchmark(seed=bad_seed)  # type: ignore[arg-type]

    for bad_path in ("not-a-path", True, 123):
        with pytest.raises(ValueError, match="output_path must be a Path or None"):
            run_community_gpu_benchmark(output_path=bad_path)  # type: ignore[arg-type]


def test_daemon_telemetry_and_game_watchlist_reject_invalid_inputs() -> None:
    tel = DaemonTelemetry(
        os_idle_seconds=45.0,
        idle_threshold_seconds=30.0,
        active_game_detected=None,
        should_yield_vram=False,
        reason="Idle mining active",
    )
    assert tel.should_yield_vram is False

    for bad_idle in (-1.0, float("nan"), float("inf"), True, "45"):
        with pytest.raises(ValueError, match="os_idle_seconds"):
            DaemonTelemetry(
                os_idle_seconds=bad_idle,  # type: ignore[arg-type]
                idle_threshold_seconds=30.0,
                active_game_detected=None,
                should_yield_vram=False,
                reason="ok",
            )

    for bad_thresh in (0.0, -1.0, float("nan"), True, "30"):
        with pytest.raises(ValueError, match="idle_threshold_seconds"):
            DaemonTelemetry(
                os_idle_seconds=10.0,
                idle_threshold_seconds=bad_thresh,  # type: ignore[arg-type]
                active_game_detected=None,
                should_yield_vram=False,
                reason="ok",
            )

    with pytest.raises(ValueError, match="should_yield_vram must be a boolean"):
        DaemonTelemetry(
            os_idle_seconds=10.0,
            idle_threshold_seconds=30.0,
            active_game_detected=None,
            should_yield_vram="no",  # type: ignore[arg-type]
            reason="ok",
        )

    for bad_reason in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="reason must be a non-empty string"):
            DaemonTelemetry(
                os_idle_seconds=10.0,
                idle_threshold_seconds=30.0,
                active_game_detected=None,
                should_yield_vram=False,
                reason=bad_reason,  # type: ignore[arg-type]
            )

    for bad_watch in ("cs2.exe", 123, {""}, {"   "}, {123}):
        with pytest.raises(ValueError, match="extra_watchlist must be a collection of non-empty strings"):
            detect_running_game_process(extra_watchlist=bad_watch)  # type: ignore[arg-type]


def test_steam_identity_account_facts_and_http_reject_invalid_inputs() -> None:
    facts = AccountFacts(steam_id64="76561197960265728", limited=False, vac_banned=False)
    assert facts.limited is False

    for bad_id in ("", "12345", True, 76561197960265728, None):
        with pytest.raises(ValueError, match="steam_id64 must be a valid SteamID64"):
            AccountFacts(steam_id64=bad_id, limited=False, vac_banned=False)  # type: ignore[arg-type]

    for bad_flag in (0, 1, "false", None):
        with pytest.raises(ValueError, match="limited and vac_banned must be booleans"):
            AccountFacts(steam_id64="76561197960265728", limited=bad_flag, vac_banned=False)  # type: ignore[arg-type]

    for bad_method in ("PUT", "DELETE", "", True, 123):
        with pytest.raises(ValueError, match="method must be 'GET' or 'POST'"):
            default_http(bad_method, "https://steamcommunity.com/openid/login")  # type: ignore[arg-type]

    for bad_url in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="url must be a non-empty string"):
            default_http("GET", bad_url)  # type: ignore[arg-type]

    for bad_form in ("not-a-mapping", ["a", "b"], 123):
        with pytest.raises(ValueError, match="form must be a Mapping or None"):
            default_http("POST", "https://steamcommunity.com/openid/login", form=bad_form)  # type: ignore[arg-type]
