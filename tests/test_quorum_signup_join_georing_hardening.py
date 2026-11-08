# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for campaign quorum, signup_page, join_page, and georing_webgl hardening."""

from __future__ import annotations

import pytest

from hivemind.cli.campaign_calendar_quorum import evaluate_quorum_readiness
from hivemind.game.georing_webgl import (
    build_guild_raid_arc_geometry,
    great_circle_arc_points,
    guild_raid_arc_color,
    lat_lon_to_xyz,
)
from hivemind.router.join_page import Flags, join_content_security_policy, render_join_page
from hivemind.router.signup_page import content_security_policy, render_signup_page


def test_campaign_quorum_rejects_raw_strings_and_non_sequences() -> None:
    res = evaluate_quorum_readiness(["h1", "h2", "h3"], ["a1", "a2"], ["d1", "d2", "d3", "d4"])
    assert res["tier1_reached"] is True

    with pytest.raises(ValueError, match="human_participants"):
        evaluate_quorum_readiness("xaviercallens", ["a1", "a2"], ["d1", "d2", "d3", "d4"])
    with pytest.raises(ValueError, match="ai_agent_participants"):
        evaluate_quorum_readiness(["h1"], "agent", ["d1"])
    with pytest.raises(ValueError, match="covered_domains"):
        evaluate_quorum_readiness(["h1"], ["a1"], "domain")


def test_signup_and_join_pages_reject_invalid_nonces_and_flags() -> None:
    csp = content_security_policy("abcdefgh12345678")
    assert "nonce-abcdefgh12345678" in csp
    with pytest.raises(ValueError, match="nonce"):
        content_security_policy("bad;nonce")
    with pytest.raises(ValueError, match="nonce"):
        render_signup_page(None, "   ")
    with pytest.raises(ValueError, match="starter_grant_hmt"):
        render_signup_page(None, "abcdefgh12345678", starter_grant_hmt=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="starter_grant_hmt"):
        render_signup_page(None, "abcdefgh12345678", starter_grant_hmt=-5.0)
    with pytest.raises(ValueError, match="auth_result"):
        render_signup_page(None, "abcdefgh12345678", auth_result=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_signup_page(None, "abcdefgh12345678", live_telemetry=["bad"])  # type: ignore[arg-type]

    jcsp = join_content_security_policy("abcdefgh12345678")
    assert "nonce-abcdefgh12345678" in jcsp
    with pytest.raises(ValueError, match="nonce"):
        join_content_security_policy("short")
    with pytest.raises(ValueError, match="google and handles"):
        Flags(google=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="google and handles"):
        render_join_page(google_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_join_page(live_telemetry=["bad"])  # type: ignore[arg-type]


def test_georing_webgl_rejects_boolean_and_out_of_range_coordinates() -> None:
    assert len(guild_raid_arc_color("GUILD-01")) == 3
    with pytest.raises(ValueError, match="guild_id"):
        guild_raid_arc_color("   ")

    p1 = lat_lon_to_xyz(48.85, 2.35)
    p2 = lat_lon_to_xyz(40.71, -74.00)
    assert len(p1) == 3
    with pytest.raises(ValueError, match="lat, lon, and radius"):
        lat_lon_to_xyz(True, 2.35)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="radius"):
        lat_lon_to_xyz(48.85, 2.35, radius=0.0)
    with pytest.raises(ValueError, match="lat must be in"):
        lat_lon_to_xyz(95.0, 0.0)
    with pytest.raises(ValueError, match="lon in"):
        lat_lon_to_xyz(0.0, 195.0)

    pts = great_circle_arc_points(p1, p2, lift=0.2, steps=10)
    assert len(pts) == 11
    with pytest.raises(ValueError, match="lift"):
        great_circle_arc_points(p1, p2, lift=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lift"):
        great_circle_arc_points(p1, p2, lift=-0.1)
    with pytest.raises(ValueError, match="steps"):
        great_circle_arc_points(p1, p2, steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3-element"):
        great_circle_arc_points([1.0, 2.0], p2)
    with pytest.raises(ValueError, match="coordinates"):
        great_circle_arc_points([1.0, True, 0.0], p2)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="steps"):
        build_guild_raid_arc_geometry([], steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps"):
        build_guild_raid_arc_geometry([], steps=0)
