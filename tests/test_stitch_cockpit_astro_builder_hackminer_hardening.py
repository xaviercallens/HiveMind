# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and validation tests for Stitch portal, v4.6 Cockpit template, Fat Steam Astro builder, and Hack-Miner hardening."""

from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.boinc.stitch_portal import render_stitch_enhanced_portal
from hivemind.game.fat_steam_astro_builder import (
    _generate_godot4_professional_engine_project,
    _generate_steam_artwork_assets,
    build_fat_steam_exoplanet_client,
)
from hivemind.game.hf_hack_miner import (
    audit_lean4_proof,
    build_agent_arena_state,
    build_guild_tournament_state,
    build_swarm_telemetry_frame,
    fetch_offline_curated_corpus,
    fetch_season4_curated_corpus,
    fetch_season5_curated_corpus,
    fetch_season6_curated_corpus,
)
from hivemind.game.stitch_v46_cockpit_template import (
    build_stitch_v46_cockpit_html,
    render_stitch_v46_cockpit_html,
)


def test_render_stitch_enhanced_portal_rejects_invalid_inputs() -> None:
    html = render_stitch_enhanced_portal({}, api_base_url="", default_screen="home")
    assert "<!DOCTYPE html>" in html or "<html" in html

    with pytest.raises(ValueError, match="state must be a dict"):
        render_stitch_enhanced_portal([])  # type: ignore[arg-type]

    for bad_url in (True, 123, None):
        with pytest.raises(ValueError, match="api_base_url must be a str"):
            render_stitch_enhanced_portal({}, api_base_url=bad_url)  # type: ignore[arg-type]

    for bad_screen in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="default_screen must be a non-empty string"):
            render_stitch_enhanced_portal({}, default_screen=bad_screen)  # type: ignore[arg-type]


def test_stitch_v46_cockpit_template_rejects_invalid_state_json() -> None:
    for bad_json in ("", "   ", True, 123, None):
        with pytest.raises(ValueError, match="state_json must be a non-empty JSON string"):
            build_stitch_v46_cockpit_html(bad_json)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="state_json must be a non-empty JSON string"):
            render_stitch_v46_cockpit_html(bad_json)  # type: ignore[arg-type]

    for malformed in ("{not-json", "[1, 2, 3]", '"just-a-string"', "42"):
        with pytest.raises(ValueError, match="state_json"):
            build_stitch_v46_cockpit_html(malformed)


def test_fat_steam_astro_builder_rejects_invalid_inputs(tmp_path: Path) -> None:
    godot_dir = _generate_godot4_professional_engine_project(tmp_path, [{"pl_name": "K2-18 b"}])
    assert (godot_dir / "project.godot").exists()

    for bad_dir in ("not-a-path", True, 123, None):
        with pytest.raises(ValueError, match="pkg_dir must be a Path"):
            _generate_godot4_professional_engine_project(bad_dir, [])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="assets_dir must be a Path"):
            _generate_steam_artwork_assets(bad_dir)  # type: ignore[arg-type]

    for bad_catalog in ("not-a-list", [{"ok": 1}, "bad-item"]):
        with pytest.raises(ValueError, match="catalog must be a list of dicts"):
            _generate_godot4_professional_engine_project(tmp_path, bad_catalog)  # type: ignore[arg-type]

    for bad_flag in ("yes", 1, None):
        with pytest.raises(ValueError, match="install_to_local_steam_library must be a bool"):
            build_fat_steam_exoplanet_client(install_to_local_steam_library=bad_flag)  # type: ignore[arg-type]


def test_hf_hack_miner_curated_corpus_rejects_invalid_inputs() -> None:
    rows = fetch_season4_curated_corpus("hivemind/qasm-circuit-opt", num_samples=3)
    assert len(rows) == 3

    for fn in (
        fetch_season4_curated_corpus,
        fetch_season5_curated_corpus,
        fetch_season6_curated_corpus,
        fetch_offline_curated_corpus,
    ):
        for bad_id in ("", "   ", True, 123, None):
            with pytest.raises(ValueError, match="dataset_id must be a non-empty string"):
                fn(bad_id, num_samples=4)  # type: ignore[arg-type]
        for bad_n in (0, -1, 10_001, True, 1.5, "4"):
            with pytest.raises(ValueError, match="num_samples must be an int"):
                fn("hivemind/qasm-circuit-opt", num_samples=bad_n)  # type: ignore[arg-type]


def test_hf_hack_miner_arenas_and_telemetry_reject_invalid_inputs() -> None:
    assert audit_lean4_proof("")["valid"] is False
    for bad_proof in (True, 123, None, ["theorem"]):
        with pytest.raises(ValueError, match="proof must be a str"):
            audit_lean4_proof(bad_proof)  # type: ignore[arg-type]

    arena = build_agent_arena_state(seed=7, rounds=2)
    assert arena["season"] == 6
    for bad_seed in (-1, True, 1.5, "7"):
        with pytest.raises(ValueError, match="seed must be a non-negative int"):
            build_agent_arena_state(seed=bad_seed, rounds=2)  # type: ignore[arg-type]
    for bad_rounds in (0, -1, 1001, True, 1.5):
        with pytest.raises(ValueError, match="rounds must be an int"):
            build_agent_arena_state(seed=7, rounds=bad_rounds)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="engine must be a GuildTournamentEngine or None"):
        build_guild_tournament_state(engine="bad-engine")  # type: ignore[arg-type]

    frame = build_swarm_telemetry_frame(seq=0)
    assert frame["seq"] == 0
    for bad_seq in (-1, True, 1.5, "0"):
        with pytest.raises(ValueError, match="seq must be a non-negative int"):
            build_swarm_telemetry_frame(seq=bad_seq)  # type: ignore[arg-type]
