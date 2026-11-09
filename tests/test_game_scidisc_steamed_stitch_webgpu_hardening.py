# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path tests for scientific_discovery_game, steam_edition, stitch_v46_cockpit_template, and webgpu_galaxy_cockpit."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.game.scientific_discovery_game import (
    ScientificMissionSpec,
    inspect_unity_steam_stack,
)
from hivemind.game.steam_edition import _generate_free_steam_readme_md
from hivemind.game.stitch_v46_cockpit_template import render_stitch_v46_cockpit_html
from hivemind.game.webgpu_galaxy_cockpit import (
    execute_server_webgpu_verification_and_sync,
    render_webgpu_galaxy_html,
)


def test_scientific_discovery_game_validation(tmp_path: Path) -> None:
    res = inspect_unity_steam_stack(project_root=tmp_path)
    assert res["all_files_present"] is False
    with pytest.raises(ValueError, match="project_root"):
        inspect_unity_steam_stack(project_root="not-a-path")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scientific_domain"):
        ScientificMissionSpec(
            mission_id="M1",
            title="T1",
            scientific_domain=" ",
            specialist_bee="hivemind/coder",
            unity_scene_script="scene.cs",
            invariant_gate="gate",
            base_scientific_pts=100,
            description="desc",
        )
    with pytest.raises(ValueError, match="description"):
        ScientificMissionSpec(
            mission_id="M1",
            title="T1",
            scientific_domain="Dom",
            specialist_bee="hivemind/coder",
            unity_scene_script="scene.cs",
            invariant_gate="gate",
            base_scientific_pts=100,
            description="",
        )


def test_steam_edition_readme_validation() -> None:
    md = _generate_free_steam_readme_md({"best_miner_champion": {"gamer_tag": "Champ1", "total_points": 5000}})
    assert "Champ1" in md
    with pytest.raises(ValueError, match="game_report"):
        _generate_free_steam_readme_md("not-a-dict")  # type: ignore[arg-type]


def test_stitch_v46_cockpit_template_validation() -> None:
    with pytest.raises(ValueError, match="state_json"):
        render_stitch_v46_cockpit_html(" ")
    with pytest.raises(ValueError, match="state_json"):
        render_stitch_v46_cockpit_html(True)  # type: ignore[arg-type]


def test_webgpu_galaxy_cockpit_validation() -> None:
    html = render_webgpu_galaxy_html("en")
    assert 'lang="en"' in html
    with pytest.raises(ValueError, match="default_lang"):
        render_webgpu_galaxy_html(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="handle"):
        execute_server_webgpu_verification_and_sync(
            handle="",
            target_planet="K2-18 b",
            gpu_adapter="WebGPU",
            backend="webgpu",
            n_points_folded=1024,
            kernel_exec_ms=12.0,
            recovered_transit_depth_ppm=2900.0,
        )
    with pytest.raises(ValueError, match="kernel_exec_ms"):
        execute_server_webgpu_verification_and_sync(
            handle="cmdr1",
            target_planet="K2-18 b",
            gpu_adapter="WebGPU",
            backend="webgpu",
            n_points_folded=1024,
            kernel_exec_ms=True,  # type: ignore[arg-type]
            recovered_transit_depth_ppm=2900.0,
        )
