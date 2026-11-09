# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path tests for fat_steam_astro_builder, georing_webgl, hf_hack_miner, and rtx_steam_tool_deployer."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from hivemind.game.fat_steam_astro_builder import (
    _draw_robotic_satellite_glyph,
    _generate_3d_planets_and_galaxy_assets,
)
from hivemind.game.georing_webgl import build_guild_raid_arc_geometry
from hivemind.game.hf_hack_miner import (
    GamerMinerProfile,
    HFBountyChallenge,
    HFDataReconScanner,
)
from hivemind.game.rtx_steam_tool_deployer import deploy_free_steam_rtx_tool_v1


def test_fat_steam_astro_builder_validation() -> None:
    with pytest.raises(ValueError, match="assets_dir"):
        _generate_3d_planets_and_galaxy_assets("not-a-path")  # type: ignore[arg-type]

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    _draw_robotic_satellite_glyph(draw, 32, 32, scale=1.0)
    with pytest.raises(ValueError, match="cx and cy"):
        _draw_robotic_satellite_glyph(draw, True, 32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale"):
        _draw_robotic_satellite_glyph(draw, 32, 32, scale=0.0)


def test_georing_webgl_validation() -> None:
    geom = build_guild_raid_arc_geometry(
        [{"arc_id": "a1", "source_guild": "GUILD-01", "target_guild": "GUILD-02", "source_lat": 48.8, "source_lon": 2.3, "target_lat": 40.7, "target_lon": -74.0}],
        steps=4,
    )
    assert len(geom) == 1 and geom[0]["steps"] == 4
    with pytest.raises(ValueError, match="arcs"):
        build_guild_raid_arc_geometry("not-an-iterable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dict"):
        build_guild_raid_arc_geometry(["not-a-dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="points"):
        build_guild_raid_arc_geometry([{"points": [[0.0, 0.0, 1.0]]}])


def test_hf_hack_miner_validation() -> None:
    with pytest.raises(ValueError, match="contract_id"):
        HFBountyChallenge(
            contract_id="",
            mission_codename="M",
            dataset_id="org/ds",
            domain_expert="Dom",
            difficulty_skulls=1,
            bounty_question_prompt="Q",
            reference_solution="A",
            target_keywords=["kw"],
        )
    with pytest.raises(ValueError, match="difficulty_skulls"):
        HFBountyChallenge(
            contract_id="C1",
            mission_codename="M",
            dataset_id="org/ds",
            domain_expert="Dom",
            difficulty_skulls=0,
            bounty_question_prompt="Q",
            reference_solution="A",
            target_keywords=["kw"],
        )
    with pytest.raises(ValueError, match="target_keywords"):
        HFBountyChallenge(
            contract_id="C1",
            mission_codename="M",
            dataset_id="org/ds",
            domain_expert="Dom",
            difficulty_skulls=1,
            bounty_question_prompt="Q",
            reference_solution="A",
            target_keywords=[],
        )

    prof = GamerMinerProfile(
        gamer_tag="G1",
        steam_id64="76561198000000001",
        node_id="node-1",
        region="EU",
        specialty_focus="Rust",
    )
    prof.update_rank_and_achievements(is_top_miner=True, num_moe_experts=3)
    assert "TOP_1_APEX_MINER" in prof.achievements_unlocked
    with pytest.raises(ValueError, match="is_top_miner"):
        prof.update_rank_and_achievements(is_top_miner="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_moe_experts"):
        prof.update_rank_and_achievements(num_moe_experts=0)

    shard, score = HFDataReconScanner.score_and_curate_shard(
        ["### Instruction: def foo(): return 1"],
        gamer_idx=0,
        num_gamers=1,
        target_keywords=["def", "return"],
    )
    assert len(shard) == 1 and score > 0.0
    with pytest.raises(ValueError, match="raw_samples"):
        HFDataReconScanner.score_and_curate_shard([], 0, 1, ["def"])
    with pytest.raises(ValueError, match="gamer_idx"):
        HFDataReconScanner.score_and_curate_shard(["abc"], -1, 1, ["def"])
    with pytest.raises(ValueError, match="num_gamers"):
        HFDataReconScanner.score_and_curate_shard(["abc"], 0, 0, ["def"])


def test_rtx_steam_tool_deployer_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_root"):
        deploy_free_steam_rtx_tool_v1(
            stress_report={"all_stages_passed": True, "certification_hmac": "cert_123"},
            output_root="not-a-path",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="stress_report"):
        deploy_free_steam_rtx_tool_v1(
            stress_report={"all_stages_passed": "yes", "certification_hmac": "cert_123"},
            output_root=tmp_path,
        )
    with pytest.raises(ValueError, match="stress_report"):
        deploy_free_steam_rtx_tool_v1(
            stress_report={"all_stages_passed": True, "certification_hmac": " "},
            output_root=tmp_path,
        )
