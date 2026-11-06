"""Season 5 Milestone HM-S5-04: Guild Wars Co-Op Tournament, Missions 17-20 & 3D WebGL Raid Arcs.

Validates:
- the four Season 5 Hack-Miner frontier missions 17-20 (`SEASON_5_HF_BOUNTY_CONTRACTS`)
  and their completion through the interactive QLoRA Micro-Forge,
- the `/api/guild-tournament` REST schema (cross-guild matchmaking of `GUILD-01`,
  `GUILD-02` and `GUILD-03` on multi-domain holdout perplexity and speculative
  tokens/sec),
- the WebGL raid-arc geometry generation used by the 3D Geo-Ring globe.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from hivemind.game.georing_webgl import (  # noqa: E402
    GUILD_RAID_ARC_LIFT,
    GUILD_RAID_ARC_STEPS,
    build_guild_raid_arc_geometry,
    great_circle_arc_points,
    guild_raid_arc_color,
    lat_lon_to_xyz,
)
from hivemind.game.hf_hack_miner import (  # noqa: E402
    SEASON_5_EXTENDED_HF_BOUNTY_CONTRACTS,
    SEASON_5_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS,
    SEASON_5_HF_BOUNTY_CONTRACTS,
    STEAM_ACHIEVEMENTS_CATALOG,
    build_guild_tournament_state,
    build_season5_mission_catalog,
    build_swarm_telemetry_frame,
    estimate_bigram_holdout_perplexity,
    execute_interactive_lora_micro_forge,
    fetch_season5_curated_corpus,
)
from hivemind.game.steam_edition import create_hack_miner_fastapi_app  # noqa: E402

SEASON_5_MISSION_IDS = [
    "CONTRACT-HF-17-CUDA-ROCM-PTX-OPT",
    "CONTRACT-HF-18-LEAN4-AUTOFORMALIZE",
    "CONTRACT-HF-19-PROTEIN-STRUCTURE-PRED",
    "CONTRACT-HF-20-AUTONOMOUS-SWE-PATCH",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_hack_miner_fastapi_app())


def test_s5_04_season5_missions_17_20_catalog() -> None:
    """The four Season 5 frontier contracts exist, are Season 5, and are catalogued."""
    ids = [c.contract_id for c in SEASON_5_HF_BOUNTY_CONTRACTS]
    assert ids == SEASON_5_MISSION_IDS
    assert all(c.season == 5 for c in SEASON_5_HF_BOUNTY_CONTRACTS)
    assert all(c.difficulty_skulls >= 4 for c in SEASON_5_HF_BOUNTY_CONTRACTS)
    assert all(c.base_bounty_pts > 0 for c in SEASON_5_HF_BOUNTY_CONTRACTS)

    domains = [c.domain_expert for c in SEASON_5_HF_BOUNTY_CONTRACTS]
    assert "CUDA_ROCm_PTX_Optimizer" in domains
    assert "Lean4_Mathlib_Autoformalizer" in domains
    assert "Biomolecular_Protein_Structure" in domains
    assert "Autonomous_SWE_Patch_Synthesizer" in domains

    # The Season 5 campaign extends (never mutates) the Season 1-4 campaigns.
    assert len(SEASON_5_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS) == 18
    assert len(SEASON_5_EXTENDED_HF_BOUNTY_CONTRACTS) == 20
    assert len(set(c.contract_id for c in SEASON_5_EXTENDED_HF_BOUNTY_CONTRACTS)) == 20

    catalog = build_season5_mission_catalog()
    assert catalog["season"] == 5
    assert catalog["num_missions"] == 4
    assert catalog["mission_ids"] == SEASON_5_MISSION_IDS
    assert catalog["bounty_pool_hive"] == 2500.0
    assert catalog["total_bounty_pts"] == sum(
        c.base_bounty_pts for c in SEASON_5_HF_BOUNTY_CONTRACTS
    )
    assert catalog["guild_tournament_endpoint"] == "/api/guild-tournament"

    # Every mission ships an offline corpus so it stays playable without network.
    for contract in SEASON_5_HF_BOUNTY_CONTRACTS:
        corpus = fetch_season5_curated_corpus(contract.dataset_id, num_samples=12)
        assert len(corpus) == 12
        assert corpus[0].startswith(contract.bounty_question_prompt)
        assert contract.reference_solution in corpus[0]


def test_s5_04_season5_missions_are_completable() -> None:
    """Each Season 5 mission can be completed through the interactive Micro-Forge."""
    for contract in SEASON_5_HF_BOUNTY_CONTRACTS:
        result = execute_interactive_lora_micro_forge(
            gamer_tag="CyberValkyrie_FR",
            contract_id=contract.contract_id,
        )
        assert result["contract_id"] == contract.contract_id
        assert result["season"] == 5
        assert result["dataset_id"] == contract.dataset_id
        assert result["domain_expert"] == contract.domain_expert
        assert result["spot_check_passed"] is True
        assert result["final_loss"] < result["initial_loss"]
        assert result["loss_reduction_pct"] > 0.0
        assert result["bounty_pts_awarded"] > contract.base_bounty_pts
        assert result["total_pts_awarded"] > contract.base_bounty_pts
        assert result["hive_minted"] > 0.0
        assert result["steam_rich_presence"]
        assert result["guild_wars_badge"] == "HF_COOP_RAID_COMMANDER"

    # The Guild Wars co-op raid badge is published in the Steam achievements catalog.
    assert "HF_COOP_RAID_COMMANDER" in STEAM_ACHIEVEMENTS_CATALOG


def test_s5_04_season5_missions_api_and_telemetry(client: TestClient) -> None:
    """`/api/season5-missions` and the live telemetry frame expose the Season 5 campaign."""
    body = client.get("/api/season5-missions").json()
    assert body["season"] == 5
    assert body["num_missions"] == 4
    assert body["missions"] == SEASON_5_MISSION_IDS
    assert body["mission_ids"] == SEASON_5_MISSION_IDS
    assert len(body["contracts"]) == 4
    assert body["bounty_pool_hive"] == 2500.0
    assert body["guild_tournament_endpoint"] == "/api/guild-tournament"
    for contract in body["contracts"]:
        assert contract["season"] == 5
        assert contract["difficulty_skulls"] >= 4
        assert contract["base_bounty_pts"] > 0

    frame = build_swarm_telemetry_frame(0)
    assert frame["stream"] == "swarm-telemetry"
    assert frame["season5_missions"] == 4
    assert frame["season5_mission_ids"] == SEASON_5_MISSION_IDS
    assert frame["guild_tournament"]["num_guilds"] == 3
    assert frame["guild_tournament"]["num_raid_arcs"] == 3


def test_s5_04_guild_tournament_api_json_schema(client: TestClient) -> None:
    """`/api/guild-tournament` publishes the documented cross-guild tournament schema."""
    resp = client.get("/api/guild-tournament")
    assert resp.status_code == 200
    body = resp.json()

    for key in (
        "schema_version",
        "tournament_id",
        "milestone",
        "season",
        "status",
        "bounty_pool_hive",
        "badge",
        "num_guilds",
        "num_matches",
        "num_raid_arcs",
        "champion_guild_id",
        "guilds",
        "standings",
        "matches",
        "raid_arcs",
        "metric_definitions",
        "missions",
    ):
        assert key in body, key

    assert body["schema_version"] == "hm-s5-04/guild-tournament/v1"
    assert body["milestone"] == "HM-S5-04"
    assert body["season"] == 5
    assert body["status"] == "live"
    assert body["badge"] == "HF_COOP_RAID_COMMANDER"
    assert body["bounty_pool_hive"] == 2500.0
    assert body["num_guilds"] == 3
    assert body["missions"]["mission_ids"] == SEASON_5_MISSION_IDS

    guild_ids = [g["guild_id"] for g in body["guilds"]]
    assert guild_ids == ["GUILD-01", "GUILD-02", "GUILD-03"]
    for guild in body["guilds"]:
        assert -90 <= guild["lat"] <= 90
        assert -180 <= guild["lon"] <= 180
        assert guild["member_miners"]
        assert guild["synergy_multiplier"] > 1.0
        assert guild["multi_domain_holdout_perplexity"] > 0.0
        assert guild["speculative_tokens_per_sec"] > 0.0
        assert guild["raid_arc_ids"]

    # Standings are ranked and sorted by total points.
    ranks = [s["rank"] for s in body["standings"]]
    assert ranks == [1, 2, 3]
    totals = [s["total_points"] for s in body["standings"]]
    assert totals == sorted(totals, reverse=True)
    assert body["champion_guild_id"] == body["standings"][0]["guild_id"]
    for standing in body["standings"]:
        assert standing["total_points"] == pytest.approx(
            standing["perplexity_points"]
            + standing["throughput_points"]
            + standing["coop_synergy_points"],
            abs=0.01,
        )

    # Deterministic round-robin matchmaking: 3 guilds => 3 matches.
    assert body["num_matches"] == 3
    assert len(body["matches"]) == 3
    seen_pairs = set()
    for match in body["matches"]:
        pair = frozenset({match["home_guild"], match["away_guild"]})
        assert pair not in seen_pairs
        seen_pairs.add(pair)
        assert match["winner_guild"] in pair
        assert match["loser_guild"] in pair
        assert match["winner_guild"] != match["loser_guild"]
        assert match["raid_arc_id"] in {a["arc_id"] for a in body["raid_arcs"]}

    # Every settled match spawns one animated raid arc with real 3D geometry.
    assert body["num_raid_arcs"] == 3
    assert len(body["raid_arcs"]) == 3
    for arc in body["raid_arcs"]:
        assert arc["kind"] == "guild_raid"
        assert arc["source_guild"] in guild_ids
        assert arc["target_guild"] in guild_ids
        assert arc["source_guild"] != arc["target_guild"]
        assert len(arc["points"]) == GUILD_RAID_ARC_STEPS + 1
        assert arc["arc_length"] > 0.0
        assert arc["max_radius"] > 1.0  # the raid arc lifts off the globe surface
        assert len(arc["color"]) == 3
        assert 0.0 < arc["intensity"] <= 1.0
        assert arc["points"][0] == pytest.approx(arc["source_xyz"], abs=1e-6)
        assert arc["points"][-1] == pytest.approx(arc["target_xyz"], abs=1e-6)


def test_s5_04_guild_tournament_matchmaking_update(client: TestClient) -> None:
    """Unauthenticated self-reports are recorded as unverified without re-ranking the champion (IP-153)."""
    initial_champion = client.get("/api/guild-tournament").json()["champion_guild_id"]
    resp = client.post(
        "/api/guild-tournament",
        json={
            "guild_id": "GUILD-03",
            "gamer_tag": "AetherPulse_JP",
            "multi_domain_holdout_perplexity": 4.25,
            "speculative_tokens_per_sec": 31.5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_guilds"] == 3
    assert body["num_raid_arcs"] == 3
    updated = next(g for g in body["guilds"] if g["guild_id"] == "GUILD-03")
    assert updated["squad_reported"] is True
    assert updated["squad_report_verified"] is False
    assert updated["self_report_status"] == "UNVERIFIED_IGNORED_FOR_RANKING"
    assert updated["reported_unverified_perplexity"] == pytest.approx(4.25)
    assert updated["reported_unverified_tokens_per_sec"] == pytest.approx(31.5)
    assert body["champion_guild_id"] == initial_champion

    unknown = client.post("/api/guild-tournament", json={"guild_id": "GUILD-99"})
    assert unknown.status_code == 404


def test_s5_04_holdout_perplexity_is_a_real_measurement() -> None:
    """The multi-domain holdout perplexity is measured from the curated corpora."""
    for contract in SEASON_5_HF_BOUNTY_CONTRACTS:
        corpus = fetch_season5_curated_corpus(contract.dataset_id, num_samples=12)
        stats = estimate_bigram_holdout_perplexity(corpus)
        assert stats["measured"] is True
        assert stats["holdout_perplexity"] > 1.0
        assert stats["train_tokens"] > 0
        assert stats["holdout_tokens"] > 0
        assert stats["vocab_size"] > 1

    # Deterministic: the same corpus always yields the same perplexity.
    corpus = fetch_season5_curated_corpus("hivemind/lean4-mathlib-autoformalizer", num_samples=12)
    first = estimate_bigram_holdout_perplexity(corpus)
    second = estimate_bigram_holdout_perplexity(corpus)
    assert first == second


def test_s5_04_guild_raid_arc_geometry_generation() -> None:
    """The WebGL raid-arc geometry is a lifted great-circle arc on the globe."""
    source = lat_lon_to_xyz(48.8566, 2.3522)
    target = lat_lon_to_xyz(35.6762, 139.6503)
    assert math.isclose(math.sqrt(sum(c * c for c in source)), 1.0, abs_tol=1e-9)
    assert math.isclose(math.sqrt(sum(c * c for c in target)), 1.0, abs_tol=1e-9)

    points = great_circle_arc_points(source, target, lift=GUILD_RAID_ARC_LIFT, steps=GUILD_RAID_ARC_STEPS)
    assert len(points) == GUILD_RAID_ARC_STEPS + 1
    assert points[0] == pytest.approx(list(source), abs=1e-9)
    assert points[-1] == pytest.approx(list(target), abs=1e-9)

    radii = [math.sqrt(sum(c * c for c in p)) for p in points]
    assert max(radii) > 1.0  # the arc lifts off the sphere
    assert math.isclose(radii[0], 1.0, abs_tol=1e-9)
    assert math.isclose(radii[-1], 1.0, abs_tol=1e-9)
    assert max(radii) == max(radii) and min(radii) == pytest.approx(1.0, abs=1e-9)

    polyline = sum(
        math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)
    )
    great_circle = math.acos(
        max(-1.0, min(1.0, sum(a * b for a, b in zip(source, target))))
    )
    assert polyline > great_circle  # the lifted arc is longer than the surface chord

    geometry = build_guild_raid_arc_geometry(
        [
            {
                "arc_id": "RAID-ARC-TEST",
                "source_guild": "GUILD-01",
                "target_guild": "GUILD-03",
                "source_lat": 48.8566,
                "source_lon": 2.3522,
                "target_lat": 35.6762,
                "target_lon": 139.6503,
            }
        ]
    )
    assert len(geometry) == 1
    arc = geometry[0]
    assert arc["arc_id"] == "RAID-ARC-TEST"
    assert arc["steps"] == GUILD_RAID_ARC_STEPS
    assert len(arc["points"]) == GUILD_RAID_ARC_STEPS + 1
    assert arc["color"] == guild_raid_arc_color("GUILD-01")
    assert arc["max_radius"] > 1.0
    assert arc["arc_length"] > great_circle

    # The published tournament geometry is exactly the generated geometry.
    state = build_guild_tournament_state()
    published = state["raid_arcs"]
    assert len(published) == state["num_raid_arcs"]
    for arc in published:
        assert arc["points"] == build_guild_raid_arc_geometry([arc])[0]["points"]


def test_s5_04_hud_renders_guild_wars_panel_and_raid_arcs(client: TestClient) -> None:
    """The HUD ships the Guild Wars panel and the animated WebGL raid-arc renderer."""
    html = client.get("/").text
    assert "season5-missions-panel" in html
    assert "guild-tournament-panel" in html
    assert "/api/guild-tournament" in html
    assert "/api/season5-missions" in html
    assert "runGuildTournamentMatch" in html
    assert "loadGuildTournament" in html
    # No unresolved merge conflict markers in the shipped HUD.
    assert "<<<<<<<" not in html
    assert ">>>>>>>" not in html

    # The 3D WebGL renderer animates the guild raid arcs.
    for needle in (
        "data-raid-arcs",
        "rebuildRaidArcs",
        "updateRaidBatch",
        "raidArcs",
        "GUILD_COLORS",
    ):
        assert needle in html, needle
