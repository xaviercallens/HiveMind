# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Issue #499: self-reports must not mint points or HIVE."""

from __future__ import annotations

from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.game.hf_hack_miner import (
    GuildTournamentEngine,
    execute_interactive_bounty_probe,
    execute_interactive_lora_micro_forge,
)
from hivemind.game.scientific_discovery_game import HiveMindScientificGameEngine


def test_tournament_override_does_not_change_champion() -> None:
    engine = GuildTournamentEngine()
    baseline = engine.build_state()
    loser = baseline["standings"][-1]["guild_id"]
    state = engine.register_squad_update(
        loser,
        multi_domain_holdout_perplexity=0.001,
        speculative_tokens_per_sec=500.0,
        gamer_tag="attacker",
    )
    assert state["champion_guild_id"] == baseline["champion_guild_id"]
    assert state["standings"] == baseline["standings"]


def test_forge_lora_mints_nothing_without_training(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    out = execute_interactive_lora_micro_forge()
    assert out["spot_check_passed"] is False
    assert out["hive_minted"] == 0
    assert out["total_pts_awarded"] == 0


def test_solve_bounty_awards_zero_points() -> None:
    out = execute_interactive_bounty_probe(prompt="def python code sort algorithm return")
    assert out["total_pts_awarded"] == 0


def test_sybil_fields_from_body_not_trusted(monkeypatch) -> None:
    monkeypatch.delenv("STEAM_WEB_API_KEY", raising=False)
    res = HiveMindScientificGameEngine().solve_mission("MISSION-SCI-02-E8-LATTICE-QUORUM")
    assert res["steam_sybil_verified"] is not True
    assert res["scientific_pts_awarded"] == 0


def test_bridge_require_web_api_rejects_self_signed_ticket() -> None:
    bridge = SteamworksSDKBridge()
    ticket = bridge.get_auth_session_ticket("76561198000000001", account_spend_usd=1e9)
    ok, _ = bridge.verify_sybil_resistance(ticket, require_web_api=True)
    assert ok is False
