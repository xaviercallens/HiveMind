"""Negative-control tests for Iteration 39 (IP-92, IP-93, IP-97 & Improvement Plan status sync)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.economy.international_launch_model import (
    SwarmLiveBaseline,
    forecast_200plus_node_international_launch,
)
from hivemind.game import fat_steam_astro_builder as builder_mod


def test_ip92_unbacked_deficit_kappa_star_and_prosumer_conversion_kappa() -> None:
    """IP-92: unbacked_deficit_kappa_star must be 0.0 when producers alone cover demand, and >0 under deficit."""
    surplus = forecast_200plus_node_international_launch(
        target_total_nodes=200,
        visitor_count=100,
        dedicated_producers=80,
        prosumer_nodes=60,
        consumer_nodes=60,
    )
    tp_surplus = surplus["throughput_and_compute_power"]
    assert tp_surplus["base_producer_only_tok_per_s"] >= tp_surplus["sustained_demand_tok_per_s"]
    stab_surplus = surplus["balancing_and_latency_sla"]["economic_and_epidemic_stabilizers"]
    assert stab_surplus["unbacked_deficit_kappa_star"] == 0.0

    calm = forecast_200plus_node_international_launch(
        target_total_nodes=265,
        visitor_count=200,
    )
    stab_calm = calm["balancing_and_latency_sla"]["economic_and_epidemic_stabilizers"]
    swarm_calm = calm["forecast_200plus_swarm"]
    expected_pros_kappa = round(
        swarm_calm["prosumer_nodes"] / max(1, swarm_calm["prosumer_nodes"] + swarm_calm["consumer_nodes"]),
        4,
    )
    assert stab_calm["prosumer_conversion_kappa"] == expected_pros_kappa
    assert 0.0 < stab_calm["unbacked_deficit_kappa_star"] < 0.0220

    overload = forecast_200plus_node_international_launch(
        target_total_nodes=265,
        visitor_count=900_000,
    )
    stab_over = overload["balancing_and_latency_sla"]["economic_and_epidemic_stabilizers"]
    assert overload["ready_for_200plus_international_launch"] is False
    assert "> 0.78" in overload["verdict_summary"]
    assert stab_over["unbacked_deficit_kappa_star"] > stab_calm["unbacked_deficit_kappa_star"]
    assert stab_over["critical_conversion_threshold_kappa_star"] >= stab_over["unbacked_deficit_kappa_star"]


def test_ip93_pioneer_location_basis_and_backfill_coordinates() -> None:
    """IP-93: pioneer_location_basis distinguishes estimated vs caller-supplied coordinates across all pioneers."""
    default_rep = forecast_200plus_node_international_launch(target_total_nodes=265, visitor_count=2400)
    globe_default = default_rep["hiveminers_on_globe"]
    assert globe_default["pioneer_location_basis"] == "estimated_regional_distribution"
    pioneers_default = [n for n in globe_default["nodes"] if n["is_current_pioneer"]]
    assert len(pioneers_default) == SwarmLiveBaseline().total_accounts
    assert all(n["location_estimated"] is True for n in pioneers_default)

    # Partial coordinates (only 10 supplied) must still report estimated_regional_distribution
    partial_coords = [
        {"city": f"PioneerCity-{i}", "lat": 48.0 + i * 0.1, "lon": 2.0 + i * 0.1, "location_estimated": False}
        for i in range(10)
    ]
    partial_rep = forecast_200plus_node_international_launch(
        target_total_nodes=265,
        visitor_count=2400,
        pioneer_coordinates=partial_coords,
    )
    assert partial_rep["hiveminers_on_globe"]["pioneer_location_basis"] == "estimated_regional_distribution"

    # Full caller-supplied coordinates for all 59 pioneers (including any backfilled pioneer slots)
    total_pioneers = SwarmLiveBaseline().total_accounts
    full_coords = [
        {"city": f"VerifiedCity-{i:02d}", "lat": round(10.0 + i * 0.5, 4), "lon": round(-20.0 + i * 0.5, 4), "location_estimated": False}
        for i in range(total_pioneers)
    ]
    full_rep = forecast_200plus_node_international_launch(
        target_total_nodes=265,
        visitor_count=2400,
        pioneer_coordinates=full_coords,
    )
    globe_full = full_rep["hiveminers_on_globe"]
    pioneers_full = [n for n in globe_full["nodes"] if n["is_current_pioneer"]]
    assert len(pioneers_full) == total_pioneers
    assert all(n["location_estimated"] is False for n in pioneers_full)
    assert globe_full["pioneer_location_basis"] == "caller_supplied_coordinates"
    assert {n["city"] for n in pioneers_full} == {c["city"] for c in full_coords}


def test_ip97_fat_steam_builder_acf_and_publish_script_free_of_hardcoded_owner_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-97: Generated appmanifest_*.acf and Publish_Steamworks_*.bat must not contain hardcoded personal Steam IDs."""
    monkeypatch.delenv("STEAM_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("STEAM_PERSONA_NAME", raising=False)
    monkeypatch.delenv("STEAM_LAST_OWNER_ID64", raising=False)

    monkeypatch.setattr(builder_mod, "PROJECT_ROOT", tmp_path)
    steam_common = tmp_path / "SteamLibrary" / "steamapps" / "common"
    steam_config = tmp_path / "Steam" / "userdata" / "1" / "config"
    steam_common.mkdir(parents=True, exist_ok=True)
    steam_config.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(builder_mod, "STEAM_LIBRARY_COMMON", steam_common)
    monkeypatch.setattr(builder_mod, "STEAM_USERDATA_CONFIG", steam_config)
    monkeypatch.setattr(
        builder_mod,
        "synthesize_gemini_or_gcp_free_tts",
        lambda **kw: {"ok": True, "cached_path": str(tmp_path / "dummy.mp3")},
    )

    res = builder_mod.build_fat_steam_exoplanet_client(install_to_local_steam_library=True)
    assert "76561198120474852" not in res["steam_account"]
    assert "xavakal" not in res["steam_account"]
    assert "hivemind_local" in res["steam_account"]

    acf_text = Path(res["steam_acf_manifest_path"]).read_text(encoding="utf-8")
    assert "76561198120474852" not in acf_text
    assert "xavakal" not in acf_text
    assert '"accountname"\t\t"hivemind_local"' in acf_text
    assert '"LastOwner"\t\t"0"' in acf_text

    bat_text = Path(res["steamworks_publish_bat"]).read_text(encoding="utf-8")
    assert "76561198120474852" not in bat_text
    assert "xavakal" not in bat_text


def test_improvement_plans_2026_11_04_and_05_document_status_on_main() -> None:
    """Verify IMPROVEMENT_PLAN_2026-11-04.md and IMPROVEMENT_PLAN_2026-11-05.md document closed status on main."""
    repo_root = Path(__file__).resolve().parents[1]
    plan_04 = (repo_root / "docs" / "IMPROVEMENT_PLAN_2026-11-04.md").read_text(encoding="utf-8")
    plan_05 = (repo_root / "docs" / "IMPROVEMENT_PLAN_2026-11-05.md").read_text(encoding="utf-8")
    assert "Status on `main` (2026-11-07):" in plan_04
    assert "Status on `main` (2026-11-07):" in plan_05
    assert "#704" in plan_04
    assert "#705" in plan_05
