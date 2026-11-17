# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input validation & negative-control hardening tests for life_as_a_bee, steam_edition, scientific_discovery_game, and exoplanet_hunter_engine."""

from __future__ import annotations

import pytest

from hivemind.game.exoplanet_hunter_engine import (
    classify_planet_taxonomy,
    compute_earth_similarity_index,
    compute_four_astrophysical_laws,
    compute_habitable_zone_au,
    compute_oolite_planetfall_parameters,
    load_real_hf_exoplanet_catalog,
    query_nasa_exoplanet_archive_tap,
    synthesize_jwst_coronagraph_4bands,
    synthesize_jwst_transmission_spectrum,
    synthesize_transit_light_curve_and_bls,
)
from hivemind.game.scientific_discovery_game import (
    HiveMindScientificGameEngine,
    ScientificMissionSpec,
    ask_la_ruche_bee_copilot_live,
    build_antigravity_game_copilot_config,
    fold_exoplanet_transit_lightcurve,
    integrate_orbital_symplectic_comparison,
    render_scientific_discovery_game_html,
    run_e8_lattice_multikrum_mission,
)
from hivemind.game.steam_edition import (
    _generate_cyberpunk_hud_html,
    bearer_token_from_header,
    build_free_steam_edition_package,
    create_hack_miner_fastapi_app,
)
from hivemind.router.life_as_a_bee import (
    BEE_LIFECYCLE_PHASES,
    LIFE_AS_A_BEE_SCENARIOS,
    BeeLifecyclePhase,
    BeeScenarioResult,
    life_as_a_bee_payload,
    render_life_as_a_bee_section,
)


def test_life_as_a_bee_dataclasses_and_render_reject_invalid_inputs() -> None:
    assert len(BEE_LIFECYCLE_PHASES) == 4
    assert len(LIFE_AS_A_BEE_SCENARIOS) >= 15

    with pytest.raises(ValueError, match="step_number"):
        BeeLifecyclePhase(
            phase_id="P1",
            step_number=0,
            title_fr="FR",
            title_en="EN",
            time_window="W",
            key_metric="M",
            summary_fr="S",
            summary_en="S",
            protocol_endpoints=("/v1/models",),
            source="NOT_MEASURED",
        )
    with pytest.raises(ValueError, match="phase_id"):
        BeeLifecyclePhase(
            phase_id="",
            step_number=1,
            title_fr="FR",
            title_en="EN",
            time_window="W",
            key_metric="M",
            summary_fr="S",
            summary_en="S",
            protocol_endpoints=("/v1/models",),
            source="NOT_MEASURED",
        )

    base_scen = LIFE_AS_A_BEE_SCENARIOS[0].to_dict()
    base_scen["aliases"] = tuple(base_scen["aliases"])
    with pytest.raises(ValueError, match="vram_footprint_gb"):
        BeeScenarioResult(**{**base_scen, "vram_footprint_gb": -1.0})
    with pytest.raises(ValueError, match="prompt_tokens"):
        BeeScenarioResult(**{**base_scen, "prompt_tokens": True})

    with pytest.raises(ValueError, match="live_telemetry"):
        life_as_a_bee_payload(["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_life_as_a_bee_section(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_steam_edition_helpers_and_factory_reject_invalid_inputs() -> None:
    assert bearer_token_from_header(None) == ""
    assert bearer_token_from_header("Bearer secret-tok") == "secret-tok"
    with pytest.raises(ValueError, match="authorization"):
        bearer_token_from_header(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="game_report"):
        _generate_cyberpunk_hud_html([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="game_report"):
        build_free_steam_edition_package(game_report="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_root"):
        build_free_steam_edition_package(output_root=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="game_report"):
        create_hack_miner_fastapi_app(game_report="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="worker_token"):
        create_hack_miner_fastapi_app(worker_token=True)  # type: ignore[arg-type]


def test_scientific_discovery_game_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="base_scientific_pts"):
        ScientificMissionSpec(
            mission_id="M1",
            title="T",
            scientific_domain="D",
            specialist_bee="hivemind/bee-stars",
            unity_scene_script="S",
            invariant_gate="G",
            base_scientific_pts=0,
            description="Desc",
        )
    with pytest.raises(ValueError, match="gateway_url"):
        build_antigravity_game_copilot_config(gateway_url="")

    with pytest.raises(ValueError, match="steps"):
        integrate_orbital_symplectic_comparison(steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt"):
        integrate_orbital_symplectic_comparison(dt=-0.01)
    with pytest.raises(ValueError, match="eccentricity"):
        integrate_orbital_symplectic_comparison(eccentricity=1.0)

    with pytest.raises(ValueError, match="true_period_days"):
        fold_exoplanet_transit_lightcurve(true_period_days=0.0)
    with pytest.raises(ValueError, match="num_cadences"):
        fold_exoplanet_transit_lightcurve(num_cadences=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        fold_exoplanet_transit_lightcurve(seed=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_honest"):
        run_e8_lattice_multikrum_mission(num_honest=0)
    with pytest.raises(ValueError, match="num_byzantine"):
        run_e8_lattice_multikrum_mission(num_byzantine=-1)
    with pytest.raises(ValueError, match="scale"):
        run_e8_lattice_multikrum_mission(scale=float("nan"))

    with pytest.raises(ValueError, match="router_api_key"):
        ask_la_ruche_bee_copilot_live(prompt="test", router_api_key=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prompt"):
        ask_la_ruche_bee_copilot_live(prompt="   ", router_api_key="hm-test")

    eng = HiveMindScientificGameEngine()
    with pytest.raises(ValueError, match="mission_id"):
        eng.solve_mission("")
    with pytest.raises(ValueError, match="steps"):
        eng.solve_mission("MISSION-SCI-01-SYMPLECTIC-EXOPLANET", steps=0)
    with pytest.raises(ValueError, match="account_spend_usd"):
        eng.solve_mission("MISSION-SCI-01-SYMPLECTIC-EXOPLANET", account_spend_usd=-5.0)
    with pytest.raises(ValueError, match="booleans"):
        eng.solve_mission("MISSION-SCI-01-SYMPLECTIC-EXOPLANET", is_vac_banned="no")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="initial_report"):
        render_scientific_discovery_game_html("not-a-dict")  # type: ignore[arg-type]


def test_exoplanet_hunter_engine_rejects_invalid_inputs() -> None:
    esi = compute_earth_similarity_index(1.0, 288.0, 1.0)
    assert 0.95 <= esi <= 1.0
    with pytest.raises(ValueError, match="radius_rearth"):
        compute_earth_similarity_index(True, 288.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="teq_k"):
        compute_earth_similarity_index(1.0, -10.0, 1.0)

    hz = compute_habitable_zone_au(5778.0, 1.0)
    assert hz["hz_inner_au"] < hz["hz_outer_au"]
    with pytest.raises(ValueError, match="st_teff_k"):
        compute_habitable_zone_au(0.0, 1.0)

    with pytest.raises(ValueError, match="disposition"):
        classify_planet_taxonomy(1.0, 288.0, "", 500.0)
    with pytest.raises(ValueError, match="transit_depth_ppm"):
        classify_planet_taxonomy(1.0, 288.0, "CONFIRMED", -1.0)

    with pytest.raises(ValueError, match="period_days"):
        synthesize_transit_light_curve_and_bls(0.0, 500.0, 2.0, 15.0, "CONFIRMED")
    with pytest.raises(ValueError, match="trial_period_days"):
        synthesize_transit_light_curve_and_bls(5.0, 500.0, 2.0, 15.0, "CONFIRMED", trial_period_days=-1.0)
    with pytest.raises(ValueError, match="seed"):
        synthesize_transit_light_curve_and_bls(5.0, 500.0, 2.0, 15.0, "CONFIRMED", seed=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="radius_rearth"):
        synthesize_jwst_transmission_spectrum(0.0, 280.0, 500.0, "CONFIRMED")
    with pytest.raises(ValueError, match="disposition"):
        synthesize_jwst_transmission_spectrum(1.2, 280.0, 500.0, "   ")

    for fn in (compute_four_astrophysical_laws, synthesize_jwst_coronagraph_4bands, compute_oolite_planetfall_parameters):
        with pytest.raises(ValueError, match="target must be a dict"):
            fn([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="live_network"):
        query_nasa_exoplanet_archive_tap("yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_extra_per_category"):
        load_real_hf_exoplanet_catalog(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_extra_per_category"):
        load_real_hf_exoplanet_catalog(-1)
