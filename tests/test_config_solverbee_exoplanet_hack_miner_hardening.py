# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for config, solverbee verifiers, exoplanet hunter, and hf_hack_miner (IP-174)."""

from __future__ import annotations

import pytest

from hivemind.config import detect_gpu_backend, detect_rocm_gcn_arch
from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app
from hivemind.game.exoplanet_hunter_engine import synthesize_gemini_or_gcp_free_tts
from hivemind.game.hf_hack_miner import (
    execute_interactive_bounty_probe,
    execute_interactive_hf_shard_scan,
    execute_interactive_joined_campaign,
    execute_interactive_lora_micro_forge,
    run_hf_hack_mining_game,
)
from hivemind.solverbee.verifier import certify_achieved_accuracy
from hivemind.solverbee.wave3_verifiers import (
    verify_astrophysics_transit_and_orbit,
    verify_genetic_elenchus_thermodynamics,
    verify_quantum_fluid_relaxation,
    verify_topology_tda_and_lattice,
)


def test_config_gpu_backend_and_rocm_arch_reject_booleans() -> None:
    with pytest.raises(ValueError, match="boolean"):
        detect_gpu_backend(True)
    with pytest.raises(ValueError, match="boolean"):
        detect_rocm_gcn_arch(False)
    assert detect_gpu_backend("cpu") == "cpu"


def test_solverbee_verifier_and_wave3_verifiers_boundary_validation() -> None:
    with pytest.raises(ValueError, match="spec_bytes"):
        certify_achieved_accuracy(b"", {}, {})
    with pytest.raises(ValueError, match="loose_result"):
        certify_achieved_accuracy(b"{}", None, {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="spec_id"):
        verify_quantum_fluid_relaxation(1.0, 0.5, 1e-12, 1.0, 1.0, spec_id="  ")
    with pytest.raises(ValueError, match="spec_id"):
        verify_astrophysics_transit_and_orbit(1.0, 1.0, 84.0, [1.0, 1.0], [1.0, 1.0], spec_id="")
    with pytest.raises(ValueError, match="spec_id"):
        verify_genetic_elenchus_thermodynamics([1.0], 1.0, 0.5, [0.0] * 8, spec_id="")
    with pytest.raises(ValueError, match="spec_id"):
        verify_topology_tda_and_lattice([1, 2, 1], 0, [0.0, 0.0], spec_id="")

    cert = verify_quantum_fluid_relaxation(
        10.0,
        5.0,
        1e-12,
        1.0,
        1.5,
        reported_defect_charge_sum=0,
        spec_id="qf-ok",
    )
    assert cert["verdict"] == "VERIFIED"


def test_exoplanet_hunter_cockpit_and_engine_boundary_validation() -> None:
    with pytest.raises(TypeError, match="portal_secret"):
        create_exoplanet_hunter_fastapi_app(portal_secret=12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="require_portal_secret"):
        create_exoplanet_hunter_fastapi_app(require_portal_secret="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="text"):
        synthesize_gemini_or_gcp_free_tts(None)  # type: ignore[arg-type]


def test_hf_hack_miner_interactive_functions_boundary_validation() -> None:
    with pytest.raises(ValueError, match="model_id"):
        run_hf_hack_mining_game(model_id="")
    with pytest.raises(ValueError, match="lora_rank"):
        run_hf_hack_mining_game(lora_rank=0)
    with pytest.raises(ValueError, match="inner_steps"):
        run_hf_hack_mining_game(inner_steps=0)
    with pytest.raises(ValueError, match="seq_len"):
        run_hf_hack_mining_game(seq_len=0)
    with pytest.raises(ValueError, match="season"):
        run_hf_hack_mining_game(season="")
    with pytest.raises(TypeError, match="build_steam_package"):
        run_hf_hack_mining_game(build_steam_package="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gamer_tag"):
        execute_interactive_hf_shard_scan(gamer_tag="")
    with pytest.raises(ValueError, match="dataset_id"):
        execute_interactive_hf_shard_scan(dataset_id="")

    with pytest.raises(ValueError, match="gamer_tag"):
        execute_interactive_bounty_probe(gamer_tag="")
    with pytest.raises(ValueError, match="prompt"):
        execute_interactive_bounty_probe(prompt="   ")
    with pytest.raises(TypeError, match="persist_leaderboard"):
        execute_interactive_bounty_probe(persist_leaderboard="no")  # type: ignore[arg-type]

    probe = execute_interactive_bounty_probe(
        gamer_tag="TestMiner",
        prompt="Write a Python function to compute block rewards",
        persist_leaderboard=False,
    )
    assert probe["gamer_tag"] == "TestMiner"
    assert probe["total_pts_awarded"] > 0

    with pytest.raises(ValueError, match="gamer_tag"):
        execute_interactive_lora_micro_forge(gamer_tag="")
    with pytest.raises(ValueError, match="contract_id"):
        execute_interactive_lora_micro_forge(contract_id="")
    with pytest.raises(TypeError, match="booleans"):
        execute_interactive_lora_micro_forge(require_cuda="no")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lead_gamer_tag"):
        execute_interactive_joined_campaign(lead_gamer_tag="")
    with pytest.raises(ValueError, match="campaign_id"):
        execute_interactive_joined_campaign(campaign_id="")
    with pytest.raises(ValueError, match="inner_steps"):
        execute_interactive_joined_campaign(inner_steps=0)
