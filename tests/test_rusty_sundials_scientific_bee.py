# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & integration tests for the Flagship BYOB Scientific Bee (``hivemind/bee-rusty-sundials``)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.boinc.real_client.boinc_server_scripts import validate_solver_sweep
from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import normalize_model_alias, resolve_producer_quorum_moe_state, tier_by_id
from hivemind.router.gateway import create_router_app
from hivemind.swarm.rusty_sundials_scientific_bee import (
    BEE_RUSTY_SUNDIALS_ID,
    BEE_RUSTY_SUNDIALS_PUBLIC_ID,
    BeeRetentionStore,
    SolverTaughtSurrogateExpert,
    analyze_physical_mathematical_problem,
    build_rusty_sundials_rescue_dataset,
    rescue_numeric_calculation,
    run_wave2_parameter_sweep_quorum,
    rusty_sundials_bee_descriptor,
    solve_wave2_method_of_lines_pde,
)


def test_analyze_physical_mathematical_problem_across_four_domains() -> None:
    astro = analyze_physical_mathematical_problem(
        "Kepler post-newtonian binary orbit secular energy drift in explicit RK4"
    )
    assert astro.domain == "astronomy"
    assert "Dual-RRK" in astro.recommended_rusty_sundials_crate or "arkode" in astro.recommended_rusty_sundials_crate

    fluid = analyze_physical_mathematical_problem(
        "3D dyadic Navier-Stokes turbulence enstrophy bound S_N^2 <= 2 Omega_N^3 and Obstruction D4"
    )
    assert fluid.domain == "fluid_mechanics"
    assert "EnstrophyProductionBound" in fluid.lean4_reference_theorem

    qf = analyze_physical_mathematical_problem(
        "Bose-Einstein condensate Projected Gross-Pitaevskii quantized vortex circulation"
    )
    assert qf.domain == "quantum_fluid"
    assert "qf-pgpe" in qf.recommended_rusty_sundials_crate

    gen = analyze_physical_mathematical_problem(
        "Stiff Hopfield-Ninio kinetic proofreading and DNA linking number Lk = Tw + Wr"
    )
    assert gen.domain == "genetics"
    assert "cvode" in gen.recommended_rusty_sundials_crate


def test_rescue_numeric_calculation_all_four_domains() -> None:
    astro_res = rescue_numeric_calculation("Rescue Kepler orbit", domain="astronomy", steps=80, dt=0.02)
    assert astro_res.rescued_succeeded is True
    assert astro_res.naive_failed is True
    assert astro_res.rescued_invariant_drift < 1e-12
    assert astro_res.improvement_factor > 1e9

    fluid_res = rescue_numeric_calculation("Rescue dyadic Navier-Stokes", domain="fluid_mechanics", steps=80, dt=0.01)
    assert fluid_res.rescued_succeeded is True
    assert fluid_res.naive_failed is True
    assert fluid_res.invariants_verified["mechanica_fluidorum_enstrophy_bound_satisfied"] is True
    assert fluid_res.invariants_verified["inviscid_flux_telescoping_residual"] < 1e-14

    qf_res = rescue_numeric_calculation("Rescue PGPE superfluid", domain="quantum_fluid", steps=80, dt=0.01)
    assert qf_res.rescued_succeeded is True
    assert qf_res.invariants_verified["k1_rel_norm_drift"] < 1e-10
    assert qf_res.invariants_verified["k4_plane_wave_max_err"] < 1e-10
    assert qf_res.invariants_verified["vortex_winding_quantization_residual"] < 1e-12

    gen_res = rescue_numeric_calculation("Rescue Hopfield-Ninio and DNA Lk", domain="genetics", steps=80, dt=0.01)
    assert gen_res.rescued_succeeded is True
    assert gen_res.naive_failed is True
    assert gen_res.invariants_verified["strict_positivity_preserved"] is True
    assert gen_res.invariants_verified["max_dna_linking_number_drift"] < 1e-12


def test_build_dataset_and_bee_retention_store(tmp_path: Path) -> None:
    ds_path = tmp_path / "rusty_sundials_dataset.jsonl"
    ds = build_rusty_sundials_rescue_dataset(ds_path)
    assert ds["bee_id"] == BEE_RUSTY_SUNDIALS_ID
    assert ds["num_records"] == 12
    assert ds_path.exists()
    lines = ds_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12

    store = BeeRetentionStore(base_dir=tmp_path / "retention")
    promoted = store.retain_verified_checkpoint(
        node_id="xavkal-test-node",
        backbone_model="Qwen/Qwen3.5-Coder-7B-Instruct-AWQ",
        lora_rank=32,
        adapter_weights_bytes=b"verified-sundials-weights",
        heldout_energy_e=4.5,
        parent_energy_e=25.0,
    )
    assert promoted["all_four_physics_domains_verified"] is True
    assert promoted["promoted_and_retained"] is True
    assert (tmp_path / "retention" / "latest_promoted.json").exists()

    rejected = store.retain_verified_checkpoint(
        node_id="xavkal-test-node",
        backbone_model="Qwen/Qwen3.5-Coder-7B-Instruct-AWQ",
        lora_rank=32,
        adapter_weights_bytes=b"regressed-sundials-weights",
        heldout_energy_e=30.0,
        parent_energy_e=25.0,
    )
    assert rejected["promoted_and_retained"] is False
    assert rejected["retained_dir"] is None


def test_catalog_and_gateway_endpoints_for_rusty_sundials_bee(tmp_path: Path) -> None:
    assert normalize_model_alias("hivemind/bee-rusty-sundials") == "bee-rusty-sundials"
    assert normalize_model_alias("rusty-sundials/numeric-7b") == "bee-rusty-sundials"
    tier = tier_by_id("bee-rusty-sundials")
    assert tier is not None
    assert tier.min_vram_gb == 6.0

    desc = rusty_sundials_bee_descriptor()
    assert desc["public_id"] == BEE_RUSTY_SUNDIALS_PUBLIC_ID

    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    resp_info = client.get("/v1/hivemind/bees/rusty-sundials")
    assert resp_info.status_code == 200
    info_json = resp_info.json()
    assert info_json["bee_id"] == "bee-rusty-sundials"
    assert set(info_json["supported_domains"]) == {
        "astronomy",
        "fluid_mechanics",
        "quantum_fluid",
        "genetics",
    }
    plan = info_json["season20_solver_bee_and_agent_plan"]
    assert plan["season"] == "HM-S20-SOLVER"
    assert len(plan["waves"]["wave_1_solver_bee_v0"]) == 9
    assert "rtx_2070_local_8gb" in plan["owner_agent_fleet"]
    assert "gcp_tesla_t4_16gb_fleet" in plan["owner_agent_fleet"]

    resp_rescue = client.post(
        "/v1/hivemind/bees/rusty-sundials/rescue",
        json={
            "prompt": "Rescue dyadic Navier-Stokes turbulence simulation from explicit blowup",
            "domain": "fluid_mechanics",
            "steps": 40,
            "dt": 0.01,
        },
    )
    assert resp_rescue.status_code == 200
    rescue_json = resp_rescue.json()
    assert rescue_json["domain"] == "fluid_mechanics"
    assert rescue_json["rescued_succeeded"] is True
    assert rescue_json["invariants_verified"]["mechanica_fluidorum_enstrophy_bound_satisfied"] is True


def test_wave2_solver_taught_surrogate_expert_and_fail_closed_fallback() -> None:
    expert = SolverTaughtSurrogateExpert()
    fit_summary = expert.fit_all_four_domains(steps=32, dt=0.01)
    assert set(fit_summary["domains_trained"]) == {"astronomy", "fluid_mechanics", "quantum_fluid", "genetics"}

    # Clean rollout: surrogate agrees with teacher within invariant tolerance -> accepted
    clean_verdict = expert.predict_and_verify(
        "Integrate Kepler binary orbit",
        domain="astronomy",
        steps=32,
        dt=0.01,
        force_surrogate_perturbation=0.0,
    )
    assert clean_verdict["surrogate_verified"] is True
    assert clean_verdict["surrogate_rejected_by_verifier"] is False
    assert clean_verdict["served_by"] == "surrogate_fast_path_verified"
    assert clean_verdict["final_verified_invariant_drift"] <= clean_verdict["max_surrogate_drift_tol"]
    assert clean_verdict["wilson_lower_bound"] > 0.0

    # Perturbed rollout: surrogate violates invariant tolerance -> fail-closed fallback to teacher
    fallback_verdict = expert.predict_and_verify(
        "Integrate Kepler binary orbit",
        domain="astronomy",
        steps=32,
        dt=0.01,
        force_surrogate_perturbation=0.5,
    )
    assert fallback_verdict["surrogate_verified"] is False
    assert fallback_verdict["surrogate_rejected_by_verifier"] is True
    assert fallback_verdict["served_by"] == "rusty_sundials_deterministic_teacher_fallback"
    assert fallback_verdict["final_verified_invariant_drift"] <= fallback_verdict["max_surrogate_drift_tol"]


def test_wave2_method_of_lines_pdes_and_quorum2_parameter_sweep() -> None:
    for pde_family in (
        "lane_emden_stellar",
        "viscous_burgers_imex",
        "pgpe_vortex_lattice",
        "fisher_kpp_genetic_wave",
    ):
        res = solve_wave2_method_of_lines_pde(pde_family, grid_points=24, steps=20, dt=0.005)
        assert res["grid_points"] == 24
        assert len(res["state_snapshots"]) >= 1
        assert float(res["invariant_drift"]) < 1e-2

    bundle = run_wave2_parameter_sweep_quorum(
        family="van_der_pol",
        parameter_grid=[0.1, 0.25, 0.5, 0.75],
        steps=24,
        dt=0.01,
    )
    verdict = validate_solver_sweep(
        bundle["spec"],
        bundle["replica_a"],
        host="cpu-ryzen-a",
        peer_meta=bundle["replica_b"],
    )
    assert verdict.valid is True
    assert "verified" in verdict.reason
    assert verdict.credit > 0.0

    # Tampered replica B should fail S-1 cross-CPU quorum agreement
    tampered_b = dict(bundle["replica_b"])
    tampered_b["sweep_results"] = [
        {**entry, "y": [[val + 0.25 for val in row] for row in entry["y"]]}
        for entry in bundle["replica_b"]["sweep_results"]
    ]
    bad_verdict = validate_solver_sweep(
        bundle["spec"],
        bundle["replica_a"],
        host="cpu-ryzen-a",
        peer_meta=tampered_b,
    )
    assert bad_verdict.valid is False
    assert "quorum" in bad_verdict.reason or "divergence" in bad_verdict.reason or "trajectory" in bad_verdict.reason


def test_wave2_gateway_surrogate_sweep_and_producer_quorum_scaling(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router_wave2.db", b"w" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    # 0 connected producers -> stage_0_standby
    q0 = client.get("/v1/hivemind/moe/quorum-status").json()
    assert q0["connected_producers"] == 0
    assert q0["stage_id"] == "stage_0_standby"
    assert q0["wave2_scientific_moe_active"] is False

    # Verify monotonic progression as 1, 2, 4, 8 producers connect
    s1 = resolve_producer_quorum_moe_state(1, total_vram_gb=8.0)
    s2 = resolve_producer_quorum_moe_state(2, total_vram_gb=16.0)
    s4 = resolve_producer_quorum_moe_state(4, total_vram_gb=32.0)
    s8 = resolve_producer_quorum_moe_state(8, total_vram_gb=64.0)
    assert s1["unlocked_moe_model_count"] < s2["unlocked_moe_model_count"] < s4["unlocked_moe_model_count"] < s8["unlocked_moe_model_count"]
    assert s2["wave2_scientific_moe_active"] is True
    assert s2["quorum_sweep_enabled"] is True
    assert "hivemind/moe-scientific-wave2" in s2["unlocked_moe_models"]
    assert "hivemind/moe-pde-fluid-quantum" in s4["unlocked_moe_models"]
    assert "hivemind/moe-glm45-air-science" in s8["unlocked_moe_models"]

    # Test /v1/hivemind/bees/rusty-sundials/surrogate
    surr_resp = client.post(
        "/v1/hivemind/bees/rusty-sundials/surrogate",
        json={"prompt": "Integrate dyadic Navier-Stokes shell model", "domain": "fluid_mechanics", "steps": 24},
    )
    assert surr_resp.status_code == 200
    assert surr_resp.json()["surrogate_accepted"] is True

    # Test /v1/hivemind/bees/rusty-sundials/sweep (both parameter sweep and MOL PDE)
    sweep_resp = client.post(
        "/v1/hivemind/bees/rusty-sundials/sweep",
        json={"domain": "quantum_fluid", "parameter_grid": [0.2, 0.4, 0.6], "steps": 20},
    )
    assert sweep_resp.status_code == 200
    assert sweep_resp.json()["quorum_verified"] is True

    pde_resp = client.post(
        "/v1/hivemind/bees/rusty-sundials/sweep",
        json={"pde_family": "viscous_burgers_imex", "grid_points": 24, "steps": 16},
    )
    assert pde_resp.status_code == 200
    assert pde_resp.json()["succeeded"] is True
