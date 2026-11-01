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
    NEXTGEN_SCIENCE_BEE_PUBLIC_ID,
    BeeRetentionStore,
    NextGenMoEScienceBee,
    OwnerFleetCoServingOrchestrator,
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


def test_nextgen_service_10_families_heldout_energy_and_accuracy_certificate() -> None:
    import hashlib
    import math
    from hivemind.solverbee.families import (
        kepler_spec,
        logistic_spec,
        pendulum_spec,
        projectile_drag_spec,
    )
    from hivemind.solverbee.service import FAMILIES, validate_spec_bytes
    from hivemind.solverbee.verifier import canonical_json as canonical_spec_bytes, certify_achieved_accuracy, verify

    assert len(FAMILIES) == 10
    for builder_spec in (
        pendulum_spec(0.5, 0.0, [1.0, 2.0]),
        projectile_drag_spec(0.0, 0.0, 10.0, 10.0, [0.5, 1.0]),
        kepler_spec(1.0, 0.0, 0.0, 1.0, [1.0, 2.0]),
        logistic_spec(10.0, [1.0, 2.0], r=0.5, K=100.0),
    ):
        validated = validate_spec_bytes(canonical_spec_bytes(builder_spec))
        assert validated["problem"]["family"] in FAMILIES

    # Held-out model="pendulum" energy invariant + auto value at y0
    pend_spec = pendulum_spec(
        0.4,
        0.0,
        [1.0],
        g_over_l=9.81,
        damping=0.0,
    )
    pend_bytes = canonical_spec_bytes(pend_spec)
    # Exact energy conserving final state: theta=0.0, omega=sqrt(2*9.81*(1 - cos(0.4)))
    omega_exact = math.sqrt(2.0 * 9.81 * (1.0 - math.cos(0.4)))
    res_good = {
        "schema": "hivemind-solver-result/1",
        "id": pend_spec["id"],
        "status": "ok",
        "method": "bdf",
        "t": [1.0],
        "y": [[0.0, omega_exact]],
        "stats": {"n_steps": 50, "n_rhs_evals": 200},
        "input_sha256": hashlib.sha256(pend_bytes).hexdigest(),
    }
    pend_problem = {
        "schema": "hivemind-solver-problem/1",
        "spec": pend_spec,
        "reference": {"t": [1.0], "y": [[0.0, omega_exact]]},
        "invariants": [{"name": "energy", "kind": "energy", "model": "pendulum", "tol": 1e-4}],
    }
    cert_pend = verify(pend_bytes, res_good, problem=pend_problem, rerun=lambda _b: res_good)
    assert cert_pend["verdict"] == "VERIFIED", cert_pend
    inv_chk = next(c for c in cert_pend["checks"] if c["name"] == "invariants")
    assert inv_chk["status"] == "pass"

    # Richardson empirical accuracy certificate + long-horizon phase warning
    long_spec = kepler_spec(
        1.0,
        0.0,
        0.0,
        1.0,
        [200.0],
        mu=1.0,
        rtol=1e-6,
        atol=1e-9,
    )
    long_bytes = canonical_spec_bytes(long_spec)
    sha = hashlib.sha256(long_bytes).hexdigest()
    # Loose trajectory has O(1) phase shift vs tight trajectory while conserving energy
    loose_res = {
        "schema": "hivemind-solver-result/1",
        "id": long_spec["id"],
        "status": "ok",
        "method": "bdf",
        "t": [200.0],
        "y": [[0.6, 0.8, -0.8, 0.6]],
        "stats": {"n_steps": 1000, "n_rhs_evals": 4000},
        "input_sha256": sha,
    }
    tight_res = {
        "schema": "hivemind-solver-result/1",
        "id": long_spec["id"],
        "status": "ok",
        "method": "bdf",
        "t": [200.0],
        "y": [[0.8, 0.6, -0.6, 0.8]],
        "stats": {"n_steps": 5000, "n_rhs_evals": 20000},
        "input_sha256": sha,
    }
    long_problem = {
        "schema": "hivemind-solver-problem/1",
        "spec": long_spec,
        "reference": {"t": [200.0], "y": [[0.8005, 0.5993, -0.5993, 0.8005]]},
        "invariants": [{"name": "energy", "kind": "energy", "model": "kepler", "tol": 1e-3}],
    }
    cert = certify_achieved_accuracy(long_bytes, loose_res, tight_res, problem=long_problem)
    assert cert["schema"] == "hivemind-accuracy-cert/1"
    assert cert["invariants_conserved"] is True
    assert cert["long_horizon_phase_warning"] is True
    assert cert["achieved_rel_ratio"] > 1.0
    assert cert["empirical_order_p"] is not None and cert["empirical_order_p"] > 0.5


def test_nextgen_slot_compiler_and_tier_c_questions_match_ground_truth() -> None:
    import json
    from hivemind.solverbee.science_bee import (
        SlotBackend,
        compile_or_validate_proposal,
        echo_interpreted_problem,
        slot_formulate,
        specs_match,
    )

    s3_path = Path(__file__).resolve().parent.parent / "artifacts" / "experiments" / "S3_questions.json"
    s3_data = json.loads(s3_path.read_text(encoding="utf-8"))
    tier_c = [q for q in s3_data["questions"] if q["tier"] == "C"]
    assert len(tier_c) == 12

    backend = SlotBackend()
    for q in tier_c:
        raw_reply = backend.complete([{"role": "user", "content": q["text"]}])
        proposal = json.loads(raw_reply)
        assert proposal.get("abstain") is not True, f"SlotBackend unexpectedly abstained on {q['id']}"
        slot_obj = slot_formulate(q["text"])
        assert slot_obj is not None, f"slot_formulate returned None on {q['id']}"
        spec, slot_trace = compile_or_validate_proposal(slot_obj)
        assert slot_trace is not None
        echo = echo_interpreted_problem(spec, language=q.get("lang", "en"), slot_trace=slot_trace)
        assert echo["schema"] == "hivemind-problem-echo/1"
        assert specs_match(spec, q["expected_spec"]), (
            f"Tier C question {q['id']} spec mismatch:\ngot={spec}\nwant={q['expected_spec']}"
        )

    # Under-specified questions must still be cleanly declined by slot_formulate
    assert slot_formulate("Simulate a pendulum.") is None
    assert slot_formulate("Modélisez une épidémie SIR.") is None


def test_nextgen_moe_science_bee_and_owner_fleet_gateway(tmp_path: Path) -> None:
    assert normalize_model_alias(NEXTGEN_SCIENCE_BEE_PUBLIC_ID) == "bee-rusty-sundials"

    bee = NextGenMoEScienceBee()
    ans = bee.ask(
        "A simple pendulum of length 2.5 m is released from rest at 30 degrees (g = 9.81 m/s^2, no damping). "
        "Track theta and omega over 5 seconds (6 points)."
    )
    assert ans["formulated"] is True
    assert ans["verdict"] in ("AUDITED", "VERIFIED")
    assert ans["spec"]["problem"]["family"] == "pendulum"
    assert ans["slot_trace"] is not None
    assert ans["interpreted_problem_echo"]["schema"] == "hivemind-problem-echo/1"
    assert ans["accuracy_certificate"]["schema"] == "hivemind-accuracy-cert/1"
    assert ans["track_record"]["n"] >= 1

    # Owner RTX 2070 + T4 fleet status & S-1b cross-CPU reproducibility
    fleet = bee.fleet_orchestrator.fleet_status()
    assert fleet["measured"] is False
    assert fleet["gcp_tesla_t4_16gb"]["serving_state"] == "NOT_MEASURED"
    assert fleet["rtx_2070_local_8gb"]["configured_vram_cap_mib"] == 6656
    distill = bee.fleet_orchestrator.distill_teacher_to_rtx2070_student(
        teacher_rank=32,
        student_rank=16,
    )
    assert distill["wddm_compliant"] is True
    assert distill["t4_free_vram_compliant"] is True
    assert distill["spectral_energy_retention"] > 0.90
    s1b = bee.fleet_orchestrator.verify_s1b_cross_cpu_reproducibility(family="van_der_pol")
    assert s1b["passed"] is True

    # Test FastAPI gateway endpoints
    store = RouterStore(tmp_path / "router_nextgen.db", b"n" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    compile_resp = client.post(
        "/v1/hivemind/bees/science-bee/slot-compile",
        json={
            "schema": "hivemind-solver-slots/1",
            "slot_family": "pendulum",
            "slots": {
                "length_m": 2.5,
                "theta0_deg": 30.0,
                "omega0": 0.0,
                "t_out": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        },
    )
    assert compile_resp.status_code == 200
    compile_json = compile_resp.json()
    assert compile_json["spec"]["problem"]["family"] == "pendulum"
    assert len(compile_json["slot_trace"]["conversions"]) >= 1
    assert len(compile_json["slot_trace"]["derived_parameters"]) >= 1
    assert compile_json["interpreted_problem_echo"]["schema"] == "hivemind-problem-echo/1"

    ask_resp = client.post(
        "/v1/hivemind/bees/science-bee/ask",
        json={
            "question": (
                "An SEIR epidemic has R0 = 3.0, incubation period 5 days, and average infectious period 10 days. "
                "Start with 0.1% exposed, 0.1% infectious, 99.8% susceptible, 0% recovered. "
                "Report at t = [30, 60, 90, 120] days."
            )
        },
    )
    assert ask_resp.status_code == 200
    ask_json = ask_resp.json()
    assert ask_json["formulated"] is True
    assert ask_json["spec"]["problem"]["kind"] == "mass_action"
    assert ask_json["accuracy_certificate"]["schema"] == "hivemind-accuracy-cert/1"

    fleet_resp = client.get("/v1/hivemind/bees/owner-fleet-status")
    assert fleet_resp.status_code == 200
    fleet_json = fleet_resp.json()
    assert fleet_json["measured"] is False
    assert fleet_json["data_source"] == "CONFIGURED_POLICY_NOT_LIVE_TELEMETRY"
    assert fleet_json["latest_nocturnal_distillation"]["status"] == "NOT_RUN"
    assert fleet_json["s1b_cross_cpu_reproducibility"]["status"] == "NOT_RUN_CROSS_HOST"
    # Negative control: fields that only a real reading could supply must be absent.
    for node in ("rtx_2070_local_8gb", "gcp_tesla_t4_16gb"):
        for fabricated in ("live_models", "serving_active", "allocated_vram_mib", "free_vram_mib"):
            assert fabricated not in fleet_json[node]
    assert "spectral_energy_retention" not in fleet_json["latest_nocturnal_distillation"]

