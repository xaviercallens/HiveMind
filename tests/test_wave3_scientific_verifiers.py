# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the 5 Wave 3 Scientific MoE Bee deterministic fail-closed verifiers."""

from __future__ import annotations

import math

import numpy as np

from hivemind.compress.lattice_quantizer import nearest_a2, nearest_e8
from hivemind.solverbee.families import named_spec
from hivemind.solverbee.wave3_verifiers import (
    CERT_SCHEMA,
    verify_astrophysics_transit_and_orbit,
    verify_genetic_elenchus_thermodynamics,
    verify_numeric_solver_trajectory,
    verify_quantum_fluid_relaxation,
    verify_topology_tda_and_lattice,
)


def test_verify_quantum_fluid_relaxation_positive_and_negative_controls() -> None:
    cert = verify_quantum_fluid_relaxation(
        energy_init=-120.5,
        energy_final=-125.9,
        max_free_force=2.5e-11,
        v_prime_scale=1.0,
        kappa=1.5,
        reported_defect_charge_sum=0,
    )
    assert cert["schema"] == CERT_SCHEMA
    assert cert["domain"] == "science_quantum_fluid"
    assert cert["verdict"] == "VERIFIED"
    assert cert["reasons"] == []

    # Negative control 1: energy increased during relaxation
    bad_energy = verify_quantum_fluid_relaxation(
        energy_init=-125.9,
        energy_final=-120.5,
        max_free_force=2.5e-11,
        v_prime_scale=1.0,
        kappa=1.5,
    )
    assert bad_energy["verdict"] == "FAILED"
    assert any("energy did not decrease" in r for r in bad_energy["reasons"])

    # Negative control 2: residual force not converged
    bad_force = verify_quantum_fluid_relaxation(
        energy_init=-120.5,
        energy_final=-125.9,
        max_free_force=1e-4,
        v_prime_scale=1.0,
        kappa=1.5,
    )
    assert bad_force["verdict"] == "FAILED"

    # Negative control 3: topological defect charge neutrality violated
    bad_charge = verify_quantum_fluid_relaxation(
        energy_init=-120.5,
        energy_final=-125.9,
        max_free_force=2.5e-11,
        v_prime_scale=1.0,
        kappa=1.5,
        reported_defect_charge_sum=1,
    )
    assert bad_charge["verdict"] == "FAILED"

    # Non-finite input -> UNVERIFIED (fail-closed)
    unverified = verify_quantum_fluid_relaxation(
        energy_init=math.nan,
        energy_final=-125.9,
        max_free_force=2.5e-11,
        v_prime_scale=1.0,
        kappa=1.5,
    )
    assert unverified["verdict"] == "UNVERIFIED"


def test_verify_astrophysics_transit_and_orbit_positive_and_negative_controls() -> None:
    # Earth-twin around Sun-twin: 84 ppm transit depth
    cert = verify_astrophysics_transit_and_orbit(
        r_planet_earth=1.0,
        r_star_sun=1.0,
        reported_depth_ppm=84.0,
        energy_series=[-0.5, -0.5000001, -0.4999999],
        ang_mom_series=[1.0, 1.0000002, 0.9999998],
    )
    assert cert["verdict"] == "VERIFIED"
    assert cert["domain"] == "science_astrophysics"

    # Negative control 1: wrong transit depth
    bad_depth = verify_astrophysics_transit_and_orbit(
        r_planet_earth=2.0,
        r_star_sun=1.0,
        reported_depth_ppm=84.0,  # should be 336 ppm
        energy_series=[-0.5, -0.5],
        ang_mom_series=[1.0, 1.0],
    )
    assert bad_depth["verdict"] == "FAILED"

    # Negative control 2: symplectic energy drift
    bad_drift = verify_astrophysics_transit_and_orbit(
        r_planet_earth=1.0,
        r_star_sun=1.0,
        reported_depth_ppm=84.0,
        energy_series=[-0.5, -0.55],
        ang_mom_series=[1.0, 1.0],
    )
    assert bad_drift["verdict"] == "FAILED"


def test_verify_genetic_elenchus_thermodynamics_positive_and_negative_controls() -> None:
    vec8 = [0.4, -0.3, 0.6, 0.1, -0.2, 0.5, 0.0, -0.1]
    opt_e8 = nearest_e8(np.asarray(vec8, dtype=float).reshape(1, 8))[0].tolist()
    cert = verify_genetic_elenchus_thermodynamics(
        prob_vector=[0.25, 0.5, 0.25],
        mean_work_kbt=3.4,
        delta_f_kbt=2.1,
        dicodon_vec_8d=vec8,
        reported_e8_codeword=opt_e8,
    )
    assert cert["verdict"] == "VERIFIED"
    assert cert["domain"] == "science_genetics"

    # Negative control 1: Jarzynski-Crooks second law violated (<W> < delta_F)
    bad_jc = verify_genetic_elenchus_thermodynamics(
        prob_vector=[0.25, 0.5, 0.25],
        mean_work_kbt=1.2,
        delta_f_kbt=2.1,
        dicodon_vec_8d=vec8,
    )
    assert bad_jc["verdict"] == "FAILED"

    # Negative control 2: probability sum != 1
    bad_prob = verify_genetic_elenchus_thermodynamics(
        prob_vector=[0.25, 0.5, 0.5],
        mean_work_kbt=3.4,
        delta_f_kbt=2.1,
        dicodon_vec_8d=vec8,
    )
    assert bad_prob["verdict"] == "FAILED"

    # Negative control 3: suboptimal E8 codeword
    bad_e8 = verify_genetic_elenchus_thermodynamics(
        prob_vector=[0.25, 0.5, 0.25],
        mean_work_kbt=3.4,
        delta_f_kbt=2.1,
        dicodon_vec_8d=vec8,
        reported_e8_codeword=[2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert bad_e8["verdict"] == "FAILED"


def test_verify_topology_tda_and_lattice_positive_and_negative_controls() -> None:
    # 2-torus Betti numbers [1, 2, 1] -> Euler characteristic 1 - 2 + 1 = 0
    pt2 = [0.8, 0.5]
    opt_a2 = nearest_a2(np.asarray(pt2, dtype=float).reshape(1, 2))[0].tolist()
    cert = verify_topology_tda_and_lattice(
        betti_numbers=[1, 2, 1],
        reported_euler_char=0,
        point=pt2,
        lattice_name="A2",
        riesz_s=3.5,
        spatial_dim=2,
        reported_nearest_point=opt_a2,
    )
    assert cert["verdict"] == "VERIFIED"
    assert cert["domain"] == "science_topology"

    # Negative control 1: Euler-Poincare mismatch
    bad_chi = verify_topology_tda_and_lattice(
        betti_numbers=[1, 2, 1],
        reported_euler_char=2,
        point=pt2,
        lattice_name="A2",
        riesz_s=3.5,
        spatial_dim=2,
    )
    assert bad_chi["verdict"] == "FAILED"

    # Negative control 2: Riesz exponent s <= spatial_dim (diverges on 2D A2 lattice)
    bad_riesz = verify_topology_tda_and_lattice(
        betti_numbers=[1, 0, 1],
        reported_euler_char=2,
        point=pt2,
        lattice_name="A2",
        riesz_s=2.0,
        spatial_dim=2,
    )
    assert bad_riesz["verdict"] == "FAILED"

    # Negative control 3: D4 non-member (odd sum of integers)
    bad_d4 = verify_topology_tda_and_lattice(
        betti_numbers=[1, 0, 1],
        reported_euler_char=2,
        point=[0.1, 0.1, 0.1, 0.1],
        lattice_name="D4",
        riesz_s=5.0,
        spatial_dim=4,
        reported_nearest_point=[1.0, 0.0, 0.0, 0.0],
    )
    assert bad_d4["verdict"] == "FAILED"


def test_verify_numeric_solver_trajectory_positive_and_negative_controls() -> None:
    spec = named_spec(
        "harmonic",
        {"omega": 2.0, "zeta": 0.0},
        [1.0, 0.0],
        [0.5, 1.0],
        spec_id="test-harmonic",
    )
    y_c = [[0.54030230586, -1.6829419696], [-0.41614683654, -1.81859485365]]
    y_f = [[0.54030230587, -1.68294196961], [-0.41614683655, -1.81859485366]]
    cert = verify_numeric_solver_trajectory(spec, y_c, y_f, invariant_drift=1e-9)
    assert cert["verdict"] == "VERIFIED"
    assert cert["domain"] == "science_numeric_solver"

    # Negative control 1: coarse/fine disagreement
    y_bad = [[0.55, -1.68], [-0.41, -1.81]]
    bad_tight = verify_numeric_solver_trajectory(spec, y_bad, y_f, invariant_drift=1e-9)
    assert bad_tight["verdict"] == "FAILED"

    # Negative control 2: invalid spec grammar
    bad_spec = dict(spec, schema="wrong-schema")
    bad_grammar = verify_numeric_solver_trajectory(bad_spec, y_c, y_f)
    assert bad_grammar["verdict"] == "FAILED"
