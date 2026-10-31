# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Wave 3 Scientific MoE Bee deterministic fail-closed verifiers (Rule A3: 0 wrong answers labeled VERIFIED).

Provides independent mathematical, physical, and topological verification certificates (`hivemind-solver-cert/1`)
for all five Wave 3 Scientific MoE Bees:
  1. `bee_quantum_fluid` (`verify_quantum_fluid_relaxation`)
  2. `bee_astrophysics_k3` (`verify_astrophysics_transit_and_orbit`)
  3. `bee_genetic_elenchus` (`verify_genetic_elenchus_thermodynamics`)
  4. `bee_topology_tda_lean` (`verify_topology_tda_and_lattice`)
  5. `bee_numeric_solver_sundials` (`verify_numeric_solver_trajectory`)
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hivemind.compress.lattice_quantizer import (
    is_lattice_point,
    nearest_a2,
    nearest_dn,
    nearest_e8,
    nearest_zn,
)
from hivemind.solverbee.science_bee import SpecInvalid, validate_spec

CERT_SCHEMA = "hivemind-solver-cert/1"
EARTH_SUN_TRANSIT_PPM_SCALE = 84.0


def _is_finite_num(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _cert(
    spec_id: str,
    domain: str,
    checks: List[Dict[str, Any]],
    reasons: List[str],
    payload_for_hash: Dict[str, Any],
    *,
    unverified: bool = False,
) -> Dict[str, Any]:
    raw_bytes = json.dumps(payload_for_hash, sort_keys=True, default=str).encode("utf-8")
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if unverified:
        verdict = "UNVERIFIED"
    elif any(c.get("status") == "fail" for c in checks):
        verdict = "FAILED"
    else:
        verdict = "VERIFIED"
    return {
        "schema": CERT_SCHEMA,
        "id": str(spec_id),
        "domain": domain,
        "verdict": verdict,
        "input_sha256": input_sha256,
        "checks": checks,
        "reasons": reasons,
    }


def verify_quantum_fluid_relaxation(
    energy_init: float,
    energy_final: float,
    max_free_force: float,
    v_prime_scale: float,
    kappa: float,
    *,
    reported_defect_charge_sum: int = 0,
    force_rel_tol: float = 1e-9,
    spec_id: str = "wave3-qf",
) -> Dict[str, Any]:
    """Fail-closed verifier for `bee_quantum_fluid` (2D Yukawa crystal / PGPE relaxation).

    Verifies:
      1. Positive screening parameter `kappa > 0` and finite force scale `|V'(a)| > 0`.
      2. Strictly decreasing energy `E_final < E_init` (or `E_final <= E_init` at exact equilibrium).
      3. Force convergence `0 <= max_free_force <= force_rel_tol * |V'(a)|`.
      4. Topological defect charge neutrality `sum(q_i) == 0` under periodic/neutral boundary conditions.
    """
    payload = {
        "energy_init": energy_init,
        "energy_final": energy_final,
        "max_free_force": max_free_force,
        "v_prime_scale": v_prime_scale,
        "kappa": kappa,
        "reported_defect_charge_sum": reported_defect_charge_sum,
    }
    if not all(_is_finite_num(v) for v in (energy_init, energy_final, max_free_force, v_prime_scale, kappa)):
        return _cert(spec_id, "science_quantum_fluid", [], ["non-finite quantum fluid parameters"], payload, unverified=True)
    if isinstance(reported_defect_charge_sum, bool) or not isinstance(reported_defect_charge_sum, int):
        return _cert(spec_id, "science_quantum_fluid", [], ["reported_defect_charge_sum must be an int"], payload, unverified=True)

    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    kappa_ok = float(kappa) > 0.0 and abs(float(v_prime_scale)) > 0.0
    checks.append({"name": "screening_and_scale", "status": "pass" if kappa_ok else "fail", "kappa": float(kappa)})
    if not kappa_ok:
        reasons.append("kappa and |v_prime_scale| must be strictly positive")

    delta_e = float(energy_final) - float(energy_init)
    energy_ok = delta_e < 0.0
    checks.append({"name": "energy_monotonic_descent", "status": "pass" if energy_ok else "fail", "delta_energy": delta_e})
    if not energy_ok:
        reasons.append(f"energy did not decrease during relaxation (delta_E={delta_e:.6g} >= 0)")

    force_bound = float(force_rel_tol) * abs(float(v_prime_scale))
    force_ok = 0.0 <= float(max_free_force) <= force_bound
    checks.append(
        {
            "name": "damped_md_force_equilibrium",
            "status": "pass" if force_ok else "fail",
            "max_free_force": float(max_free_force),
            "bound": force_bound,
        }
    )
    if not force_ok:
        reasons.append(f"max_free_force={max_free_force:.3e} exceeds bound {force_bound:.3e}")

    charge_ok = int(reported_defect_charge_sum) == 0
    checks.append({"name": "topological_charge_neutrality", "status": "pass" if charge_ok else "fail", "charge_sum": int(reported_defect_charge_sum)})
    if not charge_ok:
        reasons.append(f"topological defect charge sum is {reported_defect_charge_sum}, expected 0")

    return _cert(spec_id, "science_quantum_fluid", checks, reasons, payload)


def verify_astrophysics_transit_and_orbit(
    r_planet_earth: float,
    r_star_sun: float,
    reported_depth_ppm: float,
    energy_series: Sequence[float],
    ang_mom_series: Sequence[float],
    *,
    depth_rel_tol: float = 0.05,
    conservation_rel_tol: float = 1e-4,
    spec_id: str = "wave3-astro",
) -> Dict[str, Any]:
    """Fail-closed verifier for `bee_astrophysics_k3` (Exoplanet transit depth & symplectic orbit conservation).

    Verifies:
      1. Transit depth law `delta_F/F = 84.0 * (R_p/R_earth)^2 / (R_star/R_sun)^2` ppm within `depth_rel_tol`.
      2. Hamiltonian energy conservation `max |H(t) - H(0)| / |H(0)| <= conservation_rel_tol`.
      3. Angular momentum conservation `max |L(t) - L(0)| / |L(0)| <= conservation_rel_tol`.
    """
    payload = {
        "r_planet_earth": r_planet_earth,
        "r_star_sun": r_star_sun,
        "reported_depth_ppm": reported_depth_ppm,
        "energy_series": list(energy_series) if isinstance(energy_series, (list, tuple)) else None,
        "ang_mom_series": list(ang_mom_series) if isinstance(ang_mom_series, (list, tuple)) else None,
    }
    if not all(_is_finite_num(v) for v in (r_planet_earth, r_star_sun, reported_depth_ppm)):
        return _cert(spec_id, "science_astrophysics", [], ["non-finite transit parameters"], payload, unverified=True)
    if (
        not isinstance(energy_series, (list, tuple))
        or not isinstance(ang_mom_series, (list, tuple))
        or len(energy_series) < 2
        or len(ang_mom_series) < 2
        or not all(_is_finite_num(v) for v in energy_series)
        or not all(_is_finite_num(v) for v in ang_mom_series)
    ):
        return _cert(spec_id, "science_astrophysics", [], ["energy_series and ang_mom_series require >= 2 finite numbers"], payload, unverified=True)

    rp, rs, depth = float(r_planet_earth), float(r_star_sun), float(reported_depth_ppm)
    if rp <= 0.0 or rs <= 0.0 or depth <= 0.0:
        return _cert(spec_id, "science_astrophysics", [{"name": "positive_radii", "status": "fail"}], ["radii and depth must be > 0"], payload)

    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    expected_ppm = EARTH_SUN_TRANSIT_PPM_SCALE * ((rp / rs) ** 2)
    rel_err_ppm = abs(depth - expected_ppm) / max(expected_ppm, 1e-12)
    transit_ok = rel_err_ppm <= float(depth_rel_tol)
    checks.append(
        {
            "name": "transit_depth_law",
            "status": "pass" if transit_ok else "fail",
            "expected_ppm": expected_ppm,
            "reported_ppm": depth,
            "rel_error": rel_err_ppm,
        }
    )
    if not transit_ok:
        reasons.append(f"transit depth {depth:.2f} ppm disagrees with physical law {expected_ppm:.2f} ppm (rel_err={rel_err_ppm:.4f})")

    h0 = float(energy_series[0])
    max_dh = max(abs(float(h) - h0) for h in energy_series) / max(abs(h0), 1e-12)
    h_ok = max_dh <= float(conservation_rel_tol)
    checks.append({"name": "symplectic_energy_conservation", "status": "pass" if h_ok else "fail", "max_rel_drift": max_dh})
    if not h_ok:
        reasons.append(f"orbital energy drift {max_dh:.3e} exceeds {conservation_rel_tol:.3e}")

    l0 = float(ang_mom_series[0])
    max_dl = max(abs(float(l_val) - l0) for l_val in ang_mom_series) / max(abs(l0), 1e-12)
    l_ok = max_dl <= float(conservation_rel_tol)
    checks.append({"name": "angular_momentum_conservation", "status": "pass" if l_ok else "fail", "max_rel_drift": max_dl})
    if not l_ok:
        reasons.append(f"angular momentum drift {max_dl:.3e} exceeds {conservation_rel_tol:.3e}")

    return _cert(spec_id, "science_astrophysics", checks, reasons, payload)


def verify_genetic_elenchus_thermodynamics(
    prob_vector: Sequence[float],
    mean_work_kbt: float,
    delta_f_kbt: float,
    dicodon_vec_8d: Sequence[float],
    *,
    reported_e8_codeword: Optional[Sequence[float]] = None,
    prob_tol: float = 1e-9,
    spec_id: str = "wave3-genetic",
) -> Dict[str, Any]:
    """Fail-closed verifier for `bee_genetic_elenchus` (Pauli Master Eq, Jarzynski-Crooks bound & E8 dicodon lattice).

    Verifies:
      1. Pauli Master Equation probability simplex: `p_i >= -prob_tol` and `|sum(p_i) - 1| <= prob_tol`.
      2. Jarzynski-Crooks second-law dissipative bound: `<W> >= delta_F - prob_tol`.
      3. Conway-Sloane E8 lattice quantization of the 8D dicodon feature vector (`nearest_e8` & `is_lattice_point`).
    """
    payload = {
        "prob_vector": list(prob_vector) if isinstance(prob_vector, (list, tuple)) else None,
        "mean_work_kbt": mean_work_kbt,
        "delta_f_kbt": delta_f_kbt,
        "dicodon_vec_8d": list(dicodon_vec_8d) if isinstance(dicodon_vec_8d, (list, tuple)) else None,
        "reported_e8_codeword": list(reported_e8_codeword) if isinstance(reported_e8_codeword, (list, tuple)) else None,
    }
    if (
        not isinstance(prob_vector, (list, tuple))
        or len(prob_vector) == 0
        or not all(_is_finite_num(p) for p in prob_vector)
        or not _is_finite_num(mean_work_kbt)
        or not _is_finite_num(delta_f_kbt)
        or not isinstance(dicodon_vec_8d, (list, tuple))
        or len(dicodon_vec_8d) != 8
        or not all(_is_finite_num(x) for x in dicodon_vec_8d)
    ):
        return _cert(spec_id, "science_genetics", [], ["malformed genetic thermodynamic inputs"], payload, unverified=True)

    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    probs = [float(p) for p in prob_vector]
    prob_sum = sum(probs)
    min_prob = min(probs)
    simplex_ok = min_prob >= -float(prob_tol) and abs(prob_sum - 1.0) <= float(prob_tol)
    checks.append(
        {
            "name": "pauli_master_probability_conservation",
            "status": "pass" if simplex_ok else "fail",
            "prob_sum": prob_sum,
            "min_prob": min_prob,
        }
    )
    if not simplex_ok:
        reasons.append(f"Pauli state probabilities violate simplex (sum={prob_sum:.12g}, min={min_prob:.3e})")

    dissipated_work = float(mean_work_kbt) - float(delta_f_kbt)
    jarzynski_ok = dissipated_work >= -float(prob_tol)
    checks.append(
        {
            "name": "jarzynski_crooks_second_law",
            "status": "pass" if jarzynski_ok else "fail",
            "dissipated_work_kbt": dissipated_work,
        }
    )
    if not jarzynski_ok:
        reasons.append(f"Jarzynski-Crooks second law violated: <W> - delta_F = {dissipated_work:.6g} k_B T < 0")

    x_arr = np.asarray(dicodon_vec_8d, dtype=float).reshape(1, 8)
    ref_e8 = nearest_e8(x_arr)
    e8_ok = bool(is_lattice_point("E8", ref_e8)[0])
    if reported_e8_codeword is not None:
        if not isinstance(reported_e8_codeword, (list, tuple)) or len(reported_e8_codeword) != 8 or not all(_is_finite_num(v) for v in reported_e8_codeword):
            e8_ok = False
            reasons.append("reported_e8_codeword must be 8 finite numbers")
        else:
            rep_arr = np.asarray(reported_e8_codeword, dtype=float).reshape(1, 8)
            is_member = bool(is_lattice_point("E8", rep_arr)[0])
            dist_rep = float(np.sum((x_arr - rep_arr) ** 2))
            dist_ref = float(np.sum((x_arr - ref_e8) ** 2))
            e8_ok = is_member and (dist_rep <= dist_ref + 1e-9)
            if not e8_ok:
                reasons.append(f"reported E8 codeword is not an optimal E8 lattice point (member={is_member}, dist={dist_rep:.6g} vs {dist_ref:.6g})")
    checks.append(
        {
            "name": "conway_sloane_e8_dicodon_quantizer",
            "status": "pass" if e8_ok else "fail",
            "nearest_e8": [float(v) for v in ref_e8[0]],
        }
    )

    return _cert(spec_id, "science_genetics", checks, reasons, payload)


def verify_topology_tda_and_lattice(
    betti_numbers: Sequence[int],
    reported_euler_char: int,
    point: Sequence[float],
    lattice_name: str = "E8",
    riesz_s: float = 3.0,
    *,
    spatial_dim: int = 2,
    reported_nearest_point: Optional[Sequence[float]] = None,
    spec_id: str = "wave3-topology",
) -> Dict[str, Any]:
    """Fail-closed verifier for `bee_topology_tda_lean` (Euler-Poincare identity, Riesz s > d admissibility & A2/Dn/E8 optimality).

    Verifies:
      1. Non-negative integer Betti numbers `beta_k >= 0` and Euler-Poincare identity `chi = sum_k (-1)^k beta_k`.
      2. Riesz energy finiteness condition `s > spatial_dim` (`s > 2` for the 2D triangular lattice `A2`,
         matching `TriangularRiesz.latticeEnergy_riesz_lt_top`).
      3. Exact nearest-lattice-point optimality on `A2`, `D4`/`Dn`, `E8`, or `Z`.
    """
    payload = {
        "betti_numbers": list(betti_numbers) if isinstance(betti_numbers, (list, tuple)) else None,
        "reported_euler_char": reported_euler_char,
        "point": list(point) if isinstance(point, (list, tuple)) else None,
        "lattice_name": lattice_name,
        "riesz_s": riesz_s,
        "spatial_dim": spatial_dim,
        "reported_nearest_point": list(reported_nearest_point) if isinstance(reported_nearest_point, (list, tuple)) else None,
    }
    if (
        not isinstance(betti_numbers, (list, tuple))
        or len(betti_numbers) == 0
        or any(isinstance(b, bool) or not isinstance(b, int) for b in betti_numbers)
        or isinstance(reported_euler_char, bool)
        or not isinstance(reported_euler_char, int)
        or not isinstance(point, (list, tuple))
        or len(point) == 0
        or not all(_is_finite_num(v) for v in point)
        or not _is_finite_num(riesz_s)
    ):
        return _cert(spec_id, "science_topology", [], ["malformed topology or lattice inputs"], payload, unverified=True)

    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    betti_nonneg = all(int(b) >= 0 for b in betti_numbers)
    computed_chi = sum(((-1) ** k) * int(b) for k, b in enumerate(betti_numbers))
    euler_ok = betti_nonneg and (computed_chi == int(reported_euler_char))
    checks.append(
        {
            "name": "euler_poincare_identity",
            "status": "pass" if euler_ok else "fail",
            "computed_chi": computed_chi,
            "reported_chi": int(reported_euler_char),
        }
    )
    if not euler_ok:
        reasons.append(f"Euler-Poincare mismatch or negative Betti number: computed chi={computed_chi}, reported={reported_euler_char}")

    riesz_ok = float(riesz_s) > float(spatial_dim)
    checks.append(
        {
            "name": "riesz_energy_admissibility",
            "status": "pass" if riesz_ok else "fail",
            "riesz_s": float(riesz_s),
            "spatial_dim": int(spatial_dim),
        }
    )
    if not riesz_ok:
        reasons.append(f"Riesz exponent s={riesz_s} is not strictly greater than spatial dimension d={spatial_dim}")

    lat = str(lattice_name).upper()
    pt_arr = np.asarray(point, dtype=float).reshape(1, -1)
    dim = pt_arr.shape[1]
    lat_ok = True
    nearest_vec: List[float] = []
    try:
        if lat == "A2":
            ref_c = nearest_a2(pt_arr)
            lat_ok = bool(is_lattice_point("A2", ref_c)[0])
        elif lat == "E8":
            ref_c = nearest_e8(pt_arr)
            lat_ok = bool(is_lattice_point("E8", ref_c)[0])
        elif lat in ("D4", "DN", "D8"):
            if lat == "D4" and dim != 4:
                raise ValueError("D4 requires dimension 4")
            ref_c = nearest_dn(pt_arr)
            is_int = bool(np.all(np.abs(ref_c - np.rint(ref_c)) < 1e-9))
            even_sum = bool(abs(np.mod(np.rint(ref_c.sum()), 2.0)) < 1e-9)
            lat_ok = is_int and even_sum
        elif lat == "Z":
            ref_c = nearest_zn(pt_arr)
            lat_ok = bool(is_lattice_point("Z", ref_c)[0])
        else:
            raise ValueError(f"unsupported lattice {lattice_name!r}")
        nearest_vec = [float(v) for v in ref_c[0]]
        if reported_nearest_point is not None:
            rep_arr = np.asarray(reported_nearest_point, dtype=float).reshape(1, -1)
            if rep_arr.shape != ref_c.shape:
                lat_ok = False
                reasons.append("reported_nearest_point dimension mismatch")
            else:
                d_rep = float(np.sum((pt_arr - rep_arr) ** 2))
                d_ref = float(np.sum((pt_arr - ref_c) ** 2))
                if lat in ("A2", "E8", "Z"):
                    member_ok = bool(is_lattice_point(lat, rep_arr)[0])
                else:
                    member_ok = bool(
                        np.all(np.abs(rep_arr - np.rint(rep_arr)) < 1e-9)
                        and abs(np.mod(np.rint(rep_arr.sum()), 2.0)) < 1e-9
                    )
                if not member_ok or d_rep > d_ref + 1e-9:
                    lat_ok = False
                    reasons.append(f"reported nearest point is suboptimal or not in {lat} (d_rep={d_rep:.6g} vs d_ref={d_ref:.6g})")
    except Exception as exc:  # noqa: BLE001
        lat_ok = False
        reasons.append(f"lattice check failed: {exc}")

    checks.append({"name": "lattice_nearest_point_optimality", "status": "pass" if lat_ok else "fail", "lattice": lat, "nearest": nearest_vec})
    return _cert(spec_id, "science_topology", checks, reasons, payload)


def verify_numeric_solver_trajectory(
    spec: Dict[str, Any],
    y_coarse: Sequence[Sequence[float]],
    y_fine: Sequence[Sequence[float]],
    *,
    invariant_drift: float = 0.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    invariant_tol: float = 1e-6,
    spec_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail-closed verifier for `bee_numeric_solver_sundials` (Spec grammar + 100x tightening agreement + invariants)."""
    sid = str(spec_id or (spec.get("id") if isinstance(spec, dict) else "wave3-numeric"))
    payload = {
        "spec": spec,
        "y_coarse": y_coarse,
        "y_fine": y_fine,
        "invariant_drift": invariant_drift,
    }
    try:
        validated = validate_spec(dict(spec))
    except (SpecInvalid, TypeError, ValueError) as exc:
        return _cert(sid, "science_numeric_solver", [{"name": "spec_grammar", "status": "fail"}], [str(exc)], payload)

    n_times = len(validated["t_out"])
    n_dim = len(validated["y0"])
    if (
        not isinstance(y_coarse, (list, tuple))
        or not isinstance(y_fine, (list, tuple))
        or len(y_coarse) != n_times
        or len(y_fine) != n_times
    ):
        return _cert(sid, "science_numeric_solver", [{"name": "shape", "status": "fail"}], ["trajectory length does not match len(t_out)"], payload)

    checks: List[Dict[str, Any]] = [{"name": "spec_grammar", "status": "pass"}]
    reasons: List[str] = []

    shape_ok = True
    max_scaled_diff = 0.0
    for row_c, row_f in zip(y_coarse, y_fine):
        if (
            not isinstance(row_c, (list, tuple))
            or not isinstance(row_f, (list, tuple))
            or len(row_c) != n_dim
            or len(row_f) != n_dim
            or not all(_is_finite_num(v) for v in row_c)
            or not all(_is_finite_num(v) for v in row_f)
        ):
            shape_ok = False
            break
        for vc, vf in zip(row_c, row_f):
            scale = float(atol) + float(rtol) * max(abs(float(vc)), abs(float(vf)))
            max_scaled_diff = max(max_scaled_diff, abs(float(vc) - float(vf)) / max(scale, 1e-30))

    checks.append({"name": "shape", "status": "pass" if shape_ok else "fail"})
    if not shape_ok:
        reasons.append("trajectory rows do not match state dimension or contain non-finite values")
        return _cert(sid, "science_numeric_solver", checks, reasons, payload)

    tight_ok = max_scaled_diff <= 10.0
    checks.append({"name": "tightening", "status": "pass" if tight_ok else "fail", "max_scaled_diff": max_scaled_diff})
    if not tight_ok:
        reasons.append(f"100x tolerance tightening disagreement: max_scaled_diff={max_scaled_diff:.3f} > 10.0")

    inv_ok = _is_finite_num(invariant_drift) and abs(float(invariant_drift)) <= float(invariant_tol)
    checks.append({"name": "invariants", "status": "pass" if inv_ok else "fail", "invariant_drift": float(invariant_drift)})
    if not inv_ok:
        reasons.append(f"invariant drift {invariant_drift} exceeds tolerance {invariant_tol}")

    return _cert(sid, "science_numeric_solver", checks, reasons, payload)
