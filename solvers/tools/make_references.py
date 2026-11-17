# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regenerate the Solver Bee problem library (solvers/problems/*.json), deterministically.

This is the ONLY place in the repository where numpy / scipy / mpmath are used for the
solver library. Every reference is computed by a method that is independent of
rusty-SUNDIALS and is cross-checked by a second, different method; the agreement is
written into each file (reference.cross_check) together with the tool versions.

Methods
  exact_expm     linear systems and first-order mass-action networks: the matrix exponential of the
                 augmented matrix [[A, b], [0, 0]] in 40-digit mpmath arithmetic. Cross-check: for
                 harmonic oscillators the closed form (cos/sin/exp, mpmath); otherwise scipy.linalg.expm
                 (double precision, Pade) and a tight scipy DOP853 / Radau integration.
  taylor         non-stiff nonlinear problems: mpmath.odefun (Taylor series, 30 digits).
                 Cross-check: scipy DOP853, rtol=1e-13, atol=1e-16.
  radau          stiff nonlinear problems (Robertson, Van der Pol mu=100): scipy Radau,
                 rtol=1e-13, atol=1e-20. Cross-check: scipy LSODA, rtol=1e-13, atol=1e-20.
                 (mpmath Taylor series is impractical for these stiffness ratios.)

Usage
  python3 solvers/tools/make_references.py                 # rewrite every file
  python3 solvers/tools/make_references.py --ids a b       # only these ids
  python3 solvers/tools/make_references.py --check         # regenerate in memory, compare to disk
  python3 solvers/tools/make_references.py --out DIR       # write elsewhere
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

try:
    import mpmath as mp
    import numpy as np
    import scipy
    import scipy.integrate
    import scipy.linalg
except ImportError as exc:  # pragma: no cover
    sys.exit(f"make_references.py needs numpy, scipy and mpmath ({exc}); nothing was written")

TOOL = "solvers/tools/make_references.py"
SPEC_SCHEMA = "hivemind-solver-spec/1"
PROBLEM_SCHEMA = "hivemind-solver-problem/1"
MP_DPS = 40
POLICY = {"max_rel_err_factor": 100.0}
REFERENCE_MARGIN = 100.0  # reference must be this many times tighter than rtol*|y|+atol

mp.mp.dps = MP_DPS
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "problems"


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "mpmath": mp.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }


# ----------------------------------------------------------------------------- independent models
# These re-implement the right-hand sides in plain Python arithmetic (works for floats and mpf).

def _is_mp(x) -> bool:
    return isinstance(x, (mp.mpf, mp.mpc))


def _sin(x):
    return mp.sin(x) if _is_mp(x) else math.sin(x)


def _cos(x):
    return mp.cos(x) if _is_mp(x) else math.cos(x)


def _sqrt(x):
    return mp.sqrt(x) if _is_mp(x) else math.sqrt(x)

def rhs_named(family: str, p: dict):
    if family == "robertson":
        k1, k2, k3 = p.get("k1", 0.04), p.get("k2", 3.0e7), p.get("k3", 1.0e4)

        def f(t, y):
            a, b, c = y
            return [-k1 * a + k3 * b * c, k1 * a - k3 * b * c - k2 * b * b, k2 * b * b]
    elif family == "van_der_pol":
        mu = p.get("mu", 1.0)

        def f(t, y):
            return [y[1], mu * (1 - y[0] * y[0]) * y[1] - y[0]]
    elif family == "brusselator":
        a, b = p.get("a", 1.0), p.get("b", 3.0)

        def f(t, y):
            u, v = y
            return [a + u * u * v - (b + 1) * u, b * u - u * u * v]
    elif family == "lotka_volterra":
        al, be, de, ga = p.get("alpha", 1.5), p.get("beta", 1.0), p.get("delta", 1.0), p.get("gamma", 3.0)

        def f(t, y):
            x, z = y
            return [al * x - be * x * z, de * x * z - ga * z]
    elif family == "harmonic":
        w, ze = p.get("omega", 1.0), p.get("zeta", 0.0)

        def f(t, y):
            return [y[1], -w * w * y[0] - 2 * ze * w * y[1]]
    elif family == "pendulum":
        gl, dm = p.get("g_over_l", 9.81), p.get("damping", 0.0)

        def f(t, y):
            return [y[1], -gl * _sin(y[0]) - dm * y[1]]
    elif family == "projectile_drag":
        g, k = p.get("g", 9.81), p.get("k", 0.0)

        def f(t, y):
            s = _sqrt(y[2] * y[2] + y[3] * y[3])
            return [y[2], y[3], -k * s * y[2], -g - k * s * y[3]]
    elif family == "kepler":
        mu = p.get("mu", 1.0)

        def f(t, y):
            r2 = y[0] * y[0] + y[1] * y[1]
            r3 = r2 * _sqrt(r2)
            return [y[2], y[3], -mu * y[0] / r3, -mu * y[1] / r3]
    elif family == "logistic":
        r, K = p.get("r", 1.0), p.get("K", 1.0)

        def f(t, y):
            return [r * y[0] * (1 - y[0] / K)]
    else:
        raise ValueError(family)
    return f


def stiff_chain_matrix(n: int, k: float):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if i > 0:
            A[i][i - 1] += k
            A[i][i] -= k
        if i + 1 < n:
            A[i][i + 1] += k
            A[i][i] -= k
    return A


def mass_action_rhs(spec_problem: dict):
    species = spec_problem["species"]
    idx = {s: i for i, s in enumerate(species)}
    rxns = []
    for r in spec_problem["reactions"]:
        reac = [(idx[s], nu) for s, nu in sorted(r["reactants"].items())]
        net = [0] * len(species)
        for s, nu in r["reactants"].items():
            net[idx[s]] -= nu
        for s, nu in r["products"].items():
            net[idx[s]] += nu
        rxns.append((r["k"], reac, net))

    def f(t, y):
        out = [y[0] * 0 for _ in species]
        for k, reac, net in rxns:
            rate = k
            for i, nu in reac:
                for _ in range(nu):
                    rate = rate * y[i]
            for i, nu in enumerate(net):
                if nu:
                    out[i] = out[i] + nu * rate
        return out

    return f


def mass_action_linear(spec_problem: dict):
    """(A, b) for a purely first-order network; raises if the network is not first-order."""
    species = spec_problem["species"]
    idx = {s: i for i, s in enumerate(species)}
    n = len(species)
    A = [[0.0] * n for _ in range(n)]
    for r in spec_problem["reactions"]:
        if len(r["reactants"]) != 1 or list(r["reactants"].values()) != [1]:
            raise ValueError("not a first-order network")
        (src,) = r["reactants"]
        for s, nu in r["products"].items():
            A[idx[s]][idx[src]] += nu * r["k"]
        A[idx[src]][idx[src]] -= r["k"]
    return A, [0.0] * n


def linear_parts(spec: dict):
    """Return (A, b) if the problem is linear (or a first-order network / harmonic), else None."""
    pr = spec["problem"]
    if pr["kind"] == "linear":
        return pr["A"], pr["b"]
    if pr["kind"] == "mass_action":
        try:
            return mass_action_linear(pr)
        except ValueError:
            return None
    if pr["kind"] == "named":
        p = pr.get("params", {})
        if pr["family"] == "harmonic":
            w, ze = p.get("omega", 1.0), p.get("zeta", 0.0)
            return [[0.0, 1.0], [-w * w, -2 * ze * w]], [0.0, 0.0]
        if pr["family"] == "stiff_chain":
            n, k = int(p.get("n", 10)), p.get("k", 100.0)
            return stiff_chain_matrix(n, k), [0.0] * n
    return None


def rhs_from_spec(spec: dict):
    pr = spec["problem"]
    if pr["kind"] == "named" and pr["family"] != "stiff_chain":
        return rhs_named(pr["family"], pr.get("params", {}))
    if pr["kind"] == "mass_action":
        return mass_action_rhs(pr)
    A, b = linear_parts(spec)
    n = len(b)

    def f(t, y):
        return [sum(A[i][j] * y[j] for j in range(n)) + b[i] for i in range(n)]

    return f


# ----------------------------------------------------------------------------- reference methods

def ref_exact_expm(spec: dict):
    A, b = linear_parts(spec)
    n = len(b)
    M = mp.zeros(n + 1, n + 1)
    for i in range(n):
        for j in range(n):
            M[i, j] = mp.mpf(A[i][j])
        M[i, n] = mp.mpf(b[i])
    z0 = mp.matrix([mp.mpf(v) for v in spec["y0"]] + [mp.mpf(1)])
    out = []
    for t in spec["t_out"]:
        E = mp.expm(M * (mp.mpf(t) - mp.mpf(spec["t0"])))
        z = E * z0
        out.append([float(z[i]) for i in range(n)])
    return out


def ref_taylor_dps20(spec: dict):
    return ref_taylor(spec, dps=20)


def ref_taylor(spec: dict, dps: int = 30):
    mp.mp.dps = dps
    try:
        f = rhs_from_spec(spec)
        sol = mp.odefun(lambda t, y: f(t, y), mp.mpf(spec["t0"]), [mp.mpf(v) for v in spec["y0"]])
        return [[float(v) for v in sol(mp.mpf(t))] for t in spec["t_out"]]
    finally:
        mp.mp.dps = MP_DPS


def _scipy(spec: dict, method: str, rtol: float, atol: float):
    f = rhs_from_spec(spec)
    sol = scipy.integrate.solve_ivp(
        lambda t, y: f(t, list(y)),
        (spec["t0"], spec["t_out"][-1]),
        spec["y0"],
        method=method,
        t_eval=spec["t_out"],
        rtol=rtol,
        atol=atol,
    )
    if not sol.success or sol.y.shape[1] != len(spec["t_out"]):
        raise RuntimeError(f"scipy {method} failed: {sol.message}")
    return [[float(v) for v in sol.y[:, i]] for i in range(len(spec["t_out"]))]


def ref_radau(spec):
    return _scipy(spec, "Radau", 1e-13, 1e-20)


def cross_dop853(spec):
    return _scipy(spec, "DOP853", 1e-13, 1e-16)


def cross_lsoda(spec):
    return _scipy(spec, "LSODA", 1e-13, 1e-20)


def cross_scipy_expm(spec):
    A, b = linear_parts(spec)
    n = len(b)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = np.array(A, dtype=float)
    M[:n, n] = np.array(b, dtype=float)
    z0 = np.array(list(spec["y0"]) + [1.0])
    return [[float(v) for v in (scipy.linalg.expm(M * (t - spec["t0"])) @ z0)[:n]] for t in spec["t_out"]]


def cross_harmonic_closed_form(spec):
    p = spec["problem"].get("params", {})
    w, ze = mp.mpf(p.get("omega", 1.0)), mp.mpf(p.get("zeta", 0.0))
    x0, v0 = mp.mpf(spec["y0"][0]), mp.mpf(spec["y0"][1])
    out = []
    for t in spec["t_out"]:
        t = mp.mpf(t) - mp.mpf(spec["t0"])
        if ze == 0:
            x = x0 * mp.cos(w * t) + v0 / w * mp.sin(w * t)
            v = -x0 * w * mp.sin(w * t) + v0 * mp.cos(w * t)
        else:  # underdamped, zeta < 1
            wd = w * mp.sqrt(1 - ze * ze)
            a = ze * w
            e = mp.exp(-a * t)
            c, s = mp.cos(wd * t), mp.sin(wd * t)
            x = e * (x0 * c + (v0 + a * x0) / wd * s)
            v = e * (v0 * c - (a * v0 + (w * w) * x0) / wd * s)
        out.append([float(x), float(v)])
    return out


# ----- closed-form / special-function references (held-out set) -----

def ref_pendulum_elliptic(spec: dict):
    """Undamped pendulum released from rest: sin(theta/2) = k cd(w0 t | m=k^2), k = sin(theta0/2) (exact).
    omega is the numerical derivative of theta (mp.diff) at 40 digits."""
    p = spec["problem"].get("params", {})
    if p.get("damping", 0.0) != 0.0 or spec["y0"][1] != 0.0:
        raise ValueError("elliptic reference needs damping = 0 and omega0 = 0")
    w0 = mp.sqrt(mp.mpf(p.get("g_over_l", 9.81)))
    k = mp.sin(mp.mpf(spec["y0"][0]) / 2)
    m = k * k

    def theta(t):
        return 2 * mp.asin(k * mp.ellipfun("cd", w0 * (mp.mpf(t) - mp.mpf(spec["t0"])), m=m))

    return [[float(theta(t)), float(mp.diff(theta, mp.mpf(t)))] for t in spec["t_out"]]


def ref_projectile_closed(spec: dict):
    g = mp.mpf(spec["problem"].get("params", {}).get("g", 9.81))
    x0, y0, vx, vy = (mp.mpf(v) for v in spec["y0"])
    out = []
    for t in spec["t_out"]:
        t = mp.mpf(t) - mp.mpf(spec["t0"])
        out.append([float(x0 + vx * t), float(y0 + vy * t - g * t * t / 2), float(vx), float(vy - g * t)])
    return out


def ref_kepler_exact(spec: dict):
    """Exact two-body propagation (Lagrange f and g functions in eccentric-anomaly form, Kepler's
    equation solved by mpmath findroot at 40 digits). Valid for bound (elliptic) orbits, including e = 0."""
    mu = mp.mpf(spec["problem"].get("params", {}).get("mu", 1.0))
    x0, y0, u0, v0 = (mp.mpf(v) for v in spec["y0"])
    r0 = mp.sqrt(x0 * x0 + y0 * y0)
    energy = (u0 * u0 + v0 * v0) / 2 - mu / r0
    if energy >= 0:
        raise ValueError("kepler reference needs a bound orbit")
    a = -mu / (2 * energy)
    n = mp.sqrt(mu / a ** 3)
    ec = 1 - r0 / a                         # e cos(E0)
    es = (x0 * u0 + y0 * v0) / mp.sqrt(mu * a)   # e sin(E0)
    out = []
    for t in spec["t_out"]:
        dt = mp.mpf(t) - mp.mpf(spec["t0"])
        rhs = n * dt
        dE = mp.findroot(lambda d: d - ec * mp.sin(d) + es * (1 - mp.cos(d)) - rhs, rhs)
        cd, sd = mp.cos(dE), mp.sin(dE)
        r = a * (1 - ec * cd + es * sd)     # r = a (1 - e cos E), E = E0 + dE
        f = 1 - a / r0 * (1 - cd)
        g = dt - (dE - sd) / n
        fd = -mp.sqrt(mu * a) / (r * r0) * sd
        gd = 1 - a / r * (1 - cd)
        out.append([float(f * x0 + g * u0), float(f * y0 + g * v0), float(fd * x0 + gd * u0), float(fd * y0 + gd * v0)])
    return out


def ref_logistic_closed(spec: dict):
    p = spec["problem"].get("params", {})
    r, K = mp.mpf(p.get("r", 1.0)), mp.mpf(p.get("K", 1.0))
    n0 = mp.mpf(spec["y0"][0])
    return [[float(K / (1 + (K / n0 - 1) * mp.exp(-r * (mp.mpf(t) - mp.mpf(spec["t0"])))))] for t in spec["t_out"]]


def cross_mp_eig(spec: dict):
    """Linear homogeneous system by eigen-decomposition in mpmath (independent of mp.expm)."""
    A, b = linear_parts(spec)
    if any(v != 0 for v in b):
        raise ValueError("cross_mp_eig needs b = 0")
    n = len(b)
    M = mp.matrix([[mp.mpf(A[i][j]) for j in range(n)] for i in range(n)])
    E, ER = mp.eig(M)
    c = mp.lu_solve(ER, mp.matrix([mp.mpf(v) for v in spec["y0"]]))
    out = []
    for t in spec["t_out"]:
        dt = mp.mpf(t) - mp.mpf(spec["t0"])
        z = ER * mp.matrix([c[i] * mp.exp(E[i] * dt) for i in range(n)])
        out.append([float(mp.re(z[i])) for i in range(n)])
    return out


# ----------------------------------------------------------------------------- invariants

def inv_linear(name, weights, y0):
    value = float(sum(mp.mpf(w) * mp.mpf(y) for w, y in zip(weights, y0)))
    return {"name": name, "kind": "linear", "weights": weights, "value": value}


def inv_quadratic(name, matrix, y0):
    y = [mp.mpf(v) for v in y0]
    value = float(sum(mp.mpf(matrix[i][j]) * y[i] * y[j] for i in range(len(y)) for j in range(len(y))))
    return {"name": name, "kind": "quadratic_form", "matrix": matrix, "value": value}


def inv_lv(name, params, y0):
    al, be, de, ga = params["alpha"], params["beta"], params["delta"], params["gamma"]
    x, z = mp.mpf(y0[0]), mp.mpf(y0[1])
    value = float(de * x - ga * mp.log(x) + be * z - al * mp.log(z))
    return {"name": name, "kind": "lv_first_integral", "params": dict(params), "value": value}


def energy_value(model: str, params: dict, y):
    """Conserved energy used by the held-out set (kind 'energy'); works for floats and mpf."""
    if model == "pendulum":
        return y[1] * y[1] / 2 + params["g_over_l"] * (1 - _cos(y[0]))
    if model == "kepler":
        return (y[2] * y[2] + y[3] * y[3]) / 2 - params["mu"] / _sqrt(y[0] * y[0] + y[1] * y[1])
    if model == "projectile":
        return (y[2] * y[2] + y[3] * y[3]) / 2 + params["g"] * y[1]
    raise ValueError(model)


def inv_energy(name, model, params, y0):
    value = float(energy_value(model, params, [mp.mpf(v) for v in y0]))
    return {"name": name, "kind": "energy", "model": model, "params": dict(params), "value": value}


def finalize_invariant_tol(inv: dict, rtol: float) -> dict:
    """Pre-registered rule, fixed before any solver output was looked at:
    linear invariants: 10*rtol*max(1,|value|); nonlinear invariants: 1000*rtol*max(1,|value|)."""
    factor = 10.0 if inv["kind"] == "linear" else 1000.0
    inv["tol"] = float(f"{factor * rtol * max(1.0, abs(inv['value'])):.6g}")
    return inv


# ----------------------------------------------------------------------------- problem table

def spec_of(pid, problem, y0, t_out, rtol=1e-8, atol=1e-10, method="bdf", max_steps=100000):
    return {
        "schema": SPEC_SCHEMA,
        "id": pid,
        "problem": problem,
        "y0": y0,
        "t0": 0.0,
        "t_out": t_out,
        "rtol": rtol,
        "atol": atol,
        "method": method,
        "max_steps": max_steps,
    }


def mass(species, reactions):
    return {"kind": "mass_action", "species": species, "reactions": reactions}


def rx(reactants, products, k):
    return {"reactants": reactants, "products": products, "k": k}


LV_P = {"alpha": 1.5, "beta": 1.0, "delta": 1.0, "gamma": 3.0}

A5 = [
    [-1.0, 2.0, 0.0, 0.0, 0.5],
    [-2.0, -1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, -3.0, 1.0, 0.0],
    [0.3, 0.0, -1.0, -0.5, 1.0],
    [0.0, 0.2, 0.0, -1.0, -2.0],
]


def problems():
    P = []

    def add(pid, desc, spec, primary, cross, invariants=()):
        P.append({"id": pid, "description": desc, "spec": spec, "primary": primary, "cross": cross,
                  "invariants": list(invariants)})

    s = spec_of("harmonic_undamped", {"kind": "named", "family": "harmonic", "params": {"omega": 2.0}},
                [1.0, 0.5], [0.5, 1.0, 2.0, 5.0, 10.0])
    add("harmonic_undamped", "Undamped oscillator, omega=2; exact solution x=cos(wt)+v0/w sin(wt).", s,
        ("exact_expm", ref_exact_expm), ("closed form (mpmath cos/sin)", cross_harmonic_closed_form),
        [inv_quadratic("energy", [[4.0, 0.0], [0.0, 1.0]], s["y0"])])

    s = spec_of("harmonic_damped", {"kind": "named", "family": "harmonic", "params": {"omega": 3.0, "zeta": 0.1}},
                [1.0, 0.0], [0.5, 1.0, 2.0, 5.0, 10.0])
    add("harmonic_damped", "Underdamped oscillator, omega=3, zeta=0.1 (exponentially decaying); exact.", s,
        ("exact_expm", ref_exact_expm), ("closed form (mpmath cos/sin/exp)", cross_harmonic_closed_form))

    s = spec_of("linear_2x2", {"kind": "linear", "A": [[-1.0, 5.0], [-5.0, -1.0]], "b": [1.0, 0.5]},
                [0.0, 1.0], [0.1, 0.5, 1.0, 2.0, 5.0])
    add("linear_2x2", "x' = A x + b with A = [[-1,5],[-5,-1]] (damped rotation), b=[1,0.5]; exact matrix exponential.",
        s, ("exact_expm", ref_exact_expm), ("scipy.linalg.expm + DOP853", cross_scipy_expm))

    s = spec_of("linear_5x5", {"kind": "linear", "A": A5, "b": [1.0, 0.0, -0.5, 0.25, 0.0]},
                [1.0, -1.0, 0.5, 0.0, 2.0], [0.5, 1.0, 2.0, 5.0, 10.0])
    add("linear_5x5", "Non-symmetric 5x5 linear system with complex eigenvalues and a constant forcing; exact.", s,
        ("exact_expm", ref_exact_expm), ("scipy.linalg.expm", cross_scipy_expm))

    s = spec_of("decay_chain", mass(["A", "B", "C", "D"], [rx({"A": 1}, {"B": 1}, 1.0), rx({"B": 1}, {"C": 1}, 2.0),
                                                           rx({"C": 1}, {"D": 1}, 3.0)]),
                [1.0, 0.0, 0.0, 0.0], [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    add("decay_chain", "Radioactive-style chain A->B->C->D (k=1,2,3), first-order mass action; exact.", s,
        ("exact_expm", ref_exact_expm), ("scipy.linalg.expm", cross_scipy_expm),
        [inv_linear("mass_sum", [1.0, 1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("reversible_network", mass(["A", "B", "C"], [rx({"A": 1}, {"B": 1}, 2.0), rx({"B": 1}, {"A": 1}, 1.0),
                                                              rx({"B": 1}, {"C": 1}, 3.0), rx({"C": 1}, {"B": 1}, 0.5)]),
                [1.0, 0.0, 0.0], [0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    add("reversible_network", "Reversible isomerisation A<->B<->C, first-order mass action; exact.", s,
        ("exact_expm", ref_exact_expm), ("scipy.linalg.expm", cross_scipy_expm),
        [inv_linear("mass_sum", [1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("dimerisation", mass(["A", "D"], [rx({"A": 2}, {"D": 1}, 1.0), rx({"D": 1}, {"A": 2}, 0.5)]),
                [1.0, 0.0], [0.5, 1.0, 5.0, 20.0, 50.0])
    add("dimerisation", "Reversible dimerisation 2A<->D (nonlinear mass action, k=1/0.5); A+2D conserved.", s,
        ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853),
        [inv_linear("A_plus_2D", [1.0, 2.0], s["y0"])])

    s = spec_of("michaelis_menten", mass(["E", "S", "C", "P"],
                                         [rx({"E": 1, "S": 1}, {"C": 1}, 10.0), rx({"C": 1}, {"E": 1, "S": 1}, 1.0),
                                          rx({"C": 1}, {"E": 1, "P": 1}, 2.0)]),
                [1.0, 5.0, 0.0, 0.0], [0.1, 0.5, 1.0, 5.0, 20.0])
    add("michaelis_menten", "Enzyme kinetics E+S<->C->E+P (k=10,1,2); E+C and S+C+P conserved.", s,
        ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853),
        [inv_linear("enzyme_total", [1.0, 0.0, 1.0, 0.0], s["y0"]),
         inv_linear("substrate_total", [0.0, 1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("robertson", {"kind": "named", "family": "robertson"}, [1.0, 0.0, 0.0],
                [0.4, 4.0, 40.0, 400.0, 4000.0, 40000.0], rtol=1e-8, atol=1e-10)
    add("robertson", "Robertson stiff kinetics (k=0.04, 3e7, 1e4) to t=4e4; y1+y2+y3=1.", s,
        ("radau (scipy Radau rtol=1e-13)", ref_radau), ("scipy LSODA rtol=1e-13", cross_lsoda),
        [inv_linear("mass_sum", [1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("van_der_pol_mu1", {"kind": "named", "family": "van_der_pol", "params": {"mu": 1.0}}, [2.0, 0.0],
                [1.0, 5.0, 10.0, 20.0])
    add("van_der_pol_mu1", "Van der Pol oscillator, mu=1 (non-stiff limit cycle).", s,
        ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853))

    s = spec_of("van_der_pol_mu100", {"kind": "named", "family": "van_der_pol", "params": {"mu": 100.0}}, [2.0, 0.0],
                [1.0, 10.0, 50.0, 100.0], rtol=1e-8, atol=1e-8)
    add("van_der_pol_mu100", "Stiff Van der Pol, mu=100, short horizon t<=100 (before the first relaxation jump).", s,
        ("radau (scipy Radau rtol=1e-13)", ref_radau), ("scipy LSODA rtol=1e-13", cross_lsoda))

    s = spec_of("brusselator", {"kind": "named", "family": "brusselator", "params": {"a": 1.0, "b": 3.0}},
                [1.5, 3.0], [1.0, 5.0, 10.0, 20.0])
    add("brusselator", "Brusselator a=1, b=3 (limit cycle).", s,
        ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853))

    s = spec_of("lotka_volterra", {"kind": "named", "family": "lotka_volterra", "params": LV_P}, [1.0, 1.0],
                [1.0, 5.0, 10.0, 20.0])
    add("lotka_volterra", "Lotka-Volterra predator-prey; H = delta x - gamma ln x + beta y - alpha ln y is conserved.",
        s, ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853),
        [inv_lv("first_integral", LV_P, s["y0"])])

    s = spec_of("stiff_chain", {"kind": "named", "family": "stiff_chain", "params": {"n": 6, "k": 200.0}},
                [6.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.001, 0.01, 0.1, 1.0, 10.0])
    add("stiff_chain", "Stiff diffusion chain n=6, k=200 (eigenvalues up to ~800), reflecting ends; sum conserved; exact.",
        s, ("exact_expm", ref_exact_expm), ("scipy.linalg.expm", cross_scipy_expm),
        [inv_linear("mass_sum", [1.0] * 6, s["y0"])])

    s = spec_of("brusselator_adams", {"kind": "named", "family": "brusselator", "params": {"a": 1.0, "b": 1.5}},
                [1.0, 1.0], [1.0, 5.0, 10.0, 20.0], method="adams")
    add("brusselator_adams", "Brusselator a=1, b=1.5 (b < 1+a^2: converges to the fixed point (1, 1.5)), solved with the Adams method.", s,
        ("taylor (mpmath.odefun)", ref_taylor), ("scipy DOP853 rtol=1e-13", cross_dop853))
    return P


HELDOUT_DIR = DEFAULT_OUT / "heldout"
PI = math.pi


def heldout_problems():
    """The held-out set (solvers/problems/heldout/). Written and frozen BEFORE the solver was ever run on it.
    Deliberate mix of short and long horizons; all specs use the library defaults rtol=1e-8, atol=1e-10 unless noted."""
    P = []

    def add(pid, desc, spec, primary, cross, invariants=()):
        P.append({"id": pid, "description": desc, "spec": spec, "primary": primary, "cross": cross,
                  "invariants": list(invariants)})

    ell = ("exact (Jacobi elliptic cd, mpmath)", ref_pendulum_elliptic)
    tay = ("taylor (mpmath.odefun)", ref_taylor)
    dop = ("scipy DOP853 rtol=1e-13", cross_dop853)
    expm = ("exact_expm", ref_exact_expm)

    pn = {"g_over_l": 9.81}
    s = spec_of("pendulum_small_angle", {"kind": "named", "family": "pendulum", "params": pn}, [0.05, 0.0],
                [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    add("pendulum_small_angle", "Undamped pendulum, theta0=0.05 rad (small angle), g/l=9.81, short horizon; the reference is the exact nonlinear solution (not the linearised cos).",
        s, ell, tay, [inv_energy("energy", "pendulum", pn, s["y0"])])

    pn = {"g_over_l": 1.0}
    s = spec_of("pendulum_large_angle", {"kind": "named", "family": "pendulum", "params": pn}, [2.8, 0.0],
                [1.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    add("pendulum_large_angle", "Undamped pendulum released from 2.8 rad (near the separatrix, period about 12), g/l=1, 80 time units (about 7 periods); exact elliptic reference.",
        s, ell, tay, [inv_energy("energy", "pendulum", pn, s["y0"])])

    s = spec_of("pendulum_damped", {"kind": "named", "family": "pendulum", "params": {"g_over_l": 4.0, "damping": 0.3}},
                [1.5, 0.5], [1.0, 5.0, 10.0, 20.0, 40.0])
    add("pendulum_damped", "Damped pendulum, g/l=4, damping 0.3, theta0=1.5, omega0=0.5, decays to rest.", s, tay, dop)

    pn = {"g": 9.81}
    s = spec_of("projectile_no_drag", {"kind": "named", "family": "projectile_drag", "params": pn}, [0.0, 0.0, 30.0, 40.0],
                [0.5, 1.0, 2.0, 4.0, 6.0, 8.0])
    add("projectile_no_drag", "Drag-free projectile (k=0): exact parabola; no ground-hit event (t_out continues below y=0).", s,
        ("exact parabola (mpmath)", ref_projectile_closed), dop, [inv_energy("energy", "projectile", pn, s["y0"])])

    s = spec_of("projectile_drag", {"kind": "named", "family": "projectile_drag", "params": {"g": 9.81, "k": 0.02}},
                [0.0, 0.0, 50.0, 30.0], [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    add("projectile_drag", "Projectile with quadratic drag k=0.02 (k|v| about 1.2 initially, strongly damped), launched at (50, 30); no ground-hit event.", s, tay, dop)

    pn = {"mu": 4.0}
    s = spec_of("kepler_circular", {"kind": "named", "family": "kepler", "params": pn}, [1.0, 0.0, 0.0, 2.0],
                [0.5, 1.0, PI, 2 * PI, 5 * PI, 20 * PI])
    add("kepler_circular", "Circular Kepler orbit, mu=4, r=1 (period pi), out to 20 orbits (t=20 pi); exact propagation via Kepler's equation.", s,
        ("exact Kepler propagation (f,g functions, mpmath)", ref_kepler_exact), ("taylor (mpmath.odefun, 20 digits)", ref_taylor_dps20),
        [inv_energy("energy", "kepler", pn, s["y0"]),
         inv_quadratic("angular_momentum", [[0, 0, 0, 0.5], [0, 0, -0.5, 0], [0, -0.5, 0, 0], [0.5, 0, 0, 0]], s["y0"])])

    pn = {"mu": 1.0}
    s = spec_of("kepler_eccentric", {"kind": "named", "family": "kepler", "params": pn}, [0.4, 0.0, 0.0, 2.0],
                [0.5, 1.0, 2.0, 3.0, PI, 2 * PI, 10.0, 4 * PI, 6 * PI, 25.0])
    add("kepler_eccentric", "Eccentric Kepler orbit e=0.6, a=1, mu=1 starting at perihelion (period 2 pi), about 4 orbits.", s,
        ("exact Kepler propagation (f,g functions, mpmath)", ref_kepler_exact), ("taylor (mpmath.odefun, 20 digits)", ref_taylor_dps20),
        [inv_energy("energy", "kepler", pn, s["y0"]),
         inv_quadratic("angular_momentum", [[0, 0, 0, 0.5], [0, 0, -0.5, 0], [0, -0.5, 0, 0], [0.5, 0, 0, 0]], s["y0"])])

    s = spec_of("kepler_eccentric_long", {"kind": "named", "family": "kepler", "params": pn}, [0.5, 0.0, 0.0, 1.7320508075688772],
                [10.0, 50.0, 100.0, 250.0, 500.0, 200 * PI], max_steps=1000000)
    add("kepler_eccentric_long", "LONG-horizon eccentric Kepler orbit e=0.5 (perihelion start), 100 orbits (t=200 pi), max_steps 1e6.", s,
        ("exact Kepler propagation (f,g functions, mpmath)", ref_kepler_exact), ("taylor (mpmath.odefun, 20 digits)", ref_taylor_dps20),
        [inv_energy("energy", "kepler", pn, s["y0"]),
         inv_quadratic("angular_momentum", [[0, 0, 0, 0.5], [0, 0, -0.5, 0], [0, -0.5, 0, 0], [0.5, 0, 0, 0]], s["y0"])])

    s = spec_of("logistic_growth", {"kind": "named", "family": "logistic", "params": {"r": 0.8, "K": 50.0}}, [2.0],
                [1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    add("logistic_growth", "Logistic growth r=0.8, K=50, N0=2; closed form N=K/(1+(K/N0-1)exp(-rt)).", s,
        ("closed form (mpmath)", ref_logistic_closed), dop)

    s = spec_of("sir_epidemic", mass(["S", "I", "R"], [rx({"S": 1, "I": 1}, {"I": 2}, 0.5), rx({"I": 1}, {"R": 1}, 0.2)]),
                [0.99, 0.01, 0.0], [5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    add("sir_epidemic", "SIR epidemic in fractions (beta=0.5, gamma=0.2, R0=2.5) as mass action S+I->2I, I->R; S+I+R=1.", s, tay, dop,
        [inv_linear("population", [1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("seir_epidemic", mass(["S", "E", "I", "R"], [rx({"S": 1, "I": 1}, {"E": 1, "I": 1}, 0.9), rx({"E": 1}, {"I": 1}, 0.4),
                                                             rx({"I": 1}, {"R": 1}, 0.3)]),
                [0.98, 0.01, 0.01, 0.0], [5.0, 10.0, 20.0, 40.0, 80.0, 120.0])
    add("seir_epidemic", "SEIR epidemic in fractions (beta=0.9, sigma=0.4, gamma=0.3) as mass action S+I->E+I, E->I, I->R; S+E+I+R=1.", s, tay, dop,
        [inv_linear("population", [1.0, 1.0, 1.0, 1.0], s["y0"])])

    s = spec_of("series_rlc", {"kind": "linear", "A": [[0.0, 1.0], [-8.0, -1.6]], "b": [0.0, 4.0]}, [0.0, 0.0],
                [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    add("series_rlc", "Series RLC step response, y=[q, i], L=0.5, R=0.8, C=0.25, V=2 (q'=i, i'=(V-R i-q/C)/L), underdamped; exact.", s,
        expm, ("scipy.linalg.expm", cross_scipy_expm))

    s = spec_of("spring_mass_damper", {"kind": "linear", "A": [[0.0, 1.0], [-4.0, -0.3]], "b": [0.0, 0.5]}, [0.5, 0.0],
                [1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    add("spring_mass_damper", "Spring-mass-damper with constant force, y=[x, v], m=2, c=0.6, k=8, F=1 (x''=(F-c x'-k x)/m); exact.", s,
        expm, ("scipy.linalg.expm", cross_scipy_expm))

    s = spec_of("stiff_reaction", mass(["A", "B", "C"], [rx({"A": 1}, {"B": 1}, 1.0e4), rx({"B": 2}, {"C": 1}, 1.0e3), rx({"C": 1}, {"A": 2}, 0.5)]),
                [1.0, 0.0, 0.0], [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0])
    add("stiff_reaction", "Stiff mass-action cycle A->B (k=1e4), 2B->C (k=1e3), C->2A (k=0.5): timescales 1e-4..1e3, nonlinear; A+B+2C conserved.", s,
        ("radau (scipy Radau rtol=1e-13)", ref_radau), ("scipy LSODA rtol=1e-13", cross_lsoda),
        [inv_linear("mass", [1.0, 1.0, 2.0], s["y0"])])

    w = 1.3
    s = spec_of("harmonic_long", {"kind": "named", "family": "harmonic", "params": {"omega": w}}, [1.0, 0.2],
                [1.0, 10.0, 100.0, 500.0, 1000.0, 2000.0])
    add("harmonic_long", "LONG-horizon undamped oscillator omega=1.3 to t=2000 (about 414 periods); exact.", s,
        expm, ("closed form (mpmath cos/sin)", cross_harmonic_closed_form),
        [inv_quadratic("energy", [[w * w, 0.0], [0.0, 1.0]], s["y0"])])

    k_, c_ = 1.0, 0.3
    A = [[0, 0, 1, 0], [0, 0, 0, 1], [-(k_ + c_), c_, 0, 0], [c_, -(k_ + c_), 0, 0]]
    s = spec_of("coupled_oscillators_long", {"kind": "linear", "A": [[float(v) for v in r] for r in A], "b": [0.0] * 4},
                [1.0, 0.0, 0.0, 0.0], [1.0, 10.0, 100.0, 300.0, 600.0, 1000.0])
    add("coupled_oscillators_long", "LONG-horizon two coupled oscillators (k=1, coupling 0.3, beating) to t=1000; exact.", s,
        expm, ("eigen-decomposition (mpmath eig)", cross_mp_eig),
        [inv_quadratic("energy", [[k_ + c_, -c_, 0, 0], [-c_, k_ + c_, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], s["y0"])])
    return P


# ----------------------------------------------------------------------------- build

def compare(ref, other):
    mad = max(abs(a - b) for ra, rb in zip(ref, other) for a, b in zip(ra, rb))
    return mad


def build(entry: dict) -> dict:
    spec = entry["spec"]
    pname, pfun = entry["primary"]
    cname, cfun = entry["cross"]
    y = pfun(spec)
    y2 = cfun(spec)
    if len(y) != len(spec["t_out"]) or any(len(r) != len(spec["y0"]) for r in y):
        raise RuntimeError(f"{entry['id']}: reference shape")
    flat = [v for r in y for v in r]
    if not all(math.isfinite(v) for v in flat):
        raise RuntimeError(f"{entry['id']}: non-finite reference")
    mad = compare(y, y2)
    ymax = max(abs(v) for v in flat)
    # elementwise tolerance scale the solver is held to
    tol_min = min(spec["rtol"] * abs(v) + spec["atol"] for v in flat)
    bound = max(mad, 1.2e-16 * ymax)
    margin = tol_min / bound if bound > 0 else float("inf")
    # The 100x requirement is applied elementwise against the cross-check disagreement.
    worst_ratio = max(
        abs(a - b) / (spec["rtol"] * abs(a) + spec["atol"]) for ra, rb in zip(y, y2) for a, b in zip(ra, rb)
    )
    if worst_ratio * REFERENCE_MARGIN > 1.0:
        raise RuntimeError(
            f"{entry['id']}: reference cross-check disagreement {worst_ratio:.3g} of the spec tolerance "
            f"(needs <= {1.0 / REFERENCE_MARGIN:g}); tighten the reference or loosen the SPEC tolerance"
        )
    invs = [finalize_invariant_tol(dict(i), spec["rtol"]) for i in entry["invariants"]]
    return {
        "schema": PROBLEM_SCHEMA,
        "id": entry["id"],
        "description": entry["description"],
        "spec": spec,
        "reference": {
            "t": list(spec["t_out"]),
            "y": y,
            "max_abs_error_bound": float(f"{bound:.3g}"),
            "method": pname,
            "tool": TOOL,
            "tool_versions": versions(),
            "cross_check": {
                "method": cname,
                "max_abs_diff": float(f"{mad:.3g}"),
                "max_diff_over_spec_tolerance": float(f"{worst_ratio:.3g}"),
                "min_spec_tolerance_over_bound": float(f"{margin:.3g}"),
            },
        },
        "invariants": invs,
        "tolerance_policy": dict(POLICY),
    }


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", help="only these problem ids")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--heldout", action="store_true", help="work on the held-out set (default output solvers/problems/heldout)")
    ap.add_argument("--check", action="store_true", help="compare regenerated output with the files on disk")
    ap.add_argument("--list", action="store_true", help="print the problem ids and exit")
    args = ap.parse_args(argv)
    entries = heldout_problems() if args.heldout else problems()
    if args.out is None:
        args.out = HELDOUT_DIR if args.heldout else DEFAULT_OUT
    if args.list:
        for e in entries:
            print(e["id"])
        return 0
    if args.ids:
        known = {e["id"] for e in entries}
        unknown = [i for i in args.ids if i not in known]
        if unknown:
            print(f"unknown ids: {unknown}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e["id"] in args.ids]
    bad = 0
    for e in entries:
        text = dump(build(e))
        path = args.out / f"{e['id']}.json"
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            print(f"{'OK  ' if same else 'DIFF'} {e['id']}")
            bad += 0 if same else 1
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
