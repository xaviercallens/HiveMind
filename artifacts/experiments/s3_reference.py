#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Experiment S-3 reference answers and the frozen question set (stdlib + mpmath, deterministic, CPU only).

Usage:  python3 artifacts/experiments/s3_reference.py            writes S3_questions.json and S3_questions.sha256 next to this file.

Every expected answer is computed here, independently of the Rust solver and of the verifier:
  * logistic: closed form;  projectile without drag: closed form;
  * linear systems (series RLC, spring-mass-damper, first-order decay chains, the stiff reaction): exact matrix exponential of the augmented matrix [[A, b], [0, 0]];
  * pendulum, projectile with quadratic drag, Kepler, SIR, SEIR: mpmath.odefun (Taylor series) at 30 digits, with the right-hand sides written out below
    (NOT taken from the spec builders).
If scipy is installed the nonlinear answers are cross-checked against DOP853 (rtol 1e-13); the largest relative difference is printed, not stored.
Nothing here depends on any model, and the file is re-run to the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from hivemind.solverbee import families as fam  # noqa: E402

mp.mp.dps = 30
G = mp.mpf("9.81")
EXPECTED_RTOL, EXPECTED_ATOL = 1e-10, 1e-12
POLICY = {"rtol": 1e-8, "atol": 1e-10, "max_rel_err_factor": 100.0}


def _mf(x):
    return mp.mpf(repr(float(x)))


# ------------------------------------------------------------------ references
def _expm_solution(A, b, y0, ts):
    n = len(y0)
    M = mp.zeros(n + 1, n + 1)
    for i in range(n):
        for j in range(n):
            M[i, j] = _mf(A[i][j])
        M[i, n] = _mf(b[i])
    v0 = mp.matrix([_mf(v) for v in y0] + [mp.mpf(1)])
    out = []
    for t in ts:
        w = mp.expm(M * _mf(t)) * v0
        out.append([float(w[i]) for i in range(n)])
    return out


def _ode(F, y0, ts):
    f = mp.odefun(F, 0, [_mf(v) for v in y0])
    return [[float(v) for v in f(_mf(t))] for t in ts]


def ref_pendulum(p, y0, ts):
    gl, d = _mf(p["g_over_l"]), _mf(p["damping"])
    return _ode(lambda x, y: [y[1], -gl * mp.sin(y[0]) - d * y[1]], y0, ts)


def ref_projectile(p, y0, ts):
    g, k = _mf(p["g"]), _mf(p["k"])
    if k == 0:
        x0, yy0, vx, vy = (_mf(v) for v in y0)
        return [[float(x0 + vx * _mf(t)), float(yy0 + vy * _mf(t) - g * _mf(t) ** 2 / 2), float(vx), float(vy - g * _mf(t))] for t in ts]

    def F(x, y):
        s = mp.sqrt(y[2] ** 2 + y[3] ** 2)
        return [y[2], y[3], -k * s * y[2], -g - k * s * y[3]]
    return _ode(F, y0, ts)


def ref_kepler(p, y0, ts):
    mu = _mf(p["mu"])

    def F(x, y):
        r3 = (y[0] ** 2 + y[1] ** 2) ** mp.mpf("1.5")
        return [y[2], y[3], -mu * y[0] / r3, -mu * y[1] / r3]
    return _ode(F, y0, ts)


def ref_logistic(p, y0, ts):
    r, K, n0 = _mf(p["r"]), _mf(p["K"]), _mf(y0[0])
    return [[float(K / (1 + (K / n0 - 1) * mp.exp(-r * _mf(t))))] for t in ts]


def ref_sir(p, y0, ts):
    b, g = _mf(p["beta"]), _mf(p["gamma"])
    return _ode(lambda x, y: [-b * y[0] * y[1], b * y[0] * y[1] - g * y[1], g * y[1]], y0, ts)


def ref_seir(p, y0, ts):
    b, s, g = _mf(p["beta"]), _mf(p["sigma"]), _mf(p["gamma"])
    return _ode(lambda x, y: [-b * y[0] * y[2], b * y[0] * y[2] - s * y[1], s * y[1] - g * y[2], g * y[2]], y0, ts)


def ref_rlc(p, y0, ts):
    R, L, C, V = p["R"], p["L"], p["C"], p["V"]
    # q' = i ; i' = (V - R i - q/C) / L   (written out here from the circuit law, not from the builder)
    A = [[0.0, 1.0], [-1.0 / (L * C), -R / L]]
    return _expm_solution(A, [0.0, V / L], y0, ts)


def ref_msd(p, y0, ts):
    m, c, k, F = p["m"], p["c"], p["k"], p["F"]
    A = [[0.0, 1.0], [-k / m, -c / m]]
    return _expm_solution(A, [0.0, F / m], y0, ts)


def ref_chain(p, y0, ts):
    rates = p["rates"]
    n = len(rates) + 1
    A = [[0.0] * n for _ in range(n)]
    for i, r in enumerate(rates):
        A[i][i] = -r
        A[i + 1][i] = r
    return _expm_solution(A, [0.0] * n, y0, ts)


REFERENCE = {"pendulum": ref_pendulum, "projectile": ref_projectile, "kepler": ref_kepler, "logistic": ref_logistic, "sir": ref_sir, "seir": ref_seir,
             "rlc": ref_rlc, "msd": ref_msd, "chain": ref_chain, "stiff": ref_chain}
METHOD = {"pendulum": "mpmath.odefun Taylor, 30 digits", "projectile": "closed form (k=0) or mpmath.odefun Taylor, 30 digits", "kepler": "mpmath.odefun Taylor, 30 digits",
          "logistic": "closed form, 30 digits", "sir": "mpmath.odefun Taylor, 30 digits", "seir": "mpmath.odefun Taylor, 30 digits",
          "rlc": "exact matrix exponential (mpmath.expm, 30 digits)", "msd": "exact matrix exponential (mpmath.expm, 30 digits)",
          "chain": "exact matrix exponential (mpmath.expm, 30 digits)", "stiff": "exact matrix exponential (mpmath.expm, 30 digits)"}


def build_spec(kind, p, y0, ts, sid):
    kw = {"spec_id": sid, "rtol": EXPECTED_RTOL, "atol": EXPECTED_ATOL}
    if kind == "pendulum":
        return fam.pendulum_spec(y0[0], y0[1], ts, g_over_l=p["g_over_l"], damping=p["damping"], **kw)
    if kind == "projectile":
        return fam.projectile_drag_spec(*y0, ts, g=p["g"], k=p["k"], **kw)
    if kind == "kepler":
        return fam.kepler_spec(*y0, ts, mu=p["mu"], **kw)
    if kind == "logistic":
        return fam.logistic_spec(y0[0], ts, r=p["r"], K=p["K"], **kw)
    if kind == "sir":
        return fam.sir_spec(p["beta"], p["gamma"], y0[0], y0[1], ts, r0=y0[2], **kw)
    if kind == "seir":
        return fam.seir_spec(p["beta"], p["sigma"], p["gamma"], y0[0], y0[1], y0[2], ts, r0=y0[3], **kw)
    if kind == "rlc":
        return fam.series_rlc_spec(p["R"], p["L"], p["C"], p["V"], y0[0], y0[1], ts, **kw)
    if kind == "msd":
        return fam.spring_mass_damper_spec(p["m"], p["c"], p["k"], p["F"], y0[0], y0[1], ts, **kw)
    return fam.decay_chain_spec(p["rates"], y0, ts, **kw)


# ------------------------------------------------------------------ the question set
def n_en(x):
    return "%g" % x


def n_fr(x):
    return ("%g" % x).replace(".", ",")


Q = []  # (tier, lang, kind, params, y0, ts, text)


def add(tier, lang, kind, p, y0, ts, text):
    Q.append((tier, lang, kind, p, y0, ts, text))


# ---- tier A: direct, family and numbers stated in SI units
add("A", "en", "pendulum", dict(g_over_l=9.81, damping=0.1), [0.5, 0.0], [1, 2, 5],
    "A pendulum has g_over_l = 9.81 and damping = 0.1. It starts with theta0 = 0.5 and omega0 = 0. What are theta and omega at t = 1, 2, 5 seconds?")
add("A", "fr", "pendulum", dict(g_over_l=9.81, damping=0.2), [0.3, 0.0], [1, 2, 4],
    "Un pendule a g_over_l = 9,81 et damping = 0,2. Il part de theta0 = 0,3 avec omega0 = 0. Donne theta et omega à t = 1, 2, 4 secondes.")
add("A", "en", "projectile", dict(g=9.81, k=0.0), [0.0, 0.0, 10.0, 15.0], [0.5, 1, 2],
    "A projectile with g = 9.81 and no drag is launched from x0 = 0, y0 = 0 with vx0 = 10 and vy0 = 15. Give x, y, vx, vy at t = 0.5, 1, 2 seconds.")
add("A", "en", "projectile", dict(g=9.81, k=0.02), [0.0, 0.0, 20.0, 20.0], [1, 2, 3],
    "A projectile with g = 9.81 and drag coefficient k = 0.02 is launched from x0 = 0, y0 = 0 with vx0 = 20 and vy0 = 20. Give x, y, vx, vy at t = 1, 2, 3 seconds.")
add("A", "en", "kepler", dict(mu=1.0), [1.0, 0.0, 0.0, 1.2], [1, 3, 6],
    "A body in a planar Kepler orbit with mu = 1 starts at x0 = 1, y0 = 0 with vx0 = 0 and vy0 = 1.2. Give x, y, vx, vy at t = 1, 3, 6 seconds.")
add("A", "en", "logistic", dict(r=0.5, K=100.0), [10.0], [2, 5, 10],
    "Logistic growth with r = 0.5 and K = 100 starts from N0 = 10. What is N at t = 2, 5, 10?")
add("A", "fr", "logistic", dict(r=0.8, K=50.0), [5.0], [1, 4, 8],
    "Croissance logistique avec r = 0,8 et K = 50, N0 = 5. Quelle est N à t = 1, 4, 8 ?")
add("A", "en", "sir", dict(beta=0.4, gamma=0.1), [0.99, 0.01, 0.0], [10, 30, 60],
    "An SIR epidemic has beta = 0.4 and gamma = 0.1, with s0 = 0.99 and i0 = 0.01 (population fractions). What are S, I, R at t = 10, 30, 60 (same time unit as the rates)?")
add("A", "fr", "seir", dict(beta=0.6, sigma=0.3, gamma=0.1), [0.98, 0.01, 0.01, 0.0], [10, 20, 40],
    "Une épidémie SEIR avec beta = 0,6, sigma = 0,3, gamma = 0,1, s0 = 0,98, e0 = 0,01, i0 = 0,01. Donne S, E, I, R à t = 10, 20, 40.")
add("A", "en", "rlc", dict(R=10.0, L=0.5, C=0.01, V=5.0), [0.0, 0.0], [0.1, 0.5, 1],
    "A series RLC circuit has R = 10, L = 0.5, C = 0.01 and a constant voltage V = 5. It starts with q0 = 0 and i0 = 0. Give q and i at t = 0.1, 0.5, 1 seconds.")
add("A", "fr", "msd", dict(m=2.0, c=0.4, k=8.0, F=0.0), [1.0, 0.0], [1, 2, 4],
    "Un système masse-ressort-amortisseur a m = 2, c = 0,4, k = 8 et aucune force extérieure. Il part de x0 = 1 avec v0 = 0. Donne x et v à t = 1, 2, 4 secondes.")
add("A", "fr", "chain", dict(rates=[1.0, 0.5]), [1.0, 0.0, 0.0], [1, 2, 5],
    "Une chaîne de désintégration A -> B -> C a k1 = 1 et k2 = 0,5, avec A0 = 1 (B et C valent 0 au départ). Donne A, B, C à t = 1, 2, 5 secondes.")
add("A", "fr", "stiff", dict(rates=[1000.0, 1.0]), [1.0, 0.0, 0.0], [0.001, 0.01, 1],
    "Réaction raide A -> B -> C avec k1 = 1000 et k2 = 1 ; A0 = 1. Donne A, B, C à t = 0,001, 0,01, 1.")

# ---- tier B: paraphrased (same content, different wording and word order)
add("B", "en", "pendulum", dict(g_over_l=20.0, damping=0.0), [0.8, 0.0], [1, 3, 6],
    "Starting from rest at an angle of 0.8 radians, a pendulum with g_over_l of 20 and no friction swings. After 1, 3 and 6 seconds, what are its angle theta and angular velocity omega?")
add("B", "fr", "projectile", dict(g=9.81, k=0.0), [0.0, 0.0, 12.0, 8.0], [0.5, 1, 1.5],
    "Une balle est lancée depuis l'origine avec une vitesse horizontale de 12 et une vitesse verticale de 8 (g = 9,81, pas de frottement). Où est-elle, et à quelle vitesse, après 0,5 ; 1 et 1,5 secondes ?")
add("B", "en", "projectile", dict(g=9.81, k=0.05), [0.0, 0.0, 30.0, 25.0], [1, 2, 4],
    "Air resistance is quadratic with coefficient 0.05 (g = 9.81). A cannonball leaves the origin with horizontal speed 30 and vertical speed 25. Report position and velocity after 1, 2 and 4 seconds.")
add("B", "en", "kepler", dict(mu=1.0), [1.0, 0.0, 0.0, 0.8], [2, 5, 9],
    "Two-body motion with mu = 1: the body begins at (x, y) = (1, 0) moving with velocity (vx, vy) = (0, 0.8). Where is it, and how fast, at t = 2, 5, 9?")
add("B", "fr", "kepler", dict(mu=2.0), [2.0, 0.0, 0.0, 0.8], [1, 4, 8],
    "Une planète de paramètre gravitationnel mu = 2 part de la position (2, 0) avec la vitesse (0, 0,8). Donne sa position et sa vitesse aux instants 1, 4 et 8.")
add("B", "fr", "logistic", dict(r=0.3, K=40.0), [2.0], [3, 6, 12],
    "La population N démarre à 2 ; elle suit une loi logistique de taux de croissance 0,3 et de capacité 40. Quelle est sa valeur à t = 3, 6, 12 ?")
add("B", "en", "sir", dict(beta=0.5, gamma=0.25), [0.99, 0.01, 0.0], [5, 15, 30],
    "Epidemic model (SIR): 1% of people are infected at the start and everyone else is susceptible. The infection rate beta is 0.5 and the recovery rate gamma is 0.25. "
    "Tell me the susceptible, infected and recovered fractions at t = 5, 15, 30.")
add("B", "fr", "sir", dict(beta=0.3, gamma=0.1), [0.98, 0.02, 0.0], [20, 50, 100],
    "Sur une population normalisée, 2 % des individus sont infectés au départ et 98 % sont sains. On donne beta = 0,3 et gamma = 0,1. Que valent S, I et R à t = 20, 50, 100 ?")
add("B", "en", "seir", dict(beta=0.5, sigma=0.2, gamma=0.1), [0.98, 0.01, 0.01, 0.0], [10, 25, 50],
    "In an SEIR model with infection rate beta = 0.5, incubation rate sigma = 0.2 and recovery rate gamma = 0.1, begin with 1% exposed, 1% infectious and the rest susceptible. "
    "What are S, E, I and R at t = 10, 25, 50?")
add("B", "fr", "rlc", dict(R=20.0, L=1.0, C=0.05, V=5.0), [0.0, 0.0], [0.2, 1, 3],
    "Un circuit RLC série est branché sur une source continue de 5 V ; sa résistance est 20, son inductance 1 et sa capacité 0,05. Le condensateur est déchargé et le courant est nul au départ. "
    "Donne la charge q et le courant i à t = 0,2 ; 1 et 3.")
add("B", "en", "rlc", dict(R=2.0, L=0.5, C=0.1, V=3.0), [0.0, 0.0], [0.5, 1, 2],
    "With R = 2, L = 0.5 and C = 0.1 in series (a constant 3 V supply, empty capacitor, no initial current), how do the charge q and the current i evolve? Evaluate at t = 0.5, 1, 2.")
add("B", "en", "msd", dict(m=1.5, c=0.6, k=12.0, F=0.0), [0.2, 0.0], [1, 3, 6],
    "A 1.5 kg mass sits on a spring of stiffness 12 with damping coefficient 0.6; it is pulled 0.2 from equilibrium and released from rest. Give displacement x and velocity v at t = 1, 3, 6.")
add("B", "fr", "chain", dict(rates=[0.8, 0.3]), [2.0, 0.0, 0.0], [1, 3, 8],
    "Au départ il n'y a que 2 unités de A ; A se transforme en B avec le taux 0,8 et B en C avec le taux 0,3. Quelles quantités de A, B, C à t = 1, 3, 8 ?")
add("B", "en", "chain", dict(rates=[0.5, 0.2]), [3.0, 0.0, 0.0], [2, 6, 12],
    "Substance A decays into B (rate 0.5) and B decays into C (rate 0.2). Initially only A is present, 3 units. How much A, B and C remain at t = 2, 6, 12?")
add("B", "en", "stiff", dict(rates=[500.0, 0.4]), [1.0, 0.0, 0.0], [0.002, 0.05, 2],
    "Consider A turning into B very quickly (rate 500) and B turning into C slowly (rate 0.4); starting from 1 unit of A, what are the three amounts at t = 0.002, 0.05, 2?")

# ---- tier C: implicit (a unit conversion or a derived parameter is needed; all derived numbers are exact decimals)
add("C", "fr", "pendulum", dict(g_over_l=12.2625, damping=0.0), [0.4, 0.0], [1, 2, 4],
    "Un pendule de longueur 0,8 m (g = 9,81 m/s²) est lâché sans vitesse initiale avec un angle de 0,4 rad, sans frottement. Quel est l'angle theta et la vitesse angulaire omega à 1, 2 et 4 secondes ?")
add("C", "en", "projectile", dict(g=9.81, k=0.0), [0.0, 0.0, 40.0, 30.0], [0.5, 1, 2],
    "A ball is launched from the origin at 50 m/s along a line that rises 3 metres for every 4 metres of horizontal distance, with no air resistance (g = 9.81 m/s²). "
    "Give its position and velocity components at 0.5, 1 and 2 seconds.")
add("C", "fr", "projectile", dict(g=9.81, k=0.01), [0.0, 2.0, 20.0, 0.0], [0.5, 1, 1.5],
    "Une balle est lancée à 72 km/h horizontalement depuis une hauteur de 2 m (g = 9,81 m/s², traînée quadratique de coefficient 0,01). Donne x, y, vx, vy à 0,5 ; 1 et 1,5 s.")
add("C", "en", "kepler", dict(mu=4.0), [1.0, 0.0, 0.0, 2.0], [0.5, 1, 2],
    "A satellite is on a circular orbit of radius 1 around a body with gravitational parameter mu = 4, starting at (1, 0) and moving counter-clockwise. Where is it, and how fast, at t = 0.5, 1 and 2?")
add("C", "en", "logistic", dict(r=0.001, K=1000.0), [10.0], [60, 300, 600],
    "A bacterial culture of 10 cells grows logistically with a growth rate of 6% per minute and a carrying capacity of 1000 cells. Please work in seconds. What is the population at 60, 300 and 600 seconds?")
add("C", "fr", "sir", dict(beta=0.6, gamma=0.2), [0.99, 0.01, 0.0], [10, 30, 60],
    "Une épidémie a un R0 de 3 et une période infectieuse de 5 jours ; au départ 1 % de la population est infectée et le reste est sain. Donne les fractions S, I, R aux jours 10, 30 et 60.")
add("C", "en", "seir", dict(beta=0.48, sigma=0.25, gamma=0.2), [0.98, 0.01, 0.01, 0.0], [10, 30, 60],
    "R0 = 2.4, incubation period 4 days, infectious period 5 days. At the start 1% are infectious, 1% are exposed and the rest are susceptible. Give S, E, I, R on day 10, 30 and 60.")
add("C", "en", "rlc", dict(R=4.0, L=0.25, C=0.01, V=12.0), [0.0, 0.0], [0.05, 0.2, 0.5],
    "A series circuit has a 4 ohm resistor, a 250 mH inductor, a 10 mF capacitor and a 12 V battery switched on at t = 0 (capacitor empty, no current). Please work in seconds. "
    "Give the charge q and current i at 50, 200 and 500 milliseconds.")
add("C", "fr", "msd", dict(m=0.5, c=4.0, k=200.0, F=0.0), [0.05, 0.0], [0.05, 0.1, 0.3],
    "Une masse de 500 g est reliée à un ressort de raideur 200 N/m ; le coefficient d'amortissement vaut 4 N·s/m. On l'écarte de 5 cm et on la lâche sans vitesse. "
    "Donne le déplacement (en mètres) et la vitesse (en m/s) à 0,05 ; 0,1 et 0,3 s.")
add("C", "en", "msd", dict(m=2.0, c=3.0, k=50.0, F=10.0), [0.0, 0.0], [0.2, 0.5, 1],
    "A 2 kg block sits on a spring with k = 50 N/m; a steady 10 N force pushes on it, and a damper resists with 6 N when the block moves at 2 m/s. "
    "It starts at rest at x = 0. Give x and v at 0.2, 0.5 and 1 second.")
add("C", "en", "chain", dict(rates=[0.05, 0.02]), [1.0, 0.0, 0.0], [30, 60, 120],
    "A turns into B at 3 per minute and B turns into C at 1.2 per minute. We start with 1 unit of A. Please work in seconds. Give the amounts of A, B and C at 30, 60 and 120 seconds.")
add("C", "en", "stiff", dict(rates=[1000.0, 0.5]), [1.0, 0.0, 0.0], [0.001, 0.01, 1],
    "Species A turns into B with a time constant of 1 ms, and B turns into C with a time constant of 2 s. We start with 1 mol of A. Please work in seconds. "
    "Give the amounts of A, B and C at 0.001, 0.01 and 1 second.")


def crosscheck(kind, p, y0, ts, ref):
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError:
        return None
    if kind == "pendulum":
        def f(t, y): return [y[1], -p["g_over_l"] * np.sin(y[0]) - p["damping"] * y[1]]
    elif kind == "projectile":
        def f(t, y):
            s = (y[2] ** 2 + y[3] ** 2) ** 0.5
            return [y[2], y[3], -p["k"] * s * y[2], -p["g"] - p["k"] * s * y[3]]
    elif kind == "kepler":
        def f(t, y):
            r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
            return [y[2], y[3], -p["mu"] * y[0] / r3, -p["mu"] * y[1] / r3]
    elif kind == "sir":
        def f(t, y): return [-p["beta"] * y[0] * y[1], p["beta"] * y[0] * y[1] - p["gamma"] * y[1], p["gamma"] * y[1]]
    elif kind == "seir":
        def f(t, y): return [-p["beta"] * y[0] * y[2], p["beta"] * y[0] * y[2] - p["sigma"] * y[1], p["sigma"] * y[1] - p["gamma"] * y[2], p["gamma"] * y[2]]
    else:
        return None
    sol = solve_ivp(f, (0, ts[-1]), y0, t_eval=ts, method="DOP853", rtol=1e-13, atol=1e-15)
    worst = 0.0
    for row, rrow in zip(sol.y.T, ref):
        for a, b in zip(row, rrow):
            worst = max(worst, abs(a - b) / (abs(b) + 1e-10))
    return worst


def build():
    questions = []
    counters = {"A": 0, "B": 0, "C": 0}
    worst_cc = 0.0
    for i, (tier, lang, kind, p, y0, ts, text) in enumerate(Q, start=1):
        counters[tier] += 1
        qid = f"q{i:02d}"
        ts = [float(t) for t in ts]
        spec = build_spec(kind, p, y0, ts, f"s3-{qid}")
        ref = REFERENCE[kind](p, y0, ts)
        cc = crosscheck(kind, p, y0, ts, ref)
        if cc is not None:
            worst_cc = max(worst_cc, cc)
            assert cc < 1e-8, (qid, cc)
        questions.append({"id": qid, "tier": tier, "lang": lang, "type": kind, "text": text, "expected_spec": spec,
                          "expected_answers": {"t": ts, "y": ref, "method": METHOD[kind]}})
    return questions, counters, worst_cc


def main():
    questions, counters, worst_cc = build()
    doc = {"schema": "hivemind-s3-questions/1", "experiment_id": "S3-science-bee-formulation", "n_questions": len(questions),
           "by_lang": {l: sum(1 for q in questions if q["lang"] == l) for l in ("en", "fr")}, "by_tier": counters,
           "by_type": {k: sum(1 for q in questions if q["type"] == k) for k in sorted({q["type"] for q in questions})},
           "state_order": "the full state vector y of the expected spec is compared at every requested time (pendulum [theta, omega]; projectile and kepler [x, y, vx, vy]; "
                          "logistic [N]; SIR [S, I, R]; SEIR [S, E, I, R]; RLC [q, i]; spring-mass-damper [x, v]; chains [A, B, C])",
           "comparison_policy": POLICY, "reference_generator": "artifacts/experiments/s3_reference.py (stdlib + mpmath)", "questions": questions}
    data = (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (HERE / "S3_questions.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (HERE / "S3_questions.sha256").write_text(f"{digest}  S3_questions.json\n", encoding="utf-8")
    print(f"{len(questions)} questions, by_lang={doc['by_lang']}, by_tier={counters}")
    print("scipy cross-check max relative difference:", "not available" if worst_cc == 0.0 and not _has_scipy() else f"{worst_cc:.3e}")
    print("sha256", digest)


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
