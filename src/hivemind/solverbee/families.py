# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Family catalogue and data-only spec builders for the Science Bee (stdlib only).

Two kinds of building blocks produce ``hivemind-solver-spec/1`` dictionaries:

* NAMED families that the Rust CLI knows (``pendulum``, ``projectile_drag``, ``kepler``, ``logistic`` and the original six); the builders here only
  check the arguments and wrap them in a spec.
* PRESETS that are expressed with the generic ``mass_action`` and ``linear`` problem kinds (SIR, SEIR, decay chain, series RLC, spring-mass-damper).

Nothing here evaluates anything: every function returns plain JSON-compatible data. Whether a spec describes a real situation is never checked.

State orders are fixed here once and are part of the contract:

* series RLC: ``[q, i]`` (capacitor charge, loop current), ``A = [[0, 1], [-1/(L C), -R/L]]``, ``b = [0, V/L]``.
* spring-mass-damper: ``[x, v]``, ``A = [[0, 1], [-k/m, -c/m]]``, ``b = [0, F/m]``.
* SIR species ``[S, I, R]``; SEIR species ``[S, E, I, R]`` (fractions of the population summing to 1).
* decay chain species ``[A, B, C, ...]`` with ``A -> B -> C ...`` and rate ``k_i`` for step i.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

SPEC_SCHEMA = "hivemind-solver-spec/1"
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LIMITS_SENTENCE = "The numbers solve the equations that were set up; whether those equations describe your real situation is not checked."
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10


def _state(name: str, meaning: str, unit: str) -> Dict[str, str]:
    return {"name": name, "meaning": meaning, "unit": unit}


# name -> catalogue entry. ``params`` maps parameter name -> (default, plain-language meaning); ``dim`` is the state size (None = depends on a parameter).
FAMILIES: Dict[str, Dict[str, Any]] = {
    "pendulum": {
        "summary": "a pendulum: theta'' = -g_over_l*sin(theta) - damping*omega",
        "dim": 2,
        "params": {"g_over_l": (9.81, "gravity divided by the pendulum length (1/time^2)"), "damping": (0.0, "linear friction coefficient (1/time)")},
        "state": [_state("theta", "angle from the downward vertical", "radians"), _state("omega", "angular velocity", "radians per time unit")],
        "invariants": ["energy (only when damping is 0)"],
    },
    "projectile_drag": {
        "summary": "a projectile with quadratic air drag a = -k*|v|*v and gravity g (no ground-hit event: you choose the times)",
        "dim": 4,
        "params": {"g": (9.81, "gravitational acceleration (length/time^2)"), "k": (0.0, "quadratic drag coefficient (1/length); 0 means no drag")},
        "state": [_state("x", "horizontal position", "length unit"), _state("y", "height", "length unit"),
                  _state("vx", "horizontal velocity", "length unit per time unit"), _state("vy", "vertical velocity", "length unit per time unit")],
        "invariants": ["energy (only when k is 0)"],
    },
    "kepler": {
        "summary": "planar two-body motion around a fixed centre with gravitational parameter mu",
        "dim": 4,
        "params": {"mu": (1.0, "gravitational parameter G*M (length^3/time^2)")},
        "state": [_state("x", "x position", "length unit"), _state("y", "y position", "length unit"),
                  _state("vx", "x velocity", "length unit per time unit"), _state("vy", "y velocity", "length unit per time unit")],
        "invariants": ["energy", "angular momentum"],
    },
    "logistic": {
        "summary": "logistic growth N' = r*N*(1-N/K)",
        "dim": 1,
        "params": {"r": (1.0, "growth rate (1/time)"), "K": (1.0, "carrying capacity (same unit as N)")},
        "state": [_state("N", "population size", "population unit")],
        "invariants": [],
    },
    "robertson": {
        "summary": "the stiff Robertson chemical kinetics problem",
        "dim": 3,
        "params": {"k1": (0.04, "rate"), "k2": (3.0e7, "rate"), "k3": (1.0e4, "rate")},
        "state": [_state("y1", "concentration 1", "concentration unit"), _state("y2", "concentration 2", "concentration unit"), _state("y3", "concentration 3", "concentration unit")],
        "invariants": ["total mass y1+y2+y3"],
    },
    "van_der_pol": {
        "summary": "the Van der Pol oscillator x'' = mu*(1-x^2)*x' - x",
        "dim": 2,
        "params": {"mu": (1.0, "nonlinearity / stiffness parameter")},
        "state": [_state("x", "position", "length unit"), _state("v", "velocity", "length unit per time unit")],
        "invariants": [],
    },
    "brusselator": {
        "summary": "the Brusselator chemical oscillator",
        "dim": 2,
        "params": {"a": (1.0, "feed rate"), "b": (3.0, "feed rate")},
        "state": [_state("u", "concentration u", "concentration unit"), _state("v", "concentration v", "concentration unit")],
        "invariants": [],
    },
    "lotka_volterra": {
        "summary": "predator-prey Lotka-Volterra dynamics",
        "dim": 2,
        "params": {"alpha": (1.5, "prey growth rate"), "beta": (1.0, "predation rate"), "delta": (1.0, "predator growth per prey eaten"), "gamma": (3.0, "predator death rate")},
        "state": [_state("x", "prey population", "population unit"), _state("y", "predator population", "population unit")],
        "invariants": ["Lotka-Volterra first integral"],
    },
    "harmonic": {
        "summary": "a damped harmonic oscillator x'' = -omega^2*x - 2*zeta*omega*x'",
        "dim": 2,
        "params": {"omega": (1.0, "natural angular frequency (1/time)"), "zeta": (0.0, "damping ratio (dimensionless)")},
        "state": [_state("x", "displacement", "length unit"), _state("v", "velocity", "length unit per time unit")],
        "invariants": ["energy (only when zeta is 0)"],
    },
    "stiff_chain": {
        "summary": "a diffusion chain of n cells with rate k (reflecting ends)",
        "dim": None,
        "params": {"n": (10, "number of cells (integer 2..20)"), "k": (100.0, "exchange rate")},
        "state": [],
        "invariants": ["total amount (sum of all cells)"],
    },
}
NEW_FAMILIES = ("pendulum", "projectile_drag", "kepler", "logistic")
# Parameters that must be integers.
INTEGER_PARAMS = {"stiff_chain": {"n"}}

PRESETS: Dict[str, str] = {
    "sir": "epidemic SIR as mass_action: S+I->2I (k=beta), I->R (k=gamma); species [S, I, R], fractions summing to 1",
    "seir": "epidemic SEIR as mass_action: S+I->E+I (k=beta), E->I (k=sigma), I->R (k=gamma); species [S, E, I, R]",
    "decay_chain": "radioactive / first-order chain A->B->C... as mass_action with one rate per step",
    "series_rlc": "series R-L-C circuit as a linear system, state [q, i]",
    "spring_mass_damper": "spring-mass-damper as a linear system, state [x, v]",
}


class FamilyError(ValueError):
    """A builder argument is invalid."""


# ------------------------------------------------------------------ validation helpers
def _num(value: Any, name: str, *, positive: bool = False, nonneg: bool = False, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FamilyError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise FamilyError(f"{name} must be finite")
    if abs(v) > 1e150:
        raise FamilyError(f"{name} is too large")
    if positive and not v > 0:
        raise FamilyError(f"{name} must be > 0")
    if nonneg and v < 0:
        raise FamilyError(f"{name} must be >= 0")
    if lo is not None and v < lo:
        raise FamilyError(f"{name} must be >= {lo}")
    if hi is not None and v > hi:
        raise FamilyError(f"{name} must be <= {hi}")
    return v


def _times(t_out: Any, t0: float = 0.0) -> List[float]:
    if not isinstance(t_out, (list, tuple)) or not 1 <= len(t_out) <= 1000:
        raise FamilyError("t_out must be a list of 1..1000 numbers")
    out: List[float] = []
    prev = _num(t0, "t0")
    for i, t in enumerate(t_out):
        v = _num(t, f"t_out[{i}]")
        if not v > prev:
            raise FamilyError("t_out must be strictly increasing and greater than t0")
        out.append(v)
        prev = v
    return out


def _sid(spec_id: str) -> str:
    if not isinstance(spec_id, str) or not ID_RE.match(spec_id):
        raise FamilyError("id must match [A-Za-z0-9_.-]{1,64}")
    return spec_id


def _finish(spec_id: str, problem: Dict[str, Any], y0: Sequence[float], t_out: Any, t0: float, rtol: Optional[float], atol: Optional[float],
            method: Optional[str]) -> Dict[str, Any]:
    t0v = _num(t0, "t0")
    spec: Dict[str, Any] = {"schema": SPEC_SCHEMA, "id": _sid(spec_id), "problem": problem, "y0": [float(v) for v in y0], "t0": t0v, "t_out": _times(t_out, t0v)}
    if rtol is not None:
        spec["rtol"] = _num(rtol, "rtol", positive=True, hi=1.0)
    if atol is not None:
        spec["atol"] = _num(atol, "atol", positive=True, hi=1.0)
    if method is not None:
        if method not in ("bdf", "adams"):
            raise FamilyError("method must be 'bdf' or 'adams'")
        spec["method"] = method
    return spec


def list_families() -> List[str]:
    return sorted(FAMILIES)


def family_dim(family: str, params: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """State dimension of a named family (stiff_chain: the n parameter, default 10)."""
    entry = FAMILIES.get(family)
    if entry is None:
        raise FamilyError(f"unknown family {family!r}")
    if entry["dim"] is not None:
        return int(entry["dim"])
    n = (params or {}).get("n", entry["params"]["n"][0])
    return int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) and float(n).is_integer() else None


def check_params(family: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameter names and values of a named family; returns a clean copy (numbers as float, integer params as int)."""
    entry = FAMILIES.get(family)
    if entry is None:
        raise FamilyError(f"unknown family {family!r}; expected one of {', '.join(list_families())}")
    if not isinstance(params, dict):
        raise FamilyError("params must be an object")
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in entry["params"]:
            raise FamilyError(f"family {family!r} has no parameter {str(k)[:40]!r}; allowed: {', '.join(entry['params'])}")
        val = _num(v, f"params.{k}")
        if k in INTEGER_PARAMS.get(family, ()):
            if not val.is_integer() or not 2 <= val <= 20:
                raise FamilyError(f"params.{k} must be an integer in 2..20")
            out[k] = int(val)
            continue
        if family == "pendulum" and k == "damping" and val < 0:
            raise FamilyError("params.damping must be >= 0")
        if (family, k) in {("pendulum", "g_over_l"), ("projectile_drag", "g"), ("kepler", "mu"), ("logistic", "K"), ("harmonic", "omega")} and not val > 0:
            raise FamilyError(f"params.{k} must be > 0")
        if (family, k) in {("projectile_drag", "k")} and val < 0:
            raise FamilyError("params.k must be >= 0")
        out[k] = val
    return out


def named_spec(family: str, params: Optional[Dict[str, Any]], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", t0: float = 0.0,
               rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """A named-family spec after checking parameter names, values and the length of y0."""
    clean = check_params(family, dict(params or {}))
    dim = family_dim(family, clean)
    if not isinstance(y0, (list, tuple)) or (dim is not None and len(y0) != dim):
        raise FamilyError(f"y0 must have {dim} entries for family {family!r}")
    ys = [_num(v, f"y0[{i}]") for i, v in enumerate(y0)]
    if family == "logistic" and ys[0] < 0:
        raise FamilyError("y0[0] (N) must be >= 0")
    return _finish(spec_id, {"kind": "named", "family": family, "params": clean}, ys, t_out, t0, rtol, atol, method)


# ------------------------------------------------------------------ thin builders for the named families
def pendulum_spec(theta0: float, omega0: float, t_out: Sequence[float], *, g_over_l: float = 9.81, damping: float = 0.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("pendulum", {"g_over_l": g_over_l, "damping": damping}, [theta0, omega0], t_out, **kw)


def projectile_drag_spec(x0: float, y0: float, vx0: float, vy0: float, t_out: Sequence[float], *, g: float = 9.81, k: float = 0.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("projectile_drag", {"g": g, "k": k}, [x0, y0, vx0, vy0], t_out, **kw)


def kepler_spec(x0: float, y0: float, vx0: float, vy0: float, t_out: Sequence[float], *, mu: float = 1.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("kepler", {"mu": mu}, [x0, y0, vx0, vy0], t_out, **kw)


def logistic_spec(n0: float, t_out: Sequence[float], *, r: float = 1.0, K: float = 1.0, **kw: Any) -> Dict[str, Any]:  # noqa: N803 - K is the conventional name
    return named_spec("logistic", {"r": r, "K": K}, [n0], t_out, **kw)


# ------------------------------------------------------------------ mass_action presets
def mass_action_spec(species: Sequence[str], reactions: Sequence[Dict[str, Any]], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee",
                     t0: float = 0.0, rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """Generic mass-action spec (<= 20 species, <= 50 reactions, small positive integer stoichiometry, k >= 0)."""
    sp = list(species)
    if not 1 <= len(sp) <= 20 or len(set(sp)) != len(sp) or not all(isinstance(s, str) and 1 <= len(s) <= 64 for s in sp):
        raise FamilyError("species must be 1..20 distinct strings")
    if not 1 <= len(reactions) <= 50:
        raise FamilyError("reactions must have 1..50 entries")
    rx = []
    for r in reactions:
        if not isinstance(r, dict) or set(r) - {"reactants", "products", "k"}:
            raise FamilyError("each reaction needs reactants, products and k only")
        sides = {}
        for side in ("reactants", "products"):
            m = r.get(side)
            if not isinstance(m, dict) or len(m) > 20:
                raise FamilyError(f"reaction {side} must be an object")
            for s, c in m.items():
                if s not in sp or isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 10:
                    raise FamilyError("stoichiometric coefficients must be integers 1..10 over declared species")
            sides[side] = dict(m)
        rx.append({"reactants": sides["reactants"], "products": sides["products"], "k": _num(r.get("k"), "reaction k", nonneg=True)})
    if len(y0) != len(sp):
        raise FamilyError("y0 must have one entry per species")
    ys = [_num(v, f"y0[{i}]", nonneg=True) for i, v in enumerate(y0)]
    return _finish(spec_id, {"kind": "mass_action", "species": sp, "reactions": rx}, ys, t_out, t0, rtol, atol, method)


def _fractions(values: Sequence[float], names: Sequence[str]) -> List[float]:
    vs = [_num(v, n, nonneg=True, hi=1.0) for v, n in zip(values, names)]
    if abs(sum(vs) - 1.0) > 1e-9:
        raise FamilyError("the initial fractions (" + ", ".join(names) + ") must sum to 1")
    return vs


def sir_spec(beta: float, gamma: float, s0: float, i0: float, t_out: Sequence[float], *, r0: Optional[float] = None, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """SIR epidemic on population fractions: S+I->2I (k=beta), I->R (k=gamma)."""
    b, g = _num(beta, "beta", nonneg=True), _num(gamma, "gamma", nonneg=True)
    rr = (1.0 - s0 - i0) if r0 is None else r0
    y0 = _fractions([s0, i0, rr], ["s0", "i0", "r0"])
    rx = [{"reactants": {"S": 1, "I": 1}, "products": {"I": 2}, "k": b}, {"reactants": {"I": 1}, "products": {"R": 1}, "k": g}]
    return mass_action_spec(["S", "I", "R"], rx, y0, t_out, spec_id=spec_id, **kw)


def seir_spec(beta: float, sigma: float, gamma: float, s0: float, e0: float, i0: float, t_out: Sequence[float], *, r0: Optional[float] = None,
              spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """SEIR epidemic: S+I->E+I (k=beta), E->I (k=sigma), I->R (k=gamma)."""
    b, s, g = _num(beta, "beta", nonneg=True), _num(sigma, "sigma", nonneg=True), _num(gamma, "gamma", nonneg=True)
    rr = (1.0 - s0 - e0 - i0) if r0 is None else r0
    y0 = _fractions([s0, e0, i0, rr], ["s0", "e0", "i0", "r0"])
    rx = [{"reactants": {"S": 1, "I": 1}, "products": {"E": 1, "I": 1}, "k": b}, {"reactants": {"E": 1}, "products": {"I": 1}, "k": s},
          {"reactants": {"I": 1}, "products": {"R": 1}, "k": g}]
    return mass_action_spec(["S", "E", "I", "R"], rx, y0, t_out, spec_id=spec_id, **kw)


def decay_chain_spec(rates: Sequence[float], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """First-order chain A->B->C...: len(rates) steps, len(rates)+1 species (letters A, B, C, ...)."""
    if not isinstance(rates, (list, tuple)) or not 1 <= len(rates) <= 10:
        raise FamilyError("rates must be a list of 1..10 numbers")
    names = [chr(ord("A") + i) for i in range(len(rates) + 1)]
    rx = [{"reactants": {names[i]: 1}, "products": {names[i + 1]: 1}, "k": _num(k, f"rates[{i}]", nonneg=True)} for i, k in enumerate(rates)]
    return mass_action_spec(names, rx, y0, t_out, spec_id=spec_id, **kw)


# ------------------------------------------------------------------ linear presets
def linear_spec(A: Sequence[Sequence[float]], b: Sequence[float], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", t0: float = 0.0,  # noqa: N803
                rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    n = len(A)
    if not 1 <= n <= 20 or any(len(row) != n for row in A) or len(b) != n or len(y0) != n:
        raise FamilyError("A must be n x n (n <= 20) and b, y0 of length n")
    mat = [[_num(v, f"A[{i}][{j}]") for j, v in enumerate(row)] for i, row in enumerate(A)]
    bb = [_num(v, f"b[{i}]") for i, v in enumerate(b)]
    ys = [_num(v, f"y0[{i}]") for i, v in enumerate(y0)]
    return _finish(spec_id, {"kind": "linear", "A": mat, "b": bb}, ys, t_out, t0, rtol, atol, method)


def series_rlc_spec(R: float, L: float, C: float, V: float, q0: float, i0: float, t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:  # noqa: N803
    """Series RLC driven by a constant voltage V: state [q, i], q' = i, i' = (V - R i - q/C)/L."""
    r, l_, c, v = _num(R, "R", nonneg=True), _num(L, "L", positive=True), _num(C, "C", positive=True), _num(V, "V")
    return linear_spec([[0.0, 1.0], [-1.0 / (l_ * c), -r / l_]], [0.0, v / l_], [q0, i0], t_out, spec_id=spec_id, **kw)


def spring_mass_damper_spec(m: float, c: float, k: float, F: float, x0: float, v0: float, t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:  # noqa: N803
    """Spring-mass-damper with constant force F: state [x, v], x' = v, v' = (F - c v - k x)/m."""
    mm, cc, kk, ff = _num(m, "m", positive=True), _num(c, "c", nonneg=True), _num(k, "k", nonneg=True), _num(F, "F")
    return linear_spec([[0.0, 1.0], [-kk / mm, -cc / mm]], [0.0, ff / mm], [x0, v0], t_out, spec_id=spec_id, **kw)


# ------------------------------------------------------------------ catalogue text for prompts and explanations
def describe_family(name: str) -> str:
    e = FAMILIES[name]
    params = ", ".join(f"{k} (default {d:g}: {m})" for k, (d, m) in e["params"].items())
    state = ", ".join(f"{s['name']} = {s['meaning']} [{s['unit']}]" for s in e["state"]) or "n cells, each an amount"
    inv = "; ".join(e["invariants"]) or "none"
    return f"- {name}: {e['summary']}. params: {params}. state y: [{state}]. invariants: {inv}."


def catalogue_text(only_new: bool = False) -> str:
    names = [n for n in FAMILIES if (n in NEW_FAMILIES or not only_new)]
    return "\n".join(describe_family(n) for n in names)


def presets_text() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in PRESETS.items())


# ------------------------------------------------------------------ named-slot compiler (pre-solver unit & derived-parameter layer)
SLOT_SCHEMA = "hivemind-solver-slots/1"

SLOT_FAMILIES = (
    "pendulum",
    "projectile_drag",
    "projectile",
    "kepler",
    "logistic",
    "sir",
    "seir",
    "series_rlc",
    "rlc",
    "spring_mass_damper",
    "msd",
    "decay_chain",
    "chain",
    "stiff",
    "harmonic",
    "robertson",
    "van_der_pol",
    "brusselator",
    "lotka_volterra",
    "stiff_chain",
    "yukawa_crystal",
    "qf_yukawa",
    "gpe_vortex",
    "qf_pgpe",
    "art_exo_genetics",
    "chromatin_remodeling",
    "bao_cosmology",
    "k3_quintessence",
)


def yukawa_crystal_spec(
    kappa: float,
    x0: float,
    v0: float,
    t_out: Sequence[float],
    *,
    damping: float = 0.2,
    spec_id: str = "science-bee-yukawa",
    **kw: Any,
) -> Dict[str, Any]:
    """2D Yukawa crystal phonon/defect mode relaxation around screened-Coulomb equilibrium."""
    kap = _num(kappa, "kappa", positive=True)
    dmp = _num(damping, "damping", nonneg=True)
    omega_eff = math.sqrt(1.0 + 2.0 * (kap ** 2) * math.exp(-kap))
    zeta = dmp / (2.0 * omega_eff)
    return named_spec("harmonic", {"omega": omega_eff, "zeta": zeta}, [x0, v0], t_out, spec_id=spec_id, **kw)


def gpe_vortex_spec(
    g_int: float,
    omega_trap: float,
    x0: float,
    v0: float,
    t_out: Sequence[float],
    *,
    zeta: float = 0.0,
    spec_id: str = "science-bee-gpe",
    **kw: Any,
) -> Dict[str, Any]:
    """Gross-Pitaevskii / PGPE Bogoliubov collective mode oscillation in a harmonic trap."""
    g_v = _num(g_int, "g_int", nonneg=True)
    om_t = _num(omega_trap, "omega_trap", positive=True)
    z_v = _num(zeta, "zeta", nonneg=True)
    omega_bog = om_t * math.sqrt(2.0 + g_v)
    return named_spec("harmonic", {"omega": omega_bog, "zeta": z_v}, [x0, v0], t_out, spec_id=spec_id, **kw)


def art_exo_genetics_spec(
    k_pol: float,
    k_exo: float,
    k_ext: float,
    t0_conc: float,
    m0_conc: float,
    p0_conc: float,
    t_out: Sequence[float],
    *,
    spec_id: str = "science-bee-art-exo",
    **kw: Any,
) -> Dict[str, Any]:
    """Chimeric ART-Exo reverse transcription + exonuclease proofreading kinetic network [T, M, P]."""
    kp = _num(k_pol, "k_pol", nonneg=True)
    ke = _num(k_exo, "k_exo", nonneg=True)
    kx = _num(k_ext, "k_ext", nonneg=True)
    rx = [
        {"reactants": {"T": 1}, "products": {"M": 1}, "k": kp},
        {"reactants": {"M": 1}, "products": {"T": 1}, "k": ke},
        {"reactants": {"M": 1}, "products": {"P": 1}, "k": kx},
    ]
    return mass_action_spec(["T", "M", "P"], rx, [t0_conc, m0_conc, p0_conc], t_out, spec_id=spec_id, **kw)


def bao_cosmology_spec(
    hubble: float,
    m_eff: float,
    phi0: float,
    dphi0: float,
    t_out: Sequence[float],
    *,
    source: float = 0.0,
    spec_id: str = "science-bee-cosmo",
    **kw: Any,
) -> Dict[str, Any]:
    """Klein-Gordon / quintessence cosmological scalar mode: phi'' + 3*H*phi' + m_eff^2*phi = source."""
    h_val = _num(hubble, "hubble", nonneg=True)
    m_val = _num(m_eff, "m_eff", nonneg=True)
    s_val = _num(source, "source")
    return linear_spec(
        [[0.0, 1.0], [-(m_val ** 2), -3.0 * h_val]],
        [0.0, s_val],
        [phi0, dphi0],
        t_out,
        spec_id=spec_id,
        **kw,
    )


def _slot_time_grid(raw: Mapping[str, Any], slots: Mapping[str, Any], conversions: List[str]) -> List[float]:
    for src_name, scale, unit_desc in (
        ("t_out", 1.0, None),
        ("t_out_ms", 1e-3, "ms -> s (/ 1000)"),
        ("t_out_min", 60.0, "min -> s (* 60)"),
        ("t_out_h", 3600.0, "h -> s (* 3600)"),
        ("t_out_days", 86400.0, "days -> s (* 86400)"),
    ):
        val = slots.get(src_name, raw.get(src_name))
        if val is not None:
            if not isinstance(val, (list, tuple)) or len(val) == 0:
                raise FamilyError(f"{src_name} must be a non-empty list of numbers")
            out = [_num(v, f"{src_name}[{i}]") * scale for i, v in enumerate(val)]
            if unit_desc is not None:
                conversions.append(f"{src_name} converted via {unit_desc}: {list(val)} -> {out}")
            return out
    raise FamilyError("slot specification requires t_out (or t_out_ms / t_out_min)")


def compile_slot_spec(
    slot_obj: Mapping[str, Any],
    *,
    return_trace: bool = False,
) -> Any:
    """Compile a `hivemind-solver-slots/1` dictionary into a validated `hivemind-solver-spec/1`.

    Performs deterministic pre-conversion of physical units (cm/g/kmh/ms/min/mH/mF/deg/%)
    and derived parameters (g/L, R0*gamma, 1/period, slope velocity components, circular
    orbit speed, damper force/speed) so the LLM only fills named slots instead of doing
    mental arithmetic.
    """
    if not isinstance(slot_obj, Mapping):
        raise FamilyError("slot_obj must be a JSON object")
    schema = slot_obj.get("schema")
    if schema is not None and schema != SLOT_SCHEMA:
        raise FamilyError(f"unsupported slot schema {schema!r} (expected {SLOT_SCHEMA!r})")

    raw_family = slot_obj.get("slot_family") or slot_obj.get("family") or slot_obj.get("preset")
    if not isinstance(raw_family, str) or raw_family.strip().lower() not in SLOT_FAMILIES:
        raise FamilyError(f"unknown slot_family {raw_family!r}; expected one of {SLOT_FAMILIES}")
    slot_family = raw_family.strip().lower()

    inner = slot_obj.get("slots")
    slots: Dict[str, Any] = dict(inner) if isinstance(inner, Mapping) else dict(slot_obj)

    conversions: List[str] = []
    derived: List[str] = []
    t_out = _slot_time_grid(slot_obj, slots, conversions)
    spec_id = str(slot_obj.get("id") or slots.get("id") or "science-bee-slots")
    t0 = _num(slot_obj.get("t0", slots.get("t0", 0.0)), "t0")
    rtol = slot_obj.get("rtol", slots.get("rtol"))
    atol = slot_obj.get("atol", slots.get("atol"))
    method = slot_obj.get("method", slots.get("method"))
    kw: Dict[str, Any] = {"spec_id": spec_id, "t0": t0, "rtol": rtol, "atol": atol, "method": method}

    if slot_family == "pendulum":
        g_val = _num(slots.get("g", 9.81), "g", positive=True)
        if "g_over_l" in slots and slots["g_over_l"] is not None:
            g_over_l = _num(slots["g_over_l"], "g_over_l", positive=True)
        elif "period_s" in slots and slots["period_s"] is not None:
            per_s = _num(slots["period_s"], "period_s", positive=True)
            g_over_l = (2.0 * math.pi / per_s) ** 2
            derived.append(f"g_over_l = (2*pi / period_s)^2 = (2*pi / {per_s:g})^2 = {g_over_l:.12g}")
        elif "period_ms" in slots and slots["period_ms"] is not None:
            per_ms = _num(slots["period_ms"], "period_ms", positive=True)
            per_s = per_ms / 1000.0
            conversions.append(f"period_ms={per_ms:g} ms -> period_s={per_s:g} s")
            g_over_l = (2.0 * math.pi / per_s) ** 2
            derived.append(f"g_over_l = (2*pi / period_s)^2 = {g_over_l:.12g}")
        else:
            if "length_m" in slots and slots["length_m"] is not None:
                length_m = _num(slots["length_m"], "length_m", positive=True)
            elif "L" in slots and slots["L"] is not None:
                length_m = _num(slots["L"], "L", positive=True)
            elif "l" in slots and slots["l"] is not None:
                length_m = _num(slots["l"], "l", positive=True)
            elif "length_cm" in slots and slots["length_cm"] is not None:
                length_cm = _num(slots["length_cm"], "length_cm", positive=True)
                length_m = length_cm / 100.0
                conversions.append(f"length_cm={length_cm:g} cm -> length_m={length_m:g} m")
            else:
                length_m = 1.0
            g_over_l = g_val / length_m
            derived.append(f"g_over_l = g / length_m = {g_val:g} / {length_m:g} = {g_over_l:.12g}")
        damping = _num(slots.get("damping", slots.get("gamma", 0.0)), "damping", nonneg=True)
        if "theta0" in slots and slots["theta0"] is not None:
            theta0 = _num(slots["theta0"], "theta0")
        elif "theta0_deg" in slots and slots["theta0_deg"] is not None:
            deg = _num(slots["theta0_deg"], "theta0_deg")
            theta0 = math.radians(deg)
            conversions.append(f"theta0_deg={deg:g} deg -> theta0={theta0:.12g} rad")
        else:
            raise FamilyError("pendulum slot requires theta0 or theta0_deg")
        omega0 = _num(slots.get("omega0", 0.0), "omega0")
        spec = pendulum_spec(theta0, omega0, t_out, g_over_l=g_over_l, damping=damping, **kw)

    elif slot_family in ("projectile_drag", "projectile"):
        g_val = _num(slots.get("g", 9.81), "g", positive=True)
        k_drag = _num(slots.get("k", slots.get("k_drag", 0.0)), "k", nonneg=True)
        x0 = _num(slots.get("x0", 0.0), "x0")
        y0_pos = _num(slots.get("y0", slots.get("height_m", 0.0)), "y0")
        if "vx0" in slots and slots["vx0"] is not None and "vy0" in slots and slots["vy0"] is not None:
            vx0 = _num(slots["vx0"], "vx0")
            vy0 = _num(slots["vy0"], "vy0")
        else:
            if "speed" in slots and slots["speed"] is not None:
                speed = _num(slots["speed"], "speed", nonneg=True)
            elif "v0" in slots and slots["v0"] is not None:
                speed = _num(slots["v0"], "v0", nonneg=True)
            elif "speed_kmh" in slots and slots["speed_kmh"] is not None:
                kmh = _num(slots["speed_kmh"], "speed_kmh", nonneg=True)
                speed = kmh / 3.6
                conversions.append(f"speed_kmh={kmh:g} km/h -> speed={speed:.12g} m/s")
            else:
                raise FamilyError("projectile slot requires (vx0, vy0) or speed / speed_kmh")
            if "angle_deg" in slots and slots["angle_deg"] is not None:
                deg = _num(slots["angle_deg"], "angle_deg")
                rad = math.radians(deg)
                vx0 = speed * math.cos(rad)
                vy0 = speed * math.sin(rad)
                derived.append(f"vx0 = speed*cos({deg:g} deg) = {vx0:.12g}, vy0 = speed*sin({deg:g} deg) = {vy0:.12g}")
            elif "angle_rad" in slots and slots["angle_rad"] is not None:
                rad = _num(slots["angle_rad"], "angle_rad")
                vx0 = speed * math.cos(rad)
                vy0 = speed * math.sin(rad)
                derived.append(f"vx0 = speed*cos({rad:g}) = {vx0:.12g}, vy0 = speed*sin({rad:g}) = {vy0:.12g}")
            elif "slope_rise" in slots and slots["slope_rise"] is not None and "slope_run" in slots and slots["slope_run"] is not None:
                rise = _num(slots["slope_rise"], "slope_rise")
                run = _num(slots["slope_run"], "slope_run", positive=True)
                hypot = math.hypot(run, rise)
                vx0 = speed * (run / hypot)
                vy0 = speed * (rise / hypot)
                derived.append(
                    f"slope ({rise:g}:{run:g}, hypot={hypot:g}) -> vx0={vx0:.12g}, vy0={vy0:.12g}"
                )
            else:
                vx0 = speed
                vy0 = 0.0
                derived.append(f"horizontal launch -> vx0={vx0:.12g}, vy0=0.0")
        spec = projectile_drag_spec(x0, y0_pos, vx0, vy0, t_out, g=g_val, k=k_drag, **kw)

    elif slot_family == "kepler":
        mu = _num(slots.get("mu", 1.0), "mu", positive=True)
        if "circular_radius" in slots and slots["circular_radius"] is not None:
            r0 = _num(slots["circular_radius"], "circular_radius", positive=True)
            clockwise = bool(slots.get("clockwise", False))
            x0, y0_pos = r0, 0.0
            vx0 = 0.0
            v_circ = math.sqrt(mu / r0)
            vy0 = -v_circ if clockwise else v_circ
            derived.append(f"circular orbit r0={r0:g}, mu={mu:g} -> x0={x0:g}, y0=0, vx0=0, vy0={vy0:.12g}")
        else:
            x0 = _num(slots.get("x0", 1.0), "x0")
            y0_pos = _num(slots.get("y0", 0.0), "y0")
            vx0 = _num(slots.get("vx0", 0.0), "vx0")
            vy0 = _num(slots["vy0"], "vy0")
        spec = kepler_spec(x0, y0_pos, vx0, vy0, t_out, mu=mu, **kw)

    elif slot_family == "logistic":
        if "r" in slots and slots["r"] is not None:
            r_val = _num(slots["r"], "r", positive=True)
        elif "r_per_min" in slots and slots["r_per_min"] is not None:
            rpm = _num(slots["r_per_min"], "r_per_min", positive=True)
            r_val = rpm / 60.0
            conversions.append(f"r_per_min={rpm:g} /min -> r={r_val:.12g} /s")
        elif "r_percent_per_min" in slots and slots["r_percent_per_min"] is not None:
            pct_min = _num(slots["r_percent_per_min"], "r_percent_per_min", positive=True)
            r_val = (pct_min / 100.0) / 60.0
            conversions.append(f"r_percent_per_min={pct_min:g}%/min -> r={r_val:.12g} /s")
        elif "r_percent" in slots and slots["r_percent"] is not None:
            pct = _num(slots["r_percent"], "r_percent", positive=True)
            r_val = pct / 100.0
            conversions.append(f"r_percent={pct:g}% -> r={r_val:.12g}")
        elif "doubling_time" in slots and slots["doubling_time"] is not None:
            td = _num(slots["doubling_time"], "doubling_time", positive=True)
            r_val = math.log(2.0) / td
            derived.append(f"r = ln(2)/doubling_time = ln(2)/{td:g} = {r_val:.12g}")
        else:
            r_val = 1.0
        k_cap = _num(slots.get("K", slots.get("carrying_capacity", 1.0)), "K", positive=True)
        n0_raw = slots.get("n0", slots.get("p0", slots.get("y0", [0.1])[0] if isinstance(slots.get("y0"), list) else slots.get("y0", 0.1)))
        n0 = _num(n0_raw, "n0")
        spec = logistic_spec(n0, t_out, r=r_val, K=k_cap, **kw)

    elif slot_family == "sir":
        if "gamma" in slots and slots["gamma"] is not None:
            gamma = _num(slots["gamma"], "gamma", nonneg=True)
        elif "infectious_period" in slots and slots["infectious_period"] is not None:
            inf_p = _num(slots["infectious_period"], "infectious_period", positive=True)
            gamma = 1.0 / inf_p
            derived.append(f"gamma = 1 / infectious_period = 1 / {inf_p:g} = {gamma:.12g}")
        else:
            raise FamilyError("sir slot requires gamma or infectious_period")

        if "beta" in slots and slots["beta"] is not None:
            beta = _num(slots["beta"], "beta", nonneg=True)
        elif "R0" in slots and slots["R0"] is not None:
            r0_repro = _num(slots["R0"], "R0", nonneg=True)
            beta = r0_repro * gamma
            derived.append(f"beta = R0 * gamma = {r0_repro:g} * {gamma:.12g} = {beta:.12g}")
        else:
            raise FamilyError("sir slot requires beta or R0")

        if "i0" in slots and slots["i0"] is not None:
            i0 = _num(slots["i0"], "i0")
        elif "i0_percent" in slots and slots["i0_percent"] is not None:
            pct = _num(slots["i0_percent"], "i0_percent")
            i0 = pct / 100.0
            conversions.append(f"i0_percent={pct:g}% -> i0={i0:.12g}")
        else:
            raise FamilyError("sir slot requires i0 or i0_percent")

        if "r0" in slots and slots["r0"] is not None:
            r0_init = _num(slots["r0"], "r0")
        elif "r0_percent" in slots and slots["r0_percent"] is not None:
            pct = _num(slots["r0_percent"], "r0_percent")
            r0_init = pct / 100.0
            conversions.append(f"r0_percent={pct:g}% -> r0={r0_init:.12g}")
        else:
            r0_init = 0.0

        if "s0" in slots and slots["s0"] is not None:
            s0 = _num(slots["s0"], "s0")
        elif "s0_percent" in slots and slots["s0_percent"] is not None:
            pct = _num(slots["s0_percent"], "s0_percent")
            s0 = pct / 100.0
            conversions.append(f"s0_percent={pct:g}% -> s0={s0:.12g}")
        else:
            s0 = 1.0 - i0 - r0_init
            derived.append(f"s0 = 1 - i0 - r0 = {s0:.12g}")
        spec = sir_spec(beta, gamma, s0, i0, t_out, r0=r0_init, **kw)

    elif slot_family == "seir":
        if "gamma" in slots and slots["gamma"] is not None:
            gamma = _num(slots["gamma"], "gamma", nonneg=True)
        elif "infectious_period" in slots and slots["infectious_period"] is not None:
            inf_p = _num(slots["infectious_period"], "infectious_period", positive=True)
            gamma = 1.0 / inf_p
            derived.append(f"gamma = 1 / infectious_period = 1 / {inf_p:g} = {gamma:.12g}")
        else:
            raise FamilyError("seir slot requires gamma or infectious_period")

        if "sigma" in slots and slots["sigma"] is not None:
            sigma = _num(slots["sigma"], "sigma", nonneg=True)
        elif "incubation_period" in slots and slots["incubation_period"] is not None:
            inc_p = _num(slots["incubation_period"], "incubation_period", positive=True)
            sigma = 1.0 / inc_p
            derived.append(f"sigma = 1 / incubation_period = 1 / {inc_p:g} = {sigma:.12g}")
        else:
            raise FamilyError("seir slot requires sigma or incubation_period")

        if "beta" in slots and slots["beta"] is not None:
            beta = _num(slots["beta"], "beta", nonneg=True)
        elif "R0" in slots and slots["R0"] is not None:
            r0_repro = _num(slots["R0"], "R0", nonneg=True)
            beta = r0_repro * gamma
            derived.append(f"beta = R0 * gamma = {r0_repro:g} * {gamma:.12g} = {beta:.12g}")
        else:
            raise FamilyError("seir slot requires beta or R0")

        if "e0" in slots and slots["e0"] is not None:
            e0 = _num(slots["e0"], "e0")
        elif "e0_percent" in slots and slots["e0_percent"] is not None:
            pct = _num(slots["e0_percent"], "e0_percent")
            e0 = pct / 100.0
            conversions.append(f"e0_percent={pct:g}% -> e0={e0:.12g}")
        else:
            raise FamilyError("seir slot requires e0 or e0_percent")

        if "i0" in slots and slots["i0"] is not None:
            i0 = _num(slots["i0"], "i0")
        elif "i0_percent" in slots and slots["i0_percent"] is not None:
            pct = _num(slots["i0_percent"], "i0_percent")
            i0 = pct / 100.0
            conversions.append(f"i0_percent={pct:g}% -> i0={i0:.12g}")
        else:
            raise FamilyError("seir slot requires i0 or i0_percent")

        if "r0" in slots and slots["r0"] is not None:
            r0_init = _num(slots["r0"], "r0")
        elif "r0_percent" in slots and slots["r0_percent"] is not None:
            pct = _num(slots["r0_percent"], "r0_percent")
            r0_init = pct / 100.0
            conversions.append(f"r0_percent={pct:g}% -> r0={r0_init:.12g}")
        else:
            r0_init = 0.0

        if "s0" in slots and slots["s0"] is not None:
            s0 = _num(slots["s0"], "s0")
        elif "s0_percent" in slots and slots["s0_percent"] is not None:
            pct = _num(slots["s0_percent"], "s0_percent")
            s0 = pct / 100.0
            conversions.append(f"s0_percent={pct:g}% -> s0={s0:.12g}")
        else:
            s0 = 1.0 - e0 - i0 - r0_init
            derived.append(f"s0 = 1 - e0 - i0 - r0 = {s0:.12g}")
        spec = seir_spec(beta, sigma, gamma, s0, e0, i0, t_out, r0=r0_init, **kw)

    elif slot_family in ("series_rlc", "rlc"):
        R = _num(slots.get("R", slots.get("R_ohm", 1.0)), "R", nonneg=True)
        if "L" in slots and slots["L"] is not None:
            L = _num(slots["L"], "L", positive=True)
        elif "L_mH" in slots and slots["L_mH"] is not None:
            lmh = _num(slots["L_mH"], "L_mH", positive=True)
            L = lmh / 1000.0
            conversions.append(f"L_mH={lmh:g} mH -> L={L:.12g} H")
        else:
            raise FamilyError("series_rlc slot requires L or L_mH")

        if "C" in slots and slots["C"] is not None:
            C = _num(slots["C"], "C", positive=True)
        elif "C_mF" in slots and slots["C_mF"] is not None:
            cmf = _num(slots["C_mF"], "C_mF", positive=True)
            C = cmf / 1000.0
            conversions.append(f"C_mF={cmf:g} mF -> C={C:.12g} F")
        elif "C_uF" in slots and slots["C_uF"] is not None:
            cuf = _num(slots["C_uF"], "C_uF", positive=True)
            C = cuf / 1e6
            conversions.append(f"C_uF={cuf:g} uF -> C={C:.12g} F")
        else:
            raise FamilyError("series_rlc slot requires C, C_mF, or C_uF")

        V = _num(slots.get("V", 0.0), "V")
        q0 = _num(slots.get("q0", 0.0), "q0")
        i0 = _num(slots.get("i0", 0.0), "i0")
        spec = series_rlc_spec(R, L, C, V, q0, i0, t_out, **kw)

    elif slot_family in ("spring_mass_damper", "msd"):
        if "m" in slots and slots["m"] is not None:
            m = _num(slots["m"], "m", positive=True)
        elif "m_g" in slots and slots["m_g"] is not None:
            mg = _num(slots["m_g"], "m_g", positive=True)
            m = mg / 1000.0
            conversions.append(f"m_g={mg:g} g -> m={m:.12g} kg")
        else:
            m = 1.0

        if "c" in slots and slots["c"] is not None:
            c = _num(slots["c"], "c", nonneg=True)
        elif "damper_force" in slots and slots["damper_force"] is not None and "damper_speed" in slots and slots["damper_speed"] is not None:
            df = _num(slots["damper_force"], "damper_force", nonneg=True)
            ds = _num(slots["damper_speed"], "damper_speed", positive=True)
            c = df / ds
            derived.append(f"c = damper_force / damper_speed = {df:g} / {ds:g} = {c:.12g}")
        else:
            c = 0.0

        k = _num(slots.get("k", 1.0), "k", nonneg=True)
        F = _num(slots.get("F", 0.0), "F")
        if "x0" in slots and slots["x0"] is not None:
            x0 = _num(slots["x0"], "x0")
        elif "x0_cm" in slots and slots["x0_cm"] is not None:
            xcm = _num(slots["x0_cm"], "x0_cm")
            x0 = xcm / 100.0
            conversions.append(f"x0_cm={xcm:g} cm -> x0={x0:.12g} m")
        else:
            x0 = 0.0
        v0 = _num(slots.get("v0", 0.0), "v0")
        spec = spring_mass_damper_spec(m, c, k, F, x0, v0, t_out, **kw)

    elif slot_family in ("decay_chain", "chain", "stiff"):
        if "rates" in slots and slots["rates"] is not None:
            rates = [_num(r, f"rates[{i}]", nonneg=True) for i, r in enumerate(slots["rates"])]
        elif "rates_per_min" in slots and slots["rates_per_min"] is not None:
            rpm = [_num(r, f"rates_per_min[{i}]", nonneg=True) for i, r in enumerate(slots["rates_per_min"])]
            rates = [r / 60.0 for r in rpm]
            conversions.append(f"rates_per_min={rpm} /min -> rates={rates} /s")
        elif "time_constants_s" in slots and slots["time_constants_s"] is not None:
            taus = [_num(t_c, f"time_constants_s[{i}]", positive=True) for i, t_c in enumerate(slots["time_constants_s"])]
            rates = [1.0 / t_c for t_c in taus]
            derived.append(f"rates = [1/tau for tau in {taus}] = {rates} /s")
        else:
            raise FamilyError("decay_chain slot requires rates, rates_per_min, or time_constants_s")
        if "y0" in slots and slots["y0"] is not None:
            y0_vec = [_num(v, f"y0[{i}]") for i, v in enumerate(slots["y0"])]
        else:
            a0 = _num(slots.get("a0", 1.0), "a0")
            y0_vec = [a0] + [0.0] * len(rates)
        spec = decay_chain_spec(rates, y0_vec, t_out, **kw)

    elif slot_family in ("yukawa_crystal", "qf_yukawa"):
        kappa = _num(slots.get("kappa", 1.0), "kappa", positive=True)
        damping = _num(slots.get("damping", 0.2), "damping", nonneg=True)
        x0 = _num(slots.get("x0", 0.1), "x0")
        v0 = _num(slots.get("v0", 0.0), "v0")
        omega_eff = math.sqrt(1.0 + 2.0 * (kappa ** 2) * math.exp(-kappa))
        zeta = damping / (2.0 * omega_eff)
        derived.append(
            f"Yukawa phonon mode: omega_eff = sqrt(1 + 2*kappa^2*exp(-kappa)) = {omega_eff:.12g}, zeta = {zeta:.12g}"
        )
        spec = yukawa_crystal_spec(kappa, x0, v0, t_out, damping=damping, **kw)

    elif slot_family in ("gpe_vortex", "qf_pgpe"):
        g_int = _num(slots.get("g_int", slots.get("g", 2.0)), "g_int", nonneg=True)
        omega_trap = _num(slots.get("omega_trap", slots.get("omega", 1.0)), "omega_trap", positive=True)
        zeta = _num(slots.get("zeta", slots.get("damping", 0.0)), "zeta", nonneg=True)
        x0 = _num(slots.get("x0", 0.1), "x0")
        v0 = _num(slots.get("v0", 0.0), "v0")
        omega_bog = omega_trap * math.sqrt(2.0 + g_int)
        derived.append(f"Bogoliubov mode: omega_B = omega_trap * sqrt(2 + g_int) = {omega_bog:.12g}")
        spec = gpe_vortex_spec(g_int, omega_trap, x0, v0, t_out, zeta=zeta, **kw)

    elif slot_family in ("art_exo_genetics", "chromatin_remodeling"):
        k_pol = _num(slots.get("k_pol", 0.5), "k_pol", nonneg=True)
        k_exo = _num(slots.get("k_exo", 0.2), "k_exo", nonneg=True)
        k_ext = _num(slots.get("k_ext", 0.1), "k_ext", nonneg=True)
        t0_c = _num(slots.get("T0", slots.get("t0_conc", 1.0)), "T0", nonneg=True)
        m0_c = _num(slots.get("M0", slots.get("m0_conc", 0.0)), "M0", nonneg=True)
        p0_c = _num(slots.get("P0", slots.get("p0_conc", 0.0)), "P0", nonneg=True)
        derived.append(f"ART-Exo proofreading network: k_pol={k_pol:g}, k_exo={k_exo:g}, k_ext={k_ext:g}")
        spec = art_exo_genetics_spec(k_pol, k_exo, k_ext, t0_c, m0_c, p0_c, t_out, **kw)

    elif slot_family in ("bao_cosmology", "k3_quintessence"):
        hubble = _num(slots.get("hubble", slots.get("H", 0.1)), "hubble", nonneg=True)
        m_eff = _num(slots.get("m_eff", slots.get("mass", 1.0)), "m_eff", nonneg=True)
        phi0 = _num(slots.get("phi0", slots.get("x0", 1.0)), "phi0")
        dphi0 = _num(slots.get("dphi0", slots.get("v0", 0.0)), "dphi0")
        source = _num(slots.get("source", 0.0), "source")
        derived.append(f"Klein-Gordon / quintessence linear system: 3*H={3.0 * hubble:.12g}, m_eff^2={m_eff ** 2:.12g}")
        spec = bao_cosmology_spec(hubble, m_eff, phi0, dphi0, t_out, source=source, **kw)

    else:
        # Canonical named family in FAMILIES (harmonic, robertson, van_der_pol, brusselator, lotka_volterra, stiff_chain)
        params = dict(slots.get("params") or {})
        for k_param in FAMILIES[slot_family]["params"]:
            if k_param in slots and slots[k_param] is not None and k_param not in params:
                params[k_param] = slots[k_param]
        if slot_family == "harmonic" and "omega" not in params:
            if "k" in slots and "m" in slots and slots["k"] is not None and slots["m"] is not None:
                kk = _num(slots["k"], "k", positive=True)
                mm = _num(slots["m"], "m", positive=True)
                params["omega"] = math.sqrt(kk / mm)
                derived.append(f"omega = sqrt(k/m) = sqrt({kk:g}/{mm:g}) = {params['omega']:.12g}")
        y0_raw = slots.get("y0")
        if y0_raw is None:
            state_meta = FAMILIES[slot_family]["state"]
            if state_meta and all(s["name"] in slots for s in state_meta):
                y0_raw = [slots[s["name"]] for s in state_meta]
            elif slot_family == "harmonic" and "x0" in slots:
                y0_raw = [slots["x0"], slots.get("v0", 0.0)]
            else:
                raise FamilyError(f"{slot_family} slot requires y0")
        spec = named_spec(slot_family, params, y0_raw, t_out, **kw)

    trace = {
        "schema": SLOT_SCHEMA,
        "slot_family": slot_family,
        "problem_kind": spec["problem"]["kind"],
        "problem_family": spec["problem"].get("family"),
        "conversions": conversions,
        "derived_parameters": derived,
        "resolved_y0": list(spec["y0"]),
        "resolved_t_out": list(spec["t_out"]),
    }
    return (spec, trace) if return_trace else spec

