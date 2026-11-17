// Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
//! hivemind-solver: Solver Bee wave 1. JSON spec in, JSON result out, nothing evaluated as code.
//!
//! The six original families, `linear` and `mass_action` use only `+ - * /` (and integer powers by
//! repeated multiplication), which IEEE-754 rounds identically on every conforming CPU. The four
//! science families (pendulum, projectile_drag, kepler, logistic) additionally use `sin`, `cos` and
//! `sqrt`, and ONLY through the `mathfn` module: `sqrt` is correctly rounded, `sin`/`cos` go to the
//! system libm (not correctly rounded; see docs/solver_bee/SOLVER_CLI.md). The pinned CVODE internals
//! also call `powf`/`sqrt`.

mod mathfn;

use cvode::{Cvode, DenseMat, Method, Task};
use nvector::SerialVector;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The rusty-SUNDIALS git revision pinned in Cargo.toml (a unit test checks they match).
pub const SOLVER_REV: &str = "4e3e32de9ff188bace35c9e80e433b4e795a9d76";
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MAX_INPUT_BYTES: usize = 64 * 1024;
pub const MAX_T_OUT: usize = 1000;
pub const MAX_DIM: usize = 20;
pub const MAX_REACTIONS: usize = 50;
pub const MAX_STOICH: u32 = 6;
pub const MAX_PARAM_ABS: f64 = 1e12;
const SPEC_SCHEMA: &str = "hivemind-solver-spec/1";
const RESULT_SCHEMA: &str = "hivemind-solver-result/1";

// ---------------------------------------------------------------- spec types

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema: String,
    id: String,
    problem: Problem,
    y0: Vec<f64>,
    #[serde(default)]
    t0: f64,
    t_out: Vec<f64>,
    #[serde(default = "default_rtol")]
    rtol: f64,
    #[serde(default = "default_atol")]
    atol: Atol,
    #[serde(default)]
    method: MethodSel,
    #[serde(default = "default_max_steps")]
    max_steps: u64,
}

fn default_rtol() -> f64 {
    1e-8
}
fn default_atol() -> Atol {
    Atol::Scalar(1e-10)
}
fn default_max_steps() -> u64 {
    100_000
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Atol {
    Scalar(f64),
    Array(Vec<f64>),
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
enum MethodSel {
    #[default]
    Bdf,
    Adams,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Problem {
    Named(NamedSpec),
    Linear(LinearSpec),
    MassAction(MassSpec),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedSpec {
    family: String,
    #[serde(default)]
    params: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LinearSpec {
    #[serde(rename = "A")]
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MassSpec {
    species: Vec<String>,
    reactions: Vec<ReactionSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReactionSpec {
    #[serde(default)]
    reactants: BTreeMap<String, u32>,
    #[serde(default)]
    products: BTreeMap<String, u32>,
    k: f64,
}

// ---------------------------------------------------------------- models

/// (name, default) pairs, in the order shown by `families`.
fn family_params(family: &str) -> Option<&'static [(&'static str, f64)]> {
    Some(match family {
        "robertson" => &[("k1", 0.04), ("k2", 3.0e7), ("k3", 1.0e4)],
        "van_der_pol" => &[("mu", 1.0)],
        "brusselator" => &[("a", 1.0), ("b", 3.0)],
        "lotka_volterra" => &[("alpha", 1.5), ("beta", 1.0), ("delta", 1.0), ("gamma", 3.0)],
        "harmonic" => &[("omega", 1.0), ("zeta", 0.0)],
        "stiff_chain" => &[("n", 10.0), ("k", 100.0)],
        "pendulum" => &[("g_over_l", 9.81), ("damping", 0.0)],
        "projectile_drag" => &[("g", 9.81), ("k", 0.0)],
        "kepler" => &[("mu", 1.0)],
        "logistic" => &[("r", 1.0), ("K", 1.0)],
        _ => return None,
    })
}
const FAMILIES: [&str; 10] = [
    "robertson",
    "van_der_pol",
    "brusselator",
    "lotka_volterra",
    "harmonic",
    "stiff_chain",
    "pendulum",
    "projectile_drag",
    "kepler",
    "logistic",
];

enum Model {
    Robertson { k1: f64, k2: f64, k3: f64 },
    VanDerPol { mu: f64 },
    Brusselator { a: f64, b: f64 },
    LotkaVolterra { alpha: f64, beta: f64, delta: f64, gamma: f64 },
    Harmonic { omega: f64, zeta: f64 },
    StiffChain { n: usize, k: f64 },
    Pendulum { g_over_l: f64, damping: f64 },
    ProjectileDrag { g: f64, k: f64 },
    Kepler { mu: f64 },
    Logistic { r: f64, big_k: f64 },
    Linear { n: usize, a: Vec<f64>, b: Vec<f64> }, // a is row-major n x n
    Mass(MassModel),
}

struct MassModel {
    n: usize,
    /// per reaction: k, reactant (species, nu) pairs, net stoichiometry per species
    rxns: Vec<MassRxn>,
}
struct MassRxn {
    k: f64,
    reactants: Vec<(usize, u32)>,
    net: Vec<i32>,
}

fn ipow(x: f64, n: u32) -> f64 {
    let mut r = 1.0;
    for _ in 0..n {
        r *= x;
    }
    r
}

impl MassRxn {
    fn rate(&self, y: &[f64]) -> f64 {
        let mut r = self.k;
        for &(s, nu) in &self.reactants {
            r *= ipow(y[s], nu);
        }
        r
    }
    /// d rate / d y_s for one reactant entry index `skip`
    fn partial(&self, y: &[f64], skip: usize) -> f64 {
        let mut r = self.k;
        for (idx, &(s, nu)) in self.reactants.iter().enumerate() {
            if idx == skip {
                r *= f64::from(nu) * ipow(y[s], nu - 1);
            } else {
                r *= ipow(y[s], nu);
            }
        }
        r
    }
}

impl Model {
    fn dim(&self) -> usize {
        match self {
            Model::Robertson { .. } => 3,
            Model::VanDerPol { .. } | Model::Brusselator { .. } => 2,
            Model::LotkaVolterra { .. } | Model::Harmonic { .. } | Model::Pendulum { .. } => 2,
            Model::ProjectileDrag { .. } | Model::Kepler { .. } => 4,
            Model::Logistic { .. } => 1,
            Model::StiffChain { n, .. } | Model::Linear { n, .. } => *n,
            Model::Mass(m) => m.n,
        }
    }

    fn rhs(&self, y: &[f64], f: &mut [f64]) {
        match self {
            Model::Robertson { k1, k2, k3 } => {
                let (a, b, c) = (y[0], y[1], y[2]);
                let bc = k3 * b * c;
                let bb = k2 * b * b;
                f[0] = -k1 * a + bc;
                f[1] = k1 * a - bc - bb;
                f[2] = bb;
            }
            Model::VanDerPol { mu } => {
                f[0] = y[1];
                f[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            }
            Model::Brusselator { a, b } => {
                let (u, v) = (y[0], y[1]);
                let u2v = u * u * v;
                f[0] = a + u2v - (b + 1.0) * u;
                f[1] = b * u - u2v;
            }
            Model::LotkaVolterra { alpha, beta, delta, gamma } => {
                let (x, z) = (y[0], y[1]);
                f[0] = alpha * x - beta * x * z;
                f[1] = delta * x * z - gamma * z;
            }
            Model::Harmonic { omega, zeta } => {
                f[0] = y[1];
                f[1] = -(omega * omega) * y[0] - 2.0 * zeta * omega * y[1];
            }
            Model::Pendulum { g_over_l, damping } => {
                f[0] = y[1];
                f[1] = -g_over_l * mathfn::sin(y[0]) - damping * y[1];
            }
            Model::ProjectileDrag { g, k } => {
                // quadratic drag a = -k |v| v, gravity -g in y
                let s = mathfn::sqrt(y[2] * y[2] + y[3] * y[3]);
                f[0] = y[2];
                f[1] = y[3];
                f[2] = -k * s * y[2];
                f[3] = -g - k * s * y[3];
            }
            Model::Kepler { mu } => {
                // r^3 = r2 * sqrt(r2)
                let r2 = y[0] * y[0] + y[1] * y[1];
                let r3 = r2 * mathfn::sqrt(r2);
                f[0] = y[2];
                f[1] = y[3];
                f[2] = -mu * y[0] / r3;
                f[3] = -mu * y[1] / r3;
            }
            Model::Logistic { r, big_k } => {
                f[0] = r * y[0] * (1.0 - y[0] / big_k);
            }
            Model::StiffChain { n, k } => {
                // Reflecting (Neumann) discrete diffusion chain; sum(y) is conserved.
                for i in 0..*n {
                    let left = if i > 0 { y[i - 1] - y[i] } else { 0.0 };
                    let right = if i + 1 < *n { y[i + 1] - y[i] } else { 0.0 };
                    f[i] = k * (left + right);
                }
            }
            Model::Linear { n, a, b } => {
                for i in 0..*n {
                    let mut s = 0.0;
                    for j in 0..*n {
                        s += a[i * n + j] * y[j];
                    }
                    f[i] = s + b[i];
                }
            }
            Model::Mass(m) => {
                for v in f.iter_mut() {
                    *v = 0.0;
                }
                for r in &m.rxns {
                    let rate = r.rate(y);
                    for (s, &nu) in r.net.iter().enumerate() {
                        if nu != 0 {
                            f[s] += f64::from(nu) * rate;
                        }
                    }
                }
            }
        }
    }

    /// Analytic Jacobian, column-major: j.cols[c][r] = d f_r / d y_c.
    fn jac(&self, y: &[f64], j: &mut DenseMat) {
        let n = self.dim();
        for c in 0..n {
            for r in 0..n {
                j.cols[c][r] = 0.0;
            }
        }
        match self {
            Model::Robertson { k1, k2, k3 } => {
                let (b, c) = (y[1], y[2]);
                j.cols[0][0] = -k1;
                j.cols[1][0] = k3 * c;
                j.cols[2][0] = k3 * b;
                j.cols[0][1] = *k1;
                j.cols[1][1] = -k3 * c - 2.0 * k2 * b;
                j.cols[2][1] = -k3 * b;
                j.cols[1][2] = 2.0 * k2 * b;
            }
            Model::VanDerPol { mu } => {
                j.cols[1][0] = 1.0;
                j.cols[0][1] = -2.0 * mu * y[0] * y[1] - 1.0;
                j.cols[1][1] = mu * (1.0 - y[0] * y[0]);
            }
            Model::Brusselator { b, .. } => {
                let (u, v) = (y[0], y[1]);
                j.cols[0][0] = 2.0 * u * v - (b + 1.0);
                j.cols[1][0] = u * u;
                j.cols[0][1] = b - 2.0 * u * v;
                j.cols[1][1] = -(u * u);
            }
            Model::LotkaVolterra { alpha, beta, delta, gamma } => {
                let (x, z) = (y[0], y[1]);
                j.cols[0][0] = alpha - beta * z;
                j.cols[1][0] = -beta * x;
                j.cols[0][1] = delta * z;
                j.cols[1][1] = delta * x - gamma;
            }
            Model::Harmonic { omega, zeta } => {
                j.cols[1][0] = 1.0;
                j.cols[0][1] = -(omega * omega);
                j.cols[1][1] = -2.0 * zeta * omega;
            }
            Model::Pendulum { g_over_l, damping } => {
                j.cols[1][0] = 1.0;
                j.cols[0][1] = -g_over_l * mathfn::cos(y[0]);
                j.cols[1][1] = -damping;
            }
            Model::ProjectileDrag { k, .. } => {
                let (vx, vy) = (y[2], y[3]);
                let s = mathfn::sqrt(vx * vx + vy * vy);
                j.cols[2][0] = 1.0;
                j.cols[3][1] = 1.0;
                if s > 0.0 {
                    // d(|v| vx)/dvx = |v| + vx^2/|v|, d(|v| vx)/dvy = vx vy/|v|; the limit at v=0 is 0
                    j.cols[2][2] = -k * (s + vx * vx / s);
                    j.cols[3][2] = -k * vx * vy / s;
                    j.cols[2][3] = -k * vx * vy / s;
                    j.cols[3][3] = -k * (s + vy * vy / s);
                }
            }
            Model::Kepler { mu } => {
                let (x, yy) = (y[0], y[1]);
                let r2 = x * x + yy * yy;
                let r5 = r2 * r2 * mathfn::sqrt(r2);
                j.cols[2][0] = 1.0;
                j.cols[3][1] = 1.0;
                j.cols[0][2] = -mu * (r2 - 3.0 * x * x) / r5;
                j.cols[1][2] = 3.0 * mu * x * yy / r5;
                j.cols[0][3] = 3.0 * mu * x * yy / r5;
                j.cols[1][3] = -mu * (r2 - 3.0 * yy * yy) / r5;
            }
            Model::Logistic { r, big_k } => {
                j.cols[0][0] = r * (1.0 - 2.0 * y[0] / big_k);
            }
            Model::StiffChain { n, k } => {
                for i in 0..*n {
                    if i > 0 {
                        j.cols[i - 1][i] += k;
                        j.cols[i][i] -= k;
                    }
                    if i + 1 < *n {
                        j.cols[i + 1][i] += k;
                        j.cols[i][i] -= k;
                    }
                }
            }
            Model::Linear { n, a, .. } => {
                for c in 0..*n {
                    for r in 0..*n {
                        j.cols[c][r] = a[r * n + c];
                    }
                }
            }
            Model::Mass(m) => {
                for r in &m.rxns {
                    for (idx, &(s, _)) in r.reactants.iter().enumerate() {
                        let d = r.partial(y, idx);
                        for (row, &nu) in r.net.iter().enumerate() {
                            if nu != 0 {
                                j.cols[s][row] += f64::from(nu) * d;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- validation

fn finite(name: &str, x: f64) -> Result<(), String> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be a finite number"))
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-')
}

fn build_named(ns: &NamedSpec) -> Result<Model, String> {
    let defaults = family_params(&ns.family).ok_or_else(|| {
        format!("unknown family '{}'; expected one of {}", ns.family, FAMILIES.join(", "))
    })?;
    for (k, v) in &ns.params {
        if !defaults.iter().any(|(name, _)| name == k) {
            let valid: Vec<&str> = defaults.iter().map(|(n, _)| *n).collect();
            return Err(format!(
                "unknown parameter '{k}' for family '{}'; valid: {}",
                ns.family,
                valid.join(", ")
            ));
        }
        finite(&format!("param {k}"), *v)?;
        if v.abs() > MAX_PARAM_ABS {
            return Err(format!("param {k} exceeds the magnitude limit {MAX_PARAM_ABS:e}"));
        }
    }
    let p = |name: &str| -> f64 {
        ns.params
            .get(name)
            .copied()
            .unwrap_or_else(|| defaults.iter().find(|(n, _)| *n == name).map(|(_, d)| *d).unwrap_or(0.0))
    };
    let nonneg = |name: &str| -> Result<f64, String> {
        let v = p(name);
        if v < 0.0 {
            Err(format!("param {name} must be >= 0"))
        } else {
            Ok(v)
        }
    };
    Ok(match ns.family.as_str() {
        "robertson" => Model::Robertson { k1: nonneg("k1")?, k2: nonneg("k2")?, k3: nonneg("k3")? },
        "van_der_pol" => Model::VanDerPol { mu: nonneg("mu")? },
        "brusselator" => Model::Brusselator { a: nonneg("a")?, b: nonneg("b")? },
        "lotka_volterra" => Model::LotkaVolterra {
            alpha: p("alpha"),
            beta: p("beta"),
            delta: p("delta"),
            gamma: p("gamma"),
        },
        "harmonic" => Model::Harmonic { omega: p("omega"), zeta: nonneg("zeta")? },
        "pendulum" => Model::Pendulum { g_over_l: nonneg("g_over_l")?, damping: nonneg("damping")? },
        "projectile_drag" => Model::ProjectileDrag { g: nonneg("g")?, k: nonneg("k")? },
        "kepler" => {
            let mu = p("mu");
            if mu <= 0.0 {
                return Err("param mu must be > 0".into());
            }
            Model::Kepler { mu }
        }
        "logistic" => {
            let big_k = p("K");
            if big_k <= 0.0 {
                return Err("param K must be > 0".into());
            }
            Model::Logistic { r: p("r"), big_k }
        }
        "stiff_chain" => {
            let nf = p("n");
            if nf.fract() != 0.0 || !(2.0..=MAX_DIM as f64).contains(&nf) {
                return Err(format!("param n must be an integer in 2..={MAX_DIM}"));
            }
            Model::StiffChain { n: nf as usize, k: nonneg("k")? }
        }
        _ => unreachable!(),
    })
}

fn build_linear(ls: &LinearSpec) -> Result<Model, String> {
    let n = ls.a.len();
    if n == 0 || n > MAX_DIM {
        return Err(format!("A must be a square matrix with 1..={MAX_DIM} rows"));
    }
    if ls.b.len() != n {
        return Err(format!("b has length {} but A has {n} rows", ls.b.len()));
    }
    let mut a = Vec::with_capacity(n * n);
    for (i, row) in ls.a.iter().enumerate() {
        if row.len() != n {
            return Err(format!("A row {i} has length {} but A has {n} rows (must be square)", row.len()));
        }
        for (j, v) in row.iter().enumerate() {
            finite(&format!("A[{i}][{j}]"), *v)?;
            a.push(*v);
        }
    }
    for (i, v) in ls.b.iter().enumerate() {
        finite(&format!("b[{i}]"), *v)?;
    }
    Ok(Model::Linear { n, a, b: ls.b.clone() })
}

fn build_mass(ms: &MassSpec) -> Result<Model, String> {
    let n = ms.species.len();
    if n == 0 || n > MAX_DIM {
        return Err(format!("species must have 1..={MAX_DIM} entries"));
    }
    let mut index = BTreeMap::new();
    for (i, s) in ms.species.iter().enumerate() {
        let ok = !s.is_empty()
            && s.len() <= 32
            && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'+' || c == b'-');
        if !ok {
            return Err(format!("species name {s:?} must match [A-Za-z0-9_+-]{{1,32}}"));
        }
        if index.insert(s.clone(), i).is_some() {
            return Err(format!("duplicate species '{s}'"));
        }
    }
    if ms.reactions.is_empty() || ms.reactions.len() > MAX_REACTIONS {
        return Err(format!("reactions must have 1..={MAX_REACTIONS} entries"));
    }
    let mut rxns = Vec::new();
    for (ri, r) in ms.reactions.iter().enumerate() {
        finite(&format!("reaction {ri} k"), r.k)?;
        if r.k < 0.0 || r.k > MAX_PARAM_ABS {
            return Err(format!("reaction {ri}: k must be in [0, {MAX_PARAM_ABS:e}]"));
        }
        if r.reactants.is_empty() && r.products.is_empty() {
            return Err(format!("reaction {ri} has neither reactants nor products"));
        }
        let mut net = vec![0i32; n];
        let mut reactants = Vec::new();
        for (name, &nu) in &r.reactants {
            let s = *index.get(name).ok_or_else(|| format!("reaction {ri}: unknown species '{name}'"))?;
            if nu == 0 || nu > MAX_STOICH {
                return Err(format!("reaction {ri}: coefficient of '{name}' must be an integer in 1..={MAX_STOICH}"));
            }
            reactants.push((s, nu));
            net[s] -= nu as i32;
        }
        for (name, &nu) in &r.products {
            let s = *index.get(name).ok_or_else(|| format!("reaction {ri}: unknown species '{name}'"))?;
            if nu == 0 || nu > MAX_STOICH {
                return Err(format!("reaction {ri}: coefficient of '{name}' must be an integer in 1..={MAX_STOICH}"));
            }
            net[s] += nu as i32;
        }
        rxns.push(MassRxn { k: r.k, reactants, net });
    }
    Ok(Model::Mass(MassModel { n, rxns }))
}

struct Validated {
    id: String,
    model: Model,
    y0: Vec<f64>,
    t0: f64,
    t_out: Vec<f64>,
    rtol: f64,
    atol: f64,
    method: MethodSel,
    max_steps: usize,
}

fn validate(spec: Spec) -> Result<Validated, String> {
    if spec.schema != SPEC_SCHEMA {
        return Err(format!("schema must be \"{SPEC_SCHEMA}\""));
    }
    if !valid_id(&spec.id) {
        return Err("id must match [A-Za-z0-9_.-]{1,64}".into());
    }
    let model = match &spec.problem {
        Problem::Named(ns) => build_named(ns)?,
        Problem::Linear(ls) => build_linear(ls)?,
        Problem::MassAction(ms) => build_mass(ms)?,
    };
    let n = model.dim();
    if spec.y0.len() != n {
        return Err(format!("y0 has length {} but the problem has {n} state variables", spec.y0.len()));
    }
    for (i, v) in spec.y0.iter().enumerate() {
        finite(&format!("y0[{i}]"), *v)?;
    }
    if matches!(model, Model::Kepler { .. }) && spec.y0[0] == 0.0 && spec.y0[1] == 0.0 {
        return Err("kepler: the initial position must not be the origin (r = 0)".into());
    }
    finite("t0", spec.t0)?;
    if spec.t_out.is_empty() || spec.t_out.len() > MAX_T_OUT {
        return Err(format!("t_out must have 1..={MAX_T_OUT} entries"));
    }
    let mut prev = spec.t0;
    for (i, &t) in spec.t_out.iter().enumerate() {
        finite(&format!("t_out[{i}]"), t)?;
        if t <= prev {
            return Err(format!("t_out must be strictly increasing and > t0 (violated at index {i})"));
        }
        prev = t;
    }
    finite("rtol", spec.rtol)?;
    if !(spec.rtol > 0.0 && spec.rtol < 1.0) {
        return Err("rtol must satisfy 0 < rtol < 1".into());
    }
    let atol = match &spec.atol {
        Atol::Scalar(a) => *a,
        Atol::Array(v) => {
            if v.len() != n {
                return Err(format!("atol array has length {} but the problem has {n} state variables", v.len()));
            }
            for (i, a) in v.iter().enumerate() {
                finite(&format!("atol[{i}]"), *a)?;
            }
            let first = v[0];
            if v.iter().any(|a| *a != first) {
                return Err("per-component atol with different values is not supported by the pinned solver (scalar atol only); give equal entries or a scalar".into());
            }
            first
        }
    };
    finite("atol", atol)?;
    if atol <= 0.0 {
        return Err("atol must be > 0".into());
    }
    if spec.max_steps < 1 || spec.max_steps > 1_000_000 {
        return Err("max_steps must be in 1..=1000000".into());
    }
    Ok(Validated {
        id: spec.id,
        model,
        y0: spec.y0,
        t0: spec.t0,
        t_out: spec.t_out,
        rtol: spec.rtol,
        atol,
        method: spec.method,
        max_steps: spec.max_steps as usize,
    })
}

// ---------------------------------------------------------------- solving

#[derive(Serialize)]
struct Stats {
    steps: usize,
    rhs_evals: usize,
    jac_evals: Option<usize>,
    last_order: Option<usize>,
}

#[derive(Serialize)]
struct SolverInfo {
    name: &'static str,
    rev: &'static str,
    method: &'static str,
    crate_version: &'static str,
}

#[derive(Serialize)]
struct ResultDoc<'a> {
    schema: &'static str,
    id: Option<&'a str>,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    t: Option<&'a [f64]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<&'a [Vec<f64>]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<Stats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    solver: Option<SolverInfo>,
    input_sha256: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub fn error_result(input: &[u8], id: Option<&str>, msg: &str) -> String {
    let doc = ResultDoc {
        schema: RESULT_SCHEMA,
        id,
        status: "error",
        error: Some(msg),
        t: None,
        y: None,
        stats: None,
        solver: None,
        input_sha256: sha256_hex(input),
    };
    serde_json::to_string(&doc).unwrap_or_else(|_| String::from("{\"status\":\"error\"}"))
}

fn solve(v: &Validated) -> Result<(Vec<Vec<f64>>, Stats), String> {
    let model = Arc::new(v.model_clone());
    let m1 = Arc::clone(&model);
    let m2 = Arc::clone(&model);
    let rhs = move |_t: f64, y: &[f64], f: &mut [f64]| -> Result<(), String> {
        m1.rhs(y, f);
        if f.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err("non-finite right-hand side (state diverged)".to_string())
        }
    };
    let jac = move |_t: f64, y: &[f64], j: &mut DenseMat| -> Result<(), String> {
        m2.jac(y, j);
        Ok(())
    };
    let method = match v.method {
        MethodSel::Bdf => Method::Bdf,
        MethodSel::Adams => Method::Adams,
    };
    let mut solver = Cvode::builder(method)
        .rtol(v.rtol)
        .atol(v.atol)
        .max_steps(v.max_steps)
        .jacobian(jac)
        .build(rhs, v.t0, SerialVector::from_slice(&v.y0))
        .map_err(|e| format!("solver setup failed: {e}"))?;
    let mut ys = Vec::with_capacity(v.t_out.len());
    for &tout in &v.t_out {
        let y = match solver.solve(tout, Task::Normal) {
            Ok((_t, y)) => y.to_vec(),
            Err(e) => return Err(format!("solver failed before t = {tout}: {e}")),
        };
        if y.iter().any(|x| !x.is_finite()) {
            return Err(format!("non-finite solution at t = {tout}"));
        }
        ys.push(y);
    }
    let stats = Stats {
        steps: solver.num_steps(),
        rhs_evals: solver.num_rhs_evals(),
        jac_evals: None, // the pinned solver exposes no Jacobian-evaluation counter
        last_order: Some(solver.order()),
    };
    Ok((ys, stats))
}

impl Validated {
    fn model_clone(&self) -> Model {
        match &self.model {
            Model::Robertson { k1, k2, k3 } => Model::Robertson { k1: *k1, k2: *k2, k3: *k3 },
            Model::VanDerPol { mu } => Model::VanDerPol { mu: *mu },
            Model::Brusselator { a, b } => Model::Brusselator { a: *a, b: *b },
            Model::LotkaVolterra { alpha, beta, delta, gamma } => {
                Model::LotkaVolterra { alpha: *alpha, beta: *beta, delta: *delta, gamma: *gamma }
            }
            Model::Harmonic { omega, zeta } => Model::Harmonic { omega: *omega, zeta: *zeta },
            Model::StiffChain { n, k } => Model::StiffChain { n: *n, k: *k },
            Model::Pendulum { g_over_l, damping } => Model::Pendulum { g_over_l: *g_over_l, damping: *damping },
            Model::ProjectileDrag { g, k } => Model::ProjectileDrag { g: *g, k: *k },
            Model::Kepler { mu } => Model::Kepler { mu: *mu },
            Model::Logistic { r, big_k } => Model::Logistic { r: *r, big_k: *big_k },
            Model::Linear { n, a, b } => Model::Linear { n: *n, a: a.clone(), b: b.clone() },
            Model::Mass(m) => Model::Mass(MassModel {
                n: m.n,
                rxns: m
                    .rxns
                    .iter()
                    .map(|r| MassRxn { k: r.k, reactants: r.reactants.clone(), net: r.net.clone() })
                    .collect(),
            }),
        }
    }
}

fn id_hint(input: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(input).ok()?;
    let id = v.get("id")?.as_str()?;
    if valid_id(id) {
        Some(id.to_string())
    } else {
        None
    }
}

fn run_inner(input: &[u8]) -> String {
    if input.len() > MAX_INPUT_BYTES {
        return error_result(input, None, &format!("input exceeds {MAX_INPUT_BYTES} bytes"));
    }
    let spec: Spec = match serde_json::from_slice(input) {
        Ok(s) => s,
        Err(e) => return error_result(input, id_hint(input).as_deref(), &format!("invalid spec: {e}")),
    };
    let v = match validate(spec) {
        Ok(v) => v,
        Err(e) => return error_result(input, id_hint(input).as_deref(), &e),
    };
    match solve(&v) {
        Ok((ys, stats)) => {
            let doc = ResultDoc {
                schema: RESULT_SCHEMA,
                id: Some(&v.id),
                status: "ok",
                error: None,
                t: Some(&v.t_out),
                y: Some(&ys),
                stats: Some(stats),
                solver: Some(SolverInfo {
                    name: "rusty-sundials/cvode",
                    rev: SOLVER_REV,
                    method: match v.method {
                        MethodSel::Bdf => "bdf",
                        MethodSel::Adams => "adams",
                    },
                    crate_version: CRATE_VERSION,
                }),
                input_sha256: sha256_hex(input),
            };
            serde_json::to_string(&doc)
                .unwrap_or_else(|e| error_result(input, Some(&v.id), &format!("cannot serialise result: {e}")))
        }
        Err(e) => error_result(input, Some(&v.id), &e),
    }
}

/// Run one spec (exact input bytes). Never panics outward: an internal panic becomes status "error".
pub fn run_bytes(input: &[u8]) -> String {
    match std::panic::catch_unwind(|| run_inner(input)) {
        Ok(s) => s,
        Err(_) => error_result(input, id_hint(input).as_deref(), "internal error (panic) while solving"),
    }
}

pub fn version_json() -> String {
    serde_json::json!({"crate_version": CRATE_VERSION, "solver_rev": SOLVER_REV}).to_string()
}

pub fn families_json() -> String {
    let mut fams = Vec::new();
    for f in FAMILIES {
        let params: Vec<serde_json::Value> = family_params(f)
            .unwrap_or(&[])
            .iter()
            .map(|(n, d)| serde_json::json!({"name": n, "default": d}))
            .collect();
        let (dim, desc) = match f {
            "robertson" => ("3", "y1'=-k1 y1+k3 y2 y3; y2'=k1 y1-k3 y2 y3-k2 y2^2; y3'=k2 y2^2 (conserves y1+y2+y3)"),
            "van_der_pol" => ("2", "x'=v; v'=mu (1-x^2) v - x"),
            "brusselator" => ("2", "u'=a+u^2 v-(b+1)u; v'=b u-u^2 v"),
            "lotka_volterra" => ("2", "x'=alpha x-beta x y; y'=delta x y-gamma y (first integral delta x-gamma ln x+beta y-alpha ln y)"),
            "harmonic" => ("2", "x'=v; v'=-omega^2 x-2 zeta omega v"),
            "pendulum" => ("2", "y=[theta, omega]; theta'=omega; omega'=-g_over_l sin(theta)-damping omega (energy omega^2/2+g_over_l (1-cos theta) conserved when damping=0)"),
            "projectile_drag" => ("4", "y=[x, y, vx, vy]; x'=vx; y'=vy; v'=-k |v| v-[0,g] (quadratic drag; no ground-hit event, the caller chooses t_out)"),
            "kepler" => ("4", "y=[x, y, vx, vy]; planar two-body, x''=-mu x/r^3, y''=-mu y/r^3 (conserves energy |v|^2/2-mu/r and angular momentum x vy-y vx)"),
            "logistic" => ("1", "y=[N]; N'=r N (1-N/K)"),
            _ => ("n", "reflecting diffusion chain y_i'=k(y_{i-1}-2y_i+y_{i+1}), ends reflecting (conserves sum), n integer in 2..=20"),
        };
        fams.push(serde_json::json!({"family": f, "dim": dim, "equations": desc, "params": params}));
    }
    serde_json::json!({
        "spec_schema": SPEC_SCHEMA,
        "kinds": ["named", "linear", "mass_action"],
        "families": fams,
        "limits": {"max_input_bytes": MAX_INPUT_BYTES, "max_t_out": MAX_T_OUT, "max_dim": MAX_DIM,
                   "max_reactions": MAX_REACTIONS, "max_stoich_coefficient": MAX_STOICH}
    })
    .to_string()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn run(spec: &str) -> serde_json::Value {
        serde_json::from_str(&run_bytes(spec.as_bytes())).unwrap()
    }
    fn ys(v: &serde_json::Value) -> Vec<Vec<f64>> {
        v["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
            .collect()
    }
    fn ok(v: &serde_json::Value) {
        assert_eq!(v["status"], "ok", "{v}");
    }
    fn err_contains(v: &serde_json::Value, s: &str) {
        assert_eq!(v["status"], "error", "{v}");
        let e = v["error"].as_str().unwrap();
        assert!(e.contains(s), "error {e:?} lacks {s:?}");
        assert!(v.get("y").is_none());
    }

    #[test]
    fn rev_constant_matches_cargo_toml() {
        let toml = include_str!("../Cargo.toml");
        assert_eq!(SOLVER_REV.len(), 40);
        assert!(SOLVER_REV.bytes().all(|c| c.is_ascii_hexdigit()));
        let needle = format!("rev = \"{SOLVER_REV}\"");
        assert_eq!(toml.matches(&needle).count(), 2, "both cvode and nvector must be pinned to SOLVER_REV");
        // no unpinned rusty-SUNDIALS reference
        for line in toml.lines().filter(|l| l.contains("rusty-SUNDIALS") && l.contains("git =")) {
            assert!(line.contains(&needle), "unpinned git dependency: {line}");
        }
    }

    #[test]
    fn harmonic_matches_closed_form() {
        let w = 2.0f64;
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"h","problem":{"kind":"named","family":"harmonic","params":{"omega":2.0}},
          "y0":[1.0,0.0],"t_out":[0.5,1.0,3.0],"rtol":1e-10,"atol":1e-12}"#;
        let v = run(spec);
        ok(&v);
        for (i, t) in [0.5f64, 1.0, 3.0].iter().enumerate() {
            let y = &ys(&v)[i];
            assert!((y[0] - (w * t).cos()).abs() < 1e-8, "x at {t}");
            assert!((y[1] + w * (w * t).sin()).abs() < 1e-8, "v at {t}");
        }
    }

    #[test]
    fn damped_harmonic_energy_decays() {
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"hd","problem":{"kind":"named","family":"harmonic","params":{"omega":1.0,"zeta":0.1}},
          "y0":[1.0,0.0],"t_out":[1.0,2.0,5.0]}"#;
        let v = run(spec);
        ok(&v);
        let e: Vec<f64> = ys(&v).iter().map(|y| y[0] * y[0] + y[1] * y[1]).collect();
        assert!(e[0] < 1.0 && e[1] < e[0] && e[2] < e[1]);
    }

    #[test]
    fn linear_small_hand_checked() {
        // x' = -x + 1, x(0)=0  =>  x(t) = 1 - exp(-t);  y' = 2 (A=0,b=2) => 2t
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"lin","problem":{"kind":"linear","A":[[-1.0,0.0],[0.0,0.0]],"b":[1.0,2.0]},
          "y0":[0.0,0.0],"t_out":[1.0,2.0],"rtol":1e-10,"atol":1e-12}"#;
        let v = run(spec);
        ok(&v);
        let y = ys(&v);
        assert!((y[0][0] - (1.0 - (-1.0f64).exp())).abs() < 1e-8);
        assert!((y[1][0] - (1.0 - (-2.0f64).exp())).abs() < 1e-8);
        assert!((y[0][1] - 2.0).abs() < 1e-8 && (y[1][1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn decay_chain_mass_action_closed_form() {
        // A -> B -> C with k1=1, k2=2: A=e^-t, B=k1/(k2-k1)(e^-t - e^-2t), C=1-A-B
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"chain","problem":{"kind":"mass_action","species":["A","B","C"],
          "reactions":[{"reactants":{"A":1},"products":{"B":1},"k":1.0},{"reactants":{"B":1},"products":{"C":1},"k":2.0}]},
          "y0":[1.0,0.0,0.0],"t_out":[0.5,1.0,4.0],"rtol":1e-10,"atol":1e-13}"#;
        let v = run(spec);
        ok(&v);
        for (i, t) in [0.5f64, 1.0, 4.0].iter().enumerate() {
            let y = &ys(&v)[i];
            let a = (-t).exp();
            let b = (-t).exp() - (-2.0 * t).exp();
            assert!((y[0] - a).abs() < 1e-8, "A at {t}");
            assert!((y[1] - b).abs() < 1e-8, "B at {t}");
            assert!((y[0] + y[1] + y[2] - 1.0).abs() < 1e-9, "mass conservation at {t}");
        }
    }

    #[test]
    fn dimerisation_conserves_weighted_mass() {
        // 2A <-> D : A + 2D is conserved
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"dim","problem":{"kind":"mass_action","species":["A","D"],
          "reactions":[{"reactants":{"A":2},"products":{"D":1},"k":1.0},{"reactants":{"D":1},"products":{"A":2},"k":0.5}]},
          "y0":[1.0,0.0],"t_out":[0.5,5.0,50.0],"rtol":1e-10,"atol":1e-13}"#;
        let v = run(spec);
        ok(&v);
        for y in ys(&v) {
            assert!((y[0] + 2.0 * y[1] - 1.0).abs() < 1e-9);
            assert!(y[0] > 0.0 && y[1] > 0.0);
        }
    }

    #[test]
    fn robertson_conserves_sum() {
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"rob","problem":{"kind":"named","family":"robertson"},
          "y0":[1.0,0.0,0.0],"t_out":[0.4,40.0,4000.0],"rtol":1e-8,"atol":1e-12}"#;
        let v = run(spec);
        ok(&v);
        let y = ys(&v);
        for r in &y {
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-8);
        }
        // classic reference values at t=0.4 (Hindmarsh): y1 ~ 0.98517
        assert!((y[0][0] - 0.985172).abs() < 1e-4);
    }

    #[test]
    fn lotka_volterra_first_integral_nearly_conserved() {
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"lv","problem":{"kind":"named","family":"lotka_volterra"},
          "y0":[1.0,1.0],"t_out":[1.0,5.0,10.0],"rtol":1e-10,"atol":1e-12}"#;
        let v = run(spec);
        ok(&v);
        let h = |y: &Vec<f64>| 1.0 * y[0] - 3.0 * y[0].ln() + 1.0 * y[1] - 1.5 * y[1].ln();
        let h0 = h(&vec![1.0, 1.0]);
        for y in ys(&v) {
            assert!((h(&y) - h0).abs() < 1e-6);
        }
    }

    #[test]
    fn stiff_chain_conserves_sum_and_equilibrates() {
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"sc","problem":{"kind":"named","family":"stiff_chain","params":{"n":6,"k":50.0}},
          "y0":[6.0,0.0,0.0,0.0,0.0,0.0],"t_out":[0.01,10.0],"rtol":1e-9,"atol":1e-12}"#;
        let v = run(spec);
        ok(&v);
        let y = ys(&v);
        for r in &y {
            assert!((r.iter().sum::<f64>() - 6.0).abs() < 1e-8);
        }
        for x in &y[1] {
            assert!((x - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn van_der_pol_and_brusselator_run() {
        let v = run(r#"{"schema":"hivemind-solver-spec/1","id":"vdp","problem":{"kind":"named","family":"van_der_pol","params":{"mu":1.0}},
          "y0":[2.0,0.0],"t_out":[1.0,10.0]}"#);
        ok(&v);
        let v = run(r#"{"schema":"hivemind-solver-spec/1","id":"br","problem":{"kind":"named","family":"brusselator"},
          "y0":[1.5,3.0],"t_out":[1.0,10.0],"method":"adams"}"#);
        ok(&v);
        assert_eq!(v["solver"]["method"], "adams");
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let models = vec![
            Model::Robertson { k1: 0.04, k2: 3e7, k3: 1e4 },
            Model::VanDerPol { mu: 3.0 },
            Model::Brusselator { a: 1.0, b: 3.0 },
            Model::LotkaVolterra { alpha: 1.5, beta: 1.0, delta: 1.0, gamma: 3.0 },
            Model::Harmonic { omega: 2.0, zeta: 0.3 },
            Model::StiffChain { n: 4, k: 7.0 },
            Model::Pendulum { g_over_l: 9.81, damping: 0.4 },
            Model::ProjectileDrag { g: 9.81, k: 0.3 },
            Model::Kepler { mu: 2.5 },
            Model::Logistic { r: 0.7, big_k: 3.0 },
            build_mass(&MassSpec {
                species: vec!["A".into(), "B".into(), "C".into()],
                reactions: vec![ReactionSpec {
                    reactants: [("A".to_string(), 2u32), ("B".to_string(), 1u32)].into_iter().collect(),
                    products: [("C".to_string(), 1u32)].into_iter().collect(),
                    k: 1.3,
                }],
            })
            .unwrap(),
        ];
        for m in &models {
            let n = m.dim();
            let y: Vec<f64> = (0..n).map(|i| 0.5 + 0.37 * i as f64).collect();
            let mut j = DenseMat::zeros(n as _, n as _);
            m.jac(&y, &mut j);
            for c in 0..n {
                let h = 1e-6 * y[c].abs().max(1.0);
                let (mut yp, mut ym) = (y.clone(), y.clone());
                yp[c] += h;
                ym[c] -= h;
                let (mut fp, mut fm) = (vec![0.0; n], vec![0.0; n]);
                m.rhs(&yp, &mut fp);
                m.rhs(&ym, &mut fm);
                for r in 0..n {
                    let fd = (fp[r] - fm[r]) / (2.0 * h);
                    let tol = 1e-5 * fd.abs().max(j.cols[c][r].abs()).max(1.0) + 1e-15 * (fp[r].abs() + fm[r].abs()) / h;
                    assert!((fd - j.cols[c][r]).abs() < tol, "J[{r}][{c}] fd={fd} an={}", j.cols[c][r]);
                }
            }
        }
    }

    #[test]
    fn determinism_ten_runs_byte_identical() {
        let spec = r#"{"schema":"hivemind-solver-spec/1","id":"det","problem":{"kind":"named","family":"robertson"},
          "y0":[1.0,0.0,0.0],"t_out":[0.4,4.0,40.0,400.0],"rtol":1e-8,"atol":1e-12}"#;
        let first = run_bytes(spec.as_bytes());
        assert!(first.contains("\"status\":\"ok\""));
        for _ in 0..9 {
            assert_eq!(first, run_bytes(spec.as_bytes()));
        }
    }

    #[test]
    fn input_hash_is_of_exact_bytes() {
        let a = r#"{"schema":"hivemind-solver-spec/1","id":"x","problem":{"kind":"named","family":"harmonic"},"y0":[1,0],"t_out":[1]}"#;
        let b = format!("{a} ");
        let va = run(a);
        let vb = run(&b);
        assert_eq!(va["input_sha256"], sha256_hex(a.as_bytes()));
        assert_ne!(va["input_sha256"], vb["input_sha256"]);
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn validation_errors_are_status_error_not_panics() {
        let base = |extra: &str| {
            format!(r#"{{"schema":"hivemind-solver-spec/1","id":"e","problem":{{"kind":"named","family":"harmonic"}},"y0":[1,0],"t_out":[1]{extra}}}"#)
        };
        err_contains(&run(&base(r#","bogus":1"#)), "unknown field");
        err_contains(&run(&base(r#","rtol":0"#)), "rtol");
        err_contains(&run(&base(r#","atol":-1"#)), "atol");
        err_contains(&run(&base(r#","atol":[1e-10,1e-9]"#)), "per-component");
        err_contains(&run(&base(r#","max_steps":0"#)), "max_steps");
        err_contains(&run(&base(r#","max_steps":2000000"#)), "max_steps");
        err_contains(&run(&base(r#","method":"rk4""#)), "unknown variant");
        err_contains(&run(&base(r#","t0":1.0"#)), "strictly increasing");
        // wrong y0 size
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"harmonic"},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "y0 has length");
        // non-increasing t_out
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"harmonic"},"y0":[1,0],"t_out":[2,1]}"#;
        err_contains(&run(s), "strictly increasing");
        // empty t_out
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"harmonic"},"y0":[1,0],"t_out":[]}"#;
        err_contains(&run(s), "t_out");
        // unknown family and parameter
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"nope"},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "unknown family");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"harmonic","params":{"q":1}},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "unknown parameter");
        // wrong schema, bad id
        let s = r#"{"schema":"v2","id":"e","problem":{"kind":"named","family":"harmonic"},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "schema");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"bad id!","problem":{"kind":"named","family":"harmonic"},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "id must match");
        // non-numeric / overflow numbers and garbage
        err_contains(&run("not json"), "invalid spec");
        err_contains(&run(&base(r#","rtol":1e999"#)), "invalid spec");
        err_contains(&run(""), "invalid spec");
    }

    #[test]
    fn validation_linear_and_mass_action() {
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"linear","A":[[1,2],[3]],"b":[0,0]},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "square");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"linear","A":[[1,2],[3,4]],"b":[0]},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "b has length");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"linear","A":[],"b":[]},"y0":[],"t_out":[1]}"#;
        err_contains(&run(s), "A must be");
        let big = format!("[{}]", vec!["0"; 21].join(","));
        let rows = vec![big.clone(); 21].join(",");
        let s = format!(r#"{{"schema":"hivemind-solver-spec/1","id":"e","problem":{{"kind":"linear","A":[{rows}],"b":{big}}},"y0":{big},"t_out":[1]}}"#);
        err_contains(&run(&s), "A must be");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A","A"],"reactions":[{"reactants":{"A":1},"products":{},"k":1}]},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "duplicate species");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A"],"reactions":[{"reactants":{"Z":1},"products":{},"k":1}]},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "unknown species");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A"],"reactions":[{"reactants":{"A":0},"products":{},"k":1}]},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "coefficient");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A"],"reactions":[{"reactants":{"A":1.5},"products":{},"k":1}]},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "invalid spec");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A"],"reactions":[{"reactants":{"A":1},"products":{},"k":-1}]},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "k must be");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"mass_action","species":["A"],"reactions":[]},"y0":[1],"t_out":[1]}"#;
        err_contains(&run(s), "reactions");
        let rx = r#"{"reactants":{"A":1},"products":{},"k":1}"#;
        let many = vec![rx; 51].join(",");
        let s = format!(r#"{{"schema":"hivemind-solver-spec/1","id":"e","problem":{{"kind":"mass_action","species":["A"],"reactions":[{many}]}},"y0":[1],"t_out":[1]}}"#);
        err_contains(&run(&s), "reactions");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"stiff_chain","params":{"n":2.5}},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), "integer");
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"e","problem":{"kind":"named","family":"van_der_pol","params":{"mu":-1}},"y0":[1,0],"t_out":[1]}"#;
        err_contains(&run(s), ">= 0");
    }

    #[test]
    fn oversized_input_is_rejected() {
        let mut s = String::from(r#"{"schema":"hivemind-solver-spec/1","id":"big","pad":""#);
        s.push_str(&"x".repeat(MAX_INPUT_BYTES));
        s.push_str("\"}");
        err_contains(&run(&s), "exceeds");
    }

    #[test]
    fn max_steps_exhaustion_is_a_clean_error() {
        let s = r#"{"schema":"hivemind-solver-spec/1","id":"ms","problem":{"kind":"named","family":"van_der_pol","params":{"mu":100}},
          "y0":[2,0],"t_out":[1000],"max_steps":5}"#;
        let v = run(s);
        assert_eq!(v["status"], "error");
        assert_eq!(v["id"], "ms");
    }

    #[test]
    fn version_and_families_json_parse() {
        let v: serde_json::Value = serde_json::from_str(&version_json()).unwrap();
        assert_eq!(v["solver_rev"], SOLVER_REV);
        let f: serde_json::Value = serde_json::from_str(&families_json()).unwrap();
        assert_eq!(f["families"].as_array().unwrap().len(), 10);
    }

    // ---- science families -------------------------------------------------------------------

    fn spec_named(id: &str, family: &str, params: &str, y0: &str, t_out: &str, tol: &str) -> String {
        format!(
            r#"{{"schema":"hivemind-solver-spec/1","id":"{id}","problem":{{"kind":"named","family":"{family}","params":{params}}},"y0":{y0},"t_out":{t_out}{tol}}}"#
        )
    }

    #[test]
    fn pendulum_small_angle_matches_linear_closed_form() {
        // theta0 = 1e-3: the neglected amplitude-dependent frequency shift gives < 1e-9 absolute error.
        let spec = spec_named("pn", "pendulum", r#"{"g_over_l":4.0}"#, "[0.001,0.0]", "[0.5,1.0,2.0,3.0]",
            r#","rtol":1e-11,"atol":1e-15"#);
        let v = run(&spec);
        ok(&v);
        for (row, t) in ys(&v).iter().zip([0.5f64, 1.0, 2.0, 3.0]) {
            assert!((row[0] - 0.001 * (2.0 * t).cos()).abs() < 1e-9, "theta at {t}: {}", row[0]);
            assert!((row[1] + 0.002 * (2.0 * t).sin()).abs() < 1e-9, "omega at {t}: {}", row[1]);
        }
    }

    #[test]
    fn pendulum_conserves_energy_and_damping_dissipates() {
        let spec = spec_named("pe", "pendulum", r#"{"g_over_l":9.81}"#, "[2.5,0.0]", "[1.0,5.0,10.0]",
            r#","rtol":1e-10,"atol":1e-12"#);
        let v = run(&spec);
        ok(&v);
        let e0 = 9.81 * (1.0 - 2.5f64.cos());
        for row in ys(&v) {
            let e = 0.5 * row[1] * row[1] + 9.81 * (1.0 - row[0].cos());
            assert!((e - e0).abs() < 1e-6, "energy drift {}", e - e0);
        }
        let spec = spec_named("pd", "pendulum", r#"{"g_over_l":9.81,"damping":0.5}"#, "[1.0,0.0]", "[5.0,80.0]",
            r#","rtol":1e-9,"atol":1e-12"#);
        let rows = ys(&run(&spec));
        assert!(rows[1][0].abs() < 1e-6, "damped pendulum must settle: {:?}", rows[1]);
    }

    #[test]
    fn projectile_without_drag_is_a_parabola() {
        let spec = spec_named("pj", "projectile_drag", r#"{"g":9.81}"#, "[0.0,0.0,30.0,40.0]", "[1.0,2.0,4.0,8.0]",
            r#","rtol":1e-11,"atol":1e-12"#);
        let v = run(&spec);
        ok(&v);
        for (row, t) in ys(&v).iter().zip([1.0f64, 2.0, 4.0, 8.0]) {
            assert!((row[0] - 30.0 * t).abs() < 1e-7);
            assert!((row[1] - (40.0 * t - 0.5 * 9.81 * t * t)).abs() < 1e-7);
            assert!((row[2] - 30.0).abs() < 1e-7);
            assert!((row[3] - (40.0 - 9.81 * t)).abs() < 1e-7);
        }
    }

    #[test]
    fn projectile_with_quadratic_drag_matches_vertical_fall_closed_form() {
        // dropped from rest: vy = -vt tanh(g t / vt), vt = sqrt(g/k)
        let (g, k) = (9.81f64, 0.1f64);
        let vt = (g / k).sqrt();
        let spec = spec_named("pf", "projectile_drag", r#"{"g":9.81,"k":0.1}"#, "[0.0,0.0,0.0,0.0]", "[0.5,1.0,3.0,10.0]",
            r#","rtol":1e-11,"atol":1e-12"#);
        let v = run(&spec);
        ok(&v);
        for (row, t) in ys(&v).iter().zip([0.5f64, 1.0, 3.0, 10.0]) {
            assert!((row[3] + vt * (g * t / vt).tanh()).abs() < 1e-8, "vy at {t}");
            assert!(row[2].abs() < 1e-12 && row[0].abs() < 1e-12);
        }
    }

    #[test]
    fn kepler_circular_orbit_has_the_right_period() {
        // mu = 4, r = 1: omega = sqrt(mu/r^3) = 2, period = pi.
        let pi = std::f64::consts::PI;
        let t_out = format!("[{},{},{},{},{}]", pi / 4.0, pi / 2.0, pi, 2.0 * pi, 5.0 * pi);
        let spec = spec_named("kc", "kepler", r#"{"mu":4.0}"#, "[1.0,0.0,0.0,2.0]", &t_out, r#","rtol":1e-11,"atol":1e-13"#);
        let v = run(&spec);
        ok(&v);
        let rows = ys(&v);
        let y0 = [1.0, 0.0, 0.0, 2.0];
        // quarter period, half period, then whole periods (pi, 2 pi, 5 pi): back to the start
        let want = [[0.0, 1.0, -2.0, 0.0], [-1.0, 0.0, 0.0, -2.0], y0, y0, y0];
        for (row, w) in rows.iter().zip(want) {
            for (a, b) in row.iter().zip(w) {
                assert!((a - b).abs() < 1e-6, "circular orbit {row:?} vs {w:?}");
            }
        }
    }

    #[test]
    fn kepler_eccentric_conserves_energy_and_angular_momentum() {
        let e = 0.5f64;
        let vp = ((1.0 + e) / (1.0 - e)).sqrt();
        let y0 = format!("[{},0.0,0.0,{}]", 1.0 - e, vp);
        let spec = spec_named("ke", "kepler", "{}", &y0, "[1.0,3.0,6.0,12.0,25.0]", r#","rtol":1e-11,"atol":1e-13"#);
        let v = run(&spec);
        ok(&v);
        let en = |r: &[f64]| 0.5 * (r[2] * r[2] + r[3] * r[3]) - 1.0 / (r[0] * r[0] + r[1] * r[1]).sqrt();
        let ang = |r: &[f64]| r[0] * r[3] - r[1] * r[2];
        let (e0, l0) = (en(&[1.0 - e, 0.0, 0.0, vp]), ang(&[1.0 - e, 0.0, 0.0, vp]));
        assert!((e0 + 0.5).abs() < 1e-12, "a = 1 so E = -1/2");
        for row in ys(&v) {
            assert!((en(&row) - e0).abs() < 1e-7, "energy drift {}", en(&row) - e0);
            assert!((ang(&row) - l0).abs() < 1e-7, "angular momentum drift {}", ang(&row) - l0);
        }
    }

    #[test]
    fn logistic_matches_closed_form() {
        let (r, big_k, n0) = (0.8f64, 50.0f64, 2.0f64);
        let spec = spec_named("lg", "logistic", r#"{"r":0.8,"K":50.0}"#, "[2.0]", "[1.0,5.0,10.0,30.0]",
            r#","rtol":1e-11,"atol":1e-13"#);
        let v = run(&spec);
        ok(&v);
        for (row, t) in ys(&v).iter().zip([1.0f64, 5.0, 10.0, 30.0]) {
            let exact = big_k / (1.0 + (big_k / n0 - 1.0) * (-r * t).exp());
            assert!((row[0] - exact).abs() < 1e-8 * exact.max(1.0), "N at {t}: {} vs {exact}", row[0]);
        }
    }

    #[test]
    fn science_family_parameter_validation_is_strict() {
        let y4 = "[1.0,0.0,0.0,1.0]";
        err_contains(&run(&spec_named("e", "pendulum", r#"{"damping":-0.1}"#, "[1,0]", "[1]", "")), "damping");
        err_contains(&run(&spec_named("e", "pendulum", r#"{"g_over_l":-1}"#, "[1,0]", "[1]", "")), "g_over_l");
        err_contains(&run(&spec_named("e", "pendulum", r#"{"mu":1}"#, "[1,0]", "[1]", "")), "unknown parameter");
        err_contains(&run(&spec_named("e", "projectile_drag", r#"{"k":-1}"#, y4, "[1]", "")), "must be >= 0");
        err_contains(&run(&spec_named("e", "projectile_drag", r#"{"g":-1}"#, y4, "[1]", "")), "must be >= 0");
        err_contains(&run(&spec_named("e", "kepler", r#"{"mu":0}"#, y4, "[1]", "")), "mu");
        err_contains(&run(&spec_named("e", "kepler", r#"{"mu":-1}"#, y4, "[1]", "")), "mu");
        err_contains(&run(&spec_named("e", "kepler", "{}", "[0.0,0.0,1.0,1.0]", "[1]", "")), "origin");
        err_contains(&run(&spec_named("e", "logistic", r#"{"K":0}"#, "[1]", "[1]", "")), "K");
        err_contains(&run(&spec_named("e", "logistic", r#"{"K":-3}"#, "[1]", "[1]", "")), "K");
        err_contains(&run(&spec_named("e", "logistic", "{}", "[1,2]", "[1]", "")), "y0 has length");
        err_contains(&run(&spec_named("e", "kepler", "{}", "[1,0]", "[1]", "")), "y0 has length");
        // defaults are accepted
        ok(&run(&spec_named("d", "pendulum", "{}", "[0.1,0]", "[1]", "")));
        ok(&run(&spec_named("d", "logistic", "{}", "[0.1]", "[1]", "")));
    }

    #[test]
    fn drag_jacobian_is_finite_at_zero_velocity() {
        let m = Model::ProjectileDrag { g: 9.81, k: 0.5 };
        let mut j = DenseMat::zeros(4, 4);
        m.jac(&[0.0, 0.0, 0.0, 0.0], &mut j);
        for c in 0..4 {
            for r in 0..4 {
                assert!(j.cols[c][r].is_finite());
            }
        }
    }
}
