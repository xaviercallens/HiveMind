# Solver Bee: held-out problem set and first run (stage 2, P3)

Directory `solvers/problems/heldout/`: 16 problems, schema `hivemind-solver-problem/1`, tolerance policy `max_rel_err_factor = 100` (unchanged from the library). The set was written and committed with its sha256 manifest `FROZEN.json` in commit `6da1869` ("freeze held-out set (before any solver run)") **before the solver was ever run on it**; the numbers below come from the first recorded run (`solvers/results/heldout_run_1.json`). The git history shows the freeze commit preceding the results commit, but it cannot prove that no run happened before the freeze: the ordering is a statement of process, not a cryptographic proof. Nothing in the frozen files, the references or the policy was edited afterwards (`tests/test_solver_problems.py` fails if a frozen file changes, goes missing or an unlisted file appears).

## What is in the set

| group | problems | reference method (primary / cross-check) |
|---|---|---|
| pendulum (new family) | `pendulum_small_angle` (0.05 rad, t<=10), `pendulum_large_angle` (2.8 rad, 80 time units, about 7 periods), `pendulum_damped` | undamped: exact Jacobi elliptic solution `sin(theta/2)=k cd(w0 t, k^2)` (mpmath) / mpmath Taylor series; damped: mpmath Taylor / scipy DOP853 |
| projectile (new family) | `projectile_no_drag` (exact parabola), `projectile_drag` (k=0.02) | closed form / DOP853; Taylor / DOP853 |
| Kepler (new family) | `kepler_circular` (20 orbits), `kepler_eccentric` (e=0.6, about 4 orbits), `kepler_eccentric_long` (e=0.5, 100 orbits, t=200 pi) | exact Lagrange f,g propagation with Kepler's equation solved at 40 digits / mpmath Taylor (20 digits) |
| logistic (new family) | `logistic_growth` | closed form / DOP853 |
| epidemics (mass_action presets) | `sir_epidemic`, `seir_epidemic` (fractions, sum conserved) | mpmath Taylor / DOP853 |
| circuits and mechanics (linear presets) | `series_rlc`, `spring_mass_damper` | exact matrix exponential (mpmath, 40 digits) / `scipy.linalg.expm` |
| stiff mass action | `stiff_reaction` (A->B k=1e4, 2B->C k=1e3, C->2A k=0.5; timescales 1e-4 to 1e3) | scipy Radau (rtol 1e-13) / scipy LSODA |
| long-horizon oscillators | `harmonic_long` (t=2000, about 414 periods), `coupled_oscillators_long` (t=1000, beating) | exact matrix exponential / closed form or mpmath eigen-decomposition |

Short and long horizons are mixed on purpose (final times from 8.0 to 2000, 11 problems with a horizon of 40 or more). Every reference is at least 100x tighter than the spec tolerance (`reference.max_abs_error_bound`, checked by `check_problems.py` and by a test); every file records its tool versions and the cross-check disagreement. Invariants where they exist: `linear` (population, mass), `quadratic_form` (oscillator energy, Kepler angular momentum) and the new `energy` kind (`model` = `pendulum`, `kepler` or `projectile` with the model parameters, see below). Their tolerances follow the pre-registered rule of the library (linear 10*rtol*max(1,|value|), nonlinear 1000*rtol*max(1,|value|)).

New invariant kind `energy`: fields `model` (`pendulum`: `omega^2/2 + g_over_l (1-cos theta)`; `kepler`: `|v|^2/2 - mu/r`; `projectile`: `|v|^2/2 + g y`), `params` (exactly the model's parameters: `g_over_l`; `mu`; `g`), `value`, `tol`. It is evaluated by `solvers/tools/check_problems.py::eval_invariant`. The Python verifier of PR #274 may not know this kind yet: the calibration task (S-2b) must add it or skip those invariants explicitly.

## Result of the first run (release binary, `bdf`, this host, solver rev as pinned in `Cargo.toml`)

Metric (as in the library): max over all elements and output times of `|y - y_ref| / (rtol*|y_ref| + atol)`; the policy accepts at most 100. Invariant drift is the worst `|invariant - value| / tol`; it must be at most 1.

| problem | steps | error ratio | allowed | invariant drift / tol | verdict |
|---|---|---|---|---|---|
| coupled_oscillators_long | 37773 | 1.1e+04 | 100 | 5.34 (energy) | **OUTSIDE POLICY** |
| harmonic_long | 69204 | 1.08e+04 | 100 | 21.3 (energy) | **OUTSIDE POLICY** |
| kepler_circular | 3481 | 3.7e+07 | 100 | 1.99 (energy) | **OUTSIDE POLICY** |
| kepler_eccentric | 1951 | 1.1e+06 | 100 | 0.0721 (energy) | **OUTSIDE POLICY** |
| kepler_eccentric_long | 40575 | 5.1e+08 | 100 | 1.36 (energy) | **OUTSIDE POLICY** |
| logistic_growth | 387 | 2.95 | 100 | none defined | within policy |
| pendulum_damped | 2096 | 102 | 100 | none defined | **OUTSIDE POLICY** |
| pendulum_large_angle | 2904 | 2.23e+04 | 100 | 0.0837 (energy) | **OUTSIDE POLICY** |
| pendulum_small_angle | 756 | 222 | 100 | 0.00417 (energy) | **OUTSIDE POLICY** |
| projectile_drag | 368 | 59.5 | 100 | none defined | within policy |
| projectile_no_drag | 182 | 6.68 | 100 | 0.000479 (energy) | within policy |
| seir_epidemic | 477 | 7.77 | 100 | 4.21e-06 (population) | within policy |
| series_rlc | 984 | 40.3 | 100 | none defined | within policy |
| sir_epidemic | 456 | 9.15 | 100 | 6.73e-07 (population) | within policy |
| spring_mass_damper | 1848 | 146 | 100 | none defined | **OUTSIDE POLICY** |
| stiff_reaction | 647 | 11.4 | 100 | 1.44e-08 (mass) | within policy |

**9 of 16 problems are outside the stated policy (7 within)**, and the invariant drift exceeds its tolerance on 4 problems. Nothing was loosened. `pendulum_damped` (102) is a near miss; the worst are the three Kepler orbits (ratios 1.1e+06 to 5.1e+08), `pendulum_large_angle`, `harmonic_long` and `coupled_oscillators_long` (ratios of order 1e4), i.e. 2 to 7 orders of magnitude above the limit. This is the same mechanism as the three failures of the original library (global error accumulates over many periods at rtol 1e-8) but much stronger, because the horizons here are tens to hundreds of periods.

## How the error scales when the tolerance is tightened 100x (one extra run each)

`rtol` and `atol` of the spec are both divided by 100; the reference and the error metric are unchanged. The "multiple" column divides the tightened run's error by the SPEC tolerance scale, so values below 100 mean the tightened run would satisfy the original policy.

| problem | max abs error at spec tolerance | at tolerance/100 | improvement (x) | local order p (error ~ tol^p) | error at /100 as multiple of the spec tolerance scale | reference error bound |
|---|---|---|---|---|---|---|
| coupled_oscillators_long | 2.63e-05 | 6.07e-07 | 43.4 | 0.82 | 249 | 1.23e-16 |
| harmonic_long | 0.00014 | 3.41e-06 | 41.2 | 0.81 | 261 | 4.08e-14 |
| kepler_circular | 0.0037 | 9.25e-05 | 40 | 0.80 | 9.25e+05 | 2.4e-16 |
| kepler_eccentric | 0.000131 | 1.77e-06 | 74.1 | 0.93 | 1.37e+04 | 2.4e-16 |
| kepler_eccentric_long | 0.051 | 0.00105 | 48.4 | 0.84 | 1.05e+07 | 2.08e-16 |
| logistic_growth | 2.53e-07 | 8.83e-09 | 28.7 | 0.73 | 0.103 | 1.5e-12 |
| pendulum_damped | 1.23e-07 | 2.93e-09 | 42 | 0.81 | 2.37 | 2.29e-13 |
| pendulum_large_angle | 0.000267 | 5.15e-06 | 51.9 | 0.86 | 430 | 3.16e-16 |
| pendulum_small_angle | 8.36e-08 | 1.86e-09 | 44.8 | 0.83 | 3.58 | 1.88e-17 |
| projectile_drag | 3.84e-06 | 1.04e-07 | 36.7 | 0.78 | 1.68 | 1.99e-12 |
| projectile_no_drag | 6.1e-07 | 1.61e-08 | 37.9 | 0.79 | 0.262 | 2.7e-13 |
| seir_epidemic | 5.23e-09 | 1.33e-10 | 39.5 | 0.80 | 0.195 | 3.69e-14 |
| series_rlc | 7.94e-08 | 1.75e-09 | 45.3 | 0.83 | 0.917 | 8.88e-16 |
| sir_epidemic | 1.93e-08 | 6.48e-10 | 29.7 | 0.74 | 0.315 | 3.15e-14 |
| spring_mass_damper | 1.35e-07 | 3.22e-09 | 41.9 | 0.81 | 3.67 | 4.44e-16 |
| stiff_reaction | 6.85e-09 | 1.76e-10 | 38.9 | 0.80 | 0.294 | 9.56e-14 |

Reading: tightening by 100x reduces the error by only 28.7x to 74.1x (local order p between 0.73 and 0.93), so for these problems the global error is not proportional to the tolerance: to cut it by 100x one must tighten by more than 100x. The reference bounds (1e-12 to 1e-17) are many orders of magnitude below every measured error, so the tightened errors are solver error, not reference floor. Measured: after the 100x tightening the error is within the original policy (multiple <= 100) for 10 problems (logistic_growth, pendulum_damped, pendulum_small_angle, projectile_drag, projectile_no_drag, seir_epidemic, series_rlc, sir_epidemic, spring_mass_damper, stiff_reaction) and still outside it for 6 (coupled_oscillators_long, harmonic_long, kepler_circular, kepler_eccentric, kepler_eccentric_long, pendulum_large_angle). The calibration task (S-2b) should decide the acceptance rule from this data; this task does not change it.

## Reproduce

```
python3 solvers/tools/freeze_heldout.py --verify
python3 solvers/tools/run_problems.py --bin <path-to-hivemind-solver> --dir solvers/problems/heldout --tighten 100 --markdown
```

Raw numbers: `solvers/results/heldout_run_1.json`. One host, one run; ten-run determinism of the binary is covered by the library tests. Physical-model validity is never checked by the solver or the verifier: a spec for a pendulum that is really a different system still gets a numerically certified answer to the equations it states.
