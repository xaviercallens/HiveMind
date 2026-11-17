# Solver Bee verifier

`src/hivemind/solverbee/verifier.py` (stdlib only) turns a solver result into a `hivemind-solver-cert/1` certificate. It is independent of the solver: it only reads the spec bytes, the result, an optional problem-library entry and an optional `rerun` callable.

```python
cert = verify(spec_bytes, result, problem=None, rerun=None, binary_sha256=None)
```

`verify` never raises. Any exception inside a check becomes `status: "fail"` with the message. Results over 8 MiB, a missing or non-dict result, `status != "ok"`, a wrong schema or an unparseable spec give `UNVERIFIED`.

## Checks: what they mean and what they do not mean

| Check | Passes when | Does NOT mean |
|---|---|---|
| `shape` | schema, id equals the spec id, status ok, all numbers finite (booleans rejected), `len(t) == len(y) == len(t_out)`, each row has n entries, `t == t_out` exactly | the numbers are right |
| `hashes` | `result.input_sha256` equals sha256 of the exact spec bytes; `result_sha256` is computed over canonical JSON (sorted keys, compact separators, `ensure_ascii=False`) | the result was produced by the pinned binary (that is `binary_sha256`, supplied by the caller and only recorded) |
| `reference` | with a library entry whose `problem`, `y0`, `t0` equal the spec's: at every reference time that coincides with a `t_out` time, `\|y - y_ref\| <= factor*(rtol*\|y_ref\| + atol)`, factor from `tolerance_policy.max_rel_err_factor` (default 100). `value` is the worst ratio, `bound` is 1.0 | accuracy at times with no reference point; if no reference time coincides the check FAILS (it cannot be done) |
| `invariants` | each invariant in the library entry drifts at most `tol` (absolute) at every output time. Skipped, with detail "no invariants defined", when there are none | the trajectory between output times conserves anything |
| `tightening` | the verifier builds the spec with `rtol/100` and `atol/100` (`rtol` and `atol` are optional in a spec; when absent the verifier uses the CLI defaults 1e-8 and 1e-10, so a spec that omits them is audited like any other), calls `rerun`, validates that result like any other, and requires `\|y_loose - y_tight\| <= 100*(rtol*\|y_tight\| + atol)` using the LOOSE tolerances. Skipped when `rerun` is None | that both runs are right: a solver with a systematic bias that does not depend on the tolerance passes this check |

Why 100: the tight run is about 100x more accurate, so the difference is essentially the loose run's own error. Solvers control local error, not global error, so the allowed band is 100 times the requested tolerance, matching the library's default `max_rel_err_factor`. This is a generous band: S-2 found one injected "loosened tolerance" result (about 29 times the claimed tolerance) that lies inside it. A stricter factor would be a new, separately pre-registered experiment.

## Verdicts

The decision is one function, `decide(checks, usable, has_problem)`:

1. result not usable -> `UNVERIFIED` (fail closed)
2. any check failed -> `FAILED`
3. shape, hashes, tightening pass, and reference passes when a problem is given, and invariants pass (or none are defined) -> `VERIFIED` with a problem, `AUDITED` without one
4. otherwise (for example no `rerun` was given) -> `UNVERIFIED`

Note that `AUDITED` and `VERIFIED` both need the tightening check: calling `verify` without `rerun` yields `UNVERIFIED`.

## The caveat

Every certificate carries, verbatim: "VERIFIED means the listed checks passed. In the E16 measurement 5-7% of VERIFIED code answers still failed hidden tests; it never means 'correct'." Consumers must show it next to the verdict.

## Invariant kinds

| kind | fields | quantity compared to `value` |
|---|---|---|
| `linear` | `weights` | `w . x` |
| `quadratic_form` | `matrix` (alias `Q`) | `x^T M x` |
| `energy` | `matrix` (alias `Q`) | `0.5 x^T M x` |
| `lv_first_integral` | canonical: `params: {alpha, beta, gamma, delta}`; aliases (same four names at the invariant's top level, then `spec.problem.params`); `value` defaults to the value at `y0` | `delta x - gamma ln x + beta y - alpha ln y` for `x' = alpha x - beta x y`, `y' = delta x y - gamma y` |

The canonical layout is the one the problem library (`solvers/problems`) writes: `linear {weights, value, tol}`, `quadratic_form {matrix, value, tol}`, `lv_first_integral {params:{alpha,beta,delta,gamma}, value, tol}`. `tol` is ABSOLUTE for every kind (the check is `|invariant(x(t)) - value| / tol <= 1`). The names `Q` (for `matrix`) and top-level `alpha/beta/gamma/delta` (for `params`) are accepted only as aliases for older callers; when both `params` and a top-level name are present, `params` wins. A malformed or missing field fails the check closed. No tolerance, factor or threshold was changed by this reconciliation.

To add a kind: write `_inv_<kind>(inv, spec, x) -> float` returning the conserved quantity for one state vector `x`, raise `_CheckError` on malformed fields, register it in `_INVARIANT_KINDS`, and add a pass and a fail case to `tests/test_solverbee_verifier.py`.

## Scoring

`scoring.py` computes per-bee verdict counts, the Wilson 95% lower bound of the VERIFIED rate and `recruitment_weight`: that bound, or 0.0 with a stated reason when the bee has fewer than 5 verdicts (cold start).

## Experiment S-2

`artifacts/experiments/S2_fault_injection_prereg.json` (and amendment 1) fix the thresholds; `scripts/run_experiment_S2_fault_injection.py` measures; the report is `S2_fault_injection_report.json`.
