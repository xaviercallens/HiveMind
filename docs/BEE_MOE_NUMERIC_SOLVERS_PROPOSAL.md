# Bee MoE with numeric solver experts: a three-wave proposal

Status: **Wave 1 APPROVED by the owner on 2026-10-27; implementation in progress (tracking epic: issue #273). Waves 2 and 3 remain proposals.** The pieces are being built as separate pull requests; this document is the plan. Effort figures are estimates, not measurements. Every capability is labelled *verified here*, *planned*, or *owner decision*. IDs such as `W1-1` are proposal-local; roadmap rows are added only after the owner picks a season number.

## 1. The idea in one paragraph

A Bee MoE today routes between *neural* experts (the night MoE adapters, the coder tiers). Some questions are not best answered by a neural network at all: "integrate this stiff chemical network to t = 100 and tell me the final concentrations" has a right answer that a deterministic solver computes and a checker can confirm. The proposal adds a **Solver Bee**: a non-neural expert that runs a pinned numeric solver in the existing fail-closed sandbox and returns its result **together with a certificate** that an independent verifier re-checks. Neural bees *propose and formulate*; the solver and the verifier *decide*. This is the same pattern that produced a measured gain in our own experiments (E15: the verification loop added +7.0 points of pass@1; E16: VERIFIED precision 92.6 to 94.8 percent across the three tiers, so about 5 to 7 percent of VERIFIED answers still fail hidden tests, which is why the vocabulary stays strict).

## 2. What already exists (inventory)

Sources: a read-only survey of this machine and of the repository on 2026-10-27. README figures are **self-reported** and were not re-run.

| Asset | What it is | Licence | Evidence of maturity | Use in Wave 1 |
|---|---|---|---|---|
| `rusty-SUNDIALS` (owner's project, in the home directory) | Pure-Rust ODE solver translated from LLNL SUNDIALS CVODE (BDF and Adams) | BSD-3-Clause with a NOTICE file | README badge: 134 passing tests (self-reported) | Core solver, as a **pinned external dependency** |
| `fenris-cmrr` | Relaxes 2D mesh nodes under a Yukawa potential using rusty-SUNDIALS (BDF); small C API | MIT | 6 tests (README) | Working example of calling the solver; not shipped |
| `AutoevolveAI` | Verification-first harness for agents that claim computational results | MIT | README: 1371 tests (self-reported) | Design reference for the verifier; not a dependency |
| `lattice_quantizer.py` (in this repository) | Exact nearest-point decoders for Z^n, A2, Dn, E8 with an independent brute-force checker | AGPL-3.0 | E10 and E11 artifacts | Optional first slice; zero new dependencies |
| E15 gate (`scripts/e15_gate.py`) | Fail-closed namespace sandbox for running untrusted code | AGPL-3.0 | E15 and E16 artifacts | Runs the solver binary (needs one small extension) |
| `queenbee/dance.py` (Wilson-interval recruitment) and `verification/bee_energy.py` | Scoring of a bee's track record; graded energy of a result | AGPL-3.0 | Module docstrings; test status not re-checked | Bee scoring |
| `swarm/verified_update_gate.py` | Paired exact test for accepting a candidate over a baseline | AGPL-3.0 | Tested per module | Acceptance rule for experiments |
| `scripts/night_pack_build.py` | Pattern for CPU data packs with licence text read from the card and sha256 manifests | AGPL-3.0 | Night-MoE plan | Pattern for problem packs |

Datasets and projects found but **not used in Wave 1** (licence or provenance not established): NANOGrav 15-year tarball, `oc20_is2res_val_id`, `~/hf-data` subdirectories, bio datasets, the Hugging Face cache (FineWeb-Edu and Wikipedia carry their own terms), `SocrateAI-Scientific-Genetic-pr` (no licence file found), LeanMaster (Apache-2.0 per a README badge, no LICENSE file found).

## 3. Gaps (what does not exist)

1. No ODE, PDE, FFT, N-body or CFD solver in the repository. The "astrophysics discoveries" in `astrophysics_24h_campaign.py` are hard-coded and labelled illustrative; they are **not** measured science and are not used here.
2. The E15 sandbox is stdlib-only (`python3 -I -S`): no numpy, no scipy. A Rust binary sidesteps this; a Python solver would not run.
3. Every registered BOINC app is CUDA training or inference. A CPU numeric app needs a new plan class, a new validator branch (`boinc_server_scripts.validate` dispatches on the training apps and `hivemind_moe_infer` only; any other app returns "unknown app") and a CPU branch in the server stack (`write_plan_class_spec` and `install_app` in `src/hivemind/boinc/real_client/server_stack.py` assume NVIDIA and CUDA).
4. Spot-check validator assignment is architecture-matched by design (HM-S10-08): cross-accelerator-class replays are 'not comparable' and never flag an account. **The cross-CPU tolerance for a solver is unknown.**
5. The router catalog has no CPU tier (`min_vram_gb` is a required field and must be positive).
6. No MPI runtime is installed on this machine and no project uses one. OpenMP and rayon usage was not surveyed.

## 4. Wave 1: "Solver Bee v0" (detailed, feasible on this machine)

### 4.1 Scope

One expert type, one problem class, CPU only, no GPU, no external datasets, no wide-area network. **Initial-value problems for ordinary differential equations**, specified as data, never as code:

- named families with parameters (Robertson, Van der Pol with parameter mu, Brusselator, Lotka-Volterra, damped harmonic oscillator, a small stiff chain);
- linear systems `x' = A x + b` (reference: matrix exponential);
- mass-action reaction networks given as reaction lists (invariants from the stoichiometry).

Because the specification is data, **no user-supplied code ever runs**. This is a safety property as much as a scope limit.

### 4.2 Deliverables

| ID | Deliverable | Acceptance criterion |
|---|---|---|
| W1-1 | `hivemind-solver`: a small Rust command line built on rusty-SUNDIALS at a pinned revision. JSON spec in; JSON out with the trajectory at requested times, solver statistics, the input sha256 and the binary sha256. Built with `--locked`; no fast-math; pure-Rust math (the `libm` crate) rather than the system libm so results do not depend on the glibc version. | Builds from a clean checkout; the pinned solver's own tests and ours pass; the output is byte-identical across ten repeated runs on this host. |
| W1-2 | Problem library: at least 12 problems with references computed offline (high-precision, with the script and its provenance stored) and with invariants where they exist (Robertson mass sum, Lotka-Volterra first integral, oscillator energy, stoichiometric conservation). | Every reference is reproducible from the stored script; no external dataset is used, so no data licence is involved. |
| W1-3 | Independent verifier (Python, stdlib only). Four checks: error against the reference, invariant drift, agreement when the tolerance is tightened by a fixed factor (two runs), and hash consistency. Verdicts reuse the GWAYA vocabulary: VERIFIED, AUDITED (no reference, checks only), FAILED, UNVERIFIED (could not run, fail closed). | Clean set: 0 false rejections. |
| W1-4 | Sandbox extension: run a **binary** under the E15 limits (CPU time, address space, process count, file size, no network, read-only inputs). | Tests show the limits trigger; the existing E15 tests still pass. The known caveats (no seccomp, shared kernel, no cgroup memory limit) stay documented. |
| W1-5 | `solver_bee` service: loopback only, OpenAI-compatible `/v1/chat/completions` where the user message carries a fenced spec block; reply carries the result and the certificate; behind the producer shield. | 20 scripted requests answered correctly; malformed specs rejected with a clear error; no spec can execute code. |
| W1-6 | Pre-registered experiment **S-1: reproducibility.** (a) same binary, same host: bitwise identical; (b) same binary, different CPU class: report the largest relative difference over all problems. The thresholds are registered before the first run. | (a) must hold. For (b) the **measured value is the deliverable**, whichever way it falls. Part (b) needs at least one more CPU class; this machine has one. Part (b) needs another CPU class; the epic issue #273 asks for volunteers (issue #251 asks for different GPUs and runtimes). |
| W1-7 | Pre-registered experiment **S-2: does the verifier catch wrong answers?** Inject five fault types (wrong matrix entry, perturbed trajectory, loosened tolerance, NaN, truncated output) into the clean set. | 100 percent of injected faults flagged; 0 of the clean results rejected. Positive and negative controls follow the E-series method. |
| W1-8 | Bee scoring: log each verdict and compute the Wilson lower bound per bee with the existing module. | A short report on the 12-problem set. |
| W1-9 | Documentation and quickstart for newcomers: what a Solver Bee is, how to run it, how to add a problem. | A newcomer can reproduce W1-2 and W1-3 from the README. |

**Stretch (W1b), only if a GPU slot is approved by the owner:** a small coder model (the existing GWAYA-Lite tiers) maps plain-language problem statements to the JSON spec; pre-registered on 40 held-out prompts, exact match of the parsed spec against a template baseline, paired exact test. Not part of the core acceptance.

### 4.3 What Wave 1 does **not** claim

- It does not deliver massive parallelism or HPC. It delivers the verified building block that Wave 2 parallelises.
- It does not show that a Solver Bee beats a neural expert on any benchmark. It shows that its answers can be certified.
- It does not run on the GPU, and it does not touch the live node.

### 4.4 Capacity plan

The machine has 8 vCPUs shared with another session's long Lean jobs. Wave 1 runs a single worker, one core per solve, and starts only when load is low. Nothing here needs the T4.

### 4.5 Effort (estimate, unvalidated)

About 4 to 6 weeks part-time for one engineer; agent-assisted with review, perhaps 2 to 3 weeks. S-1 part (b) depends on getting a second CPU class and may take longer.

### 4.6 Risks

| Risk | Mitigation |
|---|---|
| Floating-point results differ across CPUs or libm versions | `libm` crate, no fast-math, no `target-cpu=native`; S-1 measures it instead of assuming |
| The solver is a translation of LLNL code and its licence notice may carry obligations | Treat it as an external pinned dependency, ship its LICENSE and NOTICE unchanged, and get the owner's licence decision before any vendoring into the AGPL tree |
| Verifier and solver share a bug | The reference solutions come from an independent high-precision computation, and S-2 injects faults |
| "VERIFIED" read as "correct" | Keep the E16 caveat in every output: VERIFIED means the stated checks passed |

### 4.7 Stage 2 of Wave 1 (after the three base pull requests): the first easy scientific bee

All items are **planned**.

- (a) Integration of the real solver, verifier and library end to end.
- (b) Science-friendly families for students and citizen scientists: nonlinear pendulum, projectile with quadratic drag, Kepler two-body, SIR and SEIR epidemics (both can be expressed in the `mass_action` kind), logistic growth, RLC circuit (`linear` kind).
- (c) An MCP server exposing `solve`, `list_problems` and `verify`, so any MCP client can call the Solver Bee. The client's own language model formulates the spec; the tool validates it and certifies the answer.
- (d) A Science Bee conversational layer: a natural-language question becomes a validated spec through a small model with strict schema validation and a bounded retry loop, then solver, verifier, and a plain-language answer with units, the checks that passed and what they do not prove.
- (e) Pre-registered experiment **S-3: formulation accuracy** on at least 40 held-out questions written by the project (answers known): exact match of the parsed spec and numerical agreement with the reference, against a template baseline, paired exact test. CPU first, then the resident coder model on the T4 only inside the night window at low request rate.
- (f) S-1 run on this machine's CPU; the cross-CPU part needs a second CPU class (the epic issue #273 asks for volunteers).
- (g) A newcomer quickstart and a short guide for students.

The model's role is only to propose a specification. The trust in an answer comes from the solver and the verifier. Whether the chosen physical model is valid for the real situation is **not checked**.

## 5. Wave 2 (planned; depends on the Wave 1 results)

- **BOINC CPU sweep app.** New CPU plan class, a validator branch for solver results, class-aware tolerances taken from S-1 (the validator forbids cross-class replay today), quorum of two for sweeps. Parameter sweeps (a Van der Pol mu grid, reaction-rate scans) are embarrassingly parallel and are the realistic route to volunteer-scale computation.
- **Surrogate experts.** Small neural surrogates trained on solver outputs, with the **solver as teacher and verifier**: the surrogate proposes, the solver checks, the track record decides routing. This is the "trained plus numeric" integration.
- **Mesh integration.** A `solver` tier in the AMCP bee mesh, using the existing task and handoff events.
- **Router.** A CPU tier in the catalog (an owner-reviewed router change).
- **Portability.** Windows and macOS builds of the Rust command line.
- **Datasets.** Only after a licence review of each candidate listed in section 2.

## 6. Wave 3 (planned; needs resources that do not exist here)

- **Tightly coupled HPC (MPI).** Needs an MPI runtime and a cluster or an allocation. Nothing on this machine or in the repository uses MPI; this is stated so it is not mistaken for something Wave 1 delivers.
- **PDE solvers** by the method of lines, and the cosmology and vacuum-decay simulations in `SocrateAI-Scientific-DualScaleSimulator`, after a review of that project (no README, licence only in `pyproject`).
- **Exact cross-hardware replay.** Fixed-point or interval arithmetic for the E12 open item (the replay kernel that does not exist yet).
- **Formal link.** A Lean statement of the verifier's soundness, building on the existing Lean work.
- **GPU numerics** for dense linear algebra.

## 7. Decisions (answered 2026-10-27)

1. Wave 1 is approved.
2. rusty-SUNDIALS is confirmed as the solver (same GitHub owner). It stays a pinned external dependency and is never vendored.
3. A CPU tier is allowed in the router catalog in addition to the GPU tiers; work moves to the GPU where that makes sense.
4. Solver Bee answers are priced in HMT but free for the coming month, until 2026-11-27 (points are non-monetary accounting units).
5. GPU time for the natural-language step is limited to the Paris night window (22:00 to 07:00, Europe/Paris) and goes through the existing serving path, because the live node serves users first and its governor policy is fixed.

## 8. Proposed issue titles

Created as the epic issue #273 with a checklist instead of one issue per title.

1. `hivemind-solver`: pinned Rust CLI around rusty-SUNDIALS (JSON in, certificate out)
2. Problem library with independent references and invariants (12 IVPs)
3. Independent verifier and the VERIFIED / AUDITED / FAILED / UNVERIFIED vocabulary for numeric results
4. E15 sandbox: run a binary under resource limits
5. `solver_bee` loopback service with an OpenAI-compatible endpoint
6. Pre-registered S-1: cross-CPU reproducibility of solver results (needs a second CPU class)
7. Pre-registered S-2: fault injection against the verifier
8. Bee scoring for solver bees (Wilson lower bound)
9. Newcomer quickstart for the Solver Bee
10. (Wave 2) BOINC CPU sweep app and a solver-result validator
11. (Wave 2) Surrogate expert with the solver as teacher and verifier
12. (Wave 2) CPU tier in the router catalog

## 9. Honest limits of this proposal

The inventory is read-only and partial: some project folders were not opened, and README test counts are self-reported. No solver was run for this document. The licence statements are what the files say, not legal advice. The effort estimates are guesses informed by the size of the earlier work in this repository.
