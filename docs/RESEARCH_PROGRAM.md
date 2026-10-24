# Project HiveMind — Research Program (scientific lead: Claude; engineering coordinator: Antigravity)

Version 1.0, 2026-10-23. This document is the scientific contract of the project. It states the
thesis, the research questions, the predictions each question makes, what observation would
refute each prediction, and the pre-registered experiments that decide them. Engineering tasks
(`HM-S9-xx` in `ROADMAP.md`) exist to produce the measurements named here; a task is finished
when its artifact answers the question it was opened for, not when code merges.

Operating rule carried over from the 2026-10-23 review (`MOE_DILOCO_REVIEW_2026-10-23.md`):
no number enters the paper without an artifact produced by a committed script, every experiment
runs a positive and a negative control, no script emits a score or verdict labelled as a
reviewer's, and every arXiv identifier is opened before it is cited.

## 1. Thesis

Volunteer consumer GPUs (8 GB, residential WAN, churn, untrusted operators) can train a sparse
Mixture-of-Experts whose held-out loss, at matched tokens, lies within a **predictable** distance
of a single-host DiLoCo run with the same number of replicas. "Predictable" means: the gap is the
sum of (a) the replica-count penalty already described by the DiLoCo scaling law of Charles et
al. (`arXiv:2503.09799`) and (b) a **volunteer tax** caused by churn, staleness and Byzantine
filtering, and the mechanisms integrated in HiveMind (Decoupled quorum + grace window + RDA,
PA-CGAD, FoMoE partial replication, spot-check verification) keep (b) bounded by the penalty of
simply having fewer, reliable replicas.

If (b) is *not* bounded that way, the system design is wrong and the paper must say so. Every
question below is a piece of that thesis.

## 2. Research questions and predictions ledger

| RQ | Question | Prediction (falsifiable) | Refuted if | Decides | Experiment |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **RQ1** Volunteer tax | How far is the held-out loss of M real WAN learners with churn from M synchronous single-host workers at matched tokens? | **P1**: with `K < M` quorum, grace window and RDA, the WAN loss is no worse than the single-host loss obtained with `M·(1−churn)` workers. | WAN loss exceeds the "fewer workers" baseline by more than the 95 % seed CI. | whether the quorum design pays for itself | E8 (needs E5 first) |
| **RQ2** RDA and outer-LR retuning | Does Radial-Directional Averaging remove the need to retune the outer learning rate with M? | **P2**: under direct averaging the best `η_outer` grows with M (≈√M when learner deltas are near-orthogonal; report the measured mean pairwise cosine); under RDA the best `η_outer` is the same grid point for M ∈ {4, 8, 16}. | best `η_outer` under RDA moves by ≥ 1 grid step between M = 4 and M = 16, or deltas are not near-orthogonal (then the prediction does not apply and must be re-derived). | RDA as default merge | E5 |
| **RQ2′** RDA as an outer-LR multiplier | Is RDA, in training, nothing but direct averaging with a per-round step multiplier `r_t = Σ w_i‖Δ_i‖ / ‖Σ w_i Δ_i‖ ≥ 1` (Proposition `prop:rda_multiplier` in the paper)? | **P9**: at the heterogeneity level where P2 applies, `log(best η_direct / best η_RDA)` lies within one √2 grid step of `log(mean r_t)` for every M. | the deviation exceeds one grid step at some M (then RDA changes the direction, not only the length, and RQ2 must be re-stated). | whether the campaign may replace RDA by a `√M`-style LR rule | E5-het |
| **RQ9** Lattice codec for outer deltas (added 2026-10-23, leverages OptimalEnergyAI) | At matched measured rate, does E₈ blockwise quantisation of learner deltas beat scalar ℤ⁸ quantisation, and does error feedback recover the unquantised loss at 1 bit/dim? | **P10a**: relMSE(E₈)/relMSE(ℤ⁸) < 1 at 1 and 2 bits/dim (Gaussian prediction 0.86); **P10b**: loss(ℤ⁸) − loss(E₈) > 0 at 1 bit/dim with EF; **P10c**: loss(E₈, 1 bit, EF) − loss(none) ≈ 0. | P10a: ratio CI includes or exceeds 1; P10b: CI < 0; P10c: CI > 0. | default outer-delta codec of the campaign (`HM-S10-01`) | E10-lattice |
| **RQ10** KV-cache codec for the inference ring (added 2026-10-23; leverages the RunuX KV study 10.5281/zenodo.22879667 and OptimalEnergyAI H1.6) | At equal bytes, does a lattice codec (A₂ per RoPE plane, or E₈ after a per-head transform) beat the FP8 paged KV cache on real `Qwen2.5-0.5B` keys/values? | **P11**: A₂ per RoPE plane lowers ΔNLL vs FP8-equivalent scalar quantisation at matched bits; E₈ does so only with the transform. | ΔNLL CI includes 0 at matched bytes, or FP8 wins. | page codec of `FP8PagedKVCache` (`HM-S10-05`) | E11 (prereg first) |
| **RQ11** Spot-check replay across hardware (added 2026-10-23 from the external review of v3.5.0) | How large is `‖θ_validator − θ_k‖∞` after replaying one 50-step segment across accelerator classes (T4, RTX 2070, TPU-VM CPU), and does fixed-point replay remove it? | **P12**: same-class divergence ≤ 1e-4 (the current ε); cross-class divergence exceeds it; fixed-point replay is exact. | same-class divergence > 1e-4 (then ε itself is wrong), or cross-class ≤ 1e-4 (then matching is unnecessary). | validator assignment rule and ε of the Byzantine pillar (`HM-S10-08`) | E12 (prereg first) |
| **RQ3** Grace window vs matched random | Is adaptive grace admission better than admitting the same expected number of extra learners at random? (lesson of WA-DiLoCo, `arXiv:2607.02544`: forecast-free policies must beat matched random) | **P3**: at equal loss, adaptive grace reaches it in fewer wall-clock seconds than matched-random admission with the same admission budget. | matched random ties or wins (then the grace window is only a budget knob and the EMA controller should be deleted). | keep or delete `AdaptiveGraceWindowController` | E5 |
| **RQ4** Staleness gating under quorum | Does PA-CGAD still help once a quorum already excludes late learners? | **P4**: CGAD's terminal-MSE gain over un-gated outer AdamW is large without quorum (E2 reproduced) and shrinks with quorum `K < M`; it never hurts. | CGAD hurts under quorum, or its gain does not shrink. | whether to ship CGAD + quorum together | E5 (factorial) |
| **RQ5** Ghost fraction and quality | Is Skip-Token quality loss governed by the *measured* ghost-assignment fraction rather than the ideal `N_le/N_e`? | **P5**: at a fixed ideal factor, perplexity degrades monotonically with the measured ghost fraction across routing-imbalance conditions. | no monotone relation, or the ideal factor predicts as well as the measured one. | whether to log and gate on the measured fraction (`HM-S9-02`) | E7 |
| **RQ6** Replica scaling and inner optimizer | Does a whitening inner optimizer flatten the loss-vs-M slope (claim of `arXiv:2610.19064`, `arXiv:2505.23725`)? | **P6**: slope of held-out loss vs log M with Muon Newton–Schulz inner preconditioning is smaller than with AdamW inner, at matched compute. | slopes are equal within CI, or Muon is worse at M = 4. | default inner optimizer for the campaign | E6 |
| **RQ7** Looped depth prior | Does a broad, learned depth prior make terminal-KV sharing safe where fixed-depth training does not (Part II, `arXiv:2610.06833`)? | **P7**: accuracy drop from full cache to terminal-KV sharing is smaller with the learned prior than with fixed depth, at equal mean recurrence. | drops are equal within CI. | whether `HM-S7-02` KV sharing is enabled by default | E7b |
| **RQ8** Adaptive Byzantine attacks | Does Multi-Krum + 2σ cosine filtering stop an ALIE-style small-perturbation attack under non-IID shards? | **P8**: it does not for `f/n ≥ 0.2`; the deterministic spot-check replay is what catches it. The paper currently claims only sign-flip/scale resistance and must not claim more until measured. | the filter alone blocks ALIE at `f/n = 0.3` (then the spot-check cost can be cut). | security section of the paper | E9 |

Status column convention (kept in `artifacts/predictions_ledger.json` once E5 starts): `untested`,
`supported`, `refuted`, `inconclusive (CI includes 0)`. A refuted prediction is reported in the
paper as refuted; it is not silently removed.

## 3. Scaling-law prior (how "predictable" is computed)

Before E5, fit the functional form used by Charles et al. (`arXiv:2503.09799`) for the loss as a
function of model size, tokens, replica count M and inner steps H to HiveMind's own single-host
runs at M ∈ {1, 2, 4, 8, 16} on the `Qwen2.5-0.5B` LoRA setting (E1 model, ≥ 3 seeds). We do not
reuse their coefficients, which were fitted on different models and data; we reuse the form. The
fitted curve is the "replica-count penalty" baseline. The volunteer tax of RQ1 is then the residual
between the WAN run and that curve. Report the fit, its residual standard error and the number of
points; if the form does not fit single-host runs (R² < 0.9), say so and fall back to the
"fewer workers" baseline only.

## 4. Pre-registered experiments

Common protocol for every experiment:

(The protocol is the one of *A laboratory method for AI-assisted science* (10.5281/zenodo.22829767): formal proof, exact arithmetic, controlled experiment and a claim ledger; the hyper-parameter search at scale uses the autoresearch screening loop of 10.5281/zenodo.23467992; the harness shape, preregister → run → ledger gate → Lean gate, is OptimalEnergyAI's.)

- **Seeds**: ≥ 3 seeds per cell; report mean and 95 % t-interval; claim an effect only if the
  interval excludes zero. E1–E4 in paper v3.2.0 use a single seed (2026) and are therefore
  demonstrations, not evidence of an effect; E5 re-runs E1's regimes with 3 seeds as its first cell.
- **Positive control**: a cell where the known effect must reproduce (e.g. un-gated AdamW under
  stale bursts must diverge as in E2). If it does not, the run is invalid, not "surprising".
- **Negative control**: a cell where the mechanism is switched off or the signal is destroyed
  (e.g. RDA with identical learner deltas must equal direct averaging; grace window with
  `γ = 0` must equal pure quorum). If the negative control shows the effect, the measurement is
  an artefact.
- **Matched-random control** for every *policy* (grace window, cadence controller, Mix2 block
  selection): same budget, random decisions.
- **Pre-registration**: the metric, the cells, the seeds and the acceptance rule are written in
  `artifacts/experiments/E<n>_prereg.json` before the run and hashed into the result artifact.
- **Artifacts**: `scripts/run_experiment_E<n>.py` → `artifacts/experiments/E<n>_report.json`
  with `receipt_sha256`; figures regenerated from the JSON; the paper's provenance lock
  (`PROV-*`) extended for every number quoted.

### E5 — Outer-loop factorial under simulated churn (single host; decides RQ2, RQ3, RQ4)

- Model/data: E1 synthetic LM for the full factorial (cheap), then the `Qwen2.5-0.5B` LoRA
  8-domain setting for the two best and two worst cells.
- Factors: merge ∈ {direct, RDA} × quorum ∈ {K = M, K = ⌈M/2⌉} × grace ∈ {off, adaptive,
  matched-random} × staleness gate ∈ {outer AdamW, PA-CGAD} × M ∈ {4, 8, 16}.
- Churn model: learner arrival times drawn from the empirical RTT and dropout distribution
  recorded by `wan_fault_benchmark.py` (not from a Gaussian), replayed identically across cells.
- Outer-LR sweep per cell: `η_outer ∈ {0.3, 0.5, 0.7, 0.9, 1.1}` (RQ2 needs the argmax).
- Metrics: held-out loss at matched tokens; wall-clock to reach the direct/K=M/no-grace loss;
  admitted-token fraction; mean pairwise cosine of learner deltas (reported, conditions P2).
- Controls: positive = E2 divergence reproduces in the {no quorum, AdamW} cell; negative =
  identical-delta learners give RDA ≡ direct; matched random for grace.
- Acceptance: P2, P3, P4 each get `supported/refuted/inconclusive` with the rule in §2.

### E5-het — the same factorial on a heterogeneity-controlled testbed (added 2026-10-23)

E5-toy returned P2 *inapplicable* (delta cosine 0.909) and every cell within the seed interval,
so a knob was added that makes learner deltas disagree by a controlled amount: learner `m` is
labelled by its own teacher `B_shared + ρ·B_m` and the held-out set mixes all teachers equally
(`scripts/run_experiment_E5_het.py`; `ρ = 0` reproduces E5-toy bit for bit, which is one of its
negative controls). The `ρ` grid was chosen from a calibration of round-0 cosines only
(`artifacts/experiments/E5_het_calibration.json`); the outer-LR grid is geometric with ratio √2 so
that the √2 shift predicted between M = 4 and M = 8 is one grid step. Pre-registration:
`artifacts/experiments/E5_het_prereg.json` (manipulation check, P2, P3, P4, P9, three controls).
This is still a toy: heterogeneity is a synthetic teacher perturbation, not domain shift in text.

### E10-lattice — E₈ vs ℤ⁸ vs sign quantisation of outer deltas (added 2026-10-23; decides RQ9)

Reuses the E5-het ρ = 4, M = 8 testbed (direct merge, admission all, ungated, best direct LR read from the
E5-het artifact). Arms: none, sign (1 bit), ℤ⁸ and E₈ at measured rates 1 and 2 bits/dim, each with and
without EF21 error feedback; scales calibrated by bisection on one round-0 delta. Controls: positive = 1-bit
sign without EF must hurt; negative = E₈ at ≥ 6 bits/dim must equal "none", the decoders must pass the
two-sided brute-force check, and the ℤ⁸/E₈ arms must realise the same rate within 0.1 bit/dim.
Pre-registration: `artifacts/experiments/E10_lattice_prereg.json`; code:
`src/hivemind/compress/lattice_quantizer.py` (decoders ported from OptimalEnergyAI with attribution).

### E11 — Lattice KV-cache codecs for the inference ring (added 2026-10-23; decides RQ10; pre-registered in `artifacts/experiments/E11_kv_lattice_prereg.json` + amendment 1, run 2026-10-23/24 — see the status log)

Real `Qwen2.5-0.5B` KV tensors captured with OptimalEnergyAI's `poc2_crystallized_attention/extract.py`
(`capture_kv`, `rope_planes`), WikiText ΔNLL as the metric. Factors: codec ∈ {FP8 per-page absmax (current
`FP8PagedKVCache`), scalar at matched bits, A₂ per RoPE plane, E₈} × transform ∈ {none, per-head KLT} × bits
∈ {2, 3, 4} × 3 seeds (sequence samples). Controls: positive = 2-bit scalar must raise ΔNLL; negative = 8-bit
A₂ must equal FP8 within CI; rate is *measured* (factorised entropy) as in E10, and a rate mismatch > 0.1
bit/dim invalidates the cell. Not run yet: nothing about E11 may appear in the paper as a result.

E8 amendment (from the external review of v3.5.0): the WAN factorial must include an *all-mechanisms-on* cell
against *each-mechanism-off* cells (NF4, 2-bit DCT+EF21, Skip-Token, Skip-Attention, PA-CGAD, fixed-point
exits) so that interactions between approximations, not only main effects, are detectable.

### E12 — Spot-check replay divergence across hardware (added 2026-10-23; decides RQ11; pre-registered in `artifacts/experiments/E12_replay_divergence_prereg.json`, run 2026-10-24 on the T4 and this host's CPU only — see the status log)

One honest worker's 50-step segment (Qwen2.5-0.5B LoRA, the PoC configuration) replayed on the T4 of this
workstation, the RTX 2070 of the PoC and the CPU of the TPU VM; metric `‖Δθ‖∞` per pair, 3 segments × 3
seeds; cells: same class, cross class, cross class with a fixed-point replay of the attention kernel.
Controls: positive = a tampered replay (one flipped sign) must exceed ε; negative = replay on the same device
must give 0.0 exactly. Nothing about E12 may be stated as a result before it runs.

### E6 — Replica scaling vs inner optimizer (decides RQ6)

- M ∈ {2, 4, 8, 16}, inner ∈ {AdamW, Muon-NS5}, H = 32 and 128, 3 seeds, matched total tokens.
- Metric: slope of held-out loss vs log₂ M by least squares; report both slopes with CI.
- Controls: M = 1 both inner optimizers (positive: Muon ≥ AdamW at M = 1 is *not* required, the
  claim is about slope); negative: shuffled-label run must show no inner-optimizer effect.

### E7 — Ghost fraction sweep (decides RQ5) and E7b looped depth prior (decides RQ7)

- E7: 128 experts, 32 resident, top-2; induce routing imbalance by temperature on the router
  logits so that the measured ghost fraction spans ≈ 0.5–0.9 at a fixed ideal factor 0.25;
  3 seeds; metric: held-out perplexity vs measured ghost fraction (Spearman ρ with CI).
- E7b: `LoopedExpertBackbone` T = 6 trained with fixed depth vs learned categorical prior
  (entropy-regularised, budget-penalised, Part II); evaluate with full cache and with terminal-KV
  sharing; metric: accuracy drop; 3 seeds.

### E8 — Multi-host residential WAN GameDay (decides RQ1; the measurement everything waits on)

- 3 machines first, then 8; real tokens from the 8-domain mixture; `K = ⌈M/2⌉`, adaptive grace,
  RDA, PA-CGAD, FoMoE `O_e = 4` for the 8-node run.
- Baselines on the same hardware: single-host synchronous DiLoCo with M workers and with
  `M·(1−churn)` workers, same seeds, same tokens.
- Metrics: held-out loss at matched tokens; p50/p95/p99 outer-round latency; admitted-token
  fraction per round; bytes on the wire per outer round; goodput (useful tokens / wall-clock).
- Controls: positive = single-host run reproduces E5's best cell; negative = a run with the
  learners' network replaced by localhost must match single-host within CI (if it does not,
  the gap is not the network).

### E9 — Adaptive Byzantine evaluation (decides RQ8)

- Attacks: sign-flip (positive control, must be blocked), scaled, ALIE (small coordinated
  perturbation within the honest variance), label-flip on one shard; `f/n ∈ {0.1, 0.2, 0.3}`;
  non-IID shards from the 8-domain mixture.
- Defences: Multi-Krum + 2σ cosine alone; + 10 % spot-check replay; + both.
- Metrics: attack success (held-out loss increase vs no-attack run) and detection rate per
  defence; negative control = honest run through the same filter must lose < 1 % of updates.

## 5. Decision rules (what each outcome changes)

| Outcome | Roadmap consequence | Paper consequence |
| :-- | :-- | :-- |
| P2 refuted | RDA stays optional; add the `√M` outer-LR rule to the campaign config | §7.2 claims only norm bookkeeping, not "no retuning" |
| P3 refuted | delete `AdaptiveGraceWindowController`, keep a fixed admission budget | remove the EMA controller from §7.2 |
| P4: CGAD hurts under quorum | CGAD off by default when `K < M` | E2 reported as a no-quorum result only |
| P5 supported | ghost fraction becomes a logged, gated metric (`HM-S9-02`) | FoMoE cost model quoted with measured factor |
| P6 refuted | AdamW remains default inner optimizer | §7.5 MuLoCo claim limited to conditioning |
| P1 refuted | stop scaling the swarm; investigate which mechanism fails (E5 cells) | the thesis is reported as not supported |
| P8 supported | spot-check replay rate becomes the security parameter, not the filter | §4 rewritten around replay, not Krum |

## 6. Milestones and gates (owned by the engineering coordinator; gates judged by the scientific lead)

| Gate | Content | Evidence required to pass |
| :-- | :-- | :-- |
| **G1** (single-host science) | E5, E6, E7, E7b complete with 3 seeds | four `E<n>_report.json` with receipts; predictions ledger updated; paper v3.3 → v3.4 numbers through provenance lock |
| **G2** (3-node WAN) | E8 at 3 machines | latency and loss artifacts; negative localhost control passes |
| **G3** (8-node WAN + security) | E8 at 8 machines, E9 | RQ1 and RQ8 decided |
| **G4** (external review) | `HM-S9-09`: human or logged-model review with committed transcripts | review transcript in `paper/review/`; no script-emitted score anywhere |

A gate is not passed by a merged PR. It is passed when the artifact named in the evidence column
exists, its receipt matches, and the scientific lead has recorded the prediction status.

## 7. Handoff protocol between the two roles

1. **Scientific lead → coordinator**: a research direction is handed over as an experiment
   spec (`artifacts/experiments/E<n>_prereg.json` + the §4 text). The spec fixes metric, cells,
   seeds, controls and acceptance rule. Engineering may change *how* it is computed, never *what*.
2. **Coordinator → engineering tasks**: the coordinator splits the spec into `HM-S9-xx` tasks,
   assigns branches, runs the 1-hour orchestrator, reviews PRs for the Definition of Done in
   `ENGINEERING_COORDINATION_BRIEF.md`, and merges.
3. **Engineering → scientific lead**: results come back only as artifacts with receipts plus a
   one-page memo: what was run, what deviated from the spec, raw numbers, no interpretation.
4. **Scientific lead → paper**: interpretation, prediction status, and paper text. Numbers are
   pasted only from artifacts and locked with a `PROV-*` check.
5. **Disagreement**: if the coordinator believes a spec is infeasible or a control is wrong, it
   opens a GitHub Discussion tagged `spec-change`; the spec is amended *before* the run, never
   after seeing results.

## 8. Status log

### 2026-10-23 — E5-toy run as pre-registered; E8 two-host pilot

- **E5-toy** (`artifacts/experiments/E5_toy_report.json`, 360 runs, 3 seeds, all three controls passed): P2 **inapplicable** (mean pairwise cosine of learner deltas 0.909 > 0.3, so RDA = direct by construction and the best outer LR sits at the top of the grid for every cell); P3 **inconclusive** (random − adaptive = −0.0004 ± 0.0022 at matched wall budget, 24 pairs); P4 **refuted at this scale** (gate changes loss by −0.00063 ± 0.00029 under quorum; exactly 0 without stale deltas). Every cell lies within 0.011 of every other while the seed interval is ±0.024: the toy LM is a non-discriminating testbed. Consequence: the G1 cell of E5 must run on the `Qwen2.5-0.5B` 8-domain setting (`HM-S9-01`) and must report the delta cosine before any merge or gate claim.
- **E8-pilot** (two real hosts: this workstation and the GCP TPU VM `gwenlaya-tpu-1`, `us-west4-a`, ICMP RTT 151 ms, HTTP RTT through the SSH reverse tunnel ≈ 320 ms; `artifacts/experiments/E8_pilot_report.json`, 3 seeds per cell): (i) with `K = 2` (wait for both learners) the cross-host run reproduces the single-host baseline loss to within 1e-5 at every seed, which validates the shipping of shards and parameters and the merge arithmetic across machines, and gives the first real WAN round latencies (p50 ≈ 4 s for ≈ 2.7 s of local compute); (ii) with `K = 1` + adaptive grace and 400 inner steps, the faster learner (the 24-CPU VM) wins every quorum and the slower one is never merged, because the pilot learner loop is round-synchronous and discards work that misses the round instead of carrying it to the next merge with its token weight as in Decoupled DiLoCo's Algorithm 1; the run ends ≈ 1.0 nats worse than the baseline. This is a defect of the pilot protocol, not a measurement of Decoupled DiLoCo, and it fixes the first requirement of `HM-S9-01`: learners must keep training across rounds and be merged late with token weights. (iii) With 8 inner steps (RTT ≫ compute) the grace window sized from compute-step durations is shorter than one round trip, so the remote learner contributes only through one-round-late deferral; the loss difference to the baseline is within the seed interval of zero.
- Ledger: `artifacts/predictions_ledger.json` (P2 inapplicable, P3 inconclusive, P4 refuted at toy scale, P1/P5–P8 untested).
- Tooling leveraged from the owner's other projects: GWAYA zero-stub audit (`scripts/gwaya_stub_audit.py`, 74 files clean after fixing one `pass`-body hook), a logged local-model review through AutoevolveAI's `APIExtractor` (`scripts/logged_model_review.py`, transcript under `paper/review/`, no score), and the TPU VM as the second WAN host.

### 2026-10-23 (evening) — E5-het, E10-lattice, independent Lean gate, other projects leveraged

- **E5-het** (`artifacts/experiments/E5_het_report.json`, 2,652 runs incl. the pre-registered grid extension, 3 seeds, all controls passed, `ρ = 0` reproduces E5-toy to 1e-9): the heterogeneity knob works (delta cosine 0.909 → 0.079; `||RDA||/||direct||` 1.04 → 2.25 at M = 8, and the common-norm formula of the paper's Proposition predicts 1.80 / 2.27 from the cosines, within 1 %). On the original LR grid every best LR was censored at the top (the toy trains far below its optimal step); the pre-registered extension (amendment 2) found the optimum at the divergence edge: direct best at η = 6.79 for both M, RDA best at 4.8 for both M. **P2 refuted by the rule** (direct's best LR does not grow with M; the arg-mins are not separated by their intervals, so the rule needs an interval-separation clause at 0.5B), **P9 refuted at M = 8** (best-LR ratio √2 vs predicted 2.25), supported at M = 4; **P3 inconclusive** (+0.003 ± 0.012); **P4 refuted at every ρ** (−0.0025 ± 0.0023 at ρ = 4). P3/P4 were measured on the original grid, i.e. at LRs far below the optimum: low power.
- **E10-lattice** (`artifacts/experiments/E10_lattice_report.json`, 36 runs, 46 s; `E10_lattice_first_execution_lr2.4.json` kept as sensitivity): **P10a supported** (E₈/ℤ⁸ relative-MSE ratio 0.79 ± 0.02 at the matched ≈1.7-bit rate; Gaussian value 0.86), **P10b inconclusive** (−0.003 ± 0.019 with EF), **P10c supported** (−0.012 ± 0.077). The 2-bit cells are invalid (rates 3.10 vs 3.36 unmatched) and all realised rates drift above target because the scale was calibrated once on a round-0 delta: per-round scale adaptation is a prerequisite of `HM-S10-01`.
- **Independent Lean gate** (`artifacts/lean_independent_runner_check.json`): AutoevolveAI's `anse.formal.lean_runner` found 13 theorems with no `#print axioms` line; added; now 120/120, 0 sorry, 0 untrusted axioms. The theorem count was right, the axiom coverage was not. Re-run 2026-10-24 on the 128-theorem file that reached `main` by direct push: 128/128, 0 sorry, 0 untrusted; Theorems 125 and 127 flagged (tautological conjunct; unused gate hypothesis) → `HM-S10-09`.
- **Decision consequences applied**: PA-CGAD stays off by default under quorum (P4); RDA stays "norm bookkeeping" in the paper's claims (P2/P9 at toy scale); the E5 rule for 0.5B gains an interval-separation clause and an LR sweep up to divergence for both merges; the E₈ codec is a selectable option, not the default, until E10b at 0.5B (`HM-S10-01`).
- **Leveraged from the owner's other projects** (ROADMAP §3.8): OptimalEnergyAI decoders and rate accounting (E10), AutoevolveAI Lean runner (PROV-27) and APIExtractor (HM-S9-09), GWAYA zero-stub audit (78 files clean), runux signSGD as the 1-bit arm's motivation (re-implemented, licence), xavuntu `tpu_trainer.py` as the TPU-learner candidate (HM-S10-03), Xbrain/brain Rust learner feasibility (HM-S10-04).
- Ledger: `artifacts/predictions_ledger.json` now carries E5-toy and E5-het decisions per prediction plus P9 and P10a–c.

### 2026-10-23 (night) — publications leverage, RDA Proposition over ℝ, E11 design

- **Formal**: the RDA-multiplier Proposition is machine-checked over real inner-product spaces in
  `lean4/mathlib/RDAMultiplier.lean` (3 theorems; `r ≥ 1`, the Gram quadratic form, `R²/M` for orthogonal
  common-norm deltas), gated by `anse.formal.lean_runner` with standard axioms only
  (`artifacts/lean_rda_multiplier_check.json`, PROV-31). The RDA item of `HM-S9-08` is closed.
- **Directions from the owner's publications** (ROADMAP §3.8): E11 KV-cache codec (RQ10, `HM-S10-05`),
  Laya-LoRA curriculum as the campaign workload (`HM-S10-06`), one shared lattice package (`HM-S10-07`); the
  method papers (10.5281/zenodo.22829767, 23467992) are now the cited source of §4's protocol.
- Paper v3.5.1 (Directions subsection, 5 self-citations verified against the Zenodo API) republished on Zenodo.
- **External review of v3.5.0 received** (provenance unstated; recorded verbatim in `paper/review/external_review_v3.5.0_2026-10-23.md` with a point-by-point response). Accepted: hardware-heterogeneity caveat on the ε-spot-check (new RQ11/E12, `HM-S10-08`), compounding-approximations threat (E8 all-on vs each-off cells), PoC-scale caveat, architecture figure, Skip-Token dropout note, PA-CGAD-under-quorum note. Declined: moving the scripted-review correction out of the manuscript. Four reported typos were PDF-text-extraction artefacts (source and PDF text checked). Paper v3.6.0.

### 2026-10-23 (late) — E12 pilot run (CPU classes), gatekeeper in place

- **E12-pilot** (`artifacts/experiments/E12_pilot_report.json`; prereg before any run): same host + same thread
  count reproduces the 50-step LoRA segment bit for bit (P12a supported); **thread count alone** (1 or 8 instead
  of 4) diverges to 2.6e-4 / 2.5e-4 > ε = 1e-4 (P12b **refuted**); the TPU-VM CPU host (other CPU, torch build,
  library versions, same 4 threads) diverges to 1.5e-4, above ε but *less* than the thread cells (P12c refuted
  as stated: reduction order, not the host, dominates); tampered control 1.25e-3 with 96/96 tensors above ε vs
  3–5/96 for honest divergence. Decision applied: the
  spot-check rule is re-specified as execution-class matched (accelerator class, build, dtype, thread count /
  deterministic-kernel flags) with a per-class ε and a tensor-count term; GPU classes remain `HM-S10-08`.
- **Merge authority**: `scripts/pr_gatekeeper.py` runs hourly (owner-only automatic merges, PR text is data);
  branch protection on `main` blocks forced pushes and deletions and requires the two CI jobs.

### 2026-10-24 — E12 pilot complete: bf16 cell (P12d)

- **P12d supported**: the bf16-autocast cell (same host, build and 4 threads; fp32 master adapter weights) replayed
  against the fp32 reference diverges from step 0 to **3.3e-3** with 96/96 tensors above ε and a max loss difference
  of 1.7e-2 — 13× the honest fp32 divergence and **2.7× the tampered control** (1.25e-3). Both statistics of the
  re-specified rule (ℓ∞ and tensor count) saturate, so an fp32 validator cannot tell an honest bf16 worker from a
  tamperer at any ε: dtype is a class boundary; bf16 workers get a bf16 validator and their own ε measured on GPU
  (`HM-S10-08`, pre-registered consequence). The cell took 19,233 s on CPU (single-threaded bf16 path) vs 3–16 min
  for fp32 cells; the GPU bf16 class is the one that matters and is not measured here. Paper v3.7.1, PROV-34.

### 2026-10-24 (morning) — Lean gate at 128; T4 branch reviewed before its PR; `main` locked to PRs

- **Independent Lean gate re-run** on the 128-theorem file that reached `main` by direct push: 128/128, 0 sorry, 0 untrusted
  axioms (`artifacts/lean_independent_runner_check.json`, schema v1.1 keeps the 23 Oct run as `previous_run`). Theorems 125
  and 127 compile but do not constrain the σ-MoE gate / factored-gossip barrier they are named after → `HM-S10-09`.
- **T4 branch review** (`docs/review/t4_branch_review_2026-10-24.md`): E10b INVALID (positive control failed, seed 3);
  E11 Qwen-7B row invalid (negative control failed, −0.47 ± 0.20 nats); E12-GPU (same-T4 exact, every class crossing
  > ε, fp64+snap proxy does not help) and E13 (P13b refuted: the all-on swarm keeps −0.05 ± 0.35 of the single-learner
  gain; the K<M quorum with a persistent straggler is the mechanism, no-quorum retention 0.94) are sound and will be
  the headline of v3.8.0 (the branch has since been merged with `main` and the manuscript re-based as v3.8.0).
- **Governance**: `main` requires a PR and enforces admins after two coordinator commits with the owner's e-mail
  bypassed the gate; gatekeeper retries merges of ACCEPTed heads (PR #125).
- **Hardening (owner's brief)**: `HiveMindFirewall` delivered with a 59-vector scripted red team (first run 47/49, two defects
  fixed, second 49/49, 0 false positives); Season 11 tasks in ROADMAP §3.9; design `docs/SECURITY_HARDENING_PHASE.md`.
- **Compute economy (owner's brief)**: BOINC-credit coin and Nash-equilibrium market proposed (`docs/COMPUTE_COIN_AND_MARKET.md`);
  pre-registered simulation: Kelly efficiency ≥ 0.80, 300/300 Tullock certified, routing 55/60 undamped → 60/60 damped (run 1 kept).
  Proposed, not adopted (`HM-S12-02`); Season 12 in ROADMAP §3.10.

### 2026-10-24 — 10-hour T4 campaign: E11, E12, E10b decided as registered; E11b exploratory sweep

Report: `docs/benchmarks/T4_10h_campaign_report.md` (generated from the receipts by `scripts/write_t4_benchmark_report.py`).
Ledger: `artifacts/predictions_ledger.json` (P11a, P11b, P12-* added; E10b attached to P10a–c as an invalid decision).

- **E11 (RQ10)** on real KV tensors of Qwen2.5-0.5B-Instruct (fp16), Qwen2.5-Coder-7B-Instruct (fp16, embeddings on
  CPU) and Gemma-4-E4B (bf16, per-layer embeddings on CPU), WikiText-2, 1024-token windows, 3 seeds:
  **the Qwen2.5-Coder-7B row is INVALID** (the negative control `negative_A2at8_equals_fp8` failed at -0.47 +- 0.20
  nats: an 8-bit lattice cannot be half a nat better than FP8 unless the FP8 arm is broken on this float16 model), so
  its verdicts (P11a refuted, P11b 'supported on one matched cell') are WITHDRAWN and no 7B number is to be quoted
  until the FP8 arm has been investigated (scientific lead's review). **P11a refuted on Qwen2.5-0.5B on one
  rate-matched cell (R = 4; R = 2 and 3 are not rate-matched)** and on Gemma (3 matched rates; all intervals include
  0); 3 seeds give low power. **P11b refuted** on Gemma (E₈ already beats ℤ without the KLT) and inconclusive on 0.5B.
  Consequence: the KV ring stays on FP8 (`HM-S10-05` keeps the pluggable codec, adopts none). Amendment 1 (per-layer
  calibration rate) was made after the first 0.5B execution and before any 7B / Gemma result; on 0.5B the
  A₂@8-vs-FP8 negative control reads -0.04 nats (e4m3 keeps 3 mantissa bits, so FP8 is coarser than an 8-bit lattice;
  a design error, kept as registered and reported), so only Gemma passed every control.
- **E12 (RQ11)**: **P12-same supported** (same T4, separate process: ‖Δθ‖∞ = 0.0 in 9/9), **P12-cross supported** (T4 vs
  x86 CPU, fp32: 2.0e-4–6.3e-4 > ε in 9/9), controls pass. The fp64 + 2⁻²⁰ snapping proxy does not reduce the
  cross-device divergence, which sits near the inner lr from step 10 on — consistent with Adam's sign-like early steps
  turning any tiny gradient difference into an ≈lr step (interpretation). RTX 2070 / TPU cells and the fixed-point
  clause remain open. `HM-S10-08`: validators are assigned by accelerator class; cross-class replays are 'not comparable'.
- **E10b (RQ9 at 0.5B)**: first execution aborted by the lead after one cell (0.72 s/step vs 0.12 assumed; the
  E12-style LoRA init made the testbed diverge — confirmed by a CPU diagnostic); amendment 2 (PEFT-scale init, 5 arms,
  R = 6, H = 20, registered fail-fast rule). The amended run is **invalid**: its positive control failed (sign without
  EF did not hurt in seed 3; all arms end within ~1e-3 nats). Descriptive: relMSE(E₈)/relMSE(ℤ⁸) = 0.869 ± 0.003 at 1
  bit/dim on real LoRA deltas (Gaussian prediction 0.86). P10a–c keep their E10 (toy) statuses.
- **E11b** (exploratory, decides nothing): planned for the same codecs on fineweb-edu, code and GSM8K and at 4096 tokens.
  Correction (2026-10-24, day): only the three fineweb-edu 1024-token configurations ran; the other sweeps were stopped
  on purpose and never re-queued, so no E11b aggregate report exists (see the campaign report, sections 1b and 5).
- Code delivered on the way (Sonnet / Haiku subagents, reviewed by the lead): GPU lattice codecs with A₂
  (`hivemind.compress.kv_lattice`), pluggable KV page codecs, validator assignment, E10b runner, E11b aggregator, a
  lazy `peft` import in `hivemind.core.quantization` (the replay path no longer needs peft), 60+ CPU tests.

### 2026-10-24 (day) - E13d, E14, E15, E16 and the GWAYA-Lite deployment

Report: `docs/benchmarks/T4_10h_campaign_report.md` (sections 6-10, extension-queue timeline in 1b; generated by `scripts/write_t4_benchmark_report.py`).
Ledger: `artifacts/predictions_ledger.json` now carries P13a-c, P14a-b, P15a-e and P16a-f (per-model / per-tier decisions under `decisions`, `per_model`, `per_tier`; `mixed` = the per-group statuses disagree, `descriptive` = registered without a status).
Statuses of E14 / E15 are computed from the receipted reports by the registered rules in `scripts/analyze_E14_E15_hypotheses.py` (`E14_hypotheses.json`, `E15_hypotheses.json`). No peer review of record exists for any of this; PTS / $HIVE are non-monetary and no reward is attached.

- **E13 (registered, earlier on 2026-10-24; first logged here)**, localhost swarm over real HTTP on one T4, no WAN claim: P13a **supported** (S1 equals S0 bit-identically on Qwen2.5-0.5B, Granite-3.1-1B, Granite-3B; also with two real OS processes at M = 2), P13c **supported** (upload 0.09 of raw fp32 on all three models), P13b **refuted** on Qwen2.5-0.5B (retention -0.051 +- 0.350 against the 0.5 bar) and on Granite-1B (-0.658 +- 0.665); Granite-3B has one seed (descriptive, -0.698). Each-off cells (Qwen): without the quorum (K = M) retention is 0.942, without the codec 0.114, without RDA 0.264. E13c (rotating straggler, exploratory): retention +0.402 +- 0.304, so starvation of the persistent-straggler domain explains part of the loss and K < M itself the rest.
- **E13d (six non-IID domains, six workers, exploratory replication, amendment 6)**: P13a **supported**, P13c **supported**, P13b **inconclusive** (retention +0.290 +- 0.391, interval [-0.100, +0.681] includes the 0.5 bar); K = M keeps 0.942. The registered refutation at M = 4 is therefore not replicated as a refutation, and not confirmed either; three seeds, thread-hosted peers.
- **E14 (fair quorum S6 and deferred carry S7 against the plain K < M swarm S2; exploratory, 3 seeds, two hypotheses per model, no correction)**: dense Qwen2.5-0.5B: **H14a supported** (S2 - S6 = +0.124 +- 0.055 nats; retention 0.879 +- 0.079 against -0.051 +- 0.350 for S2; round latency 1.28x S2) and **H14b supported** (S2 - S7 = +0.077 +- 0.066; retention 0.528 +- 0.184). Granite-3.1-1B-A400M MoE: **H14a inconclusive** (+0.016 +- 0.026; retention -0.080), **H14b inconclusive** (-0.016 +- 0.034; retention S7 -0.665, S2 -0.371). The dense-model support does not transfer clearly to the MoE at this size. Decision consequence (as registered): S6/S7 are candidate admission policies for the multi-host gameday (`HM-S9-07`) and stay opt-in (`HM-S10-13`), to be confirmed with more seeds, on MoE and on the WAN; the constants (MAX_EXCLUSION 1, MAX_STALENESS 2) were not tuned.
- **E15 (GWAYA-gated RFT of Qwen2.5-Coder-1.5B-Instruct, 7.5 GiB cap on the T4; MBPP-sanitized n = 257, HumanEval n = 164)**: **H15b supported** (the GWAYA inference loop C1 - C0 = +7.0 points, 19 problems gained and 1 lost, McNemar p = 4.0e-05); **H15a inconclusive** (RFT-tuned greedy C2 - C0 = +0.8 points, p = 0.85: no detected gain from the fine-tuning; HumanEval C2h - C0h = +0.6 points, p = 1.0, descriptive); **H15d supported** (peak reserved 4.47 GiB at most); stacking C3 - C1 = +1.2 points (p = 0.66), C3 - C0 = +8.2 points (descriptive). RFT yield: 359 of 462 train problems gave at least one accepted sample, 673 kept pairs. One training seed, one model size; the loop costs about 4.4x the GPU seconds of greedy decoding. Decision consequence (as registered): the inference-time gate alone is the efficient path; training needs more verified data before it is worth recommending.
- **E16 (GWAYA draft PR #21 P1/P2 measured on the T4, Qwen2.5-Coder 1.5B/3B/7B Q4_K_M via Ollama, MBPP n = 257)**: H16a **supported** (7B: 5483 MiB at num_ctx 4096 and 6387 MiB at 8192, both under 6656 MiB), H16b **adequate** (all measured/estimated VRAM ratios in 0.93-1.21), H16c **supported** on all tiers (the loop never lowers pass@1), H16d **supported** for 1.5B (+8.6 points) and 3B (+7.4) but **inconclusive** for 7B (+3.9 points, p = 0.002, below the +5.0 bar), H16e **supported** (precision of VERIFIED 92.6 % / 92.8 % / 94.8 %, so 5-7 % of VERIFIED solutions still fail the hidden tests), H16f met (7B decode 28.5 tok/s). Descriptive: the 1.5B -> 3B -> 7B ladder reaches 85.2 % against 85.6 % for the 7B loop alone at 78 % of its model time. Run valid (idle GPU, gate, unload).
- **GWAYA-Lite deployment**: `scripts/gwaya_lite.py` (CLI + localhost OpenAI-compatible gateway, loopback only) runs on the T4 at port 8099; the live check (`artifacts/gwaya_lite_live_check.json`) shows 3 of 3 small tasks VERIFIED (the ladder escalated to 3B for fizzbuzz), a request without tests answered AUDITED, SSE streaming, foreign Host / Origin rejected with 403, 1639 MiB of GPU memory in use afterwards (qwen2.5-coder:1.5b resident, 1514 MiB). Three small tasks are a smoke check, not a benchmark; the editor-integration examples are untested in real editors.
- Also delivered: the Swarm Lab (`HM-S10-10`), campaign checkpointing for Spot VMs (`scripts/checkpoint_state.py`, `HM-S10-12`) and a local staging script for the campaign package (`scripts/publish_campaign_to_gcs.py`); the public upload of the package is pending the owner's permission and has not been done.

Caveats that bound every statement above:

- **Thread-hosting caveat.** Where the memory preflight did not allow separate OS processes (Qwen2.5-0.5B with 4-6 workers: 15 cells of the registered E13 run, the E13d and E14 Qwen cells; the Granite runs by choice), the peers are threads of one process sharing one frozen model, talking over the same real localhost HTTP protocol. Round latencies are GPU time-sharing artefacts; the exact-identity control was repeated with two real OS processes only at M = 2. Nothing here is evidence about WAN, NAT, volunteer churn or throughput (RQ1 stays untested).
- **Sandbox caveat.** The E15 / E16 / GWAYA-Lite gate is the vendored GWAYA zero-stub auditor (unmodified, pinned commit) plus our own user/mount/net/pid-namespace sandbox with rlimits: no seccomp, no cgroups, shared kernel, not GWAYA's Bubblewrap oracle. Results are labelled GWAYA-style and it is not a defence against adversarial code.
- **Timestamp note (unresolved).** The `registered_utc` stamps of several pre-registration files are later than the stage timestamps of the runs they claim to precede (E15 prereg 13:00Z and amendment 1 14:30Z against a baseline stage written 08:24Z; E16 prereg 16:00Z against a preflight at 10:07Z; E14 prereg 11:00Z against a job started about 07:06Z; E13 amendment 6 at 10:15Z against the E13d start at 07:33Z; amendment 4 at 09:00Z against a Granite-1B run of the first queue), while the file modification times place every pre-registration before its run. The stamps are probably hand-typed or in a different clock; "registered before the run" therefore rests on the file modification times (the commit history was not checked here), not on the stamps. Not corrected here (the files are receipts).
- Statistics: 3 seeds (df = 2 intervals) for E13/E14, one training seed for E15, one seed set for E16; E14 is exploratory and uncorrected for two hypotheses per model.

Still open: the multi-host WAN GameDay (E8, `HM-S9-07`); E14 with more seeds, a larger MoE and on the WAN, and a data-leasing alternative if the quorum problem persists; E15 with several training seeds or more verified data; E16 on other runtimes and GPUs; the E11b sweeps (only fineweb-edu ran), E12 on an RTX 2070 / TPU-VM CPU and the fixed-point replay clause; a longer E10b; a real-editor test of the GWAYA-Lite gateway; publication of the campaign package and of anything on the GWAYA repository or its PR is left to the owner.

**Provenance note on pre-registration timestamps (2026-10-24).** The `registered_utc` fields inside the E13 amendments 4 and 6 and the E14, E15 and E16 preregistrations are later than some runs they precede and must not be read as registration times. The git history is the authority: the E14 prereg was committed at 06:23Z (report 07:33Z), the E15 prereg at 07:28Z (report 10:07Z) and the E16 prereg at 08:27Z (report 11:21Z), each before its result. The E15 baseline stage began after the prereg commit. Fields are left as written.
