# QueenBee: verified-feedback routing and knowledge sharing for HiveMind

Status: proposal with a working core. The Lean model, the Python package `hivemind.queenbee`, the tests, and one
simulation study (with two invalid runs on record) are in this repository. Nothing has been tested against real models
or real users yet.

Naming: the public name is **QueenBee**. The algorithm is registered as **ALIX** in the Lean specification
(`HiveMindSpec.ALIX_QueenBee`) and in `hivemind.queenbee.ALGORITHM_NAME`. Everywhere else it is called QueenBee.

## 1. The idea

A colony allocates foragers without a central plan. A scout that finds a good source dances, the dance recruits other
bees in proportion to the source's quality, and the dance fades when the source runs dry. QueenBee applies the same rule
to HiveMind's experts:

- **Queen**: the client-side router. It assigns each task to an expert. Splitting a task into subtasks, each with its own
  verifier, is roadmap item QB-4 and is not built yet.
- **Foragers**: the experts behind the OpenRouter gateway. These are models and adapters served on different GPUs.
- **Verifier**: unit tests, numeric checks, MCQ keys, labels, or Lean. A success counts only when a verifier says so.
- **Dance**: a signed, aggregate record of the form "expert E on cluster C: n verified trials, s successes, cost, epoch".
  It holds no prompts and no code.
- **Recruitment**: other clients warm-start their routers from dances. The record's weight is a pessimistic score. The
  record fades after a horizon, and the reporter must be the expert that actually served the work.

## 2. Three strategies, one manager

`StrategyManager(profile=...)` in `src/hivemind/queenbee/strategies.py` picks a strategy from the usage profile and logs
every routing decision as JSONL, without prompt text.

| Profile | Strategy | What learns | What leaves the machine |
|---|---|---|---|
| `ide_default` (**default**, the owner's preference) | a. Thompson bandit over experts per task cluster | Beta posteriors, updated only by verified outcomes | optional dance records (aggregate counts) |
| `verified_corrections` | b. preference pairs from corrections that a test confirms, for training a local adapter | a local adapter trained with DPO on (rejected, chosen) pairs; at most 30% unverified rows in the mix | nothing |
| `federated_optin` (explicit consent required) | c. clipped, noised adapter deltas, aggregated by Multi-Krum plus the DiLoCo quorum | a shared adapter | clipped, noised deltas; never raw pairs |

In b and c, routing is still done by an inner bandit: pairs and shared deltas improve the experts, and the bandit chooses
among them.

Known limitation of c: averaging LoRA factors A and B separately is not the same as averaging their product (FedEx-LoRA,
arXiv 2410.09432). Callers must pass product-space deltas or use an exact scheme. Clipping and noise are implemented, but
no (epsilon, delta) privacy guarantee is claimed.

## 3. Sharing knowledge as public data, for points

A HiveMinder can opt in to publish a verified triple: the task, the accepted solution, and the tests that accepted it,
under CC0-1.0, CC-BY-4.0, MIT or Apache-2.0 (`src/hivemind/queenbee/knowledge_market.py`). A share is admitted only if all
of the following hold:

- it is verified;
- its licence is on the permitted list;
- the sha256 of its content matches the declared one;
- the contributor is under the per-epoch identity cap (20 shares);
- it passes the HiveMindFirewall prompt-injection scan.

Each admitted share earns one knowledge point.

**Payment and the Nash equilibrium.** A fixed fraction of the validator's share of the epoch's fees becomes a bonus pool.
The pool is split in proportion to points. The bonus is a lump sum: it does not depend on how much capacity a producer
supplies or on how a consumer bids. Consequences:

- every best response, and therefore the Kelly market equilibrium of `hivemind.router.market`, is unchanged (Theorems 137-138);
- the payouts never exceed the pool (Theorem 139);
- `certify_knowledge_bonus` recomputes these facts and the win-win property before and after settlement. It never trusts
  stored numbers.

Knowledge points therefore join the existing equilibrium as a pure transfer and leave behaviour unchanged. Weighting
points by downstream use (for example Shapley-style) is deferred. The literature finds such estimates volatile (see the
review).

## 4. Lean formalisation (`lean4/HiveMindSpec.lean`, namespace `HiveMindSpec.ALIX_QueenBee`)

Core Lean only, no Mathlib. Each theorem has a `#print axioms` line. The independent gate
`anse.formal.lean_runner` reports 141 theorems in the file, none failing the whitelist {propext, Classical.choice,
Quot.sound}. All QueenBee theorems use only {propext, Quot.sound}. Receipt: `artifacts/lean_queenbee_check.json`.

| # | Theorem | Says |
|---|---|---|
| 129 | `faded_record_recruits_nobody` | an expired record has recruitment weight 0 |
| 130 | `recruit_weight_le_cap` | one record carries at most `cap` |
| 131 | `sybil_influence_le_count_times_cap` | k records carry at most k·cap |
| 132 | `score_monotone_in_successes` | more verified successes never lower the score |
| 133 | `score_antitone_in_trials` | padding trials without successes never raises it |
| 134 | `unverified_outcomes_never_move_the_count` | unverified outcomes leave the counts unchanged |
| 135 | `verified_rows_keep_mix_admissible` | adding verified rows keeps the 30% mix admissible |
| 136 | `dilution_cap_ratio` | admissible means 7u ≤ 3v |
| 137 | `lump_sum_bonus_preserves_best_response` | a lump sum does not change any best response |
| 138 | `lump_sum_bonus_preserves_nash` | ... so a Nash equilibrium stays one |
| 139 | `knowledge_bonus_payouts_never_exceed_pool` | proportional payouts sum to at most the pool |

What the proofs do not cover: they model integer scores, not the Wilson bound used in Python. The Python tests check that
the Python functions satisfy the same properties (Hypothesis), but the two are not linked mechanically.

## 5. What the experiments showed

See `docs/QUEENBEE_EXPERIMENTS.md`. In the valid simulation (QB1c):

- the bandit beats uniform and the static keyword route in 20 of 20 seeds;
- honest dances lower early regret from 22.9 to 9.7;
- a scout cross-check removes the early-regret damage from sybil dances (3.6 vs 41.3);
- **the value cap does not work as a defence** (H3 not supported). It caps the claimed level, which also cuts honest
  signal, rather than capping the evidence a record carries.

Two earlier runs are recorded as invalid. Both were caught by their own controls.

## 6. Roadmap

| Step | Deliverable | Gate |
|---|---|---|
| QB-0 (done) | Lean model, package, tests, simulation QB1c, literature review | this PR |
| QB-1 | replace the value cap with a per-signer pseudo-trial budget (influence bounded by evidence volume); a Lean theorem for it; a preregistered re-test of H3 with the cap really removed | G-R1 |
| QB-2 | scout trust: rotate scouts, require two independent scouts per cell, and test an adversary that corrupts one scout | G-R1 |
| QB-3 | bandit on the live gateway: route between the served experts on BigCodeBench-style tasks with executable tests; measure the routing plateau against the best single expert before claiming any routing gain | G-R1 |
| QB-4 | Queen decomposition: split a task into subtasks, each with its own verifier, and route each one; compare against planner-as-router baselines (arXiv 2609.22951, 2609.32917) | G-R1 |
| QB-5 | strategy b in an IDE: collect verified correction pairs from consenting developers, train a local adapter, and gate it with the verified update gate before it is served | G-R2 |
| QB-6 | strategy c across two hosts (the TPU VM and one GPU host): product-space or exact LoRA aggregation, a privacy accountant, and an attack test against Multi-Krum | G-R2 |
| QB-7 | knowledge points live: admission service behind the firewall, a public dataset with licence metadata, and settlement through the existing epoch ledger; the owner decides the bonus share | owner |

## 7. Not claimed

- No result with real models, real developers, real latency or real money.
- Theorem 131 holds, but the cap it bounds does not protect routing quality in our simulation.
- No differential-privacy guarantee for strategy c.
- Decomposition (QB-4) is not built.
