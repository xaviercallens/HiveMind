# QueenBee experiment record

This file records every QueenBee experiment run, including the invalid ones. Each run's numbers come from the JSON report
named under it; nothing here was typed in by hand. The run is reproducible with
`PYTHONPATH=src python scripts/run_experiment_QB1_bandit_routing.py` (CPU only, 20 seeds, about 15 s).

## What is simulated

Four task clusters and five experts. Each (cluster, expert) pair has a fixed success probability drawn per seed from
Uniform(0.2, 0.9). A verifier reports the Bernoulli outcome. No language model is called. These runs test the mechanism:
learning from verified outcomes, warm-starting from dance records, and defending against dishonest dancers. They say
nothing about the quality of any real expert.

## Run history

| Run | Preregistration | Report | Verdict |
|---|---|---|---|
| QB1 | `artifacts/experiments/QB1_bandit_routing_prereg.json` | `QB1_QB1b_bandit_routing_report_INVALID.json` (fields `controls`, `hypotheses`) | **INVALID.** The positive control "the oracle is never beaten on realised success" failed: arms beat the oracle in 15 of 20 seeds. Realised success is a Bernoulli count drawn on a separate random stream for each arm, so a learning router can beat the oracle by luck. |
| QB1b | `QB1b_bandit_routing_prereg.json`, registered after QB1 was seen | same file (field `qb1b`) | **INVALID.** It scored the expected success of each decision and added a discriminating control. The wiring control, which I had described as true by construction, failed. Each arm drew its own cluster sequence, so even the oracle's expected success depended on which clusters it happened to get. The control caught a real harness defect. |
| QB1c | `QB1c_bandit_routing_prereg.json`, registered after QB1 and QB1b were seen | `QB1c_bandit_routing_report.json` (field `qb1c`) | **VALID.** All three controls pass. QB1c uses one cluster sequence per seed, shared by every arm (common random numbers). Hypotheses and thresholds are QB1b's, unchanged. |

QB1c is a corrected re-test, not a blind test. It was registered after two earlier runs had been seen, and the report says
so. On all three preregistrations, the hand-typed registration time was later than the run. Each was corrected to the time
the file was first written, and the file records that correction. After the correction the run was repeated, and every
number came out identical.

## QB1c controls

- Positive (wiring): the oracle's expected last-half success is greater than or equal to every other arm's, in every seed. **Pass.**
- Positive (discriminating): a router that picks the best expert 90% of the time sits strictly below the oracle and strictly
  above uniform, in every seed. **Pass.** The metric separates routers that are known to differ.
- Negative: in a flat world (every expert 0.55), cold Thompson's realised success is 0.5567, inside uniform's 95% interval
  [0.5437, 0.5581], and its regret is 0. **Pass.** The harness does not leak the answer.

The QB1 realised-success check was evaluated again on the QB1c run: the oracle is still beaten in 11 of 20 seeds. This
confirms that the check was measuring noise.

## QB1c results (20 seeds, 2000 rounds, mean and 95% interval over seeds)

| Arm | expected last-half success | realised | regret | early regret (rounds 1-200) |
|---|---|---|---|---|
| oracle | 0.7859 [0.7692, 0.8027] | 0.7906 | 0.0 | 0.0 |
| near_oracle_90 (control) | 0.7636 [0.7468, 0.7804] | 0.7651 | 45.0 | 4.3 |
| uniform | 0.5564 [0.5332, 0.5796] | 0.5575 | 455.1 | 44.3 |
| static keyword route (35% coverage) | 0.6375 [0.6192, 0.6558] | 0.6320 | 293.4 | 29.0 |
| Thompson, cold | 0.7692 [0.7518, 0.7867] | 0.7712 | 65.9 | 22.9 |
| Thompson, warm from honest dances | 0.7767 [0.7590, 0.7945] | 0.7745 | 32.2 | 9.7 |
| warm, 20% records replaced by sybils, cap 0.5 | 0.7705 [0.7533, 0.7878] | 0.7667 | 79.1 | 34.1 |
| warm, 40% replaced by sybils, cap 0.5 | 0.7653 [0.7475, 0.7831] | 0.7648 | 108.8 | 45.5 |
| warm, 40% replaced by sybils, "no cap" | 0.7749 [0.7574, 0.7923] | 0.7800 | 86.1 | 45.2 |
| warm, 40% sybils added to honest, no scout check | 0.7772 [0.7594, 0.7949] | 0.7761 | 72.0 | 41.3 |
| warm, 40% sybils added to honest, scout check | 0.7809 [0.7636, 0.7982] | 0.7834 | 15.3 | 3.6 |

## Hypotheses (QB1c)

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | Cold Thompson beats uniform and the static route on expected last-half success in at least 18 of 20 seeds | 20/20 and 20/20 | **supported** |
| H2 | Warm start from honest dances lowers early regret compared with cold, in at least 16 of 20 seeds | 20/20 (9.7 vs 22.9) | **supported** |
| H3 | With cap 0.5, 40% sybils cost at most 0.03, and the uncapped arm loses more | capped gap 0.0115, uncapped gap 0.0019 | **not supported**: the capped arm loses more than the uncapped arm |
| H4 | The scout cross-check lowers early regret when sybils are added on top of honest scouts, in at least 16 of 20 seeds | 20/20 (3.6 vs 41.3) | **supported** |

## Why H3 fails

1. The "no cap" arm was not uncapped. `BanditStrategy.warm_start` clips each weight to [0, 1], and a Wilson lower bound
   never exceeds 1. The comparison was therefore cap 0.5 against cap 1.0.
2. The cap acts on the score, not on influence. A record's weight is its Wilson lower bound, so a cap of 0.5 also cuts
   honest records for good experts (lower bounds around 0.78 at n=50) down to 0.5. That erases the honest signal that
   separates good experts from mediocre ones, at the same rate as it limits sybils.
3. In the "replace" design, each sybil occupies exactly one record per cell. Theorem 131 (influence at most count times cap)
   holds, but it never binds, because the count is 1.

What the data supports instead: limit influence by the evidence a record is allowed to carry, not by the level it claims.
The scout cross-check (H4) does this. It drops any record whose lower bound exceeds what an independent scout measured
for the same cell, and it removes almost all of the early-regret damage (3.6 against 41.3, close to the honest 9.7 and
better than it). Its weakness is stated in the preregistration: it assumes the scouts are honest, so scouts become the
next target.

The value cap stays in the code because Theorems 130-131 bound it, but it is not presented as a working defence.
QB-2 in the proposal adds a per-signer pseudo-trial budget, which bounds influence by evidence volume. It needs its own
theorem and its own preregistered test.
