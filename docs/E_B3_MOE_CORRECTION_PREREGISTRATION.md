# E-B3-MoE-C: correcting campaign 2's general-text regression (pre-registered 23 Nov 2026, before any candidate is scored)

## What is already known when this is written

- **Campaign 2's final state** (`run_20261122T200945Z_c2_full/crowd_moe_opendiloco_round_84.safetensors`) was scored once
  by the frozen mid-row judge (`E_B3_MOE_MIDROW` spec, 240 never-used chunks):
  - J = 3.7337 against base 4.2864, a difference of −0.553 nats, 95 % CI [−0.633, −0.473];
  - per expert against the base: E0 general **+2.21 %**, E1 code −7.19 %, E2 maths −3.33 %, E5 exoplanets −9.55 %,
    E6 stars −49.42 %, E7 security −16.86 %;
  - the +1 % per-domain guard against the base **fails** on E0.
- **More general-text data is not the fix.** Learner 0 had already wrapped back into E0's training data and re-read
  about 150,000 general-text tokens at the end of campaign 2, and E0 was still +2.21 %.
- **The frozen 240 chunks have already scored about ten checkpoints.** They are not used for selection here.

## Selection split (new, disjoint)

- The same spec and rows, with `chunking.skip_eligible_rows = 40`: per expert, the **next 40 eligible rows** after the 40
  the frozen judge uses. They are in the same sources and still after each source's training range.
- The option and its test are in `scripts/crowdgpt_moe_paired_eval.py`; its default of 0 leaves the frozen judge
  unchanged.
- An expert with fewer than 20 eligible rows in this split is excluded from it and reported. E4 has no rows left and E3
  has no data, as before.

## Candidates

| id | candidate | cost |
|---|---|---|
| C2 | campaign 2's final state (reference, λ = 1) | CPU |
| W75, W50, W25 | weight interpolation θ(λ) = start + λ·(C2 − start), λ = 0.75, 0.50, 0.25 (WiSE-FT-style), over every trainable bucket (LoRA experts and router). `start` is the zero-delta upcycled state, which reproduces the base exactly. Tool: `scripts/crowdgpt_moe_interpolate.py`. | CPU |
| F1 | **only if no W candidate corrects:** continue from C2 with inner lr 1e-4 (4× lower) for 15 rounds at seq 128 (30,720 tokens per learner), on the next data. Learner 0 re-reads E0's training rows: that is a second pass, not new data. | T4, about 1 h, in the 06:20–02:40 window, with the usual guards |

## Decision rule (on the selection split only)

- **A candidate "corrects"** if, on the selection split:
  1. every scored expert's mid-row NLL is within **+1 % of the base**; and
  2. the mean MoE − base difference has a 95 % paired bootstrap CI (10,000 resamples, seed 20261123) **entirely below 0**.
- **The chosen candidate** is the one with the lowest J among those that correct. The interpolation candidates are
  scored first; F1 runs only if none of them corrects.
- **One final test:** the chosen candidate is scored **once** on the frozen 240 chunks. That test reports the result,
  including whether the guard holds there. It is not used to choose.
- **If no candidate corrects,** the result is "not corrected at this budget", and the add-on stays unpromoted.

## Execution

- CPU only for evaluation (`CUDA_VISIBLE_DEVICES=`, nice 19, 4 threads). F1, if it runs, goes through the guarded
  launcher.
- Outputs go to `hivemind_work/autoresearch_moe/correction/`.
- The receipts and a results section are added to this file afterwards. This pre-registration text is not edited.

## Results (added 23 Nov 2026, 11:30 Paris; the pre-registration above is unchanged)

Receipts: `artifacts/experiments/E_B3_MOE_CORRECTION_decision_2026-11-23.json` (all candidates) and
`E_B3_MOE_CORRECTION_W25_frozen_2026-11-23.json` (final test). Mid-row change against base Crowd-v1, %.

| Candidate | split | mean diff (nats) [95 % CI, seed 20261123] | general | code | maths | exoplanets | stars | security | corrects |
|---|---|---|---|---|---|---|---|---|---|
| C2 (λ = 1) | selection | −0.548 [−0.630, −0.466] | **+3.33** | −6.41 | −3.64 | −10.79 | −49.56 | −16.15 | no |
| W75 | selection | −0.394 [−0.460, −0.326] | **+8.30** | −6.18 | −4.93 | −9.65 | −36.60 | −11.08 | no |
| W50 | selection | −0.163 [−0.208, −0.114] | **+10.53** | −3.94 | −2.83 | −5.27 | −18.09 | −5.42 | no |
| W25 | selection | −0.083 [−0.093, −0.072] | +0.64 | −1.79 | −1.85 | −1.81 | −5.33 | −2.06 | **yes** |
| **W25, final** | **frozen 240** | **−0.079 [−0.089, −0.068]** | **+0.77** | −1.87 | −1.57 | −1.64 | −5.25 | −2.07 | guard holds |

**Outcome: corrected.** W25 is chosen (the only candidate that corrects). On the frozen 240 chunks, scored once, every
domain is within +1 % of the base and the gain is −0.079 nats. The frozen judge's own CI (seed 20261118) is
[−0.088, −0.068], and the add-on is lower on 202 of 240 chunks. The row-order and base-equivalence gates pass. The
checkpoint is `autoresearch_moe/correction/interp_lambda_0.25.safetensors` (sha256 `ab6c8fdc…`).

**Reading.**
- The correction keeps about 14 % of campaign 2's gain (−0.079 of −0.553). It is a safe operating point, not a free fix.
- General text is **not monotone** in λ: it is worse at λ = 0.75 and 0.50 than at λ = 1. Interpolation is not a simple
  dial between gain and regression on this model; only the low end of the range repaired it.
- The size of the gain (−0.079) is about the same as the screen's recipe T6 (−0.080, also passing the guard), reached
  by a different route.
