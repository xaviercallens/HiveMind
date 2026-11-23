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
