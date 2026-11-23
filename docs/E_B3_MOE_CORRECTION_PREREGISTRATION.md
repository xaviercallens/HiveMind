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

## Amendment A1: finer search between λ = 0.25 and 0.50 (owner, 23 Nov 2026, written before any new candidate is scored)

**Known when written:** the results above (W25 corrects on the selection split with J = 4.2428; W50 fails on general
text, +10.53 %; W25 on the frozen 240 chunks scores −0.079 nats).

- **New candidates:** W30, W35, W40 and W45, that is θ(λ) for λ ∈ {0.30, 0.35, 0.40, 0.45}, built with the same tool
  and the same start and C2 files.
- **Selection:** the same selection split (`skip_eligible_rows = 40`) and the same rule. A candidate corrects if every
  scored expert is within +1 % of the base and the mean CI against the base (seed 20261123) is entirely below 0. The
  chosen candidate has the lowest J among the correcting candidates, **including W25** (J = 4.2428).
- **Final test on a new, third split.** The frozen 240 chunks already scored the earlier choice. So the final test of this
  amendment uses the **next 40 eligible rows** per expert (`skip_eligible_rows = 80`), which no step has used. The
  chosen candidate and W25 are both scored there, for a paired comparison. An expert with fewer than 20 rows in this
  split is excluded and reported.
- **Outcome:** the chosen candidate replaces W25 as the recommendation only if it **corrects on the third split**.
  Otherwise W25 stays the recommendation.
- CPU only. Receipts go to `hivemind_work/autoresearch_moe/correction_a1/`.

### Amendment A1: partial results (23 Nov 2026, about 16:50 Paris; the run was stopped by the host for low memory)

Selection split, mid-row change against base Crowd-v1, %:

| Candidate | mean diff (nats) [95 % CI] | general | code | maths | exoplanets | stars | security | corrects |
|---|---|---|---|---|---|---|---|---|
| W25 (from above) | −0.083 [−0.093, −0.072] | +0.64 | −1.79 | −1.85 | −1.81 | −5.33 | −2.06 | yes |
| W30 | −0.098 [−0.114, −0.081] | **+1.87** | −2.20 | −2.17 | −2.35 | −7.16 | −2.59 | no |
| W35 | −0.112 [−0.135, −0.088] | **+3.70** | −2.61 | −2.46 | −2.96 | −9.34 | −3.17 | no |
| W40 | −0.126 [−0.156, −0.093] | **+6.00** | −3.03 | −2.56 | −3.69 | −11.88 | −3.83 | no |
| W45 | not scored (run stopped) | | | | | | | |

- **Status:** W30, W35 and W40 fail on general text, and W45 was not scored. The chosen candidate is therefore still
  **W25**. Its test on the third split (`skip_eligible_rows = 80`) has **not been run yet**.
- **Reading:** between λ = 0.25 and 0.40, general text worsens steadily as λ grows, about +1.3 points per 0.05. The
  other domains improve steadily. On this model, the trade-off is sharp just above λ = 0.25.

## Amendment A2: router-versus-expert interpolation, and the confirmation of W25 (owner, 23 Nov 2026, written before any new candidate is scored)

**Known when written:** the results above. W25 corrects on the selection split (J = 4.2428) and on the frozen 240
chunks. W30, W35 and W40 fail on general text (+1.87, +3.70 and +6.00 %). W45 was not scored, because the host stopped
the run for low memory, and W25 has not been tested on the third split (`skip_eligible_rows = 80`). The trainable
weights are 12 buckets: 8 experts (18.87 M parameters) and 4 router fragments (0.30 M, 1.6 %).

**Hypothesis (H-router).** The general-text regression comes mostly from the router, which learned to send general
tokens to specialist experts, rather than from the experts' own weights. If so, keeping the experts at a high λ while
pulling the **router** toward its start should keep most of the specialist gain with less regression.
*Prediction, informational:* at equal expert λ, a lower router λ gives a smaller general-text change.

**Candidates** (θ(λ_expert, λ_router) = start + λ·(C2 − start), per bucket group; same start and C2 files as above):

| name | λ_expert | λ_router |
|---|---|---|
| e45_r45 | 0.45 | 0.45 (the unscored W45) |
| e50_r00, e75_r00, e100_r00 | 0.50, 0.75, 1.00 | 0.00 |
| e50_r25, e75_r25, e100_r25 | 0.50, 0.75, 1.00 | 0.25 |
| e75_r50, e100_r50 | 0.75, 1.00 | 0.50 |
| e25_r100, e50_r100 | 0.25, 0.50 | 1.00 |

**Procedure and rule** (the rule of the first part, unchanged):
1. Score all 11 candidates on the **selection split** (`skip_eligible_rows = 40`). A candidate *corrects* if every
   scored expert is within +1 % of the base and the mean CI against the base (seed 20261123) is entirely below 0.
2. **Chosen** = the lowest J among the correcting candidates **and W25** (selection J = 4.2428).
3. Score the chosen candidate and W25 on the **third split** (`skip_eligible_rows = 80`). The frozen 240 chunks are not
   used again.
4. **The chosen candidate replaces W25 as the recommendation** only if it is not W25, it corrects on the third split,
   and its J there is lower than W25's. **W25 stays the recommendation** if it is chosen or if the chosen candidate does
   not meet that. In both cases W25's third-split result is reported, whether or not it corrects there.

**Execution.** CPU only. One process loads the model and the base arm once and scores each candidate (about 9 minutes
each), through `scripts/crowdgpt_moe_multi_eval.py`. It resumes from finished receipts, and it waits until at least
9 GiB of RAM is available before loading the model and before each candidate. The scoring is the judge's own mid-row
metric, without the same-row row-start control. Receipts: `hivemind_work/autoresearch_moe/correction_a2/`.

**Not started here (needs the owner's go, because it uses the T4 and stops the producer for about an hour):** if no
candidate beats W25, a short GPU continuation from C2 at a lower inner learning rate.

## Amendment A3: the GPU continuation F1 (owner's go, 23 Nov 2026 17:15 Paris, written before F1 is trained or scored)

**Known when written:** everything above. The A2 grid had not produced a single score yet: its CPU evaluator was
waiting for RAM because another project's job held about 8 GiB. The owner asked for the GPU continuation to go ahead
now, instead of only if no A2 candidate beats W25.

**Candidate F1.** Continue campaign 2's final state with the screen's recipe, except for a lower inner learning rate:
- start: `run_20261122T200945Z_c2_full/crowd_moe_opendiloco_round_84.safetensors` (C2, sha256 `fd0d41a9…` is the earlier
  round-77 package; C2 itself is the file scored above, copied to `correction/C2.safetensors`);
- `--seq-len 128`, `--lr 1e-4` (T6 used 4e-4), outer learning rate 0.2 with the adaptive rule of the campaigns,
  16 inner steps, **15 rounds** (30,720 tokens per learner), concatenated layout, no stall stop;
- data: start offset **72,149**, the campaign's natural continuation (learner 1's offset after C2's last round plus one
  round). Both learners had already wrapped around their shards, so this is a **second pass** over the first rows of
  E0 and E1, not new data;
- run on the T4 through the guarded launcher, in the 06:20–02:40 Paris window, with the producer stopped only while the
  worker runs, and only after at least 8 GiB of RAM is available. The final state is the last saved master state, which
  is the ratchet's best accepted state so far.

**Scoring and rule.** F1 is scored on the **selection split** (`skip_eligible_rows = 40`) by the same rule. It then
competes with the A2 candidates:
1. The chosen candidate of the selection split is the lowest J among all correcting candidates of A2, W25 and F1.
2. The chosen candidate and W25 are scored on the **third split** (`skip_eligible_rows = 80`). A2's runner scores its own
   choice and W25 there; if F1 is the choice, F1 is scored on the third split afterwards.
3. **F1 replaces A2's recommendation** only if F1 corrects on the selection split with a lower J than A2's chosen
   candidate, **and** corrects on the third split with a lower J than A2's recommendation there. Otherwise A2's
   recommendation stands.

The frozen 240 chunks are not used again. Receipts: `hivemind_work/autoresearch_moe/correction_f1/`. The final decision
is applied by hand from the receipts, following the text above.

### Amendment A3: results (23 Nov 2026, 19:20 Paris)

Receipts: `artifacts/experiments/E_B3_MOE_CORRECTION_F1_training_2026-11-23.json` and `..._F1_selection_2026-11-23.json`.

- **Training (T4, 17:12 to 18:48 Paris):** 15 rounds, 7 accepted, from C2 with inner lr 1e-4 at offset 72,149 (a second pass over
  the first rows of E0 and E1). Checkpoint `crowd_moe_opendiloco_round_15.safetensors`, sha256 `fe6cfaf89cbf9b59...`. The worker used
  about 2.8 GiB of RAM, free VRAM stayed at 6,957 MiB or more, and the producer was stopped only while it ran.
- **Selection split, mid-row change against the base:** mean **-0.559** nats (CI [-0.641, -0.477]); general text **+3.45 %**, code -7.39 %,
  maths -4.40 %, exoplanets -10.88 %, stars -49.91 %, security -15.77 %. **F1 does not correct** (general text above +1 %).
- **Per the rule of A3, F1 cannot replace A2's recommendation.** A lower inner learning rate for 15 more rounds did not undo
  the regression: F1 is close to C2 (selection: C2 +3.33 % on general text, F1 +3.45 %).
- *Clarification:* the "sha256 `fd0d41a9...`" remark in A3's text refers to the earlier round-77 package and not to C2. C2 is the file
  scored above, a copy of `run_20261122T200945Z_c2_full/crowd_moe_opendiloco_round_84.safetensors`.

### Amendment A2: results (23 Nov 2026, 20:47 Paris)

Receipt: `artifacts/experiments/E_B3_MOE_CORRECTION_A2_decision_2026-11-23.json` (selection and third-split results, per chunk
aggregates). Mid-row change against base Crowd-v1, %, columns general, code, maths, exoplanets, stars, security.

| candidate (experts_router) | split | J | mean diff (nats) [95 % CI] | general | code | maths | exoplanets | stars | security | corrects |
|---|---|---|---|---|---|---|---|---|---|---|
| e45_r45 | selection | 4.1845 | -0.1410 [-0.179, -0.100] | +8.37 | -3.46 | -2.67 | -4.45 | -14.82 | -4.53 | no |
| e50_r00 | selection | 4.2312 | -0.0943 [-0.105, -0.083] | +0.10 | -1.69 | -1.98 | -2.27 | -6.06 | -2.11 | yes |
| e75_r00 | selection | 4.2369 | -0.0886 [-0.112, -0.064] | +3.14 | -0.66 | -0.52 | -2.95 | -10.77 | -2.43 | no |
| e100_r00 | selection | 4.5532 | +0.2276 [+0.181, +0.278] | +14.89 | +8.01 | +11.72 | +1.01 | -6.90 | +1.44 | no |
| e50_r25 | selection | 4.1770 | -0.1485 [-0.190, -0.104] | +9.38 | -3.46 | -2.58 | -5.05 | -16.42 | -4.78 | no |
| e75_r25 | selection | 4.0321 | -0.2935 [-0.357, -0.228] | +11.21 | -4.48 | -3.03 | -8.91 | -31.44 | -8.84 | no |
| e100_r25 | selection | 3.8986 | -0.4270 [-0.507, -0.346] | +6.94 | -3.39 | -1.55 | -10.83 | -44.91 | -12.65 | no |
| e75_r50 | selection | 3.9724 | -0.3532 [-0.418, -0.286] | +9.56 | -5.67 | -4.19 | -9.44 | -34.24 | -10.15 | no |
| e100_r50 | selection | 3.8181 | -0.5075 [-0.587, -0.427] | +5.04 | -5.56 | -3.69 | -11.20 | -47.30 | -14.81 | no |
| e25_r100 | selection | 4.2255 | -0.1001 [-0.115, -0.084] | +1.53 | -2.67 | -2.49 | -1.42 | -6.83 | -2.78 | no |
| e50_r100 | selection | 4.1336 | -0.1920 [-0.242, -0.138] | +11.16 | -4.57 | -3.35 | -5.50 | -20.89 | -6.31 | no |
| W25 | selection | 4.2428 | -0.0828 [-0.093, -0.072] | +0.64 | -1.79 | -1.85 | -1.81 | -5.33 | -2.06 | yes |
| e50_r00 | **third** | 4.2252 | -0.0857 [-0.097, -0.075] | +0.63 | -1.52 | -1.90 | -1.92 | -6.03 | -2.08 | yes |
| W25 | **third** | 4.2324 | -0.0785 [-0.089, -0.068] | +1.04 | -1.88 | -1.74 | -1.78 | -5.20 | -2.02 | no |

**Outcome (rule of A2): `e50_r00` is chosen on the selection split and replaces W25 as the recommendation.** It corrects on the
third split (every domain within +1 %, general text +0.63 %, mean -0.0857 nats) with a lower J
than W25 there (4.2252 against 4.2324).
Checkpoint: `autoresearch_moe/correction_a2/interp_e50_r00.safetensors`, sha256 `80dccfa8246058ea...`: experts at 50 % of the way from the
zero-delta start to C2, **router fragments reset to their start** (λ_router = 0).

**W25 does not confirm on the third split:** its general-text change is +1.04 %, just above the +1 % limit (selection +0.64 %, frozen 240 chunks +0.77 %). The margin of this guard is thin
and the general-text figure moves by several tenths of a point between splits.

**Reading.**
- **H-router is supported only partly.** At experts λ = 0.50, router λ = 0 gives general +0.10 %, against +9.38 % at 0.25 and +11.16 % at 1.0.
  At experts λ = 0.75, router 0 gives +3.14 %, against +11.21 % at 0.25 and +9.56 % at 0.5. A router reset helps when the experts are shrunk.
  With full experts it is **harmful**: e100_r00 is +14.89 % on general text and a positive mean difference (+0.2276), because the experts were trained
  jointly with their router. So the router does carry the regression, but the experts and the router cannot be pulled apart freely.
- **The gain is small:** `e50_r00` keeps about 16 % of campaign 2's mean gain (-0.086 of -0.553), a little more than W25 (-0.079) and with more margin on general text.
- Nothing here changes the earlier conclusion that no candidate passes the guard together with the full gain of campaign 2.
- **Operational:** the evaluator's RAM gate was raised from 9 to 12 GiB (no effect on results). The first A1 run was stopped by the host for low memory
  (see above); A2 waited for RAM instead (about one hour in total).
