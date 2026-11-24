# E-MOE-C3: gentle expert interleave × router freezing, starting from `e50_r00` (pre-registration)

Owner request, 24 Nov 2026: *"Oui fais 1 2 et 3"*. Item 3 is a next campaign with gently interleaved experts, building on
what was learned about the router. This document is written and pushed **before any training or scoring** in this campaign.

## Known when written

- **Screen #948** (judged on the frozen 240 chunks):
  - T6 (seq 128, lr 4e-4, from run 2 round 49) was kept, and T7 confirmed it on data slice 2.
  - T3, interleave in 1,024-token blocks, was discarded: general text +17.5 %, stars −20.3 %.
  - The noise control T2b (same recipe, data slice 2) differed from T1 by **−0.0236 nat**, CI [−0.0305, −0.0166]. The
    judge's CI therefore does not cover run-to-run training variance.
- **Campaign 2** (T6 recipe, 428 rounds): −0.553 nat, but general text +2.21 %, which fails the +1 % guard.
- **Correction E-B3-MoE-C** (#944):
  - `e50_r00` (experts at λ 0.5, router and final LayerNorm at λ 0, i.e. their upcycle initial values) passes the guard
    on the skip-40 split (+0.10 %), the skip-80 split (+0.63 %) and the frozen 240 chunks (+0.30 %, −0.087 nat).
  - Full-strength experts with the router reset are harmful (`e100_r00`: general +14.9 %).
- **Data layout.** Each learner's shard holds about 3 experts back to back, 262,144 tokens each. Learner 0 has 786,432
  tokens and learner 1 has 870,059. One round uses 16 × 128 = 2,048 tokens per learner.
- **Row availability check** (counts only, no model, no scores), done on 24 Nov before this document: the mid-row judge's
  selection at `skip_eligible_rows = 120` yields 40 chunks for each of E0, E1, E2, E5, E6 and E7 (240 in total).

## Question

Starting from `e50_r00`, can more training improve the judged NLL beyond training noise while keeping every domain within
+1 % of the base? Which of two changes help:
- **gentle interleave**: 8,192-token blocks, so each learner switches expert every 4 rounds;
- **freezing the router gates and the final LayerNorm** during training (`--router-lr-scale 0`)?

## Arms (2 × 2), in run order

Every arm shares the following:
- init `e50_r00` (sha256 `80dccfa8…`);
- seq 128, inner lr 4e-4, 16 inner steps per round;
- **30 rounds**, i.e. 61,440 tokens per learner, from data offset 0;
- the screen's common flags (`--outer-lr 0.2 --outer-lr-decay 0.5 --outer-lr-growth 1.25 --outer-lr-min 0.02 --local-data
  --max-rows-per-source 5000 --train-tokens-per-expert 262144 --heldout-tokens-per-expert 8192 --eval-windows 224
  --stop-after-rejections 0`).

The training rows are therefore **the same rows as campaigns 1 and 2**. No new rows are read, so the judge's test rows,
which start after each source's training rows, stay unseen.

| order | arm | layout | router gates + final LN |
|---|---|---|---|
| 1 | **D** | interleave, 8,192-token blocks | frozen (`--router-lr-scale 0`) |
| 2 | **B** | interleave, 8,192-token blocks | learned (`--router-lr-scale 1`) |
| 3 | **C** | concatenated (`ILV=0`) | frozen |
| 4 | **A** | concatenated | learned (the campaign-2 recipe continued from `e50_r00`) |

**Layout confound (stated in advance).**
- In the concatenated arms, 30 rounds stay inside each learner's **first** expert: 61,440 tokens out of 262,144.
  So A and C train only 2 of the 6 judged experts.
- In the interleaved arms, every expert of both learners receives data, visited 2 or 3 times.
- The layout contrast is therefore mostly "2 experts trained versus 6", not an effect of order.
- Per-domain results will be reported in two groups: experts that received training tokens in that arm, and experts that
  did not. Untouched domains must not be read as protected by the layout.

## Code

- `--router-lr-scale` (this PR).
  - At 0, the router gates and the final LayerNorm get no gradient and stay out of gradient clipping during the inner
    steps, and their pseudo-gradient is exactly zero. They stay in the bucket layout, so checkpoints, `--init-buckets`
    and the judge are unchanged.
  - At 1, the optimizer is unchanged: a test checks that the result is bit-identical.
- Interleave: `--interleave-block-tokens` (#946, main).
- The worker runs from this branch at the commit that adds this document. The launcher
  (`launch_crowdgpt_moe_interleaved.sh`, not under version control) gains only `REPO=${REPO:-…/wt-campaign2}`, so it can
  run from this branch. Its sha256 was `9e6c5f0e…` before the edit and is `f2aa5db4…` after it; the old copy is kept
  beside it.
- Driver: `scripts/crowdgpt_t4/autoresearch_moe/campaign3.py`, committed with this document. It reuses `train()` from the
  #948 screen driver, with the SEC-01 fix. It refuses to start unless the worker checkout is at the commit that adds this
  document, the judge checkout is at `e5f6faf4`, the judge spec hash matches, and the `e50_r00` hash matches.
- T4 guards as before: HOLD file, the producer `hivemind-live-granite-moe` stopped during a segment and restarted after
  it, one worker at a time, a CUDA memory cap keeping **≥ 4,300 MiB free**, the 06:20–02:40 Paris window, and
  STOP_AT 02:40.
- If the window closes during an arm, the arm resumes in the next window from its last saved round, as the screen
  driver already does.
- **Hard stop: 25 Nov 2026, 21:00 Paris.** Arms not started by then are dropped in reverse order: A first, then C.

## Judge (fixed)

- Frozen mid-row judge `scripts/crowdgpt_moe_paired_eval.py`, through `scripts/crowdgpt_moe_multi_eval.py`, from a
  detached checkout at **`e5f6faf4`**: the code that produced the A2 and A4 numbers.
- Spec `docs/E_B3_MOE_MIDROW_PREREGISTRATION.json`, sha256 `dce3b43e…`.
- CPU only, RAM gate 12 GiB. The row-order check, the base-equivalence gate and the hash checks stay on.
- **Selection split: `skip_eligible_rows = 40`.**
  - The cached base arm (`base_skip40.json`) and the cached `e50_r00` arm (`e50_r00_skip40.json`) from A2 are reused only
    if their signature matches. Otherwise they are recomputed.
- **Confirmation split: `skip_eligible_rows = 120`.** It has never been scored. It is used **once**, in a single run that
  scores the winner, `e50_r00` and the base on the same chunks.

## Decision rule (selection split, skip 40)

- **Statistic:** per arm, the mean paired difference (arm − `e50_r00`) over the 240 chunks, with a paired bootstrap 95 % CI
  (10,000 resamples, seed 20261118). The per-expert change is measured against the base.
- **Guard:** every scored expert's mean NLL within **+1.0 %** of the base.
- **Minimum effect** (the training-noise floor from T2b, rounded): **0.025 nat**.
- **Keep** an arm if all three hold: the guard holds, the mean difference against `e50_r00` is ≤ −0.025 nat, and its CI
  upper bound is < 0.
- **Winner:** the kept arm with the lowest mean NLL.
- **If no arm is kept:** the result is *no tested recipe beats `e50_r00` by more than training noise within the guard*.
  `e50_r00` stays the recommended add-on, and the skip-120 split is **not** used.
- **Factorial contrasts** are descriptive, not a decision. Both are reported on the mean NLL and on general text %:
  - layout: (B + D)/2 − (A + C)/2;
  - router: (C + D)/2 − (A + B)/2.

  Contrasts smaller than 0.025 nat in absolute value are labelled *indistinguishable from training noise*.
- **Missing arms:** if fewer than 4 arms finish, contrasts are reported only where both sides exist, and the keep rule is
  applied to the arms that did finish.

## Confirmation rule (skip 120, once)

The winner, `e50_r00` and the base are scored together, once.
- **Confirmed:** the guard holds against the base, and the mean (winner − `e50_r00`) is ≤ −0.025 nat with CI upper bound < 0.
- **Partly confirmed:** the guard holds, the CI upper bound is < 0, but the absolute mean difference is < 0.025.
- **Not confirmed:** any other outcome.

Nothing is rerun. Nothing is promoted to serving by this campaign. Replacing `e50_r00` in #1006 would be a separate owner
decision.

## Reported in every case

- per arm: rounds run, rounds accepted, wall time, segments;
- the T4 free-VRAM minimum;
- per-domain % against the base, grouped by trained and untrained experts;
- the receipts in `artifacts/experiments/E_MOE_C3_*`.

## Deviation 1: worker interpreter (24 Nov 2026, 16:35 Paris, before any round)

**What happened.** The first launch of arm D (16:25) stopped at import: `ImportError: cannot import name 'UTC' from
'datetime'`. The launcher ran the worker with `boinc/venv`, which is Python 3.10 with torch 2.7.1+cu118, and current
main needs Python 3.11. Both segments stopped before any round, the driver stopped as designed, and the producer was
restarted. No checkpoint was written and nothing was scored.

**Change.** The launcher gains `PY=${PY:-…/boinc/venv/bin/python}`; its sha256 is now `c3ef8eb6…`. The driver sets `PY`
to `venv311`, which is Python 3.11 with torch 2.6.0+cu124, for every arm.

**Effect.**
- All four arms use the same interpreter, and the judge still runs on CPU from `e5f6faf4`, so the comparisons inside
  this campaign are unaffected.
- Training numerics may differ slightly from campaigns 1 and 2 (different torch and CUDA builds). This is one more
  reason the keep rule compares against `e50_r00` with a noise floor, not against earlier training runs.

**Unchanged:** the worker code, the arms, the judge, the splits and the rules. Only this section and the driver's
interpreter line are new in the commit that adds it, and the driver is relaunched pinned to that commit.
