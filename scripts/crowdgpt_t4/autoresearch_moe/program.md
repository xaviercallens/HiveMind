# autoresearch-MoE: choosing the recipe for MoE campaign 2 (pre-registered 21 Nov 2026, before any trial)

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (March 2026), like
`scripts/crowdgpt_t4/autoresearch/` (the JEPA screen of 14 Nov):
- **one frozen judge:** our `prepare.py` is the evaluation, and the search can't change it;
- **a fixed training budget per trial;**
- **one change per trial;**
- **keep or discard;**
- **`results.tsv` stays untracked;**
- **simpler wins ties.**

## What is different from Karpathy, and why

| autoresearch | here | why |
|---|---|---|
| An agent edits `train.py` | Each trial is a **configuration change** of the existing worker's flags | The T4 is shared with a live inference server and has a 4,300 MiB floor. Code is not edited on it during a search. |
| 5 min of wall clock | **30,720 training tokens per learner** (30 rounds × 16 inner steps × 64 tokens) | One round takes ~4 min. Equal tokens keep a trial that changes the tokens per round (seq_len) fair. |
| One run, no noise control | **The baseline runs twice**, a change must beat the champion with a **paired CI that excludes 0**, and the winner is **re-run once** | A single noisy run picks lucky winners. |
| `val_bpb` | **Mid-row held-out NLL** (see the judge below) | Lessons 16 and 17 (`LL.md`): the training ratchet's row-start metric mostly measures format and expert order. |

## The judge (frozen, never used for training or acceptance)

- **Script:** `scripts/crowdgpt_moe_paired_eval.py` from PR #944, branch `feat/ip300-paired-moe-eval` at `9bebbfd1`.
- **Spec:** `docs/E_B3_MOE_MIDROW_PREREGISTRATION.json` from the same commit, sha256 `dce3b43e3cab6086…`.
  Only `arms.moe.checkpoint`, its source and its sha256 change per trial.
- **Data:** the same 240 chunks of never-used rows (rows after each source's training range), 40 per scored expert:
  E0, E1, E2, E5, E6, E7. E4 (physics) has no rows left, and E3 (logic) has no data.
- **Scoring:** each chunk is scored on tokens 64–127 of its row, with the model reading tokens 0–127.
- **Metric J:** the mean mid-row NLL of the MoE arm over the 240 chunks (lower is better).
- **Domain guard:** no scored expert's mid-row NLL may exceed the base model's by more than 1 %.
- **Runs on:** the CPU only (`CUDA_VISIBLE_DEVICES=`, nice 19).

## Fixed recipe shared by every trial (campaign recipe, 18–19 Nov)

`--mode moe-opendiloco-round --moe-split-mode lora_domain_experts`, 8 experts, `--real-data --local-data
--train-tokens-per-expert 262144 --heldout-tokens-per-expert 8192 --max-rows-per-source 5000 --eval-windows 224
--outer-lr 0.2 --outer-lr-decay 0.5 --outer-lr-growth 1.25 --outer-lr-min 0.02 --inner-steps 16 --seq-len 64
--stop-after-rejections 0`.
- **Start:** the upcycled start (no `--init-buckets`) at data offset 0.
- **Output:** the trial's best accepted state, by the ratchet, is what the judge scores.
- **If a trial is cut short** by a yield to live inference or by the 02:40 window, it continues from its best state at
  its next data offset until the token budget is spent.

## Trials (in this order)

| id | change from the champion | hypothesis, from this week's evidence |
|---|---|---|
| T1 | none: concatenated layout (`--interleave-block-tokens 0`) | baseline A |
| T2 | none: same as T1 | baseline B, the noise control: the T2 − T1 gap is reported |
| T3 | `--interleave-block-tokens 1024` | **H1** the experts' data interleaved, so every expert trains from the first rounds (lesson 17, PR #946) |
| T4 | `--seq-len 128`, 15 rounds (same tokens) | **H3** longer windows put more weight on content than on the row opening (mid-row finding, #944) |
| T5 | `--init-buckets` run 2 round 49 (`ea09e677…`) | **H2** start point: a lightly trained state against the upcycled start |
| T6 | `--lr 4e-4` (inner lr ×2) | **H4** a faster inner learning rate, given the slow early rounds |
| T7 | none: re-run the final champion | confirmation |

## Decision rule

- **The champion starts as T1.** T2 is compared with T1 only to estimate noise; it can't become champion.
- **From T3 on,** a trial **replaces the champion** only if its paired bootstrap 95 % CI of (trial − champion), per-chunk
  mid-row NLL over 10,000 resamples with seed 20261121, is **entirely below 0**, and the trial passes the domain guard.
  Otherwise it is discarded.
- **Ties:** the simpler configuration stays.
- **Crashes:** a crash is retried once. If it fails again it is logged as `crash` and skipped.
- **Final:** the champion after T6, confirmed by T7, which must also pass the guard, becomes the recommended recipe for
  campaign 2. If T7 does not reproduce the champion's J within the T1/T2 noise gap, the recommendation says "not
  confirmed".

## Not in this search (and why)

- **STP / JEPA terms:** that branch is closed by three pre-registered negative or uninformative verdicts
  (`docs/E_STP_RESULTS.md`). Reopening it needs a new pre-registration and the owner's decision.
- **Stripping dataset templates:** this needs code, not a flag. It is a separate PR with tests, and can join a later
  screen.

## Safety and cost

- **Guards:** every trial runs through `hivemind_work/launch_crowdgpt_moe_interleaved.sh`, which keeps all the usual
  guards (HOLD, producer stopped, one worker, free − 4,300 − 700 MiB CUDA cap, floor watchdog, `STOP_AT` 02:40). Runs
  are named `run_*_ar_full`, so `training_pipeline.py` never picks them up.
- **Window:** trials run only from 06:20 to 02:40 Paris. The AutoevolveAI jobs at 03:00 and 06:00 keep the night gap.
- **Cost:** the T4 VM already runs all day for the live inference server, so the extra GCP cost is ≈ 0.
- **Time:** about 7 × (2 h training + 20 min CPU judging) ≈ 17 h, so the T4 is busy until about Sunday 22 Nov 02:40
  Paris at the latest.
