# E-B3-MOE-2 (mid-row): pre-registration (IP-300 follow-up to #944)

**Registered 2026-11-18, before any number of this experiment was computed.** The frozen spec is
[`E_B3_MOE_MIDROW_PREREGISTRATION.json`](E_B3_MOE_MIDROW_PREREGISTRATION.json). If this page and the JSON disagree, the
JSON wins. Neither file is edited after the run; results go in a separate section at the end of this page.

## Why

[E-B3-MOE-PAIRED](E_B3_MOE_PAIRED_PREREGISTRATION.md) (#944) gave PASS: MoE − base = −0.3306 nats, 95 % CI
[−0.3865, −0.2770]. Every #944 chunk was the **first** 64 tokens of a row, in the training text format. Its own caveat:
part of the gain may be the add-on learning each source's format or template (most clearly E1 code, −22 %, on short
templated CodeAlpaca rows) rather than domain content. This follow-up tests that by scoring tokens from the middle of
the row.

## What changes, and what does not

Only the scored chunk position changes (and, as a consequence, which rows are long enough). These stay as in #944:

- the base model;
- the **primary checkpoint** (`0778b9ee…af781f8`, run 101755Z round 24, byte-copied to scratch because
  `hf_moe_best` now holds a newer round);
- the split and the loader;
- the test-row rule (rows from each source's `train_rows[1]`, at most 2,000 scanned per source, same skips: E3 never
  trained, E4 exhausted);
- the statistic and the decision rule.

## Chunks

- **Rows:** a row needs at least **128** token ids. Shorter rows are skipped and the next eligible row is taken.
- **Input and scoring:** the model gets `ids[0:128]`, so it sees the row's opening and template. Only the predictions
  of **tokens 64..127** are scored (64 predictions; labels 0..63 are masked). So the model sees the template but is
  judged on content.
- **Per expert:** the target is 40 chunks.
  - An expert with fewer than **20** chunks is left out of the bootstrap, the verdict and the domain guard, and is
    listed for information only.
  - N may differ per expert.
- **INSUFFICIENT_N:** fewer than 200 chunks in total over the experts that qualify.
- **Row set:** it differs from #944 because the length requirement is longer (expect more skipped rows in E1). No
  row-length census was run before registering.

## Arms and gates

1. **Base, dense:** mid-row score, plus the unchanged #944 row-start score on the same rows.
2. **Causality gate (chunk 0):** the masked score of tokens 1..63 on the 128-token input must equal the #944
   `chunk_nll(ids[0:64])` within 1e-4. Otherwise the run aborts.
3. **Base-equivalence gate:** upcycle with zero-delta buckets. The mid-row NLL on chunk 0 must equal the dense value
   within 1e-3. Otherwise the run aborts.
4. **Primary (decisive):** load `0778b9ee…`.
5. **Secondary (reported, not decisive):** load the current package checkpoint `58696cfe…1898fa9` (round 69 of the same
   run and manifest) into the same model; every bucket is overwritten. Same chunks.

CPU only, fp32 trunk, 4 threads, `nice 19`. The GPU is never touched.

## Decision rule (primary arm, mid-row)

The statistic is unchanged from #944: a paired bootstrap of `mean(d)`, `d = NLL_moe − NLL_base`, with 10,000
resamples, `default_rng(20261118)` and a percentile 95 % CI.

| Verdict | Condition |
|---|---|
| INSUFFICIENT_N | fewer than 200 chunks over the qualifying experts |
| **PASS** | CI upper bound < 0 |
| **HARM** | CI lower bound > 0 |
| **NO_GAIN** | otherwise |

**Domain guard:** every qualifying expert has mean `NLL_moe ≤ 1.01 × NLL_base`.

## Interpretation rule (format vs content), frozen now

The threshold is half of #944's effect: **−0.1653 nats**.

| Label | Condition | Meaning |
|---|---|---|
| MOSTLY_CONTENT | PASS and mid-row mean d ≤ −0.1653 | the #944 gain mostly survives away from the row start |
| MOSTLY_FORMAT_SOME_CONTENT | PASS and mid-row mean d > −0.1653 | the #944 gain is mostly format; a smaller content gain remains |
| FORMAT | NO_GAIN or HARM | no content gain mid-row; the #944 gain is attributed to format/row-start |
| NONE | INSUFFICIENT_N | no interpretation |

**Caveat stated in advance:**

- **More context:** mid-row tokens are predicted with more context, so base NLL should be lower and absolute gaps can
  shrink even with no format effect. Relative changes are reported next to nats.
- **Same-row control:** a row-start vs mid-row comparison on the same rows is reported with its own CI and the ratio
  `d_mid / d_start`. It separates chunk position from row selection.
- **Neither changes the label,** which is decided in nats as stated.

This follow-up does not change the #944 verdict, its receipt or `scripts/mvp_scorecard.py`.

## Receipt

The receipt is `artifacts/experiments/E_B3_MOE_MIDROW_2026-11-18.json`. It records:

- the sha256 of the spec and the pre-registration and script commits;
- the rows used per source;
- the sha256 of the 128-token chunk ids;
- the per-chunk NLLs (mid-row and row-start, every arm);
- the gates;
- start and end times;
- a `deviations` list.
