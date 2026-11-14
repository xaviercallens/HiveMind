# autoresearch: LLM-JEPA tuning for Crowd-v1 (written 2026-11-14 07:40 CET, before any run)

Adapted from karpathy/autoresearch (March 2026):
- one fixed evaluation and one metric;
- a fixed wall-clock training budget;
- one change per experiment, committed;
- keep or discard; `results.tsv` stays untracked.

## Protocol

| Item | Setting |
|---|---|
| Base | Crowd-v1 rev 5ac3916 + engram (frozen fp16) |
| Adapter | LoRA r64 |
| Optimizer | AdamW lr 2e-4, time-based warmup + cosine; effective batch 16 |
| Training data | NL-RX-SYNTH train lines 0–7,499 |
| Seed | 0 |
| Budget | **600 s of training wall-clock** per experiment (5 min on an H100 ≈ 10 min on the T4). Fixed-time is compute-fair, so JEPA's ~1.8× per-step cost counts against it. |
| **Metric** | `val_ce`: mean CE (nats/token) on the answer tokens of dev = train lines 7,500–7,999 (never trained on). Lower is better. The test set is not touched. |
| Guard rail | `far_loss` (general LM, E0 windows) ≤ far_loss(sft) + 0.05, otherwise discarded. P1 found that JEPA hurts this badly. |
| Secondary | `dev_acc300`: exact match on the first 300 dev examples. |
| Keep / discard | Keep = `val_ce` < `val_ce`(sft) **and** the guard rail holds. The bar is plain SFT, since JEPA only matters if it beats SFT at equal compute. Otherwise discard and git reset. |
| Order | sft, jepa (baselines), then H1, H2, H3, H4, H6. |
| Advance | If ≥ 2 kept, one combined run (the autoresearch "advance the branch" step). |
| Hard stop | Free VRAM ≥ 4,300 MiB (t4guard); stop by 11:00 CET so the T4 is free for the weekend training. |
| Selection | Rank all tested hypotheses by `val_ce` (guard rail applied); the 3 best are reported for confirmation at full protocol (3 seeds, test set). |

## Evidence from P1 (night of 2026-11-13)

- **E1:** no accuracy gain (16.90 vs 17.07 %).
- **E2:** the JEPA term collapses (1 − cos goes from 0.68 to 0.0005).
- **E3:** general loss is +0.90 nat worse than SFT.
- **E4:** λ = 4 is worse still.

## 10 hypotheses

| # | Hypothesis | Mechanism / evidence | Tested? |
|---|---|---|---|
| H1 | λ = 0.1 + loss dropout 0.5 | JEPA pressure too strong for LoRA (E3, E4: more λ, more damage); the paper's λ optimum varies by task. | **yes** |
| H2 | stop-gradient on the target (answer) view | Classic JEPA/BYOL anti-collapse: only the query side moves toward the answer (E2). | **yes** |
| H3 | centered cosine (subtract the batch mean of both views) | The collapse may be a shared direction, i.e. anisotropy (E2); removing the mean makes "same direction for everything" useless. | **yes** |
| H4 | InfoNCE with in-batch negatives (T = 0.07; an option in the reference repo) | Negatives make the trivial solution impossible: the query must match *its own* answer (E2). | **yes** |
| H5 | k = 1 predictor token (the paper's best for NL-RX-SYNTH) | A learned predictor slot instead of identity. Needs a reserved Qwen3 token; the setup is costlier. | no (cost) |
| H6 | align layer-16 states instead of the final (LM-head input) states | Keeps the JEPA pull away from the output layer, to protect general LM (E3). | **yes** |
| H7 | mean-pooled view embeddings instead of last-token | Less sensitive to the final token; a weaker signal. | no (lower prior) |
| H8 | LoRA rank 512 | The paper's LoRA gain is reported at r512. Does not fit the 2.7 GB T4 cap (~3.3 GB of adapter + Adam). | no (memory) |
| H9 | JEPA warm-start: SFT for the first 50 %, then JEPA | Lets the task be learned before alignment. Overlaps with H1. | no (overlap) |
| H10 | VICReg variance/covariance term on view embeddings | Explicit anti-collapse. Batch of 8 is too small for stable covariance estimates. | no (batch size) |

Tested: **H1, H2, H3, H4, H6**. They are the hypotheses that directly target the measured failure (E2/E3), fit the T4,
and are each a single small change.
