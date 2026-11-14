# CrowdGPT × LLM-JEPA: complete results and improvement proposal (2026-11-13/14)

Consolidated report of every measurement made on the HiveMind Tesla T4 for `Vxtzq/Crowd-v1`. The French study with the
full reasoning is `docs/CROWDGPT_LLM_JEPA_STUDY.md`. The paper version is
`paper/crowdgpt_llm_jepa/crowdgpt_llm_jepa.pdf`. Every number below comes from a file in `artifacts/experiments/`;
nothing is estimated.

## 1. Summary

| Question | Answer (measured) |
|---|---|
| Is Crowd-v1 healthy? | Yes. Held-out CE is 3.03–3.19 nats on its own data. The earlier "8–11 nats" came from an old revision measured by a loss check that **ignored the engram table** (+0.35 nat without it), now fixed. |
| Does LLM-JEPA help Crowd-v1 (paper λ = 1)? | **No.** 16.90 ± 1.06 % vs 17.07 ± 0.93 % exact match (3 seeds), and general loss **+0.90 nat** worse. The JEPA term collapses. |
| Can tuning rescue it? | Partly. Anti-collapse (stop-gradient, centering, InfoNCE) removes the damage, and λ = 0.1 + loss dropout restores task learning, but **no variant beats SFT at equal compute**; the best ties it. |
| Is it specific to CrowdGPT? | **Open.** The paper reports +11.4 points at LoRA r64 on an *Instruct* model with lr 2e-5 and k = 1. P2 (paper recipe on Qwen3-0.6B and Crowd-v1) is ready; the T4 is reserved for training this weekend. |
| What improves Crowd-v1 most? | **The data mix.** The gap to Qwen3-0.6B is +3.84 nat on Python code and +2.44 on math, against +0.36 on its own data. |

## 2. Setup common to all experiments

| Item | Setting |
|---|---|
| Model | `Vxtzq/Crowd-v1` rev `5ac3916` (step 366, 657,787,392 trunk parameters + 350,000,640 engram parameters), Qwen3 tokenizer (vocab 151,669) |
| Hardware | the shared HiveMind Tesla T4, beside live inference |
| Safety | free VRAM always ≥ 4,300 MiB (AGENTS.md); per-process cap 2.7 GB; pause during inference requests; checkpoint/resume; yield to the owner's training |
| Fine-tuning task | NL-RX-SYNTH (natural language → regular expression); 8,000 train, test = the first 1,000 of 2,000; exact string match (as the reference `evaluate.py`) |
| Adapter | LoRA r64 (alpha 64) on wq/wk/wv/wo/w1/w2/w3; base and engram frozen (fp16) |
| General-loss check | mean CE on 12 fixed 2,048-token windows of chunk 1026 at offset 200 MB (base model: 3.194) |

## 3. E0: held-out loss of Crowd-v1 (inference)

12 windows × 2,048 tokens per source; mean (sd).

| Model | Chunk 1026 @0 | Chunk 1026 @200 MB | wikitext-2 test |
|---|---|---|---|
| Crowd-v1 + engram (fp16 = fp32 = bf16) | 3.030 (0.355) | 3.194 (0.403) | 4.855 (0.345) |
| Crowd-v1 **without** engram | 3.376 (0.330) | 3.570 (0.377) | 5.055 (0.308) |
| Qwen3-0.6B | 2.536 (0.300) | 2.776 (0.293) | 3.020 (0.268) |

- **The engram matters (+0.35 nat without it).** HiveMind's loss check now models it; the receipt binds trunk ‖ engram.
- **No memorization.** Cohort mode reads 1,000 windows of a 1,280-window sub-slice (0.78 passes). The seen/far gap is
  0.16 nat, against 0.24 for Qwen.

## 4. P0: per-domain gap (inference)

8 windows × 2,048 per domain.

| Domain | Crowd-v1 | Qwen3-0.6B | Gap |
|---|---|---|---|
| Python code (codeparrot-clean-valid) | 5.187 | 1.347 | **+3.840** |
| Math (NuminaMath-CoT test) | 3.330 | 0.888 | **+2.442** |
| Encyclopedic (wikitext-2 test) | 4.709 | 2.909 | +1.800 |
| Science (SciQ test) | 3.821 | 2.319 | +1.501 |
| Educational web (fineweb-edu) | 3.992 | 2.948 | +1.044 |
| CrowdGPT's own chunk | 3.122 | 2.758 | +0.365 |

No French source was available offline, so French was not measured.

## 5. P1: pre-registered pilot, SFT vs SFT + LLM-JEPA

Protocol written before any run (`scripts/crowdgpt_t4/P1_GOAL.md`):
- AdamW lr 2e-4 with cosine schedule; effective batch 16; 3 epochs (1,500 steps).
- JEPA arm: λ = 1, k = 0. View 1 = the query, view 2 = the regex, packed with a block-causal mask. The engram bigram is
  reset at the start of view 2. Embeddings are the last-token final hidden states; no stop-grad.

| Seed | SFT acc % | JEPA acc % | Paired Δ | SFT general | JEPA general |
|---|---|---|---|---|---|
| 0 | 16.0 | 15.7 | −0.3 | 3.789 | 4.605 |
| 1 | 17.7 | 17.7 | 0.0 | 3.719 | 4.652 |
| 2 | 17.5 | 17.3 | −0.2 | 3.830 | 4.770 |
| **Mean ± sd** | **17.07 ± 0.93** | **16.90 ± 1.06** | **−0.17** | **3.779** | **4.676** |

- **Criteria:**
  - floor (≥ 5 %): passed;
  - success (Δ > 2 × pooled sd = 1.99, and general loss ≤ SFT + 0.05): **not met**;
  - paper-like gain (≥ +10): **not met**.
- **Exploratory λ = 4:** 15.4 %, general loss 5.012.
- **Prefix match (secondary metric, seeds 1–2):** SFT 20.6 / 19.4, JEPA 19.5 / 19.8.
- **Collapse:** 1 − cos is 0.15 at step 25 and ≤ 0.0005 at the end, in every JEPA run.
- **Cost:** JEPA takes 1.45–1.61 s/step against 0.71–0.83 for SFT, about 1.9×. Peak memory is 2.25–2.27 GB vs
  2.12–2.14 GB.

## 6. autoresearch screen (after karpathy/autoresearch)

Setup:
- Fixed **600 s** training budget per experiment.
- Metric: answer-token CE on 500 dev examples (train lines 7,500–7,999, never trained on).
- Guard rail: general loss ≤ 3.446.
- One commit per experiment, kept or reset.
- 10 hypotheses written before the first run (`scripts/crowdgpt_t4/autoresearch/program.md`).

| Run | val CE | Δ vs SFT | General | Guard | Steps | JEPA term at end |
|---|---|---|---|---|---|---|
| SFT | **2.5034** | — | 3.396 | — | 496 | — |
| JEPA λ = 1 | 2.7348 | +0.231 | 4.159 | ✗ | 385 | 0.0025 |
| H1 λ = 0.1 + LD 0.5 | 2.5128 | +0.009 | 3.474 | ✗ | 545 | 0.023 |
| H2 stop-grad | 2.8030 | +0.300 | 3.399 | ✓ | 365 | 0.081 |
| H3 centered | 2.9624 | +0.459 | 3.323 | ✓ | 355 | 0.007 |
| H4 InfoNCE | 3.2738 | +0.770 | 3.253 | ✓ | 305 | 0.108 |
| H6 layer 16 | 2.6085 | +0.105 | 4.352 | ✗ | 482 | 0.0017 |
| *H1+H2 (post hoc)* | 2.5403 | +0.037 | 3.430 | ✓ | 546 | 0.179 |
| *H1+H3 (post hoc)* | 2.5048 | +0.001 | 3.452 | ✗ | 572 | 0.197 |

- **Pre-registered top 3:** H2, H3, H4.
- **Untested:** H5 (k = 1), H7 (mean pooling), H8 (r512, too big for the cap), H9 (warm start), H10 (VICReg).
- **Caveats:** one seed per run. In the loss-dropout arms, the dropout draw shared the data-order RNG, which adds noise to
  their comparison with SFT; fixed afterwards.

## 7. Is the null result specific to CrowdGPT?

The paper's Table 8 (NL-RX-SYNTH, 5 runs per cell, lr 2e-5, λ = 1, k = 1):

| LoRA rank | SFT | JEPA |
|---|---|---|
| 32 | 6.09 ± 0.55 | 7.45 ± 1.87 |
| **64** | **21.09 ± 1.90** | **32.46 ± 1.26** |
| 128 | 34.21 ± 2.82 | 48.45 ± 3.66 |
| 256 | 45.57 ± 4.52 | 60.80 ± 2.31 |
| 512 | 50.18 ± 5.15 | 72.41 ± 2.94 |

Rank therefore does not explain P1. The remaining differences:
- learning rate (2e-4 vs 2e-5);
- k (0 vs 1);
- epochs (3 vs ~4);
- the model. The paper tests Instruct models only; Crowd-v1 is a base model with an engram table.

**P2** (`scripts/crowdgpt_t4/P2_GOAL.md`, runner `run_p2.py`) runs the paper's LoRA recipe on Qwen3-0.6B and Crowd-v1 with
paired seeds, and fixes the reading of all four outcomes in advance. **Not run: the T4 is reserved for the owner's
training this weekend.**

## 8. Improvement proposal

| # | Proposal | Evidence | Where |
|---|---|---|---|
| 1 | **Add code, math and reference text to the CrowdGPT data mix** (licenses to be checked; the maintainer decides) | P0 gaps +3.84 / +2.44 / +1.80 nat | Vxtzq/CrowdGPT#103 `docs/DATA_MIX_PROPOSAL.md` |
| 2 | **Track per-domain loss on every checkpoint** | one held-out number hid these gaps | CrowdGPT `tools/domain_eval.py` |
| 3 | **Keep the loss check engram-aware** | +0.35 nat without it | this PR, `crowdgpt_loss_check.py` |
| 4 | **Do not wire JEPA into the CrowdGPT client yet**; if used, λ = 0.1 + loss dropout 0.5 + stop-gradient | P1 and the screen | `crowd_jepa.py` defaults |
| 5 | **Run P2** when the T4 is free: paper recipe on Qwen3-0.6B and Crowd-v1 | answers the specificity question | `scripts/crowdgpt_t4/run_p2.py` |
| 6 | **LoRA r512** on a 24 GB GPU (community volunteer) | the paper's largest LoRA gain | not runnable on the T4 cap |
| 7 | **Other HiveMind models**: coder (docstring ↔ code), Bee (Q ↔ A), Granite grafts (statement ↔ proof); **gated on a P2 gain on Qwen** | — | `docs/JEPA_FOR_HIVEMIND_MODELS.md` |

## 9. Reproducibility

Scripts are in `scripts/crowdgpt_t4/`:

| Script | Experiment |
|---|---|
| `e0_heldout_sanity.py` | E0 |
| `p0_domains.py` | P0 |
| `night_jepa.py` + `run_queue.py` + `criteria.py` | P1 |
| `autoresearch/` | the screen |
| `hf_model.py` + `run_p2.py` | P2 |

All experiments run under `t4guard.py`. Data and weights are pinned by revision; experiment data lives on the data disk
under `crowdgpt_exp/`.
