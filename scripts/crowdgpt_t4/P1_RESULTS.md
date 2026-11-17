# P1 night pilot: results (criteria in ../../GOAL.md; computed by scripts/criteria.py → criteria.json)

Crowd-v1 rev `5ac3916` + engram (frozen), LoRA r64, NL-RX-SYNTH, 3 epochs, 1,000 test examples, exact match.
Queue 23:02 → 02:54 CET on the shared T4. The T4 was free from 02:54. Free VRAM stayed ≥ 4,300 MiB: two guard aborts
during sft_s0, both resumed from a checkpoint.

| seed | SFT acc % | JEPA (λ=1) acc % | paired Δ | SFT far_loss | JEPA far_loss |
|---|---|---|---|---|---|
| 0 | 16.0 | 15.7 | −0.3 | 3.789 | 4.605 |
| 1 | 17.7 | 17.7 | 0.0 | 3.719 | 4.652 |
| 2 | 17.5 | 17.3 | −0.2 | 3.830 | 4.770 |
| **mean ± sd** | **17.07 ± 0.93** | **16.90 ± 1.06** | **−0.17** | **3.779** | **4.676 (+0.90)** |

Base model far_loss: 3.194.

**Criteria (as written in GOAL.md):**
1. **Floor: passed** (sft_s0 = 16.0 % ≥ 5 %).
2. **Success: not met.** Δ = −0.17 points against a threshold of > 2 × pooled sd = 1.99. The far-loss condition also fails
   (JEPA +0.90 nat vs SFT; the limit is +0.05).
3. **Paper-like gain (≥ +10 points): not met.** The paper reports +14.2 for Llama-3.2-1B-Instruct, full fine-tuning.
4. 3 seeds per arm completed, so the verdict above is final for this setup.

**Exploratory, not part of the criteria:**
- **λ = 4, seed 0:** 15.4 % (prefix 18.3 %), far_loss 5.012. Stronger λ is worse on both counts.
- **Prefix match** (secondary metric, seeds 1–2 only): SFT 20.6 / 19.4, JEPA 19.5 / 19.8. No difference either.
- **The JEPA term collapses.** 1 − cos is about 0.15 at step 25 (0.68 at step 3 in a smoke test) and 0.0004–0.0005 by the end of every JEPA run. Query
  and answer embeddings become nearly identical, while the general loss gets 0.9 nat worse than plain SFT. That pattern
  fits a shortcut: the LoRA aligns the last-token states along a shared direction instead of learning the query → regex
  mapping. Not verified: measuring the cosine between *different* examples' embeddings would settle it.
- **Cost:** JEPA training takes about 1.8× SFT (2,174–2,408 s vs 1,069–1,241 s), consistent with the paper's ~2×.

**What this does and does not show.** In this setup LLM-JEPA brings no gain, and it degrades general language modeling.
The setup differs from the paper on the points declared in GOAL.md:
- LoRA instead of full fine-tuning;
- a base model at 3 nats, not an Instruct model;
- k = 0;
- 3 epochs;
- a 1,000-example test subset.

The paper reports its large LoRA gain at rank 512, not 64. So this does **not** refute the paper. It says the method does
not carry over to Crowd-v1 + LoRA r64 as is.
