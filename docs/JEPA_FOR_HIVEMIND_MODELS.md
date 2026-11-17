# LLM-JEPA for other HiveMind models: proposal and gate (2026-11-14)

Status: **proposal, gated on P2.** Nothing is wired into production. The BOINC QLoRA workunits, the governor and the live
services are unchanged.

## What we know so far (measured on the T4)

Every Crowd-v1 result below uses LoRA r64 on NL-RX-SYNTH.

| Experiment | Result |
|---|---|
| **P1**: Crowd-v1, paper λ = 1, k = 0, lr 2e-4, 3 seeds | **No gain**: 16.90 vs 17.07 %. General loss +0.90 nat (the JEPA term collapses). |
| **autoresearch**: 5 of 10 hypotheses, 600 s budget | No variant beats SFT at equal compute. Stop-grad, centering and InfoNCE remove the collapse. λ 0.1 + loss dropout + stop-grad is the best variant that passes the guard rail. |
| **Paper, Table 8** (Llama-3.2-1B-*Instruct*, LoRA r64, lr 2e-5, λ = 1, k = 1) | **+11.4 points** (21.09 → 32.46). |

So the open question is whether the gap comes from the **model** (Crowd-v1 is a base model) or from **our recipe**
(lr 10× higher, k = 0, 3 epochs). **P2** (`scripts/crowdgpt_t4/P2_GOAL.md`) answers it: the paper's exact LoRA recipe
on Qwen3-0.6B and Crowd-v1, SFT vs JEPA, paired seeds. It is ready to run; the T4 is in use for the weekend training.

## Gate

Fixed now, before P2 results:
- **If P2 shows a JEPA gain on Qwen3-0.6B** (> 2 × pooled sd), adapt it to the models below, one at a time, each with its
  own SFT-vs-JEPA comparison.
- **If not,** stop here. No model below gets JEPA.

## Candidate models and natural views

| Model (HiveMind) | Natural pair of views | Fits where | Notes |
|---|---|---|---|
| `hive-bee-stars-3b` (Bee expert) | question ↔ answer of its domain set | ≥ 16 GB GPU (QLoRA) | not on the T4 next to inference |
| `Qwen2.5-Coder-7B` (coder tiers) | docstring / NL spec ↔ function body; test ↔ implementation | ≥ 24 GB GPU (QLoRA) | code has the most natural views |
| Granite 3.1 MoE QLoRA domain grafts (22 domains) | per-domain Q ↔ A, statement ↔ proof (Lean) | BOINC workunits | needs paired views in the workunit spec (below) |
| `Vxtzq/Crowd-v1` | regex/SQL/code pairs at SFT time | T4 (LoRA r64, ≤ 2.7 GB) | P1 + screen done; P2 pending |

## Recommended settings (from our measurements, to be re-checked by P2)

| Setting | Recommendation |
|---|---|
| Start from | the paper's LoRA recipe: lr 2e-5, λ = 1, k = 1 predictor token |
| If the JEPA term collapses (1 − cos → 0) or general loss rises | λ 0.1 + loss dropout 0.5 + **stop-gradient on the target view**. This kept general loss at the SFT level in our screen. |
| Measure | the task metric **and** a general held-out loss (guard rail ≤ SFT + 0.05 nat), at equal compute |
| Code | `crowd_jepa.py` (Vxtzq/CrowdGPT#103): model-agnostic, takes final hidden states; options `stopgrad`, `center`, deterministic loss dropout (`jepa_step`). `scripts/crowdgpt_t4/hf_model.py` shows the HF/PEFT path (separate view passes). |

## What BOINC workunits would need (design only)

`src/hivemind/boinc/real_client/moe_app.py` trains by deterministic segment replay on a raw token stream
(`replay_inner_segment`), so validators can recompute a segment. JEPA would need:

1. **Paired views in the spec.** For example `jepa_pairs: [[query_ids, answer_ids], ...]` and
   `jepa: {lam, k, pred_id, stopgrad, dropout}`, part of the signed workunit.
2. **Determinism.** The JEPA term must be inside `replay_inner_segment`. The loss-dropout decision must be a hash of
   (seed, step), never a local RNG, so a validator replays the same loss.
3. **Validation unchanged.** The held-out `eval_by_domain` (next-token loss) stays the acceptance metric; JEPA only
   changes training.

None of this should be built before the gate opens.
