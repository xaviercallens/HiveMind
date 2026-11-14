# P2: is the null result specific to CrowdGPT? (written 2026-11-14 10:45 CET, before any P2 run)

## Why P2

The paper's own LoRA table (arXiv 2509.14252 v2, Table 8, NL-RX-SYNTH, 5 runs per cell, lr 2e-5, λ = 1, k = 1) reports a
large JEPA gain **at rank 64**: SFT 21.09 ± 1.90 → JEPA 32.46 ± 1.26 (+11.4).

So LoRA rank does not explain our P1 null at r64. The remaining differences:
- the learning rate (ours 2e-4, paper 2e-5);
- k (ours 0, paper 1);
- epochs (ours 3; the paper's main runs use 4);
- the model: the paper uses Instruct models only. Crowd-v1 is a base model with an engram table.

## Design

The **paper's LoRA recipe** on two models, SFT vs JEPA, paired seeds:

| Setting | Value |
|---|---|
| LoRA | r64 (alpha 64) on all attention + MLP projections |
| Optimizer | lr **2e-5**, cosine with 3 % warmup |
| Batch | effective 16 (8 × 2) |
| Epochs | **4** |
| Data | NL-RX-SYNTH, 8,000 training examples; same raw prompt as P1; EOS 151643 |
| JEPA | λ = 1, **k = 1**. The predictor token is `<|box_start|>` (151648), absent from Crowd's data, appended after the query; the embedding is the predictor token's final state. View 2 = the regex; final-layer states; no stop-grad (as in the paper). |
| Eval | exact match on the first 1,000 test examples (as P1); general loss on the E0 "far" windows |

Models:
- **Crowd-v1** rev 5ac3916 + engram (frozen fp16), packed views with a block-causal mask (as P1).
- **Qwen3-0.6B** (the cached post-trained checkpoint, `Qwen/Qwen3-0.6B`, not `-Base`), fp16, PEFT LoRA. Views run as
  separate forward passes (the reference code's default mode); `last_hidden_state` (after the final norm).
  - This changes model strength **and** instruction tuning at once: P2 cannot separate those two.

Order: qwen sft s0, qwen jepa s0, crowd sft s0, crowd jepa s0, then seeds 1 and 2 in the same order while time allows.

## Reading the outcome

Fixed in advance, using seed 0 first and confirmed with 3 seeds where available. "Gain" means mean(JEPA) − mean(SFT) > 2 × pooled sd with ≥ 2 seeds; with 1 seed, the result is reported as an indication only.

| Qwen gains? | Crowd gains? | Interpretation |
|---|---|---|
| yes | yes | P1's null was our recipe (lr / k / epochs), not CrowdGPT. |
| yes | no | Specific to CrowdGPT (a base model, 3 nats, engram): JEPA needs a stronger or instruction-tuned base. |
| no | no | The paper's gain does not reproduce in this setup at all (scale, data budget, or implementation). Not CrowdGPT-specific. |
| no | yes | Unexpected; report as is. |

## T4 rules

- Free VRAM ≥ 4,300 MiB (t4guard), and a process cap of 2.7 GB.
- Pause during inference requests.
- **Yield to the weekend training.** Stop cleanly, with a checkpoint and no relaunch, if:
  - the governor leaves `held` or the `HOLD` file disappears; or
  - a GPU process that is not whitelisted takes more than 1 GB. Whitelisted: llama-server (gwaya), ollama llama-server,
    the 318 MiB venv python, my own runs.
- Hard stop at 23:00 CET today.
