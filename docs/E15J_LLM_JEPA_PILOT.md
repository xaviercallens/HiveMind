# E15-J: an LLM-JEPA term on GWAYA-gated fine-tuning (pilot plan)

**Status: draft plan, not frozen, not run.** The preregistration draft is
[`artifacts/experiments/E15J_llm_jepa_prereg_draft.json`](../artifacts/experiments/E15J_llm_jepa_prereg_draft.json).
It becomes a registration only when the owner freezes it. To freeze it:
1. Set `registered_utc` and `registered_before_run: true`.
2. Commit the file.
3. Record its sha256 as `design_hash` before any real-model stage runs.

No pipeline code exists for E15-J yet.

## Why this experiment

E15 (`HM-S10-11`, paper §E15) fine-tuned `Qwen2.5-Coder-1.5B-Instruct` with a rank-16 LoRA, trained on solutions that the GWAYA-style gate had accepted. It found two things:

- **The inference loop works.** C1 − C0 = +7.0 points, 19 gained / 1 lost, p = 4.0e-05.
- **Fine-tuning did not.** C2 − C0 = +0.8 points, p = 0.85.

LLM-JEPA (Huang, LeCun & Balestriero, [arXiv:2509.14252](https://arxiv.org/abs/2509.14252)) claims to fix this weak fine-tuning step for exactly this kind of data. It keeps the next-token loss and adds a term that aligns the final-layer embedding of a natural-language view with the embedding of its formal counterpart.

Its largest reported gains are natural language → regex:
- Llama-3.2-1B, full fine-tuning: 57.29 → 71.46.
- LoRA: 50.18 → 72.41.

On other tasks it gains about one point. The paper has no code-synthesis benchmark.

E15 is a near-exact match to the paper's setting: about 1.5B parameters, LoRA, problem ↔ verified solution pairs. It already has controls, receipts and a measured cost. That makes it the cheapest clean test of the claim available to the project.

## What the term is (from the reference code, not only the paper)

The reference implementation is `github.com/galilai-group/llm-jepa`, Apache-2.0, pinned at commit `ea0017c`. It was read and not run.

```
total = γ·L_LM + λ·(1 − mean cos(e_text, e_code))
```

- `e_view` is the final-layer hidden state at the last non-padding token of that view, run on its own.
- There is **no stop-gradient**: both views receive gradients. Only the language-modelling term prevents the embeddings from collapsing; the paper's γ = 0 ablation gives empty outputs.
- The predictor is k new special tokens `<|predictor_i|>`, appended to the user message.
- The default path batches three sequences per example: the full conversation, the text view and the code view.

E15-J re-implements this inside the E15 pipeline:
- The LoRA stays hand-rolled.
- One new token row is trainable.
- Every other embedding row stays frozen.

## Design in one table

| | |
|---|---|
| Fixed in advance | λ = 1, k = 1, γ = 1, JEPA on every batch. This is the paper's headline NL-RX setting. There is no configuration search, so nothing is tuned on the evaluation set. This avoids the paper's own weakness: its (λ, k) appear to be chosen by test accuracy. |
| Data | The E15 train pool (462 problems). The rejection-sampled set is regenerated **once** with E15's code and hashed, and that file trains every arm. It is not claimed to equal E15's 673 pairs. |
| Arms × seeds | **A**: plain LoRA, the E15 recipe. **J**: A plus the JEPA term. **S**: J with problem↔solution pairs shuffled by a fixed derangement. Seeds 1, 2, 3 for each arm. |
| Primary test (H15Ja) | Mean pass@1(J) − pass@1(A) over seeds on MBPP-sanitized (n = 257), with an exact sign test on per-problem solve counts. Supported iff the difference is ≥ +3.0 points and p < 0.05. |
| Specificity (H15Jb) | J against S. If S matches J, any gain is generic regularisation, not the pairing the method relies on. |
| Controls | All E15 controls are re-run. Positive control: J's text→code least-squares residual must drop below A's (paper Table 14 style). Plumbing control: a CPU unit test checks that λ = 0 reproduces the A path. The evaluation set is used for nothing else. |
| Memory | E15's 7.5 GiB per-process cap is kept. A pre-run `nvidia-smi` gate checks that other processes use ≤ 3,380 MiB, so the T4 keeps ≥ 4,300 MiB free (AGENTS.md). |
| Cost | About 5–6 T4-hours in the Spot training window. This is a planning estimate from E15's measured stage times, to be replaced by measured values. |

## What could go wrong, stated before the run

- **Power.** E15's comparison had 28 discordant problems out of 257, and +3 points is about 8 net problems. Three seeds help, but an inconclusive result is a real possibility, and it will be reported as such.
- **Transfer.** Python synthesis is a harder target than regex. The paper's +14 points may not transfer.
- **Compute.** The JEPA arms cost about 2–3× the plain training. The comparison is equal-epoch, as in the paper, and the extra compute is reported rather than equalised.
- **Serving.** Gains would be measured on fp16 + LoRA. Nothing is claimed for GGUF/Ollama serving, where E17 already notes an nf4/Q4 base mismatch.

## Decisions the result triggers

| Outcome | Consequence |
|---|---|
| H15Ja and H15Jb supported | The term becomes an option in the coding-curriculum recipe (`HM-S10-06`), and E17 (7B QLoRA) gets a JEPA arm. |
| H15Ja supported, H15Jb not | Reported as regularisation. Not adopted without a comparison against a cheaper regulariser. |
| Inconclusive or refuted | The inference-time loop remains the efficient path, as after E15. LLM-JEPA is not pursued for code at this scale. |

## Not in scope

The router (keyword intent counts and producer rotation), the volunteer swarm (inference only) and the Cloud Run services hold no trained text embeddings, so the term has nothing to act on there.
