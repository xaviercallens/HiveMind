# Night pilot P1: goal and rules, written before any result (2026-11-13, written 22:50 CET, queue started 22:55)

**Question.** Does LLM-JEPA give Crowd-v1 a gain on NL-RX-SYNTH comparable to the paper's? The paper reports
57.29 → 71.46 (+14.2 points) for Llama-3.2-1B-Instruct, full fine-tuning, λ = 1, k = 1, 4 epochs.

**Setup (fixed).**
- Model: Crowd-v1 rev `5ac3916`, engram on, frozen fp16.
- Adapter: LoRA r64 (alpha 64) on wq/wk/wv/wo/w1/w2/w3.
- Optimizer: AdamW, lr 2e-4, cosine with 3 % warmup; effective batch 16 (8 × 2); 3 epochs over the 8,000 training examples.
- Arms:
  - `sft`: CE on the answer given the prompt.
  - `jepa`: the same CE, plus λ = 1 × (1 − cos) between the last-token final hidden states of [prompt] and [answer], packed
    with a block-causal mask, k = 0.
- Seeds: 0, 1, 2. Order: sft0, jepa0, sft1, jepa1, sft2, jepa2.
- Test: the first 1,000 examples of `synth_test.jsonl` (fixed), greedy, exact string match (as in the reference
  evaluate.py).
- General-loss check: mean CE on the 12 "far" windows of E0. Base model: 3.194 nats.

**Deviations from the paper, stated up front.** LoRA instead of full fine-tuning, because 658M parameters with Adam does not
fit in the ~2.5–5 GB the T4 can give without touching inference. Also 3 epochs instead of 4, k = 0 instead of 1, a
1,000-example test subset, and a different model (Crowd-v1 is a base model, not Instruct).

**Criteria.**
1. **Floor.** If sft seed 0 scores < 5 %, the pilot is uninformative. Stop and do not interpret JEPA.
2. **Success** (JEPA retained). mean(jepa) − mean(sft) > 2 × pooled std over the seeds, **and** the far-loss of the JEPA
   runs is ≤ that of the SFT runs + 0.05 nat.
3. **"Paper-like gain" (the night's goal).** Success **and** mean(jepa) − mean(sft) ≥ +10 points.
4. With fewer than 3 seeds per arm at the deadline, report the paired differences per seed as "preliminary"; no verdict.

**Allowed adaptive step** (only if both arms of seeds 0–2 are done before 03:30): one extra JEPA arm with λ = 4, seed 0
(the paper's grid includes λ ∈ {0.5, 1, 2, 4}). It is exploratory and does not count toward criteria 2–3.

**Changes made 22:58–23:02, before any result existed; queue restarted 23:02** (from the pre-flight review workflow, 3 independent reviewers;
the aborted first sft_s0 attempt had no eval and was discarded):
- JEPA view 1 is the **bare query** (the reference's user message only), not the full instruction prompt. Before, the
  embedding sat on the same ':' template token for every example.
- The engram bigram input is reset at the start of view 2 in the packed pass, so view 2 is exactly a separate pass (no
  cross-view leak through the bigram).
- Eval decodes with `skip_special_tokens=True`, as in the reference evaluate.py.
- Checkpoint/resume on guard aborts (another tenant's ollama model loads and unloads on the T4); the 06:30 deadline is
  enforced while waiting for inference to go idle and in the far-loss check; the runner retries each run until done, and
  the floor gate is applied only after sft_s0 actually finishes.

**T4 rules.**
- Free VRAM ≥ 4,300 MiB at all times (t4guard).
- Pause while inference requests are in flight.
- No run starts if its estimated end is after **06:30 CET**, so the T4 is free for the weekend's intensive training
  (Nov 14–15). At 06:30 everything is stopped.
