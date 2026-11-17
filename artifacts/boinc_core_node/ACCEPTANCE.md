# Acceptance criteria — BOINC as a HiveMind core node (QLoRA MoE training + inference), written before the first run

Builds on `artifacts/boinc_real/ACCEPTANCE.md` (PR #178). Same three honest tiers (unit tests / standalone app run /
real stock client end-to-end). Model: `ibm-granite/granite-3.1-1b-a400m-base` (a 1.3 B-parameter MoE, cached locally,
`HF_HUB_OFFLINE=1`). Environment: isolated venv with `peft 0.21.2` + `bitsandbytes 0.50.2`; the shared site-packages
are untouched.

**What "QLoRA" means here, stated up front.** Attention projections (`q,k,v,o`) are quantised to NF4 (double
quantisation, fp16 compute) by bitsandbytes and carry fp32 LoRA adapters (r=8). Granite's experts are not
`nn.Linear`, so they stay **frozen fp16, not NF4**. This is QLoRA on the attention path of a MoE, not on the experts.

**Work-unit semantics, declared.** A workunit is `K` segments of `S` inner steps; AdamW is re-initialised at every
segment start (this is what `replay_inner_segment` implements). It makes checkpoint/resume and spot-check replay exact
by construction; it is not the same trajectory as one continuous AdamW run.

| ID | Tier | Criterion (all must hold for PASS) |
| --- | --- | --- |
| B1 | a | A stock BOINC 7.18.1 client downloads a signed token-shard workunit, runs the QLoRA MoE app on the T4, and uploads **two** output files (`out.json` + `out.pt`, tens of MB). |
| B2 | a | The validator replays a randomly chosen segment from the claimed start state on the T4 and finds max abs divergence `<= 1e-4` (expected 0.0, same class), and the claimed start state equals the deterministic LoRA init. |
| B3 | a | The validated result earns cobblestones through `verified_credit`; the adapter is assimilated (stored with its sha256). |
| B4 | a | Red-team control: a "lazy" volunteer that returns the untrained adapter as if trained is rejected with 0 credit. |
| B5 | a | Preemption: SIGKILL the client mid-WU, restart: resumes from a segment boundary (`resumed_from_segment > 0`) and the final adapter sha256 equals an uninterrupted run's. |
| B6 | b | A batch **inference** workunit (prompts in, greedy completions out, with the assimilated adapter) runs on the stock client; the validator's GPU replay gives **identical token ids**. |
| B7 | c | A loopback-only OpenAI-compatible endpoint serves `/v1/chat/completions` with the assimilated adapter; its greedy output equals the B6 batch output for the same prompt. |
| B8 | all | The GPU is free of BOINC/app processes afterwards; no system service installed; every listening socket is on 127.0.0.1. |

Informational only (reported, **not** a pass criterion, no claim of improvement is made): held-out eval loss before and
after the workunit, wall time per segment, peak VRAM.

Any failed row is reported as failed, with the log.
