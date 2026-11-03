# Real-model MVP smoke test

> **Proof of concept (PoC):** HiveMind is not a production service yet. Send only public, non-sensitive data.

`scripts/mvp_e2e_smoke.py` is the smallest end-to-end proof that the core idea works with a **real model and no mocks**:

```
real GGUF model (llama.cpp server, CPU)  <-  real producer shield (HTTP, endpoint token)
    <-  real gateway (escrow hold, dispatch, metering, settlement, hash-chained ledger)  <-  a consumer request
```

It runs on **CPU** (`-ngl 0`), so it uses no GPU memory and never disturbs a live producer (the T4 keeps at least 4,300 MiB free).

## Run it

```bash
python scripts/mvp_e2e_smoke.py \
  --gguf /path/to/qwen2.5-coder-1.5b.gguf \
  --llama-server /path/to/llama.cpp/build/bin/llama-server \
  --report mvp_smoke_report.json
```

The GGUF file of any model Ollama has pulled is in its blob store (`ollama show <tag> --modelfile` prints the path on the `FROM` line).
The pytest wrapper `tests/test_mvp_e2e_smoke.py` runs the same thing when `HIVEMIND_MVP_GGUF` (and `HIVEMIND_LLAMA_SERVER`) are set, and is skipped otherwise.

## What it checks (15 checks, exit code 0 only if all pass)

Model server starts; the shield starts, refuses requests without the endpoint token and serves the real model with it; a producer registers and
heartbeats; two different prompts return HTTP 200 with non-empty answers that **differ from each other** (not a constant) and the code prompt is
answered with code; the consumer is debited and the producer credited less than the debit (the fee); **no HMT is created or lost**; no escrow hold
is left open; the hash-chained ledger verifies; a model nobody serves is refused **without charge**.

## Measured result (2026-11-03, Tesla T4 VM, CPU only, 8 vCPU)

Model: Qwen2.5-Coder 1.5B (Ollama blob, Q4 GGUF), llama.cpp server 0.7.0-dev.

| request | tokens (prompt + completion) | wall time |
| :-- | :-- | :-- |
| Python `is_palindrome` | 46 + 58 | 2.4 s |
| "what does HTTP 404 mean" | 30 + 21 | 1.2 s |

Ledger: consumer spent 0.0383 HMT, producer earned 0.0326 HMT, all 15 checks passed.

## What this does and does not show

It shows that the metered request path (escrow, dispatch to a producer behind the shield, metering, settlement, ledger) works end to end with a real
model. It does **not** show that producers are honest (their output is still unverified), that the models are trained by HiveMind (they are
off-the-shelf), or anything about throughput at scale. See `SECURITY.md` and the catalog's `served_by` fields.

## CPU base tier: `hivemind/granite-1b-base` (2026-11-03)

The same 15 checks pass for the plain IBM Granite 3.1 1B-A400M **base** model (Q4_K_M, about 0.8 GB, about 1.2 GB RAM, about 68 tokens/s on 6 CPU threads,
0 MiB of GPU memory): `python scripts/mvp_e2e_smoke.py --gguf base-q4km.gguf --tier granite-1b-base --chat-template-file deploy/granite_cpu/completion.jinja --prompt "..."`.
Requests took 0.95 s and 0.78 s (48 tokens each); the consumer paid 0.0274 HMT and the producer earned 0.0233 HMT.
A base model cannot chat: through its default template it echoed the prompt as raw chat markup, so `deploy/granite_cpu/completion.jinja` passes the user text through and
the model continues it. Quality was not measured; it was visibly weak (wrong third planet, repetition). Serve it with `deploy/granite_cpu/start.sh`.
