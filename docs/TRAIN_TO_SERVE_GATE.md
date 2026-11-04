# Train-to-serve gate (BOINC adapter -> GGUF -> measured -> served only if it wins)

> **Proof of concept (PoC):** not a production service. Send only public, non-sensitive data.

`scripts/adapter_to_served.py` closes the loop between the BOINC QLoRA chain (`granite-3.1-1b-a400m-base`, rank-8 LoRA on q/k/v/o) and serving:

1. `prepare`: read the validated adapter from the ledger (read-only, hash-checked against `latest_adapter.json`), write a PEFT directory, convert the base and the LoRA to GGUF with llama.cpp.
2. `evaluate`: run `llama-perplexity` on held-out text (the chain's own never-trained rows, 6 domains x 64 chunks x 128 tokens) for the base GGUF and the base + LoRA GGUF, per chunk, on the files that would be served.

**Gate (fixed before running):** pooled mean NLL gain > 0, paired bootstrap 95% interval of the gain excludes 0, and no domain more than 1% worse. Only then may a tier be built from the adapter.

## Result for `hmq_00147` (best of 161 validated rounds; chain loss 2.2796 -> 2.1214)

| Served format | Pooled gain (nats/token) | 95% bootstrap | Worst domain | Servable |
| :-- | :-- | :-- | :-- | :-- |
| f16 base + f16 LoRA (primary) | -0.0017 | [-0.0144, +0.0120] | +3.7% (french; edu_web +3.6%) | **no** |
| Q4_K_M base + f16 LoRA (post-hoc, exploratory) | +0.0040 | [-0.0089, +0.0178] | +3.3% (french; edu_web +3.3%) | **no** |

Per domain the adapter helps math (about -9%) and biomed (about -3%) and is flat or worse on general, french, edu_web and news. The conversion works (the adapter clearly changes the output); the improvement the chain reports does not carry over to the served model on this evaluation. Possible contributors, not tested separately: the chain trains on an NF4 base, and each round is about 80 steps on one domain, with a plateau since round 147.

**Consequence:** no adapter-based tier is added to the catalog and nothing is deployed from this adapter. The plain base model (no adapter) can still be served as a base tier.
Raw results: `artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16.json` and `..._gate_q4km_posthoc.json`.
