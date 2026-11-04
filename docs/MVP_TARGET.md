# HiveMind MVP target

> **Proof of concept (PoC):** not a production service. Send only public, non-sensitive data.

**One sentence.** End users lend consumer GPUs (through a BOINC client and Ollama/llama.cpp, behind a security shield); a low-cost GCP control plane
orchestrates them; together they (1) **serve a Mixture-of-Experts model with inference distributed across several consumer GPUs** and (2) **train
MoE LoRA experts with an OpenDiLoCo-style recipe (QLoRA on custom datasets)**, coordinating a swarm of MoE experts over **AMCP on Kafka**.

**Rules.** Every claim has a check that a machine can run and an artifact that records the result. A criterion is *met* only when its check passes on
real models and real hardware (not mocks, not simulated numbers). Negative results are recorded as such (the train-to-serve gate on `hmq_00147` is the model:
`docs/TRAIN_TO_SERVE_GATE.md`). Scorecard: `python scripts/mvp_scorecard.py` (the loop's measure; see "The loop").

Background: DiLoCo (Douillard et al. 2023) runs H local AdamW steps per worker, then one outer Nesterov-SGD step on the averaged pseudo-gradient;
OpenDiLoCo (PrimeIntellect-ai/OpenDiLoCo, built on `hivemind`; its README gives inner AdamW lr 4e-4, H = 50 to 500 local steps, outer Nesterov SGD lr 0.7 momentum 0.9, and
the project is no longer maintained) ran DiLoCo across machines over a DHT. Its README does not state compression or utilisation figures; check the paper (arXiv:2407.07852)
before quoting any. HiveMind's version trains LoRA adapters (QLoRA, NF4 base) instead of full weights, so the exchanged tensor is megabytes, not gigabytes.

## Capabilities and acceptance checks

| ID | Capability | Acceptance check (all must hold) | Evidence artifact |
| :-- | :-- | :-- | :-- |
| **A1** | Request-level distribution of MoE inference over >= 2 consumer GPUs | Two producers on two different physical GPUs serve the same MoE tier; the gateway splits >= 20 concurrent requests between them, survives killing one producer (no failed request after the failover window), and bills correctly | `artifacts/mvp/A1_multi_gpu_serving.json` |
| **A2** | Model-level split of one MoE across 2 GPUs/hosts | One MoE GGUF served by llama.cpp with layers/experts split over two hosts (RPC), output token-identical to the single-host greedy output on 20 prompts; tokens/s for both setups recorded | `artifacts/mvp/A2_split_moe.json` |
| **B1** | Custom dataset in, tokenized shards out | `python scripts/make_custom_shards.py --jsonl my.jsonl` produces signed shards and a held-out split in the BOINC format; no data leaves the machine | `artifacts/mvp/B1_custom_shards.json` |
| **B2** | OpenDiLoCo-style QLoRA round across >= 2 real workers | Each worker runs H inner steps (QLoRA, NF4 base), sends a compressed LoRA delta, the coordinator applies an outer Nesterov step with Multi-Krum; >= 5 rounds on 2 physical hosts; deltas and sync cost recorded | `artifacts/mvp/B2_diloco_qlora.json` |
| **B3** | Training improves the model on the served files | The train-to-serve gate (`scripts/adapter_to_served.py`) passes on the final adapter: pooled held-out gain > 0, bootstrap 95% interval excludes 0, no domain worse than +1% | `artifacts/experiments/GRANITE_ADAPTER_*_gate_*.json` |
| **C1** | MoE swarm coordination over AMCP on Kafka | Task, result and delta events flow through a 3-voter KRaft cluster; killing one voter loses no committed event; a consumer group resumes from its offset | `artifacts/mvp/C1_kafka_swarm.json` |
| **C2** | Expert registry and routing | A trained adapter registered in the expert registry is selected by the router for its domain and served; unregistered or failed-gate adapters are refused | `artifacts/mvp/C2_expert_routing.json` |
| **D1** | Low-cost GCP control plane | Cloud Run flags equal the AGENTS.md rules; no idle TPU; no paid static IP for internal nodes; a monthly cost estimate from the deployed settings is recorded and stays below the owner's budget (to be set) | `artifacts/mvp/D1_gcp_cost.json` |
| **E1** | End-user onboarding (BOINC + Ollama/llama.cpp) | A fresh machine reaches "serving or training" with one documented command; preflight checks VRAM, drivers, free disk; the client stops training within seconds when the owner needs the GPU (governor) | `artifacts/mvp/E1_onboarding.json` |
| **E2** | Security of volunteer GPUs and of results | Shield refuses unauthenticated and off-allowlist requests; BOINC results are validated (replay spot-check + Multi-Krum); a poisoned delta and a lazy worker are rejected in a live test; producer usage is cross-checked against a gateway-side token count; SSRF and key handling tests pass | `artifacts/mvp/E2_security.json` |
| **F1** | Honest public numbers | Public pages show only figures present in `measured_results.json`; a CI test fails on a literal number without an artifact | tests |

## Status at the start (2026-11-04)

Met or partly met: A1 (request routing, market, escrow: real; 2 physical GPUs only recently; failover not measured), C1 (Kafka cluster built and failure-tested on VMs,
reported earlier), E2 (shield, validator, spot-check, Multi-Krum exist and have tests), F1 (benchmarks page measured-only). Not met: A2, B1, B2 (the BOINC chain
trains on one host; two-host DiLoCo exists only on a toy LM), B3 (the current adapter fails the gate), C2, D1 (no recorded cost estimate), E1.

## Loop log

| Iteration | Scorecard | What changed |
| :-- | :-- | :-- |
| 0 (2026-11-04) | 0/11 | target + scorecard written; 13 low-tier agents reviewed every criterion (`MVP review` workflow) and produced the ranked backlog |
| 1 | 1/11 (B1 pass; B3, D1, E2, F1 fail with reasons; A1, A2, B2, C1, C2, E1 not run) | W1 figure ratchet (48 typed-in page figures recorded, may only fall); W2 gate unit tests + selection-independent re-check (hmq_00147 still fails); W3 gated expert promotion (C2 code-ready, no router link yet); W7 endpoint cooldown + concurrent kill test (A1 code-ready, needs two physical GPUs); W5 custom shards (B1 pass on a real run); W4 E2 evaluation (live lazy-worker rejection and honest verification, offline Multi-Krum; poisoned-delta-live and tight billing check still open); stratified `--mix` rounds (pre-registered experiment, not run) |

| 2 | 3/11 (A1, A2, B1 pass; F1 waived) | A2: one MoE split across two hosts, 20/20 token-identical with `--no-repack` (CPU on both; 8/20 with default kernels). A1 on the production gateway, T4 + RTX 2070: 30/30 after the T4's shield was stopped, billing exact. **A1 limits:** before the kill the T4 answered only 3 of 30 (overflow past the 2070's 6 concurrent), and 8 of 30 got 502 at 10 concurrent requests; the two GPUs serve different models under `granite-moe` (2070 `granite3.1-moe:3b`, T4 the 1B-A400M base with a self-trained adapter). It shows failover and overflow, not balanced load on one model. First two runs failed and show why: `git log artifacts/mvp/A1_multi_gpu_serving.json` |

Open items and what each needs: A1 real run = two physical GPUs + permission to kill one producer; A2 = second host + RPC-enabled llama.cpp; B2 = two hosts running QLoRA rounds
(the T4 must keep 4,300 MiB free, so training there is governor-gated); B3 = the `--mix` experiment deployed by the owner; C1 = Kafka VMs; D1 = owner decision (scale-to-zero vs the AGENTS.md
single-writer rule) and a budget; E1 = a fresh volunteer machine; F1 = owner decision on which of the 48 page figures to keep.

## The loop

```
repeat:
  1. python scripts/mvp_scorecard.py          # deterministic checks per criterion -> docs/MVP_SCORECARD.md (pass / fail / not-run + reason)
  2. workflow: low-tier agents review each failing criterion against the code and propose the smallest next change (read-only)
  3. implement the top change on a branch, with a test; run the touched suites on Python 3.11
  4. re-run the scorecard; record the delta; stop when all rows pass or a row needs hardware/owner input (then say exactly what)
```

Hardware used: Tesla T4 VM (keeps >= 4,300 MiB VRAM free per AGENTS.md), RTX 2070 host (owner-operated), 3 Spot Kafka voters on GCP, Cloud Run control plane.
Out of scope for the MVP: cash value for points, production SLAs, private-data workloads.
