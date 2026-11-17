# S21: upgrading the core models (base-model catalog, VRAM estimate, promotion gate)

Module: `src/hivemind/core/base_model_catalog.py`. Catalog: `configs/base_models_s21.json`. Refresh: `scripts/refresh_base_model_catalog.py`.
Tests: `tests/test_s21_base_model_catalog.py`.

## Why

The incumbent base for coding is `Qwen/Qwen2.5-Coder-7B-Instruct`. Season 19 (`docs/QWEN35_CODING_AND_IDLE_MOE_PLAN.md`,
`qwen35_usage_moe_orchestrator.QWEN35_HARDWARE_TIERS`) planned an upgrade to Qwen 3.5 backbones, but its parameter counts and VRAM figures were typed
by hand, and three of its model ids are not served by the Hugging Face Hub (see below). This module replaces hand-typed figures with fetched ones,
gives every tier a VRAM *estimate* with a stated margin, and makes a switch of base model conditional on a paired statistical test.

## 1. Catalog (fetched, never typed)

`scripts/refresh_base_model_catalog.py` reads, for each requested id:

- `https://huggingface.co/api/models/<id>?blobs=true`: revision sha, last modification, license (`cardData.license`, else the `license:` tag),
  gating, pipeline tag, the Hub's own safetensors parameter total and per-file sizes;
- `config.json` at that exact revision: text-model geometry (`text_config` for the multimodal Qwen 3.5 checkpoints);
- the safetensors headers by HTTP range requests (no weights downloaded): parameters split into text linear weights, embeddings / lm_head,
  other text tensors (norms, convolutions), vision tower and multi-token-prediction (MTP) layers, plus the LoRA geometry (sum of in + out over the
  text linear layers).

`validate_catalog` refuses a catalog whose header total differs from the Hub's safetensors total, whose revision is not a 40-hex sha, or which lacks a
license, a retrieval date or an incumbent. A model id the Hub does not serve goes to `not_found` and is left out of `models`.

Fetch run on 2026-10-28 (`python3 scripts/refresh_base_model_catalog.py --plan`, retrieved_at `2026-10-28T07:49:30Z`); every value below is
copied from `configs/base_models_s21.json`:

| Model | Role | Revision | License | Parameters (Hub total = header total) | Vision + MTP params (header) |
|---|---|---|---|---:|---:|
| Qwen/Qwen2.5-Coder-7B-Instruct | incumbent | `c03e6d35820…` | apache-2.0 | 7,615,616,512 | 0 |
| Qwen/Qwen3.5-0.8B | candidate | `2fc06364715…` | apache-2.0 | 873,438,784 | 121,045,760 |
| Qwen/Qwen3.5-4B | candidate | `851bf6e806e…` | apache-2.0 | 4,659,865,088 | 454,113,792 |
| Qwen/Qwen3.5-9B | candidate | `c2022362357…` | apache-2.0 | 9,653,104,368 | 699,301,104 |
| Qwen/Qwen2.5-Coder-14B-Instruct | reference | `aedcc2d42b6…` | apache-2.0 | 14,770,033,664 | 0 |
| Qwen/Qwen3.5-9B-Base | named in S19 | `68c46c4b349…` | apache-2.0 | 9,653,104,368 | 699,301,104 |

**Not served by the Hub** (HTTP 401 on `/api/models/<id>`, which the Hub returns for ids that do not exist or are private): `Qwen/Qwen3.5-Coder-3B`,
`Qwen/Qwen3.5-Coder-14B-Instruct`, `Qwen/Qwen3.5-Coder-30B-A3B-MoE`. All three are named as backbones in `QWEN35_HARDWARE_TIERS`. They are not in the
catalog and the plan never recommends them. Correcting the S19 tier table is left to a follow-up (out of this PR's scope).

The Qwen 3.5 checkpoints are multimodal (`image-text-to-text`) and hybrid: 3 of every 4 layers use linear attention (`layer_types` in the config).

## 2. VRAM estimator (an ESTIMATE)

All outputs carry `kind = "ESTIMATE"`. Nothing ran on a GPU.

**NF4 QLoRA training** (`estimate_qlora_training`): text linear weights at 0.53 byte/param (the constant of `scripts/t4_model_candidates.py`, whose
header prediction for Qwen3-4B was 2,579 MiB against 2,578 MiB measured, per `docs/T4_NIGHT_MODEL_DECISION.md`); embeddings and other text tensors at
2 bytes; vision + MTP at 2 bytes when `load_aux_towers` (default on, conservative); LoRA weights + gradients in fp32 plus the optimizer state (AdamW
8 bytes/param); activations and logits with the constants of `scripts/qlora_vram_estimate.py`, with or without gradient checkpointing; one
de-quantisation buffer; and a 307 MiB CUDA-context allowance (3,716 process peak minus 3,409 allocated peak, both measured on the T4). The activation
scale is uncalibrated (1.0) in this module; `scripts/qlora_vram_estimate.py` fits one when its calibration models are in the local cache.
Known approximation: the activation formula is the dense-transformer one, applied unchanged to the linear-attention layers of Qwen 3.5.

**GGUF serving** (`estimate_gguf_serving`): bytes per weight derived from the ggml block layout (Q4_0: 18 bytes per 32 weights; Q8_0: 34 per 32),
fp32 norms, fp16 KV cache for the full-attention layers only, fp32 recurrent + convolution state for linear-attention layers, a compute buffer sized
for a 512-token micro-batch, and the same context allowance. The vision projector is not counted (served separately, if at all).

**Verdict** (`fit_verdict`): `FITS` when estimate x (1 + margin) <= budget, `MARGINAL` when only the bare estimate fits, else `DOES_NOT_FIT`. The
default margin is 20 %, the floor `scripts/qlora_vram_estimate.py` uses.

**Calibration** (`calibrate_from_measurement`, `append_calibration`): records a real measured peak (`process_peak`, or `torch_max_allocated`, which is
compared against the estimate minus the context allowance), the GPU, the source of the number, the ratio measured / estimate, and whether the
measurement stayed within the margin. **No calibration point is recorded in this PR:** no GPU was used.

## 3. Promotion gate

`promotion_gate(candidate, incumbent, guard_candidate, guard_incumbent, policy)` takes per-item booleans on the SAME held-out items (and on a separate
guard pack). It returns `PROMOTE` only when all hold:

1. both packs pair exactly and have at least `min_items` (50) / `min_guard_items` (20) items (else `INSUFFICIENT`);
2. exact two-sided McNemar p < `alpha` (0.05) on the held-out pack (binomial on the discordant pairs; same value as
   `swarm.verified_update_gate.exact_paired_p`, which a property test checks);
3. candidate wins > losses on the held-out pack;
4. on the guard pack, losses - wins <= `tolerated_guard_net_loss` (0) and no one-sided significant regression.

Otherwise `HOLD`. Controls in the tests: identical outcomes (any pattern, property-tested) never promote; a candidate with 30 wins / 0 losses on 100
items promotes; the same candidate with one guard-pack loss is held; a worse candidate is held.

## 4. Upgrade plan per GPU tier (derived by code)

`recommend_upgrade_plan(catalog)` evaluates every incumbent / candidate / reference model on a training ladder (seq 512 mb 2 → seq 512 mb 1 → + gradient
checkpointing → seq 256; rank 16 on all text linear layers, the order of the T4 decision doc's plan) and a serving ladder (Q8_0 → Q4_0 at 4,096
context). Per tier it nominates the largest *candidate* whose training fits with the margin. A nomination means "run the promotion gate", never "switch".

Tier budgets: `t4_serving_resident` 4,300 MiB (governor `--min-free-mib 4300`), `std_8gb` 6,656, `big_12gb` 11,264, `xl_16gb_plus` 22,528 (the
`max_wddm_safe_mib` values of `QWEN35_HARDWARE_TIERS`, asserted equal by a test), `t4_serving_stopped` 15,360 (T4 total).

Output of `PYTHONPATH=src python -m hivemind.core.base_model_catalog` on the committed catalog (2026-10-28), estimates in MiB before the margin:

| Tier | Budget | Train candidate (first fitting recipe, estimate) | Serve candidate | Incumbent 7B: train / serve |
|---|---:|---|---|---|
| t4_serving_resident | 4,300 | Qwen3.5-0.8B (seq512 mb1, 3,169) | Qwen3.5-4B (Q4_0, 3,286) | does not fit / does not fit |
| std_8gb | 6,656 | Qwen3.5-4B (seq256 mb1 ckpt, 5,449) | Qwen3.5-4B (Q8_0, 5,290) | does not fit / Q4_0 (5,016) |
| big_12gb | 11,264 | Qwen3.5-4B (seq512 mb1, 7,653) | Qwen3.5-9B (Q4_0, 5,857) | seq512 mb1 ckpt (7,491) / Q8_0 (8,648) |
| t4_serving_stopped | 15,360 | Qwen3.5-9B (seq512 mb1 ckpt, 11,289) | Qwen3.5-9B (Q8_0, 10,126) | seq512 mb1 (9,836) / Q8_0 (8,648) |
| xl_16gb_plus | 22,528 | Qwen3.5-9B (seq512 mb2, 17,161) | Qwen3.5-9B (Q8_0, 10,126) | seq512 mb2 (13,113) / Q8_0 (8,648) |

Reading it: on the T4 with serving stopped, the 9B Qwen 3.5 is estimated to train only with gradient checkpointing; with serving resident only the
0.8B fits. These are estimates with no calibration point on Qwen 3.5; the first real run of each must be recorded with `calibrate_from_measurement`.

## Not done here

- No GPU run, no training, no measurement; no calibration record exists yet.
- No held-out evaluation was run, so the promotion gate has not decided any real candidate.
- The S19 tier table still names the three ids the Hub does not serve; fixing it, and wiring the plan into the campaign orchestrator / expert
  registry (parallel S21 PRs), is follow-up work.
- Bits-per-byte comparison across tokenizers (required by `docs/T4_NIGHT_MODEL_DECISION.md` when the base changes) is the evaluator's job, outside
  this module: the gate consumes per-item outcomes only.
