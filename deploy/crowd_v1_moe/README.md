# Serving the Crowd-v1 MoE add-on (`hivemind/crowd-v1-moe`)

This folder describes one judged add-on, `crowd-v1-moe-e50-r00`, and how to serve it. **Nothing here is deployed.**
The owner deploys, canary first.

## What it is

- 12 buckets, 19.17 M parameters, 38 MB:
  - `expert_0..7`: 8 LoRA domain experts, rank 8, on all 24 layers;
  - `router_frag_0..3`: the top-2 sigma router gates plus the final LayerNorm.
- It sits on top of the frozen `Vxtzq/Crowd-v1` base, **pinned to revision `b63212a8`**. It means nothing on another
  revision, so the producer never follows `status.json`.
- It is the weight interpolation chosen in amendment A2 of `docs/E_B3_MOE_CORRECTION_PREREGISTRATION.md`, and confirmed
  once on the frozen 240 chunks (amendment A4). See `e50_r00_manifest.json` → `evidence`.

## What was measured, and what was not

| Measured | Value |
|---|---|
| Mean next-token NLL against the same base (tokens 64–127 of never-used rows), frozen 240 chunks | **−0.087 nat**, 95 % CI [−0.098, −0.076]; lower on 197 of 240 chunks |
| General text (FineWeb-Edu) / code / maths / exoplanets / stars / security | +0.30 / −1.82 / −1.71 / −1.57 / −6.00 / −2.18 % |
| Serving path reproduces the judge (`scripts/crowd_v1_moe_acceptance.py`, CPU float32, 12 chunks) | max gap 4.8e-7 nat; receipt in `artifacts/experiments/` |

**Not measured:**
- chat or instruction-following quality;
- safety behaviour;
- long context;
- GPU half-precision equivalence.

It is a 1 B base model with domain adapters. Describe it as that, not as an assistant.

## Files the producer needs

| File | Where | sha256 |
|---|---|---|
| `crowd_v1_moe_e50_r00.safetensors` | not uploaded yet: owner's choice of a Hugging Face repo or the private GCS bucket | `80dccfa8…22d3` |
| base `model.safetensors` @ `b63212a8` | fetched by revision from `Vxtzq/Crowd-v1` | `1f8214ca…209f` |
| `tokenizer.json` @ `b63212a8` | fetched by revision from `Vxtzq/Crowd-v1` | `aeb13307…dae4` |

The producer checks all three and refuses to start on any mismatch.

## Canary (owner)

1. Put the add-on next to the manifest, or pass `--addon PATH`. Then run the preflight, which loads, verifies and exits:

   ```bash
   python scripts/hivemind_crowd_v1_moe_producer.py --manifest deploy/crowd_v1_moe/e50_r00_manifest.json \
       --addon /path/crowd_v1_moe_e50_r00.safetensors --check-only
   ```

   Expect `"event": "crowd_v1_moe_loaded"` and `"readback_max_abs_diff": 0.0`.

2. Serve on loopback, on one producer only. The RTX 2070 is preferred: the T4 keeps its 4,300 MiB floor and the device
   guard applies either way.

   ```bash
   python scripts/hivemind_crowd_v1_moe_producer.py --manifest ... --addon ... --port 8792
   ```

   The producer advertises only `hivemind/crowd-v1-moe` and `crowd-v1-moe`. A request naming any other model gets a 404.

3. Route it through the shield and heartbeat it under its own ID:

   ```bash
   hivemind_producer_shield.py ... --route hivemind/crowd-v1-moe=http://127.0.0.1:8792
   hivemind_producer_agent.py ... --model crowd-v1-moe=hivemind/crowd-v1-moe \
       --routed-upstream hivemind/crowd-v1-moe=http://127.0.0.1:8792
   ```

   The gateway accepts any well-formed model ID in a heartbeat, so no gateway deploy is needed to route it.

4. Smoke-test through the gateway with `model: "hivemind/crowd-v1-moe"`, then watch latency and VRAM. Roll back by
   stopping the producer: nothing else changes.

## Catalog entry: deliberately not added

The catalog invariants (owner decision, 2026-11-07, `tests/test_router_catalog.py`) say:
- an unpublished entry is shown as `UNPUBLISHED_NOT_TRAINED`;
- no entry may link to `xaviercallens/*`.

A model that is trained but not yet published fits neither rule. Two ways forward:
- once the add-on is published (a repository outside `xaviercallens/*`), add a `PrewiredPartnerModel` with that
  `hf_repo`;
- or the owner decides how the catalog should show a trained-but-unpublished artefact.
