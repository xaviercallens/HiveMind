# T4 night training: base model decision (2026-10-26)

**Decision (owner):** keep **`ibm-granite/granite-3.1-1b-a400m-base`** (MoE) as the base for the night QLoRA runs. The second alternative, for other nights' tests, is a
**dense 4B model, `Qwen/Qwen3-4B-Instruct-2507`**. Nothing changes on the node for this decision; it is what already runs.

## Constraint

The governor starts training only with at least 4,300 MiB of free VRAM (`--min-free-mib 4300`). The probe approximates that with a hard allocator cap of
3,950 MiB (4,300 minus the ~350 MiB CUDA context) on the trainer's own code path (`moe_app.load_peft_model`: NF4 on `nn.Linear`, fp32 LoRA rank 8;
`replay_inner_segment`, sequence 128, micro-batch 2). Reproduce with `scripts/t4_model_probe.py <model>`; the header-based prediction is `scripts/t4_model_candidates.py`.

## Measured

| Model | Type | Resident after load | Result under the 3,950 MiB cap |
|---|---|---|---|
| **granite-3.1-1b-a400m-base** (chosen) | MoE, 1.3B total / 0.4B active | 2,444 MiB | **fits**: peak 3,409 MiB allocated (3,522 reserved); 1.9 s per step, 134 tokens/s measured while a package build shared the CPU, so a lower bound on speed |
| Qwen3-4B-Instruct-2507 (second alternative) | dense 4B | 2,578 MiB (the header prediction was 2,579) | **out of memory during training** at 3.97 GiB at micro-batch 2 |
| Phi-4-mini-instruct | dense 3.8B | 2,764 MiB | out of memory during training at 4.0 GiB |
| granite-3.1-3b-a800m-base | MoE 3.3B | would be ~6.0 GiB | out of memory loading (fused experts stay 16-bit) |
| Phi-tiny-MoE-instruct | MoE 3.8B | would be ~6.1 GiB | out of memory loading |
| OLMoE-1B-7B | MoE 6.9B | would be ~12.5 GiB | not run; far over |
| Gemma-4 E4B | dense 8B incl. per-layer embeddings | would be ~8.7 GiB (3.5 GiB even with the per-layer table on the CPU) | not run; over |
| Qwen3.5-4B | dense 4.7B | predicted 3.25 GiB | not run (the probe job was stopped for low memory) |

## Why no larger MoE fits

With transformers 5.17 every MoE checked stores its experts as fused 3-D tensors (`gate_up_proj`, `down_proj`). bitsandbytes quantizes only `nn.Linear`, so those
tensors stay 16-bit: the resident size of a MoE is about two bytes per total parameter. Only a MoE of roughly 1.5B total parameters or fewer fits. (The
checkpoint headers show per-expert linear layers, which is misleading: the fusing happens at load time. The meta-device build in the probe notes shows it.)
A dense model quantizes everything, hence 4B fits as weights.

## Plan for the alternative night (dense 4B)

The 4B weights fit; training activations do not at micro-batch 2. To test, in this order, stopping at the first that holds the cap:
1. micro-batch 1, same sequence length;
2. gradient checkpointing on top;
3. only then a shorter sequence.

Acceptance: peak at most 3,950 MiB under the cap; the workunit still validates by replay; and compare quality in **bits per byte** on held-out text, not
per-token loss, because the tokenizers differ (the fixed held-out sets are token ids from Granite's tokenizer, so workunits and eval sets must be rebuilt per model).
The BOINC package also needs that model's archive instead of Granite's. A dense 4B is not a MoE: it serves the quality question, not the MoE one.

## Notes for whoever runs the probes

- The transformers "allocator warm-up" pre-allocates about the full model size in one block and fails with only a few GiB free; the probe disables it
  (`caching_allocator_warmup`). Any larger model on the real trainer needs the same.
- Loading a 4B model reads about 8 GB of 16-bit weights through CPU memory and took 3-4 minutes; running probes next to a package build exhausted memory and
  the harness stopped both. Run them one at a time, and only while the node is idle.
- Night results for the chosen model: 98 rounds, plateau after about 30, mean held-out loss -5.8 % (see `deploy/t4_node/live/README.md`). Expect little more from the
  same model and data; the lever left is data, not model size.
