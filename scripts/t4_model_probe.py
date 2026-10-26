#!/usr/bin/env python3
"""Does a candidate base model fit the T4 night-training budget, and how fast is it?

Uses the trainer's own code path (``moe_app.load_peft_model`` -> NF4 attention + fp32 LoRA, ``replay_inner_segment``) with the workunit shape
(seq 128, micro-batch 2) on random token ids, so it measures memory and speed, NOT quality. The CUDA allocator is capped (``--cap-mib``) to approximate the
governor's ``--min-free-mib 4300`` rule minus the ~350 MiB CUDA context: a model that does not fit fails here with an out-of-memory error.
Prints one JSON line. Run one model per process:  python3 scripts/t4_model_probe.py ibm-granite/granite-3.1-3b-a800m-base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--cap-mib", type=int, default=3950)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--segment-steps", type=int, default=10)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    a = ap.parse_args()
    res: dict = {"model_id": a.model_id, "cap_mib": a.cap_mib}
    try:
        import torch

        from hivemind.boinc.real_client.moe_app import load_peft_model
        from hivemind.core.diloco_inner import replay_inner_segment
        from hivemind.core.quantization import extract_lora_state_dict

        # transformers pre-allocates (almost) the whole model size in one block before loading ("allocator warm-up"). It is only a speed-up, but with a few GiB
        # free it OOMs before a single weight is loaded, so the night trainer must skip it for any model whose 16-bit size exceeds the free VRAM.
        import transformers.modeling_utils as _mu

        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_mib * 2**20 / total))
        spec = {"model_id": a.model_id, "init_seed": 7, "rank": 8, "alpha": 16}
        t = time.time()
        model = load_peft_model(spec)
        res["load_s"] = round(time.time() - t, 1)
        res["after_load_mib"] = torch.cuda.memory_allocated() // 2**20
        res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res["total_params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)  # NF4 params report packed counts: indicative only
        vocab = int(getattr(model.config, "vocab_size", None) or model.config.get_text_config().vocab_size)
        g = torch.Generator().manual_seed(0)
        steps = a.segment_steps * a.segments
        toks = torch.randint(0, min(vocab, 30000), (steps * a.micro_batch, a.seq_len), generator=g)
        state, times = extract_lora_state_dict(model), []
        torch.cuda.reset_peak_memory_stats()
        for k in range(a.segments):
            torch.cuda.synchronize()
            t = time.time()
            state = replay_inner_segment(model, state, toks, k * a.segment_steps, (k + 1) * a.segment_steps, lr=6e-5, weight_decay=0.01, rng_seed=1, micro_batch_size=a.micro_batch)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        steady = times[1:] or times  # the first segment includes warm-up
        res.update(
            ok=True,
            peak_alloc_mib=torch.cuda.max_memory_allocated() // 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() // 2**20,
            seg_seconds=[round(x, 2) for x in times],
            ms_per_step=round(1000 * (sum(steady) / len(steady)) / a.segment_steps),
            tokens_per_s=round(a.micro_batch * a.seq_len * a.segment_steps * len(steady) / sum(steady)),
        )
    except Exception as e:  # noqa: BLE001 - the failure IS the result (e.g. OOM, unsupported architecture)
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:240]}", trace_tail=traceback.format_exc().splitlines()[-3:])
    print(json.dumps(res))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
