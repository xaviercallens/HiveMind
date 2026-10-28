#!/usr/bin/env python3
"""Does a candidate base model fit the T4 night-training budget, and how fast is it?

Uses the trainer's own code path (``moe_app.load_peft_model`` -> NF4 attention + fp32 LoRA, ``replay_inner_segment``) with the workunit shape
(seq 128, micro-batch 2) on random token ids, so it measures memory and speed, NOT quality. The CUDA allocator is capped (``--cap-mib``) to approximate the
governor's ``--min-free-mib 4300`` rule minus the ~350 MiB CUDA context: a model that does not fit fails here with an out-of-memory error.
Prints one JSON line. Run one model per process:  python3 scripts/t4_model_probe.py ibm-granite/granite-3.1-3b-a800m-base

Optional recipe flags (all default to the production workunit shape, so a run without them is unchanged): ``--rank`` (alpha defaults to 2 x rank, i.e. 16 at rank 8),
``--targets`` (LoRA target modules; default is the trainer's ``LORA_TARGETS``), ``--checkpointing`` (gradient checkpointing; reported as CHECKPOINTING_UNSUPPORTED
if the model cannot do it, never silently ignored) and ``--max-gpu-temp-c`` (abort above this temperature). The JSON line carries ``status``: OK, OOM, TOO_HOT,
CHECKPOINTING_UNSUPPORTED or ERROR. Argument parsing (``build_parser``/``build_spec``) does not import torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class CheckpointingUnsupported(RuntimeError):
    """The loaded model cannot do gradient checkpointing (reported honestly, never ignored)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--cap-mib", type=int, default=3950)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--segment-steps", type=int, default=10)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default 2 x rank: 16 at rank 8, as before)")
    ap.add_argument("--targets", default=None, help="comma-separated LoRA target modules (default: the trainer's LORA_TARGETS)")
    ap.add_argument("--checkpointing", action="store_true", help="enable gradient checkpointing (CHECKPOINTING_UNSUPPORTED if the model cannot)")
    ap.add_argument("--max-gpu-temp-c", type=int, default=83, help="abort the probe above this GPU temperature")
    return ap


def build_spec(a: argparse.Namespace) -> dict:
    """The workunit spec handed to ``load_peft_model``. With default flags it is exactly the historic ``{model_id, init_seed 7, rank 8, alpha 16}``."""
    alpha = a.alpha if a.alpha is not None else 2 * a.rank
    return {"model_id": a.model_id, "init_seed": 7, "rank": a.rank, "alpha": alpha}


def parse_targets(a: argparse.Namespace) -> tuple[str, ...] | None:
    return tuple(t.strip() for t in a.targets.split(",") if t.strip()) if a.targets else None


def classify_status(exc: BaseException) -> str:
    if isinstance(exc, CheckpointingUnsupported):
        return "CHECKPOINTING_UNSUPPORTED"
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return "OOM"
    return "ERROR"


def enable_checkpointing(model) -> None:
    """Turn on gradient checkpointing (non-reentrant, so frozen NF4 inputs need no ``requires_grad`` hook) and verify it took effect."""
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn is None:
        raise CheckpointingUnsupported("model has no gradient_checkpointing_enable")
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:  # noqa: BLE001 - any failure here means checkpointing is unavailable for this model
        raise CheckpointingUnsupported(f"{type(e).__name__}: {str(e)[:200]}") from e
    if not getattr(model, "is_gradient_checkpointing", False):
        raise CheckpointingUnsupported("gradient_checkpointing_enable ran but the model does not report is_gradient_checkpointing")
    try:
        model.config.use_cache = False  # the KV cache is incompatible with checkpointing
    except Exception:  # noqa: BLE001
        pass


class Emitter:
    """Prints the single JSON result line exactly once, whichever thread gets there first."""

    def __init__(self) -> None:
        self._lock, self.done = threading.Lock(), False

    def emit(self, res: dict) -> bool:
        with self._lock:
            if self.done:
                return False
            self.done = True
            print(json.dumps(res), flush=True)
            return True


def gpu_temp_c() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    return int(out.strip().splitlines()[0])


def watch_temperature(limit_c: int, read_temp, on_hot, stop: threading.Event, interval_s: float = 1.0) -> None:
    """Poll the GPU temperature until ``stop``; call ``on_hot(temp)`` once when it exceeds ``limit_c``. Read errors are ignored (the queue samples too)."""
    while not stop.wait(interval_s):
        try:
            t = read_temp()
        except Exception:  # noqa: BLE001
            continue
        if t > limit_c:
            on_hot(t)
            return


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    targets = parse_targets(a)
    alpha = a.alpha if a.alpha is not None else 2 * a.rank
    res: dict = {"model_id": a.model_id, "cap_mib": a.cap_mib, "status": "ERROR", "rank": a.rank, "alpha": alpha, "seq_len": a.seq_len,
                 "micro_batch": a.micro_batch, "checkpointing": bool(a.checkpointing), "targets": list(targets) if targets else None,
                 "max_gpu_temp_c": a.max_gpu_temp_c}
    emitter, stop = Emitter(), threading.Event()
    max_temp: dict = {"v": None}

    def on_hot(t: int) -> None:
        r = dict(res, ok=False, status="TOO_HOT", max_temp_c=t, error=f"GPU temperature {t} C above limit {a.max_gpu_temp_c} C")
        if emitter.emit(r):
            os._exit(4)  # the training loop is inside CUDA calls: do not wait for it

    def read_temp() -> int:
        t = gpu_temp_c()
        max_temp["v"] = t if max_temp["v"] is None else max(max_temp["v"], t)
        return t

    watcher = threading.Thread(target=watch_temperature, args=(a.max_gpu_temp_c, read_temp, on_hot, stop), daemon=True)
    try:
        import torch

        from hivemind.boinc.real_client import moe_app
        from hivemind.core.diloco_inner import replay_inner_segment
        from hivemind.core.quantization import extract_lora_state_dict

        if targets:
            moe_app.LORA_TARGETS = targets  # load_peft_model reads this module global; the default is left untouched
        # transformers pre-allocates (almost) the whole model size in one block before loading ("allocator warm-up"). It is only a speed-up, but with a few GiB
        # free it OOMs before a single weight is loaded, so the night trainer must skip it for any model whose 16-bit size exceeds the free VRAM.
        import transformers.modeling_utils as _mu

        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
        watcher.start()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_mib * 2**20 / total))
        spec = build_spec(a)
        t = time.time()
        model = moe_app.load_peft_model(spec)
        res["load_s"] = res["load_seconds"] = round(time.time() - t, 1)
        if a.checkpointing:
            enable_checkpointing(model)
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
        peak_a, peak_r = torch.cuda.max_memory_allocated() // 2**20, torch.cuda.max_memory_reserved() // 2**20
        step_s = (sum(steady) / len(steady)) / a.segment_steps
        tps = a.micro_batch * a.seq_len * a.segment_steps * len(steady) / sum(steady)
        res.update(
            ok=True,
            status="OK",
            peak_alloc_mib=peak_a,
            peak_reserved_mib=peak_r,
            torch_peak_allocated_mib=peak_a,
            torch_peak_reserved_mib=peak_r,
            seg_seconds=[round(x, 2) for x in times],
            ms_per_step=round(1000 * step_s),
            tokens_per_s=round(tps),
            seconds_per_step=round(step_s, 3),
            tokens_per_second=round(tps),
        )
    except Exception as e:  # noqa: BLE001 - the failure IS the result (e.g. OOM, unsupported architecture)
        res.update(ok=False, status=classify_status(e), error=f"{type(e).__name__}: {str(e)[:240]}", trace_tail=traceback.format_exc().splitlines()[-3:])
    finally:
        stop.set()
    if max_temp["v"] is not None:
        res["max_temp_c"] = max_temp["v"]
    emitter.emit(res)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
