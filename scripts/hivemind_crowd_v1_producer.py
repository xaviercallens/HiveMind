#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live Producer & Auto-Sync Worker for `Vxtzq/Crowd-v1` (`hivemind/crowd-v1`).

Loads the latest `Vxtzq/Crowd-v1` checkpoint (`model.safetensors` + `CrowdEngram` + `tokenizer.json`)
from Hugging Face, watches `status.json` (`globalStep` / `hfRevision`) to hot-reload new federated
checkpoints automatically, enforces RTX 2070 (`>= 1,536 MiB`) and GCP Tesla T4 (`>= 4,300 MiB`)
free VRAM floors, and serves an OpenAI-compatible HTTP endpoint (`127.0.0.1:8791`) designed to sit
behind `hivemind_producer_shield.py --route hf.co/Vxtzq/Crowd-v1=http://127.0.0.1:8791`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import torch
import uvicorn
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.crowdgpt_live_runtime import (  # noqa: E402
    CrowdGPTHuggingFaceSync,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    create_crowdgpt_producer_app,
    load_crowd_v1_safetensors_into_model,
    select_safe_execution_device,
)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="Loopback bind address (never bind 0.0.0.0 without shield)")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "hivemind-crowd-v1")
    ap.add_argument("--refresh-interval-s", type=float, default=600.0, help="HF status.json poll interval in seconds")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU execution to preserve 100% of GPU VRAM")
    ap.add_argument("--threads", type=int, default=6, help="PyTorch CPU intra-op threads when running on CPU")
    args = ap.parse_args(argv)

    if args.host in ("0.0.0.0", "::"):
        raise ValueError("Refusing wildcard bind (0.0.0.0 / ::); bind 127.0.0.1 behind hivemind_producer_shield.py")

    torch.set_num_threads(max(1, int(args.threads)))

    sync = CrowdGPTHuggingFaceSync(
        cache_dir=args.cache_dir,
        refresh_interval_s=args.refresh_interval_s,
    )
    print("[crowd-v1] Fetching latest Vxtzq/Crowd-v1 status.json & safetensors from Hugging Face...", flush=True)
    bundle = sync.ensure_latest_files(force_check=True)
    tokenizer = Tokenizer.from_file(str(bundle["tokenizer_path"]))

    cfg = CrowdV1RuntimeConfig()
    model = CrowdV1WithEngram(cfg)
    sha256 = load_crowd_v1_safetensors_into_model(model, bundle["weights_path"])
    model.eval()

    decision = select_safe_execution_device(prefer_cuda=not args.cpu_only)
    model.place_for_inference(
        device=decision.device,
        dtype=_dtype_from_name(decision.dtype_name),
        offload_engram_to_cpu=decision.offload_engram_to_cpu,
    )

    loaded_state: dict[str, Any] = {
        "hf_revision": bundle["hf_revision"],
        "global_step": bundle["global_step"],
        "weights_sha256": sha256,
        "last_loss": bundle["status"].get("lastLoss"),
        "total_tokens": bundle["status"].get("totalTokens"),
        "loaded_utc": round(time.time(), 3),
    }
    print(
        json.dumps(
            {
                "event": "crowd_v1_loaded",
                **loaded_state,
                "device_decision": decision.__dict__,
            }
        ),
        flush=True,
    )

    reload_lock = threading.Lock()
    last_poll = [time.time()]

    def _maybe_sync() -> dict[str, Any] | None:
        now = time.time()
        if now - last_poll[0] < args.refresh_interval_s:
            return None
        if not reload_lock.acquire(blocking=False):
            return None
        try:
            last_poll[0] = now
            nxt = sync.ensure_latest_files(force_check=True)
            if nxt["hf_revision"] != loaded_state["hf_revision"] or nxt["global_step"] != loaded_state["global_step"]:
                new_sha = load_crowd_v1_safetensors_into_model(model, nxt["weights_path"])
                dec = select_safe_execution_device(prefer_cuda=not args.cpu_only)
                model.place_for_inference(
                    device=dec.device,
                    dtype=_dtype_from_name(dec.dtype_name),
                    offload_engram_to_cpu=dec.offload_engram_to_cpu,
                )
                model.eval()
                loaded_state.update(
                    {
                        "hf_revision": nxt["hf_revision"],
                        "global_step": nxt["global_step"],
                        "weights_sha256": new_sha,
                        "last_loss": nxt["status"].get("lastLoss"),
                        "total_tokens": nxt["status"].get("totalTokens"),
                        "loaded_utc": round(time.time(), 3),
                    }
                )
                print(json.dumps({"event": "crowd_v1_hot_reloaded", **loaded_state}), flush=True)
            return dict(loaded_state)
        except Exception as exc:
            print(json.dumps({"event": "crowd_v1_sync_warning", "error": str(exc)}), flush=True)
            return None
        finally:
            reload_lock.release()

    app = create_crowdgpt_producer_app(
        model,
        encode_fn=lambda text: list(tokenizer.encode(text).ids),
        decode_fn=lambda ids: tokenizer.decode([int(i) for i in ids], skip_special_tokens=True),
        revision_info=loaded_state,
        device_decision=decision,
        sync_hook=_maybe_sync,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
