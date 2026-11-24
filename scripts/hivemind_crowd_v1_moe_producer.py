#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Producer for a Crowd-v1 MoE LoRA add-on (`hivemind/crowd-v1-moe`) on a pinned `Vxtzq/Crowd-v1` base.

Unlike `hivemind_crowd_v1_producer.py` it never follows `status.json`: the add-on was trained and judged on one base
revision, so the manifest pins that revision and the base weights, tokenizer and add-on sha256, and the producer
refuses to start on any mismatch. It advertises only the manifest's model IDs and answers 404 for any other model,
so it can never answer under the base model's name.

Canary first (the owner deploys):

    python scripts/hivemind_crowd_v1_moe_producer.py --manifest deploy/crowd_v1_moe/e50_r00_manifest.json \\
        --addon /path/to/crowd_v1_moe_e50_r00.safetensors --check-only

prints the integrity report and exits; without `--check-only` it serves on 127.0.0.1:8792, behind
`hivemind_producer_shield.py --route hivemind/crowd-v1-moe=http://127.0.0.1:8792`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.crowd_v1_moe_addon import (  # noqa: E402
    build_crowd_v1_moe_model,
    fetch_pinned_base,
    load_addon_manifest,
    verify_file_sha256,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="add-on manifest (schema hivemind.crowd_v1_moe_addon.v1)")
    ap.add_argument("--addon", type=Path, default=None, help="add-on safetensors (default: next to the manifest)")
    ap.add_argument("--base-weights", type=Path, default=None, help="local pinned base weights (else fetched by revision)")
    ap.add_argument("--tokenizer-json", type=Path, default=None, help="local pinned tokenizer (else fetched by revision)")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "hivemind-crowd-v1-moe",
                    help="own cache: the base producer's cache deletes superseded revisions")
    ap.add_argument("--host", default="127.0.0.1", help="loopback bind address (never 0.0.0.0 without the shield)")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--cpu-only", action="store_true", help="force CPU execution")
    ap.add_argument("--threads", type=int, default=6, help="PyTorch CPU intra-op threads")
    ap.add_argument("--check-only", action="store_true", help="load, verify, print the report and exit")
    args = ap.parse_args(argv)

    if args.host in {"0.0.0.0", "::"}:
        raise ValueError("Refusing wildcard bind (0.0.0.0 / ::); bind 127.0.0.1 behind hivemind_producer_shield.py")
    torch.set_num_threads(max(1, int(args.threads)))

    manifest = load_addon_manifest(args.manifest)
    addon = args.addon or (args.manifest.parent / manifest.addon_file)
    if args.base_weights is None or args.tokenizer_json is None:
        fetched = fetch_pinned_base(manifest, args.cache_dir)
        base_weights = args.base_weights or fetched["weights"]
        tokenizer_json = args.tokenizer_json or fetched["tokenizer"]
    else:
        base_weights, tokenizer_json = args.base_weights, args.tokenizer_json
    verify_file_sha256(tokenizer_json, manifest.tokenizer_sha256, "tokenizer")

    from hivemind.swarm.crowdgpt_live_runtime import create_crowdgpt_producer_app, select_safe_execution_device

    decision = select_safe_execution_device(prefer_cuda=not args.cpu_only)
    run_dtype = _DTYPES.get(decision.dtype_name, torch.float32)
    # build in the run dtype on GPU (RAM), in float32 on CPU (the judged path)
    model, info = build_crowd_v1_moe_model(manifest, base_weights, addon, build_dtype=run_dtype)
    model.place_for_inference(device=decision.device, dtype=run_dtype, offload_engram_to_cpu=decision.offload_engram_to_cpu)
    model.eval()

    revision_info = {
        "hf_revision": manifest.base_hf_revision,
        "global_step": None,
        "weights_sha256": info["base_weights_sha256"],
        "addon_id": manifest.addon_id,
        "addon_sha256": info["addon"]["addon_sha256"],
        "pinned": True,
        "loaded_utc": round(time.time(), 3),
    }
    report = {"event": "crowd_v1_moe_loaded", **revision_info, "addon": info["addon"],
              "advertised_model_ids": list(manifest.advertised_model_ids), "device_decision": decision.__dict__}
    print(json.dumps(report), flush=True)
    if args.check_only:
        return 0

    import uvicorn
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    app = create_crowdgpt_producer_app(
        model,
        encode_fn=lambda text: list(tokenizer.encode(text).ids),
        decode_fn=lambda ids: tokenizer.decode([int(i) for i in ids], skip_special_tokens=True),
        revision_info=revision_info,
        device_decision=decision,
        sync_hook=None,
        advertised_model_ids=manifest.advertised_model_ids,
        model_repo=f"{manifest.base_hf_repo}@{manifest.base_hf_revision[:12]}+{manifest.addon_id}",
        strict_model_ids=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
