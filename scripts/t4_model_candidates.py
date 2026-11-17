#!/usr/bin/env python3
"""Rank Hugging Face MoE / small models by what the T4 night-training loader would keep resident, from safetensors HEADERS only (no weight download).

Rule mirrored from ``moe_app.load_peft_model`` (bitsandbytes NF4 on ``nn.Linear`` only): a 2-D ``*.weight`` that is not an embedding/lm_head is quantised
(~0.53 byte/param incl. quant state); everything else (embeddings, norms, FUSED 3-D expert tensors) stays 16-bit (2 bytes/param). Optionally the per-layer
embedding table (Gemma-style ``embed_tokens_per_layer``) is counted as CPU-offloaded. The T4 budget is the governor's ``--min-free-mib 4300`` minus the CUDA
context (~350 MiB) and activations/LoRA/optimizer (~600-900 MiB at seq 128, micro-batch 2).

  python3 scripts/t4_model_candidates.py                      # search the Hub and rank
  python3 scripts/t4_model_candidates.py org/model [org/model2 ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

NF4_BYTES = 0.53
BUDGET_MIB = 4300 - 350 - 800  # weights may use this much GPU memory
EMBED_RE = re.compile(r"(embed|lm_head|wte|wpe)", re.I)
PER_LAYER_EMBED_RE = re.compile(r"embed_tokens_per_layer|per_layer", re.I)


def hub_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hivemind-probe"}), timeout=60) as r:
        return json.loads(r.read())


def header_of(repo: str) -> dict[str, tuple[list[int], str]]:
    """Tensor name -> (shape, dtype) for every safetensors shard, read with HTTP range requests (headers only)."""
    info = hub_json(f"https://huggingface.co/api/models/{repo}")
    files = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors")]
    out: dict[str, tuple[list[int], str]] = {}
    for f in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        n = int.from_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}), timeout=60).read(8), "little")
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"}), timeout=60).read()
        for k, v in json.loads(raw).items():
            if k != "__metadata__":
                out[k] = (v["shape"], v["dtype"])
    return out


def footprint(tensors: dict[str, tuple[list[int], str]]) -> dict[str, float]:
    total = quant = fp16 = fused = pl = 0
    for name, (shape, _) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        total += n
        if PER_LAYER_EMBED_RE.search(name):
            pl += n
        elif len(shape) == 2 and name.endswith(".weight") and not EMBED_RE.search(name):
            quant += n
        else:
            fp16 += n
            if len(shape) >= 3:
                fused += n
    mib = lambda b: b / 2**20  # noqa: E731
    return {
        "total_B": total / 1e9,
        "nf4_MiB": mib(quant * NF4_BYTES),
        "fp16_MiB": mib(fp16 * 2),
        "fused_experts_MiB": mib(fused * 2),
        "per_layer_embed_MiB": mib(pl * 2),
        "resident_MiB": mib(quant * NF4_BYTES + fp16 * 2),
        "resident_if_PLE_on_cpu_MiB": mib(quant * NF4_BYTES + fp16 * 2),
        "resident_if_PLE_on_gpu_MiB": mib(quant * NF4_BYTES + fp16 * 2 + pl * 2),
    }


def search() -> list[str]:
    seen: dict[str, int] = {}
    for q in ("moe", "a1b", "a3b", "a800m", "a400m", "tiny-moe", "olmoe", "granite-4", "gemma-4", "lfm2", "qwen3.5", "phi-4"):
        try:
            for m in hub_json(f"https://huggingface.co/api/models?search={q}&sort=downloads&direction=-1&limit=25&filter=text-generation"):
                seen[m["id"]] = m.get("downloads", 0)
        except Exception as e:  # noqa: BLE001
            print("search failed", q, e, file=sys.stderr)
    return [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--max-total-b", type=float, default=12.0, help="skip models larger than this (billions of parameters)")
    a = ap.parse_args()
    repos = a.repos or search()
    rows = []
    for r in repos:
        try:
            fp = footprint(header_of(r))
        except Exception as e:  # noqa: BLE001 - gated, no safetensors, etc.
            print(f"skip {r}: {type(e).__name__}", file=sys.stderr)
            continue
        if fp["total_B"] > a.max_total_b:
            continue
        rows.append((r, fp))
    rows.sort(key=lambda x: x[1]["resident_MiB"])
    print(f"budget for weights: {BUDGET_MIB} MiB  (4300 free - 350 context - 800 activations/LoRA/optimizer)")
    print(f"{'model':52s} {'total B':>7s} {'NF4':>6s} {'fp16':>6s} {'fusedExp':>8s} {'perLayerEmb':>11s} {'resident':>8s} {'fits?':>6s}")
    for r, f in rows:
        res = f["resident_MiB"]
        print(f"{r:52s} {f['total_B']:7.2f} {f['nf4_MiB']:6.0f} {f['fp16_MiB']:6.0f} {f['fused_experts_MiB']:8.0f} {f['per_layer_embed_MiB']:11.0f} {res:8.0f} {'yes' if res <= BUDGET_MIB else 'NO':>6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
