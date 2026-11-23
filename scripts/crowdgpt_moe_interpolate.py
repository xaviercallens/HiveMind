# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Weight-space interpolation of a MoE add-on toward its zero-delta start (correction screen, 23 Nov 2026).

    CUDA_VISIBLE_DEVICES= python scripts/crowdgpt_moe_interpolate.py --spec SPEC.json --base-cache-dir DIR \\
        --checkpoint C2.safetensors --lambdas 0.25 0.5 0.75 --out-dir OUT

For each lambda writes ``theta(lambda) = start + lambda * (checkpoint - start)`` over every trainable bucket (LoRA
experts and router fragments), where ``start`` is the freshly upcycled model's buckets: the zero-delta state that the
paired evaluation shows to reproduce dense Crowd-v1 exactly. lambda = 0 is the base model, lambda = 1 the checkpoint.
CPU only. The split (experts, top-k, rank, routing, fragments) is read from the spec's ``arms.moe.split``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _eval_module():
    spec = importlib.util.spec_from_file_location("crowdgpt_moe_paired_eval", REPO / "scripts" / "crowdgpt_moe_paired_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def interpolate(start: dict[str, torch.Tensor], end: dict[str, torch.Tensor], lam: float) -> dict[str, torch.Tensor]:
    """``start + lam * (end - start)`` per bucket; names and shapes must match."""
    if set(start) != set(end):
        raise ValueError("bucket names differ")
    out = {}
    for k, s in start.items():
        e = end[k]
        if tuple(s.shape) != tuple(e.shape):
            raise ValueError(f"{k}: shape {tuple(s.shape)} != {tuple(e.shape)}")
        out[k] = (s.float() + float(lam) * (e.float() - s.float())).contiguous()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--base-cache-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--lambdas", nargs="+", type=float, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("refused: run with CUDA_VISIBLE_DEVICES= (CPU only)", file=sys.stderr)
        return 2
    if any(not (0.0 <= lam <= 1.0) for lam in a.lambdas):
        print("refused: every lambda must be in [0, 1]", file=sys.stderr)
        return 2
    from safetensors.torch import save_file

    pe = _eval_module()
    split = json.loads(a.spec.read_text())["arms"]["moe"]["split"]
    worker = pe._load_worker()
    model, _ = worker._load_or_build_model(cache_dir=a.base_cache_dir, use_small_arch=False,
                                           direct_trunk_device="cpu", direct_trunk_dtype=torch.float32)
    pe.upcycle_and_load(model, split, None, worker.load_init_buckets)
    from hivemind.swarm.crowdgpt_moe_opendiloco import extract_moe_trainable_buckets

    start = {k: v.detach().cpu().float().clone() for k, v in
             extract_moe_trainable_buckets(model, num_router_fragments=int(split["num_router_fragments"])).items()}
    end = worker.load_init_buckets(str(a.checkpoint), start)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              "files": {}}
    for lam in a.lambdas:
        th = {k: v.to(torch.bfloat16) for k, v in interpolate(start, end, lam).items()}
        p = a.out_dir / f"interp_lambda_{lam:.2f}.safetensors"
        save_file(th, str(p))
        report["files"][f"{lam:.2f}"] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        print(f"lambda {lam:.2f} -> {p.name}")
    (a.out_dir / "interp_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
