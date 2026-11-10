# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S6-01 synthetic benchmark: Top-k + FP8 outer-loop compression with and without error feedback.

This is a small, deterministic CPU experiment on a synthetic multi-worker least-squares
problem. It measures how compression of DiLoCo-style pseudo-gradients changes holdout loss
for a fixed number of outer rounds, and how many bytes each scheme would put on the wire.
It does not train a language model, use real network links, or measure wall-clock speed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer

PARAM_NAME = "proj.weight"
SCHEMES = ("dense", "top5_fp16", "top1_fp8", "top1_fp8_ef")


def _require_pos_int(value: Any, name: str, *, min_value: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
        raise ValueError(f"{name} must be an integer >= {min_value}")
    return value


def _require_finite_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    val = float(value)
    if (allow_zero and val < 0.0) or (not allow_zero and val <= 0.0):
        op = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be {op} 0.0")
    return val


@dataclass(frozen=True)
class SyntheticSwarmProblem:
    w_true: torch.Tensor
    worker_x: List[torch.Tensor]
    worker_y: List[torch.Tensor]
    holdout_x: torch.Tensor
    holdout_y: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.w_true, torch.Tensor) or self.w_true.ndim != 2 or self.w_true.numel() == 0 or not bool(torch.isfinite(self.w_true).all()):
            raise ValueError("w_true must be a non-empty 2D finite torch.Tensor")
        d_out, d_in = self.w_true.shape
        if not isinstance(self.worker_x, list) or not isinstance(self.worker_y, list) or len(self.worker_x) == 0 or len(self.worker_x) != len(self.worker_y):
            raise ValueError("worker_x and worker_y must be non-empty lists of equal length")
        for x, y in zip(self.worker_x, self.worker_y):
            if (
                not isinstance(x, torch.Tensor)
                or not isinstance(y, torch.Tensor)
                or x.ndim != 2
                or y.ndim != 2
                or x.shape[0] == 0
                or x.shape[0] != y.shape[0]
                or x.shape[1] != d_in
                or y.shape[1] != d_out
                or not bool(torch.isfinite(x).all())
                or not bool(torch.isfinite(y).all())
            ):
                raise ValueError("worker_x and worker_y tensors must be 2D finite tensors matching w_true dimensions")
        if (
            not isinstance(self.holdout_x, torch.Tensor)
            or not isinstance(self.holdout_y, torch.Tensor)
            or self.holdout_x.ndim != 2
            or self.holdout_y.ndim != 2
            or self.holdout_x.shape[0] == 0
            or self.holdout_x.shape[0] != self.holdout_y.shape[0]
            or self.holdout_x.shape[1] != d_in
            or self.holdout_y.shape[1] != d_out
            or not bool(torch.isfinite(self.holdout_x).all())
            or not bool(torch.isfinite(self.holdout_y).all())
        ):
            raise ValueError("holdout_x and holdout_y must be non-empty 2D finite tensors matching w_true dimensions")


def build_problem(
    seed: int = 0,
    num_workers: int = 4,
    samples_per_worker: int = 256,
    d_in: int = 64,
    d_out: int = 16,
    noise_std: float = 0.1,
) -> SyntheticSwarmProblem:
    """Shared linear target; each worker sees inputs with its own feature scaling (non-IID)."""
    _require_pos_int(seed, "seed", min_value=0)
    _require_pos_int(num_workers, "num_workers", min_value=1)
    _require_pos_int(samples_per_worker, "samples_per_worker", min_value=1)
    _require_pos_int(d_in, "d_in", min_value=1)
    _require_pos_int(d_out, "d_out", min_value=1)
    noise_std = _require_finite_float(noise_std, "noise_std", allow_zero=True)
    gen = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_out, d_in, generator=gen) / d_in**0.5
    worker_x: List[torch.Tensor] = []
    worker_y: List[torch.Tensor] = []
    for _ in range(num_workers):
        feature_scale = torch.rand(d_in, generator=gen) * 1.5 + 0.25
        x = torch.randn(samples_per_worker, d_in, generator=gen) * feature_scale
        y = x @ w_true.T + noise_std * torch.randn(samples_per_worker, d_out, generator=gen)
        worker_x.append(x)
        worker_y.append(y)
    holdout_x = torch.cat([torch.randn(256, d_in, generator=gen) for _ in range(num_workers)])
    holdout_y = holdout_x @ w_true.T + noise_std * torch.randn(holdout_x.shape[0], d_out, generator=gen)
    return SyntheticSwarmProblem(w_true, worker_x, worker_y, holdout_x, holdout_y)


def _mse(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    if not isinstance(w, torch.Tensor) or not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("w, x, and y must be torch.Tensors")
    if w.ndim != 2 or x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or x.shape[0] != y.shape[0] or x.shape[1] != w.shape[1] or y.shape[1] != w.shape[0]:
        raise ValueError("Incompatible tensor shapes for _mse")
    if not (torch.isfinite(w).all() and torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("w, x, and y must contain finite values")
    return float(((x @ w.T - y) ** 2).mean().item())


def _local_steps(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    _require_pos_int(steps, "steps", min_value=1)
    lr = _require_finite_float(lr, "lr", allow_zero=False)
    if not isinstance(w, torch.Tensor) or not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("w, x, and y must be torch.Tensors")
    if w.ndim != 2 or x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or y.shape[1] == 0 or x.shape[0] != y.shape[0] or x.shape[1] != w.shape[1] or y.shape[1] != w.shape[0]:
        raise ValueError("Incompatible tensor shapes for _local_steps")
    w = w.clone()
    n = x.shape[0]
    for _ in range(steps):
        grad = 2.0 / n * (x @ w.T - y).T @ x / y.shape[1]
        w = w - lr * grad
    return w


def _topk_fp16(tensor: torch.Tensor, topk_ratio: float) -> tuple[torch.Tensor, int]:
    """Top-k sparsification with FP16 values and uint16 indices (no FP8, no error feedback)."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("tensor must be a non-empty finite torch.Tensor")
    topk_ratio = _require_finite_float(topk_ratio, "topk_ratio", allow_zero=False)
    if topk_ratio > 1.0:
        raise ValueError("topk_ratio must be in (0.0, 1.0]")
    flat = tensor.reshape(-1)
    k = max(1, int(round(flat.numel() * topk_ratio)))
    _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=True)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx].to(torch.float16).to(torch.float32)
    return out.reshape(tensor.shape), k * 2 + k * 2 + 8


def run_benchmark(
    seed: int = 0,
    num_workers: int = 4,
    outer_rounds: int = 200,
    inner_steps: int = 8,
    inner_lr: float = 0.05,
    outer_lr: float = 1.0,
    schemes: Sequence[str] = SCHEMES,
    log_every: int = 25,
) -> Dict[str, Any]:
    """Run every scheme on the same problem and starting point; return losses and wire bytes."""
    _require_pos_int(seed, "seed", min_value=0)
    _require_pos_int(num_workers, "num_workers", min_value=1)
    _require_pos_int(outer_rounds, "outer_rounds", min_value=1)
    _require_pos_int(inner_steps, "inner_steps", min_value=1)
    _require_pos_int(log_every, "log_every", min_value=1)
    inner_lr = _require_finite_float(inner_lr, "inner_lr", allow_zero=False)
    outer_lr = _require_finite_float(outer_lr, "outer_lr", allow_zero=False)
    if isinstance(schemes, (str, bytes)) or not isinstance(schemes, Sequence) or len(schemes) == 0:
        raise ValueError("schemes must be a non-empty sequence of scheme names")
    if not all(isinstance(s, str) for s in schemes):
        raise ValueError("each scheme must be a string")
    unknown = set(schemes) - set(SCHEMES)
    if unknown:
        raise ValueError(f"unknown schemes: {sorted(unknown)}")
    problem = build_problem(seed=seed, num_workers=num_workers)
    w0 = torch.zeros_like(problem.w_true)
    results: Dict[str, Any] = {}

    for scheme in schemes:
        w = w0.clone()
        buffers = [EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.01) for _ in range(num_workers)]
        wire_bytes = 0
        curve = []
        for step in range(1, outer_rounds + 1):
            sent = []
            for k in range(num_workers):
                w_local = _local_steps(w, problem.worker_x[k], problem.worker_y[k], inner_steps, inner_lr)
                delta = {PARAM_NAME: w - w_local}
                if scheme == "dense":
                    sent.append(delta[PARAM_NAME])
                    wire_bytes += delta[PARAM_NAME].numel() * 2
                elif scheme == "top5_fp16":
                    approx, nbytes = _topk_fp16(delta[PARAM_NAME], 0.05)
                    sent.append(approx)
                    wire_bytes += nbytes
                elif scheme == "top1_fp8":
                    pkg = compress_lora_delta(delta, f"w{k}", "bench", step, inner_steps, topk_ratio=0.01)
                    sent.append(decompress_lora_delta(pkg)[PARAM_NAME])
                    wire_bytes += pkg.compressed_bytes
                else:
                    pkg = buffers[k].compress(delta, f"w{k}", "bench", step, inner_steps)
                    sent.append(decompress_lora_delta(pkg)[PARAM_NAME])
                    wire_bytes += pkg.compressed_bytes
            w = w - outer_lr * torch.stack(sent).mean(dim=0)
            if step % log_every == 0 or step == outer_rounds:
                curve.append({"outer_round": step, "holdout_mse": round(_mse(w, problem.holdout_x, problem.holdout_y), 6)})
        results[scheme] = {
            "final_holdout_mse": round(_mse(w, problem.holdout_x, problem.holdout_y), 6),
            "wire_bytes_all_workers": int(wire_bytes),
            "curve": curve,
        }

    report: Dict[str, Any] = {
        "benchmark": "HM-S6-01 synthetic error-feedback compression",
        "scope": (
            "Synthetic linear least-squares on CPU; non-IID feature scaling per worker. "
            "Not a language model, not a real network, no timing."
        ),
        "config": {
            "seed": seed,
            "num_workers": num_workers,
            "outer_rounds": outer_rounds,
            "inner_steps": inner_steps,
            "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "parameters": int(w0.numel()),
        },
        "initial_holdout_mse": round(_mse(w0, problem.holdout_x, problem.holdout_y), 6),
        "noise_floor_mse": round(_mse(problem.w_true, problem.holdout_x, problem.holdout_y), 6),
        "results": results,
    }
    if "top5_fp16" in results and "top1_fp8_ef" in results:
        ref = results["top5_fp16"]
        ef = results["top1_fp8_ef"]
        report["comparison_top1_fp8_ef_vs_top5_fp16"] = {
            "holdout_mse_gap": round(ef["final_holdout_mse"] - ref["final_holdout_mse"], 6),
            "wire_bytes_reduction_x": round(ref["wire_bytes_all_workers"] / ef["wire_bytes_all_workers"], 3),
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-rounds", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_benchmark(seed=args.seed, outer_rounds=args.outer_rounds)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
