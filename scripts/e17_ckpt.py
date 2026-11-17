#!/usr/bin/env python3
"""Atomic, resumable training checkpoints for the QLoRA expert trainer (scripts/e17_train.py).

The GPU host is a Spot VM that Google can stop at any time, so a multi-hour adapter run must survive losing the machine
between two optimizer steps. A checkpoint holds everything the next step reads: the LoRA weights, the optimizer and
GradScaler state, the python and torch RNG states, and the loop position. Resuming from it reproduces the uninterrupted run.

A checkpoint is written to a temporary directory, flushed to disk, marked with a DONE file and only then renamed into place,
so a crash mid-write leaves either the previous checkpoint or an ignored temporary directory, never a half-written one.
Reading skips any checkpoint without its marker or that fails to load and falls back to the next older one.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch

PREFIX = "step-"
KEEP = 2
STATE_FILE = "state.pt"
DONE_FILE = "DONE"
COMPLETE_FILE = "TRAIN_COMPLETE"


def step_dir(root: Path, step: int) -> Path:
    return root / f"{PREFIX}{step:08d}"


def capture_rng() -> dict[str, Any]:
    """Python and torch (CPU and every CUDA device) generator states, as they are right now."""
    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def next_position(epoch: int, s: int, steps_per_epoch: int) -> tuple[int, int]:
    """(epoch, step-in-epoch) of the step that follows the one that has just finished at (epoch, s)."""
    if s + 1 >= steps_per_epoch:
        return epoch + 1, 0
    return epoch, s + 1


def fingerprint(*, base: str, pairs_file: Path, n_data: int, epochs: int, lr: float, accum: int, warmup: int, max_len: int, seed: int) -> dict[str, Any]:
    """What must be identical between the run that wrote a checkpoint and the run that resumes from it."""
    sha = hashlib.sha256(Path(pairs_file).read_bytes()).hexdigest()
    return {"base": base, "pairs_sha256": sha, "n_data": n_data, "epochs": epochs, "lr": lr, "accum": accum,
            "warmup": warmup, "max_len": max_len, "seed": seed}


def save_checkpoint(root: Path, step: int, payload: dict[str, Any], keep: int = KEEP) -> Path:
    """Write `payload` as checkpoint `step` under `root` atomically and keep only the newest `keep` checkpoints."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    root.mkdir(parents=True, exist_ok=True)
    final = step_dir(root, step)
    tmp = root / f".tmp-{step:08d}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with (tmp / STATE_FILE).open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    (tmp / DONE_FILE).write_text(str(step), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    for old in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(PREFIX))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def latest_checkpoint(root: Path) -> tuple[int, dict[str, Any]] | None:
    """Newest complete, loadable checkpoint as (step, payload), or None. Unfinished or corrupt ones are skipped."""
    if not root.is_dir():
        return None
    for cand in sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith(PREFIX)), reverse=True):
        if not (cand / DONE_FILE).is_file():
            continue
        try:
            payload = torch.load(cand / STATE_FILE, map_location="cpu", weights_only=True)
            return int(cand.name[len(PREFIX):]), payload
        except Exception:  # truncated or corrupt file: fall back to the next older checkpoint
            continue
    return None


def mark_complete(out_dir: Path, info: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COMPLETE_FILE).write_text(json.dumps(info, sort_keys=True), encoding="utf-8")


def is_complete(out_dir: Path) -> bool:
    return (out_dir / COMPLETE_FILE).is_file()
