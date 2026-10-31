#!/usr/bin/env python3
"""Qwen3.5 QLoRA run for the T4: the 0.8B smoke test first, then the 4B night run.

Resolves the base from configs/base_models_s21.json at its pinned revision, downloads that snapshot, and hands the pairs
to scripts/e17_train.py (nf4 base, LoRA on all linear layers, assistant-token loss). The GPU run is refused unless
--i-confirm-serving-stopped is given, because the T4 must not serve inference while it trains.

Usage: python qwen35_qlora_train.py --size 0.8B --pairs pairs.jsonl --out DIR [--smoke] [--dry-run]
       python qwen35_qlora_train.py --size 4B --pairs pairs.jsonl --out DIR --i-confirm-serving-stopped
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "configs" / "base_models_s21.json"
E17_TRAIN = Path(__file__).resolve().parent / "e17_train.py"
MIN_RECORDS = 50
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SIZE_TO_MODEL_ID = {"0.8B": "Qwen/Qwen3.5-0.8B", "4B": "Qwen/Qwen3.5-4B"}


def resolve_base(model_id: str, catalogue: Path = CATALOGUE) -> tuple[str, str]:
    """Return (model_id, pinned revision SHA) from the catalogue; refuse an unknown model or an unpinned revision."""
    data = json.loads(catalogue.read_text(encoding="utf-8"))
    for entry in data.get("models", []):
        if entry.get("model_id") == model_id:
            sha = str(entry.get("revision_sha", ""))
            if not SHA_RE.match(sha):
                raise ValueError(f"{model_id}: no pinned 40-hex revision_sha in {catalogue.name}")
            return model_id, sha
    raise ValueError(f"{model_id} is not in {catalogue.name}")


def load_pairs(path: Path, min_records: int = MIN_RECORDS) -> list[dict[str, Any]]:
    """Read JSONL pairs; each record needs a non-empty messages list and a non-empty completion."""
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        messages = rec.get("messages")
        completion = rec.get("completion")
        if not isinstance(messages, list) or not messages or not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{path.name}:{lineno}: each record needs a non-empty messages list and a non-empty completion")
        records.append(rec)
    if len(records) < min_records:
        raise ValueError(f"{len(records)} records; at least {min_records} are needed to train")
    return records


def training_plan(n_records: int, accum: int, epochs: int) -> dict[str, int]:
    """Optimizer steps per epoch and in total, as scripts/e17_train.py counts them."""
    if n_records <= 0 or accum <= 0 or epochs <= 0:
        raise ValueError("n_records, accum and epochs must be positive")
    steps_per_epoch = math.ceil(n_records / accum)
    return {"steps_per_epoch": steps_per_epoch, "total_steps": steps_per_epoch * epochs}


def fetch_snapshot(model_id: str, revision: str, cache_dir: Path) -> Path:
    """Download the pinned snapshot (public, not gated) and return its local directory."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_id, revision=revision, cache_dir=str(cache_dir)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5 QLoRA run (see module docstring).")
    ap.add_argument("--size", choices=sorted(SIZE_TO_MODEL_ID), required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "hivemind-qwen35")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true", help="one epoch capped at 15 minutes: the first GPU run on the 0.8B base")
    ap.add_argument("--i-confirm-serving-stopped", action="store_true", help="required for a GPU run: the T4 must not serve inference")
    ap.add_argument("--dry-run", action="store_true", help="validate the inputs and print the plan; no download, no GPU")
    args = ap.parse_args(argv)

    model_id, revision = resolve_base(SIZE_TO_MODEL_ID[args.size])
    records = load_pairs(args.pairs)
    epochs = 1 if args.smoke else args.epochs
    max_hours = 0.25 if args.smoke else args.max_hours
    plan = training_plan(len(records), args.accum, epochs)
    print(json.dumps({"base": model_id, "revision": revision, "records": len(records), "epochs": epochs,
                      "max_hours": max_hours, **plan}), flush=True)
    if args.dry_run:
        return 0
    if not args.i_confirm_serving_stopped:
        print("refusing to train: confirm the T4 is not serving inference with --i-confirm-serving-stopped", file=sys.stderr)
        return 4

    snapshot = fetch_snapshot(model_id, revision, args.cache_dir)
    cmd = [sys.executable, str(E17_TRAIN), "--base", str(snapshot), "--pairs", str(args.pairs), "--out", str(args.out),
           "--epochs", str(epochs), "--lr", str(args.lr), "--accum", str(args.accum), "--max-len", str(args.max_len),
           "--max-hours", str(max_hours)]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
