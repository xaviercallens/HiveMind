# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Fail-closed guard against loading 0-key / uninitialized Crowd-v1 safetensors files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_worker_module():
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location(
        "gcp_t4_crowd_diloco_worker_failclosed",
        ROOT / "scripts" / "gcp_t4_crowd_diloco_worker.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        raw = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(chunks))


def test_stream_safetensors_refuses_zero_matched_keys(tmp_path: Path) -> None:
    """Loading a raw 1-key blob ('weights') with 0 matching Crowd-v1 keys must fail closed."""
    w = _load_worker_module()
    model, _ = w._load_or_build_model(cache_dir=tmp_path, use_small_arch=True)
    blob = tmp_path / "raw_one_key.safetensors"
    _write_safetensors(blob, {"weights": torch.zeros(4, dtype=torch.float32)})
    assert w._safetensors_key_count(blob) == 1

    with pytest.raises(RuntimeError, match="Refusing uninitialized Crowd-v1 load"):
        w._stream_safetensors_no_mmap(model, blob, require_match=True)


def test_stream_safetensors_accepts_and_reports_matched_keys(tmp_path: Path) -> None:
    """Multi-key matching checkpoint loads cleanly and reports matched parameter keys."""
    w = _load_worker_module()
    model, _ = w._load_or_build_model(cache_dir=tmp_path, use_small_arch=True)
    target_key = next(iter(dict(model.named_parameters()).keys()))
    target_param = dict(model.named_parameters())[target_key]
    good = tmp_path / "model_named.safetensors"
    _write_safetensors(
        good,
        {
            target_key: torch.ones_like(target_param, dtype=torch.float32) * 0.25,
            "extra_unused": torch.zeros(2, dtype=torch.float32),
        },
    )
    assert w._safetensors_key_count(good) == 2

    sha, matched = w._stream_safetensors_no_mmap(model, good, require_match=True)
    assert len(sha) == 64
    assert matched == {target_key}
    assert torch.allclose(target_param.detach().cpu().float(), torch.full_like(target_param.float(), 0.25))
