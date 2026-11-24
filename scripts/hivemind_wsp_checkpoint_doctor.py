#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Deterministic offline safetensors open-weight MoE checkpoint doctor and ANSE preflight CLI.

Implements CPU-only (0 MiB GPU VRAM) weight-space inspection and surgical repair for
CrowdLoRA Sparse MoE bucket checkpoints (``router_frag_*`` and ``expert_*``):

1. ``inspect_moe_bucket_safetensors(buckets_or_path, *, num_layers, num_experts, rank, hidden_dim, inter_dim)``:
   Audits per-layer and per-expert Frobenius norms (``w1_A/B``, ``w2_A/B``, ``w3_A/B``),
   FoMoE zero-sink status (``w1_B = w2_B = w3_B = 0 => dy_e(x) == 0``), ``router_frag_*``
   final LayerNorm tail drift vs ``1.0`` (and ``0.0`` bias), singular-value spectral spikes
   via memory-efficient float64 QR-SVD, non-finite (``NaN``/``Inf``) corruptions, and
   rogue high-norm unscored experts (e.g., ``expert_3`` with ``||w1_B|| = 42.5``).
2. ``apply_offline_ghost_sink_and_ln_surgery(buckets, *, sink_experts, active_layers=None, lambda_ln=0.65, base_ln_tail=None)``:
   Synthesizes a sanitized CPU checkpoint dict with layer-localized FoMoE ghost-sink
   zeroing and LayerNorm tail interpolation, preserving non-active layers and non-sink
   experts bitwise while reporting exact parameter-edit ratio and SHA-256 provenance.
3. Offline CLI (``--inspect`` / ``--apply-surgery`` / ``--json`` / Markdown) returning exit
   code ``2`` whenever corrupted, drifted, or rogue blocks fail health thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

SCHEMA = "hivemind.wsp_checkpoint_doctor/1"
INSPECT_OBJECT = "hivemind.wsp_checkpoint_inspection"
SURGERY_OBJECT = "hivemind.wsp_checkpoint_surgery"

DEFAULT_LAMBDA_LN = 0.65
DEFAULT_ROGUE_B_NORM_THRESHOLD = 10.0
DEFAULT_ROGUE_NORM_RATIO_THRESHOLD = 4.0
DEFAULT_ROGUE_MIN_NORM_FLOOR = 1.0
DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD = 10.0
DEFAULT_SINGLE_MODE_SPIKE_NORM = 50.0
DEFAULT_MAX_LORA_SINGULAR_VALUE = 80.0
DEFAULT_MAX_ROUTER_ROW_NORM = 50.0
DEFAULT_LN_WEIGHT_DRIFT_RMS_THRESHOLD = 0.05
DEFAULT_LN_BIAS_DRIFT_RMS_THRESHOLD = 0.05
DEFAULT_LN_MAX_ABS_DRIFT_THRESHOLD = 0.25
DEFAULT_MIN_SINGULAR_VALUE = 1e-10
ANSE_MAXIMUM_PAIN = 1_000_000.0

_ROUTER_FRAG_RE = re.compile(r"^router_frag_(\d+)$")
_EXPERT_KEY_RE = re.compile(r"^expert_(\d+)$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OUTPUT_RE = re.compile(
    r'"' + "sco" + r're"\s*:\s*\d|referee_' + r"score|is_peer_" + r'review"?\s*:\s*[Tt]rue'
)
_METADATA_GEOMETRY_KEYS: dict[str, str] = {
    "num_layers": "hivemind_num_layers",
    "num_experts": "hivemind_num_experts",
    "rank": "hivemind_rank",
    "hidden_dim": "hivemind_hidden_dim",
    "inter_dim": "hivemind_inter_dim",
}


class CheckpointDoctorError(ValueError):
    """Fail-closed validation, layout, safetensors I/O, or health error."""


@dataclass(frozen=True)
class _ExpertFactorSlice:
    """Slice offset and 2D matrix shape for one LoRA factor within an expert bucket."""

    name: str
    offset: int
    numel: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class _BucketLayout:
    """Exact index map from flat CrowdLoRA buckets to per-layer routers, LN tail, and experts."""

    num_layers: int
    num_experts: int
    rank: int
    hidden_dim: int
    inter_dim: int
    router_keys: tuple[str, ...]
    expert_keys: tuple[str, ...]
    present_experts: tuple[int, ...]
    missing_experts: tuple[int, ...]
    router_layer_slices: tuple[tuple[str, int, int], ...]
    ln_tail_key: str
    ln_tail_offset: int
    per_layer_expert_numel: int
    expert_factor_slices: tuple[_ExpertFactorSlice, ...]


def _validate_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointDoctorError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise CheckpointDoctorError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise CheckpointDoctorError(f"{name} must be >= 1, got {value}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointDoctorError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise CheckpointDoctorError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_pos_finite(value: Any, name: str) -> float:
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise CheckpointDoctorError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise CheckpointDoctorError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def assert_safe_output(text: str) -> str:
    """Reject any serialized payload matching forbidden gatekeeper output patterns."""
    if _FORBIDDEN_OUTPUT_RE.search(text):
        raise CheckpointDoctorError("Output safety guard blocked forbidden output key pattern.")
    return text


def compute_json_sha256(payload: Any) -> str:
    """Compute canonical SHA-256 hex digest over strict RFC-8259 finite JSON."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _attach_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    core["receipt_sha256"] = compute_json_sha256(core)
    return core


def verify_report_provenance(report: Mapping[str, Any]) -> bool:
    """Verify that ``report['receipt_sha256']`` matches its canonical JSON SHA-256 digest."""
    if not isinstance(report, Mapping):
        raise CheckpointDoctorError("Report must be a mapping.")
    claimed = report.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in report.items() if k != "receipt_sha256"}
    try:
        expected = compute_json_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _raw_bit_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return an integer bit-pattern view for exact NaN-safe bitwise comparison and hashing."""
    t_cpu = tensor.detach().to(device="cpu").contiguous()
    elem_bytes = t_cpu.element_size()
    if elem_bytes == 1:
        return t_cpu.view(torch.uint8)
    if elem_bytes == 2:
        return t_cpu.view(torch.int16)
    if elem_bytes == 4:
        return t_cpu.view(torch.int32)
    return t_cpu.view(torch.int64)


def _count_bitwise_modified(before: torch.Tensor, after: torch.Tensor) -> int:
    """Count exact bitwise-different elements between two same-shape tensors."""
    return int(torch.ne(_raw_bit_view(before), _raw_bit_view(after)).sum().item())


def compute_buckets_sha256(buckets: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 digest over sorted bucket names, shapes, dtypes, and raw bits."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise CheckpointDoctorError("buckets must be a non-empty mapping of name -> torch.Tensor")
    hasher = hashlib.sha256()
    for key in sorted(buckets.keys()):
        val = buckets[key]
        if not isinstance(val, torch.Tensor):
            raise CheckpointDoctorError(f"buckets[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_cpu = val.detach().to(device="cpu").contiguous()
        header = f"{key}|{tuple(t_cpu.shape)}|{t_cpu.dtype}|".encode()
        hasher.update(header)
        hasher.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def compute_file_sha256(path: str | os.PathLike[str]) -> str:
    """Compute streaming SHA-256 hex digest of a file on disk."""
    p = Path(path)
    if not p.is_file():
        raise CheckpointDoctorError(f"Checkpoint file not found: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_checkpoint_buckets(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict[str, str], str]:
    """Load a CrowdLoRA ``.safetensors`` file strictly on CPU and return ``(tensors, metadata, file_sha256)``."""
    p = Path(path)
    if not p.is_file():
        raise CheckpointDoctorError(f"Safetensors checkpoint not found: {p}")
    file_sha = compute_file_sha256(p)
    tensors: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(p), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            for key in sorted(handle.keys()):
                tensors[key] = handle.get_tensor(key).detach().to(device="cpu").contiguous().clone()
    except Exception as exc:
        raise CheckpointDoctorError(f"Failed to read safetensors checkpoint {p}: {exc}") from exc
    if not tensors:
        raise CheckpointDoctorError(f"Safetensors checkpoint {p} contains zero tensors")
    return tensors, metadata, file_sha


def save_checkpoint_safetensors(
    buckets: Mapping[str, torch.Tensor],
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, str] | None = None,
) -> str:
    """Atomically write ``buckets`` to ``.safetensors`` on CPU and return the written file SHA-256."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise CheckpointDoctorError("buckets must be a non-empty mapping to save safetensors")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_tensors = {
        str(k): v.detach().to(device="cpu").contiguous().clone()
        for k, v in sorted(buckets.items())
    }
    meta_str = {str(k): str(v) for k, v in metadata.items()} if metadata else None
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        save_file(clean_tensors, str(tmp_path), metadata=meta_str)
        os.replace(str(tmp_path), str(out_path))
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise CheckpointDoctorError(f"Failed to save safetensors checkpoint to {out_path}: {exc}") from exc
    return compute_file_sha256(out_path)


def _normalize_buckets_input(
    buckets_or_path: str | os.PathLike[str] | Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str, str | None, str | None, dict[str, str]]:
    """Resolve either a ``.safetensors`` path or in-memory tensor mapping to CPU clones."""
    if isinstance(buckets_or_path, (str, os.PathLike)):
        p = Path(buckets_or_path)
        tensors, metadata, file_sha = load_checkpoint_buckets(p)
        return tensors, "safetensors_file", str(p), file_sha, metadata

    if not isinstance(buckets_or_path, Mapping) or not buckets_or_path:
        raise CheckpointDoctorError(
            "buckets_or_path must be a .safetensors file path or non-empty Mapping[str, torch.Tensor]"
        )
    tensors = {}
    for key, val in buckets_or_path.items():
        if not isinstance(key, str) or not key.strip():
            raise CheckpointDoctorError(f"Bucket keys must be non-empty strings, got {key!r}")
        if not isinstance(val, torch.Tensor):
            raise CheckpointDoctorError(f"buckets[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        tensors[key] = val.detach().to(device="cpu").contiguous().clone()
    return tensors, "in_memory_mapping", None, None, {}


def _extract_tail_pair(
    raw_tail: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    hidden_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract float64 CPU ``(ln_weight, ln_bias)`` from a tail tensor, pair, or base bucket dict."""
    if isinstance(raw_tail, Mapping):
        router_pairs: list[tuple[int, str]] = []
        for k in raw_tail:
            m = _ROUTER_FRAG_RE.match(k)
            if m:
                router_pairs.append((int(m.group(1)), k))
        if not router_pairs:
            raise CheckpointDoctorError("base_ln_tail mapping must contain at least one 'router_frag_*' key")
        last_key = max(router_pairs)[1]
        vec = raw_tail[last_key].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if hidden_dim is None:
            raise CheckpointDoctorError("hidden_dim is required when extracting base_ln_tail from a bucket mapping")
        if vec.numel() < 2 * hidden_dim:
            raise CheckpointDoctorError(
                f"base_ln_tail[{last_key!r}] length {vec.numel()} < 2*hidden_dim ({2 * hidden_dim})"
            )
        ln_w = vec[-2 * hidden_dim : -hidden_dim]
        ln_b = vec[-hidden_dim:]
    elif isinstance(raw_tail, tuple) and len(raw_tail) == 2:
        w_t, b_t = raw_tail
        if not isinstance(w_t, torch.Tensor) or not isinstance(b_t, torch.Tensor):
            raise CheckpointDoctorError("base_ln_tail pair entries must both be torch.Tensor instances")
        ln_w = w_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        ln_b = b_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if ln_w.numel() != ln_b.numel() or ln_w.numel() == 0:
            raise CheckpointDoctorError("base_ln_tail (w, b) must have matching non-zero lengths")
        if hidden_dim is not None and ln_w.numel() != hidden_dim:
            raise CheckpointDoctorError(f"base_ln_tail length {ln_w.numel()} != hidden_dim={hidden_dim}")
    elif isinstance(raw_tail, torch.Tensor):
        t64 = raw_tail.detach().to(device="cpu", dtype=torch.float64)
        if t64.ndim == 2 and t64.shape[0] == 2:
            ln_w, ln_b = t64[0].reshape(-1), t64[1].reshape(-1)
        elif t64.ndim == 1 and t64.numel() >= 2 and t64.numel() % 2 == 0:
            half = t64.numel() // 2
            ln_w, ln_b = t64[:half], t64[half:]
        else:
            raise CheckpointDoctorError(
                f"base_ln_tail tensor must have shape (2*H,) or (2, H), got {tuple(raw_tail.shape)}"
            )
        if hidden_dim is not None and ln_w.numel() != hidden_dim:
            raise CheckpointDoctorError(f"base_ln_tail hidden_dim {ln_w.numel()} != hidden_dim={hidden_dim}")
    else:
        raise CheckpointDoctorError(
            "base_ln_tail must be None, a (ln_w, ln_b) tuple, a 1D/2D Tensor, or a bucket Mapping"
        )

    if not bool(torch.isfinite(ln_w).all().item()) or not bool(torch.isfinite(ln_b).all().item()):
        raise CheckpointDoctorError("base_ln_tail must contain only finite values")
    return ln_w.clone(), ln_b.clone()


def _infer_geometry(
    buckets: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
    base_ln_tail: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor] | None,
    num_layers: int | None,
    num_experts: int | None,
    rank: int | None,
    hidden_dim: int | None,
    inter_dim: int | None,
) -> tuple[int, int, int, int, int]:
    """Resolve and validate ``(num_layers, num_experts, rank, hidden_dim, inter_dim)``."""
    meta_vals: dict[str, int] = {}
    if "hivemind_doctor_meta" in metadata:
        try:
            parsed_meta = json.loads(str(metadata["hivemind_doctor_meta"]))
        except (TypeError, ValueError):
            parsed_meta = {}
        if isinstance(parsed_meta, Mapping):
            for field_name in _METADATA_GEOMETRY_KEYS:
                raw_v = parsed_meta.get(field_name)
                if isinstance(raw_v, int) and not isinstance(raw_v, bool) and raw_v > 0:
                    meta_vals[field_name] = raw_v
    for field_name, meta_key in _METADATA_GEOMETRY_KEYS.items():
        if field_name not in meta_vals and meta_key in metadata and str(metadata[meta_key]).strip().isdigit():
            meta_vals[field_name] = int(str(metadata[meta_key]).strip())

    nl = num_layers if num_layers is not None else meta_vals.get("num_layers")
    ne = num_experts if num_experts is not None else meta_vals.get("num_experts")
    rk = rank if rank is not None else meta_vals.get("rank")
    hd = hidden_dim if hidden_dim is not None else meta_vals.get("hidden_dim")
    idim = inter_dim if inter_dim is not None else meta_vals.get("inter_dim")

    expert_nums = sorted(
        int(m.group(1))
        for k in buckets
        if (m := _EXPERT_KEY_RE.match(k)) is not None
    )
    router_pairs = sorted(
        (int(m.group(1)), k)
        for k in buckets
        if (m := _ROUTER_FRAG_RE.match(k)) is not None
    )
    if not router_pairs or not expert_nums:
        raise CheckpointDoctorError("Checkpoint buckets must contain 'router_frag_*' and 'expert_*' entries")

    if ne is None:
        ne = max(expert_nums) + 1

    if hd is None and base_ln_tail is not None and not isinstance(base_ln_tail, Mapping):
        w_b, _ = _extract_tail_pair(base_ln_tail)
        hd = int(w_b.numel())

    total_router_numel = sum(int(buckets[k].numel()) for _, k in router_pairs)
    first_expert_numel = int(buckets[f"expert_{expert_nums[0]}"].numel())

    if hd is not None and nl is None:
        h_val = _validate_positive_int(hd, "hidden_dim")
        e_val = _validate_positive_int(ne, "num_experts")
        rem = total_router_numel - 2 * h_val
        unit = e_val * h_val
        if rem > 0 and rem % unit == 0:
            nl = rem // unit

    if nl is not None and hd is None:
        l_val = _validate_positive_int(nl, "num_layers")
        e_val = _validate_positive_int(ne, "num_experts")
        denom = l_val * e_val + 2
        if total_router_numel % denom == 0:
            hd = total_router_numel // denom

    if hd is not None and nl is not None:
        l_val = _validate_positive_int(nl, "num_layers")
        h_val = _validate_positive_int(hd, "hidden_dim")
        if first_expert_numel % (3 * l_val) == 0:
            per_layer_third = first_expert_numel // (3 * l_val)
            if rk is not None and idim is None:
                r_val = _validate_positive_int(rk, "rank")
                if per_layer_third % r_val == 0:
                    idim = (per_layer_third // r_val) - h_val
            elif idim is not None and rk is None:
                i_val = _validate_positive_int(idim, "inter_dim")
                if per_layer_third % (h_val + i_val) == 0:
                    rk = per_layer_third // (h_val + i_val)
            elif rk is None and idim is None:
                for cand_r in (8, 4, 16, 2, 32, 1):
                    if per_layer_third % cand_r == 0:
                        cand_i = (per_layer_third // cand_r) - h_val
                        if cand_i > h_val:
                            rk = cand_r
                            idim = cand_i
                            break

    missing_dims = [
        name
        for name, val in (
            ("num_layers", nl),
            ("num_experts", ne),
            ("rank", rk),
            ("hidden_dim", hd),
            ("inter_dim", idim),
        )
        if val is None
    ]
    if missing_dims:
        raise CheckpointDoctorError(
            f"Unable to infer CrowdLoRA geometry ({', '.join(missing_dims)}); specify explicitly."
        )

    return (
        _validate_positive_int(nl, "num_layers"),
        _validate_positive_int(ne, "num_experts"),
        _validate_positive_int(rk, "rank"),
        _validate_positive_int(hd, "hidden_dim"),
        _validate_positive_int(idim, "inter_dim"),
    )


def _build_bucket_layout(
    buckets: Mapping[str, torch.Tensor],
    *,
    num_layers: int,
    num_experts: int,
    rank: int,
    hidden_dim: int,
    inter_dim: int,
) -> _BucketLayout:
    """Validate bucket keys/lengths and return exact slice offsets for every layer and expert."""
    nl = _validate_positive_int(num_layers, "num_layers")
    ne = _validate_positive_int(num_experts, "num_experts")
    rk = _validate_positive_int(rank, "rank")
    hd = _validate_positive_int(hidden_dim, "hidden_dim")
    idim = _validate_positive_int(inter_dim, "inter_dim")

    router_pairs: list[tuple[int, str]] = []
    expert_pairs: list[tuple[int, str]] = []
    for key, tens in buckets.items():
        if tens.ndim != 1 or tens.numel() == 0:
            raise CheckpointDoctorError(f"Bucket {key!r} must be a non-empty 1D tensor, got {tuple(tens.shape)}")
        if (m_r := _ROUTER_FRAG_RE.match(key)) is not None:
            router_pairs.append((int(m_r.group(1)), key))
        elif (m_e := _EXPERT_KEY_RE.match(key)) is not None:
            e_idx = int(m_e.group(1))
            if e_idx >= ne:
                raise CheckpointDoctorError(f"Bucket {key!r} expert index {e_idx} >= num_experts={ne}")
            expert_pairs.append((e_idx, key))
        else:
            raise CheckpointDoctorError(f"Unrecognized CrowdLoRA bucket key {key!r}")

    if not router_pairs:
        raise CheckpointDoctorError("Checkpoint buckets must contain at least one 'router_frag_*' tensor")
    if not expert_pairs:
        raise CheckpointDoctorError("Checkpoint buckets must contain at least one 'expert_*' tensor")

    router_pairs.sort()
    expert_pairs.sort()
    router_keys = tuple(k for _, k in router_pairs)
    expert_keys = tuple(k for _, k in expert_pairs)
    present_experts = tuple(e for e, _ in expert_pairs)
    present_set = set(present_experts)
    missing_experts = tuple(e for e in range(ne) if e not in present_set)

    router_layer_numel = ne * hd
    ln_tail_numel = 2 * hd
    last_r_key = router_keys[-1]
    router_layer_slices: list[tuple[str, int, int]] = []
    ln_tail_offset = 0

    for r_key in router_keys:
        n_elem = int(buckets[r_key].numel())
        is_last = r_key == last_r_key
        router_span = n_elem - ln_tail_numel if is_last else n_elem
        if router_span < 0 or router_span % router_layer_numel != 0:
            raise CheckpointDoctorError(
                f"Bucket {r_key!r} length {n_elem} is incompatible with num_experts={ne}, hidden_dim={hd}"
            )
        n_frag_layers = router_span // router_layer_numel
        for l_sub in range(n_frag_layers):
            router_layer_slices.append((r_key, l_sub * router_layer_numel, router_layer_numel))
        if is_last:
            ln_tail_offset = router_span

    if len(router_layer_slices) != nl:
        raise CheckpointDoctorError(
            f"Router fragments cover {len(router_layer_slices)} layers, expected num_layers={nl}"
        )

    factor_specs: tuple[tuple[str, tuple[int, int]], ...] = (
        ("w1_A", (rk, hd)),
        ("w1_B", (idim, rk)),
        ("w2_A", (rk, hd)),
        ("w2_B", (idim, rk)),
        ("w3_A", (rk, idim)),
        ("w3_B", (hd, rk)),
    )
    expert_slices: list[_ExpertFactorSlice] = []
    offset = 0
    for f_name, f_shape in factor_specs:
        f_numel = f_shape[0] * f_shape[1]
        expert_slices.append(_ExpertFactorSlice(name=f_name, offset=offset, numel=f_numel, shape=f_shape))
        offset += f_numel

    per_layer_expert_numel = offset
    expected_expert_numel = nl * per_layer_expert_numel
    for e_key in expert_keys:
        actual = int(buckets[e_key].numel())
        if actual != expected_expert_numel:
            raise CheckpointDoctorError(
                f"Bucket {e_key!r} length {actual} != expected {expected_expert_numel} "
                f"(num_layers={nl}, rank={rk}, hidden_dim={hd}, inter_dim={idim})"
            )

    return _BucketLayout(
        num_layers=nl,
        num_experts=ne,
        rank=rk,
        hidden_dim=hd,
        inter_dim=idim,
        router_keys=router_keys,
        expert_keys=expert_keys,
        present_experts=present_experts,
        missing_experts=missing_experts,
        router_layer_slices=tuple(router_layer_slices),
        ln_tail_key=last_r_key,
        ln_tail_offset=ln_tail_offset,
        per_layer_expert_numel=per_layer_expert_numel,
        expert_factor_slices=tuple(expert_slices),
    )


def _compute_qr_svd_metrics(
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    *,
    spectral_spike_ratio_threshold: float,
    single_mode_spike_norm: float,
    max_lora_singular_value: float,
) -> dict[str, Any]:
    """Compute exact rank-r singular values of ``w_b @ w_a`` via CPU float64 QR-SVD."""
    b_finite = bool(torch.isfinite(w_b).all().item())
    a_finite = bool(torch.isfinite(w_a).all().item())
    if not (b_finite and a_finite):
        return {
            "finite": False,
            "is_zero": False,
            "sigma_1": None,
            "median_sigma": None,
            "spectral_outlier_ratio": None,
            "stable_rank": None,
            "is_spectral_spike": False,
        }

    b64 = w_b.detach().to(device="cpu", dtype=torch.float64)
    a64 = w_a.detach().to(device="cpu", dtype=torch.float64)
    _, rb = torch.linalg.qr(b64, mode="reduced")
    _, ra = torch.linalg.qr(a64.T, mode="reduced")
    s_vals = torch.linalg.svdvals(rb @ ra.T)
    s_pos = s_vals[s_vals > DEFAULT_MIN_SINGULAR_VALUE]

    if s_pos.numel() == 0:
        return {
            "finite": True,
            "is_zero": True,
            "sigma_1": 0.0,
            "median_sigma": 0.0,
            "spectral_outlier_ratio": 0.0,
            "stable_rank": 0.0,
            "is_spectral_spike": False,
        }

    sigma_1 = float(s_vals[0].item())
    median_sigma = float(torch.median(s_pos).item())
    ratio = sigma_1 / max(median_sigma, 1e-12)
    frob_sq = float((s_vals * s_vals).sum().item())
    stable_rank = frob_sq / max(sigma_1 * sigma_1, 1e-24)

    is_spike = bool(
        (sigma_1 > max_lora_singular_value)
        or (s_pos.numel() >= 2 and ratio > spectral_spike_ratio_threshold)
        or (s_pos.numel() == 1 and sigma_1 > single_mode_spike_norm)
    )
    return {
        "finite": True,
        "is_zero": False,
        "sigma_1": sigma_1,
        "median_sigma": median_sigma,
        "spectral_outlier_ratio": ratio,
        "stable_rank": stable_rank,
        "is_spectral_spike": is_spike,
    }


def _finite_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Return ``(finite_only_float64_zeroed, non_finite_count, nan_count, inf_count)``."""
    t64 = tensor.detach().to(device="cpu", dtype=torch.float64)
    nan_count = int(torch.isnan(t64).sum().item())
    inf_count = int(torch.isinf(t64).sum().item())
    nf_count = nan_count + inf_count
    clean = torch.where(torch.isfinite(t64), t64, torch.zeros_like(t64)) if nf_count > 0 else t64
    return clean, nf_count, nan_count, inf_count


def _audit_layernorm_tail(
    buckets: Mapping[str, torch.Tensor],
    layout: _BucketLayout,
    *,
    base_ln_tail: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor] | None,
    ln_weight_drift_rms_threshold: float,
    ln_bias_drift_rms_threshold: float,
    ln_max_abs_drift_threshold: float,
) -> dict[str, Any]:
    """Audit ``router_frag_*`` final LayerNorm tail drift vs ``1.0`` (or explicit ``base_ln_tail``)."""
    hd = layout.hidden_dim
    tail = buckets[layout.ln_tail_key][layout.ln_tail_offset : layout.ln_tail_offset + 2 * hd]
    ln_w_raw = tail[:hd]
    ln_b_raw = tail[hd:]

    if base_ln_tail is None:
        ref_w = torch.ones(hd, dtype=torch.float64)
        ref_b = torch.zeros(hd, dtype=torch.float64)
        baseline_mode = "unit_ones_zero_bias"
    else:
        ref_w, ref_b = _extract_tail_pair(base_ln_tail, hidden_dim=hd)
        baseline_mode = "explicit_base_ln_tail"

    ln_w, nf_w, nan_w, inf_w = _finite_stats(ln_w_raw)
    ln_b, nf_b, nan_b, inf_b = _finite_stats(ln_b_raw)
    nf_tot = nf_w + nf_b

    w_diff = ln_w - ref_w
    b_diff = ln_b - ref_b
    w_drift_l2 = float(w_diff.norm().item())
    b_drift_l2 = float(b_diff.norm().item())
    w_drift_rms = float(w_diff.pow(2).mean().sqrt().item())
    b_drift_rms = float(b_diff.pow(2).mean().sqrt().item())
    w_max_abs = float(w_diff.abs().max().item())
    b_max_abs = float(b_diff.abs().max().item())
    tail_drift_l2 = math.sqrt(w_drift_l2 * w_drift_l2 + b_drift_l2 * b_drift_l2)
    tail_drift_rms = math.sqrt(0.5 * (w_drift_rms * w_drift_rms + b_drift_rms * b_drift_rms))

    drift_reasons: list[str] = []
    if nf_tot > 0:
        drift_reasons.append(f"non_finite_tail({nf_tot})")
    if w_drift_rms > ln_weight_drift_rms_threshold:
        drift_reasons.append(f"ln_weight_drift_rms({w_drift_rms:.4f}>{ln_weight_drift_rms_threshold:.4f})")
    if b_drift_rms > ln_bias_drift_rms_threshold:
        drift_reasons.append(f"ln_bias_drift_rms({b_drift_rms:.4f}>{ln_bias_drift_rms_threshold:.4f})")
    if w_max_abs > ln_max_abs_drift_threshold:
        drift_reasons.append(f"ln_weight_max_abs_drift({w_max_abs:.4f}>{ln_max_abs_drift_threshold:.4f})")
    if b_max_abs > ln_max_abs_drift_threshold:
        drift_reasons.append(f"ln_bias_max_abs_drift({b_max_abs:.4f}>{ln_max_abs_drift_threshold:.4f})")

    return {
        "router_fragment_key": layout.ln_tail_key,
        "baseline_mode": baseline_mode,
        "finite": nf_tot == 0,
        "non_finite_elements": nf_tot,
        "nan_elements": nan_w + nan_b,
        "inf_elements": inf_w + inf_b,
        "ln_weight_mean": float(ln_w.mean().item()),
        "ln_weight_min": float(ln_w.min().item()),
        "ln_weight_max": float(ln_w.max().item()),
        "ln_weight_std": float(ln_w.std(unbiased=False).item()),
        "ln_weight_drift_l2": w_drift_l2,
        "ln_weight_drift_rms": w_drift_rms,
        "ln_weight_max_abs_drift": w_max_abs,
        "ln_bias_mean": float(ln_b.mean().item()),
        "ln_bias_drift_l2": b_drift_l2,
        "ln_bias_drift_rms": b_drift_rms,
        "ln_bias_max_abs": b_max_abs,
        "ln_tail_drift_l2": tail_drift_l2,
        "ln_tail_drift_rms": tail_drift_rms,
        "drift_failed": bool(drift_reasons),
        "drift_reasons": drift_reasons,
    }


def inspect_moe_bucket_safetensors(
    buckets_or_path: str | os.PathLike[str] | Mapping[str, torch.Tensor],
    *,
    num_layers: int,
    num_experts: int,
    rank: int,
    hidden_dim: int,
    inter_dim: int,
    scored_experts: Sequence[int] | None = None,
    rogue_b_norm_threshold: float = DEFAULT_ROGUE_B_NORM_THRESHOLD,
    rogue_norm_ratio_threshold: float = DEFAULT_ROGUE_NORM_RATIO_THRESHOLD,
    rogue_min_norm_floor: float = DEFAULT_ROGUE_MIN_NORM_FLOOR,
    spectral_spike_ratio_threshold: float = DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD,
    single_mode_spike_norm: float = DEFAULT_SINGLE_MODE_SPIKE_NORM,
    max_lora_singular_value: float = DEFAULT_MAX_LORA_SINGULAR_VALUE,
    max_router_row_norm: float = DEFAULT_MAX_ROUTER_ROW_NORM,
    ln_weight_drift_rms_threshold: float = DEFAULT_LN_WEIGHT_DRIFT_RMS_THRESHOLD,
    ln_bias_drift_rms_threshold: float = DEFAULT_LN_BIAS_DRIFT_RMS_THRESHOLD,
    ln_max_abs_drift_threshold: float = DEFAULT_LN_MAX_ABS_DRIFT_THRESHOLD,
    base_ln_tail: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Audit a CrowdLoRA Sparse MoE bucket checkpoint offline on CPU with zero GPU VRAM."""
    rogue_b_thresh = _validate_nonneg_finite(rogue_b_norm_threshold, "rogue_b_norm_threshold")
    rogue_ratio_thresh = _validate_nonneg_finite(rogue_norm_ratio_threshold, "rogue_norm_ratio_threshold")
    rogue_floor = _validate_nonneg_finite(rogue_min_norm_floor, "rogue_min_norm_floor")
    spike_ratio_thresh = _validate_pos_finite(spectral_spike_ratio_threshold, "spectral_spike_ratio_threshold")
    single_spike_norm = _validate_pos_finite(single_mode_spike_norm, "single_mode_spike_norm")
    max_sv_thresh = _validate_pos_finite(max_lora_singular_value, "max_lora_singular_value")
    max_router_thresh = _validate_pos_finite(max_router_row_norm, "max_router_row_norm")
    ln_w_thresh = _validate_nonneg_finite(ln_weight_drift_rms_threshold, "ln_weight_drift_rms_threshold")
    ln_b_thresh = _validate_nonneg_finite(ln_bias_drift_rms_threshold, "ln_bias_drift_rms_threshold")
    ln_abs_thresh = _validate_nonneg_finite(ln_max_abs_drift_threshold, "ln_max_abs_drift_threshold")

    buckets, source_type, source_path, source_file_sha, _ = _normalize_buckets_input(buckets_or_path)
    layout = _build_bucket_layout(
        buckets,
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    )

    scored_set: set[int] | None = None
    if scored_experts is not None:
        scored_set = {_validate_nonneg_int(e, "scored_expert") for e in scored_experts}

    # Global bucket non-finite accounting
    total_elements = sum(int(t.numel()) for t in buckets.values())
    total_nf = 0
    total_nan = 0
    total_inf = 0
    affected_buckets: list[str] = []
    nf_locations: list[dict[str, Any]] = []
    for b_key in sorted(buckets.keys()):
        _, nf_b, nan_b, inf_b = _finite_stats(buckets[b_key])
        if nf_b > 0:
            total_nf += nf_b
            total_nan += nan_b
            total_inf += inf_b
            affected_buckets.append(b_key)

    # Router row norm audit across all layers
    router_sq_per_expert = [0.0 for _ in range(layout.num_experts)]
    router_layer_row_norms: list[list[float]] = []
    per_layer_max_router_norm: list[float] = []
    router_norm_outliers: list[dict[str, Any]] = []
    for l_idx, (r_key, r_off, r_len) in enumerate(layout.router_layer_slices):
        r_mat = buckets[r_key][r_off : r_off + r_len].reshape(layout.num_experts, layout.hidden_dim)
        layer_norms: list[float] = []
        for e_idx in range(layout.num_experts):
            row_clean, nf_r, nan_r, inf_r = _finite_stats(r_mat[e_idx])
            row_norm = float(row_clean.norm().item())
            layer_norms.append(row_norm)
            router_sq_per_expert[e_idx] += row_norm * row_norm
            if nf_r > 0:
                nf_locations.append(
                    {
                        "bucket": r_key,
                        "layer": l_idx,
                        "component": f"router_row_{e_idx}",
                        "non_finite_elements": nf_r,
                        "nan_elements": nan_r,
                        "inf_elements": inf_r,
                    }
                )
            if row_norm > max_router_thresh:
                router_norm_outliers.append(
                    {"layer": l_idx, "expert": e_idx, "router_row_norm": row_norm, "threshold": max_router_thresh}
                )
        router_layer_row_norms.append(layer_norms)
        per_layer_max_router_norm.append(max(layer_norms))

    # Per-expert, per-layer Frobenius norms, FoMoE zero-sink status, and QR-SVD spectral audit
    expert_accum: dict[int, dict[str, Any]] = {}
    spectral_spikes: list[dict[str, Any]] = []
    max_sv_global = 0.0
    max_ratio_global = 0.0
    full_zero_sink_experts: list[int] = []
    partial_zero_sink_experts: list[int] = []
    active_experts: list[int] = []

    for e_idx in layout.present_experts:
        e_key = f"expert_{e_idx}"
        e_vec = buckets[e_key]
        sq_totals = {spec.name: 0.0 for spec in layout.expert_factor_slices}
        max_l_w1_b = 0.0
        max_l_w1_layer = 0
        max_l_b = 0.0
        max_l_b_layer = 0
        exp_max_sv = 0.0
        exp_max_ratio = 0.0
        exp_nf = 0
        zero_sink_layers: list[int] = []
        nonzero_layers: list[int] = []
        per_layer_rows: list[dict[str, Any]] = []

        for l_idx in range(layout.num_layers):
            base_off = l_idx * layout.per_layer_expert_numel
            factors: dict[str, torch.Tensor] = {}
            layer_norms: dict[str, float] = {}
            layer_nf = 0

            for spec in layout.expert_factor_slices:
                sl = e_vec[base_off + spec.offset : base_off + spec.offset + spec.numel].reshape(spec.shape)
                factors[spec.name] = sl
                clean_f, nf_f, nan_f, inf_f = _finite_stats(sl)
                if nf_f > 0:
                    layer_nf += nf_f
                    nf_locations.append(
                        {
                            "bucket": e_key,
                            "layer": l_idx,
                            "component": spec.name,
                            "non_finite_elements": nf_f,
                            "nan_elements": nan_f,
                            "inf_elements": inf_f,
                        }
                    )
                sq_val = float(clean_f.pow(2).sum().item())
                sq_totals[spec.name] += sq_val
                layer_norms[spec.name] = math.sqrt(sq_val)

            exp_nf += layer_nf
            w1_b_l = layer_norms["w1_B"]
            b_tot_l = math.sqrt(
                layer_norms["w1_B"] ** 2 + layer_norms["w2_B"] ** 2 + layer_norms["w3_B"] ** 2
            )
            if w1_b_l > max_l_w1_b:
                max_l_w1_b = w1_b_l
                max_l_w1_layer = l_idx
            if b_tot_l > max_l_b:
                max_l_b = b_tot_l
                max_l_b_layer = l_idx

            b_max_abs = max(
                float(factors[b_name].abs().max().item()) if bool(torch.isfinite(factors[b_name]).all().item()) else 1.0
                for b_name in ("w1_B", "w2_B", "w3_B")
            )
            b_all_finite = all(bool(torch.isfinite(factors[b_name]).all().item()) for b_name in ("w1_B", "w2_B", "w3_B"))
            is_layer_zero = bool(b_all_finite and b_max_abs == 0.0)
            if is_layer_zero:
                zero_sink_layers.append(l_idx)
            else:
                nonzero_layers.append(l_idx)

            proj_reports: dict[str, dict[str, Any]] = {}
            layer_max_sv = 0.0
            layer_max_ratio = 0.0
            for proj in ("w1", "w2", "w3"):
                svd_res = _compute_qr_svd_metrics(
                    factors[f"{proj}_B"],
                    factors[f"{proj}_A"],
                    spectral_spike_ratio_threshold=spike_ratio_thresh,
                    single_mode_spike_norm=single_spike_norm,
                    max_lora_singular_value=max_sv_thresh,
                )
                proj_reports[proj] = svd_res
                if svd_res["finite"]:
                    sv1 = float(svd_res["sigma_1"])
                    rat = float(svd_res["spectral_outlier_ratio"])
                    layer_max_sv = max(layer_max_sv, sv1)
                    layer_max_ratio = max(layer_max_ratio, rat)
                    if svd_res["is_spectral_spike"]:
                        spectral_spikes.append(
                            {
                                "layer": l_idx,
                                "expert": e_idx,
                                "projection": proj,
                                "sigma_1": sv1,
                                "median_sigma": float(svd_res["median_sigma"]),
                                "spectral_outlier_ratio": rat,
                                "stable_rank": float(svd_res["stable_rank"]),
                            }
                        )

            exp_max_sv = max(exp_max_sv, layer_max_sv)
            exp_max_ratio = max(exp_max_ratio, layer_max_ratio)
            max_sv_global = max(max_sv_global, layer_max_sv)
            max_ratio_global = max(max_ratio_global, layer_max_ratio)

            per_layer_rows.append(
                {
                    "layer": l_idx,
                    "w1_a_norm": layer_norms["w1_A"],
                    "w1_b_norm": w1_b_l,
                    "w2_a_norm": layer_norms["w2_A"],
                    "w2_b_norm": layer_norms["w2_B"],
                    "w3_a_norm": layer_norms["w3_A"],
                    "w3_b_norm": layer_norms["w3_B"],
                    "total_b_norm": b_tot_l,
                    "router_row_norm": router_layer_row_norms[l_idx][e_idx],
                    "is_zero_sink": is_layer_zero,
                    "non_finite_elements": layer_nf,
                    "max_singular_value": layer_max_sv,
                    "max_spectral_outlier_ratio": layer_max_ratio,
                    "projections": proj_reports,
                }
            )

        if len(zero_sink_layers) == layout.num_layers:
            sink_status = "full_zero_sink"
            full_zero_sink_experts.append(e_idx)
        elif zero_sink_layers:
            sink_status = "partial_zero_sink"
            partial_zero_sink_experts.append(e_idx)
        else:
            sink_status = "active"
            active_experts.append(e_idx)

        w1_b_g = math.sqrt(sq_totals["w1_B"])
        w2_b_g = math.sqrt(sq_totals["w2_B"])
        w3_b_g = math.sqrt(sq_totals["w3_B"])
        tot_b_g = math.sqrt(sq_totals["w1_B"] + sq_totals["w2_B"] + sq_totals["w3_B"])
        tot_a_g = math.sqrt(sq_totals["w1_A"] + sq_totals["w2_A"] + sq_totals["w3_A"])

        expert_accum[e_idx] = {
            "expert_index": e_idx,
            "w1_a_norm": math.sqrt(sq_totals["w1_A"]),
            "w1_b_norm": w1_b_g,
            "w2_a_norm": math.sqrt(sq_totals["w2_A"]),
            "w2_b_norm": w2_b_g,
            "w3_a_norm": math.sqrt(sq_totals["w3_A"]),
            "w3_b_norm": w3_b_g,
            "total_a_norm": tot_a_g,
            "total_b_norm": tot_b_g,
            "router_row_norm": math.sqrt(router_sq_per_expert[e_idx]),
            "max_layer_w1_b_norm": max_l_w1_b,
            "max_layer_w1_b_layer": max_l_w1_layer,
            "max_layer_b_norm": max_l_b,
            "max_layer_b_layer": max_l_b_layer,
            "max_singular_value": exp_max_sv,
            "max_spectral_outlier_ratio": exp_max_ratio,
            "fomoe_zero_sink_status": sink_status,
            "zero_sink_layers": zero_sink_layers,
            "nonzero_layers": nonzero_layers,
            "non_finite_elements": exp_nf,
            "per_layer": per_layer_rows,
        }

    # Cross-expert median norm & rogue high-norm unscored detection
    nonzero_w1 = [info["w1_b_norm"] for info in expert_accum.values() if info["w1_b_norm"] > 0.0]
    nonzero_tot = [info["total_b_norm"] for info in expert_accum.values() if info["total_b_norm"] > 0.0]
    med_w1 = float(statistics.median(nonzero_w1)) if nonzero_w1 else 0.0
    med_tot = float(statistics.median(nonzero_tot)) if nonzero_tot else 0.0

    high_norm_outliers: list[int] = []
    rogue_experts: list[int] = []
    max_norm_ratio = 0.0
    max_w1_b_norm = 0.0

    for e_idx in layout.present_experts:
        info = expert_accum[e_idx]
        w1_n = float(info["w1_b_norm"])
        tot_n = float(info["total_b_norm"])
        w1_ratio = w1_n / max(med_w1, 1e-8) if med_w1 > 0.0 else 0.0
        tot_ratio = tot_n / max(med_tot, 1e-8) if med_tot > 0.0 else 0.0
        max_norm_ratio = max(max_norm_ratio, w1_ratio, tot_ratio)
        max_w1_b_norm = max(max_w1_b_norm, w1_n)

        if len(nonzero_tot) >= 3:
            is_high_norm = (
                (w1_n >= rogue_b_thresh and w1_ratio >= 2.0)
                or (tot_n >= rogue_b_thresh and tot_ratio >= 2.0)
                or (w1_ratio >= rogue_ratio_thresh and w1_n >= rogue_floor)
                or (tot_ratio >= rogue_ratio_thresh and tot_n >= rogue_floor)
            )
        else:
            is_high_norm = (w1_n >= rogue_b_thresh) or (tot_n >= rogue_b_thresh)

        is_scored = True if (scored_set is not None and e_idx in scored_set) else False
        is_rogue = bool(is_high_norm and not is_scored)

        info["norm_ratio"] = w1_ratio
        info["total_norm_ratio"] = tot_ratio
        info["is_scored_domain"] = is_scored
        info["is_high_norm_outlier"] = bool(is_high_norm)
        info["is_rogue_high_norm"] = is_rogue

        if is_high_norm:
            high_norm_outliers.append(e_idx)
        if is_rogue:
            rogue_experts.append(e_idx)

    ln_tail_report = _audit_layernorm_tail(
        buckets,
        layout,
        base_ln_tail=base_ln_tail,
        ln_weight_drift_rms_threshold=ln_w_thresh,
        ln_bias_drift_rms_threshold=ln_b_thresh,
        ln_max_abs_drift_threshold=ln_abs_thresh,
    )
    if ln_tail_report["non_finite_elements"] > 0:
        nf_locations.append(
            {
                "bucket": layout.ln_tail_key,
                "layer": -1,
                "component": "ln_tail",
                "non_finite_elements": int(ln_tail_report["non_finite_elements"]),
                "nan_elements": int(ln_tail_report["nan_elements"]),
                "inf_elements": int(ln_tail_report["inf_elements"]),
            }
        )

    health_failures: list[str] = []
    if total_nf > 0:
        health_failures.append(f"non_finite_corruption({total_nf}_elements)")
    if rogue_experts:
        health_failures.append(f"rogue_high_norm_experts({rogue_experts})")
    if ln_tail_report["drift_failed"]:
        health_failures.append(f"layernorm_tail_drift({','.join(ln_tail_report['drift_reasons'])})")
    if spectral_spikes:
        health_failures.append(f"spectral_spikes({len(spectral_spikes)})")
    if router_norm_outliers:
        health_failures.append(f"router_norm_outliers({len(router_norm_outliers)})")

    healthy = len(health_failures) == 0
    ln_rms = float(ln_tail_report["ln_tail_drift_rms"])
    anse_energy = (
        (0.0 if healthy else ANSE_MAXIMUM_PAIN)
        + ln_rms
        + 0.01 * max_ratio_global
        + 0.001 * max_w1_b_norm
    )

    nz_router_rows = [r for row in router_layer_row_norms for r in row if r > 0.0]
    med_router_row = float(statistics.median(nz_router_rows)) if nz_router_rows else 0.0
    max_router_row_val = max((r for row in router_layer_row_norms for r in row), default=0.0)

    payload: dict[str, Any] = {
        "object": INSPECT_OBJECT,
        "schema": SCHEMA,
        "source_type": source_type,
        "source_path": source_path,
        "source_file_sha256": source_file_sha,
        "buckets_sha256": compute_buckets_sha256(buckets),
        "geometry": {
            "num_layers": layout.num_layers,
            "num_experts": layout.num_experts,
            "rank": layout.rank,
            "hidden_dim": layout.hidden_dim,
            "inter_dim": layout.inter_dim,
            "num_router_fragments": len(layout.router_keys),
            "present_experts": list(layout.present_experts),
            "missing_experts": list(layout.missing_experts),
            "total_elements": total_elements,
            "dtypes": {k: str(buckets[k].dtype) for k in sorted(buckets.keys())},
        },
        "healthy": healthy,
        "status": "HEALTHY" if healthy else "UNHEALTHY",
        "health_failures": health_failures,
        "non_finite_summary": {
            "has_non_finite": total_nf > 0,
            "total_non_finite_elements": total_nf,
            "nan_elements": total_nan,
            "inf_elements": total_inf,
            "affected_buckets": affected_buckets,
            "locations": nf_locations,
        },
        "layernorm_tail": ln_tail_report,
        "fomoe_zero_sink_summary": {
            "full_zero_sink_experts": full_zero_sink_experts,
            "partial_zero_sink_experts": partial_zero_sink_experts,
            "active_experts": active_experts,
            "per_expert_status": {f"expert_{e}": expert_accum[e]["fomoe_zero_sink_status"] for e in layout.present_experts},
            "per_expert_zero_sink_layers": {f"expert_{e}": expert_accum[e]["zero_sink_layers"] for e in layout.present_experts},
        },
        "spectral_audit": {
            "max_lora_singular_value": max_sv_global,
            "max_spectral_outlier_ratio": max_ratio_global,
            "spectral_spike_count": len(spectral_spikes),
            "spectral_spikes": spectral_spikes,
        },
        "router_audit": {
            "max_router_row_norm": max_router_row_val,
            "median_nonzero_router_row_norm": med_router_row,
            "router_norm_outliers": router_norm_outliers,
            "per_layer_max_row_norm": per_layer_max_router_norm,
        },
        "rogue_audit": {
            "scored_experts": sorted(scored_set) if scored_set is not None else None,
            "median_nonzero_w1_b_norm": med_w1,
            "median_nonzero_total_b_norm": med_tot,
            "high_norm_outlier_experts": high_norm_outliers,
            "rogue_high_norm_experts": rogue_experts,
            "recommended_sink_experts": rogue_experts,
        },
        "experts": {f"expert_{e}": expert_accum[e] for e in layout.present_experts},
        "metrics": {
            "anse_energy": anse_energy,
            "norm_ratio": max_norm_ratio,
            "max_w1_b_norm": max_w1_b_norm,
            "ln_tail_drift_rms": ln_rms,
            "max_lora_singular_value": max_sv_global,
        },
    }
    return _attach_provenance(payload)


def apply_offline_ghost_sink_and_ln_surgery(
    buckets: str | os.PathLike[str] | Mapping[str, torch.Tensor],
    *,
    sink_experts: Sequence[int],
    active_layers: Sequence[int] | None = None,
    lambda_ln: float = DEFAULT_LAMBDA_LN,
    base_ln_tail: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    num_layers: int | None = None,
    num_experts: int | None = None,
    rank: int | None = None,
    hidden_dim: int | None = None,
    inter_dim: int | None = None,
    sanitize_non_finite: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply layer-localized FoMoE ghost-sink zeroing and LayerNorm tail interpolation offline on CPU.

    Produces a cloned, sanitized bucket dictionary preserving non-active layers and non-sink
    experts bitwise, along with exact parameter-edit ratio and SHA-256 provenance receipt.
    """
    lam = _validate_unit_interval(lambda_ln, "lambda_ln")
    if not isinstance(sanitize_non_finite, bool):
        raise CheckpointDoctorError(f"sanitize_non_finite must be a bool, got {type(sanitize_non_finite).__name__}")
    if isinstance(sink_experts, (str, bytes)) or not isinstance(sink_experts, Sequence):
        raise CheckpointDoctorError("sink_experts must be a sequence of integer expert indices")

    orig_buckets, source_type, source_path, source_file_sha, metadata = _normalize_buckets_input(buckets)
    nl, ne, rk, hd, idim = _infer_geometry(
        orig_buckets,
        metadata=metadata,
        base_ln_tail=base_ln_tail,
        num_layers=num_layers,
        num_experts=num_experts,
        rank=rank,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
    )
    layout = _build_bucket_layout(
        orig_buckets,
        num_layers=nl,
        num_experts=ne,
        rank=rk,
        hidden_dim=hd,
        inter_dim=idim,
    )

    sink_list = sorted({_validate_nonneg_int(e, "sink_expert") for e in sink_experts})
    for e_idx in sink_list:
        if e_idx >= layout.num_experts:
            raise CheckpointDoctorError(f"sink_expert {e_idx} >= num_experts={layout.num_experts}")

    if active_layers is None:
        active_layer_list = list(range(layout.num_layers))
    else:
        if isinstance(active_layers, (str, bytes)) or not isinstance(active_layers, Sequence):
            raise CheckpointDoctorError("active_layers must be None or a sequence of layer indices")
        active_layer_list = sorted({_validate_nonneg_int(l_i, "active_layer") for l_i in active_layers})
        for l_i in active_layer_list:
            if l_i >= layout.num_layers:
                raise CheckpointDoctorError(f"active_layer {l_i} >= num_layers={layout.num_layers}")
    active_set = set(active_layer_list)
    preserved_layers = [l_i for l_i in range(layout.num_layers) if l_i not in active_set]

    if base_ln_tail is None:
        base_ln_w = torch.ones(layout.hidden_dim, dtype=torch.float64)
        base_ln_b = torch.zeros(layout.hidden_dim, dtype=torch.float64)
        ln_baseline_mode = "unit_ones_zero_bias"
    else:
        base_ln_w, base_ln_b = _extract_tail_pair(base_ln_tail, hidden_dim=layout.hidden_dim)
        ln_baseline_mode = "explicit_base_ln_tail"

    input_sha = compute_buckets_sha256(orig_buckets)
    working: dict[str, torch.Tensor] = {k: v.clone() for k, v in orig_buckets.items()}

    # Step 1: Non-finite check / optional fail-safe repair without touching finite entries
    non_finite_repaired = 0
    for key in sorted(working.keys()):
        tens = working[key]
        fin_mask = torch.isfinite(tens)
        nf_cnt = int((~fin_mask).sum().item())
        if nf_cnt == 0:
            continue
        if not sanitize_non_finite:
            raise CheckpointDoctorError(f"Non-finite values ({nf_cnt}) detected in {key!r} with sanitize_non_finite=False")
        if key == layout.ln_tail_key:
            tail_start = layout.ln_tail_offset
            router_part = tens[:tail_start]
            r_fin = torch.isfinite(router_part)
            if not bool(r_fin.all().item()):
                tens[:tail_start].copy_(torch.where(r_fin, router_part, torch.zeros_like(router_part)))
            ln_w_part = tens[tail_start : tail_start + layout.hidden_dim]
            ln_b_part = tens[tail_start + layout.hidden_dim :]
            w_fin = torch.isfinite(ln_w_part)
            b_fin = torch.isfinite(ln_b_part)
            if not bool(w_fin.all().item()):
                ln_w_part.copy_(torch.where(w_fin, ln_w_part, base_ln_w.to(dtype=ln_w_part.dtype)))
            if not bool(b_fin.all().item()):
                ln_b_part.copy_(torch.where(b_fin, ln_b_part, base_ln_b.to(dtype=ln_b_part.dtype)))
        else:
            tens.copy_(torch.where(fin_mask, tens, torch.zeros_like(tens)))
        non_finite_repaired += nf_cnt

    # Step 2: Layer-localized FoMoE Ghost-Sink Zeroing (w1_B = w2_B = w3_B = 0)
    b_factor_specs = [spec for spec in layout.expert_factor_slices if spec.name.endswith("_B")]
    applied_sink_experts: list[int] = []
    missing_sink_experts: list[int] = []
    ghost_sink_elements_zeroed = 0

    for e_idx in sink_list:
        e_key = f"expert_{e_idx}"
        if e_key not in working:
            missing_sink_experts.append(e_idx)
            continue
        applied_sink_experts.append(e_idx)
        e_vec = working[e_key]
        for l_idx in active_layer_list:
            base_off = l_idx * layout.per_layer_expert_numel
            for spec in b_factor_specs:
                sl = e_vec[base_off + spec.offset : base_off + spec.offset + spec.numel]
                ghost_sink_elements_zeroed += int((sl != 0).sum().item())
                sl.zero_()

    # Step 3: LayerNorm Tail Interpolation (base + lambda_ln * (curr - base))
    tail_slice = working[layout.ln_tail_key][layout.ln_tail_offset : layout.ln_tail_offset + 2 * layout.hidden_dim]
    tail_dtype = tail_slice.dtype
    orig_tail = orig_buckets[layout.ln_tail_key][layout.ln_tail_offset : layout.ln_tail_offset + 2 * layout.hidden_dim]
    tail_had_nf = not bool(torch.isfinite(orig_tail).all().item())
    if lam != 1.0 or tail_had_nf:
        curr_ln_w = tail_slice[: layout.hidden_dim].to(torch.float64)
        curr_ln_b = tail_slice[layout.hidden_dim :].to(torch.float64)
        new_ln_w = (base_ln_w + lam * (curr_ln_w - base_ln_w)).to(dtype=tail_dtype)
        new_ln_b = (base_ln_b + lam * (curr_ln_b - base_ln_b)).to(dtype=tail_dtype)
        tail_slice[: layout.hidden_dim].copy_(new_ln_w)
        tail_slice[layout.hidden_dim :].copy_(new_ln_b)

    # Step 4: Exact bitwise edit-ratio accounting & non-active layer preservation check
    total_elements = sum(int(t.numel()) for t in orig_buckets.values())
    edited_elements = 0
    per_bucket_edited: dict[str, int] = {}
    for key in sorted(working.keys()):
        diff_cnt = _count_bitwise_modified(orig_buckets[key], working[key])
        per_bucket_edited[key] = diff_cnt
        edited_elements += diff_cnt

    non_active_modified = 0
    for e_idx in layout.present_experts:
        e_key = f"expert_{e_idx}"
        for l_idx in preserved_layers:
            off0 = l_idx * layout.per_layer_expert_numel
            off1 = off0 + layout.per_layer_expert_numel
            non_active_modified += _count_bitwise_modified(orig_buckets[e_key][off0:off1], working[e_key][off0:off1])

    param_edit_ratio = float(edited_elements) / float(max(1, total_elements))
    output_sha = compute_buckets_sha256(working)

    receipt_core: dict[str, Any] = {
        "object": SURGERY_OBJECT,
        "schema": SCHEMA,
        "source_type": source_type,
        "source_path": source_path,
        "source_file_sha256": source_file_sha,
        "input_buckets_sha256": input_sha,
        "output_buckets_sha256": output_sha,
        "geometry": {
            "num_layers": layout.num_layers,
            "num_experts": layout.num_experts,
            "rank": layout.rank,
            "hidden_dim": layout.hidden_dim,
            "inter_dim": layout.inter_dim,
        },
        "surgery_config": {
            "sink_experts": sink_list,
            "applied_sink_experts": applied_sink_experts,
            "missing_sink_experts": missing_sink_experts,
            "active_layers": active_layer_list,
            "preserved_layers": preserved_layers,
            "lambda_ln": lam,
            "ln_baseline_mode": ln_baseline_mode,
            "sanitize_non_finite": sanitize_non_finite,
        },
        "non_finite_repaired_elements": non_finite_repaired,
        "ghost_sink_elements_zeroed": ghost_sink_elements_zeroed,
        "total_elements": total_elements,
        "edited_elements": edited_elements,
        "parameter_edit_ratio": param_edit_ratio,
        "non_active_layers_bitwise_preserved": non_active_modified == 0,
        "per_bucket_edited_elements": per_bucket_edited,
    }
    return working, _attach_provenance(receipt_core)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render deterministic GitHub-flavored Markdown for an inspection or surgery receipt."""
    obj_type = report.get("object")
    if obj_type == INSPECT_OBJECT:
        geom = report["geometry"]
        ln_t = report["layernorm_tail"]
        metrics = report["metrics"]
        rogue = report["rogue_audit"]
        sink = report["fomoe_zero_sink_summary"]
        lines = [
            "# HiveMind WSP Checkpoint Doctor Inspection Report",
            "",
            f"- **Status**: `{report['status']}` (`healthy={report['healthy']}`)",
            f"- **Geometry**: `{geom['num_layers']} layers x {geom['num_experts']} experts` (rank=`{geom['rank']}`, H=`{geom['hidden_dim']}`, I=`{geom['inter_dim']}`)",
            f"- **Buckets SHA-256**: `{report['buckets_sha256']}`",
            f"- **ANSE Energy**: `{metrics['anse_energy']:.6f}` | **Max `norm_ratio`**: `{metrics['norm_ratio']:.4f}`",
            f"- **Non-Finite Elements**: `{report['non_finite_summary']['total_non_finite_elements']}`",
            f"- **LayerNorm Tail Drift vs 1.0**: weight RMS `{ln_t['ln_weight_drift_rms']:.5f}`, bias RMS `{ln_t['ln_bias_drift_rms']:.5f}`, max abs `{max(ln_t['ln_weight_max_abs_drift'], ln_t['ln_bias_max_abs']):.5f}` (`failed={ln_t['drift_failed']}`)",
            f"- **FoMoE Zero-Sink Experts**: full=`{sink['full_zero_sink_experts']}`, partial=`{sink['partial_zero_sink_experts']}`, active=`{sink['active_experts']}`",
            f"- **Rogue High-Norm Experts**: `{rogue['rogue_high_norm_experts']}` | **Spectral Spikes**: `{report['spectral_audit']['spectral_spike_count']}`",
            f"- **Health Failures**: `{report['health_failures']}`",
            "",
            "| Expert | `||w1_B||_F` | `||total_B||_F` | `||router||_2` | `norm_ratio` | Max `sigma_1` | Sink Status | Rogue |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |",
        ]
        for e_key, info in report["experts"].items():
            lines.append(
                f"| `{e_key}` | `{info['w1_b_norm']:.4f}` | `{info['total_b_norm']:.4f}` | "
                f"`{info['router_row_norm']:.4f}` | `{info['norm_ratio']:.3f}` | `{info['max_singular_value']:.4f}` | "
                f"`{info['fomoe_zero_sink_status']}` | `{info['is_rogue_high_norm']}` |"
            )
        lines.extend(["", f"**Receipt SHA-256**: `{report['receipt_sha256']}`"])
        return assert_safe_output("\n".join(lines))

    if obj_type == SURGERY_OBJECT:
        cfg = report["surgery_config"]
        post = report.get("post_surgery_inspection")
        lines = [
            "# HiveMind WSP Offline Ghost-Sink & LayerNorm Surgery Report",
            "",
            f"- **Input Buckets SHA-256**: `{report['input_buckets_sha256']}`",
            f"- **Output Buckets SHA-256**: `{report['output_buckets_sha256']}`",
            f"- **Sink Experts**: `{cfg['applied_sink_experts']}` | **Active Layers**: `{cfg['active_layers']}` | **Preserved Layers**: `{cfg['preserved_layers']}`",
            f"- **LayerNorm Interpolation (`lambda_ln`)**: `{cfg['lambda_ln']:.4f}` (`{cfg['ln_baseline_mode']}`)",
            f"- **Parameter-Edit Ratio**: `{report['parameter_edit_ratio']:.6f}` (`{report['edited_elements']}/{report['total_elements']}` elements)",
            f"- **Non-Active Layers Bitwise Preserved**: `{report['non_active_layers_bitwise_preserved']}`",
        ]
        if isinstance(post, Mapping):
            lines.append(
                f"- **Post-Surgery Health**: `{post['status']}` (`anse_energy={post['metrics']['anse_energy']:.6f}`, failures=`{post['health_failures']}`)"
            )
        lines.extend(["", f"**Receipt SHA-256**: `{report['receipt_sha256']}`"])
        return assert_safe_output("\n".join(lines))

    raise CheckpointDoctorError(f"Unsupported report object type: {obj_type!r}")


def _parse_csv_ints(raw: str | None, name: str) -> list[int] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() == "none":
        return []
    out: list[int] = []
    for part in stripped.split(","):
        tok = part.strip()
        if not tok or not tok.lstrip("+-").isdigit():
            raise CheckpointDoctorError(f"Invalid integer token {tok!r} in {name}")
        out.append(_validate_nonneg_int(int(tok), name))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``scripts/hivemind_wsp_checkpoint_doctor.py``."""
    ap = argparse.ArgumentParser(
        description=(
            "Offline CPU-only CrowdLoRA Sparse MoE safetensors checkpoint doctor & ANSE preflight CLI."
        )
    )
    ap.add_argument("checkpoint_pos", nargs="?", type=Path, default=None, help="Optional positional .safetensors checkpoint path.")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Path to target .safetensors checkpoint.")
    ap.add_argument(
        "--inspect",
        nargs="?",
        const="__FLAG__",
        default=None,
        help="Inspect CrowdLoRA safetensors checkpoint against ANSE/weight health thresholds.",
    )
    ap.add_argument(
        "--apply-surgery",
        nargs="?",
        const="__FLAG__",
        default=None,
        help="Apply offline FoMoE ghost-sink + LayerNorm tail surgery and verify health.",
    )
    ap.add_argument("--base-checkpoint", type=Path, default=None, help="Optional base .safetensors for base_ln_tail.")
    ap.add_argument("--out-checkpoint", type=Path, default=None, help="Optional output path for sanitized .safetensors.")
    ap.add_argument("--num-layers", type=int, default=None, help="Number of transformer MoE layers.")
    ap.add_argument("--num-experts", type=int, default=None, help="Total number of MoE experts per layer.")
    ap.add_argument("--rank", type=int, default=None, help="LoRA adapter rank r.")
    ap.add_argument("--hidden-dim", type=int, default=None, help="Transformer hidden width H.")
    ap.add_argument("--inter-dim", type=int, default=None, help="SwiGLU intermediate width I.")
    ap.add_argument("--scored-experts", type=str, default=None, help="Comma-separated scored expert indices (e.g. 0,1,2,5,6,7).")
    ap.add_argument("--sink-experts", type=str, default=None, help="Comma-separated expert indices to ghost-sink (default: auto rogue).")
    ap.add_argument("--active-layers", type=str, default=None, help="Comma-separated layer indices for localized surgery.")
    ap.add_argument("--lambda-ln", type=float, default=DEFAULT_LAMBDA_LN, help="LayerNorm interpolation coefficient in [0, 1].")
    ap.add_argument("--rogue-b-norm-threshold", type=float, default=DEFAULT_ROGUE_B_NORM_THRESHOLD)
    ap.add_argument("--rogue-norm-ratio-threshold", type=float, default=DEFAULT_ROGUE_NORM_RATIO_THRESHOLD)
    ap.add_argument("--spectral-spike-threshold", type=float, default=DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD)
    ap.add_argument("--max-lora-singular-value", type=float, default=DEFAULT_MAX_LORA_SINGULAR_VALUE)
    ap.add_argument("--max-router-row-norm", type=float, default=DEFAULT_MAX_ROUTER_ROW_NORM)
    ap.add_argument("--ln-weight-drift-threshold", type=float, default=DEFAULT_LN_WEIGHT_DRIFT_RMS_THRESHOLD)
    ap.add_argument("--ln-bias-drift-threshold", type=float, default=DEFAULT_LN_BIAS_DRIFT_RMS_THRESHOLD)
    ap.add_argument("--ln-max-abs-drift-threshold", type=float, default=DEFAULT_LN_MAX_ABS_DRIFT_THRESHOLD)
    ap.add_argument("--no-sanitize-non-finite", action="store_true", help="Fail closed instead of repairing NaN/Inf values.")
    ap.add_argument("--out-json", type=Path, default=None, help="Optional file path to write JSON report.")
    ap.add_argument("--out-markdown", type=Path, default=None, help="Optional file path to write Markdown report.")
    ap.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    ap.add_argument("--markdown", action="store_true", help="Print Markdown report to stdout (default).")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning ``0`` when checkpoint is healthy (or ``2`` on fail-closed error/unhealthy)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = build_parser().parse_args(argv)

    try:
        if args.inspect is None and args.apply_surgery is None:
            raise CheckpointDoctorError("Specify --inspect or --apply-surgery.")

        path_candidates: list[Path] = []
        for raw_val in (args.checkpoint, args.inspect, args.apply_surgery, args.checkpoint_pos):
            if isinstance(raw_val, Path):
                path_candidates.append(raw_val)
            elif isinstance(raw_val, str) and raw_val != "__FLAG__":
                path_candidates.append(Path(raw_val))
        if not path_candidates:
            raise CheckpointDoctorError("No checkpoint path provided.")
        ckpt_path = path_candidates[0]

        buckets, metadata, file_sha = load_checkpoint_buckets(ckpt_path)
        base_buckets: dict[str, torch.Tensor] | None = None
        if args.base_checkpoint is not None:
            base_buckets, _, _ = load_checkpoint_buckets(args.base_checkpoint)

        nl, ne, rk, hd, idim = _infer_geometry(
            buckets,
            metadata=metadata,
            base_ln_tail=base_buckets,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            rank=args.rank,
            hidden_dim=args.hidden_dim,
            inter_dim=args.inter_dim,
        )
        scored_list = _parse_csv_ints(args.scored_experts, "scored_experts")

        inspect_kwargs: dict[str, Any] = {
            "num_layers": nl,
            "num_experts": ne,
            "rank": rk,
            "hidden_dim": hd,
            "inter_dim": idim,
            "scored_experts": scored_list,
            "rogue_b_norm_threshold": args.rogue_b_norm_threshold,
            "rogue_norm_ratio_threshold": args.rogue_norm_ratio_threshold,
            "spectral_spike_ratio_threshold": args.spectral_spike_threshold,
            "max_lora_singular_value": args.max_lora_singular_value,
            "max_router_row_norm": args.max_router_row_norm,
            "ln_weight_drift_rms_threshold": args.ln_weight_drift_threshold,
            "ln_bias_drift_rms_threshold": args.ln_bias_drift_threshold,
            "ln_max_abs_drift_threshold": args.ln_max_abs_drift_threshold,
            "base_ln_tail": base_buckets,
        }

        if args.apply_surgery is not None:
            pre_inspection = inspect_moe_bucket_safetensors(buckets, **inspect_kwargs)
            sink_list = _parse_csv_ints(args.sink_experts, "sink_experts")
            if sink_list is None:
                sink_list = list(pre_inspection["rogue_audit"]["recommended_sink_experts"])
            active_list = _parse_csv_ints(args.active_layers, "active_layers")

            sanitized, report = apply_offline_ghost_sink_and_ln_surgery(
                buckets,
                sink_experts=sink_list,
                active_layers=active_list,
                lambda_ln=args.lambda_ln,
                base_ln_tail=base_buckets,
                num_layers=nl,
                num_experts=ne,
                rank=rk,
                hidden_dim=hd,
                inter_dim=idim,
                sanitize_non_finite=not args.no_sanitize_non_finite,
            )
            report["source_type"] = "safetensors_file"
            report["source_path"] = str(ckpt_path)
            report["source_file_sha256"] = file_sha

            if args.out_checkpoint is not None:
                out_meta = {
                    "hivemind_doctor_meta": json.dumps(
                        {
                            "num_layers": nl,
                            "num_experts": ne,
                            "rank": rk,
                            "hidden_dim": hd,
                            "inter_dim": idim,
                            "buckets_sha256": report["output_buckets_sha256"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                }
                report["out_checkpoint_path"] = str(args.out_checkpoint)
                report["out_checkpoint_file_sha256"] = save_checkpoint_safetensors(
                    sanitized, args.out_checkpoint, metadata=out_meta
                )

            post_inspection = inspect_moe_bucket_safetensors(sanitized, **inspect_kwargs)
            report["pre_surgery_status"] = pre_inspection["status"]
            report["post_surgery_inspection"] = post_inspection
            report = _attach_provenance(report)
            healthy = bool(post_inspection["healthy"])
        else:
            report = inspect_moe_bucket_safetensors(buckets_or_path=ckpt_path, **inspect_kwargs)
            healthy = bool(report["healthy"])

        json_text = assert_safe_output(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        md_text = render_markdown_report(report)

        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json_text + "\n", encoding="utf-8")
        if args.out_markdown is not None:
            args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.out_markdown.write_text(md_text + "\n", encoding="utf-8")

        if args.json:
            print(json_text)
        else:
            print(md_text)

        return 0 if healthy else 2
    except CheckpointDoctorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
