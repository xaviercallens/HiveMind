# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""StreamingLLM Attention-Sink & GQA Value-Output Null-Space Stabilizer for Open-Weight MoEs.

Implements training-free Grouped-Query Attention (GQA) massive-activation / attention-sink
detection and ``o_proj`` null-space weight surgery for open-weight Sparse MoE models
(``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP`` / ``CrowdFineGrainedMoEMLP`` / GQA blocks),
inspired by Sun et al. (2024, "Massive Activations in Large Language Models") and
Xiao et al. ("StreamingLLM"):

1. ``detect_massive_sink_heads_and_channels(attn_head_outputs, token_positions, *, sink_positions=(0, 1, 2, 3), spike_ratio_threshold=5.0)``:
   Separates GQA query/KV heads and channels carrying benign BOS attention sinks at
   anchor positions from heads and residual channels emitting rogue mid-sequence spikes.
2. ``project_oproj_rogue_spike_into_prose_nullspace(o_proj_weight, rogue_head_slice, rogue_dir, preserve_sink_acts, *, damp_alpha=0.85, max_norm_delta_ratio=0.03)``:
   Surgically damps mid-sequence rogue head output directions inside the null-space of
   BOS/anchor sink activations and general-prose keys without altering KV-cache shapes
   or RoPE frequencies, bounded strictly by ``max_norm_delta_ratio``.
3. ``execute_verified_attention_sink_surgery(...)``:
   Executes sequential layer-wise ``o_proj`` null-space stabilization under an ANSE
   thermodynamic gate, committing edits if and only if ``Delta E < 0``, BOS anchor
   drift ``<= 1e-5``, general/long-context/worst-domain NLL drift ``<= +1.0%``, and
   KV-cache/RoPE invariants hold; otherwise restores the exact parent state bit-for-bit
   with Maximum Pain accounting and a canonical SHA-256 provenance receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.marchenko_pastur_covariance_guard import shrink_moment_oracle_oas

SURGEON_SCHEMA: str = "hivemind.attention_sink_kv_surgeon/1"
SURGEON_RECEIPT_OBJECT: str = "hivemind.attention_sink_kv_surgeon_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
LONG_CONTEXT_PREFIX: str = "LONG_"
DEFAULT_SINK_POSITIONS: tuple[int, ...] = (0, 1, 2, 3)
DEFAULT_SPIKE_RATIO_THRESHOLD: float = 5.0
DEFAULT_DAMP_ALPHA: float = 0.85
DEFAULT_MAX_NORM_DELTA_RATIO: float = 0.03
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-6
DEFAULT_PRESERVE_REG: float = 0.05
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_GENERAL_GUARD_MAX_RATIO: float = 1.01
DEFAULT_BOS_ANCHOR_MAX_DRIFT: float = 1e-5
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SPIKE_ENERGY_WEIGHT: float = 0.05
DEFAULT_SCORE_FROM: int = 4
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_MIN_ABS_MAGNITUDE: float = 1e-4
VALID_PROJECTION_SPACES: frozenset[str] = frozenset({"auto", "head", "residual"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` lies inside ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _validate_sink_positions(sink_positions: Sequence[int]) -> tuple[int, ...]:
    """Validate non-empty sequence of unique non-negative sink token positions."""
    if isinstance(sink_positions, (str, bytes, bytearray)) or not isinstance(sink_positions, Sequence):
        raise TypeError("sink_positions must be a non-empty sequence of non-negative ints")
    if len(sink_positions) == 0:
        raise ValueError("sink_positions must be non-empty")
    seen: list[int] = []
    for idx, pos in enumerate(sink_positions):
        val = _validate_nonneg_int(pos, f"sink_positions[{idx}]")
        if val not in seen:
            seen.append(val)
    return tuple(sorted(seen))


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically while rejecting non-finite numbers."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Compute lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute bitwise SHA-256 digest over a tensor's dtype, shape, and raw bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    t_cpu = tensor.detach().to(device="cpu").contiguous()
    if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
        raise ValueError("tensor contains non-finite values")
    digest = hashlib.sha256()
    digest.update(f"{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
    digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic bitwise SHA-256 digest over a non-empty tensor mapping."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty Mapping[str, torch.Tensor]")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"State key must be a non-empty string, got {key!r}")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        t_cpu = tensor.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_attention_sink_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify ``receipt_sha256`` and schema provenance on a surgery receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != SURGEON_RECEIPT_OBJECT or receipt.get("schema") != SURGEON_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class AttentionSinkSurgeonConfig:
    """Validated hyperparameters for GQA attention-sink detection and null-space surgery."""

    sink_positions: tuple[int, ...] = DEFAULT_SINK_POSITIONS
    spike_ratio_threshold: float = DEFAULT_SPIKE_RATIO_THRESHOLD
    damp_alpha: float = DEFAULT_DAMP_ALPHA
    max_norm_delta_ratio: float = DEFAULT_MAX_NORM_DELTA_RATIO
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    reg_preserve: float = DEFAULT_PRESERVE_REG
    shrink_preserve: bool = True
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    general_guard_max_ratio: float = DEFAULT_GENERAL_GUARD_MAX_RATIO
    bos_anchor_max_drift: float = DEFAULT_BOS_ANCHOR_MAX_DRIFT
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    spike_energy_weight: float = DEFAULT_SPIKE_ENERGY_WEIGHT
    general_prefix: str = GENERAL_EXPERT_PREFIX
    long_context_prefix: str = LONG_CONTEXT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "sink_positions", _validate_sink_positions(self.sink_positions))
        thr = _validate_positive_finite(self.spike_ratio_threshold, "spike_ratio_threshold")
        if thr <= 1.0:
            raise ValueError(f"spike_ratio_threshold must be > 1.0, got {thr}")
        _validate_unit_interval(self.damp_alpha, "damp_alpha")
        norm_cap = _validate_positive_finite(self.max_norm_delta_ratio, "max_norm_delta_ratio")
        if norm_cap > 1.0:
            raise ValueError(f"max_norm_delta_ratio must be <= 1.0, got {norm_cap}")
        sv_tol = _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_rel_tol must be < 1.0, got {sv_tol}")
        _validate_nonneg_finite(self.reg_preserve, "reg_preserve")
        _validate_bool(self.shrink_preserve, "shrink_preserve")
        dom_g = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        gen_g = _validate_positive_finite(self.general_guard_max_ratio, "general_guard_max_ratio")
        safe_g = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if dom_g < 1.0 or gen_g < 1.0 or safe_g < 1.0:
            raise ValueError("Guard ratio thresholds must all be >= 1.0")
        _validate_nonneg_finite(self.bos_anchor_max_drift, "bos_anchor_max_drift")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.spike_energy_weight, "spike_energy_weight")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        if not isinstance(self.long_context_prefix, str) or not self.long_context_prefix.strip():
            raise ValueError("long_context_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


# ---------------------------------------------------------------------------
# 2. GQA Benign BOS Sink vs Mid-Sequence Rogue Spike Detector
# ---------------------------------------------------------------------------


def _normalize_head_outputs_4d(
    attn_head_outputs: Any,
    *,
    head_dim: int | None,
) -> torch.Tensor:
    """Validate and normalize attention head outputs to CPU ``float64`` ``(B, T, H_q, D_h)``."""
    if not isinstance(attn_head_outputs, torch.Tensor):
        raise TypeError(
            f"attn_head_outputs must be a torch.Tensor, got {type(attn_head_outputs).__name__}"
        )
    if any(dim == 0 for dim in attn_head_outputs.shape):
        raise ValueError(
            f"attn_head_outputs must be non-empty, got shape {tuple(attn_head_outputs.shape)}"
        )
    if not attn_head_outputs.is_floating_point():
        raise TypeError(
            f"attn_head_outputs must have a floating-point dtype, got {attn_head_outputs.dtype}"
        )

    t64 = attn_head_outputs.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(t64).all().item()):
        raise ValueError("attn_head_outputs must contain only finite values")

    if t64.ndim == 4:
        if head_dim is not None:
            hd = _validate_positive_int(head_dim, "head_dim")
            if int(t64.shape[3]) != hd:
                raise ValueError(f"attn_head_outputs head_dim {t64.shape[3]} != {hd}")
        return t64
    if t64.ndim == 3:
        if head_dim is not None:
            hd = _validate_positive_int(head_dim, "head_dim")
            d_model = int(t64.shape[2])
            if d_model % hd != 0:
                raise ValueError(f"Last dimension {d_model} is not divisible by head_dim={hd}")
            return t64.view(int(t64.shape[0]), int(t64.shape[1]), d_model // hd, hd)
        return t64.unsqueeze(0)

    raise ValueError(
        f"attn_head_outputs must have 3 or 4 dims, got shape {tuple(attn_head_outputs.shape)}"
    )


def _normalize_token_positions(
    token_positions: Any,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Validate 1D ``(T,)`` or 2D ``(B, T)`` non-negative integer position IDs."""
    if isinstance(token_positions, torch.Tensor):
        if token_positions.is_floating_point() or token_positions.dtype == torch.bool:
            raise TypeError(f"token_positions must have an integer dtype, got {token_positions.dtype}")
        pos_t = token_positions.detach().to(device="cpu", dtype=torch.long)
    elif isinstance(token_positions, Sequence) and not isinstance(token_positions, (str, bytes, bytearray)):
        if len(token_positions) == 0:
            raise ValueError("token_positions must be non-empty")
        pos_t = torch.tensor(token_positions, dtype=torch.long)
    else:
        raise TypeError("token_positions must be an integer torch.Tensor or sequence")

    if bool((pos_t < 0).any().item()):
        raise ValueError("token_positions must contain only non-negative integers")

    if pos_t.ndim == 1:
        if int(pos_t.shape[0]) != seq_len:
            raise ValueError(f"1D token_positions length {pos_t.shape[0]} must match seq_len={seq_len}")
        return pos_t.unsqueeze(0).expand(batch_size, seq_len).contiguous()
    if pos_t.ndim == 2:
        if tuple(pos_t.shape) != (batch_size, seq_len):
            raise ValueError(
                f"2D token_positions shape {tuple(pos_t.shape)} must match ({batch_size}, {seq_len})"
            )
        return pos_t.contiguous()
    raise ValueError(f"token_positions must be 1D or 2D, got shape {tuple(pos_t.shape)}")


def _extract_dominant_direction(vectors_2d: torch.Tensor) -> torch.Tensor:
    """Return canonical unit top right singular direction of non-empty 2D ``vectors_2d``."""
    d = int(vectors_2d.shape[1])
    if float(torch.linalg.norm(vectors_2d).item()) <= 1e-12:
        return torch.zeros(d, dtype=torch.float64)
    _, s_vals, vh = torch.linalg.svd(vectors_2d, full_matrices=False)
    if float(s_vals[0].item()) <= 1e-12:
        return torch.zeros(d, dtype=torch.float64)
    vec = vh[0]
    row_norms = torch.linalg.vector_norm(vectors_2d, dim=-1)
    anchor_row = vectors_2d[int(torch.argmax(row_norms).item())]
    if float((vec @ anchor_row).item()) < 0.0:
        vec = -vec
    return (vec / vec.norm().clamp_min(1e-12)).contiguous()


def detect_massive_sink_heads_and_channels(
    attn_head_outputs: torch.Tensor,
    token_positions: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
    *,
    sink_positions: Sequence[int] = DEFAULT_SINK_POSITIONS,
    spike_ratio_threshold: float = DEFAULT_SPIKE_RATIO_THRESHOLD,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    o_proj_weight: torch.Tensor | None = None,
    min_abs_magnitude: float = DEFAULT_MIN_ABS_MAGNITUDE,
) -> dict[str, Any]:
    """Identify GQA query/KV heads and channels carrying benign BOS sinks vs mid-sequence spikes.

    Distinguishes load-bearing StreamingLLM anchor sinks localized to ``sink_positions``
    (e.g. BOS positions ``0..3``) from rogue massive activations erupting at mid-sequence
    positions (Sun et al., 2024) that would otherwise hijack downstream MoE routers.
    """
    sink_tuple = _validate_sink_positions(sink_positions)
    threshold = _validate_positive_finite(spike_ratio_threshold, "spike_ratio_threshold")
    if threshold <= 1.0:
        raise ValueError(f"spike_ratio_threshold must be > 1.0, got {threshold}")
    min_abs = _validate_positive_finite(min_abs_magnitude, "min_abs_magnitude")

    z64 = _normalize_head_outputs_4d(attn_head_outputs, head_dim=head_dim)
    batch_size, seq_len, num_heads, d_h = z64.shape
    pos_2d = _normalize_token_positions(token_positions, batch_size, seq_len)

    if num_kv_heads is None:
        kv_heads = num_heads
    else:
        kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
        if kv_heads > num_heads or num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({kv_heads})"
            )
    group_size = num_heads // kv_heads

    w_o64: torch.Tensor | None = None
    if o_proj_weight is not None:
        w_o64 = _validate_oproj_weight_tensor(o_proj_weight, "o_proj_weight").detach().to("cpu", torch.float64)
        if int(w_o64.shape[1]) != num_heads * d_h:
            raise ValueError(
                f"o_proj_weight input width {w_o64.shape[1]} != num_heads * head_dim ({num_heads * d_h})"
            )

    sink_tensor = torch.tensor(sink_tuple, dtype=torch.long)
    sink_mask = torch.isin(pos_2d, sink_tensor)
    mid_mask = ~sink_mask
    if not bool(mid_mask.any().item()):
        raise ValueError("token_positions must include at least one non-sink mid-sequence position")

    norms = torch.linalg.vector_norm(z64, dim=-1)  # (B, T, H_q)
    mid_norms = norms[mid_mask]  # (N_mid, H_q)
    active_mid = mid_norms[mid_norms > min_abs]
    global_mid_baseline = (
        max(float(torch.median(active_mid).item()), min_abs) if active_mid.numel() > 0 else min_abs
    )

    mid_abs = z64[mid_mask].abs()
    active_ch = mid_abs[mid_abs > min_abs]
    ch_floor = global_mid_baseline / math.sqrt(float(d_h))
    channel_baseline = (
        max(float(torch.median(active_ch).item()), ch_floor, min_abs)
        if active_ch.numel() > 0
        else max(ch_floor, min_abs)
    )

    has_sink_tokens = bool(sink_mask.any().item())
    sink_norms = norms[sink_mask] if has_sink_tokens else torch.empty((0, num_heads), dtype=torch.float64)

    benign_bos_sink_heads: list[int] = []
    rogue_spike_heads: list[int] = []
    dual_role_heads: list[int] = []
    rogue_head_slices: dict[int, slice] = {}
    rogue_head_directions: dict[int, torch.Tensor] = {}
    rogue_residual_directions: dict[int, torch.Tensor] = {}
    rogue_channels_by_head: dict[int, list[int]] = {}
    bos_sink_channels_by_head: dict[int, list[int]] = {}
    rogue_flat_channels: list[int] = []
    per_head_stats: list[dict[str, Any]] = []

    for h_idx in range(num_heads):
        kv_idx = h_idx // group_size
        h_mid_norms = mid_norms[:, h_idx]
        h_mid_med = float(torch.median(h_mid_norms).item())
        h_mid_max = float(h_mid_norms.max().item())
        denom_h = (
            min(h_mid_med, global_mid_baseline)
            if h_mid_med >= 0.25 * global_mid_baseline
            else global_mid_baseline
        )
        denom_h = max(denom_h, min_abs)
        mid_spike_ratio = h_mid_max / denom_h

        if has_sink_tokens:
            h_sink_norms = sink_norms[:, h_idx]
            h_bos_mean = float(h_sink_norms.mean().item())
            h_bos_max = float(h_sink_norms.max().item())
            bos_sink_ratio = h_bos_max / max(global_mid_baseline, min_abs)
        else:
            h_bos_mean = 0.0
            h_bos_max = 0.0
            bos_sink_ratio = 0.0

        is_rogue = mid_spike_ratio >= threshold
        is_bos_sink = bos_sink_ratio >= threshold

        ch_mid_max = z64[mid_mask, h_idx, :].abs().max(dim=0).values
        ch_ratios = ch_mid_max / channel_baseline
        r_chs = [int(c) for c in range(d_h) if float(ch_ratios[c].item()) >= threshold]
        if is_rogue and not r_chs:
            top_val = float(ch_mid_max.max().item())
            r_chs = [int(c) for c in range(d_h) if float(ch_mid_max[c].item()) >= 0.5 * top_val]

        if has_sink_tokens:
            ch_bos_max = z64[sink_mask, h_idx, :].abs().max(dim=0).values
            b_chs = [int(c) for c in range(d_h) if float(ch_bos_max[c].item() / channel_baseline) >= threshold]
        else:
            b_chs = []

        if b_chs:
            bos_sink_channels_by_head[h_idx] = b_chs

        if is_rogue and is_bos_sink:
            classification = "dual_bos_and_rogue"
            dual_role_heads.append(h_idx)
            rogue_spike_heads.append(h_idx)
        elif is_rogue:
            classification = "mid_seq_rogue_spike"
            rogue_spike_heads.append(h_idx)
        elif is_bos_sink:
            classification = "benign_bos_sink"
            benign_bos_sink_heads.append(h_idx)
        else:
            classification = "normal"

        if is_rogue:
            rogue_head_slices[h_idx] = slice(h_idx * d_h, (h_idx + 1) * d_h)
            rogue_channels_by_head[h_idx] = r_chs
            rogue_flat_channels.extend(h_idx * d_h + c for c in r_chs)
            spiked_tok_mask = h_mid_norms >= threshold * denom_h
            z_sp = z64[mid_mask, h_idx, :][spiked_tok_mask]
            rogue_head_directions[h_idx] = _extract_dominant_direction(z_sp)
            if w_o64 is not None:
                w_h = w_o64[:, rogue_head_slices[h_idx]]
                y_from_head = w_h @ rogue_head_directions[h_idx]
                y_norm = float(y_from_head.norm().item())
                if y_norm > 1e-12:
                    rogue_residual_directions[h_idx] = (y_from_head / y_norm).contiguous()
                else:
                    rogue_residual_directions[h_idx] = _extract_dominant_direction(z_sp @ w_h.T)

        per_head_stats.append(
            {
                "head_idx": h_idx,
                "kv_head_idx": kv_idx,
                "bos_mean_norm": h_bos_mean,
                "bos_max_norm": h_bos_max,
                "bos_sink_ratio": bos_sink_ratio,
                "mid_median_norm": h_mid_med,
                "mid_max_norm": h_mid_max,
                "sink_norm_ratio": mid_spike_ratio,
                "classification": classification,
                "rogue_channels": r_chs if is_rogue else [],
                "bos_sink_channels": b_chs,
            }
        )

    rogue_residual_channels: list[int] = []
    bos_sink_residual_channels: list[int] = []
    if w_o64 is not None:
        y_tot = z64.reshape(batch_size, seq_len, num_heads * d_h) @ w_o64.T
        y_mid_abs = y_tot[mid_mask].abs()
        active_res = y_mid_abs[y_mid_abs > min_abs]
        res_base = max(float(torch.median(active_res).item()), min_abs) if active_res.numel() > 0 else min_abs
        res_mid_max = y_mid_abs.max(dim=0).values
        rogue_residual_channels = [
            int(d) for d in range(w_o64.shape[0]) if float(res_mid_max[d].item() / res_base) >= threshold
        ]
        if has_sink_tokens:
            res_bos_max = y_tot[sink_mask].abs().max(dim=0).values
            bos_sink_residual_channels = [
                int(d) for d in range(w_o64.shape[0]) if float(res_bos_max[d].item() / res_base) >= threshold
            ]

    return {
        "num_heads": num_heads,
        "num_kv_heads": kv_heads,
        "gqa_group_size": group_size,
        "head_dim": d_h,
        "sink_positions": list(sink_tuple),
        "spike_ratio_threshold": threshold,
        "global_mid_baseline_norm": global_mid_baseline,
        "channel_baseline_magnitude": channel_baseline,
        "benign_bos_sink_heads": benign_bos_sink_heads,
        "benign_bos_sink_kv_heads": sorted({h // group_size for h in benign_bos_sink_heads}),
        "rogue_spike_heads": rogue_spike_heads,
        "rogue_spike_kv_heads": sorted({h // group_size for h in rogue_spike_heads}),
        "dual_role_heads": dual_role_heads,
        "rogue_head_slices": rogue_head_slices,
        "rogue_head_directions": rogue_head_directions,
        "rogue_residual_directions": rogue_residual_directions,
        "rogue_channels_by_head": rogue_channels_by_head,
        "bos_sink_channels_by_head": bos_sink_channels_by_head,
        "rogue_flat_channels": sorted(rogue_flat_channels),
        "rogue_residual_channels": rogue_residual_channels,
        "bos_sink_residual_channels": bos_sink_residual_channels,
        "max_mid_spike_ratio": max(float(s["sink_norm_ratio"]) for s in per_head_stats),
        "max_bos_sink_ratio": max(float(s["bos_sink_ratio"]) for s in per_head_stats),
        "per_head_stats": per_head_stats,
    }


# ---------------------------------------------------------------------------
# 3. Protected BOS & Prose Null-Space Projection for Attention o_proj
# ---------------------------------------------------------------------------


def _validate_oproj_weight_tensor(target: Any, name: str) -> torch.Tensor:
    """Resolve and validate a 2D floating-point ``o_proj`` weight tensor."""
    w_tensor = target.weight if isinstance(target, nn.Module) and hasattr(target, "weight") else target
    if not isinstance(w_tensor, torch.Tensor):
        raise TypeError(f"{name} must be a 2D torch.Tensor or nn.Linear, got {type(target).__name__}")
    if w_tensor.ndim != 2 or w_tensor.shape[0] == 0 or w_tensor.shape[1] == 0:
        raise ValueError(f"{name} weight must be non-empty 2D, got shape {tuple(w_tensor.shape)}")
    if not w_tensor.is_floating_point():
        raise TypeError(f"{name} weight must have a floating-point dtype, got {w_tensor.dtype}")
    if not bool(torch.isfinite(w_tensor.detach()).all().item()):
        raise ValueError(f"{name} weight must contain only finite values")
    return w_tensor


def _normalize_head_column_indices(rogue_head_slice: Any, d_in: int) -> list[int]:
    """Validate and expand ``rogue_head_slice`` into ordered column indices in ``[0, d_in)``."""
    if isinstance(rogue_head_slice, slice):
        start, stop, step = rogue_head_slice.indices(d_in)
        if step != 1:
            raise ValueError(f"rogue_head_slice step must be 1, got {step}")
        indices = list(range(start, stop))
    elif isinstance(rogue_head_slice, Sequence) and not isinstance(rogue_head_slice, (str, bytes, bytearray)):
        indices = [_validate_nonneg_int(v, "rogue_head_slice") for v in rogue_head_slice]
    else:
        raise TypeError(
            f"rogue_head_slice must be a slice or sequence of column indices, got {type(rogue_head_slice).__name__}"
        )

    if not indices:
        raise ValueError("rogue_head_slice resolves to an empty column selection")
    if len(set(indices)) != len(indices):
        raise ValueError("rogue_head_slice must not contain duplicate column indices")
    if any(idx < 0 or idx >= d_in for idx in indices):
        raise ValueError(f"rogue_head_slice indices must lie in [0, {d_in}), got {indices}")
    return indices


def _normalize_preserve_acts_2d(
    preserve_sink_acts: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Validate and flatten protected BOS/prose activations to CPU ``float64`` ``(N, D)``."""
    if preserve_sink_acts is None:
        return None
    if isinstance(preserve_sink_acts, torch.Tensor):
        items = [("preserve_sink_acts", preserve_sink_acts)]
    elif isinstance(preserve_sink_acts, Mapping):
        if not preserve_sink_acts:
            raise ValueError("preserve_sink_acts mapping must not be empty")
        items = [(f"preserve_sink_acts[{k!r}]", v) for k, v in sorted(preserve_sink_acts.items())]
    elif isinstance(preserve_sink_acts, Sequence) and not isinstance(preserve_sink_acts, (str, bytes, bytearray)):
        if not preserve_sink_acts:
            raise ValueError("preserve_sink_acts sequence must not be empty")
        items = [(f"preserve_sink_acts[{idx}]", v) for idx, v in enumerate(preserve_sink_acts)]
    else:
        raise TypeError(
            f"preserve_sink_acts must be a Tensor, Sequence[Tensor], Mapping[str, Tensor], or None; "
            f"got {type(preserve_sink_acts).__name__}"
        )

    parts: list[torch.Tensor] = []
    width: int | None = None
    for label, ten in items:
        if not isinstance(ten, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(ten).__name__}")
        if ten.ndim not in (2, 3) or any(dim == 0 for dim in ten.shape):
            raise ValueError(f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(ten.shape)}")
        if not ten.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {ten.dtype}")
        flat = ten.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(ten.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        if width is None:
            width = int(flat.shape[1])
        elif int(flat.shape[1]) != width:
            raise ValueError(f"{label} width {flat.shape[1]} must match expected width {width}")
        parts.append(flat)
    return torch.cat(parts, dim=0).contiguous()


def _project_vector_into_nullspace(
    dir_vec: torch.Tensor,
    preserve_2d: torch.Tensor | None,
    *,
    nullspace_sv_rel_tol: float,
    reg_preserve: float,
    shrink_preserve: bool,
) -> tuple[torch.Tensor, str, int]:
    """Project 1D direction into exact or OAS-regularized null-space of ``preserve_2d``."""
    d = int(dir_vec.numel())
    d_norm = float(dir_vec.norm().item())
    if d_norm <= 1e-12:
        return torch.zeros(d, dtype=torch.float64), "zero_rogue_dir", 0

    r_unit = dir_vec / d_norm
    if preserve_2d is None:
        return r_unit, "unconstrained", d

    n_pres = int(preserve_2d.shape[0])
    m_emp = (preserve_2d.T @ preserve_2d) / float(n_pres)
    m_emp = 0.5 * (m_emp + m_emp.T)
    evals, evecs = torch.linalg.eigh(m_emp)
    lam_max = max(float(evals[-1].item()), 0.0)
    if lam_max <= 1e-24:
        return r_unit, "unconstrained", d

    null_mask = evals <= nullspace_sv_rel_tol * lam_max
    n_null = int(null_mask.sum().item())
    if n_null > 0:
        v_null = evecs[:, null_mask]
        u_raw = v_null @ (v_null.T @ r_unit)
        u_norm = float(u_raw.norm().item())
        if u_norm <= 1e-12:
            return torch.zeros(d, dtype=torch.float64), "rogue_dir_in_preserved_span", n_null
        return (u_raw / u_norm).contiguous(), "exact_nullspace", n_null

    m_base = shrink_moment_oracle_oas(m_emp, n_pres)[0] if shrink_preserve else m_emp
    tr_p = max(float(torch.diagonal(m_base).mean().item()), 1e-8)
    b_mat = m_base + max(reg_preserve * tr_p, 1e-8) * torch.eye(d, dtype=torch.float64)
    u_raw = torch.linalg.solve(0.5 * (b_mat + b_mat.T), r_unit)
    u_norm = float(u_raw.norm().item())
    if u_norm <= 1e-12:
        return torch.zeros(d, dtype=torch.float64), "degenerate_rayleigh", 0
    u_ctrl = u_raw / u_norm
    if float((u_ctrl @ r_unit).item()) < 0.0:
        u_ctrl = -u_ctrl
    return u_ctrl.contiguous(), "generalized_rayleigh", 0


def _build_oproj_noop_report(
    *,
    projection_mode: str,
    nullspace_mode: str,
    nullspace_dim: int,
    norm_before: float,
    head_norm_before: float,
    max_norm_delta_ratio: float,
    damp_alpha: float,
    noop_reason: str,
) -> dict[str, Any]:
    """Build standardized diagnostics payload when ``o_proj`` surgery is a safe no-op."""
    return {
        "projection_mode": projection_mode,
        "nullspace_mode": nullspace_mode,
        "nullspace_dim": int(nullspace_dim),
        "norm_before": float(norm_before),
        "norm_after": float(norm_before),
        "head_norm_before": float(head_norm_before),
        "head_norm_after": float(head_norm_before),
        "raw_norm_delta": 0.0,
        "raw_norm_delta_ratio": 0.0,
        "applied_norm_delta": 0.0,
        "applied_norm_delta_ratio": 0.0,
        "head_norm_delta_ratio": 0.0,
        "max_norm_delta_ratio": float(max_norm_delta_ratio),
        "budget_clamped": False,
        "budget_scale": 1.0,
        "damp_alpha": float(damp_alpha),
        "pre_spike_readout_norm": 0.0,
        "post_spike_readout_norm": 0.0,
        "spike_suppression_ratio": 0.0,
        "preserved_rms_drift": 0.0,
        "non_target_columns_bitwise_unchanged": True,
        "noop_reason": noop_reason,
    }


@torch.no_grad()
def project_oproj_rogue_spike_into_prose_nullspace(
    o_proj_weight: torch.Tensor | nn.Module,
    rogue_head_slice: slice | Sequence[int],
    rogue_dir: torch.Tensor,
    preserve_sink_acts: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None,
    *,
    damp_alpha: float = DEFAULT_DAMP_ALPHA,
    max_norm_delta_ratio: float = DEFAULT_MAX_NORM_DELTA_RATIO,
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL,
    reg_preserve: float = DEFAULT_PRESERVE_REG,
    shrink_preserve: bool = True,
    inplace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Damp mid-sequence rogue head output directions inside the BOS & prose null-space.

    Operates exclusively on columns ``rogue_head_slice`` of ``o_proj_weight``, leaving all
    other heads, KV-cache shapes, and RoPE frequencies bitwise untouched.
    """
    alpha = _validate_unit_interval(damp_alpha, "damp_alpha")
    norm_cap = _validate_positive_finite(max_norm_delta_ratio, "max_norm_delta_ratio")
    if norm_cap > 1.0:
        raise ValueError(f"max_norm_delta_ratio must be <= 1.0, got {norm_cap}")
    sv_tol = _validate_nonneg_finite(nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
    if sv_tol >= 1.0:
        raise ValueError(f"nullspace_sv_rel_tol must be < 1.0, got {sv_tol}")
    reg_p = _validate_nonneg_finite(reg_preserve, "reg_preserve")
    _validate_bool(shrink_preserve, "shrink_preserve")
    _validate_bool(inplace, "inplace")

    w_tensor = _validate_oproj_weight_tensor(o_proj_weight, "o_proj_weight")
    d_out, d_in = int(w_tensor.shape[0]), int(w_tensor.shape[1])
    col_indices = _normalize_head_column_indices(rogue_head_slice, d_in)
    d_h = len(col_indices)

    if not isinstance(rogue_dir, torch.Tensor):
        raise TypeError(f"rogue_dir must be a torch.Tensor, got {type(rogue_dir).__name__}")
    if rogue_dir.ndim != 1 or rogue_dir.numel() == 0:
        raise ValueError(f"rogue_dir must be a non-empty 1D tensor, got shape {tuple(rogue_dir.shape)}")
    if not rogue_dir.is_floating_point():
        raise TypeError(f"rogue_dir must have a floating-point dtype, got {rogue_dir.dtype}")
    r64 = rogue_dir.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(r64).all().item()):
        raise ValueError("rogue_dir must contain only finite values")

    d_rogue = int(r64.numel())
    if d_rogue not in (d_h, d_out):
        raise ValueError(
            f"rogue_dir length {d_rogue} must match head slice width {d_h} or residual dim {d_out}"
        )

    pres_raw = _normalize_preserve_acts_2d(preserve_sink_acts)
    if pres_raw is not None:
        d_pres = int(pres_raw.shape[1])
        if d_pres not in (d_h, d_out, d_in):
            raise ValueError(
                f"preserve_sink_acts width {d_pres} must match head dim {d_h}, residual dim {d_out}, or in dim {d_in}"
            )
    else:
        d_pres = 0

    w64 = w_tensor.detach().to(device="cpu", dtype=torch.float64)
    w_h64 = w64[:, col_indices]
    norm_before = float(torch.linalg.norm(w64).item())
    head_norm_before = float(torch.linalg.norm(w_h64).item())

    if d_rogue == d_h and (d_h != d_out or pres_raw is None or d_pres in (d_h, d_in)):
        if pres_raw is not None and d_pres == d_out and d_out != d_h:
            projection_mode = "residual_output_nullspace"
            target_dir = w_h64 @ r64
            pres_mat = pres_raw
        else:
            projection_mode = "head_input_nullspace"
            target_dir = r64
            if pres_raw is None:
                pres_mat = None
            elif d_pres == d_in and d_in != d_h:
                pres_mat = pres_raw[:, col_indices].contiguous()
            else:
                pres_mat = pres_raw
    else:
        projection_mode = "residual_output_nullspace"
        target_dir = r64
        if pres_raw is None:
            pres_mat = None
        elif d_pres == d_out:
            pres_mat = pres_raw
        elif d_pres == d_h:
            pres_mat = (pres_raw @ w_h64.T).contiguous()
        else:
            pres_mat = (pres_raw @ w64.T).contiguous()

    def _noop(ns_mode: str, ns_dim: int, reason: str) -> tuple[torch.Tensor, dict[str, Any]]:
        out_t = w_tensor if inplace else w_tensor.clone()
        return out_t, _build_oproj_noop_report(
            projection_mode=projection_mode,
            nullspace_mode=ns_mode,
            nullspace_dim=ns_dim,
            norm_before=norm_before,
            head_norm_before=head_norm_before,
            max_norm_delta_ratio=norm_cap,
            damp_alpha=alpha,
            noop_reason=reason,
        )

    if norm_before <= 1e-15 or head_norm_before <= 1e-15:
        return _noop("noop", 0, "zero_weight_norm")
    if alpha == 0.0:
        return _noop("noop", 0, "zero_damp_alpha")

    u_ctrl, ns_mode, ns_dim = _project_vector_into_nullspace(
        target_dir,
        pres_mat,
        nullspace_sv_rel_tol=sv_tol,
        reg_preserve=reg_p,
        shrink_preserve=shrink_preserve,
    )
    if float(u_ctrl.norm().item()) <= 1e-12:
        return _noop(ns_mode, ns_dim, ns_mode)

    if projection_mode == "head_input_nullspace":
        pre_col = w_h64 @ u_ctrl
        delta_w_h = -alpha * torch.outer(pre_col, u_ctrl)
    else:
        pre_row = u_ctrl @ w_h64
        delta_w_h = -alpha * torch.outer(u_ctrl, pre_row)

    raw_delta_norm = float(torch.linalg.norm(delta_w_h).item())
    raw_delta_ratio = raw_delta_norm / norm_before
    if raw_delta_norm <= 1e-15:
        return _noop(ns_mode, ns_dim, "zero_projection_overlap")

    tau_budget = norm_cap * norm_before
    if raw_delta_norm > tau_budget:
        budget_scale = tau_budget / raw_delta_norm
        delta_w_h = delta_w_h * budget_scale
        budget_clamped = True
    else:
        budget_scale = 1.0
        budget_clamped = False

    def _assemble(delta: torch.Tensor) -> torch.Tensor:
        w_cand64 = w64.clone()
        w_cand64[:, col_indices] += delta
        return w_cand64.to(device=w_tensor.device, dtype=w_tensor.dtype)

    w_out = _assemble(delta_w_h)
    if w_tensor.dtype != torch.float64:
        for shrink_factor in (0.999, 0.995, 0.985, 0.97):
            cand_norm = float(torch.linalg.norm(w_out.detach().to("cpu", dtype=torch.float64) - w64).item())
            if cand_norm <= tau_budget + 1e-12:
                break
            delta_w_h = delta_w_h * shrink_factor
            budget_scale *= shrink_factor
            budget_clamped = True
            w_out = _assemble(delta_w_h)

    if inplace:
        w_tensor.copy_(w_out)
        result_tensor = w_tensor
    else:
        result_tensor = w_out

    w_after64 = result_tensor.detach().to(device="cpu", dtype=torch.float64)
    w_h_after64 = w_after64[:, col_indices]
    applied_delta = float(torch.linalg.norm(w_after64 - w64).item())
    applied_ratio = applied_delta / norm_before
    head_delta_ratio = float(torch.linalg.norm(w_h_after64 - w_h64).item()) / head_norm_before

    if projection_mode == "head_input_nullspace":
        pre_norm = float(torch.linalg.norm(w_h64 @ u_ctrl).item())
        post_norm = float(torch.linalg.norm(w_h_after64 @ u_ctrl).item())
        if pres_mat is not None:
            p_base = float(torch.linalg.norm(pres_mat @ w_h64.T).item())
            p_diff = float(torch.linalg.norm(pres_mat @ (w_h_after64 - w_h64).T).item())
            pres_drift = (p_diff / p_base) if p_base > 1e-6 else p_diff
        else:
            pres_drift = 0.0
    else:
        pre_norm = float(torch.linalg.norm(u_ctrl @ w_h64).item())
        post_norm = float(torch.linalg.norm(u_ctrl @ w_h_after64).item())
        if pres_mat is not None:
            p_base = float(torch.linalg.norm(pres_mat).item())
            p_diff = float(torch.linalg.norm((pres_mat @ u_ctrl).unsqueeze(1) * (u_ctrl @ (w_h_after64 - w_h64))).item())
            pres_drift = (p_diff / p_base) if p_base > 1e-6 else p_diff
        else:
            pres_drift = 0.0

    suppression = max(0.0, 1.0 - (post_norm / pre_norm)) if pre_norm > 1e-12 else 0.0

    other_cols = [idx for idx in range(d_in) if idx not in set(col_indices)]
    non_target_ok = bool(torch.equal(w_after64[:, other_cols], w64[:, other_cols])) if other_cols else True

    return result_tensor, {
        "projection_mode": projection_mode,
        "nullspace_mode": ns_mode,
        "nullspace_dim": int(ns_dim),
        "norm_before": norm_before,
        "norm_after": float(torch.linalg.norm(w_after64).item()),
        "head_norm_before": head_norm_before,
        "head_norm_after": float(torch.linalg.norm(w_h_after64).item()),
        "raw_norm_delta": raw_delta_norm,
        "raw_norm_delta_ratio": raw_delta_ratio,
        "applied_norm_delta": applied_delta,
        "applied_norm_delta_ratio": applied_ratio,
        "head_norm_delta_ratio": head_delta_ratio,
        "max_norm_delta_ratio": norm_cap,
        "budget_clamped": budget_clamped,
        "budget_scale": float(budget_scale),
        "damp_alpha": alpha,
        "pre_spike_readout_norm": pre_norm,
        "post_spike_readout_norm": post_norm,
        "spike_suppression_ratio": suppression,
        "preserved_rms_drift": pres_drift,
        "non_target_columns_bitwise_unchanged": non_target_ok,
        "noop_reason": None,
    }


# ---------------------------------------------------------------------------
# 4. End-to-End Verified Attention-Sink Surgery with ANSE Gate & Rollback
# ---------------------------------------------------------------------------


def _validate_model_compatible(model: nn.Module) -> None:
    """Verify required CrowdV1/GQA-compatible transformer attributes on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")
    for idx, blk in enumerate(blocks):
        for b_attr in ("ln1", "attn", "ln2", "mlp"):
            if not hasattr(blk, b_attr):
                raise ValueError(f"model.blocks[{idx}] must expose '{b_attr}'")
        _resolve_attn_interface(blk.attn, block_idx=idx)


def _resolve_attn_interface(attn_mod: nn.Module, *, block_idx: int = 0) -> tuple[nn.Module, int, int, int]:
    """Return ``(o_proj_module, num_heads, num_kv_heads, head_dim)`` for Crowd or HF GQA blocks."""
    o_mod = getattr(attn_mod, "o", None) or getattr(attn_mod, "o_proj", None)
    if not isinstance(o_mod, nn.Module) or not isinstance(getattr(o_mod, "weight", None), torch.Tensor):
        raise ValueError(f"model.blocks[{block_idx}].attn must expose 'o' or 'o_proj' with a weight tensor")

    num_heads = getattr(attn_mod, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn_mod, "num_attention_heads", None)
    num_kv_heads = getattr(attn_mod, "num_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(attn_mod, "num_key_value_heads", None)
    head_dim = getattr(attn_mod, "head_dim", None)

    if num_heads is None or num_kv_heads is None or head_dim is None:
        raise ValueError(
            f"model.blocks[{block_idx}].attn must expose num_heads/num_kv_heads/head_dim metadata"
        )
    h_q = _validate_positive_int(int(num_heads), f"blocks[{block_idx}].attn.num_heads")
    h_kv = _validate_positive_int(int(num_kv_heads), f"blocks[{block_idx}].attn.num_kv_heads")
    d_h = _validate_positive_int(int(head_dim), f"blocks[{block_idx}].attn.head_dim")
    if h_q % h_kv != 0:
        raise ValueError(f"blocks[{block_idx}].attn num_heads ({h_q}) % num_kv_heads ({h_kv}) != 0")
    if int(o_mod.weight.shape[1]) != h_q * d_h:
        raise ValueError(
            f"blocks[{block_idx}].attn o_proj width {o_mod.weight.shape[1]} != {h_q} * {d_h}"
        )
    return o_mod, h_q, h_kv, d_h


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    vocab_size: int,
    score_from: int,
    general_prefix: str,
    require_general_and_other: bool,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunk dicts and stack IDs to CPU ``torch.long``."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    start_pos = max(0, score_from - 1)
    rows: list[torch.Tensor] = []
    domains: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping with 'expert_id' and 'ids'")
        eid = ch["expert_id"]
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        raw_ids = ch["ids"]
        if isinstance(raw_ids, torch.Tensor):
            if raw_ids.is_floating_point() or raw_ids.dtype == torch.bool:
                raise TypeError(f"{name}[{idx}]['ids'] must have an integer dtype")
            ids_t = raw_ids.detach().to(device="cpu", dtype=torch.long)
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            for elem in raw_ids:
                _validate_nonneg_int(elem, f"{name}[{idx}]['ids'] element")
            ids_t = torch.tensor(list(raw_ids), dtype=torch.long)
        else:
            raise TypeError(f"{name}[{idx}]['ids'] must be a sequence or 1D integer tensor")

        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
            if start_pos >= seq_len - 1:
                raise ValueError(f"{name} sequence length {seq_len} is too short for score_from={score_from}")
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}, got {ids_t.numel()}")
        rows.append(ids_t)
        domains.append(eid)

    ids_2d = torch.stack(rows, dim=0)
    is_e0 = np.array([d.startswith(general_prefix) for d in domains], dtype=bool)
    if require_general_and_other and (not bool(np.any(is_e0)) or not bool(np.any(~is_e0))):
        raise ValueError(
            f"{name} must contain both general ('{general_prefix}*') and non-general specialist chunks"
        )
    return ids_2d, is_e0, domains


def _compute_initial_embeddings(model: nn.Module, ids_2d: torch.Tensor) -> torch.Tensor:
    """Compute initial token embeddings (plus optional CrowdEngram) in ``float32``."""
    x = model.wte(ids_2d).to(torch.float32)
    engram_mod = getattr(model, "engram", None)
    engram_buckets = int(getattr(getattr(model, "cfg", None), "engram_buckets", 0))
    if engram_mod is not None and engram_buckets > 0:
        x = x + engram_mod(ids_2d).to(device=x.device, dtype=torch.float32)
    return x


def _apply_final_norm(ln_mod: nn.Module, h_pre_ln: torch.Tensor) -> torch.Tensor:
    """Apply final ``LayerNorm`` or biasless ``RMSNorm`` in ``float32``."""
    w = ln_mod.weight.to(device=h_pre_ln.device, dtype=torch.float32)
    b_raw = getattr(ln_mod, "bias", None)
    dim = int(w.numel())
    eps = float(getattr(ln_mod, "eps", getattr(ln_mod, "variance_epsilon", 1e-5)))
    if b_raw is None and (isinstance(ln_mod, nn.RMSNorm) or hasattr(ln_mod, "variance_epsilon") or "rms" in type(ln_mod).__name__.lower()):
        return F.rms_norm(h_pre_ln.to(torch.float32), (dim,), w, eps)
    b = b_raw.to(device=h_pre_ln.device, dtype=torch.float32) if b_raw is not None else None
    return F.layer_norm(h_pre_ln.to(torch.float32), (dim,), w, b, eps=eps)


@torch.no_grad()
def _forward_block_with_capture(
    blk: nn.Module,
    x_seq: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    o_mod: nn.Module,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Run one transformer block capturing pre-``o_proj`` head outputs, attention residual, and KV shape."""
    blk_dtype = blk.ln1.weight.dtype
    head_parts: list[torch.Tensor] = []
    attn_parts: list[torch.Tensor] = []
    out_parts: list[torch.Tensor] = []
    kv_shape: tuple[int, ...] = ()

    for b0 in range(0, x_seq.shape[0], batch_size):
        xb = x_seq[b0 : b0 + batch_size].to(dtype=blk_dtype)
        captured: list[torch.Tensor] = []

        def _pre_hook(
            _mod: nn.Module,
            inputs: tuple[Any, ...],
            bucket: list[torch.Tensor] = captured,
        ) -> None:
            bucket.append(inputs[0].detach().to("cpu", dtype=torch.float32))

        handle = o_mod.register_forward_pre_hook(_pre_hook)
        try:
            attn_out, next_kv = blk.attn(blk.ln1(xb), cos, sin, past_kv=None, use_cache=True)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError("o_proj forward pre-hook failed to capture attention head tensor")
        z_flat = captured[0]
        bsz, slen, _ = xb.shape
        head_parts.append(z_flat.reshape(bsz, slen, num_heads, head_dim))
        attn_parts.append(attn_out.detach().to("cpu", dtype=torch.float32))

        if next_kv is None or not isinstance(next_kv, tuple) or len(next_kv) != 2:
            raise ValueError("Attention module must return (k, v) cache tuple when use_cache=True")
        cur_kv_shape = tuple(int(d) for d in next_kv[0].shape)
        if tuple(next_kv[1].shape) != cur_kv_shape:
            raise ValueError(f"Mismatched K/V cache shapes: {cur_kv_shape} vs {tuple(next_kv[1].shape)}")
        if not kv_shape:
            kv_shape = (int(x_seq.shape[0]), cur_kv_shape[1], cur_kv_shape[2], cur_kv_shape[3])

        x_mid = xb + attn_out
        mlp_out = blk.mlp(blk.ln2(x_mid))
        out_parts.append((x_mid + mlp_out).detach().to("cpu", dtype=torch.float32))

    return (
        torch.cat(head_parts, dim=0),
        torch.cat(attn_parts, dim=0),
        torch.cat(out_parts, dim=0),
        kv_shape,
    )


@torch.no_grad()
def _evaluate_model_trajectory(
    model: nn.Module,
    ids_2d: torch.Tensor,
    *,
    sink_positions: Sequence[int],
    score_from: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run full model forward capturing per-chunk NLL, BOS sink representations, and KV shapes."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    sink_idx = torch.tensor(list(sink_positions), dtype=torch.long)
    sink_idx = sink_idx[sink_idx < int(ids_2d.shape[1])]

    x_parts = [
        _compute_initial_embeddings(model, ids_2d[b0 : b0 + batch_size].to(dev))
        for b0 in range(0, ids_2d.shape[0], batch_size)
    ]
    xb = torch.cat(x_parts, dim=0)

    kv_shapes: list[tuple[int, ...]] = []
    bos_attn_list: list[torch.Tensor] = []
    bos_res_list: list[torch.Tensor] = []
    per_layer_heads: list[torch.Tensor] = []

    for l_idx, blk in enumerate(model.blocks):
        o_mod, h_q, _, d_h = _resolve_attn_interface(blk.attn, block_idx=l_idx)
        z_heads, attn_out, xb_next, kv_shape = _forward_block_with_capture(
            blk,
            xb.to(dev),
            cos,
            sin,
            num_heads=h_q,
            head_dim=d_h,
            o_mod=o_mod,
            batch_size=batch_size,
        )
        kv_shapes.append(kv_shape)
        per_layer_heads.append(z_heads)
        if sink_idx.numel() > 0:
            bos_attn_list.append(attn_out[:, sink_idx, :].contiguous())
            bos_res_list.append(xb_next[:, sink_idx, :].contiguous())
        xb = xb_next

    if not bool(torch.isfinite(xb).all().item()):
        return {
            "finite": False,
            "losses": np.full(int(ids_2d.shape[0]), float("nan"), dtype=np.float64),
            "kv_shapes": kv_shapes,
            "bos_attn": bos_attn_list,
            "bos_res": bos_res_list,
            "per_layer_heads": per_layer_heads,
        }

    h_norm = _apply_final_norm(model.ln, xb.to(dev))
    start_pos = max(0, score_from - 1)
    logits = model.head(h_norm[:, start_pos:-1, :].to(dtype=model.head.weight.dtype)).to(torch.float32)
    targets = ids_2d[:, start_pos + 1 :].to(dev)
    vocab = int(logits.shape[-1])
    tok_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    losses = tok_loss.view(int(ids_2d.shape[0]), -1).mean(dim=1).detach().cpu().numpy().astype(np.float64)

    return {
        "finite": bool(np.isfinite(losses).all()),
        "losses": losses,
        "kv_shapes": kv_shapes,
        "bos_attn": bos_attn_list,
        "bos_res": bos_res_list,
        "per_layer_heads": per_layer_heads,
    }


def _compute_bos_anchor_drift(
    pre_attn: Sequence[torch.Tensor],
    post_attn: Sequence[torch.Tensor],
    pre_res: Sequence[torch.Tensor],
    post_res: Sequence[torch.Tensor],
) -> tuple[float, float]:
    """Compute max absolute and relative L2 drift across BOS sink anchor representations."""
    max_abs = 0.0
    max_rel = 0.0
    for a0, a1 in zip(pre_attn, post_attn, strict=True):
        diff = (a1 - a0).to(torch.float64)
        max_abs = max(max_abs, float(diff.abs().max().item()))
        base_n = max(float(torch.linalg.norm(a0.to(torch.float64)).item()), 1e-12)
        max_rel = max(max_rel, float(torch.linalg.norm(diff).item()) / base_n)
    for r0, r1 in zip(pre_res, post_res, strict=True):
        diff = (r1 - r0).to(torch.float64)
        max_abs = max(max_abs, float(diff.abs().max().item()))
        base_n = max(float(torch.linalg.norm(r0.to(torch.float64)).item()), 1e-12)
        max_rel = max(max_rel, float(torch.linalg.norm(diff).item()) / base_n)
    return max_abs, max_rel


def _measure_target_spike_residual_rms(
    model: nn.Module,
    traj: Mapping[str, Any],
    edited_targets: Sequence[Mapping[str, Any]],
    pos_2d: torch.Tensor,
    sink_positions: Sequence[int],
) -> float:
    """Measure RMS of post-``o_proj`` residual spikes along detected rogue head directions."""
    if not edited_targets:
        return 0.0
    sink_mask = torch.isin(pos_2d, torch.tensor(list(sink_positions), dtype=torch.long))
    mid_mask = ~sink_mask
    sq_sum = 0.0
    count = 0
    for tgt in edited_targets:
        l_idx = int(tgt["layer"])
        h_slice: slice = tgt["head_slice"]
        dir_vec: torch.Tensor = tgt["direction"]
        proj_mode: str = str(tgt["projection_mode"])
        h_idx = int(tgt["head_idx"])
        o_mod, _, _, _ = _resolve_attn_interface(model.blocks[l_idx].attn, block_idx=l_idx)
        w_h = o_mod.weight[:, h_slice].detach().to("cpu", dtype=torch.float64)
        z_mid = traj["per_layer_heads"][l_idx][mid_mask, h_idx, :].to(torch.float64)
        if proj_mode == "head_input_nullspace":
            proj = (z_mid @ dir_vec).unsqueeze(1) * (w_h @ dir_vec).unsqueeze(0)
        else:
            y_mid = z_mid @ w_h.T
            proj = (y_mid @ dir_vec).unsqueeze(1) * dir_vec.unsqueeze(0)
        sq_sum += float(proj.pow(2).sum().item())
        count += int(proj.shape[0])
    return math.sqrt(sq_sum / max(count, 1))


def _summarize_anse_metrics(
    base_losses: np.ndarray,
    cand_losses: np.ndarray,
    domains: Sequence[str],
    *,
    long_base_losses: np.ndarray | None,
    long_cand_losses: np.ndarray | None,
    bos_max_abs_drift: float,
    kv_and_rope_ok: bool,
    spike_rms: float,
    cfg: AttentionSinkSurgeonConfig,
) -> dict[str, Any]:
    """Evaluate per-domain drifts, BOS anchor guard, long-context guard, and ANSE energy."""
    finite_ok = bool(np.isfinite(cand_losses).all())
    if long_cand_losses is not None:
        finite_ok = finite_ok and bool(np.isfinite(long_cand_losses).all())

    bos_guard_ok = bool(bos_max_abs_drift <= cfg.bos_anchor_max_drift + 1e-12)
    if not finite_ok:
        return {
            "finite": False,
            "nll_mean": None,
            "base_nll_mean": float(base_losses.mean()),
            "general_pct_vs_base": None,
            "worst_domain_id": "",
            "worst_domain_ratio": None,
            "worst_domain_pct_vs_base": None,
            "long_context_pct_vs_base": None,
            "domain_guard_ok": False,
            "general_guard_ok": False,
            "long_context_guard_ok": False,
            "bos_guard_ok": bos_guard_ok,
            "kv_and_rope_ok": kv_and_rope_ok,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain_energy),
            "per_domain": {},
        }

    grouped: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        grouped.setdefault(dom, []).append(idx)

    per_domain: dict[str, dict[str, Any]] = {}
    for dom, ix in sorted(grouped.items()):
        b_m = float(base_losses[ix].mean())
        c_m = float(cand_losses[ix].mean())
        ratio = c_m / max(b_m, 1e-12)
        cap = cfg.general_guard_max_ratio if dom.startswith(cfg.general_prefix) else cfg.domain_guard_max_ratio
        per_domain[dom] = {
            "n": len(ix),
            "base_nll": b_m,
            "nll_mean": c_m,
            "mean_diff": c_m - b_m,
            "ratio": ratio,
            "pct_vs_base": 100.0 * (ratio - 1.0),
            "guard_ok": bool(ratio <= cap + 1e-12),
        }

    worst_dom, worst_info = max(per_domain.items(), key=lambda kv: float(kv[1]["ratio"]))
    worst_ratio = float(worst_info["ratio"])
    worst_pct = float(worst_info["pct_vs_base"])
    domain_guard_ok = all(bool(v["guard_ok"]) for v in per_domain.values())

    gen_ratios = [float(v["ratio"]) for d, v in per_domain.items() if d.startswith(cfg.general_prefix)]
    gen_ratio = max(gen_ratios) if gen_ratios else worst_ratio
    general_guard_ok = bool(gen_ratio <= cfg.general_guard_max_ratio + 1e-12)

    long_ratios = [float(v["ratio"]) for d, v in per_domain.items() if d.startswith(cfg.long_context_prefix)]
    if long_base_losses is not None and long_cand_losses is not None:
        long_ratio_extra = float(long_cand_losses.mean()) / max(float(long_base_losses.mean()), 1e-12)
        long_ratios.append(long_ratio_extra)
    long_ratio = max(long_ratios) if long_ratios else gen_ratio
    long_context_guard_ok = bool(long_ratio <= cfg.domain_guard_max_ratio + 1e-12)

    guard_ok = bool(domain_guard_ok and general_guard_ok and long_context_guard_ok and bos_guard_ok and kv_and_rope_ok)
    cand_mean = float(cand_losses.mean())
    base_mean = float(base_losses.mean())
    pain = (
        cfg.maximum_pain_energy
        if not guard_ok
        else (cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0)
    )
    anse_energy = float(cand_mean + cfg.spike_energy_weight * spike_rms + pain)

    return {
        "finite": True,
        "nll_mean": cand_mean,
        "base_nll_mean": base_mean,
        "general_pct_vs_base": 100.0 * (gen_ratio - 1.0),
        "worst_domain_id": worst_dom,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": worst_pct,
        "long_context_pct_vs_base": 100.0 * (long_ratio - 1.0),
        "domain_guard_ok": domain_guard_ok,
        "general_guard_ok": general_guard_ok,
        "long_context_guard_ok": long_context_guard_ok,
        "bos_guard_ok": bos_guard_ok,
        "kv_and_rope_ok": kv_and_rope_ok,
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


def _resolve_surgery_config(
    config: AttentionSinkSurgeonConfig | None,
    *,
    sink_positions: Sequence[int] | None,
    spike_ratio_threshold: float | None,
    damp_alpha: float | None,
    max_norm_delta_ratio: float | None,
    domain_guard_max_ratio: float | None,
    general_guard_max_ratio: float | None,
    bos_anchor_max_drift: float | None,
    score_from: int | None,
) -> AttentionSinkSurgeonConfig:
    """Merge optional keyword overrides into a validated ``AttentionSinkSurgeonConfig``."""
    if config is not None and not isinstance(config, AttentionSinkSurgeonConfig):
        raise TypeError(
            f"config must be AttentionSinkSurgeonConfig or None, got {type(config).__name__}"
        )
    base = config or AttentionSinkSurgeonConfig()
    return AttentionSinkSurgeonConfig(
        sink_positions=tuple(sink_positions) if sink_positions is not None else base.sink_positions,
        spike_ratio_threshold=(
            base.spike_ratio_threshold if spike_ratio_threshold is None else spike_ratio_threshold
        ),
        damp_alpha=base.damp_alpha if damp_alpha is None else damp_alpha,
        max_norm_delta_ratio=(
            base.max_norm_delta_ratio if max_norm_delta_ratio is None else max_norm_delta_ratio
        ),
        nullspace_sv_rel_tol=base.nullspace_sv_rel_tol,
        reg_preserve=base.reg_preserve,
        shrink_preserve=base.shrink_preserve,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio if domain_guard_max_ratio is None else domain_guard_max_ratio
        ),
        general_guard_max_ratio=(
            base.general_guard_max_ratio if general_guard_max_ratio is None else general_guard_max_ratio
        ),
        bos_anchor_max_drift=(
            base.bos_anchor_max_drift if bos_anchor_max_drift is None else bos_anchor_max_drift
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        spike_energy_weight=base.spike_energy_weight,
        general_prefix=base.general_prefix,
        long_context_prefix=base.long_context_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        require_strict_energy_drop=base.require_strict_energy_drop,
    )


@torch.no_grad()
def execute_verified_attention_sink_surgery(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    long_context_chunks: Sequence[Mapping[str, Any]] | None = None,
    config: AttentionSinkSurgeonConfig | None = None,
    layer_indices: Sequence[int] | None = None,
    sink_positions: Sequence[int] | None = None,
    spike_ratio_threshold: float | None = None,
    damp_alpha: float | None = None,
    max_norm_delta_ratio: float | None = None,
    domain_guard_max_ratio: float | None = None,
    general_guard_max_ratio: float | None = None,
    bos_anchor_max_drift: float | None = None,
    score_from: int | None = None,
    protect_bos_sinks: bool = True,
    protect_general_prose: bool = True,
    projection_space: str = "auto",
) -> dict[str, Any]:
    """Run verified sequential GQA ``o_proj`` null-space stabilization with ANSE rollback.

    Detects GQA heads carrying mid-sequence massive activation spikes, projects their
    ``o_proj`` readout into the null-space of BOS attention-sink anchor activations and
    general-prose representations, and commits edits iff ``Delta E < 0``, BOS anchor
    drift ``<= bos_anchor_max_drift``, general/long-context NLL drift ``<= +1.0%``, and
    KV-cache shapes and RoPE frequencies are unchanged; otherwise rolls back bit-for-bit.
    """
    _validate_model_compatible(model)
    _validate_bool(protect_bos_sinks, "protect_bos_sinks")
    _validate_bool(protect_general_prose, "protect_general_prose")
    if not isinstance(projection_space, str) or projection_space not in VALID_PROJECTION_SPACES:
        raise ValueError(
            f"projection_space must be one of {sorted(VALID_PROJECTION_SPACES)}, got {projection_space!r}"
        )

    cfg = _resolve_surgery_config(
        config,
        sink_positions=sink_positions,
        spike_ratio_threshold=spike_ratio_threshold,
        damp_alpha=damp_alpha,
        max_norm_delta_ratio=max_norm_delta_ratio,
        domain_guard_max_ratio=domain_guard_max_ratio,
        general_guard_max_ratio=general_guard_max_ratio,
        bos_anchor_max_drift=bos_anchor_max_drift,
        score_from=score_from,
    )

    n_blocks = len(model.blocks)
    if layer_indices is None:
        target_layer_set = set(range(n_blocks))
    else:
        if isinstance(layer_indices, (str, bytes, bytearray)) or not isinstance(layer_indices, Sequence):
            raise TypeError("layer_indices must be a sequence of block indices or None")
        t_layers = sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if not t_layers or any(idx >= n_blocks for idx in t_layers):
            raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
        target_layer_set = set(t_layers)

    vocab_size = int(model.wte.weight.shape[0])
    probe_ids, probe_is_e0, _ = _prepare_chunks(
        probe_chunks,
        name="probe_chunks",
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        general_prefix=cfg.general_prefix,
        require_general_and_other=True,
    )
    if eval_chunks is not None:
        verify_ids, _, verify_domains = _prepare_chunks(
            eval_chunks,
            name="eval_chunks",
            vocab_size=vocab_size,
            score_from=cfg.score_from,
            general_prefix=cfg.general_prefix,
            require_general_and_other=True,
        )
    else:
        verify_ids = probe_ids
        verify_domains = [str(ch["expert_id"]) for ch in probe_chunks]

    long_ids: torch.Tensor | None = None
    if long_context_chunks is not None:
        long_ids, _, _ = _prepare_chunks(
            long_context_chunks,
            name="long_context_chunks",
            vocab_size=vocab_size,
            score_from=cfg.score_from,
            general_prefix=cfg.general_prefix,
            require_general_and_other=False,
        )

    was_training = model.training
    model.eval()
    try:
        parent_snapshot = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)
        pre_cos_sha = compute_tensor_sha256(model.cos_cached)
        pre_sin_sha = compute_tensor_sha256(model.sin_cached)

        pre_traj = _evaluate_model_trajectory(
            model,
            verify_ids,
            sink_positions=cfg.sink_positions,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        if not pre_traj["finite"]:
            raise ValueError("Model produced non-finite baseline verification states or losses")

        pre_long_traj = (
            _evaluate_model_trajectory(
                model,
                long_ids,
                sink_positions=cfg.sink_positions,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )
            if long_ids is not None
            else None
        )
        if pre_long_traj is not None and not pre_long_traj["finite"]:
            raise ValueError("Model produced non-finite baseline long-context states or losses")

        dev = model.wte.weight.device
        cos = model.cos_cached.to(dev)
        sin = model.sin_cached.to(dev)
        probe_seq_len = int(probe_ids.shape[1])
        probe_pos_2d = torch.arange(probe_seq_len, dtype=torch.long).unsqueeze(0).expand(int(probe_ids.shape[0]), probe_seq_len)
        probe_sink_mask = torch.isin(probe_pos_2d, torch.tensor(list(cfg.sink_positions), dtype=torch.long))
        probe_mid_mask = ~probe_sink_mask
        e0_mask_t = torch.from_numpy(probe_is_e0).unsqueeze(1).expand_as(probe_mid_mask)
        prose_mid_mask = probe_mid_mask & e0_mask_t

        verify_seq_len = int(verify_ids.shape[1])
        verify_pos_2d = torch.arange(verify_seq_len, dtype=torch.long).unsqueeze(0).expand(int(verify_ids.shape[0]), verify_seq_len)

        xb_probe = torch.cat(
            [
                _compute_initial_embeddings(model, probe_ids[b0 : b0 + cfg.batch_size].to(dev))
                for b0 in range(0, probe_ids.shape[0], cfg.batch_size)
            ],
            dim=0,
        )

        layer_reports: list[dict[str, Any]] = []
        edited_targets: list[dict[str, Any]] = []
        all_detections: list[dict[str, Any]] = []

        for l_idx, blk in enumerate(model.blocks):
            o_mod, h_q, h_kv, d_h = _resolve_attn_interface(blk.attn, block_idx=l_idx)
            z_heads, _, xb_next, _ = _forward_block_with_capture(
                blk,
                xb_probe.to(dev),
                cos,
                sin,
                num_heads=h_q,
                head_dim=d_h,
                o_mod=o_mod,
                batch_size=cfg.batch_size,
            )

            det = detect_massive_sink_heads_and_channels(
                z_heads,
                probe_pos_2d,
                sink_positions=cfg.sink_positions,
                spike_ratio_threshold=cfg.spike_ratio_threshold,
                num_kv_heads=h_kv,
                o_proj_weight=o_mod.weight,
            )
            all_detections.append(
                {
                    "layer": l_idx,
                    "benign_bos_sink_heads": det["benign_bos_sink_heads"],
                    "benign_bos_sink_kv_heads": det["benign_bos_sink_kv_heads"],
                    "rogue_spike_heads": det["rogue_spike_heads"],
                    "rogue_spike_kv_heads": det["rogue_spike_kv_heads"],
                    "dual_role_heads": det["dual_role_heads"],
                    "max_mid_spike_ratio": det["max_mid_spike_ratio"],
                    "max_bos_sink_ratio": det["max_bos_sink_ratio"],
                }
            )

            if l_idx in target_layer_set and det["rogue_spike_heads"]:
                for h_idx in det["rogue_spike_heads"]:
                    h_slice: slice = det["rogue_head_slices"][h_idx]
                    w_h_before = o_mod.weight[:, h_slice].detach().to("cpu", dtype=torch.float64)
                    z_h_all = z_heads[:, :, h_idx, :].to(torch.float64)

                    head_pres_parts: list[torch.Tensor] = []
                    res_pres_parts: list[torch.Tensor] = []
                    if protect_bos_sinks and bool(probe_sink_mask.any().item()):
                        z_bos_h = z_h_all[probe_sink_mask]
                        head_pres_parts.append(z_bos_h)
                        res_pres_parts.append(z_bos_h @ w_h_before.T)
                    if protect_general_prose and bool(prose_mid_mask.any().item()):
                        z_prose_h = z_h_all[prose_mid_mask]
                        head_pres_parts.append(z_prose_h)
                        res_pres_parts.append(z_prose_h @ w_h_before.T)

                    head_pres = torch.cat(head_pres_parts, dim=0) if head_pres_parts else None
                    res_pres = torch.cat(res_pres_parts, dim=0) if res_pres_parts else None

                    r_head = det["rogue_head_directions"][h_idx]
                    r_res = det["rogue_residual_directions"][h_idx]

                    chosen_space = projection_space
                    if chosen_space == "auto":
                        _, hm, _ = _project_vector_into_nullspace(
                            r_head,
                            head_pres,
                            nullspace_sv_rel_tol=cfg.nullspace_sv_rel_tol,
                            reg_preserve=cfg.reg_preserve,
                            shrink_preserve=cfg.shrink_preserve,
                        )
                        _, rm, _ = _project_vector_into_nullspace(
                            r_res,
                            res_pres,
                            nullspace_sv_rel_tol=cfg.nullspace_sv_rel_tol,
                            reg_preserve=cfg.reg_preserve,
                            shrink_preserve=cfg.shrink_preserve,
                        )
                        chosen_space = "residual" if (rm == "exact_nullspace" and hm != "exact_nullspace") else "head"

                    if chosen_space == "head":
                        r_dir, pres_acts = r_head, head_pres
                    else:
                        r_dir, pres_acts = r_res, res_pres

                    _, proj_rep = project_oproj_rogue_spike_into_prose_nullspace(
                        o_mod.weight,
                        h_slice,
                        r_dir,
                        pres_acts,
                        damp_alpha=cfg.damp_alpha,
                        max_norm_delta_ratio=cfg.max_norm_delta_ratio,
                        nullspace_sv_rel_tol=cfg.nullspace_sv_rel_tol,
                        reg_preserve=cfg.reg_preserve,
                        shrink_preserve=cfg.shrink_preserve,
                        inplace=True,
                    )
                    layer_reports.append(
                        {
                            "layer": l_idx,
                            "head_idx": h_idx,
                            "kv_head_idx": h_idx // det["gqa_group_size"],
                            **proj_rep,
                        }
                    )
                    if float(proj_rep["applied_norm_delta"]) > 0.0:
                        edited_targets.append(
                            {
                                "layer": l_idx,
                                "head_idx": h_idx,
                                "head_slice": h_slice,
                                "direction": r_dir,
                                "projection_mode": proj_rep["projection_mode"],
                            }
                        )

                # Refresh sequential trunk states after editing layer l_idx
                _, _, xb_probe, _ = _forward_block_with_capture(
                    blk,
                    xb_probe.to(dev),
                    cos,
                    sin,
                    num_heads=h_q,
                    head_dim=d_h,
                    o_mod=o_mod,
                    batch_size=cfg.batch_size,
                )
            else:
                xb_probe = xb_next

        candidate_sha = compute_state_sha256(model.state_dict())
        post_cos_sha = compute_tensor_sha256(model.cos_cached)
        post_sin_sha = compute_tensor_sha256(model.sin_cached)
        rope_unchanged = (post_cos_sha == pre_cos_sha) and (post_sin_sha == pre_sin_sha)

        cand_traj = _evaluate_model_trajectory(
            model,
            verify_ids,
            sink_positions=cfg.sink_positions,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        cand_long_traj = (
            _evaluate_model_trajectory(
                model,
                long_ids,
                sink_positions=cfg.sink_positions,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )
            if long_ids is not None
            else None
        )

        kv_shape_unchanged = cand_traj["kv_shapes"] == pre_traj["kv_shapes"]
        if cand_long_traj is not None and pre_long_traj is not None:
            kv_shape_unchanged = kv_shape_unchanged and (cand_long_traj["kv_shapes"] == pre_long_traj["kv_shapes"])
        kv_and_rope_ok = bool(kv_shape_unchanged and rope_unchanged)

        bos_max_abs, bos_max_rel = _compute_bos_anchor_drift(
            pre_traj["bos_attn"],
            cand_traj["bos_attn"],
            pre_traj["bos_res"],
            cand_traj["bos_res"],
        )

        # Measure residual spike RMS pre- vs post-surgery using parent snapshot weights for pre
        post_spike_rms = _measure_target_spike_residual_rms(
            model, cand_traj, edited_targets, verify_pos_2d, cfg.sink_positions
        )
        if edited_targets:
            cur_sd = model.state_dict()
            cand_snap = {k: v.detach().to("cpu").clone() for k, v in cur_sd.items()}
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )
            pre_spike_rms = _measure_target_spike_residual_rms(
                model, pre_traj, edited_targets, verify_pos_2d, cfg.sink_positions
            )
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in cand_snap.items()}
            )
        else:
            pre_spike_rms = 0.0

        spike_suppression_ratio = (
            max(0.0, 1.0 - (post_spike_rms / pre_spike_rms)) if pre_spike_rms > 1e-12 else 0.0
        )

        parent_eval = _summarize_anse_metrics(
            pre_traj["losses"],
            pre_traj["losses"],
            verify_domains,
            long_base_losses=pre_long_traj["losses"] if pre_long_traj is not None else None,
            long_cand_losses=pre_long_traj["losses"] if pre_long_traj is not None else None,
            bos_max_abs_drift=0.0,
            kv_and_rope_ok=True,
            spike_rms=pre_spike_rms,
            cfg=cfg,
        )
        cand_eval = _summarize_anse_metrics(
            pre_traj["losses"],
            cand_traj["losses"],
            verify_domains,
            long_base_losses=pre_long_traj["losses"] if pre_long_traj is not None else None,
            long_cand_losses=cand_long_traj["losses"] if cand_long_traj is not None else None,
            bos_max_abs_drift=bos_max_abs,
            kv_and_rope_ok=kv_and_rope_ok,
            spike_rms=post_spike_rms,
            cfg=cfg,
        )

        parent_e = float(parent_eval["anse_energy"])
        cand_e = float(cand_eval["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED" if cand_eval["guard_ok"] else "ROLLED_BACK_GUARD_VIOLATION"
            accepted = False
            rolled_back = False
        elif bool(cand_eval["guard_ok"]) and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if not bool(cand_eval["bos_guard_ok"]):
                status = "ROLLED_BACK_BOS_ANCHOR_GUARD_VIOLATION"
            elif not bool(cand_eval["kv_and_rope_ok"]):
                status = "ROLLED_BACK_KV_OR_ROPE_INVARIANT_VIOLATION"
            elif not bool(cand_eval["guard_ok"]):
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            cur_sd = model.state_dict()
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_state_sha256(model.state_dict())
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
    finally:
        model.train(was_training)

    active_eval = cand_eval if accepted else parent_eval
    active_spike_rms = post_spike_rms if accepted else pre_spike_rms
    active_suppression = spike_suppression_ratio if accepted else 0.0

    receipt: dict[str, Any] = {
        "object": SURGEON_RECEIPT_OBJECT,
        "schema": SURGEON_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "domain_guard_ok": bool(cand_eval["domain_guard_ok"]),
        "general_guard_ok": bool(cand_eval["general_guard_ok"]),
        "long_context_guard_ok": bool(cand_eval["long_context_guard_ok"]),
        "bos_guard_ok": bool(cand_eval["bos_guard_ok"]),
        "kv_shape_unchanged": bool(kv_shape_unchanged),
        "rope_frequencies_unchanged": bool(rope_unchanged),
        "kv_shapes_per_layer": [list(s) for s in pre_traj["kv_shapes"]],
        "nll_mean": float(active_eval["nll_mean"]),
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "sink_norm_ratio": float(max((d["max_mid_spike_ratio"] for d in all_detections), default=0.0)),
        "pre_spike_rms": float(pre_spike_rms),
        "post_spike_rms": float(post_spike_rms),
        "active_spike_rms": float(active_spike_rms),
        "spike_suppression_ratio": float(active_suppression),
        "candidate_spike_suppression_ratio": float(spike_suppression_ratio),
        "bos_anchor_max_abs_drift": float(bos_max_abs),
        "bos_anchor_max_rel_drift": float(bos_max_rel),
        "anse_energy": float(cand_e),
        "parent_anse_energy": float(parent_e),
        "final_anse_energy": float(cand_e if accepted else parent_e),
        "delta_e": float(delta_e),
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "long_context_pct_vs_base": cand_eval["long_context_pct_vs_base"],
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "per_domain": cand_eval["per_domain"],
        "detections_per_layer": all_detections,
        "layer_reports": layer_reports,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BOS_ANCHOR_MAX_DRIFT",
    "DEFAULT_DAMP_ALPHA",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GENERAL_GUARD_MAX_RATIO",
    "DEFAULT_MAX_NORM_DELTA_RATIO",
    "DEFAULT_MIN_ABS_MAGNITUDE",
    "DEFAULT_NULLSPACE_SV_REL_TOL",
    "DEFAULT_PRESERVE_REG",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SINK_POSITIONS",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SPIKE_ENERGY_WEIGHT",
    "DEFAULT_SPIKE_RATIO_THRESHOLD",
    "GENERAL_EXPERT_PREFIX",
    "LONG_CONTEXT_PREFIX",
    "SURGEON_RECEIPT_OBJECT",
    "SURGEON_SCHEMA",
    "VALID_PROJECTION_SPACES",
    "AttentionSinkSurgeonConfig",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "compute_state_sha256",
    "compute_tensor_sha256",
    "detect_massive_sink_heads_and_channels",
    "execute_verified_attention_sink_surgery",
    "project_oproj_rogue_spike_into_prose_nullspace",
    "verify_attention_sink_receipt_provenance",
]
