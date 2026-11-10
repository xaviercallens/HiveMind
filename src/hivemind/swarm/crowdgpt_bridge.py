# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CrowdGPT x HiveMind DiLoCo, Multi-Krum & Loss-Ratchet Synergy Bridge.

Provides drop-in interoperability between CrowdGPT (`Vxtzq/CrowdGPT`, `Vxtzq/Crowd-v1`)
and HiveMind (`xaviercallens/HiveMind`):
1. Exact binary parser/packer for CrowdGPT's gzipped delta stream
   (`struct.pack('<I', len(meta_json)) + meta_json + bf16_uint16_delta + engram_sparse`).
2. Chunked Multi-Krum geometric Byzantine filtering over 1D `bfloat16` deltas or
   `safetensors` state-dicts (`dict[str, torch.Tensor]`), weighted by `tokensProcessed`.
3. Monotonic Held-Out Loss Ratchet (`LossRatchetGate`) with SHA-256 checkpoint receipts.
4. Zero-cost OpenAI-compatible playground helper for `hivemind/crowd-v1` (`Vxtzq/Crowd-v1`).
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import statistics
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import torch

DEFAULT_HIVEMIND_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
CROWDGPT_HF_REPO_ID = "Vxtzq/Crowd-v1"
HIVEMIND_CROWD_MODEL_ID = "hivemind/crowd-v1"


@dataclass(frozen=True)
class CrowdGPTDeltaPayload:
    """Unpacked representation of a CrowdGPT client delta upload."""

    metadata: dict[str, Any]
    base_delta_bf16: torch.Tensor
    engram_indices: np.ndarray
    engram_values_uint16: np.ndarray
    payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if not isinstance(self.base_delta_bf16, torch.Tensor) or self.base_delta_bf16.dtype != torch.bfloat16:
            raise TypeError("base_delta_bf16 must be a bfloat16 torch.Tensor")
        if not isinstance(self.engram_indices, np.ndarray) or not isinstance(self.engram_values_uint16, np.ndarray):
            raise TypeError("engram_indices and engram_values_uint16 must be numpy.ndarray instances")
        if not isinstance(self.payload_sha256, str) or len(self.payload_sha256.strip()) != 64:
            raise ValueError("payload_sha256 must be a 64-character hex string")


def pack_crowdgpt_delta_stream(
    metadata: Mapping[str, Any],
    base_delta_bf16: torch.Tensor,
    engram_indices: np.ndarray | None = None,
    engram_values_uint16: np.ndarray | None = None,
    *,
    compresslevel: int = 6,
) -> bytes:
    """Pack a CrowdGPT delta in the exact wire format produced by `client.py`."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a Mapping.")
    if not isinstance(base_delta_bf16, torch.Tensor):
        raise TypeError("base_delta_bf16 must be a torch.Tensor.")
    if isinstance(compresslevel, bool) or not isinstance(compresslevel, int) or not (0 <= compresslevel <= 9):
        raise ValueError("compresslevel must be an integer in [0, 9].")
    if base_delta_bf16.dtype != torch.bfloat16:
        base_delta_bf16 = base_delta_bf16.to(torch.bfloat16)
    flat_uint16 = (
        base_delta_bf16.detach()
        .to("cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    si = (
        np.ascontiguousarray(engram_indices, dtype=np.uint32)
        if engram_indices is not None and len(engram_indices) > 0
        else np.array([], dtype=np.uint32)
    )
    sv = (
        np.ascontiguousarray(engram_values_uint16, dtype=np.uint16)
        if engram_values_uint16 is not None and len(engram_values_uint16) > 0
        else np.array([], dtype=np.uint16)
    )
    meta_dict = dict(metadata)
    meta_dict.setdefault("loraRank", 0)
    meta_dict.setdefault("isDelta", True)
    meta_dict.setdefault("weightFormat", "bf16")
    meta_dict["hasEngram"] = bool(len(si) > 0)
    meta_dict["engramSparseCount"] = int(len(si))

    pl_bytes = json.dumps(meta_dict, separators=(",", ":")).encode("utf-8")
    header = struct.pack("<I", len(pl_bytes)) + pl_bytes

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=compresslevel) as gz:
        gz.write(header)
        for arr in (flat_uint16, si, sv):
            mv = memoryview(np.ascontiguousarray(arr)).cast("B")
            gz.write(mv)
    return buf.getvalue()


def unpack_crowdgpt_delta_stream(
    compressed_data: bytes,
    *,
    engram_row_dim: int = 1536,
) -> CrowdGPTDeltaPayload:
    """Unpack a gzipped CrowdGPT client delta stream and verify its structure."""
    if not isinstance(compressed_data, (bytes, bytearray)) or not compressed_data:
        raise ValueError("compressed_data must be non-empty bytes.")
    if isinstance(engram_row_dim, bool) or not isinstance(engram_row_dim, int) or engram_row_dim < 1:
        raise ValueError("engram_row_dim must be an integer >= 1.")
    payload_sha256 = hashlib.sha256(compressed_data).hexdigest()
    raw = gzip.decompress(compressed_data)
    if len(raw) < 4:
        raise ValueError("CrowdGPT delta stream is shorter than the 4-byte header prefix.")

    meta_len = struct.unpack("<I", raw[:4])[0]
    meta_end = 4 + meta_len
    if len(raw) < meta_end:
        raise ValueError("CrowdGPT delta stream truncated inside JSON metadata header.")

    metadata = json.loads(raw[4:meta_end].decode("utf-8"))
    body = memoryview(raw)[meta_end:]

    has_engram = bool(metadata.get("hasEngram", False))
    sparse_count = int(metadata.get("engramSparseCount", 0)) if has_engram else 0
    if sparse_count < 0:
        raise ValueError("engramSparseCount cannot be negative.")
    engram_idx_bytes = sparse_count * 4
    engram_val_bytes = sparse_count * engram_row_dim * 2
    engram_total_bytes = engram_idx_bytes + engram_val_bytes

    if len(body) < engram_total_bytes:
        raise ValueError("CrowdGPT delta stream smaller than declared sparse engram tail.")

    base_bytes_len = len(body) - engram_total_bytes
    if base_bytes_len % 2 != 0:
        raise ValueError("CrowdGPT bf16 base delta byte length must be divisible by 2.")

    base_int16 = np.frombuffer(body[:base_bytes_len], dtype=np.int16).copy()
    base_bf16 = torch.from_numpy(base_int16).view(torch.bfloat16)

    if sparse_count > 0:
        si = np.frombuffer(
            body[base_bytes_len : base_bytes_len + engram_idx_bytes], dtype=np.uint32
        ).copy()
        sv = np.frombuffer(
            body[base_bytes_len + engram_idx_bytes :], dtype=np.uint16
        ).copy()
    else:
        si = np.array([], dtype=np.uint32)
        sv = np.array([], dtype=np.uint16)

    return CrowdGPTDeltaPayload(
        metadata=metadata,
        base_delta_bf16=base_bf16,
        engram_indices=si,
        engram_values_uint16=sv,
        payload_sha256=payload_sha256,
    )


def _flatten_delta(
    delta: torch.Tensor | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(delta, torch.Tensor):
        return delta.detach().to("cpu", dtype=torch.float32).reshape(-1)
    parts = [
         delta[k].detach().to("cpu", dtype=torch.float32).reshape(-1)
        for k in sorted(delta.keys())
    ]
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.float32)


def chunked_pairwise_sq_l2(
    deltas: Sequence[torch.Tensor | Mapping[str, torch.Tensor]],
    *,
    chunk_elements: int = 1_048_576,
) -> torch.Tensor:
    """Compute the N x N pairwise squared L2 distance matrix in memory-bounded chunks."""
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int) or chunk_elements < 1:
        raise ValueError("chunk_elements must be an integer >= 1.")
    n = len(deltas)
    if n == 0:
        raise ValueError("deltas must be non-empty.")
    flat = [_flatten_delta(d) for d in deltas]
    dim = flat[0].numel()
    for idx, vec in enumerate(flat):
        if vec.numel() != dim:
            raise ValueError(f"Delta {idx} has {vec.numel()} elements, expected {dim}.")
        if not bool(torch.isfinite(vec).all()):
            raise ValueError(f"Delta {idx} contains non-finite (NaN/Inf) values.")

    dist_sq = torch.zeros((n, n), dtype=torch.float64)
    for start in range(0, dim, chunk_elements):
        end = min(dim, start + chunk_elements)
        block = torch.stack([v[start:end] for v in flat], dim=0).to(torch.float64)
        norms = (block * block).sum(dim=1, keepdim=True)
        gram = block @ block.T
        chunk_d2 = torch.clamp(norms + norms.T - 2.0 * gram, min=0.0)
        dist_sq += chunk_d2
    dist_sq.fill_diagonal_(0.0)
    return dist_sq.to(torch.float32)


MAX_TOKEN_WEIGHT_MEDIAN_MULTIPLE: float = 1.25


@dataclass(frozen=True)
class MultiKrumAggregationResult:
    """Result of Multi-Krum filtering and token-weighted DiLoCo aggregation."""

    selected_indices: list[int]
    rejected_indices: list[int]
    krum_scores: list[float]
    aggregated_delta_bf16: torch.Tensor
    effective_tokens: int
    byzantine_filter_active: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selected_indices, list)
            or not self.selected_indices
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.selected_indices)
        ):
            raise ValueError("selected_indices must be a non-empty list of non-negative integers")
        if (
            not isinstance(self.rejected_indices, list)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.rejected_indices)
        ):
            raise ValueError("rejected_indices must be a list of non-negative integers")
        if (
            not isinstance(self.krum_scores, list)
            or not self.krum_scores
            or any(
                isinstance(s, bool)
                or not isinstance(s, (int, float))
                or not math.isfinite(float(s))
                or float(s) < 0.0
                for s in self.krum_scores
            )
        ):
            raise ValueError("krum_scores must be a non-empty list of finite non-negative numbers")
        if not isinstance(self.aggregated_delta_bf16, torch.Tensor) or self.aggregated_delta_bf16.dtype != torch.bfloat16:
            raise TypeError("aggregated_delta_bf16 must be a bfloat16 torch.Tensor")
        if isinstance(self.effective_tokens, bool) or not isinstance(self.effective_tokens, int) or self.effective_tokens < 1:
            raise ValueError("effective_tokens must be a positive integer")
        if not isinstance(self.byzantine_filter_active, bool):
            raise ValueError("byzantine_filter_active must be a boolean")


def _validate_and_clamp_token_weights(
    token_weights: Sequence[int | float],
    expected_len: int,
) -> list[float]:
    """Validate finite positive token weights and clamp outliers to a multiple of the median."""
    if len(token_weights) != expected_len:
        raise ValueError("token_weights length must match deltas length.")
    parsed: list[float] = []
    for w in token_weights:
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"token_weights must be finite and positive, got {w!r}")
        fw = float(w)
        if not math.isfinite(fw) or fw <= 0.0:
            raise ValueError(f"token_weights must be finite and positive, got {w!r}")
        parsed.append(max(1.0, fw))
    med_w = float(statistics.median(parsed))
    cap_w = max(1.0, MAX_TOKEN_WEIGHT_MEDIAN_MULTIPLE * med_w)
    return [min(w, cap_w) for w in parsed]


def multi_krum_select_and_aggregate(
    deltas: Sequence[torch.Tensor | Mapping[str, torch.Tensor]],
    *,
    f_byzantine: int = 1,
    m_select: int | None = None,
    token_weights: Sequence[int | float] | None = None,
    chunk_elements: int = 1_048_576,
) -> MultiKrumAggregationResult:
    """Select `m_select` Byzantine-resilient deltas via Multi-Krum and aggregate in bf16.

    When `n < 2 * f_byzantine + 3`, automatically falls back to `f_byzantine = max(0, (n - 3) // 2)`
    so small contributor rounds still complete cleanly.
    """
    n = len(deltas)
    if n == 0:
        raise ValueError("At least one delta is required.")

    clamped_all_w = (
        _validate_and_clamp_token_weights(token_weights, n)
        if token_weights is not None
        else None
    )

    flat = [_flatten_delta(d) for d in deltas]
    if n == 1:
        tok = int(round(clamped_all_w[0])) if clamped_all_w is not None else 1
        return MultiKrumAggregationResult(
            selected_indices=[0],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=flat[0].to(torch.bfloat16),
            effective_tokens=max(1, tok),
            byzantine_filter_active=False,
        )

    eff_f = min(max(0, f_byzantine), max(0, (n - 3) // 2))
    neighbors_k = max(1, n - eff_f - 2)
    m = m_select if m_select is not None else max(1, n - eff_f)
    m = min(n, max(1, m))

    dist_sq = chunked_pairwise_sq_l2(flat, chunk_elements=chunk_elements)
    scores: list[float] = []
    for i in range(n):
        row = torch.cat([dist_sq[i, :i], dist_sq[i, i + 1 :]])
        nearest, _ = torch.topk(row, k=min(neighbors_k, row.numel()), largest=False)
        scores.append(float(nearest.sum().item()))

    ranked = sorted(range(n), key=lambda idx: (scores[idx], idx))
    candidate_ranked = ranked[:m]
    extra_rejected = ranked[m:]

    if m_select is None and eff_f > 0 and len(candidate_ranked) > 1:
        core_count = max(1, (n + 1) // 2)
        cluster_med = float(statistics.median(scores[idx] for idx in ranked[:core_count]))
        score_ceiling = 4.0 * max(cluster_med, 1e-9)
        min_keep = max(1, n - 2 * eff_f)
        filtered: list[int] = []
        pruned: list[int] = []
        for pos, idx in enumerate(candidate_ranked):
            if pos < min_keep or scores[idx] <= score_ceiling:
                filtered.append(idx)
            else:
                pruned.append(idx)
        selected = sorted(filtered)
        rejected = sorted(pruned + extra_rejected)
    else:
        selected = sorted(candidate_ranked)
        rejected = sorted(extra_rejected)

    if clamped_all_w is not None:
        sel_med = float(statistics.median(clamped_all_w[i] for i in selected))
        sel_cap = max(1.0, MAX_TOKEN_WEIGHT_MEDIAN_MULTIPLE * sel_med)
        raw_w = [min(clamped_all_w[i], sel_cap) for i in selected]
    else:
        raw_w = [1.0 for _ in selected]

    total_w = sum(raw_w)
    agg = torch.zeros_like(flat[0], dtype=torch.float32)
    for idx, w in zip(selected, raw_w):
        agg.add_(flat[idx], alpha=w / total_w)

    eff_tokens = (
        int(sum(int(round(w)) for w in raw_w))
        if clamped_all_w is not None
        else len(selected)
    )
    filter_active = bool(eff_f > 0 and len(selected) < n)
    return MultiKrumAggregationResult(
        selected_indices=selected,
        rejected_indices=rejected,
        krum_scores=scores,
        aggregated_delta_bf16=agg.to(torch.bfloat16),
        effective_tokens=eff_tokens,
        byzantine_filter_active=filter_active,
    )


@dataclass(frozen=True)
class LossRatchetDecision:
    """Audit receipt for a held-out Loss Ratchet evaluation."""

    accepted: bool
    loss_before: float
    loss_after: float
    delta_loss: float
    checkpoint_sha256: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        for f_name in ("loss_before", "loss_after", "delta_loss"):
            val = getattr(self, f_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{f_name} must be a number")
            if self.accepted and not math.isfinite(float(val)):
                raise ValueError(f"{f_name} must be finite when accepted is True")
        for s_name in ("checkpoint_sha256", "reason"):
            val = getattr(self, s_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{s_name} must be a non-empty string")


class LossRatchetGate:
    """Monotonic held-out validation gate for outer DiLoCo / CrowdGPT merges."""

    def __init__(self, *, max_regression_tolerance: float = 0.0) -> None:
        if (
            isinstance(max_regression_tolerance, bool)
            or not isinstance(max_regression_tolerance, (int, float))
            or not math.isfinite(float(max_regression_tolerance))
        ):
            raise ValueError("max_regression_tolerance must be a finite number.")
        self.max_regression_tolerance = float(max_regression_tolerance)

    def evaluate(
        self,
        *,
        loss_before: float,
        loss_after: float,
        candidate_delta_bf16: torch.Tensor,
    ) -> LossRatchetDecision:
        raw_bytes = (
            candidate_delta_bf16.detach()
            .to("cpu", dtype=torch.bfloat16)
            .contiguous()
            .view(torch.int16)
            .numpy()
            .tobytes()
        )
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if not (np.isfinite(loss_before) and np.isfinite(loss_after)):
            return LossRatchetDecision(
                accepted=False,
                loss_before=float(loss_before),
                loss_after=float(loss_after),
                delta_loss=float("nan"),
                checkpoint_sha256=digest,
                reason="REJECTED_NON_FINITE_LOSS",
            )
        delta_loss = float(loss_after - loss_before)
        accepted = delta_loss <= self.max_regression_tolerance
        reason = (
            f"ACCEPTED (delta_loss={delta_loss:+.5f} <= tol={self.max_regression_tolerance:.5f})"
            if accepted
            else f"HELD_PREVIOUS_HEAD (delta_loss={delta_loss:+.5f} > tol={self.max_regression_tolerance:.5f})"
        )
        return LossRatchetDecision(
            accepted=accepted,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            delta_loss=delta_loss,
            checkpoint_sha256=digest,
            reason=reason,
        )


@dataclass(frozen=True)
class WeightRmsGuardDecision:
    """Audit decision for the CrowdSwarm v2 Anti-Lobotomy Weight RMS Collapse Guard (`HM-S17-CW-02`).

    Prevents the Step-282 weight-halving failure mode diagnosed by `@jpmlima` in `Vxtzq/CrowdGPT#48`,
    where averaging a zeroed participant tensor scaled trunk weights by 0.5x (`rms 0.0387 -> 0.0195`)
    and reset held-out loss to ~12.6.
    """

    accepted: bool
    rms_old: float
    rms_new: float
    rms_ratio: float
    halving_residual_rms: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        for f_name in ("rms_old", "rms_new", "rms_ratio", "halving_residual_rms"):
            val = getattr(self, f_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{f_name} must be a number")
            if self.accepted:
                if not math.isfinite(float(val)):
                    raise ValueError(f"{f_name} must be finite when accepted is True")
                if f_name == "halving_residual_rms" and float(val) < 0.0:
                    raise ValueError("halving_residual_rms must be non-negative")
                if f_name != "halving_residual_rms" and float(val) <= 0.0:
                    raise ValueError(f"{f_name} must be positive when accepted is True")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def check_weight_rms_collapse_guard(
    old_weights: torch.Tensor | Mapping[str, torch.Tensor],
    new_weights: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    min_rms_ratio: float = 0.80,
    max_rms_ratio: float = 1.25,
    halving_residual_tol: float = 1e-3,
) -> WeightRmsGuardDecision:
    """Reject aggregated checkpoints whose global RMS norm collapses, explodes, or halves (`0.5 * old`)."""
    if (
        isinstance(min_rms_ratio, bool)
        or not isinstance(min_rms_ratio, (int, float))
        or not math.isfinite(float(min_rms_ratio))
        or float(min_rms_ratio) <= 0.0
        or isinstance(max_rms_ratio, bool)
        or not isinstance(max_rms_ratio, (int, float))
        or not math.isfinite(float(max_rms_ratio))
        or float(max_rms_ratio) < float(min_rms_ratio)
        or isinstance(halving_residual_tol, bool)
        or not isinstance(halving_residual_tol, (int, float))
        or not math.isfinite(float(halving_residual_tol))
        or float(halving_residual_tol) < 0.0
    ):
        raise ValueError("min_rms_ratio, max_rms_ratio, and halving_residual_tol must be valid finite bounds.")
    flat_old = _flatten_delta(old_weights)
    flat_new = _flatten_delta(new_weights)
    if flat_old.numel() == 0 or flat_old.numel() != flat_new.numel():
        raise ValueError("old_weights and new_weights must be non-empty and have identical element counts.")

    if not (bool(torch.isfinite(flat_old).all()) and bool(torch.isfinite(flat_new).all())):
        return WeightRmsGuardDecision(
            accepted=False,
            rms_old=float("nan"),
            rms_new=float("nan"),
            rms_ratio=float("nan"),
            halving_residual_rms=float("nan"),
            reason="REJECTED_NON_FINITE_WEIGHTS",
        )

    old_f64 = flat_old.to(torch.float64)
    new_f64 = flat_new.to(torch.float64)
    rms_old = float(torch.sqrt(torch.mean(old_f64 * old_f64)).item())
    rms_new = float(torch.sqrt(torch.mean(new_f64 * new_f64)).item())
    if rms_old <= 1e-12:
        raise ValueError("old_weights RMS norm must be strictly positive.")

    rms_ratio = rms_new / rms_old
    diff_half = new_f64 - 0.5 * old_f64
    halving_residual_rms = float(torch.sqrt(torch.mean(diff_half * diff_half)).item())

    if halving_residual_rms <= halving_residual_tol and rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(
            accepted=False,
            rms_old=rms_old,
            rms_new=rms_new,
            rms_ratio=rms_ratio,
            halving_residual_rms=halving_residual_rms,
            reason=(
                f"REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING "
                f"(rms_ratio={rms_ratio:.4f}, halving_residual_rms={halving_residual_rms:.6f})"
            ),
        )
    if rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(
            accepted=False,
            rms_old=rms_old,
            rms_new=rms_new,
            rms_ratio=rms_ratio,
            halving_residual_rms=halving_residual_rms,
            reason=f"REJECTED_RMS_COLLAPSE (rms_ratio={rms_ratio:.4f} < min={min_rms_ratio:.4f})",
        )
    if rms_ratio > max_rms_ratio:
        return WeightRmsGuardDecision(
            accepted=False,
            rms_old=rms_old,
            rms_new=rms_new,
            rms_ratio=rms_ratio,
            halving_residual_rms=halving_residual_rms,
            reason=f"REJECTED_RMS_EXPLOSION (rms_ratio={rms_ratio:.4f} > max={max_rms_ratio:.4f})",
        )

    return WeightRmsGuardDecision(
        accepted=True,
        rms_old=rms_old,
        rms_new=rms_new,
        rms_ratio=rms_ratio,
        halving_residual_rms=halving_residual_rms,
        reason=f"ACCEPTED_RMS_HEALTHY (rms_ratio={rms_ratio:.4f} in [{min_rms_ratio:.2f}, {max_rms_ratio:.2f}])",
    )


@dataclass(frozen=True)
class GpuPrecisionCohort:
    """Hardware-stratified precision & optimizer cohort assignment (`Vxtzq/CrowdGPT#44`, `#51`, HiveMind `E12`)."""

    vendor: str
    compute_capability: tuple[int, int]
    effective_dtype: str
    optimizer_family: str
    pre_ampere_fp32_enforced: bool
    cohort_key: str

    def __post_init__(self) -> None:
        for field_name, val in (
            ("vendor", self.vendor),
            ("effective_dtype", self.effective_dtype),
            ("optimizer_family", self.optimizer_family),
            ("cohort_key", self.cohort_key),
        ):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.compute_capability, tuple)
            or len(self.compute_capability) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.compute_capability)
        ):
            raise ValueError("compute_capability must be a 2-tuple of non-negative integers")
        if not isinstance(self.pre_ampere_fp32_enforced, bool):
            raise ValueError("pre_ampere_fp32_enforced must be a boolean")


def classify_gpu_precision_cohort(
    vendor: str,
    compute_capability: tuple[int, int] = (0, 0),
    *,
    requested_dtype: str = "auto",
    optimizer_family: str = "adamw_8bit",
) -> GpuPrecisionCohort:
    """Assign a volunteer GPU to a homogeneous numerical cohort and enforce `fp32` on pre-Ampere CUDA (`sm < 80`)."""
    if not isinstance(vendor, str) or not vendor.strip():
        raise ValueError("vendor must be a non-empty string.")
    if (
        not isinstance(compute_capability, tuple)
        or len(compute_capability) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in compute_capability)
    ):
        raise ValueError("compute_capability must be a 2-tuple of non-negative integers.")
    norm_vendor = vendor.strip().lower()
    major, minor = int(compute_capability[0]), int(compute_capability[1])
    req = requested_dtype.strip().lower()
    opt = optimizer_family.strip().lower() or "adamw"

    pre_ampere_enforced = False
    if norm_vendor in ("nvidia", "cuda"):
        norm_vendor = "nvidia"
        if major < 8:
            effective_dtype = "fp32"
            pre_ampere_enforced = req in ("auto", "bf16", "bfloat16")
            arch_tier = f"sm{major}{minor}_pre_ampere"
        else:
            effective_dtype = "fp32" if req in ("fp32", "float32") else "bf16"
            arch_tier = "sm80_plus_ampere"
    elif norm_vendor in ("amd", "rocm", "hip"):
        norm_vendor = "amd_rocm"
        effective_dtype = "bf16" if req in ("bf16", "bfloat16") else "fp32"
        arch_tier = "rocm"
    elif norm_vendor in ("apple", "mps", "metal"):
        norm_vendor = "apple_mps"
        effective_dtype = "fp32"
        arch_tier = "mps"
    else:
        norm_vendor = norm_vendor or "directml"
        effective_dtype = "fp32"
        arch_tier = "generic"

    cohort_key = f"{norm_vendor}:{arch_tier}:{effective_dtype}:{opt}"
    return GpuPrecisionCohort(
        vendor=norm_vendor,
        compute_capability=(major, minor),
        effective_dtype=effective_dtype,
        optimizer_family=opt,
        pre_ampere_fp32_enforced=pre_ampere_enforced,
        cohort_key=cohort_key,
    )


@dataclass(frozen=True)
class PoolDispatchDecision:
    """Routing decision for a CrowdSwarm cohort pool (`Vxtzq/CrowdGPT#44` forming-pool starvation fix)."""

    mode: str
    cohort_key: str
    active_members: int
    target_pool_size: int
    requires_multi_krum: bool
    requires_loss_ratchet: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("mode must be a non-empty string")
        if not isinstance(self.cohort_key, str) or not self.cohort_key.strip():
            raise ValueError("cohort_key must be a non-empty string")
        if isinstance(self.active_members, bool) or not isinstance(self.active_members, int) or self.active_members < 0:
            raise ValueError("active_members must be a non-negative integer")
        if isinstance(self.target_pool_size, bool) or not isinstance(self.target_pool_size, int) or self.target_pool_size < 1:
            raise ValueError("target_pool_size must be a positive integer")
        if not isinstance(self.requires_multi_krum, bool) or not isinstance(self.requires_loss_ratchet, bool):
            raise ValueError("requires_multi_krum and requires_loss_ratchet must be booleans")


def resolve_forming_pool_starvation(
    *,
    current_members: int,
    target_pool_size: int = 3,
    waited_seconds: float,
    forming_timeout_s: float = 180.0,
    cohort_key: str = "nvidia:sm75_pre_ampere:fp32:adamw",
) -> PoolDispatchDecision:
    """Prevent `1/3` or `2/3` forming pools from starving minority `fp32` / ROCm cohorts after timeout."""
    if isinstance(current_members, bool) or not isinstance(current_members, int):
        raise ValueError("current_members must be an integer.")
    if isinstance(target_pool_size, bool) or not isinstance(target_pool_size, int) or target_pool_size < 1:
        raise ValueError("target_pool_size must be a positive integer.")
    if (
        isinstance(waited_seconds, bool)
        or not isinstance(waited_seconds, (int, float))
        or not math.isfinite(float(waited_seconds))
        or float(waited_seconds) < 0.0
    ):
        raise ValueError("waited_seconds must be a finite non-negative number.")
    if (
        isinstance(forming_timeout_s, bool)
        or not isinstance(forming_timeout_s, (int, float))
        or not math.isfinite(float(forming_timeout_s))
        or float(forming_timeout_s) <= 0.0
    ):
        raise ValueError("forming_timeout_s must be a positive finite number.")
    if current_members <= 0:
        return PoolDispatchDecision(
            mode="IDLE_EMPTY_POOL",
            cohort_key=cohort_key,
            active_members=0,
            target_pool_size=target_pool_size,
            requires_multi_krum=False,
            requires_loss_ratchet=False,
        )
    if current_members >= target_pool_size:
        return PoolDispatchDecision(
            mode="COHORT_QUORUM_READY",
            cohort_key=cohort_key,
            active_members=current_members,
            target_pool_size=target_pool_size,
            requires_multi_krum=True,
            requires_loss_ratchet=True,
        )
    if waited_seconds >= forming_timeout_s:
        return PoolDispatchDecision(
            mode="SOLO_LOSS_RATCHET_FALLBACK",
            cohort_key=cohort_key,
            active_members=current_members,
            target_pool_size=target_pool_size,
            requires_multi_krum=True,
            requires_loss_ratchet=True,
        )
    return PoolDispatchDecision(
        mode="FORMING_WAIT",
        cohort_key=cohort_key,
        active_members=current_members,
        target_pool_size=target_pool_size,
        requires_multi_krum=True,
        requires_loss_ratchet=True,
    )


def evaluate_client_resilience_policy(
    *,
    batch_size: int,
    encountered_oom: bool = False,
    http_status: int | None = None,
    retry_attempt: int = 0,
    base_backoff_s: float = 2.0,
    max_backoff_s: float = 60.0,
) -> dict[str, Any]:
    """Client circuit-breaker for OOM at `batch_size=1` (`#52`) and HTTP 502/503 retry backoff (`#50`)."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1.")
    if isinstance(retry_attempt, bool) or not isinstance(retry_attempt, int) or retry_attempt < 0:
        raise ValueError("retry_attempt must be a non-negative integer.")
    if (
        isinstance(base_backoff_s, bool)
        or not isinstance(base_backoff_s, (int, float))
        or not math.isfinite(float(base_backoff_s))
        or float(base_backoff_s) <= 0.0
        or isinstance(max_backoff_s, bool)
        or not isinstance(max_backoff_s, (int, float))
        or not math.isfinite(float(max_backoff_s))
        or float(max_backoff_s) < float(base_backoff_s)
    ):
        raise ValueError("base_backoff_s and max_backoff_s must be finite positive numbers with max >= base.")
    if encountered_oom:
        if batch_size <= 1:
            return {
                "action": "ABORT_INSUFFICIENT_VRAM",
                "next_batch_size": 1,
                "redownload_checkpoint": False,
                "backoff_s": 0.0,
            }
        return {
            "action": "HALVE_BATCH_SIZE_AND_RETRY_LOCAL",
            "next_batch_size": max(1, batch_size // 2),
            "redownload_checkpoint": False,
            "backoff_s": 0.0,
        }

    if http_status in (502, 503, 504, 429):
        exp_delay = min(max_backoff_s, base_backoff_s * (2 ** max(0, retry_attempt)))
        jitter = 0.125 * exp_delay
        return {
            "action": "RETRY_FETCH_TASK_WITH_BACKOFF",
            "next_batch_size": batch_size,
            "redownload_checkpoint": False,
            "backoff_s": round(exp_delay + jitter, 3),
        }

    return {
        "action": "CONTINUE_NORMAL",
        "next_batch_size": batch_size,
        "redownload_checkpoint": False,
        "backoff_s": 0.0,
    }
