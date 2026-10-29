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


def pack_crowdgpt_delta_stream(
    metadata: Mapping[str, Any],
    base_delta_bf16: torch.Tensor,
    engram_indices: np.ndarray | None = None,
    engram_values_uint16: np.ndarray | None = None,
    *,
    compresslevel: int = 6,
) -> bytes:
    """Pack a CrowdGPT delta in the exact wire format produced by `client.py`."""
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


@dataclass(frozen=True)
class MultiKrumAggregationResult:
    """Result of Multi-Krum filtering and token-weighted DiLoCo aggregation."""

    selected_indices: list[int]
    rejected_indices: list[int]
    krum_scores: list[float]
    aggregated_delta_bf16: torch.Tensor
    effective_tokens: int


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

    flat = [_flatten_delta(d) for d in deltas]
    if n == 1:
        tok = int(token_weights[0]) if token_weights else 1
        return MultiKrumAggregationResult(
            selected_indices=[0],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=flat[0].to(torch.bfloat16),
            effective_tokens=max(1, tok),
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
    selected = sorted(ranked[:m])
    rejected = sorted(ranked[m:])

    if token_weights is not None:
        if len(token_weights) != n:
            raise ValueError("token_weights length must match deltas length.")
        raw_w = [max(1.0, float(token_weights[i])) for i in selected]
    else:
        raw_w = [1.0 for _ in selected]

    total_w = sum(raw_w)
    agg = torch.zeros_like(flat[0], dtype=torch.float32)
    for idx, w in zip(selected, raw_w):
        agg.add_(flat[idx], alpha=w / total_w)

    eff_tokens = int(sum(int(token_weights[i]) for i in selected)) if token_weights else len(selected)
    return MultiKrumAggregationResult(
        selected_indices=selected,
        rejected_indices=rejected,
        krum_scores=scores,
        aggregated_delta_bf16=agg.to(torch.bfloat16),
        effective_tokens=eff_tokens,
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


class LossRatchetGate:
    """Monotonic held-out validation gate for outer DiLoCo / CrowdGPT merges."""

    def __init__(self, *, max_regression_tolerance: float = 0.0) -> None:
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


def check_weight_rms_collapse_guard(
    old_weights: torch.Tensor | Mapping[str, torch.Tensor],
    new_weights: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    min_rms_ratio: float = 0.80,
    max_rms_ratio: float = 1.25,
    halving_residual_tol: float = 1e-3,
) -> WeightRmsGuardDecision:
    """Reject aggregated checkpoints whose global RMS norm collapses, explodes, or halves (`0.5 * old`)."""
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


def classify_gpu_precision_cohort(
    vendor: str,
    compute_capability: tuple[int, int] = (0, 0),
    *,
    requested_dtype: str = "auto",
    optimizer_family: str = "adamw_8bit",
) -> GpuPrecisionCohort:
    """Assign a volunteer GPU to a homogeneous numerical cohort and enforce `fp32` on pre-Ampere CUDA (`sm < 80`)."""
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


def resolve_forming_pool_starvation(
    *,
    current_members: int,
    target_pool_size: int = 3,
    waited_seconds: float,
    forming_timeout_s: float = 180.0,
    cohort_key: str = "nvidia:sm75_pre_ampere:fp32:adamw",
) -> PoolDispatchDecision:
    """Prevent `1/3` or `2/3` forming pools from starving minority `fp32` / ROCm cohorts after timeout."""
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
