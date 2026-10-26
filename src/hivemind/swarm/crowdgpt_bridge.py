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
