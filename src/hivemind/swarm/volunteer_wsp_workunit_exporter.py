# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Volunteer 8 GB Gamer-Whisper WSP Calibration Workunit & 4-Bit NF4 Edge Exporter.

Implements Season 35 ``HM-S35-06`` and ``HM-S35-07`` for decentralized open-weight
Weight-Space Programming (WSP) on consumer 8 GB GPUs (``std_8gb`` / RTX 2070 WDDM):

1. **HM-S35-06 — Volunteer 8 GB Gamer-Whisper WSP Calibration Workunit Policy & Null-Space
   Byzantine Aggregation** (``VolunteerWSPWorkunitSpec``, ``VolunteerWSPDeltaSubmission``,
   ``preempt_volunteer_workunit_for_game``, ``aggregate_volunteer_wsp_workunits_nullspace``):
   - Enforces the ``std_8gb <= 6656 MiB`` WDDM VRAM ceiling, ``< 1500 ms`` foreground game
     preemption checkpointing, strict refusal of corporate/managed hosts
     (``is_corporate_managed=True`` immediately rejected), and ``< 50.0 MB/h`` rank-bounded
     compressed WSP delta payload budgets.
   - Pre-screens untrusted volunteer submissions for policy breaches, ``NaN``/``Inf``
     corruption, rank exceedance, and L2 norm spikes, filters remaining Byzantine peers via
     exact CPU ``float64`` Multi-Krum, and projects the surviving consensus delta strictly
     into AlphaEdit general-prose null-space ``P_0`` so protected prose keys experience
     ``< 1e-6`` drift.

2. **HM-S35-07 — Canonical 16-Level NormalFloat4 (NF4) Edge Exporter with GPTQ-Style
   Diagonal-Hessian Intra-Block Error Feedback** (``export_nf4_compensated_edge_blocks``,
   ``dequantize_nf4_compensated_edge_blocks``, ``export_s5_model_to_nf4_edge``):
   - Folds S5 continuous edited weights into packed 4-bit NF4 blocks (``block_size=64``)
     using Hessian-weighted optimal block scale search + alternating weighted least squares
     and descending-Hessian intra-block error feedback (with exact Cholesky compensation when
     a full block Hessian is provided), strictly beating naive round-to-nearest NF4 error
     and retaining ``< 0.02`` nat held-out NLL gap vs FP32 within the 8 GB WDDM VRAM ceiling.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    compute_payload_sha256,
    multi_krum_nullspace_aggregate,
)

SCHEMA_VERSION: str = "hivemind.volunteer_wsp_workunit_exporter/1"
WORKUNIT_RECEIPT_OBJECT: str = "hivemind.volunteer_wsp_workunit_receipt"
PREEMPTION_RECEIPT_OBJECT: str = "hivemind.volunteer_game_preemption_receipt"
EDGE_EXPORT_RECEIPT_OBJECT: str = "hivemind.nf4_edge_export_receipt"
MODEL_EDGE_EXPORT_RECEIPT_OBJECT: str = "hivemind.nf4_model_edge_export_receipt"

STD_8GB_TIER: str = "std_8gb"
STD_8GB_WDDM_CEILING_MIB: int = 6656
MAX_GAME_PREEMPTION_LATENCY_MS: float = 1500.0
MAX_OUTBOUND_MB_PER_HOUR: float = 50.0
MAX_NF4_NLL_GAP_NATS: float = 0.02
DEFAULT_NF4_BLOCK_SIZE: int = 64
DEFAULT_MAX_NORM_SPIKE_RATIO: float = 4.0
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
SUPPORTED_COMPRESSION_BITS: tuple[int, ...] = (4, 8, 16)

CANONICAL_NF4_LEVELS: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

CORPORATE_HOST_MARKERS: frozenset[str] = frozenset(
    {
        "amadeus",
        "corp",
        "corporate",
        "enterprise",
        "intune",
        "jamf",
        "managed",
        "mdm",
        "work_laptop",
        "workplace",
    }
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_BLOB_KEYS: frozenset[str] = frozenset({"packed_nibbles", "block_scales"})


class VolunteerPolicyViolationError(ValueError):
    """Base exception raised when a volunteer host or workunit violates HiveMind policy."""

    def __init__(self, message: str, *, reason: str = "policy_violation") -> None:
        super().__init__(message)
        self.reason = reason


class CorporateHostRefusedError(VolunteerPolicyViolationError):
    """Raised immediately when a corporate or centrally managed machine attempts participation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="corporate_managed_host")


class VRAMCeilingExceededError(VolunteerPolicyViolationError):
    """Raised when a volunteer workunit exceeds the 8 GB WDDM 6656 MiB ceiling."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="vram_ceiling_exceeded")


class PreemptionBudgetExceededError(VolunteerPolicyViolationError):
    """Raised when foreground game preemption checkpointing fails the <1.5s budget."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="preemption_budget_exceeded")


class PayloadBudgetExceededError(VolunteerPolicyViolationError):
    """Raised when rank-bounded WSP delta upload exceeds the <50 MB/h outbound budget."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="payload_budget_exceeded")


# ---------------------------------------------------------------------------
# 1. Validation & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty string without control characters."""
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{name} must be a non-empty string without control characters")
    return value.strip()


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate a strictly positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate a finite real number (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite scalar inside ``(0, 1]`` or ``[0, 1]``."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_compression_bits(bits: Any) -> int:
    """Validate supported WSP payload compression bit-widths."""
    b_int = _validate_positive_int(bits, "compression_bits")
    if b_int not in SUPPORTED_COMPRESSION_BITS:
        raise ValueError(f"compression_bits must be in {SUPPORTED_COMPRESSION_BITS}, got {bits}")
    return b_int


def _validate_host_policy(
    *,
    hardware_tier: str,
    vram_budget_mib: float,
    preemption_latency_ms: float,
    is_corporate_managed: bool,
    host_markers: Sequence[str],
) -> tuple[str, float, float, tuple[str, ...]]:
    """Enforce personal-PC, 6656 MiB WDDM VRAM ceiling, and <1.5s game preemption policy."""
    _validate_bool(is_corporate_managed, "is_corporate_managed")
    if isinstance(host_markers, (str, bytes, bytearray)) or not isinstance(host_markers, Sequence):
        raise TypeError("host_markers must be a sequence of strings")
    cleaned_markers = tuple(str(m).strip().lower() for m in host_markers if str(m).strip())

    if is_corporate_managed:
        raise CorporateHostRefusedError(
            "Corporate or managed machines are strictly refused (100% personal home PCs only)."
        )
    matched = [
        marker
        for marker in cleaned_markers
        if any(corp in marker for corp in CORPORATE_HOST_MARKERS)
    ]
    if matched:
        raise CorporateHostRefusedError(
            f"Corporate/managed host markers detected ({matched}); personal home PCs only."
        )

    tier = _validate_nonempty_str(hardware_tier, "hardware_tier")
    if tier != STD_8GB_TIER:
        raise ValueError(f"Only {STD_8GB_TIER!r} tier is permitted for 8 GB volunteer WSP, got {tier!r}")

    vram = _validate_positive_finite(vram_budget_mib, "vram_budget_mib")
    if vram > float(STD_8GB_WDDM_CEILING_MIB):
        raise VRAMCeilingExceededError(
            f"vram_budget_mib ({vram:.1f} MiB) exceeds std_8gb WDDM ceiling ({STD_8GB_WDDM_CEILING_MIB} MiB)"
        )

    latency = _validate_nonneg_finite(preemption_latency_ms, "preemption_latency_ms")
    if latency >= MAX_GAME_PREEMPTION_LATENCY_MS:
        raise PreemptionBudgetExceededError(
            f"preemption_latency_ms ({latency:.1f} ms) must be < {MAX_GAME_PREEMPTION_LATENCY_MS:.1f} ms"
        )

    return tier, vram, latency, cleaned_markers


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute a deterministic bitwise SHA-256 hex digest over a single CPU tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    t_cpu = tensor.detach().to(device="cpu").contiguous()
    if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
        raise ValueError("Cannot hash tensor containing non-finite values")
    digest = hashlib.sha256()
    digest.update(f"{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
    digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compute_tensor_map_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic bitwise SHA-256 over a sorted mapping of finite tensors."""
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("tensors must be a non-empty mapping")
    digest = hashlib.sha256()
    for key in sorted(tensors.keys()):
        k_str = _validate_nonempty_str(key, "tensor_key")
        val = tensors[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"tensors[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_cpu = val.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"tensors[{key!r}] contains non-finite values")
        digest.update(f"{k_str}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _seal_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical ``receipt_sha256`` over all JSON-serializable fields (excluding tensor blobs)."""
    core = {
        k: v
        for k, v in payload.items()
        if k != "receipt_sha256" and k not in _TENSOR_BLOB_KEYS
    }
    sealed = dict(payload)
    sealed["receipt_sha256"] = compute_payload_sha256(core)
    return sealed


def verify_workunit_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical ``receipt_sha256`` provenance on a workunit or preemption receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def verify_edge_export_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical ``receipt_sha256`` and tensor-blob digests on an NF4 edge-export receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False

    for blob_key in _TENSOR_BLOB_KEYS:
        if blob_key in receipt:
            blob = receipt[blob_key]
            digest_key = f"{blob_key}_sha256"
            if not isinstance(blob, torch.Tensor):
                return False
            try:
                actual_blob_sha = compute_tensor_sha256(blob)
            except (TypeError, ValueError):
                return False
            if receipt.get(digest_key) != actual_blob_sha:
                return False

    core = {
        k: v
        for k, v in receipt.items()
        if k != "receipt_sha256" and k not in _TENSOR_BLOB_KEYS
    }
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. HM-S35-06 — Volunteer 8 GB Gamer-Whisper Spec, Payload & Preemption
# ---------------------------------------------------------------------------


def estimate_rank_bounded_wsp_payload_bytes(
    *,
    num_layers: int,
    hidden_size: int,
    rank: int,
    num_factors_per_layer: int = 3,
    compression_bits: int = 4,
    sparse_keep_ratio: float = 1.0,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    header_bytes: int = 1024,
) -> int:
    """Estimate compressed byte payload for one rank-``r`` volunteer WSP delta upload."""
    layers_i = _validate_positive_int(num_layers, "num_layers")
    dim_i = _validate_positive_int(hidden_size, "hidden_size")
    rank_i = _validate_positive_int(rank, "rank")
    factors_i = _validate_positive_int(num_factors_per_layer, "num_factors_per_layer")
    bits_i = _validate_compression_bits(compression_bits)
    keep_f = _validate_unit_interval(sparse_keep_ratio, "sparse_keep_ratio", allow_zero=False)
    block_i = _validate_positive_int(block_size, "block_size")
    hdr_i = _validate_nonneg_int(header_bytes, "header_bytes")

    total_elements = layers_i * factors_i * rank_i * dim_i
    return estimate_elements_payload_bytes(
        total_elements,
        compression_bits=bits_i,
        sparse_keep_ratio=keep_f,
        block_size=block_i,
        header_bytes=hdr_i,
    )


def estimate_elements_payload_bytes(
    total_elements: int,
    *,
    compression_bits: int = 4,
    sparse_keep_ratio: float = 1.0,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    header_bytes: int = 1024,
) -> int:
    """Estimate compressed byte size for ``total_elements`` quantized WSP coefficients."""
    elems_i = _validate_positive_int(total_elements, "total_elements")
    bits_i = _validate_compression_bits(compression_bits)
    keep_f = _validate_unit_interval(sparse_keep_ratio, "sparse_keep_ratio", allow_zero=False)
    block_i = _validate_positive_int(block_size, "block_size")
    hdr_i = _validate_nonneg_int(header_bytes, "header_bytes")

    active_elements = max(1, math.ceil(elems_i * keep_f))
    coeff_bytes = math.ceil((active_elements * bits_i) / 8.0)
    num_blocks = math.ceil(active_elements / float(block_i))
    scale_bytes = num_blocks * 2  # fp16 block scale per group
    bitmap_bytes = math.ceil(elems_i / 8.0) if keep_f < 1.0 else 0
    return int(hdr_i + coeff_bytes + scale_bytes + bitmap_bytes)


def estimate_hourly_payload_mb(payload_bytes: int, workunits_per_hour: float) -> float:
    """Compute hourly outbound upload rate in decimal MB/h (``1e6`` bytes/MB)."""
    p_bytes = _validate_nonneg_int(payload_bytes, "payload_bytes")
    rate = _validate_positive_finite(workunits_per_hour, "workunits_per_hour")
    return (float(p_bytes) * rate) / 1_000_000.0


@torch.no_grad()
def compress_rank_bounded_wsp_delta(
    delta: torch.Tensor,
    *,
    max_rank: int,
    compression_bits: int = 8,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Truncate a 2D delta matrix to ``rank <= max_rank`` and quantize to ``compression_bits``."""
    if not isinstance(delta, torch.Tensor) or delta.ndim != 2 or delta.numel() == 0:
        raise ValueError("delta must be a non-empty 2D torch.Tensor")
    if not delta.is_floating_point():
        raise TypeError(f"delta must have floating-point dtype, got {delta.dtype}")
    d64 = delta.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(d64).all().item()):
        raise ValueError("delta must contain only finite values")

    r_cap = _validate_positive_int(max_rank, "max_rank")
    bits_i = _validate_compression_bits(compression_bits)
    block_i = _validate_positive_int(block_size, "block_size")

    rows, cols = int(d64.shape[0]), int(d64.shape[1])
    eff_rank = min(r_cap, rows, cols)
    u_mat, s_vals, vh_mat = torch.linalg.svd(d64, full_matrices=False)
    u_r = u_mat[:, :eff_rank]
    s_r = s_vals[:eff_rank]
    vh_r = vh_mat[:eff_rank, :]

    factor_b = u_r * s_r.sqrt().unsqueeze(0)
    factor_a = s_r.sqrt().unsqueeze(1) * vh_r

    if bits_i == 16:
        fb_q = factor_b.to(torch.float16).to(torch.float64)
        fa_q = factor_a.to(torch.float16).to(torch.float64)
    else:
        q_max = float((1 << (bits_i - 1)) - 1)
        fb_q = _symmetric_block_quant_dequant(factor_b, q_max=q_max, block_size=block_i)
        fa_q = _symmetric_block_quant_dequant(factor_a, q_max=q_max, block_size=block_i)

    recon = (fb_q @ fa_q).to(device=delta.device, dtype=delta.dtype).contiguous()
    factor_elements = int(factor_b.numel() + factor_a.numel())
    payload_bytes = estimate_elements_payload_bytes(
        factor_elements,
        compression_bits=bits_i,
        block_size=block_i,
        header_bytes=256,
    )
    recon_err = float(torch.linalg.matrix_norm(d64 - (fb_q @ fa_q), ord="fro").item())
    return recon, {
        "effective_rank": eff_rank,
        "compression_bits": bits_i,
        "factor_elements": factor_elements,
        "payload_bytes": payload_bytes,
        "reconstruction_fro_error": recon_err,
    }


def _symmetric_block_quant_dequant(mat: torch.Tensor, *, q_max: float, block_size: int) -> torch.Tensor:
    """Group-quantize a 2D ``float64`` matrix along columns and dequantize."""
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    pad_cols = (block_size - (cols % block_size)) % block_size
    padded = F.pad(mat, (0, pad_cols)) if pad_cols > 0 else mat
    grouped = padded.reshape(rows, -1, block_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True) / q_max
    safe_scales = torch.where(scales > 0.0, scales, torch.ones_like(scales))
    q_int = torch.round(grouped / safe_scales).clamp(-q_max, q_max)
    deq = torch.where(scales > 0.0, q_int * scales, torch.zeros_like(grouped))
    return deq.reshape(rows, -1)[:, :cols].contiguous()


@dataclass(frozen=True)
class VolunteerWSPWorkunitSpec:
    """Specification and admission contract for an 8 GB Gamer-Whisper WSP calibration workunit."""

    workunit_id: str
    peer_id: str
    hardware_tier: str = STD_8GB_TIER
    vram_budget_mib: float = 6144.0
    preemption_latency_ms: float = 420.0
    is_corporate_managed: bool = False
    game_active: bool = False
    host_markers: tuple[str, ...] = ()
    max_rank: int = 8
    num_layers: int = 24
    hidden_size: int = 1536
    num_factors_per_layer: int = 3
    compression_bits: int = 4
    sparse_keep_ratio: float = 1.0
    block_size: int = DEFAULT_NF4_BLOCK_SIZE
    workunits_per_hour: float = 4.0
    max_mb_per_hour: float = MAX_OUTBOUND_MB_PER_HOUR

    def __post_init__(self) -> None:
        wu_id = _validate_nonempty_str(self.workunit_id, "workunit_id")
        p_id = _validate_nonempty_str(self.peer_id, "peer_id")
        tier, vram, latency, cleaned_markers = _validate_host_policy(
            hardware_tier=self.hardware_tier,
            vram_budget_mib=self.vram_budget_mib,
            preemption_latency_ms=self.preemption_latency_ms,
            is_corporate_managed=self.is_corporate_managed,
            host_markers=self.host_markers,
        )
        _validate_bool(self.game_active, "game_active")
        rank_i = _validate_positive_int(self.max_rank, "max_rank")
        layers_i = _validate_positive_int(self.num_layers, "num_layers")
        dim_i = _validate_positive_int(self.hidden_size, "hidden_size")
        factors_i = _validate_positive_int(self.num_factors_per_layer, "num_factors_per_layer")
        bits_i = _validate_compression_bits(self.compression_bits)
        keep_f = _validate_unit_interval(self.sparse_keep_ratio, "sparse_keep_ratio", allow_zero=False)
        block_i = _validate_positive_int(self.block_size, "block_size")
        rate_f = _validate_positive_finite(self.workunits_per_hour, "workunits_per_hour")
        cap_mb_h = _validate_positive_finite(self.max_mb_per_hour, "max_mb_per_hour")
        if cap_mb_h > MAX_OUTBOUND_MB_PER_HOUR:
            raise PayloadBudgetExceededError(
                f"max_mb_per_hour ({cap_mb_h:.2f}) cannot exceed {MAX_OUTBOUND_MB_PER_HOUR:.1f} MB/h"
            )

        object.__setattr__(self, "workunit_id", wu_id)
        object.__setattr__(self, "peer_id", p_id)
        object.__setattr__(self, "hardware_tier", tier)
        object.__setattr__(self, "vram_budget_mib", vram)
        object.__setattr__(self, "preemption_latency_ms", latency)
        object.__setattr__(self, "host_markers", cleaned_markers)
        object.__setattr__(self, "max_rank", rank_i)
        object.__setattr__(self, "num_layers", layers_i)
        object.__setattr__(self, "hidden_size", dim_i)
        object.__setattr__(self, "num_factors_per_layer", factors_i)
        object.__setattr__(self, "compression_bits", bits_i)
        object.__setattr__(self, "sparse_keep_ratio", keep_f)
        object.__setattr__(self, "block_size", block_i)
        object.__setattr__(self, "workunits_per_hour", rate_f)
        object.__setattr__(self, "max_mb_per_hour", cap_mb_h)

        est_mb_h = self.estimated_mb_per_hour
        if est_mb_h >= cap_mb_h:
            raise PayloadBudgetExceededError(
                f"Estimated outbound WSP payload ({est_mb_h:.2f} MB/h) must be < {cap_mb_h:.2f} MB/h"
            )

    @property
    def estimated_payload_bytes(self) -> int:
        """Estimated compressed byte size of one rank-bounded WSP delta upload."""
        return estimate_rank_bounded_wsp_payload_bytes(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            rank=self.max_rank,
            num_factors_per_layer=self.num_factors_per_layer,
            compression_bits=self.compression_bits,
            sparse_keep_ratio=self.sparse_keep_ratio,
            block_size=self.block_size,
        )

    @property
    def estimated_mb_per_hour(self) -> float:
        """Estimated outbound telemetry/delta bandwidth in decimal MB/h."""
        return estimate_hourly_payload_mb(self.estimated_payload_bytes, self.workunits_per_hour)

    def to_dict(self) -> dict[str, Any]:
        """Serialize workunit specification to a canonical JSON-safe dictionary."""
        data = asdict(self)
        data["host_markers"] = list(self.host_markers)
        data["estimated_payload_bytes"] = self.estimated_payload_bytes
        data["estimated_mb_per_hour"] = self.estimated_mb_per_hour
        return data


def evaluate_volunteer_workunit_admission(spec: VolunteerWSPWorkunitSpec) -> dict[str, Any]:
    """Evaluate runtime dispatch eligibility for a validated ``VolunteerWSPWorkunitSpec``."""
    if not isinstance(spec, VolunteerWSPWorkunitSpec):
        raise TypeError(f"spec must be VolunteerWSPWorkunitSpec, got {type(spec).__name__}")

    if spec.game_active:
        payload = {
            "object": WORKUNIT_RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "workunit_id": spec.workunit_id,
            "peer_id": spec.peer_id,
            "hardware_tier": spec.hardware_tier,
            "admitted": False,
            "paused": True,
            "game_active": True,
            "reason": "game_active_preemption_pause",
            "vram_budget_mib": 0.0,
            "preemption_latency_ms": spec.preemption_latency_ms,
            "estimated_payload_bytes": spec.estimated_payload_bytes,
            "estimated_mb_per_hour": spec.estimated_mb_per_hour,
        }
        return _seal_receipt(payload)

    payload = {
        "object": WORKUNIT_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "workunit_id": spec.workunit_id,
        "peer_id": spec.peer_id,
        "hardware_tier": spec.hardware_tier,
        "admitted": True,
        "paused": False,
        "game_active": False,
        "reason": "admitted_std_8gb_idle",
        "vram_budget_mib": spec.vram_budget_mib,
        "preemption_latency_ms": spec.preemption_latency_ms,
        "estimated_payload_bytes": spec.estimated_payload_bytes,
        "estimated_mb_per_hour": spec.estimated_mb_per_hour,
    }
    return _seal_receipt(payload)


@torch.no_grad()
def preempt_volunteer_workunit_for_game(
    spec: VolunteerWSPWorkunitSpec,
    *,
    trigger_process: str = "steam_game.exe",
    inflight_deltas: Mapping[str, torch.Tensor] | None = None,
    pcie_dma_bandwidth_gbps: float = 12.8,
    kernel_drain_ms: float = 45.0,
) -> dict[str, Any]:
    """Pause a volunteer WSP workunit on foreground game launch and checkpoint to Host RAM < 1.5s."""
    if not isinstance(spec, VolunteerWSPWorkunitSpec):
        raise TypeError(f"spec must be VolunteerWSPWorkunitSpec, got {type(spec).__name__}")
    proc = _validate_nonempty_str(trigger_process, "trigger_process")
    bw = _validate_positive_finite(pcie_dma_bandwidth_gbps, "pcie_dma_bandwidth_gbps")
    drain = _validate_nonneg_finite(kernel_drain_ms, "kernel_drain_ms")

    dma_flush_ms = (spec.vram_budget_mib / 1024.0) / bw * 1000.0
    total_latency_ms = round(max(drain + dma_flush_ms, spec.preemption_latency_ms), 3)
    if total_latency_ms >= MAX_GAME_PREEMPTION_LATENCY_MS:
        raise PreemptionBudgetExceededError(
            f"Preemption eviction latency ({total_latency_ms:.1f} ms) exceeds <{MAX_GAME_PREEMPTION_LATENCY_MS:.1f} ms budget"
        )

    checkpoint_tensors: dict[str, torch.Tensor] = {}
    if inflight_deltas is not None:
        if not isinstance(inflight_deltas, Mapping) or not inflight_deltas:
            raise ValueError("inflight_deltas must be None or a non-empty mapping of tensors")
        for key, tensor in sorted(inflight_deltas.items()):
            k_str = _validate_nonempty_str(key, "inflight_key")
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise ValueError(f"inflight_deltas[{key!r}] must be a non-empty torch.Tensor")
            checkpoint_tensors[k_str] = tensor.detach().to(device="cpu").contiguous().clone()
        checkpoint_sha256 = compute_tensor_map_sha256(checkpoint_tensors)
    else:
        checkpoint_sha256 = compute_payload_sha256({"workunit_id": spec.workunit_id, "empty_checkpoint": True})

    payload = {
        "object": PREEMPTION_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "workunit_id": spec.workunit_id,
        "peer_id": spec.peer_id,
        "trigger_process": proc,
        "admitted": False,
        "paused": True,
        "game_active": True,
        "vram_before_mib": spec.vram_budget_mib,
        "vram_budget_mib": 0.0,
        "eviction_latency_ms": total_latency_ms,
        "preemption_satisfied": True,
        "checkpoint_tensor_count": len(checkpoint_tensors),
        "checkpoint_sha256": checkpoint_sha256,
    }
    return _seal_receipt(payload)


@dataclass(frozen=True)
class VolunteerWSPDeltaSubmission:
    """Validated volunteer WSP delta submission respecting host, VRAM, preemption, and <50 MB/h rules."""

    peer_id: str
    workunit_id: str
    delta_tensors: Mapping[str, torch.Tensor] | torch.Tensor
    rank: int = 8
    hardware_tier: str = STD_8GB_TIER
    vram_budget_mib: float = 6144.0
    preemption_latency_ms: float = 420.0
    is_corporate_managed: bool = False
    game_active: bool = False
    host_markers: tuple[str, ...] = ()
    compression_bits: int = 4
    sparse_keep_ratio: float = 1.0
    block_size: int = DEFAULT_NF4_BLOCK_SIZE
    workunits_per_hour: float = 4.0
    reported_payload_bytes: int | None = None
    peer_weight: float = 1.0

    def __post_init__(self) -> None:
        p_id = _validate_nonempty_str(self.peer_id, "peer_id")
        wu_id = _validate_nonempty_str(self.workunit_id, "workunit_id")
        tier, vram, latency, cleaned_markers = _validate_host_policy(
            hardware_tier=self.hardware_tier,
            vram_budget_mib=self.vram_budget_mib,
            preemption_latency_ms=self.preemption_latency_ms,
            is_corporate_managed=self.is_corporate_managed,
            host_markers=self.host_markers,
        )
        _validate_bool(self.game_active, "game_active")
        rank_i = _validate_positive_int(self.rank, "rank")
        bits_i = _validate_compression_bits(self.compression_bits)
        keep_f = _validate_unit_interval(self.sparse_keep_ratio, "sparse_keep_ratio", allow_zero=False)
        block_i = _validate_positive_int(self.block_size, "block_size")
        rate_f = _validate_positive_finite(self.workunits_per_hour, "workunits_per_hour")
        weight_f = _validate_positive_finite(self.peer_weight, "peer_weight")

        if isinstance(self.delta_tensors, torch.Tensor):
            raw_map: dict[str, torch.Tensor] = {"delta": self.delta_tensors}
        elif isinstance(self.delta_tensors, Mapping) and self.delta_tensors:
            raw_map = dict(self.delta_tensors)
        else:
            raise TypeError("delta_tensors must be a non-empty Mapping[str, torch.Tensor] or torch.Tensor")

        clean_map: dict[str, torch.Tensor] = {}
        total_elements = 0
        for key in sorted(raw_map.keys()):
            k_str = _validate_nonempty_str(key, "delta_key")
            val = raw_map[key]
            if not isinstance(val, torch.Tensor) or val.numel() == 0:
                raise ValueError(f"delta_tensors[{key!r}] must be a non-empty torch.Tensor")
            if not val.is_floating_point():
                raise TypeError(f"delta_tensors[{key!r}] must be floating-point, got {val.dtype}")
            clean_map[k_str] = val.detach().clone()
            total_elements += int(val.numel())

        est_bytes = estimate_elements_payload_bytes(
            total_elements,
            compression_bits=bits_i,
            sparse_keep_ratio=keep_f,
            block_size=block_i,
            header_bytes=256,
        )
        if self.reported_payload_bytes is not None:
            rep_i = _validate_positive_int(self.reported_payload_bytes, "reported_payload_bytes")
            eff_bytes = max(est_bytes, rep_i)
        else:
            eff_bytes = est_bytes

        mb_h = estimate_hourly_payload_mb(eff_bytes, rate_f)
        if mb_h >= MAX_OUTBOUND_MB_PER_HOUR:
            raise PayloadBudgetExceededError(
                f"Submission {p_id!r} outbound payload ({mb_h:.2f} MB/h) exceeds <{MAX_OUTBOUND_MB_PER_HOUR:.1f} MB/h budget"
            )

        object.__setattr__(self, "peer_id", p_id)
        object.__setattr__(self, "workunit_id", wu_id)
        object.__setattr__(self, "delta_tensors", clean_map)
        object.__setattr__(self, "rank", rank_i)
        object.__setattr__(self, "hardware_tier", tier)
        object.__setattr__(self, "vram_budget_mib", vram)
        object.__setattr__(self, "preemption_latency_ms", latency)
        object.__setattr__(self, "host_markers", cleaned_markers)
        object.__setattr__(self, "compression_bits", bits_i)
        object.__setattr__(self, "sparse_keep_ratio", keep_f)
        object.__setattr__(self, "block_size", block_i)
        object.__setattr__(self, "workunits_per_hour", rate_f)
        object.__setattr__(self, "reported_payload_bytes", eff_bytes)
        object.__setattr__(self, "peer_weight", weight_f)

    @property
    def estimated_payload_bytes(self) -> int:
        """Compressed payload byte size for this delta submission."""
        assert self.reported_payload_bytes is not None
        return int(self.reported_payload_bytes)

    @property
    def estimated_mb_per_hour(self) -> float:
        """Hourly outbound bandwidth estimate in decimal MB/h."""
        return estimate_hourly_payload_mb(self.estimated_payload_bytes, self.workunits_per_hour)


# ---------------------------------------------------------------------------
# 3. HM-S35-06 — Byzantine Multi-Krum & AlphaEdit Null-Space Workunit Aggregator
# ---------------------------------------------------------------------------


def _extract_submission_record(item: VolunteerWSPDeltaSubmission | Mapping[str, Any], index: int) -> dict[str, Any]:
    """Normalize a ``VolunteerWSPDeltaSubmission`` or mapping submission into a validated record."""
    if isinstance(item, VolunteerWSPDeltaSubmission):
        return {
            "peer_id": item.peer_id,
            "workunit_id": item.workunit_id,
            "delta_tensors": dict(item.delta_tensors),
            "rank": item.rank,
            "vram_budget_mib": item.vram_budget_mib,
            "preemption_latency_ms": item.preemption_latency_ms,
            "is_corporate_managed": item.is_corporate_managed,
            "host_markers": item.host_markers,
            "payload_bytes": item.estimated_payload_bytes,
            "mb_per_hour": item.estimated_mb_per_hour,
            "peer_weight": item.peer_weight,
            "policy_reasons": [],
        }

    if not isinstance(item, Mapping):
        raise TypeError(
            f"submissions[{index}] must be VolunteerWSPDeltaSubmission or Mapping, got {type(item).__name__}"
        )

    peer_id = _validate_nonempty_str(item.get("peer_id", f"peer_{index}"), f"submissions[{index}].peer_id")
    workunit_id = _validate_nonempty_str(
        item.get("workunit_id", f"wu_{index}"), f"submissions[{index}].workunit_id"
    )
    raw_deltas = item.get("delta_tensors")
    if isinstance(raw_deltas, torch.Tensor):
        delta_map = {"delta": raw_deltas.detach().clone()}
    elif isinstance(raw_deltas, Mapping) and raw_deltas:
        delta_map = {
            _validate_nonempty_str(k, "delta_key"): v.detach().clone()
            for k, v in raw_deltas.items()
            if isinstance(v, torch.Tensor) and v.numel() > 0 and v.is_floating_point()
        }
        if len(delta_map) != len(raw_deltas):
            raise ValueError(f"submissions[{index}] delta_tensors must contain non-empty floating-point tensors")
    else:
        raise ValueError(f"submissions[{index}].delta_tensors must be a non-empty tensor or tensor mapping")

    policy_reasons: list[str] = []
    try:
        _, vram, latency, _ = _validate_host_policy(
            hardware_tier=str(item.get("hardware_tier", STD_8GB_TIER)),
            vram_budget_mib=float(item.get("vram_budget_mib", 6144.0)),
            preemption_latency_ms=float(item.get("preemption_latency_ms", 420.0)),
            is_corporate_managed=bool(item.get("is_corporate_managed", False)),
            host_markers=tuple(item.get("host_markers", ())),
        )
    except VolunteerPolicyViolationError as exc:
        policy_reasons.append(exc.reason)
        vram = float(item.get("vram_budget_mib", 0.0))
        latency = float(item.get("preemption_latency_ms", 0.0))

    rank_i = _validate_positive_int(int(item.get("rank", 8)), "rank")
    total_elements = sum(int(t.numel()) for t in delta_map.values())
    payload_bytes = int(
        item.get(
            "payload_bytes",
            estimate_elements_payload_bytes(total_elements, compression_bits=4, header_bytes=256),
        )
    )
    rate_f = _validate_positive_finite(float(item.get("workunits_per_hour", 4.0)), "workunits_per_hour")
    mb_h = estimate_hourly_payload_mb(payload_bytes, rate_f)
    if mb_h >= MAX_OUTBOUND_MB_PER_HOUR:
        policy_reasons.append("payload_budget_exceeded")

    return {
        "peer_id": peer_id,
        "workunit_id": workunit_id,
        "delta_tensors": delta_map,
        "rank": rank_i,
        "vram_budget_mib": vram,
        "preemption_latency_ms": latency,
        "is_corporate_managed": bool(item.get("is_corporate_managed", False)),
        "host_markers": tuple(item.get("host_markers", ())),
        "payload_bytes": payload_bytes,
        "mb_per_hour": mb_h,
        "peer_weight": _validate_positive_finite(float(item.get("peer_weight", 1.0)), "peer_weight"),
        "policy_reasons": policy_reasons,
    }


def _numerical_matrix_rank(tensor: torch.Tensor, *, tol: float = 1e-7) -> int:
    """Compute effective numerical rank of a 2D finite matrix."""
    if tensor.ndim < 2:
        return 1
    mat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    sv = torch.linalg.svdvals(mat)
    max_sv = float(sv[0].item()) if sv.numel() > 0 else 0.0
    if max_sv <= 1e-12:
        return 0
    return int((sv > tol * max_sv).sum().item())


def _compute_prose_key_drift(
    delta_map: Mapping[str, torch.Tensor],
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> float:
    """Measure max absolute projection norm ``||K_0 @ Delta^T||_inf`` across prose keys."""
    if prose_keys is None:
        return 0.0
    max_drift = 0.0
    for key, val in delta_map.items():
        if isinstance(prose_keys, torch.Tensor):
            k_mat = prose_keys.detach().to(device="cpu", dtype=torch.float64)
        elif key in prose_keys:
            k_mat = prose_keys[key].detach().to(device="cpu", dtype=torch.float64)
        else:
            continue
        t64 = val.detach().to(device="cpu", dtype=torch.float64)
        if k_mat.ndim != 2 or t64.ndim < 1 or int(k_mat.shape[1]) != int(t64.shape[-1]):
            continue
        prod = k_mat @ t64.T if t64.ndim == 2 else k_mat @ t64
        drift = float(prod.abs().max().item()) if prod.numel() > 0 else 0.0
        max_drift = max(max_drift, drift)
    return max_drift


@torch.no_grad()
def aggregate_volunteer_wsp_workunits_nullspace(
    submissions: Sequence[VolunteerWSPDeltaSubmission | Mapping[str, Any]],
    p_null: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    max_byzantine_peers: int = 1,
    max_norm_spike_ratio: float = DEFAULT_MAX_NORM_SPIKE_RATIO,
    max_l2_norm: float | None = None,
    max_allowed_rank: int | None = None,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pre-screen volunteer WSP workunits, filter Byzantine peers via Multi-Krum, and project into ``P_0``."""
    if isinstance(submissions, (str, bytes, bytearray)) or not isinstance(submissions, Sequence) or not submissions:
        raise TypeError("submissions must be a non-empty sequence of VolunteerWSPDeltaSubmission or mappings")
    max_byz = _validate_nonneg_int(max_byzantine_peers, "max_byzantine_peers")
    spike_cap = _validate_positive_finite(max_norm_spike_ratio, "max_norm_spike_ratio")
    norm_cap = _validate_positive_finite(max_l2_norm, "max_l2_norm") if max_l2_norm is not None else None
    rank_cap = _validate_positive_int(max_allowed_rank, "max_allowed_rank") if max_allowed_rank is not None else None

    n_total = len(submissions)
    if n_total < 2 * max_byz + 3:
        raise ValueError(
            f"Multi-Krum quorum requires n >= 2*f + 3 (n >= {2 * max_byz + 3}); got n={n_total}, max_byzantine_peers={max_byz}"
        )

    records = [_extract_submission_record(item, idx) for idx, item in enumerate(submissions)]
    seen_peers: set[str] = set()
    for rec in records:
        pid = rec["peer_id"]
        if pid in seen_peers:
            raise ValueError(f"Duplicate peer_id {pid!r} in submissions")
        seen_peers.add(pid)

    ref_keys = tuple(sorted(records[0]["delta_tensors"].keys()))
    ref_shapes = {k: tuple(records[0]["delta_tensors"][k].shape) for k in ref_keys}

    # Pre-compute L2 norms and schema/finite checks
    prelim: list[dict[str, Any]] = []
    for rec in records:
        reasons = list(rec["policy_reasons"])
        d_map = rec["delta_tensors"]
        if tuple(sorted(d_map.keys())) != ref_keys or any(
            tuple(d_map[k].shape) != ref_shapes[k] for k in ref_keys
        ):
            reasons.append("schema_or_shape_mismatch")
            prelim.append({"rec": rec, "finite": False, "l2_norm": None, "reasons": reasons})
            continue

        all_finite = all(bool(torch.isfinite(d_map[k].detach()).all().item()) for k in ref_keys)
        if not all_finite:
            reasons.append("non_finite_values")
            prelim.append({"rec": rec, "finite": False, "l2_norm": None, "reasons": reasons})
            continue

        l2_sq = sum(float(d_map[k].detach().to(torch.float64).pow(2).sum().item()) for k in ref_keys)
        l2_norm = math.sqrt(l2_sq)
        if norm_cap is not None and l2_norm > norm_cap:
            reasons.append("l2_norm_exceeded")
        if rank_cap is not None:
            obs_rank = max(_numerical_matrix_rank(d_map[k]) for k in ref_keys)
            if obs_rank > rank_cap:
                reasons.append("rank_bound_exceeded")

        prelim.append({"rec": rec, "finite": True, "l2_norm": l2_norm, "reasons": reasons})

    finite_norms = [
        float(entry["l2_norm"])
        for entry in prelim
        if entry["finite"] and not entry["reasons"] and float(entry["l2_norm"]) > 1e-12
    ]
    if len(finite_norms) < 3:
        finite_norms = [
            float(entry["l2_norm"])
            for entry in prelim
            if entry["finite"] and float(entry["l2_norm"]) > 1e-12
        ]
    median_norm = float(np.median(finite_norms)) if len(finite_norms) >= 3 else 0.0

    surviving_by_peer: dict[str, dict[str, torch.Tensor]] = {}
    surviving_weights: list[float] = []
    surviving_records: list[dict[str, Any]] = []
    rejected_reasons: dict[str, list[str]] = {}

    for entry in prelim:
        rec = entry["rec"]
        pid = rec["peer_id"]
        reasons = list(entry["reasons"])
        if entry["finite"] and median_norm > 1e-12:
            if float(entry["l2_norm"]) > spike_cap * median_norm:
                reasons.append("norm_spike_exceeded")
        if reasons:
            rejected_reasons[pid] = reasons
        else:
            surviving_by_peer[pid] = rec["delta_tensors"]
            surviving_weights.append(float(rec["peer_weight"]))
            surviving_records.append(rec)

    pre_rejected_count = len(rejected_reasons)
    if pre_rejected_count > max_byz:
        raise ValueError(
            f"Pre-screening rejected {pre_rejected_count} peers, exceeding max_byzantine_peers={max_byz}"
        )

    remaining_f = max_byz - pre_rejected_count
    n_surv = len(surviving_by_peer)

    consensus_map, krum_meta = multi_krum_nullspace_aggregate(
        surviving_by_peer,
        num_byzantine_f=remaining_f,
        select_m=n_surv - remaining_f,
        p_null=p_null,
        peer_weights=surviving_weights,
        enforce_byzantine_bound=True,
        protect_layernorm=protect_layernorm,
        protect_general_expert=protect_general_expert,
        general_expert_index=general_expert_index,
    )
    assert isinstance(consensus_map, dict)

    for rej_pid in krum_meta["rejected_peer_ids"]:
        rejected_reasons[rej_pid] = ["multi_krum_distance_outlier"]

    accepted_ids = list(krum_meta["selected_peer_ids"])
    accepted_set = set(accepted_ids)
    accepted_recs = [r for r in surviving_records if r["peer_id"] in accepted_set]

    prose_drift = _compute_prose_key_drift(consensus_map, prose_keys)
    proj_l2 = float(krum_meta["projected_l2_norm"])
    anse_energy = ANSE_MAXIMUM_PAIN if prose_drift > 1e-6 else round(-proj_l2 + prose_drift, 9)

    receipt_payload: dict[str, Any] = {
        "object": WORKUNIT_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "num_submissions": n_total,
        "num_accepted": len(accepted_ids),
        "num_rejected": len(rejected_reasons),
        "max_byzantine_peers": max_byz,
        "accepted_peer_ids": accepted_ids,
        "rejected_peer_ids": sorted(rejected_reasons.keys()),
        "rejection_reasons": {k: rejected_reasons[k] for k in sorted(rejected_reasons.keys())},
        "krum_distances_by_peer": dict(krum_meta["krum_distances_by_peer"]),
        "projected_keys": list(krum_meta["projected_keys"]),
        "skipped_keys": list(krum_meta["skipped_keys"]),
        "raw_l2_norm": float(krum_meta["raw_l2_norm"]),
        "projected_l2_norm": proj_l2,
        "retained_norm_ratio": float(krum_meta["retained_norm_ratio"]),
        "max_prose_key_drift": prose_drift,
        "total_payload_bytes": sum(int(r["payload_bytes"]) for r in accepted_recs),
        "max_peer_mb_per_hour": max((float(r["mb_per_hour"]) for r in accepted_recs), default=0.0),
        "vram_budget_mib": max((float(r["vram_budget_mib"]) for r in accepted_recs), default=0.0),
        "max_preemption_latency_ms": max((float(r["preemption_latency_ms"]) for r in accepted_recs), default=0.0),
        "anse_energy": anse_energy,
        "delta_sha256": compute_tensor_map_sha256(consensus_map),
    }
    return consensus_map, _seal_receipt(receipt_payload)


# ---------------------------------------------------------------------------
# 4. HM-S35-07 — Canonical 16-Level NF4 Edge Exporter with Diagonal-Hessian Feedback
# ---------------------------------------------------------------------------


def _normalize_hessian_diag(
    hessian_diag: torch.Tensor,
    *,
    rows: int,
    cols: int,
    hessian_damping: float,
) -> torch.Tensor:
    """Validate and broadcast ``hessian_diag`` to CPU ``float64`` ``(h_rows, cols)``."""
    if not isinstance(hessian_diag, torch.Tensor) or not hessian_diag.is_floating_point():
        raise TypeError("hessian_diag must be a floating-point torch.Tensor")
    h64 = hessian_diag.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(h64).all().item()):
        raise ValueError("hessian_diag must contain only finite values")

    if h64.ndim == 1 and int(h64.shape[0]) == cols:
        h_mat = h64.unsqueeze(0)
    elif h64.ndim == 2 and tuple(h64.shape) == (rows, cols):
        h_mat = h64
    elif h64.ndim == 2 and tuple(h64.shape) == (cols, cols):
        h_mat = torch.diagonal(h64).unsqueeze(0)
    else:
        raise ValueError(
            f"hessian_diag shape {tuple(h64.shape)} must match ({cols},), ({rows}, {cols}), or ({cols}, {cols})"
        )

    if bool((h_mat < 0.0).any().item()):
        raise ValueError("hessian_diag must contain only non-negative entries")

    h_mean = max(float(h_mat.mean().item()), 1e-12)
    damp_floor = max(hessian_damping * h_mean, 1e-12)
    return (h_mat + damp_floor).contiguous()


def _pack_nf4_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit NF4 indices in ``[0, 15]`` into compact ``torch.uint8`` bytes (2 per byte)."""
    flat = indices.reshape(-1).to(dtype=torch.uint8)
    if flat.numel() % 2 != 0:
        flat = F.pad(flat, (0, 1), mode="constant", value=7)
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def _unpack_nf4_nibbles(packed_nibbles: torch.Tensor, total_elements: int) -> torch.Tensor:
    """Unpack ``torch.uint8`` packed NF4 nibble stream back to ``total_elements`` indices."""
    flat_u8 = packed_nibbles.reshape(-1).to(dtype=torch.long)
    low = flat_u8 & 0x0F
    high = (flat_u8 >> 4) & 0x0F
    interleaved = torch.stack([low, high], dim=1).reshape(-1)
    return interleaved[:total_elements].contiguous()


def _quantize_nf4_nearest(norm_vals: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return nearest canonical 16-level NF4 codebook indices for values in ``[-1, 1]``."""
    clipped = norm_vals.clamp(-1.0, 1.0)
    return (clipped.unsqueeze(-1) - codebook).abs().argmin(dim=-1)


@torch.no_grad()
def dequantize_nf4_compensated_edge_blocks(
    packed_nibbles: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    shape: tuple[int, int],
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct an NF4-exported 2D tensor from packed 4-bit nibbles and block scales."""
    if not isinstance(packed_nibbles, torch.Tensor) or packed_nibbles.dtype != torch.uint8:
        raise TypeError("packed_nibbles must be a torch.uint8 tensor")
    if not isinstance(block_scales, torch.Tensor) or not block_scales.is_floating_point():
        raise TypeError("block_scales must be a floating-point tensor")
    if len(shape) != 2:
        raise ValueError(f"shape must be a 2-tuple (rows, cols), got {shape!r}")
    rows = _validate_positive_int(shape[0], "shape[0]")
    cols = _validate_positive_int(shape[1], "shape[1]")
    blk = _validate_positive_int(block_size, "block_size")

    num_blocks = (cols + blk - 1) // blk
    padded_cols = num_blocks * blk
    if tuple(block_scales.shape) != (rows, num_blocks):
        raise ValueError(
            f"block_scales shape {tuple(block_scales.shape)} does not match ({rows}, {num_blocks})"
        )

    needed_bytes = (rows * padded_cols + 1) // 2
    if int(packed_nibbles.numel()) < needed_bytes:
        raise ValueError(f"packed_nibbles length ({packed_nibbles.numel()}) < required ({needed_bytes})")

    codebook = torch.tensor(CANONICAL_NF4_LEVELS, dtype=torch.float64)
    idx = _unpack_nf4_nibbles(packed_nibbles, rows * padded_cols).reshape(rows, num_blocks, blk)
    scales64 = block_scales.detach().to(device="cpu", dtype=torch.float64).unsqueeze(-1)
    deq = codebook[idx] * scales64
    return deq.reshape(rows, padded_cols)[:, :cols].to(dtype=dtype).contiguous()


def _gptq_cholesky_block_quantize(
    w_blk: torch.Tensor,
    h_sub: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact GPTQ intra-block Cholesky error feedback over columns in descending diagonal order."""
    rows, blk = int(w_blk.shape[0]), int(w_blk.shape[1])
    h_sym = 0.5 * (h_sub + h_sub.T)
    diag = torch.diagonal(h_sym)
    damp = max(0.01 * float(diag.mean().item()), 1e-8)
    h_reg = h_sym + damp * torch.eye(blk, dtype=torch.float64)

    order = torch.argsort(torch.diagonal(h_reg), descending=True)
    h_perm = h_reg.index_select(0, order).index_select(1, order)
    chol = torch.linalg.cholesky(h_perm)
    h_inv_upper = torch.cholesky_inverse(chol)
    h_inv_chol = torch.linalg.cholesky(h_inv_upper, upper=True)

    w_work = w_blk[:, order].clone()
    idx_perm = torch.empty((rows, blk), dtype=torch.long)
    s_vec = scales[:, 0]

    for j in range(blk):
        safe_s = torch.where(s_vec > 0.0, s_vec, torch.ones_like(s_vec))
        u_j = (w_work[:, j] / safe_s).clamp(-1.0, 1.0)
        idx_j = _quantize_nf4_nearest(u_j, codebook)
        idx_j = torch.where(s_vec > 0.0, idx_j, torch.full_like(idx_j, 7))
        idx_perm[:, j] = idx_j
        q_j = codebook[idx_j] * s_vec
        err_j = (w_work[:, j] - q_j) / h_inv_chol[j, j].clamp_min(1e-12)
        if j + 1 < blk:
            w_work[:, j + 1 :] -= err_j.unsqueeze(1) * h_inv_chol[j, j + 1 :].unsqueeze(0)

    inv_order = torch.argsort(order)
    idx_out = idx_perm[:, inv_order].contiguous()
    deq_out = codebook[idx_out] * scales
    return idx_out, deq_out


@torch.no_grad()
def export_nf4_compensated_edge_blocks(
    fp32_edited_weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    *,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    hessian_damping: float = 1e-6,
    hessian_matrix: torch.Tensor | None = None,
    intra_block_coupling: float = 0.02,
    vram_budget_mib: float = 6144.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fold S5 continuous weights into 16-level NF4 blocks with Hessian error compensation."""
    if not isinstance(fp32_edited_weight, torch.Tensor) or fp32_edited_weight.ndim != 2 or fp32_edited_weight.numel() == 0:
        raise ValueError("fp32_edited_weight must be a non-empty 2D torch.Tensor")
    if not fp32_edited_weight.is_floating_point():
        raise TypeError(f"fp32_edited_weight must be floating-point, got {fp32_edited_weight.dtype}")
    w64 = fp32_edited_weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(w64).all().item()):
        raise ValueError("fp32_edited_weight must contain only finite values")

    blk = _validate_positive_int(block_size, "block_size")
    damp_f = _validate_nonneg_finite(hessian_damping, "hessian_damping")
    coupling_f = _validate_nonneg_finite(intra_block_coupling, "intra_block_coupling")
    vram_f = _validate_positive_finite(vram_budget_mib, "vram_budget_mib")
    if vram_f > float(STD_8GB_WDDM_CEILING_MIB):
        raise VRAMCeilingExceededError(
            f"vram_budget_mib ({vram_f:.1f}) exceeds std_8gb WDDM ceiling ({STD_8GB_WDDM_CEILING_MIB} MiB)"
        )

    rows, cols = int(w64.shape[0]), int(w64.shape[1])
    h_eff = _normalize_hessian_diag(hessian_diag, rows=rows, cols=cols, hessian_damping=damp_f)
    h_mat_full: torch.Tensor | None = None
    if hessian_matrix is not None:
        if not isinstance(hessian_matrix, torch.Tensor) or tuple(hessian_matrix.shape) != (cols, cols):
            raise ValueError(f"hessian_matrix must have shape ({cols}, {cols})")
        h_mat_full = hessian_matrix.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(h_mat_full).all().item()):
            raise ValueError("hessian_matrix must contain only finite values")

    num_blocks = (cols + blk - 1) // blk
    padded_cols = num_blocks * blk
    pad_cols = padded_cols - cols

    w_pad = F.pad(w64, (0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else w64
    h_pad = F.pad(h_eff, (0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else h_eff

    w_blk = w_pad.reshape(rows, num_blocks, blk)
    h_blk = h_pad.reshape(int(h_eff.shape[0]), num_blocks, blk)
    codebook = torch.tensor(CANONICAL_NF4_LEVELS, dtype=torch.float64)

    # 1. Naive absmax RTN NF4 baseline
    s_rtn = w_blk.abs().amax(dim=-1, keepdim=True)
    safe_s_rtn = torch.where(s_rtn > 0.0, s_rtn, torch.ones_like(s_rtn))
    idx_rtn = _quantize_nf4_nearest(w_blk / safe_s_rtn, codebook)
    idx_rtn = torch.where(s_rtn > 0.0, idx_rtn, torch.full_like(idx_rtn, 7))
    deq_rtn = codebook[idx_rtn] * s_rtn
    err_rtn_mat = h_blk * (w_blk - deq_rtn).pow(2)
    naive_h_err = float(err_rtn_mat.sum().item())
    naive_fro_err = float(torch.linalg.matrix_norm(w64 - deq_rtn.reshape(rows, padded_cols)[:, :cols], ord="fro").item())

    # 2. Hessian-weighted optimal block scale search + alternating WLS refinement
    best_blk_err = err_rtn_mat.sum(dim=-1, keepdim=True)
    best_scales = s_rtn.clone()
    best_idx = idx_rtn.clone()
    best_deq = deq_rtn.clone()

    for alpha in torch.linspace(0.45, 1.05, 25, dtype=torch.float64):
        cand_s = s_rtn * float(alpha.item())
        for _ in range(4):
            safe_s = torch.where(cand_s > 0.0, cand_s, torch.ones_like(cand_s))
            cand_idx = _quantize_nf4_nearest(w_blk / safe_s, codebook)
            cand_idx = torch.where(cand_s > 0.0, cand_idx, torch.full_like(cand_idx, 7))
            codes = codebook[cand_idx]
            num = (h_blk * w_blk * codes).sum(dim=-1, keepdim=True)
            den = (h_blk * codes * codes).sum(dim=-1, keepdim=True)
            cand_s = torch.where((s_rtn > 0.0) & (den > 1e-12), (num / den.clamp_min(1e-12)).clamp_min(1e-12), cand_s)
        cand_deq = codebook[cand_idx] * cand_s
        cand_err = (h_blk * (w_blk - cand_deq).pow(2)).sum(dim=-1, keepdim=True)
        better = cand_err < best_blk_err
        best_blk_err = torch.where(better, cand_err, best_blk_err)
        best_scales = torch.where(better, cand_s, best_scales)
        best_idx = torch.where(better, cand_idx, best_idx)
        best_deq = torch.where(better, cand_deq, best_deq)

    # 3. GPTQ-style descending-Hessian intra-block error feedback pass (Cholesky or Sherman-Morrison)
    if h_mat_full is not None:
        h_mat_pad = F.pad(h_mat_full, (0, pad_cols, 0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else h_mat_full
        for b_idx in range(num_blocks):
            c0, c1 = b_idx * blk, (b_idx + 1) * blk
            h_sub = h_mat_pad[c0:c1, c0:c1]
            fb_idx, fb_deq = _gptq_cholesky_block_quantize(
                w_blk[:, b_idx, :], h_sub, best_scales[:, b_idx, :], codebook
            )
            best_idx[:, b_idx, :] = fb_idx
            best_deq[:, b_idx, :] = fb_deq
    elif coupling_f > 0.0:
        h_col = h_blk.mean(dim=0)  # (num_blocks, blk)
        for b_idx in range(num_blocks):
            hb = h_col[b_idx].clamp_min(1e-12)
            order = torch.argsort(hb, descending=True)
            inv_h = 1.0 / hb
            wb_work = w_blk[:, b_idx, :].clone()
            s_vec = best_scales[:, b_idx, 0]
            fb_idx = torch.empty((rows, blk), dtype=torch.long)
            fb_deq = torch.empty((rows, blk), dtype=torch.float64)
            for step in range(blk):
                j = int(order[step].item())
                safe_s = torch.where(s_vec > 0.0, s_vec, torch.ones_like(s_vec))
                u_j = (wb_work[:, j] / safe_s).clamp(-1.0, 1.0)
                idx_j = torch.where(s_vec > 0.0, _quantize_nf4_nearest(u_j, codebook), torch.full((rows,), 7, dtype=torch.long))
                q_j = codebook[idx_j] * s_vec
                fb_idx[:, j] = idx_j
                fb_deq[:, j] = q_j
                if step + 1 < blk:
                    rem = order[step + 1 :]
                    err_j = wb_work[:, j] - q_j
                    w_rem = inv_h[rem]
                    denom = (1.0 / coupling_f) * inv_h[j] + w_rem.sum()
                    wb_work[:, rem] += (err_j.unsqueeze(1) * w_rem.unsqueeze(0)) / denom
            fb_err = (h_blk[:, b_idx, :] * (w_blk[:, b_idx, :] - fb_deq).pow(2)).sum(dim=-1, keepdim=True)
            use_fb = fb_err < best_blk_err[:, b_idx, :]
            best_idx[:, b_idx, :] = torch.where(use_fb, fb_idx, best_idx[:, b_idx, :])
            best_deq[:, b_idx, :] = torch.where(use_fb, fb_deq, best_deq[:, b_idx, :])

    # 4. Pack exact 4-bit nibbles + float32 block scales and reconstruct bitwise
    scales_f32 = best_scales.squeeze(-1).to(dtype=torch.float32).contiguous()
    packed_nibbles = _pack_nf4_nibbles(best_idx)
    w_deq_out = dequantize_nf4_compensated_edge_blocks(
        packed_nibbles,
        scales_f32,
        shape=(rows, cols),
        block_size=blk,
        dtype=fp32_edited_weight.dtype,
    ).to(device=fp32_edited_weight.device)

    w_deq64 = w_deq_out.detach().to(device="cpu", dtype=torch.float64)
    comp_h_err = float((h_eff * (w64 - w_deq64).pow(2)).sum().item())
    comp_fro_err = float(torch.linalg.matrix_norm(w64 - w_deq64, ord="fro").item())
    reduction = ((naive_h_err - comp_h_err) / naive_h_err) if naive_h_err > 1e-18 else 0.0

    receipt_payload: dict[str, Any] = {
        "object": EDGE_EXPORT_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "quant_format": "nf4",
        "block_size": blk,
        "shape": [rows, cols],
        "padded_cols": padded_cols,
        "num_blocks_per_row": num_blocks,
        "total_blocks": rows * num_blocks,
        "nf4_levels": list(CANONICAL_NF4_LEVELS),
        "packed_nibbles": packed_nibbles,
        "block_scales": scales_f32,
        "packed_nibbles_sha256": compute_tensor_sha256(packed_nibbles),
        "block_scales_sha256": compute_tensor_sha256(scales_f32),
        "dequant_weight_sha256": compute_tensor_sha256(w_deq_out),
        "hessian_weighted_error": comp_h_err,
        "naive_hessian_weighted_error": naive_h_err,
        "hessian_error_reduction_ratio": reduction,
        "frobenius_error": comp_fro_err,
        "naive_frobenius_error": naive_fro_err,
        "bits_per_weight": float((4.0 * padded_cols + 32.0 * num_blocks) / float(cols)),
        "edge_payload_bytes": int(packed_nibbles.numel() + scales_f32.numel() * 4),
        "vram_budget_mib": vram_f,
        "anse_energy": round(comp_h_err - naive_h_err, 9),
    }
    return w_deq_out, _seal_receipt(receipt_payload)


@torch.no_grad()
def export_s5_model_to_nf4_edge(
    weights: Mapping[str, torch.Tensor],
    hessian_diags: Mapping[str, torch.Tensor],
    *,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    eval_fn: Callable[[Mapping[str, torch.Tensor]], float | Mapping[str, float]] | None = None,
    fp32_baseline_nll: float | None = None,
    max_nll_gap_nats: float = MAX_NF4_NLL_GAP_NATS,
    vram_budget_mib: float = 6144.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Export S5 continuous weight mapping into 4-bit NF4 edge blocks and verify <0.02 nat NLL gap."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a non-empty mapping of named tensors")
    if not isinstance(hessian_diags, Mapping):
        raise TypeError("hessian_diags must be a mapping")
    blk = _validate_positive_int(block_size, "block_size")
    max_gap = _validate_positive_finite(max_nll_gap_nats, "max_nll_gap_nats")
    vram_f = _validate_positive_finite(vram_budget_mib, "vram_budget_mib")
    if vram_f > float(STD_8GB_WDDM_CEILING_MIB):
        raise VRAMCeilingExceededError(
            f"vram_budget_mib ({vram_f:.1f}) exceeds std_8gb WDDM ceiling ({STD_8GB_WDDM_CEILING_MIB} MiB)"
        )

    quantized_weights: dict[str, torch.Tensor] = {}
    per_tensor_meta: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    total_h_err = 0.0
    total_naive_h_err = 0.0

    for name in sorted(weights.keys()):
        w_t = weights[name]
        if not isinstance(w_t, torch.Tensor):
            raise TypeError(f"weights[{name!r}] must be a torch.Tensor")
        if w_t.ndim == 2 and name in hessian_diags:
            w_q, rec = export_nf4_compensated_edge_blocks(
                w_t, hessian_diags[name], block_size=blk, vram_budget_mib=vram_f
            )
            quantized_weights[name] = w_q
            total_bytes += int(rec["edge_payload_bytes"])
            total_h_err += float(rec["hessian_weighted_error"])
            total_naive_h_err += float(rec["naive_hessian_weighted_error"])
            per_tensor_meta[name] = {
                "quantized": True,
                "edge_payload_bytes": int(rec["edge_payload_bytes"]),
                "hessian_weighted_error": float(rec["hessian_weighted_error"]),
                "naive_hessian_weighted_error": float(rec["naive_hessian_weighted_error"]),
                "receipt_sha256": rec["receipt_sha256"],
            }
        else:
            w_cpu = w_t.detach().to(device="cpu").contiguous()
            quantized_weights[name] = w_cpu.clone()
            raw_bytes = int(w_cpu.numel() * w_cpu.element_size())
            total_bytes += raw_bytes
            per_tensor_meta[name] = {"quantized": False, "edge_payload_bytes": raw_bytes}

    nll_fp32: float | None = (
        _validate_finite_float(fp32_baseline_nll, "fp32_baseline_nll")
        if fp32_baseline_nll is not None
        else None
    )
    nll_nf4: float | None = None
    nll_gap_nats = 0.0

    if eval_fn is not None:
        if nll_fp32 is None:
            fp32_res = eval_fn(weights)
            nll_fp32 = _extract_scalar_nll(fp32_res, "fp32_eval")
        nf4_res = eval_fn(quantized_weights)
        nll_nf4 = _extract_scalar_nll(nf4_res, "nf4_eval")
        nll_gap_nats = nll_nf4 - nll_fp32
        if nll_gap_nats >= max_gap:
            raise ValueError(
                f"NF4 edge export NLL gap ({nll_gap_nats:.5f} nats) exceeds maximum allowed ({max_gap:.5f} nats)"
            )

    edge_vram_mib = total_bytes / (1024.0 * 1024.0)
    if edge_vram_mib > vram_f:
        raise VRAMCeilingExceededError(
            f"Exported NF4 edge footprint ({edge_vram_mib:.2f} MiB) exceeds budget ({vram_f:.1f} MiB)"
        )

    anse_energy = round(nll_gap_nats + (total_h_err - total_naive_h_err), 9)
    receipt_payload: dict[str, Any] = {
        "object": MODEL_EDGE_EXPORT_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "block_size": blk,
        "num_tensors": len(quantized_weights),
        "edge_payload_bytes": total_bytes,
        "edge_vram_mib": round(edge_vram_mib, 6),
        "vram_budget_mib": vram_f,
        "fp32_nll_nats": nll_fp32,
        "nf4_nll_nats": nll_nf4,
        "nll_gap_nats": nll_gap_nats,
        "hessian_weighted_error": total_h_err,
        "naive_hessian_weighted_error": total_naive_h_err,
        "anse_energy": anse_energy,
        "per_tensor": per_tensor_meta,
        "exported_weights_sha256": compute_tensor_map_sha256(quantized_weights),
    }
    return quantized_weights, _seal_receipt(receipt_payload)


def _extract_scalar_nll(raw: float | Mapping[str, float], label: str) -> float:
    """Extract finite scalar NLL from float or mapping returned by ``eval_fn``."""
    if isinstance(raw, Mapping):
        if "nll_mean" in raw:
            return _validate_finite_float(raw["nll_mean"], f"{label}.nll_mean")
        vals = [_validate_finite_float(v, f"{label}[{k!r}]") for k, v in raw.items()]
        if not vals:
            raise ValueError(f"{label} mapping must not be empty")
        return float(sum(vals) / float(len(vals)))
    return _validate_finite_float(raw, label)
