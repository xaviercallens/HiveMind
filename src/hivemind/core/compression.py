# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pseudo-Gradient Delta Compression via Top-k Sparsification & FP8 (E4M3) Quantization.

Implements the Project HiveMind communication compression layer:
  Delta_k^(t) = theta^(t-1) - theta_k^(t, H)
Compressed via Top-k magnitude selection and FP8 (float8_e4m3fn) quantization,
achieving 4x-10x wire reduction on top of DiLoCo's 500x step-frequency reduction.

Season 8 (`HM-S8-05`, `SI-09C`) adds the `DeMo`/`SparseLoCo` path: an orthonormal
2D chunked Type-II DCT decorrelates spatially structured LoRA pseudo-gradients, only
the top-`k` frequency coefficients are retained, and those coefficients are quantized
through a 2-bit (`4`-bin) statistical centroid LUT packed four codes per `uint8`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
import torch.nn.functional as F


FP8_E4M3_MAX = 448.0
COMPRESSION_MODES = ("topk_fp8", "dct_sparse_2bit")
DCT_SPARSE_2BIT_MIN_COMPRESSION_RATIO = 16.0


@dataclass
class CompressedTensorPayload:
    """Wire representation of a single Top-k + FP8 compressed LoRA parameter delta."""

    original_shape: tuple[int, ...]
    original_numel: int
    indices: torch.Tensor       # int32 indices of top-k entries
    fp8_values: torch.Tensor    # float8_e4m3fn (or uint8-packed) quantized values
    scale: float                # absmax / FP8_E4M3_MAX dequantization factor

    @property
    def compressed_bytes(self) -> int:
        # 1 byte per FP8 value + 2 bytes per delta-coded/uint16 index in block-sparse wire format
        return int(self.fp8_values.numel() * 1 + self.indices.numel() * 2 + 8)


@dataclass
class DCTSparse2BitTensorPayload:
    """Wire representation of one chunked-DCT + 2-bit centroid compressed LoRA delta.

    `indices` addresses the flat, padded DCT coefficient space. `packed_2bit` holds four
    2-bit LUT codes per `uint8` byte (`0.25 B` per retained coefficient), and `lookup`
    stores the four empirical centroid values (`float16` on the wire).
    """

    original_shape: tuple[int, ...]
    original_numel: int
    padded_shape: tuple[int, int]
    indices: torch.Tensor       # int32 flat indices into the chunked DCT coefficient space
    packed_2bit: torch.Tensor   # uint8, four 2-bit LUT codes per byte
    lookup: torch.Tensor        # float16 empirical centroid LUT (4 bins)
    shift: float                # per-tensor mean removed before centroid quantization
    scale: float                # statistical centroid bin width (sigma-derived)
    n_coefficients: int
    chunk_size: int
    index_width: int            # 1, 2 or 4 bytes per transmitted index
    dct_energy_retained_fraction: float

    @property
    def compressed_bytes(self) -> int:
        """Accurate wire size: packed codes + indices + float16 LUT + 12-byte header.

        The header carries the `float32` shift and scale plus four format bytes
        (`chunk_size`, `n_bins`, `index_width`, and a reserved flags byte).
        """
        return int(
            self.packed_2bit.numel()
            + self.indices.numel() * self.index_width
            + self.lookup.numel() * 2
            + 12
        )


@dataclass
class CompressedDeltaPackage:
    """Complete compressed pseudo-gradient package uploaded by a BOINC worker node."""

    worker_id: str
    branch_name: str
    outer_step: int
    inner_steps_completed: int
    topk_ratio: float
    tensors: dict[str, CompressedTensorPayload | DCTSparse2BitTensorPayload]
    uncompressed_bytes: int
    compressed_bytes: int
    compression_mode: str = "topk_fp8"

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 1.0
        return round(self.uncompressed_bytes / self.compressed_bytes, 2)


def compute_pseudo_gradient_delta(
    theta_init: dict[str, torch.Tensor],
    theta_final: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the DiLoCo pseudo-gradient: Delta_k^(t) = theta^(t-1) - theta_k^(t, H)."""
    delta: dict[str, torch.Tensor] = {}
    for name, init_t in theta_init.items():
        if name not in theta_final:
            raise KeyError(f"Missing LoRA parameter '{name}' in final worker state.")
        fin_t = theta_final[name]
        delta[name] = (init_t.detach().to(torch.float32).cpu() - fin_t.detach().to(torch.float32).cpu())
    return delta


def compress_tensor_topk_fp8(
    tensor: torch.Tensor,
    topk_ratio: float = 0.20,
) -> CompressedTensorPayload:
    """Compress a 1D/2D LoRA delta tensor using Top-k sparsification and FP8 E4M3 quantization."""
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
    numel = flat.numel()
    k = max(1, int(round(numel * topk_ratio)))

    if k >= numel:
        indices = torch.arange(numel, dtype=torch.int32)
        selected = flat
    else:
        abs_vals = flat.abs()
        # Deterministic top-k selection
        _, top_idx = torch.topk(abs_vals, k=k, largest=True, sorted=True)
        indices = top_idx.to(torch.int32)
        selected = flat[top_idx]

    absmax = float(selected.abs().max().item()) if selected.numel() > 0 else 0.0
    if absmax == 0.0:
        scale = 1.0
        scaled = torch.zeros_like(selected)
    else:
        scale = absmax / FP8_E4M3_MAX
        scaled = (selected / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    if hasattr(torch, "float8_e4m3fn"):
        fp8_vals = scaled.to(torch.float8_e4m3fn)
    else:
        # Fallback uniform int8 quantization if float8 dtype unavailable
        fp8_vals = torch.round(scaled / (FP8_E4M3_MAX / 127.0)).to(torch.int8)

    return CompressedTensorPayload(
        original_shape=tuple(tensor.shape),
        original_numel=numel,
        indices=indices,
        fp8_values=fp8_vals,
        scale=scale,
    )


def decompress_tensor_topk_fp8(
    payload: CompressedTensorPayload,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decompress a Top-k + FP8 payload back into a dense tensor of original_shape."""
    out = torch.zeros(payload.original_numel, dtype=torch.float32, device="cpu")
    if hasattr(torch, "float8_e4m3fn") and payload.fp8_values.dtype == torch.float8_e4m3fn:
        dequant = payload.fp8_values.to(torch.float32) * payload.scale
    else:
        dequant = payload.fp8_values.to(torch.float32) * (FP8_E4M3_MAX / 127.0) * payload.scale

    out.index_copy_(0, payload.indices.to(torch.long), dequant)
    return out.reshape(payload.original_shape).to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _orthonormal_dct_basis(chunk_size: int) -> torch.Tensor:
    """Return the cached `chunk_size x chunk_size` orthonormal Type-II DCT basis.

    Reuses the `SI-09C` primitive in `hivemind.swarm.diloco_nextgen`; the import is
    function-local so the core compression layer does not import the swarm package at
    module-import time.
    """
    from hivemind.swarm.diloco_nextgen import ChunkedOrthonormalDCTTransform

    return ChunkedOrthonormalDCTTransform(chunk_size=chunk_size).basis


def _prepare_chunked_dct_matrix(
    tensor: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reshape an N-D delta to 2D and zero-pad it to whole DCT chunks."""
    if tensor.numel() == 0:
        raise ValueError("cannot DCT-compress an empty tensor")
    matrix = tensor if tensor.ndim == 2 else tensor.reshape(1, -1)
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    pad_rows = (-rows) % chunk_size
    pad_cols = (-cols) % chunk_size
    if pad_rows or pad_cols:
        matrix = F.pad(matrix, (0, pad_cols, 0, pad_rows))
    return matrix.contiguous(), (rows + pad_rows, cols + pad_cols)


def _chunked_dct_forward(matrix: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project a padded 2D matrix onto the flat chunked DCT coefficient space."""
    n = int(basis.shape[0])
    rows, cols = matrix.shape
    blocks = matrix.reshape(rows // n, n, cols // n, n).permute(0, 2, 1, 3).contiguous()
    transformed = torch.matmul(torch.matmul(basis, blocks), basis.t())
    return transformed.reshape(-1)


def _chunked_dct_inverse(
    coefficients: torch.Tensor,
    padded_shape: tuple[int, int],
    basis: torch.Tensor,
) -> torch.Tensor:
    """Invert a flat chunked DCT coefficient vector back to the padded 2D matrix."""
    n = int(basis.shape[0])
    rows, cols = padded_shape
    blocks = coefficients.reshape(rows // n, cols // n, n, n)
    spatial = torch.matmul(torch.matmul(basis.t(), blocks), basis)
    return spatial.permute(0, 2, 1, 3).contiguous().reshape(rows, cols)


def _index_width(numel: int) -> int:
    """Smallest unsigned wire width that can address every coefficient of `numel`."""
    if numel <= 0x100:
        return 1
    if numel <= 0x10000:
        return 2
    return 4


def _pack_2bit_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack four 2-bit codes into each `uint8` byte (least-significant code first)."""
    packed_input = codes.detach().to(torch.uint8).reshape(-1)
    padding = (-packed_input.numel()) % 4
    if padding:
        packed_input = torch.cat(
            [packed_input, torch.zeros(padding, dtype=torch.uint8)]
        )
    quads = packed_input.reshape(-1, 4).to(torch.int16)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    return packed.to(torch.uint8)


def _unpack_2bit_codes(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack `count` 2-bit codes from four-codes-per-byte `uint8` payloads."""
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int16)
    codes = (packed.detach().to(torch.int16).reshape(-1, 1) >> shifts) & 0x03
    return codes.reshape(-1)[:count].to(torch.long)


def _quantize_2bit_centroids(
    values: torch.Tensor,
    n_bins: int,
    range_in_sigmas: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Quantize values with a `4`-bin empirical statistical-centroid LUT (2 bits/value)."""
    if n_bins != 4:
        raise ValueError(f"2-bit compression requires exactly 4 bins, got {n_bins}")
    if range_in_sigmas <= 0.0:
        raise ValueError("range_in_sigmas must be positive")

    count = int(values.numel())
    shift = float(values.mean().item())
    centered = values - shift
    std = float(centered.norm().item() / math.sqrt(max(1, count - 1)))
    if std <= 1e-12:
        return (
            torch.zeros(count, dtype=torch.uint8),
            torch.zeros(n_bins, dtype=torch.float16),
            shift,
            1.0,
        )

    scale = float(range_in_sigmas * std / n_bins)
    center_offset = 0.5 * float(n_bins - 1)
    codes = ((centered / scale) + center_offset).round().clamp(0, n_bins - 1).to(torch.uint8)

    lookup = torch.empty(n_bins, dtype=torch.float32)
    for bin_index in range(n_bins):
        in_bin = codes == bin_index
        if bool(in_bin.any().item()):
            lookup[bin_index] = centered[in_bin].mean()
        else:
            # Preserve the symmetric bin centre for empty bins instead of collapsing it to zero (IP-37).
            lookup[bin_index] = (float(bin_index) - center_offset) * scale
    lookup = lookup.clamp(-65504.0, 65504.0).to(torch.float16)
    return codes, lookup, shift, scale


def compress_tensor_dct_sparse_2bit(
    tensor: torch.Tensor,
    topk_ratio: float = 0.02,
    chunk_size: int = 8,
    n_bins: int = 4,
    range_in_sigmas: float = 4.0,
) -> DCTSparse2BitTensorPayload:
    """Compress one LoRA delta tensor via chunked orthonormal DCT + 2-bit centroid sparsity."""
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
    if chunk_size < 2:
        raise ValueError("chunk_size must be >= 2")
    if n_bins != 4:
        raise ValueError(f"2-bit compression requires exactly 4 bins, got {n_bins}")

    delta = tensor.detach().to(torch.float32).cpu()
    matrix, padded_shape = _prepare_chunked_dct_matrix(delta, int(chunk_size))
    basis = _orthonormal_dct_basis(int(chunk_size))
    frequencies = _chunked_dct_forward(matrix, basis)

    total = int(frequencies.numel())
    k = min(total, max(1, int(round(total * float(topk_ratio)))))
    _, indices = torch.topk(frequencies.abs(), k=k, largest=True, sorted=True)
    selected = frequencies[indices]
    codes, lookup, shift, scale = _quantize_2bit_centroids(
        selected,
        n_bins=int(n_bins),
        range_in_sigmas=float(range_in_sigmas),
    )

    total_energy = float(frequencies.pow(2).sum().item())
    retained = (
        float(selected.pow(2).sum().item()) / total_energy if total_energy > 0.0 else 1.0
    )
    return DCTSparse2BitTensorPayload(
        original_shape=tuple(int(dim) for dim in tensor.shape),
        original_numel=int(tensor.numel()),
        padded_shape=padded_shape,
        indices=indices.to(torch.int32),
        packed_2bit=_pack_2bit_codes(codes),
        lookup=lookup,
        shift=shift,
        scale=scale,
        n_coefficients=k,
        chunk_size=int(chunk_size),
        index_width=_index_width(total),
        dct_energy_retained_fraction=round(retained, 6),
    )


def decompress_tensor_dct_sparse_2bit(
    payload: DCTSparse2BitTensorPayload,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reconstruct a dense tensor from a chunked-DCT + 2-bit centroid payload."""
    if not isinstance(payload, DCTSparse2BitTensorPayload):
        raise TypeError("payload must be a DCTSparse2BitTensorPayload")
    if payload.n_coefficients != int(payload.indices.numel()):
        raise ValueError("DCT payload coefficient count does not match its index tensor")

    codes = _unpack_2bit_codes(payload.packed_2bit, payload.n_coefficients)
    values = payload.lookup.to(torch.float32)[codes] + float(payload.shift)
    frequencies = torch.zeros(
        int(payload.padded_shape[0]) * int(payload.padded_shape[1]), dtype=torch.float32
    )
    frequencies.index_copy_(0, payload.indices.to(torch.long), values)
    matrix = _chunked_dct_inverse(
        frequencies,
        payload.padded_shape,
        _orthonormal_dct_basis(payload.chunk_size),
    )
    if len(payload.original_shape) == 2:
        # Padded 2-D matrices must be cropped on both axes; flattening first would
        # splice padded column values into later rows.
        original_rows, original_cols = payload.original_shape
        cropped = matrix[:original_rows, :original_cols]
    else:
        cropped = matrix.reshape(-1)[: payload.original_numel]
    return cropped.reshape(payload.original_shape).to(device=device, dtype=dtype)


def compress_lora_delta(
    delta_dict: dict[str, torch.Tensor],
    worker_id: str,
    branch_name: str,
    outer_step: int,
    inner_steps_completed: int = 500,
    topk_ratio: float | None = None,
    compression_mode: str = "topk_fp8",
    dct_chunk_size: int = 8,
    dct_n_bins: int = 4,
) -> CompressedDeltaPackage:
    """Compress an entire dictionary of LoRA pseudo-gradients for BOINC upload.

    `compression_mode="topk_fp8"` keeps the legacy Top-k + FP8 E4M3 wire format and its
    `0.20` default density. `compression_mode="dct_sparse_2bit"` selects the `SI-09C`
    chunked-DCT + 2-bit statistical-centroid format, whose lower default density (`2%`)
    is required to clear the `16x` BF16 wire-reduction target once sparse indices and
    the per-tensor LUT are counted honestly.
    """
    if compression_mode not in COMPRESSION_MODES:
        raise ValueError(
            f"compression_mode must be one of {COMPRESSION_MODES}, got {compression_mode!r}"
        )
    if topk_ratio is None:
        topk_ratio = 0.02 if compression_mode == "dct_sparse_2bit" else 0.20
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    if compression_mode == "dct_sparse_2bit":
        return compress_lora_delta_dct_sparse_2bit(
            delta_dict=delta_dict,
            worker_id=worker_id,
            branch_name=branch_name,
            outer_step=outer_step,
            inner_steps_completed=inner_steps_completed,
            topk_ratio=topk_ratio,
            chunk_size=dct_chunk_size,
            n_bins=dct_n_bins,
        )

    tensors: dict[str, CompressedTensorPayload] = {}
    uncompressed_bytes = 0
    compressed_bytes = 0

    for name, tensor in delta_dict.items():
        # Reference uncompressed size in BF16/FP16 (2 bytes per parameter)
        uncompressed_bytes += tensor.numel() * 2
        payload = compress_tensor_topk_fp8(tensor, topk_ratio=topk_ratio)
        tensors[name] = payload
        compressed_bytes += payload.compressed_bytes

    return CompressedDeltaPackage(
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        inner_steps_completed=inner_steps_completed,
        topk_ratio=topk_ratio,
        tensors=tensors,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        compression_mode="topk_fp8",
    )


def compress_lora_delta_dct_sparse_2bit(
    delta_dict: dict[str, torch.Tensor],
    worker_id: str,
    branch_name: str,
    outer_step: int,
    inner_steps_completed: int = 500,
    topk_ratio: float = 0.02,
    chunk_size: int = 8,
    n_bins: int = 4,
) -> CompressedDeltaPackage:
    """Compress LoRA pseudo-gradients via chunked orthonormal DCT + 2-bit centroid sparsity."""
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    tensors: dict[str, DCTSparse2BitTensorPayload] = {}
    uncompressed_bytes = 0
    compressed_bytes = 0

    for name, tensor in delta_dict.items():
        # Reference uncompressed size in BF16/FP16 (2 bytes per parameter)
        uncompressed_bytes += tensor.numel() * 2
        payload = compress_tensor_dct_sparse_2bit(
            tensor,
            topk_ratio=topk_ratio,
            chunk_size=chunk_size,
            n_bins=n_bins,
        )
        tensors[name] = payload
        compressed_bytes += payload.compressed_bytes

    return CompressedDeltaPackage(
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        inner_steps_completed=inner_steps_completed,
        topk_ratio=topk_ratio,
        tensors=tensors,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        compression_mode="dct_sparse_2bit",
    )


def decompress_lora_delta(
    package: CompressedDeltaPackage,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Reconstruct dense LoRA pseudo-gradient dictionary from a CompressedDeltaPackage."""
    if package.compression_mode == "topk_fp8":
        return {
            name: decompress_tensor_topk_fp8(payload, dtype=dtype, device=device)
            for name, payload in package.tensors.items()
        }
    if package.compression_mode == "dct_sparse_2bit":
        return decompress_lora_delta_dct_sparse_2bit(package, dtype=dtype, device=device)
    raise ValueError(
        f"unsupported package compression_mode={package.compression_mode!r}; "
        f"expected one of {COMPRESSION_MODES}"
    )


def decompress_lora_delta_dct_sparse_2bit(
    package: CompressedDeltaPackage,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Reconstruct a chunked-DCT + 2-bit compressed LoRA pseudo-gradient package."""
    if package.compression_mode != "dct_sparse_2bit":
        raise ValueError(
            "decompress_lora_delta_dct_sparse_2bit requires "
            f"compression_mode='dct_sparse_2bit', got {package.compression_mode!r}"
        )

    restored: dict[str, torch.Tensor] = {}
    for name, payload in package.tensors.items():
        if not isinstance(payload, DCTSparse2BitTensorPayload):
            raise TypeError(
                f"tensor '{name}' is not a DCTSparse2BitTensorPayload; "
                "package mode and payload wire format disagree"
            )
        restored[name] = decompress_tensor_dct_sparse_2bit(
            payload, dtype=dtype, device=device
        )
    return restored
