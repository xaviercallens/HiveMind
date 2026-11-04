# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pluggable KV page codecs for the inference ring (roadmap ``HM-S10-05``).

A ``KVPageCodec`` turns one K or V page (last axis = head_dim) into an opaque payload and back, and reports the
bits it actually stores (side information included). ``PluggableKVPageCache`` is ``FP8PagedKVCache`` with the
codec swappable. The default codec is ``FP8E4M3PageCodec`` (the Season-5 FP8 path, unchanged) and stays the default:
whether any other codec is adopted is decided by experiment E11 (``scripts/run_experiment_E11_kv_lattice.py``,
preregistered, not finished). Nothing here is evidence for adopting a non-FP8 codec.

References: ``hivemind.core.btm_moe.FP8PagedKVCache`` (FP8 path), ``hivemind.compress.kv_lattice`` (lattice
decoders, factorised rate), Conway & Sloane 1982 (nearest-point algorithms).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import torch

from hivemind.compress.kv_lattice import DECODERS_T, factorised_rate_t, from_rows, to_rows
from hivemind.core.btm_moe import FP8PagedKVCache

Tensor = torch.Tensor


@runtime_checkable
class KVPageCodec(Protocol):
    name: str

    def encode(self, page: Tensor) -> Any:
        """Encode a page tensor into codec payload."""
        raise TypeError(f"{type(self).__name__} must implement encode()")

    def decode(self, payload: Any, dtype: torch.dtype = torch.float16) -> Tensor:
        """Decode codec payload back to tensor."""
        raise TypeError(f"{type(self).__name__} must implement decode()")

    def payload_bits(self, payload: Any) -> float:
        """Compute bit cost of payload."""
        raise TypeError(f"{type(self).__name__} must implement payload_bits()")


class FP8E4M3PageCodec:
    """Per-row absmax FP8 (e4m3) with an fp16 scale: delegates to the unchanged ``FP8PagedKVCache`` statics."""

    name = "fp8_e4m3"

    def encode(self, page: Tensor) -> Tuple[Tensor, Tensor]:
        return FP8PagedKVCache._quantize_page_fp8(page)

    def decode(self, payload: Tuple[Tensor, Tensor], dtype: torch.dtype = torch.float16) -> Tensor:
        q, scale = payload
        return FP8PagedKVCache._dequantize_page_fp8(q, scale, target_dtype=dtype)

    def payload_bits(self, payload: Tuple[Tensor, Tensor]) -> float:
        q, scale = payload
        return float(q.numel() * q.element_size() * 8 + scale.numel() * scale.element_size() * 8)


def _require_encodable(codec: str, page: Tensor, *side: Tensor) -> None:
    """Refuse a page the fp16 side information cannot represent, instead of returning a payload that decodes to inf/NaN.

    NaN or inf in the page, or a finite page whose per-row min/max/RMS overflows fp16 (above 65504), would otherwise be
    encoded silently and `payload_bits` would still report a cost for a payload that is unusable."""
    if not bool(torch.isfinite(page).all()):
        raise ValueError(f"{codec}: page contains NaN or inf")
    if not all(bool(torch.isfinite(t).all()) for t in side):
        raise ValueError(f"{codec}: per-row side information overflows fp16 (page magnitude above 65504)")


class IntMinMaxPageCodec:
    """Asymmetric per-row min-max integer quantiser: ``2**bits`` levels, fp16 min and max stored per row."""

    def __init__(self, bits: int) -> None:
        if not 1 <= int(bits) <= 16:
            raise ValueError("bits must be in 1..16")
        self.bits = int(bits)
        self.name = f"int{self.bits}_minmax"

    def encode(self, page: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = page.detach().to(torch.float32)
        mn = x.amin(dim=-1, keepdim=True).to(torch.float16)
        mx = x.amax(dim=-1, keepdim=True).to(torch.float16)
        _require_encodable(self.name, x, mn, mx)
        step = ((mx.to(torch.float32) - mn.to(torch.float32)) / float((1 << self.bits) - 1)).clamp_min(1e-12)
        codes = torch.round((x - mn.to(torch.float32)) / step).clamp(0, (1 << self.bits) - 1)
        return codes.to(torch.int32), mn, mx

    def decode(self, payload: Tuple[Tensor, Tensor, Tensor], dtype: torch.dtype = torch.float16) -> Tensor:
        codes, mn, mx = payload
        mn32, mx32 = mn.to(torch.float32), mx.to(torch.float32)
        step = ((mx32 - mn32) / float((1 << self.bits) - 1)).clamp_min(1e-12)
        return (mn32 + codes.to(torch.float32) * step).to(dtype)

    def payload_bits(self, payload: Tuple[Tensor, Tensor, Tensor]) -> float:
        codes, mn, mx = payload
        return float(codes.numel() * self.bits + (mn.numel() + mx.numel()) * 16)


class LatticePageCodec:
    """Lattice quantiser (Z blocks of 8, A2 per RoPE plane, or E8) on per-row RMS-normalised vectors.

    Each head_dim vector is divided by its fp16 RMS, quantised on ``scale * L``, and decoded as ``rms * c * scale``.
    ``payload_bits`` = measured factorised rate (bits/coord, plug-in entropy of this payload's own codewords given
    the in-head group; biased low on small pages) * coords + 16 bits of RMS per row. Codewords are kept as
    float64 unscaled lattice points (simulation container; the accounted size is ``payload_bits``).
    """

    def __init__(self, kind: str, scale: float) -> None:
        if kind not in DECODERS_T:
            raise ValueError(f"kind must be one of {sorted(DECODERS_T)}")
        if not scale > 0:
            raise ValueError("scale must be positive")
        self.kind = kind
        self.scale = float(scale)
        self.name = f"lattice_{kind}_s{self.scale:g}"

    def encode(self, page: Tensor) -> Dict[str, Any]:
        x = page.detach().to(torch.float64)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6).to(torch.float16)
        _require_encodable(self.name, x, rms)
        rows, ctx = to_rows(self.kind, x / rms.to(torch.float64))
        c = DECODERS_T[self.kind](rows / self.scale)
        rate = factorised_rate_t(self.kind, c, ctx)
        return {"codewords": c, "rms": rms, "shape": tuple(page.shape), "rate": rate, "coords": page.numel()}

    def decode(self, payload: Dict[str, Any], dtype: torch.dtype = torch.float16) -> Tensor:
        xn = from_rows(self.kind, payload["codewords"] * self.scale, torch.Size(payload["shape"]))
        return (xn * payload["rms"].to(torch.float64)).to(dtype)

    def payload_bits(self, payload: Dict[str, Any]) -> float:
        return float(payload["rate"] * payload["coords"] + 16 * payload["rms"].numel())


class PluggableKVPageCache:
    """``FP8PagedKVCache`` API with a swappable page codec (``codec=None`` -> ``FP8E4M3PageCodec``)."""

    def __init__(
        self,
        num_layers: int = 12,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        page_size_tokens: int = 16,
        max_pages: int = 128,
        device: torch.device | str = "cpu",
        codec: KVPageCodec | None = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size_tokens = int(page_size_tokens)
        self.max_pages = int(max_pages)
        self.device = torch.device(device)
        self.codec: KVPageCodec = codec if codec is not None else FP8E4M3PageCodec()
        self._pages_k: Dict[Tuple[int, int], Any] = {}
        self._pages_v: Dict[Tuple[int, int], Any] = {}

    def write_kv_page(self, layer_idx: int, page_idx: int, k_tensor: Tensor, v_tensor: Tensor) -> Dict[str, Any]:
        """Encode and store a ``(K, V)`` page; evicts the oldest page when full (same rule as FP8PagedKVCache)."""
        if len(self._pages_k) >= self.max_pages and (layer_idx, page_idx) not in self._pages_k:
            oldest_key = next(iter(self._pages_k))
            self._pages_k.pop(oldest_key, None)
            self._pages_v.pop(oldest_key, None)
        pk = self.codec.encode(k_tensor.to(self.device))
        pv = self.codec.encode(v_tensor.to(self.device))
        self._pages_k[(layer_idx, page_idx)] = pk
        self._pages_v[(layer_idx, page_idx)] = pv
        fp16_bytes = (k_tensor.numel() + v_tensor.numel()) * 2
        paged_bytes = self._bytes(pk) + self._bytes(pv)
        return {
            "layer_idx": layer_idx,
            "page_idx": page_idx,
            "codec": self.codec.name,
            "fp16_equivalent_bytes": fp16_bytes,
            "fp8_paged_bytes": paged_bytes,  # key kept for FP8PagedKVCache compatibility; bytes of the active codec
            "compression_ratio": round(fp16_bytes / max(1, paged_bytes), 4),
        }

    def read_kv_page(
        self, layer_idx: int, page_idx: int, target_dtype: torch.dtype = torch.float16
    ) -> Tuple[Tensor, Tensor]:
        pk = self._pages_k[(layer_idx, page_idx)]
        pv = self._pages_v[(layer_idx, page_idx)]
        return self.codec.decode(pk, target_dtype), self.codec.decode(pv, target_dtype)

    def total_kv_cache_bytes(self) -> int:
        return sum(self._bytes(p) for p in list(self._pages_k.values()) + list(self._pages_v.values()))

    def _bytes(self, payload: Any) -> int:
        return int(math.ceil(self.codec.payload_bits(payload) / 8.0))
