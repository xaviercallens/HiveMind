# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""GPU lattice codecs for KV-cache tensors (experiment E11, ``HM-S10-05``).

Torch ports of the nearest-point decoders of :mod:`hivemind.compress.lattice_quantizer` (Conway & Sloane 1982, via
the author's OptimalEnergyAI ``oai_ml.core.quantise``) so that whole KV tensors of a 7B model can be quantised on a
16 GB T4 inside the forward pass. The tie rules are the reference ones and the ports are checked bit for bit
against the numpy decoders (``tests/test_kv_lattice.py`` and the E11 controls): round half to even
(``torch.round``), the Dₙ parity fix flips the coordinate with the largest rounding error, lowest index first,
towards the input; between two cosets the first wins on an equal distance. Decoding runs in float64 so that the
rounding decisions are the reference ones.

New here: the hexagonal lattice A₂ (minimum distance 1) as the union of two cosets of the rectangular lattice
ℤ × √3ℤ, applied per RoPE plane. A rotary embedding rotates the coordinate pairs ``(i, i + d/2)`` (the
``rotate_half`` convention of Qwen2 and Gemma), so A₂ on those planes is rotation-covariant up to the lattice's
6-fold symmetry. NSM: ℤ 1/12 = 0.0833, A₂ 5/(36√3) = 0.0802 (0.17 dB), E₈ 929/12960 = 0.0717 (0.65 dB).

Rate is measured, not assumed, with the factorisation of ``factorised_rate_bits_per_dim`` extended by a context
(the channel group a row came from, e.g. the block index inside a head): the coset is coded given the context,
each coordinate given (context, coset), the last E₈ coordinate also given the parity of the others. The same
context is used for every lattice, so ℤ, A₂ and E₈ are compared fairly. The estimate is the plug-in entropy, which
is biased low when the number of distinct codewords approaches the number of rows per context; at the 2-8 bits/dim
of E11 with ~10⁵-10⁶ rows per layer this bias is negligible, but at very fine scales on small samples it saturates.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

SQRT3 = math.sqrt(3.0)
NSM: Dict[str, float] = {"Z": 1.0 / 12.0, "A2": 5.0 / (36.0 * SQRT3), "E8": 929.0 / 12960.0}
BLOCK: Dict[str, int] = {"Z": 8, "A2": 2, "E8": 8}


def nearest_zn_t(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a non-empty finite torch.Tensor")
    return torch.round(x)


def nearest_dn_t(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a non-empty finite torch.Tensor")
    f = torch.round(x)
    odd = torch.remainder(f.sum(dim=-1), 2.0) != 0
    d = x - f
    i = torch.argmax(d.abs(), dim=-1, keepdim=True)
    step = torch.where(torch.gather(d, -1, i) >= 0.0, 1.0, -1.0).to(x.dtype)
    g = f.scatter_add(-1, i, step)
    return torch.where(odd.unsqueeze(-1), g, f)


def _closer_t(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    da = ((x - a) ** 2).sum(dim=-1)
    db = ((x - b) ** 2).sum(dim=-1)
    return torch.where((db < da).unsqueeze(-1), b, a)


def nearest_e8_t(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a non-empty finite torch.Tensor")
    if x.shape[-1] != 8:
        raise ValueError("E8 lives in dimension 8")
    return _closer_t(nearest_dn_t(x), nearest_dn_t(x - 0.5) + 0.5, x)


def nearest_a2_t(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of A₂ = R ∪ (R + (½, √3/2)), R = ℤ × √3ℤ; rows of ``x`` have 2 columns."""
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a non-empty finite torch.Tensor")
    if x.shape[-1] != 2:
        raise ValueError("A2 lives in dimension 2")
    scale = torch.tensor([1.0, SQRT3], dtype=x.dtype, device=x.device)
    shift = torch.tensor([0.5, 0.5 * SQRT3], dtype=x.dtype, device=x.device)
    a = torch.round(x / scale) * scale
    b = torch.round((x - shift) / scale) * scale + shift
    return _closer_t(a, b, x)


DECODERS_T = {"Z": nearest_zn_t, "E8": nearest_e8_t, "A2": nearest_a2_t}


def is_a2_point(c: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """Membership in A₂ (rows): integer coordinates in the basis (1, 0), (½, √3/2)."""
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be a strictly positive finite float")
    if not isinstance(c, torch.Tensor) or c.numel() == 0 or c.shape[-1] != 2 or not bool(torch.isfinite(c).all()):
        raise ValueError("A2 points must have last dimension 2 and be non-empty")
    b = c[..., 1] / (0.5 * SQRT3)
    a = c[..., 0] - 0.5 * b
    return ((a - torch.round(a)).abs() < float(tol)) & ((b - torch.round(b)).abs() < float(tol))


def _entropy_bits(key: torch.Tensor) -> float:
    if not isinstance(key, torch.Tensor) or key.numel() == 0:
        raise ValueError("key must be a non-empty torch.Tensor")
    _, counts = torch.unique(key.reshape(-1), return_counts=True)
    p = counts.to(torch.float64) / counts.sum()
    return float(-(p * torch.log2(p)).sum())


def _joint_key(*cols: torch.Tensor) -> torch.Tensor:
    """Injective int64 key of several integer columns (mixed radix on their observed ranges)."""
    if not cols or any(not isinstance(c, torch.Tensor) or c.numel() == 0 for c in cols):
        raise ValueError("cols must be non-empty tensors")
    key = torch.zeros_like(cols[0], dtype=torch.int64)
    for c in cols:
        c = c.to(torch.int64)
        lo = int(c.min())
        span = int(c.max()) - lo + 1
        key = key * span + (c - lo)
    return key


def _cond_entropy_bits(x: torch.Tensor, *cond: torch.Tensor) -> float:
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        raise ValueError("x must be a non-empty torch.Tensor")
    if not cond:
        return _entropy_bits(x)
    return _entropy_bits(_joint_key(x, *cond)) - _entropy_bits(_joint_key(*cond))


def integer_coordinates(kind: str, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(coset index per row, integer coordinates per row) of unscaled codewords."""
    if not isinstance(c, torch.Tensor) or c.numel() == 0 or not bool(torch.isfinite(c).all()):
        raise ValueError("c must be a non-empty finite torch.Tensor")
    if kind == "Z":
        return torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device), torch.round(c).to(torch.int64)
    if kind == "E8":
        coset = ((c[..., 0] - torch.round(c[..., 0])).abs() > 1e-9).to(torch.int64)
        return coset, torch.round(c - 0.5 * coset.unsqueeze(-1).to(c.dtype)).to(torch.int64)
    if kind == "A2":
        y = c[..., 1] / SQRT3
        coset = ((y - torch.round(y)).abs() > 1e-9).to(torch.int64)
        half = 0.5 * coset.to(c.dtype)
        return coset, torch.stack([torch.round(c[..., 0] - half), torch.round(y - half)], dim=-1).to(torch.int64)
    raise KeyError(kind)


def factorised_rate_t(kind: str, codewords: torch.Tensor, context: torch.Tensor) -> float:
    """Bits per dimension of unscaled codewords (rows, shape (N, n)) given an integer context per row (N,)."""
    if not isinstance(codewords, torch.Tensor) or codewords.ndim != 2 or codewords.shape[0] == 0 or codewords.shape[1] == 0:
        raise ValueError("codewords must be a non-empty 2D tensor")
    if not isinstance(context, torch.Tensor) or context.ndim != 1 or context.shape[0] != codewords.shape[0] or not bool(torch.isfinite(context).all()):
        raise ValueError("context must be a 1D tensor matching codewords.shape[0]")
    n = codewords.shape[-1]
    coset, z = integer_coordinates(kind, codewords)
    ctx = context.to(torch.int64)
    bits = _cond_entropy_bits(coset, ctx) if kind != "Z" else 0.0
    last = n - 1 if kind == "E8" else n
    for i in range(last):
        bits += _cond_entropy_bits(z[:, i], ctx, coset)
    if kind == "E8":
        parity = torch.remainder(z[:, :7].sum(dim=1), 2)
        bits += _cond_entropy_bits(z[:, 7], ctx, coset, parity)
    return bits / n


def to_rows(kind: str, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group the last axis (head_dim d) into lattice rows; return rows (N, n) and the in-head group index (N,).

    ℤ and E₈: consecutive blocks of 8 channels. A₂: RoPE planes (i, i + d/2), i < d/2."""
    if kind not in DECODERS_T:
        raise KeyError(kind)
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be non-empty")
    d = x.shape[-1]
    if kind == "A2":
        if d % 2:
            raise ValueError("A2 needs an even head_dim")
        h = d // 2
        rows = torch.stack([x[..., :h], x[..., h:]], dim=-1)  # (..., d/2, 2)
        groups = h
    else:
        if d % 8:
            raise ValueError("Z/E8 blocks need head_dim % 8 == 0")
        rows = x.reshape(*x.shape[:-1], d // 8, 8)
        groups = d // 8
    ctx = torch.arange(groups, device=x.device).expand(rows.shape[:-1])
    return rows.reshape(-1, rows.shape[-1]), ctx.reshape(-1)


def from_rows(kind: str, rows: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if kind not in DECODERS_T:
        raise KeyError(kind)
    if not isinstance(rows, torch.Tensor) or rows.numel() == 0:
        raise ValueError("rows must be a non-empty torch.Tensor")
    if (
        not isinstance(shape, (tuple, torch.Size))
        or len(shape) < 1
        or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape)
    ):
        raise ValueError("shape must be a non-empty tuple/Size of positive integers")
    if kind == "A2":
        r = rows.reshape(*shape[:-1], shape[-1] // 2, 2)
        return torch.cat([r[..., 0], r[..., 1]], dim=-1)
    return rows.reshape(shape)


def lattice_quantize(kind: str, x: torch.Tensor, scale: float, measure_rate: bool = False) -> Tuple[torch.Tensor, float]:
    """Quantise ``x`` (last axis = head_dim, already normalised) on ``scale · L``; return (x̂ float64, rate or nan)."""
    if kind not in DECODERS_T:
        raise KeyError(kind)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be positive and finite")
    if not isinstance(measure_rate, bool):
        raise ValueError("measure_rate must be a boolean")
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        raise ValueError("x must be a non-empty torch.Tensor")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x contains NaN or inf; refusing to quantise it")
    rows, ctx = to_rows(kind, x.to(torch.float64))
    c = DECODERS_T[kind](rows / float(scale))
    rate = factorised_rate_t(kind, c, ctx) if measure_rate else float("nan")
    return from_rows(kind, c * float(scale), x.shape), rate
