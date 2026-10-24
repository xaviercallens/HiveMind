"""Lattice and sign quantisers for DiLoCo outer deltas (experiment E10).

The nearest-point decoders for ℤⁿ, Dₙ and E₈ = D₈ ∪ (D₈ + ½) follow Conway & Sloane, "Fast quantizing and
decoding algorithms for lattice quantizers and codes", IEEE Trans. Inf. Theory 28(2):227–232 (1982), and are
ported from the reference numpy implementation ``oai_ml.core.quantise`` of the author's OptimalEnergyAI project
(X. Callens, 2026; companion of the textbook doi:10.5281/zenodo.23295142), whose decoders were checked against
exhaustive search with 0 mismatches in 3·10⁶ queries per lattice. The tie rules are copied unchanged: coordinates
round half to even (``np.rint``); the Dₙ parity fix flips the coordinate with the largest rounding error, lowest
index first, towards the input (+1 when the input is an integer); between the two E₈ cosets the first wins on an
equal distance.

Rate is measured, not assumed: ``factorised_rate_bits_per_dim`` is the achievable rate of an entropy coder that
codes the coset, then each coordinate given the coset, then the last coordinate given the parity of the others
(the Dₙ / E₈ constraint), as in ``oai_ml.core.rate``. It ignores the same inter-coordinate correlations for every
lattice, so ℤ⁸ and E₈ are compared fairly. The 1-bit sign quantiser is the signSGD compressor of Bernstein et al.
(arXiv:1802.04434) with the mean absolute value as scale; its error-feedback variant carries the residual to the
next round (EF21 telescoping identity, HiveMind Lean Theorem 107).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

LATTICE_QUANTIZER_REFERENCES: Dict[str, str] = {
    "conway_sloane_1982": "Conway & Sloane, IEEE Trans. IT 28(2):227-232, 1982 (fast nearest-point decoders)",
    "optimal_energy_ai": "X. Callens, OptimalEnergyAI, oai_ml.core.quantise / oai_ml.core.rate (reference decoders; 0 mismatches in 3e6 brute-force queries), 2026",
    "triangular_lattice_textbook": "doi:10.5281/zenodo.23295142 (universal optimality textbook the decoders accompany)",
    "signsgd": "Bernstein, Wang, Azizzadenesheli & Anandkumar, signSGD, arXiv:1802.04434, ICML 2018",
}

NSM_REFERENCE: Dict[str, float] = {
    "Z": 1.0 / 12.0,
    "A2": 5.0 / (36.0 * math.sqrt(3.0)),
    "E8": 929.0 / 12960.0,
}

_SQRT3 = math.sqrt(3.0)


def nearest_zn(x: np.ndarray) -> np.ndarray:
    """Nearest point of ℤⁿ (round half to even)."""
    return np.rint(np.asarray(x, dtype=float))


def nearest_a2(x: np.ndarray) -> np.ndarray:
    """Nearest point of the 2D hexagonal lattice A₂ = (ℤ × √3·ℤ) ∪ ((ℤ + ½) × √3·(ℤ + ½)); rows of ``x`` must have 2 columns."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 2:
        raise ValueError("A2 lives in dimension 2")
    c0 = np.column_stack([np.rint(x[:, 0]), _SQRT3 * np.rint(x[:, 1] / _SQRT3)])
    c1 = np.column_stack(
        [
            np.rint(x[:, 0] - 0.5) + 0.5,
            _SQRT3 * (np.rint(x[:, 1] / _SQRT3 - 0.5) + 0.5),
        ]
    )
    return _closer(c0, c1, x)


def nearest_dn(x: np.ndarray) -> np.ndarray:
    """Nearest point of Dₙ (integer vectors with even coordinate sum); rows of ``x``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    f = np.rint(x)
    odd = np.mod(f.sum(axis=-1), 2.0) != 0
    if not odd.any():
        return f
    d = x - f
    i = np.argmax(np.abs(d), axis=-1)
    rows = np.arange(len(x))
    step = np.where(d[rows, i] >= 0.0, 1.0, -1.0)
    g = f.copy()
    g[rows, i] += step
    return np.where(odd[:, None], g, f)


def _closer(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    da = np.einsum("ij,ij->i", x - a, x - a)
    db = np.einsum("ij,ij->i", x - b, x - b)
    return np.where((db < da)[:, None], b, a)


def nearest_e8(x: np.ndarray) -> np.ndarray:
    """Nearest point of E₈ = D₈ ∪ (D₈ + ½); rows of ``x`` must have 8 columns."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 8:
        raise ValueError("E8 lives in dimension 8")
    return _closer(nearest_dn(x), nearest_dn(x - 0.5) + 0.5, x)


DECODERS = {"Z": nearest_zn, "A2": nearest_a2, "E8": nearest_e8}
LATTICE_DIM = {"Z": 8, "A2": 2, "E8": 8}


def is_lattice_point(kind: str, c: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Membership test (per row): ℤⁿ integer; A₂ rectangular cosets; E₈ integer or half-integer with even sum."""
    c = np.atleast_2d(np.asarray(c, dtype=float))
    if kind == "Z":
        return np.all(np.abs(c - np.rint(c)) < tol, axis=1)
    if kind == "A2":
        if c.shape[1] != 2:
            raise ValueError("A2 lives in dimension 2")
        y0 = c[:, 0]
        y1 = c[:, 1] / _SQRT3
        coset0 = (np.abs(y0 - np.rint(y0)) < tol) & (np.abs(y1 - np.rint(y1)) < tol)
        coset1 = (np.abs(y0 - 0.5 - np.rint(y0 - 0.5)) < tol) & (
            np.abs(y1 - 0.5 - np.rint(y1 - 0.5)) < tol
        )
        return coset0 | coset1
    if kind == "E8":
        integral = np.all(np.abs(c - np.rint(c)) < tol, axis=1)
        half = np.all(np.abs(c - 0.5 - np.rint(c - 0.5)) < tol, axis=1)
        even_sum = np.abs(np.mod(np.rint(c.sum(axis=1)), 2.0)) < tol
        return (integral | half) & even_sum
    raise KeyError(kind)


def _entropy_bits(cols: List[np.ndarray]) -> float:
    """Plug-in joint entropy (bits) of integer columns."""
    keys = np.stack([np.asarray(c).astype(np.int64) for c in cols], axis=1)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _cond_entropy_bits(x: np.ndarray, *cond: np.ndarray) -> float:
    if not cond:
        return _entropy_bits([x])
    return _entropy_bits([x, *cond]) - _entropy_bits(list(cond))


def factorised_rate_bits_per_dim(kind: str, codewords: np.ndarray) -> float:
    """Achievable bits per dimension of unscaled codewords (rows): ℤⁿ Σ H(xᵢ); A₂ H(coset)+H(z₀|coset)+H(z₁|coset); E₈ H(coset)+Σ H(zᵢ|coset)+H(z₈|coset,parity)."""
    c = np.atleast_2d(np.asarray(codewords, dtype=float))
    n = c.shape[1]
    if kind == "Z":
        return sum(_entropy_bits([np.rint(c[:, i])]) for i in range(n)) / n
    if kind == "A2":
        coset = (np.abs(c[:, 0] - np.rint(c[:, 0])) > 1e-9).astype(np.int64)
        z0 = np.rint(c[:, 0] - 0.5 * coset).astype(np.int64)
        z1 = np.rint(c[:, 1] / _SQRT3 - 0.5 * coset).astype(np.int64)
        bits = (
            _entropy_bits([coset])
            + _cond_entropy_bits(z0, coset)
            + _cond_entropy_bits(z1, coset)
        )
        return bits / 2.0
    if kind == "E8":
        coset = (np.abs(c[:, 0] - np.rint(c[:, 0])) > 1e-9).astype(np.int64)  # 1 = half-integer coset
        z = np.rint(c - 0.5 * coset[:, None]).astype(np.int64)
        bits = _entropy_bits([coset])
        for i in range(7):
            bits += _cond_entropy_bits(z[:, i], coset)
        parity = np.mod(z[:, :7].sum(axis=1), 2)
        bits += _cond_entropy_bits(z[:, 7], coset, parity)
        return bits / 8.0
    raise KeyError(kind)


@dataclass(frozen=True)
class QuantizedDelta:
    dequantized: torch.Tensor
    rate_bits_per_dim: float
    relative_mse: float
    residual: torch.Tensor  # v - Q(v) (carried when error feedback is on)


class LatticeDeltaQuantizer:
    """Quantise a flat delta blockwise on ``scale · L`` (L ∈ {ℤ⁸, A₂, E₈}); optional EF21 error feedback."""

    def __init__(self, kind: str, scale: float, block: Optional[int] = None, error_feedback: bool = False) -> None:
        if kind not in DECODERS:
            raise KeyError(kind)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("scale must be positive and finite")
        eff_block = int(block) if block is not None else LATTICE_DIM[kind]
        self.kind, self.scale, self.block, self.error_feedback = kind, float(scale), eff_block, bool(error_feedback)
        self._carry: Optional[torch.Tensor] = None

    def quantize(self, vec: torch.Tensor) -> QuantizedDelta:
        v = vec.detach().to(torch.float32).reshape(-1)
        if self.error_feedback and self._carry is not None:
            v = v + self._carry
        n = v.numel()
        pad = (-n) % self.block
        x = torch.cat([v, torch.zeros(pad)]) if pad else v
        rows = x.numpy().astype(float).reshape(-1, self.block) / self.scale
        c = DECODERS[self.kind](rows)
        deq = torch.from_numpy((c * self.scale).reshape(-1)[:n].astype(np.float32))
        residual = v - deq
        if self.error_feedback:
            self._carry = residual.clone()
        vn = float(torch.dot(v, v))
        rel = float(torch.dot(residual, residual)) / vn if vn > 0.0 else 0.0
        return QuantizedDelta(deq, factorised_rate_bits_per_dim(self.kind, c), rel, residual)


class SignDeltaQuantizer:
    """1-bit sign compressor: Q(v) = mean|v| · sign(v) (signSGD scale rule); optional error feedback."""

    def __init__(self, error_feedback: bool = False) -> None:
        self.error_feedback = bool(error_feedback)
        self._carry: Optional[torch.Tensor] = None

    def quantize(self, vec: torch.Tensor) -> QuantizedDelta:
        v = vec.detach().to(torch.float32).reshape(-1)
        if self.error_feedback and self._carry is not None:
            v = v + self._carry
        deq = v.abs().mean() * torch.sign(v)
        residual = v - deq
        if self.error_feedback:
            self._carry = residual.clone()
        vn = float(torch.dot(v, v))
        rel = float(torch.dot(residual, residual)) / vn if vn > 0.0 else 0.0
        return QuantizedDelta(deq, 1.0, rel, residual)


def calibrate_scale_for_rate(kind: str, sample: torch.Tensor, target_bits_per_dim: float, block: Optional[int] = None, tol: float = 0.02, iters: int = 60) -> Tuple[float, float]:
    """Bisection on log-scale so that the measured factorised rate of ``sample`` hits ``target`` (rate falls as scale grows)."""
    eff_block = int(block) if block is not None else LATTICE_DIM.get(kind, 8)
    v = sample.detach().to(torch.float32).reshape(-1)
    n = v.numel()
    pad = (-n) % eff_block
    x = (torch.cat([v, torch.zeros(pad)]) if pad else v).numpy().astype(float).reshape(-1, eff_block)
    sd = float(np.std(x)) or 1.0

    def rate_at(s: float) -> float:
        return factorised_rate_bits_per_dim(kind, DECODERS[kind](x / s))

    lo, hi = math.log(sd * 1e-3), math.log(sd * 1e3)
    r_lo, r_hi = rate_at(math.exp(lo)), rate_at(math.exp(hi))
    if not (r_hi <= target_bits_per_dim <= r_lo):
        raise ValueError(f"target {target_bits_per_dim} bits/dim outside the reachable range [{r_hi:.3f}, {r_lo:.3f}]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rate_at(math.exp(mid))
        if abs(r - target_bits_per_dim) <= tol:
            return math.exp(mid), r
        if r > target_bits_per_dim:
            lo = mid
        else:
            hi = mid
    s = math.exp(0.5 * (lo + hi))
    return s, rate_at(s)


def _lattice_points_in_ball(kind: str, radius: float) -> np.ndarray:
    """All points of ℤ⁸ / A₂ / E₈ with norm ≤ radius (exhaustive; for the brute-force check)."""
    r = int(math.ceil(radius))
    grid = np.arange(-r, r + 1, dtype=float)
    if kind == "A2":
        mesh = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
        c0 = np.column_stack([mesh[:, 0], _SQRT3 * mesh[:, 1]])
        c1 = np.column_stack([mesh[:, 0] + 0.5, _SQRT3 * (mesh[:, 1] + 0.5)])
        both = np.concatenate([c0, c1], axis=0)
        return both[np.einsum("ij,ij->i", both, both) <= radius * radius + 1e-9]
    mesh = np.stack(np.meshgrid(*([grid] * 8), indexing="ij"), axis=-1).reshape(-1, 8)
    pts = mesh[np.einsum("ij,ij->i", mesh, mesh) <= radius * radius + 1e-9]
    if kind == "Z":
        return pts
    even = pts[np.mod(pts.sum(axis=1), 2.0) == 0]
    half = mesh + 0.5
    half = half[(np.einsum("ij,ij->i", half, half) <= radius * radius + 1e-9) & (np.mod(half.sum(axis=1), 2.0) == 0)]
    return np.concatenate([even, half], axis=0)


def brute_force_decoder_check(kind: str, n_queries: int, seed: int, radius: float = 3.0) -> Dict[str, object]:
    """Two-sided check (lesson recorded in OptimalEnergyAI h0.md): the decoder output must be a lattice point AND
    its distance must equal the exhaustive minimum over all lattice points in a ball that contains the answer.

    Queries are uniform in the cube [-1/2, 1/2]^d; the covering radius of all supported lattices is
    <= sqrt 2, so every nearest point has norm <= 2 sqrt 2 < 3 and the ball of radius 3 contains the answer."""
    rng = np.random.default_rng(seed)
    dim = LATTICE_DIM.get(kind, 8)
    q = rng.uniform(-0.5, 0.5, size=(n_queries, dim))
    pts = _lattice_points_in_ball(kind, radius)
    c = DECODERS[kind](q)
    member = is_lattice_point(kind, c)
    d_dec = np.einsum("ij,ij->i", q - c, q - c)
    p2 = np.einsum("ij,ij->i", pts, pts)
    d_min = np.empty(n_queries)
    for s in range(0, n_queries, 2048):
        qs = q[s : s + 2048]
        d_min[s : s + 2048] = (p2[None, :] - 2.0 * qs @ pts.T).min(axis=1) + np.einsum("ij,ij->i", qs, qs)
    mismatches = int(np.sum((np.abs(d_dec - d_min) > 1e-9) | ~member))
    return {"kind": kind, "n_queries": int(n_queries), "lattice_points_in_ball": int(len(pts)), "non_members": int(np.sum(~member)), "mismatches": mismatches, "passed": mismatches == 0}
