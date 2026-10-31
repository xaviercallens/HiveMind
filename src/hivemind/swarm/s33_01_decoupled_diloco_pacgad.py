# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S33-01 — Straggler-Free Decoupled DiLoCo with Staleness-Aware PA-CGAD & Radial-Directional Averaging.

Implements asynchronous straggler-free outer optimization inspired by Decoupled DiLoCo (arXiv:2604.21428):
1. Accepts delayed pseudo-gradients ``Delta_i`` from heterogeneous consumer GPUs (RTX 2070, RTX 3060, RTX 4090)
   without blocking on stragglers.
2. Applies staleness-decay weights ``w(tau_i) = (1 + tau_i)^(-alpha)``.
3. Decouples the outer step into:
   - Directional consensus (staleness-weighted unit vector average, normalized on the unit sphere), and
   - Radial magnitude consensus (staleness-weighted L2 norm average),
   preventing destructive cancellation between slightly out-of-phase consumer workers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class WorkerPseudoGradient:
    """Pseudo-gradient contribution from an asynchronous DiLoCo worker."""

    worker_id: str
    staleness_steps: int
    inner_steps: int
    pseudo_gradient: tuple[float, ...]


@dataclass(frozen=True)
class DecoupledDiLoCoStepResult:
    """Aggregated outer step from Straggler-Free Decoupled DiLoCo + PA-CGAD."""

    aggregated_vector: tuple[float, ...]
    radial_norm: float
    naive_mean_norm: float
    norm_preservation_ratio: float
    effective_worker_count: float
    max_staleness_steps: int
    Updated_parameters: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


class StragglerFreeDecoupledDiLoCoAggregator:
    """Staleness-aware PA-CGAD radial-directional outer optimizer."""

    def __init__(
        self,
        *,
        staleness_alpha: float = 0.5,
        outer_lr: float = 0.7,
        max_staleness_steps: int = 16,
    ) -> None:
        if staleness_alpha < 0.0:
            raise ValueError("staleness_alpha must be >= 0")
        if outer_lr <= 0.0:
            raise ValueError("outer_lr must be > 0")
        if max_staleness_steps < 0:
            raise ValueError("max_staleness_steps must be >= 0")
        self.staleness_alpha = float(staleness_alpha)
        self.outer_lr = float(outer_lr)
        self.max_staleness_steps = int(max_staleness_steps)

    def staleness_weight(self, staleness_steps: int) -> float:
        if staleness_steps < 0 or staleness_steps > self.max_staleness_steps:
            return 0.0
        return (1.0 + float(staleness_steps)) ** (-self.staleness_alpha)

    def aggregate_and_step(
        self,
        current_params: Sequence[float],
        contributions: Sequence[WorkerPseudoGradient],
    ) -> DecoupledDiLoCoStepResult:
        """Aggregate delayed worker pseudo-gradients via PA-CGAD radial-directional decoupling."""
        dim = len(current_params)
        if dim == 0:
            raise ValueError("current_params must be non-empty")
        valid = [
            c for c in contributions
            if len(c.pseudo_gradient) == dim and 0 <= c.staleness_steps <= self.max_staleness_steps and _l2_norm(c.pseudo_gradient) > 1e-12
        ]
        if not valid:
            raise ValueError("no valid non-zero worker pseudo-gradients within max_staleness_steps")

        weights = [self.staleness_weight(c.staleness_steps) * max(1, c.inner_steps) for c in valid]
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # 1. Radial magnitude consensus: weighted average of individual L2 norms
        norms = [_l2_norm(c.pseudo_gradient) for c in valid]
        radial_norm = sum(nw * r for nw, r in zip(norm_weights, norms))

        # 2. Directional consensus: weighted sum of unit vectors, re-normalized to unit length
        dir_sum = [0.0] * dim
        naive_mean = [0.0] * dim
        for nw, c, r in zip(norm_weights, valid, norms):
            for j in range(dim):
                dir_sum[j] += nw * (float(c.pseudo_gradient[j]) / r)
                naive_mean[j] += nw * float(c.pseudo_gradient[j])

        dir_norm = max(_l2_norm(dir_sum), 1e-12)
        aggregated = tuple(radial_norm * (d / dir_norm) for d in dir_sum)
        naive_norm = _l2_norm(naive_mean)
        updated = tuple(float(p) - self.outer_lr * g for p, g in zip(current_params, aggregated))

        return DecoupledDiLoCoStepResult(
            aggregated_vector=aggregated,
            radial_norm=radial_norm,
            naive_mean_norm=naive_norm,
            norm_preservation_ratio=radial_norm / max(naive_norm, 1e-12),
            effective_worker_count=float(len(valid)),
            max_staleness_steps=max(c.staleness_steps for c in valid),
            Updated_parameters=updated,
        )
