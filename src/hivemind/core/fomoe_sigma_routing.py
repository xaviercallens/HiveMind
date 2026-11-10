# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""FoMoE sigma-MoE routing with Ghost-Expert Skip-Token accounting (`arXiv:2606.19025`).

FoMoE (Sani, Cao, Kurmanji, Iacob, Jovanovic, Gao, Zhao & Lane, CaMLSys / Flower Labs,
2026) trains decoder-only Transformers whose FFNs are `sigma-MoE` modules: experts are
selected by *independent sigmoid* gates (Csordas et al., 2023) instead of a competitive
softmax, and only a local subset of experts is resident on each worker. When the router is
left unpartitioned, tokens may be assigned to non-resident **ghost experts**; the worker then
**skips** that expert path (Skip-Token) instead of issuing a remote FFN call.

FoMoE's cost model (their Eq. 2-3) uses the *ideal* scaling factor `sigma_st = N_le / N_e`,
and the paper explicitly recommends replacing it with the *empirical* ghost-assignment
frequency when routing is imbalanced. The existing `FoMoEPartialReplicaPlanner` in
`hivemind.swarm.diloco_nextgen` only models the ideal factor. This module adds:

* :func:`sigmoid_independent_topk_routing` — sigma-MoE top-k over sigmoid scores with a
  resident-expert mask that *skips* ghost assignments after routing (router stays global),
  returning the realised ghost-assignment fraction per batch.
* :func:`skip_token_flops_per_token` — FoMoE Eq. 3 with either the ideal `N_le / N_e`
  factor or the measured ghost fraction, so the cost estimate can be gated on a measurement.

Nothing here has been evaluated on a trained model; the functions are exact bookkeeping
over whatever logits they are given.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class SigmaMoERoutingResult:
    """Per-batch sigma-MoE routing decision with ghost-expert skip accounting."""

    expert_indices: torch.Tensor  # [tokens, k] long
    gate_values: torch.Tensor  # [tokens, k] float32, zero where the slot was skipped
    skipped_slots: torch.Tensor  # [tokens, k] bool, True where the chosen expert is a ghost
    total_experts: int
    resident_experts: int
    top_k: int
    ghost_assignment_fraction: float
    ideal_skip_token_scaling: float
    empirical_skip_token_scaling: float

    def __post_init__(self) -> None:
        if not isinstance(self.expert_indices, torch.Tensor) or self.expert_indices.ndim != 2 or self.expert_indices.numel() == 0:
            raise ValueError("expert_indices must be a non-empty 2D torch.Tensor")
        if not isinstance(self.gate_values, torch.Tensor) or self.gate_values.shape != self.expert_indices.shape:
            raise ValueError("gate_values must match expert_indices.shape")
        if not isinstance(self.skipped_slots, torch.Tensor) or self.skipped_slots.shape != self.expert_indices.shape:
            raise ValueError("skipped_slots must match expert_indices.shape")
        if isinstance(self.total_experts, bool) or not isinstance(self.total_experts, int) or self.total_experts < 1:
            raise ValueError("total_experts must be an integer >= 1")
        if (
            isinstance(self.resident_experts, bool)
            or not isinstance(self.resident_experts, int)
            or not (1 <= self.resident_experts <= self.total_experts)
        ):
            raise ValueError("resident_experts must satisfy 1 <= resident_experts <= total_experts")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not (1 <= self.top_k <= self.total_experts)
        ):
            raise ValueError("top_k must satisfy 1 <= top_k <= total_experts")
        for field_name in ("ghost_assignment_fraction", "ideal_skip_token_scaling", "empirical_skip_token_scaling"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{field_name} must be a finite number in [0.0, 1.0]")


def sigmoid_independent_topk_routing(
    router_logits: torch.Tensor,
    top_k: int,
    resident_mask: Optional[torch.Tensor] = None,
) -> SigmaMoERoutingResult:
    """sigma-MoE routing: top-`k` over *independent sigmoid* scores, then skip ghost experts.

    The router is evaluated over all `E` experts (FoMoE's unpartitioned-router variant), so
    the selection does not depend on which experts happen to be resident. Slots whose chosen
    expert is not resident are marked skipped and their gate is zeroed; the realised
    ghost-assignment fraction is `skipped_slots / (tokens * k)`.
    """
    if not isinstance(router_logits, torch.Tensor) or router_logits.dim() != 2:
        raise ValueError("router_logits must be [tokens, experts]")
    tokens, num_experts = router_logits.shape
    if tokens < 1 or num_experts < 1:
        raise ValueError("router_logits must be non-empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= num_experts):
        raise ValueError("top_k must satisfy 1 <= k <= experts")
    if not bool(torch.isfinite(router_logits).all().item()):
        raise ValueError("router_logits must be finite")
    if resident_mask is None:
        resident = torch.ones(num_experts, dtype=torch.bool, device=router_logits.device)
    else:
        if (
            not isinstance(resident_mask, torch.Tensor)
            or resident_mask.shape != (num_experts,)
            or resident_mask.dtype != torch.bool
        ):
            raise ValueError("resident_mask must be a bool tensor of shape [experts]")
        resident = resident_mask.to(router_logits.device)
    resident_count = int(resident.sum().item())
    if resident_count < 1:
        raise ValueError("at least one expert must be resident")

    scores = torch.sigmoid(router_logits.detach().to(torch.float32))
    top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
    skipped = ~resident[top_idx]
    gates = torch.where(skipped, torch.zeros_like(top_scores), top_scores)

    ghost_fraction = float(skipped.sum().item()) / float(tokens * top_k)
    ideal = resident_count / float(num_experts)
    return SigmaMoERoutingResult(
        expert_indices=top_idx,
        gate_values=gates,
        skipped_slots=skipped,
        total_experts=num_experts,
        resident_experts=resident_count,
        top_k=top_k,
        ghost_assignment_fraction=ghost_fraction,
        ideal_skip_token_scaling=ideal,
        empirical_skip_token_scaling=1.0 - ghost_fraction,
    )


def skip_token_flops_per_token(
    dense_flops: float,
    ffn_flops_per_layer: Sequence[float],
    gate_flops_per_layer: Sequence[float],
    top_k: int,
    local_experts_per_layer: Sequence[int],
    total_experts_per_layer: Sequence[int],
    empirical_ghost_fraction_per_layer: Optional[Sequence[float]] = None,
    router_partitioned: bool = False,
) -> Dict[str, object]:
    """FoMoE per-token FLOPs (Eq. 3):

    `F_token ~ F_dense + sum_l [ sigma_st,l * min(k, N_le,l) / k * F_FFN,l + g_l * F_gate,l ]`

    where `sigma_st,l = N_le,l / N_e,l` (ideal, balanced routing) or `1 - ghost_fraction_l`
    when a measured ghost-assignment frequency is supplied, and the gating factor `g_l` is
    `N_le,l / N_e,l` for a partitioned router or `1` for FoMoE's unpartitioned ghost-expert
    router. Returns both the full-replica and partial-replica estimates.
    """
    if not isinstance(router_partitioned, bool):
        raise ValueError("router_partitioned must be a bool")
    if not isinstance(ffn_flops_per_layer, Sequence) or isinstance(ffn_flops_per_layer, (str, bytes)):
        raise ValueError("ffn_flops_per_layer must be a sequence")
    layers = len(ffn_flops_per_layer)
    if layers == 0:
        raise ValueError("at least one layer is required")
    for name, seq in (
        ("gate_flops_per_layer", gate_flops_per_layer),
        ("local_experts_per_layer", local_experts_per_layer),
        ("total_experts_per_layer", total_experts_per_layer),
    ):
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)) or len(seq) != layers:
            raise ValueError(f"{name} must have one entry per layer")
    if empirical_ghost_fraction_per_layer is not None:
        if (
            not isinstance(empirical_ghost_fraction_per_layer, Sequence)
            or isinstance(empirical_ghost_fraction_per_layer, (str, bytes))
            or len(empirical_ghost_fraction_per_layer) != layers
        ):
            raise ValueError("empirical_ghost_fraction_per_layer must have one entry per layer")
    if (
        isinstance(dense_flops, bool)
        or not isinstance(dense_flops, (int, float))
        or not math.isfinite(float(dense_flops))
        or dense_flops < 0.0
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError("dense_flops must be >= 0 and top_k >= 1")

    partial_total = float(dense_flops)
    full_total = float(dense_flops)
    per_layer: List[Dict[str, float]] = []
    for layer in range(layers):
        raw_le = local_experts_per_layer[layer]
        raw_e = total_experts_per_layer[layer]
        if isinstance(raw_le, bool) or isinstance(raw_e, bool) or not isinstance(raw_le, int) or not isinstance(raw_e, int):
            raise ValueError("local experts must satisfy 1 <= N_le <= N_e")
        n_le = int(raw_le)
        n_e = int(raw_e)
        if n_e < 1 or not (1 <= n_le <= n_e):
            raise ValueError("local experts must satisfy 1 <= N_le <= N_e")
        f_ffn = float(ffn_flops_per_layer[layer])
        f_gate = float(gate_flops_per_layer[layer])
        if not math.isfinite(f_ffn) or not math.isfinite(f_gate) or f_ffn < 0.0 or f_gate < 0.0:
            raise ValueError("FLOP counts must be >= 0")
        ideal_sigma = n_le / n_e
        if empirical_ghost_fraction_per_layer is None:
            sigma_st = ideal_sigma
        else:
            ghost = float(empirical_ghost_fraction_per_layer[layer])
            if not (0.0 <= ghost <= 1.0):
                raise ValueError("ghost fractions must be in [0, 1]")
            sigma_st = 1.0 - ghost
        active_fraction = min(top_k, n_le) / float(top_k)
        gate_factor = ideal_sigma if router_partitioned else 1.0
        expert_flops = sigma_st * active_fraction * f_ffn
        gate_flops = gate_factor * f_gate
        partial_total += expert_flops + gate_flops
        full_total += f_ffn + f_gate
        per_layer.append(
            {
                "layer": float(layer),
                "ideal_skip_token_scaling": ideal_sigma,
                "applied_skip_token_scaling": sigma_st,
                "active_expert_fraction": active_fraction,
                "gate_factor": gate_factor,
                "expert_flops": expert_flops,
                "gate_flops": gate_flops,
            }
        )
    return {
        "full_replica_flops_per_token": full_total,
        "partial_replica_flops_per_token": partial_total,
        "flop_reduction_x": (full_total / partial_total) if partial_total > 0.0 else float("inf"),
        "scaling_source": "empirical_ghost_fraction" if empirical_ghost_fraction_per_layer is not None else "ideal_N_le_over_N_e",
        "per_layer": per_layer,
    }
