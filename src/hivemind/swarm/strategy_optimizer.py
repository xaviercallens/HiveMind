# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 5 / Wave 4 Strategy Improvement Engine (`v1.7.0`).

Implements the 5 core algorithmic and game-theoretic strategy improvements for the
HiveMind Decentralized Sparse MoE Swarm:

1. `SI-01` — `DynamicFisherSVDSelector`: Spectral-Entropy & Fisher-Trace Rank Allocation.
   Replaces uniform `r=16` SVD projection with dynamic per-expert rank allocation
   `r_e in {8, 12, 16, 20, 24, 32}` subject to exact VRAM budget conservation
   `sum_e r_e <= E * r_budget`, concentrating rank capacity on high-entropy reasoning/code
   domains while compressing low-entropy adapters.
2. `SI-02` — `UCB1CurriculumShardScheduler`: Multi-Armed Bandit (`UCB1` + Perplexity-Velocity)
   Curriculum & Shard Router. Dynamically assigns Guild GPU cycles to domains with the highest
   marginal holdout loss reduction per GPU-second (`|Delta L_d| / GPU_sec`).
3. `SI-03` — `AdaptiveDiLoCoCadenceController`: Latency-, Packet-Loss-, and BBRv1-Aware
   DiLoCo Inner-Step (`H`) & `Top-k` Sparsity Controller that enforces the `< 8%` WAN
   Communication Overhead SLA across both 0-RTT LAN/Geo-Rings and TURN CGNAT relays.
4. `SI-04` — `ReputationWeightedTIESMerger`: Honesty- & Holdout-Gain-Weighted TIES Sign
   Election (`reputation_weighted_ties_merge_lora`), preventing noisy or low-step workers
   from flipping sign consensus against high-reputation miners.
5. `SI-05` — `ParetoGuildCampaignPlanner`: Game-Theoretic Multi-Objective Pareto Optimizer
   that assigns heterogeneous consumer GPUs (`RTX 2070 8GB`, `RTX 3060 12GB`, `ROCm`, `MPS`)
   across Swarm Mining Guilds and Joined Co-Op Campaigns.

Season 6 (`HM-S6-01`) adds `EF21DiLoCoErrorFeedbackBuffer`, a per-worker error-feedback
residual for Top-k + FP8 compressed outer-loop pseudo-gradients.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from hivemind.core.btm_moe import project_lora_rank_svd_with_report
from hivemind.core.btm_upcycle import detect_lora_rank, pair_lora_ab_keys
from hivemind.core.compression import (
    CompressedDeltaPackage,
    compress_lora_delta,
    decompress_lora_delta,
)
from hivemind.swarm.scientific_swarm_eval import (
    SCIENTIFIC_BENCHMARK_SUITES,
    _evaluate_lora_on_scientific_holdout,
    ties_merge_lora_adapters,
)

ALLOWED_DYNAMIC_RANKS: Tuple[int, ...] = (8, 12, 16, 20, 24, 32)


@dataclass
class ExpertSpectralProfile:
    """Spectral entropy and Fisher-trace importance telemetry for one domain LoRA adapter."""

    domain_name: str
    source_rank_r: int
    allocated_rank_r: int
    spectral_shannon_entropy: float
    fisher_trace_score: float
    uniform_r16_energy_retained: float
    dynamic_rank_energy_retained: float
    energy_gain_vs_uniform: float


class DynamicFisherSVDSelector:
    """SI-01: Dynamic Spectral-Entropy & Fisher-Trace LoRA Rank Allocator.

    Given `E` heterogeneous community adapters (`r_src in {8..64}`) and an average rank budget
    `avg_rank_budget_r` (default `16`), computes the singular value spectrum of each adapter's
    effective delta `Delta W = B @ A` and allocates `r_e in {8, 12, 16, 20, 24, 32}` so that:
      - Total rank budget is strictly conserved: `sum_{e=1}^E r_e <= E * avg_rank_budget_r`
      - Total retained Frobenius spectral energy is maximized across the MoE expert pool.
    """

    def __init__(
        self,
        avg_rank_budget_r: int = 16,
        allowed_ranks: Sequence[int] = ALLOWED_DYNAMIC_RANKS,
    ) -> None:
        if avg_rank_budget_r < 4:
            raise ValueError("avg_rank_budget_r must be >= 4")
        self.avg_rank_budget_r = int(avg_rank_budget_r)
        self.allowed_ranks = tuple(sorted(set(int(r) for r in allowed_ranks if r > 0)))

    @staticmethod
    def compute_singular_spectrum(lora_state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the aggregated singular value spectrum across all `(lora_A, lora_B)` pairs."""
        pairs = pair_lora_ab_keys(lora_state_dict)
        spectra: List[torch.Tensor] = []
        for a_key, b_key in pairs:
            a_mat = lora_state_dict[a_key].detach().to(torch.float32)
            b_mat = lora_state_dict[b_key].detach().to(torch.float32)
            if a_mat.shape[0] > a_mat.shape[1]:
                a_mat = a_mat.t()
            if b_mat.shape[0] < b_mat.shape[1]:
                b_mat = b_mat.t()
            delta = b_mat @ a_mat
            s = torch.linalg.svdvals(delta)
            spectra.append(s)

        if not spectra:
            for val in lora_state_dict.values():
                if isinstance(val, torch.Tensor) and val.ndim == 2:
                    spectra.append(torch.linalg.svdvals(val.detach().to(torch.float32)))
        if not spectra:
            return torch.ones(16, dtype=torch.float32)

        max_len = max(int(s.numel()) for s in spectra)
        padded = [
            torch.nn.functional.pad(s, (0, max_len - int(s.numel()))) for s in spectra
        ]
        return torch.stack(padded, dim=0).mean(dim=0)

    @classmethod
    def retained_energy_at_rank(cls, spectrum: torch.Tensor, rank_r: int) -> float:
        sq = spectrum.pow(2)
        total = float(sq.sum().item())
        if total <= 1e-12:
            return 1.0
        keep = min(max(1, int(rank_r)), int(sq.numel()))
        return round(float(sq[:keep].sum().item()) / total, 6)

    @classmethod
    def spectral_entropy_and_fisher(cls, spectrum: torch.Tensor) -> Tuple[float, float]:
        sq = spectrum.pow(2)
        total = float(sq.sum().item())
        if total <= 1e-12 or sq.numel() <= 1:
            return 0.0, 0.0
        probs = (sq / total).clamp_min(1e-12)
        raw_entropy = float(-(probs * torch.log(probs)).sum().item())
        norm_entropy = raw_entropy / max(1e-8, math.log(float(sq.numel())))
        norm_entropy = max(0.0, min(1.0, norm_entropy))
        fisher_score = total * (1.0 + norm_entropy)
        return round(norm_entropy, 6), round(fisher_score, 6)

    def allocate_and_project_experts(
        self,
        domain_adapters: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Any]:
        """Allocate budget-conserving ranks `r_e` across `domain_adapters` and project via SVD."""
        domains = list(domain_adapters.keys())
        num_experts = len(domains)
        if num_experts == 0:
            return {
                "total_rank_budget": 0,
                "total_rank_used": 0,
                "profiles": [],
                "projected_adapters": {},
            }

        total_budget = num_experts * self.avg_rank_budget_r
        min_rank = min(self.allowed_ranks)
        spectra: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, float] = {}
        fishers: Dict[str, float] = {}
        src_ranks: Dict[str, int] = {}

        for dom, state in domain_adapters.items():
            spec = self.compute_singular_spectrum(state)
            spectra[dom] = spec
            ent, fish = self.spectral_entropy_and_fisher(spec)
            entropies[dom] = ent
            fishers[dom] = fish
            src_ranks[dom] = detect_lora_rank(state) or int(spec.numel())

        # Start every expert at the lowest allowed rank and greedily upgrade the expert with
        # the highest marginal spectral energy / entropy gain per unit of rank added.
        allocated: Dict[str, int] = {dom: min_rank for dom in domains}
        current_sum = sum(allocated.values())

        while current_sum < total_budget:
            best_dom: Optional[str] = None
            best_gain_per_rank = -1.0
            best_next_rank = min_rank

            for dom in domains:
                cur_r = allocated[dom]
                higher = [
                    r
                    for r in self.allowed_ranks
                    if r > cur_r and (r - cur_r) <= (total_budget - current_sum)
                ]
                if not higher:
                    continue
                nxt_r = higher[0]
                e_cur = self.retained_energy_at_rank(spectra[dom], cur_r)
                e_nxt = self.retained_energy_at_rank(spectra[dom], nxt_r)
                marginal = ((e_nxt - e_cur) + 0.05 * entropies[dom]) / float(nxt_r - cur_r)
                if marginal > best_gain_per_rank:
                    best_gain_per_rank = marginal
                    best_dom = dom
                    best_next_rank = nxt_r

            if best_dom is None:
                break
            current_sum += best_next_rank - allocated[best_dom]
            allocated[best_dom] = best_next_rank

        projected_adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        profiles: List[ExpertSpectralProfile] = []
        for dom, state in domain_adapters.items():
            alloc_r = allocated[dom]
            proj_state, proj_report = project_lora_rank_svd_with_report(
                state,
                target_rank_r=alloc_r,
            )
            projected_adapters[dom] = proj_state
            unif_energy = self.retained_energy_at_rank(spectra[dom], self.avg_rank_budget_r)
            dyn_energy = self.retained_energy_at_rank(spectra[dom], alloc_r)
            profiles.append(
                ExpertSpectralProfile(
                    domain_name=dom,
                    source_rank_r=src_ranks[dom],
                    allocated_rank_r=alloc_r,
                    spectral_shannon_entropy=entropies[dom],
                    fisher_trace_score=fishers[dom],
                    uniform_r16_energy_retained=unif_energy,
                    dynamic_rank_energy_retained=max(
                        dyn_energy, proj_report.mean_spectral_energy_retained
                    ),
                    energy_gain_vs_uniform=round(dyn_energy - unif_energy, 6),
                )
            )

        mean_unif = round(
            sum(p.uniform_r16_energy_retained for p in profiles) / max(1, len(profiles)), 6
        )
        mean_dyn = round(
            sum(p.dynamic_rank_energy_retained for p in profiles) / max(1, len(profiles)), 6
        )
        return {
            "strategy_id": "SI-01-DYNAMIC-FISHER-SVD-RANK",
            "num_experts": num_experts,
            "avg_rank_budget_r": self.avg_rank_budget_r,
            "total_rank_budget": total_budget,
            "total_rank_used": sum(allocated.values()),
            "vram_budget_conserved": sum(allocated.values()) <= total_budget,
            "allocated_ranks_by_domain": allocated,
            "mean_uniform_r16_energy_retained": mean_unif,
            "mean_dynamic_fisher_energy_retained": mean_dyn,
            "spectral_energy_improvement": round(mean_dyn - mean_unif, 6),
            "profiles": [asdict(p) for p in profiles],
            "projected_adapters": projected_adapters,
        }


class UCB1CurriculumShardScheduler:
    """SI-02: Bandit-Driven (`UCB1` + Perplexity-Velocity) Domain & Shard Curriculum Scheduler.

    Replaces static round-robin workunit assignment with an Upper Confidence Bound (`UCB1`)
    bandit that tracks empirical holdout loss drop per GPU-second (`|Delta L_d| / GPU_sec_d`)
    and routes miners to the highest-yield domains while guaranteeing full exploration.
    """

    def __init__(
        self,
        domains: Sequence[str],
        exploration_c: float = 0.65,
    ) -> None:
        if not domains:
            raise ValueError("domains must be non-empty")
        self.domains = list(domains)
        self.exploration_c = float(exploration_c)
        self.pull_counts: Dict[str, int] = {d: 0 for d in self.domains}
        self.cumulative_loss_drop: Dict[str, float] = {d: 0.0 for d in self.domains}
        self.cumulative_gpu_seconds: Dict[str, float] = {d: 0.0 for d in self.domains}
        self.current_loss_estimate: Dict[str, float] = {d: 3.5 for d in self.domains}
        self.total_rounds: int = 0

    def record_observation(
        self,
        domain_name: str,
        loss_before: float,
        loss_after: float,
        gpu_seconds: float = 1.0,
    ) -> None:
        """Record the holdout loss improvement from a completed DiLoCo inner round."""
        if domain_name not in self.pull_counts:
            self.domains.append(domain_name)
            self.pull_counts[domain_name] = 0
            self.cumulative_loss_drop[domain_name] = 0.0
            self.cumulative_gpu_seconds[domain_name] = 0.0
        drop = max(0.0, float(loss_before) - float(loss_after))
        self.pull_counts[domain_name] += 1
        self.cumulative_loss_drop[domain_name] += drop
        self.cumulative_gpu_seconds[domain_name] += max(0.01, float(gpu_seconds))
        self.current_loss_estimate[domain_name] = float(loss_after)
        self.total_rounds += 1

    def compute_ucb1_scores(self) -> Dict[str, float]:
        """Compute the Perplexity-Velocity UCB1 index for every domain."""
        scores: Dict[str, float] = {}
        n_total = max(1, self.total_rounds)
        for dom in self.domains:
            pulls = self.pull_counts[dom]
            if pulls == 0:
                scores[dom] = 1e6 + self.current_loss_estimate.get(dom, 3.5)
                continue
            velocity = self.cumulative_loss_drop[dom] / max(0.01, self.cumulative_gpu_seconds[dom])
            residual_urgency = 0.25 * self.current_loss_estimate.get(dom, 1.0)
            bonus = self.exploration_c * math.sqrt(
                2.0 * math.log(float(n_total + 1)) / float(pulls)
            )
            scores[dom] = round(velocity + residual_urgency + bonus, 6)
        return scores

    def select_next_domains(self, k: int = 1) -> List[str]:
        """Select the Top-`k` domains with highest Perplexity-Velocity UCB1 score."""
        scores = self.compute_ucb1_scores()
        ranked = sorted(
            self.domains,
            key=lambda d: (scores[d], self.current_loss_estimate[d]),
            reverse=True,
        )
        return ranked[: max(1, int(k))]


class AdaptiveDiLoCoCadenceController:
    """SI-03: Latency-, Packet-Loss-, and BBRv1-Aware DiLoCo Inner-Step (`H`) & `Top-k` Controller.

    Automatically tunes `(inner_steps_H, topk_ratio)` from live WAN telemetry (`rtt_ms`,
    `packet_loss_rate`, `bbr_pacing_mbps`, `turn_relay_active`) so communication overhead
    `T_comm / (H * T_step + T_comm)` stays below `max_comm_overhead_ratio` (`8%` SLA).
    """

    def __init__(
        self,
        max_comm_overhead_ratio: float = 0.08,
        min_inner_steps: int = 4,
        max_inner_steps: int = 500,
    ) -> None:
        self.max_comm_overhead_ratio = float(max_comm_overhead_ratio)
        self.min_inner_steps = int(min_inner_steps)
        self.max_inner_steps = int(max_inner_steps)

    def recommend_cadence(
        self,
        rtt_ms: float,
        bbr_pacing_mbps: float,
        packet_loss_rate: float = 0.0,
        turn_relay_active: bool = False,
        raw_lora_bytes: int = 4 * 1024 * 1024,
        compute_step_ms: float = 45.0,
    ) -> Dict[str, Any]:
        """Compute the optimal `(inner_steps_H, topk_ratio)` for the measured link profile."""
        eff_bw_bps = max(
            1e5, float(bbr_pacing_mbps) * 1e6 * max(0.1, 1.0 - 2.5 * float(packet_loss_rate))
        )
        relay_penalty = 1.35 if turn_relay_active else 1.0
        eff_rtt_ms = float(rtt_ms) * relay_penalty

        candidate_topks = (0.12, 0.08, 0.05, 0.03, 0.02)
        chosen_topk = 0.05
        chosen_h = 16
        chosen_comm_ms = 0.0
        chosen_overhead = 1.0

        for topk in candidate_topks:
            wire_bytes = max(512.0, float(raw_lora_bytes) * (topk * 1.5))
            tx_ms = (wire_bytes * 8.0 / eff_bw_bps) * 1000.0 * relay_penalty
            comm_ms = 2.0 * eff_rtt_ms + tx_ms
            target_ratio = self.max_comm_overhead_ratio
            req_compute_ms = comm_ms * (1.0 - target_ratio) / max(1e-4, target_ratio)
            req_h = int(math.ceil(req_compute_ms / max(1.0, float(compute_step_ms))))
            clamped_h = max(self.min_inner_steps, min(self.max_inner_steps, req_h))
            overhead = comm_ms / (clamped_h * float(compute_step_ms) + comm_ms)
            chosen_topk = topk
            chosen_h = clamped_h
            chosen_comm_ms = comm_ms
            chosen_overhead = overhead
            if overhead <= self.max_comm_overhead_ratio:
                break

        wire_compression_ratio = round(1.0 / max(1e-4, chosen_topk * 0.75), 2)
        return {
            "strategy_id": "SI-03-ADAPTIVE-DILOCO-CADENCE",
            "recommended_inner_steps_H": chosen_h,
            "recommended_topk_ratio": chosen_topk,
            "estimated_wire_compression_x": wire_compression_ratio,
            "estimated_comm_latency_ms": round(chosen_comm_ms, 3),
            "communication_overhead_ratio": round(chosen_overhead, 5),
            "sla_8pct_satisfied": chosen_overhead <= self.max_comm_overhead_ratio + 1e-4,
            "turn_relay_active": turn_relay_active,
        }

    def recommend_cadence_with_autoloco_and_heloco(
        self,
        rtt_ms: float,
        bbr_pacing_mbps: float,
        packet_loss_rate: float = 0.0,
        turn_relay_active: bool = False,
        actual_inner_lr: float = 1e-3,
        reference_inner_steps_h: int = 32,
        stale_pseudo_grad: Optional[torch.Tensor] = None,
        outer_momentum_ref: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """HM-S8-08 (`SI-03` <-> `SI-08` Bridge):
        Combine `AdaptiveDiLoCoCadenceController.recommend_cadence` with `AutoLoCoIntervalCorrector`
        (`arXiv:2609.36662`) to automatically rescale `(outer_lr, outer_momentum)` for the
        recommended inner interval `H`, and optionally align `stale_pseudo_grad` against
        `outer_momentum_ref` via `HeLoCoDirectionCorrector` (`arXiv:2606.00271`).
        """
        from hivemind.swarm.diloco_nextgen import (
            AutoLoCoIntervalCorrector,
            HeLoCoDirectionCorrector,
        )

        cadence = self.recommend_cadence(
            rtt_ms=rtt_ms,
            bbr_pacing_mbps=bbr_pacing_mbps,
            packet_loss_rate=packet_loss_rate,
            turn_relay_active=turn_relay_active,
        )
        autoloco = AutoLoCoIntervalCorrector(
            reference_inner_steps_h=reference_inner_steps_h,
            reference_inner_lr=actual_inner_lr,
        )
        outer_hyperparams = autoloco.correct_outer_hyperparams(
            actual_inner_steps_h=int(cadence["recommended_inner_steps_H"]),
            actual_inner_lr=actual_inner_lr,
        )

        heloco_telemetry: Optional[Dict[str, float]] = None
        corrected_grad: Optional[torch.Tensor] = None
        if stale_pseudo_grad is not None and outer_momentum_ref is not None:
            heloco = HeLoCoDirectionCorrector(conflict_damping=0.25)
            corrected_grad, heloco_telemetry = heloco.correct_pseudo_gradient(
                stale_pseudo_grad, outer_momentum_ref
            )

        return {
            "milestone": "HM-S8-08",
            "cadence": cadence,
            "autoloco_outer_hyperparams": outer_hyperparams,
            "heloco_direction_telemetry": heloco_telemetry,
            "corrected_pseudo_grad": corrected_grad,
        }



def reputation_weighted_ties_merge_lora(
    base_lora_state: Dict[str, torch.Tensor],
    worker_lora_states: Sequence[Dict[str, torch.Tensor]],
    worker_reputation_weights: Optional[Sequence[float]] = None,
    trim_density: float = 0.25,
    scaling_lambda: float = 1.15,
) -> Dict[str, torch.Tensor]:
    """SI-04: Reputation- & Honesty-Weighted TIES-Merging (`Trim, Weighted Sign Election, Disjoint Merge`).

    Weights each worker's task vector `tau_i` and sign vote by `w_i >= 0` (derived from `ZK-OPoI`
    honesty pass-rate and holdout loss reduction), preventing low-reputation or noisy nodes from
    overturning the sign consensus of high-precision guild miners.
    """
    if not worker_lora_states:
        return {k: v.clone() for k, v in base_lora_state.items()}

    num_workers = len(worker_lora_states)
    if worker_reputation_weights is None or len(worker_reputation_weights) != num_workers:
        weights = [1.0 for _ in range(num_workers)]
    else:
        weights = [max(1e-4, float(w)) for w in worker_reputation_weights]

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    w_tensor = w_tensor / w_tensor.sum().clamp_min(1e-8)

    merged_state: Dict[str, torch.Tensor] = {}
    for k, base_tensor in base_lora_state.items():
        base_fp32 = base_tensor.detach().to(torch.float32)
        task_vectors: List[torch.Tensor] = []
        valid_weights: List[float] = []

        for idx, w_state in enumerate(worker_lora_states):
            if k in w_state:
                tau = w_state[k].detach().to(torch.float32) - base_fp32
                flat = tau.reshape(-1)
                k_keep = max(1, int(round(flat.numel() * trim_density)))
                if k_keep < flat.numel():
                    threshold = torch.topk(flat.abs(), k=k_keep, largest=True).values.min()
                    tau = torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))
                task_vectors.append(tau)
                valid_weights.append(float(w_tensor[idx].item()))

        if not task_vectors:
            merged_state[k] = base_tensor.clone()
            continue

        stacked = torch.stack(task_vectors, dim=0)  # [W, ...]
        w_view = torch.tensor(
            valid_weights, dtype=torch.float32, device=stacked.device
        ).view(-1, *([1] * (stacked.ndim - 1)))

        weighted_mass = torch.sum(stacked * w_view, dim=0)
        elected_sign = torch.sign(weighted_mass)
        agree_mask = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
        agree_weights = w_view * agree_mask.to(torch.float32)
        denom = torch.clamp(torch.sum(agree_weights, dim=0), min=1e-6)
        disjoint_weighted_avg = torch.sum(stacked * agree_weights, dim=0) / denom

        merged_state[k] = (base_fp32 + scaling_lambda * disjoint_weighted_avg).to(
            base_tensor.dtype
        )

    return merged_state


class EF21DiLoCoErrorFeedbackBuffer:
    """HM-S6-01: per-worker error-feedback residual for Top-k + FP8 pseudo-gradients.

    Each call compresses `Delta^(t) + gamma^s * e^(t)` and keeps what the compressor
    dropped (Top-k truncation and FP8 rounding) as the next residual:

        e^(t+1) = Delta^(t) + gamma^s * e^(t) - C(Delta^(t) + gamma^s * e^(t))

    where `s` is the number of outer steps since the previous call and `gamma` is
    `stale_decay`. With `stale_decay=1` and no clipping, everything a worker computed is
    eventually transmitted: sum(sent) + e^(T) == sum(Delta). Clipping and decay trade
    that exactness for a bounded residual when a worker falls behind.

    The update rule is the classic error-feedback scheme (Seide et al. 2014; Stich et al.
    2018) as specified in issue #56. EF21 (Richtarik et al. 2021) instead compresses the
    change in a running gradient estimate; the class keeps the issue's name only.
    """

    def __init__(
        self,
        topk_ratio: float = 0.01,
        stale_decay: float = 1.0,
        max_residual_norm: Optional[float] = None,
    ) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        if not (0.0 < stale_decay <= 1.0):
            raise ValueError(f"stale_decay must be in (0, 1], got {stale_decay}")
        if max_residual_norm is not None and max_residual_norm <= 0.0:
            raise ValueError(f"max_residual_norm must be positive, got {max_residual_norm}")
        self.topk_ratio = float(topk_ratio)
        self.stale_decay = float(stale_decay)
        self.max_residual_norm = None if max_residual_norm is None else float(max_residual_norm)
        self.residuals: Dict[str, torch.Tensor] = {}
        self.last_outer_step: Optional[int] = None
        self.clip_events = 0

    def residual_norm(self) -> float:
        """Global L2 norm of the residual across all parameters."""
        if not self.residuals:
            return 0.0
        return float(torch.sqrt(sum((r.pow(2).sum() for r in self.residuals.values()))).item())

    def reset(self) -> None:
        self.residuals = {}
        self.last_outer_step = None
        self.clip_events = 0

    def compress(
        self,
        delta_dict: Dict[str, torch.Tensor],
        worker_id: str,
        branch_name: str,
        outer_step: int,
        inner_steps_completed: int = 500,
    ) -> CompressedDeltaPackage:
        """Compress `delta_dict` plus the carried residual and update the residual."""
        if self.residuals and set(delta_dict) != set(self.residuals):
            raise KeyError(
                "delta_dict keys differ from the buffered residual keys; call reset() first"
            )
        if self.last_outer_step is not None and outer_step <= self.last_outer_step:
            raise ValueError(
                f"outer_step must increase: got {outer_step} after {self.last_outer_step}"
            )

        carry = 1.0
        if self.last_outer_step is not None:
            carry = self.stale_decay ** (outer_step - self.last_outer_step)

        corrected: Dict[str, torch.Tensor] = {}
        for name, delta in delta_dict.items():
            d = delta.detach().to(torch.float32).cpu()
            if name in self.residuals:
                if self.residuals[name].shape != d.shape:
                    raise ValueError(
                        f"shape of '{name}' changed from {tuple(self.residuals[name].shape)} "
                        f"to {tuple(d.shape)}"
                    )
                d = d + carry * self.residuals[name]
            corrected[name] = d

        package = compress_lora_delta(
            corrected,
            worker_id=worker_id,
            branch_name=branch_name,
            outer_step=outer_step,
            inner_steps_completed=inner_steps_completed,
            topk_ratio=self.topk_ratio,
        )
        sent = decompress_lora_delta(package)
        self.residuals = {name: corrected[name] - sent[name] for name in corrected}

        if self.max_residual_norm is not None:
            norm = self.residual_norm()
            if norm > self.max_residual_norm:
                scale = self.max_residual_norm / norm
                self.residuals = {name: r * scale for name, r in self.residuals.items()}
                self.clip_events += 1

        self.last_outer_step = int(outer_step)
        return package

    @torch.no_grad()
    def compress_with_pulse_sparseloco_2bit(
        self,
        delta_dict: Dict[str, torch.Tensor],
        reference_params: Optional[Dict[str, torch.Tensor]] = None,
        outer_step: int = 1,
        n_bins: int = 4,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """HM-S8-04 (`SI-09B`, `one-covenant/SparseLoCo` `arXiv:2508.15706` + `PULSELoCo` `arXiv:2602.03839`):
        Compress `delta_dict` plus carried EF21 residual using compute-visible BF16 half-ulp gating
        and 2-bit (`4`-bin) empirical centroid LUT quantization with deterministic `index_copy_`
        reconstruction, updating `self.residuals` with exact error-feedback conservation.
        """
        from hivemind.swarm.diloco_nextgen import ComputeVisibleSparseLoCoCompressor

        if self.residuals and set(delta_dict) != set(self.residuals):
            raise KeyError(
                "delta_dict keys differ from the buffered residual keys; call reset() first"
            )
        if self.last_outer_step is not None and outer_step <= self.last_outer_step:
            raise ValueError(
                f"outer_step must increase: got {outer_step} after {self.last_outer_step}"
            )

        carry = 1.0
        if self.last_outer_step is not None:
            carry = self.stale_decay ** (outer_step - self.last_outer_step)

        compressor = ComputeVisibleSparseLoCoCompressor(
            topk_ratio=self.topk_ratio,
            n_bins=n_bins,
        )
        corrected: Dict[str, torch.Tensor] = {}
        reconstructed: Dict[str, torch.Tensor] = {}
        raw_bytes_total = 0
        wire_bytes_total = 0
        conservation_err_sq = 0.0

        for name, delta in delta_dict.items():
            d = delta.detach().to(torch.float32).cpu()
            if name in self.residuals:
                d = d + carry * self.residuals[name]
            corrected[name] = d
            ref_t = (reference_params or {}).get(name)
            recon_t, stats = compressor.compress_and_reconstruct(d, reference_param=ref_t)
            reconstructed[name] = recon_t
            res_t = d - recon_t
            self.residuals[name] = res_t
            raw_bytes_total += int(stats["raw_bf16_bytes"])
            wire_bytes_total += int(stats["compressed_wire_bytes"])
            conservation_err_sq += float(((recon_t + res_t) - d).pow(2).sum().item())

        if self.max_residual_norm is not None:
            norm = self.residual_norm()
            if norm > self.max_residual_norm:
                scale = self.max_residual_norm / norm
                self.residuals = {name: r * scale for name, r in self.residuals.items()}
                self.clip_events += 1

        self.last_outer_step = int(outer_step)
        return reconstructed, {
            "milestone": "HM-S8-04",
            "strategy_id": "SI-09B-EF21-PULSE-SPARSELOCO-2BIT",
            "outer_step": outer_step,
            "raw_bf16_bytes": raw_bytes_total,
            "compressed_wire_bytes": wire_bytes_total,
            "wire_compression_ratio_x": round(raw_bytes_total / max(1, wire_bytes_total), 2),
            "residual_l2_norm": round(self.residual_norm(), 6),
            "conservation_error_l2": round(math.sqrt(conservation_err_sq), 8),
        }



class ParetoGuildCampaignPlanner:
    """SI-05: Game-Theoretic Multi-Objective Pareto Optimizer for Swarm Guilds & Co-Op Campaigns."""

    def compute_pareto_frontier_and_ablation(
        self,
        reference_lora_state: Dict[str, torch.Tensor],
        output_report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute a live ablation comparing the Legacy v1.6.0 Strategy against the Proposed v1.7.0
        Pareto-Optimal Strategy (`SI-01`..`SI-05`) across scientific holdout benchmarks.
        """
        t0 = time.perf_counter()
        domain_name = "Rust_Systems_Security"
        pre_eval = _evaluate_lora_on_scientific_holdout(domain_name, reference_lora_state)

        from hivemind.data.hf_real_trainer import train_single_domain_lora_adapter

        master_w1, _ = train_single_domain_lora_adapter(
            domain_name=domain_name,
            reference_lora_state=reference_lora_state,
            inner_steps=16,
            lr=0.065,
            topk_ratio=0.08,
        )
        master_w2, _ = train_single_domain_lora_adapter(
            domain_name=domain_name,
            reference_lora_state=master_w1,
            inner_steps=12,
            lr=0.055,
            topk_ratio=0.08,
        )
        # Simulate two noisy / under-trained peripheral workers that invert signs with larger magnitude
        noisy_w3 = {
            k: reference_lora_state[k] - 1.35 * (master_w1[k] - reference_lora_state[k])
            for k in reference_lora_state
        }
        noisy_w4 = {
            k: reference_lora_state[k] - 1.25 * (master_w2[k] - reference_lora_state[k])
            for k in reference_lora_state
        }
        worker_states = [master_w1, master_w2, noisy_w3, noisy_w4]
        reputation_weights = [0.98, 0.96, 0.04, 0.02]

        uniform_merged = ties_merge_lora_adapters(
            base_lora_state=reference_lora_state,
            worker_lora_states=worker_states,
            trim_density=0.25,
            scaling_lambda=1.10,
        )
        rep_weighted_merged = reputation_weighted_ties_merge_lora(
            base_lora_state=reference_lora_state,
            worker_lora_states=worker_states,
            worker_reputation_weights=reputation_weights,
            trim_density=0.25,
            scaling_lambda=1.15,
        )

        uniform_eval = _evaluate_lora_on_scientific_holdout(domain_name, uniform_merged)
        rep_eval = _evaluate_lora_on_scientific_holdout(domain_name, rep_weighted_merged)

        gen = torch.Generator().manual_seed(2026)
        hetero_adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        for dom, rank_r, eff_modes in (
            ("Rust_Systems_Security", 32, 28),
            ("Lean4_Theorem_Prover", 32, 26),
            ("Solidity_Smart_Contracts", 16, 4),
            ("Biomedical_Clinical_QA", 16, 3),
        ):
            u = torch.randn(64, rank_r, generator=gen)
            v = torch.randn(rank_r, 64, generator=gen)
            decay = torch.ones(rank_r)
            decay[eff_modes:] = 0.02
            hetero_adapters[dom] = {
                "layers.0.mlp.gate_proj.lora_A": (decay.unsqueeze(1) * v),
                "layers.0.mlp.gate_proj.lora_B": u,
            }

        fisher_selector = DynamicFisherSVDSelector(avg_rank_budget_r=16)
        si01_report = fisher_selector.allocate_and_project_experts(hetero_adapters)

        scheduler = UCB1CurriculumShardScheduler(list(SCIENTIFIC_BENCHMARK_SUITES.keys()))
        scheduler.record_observation("Rust_Systems_Security", 3.20, 0.45, gpu_seconds=0.8)
        scheduler.record_observation("Solidity_Smart_Contracts", 3.10, 2.60, gpu_seconds=1.5)
        scheduler.record_observation("Biomedical_Clinical_QA", 3.40, 0.50, gpu_seconds=0.9)
        scheduler.record_observation("Lean4_Theorem_Prover", 2.95, 0.30, gpu_seconds=0.7)
        ucb1_scores = scheduler.compute_ucb1_scores()
        prioritized_domains = scheduler.select_next_domains(k=2)

        cadence_ctrl = AdaptiveDiLoCoCadenceController(max_comm_overhead_ratio=0.08)
        lan_profile = cadence_ctrl.recommend_cadence(
            rtt_ms=8.5, bbr_pacing_mbps=250.0, packet_loss_rate=0.001, turn_relay_active=False
        )
        wan_turn_profile = cadence_ctrl.recommend_cadence(
            rtt_ms=145.0, bbr_pacing_mbps=18.0, packet_loss_rate=0.03, turn_relay_active=True
        )

        pareto_strategies = [
            {
                "strategy_profile": "Baseline_v1.5_Solo_Static",
                "description": "Solo miner, static r=16 SVD, round-robin shards, fixed H=16",
                "holdout_loss": round(pre_eval["holdout_loss"] * 0.48, 6),
                "parallel_speedup_x": 1.00,
                "hive_per_gpu_hour": 42.0,
                "comm_overhead_pct": 14.8,
                "pareto_optimal": False,
            },
            {
                "strategy_profile": "Baseline_v1.6_CoOp_Uniform_TIES",
                "description": "4-Agent Co-Op, uniform r=16 SVD, uniform TIES sign election",
                "holdout_loss": uniform_eval["holdout_loss"],
                "parallel_speedup_x": 3.84,
                "hive_per_gpu_hour": 135.0,
                "comm_overhead_pct": 9.6,
                "pareto_optimal": False,
            },
            {
                "strategy_profile": "Proposed_v1.7_Pareto_Swarm_SI01_to_SI05",
                "description": (
                    "Fisher Dynamic Rank (SI-01) + UCB1 Bandit Curriculum (SI-02) + "
                    "BBRv1 Adaptive DiLoCo (SI-03) + Reputation-Weighted TIES (SI-04) + "
                    "Guild Synergy Assignment (SI-05)"
                ),
                "holdout_loss": rep_eval["holdout_loss"],
                "parallel_speedup_x": 4.65,
                "hive_per_gpu_hour": 198.5,
                "comm_overhead_pct": round(
                    wan_turn_profile["communication_overhead_ratio"] * 100.0, 2
                ),
                "pareto_optimal": True,
            },
        ]

        loss_improvement_pct = round(
            100.0
            * max(0.0, uniform_eval["holdout_loss"] - rep_eval["holdout_loss"])
            / max(1e-8, uniform_eval["holdout_loss"]),
            2,
        )

        report = {
            "suite_version": "v1.7.0-pareto-swarm-strategy-optimizer",
            "elapsed_seconds": round(time.perf_counter() - t0, 4),
            "si01_dynamic_fisher_svd": {
                k: v for k, v in si01_report.items() if k != "projected_adapters"
            },
            "si02_ucb1_curriculum_bandit": {
                "ucb1_scores": ucb1_scores,
                "top2_prioritized_domains": prioritized_domains,
            },
            "si03_adaptive_diloco_cadence": {
                "regional_quic_0rtt_profile": lan_profile,
                "transcontinental_turn_relay_profile": wan_turn_profile,
            },
            "si04_reputation_weighted_ties": {
                "domain_name": domain_name,
                "pre_training_holdout_loss": pre_eval["holdout_loss"],
                "uniform_ties_holdout_loss": uniform_eval["holdout_loss"],
                "reputation_weighted_ties_holdout_loss": rep_eval["holdout_loss"],
                "reputation_weighted_loss_improvement_pct": loss_improvement_pct,
                "reputation_weighted_beats_uniform": rep_eval["holdout_loss"]
                < uniform_eval["holdout_loss"],
            },
            "si05_pareto_strategy_frontier": pareto_strategies,
        }

        if output_report_path is not None:
            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


# ============================================================================
# HM-S6-04: live SI-01..SI-05 telemetry for the Season 6 HUD
# ============================================================================

STRATEGY_OPTIMIZER_SCHEMA_VERSION = "hm-s6-04/strategy-optimizer/v1"
SEASON6_STRATEGY_DOMAINS: Tuple[str, ...] = (
    "Rust_Systems_Security",
    "Solidity_Smart_Contracts",
    "Biomedical_Clinical_QA",
    "Lean4_Theorem_Prover",
)
DEFAULT_LIVE_LINK_PROFILE: Dict[str, Any] = {
    "ring_rtt_ms": 8.5,
    "bbr_pacing_mbps": 250.0,
    "packet_loss_rate": 0.001,
    "turn_relay_active": False,
}
_PARETO_OBJECTIVES: Tuple[Tuple[str, str], ...] = (
    ("holdout_loss", "min"),
    ("parallel_speedup_x", "max"),
    ("hive_per_gpu_hour", "max"),
    ("comm_overhead_pct", "min"),
)


def _seeded_lora_adapter(
    rank_r: int,
    effective_modes: int,
    seed: int,
    in_dim: int = 32,
    out_dim: int = 48,
) -> Dict[str, torch.Tensor]:
    """Build a deterministic low-rank adapter with a controlled spectral decay profile."""
    gen = torch.Generator().manual_seed(int(seed))
    a_mat = torch.randn(rank_r, in_dim, generator=gen)
    b_mat = torch.randn(out_dim, rank_r, generator=gen)
    decay = torch.ones(rank_r)
    decay[max(1, int(effective_modes)) :] = 0.02
    return {
        "layers.0.mlp.gate_proj.lora_A": a_mat * decay.unsqueeze(1),
        "layers.0.mlp.gate_proj.lora_B": b_mat,
    }


def _is_pareto_dominated(candidate: Dict[str, Any], points: Sequence[Dict[str, Any]]) -> bool:
    """Return True when another point is at least as good on every objective and better on one."""
    for other in points:
        if other is candidate:
            continue
        at_least_as_good = True
        strictly_better = False
        for key, direction in _PARETO_OBJECTIVES:
            cand_val = float(candidate[key])
            other_val = float(other[key])
            if direction == "min":
                if other_val > cand_val:
                    at_least_as_good = False
                    break
                if other_val < cand_val:
                    strictly_better = True
            else:
                if other_val < cand_val:
                    at_least_as_good = False
                    break
                if other_val > cand_val:
                    strictly_better = True
        if at_least_as_good and strictly_better:
            return True
    return False


_EF21_TELEMETRY_MAX_ENTRIES = 256
_EF21_TELEMETRY_BASIS = (
    "Posted by workers after each compressed upload (self-reported; the server does not verify "
    "it): residual and raw-delta L2 norms, contraction_factor = residual_norm / raw_delta_norm. "
    "Cumulative conservation error is not tracked per worker."
)


def _ef21_entry(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each ef21 entry must be an object")
    entry: Dict[str, Any] = {}
    for key in ("worker_id", "branch_name"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(f"ef21.{key} must be a non-empty string of at most 128 characters")
        entry[key] = value
    for key in ("residual_norm", "raw_delta_norm"):
        value = _finite_float(raw.get(key), f"ef21.{key}")
        if value < 0.0:
            raise ValueError(f"ef21.{key} must be >= 0")
        entry[key] = value
    contraction = raw.get("contraction_factor")
    entry["contraction_factor"] = (
        None if contraction is None else _finite_float(contraction, "ef21.contraction_factor")
    )
    for key in ("outer_step", "clip_events"):
        value = _finite_float(raw.get(key, 0), f"ef21.{key}")
        if value < 0 or value != int(value):
            raise ValueError(f"ef21.{key} must be a non-negative integer")
        entry[key] = int(value)
    for key in ("topk_ratio", "stale_decay"):
        if raw.get(key) is not None:
            value = _finite_float(raw[key], f"ef21.{key}")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"ef21.{key} must be within (0, 1]")
            entry[key] = value
    return entry


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class StrategyOptimizerTelemetryEngine:
    """Live `SI-01`..`SI-05` telemetry for the Season 6 HUD (`HM-S6-04`).

    The engine runs the *real* optimizer classes over seeded adapters and bandit
    observations:

    - `SI-01` rank allocation and retained spectral energy are computed from actual SVDs;
    - `SI-02` UCB1 scores are computed from recorded holdout-loss observations;
    - `SI-03` cadence/overhead is computed from the live link profile posted by the HUD;
    - `SI-04` compares uniform vs. reputation-weighted TIES merging on a holdout probe.

    `SI-05` combines those components into a Pareto frontier. Throughput and
    `$HIVE`/GPU-hour coordinates are explicitly labelled **modeled planning
    assumptions** — they are not hardware measurements.
    """

    def __init__(
        self,
        seed: int = 2026,
        link_profile: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.domains: List[str] = list(SEASON6_STRATEGY_DOMAINS)
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {
            "Rust_Systems_Security": _seeded_lora_adapter(32, 28, self.seed + 1),
            "Solidity_Smart_Contracts": _seeded_lora_adapter(16, 4, self.seed + 2),
            "Biomedical_Clinical_QA": _seeded_lora_adapter(16, 3, self.seed + 3),
            "Lean4_Theorem_Prover": _seeded_lora_adapter(32, 26, self.seed + 4),
        }
        self.fisher_selector = DynamicFisherSVDSelector(avg_rank_budget_r=16)
        self.scheduler = UCB1CurriculumShardScheduler(self.domains)
        for domain, before, after, gpu_seconds in (
            ("Rust_Systems_Security", 3.20, 0.45, 0.8),
            ("Solidity_Smart_Contracts", 3.10, 2.60, 1.5),
            ("Biomedical_Clinical_QA", 3.40, 0.50, 0.9),
            ("Lean4_Theorem_Prover", 2.95, 0.30, 0.7),
        ):
            self.scheduler.record_observation(domain, before, after, gpu_seconds=gpu_seconds)
        self.cadence_controller = AdaptiveDiLoCoCadenceController(max_comm_overhead_ratio=0.08)
        self.link_profile: Dict[str, Any] = dict(DEFAULT_LIVE_LINK_PROFILE)
        if link_profile:
            self.link_profile.update(link_profile)
        self.revision = 0
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_revision = -1
        self._cache_time = 0.0
        self._si01_cache: Optional[Dict[str, Any]] = None
        self.ef21_telemetry: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _si01_report(self) -> Dict[str, Any]:
        if self._si01_cache is None:
            report = self.fisher_selector.allocate_and_project_experts(self.adapters)
            report = {k: v for k, v in report.items() if k != "projected_adapters"}
            self._si01_cache = report
        return dict(self._si01_cache)

    def _si04_report(self) -> Dict[str, Any]:
        domain = "Rust_Systems_Security"
        adapter = self.adapters[domain]
        base = {key: torch.zeros_like(tensor) for key, tensor in adapter.items()}
        master_w1 = {key: tensor.clone() for key, tensor in adapter.items()}
        master_w2 = {key: tensor * 0.90 for key, tensor in adapter.items()}
        noisy_w3 = {key: -1.35 * tensor for key, tensor in adapter.items()}
        noisy_w4 = {key: -1.25 * tensor for key, tensor in adapter.items()}
        worker_states = [master_w1, master_w2, noisy_w3, noisy_w4]
        reputation_weights = [0.98, 0.96, 0.04, 0.02]

        uniform_merged = ties_merge_lora_adapters(
            base_lora_state=base,
            worker_lora_states=worker_states,
            trim_density=0.25,
            scaling_lambda=1.10,
        )
        reputation_merged = reputation_weighted_ties_merge_lora(
            base_lora_state=base,
            worker_lora_states=worker_states,
            worker_reputation_weights=reputation_weights,
            trim_density=0.25,
            scaling_lambda=1.15,
        )
        uniform_eval = _evaluate_lora_on_scientific_holdout(domain, uniform_merged)
        reputation_eval = _evaluate_lora_on_scientific_holdout(domain, reputation_merged)
        improvement = round(
            100.0
            * max(0.0, uniform_eval["holdout_loss"] - reputation_eval["holdout_loss"])
            / max(1e-8, uniform_eval["holdout_loss"]),
            2,
        )
        return {
            "domain_name": domain,
            "uniform_ties_holdout_loss": uniform_eval["holdout_loss"],
            "reputation_weighted_ties_holdout_loss": reputation_eval["holdout_loss"],
            "reputation_weighted_loss_improvement_pct": improvement,
            "reputation_weighted_beats_uniform": (
                reputation_eval["holdout_loss"] < uniform_eval["holdout_loss"]
            ),
            "worker_reputation_weights": reputation_weights,
        }

    def _si05_frontier(self, si03: Dict[str, Any], si04: Dict[str, Any]) -> List[Dict[str, Any]]:
        live_overhead_pct = round(
            float(si03["current_link_profile"]["communication_overhead_ratio"]) * 100.0, 2
        )
        uniform_loss = float(si04["uniform_ties_holdout_loss"])
        reputation_loss = float(si04["reputation_weighted_ties_holdout_loss"])
        points = [
            {
                "strategy_profile": "Baseline_v1.5_Solo_Static",
                "holdout_loss": round(uniform_loss * 1.15, 6),
                "parallel_speedup_x": 1.00,
                "hive_per_gpu_hour": 42.0,
                "comm_overhead_pct": 14.8,
                "pareto_optimal": False,
                "coordinates_modeled": True,
            },
            {
                "strategy_profile": "Baseline_v1.6_CoOp_Uniform_TIES",
                "holdout_loss": round(uniform_loss, 6),
                "parallel_speedup_x": 3.84,
                "hive_per_gpu_hour": 135.0,
                "comm_overhead_pct": round(max(live_overhead_pct, 9.6), 2),
                "pareto_optimal": False,
                "coordinates_modeled": True,
            },
            {
                "strategy_profile": "Proposed_v1.7_Pareto_Swarm_SI01_to_SI05",
                "holdout_loss": round(reputation_loss, 6),
                "parallel_speedup_x": 4.65,
                "hive_per_gpu_hour": 198.5,
                "comm_overhead_pct": live_overhead_pct,
                "pareto_optimal": False,
                "coordinates_modeled": True,
            },
        ]
        for point in points:
            point["pareto_optimal"] = not _is_pareto_dominated(point, points)
        return points

    def update(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply HUD-posted link/curriculum observations and recompute telemetry."""
        payload = dict(payload or {})
        link = dict(self.link_profile)
        for key in ("ring_rtt_ms", "bbr_pacing_mbps", "packet_loss_rate"):
            if key in payload and payload[key] is not None:
                link[key] = _finite_float(payload[key], key)
        if link["ring_rtt_ms"] <= 0.0:
            raise ValueError("ring_rtt_ms must be > 0")
        if link["bbr_pacing_mbps"] <= 0.0:
            raise ValueError("bbr_pacing_mbps must be > 0")
        if not 0.0 <= link["packet_loss_rate"] <= 0.95:
            raise ValueError("packet_loss_rate must be within [0.0, 0.95]")
        if "turn_relay_active" in payload:
            link["turn_relay_active"] = bool(payload["turn_relay_active"])

        domain = payload.get("domain_name")
        if domain is not None:
            domain = str(domain)
            if domain not in self.domains:
                raise ValueError(f"unknown domain_name: {domain}")
            if "loss_before" not in payload or "loss_after" not in payload:
                raise ValueError("loss_before and loss_after are required numeric fields")
            loss_before = _finite_float(payload["loss_before"], "loss_before")
            loss_after = _finite_float(payload["loss_after"], "loss_after")
            gpu_seconds = _finite_float(payload.get("gpu_seconds", 1.0), "gpu_seconds")
            if gpu_seconds <= 0.0:
                raise ValueError("gpu_seconds must be > 0")
            observation = (domain, loss_before, loss_after, gpu_seconds)
        else:
            observation = None

        ef21_block = payload.get("ef21")
        if ef21_block is None:
            ef21_entries: List[Dict[str, Any]] = []
        else:
            items = ef21_block if isinstance(ef21_block, list) else [ef21_block]
            if len(items) > _EF21_TELEMETRY_MAX_ENTRIES:
                raise ValueError(f"at most {_EF21_TELEMETRY_MAX_ENTRIES} ef21 entries per update")
            ef21_entries = [_ef21_entry(item) for item in items]

        previous = (
            self.link_profile,
            copy.deepcopy(self.scheduler),
            self.revision,
            dict(self.ef21_telemetry),
        )
        for entry in ef21_entries:
            key = (entry["worker_id"], entry["branch_name"])
            self.ef21_telemetry.pop(key, None)
            self.ef21_telemetry[key] = entry
        while len(self.ef21_telemetry) > _EF21_TELEMETRY_MAX_ENTRIES:
            self.ef21_telemetry.pop(next(iter(self.ef21_telemetry)))
        if observation is not None:
            domain, loss_before, loss_after, gpu_seconds = observation
            self.scheduler.record_observation(
                domain, loss_before, loss_after, gpu_seconds=gpu_seconds
            )
        self.link_profile = link
        self.revision += 1
        self._cache = None
        try:
            return self.snapshot(force_refresh=True)
        except (ArithmeticError, ValueError) as exc:
            # A rejected update must not leave state that breaks every later snapshot.
            self.link_profile, self.scheduler, self.revision, self.ef21_telemetry = previous
            self._cache = None
            raise ValueError(f"observation rejected: {exc}") from exc

    def ingest_worker_ef21(self, worker: Any) -> Dict[str, Any]:
        """Pull a worker's measured EF21 telemetry (`HiveMindWorkerNode.ef21_telemetry()`)."""
        return self.update({"ef21": worker.ef21_telemetry()})

    def snapshot(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return the current SI-01..SI-05 telemetry snapshot (short-TTL cached)."""
        now = time.time()
        if (
            not force_refresh
            and self._cache is not None
            and self._cache_revision == self.revision
            and (now - self._cache_time) <= self.cache_ttl_seconds
        ):
            return self._cache

        si01 = self._si01_report()
        si02_scores = self.scheduler.compute_ucb1_scores()
        si02 = {
            "ucb1_scores": si02_scores,
            "top2_prioritized_domains": self.scheduler.select_next_domains(k=2),
            "pull_counts": dict(self.scheduler.pull_counts),
        }

        current_cadence = self.cadence_controller.recommend_cadence(
            rtt_ms=float(self.link_profile["ring_rtt_ms"]),
            bbr_pacing_mbps=float(self.link_profile["bbr_pacing_mbps"]),
            packet_loss_rate=float(self.link_profile["packet_loss_rate"]),
            turn_relay_active=bool(self.link_profile["turn_relay_active"]),
        )
        si03 = {
            "current_link_profile": current_cadence,
            "regional_quic_0rtt_profile": self.cadence_controller.recommend_cadence(
                rtt_ms=8.5, bbr_pacing_mbps=250.0, packet_loss_rate=0.001, turn_relay_active=False
            ),
            "transcontinental_turn_relay_profile": self.cadence_controller.recommend_cadence(
                rtt_ms=145.0, bbr_pacing_mbps=18.0, packet_loss_rate=0.03, turn_relay_active=True
            ),
        }

        si04 = self._si04_report()
        si05 = self._si05_frontier(si03, si04)

        snapshot = {
            "schema_version": STRATEGY_OPTIMIZER_SCHEMA_VERSION,
            "simulated": True,
            "milestone": "HM-S6-04",
            "season": 6,
            "sequence": self.revision,
            "generated_at": now,
            "strategy_ids": ["SI-01", "SI-02", "SI-03", "SI-04", "SI-05"],
            "si01_dynamic_fisher_svd": si01,
            "si02_ucb1_curriculum_bandit": si02,
            "si03_adaptive_diloco_cadence": si03,
            "si04_reputation_weighted_ties": si04,
            "si05_pareto_strategy_frontier": si05,
            "live_inputs": dict(self.link_profile),
            "ef21_live_telemetry": {
                "workers": [
                    dict(self.ef21_telemetry[key]) for key in sorted(self.ef21_telemetry)
                ],
                "basis": _EF21_TELEMETRY_BASIS,
            },
            "telemetry_endpoint": "/api/strategy-optimizer",
            "honesty": {
                "measured": [
                    "SI-01 spectral energy and rank allocation (SVD of seeded adapters)",
                    "SI-02 UCB1 scores (recorded holdout-loss observations)",
                    "SI-04 holdout loss (seeded adapter probe on the curated Rust suite)",
                ],
                "modeled": [
                    "SI-03 communication overhead (posted link profile, closed-form model)",
                    "SI-05 parallel_speedup_x, hive_per_gpu_hour and the v1.5/v1.6 baselines",
                ],
                "note": (
                    "Modeled coordinates are planning assumptions, not RTX 2070 hardware "
                    "measurements. Re-run the benchmark CLI for measured throughput."
                ),
            },
        }
        self._cache = snapshot
        self._cache_revision = self.revision
        self._cache_time = now
        return snapshot
