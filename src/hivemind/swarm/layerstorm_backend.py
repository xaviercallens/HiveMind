# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 25 (HM-S25-01..05): LayerStoRm x HiveMind PCIe Expert Streaming & Two-Zone VRAM Cache Bridge.

Integrates the core architectural primitives of LayerStoRm (kkontosis/LayerStoRm) with
Project HiveMind's Bi-Layer Honeybee Agent Mesh:
1. TwoZoneVRAMCachePlanner (HM-S25-02 & HM-S25-04):
   - Enforces the strict consumer/prosumer VRAM ceiling:
     VRAM_dense + VRAM_KV + VRAM_ZoneA + VRAM_ZoneB <= VRAM_cap_mib.
   - Pre-warms Zone A with domain-matched experts during the Kafka Waggle Dance
     quorum lock window (< 450 ms) before token 1 arrives.
   - Manages Zone B as a deterministic LRU ring buffer for token-routed experts
     streamed from pinned NUMA host RAM.
2. PCIeWaterlineStreamScheduler (HM-S25-01 & HM-S25-03):
   - Schedules deduplicated PCIe H2D DMA transfers from pinned NUMA host RAM.
   - Overlaps layer L+1 expert H2D transfers with layer L dense attention using
     FoMoE Skip-Attention speculative routing predictions.
3. LayerStoRmSwarmBridge (HM-S25-05):
   - Places the P2P SwarmNectarCache (HM-S23) upstream of LayerStoRm to bypass
     PCIe streaming on recurring sub-tasks.
   - Implements the < 0.5s hardware gamer preemption circuit-breaker that drains
     in-flight DMA queues and yields the GPU immediately when a 3D application launches.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Any

import torch

from hivemind.swarm.swarm_nectar_cache import SwarmNectarCache


DEFAULT_DOMAIN_HOT_EXPERTS: dict[str, tuple[int, ...]] = {
    "systems_deeptech": (0, 1, 2, 3, 4, 5, 6, 7),
    "cyber_defense": (8, 9, 10, 11, 12, 13, 14, 15),
    "stem_scholar": (16, 17, 18, 19, 20, 21, 22, 23),
    "life_sciences": (24, 25, 26, 27, 28, 29, 30, 31),
}


@dataclass(frozen=True)
class TwoZoneVRAMBudget:
    """Verified memory partition for a LayerStoRm node under a hard VRAM ceiling."""

    vram_cap_mib: float
    dense_attention_mib: float
    kv_cache_mib: float
    expert_size_mib: float
    zone_a_slots: int
    zone_b_slots: int
    host_numa_ram_cap_mib: float = 65536.0

    @property
    def zone_a_mib(self) -> float:
        return self.zone_a_slots * self.expert_size_mib

    @property
    def zone_b_mib(self) -> float:
        return self.zone_b_slots * self.expert_size_mib

    @property
    def total_vram_mib(self) -> float:
        return (
            self.dense_attention_mib
            + self.kv_cache_mib
            + self.zone_a_mib
            + self.zone_b_mib
        )

    @property
    def headroom_mib(self) -> float:
        return self.vram_cap_mib - self.total_vram_mib

    def validate(self) -> None:
        """Fail-closed check of the Golden VRAM Ceiling invariant."""
        if self.vram_cap_mib <= 0:
            raise ValueError(f"vram_cap_mib must be positive, got {self.vram_cap_mib}")
        if self.expert_size_mib <= 0:
            raise ValueError(f"expert_size_mib must be positive, got {self.expert_size_mib}")
        if self.zone_a_slots < 0 or self.zone_b_slots < 0:
            raise ValueError("Zone A and Zone B slot counts must be non-negative")
        if self.total_vram_mib > self.vram_cap_mib:
            raise ValueError(
                f"Two-Zone VRAM overflow: total={self.total_vram_mib:.2f} MiB exceeds "
                f"cap={self.vram_cap_mib:.2f} MiB (dense={self.dense_attention_mib:.1f}, "
                f"kv={self.kv_cache_mib:.1f}, zone_a={self.zone_a_mib:.1f}, zone_b={self.zone_b_mib:.1f})"
            )


class TwoZoneVRAMCache:
    """Two-Zone GPU VRAM Expert Cache (Zone A: Waggle-Warm Pinned; Zone B: Dynamic LRU)."""

    def __init__(self, budget: TwoZoneVRAMBudget) -> None:
        budget.validate()
        self.budget = budget
        self._zone_a: set[int] = set()
        self._zone_b: OrderedDict[int, bool] = OrderedDict()
        self.active_domain: str | None = None
        self.zone_a_hits: int = 0
        self.zone_b_hits: int = 0
        self.pcie_misses: int = 0

    @property
    def zone_a_experts(self) -> frozenset[int]:
        return frozenset(self._zone_a)

    @property
    def zone_b_experts(self) -> tuple[int, ...]:
        return tuple(self._zone_b.keys())

    def prewarm_zone_a_from_waggle(
        self,
        domain: str,
        explicit_experts: list[int] | tuple[int, ...] | None = None,
    ) -> list[int]:
        """Pre-populate Zone A during the Kafka Waggle Dance quorum lock window."""
        candidates = (
            list(explicit_experts)
            if explicit_experts is not None
            else list(DEFAULT_DOMAIN_HOT_EXPERTS.get(domain, ()))
        )
        selected = candidates[: self.budget.zone_a_slots]
        self._zone_a = set(selected)
        self.active_domain = domain
        for exp_id in self._zone_a:
            self._zone_b.pop(exp_id, None)
        return selected

    def access_expert(self, expert_id: int) -> str:
        """Query an expert; returns 'zone_a_hit', 'zone_b_hit', or 'pcie_stream_miss'."""
        if expert_id in self._zone_a:
            self.zone_a_hits += 1
            return "zone_a_hit"
        if expert_id in self._zone_b:
            self.zone_b_hits += 1
            self._zone_b.move_to_end(expert_id)
            return "zone_b_hit"

        self.pcie_misses += 1
        if self.budget.zone_b_slots > 0:
            if len(self._zone_b) >= self.budget.zone_b_slots:
                self._zone_b.popitem(last=False)
            self._zone_b[expert_id] = True
        return "pcie_stream_miss"

    def clear_on_preempt(self) -> None:
        """Release all cached expert slots immediately on hardware gamer preemption."""
        self._zone_a.clear()
        self._zone_b.clear()
        self.active_domain = None


@dataclass
class PCIeStepTelemetry:
    """Telemetry for a single MoE layer or token decode step over PCIe DMA."""

    requested_experts: tuple[int, ...]
    streamed_experts: tuple[int, ...]
    pcie_h2d_mib: float
    raw_transfer_ms: float
    overlapped_stall_ms: float
    total_step_ms: float


class PCIeWaterlineStreamScheduler:
    """Models LayerStoRm deduplicated PCIe DMA streaming with FoMoE speculative overlap."""

    def __init__(
        self,
        vram_cache: TwoZoneVRAMCache,
        pcie_bandwidth_gbps: float = 24.0,
        layer_attention_ms: float = 1.8,
        fomoe_speculative_overlap_ratio: float = 0.858,
    ) -> None:
        if pcie_bandwidth_gbps <= 0:
            raise ValueError("pcie_bandwidth_gbps must be positive")
        self.vram_cache = vram_cache
        self.pcie_bandwidth_mib_per_ms = (pcie_bandwidth_gbps * 1024.0) / 1000.0
        self.layer_attention_ms = max(0.0, layer_attention_ms)
        self.fomoe_speculative_overlap_ratio = max(0.0, min(1.0, fomoe_speculative_overlap_ratio))
        self.cumulative_h2d_mib: float = 0.0
        self.cumulative_steps: int = 0

    def execute_layer_step(
        self,
        activated_experts: list[int] | tuple[int, ...],
        use_waterline_overlap: bool = True,
    ) -> PCIeStepTelemetry:
        """Deduplicate activated experts, stream misses over PCIe, and overlap with attention."""
        unique_experts = tuple(dict.fromkeys(activated_experts))
        streamed: list[int] = []
        for exp_id in unique_experts:
            outcome = self.vram_cache.access_expert(exp_id)
            if outcome == "pcie_stream_miss":
                streamed.append(exp_id)

        h2d_mib = len(streamed) * self.vram_cache.budget.expert_size_mib
        raw_transfer_ms = h2d_mib / self.pcie_bandwidth_mib_per_ms if h2d_mib > 0 else 0.0

        if use_waterline_overlap and raw_transfer_ms > 0:
            overlapped_window_ms = self.layer_attention_ms * self.fomoe_speculative_overlap_ratio
            stall_ms = max(0.0, raw_transfer_ms - overlapped_window_ms)
        else:
            stall_ms = raw_transfer_ms

        total_step_ms = self.layer_attention_ms + stall_ms
        self.cumulative_h2d_mib += h2d_mib
        self.cumulative_steps += 1

        return PCIeStepTelemetry(
            requested_experts=unique_experts,
            streamed_experts=tuple(streamed),
            pcie_h2d_mib=round(h2d_mib, 4),
            raw_transfer_ms=round(raw_transfer_ms, 4),
            overlapped_stall_ms=round(stall_ms, 4),
            total_step_ms=round(total_step_ms, 4),
        )


@dataclass
class LayerStoRmSwarmBridge:
    """End-to-end Season 25 bridge coupling SwarmNectarCache, Waggle-Warm Zone A, and LayerStoRm PCIe DMA."""

    scheduler: PCIeWaterlineStreamScheduler
    nectar_cache: SwarmNectarCache = field(default_factory=SwarmNectarCache)
    is_preempted: bool = False
    last_preemption_latency_ms: float = 0.0

    def serve_request(
        self,
        domain: str,
        query: str,
        token_expert_traces: list[tuple[int, ...]],
        discoverer_bee_id: str = "bee-layerstorm-prosumer-01",
        prewarm_on_waggle: bool = True,
        use_waterline_overlap: bool = True,
    ) -> dict[str, Any]:
        """Serve a multi-token MoE request through Nectar Cache -> Waggle Prewarm -> LayerStoRm PCIe."""
        if self.is_preempted:
            raise RuntimeError("Node is preempted by active local game/3D application")

        cached = self.nectar_cache.get(tool_name=domain, query=query)
        if cached is not None:
            return {
                "status": "nectar_cache_hit",
                "domain": domain,
                "pcie_h2d_mib": 0.0,
                "ttft_ms": 0.45,
                "total_decode_ms": 0.45,
                "nectar_hash": cached.query_hash,
                "vram_allocated_mib": self.scheduler.vram_cache.budget.total_vram_mib,
            }

        if prewarm_on_waggle and self.scheduler.vram_cache.active_domain != domain:
            self.scheduler.vram_cache.prewarm_zone_a_from_waggle(domain=domain)

        step_telemetries: list[PCIeStepTelemetry] = []
        for trace in token_expert_traces:
            step_telemetries.append(
                self.scheduler.execute_layer_step(
                    activated_experts=trace,
                    use_waterline_overlap=use_waterline_overlap,
                )
            )

        total_h2d_mib = sum(s.pcie_h2d_mib for s in step_telemetries)
        ttft_ms = step_telemetries[0].total_step_ms if step_telemetries else 0.0
        total_decode_ms = sum(s.total_step_ms for s in step_telemetries)

        entry = self.nectar_cache.put(
            tool_name=domain,
            query=query,
            content={"completion": f"layerstorm_moe_completion:{domain}:{len(token_expert_traces)}_steps"},
            discovered_by_bee=discoverer_bee_id,
            nectar_richness=0.94,
        )

        return {
            "status": "layerstorm_pcie_streamed",
            "domain": domain,
            "pcie_h2d_mib": round(total_h2d_mib, 4),
            "ttft_ms": round(ttft_ms, 4),
            "total_decode_ms": round(total_decode_ms, 4),
            "nectar_hash": entry.query_hash,
            "vram_allocated_mib": self.scheduler.vram_cache.budget.total_vram_mib,
        }

    def trigger_gamer_preemption(self) -> dict[str, Any]:
        """Abort in-flight PCIe DMA queues and release Zone A/B VRAM in < 500 ms (P19c)."""
        t0 = time.perf_counter()
        self.is_preempted = True
        self.scheduler.vram_cache.clear_on_preempt()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_preemption_latency_ms = round(elapsed_ms, 4)
        return {
            "preempted": True,
            "yield_latency_ms": self.last_preemption_latency_ms,
            "sla_met_under_500ms": self.last_preemption_latency_ms < 500.0,
            "zone_a_remaining": len(self.scheduler.vram_cache.zone_a_experts),
            "zone_b_remaining": len(self.scheduler.vram_cache.zone_b_experts),
        }

    def resume_after_game(self) -> None:
        """Re-enable LayerStoRm PCIe serving once the local game exits."""
        self.is_preempted = False


class RealCUDAPCIeExpertStreamer:
    """Hardware-backed LayerStoRm Two-Zone PCIe H2D Expert Streamer (HM-S25-01 / E19).

    Allocates expert weights in pinned host RAM (`pin_memory=True` when CUDA is available)
    and streams misses over PCIe into a Two-Zone GPU tensor cache using a dedicated
    `torch.cuda.Stream()` overlapped with dense attention matmul compute. Falls back
    cleanly to CPU tensors when CUDA is not available.
    """

    def __init__(
        self,
        num_total_experts: int = 32,
        zone_a_slots: int = 8,
        zone_b_slots: int = 8,
        expert_rows: int = 512,
        expert_cols: int = 1024,
        device: str | None = None,
    ) -> None:
        if zone_a_slots < 0 or zone_b_slots < 0 or num_total_experts <= 0:
            raise ValueError("Invalid expert/slot counts for RealCUDAPCIeExpertStreamer")
        self.use_cuda = (device == "cuda") if device is not None else torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_total_experts = num_total_experts
        self.zone_a_slots = zone_a_slots
        self.zone_b_slots = zone_b_slots
        self.expert_rows = expert_rows
        self.expert_cols = expert_cols
        self.dtype = torch.float16 if self.use_cuda else torch.float32
        bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        self.expert_size_mib = (expert_rows * expert_cols * bytes_per_elem) / (1024.0 * 1024.0)

        g = torch.Generator(device="cpu").manual_seed(20261029)
        self.host_experts: dict[int, torch.Tensor] = {}
        for exp_id in range(num_total_experts):
            t = torch.randn((expert_rows, expert_cols), generator=g, dtype=self.dtype)
            if self.use_cuda:
                t = t.pin_memory()
            self.host_experts[exp_id] = t

        self.dense_attn_weight = torch.randn(
            (expert_rows, expert_rows), dtype=self.dtype, device=self.device
        )
        self.zone_a_gpu: dict[int, torch.Tensor] = {}
        self.zone_b_gpu: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.dma_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self.total_h2d_mib: float = 0.0
        self.zone_a_hits: int = 0
        self.zone_b_hits: int = 0
        self.pcie_misses: int = 0

    def reset_gpu_zones(self) -> None:
        """Clear Zone A and Zone B GPU caches while retaining pinned host RAM experts."""
        self.zone_a_gpu.clear()
        self.zone_b_gpu.clear()
        self.total_h2d_mib = 0.0
        self.zone_a_hits = 0
        self.zone_b_hits = 0
        self.pcie_misses = 0

    def prewarm_zone_a(self, domain: str, explicit_experts: list[int] | tuple[int, ...] | None = None) -> list[int]:
        """Pre-stage domain hot experts into Zone A GPU memory before token 1 arrives."""
        candidates = (
            list(explicit_experts)
            if explicit_experts is not None
            else list(DEFAULT_DOMAIN_HOT_EXPERTS.get(domain, ()))
        )
        selected = [e % self.num_total_experts for e in candidates[: self.zone_a_slots]]
        self.zone_a_gpu.clear()
        for exp_id in selected:
            self.zone_b_gpu.pop(exp_id, None)
            self.zone_a_gpu[exp_id] = self.host_experts[exp_id].to(
                device=self.device, non_blocking=self.use_cuda
            )
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        return selected

    def forward_step(
        self,
        hidden_state: torch.Tensor,
        activated_experts: list[int] | tuple[int, ...],
        overlap_dma_with_attention: bool = True,
    ) -> dict[str, Any]:
        """Execute one dense-attention + streamed-MoE step and return measured telemetry."""
        unique_experts = tuple(dict.fromkeys(e % self.num_total_experts for e in activated_experts))
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        streamed_ids: list[int] = []
        misses_to_fetch: list[int] = []
        for exp_id in unique_experts:
            if exp_id in self.zone_a_gpu:
                self.zone_a_hits += 1
            elif exp_id in self.zone_b_gpu:
                self.zone_b_hits += 1
                self.zone_b_gpu.move_to_end(exp_id)
            else:
                self.pcie_misses += 1
                misses_to_fetch.append(exp_id)

        if overlap_dma_with_attention and self.use_cuda and self.dma_stream is not None:
            with torch.cuda.stream(self.dma_stream):
                for exp_id in misses_to_fetch:
                    if self.zone_b_slots > 0 and len(self.zone_b_gpu) >= self.zone_b_slots:
                        self.zone_b_gpu.popitem(last=False)
                    self.zone_b_gpu[exp_id] = self.host_experts[exp_id].to(
                        device=self.device, non_blocking=True
                    )
                    streamed_ids.append(exp_id)
            attn_out = hidden_state @ self.dense_attn_weight
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
        else:
            attn_out = hidden_state @ self.dense_attn_weight
            if self.use_cuda:
                torch.cuda.synchronize(self.device)
            for exp_id in misses_to_fetch:
                if self.zone_b_slots > 0 and len(self.zone_b_gpu) >= self.zone_b_slots:
                    self.zone_b_gpu.popitem(last=False)
                self.zone_b_gpu[exp_id] = self.host_experts[exp_id].to(
                    device=self.device, non_blocking=False
                )
                streamed_ids.append(exp_id)

        moe_accum = torch.zeros(
            (hidden_state.shape[0], self.expert_cols), dtype=self.dtype, device=self.device
        )
        for exp_id in unique_experts:
            w = self.zone_a_gpu.get(exp_id)
            if w is None:
                w = self.zone_b_gpu[exp_id]
            moe_accum = moe_accum + (attn_out @ w)

        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        h2d_mib = len(streamed_ids) * self.expert_size_mib
        self.total_h2d_mib += h2d_mib

        return {
            "step_latency_ms": round(elapsed_ms, 4),
            "pcie_h2d_mib": round(h2d_mib, 4),
            "streamed_experts": tuple(streamed_ids),
            "output_norm": round(float(moe_accum.float().norm().item()), 4),
        }

    def preempt_and_release_vram(self) -> dict[str, Any]:
        """Drain CUDA stream and release Zone A & Zone B tensors in < 500 ms (P19c)."""
        t0 = time.perf_counter()
        if self.use_cuda and self.dma_stream is not None:
            self.dma_stream.synchronize()
        self.zone_a_gpu.clear()
        self.zone_b_gpu.clear()
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        yield_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "yield_latency_ms": round(yield_ms, 4),
            "sla_met_under_500ms": yield_ms < 500.0,
            "zone_a_remaining": len(self.zone_a_gpu),
            "zone_b_remaining": len(self.zone_b_gpu),
        }


class FoMoESkipAttentionLookaheadRouter:
    """FoMoE Skip-Attention router lookahead + Cache-Aware Routing (CAR) fallback (HM-S25-03).

    Predicts Layer L+1 top-k experts from Layer L pre-attention hidden states so that
    PCIe H2D DMA transfers run concurrently behind Layer L attention (PCIe Waterline
    Scheduler). When a post-attention router selects a non-resident expert whose router
    logit exceeds the best resident Zone A/B expert by at most ``car_margin_threshold``,
    CAR substitutes the resident expert to eliminate the PCIe stall without degrading
    routing margin.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_experts: int = 64,
        top_k: int = 4,
        car_margin_threshold: float = 0.35,
        seed: int = 42,
    ) -> None:
        import torch

        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.car_margin_threshold = float(car_margin_threshold)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.lookahead_proj = torch.randn(
            (self.hidden_dim, self.num_experts), generator=gen, dtype=torch.float32
        ) / (self.hidden_dim**0.5)

    def predict_next_layer_logits(self, pre_attn_hidden: Any) -> Any:
        """Compute Layer L+1 lookahead router logits from Layer L pre-attention states."""
        import torch

        h = pre_attn_hidden.detach().to(device="cpu", dtype=torch.float32)
        token_logits = h @ self.lookahead_proj
        if token_logits.ndim == 2:
            return token_logits.max(dim=0).values
        return token_logits.reshape(-1)

    def predict_next_layer_experts(
        self,
        pre_attn_hidden: Any,
        top_k: int | None = None,
    ) -> tuple[int, ...]:
        """Predict top-k experts for Layer L+1 prior to Layer L attention execution."""
        import torch

        k = self.top_k if top_k is None else int(top_k)
        logits = self.predict_next_layer_logits(pre_attn_hidden)
        topk_idx = torch.topk(logits, k=min(k, self.num_experts)).indices.tolist()
        return tuple(int(i) for i in topk_idx)

    def apply_cache_aware_routing(
        self,
        post_attn_logits: Any,
        resident_experts: Sequence[int] | set[int],
        top_k: int | None = None,
    ) -> tuple[tuple[int, ...], int]:
        """Apply Cache-Aware Routing (CAR) fallback on Layer L+1 post-attention logits.

        Returns ``(selected_experts, car_swaps_count)``.
        """
        import torch

        k = self.top_k if top_k is None else int(top_k)
        logits = post_attn_logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        raw_top = torch.topk(logits, k=min(k, self.num_experts)).indices.tolist()
        resident_set = {int(e) for e in resident_experts if 0 <= int(e) < self.num_experts}
        if not resident_set:
            return tuple(int(i) for i in raw_top), 0

        selected: list[int] = [int(i) for i in raw_top]
        used = set(selected)
        candidate_residents = sorted(
            (e for e in resident_set if e not in used),
            key=lambda e: float(logits[e].item()),
            reverse=True,
        )
        car_swaps = 0
        for idx, exp_id in enumerate(selected):
            if exp_id in resident_set or not candidate_residents:
                continue
            best_res = candidate_residents[0]
            margin = float(logits[exp_id].item() - logits[best_res].item())
            if margin <= self.car_margin_threshold:
                selected[idx] = best_res
                candidate_residents.pop(0)
                car_swaps += 1
        return tuple(selected), car_swaps

    def run_waterline_schedule(
        self,
        scheduler: PCIeWaterlineStreamScheduler,
        pre_attn_hiddens: Sequence[Any],
        residual_perturbations: Sequence[Any] | None = None,
        use_skip_attention_lookahead: bool = True,
        use_car_fallback: bool = True,
    ) -> dict[str, Any]:
        """Execute a multi-layer MoE decode pipeline with FoMoE lookahead + CAR fallback.

        Evaluates HM-S25-03 gate: PCIe transfer stall ratio < 15% of total step time.
        """
        import torch

        total_step_ms = 0.0
        total_stall_ms = 0.0
        total_h2d_mib = 0.0
        total_car_swaps = 0
        lookahead_hits = 0
        lookahead_total = 0

        for layer_idx, h_pre in enumerate(pre_attn_hiddens):
            pred_logits = self.predict_next_layer_logits(h_pre)
            if residual_perturbations is not None and layer_idx < len(residual_perturbations):
                noise = residual_perturbations[layer_idx].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                true_logits = pred_logits + noise
            else:
                true_logits = pred_logits

            raw_predicted_experts = self.predict_next_layer_experts(h_pre)
            resident_before = set(scheduler.vram_cache.zone_a_experts) | set(
                scheduler.vram_cache.zone_b_experts
            )
            if use_car_fallback:
                predicted_experts, pre_swaps = self.apply_cache_aware_routing(
                    pred_logits, resident_experts=resident_before
                )
                total_car_swaps += pre_swaps
            else:
                predicted_experts = raw_predicted_experts

            newly_prefetched: list[int] = []
            if use_skip_attention_lookahead:
                # Prefetch predicted experts into Zone B during Layer L attention
                for exp_id in predicted_experts:
                    if exp_id not in resident_before:
                        scheduler.vram_cache.access_expert(exp_id)
                        newly_prefetched.append(exp_id)

            resident_now = set(scheduler.vram_cache.zone_a_experts) | set(
                scheduler.vram_cache.zone_b_experts
            )
            raw_true_top = tuple(
                int(i) for i in torch.topk(true_logits, k=min(self.top_k, self.num_experts)).indices.tolist()
            )
            lookahead_hits += len(set(raw_predicted_experts) & set(raw_true_top))
            lookahead_total += len(raw_true_top)

            if use_car_fallback:
                final_experts, swaps = self.apply_cache_aware_routing(
                    true_logits, resident_experts=resident_now
                )
                total_car_swaps += swaps
            else:
                final_experts = raw_true_top

            step = scheduler.execute_layer_step(
                final_experts,
                use_waterline_overlap=use_skip_attention_lookahead,
            )
            if use_skip_attention_lookahead:
                # Prefetched transfers overlapped with layer_attention_ms * fomoe_speculative_overlap_ratio
                prefetch_mib = len(newly_prefetched) * scheduler.vram_cache.budget.expert_size_mib
                prefetch_ms = (
                    prefetch_mib / scheduler.pcie_bandwidth_mib_per_ms if prefetch_mib > 0 else 0.0
                )
                overlapped_window_ms = (
                    scheduler.layer_attention_ms * scheduler.fomoe_speculative_overlap_ratio
                )
                extra_stall = max(0.0, prefetch_ms - overlapped_window_ms)
                step_stall_ms = step.overlapped_stall_ms + extra_stall
                step_total_ms = step.total_step_ms + extra_stall
                step_h2d_mib = step.pcie_h2d_mib + prefetch_mib
            else:
                step_stall_ms = step.overlapped_stall_ms
                step_total_ms = step.total_step_ms
                step_h2d_mib = step.pcie_h2d_mib

            total_step_ms += step_total_ms
            total_stall_ms += step_stall_ms
            total_h2d_mib += step_h2d_mib

        stall_ratio_pct = (100.0 * total_stall_ms / total_step_ms) if total_step_ms > 0.0 else 0.0
        lookahead_recall = (lookahead_hits / lookahead_total) if lookahead_total > 0 else 1.0
        return {
            "layers_evaluated": len(pre_attn_hiddens),
            "total_step_ms": round(total_step_ms, 4),
            "total_pcie_stall_ms": round(total_stall_ms, 4),
            "pcie_stall_ratio_pct": round(stall_ratio_pct, 4),
            "stall_gate_under_15pct": stall_ratio_pct < 15.0,
            "lookahead_recall": round(lookahead_recall, 4),
            "car_swaps": total_car_swaps,
            "total_pcie_h2d_mib": round(total_h2d_mib, 4),
        }


class HybridHiSparseE8KVCache:
    """Hybrid HiSparse + Gosset E8 Lattice KV Cache for 32k-128k contexts (HM-S25-04).

    Combines HiSparse hierarchical KV eviction (sink tokens + local sliding window +
    heavy-hitter tokens retained in hot VRAM; cold tokens offloaded to pinned host RAM)
    with HiveMind's Gosset E8 lattice vector quantizer (Conway & Sloane 1982,
    ``src/hivemind/compress/lattice_quantizer.py``) on the hot VRAM tier.
    """

    def __init__(
        self,
        sink_tokens: int = 32,
        sliding_window_tokens: int = 256,
        heavy_hitter_tokens: int = 256,
        e8_scale: float = 0.08,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
    ) -> None:
        from hivemind.compress.lattice_quantizer import LatticeDeltaQuantizer

        self.sink_tokens = int(sink_tokens)
        self.sliding_window_tokens = int(sliding_window_tokens)
        self.heavy_hitter_tokens = int(heavy_hitter_tokens)
        self.e8_scale = float(e8_scale)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.quantizer = LatticeDeltaQuantizer("E8", scale=self.e8_scale)

    @property
    def max_hot_vram_tokens(self) -> int:
        return self.sink_tokens + self.sliding_window_tokens + self.heavy_hitter_tokens

    def estimate_vram_kv_mib(self, seq_len: int, bits_per_dim: float = 4.0) -> dict[str, float]:
        """Compute FP16 full-VRAM KV footprint vs Hybrid HiSparse + E8 hot-VRAM footprint."""
        total_tokens = max(1, int(seq_len))
        hot_tokens = min(total_tokens, self.max_hot_vram_tokens)
        cold_host_tokens = max(0, total_tokens - hot_tokens)
        # 2 (K and V) * num_layers * num_kv_heads * head_dim
        dims_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim
        fp16_full_mib = (total_tokens * dims_per_token * 2.0) / (1024.0 * 1024.0)
        e8_hot_vram_mib = (hot_tokens * dims_per_token * (bits_per_dim / 8.0)) / (1024.0 * 1024.0)
        cold_host_fp16_mib = (cold_host_tokens * dims_per_token * 2.0) / (1024.0 * 1024.0)
        return {
            "seq_len": float(total_tokens),
            "hot_vram_tokens": float(hot_tokens),
            "cold_host_tokens": float(cold_host_tokens),
            "fp16_full_vram_mib": round(fp16_full_mib, 4),
            "hisparse_e8_vram_mib": round(e8_hot_vram_mib, 4),
            "cold_host_pinned_mib": round(cold_host_fp16_mib, 4),
            "vram_compression_ratio": round(fp16_full_mib / max(e8_hot_vram_mib, 1e-9), 2),
        }

    def select_hot_indices(
        self,
        attn_weights: Any,
        random_control: bool = False,
        seed: int = 42,
    ) -> Any:
        """Select hot VRAM token indices using HiSparse (sink + window + heavy hitters) or random control."""
        import torch

        seq_len = int(attn_weights.shape[-1])
        budget = min(seq_len, self.max_hot_vram_tokens)
        if seq_len <= budget:
            return torch.arange(seq_len, device=attn_weights.device)

        if random_control:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(seq_len, generator=gen)[:budget]
            return torch.sort(perm).values.to(attn_weights.device)

        sink_end = min(self.sink_tokens, seq_len)
        win_start = max(sink_end, seq_len - self.sliding_window_tokens)
        sink_idx = torch.arange(0, sink_end, device=attn_weights.device)
        win_idx = torch.arange(win_start, seq_len, device=attn_weights.device)

        middle_len = win_start - sink_end
        hh_k = min(self.heavy_hitter_tokens, max(0, middle_len))
        if hh_k > 0:
            scores = attn_weights[..., sink_end:win_start]
            while scores.ndim > 1:
                scores = scores.sum(dim=0)
            top_mid = torch.topk(scores, k=hh_k).indices + sink_end
            combined = torch.cat([sink_idx, top_mid, win_idx], dim=0)
        else:
            combined = torch.cat([sink_idx, win_idx], dim=0)
        return torch.unique(torch.sort(combined).values)

    def evaluate_attention_fidelity(
        self,
        query: Any,
        key: Any,
        value: Any,
        random_control: bool = False,
    ) -> dict[str, Any]:
        """Compare exact full-context attention against Hybrid HiSparse + E8 Lattice KV Cache.

        Verifies HM-S25-04 gate: relative attention / NLL proxy delta < 0.05 on long contexts,
        while the planted random-eviction negative control degrades sharply.
        """
        import math
        import torch

        q = query.detach().to(dtype=torch.float32)
        k = key.detach().to(dtype=torch.float32)
        v = value.detach().to(dtype=torch.float32)
        d_k = q.shape[-1]

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        full_probs = torch.softmax(logits, dim=-1)
        exact_out = full_probs @ v

        hot_idx = self.select_hot_indices(full_probs, random_control=random_control)
        k_hot = k.index_select(dim=-2, index=hot_idx)
        v_hot = v.index_select(dim=-2, index=hot_idx)

        # Quantize hot KV tokens on Gosset E8 lattice (Conway & Sloane 1982)
        q_k = self.quantizer.quantize(k_hot.cpu())
        q_v = self.quantizer.quantize(v_hot.cpu())
        k_hot_e8 = q_k.dequantized.reshape(k_hot.shape).to(q.device)
        v_hot_e8 = q_v.dequantized.reshape(v_hot.shape).to(q.device)

        logits_sparse = (q @ k_hot_e8.transpose(-2, -1)) / math.sqrt(d_k)
        sparse_probs = torch.softmax(logits_sparse, dim=-1)
        approx_out = sparse_probs @ v_hot_e8

        diff_norm = float(torch.norm(exact_out - approx_out).item())
        base_norm = max(float(torch.norm(exact_out).item()), 1e-9)
        rel_l2_delta = diff_norm / base_norm
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                exact_out.reshape(1, -1), approx_out.reshape(1, -1)
            ).item()
        )
        # Retained attention mass on hot indices
        retained_mass = float(full_probs.index_select(dim=-1, index=hot_idx).sum(dim=-1).mean().item())
        return {
            "seq_len": int(k.shape[-2]),
            "hot_vram_tokens": int(hot_idx.numel()),
            "random_control": bool(random_control),
            "retained_attention_mass": round(retained_mass, 4),
            "e8_rate_bits_per_dim": round(0.5 * (q_k.rate_bits_per_dim + q_v.rate_bits_per_dim), 3),
            "relative_l2_delta": round(rel_l2_delta, 4),
            "cosine_similarity": round(cos_sim, 5),
            "passed_delta_under_0_05": rel_l2_delta < 0.05,
        }


def build_layerstorm_byob_preset(
    vram_gb: float = 8.0,
    domain: str = "systems_deeptech",
    context_length: int = 32768,
) -> dict[str, Any]:
    """Build the BYOB ``layerstorm-frontier-moe`` preset for consumer 8 GB / 16 GB GPUs (HM-S25-05)."""
    vram_cap_mib = 6656.0 if vram_gb <= 8.5 else round(vram_gb * 1024.0 * 0.85, 1)
    kv_cache = HybridHiSparseE8KVCache(
        sink_tokens=32,
        sliding_window_tokens=256,
        heavy_hitter_tokens=256,
    )
    kv_est = kv_cache.estimate_vram_kv_mib(context_length)
    kv_budget_mib = max(256.0, min(1024.0, round(kv_est["hisparse_e8_vram_mib"] + 256.0, 1)))
    zone_a_slots = 8 if vram_gb <= 8.5 else 16
    zone_b_slots = 12 if vram_gb <= 8.5 else 24

    budget = TwoZoneVRAMBudget(
        vram_cap_mib=vram_cap_mib,
        dense_attention_mib=2400.0,
        kv_cache_mib=kv_budget_mib,
        expert_size_mib=128.0,
        zone_a_slots=zone_a_slots,
        zone_b_slots=zone_b_slots,
    )
    budget.validate()
    return {
        "preset_id": "layerstorm-frontier-moe",
        "tier": "moe-layerstorm",
        "virtual_model_id": "hivemind/moe-layerstorm",
        "backend": "layerstorm-pcie",
        "target_model_family": "Qwen3-30B-A3B-Instruct-2507",
        "domain": domain,
        "context_length": int(context_length),
        "vram_budget": {
            "vram_cap_mib": budget.vram_cap_mib,
            "total_vram_mib": budget.total_vram_mib,
            "headroom_mib": budget.headroom_mib,
            "zone_a_slots": budget.zone_a_slots,
            "zone_b_slots": budget.zone_b_slots,
            "kv_cache_mib": budget.kv_cache_mib,
        },
        "hisparse_e8_kv": kv_est,
        "gamer_preemption_sla_ms": 500.0,
    }

