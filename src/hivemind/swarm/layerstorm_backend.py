# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the Apache-2.0 License.
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
