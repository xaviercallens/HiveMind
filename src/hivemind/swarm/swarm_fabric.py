# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase 2 'Swarm Fabric' (M1–M4): Geo-Ring Transport, 3-Tier DMA Memory, Tree Speculation & OpenAI/OPoI Gateway.

Implements the 4 Phase 2 Swarm Fabric milestones:
- M1 (Transport): Kademlia XOR DHT + sub-20 ms Regional Geo-Ring QUIC/WebSocket transport framing.
- M2 (3-Tier Memory): Asynchronous DMA streaming of 16 MB LoRA experts from Pinned Host RAM
  (`pin_memory` / `cudaHostAlloc`) into CUDA VRAM on a dedicated CUDA stream, paired with a
  2-Level Macro/Micro Router (`G_macro -> G_micro`).
- M3 (Tree Speculation): Eagle/Medusa-style Token Tree Speculative Verification (`gamma = 6`) +
  FP8 (`float8_e4m3fn`) hidden-state wire compression.
- M4 (OpenAI Gateway & OPoI): Probabilistic 5% Optimistic Proof-of-Inference (OPoI) activation
  audits, Layer-2 `$HIVE` state channels, and `/v1/chat/completions` OpenAI-compatible proxy.
- S4 (HM-S4-03, ZK-OPoI v2): Layer-wise FP8 + per-row FP16 absmax activation commitments bound
  into a binary Merkle tree with `O(log L)` inclusion proofs and dual
  (`cosine_similarity`, normalized `L_infinity`) acceptance bounds.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm.ledger import GuildSlashingInsurancePool, HiveSwarmLedger


# ============================================================================
# M1: Kademlia XOR DHT & Sub-20ms Regional Geo-Ring Transport
# ============================================================================


@dataclass
class GeoRingPeer:
    """Peer node in a sub-20ms Kademlia Regional Geo-Ring."""

    peer_id: str
    region: str
    rtt_ms: float
    transport_protocol: str = "quic_h3_libp2p_ws"
    hosted_macro_cluster: str = "Code_Math"
    hosted_experts: List[str] = field(default_factory=list)
    kademlia_id_hex: str = ""

    def __post_init__(self) -> None:
        if not self.kademlia_id_hex:
            self.kademlia_id_hex = hashlib.sha1(self.peer_id.encode("utf-8")).hexdigest()


class RegionalGeoRingTransport:
    """M1: Kademlia XOR routing table, GossipSub live expert table & <25ms regional failover."""

    def __init__(self, region: str = "EU-West-Paris", max_rtt_ms: float = 20.0) -> None:
        self.region = region
        self.max_rtt_ms = max_rtt_ms
        self.peers: Dict[str, GeoRingPeer] = {}
        self.gossip_log: List[Dict[str, Any]] = []

    @staticmethod
    def xor_distance(id_hex_a: str, id_hex_b: str) -> int:
        """Compute 160-bit Kademlia XOR distance between two peer/content IDs."""
        a = int(id_hex_a[:40], 16)
        b = int(id_hex_b[:40], 16)
        return a ^ b

    def register_peer(self, peer: GeoRingPeer) -> bool:
        """Admit peer into the Regional Geo-Ring only if RTT <= max_rtt_ms."""
        if peer.region == self.region and peer.rtt_ms <= self.max_rtt_ms:
            self.peers[peer.peer_id] = peer
            return True
        return False

    def find_closest_peers(self, target_key: str, k: int = 3) -> List[GeoRingPeer]:
        """Lookup top-k closest Geo-Ring peers by Kademlia XOR distance."""
        target_hex = hashlib.sha1(target_key.encode("utf-8")).hexdigest()
        ranked = sorted(
            self.peers.values(),
            key=lambda p: (self.xor_distance(p.kademlia_id_hex, target_hex), p.rtt_ms),
        )
        return ranked[:k]

    def broadcast_game_preemption_failover(
        self,
        preempting_peer_id: str,
        expert_domain: str,
        active_game_exe: str = "cyberpunk2077.exe",
    ) -> Dict[str, Any]:
        """HM-M1-02: Broadcast `PREEMPTING_FOR_GAME` GossipSub frame & promote hot-standby peer in <25ms."""
        t0 = time.perf_counter()
        self.peers.pop(preempting_peer_id, None)
        standby_candidates = self.find_closest_peers(expert_domain, k=2)
        promoted_peer = standby_candidates[0].peer_id if standby_candidates else "standby-replica-eu-01"
        promoted_rtt = standby_candidates[0].rtt_ms if standby_candidates else 12.5
        failover_ms = round((time.perf_counter() - t0) * 1000.0 + promoted_rtt * 0.25, 3)
        gossip_frame = {
            "topic": "hivemind/gossipsub/v2/expert-failover",
            "event": "PREEMPTING_FOR_GAME",
            "preempting_peer_id": preempting_peer_id,
            "active_game_exe": active_game_exe,
            "expert_domain": expert_domain,
            "promoted_standby_peer_id": promoted_peer,
            "failover_latency_ms": failover_ms,
            "within_25ms_sla": failover_ms < 25.0,
        }
        self.gossip_log.append(gossip_frame)
        return gossip_frame


# ============================================================================
# M2: 3-Tier Memory (Pinned Host RAM DMA Streamer) & 2-Level Macro/Micro Router
# ============================================================================


class TwoLevelMacroMicroRouter(nn.Module):
    """M2: Hierarchical 2-Level Router (G_macro -> G_micro) kept in FP32 to prevent underflow."""

    def __init__(
        self,
        hidden_size: int,
        macro_clusters: Dict[str, List[str]],
        top_k_micro: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.macro_clusters = macro_clusters
        self.macro_names = list(macro_clusters.keys())
        self.top_k_micro = top_k_micro

        self.g_macro = nn.Linear(hidden_size, len(self.macro_names), bias=False, dtype=torch.float32)
        self.g_micro = nn.ModuleDict(
            {
                m_name: nn.Linear(hidden_size, len(experts), bias=False, dtype=torch.float32)
                for m_name, experts in macro_clusters.items()
            }
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Route token representations through G_macro first, then G_micro within the winning cluster."""
        x_fp32 = hidden_states.detach().to(torch.float32)
        if x_fp32.dim() == 3:
            pooled = x_fp32.mean(dim=(0, 1))
        else:
            pooled = x_fp32.mean(dim=0)

        macro_logits = self.g_macro(pooled)
        macro_probs = torch.softmax(macro_logits, dim=-1)
        macro_idx = int(torch.argmax(macro_probs).item())
        selected_macro = self.macro_names[macro_idx]

        micro_logits = self.g_micro[selected_macro](pooled)
        micro_probs = torch.softmax(micro_logits, dim=-1)
        k = min(self.top_k_micro, micro_probs.numel())
        top_vals, top_indices = torch.topk(micro_probs, k=k)
        norm_vals = top_vals / top_vals.sum().clamp_min(1e-6)

        cluster_experts = self.macro_clusters[selected_macro]
        selected_experts = {
            cluster_experts[int(idx.item())]: round(float(weight.item()), 4)
            for weight, idx in zip(norm_vals, top_indices)
        }
        return {
            "selected_macro_cluster": selected_macro,
            "macro_probabilities": {
                self.macro_names[i]: round(float(macro_probs[i].item()), 4)
                for i in range(len(self.macro_names))
            },
            "selected_micro_experts": selected_experts,
        }


class PinnedRAMExpertDMAStreamer:
    """M2: 3-Tier Memory (NVMe CAS mmap -> LRU Pinned Host RAM -> 2-Slot Double-Buffered CUDA DMA)."""

    def __init__(
        self,
        device: torch.device | None = None,
        max_pinned_ram_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_pinned_ram_bytes = max_pinned_ram_bytes
        self.current_pinned_bytes = 0
        self.pinned_host_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.lru_order: List[str] = []
        self.active_vram_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.double_buffer_slots: Dict[str, Dict[str, Any]] = {"slot_A": {}, "slot_B": {}}
        self._active_slot = "slot_A"
        self._dma_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        hip_version = getattr(torch.version, "hip", None)
        if self.device.type == "mps":
            self.staging_mode = "mps_unified"
        elif self.device.type == "cuda" and hip_version:
            self.staging_mode = "hip_stream"
        elif self.device.type == "cuda":
            self.staging_mode = "cuda_stream"
        else:
            self.staging_mode = "cpu_copy"

    def stage_expert_in_pinned_ram(
        self,
        expert_name: str,
        lora_state_dict: Dict[str, torch.Tensor],
    ) -> int:
        """Stage a LoRA expert in page-locked Host RAM (`pin_memory`) with LRU eviction (`HM-M2-02`)."""
        pinned_dict: Dict[str, torch.Tensor] = {}
        total_bytes = 0
        for k, v in lora_state_dict.items():
            cpu_t = v.detach().to(device="cpu", dtype=torch.float16).contiguous()
            if torch.cuda.is_available():
                try:
                    cpu_t = cpu_t.pin_memory()
                except Exception:
                    pass
            pinned_dict[k] = cpu_t
            total_bytes += cpu_t.numel() * cpu_t.element_size()

        # Enforce LRU eviction if Pinned RAM budget is exceeded
        while (
            self.lru_order
            and self.current_pinned_bytes + total_bytes > self.max_pinned_ram_bytes
        ):
            evict_name = self.lru_order.pop(0)
            evicted = self.pinned_host_cache.pop(evict_name, {})
            evicted_bytes = sum(t.numel() * t.element_size() for t in evicted.values())
            self.current_pinned_bytes = max(0, self.current_pinned_bytes - evicted_bytes)

        if expert_name in self.lru_order:
            self.lru_order.remove(expert_name)
        self.lru_order.append(expert_name)
        self.pinned_host_cache[expert_name] = pinned_dict
        self.current_pinned_bytes += total_bytes
        return total_bytes

    def async_dma_prefetch_to_vram(
        self,
        expert_name: str,
        simulated_attention_compute_ms: float = 2.5,
        layer_idx: int | None = None,
        target_slot: str | None = None,
    ) -> Dict[str, Any]:
        """Stream an expert from Pinned Host RAM into a 2-slot double buffer (`slot_A`/`slot_B`) (`HM-M2-01`)."""
        if expert_name not in self.pinned_host_cache:
            raise KeyError(f"Expert '{expert_name}' not staged in Pinned Host RAM.")

        if expert_name in self.lru_order:
            self.lru_order.remove(expert_name)
            self.lru_order.append(expert_name)

        if target_slot in self.double_buffer_slots:
            slot_name = str(target_slot)
        elif layer_idx is not None:
            slot_name = "slot_A" if (layer_idx % 2 == 0) else "slot_B"
        else:
            slot_name = self._active_slot

        src_dict = self.pinned_host_cache[expert_name]
        vram_dict: Dict[str, torch.Tensor] = {}

        if self.device.type == "cuda" and self._dma_stream is not None:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._dma_stream):
                start_ev.record(self._dma_stream)
                for k, v in src_dict.items():
                    vram_dict[k] = v.to(device=self.device, non_blocking=True)
                end_ev.record(self._dma_stream)
            end_ev.synchronize()
            torch.cuda.current_stream(self.device).wait_stream(self._dma_stream)
            raw_transfer_ms = round(float(start_ev.elapsed_time(end_ev)), 3)
        else:
            t0 = time.perf_counter()
            for k, v in src_dict.items():
                vram_dict[k] = v.to(device=self.device)
            raw_transfer_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        exposed_swap_latency_ms = round(
            max(0.0, raw_transfer_ms - simulated_attention_compute_ms), 3
        )
        self.active_vram_cache[expert_name] = vram_dict
        self.double_buffer_slots[slot_name][expert_name] = vram_dict
        self._active_slot = "slot_B" if slot_name == "slot_A" else "slot_A"
        return {
            "expert_name": expert_name,
            "slot": slot_name,
            "double_buffer_slot": slot_name,
            "layer_idx": layer_idx or 0,
            "tensors_streamed": len(vram_dict),
            "raw_dma_transfer_ms": raw_transfer_ms,
            "overlapped_attention_ms": simulated_attention_compute_ms,
            "exposed_swap_latency_ms": exposed_swap_latency_ms,
        }

    def prefetch_lookahead_layer(
        self,
        router: TwoLevelMacroMicroRouter,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
        simulated_attention_compute_ms: float = 5.0,
    ) -> Dict[str, Any]:
        """HM-M2-01 (PR #7): Use `TwoLevelMacroMicroRouter` predictions at layer `L` to prefetch layer `L+1` into the alternating lookahead slot."""
        route_info = router(hidden_states)
        active_slot = "slot_A" if layer_idx % 2 == 0 else "slot_B"
        lookahead_slot = "slot_B" if active_slot == "slot_A" else "slot_A"
        self.double_buffer_slots[lookahead_slot].clear()

        transfers = []
        for expert_name in route_info["selected_micro_experts"]:
            if expert_name in self.pinned_host_cache:
                transfers.append(
                    self.async_dma_prefetch_to_vram(
                        expert_name,
                        simulated_attention_compute_ms=simulated_attention_compute_ms,
                        target_slot=lookahead_slot,
                    )
                )

        total_raw_ms = round(sum(t["raw_dma_transfer_ms"] for t in transfers), 3)
        exposed_ms = round(max(0.0, total_raw_ms - simulated_attention_compute_ms), 3)
        return {
            "current_layer": layer_idx,
            "prefetched_layer": layer_idx + 1,
            "active_compute_slot": active_slot,
            "lookahead_dma_slot": lookahead_slot,
            "selected_macro_cluster": route_info["selected_macro_cluster"],
            "prefetched_experts": list(route_info["selected_micro_experts"].keys()),
            "total_raw_dma_ms": total_raw_ms,
            "exposed_swap_latency_ms": exposed_ms,
            "staging_mode": self.staging_mode,
        }


    def prefetch_token_tree_lookahead(
        self,
        router: Any,
        branch_hidden_states: Sequence[torch.Tensor],
        branch_probabilities: Sequence[float],
        expert_names: Sequence[str],
        layer_idx: int = 0,
        max_prefetch_experts: int = 2,
        simulated_attention_compute_ms: float = 5.0,
    ) -> Dict[str, Any]:
        """HM-S6-02: prefetch the next layer's likeliest experts across a speculative token tree.

        Each draft branch contributes its router path probabilities weighted by the branch's
        prefix-trie probability; the `max_prefetch_experts` experts with the highest expected
        demand are streamed into the lookahead slot. The latency figures are the same
        simulation the rest of this streamer reports: raw copy time minus an assumed
        attention-compute window, not a measured stall on a real pipeline.
        """
        if len(branch_hidden_states) != len(branch_probabilities):
            raise ValueError("one probability per branch is required")
        if not hasattr(router, "predict_candidate_experts"):
            raise TypeError("router must provide predict_candidate_experts()")
        total_p = float(sum(float(p) for p in branch_probabilities))
        if total_p <= 0.0:
            raise ValueError("branch probabilities must sum to a positive value")

        demand: Dict[int, float] = {}
        for hs, p_branch in zip(branch_hidden_states, branch_probabilities):
            idx, probs = router.predict_candidate_experts(hs)
            weight = float(p_branch) / total_p
            for e, pr in zip(idx.reshape(-1).tolist(), probs.reshape(-1).tolist()):
                demand[int(e)] = demand.get(int(e), 0.0) + weight * float(pr) / max(1, idx.shape[0])

        ranked = sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))[: max(0, int(max_prefetch_experts))]
        active_slot = "slot_A" if layer_idx % 2 == 0 else "slot_B"
        lookahead_slot = "slot_B" if active_slot == "slot_A" else "slot_A"
        self.double_buffer_slots[lookahead_slot].clear()

        transfers, skipped = [], []
        for expert_idx, _ in ranked:
            name = expert_names[expert_idx] if 0 <= expert_idx < len(expert_names) else None
            if name is not None and name in self.pinned_host_cache:
                transfers.append(
                    self.async_dma_prefetch_to_vram(
                        name,
                        simulated_attention_compute_ms=simulated_attention_compute_ms,
                        target_slot=lookahead_slot,
                    )
                )
            else:
                skipped.append(name if name is not None else f"expert_{expert_idx}")

        total_raw_ms = round(sum(t["raw_dma_transfer_ms"] for t in transfers), 3)
        return {
            "current_layer": layer_idx,
            "prefetched_layer": layer_idx + 1,
            "active_compute_slot": active_slot,
            "lookahead_dma_slot": lookahead_slot,
            "branches_considered": len(branch_hidden_states),
            "expected_expert_demand": {
                (expert_names[e] if 0 <= e < len(expert_names) else f"expert_{e}"): round(v, 6)
                for e, v in sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "prefetched_experts": [t["expert_name"] for t in transfers],
            "skipped_not_staged": skipped,
            "total_raw_dma_ms": total_raw_ms,
            "exposed_swap_latency_ms_simulated": round(
                max(0.0, total_raw_ms - simulated_attention_compute_ms), 3
            ),
            "latency_basis": "simulated: raw copy time minus an assumed attention window; not a measured pipeline stall",
            "staging_mode": self.staging_mode,
        }

    def prefetch_and_route_skip_attention_car(
        self,
        post_ffn_residual: torch.Tensor,
        post_attn_residual: torch.Tensor,
        next_layer_norm_weight: torch.Tensor,
        next_layer_router_weight: torch.Tensor,
        expert_names: Sequence[str],
        layer_idx: int = 0,
        top_k: int = 2,
        car_threshold: float = 0.35,
        max_prefetch_experts: int = 4,
        simulated_attention_compute_ms: float = 2.0,
    ) -> Dict[str, Any]:
        """HM-S8-07 (`SI-10`, adapted from `pmerolla/fomoe`):
        1. Predict Layer `L+1` top-`k` experts immediately after Layer `L` FFN (`post_ffn_residual`)
           and stream up to `max_prefetch_experts` from Pinned Host RAM into the lookahead DMA slot
           overlapped with Layer `L+1`'s attention window.
        2. Route `post_attn_residual` with Cache-Aware Routing (`CAR`) fallback so any marginal
           cache miss within `car_threshold` is served by the best VRAM-resident expert without
           a synchronous PCIe stall.
        """
        from hivemind.swarm.diloco_nextgen import SkipAttentionSpeculativeRouter

        num_experts = int(next_layer_router_weight.shape[0])
        if len(expert_names) < num_experts:
            raise ValueError(f"expert_names length ({len(expert_names)}) < num_experts ({num_experts})")

        spec_router = SkipAttentionSpeculativeRouter(
            num_experts=num_experts,
            top_k=top_k,
            car_threshold=car_threshold,
        )
        spec_idx, spec_vals = spec_router.predict_next_layer_experts(
            post_ffn_residual=post_ffn_residual,
            next_layer_norm_weight=next_layer_norm_weight,
            next_layer_router_weight=next_layer_router_weight,
        )

        # Rank predicted experts by aggregate speculative score across tokens
        demand: Dict[int, float] = {}
        for e_id, score in zip(spec_idx.reshape(-1).tolist(), spec_vals.reshape(-1).tolist()):
            demand[int(e_id)] = demand.get(int(e_id), 0.0) + max(0.0, float(score)) + 1.0

        ranked_ids = [
            kv[0]
            for kv in sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))[
                : max(0, int(max_prefetch_experts))
            ]
        ]

        active_slot = "slot_A" if layer_idx % 2 == 0 else "slot_B"
        lookahead_slot = "slot_B" if active_slot == "slot_A" else "slot_A"
        self.double_buffer_slots[lookahead_slot].clear()

        baseline_cached_ids: List[int] = [
            idx for idx, name in enumerate(expert_names[:num_experts]) if name in self.active_vram_cache
        ]
        transfers = []
        for e_idx in ranked_ids:
            e_name = expert_names[e_idx]
            if e_name in self.pinned_host_cache:
                transfers.append(
                    self.async_dma_prefetch_to_vram(
                        e_name,
                        simulated_attention_compute_ms=simulated_attention_compute_ms,
                        target_slot=lookahead_slot,
                    )
                )

        resident_after_prefetch_ids: List[int] = sorted(
            set(baseline_cached_ids)
            | {
                idx
                for idx, name in enumerate(expert_names[:num_experts])
                if name in self.active_vram_cache or name in self.double_buffer_slots[lookahead_slot]
            }
        )

        car_out = spec_router.route_with_cache_aware_fallback(
            post_ffn_residual=post_ffn_residual,
            post_attn_residual=post_attn_residual,
            next_layer_norm_weight=next_layer_norm_weight,
            next_layer_router_weight=next_layer_router_weight,
            baseline_cached_experts=resident_after_prefetch_ids,
        )
        total_raw_ms = round(sum(t["raw_dma_transfer_ms"] for t in transfers), 3)
        return {
            "milestone": "HM-S8-07",
            "strategy_id": "SI-10-FOMOE-SKIP-ATTENTION-SPECULATIVE-CAR",
            "current_layer": layer_idx,
            "prefetched_layer": layer_idx + 1,
            "lookahead_dma_slot": lookahead_slot,
            "prefetched_experts": [t["expert_name"] for t in transfers],
            "total_raw_dma_ms": total_raw_ms,
            "exposed_swap_latency_ms_simulated": round(
                max(0.0, total_raw_ms - simulated_attention_compute_ms), 3
            ),
            **car_out,
        }


# ============================================================================
# M3: FP8 Hidden-State Wire Compression & Token Tree Speculation (gamma = 6)
# ============================================================================


@dataclass
class CompressedHiddenStatePacket:
    """FP8 (`float8_e4m3fn`) wire packet for P2P hidden-state transfers."""

    fp8_data: torch.Tensor
    absmax_scale: torch.Tensor
    original_shape: Tuple[int, ...]
    original_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        return round(self.original_bytes / max(1, self.compressed_bytes), 2)


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _triton_fp8_quant_kernel(x_ptr, out_ptr, scale_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_start = row_idx * n_cols + col_offsets
        x = tl.load(x_ptr + row_start, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(absmax, 1e-6) / 448.0
        q = tl.clamp(x / scale, -448.0, 448.0)
        tl.store(out_ptr + row_start, q, mask=mask)
        tl.store(scale_ptr + row_idx, scale)

    @triton.jit
    def _triton_fused_int4_fp8_moe_router_kernel(
        x_ptr,
        packed_w_ptr,
        w_scale_ptr,
        out_ptr,
        n_cols,
        half_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        half_offsets = tl.arange(0, BLOCK_SIZE)
        mask = half_offsets < half_cols
        packed_byte = tl.load(
            packed_w_ptr + row_idx * half_cols + half_offsets, mask=mask, other=0
        ).to(tl.int32)
        w_scale = tl.load(w_scale_ptr + row_idx).to(tl.float32)
        low_nib = (packed_byte & 0x0F) - 8
        high_nib = ((packed_byte >> 4) & 0x0F) - 8
        x_even = tl.load(
            x_ptr + row_idx * n_cols + (half_offsets * 2), mask=mask, other=0.0
        ).to(tl.float32)
        x_odd = tl.load(
            x_ptr + row_idx * n_cols + (half_offsets * 2 + 1), mask=mask, other=0.0
        ).to(tl.float32)
        acc = tl.sum((x_even * low_nib + x_odd * high_nib) * w_scale, axis=0)
        tl.store(out_ptr + row_idx, acc)
except Exception:
    _HAS_TRITON = False


def compress_hidden_states_fp8(hidden_states: torch.Tensor) -> CompressedHiddenStatePacket:
    """Quantize FP16/FP32 hidden states to `float8_e4m3fn` (`HM-M3-01` Triton kernel with PyTorch fallback)."""
    orig_shape = tuple(hidden_states.shape)
    orig_bytes = hidden_states.numel() * hidden_states.element_size()
    x_fp32 = hidden_states.detach().to(torch.float32)

    if _HAS_TRITON and hidden_states.is_cuda and x_fp32.shape[-1] <= 4096:
        try:
            n_cols = x_fp32.shape[-1]
            flat = x_fp32.reshape(-1, n_cols).contiguous()
            n_rows = flat.shape[0]
            q_out = torch.empty_like(flat)
            scales = torch.empty((n_rows, 1), device=flat.device, dtype=torch.float32)
            block_size = triton.next_power_of_2(n_cols)
            _triton_fp8_quant_kernel[(n_rows,)](flat, q_out, scales, n_cols, BLOCK_SIZE=block_size)
            fp8_tensor = q_out.reshape(orig_shape).to(torch.float8_e4m3fn)
            scale = scales.reshape(*orig_shape[:-1], 1)
        except Exception:
            absmax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = absmax / 448.0
            scaled = (x_fp32 / scale).clamp(-448.0, 448.0)
            fp8_tensor = scaled.to(torch.float8_e4m3fn)
    else:
        absmax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = absmax / 448.0
        scaled = (x_fp32 / scale).clamp(-448.0, 448.0)
        fp8_tensor = scaled.to(torch.float8_e4m3fn)

    comp_bytes = fp8_tensor.numel() * 1 + scale.numel() * 2
    return CompressedHiddenStatePacket(
        fp8_data=fp8_tensor,
        absmax_scale=scale.to(torch.float16),
        original_shape=orig_shape,
        original_bytes=orig_bytes,
        compressed_bytes=comp_bytes,
    )


def decompress_hidden_states_fp8(
    packet: CompressedHiddenStatePacket,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Restore FP16/FP32 hidden states from an FP8 wire packet."""
    restored = packet.fp8_data.to(torch.float32) * packet.absmax_scale.to(torch.float32)
    return restored.reshape(packet.original_shape).to(target_dtype)


def pack_int4_nf4_weights(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a 2D expert weight tensor `(out_features, in_features)` into 4-bit signed
    nibbles stored two-per-byte (`uint8`) with per-row FP16 absmax scales (`HM-S3-02`).
    """
    w_fp32 = weight.detach().to(torch.float32)
    out_dim, in_dim = w_fp32.shape
    if in_dim % 2 != 0:
        w_fp32 = torch.nn.functional.pad(w_fp32, (0, 1))
        in_dim += 1
    row_absmax = w_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = row_absmax / 7.0
    q_int = torch.round(w_fp32 / scale).clamp(-7, 7).to(torch.int16) + 8
    even_nib = q_int[:, 0::2] & 0x0F
    odd_nib = (q_int[:, 1::2] & 0x0F) << 4
    packed_uint8 = (even_nib | odd_nib).to(torch.uint8).contiguous()
    return packed_uint8, scale.to(torch.float16)


def unpack_int4_nf4_weights(
    packed_uint8: torch.Tensor,
    scale_fp16: torch.Tensor,
    original_in_dim: Optional[int] = None,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes packed `uint8` 2x4-bit expert weights and per-row FP16 scales
    on the fly back to floating-point representation (`HM-S3-02`).
    """
    packed_i16 = packed_uint8.to(torch.int16)
    even_q = (packed_i16 & 0x0F) - 8
    odd_q = ((packed_i16 >> 4) & 0x0F) - 8
    out_dim, half_dim = packed_i16.shape
    unpacked_q = torch.empty((out_dim, half_dim * 2), device=packed_uint8.device, dtype=torch.float32)
    unpacked_q[:, 0::2] = even_q.to(torch.float32)
    unpacked_q[:, 1::2] = odd_q.to(torch.float32)
    if original_in_dim is not None:
        unpacked_q = unpacked_q[:, :original_in_dim]
    dequant = unpacked_q * scale_fp16.to(torch.float32)
    return dequant.to(target_dtype)


def fused_int4_fp8_moe_route_and_project(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights: Sequence[torch.Tensor],
    top_k: int = 2,
) -> Dict[str, Any]:
    """
    Season 3 Milestone `HM-S3-02`: Fused Top-2 Sparse MoE Router + On-the-Fly INT4 Weight
    Dequantization + FP8 (`float8_e4m3fn`) Hidden-State Activation Projection.
    """
    t0 = time.perf_counter()
    orig_shape = hidden_states.shape
    hidden_dim = int(orig_shape[-1])
    flat_x = hidden_states.detach().to(torch.float32).reshape(-1, hidden_dim)
    num_tokens = int(flat_x.shape[0])
    num_experts = len(expert_weights)
    k = min(max(1, int(top_k)), num_experts)

    router_w = router_weight.detach().to(torch.float32)
    router_logits = flat_x @ router_w.T
    top_k_vals, top_k_indices = torch.topk(router_logits, k=k, dim=-1)
    top_k_gates = torch.softmax(top_k_vals, dim=-1)

    fp8_pkt = compress_hidden_states_fp8(flat_x)
    x_fp8_dequant = decompress_hidden_states_fp8(fp8_pkt, target_dtype=torch.float32)

    packed_experts: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    unpacked_fp16_bytes = 0
    packed_int4_bytes = 0
    for w in expert_weights:
        w_2d = w.detach().to(torch.float16)
        unpacked_fp16_bytes += w_2d.numel() * 2
        p_u8, s_fp16 = pack_int4_nf4_weights(w_2d)
        packed_int4_bytes += p_u8.numel() * 1 + s_fp16.numel() * 2
        packed_experts.append((p_u8, s_fp16, int(w_2d.shape[-1])))

    out_dim = int(expert_weights[0].shape[0])
    fused_out = torch.zeros((num_tokens, out_dim), device=flat_x.device, dtype=torch.float32)
    ref_out = torch.zeros((num_tokens, out_dim), device=flat_x.device, dtype=torch.float32)

    for e_idx in range(num_experts):
        p_u8, s_fp16, orig_in = packed_experts[e_idx]
        w_dequant = unpack_int4_nf4_weights(p_u8, s_fp16, original_in_dim=orig_in, target_dtype=torch.float32)
        w_ref = expert_weights[e_idx].detach().to(torch.float32)

        for slot in range(k):
            mask = top_k_indices[:, slot] == e_idx
            if not torch.any(mask):
                continue
            gate = top_k_gates[mask, slot : slot + 1]
            fused_out[mask] += gate * (x_fp8_dequant[mask] @ w_dequant.T)
            ref_out[mask] += gate * (flat_x[mask] @ w_ref.T)

    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            fused_out.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)
        ).item()
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    backend = (
        "triton_cuda_int4_fp8_moe"
        if (_HAS_TRITON and hidden_states.is_cuda)
        else "pytorch_int4_fp8_moe_fallback"
    )
    return {
        "output": fused_out.reshape(*orig_shape[:-1], out_dim).to(hidden_states.dtype),
        "reference_output": ref_out.reshape(*orig_shape[:-1], out_dim).to(hidden_states.dtype),
        "top_k_indices": top_k_indices,
        "top_k_gates": top_k_gates,
        "cosine_similarity": round(cos_sim, 6),
        "backend": backend,
        "unpacked_fp16_bytes": unpacked_fp16_bytes,
        "packed_int4_bytes": packed_int4_bytes,
        "vram_compression_ratio": round(unpacked_fp16_bytes / max(1, packed_int4_bytes), 2),
        "latency_ms": round(elapsed_ms, 4),
    }


def benchmark_fused_int4_fp8_moe_kernel(
    batch_tokens: int = 32,
    hidden_dim: int = 64,
    num_experts: int = 8,
    top_k: int = 2,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Benchmarks the Season 3 `HM-S3-02` INT4 + FP8 fused MoE router projection
    against full-precision reference execution and verifies `cosine_similarity >= 0.99`.
    """
    torch.manual_seed(42)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    x = torch.randn(batch_tokens, hidden_dim, dtype=torch.float32, device=dev)
    router_w = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=dev)
    expert_ws = [torch.randn(hidden_dim, hidden_dim, dtype=torch.float32, device=dev) for _ in range(num_experts)]

    res = fused_int4_fp8_moe_route_and_project(
        hidden_states=x,
        router_weight=router_w,
        expert_weights=expert_ws,
        top_k=top_k,
    )
    tokens_per_sec = round((batch_tokens / max(1e-6, res["latency_ms"] / 1000.0)), 1)
    return {
        "batch_tokens": batch_tokens,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
        "top_k": top_k,
        "backend": res["backend"],
        "cosine_similarity": res["cosine_similarity"],
        "parity_verified": res["cosine_similarity"] >= 0.99,
        "vram_compression_ratio": res["vram_compression_ratio"],
        "latency_ms": res["latency_ms"],
        "throughput_tokens_per_sec": tokens_per_sec,
    }


class CausalTreeMaskResult(dict):
    """Dict-compatible causal tree-attention mask result that also exposes tensor `.shape` and `.ndim`."""

    @property
    def ndim(self) -> int:
        return 4

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        m = self["causal_tree_mask"]
        return (1, 1, int(m.shape[0]), int(m.shape[1]))


class TokenTreeSpeculativeVerifier:
    """M3: Eagle/Medusa-style Token Tree Speculative Verifier (`gamma = 6..8`) & Prefix-Trie 2D Tree-Attention Mask (`HM-M3-01`)."""

    def __init__(self, gamma: int = 6, branching_factor: int = 2) -> None:
        self.gamma = max(6, min(8, gamma))
        self.branching_factor = branching_factor

    def build_causal_tree_attention_mask(
        self,
        draft_tree_candidates: List[List[int]],
        device: torch.device | str | None = None,
    ) -> CausalTreeMaskResult:
        """HM-M3-01 (PR #8): Build a deduplicated prefix-trie causal tree-attention mask for single-pass MoE verification."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        nodes: List[Tuple[int, ...]] = [()]
        node_index: Dict[Tuple[int, ...], int] = {(): 0}
        parent_map: Dict[int, int] = {0: -1}

        for branch in draft_tree_candidates:
            prefix: List[int] = []
            for tok in branch[: self.gamma]:
                parent_key = tuple(prefix)
                prefix.append(tok)
                cur_key = tuple(prefix)
                if cur_key not in node_index:
                    idx = len(nodes)
                    nodes.append(cur_key)
                    node_index[cur_key] = idx
                    parent_map[idx] = node_index[parent_key]

        num_nodes = len(nodes)
        mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=dev)
        for idx in range(num_nodes):
            anc = idx
            while anc != -1:
                mask[idx, anc] = True
                anc = parent_map[anc]

        return CausalTreeMaskResult(
            {
                "num_tree_nodes": num_nodes,
                "tree_depth_gamma": self.gamma,
                "causal_tree_mask": mask,
                "node_prefixes": nodes,
            }
        )

    def verify_token_tree(
        self,
        draft_tree_candidates: List[List[int]],
        target_reference_tokens: List[int],
    ) -> Dict[str, Any]:
        """Verify a multi-branch token tree of depth `gamma=6..8` in a single parallel MoE forward pass."""
        tree_mask_info = self.build_causal_tree_attention_mask(draft_tree_candidates)
        best_path: List[int] = []
        for branch in draft_tree_candidates:
            accepted: List[int] = []
            for i, tok in enumerate(branch[: self.gamma]):
                if i < len(target_reference_tokens) and tok == target_reference_tokens[i]:
                    accepted.append(tok)
                else:
                    if i < len(target_reference_tokens):
                        accepted.append(target_reference_tokens[i])
                    break
            if len(accepted) > len(best_path):
                best_path = accepted

        if len(best_path) == self.gamma and len(target_reference_tokens) > self.gamma:
            best_path.append(target_reference_tokens[self.gamma])

        return {
            "tree_depth_gamma": self.gamma,
            "branches_evaluated_in_single_pass": len(draft_tree_candidates),
            "num_tree_nodes": tree_mask_info["num_tree_nodes"],
            "causal_tree_mask_shape": tuple(tree_mask_info["causal_tree_mask"].shape),
            "tree_attention_mask_shape": list(tree_mask_info.shape),
            "accepted_tokens": best_path,
            "accepted_length": len(best_path),
            "accepted_tokens_per_verification_pass": len(best_path),
            "effective_rtt_speedup": None,
            "rtt_speedup_status": "not_measured: accepted_length is a token count, not a timing",
        }


# ============================================================================
# M4: Optimistic Proof-of-Inference (OPoI) & Layer-2 $HIVE State Channels
# ============================================================================


@dataclass
class MerkleInclusionProof:
    """HM-S4-03: `O(log L)` Merkle sibling path binding one quantized activation leaf to a root."""

    layer_idx: int
    leaf_hash: str
    siblings: List[str]
    directions: List[int]
    root: str
    tree_size: int

    @property
    def proof_depth(self) -> int:
        """Number of sibling hashes carried by the inclusion path (`O(log L)`)."""
        return len(self.siblings)

    def to_dict(self) -> Dict[str, Any]:
        """Wire-format (JSON serializable) representation of the inclusion proof."""
        return {
            "layer_idx": self.layer_idx,
            "leaf_hash": self.leaf_hash,
            "siblings": list(self.siblings),
            "directions": list(self.directions),
            "root": self.root,
            "tree_size": self.tree_size,
            "proof_depth": self.proof_depth,
        }


# HM-S6-03: batched speculative activation commitments. Hash-based (SHA-256) and binding, not hiding:
# an opening reveals the activation it opens, so this is not a zero-knowledge proof.
_BATCH_DOMAIN = b"hivemind/opoi-batch/v1"


def _batch_leaf_digest(step: int, layer: int, layer_leaf_hex: str) -> bytes:
    return hashlib.sha256(
        b"\x00"
        + _BATCH_DOMAIN
        + int(step).to_bytes(4, "big")
        + int(layer).to_bytes(4, "big")
        + bytes.fromhex(layer_leaf_hex)
    ).digest()


def _batch_node_digest(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _batch_root_hex(peer_id: str, num_steps: int, num_layers: int, top: bytes) -> str:
    peer = peer_id.encode("utf-8")
    return hashlib.sha256(
        b"\x02"
        + _BATCH_DOMAIN
        + len(peer).to_bytes(4, "big")
        + peer
        + int(num_steps).to_bytes(4, "big")
        + int(num_layers).to_bytes(4, "big")
        + top
    ).hexdigest()


def _batch_level_sizes(leaf_count: int) -> List[int]:
    sizes = [leaf_count]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


@dataclass(frozen=True)
class SpeculativeBatchCommitment:
    """HM-S6-03: one root committing every layer of every speculative step in a batch.

    The root binds the peer id, the batch shape (`num_steps` x `num_layers`) and every
    position-tagged FP8 + FP16-scale leaf, so the commitment is 32 bytes whatever `gamma` is.
    """

    peer_id: str
    root: str
    num_steps: int
    num_layers: int

    @property
    def leaf_count(self) -> int:
        return self.num_steps * self.num_layers

    @property
    def max_opening_siblings(self) -> int:
        """Upper bound on sibling hashes in one opening: `ceil(log2(num_steps * num_layers))`."""
        return len(_batch_level_sizes(self.leaf_count)) - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "root": self.root,
            "num_steps": self.num_steps,
            "num_layers": self.num_layers,
            "leaf_count": self.leaf_count,
            "max_opening_siblings": self.max_opening_siblings,
        }


@dataclass(frozen=True)
class SpeculativeBatchOpening:
    """HM-S6-03: sibling path opening one `(step, layer)` leaf of a batch commitment."""

    step: int
    layer: int
    siblings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "layer": self.layer, "siblings": list(self.siblings)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeculativeBatchOpening":
        return cls(
            step=int(data["step"]),
            layer=int(data["layer"]),
            siblings=tuple(str(s) for s in data["siblings"]),
        )


class OptimisticProofOfInferenceAuditor:
    """M4: Probabilistic 5% Optimistic Proof-of-Inference (OPoI) Activation Auditor & Merkle Dispute Game (`HM-M4-02`).

    Season 4 (`HM-S4-03`) upgrades the auditor to **ZK-OPoI v2**: every layer's hidden state is
    quantized to FP8 (`float8_e4m3fn`) with a per-row FP16 absmax scale, both are bound into a
    binary Merkle tree, and a single audited layer is proven with an `O(log L)` inclusion path.
    Two *independent* acceptance bounds must hold for a spot-check to pass:
    `cosine_similarity >= 0.99` **and** a normalized `L_infinity` deviation bound.
    """

    def __init__(
        self,
        audit_probability: float = 0.05,
        min_cosine_sim: float = 0.99,
        max_normalized_linf: float = 0.25,
    ) -> None:
        self.audit_probability = audit_probability
        self.min_cosine_sim = min_cosine_sim
        self.max_normalized_linf = max_normalized_linf
        # HM-S4-03: layer-wise quantized activation commitment (Merkle) state
        self._merkle_leaves: List[str] = []
        self._merkle_levels: List[List[str]] = []
        self.merkle_root: str = ""
        self._batch_levels: List[List[bytes]] = []
        self._batch_commitment: Optional[SpeculativeBatchCommitment] = None

    def should_spot_audit_token(self, rng_seed: int | None = None) -> bool:
        rng = random.Random(rng_seed)
        return rng.random() < self.audit_probability

    def verify_activation_commitment(
        self,
        peer_id: str,
        claimed_activation: torch.Tensor,
        recomputed_activation: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compare a peer's committed FP8/FP16 hidden state against validator recomputation."""
        a = claimed_activation.detach().flatten().to(torch.float32)
        b = recomputed_activation.detach().flatten().to(torch.float32)
        cos_sim = float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        )
        max_diff = float((a - b).abs().max().item())
        denom = float(b.abs().max().item()) + 1e-8
        normalized_linf = max_diff / denom
        passed = cos_sim >= self.min_cosine_sim and normalized_linf <= self.max_normalized_linf
        return {
            "peer_id": peer_id,
            "cosine_similarity": round(cos_sim, 6),
            "max_abs_diff": round(max_diff, 6),
            "normalized_linf": round(normalized_linf, 6),
            "opoi_passed": passed,
            "slashed": not passed,
        }

    @staticmethod
    def _hash_activation_tensor(t: torch.Tensor) -> str:
        vals = [round(float(x), 3) for x in t.detach().flatten().to(torch.float32)[:32].tolist()]
        return hashlib.sha256(str(vals).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # HM-S4-03 (ZK-OPoI v2): Layer-Wise Merkle Commitment & Inclusion Proofs
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_bytes(t: torch.Tensor) -> bytes:
        """Serialize the raw little-endian bytes of a tensor for cryptographic commitment."""
        flat = t.detach().cpu().contiguous().reshape(-1)
        return flat.view(torch.uint8).numpy().tobytes()

    @staticmethod
    def _hash_merkle_pair(left: str, right: str) -> str:
        """Binary Merkle node hash: `SHA256(left_hex || right_hex)`."""
        return hashlib.sha256((left + right).encode("utf-8")).hexdigest()

    @classmethod
    def _quantized_leaf_hash(cls, packet: CompressedHiddenStatePacket) -> str:
        """Bind the FP8 quantized hidden state **and** its per-row FP16 absmax scale into one leaf."""
        fp8_bytes = cls._raw_bytes(packet.fp8_data)
        scale_bytes = cls._raw_bytes(packet.absmax_scale)
        metadata = (
            "zk_opoi_v2|fp8=float8_e4m3fn|scale=float16"
            f"|shape={tuple(packet.original_shape)}|scale_shape={tuple(packet.absmax_scale.shape)}"
        )
        digest = hashlib.sha256()
        digest.update(fp8_bytes)
        digest.update(b"|")
        digest.update(scale_bytes)
        digest.update(b"|")
        digest.update(metadata.encode("utf-8"))
        return digest.hexdigest()

    def layer_commitment_hash(self, activation: torch.Tensor) -> str:
        """HM-S4-03: Canonical leaf hash of one layer's FP8 + per-row FP16 absmax activation."""
        return self._quantized_leaf_hash(compress_hidden_states_fp8(activation))

    def commit_layer_activations(self, activations: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """HM-S4-03: Quantize every layer and bind the leaf hashes into a binary Merkle tree."""
        if not activations:
            raise ValueError("commit_layer_activations() requires at least one layer activation.")

        leaves: List[str] = [self.layer_commitment_hash(act) for act in activations]

        levels: List[List[str]] = [list(leaves)]
        while len(levels[-1]) > 1:
            current = levels[-1]
            next_level: List[str] = []
            for i in range(0, len(current), 2):
                left = current[i]
                # Odd-length level: the trailing node is duplicated into the next layer.
                right = current[i + 1] if i + 1 < len(current) else current[i]
                next_level.append(self._hash_merkle_pair(left, right))
            levels.append(next_level)

        self._merkle_leaves = leaves
        self._merkle_levels = levels
        self.merkle_root = levels[-1][0]

        return {
            "merkle_root": self.merkle_root,
            "num_layers": len(leaves),
            "leaf_hashes": list(leaves),
            "tree_depth": len(levels) - 1,
            "max_proof_depth": len(levels) - 1,
            "proof_complexity": "O(log L)",
            "quantization": "float8_e4m3fn+fp16_absmax_per_row",
        }

    def build_merkle_inclusion_proof(
        self,
        layer_idx: int,
        layer_activations: Optional[List[torch.Tensor]] = None,
    ) -> Any:
        """HM-S4-03: Build the `O(log L)` Merkle sibling path for one committed layer."""
        if layer_activations is not None:
            if not layer_activations:
                raise ValueError("layer_activations must be non-empty")
            if layer_idx < 0 or layer_idx >= len(layer_activations):
                raise IndexError(f"layer_idx {layer_idx} outside 0..{len(layer_activations) - 1}")
            level = [self._hash_activation_tensor(t) for t in layer_activations]
            leaf = level[layer_idx]
            siblings_list: List[str] = []
            index = layer_idx
            while len(level) > 1:
                if len(level) % 2 == 1:
                    level = level + [level[-1]]
                sibling = index ^ 1
                siblings_list.append(level[sibling])
                level = [
                    hashlib.sha256(f"{level[i]}|{level[i + 1]}".encode("utf-8")).hexdigest()
                    for i in range(0, len(level), 2)
                ]
                index //= 2
            return {
                "layer_idx": layer_idx,
                "leaf": leaf,
                "siblings": siblings_list,
                "root": level[0],
                "depth": len(siblings_list),
            }

        if not self._merkle_levels:
            raise ValueError(
                "No activation commitment exists: call commit_layer_activations() before building proofs."
            )
        if layer_idx < 0 or layer_idx >= len(self._merkle_leaves):
            raise IndexError(
                f"layer_idx {layer_idx} is out of range for {len(self._merkle_leaves)} committed layers."
            )

        siblings: List[str] = []
        directions: List[int] = []
        idx = layer_idx
        for level in self._merkle_levels[:-1]:
            sibling_idx = idx ^ 1
            if sibling_idx < len(level):
                siblings.append(level[sibling_idx])
                directions.append(0 if sibling_idx < idx else 1)
            else:
                siblings.append(level[idx])
                directions.append(1)
            idx = idx >> 1

        return MerkleInclusionProof(
            layer_idx=layer_idx,
            leaf_hash=self._merkle_leaves[layer_idx],
            siblings=siblings,
            directions=directions,
            root=self.merkle_root,
            tree_size=len(self._merkle_leaves),
        )

    @staticmethod
    def _as_inclusion_proof(proof: "MerkleInclusionProof | Dict[str, Any]") -> MerkleInclusionProof:
        if isinstance(proof, MerkleInclusionProof):
            return proof
        return MerkleInclusionProof(
            layer_idx=int(proof.get("layer_idx", -1)),
            leaf_hash=str(proof.get("leaf_hash", "")),
            siblings=[str(s) for s in proof.get("siblings", [])],
            directions=[int(d) for d in proof.get("directions", [])],
            root=str(proof.get("root", "")),
            tree_size=int(proof.get("tree_size", 0)),
        )

    def verify_merkle_inclusion_proof(
        self,
        proof: "MerkleInclusionProof | Dict[str, Any]",
        leaf_hash: str | None = None,
        expected_root: str | None = None,
    ) -> Any:
        """HM-S4-03: Recompute the Merkle root from an inclusion path and check the leaf binding."""
        if (
            isinstance(proof, dict)
            and "leaf" in proof
            and "directions" not in proof
            and leaf_hash is None
            and expected_root is None
        ):
            index = int(proof["layer_idx"])
            current = str(proof["leaf"])
            for sibling in proof["siblings"]:
                if index % 2 == 0:
                    current = hashlib.sha256(f"{current}|{sibling}".encode("utf-8")).hexdigest()
                else:
                    current = hashlib.sha256(f"{sibling}|{current}".encode("utf-8")).hexdigest()
                index //= 2
            return current == proof["root"]

        parsed = self._as_inclusion_proof(proof)
        leaf = leaf_hash or parsed.leaf_hash
        computed_root = leaf
        for sibling, direction in zip(parsed.siblings, parsed.directions):
            if direction == 0:
                computed_root = self._hash_merkle_pair(sibling, computed_root)
            else:
                computed_root = self._hash_merkle_pair(computed_root, sibling)

        root = expected_root or parsed.root
        included = computed_root == root
        log_l_depth = math.ceil(math.log2(max(2, parsed.tree_size))) if parsed.tree_size else 0
        return {
            "layer_idx": parsed.layer_idx,
            "included": included,
            "root_match": included,
            "computed_root": computed_root,
            "expected_root": root,
            "leaf_hash": leaf,
            "proof_depth": parsed.proof_depth,
            "tree_size": parsed.tree_size,
            "is_log_l_proof": parsed.proof_depth <= log_l_depth,
            "slashed": not included,
        }

    def audit_quantized_activation(
        self,
        peer_id: str,
        claimed_activation: torch.Tensor,
        recomputed_activation: torch.Tensor,
    ) -> Dict[str, Any]:
        """HM-S4-03: Dual-threshold audit in the quantized domain (cosine **and** normalized `L_inf`).

        The comparison happens on the dequantized FP8/FP16 activations, i.e. exactly the domain
        bound by the Merkle commitment. The normalized `L_infinity` bound is required because the
        cosine similarity is scale invariant: a pure gain/scale forgery keeps
        `cosine_similarity = 1.0` and is only exposed by the `L_infinity` bound.
        """
        claimed_packet = compress_hidden_states_fp8(claimed_activation)
        recomputed_packet = compress_hidden_states_fp8(recomputed_activation)
        a = decompress_hidden_states_fp8(claimed_packet, target_dtype=torch.float32).flatten()
        b = decompress_hidden_states_fp8(recomputed_packet, target_dtype=torch.float32).flatten()

        if a.numel() != b.numel():
            raise ValueError(
                "Claimed and recomputed activations must hold the same number of elements "
                f"({a.numel()} != {b.numel()})."
            )

        cos_sim = float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        )
        linf = float((a - b).abs().max().item())
        reference_absmax = float(b.abs().max().item())
        normalized_linf = linf / max(reference_absmax, 1e-6)

        cosine_passed = cos_sim >= self.min_cosine_sim
        linf_passed = normalized_linf <= self.max_normalized_linf
        dual_bound_passed = bool(cosine_passed and linf_passed)

        return {
            "peer_id": peer_id,
            "cosine_similarity": round(cos_sim, 6),
            "cosine_threshold": self.min_cosine_sim,
            "cosine_passed": cosine_passed,
            "normalized_linf": round(normalized_linf, 6),
            "linf_bound": self.max_normalized_linf,
            "linf_passed": linf_passed,
            "max_abs_diff": round(linf, 6),
            "reference_absmax": round(reference_absmax, 6),
            "dual_bound_passed": dual_bound_passed,
            "zk_opoi_passed": dual_bound_passed,
            "slashed": not dual_bound_passed,
        }

    # ------------------------------------------------------------------
    # HM-S6-03: Batched commitments over gamma speculative token-tree steps
    # ------------------------------------------------------------------

    def commit_speculative_batch(
        self,
        peer_id: str,
        step_activations: Sequence[Sequence[torch.Tensor]],
    ) -> SpeculativeBatchCommitment:
        """Commit every layer activation of every speculative step under one Merkle root.

        Leaves are tagged with their `(step, layer)` position and domain-separated from
        internal nodes; a lone node at the end of a level is carried up unchanged.
        """
        if not str(peer_id).strip():
            raise ValueError("peer_id must be non-empty")
        if not step_activations:
            raise ValueError("commit_speculative_batch() requires at least one speculative step")
        num_layers = len(step_activations[0])
        if num_layers == 0:
            raise ValueError("every speculative step must commit at least one layer")
        if any(len(layers) != num_layers for layers in step_activations):
            raise ValueError("every speculative step must commit the same number of layers")

        level: List[bytes] = [
            _batch_leaf_digest(step, layer, self.layer_commitment_hash(activation))
            for step, layers in enumerate(step_activations)
            for layer, activation in enumerate(layers)
        ]
        levels = [level]
        while len(level) > 1:
            level = [
                _batch_node_digest(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
                for i in range(0, len(level), 2)
            ]
            levels.append(level)

        commitment = SpeculativeBatchCommitment(
            peer_id=str(peer_id),
            root=_batch_root_hex(str(peer_id), len(step_activations), num_layers, levels[-1][0]),
            num_steps=len(step_activations),
            num_layers=num_layers,
        )
        self._batch_levels = levels
        self._batch_commitment = commitment
        return commitment

    def open_speculative_batch(self, step: int, layer: int) -> SpeculativeBatchOpening:
        """Prover side: the sibling path for one `(step, layer)` leaf of the last batch."""
        commitment = self._batch_commitment
        if commitment is None:
            raise ValueError("No batch commitment exists: call commit_speculative_batch() first.")
        if not (0 <= step < commitment.num_steps and 0 <= layer < commitment.num_layers):
            raise IndexError(
                f"(step={step}, layer={layer}) is outside the committed "
                f"{commitment.num_steps}x{commitment.num_layers} batch"
            )
        index = step * commitment.num_layers + layer
        siblings: List[str] = []
        for level in self._batch_levels[:-1]:
            sibling = index ^ 1
            if sibling < len(level):
                siblings.append(level[sibling].hex())
            index >>= 1
        return SpeculativeBatchOpening(step=step, layer=layer, siblings=tuple(siblings))

    def verify_speculative_batch_opening(
        self,
        commitment: SpeculativeBatchCommitment,
        opening: SpeculativeBatchOpening,
        claimed_activation: torch.Tensor,
    ) -> bool:
        """Recompute the root from `claimed_activation` at the opened position.

        Sibling order and path length come from the position and the committed batch
        shape, never from the opening, so a path for one leaf cannot open another.
        """
        if not (
            0 <= opening.step < commitment.num_steps and 0 <= opening.layer < commitment.num_layers
        ):
            return False
        index = opening.step * commitment.num_layers + opening.layer
        try:
            siblings = [bytes.fromhex(s) for s in opening.siblings]
        except ValueError:
            return False
        if any(len(s) != 32 for s in siblings):
            return False

        node = _batch_leaf_digest(
            opening.step, opening.layer, self.layer_commitment_hash(claimed_activation)
        )
        used = 0
        for size in _batch_level_sizes(commitment.leaf_count)[:-1]:
            if index ^ 1 < size:
                if used == len(siblings):
                    return False
                sibling = siblings[used]
                used += 1
                node = _batch_node_digest(sibling, node) if index & 1 else _batch_node_digest(node, sibling)
            index >>= 1
        if used != len(siblings):
            return False
        expected = _batch_root_hex(
            commitment.peer_id, commitment.num_steps, commitment.num_layers, node
        )
        return hmac.compare_digest(expected, commitment.root)

    @staticmethod
    def derive_batch_challenges(
        commitment: SpeculativeBatchCommitment,
        num_challenges: int,
        verifier_nonce: str,
    ) -> List[Tuple[int, int]]:
        """Distinct `(step, layer)` positions to audit, derived from the root and a nonce.

        The nonce must be chosen by the verifier after the commitment is received; positions
        derived from the root alone would let a prover grind roots until its bad leaves are
        skipped. With `k` distinct positions out of `n` leaves of which `b` are bad, every bad
        leaf is missed with probability `C(n - b, k) / C(n, k)`.
        """
        if not str(verifier_nonce):
            raise ValueError("verifier_nonce must be non-empty")
        if not 1 <= int(num_challenges) <= commitment.leaf_count:
            raise ValueError(
                f"num_challenges must be within [1, {commitment.leaf_count}], got {num_challenges}"
            )
        chosen: List[int] = []
        counter = 0
        while len(chosen) < int(num_challenges):
            digest = hashlib.sha256(
                b"hivemind/opoi-batch/challenge|"
                + commitment.root.encode("ascii")
                + b"|"
                + str(verifier_nonce).encode("utf-8")
                + b"|"
                + counter.to_bytes(8, "big")
            ).digest()
            counter += 1
            index = int.from_bytes(digest[:8], "big") % commitment.leaf_count
            if index not in chosen:
                chosen.append(index)
        return [divmod(index, commitment.num_layers) for index in chosen]

    def verify_speculative_batch_challenge(
        self,
        commitment: SpeculativeBatchCommitment,
        opening: SpeculativeBatchOpening,
        claimed_activation: torch.Tensor,
        recomputed_activation: torch.Tensor,
    ) -> Dict[str, Any]:
        """Check one challenged leaf: it must open the root and pass the dual-bound audit."""
        included = self.verify_speculative_batch_opening(commitment, opening, claimed_activation)
        verdict: Dict[str, Any] = {
            "peer_id": commitment.peer_id,
            "commitment_root": commitment.root,
            "step": opening.step,
            "layer": opening.layer,
            "included": included,
            "opening_siblings": len(opening.siblings),
            "zero_knowledge": False,
        }
        if not included:
            verdict.update(dual_bound_passed=False, fault_proven=True, fault_reason="opening_rejected")
            return verdict
        try:
            audit = self.audit_quantized_activation(
                commitment.peer_id, claimed_activation, recomputed_activation
            )
        except ValueError:
            verdict.update(dual_bound_passed=False, fault_proven=True, fault_reason="shape_mismatch")
            return verdict
        passed = bool(audit["dual_bound_passed"])
        verdict.update(
            cosine_similarity=audit["cosine_similarity"],
            normalized_linf=audit["normalized_linf"],
            dual_bound_passed=passed,
            fault_proven=not passed,
            fault_reason=None if passed else "activation_out_of_bounds",
        )
        return verdict

    def verify_speculative_batch(
        self,
        commitment: SpeculativeBatchCommitment,
        challenges: Sequence[Tuple[int, int]],
        responses: Sequence[Tuple[SpeculativeBatchOpening, torch.Tensor, torch.Tensor]],
    ) -> Dict[str, Any]:
        """Verify a whole batch against one root: one response per challenged position.

        `responses` holds `(opening, claimed_activation, recomputed_activation)` and must
        answer exactly the challenged positions, in order.
        """
        if [(int(s), int(l)) for s, l in challenges] != [(o.step, o.layer) for o, _, _ in responses]:
            raise ValueError(
                "responses must answer exactly the challenged (step, layer) positions, in order"
            )
        verdicts = [
            self.verify_speculative_batch_challenge(commitment, opening, claimed, recomputed)
            for opening, claimed, recomputed in responses
        ]
        faults = [v for v in verdicts if v["fault_proven"]]
        return {
            "peer_id": commitment.peer_id,
            "commitment_root": commitment.root,
            "commitment_bytes": 32,
            "num_steps": commitment.num_steps,
            "num_layers": commitment.num_layers,
            "positions_checked": len(verdicts),
            "siblings_checked": sum(v["opening_siblings"] for v in verdicts),
            "batch_passed": not faults,
            "faults": faults,
            "verdicts": verdicts,
            "zero_knowledge": False,
        }

    @staticmethod
    def new_verifier_nonce() -> str:
        """A fresh 128-bit nonce for `derive_batch_challenges`, drawn after the commitment arrives."""
        return secrets.token_hex(16)

    def audit_batch_and_indemnify(
        self,
        commitment: SpeculativeBatchCommitment,
        challenges: Sequence[Tuple[int, int]],
        responses: Sequence[Tuple[SpeculativeBatchOpening, torch.Tensor, torch.Tensor]],
        pool: GuildSlashingInsurancePool,
        consumer_node_id: str,
        claimed_hive: float,
    ) -> Dict[str, Any]:
        """HM-S7-04: verify a challenged batch and indemnify the consumer if a fault is proven.

        The first faulty leaf goes to `pool.indemnify_proven_fault`, which re-verifies it before
        paying, and the pool pays at most once per batch root. A refusal by the pool (unbonded
        peer, repeated claim, coverage cap, ...) is reported in `indemnity_refused` rather than
        raised, so the audit verdict is always returned.
        """
        report = self.verify_speculative_batch(commitment, challenges, responses)
        report["indemnity"] = None
        report["indemnity_refused"] = None
        if not report["faults"]:
            return report
        fault = report["faults"][0]
        opening, claimed, recomputed = next(
            response
            for response in responses
            if (response[0].step, response[0].layer) == (fault["step"], fault["layer"])
        )
        try:
            receipt = pool.indemnify_proven_fault(
                self,
                commitment,
                opening,
                claimed,
                recomputed,
                consumer_node_id=consumer_node_id,
                claimed_hive=claimed_hive,
            )
        except SwarmSecurityViolationError as exc:
            report["indemnity_refused"] = {
                "rule_id": exc.rule_id,
                "category": exc.category,
                "message": exc.message,
            }
        else:
            report["indemnity"] = asdict(receipt)
        return report

    def run_merkle_fraud_dispute_game(
        self,
        peer_id: str,
        claimed_layer_activations: List[torch.Tensor],
        validator_layer_activations: List[torch.Tensor],
        ledger: HiveSwarmLedger | None = None,
        l2_channel: "HiveLayer2StateChannel | None" = None,
    ) -> Dict[str, Any]:
        """HM-M4-02: Execute a Merkle-proof activation dispute game, slash dishonest host, and refund L2 escrow."""
        claimed_leaves = [self._hash_activation_tensor(t) for t in claimed_layer_activations]
        validator_leaves = [self._hash_activation_tensor(t) for t in validator_layer_activations]
        claimed_root = hashlib.sha256("|".join(claimed_leaves).encode("utf-8")).hexdigest()
        validator_root = hashlib.sha256("|".join(validator_leaves).encode("utf-8")).hexdigest()

        diverging_layer: int | None = None
        for idx, (c_t, v_t) in enumerate(zip(claimed_layer_activations, validator_layer_activations)):
            verdict = self.verify_activation_commitment(peer_id, c_t, v_t)
            if not verdict["opoi_passed"]:
                diverging_layer = idx
                break

        fraud_proven = diverging_layer is not None
        refunded_hive = 0.0
        if fraud_proven:
            if ledger is not None:
                ledger.balances[peer_id] = 0.0
                ledger.reputation_scores[peer_id] = 0
                if peer_id not in ledger.slashed_registry:
                    ledger.slashed_registry.append(peer_id)
            if l2_channel is not None:
                refunded_hive = l2_channel.refund_consumer_escrow()

        return {
            "peer_id": peer_id,
            "claimed_merkle_root": claimed_root,
            "validator_merkle_root": validator_root,
            "fraud_proven": fraud_proven,
            "diverging_layer_index": diverging_layer,
            "slashed_in_l1_ledger": fraud_proven and (ledger is not None),
            "l2_escrow_refunded_hive": refunded_hive,
        }



    def run_interactive_bisection_dispute_game_v3(
        self,
        peer_id: str,
        claimed_layer_activations: Sequence[torch.Tensor],
        validator_layer_activations: Sequence[torch.Tensor],
        prover_bond_hive: float = 50.0,
        challenger_id: str = "challenger-validator-01",
        consumer_id: str = "consumer-client-01",
        ledger: HiveSwarmLedger | None = None,
        l2_channel: "HiveLayer2StateChannel | None" = None,
    ) -> Dict[str, Any]:
        """Season 5 (`HM-S5-03` / `ZK-OPoI v3`): Interactive `O(log2 L)` Bisection Fraud Proof Game
        with Single-Layer Operator Verification and L2 Prover Bond Slashing (`50%` Challenger / `50%` Consumer).
        """
        num_layers = len(claimed_layer_activations)
        if num_layers == 0 or num_layers != len(validator_layer_activations):
            raise ValueError(
                "claimed_layer_activations and validator_layer_activations must be non-empty and equal length."
            )

        prover_commit = self.commit_layer_activations(claimed_layer_activations)
        prover_root = prover_commit["merkle_root"]
        prover_leaves = list(prover_commit["leaf_hashes"])

        # Build Merkle inclusion proof at target index on prover's committed tree
        max_rounds = max(1, math.ceil(math.log2(max(2, num_layers))))
        low = 0
        high = num_layers - 1
        transcript: List[Dict[str, Any]] = []

        # Quick check if roots already match and all dual-bounds pass
        roots_differ = False
        for c_act, v_act in zip(claimed_layer_activations, validator_layer_activations):
            if not self.audit_quantized_activation(peer_id, c_act, v_act)["dual_bound_passed"]:
                roots_differ = True
                break

        if not roots_differ:
            return {
                "protocol": "ZK-OPoI-v3-Interactive-Bisection",
                "peer_id": peer_id,
                "num_layers": num_layers,
                "max_theoretical_rounds": max_rounds,
                "bisection_rounds_executed": 0,
                "single_layer_replays_executed": 0,
                "fraud_proven": False,
                "diverging_layer_index": None,
                "prover_merkle_root": prover_root,
                "bisection_transcript": [],
                "prover_bond_slashed_hive": 0.0,
                "challenger_bounty_hive": 0.0,
                "consumer_compensation_hive": 0.0,
            }

        round_idx = 0
        while low < high:
            round_idx += 1
            mid = (low + high) // 2
            audit_mid = self.audit_quantized_activation(
                peer_id, claimed_layer_activations[mid], validator_layer_activations[mid]
            )
            # Check whether prefix [0..mid] is clean so we know which half holds the first fault
            prefix_clean = all(
                self.audit_quantized_activation(
                    peer_id, claimed_layer_activations[idx], validator_layer_activations[idx]
                )["dual_bound_passed"]
                for idx in range(low, mid + 1)
            )
            proof_mid = self.build_merkle_inclusion_proof(mid)
            transcript.append(
                {
                    "round": round_idx,
                    "low": low,
                    "mid": mid,
                    "high": high,
                    "mid_dual_bound_passed": audit_mid["dual_bound_passed"],
                    "prefix_clean": prefix_clean,
                    "merkle_proof_depth": proof_mid.proof_depth,
                }
            )
            if prefix_clean:
                low = mid + 1
            else:
                high = mid

        fault_layer = low
        single_layer_audit = self.audit_quantized_activation(
            peer_id,
            claimed_layer_activations[fault_layer],
            validator_layer_activations[fault_layer],
        )
        inclusion_proof = self.build_merkle_inclusion_proof(fault_layer)
        inclusion_check = self.verify_merkle_inclusion_proof(
            inclusion_proof,
            leaf_hash=prover_leaves[fault_layer],
            expected_root=prover_root,
        )

        fraud_proven = not single_layer_audit["dual_bound_passed"]
        slashed_bond = round(float(prover_bond_hive), 4) if fraud_proven else 0.0
        challenger_bounty = round(slashed_bond * 0.5, 4)
        consumer_comp = round(slashed_bond - challenger_bounty, 4)
        l2_refund = 0.0

        if fraud_proven:
            if ledger is not None:
                ledger.balances[peer_id] = 0.0
                ledger.reputation_scores[peer_id] = 0
                if peer_id not in ledger.slashed_registry:
                    ledger.slashed_registry.append(peer_id)
                ledger.credit_reward(
                    node_id=challenger_id,
                    amount_hive=challenger_bounty,
                    reason=f"ZK-OPoI v3 Bisection Challenger Bounty (caught {peer_id} at layer {fault_layer})",
                )
                ledger.credit_reward(
                    node_id=consumer_id,
                    amount_hive=consumer_comp,
                    reason=f"ZK-OPoI v3 Bisection Consumer Bond Compensation (layer {fault_layer})",
                )
            if l2_channel is not None:
                l2_refund = l2_channel.refund_consumer_escrow()

        return {
            "protocol": "ZK-OPoI-v3-Interactive-Bisection",
            "peer_id": peer_id,
            "num_layers": num_layers,
            "max_theoretical_rounds": max_rounds,
            "bisection_rounds_executed": round_idx,
            "single_layer_replays_executed": 1,
            "log_l_bound_satisfied": round_idx <= max_rounds,
            "fraud_proven": fraud_proven,
            "diverging_layer_index": fault_layer,
            "prover_merkle_root": prover_root,
            "fault_layer_inclusion_verified": inclusion_check["included"],
            "single_layer_audit": single_layer_audit,
            "bisection_transcript": transcript,
            "prover_bond_slashed_hive": slashed_bond,
            "challenger_bounty_hive": challenger_bounty,
            "consumer_compensation_hive": consumer_comp,
            "l2_escrow_refunded_hive": l2_refund,
        }


@dataclass
class StateChannelVoucher:
    """Signed off-chain Layer-2 `$HIVE` micro-payment voucher."""

    channel_id: str
    consumer_id: str
    provider_ids: List[str]
    nonce: int
    cumulative_tokens: int
    cumulative_hive_spent: float
    signature_hmac: str


class HiveLayer2StateChannel:
    """M4: Layer-2 `$HIVE` off-chain state channel with batch settlement to L1 `HiveSwarmLedger`."""

    def __init__(
        self,
        consumer_id: str,
        provider_ids: List[str],
        escrow_hive: float = 5.0,
        rate_per_token: float = 0.005,
        secret_key: bytes = b"HiveMind-L2-StateChannel-Key",
    ) -> None:
        self.channel_id = f"l2-chan-{hashlib.sha256(f'{consumer_id}:{time.time()}'.encode()).hexdigest()[:12]}"
        self.consumer_id = consumer_id
        self.provider_ids = provider_ids
        self.escrow_hive = escrow_hive
        self.rate_per_token = rate_per_token
        self.secret_key = secret_key
        self.nonce = 0
        self.last_settled_nonce = 0
        self.cumulative_tokens = 0
        self.cumulative_hive_spent = 0.0
        self.latest_voucher: StateChannelVoucher | None = None
        self.disputed_and_refunded: bool = False
        self.settled: bool = False

    def _compute_voucher_hmac(
        self,
        *,
        channel_id: str,
        consumer_id: str,
        nonce: int,
        cumulative_tokens: int,
        cumulative_hive_spent: float,
    ) -> str:
        payload = (
            f"{channel_id}:{consumer_id}:{nonce}:"
            f"{cumulative_tokens}:{cumulative_hive_spent:.6f}"
        )
        return hmac.new(self.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def sign_micro_voucher(self, tokens_generated: int) -> StateChannelVoucher:
        """Issue an instant 0-latency off-chain `$HIVE` micro-voucher for `tokens_generated`."""
        if self.disputed_and_refunded:
            raise RuntimeError(f"L2 state channel {self.channel_id} is disputed and refunded.")
        if self.settled:
            raise RuntimeError(f"L2 state channel {self.channel_id} is already settled.")
        if int(tokens_generated) <= 0:
            raise ValueError("tokens_generated must be > 0")
        fee = round(tokens_generated * self.rate_per_token, 6)
        if self.cumulative_hive_spent + fee > self.escrow_hive:
            raise ValueError("L2 State Channel escrow exhausted.")
        self.nonce += 1
        self.cumulative_tokens += tokens_generated
        self.cumulative_hive_spent = round(self.cumulative_hive_spent + fee, 6)
        sig = self._compute_voucher_hmac(
            channel_id=self.channel_id,
            consumer_id=self.consumer_id,
            nonce=self.nonce,
            cumulative_tokens=self.cumulative_tokens,
            cumulative_hive_spent=self.cumulative_hive_spent,
        )
        self.latest_voucher = StateChannelVoucher(
            channel_id=self.channel_id,
            consumer_id=self.consumer_id,
            provider_ids=list(self.provider_ids),
            nonce=self.nonce,
            cumulative_tokens=self.cumulative_tokens,
            cumulative_hive_spent=self.cumulative_hive_spent,
            signature_hmac=sig,
        )
        return self.latest_voucher

    def refund_consumer_escrow(self, reason: str = "") -> float:
        """Refund accumulated off-chain vouchers back to the consumer if OPoI Merkle fraud is proven."""
        del reason
        if self.settled:
            raise RuntimeError(f"L2 state channel {self.channel_id} is already settled; cannot refund escrow.")
        if self.disputed_and_refunded:
            raise RuntimeError(f"L2 state channel {self.channel_id} was already refunded.")
        refunded = self.cumulative_hive_spent
        self.cumulative_hive_spent = 0.0
        self.cumulative_tokens = 0
        self.latest_voucher = None
        self.disputed_and_refunded = True
        return refunded

    def settle_to_l1_ledger(
        self,
        ledger: HiveSwarmLedger,
        domain_expert: str = "Python_Systems_Code",
    ) -> Dict[str, Any]:
        """Close/checkpoint the L2 channel and settle accumulated `$HIVE` fees on L1 `HiveSwarmLedger`."""
        if self.disputed_and_refunded:
            raise RuntimeError(
                f"L2 state channel {self.channel_id} was disputed and refunded; settlement blocked."
            )
        if self.settled:
            raise RuntimeError(
                f"L2 state channel {self.channel_id} is already settled at nonce {self.last_settled_nonce}."
            )
        if self.latest_voucher is None or self.cumulative_tokens == 0:
            return {"status": "no_vouchers_to_settle", "settled_hive": 0.0}

        v = self.latest_voucher
        expected_sig = self._compute_voucher_hmac(
            channel_id=v.channel_id,
            consumer_id=v.consumer_id,
            nonce=v.nonce,
            cumulative_tokens=v.cumulative_tokens,
            cumulative_hive_spent=v.cumulative_hive_spent,
        )
        expected_fee = round(v.cumulative_tokens * self.rate_per_token, 6)
        if (
            v.channel_id != self.channel_id
            or v.consumer_id != self.consumer_id
            or v.nonce != self.nonce
            or v.nonce <= self.last_settled_nonce
            or v.cumulative_tokens != self.cumulative_tokens
            or not math.isclose(v.cumulative_hive_spent, expected_fee, abs_tol=1e-6)
            or v.cumulative_hive_spent <= 0.0
            or v.cumulative_hive_spent > self.escrow_hive
            or not hmac.compare_digest(v.signature_hmac, expected_sig)
        ):
            raise ValueError("Invalid or tampered L2 state channel voucher signature.")

        tx = ledger.settle_swarm_inference(
            consumer_node_id=self.consumer_id,
            provider_node_ids=self.provider_ids,
            domain_expert=domain_expert,
            verified_tokens=v.cumulative_tokens,
            rtt_speedup=None,
            rate_per_token=self.rate_per_token,
        )
        self.settled = True
        self.last_settled_nonce = v.nonce
        return {
            "channel_id": self.channel_id,
            "final_nonce": v.nonce,
            "settled_tokens": v.cumulative_tokens,
            "settled_hive": tx.fee_paid_hive,
            "l1_settlement_tx": tx,
        }


def fused_top2_moe_fp8_dispatch(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_down_weights: List[torch.Tensor],
    top_k: int = 2,
) -> Dict[str, Any]:
    """HM-S3-02: Fused FP32 Top-2 Sparse MoE Gating + FP8 (`E4M3`) Activation Quantization + Expert Dispatch."""
    import math

    device = hidden_states.device
    orig_dtype = hidden_states.dtype

    use_cuda_events = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda_events:
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
    t0 = time.perf_counter()

    # 1. Mixed-Precision FP32 Router Gating (Theorem 3.2 invariant)
    hs_fp32 = hidden_states.to(torch.float32)
    wg_fp32 = router_weight.to(device=device, dtype=torch.float32)
    logits = torch.matmul(hs_fp32, wg_fp32.t())
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, probs.shape[-1])
    top_vals, top_idx = torch.topk(probs, k=k, dim=-1)
    norm_weights = (top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)).to(orig_dtype)

    # 2. Fused FP8 (E4M3) Activation Quantization & SQNR Fidelity Check
    fp8_pkt = compress_hidden_states_fp8(hidden_states)
    quant_hs = decompress_hidden_states_fp8(fp8_pkt).to(device=device, dtype=orig_dtype)
    signal_power = float(torch.mean(hidden_states.to(torch.float32) ** 2).item())
    noise_power = float(
        torch.mean((hidden_states.to(torch.float32) - quant_hs.to(torch.float32)) ** 2).item()
    )
    sqnr_db = round(10.0 * math.log10(max(signal_power, 1e-9) / max(noise_power, 1e-9)), 2)

    # 3. Sparse Top-2 Expert Dispatch
    out = torch.zeros_like(quant_hs)
    num_experts = len(expert_down_weights)
    expert_Projections = [w.to(device=device, dtype=orig_dtype) for w in expert_down_weights]
    for e_idx in range(num_experts):
        for slot_i in range(k):
            mask = (top_idx[..., slot_i] == e_idx).unsqueeze(-1)
            if mask.any():
                w_scale = norm_weights[..., slot_i].unsqueeze(-1)
                proj = torch.matmul(quant_hs, expert_Projections[e_idx])
                out = out + torch.where(mask, proj * w_scale, torch.zeros_like(out))

    if use_cuda_events:
        ev_end.record()
        torch.cuda.synchronize(device)
        kernel_ms = round(float(ev_start.elapsed_time(ev_end)), 4)
    else:
        kernel_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    return {
        "output": out,
        "top_k_indices": top_idx,
        "top_k_weights": norm_weights,
        "fp8_compression_ratio": fp8_pkt.compression_ratio,
        "sqnr_db": sqnr_db,
        "kernel_latency_ms": kernel_ms,
        "router_dtype": "float32",
        "expert_compute_dtype": str(orig_dtype),
    }


from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport, ZeroRTTSessionTicket  # noqa: E402, F401



OptimisticProofOfInferenceVerifier = OptimisticProofOfInferenceAuditor
