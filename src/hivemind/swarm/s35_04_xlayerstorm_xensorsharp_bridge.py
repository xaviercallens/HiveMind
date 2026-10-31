# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-04: xLayerStoRm (PCIe MoE Expert Streaming) x xensorSharp (.NET/Native Worker) x RunuX TurboQuant Bridge.

Integrates:
1. xLayerStoRm Fork Bridge (https://github.com/xaviercallens/xLayerStoRm):
   - Double-buffered PCIe expert streaming so frontier 32-expert Sparse MoEs execute on 8 GB consumer
     GPUs without exceeding the 1,536 MiB HiveMind citizen VRAM ceiling.
2. xensorSharp Fork Bridge (https://github.com/xaviercallens/xensorSharp):
   - Cross-platform native .NET / Windows / macOS / iOS / Linux GGUF worker manifest with zero WSL2
     or Python dependency overhead for gamer/homelab onboarding.
3. runux-ai-runtime (/home/xavkal/xdev/runux-ai-runtime):
   - TurboQuant polar/radial KV-cache compression reducing resident KV memory by 4.8x.
"""

from __future__ import annotations

from dataclasses import dataclass

CITIZEN_VRAM_CAP_MIB: float = 1536.0


@dataclass(frozen=True)
class PCIeMoEStreamingPlan:
    """Memory and latency schedule for xLayerStoRm + TurboQuant MoE execution under <= 1,536 MiB."""

    total_experts: int
    active_top_k_experts: int
    single_expert_mib: float
    shared_prelude_mib: float
    turboquant_kv_cache_mib: float
    double_buffer_staging_mib: float
    peak_active_vram_mib: float
    vram_headroom_mib: float
    pcie_bandwidth_gbps: float
    overlapped_compute_ms: float
    raw_pcie_transfer_ms: float
    effective_stall_ms: float
    overlap_efficiency_pct: float
    within_citizen_vram_cap: bool


@dataclass(frozen=True)
class NativeCrossPlatformWorkerSpec:
    """xensorSharp + HiveMind native 1-click citizen worker manifest."""

    os_family: str  # "Windows-x64-DX12/CUDA" | "macOS-ARM64-Metal" | "Linux-x64-CUDA" | "RISC-V-RunuX"
    runtime_engine: str
    requires_python_or_wsl: bool
    active_vram_cap_mib: float
    steam_preemption_ms: float
    openai_compatible_endpoint: str


def compute_xlayerstorm_turboquant_schedule(
    total_experts: int = 32,
    active_top_k_experts: int = 2,
    single_expert_fp16_mib: float = 280.0,
    e8_quant_compression: float = 3.2,
    shared_prelude_mib: float = 620.0,
    raw_kv_cache_mib: float = 1440.0,
    turboquant_ratio: float = 4.8,
    pcie_bandwidth_gbps: float = 14.0,
    layer_compute_ms: float = 11.5,
) -> PCIeMoEStreamingPlan:
    """Compute resident VRAM and overlapped PCIe streaming latency for a sparse MoE on a citizen GPU."""
    if active_top_k_experts < 1 or active_top_k_experts > total_experts:
        raise ValueError("Invalid active_top_k_experts")

    quant_expert_mib = single_expert_fp16_mib / e8_quant_compression
    active_experts_mib = active_top_k_experts * quant_expert_mib
    # Double-buffer staging slot holds the next layer's Top-K experts in pinned VRAM ring
    staging_mib = active_experts_mib
    turboquant_kv_mib = raw_kv_cache_mib / turboquant_ratio

    peak_vram = shared_prelude_mib + active_experts_mib + staging_mib + turboquant_kv_mib
    headroom = CITIZEN_VRAM_CAP_MIB - peak_vram

    transfer_mib = active_experts_mib
    raw_transfer_ms = (transfer_mib / 1024.0) / max(0.1, pcie_bandwidth_gbps) * 1000.0
    effective_stall_ms = max(0.0, raw_transfer_ms - layer_compute_ms)
    hidden_ms = min(raw_transfer_ms, layer_compute_ms)
    overlap_pct = 100.0 * (hidden_ms / max(1e-9, raw_transfer_ms))

    return PCIeMoEStreamingPlan(
        total_experts=total_experts,
        active_top_k_experts=active_top_k_experts,
        single_expert_mib=round(quant_expert_mib, 2),
        shared_prelude_mib=round(shared_prelude_mib, 2),
        turboquant_kv_cache_mib=round(turboquant_kv_mib, 2),
        double_buffer_staging_mib=round(staging_mib, 2),
        peak_active_vram_mib=round(peak_vram, 2),
        vram_headroom_mib=round(headroom, 2),
        pcie_bandwidth_gbps=pcie_bandwidth_gbps,
        overlapped_compute_ms=layer_compute_ms,
        raw_pcie_transfer_ms=round(raw_transfer_ms, 4),
        effective_stall_ms=round(effective_stall_ms, 4),
        overlap_efficiency_pct=round(overlap_pct, 2),
        within_citizen_vram_cap=(peak_vram <= CITIZEN_VRAM_CAP_MIB),
    )


def build_native_worker_fleet_manifests() -> tuple[NativeCrossPlatformWorkerSpec, ...]:
    """Return 1-click native citizen worker specifications across Windows, macOS, Linux, and RISC-V."""
    return (
        NativeCrossPlatformWorkerSpec(
            os_family="Windows-x64-DX12/CUDA",
            runtime_engine="xensorSharp-GGUF + xLayerStoRm-PCIe",
            requires_python_or_wsl=False,
            active_vram_cap_mib=1380.0,
            steam_preemption_ms=480.0,
            openai_compatible_endpoint="http://127.0.0.1:8787/v1",
        ),
        NativeCrossPlatformWorkerSpec(
            os_family="macOS-ARM64-Metal",
            runtime_engine="xensorSharp-Metal + OptimalEnergyAI-E8",
            requires_python_or_wsl=False,
            active_vram_cap_mib=1320.0,
            steam_preemption_ms=410.0,
            openai_compatible_endpoint="http://127.0.0.1:8787/v1",
        ),
        NativeCrossPlatformWorkerSpec(
            os_family="Linux-x64-CUDA",
            runtime_engine="HiveMind-LaRuche + xLayerStoRm-PCIe",
            requires_python_or_wsl=False,
            active_vram_cap_mib=1270.0,
            steam_preemption_ms=510.0,
            openai_compatible_endpoint="http://127.0.0.1:8787/v1",
        ),
        NativeCrossPlatformWorkerSpec(
            os_family="RISC-V-SpacemiT-K1/K3",
            runtime_engine="RunuX-AI-Runtime-v0.3 + TurboQuant",
            requires_python_or_wsl=False,
            active_vram_cap_mib=1024.0,
            steam_preemption_ms=320.0,
            openai_compatible_endpoint="http://127.0.0.1:8787/v1",
        ),
    )
