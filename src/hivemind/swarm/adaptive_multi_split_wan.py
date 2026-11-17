# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S29-01: Adaptive Multi-Split ML Protocol over Constrained WAN Links (Experiment E20).

Implements the surgical neural graph splitting controller specified in
`docs/VISION_AND_NEXTGEN_SWARM_ROADMAP.md` (§2 Rupture 1.1 & §3 HM-S29-01):
- Dynamically partitions execution between a citizen client Prelude (`<= 1,536 MiB`
  background VRAM ceiling) and a regional Edge Core (`Tier 2 / Tier 3` Super-Peer)
  across candidate cut-layers `L_c in {1, 2, 3, 4}`.
- Adapts `L_c` and boundary activation quantization (`INT8` vs `FP16`) in real time
  from measured QUIC uplink bandwidth (`10 Mbps` to `1,000 Mbps`), RTT (`10 ms` to
  `250 ms`), and jitter (`ms`).
- Enforces strict cohort isolation between pre-Ampere GPUs (`SM < 8.0`, e.g. GTX 1080 Ti
  running `FP32` master math) and Ampere/Ada/Blackwell GPUs (`SM >= 8.0` running `BF16`).
- Verifies Pre-Registered Experiment `E20` (step-time reduction under a constrained
  `20 Mbps / 60 ms RTT` residential WAN link vs. monolithic DiLoCo sync).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Sequence


MAX_PRELUDE_CLIENT_VRAM_MIB: float = 1536.0
MAX_GAMER_PREEMPTION_SLA_MS: float = 412.0
VALID_CUT_LAYERS: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class WANLinkTelemetry:
    """Measured QUIC WAN link state between a citizen Prelude node and an Edge Core peer."""

    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    jitter_ms: float = 2.0
    packet_loss_rate: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.uplink_mbps, bool)
            or not isinstance(self.uplink_mbps, (int, float))
            or self.uplink_mbps <= 0.0
            or not math.isfinite(self.uplink_mbps)
        ):
            raise ValueError("uplink_mbps must be finite and > 0")
        if (
            isinstance(self.downlink_mbps, bool)
            or not isinstance(self.downlink_mbps, (int, float))
            or self.downlink_mbps <= 0.0
            or not math.isfinite(self.downlink_mbps)
        ):
            raise ValueError("downlink_mbps must be finite and > 0")
        if (
            isinstance(self.rtt_ms, bool)
            or not isinstance(self.rtt_ms, (int, float))
            or self.rtt_ms < 0.0
            or not math.isfinite(self.rtt_ms)
        ):
            raise ValueError("rtt_ms must be finite and >= 0")
        if (
            isinstance(self.jitter_ms, bool)
            or not isinstance(self.jitter_ms, (int, float))
            or self.jitter_ms < 0.0
            or not math.isfinite(self.jitter_ms)
        ):
            raise ValueError("jitter_ms must be finite and >= 0")
        if (
            isinstance(self.packet_loss_rate, bool)
            or not isinstance(self.packet_loss_rate, (int, float))
            or not math.isfinite(self.packet_loss_rate)
            or not (0.0 <= self.packet_loss_rate < 1.0)
        ):
            raise ValueError("packet_loss_rate must be in [0.0, 1.0)")


@dataclass(frozen=True)
class CitizenHardwareProfile:
    """Hardware descriptor for a citizen GPU node participating in Adaptive Multi-Split ML."""

    node_id: str
    gpu_name: str
    compute_capability_sm: float
    vram_budget_mib: float = 1410.0
    local_tflops: float = 11.3
    foreground_game_active: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.gpu_name, str) or not self.gpu_name.strip():
            raise ValueError("gpu_name must be a non-empty string")
        if (
            isinstance(self.compute_capability_sm, bool)
            or not isinstance(self.compute_capability_sm, (int, float))
            or not math.isfinite(self.compute_capability_sm)
            or self.compute_capability_sm <= 0.0
        ):
            raise ValueError("compute_capability_sm must be finite and > 0")
        if (
            isinstance(self.vram_budget_mib, bool)
            or not isinstance(self.vram_budget_mib, (int, float))
            or not math.isfinite(self.vram_budget_mib)
            or self.vram_budget_mib <= 0.0
        ):
            raise ValueError("vram_budget_mib must be finite and > 0")
        if (
            isinstance(self.local_tflops, bool)
            or not isinstance(self.local_tflops, (int, float))
            or not math.isfinite(self.local_tflops)
            or self.local_tflops <= 0.0
        ):
            raise ValueError("local_tflops must be finite and > 0")
        if not isinstance(self.foreground_game_active, bool):
            raise ValueError("foreground_game_active must be a boolean")

    @property
    def precision_cohort(self) -> str:
        """Isolate pre-Ampere (`SM < 8.0` -> `FP32`) from Ampere+ (`SM >= 8.0` -> `BF16`)."""
        return "cohort_bf16_ampere_plus" if self.compute_capability_sm >= 8.0 else "cohort_fp32_pre_ampere"

    @property
    def dtype_bits(self) -> int:
        return 16 if self.compute_capability_sm >= 8.0 else 32


@dataclass(frozen=True)
class CutLayerDecision:
    """Real-time cut-layer selection and memory/latency budget for a single training/inference step."""

    node_id: str
    precision_cohort: str
    cut_layer_index: int
    activation_bits: int
    client_prelude_vram_mib: float
    boundary_payload_kb: float
    prelude_compute_ms: float
    wan_transfer_ms: float
    edge_core_compute_ms: float
    total_step_ms: float
    monolithic_baseline_step_ms: float
    step_time_reduction_pct: float
    preemption_latency_ms: float
    vram_compliant: bool
    preemption_compliant: bool

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.precision_cohort, str) or not self.precision_cohort.strip():
            raise ValueError("precision_cohort must be a non-empty string")
        if isinstance(self.cut_layer_index, bool) or self.cut_layer_index not in VALID_CUT_LAYERS:
            raise ValueError(f"cut_layer_index must be one of {VALID_CUT_LAYERS}")
        if isinstance(self.activation_bits, bool) or self.activation_bits not in (8, 16):
            raise ValueError("activation_bits must be 8 or 16")
        for field_name, num in (
            ("client_prelude_vram_mib", self.client_prelude_vram_mib),
            ("boundary_payload_kb", self.boundary_payload_kb),
            ("prelude_compute_ms", self.prelude_compute_ms),
            ("wan_transfer_ms", self.wan_transfer_ms),
            ("edge_core_compute_ms", self.edge_core_compute_ms),
            ("total_step_ms", self.total_step_ms),
            ("monolithic_baseline_step_ms", self.monolithic_baseline_step_ms),
            ("preemption_latency_ms", self.preemption_latency_ms),
        ):
            if isinstance(num, bool) or not isinstance(num, (int, float)) or not math.isfinite(float(num)) or float(num) <= 0.0:
                raise ValueError(f"{field_name} must be a positive finite number")
        if (
            isinstance(self.step_time_reduction_pct, bool)
            or not isinstance(self.step_time_reduction_pct, (int, float))
            or not math.isfinite(float(self.step_time_reduction_pct))
            or not (0.0 <= float(self.step_time_reduction_pct) <= 100.0)
        ):
            raise ValueError("step_time_reduction_pct must be a finite number in [0.0, 100.0]")
        if not isinstance(self.vram_compliant, bool) or not isinstance(self.preemption_compliant, bool):
            raise ValueError("vram_compliant and preemption_compliant must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Canonical cut-layer boundary profiles:
# Each cut layer L_c in {1..4} trades slightly more client Prelude compute/VRAM
# for a sharper bottleneck dimensionality reduction at the cut boundary.
CUT_LAYER_PROFILES: dict[int, dict[str, float]] = {
    1: {
        "base_vram_mib": 680.0,
        "bottleneck_dim": 768.0,
        "prelude_flops_g": 18.0,
        "edge_core_flops_g": 142.0,
    },
    2: {
        "base_vram_mib": 920.0,
        "bottleneck_dim": 512.0,
        "prelude_flops_g": 34.0,
        "edge_core_flops_g": 126.0,
    },
    3: {
        "base_vram_mib": 1160.0,
        "bottleneck_dim": 384.0,
        "prelude_flops_g": 52.0,
        "edge_core_flops_g": 108.0,
    },
    4: {
        "base_vram_mib": 1380.0,
        "bottleneck_dim": 256.0,
        "prelude_flops_g": 72.0,
        "edge_core_flops_g": 88.0,
    },
}


class AdaptiveMultiSplitWANController:
    """Real-time cut-layer controller for HM-S29-01 Adaptive Multi-Split ML over WAN."""

    def __init__(
        self,
        *,
        sequence_length: int = 512,
        micro_batch_size: int = 2,
        edge_core_tflops: float = 82.5,
        monolithic_delta_mb: float = 48.0,
        max_client_vram_mib: float = MAX_PRELUDE_CLIENT_VRAM_MIB,
    ) -> None:
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or sequence_length <= 0
            or micro_batch_size <= 0
        ):
            raise ValueError("sequence_length and micro_batch_size must be positive")
        if (
            isinstance(edge_core_tflops, bool)
            or not isinstance(edge_core_tflops, (int, float))
            or not math.isfinite(edge_core_tflops)
            or edge_core_tflops <= 0.0
        ):
            raise ValueError("edge_core_tflops must be finite and > 0")
        if (
            isinstance(monolithic_delta_mb, bool)
            or not isinstance(monolithic_delta_mb, (int, float))
            or not math.isfinite(monolithic_delta_mb)
            or monolithic_delta_mb <= 0.0
        ):
            raise ValueError("monolithic_delta_mb must be finite and > 0")
        if (
            isinstance(max_client_vram_mib, bool)
            or not isinstance(max_client_vram_mib, (int, float))
            or not math.isfinite(max_client_vram_mib)
            or max_client_vram_mib <= 0.0
        ):
            raise ValueError("max_client_vram_mib must be finite and > 0")
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self.edge_core_tflops = float(edge_core_tflops)
        self.monolithic_delta_mb = float(monolithic_delta_mb)
        self.max_client_vram_mib = float(max_client_vram_mib)

    def estimate_cut_layer_cost(
        self,
        cut_layer: int,
        link: WANLinkTelemetry,
        hw: CitizenHardwareProfile,
        *,
        activation_bits: int | None = None,
    ) -> CutLayerDecision:
        """Compute exact VRAM, wire payload, and step latency for a candidate cut-layer `L_c`."""
        if isinstance(cut_layer, bool) or cut_layer not in CUT_LAYER_PROFILES:
            raise ValueError(f"Unsupported cut_layer={cut_layer}; expected one of {VALID_CUT_LAYERS}")
        if not isinstance(link, WANLinkTelemetry):
            raise TypeError("link must be a WANLinkTelemetry")
        if not isinstance(hw, CitizenHardwareProfile):
            raise TypeError("hw must be a CitizenHardwareProfile")
        if activation_bits is not None and (
            isinstance(activation_bits, bool) or activation_bits not in (8, 16)
        ):
            raise ValueError("activation_bits must be 8 or 16")
        prof = CUT_LAYER_PROFILES[cut_layer]

        # When foreground gaming is active, force minimal VRAM footprint and fast preemption
        act_bits = activation_bits if activation_bits is not None else (8 if link.uplink_mbps < 150.0 else 16)
        dtype_scale = 1.08 if hw.dtype_bits == 32 else 1.0
        vram_mib = round(prof["base_vram_mib"] * dtype_scale, 2)

        # Forward activation + backward cut gradient across the boundary (2 passes per step)
        tokens_per_step = self.sequence_length * self.micro_batch_size
        raw_bits = 2.0 * tokens_per_step * prof["bottleneck_dim"] * act_bits
        retransmit_factor = 1.0 / max(0.25, 1.0 - link.packet_loss_rate)
        boundary_payload_kb = round((raw_bits / 8.0 / 1024.0) * retransmit_factor, 2)

        effective_bw_mbps = max(1.0, min(link.uplink_mbps, link.downlink_mbps) * (1.0 - 0.5 * link.packet_loss_rate))
        serialization_ms = (boundary_payload_kb * 8.0 / 1024.0) / effective_bw_mbps * 1000.0
        # Split step crosses WAN twice (forward cut activations + backward cut gradients)
        wan_transfer_ms = round(2.0 * link.rtt_ms + 1.5 * link.jitter_ms + serialization_ms, 2)

        prelude_compute_ms = round((prof["prelude_flops_g"] / max(1.0, hw.local_tflops)) * 12.0, 2)
        edge_core_compute_ms = round((prof["edge_core_flops_g"] / max(5.0, self.edge_core_tflops)) * 12.0, 2)
        total_step_ms = round(prelude_compute_ms + wan_transfer_ms + edge_core_compute_ms, 2)

        # Monolithic baseline: client computes entire branch and amortizes outer DiLoCo delta upload across H=12 inner steps
        amortized_delta_mb = self.monolithic_delta_mb / 12.0
        mono_compute_ms = round(((prof["prelude_flops_g"] + prof["edge_core_flops_g"]) / max(1.0, hw.local_tflops)) * 12.0, 2)
        mono_wan_ms = round(link.rtt_ms + (amortized_delta_mb * 8.0 / effective_bw_mbps) * 1000.0, 2)
        monolithic_step_ms = round(mono_compute_ms + mono_wan_ms, 2)

        reduction_pct = round(
            max(0.0, (monolithic_step_ms - total_step_ms) / max(1e-6, monolithic_step_ms) * 100.0),
            2,
        )
        preemption_ms = round(140.0 + 0.16 * vram_mib, 2)
        allowed_vram = min(self.max_client_vram_mib, hw.vram_budget_mib)

        return CutLayerDecision(
            node_id=hw.node_id,
            precision_cohort=hw.precision_cohort,
            cut_layer_index=cut_layer,
            activation_bits=act_bits,
            client_prelude_vram_mib=vram_mib,
            boundary_payload_kb=boundary_payload_kb,
            prelude_compute_ms=prelude_compute_ms,
            wan_transfer_ms=wan_transfer_ms,
            edge_core_compute_ms=edge_core_compute_ms,
            total_step_ms=total_step_ms,
            monolithic_baseline_step_ms=monolithic_step_ms,
            step_time_reduction_pct=reduction_pct,
            preemption_latency_ms=preemption_ms,
            vram_compliant=(vram_mib <= allowed_vram),
            preemption_compliant=(preemption_ms <= MAX_GAMER_PREEMPTION_SLA_MS),
        )

    def select_optimal_cut_layer(
        self,
        link: WANLinkTelemetry,
        hw: CitizenHardwareProfile,
    ) -> CutLayerDecision:
        """Select the cut layer `L_c in {1..4}` minimizing total step time under VRAM and preemption ceilings."""
        if not isinstance(link, WANLinkTelemetry):
            raise TypeError("link must be a WANLinkTelemetry")
        if not isinstance(hw, CitizenHardwareProfile):
            raise TypeError("hw must be a CitizenHardwareProfile")
        if hw.foreground_game_active:
            # Immediate gamer-safe shallowest cut layer L_c = 1 with INT8 boundary
            return self.estimate_cut_layer_cost(1, link, hw, activation_bits=8)

        candidates = [self.estimate_cut_layer_cost(lc, link, hw) for lc in VALID_CUT_LAYERS]
        compliant = [c for c in candidates if c.vram_compliant and c.preemption_compliant]
        pool = compliant if compliant else candidates
        return min(pool, key=lambda c: c.total_step_ms)

    def quantize_cut_activations_int8(self, activations: Sequence[float]) -> dict[str, Any]:
        """Symmetric per-tensor INT8 quantization at the Prelude -> Edge Core cut boundary."""
        if isinstance(activations, (str, bytes)) or not isinstance(activations, Sequence):
            raise TypeError("activations must be a Sequence of finite numbers")
        vals: list[float] = []
        for x in activations:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise ValueError("activations must contain finite numbers")
            vals.append(float(x))
        if not vals:
            return {"q_int8": [], "scale": 1.0, "dequantized": [], "relative_l2_error": 0.0, "sha256": ""}
        max_abs = max(abs(x) for x in vals)
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
        q_int8 = [max(-127, min(127, int(round(x / scale)))) for x in vals]
        dequant = [round(q * scale, 6) for q in q_int8]
        num = math.sqrt(sum((a - b) ** 2 for a, b in zip(vals, dequant)))
        den = math.sqrt(sum(a * a for a in vals)) or 1.0
        digest = hashlib.sha256(",".join(str(q) for q in q_int8).encode("utf-8")).hexdigest()
        return {
            "q_int8": q_int8,
            "scale": round(scale, 8),
            "dequantized": dequant,
            "relative_l2_error": round(num / den, 6),
            "compression_vs_fp32": 4.0,
            "sha256": digest,
        }


def run_e20_adaptive_multi_split_experiment(
    link: WANLinkTelemetry | None = None,
    hardware_profiles: Sequence[CitizenHardwareProfile] | None = None,
) -> dict[str, Any]:
    """Run Pre-Registered Experiment E20 across supplied or default WAN and hardware profiles."""
    if link is not None and not isinstance(link, WANLinkTelemetry):
        raise TypeError("link must be a WANLinkTelemetry or None")
    if hardware_profiles is not None and (
        isinstance(hardware_profiles, (str, bytes))
        or not isinstance(hardware_profiles, Sequence)
        or not all(isinstance(h, CitizenHardwareProfile) for h in hardware_profiles)
    ):
        raise TypeError("hardware_profiles must be a Sequence of CitizenHardwareProfile or None")
    controller = AdaptiveMultiSplitWANController()
    target_link = link if link is not None else WANLinkTelemetry(
        uplink_mbps=20.0, downlink_mbps=50.0, rtt_ms=60.0, jitter_ms=6.0
    )
    fiber_link = WANLinkTelemetry(uplink_mbps=300.0, downlink_mbps=500.0, rtt_ms=14.0, jitter_ms=1.2)

    if hardware_profiles is not None and len(hardware_profiles) >= 2:
        pascal_node = hardware_profiles[0]
        ampere_node = hardware_profiles[1]
    else:
        pascal_node = CitizenHardwareProfile(
            node_id="node-gtx1080ti-pascal",
            gpu_name="NVIDIA GeForce GTX 1080 Ti",
            compute_capability_sm=6.1,
            vram_budget_mib=1500.0,
            local_tflops=11.3,
        )
        ampere_node = CitizenHardwareProfile(
            node_id="node-rtx3060-ampere",
            gpu_name="NVIDIA GeForce RTX 3060",
            compute_capability_sm=8.6,
            vram_budget_mib=1500.0,
            local_tflops=12.7,
        )

    constrained_pascal = controller.select_optimal_cut_layer(target_link, pascal_node)
    constrained_ampere = controller.select_optimal_cut_layer(target_link, ampere_node)
    fiber_ampere = controller.select_optimal_cut_layer(fiber_link, ampere_node)

    sample_act = [math.sin(i * 0.17) * 2.4 for i in range(64)]
    q_stats = controller.quantize_cut_activations_int8(sample_act)

    mean_reduction_20mbps = round(
        (constrained_pascal.step_time_reduction_pct + constrained_ampere.step_time_reduction_pct) / 2.0,
        2,
    )
    e20_gate_passed = (
        mean_reduction_20mbps >= 55.0
        and constrained_pascal.vram_compliant
        and constrained_ampere.vram_compliant
        and constrained_pascal.preemption_compliant
        and constrained_ampere.preemption_compliant
        and constrained_pascal.precision_cohort != constrained_ampere.precision_cohort
    )

    return {
        "experiment_id": "E20",
        "task_id": "HM-S29-01",
        "measurement_kind": "closed_form_model",
        "e20_gate_passed": e20_gate_passed,
        "mean_step_time_reduction_pct_at_20mbps_60ms": mean_reduction_20mbps,
        "constrained_20mbps_pascal": constrained_pascal.to_dict(),
        "constrained_20mbps_ampere": constrained_ampere.to_dict(),
        "fiber_300mbps_ampere": fiber_ampere.to_dict(),
        "int8_boundary_quantization": q_stats,
    }
