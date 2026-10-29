# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Swarm Fabric M1: AsyncQUICGeoRingTransport with 0-RTT Session Resumption & WebSocket Fallback (`HM-M1-01`).

Integrates contributor PRs #9 and #22 (@clark719664), closing Issues #1 and #14:
- `CompressedHiddenStatePacket` frames over real QUIC (`aioquic`) with TLS 1.3 0-RTT session
  resumption, and a libp2p multistream-select WebSocket fallback when UDP is blocked (`quic_link`)
- Binary frame format (`MAGIC_QUIC_0RTT = b"HQ0R"`, `MAGIC_WS_FALLBACK = b"HWSF"`)
- RFC 5389 STUN discovery, RFC 5780 mapping classification, UDP hole punching and RFC 5766 TURN
  relay fallback (`nat_traversal`), plus ICE (RFC 8445/8839) host and server-reflexive candidates

Season 4 Milestone `HM-S4-02` (Issue #27, branch `feat/s4-hierarchical-super-ring`) extends this
transport with the 2-Tier Kademlia Continental Super-Ring:
- `ContinentalSuperRingRouter` aggregates regional Geo-Rings (`EU`/`US`/`APAC`) and elects the
  lowest-RTT peer of every region that the caller has flagged `stun_verified` (this module does
  not run STUN itself) as a *Continental Bridge* (Tier-2), while
  intra-region traffic stays inside the `< 20 ms` Tier-1 Regional Geo-Ring.
- `AdaptiveCongestionController` implements a BBR-inspired (BBRv1 delivery-rate model) adaptive
  congestion window for `CompressedHiddenStatePacket` bursts driven by measured RTT jitter and
  packet loss.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
import socket
import struct
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch

from hivemind.swarm.nat_traversal import (
    DEFAULT_STUN_SERVERS,
    Address,
    PunchResult,
    ServerSpec,
    StunError,
    TurnClient,
    classify_nat_mapping,
    outbound_interface_ip,
    resolve,
    stun_binding,
    udp_hole_punch,
    udp_socket,
)
try:
    from hivemind.swarm.quic_link import LinkRun, run_local_link
except ImportError:
    LinkRun = Any  # type: ignore[misc,assignment]
    run_local_link = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from hivemind.swarm.quic_link import LinkEndpoint, RemoteHiddenStateLink

from hivemind.swarm.swarm_fabric import (
    CompressedHiddenStatePacket,
    RegionalGeoRingTransport,
    compress_hidden_states_fp8,
    decompress_hidden_states_fp8,
)

MAGIC_QUIC_0RTT = b"HQ0R"
MAGIC_WS_FALLBACK = b"HWSF"

_T = TypeVar("_T")


def _run_coroutine(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run `coro` to completion from synchronous code, even if this thread already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    outcome: Dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=runner)
    worker.start()
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@dataclass
class ZeroRTTSessionTicket:
    """0-RTT session ticket issued for low-latency QUIC hidden-state streaming (`HM-M1-01` & `HM-S3-01`)."""

    peer_id: str
    session_token: bytes
    issued_at: float
    resumed_count: int = 0
    reflexive_ip: str = "127.0.0.1"
    reflexive_port: int = 4433
    nat_type: str = "full_cone"
    turn_relay_endpoint: Optional[str] = None
    last_accepted_nonce: int = 0


# ============================================================================
# HM-S4-02: 2-Tier Kademlia Continental Super-Ring & BBR-Inspired Congestion Control
# ============================================================================

CONTINENT_CODES: Tuple[str, ...] = ("EU", "US", "APAC")
"""Continental Geo-Ring buckets aggregated by the Tier-2 Super-Ring."""

CONTINENT_ALIASES: Dict[str, str] = {
    "AP": "APAC",
    "ASIA": "APAC",
    "AU": "APAC",
    "CN": "APAC",
    "IN": "APAC",
    "JP": "APAC",
    "KR": "APAC",
    "SG": "APAC",
    "CA": "US",
    "MX": "US",
    "US": "US",
    "USA": "US",
    "DE": "EU",
    "ES": "EU",
    "EU": "EU",
    "FR": "EU",
    "NL": "EU",
    "UK": "EU",
}
"""Region-prefix -> continent alias table used by the continental Geo-Ring aggregator."""

DEFAULT_INTER_CONTINENT_RTT_MS: Dict[str, Dict[str, float]] = {
    "EU": {"US": 78.0, "APAC": 168.0},
    "US": {"EU": 78.0, "APAC": 132.0},
    "APAC": {"EU": 168.0, "US": 132.0},
}
"""Default WAN round-trip estimates between Continental Bridge pairs (Paris/Virginia/Tokyo)."""


def classify_continent(region: str) -> str:
    """Map a regional Geo-Ring label (e.g. `EU-West-Paris`) to its continental bucket.

    Returns one of `EU`, `US`, `APAC` or `OTHER` when the region cannot be mapped to a
    known continental super-ring (`HM-S4-02`).
    """
    head = (region or "").strip().split("-", 1)[0].upper()
    if head in CONTINENT_CODES:
        return head
    return CONTINENT_ALIASES.get(head, "OTHER")


@dataclass
class SuperRingPeerNode:
    """Tier-1 Regional Geo-Ring peer tracked by the 2-Tier Continental Super-Ring (`HM-S4-02`)."""

    peer_id: str
    region: str
    rtt_ms: float
    jitter_ms: float = 0.0
    packet_loss_ratio: float = 0.0
    stun_verified: bool = False
    nat_type: str = "full_cone"
    kademlia_id_hex: str = ""

    def __post_init__(self) -> None:
        if not self.kademlia_id_hex:
            self.kademlia_id_hex = hashlib.sha1(self.peer_id.encode("utf-8")).hexdigest()

    @property
    def continent(self) -> str:
        """Continental super-ring bucket (`EU`/`US`/`APAC`) this peer belongs to."""
        return classify_continent(self.region)

    @property
    def bridge_eligible(self) -> bool:
        """Bridge candidates must be flagged `stun_verified` by whoever built the peer list and have a measured RTT."""
        return bool(self.stun_verified) and self.rtt_ms > 0.0


@dataclass
class RegionalBridgePeer:
    """Candidate continental bridge (`HM-S4-02`). `stun_verified` is caller-supplied; no STUN probe runs here."""

    peer_id: str
    continent: str
    rtt_ms: float
    stun_verified: bool = True


class ContinentalSuperRingRouter:
    """HM-S4-02: 2-Tier Kademlia Super-Ring (`Continental Bridge -> Regional <20 ms Geo-Ring`).

    Tier 1 groups peers into regional Geo-Rings (`EU-West-Paris`, `US-East-Virginia`,
    `APAC-East-Tokyo`, ...) that always stay below the `< 20 ms` intra-region SLA. Tier 2
    aggregates those rings per continent (`EU`, `US`, `APAC`) through an elected *Continental
    Bridge*: the lowest-RTT **STUN-verified** peer of the region (and, per continent, of all
    its regions). Cross-continent traffic is routed `source -> Continental Bridge (src)
    -> WAN link -> Continental Bridge (dst) -> target`.
    """

    CONTINENTS = ("EU", "US", "APAC")

    def __init__(
        self,
        max_regional_rtt_ms: Any = 20.0,
        inter_continent_rtt_ms: Optional[Dict[str, Dict[str, float]]] = None,
        max_bridge_jitter_ms: float = 5.0,
    ) -> None:
        self.bridge_peers: List[RegionalBridgePeer] = []
        if isinstance(max_regional_rtt_ms, (list, tuple)):
            self.bridge_peers = list(max_regional_rtt_ms)
            self.max_regional_rtt_ms = 20.0
        else:
            self.max_regional_rtt_ms = float(max_regional_rtt_ms)
        self.max_bridge_jitter_ms = float(max_bridge_jitter_ms)
        self.inter_continent_rtt_table: Dict[str, Dict[str, float]] = (
            dict(inter_continent_rtt_ms)
            if inter_continent_rtt_ms is not None
            else {k: dict(v) for k, v in DEFAULT_INTER_CONTINENT_RTT_MS.items()}
        )
        self.peers: Dict[str, SuperRingPeerNode] = {}
        self.rings: Dict[str, List[SuperRingPeerNode]] = {}
        self.regional_bridges: Dict[str, str] = {}
        self.continental_bridges: Dict[str, str] = {}
        self.routing_log: List[Dict[str, Any]] = []
        for bp in self.bridge_peers:
            self.register_peer(
                SuperRingPeerNode(
                    peer_id=bp.peer_id,
                    region=bp.continent,
                    rtt_ms=float(bp.rtt_ms),
                    stun_verified=bool(bp.stun_verified),
                )
            )

    def elect_bridges(self) -> Dict[str, RegionalBridgePeer]:
        """Elect the lowest-RTT peer flagged `stun_verified` in each continent (`HM-S4-02`)."""
        bridges: Dict[str, RegionalBridgePeer] = {}
        source_pool = self.bridge_peers or [
            RegionalBridgePeer(
                peer_id=p.peer_id,
                continent=p.continent,
                rtt_ms=p.rtt_ms,
                stun_verified=p.stun_verified,
            )
            for p in self.peers.values()
        ]
        for continent in self.CONTINENTS:
            verified = [
                peer for peer in source_pool if peer.continent == continent and peer.stun_verified
            ]
            if verified:
                bridges[continent] = min(verified, key=lambda peer: peer.rtt_ms)
        return bridges

    @staticmethod
    def bbr_window(base_window: int, jitter_ms: float, loss: float) -> int:
        """Shrink the burst window when RTT jitter or loss rises. Floor is 1 packet (`HM-S4-02`)."""
        if base_window < 1:
            raise ValueError("base_window must be positive")
        if loss < 0.0 or loss > 1.0:
            raise ValueError("loss must be in [0, 1]")
        scale = 1.0
        if jitter_ms > 5.0:
            scale *= 0.5
        elif jitter_ms > 1.0:
            scale *= 0.75
        scale *= max(0.25, 1.0 - loss)
        return max(1, int(base_window * scale))

    # ------------------------------------------------------------------
    # Tier 1: Regional Geo-Ring membership
    # ------------------------------------------------------------------
    def register_peer(self, node: SuperRingPeerNode) -> bool:
        """Admit `node` into its Tier-1 Regional Geo-Ring (only if the `<20 ms` SLA holds)."""
        if node.rtt_ms > self.max_regional_rtt_ms:
            return False
        existing = self.peers.get(node.peer_id)
        if existing is not None:
            self.rings[existing.region].remove(existing)
        self.peers[node.peer_id] = node
        ring = self.rings.setdefault(node.region, [])
        if node not in ring:
            ring.append(node)
        return True

    def remove_peer(self, peer_id: str) -> bool:
        """Evict a peer (e.g. after a failed STUN re-probe) and re-elect the bridges."""
        node = self.peers.pop(peer_id, None)
        if node is None:
            return False
        ring = self.rings.get(node.region, [])
        if node in ring:
            ring.remove(node)
        if not ring:
            self.rings.pop(node.region, None)
        self.refresh_bridges()
        return True

    def peers_in_region(self, region: str) -> List[SuperRingPeerNode]:
        """All Tier-1 peers currently registered in `region`."""
        return list(self.rings.get(region, []))

    def find_peer_in_region(self, region: str) -> Optional[SuperRingPeerNode]:
        """Lowest-RTT peer of `region` (used as the default regional entry point)."""
        members = self.peers_in_region(region)
        if not members:
            return None
        return sorted(members, key=lambda p: (p.rtt_ms, p.jitter_ms, p.peer_id))[0]

    # ------------------------------------------------------------------
    # Tier 2: Continental Bridge election
    # ------------------------------------------------------------------
    def elect_regional_bridges(self) -> Dict[str, str]:
        """Elect the lowest-RTT STUN-verified peer of every regional Geo-Ring."""
        bridges: Dict[str, str] = {}
        for region, members in self.rings.items():
            candidates = [p for p in members if p.bridge_eligible]
            if not candidates:
                continue
            candidates.sort(key=lambda p: (p.rtt_ms, p.jitter_ms, p.peer_id))
            bridges[region] = candidates[0].peer_id
        self.regional_bridges = bridges
        return dict(bridges)

    def elect_continental_bridges(self) -> Dict[str, str]:
        """Aggregate the regional bridges into one Continental Bridge per continent."""
        if not self.regional_bridges:
            self.elect_regional_bridges()
        per_continent: Dict[str, List[SuperRingPeerNode]] = {}
        for region, peer_id in self.regional_bridges.items():
            node = self.peers.get(peer_id)
            if node is None:
                continue
            per_continent.setdefault(node.continent, []).append(node)
        bridges: Dict[str, str] = {}
        for continent, nodes in per_continent.items():
            nodes.sort(key=lambda p: (p.rtt_ms, p.jitter_ms, p.peer_id))
            bridges[continent] = nodes[0].peer_id
        self.continental_bridges = bridges
        return dict(bridges)

    def refresh_bridges(self) -> Dict[str, str]:
        """Re-run both election rounds (regional -> continental) and return the Tier-2 bridges."""
        self.elect_regional_bridges()
        return self.elect_continental_bridges()

    def inter_continent_rtt_ms(self, source_continent: str, target_continent: str) -> float:
        """Measured (or default) WAN RTT between two continental bridge rings."""
        if source_continent == target_continent:
            return 0.0
        table = self.inter_continent_rtt_table.get(source_continent, {})
        if target_continent in table:
            return float(table[target_continent])
        reverse = self.inter_continent_rtt_table.get(target_continent, {})
        if source_continent in reverse:
            return float(reverse[source_continent])
        return 0.0

    def record_inter_continent_rtt(
        self, source_continent: str, target_continent: str, rtt_ms: float
    ) -> None:
        """Store a freshly measured WAN RTT between two continental bridge rings."""
        self.inter_continent_rtt_table.setdefault(source_continent, {})[target_continent] = float(
            rtt_ms
        )
        self.inter_continent_rtt_table.setdefault(target_continent, {})[source_continent] = float(
            rtt_ms
        )

    # ------------------------------------------------------------------
    # 2-Tier routing
    # ------------------------------------------------------------------
    def route(self, source_peer_id: str, target_peer_id: str) -> Dict[str, Any]:
        """Resolve the 2-Tier path between two peers (or two continents) of the Super-Ring.

        Same-region peers stay inside the Tier-1 Regional Geo-Ring (`< 20 ms`), while
        cross-continent peers are routed through the elected Continental Bridges.
        """
        if source_peer_id not in self.peers and target_peer_id not in self.peers:
            bridges = self.elect_bridges()
            if source_peer_id == target_peer_id:
                local = bridges.get(source_peer_id)
                rtt = 0.0 if local is None else local.rtt_ms
                return {
                    "hops": ["regional"],
                    "hop_count": 1,
                    "rtt_ms": round(rtt, 3),
                    "bridge_peer_id": None,
                }
            if source_peer_id not in bridges or target_peer_id not in bridges:
                raise KeyError(f"no STUN-verified bridge for {source_peer_id} -> {target_peer_id}")
            src = bridges[source_peer_id]
            dst = bridges[target_peer_id]
            bridge_rtt = max(src.rtt_ms, dst.rtt_ms)
            total = src.rtt_ms + bridge_rtt + dst.rtt_ms
            return {
                "hops": ["regional", "continental_bridge", "remote_regional"],
                "hop_count": 3,
                "rtt_ms": round(total, 3),
                "bridge_peer_id": src.peer_id if src.rtt_ms <= dst.rtt_ms else dst.peer_id,
                "bound_rtt_ms": round(src.rtt_ms + bridge_rtt + dst.rtt_ms, 3),
            }

        source = self.peers.get(source_peer_id)
        target = self.peers.get(target_peer_id)
        if source is None:
            raise KeyError(f"Unknown Super-Ring peer: {source_peer_id}")
        if target is None:
            raise KeyError(f"Unknown Super-Ring peer: {target_peer_id}")

        self.refresh_bridges()

        if source.region == target.region:
            estimated = source.rtt_ms + target.rtt_ms
            route = {
                "tier": "regional_geo_ring",
                "cross_continent": False,
                "source_peer_id": source_peer_id,
                "target_peer_id": target_peer_id,
                "source_region": source.region,
                "target_region": target.region,
                "source_continent": source.continent,
                "target_continent": target.continent,
                "path": [source_peer_id, target_peer_id],
                "regional_bridge_peer_ids": [],
                "continental_bridge_peer_ids": [],
                "inter_continent_rtt_ms": 0.0,
                "estimated_rtt_ms": round(estimated, 3),
                "regional_sla_ms": self.max_regional_rtt_ms,
                "within_regional_sla": estimated < self.max_regional_rtt_ms,
                "wan_transit": False,
            }
        else:
            source_bridge_id = self.continental_bridges.get(source.continent)
            target_bridge_id = self.continental_bridges.get(target.continent)
            if source_bridge_id is None or target_bridge_id is None:
                # No STUN-verified bridge for one of the continents: stay on Tier-1.
                estimated = source.rtt_ms + target.rtt_ms
                route = {
                    "tier": "regional_geo_ring",
                    "cross_continent": False,
                    "source_peer_id": source_peer_id,
                    "target_peer_id": target_peer_id,
                    "source_region": source.region,
                    "target_region": target.region,
                    "source_continent": source.continent,
                    "target_continent": target.continent,
                    "path": [source_peer_id, target_peer_id],
                    "regional_bridge_peer_ids": [],
                    "continental_bridge_peer_ids": [],
                    "inter_continent_rtt_ms": 0.0,
                    "estimated_rtt_ms": round(estimated, 3),
                    "regional_sla_ms": self.max_regional_rtt_ms,
                    "within_regional_sla": estimated < self.max_regional_rtt_ms,
                    "wan_transit": False,
                }
            else:
                wan_rtt = self.inter_continent_rtt_ms(source.continent, target.continent)
                source_leg = (
                    0.0
                    if source_bridge_id == source_peer_id
                    else self.peers[source_bridge_id].rtt_ms
                )
                target_leg = (
                    0.0
                    if target_bridge_id == target_peer_id
                    else self.peers[target_bridge_id].rtt_ms
                )
                estimated = source_leg + wan_rtt + target_leg
                path = [source_peer_id]
                if source_bridge_id != source_peer_id:
                    path.append(source_bridge_id)
                if target_bridge_id != source_peer_id:
                    path.append(target_bridge_id)
                path.append(target_peer_id)
                route = {
                    "tier": "continental_bridge",
                    "cross_continent": True,
                    "source_peer_id": source_peer_id,
                    "target_peer_id": target_peer_id,
                    "source_region": source.region,
                    "target_region": target.region,
                    "source_continent": source.continent,
                    "target_continent": target.continent,
                    "path": path,
                    "regional_bridge_peer_ids": [
                        p
                        for p in (source_bridge_id, target_bridge_id)
                        if p in self.regional_bridges.values()
                    ],
                    "continental_bridge_peer_ids": [source_bridge_id, target_bridge_id],
                    "inter_continent_rtt_ms": round(wan_rtt, 3),
                    "estimated_rtt_ms": round(estimated, 3),
                    "regional_sla_ms": self.max_regional_rtt_ms,
                    "within_regional_sla": estimated < self.max_regional_rtt_ms,
                    "wan_transit": True,
                }

        route["hop_count"] = max(0, len(route["path"]) - 1)
        self.routing_log.append(route)
        return route

    def find_closest_continental_bridges(
        self, target_key: str, k: int = 2
    ) -> List[SuperRingPeerNode]:
        """Kademlia XOR lookup over the Tier-2 Continental Bridge ring."""
        self.refresh_bridges()
        target_hex = hashlib.sha1(target_key.encode("utf-8")).hexdigest()
        bridges = [
            self.peers[pid]
            for pid in self.continental_bridges.values()
            if pid in self.peers
        ]
        bridges.sort(
            key=lambda p: (
                RegionalGeoRingTransport.xor_distance(p.kademlia_id_hex, target_hex),
                p.rtt_ms,
            )
        )
        return bridges[: max(1, k)]

    def describe_topology(self) -> Dict[str, Any]:
        """Snapshot of the 2-Tier Super-Ring topology (Tier-1 rings + Tier-2 bridges)."""
        self.refresh_bridges()
        regions: Dict[str, Dict[str, Any]] = {}
        for region, members in self.rings.items():
            bridge_id = self.regional_bridges.get(region)
            bridge = self.peers.get(bridge_id) if bridge_id else None
            regions[region] = {
                "continent": classify_continent(region),
                "peer_count": len(members),
                "regional_bridge_peer_id": bridge_id,
                "regional_bridge_rtt_ms": round(bridge.rtt_ms, 3) if bridge else None,
                "stun_verified_peers": sum(1 for p in members if p.stun_verified),
                "mean_rtt_ms": round(sum(p.rtt_ms for p in members) / len(members), 3)
                if members
                else None,
            }
        continents: Dict[str, Dict[str, Any]] = {}
        for continent, bridge_id in self.continental_bridges.items():
            bridge = self.peers.get(bridge_id)
            continents[continent] = {
                "continental_bridge_peer_id": bridge_id,
                "continental_bridge_region": bridge.region if bridge else None,
                "continental_bridge_rtt_ms": round(bridge.rtt_ms, 3) if bridge else None,
                "jitter_ms": round(bridge.jitter_ms, 3) if bridge else None,
                "regional_bridges": sorted(
                    region
                    for region, pid in self.regional_bridges.items()
                    if pid in self.peers and self.peers[pid].continent == continent
                ),
            }
        return {
            "tier1_regional_rings": regions,
            "tier2_continental_bridges": continents,
            "regional_ring_count": len(self.rings),
            "continental_bridge_count": len(self.continental_bridges),
            "total_peer_count": len(self.peers),
            "regional_sla_ms": self.max_regional_rtt_ms,
        }


@dataclass
class CongestionWindowTelemetry:
    """Snapshot of the BBR-inspired congestion controller state (`HM-S4-02`)."""

    cwnd_bytes: int
    cwnd_packets: int
    btl_bw_bps: float
    rtprop_ms: float
    jitter_ms: float
    loss_ratio: float
    gain: float
    mtu_bytes: int
    mode: str = "steady"

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the telemetry (JSON friendly)."""
        return {
            "cwnd_bytes": int(self.cwnd_bytes),
            "cwnd_packets": int(self.cwnd_packets),
            "btl_bw_bps": round(float(self.btl_bw_bps), 3),
            "rtprop_ms": round(float(self.rtprop_ms), 3),
            "jitter_ms": round(float(self.jitter_ms), 3),
            "loss_ratio": round(float(self.loss_ratio), 4),
            "gain": round(float(self.gain), 4),
            "mtu_bytes": int(self.mtu_bytes),
            "mode": self.mode,
        }


class AdaptiveCongestionController:
    """HM-S4-02: BBR-inspired adaptive congestion window for FP8 hidden-state bursts.

    The controller mirrors the BBRv1 delivery-rate model:
    - `BTlBw` (bottleneck bandwidth) = maximum delivery rate over the last
      `bandwidth_window` samples (BBR *max filter*).
    - `RTprop` (round-trip propagation time) = minimum RTT over the last `rtt_window`
      samples (BBR *min filter*).
    - `cwnd = gain * BTlBw * RTprop` with `gain = 2.0` during startup and `gain = 1.25`
      in steady state.

    Measured RTT **jitter** inflates the queueing-delay estimate and therefore shrinks the
    gain, while measured **packet loss** triggers a multiplicative decrease (`loss_beta`).
    """

    def __init__(
        self,
        mtu_bytes: int = 1200,
        initial_cwnd_packets: int = 10,
        startup_gain: float = 2.0,
        steady_gain: float = 1.25,
        loss_beta: float = 0.7,
        loss_sensitivity: float = 2.0,
        min_cwnd_packets: int = 2,
        max_cwnd_packets: int = 512,
        bandwidth_window: int = 10,
        rtt_window: int = 10,
        startup_packets: int = 20,
    ) -> None:
        self.mtu_bytes = int(mtu_bytes)
        self.initial_cwnd_packets = int(initial_cwnd_packets)
        self.startup_gain = float(startup_gain)
        self.steady_gain = float(steady_gain)
        self.loss_beta = float(loss_beta)
        self.loss_sensitivity = float(loss_sensitivity)
        self.min_cwnd_packets = int(min_cwnd_packets)
        self.max_cwnd_packets = int(max_cwnd_packets)
        self.bandwidth_window = int(bandwidth_window)
        self.rtt_window = int(rtt_window)
        self.startup_packets = int(startup_packets)

        self.rtt_samples: List[float] = []
        self.delivery_samples: List[Tuple[float, float]] = []
        self.btl_bw_bps: float = 0.0
        self.rtprop_ms: float = 0.0
        self.jitter_ms: float = 0.0
        self.loss_ratio: float = 0.0
        self.cwnd_packets: float = float(initial_cwnd_packets)
        self.packets_acked: int = 0
        self.in_startup: bool = True

    # ------------------------------------------------------------------
    # Feedback signals
    # ------------------------------------------------------------------
    def record_rtt_sample(self, rtt_ms: float) -> None:
        """Feed an RTT sample into the `RTprop` min-filter and the jitter estimator."""
        rtt = float(rtt_ms)
        if rtt <= 0.0:
            return
        self.rtt_samples.append(rtt)
        if len(self.rtt_samples) > self.rtt_window:
            self.rtt_samples.pop(0)
        self.rtprop_ms = min(self.rtt_samples)
        mean = sum(self.rtt_samples) / len(self.rtt_samples)
        deviation = sum(abs(s - mean) for s in self.rtt_samples) / len(self.rtt_samples)
        self.jitter_ms = deviation

    def record_delivery(self, bytes_delivered: int, elapsed_ms: float) -> None:
        """Record a delivery-rate sample (`bytes / elapsed`) for the `BTlBw` max-filter."""
        if bytes_delivered <= 0 or elapsed_ms <= 0:
            return
        self.delivery_samples.append((float(bytes_delivered), float(elapsed_ms)))
        if len(self.delivery_samples) > self.bandwidth_window:
            self.delivery_samples.pop(0)
        rates = [
            (b * 8.0) / (ms / 1000.0) for b, ms in self.delivery_samples
        ]
        self.btl_bw_bps = max(rates)
        self.packets_acked += max(1, int(bytes_delivered) // self.mtu_bytes)
        if self.packets_acked >= self.startup_packets:
            self.in_startup = False

    def record_loss(self, lost_packets: int, sent_packets: int) -> float:
        """Fold a loss report into the loss EWMA and apply a multiplicative decrease."""
        if sent_packets <= 0:
            return self.loss_ratio
        ratio = max(0.0, min(1.0, float(lost_packets) / float(sent_packets)))
        self.loss_ratio = 0.75 * self.loss_ratio + 0.25 * ratio
        if ratio > 0.0:
            self.cwnd_packets = max(
                float(self.min_cwnd_packets), self.cwnd_packets * self.loss_beta
            )
            self.btl_bw_bps *= 1.0 - min(0.5, ratio)
        return self.loss_ratio

    # ------------------------------------------------------------------
    # Window computation
    # ------------------------------------------------------------------
    def compute_target_window(
        self,
        rtt_ms: Optional[float] = None,
        jitter_ms: Optional[float] = None,
        loss_ratio: Optional[float] = None,
    ) -> CongestionWindowTelemetry:
        """Compute the BBR-inspired congestion window for the current path conditions."""
        rtprop = float(rtt_ms) if (rtt_ms is not None and rtt_ms > 0.0) else (self.rtprop_ms or 20.0)
        jitter = float(jitter_ms) if jitter_ms is not None else self.jitter_ms
        loss = float(loss_ratio) if loss_ratio is not None else self.loss_ratio
        loss = max(0.0, min(1.0, loss))
        jitter = max(0.0, jitter)

        base_gain = self.startup_gain if self.in_startup else self.steady_gain
        # Jitter inflates the queueing-delay estimate -> shrink the pacing gain.
        jitter_penalty = 1.0 / (1.0 + (jitter / max(1.0, rtprop)))
        # Packet loss triggers the BBR/BBR-style multiplicative decrease.
        loss_penalty = max(self.loss_beta, 1.0 - loss * self.loss_sensitivity)
        gain = base_gain * jitter_penalty * loss_penalty

        btl_bw = self.btl_bw_bps
        if btl_bw <= 0.0:
            # Bootstrap the delivery-rate model with the canonical 10-packets-per-RTT estimate.
            btl_bw = (self.initial_cwnd_packets * self.mtu_bytes * 8.0) / (rtprop / 1000.0)

        cwnd_bytes = gain * (btl_bw / 8.0) * (rtprop / 1000.0)
        cwnd_packets = int(round(cwnd_bytes / self.mtu_bytes))
        cwnd_packets = max(self.min_cwnd_packets, min(self.max_cwnd_packets, cwnd_packets))

        return CongestionWindowTelemetry(
            cwnd_bytes=cwnd_packets * self.mtu_bytes,
            cwnd_packets=cwnd_packets,
            btl_bw_bps=btl_bw,
            rtprop_ms=rtprop,
            jitter_ms=jitter,
            loss_ratio=loss,
            gain=gain,
            mtu_bytes=self.mtu_bytes,
            mode="startup" if self.in_startup else "steady",
        )

    def plan_hidden_state_burst(
        self,
        packet: CompressedHiddenStatePacket,
        rtt_ms: Optional[float] = None,
        jitter_ms: Optional[float] = None,
        loss_ratio: Optional[float] = None,
        total_packets: int = 1,
    ) -> Dict[str, Any]:
        """Size the in-flight burst of `CompressedHiddenStatePacket` frames allowed by the window."""
        window = self.compute_target_window(
            rtt_ms=rtt_ms, jitter_ms=jitter_ms, loss_ratio=loss_ratio
        )
        packet_bytes = max(1, int(packet.compressed_bytes))
        total = max(1, int(total_packets))
        burst_packets = max(1, int(window.cwnd_bytes // packet_bytes))
        burst_packets = min(burst_packets, total)
        return {
            **window.to_dict(),
            "packet_bytes": packet_bytes,
            "total_packets": total,
            "total_bytes": packet_bytes * total,
            "burst_packets": burst_packets,
            "burst_bytes": burst_packets * packet_bytes,
            "rounds_to_flush": max(1, (total + burst_packets - 1) // burst_packets),
            "bbr_model": "bbr_v1_delivery_rate_max_filter",
        }


class AsyncQUICGeoRingTransport(RegionalGeoRingTransport):
    """M1 (`HM-M1-01`) & S3 (`HM-S3-01`): QUIC 0-RTT hidden-state transport with STUN/TURN NAT traversal.

    Frames ride real QUIC streams (`hivemind.swarm.quic_link`) with TLS 1.3 0-RTT resumption and
    fall back to a libp2p-negotiated WebSocket when UDP is unusable. Peers learn reflexive
    addresses with RFC 5389 STUN, punch UDP holes directly when their NATs allow it, and relay
    through an RFC 5766 TURN allocation when they do not (`hivemind.swarm.nat_traversal`).

    Remote hidden-state sends require an explicit trusted `LinkEndpoint`; the NAT
    probe/relay APIs do not automatically configure that link. Methods ending in
    `_loopback` and `benchmark_loopback_0rtt` remain explicit local demonstrations.
    """

    def __init__(
        self,
        region: str = "EU-West-Paris",
        max_rtt_ms: float = 20.0,
        bind_host: str = "127.0.0.1",
        signing_secret: bytes = b"HiveMind-QUIC-0RTT-Secret",
        *,
        remote_endpoint: "LinkEndpoint | None" = None,
    ) -> None:
        super().__init__(region=region, max_rtt_ms=max_rtt_ms)
        self.bind_host = bind_host
        self.signing_secret = signing_secret
        self.remote_endpoint = remote_endpoint
        self._remote_links: Dict["LinkEndpoint", "RemoteHiddenStateLink"] = {}
        self._remote_links_lock = threading.Lock()
        self.zero_rtt_tickets: Dict[str, ZeroRTTSessionTicket] = {}
        self.session_tickets_0rtt: Dict[str, str] = {}
        self.anti_replay_max_nonce: Dict[str, int] = {}
        self._nonce_lock = threading.Lock()
        self.nat_endpoints: Dict[str, Dict[str, Any]] = {}
        self.turn_clients: Dict[str, TurnClient] = {}
        self.total_udp_hole_punches: int = 0
        self.total_turn_fallbacks: int = 0
        self.congestion_controller: AdaptiveCongestionController = AdaptiveCongestionController()

    def issue_0rtt_ticket(self, peer_id: str) -> ZeroRTTSessionTicket:
        """Issue the application-level resumption record that keys `peer_id`'s anti-replay state.

        The TLS 1.3 session ticket that actually enables 0-RTT is issued by the QUIC server in
        `quic_link`; this record carries the swarm-side nonce window and NAT metadata.
        """
        with self._nonce_lock:
            previous = self.zero_rtt_tickets.get(peer_id)
            # A new ticket refreshes its lifetime, never the peer's strike register.
            max_seen = max(
                self.anti_replay_max_nonce.get(peer_id, 0),
                previous.last_accepted_nonce if previous is not None else 0,
            )
            token = secrets.token_bytes(12)
            ticket = ZeroRTTSessionTicket(
                peer_id=peer_id,
                session_token=token,
                issued_at=time.time(),
                last_accepted_nonce=max_seen,
            )
            self.zero_rtt_tickets[peer_id] = ticket
            self.session_tickets_0rtt[peer_id] = token.hex()
            self.anti_replay_max_nonce[peer_id] = max_seen
            return ticket

    def verify_0rtt_ticket_anti_replay(
        self,
        peer_id: str,
        nonce: int,
        timestamp: float | None = None,
        max_age_sec: float = 600.0,
    ) -> Tuple[bool, str]:
        """Check and atomically advance the application-level monotonic strike register.

        ``timestamp`` is the verifier's trusted current time, not a sender-supplied
        packet timestamp. Tickets are valid on the inclusive interval
        ``[issued_at, issued_at + max_age_sec]``; future-issued tickets fail closed.
        This in-process register is not a substitute for QUIC/TLS authentication or
        a shared, durable replay store across servers and process restarts.
        """
        with self._nonce_lock:
            ticket = self.zero_rtt_tickets.get(peer_id)
            if ticket is None:
                return False, "missing_0rtt_session_ticket"
            if isinstance(nonce, bool) or not isinstance(nonce, Integral) or nonce <= 0:
                return False, "invalid_0rtt_nonce"
            now = timestamp if timestamp is not None else time.time()
            if (
                not math.isfinite(now)
                or not math.isfinite(ticket.issued_at)
                or not math.isfinite(max_age_sec)
                or max_age_sec < 0
            ):
                return False, "invalid_0rtt_ticket_time"
            if now < ticket.issued_at:
                return False, "future_0rtt_session_ticket"
            if now - ticket.issued_at > max_age_sec:
                return False, "expired_0rtt_session_ticket"
            # Keep the public ticket field and legacy dictionary synchronized, also
            # when restoring an older record that persisted only one of them.
            max_seen = max(self.anti_replay_max_nonce.get(peer_id, 0), ticket.last_accepted_nonce)
            self.anti_replay_max_nonce[peer_id] = max_seen
            ticket.last_accepted_nonce = max_seen
            if nonce <= max_seen:
                return False, f"replay_nonce_detected(nonce={nonce}<=max_seen={max_seen})"
            self.anti_replay_max_nonce[peer_id] = int(nonce)
            ticket.last_accepted_nonce = int(nonce)
            ticket.resumed_count += 1
            return True, "0rtt_nonce_accepted"

    @staticmethod
    def _punch_pair(
        sock_a: socket.socket,
        sock_b: socket.socket,
        candidates_a: List[Address],
        candidates_b: List[Address],
        timeout: float,
    ) -> Tuple[PunchResult, PunchResult]:
        """Run both sides of a simultaneous-open hole punch at once (B on a worker thread)."""
        session = secrets.token_bytes(8)
        result_b: List[PunchResult] = []
        worker = threading.Thread(
            target=lambda: result_b.append(udp_hole_punch(sock_b, candidates_a, session, timeout))
        )
        worker.start()
        try:
            result_a = udp_hole_punch(sock_a, candidates_b, session, timeout)
        finally:
            worker.join()
        return result_a, result_b[0]

    def _open_peer_socket(
        self, peer_id: str, stun_server: Optional[ServerSpec]
    ) -> Tuple[socket.socket, Address, Dict[str, Any]]:
        """Bind a peer socket and return it with its candidate (STUN-reflexive or host) address."""
        if stun_server is None:
            sock = udp_socket((self.bind_host, 0))
            candidate: Address = sock.getsockname()
            return sock, candidate, {"peer_id": peer_id, "candidate": candidate, "nat_type": "unknown"}
        sock = udp_socket(("0.0.0.0", 0))
        try:
            mapping = classify_nat_mapping(sock, [stun_server])
        except BaseException:
            sock.close()
            raise
        endpoint = {
            "peer_id": peer_id,
            "candidate": mapping.mapped[0],
            "nat_type": mapping.nat_type,
            "mapping_behaviour": mapping.behaviour,
        }
        return sock, mapping.mapped[0], endpoint

    def perform_udp_hole_punch(
        self,
        local_peer_id: str,
        remote_peer_id: str,
        stun_server: Optional[ServerSpec] = None,
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """Punch a UDP path between two local peers with simultaneous-open checks (`HM-S3-01`).

        Candidates are STUN server-reflexive addresses when `stun_server` is given, otherwise the
        sockets' host addresses. Use `execute_udp_hole_punch` for TURN fallback.
        """
        self.issue_0rtt_ticket(local_peer_id)
        self.issue_0rtt_ticket(remote_peer_id)
        with ExitStack() as sockets:
            sock_a, cand_a, _ = self._open_peer_socket(local_peer_id, stun_server)
            sockets.callback(sock_a.close)
            sock_b, cand_b, _ = self._open_peer_socket(remote_peer_id, stun_server)
            sockets.callback(sock_b.close)
            res_a, res_b = self._punch_pair(sock_a, sock_b, [cand_a], [cand_b], timeout)
        verified = res_a.succeeded and res_b.succeeded
        rtts = [r.rtt_ms for r in (res_a, res_b) if r.rtt_ms is not None]
        return {
            "local_peer_id": local_peer_id,
            "remote_peer_id": remote_peer_id,
            "local_candidate": f"{cand_a[0]}:{cand_a[1]}",
            "remote_candidate": f"{cand_b[0]}:{cand_b[1]}",
            "local_ephemeral_port": cand_a[1],
            "remote_ephemeral_port": cand_b[1],
            "hole_punch_succeeded": verified,
            "transport_mode": "direct_udp" if verified else "failed",
            "effective_rtt_ms": round(max(rtts), 3) if rtts else None,
            "probes_sent": res_a.probes_sent + res_b.probes_sent,
            "handshake_verified": verified,
        }

    @staticmethod
    def _ice_candidate(
        foundation: int, ip: str, port: int, kind: str, related: Optional[Address] = None
    ) -> str:
        """RFC 8839 candidate line with an RFC 8445 §5.1.2.1 priority (component 1, UDP)."""
        type_preference = {"host": 126, "srflx": 100, "relay": 0}[kind]
        priority = (type_preference << 24) + (65535 << 8) + (256 - 1)
        line = f"candidate:{foundation} 1 UDP {priority} {ip} {port} typ {kind}"
        if related is not None:
            line += f" raddr {related[0]} rport {related[1]}"
        return line

    def negotiate_stun_ice_candidate(
        self,
        peer_id: str,
        local_port: int = 0,
        stun_server: Optional[ServerSpec] = None,
    ) -> Dict[str, Any]:
        """Gather ICE candidates for `peer_id` and issue its 0-RTT resumption record.

        Always yields the host candidate of a socket bound on `bind_host`; with `stun_server` it
        also performs a Binding request and adds the server-reflexive candidate.
        """
        try:
            sock = udp_socket(("0.0.0.0" if stun_server else self.bind_host, local_port))
        except OSError:
            sock = udp_socket(("0.0.0.0" if stun_server else self.bind_host, 0))
        try:
            host_ip, port = sock.getsockname()
            if host_ip == "0.0.0.0":
                host_ip = outbound_interface_ip(resolve(stun_server)) if stun_server else self.bind_host
            candidates = [self._ice_candidate(1, host_ip, port, "host")]
            reflexive: Optional[Address] = None
            if stun_server is not None:
                reflexive = stun_binding(sock, stun_server).mapped
                candidates.append(
                    self._ice_candidate(2, reflexive[0], reflexive[1], "srflx", related=(host_ip, port))
                )
        finally:
            sock.close()
        ticket_obj = self.issue_0rtt_ticket(peer_id)
        ticket_hex = hashlib.sha256(
            self.signing_secret + f":{peer_id}:{port}".encode("utf-8")
        ).hexdigest()[:24]
        self.session_tickets_0rtt[peer_id] = ticket_hex
        return {
            "peer_id": peer_id,
            "ice_candidates": candidates,
            "ice_candidate": candidates[-1],
            "candidate_type": "srflx" if reflexive else "host",
            "bind_host": self.bind_host,
            "local_port": port,
            "reflexive_address": f"{reflexive[0]}:{reflexive[1]}" if reflexive else None,
            "session_ticket_0rtt": ticket_hex,
            "session_token_bytes_hex": ticket_obj.session_token.hex(),
            "protocol": "quic_0rtt",
            "fallback_protocol": "libp2p_websocket",
        }

    def serialize_packet_0rtt(
        self,
        peer_id: str,
        packet: CompressedHiddenStatePacket,
        use_websocket_fallback: bool = False,
    ) -> bytes:
        """Pack an FP8 `CompressedHiddenStatePacket` into the binary frame carried by QUIC or WebSocket."""
        ticket = self.zero_rtt_tickets.get(peer_id) or self.issue_0rtt_ticket(peer_id)
        magic = MAGIC_WS_FALLBACK if use_websocket_fallback else MAGIC_QUIC_0RTT
        shape = packet.original_shape
        ndim = len(shape)
        fp8_bytes = packet.fp8_data.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        scale_bytes = packet.absmax_scale.contiguous().cpu().numpy().tobytes()
        header = struct.pack(
            f"!4s12sB{ndim}III",
            magic,
            ticket.session_token,
            ndim,
            *shape,
            len(fp8_bytes),
            len(scale_bytes),
        )
        return header + fp8_bytes + scale_bytes

    def deserialize_packet_0rtt(
        self, frame: bytes
    ) -> Tuple[CompressedHiddenStatePacket, str]:
        """Validate and unpack a hidden-state frame; this parser does not authenticate its token."""
        if len(frame) < 17:
            raise ValueError("truncated hidden-state frame header")
        magic, _token, ndim = struct.unpack("!4s12sB", frame[:17])
        if magic not in (MAGIC_QUIC_0RTT, MAGIC_WS_FALLBACK):
            raise ValueError("unknown hidden-state frame magic")
        if not 1 <= ndim <= 64:
            raise ValueError("hidden-state frame must have 1 to 64 dimensions")
        protocol = "libp2p_websocket" if magic == MAGIC_WS_FALLBACK else "quic"
        fmt = f"!{ndim}III"
        header_len = 17 + struct.calcsize(fmt)
        if len(frame) < header_len:
            raise ValueError("truncated hidden-state shape header")
        unpacked = struct.unpack(fmt, frame[17:header_len])
        shape = tuple(unpacked[:ndim])
        fp8_len, scale_len = unpacked[ndim], unpacked[ndim + 1]
        if any(dimension == 0 for dimension in shape):
            raise ValueError("hidden-state frame dimensions must be positive")
        if fp8_len != math.prod(shape) or scale_len != math.prod(shape[:-1]) * 2:
            raise ValueError("hidden-state payload lengths do not match the declared shape")
        if len(frame) != header_len + fp8_len + scale_len:
            raise ValueError("hidden-state frame is truncated or contains trailing data")
        fp8_raw = frame[header_len : header_len + fp8_len]
        scale_raw = frame[header_len + fp8_len : header_len + fp8_len + scale_len]

        fp8_u8 = torch.frombuffer(bytearray(fp8_raw), dtype=torch.uint8).reshape(shape)
        fp8_tensor = fp8_u8.view(torch.float8_e4m3fn)
        scale_shape = (*shape[:-1], 1)
        scale_tensor = torch.frombuffer(bytearray(scale_raw), dtype=torch.float16).reshape(
            scale_shape
        )
        orig_bytes = math.prod(shape) * 2
        pkt = CompressedHiddenStatePacket(
            fp8_data=fp8_tensor,
            absmax_scale=scale_tensor,
            original_shape=shape,
            original_bytes=orig_bytes,
            compressed_bytes=fp8_len + scale_len,
        )
        return pkt, protocol

    async def _send_remote_frames(
        self,
        peer_id: str,
        frames: Sequence[bytes],
        endpoint: "LinkEndpoint | None",
    ) -> LinkRun:
        from hivemind.swarm.quic_link import LinkEndpoint, RemoteHiddenStateLink

        selected = endpoint if endpoint is not None else self.remote_endpoint
        if selected is None:
            raise ValueError("configure remote_endpoint or supply an explicit trusted LinkEndpoint")
        if not isinstance(selected, LinkEndpoint):
            raise TypeError("remote endpoint must be a LinkEndpoint")
        if not frames:
            raise ValueError("at least one hidden-state packet is required")
        with self._remote_links_lock:
            link = self._remote_links.get(selected)
            if link is None:
                link = RemoteHiddenStateLink(selected)
                self._remote_links[selected] = link
        run = await link.send_frames(frames)
        if run.session_resumed:
            with self._nonce_lock:
                ticket = self.zero_rtt_tickets.get(peer_id)
                if ticket is not None:
                    ticket.resumed_count += 1
        return run

    async def send_compressed_hidden_states(
        self,
        peer_id: str,
        packets: Sequence[CompressedHiddenStatePacket],
        *,
        endpoint: "LinkEndpoint | None" = None,
    ) -> LinkRun:
        """Send packets to an explicitly configured, independently running receiver.

        The endpoint supplies the trusted CA and server name. Its client caches
        single-use TLS session tickets between calls on this transport instance;
        no local receiver or warmup frame is created. A receiver acknowledgment
        confirms its delivered-frame count, not model execution or tensor quality.
        STUN discovery and TURN allocation are separate APIs and do not wire a NAT
        path into this link. Application nonce checks also remain separate.
        """
        frames = [self.serialize_packet_0rtt(peer_id, packet) for packet in packets]
        return await self._send_remote_frames(peer_id, frames, endpoint)

    def transmit_compressed_hidden_state(
        self,
        peer_id: str,
        packet: CompressedHiddenStatePacket,
        *,
        endpoint: "LinkEndpoint | None" = None,
    ) -> Dict[str, Any]:
        """Synchronous remote send; use send_compressed_hidden_states in async code.

        Unlike the explicitly named loopback demonstration, this requires a
        configured receiver. Certificate and uncertain-delivery failures propagate.
        """
        selected = endpoint if endpoint is not None else self.remote_endpoint
        if selected is None:
            raise ValueError("configure remote_endpoint or supply an explicit trusted LinkEndpoint")
        frame = self.serialize_packet_0rtt(peer_id, packet)
        run = _run_coroutine(self._send_remote_frames(peer_id, [frame], selected))
        return {
            "transport_used": {
                "quic_0rtt": "quic_0rtt_stream",
                "quic": "quic_stream",
                "libp2p_websocket": "libp2p_websocket_fallback",
            }[run.protocol],
            "endpoint_host": selected.host,
            "endpoint_port": selected.port,
            "tls_server_name": selected.server_name,
            "frame_bytes": len(frame),
            "frames_sent": run.frames_sent,
            "frames_delivered": run.frames_delivered,
            "frame_acknowledged": run.frames_delivered == 1,
            "transport_latency_ms": round(run.elapsed_s * 1000.0, 3),
            "session_resumed": run.session_resumed,
            "early_data_accepted": run.early_data_accepted,
            "fallback_reason": run.fallback_reason,
        }

    def _send_over_link(
        self, frames: List[bytes], expected_shape: Tuple[int, ...], udp_blocked: bool
    ) -> Tuple[LinkRun, int]:
        """Push `frames` through a local QUIC link (WebSocket on fallback); count intact arrivals."""
        intact: List[bool] = []

        def on_frame(frame: bytes) -> None:
            if frame == b"warmup":
                return
            restored, _ = self.deserialize_packet_0rtt(frame)
            intact.append(restored.original_shape == expected_shape)

        run = _run_coroutine(
            run_local_link(frames, host=self.bind_host, on_frame=on_frame, udp_blocked=udp_blocked)
        )
        return run, sum(intact)

    def benchmark_loopback_0rtt(
        self,
        hidden_states: torch.Tensor,
        num_frames: int = 120,
        simulate_udp_blocked: bool = False,
    ) -> Dict[str, Any]:
        """Stream `num_frames` FP8 hidden-state frames over a resumed 0-RTT QUIC connection on `bind_host`.

        `simulate_udp_blocked` aims the QUIC client at a UDP port nothing listens on, so the
        handshake genuinely times out and the libp2p WebSocket fallback carries the frames.
        """
        pkt = compress_hidden_states_fp8(hidden_states)
        peer_id = "peer-paris-01"
        self.issue_0rtt_ticket(peer_id)

        t_ser0 = time.perf_counter()
        wire_frame = self.serialize_packet_0rtt(
            peer_id, pkt, use_websocket_fallback=simulate_udp_blocked
        )
        restored_pkt, _ = self.deserialize_packet_0rtt(wire_frame)
        ser_ms = (time.perf_counter() - t_ser0) * 1000.0
        _ = decompress_hidden_states_fp8(restored_pkt)

        run, intact = self._send_over_link([wire_frame] * num_frames, pkt.original_shape, simulate_udp_blocked)
        if run.session_resumed:
            self.zero_rtt_tickets[peer_id].resumed_count += 1
        return {
            "protocol": run.protocol,
            "frames_transmitted": num_frames,
            "frames_delivered": run.frames_delivered,
            "frames_verified": intact,
            "frames_per_sec": round(run.frames_per_sec, 1),
            "serialization_overhead_ms": round(ser_ms, 3),
            "session_resumed": run.session_resumed,
            "early_data_accepted": run.early_data_accepted,
            "server_early_frames": run.server_early_frames,
            "fallback_reason": run.fallback_reason,
            "zero_rtt_resumptions": self.zero_rtt_tickets[peer_id].resumed_count,
        }

    def transmit_compressed_hidden_state_loopback(
        self,
        packet: CompressedHiddenStatePacket,
        simulate_udp_blocked: bool = False,
    ) -> Dict[str, Any]:
        """Send one FP8 hidden-state packet over a resumed 0-RTT QUIC link (or the WebSocket fallback)."""
        t0 = time.perf_counter()
        wire_frame = self.serialize_packet_0rtt(
            "peer-paris-01", packet, use_websocket_fallback=simulate_udp_blocked
        )
        run, intact = self._send_over_link([wire_frame], packet.original_shape, simulate_udp_blocked)
        if run.session_resumed:
            self.zero_rtt_tickets["peer-paris-01"].resumed_count += 1
        transport_used = {
            "quic_0rtt": "quic_0rtt_stream",
            "quic": "quic_stream",
            "libp2p_websocket": "libp2p_websocket_fallback",
        }[run.protocol]
        return {
            "transport_used": transport_used,
            "frame_bytes": len(wire_frame),
            "loopback_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "frame_verified": intact == 1 and run.frames_delivered == 1,
            "early_data_accepted": run.early_data_accepted,
        }

    def discover_stun_turn_endpoint(
        self,
        peer_id: str,
        local_port: int = 0,
        stun_servers: Union[ServerSpec, Sequence[ServerSpec]] = DEFAULT_STUN_SERVERS,
        turn_server: Optional[ServerSpec] = None,
        turn_username: Optional[str] = None,
        turn_password: Optional[str] = None,
        timeout: float = 2.0,
    ) -> Dict[str, Any]:
        """RFC 5389 STUN discovery (+ RFC 5780 mapping test) and optional RFC 5766 TURN allocation (`HM-S3-01`).

        With two answering STUN servers the mapping behaviour tells cone NATs (hole-punchable)
        from symmetric ones. When `turn_server` and credentials are given, a relay is allocated
        and kept alive in `turn_clients[peer_id]` until `release_turn_allocations()`.
        """
        # A tuple can be one (host, port) address or a sequence of server specs.
        # In particular DEFAULT_STUN_SERVERS is a tuple of two URL strings.
        is_address = (
            isinstance(stun_servers, tuple)
            and len(stun_servers) == 2
            and isinstance(stun_servers[0], str)
            and isinstance(stun_servers[1], Integral)
            and not isinstance(stun_servers[1], bool)
        )
        servers = [stun_servers] if isinstance(stun_servers, str) or is_address else list(stun_servers)
        if not servers:
            raise ValueError("at least one STUN server is required")
        sock = udp_socket(("0.0.0.0", local_port))
        try:
            mapping = classify_nat_mapping(sock, servers, timeout)
        finally:
            sock.close()
        relay: Optional[Address] = None
        if turn_server is not None and turn_username is not None:
            previous = self.turn_clients.pop(peer_id, None)
            if previous is not None:
                previous.close()
            client = TurnClient(turn_server, turn_username, turn_password or "", timeout=timeout)
            try:
                relay = client.allocate()
            except BaseException:
                client.close()
                raise
            self.turn_clients[peer_id] = client

        reflexive = mapping.mapped[0]
        endpoint = {
            "peer_id": peer_id,
            "local_port": mapping.local[1],
            "local_address": f"{mapping.local[0]}:{mapping.local[1]}",
            "reflexive_ip": reflexive[0],
            "reflexive_port": reflexive[1],
            "nat_type": mapping.nat_type,
            "mapping_behaviour": mapping.behaviour,
            "hole_punchable": mapping.hole_punchable,
            "stun_server": f"{mapping.servers[0][0]}:{mapping.servers[0][1]}",
            "stun_servers_answered": len(mapping.servers),
            "stun_rtt_ms": round(mapping.rtt_ms[0], 3),
            "turn_server": turn_server if turn_server is None or isinstance(turn_server, str) else f"{turn_server[0]}:{turn_server[1]}",
            "turn_relay_endpoint": f"{relay[0]}:{relay[1]}" if relay else None,
        }
        self.nat_endpoints[peer_id] = endpoint

        ticket = self.zero_rtt_tickets.get(peer_id) or self.issue_0rtt_ticket(peer_id)
        ticket.reflexive_ip = reflexive[0]
        ticket.reflexive_port = int(reflexive[1])
        ticket.nat_type = mapping.nat_type
        ticket.turn_relay_endpoint = endpoint["turn_relay_endpoint"]
        return endpoint

    def release_turn_allocations(self) -> None:
        """Deallocate every TURN relay opened by `discover_stun_turn_endpoint`."""
        for client in self.turn_clients.values():
            client.close()
        self.turn_clients.clear()

    def verify_and_advance_0rtt_nonce(self, peer_id: str, incoming_nonce: int) -> bool:
        """Boolean wrapper using the same ticket lifetime and strike register as verification.

        A ticket must already have been issued; verification never creates or
        refreshes one on behalf of an unrecognized incoming peer.
        """
        accepted, _ = self.verify_0rtt_ticket_anti_replay(peer_id, incoming_nonce)
        return accepted

    @staticmethod
    def _relay_pair(
        turn_server: ServerSpec, username: str, password: str, timeout: float = 2.0
    ) -> Tuple[bool, float, str]:
        """Allocate a TURN relay for each peer and exchange a probe both ways through them."""
        with ExitStack() as clients:
            client_a = TurnClient(turn_server, username, password, timeout=timeout)
            clients.callback(client_a.close)
            client_b = TurnClient(turn_server, username, password, timeout=timeout)
            clients.callback(client_b.close)
            relay_a, relay_b = client_a.allocate(), client_b.allocate()
            client_a.create_permission(relay_b[0])
            client_b.create_permission(relay_a[0])
            probe = b"HMRELAY" + secrets.token_bytes(8)
            t0 = time.perf_counter()
            client_a.send_to(relay_b, probe)
            from_a, data_at_b = client_b.recv_from(timeout)
            client_b.send_to(relay_a, probe[::-1])
            from_b, data_at_a = client_a.recv_from(timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            verified = (
                data_at_b == probe and data_at_a == probe[::-1] and from_a == relay_a and from_b == relay_b
            )
            return verified, elapsed_ms, f"{relay_b[0]}:{relay_b[1]}"

    def execute_udp_hole_punch(
        self,
        peer_a_id: str,
        peer_b_id: str,
        stun_server: Optional[ServerSpec] = None,
        turn_server: Optional[ServerSpec] = None,
        turn_username: Optional[str] = None,
        turn_password: Optional[str] = None,
        direct_timeout: float = 1.0,
        transport_policy: str = "all",
    ) -> Dict[str, Any]:
        """Connect two peers: direct UDP hole punch first, TURN relay when that fails (`HM-S3-01`).

        `transport_policy="relay"` skips the direct attempt and always uses TURN, mirroring
        WebRTC's `iceTransportPolicy: "relay"` for peers that must not expose their address.
        """
        direct_ok = False
        relay_verified = False
        latency_ms: Optional[float] = None
        active_endpoint: Optional[str] = None
        with ExitStack() as sockets:
            sock_a, cand_a, ep_a = self._open_peer_socket(peer_a_id, stun_server)
            sockets.callback(sock_a.close)
            sock_b, cand_b, ep_b = self._open_peer_socket(peer_b_id, stun_server)
            sockets.callback(sock_b.close)
            if transport_policy != "relay":
                res_a, res_b = self._punch_pair(sock_a, sock_b, [cand_a], [cand_b], direct_timeout)
                direct_ok = res_a.succeeded and res_b.succeeded
                if direct_ok and res_a.peer is not None:
                    latency_ms = max(r.rtt_ms or 0.0 for r in (res_a, res_b))
                    active_endpoint = f"{res_a.peer[0]}:{res_a.peer[1]}"
                    self.total_udp_hole_punches += 1
        if not direct_ok and turn_server is not None and turn_username is not None:
            relay_verified, latency_ms, active_endpoint = self._relay_pair(
                turn_server, turn_username, turn_password or ""
            )
            if relay_verified:
                self.total_turn_fallbacks += 1
        if direct_ok:
            transport_mode = "direct_udp_hole_punch"
        elif relay_verified:
            transport_mode = "turn_relay"
        else:
            transport_mode = "failed"
        return {
            "peer_a_id": peer_a_id,
            "peer_b_id": peer_b_id,
            "nat_type_a": ep_a["nat_type"],
            "nat_type_b": ep_b["nat_type"],
            "direct_hole_punch_succeeded": direct_ok,
            "fallback_to_turn": transport_mode == "turn_relay",
            "relay_verified": relay_verified,
            "transport_mode": transport_mode,
            "active_endpoint": active_endpoint,
            "hole_punch_latency_ms": round(latency_ms, 4) if latency_ms is not None else None,
            "sub_5ms_verified": latency_ms is not None and latency_ms < 5.0,
            "endpoint_a": ep_a,
            "endpoint_b": ep_b,
        }

    # ------------------------------------------------------------------
    # HM-S4-02: 2-Tier Continental Super-Ring & adaptive congestion window
    # ------------------------------------------------------------------
    def register_super_ring_peer(
        self,
        router: ContinentalSuperRingRouter,
        peer_id: str,
        region: str,
        rtt_ms: float,
        jitter_ms: float = 0.0,
        packet_loss_ratio: float = 0.0,
        stun_servers: Optional[Sequence[ServerSpec]] = None,
    ) -> bool:
        """Register a peer into the Tier-1 ring of the 2-Tier Continental Super-Ring.

        With `stun_servers` the peer is marked `stun_verified` only if STUN discovery succeeds,
        and its NAT type comes from the RFC 5780 mapping test. The peer is admitted only when
        its measured RTT satisfies the regional `< 20 ms` Geo-Ring SLA (`HM-S4-02`).
        """
        stun_verified = False
        nat_type = "unknown"
        if stun_servers:
            try:
                endpoint = self.discover_stun_turn_endpoint(peer_id, stun_servers=stun_servers)
                stun_verified = True
                nat_type = str(endpoint["nat_type"])
            except (StunError, OSError):
                pass
        node = SuperRingPeerNode(
            peer_id=peer_id,
            region=region,
            rtt_ms=float(rtt_ms),
            jitter_ms=float(jitter_ms),
            packet_loss_ratio=float(packet_loss_ratio),
            stun_verified=stun_verified,
            nat_type=nat_type,
        )
        return router.register_peer(node)

    def measure_adaptive_window(
        self,
        packet: CompressedHiddenStatePacket,
        rtt_ms: float,
        jitter_ms: float = 0.0,
        loss_ratio: float = 0.0,
        total_packets: int = 1,
    ) -> Dict[str, Any]:
        """Size the FP8 hidden-state burst permitted by the BBR-inspired congestion window."""
        self.congestion_controller.record_rtt_sample(float(rtt_ms))
        return self.congestion_controller.plan_hidden_state_burst(
            packet,
            rtt_ms=float(rtt_ms),
            jitter_ms=float(jitter_ms),
            loss_ratio=float(loss_ratio),
            total_packets=total_packets,
        )

    def measure_hidden_state_burst_window(
        self,
        hidden_states: torch.Tensor,
        rtt_ms: float,
        jitter_ms: float = 0.0,
        loss_ratio: float = 0.0,
        total_packets: int = 1,
    ) -> Dict[str, Any]:
        """Compress `hidden_states` to FP8 and report the adaptive burst window for the burst."""
        packet = compress_hidden_states_fp8(hidden_states)
        return self.measure_adaptive_window(
            packet,
            rtt_ms=rtt_ms,
            jitter_ms=jitter_ms,
            loss_ratio=loss_ratio,
            total_packets=total_packets,
        )

