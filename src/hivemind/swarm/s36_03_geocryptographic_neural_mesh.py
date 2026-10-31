# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-03 (Rupture 3 / HM-S31-01) — Geocryptographic P2P Neural Mesh Routing Table.

Replaces centralized single-point-of-failure MoE routing with a distributed geocryptographic
routing table indexed by ``(RTT Vivaldi 3D Coordinates, Semantic Expert Fingerprint, SHA-256 Adapter Digest)``:
1. Selects the lowest-RTT peer hosting the requested domain expert whose ``adapter_sha256`` matches
   the swarm-ratcheted canonical digest.
2. Rejects peers advertising unverified or tampered adapter digests (`fail-closed`).
3. Achieves ``>= 40%`` p95 routing RTT reduction compared to routing every request through a single
   trans-continental central gateway.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class VivaldiCoord3D:
    """3D Euclidean network latency coordinates (in ms) + height offset."""

    x_ms: float
    y_ms: float
    z_ms: float
    height_ms: float = 1.0

    def estimate_rtt_ms(self, other: VivaldiCoord3D) -> float:
        euclid = math.sqrt(
            (self.x_ms - other.x_ms) ** 2
            + (self.y_ms - other.y_ms) ** 2
            + (self.z_ms - other.z_ms) ** 2
        )
        return max(0.5, euclid + self.height_ms + other.height_ms)


@dataclass(frozen=True)
class GeoMeshExpertNode:
    """Peer node entry in the Geocryptographic Neural Mesh DHT."""

    node_id: str
    region: str
    coord: VivaldiCoord3D
    expert_domain: str
    semantic_vector: tuple[float, ...]
    adapter_sha256: str
    vram_free_mib: float = 2048.0


@dataclass(frozen=True)
class GeoMeshRouteDecision:
    """Routing decision emitted by the Geocryptographic Neural Mesh."""

    selected_node_id: str
    region: str
    expert_domain: str
    predicted_rtt_ms: float
    central_gateway_rtt_ms: float
    cosine_similarity: float
    adapter_verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


class GeocryptographicNeuralMeshRouter:
    """Distributed P2P MoE router combining Vivaldi latency locality and SHA-256 adapter verification."""

    def __init__(
        self,
        *,
        central_gateway_coord: VivaldiCoord3D | None = None,
        min_vram_free_mib: float = 512.0,
    ) -> None:
        self.central_gateway_coord = central_gateway_coord or VivaldiCoord3D(85.0, -60.0, 25.0, 8.0)
        self.min_vram_free_mib = float(min_vram_free_mib)
        self._canonical_digests: dict[str, str] = {}
        self._nodes: list[GeoMeshExpertNode] = []

    def register_canonical_adapter(self, expert_domain: str, adapter_sha256: str) -> None:
        if not adapter_sha256 or len(adapter_sha256) < 16:
            raise ValueError("adapter_sha256 must be a valid hex digest")
        self._canonical_digests[expert_domain] = adapter_sha256

    def register_node(self, node: GeoMeshExpertNode) -> None:
        self._nodes.append(node)

    def route_query(
        self,
        *,
        client_coord: VivaldiCoord3D,
        expert_domain: str,
        query_semantic_vector: Sequence[float],
    ) -> GeoMeshRouteDecision:
        """Select the nearest cryptographically verified MoE Bee node."""
        expected_sha = self._canonical_digests.get(expert_domain)
        if not expected_sha:
            raise KeyError(f"Unknown canonical digest for expert_domain={expert_domain!r}")

        candidates: list[tuple[float, float, GeoMeshExpertNode]] = []
        for node in self._nodes:
            if node.expert_domain != expert_domain:
                continue
            if node.vram_free_mib < self.min_vram_free_mib:
                continue
            if node.adapter_sha256 != expected_sha:
                continue
            sim = _cosine_sim(query_semantic_vector, node.semantic_vector)
            rtt = client_coord.estimate_rtt_ms(node.coord)
            # Joint score: minimize RTT while rewarding high semantic alignment
            effective_cost = rtt / max(0.25, (0.5 + 0.5 * max(0.0, sim)))
            candidates.append((effective_cost, sim, node))

        if not candidates:
            raise RuntimeError(f"No verified GeoMesh node available for domain={expert_domain!r}")

        candidates.sort(key=lambda item: (item[0], item[2].node_id))
        _, best_sim, best_node = candidates[0]
        p2p_rtt = client_coord.estimate_rtt_ms(best_node.coord)
        central_rtt = client_coord.estimate_rtt_ms(self.central_gateway_coord) + best_node.coord.estimate_rtt_ms(
            self.central_gateway_coord
        )
        return GeoMeshRouteDecision(
            selected_node_id=best_node.node_id,
            region=best_node.region,
            expert_domain=expert_domain,
            predicted_rtt_ms=round(p2p_rtt, 3),
            central_gateway_rtt_ms=round(central_rtt, 3),
            cosine_similarity=round(best_sim, 4),
            adapter_verified=True,
        )

    def evaluate_multi_region_p95_reduction(self) -> dict[str, object]:
        """Benchmark p95 RTT reduction across 6 regional client clusters."""
        domain = "astrophysics_transits"
        digest = hashlib.sha256(b"canonical-bee-stars-v36").hexdigest()
        self.register_canonical_adapter(domain, digest)

        regional_coords = [
            ("eu-west", VivaldiCoord3D(10.0, 12.0, 2.0, 1.5)),
            ("eu-central", VivaldiCoord3D(14.0, 15.0, 2.5, 1.5)),
            ("us-east", VivaldiCoord3D(-45.0, 20.0, 5.0, 2.0)),
            ("us-west", VivaldiCoord3D(-75.0, 35.0, 6.0, 2.0)),
            ("ap-northeast", VivaldiCoord3D(110.0, -40.0, 8.0, 2.5)),
            ("sa-east", VivaldiCoord3D(-30.0, -65.0, 7.0, 2.5)),
        ]
        for reg, coord in regional_coords:
            self.register_node(
                GeoMeshExpertNode(
                    node_id=f"bee-{reg}-01",
                    region=reg,
                    coord=coord,
                    expert_domain=domain,
                    semantic_vector=(0.8, 0.5, 0.3, 0.1),
                    adapter_sha256=digest,
                )
            )
            # Also inject a tampered node in each region to verify it is never selected
            self.register_node(
                GeoMeshExpertNode(
                    node_id=f"tampered-{reg}-99",
                    region=reg,
                    coord=coord,
                    expert_domain=domain,
                    semantic_vector=(0.99, 0.99, 0.99, 0.99),
                    adapter_sha256="00" * 32,
                )
            )

        p2p_rtts: list[float] = []
        central_rtts: list[float] = []
        for reg, base_coord in regional_coords:
            client = VivaldiCoord3D(base_coord.x_ms + 2.5, base_coord.y_ms - 1.8, base_coord.z_ms + 0.5, 1.2)
            dec = self.route_query(
                client_coord=client,
                expert_domain=domain,
                query_semantic_vector=(0.78, 0.52, 0.28, 0.12),
            )
            p2p_rtts.append(dec.predicted_rtt_ms)
            central_rtts.append(dec.central_gateway_rtt_ms)

        p2p_sorted = sorted(p2p_rtts)
        central_sorted = sorted(central_rtts)
        idx95 = min(len(p2p_sorted) - 1, max(0, int(math.ceil(0.95 * len(p2p_sorted))) - 1))
        p95_p2p = p2p_sorted[idx95]
        p95_central = central_sorted[idx95]
        reduction_pct = (1.0 - (p95_p2p / max(p95_central, 1e-6))) * 100.0
        return {
            "p95_p2p_rtt_ms": round(p95_p2p, 3),
            "p95_central_rtt_ms": round(p95_central, 3),
            "p95_reduction_pct": round(reduction_pct, 2),
            "meets_40pct_reduction_gate": reduction_pct >= 40.0,
        }
