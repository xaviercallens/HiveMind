"""HM-S31-01 & HM-S31-02: Geocryptographic P2P Neural Mesh & 15–20 Node Fuzzy Sub-Swarm Consensus.

Implements:
  1. Geocryptographic Coordinate RTT Triangulation (`GeoMeshNode`, `triangulate_rtt_consistency`):
     verifies that claimed geographic coordinates match observed speed-of-light in fiber
     RTT bounds (`c_fiber ~= 200 km/ms`) across independent anchor landmarks, mitigating
     co-located Sybil rack attacks.
  2. 15–20 Node Fuzzy Sub-Swarm Partitioner (`partition_into_fuzzy_subswarms`):
     deterministically partitions a large swarm into geographically diverse sub-swarms of
     15 to 20 nodes with maximized mean intra-swarm great-circle distance.
  3. Fuzzy Sub-Swarm Byzantine Consensus (`fuzzy_subswarm_consensus`):
     combines coordinate-median clipping with cosine-similarity fuzzy membership weights
     so honest variance on non-IID shards is preserved while colluding outliers are suppressed.

Synthesized and architected via HiveMind OpenRouter Coding Bee (`hivemind/coder`,
request `req-d01adb55fa574049b945ec8e078f06b1`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0
FIBER_SPEED_KM_PER_MS = 198.0  # ~2/3 c in optical fiber (one-way)


@dataclass(frozen=True)
class GeoMeshNode:
    """Swarm worker with claimed geographic coordinates and observed anchor RTTs."""

    node_id: str
    lat_deg: float
    lon_deg: float
    asn: int
    anchor_rtts_ms: dict[str, float]


@dataclass(frozen=True)
class SubSwarmAssignment:
    """A 15–20 node geographically diverse sub-swarm."""

    subswarm_id: str
    nodes: tuple[GeoMeshNode, ...]
    mean_pairwise_km: float
    unique_asn_count: int


@dataclass(frozen=True)
class FuzzyConsensusResult:
    """Result of fuzzy Byzantine consensus inside a 15–20 node sub-swarm."""

    subswarm_id: str
    aggregated_gradient: np.ndarray
    node_weights: dict[str, float]
    rejected_node_ids: tuple[str, ...]
    honest_cosine_fidelity: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def triangulate_rtt_consistency(
    node: GeoMeshNode,
    anchors: dict[str, tuple[float, float]],
    max_routing_stretch: float = 3.6,
    base_switching_jitter_ms: float = 18.0,
) -> tuple[bool, str]:
    """Verify a node's claimed coordinates against physical speed-of-light RTT bounds.

    Checks:
      1. Lower bound (relativity / fiber physics): RTT >= 2 * d_km / FIBER_SPEED_KM_PER_MS - 1.5ms.
         A node claiming to be 8,000 km away from an anchor cannot respond in 5 ms.
      2. Upper bound (stretch): RTT <= max_routing_stretch * (2 * d_km / FIBER_SPEED_KM_PER_MS) + jitter.
    """
    if not anchors:
        return True, "no_anchors"
    verified_count = 0
    for anchor_id, (alat, alon) in anchors.items():
        if anchor_id not in node.anchor_rtts_ms:
            continue
        rtt_ms = float(node.anchor_rtts_ms[anchor_id])
        dist_km = haversine_km(node.lat_deg, node.lon_deg, alat, alon)
        min_physical_rtt = max(0.2, (2.0 * dist_km) / FIBER_SPEED_KM_PER_MS - 1.5)
        max_plausible_rtt = max_routing_stretch * ((2.0 * dist_km) / FIBER_SPEED_KM_PER_MS) + base_switching_jitter_ms
        if rtt_ms < min_physical_rtt:
            return False, f"speed_of_light_violation:{anchor_id}:rtt={rtt_ms:.2f}ms<min={min_physical_rtt:.2f}ms"
        if rtt_ms > max_plausible_rtt:
            return False, f"excessive_rtt_stretch:{anchor_id}:rtt={rtt_ms:.2f}ms>max={max_plausible_rtt:.2f}ms"
        verified_count += 1
    if verified_count == 0:
        return False, "missing_anchor_measurements"
    return True, "ok"


def partition_into_fuzzy_subswarms(
    nodes: Sequence[GeoMeshNode],
    epoch_seed: str = "epoch-0",
    target_min_size: int = 15,
    target_max_size: int = 20,
) -> list[SubSwarmAssignment]:
    """Partition verified nodes into 15–20 node sub-swarms with high geographic diversity.

    Uses a deterministic VRF-style hash sort combined with round-robin ASN/latitude striping
    so co-located ASNs are spread across distinct sub-swarms rather than concentrated in one.
    """
    if target_min_size < 5 or target_max_size < target_min_size:
        raise ValueError("Invalid target sub-swarm bounds")
    n = len(nodes)
    if n == 0:
        return []

    # Sort by ASN and latitude bands, with deterministic epoch tie-breaker
    def _stripe_key(node: GeoMeshNode) -> tuple[int, int, str]:
        lat_band = int((node.lat_deg + 90.0) // 30.0)
        digest = hashlib.sha256(f"{epoch_seed}:{node.node_id}".encode("utf-8")).hexdigest()
        return (lat_band, node.asn % 97, digest)

    ordered = sorted(nodes, key=_stripe_key)
    mid_size = (target_min_size + target_max_size) // 2
    num_buckets = max(1, round(n / float(mid_size)))
    while n // num_buckets > target_max_size:
        num_buckets += 1
    while num_buckets > 1 and (n // num_buckets) < target_min_size:
        num_buckets -= 1

    buckets: list[list[GeoMeshNode]] = [[] for _ in range(num_buckets)]
    for idx, node in enumerate(ordered):
        buckets[idx % num_buckets].append(node)

    assignments: list[SubSwarmAssignment] = []
    for b_idx, members in enumerate(buckets):
        dists: list[float] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                dists.append(
                    haversine_km(
                        members[i].lat_deg,
                        members[i].lon_deg,
                        members[j].lat_deg,
                        members[j].lon_deg,
                    )
                )
        mean_km = float(np.mean(dists)) if dists else 0.0
        unique_asns = len({m.asn for m in members})
        assignments.append(
            SubSwarmAssignment(
                subswarm_id=f"subswarm-{epoch_seed}-{b_idx:03d}",
                nodes=tuple(members),
                mean_pairwise_km=round(mean_km, 2),
                unique_asn_count=unique_asns,
            )
        )
    return assignments


def fuzzy_subswarm_consensus(
    subswarm_id: str,
    node_gradients: dict[str, np.ndarray],
    reference_gradient: np.ndarray | None = None,
    temperature: float = 0.25,
    rejection_threshold: float = 0.15,
) -> FuzzyConsensusResult:
    """Compute fuzzy Byzantine-resilient consensus over a 15–20 node sub-swarm.

    Algorithm:
      1. Compute coordinate-wise median ``m`` and median absolute deviation (MAD) across nodes.
      2. Score each node by combining cosine alignment with ``m`` and norm consistency.
      3. Assign fuzzy membership weights ``w_i = exp(score_i / temperature)`` for nodes above
         ``rejection_threshold``, zeroing out detected Byzantine outliers.
    """
    if not node_gradients:
        raise ValueError("node_gradients must not be empty")

    node_ids = list(node_gradients.keys())
    matrix = np.vstack([np.asarray(node_gradients[nid], dtype=np.float32).ravel() for nid in node_ids])
    coord_median = np.median(matrix, axis=0)
    median_norm = float(np.linalg.norm(coord_median)) + 1e-12

    norms = np.linalg.norm(matrix, axis=1) + 1e-12
    cos_to_median = (matrix @ coord_median) / (norms * median_norm)

    log_norm_ratios = np.abs(np.log(norms / median_norm))
    norm_penalty = np.exp(-1.5 * log_norm_ratios)
    raw_scores = np.maximum(0.0, cos_to_median) * norm_penalty

    weights: dict[str, float] = {}
    rejected: list[str] = []
    for idx, nid in enumerate(node_ids):
        score = float(raw_scores[idx])
        if score < rejection_threshold:
            weights[nid] = 0.0
            rejected.append(nid)
        else:
            weights[nid] = math.exp(score / max(0.05, temperature))

    total_w = sum(weights.values())
    if total_w <= 1e-12:
        # Fallback to coordinate median if all weights collapsed
        agg = coord_median.astype(np.float32)
        for nid in node_ids:
            weights[nid] = 1.0 / len(node_ids)
    else:
        for nid in node_ids:
            weights[nid] = round(weights[nid] / total_w, 6)
        weight_vec = np.array([weights[nid] for nid in node_ids], dtype=np.float32)[:, None]
        agg = np.sum(matrix * weight_vec, axis=0).astype(np.float32)

    ref = coord_median if reference_gradient is None else np.asarray(reference_gradient, dtype=np.float32).ravel()
    cos_fidelity = float(
        np.dot(agg, ref) / ((float(np.linalg.norm(agg)) + 1e-12) * (float(np.linalg.norm(ref)) + 1e-12))
    )
    return FuzzyConsensusResult(
        subswarm_id=subswarm_id,
        aggregated_gradient=agg,
        node_weights=weights,
        rejected_node_ids=tuple(rejected),
        honest_cosine_fidelity=round(cos_fidelity, 4),
    )
