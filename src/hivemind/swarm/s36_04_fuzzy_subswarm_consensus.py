# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-04 (Rupture 3 / HM-S31-02) — Localized Fuzzy Sub-Swarm Consensus (15–20 Neighboring Nodes).

Replaces global O(N) broadcast validation with localized 15–20 node sub-swarm fuzzy agreement
inspired by Honeybee Nesting Scrutiny & Waggle-Dance Observer Fitness:
1. Validates sub-swarm size strictly within ``[15, 20]`` neighboring nodes.
2. Computes fuzzy confidence & semantic cross-consistency in ``< 50 ms`` with ``O(log N)`` scaling.
3. Computes the exact hypergeometric upper bound on Byzantine collusion probability under
   geocryptographic random peer sampling.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

MIN_SUBSWARM_NODES = 15
MAX_SUBSWARM_NODES = 20


@dataclass(frozen=True)
class SubSwarmVote:
    """Single neighbor node evaluation inside a 15–20 node localized sub-swarm."""

    node_id: str
    answer_digest: str
    semantic_confidence: float
    observer_reputation: float


@dataclass(frozen=True)
class FuzzySubSwarmConsensusReport:
    """Result of a 15–20 node localized fuzzy sub-swarm consensus round."""

    subswarm_size: int
    winning_digest: str
    fuzzy_agreement_score: float
    quorum_reached: bool
    consensus_latency_ms: float
    within_50ms_sla: bool
    collusion_probability_bound: float
    complexity_hops_log2_n: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def hypergeometric_collusion_probability(
    total_nodes_n: int,
    byzantine_nodes_b: int,
    subswarm_size_k: int,
    supermajority_m: int,
) -> float:
    """Compute exact probability that >= m out of k sampled nodes are Byzantine without replacement."""
    if total_nodes_n <= 0 or subswarm_size_k <= 0 or subswarm_size_k > total_nodes_n:
        raise ValueError("Invalid population or sample size")
    if byzantine_nodes_b < supermajority_m:
        return 0.0
    total_ways = math.comb(total_nodes_n, subswarm_size_k)
    if total_ways == 0:
        return 0.0
    honest_nodes = total_nodes_n - byzantine_nodes_b
    favorable = 0
    for j in range(supermajority_m, min(subswarm_size_k, byzantine_nodes_b) + 1):
        if subswarm_size_k - j <= honest_nodes:
            favorable += math.comb(byzantine_nodes_b, j) * math.comb(honest_nodes, subswarm_size_k - j)
    return favorable / total_ways


class LocalizedFuzzySubSwarmConsensus:
    """15–20 node localized fuzzy sub-swarm consensus engine."""

    def __init__(self, *, min_fuzzy_quorum: float = 0.67) -> None:
        self.min_fuzzy_quorum = float(min_fuzzy_quorum)

    def evaluate_subswarm(
        self,
        votes: Sequence[SubSwarmVote],
        *,
        total_network_nodes_n: int = 10_000,
        assumed_byzantine_fraction: float = 0.15,
    ) -> FuzzySubSwarmConsensusReport:
        """Compute waggle-dance weighted fuzzy agreement across 15–20 neighboring nodes."""
        k = len(votes)
        if not (MIN_SUBSWARM_NODES <= k <= MAX_SUBSWARM_NODES):
            raise ValueError(f"Sub-swarm size must be in [{MIN_SUBSWARM_NODES}, {MAX_SUBSWARM_NODES}], got {k}")

        t0 = time.perf_counter()
        bucket_weights: dict[str, float] = {}
        total_weight = 0.0
        for v in votes:
            conf = max(0.0, min(1.0, float(v.semantic_confidence)))
            rep = max(0.01, min(1.0, float(v.observer_reputation)))
            # Fuzzy triangular t-norm weight = confidence * reputation
            w = conf * rep
            bucket_weights[v.answer_digest] = bucket_weights.get(v.answer_digest, 0.0) + w
            total_weight += w

        if total_weight <= 1e-12:
            winning_digest = ""
            agreement = 0.0
        else:
            winning_digest, best_w = max(bucket_weights.items(), key=lambda kv: (kv[1], kv[0]))
            agreement = best_w / total_weight

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        byz_count = int(round(total_network_nodes_n * assumed_byzantine_fraction))
        supermaj = int(math.ceil(self.min_fuzzy_quorum * k))
        collusion_prob = hypergeometric_collusion_probability(
            total_nodes_n=total_network_nodes_n,
            byzantine_nodes_b=byz_count,
            subswarm_size_k=k,
            supermajority_m=supermaj,
        )
        hops = max(1, int(math.ceil(math.log2(max(2, total_network_nodes_n)))))
        return FuzzySubSwarmConsensusReport(
            subswarm_size=k,
            winning_digest=winning_digest,
            fuzzy_agreement_score=round(agreement, 6),
            quorum_reached=agreement >= self.min_fuzzy_quorum,
            consensus_latency_ms=round(elapsed_ms, 4),
            within_50ms_sla=elapsed_ms < 50.0,
            collusion_probability_bound=round(collusion_prob, 10),
            complexity_hops_log2_n=hops,
        )
