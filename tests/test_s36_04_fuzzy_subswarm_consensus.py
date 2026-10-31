"""Unit tests for HM-S36-04 Localized 15-20 Node Fuzzy Sub-Swarm Consensus."""

from __future__ import annotations

import pytest

from hivemind.swarm.s36_04_fuzzy_subswarm_consensus import (
    LocalizedFuzzySubSwarmConsensus,
    SubSwarmVote,
    hypergeometric_collusion_probability,
)


def test_fuzzy_subswarm_consensus_reaches_quorum_under_50ms() -> None:
    engine = LocalizedFuzzySubSwarmConsensus()
    votes = [
        SubSwarmVote(
            node_id=f"node-{i:02d}",
            answer_digest="digest-honest" if i < 14 else "digest-byzantine",
            semantic_confidence=0.94 if i < 14 else 0.50,
            observer_reputation=0.90 if i < 14 else 0.30,
        )
        for i in range(18)
    ]
    report = engine.evaluate_subswarm(votes, total_network_nodes_n=10_000, assumed_byzantine_fraction=0.15)
    assert report.quorum_reached is True
    assert report.winning_digest == "digest-honest"
    assert report.within_50ms_sla is True
    assert report.collusion_probability_bound < 1e-4
    assert report.complexity_hops_log2_n == 14


def test_fuzzy_subswarm_enforces_15_to_20_node_bounds() -> None:
    engine = LocalizedFuzzySubSwarmConsensus()
    too_small = [
        SubSwarmVote(node_id=f"n-{i}", answer_digest="d", semantic_confidence=0.9, observer_reputation=0.9)
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="Sub-swarm size"):
        engine.evaluate_subswarm(too_small)

    assert hypergeometric_collusion_probability(100, 5, 15, 10) == 0.0
