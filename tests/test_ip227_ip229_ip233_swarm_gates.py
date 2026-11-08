# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Fail-closed swarm gates: IP-227 (ratchet), IP-229 (geomesh triangulation), IP-233 (QUIC register_peer)."""

from __future__ import annotations

import pytest

from hivemind.swarm.geomesh_fuzzy_consensus import GeoMeshNode, triangulate_rtt_consistency
from hivemind.swarm.quic_transport import ContinentalSuperRingRouter, SuperRingPeerNode
from hivemind.swarm.s32_02_lifelong_decentralized_ratchet import (
    AnchorEvalSample,
    DailyCorpusIngestionBatch,
    LifelongDecentralizedRatchet,
)


def _batch(candidate_domain_nll: float = 1.31) -> DailyCorpusIngestionBatch:
    return DailyCorpusIngestionBatch(
        batch_id="ip227-batch",
        source="test",
        document_count=8,
        baseline_domain_nll=1.45,
        candidate_domain_nll=candidate_domain_nll,
    )


def _anchors(candidate_nll: float) -> list[AnchorEvalSample]:
    return [
        AnchorEvalSample("a1", "astrophysics", 1.20, candidate_nll, weight=1.0),
        AnchorEvalSample("a2", "python_coding", 0.95, 0.95, weight=1.0),
    ]


def test_ip227_nan_candidate_nll_raises_instead_of_promoting() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    with pytest.raises(ValueError, match="non-finite NLL"):
        ratchet.evaluate_checkpoint("ckpt-nan", _anchors(float("nan")), _batch())


def test_ip227_positive_inf_candidate_nll_raises() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    with pytest.raises(ValueError, match="non-finite NLL"):
        ratchet.evaluate_checkpoint("ckpt-inf", _anchors(float("inf")), _batch())


def test_ip227_non_finite_daily_batch_nll_raises() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    with pytest.raises(ValueError, match="must be finite"):
        ratchet.evaluate_checkpoint("ckpt-batch-nan", _anchors(1.20), _batch(float("nan")))


def test_ip227_no_ewc_inputs_gives_no_penalty_and_no_invented_defaults() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    verdict = ratchet.evaluate_checkpoint("ckpt-no-ewc", _anchors(1.20), _batch(1.31))
    assert verdict.ewc_fisher_penalty is None
    assert verdict.domain_nll_delta == pytest.approx(1.31 - 1.45)
    assert verdict.promoted is True


def test_ip227_partial_ewc_inputs_raise() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    with pytest.raises(ValueError, match="supplied together"):
        ratchet.evaluate_checkpoint(
            "ckpt-partial",
            _anchors(1.20),
            _batch(),
            fisher_diag=(1.0, 1.0),
        )


def test_ip229_no_anchors_is_rejected_not_accepted() -> None:
    node = GeoMeshNode("sybil-no-anchor", 48.8566, 2.3522, 3215, {})
    ok, reason = triangulate_rtt_consistency(node, {})
    assert ok is False
    assert reason == "no_anchors"


def test_ip229_nan_rtt_is_rejected() -> None:
    anchors = {"eu-west": (50.1109, 8.6821)}
    node = GeoMeshNode("nan-rtt", 48.8566, 2.3522, 3215, {"eu-west": float("nan")})
    ok, reason = triangulate_rtt_consistency(node, anchors)
    assert ok is False
    assert reason.startswith("non_finite_rtt")


def test_ip233_nan_rtt_peer_is_refused_by_register_peer() -> None:
    router = ContinentalSuperRingRouter()
    nan_peer = SuperRingPeerNode("nan-peer", "EU-West-Paris", float("nan"))
    assert router.register_peer(nan_peer) is False
    assert "nan-peer" not in router.peers


def test_ip233_finite_low_rtt_peer_is_still_admitted() -> None:
    router = ContinentalSuperRingRouter()
    ok_peer = SuperRingPeerNode("ok-peer", "EU-West-Paris", 5.0)
    assert router.register_peer(ok_peer) is True
    assert "ok-peer" in router.peers
