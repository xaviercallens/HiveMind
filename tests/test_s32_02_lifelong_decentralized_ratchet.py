# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S32-02 Lifelong Decentralized Ratchet & Anti-Forgetting Anchor Buffer."""

from __future__ import annotations

from hivemind.swarm.s32_02_lifelong_decentralized_ratchet import (
    AnchorEvalSample,
    DailyCorpusIngestionBatch,
    LifelongDecentralizedRatchet,
)


def _anchor_samples(candidate_shift: float = 0.001) -> list[AnchorEvalSample]:
    return [
        AnchorEvalSample("a1", "astrophysics", 1.20, 1.20 + candidate_shift, weight=1.0),
        AnchorEvalSample("a2", "python_coding", 0.95, 0.95 + candidate_shift, weight=1.5),
        AnchorEvalSample("a3", "lean4_proofs", 1.10, 1.10 + candidate_shift, weight=1.0),
    ]


def test_lifelong_ratchet_promotes_when_anchor_drift_within_half_percent_and_domain_improves() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    batch = DailyCorpusIngestionBatch(
        batch_id="arxiv-jwst-2026-10-31",
        source="arXiv+JWST+TESS",
        document_count=128,
        baseline_domain_nll=1.45,
        candidate_domain_nll=1.31,
    )
    verdict = ratchet.evaluate_checkpoint("ckpt-s32-02-good", _anchor_samples(0.002), batch)
    assert verdict.promoted is True
    assert verdict.relative_anchor_perplexity_drift <= 0.005
    assert verdict.domain_nll_delta < 0.0
    assert len(verdict.ratchet_certificate_sha256) == 64


def test_lifelong_ratchet_rejects_catastrophic_forgetting_above_half_percent() -> None:
    ratchet = LifelongDecentralizedRatchet(max_anchor_drift_ratio=0.005, ewc_lambda=0.1)
    batch = DailyCorpusIngestionBatch(
        batch_id="arxiv-jwst-2026-10-31",
        source="arXiv+JWST+TESS",
        document_count=128,
        baseline_domain_nll=1.45,
        candidate_domain_nll=1.20,
    )
    verdict = ratchet.evaluate_checkpoint("ckpt-s32-02-forgetful", _anchor_samples(0.02), batch)
    assert verdict.promoted is False
    assert "REJECTED_ANCHOR_DRIFT" in verdict.reason
