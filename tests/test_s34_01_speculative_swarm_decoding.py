# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S34-01 Speculative Multi-Token Draft-Verify Decoding."""

from __future__ import annotations

from hivemind.swarm.s34_01_speculative_swarm_decoding import (
    DraftTokenCandidate,
    SpeculativeSwarmDecoder,
)


def test_speculative_decoder_accepts_high_agreement_draft_and_achieves_speedup() -> None:
    decoder = SpeculativeSwarmDecoder(
        draft_step_ms=1.0,
        target_parallel_verify_ms=13.0,
        target_single_token_ms=12.0,
    )
    window = [
        DraftTokenCandidate(101, "def", draft_prob=0.80, target_prob=0.85, uniform_draw=0.4),
        DraftTokenCandidate(102, " verlet", draft_prob=0.75, target_prob=0.78, uniform_draw=0.5),
        DraftTokenCandidate(103, "(r", draft_prob=0.70, target_prob=0.68, uniform_draw=0.90),
        DraftTokenCandidate(104, ", v):", draft_prob=0.85, target_prob=0.90, uniform_draw=0.2),
    ]
    res = decoder.verify_draft_window(window, fallback_or_bonus_token_id=105)
    assert res.accepted_count == 4
    assert res.emitted_tokens == (101, 102, 103, 104, 105)
    assert res.effective_speedup_factor > 3.0


def test_speculative_decoder_stops_at_first_rejected_token() -> None:
    decoder = SpeculativeSwarmDecoder()
    window = [
        DraftTokenCandidate(10, "ok", draft_prob=0.80, target_prob=0.82, uniform_draw=0.3),
        DraftTokenCandidate(11, "bad", draft_prob=0.90, target_prob=0.09, uniform_draw=0.5),  # p_acc = 0.10 < 0.5
        DraftTokenCandidate(12, "after", draft_prob=0.95, target_prob=0.95, uniform_draw=0.1),
    ]
    res = decoder.verify_draft_window(window, fallback_or_bonus_token_id=99)
    assert res.accepted_count == 1
    assert res.emitted_tokens == (10, 99)
