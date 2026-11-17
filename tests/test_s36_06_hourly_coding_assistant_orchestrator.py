"""Unit tests for HM-S36-06 Hourly HiveMind Coding Assistant Orchestrator & IP-21/IP-67 fixes."""

from __future__ import annotations

from hivemind.swarm.s34_01_speculative_swarm_decoding import (
    DraftTokenCandidate,
    SpeculativeSwarmDecoder,
    sample_residual_token_distribution,
)
from hivemind.swarm.s36_06_hourly_coding_assistant_orchestrator import (
    HourlyHiveMindCodingAssistantOrchestrator,
)


def test_ip21_residual_distribution_sampling_matches_target_minus_draft() -> None:
    # Target favors token 42 (0.80 vs draft 0.10); token 7 has target < draft (0.10 vs 0.80 -> 0 residual)
    target = {7: 0.10, 42: 0.80, 99: 0.10}
    draft = {7: 0.80, 42: 0.10, 99: 0.10}
    for u in (0.05, 0.25, 0.50, 0.75, 0.95):
        tok = sample_residual_token_distribution(target, draft, uniform_draw=u)
        assert tok == 42

    decoder = SpeculativeSwarmDecoder()
    step = decoder.verify_draft_window(
        [DraftTokenCandidate(token_id=7, token_text="bad", draft_prob=0.80, target_prob=0.10, uniform_draw=0.90)],
        target_vocab_probs=target,
        draft_vocab_probs=draft,
        residual_uniform_draw=0.5,
    )
    assert step.accepted_count == 0
    assert step.bonus_token_id == 42
    assert step.speedup_measurement_kind == "model"


def test_hourly_coding_assistant_orchestrator_verifies_all_s36_and_ip_tasks() -> None:
    orchestrator = HourlyHiveMindCodingAssistantOrchestrator()
    report = orchestrator.run_hourly_improvement_cycle(call_live_openrouter=False)
    assert report.all_gates_passed is True
    assert report.tasks_verified == 6
    assert report.tasks_total == 6
    assert report.ip21_residual_sampling_verified is True
    assert report.ip67_no_hardcoded_xdev_verified is True
    assert len(report.merkle_root_sha256) == 64
