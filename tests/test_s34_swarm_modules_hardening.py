# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening suite for HM-S34-01..04 swarm modules."""

from __future__ import annotations

import pytest

from hivemind.swarm.s34_01_speculative_swarm_decoding import (
    DraftTokenCandidate,
    SpeculativeSwarmDecoder,
    sample_residual_token_distribution,
)
from hivemind.swarm.s34_02_game_preemption_ef21_flush import GamePreemptionEF21CircuitBreaker
from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import (
    GossetE8LatticeQuantizer,
    _nearest_d8,
    is_valid_e8_lattice_point,
)
from hivemind.swarm.s34_04_self_evolving_roadmap_supervisor import (
    RoadmapPRVerificationEntry,
    SelfEvolvingRoadmapSupervisor,
)


def test_s34_01_speculative_swarm_decoding_validation() -> None:
    with pytest.raises(ValueError, match="token_id"):
        DraftTokenCandidate(-1, "tok", 0.5, 0.5)
    with pytest.raises(ValueError, match="token_id"):
        DraftTokenCandidate(True, "tok", 0.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_text"):
        DraftTokenCandidate(1, 123, 0.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draft_prob"):
        DraftTokenCandidate(1, "tok", 0.0, 0.5)
    with pytest.raises(ValueError, match="target_prob"):
        DraftTokenCandidate(1, "tok", 0.5, 1.5)
    with pytest.raises(ValueError, match="uniform_draw"):
        DraftTokenCandidate(1, "tok", 0.5, 0.5, uniform_draw=float("nan"))

    with pytest.raises(ValueError, match="target_probs"):
        sample_residual_token_distribution({}, {1: 0.5})
    with pytest.raises(ValueError, match="draft_probs"):
        sample_residual_token_distribution({1: 1.0}, "not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uniform_draw"):
        sample_residual_token_distribution({1: 1.0}, {1: 0.5}, uniform_draw=True)  # type: ignore[arg-type]

    for kw in ("draft_step_ms", "target_parallel_verify_ms", "target_single_token_ms"):
        with pytest.raises(ValueError, match="latency"):
            SpeculativeSwarmDecoder(**{kw: True})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="latency"):
            SpeculativeSwarmDecoder(**{kw: float("nan")})

    with pytest.raises(ValueError, match="draft_prob"):
        SpeculativeSwarmDecoder.acceptance_probability(True, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draft_prob"):
        SpeculativeSwarmDecoder.acceptance_probability(0.5, float("nan"))

    dec = SpeculativeSwarmDecoder()
    cand = DraftTokenCandidate(1, "def", 0.8, 0.85, 0.2)
    with pytest.raises(ValueError, match="candidates"):
        dec.verify_draft_window([])
    with pytest.raises(ValueError, match="DraftTokenCandidate"):
        dec.verify_draft_window(["not-a-candidate"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="fallback_or_bonus_token_id"):
        dec.verify_draft_window([cand], fallback_or_bonus_token_id=-1)
    with pytest.raises(ValueError, match="fallback_or_bonus_token_id"):
        dec.verify_draft_window([cand], fallback_or_bonus_token_id=True)  # type: ignore[arg-type]


def test_s34_02_game_preemption_ef21_flush_validation() -> None:
    with pytest.raises(ValueError, match="pcie_dma_bandwidth_gbps"):
        GamePreemptionEF21CircuitBreaker(pcie_dma_bandwidth_gbps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k_ratio"):
        GamePreemptionEF21CircuitBreaker(top_k_ratio=float("nan"))

    cb = GamePreemptionEF21CircuitBreaker()
    with pytest.raises(ValueError, match="gradient"):
        cb.compress_ef21([float("nan"), 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="prev_error_buffer"):
        cb.compress_ef21([1.0, 1.0], [True, 0.0])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="trigger_process"):
        cb.trigger_game_preemption(
            trigger_process="",
            allocated_vram_mib=4096,
            inflight_gradient=[0.1, 0.2],
            prev_error_buffer=[0.0, 0.0],
        )
    with pytest.raises(ValueError, match="allocated_vram_mib"):
        cb.trigger_game_preemption(
            trigger_process="game",
            allocated_vram_mib=True,  # type: ignore[arg-type]
            inflight_gradient=[0.1, 0.2],
            prev_error_buffer=[0.0, 0.0],
        )
    with pytest.raises(ValueError, match="kernel_drain_ms"):
        cb.trigger_game_preemption(
            trigger_process="game",
            allocated_vram_mib=4096,
            inflight_gradient=[0.1, 0.2],
            prev_error_buffer=[0.0, 0.0],
            kernel_drain_ms=-1.0,
        )
    with pytest.raises(ValueError, match="cuda_device_available"):
        cb.trigger_game_preemption(
            trigger_process="game",
            allocated_vram_mib=4096,
            inflight_gradient=[0.1, 0.2],
            prev_error_buffer=[0.0, 0.0],
            cuda_device_available="yes",  # type: ignore[arg-type]
        )


def test_s34_03_gosset_e8_lattice_quantizer_validation() -> None:
    with pytest.raises(ValueError, match="codebook_radius"):
        GossetE8LatticeQuantizer(codebook_radius=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="codebook_radius"):
        GossetE8LatticeQuantizer(codebook_radius=float("nan"))

    with pytest.raises(ValueError, match="tol"):
        is_valid_e8_lattice_point([0.0] * 8, tol=0.0)
    assert is_valid_e8_lattice_point([float("nan")] * 8) is False
    assert is_valid_e8_lattice_point([True] * 8) is False  # type: ignore[list-item]

    with pytest.raises(ValueError, match="D8 projection"):
        _nearest_d8([float("nan")] + [0.0] * 7)

    q = GossetE8LatticeQuantizer()
    with pytest.raises(ValueError, match="E8 projection"):
        q.project_to_e8([float("inf")] + [0.0] * 7)
    with pytest.raises(ValueError, match="vec"):
        q.quantize_8d([True] + [0.0] * 7)  # type: ignore[list-item]


def test_s34_04_self_evolving_roadmap_supervisor_validation() -> None:
    with pytest.raises(ValueError, match="order"):
        RoadmapPRVerificationEntry(0, "HM-S32-01", "title", True, "metric", 1.0, "req", "a" * 64)
    with pytest.raises(ValueError, match="ticket_id"):
        RoadmapPRVerificationEntry(1, "", "title", True, "metric", 1.0, "req", "a" * 64)
    with pytest.raises(ValueError, match="invariant_passed"):
        RoadmapPRVerificationEntry(1, "HM-S32-01", "title", "yes", "metric", 1.0, "req", "a" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key_metric_value"):
        RoadmapPRVerificationEntry(1, "HM-S32-01", "title", True, "metric", float("nan"), "req", "a" * 64)
    with pytest.raises(ValueError, match="entry_hash_sha256"):
        RoadmapPRVerificationEntry(1, "HM-S32-01", "title", True, "metric", 1.0, "req", "short")

    sup = SelfEvolvingRoadmapSupervisor()
    with pytest.raises(ValueError, match="openrouter_request_ids"):
        sup.verify_all_10_roadmap_modules(openrouter_request_ids="not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="openrouter_request_ids"):
        sup.verify_all_10_roadmap_modules(openrouter_request_ids=[123])  # type: ignore[list-item]
