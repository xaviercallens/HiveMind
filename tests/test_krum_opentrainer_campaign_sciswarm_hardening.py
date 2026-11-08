# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant hardening tests for krum_vs_zkopoi_v5_benchmark, opentrainer_orchestrator, campaign_orchestrator, and scientific_swarm_eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.swarm.amcp_kafka_campaign_bus import InMemoryKafkaBridge
from hivemind.swarm.amcp_kafka_transport import EnvelopeVerifier
from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    CampaignPolicy,
    CampaignRequest,
    DreamPolicy,
    max_unverified_rows,
    select_dream_teacher_for_job,
    sign_campaign_request,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.krum_vs_zkopoi_v5_benchmark import (
    ScalingBenchmarkReport,
    ScalingRegimePoint,
    evaluate_defense_regime,
    multi_krum_select,
    run_e21_scaling_benchmark,
)
from hivemind.swarm.opentrainer_orchestrator import (
    LowCostVMProfile,
    OpenTrainerConfig,
    execute_http_learner_round,
    resolve_opentrainer_keyring,
    verify_federation_envelope,
)
from hivemind.swarm.scientific_swarm_eval import (
    JoinedCampaignProposal,
    JoinedSwarmCampaignEngine,
    SwarmKnowledgeTransferEngine,
    _evaluate_lora_on_scientific_holdout,
    ties_merge_lora_adapters,
)


# ---------------------------------------------------------------------------
# 1. krum_vs_zkopoi_v5_benchmark hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_workers": 4, "byzantine_workers": 1, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": True, "byzantine_workers": 1, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 10, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 0, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 32, "mode": "unknown_mode", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": -0.1, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": float("nan"), "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.5, "honest_cosine_fidelity": 0.99, "memory_mb": 1.0},
        {"n_workers": 10, "byzantine_workers": 2, "grad_dim": 32, "mode": "multi_krum_plus_10pct_replay", "wall_clock_ms": 1.0, "verifier_ms": 0.1, "byzantine_rejection_rate": 1.0, "honest_cosine_fidelity": 2.0, "memory_mb": 1.0},
    ],
)
def test_scaling_regime_point_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ScalingRegimePoint(**kwargs)


def test_scaling_benchmark_report_and_multi_krum_validation() -> None:
    pt = ScalingRegimePoint(32, 6, 64, "multi_krum_plus_10pct_replay", 1.0, 0.2, 1.0, 0.99, 0.5)
    with pytest.raises(ValueError, match="experiment_id"):
        ScalingBenchmarkReport("", "HM-S30-02", (pt,), {32: "multi_krum_plus_10pct_replay"}, 512, True)
    with pytest.raises(ValueError, match="points"):
        ScalingBenchmarkReport("E21", "HM-S30-02", (), {32: "multi_krum_plus_10pct_replay"}, 512, True)
    with pytest.raises(ValueError, match="pareto_winner_by_n"):
        ScalingBenchmarkReport("E21", "HM-S30-02", (pt,), {32: "bad_mode"}, 512, True)
    with pytest.raises(ValueError, match="all_invariants_passed"):
        ScalingBenchmarkReport("E21", "HM-S30-02", (pt,), {32: "multi_krum_plus_10pct_replay"}, 512, 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="2D numpy array"):
        multi_krum_select(np.ones((2, 8), dtype=np.float32), num_byzantine=0)
    with pytest.raises(ValueError, match="finite"):
        multi_krum_select(np.full((6, 8), np.nan, dtype=np.float32), num_byzantine=1)
    with pytest.raises(ValueError, match="num_byzantine"):
        multi_krum_select(np.ones((6, 8), dtype=np.float32), num_byzantine=-1)
    with pytest.raises(ValueError, match="m_select"):
        multi_krum_select(np.ones((6, 8), dtype=np.float32), num_byzantine=1, m_select=0)
    with pytest.raises(ValueError, match="m_select cannot exceed"):
        multi_krum_select(np.ones((6, 8), dtype=np.float32), num_byzantine=1, m_select=10)

    with pytest.raises(ValueError, match="byzantine_ratio"):
        evaluate_defense_regime(n_workers=16, byzantine_ratio=0.6)
    with pytest.raises(ValueError, match="grad_dim"):
        evaluate_defense_regime(n_workers=16, grad_dim=0)
    with pytest.raises(ValueError, match="swarm_sizes"):
        run_e21_scaling_benchmark(swarm_sizes=())


# ---------------------------------------------------------------------------
# 2. opentrainer_orchestrator hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vm_name": ""},
        {"vcpus": 0},
        {"vcpus": True},
        {"memory_mib": 128},
        {"accelerators_count": -1},
        {"boot_disk_gb": 0},
        {"requires_t4_gpu": "no"},  # type: ignore[dict-item]
        {"estimated_monthly_cost_usd": -1.0},
        {"estimated_monthly_cost_usd": float("nan")},
    ],
)
def test_lowcost_vm_profile_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LowCostVMProfile(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": ""},
        {"learners": ()},
        {"learners": "l0"},  # type: ignore[dict-item]
        {"learners": ("l0", "l0")},
        {"learners": ("l0", "")},
        {"quorum": 0},
        {"quorum": 5},
        {"rounds": 0},
        {"seed": -1},
        {"inner_steps": 0},
        {"batch_size": 0},
        {"shard_size": 0},
        {"inner_lr": 0.0},
        {"outer_lr": -0.5},
        {"outer_mu": 1.0},
        {"outer_mu": -0.1},
        {"max_delta_l2": 0.0},
        {"round_deadline_s": 0.0},
        {"cpu_sentinel_when_t4_offline": 1},  # type: ignore[dict-item]
    ],
)
def test_opentrainer_config_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        OpenTrainerConfig(**kwargs)


def test_opentrainer_helpers_validate_inputs() -> None:
    with pytest.raises(TypeError, match="HMACKeyring"):
        resolve_opentrainer_keyring("not-a-keyring")  # type: ignore[arg-type]

    kr = resolve_opentrainer_keyring()
    verifier = EnvelopeVerifier(kr)
    with pytest.raises(TypeError, match="EnvelopeVerifier"):
        verify_federation_envelope("bad", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        verify_federation_envelope(verifier, "not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="orchestrator_url"):
        execute_http_learner_round("ftp://bad.url", learner_id="l0", learner_index=0, keyring=kr)
    with pytest.raises(ValueError, match="learner_id"):
        execute_http_learner_round("http://localhost:8000", learner_id="  ", learner_index=0, keyring=kr)
    with pytest.raises(ValueError, match="learner_index"):
        execute_http_learner_round("http://localhost:8000", learner_id="l0", learner_index=-1, keyring=kr)
    with pytest.raises(TypeError, match="HMACKeyring"):
        execute_http_learner_round("http://localhost:8000", learner_id="l0", learner_index=0, keyring="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_free_mib"):
        execute_http_learner_round("http://localhost:8000", learner_id="l0", learner_index=0, keyring=kr, min_free_mib=0.0)


# ---------------------------------------------------------------------------
# 3. campaign_orchestrator hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_members_per_campaign": 0},
        {"max_members_per_campaign": True},
        {"max_active_campaigns_per_participant": 0},
        {"request_max_skew_s": 0.0},
        {"request_max_skew_s": float("nan")},
        {"join_lead_s": -1.0},
    ],
)
def test_campaign_policy_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CampaignPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_new_verified": 0},
        {"min_heldout_eval_items": 0},
        {"max_jobs": 0},
        {"train_cap_h": 0.0},
        {"control_cap_h": -0.1},
        {"eval_cap_h": 0.0},
        {"governor_max_age_s": 0.0},
        {"governor_max_future_s": -1.0},
        {"eligible_availability": frozenset()},
        {"eligible_availability": frozenset({"invalid_slot"})},
        {"start_h": 7, "end_h": 7},
        {"start_h": 24, "end_h": 7},
    ],
)
def test_dream_policy_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DreamPolicy(**kwargs)


def test_campaign_orchestrator_rejects_bool_vram_and_nan_timestamp(tmp_path: Path) -> None:
    secret = b"s21-orch-test-secret-0123456789ab"
    bridge = InMemoryKafkaBridge()
    orch = CampaignOrchestrator(tmp_path / "state.json", secret, bridge.publish)
    now = 1793223000.0

    r_bool_vram = sign_campaign_request(
        CampaignRequest("join", "prod-1", "gpu_producer", "SUBJ-01-PYTHON-RUST-CODER", True, "eu_evening_cet", now, "nonce-bool01"),  # type: ignore[arg-type]
        secret,
    )
    d1 = orch.handle(r_bool_vram, now)
    assert d1.decision == "REJECTED" and "declared_vram_mib" in d1.reason

    r_nan_ts = CampaignRequest("join", "prod-1", "gpu_producer", "SUBJ-01-PYTHON-RUST-CODER", 16384, "eu_evening_cet", float("nan"), "nonce-nan001", "deadbeef")
    d2 = orch.handle(r_nan_ts, now)
    assert d2.decision == "REJECTED" and "allowed skew" in d2.reason

    with pytest.raises(ValueError):
        max_unverified_rows(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_controls"):
        DreamPolicy().job_budget_h(-1)
    with pytest.raises(ValueError, match="domain"):
        select_dream_teacher_for_job("  ", 8192)
    with pytest.raises(ValueError, match="vram_mib"):
        select_dream_teacher_for_job("coding_python", -1)
    with pytest.raises(ValueError, match="vram_mib"):
        select_dream_teacher_for_job("coding_python", True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. scientific_swarm_eval hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"campaign_id": "", "codename": "C", "strategy_mode": "S", "target_domains": ["D"], "min_participating_agents": 2, "expected_speedup_factor": 2.0, "expected_quality_synergy_pct": 10.0, "reward_pool_hive": 50.0, "reward_pool_pts": 100, "scientific_rationale": "R"},
        {"campaign_id": "ID", "codename": "C", "strategy_mode": "S", "target_domains": [], "min_participating_agents": 2, "expected_speedup_factor": 2.0, "expected_quality_synergy_pct": 10.0, "reward_pool_hive": 50.0, "reward_pool_pts": 100, "scientific_rationale": "R"},
        {"campaign_id": "ID", "codename": "C", "strategy_mode": "S", "target_domains": ["D"], "min_participating_agents": 0, "expected_speedup_factor": 2.0, "expected_quality_synergy_pct": 10.0, "reward_pool_hive": 50.0, "reward_pool_pts": 100, "scientific_rationale": "R"},
        {"campaign_id": "ID", "codename": "C", "strategy_mode": "S", "target_domains": ["D"], "min_participating_agents": 2, "expected_speedup_factor": 0.0, "expected_quality_synergy_pct": 10.0, "reward_pool_hive": 50.0, "reward_pool_pts": 100, "scientific_rationale": "R"},
        {"campaign_id": "ID", "codename": "C", "strategy_mode": "S", "target_domains": ["D"], "min_participating_agents": 2, "expected_speedup_factor": 2.0, "expected_quality_synergy_pct": -1.0, "reward_pool_hive": 50.0, "reward_pool_pts": 100, "scientific_rationale": "R"},
    ],
)
def test_joined_campaign_proposal_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        JoinedCampaignProposal(**kwargs)


def test_scientific_swarm_eval_functions_and_engines_validate_inputs(tmp_path: Path) -> None:
    ref = {"layer0.lora_A": torch.zeros(4, 8), "layer0.lora_B": torch.zeros(8, 4)}

    with pytest.raises(ValueError, match="domain_name"):
        _evaluate_lora_on_scientific_holdout("", ref)
    with pytest.raises(ValueError, match="lora_state"):
        _evaluate_lora_on_scientific_holdout("Rust_Systems_Security", {})
    with pytest.raises(ValueError, match="lora_rank_r"):
        _evaluate_lora_on_scientific_holdout("Rust_Systems_Security", ref, lora_rank_r=0)

    with pytest.raises(ValueError, match="base_lora_state"):
        ties_merge_lora_adapters({}, [])
    with pytest.raises(ValueError, match="trim_density"):
        ties_merge_lora_adapters(ref, [], trim_density=0.0)
    with pytest.raises(ValueError, match="scaling_lambda"):
        ties_merge_lora_adapters(ref, [], scaling_lambda=-1.0)

    with pytest.raises(TypeError, match="ContentAddressedSwarmStore"):
        SwarmKnowledgeTransferEngine(store="not-a-store")  # type: ignore[arg-type]

    store = ContentAddressedSwarmStore(tmp_path / "cas")
    with pytest.raises(ValueError, match="peer_agent_ids"):
        SwarmKnowledgeTransferEngine(store=store, peer_agent_ids="not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        SwarmKnowledgeTransferEngine(store=store, peer_agent_ids=["a", "a"])

    transfer = SwarmKnowledgeTransferEngine(store=store)
    with pytest.raises(ValueError, match="inner_steps"):
        transfer.verify_swarm_peer_knowledge_transfer("Rust_Systems_Security", "CyberValkyrie_FR", ref, inner_steps=0)
    with pytest.raises(ValueError, match="topk_ratio"):
        transfer.verify_swarm_peer_knowledge_transfer("Rust_Systems_Security", "CyberValkyrie_FR", ref, topk_ratio=1.5)
    with pytest.raises(ValueError, match="royalty_per_peer_hive"):
        transfer.verify_swarm_peer_knowledge_transfer("Rust_Systems_Security", "CyberValkyrie_FR", ref, royalty_per_peer_hive=-1.0)

    with pytest.raises(TypeError, match="SwarmKnowledgeTransferEngine"):
        JoinedSwarmCampaignEngine("not-an-engine")  # type: ignore[arg-type]

    campaign = JoinedSwarmCampaignEngine(transfer)
    with pytest.raises(ValueError, match="participating_agents"):
        campaign.execute_joined_shard_parallel_campaign("Rust_Systems_Security", ref, participating_agents=())
    with pytest.raises(ValueError, match="total_inner_steps"):
        campaign.execute_joined_shard_parallel_campaign("Rust_Systems_Security", ref, participating_agents=["CyberValkyrie_FR"], total_inner_steps=0)
    with pytest.raises(ValueError, match="inner_steps"):
        campaign.execute_full_scientific_swarm_campaign(ref, inner_steps=0)
