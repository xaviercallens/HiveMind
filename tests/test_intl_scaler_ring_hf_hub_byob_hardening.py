# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path hardening tests for Iteration 155.

Covers:
1. `hivemind.swarm.international_launch_scaler`:
   - `AdaptiveMultiSplitDecision.__post_init__` cut index, string, boolean, and latency/bandwidth bounds,
   - `HierarchicalFanoutPlan.__post_init__` node count, reduction factor, boolean, and regional breakdown guards,
   - `CoalescedLedgerBatch.__post_init__` event/row count, IOPS reduction percentage, Merkle root digest, and entries guards.
2. `hivemind.swarm.inference_ring`:
   - `SpeculativeSwarmResult.__post_init__` telemetry invariant guards,
   - `RegionalDHTSwarmRing.register_peer` and `register_super_ring_peer` `SwarmPeerNode` type guards.
3. `hivemind.swarm.hf_community_hub`:
   - `SharedCommunityExpertEntry.__post_init__`, `CommunityAdapterBundle.__post_init__`, and `SwarmMiningGuild.__post_init__` invariant guards.
4. `hivemind.swarm.exemplar_byob_bees`:
   - `ExemplarBYOBBee.__post_init__` control-character rejection,
   - `ExemplarBYOBRegistryManager.__init__` `expert_registry` type guard and `get_bee` control-character rejection.
"""

from __future__ import annotations

import pytest

from hivemind.swarm.exemplar_byob_bees import (
    EXEMPLAR_BYOB_BEES,
    ExemplarBYOBBee,
    ExemplarBYOBRegistryManager,
)
from hivemind.swarm.hf_community_hub import (
    DEFAULT_SWARM_GUILDS,
    CommunityAdapterBundle,
    SharedCommunityExpertEntry,
    SwarmMiningGuild,
)
from hivemind.swarm.inference_ring import (
    RegionalDHTSwarmRing,
    SpeculativeSwarmResult,
)
from hivemind.swarm.international_launch_scaler import (
    AdaptiveMultiSplitDecision,
    CoalescedLedgerBatch,
    HierarchicalFanoutPlan,
    coalesce_ledger_settlement_batch,
    compute_hierarchical_fanout_plan,
    select_adaptive_multisplit_cut,
)


def test_international_launch_scaler_dataclasses_hardening() -> None:
    dec = select_adaptive_multisplit_cut(uplink_mbps=50.0, rtt_ms=25.0)
    assert dec.selected_cut_index >= 1

    with pytest.raises(ValueError, match="selected_cut_index"):
        AdaptiveMultiSplitDecision(**{**dec.__dict__, "selected_cut_index": 0})
    with pytest.raises(ValueError, match="selected_cut_name"):
        AdaptiveMultiSplitDecision(**{**dec.__dict__, "selected_cut_name": " "})
    with pytest.raises(ValueError, match="within_1500mb_background_ceiling"):
        AdaptiveMultiSplitDecision(**{**dec.__dict__, "within_1500mb_background_ceiling": 1})
    with pytest.raises(ValueError, match="uplink_mbps"):
        AdaptiveMultiSplitDecision(**{**dec.__dict__, "uplink_mbps": 0.0})
    with pytest.raises(ValueError, match="step_latency_reduction_pct"):
        AdaptiveMultiSplitDecision(**{**dec.__dict__, "step_latency_reduction_pct": float("nan")})

    plan = compute_hierarchical_fanout_plan(total_leaf_nodes=120)
    assert plan.regions_active == 6

    with pytest.raises(ValueError, match="total_leaf_nodes"):
        HierarchicalFanoutPlan(**{**plan.__dict__, "total_leaf_nodes": 0})
    with pytest.raises(ValueError, match="ingress_reduction_factor"):
        HierarchicalFanoutPlan(**{**plan.__dict__, "ingress_reduction_factor": 0.0})
    with pytest.raises(ValueError, match="within_openrouter_concurrency_cap"):
        HierarchicalFanoutPlan(**{**plan.__dict__, "within_openrouter_concurrency_cap": "yes"})
    with pytest.raises(ValueError, match="regional_breakdown"):
        HierarchicalFanoutPlan(**{**plan.__dict__, "regional_breakdown": []})

    batch = coalesce_ledger_settlement_batch([{"contributor_handle": "@bee", "hmt_points": 50}])
    assert batch.hmt_conservation_exact is True

    with pytest.raises(ValueError, match="raw_event_count"):
        CoalescedLedgerBatch(**{**batch.__dict__, "raw_event_count": -1})
    with pytest.raises(ValueError, match="write_iops_reduction_pct"):
        CoalescedLedgerBatch(**{**batch.__dict__, "write_iops_reduction_pct": 120.0})
    with pytest.raises(ValueError, match="hmt_conservation_exact"):
        CoalescedLedgerBatch(**{**batch.__dict__, "hmt_conservation_exact": 1})
    with pytest.raises(ValueError, match="batch_merkle_root_sha256"):
        CoalescedLedgerBatch(**{**batch.__dict__, "batch_merkle_root_sha256": "bad"})
    with pytest.raises(ValueError, match="coalesced_entries"):
        CoalescedLedgerBatch(**{**batch.__dict__, "coalesced_entries": ["not-a-dict"]})


def test_inference_ring_hardening() -> None:
    res = SpeculativeSwarmResult(
        region_ring="EU-West-Paris",
        active_ring_peers=4,
        mean_ring_ping_ms=11.5,
        prompt_length=8,
        generated_tokens=[10, 11, 12],
        draft_gamma=5,
        swarm_verification_passes=1,
        mean_acceptance_rate=0.8,
        effective_tokens_per_sec=22.4,
        latency_reduction_factor=3.0,
    )
    assert res.region_ring == "EU-West-Paris"

    with pytest.raises(ValueError, match="region_ring"):
        SpeculativeSwarmResult(**{**res.__dict__, "region_ring": ""})
    with pytest.raises(ValueError, match="prompt_length"):
        SpeculativeSwarmResult(**{**res.__dict__, "prompt_length": 0})
    with pytest.raises(ValueError, match="draft_gamma"):
        SpeculativeSwarmResult(**{**res.__dict__, "draft_gamma": 20})
    with pytest.raises(ValueError, match="generated_tokens"):
        SpeculativeSwarmResult(**{**res.__dict__, "generated_tokens": [True]})
    with pytest.raises(ValueError, match="mean_acceptance_rate"):
        SpeculativeSwarmResult(**{**res.__dict__, "mean_acceptance_rate": 1.2})
    with pytest.raises(ValueError, match="expert_prefetch_enabled"):
        SpeculativeSwarmResult(**{**res.__dict__, "expert_prefetch_enabled": 1})
    with pytest.raises(ValueError, match="expert_prefetch_hit_rate"):
        SpeculativeSwarmResult(**{**res.__dict__, "expert_prefetch_hit_rate": 1.5})

    ring = RegionalDHTSwarmRing(region="EU-West-Paris")
    with pytest.raises(TypeError, match="peer must be a SwarmPeerNode"):
        ring.register_peer("not-a-peer")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="peer must be a SwarmPeerNode"):
        ring.register_super_ring_peer("not-a-peer")  # type: ignore[arg-type]


def test_hf_community_hub_dataclasses_hardening() -> None:
    entry = SharedCommunityExpertEntry(
        domain_name="Python_Systems_Code",
        dataset_id="hivemind/python_systems",
        author_gamer_tag="CyberValkyrie_FR",
        author_contributor_handle="@xaviercallens",
        expert_cid="cid://sha256/" + "a" * 64,
        source_rank_r=16,
        canonical_rank_r=16,
        lora_alpha=32,
        spectral_energy_retained=0.99,
        baseline_loss=4.5,
        trained_loss=1.2,
        relative_loss_reduction_pct=73.33,
        tensor_sha256="b" * 64,
    )
    assert entry.canonical_rank_r == 16

    with pytest.raises(ValueError, match="domain_name"):
        SharedCommunityExpertEntry(**{**entry.__dict__, "domain_name": ""})
    with pytest.raises(ValueError, match="source_rank_r"):
        SharedCommunityExpertEntry(**{**entry.__dict__, "source_rank_r": 0})
    with pytest.raises(ValueError, match="spectral_energy_retained"):
        SharedCommunityExpertEntry(**{**entry.__dict__, "spectral_energy_retained": 1.5})
    with pytest.raises(ValueError, match="baseline_loss"):
        SharedCommunityExpertEntry(**{**entry.__dict__, "baseline_loss": float("nan")})

    bundle = CommunityAdapterBundle(
        bundle_id="BUNDLE-01",
        bundle_title="Bundle 01",
        hf_repo_id="hivemind-swarm/bundle-01",
        base_model_id="Qwen/Qwen2.5-0.5B",
        canonical_rank_r=16,
        created_at_epoch=1700000000.0,
        guild_id="GUILD-01",
        authors=["CyberValkyrie_FR"],
        contributors=["@xaviercallens"],
        entries=[entry],
        bundle_sha256="c" * 64,
    )
    assert bundle.canonical_rank_r == 16

    with pytest.raises(ValueError, match="bundle_id"):
        CommunityAdapterBundle(**{**bundle.__dict__, "bundle_id": " "})
    with pytest.raises(ValueError, match="canonical_rank_r"):
        CommunityAdapterBundle(**{**bundle.__dict__, "canonical_rank_r": 0})
    with pytest.raises(ValueError, match="authors"):
        CommunityAdapterBundle(**{**bundle.__dict__, "authors": []})
    with pytest.raises(ValueError, match="entries"):
        CommunityAdapterBundle(**{**bundle.__dict__, "entries": ["bad"]})
    with pytest.raises(ValueError, match="ties_merged_included"):
        CommunityAdapterBundle(**{**bundle.__dict__, "ties_merged_included": 1})

    guild = DEFAULT_SWARM_GUILDS[0]
    assert guild.synergy_multiplier >= 1.0
    with pytest.raises(ValueError, match="guild_id"):
        SwarmMiningGuild(**{**guild.__dict__, "guild_id": ""})
    with pytest.raises(ValueError, match="member_miners"):
        SwarmMiningGuild(**{**guild.__dict__, "member_miners": []})
    with pytest.raises(ValueError, match="synergy_multiplier"):
        SwarmMiningGuild(**{**guild.__dict__, "synergy_multiplier": 0.9})
    with pytest.raises(ValueError, match="guild_total_pts"):
        SwarmMiningGuild(**{**guild.__dict__, "guild_total_pts": -1})


def test_exemplar_byob_bees_hardening() -> None:
    bee = EXEMPLAR_BYOB_BEES[0]
    with pytest.raises(ValueError, match="control characters"):
        ExemplarBYOBBee(**{**bee.__dict__, "bee_id": "bee-quantum\n"})

    with pytest.raises(TypeError, match="expert_registry"):
        ExemplarBYOBRegistryManager(expert_registry="bad")  # type: ignore[arg-type]

    mgr = ExemplarBYOBRegistryManager()
    with pytest.raises(ValueError, match="bee_id"):
        mgr.get_bee("bee\x00bad")
