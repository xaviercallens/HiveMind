# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path hardening tests for Iteration 154.

Covers:
1. `hivemind.swarm.opentrainer_orchestrator`:
   - `PermanentOpenTrainerOrchestrator.__init__` `vm_profile` type guard,
   - `_validate_optional_now` timestamp bounds in `submit_result_envelope`, `run_cpu_sentinel_step_if_needed`, and `tick`,
   - `tick` boolean `run_sentinel_if_t4_absent` guard,
   - `create_opentrainer_router` and `create_opentrainer_app` argument type guards.
2. `hivemind.swarm.layerstorm_backend`:
   - `TwoZoneVRAMCache` budget type guard,
   - `PCIeStepTelemetry.__post_init__` expert tuple and metric finiteness/non-negativity guards,
   - `PCIeWaterlineStreamScheduler` `vram_cache` type guard,
   - `LayerStoRmSwarmBridge.__post_init__` and `serve_request` `token_expert_traces` guards.
3. `hivemind.swarm.la_ruche_vault`:
   - `RucheHoneycombCell.__post_init__` string, coordinate, rank/size, spectral energy, NLL/perplexity, and boolean guards,
   - `RucheCommunityMoERelease.__post_init__` manifest invariant guards.
4. `hivemind.swarm.krum_vs_zkopoi_v5_benchmark`:
   - `ScalingBenchmarkReport.__post_init__` control-character rejection on `experiment_id` and `roadmap_id`,
   - `multi_krum_select` rejection of boolean and non-numeric numpy arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hivemind.swarm.krum_vs_zkopoi_v5_benchmark import (
    ScalingBenchmarkReport,
    ScalingRegimePoint,
    multi_krum_select,
)
from hivemind.swarm.la_ruche_vault import (
    RucheCommunityMoERelease,
    RucheHoneycombCell,
)
from hivemind.swarm.layerstorm_backend import (
    LayerStoRmSwarmBridge,
    PCIeStepTelemetry,
    PCIeWaterlineStreamScheduler,
    TwoZoneVRAMBudget,
    TwoZoneVRAMCache,
)
from hivemind.swarm.opentrainer_orchestrator import (
    OpenTrainerConfig,
    PermanentOpenTrainerOrchestrator,
    create_opentrainer_app,
    create_opentrainer_router,
)


def test_opentrainer_orchestrator_hardening(tmp_path: Path) -> None:
    cfg = OpenTrainerConfig(
        run_id="hardening-ot-154",
        learners=("l0",),
        quorum=1,
        rounds=1,
        inner_steps=2,
        checkpoint_path=tmp_path / "ot.pt",
    )
    with pytest.raises(TypeError, match="vm_profile"):
        PermanentOpenTrainerOrchestrator(cfg, vm_profile="bad")  # type: ignore[arg-type]

    orch = PermanentOpenTrainerOrchestrator(cfg)
    with pytest.raises(ValueError, match="run_sentinel_if_t4_absent"):
        orch.tick(run_sentinel_if_t4_absent=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now"):
        orch.tick(now=0.0)
    with pytest.raises(ValueError, match="now"):
        orch.tick(now=float("nan"))
    with pytest.raises(ValueError, match="now"):
        orch.run_cpu_sentinel_step_if_needed(now=-5.0)
    with pytest.raises(ValueError, match="now"):
        orch.submit_result_envelope({}, now=0.0)

    with pytest.raises(TypeError, match="orchestrator"):
        create_opentrainer_router("not-an-orchestrator")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="orchestrator"):
        create_opentrainer_app(orchestrator="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        create_opentrainer_app(config="bad")  # type: ignore[arg-type]


def test_layerstorm_backend_hardening() -> None:
    with pytest.raises(TypeError, match="budget"):
        TwoZoneVRAMCache("not-a-budget")  # type: ignore[arg-type]

    telem = PCIeStepTelemetry(
        requested_experts=(0, 1),
        streamed_experts=(1,),
        pcie_h2d_mib=128.0,
        raw_transfer_ms=5.2,
        overlapped_stall_ms=3.6,
        total_step_ms=5.4,
    )
    assert telem.pcie_h2d_mib == 128.0

    with pytest.raises(ValueError, match="requested_experts"):
        PCIeStepTelemetry(
            requested_experts=[0, 1],  # type: ignore[arg-type]
            streamed_experts=(1,),
            pcie_h2d_mib=128.0,
            raw_transfer_ms=5.2,
            overlapped_stall_ms=3.6,
            total_step_ms=5.4,
        )
    with pytest.raises(ValueError, match="streamed_experts"):
        PCIeStepTelemetry(
            requested_experts=(0, 1),
            streamed_experts=(True,),  # type: ignore[arg-type]
            pcie_h2d_mib=128.0,
            raw_transfer_ms=5.2,
            overlapped_stall_ms=3.6,
            total_step_ms=5.4,
        )
    with pytest.raises(ValueError, match="pcie_h2d_mib"):
        PCIeStepTelemetry(
            requested_experts=(0, 1),
            streamed_experts=(1,),
            pcie_h2d_mib=-1.0,
            raw_transfer_ms=5.2,
            overlapped_stall_ms=3.6,
            total_step_ms=5.4,
        )

    with pytest.raises(TypeError, match="vram_cache"):
        PCIeWaterlineStreamScheduler("bad")  # type: ignore[arg-type]

    budget = TwoZoneVRAMBudget(
        vram_cap_mib=6656.0,
        dense_attention_mib=2400.0,
        kv_cache_mib=1024.0,
        expert_size_mib=128.0,
        zone_a_slots=8,
        zone_b_slots=12,
    )
    cache = TwoZoneVRAMCache(budget)
    sched = PCIeWaterlineStreamScheduler(cache)

    with pytest.raises(TypeError, match="scheduler"):
        LayerStoRmSwarmBridge(scheduler="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nectar_cache"):
        LayerStoRmSwarmBridge(scheduler=sched, nectar_cache="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_preempted"):
        LayerStoRmSwarmBridge(scheduler=sched, is_preempted=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="last_preemption_latency_ms"):
        LayerStoRmSwarmBridge(scheduler=sched, last_preemption_latency_ms=-0.1)

    bridge = LayerStoRmSwarmBridge(scheduler=sched)
    with pytest.raises(ValueError, match="token_expert_traces"):
        bridge.serve_request(domain="systems_deeptech", query="q1", token_expert_traces="bad")  # type: ignore[arg-type]


def _make_valid_ruche_cell() -> RucheHoneycombCell:
    return RucheHoneycombCell(
        cell_id="ALVEOLE-01-CODE-PYTHON",
        domain_name="code_python",
        display_title="Python & Numeric Kernels",
        hex_q=1,
        hex_r=0,
        accent_hex="#38bdf8",
        contributor_handle="bee-01",
        hardware_device="RTX-2070-8GB",
        dataset_shard_cid="cid-shard-01",
        expert_cas_cid="cid-expert-01",
        safetensors_rel_path="cells/code_python/alveole.safetensors",
        safetensors_sha256="a" * 64,
        safetensors_bytes=4096,
        source_rank_r=4,
        canonical_rank_r=4,
        spectral_energy_retained=0.98,
        train_nll_before=2.4,
        train_nll_after=1.8,
        holdout_nll_before=2.5,
        holdout_nll_after=1.9,
        holdout_perplexity_before=12.18,
        holdout_perplexity_after=6.69,
        perplexity_reduction_pct=45.07,
        wire_codec="e8_2bit",
        deposited_at_epoch=1700000000.0,
        verified=True,
    )


def test_la_ruche_vault_dataclasses_hardening() -> None:
    cell = _make_valid_ruche_cell()
    assert cell.domain_name == "code_python"

    with pytest.raises(ValueError, match="cell_id"):
        RucheHoneycombCell(**{**cell.__dict__, "cell_id": " "})
    with pytest.raises(ValueError, match="hex_q"):
        RucheHoneycombCell(**{**cell.__dict__, "hex_q": True})
    with pytest.raises(ValueError, match="safetensors_bytes"):
        RucheHoneycombCell(**{**cell.__dict__, "safetensors_bytes": 0})
    with pytest.raises(ValueError, match="spectral_energy_retained"):
        RucheHoneycombCell(**{**cell.__dict__, "spectral_energy_retained": 1.5})
    with pytest.raises(ValueError, match="holdout_nll_after"):
        RucheHoneycombCell(**{**cell.__dict__, "holdout_nll_after": 0.0})
    with pytest.raises(ValueError, match="deposited_at_epoch"):
        RucheHoneycombCell(**{**cell.__dict__, "deposited_at_epoch": -1.0})
    with pytest.raises(ValueError, match="verified"):
        RucheHoneycombCell(**{**cell.__dict__, "verified": 1})

    rel = RucheCommunityMoERelease(
        release_id="RUCHE-MOE-V2_6_0-12345678",
        release_version="v2.6.0",
        base_architecture="HiveMind-BTM-UpcycledSparseMoE-8xQLoRA",
        num_domains=1,
        top_k_active=1,
        canonical_rank_r=4,
        hidden_size=32,
        vocab_size=259,
        router_type="SigmaMoE",
        three_tier_enabled=True,
        ortho_inj_enabled=True,
        combined_safetensors_path="community_moe_release/moe.safetensors",
        combined_safetensors_sha256="b" * 64,
        combined_safetensors_bytes=8192,
        model_card_path="community_moe_release/README.md",
        manifest_path="community_moe_release/ruche_manifest.json",
        mean_holdout_perplexity_before=12.18,
        mean_holdout_perplexity_after=6.69,
        mean_perplexity_reduction_pct=45.07,
        contributors=["bee-01"],
        cells=[cell],
        created_at_epoch=1700000000.0,
    )
    assert rel.num_domains == 1

    with pytest.raises(ValueError, match="release_id"):
        RucheCommunityMoERelease(**{**rel.__dict__, "release_id": ""})
    with pytest.raises(ValueError, match="hidden_size"):
        RucheCommunityMoERelease(**{**rel.__dict__, "hidden_size": 2})
    with pytest.raises(ValueError, match="three_tier_enabled"):
        RucheCommunityMoERelease(**{**rel.__dict__, "three_tier_enabled": 1})
    with pytest.raises(ValueError, match="mean_holdout_perplexity_before"):
        RucheCommunityMoERelease(**{**rel.__dict__, "mean_holdout_perplexity_before": 0.0})
    with pytest.raises(ValueError, match="contributors"):
        RucheCommunityMoERelease(**{**rel.__dict__, "contributors": []})
    with pytest.raises(ValueError, match="cells"):
        RucheCommunityMoERelease(**{**rel.__dict__, "cells": ["bad"]})


def test_krum_vs_zkopoi_v5_benchmark_hardening() -> None:
    pt = ScalingRegimePoint(
        n_workers=32,
        byzantine_workers=6,
        grad_dim=64,
        mode="multi_krum_plus_10pct_replay",
        wall_clock_ms=2.5,
        verifier_ms=0.4,
        byzantine_rejection_rate=1.0,
        honest_cosine_fidelity=0.99,
        memory_mb=0.5,
    )
    rep = ScalingBenchmarkReport(
        experiment_id="E21",
        roadmap_id="HM-S30-02",
        points=(pt,),
        pareto_winner_by_n={32: "multi_krum_plus_10pct_replay"},
        crossover_swarm_size_n=512,
        all_invariants_passed=True,
    )
    assert rep.to_dict()["experiment_id"] == "E21"

    with pytest.raises(ValueError, match="experiment_id"):
        ScalingBenchmarkReport(
            experiment_id="E21\n",
            roadmap_id="HM-S30-02",
            points=(pt,),
            pareto_winner_by_n={32: "multi_krum_plus_10pct_replay"},
            crossover_swarm_size_n=512,
            all_invariants_passed=True,
        )
    with pytest.raises(ValueError, match="roadmap_id"):
        ScalingBenchmarkReport(
            experiment_id="E21",
            roadmap_id="HM-S30-02\x00",
            points=(pt,),
            pareto_winner_by_n={32: "multi_krum_plus_10pct_replay"},
            crossover_swarm_size_n=512,
            all_invariants_passed=True,
        )

    with pytest.raises(ValueError, match="gradients must be a 2D numpy array"):
        multi_krum_select(np.ones((5, 4), dtype=bool), num_byzantine=1)
    with pytest.raises(ValueError, match="gradients must be a 2D numpy array"):
        multi_krum_select(np.array([["a", "b"], ["c", "d"], ["e", "f"]]), num_byzantine=0)
