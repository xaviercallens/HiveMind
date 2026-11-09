# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and input-validation tests for decoupled DiLoCo, DHT storage, DiLoCo nextgen, and diurnal MoE engine."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.swarm.decoupled_diloco import (
    ContinuousAsyncLearnerCoordinator,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    radial_directional_average,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.diloco_nextgen import (
    CosineGatedAdamDecayOuterOptimizer,
    HeLoCoDirectionCorrector,
    ReputationWeightedIsoCTSMerger,
)
from hivemind.swarm.dynamic_diurnal_moe_engine import (
    DiurnalMoETrainingScheduler,
    DiurnalProducerProfile,
)


def test_decoupled_diloco_rda_and_syncer_validation() -> None:
    v1 = torch.tensor([3.0, 0.0], dtype=torch.float32)
    v2 = torch.tensor([0.0, 3.0], dtype=torch.float32)
    merged = radial_directional_average([v1, v2], eps=1e-9)
    assert merged.shape == (2,)

    with pytest.raises(ValueError, match="eps must be a positive finite number"):
        radial_directional_average([v1, v2], eps=0.0)

    with pytest.raises(ValueError, match="eps must be a positive finite number"):
        radial_directional_average([v1, v2], eps=True)  # type: ignore[arg-type]

    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=4, min_quorum_k=2)
    arr = [
        LearnerArrival("l1", 1.0, 10, 100),
        LearnerArrival("l2", 1.5, 10, 100),
    ]
    g_prev = torch.tensor([1.0, 2.0], dtype=torch.float32)
    frags = {"l1": torch.tensor([0.5, 1.5]), "l2": torch.tensor([0.6, 1.4])}
    delta, sel, _ = syncer.merge_fragment("layer.0", g_prev, frags, arr)
    assert sel.reached is True and delta is not None

    with pytest.raises(ValueError, match="fragment_name must be a non-empty string"):
        syncer.merge_fragment("   ", g_prev, frags, arr)

    with pytest.raises(TypeError, match="global_prev must be a torch.Tensor"):
        syncer.merge_fragment("layer.0", [1.0, 2.0], frags, arr)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="learner_fragments must be a dict"):
        syncer.merge_fragment("layer.0", g_prev, [frags], arr)  # type: ignore[arg-type]


def test_continuous_async_learner_coordinator_validation() -> None:
    coord = ContinuousAsyncLearnerCoordinator(num_learners_m=3, min_quorum_k=2)
    res = coord.submit_learner_delta("l1", 0, torch.tensor([1.0, -1.0]), 64, 4, 1.0)
    assert res["accepted"] is True

    with pytest.raises(ValueError, match="learner_id must be a non-empty string"):
        coord.submit_learner_delta("  ", 0, torch.tensor([1.0]), 64, 4)

    with pytest.raises(ValueError, match="origin_round must be a non-negative integer"):
        coord.submit_learner_delta("l2", -1, torch.tensor([1.0]), 64, 4)

    with pytest.raises(ValueError, match="origin_round must be a non-negative integer"):
        coord.submit_learner_delta("l2", True, torch.tensor([1.0]), 64, 4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="delta must be a non-empty torch.Tensor"):
        coord.submit_learner_delta("l2", 0, torch.tensor([]), 64, 4)


def test_dht_storage_proof_and_manifest_validation(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(root_dir=tmp_path / "vault")

    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        store.verify_proof_of_storage("  ", "cid://sha256/abc", tmp_path / "missing", "block123")

    with pytest.raises(ValueError, match="shard_cid must be a non-empty string"):
        store.verify_proof_of_storage("node-1", "  ", tmp_path / "missing", "block123")

    with pytest.raises(ValueError, match="prev_block_hash must be a non-empty string"):
        store.verify_proof_of_storage("node-1", "cid://sha256/abc", tmp_path / "missing", "")

    with pytest.raises(ValueError, match="bundle_id must be a non-empty string"):
        store.load_seed_manifest("   ")


def test_diloco_nextgen_cgad_isoc_and_heloco_validation() -> None:
    opt = CosineGatedAdamDecayOuterOptimizer(lr=1e-3, alpha=0.2, tau_cut=16.0)
    gamma, sigma = opt.compute_gate(4.0)
    assert 0.0 < sigma <= gamma <= 1.0

    with pytest.raises(ValueError, match="effective_tau must be a finite number"):
        opt.compute_gate(float("nan"))

    with pytest.raises(ValueError, match="effective_tau must be a finite number"):
        opt.compute_gate(True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="matrix must be a torch.Tensor"):
        ReputationWeightedIsoCTSMerger.apply_isoc_2d([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    corrector = HeLoCoDirectionCorrector(conflict_damping=0.25)
    g = torch.tensor([1.0, -1.0])
    m = torch.tensor([-1.0, 1.0])
    corrected, meta = corrector.correct_pseudo_gradient(g, m)
    assert corrected.shape == g.shape and meta["conflict_detected"] == 1.0

    with pytest.raises(TypeError, match="pseudo_grad and outer_momentum_ref must be torch.Tensor instances"):
        corrector.correct_pseudo_gradient([1.0, -1.0], m)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pseudo_grad and outer_momentum_ref must have the same shape"):
        corrector.correct_pseudo_gradient(g, torch.tensor([1.0, 2.0, 3.0]))


def test_diurnal_moe_scheduler_profile_validation() -> None:
    sched = DiurnalMoETrainingScheduler()
    prof = DiurnalProducerProfile(
        account_id="prod-01",
        handle="alpha.bee",
        vram_gb=16.0,
        hardware_class="RTX_4080",
        timezone_offset_hours=1.0,
    )
    sched.register_producer_profile(prof)
    assert sched.get_producer_profile("prod-01") == prof

    with pytest.raises(TypeError, match="profile must be a DiurnalProducerProfile"):
        sched.register_producer_profile({"account_id": "prod-01"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="account_id must be a non-empty string"):
        sched.get_producer_profile("   ")
