# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 194 negative-control and invariant hardening tests across:
- `hivemind.swarm.exemplar_byob_bees`
- `hivemind.swarm.geomesh_fuzzy_consensus`
- `hivemind.swarm.krum_vs_zkopoi_v5_benchmark`
- `hivemind.swarm.crowdgpt_loss_check_service`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hivemind.swarm.crowdgpt_loss_check_service import LossCheckCoordinator, LossCheckJob
from hivemind.swarm.exemplar_byob_bees import (
    EXEMPLAR_BYOB_BEES,
    ExemplarBYOBBee,
    ExemplarBYOBRegistryManager,
)
from hivemind.swarm.geomesh_fuzzy_consensus import (
    FuzzyConsensusResult,
    GeoMeshNode,
    SubSwarmAssignment,
    fuzzy_subswarm_consensus,
    partition_into_fuzzy_subswarms,
)
from hivemind.swarm.krum_vs_zkopoi_v5_benchmark import ScalingRegimePoint


def test_exemplar_byob_bee_invariants_and_paired_outcomes_sequence_guard() -> None:
    base = EXEMPLAR_BYOB_BEES[0]
    base_kw = dict(base.__dict__)

    with pytest.raises(ValueError, match="public_id must equal 'hivemind/<bee_id>'"):
        ExemplarBYOBBee(**{**base_kw, "public_id": "hivemind/wrong-bee-id"})

    with pytest.raises(ValueError, match="revision must be a 40-character hexadecimal commit SHA"):
        ExemplarBYOBBee(**{**base_kw, "revision": "short_rev"})

    with pytest.raises(ValueError, match="revision must be a 40-character hexadecimal commit SHA"):
        ExemplarBYOBBee(**{**base_kw, "revision": "z" * 40})

    with pytest.raises(ValueError, match="test_accuracy_gain must be in"):
        ExemplarBYOBBee(**{**base_kw, "test_accuracy_gain": 1.5})

    with pytest.raises(ValueError, match="test_accuracy_gain must be in"):
        ExemplarBYOBBee(**{**base_kw, "test_accuracy_gain": -1.0})

    mgr = ExemplarBYOBRegistryManager()
    with pytest.raises(ValueError, match="paired_outcomes must not be empty"):
        mgr.register_and_attest_bee(base, paired_outcomes="not-a-pair-seq")  # type: ignore[arg-type]


def test_geomesh_fuzzy_consensus_ip194_hardening() -> None:
    with pytest.raises(ValueError, match="node_id must be a non-empty string without control characters"):
        GeoMeshNode("node\n1", 48.85, 2.35, 3215, {"eu": 12.0})

    with pytest.raises(ValueError, match="anchor_rtts_ms keys must be non-empty strings"):
        GeoMeshNode("node-1", 48.85, 2.35, 3215, {"": 12.0})

    with pytest.raises(ValueError, match="anchor_rtts_ms keys must be non-empty strings"):
        GeoMeshNode("node-1", 48.85, 2.35, 3215, {"eu\x00": 12.0})

    n1 = GeoMeshNode("node-1", 48.85, 2.35, 3215, {"eu": 12.0})
    with pytest.raises(ValueError, match="subswarm_id must be a non-empty string without control characters"):
        SubSwarmAssignment("sub\n1", (n1,), 0.0, 1)

    with pytest.raises(ValueError, match="unique_asn_count must be an integer >= 1 and <= len\\(nodes\\)"):
        SubSwarmAssignment("sub-1", (n1,), 0.0, 2)

    with pytest.raises(ValueError, match="aggregated_gradient must be a non-empty finite numeric"):
        FuzzyConsensusResult(
            subswarm_id="sub-1",
            aggregated_gradient=np.array([True, False], dtype=bool),
            node_weights={"node-1": 1.0},
            rejected_node_ids=(),
            honest_cosine_fidelity=1.0,
        )

    with pytest.raises(ValueError, match="rejected_node_ids must not contain duplicates"):
        FuzzyConsensusResult(
            subswarm_id="sub-1",
            aggregated_gradient=np.array([1.0, 2.0], dtype=np.float32),
            node_weights={"node-1": 1.0, "node-2": 0.0},
            rejected_node_ids=("node-2", "node-2"),
            honest_cosine_fidelity=1.0,
        )

    with pytest.raises(TypeError, match="nodes must be a sequence of GeoMeshNode"):
        partition_into_fuzzy_subswarms(["not-a-node"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="epoch_seed must be a non-empty string without control characters"):
        partition_into_fuzzy_subswarms([n1], epoch_seed="epoch\n0")

    g1 = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="numeric \\(non-boolean\\)"):
        fuzzy_subswarm_consensus("sub-1", {"n1": g1, "n2": np.array([True, False], dtype=bool)})

    with pytest.raises(ValueError, match="reference_gradient must be numeric"):
        fuzzy_subswarm_consensus(
            "sub-1",
            {"n1": g1, "n2": g1},
            reference_gradient=np.array([True, False], dtype=bool),
        )


def test_krum_vs_zkopoi_v5_scaling_regime_point_resilience_and_latency_bounds() -> None:
    pt = ScalingRegimePoint(
        n_workers=15,
        byzantine_workers=6,
        grad_dim=32,
        mode="multi_krum_plus_10pct_replay",
        wall_clock_ms=2.0,
        verifier_ms=0.5,
        byzantine_rejection_rate=1.0,
        honest_cosine_fidelity=0.99,
        memory_mb=0.5,
    )
    assert pt.n_workers == 15

    with pytest.raises(ValueError, match="Multi-Krum resilience bound"):
        ScalingRegimePoint(
            n_workers=14,
            byzantine_workers=6,
            grad_dim=32,
            mode="multi_krum_plus_10pct_replay",
            wall_clock_ms=2.0,
            verifier_ms=0.5,
            byzantine_rejection_rate=1.0,
            honest_cosine_fidelity=0.99,
            memory_mb=0.5,
        )

    with pytest.raises(ValueError, match="verifier_ms cannot exceed wall_clock_ms"):
        ScalingRegimePoint(
            n_workers=15,
            byzantine_workers=6,
            grad_dim=32,
            mode="multi_krum_plus_10pct_replay",
            wall_clock_ms=0.5,
            verifier_ms=2.0,
            byzantine_rejection_rate=1.0,
            honest_cosine_fidelity=0.99,
            memory_mb=0.5,
        )


def test_crowdgpt_loss_check_service_ip194_hardening(tmp_path: Path) -> None:
    valid_kw = dict(
        job_id="job-194",
        base_url="https://example.org/base.st",
        base_sha256="a" * 64,
        candidate_url="https://example.org/cand.st",
        candidate_sha256="b" * 64,
        heldout_url="https://example.org/held.bin",
        heldout_sha256="c" * 64,
        window=16,
        n=8,
        seed=7,
        created_at=10.0,
        expires_at=20.0,
    )
    job = LossCheckJob(**valid_kw)
    assert LossCheckJob.from_wire(job.to_wire()) == job

    with pytest.raises(ValueError, match="must not contain control characters"):
        LossCheckJob(**{**valid_kw, "job_id": "job\n194"})

    with pytest.raises(ValueError, match="base_sha256 must be a 64-character hex string"):
        LossCheckJob(**{**valid_kw, "base_sha256": "short"})

    with pytest.raises(ValueError, match="candidate_sha256 must be a 64-character hex string"):
        LossCheckJob(**{**valid_kw, "candidate_sha256": "g" * 64})

    with pytest.raises(ValueError, match="heldout_range must be a string without control characters"):
        LossCheckJob(**{**valid_kw, "heldout_range": "0-100\x00"})

    with pytest.raises(TypeError, match="wire payload must be a mapping"):
        LossCheckJob.from_wire(["not", "a", "mapping"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="evaluator_keys must map non-empty evaluator IDs"):
        LossCheckCoordinator(
            evaluator_keys={"": b"k"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=lambda _j: (None, None, None),
        )

    with pytest.raises(ValueError, match="evaluator_keys must map non-empty evaluator IDs"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b""},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=lambda _j: (None, None, None),
        )

    with pytest.raises(TypeError, match="loader must be callable"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader="not-callable",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="clock must be callable"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=lambda _j: (None, None, None),
            clock="not-callable",  # type: ignore[arg-type]
        )
