# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 158 hardening tests for BYOB, CampaignOrchestrator, Decoupled DiLoCo, and DHT CAS storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import pytest

from hivemind.swarm.byob import (
    EvaluationReport,
    IsolatedExecutor,
    Reason,
    ReasonCode,
    RepoFile,
    StaticReport,
    Verdict,
)
from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    DomainEvidence,
    JoinDecision,
    NightWindow,
)
from hivemind.swarm.decoupled_diloco import (
    DecoupledDiLoCoQuorumSyncer,
    QuorumSelection,
    SyncerRoundTelemetry,
)
from hivemind.swarm.dht_storage import (
    BundleSeedManifest,
    ChunkedFileEntry,
    ContentAddressedSwarmStore,
    DataShardLease,
)


def test_byob_dataclass_invariants() -> None:
    rf = RepoFile(path="adapter_model.safetensors", size=1024, sha256="abc123")
    assert rf.to_dict()["size"] == 1024

    with pytest.raises(TypeError, match="path"):
        RepoFile(path=123, size=1024, sha256="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size"):
        RepoFile(path="a.safetensors", size=-1, sha256="abc")
    with pytest.raises(TypeError, match="sha256"):
        RepoFile(path="a.safetensors", size=10, sha256=None)  # type: ignore[arg-type]

    sr = StaticReport(verdict=Verdict.ACCEPT, files=[rf], weight_files=[rf], weight_bytes=1024, resident_gb=4.5)
    assert sr.to_dict()["verdict"] == "ACCEPT"

    with pytest.raises(TypeError, match="verdict"):
        StaticReport(verdict="ACCEPT")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reasons"):
        StaticReport(verdict=Verdict.REJECT, reasons=["bad"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="files"):
        StaticReport(verdict=Verdict.REJECT, files=["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="weight_bytes"):
        StaticReport(verdict=Verdict.REJECT, weight_bytes=-5)
    with pytest.raises(ValueError, match="resident_gb"):
        StaticReport(verdict=Verdict.REJECT, resident_gb=-1.0)

    ex = IsolatedExecutor(name="gvisor-gpu", isolated=True, answer_pack=lambda m, f, p: ["ok" for _ in p])
    assert ex.isolated is True
    with pytest.raises(ValueError, match="name"):
        IsolatedExecutor(name=" ", isolated=True, answer_pack=lambda m, f, p: [])
    with pytest.raises(TypeError, match="isolated"):
        IsolatedExecutor(name="gvisor-gpu", isolated=1, answer_pack=lambda m, f, p: [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="answer_pack"):
        IsolatedExecutor(name="gvisor-gpu", isolated=True, answer_pack="not-callable")  # type: ignore[arg-type]

    er = EvaluationReport(
        verdict=Verdict.ACCEPT,
        reasons=[Reason(ReasonCode.STATIC_NOT_PASSED, "ok")],
        pack_id="pack-1",
        n_items=5,
        pass_rate=0.8,
        executor="gvisor-gpu",
    )
    assert er.to_dict()["pass_rate"] == 0.8
    with pytest.raises(ValueError, match="n_items"):
        EvaluationReport(verdict=Verdict.REJECT, n_items=-1)
    with pytest.raises(ValueError, match="pass_rate"):
        EvaluationReport(verdict=Verdict.REJECT, pass_rate=1.5)


def test_campaign_orchestrator_invariants(tmp_path: Path) -> None:
    jd = JoinDecision(
        decision="ADMITTED",
        reason="admitted",
        campaign_id="SUBJ-01",
        participant_id="peer-1",
    )
    assert jd.accepted is True

    with pytest.raises(ValueError, match="invalid decision"):
        JoinDecision(decision="UNKNOWN", reason="ok", campaign_id="SUBJ-01", participant_id="peer-1")
    with pytest.raises(ValueError, match="reason"):
        JoinDecision(decision="REJECTED", reason="  ", campaign_id="SUBJ-01", participant_id="peer-1")
    with pytest.raises(TypeError, match="event and delivery"):
        JoinDecision(decision="REJECTED", reason="bad", campaign_id="SUBJ-01", participant_id="peer-1", event=None)  # type: ignore[arg-type]

    t0 = datetime(2026, 10, 28, 22, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=9)
    nw = NightWindow(start=t0, end=t1, tz="UTC")
    assert nw.duration_h == 9.0

    with pytest.raises(ValueError, match="timezone-aware"):
        NightWindow(start=datetime(2026, 10, 28, 22, 0), end=t1, tz="UTC")
    with pytest.raises(ValueError, match="strictly after"):
        NightWindow(start=t1, end=t0, tz="UTC")
    with pytest.raises(ValueError, match="tz"):
        NightWindow(start=t0, end=t1, tz="")

    de = DomainEvidence(domain="coding_python", verified_accept=100, verified_reject=10, unverified=20, heldout_eval_items=50)
    assert de.new_verified == 100
    with pytest.raises(ValueError, match="invalid domain name"):
        DomainEvidence(domain="Bad Domain", verified_accept=10, verified_reject=0, unverified=0, heldout_eval_items=10)
    with pytest.raises(ValueError, match="counts must be non-negative"):
        DomainEvidence(domain="coding_python", verified_accept=-1, verified_reject=0, unverified=0, heldout_eval_items=10)

    with pytest.raises(TypeError, match="publish must be callable"):
        CampaignOrchestrator(tmp_path / "state.json", b"0123456789abcdef", publish="not-callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy must be a CampaignPolicy"):
        CampaignOrchestrator(tmp_path / "state.json", b"0123456789abcdef", publish=lambda e: {}, policy="bad")  # type: ignore[arg-type]


def test_decoupled_diloco_invariants() -> None:
    qs = QuorumSelection(
        reached=True,
        min_quorum_k=2,
        grace_window_s=0.5,
        admitted_learner_ids=("l1", "l2"),
        deferred_learner_ids=(),
        quorum_reached_at_s=1.0,
        window_close_s=1.5,
    )
    assert qs.reached is True

    with pytest.raises(TypeError, match="reached"):
        QuorumSelection(
            reached=1,  # type: ignore[arg-type]
            min_quorum_k=2,
            grace_window_s=0.5,
            admitted_learner_ids=("l1", "l2"),
            deferred_learner_ids=(),
            quorum_reached_at_s=1.0,
            window_close_s=1.5,
        )
    with pytest.raises(ValueError, match="min_quorum_k"):
        QuorumSelection(
            reached=False,
            min_quorum_k=0,
            grace_window_s=0.0,
            admitted_learner_ids=(),
            deferred_learner_ids=(),
            quorum_reached_at_s=None,
            window_close_s=None,
        )
    with pytest.raises(ValueError, match="admitted_learner_ids"):
        QuorumSelection(
            reached=True,
            min_quorum_k=1,
            grace_window_s=0.0,
            admitted_learner_ids=("",),
            deferred_learner_ids=(),
            quorum_reached_at_s=1.0,
            window_close_s=1.0,
        )

    srt = SyncerRoundTelemetry(
        fragment_name="layers.0.q_proj",
        merge_mode="rda",
        num_arrivals=2,
        num_admitted=2,
        num_deferred=0,
        min_quorum_k=2,
        grace_window_s=0.25,
        admitted_learner_ids=("l1", "l2"),
        learner_weights={"l1": 1.0, "l2": 1.0},
        admitted_tokens=1024,
        deferred_tokens=0,
        merged_delta_norm=0.5,
        direct_average_norm=0.4,
        mean_learner_delta_norm=0.5,
    )
    assert srt.num_admitted == 2

    with pytest.raises(ValueError, match="fragment_name"):
        SyncerRoundTelemetry(
            fragment_name="",
            merge_mode="rda",
            num_arrivals=0,
            num_admitted=0,
            num_deferred=0,
            min_quorum_k=1,
            grace_window_s=0.0,
            admitted_learner_ids=(),
            learner_weights={},
            admitted_tokens=0,
            deferred_tokens=0,
            merged_delta_norm=0.0,
            direct_average_norm=0.0,
            mean_learner_delta_norm=0.0,
        )
    with pytest.raises(TypeError, match="grace_controller"):
        DecoupledDiLoCoQuorumSyncer(num_learners_m=4, min_quorum_k=2, grace_controller="bad")  # type: ignore[arg-type]


def test_dht_storage_invariants(tmp_path: Path) -> None:
    cfe = ChunkedFileEntry(
        relative_path="adapter.safetensors",
        size=1024,
        sha256="a" * 64,
        chunk_size=65536,
        num_chunks=1,
        chunk_hashes=["b" * 64],
        merkle_root="c" * 64,
    )
    assert cfe.num_chunks == 1

    with pytest.raises(ValueError, match="chunk_hashes"):
        ChunkedFileEntry(
            relative_path="adapter.safetensors",
            size=1024,
            sha256="a" * 64,
            chunk_size=65536,
            num_chunks=2,
            chunk_hashes=["b" * 64],
            merkle_root="c" * 64,
        )

    bsm = BundleSeedManifest(
        bundle_id="bundle-1",
        source_node_id="node-1",
        created_at_epoch=1700000000.0,
        chunk_size=65536,
        total_bytes=1024,
        num_chunks=1,
        bundle_merkle_root="d" * 64,
        files=[cfe],
    )
    assert bsm.total_bytes == 1024

    with pytest.raises(TypeError, match="files"):
        BundleSeedManifest(
            bundle_id="bundle-1",
            source_node_id="node-1",
            created_at_epoch=1700000000.0,
            chunk_size=65536,
            total_bytes=1024,
            num_chunks=1,
            bundle_merkle_root="d" * 64,
            files=["not-an-entry"],  # type: ignore[list-item]
        )

    lease = DataShardLease(
        lease_id="lease-1",
        shard_cid="cid-1",
        worker_id="worker-1",
        granted_at_s=10.0,
        ttl_s=30.0,
        expires_at_s=40.0,
    )
    assert lease.status == "active"

    with pytest.raises(ValueError, match="status"):
        DataShardLease(
            lease_id="lease-1",
            shard_cid="cid-1",
            worker_id="worker-1",
            granted_at_s=10.0,
            ttl_s=30.0,
            expires_at_s=40.0,
            status="bogus",
        )
    with pytest.raises(TypeError, match="security_guard"):
        ContentAddressedSwarmStore(tmp_path / "cas", security_guard="bad")  # type: ignore[arg-type]
