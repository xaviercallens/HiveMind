# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for PR-IP-16 (Swarm Integrity):
IP-107, IP-111, IP-112, IP-113, IP-114, IP-115, IP-116, IP-117, IP-118, IP-132, IP-133, IP-134, IP-137.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hivemind.swarm.amcp_kafka_campaign_bus import JOINT_SUBJECT_CAMPAIGNS
from hivemind.swarm.amcp_kafka_transport import (
    AMCPConsumer,
    AMCPPublisher,
    DLQ_TOPIC,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
)
from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    CampaignRequest,
    parse_campaign_utc,
    sign_campaign_request,
    verify_campaign_request_signature,
)
from hivemind.swarm.decoupled_diloco import (
    ContinuousAsyncLearnerCoordinator,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    token_weighted_quality_weight,
)
from hivemind.swarm.inference_ring import (
    RegionalDHTSwarmRing,
    SwarmPeerNode,
)
from hivemind.swarm.la_ruche_vault import load_community_moe_from_gcs_or_https
from hivemind.swarm.ledger import (
    GuildSlashingInsurancePool,
    HiveSwarmLedger,
    StorageProofReceipt,
    SwarmSecurityViolationError,
    WorkReceipt,
)
from hivemind.swarm.s32_03_open_bee_marketplace_registry import (
    OpenBeeMarketplaceRegistry,
    SignedBeeManifest,
    sign_bee_manifest,
)
from hivemind.swarm.s33_01_decoupled_diloco_pacgad import (
    StragglerFreeDecoupledDiLoCoAggregator,
    WorkerPseudoGradient,
)
from hivemind.swarm.s33_02_multibee_amcp_debate import (
    BeeDebateTurn,
    MultiBeeAMCPDebateArena,
)
from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import MAXIMUM_PAIN_ENERGY
from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    audit_lean4_source_elenchus,
    verify_robertson_sundials_tier_x,
)
from hivemind.swarm.s36_02_e21_multikrum_vs_zk_crossover import (
    run_e21_multikrum_vs_zk_crossover,
)
from hivemind.swarm.s36_04_fuzzy_subswarm_consensus import (
    LocalizedFuzzySubSwarmConsensus,
    SubSwarmVote,
)
from hivemind.swarm.s36_06_hourly_coding_assistant_orchestrator import (
    HourlyHiveMindCodingAssistantOrchestrator,
)
from hivemind.swarm.swarm_fabric import (
    OptimisticProofOfInferenceAuditor,
    SpeculativeBatchCommitment,
)


def test_ip107_unattributed_or_forged_commitment_rejected_before_slashing() -> None:
    import torch

    gen = torch.Generator().manual_seed(7)
    honest = [[torch.randn(2, 8, generator=gen) for _ in range(2)] for _ in range(3)]
    claimed = [list(step) for step in honest]
    claimed[1][0] = honest[1][0] * 4.0

    auditor = OptimisticProofOfInferenceAuditor()
    secret = hashlib.sha256(b"ip107-victim-secret").digest()

    ledger = HiveSwarmLedger()
    ledger.balances.update({"victim-node": 100.0, "consumer": 0.0})
    pool = GuildSlashingInsurancePool(ledger)
    pool.bond_stake("guild-alpha", "victim-node", 50.0)

    # 1. Forged commitment constructed outside auditor.commit_speculative_batch
    forged = SpeculativeBatchCommitment(
        peer_id="victim-node",
        root="f" * 64,
        num_steps=3,
        num_layers=2,
    )
    opening_dummy = auditor.commit_speculative_batch("other-peer", claimed)
    op = auditor.open_speculative_batch(1, 0)
    with pytest.raises(SwarmSecurityViolationError) as exc_info:
        pool.indemnify_proven_fault(
            auditor,
            forged,
            op,
            claimed[1][0],
            honest[1][0],
            consumer_node_id="consumer",
            claimed_hive=20.0,
        )
    assert exc_info.value.rule_id == "LEDGER-05"
    assert exc_info.value.category == "UNATTRIBUTED_COMMITMENT"
    assert pool.peer_stakes["victim-node"] == pytest.approx(50.0)

    # 2. Issued commitment without valid peer signature when peer secret is registered
    unsigned_issued = auditor.commit_speculative_batch("victim-node", claimed)
    op2 = auditor.open_speculative_batch(1, 0)
    auditor.register_peer_secret("victim-node", secret)
    with pytest.raises(SwarmSecurityViolationError) as exc_unsigned:
        pool.indemnify_proven_fault(
            auditor,
            unsigned_issued,
            op2,
            claimed[1][0],
            honest[1][0],
            consumer_node_id="consumer",
            claimed_hive=20.0,
        )
    assert exc_unsigned.value.category == "UNATTRIBUTED_COMMITMENT"

    # 3. Legitimately issued and peer-signed commitment with proven fault is indemnified
    signed_issued = auditor.commit_speculative_batch("victim-node", claimed)
    op3 = auditor.open_speculative_batch(1, 0)
    receipt = pool.indemnify_proven_fault(
        auditor,
        signed_issued,
        op3,
        claimed[1][0],
        honest[1][0],
        consumer_node_id="consumer",
        claimed_hive=20.0,
    )
    assert receipt.indemnified_hive == pytest.approx(20.0)
    assert "victim-node" in ledger.slashed_registry


def test_ip111_la_ruche_vault_blocks_path_traversal_and_verifies_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import urllib.request

    cache_dir = tmp_path / "vault_cache"
    outside_target = tmp_path / "escaped.safetensors"
    payload_bytes = b"SAFE_WEIGHTS_BLOB"
    valid_sha = hashlib.sha256(payload_bytes).hexdigest()

    def _fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("ruche_manifest.json"):
            data = json.dumps(
                {
                    "combined_safetensors_path": "../escaped.safetensors",
                    "combined_safetensors_sha256": valid_sha,
                }
            ).encode("utf-8")
            return io.BytesIO(data)
        return io.BytesIO(payload_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(ValueError, match="Unsafe combined_safetensors_path"):
        load_community_moe_from_gcs_or_https(
            cache_dir=cache_dir,
            gcs_public_base_url="https://example.org",
        )
    assert not outside_target.exists()

    # Valid path but pinned expected_safetensors_sha256 mismatch
    def _fake_urlopen_clean_path(req: Any, timeout: float = 30.0) -> Any:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("ruche_manifest.json"):
            data = json.dumps(
                {
                    "combined_safetensors_path": "sub/weights.safetensors",
                    "combined_safetensors_sha256": valid_sha,
                }
            ).encode("utf-8")
            return io.BytesIO(data)
        return io.BytesIO(payload_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_clean_path)
    with pytest.raises(ValueError, match="does not match pinned digest"):
        load_community_moe_from_gcs_or_https(
            cache_dir=cache_dir,
            gcs_public_base_url="https://example.org",
            expected_safetensors_sha256="0" * 64,
        )


def test_ip112_marketplace_nan_score_and_overwrite_blocked_and_debate_nan_vetoed() -> None:
    secret = hashlib.sha256(b"ip112-marketplace-secret").digest()
    reg = OpenBeeMarketplaceRegistry(signing_secret=secret)
    m_owner1 = sign_bee_manifest(
        bee_id="hivemind/bee-shared",
        owner_handle="@alice",
        domain="math",
        base_backbone="Qwen/Qwen2.5-7B",
        lora_rank=16,
        vram_required_mib=4096,
        weights_sha256="a" * 64,
        benchmark_score=0.91,
        signing_secret=secret,
    )
    r1 = reg.verify_and_register(m_owner1)
    assert r1["accepted"] is True

    # Second owner attempting to overwrite @alice's bee_id is rejected
    m_hijack = sign_bee_manifest(
        bee_id="hivemind/bee-shared",
        owner_handle="@mallory",
        domain="math",
        base_backbone="Qwen/Qwen2.5-7B",
        lora_rank=16,
        vram_required_mib=4096,
        weights_sha256="b" * 64,
        benchmark_score=0.99,
        signing_secret=secret,
    )
    r_hijack = reg.verify_and_register(m_hijack)
    assert r_hijack["accepted"] is False
    assert r_hijack["reason"] == "BEE_ID_OWNED"
    assert reg.list_registered_bees()[0]["owner_handle"] == "@alice"

    # NaN benchmark_score is rejected
    m_nan = SignedBeeManifest(
        bee_id="hivemind/bee-nan",
        owner_handle="@bob",
        domain="math",
        base_backbone="Qwen/Qwen2.5-7B",
        lora_rank=16,
        vram_required_mib=4096,
        weights_sha256="c" * 64,
        benchmark_score=float("nan"),
        signature_hex="0" * 64,
    )
    r_nan = reg.verify_and_register(m_nan)
    assert r_nan["accepted"] is False
    assert r_nan["reason"] == "INVALID_SCORE"

    # Debate arena vetoes NaN residual_error
    arena = MultiBeeAMCPDebateArena()
    res = arena.adjudicate_debate(
        subject_id="prob-nan",
        turns=[
            BeeDebateTurn(
                bee_id="hivemind/bee-nan",
                role="solver",
                passed_invariants=True,
                confidence=0.95,
                residual_error=float("nan"),
                evidence_summary="nan residual",
            ),
            BeeDebateTurn(
                bee_id="hivemind/bee-honest",
                role="prover",
                passed_invariants=True,
                confidence=0.95,
                residual_error=1e-6,
                evidence_summary="clean",
            ),
        ],
    )
    assert res.accepted is False
    assert "hivemind/bee-nan" in res.vetoed_by


def test_ip113_and_ip132_diloco_weight_cap_and_nonfinite_rejection() -> None:
    import torch

    # Quadratic inflation capped
    capped_weight = token_weighted_quality_weight(100_000, 1)
    assert capped_weight <= 65_536.0 * 256.0

    # DecoupledDiLoCoQuorumSyncer drops NaN fragment without poisoning merged delta
    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=3, min_quorum_k=2)
    delta_p, _sel, telemetry = syncer.merge_fragment(
        fragment_name="frag-0",
        global_prev=torch.tensor([1.0, 2.0]),
        learner_fragments={
            "bad-peer": torch.tensor([float("nan"), 0.1]),
            "honest-1": torch.tensor([0.8, 1.8]),
            "honest-2": torch.tensor([0.9, 1.9]),
        },
        arrivals=[
            LearnerArrival("bad-peer", 1.0, 4, 512),
            LearnerArrival("honest-1", 1.1, 4, 512),
            LearnerArrival("honest-2", 1.2, 4, 512),
        ],
        grace_window_s=1.0,
    )
    assert delta_p is not None
    assert bool(torch.isfinite(delta_p).all())
    assert "bad-peer" not in telemetry.admitted_learner_ids

    # StragglerFreeDecoupledDiLoCoAggregator drops NaN fragment without poisoning Updated_parameters
    agg = StragglerFreeDecoupledDiLoCoAggregator()
    round_res = agg.aggregate_and_step(
        current_params=[1.0, 1.0, 1.0, 1.0],
        contributions=[
            WorkerPseudoGradient(
                worker_id="poison-peer",
                staleness_steps=0,
                inner_steps=4,
                pseudo_gradient=(float("nan"), 1.0, 1.0, 1.0),
            ),
            WorkerPseudoGradient(
                worker_id="honest-1",
                staleness_steps=0,
                inner_steps=4,
                pseudo_gradient=(0.2, 0.2, 0.2, 0.2),
            ),
            WorkerPseudoGradient(
                worker_id="honest-2",
                staleness_steps=1,
                inner_steps=4,
                pseudo_gradient=(0.2, 0.2, 0.2, 0.2),
            ),
        ],
    )
    assert all(abs(w) < 10.0 for w in round_res.Updated_parameters)

    # IP-132: ContinuousAsyncLearnerCoordinator rejects NaN before mutating counters and never wedges
    coord = ContinuousAsyncLearnerCoordinator(num_learners_m=3, min_quorum_k=2)
    with pytest.raises(ValueError, match="finite"):
        coord.submit_learner_delta(
            "nan-node",
            origin_round=0,
            delta=torch.tensor([float("nan"), 0.1]),
            tokens_since_sync=100,
            steps_since_sync=2,
        )
    assert coord.total_submitted_tokens == 0

    # Zero-step submission does not wedge quorum round when K honest learners also submit
    coord.submit_learner_delta(
        "zero-step-node",
        origin_round=0,
        delta=torch.tensor([0.1, 0.1]),
        tokens_since_sync=100,
        steps_since_sync=0,
        arrival_time_s=0.5,
    )
    coord.submit_learner_delta(
        "h1",
        origin_round=0,
        delta=torch.tensor([0.2, 0.2]),
        tokens_since_sync=200,
        steps_since_sync=4,
        arrival_time_s=1.0,
    )
    coord.submit_learner_delta(
        "h2",
        origin_round=0,
        delta=torch.tensor([0.2, 0.2]),
        tokens_since_sync=200,
        steps_since_sync=4,
        arrival_time_s=1.1,
    )
    updated_p, meta = coord.close_quorum_round(torch.tensor([1.0, 1.0]), grace_window_s=1.0)
    assert meta["quorum_reached"] is True
    assert bool(torch.isfinite(updated_p).all())
    assert coord.current_round == 1


def test_ip114_inference_ring_tracks_draft_rejections_and_suppresses_bonus_token() -> None:
    from types import SimpleNamespace
    import torch
    import torch.nn as nn

    class _DraftMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor) -> Any:
            b, t = input_ids.shape
            logits = torch.full((b, t, 8), -10.0)
            logits[:, :, 2] = 10.0  # draft always proposes token 2
            return SimpleNamespace(logits=logits)

    class _TargetMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def reset_routing_telemetry(self) -> None:
            pass

        def get_expert_utilization(self) -> dict[str, float]:
            return {"math": 1.0}

        def forward(self, input_ids: torch.Tensor) -> Any:
            b, t = input_ids.shape
            logits = torch.full((b, t, 8), -20.0)
            logits[:, :, 5] = 20.0
            return SimpleNamespace(logits=logits)

    ring = RegionalDHTSwarmRing()
    ring.register_peer(
        SwarmPeerNode(
            peer_id="peer-eu-1",
            steam_id64="76561198000000001",
            region="EU-West-Paris",
            ping_ms=10.0,
            vram_gb=8.0,
            hosted_expert_domain="math",
            layer_range=(0, 4),
        )
    )
    res = ring.run_speculative_inference(
        moe_target_model=_TargetMod(),  # type: ignore[arg-type]
        draft_model=_DraftMod(),
        prompt_ids=torch.tensor([[1, 1]]),
        max_new_tokens=2,
        draft_gamma=3,
    )
    # Draft proposed token 2, target preferred token 5 -> rejected on first token of pass (acceptance rate = 0.0 < 1.0)
    assert res.mean_acceptance_rate < 1.0
    assert res.generated_tokens == [5, 5]
    # Because draft_accepted_in_pass < draft_gamma, each pass emitted only 1 correction token (no bonus token)
    assert res.swarm_verification_passes == 2


def test_ip115_elenchus_rejects_axioms_and_unattached_robertson_call() -> None:
    axiom_lean = "axiom magic_false : False\ntheorem anything : False := magic_false\n"
    audit = audit_lean4_source_elenchus(axiom_lean)
    assert audit.quarantined_axiom_count == 1
    assert audit.is_kernel_clean is False
    assert audit.classification == "QUARANTINED_AXIOM"

    args = {"problem": "robertson", "rtol": 1e-9, "atol": 1e-12}
    unattached = verify_robertson_sundials_tier_x(sent_arguments=args, received_arguments=args)
    assert unattached.accepted_by_elenchus_bridge is False
    assert unattached.evidentiary_value == "none"


def test_ip116_and_ip137_e21_crossover_estimation_mode_and_forged_control() -> None:
    report = run_e21_multikrum_vs_zk_crossover()
    payload = report.to_dict()
    assert payload["estimation_mode"] == "model-based estimate"
    assert payload["measured_flop_ms_coeff"] > 0.0
    assert report.gate_passed is True

    # Negative controls: forged proof or overriding single_zk_verify_ms above Multi-Krum causes gate_passed=False
    forged_report = run_e21_multikrum_vs_zk_crossover(forged_proof_control=True)
    assert forged_report.gate_passed is False
    bad_report = run_e21_multikrum_vs_zk_crossover(single_zk_verify_ms_override=1e9)
    assert bad_report.gate_passed is False


def test_ip117_hourly_orchestrator_blocks_when_offline_and_runs_task_assertions() -> None:
    orch = HourlyHiveMindCodingAssistantOrchestrator(api_key=None)
    rep = orch.run_hourly_improvement_cycle(call_live_openrouter=False)
    assert rep.live_openrouter_called is False
    assert rep.status == "BLOCKED"
    assert rep.blocked_reason == "model not called"

    # Failing task assertion sets unit_verification_passed=False and MAXIMUM_PAIN_ENERGY
    rep_fail = orch.run_hourly_improvement_cycle(
        call_live_openrouter=False,
        task_assertions={"HM-S36-01": lambda _mod: False},
    )
    r0 = next(r for r in rep_fail.task_receipts if r.task_id == "HM-S36-01")
    assert r0.unit_verification_passed is False
    assert r0.anse_energy == MAXIMUM_PAIN_ENERGY
    assert rep_fail.all_gates_passed is False


def test_ip118_fuzzy_subswarm_rejects_duplicate_node_ids_and_self_reported_reputation() -> None:
    consensus = LocalizedFuzzySubSwarmConsensus()
    dup_votes = [
        SubSwarmVote("peer-00", "ans-a", 0.9, 0.9) for _ in range(15)
    ]
    with pytest.raises(ValueError, match="Duplicate node_id"):
        consensus.evaluate_subswarm(dup_votes)

    # 8 Sybil voters claim observer_reputation=1.0, 10 honest voters claim observer_reputation=0.05;
    # without a reputation_registry, self-reported reputation is ignored (uniform 1.0), so the 10 honest voters win.
    votes = [
        SubSwarmVote(
            node_id=f"peer-{i:02d}",
            answer_digest="ans-sybil" if i < 8 else "ans-honest",
            semantic_confidence=0.95,
            observer_reputation=1.0 if i < 8 else 0.05,
        )
        for i in range(18)
    ]
    rep = consensus.evaluate_subswarm(votes)
    assert rep.winning_digest == "ans-honest"


def test_ip133_campaign_orchestrator_non_ascii_signature_and_amcp_kafka_roundtrip(
    tmp_path: Path,
) -> None:
    secret = hashlib.sha256(b"ip133-orchestrator-secret").digest()
    campaign = JOINT_SUBJECT_CAMPAIGNS[0]
    now_utc = parse_campaign_utc(campaign.dtend_utc) - 10.0

    bad_req = CampaignRequest(
        action="join",
        participant_id="alice",
        participant_kind="gpu_producer",
        campaign_id=campaign.subject_id,
        declared_vram_mib=16384,
        availability="eu_evening_cet",
        issued_at_utc=now_utc,
        nonce="nonce-nonascii-01",
        signature="é" * 64,
    )
    assert verify_campaign_request_signature(bad_req, secret) is False

    transport = InMemoryTransport()
    keyring = HMACKeyring(keys={"k1": secret}, active_kid="k1")
    publisher = AMCPPublisher(transport, keyring)
    orch = CampaignOrchestrator(
        state_path=tmp_path / "campaigns_state.json",
        secret=secret,
        publish=publisher.publish,
    )

    # 1. Non-ASCII signature returns REJECTED cleanly and publishes valid AMCP event
    dec_bad = orch.handle(bad_req, now_utc)
    assert dec_bad.decision == "REJECTED"
    assert dec_bad.delivery["status"] == "DELIVERED"

    # 2. Valid join -> ADMITTED, duplicate join -> ALREADY_MEMBER, leave -> LEFT, expire -> EXPIRED
    join_req = sign_campaign_request(
        CampaignRequest(
            action="join",
            participant_id="alice",
            participant_kind="gpu_producer",
            campaign_id=campaign.subject_id,
            declared_vram_mib=16384,
            availability="eu_evening_cet",
            issued_at_utc=now_utc,
            nonce="nonce-valid-join-01",
        ),
        secret,
    )
    dec_join = orch.handle(join_req, now_utc)
    assert dec_join.decision == "ADMITTED"
    assert dec_join.delivery["status"] == "DELIVERED"

    end_utc = parse_campaign_utc(campaign.dtend_utc) + 5.0
    expired_list = orch.expire(end_utc)
    assert len(expired_list) == 1
    assert expired_list[0].decision == "EXPIRED"
    assert expired_list[0].delivery["status"] == "DELIVERED"

    # Consume all emitted events through AMCPConsumer: 0 DLQ records
    seen_events: list[str] = []
    verifier = EnvelopeVerifier(keyring, clock=lambda: end_utc)
    consumer = AMCPConsumer(
        transport,
        verifier,
        clock=lambda: end_utc,
    )
    consumer.register_topic("campaigns", lambda ve: seen_events.append(ve.event.event_type))
    consumer.run_once()
    assert len(transport.records(DLQ_TOPIC)) == 0
    assert "io.hivemind.amcp.campaign.admitted" in seen_events
    assert "io.hivemind.amcp.campaign.expired" in seen_events


def test_ip134_ledger_persists_full_state_and_validates_storage_before_mutation(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "ledger_state.json"
    ledger = HiveSwarmLedger(ledger_path=state_file)
    ledger.credit_reward("peer-a", 100.0)
    pool = GuildSlashingInsurancePool(ledger)
    pool.bond_stake("guild-1", "peer-a", 40.0)
    pool.contribute_reserve("guild-1", "peer-a", 20.0)
    supply_before = ledger.total_minted_supply()
    assert ledger.get_balance("peer-a") + pool.total_holdings() == pytest.approx(supply_before)

    # Reload from state_file and verify full conservation
    reloaded = HiveSwarmLedger(ledger_path=state_file)
    reloaded_pool = GuildSlashingInsurancePool(reloaded)
    assert reloaded.get_balance("peer-a") == pytest.approx(40.0)
    assert reloaded_pool.peer_stakes["peer-a"] == pytest.approx(40.0)
    assert reloaded_pool.reserve_hive == pytest.approx(20.0)
    assert reloaded_pool.total_holdings() == pytest.approx(60.0)
    assert reloaded.total_minted_supply() == pytest.approx(supply_before)

    # Negative storage receipt raises SwarmSecurityViolationError before slashing or mutating peer balances
    bal_before = reloaded.get_balance("peer-a")
    with pytest.raises(SwarmSecurityViolationError) as exc_info:
        reloaded.mine_pouw_block(
            outer_step=1,
            expert_cid="cid://expert-math",
            work_receipts=[
                WorkReceipt(
                    node_id="peer-a",
                    steam_id64="76561198000000001",
                    branch_name="math",
                    flops_contributed=1e12,
                    cosine_alignment=0.9,
                    spot_check_passed=False,
                    multikrum_accepted=False,
                    delta_sha256="d" * 64,
                )
            ],
            storage_receipts=[
                StorageProofReceipt(
                    node_id="peer-a",
                    shard_cid="cid-bad",
                    challenged_chunk_index=0,
                    merkle_proof_hash="e" * 64,
                    bytes_stored=-1024,
                    verified=True,
                )
            ],
            validator_node_ids=["val-1"],
        )
    assert exc_info.value.category == "NEGATIVE_STORAGE_BYTES_RECEIPT"
    assert reloaded.get_balance("peer-a") == pytest.approx(bal_before)
    assert "peer-a" not in reloaded.slashed_registry
