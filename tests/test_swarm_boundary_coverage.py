"""Swarm boundary contracts; transport tests use only in-memory doubles.

This module is authored for execution in the reviewed remote quarantine only.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
import torch

from hivemind.swarm import architecture_hypothesis_verifier as architecture
from hivemind.swarm import decoupled_diloco as decoupled
from hivemind.swarm import dht_storage as storage
from hivemind.swarm import diloco_nextgen as diloco
from hivemind.swarm import inference_ring as inference
from hivemind.swarm import nat_traversal as nat
from hivemind.swarm import quic_link as link
from hivemind.swarm import quic_transport as transport
from hivemind.swarm import scientific_swarm_eval as scientific
from hivemind.swarm import strategy_optimizer as strategy
from hivemind.swarm import wan_fault_benchmark as wan
from hivemind.swarm.ledger import HiveSwarmLedger


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"learner_id": ""}, "learner_id"),
        ({"arrival_time_s": float("nan")}, "arrival_time_s"),
        ({"arrival_time_s": float("inf")}, "arrival_time_s"),
        ({"steps_since_sync": -1}, "steps_since_sync"),
        ({"tokens_since_sync": -1}, "tokens_since_sync"),
    ],
)
def test_arrival_rejects_invalid_identity_time_and_counters(kwargs, message):
    values = dict(learner_id="a", arrival_time_s=0.0, steps_since_sync=1, tokens_since_sync=4)
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        decoupled.LearnerArrival(**values)


@pytest.mark.parametrize("grace", [-1.0, float("inf"), float("nan")])
def test_quorum_rejects_invalid_grace(grace):
    with pytest.raises(ValueError, match="grace_window_s"):
        decoupled.select_quorum([], min_quorum_k=1, grace_window_s=grace)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"overlap_steps_tau": 0}, "overlap_steps_tau"),
        ({"ema_beta": 1.0}, "ema_beta"),
        ({"ema_beta": float("nan")}, "ema_beta"),
        ({"sync_duration_s": -1.0}, "sync_duration_s"),
        ({"sync_duration_s": float("inf")}, "sync_duration_s"),
    ],
)
def test_grace_controller_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        decoupled.AdaptiveGraceWindowController(**kwargs)


def test_grace_telemetry_reports_measured_slack_and_grace():
    controller = decoupled.AdaptiveGraceWindowController(
        overlap_steps_tau=3, safety_margin_gamma=0.25, sync_duration_s=0.5
    )
    assert controller.telemetry()["step_duration_ema_s"] == 0.0
    controller.observe_step_duration(2.0)
    controller.observe_quorum_duration(1.5)
    assert controller.telemetry() == {
        "overlap_steps_tau": 3.0,
        "safety_margin_gamma": 0.25,
        "step_duration_ema_s": 2.0,
        "quorum_duration_ema_s": 1.5,
        "sync_duration_s": 0.5,
        "slack_s": 4.0,
        "grace_window_s": 1.0,
    }


@pytest.mark.parametrize("average", [decoupled.direct_weighted_average, decoupled.radial_directional_average])
@pytest.mark.parametrize(
    "values,weights,message",
    [
        ([], None, "at least one vector"),
        ([[1.0], [2.0]], [1.0], "one entry per vector"),
        ([[1.0]], [-1.0], "finite and >= 0"),
        ([[1.0]], [float("inf")], "finite and >= 0"),
        ([[1.0]], [float("nan")], "finite and >= 0"),
    ],
)
def test_averages_reject_invalid_vector_weights(average, values, weights, message):
    with pytest.raises(ValueError, match=message):
        average([torch.tensor(value) for value in values], weights=weights)


def test_syncer_rejects_impossible_swarm_and_excess_arrivals():
    with pytest.raises(ValueError, match="num_learners_m"):
        decoupled.DecoupledDiLoCoQuorumSyncer(0, 1)
    syncer = decoupled.DecoupledDiLoCoQuorumSyncer(1, 1)
    arrivals = [decoupled.LearnerArrival(name, 0.0, 1, 1) for name in ("a", "b")]
    with pytest.raises(ValueError, match="more arrivals"):
        syncer.merge_fragment("w", torch.zeros(1), {}, arrivals)


def _completed_async_round():
    coordinator = decoupled.ContinuousAsyncLearnerCoordinator(2, 1, merge_mode="direct")
    coordinator.submit_learner_delta("a", 0, torch.tensor([1.0]), 2, 1)
    updated, report = coordinator.close_quorum_round(torch.tensor([10.0]), outer_lr=1.0)
    torch.testing.assert_close(updated, torch.tensor([9.0]))
    assert report["token_conservation_holds"]
    return coordinator


def test_async_stale_updates_accumulate_then_join_fresh_delta_without_losing_tokens():
    coordinator = _completed_async_round()
    for delta, tokens in [(2.0, 3), (3.0, 4)]:
        result = coordinator.submit_learner_delta("b", 0, torch.tensor([delta]), tokens, 1)
        assert result["disposition"] == "deferred_carry_with_token_weight"
    result = coordinator.submit_learner_delta("b", 1, torch.tensor([4.0]), 5, 1)
    assert result["disposition"] == "fresh_inbox"
    updated, report = coordinator.close_quorum_round(torch.tensor([20.0]), outer_lr=1.0)
    torch.testing.assert_close(updated, torch.tensor([11.0]))
    assert report["admitted_learners"] == ["b"]
    assert report["merged_tokens"] == report["total_submitted_tokens"] == 14
    assert report["pending_tokens"] == report["dropped_tokens"] == 0
    assert report["token_conservation_holds"]


def test_async_fresh_then_stale_same_learner_combines_at_round_close():
    coordinator = _completed_async_round()
    coordinator.submit_learner_delta("b", 1, torch.tensor([2.0]), 3, 1, arrival_time_s=0.5)
    coordinator.submit_learner_delta("b", 0, torch.tensor([3.0]), 4, 2)
    updated, report = coordinator.close_quorum_round(torch.tensor([20.0]), outer_lr=1.0)
    torch.testing.assert_close(updated, torch.tensor([15.0]))
    assert report["admitted_learners"] == ["b"]
    assert report["total_submitted_tokens"] == report["merged_tokens"] == 9
    assert report["token_conservation_holds"]


@pytest.mark.parametrize("straggler_mode", decoupled.ContinuousAsyncLearnerCoordinator.STRAGGLER_MODES)
def test_async_full_quorum_defers_or_drops_late_tokens(straggler_mode):
    coordinator = decoupled.ContinuousAsyncLearnerCoordinator(
        3, 2, straggler_mode=straggler_mode, merge_mode="direct"
    )
    for learner_id, arrival, delta, tokens in [
        ("a", 0.0, [2.0, 0.0], 100),
        ("b", 1.0, [2.0, 0.0], 200),
        ("c", 5.0, [7.0, 0.0], 300),
    ]:
        coordinator.submit_learner_delta(
            learner_id, 0, torch.tensor(delta), tokens, 1, arrival_time_s=arrival
        )
    initial = torch.zeros(2)
    updated, report = coordinator.close_quorum_round(
        initial, outer_lr=0.5, grace_window_s=0.0
    )
    torch.testing.assert_close(updated, torch.tensor([-1.0, 0.0]))
    torch.testing.assert_close(initial, torch.zeros(2))
    assert report["admitted_learners"] == ["a", "b"]
    assert report["deferred_learners"] == ["c"]
    assert report["merged_tokens"] == 300
    assert report["dropped_tokens"] == (300 if straggler_mode == "drop" else 0)
    assert report["total_submitted_tokens"] == 600
    assert report["pending_tokens"] == (0 if straggler_mode == "drop" else 300)
    assert report["token_conservation_holds"] is True
    if straggler_mode != "drop":
        coordinator.submit_learner_delta("a", 1, torch.tensor([1.0, 0.0]), 300, 1)
        carried, next_report = coordinator.close_quorum_round(updated, outer_lr=0.5)
        torch.testing.assert_close(carried, torch.tensor([-3.0, 0.0]))
        assert set(next_report["admitted_learners"]) == {"a", "c"}
        assert next_report["merged_tokens"] == next_report["total_submitted_tokens"] == 900
        assert next_report["pending_tokens"] == next_report["dropped_tokens"] == 0
        assert next_report["token_conservation_holds"] is True


def test_campaign_delta_analysis_rejects_single_learner_without_a_pair():
    base = {"w": torch.zeros(2)}
    branch = {"w": torch.tensor([1.0, 2.0])}
    with pytest.raises(ValueError, match="at least 2 learner branches are required"):
        decoupled.analyze_campaign_rda_and_delta_cosine(base, {"only": branch})
    torch.testing.assert_close(base["w"], torch.zeros(2))
    torch.testing.assert_close(branch["w"], torch.tensor([1.0, 2.0]))


def test_campaign_delta_analysis_matches_orthogonal_equal_norm_vectors():
    report = decoupled.analyze_campaign_rda_and_delta_cosine(
        {"w": torch.zeros(2)},
        {"a": {"w": torch.tensor([-2.0, 0.0])}, "b": {"w": torch.tensor([0.0, -2.0])}},
    )
    assert report["num_pairs"] == 1
    assert report["mean_pairwise_delta_cosine"] == 0.0
    assert report["rda_merged_norm"] == pytest.approx(2.0)
    assert report["direct_merged_norm"] == pytest.approx(2.0 ** 0.5)
    assert report["rda_to_direct_norm_ratio_rt"] == pytest.approx(2.0 ** 0.5)
    assert report["theoretical_common_norm_ratio"] == pytest.approx(2.0 ** 0.5)
    assert report["low_cosine_regime_active"] is True


@pytest.mark.parametrize("field,bad", [("alpha", 0.0), ("alpha", float("nan")), ("t_step_s", -1.0), ("t_step_s", float("inf")), ("rtt_s", -1.0), ("rtt_s", float("nan"))])
def test_lease_ttl_rejects_invalid_time_components(field, bad):
    values = dict(alpha=2.0, t_step_s=3.0, rtt_s=0.25)
    values[field] = bad
    with pytest.raises(ValueError, match=field):
        storage.compute_lease_ttl_s(**values)


@pytest.fixture
def swarm_store(tmp_path):
    return storage.ContentAddressedSwarmStore(tmp_path / "vault", node_id="boundary-peer")


def test_checkpoint_rejects_non_dictionary_and_mismatched_index_cid(swarm_store, tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save([torch.ones(1)], checkpoint)
    with pytest.raises(storage.ChunkIntegrityError, match="not a dictionary"):
        swarm_store.load_lora_expert_checkpoint(checkpoint)
    torch.save({"cid": "cid://wrong"}, checkpoint)
    swarm_store.cid_to_path["cid://expected"] = checkpoint
    with pytest.raises(storage.ChunkIntegrityError, match="CID mismatch"):
        swarm_store.load_lora_expert_checkpoint("cid://expected")


def test_chunk_and_bundle_limits_reject_before_writing(swarm_store, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "MAX_CHUNK_BYTES", 3)
    with pytest.raises(storage.ChunkIntegrityError, match="exceeds MAX_CHUNK_BYTES"):
        swarm_store.put_chunk(b"four")
    assert list(swarm_store.chunks_dir.iterdir()) == []
    manifest = storage.BundleSeedManifest(
        bundle_id="oversized", source_node_id="peer", created_at_epoch=0.0,
        chunk_size=1, total_bytes=storage.MAX_BUNDLE_BYTES + 1, num_chunks=0,
        bundle_merkle_root="", files=[],
    )
    target = tmp_path / "never-created"
    with pytest.raises(storage.ChunkIntegrityError, match="exceeds MAX_BUNDLE_BYTES"):
        swarm_store.download_bundle_chunks(manifest, target)
    assert not target.exists()


@pytest.mark.parametrize("shard,worker", [("", "worker"), ("shard", "")])
def test_shard_lease_requires_identity_without_creating_state(swarm_store, shard, worker):
    with pytest.raises(ValueError, match="non-empty"):
        swarm_store.grant_shard_lease(shard, worker, now_s=0.0)
    assert swarm_store.lease_ledger_summary()["total_granted"] == 0


def test_active_lease_is_exclusive_and_reclaimed_shard_can_be_leased_again(swarm_store):
    lease = swarm_store.grant_shard_lease("shard", "a", now_s=0.0)
    with pytest.raises(ValueError, match="already has an active lease"):
        swarm_store.grant_shard_lease("shard", "b", now_s=0.5)
    assert swarm_store.reclaim_expired_leases(now_s=3.0) == [lease]
    replacement = swarm_store.grant_shard_lease("shard", "b", now_s=4.0)
    assert replacement.lease_id != lease.lease_id
    assert replacement.worker_id == "b"
    assert "shard" not in swarm_store._reclaim_queue
    assert swarm_store.lease_ledger_summary()["active_count"] == 1


@pytest.mark.parametrize("operation", ["heartbeat", "commit"])
def test_unknown_lease_does_not_create_state(swarm_store, operation):
    with pytest.raises(KeyError, match="Unknown lease_id"):
        if operation == "heartbeat":
            swarm_store.heartbeat_shard_lease("missing", now_s=1.0)
        else:
            swarm_store.commit_shard_lease("missing", "worker", now_s=1.0)
    assert swarm_store.lease_ledger_summary()["total_granted"] == 0


def test_lease_expiry_and_ownership_failures_preserve_active_state(swarm_store):
    lease = swarm_store.grant_shard_lease("shard", "owner", now_s=10.0)
    expiry = lease.expires_at_s
    with pytest.raises(ValueError, match="does not own"):
        swarm_store.commit_shard_lease(lease.lease_id, "intruder", now_s=11.0)
    with pytest.raises(ValueError, match="expired"):
        swarm_store.heartbeat_shard_lease(lease.lease_id, now_s=expiry + 0.1)
    with pytest.raises(ValueError, match="expired"):
        swarm_store.commit_shard_lease(lease.lease_id, "owner", now_s=expiry + 0.1)
    assert lease.expires_at_s == expiry
    assert lease.heartbeat_count == 0
    assert lease.status == "active"
    assert swarm_store.lease_ledger_summary()["committed_count"] == 0


@pytest.mark.parametrize("status", ["committed", "reclaimed"])
def test_terminal_lease_refuses_heartbeat_and_second_commit(swarm_store, status):
    lease = swarm_store.grant_shard_lease("shard", "owner", now_s=0.0)
    if status == "committed":
        swarm_store.commit_shard_lease(lease.lease_id, "owner", now_s=1.0)
    else:
        swarm_store.reclaim_expired_leases(now_s=3.0)
    with pytest.raises(ValueError, match="status"):
        swarm_store.heartbeat_shard_lease(lease.lease_id, now_s=4.0)
    with pytest.raises(ValueError, match="not active"):
        swarm_store.commit_shard_lease(lease.lease_id, "owner", now_s=4.0)
    assert lease.status == status


@pytest.mark.parametrize(
    "factory,kwargs,message",
    [
        (diloco.CosineGatedAdamDecayOuterOptimizer, {"momentum_restart_period": 0}, "momentum_restart_period"),
        (diloco.ComputeVisibleSparseLoCoCompressor, {"codec": "unknown"}, "codec"),
        (diloco.ComputeVisibleSparseLoCoCompressor, {"target_bits_per_dim": 0.0}, "target_bits_per_dim"),
        (diloco.ComputeVisibleSparseLoCoCompressor, {"target_bits_per_dim": float("nan")}, "target_bits_per_dim"),
        (diloco.ComputeVisibleSparseLoCoCompressor, {"lattice_scale": -1.0}, "lattice_scale"),
        (diloco.ComputeVisibleSparseLoCoCompressor, {"lattice_scale": float("inf")}, "lattice_scale"),
        (diloco.FactoredGossipMix1Mix2Coordinator, {"alpha_mix1": 0.0}, "alpha_mix1"),
        (diloco.FactoredGossipMix1Mix2Coordinator, {"js_threshold": -1.0}, "js_threshold"),
        (diloco.FactoredGossipMix1Mix2Coordinator, {"outer_lr_mix2": 0.0}, "outer_lr_mix2"),
        (diloco.FactoredGossipMix1Mix2Coordinator, {"outer_momentum_mix2": 1.0}, "outer_momentum_mix2"),
    ],
)
def test_nextgen_configuration_boundaries(factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_compressor_fixed_degenerate_and_failed_calibration_scale(monkeypatch):
    fixed = diloco.ComputeVisibleSparseLoCoCompressor(lattice_scale=0.25, adapt_scale_per_round=False)
    assert fixed._resolve_round_scale("Z", torch.tensor([1.0, 3.0])) == 0.25
    automatic = diloco.ComputeVisibleSparseLoCoCompressor(target_bits_per_dim=2.0)
    assert automatic._resolve_round_scale("Z", torch.zeros(2)) == 0.5
    calibrate = Mock(side_effect=ValueError("rate is unavailable"))
    monkeypatch.setattr(diloco, "calibrate_scale_for_rate", calibrate)
    vector = torch.tensor([-1.0, 1.0])
    expected = float(vector.std()) / 2.0
    assert automatic._resolve_round_scale("Z", vector) == pytest.approx(expected)
    calibrate.assert_called_once_with(kind="Z", sample=vector, target_bits_per_dim=2.0)
    configured = diloco.ComputeVisibleSparseLoCoCompressor(lattice_scale=0.75)
    assert configured._resolve_round_scale("Z", vector) == 0.75


def test_lattice_entry_point_delegates_centroid_and_rejects_unknown_codec(monkeypatch):
    compressor = diloco.ComputeVisibleSparseLoCoCompressor()
    delta = torch.tensor([1.0])
    expected = (delta.clone(), {"codec": "centroid"})
    centroid = Mock(return_value=expected)
    monkeypatch.setattr(compressor, "compress_and_reconstruct", centroid)
    assert compressor.compress_with_lattice_codec(delta) is expected
    centroid.assert_called_once_with(delta, reference_param=None, _force_centroid=True)
    with pytest.raises(ValueError, match="Unsupported lattice codec"):
        compressor.compress_with_lattice_codec(delta, codec="unknown")


def test_lattice_codec_reuses_quantizer_and_updates_scale_without_reference():
    compressor = diloco.ComputeVisibleSparseLoCoCompressor(
        codec="Z", lattice_scale=1.0, adapt_scale_per_round=False
    )
    delta = torch.tensor([1.0, -2.0, 3.0, -4.0, 1.0, -2.0, 3.0, -4.0])
    first, report = compressor.compress_with_lattice_codec(delta, fragment_name="w")
    quantizer = compressor._lattice_quantizers["Z::w"]
    torch.testing.assert_close(first, delta)
    assert report["compute_invisible_fraction"] == 0.0
    compressor.lattice_scale = 0.5
    second, report = compressor.compress_with_lattice_codec(delta, fragment_name="w")
    assert compressor._lattice_quantizers["Z::w"] is quantizer
    assert quantizer.scale == 0.5
    torch.testing.assert_close(second, delta)
    assert report["adapted_scale"] == 0.5
    assert len(compressor.fragment_rate_history["w"]) == 2


@pytest.mark.parametrize("broken_velocity", [None, torch.ones(1)])
def test_flat_bucket_reset_recreates_missing_or_wrong_size_velocity(broken_velocity):
    bucket = diloco.CPUOffloadedFlatBucketOuterState({"w": torch.tensor([3.0, 4.0])})
    bucket.flat_velocity_cpu = broken_velocity
    bucket.reset_velocity()
    torch.testing.assert_close(bucket.flat_velocity_cpu, torch.zeros(2))
    torch.testing.assert_close(bucket.flat_master_cpu, torch.tensor([3.0, 4.0]))


def test_flat_bucket_rejects_bad_restart_then_initializes_velocity_for_nesterov():
    bucket = diloco.CPUOffloadedFlatBucketOuterState({"w": torch.tensor([10.0, 20.0])})
    gradients = {"w": torch.tensor([2.0, 4.0])}
    with pytest.raises(ValueError, match="momentum_restart_period"):
        bucket.outer_nesterov_step(gradients, momentum_restart_period=0)
    assert bucket.outer_step_count == 0
    torch.testing.assert_close(bucket.flat_master_cpu, torch.tensor([10.0, 20.0]))
    bucket.flat_velocity_cpu = None
    updated, velocity = bucket.outer_nesterov_step(gradients, lr=0.5, momentum=0.5)
    torch.testing.assert_close(updated["w"], torch.tensor([8.5, 17.0]))
    torch.testing.assert_close(velocity["w"], torch.tensor([2.0, 4.0]))
    assert bucket.outer_step_count == 1


def test_factored_gossip_handles_no_disagreement_and_missing_fragments():
    coordinator = diloco.FactoredGossipMix1Mix2Coordinator(alpha_mix1=0.5)
    assert coordinator.jensen_shannon_router_disagreement([]) == 0.0
    assert coordinator.jensen_shannon_router_disagreement([torch.tensor([0.4, 0.6])]) == 0.0
    with pytest.raises(ValueError, match="master_params"):
        coordinator.step_factored_gossip({}, [{"w": torch.ones(1)}])
    with pytest.raises(ValueError, match="worker_param_states"):
        coordinator.step_factored_gossip({"w": torch.ones(1)}, [])
    master = {"missing": torch.tensor([7.0]), "w": torch.tensor([2.0])}
    updated, report = coordinator.step_factored_gossip(master, [{"w": torch.tensor([6.0])}])
    torch.testing.assert_close(updated["missing"], master["missing"])
    assert updated["missing"].data_ptr() != master["missing"].data_ptr()
    torch.testing.assert_close(updated["w"], torch.tensor([4.0]))
    assert report["mix1_fragment_count"] == 1
    assert report["mix2_fragment_count"] == 0


def test_prefix_probabilities_reject_missing_branch_and_nonpositive_total_weight():
    with pytest.raises(ValueError, match="one probability list per draft branch"):
        inference.prefix_trie_branch_probabilities([[1]], [])
    with pytest.raises(ValueError, match="positive total weight"):
        inference.prefix_trie_branch_probabilities([[1]], [[-1.0]])


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_prefetch_hook_rejects_non_single_sequence_before_streaming(shape):
    streamer = Mock()
    sink = []
    hook = inference.RegionalDHTSwarmRing._make_prefetch_hook(
        streamer, Mock(), ["expert"], 1, [()], [1.0], [0], 1, sink
    )
    with pytest.raises(ValueError, match="single-sequence"):
        hook(None, (), {"hidden_states": torch.zeros(shape)})
    streamer.prefetch_token_tree_lookahead.assert_not_called()
    assert sink == []


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_ring_rejects_nonfinite_target_logits_before_sampling(poison):
    class TinyLogitModel(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.value = value
            self.reset_routing_telemetry = Mock()

        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.full((*input_ids.shape, 2), self.value))

    target = TinyLogitModel(poison)
    draft = TinyLogitModel(0.0)
    with pytest.raises(inference.SwarmSecurityViolationError, match="POISONED_MOE_RING_LOGITS_DETECTED"):
        inference.RegionalDHTSwarmRing().run_speculative_inference(
            target, draft, torch.tensor([[1]]), max_new_tokens=1, draft_gamma=1
        )
    target.reset_routing_telemetry.assert_called_once_with()


def test_empty_ledger_merkle_root_is_stable_domain_hash():
    assert HiveSwarmLedger._compute_merkle_root([]) == hashlib.sha256(b"empty").hexdigest()


def test_architecture_cli_passes_artifact_path_and_prints_receipt(monkeypatch, capsys):
    verifier = Mock()
    verifier.run_all_verifications.return_value = dict(
        passed_hypotheses=9, total_hypotheses=10, elapsed_seconds=1.25,
        receipt_sha256="0123456789abcdef" + "f" * 48,
    )
    factory = Mock(return_value=verifier)
    monkeypatch.setattr(architecture, "ArchitectureHypothesisVerifier", factory)
    architecture.main()
    factory.assert_called_once_with(seed=2026)
    verifier.run_all_verifications.assert_called_once_with(
        output_path=Path(architecture.__file__).resolve().parents[3]
        / "artifacts" / "architecture_hypothesis_verification_report.json"
    )
    assert capsys.readouterr().out == (
        "[ArchitectureHypothesisVerifier] 9/10 hypotheses PASSED in 1.25s "
        "| receipt_sha256=0123456789abcdef...\n"
    )


def test_windows_udp_socket_disables_connection_reset_before_bind(monkeypatch):
    sock = Mock()
    socket_api = SimpleNamespace(socket=Mock(return_value=sock), AF_INET=2, SOCK_DGRAM=2, SIO_UDP_CONNRESET=123)
    monkeypatch.setattr(nat, "socket", socket_api)
    assert nat.udp_socket(("192.0.2.1", 1234)) is sock
    socket_api.socket.assert_called_once_with(2, 2)
    assert sock.mock_calls == [call.ioctl(123, False), call.bind(("192.0.2.1", 1234))]


def _mock_clock(monkeypatch, module):
    ticks = itertools.count()
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: next(ticks) * 0.001,
        perf_counter=Mock(side_effect=[10.0, 10.025]),
    ))


def test_stun_transaction_recovers_receive_error_and_accepts_matching_response(monkeypatch):
    txid = b"t" * 12
    request = nat.StunMessage(nat.BINDING, nat.REQUEST, txid).encode()
    response = nat.StunMessage(nat.BINDING, nat.SUCCESS, txid).encode()
    server = ("192.0.2.1", 3478)
    sock = Mock()
    sock.recvfrom.side_effect = [OSError("transient receive"), (response, server)]
    monkeypatch.setattr(nat, "select", SimpleNamespace(select=Mock(return_value=([sock], [], []))))
    _mock_clock(monkeypatch, nat)
    message, raw, rtt = nat.stun_transaction(sock, server, request, txid, timeout=1.0)
    assert message.transaction_id == txid
    assert message.cls == nat.SUCCESS
    assert raw == response
    assert rtt == pytest.approx(25.0)
    assert sock.recvfrom.call_count == 2
    sock.sendto.assert_called_once_with(request, server)


def test_hole_punch_recovers_send_receive_and_ack_failures(monkeypatch):
    peer = ("192.0.2.1", 5000)
    session = b"session"
    syn, ack = nat.PUNCH_MAGIC + b"S" + session, nat.PUNCH_MAGIC + b"A" + session
    sock = Mock()
    sock.sendto.side_effect = [OSError("probe"), None, OSError("ack"), None]
    sock.recvfrom.side_effect = [OSError("receive"), (syn, peer), (ack, peer)]
    monkeypatch.setattr(nat, "select", SimpleNamespace(select=Mock(return_value=([sock], [], []))))
    _mock_clock(monkeypatch, nat)
    result = nat.udp_hole_punch(sock, [peer], session, timeout=1.0, interval=0.0, linger=0.0)
    assert result.succeeded is True
    assert result.peer == peer
    assert result.probes_sent == 2
    assert result.rtt_ms == pytest.approx(25.0)
    assert sock.sendto.call_args_list == [call(syn, peer), call(syn, peer), call(ack, peer), call(syn, peer)]


@pytest.mark.parametrize("failure", [OSError("closed"), ValueError("invalid descriptor")])
def test_hole_punch_stops_on_select_failure_without_claiming_success(monkeypatch, failure):
    sock = Mock()
    monkeypatch.setattr(nat, "select", SimpleNamespace(select=Mock(side_effect=failure)))
    _mock_clock(monkeypatch, nat)
    result = nat.udp_hole_punch(sock, [("192.0.2.1", 5000)], b"x", timeout=1.0)
    assert result.succeeded is False
    assert result.peer is result.rtt_ms is None
    assert result.probes_sent == 1
    sock.recvfrom.assert_not_called()


def test_fault_proxy_close_continues_after_socket_close_errors():
    proxy = nat.DeterministicFaultProxy(("192.0.2.1", 3478))
    sockets = [Mock() for _ in range(3)]
    for name, sock in zip(("_client_sock", "_upstream_sock", "_pending_upstream_sock"), sockets):
        sock.close.side_effect = OSError("already closed")
        setattr(proxy, name, sock)
    thread = Mock()
    proxy._thread = thread
    proxy.close()
    assert proxy._stop.is_set()
    assert proxy._thread is None
    thread.join.assert_called_once_with(timeout=2.0)
    for name, sock in zip(("_client_sock", "_upstream_sock", "_pending_upstream_sock"), sockets):
        sock.close.assert_called_once_with()
        assert getattr(proxy, name) is None


def test_fault_proxy_without_sockets_waits_once_then_stops(monkeypatch):
    proxy = nat.DeterministicFaultProxy(("192.0.2.1", 3478))
    pause = Mock(side_effect=lambda _: proxy._stop.set())
    monkeypatch.setattr(nat, "time", SimpleNamespace(sleep=pause))
    proxy._serve()
    pause.assert_called_once_with(0.005)
    assert proxy.forwarded == 0


def test_fault_proxy_installs_rebinding_even_if_old_socket_close_fails():
    proxy = nat.DeterministicFaultProxy(("192.0.2.1", 3478))
    old, replacement = Mock(), Mock()
    old.close.side_effect = OSError("already closed")
    proxy._upstream_sock = old
    proxy._pending_upstream_sock = replacement
    proxy._apply_pending_rebind()
    assert proxy._upstream_sock is replacement
    assert proxy._pending_upstream_sock is None
    old.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_resumed_quic_waits_for_handshake_and_cancels_unused_connected_waiter(monkeypatch):
    async def exercise():
        loop = asyncio.get_running_loop()
        protocol = SimpleNamespace(
            reply=loop.create_future(), connected=loop.create_future(), handshake=None,
            handshake_latency_s=None, connect=Mock(), close=Mock(),
            wait_closed=AsyncMock(), wait_connected=AsyncMock(),
        )
        protocol.transmit = Mock(side_effect=lambda: protocol.reply.set_result(struct.pack("!I", 1)))

        async def complete_handshake(_delay):
            protocol.handshake = SimpleNamespace(session_resumed=True, early_data_accepted=True)

        sleep = AsyncMock(side_effect=complete_handshake)
        datagram_transport = Mock()
        fake_loop = SimpleNamespace(
            getaddrinfo=AsyncMock(return_value=[(2, 2, 17, "", ("192.0.2.1", 4433))]),
            create_datagram_endpoint=AsyncMock(return_value=(datagram_transport, protocol)),
        )
        monkeypatch.setattr(link, "asyncio", SimpleNamespace(
            get_running_loop=lambda: fake_loop, timeout=asyncio.timeout,
            wait_for=asyncio.wait_for, sleep=sleep,
        ))
        configuration, connection = Mock(), Mock()
        connection.get_next_available_stream_id.return_value = 0
        monkeypatch.setattr(link, "QuicConfiguration", Mock(return_value=configuration))
        monkeypatch.setattr(link, "QuicConnection", Mock(return_value=connection))
        report = await link.quic_send_frames(
            "192.0.2.1", 4433, [b"data"], "test CA", session_ticket=SimpleNamespace(is_valid=True)
        )
        assert report.delivered == 1
        assert report.session_resumed and report.early_data_accepted
        sleep.assert_awaited_once_with(0.002)
        protocol.wait_connected.assert_not_awaited()
        connection.send_stream_data.assert_called_once_with(0, struct.pack("!I", 4) + b"data", end_stream=True)
        protocol.close.assert_called_once_with()
        protocol.wait_closed.assert_awaited_once_with()
        assert protocol.connected.cancelled()
        datagram_transport.close.assert_called_once_with()

    asyncio.run(exercise())


def test_loopback_link_propagates_delivery_uncertainty_and_closes_both_servers(monkeypatch):
    quic_server, ws_server = Mock(), Mock()
    quic_server.start = AsyncMock(return_value=quic_server)
    ws_server.start = AsyncMock(return_value=ws_server)
    ws_server.close = AsyncMock()
    monkeypatch.setattr(link, "LinkIdentity", SimpleNamespace(generate=Mock(return_value=SimpleNamespace(ca_pem="CA"))))
    monkeypatch.setattr(link, "QuicHiddenStateServer", Mock(return_value=quic_server))
    monkeypatch.setattr(link, "WebSocketHiddenStateServer", Mock(return_value=ws_server))
    send = AsyncMock(side_effect=link.DeliveryUncertainError("unconfirmed"))
    fallback = AsyncMock()
    monkeypatch.setattr(link, "quic_send_frames", send)
    monkeypatch.setattr(link, "websocket_send_frames", fallback)
    with pytest.raises(link.DeliveryUncertainError, match="unconfirmed"):
        asyncio.run(link.run_local_link([b"frame"]))
    fallback.assert_not_awaited()
    quic_server.close.assert_called_once_with()
    ws_server.close.assert_awaited_once_with()


def test_continental_election_ignores_a_departed_regional_bridge():
    router = transport.ContinentalSuperRingRouter()
    node = transport.SuperRingPeerNode("live", "EU-West-Paris", 5.0, stun_verified=True)
    assert router.register_peer(node)
    router.regional_bridges = {"EU-West-Paris": "live", "US-East": "departed"}
    assert router.elect_continental_bridges() == {"EU": "live"}


@pytest.mark.parametrize("use_stun", [False, True])
def test_ice_wildcard_binding_uses_advertisable_host_and_closes_socket(monkeypatch, use_stun):
    sock = Mock()
    sock.getsockname.return_value = ("0.0.0.0", 4321)
    socket_factory = Mock(return_value=sock)
    monkeypatch.setattr(transport, "udp_socket", socket_factory)
    resolve = Mock(return_value=("192.0.2.9", 3478))
    outbound = Mock(return_value="192.0.2.5")
    binding = Mock(return_value=SimpleNamespace(mapped=("198.51.100.5", 5432)))
    monkeypatch.setattr(transport, "resolve", resolve)
    monkeypatch.setattr(transport, "outbound_interface_ip", outbound)
    monkeypatch.setattr(transport, "stun_binding", binding)
    instance = transport.AsyncQUICGeoRingTransport(bind_host="192.0.2.4")
    result = instance.negotiate_stun_ice_candidate("peer", stun_server="stun.invalid:3478" if use_stun else None)
    host = "192.0.2.5" if use_stun else "192.0.2.4"
    assert f" {host} 4321 typ host" in result["ice_candidates"][0]
    assert result["candidate_type"] == ("srflx" if use_stun else "host")
    assert result["local_port"] == 4321
    sock.close.assert_called_once_with()
    if use_stun:
        outbound.assert_called_once_with(("192.0.2.9", 3478))
        binding.assert_called_once_with(sock, "stun.invalid:3478")
        assert f"raddr {host} rport 4321" in result["ice_candidate"]
    else:
        outbound.assert_not_called()
        binding.assert_not_called()


def test_remote_send_requires_an_explicit_endpoint_before_opening_link(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(link, "RemoteHiddenStateLink", constructor)
    instance = transport.AsyncQUICGeoRingTransport()
    with pytest.raises(ValueError, match="remote_endpoint"):
        asyncio.run(instance._send_remote_frames("peer", [b"frame"], None))
    constructor.assert_not_called()


def test_full_scientific_campaign_hotplugs_only_missing_domains(swarm_store):
    transfer = scientific.SwarmKnowledgeTransferEngine(store=swarm_store, device=torch.device("cpu"))
    engine = scientific.JoinedSwarmCampaignEngine(transfer)
    domains = list(scientific.SCIENTIFIC_BENCHMARK_SUITES)
    model = SimpleNamespace(expert_domains=domains[:2], hotplug_domain_expert=Mock())
    reference = {"layer0.lora_A": torch.zeros(4, 8), "layer0.lora_B": torch.zeros(8, 4)}
    report = engine.execute_full_scientific_swarm_campaign(reference, moe_model=model, inner_steps=1)
    calls = model.hotplug_domain_expert.call_args_list
    assert [item.kwargs["domain_name"] for item in calls] == domains[2:]
    for item in calls:
        assert item.kwargs["lora_rank_r"] == 4
        assert item.kwargs["lora_alpha"] == 32
        assert all(t.dtype == torch.float16 for t in item.kwargs["lora_state_dict"].values())
    assert report["hardware_validation_status"] == "not_run"
    assert report["measurement_kind"] == "synthetic_regression"


def test_pareto_candidate_is_not_dominated_by_lower_maximization_objective():
    candidate = dict(holdout_loss=1.0, parallel_speedup_x=2.0, hive_per_gpu_hour=3.0, comm_overhead_pct=4.0)
    slower = dict(candidate, parallel_speedup_x=1.0)
    assert strategy._is_pareto_dominated(candidate, [candidate, slower]) is False
    assert strategy._is_pareto_dominated(slower, [candidate, slower]) is True


def _fault_scenario():
    return wan.FaultScenario("mock-boundary", nat.FaultProfile(), nat_rebinds=1, frames=1, frame_bytes=4)


def test_wan_quic_failed_probe_then_failed_ticket_refresh_preserves_failure(monkeypatch):
    scenario = _fault_scenario()
    server = SimpleNamespace(identity=SimpleNamespace(ca_pem="CA"))
    proxy = SimpleNamespace(client_address=("192.0.2.1", 4433), stats=lambda: {})
    ticket = object()
    send = AsyncMock(side_effect=[SimpleNamespace(new_ticket=ticket), OSError("refresh failed")])
    probe = AsyncMock(return_value=SimpleNamespace(frame_latencies_ms=[], frames_delivered=0, ok=False))
    monkeypatch.setattr(wan, "quic_send_frames", send)
    monkeypatch.setattr(wan, "quic_0rtt_resumption_probe", probe)
    report = asyncio.run(wan._run_quic_leg(scenario, wan.StrikeRegister(), server, proxy))
    assert report["ok"] is False
    assert report["error"] == "OSError"
    assert report["attempts"] == 1
    assert report["frames_sent"] == 1
    assert report["frames_delivered"] == 0
    assert send.await_count == 2
    assert send.await_args_list[0].args[3] == "CA"
    assert probe.await_args.kwargs["session_ticket"] is ticket


def test_wan_relay_discards_non_envelope_before_counting_delivery(monkeypatch):
    sender, receiver = Mock(), Mock()
    peer = ("192.0.2.1", 5000)
    valid = link.FrameEnvelope(0, 100.0, b"data").encode()
    # There is no magic marker: arbitrary text can be a valid timestamp header.
    truncated = valid[: link.ENVELOPE_HEADER.size - 1]
    with pytest.raises(link.LinkProtocolError, match="truncated frame envelope"):
        link.FrameEnvelope.decode(truncated)
    receiver.recv_from.side_effect = [(peer, truncated), (peer, valid)]
    monkeypatch.setattr(wan, "_envelope_frames", Mock(return_value=[valid]))
    monkeypatch.setattr(wan, "time", SimpleNamespace(monotonic=lambda: 0.0, time=lambda: 100.025))
    assert wan._relay_frames(sender, receiver, peer, _fault_scenario(), timeout=1.0) == pytest.approx([25.0])
    sender.send_to.assert_called_once_with(peer, valid)
    assert receiver.recv_from.call_count == 2


def test_wan_turn_reallocation_failure_stops_retrying_and_closes_clients(monkeypatch):
    alice, bob = Mock(), Mock()
    alice.allocate.side_effect = [("198.51.100.1", 6000), nat.StunError("allocation lost")]
    bob.allocate.return_value = ("198.51.100.2", 6001)
    alice.bind_channel.return_value = 0x4000
    proxies = [Mock(), Mock()]
    for proxy in proxies:
        proxy.start.return_value = proxy
        proxy.client_address = ("192.0.2.1", 5000)
        proxy.stats.return_value = {}
    monkeypatch.setattr(wan, "DeterministicFaultProxy", Mock(side_effect=proxies))
    monkeypatch.setattr(wan, "TurnClient", Mock(side_effect=[alice, bob]))
    relay = Mock(side_effect=[[2.0], [3.0], []])
    monkeypatch.setattr(wan, "_relay_frames", relay)
    endpoints = SimpleNamespace(has_turn=True, turn_host="192.0.2.1", turn_port=3478,
                                turn_username="user", turn_password="password")
    report = wan._run_turn_leg(_fault_scenario(), endpoints=endpoints)
    assert report["nat_rebind_events"] == 1
    assert report["reallocations_after_rebind"] == 0
    assert report["frames_lost_after_rebind"] == 1
    assert report["post_rebind_frames_delivered"] == 0
    assert alice.allocate.call_count == 2
    assert relay.call_count == 3
    alice.close.assert_called_once_with()
    bob.close.assert_called_once_with()
    for proxy in proxies:
        proxy.close.assert_called_once_with()


def test_wan_websocket_reports_lossy_quic_failure_and_closes_listener(monkeypatch):
    server = SimpleNamespace(url="ws://mock.invalid", start=AsyncMock(), close=AsyncMock())
    server.start.return_value = server
    server_factory = Mock(return_value=server)
    monkeypatch.setattr(wan, "WebSocketHiddenStateServer", server_factory)

    async def websocket_send(_url, frames, **_kwargs):
        for frame in frames:
            server_factory.call_args.kwargs["on_frame"](frame)
        return len(frames)

    fallback = SimpleNamespace(protocol="libp2p_websocket", fallback_reason="TimeoutError",
                               frames_sent=1, frames_delivered=1)
    links = [SimpleNamespace(send_frames=AsyncMock(return_value=fallback)),
             SimpleNamespace(send_frames=AsyncMock(side_effect=ConnectionError("lost")))]
    monkeypatch.setattr(wan, "websocket_send_frames", AsyncMock(side_effect=websocket_send))
    monkeypatch.setattr(wan, "_unused_udp_port", Mock(return_value=5001))
    monkeypatch.setattr(wan, "LinkEndpoint", Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(wan, "RemoteHiddenStateLink", Mock(side_effect=links))
    report = asyncio.run(wan._run_websocket_leg(_fault_scenario(), "CA", ("192.0.2.1", 5002)))
    assert report["ok"] is True
    assert report["frames_delivered"] == 1
    assert report["lossy_quic_path"]["error"] == "ConnectionError"
    assert report["lossy_quic_path"]["frames_delivered"] == 0
    links[1].send_frames.assert_awaited_once()
    server.close.assert_awaited_once_with()
