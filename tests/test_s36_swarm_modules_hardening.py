# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control hardening tests for HM-S36-01 through HM-S36-06 swarm modules."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import ZKOPoIV5Int8TransitionVerifier
from hivemind.swarm.s36_02_e21_multikrum_vs_zk_crossover import (
    _measure_pairwise_l2_flop_ms_coeff,
    run_e21_multikrum_vs_zk_crossover,
)
from hivemind.swarm.s36_03_geocryptographic_neural_mesh import (
    GeocryptographicNeuralMeshRouter,
    GeoMeshExpertNode,
    VivaldiCoord3D,
    _cosine_sim,
)
from hivemind.swarm.s36_04_fuzzy_subswarm_consensus import (
    LocalizedFuzzySubSwarmConsensus,
    SubSwarmVote,
    hypergeometric_collusion_probability,
)
from hivemind.swarm.s36_05_darl_cas_shard_leasing import DARLShardLeaseCoordinator
from hivemind.swarm.s36_06_hourly_coding_assistant_orchestrator import (
    HourlyHiveMindCodingAssistantOrchestrator,
    resolve_hivemind_api_key,
)


def test_s36_01_zk_opoi_v5_int8_verifier_rejects_malformed_and_delimiter_inputs() -> None:
    for bad_secret in ("", "   ", "sec\x00ret", 123):
        with pytest.raises(ValueError, match="coordinator_secret"):
            ZKOPoIV5Int8TransitionVerifier(coordinator_secret=bad_secret)  # type: ignore[arg-type]

    for bad_norm in (True, False, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_l2_norm"):
            ZKOPoIV5Int8TransitionVerifier(max_l2_norm=bad_norm)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_linf_norm"):
            ZKOPoIV5Int8TransitionVerifier(max_linf_norm=bad_norm)  # type: ignore[arg-type]

    for bad_scale_q in (True, False, 0, -5, 127.0):
        with pytest.raises(ValueError, match="scale_q"):
            ZKOPoIV5Int8TransitionVerifier(scale_q=bad_scale_q)  # type: ignore[arg-type]

    verifier = ZKOPoIV5Int8TransitionVerifier(coordinator_secret="shared-secret-key")
    with pytest.raises(ValueError, match="quant_bits"):
        verifier.quantize_to_int8([0.1, 0.2], quant_bits=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="values"):
        verifier.quantize_to_int8([])
    with pytest.raises(ValueError, match="boolean"):
        verifier.quantize_to_int8([True, 0.2])  # type: ignore[list-item]

    theta0 = [0.1, -0.2, 0.3, 0.4]
    delta = [0.05, -0.05, 0.08, -0.06]
    for bad_id in ("", "  ", "worker|inject", "worker\n1"):
        with pytest.raises(ValueError):
            verifier.generate_proof(
                worker_id=bad_id,
                workunit_id="wu-01",
                shard_sha256="shard-01",
                theta0=theta0,
                delta_weights=delta,
            )
        with pytest.raises(ValueError):
            verifier.generate_proof(
                worker_id="w-01",
                workunit_id=bad_id,
                shard_sha256="shard-01",
                theta0=theta0,
                delta_weights=delta,
            )
        with pytest.raises(ValueError):
            verifier.generate_proof(
                worker_id="w-01",
                workunit_id="wu-01",
                shard_sha256=bad_id,
                theta0=theta0,
                delta_weights=delta,
            )

    with pytest.raises(ValueError, match="inner_steps_h"):
        verifier.generate_proof(
            worker_id="w-01",
            workunit_id="wu-01",
            shard_sha256="shard-01",
            theta0=theta0,
            delta_weights=delta,
            inner_steps_h=True,  # type: ignore[arg-type]
        )

    receipt = verifier.generate_proof(
        worker_id="w-01",
        workunit_id="wu-01",
        shard_sha256="shard-01",
        theta0=theta0,
        delta_weights=delta,
        inner_steps_h=10,
    )
    assert (
        verifier.verify_proof(
            replace(receipt, worker_id="bad|id"),
            expected_shard_sha256="shard-01",
            expected_theta0_sha256=receipt.theta0_sha256,
        ).rejection_reason
        == "invalid_identifier"
    )
    assert (
        verifier.verify_proof(
            replace(receipt, quant_bits=6),
            expected_shard_sha256="shard-01",
            expected_theta0_sha256=receipt.theta0_sha256,
        ).rejection_reason
        == "quant_bits_invalid"
    )
    assert (
        verifier.verify_proof(
            replace(receipt, scale_q=63),
            expected_shard_sha256="shard-01",
            expected_theta0_sha256=receipt.theta0_sha256,
        ).rejection_reason
        == "scale_q_mismatch"
    )
    assert (
        verifier.verify_proof(
            replace(receipt, quantized_sketch_int8=()),
            expected_shard_sha256="shard-01",
            expected_theta0_sha256=receipt.theta0_sha256,
        ).rejection_reason
        == "sketch_invalid"
    )
    assert (
        verifier.verify_proof(
            replace(receipt, commitment_hmac="é" * 64),
            expected_shard_sha256="shard-01",
            expected_theta0_sha256=receipt.theta0_sha256,
        ).rejection_reason
        == "commitment_hmac_invalid"
    )


def test_s36_02_e21_crossover_rejects_malformed_parameters() -> None:
    with pytest.raises(ValueError, match="k_calib"):
        _measure_pairwise_l2_flop_ms_coeff(k_calib=1)
    with pytest.raises(ValueError, match="d_calib"):
        _measure_pairwise_l2_flop_ms_coeff(d_calib=0)

    for bad_sizes in ([], "100", [1], [True, 100]):
        with pytest.raises(ValueError, match="swarm_sizes"):
            run_e21_multikrum_vs_zk_crossover(swarm_sizes=bad_sizes)  # type: ignore[arg-type]

    for bad_dim in (True, 0, -1, 128.0):
        with pytest.raises(ValueError, match="lora_dim_d"):
            run_e21_multikrum_vs_zk_crossover(lora_dim_d=bad_dim)  # type: ignore[arg-type]

    for bad_replay in (True, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gpu_step_replay_ms_per_wu"):
            run_e21_multikrum_vs_zk_crossover(gpu_step_replay_ms_per_wu=bad_replay)  # type: ignore[arg-type]

    for bad_ratio in (True, False, 0.0, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError, match="spot_check_ratio"):
            run_e21_multikrum_vs_zk_crossover(spot_check_ratio=bad_ratio)  # type: ignore[arg-type]

    for bad_ov in (True, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="single_zk_verify_ms_override"):
            run_e21_multikrum_vs_zk_crossover(single_zk_verify_ms_override=bad_ov)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="flop_ms_coeff_override"):
            run_e21_multikrum_vs_zk_crossover(flop_ms_coeff_override=bad_ov)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="forged_proof_control"):
        run_e21_multikrum_vs_zk_crossover(forged_proof_control=1)  # type: ignore[arg-type]


def test_s36_03_geocryptographic_mesh_rejects_malformed_coordinates_and_nodes() -> None:
    for bad_c in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            VivaldiCoord3D(bad_c, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="height_ms"):
        VivaldiCoord3D(1.0, 2.0, 3.0, height_ms=-0.1)

    c = VivaldiCoord3D(1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="VivaldiCoord3D"):
        c.estimate_rtt_ms("bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        GeoMeshExpertNode(
            node_id="",
            region="eu-west",
            coord=c,
            expert_domain="math",
            semantic_vector=(1.0, 0.0),
            adapter_sha256="a" * 64,
        )
    with pytest.raises(ValueError):
        GeoMeshExpertNode(
            node_id="n1",
            region="eu-west",
            coord=c,
            expert_domain="math",
            semantic_vector=(True, 0.0),  # type: ignore[arg-type]
            adapter_sha256="a" * 64,
        )

    with pytest.raises(ValueError):
        _cosine_sim([True, 0.5], [0.5, 0.5])  # type: ignore[list-item]

    router = GeocryptographicNeuralMeshRouter()
    with pytest.raises(ValueError, match="GeoMeshExpertNode"):
        router.register_node("not-a-node")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="client_coord"):
        router.route_query(
            client_coord="bad",  # type: ignore[arg-type]
            expert_domain="math",
            query_semantic_vector=(1.0, 0.0),
        )


def test_s36_04_fuzzy_subswarm_consensus_rejects_malformed_inputs() -> None:
    for bad_args in (
        (True, 5, 15, 10),
        (100, -1, 15, 10),
        (100, 101, 15, 10),
        (100, 5, 15, 0),
        (100, 5, 15, 16),
    ):
        with pytest.raises(ValueError):
            hypergeometric_collusion_probability(*bad_args)  # type: ignore[arg-type]

    for bad_q in (True, False, 0.0, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError, match="min_fuzzy_quorum"):
            LocalizedFuzzySubSwarmConsensus(min_fuzzy_quorum=bad_q)  # type: ignore[arg-type]

    engine = LocalizedFuzzySubSwarmConsensus()
    valid_votes = [
        SubSwarmVote(node_id=f"n-{i:02d}", answer_digest="ans-1", semantic_confidence=0.9, observer_reputation=0.9)
        for i in range(15)
    ]
    with pytest.raises(ValueError, match="total_network_nodes_n"):
        engine.evaluate_subswarm(valid_votes, total_network_nodes_n=10)
    with pytest.raises(ValueError, match="assumed_byzantine_fraction"):
        engine.evaluate_subswarm(valid_votes, assumed_byzantine_fraction=1.5)

    bad_vote_bool = list(valid_votes)
    bad_vote_bool[0] = SubSwarmVote(
        node_id="n-00",
        answer_digest="ans-1",
        semantic_confidence=True,  # type: ignore[arg-type]
        observer_reputation=0.9,
    )
    with pytest.raises(ValueError, match="semantic_confidence"):
        engine.evaluate_subswarm(bad_vote_bool)


def test_s36_05_and_s36_06_darl_and_orchestrator_reject_malformed_inputs() -> None:
    for bad_shards in ("s1s2", ["s1", ""], ["s1", "s\n2"]):
        with pytest.raises(ValueError, match="shard_ids"):
            DARLShardLeaseCoordinator(bad_shards)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        DARLShardLeaseCoordinator(["s1"], alpha=True)  # type: ignore[arg-type]

    coord = DARLShardLeaseCoordinator(["s1", "s2"])
    with pytest.raises(ValueError, match="worker_id"):
        coord.grant_lease("bad\nworker", now_s=0.0)
    with pytest.raises(ValueError, match="now_s"):
        coord.grant_lease("w1", now_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="t_step_s"):
        coord.grant_lease("w1", now_s=0.0, t_step_s=0.0)
    with pytest.raises(ValueError, match="rtt_s"):
        coord.grant_lease("w1", now_s=0.0, rtt_s=-0.1)
    with pytest.raises(ValueError, match="lease_id"):
        coord.commit_lease(0, "w1", now_s=1.0)
    with pytest.raises(ValueError, match="lease_id"):
        coord.renew_heartbeat(True, "w1", now_s=1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="explicit_key"):
        resolve_hivemind_api_key("bad\nkey")
    with pytest.raises(ValueError, match="base_url"):
        HourlyHiveMindCodingAssistantOrchestrator(base_url="ftp://bad-url")
    orch = HourlyHiveMindCodingAssistantOrchestrator(api_key=None)
    with pytest.raises(ValueError, match="call_live_openrouter"):
        orch.run_hourly_improvement_cycle(call_live_openrouter=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="task_assertions"):
        orch.run_hourly_improvement_cycle(task_assertions={"HM-S36-01": "not-callable"})  # type: ignore[dict-item]
