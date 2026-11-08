# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 68: Negative-control and boundary tests for canary_probe, crowdgpt_loss_check, crowdgpt_loss_check_service, and crowdgpt_bridge."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.swarm.crowdgpt_bridge import (
    LossRatchetGate,
    check_weight_rms_collapse_guard,
    chunked_pairwise_sq_l2,
    evaluate_client_resilience_policy,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
    unpack_crowdgpt_delta_stream,
)
from hivemind.swarm.crowdgpt_loss_check import (
    LossCheckReceipt,
    detection_probability,
    heldout_windows,
    receipts_agree,
    spot_check_picks,
)
from hivemind.swarm.crowdgpt_loss_check_service import LossCheckCoordinator
from hivemind.verification.canary_probe import (
    NodeCanaryState,
    run_planted_canary_evaluation,
    wilson_confidence_interval,
)


def test_canary_probe_boundary_validation() -> None:
    with pytest.raises(ValueError, match="integers"):
        wilson_confidence_interval(True, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integers"):
        wilson_confidence_interval(5, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="z must be"):
        wilson_confidence_interval(5, 10, z=0.0)
    with pytest.raises(ValueError, match="z must be"):
        wilson_confidence_interval(5, 10, z=True)  # type: ignore[arg-type]

    low, high = wilson_confidence_interval(8, 10)
    assert 0.0 < low < high <= 1.0

    with pytest.raises(ValueError, match="node_id"):
        NodeCanaryState(node_id="   ", planted_role="honest")
    with pytest.raises(ValueError, match="planted_role"):
        NodeCanaryState(node_id="n1", planted_role="   ")
    with pytest.raises(ValueError, match="successes"):
        NodeCanaryState(node_id="n1", planted_role="honest", successes=5, total=3)
    with pytest.raises(ValueError, match="min_trials"):
        NodeCanaryState(node_id="n1", planted_role="honest", min_trials=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wilson_lb_threshold"):
        NodeCanaryState(node_id="n1", planted_role="honest", wilson_lb_threshold=1.0)

    with pytest.raises(ValueError, match="integers"):
        run_planted_canary_evaluation(num_honest_nodes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="honest_pass_prob"):
        run_planted_canary_evaluation(honest_pass_prob=0.0)
    with pytest.raises(ValueError, match="wilson_lb_threshold"):
        run_planted_canary_evaluation(wilson_lb_threshold=True)  # type: ignore[arg-type]


def test_crowdgpt_loss_check_and_service_boundary_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integers"):
        heldout_windows(True, 4, 2, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integers"):
        heldout_windows(100, True, 2, 0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="evaluator_id"):
        LossCheckReceipt(
            evaluator_id="   ",
            base_sha256="a" * 64,
            candidate_sha256="b" * 64,
            heldout_sha256="c" * 64,
            window=8,
            seed=1,
            starts=(0,),
            losses_before=(2.0,),
            losses_after=(1.9,),
            compute_dtype="float32",
            device="cpu",
        )
    with pytest.raises(ValueError, match="window"):
        LossCheckReceipt(
            evaluator_id="ev1",
            base_sha256="a" * 64,
            candidate_sha256="b" * 64,
            heldout_sha256="c" * 64,
            window=True,  # type: ignore[arg-type]
            seed=1,
            starts=(0,),
            losses_before=(2.0,),
            losses_after=(1.9,),
            compute_dtype="float32",
            device="cpu",
        )

    with pytest.raises(ValueError, match="commitment"):
        spot_check_picks("   ", b"secret", 8, 2)
    with pytest.raises(ValueError, match="server_secret"):
        spot_check_picks("abc", b"", 8, 2)
    with pytest.raises(ValueError, match="n_windows"):
        spot_check_picks("abc", b"secret", True, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be"):
        spot_check_picks("abc", b"secret", 8, -1)

    with pytest.raises(ValueError, match="integers"):
        detection_probability(True, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integers"):
        detection_probability(4, 5, 1)

    r1 = LossCheckReceipt(
        evaluator_id="ev1",
        base_sha256="a" * 64,
        candidate_sha256="b" * 64,
        heldout_sha256="c" * 64,
        window=8,
        seed=1,
        starts=(0,),
        losses_before=(2.0,),
        losses_after=(1.9,),
        compute_dtype="float32",
        device="cpu",
    )
    r2 = LossCheckReceipt(
        evaluator_id="ev2",
        base_sha256="a" * 64,
        candidate_sha256="b" * 64,
        heldout_sha256="c" * 64,
        window=8,
        seed=1,
        starts=(0,),
        losses_before=(2.0,),
        losses_after=(1.9,),
        compute_dtype="float32",
        device="cpu",
    )
    with pytest.raises(ValueError, match="tol"):
        receipts_agree(r1, r2, tol=True)  # type: ignore[arg-type]

    loader = lambda _job: (None, None, None)
    with pytest.raises(ValueError, match="required_receipts"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k1"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=loader,
            required_receipts=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="server_secret"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k1"},
            server_secret=b"",
            state_path=tmp_path / "s.json",
            loader=loader,
        )
    with pytest.raises(ValueError, match="tol"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k1"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=loader,
            tol=-0.1,
        )
    with pytest.raises(ValueError, match="job_ttl_s"):
        LossCheckCoordinator(
            evaluator_keys={"ev1": b"k1"},
            server_secret=b"s",
            state_path=tmp_path / "s.json",
            loader=loader,
            job_ttl_s=0.0,
        )

    coord = LossCheckCoordinator(
        evaluator_keys={"ev1": b"k1"},
        server_secret=b"s",
        state_path=tmp_path / "s.json",
        loader=loader,
    )
    with pytest.raises(ValueError, match="window"):
        coord.open_job(
            base_url="u1",
            base_sha256="a" * 64,
            candidate_url="u2",
            candidate_sha256="b" * 64,
            heldout_url="u3",
            heldout_sha256="c" * 64,
            window=1,
        )


def test_crowdgpt_bridge_boundary_validation() -> None:
    packed = pack_crowdgpt_delta_stream({"step": 1}, torch.ones(4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="engram_row_dim"):
        unpack_crowdgpt_delta_stream(packed, engram_row_dim=0)
    with pytest.raises(ValueError, match="engram_row_dim"):
        unpack_crowdgpt_delta_stream(packed, engram_row_dim=True)  # type: ignore[arg-type]

    d1 = torch.tensor([1.0, 2.0])
    d2 = torch.tensor([2.0, 3.0])
    with pytest.raises(ValueError, match="chunk_elements"):
        chunked_pairwise_sq_l2([d1, d2], chunk_elements=0)
    with pytest.raises(ValueError, match="chunk_elements"):
        chunked_pairwise_sq_l2([d1, d2], chunk_elements=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="token_weights"):
        multi_krum_select_and_aggregate([d1, d2], token_weights=[True, 10])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="max_regression_tolerance"):
        LossRatchetGate(max_regression_tolerance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_regression_tolerance"):
        LossRatchetGate(max_regression_tolerance=float("nan"))

    with pytest.raises(ValueError, match="min_rms_ratio"):
        check_weight_rms_collapse_guard(d1, d2, min_rms_ratio=0.0)
    with pytest.raises(ValueError, match="min_rms_ratio"):
        check_weight_rms_collapse_guard(d1, d2, min_rms_ratio=1.5, max_rms_ratio=1.2)

    with pytest.raises(ValueError, match="batch_size"):
        evaluate_client_resilience_policy(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_client_resilience_policy(batch_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retry_attempt"):
        evaluate_client_resilience_policy(batch_size=2, retry_attempt=-1)
    with pytest.raises(ValueError, match="base_backoff_s"):
        evaluate_client_resilience_policy(batch_size=2, base_backoff_s=10.0, max_backoff_s=5.0)
