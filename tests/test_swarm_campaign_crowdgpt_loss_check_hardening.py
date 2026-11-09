# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and input-validation tests for campaign orchestrator, CrowdGPT bridge, and loss-check service."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from hivemind.swarm.campaign_orchestrator import (
    CampaignRequest,
    atomic_write_json,
    parse_campaign_utc,
    sign_campaign_request,
    verify_campaign_request_signature,
)
from hivemind.swarm.crowdgpt_bridge import (
    classify_gpu_precision_cohort,
    pack_crowdgpt_delta_stream,
    resolve_forming_pool_starvation,
    unpack_crowdgpt_delta_stream,
)
from hivemind.swarm.crowdgpt_loss_check import (
    LossCheckReceipt,
    receipts_agree,
    tokens_sha256,
    weights_sha256,
)
from hivemind.swarm.crowdgpt_loss_check_service import LossCheckCoordinator


def _make_campaign_request() -> CampaignRequest:
    return CampaignRequest(
        action="join",
        participant_id="node-alpha",
        participant_kind="volunteer_gpu",
        campaign_id="camp-01",
        declared_vram_mib=8192,
        availability="24x7_autonomous",
        issued_at_utc=1760000000.0,
        nonce="nonce-12345678",
    )


def test_campaign_orchestrator_sign_and_verify_validation() -> None:
    req = _make_campaign_request()
    secret = b"0123456789abcdef"
    signed = sign_campaign_request(req, secret)
    assert verify_campaign_request_signature(signed, secret) is True

    with pytest.raises(TypeError, match="request must be a CampaignRequest"):
        sign_campaign_request({"action": "join"}, secret)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="secret must be bytes of length >= 16"):
        sign_campaign_request(req, b"short")

    with pytest.raises(TypeError, match="request must be a CampaignRequest"):
        verify_campaign_request_signature("bad", secret)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="secret must be bytes of length >= 16"):
        verify_campaign_request_signature(signed, b"short")


def test_campaign_orchestrator_utc_and_atomic_write_validation(tmp_path: Path) -> None:
    ts = parse_campaign_utc("20261028T180000Z")
    assert ts > 0

    with pytest.raises(ValueError, match="stamp must be a non-empty string"):
        parse_campaign_utc("   ")

    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": True})
    assert target.exists()

    with pytest.raises(ValueError, match="path must be a non-empty Path or string"):
        atomic_write_json("   ", {"ok": True})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="payload must be a Mapping"):
        atomic_write_json(target, [1, 2, 3])  # type: ignore[arg-type]


def test_crowdgpt_bridge_pack_and_unpack_validation() -> None:
    delta = torch.tensor([0.25, -0.5, 1.0, 2.0], dtype=torch.bfloat16)
    blob = pack_crowdgpt_delta_stream({"clientId": "c1"}, delta, compresslevel=1)
    unpacked = unpack_crowdgpt_delta_stream(blob)
    assert unpacked.metadata["clientId"] == "c1"
    assert unpacked.base_delta_bf16.numel() == 4

    with pytest.raises(TypeError, match="metadata must be a Mapping"):
        pack_crowdgpt_delta_stream(["bad"], delta)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="base_delta_bf16 must be a torch.Tensor"):
        pack_crowdgpt_delta_stream({}, [1.0, 2.0])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="compresslevel must be an integer"):
        pack_crowdgpt_delta_stream({}, delta, compresslevel=10)

    with pytest.raises(ValueError, match="compressed_data must be non-empty bytes"):
        unpack_crowdgpt_delta_stream(b"")


def test_crowdgpt_bridge_cohort_and_pool_starvation_validation() -> None:
    cohort = classify_gpu_precision_cohort("nvidia", (8, 6))
    assert cohort.effective_dtype == "bf16"

    with pytest.raises(ValueError, match="vendor must be a non-empty string"):
        classify_gpu_precision_cohort("  ", (8, 0))

    with pytest.raises(ValueError, match="compute_capability must be a 2-tuple of non-negative integers"):
        classify_gpu_precision_cohort("nvidia", (8, -1))

    with pytest.raises(ValueError, match="compute_capability must be a 2-tuple of non-negative integers"):
        classify_gpu_precision_cohort("nvidia", (True, 0))  # type: ignore[arg-type]

    dec = resolve_forming_pool_starvation(
        current_members=2,
        target_pool_size=3,
        waited_seconds=200.0,
        forming_timeout_s=180.0,
    )
    assert dec.active_members == 2

    with pytest.raises(ValueError, match="current_members must be an integer"):
        resolve_forming_pool_starvation(current_members=True, waited_seconds=10.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target_pool_size must be a positive integer"):
        resolve_forming_pool_starvation(current_members=1, target_pool_size=0, waited_seconds=10.0)

    with pytest.raises(ValueError, match="waited_seconds must be a finite non-negative number"):
        resolve_forming_pool_starvation(current_members=1, waited_seconds=-1.0)

    with pytest.raises(ValueError, match="forming_timeout_s must be a positive finite number"):
        resolve_forming_pool_starvation(current_members=1, waited_seconds=1.0, forming_timeout_s=0.0)


def test_crowdgpt_loss_check_validation() -> None:
    w_digest = weights_sha256(torch.tensor([1.0, 2.0], dtype=torch.float32))
    t_digest = tokens_sha256(np.array([1, 2, 3, 4], dtype=np.uint16))
    assert len(w_digest) == 64 and len(t_digest) == 64

    with pytest.raises(ValueError, match="flat must be a non-empty torch.Tensor"):
        weights_sha256(torch.tensor([], dtype=torch.float32))

    with pytest.raises(ValueError, match="tokens must be a non-empty numpy ndarray"):
        tokens_sha256(np.array([], dtype=np.uint16))

    r1 = LossCheckReceipt(
        evaluator_id="eval-1",
        base_sha256=w_digest,
        candidate_sha256=w_digest,
        heldout_sha256=t_digest,
        window=4,
        seed=7,
        starts=(0, 4),
        losses_before=(2.1, 2.2),
        losses_after=(2.0, 2.1),
        compute_dtype="float32",
        device="cpu",
    )
    r2 = LossCheckReceipt(
        evaluator_id="eval-2",
        base_sha256=w_digest,
        candidate_sha256=w_digest,
        heldout_sha256=t_digest,
        window=4,
        seed=7,
        starts=(0, 4),
        losses_before=(2.1, 2.2),
        losses_after=(2.0, 2.1),
        compute_dtype="float32",
        device="cpu",
    )
    signed = r1.signed(b"eval-key-1")
    assert signed.verify_signature(b"eval-key-1") is True

    with pytest.raises(ValueError, match="key must be non-empty bytes"):
        r1.signed(b"")

    with pytest.raises(ValueError, match="key must be non-empty bytes"):
        signed.verify_signature(b"")

    ok, _ = receipts_agree(r1, r2)
    assert ok is True

    with pytest.raises(TypeError, match="a and b must be LossCheckReceipt instances"):
        receipts_agree(r1, "bad")  # type: ignore[arg-type]


def test_crowdgpt_loss_check_service_validation(tmp_path: Path) -> None:
    coord = LossCheckCoordinator(
        evaluator_keys={"eval-1": b"secret-eval-1"},
        server_secret=b"server-secret-key",
        state_path=tmp_path / "loss_check_state.json",
        loader=lambda _job: (None, None, None),
    )
    job = coord.open_job(
        base_url="https://example.org/base.safetensors",
        base_sha256="a" * 64,
        candidate_url="https://example.org/cand.safetensors",
        candidate_sha256="b" * 64,
        heldout_url="https://example.org/heldout.bin",
        heldout_sha256="c" * 64,
        window=64,
        n=8,
    )
    assert coord.decision(job.job_id) is None
    assert coord.next_job("eval-1") is not None

    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        coord.open_job(
            base_url="  ",
            base_sha256="a" * 64,
            candidate_url="https://example.org/cand.safetensors",
            candidate_sha256="b" * 64,
            heldout_url="https://example.org/heldout.bin",
            heldout_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match="job_id must be a non-empty string"):
        coord.decision("  ")

    with pytest.raises(ValueError, match="evaluator_id must be a non-empty string"):
        coord.next_job("")

    with pytest.raises(TypeError, match="body must be bytes or str"):
        coord.handle_post_receipt(12345)  # type: ignore[arg-type]
