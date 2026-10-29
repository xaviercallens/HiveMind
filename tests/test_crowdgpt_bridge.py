"""Tests for the CrowdGPT x HiveMind DiLoCo, Multi-Krum & Loss-Ratchet bridge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.swarm.crowdgpt_bridge import (
    LossRatchetGate,
    check_weight_rms_collapse_guard,
    classify_gpu_precision_cohort,
    evaluate_client_resilience_policy,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
    resolve_forming_pool_starvation,
    unpack_crowdgpt_delta_stream,
)


def test_crowdgpt_delta_stream_pack_and_unpack_roundtrip() -> None:
    meta = {
        "taskId": "round-42-slot-3",
        "loss": 2.345,
        "localSteps": 420,
        "tokensProcessed": 860160,
        "loraRank": 0,
    }
    delta = torch.tensor([0.125, -0.5, 1.25, 0.0, -2.0, 3.5], dtype=torch.bfloat16)
    si = np.array([3, 19], dtype=np.uint32)
    sv = np.ones((2 * 8,), dtype=np.uint16)

    compressed = pack_crowdgpt_delta_stream(meta, delta, si, sv)
    unpacked = unpack_crowdgpt_delta_stream(compressed, engram_row_dim=8)

    assert unpacked.metadata["taskId"] == "round-42-slot-3"
    assert unpacked.metadata["tokensProcessed"] == 860160
    assert unpacked.metadata["hasEngram"] is True
    assert unpacked.metadata["engramSparseCount"] == 2
    assert torch.equal(unpacked.base_delta_bf16, delta)
    assert np.array_equal(unpacked.engram_indices, si)
    assert np.array_equal(unpacked.engram_values_uint16, sv)
    assert len(unpacked.payload_sha256) == 64


def test_multi_krum_rejects_poisoned_crowdgpt_delta() -> None:
    torch.manual_seed(7)
    true_direction = torch.linspace(-0.05, 0.05, steps=2048, dtype=torch.float32)
    honest = [
        (true_direction + 0.001 * torch.randn_like(true_direction)).to(torch.bfloat16)
        for _ in range(4)
    ]
    byzantine = (-50.0 * true_direction).to(torch.bfloat16)
    deltas = honest + [byzantine]
    tokens = [100_000, 120_000, 110_000, 90_000, 5_000_000]

    res = multi_krum_select_and_aggregate(
        deltas,
        f_byzantine=1,
        m_select=4,
        token_weights=tokens,
        chunk_elements=512,
    )

    assert res.selected_indices == [0, 1, 2, 3]
    assert res.rejected_indices == [4]
    assert res.effective_tokens == 420_000
    cos = torch.nn.functional.cosine_similarity(
        res.aggregated_delta_bf16.float().unsqueeze(0),
        true_direction.unsqueeze(0),
    ).item()
    assert cos > 0.999


def test_loss_ratchet_gate_accepts_improvement_and_holds_regression() -> None:
    gate = LossRatchetGate(max_regression_tolerance=0.0)
    delta = torch.tensor([0.01, -0.02, 0.03], dtype=torch.bfloat16)

    ok = gate.evaluate(loss_before=2.2705, loss_after=2.1380, candidate_delta_bf16=delta)
    assert ok.accepted is True
    assert ok.delta_loss == pytest.approx(2.1380 - 2.2705)
    assert len(ok.checkpoint_sha256) == 64

    held = gate.evaluate(loss_before=2.1380, loss_after=2.1447, candidate_delta_bf16=delta)
    assert held.accepted is False
    assert "HELD_PREVIOUS_HEAD" in held.reason


def test_crowdswarm_v2_weight_rms_collapse_and_step282_halving_guard() -> None:
    """Verify Vxtzq/CrowdGPT#48 Step-282 weight halving (0.5 * old_weights) and RMS collapse are rejected."""
    torch.manual_seed(48)
    old_w = torch.randn(4096, dtype=torch.float32) * 0.0387

    # Case 1: Exact Step-282 bug where averaging a zeroed participant halves trunk weights
    halved_w = 0.5 * old_w + 1e-5 * torch.randn_like(old_w)
    res_halved = check_weight_rms_collapse_guard(old_w, halved_w)
    assert res_halved.accepted is False
    assert "REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING" in res_halved.reason
    assert res_halved.rms_ratio == pytest.approx(0.5, abs=0.01)

    # Case 2: Exploding weights (rms_ratio > 1.25)
    exploded_w = 1.6 * old_w
    res_exploded = check_weight_rms_collapse_guard(old_w, exploded_w)
    assert res_exploded.accepted is False
    assert "REJECTED_RMS_EXPLOSION" in res_exploded.reason

    # Case 3: Healthy outer DiLoCo update within [0.80, 1.25]
    healthy_w = 1.01 * old_w + 0.0005 * torch.randn_like(old_w)
    res_ok = check_weight_rms_collapse_guard(old_w, healthy_w)
    assert res_ok.accepted is True
    assert "ACCEPTED_RMS_HEALTHY" in res_ok.reason


def test_crowdswarm_v2_precision_cohort_and_forming_pool_starvation_fallback() -> None:
    """Verify Vxtzq/CrowdGPT#44/#51 pre-Ampere fp32 enforcement and forming-pool timeout fallback."""
    # Pre-Ampere Turing RTX 2070 / Tesla T4 (sm_75) requesting bf16 must be forced to fp32
    turing = classify_gpu_precision_cohort("nvidia", (7, 5), requested_dtype="bf16", optimizer_family="adamw")
    assert turing.effective_dtype == "fp32"
    assert turing.pre_ampere_fp32_enforced is True
    assert turing.cohort_key == "nvidia:sm75_pre_ampere:fp32:adamw"

    # Ampere RTX 3090 / A100 (sm_80+) stays in native bf16 cohort
    ampere = classify_gpu_precision_cohort("nvidia", (8, 6), requested_dtype="bf16", optimizer_family="adamw")
    assert ampere.effective_dtype == "bf16"
    assert ampere.pre_ampere_fp32_enforced is False
    assert ampere.cohort_key != turing.cohort_key

    # Forming pool at 1/3 before timeout waits; after timeout falls back to solo Multi-Krum + Loss-Ratchet
    wait_dec = resolve_forming_pool_starvation(
        current_members=1, target_pool_size=3, waited_seconds=45.0, forming_timeout_s=180.0, cohort_key=turing.cohort_key
    )
    assert wait_dec.mode == "FORMING_WAIT"

    solo_dec = resolve_forming_pool_starvation(
        current_members=1, target_pool_size=3, waited_seconds=185.0, forming_timeout_s=180.0, cohort_key=turing.cohort_key
    )
    assert solo_dec.mode == "SOLO_LOSS_RATCHET_FALLBACK"
    assert solo_dec.requires_loss_ratchet is True
    assert solo_dec.requires_multi_krum is True


def test_crowdswarm_v2_client_oom_and_502_503_resilience_policy() -> None:
    """Verify Vxtzq/CrowdGPT#52 OOM-at-BS1 circuit breaker and #50 HTTP 502/503 backoff."""
    oom_bs4 = evaluate_client_resilience_policy(batch_size=4, encountered_oom=True)
    assert oom_bs4["action"] == "HALVE_BATCH_SIZE_AND_RETRY_LOCAL"
    assert oom_bs4["next_batch_size"] == 2
    assert oom_bs4["redownload_checkpoint"] is False

    oom_bs1 = evaluate_client_resilience_policy(batch_size=1, encountered_oom=True)
    assert oom_bs1["action"] == "ABORT_INSUFFICIENT_VRAM"
    assert oom_bs1["redownload_checkpoint"] is False

    retry_503 = evaluate_client_resilience_policy(batch_size=2, http_status=503, retry_attempt=2)
    assert retry_503["action"] == "RETRY_FETCH_TASK_WITH_BACKOFF"
    assert retry_503["backoff_s"] > 8.0
