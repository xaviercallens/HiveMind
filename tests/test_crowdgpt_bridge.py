"""Tests for the CrowdGPT x HiveMind DiLoCo, Multi-Krum & Loss-Ratchet bridge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.swarm.crowdgpt_bridge import (
    LossRatchetGate,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
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
