# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and negative-control tests for IP-126 swarm fabric, SwaT, DCT, and update gate hardening."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.swarm.swarm_fabric import (
    GeoRingPeer,
    PinnedRAMExpertDMAStreamer,
    RegionalGeoRingTransport,
)
from hivemind.swarm.swat_telepathy_bridge import (
    audit_latent_kv_safety,
    compute_virtual_positions,
)
from hivemind.swarm.top1pct_dct_ef21_compressor import _ortho_dct1d, _ortho_idct1d
from hivemind.swarm.verified_update_gate import (
    PASS,
    HeldOutTask,
    UpdatePolicy,
    decide,
    evaluate_tasks,
)


def test_georing_peer_and_xor_distance_reject_invalid_inputs() -> None:
    peer = GeoRingPeer(peer_id="peer-eu-1", region="EU-West-Paris", rtt_ms=8.5)
    assert len(peer.kademlia_id_hex) == 40
    assert RegionalGeoRingTransport.xor_distance("aa", "ff") == (0xAA ^ 0xFF)

    with pytest.raises(ValueError, match="peer_id"):
        GeoRingPeer(peer_id="  ", region="EU-West-Paris", rtt_ms=8.5)
    with pytest.raises(ValueError, match="region"):
        GeoRingPeer(peer_id="peer-eu-1", region="", rtt_ms=8.5)
    with pytest.raises(ValueError, match="rtt_ms"):
        GeoRingPeer(peer_id="peer-eu-1", region="EU-West-Paris", rtt_ms=-1.0)
    with pytest.raises(ValueError, match="rtt_ms"):
        GeoRingPeer(peer_id="peer-eu-1", region="EU-West-Paris", rtt_ms=float("nan"))
    with pytest.raises(ValueError, match="id_hex_a"):
        RegionalGeoRingTransport.xor_distance("", "ff")


def test_pinned_ram_dma_streamer_prefetch_rejects_invalid_args() -> None:
    streamer = PinnedRAMExpertDMAStreamer()
    streamer.stage_expert_in_pinned_ram(
        "exp_code",
        {"lora_A": torch.ones((4, 8), dtype=torch.float32)},
    )
    out = streamer.async_dma_prefetch_to_vram("exp_code", simulated_attention_compute_ms=1.5, layer_idx=0)
    assert out["expert_name"] == "exp_code"

    with pytest.raises(ValueError, match="simulated_attention_compute_ms"):
        streamer.async_dma_prefetch_to_vram("exp_code", simulated_attention_compute_ms=-1.0)
    with pytest.raises(ValueError, match="simulated_attention_compute_ms"):
        streamer.async_dma_prefetch_to_vram("exp_code", simulated_attention_compute_ms=float("nan"))
    with pytest.raises(ValueError, match="layer_idx"):
        streamer.async_dma_prefetch_to_vram("exp_code", layer_idx=-1)


def test_swat_compute_virtual_positions_and_audit_latent_kv_safety_validation() -> None:
    pos = compute_virtual_positions([(0, 4), (4, 8)], [(0, 4), (4, 8)])
    assert pos.tolist() == [0.0, 1.0]

    with pytest.raises(ValueError, match="sender_offsets"):
        compute_virtual_positions([(4, 2)], [(0, 4)])
    with pytest.raises(ValueError, match="sender_offsets"):
        compute_virtual_positions("bad", [(0, 4)])  # type: ignore[arg-type]

    kv = np.ones((1, 4, 2, 2, 2, 8), dtype=np.float64)
    key_rms = np.ones((2, 2), dtype=np.float64)
    val_rms = np.ones((2, 2), dtype=np.float64)
    res = audit_latent_kv_safety(kv, key_rms, val_rms)
    assert res["accepted"] is True

    with pytest.raises(ValueError, match="max_rms_ratio"):
        audit_latent_kv_safety(kv, key_rms, val_rms, max_rms_ratio=float("nan"))
    with pytest.raises(ValueError, match="rms ratio bounds"):
        audit_latent_kv_safety(kv, key_rms, val_rms, max_rms_ratio=0.5, min_rms_ratio=0.8)
    with pytest.raises(ValueError, match="min_refusal_cosine"):
        audit_latent_kv_safety(kv, key_rms, val_rms, min_refusal_cosine=1.5)
    with pytest.raises(ValueError, match="pred_kv"):
        audit_latent_kv_safety(np.ones((2, 2)), key_rms, val_rms)


def test_ortho_dct_and_idct_reject_invalid_inputs() -> None:
    vec = [1.0, -0.5, 0.25, 0.75]
    spec = _ortho_dct1d(vec)
    recon = _ortho_idct1d(spec)
    assert pytest.approx(recon, rel=1e-6, abs=1e-6) == vec

    with pytest.raises(ValueError, match="vec"):
        _ortho_dct1d("1234")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vec elements"):
        _ortho_dct1d([1.0, float("nan")])
    with pytest.raises(ValueError, match="coeffs elements"):
        _ortho_idct1d([1.0, True])  # type: ignore[list-item]


def test_verified_update_gate_decide_and_evaluate_tasks_validation() -> None:
    outcomes = {f"t{i}": PASS for i in range(5)}
    receipt = decide(outcomes, outcomes, UpdatePolicy(min_n=5))
    assert receipt.accepted is True

    with pytest.raises(ValueError, match="Mapping"):
        decide([], outcomes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="UpdatePolicy"):
        decide(outcomes, outcomes, policy="bad")  # type: ignore[arg-type]

    task = HeldOutTask(key="t1", prompt="def f():", tests=["assert f() == 1"])
    with pytest.raises(ValueError, match="HeldOutTask"):
        evaluate_tasks(["not-a-task"], lambda t: "def f(): return 1", lambda c, ts, s: None)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="callable"):
        evaluate_tasks([task], "not-callable", lambda c, ts, s: None)  # type: ignore[arg-type]
