# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 160 hardening tests for QUIC link/transport, SwarmFabric packets/Merkle/batch openings, and SwaT bridge."""

from __future__ import annotations

import pytest
import torch

from hivemind.swarm.quic_link import LinkIdentity, Quic0RttProbe
from hivemind.swarm.quic_transport import CongestionWindowTelemetry, RegionalBridgePeer
from hivemind.swarm.swarm_fabric import (
    CompressedHiddenStatePacket,
    MerkleInclusionProof,
    SpeculativeBatchCommitment,
    SpeculativeBatchOpening,
)
from hivemind.swarm.swat_telepathy_bridge import WakeToSleepConsolidationBridge


def test_quic_link_identity_and_probe_invariants() -> None:
    ident = LinkIdentity.generate()
    assert "BEGIN CERTIFICATE" in ident.ca_pem

    with pytest.raises(TypeError, match="ca_certificate"):
        LinkIdentity(ca_certificate="bad", certificate=ident.certificate, private_key=ident.private_key)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="private_key"):
        LinkIdentity(ca_certificate=ident.ca_certificate, certificate=ident.certificate, private_key="bad")  # type: ignore[arg-type]

    probe = Quic0RttProbe(
        frames_sent=3,
        frames_delivered=3,
        session_resumed=True,
        early_data_accepted=True,
        server_early_frames=3,
        handshake_latency_ms=4.0,
        resumption_latency_ms=1.5,
    )
    assert probe.ok is True

    with pytest.raises(ValueError, match="frames_sent"):
        Quic0RttProbe(
            frames_sent=-1,
            frames_delivered=0,
            session_resumed=False,
            early_data_accepted=False,
            server_early_frames=0,
            handshake_latency_ms=None,
            resumption_latency_ms=None,
        )
    with pytest.raises(TypeError, match="session_resumed"):
        Quic0RttProbe(
            frames_sent=1,
            frames_delivered=1,
            session_resumed=1,  # type: ignore[arg-type]
            early_data_accepted=True,
            server_early_frames=1,
            handshake_latency_ms=None,
            resumption_latency_ms=None,
        )


def test_quic_transport_peer_and_cwnd_invariants() -> None:
    peer = RegionalBridgePeer(peer_id="paris-1", continent="EU", rtt_ms=8.5, stun_verified=True)
    assert peer.stun_verified is True

    with pytest.raises(ValueError, match="peer_id"):
        RegionalBridgePeer(peer_id=" ", continent="EU", rtt_ms=8.5)
    with pytest.raises(ValueError, match="rtt_ms"):
        RegionalBridgePeer(peer_id="paris-1", continent="EU", rtt_ms=0.0)
    with pytest.raises(TypeError, match="stun_verified"):
        RegionalBridgePeer(peer_id="paris-1", continent="EU", rtt_ms=8.5, stun_verified=1)  # type: ignore[arg-type]

    cwnd = CongestionWindowTelemetry(
        cwnd_bytes=12000,
        cwnd_packets=10,
        btl_bw_bps=1e8,
        rtprop_ms=12.0,
        jitter_ms=1.0,
        loss_ratio=0.0,
        gain=1.0,
        mtu_bytes=1200,
    )
    assert cwnd.to_dict()["cwnd_packets"] == 10

    with pytest.raises(ValueError, match="cwnd_bytes"):
        CongestionWindowTelemetry(
            cwnd_bytes=0,
            cwnd_packets=10,
            btl_bw_bps=1e8,
            rtprop_ms=12.0,
            jitter_ms=1.0,
            loss_ratio=0.0,
            gain=1.0,
            mtu_bytes=1200,
        )
    with pytest.raises(ValueError, match="mode"):
        CongestionWindowTelemetry(
            cwnd_bytes=12000,
            cwnd_packets=10,
            btl_bw_bps=1e8,
            rtprop_ms=12.0,
            jitter_ms=1.0,
            loss_ratio=0.0,
            gain=1.0,
            mtu_bytes=1200,
            mode="",
        )


def test_swarm_fabric_and_swat_bridge_invariants() -> None:
    pkt = CompressedHiddenStatePacket(
        fp8_data=torch.zeros(4, dtype=torch.uint8),
        absmax_scale=torch.ones(1, dtype=torch.float16),
        original_shape=(1, 4),
        original_bytes=8,
        compressed_bytes=4,
    )
    assert pkt.compression_ratio == 2.0

    with pytest.raises(TypeError, match="fp8_data"):
        CompressedHiddenStatePacket(
            fp8_data="bad",  # type: ignore[arg-type]
            absmax_scale=torch.ones(1),
            original_shape=(1, 4),
            original_bytes=8,
            compressed_bytes=4,
        )
    with pytest.raises(ValueError, match="original_shape"):
        CompressedHiddenStatePacket(
            fp8_data=torch.zeros(4),
            absmax_scale=torch.ones(1),
            original_shape=(),
            original_bytes=8,
            compressed_bytes=4,
        )

    mip = MerkleInclusionProof(
        layer_idx=0,
        leaf_hash="a" * 64,
        siblings=["b" * 64],
        directions=[0],
        root="c" * 64,
        tree_size=2,
    )
    assert mip.proof_depth == 1
    with pytest.raises(ValueError, match="siblings and directions"):
        MerkleInclusionProof(
            layer_idx=0,
            leaf_hash="a" * 64,
            siblings=["b" * 64],
            directions=[],
            root="c" * 64,
            tree_size=2,
        )

    sbc = SpeculativeBatchCommitment(peer_id="peer-1", root="a" * 64, num_steps=4, num_layers=8)
    assert sbc.leaf_count == 32
    with pytest.raises(ValueError, match="num_steps"):
        SpeculativeBatchCommitment(peer_id="peer-1", root="a" * 64, num_steps=0, num_layers=8)

    sbo = SpeculativeBatchOpening(step=0, layer=1, siblings=("a" * 64,))
    assert sbo.to_dict()["layer"] == 1
    with pytest.raises(ValueError, match="step"):
        SpeculativeBatchOpening(step=-1, layer=0, siblings=())

    w2s = WakeToSleepConsolidationBridge()
    assert w2s.swat_bridge is not None
    with pytest.raises(TypeError, match="swat_bridge"):
        WakeToSleepConsolidationBridge(swat_bridge="bad")  # type: ignore[arg-type]
