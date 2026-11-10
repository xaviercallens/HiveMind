# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for CrowdGPT bridge, International Launch Scaler, WAN Fault Benchmark, and NAT Traversal (IP-161)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.swarm.crowdgpt_bridge import (
    LossRatchetDecision,
    LossRatchetGate,
    WeightRmsGuardDecision,
    check_weight_rms_collapse_guard,
)
from hivemind.swarm.international_launch_scaler import (
    Top1PctDctEf21Packet,
    compress_top1pct_dct_ef21,
)
from hivemind.swarm.nat_traversal import (
    BINDING,
    REQUEST,
    NatMapping,
    PunchResult,
    StunBindingResult,
    StunMessage,
)
from hivemind.swarm.wan_fault_benchmark import ExternalEndpoints


def test_loss_ratchet_decision_and_weight_rms_guard_decision_validation() -> None:
    gate = LossRatchetGate(max_regression_tolerance=0.01)
    delta = torch.ones(8, dtype=torch.bfloat16)
    ok_dec = gate.evaluate(loss_before=2.5, loss_after=2.4, candidate_delta_bf16=delta)
    assert ok_dec.accepted is True
    nan_dec = gate.evaluate(loss_before=float("nan"), loss_after=2.4, candidate_delta_bf16=delta)
    assert nan_dec.accepted is False

    with pytest.raises(ValueError, match="accepted must be a boolean"):
        LossRatchetDecision(
            accepted=1,  # type: ignore[arg-type]
            loss_before=2.5,
            loss_after=2.4,
            delta_loss=-0.1,
            checkpoint_sha256="a" * 64,
            reason="ok",
        )
    with pytest.raises(ValueError, match="loss_before must be a number"):
        LossRatchetDecision(
            accepted=True,
            loss_before=True,  # type: ignore[arg-type]
            loss_after=2.4,
            delta_loss=-0.1,
            checkpoint_sha256="a" * 64,
            reason="ok",
        )
    with pytest.raises(ValueError, match="loss_after must be finite when accepted is True"):
        LossRatchetDecision(
            accepted=True,
            loss_before=2.5,
            loss_after=float("nan"),
            delta_loss=-0.1,
            checkpoint_sha256="a" * 64,
            reason="ok",
        )
    with pytest.raises(ValueError, match="checkpoint_sha256 must be a non-empty string"):
        LossRatchetDecision(
            accepted=True,
            loss_before=2.5,
            loss_after=2.4,
            delta_loss=-0.1,
            checkpoint_sha256="   ",
            reason="ok",
        )

    old_w = torch.ones(16, dtype=torch.float32) * 0.5
    new_w = torch.ones(16, dtype=torch.float32) * 0.48
    rms_ok = check_weight_rms_collapse_guard(old_w, new_w)
    assert rms_ok.accepted is True
    rms_nan = check_weight_rms_collapse_guard(old_w, torch.tensor([float("nan")] * 16))
    assert rms_nan.accepted is False

    with pytest.raises(ValueError, match="accepted must be a boolean"):
        WeightRmsGuardDecision(
            accepted="yes",  # type: ignore[arg-type]
            rms_old=0.5,
            rms_new=0.48,
            rms_ratio=0.96,
            halving_residual_rms=0.23,
            reason="ok",
        )
    with pytest.raises(ValueError, match="rms_old must be positive when accepted is True"):
        WeightRmsGuardDecision(
            accepted=True,
            rms_old=0.0,
            rms_new=0.48,
            rms_ratio=0.96,
            halving_residual_rms=0.23,
            reason="ok",
        )
    with pytest.raises(ValueError, match="halving_residual_rms must be non-negative"):
        WeightRmsGuardDecision(
            accepted=True,
            rms_old=0.5,
            rms_new=0.48,
            rms_ratio=0.96,
            halving_residual_rms=-0.01,
            reason="ok",
        )
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        WeightRmsGuardDecision(
            accepted=True,
            rms_old=0.5,
            rms_new=0.48,
            rms_ratio=0.96,
            halving_residual_rms=0.23,
            reason="",
        )


def test_top1pct_dct_ef21_packet_validation() -> None:
    mat = np.array([[0.5, -0.25], [0.75, 0.1]], dtype=np.float64)
    pkt, _, _ = compress_top1pct_dct_ef21(mat, top_k_fraction=0.5)
    assert pkt.total_elements == 4

    with pytest.raises(ValueError, match="original_shape must be a 2-tuple of integers"):
        Top1PctDctEf21Packet(
            original_shape=(2,),  # type: ignore[arg-type]
            total_elements=pkt.total_elements,
            retained_k=pkt.retained_k,
            top_k_fraction=pkt.top_k_fraction,
            flat_indices=pkt.flat_indices,
            quantized_codes_2bit=pkt.quantized_codes_2bit,
            codebook_4level=pkt.codebook_4level,
            uncompressed_fp32_bytes=pkt.uncompressed_fp32_bytes,
            compressed_wire_bytes=pkt.compressed_wire_bytes,
            compression_ratio=pkt.compression_ratio,
            residual_conservation_l2_error=pkt.residual_conservation_l2_error,
            packet_sha256=pkt.packet_sha256,
        )
    with pytest.raises(ValueError, match="retained_k cannot exceed total_elements"):
        Top1PctDctEf21Packet(
            original_shape=pkt.original_shape,
            total_elements=2,
            retained_k=4,
            top_k_fraction=pkt.top_k_fraction,
            flat_indices=pkt.flat_indices,
            quantized_codes_2bit=pkt.quantized_codes_2bit,
            codebook_4level=pkt.codebook_4level,
            uncompressed_fp32_bytes=pkt.uncompressed_fp32_bytes,
            compressed_wire_bytes=pkt.compressed_wire_bytes,
            compression_ratio=pkt.compression_ratio,
            residual_conservation_l2_error=pkt.residual_conservation_l2_error,
            packet_sha256=pkt.packet_sha256,
        )
    with pytest.raises(ValueError, match="codebook_4level must have shape"):
        Top1PctDctEf21Packet(
            original_shape=pkt.original_shape,
            total_elements=pkt.total_elements,
            retained_k=pkt.retained_k,
            top_k_fraction=pkt.top_k_fraction,
            flat_indices=pkt.flat_indices,
            quantized_codes_2bit=pkt.quantized_codes_2bit,
            codebook_4level=np.zeros(3, dtype=np.float32),
            uncompressed_fp32_bytes=pkt.uncompressed_fp32_bytes,
            compressed_wire_bytes=pkt.compressed_wire_bytes,
            compression_ratio=pkt.compression_ratio,
            residual_conservation_l2_error=pkt.residual_conservation_l2_error,
            packet_sha256=pkt.packet_sha256,
        )
    with pytest.raises(ValueError, match="packet_sha256 must be a non-empty string"):
        Top1PctDctEf21Packet(
            original_shape=pkt.original_shape,
            total_elements=pkt.total_elements,
            retained_k=pkt.retained_k,
            top_k_fraction=pkt.top_k_fraction,
            flat_indices=pkt.flat_indices,
            quantized_codes_2bit=pkt.quantized_codes_2bit,
            codebook_4level=pkt.codebook_4level,
            uncompressed_fp32_bytes=pkt.uncompressed_fp32_bytes,
            compressed_wire_bytes=pkt.compressed_wire_bytes,
            compression_ratio=pkt.compression_ratio,
            residual_conservation_l2_error=pkt.residual_conservation_l2_error,
            packet_sha256="   ",
        )


def test_external_endpoints_and_nat_traversal_dataclasses_validation() -> None:
    ep = ExternalEndpoints(quic_host="127.0.0.1", quic_port=4433, quic_ca_pem="PEM")
    assert ep.has_quic is True
    with pytest.raises(ValueError, match="quic_host must be None or a non-empty string"):
        ExternalEndpoints(quic_host="  ")
    with pytest.raises(ValueError, match="quic_port must be None or an integer in 1..65535"):
        ExternalEndpoints(quic_port=70000)
    with pytest.raises(ValueError, match="turn_port must be None or an integer in 1..65535"):
        ExternalEndpoints(turn_port=True)  # type: ignore[arg-type]

    msg = StunMessage(BINDING, REQUEST)
    assert len(msg.transaction_id) == 12
    with pytest.raises(ValueError, match="method must be an integer"):
        StunMessage(-1, REQUEST)
    with pytest.raises(ValueError, match="cls must be an integer in 0..3"):
        StunMessage(BINDING, 4)
    with pytest.raises(ValueError, match="transaction_id must be 12 bytes"):
        StunMessage(BINDING, REQUEST, b"short")
    with pytest.raises(ValueError, match="attributes must be a list"):
        StunMessage(BINDING, REQUEST, b"a" * 12, attributes=[("bad", b"x")])  # type: ignore[list-item]

    res = StunBindingResult(
        server=("198.51.100.1", 3478),
        local=("192.168.1.2", 50000),
        mapped=("203.0.113.5", 50000),
        rtt_ms=12.4,
    )
    assert res.rtt_ms == 12.4
    with pytest.raises(ValueError, match="server must be a"):
        StunBindingResult(
            server=("", 3478),
            local=("192.168.1.2", 50000),
            mapped=("203.0.113.5", 50000),
            rtt_ms=12.4,
        )
    with pytest.raises(ValueError, match="rtt_ms must be a finite non-negative number"):
        StunBindingResult(
            server=("198.51.100.1", 3478),
            local=("192.168.1.2", 50000),
            mapped=("203.0.113.5", 50000),
            rtt_ms=-1.0,
        )

    mapping = NatMapping(
        behaviour="endpoint_independent",
        local=("192.168.1.2", 50000),
        mapped=[("203.0.113.5", 50000)],
        servers=[("198.51.100.1", 3478)],
        rtt_ms=[10.5],
    )
    assert mapping.hole_punchable is True
    with pytest.raises(ValueError, match="behaviour must be one of"):
        NatMapping(
            behaviour="invalid_mode",
            local=("192.168.1.2", 50000),
            mapped=[("203.0.113.5", 50000)],
            servers=[("198.51.100.1", 3478)],
        )
    with pytest.raises(ValueError, match="mapped must be a non-empty list"):
        NatMapping(
            behaviour="no_nat",
            local=("192.168.1.2", 50000),
            mapped=[],
            servers=[("198.51.100.1", 3478)],
        )

    punch_ok = PunchResult(succeeded=True, peer=("203.0.113.9", 51000), rtt_ms=14.2, probes_sent=3)
    assert punch_ok.succeeded is True
    punch_fail = PunchResult(succeeded=False, peer=None, rtt_ms=None, probes_sent=10)
    assert punch_fail.succeeded is False
    with pytest.raises(ValueError, match="peer cannot be None when succeeded is True"):
        PunchResult(succeeded=True, peer=None, rtt_ms=14.2, probes_sent=3)
    with pytest.raises(ValueError, match="rtt_ms cannot be None when succeeded is True"):
        PunchResult(succeeded=True, peer=("203.0.113.9", 51000), rtt_ms=None, probes_sent=3)
    with pytest.raises(ValueError, match="probes_sent must be a non-negative integer"):
        PunchResult(succeeded=False, peer=None, rtt_ms=None, probes_sent=-1)
