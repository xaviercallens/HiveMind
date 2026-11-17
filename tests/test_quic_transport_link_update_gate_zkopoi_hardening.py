# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 67: Negative-control and boundary tests for verified_update_gate, zk_opoi_v5_circuit, quic_link, and quic_transport."""

from __future__ import annotations

import pytest

from hivemind.swarm.quic_link import (
    FrameEnvelope,
    LinkEndpoint,
    LinkIdentity,
    RemoteHiddenStateLink,
    _uvarint,
    _validate_timeout,
)
from hivemind.swarm.quic_transport import (
    AsyncQUICGeoRingTransport,
    ContinentalSuperRingRouter,
    SuperRingPeerNode,
)
from hivemind.swarm.verified_update_gate import (
    HeldOutTask,
    UpdatePolicy,
    exact_paired_p,
    extract_code,
)
from hivemind.swarm.zk_opoi_v5_circuit import ZKOPoIV5FixedPointCircuit


def test_verified_update_gate_boundary_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UpdatePolicy(alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_n"):
        UpdatePolicy(min_n=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_n"):
        UpdatePolicy(min_n=10.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_unverified_fraction"):
        UpdatePolicy(max_unverified_fraction=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerated_net_loss"):
        UpdatePolicy(tolerated_net_loss=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="require_improvement"):
        UpdatePolicy(require_improvement=1)  # type: ignore[arg-type]

    for bad_b, bad_c in ((True, 1), (1, False), (-1, 1), (1, -1), (1.5, 2)):
        with pytest.raises(ValueError, match="non-negative integers"):
            exact_paired_p(bad_b, bad_c)  # type: ignore[arg-type]

    assert exact_paired_p(0, 0) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="key"):
        HeldOutTask(key="   ", prompt="def f():", tests=["assert True"])
    with pytest.raises(ValueError, match="tests"):
        HeldOutTask(key="t1", prompt="def f():", tests="assert True")
    with pytest.raises(ValueError, match="tests"):
        HeldOutTask(key="t1", prompt="def f():", tests=[])

    with pytest.raises(TypeError, match="string"):
        extract_code(123)  # type: ignore[arg-type]


def test_zk_opoi_v5_circuit_boundary_validation() -> None:
    with pytest.raises(ValueError, match="quantization_bits"):
        ZKOPoIV5FixedPointCircuit(quantization_bits=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixed_point_scale"):
        ZKOPoIV5FixedPointCircuit(fixed_point_scale=0)
    with pytest.raises(ValueError, match="fixed_point_scale"):
        ZKOPoIV5FixedPointCircuit(fixed_point_scale=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lr_num"):
        ZKOPoIV5FixedPointCircuit(lr_den=0)
    with pytest.raises(ValueError, match="lr_num"):
        ZKOPoIV5FixedPointCircuit(lr_num=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_delta_l2_norm"):
        ZKOPoIV5FixedPointCircuit(max_delta_l2_norm=0.0)
    with pytest.raises(ValueError, match="max_delta_l2_norm"):
        ZKOPoIV5FixedPointCircuit(max_delta_l2_norm=float("nan"))
    with pytest.raises(ValueError, match="max_spectral_bound"):
        ZKOPoIV5FixedPointCircuit(max_spectral_bound=0.0)
    with pytest.raises(ValueError, match="coordinator_secret"):
        ZKOPoIV5FixedPointCircuit(coordinator_secret="   ")

    circuit = ZKOPoIV5FixedPointCircuit(coordinator_secret="secret-test-key")
    with pytest.raises(ValueError, match="non-finite"):
        circuit.quantize_to_fixed_point([True, 0.2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="node_id"):
        circuit.prove_quantized_diloco_transition(
            node_id="   ",
            assigned_shard_sha256="a" * 64,
            theta0_float=[0.5, -0.5],
            step_gradients_float=[[0.1, -0.1]],
        )
    with pytest.raises(ValueError, match="assigned_shard_sha256"):
        circuit.prove_quantized_diloco_transition(
            node_id="worker-1",
            assigned_shard_sha256="   ",
            theta0_float=[0.5, -0.5],
            step_gradients_float=[[0.1, -0.1]],
        )


def test_quic_link_boundary_validation() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _validate_timeout(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="server_name"):
        LinkIdentity.generate(server_name="   ")
    with pytest.raises(ValueError, match="days"):
        LinkIdentity.generate(days=0)
    with pytest.raises(ValueError, match="days"):
        LinkIdentity.generate(days=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="timestamp"):
        FrameEnvelope(sequence=0, sent_at=True, payload=b"abc").encode()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp"):
        FrameEnvelope(sequence=0, sent_at=0.0, payload=b"abc").encode()
    with pytest.raises(TypeError, match="bytes"):
        FrameEnvelope(sequence=0, sent_at=100.0, payload="abc").encode()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsigned 32-bit"):
        _uvarint(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="required"):
        LinkEndpoint(host="   ", port=4433, ca_pem="PEM")
    with pytest.raises(ValueError, match="required"):
        LinkEndpoint(host="127.0.0.1", port=4433, ca_pem="   ")

    ep = LinkEndpoint(host="127.0.0.1", port=4433, ca_pem="PEM")
    with pytest.raises(ValueError, match="timeout"):
        RemoteHiddenStateLink(ep, timeout=True)  # type: ignore[arg-type]


def test_quic_transport_boundary_validation() -> None:
    with pytest.raises(ValueError, match="max_regional_rtt_ms"):
        ContinentalSuperRingRouter(max_regional_rtt_ms=0.0)
    with pytest.raises(ValueError, match="max_regional_rtt_ms"):
        ContinentalSuperRingRouter(max_regional_rtt_ms=True)
    with pytest.raises(ValueError, match="max_bridge_jitter_ms"):
        ContinentalSuperRingRouter(max_bridge_jitter_ms=-1.0)
    with pytest.raises(ValueError, match="max_bridge_jitter_ms"):
        ContinentalSuperRingRouter(max_bridge_jitter_ms=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="base_window"):
        ContinentalSuperRingRouter.bbr_window(True, 1.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="jitter_ms"):
        ContinentalSuperRingRouter.bbr_window(4, -1.0, 0.0)
    with pytest.raises(ValueError, match="jitter_ms"):
        ContinentalSuperRingRouter.bbr_window(4, True, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="loss"):
        ContinentalSuperRingRouter.bbr_window(4, 1.0, True)  # type: ignore[arg-type]

    router = ContinentalSuperRingRouter()
    assert router.register_peer(SuperRingPeerNode(peer_id="   ", region="EU-West", rtt_ms=5.0)) is False
    assert router.register_peer(SuperRingPeerNode(peer_id="p1", region="EU-West", rtt_ms=0.0)) is False
    assert router.register_peer(SuperRingPeerNode(peer_id="p1", region="EU-West", rtt_ms=True)) is False  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source_continent"):
        router.record_inter_continent_rtt("   ", "US", 80.0)
    with pytest.raises(ValueError, match="rtt_ms"):
        router.record_inter_continent_rtt("EU", "US", 0.0)
    with pytest.raises(ValueError, match="rtt_ms"):
        router.record_inter_continent_rtt("EU", "US", True)  # type: ignore[arg-type]

    transport = AsyncQUICGeoRingTransport()
    with pytest.raises(ValueError, match="peer_id"):
        transport.issue_0rtt_ticket("   ")
    transport.issue_0rtt_ticket("peer-1")
    ok, reason = transport.verify_0rtt_ticket_anti_replay("peer-1", 1, timestamp=True)  # type: ignore[arg-type]
    assert ok is False
    assert reason == "invalid_0rtt_ticket_time"
    ok2, reason2 = transport.verify_0rtt_ticket_anti_replay("peer-1", 1, max_age_sec=True)  # type: ignore[arg-type]
    assert ok2 is False
    assert reason2 == "invalid_0rtt_ticket_time"
