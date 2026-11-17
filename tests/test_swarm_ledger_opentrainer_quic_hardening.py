# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and negative-control tests for swarm ledger, opentrainer_orchestrator, quic_link, and quic_transport hardening."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm.ledger import HiveSwarmLedger, WorkReceipt
from hivemind.swarm.opentrainer_orchestrator import (
    OpenTrainerConfig,
    PermanentOpenTrainerOrchestrator,
    _atomic_write_bytes,
    _atomic_write_json,
)
from hivemind.swarm.quic_link import (
    LinkEndpoint,
    LinkIdentity,
    QuicHiddenStateServer,
    RemoteHiddenStateLink,
    WebSocketHiddenStateServer,
    await_quic_closed,
)
from hivemind.swarm.quic_transport import (
    AdaptiveCongestionController,
    AsyncQUICGeoRingTransport,
)


def test_ledger_mine_pouw_block_and_get_balance_validation() -> None:
    ledger = HiveSwarmLedger()
    with pytest.raises(ValueError, match="outer_step"):
        ledger.mine_pouw_block(-1, "cid://expert", [], [], [])
    with pytest.raises(ValueError, match="outer_step"):
        ledger.mine_pouw_block(True, "cid://expert", [], [], [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expert_cid"):
        ledger.mine_pouw_block(1, "   ", [], [], [])
    with pytest.raises(SwarmSecurityViolationError, match="INVALID_NEGATIVE_OR_NAN_FLOPS_RECEIPT"):
        ledger.mine_pouw_block(
            1,
            "cid://expert",
            [
                WorkReceipt(
                    node_id="w1",
                    steam_id64="76561198000000001",
                    branch_name="main",
                    flops_contributed=True,  # type: ignore[arg-type]
                    cosine_alignment=0.95,
                    spot_check_passed=True,
                    multikrum_accepted=True,
                    delta_sha256="a" * 64,
                )
            ],
            [],
            [],
        )
    with pytest.raises(SwarmSecurityViolationError, match="INVALID_NAN_OR_INF_COSINE_ALIGNMENT"):
        ledger.mine_pouw_block(
            1,
            "cid://expert",
            [
                WorkReceipt(
                    node_id="w1",
                    steam_id64="76561198000000001",
                    branch_name="main",
                    flops_contributed=1e12,
                    cosine_alignment=float("nan"),
                    spot_check_passed=True,
                    multikrum_accepted=True,
                    delta_sha256="a" * 64,
                )
            ],
            [],
            [],
        )
    with pytest.raises(ValueError, match="node_id"):
        ledger.get_balance("")
    assert ledger.get_balance("w1") == 0.0


def test_opentrainer_orchestrator_atomic_io_and_init_validation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="data must be bytes"):
        _atomic_write_bytes(tmp_path / "out.bin", "not-bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="payload must be a Mapping"):
        _atomic_write_json(tmp_path / "out.json", ["not-a-dict"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be an OpenTrainerConfig"):
        PermanentOpenTrainerOrchestrator("not-a-config")  # type: ignore[arg-type]

    cfg = OpenTrainerConfig(rounds=2, inner_steps=1, shard_size=4)
    orch = PermanentOpenTrainerOrchestrator(cfg)
    with pytest.raises(ValueError, match="now must be a finite positive timestamp"):
        orch._open_new_round_offer(now=0.0)
    with pytest.raises(ValueError, match="now must be a finite positive timestamp"):
        orch._open_new_round_offer(now=float("nan"))


def test_quic_link_server_client_and_await_closed_validation() -> None:
    identity = LinkIdentity.generate()
    with pytest.raises(TypeError, match="identity must be a LinkIdentity"):
        QuicHiddenStateServer("not-identity")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        QuicHiddenStateServer(identity, host="  ")
    with pytest.raises(ValueError, match="port must be an integer"):
        QuicHiddenStateServer(identity, port=70000)
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        WebSocketHiddenStateServer(host="")
    with pytest.raises(ValueError, match="port must be an integer"):
        WebSocketHiddenStateServer(port=-1)
    with pytest.raises(TypeError, match="endpoint must be a LinkEndpoint"):
        RemoteHiddenStateLink("not-an-endpoint")  # type: ignore[arg-type]

    endpoint = LinkEndpoint(host="127.0.0.1", port=4433, ca_pem=identity.ca_pem)
    link = RemoteHiddenStateLink(endpoint, timeout=2.0)
    assert link.endpoint.port == 4433

    with pytest.raises(ValueError, match="link timeout must be finite and positive"):
        asyncio.run(await_quic_closed(None, timeout=0.0))  # type: ignore[arg-type]


def test_quic_transport_congestion_controller_and_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="mtu_bytes"):
        AdaptiveCongestionController(mtu_bytes=0)
    with pytest.raises(ValueError, match="max_cwnd_packets"):
        AdaptiveCongestionController(min_cwnd_packets=10, max_cwnd_packets=5)
    with pytest.raises(ValueError, match="startup_gain"):
        AdaptiveCongestionController(startup_gain=0.0)
    with pytest.raises(ValueError, match="loss_beta"):
        AdaptiveCongestionController(loss_beta=0.0)

    cc = AdaptiveCongestionController()
    with pytest.raises(ValueError, match="rtt_ms"):
        cc.record_rtt_sample(float("nan"))
    with pytest.raises(ValueError, match="rtt_ms"):
        cc.record_rtt_sample(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bytes_delivered"):
        cc.record_delivery(1200, float("inf"))
    cc.record_rtt_sample(12.5)
    cc.record_delivery(2400, 10.0)
    assert cc.rtprop_ms == pytest.approx(12.5)

    transport = AsyncQUICGeoRingTransport()
    with pytest.raises(ValueError, match="num_frames"):
        transport.benchmark_loopback_0rtt(torch.randn(2, 8), num_frames=0)
    with pytest.raises(ValueError, match="num_frames"):
        transport.benchmark_loopback_0rtt(torch.randn(2, 8), num_frames=True)  # type: ignore[arg-type]
