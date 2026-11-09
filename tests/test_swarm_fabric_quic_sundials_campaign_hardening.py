# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and validation hardening tests for swarm_fabric, quic_transport, rusty_sundials_scientific_bee, and campaign_orchestrator."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.swarm.amcp_kafka_campaign_bus import JOINT_SUBJECT_CAMPAIGNS
from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    night_window,
)
from hivemind.swarm.quic_transport import (
    AdaptiveCongestionController,
    ContinentalSuperRingRouter,
    ZeroRTTSessionTicket,
)
from hivemind.swarm.rusty_sundials_scientific_bee import (
    _extract_prompt_param,
    _rescue_astronomy_kepler,
    _rescue_fluid_mechanics_dyadic_ns,
    _rescue_genetics_stiff_hopfield_and_dna_lk,
    _rescue_quantum_fluid_pgpe,
    build_rusty_sundials_rescue_dataset,
)
from hivemind.swarm.swarm_fabric import (
    RegionalGeoRingTransport,
    benchmark_fused_int4_fp8_moe_kernel,
    compress_hidden_states_fp8,
    decompress_hidden_states_fp8,
    fused_int4_fp8_moe_route_and_project,
    fused_top2_moe_fp8_dispatch,
    pack_int4_nf4_weights,
    unpack_int4_nf4_weights,
)


def test_swarm_fabric_broadcast_and_fp8_int4_validation() -> None:
    transport = RegionalGeoRingTransport()
    with pytest.raises(ValueError, match="preempting_peer_id"):
        transport.broadcast_game_preemption_failover("", "coder")
    with pytest.raises(ValueError, match="expert_domain"):
        transport.broadcast_game_preemption_failover("peer-1", "   ")
    with pytest.raises(ValueError, match="active_game_exe"):
        transport.broadcast_game_preemption_failover("peer-1", "coder", active_game_exe="")
    frame = transport.broadcast_game_preemption_failover("peer-1", "coder", active_game_exe="game.exe")
    assert frame["event"] == "PREEMPTING_FOR_GAME"

    with pytest.raises(ValueError, match="hidden_states"):
        compress_hidden_states_fp8(torch.tensor([]))
    with pytest.raises(ValueError, match="hidden_states"):
        compress_hidden_states_fp8(torch.tensor([float("nan"), 1.0]))
    pkt = compress_hidden_states_fp8(torch.ones((2, 4), dtype=torch.float32))
    with pytest.raises(TypeError, match="packet"):
        decompress_hidden_states_fp8("not-a-packet")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_dtype"):
        decompress_hidden_states_fp8(pkt, target_dtype="float16")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="weight"):
        pack_int4_nf4_weights(torch.ones((4,)))
    p_u8, s_fp16 = pack_int4_nf4_weights(torch.ones((4, 4)))
    with pytest.raises(ValueError, match="packed_uint8"):
        unpack_int4_nf4_weights(torch.ones((4,), dtype=torch.uint8), s_fp16)
    with pytest.raises(ValueError, match="scale_fp16"):
        unpack_int4_nf4_weights(p_u8, torch.tensor([]))
    with pytest.raises(ValueError, match="original_in_dim"):
        unpack_int4_nf4_weights(p_u8, s_fp16, original_in_dim=True)  # type: ignore[arg-type]


def test_swarm_fabric_moe_kernels_validation() -> None:
    x = torch.ones((4, 8), dtype=torch.float32)
    rw = torch.ones((2, 8), dtype=torch.float32)
    ews = [torch.ones((8, 8), dtype=torch.float32), torch.ones((8, 8), dtype=torch.float32)]

    with pytest.raises(ValueError, match="hidden_states"):
        fused_int4_fp8_moe_route_and_project(torch.tensor([]), rw, ews)
    with pytest.raises(ValueError, match="router_weight"):
        fused_int4_fp8_moe_route_and_project(x, torch.ones((8,)), ews)
    with pytest.raises(ValueError, match="expert_weights"):
        fused_int4_fp8_moe_route_and_project(x, rw, [])
    with pytest.raises(ValueError, match="top_k"):
        fused_int4_fp8_moe_route_and_project(x, rw, ews, top_k=0)

    with pytest.raises(ValueError, match="batch_tokens"):
        benchmark_fused_int4_fp8_moe_kernel(batch_tokens=0)
    with pytest.raises(ValueError, match="top_k"):
        benchmark_fused_int4_fp8_moe_kernel(top_k=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hidden_states"):
        fused_top2_moe_fp8_dispatch(torch.tensor([]), rw, ews)
    with pytest.raises(ValueError, match="router_weight"):
        fused_top2_moe_fp8_dispatch(x, torch.ones((8,)), ews)
    with pytest.raises(ValueError, match="expert_down_weights"):
        fused_top2_moe_fp8_dispatch(x, rw, [])
    with pytest.raises(ValueError, match="top_k"):
        fused_top2_moe_fp8_dispatch(x, rw, ews, top_k=True)  # type: ignore[arg-type]


def test_quic_transport_ticket_router_and_cc_validation() -> None:
    with pytest.raises(ValueError, match="peer_id"):
        ZeroRTTSessionTicket(peer_id=" ", session_token=b"tok", issued_at=1.0)
    with pytest.raises(ValueError, match="session_token"):
        ZeroRTTSessionTicket(peer_id="p1", session_token=b"", issued_at=1.0)
    with pytest.raises(ValueError, match="issued_at"):
        ZeroRTTSessionTicket(peer_id="p1", session_token=b"tok", issued_at=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reflexive_port"):
        ZeroRTTSessionTicket(peer_id="p1", session_token=b"tok", issued_at=1.0, reflexive_port=70000)

    router = ContinentalSuperRingRouter()
    with pytest.raises(ValueError, match="peer_id"):
        router.remove_peer("")
    with pytest.raises(ValueError, match="region"):
        router.peers_in_region("  ")
    with pytest.raises(ValueError, match="region"):
        router.find_peer_in_region("")
    with pytest.raises(ValueError, match="target_key"):
        router.find_closest_continental_bridges("")
    with pytest.raises(ValueError, match="k must be a positive integer"):
        router.find_closest_continental_bridges("key", k=0)

    cc = AdaptiveCongestionController()
    with pytest.raises(ValueError, match="lost_packets"):
        cc.record_loss(-1, 10)
    with pytest.raises(ValueError, match="sent_packets"):
        cc.record_loss(0, True)  # type: ignore[arg-type]
    pkt = compress_hidden_states_fp8(torch.ones((2, 4), dtype=torch.float32))
    with pytest.raises(TypeError, match="packet"):
        cc.plan_hidden_state_burst("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_packets"):
        cc.plan_hidden_state_burst(pkt, total_packets=0)


def test_rusty_sundials_helpers_and_campaign_orchestrator_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt"):
        _extract_prompt_param(123, ("ecc",), 0.5, 0.1, 0.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="names"):
        _extract_prompt_param("ecc=0.4", (), 0.5, 0.1, 0.9)
    with pytest.raises(ValueError, match="low must be <= high"):
        _extract_prompt_param("ecc=0.4", ("ecc",), 0.5, 0.9, 0.1)

    with pytest.raises(ValueError, match="steps"):
        _rescue_astronomy_kepler(steps=0)
    with pytest.raises(ValueError, match="nu"):
        _rescue_fluid_mechanics_dyadic_ns(nu=0.0)
    with pytest.raises(ValueError, match="amp"):
        _rescue_quantum_fluid_pgpe(amp=-1.0)
    with pytest.raises(ValueError, match="lk0"):
        _rescue_genetics_stiff_hopfield_and_dna_lk(lk0=0.0)
    with pytest.raises(ValueError, match="output_jsonl_path"):
        build_rusty_sundials_rescue_dataset(output_jsonl_path="")  # type: ignore[arg-type]

    orch = CampaignOrchestrator(
        state_path=tmp_path / "state.json",
        secret=b"0123456789abcdef0123456789abcdef",
        publish=lambda _ev: {"topic": "t", "partition": 0, "offset": 1},
    )
    cid = JOINT_SUBJECT_CAMPAIGNS[0].subject_id
    with pytest.raises(ValueError, match="campaign_id"):
        orch.campaign_window(" ")
    with pytest.raises(ValueError, match="now_utc"):
        orch.is_open(cid, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="campaign_id"):
        orch.active_members(1700000000.0, campaign_id="")
    with pytest.raises(ValueError, match="now_utc"):
        orch.expire(float("nan"))

    with pytest.raises(ValueError, match="now_utc"):
        night_window(False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tz"):
        night_window(1700000000.0, tz=" ")
    with pytest.raises(ValueError, match="invalid night window"):
        night_window(1700000000.0, start_h=True)  # type: ignore[arg-type]
