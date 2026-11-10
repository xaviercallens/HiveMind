# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for Iteration 157 hardening across:
- `hivemind.swarm.amcp_eda_traffic_sustainer`
- `hivemind.swarm.adaptive_multi_split_wan`
- `hivemind.swarm.abc_moe_agent_mesh`
- `hivemind.swarm.amcp_kafka_transport`
"""

from __future__ import annotations

import pytest

from hivemind.swarm.abc_moe_agent_mesh import (
    ABCMoEAgent,
    AgentMeshOrchestrator,
)
from hivemind.swarm.adaptive_multi_split_wan import (
    AdaptiveMultiSplitWANController,
    CitizenHardwareProfile,
    CutLayerDecision,
    WANLinkTelemetry,
)
from hivemind.swarm.amcp_eda_traffic_sustainer import (
    AMCPEventDrivenTrafficSustainer,
    AMCPMobilityCommand,
    CloudEventEnvelope,
    ElasticEDAScalingResult,
)
from hivemind.swarm.amcp_kafka_campaign_bus import build_amcp_kafka_event
from hivemind.swarm.amcp_kafka_transport import (
    AMCPPublisher,
    ConsumerStats,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
    TransportRecord,
    VerifiedEvent,
)


def test_amcp_eda_traffic_sustainer_dataclasses_hardening() -> None:
    sustainer = AMCPEventDrivenTrafficSustainer()
    env = sustainer.publish_cloudevent(
        event_type="io.hivemind.amcp.test",
        source="urn:hivemind:test",
        subject="subj",
        data={"ok": True},
    )
    assert env.to_dict()["specversion"] == "1.0"

    with pytest.raises(ValueError, match="id must be a non-empty string"):
        CloudEventEnvelope(
            id="",
            source="urn:hivemind:test",
            type="io.hivemind.amcp.test",
            time=100.0,
            correlation_id="corr-1",
            subject="subj",
            data={},
        )
    with pytest.raises(TypeError, match="data must be a dict"):
        CloudEventEnvelope(
            id="ce-1",
            source="urn:hivemind:test",
            type="io.hivemind.amcp.test",
            time=100.0,
            correlation_id="corr-1",
            subject="subj",
            data="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Unsupported AMCPMobilityVerb"):
        AMCPMobilityCommand(
            verb="teleport",  # type: ignore[arg-type]
            bee_id="bee-1",
            model_id="hivemind/coder",
            source_node_id="src",
            target_node_id="dst",
            adapter_size_mib=18.0,
            expected_warm_latency_s=1.2,
            reason="test",
            event_id="ev-1",
        )
    with pytest.raises(ValueError, match="adapter_size_mib must be a positive finite number"):
        AMCPMobilityCommand(
            verb="replicate",
            bee_id="bee-1",
            model_id="hivemind/coder",
            source_node_id="src",
            target_node_id="dst",
            adapter_size_mib=0.0,
            expected_warm_latency_s=1.2,
            reason="test",
            event_id="ev-1",
        )

    res = sustainer.plan_elastic_bee_mobility(
        wave_name="Wave-1",
        base_resident_producers_c0=10,
        standby_prosumer_pool=20,
        peak_arrival_req_per_s=5.0,
    )
    assert res.effective_active_producers_c >= 10

    with pytest.raises(ValueError, match="base_resident_producers_c0 must be a positive integer"):
        ElasticEDAScalingResult(
            wave_name="Wave-1",
            base_resident_producers_c0=0,
            standby_prosumer_pool=10,
            activated_amcp_replicas=2,
            effective_active_producers_c=10,
            peak_arrival_req_per_s=5.0,
            static_service_rate_req_per_s=3.0,
            elastic_service_rate_req_per_s=4.0,
            static_utilization_rho=0.8,
            elastic_eda_utilization_rho=0.5,
            eda_buffered_burst_requests=60,
            cold_start_504_prevented_by_pr249=28,
            mobility_commands_emitted=[],
            sustained_without_drops=True,
        )
    with pytest.raises(TypeError, match="mobility_commands_emitted must be a list of dicts"):
        ElasticEDAScalingResult(
            wave_name="Wave-1",
            base_resident_producers_c0=10,
            standby_prosumer_pool=10,
            activated_amcp_replicas=2,
            effective_active_producers_c=12,
            peak_arrival_req_per_s=5.0,
            static_service_rate_req_per_s=3.0,
            elastic_service_rate_req_per_s=4.0,
            static_utilization_rho=0.8,
            elastic_eda_utilization_rho=0.5,
            eda_buffered_burst_requests=60,
            cold_start_504_prevented_by_pr249=28,
            mobility_commands_emitted=["not-dict"],  # type: ignore[list-item]
            sustained_without_drops=True,
        )


def test_adaptive_multi_split_wan_cut_layer_decision_hardening() -> None:
    ctrl = AdaptiveMultiSplitWANController()
    link = WANLinkTelemetry(uplink_mbps=20.0, downlink_mbps=50.0, rtt_ms=60.0)
    hw = CitizenHardwareProfile(node_id="node-1", gpu_name="RTX 3060", compute_capability_sm=8.6)
    dec = ctrl.select_optimal_cut_layer(link, hw)
    assert dec.cut_layer_index in (1, 2, 3, 4)

    with pytest.raises(ValueError, match="cut_layer_index must be one of"):
        CutLayerDecision(
            node_id="node-1",
            precision_cohort="cohort_bf16_ampere_plus",
            cut_layer_index=9,
            activation_bits=8,
            client_prelude_vram_mib=900.0,
            boundary_payload_kb=512.0,
            prelude_compute_ms=20.0,
            wan_transfer_ms=150.0,
            edge_core_compute_ms=15.0,
            total_step_ms=185.0,
            monolithic_baseline_step_ms=600.0,
            step_time_reduction_pct=69.1,
            preemption_latency_ms=284.0,
            vram_compliant=True,
            preemption_compliant=True,
        )
    with pytest.raises(ValueError, match="step_time_reduction_pct must be a finite number in \\[0.0, 100.0\\]"):
        CutLayerDecision(
            node_id="node-1",
            precision_cohort="cohort_bf16_ampere_plus",
            cut_layer_index=2,
            activation_bits=8,
            client_prelude_vram_mib=900.0,
            boundary_payload_kb=512.0,
            prelude_compute_ms=20.0,
            wan_transfer_ms=150.0,
            edge_core_compute_ms=15.0,
            total_step_ms=185.0,
            monolithic_baseline_step_ms=600.0,
            step_time_reduction_pct=120.0,
            preemption_latency_ms=284.0,
            vram_compliant=True,
            preemption_compliant=True,
        )


def test_abc_moe_agent_mesh_constructor_hardening() -> None:
    with pytest.raises(TypeError, match="web_search_fn must be callable or None"):
        ABCMoEAgent(web_search_fn="not-callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nectar_cache must be a SwarmNectarCache or None"):
        ABCMoEAgent(nectar_cache="not-a-cache")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="available_peers must be a list of peer mappings or None"):
        AgentMeshOrchestrator(available_peers=["not-a-mapping"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="nectar_cache must be a SwarmNectarCache or None"):
        AgentMeshOrchestrator(nectar_cache="not-a-cache")  # type: ignore[arg-type]


def test_amcp_kafka_transport_dataclasses_and_init_hardening() -> None:
    keyring = HMACKeyring(keys={"k1": b"a" * 32}, active_kid="k1")
    transport = InMemoryTransport()
    ev = build_amcp_kafka_event(
        topic_key="human_agent",
        event_type="io.hivemind.amcp.message",
        sender_id="agent-1",
        sender_kind="ai_agent",
        now_utc=1700000000.0,
    )
    rec = TransportRecord("hivemind.amcp.human_agent.v1", 0, 0, b"k", b"{}")
    ve = VerifiedEvent(event=ev, key_id="k1", record=rec)
    assert ve.key_id == "k1"

    with pytest.raises(TypeError, match="event must be an AMCPKafkaEvent"):
        VerifiedEvent(event="bad", key_id="k1", record=rec)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key_id must be a non-empty string"):
        VerifiedEvent(event=ev, key_id="", record=rec)
    with pytest.raises(TypeError, match="record must be a TransportRecord"):
        VerifiedEvent(event=ev, key_id="k1", record="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="keyring must be an HMACKeyring"):
        EnvelopeVerifier("not-keyring")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keyring must be an HMACKeyring"):
        AMCPPublisher(transport, "not-keyring")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mirror must be an InMemoryKafkaBridge or None"):
        AMCPPublisher(transport, keyring, mirror="not-bridge")  # type: ignore[arg-type]

    stats = ConsumerStats()
    assert stats.to_dict()["polled"] == 0
    with pytest.raises(ValueError, match="polled must be a non-negative integer"):
        ConsumerStats(polled=-1)
    with pytest.raises(TypeError, match="dead_lettered must be a dict"):
        ConsumerStats(dead_lettered="bad")  # type: ignore[arg-type]
