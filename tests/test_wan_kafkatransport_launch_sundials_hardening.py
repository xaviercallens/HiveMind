from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from hivemind.swarm.wan_fault_benchmark import (
    FaultProfile,
    FaultScenario,
    StrikeRegister,
    _envelope_frames,
    _throughput_mbps,
    default_scenarios,
    latency_summary,
    percentile,
)
from hivemind.swarm.amcp_kafka_transport import (
    AMCPConsumer,
    AMCPKafkaEvent,
    AMCPPublisher,
    EnvelopeVerifier,
    FieldRule,
    HMACKeyring,
    InMemoryTransport,
    ReplayWindow,
    TopicSchema,
    TransportRecord,
    sign_envelope,
)
from hivemind.swarm.international_launch_scaler import (
    Top1PctDctEf21Packet,
    _orthonormal_dct_1d_matrix,
    coalesce_ledger_settlement_batch,
    compress_top1pct_dct_ef21,
    compute_hierarchical_fanout_plan,
    decompress_top1pct_dct,
    enroll_launch_participant,
    monitor_nocturnal_image_moe_evolution,
    query_local_tesla_t4_vram_guard,
    route_follow_the_sun_workload,
    select_adaptive_multisplit_cut,
    select_boinc_plan_class_for_participant,
)
from hivemind.swarm.rusty_sundials_scientific_bee import (
    BeeRetentionStore,
    OwnerFleetCoServingOrchestrator,
    SolverTaughtSurrogateExpert,
    _wilson_lower_bound_95,
    analyze_physical_mathematical_problem,
    rescue_numeric_calculation,
    run_wave2_parameter_sweep_quorum,
    solve_wave2_method_of_lines_pde,
)


# ---------------------------------------------------------------------------
# 1. wan_fault_benchmark.py
# ---------------------------------------------------------------------------


def test_wan_fault_benchmark_percentile_and_latency_summary_validation() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0, 2.0], True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0, 2.0], 1.1)
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0, 2.0], float("nan"))
    with pytest.raises(ValueError, match="finite"):
        percentile([1.0, True], 0.5)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        percentile([1.0, float("inf")], 0.5)
    with pytest.raises(ValueError, match="sequence"):
        percentile("123", 0.5)  # type: ignore[arg-type]

    s = latency_summary([10.0, 20.0, 30.0])
    assert s["p50_ms"] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="finite"):
        latency_summary([10.0, False])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        latency_summary([10.0, float("nan")])


def test_wan_fault_benchmark_throughput_and_scenario_validation() -> None:
    assert _throughput_mbps(1_000_000, 1.0) == pytest.approx(8.0)
    with pytest.raises(ValueError, match="payload_bytes"):
        _throughput_mbps(True, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload_bytes"):
        _throughput_mbps(-1, 1.0)
    with pytest.raises(ValueError, match="elapsed_s"):
        _throughput_mbps(100, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="elapsed_s"):
        _throughput_mbps(100, float("nan"))

    prof = default_scenarios(seed=7)[0].profile
    sc = FaultScenario(scenario_id="ok", profile=prof)
    assert sc.scenario_id == "ok"
    with pytest.raises(ValueError, match="scenario_id"):
        FaultScenario(scenario_id=" ", profile=prof)
    with pytest.raises(ValueError, match="profile"):
        FaultScenario(scenario_id="ok", profile="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nat_rebinds"):
        FaultScenario(scenario_id="ok", profile=prof, nat_rebinds=-1)
    with pytest.raises(ValueError, match="frames"):
        FaultScenario(scenario_id="ok", profile=prof, frames=0)
    with pytest.raises(ValueError, match="frame_bytes"):
        FaultScenario(scenario_id="ok", profile=prof, frame_bytes=True)  # type: ignore[arg-type]

    assert len(default_scenarios(seed=42)) == 4
    with pytest.raises(ValueError, match="seed"):
        default_scenarios(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="loss_rate"):
        default_scenarios(loss_rates=[1.0])


def test_wan_fault_benchmark_strike_register_and_frames_validation() -> None:
    sr = StrikeRegister(peer_id="peer-a")
    assert sr.register(1) is True
    assert sr.replay(1) is False

    with pytest.raises(ValueError, match="peer_id"):
        StrikeRegister(peer_id=" ")
    with pytest.raises(ValueError, match="nonce"):
        sr.register(0)
    with pytest.raises(ValueError, match="nonce"):
        sr.register(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonce"):
        sr.replay(0)

    frames = _envelope_frames(4, 128)
    assert len(frames) == 4
    with pytest.raises(ValueError, match="count"):
        _envelope_frames(-1, 128)
    with pytest.raises(ValueError, match="count"):
        _envelope_frames(True, 128)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload_bytes"):
        _envelope_frames(4, 0)


# ---------------------------------------------------------------------------
# 2. amcp_kafka_transport.py
# ---------------------------------------------------------------------------


def test_amcp_kafka_transport_record_and_inmemory_validation() -> None:
    rec = TransportRecord(topic="amcp.v1", partition=0, offset=0, key=b"k1", value=b"{}")
    assert rec.topic == "amcp.v1"

    with pytest.raises(ValueError, match="topic"):
        TransportRecord(topic="", partition=0, offset=0, key=b"k1", value=b"{}")
    with pytest.raises(ValueError, match="partition"):
        TransportRecord(topic="t", partition=-1, offset=0, key=b"k1", value=b"{}")
    with pytest.raises(ValueError, match="offset"):
        TransportRecord(topic="t", partition=0, offset=True, key=b"k1", value=b"{}")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key"):
        TransportRecord(topic="t", partition=0, offset=0, key="str-key", value=b"{}")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value"):
        TransportRecord(topic="t", partition=0, offset=0, key=b"k", value="str-val")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_partitions"):
        InMemoryTransport(num_partitions=0)
    with pytest.raises(ValueError, match="num_partitions"):
        InMemoryTransport(num_partitions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retention_per_partition"):
        InMemoryTransport(retention_per_partition=0)

    tr = InMemoryTransport(num_partitions=2)
    with pytest.raises(ValueError, match="key"):
        tr.partition_for("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="topic"):
        tr.publish("", b"k", b"v")
    with pytest.raises(ValueError, match="bytes"):
        tr.publish("t", "k", b"v")  # type: ignore[arg-type]

    meta = tr.publish("t", b"k", b"v")
    assert meta["offset"] == 0

    with pytest.raises(ValueError, match="max_records"):
        tr.poll(0, 0.1)
    with pytest.raises(ValueError, match="max_records"):
        tr.poll(True, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout_s"):
        tr.poll(10, -0.1)
    with pytest.raises(ValueError, match="timeout_s"):
        tr.poll(10, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="TransportRecord"):
        tr.commit({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="offset"):
        tr.seek("t", 0, -1)
    with pytest.raises(ValueError, match="partition"):
        tr.seek("t", True, 0)  # type: ignore[arg-type]


def test_amcp_kafka_transport_signing_schema_and_consumer_validation() -> None:
    keyring = HMACKeyring({"k1": b"0123456789abcdef0123456789abcdef"}, active_kid="k1")
    ev = AMCPKafkaEvent(
        event_id="ev-1",
        event_type="io.hivemind.amcp.trade.order",
        topic="hivemind.amcp.trade.v1",
        sender_kind="bee",
        sender_id="bee-1",
        recipient_id="coord-1",
        subject_id="bee-1",
        correlation_id="c-1",
        utc=1700000000.0,
        payload={"order_id": "o1", "bee_id": "bee-1", "gpu_hours": 1.0, "price_eur": 0.0},
    )
    env = sign_envelope(ev, keyring, ttl_s=60.0)
    assert "hmsig" in env

    with pytest.raises(ValueError, match="ttl_s"):
        sign_envelope(ev, keyring, ttl_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ttl_s"):
        sign_envelope(ev, keyring, ttl_s=0.0)

    with pytest.raises(ValueError, match="FieldRule.kind"):
        FieldRule(kind="unknown")
    with pytest.raises(ValueError, match="FieldRule.max_len"):
        FieldRule(kind="str", max_len=0)
    with pytest.raises(ValueError, match="FieldRule.ge"):
        FieldRule(kind="number", ge=10.0, le=5.0)
    with pytest.raises(ValueError, match="FieldRule.gt"):
        FieldRule(kind="number", gt=5.0, le=5.0)

    with pytest.raises(ValueError, match="topic_key"):
        TopicSchema(topic_key="", payloads={})
    with pytest.raises(ValueError, match="payloads"):
        TopicSchema(topic_key="trade", payloads=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="open_event_types"):
        TopicSchema(topic_key="trade", payloads={}, open_event_types="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_ttl_s"):
        EnvelopeVerifier(keyring, max_ttl_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clock"):
        EnvelopeVerifier(keyring, clock="not-callable")  # type: ignore[arg-type]

    tr = InMemoryTransport()
    with pytest.raises(ValueError, match="ttl_s"):
        AMCPPublisher(tr, keyring, ttl_s=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="capacity"):
        ReplayWindow(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        ReplayWindow(capacity=True)  # type: ignore[arg-type]
    rw = ReplayWindow(capacity=8)
    with pytest.raises(ValueError, match="event_id"):
        rw.seen("", 100.0)
    with pytest.raises(ValueError, match="now"):
        rw.mark("e1", True)  # type: ignore[arg-type]

    verifier = EnvelopeVerifier(keyring)
    with pytest.raises(ValueError, match="max_batch"):
        AMCPConsumer(tr, verifier, max_batch=0)
    with pytest.raises(ValueError, match="max_attempts"):
        AMCPConsumer(tr, verifier, max_attempts=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="poll_timeout_s"):
        AMCPConsumer(tr, verifier, poll_timeout_s=-1.0)
    with pytest.raises(ValueError, match="clock"):
        AMCPConsumer(tr, verifier, clock="bad")  # type: ignore[arg-type]

    consumer = AMCPConsumer(tr, verifier)
    with pytest.raises(ValueError, match="event_type"):
        consumer.register("bad.prefix", lambda _e: None)
    with pytest.raises(ValueError, match="callable"):
        consumer.register("io.hivemind.amcp.trade.order", "not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="callable"):
        consumer.register_topic("trade", "not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_batches"):
        consumer.run(max_batches=0)
    with pytest.raises(ValueError, match="idle_sleep_s"):
        consumer.run(max_batches=1, idle_sleep_s=-0.1)


# ---------------------------------------------------------------------------
# 3. international_launch_scaler.py
# ---------------------------------------------------------------------------


def test_international_launch_scaler_multisplit_and_dct_validation() -> None:
    cut = select_adaptive_multisplit_cut(uplink_mbps=50.0, rtt_ms=40.0, client_vram_limit_mib=1536.0)
    assert cut is not None

    with pytest.raises(ValueError, match="uplink_mbps"):
        select_adaptive_multisplit_cut(uplink_mbps=0.0, rtt_ms=40.0)
    with pytest.raises(ValueError, match="uplink_mbps"):
        select_adaptive_multisplit_cut(uplink_mbps=True, rtt_ms=40.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rtt_ms"):
        select_adaptive_multisplit_cut(uplink_mbps=50.0, rtt_ms=-1.0)
    with pytest.raises(ValueError, match="client_vram_limit_mib"):
        select_adaptive_multisplit_cut(uplink_mbps=50.0, rtt_ms=40.0, client_vram_limit_mib=float("nan"))

    with pytest.raises(ValueError, match="n must be"):
        _orthonormal_dct_1d_matrix(0)
    with pytest.raises(ValueError, match="n must be"):
        _orthonormal_dct_1d_matrix(True)  # type: ignore[arg-type]

    mat = np.array([[0.5, -0.25], [0.75, 0.1]], dtype=np.float64)
    packet, recon, next_res = compress_top1pct_dct_ef21(mat, top_k_fraction=0.5)
    restored = decompress_top1pct_dct(packet)
    assert restored.shape == (2, 2)
    assert recon.shape == (2, 2)
    assert next_res.shape == (2, 2)

    with pytest.raises(ValueError, match="top_k_fraction"):
        compress_top1pct_dct_ef21(mat, top_k_fraction=0.0)
    with pytest.raises(ValueError, match="top_k_fraction"):
        compress_top1pct_dct_ef21(mat, top_k_fraction=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delta_matrix"):
        compress_top1pct_dct_ef21(np.array([]), top_k_fraction=0.5)
    with pytest.raises(ValueError, match="ef21_residual_buffer"):
        compress_top1pct_dct_ef21(mat, ef21_residual_buffer=np.array([1.0]))

    with pytest.raises(ValueError, match="Top1PctDctEf21Packet"):
        decompress_top1pct_dct({})  # type: ignore[arg-type]
    bad_pkt = dataclasses.replace(packet, original_shape=(0, 2))
    with pytest.raises(ValueError, match="original_shape"):
        decompress_top1pct_dct(bad_pkt)


def test_international_launch_scaler_fanout_ledger_routing_and_boinc_validation() -> None:
    plan = compute_hierarchical_fanout_plan(total_leaf_nodes=120, branching_factor=16)
    assert plan.total_leaf_nodes >= 120
    with pytest.raises(ValueError, match="total_leaf_nodes"):
        compute_hierarchical_fanout_plan(total_leaf_nodes=0)
    with pytest.raises(ValueError, match="branching_factor"):
        compute_hierarchical_fanout_plan(total_leaf_nodes=10, branching_factor=True)  # type: ignore[arg-type]

    batch = coalesce_ledger_settlement_batch(
        [{"contributor_handle": "@n1", "hmt_points": 5, "workunits": 1}, {"contributor_handle": "@n1", "hmt_points": 3, "workunits": 2}],
        window_ms=500,
    )
    assert batch is not None
    with pytest.raises(ValueError, match="window_ms"):
        coalesce_ledger_settlement_batch([], window_ms=0)
    with pytest.raises(ValueError, match="settlement_events"):
        coalesce_ledger_settlement_batch("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hmt_points"):
        coalesce_ledger_settlement_batch([{"contributor_handle": "@n1", "hmt_points": -1}])
    with pytest.raises(ValueError, match="workunits"):
        coalesce_ledger_settlement_batch([{"contributor_handle": "@n1", "hmt_points": 5, "workunits": 0}])

    routed = route_follow_the_sun_workload(utc_hour=3)
    assert routed["utc_hour"] == 3
    with pytest.raises(ValueError, match="utc_hour"):
        route_follow_the_sun_workload(utc_hour=True)  # type: ignore[arg-type]

    boinc = select_boinc_plan_class_for_participant(vram_mib=8192.0)
    assert boinc is not None
    with pytest.raises(ValueError, match="vram_mib"):
        select_boinc_plan_class_for_participant(vram_mib=-1.0)
    with pytest.raises(ValueError, match="vram_mib"):
        select_boinc_plan_class_for_participant(vram_mib=True)  # type: ignore[arg-type]

    enrolled = enroll_launch_participant(handle="@node-1", vram_mib=8192.0, uplink_mbps=100.0, rtt_ms=25.0)
    assert enrolled is not None
    with pytest.raises(ValueError, match="handle"):
        enroll_launch_participant(handle="", vram_mib=8192.0)
    with pytest.raises(ValueError, match="vram_mib"):
        enroll_launch_participant(handle="@node-1", vram_mib=0.0)

    with pytest.raises(ValueError, match="min_free_mib"):
        query_local_tesla_t4_vram_guard(min_free_mib=-1.0)
    with pytest.raises(ValueError, match="min_free_mib"):
        query_local_tesla_t4_vram_guard(min_free_mib=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="generations"):
        monitor_nocturnal_image_moe_evolution(generations=0)
    with pytest.raises(ValueError, match="base_inner_steps"):
        monitor_nocturnal_image_moe_evolution(base_inner_steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="save_artifacts"):
        monitor_nocturnal_image_moe_evolution(save_artifacts="no")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. rusty_sundials_scientific_bee.py
# ---------------------------------------------------------------------------


def test_rusty_sundials_bee_analysis_rescue_and_retention_validation(tmp_path: Path) -> None:
    diag = analyze_physical_mathematical_problem("Integrate 3-body gravitational orbit")
    assert diag.domain == "astronomy"
    with pytest.raises(ValueError, match="prompt_or_equation"):
        analyze_physical_mathematical_problem("   ")
    with pytest.raises(ValueError, match="domain_hint"):
        analyze_physical_mathematical_problem("orbit", domain_hint=123)  # type: ignore[arg-type]

    res = rescue_numeric_calculation("Integrate orbit", domain="astronomy", steps=20, dt=0.01)
    assert res.rescued_succeeded is True
    with pytest.raises(ValueError, match="steps"):
        rescue_numeric_calculation("Integrate orbit", steps=0)
    with pytest.raises(ValueError, match="steps"):
        rescue_numeric_calculation("Integrate orbit", steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt"):
        rescue_numeric_calculation("Integrate orbit", dt=-0.01)

    store = BeeRetentionStore(base_dir=tmp_path / "bee_store")
    manifest = store.retain_verified_checkpoint(
        node_id="node-1",
        backbone_model="Qwen/Qwen2.5-Coder-7B",
        lora_rank=16,
        adapter_weights_bytes=b"weights-bytes",
        heldout_energy_e=12.5,
        parent_energy_e=20.0,
    )
    assert manifest["promoted_and_retained"] is True
    with pytest.raises(ValueError, match="node_id"):
        store.retain_verified_checkpoint(
            node_id="",
            backbone_model="m",
            lora_rank=16,
            adapter_weights_bytes=b"w",
            heldout_energy_e=12.0,
        )
    with pytest.raises(ValueError, match="lora_rank"):
        store.retain_verified_checkpoint(
            node_id="n",
            backbone_model="m",
            lora_rank=0,
            adapter_weights_bytes=b"w",
            heldout_energy_e=12.0,
        )
    with pytest.raises(ValueError, match="adapter_weights_bytes"):
        store.retain_verified_checkpoint(
            node_id="n",
            backbone_model="m",
            lora_rank=16,
            adapter_weights_bytes=b"",
            heldout_energy_e=12.0,
        )
    with pytest.raises(ValueError, match="heldout_energy_e"):
        store.retain_verified_checkpoint(
            node_id="n",
            backbone_model="m",
            lora_rank=16,
            adapter_weights_bytes=b"w",
            heldout_energy_e=float("nan"),
        )


def test_rusty_sundials_wilson_surrogate_pde_sweep_and_fleet_validation() -> None:
    assert _wilson_lower_bound_95(8, 10) > 0.0
    with pytest.raises(ValueError, match="passes"):
        _wilson_lower_bound_95(-1, 10)
    with pytest.raises(ValueError, match="passes cannot exceed trials"):
        _wilson_lower_bound_95(11, 10)
    with pytest.raises(ValueError, match="z"):
        _wilson_lower_bound_95(5, 10, z=0.0)

    with pytest.raises(ValueError, match="expert_id"):
        SolverTaughtSurrogateExpert(expert_id="")
    sur = SolverTaughtSurrogateExpert()
    with pytest.raises(TypeError, match="NumericRescueResult"):
        sur.fit_from_verified_rescue("astronomy", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps"):
        sur.fit_all_four_domains(steps=0)
    with pytest.raises(ValueError, match="max_surrogate_drift_tol"):
        sur.predict_and_verify("orbit", max_surrogate_drift_tol=-1e-6)

    pde = solve_wave2_method_of_lines_pde("viscous_burgers_imex", grid_points=16, steps=10, dt=0.005)
    assert pde["pde_family"] == "viscous_burgers_imex"
    with pytest.raises(ValueError, match="pde_family"):
        solve_wave2_method_of_lines_pde("")
    with pytest.raises(ValueError, match="grid_points"):
        solve_wave2_method_of_lines_pde("viscous_burgers_imex", grid_points=2)
    with pytest.raises(ValueError, match="steps"):
        solve_wave2_method_of_lines_pde("viscous_burgers_imex", steps=0)
    with pytest.raises(ValueError, match="dt"):
        solve_wave2_method_of_lines_pde("viscous_burgers_imex", dt=0.0)
    with pytest.raises(ValueError, match="param"):
        solve_wave2_method_of_lines_pde("viscous_burgers_imex", param=True)  # type: ignore[arg-type]

    sweep = run_wave2_parameter_sweep_quorum("van_der_pol", parameter_grid=[0.5, 1.0], steps=8, dt=0.05)
    assert sweep["quorum_verified"] is True
    with pytest.raises(ValueError, match="parameter_grid"):
        run_wave2_parameter_sweep_quorum("van_der_pol", parameter_grid=[])
    with pytest.raises(ValueError, match="parameter_grid"):
        run_wave2_parameter_sweep_quorum("van_der_pol", parameter_grid=[0.5, True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="wu_id"):
        run_wave2_parameter_sweep_quorum("van_der_pol", wu_id="")

    orch = OwnerFleetCoServingOrchestrator()
    receipt = orch.distill_teacher_to_rtx2070_student(teacher_rank=32, student_rank=16, verified_trajectories_count=12)
    assert receipt["wddm_compliant"] is True
    with pytest.raises(ValueError, match="student_rank"):
        orch.distill_teacher_to_rtx2070_student(teacher_rank=True, student_rank=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verified_trajectories_count"):
        orch.distill_teacher_to_rtx2070_student(teacher_rank=16, student_rank=8, verified_trajectories_count=0)

    with pytest.raises(ValueError, match="rtol"):
        OwnerFleetCoServingOrchestrator.verify_s1b_cross_cpu_reproducibility(rtol=0.0)
    with pytest.raises(ValueError, match="atol"):
        OwnerFleetCoServingOrchestrator.verify_s1b_cross_cpu_reproducibility(atol=-1e-9)
