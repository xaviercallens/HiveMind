from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.boinc.mesh_events import (
    AMCPMeshEventBus,
    AMCPMeshHostView,
    AdapterPointer,
    BOINCChainLedger,
    BOINCMeshEvent,
    TOPIC_VALIDATED,
    publish_validated_and_assimilated_round,
)
from hivemind.boinc.validator_assignment import (
    AcceleratorFingerprint,
    EpsilonTable,
    FallbackPolicy,
    ValidatorRegistry,
    _major_minor,
)
from hivemind.boinc.validator import (
    DeterministicSpotCheckValidator,
    ExecutionClassProfile,
    quantize_fixed_point_q16_state,
    verify_fixed_point_q16_digest,
)
from hivemind.boinc.coordinator_api import (
    BOINCCoordinatorService,
    _anonymous_id,
)
from hivemind.config import HiveMindConfig


# ---------------------------------------------------------------------------
# 1. mesh_events.py
# ---------------------------------------------------------------------------


def test_boinc_mesh_events_pointer_event_ledger_and_bus_validation() -> None:
    sha = "a" * 64
    ptr = AdapterPointer(
        chain_stem="latest_adapter",
        round_index=1,
        adapter_rel_path="adapters/r1.bin",
        adapter_sha256=sha,
        wu_name="wu-1",
        domain="code",
        timestamp_utc="2026-11-08T12:00:00Z",
    )
    assert ptr.round_index == 1

    with pytest.raises(ValueError, match="chain_stem"):
        AdapterPointer(
            chain_stem="",
            round_index=1,
            adapter_rel_path="a",
            adapter_sha256=sha,
            wu_name="w",
            domain="d",
            timestamp_utc="t",
        )
    with pytest.raises(ValueError, match="round_index"):
        AdapterPointer(
            chain_stem="c",
            round_index=True,  # type: ignore[arg-type]
            adapter_rel_path="a",
            adapter_sha256=sha,
            wu_name="w",
            domain="d",
            timestamp_utc="t",
        )
    with pytest.raises(ValueError, match="adapter_sha256"):
        AdapterPointer(
            chain_stem="c",
            round_index=1,
            adapter_rel_path="a",
            adapter_sha256="z" * 64,
            wu_name="w",
            domain="d",
            timestamp_utc="t",
        )

    with pytest.raises(ValueError, match="event_type"):
        BOINCMeshEvent(
            event_id="e1",
            topic=TOPIC_VALIDATED,
            event_type="unknown",
            wu_name="w",
            host_id="h",
            chain_stem="c",
            round_index=0,
            adapter_rel_path="a",
            adapter_sha256=sha,
            domain="d",
            timestamp_utc="t",
            payload_sha256=sha,
        )

    with pytest.raises(TypeError, match="initial_credits"):
        BOINCChainLedger(initial_credits=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="credit account"):
        BOINCChainLedger(initial_credits={"": 10.0})
    with pytest.raises(ValueError, match="credit balance"):
        BOINCChainLedger(initial_credits={"u1": -1.0})
    with pytest.raises(ValueError, match="credit balance"):
        BOINCChainLedger(initial_credits={"u1": True})  # type: ignore[dict-item]

    ledger = BOINCChainLedger(initial_credits={"u1": 15.0})
    with pytest.raises(TypeError, match="AdapterPointer"):
        ledger.record_validated_workunit({})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AdapterPointer"):
        ledger.record_assimilated_workunit({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chain_stem"):
        ledger.latest_adapter_pointer("")

    with pytest.raises(TypeError, match="BOINCChainLedger"):
        AMCPMeshEventBus("bad")  # type: ignore[arg-type]
    bus = AMCPMeshEventBus(ledger)
    with pytest.raises(TypeError, match="callable"):
        bus.subscribe(TOPIC_VALIDATED, "not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="round_index"):
        bus.publish_raw_dict(
            {
                "event_type": "validated",
                "wu_name": "w1",
                "host_id": "h1",
                "chain_stem": "c1",
                "round_index": True,
                "adapter_rel_path": "a",
                "adapter_sha256": sha,
                "domain": "d",
                "timestamp_utc": "t",
            }
        )
    with pytest.raises(ValueError, match="adapter_sha256"):
        bus.publish_raw_dict(
            {
                "event_type": "validated",
                "wu_name": "w1",
                "host_id": "h1",
                "chain_stem": "c1",
                "round_index": 1,
                "adapter_rel_path": "a",
                "adapter_sha256": "g" * 64,
                "domain": "d",
                "timestamp_utc": "t",
            }
        )

    with pytest.raises(ValueError, match="host_id"):
        AMCPMeshHostView("")
    view = AMCPMeshHostView("host-1")
    with pytest.raises(TypeError, match="AMCPMeshEventBus"):
        view.attach("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BOINCMeshEvent"):
        view.on_event({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chain_stem"):
        view.latest_adapter_pointer(" ")

    view.attach(bus)
    ev_v, ev_a = publish_validated_and_assimilated_round(
        ledger=ledger,
        bus=bus,
        wu_name="wu-1",
        host_id="host-1",
        chain_stem="latest_adapter",
        round_index=2,
        adapter_rel_path="adapters/r2.bin",
        adapter_sha256=sha,
        domain="code",
        timestamp_utc="2026-11-08T12:00:00Z",
    )
    assert ev_v.round_index == 2
    assert ev_a.round_index == 2
    assert view.latest_adapter_pointer("latest_adapter") == ptr.__class__(
        chain_stem="latest_adapter",
        round_index=2,
        adapter_rel_path="adapters/r2.bin",
        adapter_sha256=sha,
        wu_name="wu-1",
        domain="code",
        timestamp_utc="2026-11-08T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# 2. validator_assignment.py
# ---------------------------------------------------------------------------


def test_validator_assignment_fingerprint_registry_and_eps_table_validation() -> None:
    assert _major_minor("2.4.1+cu121") == "2.4"
    with pytest.raises(ValueError, match="version"):
        _major_minor("")

    fp = AcceleratorFingerprint("cuda", "T4", "7.5", "2.4", "12.1", "float32")
    assert fp.class_key() == ("cuda", "7.5", "2.4", "float32")
    with pytest.raises(ValueError, match="device_name"):
        AcceleratorFingerprint("cuda", "", "7.5", "2.4", "12.1", "float32")
    with pytest.raises(ValueError, match="compute_capability"):
        AcceleratorFingerprint("cuda", "T4", " ", "2.4", "12.1", "float32")
    with pytest.raises(ValueError, match="dtype"):
        AcceleratorFingerprint.from_local(dtype="")

    with pytest.raises(ValueError, match="FallbackPolicy"):
        ValidatorRegistry(default_fallback="bad")  # type: ignore[arg-type]
    reg = ValidatorRegistry()
    with pytest.raises(ValueError, match="validator_id"):
        reg.register("", fp)
    with pytest.raises(TypeError, match="AcceleratorFingerprint"):
        reg.register("v1", {})  # type: ignore[arg-type]
    reg.register("v1", fp)
    vid, pol, _ = reg.assign(fp, rng_seed=42)
    assert vid == "v1" and pol is None

    with pytest.raises(TypeError, match="worker_fp"):
        reg.assign({}, rng_seed=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng_seed"):
        reg.assign(fp, rng_seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exclude"):
        reg.assign(fp, rng_seed=42, exclude="v1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="same_class_eps"):
        EpsilonTable(same_class_eps=0.0)
    with pytest.raises(ValueError, match="same_class_eps"):
        EpsilonTable(same_class_eps=True)  # type: ignore[arg-type]
    tbl = EpsilonTable()
    with pytest.raises(ValueError, match="eps"):
        tbl.set_eps(fp.class_key(), fp.class_key(), -1e-4)
    with pytest.raises(ValueError, match="eps"):
        tbl.set_eps(fp.class_key(), fp.class_key(), True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. validator.py
# ---------------------------------------------------------------------------


def test_boinc_validator_profile_quantize_and_two_statistic_validation() -> None:
    prof = ExecutionClassProfile()
    assert prof.thread_count == 4
    with pytest.raises(ValueError, match="accelerator_class"):
        ExecutionClassProfile(accelerator_class="")
    with pytest.raises(ValueError, match="torch_build"):
        ExecutionClassProfile(torch_build=" ")
    with pytest.raises(ValueError, match="thread_count"):
        ExecutionClassProfile(thread_count=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deterministic_flags"):
        ExecutionClassProfile(deterministic_flags=1)  # type: ignore[arg-type]

    state = {"w": torch.tensor([0.25, -0.5], dtype=torch.float32)}
    q = quantize_fixed_point_q16_state(state, frac_bits=16)
    assert "w" in q
    with pytest.raises(ValueError, match="frac_bits"):
        quantize_fixed_point_q16_state(state, frac_bits=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state_dict"):
        quantize_fixed_point_q16_state([], frac_bits=16)  # type: ignore[arg-type]

    dig = verify_fixed_point_q16_digest(state, state, frac_bits=16, max_ulp_drift=1)
    assert dig["passed"] is True
    with pytest.raises(ValueError, match="frac_bits"):
        verify_fixed_point_q16_digest(state, state, frac_bits=0)
    with pytest.raises(ValueError, match="max_ulp_drift"):
        verify_fixed_point_q16_digest(state, state, max_ulp_drift=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tolerance_eps"):
        DeterministicSpotCheckValidator(tolerance_eps=0.0)
    with pytest.raises(ValueError, match="tolerance_eps"):
        DeterministicSpotCheckValidator(tolerance_eps=True)  # type: ignore[arg-type]

    val = DeterministicSpotCheckValidator()
    with pytest.raises(TypeError, match="worker_class"):
        val.assign_matching_validator("bad", [prof])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validator_pool"):
        val.assign_matching_validator(prof, ["bad"])  # type: ignore[list-item]

    res = val.evaluate_two_statistic_replay(state, state)
    assert res.passed is True
    with pytest.raises(ValueError, match="base_eps"):
        val.evaluate_two_statistic_replay(state, state, base_eps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cross_thread_eps"):
        val.evaluate_two_statistic_replay(state, state, cross_thread_eps=0.0)
    with pytest.raises(ValueError, match="max_tensor_fraction_above_eps"):
        val.evaluate_two_statistic_replay(state, state, max_tensor_fraction_above_eps=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. coordinator_api.py
# ---------------------------------------------------------------------------


def test_boinc_coordinator_service_validation(tmp_path: Path) -> None:
    cfg = HiveMindConfig()
    lora = {"lora_A": torch.zeros((2, 4), dtype=torch.float32)}

    with pytest.raises(ValueError, match="base_model_sha256"):
        BOINCCoordinatorService(cfg, lora, base_model_sha256="", shards_dir=tmp_path)
    with pytest.raises(ValueError, match="vocab_size"):
        BOINCCoordinatorService(cfg, lora, base_model_sha256="sha", shards_dir=tmp_path, vocab_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workunit_ttl_s"):
        BOINCCoordinatorService(cfg, lora, base_model_sha256="sha", shards_dir=tmp_path, workunit_ttl_s=0.0)

    coord = BOINCCoordinatorService(cfg, lora, base_model_sha256="sha-model", shards_dir=tmp_path)
    with pytest.raises(ValueError, match="now_s"):
        coord.prune_active_workunits(now_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_sequences"):
        coord.assign_workunit("w1", "76561198000000001", num_sequences=0)
    with pytest.raises(ValueError, match="seq_len"):
        coord.assign_workunit("w1", "76561198000000001", seq_len=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng_seed"):
        coord.assign_workunit("w1", "76561198000000001", rng_seed=True)  # type: ignore[arg-type]

    first_branch = next(iter(coord.branches.keys()))
    with pytest.raises(ValueError, match="branch_name"):
        coord.inject_raw_delta_for_aggregation("nonexistent_branch", "w1", lora)
    with pytest.raises(ValueError, match="worker_id"):
        coord.inject_raw_delta_for_aggregation(first_branch, "", lora)
    with pytest.raises(ValueError, match="delta_dict"):
        coord.inject_raw_delta_for_aggregation(first_branch, "w1", {})
    with pytest.raises(ValueError, match="branch_name"):
        coord.step_branch_outer_loop("nonexistent_branch")

    assert len(_anonymous_id("76561198000000001")) == 16
    with pytest.raises(ValueError, match="steam_id64"):
        _anonymous_id("")
