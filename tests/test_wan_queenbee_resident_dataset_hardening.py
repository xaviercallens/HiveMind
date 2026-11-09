"""Negative-control and positive validation tests for wan_fault_benchmark, queenbee.strategies,
boinc.real_client.resident_validator, and data.dataset_registry input hardening."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hivemind.boinc.real_client.resident_validator import (
    ModelCache,
    ResidentValidatorClient,
    _make_validator,
    default_cache,
    forward_validate,
)
from hivemind.data.dataset_registry import (
    SourceSpec,
    _excluded,
    _label_filtered,
    _map_fields,
    _passes_filters,
    entity_aliases,
    map_row,
    normalize_entity_key,
    normalize_tokens,
    text_hash,
)
from hivemind.queenbee.strategies import (
    BanditStrategy,
    FederatedShareStrategy,
    PreferencePairStrategy,
    PrivacyAccountant,
    VerifiedIDECorrectionFlow,
)
from hivemind.swarm.nat_traversal import FaultProfile
from hivemind.swarm.wan_fault_benchmark import (
    FaultScenario,
    StrikeRegister,
    _bbr_pacing,
    _decode_latency,
    _pacing_adaptation,
    _relay_frames,
    _run_quic_leg,
    _run_turn_leg,
    _run_websocket_leg,
    _unused_udp_port,
    run_fault_scenario,
    run_wan_quic_turn_fault_benchmark,
    write_fault_report,
)


def test_wan_fault_benchmark_hardening(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        _unused_udp_port("")
    port = _unused_udp_port("127.0.0.1")
    assert isinstance(port, int) and port > 0

    with pytest.raises(TypeError, match="payload must be bytes"):
        _decode_latency("not-bytes")  # type: ignore[arg-type]
    assert _decode_latency(b"invalid") is None

    scen = FaultScenario(scenario_id="s0", profile=FaultProfile(), frames=2, frame_bytes=64)
    strike = StrikeRegister()

    with pytest.raises(TypeError, match="scenario must be a FaultScenario"):
        asyncio.run(_run_quic_leg("bad", strike, None, None))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strike must be a StrikeRegister"):
        asyncio.run(_run_quic_leg(scen, "bad", None, None))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="proxy must be a DeterministicFaultProxy"):
        asyncio.run(_run_quic_leg(scen, strike, None, "bad"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="sender and receiver must be TurnClient instances"):
        _relay_frames("bad", "bad", ("127.0.0.1", 1234), scen, 1.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="scenario must be a FaultScenario"):
        _run_turn_leg("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        _run_turn_leg(scen, timeout=0.0)
    with pytest.raises(TypeError, match="endpoints must be an ExternalEndpoints or None"):
        _run_turn_leg(scen, endpoints="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="scenario must be a FaultScenario"):
        asyncio.run(_run_websocket_leg("bad", "pem", ("127.0.0.1", 1234)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ca_pem must be a non-empty string"):
        asyncio.run(_run_websocket_leg(scen, "   ", ("127.0.0.1", 1234)))
    with pytest.raises(ValueError, match="timeout and fallback_timeout must be finite positive numbers"):
        asyncio.run(_run_websocket_leg(scen, "pem", ("127.0.0.1", 1234), timeout=-1.0))

    with pytest.raises(TypeError, match="samples must be a sequence of dicts"):
        _bbr_pacing("bad", 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frames_sent and frames_delivered must be non-negative integers"):
        _bbr_pacing([], -1, 0)

    with pytest.raises(TypeError, match="scenario must be a FaultScenario"):
        asyncio.run(run_fault_scenario("bad"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        asyncio.run(run_fault_scenario(scen, timeout=0.0))

    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        asyncio.run(run_wan_quic_turn_fault_benchmark([scen], timeout=0.0))
    with pytest.raises(TypeError, match="scenarios must be a sequence of FaultScenario instances"):
        asyncio.run(run_wan_quic_turn_fault_benchmark(["bad"]))  # type: ignore[list-item]

    with pytest.raises(TypeError, match="results must be a sequence of dicts"):
        _pacing_adaptation("bad")  # type: ignore[arg-type]
    assert _pacing_adaptation([]) == {}

    with pytest.raises(TypeError, match="report must be a dict"):
        write_fault_report([], tmp_path / "r.json")  # type: ignore[arg-type]
    out_p = write_fault_report({"ok": True}, tmp_path / "r.json")
    assert out_p.exists()


def test_queenbee_strategies_hardening(tmp_path: Path) -> None:
    b = BanditStrategy()
    with pytest.raises(TypeError, match="served_by and cost_per_1k must be Mappings"):
        b.dance_records(
            served_by=[],  # type: ignore[arg-type]
            cost_per_1k={},
            epoch=1,
            verifier_kind="unit_tests",
            signer_id="s1",
            secret=b"secret",
        )
    with pytest.raises(ValueError, match="epoch must be a non-negative integer"):
        b.dance_records(
            served_by={},
            cost_per_1k={},
            epoch=-1,
            verifier_kind="unit_tests",
            signer_id="s1",
            secret=b"secret",
        )
    with pytest.raises(ValueError, match="secret must be non-empty bytes"):
        b.dance_records(
            served_by={},
            cost_per_1k={},
            epoch=1,
            verifier_kind="unit_tests",
            signer_id="s1",
            secret=b"",
        )

    pps = PreferencePairStrategy()
    with pytest.raises(ValueError, match="path must be a non-empty Path or string"):
        pps.export_jsonl("   ")
    assert pps.export_jsonl(tmp_path / "pairs.jsonl") == 0

    acc = PrivacyAccountant()
    with pytest.raises(ValueError, match="alpha must be a finite number > 1.0"):
        acc._step_rdp(1.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="clip_norm must be finite and > 0"):
        acc._step_rdp(2.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="sigma must be finite and >= 0"):
        acc._step_rdp(2.0, 1.0, -0.1)

    fss = FederatedShareStrategy(clip_norm=1.0, sigma=0.1)
    with pytest.raises(ValueError, match="delta must be a non-empty Mapping"):
        fss.prepare_share({}, None)

    flow = VerifiedIDECorrectionFlow(opt_in=False)
    with pytest.raises(ValueError, match="prompt_id and cluster_id must be non-empty strings"):
        flow.record_ide_correction("", "c1", "r", "c", verifier_kind="unit_tests", chosen_verified=True)
    with pytest.raises(ValueError, match="chosen_verified must be a boolean"):
        flow.record_ide_correction(
            "p1", "c1", "r", "c", verifier_kind="unit_tests", chosen_verified=1  # type: ignore[arg-type]
        )


def test_resident_validator_hardening(tmp_path: Path) -> None:
    cache = ModelCache(loader=lambda k: k)
    with pytest.raises(TypeError, match="cache must be a ModelCache"):
        _make_validator("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="on_spec must be callable or None"):
        _make_validator(cache, on_spec="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        default_cache(max_entries=0)
    with pytest.raises(ValueError, match="max_bytes must be a positive integer when provided"):
        default_cache(max_entries=1, max_bytes=0)

    with pytest.raises(ValueError, match="socket_path must be a non-empty string"):
        forward_validate("   ", ["f1"])
    with pytest.raises(ValueError, match="args must be a list of strings"):
        forward_validate(str(tmp_path / "sock"), "not-a-list")  # type: ignore[arg-type]


def test_dataset_registry_hardening() -> None:
    with pytest.raises(TypeError, match="row and mapping must be Mappings"):
        _passes_filters([], {})  # type: ignore[arg-type]
    assert _passes_filters({"a": 1}, {}) is True

    with pytest.raises(TypeError, match="prompt and completion must be strings"):
        _excluded(123, "c", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping must be a Mapping"):
        _excluded("p", "c", [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="row and m must be Mappings"):
        _map_fields([], {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="row and m must be Mappings"):
        _label_filtered([], {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="style must be one of 'koi', 'star', or 'raw'"):
        normalize_entity_key("5218.01", "invalid")
    assert normalize_entity_key("5218.01", "koi") == "K05218.01"

    with pytest.raises(ValueError, match="key must be a non-empty string"):
        entity_aliases("   ")
    assert "koi-host-K05218" in entity_aliases("K05218.01")

    spec = SourceSpec(
        dataset_id="org/ds",
        domain="math",
        config="default",
        split="train",
        verified_signal=True,
        signal_basis="unit_tests",
        mapping={"kind": "fields", "prompt_fields": ["q"], "completion_field": "a"},
    )
    with pytest.raises(TypeError, match="spec must be a SourceSpec"):
        map_row("bad", {"q": "1+1", "a": "2"}, "sys", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="row must be a Mapping"):
        map_row(spec, [], "sys", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index must be a non-negative integer"):
        map_row(spec, {"q": "1+1", "a": "2"}, "sys", -1)

    with pytest.raises(TypeError, match="text must be a string"):
        normalize_tokens(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text must be a string"):
        text_hash(None)  # type: ignore[arg-type]
    assert normalize_tokens("Hello, World!") == ["hello", "world"]
