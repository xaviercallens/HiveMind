# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for hivemind_producer_agent, solverbee families, crowdgpt_loss_check, and queenbee strategies (IP-173)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hivemind.queenbee.strategies import aggregate_shared_updates
from hivemind.router.hivemind_producer_agent import (
    apply_control,
    apply_residency,
    detect_vram_gb,
    fetch_catalog,
    fetch_remote_config,
    filter_models_by_vram_budget,
    parse_model_spec,
    preflight,
    probe_model_sizes_mib,
    probe_residency,
    send_heartbeat,
    write_control_state,
)
from hivemind.solverbee.families import FamilyError, seir_spec, sir_spec
from hivemind.swarm.crowdgpt_loss_check import load_flat_safetensors


def test_hivemind_producer_agent_boundary_validation(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_spec(None)  # type: ignore[arg-type]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_spec("missing-equals")
    assert parse_model_spec("qwen=qwen2.5:3b") == {"id": "qwen", "local_name": "qwen2.5:3b"}

    with pytest.raises(ValueError, match="local_endpoint"):
        probe_residency("")
    with pytest.raises(ValueError, match="timeout_s"):
        probe_residency("http://127.0.0.1:11434", timeout_s=0.0)

    with pytest.raises(ValueError, match="local_endpoint"):
        probe_model_sizes_mib("")
    with pytest.raises(ValueError, match="timeout_s"):
        probe_model_sizes_mib("http://127.0.0.1:11434", timeout_s=-1.0)

    with pytest.raises(TypeError, match="models"):
        filter_models_by_vram_budget("not-a-seq", None, gpu_vram_mib=8192)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model_sizes_mib"):
        filter_models_by_vram_budget([], "not-a-mapping", gpu_vram_mib=8192)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="payload"):
        apply_residency(None, None, cold_start_s=10.0, gpu_vram_mib=8192)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resident"):
        apply_residency({"models": []}, "bad", cold_start_s=10.0, gpu_vram_mib=8192)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cold_start_s"):
        apply_residency({"models": []}, None, cold_start_s=-1.0, gpu_vram_mib=8192)

    with pytest.raises(ValueError, match="path"):
        write_control_state("  ", {"paused": False, "vram_cap_mib": None})
    with pytest.raises(TypeError, match="control"):
        write_control_state(str(tmp_path / "ctrl.json"), None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="payload"):
        apply_control(None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="control"):
        apply_control({"models": []}, "bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="client"):
        send_heartbeat(object(), "https://gw", "key", {})  # type: ignore[arg-type]

    class _DummyClient:
        def post(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("unused")

        def get(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("unused")

    c = _DummyClient()
    with pytest.raises(ValueError, match="gateway"):
        send_heartbeat(c, "", "key", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_key"):
        send_heartbeat(c, "https://gw", "", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="payload"):
        send_heartbeat(c, "https://gw", "key", None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="client"):
        fetch_catalog(object(), "https://gw")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gateway"):
        fetch_catalog(c, "")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="client"):
        fetch_remote_config(object(), "https://gw", "key")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gateway"):
        fetch_remote_config(c, "", "key")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_key"):
        fetch_remote_config(c, "https://gw", "")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="run"):
        detect_vram_gb(run="not-callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="client"):
        preflight(object(), "http://127.0.0.1:11434", None, ["m1"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="endpoint"):
        preflight(c, "", None, ["m1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="local_names"):
        preflight(c, "http://127.0.0.1:11434", None, "not-a-seq")  # type: ignore[arg-type]


def test_solverbee_families_sir_and_seir_pre_subtraction_validation() -> None:
    with pytest.raises(FamilyError, match="s0"):
        sir_spec(0.3, 0.1, "bad", 0.01, [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(FamilyError, match="i0"):
        sir_spec(0.3, 0.1, 0.99, None, [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(FamilyError, match="e0"):
        seir_spec(0.3, 0.2, 0.1, 0.98, "bad", 0.01, [1.0, 2.0])  # type: ignore[arg-type]

    sir = sir_spec(0.3, 0.1, 0.99, 0.01, [1.0, 2.0])
    assert sir["problem"]["kind"] == "mass_action"
    seir = seir_spec(0.3, 0.2, 0.1, 0.98, 0.01, 0.01, [1.0, 2.0])
    assert seir["problem"]["kind"] == "mass_action"


def test_crowdgpt_and_queenbee_strategies_validation() -> None:
    with pytest.raises(ValueError, match="path"):
        load_flat_safetensors("")
    with pytest.raises(ValueError, match="path"):
        load_flat_safetensors(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Mappings"):
        aggregate_shared_updates(
            None,  # type: ignore[arg-type]
            {},
            min_quorum_k=2,
            grace_window_s=1.0,
            max_byzantine_f=0,
        )
    with pytest.raises(TypeError, match="exact_lora_product"):
        aggregate_shared_updates(
            {},
            {},
            min_quorum_k=2,
            grace_window_s=1.0,
            max_byzantine_f=0,
            exact_lora_product="yes",  # type: ignore[arg-type]
        )
