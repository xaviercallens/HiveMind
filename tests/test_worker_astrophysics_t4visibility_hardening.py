# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control boundary suite for worker_node, local_astrophysics_boinc, astrophysics_24h_campaign, and t4_visibility."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from hivemind.boinc.astrophysics_24h_campaign import (
    _eval_loss,
    _insert_after_key,
    create_signed_6domain_parquet_shard,
)
from hivemind.boinc.local_astrophysics_boinc import (
    _evaluate_model_loss_on_shard,
    _find_cached_hf_dataset_file,
    create_signed_astrophysics_parquet_shard,
    run_local_boinc_astrophysics_campaign,
)
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.router.t4_visibility import (
    _check_like,
    _require_finite_numbers,
    _validated_section,
)


def test_worker_node_constructor_and_method_boundaries() -> None:
    model = nn.Linear(4, 4)
    sid = "76561198000000001"
    node = HiveMindWorkerNode("worker-1", sid, "Ada", model, enable_error_feedback=True)
    buf = node._error_feedback_for("code", topk_ratio=0.25)
    assert buf is not None
    assert isinstance(node.get_seti_lifecycle_state(False), str)

    with pytest.raises(TypeError, match="model must be an nn.Module"):
        HiveMindWorkerNode("w1", sid, "Ada", "not-a-model")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="steamworks"):
        HiveMindWorkerNode("w1", sid, "Ada", model, steamworks="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="daemon"):
        HiveMindWorkerNode("w1", sid, "Ada", model, daemon="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="branch_name"):
        node._error_feedback_for(" ", topk_ratio=0.25)
    with pytest.raises(ValueError, match="topk_ratio"):
        node._error_feedback_for("code", topk_ratio=0.0)
    with pytest.raises(ValueError, match="topk_ratio"):
        node._error_feedback_for("code", topk_ratio=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="result must be an InnerLoopResult"):
        node._record_ef21_telemetry(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="result must be an InnerLoopResult"):
        node._record_ef21_telemetry("bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="workunit must be a BOINCWorkunit"):
        node.execute_workunit("bad", {})  # type: ignore[arg-type]
    with pytest.raises( ValueError, match="consumer_query_active"):
        node.get_seti_lifecycle_state("yes")  # type: ignore[arg-type]

    toks = torch.ones((2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="domain_name"):
        node.execute_cas_shard(" ", toks, {})
    with pytest.raises(ValueError, match="token_sequences"):
        node.execute_cas_shard("math", torch.ones((4,), dtype=torch.long), {})
    with pytest.raises(TypeError, match="global_lora_state"):
        node.execute_cas_shard("math", toks, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outer_step"):
        node.execute_cas_shard("math", toks, {}, outer_step=0)
    with pytest.raises(ValueError, match="inner_steps_H"):
        node.execute_cas_shard("math", toks, {}, inner_steps_H=0)
    with pytest.raises(ValueError, match="spot_check_interval_s"):
        node.execute_cas_shard("math", toks, {}, spot_check_interval_s=0)
    with pytest.raises(ValueError, match="rng_seed"):
        node.execute_cas_shard("math", toks, {}, rng_seed=-1)
    with pytest.raises(ValueError, match="use_muon_lora_preconditioner"):
        node.execute_cas_shard("math", toks, {}, use_muon_lora_preconditioner="no")  # type: ignore[arg-type]


def test_astrophysics_local_and_24h_campaign_boundaries(tmp_path: Path) -> None:
    assert _find_cached_hf_dataset_file("nonexistent/repo", "data.parquet") is None
    with pytest.raises(ValueError, match="repo_id"):
        _find_cached_hf_dataset_file(" ", "data.parquet")
    with pytest.raises(ValueError, match="filename_suffix"):
        _find_cached_hf_dataset_file("repo/name", "")

    with pytest.raises(ValueError, match="shard_path"):
        create_signed_astrophysics_parquet_shard("", "Kepler_Exoplanet_Transit_Classifier", "ds", 42)
    with pytest.raises(ValueError, match="shard_path"):
        create_signed_6domain_parquet_shard("   ", "Kepler_Exoplanet_Transit_Classifier", "ds", 42)

    dummy_model = nn.Linear(4, 4)
    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        _evaluate_model_loss_on_shard("bad", torch.ones((2, 4), dtype=torch.long), torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_sequences"):
        _evaluate_model_loss_on_shard(dummy_model, torch.ones((4,), dtype=torch.long), torch.device("cpu"))

    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        _eval_loss("bad", torch.ones((2, 4), dtype=torch.long), torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_sequences"):
        _eval_loss(dummy_model, torch.empty((0, 4), dtype=torch.long), torch.device("cpu"))

    with pytest.raises(ValueError, match="output_json_path"):
        run_local_boinc_astrophysics_campaign(output_json_path="   ")

    assert _insert_after_key({"a": 1, "b": 2}, "a", "k", 99) == {"a": 1, "k": 99, "b": 2}
    with pytest.raises(TypeError, match="entry must be a dict"):
        _insert_after_key("bad", "a", "k", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after"):
        _insert_after_key({"a": 1}, " ", "k", 1)
    with pytest.raises(ValueError, match="key"):
        _insert_after_key({"a": 1}, "a", "", 1)


def test_t4_visibility_helper_boundaries() -> None:
    _require_finite_numbers("gpu", {"used_mib": 4096, "temp_c": 65.0}, ("used_mib", "temp_c"))
    with pytest.raises(ValueError, match="section must be a non-empty string"):
        _require_finite_numbers(" ", {}, ("used_mib",))
    with pytest.raises(ValueError, match="values must be a dict"):
        _require_finite_numbers("gpu", "bad", ("used_mib",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields must be a tuple"):
        _require_finite_numbers("gpu", {}, ["used_mib"])  # type: ignore[arg-type]

    out = _validated_section("gpu", {"used_mib": 100}, {"used_mib": 200})
    assert out == {"used_mib": 200}
    with pytest.raises(ValueError, match="section must be a non-empty string"):
        _validated_section("", {"used_mib": 100}, {})
    with pytest.raises(ValueError, match="current and update must be dicts"):
        _validated_section("gpu", "bad", {})  # type: ignore[arg-type]

    _check_like("gpu.used_mib", 100, 250)
    with pytest.raises(ValueError, match="path must be a non-empty string"):
        _check_like("  ", 100, 250)
