# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control test suite for BOINC real_client bridge, contributor_policy, moe_profiles, and node_endpoint."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from hivemind.boinc.real_client import bridge, contributor_policy, moe_profiles, node_endpoint


def test_bridge_peak_and_validate_result_validation() -> None:
    assert bridge._peak("Tesla_T4") == 8000.0
    assert bridge._peak("unknown_gpu") == 50.0
    with pytest.raises(ValueError, match="fingerprint_class must be a non-empty string"):
        bridge._peak("  ")

    with pytest.raises(ValueError, match="non-empty string wu_id"):
        bridge.validate_result({}, {}, "host1")
    with pytest.raises(ValueError, match="result must be a dict"):
        bridge.validate_result({"wu_id": "wu1"}, "bad", "host1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        bridge.validate_result({"wu_id": "wu1"}, {}, "  ")
    with pytest.raises(ValueError, match="replay_devices must be a non-empty sequence"):
        bridge.validate_result({"wu_id": "wu1"}, {}, "host1", replay_devices=("tpu",))

    # Missing or malformed fingerprint in result is cleanly rejected without crashing
    spec = {"wu_id": "wu1", "rank": 2, "d_in": 2, "d_out": 2}
    state_b64 = {
        "a": base64.b64encode(np.zeros((2, 2), dtype=np.float32).tobytes()).decode(),
        "b": base64.b64encode(np.zeros((2, 2), dtype=np.float32).tobytes()).decode(),
    }
    res_no_fp = {
        "wu_id": "wu1",
        "app": "hivemind_inner_loop",
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "state_b64": state_b64,
        "fingerprint": {},
    }
    verdict = bridge.validate_result(spec, res_no_fp, "host1", replay_devices=("cpu",))
    assert verdict.valid is False
    assert "missing or invalid fingerprint" in verdict.reason


def test_contributor_policy_night_window_policy_and_host_directory_validation() -> None:
    now_utc = datetime(2026, 10, 26, 22, 0, tzinfo=timezone.utc)
    assert contributor_policy.in_european_night(now_utc) is True

    with pytest.raises(ValueError, match="now must be timezone-aware"):
        contributor_policy.in_european_night("2026-10-26")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tz must be a non-empty timezone name"):
        contributor_policy.in_european_night(now_utc, tz="   ")
    with pytest.raises(ValueError, match="start_h and end_h must be integers"):
        contributor_policy.in_european_night(now_utc, start_h=24)
    with pytest.raises(ValueError, match="start_h and end_h must be integers"):
        contributor_policy.in_european_night(now_utc, end_h=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="probation_valid_results must be a non-negative integer"):
        contributor_policy.ContributorPolicy(probation_valid_results=-1)
    with pytest.raises(ValueError, match="owner_user_ids must contain non-negative integers"):
        contributor_policy.ContributorPolicy(owner_user_ids=frozenset({-1}))

    pol = contributor_policy.ContributorPolicy(probation_valid_results=3, owner_user_ids=frozenset({1}))
    assert pol.replay_all(1, 0) is False
    assert pol.replay_all(2, 2) is True
    assert pol.replay_all(2, 3) is False
    with pytest.raises(ValueError, match="valid_results must be a non-negative integer"):
        pol.replay_all(2, -1)
    with pytest.raises(ValueError, match="user_id must be None or a non-negative integer"):
        pol.replay_all(True, 0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sql must be callable"):
        contributor_policy.BoincHostDirectory("not-callable")  # type: ignore[arg-type]

    hd = contributor_policy.BoincHostDirectory(lambda q: "10 5\n")
    assert hd.lookup(True) == (None, 0)
    assert hd.result_workunit_and_elapsed(False) == (None, None)
    assert hd.workunit_name(True) is None


def test_moe_profiles_validation() -> None:
    assert moe_profiles.profile_for_app("hivemind_qlora_moe")["rank"] == 8
    assert moe_profiles.profiles_for_gpu(8192) == ["standard"]
    assert moe_profiles.profiles_for_gpu(16384) == ["big", "standard"]

    with pytest.raises(ValueError, match="app must be a non-empty string"):
        moe_profiles.profile_for_app("   ")
    with pytest.raises(KeyError, match="not a training app"):
        moe_profiles.profile_for_app("unknown_app")
    with pytest.raises(ValueError, match="profile must be a non-empty string"):
        moe_profiles.flops_estimate("")
    with pytest.raises(ValueError, match="total_mib must be a non-negative integer"):
        moe_profiles.profiles_for_gpu(-1)
    with pytest.raises(ValueError, match="total_mib must be a non-negative integer"):
        moe_profiles.profiles_for_gpu(True)  # type: ignore[arg-type]


def test_node_endpoint_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="client must be a non-empty string"):
        node_endpoint.log_request("  ", "%s", "ok")
    with pytest.raises(ValueError, match="fmt must be a string"):
        node_endpoint.log_request("127.0.0.1", 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty string model_id"):
        node_endpoint.NodeEndpoint({}, tmp_path / "a.pt")
    with pytest.raises(ValueError, match="port must be an integer"):
        node_endpoint.NodeEndpoint({"model_id": "m"}, tmp_path / "a.pt", port=-1)
    with pytest.raises(ValueError, match="served_name must be a non-empty string"):
        node_endpoint.NodeEndpoint({"model_id": "m"}, tmp_path / "a.pt", served_name="  ")

    dummy = object.__new__(node_endpoint.NodeEndpoint)
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        dummy.complete("   ", 16)
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        dummy.complete("Hello", 0)
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        dummy.complete("Hello", True)  # type: ignore[arg-type]
