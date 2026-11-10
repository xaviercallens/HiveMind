# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant hardening tests for BOINC server scripts, OpenTrainer, and SolverBee MCP."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.boinc.real_client.boinc_server_scripts import (
    assimilate,
    classify_workunit_eval_gain,
    lookup_verdict,
    run_validation,
    validate,
    validate_solver_sweep,
    wait_for_idle,
)
from hivemind.boinc.real_client.contributor_policy import replay_all_for_result
from hivemind.boinc.real_client.server_stack import copy_tree
from hivemind.router.opentrainer import (
    build_opentrainer_status_payload,
    build_opentrainer_training_round_table,
    compute_default_code_pin,
    is_allowed_opentrainer_campaign_id,
    summarize_moe_self_training_progress,
)
from hivemind.solverbee.mcp import build_server


def test_boinc_server_scripts_validation() -> None:
    with pytest.raises(ValueError, match="quiet_s"):
        wait_for_idle(quiet_s=-1.0)
    with pytest.raises(ValueError, match="max_wait_s"):
        wait_for_idle(max_wait_s=-5.0)
    with pytest.raises(TypeError, match="clock"):
        wait_for_idle(clock="not-callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="spec and meta"):
        validate_solver_sweep("not-a-dict", {}, "host-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host"):
        validate_solver_sweep({}, {}, "   ")
    with pytest.raises(TypeError, match="peer_meta"):
        validate_solver_sweep({}, {}, "host-1", peer_meta="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="args"):
        run_validation("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="log"):
        run_validation(["a.json"], log="not-callable")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="args"):
        validate("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="args"):
        assimilate("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="args"):
        lookup_verdict("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="log"):
        lookup_verdict(["123"], log="not-callable")  # type: ignore[arg-type]

    info = classify_workunit_eval_gain({"wu_id": "hmq_00180", "eval_loss_before": 2.0, "eval_loss_after": 1.9})
    assert info["improved"] is True
    with pytest.raises(TypeError, match="record"):
        classify_workunit_eval_gain("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        classify_workunit_eval_gain({}, tolerance=-1e-3)
    with pytest.raises(ValueError, match="baseline_nll"):
        classify_workunit_eval_gain({}, baseline_nll=0.0)


def test_contributor_policy_and_server_stack_validation(tmp_path: Path) -> None:
    assert replay_all_for_result(123, None, env={}) is False
    with pytest.raises(ValueError, match="result_id"):
        replay_all_for_result(True, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="env"):
        replay_all_for_result(123, None, env="not-a-mapping")  # type: ignore[arg-type]

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "f.txt").write_text("ok", encoding="utf-8")
    copy_tree(src, dst)
    assert (dst / "f.txt").read_text(encoding="utf-8") == "ok"

    with pytest.raises(TypeError, match="src and dst"):
        copy_tree(str(src), dst)  # type: ignore[arg-type]


def test_opentrainer_validation() -> None:
    assert is_allowed_opentrainer_campaign_id("byom-demo-1") is True
    with pytest.raises(TypeError, match="campaign_id"):
        is_allowed_opentrainer_campaign_id(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="registered_account_campaigns"):
        is_allowed_opentrainer_campaign_id("byom-demo-1", registered_account_campaigns="not-a-seq")  # type: ignore[arg-type]

    assert build_opentrainer_training_round_table(None) == []
    with pytest.raises(TypeError, match="ledger_dir"):
        build_opentrainer_training_round_table("/tmp")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prefix"):
        build_opentrainer_training_round_table(None, prefix="../bad")

    pin = compute_default_code_pin()
    assert len(pin) == 64
    with pytest.raises(TypeError, match="script_path"):
        compute_default_code_pin("/tmp/script.py")  # type: ignore[arg-type]

    prog = summarize_moe_self_training_progress(recent_rounds=[], opentrainer_nodes=[], now_utc=1700000000.0)
    assert prog["rounds_recorded"] == 0
    with pytest.raises(TypeError, match="recent_rounds"):
        summarize_moe_self_training_progress(recent_rounds="bad", opentrainer_nodes=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="opentrainer_nodes"):
        summarize_moe_self_training_progress(recent_rounds=[], opentrainer_nodes="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc"):
        summarize_moe_self_training_progress(recent_rounds=[], opentrainer_nodes=[], now_utc=float("nan"))

    status = build_opentrainer_status_payload(
        trainer_accounts=[], opentrainer_nodes=[], recent_rounds=[], now_utc=1700000000.0
    )
    assert status["status"] == "COORDINATOR_UP_NO_TRAINERS"
    with pytest.raises(TypeError, match="trainer_accounts"):
        build_opentrainer_status_payload(trainer_accounts="bad", opentrainer_nodes=[], recent_rounds=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc"):
        build_opentrainer_status_payload(trainer_accounts=[], opentrainer_nodes=[], recent_rounds=[], now_utc=True)  # type: ignore[arg-type]


def test_solverbee_mcp_build_server_validation() -> None:
    with pytest.raises(ValueError, match="binary"):
        build_server("   ")
    with pytest.raises(ValueError, match="problems_dir"):
        build_server("/bin/true", problems_dir="   ")
    with pytest.raises(ValueError, match="timeout_s"):
        build_server("/bin/true", timeout_s=0.0)
    with pytest.raises(ValueError, match="allow_unsandboxed"):
        build_server("/bin/true", allow_unsandboxed="yes")  # type: ignore[arg-type]
