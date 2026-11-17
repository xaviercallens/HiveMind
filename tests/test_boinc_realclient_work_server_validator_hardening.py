# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control test suite for BOINC real_client work, server, validator, and stack modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.boinc.real_client import moe_work, project_server, resident_validator, server_stack


def test_moe_work_domain_ids_chunks_and_mix_counts_validation() -> None:
    assert moe_work._chunks([1, 2, 3, 4, 5, 6], 2, 2) == [[1, 2], [3, 4]]
    assert moe_work._chunks([1, 2, 3, 4], 2, 0) == []
    assert moe_work.mix_counts(14, 6) == [3, 3, 2, 2, 2, 2]

    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        moe_work._chunks([1, 2], 0, 1)
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        moe_work._chunks([1, 2], True, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        moe_work._chunks([1, 2], 2, -1)
    with pytest.raises(ValueError, match="n and k must be integers"):
        moe_work.mix_counts(True, 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown domain"):
        moe_work.domain_ids(None, "nonexistent_domain", "train", 0, 128)
    with pytest.raises(ValueError, match="role must be 'train' or 'eval'"):
        moe_work.domain_ids(None, "general", "test", 0, 128)
    with pytest.raises(ValueError, match="shard_index must be a non-negative integer"):
        moe_work.domain_ids(None, "general", "train", -1, 128)
    with pytest.raises(ValueError, match="shard_index must be a non-negative integer"):
        moe_work.domain_ids(None, "general", "train", True, 128)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="need must be a positive integer"):
        moe_work.domain_ids(None, "general", "train", 0, 0)


def test_moe_work_build_infer_spec_validation() -> None:
    spec = moe_work.build_infer_spec("wu-inf-1", ["Hello world"], max_new_tokens=16, rank=8, alpha=16, init_seed=7)
    assert spec["wu_id"] == "wu-inf-1"
    assert spec["max_new_tokens"] == 16

    with pytest.raises(ValueError, match="wu_id must be a non-empty string"):
        moe_work.build_infer_spec("   ", ["Hello"])
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        moe_work.build_infer_spec("wu-1", ["Hello"], model_id="")
    with pytest.raises(ValueError, match="prompts must be a non-empty list of non-empty strings"):
        moe_work.build_infer_spec("wu-1", [])
    with pytest.raises(ValueError, match="prompts must be a non-empty list of non-empty strings"):
        moe_work.build_infer_spec("wu-1", ["ok", "  "])
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        moe_work.build_infer_spec("wu-1", ["Hello"], max_new_tokens=0)
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        moe_work.build_infer_spec("wu-1", ["Hello"], rank=0)
    with pytest.raises(ValueError, match="alpha must be a positive integer"):
        moe_work.build_infer_spec("wu-1", ["Hello"], alpha=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="init_seed must be an integer"):
        moe_work.build_infer_spec("wu-1", ["Hello"], init_seed=True)  # type: ignore[arg-type]


def test_project_server_init_create_work_output_path_and_wait_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback interface"):
        project_server.ProjectServer(tmp_path, host="0.0.0.0")
    with pytest.raises(ValueError, match="port must be an integer"):
        project_server.ProjectServer(tmp_path, port=-1)
    with pytest.raises(ValueError, match="port must be an integer"):
        project_server.ProjectServer(tmp_path, port=70000)
    with pytest.raises(ValueError, match="validate must be callable"):
        project_server.ProjectServer(tmp_path, validate="not-callable")  # type: ignore[arg-type]

    srv = project_server.ProjectServer(tmp_path)
    try:
        wu_id = srv.create_work({"wu_id": "wu_test_1"}, outputs=("out.json", "adapter.pt"), max_nbytes=1024)
        assert wu_id == "wu_test_1"
        assert srv.output_path("wu_test_1", "adapter.pt").name == "wu_test_1_0_1"
        assert srv.wait_all(0.0) is False

        with pytest.raises(ValueError, match="non-empty string wu_id"):
            srv.create_work({"wu_id": "  "})
        with pytest.raises(ValueError, match="app must be a non-empty string"):
            srv.create_work({"wu_id": "wu_2"}, app="")
        with pytest.raises(ValueError, match="max_nbytes must be a positive integer"):
            srv.create_work({"wu_id": "wu_2"}, max_nbytes=0)
        with pytest.raises(ValueError, match="outputs must be a non-empty sequence"):
            srv.create_work({"wu_id": "wu_2"}, outputs=("../escape.json",))
        with pytest.raises(KeyError, match="unknown wu_id"):
            srv.output_path("missing_wu", "out.json")
        with pytest.raises(ValueError, match="unknown output open_name"):
            srv.output_path("wu_test_1", "nonexistent.json")
        with pytest.raises(ValueError, match="timeout must be a finite non-negative number"):
            srv.wait_all(-0.1)
        with pytest.raises(ValueError, match="timeout must be a finite non-negative number"):
            srv.wait_all(float("nan"))
    finally:
        srv.stop()


def test_resident_validator_cache_key_server_and_client_validation(tmp_path: Path) -> None:
    cache = resident_validator.ModelCache(lambda k: k, max_entries=1, max_bytes=1024)
    assert len(cache) == 0

    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        resident_validator.ModelCache(lambda k: k, max_entries=0)
    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        resident_validator.ModelCache(lambda k: k, max_entries=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        resident_validator.ModelCache(lambda k: k, max_bytes=0)
    with pytest.raises(ValueError, match="callable"):
        resident_validator.ModelCache("not_callable")  # type: ignore[arg-type]

    key = resident_validator.model_key({"model_id": "ibm-granite/model", "rank": 8, "alpha": 16, "init_seed": 42})
    assert key == ("ibm-granite/model", "nf4-dq-fp16|r8|a16|s42", "cuda:0")

    with pytest.raises(ValueError, match="spec must be a dict"):
        resident_validator.model_key({"model_id": "m"})
    with pytest.raises(ValueError, match="non-empty string"):
        resident_validator.model_key({"model_id": "  ", "rank": 8, "alpha": 16, "init_seed": 42})
    with pytest.raises(ValueError, match="device must be a non-empty string"):
        resident_validator.model_key({"model_id": "m", "rank": 8, "alpha": 16, "init_seed": 42}, device="")

    with pytest.raises(ValueError, match="request_timeout_s must be a finite positive number"):
        resident_validator.ResidentValidatorServer(tmp_path / "rv.sock", tmp_path, request_timeout_s=0.0)
    with pytest.raises(ValueError, match="io_timeout_s must be a finite positive number"):
        resident_validator.ResidentValidatorServer(tmp_path / "rv.sock", tmp_path, io_timeout_s=-1.0)
    with pytest.raises(ValueError, match="max_message_bytes must be a positive integer"):
        resident_validator.ResidentValidatorServer(tmp_path / "rv.sock", tmp_path, max_message_bytes=0)

    with pytest.raises(ValueError, match="socket_path must be non-empty"):
        resident_validator.ResidentValidatorClient("   ")
    with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
        resident_validator.ResidentValidatorClient(tmp_path / "rv.sock", timeout_s=0.0)
    with pytest.raises(ValueError, match="max_message_bytes must be a positive integer"):
        resident_validator.ResidentValidatorClient(tmp_path / "rv.sock", max_message_bytes=False)  # type: ignore[arg-type]


def test_server_stack_sh_stack_stage_deprecate_and_launcher_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cmd must be a non-empty list"):
        server_stack.sh([])
    with pytest.raises(ValueError, match="cmd must be a non-empty list"):
        server_stack.sh([""])
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        server_stack.sh(["true"], timeout=0)

    with pytest.raises(ValueError, match="port must be an integer"):
        server_stack.Stack(tmp_path, tmp_path, "/usr/bin/python3", tmp_path, port=0)
    with pytest.raises(ValueError, match="alphanumeric/underscore identifier"):
        server_stack.Stack(tmp_path, tmp_path, "/usr/bin/python3", tmp_path, name="bad;drop")
    with pytest.raises(ValueError, match="venv_python must be a non-empty string"):
        server_stack.Stack(tmp_path, tmp_path, "   ", tmp_path)

    st = server_stack.Stack(tmp_path, tmp_path, "/usr/bin/python3", tmp_path, port=8081, name="hmproj_test")
    st.stage("wu_in_json", b'{"ok":true}')
    assert (st.root / "download" / "wu_in_json").read_bytes() == b'{"ok":true}'

    with pytest.raises(ValueError, match="version must be a non-negative integer"):
        st.deprecate_versions_below(-1)
    with pytest.raises(ValueError, match="stage name must be a safe single filename"):
        st.stage("../escape", b"x")
    with pytest.raises(ValueError, match="data must be bytes"):
        st.stage("valid_name", "not-bytes")  # type: ignore[arg-type]

    script = server_stack.launcher("/usr/bin/python3", tmp_path, "hivemind.boinc.real_client.moe_app", ["run"], {"CUDA_VISIBLE_DEVICES": "0"})
    assert "exec /usr/bin/python3 -W ignore -m hivemind.boinc.real_client.moe_app run" in script

    with pytest.raises(ValueError, match="venv_python must be a non-empty string"):
        server_stack.launcher("", tmp_path, "hivemind.boinc.real_client.moe_app", [], {})
    with pytest.raises(ValueError, match="module must be a valid dotted Python module path"):
        server_stack.launcher("/usr/bin/python3", tmp_path, "bad;module", [], {})
