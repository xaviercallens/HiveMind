# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control boundary suite for client_setup, moe_bridge, project_server, and night_moe_page."""

from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.boinc.night_moe_page import _fmt, render_night_moe_html
from hivemind.boinc.real_client.client_setup import Client, prepare_data_dir, toy_app
from hivemind.boinc.real_client.moe_bridge import MoEValidator, _resolve_server_elapsed
from hivemind.boinc.real_client.project_server import ProjectServer


def test_client_setup_prepare_data_dir_and_client_method_boundaries(tmp_path: Path) -> None:
    res = prepare_data_dir(
        str(tmp_path / "data"),
        "http://127.0.0.1:8080/hivemind/",
        str(tmp_path / "src"),
        authenticator="auth123",
        apps=[toy_app("python3")],
    )
    assert res["authenticator"] == "auth123"
    assert res["project_dir"].is_dir()

    with pytest.raises(ValueError, match="data_dir"):
        prepare_data_dir("", "http://127.0.0.1/", tmp_path)
    with pytest.raises(ValueError, match="src_root"):
        prepare_data_dir(tmp_path / "d1", "http://127.0.0.1/", "   ")
    with pytest.raises(ValueError, match="authenticator"):
        prepare_data_dir(tmp_path / "d1", "http://127.0.0.1/", tmp_path, authenticator="  ")
    with pytest.raises(ValueError, match="apps"):
        prepare_data_dir(tmp_path / "d1", "http://127.0.0.1/", tmp_path, apps=[])

    with pytest.raises(ValueError, match="boinc_bin"):
        Client("", tmp_path / "lib", tmp_path / "data", rpc_port=31416)
    with pytest.raises(ValueError, match="lib_dir"):
        Client(tmp_path / "bin", " ", tmp_path / "data", rpc_port=31416)
    with pytest.raises(ValueError, match="data_dir"):
        Client(tmp_path / "bin", tmp_path / "lib", "", rpc_port=31416)

    c = Client(tmp_path / "bin", tmp_path / "lib", tmp_path / "data", rpc_port=31416)
    with pytest.raises(ValueError, match="args must be non-empty strings"):
        c.cmd("")
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        c.cmd("--get_state", timeout=0)
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        c.cmd("--get_state", timeout=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        c.quit(timeout=0)
    with pytest.raises(ValueError, match="timeout must be a finite non-negative number"):
        c.wait_rpc(timeout=-1)


def test_moe_bridge_server_elapsed_and_validator_boundaries(tmp_path: Path) -> None:
    assert _resolve_server_elapsed({}, 8.5) == pytest.approx(8.5)
    assert _resolve_server_elapsed({"server_elapsed_seconds": -1.0}, None) is None
    with pytest.raises(ValueError, match="server_elapsed_seconds"):
        _resolve_server_elapsed({}, 0.0)
    with pytest.raises(ValueError, match="server_elapsed_seconds"):
        _resolve_server_elapsed({}, float("nan"))
    with pytest.raises(ValueError, match="server_elapsed_seconds"):
        _resolve_server_elapsed({}, True)  # type: ignore[arg-type]

    validator = MoEValidator()
    with pytest.raises(TypeError, match="spec must be a dict"):
        validator._ensure("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing required key"):
        validator._ensure({"model_id": "m"})
    with pytest.raises(ValueError, match="model_id"):
        validator._ensure({"model_id": " ", "rank": 8, "alpha": 16, "init_seed": 1})
    with pytest.raises(ValueError, match="rank"):
        validator._ensure({"model_id": "m", "rank": 0, "alpha": 16, "init_seed": 1})
    with pytest.raises(ValueError, match="alpha"):
        validator._ensure({"model_id": "m", "rank": 8, "alpha": 0.0, "init_seed": 1})
    with pytest.raises(ValueError, match="init_seed"):
        validator._ensure({"model_id": "m", "rank": 8, "alpha": 16, "init_seed": True})

    with pytest.raises(ValueError, match="wu_id"):
        validator.validate_train({}, tmp_path / "m.json", tmp_path / "s.pt", "host1")
    with pytest.raises(ValueError, match="meta_path"):
        validator.validate_train({"wu_id": "w1"}, " ", tmp_path / "s.pt", "host1")
    with pytest.raises(ValueError, match="states_path"):
        validator.validate_train({"wu_id": "w1"}, tmp_path / "m.json", "", "host1")
    with pytest.raises(ValueError, match="host"):
        validator.validate_train({"wu_id": "w1"}, tmp_path / "m.json", tmp_path / "s.pt", " ")
    with pytest.raises(ValueError, match="replay_all"):
        validator.validate_train({"wu_id": "w1"}, tmp_path / "m.json", tmp_path / "s.pt", "h1", replay_all="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="server_elapsed_seconds"):
        validator.validate_train({"wu_id": "w1"}, tmp_path / "m.json", tmp_path / "s.pt", "h1", server_elapsed_seconds=0.0)

    with pytest.raises(ValueError, match="wu_id"):
        validator.validate_infer({}, tmp_path / "m.json", {}, "host1")
    with pytest.raises(ValueError, match="meta_path"):
        validator.validate_infer({"wu_id": "w1"}, "", {}, "host1")
    with pytest.raises(TypeError, match="adapter"):
        validator.validate_infer({"wu_id": "w1"}, tmp_path / "m.json", "bad", "host1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host"):
        validator.validate_infer({"wu_id": "w1"}, tmp_path / "m.json", {}, "")
    with pytest.raises(ValueError, match="server_elapsed_seconds"):
        validator.validate_infer({"wu_id": "w1"}, tmp_path / "m.json", {}, "h1", server_elapsed_seconds=-1.0)


def test_project_server_and_night_moe_page_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workdir"):
        ProjectServer("")

    srv = ProjectServer(tmp_path)
    try:
        with pytest.raises(ValueError, match="command_line"):
            srv.create_work({"wu_id": "w1"}, command_line=123)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="flops_est"):
            srv.create_work({"wu_id": "w1"}, flops_est=0.0)
        with pytest.raises(ValueError, match="flops_est"):
            srv.create_work({"wu_id": "w1"}, flops_est=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="extra_inputs"):
            srv.create_work({"wu_id": "w1"}, extra_inputs={"../escape": b"x"})
        with pytest.raises(ValueError, match="extra_inputs"):
            srv.create_work({"wu_id": "w1"}, extra_inputs={"ok.bin": "not-bytes"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="wu_id"):
            srv.output_path(" ", "out.json")
        with pytest.raises(ValueError, match="open_name"):
            srv.output_path("w1", "")
        with pytest.raises(TypeError, match="text must be a string"):
            srv.scheduler_rpc(b"not-str")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="body must be bytes"):
            srv.upload("not-bytes")  # type: ignore[arg-type]
    finally:
        srv.stop()

    assert _fmt(0.125) == "0.125"
    with pytest.raises(ValueError, match="finite float"):
        _fmt(float("nan"))

    valid_report = {
        "base_model": "ibm-granite/granite-3.1-1b-a400m-instruct",
        "alpha": 0.05,
        "bonferroni_alpha": 0.005,
        "n_other_cells": 10,
        "subjects": {},
    }
    html = render_night_moe_html(valid_report)
    assert "NIGHT-MOE-1 results" in html

    with pytest.raises(ValueError, match="base_model"):
        render_night_moe_html({**valid_report, "base_model": "  "})
    with pytest.raises(ValueError, match="alpha"):
        render_night_moe_html({**valid_report, "alpha": 0.0})
    with pytest.raises(ValueError, match="bonferroni_alpha"):
        render_night_moe_html({**valid_report, "bonferroni_alpha": 1.5})
    with pytest.raises(ValueError, match="n_other_cells"):
        render_night_moe_html({**valid_report, "n_other_cells": -1})
    with pytest.raises(ValueError, match="n_other_cells"):
        render_night_moe_html({**valid_report, "n_other_cells": True})
