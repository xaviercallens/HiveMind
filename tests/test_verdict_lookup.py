# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The always-on BOINC server has no GPU: it only reads verdicts a GPU host produced. A missing or unreadable verdict must mean "retry later", never "invalid"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivemind.boinc.real_client import boinc_server_scripts as bss


@pytest.fixture()
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(bss, "VERDICT_DIR", tmp_path / "verdicts")
    monkeypatch.setattr(bss, "LEDGER_DIR", tmp_path / "ledger")
    (tmp_path / "verdicts").mkdir()
    return tmp_path


def _write(d: Path, rid: str, body: str) -> None:
    (d / "verdicts" / f"{rid}.json").write_text(body, encoding="utf-8")


def test_no_verdict_yet_is_a_retry_not_a_rejection(dirs: Path) -> None:
    assert bss.lookup_verdict(["a.bin", "b.bin", "1234"]) == bss.RETRY_LATER == 75


def test_a_valid_verdict_passes_and_is_recorded_in_the_ledger(dirs: Path) -> None:
    _write(dirs, "1234", json.dumps({"valid": True, "wu_id": "hmq_00171", "reason": "verified"}))
    assert bss.lookup_verdict(["a.bin", "1234"]) == 0
    assert len(list((dirs / "ledger").glob("verdict_hmq_00171_*.json"))) == 1


def test_an_invalid_verdict_rejects(dirs: Path) -> None:
    _write(dirs, "55", json.dumps({"valid": False, "wu_id": "hmq_1", "reason": "divergence"}))
    assert bss.lookup_verdict(["a.bin", "55"]) == 1


@pytest.mark.parametrize("body", ["", "{not json", "[]", '{"wu_id": "x"}', '{"valid": "yes"}'])
def test_a_partial_or_malformed_verdict_is_a_retry(dirs: Path, body: str) -> None:
    _write(dirs, "7", body)
    assert bss.lookup_verdict(["a.bin", "7"]) == bss.RETRY_LATER


@pytest.mark.parametrize("args", [[], ["a.bin"], ["a.bin", "../../etc"], ["a.bin", "12x"]])
def test_a_bad_result_id_never_reads_outside_the_folder(dirs: Path, args: list[str]) -> None:
    assert bss.lookup_verdict(args) == bss.RETRY_LATER


def test_the_patch_uses_the_same_exit_code() -> None:
    patch = (Path(__file__).resolve().parents[1] / "deploy/boinc_always_on/script_validator_retry.patch").read_text(encoding="utf-8")
    assert f"WEXITSTATUS(retval) == {bss.RETRY_LATER}" in patch and "ERR_OPENDIR" in patch


def test_validate_solver_sweep_and_assimilate_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bss, "SPEC_DIR", tmp_path / "specs")
    monkeypatch.setattr(bss, "LEDGER_DIR", tmp_path / "ledger")
    (tmp_path / "specs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ledger").mkdir(parents=True, exist_ok=True)

    # _trajectory_max_rel_diff edge branches
    assert bss._trajectory_max_rel_diff([[1.0]], [[1.0], [2.0]]) == (float("inf"), float("inf"))
    assert bss._trajectory_max_rel_diff([[1.0]], [[1.0, 2.0]]) == (float("inf"), float("inf"))
    assert bss._trajectory_max_rel_diff([["bad"]], [[1.0]]) == (float("inf"), float("inf"))  # type: ignore[list-item]
    assert bss._trajectory_max_rel_diff([[float("nan")]], [[1.0]]) == (float("inf"), float("inf"))

    spec = {
        "wu_id": "hm_sw_01",
        "app": "hivemind_solver_sweep",
        "family": "van_der_pol",
        "parameter_grid": [0.5],
        "t_out": [0.01],
        "spec_sha256": "sha_abc",
        "max_invariant_drift": 1e-3,
    }
    (tmp_path / "specs" / "hm_sw_01.json").write_text(json.dumps(spec), encoding="utf-8")

    # Compute exact van_der_pol reference step for mu=0.5, dt=0.01 -> [0.99995, -0.01]
    v_valid = bss.validate_solver_sweep(
        spec,
        {
            "wu_id": "hm_sw_01",
            "app": "hivemind_solver_sweep",
            "spec_sha256": "sha_abc",
            "wall_seconds": 0.5,
            "flops_claimed": 1e9,
            "sweep_results": [
                {
                    "status": "ok",
                    "param_value": 0.5,
                    "t": [0.01],
                    "y": [[0.99995, -0.01]],
                    "invariant_drift": 1e-6,
                }
            ],
        },
        "host-1",
    )
    assert v_valid.valid is True

    # S-1 reference tolerance rejection branch
    v_tol_fail = bss.validate_solver_sweep(
        spec,
        {
            "wu_id": "hm_sw_01",
            "app": "hivemind_solver_sweep",
            "spec_sha256": "sha_abc",
            "wall_seconds": 0.5,
            "flops_claimed": 1e9,
            "sweep_results": [
                {
                    "status": "ok",
                    "param_value": 0.5,
                    "t": [0.01],
                    "y": [[0.99995, -0.00999975]],
                    "invariant_drift": 1e-6,
                }
            ],
        },
        "host-1",
    )
    assert v_tol_fail.valid is False

    # Rejection branches in validate_solver_sweep
    assert not bss.validate_solver_sweep(spec, {"wu_id": "other", "app": "hivemind_solver_sweep"}, "h").valid
    assert not bss.validate_solver_sweep(
        spec, {"wu_id": "hm_sw_01", "app": "hivemind_solver_sweep", "spec_sha256": "wrong"}, "h"
    ).valid
    assert not bss.validate_solver_sweep(
        spec, {"wu_id": "hm_sw_01", "app": "hivemind_solver_sweep", "spec_sha256": "sha_abc", "sweep_results": []}, "h"
    ).valid
    assert not bss.validate_solver_sweep(
        spec,
        {
            "wu_id": "hm_sw_01",
            "app": "hivemind_solver_sweep",
            "spec_sha256": "sha_abc",
            "sweep_results": [{}, {}],
        },
        "h",
    ).valid

    # run_validation and assimilate for solver sweep & error/completions
    res_file = tmp_path / "res_sw.json"
    res_file.write_text(
        json.dumps(
            {
                "wu_id": "hm_sw_01",
                "app": "hivemind_solver_sweep",
                "spec_sha256": "sha_abc",
                "wall_seconds": 0.5,
                "flops_claimed": 1e9,
                "sweep_results": [
                    {
                        "status": "ok",
                        "param_value": 0.5,
                        "t": [0.01],
                        "y": [[0.99995, -0.01]],
                        "invariant_drift": 1e-6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert bss.run_validation(["--wu-name", "hm_sw_01", str(res_file), "99"]) == 0
    assert bss.run_validation(["--wu-name=wrong_wu", str(res_file), "99"]) == 2
    assert bss.assimilate(["--error", "4", "101"]) == 0
    assert bss.assimilate(["101", "--wu-name", "hm_sw_01", str(res_file)]) == 0
    assert bss.assimilate(["--wu-name", "mismatch_wu", str(res_file)]) == 1

    # classify_workunit_eval_gain branches
    g1 = bss.classify_workunit_eval_gain(
        {"wu_id": "hmq_00175", "domain": "mix", "eval_loss_before": 2.10, "eval_loss_after": 2.05}
    )
    assert g1["improved"] is True
    assert g1["chain_head_ratchet_pass"] is True

    g2 = bss.classify_workunit_eval_gain(
        {"wu_id": "hmq_00176", "domain": "code", "eval_loss_before": 2.05, "eval_loss_after": 2.09}
    )
    assert g2["improved"] is False
    assert g2["chain_head_ratchet_pass"] is False
