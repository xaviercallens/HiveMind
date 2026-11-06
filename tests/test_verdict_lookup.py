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
