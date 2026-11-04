# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The MVP scorecard is the loop's measure: it must not pass without evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_scorecard as sc  # noqa: E402


def test_every_criterion_has_a_row_and_none_passes_without_an_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    rows = {r.cid: r for r in sc.run()}
    assert set(rows) == set(sc.CRITERIA)
    assert all(r.status != "PASS" for r in rows.values())  # an empty repository passes nothing


def test_an_artifact_must_say_passed_true(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    (tmp_path / "artifacts/mvp").mkdir(parents=True)
    (tmp_path / "artifacts/mvp/A1_x.json").write_text(json.dumps({"passed": False, "reason": "failover lost 3 requests"}))
    (tmp_path / "artifacts/mvp/C1_x.json").write_text(json.dumps({"passed": True}))
    assert sc._artifact_rows("A1").status == "FAIL" and "failover lost 3" in sc._artifact_rows("A1").reason
    assert sc._artifact_rows("C1").status == "PASS"
    assert sc._artifact_rows("A2").status == "NOT-RUN"


def test_b3_reads_the_primary_gate_not_the_post_hoc_one() -> None:
    row = sc.check_b3()
    assert row.cid == "B3" and row.status in {"FAIL", "PASS"}
    if row.status == "FAIL":
        assert "servable=False" in row.reason


def test_the_committed_scorecard_matches_the_current_evidence() -> None:
    committed = (Path(__file__).resolve().parents[1] / "docs/MVP_SCORECARD.md")
    if committed.exists():
        assert committed.read_text(encoding="utf-8") == sc.render(sc.run()), "run: python scripts/mvp_scorecard.py --write"
