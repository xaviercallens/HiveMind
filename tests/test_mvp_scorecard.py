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


def test_regressing_workunit_record_is_labeled_assimilated_no_held_out_gain(tmp_path: Path) -> None:
    """Issue #633: A synthetic work-unit record where eval_loss_after > eval_loss_before is labeled 'assimilated, no held-out gain'."""
    import t4_monitor
    from hivemind.boinc.real_client.boinc_server_scripts import classify_workunit_eval_gain
    from hivemind.router.t4_visibility import format_workunit_counter_label

    synthetic_wu = {
        "wu_id": "hmq_00181",
        "wu": "hmq_00181",
        "app": "hivemind_qlora_moe",
        "domain": "mix",
        "round": 181,
        "loss_before": 1.664,
        "loss_after": 1.3849,
        "eval_loss_before": 1.8832,
        "eval_loss_after": 2.2889,
    }
    verdict = classify_workunit_eval_gain(synthetic_wu)
    assert verdict["held_out_gain"] < 0
    assert verdict["improved"] is False
    assert verdict["progress_label"] == "assimilated, no held-out gain"
    assert verdict["assimilator_action"] == "counted_in_ledger_not_improvement_chain_head_held"
    assert verdict["eval_basis"]["basis_id"] == "6_domain_mix_stratified"

    b3_row = sc.check_b3(wu_record=synthetic_wu)
    assert b3_row.status == "FAIL"
    assert "assimilated, no held-out gain" in b3_row.reason
    assert "basis=6_domain_mix_stratified" in b3_row.reason

    (tmp_path / "assimilated.jsonl").write_text(json.dumps(synthetic_wu) + "\n", encoding="utf-8")
    prog = t4_monitor.progress(tmp_path)
    assert prog["last"]["progress_label"] == "assimilated, no held-out gain"
    assert prog["assimilated_no_held_out_gain"] == 1
    assert prog["counter_label"] == "assimilated, no held-out gain"
    assert format_workunit_counter_label(synthetic_wu) == "assimilated, no held-out gain"
