# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The E-STP verdicts follow Amendment A1 exactly (floor, primary fraction or two fractions, guard rail)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _crit():
    spec = importlib.util.spec_from_file_location("estp_criteria", ROOT / "scripts" / "crowdgpt_t4" / "estp_criteria.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write(d: Path, model: str, frac: float, arm: str, accs: list[float], fars: list[float]) -> None:
    for s, (acc, far) in enumerate(zip(accs, fars)):
        (d / f"{model}_f{frac:g}_{arm}_s{s}.json").write_text(json.dumps({"status": "done", "accuracy": acc, "far_loss": far}))


def _full(d: Path, model: str, sft: dict, stp: dict, far_sft=3.2, far_stp=3.2) -> None:
    for f in (1.0, 0.25, 0.0625):
        _write(d, model, f, "sft", sft[f], [far_sft] * 3)
        _write(d, model, f, "stp", stp[f], [far_stp] * 3)


def test_incomplete_until_all_seeds_are_done(tmp_path) -> None:
    c = _crit()
    _write(tmp_path, "crowd", 1.0, "sft", [10, 11], [3.2, 3.2])
    assert c.evaluate_model(tmp_path, "crowd")["verdict"] == "incomplete"


def test_primary_fraction_gain_is_a_success_conditional_on_health(tmp_path) -> None:
    c = _crit()
    sft = {1.0: [30, 31, 29], 0.25: [20, 21, 19], 0.0625: [10, 11, 9]}
    stp = {1.0: [30, 31, 29], 0.25: [20, 21, 19], 0.0625: [15, 16, 14]}
    _full(tmp_path, "crowd", sft, stp)
    r = c.evaluate_model(tmp_path, "crowd")
    assert r["primary_fraction_gain"] and r["verdict"] == "success_conditional_on_health"


def test_one_non_primary_fraction_is_not_enough(tmp_path) -> None:
    c = _crit()
    sft = {1.0: [30, 31, 29], 0.25: [20, 21, 19], 0.0625: [10, 11, 9]}
    stp = {1.0: [36, 37, 35], 0.25: [20, 21, 19], 0.0625: [10, 11, 9]}
    _full(tmp_path, "crowd", sft, stp)
    assert c.evaluate_model(tmp_path, "crowd")["verdict"] == "no_gain"


def test_floor_makes_a_fraction_uninformative_and_guard_rail_can_fail(tmp_path) -> None:
    c = _crit()
    sft = {1.0: [3, 4, 2], 0.25: [2, 3, 1], 0.0625: [1, 2, 0]}
    stp = {1.0: [9, 10, 8], 0.25: [9, 10, 8], 0.0625: [9, 10, 8]}
    _full(tmp_path, "crowd", sft, stp)
    assert c.evaluate_model(tmp_path, "crowd")["verdict"] == "uninformative"
    d2 = tmp_path / "g"
    d2.mkdir()
    sft2 = {1.0: [30, 31, 29], 0.25: [20, 21, 19], 0.0625: [10, 11, 9]}
    stp2 = {1.0: [30, 31, 29], 0.25: [20, 21, 19], 0.0625: [15, 16, 14]}
    _full(d2, "crowd", sft2, stp2, far_sft=3.20, far_stp=3.30)
    assert c.evaluate_model(d2, "crowd")["verdict"] == "fail_guard_rail"


def test_m4_gate(tmp_path) -> None:
    c = _crit()
    win = {"verdict": "success_conditional_on_health"}
    assert c.m4_gate(win, {"verdict": "no_gain"}) == "open_pending_M3"
    assert c.m4_gate({"verdict": "no_gain"}, {"verdict": "incomplete"}) == "undecided"
    assert c.m4_gate({"verdict": "no_gain"}, {"verdict": "fail_guard_rail"}) == "closed"
    assert c.main(["--results", str(tmp_path)]) == 0 and (tmp_path / "criteria_estp.json").exists()
