"""NIGHT-MOE-1 report builder, verdict rule and dashboard routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_night_moe_report as nm  # noqa: E402

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app  # noqa: E402
from hivemind.boinc.night_moe_page import load_night_moe, render_night_moe_html  # noqa: E402


def _s(tag: str, n: int, correct: int, only_a: int | None = None, only_b: int | None = None, p: float | None = None) -> dict:
    out = {"tag": tag, "n": n, "correct": correct, "accuracy": round(correct / n, 4), "wilson95": [0, 1], "unparsable": 0}
    if p is not None:
        out["vs_baseline"] = {"only_this_right": only_a, "only_baseline_right": only_b, "mcnemar_exact_p": p}
    return out


def _bundle(control_gain: bool) -> dict[str, dict]:
    ctrl = _s("a_ctrl_p", 100, 60, 20, 10, 0.03) if control_gain else _s("a_ctrl_p", 100, 30, 5, 30, 0.0001)
    return {
        "base_p": _s("base_p", 100, 50), "base_q": _s("base_q", 100, 40),
        "a_p": _s("a_p", 100, 90, 41, 1, 0.0), "a_ctrl_p": ctrl,
        "x_a_q": _s("x_a_q", 40, 10, 8, 0, 0.0078),
    }


def test_control_that_gains_voids_the_subject_and_a_harming_control_does_not() -> None:
    nm.SUBJECTS.clear()
    nm.SUBJECTS["a"] = "p"
    voided = nm.build(_bundle(control_gain=True))["subjects"]["a"]
    valid = nm.build(_bundle(control_gain=False))["subjects"]["a"]
    assert voided["verdict"] == "VOID" and valid["verdict"] == "VALID"
    assert valid["control"]["flag"] == "HARM" and voided["control"]["flag"] == "GAIN"


def test_delta_comes_from_paired_counts_on_the_scored_subset() -> None:
    nm.SUBJECTS.clear()
    nm.SUBJECTS["a"] = "p"
    cell = nm.build(_bundle(control_gain=False))["subjects"]["a"]["other_packs"]["q"]
    assert cell["delta_vs_base"] == 0.2 and cell["base_accuracy_same_items"] == 0.05
    assert cell["kind"] == "interference" and cell["survives_bonferroni"] is True  # one cell: Bonferroni alpha is 0.05 and p 0.0078 is below it
    assert nm.build(_bundle(control_gain=False))["bonferroni_alpha"] == 0.05


def test_zero_correct_control_is_noted_as_unable_to_gain_and_missing_cells_listed() -> None:
    nm.SUBJECTS.clear()
    nm.SUBJECTS["a"] = "p"
    data = _bundle(control_gain=False)
    data["a_ctrl_p"] = _s("a_ctrl_p", 100, 0, 0, 50, 0.0)
    subj = nm.build(data)["subjects"]["a"]
    assert "cannot gain" in subj["control_note"]
    assert nm.build(_bundle(control_gain=False))["subjects"]["a"]["missing_cells"] == []


def test_dashboard_routes_serve_the_report_and_escape_text(tmp_path: Path) -> None:
    nm.SUBJECTS.clear()
    nm.SUBJECTS["a"] = "p"
    report = nm.build(_bundle(control_gain=False))
    report["subjects"]["a"]["reason"] = "<script>alert(1)</script>"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    client = TestClient(create_hivemind_at_home_app(night_moe_results_path=path))
    api = client.get("/api/v1/night-moe")
    page = client.get("/night-moe")
    assert api.status_code == 200 and api.json()["subjects"]["a"]["verdict"] == "VALID"
    assert page.status_code == 200 and "<script>alert(1)</script>" not in page.text and "&lt;script&gt;" in page.text


def test_missing_results_file_is_a_404_for_the_api_and_a_notice_for_the_page(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert load_night_moe(missing) is None
    client = TestClient(create_hivemind_at_home_app(night_moe_results_path=missing))
    assert client.get("/api/v1/night-moe").status_code == 404
    assert "No results file" in client.get("/night-moe").text and "No results file" in render_night_moe_html(None)


def test_routed_bee_moe_system_self_solution_and_bonferroni_grid_hm_s14_02_03_04() -> None:
    root = Path(__file__).resolve().parents[1]
    stored = json.loads((root / "artifacts" / "experiments" / "NIGHT_MOE_results.json").read_text(encoding="utf-8"))
    routed = nm.evaluate_routed_bee_moe_system(stored)
    assert routed["admitted_adapters"] == ["exoplanet", "stars"]
    assert routed["system_totals"]["gated_routed_correct"] > routed["system_totals"]["ungated_routed_correct"]
    assert routed["system_totals"]["gated_routed_correct"] > routed["routing_plateau"]["correct"]
    assert routed["gated_vs_base_receipt"]["decision"] == "ACCEPT"
    assert routed["gated_vs_best_single_plateau_receipt"]["decision"] == "ACCEPT"
    assert routed["corrupted_misrouted_negative_control_receipt"]["decision"] == "REJECT"

    base_map = {f"k{i}": i < 35 for i in range(50)}
    gold_map = {f"k{i}": i < 20 for i in range(50)}
    self_map = {f"k{i}": i < 47 for i in range(50)}
    cross_map = {f"k{i}": i < 25 for i in range(50)}
    comp = nm.compare_self_solution_vs_gold_adapter(base_map, gold_map, self_map, cross_map, base_unparsable_keys=["k48", "k49"])
    assert comp["supported"] is True
    assert comp["self_solution_gate_receipt"]["decision"] == "ACCEPT"
    assert comp["gold_adapter_gate_receipt"]["decision"] == "REJECT"

    summaries = {
        "base_p": _s("base_p", 100, 50),
        "base_q": _s("base_q", 100, 40),
        "a_p": _s("a_p", 100, 80, 32, 2, 0.0),
        "x_a_q": _s("x_a_q", 100, 65, 27, 2, 0.0),
        "b_q": _s("b_q", 100, 75, 36, 1, 0.0),
        "x_b_p": _s("x_b_p", 100, 51, 3, 2, 1.0),
    }
    grid = nm.evaluate_full_interference_bonferroni_grid(summaries, subjects={"a": "p", "b": "q"})
    assert grid["complete_full_size_grid"] is True and grid["bonferroni_alpha"] == 0.025
    assert grid["bonferroni_survivors"] == ["a->q:GAIN"]
