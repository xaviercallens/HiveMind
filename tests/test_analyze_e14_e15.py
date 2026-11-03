"""Unit tests of scripts/analyze_E14_E15_hypotheses.py with synthetic reports (CPU only, fast; no real experiment file is read except in one skippable check)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("analyze_E14_E15_hypotheses", REPO / "scripts" / "analyze_E14_E15_hypotheses.py")
assert _spec is not None and _spec.loader is not None
an = importlib.util.module_from_spec(_spec)
# The script under test sets CUDA_VISIBLE_DEVICES when it is imported; put the value back right after loading it so
# collecting this module does not hide the GPU from the modules collected after it (IP-179).
_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
_spec.loader.exec_module(an)
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


def cell(finals: list[float], initial: float = 3.0, p50: float = 10.0) -> dict[str, Any]:
    return {"seeds": [1, 2, 3][: len(finals)], "initial_loss_per_seed": [initial] * len(finals), "final_loss_per_seed": finals,
            "round_latency_s": {"p50": p50}, "admitted_fraction_mean": 0.5, "late_discards_total": 1}


def model(s2: list[float], s6: list[float], s7: list[float]) -> dict[str, Any]:
    return {"cells": {"S0": cell([2.0, 2.0, 2.0]), "S2": cell(s2), "S6": cell(s6, p50=15.0), "S7": cell(s7)},
            "controls": {"s0_learns": {"passed": True}, "s1_equals_s0": {"passed": None}, "codec_rate_on_target": {"passed": True}}, "controls_all_passed": False}


def test_t_interval_matches_hand_computation() -> None:
    iv = an.t_interval([0.1, 0.2, 0.3])
    assert iv["mean"] == pytest.approx(0.2)
    assert iv["half_width"] == pytest.approx(4.302652729911275 * 0.1 / math.sqrt(3))
    assert iv["interval"][0] == pytest.approx(0.2 - iv["half_width"])
    assert an.t_interval([0.1, 0.2])["decidable"] is False


@pytest.mark.parametrize("iv,status", [
    ({"decidable": True, "interval": [0.01, 0.2]}, "supported"),
    ({"decidable": True, "interval": [-0.2, -0.01]}, "refuted"),
    ({"decidable": True, "interval": [-0.01, 0.2]}, "inconclusive"),
    ({"decidable": True, "interval": [0.0, 0.2]}, "inconclusive"),  # an interval touching 0 is not above 0
    ({"decidable": False}, "inconclusive (needs exactly 3 paired seeds)"),
])
def test_interval_status_branches(iv: dict[str, Any], status: str) -> None:
    assert an.interval_status(iv) == status


def test_e14_supported_refuted_inconclusive_per_cell() -> None:
    # S6 clearly better than S2 (supported), S7 clearly worse (refuted)
    d = an.e14_model_decision(model([2.5, 2.5, 2.5001], [2.2, 2.2001, 2.2002], [2.9, 2.9001, 2.9002]), twin_bit_identical=True)
    assert d["H14a"]["status"] == "supported" and d["H14b"]["status"] == "refuted"
    assert d["H14a"]["S2_minus_cell_per_seed"]["1"] == pytest.approx(0.3)
    # noisy differences around zero: inconclusive
    d2 = an.e14_model_decision(model([2.3, 2.2, 2.25], [2.2, 2.3, 2.25], [2.25, 2.2, 2.3]))
    assert d2["H14a"]["status"] == "inconclusive" and d2["H14b"]["status"] == "inconclusive"


def test_e14_retention_formula_latency_and_controls() -> None:
    d = an.e14_model_decision(model([2.5, 2.5, 2.5], [2.1, 2.1, 2.1], [2.2, 2.2, 2.2]), twin_bit_identical=True)
    # rho = (3.0 - L_cell) / (3.0 - 2.0)
    assert d["retention_vs_S0"]["S6"]["mean"] == pytest.approx(0.9) and d["retention_vs_S0"]["S2"]["mean"] == pytest.approx(0.5)
    assert d["descriptive_costs"]["S6_over_S2_latency"] == pytest.approx(1.5)
    assert d["controls"]["valid"] is True and d["controls"]["not_evaluated"] == ["s1_equals_s0"] and d["controls"]["failed"] == []
    bad = model([2.5] * 3, [2.1] * 3, [2.2] * 3)
    bad["controls"]["s0_learns"]["passed"] = False
    assert an.e14_model_decision(bad)["controls"]["valid"] is False
    assert an.e14_model_decision(model([2.5] * 3, [2.1] * 3, [2.2] * 3), twin_bit_identical=False)["controls"]["valid"] is False


def test_e14_missing_cell_or_seeds_is_not_decided() -> None:
    m = model([2.5] * 3, [2.1] * 3, [2.2] * 3)
    del m["cells"]["S7"]
    assert an.e14_model_decision(m)["H14b"]["status"].startswith("inconclusive")
    m2 = model([2.5, 2.5], [2.1, 2.1], [2.2, 2.2])
    assert an.e14_model_decision(m2)["H14a"]["status"].startswith("inconclusive (needs exactly 3")


def test_mcnemar_exact_values() -> None:
    assert an.mcnemar_exact(0, 0) == 1.0
    assert an.mcnemar_exact(5, 5) == 1.0
    assert an.mcnemar_exact(19, 1) == pytest.approx(4.00543212890625e-05)
    assert an.mcnemar_exact(15, 13) == pytest.approx(0.8505540192127228)
    assert an.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)


@pytest.mark.parametrize("points,p,status", [
    (3.0, 0.049, "supported"), (3.0, 0.05, "inconclusive"), (2.9, 0.001, "inconclusive"), (0.0, 1.0, "inconclusive"),
    (-0.5, 0.2, "inconclusive"), (-3.0, 0.01, "refuted"), (-3.0, 0.05, "inconclusive"),
])
def test_h15a_branches(points: float, p: float, status: str) -> None:
    assert an.h15a_status(points, p) == status


@pytest.mark.parametrize("points,p,status", [
    (5.0, 0.04, "supported"), (5.0, 0.06, "inconclusive"), (4.9, 0.001, "inconclusive"), (0.1, 0.5, "inconclusive"),
    (0.0, 1.0, "refuted"), (-2.0, 0.5, "refuted"),
])
def test_h15b_branches(points: float, p: float, status: str) -> None:
    assert an.h15b_status(points, p) == status


def test_h15d_branches() -> None:
    ok = {"a": "ok", "b": "ok"}
    assert an.h15d_status(ok, {"a": 4.2, "b": 8.0}) == "supported"
    assert an.h15d_status(ok, {"a": 4.2, "b": 8.01}) == "refuted"
    assert an.h15d_status({"a": "ok", "b": "failed: CUDA out of memory"}, {"a": 4.0}) == "refuted"
    assert an.h15d_status({"a": "ok", "b": "failed"}, {"a": 4.0}).startswith("inconclusive")
    assert an.h15d_status(ok, {"a": None}).startswith("inconclusive")


def e15_report(c2_passes: int, c1_passes: int, comps: dict[str, tuple[int, int]], peak: float = 4.4, controls_ok: bool = True) -> dict[str, Any]:
    n = 100

    def c(label: str, passes: int, nn: int = n) -> dict[str, Any]:
        return {"label": label, "n": nn, "passes": passes, "pass_at_1": passes / nn, "wilson95": [0.0, 1.0]}

    conds = {"C0": c("C0", 60), "C1": c("C1", c1_passes), "C2": c("C2", c2_passes), "C3": c("C3", 70), "C0h": c("C0h", 50, 50), "C2h": c("C2h", 51, 50)}

    def comp(key: str, a: str, b: str) -> dict[str, Any]:
        bo, ao = comps[key]
        return {"n": conds[a]["n"], "diff": (conds[b]["passes"] - conds[a]["passes"]) / conds[a]["n"], "b_only": bo, "a_only": ao, "discordant": bo + ao, "mcnemar_p": 0.0}

    return {"conditions": conds, "controls": {"all_passed": controls_ok}, "controls_all_passed": controls_ok,
            "paired_comparisons": {"C2-C0": comp("C2-C0", "C0", "C2"), "C1-C0": comp("C1-C0", "C0", "C1"), "C3-C1": comp("C3-C1", "C1", "C3"),
                                   "C3-C0": comp("C3-C0", "C0", "C3"), "C2h-C0h (HumanEval)": comp("C2h-C0h (HumanEval)", "C0h", "C2h")},
            "stages": {"baseline": {"status": "ok", "meter": {"peak_reserved_gib": peak}}, "controls": {"status": "ok"}}}


ALL = {"C2-C0": (3, 2), "C1-C0": (12, 2), "C3-C1": (4, 3), "C3-C0": (11, 1), "C2h-C0h (HumanEval)": (1, 0)}


def test_e15_report_statuses_and_recomputed_p() -> None:
    # C2 = 70 vs C0 = 60 with 12 vs 2 discordant: +10 points, p = 0.0129 -> supported; C1 = 61: +1 point -> inconclusive
    out = an.build_e15_from(e15_report(70, 61, {**ALL, "C2-C0": (12, 2), "C1-C0": (4, 3)}))
    assert out["valid"] is True and out["H15a"]["status"] == "supported" and out["H15b"]["status"] == "inconclusive"
    assert out["H15a"]["points"] == pytest.approx(10.0) and out["H15a"]["mcnemar_p"] == pytest.approx(an.mcnemar_exact(12, 2))
    assert out["H15d"]["status"] == "supported" and out["H15d"]["max_peak_reserved_gib"] == 4.4
    assert out["H15c_descriptive"]["status"].startswith("descriptive") and out["H15e_descriptive"]["points"] == pytest.approx(2.0)


def test_e15_refuted_branches_and_hurt_by_finetuning() -> None:
    out = an.build_e15_from(e15_report(40, 55, {**ALL, "C2-C0": (1, 21), "C1-C0": (0, 6)}))
    assert out["H15a"]["status"] == "refuted" and out["H15b"]["status"] == "refuted"
    assert out["H15a"]["points"] < 0 and out["H15b"]["points"] < 0


def test_e15_invalid_when_controls_fail_but_computed_word_is_kept() -> None:
    out = an.build_e15_from(e15_report(70, 70, {**ALL, "C2-C0": (12, 2), "C1-C0": (12, 2)}, controls_ok=False))
    assert out["valid"] is False and out["H15a"]["computed_status"] == "supported" and out["H15a"]["status"].startswith("invalid")
    assert out["H15b"]["status"].startswith("invalid")


def test_e15_memory_limit_exceeded_is_refuted() -> None:
    assert an.build_e15_from(e15_report(61, 61, ALL, peak=8.5))["H15d"]["status"] == "refuted"


def test_end_to_end_files_and_receipt(tmp_path: Path) -> None:
    art = tmp_path / "art"
    art.mkdir()
    rep14 = {"models": {"m": model([2.5, 2.5, 2.5001], [2.2, 2.2001, 2.2002], [2.3, 2.3001, 2.2999])}, "receipt_sha256": "r" * 64}
    (art / "E14_report_qwen0.5b.json").write_text(json.dumps({"models": {"qwen0.5b": rep14["models"]["m"]}, "receipt_sha256": "r" * 64}), encoding="utf-8")
    (art / "E15_gwaya_rft_report.json").write_text(json.dumps({**e15_report(70, 70, {**ALL, "C2-C0": (12, 2), "C1-C0": (12, 2)}), "receipt_sha256": "q" * 64}), encoding="utf-8")
    assert an.main(["--artifacts", str(art), "--out-dir", str(tmp_path / "out")]) == 0
    e14 = json.loads((tmp_path / "out" / "E14_hypotheses.json").read_text(encoding="utf-8"))
    e15 = json.loads((tmp_path / "out" / "E15_hypotheses.json").read_text(encoding="utf-8"))
    assert e14["models"]["qwen0.5b"]["H14a"]["status"] == "supported" and e14["sources"]["qwen0.5b"]["receipt_sha256"] == "r" * 64
    assert e15["hypotheses"]["H15b"]["status"] == "supported"
    for doc in (e14, e15):
        assert doc["receipt_sha256"] == an.receipt(doc)


def test_real_reports_give_the_receipted_statuses() -> None:
    """The committed reports (skipped when absent): the registered rules applied to them."""
    art = REPO / "artifacts" / "experiments"
    if not (art / "E15_gwaya_rft_report.json").exists() or not (art / "E14_report_qwen0.5b.json").exists():
        pytest.skip("real E14/E15 reports not in this checkout")
    e14, e15 = an.build_e14(art), an.build_e15(art)
    q = e14["models"]["qwen0.5b"]
    assert (q["H14a"]["status"], q["H14b"]["status"]) == ("supported", "supported")
    assert q["H14a"]["report_check"]["agrees"] and q["H14b"]["report_check"]["agrees"]
    if "granite1b" in e14["models"]:
        g = e14["models"]["granite1b"]
        assert (g["H14a"]["status"], g["H14b"]["status"]) == ("inconclusive", "inconclusive")
    h = e15["hypotheses"]
    assert (h["H15a"]["status"], h["H15b"]["status"], h["H15d"]["status"]) == ("inconclusive", "supported", "supported")
    assert h["H15a"]["report_agrees"] and h["H15b"]["report_agrees"]
