"""Unit tests of scripts/build_site_data.py with synthetic artifacts in tmp_path (CPU only, no real experiment file is read)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_site_data", REPO / "scripts" / "build_site_data.py")
assert _spec is not None and _spec.loader is not None
bsd = importlib.util.module_from_spec(_spec)
# The script under test sets CUDA_VISIBLE_DEVICES when it is imported; put the value back right after loading it so
# collecting this module does not hide the GPU from the modules collected after it (IP-179).
_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
_spec.loader.exec_module(bsd)
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE

UTC = "2026-01-02T03:04:05Z"
RECEIPT = "abcdef0123456789" * 4


def cell(name: str, finals: list[float], initial: float = 3.0, adm: float = 1.0) -> dict[str, Any]:
    return {"name": name, "status": "ok", "seeds": [1, 2, 3][: len(finals)], "initial_loss_per_seed": [initial] * len(finals), "final_loss_per_seed": finals,
            "per_domain_final_mean": {"code": 2.0, "wiki": 3.0}, "admitted_fraction_mean": adm, "late_discards_total": 0,
            "wire": {"compression_ratio_actual": 11.0, "compression_ratio_ideal": 16.0, "int16_posts_total": 0, "secret_extra": 1},
            "round_latency_s": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "n": 9, "note": "descriptive only"}}


def write(art: Path, name: str, doc: dict[str, Any]) -> None:
    art.mkdir(parents=True, exist_ok=True)
    (art / name).write_text(json.dumps(doc), encoding="utf-8")


def e13_report(models: dict[str, Any], receipt: str = RECEIPT) -> dict[str, Any]:
    return {"schema_version": "x", "models": models, "receipt_sha256": receipt, "controls_all_passed": True}


def model_with_cells(cells: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"cells": cells, "controls": {"s0_learns": {"passed": True}, "s1_equals_s0": {"passed": None, "rows": []}}, "controls_all_passed": True,
            "paired_vs_S0": {}, **extra}


def test_empty_directory_gives_available_false_everywhere(tmp_path: Path) -> None:
    data = bsd.build(tmp_path, UTC)
    assert data["e13"]["registered"]["available"] is False
    assert data["e13"]["rotating_straggler"]["available"] is False
    assert data["e13"]["m2_process_identity"]["available"] is False
    assert data["e14"]["available"] is False and data["e15"]["available"] is False
    assert data["e14"]["prereg"]["available"] is False
    assert data["generated_utc"] == UTC and data["schema"] == bsd.SCHEMA


def test_e13_registered_copies_statuses_verbatim_and_tags_receipt(tmp_path: Path) -> None:
    pred = {"source_report": "E13_swarm_local_report.json", "source_report_receipt": RECEIPT, "receipt_sha256": "0123456789abcdeffedcba" + "0" * 42, "rules": "R",
            "models": {"qwen0.5b": {"valid": True, "controls_failed": [], "P13a": {"status": "supported"},
                                    "P13b": {"status": "refuted", "mean": -0.05, "half_width": float("nan"), "interval": [-0.4, 0.3], "bar": 0.5},
                                    "P13c": {"status": "supported", "upload_ratio_actual": 0.09}, "not_a_field": 1}}}
    write(tmp_path, "E13_swarm_local_predictions.json", pred)
    write(tmp_path, "E13_swarm_local_report.json", e13_report({"qwen0.5b": model_with_cells({"S0": cell("sync", [2.2, 2.2, 2.2]), "S2": cell("k2", [2.4, 2.4, 2.3], adm=0.5)})}))
    blk = bsd.build(tmp_path, UTC)["e13"]["registered"]
    assert blk["available"] is True and blk["source"] == "E13_swarm_local_predictions.json" and blk["receipt12"] == "0123456789ab"
    assert blk["generated_utc"] == UTC and blk["report"]["receipt12"] == RECEIPT[:12]
    m = blk["models"]["qwen0.5b"]
    assert m["P13a"]["status"] == "supported" and m["P13b"]["status"] == "refuted" and m["P13b"]["mean"] == -0.05
    assert m["P13b"]["half_width"] is None  # NaN is not valid JSON
    assert "not_a_field" not in m
    assert m["report_cells"]["S2"]["admitted_fraction_mean"] == 0.5 and "secret_extra" not in m["report_cells"]["S2"]["wire"]


def test_rotating_straggler_derives_retention_with_flag_and_lists_unevaluated_controls(tmp_path: Path) -> None:
    s0, s2 = cell("sync", [2.0, 2.0, 2.0], initial=3.0), cell("k2", [2.5, 2.5, 2.5], initial=3.0, adm=0.5)
    paired = {"S2": {"final_loss_minus_S0_per_seed": [0.5, 0.5, 0.5], "mean": 0.5, "ci95": 0.1, "t_crit_df2": 4.3, "ci_excludes_0": True, "direction": "worse than S0"}}
    write(tmp_path, "E13_swarm_local_report_rotstraggle.json",
          {**e13_report({"qwen0.5b": {**model_with_cells({"S0": s0, "S2": s2}), "paired_vs_S0": paired}}), "controls_all_passed": False, "smoke": False})
    blk = bsd.build(tmp_path, UTC)["e13"]["rotating_straggler"]
    assert blk["available"] and blk["controls_all_passed"] is False and blk["status"] == "exploratory diagnostic, decides nothing" and "campaign_report" in blk["status_source"]
    m = blk["models"]["qwen0.5b"]
    assert m["paired_S2_minus_S0"]["direction"] == "worse than S0"
    assert m["controls_not_evaluated"] == ["s1_equals_s0"]
    der = m["retention_S2_derived"]
    assert der["derived"] is True and der["per_seed"]["1"] == pytest.approx(0.5) and der["mean"] == pytest.approx(0.5)


def test_m2_process_identity_block(tmp_path: Path) -> None:
    ident = {"rows": [{"max_abs_diff": 0.0, "bit_identical": True, "passed": True}], "tolerance": 1e-6, "passed": True}
    model = model_with_cells({"S0": cell("sync", [2.4], initial=2.5), "S1": cell("s1", [2.4], initial=2.5)})
    model["controls"]["s1_equals_s0"] = ident
    write(tmp_path, "E13_swarm_local_report_m2proc.json", e13_report({"qwen0.5b": model}))
    blk = bsd.build(tmp_path, UTC)["e13"]["m2_process_identity"]
    assert blk["available"] and blk["models"]["qwen0.5b"]["s1_equals_s0"]["passed"] is True and blk["receipt12"] == RECEIPT[:12]
    assert blk["status"] == "identity control only" and "campaign_report" in blk["status_source"]


def test_e14_absent_keeps_prereg_but_no_numbers(tmp_path: Path) -> None:
    write(tmp_path, "E14_swarm_fairness_prereg.json", {"registered_utc": "t", "status": "exploratory", "hypotheses_and_rules": {"H14a": "text"}, "scope": "s", "motivation": "ignored"})
    blk = bsd.build(tmp_path, UTC)["e14"]
    assert blk["available"] is False and "models" not in blk
    assert blk["prereg"]["available"] and blk["prereg"]["hypotheses_and_rules"] == {"H14a": "text"} and "motivation" not in blk["prereg"]


def test_e14_present_maps_direction_to_registered_status(tmp_path: Path) -> None:
    paired = {"S6": {"S2_minus_cell_final_loss_per_seed": [0.1, 0.1, 0.1], "mean": 0.1, "ci95": 0.05, "t_crit_df2": 4.3, "ci_excludes_0": True, "direction": "cell better than S2"},
              "S7": {"status": "needs exactly 3 paired seeds with S2 (have 2)", "seeds": [1, 2]}}
    model = model_with_cells({"S0": cell("sync", [2.0] * 3), "S2": cell("k2", [2.5] * 3), "S6": cell("fair", [2.1] * 3), "S7": cell("def", [2.2] * 3)},
                             e14={"cells": ["S6", "S7"], "paired_vs_S2": paired, "retention_vs_S0": {"S6": {"mean": 0.8}}, "note": "exploratory"})
    write(tmp_path, "E14_report_qwen0.5b.json", e13_report({"qwen0.5b": model}))
    blk = bsd.build(tmp_path, UTC)["e14"]
    assert blk["available"] and blk["source"] == ["E14_report_qwen0.5b.json"] and blk["receipt12"] == [RECEIPT[:12]]
    h = blk["models"]["qwen0.5b"]["e14"]["hypotheses"]
    assert h["H14a"]["direction"] == "cell better than S2" and h["H14a"]["hypothesis_status"] == "supported"
    assert "hypothesis_status" not in h["H14b"] and h["H14b"]["status"].startswith("needs exactly 3")
    assert blk["models"]["qwen0.5b"]["e14"]["retention_vs_S0"] == {"S6": {"mean": 0.8}}


@pytest.mark.parametrize("direction,status", [("cell worse than S2", "refuted"), ("within CI of S2", "inconclusive")])
def test_e14_direction_mapping(direction: str, status: str) -> None:
    model = model_with_cells({}, e14={"cells": ["S6"], "paired_vs_S2": {"S6": {"direction": direction, "mean": -0.2}}})
    out = bsd.e14_model({"models": {"m": model}}, "m")
    assert out is not None and out["e14"]["hypotheses"]["H14a"]["hypothesis_status"] == status


def test_e15_absent_and_present(tmp_path: Path) -> None:
    write(tmp_path, "E15_gwaya_rft_prereg.json", {"registered_utc": "t", "title": "T", "hypotheses": {"H15a": "x"}, "scope": "s"})
    assert bsd.build(tmp_path, UTC)["e15"]["available"] is False
    assert bsd.build(tmp_path, UTC)["e15"]["prereg"]["title"] == "T"
    write(tmp_path, "E15_gwaya_rft_report.json",
          {"receipt_sha256": RECEIPT, "status": "complete", "smoke": True, "model": "tiny", "controls_all_passed": True, "private": 1, "notes": ["Numbers only"],
           "conditions": {"C0": {"label": "C0", "n": 10, "passes": 4, "pass_at_1": 0.4, "wilson95": [0.17, 0.69], "hidden": 1}},
           "paired_comparisons": {"C2-C0": {"n": 10, "diff": 0.1, "b_only": 2, "a_only": 1, "discordant": 3, "mcnemar_p": 1.0, "x": 0}}})
    blk = bsd.build(tmp_path, UTC)["e15"]
    assert blk["available"] and blk["receipt12"] == RECEIPT[:12]
    assert blk["run_status"] == "complete" and blk["smoke"] is True and "private" not in blk["report_excerpt"]
    assert blk["conditions"]["C0"] == {"label": "C0", "n": 10, "passes": 4, "pass_at_1": 0.4, "wilson95": [0.17, 0.69]}
    assert blk["paired_comparisons"]["C2-C0"]["mcnemar_p"] == 1.0 and "x" not in blk["paired_comparisons"]["C2-C0"]
    assert "H15" in blk["verdicts_note"] and "not computed here" in blk["verdicts_note"]  # no verdict is ever invented


def test_unreadable_or_non_object_json_is_treated_as_missing(tmp_path: Path) -> None:
    (tmp_path / "E13_swarm_local_predictions.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "E13_swarm_local_report_m2proc.json").write_text("[1, 2]", encoding="utf-8")
    data = bsd.build(tmp_path, UTC)
    assert data["e13"]["registered"]["available"] is False and data["e13"]["m2_process_identity"]["available"] is False


def test_file_without_receipt_gets_file_hash_tag(tmp_path: Path) -> None:
    write(tmp_path, "E13_swarm_local_report_m2proc.json", {"models": {}})
    blk = bsd.build(tmp_path, UTC)["e13"]["m2_process_identity"]
    assert blk["receipt12"] is None and len(blk["file_sha256_12"]) == 12


def test_clean_and_render_are_strict_json_and_js_wrapper(tmp_path: Path) -> None:
    assert bsd.clean({"a": [float("inf"), 1.0, float("nan")]}) == {"a": [None, 1.0, None]}
    text = bsd.render(bsd.build(tmp_path, UTC))
    assert text.startswith("// Generated") and "window.HM_DATA = {" in text and text.rstrip().endswith("};")
    payload = text.split("window.HM_DATA = ", 1)[1].rstrip().rstrip(";")
    assert json.loads(payload)["generated_utc"] == UTC
    assert not math.isnan(0.0)


def test_main_writes_file_and_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    art, out = tmp_path / "art", tmp_path / "site" / "data.js"
    art.mkdir()
    assert bsd.main(["--artifacts", str(art), "--out", str(out), "--utc", UTC]) == 0
    first = out.read_text(encoding="utf-8")
    assert bsd.main(["--artifacts", str(art), "--out", str(out), "--utc", UTC]) == 0
    assert out.read_text(encoding="utf-8") == first
    assert "wrote" in capsys.readouterr().out


def test_real_artifacts_build_without_inventing_numbers() -> None:
    """Smoke test on the repository's own artifacts (skipped when the directory is not present): every numeric leaf of the P13b block equals the source."""
    art = REPO / "artifacts" / "experiments"
    src = art / "E13_swarm_local_predictions.json"
    if not src.exists():
        pytest.skip("no real E13 predictions in this checkout")
    data = bsd.build(art, UTC)["e13"]["registered"]
    raw = json.loads(src.read_text(encoding="utf-8"))
    for name, m in raw["models"].items():
        if "P13b" in m and "mean" in m["P13b"]:
            assert data["models"][name]["P13b"]["mean"] == m["P13b"]["mean"]
            assert data["models"][name]["P13b"]["status"] == m["P13b"]["status"]


def test_new_blocks_absent_on_empty_directory(tmp_path: Path) -> None:
    data = bsd.build(tmp_path / "experiments", UTC)
    assert data["e13d"]["available"] is False and data["e16"]["available"] is False and data["gwaya_lite_live"]["available"] is False
    assert data["e16"]["prereg"]["available"] is False


def test_e13d_uses_m6_files_and_copies_status(tmp_path: Path) -> None:
    write(tmp_path, "E13_swarm_local_predictions_m6.json", {"source_report": "E13_swarm_local_report_m6.json", "source_report_receipt": RECEIPT, "receipt_sha256": RECEIPT,
                                                           "models": {"qwen0.5b": {"valid": True, "P13b": {"status": "inconclusive", "mean": 0.29, "half_width": 0.39, "interval": [-0.1, 0.68], "bar": 0.5}}}})
    write(tmp_path, "E13_swarm_local_report_m6.json", e13_report({"qwen0.5b": model_with_cells({"S0": cell("sync", [2.6] * 3), "S5": cell("noq", [2.62] * 3)})}))
    blk = bsd.build(tmp_path, UTC)
    assert blk["e13"]["registered"]["available"] is False  # the registered E13 files are separate
    d = blk["e13d"]
    assert d["available"] and d["models"]["qwen0.5b"]["P13b"]["status"] == "inconclusive" and d["models"]["qwen0.5b"]["P13b"]["mean"] == 0.29
    assert "replication" in d["status"] and "S5" in d["models"]["qwen0.5b"]["report_cells"]


def test_e16_block_copies_statuses_and_numbers_verbatim(tmp_path: Path) -> None:
    tier = {"R4096": {"label": "R", "n": 10, "passes": 6, "pass_at_1": 0.6, "wilson95": [0.3, 0.8], "accepted_gwaya": 6, "extra": 1},
            "L4096": {"label": "L", "n": 10, "passes": 8, "pass_at_1": 0.8, "wilson95": [0.5, 0.9], "accepted_gwaya": 8},
            "paired_L_minus_R": {"n": 10, "diff": 0.2, "b_only": 2, "a_only": 0, "mcnemar_p": 0.5, "x": 1},
            "p2_gate_calibration": {"precision": 0.9, "precision_wilson95": [0.7, 0.97], "recall": 1.0, "verified": 9, "status_H16e": "supported", "tp": 8},
            "determinism": {"fraction_identical": 1.0}}
    h = {"H16a": {"4096": {"measured_mib": 5483, "threshold_mib": 6656, "status": "supported"}}, "H16c": {"7b": "supported"}, "H16d": {"7b": "inconclusive"}}
    write(tmp_path, "E16_gwaya_lite_t4_report.json", {"receipt_sha256": RECEIPT, "status": "complete", "smoke": False, "run_valid": True, "n_problems": 10,
                                                      "environment": {"ollama_version": "0.34.1", "python": "x"}, "analysis": {"H16": h, "tiers": {"7b": tier}, "escalation_ladder": {"n": 10}},
                                                      "probes": [{"key": "7b:4096", "tier": "7b", "ctx": 4096, "measured_mib": 5483, "estimator_mib": 5290, "ratio_measured_over_estimate": 1.04,
                                                                  "decode_tps_median": 28.5, "prompt_eval_tps_median": 812.0, "gpu_pct": 100.0, "secret": 1}],
                                                      "controls": {"idle_gpu": {"pass": True, "x": 1}}, "notes": ["Numbers only"]})
    blk = bsd.build(tmp_path, UTC)["e16"]
    assert blk["available"] and blk["receipt12"] == RECEIPT[:12] and blk["run_valid"] is True and blk["hypotheses"] == h
    t = blk["tiers"]["7b"]
    assert "extra" not in t["R4096"] and t["paired_L_minus_R"] == {"n": 10, "diff": 0.2, "b_only": 2, "a_only": 0, "mcnemar_p": 0.5}
    assert t["p2_gate_calibration"]["status_H16e"] == "supported" and "tp" not in t["p2_gate_calibration"]
    assert blk["probes"][0]["measured_mib"] == 5483 and "secret" not in blk["probes"][0] and blk["controls"] == {"idle_gpu": True}


def test_live_check_block_found_in_artifacts_dir_or_its_parent(tmp_path: Path) -> None:
    art = tmp_path / "artifacts" / "experiments"
    art.mkdir(parents=True)
    doc = {"receipt_sha256": RECEIPT, "url": "http://127.0.0.1:8099", "all_ok": True, "tasks": [{"task": "t", "verdict": "VERIFIED"}], "untested_request": {"verdict": "AUDITED"}, "other": 1}
    (tmp_path / "artifacts" / "gwaya_lite_live_check.json").write_text(json.dumps(doc), encoding="utf-8")
    blk = bsd.build(art, UTC)["gwaya_lite_live"]
    assert blk["listen_address"] == "127.0.0.1:8099" and "url" not in blk and "http" not in json.dumps(blk)
    assert blk["available"] and blk["all_ok"] is True and blk["tasks"][0]["verdict"] == "VERIFIED" and blk["untested_request"]["verdict"] == "AUDITED" and "other" not in blk
    assert blk["receipt12"] == RECEIPT[:12]


def test_registered_hypotheses_from_analysis_files(tmp_path: Path) -> None:
    write(tmp_path, "E14_report_qwen0.5b.json", e13_report({"qwen0.5b": model_with_cells({"S0": cell("sync", [2.0] * 3)}, e14={"cells": ["S6"], "paired_vs_S2": {}})}))
    write(tmp_path, "E14_hypotheses.json", {"receipt_sha256": RECEIPT, "models": {"qwen0.5b": {"H14a": {"cell": "S6", "status": "supported", "mean": 0.12, "half_width": 0.05, "interval": [0.07, 0.17], "junk": 1},
                                                                                    "H14b": {"cell": "S7", "status": "inconclusive", "mean": 0.0, "half_width": 0.1, "interval": [-0.1, 0.1]}}}})
    write(tmp_path, "E15_gwaya_rft_report.json", {"receipt_sha256": RECEIPT, "status": "complete", "stages": {"rft": {"problems": 462, "problems_with_accepted": 359, "mean_yield": 0.54, "kept_pairs": 673,
                                                                                                            "unverified_samples": 0, "meter": {"wall_s": 1.0, "peak_reserved_gib": 4.4, "energy_wh": 2.0, "x": 1}}}})
    write(tmp_path, "E15_hypotheses.json", {"receipt_sha256": RECEIPT, "hypotheses": {"valid": True, "H15a": {"status": "inconclusive", "points": 0.78, "mcnemar_p": 0.85, "b_only": 15, "a_only": 13, "z": 1},
                                                                                     "H15b": {"status": "supported", "points": 7.0, "mcnemar_p": 4e-5}, "H15d": {"status": "supported", "max_peak_reserved_gib": 4.47, "limit_gib": 8.0},
                                                                                     "H15c_descriptive": {"C3_minus_C1": {"points": 1.2, "mcnemar_p": 0.66}, "C3_minus_C0": {"points": 8.2, "mcnemar_p": 2e-4}},
                                                                                     "H15e_descriptive": {"points": 0.6, "mcnemar_p": 1.0}}})
    data = bsd.build(tmp_path, UTC)
    r14 = data["e14"]["registered_hypotheses"]
    assert r14["available"] and r14["receipt12"] == RECEIPT[:12] and r14["models"]["qwen0.5b"]["H14a"]["status"] == "supported" and "junk" not in r14["models"]["qwen0.5b"]["H14a"]
    r15 = data["e15"]["registered_hypotheses"]
    assert r15["hypotheses"]["H15a"] == {"status": "inconclusive", "points": 0.78, "mcnemar_p": 0.85, "b_only": 15, "a_only": 13} and r15["hypotheses"]["H15b"]["status"] == "supported"
    assert r15["descriptive"]["H15e"]["points"] == 0.6 and data["e15"]["rft_yield"]["problems_with_accepted"] == 359
    assert data["e15"]["stage_meters"]["rft"] == {"wall_s": 1.0, "peak_reserved_gib": 4.4, "energy_wh": 2.0}
    assert "not computed here" in data["e15"]["verdicts_note"]  # the report itself still carries numbers only


def test_registered_hypotheses_absent_when_analysis_file_missing(tmp_path: Path) -> None:
    write(tmp_path, "E15_gwaya_rft_report.json", {"receipt_sha256": RECEIPT, "status": "complete"})
    assert bsd.build(tmp_path, UTC)["e15"]["registered_hypotheses"]["available"] is False
