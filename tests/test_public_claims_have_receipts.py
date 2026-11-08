# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-03-D3 (Public Claims & Prompt-Visibility Notice Backed by Receipts),
PS-01-D2 (E24 Chain Regression Report), PS-06-D2 (E26 Residential Compression Report),
and PS-09-D2 (E27 1,000-Node Heterogeneous Testbed Report)."""

from __future__ import annotations

import json
from pathlib import Path

from hivemind.router.benchmarks_page import (
    SUPPORTED_BENCHMARK_LANGS,
    benchmarks_payload,
    load_measured_results,
    render_benchmarks_page,
)
from hivemind.router.life_as_a_bee import LIFE_AS_A_BEE_SCENARIOS, render_life_as_a_bee_section
from hivemind.router.portal_page import render_portal_page

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ps03_d3_every_chat_and_portal_page_contains_prompt_visibility_notice() -> None:
    """PS-03-D3: Every page offering or documenting the chat API states the PoC / prompt-visibility notice."""
    portal_html = render_portal_page("", "test-nonce")
    assert 'id="poc-notice"' in portal_html
    assert "Prompts are processed by volunteer-run nodes" in portal_html
    assert "no secrets, passwords, API keys, personal information" in portal_html

    bee_html = render_life_as_a_bee_section()
    assert "artifacts/all_bees_e2e_and_llm_comparison_report.json" in bee_html

    for lang in SUPPORTED_BENCHMARK_LANGS:
        bench_html = render_benchmarks_page(lang=lang)
        assert "id='poc-notice'" in bench_html


def test_ps03_d3_every_benchmarks_and_scenario_number_cites_existing_receipt_path(tmp_path: Path) -> None:
    """PS-03-D3: Every measured number on the benchmarks page and in Life-as-a-Bee cites an existing receipt file."""
    measured = load_measured_results()
    assert len(measured) >= 3
    bench_html = render_benchmarks_page(lang="en")
    payload = benchmarks_payload()
    assert payload["comparison_with_other_vendors"].startswith("none:")

    for entry in measured:
        src_rel = str(entry.get("source") or "").strip()
        assert src_rel, f"Entry {entry.get('id')} missing source receipt path"
        src_path = REPO_ROOT / src_rel
        assert src_path.is_file(), f"Cited receipt {src_rel!r} does not exist on disk"
        assert f"<code>{src_rel}</code>" in bench_html

    # Every Life-as-a-Bee scenario either cites an existing artifacts/... receipt or is explicitly labelled NOT_SOURCED
    for scenario in LIFE_AS_A_BEE_SCENARIOS:
        if "artifacts/" in scenario.source:
            artifact_rel = scenario.source.split("artifacts/", 1)[1].split()[0].rstrip(",)")
            artifact_path = REPO_ROOT / "artifacts" / artifact_rel
            assert artifact_path.is_file(), f"Scenario {scenario.scenario_id} cites missing {artifact_path}"
        else:
            assert scenario.source.startswith("NOT_SOURCED:")

    # Negative control: load_measured_results drops any entry without a source receipt path
    bad_json = tmp_path / "bad_measured.json"
    bad_json.write_text(
        json.dumps(
            {
                "results": [
                    {"id": "unbacked-claim", "model_id": "hivemind/coder", "accuracy": 0.99},
                    {
                        "id": "backed-claim",
                        "model_id": "hivemind/coder-lite",
                        "source": "docs/MVP_E2E_SMOKE.md",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    filtered = load_measured_results(bad_json)
    assert [r["id"] for r in filtered] == ["backed-claim"]


def test_ps01_d2_e24_chain_regression_report() -> None:
    """PS-01-D2: E24 report has decision in ('CAUSE_ATTRIBUTED', 'ALL_REFUTED') and passes eval sanity gate."""
    path = REPO_ROOT / "artifacts" / "experiments" / "E24_chain_regression_report.json"
    assert path.is_file()
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-01-D2"
    assert r["decision"] in ("CAUSE_ATTRIBUTED", "ALL_REFUTED")
    assert r["eval_sanity_gate_passed"] is True
    assert r["max_abs_noop_delta_nats"] <= 1e-6
    assert r["arms"]["E24_A1_ratchet_acceptance_rule"]["passes_20_round_window_gate"] is True
    assert r["arms"]["E24_A3_stratified_mixing"]["passes_20_round_window_gate"] is True
    assert r["arms"]["BASELINE_post_hmq_00171_unstratified_80steps"]["passes_20_round_window_gate"] is False


def test_ps06_d2_e26_residential_compression_report() -> None:
    """PS-06-D2: E26 report has >= 5 residential links, >= 50x compression, and <= 0.010 bpb delta."""
    path = REPO_ROOT / "artifacts" / "experiments" / "E26_residential_compression_report.json"
    assert path.is_file()
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-06-D2"
    assert len(r["links"]) >= 5
    assert r["summary"]["median_compression_ratio"] >= 50.0
    assert r["summary"]["max_bpb_delta_vs_control"] <= 0.010
    assert r["verdict"] == "PASS"
    for link in r["links"]:
        assert link["within_registered_bpb_margin"] is True
        assert link["negative_control_no_ef21_bpb_delta"] > 0.010


def test_ps09_d2_e27_testbed_1000_report() -> None:
    """PS-09-D2: E27 report has >= 3 runs for each node count in (100, 300, 1000) and passes all gates."""
    path = REPO_ROOT / "artifacts" / "experiments" / "E27_testbed_1000_report.json"
    assert path.is_file()
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-09-D2"
    assert all(len(r["runs"][str(n)]) >= 3 for n in (100, 300, 1000))
    assert r["verdict"] == "PASS"
    assert r["cloud_spend_usd"] <= r["monthly_budget_ceiling_usd"] <= 150.0
