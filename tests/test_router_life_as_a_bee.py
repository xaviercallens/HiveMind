# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Life as a Bee dashboard, 15-scenario end-user benchmark payload, and gateway endpoints."""

from __future__ import annotations

import html
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.life_as_a_bee import (
    BEE_DECISION_MATRIX,
    BEE_LIFECYCLE_PHASES,
    LIFE_AS_A_BEE_SCENARIOS,
    life_as_a_bee_payload,
    render_life_as_a_bee_section,
)


def test_life_as_a_bee_scenarios_cover_all_15_models() -> None:
    assert len(LIFE_AS_A_BEE_SCENARIOS) == 15
    model_ids = {s.model_id for s in LIFE_AS_A_BEE_SCENARIOS}
    expected = {
        "hivemind/auto",
        "hivemind/coder",
        "hivemind/coder-lite",
        "hivemind/crowd-v1",
        "hivemind/gemma-core",
        "hivemind/mistral-core",
        "hivemind/qwen35-core",
        "hivemind/bee-stars",
        "hivemind/bee-rusty-sundials",
        "hivemind/bee-lean-prover",
        "hivemind/coder-plus",
        "hivemind/bee-ruche-moe",
        "hivemind/moe-qwen35-bee",
        "hivemind/moe-general",
        "hivemind/moe-coder",
    }
    assert model_ids == expected
    for scen in LIFE_AS_A_BEE_SCENARIOS:
        assert scen.http_status == 200
        assert scen.tokens_per_sec > 0
        assert scen.warm_latency_ms > 0
        assert scen.total_tokens == scen.prompt_tokens + scen.completion_tokens
        assert scen.when_to_use_fr
        assert scen.when_to_use_en
        assert scen.when_not_to_use
        assert scen.how_to_invoke


def test_life_as_a_bee_payload_and_html_section() -> None:
    payload = life_as_a_bee_payload()
    assert payload["object"] == "hivemind.life_as_a_bee"
    assert payload["summary"]["total_scenarios_executed"] == 15
    assert payload["summary"]["all_passed_200_ok"] is True
    assert payload["summary"]["total_tokens_processed"] == 2291
    assert len(payload["lifecycle_phases"]) == len(BEE_LIFECYCLE_PHASES) == 4
    assert len(payload["decision_matrix"]) == len(BEE_DECISION_MATRIX) == 8
    assert len(payload["scenarios"]) == 15

    section_html = render_life_as_a_bee_section()
    assert 'id="life-as-a-bee"' in section_html
    for scen in LIFE_AS_A_BEE_SCENARIOS:
        assert scen.model_id in section_html
        assert html.escape(scen.scenario_title) in section_html


def test_gateway_life_as_a_bee_endpoints(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.sqlite3", b"s" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    r_json = client.get("/v1/hivemind/life-as-a-bee")
    assert r_json.status_code == 200
    body = r_json.json()
    assert body["object"] == "hivemind.life_as_a_bee"
    assert len(body["scenarios"]) == 15

    r_html = client.get("/life-as-a-bee")
    assert r_html.status_code == 200
    assert "Life as a Bee Dashboard" in r_html.text
    assert "hivemind/bee-ruche-moe" in r_html.text
    assert "hivemind/gemma-core" in r_html.text
    assert "hivemind/qwen35-core" in r_html.text


def test_phase_2_does_not_claim_a_measured_yield_time() -> None:
    """No artifact holds a yield measured with a BOINC task running, so the page must not state one."""
    from hivemind.router.life_as_a_bee import BEE_LIFECYCLE_PHASES

    phase = next(p for p in BEE_LIFECYCLE_PHASES if p.phase_id == "PHASE-2-DAYTIME-NECTAR")
    text = " ".join([phase.key_metric, phase.title_fr, phase.title_en, phase.summary_fr, phase.summary_en])
    assert "1,13" not in text and "1.13" not in text
    assert "non mesuré" in phase.key_metric
    # The preemption figure may appear only as a target.
    assert "objectif" in phase.title_fr and "target" in phase.title_en


def test_every_phase_names_the_source_of_its_figures() -> None:
    """Each lifecycle phase carries a source; any repo path it names exists; constants are not dressed as live readings."""
    import re
    from pathlib import Path

    from hivemind.router.life_as_a_bee import BEE_LIFECYCLE_PHASES, life_as_a_bee_payload

    root = Path(__file__).resolve().parents[1]
    for phase in BEE_LIFECYCLE_PHASES:
        assert phase.source.strip(), phase.phase_id
        for rel in re.findall(r"(?:docs|artifacts)/[\w./-]+", phase.source):
            assert (root / rel).exists(), (phase.phase_id, rel)
    p1 = next(p for p in BEE_LIFECYCLE_PHASES if p.phase_id == "PHASE-1-POLLEN-INTENT")
    assert "Temps réel" not in p1.time_window and "ms" not in p1.time_window
    assert "0 HTTP 402/5xx" not in p1.key_metric
    p4 = next(p for p in BEE_LIFECYCLE_PHASES if p.phase_id == "PHASE-4-NOCTURNAL-ROYAL-JELLY")
    assert "%" not in p4.key_metric
    payload = life_as_a_bee_payload()
    assert all("source" in ph for ph in payload["lifecycle_phases"])


def test_every_scenario_speed_is_sourced_or_marked_not_sourced() -> None:
    """A scenario's tokens_per_sec either appears in the artifact its source names, or the source says NOT_SOURCED."""
    from pathlib import Path

    from hivemind.router.life_as_a_bee import LIFE_AS_A_BEE_SCENARIOS, life_as_a_bee_payload

    root = Path(__file__).resolve().parents[1]
    artifact_text = (root / "artifacts" / "all_bees_e2e_and_llm_comparison_report.json").read_text(encoding="utf-8")
    sourced = 0
    for scen in LIFE_AS_A_BEE_SCENARIOS:
        assert scen.source.strip(), scen.scenario_id
        if scen.source.startswith("artifacts/"):
            assert str(scen.tokens_per_sec) in artifact_text, (scen.scenario_id, scen.tokens_per_sec)
            sourced += 1
        else:
            assert scen.source.startswith("NOT_SOURCED"), scen.scenario_id
    assert sourced >= 1
    payload = life_as_a_bee_payload()
    assert payload["summary"]["fastest_bee_tokens_per_sec"] == max(s.tokens_per_sec for s in LIFE_AS_A_BEE_SCENARIOS)
    assert payload["summary"]["scenarios_with_artifact_source"] == sourced
    assert all("source" in sc for sc in payload["scenarios"])


def test_summary_figures_are_derived_from_the_scenario_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing one scenario's speed must change the rendered mean; a hand-typed constant would not move."""
    import dataclasses

    from hivemind.router import life_as_a_bee as lab

    before = lab.render_life_as_a_bee_section()
    assert "calculée sur 15 scénarios" in before
    first = lab.LIFE_AS_A_BEE_SCENARIOS[0]
    bumped = (dataclasses.replace(first, tokens_per_sec=first.tokens_per_sec + 150.0),) + tuple(lab.LIFE_AS_A_BEE_SCENARIOS[1:])
    monkeypatch.setattr(lab, "LIFE_AS_A_BEE_SCENARIOS", bumped)
    after = lab.render_life_as_a_bee_section()
    expected = f"{sum(sc.tokens_per_sec for sc in bumped) / len(bumped):.2f}".replace(".", ",")
    assert f"Moyenne : {expected} tok/s" in after
    assert "Moyenne : 25,64 tok/s" not in after


def test_moe_ratio_and_tier_list_are_derived_from_the_scenario_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MoE-versus-dense ratio uses the mean of all MoE bees, and moving one MoE latency moves it."""
    import dataclasses

    from hivemind.router import life_as_a_bee as lab

    html_before = lab.render_life_as_a_bee_section()
    moe = [sc for sc in lab.LIFE_AS_A_BEE_SCENARIOS if sc.category == "moe_bee"]
    dense = next(sc for sc in lab.LIFE_AS_A_BEE_SCENARIOS if "14B" in sc.architecture)
    ratio = dense.warm_latency_ms / (sum(sc.warm_latency_ms for sc in moe) / len(moe))
    assert f"{ratio:.2f}".replace(".", ",") + "x plus rapide" in html_before
    assert "1,83x Plus Rapide" not in html_before and "~15,3s" not in html_before
    bumped = tuple(dataclasses.replace(sc, warm_latency_ms=sc.warm_latency_ms * 2) if sc.category == "moe_bee" else sc for sc in lab.LIFE_AS_A_BEE_SCENARIOS)
    monkeypatch.setattr(lab, "LIFE_AS_A_BEE_SCENARIOS", bumped)
    html_after = lab.render_life_as_a_bee_section()
    assert f"{ratio / 2:.2f}".replace(".", ",") + "x plus rapide" in html_after
    fastest = max((sc for sc in lab.LIFE_AS_A_BEE_SCENARIOS if sc.gpu_residency.startswith("100% GPU")), key=lambda sc: sc.tokens_per_sec)
    assert f"<code>{fastest.model_id.split('/')[-1]}</code>" in html_before
