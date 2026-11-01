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
