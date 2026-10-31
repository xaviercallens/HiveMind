# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Multilingual End-to-End Bee Benchmark, Honest LLM Comparison & Infographic Page."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.benchmarks_page import (
    ALL_BEES_E2E_BENCHMARKS,
    SUPPORTED_BENCHMARK_LANGS,
    TASK_DOMAIN_COMPARISON_MATRIX,
    benchmarks_payload,
    render_benchmarks_page,
    render_hivemind_capacity_infographic_svg,
)
from hivemind.router.gateway import create_router_app


def test_all_18_bees_e2e_benchmarks_and_honest_comparison_payload() -> None:
    assert len(ALL_BEES_E2E_BENCHMARKS) == 18
    assert len(TASK_DOMAIN_COMPARISON_MATRIX) == 7
    assert SUPPORTED_BENCHMARK_LANGS == ("fr", "en", "es", "de", "zh")

    payload = benchmarks_payload()
    assert payload["object"] == "hivemind.benchmarks"
    assert payload["swarm_capacity_summary"]["total_bees_tested_e2e"] == 18
    assert payload["swarm_capacity_summary"]["all_passed_http_200"] is True

    comp = payload["honest_llm_comparison_summary"]
    assert comp["hivemind_cash_cost_usd_per_1m"] == 0.0
    assert comp["mean_daily_usage_sufficiency_pct"] >= 92.0
    assert len(comp["domains_where_hivemind_beats_frontier_outright"]) == 3

    for bee in ALL_BEES_E2E_BENCHMARKS:
        assert bee.http_status == 200
        assert bee.tokens_per_sec > 0
        assert bee.warm_latency_ms > 0
        assert 85.0 <= bee.daily_usage_sufficiency_pct <= 100.0
        href = bee.playground_href()
        assert href.startswith("/?model=") and href.endswith("#playground")


def test_multilingual_benchmarks_page_and_svg_infographic_routes(tmp_path: Path) -> None:
    svg = render_hivemind_capacity_infographic_svg()
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "PROJECT HIVEMIND" in svg
    assert "224.0 GB Aggregate VRAM" in svg

    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    # Test all 5 languages on /benchmarks
    for lang in SUPPORTED_BENCHMARK_LANGS:
        res = client.get(f"/benchmarks?lang={lang}")
        assert res.status_code == 200
        assert res.headers["content-language"] == lang
        assert "hivemind/moe-coder-polyglot-3tome" in res.text
        assert "hivemind/moe-image-gen" in res.text
        assert "hivemind/video-moe" in res.text
        assert "<svg" in res.text

    # Test French alias /comparatif
    res_fr = client.get("/comparatif")
    assert res_fr.status_code == 200
    assert res_fr.headers["content-language"] == "fr"

    # Test JSON API and SVG Infographic endpoint
    res_json = client.get("/v1/hivemind/benchmarks")
    assert res_json.status_code == 200
    assert res_json.json()["swarm_capacity_summary"]["total_bees_tested_e2e"] == 18

    res_svg = client.get("/v1/hivemind/benchmarks/infographic.svg")
    assert res_svg.status_code == 200
    assert "image/svg+xml" in res_svg.headers["content-type"]

    # Test slim-container fixed routes
    assert client.get("/v1/hivemind/swarm-plan").status_code == 200
    assert client.post("/v1/hivemind/bees/rusty-sundials/sweep", json={"family": "van_der_pol"}).status_code == 200
