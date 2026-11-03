# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The benchmarks page shows measured results only: every figure traces to measured_results.json, unmeasured tiers say so,
and no comparison with another vendor is shown (none was ever run)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router import benchmarks_page as bp
from hivemind.router.accounts import RouterStore
from hivemind.router.benchmarks_page import (
    SUPPORTED_BENCHMARK_LANGS,
    benchmarks_payload,
    load_measured_results,
    render_benchmarks_page,
    render_hivemind_capacity_infographic_svg,
)
from hivemind.router.catalog import TIERS
from hivemind.router.gateway import create_router_app

ROOT = Path(__file__).resolve().parents[1]


def test_measured_results_are_complete_and_match_their_source_artifacts() -> None:
    results = load_measured_results()
    assert {r["model_id"] for r in results} >= {"hivemind/bee-stars", "hivemind/coder-lite"}
    for r in results:
        assert r["date"] and r["method"] and r["source"] and r["hardware"]
    stars = next(r for r in results if r["model_id"] == "hivemind/bee-stars")
    artifact = ROOT / stars["source"]
    if artifact.exists():  # the artifacts folder is not part of the gateway image; when it is present the copy must agree exactly
        raw = json.loads(artifact.read_text(encoding="utf-8"))
        assert (stars["n"], stars["correct"], stars["accuracy"]) == (raw["night"]["n"], raw["night"]["correct"], raw["night"]["accuracy"])
        assert stars["wilson95"] == raw["night"]["wilson95"]
        assert stars["served_through_router"]["correct"] == raw["served_through_router"]["correct"]
    assert 0 < stars["wilson95"][0] < stars["accuracy"] < stars["wilson95"][1] < 1
    assert abs(stars["accuracy"] - stars["correct"] / stars["n"]) < 5e-4


def test_payload_has_no_vendor_comparison_and_lists_every_unmeasured_tier() -> None:
    payload = benchmarks_payload()
    assert payload["status"] == "measured_only" and "none" in payload["comparison_with_other_vendors"]
    text = json.dumps(payload).lower()
    for banned in ("gpt-4o", "claude", "gemini", "frontier", "daily_usage_sufficiency", "beats"):
        assert banned not in text.replace("gpt-4o, claude or gemini", ""), banned
    measured_ids = {r["model_id"] for r in payload["measured_results"]}
    unmeasured_ids = {t["model_id"] for t in payload["tiers_without_measured_result"]}
    assert unmeasured_ids == {f"hivemind/{t.tier_id}" for t in TIERS} - measured_ids
    assert all(t["status"] == "not measured" for t in payload["tiers_without_measured_result"])


def test_a_missing_or_malformed_results_file_shows_nothing_rather_than_inventing(tmp_path: Path) -> None:
    assert load_measured_results(tmp_path / "absent.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_measured_results(bad) == []
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"results": [{"id": "x", "model_id": "hivemind/x"}]}), encoding="utf-8")  # no source: not accepted
    assert load_measured_results(partial) == []


def test_every_number_on_the_page_comes_from_the_results_file_or_live_telemetry() -> None:
    page = render_benchmarks_page(lang="en", nonce="n")
    plain = re.sub(r"<[^>]+>", " ", page)
    for fabricated in ("85.0", "1015.8", "71.87", "76.5", "89%", "1.4x", "224", "GPT-4o:", "100% vs 64%"):
        assert fabricated not in plain, fabricated
    assert "283/317" in plain and "0.893" in plain and "id='poc-notice'" in page


def test_all_five_languages_render_with_the_notice_and_the_unmeasured_list(tmp_path: Path) -> None:
    client = TestClient(create_router_app(RouterStore(tmp_path / "router.db", b"k" * 32), allow_private_producers=True))
    for lang in SUPPORTED_BENCHMARK_LANGS:
        res = client.get(f"/benchmarks?lang={lang}")
        assert res.status_code == 200 and res.headers["content-language"] == lang
        assert "poc-notice" in res.text and "hivemind/bee-stars" in res.text and "hivemind/coder-lite" in res.text
        assert "hivemind/moe-image-gen" not in res.text and "hivemind/video-moe" not in res.text  # nothing serves these tiers
    assert client.get("/comparatif").headers["content-language"] == "fr"
    body = client.get("/v1/hivemind/benchmarks").json()
    assert body["status"] == "measured_only" and len(body["measured_results"]) >= 2
    svg = client.get("/v1/hivemind/benchmarks/infographic.svg")
    assert svg.status_code == 200 and "image/svg+xml" in svg.headers["content-type"]
    assert render_hivemind_capacity_infographic_svg().startswith("<svg") and "224" not in svg.text


def test_the_old_hard_coded_tables_are_gone() -> None:
    assert not hasattr(bp, "ALL_BEES_E2E_BENCHMARKS") and not hasattr(bp, "TASK_DOMAIN_COMPARISON_MATRIX")
    assert len((ROOT / "src/hivemind/router/benchmarks_page.py").read_text(encoding="utf-8").splitlines()) < 400
