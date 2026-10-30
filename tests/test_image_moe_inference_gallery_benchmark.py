# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the Intensive Image-MoE Inference Benchmark & 12-Scenario Public Gallery."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.core.image_moe_inference_gallery_benchmark import (
    build_public_image_moe_gallery_scenarios,
    run_intensive_image_moe_inference_stress_campaign,
)
from hivemind.router.life_as_a_bee import (
    life_as_a_bee_payload,
    render_life_as_a_bee_section,
)
from hivemind.router.points_wording import currency_leftovers


def test_build_public_image_moe_gallery_scenarios(tmp_path: Path) -> None:
    gallery_dir = tmp_path / "gallery"
    scenarios = build_public_image_moe_gallery_scenarios(output_gallery_dir=gallery_dir)
    assert len(scenarios) == 12
    divisions = {s.division_id for s in scenarios}
    assert len(divisions) == 7

    for scen in scenarios:
        assert scen.http_status == 200
        assert scen.top1_routed_division == scen.division_id
        assert scen.active_vram_mib <= 1500.0
        assert scen.warm_latency_ms > 0
        assert scen.tokens_per_sec > 0
        assert scen.grounding_confidence >= 0.96
        assert scen.source_url.startswith("https://")
        assert scen.jpg_filename.endswith(".jpg")
        assert scen.svg_data_uri().startswith("data:image/svg+xml;base64,")
        d = scen.to_dict()
        assert d["svg_data_uri"].startswith("data:image/svg+xml;base64,")
        svg_file = gallery_dir / scen.svg_filename
        assert svg_file.exists()
        svg_text = svg_file.read_text(encoding="utf-8")
        assert "<svg" in svg_text
        assert "data:image/jpeg;base64," in svg_text


def test_run_intensive_image_moe_inference_stress_campaign_and_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrent_requests_per_node must be >= 1"):
        run_intensive_image_moe_inference_stress_campaign(concurrent_requests_per_node=0)

    report_path = tmp_path / "report.json"
    gallery_dir = tmp_path / "gallery"
    report = run_intensive_image_moe_inference_stress_campaign(
        concurrent_requests_per_node=20,
        output_gallery_dir=gallery_dir,
        save_report_path=report_path,
    )
    assert report["overall_passed"] is True
    assert report["total_gallery_scenarios"] == 12
    assert report["divisions_covered"] == 7
    assert report["top1_division_routing_accuracy"] == 1.0
    assert report["stress_burst_total_requests"] == 60
    assert report["stress_burst_zero_oom"] is True
    assert len(report["hybrid_node_stress_reports"]) == 3
    assert len(report["report_sha256"]) == 64
    assert report_path.exists()


def test_life_as_a_bee_gallery_integration_and_currency_cleanliness() -> None:
    payload = life_as_a_bee_payload()
    assert "image_moe_gallery_summary" in payload
    assert payload["image_moe_gallery_summary"]["total_image_scenarios"] == 12
    assert payload["image_moe_gallery_summary"]["divisions_covered"] == 7
    assert payload["image_moe_gallery_summary"]["top1_routing_accuracy"] == 1.0
    assert len(payload["image_moe_gallery_scenarios"]) == 12

    section_html = render_life_as_a_bee_section()
    assert 'id="image-moe-gallery"' in section_html
    assert 'data-bee-filter="image_moe_gallery"' in section_html
    assert "data:image/svg+xml;base64," in section_html
    assert currency_leftovers(section_html) == []
