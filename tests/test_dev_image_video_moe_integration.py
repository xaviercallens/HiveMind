# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for End-User Reddit Post Attribution, /webgpu 200 OK fix, and Developer/Image/Video MoE Integration."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.core.dev_image_video_moe_integration_benchmark import (
    DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS,
    END_USER_POST_ATTRIBUTIONS,
    REDDIT_POST_COHORT_FUNNELS,
    attribute_end_users_to_posts_and_learn_conversion,
    run_intense_dev_image_video_moe_benchmark,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import (
    catalog_payload,
    normalize_model_alias,
    resolve_producer_quorum_moe_state,
    tier_by_id,
)
from hivemind.router.gateway import create_router_app
from hivemind.router.life_as_a_bee import (
    SECTORAL_MOE_SCENARIOS,
    life_as_a_bee_payload,
    render_life_as_a_bee_section,
)


def test_end_user_post_attribution_and_conversion_learning() -> None:
    report = attribute_end_users_to_posts_and_learn_conversion()
    assert report["campaign_id"] == "HM-S30-END-USER-ATTRIBUTION-AND-CONVERSION-LEARNING"
    assert report["global_community_totals"]["registered_accounts_total"] == 55
    assert report["global_community_totals"]["unique_browser_visitor_ips_48h"] == 132
    assert report["global_community_totals"]["aggregate_vram_gb"] == 224.0
    assert len(END_USER_POST_ATTRIBUTIONS) == 7
    assert len(REDDIT_POST_COHORT_FUNNELS) == 3

    handles = {u.user_handle for u in END_USER_POST_ATTRIBUTIONS}
    assert "cmdr_rtx2070_steam" in handles
    assert "tibrib + tibrib-chat" in handles
    assert "quebec-tester-2147" in handles
    assert "t4-big-gpu-moe-20261028" in handles

    assert report["funnel_leak_remediation"]["verified_http_status"] == 200
    assert len(report["next_post_conversion_playbook"]) == 4


def test_new_dev_image_video_moe_models_are_integrated_in_catalog() -> None:
    # 1. Polyglot 3-Tome Coding MoE aliases -> moe-coder
    for alias in (
        "hivemind/moe-coder-polyglot-3tome",
        "moe-coder-polyglot-3tome",
        "hivemind/polyglot-coder-moe",
        "hivemind/dev-moe",
        "dev-moe",
    ):
        assert normalize_model_alias(alias) == "moe-coder"
        tier = tier_by_id(alias)
        assert tier is not None and tier.tier_id == "moe-coder"

    # 2. Image Generation & Vision-MoE aliases -> bee-ruche-moe
    for alias in (
        "hivemind/moe-image-gen",
        "moe-image-gen",
        "hivemind/vision-moe",
        "vision-moe",
        "hivemind/bee-reddit-vision",
        "hivemind/bee-exoplanet-vmoe",
        "hivemind/image-moe",
    ):
        assert normalize_model_alias(alias) == "bee-ruche-moe"
        tier = tier_by_id(alias)
        assert tier is not None and tier.tier_id == "bee-ruche-moe"

    # 3. Spatio-Temporal Video Generation MoE aliases -> moe-general
    for alias in (
        "hivemind/video-moe",
        "video-moe",
        "hivemind/moe-video-gen",
        "hivemind/wan21-video-moe",
    ):
        assert normalize_model_alias(alias) == "moe-general"
        tier = tier_by_id(alias)
        assert tier is not None and tier.tier_id == "moe-general"

    cat = catalog_payload()
    partner_public_ids = {m["public_id"] for m in cat["prewired_partner_models"]}
    assert "hivemind/moe-coder-polyglot-3tome" in partner_public_ids
    assert "hivemind/moe-image-gen" in partner_public_ids
    assert "hivemind/video-moe" in partner_public_ids

    quorum_state = resolve_producer_quorum_moe_state(8, total_vram_gb=64.0)
    assert "hivemind/moe-coder-polyglot-3tome" in quorum_state["unlocked_moe_models"]
    assert "hivemind/moe-image-gen" in quorum_state["unlocked_moe_models"]
    assert "hivemind/video-moe" in quorum_state["unlocked_moe_models"]


def test_life_as_a_bee_includes_dev_image_and_video_moe_scenarios() -> None:
    sectoral_ids = {s.scenario_id for s in SECTORAL_MOE_SCENARIOS}
    assert "SCEN-MOE-POLYGLOT-DEV-3TOME" in sectoral_ids
    assert "SCEN-MOE-IMAGE-GEN-STUDIO" in sectoral_ids
    assert "SCEN-MOE-VIDEO-SPATIOTEMPORAL" in sectoral_ids

    payload = life_as_a_bee_payload()
    assert "dev_image_video_moe_summary" in payload
    summary = payload["dev_image_video_moe_summary"]
    assert summary["polyglot_3tome_coding_moe"]["tome1_to_tome3_hard_loss_reduction_pct"] == 99.07
    assert summary["image_generation_vision_moe"]["citizen_micro_vram_mib"] <= 1500.0
    assert summary["video_spatiotemporal_moe"]["dct_3d_compression_ratio_x"] == 225.4

    html_out = render_life_as_a_bee_section()
    assert "hivemind/moe-coder-polyglot-3tome" in html_out
    assert "hivemind/moe-image-gen" in html_out
    assert "hivemind/video-moe" in html_out


def test_intense_dev_image_video_moe_benchmark_and_webgpu_endpoint(tmp_path: Path) -> None:
    report = run_intense_dev_image_video_moe_benchmark()
    assert report["status"] == "COMPLETED_ALL_GATES_PASSED"
    assert len(DEV_IMAGE_VIDEO_BENCHMARK_SCENARIOS) == 18

    summary = report["overall_summary"]
    assert summary["total_scenarios"] == 18
    assert summary["total_stress_requests"] == 360
    assert summary["stress_http_200_rate_pct"] == 100.0

    # Verify gains across all 3 tracks
    dev_gain = summary["developer_coding_3tome_gain"]
    assert dev_gain["mean_quality_gain_pct"] > 25.0
    assert dev_gain["mean_speedup_x"] > 3.5
    assert dev_gain["max_active_vram_mib"] <= 6656.0  # Invariant I1 (8 GB GPU ceiling)

    img_gain = summary["image_generation_vision_moe_gain"]
    assert img_gain["mean_quality_gain_pct"] > 20.0
    assert img_gain["mean_speedup_x"] > 3.3
    assert img_gain["max_active_vram_mib"] <= 1500.0  # <1.5 GB citizen micro-VRAM cap

    vid_gain = summary["video_generation_spatiotemporal_moe_gain"]
    assert vid_gain["mean_quality_gain_pct"] > 25.0
    assert vid_gain["mean_speedup_x"] > 2.8
    assert vid_gain["fvd_reduction_pct"] > 35.0
    assert vid_gain["max_active_vram_mib"] <= 1500.0  # Citizen keyframe verifier <1.5 GB

    # Verify gateway routes /webgpu, /life-as-a-bee, and /v1/hivemind/life-as-a-bee with HTTP 200
    store = RouterStore(tmp_path / "router.db", b"s" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    r_webgpu = client.get("/webgpu")
    assert r_webgpu.status_code == 200
    assert "WebGPU" in r_webgpu.text

    r_laab = client.get("/life-as-a-bee")
    assert r_laab.status_code == 200
    assert "hivemind/moe-coder-polyglot-3tome" in r_laab.text
    assert "hivemind/moe-image-gen" in r_laab.text
    assert "hivemind/video-moe" in r_laab.text
