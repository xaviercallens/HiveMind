# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control tests for auto_improvement, base_model_catalog, compression, and dev_image_video_moe_integration_benchmark (PR-IP102)."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.auto_improvement import NocturnalAutoImprovementScheduler
from hivemind.core.base_model_catalog import GpuTier, classify_tensor, summarize_header
from hivemind.core.compression import (
    compress_lora_delta,
    compress_lora_delta_dct_sparse_2bit,
    compute_pseudo_gradient_delta,
    decompress_lora_delta,
    decompress_lora_delta_dct_sparse_2bit,
)
from hivemind.core.dev_image_video_moe_integration_benchmark import (
    EndUserPostAttribution,
    ModalityGainBenchmarkScenario,
    RedditPostCohortFunnel,
)


def test_auto_improvement_scheduler_rejects_invalid_inputs() -> None:
    sched = NocturnalAutoImprovementScheduler()
    for bad_now in (True, False, -1.0, float("nan"), float("inf"), "now"):
        with pytest.raises(ValueError, match="now_utc must be a non-negative finite number"):
            sched.is_european_night(bad_now)  # type: ignore[arg-type]

    for bad_seq in ("not_a_list", b"bytes", 123, None):
        with pytest.raises(ValueError, match="connected_producers must be a sequence"):
            sched.filter_eligible_producers(bad_seq)  # type: ignore[arg-type]

    for bad_vram in (True, False, 0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_vram_gb must be a positive finite number"):
            sched.filter_eligible_producers([], min_vram_gb=bad_vram)  # type: ignore[arg-type]

    for bad_age in (True, False, 0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_heartbeat_age_s must be a positive finite number"):
            sched.filter_eligible_producers([], max_heartbeat_age_s=bad_age)  # type: ignore[arg-type]

    for bad_wu in (None, {}, "wu"):
        with pytest.raises(ValueError, match="workunit must be an AutoImprovementWorkunit"):
            sched.verify_workunit_execution(bad_wu, "worker-1", "acct-1")  # type: ignore[arg-type]

    wu = sched.create_domain_workunit("maths_formal", num_samples=4, inner_steps=2, seed=42)
    for bad_str in ("", "   ", "bad\nstr", None, 123):
        with pytest.raises(ValueError, match="worker_id must be a non-empty printable string"):
            sched.verify_workunit_execution(wu, bad_str, "acct-1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="account_id must be a non-empty printable string"):
            sched.verify_workunit_execution(wu, "worker-1", bad_str)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="execution_class must be a non-empty printable string"):
            sched.verify_workunit_execution(wu, "worker-1", "acct-1", execution_class=bad_str)  # type: ignore[arg-type]


def test_base_model_catalog_rejects_invalid_inputs() -> None:
    for bad_name in ("", "  ", None, 123):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            classify_tensor(bad_name, [64, 64])  # type: ignore[arg-type]

    for bad_shape in ("64x64", b"64", None, 123):
        with pytest.raises(ValueError, match="shape must be a sequence of integers"):
            classify_tensor("layer.weight", bad_shape)  # type: ignore[arg-type]

    for bad_tensors in (None, [], "invalid", 123):
        with pytest.raises(ValueError, match="tensors must be a Mapping"):
            summarize_header(bad_tensors)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tier_id must be a non-empty string"):
        GpuTier("", 4096, "src")
    for bad_budget in (True, False, 0, -100, 4.5, "4096"):
        with pytest.raises(ValueError, match="budget_mib must be a positive integer"):
            GpuTier("std_8gb", bad_budget, "src")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        GpuTier("std_8gb", 4096, "   ")


def test_compression_rejects_invalid_dicts_and_packages() -> None:
    valid_d = {"w": torch.ones((4, 4))}
    for bad in (None, {}, [], "invalid"):
        with pytest.raises(ValueError, match="theta_init must be a non-empty dict"):
            compute_pseudo_gradient_delta(bad, valid_d)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="delta_dict must be a non-empty dict"):
            compress_lora_delta(bad, "w1", "main", 1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="delta_dict must be a non-empty dict"):
            compress_lora_delta_dct_sparse_2bit(bad, "w1", "main", 1)  # type: ignore[arg-type]
    for bad_final in (None, [], "invalid", 123):
        with pytest.raises(ValueError, match="theta_final must be a dict"):
            compute_pseudo_gradient_delta(valid_d, bad_final)  # type: ignore[arg-type]

    for bad_pkg in (None, {}, "pkg", 123):
        with pytest.raises(ValueError, match="package must be a CompressedDeltaPackage"):
            decompress_lora_delta(bad_pkg)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="package must be a CompressedDeltaPackage"):
            decompress_lora_delta_dct_sparse_2bit(bad_pkg)  # type: ignore[arg-type]


def test_dev_image_video_moe_benchmark_dataclasses_reject_invalid_fields() -> None:
    with pytest.raises(ValueError, match="user_handle must be a non-empty string"):
        EndUserPostAttribution(
            user_handle="",
            role="prosumer",
            registered_utc="2026-10-30",
            hardware_profile="RTX 2070",
            declared_capacity_1k=80.0,
            hmt_balance=100.0,
            attributed_post_id="POST-1",
            subreddit="r/SelfHostedAI",
            post_archetype="Gamer",
            attribution_confidence=0.95,
            attribution_signals=("sig",),
            observed_behavior="beh",
            conversion_lesson="lesson",
        )

    for bad_conf in (True, False, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="attribution_confidence must be in"):
            EndUserPostAttribution(
                user_handle="u1",
                role="prosumer",
                registered_utc="2026-10-30",
                hardware_profile="RTX 2070",
                declared_capacity_1k=80.0,
                hmt_balance=100.0,
                attributed_post_id="POST-1",
                subreddit="r/SelfHostedAI",
                post_archetype="Gamer",
                attribution_confidence=bad_conf,  # type: ignore[arg-type]
                attribution_signals=("sig",),
                observed_behavior="beh",
                conversion_lesson="lesson",
            )

    for bad_visitors in (True, False, -1, 2.5):
        with pytest.raises(ValueError, match="unique_browser_visitors must be a non-negative integer"):
            RedditPostCohortFunnel(
                post_id="POST-1",
                subreddit="r/SelfHostedAI",
                language="en",
                title_summary="title",
                unique_browser_visitors=bad_visitors,  # type: ignore[arg-type]
                join_page_views=10,
                life_as_a_bee_views=5,
                registered_accounts=2,
                active_api_or_gpu_users=1,
                visitor_to_signup_rate_pct=20.0,
                signup_to_active_rate_pct=50.0,
                total_vram_contributed_gb=16.0,
                top_converting_hook="hook",
                primary_friction_point="friction",
                remediation_applied="fix",
            )

    for bad_lat in (True, False, 0.0, -10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="moe_latency_ms must be a positive finite number"):
            ModalityGainBenchmarkScenario(
                scenario_id="SC-1",
                modality="developer_coding_moe",
                model_id="hivemind/moe-coder-polyglot-3tome",
                title="Test",
                target_persona="Dev",
                active_experts=("e1",),
                baseline_model="base",
                baseline_quality_score=70.0,
                moe_quality_score=90.0,
                quality_gain_pct=28.5,
                baseline_latency_ms=1000.0,
                moe_latency_ms=bad_lat,  # type: ignore[arg-type]
                speedup_x=2.0,
                baseline_tokens_per_sec=20.0,
                moe_tokens_per_sec=40.0,
                active_vram_mib=4000.0,
                citizen_node_compatible=True,
                key_artifact_or_output="out",
            )
