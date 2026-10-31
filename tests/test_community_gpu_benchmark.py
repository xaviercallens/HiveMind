"""Tests for the opt-in Community GPU Benchmark & Non-Monetary Compute-Reciprocity Receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hivemind.cli.community_gpu_benchmark import (
    NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE,
    detect_community_accelerator,
    main,
    run_community_gpu_benchmark,
)


def test_detect_community_accelerator_auto_and_cpu() -> None:
    auto_info = detect_community_accelerator("auto")
    assert auto_info["backend"] in {"cuda", "rocm_hip", "apple_mps", "cpu_fallback"}
    assert auto_info["wddm_usable_budget_mib"] <= 6656.0

    cpu_info = detect_community_accelerator("cpu")
    assert cpu_info["device"] == "cpu"
    assert cpu_info["backend"] == "cpu_fallback"
    assert cpu_info["is_gpu"] is False
    assert "nvml_detected_gpu" not in cpu_info

    with pytest.raises(ValueError, match="Unsupported preferred_device"):
        detect_community_accelerator("tpu")


def test_run_community_gpu_benchmark_emits_verified_non_monetary_receipt(tmp_path: Path) -> None:
    out_file = tmp_path / "receipt.json"
    receipt = run_community_gpu_benchmark(
        contributor_handle="test-dev-gpu",
        preferred_device="auto",
        token_steps=6,
        router_tokens=32,
        hidden_size=32,
        seed=42,
        output_path=out_file,
    )
    assert out_file.exists()
    persisted = json.loads(out_file.read_text(encoding="utf-8"))
    assert persisted["receipt_sha256"] == receipt["receipt_sha256"]
    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["overall_passed"] is True

    # Verify non-monetary compute-reciprocity charter fields
    charter = receipt["non_monetary_charter"]
    assert charter["financial_payment_usd_eur"] == 0.0
    assert charter["no_money_involved"] is True
    assert charter["reward_nature"] == "non_monetary_compute_reciprocity_points_only"
    assert "live production HiveMind Swarm is deployed" in charter["redemption_condition"]
    assert charter["legal_notice"] == NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE

    # Verify all 4 technical stages passed
    assert receipt["stage1_accelerator"]["within_8gb_wddm_envelope"] is True
    assert (
        receipt["stage2_real_token_and_ef21"]["trained_holdout_nll"]
        < receipt["stage2_real_token_and_ef21"]["baseline_holdout_nll"]
    )
    assert receipt["stage2_real_token_and_ef21"]["ef21_conservation_error_l2"] < 1e-4
    assert receipt["stage3_three_tier_128_moe_and_prefetch"]["num_experts"] == 128
    assert receipt["stage3_three_tier_128_moe_and_prefetch"]["gating_reduction_factor"] == 8.0
    assert receipt["stage3_three_tier_128_moe_and_prefetch"]["prefetch_calls"] > 0
    assert receipt["stage4_zk_opoi_v4_and_insurance"]["batch_passed"] is True
    assert receipt["stage4_zk_opoi_v4_and_insurance"]["insurance_pool_reserve_hive"] == 200.0
    assert receipt["stage4_zk_opoi_v4_and_insurance"]["fault_drill"]["passed"] is True
    assert receipt["stage4_zk_opoi_v4_and_insurance"]["fault_drill"]["faults_found"] == [[2, 1]]


def test_community_gpu_benchmark_cli_entrypoint(tmp_path: Path) -> None:
    out_file = tmp_path / "cli_receipt.json"
    rc = main(
        [
            "--contributor-handle",
            "cli-tester",
            "--device",
            "cpu",
            "--token-steps",
            "4",
            "--output",
            str(out_file),
        ]
    )
    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["contributor_handle"] == "cli-tester"
    assert data["stage1_accelerator"]["device"] == "cpu"
    assert data["non_monetary_charter"]["symbolic_points_preview_pts"] == 250


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contributor_handle": "   "},
        {"token_steps": 0},
        {"router_tokens": 2},
        {"hidden_size": 12},
    ],
)
def test_community_gpu_benchmark_rejects_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        run_community_gpu_benchmark(output_path=None, **kwargs)
