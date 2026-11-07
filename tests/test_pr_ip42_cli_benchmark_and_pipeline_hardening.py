# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for PR-IP-42: CLI benchmark receipt & pipeline hardening."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hivemind.cli.community_gpu_benchmark import (
    run_community_gpu_benchmark,
    verify_community_benchmark_receipt,
)
from hivemind.cli.poc_rtx2070 import (
    _create_tokenized_parquet_shard,
    fetch_hf_domain_subset,
    run_huggingface_rtx2070_poc,
)
from hivemind.cli.simulate_e2e import (
    estimate_workunit_tflops,
    run_end_to_end_simulation,
)


def _rehash_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    canonical = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        **body,
        "receipt_sha256": hashlib.sha256(canonical).hexdigest(),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contributor_handle": ""},
        {"contributor_handle": "   "},
        {"token_steps": True},
        {"token_steps": 0},
        {"router_tokens": True},
        {"router_tokens": 2},
        {"hidden_size": True},
        {"hidden_size": 16, "seed": True},
        {"pool_holdings_override": float("nan")},
        {"pool_holdings_override": float("inf")},
        {"pool_holdings_override": -1.0},
        {"pool_holdings_override": True},
    ],
)
def test_community_gpu_benchmark_rejects_non_finite_and_boolean_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        run_community_gpu_benchmark(
            preferred_device="cpu",
            output_path=None,
            **kwargs,
        )


def test_verify_community_benchmark_receipt_blocks_rehashed_forged_receipts() -> None:
    genuine = run_community_gpu_benchmark(
        contributor_handle="ip42-honest-cpu",
        preferred_device="cpu",
        token_steps=2,
        router_tokens=16,
        hidden_size=16,
        output_path=None,
    )
    assert verify_community_benchmark_receipt(genuine) is True

    # 1. Forging gpu_points_awarded=500 on a CPU receipt and recomputing SHA-256 must fail closed
    forged_gpu_pts = copy.deepcopy(genuine)
    forged_gpu_pts["gpu_points_awarded"] = 500
    forged_gpu_pts["non_monetary_charter"]["gpu_points_awarded"] = 500
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_gpu_pts)) is False

    # 2. Flipping overall_passed=True on a failed pool reserve receipt and recomputing SHA-256 must fail closed
    failed_pool = run_community_gpu_benchmark(
        contributor_handle="ip42-failed-pool",
        preferred_device="cpu",
        token_steps=2,
        router_tokens=16,
        hidden_size=16,
        output_path=None,
        pool_holdings_override=150.0,
    )
    assert failed_pool["overall_passed"] is False
    forged_pool = copy.deepcopy(failed_pool)
    forged_pool["overall_passed"] = True
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_pool)) is False

    # 3. Altering non-monetary charter fields and recomputing SHA-256 must fail closed
    forged_charter = copy.deepcopy(genuine)
    forged_charter["non_monetary_charter"]["financial_payment_usd_eur"] = 10.0
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_charter)) is False

    # 4. Stage 2 non-improving holdout NLL or invalid EF21 contraction must fail closed even if rehashed
    forged_stage2 = copy.deepcopy(genuine)
    forged_stage2["stage2_real_token_and_ef21"]["trained_holdout_nll"] = (
        forged_stage2["stage2_real_token_and_ef21"]["baseline_holdout_nll"] + 0.25
    )
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_stage2)) is False

    # 5. Stage 3 invalid gating reduction factor or zero prefetch calls must fail closed even if rehashed
    forged_stage3 = copy.deepcopy(genuine)
    forged_stage3["stage3_three_tier_128_moe_and_prefetch"]["prefetch_calls"] = 0
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_stage3)) is False

    # 6. Stage 4 failed fault drill must fail closed even if rehashed
    forged_stage4 = copy.deepcopy(genuine)
    forged_stage4["stage4_zk_opoi_v4_and_insurance"]["fault_drill"]["passed"] = False
    assert verify_community_benchmark_receipt(_rehash_receipt(forged_stage4)) is False

    # 7. NaN injected into receipt fails closed without raising uncaught ValueError
    nan_receipt = copy.deepcopy(genuine)
    nan_receipt["elapsed_total_ms"] = float("nan")
    assert verify_community_benchmark_receipt(nan_receipt) is False


def test_simulate_e2e_and_estimate_workunit_tflops_guardrails() -> None:
    valid_wu = SimpleNamespace(inner_steps_H=10, seq_len=32)
    tflops = estimate_workunit_tflops(valid_wu, total_params=100_000_000, micro_batch_size=2)
    assert tflops > 0.0

    with pytest.raises(ValueError, match="total_params"):
        estimate_workunit_tflops(valid_wu, total_params=-1, micro_batch_size=2)
    with pytest.raises(ValueError, match="total_params"):
        estimate_workunit_tflops(valid_wu, total_params=True, micro_batch_size=2)
    with pytest.raises(ValueError, match="micro_batch_size"):
        estimate_workunit_tflops(valid_wu, total_params=0, micro_batch_size=0)
    with pytest.raises(ValueError, match="micro_batch_size"):
        estimate_workunit_tflops(valid_wu, total_params=0, micro_batch_size=True)
    with pytest.raises(ValueError, match="inner_steps_H"):
        estimate_workunit_tflops(SimpleNamespace(inner_steps_H=0, seq_len=32))
    with pytest.raises(ValueError, match="seq_len"):
        estimate_workunit_tflops(SimpleNamespace(inner_steps_H=10, seq_len=0))

    with pytest.raises(ValueError, match="profile_name"):
        run_end_to_end_simulation(profile_name="   ")
    with pytest.raises(ValueError, match="inner_steps_H"):
        run_end_to_end_simulation(inner_steps_H=0)
    with pytest.raises(ValueError, match="inner_steps_H"):
        run_end_to_end_simulation(inner_steps_H=True)
    with pytest.raises(ValueError, match="spot_check_interval_s"):
        run_end_to_end_simulation(spot_check_interval_s=0)
    with pytest.raises(ValueError, match="spot_check_interval_s"):
        run_end_to_end_simulation(spot_check_interval_s=True)


def test_poc_rtx2070_guardrails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        fetch_hf_domain_subset(dataset_id="   ", num_samples=4)
    with pytest.raises(ValueError, match="num_samples"):
        fetch_hf_domain_subset(dataset_id="sahil2801/CodeAlpaca-20k", num_samples=0)
    with pytest.raises(ValueError, match="num_samples"):
        fetch_hf_domain_subset(dataset_id="sahil2801/CodeAlpaca-20k", num_samples=True)

    dummy_tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3, 4])
    with pytest.raises(ValueError, match="domain"):
        _create_tokenized_parquet_shard(dummy_tok, "  ", ["hello"], tmp_path / "s.parquet")
    with pytest.raises(ValueError, match="texts"):
        _create_tokenized_parquet_shard(dummy_tok, "Code", [], tmp_path / "s.parquet")
    with pytest.raises(ValueError, match="seq_len"):
        _create_tokenized_parquet_shard(dummy_tok, "Code", ["hello"], tmp_path / "s.parquet", seq_len=0)
    with pytest.raises(ValueError, match="num_sequences"):
        _create_tokenized_parquet_shard(dummy_tok, "Code", ["hello"], tmp_path / "s.parquet", num_sequences=0)

    for bad_kwargs in (
        {"model_id": "  "},
        {"hf_dataset_id": "  "},
        {"hf_subset_size": 0},
        {"hf_subset_size": True},
        {"lora_rank_r": 0},
        {"lora_rank_r": True},
        {"lora_alpha": 0},
        {"inner_steps_H": 0},
        {"spot_check_interval_s": 0},
    ):
        with pytest.raises(ValueError):
            run_huggingface_rtx2070_poc(**bad_kwargs)
