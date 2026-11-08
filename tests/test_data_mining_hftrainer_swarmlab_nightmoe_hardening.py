# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening suite for dataset_mining_campaign, hf_real_trainer, swarm_lab, and night_moe_page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from hivemind.boinc.night_moe_page import load_night_moe, render_night_moe_html
from hivemind.data.dataset_mining_campaign import (
    MinedDatasetShardReport,
    _compute_max_cross_split_4gram_jaccard,
    _compute_shannon_entropy_bits,
    run_dataset_mining_and_ruche_campaign,
)
from hivemind.data.hf_real_trainer import (
    _encode_synthetic_regression_batch,
    train_and_publish_season3_domain_experts,
    train_single_domain_lora_adapter,
)
from hivemind.lab.swarm_lab import Rng, Scenario, sparkline


def _valid_shard_report(**overrides: Any) -> MinedDatasetShardReport:
    base: dict[str, Any] = {
        "domain_name": "code_python",
        "shard_id": "SHARD-S10-CODE_PYTHON",
        "shard_cas_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        "shard_sha256": "a" * 64,
        "assigned_contributor": "@xaviercallens",
        "lease_worker_id": "gpu-miner-xaviercallens-01",
        "lease_version_tag": 1,
        "lease_ttl_s": 4.25,
        "train_documents": 3,
        "holdout_documents": 2,
        "total_utf8_bytes": 512,
        "shannon_entropy_bits_per_byte": 4.5,
        "cross_split_jaccard_max": 0.1,
        "dedup_clean": True,
    }
    base.update(overrides)
    return MinedDatasetShardReport(**base)


def test_mined_dataset_shard_report_and_campaign_validation() -> None:
    report = _valid_shard_report()
    assert report.domain_name == "code_python"

    for str_field in (
        "domain_name",
        "shard_id",
        "shard_cas_cid",
        "shard_sha256",
        "assigned_contributor",
        "lease_worker_id",
    ):
        with pytest.raises(ValueError, match=str_field):
            _valid_shard_report(**{str_field: ""})

    for int_field in ("lease_version_tag", "train_documents", "holdout_documents", "total_utf8_bytes"):
        with pytest.raises(ValueError, match=int_field):
            _valid_shard_report(**{int_field: 0})
        with pytest.raises(ValueError, match=int_field):
            _valid_shard_report(**{int_field: True})

    with pytest.raises(ValueError, match="lease_ttl_s"):
        _valid_shard_report(lease_ttl_s=0.0)
    with pytest.raises(ValueError, match="shannon_entropy_bits_per_byte"):
        _valid_shard_report(shannon_entropy_bits_per_byte=-0.1)
    with pytest.raises(ValueError, match="cross_split_jaccard_max"):
        _valid_shard_report(cross_split_jaccard_max=1.5)
    with pytest.raises(ValueError, match="dedup_clean"):
        _valid_shard_report(dedup_clean="yes")

    with pytest.raises(ValueError, match="texts"):
        _compute_shannon_entropy_bits("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train_texts"):
        _compute_max_cross_split_4gram_jaccard("not-a-list", ["a b c d"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="holdout_texts"):
        _compute_max_cross_split_4gram_jaccard(["a b c d"], [123])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="vault_dir"):
        run_dataset_mining_and_ruche_campaign(vault_dir="")
    with pytest.raises(ValueError, match="report_json_path"):
        run_dataset_mining_and_ruche_campaign(report_json_path="")
    with pytest.raises(ValueError, match="training_steps"):
        run_dataset_mining_and_ruche_campaign(training_steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        run_dataset_mining_and_ruche_campaign(learning_rate=0.0)
    with pytest.raises(ValueError, match="base_seed"):
        run_dataset_mining_and_ruche_campaign(base_seed=-1)


def test_hf_real_trainer_validation() -> None:
    with pytest.raises(ValueError, match="samples"):
        _encode_synthetic_regression_batch("not-a-sequence", 8, 8, 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="in_features"):
        _encode_synthetic_regression_batch(["s"], 0, 8, 42)
    with pytest.raises(ValueError, match="out_features"):
        _encode_synthetic_regression_batch(["s"], 8, False, 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        _encode_synthetic_regression_batch(["s"], 8, 8, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_holdout"):
        _encode_synthetic_regression_batch(["s"], 8, 8, 42, is_holdout="no")  # type: ignore[arg-type]

    x, y = _encode_synthetic_regression_batch(["sample one", "sample two"], 8, 4, 42)
    assert x.shape == (12, 8) and y.shape == (12, 4)

    ref_state = {
        "layer.0.lora_A": torch.randn(4, 8),
        "layer.0.lora_B": torch.zeros(8, 4),
    }
    with pytest.raises(ValueError, match="domain_name"):
        train_single_domain_lora_adapter("", ref_state)
    with pytest.raises(ValueError, match="reference_lora_state"):
        train_single_domain_lora_adapter("Rust_Systems_Security", {})
    with pytest.raises(ValueError, match="lora_rank_r"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, lora_rank_r=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lora_alpha"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, lora_alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, inner_steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lr"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, lr=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="topk_ratio"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, topk_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="custom_corpus"):
        train_single_domain_lora_adapter("Rust_Systems_Security", ref_state, custom_corpus=[""])

    with pytest.raises(ValueError, match="store"):
        train_and_publish_season3_domain_experts("not-a-store", ref_state)  # type: ignore[arg-type]


def test_swarm_lab_rng_scenario_and_sparkline_validation() -> None:
    for bad_seed in (True, False, 1.5, "1"):
        with pytest.raises(ValueError, match="seed"):
            Rng(bad_seed)  # type: ignore[arg-type]

    for int_field in ("M", "K", "rounds", "H", "dim", "seed", "straggle_every", "churn_start", "churn_end"):
        with pytest.raises(ValueError, match=int_field):
            Scenario(**{int_field: True}).validate()  # type: ignore[arg-type]

    for float_field in ("eta", "sigma", "spread", "offset", "penalty", "grace", "outer_lr", "outer_momentum"):
        with pytest.raises(ValueError, match=float_field):
            Scenario(**{float_field: True}).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="values"):
        sparkline([])
    with pytest.raises(ValueError, match="values"):
        sparkline([1.0, float("nan")])
    with pytest.raises(ValueError, match="width"):
        sparkline([1.0, 2.0], width=0)
    with pytest.raises(ValueError, match="width"):
        sparkline([1.0, 2.0], width=True)  # type: ignore[arg-type]

    assert len(sparkline([1.0, 2.0, 3.0], width=8)) == 3


def test_night_moe_page_validation(tmp_path: Path) -> None:
    for bad_path in ("", "   ", 123):
        with pytest.raises(ValueError, match="path"):
            load_night_moe(bad_path)  # type: ignore[arg-type]

    assert load_night_moe(tmp_path / "missing.json") is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"subjects": "not-a-dict"}), encoding="utf-8")
    assert load_night_moe(bad_json) is None

    assert "No results file is deployed" in render_night_moe_html(None)

    with pytest.raises(ValueError, match="report"):
        render_night_moe_html({"subjects": "not-a-dict"})
    with pytest.raises(ValueError, match="missing required key"):
        render_night_moe_html({"subjects": {}})

    valid_report = {
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "alpha": 0.05,
        "bonferroni_alpha": 0.005,
        "n_other_cells": 10,
        "subjects": {},
    }
    html = render_night_moe_html(valid_report)
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in html
