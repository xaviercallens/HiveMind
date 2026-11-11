# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for Iteration 189:
- `hivemind.data.hf_real_trainer`
- `hivemind.data.token_training_eval`
- `hivemind.economy.compute_coin`
- `hivemind.economy.international_launch_model`
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hivemind.data.hf_real_trainer import (
    _encode_synthetic_regression_batch,
    train_and_publish_astrophysics_domain_experts,
    train_and_publish_season3_domain_experts,
    train_single_domain_lora_adapter,
)
from hivemind.data.token_training_eval import (
    evaluate_tokens,
    tensor_bytes,
    token_nll,
    validate_splits,
)
from hivemind.economy import compute_coin as cc
from hivemind.economy.international_launch_model import (
    build_foundation_paper_v42_portal_payload,
    forecast_200plus_node_international_launch,
    project_international_launch_wave,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore


def test_hf_real_trainer_null_bytes_unmatched_lora_and_duplicate_domains(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="samples must be a sequence of strings"):
        _encode_synthetic_regression_batch(["ok", "bad\x00sample"], 8, 4, 42)

    # No lora_A / lora_B keys
    with pytest.raises(ValueError, match="must contain at least one \\(lora_A, lora_B\\) adapter pair"):
        train_single_domain_lora_adapter(
            "Rust_Systems_Security",
            {"layer.0.weight": torch.ones((4, 8), dtype=torch.float32) * 0.02},
        )

    # Unmatched lora_A without lora_B
    with pytest.raises(ValueError, match="missing a matched lora_A/lora_B pair"):
        train_single_domain_lora_adapter(
            "Rust_Systems_Security",
            {"layer.0.lora_A": torch.ones((4, 8), dtype=torch.float32) * 0.02},
        )

    # Mismatched rank between lora_A (4, 8) and lora_B (8, 6)
    with pytest.raises(ValueError, match="matching rank"):
        train_single_domain_lora_adapter(
            "Rust_Systems_Security",
            {
                "layer.0.lora_A": torch.ones((4, 8), dtype=torch.float32) * 0.02,
                "layer.0.lora_B": torch.ones((8, 6), dtype=torch.float32) * 0.02,
            },
        )

    store = ContentAddressedSwarmStore(tmp_path / "cas")
    ref_state = {
        "layer.0.lora_A": torch.ones((4, 8), dtype=torch.float32) * 0.02,
        "layer.0.lora_B": torch.zeros((8, 4), dtype=torch.float32),
    }
    for bad_domains in (
        ["Rust_Systems_Security", "Rust_Systems_Security"],
        ["Rust_Systems_Security\nInjected"],
        ["Rust_Systems_Security\x00"],
    ):
        with pytest.raises(ValueError, match="domains must be None or a non-empty sequence of unique non-empty strings"):
            train_and_publish_season3_domain_experts(store, ref_state, domains=bad_domains)
        with pytest.raises(ValueError, match="domains must be None or a non-empty sequence of unique non-empty strings"):
            train_and_publish_astrophysics_domain_experts(store, ref_state, domains=bad_domains)


def test_token_training_eval_tensor_bytes_splits_and_batch_shape_hardening() -> None:
    for bad_tensor in (
        torch.empty((0, 2)),
        torch.ones((2, 2), dtype=torch.bool),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, float("inf")]]),
    ):
        with pytest.raises(
            ValueError,
            match="state must map non-empty string keys to non-empty finite numeric torch.Tensor values",
        ):
            tensor_bytes({"w": bad_tensor})

    for bad_fixture in (
        {"train": "not-seq", "holdout": [{"id": "h1", "text": "holdout text"}]},
        {"train": ["not-a-dict"], "holdout": [{"id": "h1", "text": "holdout text"}]},
        {"train": [{"id": "t1"}], "holdout": [{"id": "h1", "text": "holdout text"}]},
        {"train": [{"id": "   ", "text": "train text"}], "holdout": [{"id": "h1", "text": "holdout text"}]},
    ):
        with pytest.raises(
            ValueError,
            match="must not be empty|must be a mapping containing 'id' and 'text'|empty or duplicate document/ID",
        ):
            validate_splits(bad_fixture)

    with pytest.raises(ValueError, match="logits must be a 3D numeric torch.Tensor"):
        token_nll(torch.ones((2, 4, 8), dtype=torch.bool), torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="labels must be a 2D integer torch.Tensor"):
        token_nll(torch.ones((2, 4, 8)), torch.ones((2, 4), dtype=torch.bool))
    with pytest.raises(ValueError, match="logits batch/sequence dimensions must match labels shape"):
        token_nll(torch.ones((2, 5, 8)), torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="logits batch/sequence dimensions must match labels shape"):
        token_nll(torch.ones((2, 1, 8)), torch.ones((2, 1), dtype=torch.long))

    dummy_model = torch.nn.Linear(2, 2)
    for bad_batch in (
        {
            "input_ids": torch.ones((4,), dtype=torch.long),
            "attention_mask": torch.ones((4,), dtype=torch.long),
            "labels": torch.ones((4,), dtype=torch.long),
        },
        {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.ones((2, 4), dtype=torch.long),
        },
        {
            "input_ids": torch.empty((0, 4), dtype=torch.long),
            "attention_mask": torch.empty((0, 4), dtype=torch.long),
            "labels": torch.empty((0, 4), dtype=torch.long),
        },
    ):
        with pytest.raises(
            ValueError,
            match="batch tensors 'input_ids', 'attention_mask', and 'labels' must be non-empty 2D tensors of matching shape",
        ):
            evaluate_tokens(dummy_model, bad_batch)


def test_compute_coin_sequence_bool_and_settlement_mapping_hardening() -> None:
    for bad_seq in ("12", b"12", 123, {1.0, 2.0}):
        with pytest.raises(ValueError, match="valuations must be a sequence"):
            cc.kelly_equilibrium(bad_seq)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="costs must be a sequence"):
            cc.tullock_equilibrium(100.0, bad_seq)  # type: ignore[arg-type]

    eq = cc.tullock_equilibrium(100.0, [1.0, 2.0])
    zero_supply_eq = cc.TullockEquilibrium(
        pool_hive=100.0,
        costs=(1.0, 2.0),
        supplies=(0.0, 0.0),
        payoffs=(0.0, 0.0),
        total_supply=0.0,
        active=(0, 1),
        price_per_cobblestone=math.inf,
    )
    with pytest.raises(ValueError, match="eq.supplies must have a positive sum"):
        cc.is_epsilon_nash(zero_supply_eq)

    bool_pool_eq = cc.TullockEquilibrium(
        pool_hive=True,  # type: ignore[arg-type]
        costs=(1.0, 2.0),
        supplies=eq.supplies,
        payoffs=eq.payoffs,
        total_supply=eq.total_supply,
        active=eq.active,
        price_per_cobblestone=eq.price_per_cobblestone,
    )
    with pytest.raises(ValueError, match="eq.pool_hive"):
        cc.is_epsilon_nash(bool_pool_eq)

    bad_mint_map = cc.EpochSettlement(0, 1.0, ["not-map"], {}, 0.0, 0.0, False, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settlement.minted_by_node must be a mapping"):
        cc.apply_settlement(bad_mint_map, lambda n, a: a)

    bad_fee_map = cc.EpochSettlement(0, 1.0, {}, ["not-map"], 0.0, 0.0, False, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settlement.fees_to_workers must be a mapping"):
        cc.apply_settlement(bad_fee_map, lambda n, a: a)

    bool_mint = cc.EpochSettlement(0, 1.0, {"n1": True}, {}, 0.0, 0.0, False, 1.0)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="minted_by_node"):
        cc.apply_settlement(bool_mint, lambda n, a: a)

    bool_fee = cc.EpochSettlement(0, 1.0, {}, {"n1": True}, 0.0, 0.0, False, 1.0)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="fees_to_workers"):
        cc.apply_settlement(bool_fee, lambda n, a: a)


def test_international_launch_model_forecast_and_portal_url_hardening() -> None:
    with pytest.raises(TypeError, match="baseline must be a SwarmLiveBaseline"):
        project_international_launch_wave("Wave 1", 1000, baseline="not-baseline")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="baseline must be a SwarmLiveBaseline"):
        forecast_200plus_node_international_launch(baseline="not-baseline")  # type: ignore[arg-type]

    for kw in (
        {"target_total_nodes": True},
        {"visitor_count": False},
        {"utc_hour": True},
        {"dedicated_producers": True},
        {"prosumer_nodes": False},
        {"consumer_nodes": True},
    ):
        with pytest.raises(ValueError, match="must be an integer"):
            forecast_200plus_node_international_launch(**kw)  # type: ignore[arg-type]

    for bad_cv in (-0.1, True, False, math.nan, math.inf):
        with pytest.raises(ValueError, match="token_length_cv and arrival_cv must be finite and >= 0"):
            forecast_200plus_node_international_launch(token_length_cv=bad_cv)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="token_length_cv and arrival_cv must be finite and >= 0"):
            forecast_200plus_node_international_launch(arrival_cv=bad_cv)  # type: ignore[arg-type]

    for bad_coords in ("not-seq", ["not-a-dict"], 123):
        with pytest.raises(ValueError, match="pioneer_coordinates must be None or a sequence of mappings"):
            forecast_200plus_node_international_launch(pioneer_coordinates=bad_coords)  # type: ignore[arg-type]

    for bad_url in ("ftp://example.org", "javascript:alert(1)", "https://example.org\nInjected: 1"):
        with pytest.raises(ValueError, match="base_gcp_url must be a non-empty http\\(s\\) URL string"):
            build_foundation_paper_v42_portal_payload(base_gcp_url=bad_url)
