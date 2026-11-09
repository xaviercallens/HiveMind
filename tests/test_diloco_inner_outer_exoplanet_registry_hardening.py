# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control tests for diloco_inner, diloco_outer, exoplanet_astro_vmoe_poc, and expert_registry hardening (PR-IP103)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hivemind.core.diloco_inner import (
    build_inner_optimizer,
    get_deterministic_batch,
    replay_inner_segment,
)
from hivemind.core.diloco_outer import (
    DiLoCoOuterOptimizer,
    filter_byzantine_deltas_multikrum,
    flatten_delta_dict,
)
from hivemind.core.exoplanet_astro_vmoe_poc import (
    asymmetric_focal_loss,
    compute_citizen_science_entropy_weights,
    synthesize_exoplanet_transit_and_router_response,
)
from hivemind.core.expert_registry import (
    ExpertRegistry,
    HierarchicalExpertRouter,
)


def test_diloco_inner_rejects_invalid_inputs() -> None:
    p = nn.Parameter(torch.ones((2, 2)))
    for bad_params in (None, [], "invalid", (p,)):
        with pytest.raises(ValueError, match="trainable_params must be a non-empty list"):
            build_inner_optimizer(bad_params)  # type: ignore[arg-type]

    for bad_lr in (True, False, 0.0, -1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lr must be a positive finite float"):
            build_inner_optimizer([p], lr=bad_lr)

    for bad_wd in (True, False, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight_decay must be a non-negative finite float"):
            build_inner_optimizer([p], weight_decay=bad_wd)

    for bad_flag in (None, 1, "true"):
        with pytest.raises(ValueError, match="use_paged_8bit_adamw must be a bool"):
            build_inner_optimizer([p], use_paged_8bit_adamw=bad_flag)  # type: ignore[arg-type]

    tokens = torch.ones((4, 8), dtype=torch.long)
    dev = torch.device("cpu")
    for bad_seq in (None, [[1, 2]], torch.ones((4,), dtype=torch.long), torch.empty((0, 8), dtype=torch.long)):
        with pytest.raises(ValueError, match="token_sequences must be a non-empty tensor with ndim >= 2"):
            get_deterministic_batch(bad_seq, 1, 42, 1, dev)  # type: ignore[arg-type]

    for bad_step in (True, False, -1, 1.5, "1"):
        with pytest.raises(ValueError, match="step must be a non-negative integer"):
            get_deterministic_batch(tokens, bad_step, 42, 1, dev)  # type: ignore[arg-type]

    for bad_seed in (True, False, -1, 1.5, "42"):
        with pytest.raises(ValueError, match="rng_seed must be a non-negative integer"):
            get_deterministic_batch(tokens, 1, bad_seed, 1, dev)  # type: ignore[arg-type]

    for bad_state in (None, [], "invalid"):
        with pytest.raises(ValueError, match="start_lora_state must be a dict"):
            replay_inner_segment(nn.Linear(2, 2), bad_state, tokens, 0, 1)  # type: ignore[arg-type]


def test_diloco_outer_rejects_invalid_inputs() -> None:
    for bad_dict in (None, {}, [], "invalid"):
        with pytest.raises(ValueError, match="delta_dict must not be empty"):
            flatten_delta_dict(bad_dict)  # type: ignore[arg-type]

    for bad_wd in (None, [], "invalid", 123):
        with pytest.raises(ValueError, match="worker_deltas must be a dict"):
            filter_byzantine_deltas_multikrum(bad_wd)  # type: ignore[arg-type]

    opt = DiLoCoOuterOptimizer()
    valid_theta = {"w": torch.ones((2, 2))}
    valid_deltas = {"w1": {"w": torch.ones((2, 2))}}
    for bad_theta in (None, {}, [], "invalid"):
        with pytest.raises(ValueError, match="theta_prev must be a non-empty dict"):
            opt.step("main", 1, bad_theta, valid_deltas)  # type: ignore[arg-type]

    for bad_wdeltas in (None, {}, [], "invalid"):
        with pytest.raises(ValueError, match="worker_deltas must be a non-empty dict"):
            opt.step("main", 1, valid_theta, bad_wdeltas)  # type: ignore[arg-type]

    for bad_grace in (True, False, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="grace_window_s must be a non-negative finite number"):
            opt.step("main", 1, valid_theta, valid_deltas, grace_window_s=bad_grace)


def test_exoplanet_astro_vmoe_poc_rejects_invalid_inputs() -> None:
    valid_logits = torch.randn((4, 3))
    valid_targets = torch.zeros((4,), dtype=torch.long)
    for bad_t in (None, [1.0, 2.0], torch.empty((0, 3))):
        with pytest.raises(ValueError, match="logits must be a non-empty torch.Tensor"):
            asymmetric_focal_loss(bad_t, valid_targets)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="targets must be a non-empty torch.Tensor"):
            asymmetric_focal_loss(valid_logits, bad_t)  # type: ignore[arg-type]

    for bad_l in (None, [1.0, 2.0], torch.ones((4,)), torch.empty((0, 3))):
        with pytest.raises(ValueError, match="logits must be a non-empty 2D\\+ torch.Tensor"):
            compute_citizen_science_entropy_weights(bad_l)  # type: ignore[arg-type]

    for bad_shift in (True, False, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="centroid_shift_arcsec must be a finite non-negative number"):
            synthesize_exoplanet_transit_and_router_response(1.0, 1.0, centroid_shift_arcsec=bad_shift)


def test_expert_registry_rejects_invalid_inputs() -> None:
    for bad_strict in (None, 1, "true"):
        with pytest.raises(ValueError, match="strict must be a bool"):
            ExpertRegistry(strict=bad_strict)  # type: ignore[arg-type]

    reg = ExpertRegistry()
    for bad_rec in (None, {}, "rec", 123):
        with pytest.raises(ValueError, match="record must be an ExpertRecord"):
            reg.register(bad_rec)  # type: ignore[arg-type]

    for bad_bytes in (None, "not_bytes", [1, 2], 123):
        with pytest.raises(ValueError, match="data must be bytes or bytearray"):
            reg.verify_adapter_bytes("any_id", bad_bytes)  # type: ignore[arg-type]

    router = HierarchicalExpertRouter(reg, base_model_id="Qwen/Qwen3-4B-Instruct-2507")
    for bad_prompt in (None, 123, ["prompt"], b"bytes"):
        with pytest.raises(ValueError, match="prompt must be a str"):
            router.prompt_hits(bad_prompt)  # type: ignore[arg-type]
