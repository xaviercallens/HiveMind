# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control boundary validation tests for QueenBee strategies, CrowdGPT loss check, AMCP collab, and SolverBee."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from hivemind.queenbee.strategies import (
    BanditStrategy,
    BeeMoESystemRouter,
    FederatedShareStrategy,
    TaskDecompositionRouter,
    VerifiedIDECorrectionFlow,
    VerifiedSubtask,
    clip_and_noise,
)
from hivemind.solverbee.families import FamilyError, _slot_time_grid
from hivemind.swarm.amcp_collab import dev_keyring
from hivemind.swarm.crowdgpt_loss_check import (
    CrowdV1,
    LossCheckReceipt,
    window_losses,
)


def test_queenbee_strategies_boundary_validation() -> None:
    bandit = BanditStrategy()
    with pytest.raises(TypeError, match="rng"):
        bandit.choose("c1", ["e1", "e2"], "not-an-rng")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clip_norm"):
        clip_and_noise({"w": torch.ones(2)}, True, 0.1, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sigma"):
        clip_and_noise({"w": torch.ones(2)}, 1.0, True, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delta"):
        clip_and_noise([], 1.0, 0.1, None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="router"):
        FederatedShareStrategy(1.0, 0.1, router="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="accountant"):
        FederatedShareStrategy(1.0, 0.1, accountant="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="bandit"):
        TaskDecompositionRouter(bandit="bad")  # type: ignore[arg-type]
    td_router = TaskDecompositionRouter()
    rng = random.Random(42)
    with pytest.raises(ValueError, match="subtasks"):
        td_router.route_subtasks(["not-a-subtask"], ["e1"], rng)  # type: ignore[list-item]
    st = VerifiedSubtask("s1", "c1", "unit_tests")
    with pytest.raises(TypeError, match="assignments and outcomes"):
        td_router.observe_subtasks([st], [], {})  # type: ignore[arg-type]

    flow = VerifiedIDECorrectionFlow(opt_in=True)
    with pytest.raises(TypeError, match="baseline_outcomes"):
        flow.train_and_gate_local_adapter([], [], {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_size"):
        flow.train_and_gate_local_adapter([], {"t1": "PASS"}, {"t1": "PASS"}, step_size=0.0)

    with pytest.raises(ValueError, match="admitted_adapters"):
        BeeMoESystemRouter(admitted_adapters="not-a-seq")
    with pytest.raises(TypeError, match="pack_to_adapter"):
        BeeMoESystemRouter(pack_to_adapter=["bad"])  # type: ignore[arg-type]
    moe_router = BeeMoESystemRouter()
    with pytest.raises(TypeError, match="messages"):
        moe_router.classify_prompt_domain("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pack"):
        moe_router.route_pack("")
    with pytest.raises(TypeError, match="enforce_gate"):
        moe_router.route_pack("exoplanet", enforce_gate="yes")  # type: ignore[arg-type]


def test_crowdgpt_loss_check_and_amcp_solverbee_boundary_validation() -> None:
    with pytest.raises(TypeError, match="config"):
        CrowdV1("not-a-mapping")  # type: ignore[arg-type]
    tiny_cfg = {
        "vocabSize": 32,
        "dim": 16,
        "nLayers": 1,
        "nHeads": 2,
        "nKvHeads": 1,
        "headDim": 8,
        "maxSeqLen": 16,
        "mlpHidden": 32,
        "weightTying": True,
    }
    model = CrowdV1(tiny_cfg)
    with pytest.raises(TypeError, match="flat"):
        model.load_flat([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="idx"):
        model.forward(torch.ones(4, dtype=torch.long))

    tokens = np.zeros(32, dtype=np.uint32)
    with pytest.raises(TypeError, match="model"):
        window_losses("not-a-model", tokens, [0], 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="starts"):
        window_losses(model, tokens, [-1], 4)
    with pytest.raises(ValueError, match="window"):
        window_losses(model, tokens, [0], 1)
    with pytest.raises(ValueError, match="batch"):
        window_losses(model, tokens, [0], 4, batch=0)
    with pytest.raises(TypeError, match="stop"):
        window_losses(model, tokens, [0], 4, stop="not-callable")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="text"):
        LossCheckReceipt.from_json("")
    with pytest.raises(ValueError, match="object"):
        LossCheckReceipt.from_json("[1, 2, 3]")

    with pytest.raises(ValueError, match="path"):
        dev_keyring("   ")

    with pytest.raises(FamilyError, match="raw and slots"):
        _slot_time_grid([], {}, [])  # type: ignore[arg-type]
    with pytest.raises(FamilyError, match="conversions"):
        _slot_time_grid({}, {}, "not-a-list")  # type: ignore[arg-type]
