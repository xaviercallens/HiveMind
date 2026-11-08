# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for Switch Transformers MoE load-balancing gradient (#394) and byte-literal secret hygiene (#392)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.boinc.workunit import sign_shard_bytes
from hivemind.core.exoplanet_astro_vmoe_poc import (
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    FrozenAstroVisionBackbone,
    UpcycledAstroSparseVisionMoE,
)
from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MASSIVE_IMAGE_MOE_DIVISIONS,
    MassiveSwarmSparseImageMoE,
)
from hivemind.swarm.abc_moe_agent_mesh import WaggleDanceMessage


def _build_astro_vmoe(alpha: float) -> UpcycledAstroSparseVisionMoE:
    torch.manual_seed(42)
    backbone = FrozenAstroVisionBackbone(
        patch_dim=16,
        hidden_dim=32,
        intermediate_dim=64,
        num_classes=12,
        seed=42,
    )
    deltas = {
        b_id: np.zeros((32, 32), dtype=np.float32)
        for b_id in EXOPLANET_ASTRO_EXPERT_BRANCHES
    }
    return UpcycledAstroSparseVisionMoE(
        backbone=backbone,
        branch_deltas=deltas,
        top_k=2,
        load_balance_alpha=alpha,
    )


def _build_massive_vmoe(alpha: float) -> MassiveSwarmSparseImageMoE:
    torch.manual_seed(42)
    backbone = FrozenAstroVisionBackbone(
        patch_dim=16,
        hidden_dim=32,
        intermediate_dim=64,
        num_classes=21,
        seed=42,
    )
    deltas = {
        d_id: np.zeros((32, 32), dtype=np.float32)
        for d_id in MASSIVE_IMAGE_MOE_DIVISIONS
    }
    return MassiveSwarmSparseImageMoE(
        backbone=backbone,
        division_deltas=deltas,
        top_k=2,
        load_balance_alpha=alpha,
    )


def test_upcycled_astro_vmoe_load_balance_loss_retains_grad_and_shifts_router_grad() -> None:
    """Issue #394: UpcycledAstroSparseVisionMoE load_balance_loss_weighted retains grad and modifies router gradient."""
    torch.manual_seed(7)
    patches = torch.randn(4, 6, 16)
    labels = torch.tensor([0, 3, 6, 9], dtype=torch.long)

    model_with_aux = _build_astro_vmoe(alpha=0.01)
    out_with = model_with_aux(patches)
    assert isinstance(out_with["load_balance_loss_weighted"], torch.Tensor)
    assert out_with["load_balance_loss_weighted"].requires_grad is True
    loss_with = torch.nn.functional.cross_entropy(out_with["logits"], labels) + out_with["load_balance_loss_weighted"]
    loss_with.backward()
    grad_with = model_with_aux.router.weight.grad.detach().clone()

    model_no_aux = _build_astro_vmoe(alpha=0.0)
    out_no = model_no_aux(patches)
    loss_no = torch.nn.functional.cross_entropy(out_no["logits"], labels) + out_no["load_balance_loss_weighted"]
    loss_no.backward()
    grad_no = model_no_aux.router.weight.grad.detach().clone()

    assert not torch.allclose(grad_with, grad_no)


def test_massive_swarm_image_moe_load_balance_loss_retains_grad_and_shifts_router_grad() -> None:
    """Issue #394: MassiveSwarmSparseImageMoE load_balance_loss_weighted retains grad and modifies router gradient."""
    torch.manual_seed(11)
    patches = torch.randn(4, 6, 16)
    labels = torch.tensor([0, 4, 8, 12], dtype=torch.long)

    model_with_aux = _build_massive_vmoe(alpha=0.01)
    out_with = model_with_aux(patches)
    assert isinstance(out_with["load_balance_loss_weighted"], torch.Tensor)
    assert out_with["load_balance_loss_weighted"].requires_grad is True
    loss_with = torch.nn.functional.cross_entropy(out_with["logits"], labels) + out_with["load_balance_loss_weighted"]
    loss_with.backward()
    grad_with = model_with_aux.router.weight.grad.detach().clone()

    model_no_aux = _build_massive_vmoe(alpha=0.0)
    out_no = model_no_aux(patches)
    loss_no = torch.nn.functional.cross_entropy(out_no["logits"], labels) + out_no["load_balance_loss_weighted"]
    loss_no.backward()
    grad_no = model_no_aux.router.weight.grad.detach().clone()

    assert not torch.allclose(grad_with, grad_no)


def test_no_hardcoded_secret_or_key_byte_literals_in_src_and_empty_secrets_rejected() -> None:
    """Issue #392: No b'...key...' / b'...secret...' byte literals remain in src/, and signing helpers reject empty secrets."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    pattern = re.compile(r'b"[^"]*(?:Key|key|SECRET|secret)[^"]*"')
    matches: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{py_file.relative_to(src_root)}:{line_no}:{line.strip()}")
    assert matches == []

    with pytest.raises(ValueError, match="non-empty"):
        sign_shard_bytes(b"parquet-payload", rng_seed=42, secret_key=b"")

    msg = WaggleDanceMessage(
        task_id="t-1",
        scout_bee_id="bee-1",
        task_domain="code_generation",
        confidence_score=0.9,
        dance_duration_ms=900,
        estimated_tokens_cost=128,
    )
    with pytest.raises(ValueError, match="non-empty"):
        msg.sign(secret_key="")
    msg.sign(secret_key="key-alpha")
    assert msg.verify_signature(secret_key="key-alpha") is True
    assert msg.verify_signature(secret_key="key-beta") is False
    assert msg.verify_signature(secret_key="") is False
