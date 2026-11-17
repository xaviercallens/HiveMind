# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and regression tests for #394 (IP-35), IP-81, and IP-84."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hivemind.core.exoplanet_astro_vmoe_poc import (
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    FrozenAstroVisionBackbone,
    UpcycledAstroSparseVisionMoE,
)
from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MASSIVE_IMAGE_MOE_DIVISIONS,
    MassiveSwarmSparseImageMoE,
)
from hivemind.solverbee.families import named_spec
from hivemind.solverbee.wave3_verifiers import (
    verify_numeric_solver_trajectory,
    verify_quantum_fluid_relaxation,
)
from hivemind.verification.bee_energy import (
    MAX_PAIN_ENERGY,
    calibrate_noise_margin,
    evaluate_bee_candidate_energy,
    evaluate_child_bee_promotion,
)


def test_astro_vmoe_load_balance_loss_retains_router_gradient() -> None:
    """#394 / IP-35: load_balance_loss_weighted must be a differentiable Tensor affecting router gradients."""
    backbone = FrozenAstroVisionBackbone(
        patch_dim=48, hidden_dim=64, intermediate_dim=128, num_classes=12, seed=101
    )
    deltas = {
        b_id: np.zeros((64, 64), dtype=np.float32)
        for b_id in EXOPLANET_ASTRO_EXPERT_BRANCHES
    }
    x = torch.randn((8, 4, 48), dtype=torch.float32)

    vmoe = UpcycledAstroSparseVisionMoE(
        backbone=backbone, branch_deltas=deltas, top_k=2, load_balance_alpha=0.5
    )
    with torch.no_grad():
        vmoe.router.bias.copy_(torch.tensor([1.5, -0.8, 0.2, -0.9], dtype=torch.float32))

    out = vmoe(x)
    lb_loss = out["load_balance_loss_weighted"]
    assert isinstance(lb_loss, torch.Tensor), "load_balance_loss_weighted must be a torch.Tensor"
    assert lb_loss.requires_grad is True

    vmoe.zero_grad()
    lb_loss.backward()
    assert vmoe.router.weight.grad is not None
    assert float(torch.norm(vmoe.router.weight.grad).item()) > 0.0


def test_massive_swarm_vmoe_load_balance_loss_retains_router_gradient() -> None:
    """#394 / IP-35: MassiveSwarmUpcycledVisionMoE load_balance_loss_weighted must supply router gradients."""
    backbone = FrozenAstroVisionBackbone(
        patch_dim=48, hidden_dim=64, intermediate_dim=128, num_classes=21, seed=202
    )
    deltas = {
        div_id: np.zeros((64, 64), dtype=np.float32)
        for div_id in MASSIVE_IMAGE_MOE_DIVISIONS
    }
    x = torch.randn((8, 4, 48), dtype=torch.float32)

    massive_moe = MassiveSwarmSparseImageMoE(
        backbone=backbone, division_deltas=deltas, top_k=2, load_balance_alpha=0.5
    )
    with torch.no_grad():
        massive_moe.router.bias.copy_(
            torch.tensor([1.8, -0.5, 0.4, -0.7, 0.1, -0.9, 0.2], dtype=torch.float32)
        )

    out = massive_moe(x)
    lb_loss = out["load_balance_loss_weighted"]
    assert isinstance(lb_loss, torch.Tensor), "load_balance_loss_weighted must be a torch.Tensor"
    assert lb_loss.requires_grad is True

    massive_moe.zero_grad()
    lb_loss.backward()
    assert massive_moe.router.weight.grad is not None
    assert float(torch.norm(massive_moe.router.weight.grad).item()) > 0.0


def test_wave3_verifiers_reject_self_reference_and_missing_invariants() -> None:
    """IP-81: verify_numeric_solver_trajectory must reject self-reference and missing invariant_drift."""
    spec = named_spec(
        "harmonic",
        {"omega": 2.0, "zeta": 0.0},
        [1.0, 0.0],
        [0.5, 1.0],
        spec_id="test-harmonic-ip81",
    )
    y_c = [[0.54030230586, -1.6829419696], [-0.41614683654, -1.81859485365]]
    y_f = [[0.54030230587, -1.68294196961], [-0.41614683655, -1.81859485366]]

    # Self-referential trajectory (same object) must fail closed
    self_ref_cert = verify_numeric_solver_trajectory(
        spec, y_c, y_c, invariant_drift=1e-9
    )
    assert self_ref_cert["verdict"] == "FAILED"

    # Omitted invariant_drift must return UNVERIFIED, never VERIFIED
    omitted_drift_cert = verify_numeric_solver_trajectory(spec, y_c, y_f)
    assert omitted_drift_cert["verdict"] == "UNVERIFIED"

    # Invalid tolerances (rtol <= 0 or excessively loose) must fail closed
    bad_tol_cert = verify_numeric_solver_trajectory(
        spec, y_c, y_f, invariant_drift=1e-9, rtol=-1e-8
    )
    assert bad_tol_cert["verdict"] in ("FAILED", "UNVERIFIED")

    # Quantum fluid: omitted reported_defect_charge_sum must return UNVERIFIED, never VERIFIED
    qf_omitted_charge = verify_quantum_fluid_relaxation(
        energy_init=-120.5,
        energy_final=-125.9,
        max_free_force=2.5e-11,
        v_prime_scale=1.0,
        kappa=1.5,
    )
    assert qf_omitted_charge["verdict"] == "UNVERIFIED"


def test_bee_energy_rejects_nan_residual_and_unverified_promotion() -> None:
    """IP-84: evaluate_bee_candidate_energy must reject NaN/Inf/negative residuals with MAX_PAIN_ENERGY."""
    nan_eval = evaluate_bee_candidate_energy(
        candidate_id="c_nan",
        response_text="def solve(x):\n    return x * 2\n",
        tests_passed=4,
        tests_total=4,
        latency_ms=10.0,
        memory_mb=16.0,
        invariant_residual=math.nan,
    )
    assert nan_eval.is_stub_or_invalid is True
    assert nan_eval.total_energy == MAX_PAIN_ENERGY
    assert nan_eval.passed_all_tests is False

    neg_eval = evaluate_bee_candidate_energy(
        candidate_id="c_neg",
        response_text="def solve(x):\n    return x * 2\n",
        tests_passed=4,
        tests_total=4,
        latency_ms=10.0,
        memory_mb=16.0,
        invariant_residual=-0.5,
    )
    assert neg_eval.is_stub_or_invalid is True
    assert neg_eval.total_energy == MAX_PAIN_ENERGY
    assert neg_eval.passed_all_tests is False

    # Promotion gate must not promote without paired_discordant McNemar evidence
    margin = calibrate_noise_margin([0.1, -0.2, 0.15])
    unverified_promo = evaluate_child_bee_promotion(
        parent_energies=[25.0, 22.0, 24.0, 20.0, 23.0],
        child_energies=[2.0, 1.5, 2.2, 1.8, 1.9],
        noise_margin=margin,
    )
    assert unverified_promo.promoted is False

    with pytest.raises(ValueError):
        evaluate_child_bee_promotion(
            parent_energies=[25.0, 22.0, 24.0],
            child_energies=[2.0, 1.5, 2.2],
            noise_margin=0.0,
            paired_discordant=(20, 1),
        )
