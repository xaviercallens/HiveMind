# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for Wave 3 Scientific MoE Bee S21 ExpertRegistry, routing, residency, and promotion gates."""

from __future__ import annotations

import pytest

from hivemind.core.expert_registry import (
    CANDIDATE,
    LIVE,
    SHADOW,
    WAVE3_PAIRED_MOE_BASE_MODEL_ID,
    WAVE3_PRIMARY_BASE_MODEL_ID,
    WAVE3_SCIENTIFIC_EXPERT_SPECS,
    ExpertRegistry,
    HierarchicalExpertRouter,
    ServingNode,
    evaluate_wave3_promotion_gates,
    plan_adapter_residency,
    validate_residency_plan,
)


def test_register_wave3_scientific_experts_qwen3_and_granite() -> None:
    assert len(WAVE3_SCIENTIFIC_EXPERT_SPECS) == 5

    reg_qwen = ExpertRegistry()
    added_qwen = reg_qwen.register_wave3_scientific_experts(base_model_id=WAVE3_PRIMARY_BASE_MODEL_ID)
    assert len(added_qwen) == 5
    assert set(reg_qwen.ids_in_state(CANDIDATE)) == set(WAVE3_SCIENTIFIC_EXPERT_SPECS)

    reg_granite = ExpertRegistry()
    added_granite = reg_granite.import_wave3_scientific_experts(
        base_model_id=WAVE3_PAIRED_MOE_BASE_MODEL_ID,
        promote_to_shadow=True,
    )
    assert len(added_granite) == 5
    assert set(reg_granite.ids_in_state(SHADOW)) == set(WAVE3_SCIENTIFIC_EXPERT_SPECS)

    with pytest.raises(ValueError, match="not in WAVE3_SUPPORTED_BASE_MODELS"):
        reg_qwen.register_wave3_scientific_experts(base_model_id="unsupported/model")


def test_evaluate_wave3_promotion_gates_g1_g2_g3() -> None:
    passed = evaluate_wave3_promotion_gates(
        in_domain_acc=0.88,
        base_acc=0.62,
        cross_domain_acc=0.59,
        mcnemar_p_value=0.0012,
        wrong_verified_count=0,
    )
    assert passed["passed"] is True
    assert passed["g1_in_domain_lift"] is True
    assert passed["g2_cross_domain_specialization"] is True
    assert passed["g3_zero_wrong_verified"] is True

    # Fail G1 (insufficient lift over base)
    fail_g1 = evaluate_wave3_promotion_gates(
        in_domain_acc=0.68,
        base_acc=0.62,
        cross_domain_acc=0.50,
        mcnemar_p_value=0.001,
        wrong_verified_count=0,
    )
    assert fail_g1["passed"] is False
    assert fail_g1["g1_in_domain_lift"] is False

    # Fail G2 (insufficient cross-domain specialization)
    fail_g2 = evaluate_wave3_promotion_gates(
        in_domain_acc=0.82,
        base_acc=0.60,
        cross_domain_acc=0.78,
        mcnemar_p_value=0.001,
        wrong_verified_count=0,
    )
    assert fail_g2["passed"] is False
    assert fail_g2["g2_cross_domain_specialization"] is False

    # Fail G3 (non-zero wrong answers labeled VERIFIED)
    fail_g3 = evaluate_wave3_promotion_gates(
        in_domain_acc=0.90,
        base_acc=0.60,
        cross_domain_acc=0.55,
        mcnemar_p_value=0.0005,
        wrong_verified_count=1,
    )
    assert fail_g3["passed"] is False
    assert fail_g3["g3_zero_wrong_verified"] is False


def test_wave3_hierarchical_router_and_t4_residency() -> None:
    reg = ExpertRegistry()
    reg.register_wave3_scientific_experts(
        base_model_id=WAVE3_PRIMARY_BASE_MODEL_ID,
        promote_to_shadow=True,
    )
    for eid in WAVE3_SCIENTIFIC_EXPERT_SPECS:
        reg.transition(eid, LIVE, "promoted to live for routing test")

    # Local Tesla T4 node: 15,360 MiB total, leaving >= 4,300 MiB free VRAM headroom
    node = ServingNode(
        node_id="socreateai-agora-hermes-node1",
        vram_mib=15360 - 4300,
        base_model_id=WAVE3_PRIMARY_BASE_MODEL_ID,
        base_resident_mib=3716,
        max_lora_slots=8,
    )
    footprints = reg.residency_inputs(states=(LIVE,))
    demand = {eid: 10.0 for eid in WAVE3_SCIENTIFIC_EXPERT_SPECS}
    plan = plan_adapter_residency([node], footprints, demand)
    assert validate_residency_plan(plan, [node], footprints) == []
    assert set(plan.hot) == set(WAVE3_SCIENTIFIC_EXPERT_SPECS)

    router = HierarchicalExpertRouter(reg, WAVE3_PRIMARY_BASE_MODEL_ID, resident=plan.hot)
    assert router.route("Compute Gross-Pitaevskii quantized vortex and Yukawa crystal relaxation").expert_id == "bee_quantum_fluid"
    assert router.route("Calculate exoplanet transit depth and BAO quintessence Hubble expansion").expert_id == "bee_astrophysics_k3"
    assert router.route("Analyze Pauli master equation chromatin dicodon Jarzynski work").expert_id == "bee_genetic_elenchus"
    assert router.route("Verify persistent homology Betti number Euler characteristic and E8 lattice").expert_id == "bee_topology_tda_lean"
    assert router.route("Solve stiff ODE Robertson and Brusselator IVP with SUNDIALS BDF").expert_id == "bee_numeric_solver_sundials"
