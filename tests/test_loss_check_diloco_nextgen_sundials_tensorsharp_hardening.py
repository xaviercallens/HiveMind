# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 159 hardening tests for CrowdGPT loss check, DiLoCo next-gen, rusty-SUNDIALS bee, and TensorSharp bridge."""

from __future__ import annotations

import pytest

from hivemind.swarm.crowdgpt_loss_check import LossCheckVerdict
from hivemind.swarm.diloco_nextgen import CGADStepTelemetry
from hivemind.swarm.rusty_sundials_scientific_bee import (
    NumericRescueResult,
    PhysicalMathematicalDiagnosis,
    analyze_physical_mathematical_problem,
    rescue_numeric_calculation,
)
from hivemind.swarm.tensorsharp_collab_bridge import (
    TENSORSHARP_BENCHMARK_CATALOG,
    TensorSharpMoEOffloadPlan,
    plan_tensorsharp_moe_offload,
)


def test_loss_check_verdict_invariants() -> None:
    v = LossCheckVerdict(accepted=True, reason="ACCEPTED", loss_before=2.1, loss_after=2.0, delta_loss=-0.1)
    assert v.accepted is True

    with pytest.raises(TypeError, match="accepted"):
        LossCheckVerdict(accepted=1, reason="ACCEPTED")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason"):
        LossCheckVerdict(accepted=False, reason="  ")
    with pytest.raises(TypeError, match="loss_before"):
        LossCheckVerdict(accepted=False, reason="REJECTED", loss_before=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="checked"):
        LossCheckVerdict(accepted=False, reason="REJECTED", checked=("not-a-dict",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="detail"):
        LossCheckVerdict(accepted=False, reason="REJECTED", detail=None)  # type: ignore[arg-type]


def test_cgad_step_telemetry_invariants() -> None:
    t = CGADStepTelemetry(
        fragment_name="layers.0.q_proj",
        worker_delay_tau=1.0,
        fragment_age_rounds=0.5,
        effective_tau=1.5,
        cosine_gate_gamma=0.8,
        decay_gate_sigma=0.7,
        dropped_by_cutoff=False,
        staleness_bias_bound=0.36,
        step_linf_norm=0.01,
    )
    assert t.dropped_by_cutoff is False

    with pytest.raises(ValueError, match="fragment_name"):
        CGADStepTelemetry(
            fragment_name=" ",
            worker_delay_tau=1.0,
            fragment_age_rounds=0.5,
            effective_tau=1.5,
            cosine_gate_gamma=0.8,
            decay_gate_sigma=0.7,
            dropped_by_cutoff=False,
            staleness_bias_bound=0.36,
            step_linf_norm=0.01,
        )
    with pytest.raises(TypeError, match="dropped_by_cutoff"):
        CGADStepTelemetry(
            fragment_name="layers.0.q_proj",
            worker_delay_tau=1.0,
            fragment_age_rounds=0.5,
            effective_tau=1.5,
            cosine_gate_gamma=0.8,
            decay_gate_sigma=0.7,
            dropped_by_cutoff=0,  # type: ignore[arg-type]
            staleness_bias_bound=0.36,
            step_linf_norm=0.01,
        )
    with pytest.raises(ValueError, match="step_linf_norm"):
        CGADStepTelemetry(
            fragment_name="layers.0.q_proj",
            worker_delay_tau=1.0,
            fragment_age_rounds=0.5,
            effective_tau=1.5,
            cosine_gate_gamma=0.8,
            decay_gate_sigma=0.7,
            dropped_by_cutoff=False,
            staleness_bias_bound=0.36,
            step_linf_norm=-0.1,
        )


def test_rusty_sundials_diagnosis_and_rescue_invariants() -> None:
    diag = analyze_physical_mathematical_problem("Kepler orbit symplectic Hamiltonian")
    assert isinstance(diag, PhysicalMathematicalDiagnosis)
    res = rescue_numeric_calculation("Kepler orbit symplectic Hamiltonian")
    assert isinstance(res, NumericRescueResult)

    with pytest.raises(ValueError, match="domain"):
        PhysicalMathematicalDiagnosis(
            domain="",
            problem_subtype="kepler",
            equation_class="ODE",
            stiffness_ratio_estimate=1.0,
            is_stiff=False,
            underlying_mathematics="Hamiltonian",
            underlying_physics="Gravity",
            naive_failure_mechanism="Drift",
            conserved_invariants=["Energy"],
            recommended_rusty_sundials_crate="arkode",
            recommended_solver_scheme="Dual-RRK",
            lean4_reference_theorem="Kepler.energy",
        )
    with pytest.raises(TypeError, match="diagnosis"):
        NumericRescueResult(
            domain="astronomy",
            diagnosis="not-a-diagnosis",  # type: ignore[arg-type]
            steps_executed=10,
            dt=0.01,
            naive_scheme_name="Euler",
            naive_failed=True,
            naive_failure_reason="Drift",
            naive_invariant_drift=1.0,
            rescued_scheme_name="Dual-RRK",
            rescued_succeeded=True,
            rescued_invariant_drift=1e-9,
            improvement_factor=1e9,
            invariants_verified={},
            sample_trajectory=[],
            rust_reference_snippet="fn main() {}",
            energy_score_e=0.01,
        )


def test_tensorsharp_moe_offload_plan_invariants() -> None:
    first_model_id = next(iter(TENSORSHARP_BENCHMARK_CATALOG))
    plan = plan_tensorsharp_moe_offload(first_model_id)
    assert isinstance(plan, TensorSharpMoEOffloadPlan)
    assert plan.expected_pp8192_tok_s > 0.0

    with pytest.raises(ValueError, match="model_id"):
        TensorSharpMoEOffloadPlan(
            model_id="",
            v1_models_id="Qwen3",
            gpu_name="T4",
            compute_capability="compute_75",
            total_vram_mib=15360,
            reserved_free_vram_mib=2048,
            usable_vram_mib=13312,
            total_moe_layers=48,
            n_cpu_moe_layers=24,
            estimated_peak_vram_mib=12000,
            estimated_pp8192_tok_s=100.0,
            estimated_tg128_tok_s=20.0,
            fits_vram_budget=True,
            cli_flags=["--backend", "ggml_cuda"],
        )
    with pytest.raises(TypeError, match="cli_flags"):
        TensorSharpMoEOffloadPlan(
            model_id="qwen3-coder-30b-a3b-iq4-xs",
            v1_models_id="Qwen3",
            gpu_name="T4",
            compute_capability="compute_75",
            total_vram_mib=15360,
            reserved_free_vram_mib=2048,
            usable_vram_mib=13312,
            total_moe_layers=48,
            n_cpu_moe_layers=24,
            estimated_peak_vram_mib=12000,
            estimated_pp8192_tok_s=100.0,
            estimated_tg128_tok_s=20.0,
            fits_vram_budget=True,
            cli_flags=[123],  # type: ignore[list-item]
        )
