# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening tests for core catalog, Qwen 3.5, auto-improvement, and coding MoE.

Covers:
1. `hivemind.core.base_model_catalog`:
   - `TrainRecipe`, `ServeConfig`, `fit_verdict`, `calibrate_from_measurement`, `binom_half_lower_tail`,
     `mcnemar_exact_p`, and `PromotionPolicy` reject boolean-as-numeric, non-finite, and out-of-range inputs.
2. `hivemind.core.qwen35_usage_moe_orchestrator`:
   - `ProducerNodeCapability`, `UsageDrivenQwen35MoEOrchestrator`, and `project_lora_rank_svd` reject
     empty/control-char strings, boolean-as-numeric fields, non-finite values, and invalid rank bounds.
3. `hivemind.core.auto_improvement`:
   - `select_dream_phase_teacher_for_vram`, `resolve_domain_canonical`, `AutoImprovementWorkunit`,
     `NocturnalAutoImprovementScheduler`, and `PointsRetrofitManager` reject boolean-as-numeric, non-finite,
     and control-char inputs.
4. `hivemind.core.nocturnal_coding_moe`:
   - `multi_krum_select`, `decompose_engineering_epic_to_dag`, `PolyglotCodingBeeAdapter`, and
     `run_nocturnal_coding_moe_campaign` reject boolean-as-numeric, non-finite, and out-of-range inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.core.auto_improvement import (
    AutoImprovementWorkunit,
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
    resolve_domain_canonical,
    select_dream_phase_teacher_for_vram,
)
from hivemind.core.base_model_catalog import (
    PromotionPolicy,
    ServeConfig,
    TrainRecipe,
    VramEstimate,
    binom_half_lower_tail,
    calibrate_from_measurement,
    fit_verdict,
    mcnemar_exact_p,
)
from hivemind.core.nocturnal_coding_moe import (
    PolyglotCodingBeeAdapter,
    decompose_engineering_epic_to_dag,
    multi_krum_select,
    run_nocturnal_coding_moe_campaign,
)
from hivemind.core.qwen35_usage_moe_orchestrator import (
    ProducerNodeCapability,
    UsageDrivenQwen35MoEOrchestrator,
    project_lora_rank_svd,
)


def test_base_model_catalog_train_and_serve_config_validation() -> None:
    recipe = TrainRecipe(seq_len=512, micro_batch=1, lora_rank=16)
    assert recipe.label == "seq512 mb1 r16 all_linear"

    with pytest.raises(ValueError, match="seq_len, micro_batch and lora_rank must be >= 1"):
        TrainRecipe(seq_len=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seq_len, micro_batch and lora_rank must be >= 1"):
        TrainRecipe(micro_batch=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seq_len, micro_batch and lora_rank must be >= 1"):
        TrainRecipe(lora_rank=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gradient_checkpointing and load_aux_towers must be bool"):
        TrainRecipe(gradient_checkpointing=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="act_scale must be > 0"):
        TrainRecipe(act_scale=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="act_scale must be > 0"):
        TrainRecipe(act_scale=float("nan"))

    serve = ServeConfig(quant="Q4_0", context_len=4096, n_parallel=1)
    assert serve.label == "Q4_0 ctx4096 x1"

    with pytest.raises(ValueError, match="context_len and n_parallel must be >= 1"):
        ServeConfig(context_len=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="context_len and n_parallel must be >= 1"):
        ServeConfig(n_parallel=True)  # type: ignore[arg-type]


def test_base_model_catalog_verdict_calibration_and_mcnemar_validation() -> None:
    v = fit_verdict(4000.0, 6000.0, margin=0.20)
    assert v.fits is True

    with pytest.raises(ValueError, match="margin >= 0, budget > 0 and estimate >= 0"):
        fit_verdict(True, 6000.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="margin >= 0, budget > 0 and estimate >= 0"):
        fit_verdict(4000.0, float("inf"))

    with pytest.raises(ValueError, match="margin >= 0, budget > 0 and estimate >= 0"):
        fit_verdict(4000.0, 6000.0, margin=True)  # type: ignore[arg-type]

    est = VramEstimate(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        mode="qlora_nf4_training",
        config="seq512 mb1 r16 all_linear",
        components_mib={"nf4_weights": 3500.0, "cuda_context_allowance": 307.0},
        total_mib=3807.0,
    )
    cal = calibrate_from_measurement(
        est,
        3900.0,
        measured_kind="process_peak",
        gpu="RTX 2070",
        source="pytest",
    )
    assert cal.within_margin is True

    with pytest.raises(ValueError, match="measured_peak_mib"):
        calibrate_from_measurement(
            est,
            True,  # type: ignore[arg-type]
            measured_kind="process_peak",
            gpu="RTX 2070",
            source="pytest",
        )

    with pytest.raises(ValueError, match="margin must be a non-negative finite number"):
        calibrate_from_measurement(
            est,
            3900.0,
            measured_kind="process_peak",
            gpu="RTX 2070",
            source="pytest",
            margin=True,  # type: ignore[arg-type]
        )

    assert binom_half_lower_tail(0, 4) == pytest.approx(1.0 / 16.0)
    with pytest.raises(ValueError, match="k and n must be integers"):
        binom_half_lower_tail(True, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k and n must be integers"):
        binom_half_lower_tail(1, -1)

    with pytest.raises(ValueError, match="discordant counts must be >= 0"):
        mcnemar_exact_p(True, 5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="alpha must be in"):
        PromotionPolicy(alpha=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="min_items, min_guard_items >= 1"):
        PromotionPolicy(min_items=True)  # type: ignore[arg-type]


def test_qwen35_producer_capability_and_orchestrator_validation() -> None:
    cap = ProducerNodeCapability(
        node_id="node-rtx2070-01",
        gpu_model="RTX 2070",
        vram_mib=8192,
        fp16_tflops=14.9,
        idle_seconds=120.0,
        is_european_night=True,
        active_inference_requests=0,
    )
    assert cap.hardware_tier == "std_8gb"
    assert cap.training_mode == "nocturnal_deep_qlora"

    with pytest.raises(ValueError, match="node_id"):
        ProducerNodeCapability(
            node_id="bad\x00id",
            gpu_model="RTX 2070",
            vram_mib=8192,
            fp16_tflops=14.9,
            idle_seconds=120.0,
            is_european_night=True,
        )

    with pytest.raises(ValueError, match="vram_mib"):
        ProducerNodeCapability(
            node_id="node-01",
            gpu_model="RTX 2070",
            vram_mib=True,  # type: ignore[arg-type]
            fp16_tflops=14.9,
            idle_seconds=120.0,
            is_european_night=True,
        )

    with pytest.raises(ValueError, match="fp16_tflops"):
        ProducerNodeCapability(
            node_id="node-01",
            gpu_model="RTX 2070",
            vram_mib=8192,
            fp16_tflops=float("nan"),
            idle_seconds=120.0,
            is_european_night=True,
        )

    with pytest.raises(ValueError, match="is_european_night"):
        ProducerNodeCapability(
            node_id="node-01",
            gpu_model="RTX 2070",
            vram_mib=8192,
            fp16_tflops=14.9,
            idle_seconds=120.0,
            is_european_night=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="active_inference_requests"):
        ProducerNodeCapability(
            node_id="node-01",
            gpu_model="RTX 2070",
            vram_mib=8192,
            fp16_tflops=14.9,
            idle_seconds=120.0,
            is_european_night=True,
            active_inference_requests=-1,
        )

    with pytest.raises(ValueError, match="observed_completions_by_cluster"):
        UsageDrivenQwen35MoEOrchestrator(observed_completions_by_cluster={"coding": True})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="energy_failure_rate_by_bee"):
        UsageDrivenQwen35MoEOrchestrator(energy_failure_rate_by_bee={"bee-qwen35-python-swe": 1.5})

    orch = UsageDrivenQwen35MoEOrchestrator()
    with pytest.raises(ValueError, match="round_index"):
        orch.schedule_workunits_for_producers([cap], round_index=0)
    with pytest.raises(ValueError, match="round_index"):
        orch.schedule_workunits_for_producers([cap], round_index=True)  # type: ignore[arg-type]


def test_qwen35_project_lora_rank_svd_validation() -> None:
    torch.manual_seed(7)
    a = torch.randn(32, 64)
    b = torch.randn(64, 32)
    sa, sb, ratio = project_lora_rank_svd(a, b, target_rank=16)
    assert sa.shape == (16, 64)
    assert sb.shape == (64, 16)
    assert 0.0 < ratio <= 1.0

    with pytest.raises(ValueError, match="target_rank"):
        project_lora_rank_svd(a, b, target_rank=True)  # type: ignore[arg-type]

    bad_a = a.clone()
    bad_a[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        project_lora_rank_svd(bad_a, b, target_rank=16)


def test_auto_improvement_validation() -> None:
    teacher = select_dream_phase_teacher_for_vram(8.0, "coding", system_ram_gb=16.0)
    assert teacher["execution_mode"] == "moe_expert_cpu_ram_split_8gb_wddm_safe"

    with pytest.raises(ValueError, match="vram_gb"):
        select_dream_phase_teacher_for_vram(True, "coding")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="system_ram_gb"):
        select_dream_phase_teacher_for_vram(8.0, "coding", system_ram_gb=-4.0)

    with pytest.raises(ValueError, match="domain_or_cluster"):
        select_dream_phase_teacher_for_vram(8.0, "bad\ncluster")

    assert resolve_domain_canonical("maths") == "maths_olympiad"
    with pytest.raises(ValueError, match="domain must be a non-empty printable string"):
        resolve_domain_canonical("   ")

    with pytest.raises(ValueError, match="rank must be an integer >= 1"):
        AutoImprovementWorkunit(
            wu_id="wu-1",
            domain="maths_formal",
            base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            target_adapter_name="adapter-1",
            rank=True,  # type: ignore[arg-type]
            num_samples=10,
            inner_steps=10,
            min_vram_gb=6.0,
            time_budget_minutes=15.0,
            seed=42,
            created_utc=1700000000.0,
        )

    with pytest.raises(ValueError, match="default_rank"):
        NocturnalAutoImprovementScheduler(default_rank=0)

    with pytest.raises(ValueError, match="target_vram_ceiling_gb"):
        NocturnalAutoImprovementScheduler(target_vram_ceiling_gb=True)  # type: ignore[arg-type]

    sched = NocturnalAutoImprovementScheduler()
    with pytest.raises(ValueError, match="num_samples"):
        sched.create_domain_workunit("maths_formal", num_samples=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="time_budget_minutes"):
        sched.create_domain_workunit("maths_formal", time_budget_minutes=-5.0)

    with pytest.raises(ValueError, match="base_correct"):
        sched.evaluate_mcnemar_gate(
            base_correct=True,  # type: ignore[arg-type]
            adapted_correct=120,
            base_only_correct=5,
            adapted_only_correct=25,
            total_eval_samples=200,
        )

    with pytest.raises(ValueError, match="significance_alpha"):
        sched.evaluate_mcnemar_gate(
            base_correct=100,
            adapted_correct=120,
            base_only_correct=5,
            adapted_only_correct=25,
            total_eval_samples=200,
            significance_alpha=1.0,
        )

    with pytest.raises(ValueError, match="base_wu_reward"):
        PointsRetrofitManager(base_wu_reward=-1.0)

    with pytest.raises(ValueError, match="step_rate"):
        PointsRetrofitManager(step_rate=True)  # type: ignore[arg-type]


def test_nocturnal_coding_moe_validation() -> None:
    vecs = [np.ones(8, dtype=np.float32) * i for i in range(6)]
    sel = multi_krum_select(vecs, num_byzantine=1, num_select=3)
    assert len(sel) == 3

    with pytest.raises(ValueError, match="vectors must be non-empty"):
        multi_krum_select([], num_byzantine=0, num_select=1)

    with pytest.raises(ValueError, match="num_byzantine"):
        multi_krum_select(vecs, num_byzantine=True, num_select=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_select"):
        multi_krum_select(vecs, num_byzantine=1, num_select=0)

    bad_vecs = [np.array([1.0, float("nan")], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32)]
    with pytest.raises(ValueError, match="finite"):
        multi_krum_select(bad_vecs, num_byzantine=0, num_select=1)

    dag = decompose_engineering_epic_to_dag("Hardened Polyglot Kernel")
    assert len(dag) == 8

    with pytest.raises(ValueError, match="epic_title"):
        decompose_engineering_epic_to_dag("   ")

    with pytest.raises(ValueError, match="d_model"):
        PolyglotCodingBeeAdapter(d_model=0)

    with pytest.raises(ValueError, match="lora_alpha"):
        PolyglotCodingBeeAdapter(lora_alpha=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="inner_steps"):
        run_nocturnal_coding_moe_campaign(inner_steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lora_alpha"):
        run_nocturnal_coding_moe_campaign(lora_alpha=-1.0)
