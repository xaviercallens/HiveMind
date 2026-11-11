# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 187 negative-control & invariant hardening tests for Massive Image-MoE, Nocturnal Coding MoE, Quantization, and Qwen3.5 Orchestrator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from hivemind.core.exoplanet_astro_vmoe_poc import FrozenAstroVisionBackbone
from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MASSIVE_IMAGE_MOE_DIVISIONS,
    MassiveSwarmSparseImageMoE,
    SwarmDataShardOrchestrator,
    build_massive_division_visual_shard,
)
from hivemind.core.nocturnal_coding_moe import (
    BIG_GPU_CODING_FLEET,
    CodingQualityLoopIteration,
    CodingSubtaskNode,
    PolyglotCodingBeeAdapter,
    decompose_engineering_epic_to_dag,
    multi_krum_select,
)
from hivemind.core.quantization import (
    NF4QuantState,
    QuantizedModelBundle,
    compute_state_dict_sha256,
    create_lora_config,
    extract_lora_state_dict,
    load_lora_state_dict,
    measure_vram_footprint,
    quantize_tensor_nf4_double_quant,
    resolve_compute_dtype,
)
from hivemind.core.qwen35_usage_moe_orchestrator import (
    ProducerNodeCapability,
    UsageDrivenQwen35MoEOrchestrator,
    build_seven_day_orchestration_plan,
    project_lora_rank_svd,
)


# ============================================================================
# 1. massive_image_moe_swarm_orchestrator.py
# ============================================================================


def test_swarm_shard_assignment_and_orchestrator_reject_malformed_inputs() -> None:
    orch = SwarmDataShardOrchestrator()
    assignment = orch.assign_shard_to_bee(
        "bee-worker-01",
        available_vram_mib=1500.0,
        preferred_division_id="div_1_aesthetics_digital_art",
    )
    assert assignment.division_id == "div_1_aesthetics_digital_art"
    assert assignment.hardware_tier == "tier_1_la_ruche_background_1500mb"

    for str_field in (
        "bee_id",
        "hardware_tier",
        "backend",
        "division_id",
        "division_title",
        "assigned_backbone",
        "hf_dataset_id",
        "shard_uri",
        "optimizer",
        "transport_codec",
    ):
        with pytest.raises(ValueError, match=str_field):
            replace(assignment, **{str_field: ""})
        with pytest.raises(ValueError, match=str_field):
            replace(assignment, **{str_field: "   "})

    with pytest.raises(ValueError, match="preferred_division_id"):
        orch.assign_shard_to_bee("bee-worker-01", preferred_division_id="unknown_div")
    with pytest.raises(ValueError, match="preferred_division_id"):
        orch.assign_shard_to_bee("bee-worker-01", preferred_division_id=123)  # type: ignore[arg-type]


def test_build_massive_division_visual_shard_and_moe_forward_validation() -> None:
    # Small patch_dim (< 7) works cleanly with sliced signature
    patches, labels = build_massive_division_visual_shard(
        division_id="div_1_aesthetics_digital_art",
        num_samples=6,
        num_patches=4,
        patch_dim=4,
        seed=42,
    )
    assert patches.shape == (6, 4, 4)
    assert labels.shape == (6,)

    backbone = FrozenAstroVisionBackbone(
        patch_dim=8,
        hidden_dim=8,
        intermediate_dim=16,
        num_classes=21,
        seed=42,
    )
    delta = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="division_deltas"):
        MassiveSwarmSparseImageMoE(backbone, {"": delta})
    with pytest.raises(ValueError, match="division_deltas"):
        MassiveSwarmSparseImageMoE(backbone, {"   ": delta})

    first_div = next(iter(MASSIVE_IMAGE_MOE_DIVISIONS.keys()))
    moe = MassiveSwarmSparseImageMoE(backbone, {first_div: delta}, top_k=1)
    valid_x = torch.randn(2, 4, 8)
    out = moe(valid_x)
    assert out["logits"].shape == (2, 21)

    with pytest.raises(ValueError, match="visual_patches"):
        moe(torch.ones((2, 4, 8), dtype=torch.bool))
    with pytest.raises(ValueError, match="visual_patches"):
        moe(torch.randn(2, 4, 10))
    bad_x = valid_x.clone()
    bad_x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="visual_patches"):
        moe(bad_x)


# ============================================================================
# 2. nocturnal_coding_moe.py
# ============================================================================


def test_nocturnal_coding_moe_multi_krum_and_specs_reject_malformed_inputs() -> None:
    v0 = np.array([1.0, 2.0], dtype=np.float32)
    v1 = np.array([1.1, 2.1], dtype=np.float32)
    with pytest.raises(ValueError, match=r"vectors\[0\]"):
        multi_krum_select([np.array([True, False]), v1], num_byzantine=0, num_select=1)
    with pytest.raises(ValueError, match=r"vectors\[1\]"):
        multi_krum_select([v0, np.array(["a", "b"])], num_byzantine=0, num_select=1)
    with pytest.raises(ValueError, match="shape"):
        multi_krum_select([v0, np.array([1.0, 2.0, 3.0])], num_byzantine=0, num_select=1)

    spec = BIG_GPU_CODING_FLEET[0]
    for tup_field in ("assigned_cluster_ids", "assigned_bee_ids", "primary_backbones"):
        with pytest.raises(ValueError, match=tup_field):
            replace(spec, **{tup_field: ()})
        with pytest.raises(ValueError, match=tup_field):
            replace(spec, **{tup_field: ("",)})
        with pytest.raises(ValueError, match=tup_field):
            replace(spec, **{tup_field: ["valid"]})  # type: ignore[arg-type]


def test_polyglot_coding_bee_adapter_and_quality_loop_validation() -> None:
    adapter = PolyglotCodingBeeAdapter(d_model=16, d_ff=32, num_classes=8, lora_rank=4, seed=7)
    assert adapter.verify_base_frozen() is True
    x = torch.randn(4, 16)
    logits = adapter(x)
    assert logits.shape == (4, 8)
    vec = adapter.export_lora_delta_vector()
    assert vec.ndim == 1 and vec.size > 0

    with pytest.raises(ValueError, match="x must be a finite 2D tensor"):
        adapter(torch.ones((4, 16), dtype=torch.bool))
    with pytest.raises(ValueError, match="x must be a finite 2D tensor"):
        adapter(torch.randn(4, 8))
    with pytest.raises(ValueError, match="x must be a finite 2D tensor"):
        adapter(torch.randn(4, 16, 1))
    nan_x = x.clone()
    nan_x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="x must be a finite 2D tensor"):
        adapter(nan_x)

    loop_it = CodingQualityLoopIteration(
        loop_index=1,
        loop_title="Tome 1",
        refinement_strategy="DiLoCo",
        distributed_nodes_active=8,
        inner_steps_executed=500,
        heldout_hard_eval_loss=1.25,
        heldout_hard_routing_acc=0.88,
        heldout_top2_expert_acc=0.95,
        delta_energy_vs_prev_loop=-0.35,
        monotone_ratchet_accepted=True,
        mean_bee_sqnr_db=22.4,
        mean_wire_compression_x=18.5,
        all_zk_opoi_verified=True,
        all_node_vram_within_budget=True,
        polyglot_loop_benchmarks={"Python": {"PassAt1": 85.0}},
    )
    with pytest.raises(ValueError, match="loop_index"):
        replace(loop_it, loop_index=0)
    with pytest.raises(ValueError, match="heldout_hard_eval_loss"):
        replace(loop_it, heldout_hard_eval_loss=0.0)
    with pytest.raises(ValueError, match="monotone_ratchet_accepted"):
        replace(loop_it, monotone_ratchet_accepted=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="polyglot_loop_benchmarks"):
        replace(loop_it, polyglot_loop_benchmarks={})

    subtask = decompose_engineering_epic_to_dag("Test Epic", target_languages=("Rust", "Python"))[0]
    assert isinstance(subtask, CodingSubtaskNode)
    with pytest.raises(ValueError, match="target_languages"):
        decompose_engineering_epic_to_dag("Test Epic", target_languages="Rust")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_languages"):
        decompose_engineering_epic_to_dag("Test Epic", target_languages=("",))


# ============================================================================
# 3. quantization.py
# ============================================================================


def test_nf4_quant_state_and_quantized_model_bundle_reject_malformed_inputs() -> None:
    state = NF4QuantState(
        packed_4bit=torch.zeros(8, dtype=torch.uint8),
        q_absmax_uint8=torch.ones(2, dtype=torch.uint8),
        superblock_scale=0.5,
        original_shape=(4, 4),
        block_size=8,
    )
    assert state.storage_bytes == 14

    with pytest.raises(ValueError, match="packed_4bit"):
        replace(state, packed_4bit=torch.ones(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="q_absmax_uint8"):
        replace(state, q_absmax_uint8=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="superblock_scale"):
        replace(state, superblock_scale=0.0)
    with pytest.raises(ValueError, match="original_shape"):
        replace(state, original_shape=(4, 0))
    with pytest.raises(ValueError, match="block_size"):
        replace(state, block_size=0)

    lin = nn.Linear(4, 4)
    bundle = QuantizedModelBundle(
        model=lin,
        base_model_id="Qwen/Qwen2.5-0.5B",
        base_model_sha256="a" * 64,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        is_4bit_nf4=True,
        nf4_compressed_base_mb=12.5,
        trainable_params=16,
        total_params=36,
        lora_size_mb=0.25,
    )
    with pytest.raises(ValueError, match="model must be an nn.Module"):
        replace(bundle, model="not_a_module")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device must be a torch.device"):
        replace(bundle, device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compute_dtype must be a torch.dtype"):
        replace(bundle, compute_dtype="float32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nf4_compressed_base_mb"):
        replace(bundle, nf4_compressed_base_mb=-0.1)
    with pytest.raises(ValueError, match="nf4_compressed_base_mb"):
        replace(bundle, nf4_compressed_base_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lora_size_mb"):
        replace(bundle, lora_size_mb=float("nan"))
    with pytest.raises(ValueError, match="total_params"):
        replace(bundle, total_params=8)


def test_quantization_functions_reject_boolean_empty_and_non_finite_tensors() -> None:
    with pytest.raises(ValueError, match="weight"):
        quantize_tensor_nf4_double_quant(torch.ones((4, 4), dtype=torch.bool), block_size=4)
    with pytest.raises(ValueError, match="weight"):
        quantize_tensor_nf4_double_quant(torch.tensor([[1.0, float("nan")]]), block_size=4)
    with pytest.raises(ValueError, match="block_size"):
        quantize_tensor_nf4_double_quant(torch.ones((4, 4)), block_size=0)

    with pytest.raises(ValueError, match="state_dict"):
        compute_state_dict_sha256({"": torch.ones(4)})
    with pytest.raises(ValueError, match="state_dict"):
        compute_state_dict_sha256({"w": torch.ones(4, dtype=torch.bool)})
    with pytest.raises(ValueError, match="state_dict"):
        compute_state_dict_sha256({"w": torch.tensor([1.0, float("inf")])})

    lin = nn.Linear(4, 4, bias=False)
    extracted = extract_lora_state_dict(lin)
    assert "weight" in extracted

    with pytest.raises(ValueError, match="lora_state"):
        load_lora_state_dict(lin, {"": torch.ones(4, 4)})
    with pytest.raises(ValueError, match="lora_state"):
        load_lora_state_dict(lin, {"weight": torch.ones((4, 4), dtype=torch.bool)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_lora_state_dict(lin, {"weight": torch.ones((2, 2))})

    with pytest.raises(ValueError, match="rank_r"):
        create_lora_config(rank_r=0)
    with pytest.raises(ValueError, match="lora_alpha"):
        create_lora_config(lora_alpha=0)
    with pytest.raises(ValueError, match="lora_dropout"):
        create_lora_config(lora_dropout=1.0)

    vram_info = measure_vram_footprint(torch.device("cpu"))
    assert vram_info["allocated_mb"] == 0.0
    assert resolve_compute_dtype("bfloat16", torch.device("cpu")) == torch.float32


# ============================================================================
# 4. qwen35_usage_moe_orchestrator.py
# ============================================================================


def test_qwen35_orchestrator_and_svd_projection_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="observed_completions_by_cluster must be a mapping"):
        UsageDrivenQwen35MoEOrchestrator(observed_completions_by_cluster=["coding"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observed_completions_by_cluster keys"):
        UsageDrivenQwen35MoEOrchestrator(observed_completions_by_cluster={" ": 10})
    with pytest.raises(ValueError, match="energy_failure_rate_by_bee must be a mapping"):
        UsageDrivenQwen35MoEOrchestrator(energy_failure_rate_by_bee=["bee"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="energy_failure_rate_by_bee keys"):
        UsageDrivenQwen35MoEOrchestrator(energy_failure_rate_by_bee={"": 0.2})

    orch = UsageDrivenQwen35MoEOrchestrator()
    shares = orch.compute_cluster_target_shares()
    assert pytest.approx(sum(shares.values()), abs=1e-3) == 1.0

    # Test CPU audit and inference priority nodes are skipped during workunit scheduling
    cpu_node = ProducerNodeCapability(
        node_id="node-cpu",
        gpu_model="CPU Only",
        vram_mib=2048,
        fp16_tflops=1.0,
        idle_seconds=120.0,
        is_european_night=True,
    )
    busy_node = ProducerNodeCapability(
        node_id="node-busy",
        gpu_model="RTX 3090",
        vram_mib=24576,
        fp16_tflops=35.0,
        idle_seconds=120.0,
        is_european_night=True,
        active_inference_requests=2,
    )
    assert cpu_node.training_mode == "zk_opoi_cpu_audit"
    assert busy_node.training_mode == "inference_priority"
    assert orch.schedule_workunits_for_producers([cpu_node, busy_node], round_index=1) == []

    # Test SVD rank projection validation
    a = torch.randn(8, 4)
    b = torch.randn(4, 8)
    with pytest.raises(ValueError, match="non-boolean"):
        project_lora_rank_svd(torch.ones((4, 4), dtype=torch.bool), torch.ones((4, 4)), target_rank=2)
    with pytest.raises(ValueError, match="Inner rank mismatch"):
        project_lora_rank_svd(torch.randn(4, 8), torch.randn(8, 6), target_rank=2)
    # target_rank=6 > min(8, 4, 4) = 4 should raise ValueError
    with pytest.raises(ValueError, match="target_rank"):
        project_lora_rank_svd(a, b, target_rank=6)

    plan: dict[str, Any] = build_seven_day_orchestration_plan()
    assert len(plan["days"]) == 6
    assert len(plan["plan_sha256"]) == 64
