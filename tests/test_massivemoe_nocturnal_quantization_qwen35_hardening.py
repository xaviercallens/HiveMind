# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for Massive Image-MoE, Nocturnal Coding MoE, Quantization, and Qwen3.5 Usage Orchestrator (PR-IP105)."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch
import torch.nn as nn

from hivemind.core.exoplanet_astro_vmoe_poc import FrozenAstroVisionBackbone
from hivemind.core.massive_image_moe_swarm_orchestrator import (
    MassiveSwarmSparseImageMoE,
    SwarmDataShardOrchestrator,
)
from hivemind.core.nocturnal_coding_moe import (
    BIG_GPU_CODING_FLEET,
    CodingSubtaskNode,
    multi_krum_select,
)
from hivemind.core.quantization import (
    compute_state_dict_sha256,
    load_lora_state_dict,
    quantize_tensor_nf4_double_quant,
    resolve_compute_dtype,
)
from hivemind.core.qwen35_usage_moe_orchestrator import (
    ProducerNodeCapability,
    UsageDrivenQwen35MoEOrchestrator,
    project_lora_rank_svd,
)


def test_massive_image_moe_swarm_orchestrator_rejects_malformed_inputs() -> None:
    orch = SwarmDataShardOrchestrator()
    assignment = orch.assign_shard_to_bee("bee-test-01", available_vram_mib=1500.0)
    assert assignment.samples_in_shard == 5000

    with pytest.raises(ValueError, match="bee_id"):
        replace(assignment, bee_id="")
    with pytest.raises(ValueError, match="available_vram_mib"):
        replace(assignment, available_vram_mib=0.0)
    with pytest.raises(ValueError, match="available_vram_mib"):
        replace(assignment, available_vram_mib=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shard_index"):
        replace(assignment, shard_index=-1)
    with pytest.raises(ValueError, match="samples_in_shard"):
        replace(assignment, samples_in_shard=0)

    backbone = FrozenAstroVisionBackbone(patch_dim=16, hidden_dim=16, intermediate_dim=32, num_classes=3, seed=42)
    delta = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="backbone"):
        MassiveSwarmSparseImageMoE("bad_backbone", {"custom_div": delta})  # type: ignore[arg-type]

    moe = MassiveSwarmSparseImageMoE(backbone, {"custom_div": delta}, top_k=1)
    with pytest.raises(ValueError, match="visual_patches"):
        moe.forward(torch.randn(4, 16))


def test_nocturnal_coding_moe_rejects_malformed_inputs() -> None:
    v0 = np.array([1.0, 2.0])
    v1 = np.array([1.1, 2.1])
    v2 = np.array([100.0, -100.0])
    sel = multi_krum_select([v0, v1, v2], num_byzantine=1, num_select=2)
    assert set(sel) == {0, 1}

    with pytest.raises(ValueError, match="vectors"):
        multi_krum_select("not-a-list", num_byzantine=0, num_select=1)  # type: ignore[arg-type]

    subtask = CodingSubtaskNode(
        subtask_id="sub-1",
        role_slug="coding",
        assigned_cluster_id="expert_5_polyglot_implementation",
        target_language="Rust",
        description="Implement SIMD kernel",
        depends_on=["sub-0"],
        verification_benchmark="cargo test",
    )
    with pytest.raises(ValueError, match="subtask_id"):
        replace(subtask, subtask_id=" ")
    with pytest.raises(ValueError, match="depends_on"):
        replace(subtask, depends_on=[""])

    base_node = BIG_GPU_CODING_FLEET[0]
    with pytest.raises(ValueError, match="node_id"):
        replace(base_node, node_id="")
    with pytest.raises(ValueError, match="total_vram_mib"):
        replace(base_node, total_vram_mib=-1.0)
    with pytest.raises(ValueError, match="lora_rank"):
        replace(base_node, lora_rank=0)


def test_quantization_rejects_malformed_inputs() -> None:
    w = torch.randn(8, 16)
    state, recon = quantize_tensor_nf4_double_quant(w, block_size=8)
    assert recon.shape == (8, 16)
    assert state.storage_bytes > 0

    with pytest.raises(ValueError, match="weight"):
        quantize_tensor_nf4_double_quant([[1.0, 2.0]], block_size=8)  # type: ignore[arg-type]

    assert resolve_compute_dtype("float16", torch.device("cpu")) == torch.float32
    with pytest.raises(ValueError, match="dtype_name"):
        resolve_compute_dtype(" ", torch.device("cpu"))
    with pytest.raises(ValueError, match="device"):
        resolve_compute_dtype("float16", "cpu")  # type: ignore[arg-type]

    digest = compute_state_dict_sha256({"w": w})
    assert len(digest) == 64
    with pytest.raises(ValueError, match="state_dict"):
        compute_state_dict_sha256({})
    with pytest.raises(ValueError, match="state_dict"):
        compute_state_dict_sha256({"w": [1.0]})  # type: ignore[dict-item]

    lin = nn.Linear(4, 4, bias=False)
    load_lora_state_dict(lin, {"weight": torch.ones(4, 4)})
    with pytest.raises(ValueError, match="model"):
        load_lora_state_dict("not_a_module", {"weight": torch.ones(4, 4)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lora_state"):
        load_lora_state_dict(lin, {})


def test_qwen35_usage_moe_orchestrator_rejects_malformed_inputs() -> None:
    orch = UsageDrivenQwen35MoEOrchestrator()
    node = ProducerNodeCapability(
        node_id="node-01",
        gpu_model="RTX 2070",
        vram_mib=8192,
        fp16_tflops=14.9,
        idle_seconds=120.0,
        is_european_night=True,
    )
    wus = orch.schedule_workunits_for_producers([node], round_index=1)
    assert len(wus) == 1
    wu = wus[0]

    with pytest.raises(ValueError, match="workunit_id"):
        replace(wu, workunit_id="")
    with pytest.raises(ValueError, match="lora_rank"):
        replace(wu, lora_rank=0)
    with pytest.raises(ValueError, match="prorated_reward_hmt"):
        replace(wu, prorated_reward_hmt=0.0)

    with pytest.raises(ValueError, match="producers"):
        orch.schedule_workunits_for_producers("bad", round_index=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="producers"):
        orch.schedule_workunits_for_producers(["not_a_node"], round_index=1)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="lora_a and lora_b"):
        project_lora_rank_svd([[1.0]], torch.ones(4, 4), target_rank=2)  # type: ignore[arg-type]
