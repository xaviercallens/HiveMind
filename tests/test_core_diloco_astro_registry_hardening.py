from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.core.dev_image_video_moe_integration_benchmark import (
    _model_concurrent_modality_stress,
    run_intense_dev_image_video_moe_benchmark,
)
from hivemind.core.diloco_inner import (
    dequantize_tensor_blockwise_8bit,
    quantize_tensor_blockwise_8bit,
)
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer, flatten_delta_dict
from hivemind.core.exoplanet_astro_vmoe_poc import (
    apply_nesterov_outer_momentum,
    build_astrophysics_visual_shard,
)
from hivemind.core.expert_registry import (
    AdapterFootprint,
    ExpertRegistry,
    HierarchicalExpertRouter,
    ServingNode,
    plan_adapter_residency,
    validate_residency_plan,
)


def test_dev_image_video_moe_integration_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="scenarios must be a non-empty sequence"):
        _model_concurrent_modality_stress("developer_coding_moe", "bad-str")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenarios must be a non-empty sequence"):
        _model_concurrent_modality_stress("developer_coding_moe", ["not-a-scenario"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="scenarios must be a sequence"):
        run_intense_dev_image_video_moe_benchmark(scenarios="bad-str")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenarios must be non-empty"):
        run_intense_dev_image_video_moe_benchmark(scenarios=["not-a-scenario"])  # type: ignore[list-item]

    rep = run_intense_dev_image_video_moe_benchmark()
    assert rep["overall_summary"]["total_scenarios"] == 18


def test_diloco_inner_blockwise_8bit_validation() -> None:
    x = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
    q, absmax, shape = quantize_tensor_blockwise_8bit(x, block_size=32, signed=True)
    deq = dequantize_tensor_blockwise_8bit(q, absmax, shape, signed=True)
    assert deq.shape == x.shape

    with pytest.raises(ValueError, match="tensor must be a torch.Tensor"):
        quantize_tensor_blockwise_8bit([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="signed must be a bool"):
        quantize_tensor_blockwise_8bit(x, signed="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="q and absmax must be torch.Tensor"):
        dequantize_tensor_blockwise_8bit([[1]], absmax, shape)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="signed must be a bool"):
        dequantize_tensor_blockwise_8bit(q, absmax, shape, signed=1)  # type: ignore[arg-type]


def test_diloco_outer_flatten_and_init_validation() -> None:
    flat = flatten_delta_dict({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])})
    assert flat.numel() == 3

    with pytest.raises(ValueError, match="must be a non-empty torch.Tensor"):
        flatten_delta_dict({"a": "not-a-tensor"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must contain only finite values"):
        flatten_delta_dict({"a": torch.tensor([1.0, float("nan")])})

    with pytest.raises(ValueError, match="use_cpu_flat_bucket must be a bool"):
        DiLoCoOuterOptimizer(use_cpu_flat_bucket="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outer_lr_eta must be finite and positive"):
        DiLoCoOuterOptimizer(outer_lr_eta=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outer_momentum_mu must be finite"):
        DiLoCoOuterOptimizer(outer_momentum_mu=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="beta1 and beta2 must be in"):
        DiLoCoOuterOptimizer(beta1=True)  # type: ignore[arg-type]


def test_exoplanet_astro_vmoe_shard_and_nesterov_validation() -> None:
    patches, labels = build_astrophysics_visual_shard(
        "expert_1_stellar_transits", num_samples=6, num_patches=4, patch_dim=16, seed=7
    )
    assert patches.shape == (6, 4, 16) and labels.shape == (6,)

    with pytest.raises(ValueError, match="branch_id must be one of"):
        build_astrophysics_visual_shard("unknown_branch")
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        build_astrophysics_visual_shard("expert_1_stellar_transits", num_samples=True)  # type: ignore[arg-type]

    m = np.ones((4, 4), dtype=np.float32)
    v = np.zeros((4, 4), dtype=np.float32)
    g = np.full((4, 4), 0.1, dtype=np.float32)
    m_new, v_new = apply_nesterov_outer_momentum(m, v, g)
    assert m_new.shape == (4, 4) and v_new.shape == (4, 4)

    with pytest.raises(ValueError, match="master_delta must be a non-empty finite numpy.ndarray"):
        apply_nesterov_outer_momentum([[1.0]], v, g)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identical shapes"):
        apply_nesterov_outer_momentum(m, np.zeros((2, 2), dtype=np.float32), g)
    with pytest.raises(ValueError, match="momentum must be finite"):
        apply_nesterov_outer_momentum(m, v, g, momentum=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outer_lr must be a finite positive"):
        apply_nesterov_outer_momentum(m, v, g, outer_lr=0.0)


def test_expert_registry_residency_and_router_validation() -> None:
    node = ServingNode("n1", 8192, "Qwen/Qwen3-4B-Instruct-2507", 4096, 4)
    adapters = {"e1": AdapterFootprint(48, "Qwen/Qwen3-4B-Instruct-2507")}
    plan = plan_adapter_residency([node], adapters, {"e1": 10.0})
    assert validate_residency_plan(plan, [node], adapters) == []

    with pytest.raises(ValueError, match="nodes must be a Sequence of ServingNode"):
        plan_adapter_residency("bad-nodes", adapters, {"e1": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="adapters must be a Mapping of AdapterFootprint"):
        plan_adapter_residency([node], {"e1": "bad"}, {"e1": 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="plan must be a ResidencyPlan"):
        validate_residency_plan("not-a-plan", [node], adapters)  # type: ignore[arg-type]

    reg = ExpertRegistry()
    with pytest.raises(ValueError, match="registry must be an ExpertRegistry"):
        HierarchicalExpertRouter("not-a-reg", "Qwen/Qwen3-4B-Instruct-2507")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_cold_load must be a bool"):
        HierarchicalExpertRouter(reg, "Qwen/Qwen3-4B-Instruct-2507", allow_cold_load="yes")  # type: ignore[arg-type]
