"""Regression tests for IP-185 boundary validation across diloco_inner, diloco_outer, exoplanet_astro_vmoe_poc, and expert_registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hivemind.core.compression import compress_lora_delta
from hivemind.core.diloco_inner import (
    Blockwise8BitAdamW,
    InnerLoopResult,
    dequantize_tensor_blockwise_8bit,
    get_deterministic_batch,
    quantize_tensor_blockwise_8bit,
    replay_inner_segment,
    run_diloco_inner_loop,
)
from hivemind.core.diloco_outer import (
    ByzantineFilterReport,
    DiLoCoOuterOptimizer,
    OuterStepResult,
    flatten_delta_dict,
)
from hivemind.core.exoplanet_astro_vmoe_poc import (
    FrozenAstroVisionBackbone,
    SwarmFFNAdapter,
    UpcycledAstroSparseVisionMoE,
    asymmetric_focal_loss,
    compute_citizen_science_entropy_weights,
)
from hivemind.core.expert_registry import (
    LIVE,
    LORA_DOMAIN,
    SERVING,
    SHADOW,
    AdapterFootprint,
    ExpertRecord,
    ExpertRegistry,
    HierarchicalExpertRouter,
    ResidencyPlan,
    ServingNode,
)


def _dummy_compressed_pkg() -> object:
    return compress_lora_delta(
        delta_dict={"lora_A": torch.ones((2, 4), dtype=torch.float32)},
        worker_id="w1",
        branch_name="math",
        outer_step=1,
    )


def test_diloco_inner_hardening_guards() -> None:
    pkg = _dummy_compressed_pkg()
    res = InnerLoopResult(
        worker_id="w1",
        branch_name="math",
        outer_step=1,
        steps_completed=2,
        initial_loss=1.2,
        final_loss=1.0,
        elapsed_seconds=0.5,
        theta_init={"lora_A": torch.zeros((2, 4))},
        theta_final={"lora_A": torch.ones((2, 4))},
        raw_delta={"lora_A": torch.ones((2, 4))},
        compressed_delta=pkg,  # type: ignore[arg-type]
        micro_checkpoints={},
        error_feedback_residual_norm=0.25,
    )
    assert res.error_feedback_residual_norm == pytest.approx(0.25)

    with pytest.raises(ValueError, match="error_feedback_residual_norm must be None or a non-negative finite number"):
        InnerLoopResult(
            worker_id="w1",
            branch_name="math",
            outer_step=1,
            steps_completed=1,
            initial_loss=1.0,
            final_loss=0.9,
            elapsed_seconds=0.1,
            theta_init={"lora_A": torch.zeros((2, 4))},
            theta_final={"lora_A": torch.ones((2, 4))},
            raw_delta={"lora_A": torch.ones((2, 4))},
            compressed_delta=pkg,  # type: ignore[arg-type]
            micro_checkpoints={},
            error_feedback_residual_norm=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="circuit_breaker_reason must be None or a non-empty string"):
        InnerLoopResult(
            worker_id="w1",
            branch_name="math",
            outer_step=1,
            steps_completed=0,
            initial_loss=1.0,
            final_loss=1.0,
            elapsed_seconds=0.1,
            theta_init={"lora_A": torch.zeros((2, 4))},
            theta_final={"lora_A": torch.ones((2, 4))},
            raw_delta={"lora_A": torch.ones((2, 4))},
            compressed_delta=pkg,  # type: ignore[arg-type]
            micro_checkpoints={},
            circuit_breaker_reason="   ",
        )

    with pytest.raises(ValueError, match="emergency_dump_path must be None or a non-empty Path/str"):
        InnerLoopResult(
            worker_id="w1",
            branch_name="math",
            outer_step=1,
            steps_completed=0,
            initial_loss=1.0,
            final_loss=1.0,
            elapsed_seconds=0.1,
            theta_init={"lora_A": torch.zeros((2, 4))},
            theta_final={"lora_A": torch.ones((2, 4))},
            raw_delta={"lora_A": torch.ones((2, 4))},
            compressed_delta=pkg,  # type: ignore[arg-type]
            micro_checkpoints={},
            emergency_dump_path="",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="tensor must be a torch.Tensor"):
        quantize_tensor_blockwise_8bit(torch.tensor([True, False]))

    with pytest.raises(ValueError, match="q and absmax must be torch.Tensor instances"):
        dequantize_tensor_blockwise_8bit(torch.tensor([[True, False]]), torch.tensor([1.0]), torch.Size([2]))

    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="block_size must be an integer >= 1"):
        Blockwise8BitAdamW(model.parameters(), block_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="token_sequences must be a non-empty tensor with ndim >= 2"):
        get_deterministic_batch(torch.tensor([[True, False]]), 0, 42, 1, torch.device("cpu"))

    valid_tokens = torch.randint(0, 8, (4, 6), dtype=torch.long)
    embed = nn.Sequential(nn.Embedding(16, 4), nn.Linear(4, 16))
    with pytest.raises(ValueError, match="lr must be finite and > 0"):
        run_diloco_inner_loop(
            embed,
            valid_tokens,
            worker_id="w1",
            branch_name="math",
            outer_step=1,
            inner_steps_H=1,
            lr=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="use_paged_8bit_adamw must be a bool"):
        run_diloco_inner_loop(
            embed,
            valid_tokens,
            worker_id="w1",
            branch_name="math",
            outer_step=1,
            inner_steps_H=1,
            use_paged_8bit_adamw="yes",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="micro_batch_size must be an integer >= 1"):
        replay_inner_segment(
            embed,
            {"0.weight": torch.zeros((16, 4)), "1.weight": torch.zeros((16, 4)), "1.bias": torch.zeros(16)},
            valid_tokens,
            start_step=0,
            end_step=1,
            micro_batch_size=True,  # type: ignore[arg-type]
        )


def test_diloco_outer_hardening_guards() -> None:
    report = ByzantineFilterReport(
        total_submissions=2,
        max_byzantine_f=0,
        accepted_workers=["n1", "n2"],
        rejected_workers=[],
        krum_scores={"n1": 0.1, "n2": 0.2},
        mean_cosine_similarities={"n1": 0.9, "n2": 0.85},
        median_cosine=0.875,
        cosine_threshold_lower=0.5,
    )
    assert len(report.accepted_workers) == 2

    with pytest.raises(ValueError, match="krum_scores values must be finite numbers"):
        ByzantineFilterReport(
            total_submissions=1,
            max_byzantine_f=0,
            accepted_workers=["n1"],
            rejected_workers=[],
            krum_scores={"n1": True},  # type: ignore[dict-item]
            mean_cosine_similarities={"n1": 0.9},
            median_cosine=0.9,
            cosine_threshold_lower=0.5,
        )

    with pytest.raises(ValueError, match="quarantined_pre_krum keys and values must be non-empty strings"):
        ByzantineFilterReport(
            total_submissions=1,
            max_byzantine_f=0,
            accepted_workers=["n1"],
            rejected_workers=[],
            krum_scores={"n1": 0.1},
            mean_cosine_similarities={"n1": 0.9},
            median_cosine=0.9,
            cosine_threshold_lower=0.5,
            quarantined_pre_krum={"n2": "   "},
        )

    with pytest.raises(ValueError, match="cgad_telemetry must be a list"):
        OuterStepResult(
            branch_name="math",
            outer_step=1,
            theta_updated={"w": torch.ones(2)},
            velocity_updated={"w": torch.zeros(2)},
            aggregated_delta_norm=0.1,
            filter_report=report,
            cgad_telemetry="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match=r"delta_dict\['w'\] must be a non-empty torch\.Tensor"):
        flatten_delta_dict({"w": torch.tensor([True, False])})

    with pytest.raises(ValueError, match="cgad_alpha must be a finite non-negative number"):
        DiLoCoOuterOptimizer(cgad_alpha=True)  # type: ignore[arg-type]

    opt = DiLoCoOuterOptimizer()
    valid_theta = {"w": torch.ones((2, 2))}
    valid_deltas = {"w1": {"w": torch.ones((2, 2))}}
    with pytest.raises(ValueError, match="min_quorum_k must be a positive integer or None"):
        opt.step("main", 1, valid_theta, valid_deltas, min_quorum_k=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="quorum_merge_mode must be a non-empty string"):
        opt.step("main", 1, valid_theta, valid_deltas, quorum_merge_mode="   ")


def test_exoplanet_astro_vmoe_hardening_guards() -> None:
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    loss = asymmetric_focal_loss(logits, targets, sample_weights=torch.ones(4))
    assert float(loss.item()) > 0.0

    with pytest.raises(ValueError, match="logits must be a non-empty torch.Tensor"):
        asymmetric_focal_loss(torch.tensor([[True, False]]), torch.tensor([0]))

    with pytest.raises(ValueError, match="sample_weights must be a non-empty finite numeric torch.Tensor"):
        asymmetric_focal_loss(logits, targets, sample_weights=torch.tensor([True, False, True, False]))

    with pytest.raises(ValueError, match="logits must be a non-empty 2D\\+ torch.Tensor"):
        compute_citizen_science_entropy_weights(torch.tensor([[True, False], [False, True]]))

    adapter = SwarmFFNAdapter(hidden_dim=8, intermediate_dim=16, rank=4)
    with pytest.raises(ValueError, match="h must be a non-empty finite numeric torch.Tensor with last dim 8"):
        adapter(torch.randn(2, 6))

    with pytest.raises(ValueError, match="delta must be a finite numeric numpy.ndarray of shape \\(8, 8\\)"):
        adapter.set_effective_delta(np.ones((4, 4), dtype=np.float32))

    backbone = FrozenAstroVisionBackbone(patch_dim=12, hidden_dim=8, intermediate_dim=16, num_classes=4)
    with pytest.raises(ValueError, match="visual_patches must be a non-empty finite numeric torch.Tensor with last dim 12"):
        backbone.encode_patches(torch.randn(2, 4, 10))

    with pytest.raises(ValueError, match="backbone must be a FrozenAstroVisionBackbone"):
        UpcycledAstroSparseVisionMoE("invalid", {"b0": np.zeros((8, 8), dtype=np.float32)})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="branch_deltas must be a non-empty Mapping"):
        UpcycledAstroSparseVisionMoE(backbone, {})

    moe = UpcycledAstroSparseVisionMoE(backbone, {"b0": np.zeros((8, 8), dtype=np.float32)}, top_k=1)
    out = moe(torch.randn(2, 4, 12))
    assert out["logits"].shape == (2, 4)

    with pytest.raises(ValueError, match="visual_patches must be a non-empty 3D finite numeric torch.Tensor"):
        moe(torch.randn(2, 12))


def test_expert_registry_hardening_guards() -> None:
    with pytest.raises(ValueError, match="expert_id, cluster_id and base_model_id must be non-empty"):
        ExpertRecord(123, SERVING, "code", "base-v1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="keywords must be an iterable of strings"):
        ExpertRecord("e1", SERVING, "code", "base-v1", keywords="not-a-tuple")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="serving expert 'e1' has no adapter: adapter_sha256 must be None and size 0"):
        ExpertRecord("e1", SERVING, "code", "base-v1", adapter_size_mib=False)  # type: ignore[arg-type]

    reg = ExpertRegistry()
    sha = "a" * 64
    rec = reg.register(
        ExpertRecord(
            expert_id="exp_math",
            kind=LORA_DOMAIN,
            cluster_id="math",
            base_model_id="base-v1",
            adapter_sha256=sha,
            adapter_size_mib=32,
            keywords=("calculus",),
        )
    )
    assert reg.get("exp_math").expert_id == rec.expert_id

    with pytest.raises(ValueError, match="expert_id must be a non-empty string"):
        reg.get("   ")

    with pytest.raises(ValueError, match="adapter_sha256 must be a non-empty string"):
        reg.find_by_adapter("")

    with pytest.raises(ValueError, match="base_model_id must be a non-empty string"):
        reg.check_attachable("exp_math", "   ")

    with pytest.raises(ValueError, match="base_model_id must be non-empty"):
        AdapterFootprint(size_mib=16, base_model_id=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="node_id and base_model_id must be non-empty"):
        ServingNode(node_id=None, vram_mib=8192, base_model_id="base-v1", base_resident_mib=4096, max_lora_slots=4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="placed_demand must be a finite non-negative number"):
        ResidencyPlan(
            placements={},
            hot={},
            cold={},
            node_free_mib={},
            node_free_slots={},
            blocked_nodes={},
            placed_demand=True,  # type: ignore[arg-type]
            total_demand=1.0,
            ordering="demand",
        )

    reg.transition("exp_math", SHADOW, "validated")
    reg.transition("exp_math", LIVE, "promoted")
    router = HierarchicalExpertRouter(reg, base_model_id="base-v1")
    decision = router.route("solve calculus integral")
    assert decision.expert_id == "exp_math"

    with pytest.raises(ValueError, match="cluster_id must be None or a non-empty string"):
        router.route("solve calculus", cluster_id="   ")

    with pytest.raises(ValueError, match="recruitment must be None or a Mapping"):
        router.route("solve calculus", recruitment=["bad"])  # type: ignore[arg-type]
