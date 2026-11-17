# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 195 negative-control and invariant hardening tests across:
- `hivemind.core.expert_registry`
- `hivemind.core.serving_experts`
- `hivemind.core.base_model_catalog`
- `hivemind.core.fomoe_sigma_routing`
"""

from __future__ import annotations

import pytest
import torch

from hivemind.core.base_model_catalog import (
    FitVerdict,
    GpuTier,
    ModelGeometry,
    PromotionReceipt,
    VramEstimate,
)
from hivemind.core.expert_registry import (
    SERVING,
    AdapterFootprint,
    ExpertRecord,
    ExpertRegistry,
    HierarchicalExpertRouter,
    ResidencyPlan,
    RouteDecision,
    ServingNode,
)
from hivemind.core.fomoe_sigma_routing import (
    SigmaMoERoutingResult,
    sigmoid_independent_topk_routing,
)
from hivemind.core.serving_experts import load_gwaya_code_expert


def test_expert_registry_ip195_hardening() -> None:
    with pytest.raises(ValueError, match="expert_id, cluster_id and base_model_id must be non-empty"):
        ExpertRecord("exp\n1", SERVING, "code", "Qwen/Qwen3-4B")

    with pytest.raises(ValueError, match="source must be a string without control characters"):
        ExpertRecord("exp1", SERVING, "code", "Qwen/Qwen3-4B", source="bad\x00source")

    with pytest.raises(ValueError, match="base_model_id must be non-empty"):
        AdapterFootprint(size_mib=16, base_model_id="Qwen/Qwen3-4B\n")

    with pytest.raises(ValueError, match="node_id and base_model_id must be non-empty"):
        ServingNode("node\x7f1", 8192, "Qwen/Qwen3-4B", 4096, 4)

    with pytest.raises(ValueError, match="placed_demand cannot exceed total_demand"):
        ResidencyPlan(
            placements={},
            hot={},
            cold={},
            node_free_mib={},
            node_free_slots={},
            blocked_nodes={},
            placed_demand=5.0,
            total_demand=2.0,
            ordering="demand",
        )

    with pytest.raises(ValueError, match="load_on_demand cannot be True when target is 'base'"):
        RouteDecision(target="base", expert_id=None, cluster_id=None, reason="fallback", load_on_demand=True)

    with pytest.raises(ValueError, match="expert_id must be a non-empty string when target is 'expert'"):
        RouteDecision(target="expert", expert_id="bee\n1", cluster_id="code", reason="ok")

    reg = ExpertRegistry()
    with pytest.raises(ValueError, match="resident must be None or a Collection of expert IDs"):
        HierarchicalExpertRouter(reg, "Qwen/Qwen3-4B", resident="not-a-collection")  # type: ignore[arg-type]


def test_serving_experts_ip195_del_control_character_guards() -> None:
    with pytest.raises(ValueError, match="large_url must be a non-empty http\\(s\\) URL string"):
        load_gwaya_code_expert("http://localhost:8080\x7f")

    with pytest.raises(ValueError, match="large_model must be a non-empty string"):
        load_gwaya_code_expert("http://localhost:8080", large_model="qwen\x7f")

    with pytest.raises(ValueError, match="small_url must be a non-empty http\\(s\\) URL string"):
        load_gwaya_code_expert("http://localhost:8080", small_url="http://localhost:8081\x7f")

    with pytest.raises(ValueError, match="small_model must be a non-empty string"):
        load_gwaya_code_expert("http://localhost:8080", small_model="qwen-small\x7f")


def test_base_model_catalog_ip195_hardening() -> None:
    base_geom_kw = dict(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_text_linear=7_000_000_000,
        params_embedding=500_000_000,
        params_text_other=100_000_000,
        params_vision=0,
        params_mtp=0,
        lora_in_plus_out_all_linear=1_000_000,
        lora_in_plus_out_mlp=600_000,
        max_linear_elements=50_000_000,
        hidden=3584,
        layers=28,
        intermediate=18944,
        vocab=152064,
        n_kv_heads=4,
        head_dim=128,
        n_full_attention_layers=28,
        n_linear_attention_layers=0,
        linear_num_key_heads=0,
        linear_num_value_heads=0,
        linear_key_head_dim=0,
        linear_value_head_dim=0,
        linear_conv_kernel_dim=0,
    )
    geom = ModelGeometry(**base_geom_kw)
    assert geom.layers == 28

    with pytest.raises(ValueError, match="n_full_attention_layers \\+ n_linear_attention_layers must equal layers"):
        ModelGeometry(**{**base_geom_kw, "n_full_attention_layers": 20, "n_linear_attention_layers": 4})

    with pytest.raises(ValueError, match="lora_in_plus_out_mlp cannot exceed lora_in_plus_out_all_linear"):
        ModelGeometry(**{**base_geom_kw, "lora_in_plus_out_mlp": 2_000_000})

    with pytest.raises(ValueError, match="components_mib must map non-empty component names to finite non-negative"):
        VramEstimate("m", "qlora_nf4_training", "c", {"nf4": -1.0}, 100.0)

    with pytest.raises(ValueError, match="components_mib must map non-empty component names to finite non-negative"):
        VramEstimate("m", "qlora_nf4_training", "c", {"nf4": True}, 100.0)  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="fits must equal \\(verdict == 'FITS'\\)"):
        FitVerdict("FITS", False, 3500.0, 4200.0, 6000.0, 0.2)

    with pytest.raises(ValueError, match="required_with_margin_mib must be >= estimate_mib"):
        FitVerdict("FITS", True, 4200.0, 3500.0, 6000.0, 0.2)

    with pytest.raises(ValueError, match="wins \\+ losses cannot exceed n_items"):
        PromotionReceipt("HOLD", [], 10, 6, 6, 0.5, 5, 0, 0, 1.0)

    with pytest.raises(ValueError, match="guard_wins \\+ guard_losses cannot exceed guard_items"):
        PromotionReceipt("HOLD", [], 10, 2, 2, 0.5, 5, 3, 3, 1.0)

    with pytest.raises(ValueError, match="tier_id must be a non-empty string"):
        GpuTier("tier\n1", 4096, "src")


def test_fomoe_sigma_routing_ip195_hardening() -> None:
    logits = torch.tensor([[0.5, -0.2, 1.1, 0.0], [-0.4, 0.9, 0.2, -1.2]])
    res = sigmoid_independent_topk_routing(logits, top_k=2)

    with pytest.raises(ValueError, match="expert_indices second dimension must equal top_k"):
        SigmaMoERoutingResult(
            expert_indices=torch.tensor([[0], [1]]),
            gate_values=torch.tensor([[0.5], [0.5]]),
            skipped_slots=torch.tensor([[False], [False]]),
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.0,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=1.0,
        )

    with pytest.raises(ValueError, match="expert_indices entries must be in \\[0, total_experts\\)"):
        SigmaMoERoutingResult(
            expert_indices=torch.tensor([[0, 4], [1, 2]]),
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.0,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=1.0,
        )

    with pytest.raises(ValueError, match="empirical_skip_token_scaling must equal 1 - ghost_assignment_fraction"):
        SigmaMoERoutingResult(
            expert_indices=res.expert_indices,
            gate_values=res.gate_values,
            skipped_slots=res.skipped_slots,
            total_experts=4,
            resident_experts=4,
            top_k=2,
            ghost_assignment_fraction=0.25,
            ideal_skip_token_scaling=1.0,
            empirical_skip_token_scaling=0.25,
        )
