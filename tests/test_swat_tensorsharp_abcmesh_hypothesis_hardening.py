# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control tests for SwaT telepathy bridge, TensorSharp collab bridge, ABC MoE agent mesh, and Architecture Hypothesis Verifier."""

from __future__ import annotations

import math
import numpy as np
import pytest

from hivemind.swarm.abc_moe_agent_mesh import (
    ABCMoEAgent,
    AgentMeshOrchestrator,
    FoodSource,
    WaggleDanceMessage,
    WaggleDanceScrutinyProtocol,
)
from hivemind.swarm.architecture_hypothesis_verifier import (
    ArchitectureHypothesisVerifier,
    TinyDeterministicLoRALM,
    TinyLoRALinear,
    _build_tiny_qwen_and_branches,
)
from hivemind.swarm.swat_telepathy_bridge import (
    ModelKVSpec,
    SwaTTelepathyBridge,
    compact_kv_with_attention_matching,
    compute_cms_chunk_periods,
    compute_layer_group_slices,
    select_rem_dreams_by_gradient_norm,
)
from hivemind.swarm.tensorsharp_collab_bridge import (
    TensorSharpBeeLoRARouter,
    TensorSharpDistributedClusterSpec,
    TensorSharpSingleModelProducerSpec,
    plan_tensorsharp_moe_offload,
)


class TestSwaTTelepathyBridgeHardening:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"model_id": "", "family": "qwen", "num_layers": 24, "num_kv_heads": 2, "head_dim": 64, "hidden_size": 1024}, "model_id"),
            ({"model_id": "m1", "family": "  ", "num_layers": 24, "num_kv_heads": 2, "head_dim": 64, "hidden_size": 1024}, "family"),
            ({"model_id": "m1", "family": "qwen", "num_layers": True, "num_kv_heads": 2, "head_dim": 64, "hidden_size": 1024}, "num_layers"),
            ({"model_id": "m1", "family": "qwen", "num_layers": 24, "num_kv_heads": 2, "head_dim": 64, "hidden_size": 1024, "rope_theta": 0.0}, "rope_theta"),
            ({"model_id": "m1", "family": "qwen", "num_layers": 24, "num_kv_heads": 2, "head_dim": 64, "hidden_size": 1024, "is_vlm": 1}, "is_vlm"),
        ],
    )
    def test_model_kv_spec_rejects_invalid_fields(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ModelKVSpec(**kwargs)

    def test_kv_cache_mib_and_layer_group_slices_validation(self) -> None:
        spec = ModelKVSpec("m1", "qwen", 24, 2, 64, 1024)
        with pytest.raises(ValueError, match="seq_len"):
            spec.kv_cache_mib(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bytes_per_elem"):
            spec.kv_cache_mib(128, bytes_per_elem=0)

        with pytest.raises(ValueError, match="n_blocks"):
            compute_layer_group_slices(24, n_blocks=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="total_layers"):
            compute_layer_group_slices(True, n_blocks=1)  # type: ignore[arg-type]

    def test_plan_transfer_cms_rem_and_am_compaction_validation(self) -> None:
        bridge = SwaTTelepathyBridge()
        with pytest.raises(ValueError, match="am_compaction_ratio"):
            bridge.plan_transfer(
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-4B",
                [(0, 4)],
                [(0, 4)],
                am_compaction_ratio=True,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="rtx2070_used_vram_mib"):
            bridge.plan_transfer(
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-4B",
                [(0, 4)],
                [(0, 4)],
                rtx2070_used_vram_mib=-1.0,
            )
        with pytest.raises(ValueError, match="ocr_text_density"):
            bridge.plan_transfer(
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-4B",
                [(0, 4)],
                [(0, 4)],
                ocr_text_density=1.5,
            )

        with pytest.raises(ValueError, match="max_chunk_tokens"):
            compute_cms_chunk_periods([10.0, 2.0], max_chunk_tokens=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="CMS frequency"):
            compute_cms_chunk_periods([float("nan")])
        with pytest.raises(ValueError, match="CMS frequency"):
            compute_cms_chunk_periods([True])  # type: ignore[list-item]

        with pytest.raises(ValueError, match="top_k"):
            select_rem_dreams_by_gradient_norm(["d1"], [1.0], [0], top_k=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed"):
            select_rem_dreams_by_gradient_norm(["d1"], [1.0], [0], seed=-1)
        with pytest.raises(ValueError, match="sft_gradient_norms"):
            select_rem_dreams_by_gradient_norm(["d1"], [float("nan")], [0])

        kv = np.ones((1, 8, 2, 2, 2, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="target_ratio"):
            compact_kv_with_attention_matching(kv, target_ratio=float("nan"))
        with pytest.raises(ValueError, match="target_ratio"):
            compact_kv_with_attention_matching(kv, target_ratio=True)  # type: ignore[arg-type]
        bad_kv = kv.copy()
        bad_kv[0, 0, 0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="pred_kv"):
            compact_kv_with_attention_matching(bad_kv)


class TestTensorSharpCollabBridgeHardening:
    def test_plan_tensorsharp_moe_offload_rejects_invalid_args(self) -> None:
        with pytest.raises(ValueError, match="total_vram_mib"):
            plan_tensorsharp_moe_offload("gpt-oss-20b", total_vram_mib=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reserved_free_vram_mib"):
            plan_tensorsharp_moe_offload("gpt-oss-20b", reserved_free_vram_mib=-1)
        with pytest.raises(ValueError, match="gpu_name"):
            plan_tensorsharp_moe_offload("gpt-oss-20b", gpu_name="   ")
        with pytest.raises(ValueError, match="compute_capability"):
            plan_tensorsharp_moe_offload("gpt-oss-20b", compute_capability="")

    def test_single_model_distributed_and_lora_router_validation(self) -> None:
        spec = TensorSharpSingleModelProducerSpec(
            model_path="model.gguf",
            v1_models_id="Qwen3.5-9B-Q4_K_M",
            public_tier_id="hivemind/qwen35-core",
            public_https_shield_url="https://shield.example.com",
            endpoint_token="a" * 16,
            hivemind_api_key="hm_key_123",
            server_port=70000,
        )
        with pytest.raises(ValueError, match="server_port"):
            spec.validate()

        dist = TensorSharpDistributedClusterSpec(
            model_path="model.gguf",
            mode="tp",
            local_degree=True,  # type: ignore[arg-type]
            node_id=0,
            peer_endpoints=["127.0.0.1:9000"],
        )
        with pytest.raises(ValueError, match="local_degree"):
            dist.validate()

        router = TensorSharpBeeLoRARouter()
        with pytest.raises(ValueError, match="messages"):
            router.build_openai_chat_request(public_model_id="hivemind/bee-code", messages=[])
        with pytest.raises(ValueError, match="max_tokens"):
            router.build_openai_chat_request(
                public_model_id="hivemind/bee-code",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=True,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="temperature"):
            router.build_openai_chat_request(
                public_model_id="hivemind/bee-code",
                messages=[{"role": "user", "content": "hi"}],
                temperature=-0.1,
            )
        with pytest.raises(ValueError, match="gpu_vram_mib"):
            router.build_openrouter_heartbeat(
                public_endpoint_url="https://shield.example.com",
                gpu_vram_mib=0,
            )


class TestABCMoEAgentMeshHardening:
    def test_waggle_dance_food_source_and_orchestrator_validation(self) -> None:
        msg = WaggleDanceMessage(
            task_id="t1",
            scout_bee_id="b1",
            task_domain="sci",
            confidence_score=0.8,
            dance_duration_ms=800,
            estimated_tokens_cost=100,
        )
        with pytest.raises(ValueError, match="secret_key"):
            msg.sign("   ")
        assert not msg.verify_signature("   ")

        with pytest.raises(ValueError, match="nectar_richness"):
            FoodSource("s1", "initial_prompt", True, {})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="trial_count"):
            FoodSource("s1", "initial_prompt", 0.5, {}, trial_count=-1)

        with pytest.raises(ValueError, match="tau_0_ms"):
            WaggleDanceScrutinyProtocol(tau_0_ms=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="secret_key"):
            WaggleDanceScrutinyProtocol(secret_key="   ")

        proto = WaggleDanceScrutinyProtocol()
        with pytest.raises(ValueError, match="confidence_score"):
            proto.create_waggle_dance("t1", "b1", "sci", True, 100, [], {})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="quorum_threshold"):
            proto.create_waggle_dance("t1", "b1", "sci", 0.8, 100, [], {}, quorum_threshold=0)

        with pytest.raises(ValueError, match="abandonment_limit"):
            ABCMoEAgent(abandonment_limit=0)
        with pytest.raises(ValueError, match="quality_threshold"):
            ABCMoEAgent(quality_threshold=0.0)

        orch = AgentMeshOrchestrator()
        with pytest.raises(ValueError, match="task_id"):
            orch.execute_mesh_workflow("  ", "query", {"a": "str"})
        with pytest.raises(ValueError, match="confidence_prior"):
            orch.execute_mesh_workflow("t1", "query", {"a": "str"}, confidence_prior=1.5)


class TestArchitectureHypothesisVerifierHardening:
    def test_tiny_lora_and_verifier_reject_invalid_parameters(self) -> None:
        with pytest.raises(ValueError, match="in_features"):
            TinyLoRALinear(0, 16)
        with pytest.raises(ValueError, match="rank_r"):
            TinyLoRALinear(16, 16, rank_r=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="vocab_size"):
            TinyDeterministicLoRALM(vocab_size=1)
        with pytest.raises(ValueError, match="num_branches"):
            _build_tiny_qwen_and_branches(num_branches=5)
        with pytest.raises(ValueError, match="hidden_size"):
            _build_tiny_qwen_and_branches(hidden_size=30)
        with pytest.raises(ValueError, match="seed"):
            ArchitectureHypothesisVerifier(seed=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed"):
            ArchitectureHypothesisVerifier(seed=-1)
