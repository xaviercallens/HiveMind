# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Strict boundary and negative-control tests for Iteration 142 swarm hardening."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.swarm.abc_moe_agent_mesh import ABCMoEAgent, ABCState, FoodSource
from hivemind.swarm.architecture_hypothesis_verifier import (
    ArchitectureHypothesisVerifier,
    HypothesisVerificationRecord,
    TinyDeterministicLoRALM,
    TinyLoRALinear,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.scientific_swarm_eval import (
    JoinedSwarmCampaignEngine,
    PeerKnowledgeTransferAudit,
    SwarmKnowledgeTransferEngine,
)
from hivemind.swarm.tensorsharp_collab_bridge import (
    TensorSharpBeeLoRARouter,
    TensorSharpBenchmarkProfile,
    TensorSharpDistributedClusterSpec,
    TensorSharpT4MeasuredRun,
)


# ---------------------------------------------------------------------------
# 1. scientific_swarm_eval.py
# ---------------------------------------------------------------------------


def _valid_audit_kwargs() -> dict:
    return {
        "domain_name": "Rust_Systems_Security",
        "benchmark_id": "HF-SCI-BENCH-01-RUST-BORROW",
        "trainer_agent_id": "CyberVal_FR",
        "expert_cid": "bafkrei_test_cid",
        "trainer_initial_loss": 1.2,
        "trainer_final_loss": 0.4,
        "trainer_loss_reduction_pct": 66.67,
        "peer_evaluations": [],
        "mean_peer_pre_transfer_loss": None,
        "mean_peer_post_transfer_loss": None,
        "mean_peer_pre_perplexity": None,
        "mean_peer_post_perplexity": None,
        "mean_peer_loss_reduction_pct": None,
        "knowledge_transfer_efficiency_ratio": None,
        "mean_hotswap_latency_ms": None,
        "total_peer_gpu_steps_saved": None,
        "royalties_paid_to_trainer_hive": 7.5,
        "all_peers_leveraged_successfully": True,
    }


def test_peer_knowledge_transfer_audit_valid_and_invalid() -> None:
    audit = PeerKnowledgeTransferAudit(**_valid_audit_kwargs())
    assert audit.domain_name == "Rust_Systems_Security"

    for field_name in ("domain_name", "benchmark_id", "trainer_agent_id", "expert_cid"):
        bad = _valid_audit_kwargs()
        bad[field_name] = "   "
        with pytest.raises(ValueError, match=field_name):
            PeerKnowledgeTransferAudit(**bad)

    bad_evals = _valid_audit_kwargs()
    bad_evals["peer_evaluations"] = "not-a-list"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="peer_evaluations"):
        PeerKnowledgeTransferAudit(**bad_evals)

    for bad_royalty in (-1.0, float("nan"), float("inf"), True):
        bad_r = _valid_audit_kwargs()
        bad_r["royalties_paid_to_trainer_hive"] = bad_royalty
        with pytest.raises(ValueError, match="royalties_paid_to_trainer_hive"):
            PeerKnowledgeTransferAudit(**bad_r)

    bad_bool = _valid_audit_kwargs()
    bad_bool["all_peers_leveraged_successfully"] = 1  # type: ignore[assignment]
    with pytest.raises(ValueError, match="all_peers_leveraged_successfully"):
        PeerKnowledgeTransferAudit(**bad_bool)


def test_swarm_knowledge_transfer_and_campaign_engine_guards(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(tmp_path / "cas")
    engine = SwarmKnowledgeTransferEngine(store=store)
    ref_lora = {
        "layers.0.lora_A": torch.ones((4, 8)),
        "layers.0.lora_B": torch.ones((8, 4)),
    }
    with pytest.raises(ValueError, match="trainer_agent_id"):
        engine.verify_swarm_peer_knowledge_transfer(
            domain_name="Rust_Systems_Security",
            trainer_agent_id="   ",
            reference_lora_state=ref_lora,
            inner_steps=2,
        )

    campaign = JoinedSwarmCampaignEngine(transfer_engine=engine)
    with pytest.raises(ValueError, match="moe_model"):
        campaign.execute_full_scientific_swarm_campaign(
            reference_lora_state=ref_lora,
            moe_model=object(),  # type: ignore[arg-type]
            inner_steps=2,
        )


# ---------------------------------------------------------------------------
# 2. architecture_hypothesis_verifier.py
# ---------------------------------------------------------------------------


def test_hypothesis_verification_record_guards() -> None:
    rec = HypothesisVerificationRecord(
        hypothesis_id="H1",
        title="Flat Bucket",
        category="DiLoCo",
        arxiv_references=["2407.07852"],
        github_references=["PrimeIntellect-ai/prime-diloco"],
        lean4_theorems=["h1_thm"],
        passed=True,
        key_metrics={"vram": 0},
    )
    assert rec.passed is True

    for field_name in ("hypothesis_id", "title", "category"):
        kwargs = {
            "hypothesis_id": "H1",
            "title": "Flat Bucket",
            "category": "DiLoCo",
            "arxiv_references": [],
            "github_references": [],
            "lean4_theorems": [],
            "passed": True,
            "key_metrics": {},
        }
        kwargs[field_name] = ""
        with pytest.raises(ValueError, match=field_name):
            HypothesisVerificationRecord(**kwargs)

    with pytest.raises(ValueError, match="passed"):
        HypothesisVerificationRecord(
            hypothesis_id="H1",
            title="Flat Bucket",
            category="DiLoCo",
            arxiv_references=[],
            github_references=[],
            lean4_theorems=[],
            passed=1,  # type: ignore[arg-type]
            key_metrics={},
        )

    with pytest.raises(ValueError, match="key_metrics"):
        HypothesisVerificationRecord(
            hypothesis_id="H1",
            title="Flat Bucket",
            category="DiLoCo",
            arxiv_references=[],
            github_references=[],
            lean4_theorems=[],
            passed=True,
            key_metrics=[],  # type: ignore[arg-type]
        )


def test_tiny_lora_linear_and_lm_forward_guards() -> None:
    layer = TinyLoRALinear(in_features=8, out_features=4, rank_r=2)
    out = layer(torch.ones((2, 8)))
    assert out.shape == (2, 4)

    with pytest.raises(ValueError, match="non-empty torch.Tensor"):
        layer("not-a-tensor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty torch.Tensor"):
        layer(torch.empty((0, 8)))
    with pytest.raises(ValueError, match="in_features"):
        layer(torch.ones((2, 6)))

    lm = TinyDeterministicLoRALM(vocab_size=32, hidden_size=8, rank_r=2)
    ids = torch.randint(0, 32, (2, 5))
    res = lm(ids, labels=ids)
    assert res.loss is not None

    with pytest.raises(ValueError, match="input_ids"):
        lm(torch.ones((5,), dtype=torch.long))
    with pytest.raises(ValueError, match="labels"):
        lm(ids, labels=torch.randint(0, 32, (2, 4)))


def test_verifier_h5_and_run_all_verifications_guards() -> None:
    verifier = ArchitectureHypothesisVerifier(seed=2026)
    for bad_ms in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="simulated_attention_compute_ms"):
            verifier.verify_h5_skip_attention_car_and_async_dma(
                simulated_attention_compute_ms=bad_ms
            )
        with pytest.raises(ValueError, match="min_transfer_latency_ms"):
            verifier.verify_h5_skip_attention_car_and_async_dma(
                min_transfer_latency_ms=bad_ms
            )

    with pytest.raises(ValueError, match="output_path"):
        verifier.run_all_verifications(output_path="   ")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. tensorsharp_collab_bridge.py
# ---------------------------------------------------------------------------


def _valid_benchmark_profile_kwargs() -> dict:
    return {
        "model_id": "gemma-4-26b-a4b",
        "v1_models_id": "gemma-4-26B-A4B-it-UD-IQ4_XS",
        "hivemind_tier_id": "hivemind/gemma-4-26b-a4b-moe",
        "total_params_b": 26.0,
        "active_params_b": 4.0,
        "total_moe_layers": 30,
        "resident_vram_mib": 14650,
        "min_offload_vram_mib": 1732,
        "n_cpu_moe_recommended": 15,
        "tensorsharp_pp8192_tok_s": 544.0,
        "llamacpp_pp8192_tok_s": 135.0,
        "tensorsharp_tg128_tok_s": 40.0,
        "llamacpp_tg128_tok_s": 31.0,
        "gguf_artifact": "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    }


def test_tensorsharp_benchmark_profile_and_t4_run_guards() -> None:
    prof = TensorSharpBenchmarkProfile(**_valid_benchmark_profile_kwargs())
    assert prof.prefill_speedup_vs_llamacpp > 1.0

    for str_field in ("model_id", "v1_models_id", "hivemind_tier_id", "gguf_artifact"):
        bad = _valid_benchmark_profile_kwargs()
        bad[str_field] = ""
        with pytest.raises(ValueError, match=str_field):
            TensorSharpBenchmarkProfile(**bad)

    for num_field in ("total_params_b", "active_params_b"):
        bad = _valid_benchmark_profile_kwargs()
        bad[num_field] = 0.0
        with pytest.raises(ValueError, match=num_field):
            TensorSharpBenchmarkProfile(**bad)

    for int_field in ("total_moe_layers", "resident_vram_mib"):
        bad = _valid_benchmark_profile_kwargs()
        bad[int_field] = 0
        with pytest.raises(ValueError, match=int_field):
            TensorSharpBenchmarkProfile(**bad)

    valid_t4 = {
        "run_key": "qwen3.5-9b-q4_k_m-resident",
        "v1_models_id": "Qwen3.5-9B-Q4_K_M",
        "gguf_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "gguf_size_bytes": 5680522464,
        "gguf_sha256": "a" * 64,
        "gguf_source": "unsloth/Qwen3.5-9B-GGUF",
        "backend": "ggml_cuda",
        "n_cpu_moe": None,
        "max_context": 4096,
        "idle_vram_mib": 5420,
        "peak_vram_mib": 6062,
        "prompt_tokens": 128,
        "completion_tokens": 64,
        "sse_chunks": 64,
        "ttft_ms": 120.0,
        "measured_prefill_tok_s": 310.0,
        "measured_stream_decode_tok_s": 42.0,
        "measured_non_stream_e2e_tok_s": 40.0,
    }
    run = TensorSharpT4MeasuredRun(**valid_t4)
    assert run.peak_vram_mib >= run.idle_vram_mib

    bad_sha = dict(valid_t4, gguf_sha256="short")
    with pytest.raises(ValueError, match="gguf_sha256"):
        TensorSharpT4MeasuredRun(**bad_sha)

    bad_peak = dict(valid_t4, peak_vram_mib=5000)
    with pytest.raises(ValueError, match="peak_vram_mib"):
        TensorSharpT4MeasuredRun(**bad_peak)


def test_tensorsharp_cluster_spec_and_lora_router_guards() -> None:
    spec = TensorSharpDistributedClusterSpec(
        model_path="/models/test.gguf",
        mode="tp",
        local_degree=1,
        node_id=0,
        peer_endpoints=["127.0.0.1:9000"],
    )
    spec.validate()

    with pytest.raises(ValueError, match="mode"):
        TensorSharpDistributedClusterSpec(
            model_path="/models/test.gguf",
            mode="invalid",  # type: ignore[arg-type]
            local_degree=1,
            node_id=0,
            peer_endpoints=["127.0.0.1:9000"],
        ).validate()

    with pytest.raises(ValueError, match="all_reduce"):
        TensorSharpDistributedClusterSpec(
            model_path="/models/test.gguf",
            mode="tp",
            local_degree=1,
            node_id=0,
            peer_endpoints=["127.0.0.1:9000"],
            all_reduce="ring",  # type: ignore[arg-type]
        ).validate()

    with pytest.raises(ValueError, match="n_cpu_moe"):
        TensorSharpDistributedClusterSpec(
            model_path="/models/test.gguf",
            mode="tp",
            local_degree=1,
            node_id=0,
            peer_endpoints=["127.0.0.1:9000"],
            n_cpu_moe=-1,
        ).validate()

    with pytest.raises(ValueError, match="loras_dir"):
        TensorSharpBeeLoRARouter(loras_dir="   ")

    with pytest.raises(ValueError, match="registered_bee_adapters"):
        TensorSharpBeeLoRARouter(registered_bee_adapters=["not-a-dict"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. abc_moe_agent_mesh.py
# ---------------------------------------------------------------------------


def test_abc_state_and_agent_step_guards() -> None:
    state = ABCState(
        task_id="task-01",
        user_query="battery electrolyte stability",
        target_schema={"formulation": "str"},
    )
    assert state.quality_threshold == 0.75

    with pytest.raises(ValueError, match="task_id"):
        ABCState(task_id="", user_query="valid")
    with pytest.raises(ValueError, match="user_query"):
        ABCState(task_id="task-01", user_query="   ")
    with pytest.raises(ValueError, match="target_schema"):
        ABCState(task_id="task-01", user_query="valid", target_schema=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quality_threshold"):
        ABCState(task_id="task-01", user_query="valid", quality_threshold=0.0)
    with pytest.raises(ValueError, match="abandonment_limit"):
        ABCState(task_id="task-01", user_query="valid", abandonment_limit=0)
    with pytest.raises(ValueError, match="max_cycles"):
        ABCState(task_id="task-01", user_query="valid", max_cycles=0)
    with pytest.raises(ValueError, match="cycle_count"):
        ABCState(task_id="task-01", user_query="valid", cycle_count=-1)

    agent = ABCMoEAgent(agent_id="bee-01")
    with pytest.raises(ValueError, match="query"):
        agent._default_mock_web_search("   ")
    with pytest.raises(ValueError, match="state"):
        agent.employed_exploit_step("not-a-state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state"):
        agent.observer_evaluate_step("not-a-state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="food_sources"):
        agent.observer_evaluate_step(state)
    with pytest.raises(ValueError, match="state"):
        agent.scout_explore_step("not-a-state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_state"):
        agent.run("not-a-state")  # type: ignore[arg-type]

    final_state = agent.run(state)
    assert final_state.current_phase == "COMPLETED"
