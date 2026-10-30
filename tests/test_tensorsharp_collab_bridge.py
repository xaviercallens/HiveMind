# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the TensorSharp x HiveMind Co-Lab Bridge (`hivemind.swarm.tensorsharp_collab_bridge`)."""

from __future__ import annotations

import pytest

from hivemind.swarm.tensorsharp_collab_bridge import (
    BLACKWELL_REFERENCE_HARDWARE,
    RTX2070_TOTAL_VRAM_MIB,
    SUPPORTED_TENSORSHARP_TEXT_LORA_ARCHITECTURES,
    T4_REQUIRED_FREE_VRAM_MIB,
    T4_TOTAL_VRAM_MIB,
    TENSORSHARP_BENCHMARK_CATALOG,
    TENSORSHARP_T4_MEASURED_VALIDATION,
    TENSORSHARP_VALIDATED_COMMIT,
    TensorSharpBeeLoRARouter,
    TensorSharpDistributedClusterSpec,
    TensorSharpSingleModelProducerSpec,
    build_zhongkaifu_collab_summary,
    plan_tensorsharp_moe_offload,
)


def test_tensorsharp_reference_benchmarks_and_quantizations() -> None:
    muse = TENSORSHARP_BENCHMARK_CATALOG["muse-spark-816b"]
    assert muse.total_params_b == 816.0
    assert muse.active_params_b == 42.0
    assert muse.quantization == "UD-IQ3_XXS"
    assert muse.prefill_speedup_vs_llamacpp == pytest.approx(1.63, abs=0.02)
    assert muse.decode_speedup_vs_llamacpp == pytest.approx(1.40, abs=0.02)

    qwen35 = TENSORSHARP_BENCHMARK_CATALOG["qwen3.5-35b-a3b"]
    assert qwen35.quantization == "UD-IQ4_XS"
    assert qwen35.v1_models_id == "Qwen3.5-35B-A3B-UD-IQ4_XS"
    assert qwen35.reference_hardware == BLACKWELL_REFERENCE_HARDWARE
    assert qwen35.prefill_speedup_vs_llamacpp > 10.5
    assert qwen35.decode_speedup_vs_llamacpp > 3.2

    gemma4 = TENSORSHARP_BENCHMARK_CATALOG["gemma4-26b-a4b"]
    assert gemma4.quantization == "UD-IQ4_XS"
    assert gemma4.v1_models_id == "gemma-4-26B-A4B-it-UD-IQ4_XS"
    assert gemma4.reference_hardware == BLACKWELL_REFERENCE_HARDWARE

    gpt_oss = TENSORSHARP_BENCHMARK_CATALOG["gpt-oss-20b"]
    assert gpt_oss.quantization == "Q8_0 / MXFP4"
    assert gpt_oss.v1_models_id == "gpt-oss-20b-MXFP4"


def test_tesla_t4_measured_validation_and_interpolated_estimates_separated() -> None:
    # Verify real T4 measurements are recorded with exact commit, SHA-256, and peak VRAM
    qwen9b_t4 = TENSORSHARP_T4_MEASURED_VALIDATION["qwen3.5-9b-q4_k_m-resident"]
    assert qwen9b_t4.commit == TENSORSHARP_VALIDATED_COMMIT
    assert qwen9b_t4.v1_models_id == "Qwen3.5-9B-Q4_K_M"
    assert qwen9b_t4.peak_vram_mib == 6_062
    assert qwen9b_t4.measured_prefill_tok_s == pytest.approx(952.84, abs=0.1)
    assert qwen9b_t4.measured_stream_decode_tok_s == pytest.approx(32.75, abs=0.1)
    assert qwen9b_t4.auth_checks_passed is True

    gpt20b_18 = TENSORSHARP_T4_MEASURED_VALIDATION["gpt-oss-20b-mxfp4-ncpumoe18"]
    assert gpt20b_18.v1_models_id == "gpt-oss-20b-MXFP4"
    assert gpt20b_18.n_cpu_moe == 18
    assert gpt20b_18.peak_vram_mib == 7_042
    assert gpt20b_18.measured_stream_decode_tok_s == pytest.approx(13.04, abs=0.1)

    gpt20b_24 = TENSORSHARP_T4_MEASURED_VALIDATION["gpt-oss-20b-mxfp4-ncpumoe24"]
    assert gpt20b_24.n_cpu_moe == 24
    assert gpt20b_24.peak_vram_mib == 4_610
    assert gpt20b_24.measured_stream_decode_tok_s == pytest.approx(10.76, abs=0.1)

    # Verify interpolated sizing plans are explicitly marked as estimates
    plan_gemma = plan_tensorsharp_moe_offload(
        "gemma4-26b-a4b",
        total_vram_mib=T4_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=T4_REQUIRED_FREE_VRAM_MIB,
    )
    assert plan_gemma.is_interpolated_estimate is True
    assert plan_gemma.reference_hardware == BLACKWELL_REFERENCE_HARDWARE
    assert plan_gemma.reference_quantization == "UD-IQ4_XS"
    assert "ESTIMATE ONLY" in plan_gemma.estimation_note
    assert plan_gemma.usable_vram_mib == 11_060
    assert plan_gemma.fits_vram_budget is True
    assert "--host" in plan_gemma.cli_flags and "127.0.0.1" in plan_gemma.cli_flags

    plan_rtx = plan_tensorsharp_moe_offload(
        "gpt-oss-20b",
        total_vram_mib=RTX2070_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=1_024,
    )
    assert plan_rtx.is_interpolated_estimate is True
    assert plan_rtx.usable_vram_mib == 7_168
    assert plan_rtx.fits_vram_budget is True
    assert 1 <= plan_rtx.n_cpu_moe_layers <= 24


def test_single_model_producer_security_and_v1_models_id_consistency() -> None:
    spec = TensorSharpSingleModelProducerSpec(
        model_path="/models/Qwen3.5-9B-Q4_K_M.gguf",
        v1_models_id="Qwen3.5-9B-Q4_K_M",
        public_tier_id="hivemind/qwen3.5-9b",
        public_https_shield_url="https://t4-shield.example.com",
        endpoint_token="hm_shield_secret_token_1234567890",
        hivemind_api_key="hm_live_producer_key_abcdef",
    )
    srv_cmd = spec.build_server_command()
    shield_cmd = spec.build_shield_command()
    agent_cmd = spec.build_agent_command()
    creds_env = spec.build_credentials_env()

    assert "--host" in srv_cmd and "127.0.0.1" in srv_cmd
    assert "--no-webui" in srv_cmd
    assert "--host" in shield_cmd and "--port" in shield_cmd
    assert "--listen" not in shield_cmd
    assert "Qwen3.5-9B-Q4_K_M" in shield_cmd
    assert "hivemind/qwen3.5-9b=Qwen3.5-9B-Q4_K_M" in agent_cmd
    assert creds_env["HIVEMIND_ENDPOINT_TOKEN"] == "hm_shield_secret_token_1234567890"
    assert creds_env["HIVEMIND_API_KEY"] == "hm_live_producer_key_abcdef"

    # Reject non-loopback TensorSharp binding
    with pytest.raises(ValueError, match="loopback"):
        TensorSharpSingleModelProducerSpec(
            model_path="/models/Qwen3.5-9B-Q4_K_M.gguf",
            v1_models_id="Qwen3.5-9B-Q4_K_M",
            public_tier_id="hivemind/qwen3.5-9b",
            public_https_shield_url="https://t4-shield.example.com",
            endpoint_token="hm_shield_secret_token_1234567890",
            hivemind_api_key="hm_live_producer_key_abcdef",
            server_host="0.0.0.0",
        ).build_server_command()

    # Reject unmapped Ollama-style alias with colon
    with pytest.raises(ValueError, match="Ollama-style alias"):
        TensorSharpSingleModelProducerSpec(
            model_path="/models/Qwen3.5-9B-Q4_K_M.gguf",
            v1_models_id="qwen3.5:9b",
            public_tier_id="hivemind/qwen3.5-9b",
            public_https_shield_url="https://t4-shield.example.com",
            endpoint_token="hm_shield_secret_token_1234567890",
            hivemind_api_key="hm_live_producer_key_abcdef",
        ).build_shield_command()

    # Reject non-HTTPS shield exposure
    with pytest.raises(ValueError, match="https://"):
        TensorSharpSingleModelProducerSpec(
            model_path="/models/Qwen3.5-9B-Q4_K_M.gguf",
            v1_models_id="Qwen3.5-9B-Q4_K_M",
            public_tier_id="hivemind/qwen3.5-9b",
            public_https_shield_url="http://t4-shield.example.com",
            endpoint_token="hm_shield_secret_token_1234567890",
            hivemind_api_key="hm_live_producer_key_abcdef",
        ).build_agent_command()


def test_distributed_cluster_spec_commands_and_validation() -> None:
    driver = TensorSharpDistributedClusterSpec(
        model_path="/models/Muse-Spark-816B-UD-IQ3_XXS-00001-of-00008.gguf",
        mode="tp",
        local_degree=5,
        node_id=0,
        peer_endpoints=["10.150.0.10:9500", "10.150.0.11:9500"],
        all_reduce="symmetric",
        server_host_mode=True,
        http_port=8787,
    )
    cmd = driver.build_command()
    env = driver.build_env()
    assert "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--tp" in cmd and "5" in cmd
    assert env["TENSORSHARP_TP_PEERS"] == "10.150.0.10:9500,10.150.0.11:9500"
    assert env["TENSORSHARP_TP_ALLREDUCE"] == "symmetric"

    # Worker node (node_id=1) cannot run Server.Host
    with pytest.raises(ValueError, match="node_id 0"):
        TensorSharpDistributedClusterSpec(
            model_path="/models/model.gguf",
            mode="layer-split",
            local_degree=2,
            node_id=1,
            peer_endpoints=["10.150.0.10:9500", "10.150.0.11:9500"],
            server_host_mode=True,
        ).build_command()


def test_bee_lora_router_contract_and_collab_summary() -> None:
    assert SUPPORTED_TENSORSHARP_TEXT_LORA_ARCHITECTURES == (
        "nemotron_h",
        "mistral3",
        "hunyuan-dense",
    )

    router = TensorSharpBeeLoRARouter(
        base_model_id="mistral3-instruct",
        v1_models_id="Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M",
        architecture="mistral3",
        backend="ggml_cuda",
        loras_dir="/opt/hivemind/loras",
    )
    assert router.build_server_lora_flags() == ["--loras-dir", "/opt/hivemind/loras"]

    chat_req = router.build_openai_chat_request(
        public_model_id="hivemind/bee-stars",
        messages=[{"role": "user", "content": "Classify Balmer spectrum."}],
    )
    assert chat_req["model"] == "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M"
    assert chat_req["lora"] == "bee-stars"
    assert "lora_adapter" not in chat_req

    hb = router.build_openrouter_heartbeat(public_endpoint_url="https://t4-tensorsharp.example.com")
    assert hb["execution_class"] == "t4"
    assert any(
        m["id"] == "hivemind/bee-stars"
        and m["local_name"] == "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M"
        and m.get("lora") == "bee-stars"
        for m in hb["models"]
    )

    # Unsupported architecture (e.g. qwen3.5) or multi-GPU must raise ValueError
    with pytest.raises(ValueError, match="not supported by TensorSharp text-LoRA"):
        TensorSharpBeeLoRARouter(
            base_model_id="qwen3.5-35b-a3b",
            v1_models_id="Qwen3.5-35B-A3B-UD-IQ4_XS",
            architecture="qwen35",
        )

    with pytest.raises(ValueError, match="single device"):
        TensorSharpBeeLoRARouter(
            base_model_id="mistral3-instruct",
            v1_models_id="Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M",
            architecture="mistral3",
            multi_gpu=True,
        )

    summary = build_zhongkaifu_collab_summary()
    assert summary["reddit_post"]["github_handle"] == "zhongkaifu"
    assert summary["interpolated_hardware_estimates"]["is_interpolated_estimate"] is True
    assert summary["tesla_t4_measured_single_model_validation"]["commit"] == TENSORSHARP_VALIDATED_COMMIT
    assert len(summary["collab_pillars"]) == 5
