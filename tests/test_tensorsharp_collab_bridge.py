# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the TensorSharp x HiveMind Co-Lab Bridge (`hivemind.swarm.tensorsharp_collab_bridge`)."""

from __future__ import annotations

import pytest

from hivemind.swarm.tensorsharp_collab_bridge import (
    RTX2070_TOTAL_VRAM_MIB,
    T4_REQUIRED_FREE_VRAM_MIB,
    T4_TOTAL_VRAM_MIB,
    TENSORSHARP_BENCHMARK_CATALOG,
    TensorSharpBeeLoRARouter,
    TensorSharpDistributedClusterSpec,
    build_zhongkaifu_collab_summary,
    plan_tensorsharp_moe_offload,
)


def test_tensorsharp_benchmark_catalog_speedups() -> None:
    muse = TENSORSHARP_BENCHMARK_CATALOG["muse-spark-816b"]
    assert muse.total_params_b == 816.0
    assert muse.active_params_b == 42.0
    assert muse.prefill_speedup_vs_llamacpp == pytest.approx(1.63, abs=0.02)
    assert muse.decode_speedup_vs_llamacpp == pytest.approx(1.40, abs=0.02)

    qwen35 = TENSORSHARP_BENCHMARK_CATALOG["qwen3.5-35b-a3b"]
    assert qwen35.prefill_speedup_vs_llamacpp > 10.5
    assert qwen35.decode_speedup_vs_llamacpp > 3.2


def test_tesla_t4_vram_headroom_guard_with_gemma4_and_gpt_oss() -> None:
    # On Tesla T4 (15,360 MiB) with >= 4,300 MiB free VRAM guard -> 11,060 MiB usable VRAM
    plan_gemma = plan_tensorsharp_moe_offload(
        "gemma4-26b-a4b",
        total_vram_mib=T4_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=T4_REQUIRED_FREE_VRAM_MIB,
    )
    assert plan_gemma.usable_vram_mib == 11_060
    assert plan_gemma.fits_vram_budget is True
    assert plan_gemma.estimated_peak_vram_mib <= plan_gemma.usable_vram_mib
    assert T4_TOTAL_VRAM_MIB - plan_gemma.estimated_peak_vram_mib >= T4_REQUIRED_FREE_VRAM_MIB
    assert plan_gemma.n_cpu_moe_layers == 30

    # On RTX 2070 (8,192 MiB) with 1,024 MiB reserve -> 7,168 MiB usable VRAM
    plan_rtx = plan_tensorsharp_moe_offload(
        "gpt-oss-20b",
        total_vram_mib=RTX2070_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=1_024,
    )
    assert plan_rtx.usable_vram_mib == 7_168
    assert plan_rtx.fits_vram_budget is True
    assert 1 <= plan_rtx.n_cpu_moe_layers <= 24
    assert "--n-cpu-moe" in plan_rtx.cli_flags


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


def test_bee_lora_router_and_collab_summary() -> None:
    router = TensorSharpBeeLoRARouter(base_model_id="qwen3.5-35b-a3b")
    chat_req = router.build_openai_chat_request(
        public_model_id="hivemind/bee-stars",
        messages=[{"role": "user", "content": "Classify Balmer spectrum."}],
    )
    assert chat_req["model"] == "qwen3.5-35b-a3b"
    assert chat_req["lora_adapter"].endswith("bee-stars-qwen35-lora.gguf")
    assert chat_req["hivemind_bee_specialist"] == "hivemind/bee-stars"

    hb = router.build_openrouter_heartbeat(public_endpoint_url="https://t4-tensorsharp.example.com")
    assert hb["execution_class"] == "t4"
    assert any(m["id"] == "hivemind/bee-stars" for m in hb["models"])

    summary = build_zhongkaifu_collab_summary()
    assert summary["reddit_post"]["github_handle"] == "zhongkaifu"
    assert len(summary["collab_pillars"]) == 5
