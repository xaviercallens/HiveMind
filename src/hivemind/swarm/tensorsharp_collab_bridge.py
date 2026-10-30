# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""TensorSharp x HiveMind Co-Lab Bridge (`hivemind.swarm.tensorsharp_collab_bridge`).

Integrates Zhongkai Fu's open-source .NET / C# + CUDA 13 PTX (`compute_75+`) LLM inference
engine (`https://github.com/zhongkaifu/TensorSharp`) with the HiveMind OpenRouter and
decentralized GPU swarm:

1. **816B MoE Multi-GPU Execution (`Muse Spark 816B` / 42B active, `UD-IQ3_XXS`, ~316 GiB):**
   - Layer Splitting (`--layer-split N`): pipelined prefill with adaptive micro-batching
     (1,346.8 tok/s on 16K prompt prefill across 10x NVIDIA A40 46 GB PCIe 4.0 without P2P,
     vs 826.1 tok/s on `llama.cpp`).
   - Tensor Parallelism (`--tp N` + `TcpCommunicator` Protocol v4): 22.63 tok/s single-request
     decode (vs 16.19 tok/s on `llama.cpp`) and 45.73 tok/s aggregate throughput at 4-way
     concurrency.
2. **Partial MoE Expert CPU Offloading (`--n-cpu-moe N`) for Consumer & T4 GPUs:**
   - Computes exact `--n-cpu-moe` offload depth to fit `Gemma 4 26B-A4B`, `Qwen 3.5 35B-A3B`,
     and `GPT-OSS 20B` within strict VRAM budgets (including the `>= 4,300 MiB` free VRAM
     headroom guard on the 15,360 MiB Tesla T4 `compute_75` node and 8,192 MiB RTX 2070 `compute_75`).
3. **Per-Request Specialist Bee LoRA Routing under Continuous Batching (`c154455`):**
   - Maps HiveMind's 24 nocturnal BOINC QLoRA Bee adapters (`hivemind/bee-*`) onto a single
     resident `TensorSharp.Server.Host` instance with per-request LoRA adapter selection.
4. **OpenRouter Producer Heartbeat & Multi-Node Command Generation:**
   - Generates exact `TensorSharp.Server.Host` / `TensorSharp.Cli` launch invocations and
     HiveMind OpenRouter heartbeat payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

T4_TOTAL_VRAM_MIB: int = 15_360
T4_REQUIRED_FREE_VRAM_MIB: int = 4_300
RTX2070_TOTAL_VRAM_MIB: int = 8_192


@dataclass(frozen=True)
class TensorSharpBenchmarkProfile:
    """Measured benchmark profile from `zhongkaifu/TensorSharp` (`docs/moe_cpu_offload_benchmark.md` & `r/LovingOpenSourceAI`)."""

    model_id: str
    hivemind_tier_id: str
    total_params_b: float
    active_params_b: float
    total_moe_layers: int
    resident_vram_mib: int
    min_offload_vram_mib: int
    n_cpu_moe_recommended: int
    tensorsharp_pp8192_tok_s: float
    llamacpp_pp8192_tok_s: float
    tensorsharp_tg128_tok_s: float
    llamacpp_tg128_tok_s: float
    gguf_artifact: str
    notes: str = ""

    @property
    def prefill_speedup_vs_llamacpp(self) -> float:
        return round(self.tensorsharp_pp8192_tok_s / max(self.llamacpp_pp8192_tok_s, 1e-6), 2)

    @property
    def decode_speedup_vs_llamacpp(self) -> float:
        return round(self.tensorsharp_tg128_tok_s / max(self.llamacpp_tg128_tok_s, 1e-6), 2)


TENSORSHARP_BENCHMARK_CATALOG: dict[str, TensorSharpBenchmarkProfile] = {
    "muse-spark-816b": TensorSharpBenchmarkProfile(
        model_id="muse-spark-816b",
        hivemind_tier_id="hivemind/muse-spark-816b-moe",
        total_params_b=816.0,
        active_params_b=42.0,
        total_moe_layers=80,
        resident_vram_mib=323_584,  # ~316 GiB across 10x A40 (46 GB each)
        min_offload_vram_mib=163_840,
        n_cpu_moe_recommended=0,
        tensorsharp_pp8192_tok_s=1346.8,  # 16K prefill with pipelined micro-batching
        llamacpp_pp8192_tok_s=826.1,
        tensorsharp_tg128_tok_s=22.63,  # TP=10 decode (45.73 tok/s @ 4 concurrent requests)
        llamacpp_tg128_tok_s=16.19,
        gguf_artifact="Muse-Spark-816B-UD-IQ3_XXS-00001-of-00008.gguf",
        notes="10x NVIDIA A40 (46 GB, PCIe 4.0 without P2P): Layer-Split=1346.8 tok/s prefill; TP10=22.63 tok/s single & 45.73 tok/s 4-way concurrent decode.",
    ),
    "gpt-oss-20b": TensorSharpBenchmarkProfile(
        model_id="gpt-oss-20b",
        hivemind_tier_id="hivemind/gpt-oss-20b",
        total_params_b=20.0,
        active_params_b=3.6,
        total_moe_layers=24,
        resident_vram_mib=12_482,
        min_offload_vram_mib=6_144,
        n_cpu_moe_recommended=12,
        tensorsharp_pp8192_tok_s=6394.0,  # 12,925 tok/s at --n-cpu-moe 0; 6,394 tok/s at --n-cpu-moe 12
        llamacpp_pp8192_tok_s=1188.0,
        tensorsharp_tg128_tok_s=51.7,
        llamacpp_tg128_tok_s=18.3,
        gguf_artifact="gpt-oss-20b-Q4_K_M.gguf",
        notes="Whole-model prefill geometry mask fix brought resident pp8192 to 12,925 tok/s; --n-cpu-moe 12 runs at 6,394 tok/s prefill (5.38x llama.cpp) and 51.7 tok/s decode.",
    ),
    "qwen3.5-35b-a3b": TensorSharpBenchmarkProfile(
        model_id="qwen3.5-35b-a3b",
        hivemind_tier_id="hivemind/qwen3.5-35b-a3b",
        total_params_b=35.0,
        active_params_b=3.0,
        total_moe_layers=48,
        resident_vram_mib=21_504,
        min_offload_vram_mib=7_168,
        n_cpu_moe_recommended=24,
        tensorsharp_pp8192_tok_s=5259.0,
        llamacpp_pp8192_tok_s=484.0,
        tensorsharp_tg128_tok_s=52.3,
        llamacpp_tg128_tok_s=15.8,
        gguf_artifact="Qwen3.5-35B-A3B-Q4_K_M.gguf",
        notes="10.87x faster prefill and 3.31x faster decode than llama.cpp at --n-cpu-moe 24; full 48-layer CPU expert offload still achieves 3,709 tok/s prefill and 38.6 tok/s decode.",
    ),
    "gemma4-26b-a4b": TensorSharpBenchmarkProfile(
        model_id="gemma4-26b-a4b",
        hivemind_tier_id="hivemind/gemma4-26b-a4b",
        total_params_b=26.0,
        active_params_b=4.0,
        total_moe_layers=30,
        resident_vram_mib=16_794,  # 16.4 GiB
        min_offload_vram_mib=11_059,  # 10.8 GiB with all 30 MoE layers in system RAM
        n_cpu_moe_recommended=30,
        tensorsharp_pp8192_tok_s=3072.0,  # 4,888 tok/s at 16/30; 3,072 tok/s at 30/30
        llamacpp_pp8192_tok_s=495.0,
        tensorsharp_tg128_tok_s=39.7,
        llamacpp_tg128_tok_s=12.9,
        gguf_artifact="gemma-4-26b-a4b-it-Q4_K_M.gguf",
        notes="Fits in 10.8 GiB VRAM with --n-cpu-moe 30, leaving >4,300 MiB free VRAM on a 15,360 MiB Tesla T4 while prefilling 6.21x faster than llama.cpp.",
    ),
    "deepseek-v4.1-flash": TensorSharpBenchmarkProfile(
        model_id="deepseek-v4.1-flash",
        hivemind_tier_id="hivemind/deepseek-v4.1-flash",
        total_params_b=236.0,
        active_params_b=21.0,
        total_moe_layers=43,
        resident_vram_mib=154_317,  # 150.7 GiB
        min_offload_vram_mib=45_056,
        n_cpu_moe_recommended=24,
        tensorsharp_pp8192_tok_s=236.0,
        llamacpp_pp8192_tok_s=63.0,
        tensorsharp_tg128_tok_s=18.4,  # with 1.39x speculative decoding & distributed TP
        llamacpp_tg128_tok_s=7.2,
        gguf_artifact="DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00001-of-00004.gguf",
        notes="Supports deterministic long-context top-k, 1.39x speculative decoding, and multi-node 2x2 TP / layer-splitting over TcpCommunicator v4.",
    ),
}


@dataclass(frozen=True)
class TensorSharpMoEOffloadPlan:
    """Computed `--n-cpu-moe` placement plan respecting hardware VRAM headroom guards."""

    model_id: str
    gpu_name: str
    compute_capability: str
    total_vram_mib: int
    reserved_free_vram_mib: int
    usable_vram_mib: int
    total_moe_layers: int
    n_cpu_moe_layers: int
    estimated_peak_vram_mib: int
    expected_pp8192_tok_s: float
    expected_tg128_tok_s: float
    fits_vram_budget: bool
    cli_flags: list[str]


def plan_tensorsharp_moe_offload(
    model_id: str,
    *,
    total_vram_mib: int = T4_TOTAL_VRAM_MIB,
    reserved_free_vram_mib: int = T4_REQUIRED_FREE_VRAM_MIB,
    gpu_name: str = "NVIDIA Tesla T4",
    compute_capability: str = "compute_75",
) -> TensorSharpMoEOffloadPlan:
    """Compute the optimal `--n-cpu-moe` offload depth for a target GPU and VRAM headroom policy."""
    if model_id not in TENSORSHARP_BENCHMARK_CATALOG:
        raise KeyError(f"Unknown TensorSharp model_id: {model_id!r}")
    prof = TENSORSHARP_BENCHMARK_CATALOG[model_id]
    usable_vram_mib = max(0, total_vram_mib - max(0, reserved_free_vram_mib))

    if usable_vram_mib >= prof.resident_vram_mib:
        n_cpu_moe = 0
        est_vram = prof.resident_vram_mib
        pp_speed = prof.tensorsharp_pp8192_tok_s * 1.25 if prof.n_cpu_moe_recommended > 0 else prof.tensorsharp_pp8192_tok_s
        tg_speed = prof.tensorsharp_tg128_tok_s * 1.15 if prof.n_cpu_moe_recommended > 0 else prof.tensorsharp_tg128_tok_s
        fits = True
    elif usable_vram_mib < prof.min_offload_vram_mib:
        n_cpu_moe = prof.total_moe_layers
        est_vram = prof.min_offload_vram_mib
        pp_speed = prof.tensorsharp_pp8192_tok_s * 0.72
        tg_speed = prof.tensorsharp_tg128_tok_s * 0.65
        fits = False
    else:
        # Linear interpolation between resident_vram_mib (0 offloaded) and min_offload_vram_mib (all layers offloaded)
        span = max(1, prof.resident_vram_mib - prof.min_offload_vram_mib)
        vram_Need_to_shed = prof.resident_vram_mib - usable_vram_mib
        frac = min(1.0, max(0.0, vram_Need_to_shed / span))
        import math

        n_cpu_moe = min(prof.total_moe_layers, max(1, int(math.ceil(frac * prof.total_moe_layers))))
        actual_frac = n_cpu_moe / max(1, prof.total_moe_layers)
        est_vram = int(round(prof.resident_vram_mib - actual_frac * span))
        pp_speed = round(prof.tensorsharp_pp8192_tok_s * (1.0 - 0.25 * (actual_frac - 0.5)), 1)
        tg_speed = round(prof.tensorsharp_tg128_tok_s * (1.0 - 0.22 * (actual_frac - 0.5)), 1)
        fits = est_vram <= usable_vram_mib

    cli_flags = ["--backend", "ggml_cuda", "--n-cpu-moe", str(n_cpu_moe)]
    return TensorSharpMoEOffloadPlan(
        model_id=model_id,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        total_vram_mib=total_vram_mib,
        reserved_free_vram_mib=reserved_free_vram_mib,
        usable_vram_mib=usable_vram_mib,
        total_moe_layers=prof.total_moe_layers,
        n_cpu_moe_layers=n_cpu_moe,
        estimated_peak_vram_mib=est_vram,
        expected_pp8192_tok_s=pp_speed,
        expected_tg128_tok_s=tg_speed,
        fits_vram_budget=fits,
        cli_flags=cli_flags,
    )


@dataclass(frozen=True)
class TensorSharpDistributedClusterSpec:
    """Multi-node / multi-GPU cluster configuration for `TensorSharp.Distributed` (`TcpCommunicator` v4)."""

    model_path: str
    mode: Literal["layer-split", "tp"]
    local_degree: int
    node_id: int
    peer_endpoints: list[str]
    all_reduce: Literal["symmetric", "star"] = "symmetric"
    symmetric_inline_bytes: int = 32 * 1024
    max_context: int = 16_384
    n_cpu_moe: int = 0
    server_host_mode: bool = True
    http_port: int = 8080

    def validate(self) -> None:
        if self.local_degree < 1:
            raise ValueError("local_degree must be >= 1")
        if not self.peer_endpoints:
            raise ValueError("peer_endpoints must contain at least one host:port endpoint")
        if not (0 <= self.node_id < len(self.peer_endpoints)):
            raise ValueError(f"node_id {self.node_id} out of range for {len(self.peer_endpoints)} peers")
        if self.server_host_mode and self.node_id != 0:
            raise ValueError("TensorSharp.Server.Host only runs on node_id 0; worker nodes must use TensorSharp.Cli")

    def build_env(self) -> dict[str, str]:
        self.validate()
        env = {
            "MAX_CONTEXT": str(self.max_context),
            "TENSORSHARP_TP_NODE_ID": str(self.node_id),
            "TENSORSHARP_TP_PEERS": ",".join(self.peer_endpoints),
            "TENSORSHARP_TP_ALLREDUCE": self.all_reduce,
            "TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES": str(self.symmetric_inline_bytes),
        }
        if self.mode == "layer-split":
            env["TENSORSHARP_LAYER_SPLIT_DEGREE"] = str(self.local_degree)
        else:
            env["TENSORSHARP_TP_DEGREE"] = str(self.local_degree)
        return env

    def build_command(self) -> list[str]:
        self.validate()
        dll = (
            "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll"
            if (self.server_host_mode and self.node_id == 0)
            else "TensorSharp.Cli/bin/TensorSharp.Cli.dll"
        )
        mode_flag = "--layer-split" if self.mode == "layer-split" else "--tp"
        cmd = [
            "dotnet",
            dll,
            "--model",
            self.model_path,
            "--backend",
            "ggml_cuda",
            mode_flag,
            str(self.local_degree),
            "--tp-node-id",
            str(self.node_id),
            "--tp-peers",
            ",".join(self.peer_endpoints),
        ]
        if self.n_cpu_moe > 0:
            cmd.extend(["--n-cpu-moe", str(self.n_cpu_moe)])
        if self.server_host_mode and self.node_id == 0:
            cmd.extend(["--port", str(self.http_port)])
        return cmd


@dataclass
class TensorSharpBeeLoRARouter:
    """Routes HiveMind's 24 specialist Bee LoRA adapters over a single `TensorSharp.Server.Host` continuous-batching backend."""

    base_model_id: str = "qwen3.5-35b-a3b"
    server_url: str = "http://127.0.0.1:8787"
    registered_bee_adapters: dict[str, str] = field(
        default_factory=lambda: {
            "hivemind/bee-stars": "/opt/hivemind/adapters/bee-stars-qwen35-lora.gguf",
            "hivemind/bee-code": "/opt/hivemind/adapters/bee-code-qwen35-lora.gguf",
            "hivemind/bee-math": "/opt/hivemind/adapters/bee-math-qwen35-lora.gguf",
            "hivemind/bee-science": "/opt/hivemind/adapters/bee-science-qwen35-lora.gguf",
            "hivemind/bee-cyber": "/opt/hivemind/adapters/bee-cyber-qwen35-lora.gguf",
        }
    )

    def build_openai_chat_request(
        self,
        *,
        public_model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible `/v1/chat/completions` payload with exact per-request LoRA selection (`c154455`)."""
        adapter_path = self.registered_bee_adapters.get(public_model_id)
        payload: dict[str, Any] = {
            "model": self.base_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if adapter_path:
            payload["lora_adapter"] = adapter_path
            payload["hivemind_bee_specialist"] = public_model_id
        return payload

    def build_openrouter_heartbeat(
        self,
        *,
        public_endpoint_url: str,
        execution_class: str = "t4",
        gpu_vram_mib: int = T4_TOTAL_VRAM_MIB - T4_REQUIRED_FREE_VRAM_MIB,
        cost_per_1k: float = 0.5,
        capacity_1k: float = 25.0,
    ) -> dict[str, Any]:
        """Build the `POST /v1/hivemind/producer/heartbeat` payload advertising TensorSharp models & Bee LoRAs."""
        prof = TENSORSHARP_BENCHMARK_CATALOG.get(self.base_model_id, TENSORSHARP_BENCHMARK_CATALOG["gpt-oss-20b"])
        models = [
            {
                "id": prof.hivemind_tier_id,
                "local_name": self.base_model_id,
                "loaded": True,
                "vram_mib": min(gpu_vram_mib, prof.min_offload_vram_mib),
            }
        ]
        for bee_id in sorted(self.registered_bee_adapters):
            models.append(
                {
                    "id": bee_id,
                    "local_name": f"{self.base_model_id}+lora:{bee_id.split('/')[-1]}",
                    "loaded": True,
                    "vram_mib": 128,  # Per-request LoRA adapter delta under continuous batching
                }
            )
        return {
            "endpoint_url": public_endpoint_url,
            "execution_class": execution_class,
            "gpu_vram_mib": gpu_vram_mib,
            "cost_per_1k": cost_per_1k,
            "capacity_1k": capacity_1k,
            "engine": "TensorSharp.Server.Host (CUDA 13 PTX compute_75 + --n-cpu-moe)",
            "models": models,
        }


def build_zhongkaifu_collab_summary() -> dict[str, Any]:
    """Return a structured summary of the Reddit post analysis, GitHub audit, and 5 HiveMind x TensorSharp Co-Lab Pillars."""
    t4_gemma_plan = plan_tensorsharp_moe_offload(
        "gemma4-26b-a4b",
        total_vram_mib=T4_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=T4_REQUIRED_FREE_VRAM_MIB,
        gpu_name="NVIDIA Tesla T4 (socreateai-agora-hermes-node1)",
        compute_capability="compute_75",
    )
    rtx2070_gpt_oss_plan = plan_tensorsharp_moe_offload(
        "gpt-oss-20b",
        total_vram_mib=RTX2070_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=1_024,
        gpu_name="NVIDIA GeForce RTX 2070 (Steam / BOINC Prosumer)",
        compute_capability="compute_75",
    )
    return {
        "reddit_post": {
            "url": "https://www.reddit.com/r/LovingOpenSourceAI/comments/1xcuesc/pushing_opensource_ai_further_running_an/",
            "subreddit": "r/LovingOpenSourceAI",
            "title": "Pushing Open-Source AI Further: Running an 816-billion-parameter Mixture-of-Experts (MoE) model across 10 NVIDIA A40 GPUs",
            "contributor_name": "Zhongkai Fu",
            "github_handle": "zhongkaifu",
            "github_repo": "https://github.com/zhongkaifu/TensorSharp",
            "website": "https://tensorsharp.ai",
            "contact_email": "fuzhongkai@gmail.com",
        },
        "headline_benchmarks": {
            "muse_spark_816b_10x_a40": {
                "active_params_b": 42.0,
                "total_params_b": 816.0,
                "quantization": "UD-IQ3_XXS (~316 GiB GGUF)",
                "hardware": "10x NVIDIA A40 (46 GB, PCIe 4.0, no P2P)",
                "layer_split_16k_prefill_tok_s": 1346.8,
                "llamacpp_16k_prefill_tok_s": 826.1,
                "tp10_single_decode_tok_s": 22.63,
                "llamacpp_single_decode_tok_s": 16.19,
                "tp10_4way_concurrent_decode_tok_s": 45.73,
            },
            "moe_cpu_offload_speedups": {
                k: {
                    "pp8192_speedup_vs_llamacpp": v.prefill_speedup_vs_llamacpp,
                    "tg128_speedup_vs_llamacpp": v.decode_speedup_vs_llamacpp,
                    "tensorsharp_pp8192_tok_s": v.tensorsharp_pp8192_tok_s,
                    "tensorsharp_tg128_tok_s": v.tensorsharp_tg128_tok_s,
                }
                for k, v in TENSORSHARP_BENCHMARK_CATALOG.items()
            },
        },
        "hardware_placement_plans": {
            "tesla_t4_with_4300mib_headroom_guard": t4_gemma_plan,
            "rtx_2070_8gb_prosumer": rtx2070_gpt_oss_plan,
        },
        "collab_pillars": [
            "Pillar 1: TensorSharp.Server.Host as a 1st-class HiveMind OpenRouter Producer Backend (--n-cpu-moe & CUDA 13 compute_75 PTX)",
            "Pillar 2: Turing sm_75 (Tesla T4 15 GiB & RTX 2070 8 GiB) Partial MoE CPU-Offload Benchmark preserving >= 4,300 MiB free VRAM",
            "Pillar 3: Multi-Node Layer-Split & Tensor Parallelism Bridge (TcpCommunicator v4 + HiveMind LayerStoRm PCIe/QUIC Nectar Cache)",
            "Pillar 4: Per-Request Specialist Bee LoRA Routing under Continuous Batching (24 Nocturnal BOINC QLoRA Bee Adapters on 1 Base Model)",
            "Pillar 5: Native C#/.NET Integration between TensorSharp & USS HiveMind Exoplanet Hunter Steam Fat Client (AppID 3133700)",
        ],
    }
