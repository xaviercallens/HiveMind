# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""TensorSharp x HiveMind Co-Lab Bridge (`hivemind.swarm.tensorsharp_collab_bridge`).

Integrates Zhongkai Fu's open-source .NET / C# + CUDA (`compute_75+`) LLM inference
engine (`https://github.com/zhongkaifu/TensorSharp`) with the HiveMind OpenRouter and
decentralized GPU swarm:

1. **Single-Model Producer Security & `/v1/models` ID Consistency:**
   - Binds `TensorSharp.Server.Host` explicitly to loopback (`--host 127.0.0.1 --port 11436 --no-webui`)
     because its default listener (`0.0.0.0:5000`) binds to all interfaces without authentication or TLS.
   - Binds `hivemind_producer_shield.py` with `--host 127.0.0.1 --port 8787` (requiring
     `HIVEMIND_ENDPOINT_TOKEN` >= 16 chars) and exposes **only** the shield via an authenticated,
     TLS-encrypted `https://` connection paired with matching `HIVEMIND_ENDPOINT_TOKEN` and
     `HIVEMIND_API_KEY` credentials in `hivemind_producer_agent.py`.
   - Uses the exact model ID returned by `GET /v1/models` (the GGUF filename stem, such as
     `"Qwen3.5-9B-Q4_K_M"` or `"gpt-oss-20b-MXFP4"`) consistently across chat requests,
     `--allow-model`, and `--model <tier>=<v1_models_id>` producer registration.
2. **Strict Separation of Reference Benchmarks, Interpolated Estimates & Live Tesla T4 Measurements:**
   - `TENSORSHARP_BENCHMARK_CATALOG`: Reference benchmarks from `docs/moe_cpu_offload_benchmark.md`
     recorded on **2x NVIDIA RTX PRO 6000 Blackwell Server Edition (`97,887 MiB` each, PCIe 5.0 x16,
     2x Intel Xeon 6952P, `1,511 GiB` RAM, CUDA 12.8)** using `UD-IQ4_XS` for `Gemma 4 26B-A4B`
     and `Qwen 3.5 35B-A3B` and `Q8_0 / MXFP4` for `GPT-OSS 20B` (plus 10x A40 PCIe 4.0 reference
     numbers for `Muse Spark 816B UD-IQ3_XXS`).
   - `plan_tensorsharp_moe_offload`: Explicitly marks interpolated VRAM and throughput figures as
     **linear interpolation estimates** (`is_interpolated_estimate=True`).
   - `TENSORSHARP_T4_MEASURED_VALIDATION`: Direct end-to-end single-model producer measurements
     recorded on a live **15,360 MiB NVIDIA Tesla T4 (`compute_75`)** (`v2026.10.23`, commit
     `a37448127f2fcd70079889c7050bdafb6b4bbfc7`).
3. **Exact `TensorSharp.Server.Host` Text-LoRA API Contract:**
   - Uses `--loras-dir <dir>` (or `TS_LORAS_DIR` / `POST /v1/loras`) and the request field
     `"lora": "<registered_adapter_name>"` (not `"lora_adapter"` with a file path).
   - Enforces TensorSharp's supported single-device text-LoRA architecture contract:
     `("nemotron_h", "mistral3", "hunyuan-dense")` on a single device (`cpu`, `ggml_cpu`,
     `ggml_cuda`, `cuda`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

T4_TOTAL_VRAM_MIB: int = 15_360
T4_REQUIRED_FREE_VRAM_MIB: int = 4_300
RTX2070_TOTAL_VRAM_MIB: int = 8_192

BLACKWELL_REFERENCE_HARDWARE: str = (
    "2x NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, PCIe 5.0 x16, "
    "2x Intel Xeon 6952P, 1,511 GiB RAM, CUDA 12.8)"
)
A40_CLUSTER_REFERENCE_HARDWARE: str = "10x NVIDIA A40 (46 GB each, PCIe 4.0 without P2P)"
T4_MEASURED_HARDWARE: str = (
    "1x NVIDIA Tesla T4 (15,360 MiB GDDR6, Turing compute_75, Driver 580.178.04, CUDA 13.0, "
    "8x vCPU Intel Xeon @ 2.20 GHz, 29 GiB RAM)"
)
TENSORSHARP_VALIDATED_COMMIT: str = "a37448127f2fcd70079889c7050bdafb6b4bbfc7"
TENSORSHARP_VALIDATED_RELEASE: str = "v2026.10.23"

SUPPORTED_TENSORSHARP_TEXT_LORA_ARCHITECTURES: tuple[str, ...] = (
    "nemotron_h",
    "mistral3",
    "hunyuan-dense",
)
SUPPORTED_TENSORSHARP_TEXT_LORA_BACKENDS: tuple[str, ...] = (
    "cpu",
    "ggml_cpu",
    "ggml_cuda",
    "cuda",
)


@dataclass(frozen=True)
class TensorSharpBenchmarkProfile:
    """Reference benchmark profile from `zhongkaifu/TensorSharp` (`docs/moe_cpu_offload_benchmark.md` & `r/LovingOpenSourceAI`).

    Note: The `--n-cpu-moe` reference profiles below were measured on `BLACKWELL_REFERENCE_HARDWARE`
    using `UD-IQ4_XS` (`Gemma 4 26B-A4B`, `Qwen 3.5 35B-A3B`) or `Q8_0 / MXFP4` (`GPT-OSS 20B`).
    """

    model_id: str
    v1_models_id: str
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
    quantization: str = "UD-IQ4_XS"
    reference_hardware: str = BLACKWELL_REFERENCE_HARDWARE
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
        v1_models_id="Muse-Spark-816B-UD-IQ3_XXS-00001-of-00008",
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
        quantization="UD-IQ3_XXS",
        reference_hardware=A40_CLUSTER_REFERENCE_HARDWARE,
        notes=(
            "Reference benchmark on 10x NVIDIA A40 (46 GB, PCIe 4.0 without P2P): "
            "Layer-Split=1346.8 tok/s prefill; TP10=22.63 tok/s single & 45.73 tok/s 4-way concurrent decode."
        ),
    ),
    "gpt-oss-20b": TensorSharpBenchmarkProfile(
        model_id="gpt-oss-20b",
        v1_models_id="gpt-oss-20b-MXFP4",
        hivemind_tier_id="hivemind/gpt-oss-20b",
        total_params_b=20.0,
        active_params_b=3.6,
        total_moe_layers=24,
        resident_vram_mib=11_776,  # ~11.5 GiB resident on RTX PRO 6000 Blackwell
        min_offload_vram_mib=2_253,  # ~2.2 GiB at --n-cpu-moe 24 on RTX PRO 6000 Blackwell
        n_cpu_moe_recommended=12,
        tensorsharp_pp8192_tok_s=6394.0,  # 12,925 tok/s at --n-cpu-moe 0; 6,394 tok/s at --n-cpu-moe 12
        llamacpp_pp8192_tok_s=1188.0,
        tensorsharp_tg128_tok_s=51.7,
        llamacpp_tg128_tok_s=18.3,
        gguf_artifact="gpt-oss-20b-MXFP4.gguf",
        quantization="Q8_0 / MXFP4",
        reference_hardware=BLACKWELL_REFERENCE_HARDWARE,
        notes=(
            "Reference benchmark on RTX PRO 6000 Blackwell (Q8_0 / MXFP4, 11.3 GiB GGUF): "
            "--n-cpu-moe 12 runs at 6,394 tok/s prefill (5.38x llama.cpp) and 51.7 tok/s decode; "
            "--n-cpu-moe 24 runs at 3,798 tok/s prefill (6.93x) and 27.7 tok/s decode (~2.2 GiB VRAM)."
        ),
    ),
    "qwen3.5-35b-a3b": TensorSharpBenchmarkProfile(
        model_id="qwen3.5-35b-a3b",
        v1_models_id="Qwen3.5-35B-A3B-UD-IQ4_XS",
        hivemind_tier_id="hivemind/qwen3.5-35b-a3b",
        total_params_b=35.0,
        active_params_b=3.0,
        total_moe_layers=48,
        resident_vram_mib=21_811,  # ~21.3 GiB resident on RTX PRO 6000 Blackwell
        min_offload_vram_mib=13_210,  # ~12.9 GiB at --n-cpu-moe 48 on RTX PRO 6000 Blackwell
        n_cpu_moe_recommended=24,
        tensorsharp_pp8192_tok_s=5259.0,
        llamacpp_pp8192_tok_s=484.0,
        tensorsharp_tg128_tok_s=52.3,
        llamacpp_tg128_tok_s=15.8,
        gguf_artifact="Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
        quantization="UD-IQ4_XS",
        reference_hardware=BLACKWELL_REFERENCE_HARDWARE,
        notes=(
            "Reference benchmark on RTX PRO 6000 Blackwell (UD-IQ4_XS, 17.7 GiB GGUF): "
            "10.87x faster prefill and 3.31x faster decode than llama.cpp at --n-cpu-moe 24; "
            "full 48-layer CPU expert offload achieves 3,709 tok/s prefill and 38.6 tok/s decode."
        ),
    ),
    "gemma4-26b-a4b": TensorSharpBenchmarkProfile(
        model_id="gemma4-26b-a4b",
        v1_models_id="gemma-4-26B-A4B-it-UD-IQ4_XS",
        hivemind_tier_id="hivemind/gemma4-26b-a4b",
        total_params_b=26.0,
        active_params_b=4.0,
        total_moe_layers=30,
        resident_vram_mib=16_794,  # 16.4 GiB resident on RTX PRO 6000 Blackwell (UD-IQ4_XS)
        min_offload_vram_mib=11_059,  # 10.8 GiB at --n-cpu-moe 30 on RTX PRO 6000 Blackwell (UD-IQ4_XS)
        n_cpu_moe_recommended=30,
        tensorsharp_pp8192_tok_s=3072.0,  # 4,888 tok/s at 16/30; 3,072 tok/s at 30/30
        llamacpp_pp8192_tok_s=495.0,
        tensorsharp_tg128_tok_s=39.7,
        llamacpp_tg128_tok_s=12.9,
        gguf_artifact="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
        quantization="UD-IQ4_XS",
        reference_hardware=BLACKWELL_REFERENCE_HARDWARE,
        notes=(
            "Reference benchmark on RTX PRO 6000 Blackwell (UD-IQ4_XS, 12.8 GiB GGUF): "
            "10.8 GiB VRAM at --n-cpu-moe 30 with 3,072 tok/s prefill (6.21x llama.cpp) and 39.7 tok/s decode."
        ),
    ),
    "deepseek-v4.1-flash": TensorSharpBenchmarkProfile(
        model_id="deepseek-v4.1-flash",
        v1_models_id="DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00001-of-00004",
        hivemind_tier_id="hivemind/deepseek-v4.1-flash",
        total_params_b=236.0,
        active_params_b=21.0,
        total_moe_layers=43,
        resident_vram_mib=154_317,  # 150.7 GiB across 2x RTX PRO 6000 Blackwell
        min_offload_vram_mib=45_056,
        n_cpu_moe_recommended=24,
        tensorsharp_pp8192_tok_s=236.0,
        llamacpp_pp8192_tok_s=63.0,
        tensorsharp_tg128_tok_s=18.4,
        llamacpp_tg128_tok_s=7.2,
        gguf_artifact="DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00001-of-00004.gguf",
        quantization="UD-IQ3_XXS",
        reference_hardware=BLACKWELL_REFERENCE_HARDWARE,
        notes="Reference benchmark on 2x RTX PRO 6000 Blackwell with deterministic top-k and 1.39x speculative decoding.",
    ),
}


@dataclass(frozen=True)
class TensorSharpT4MeasuredRun:
    """Direct single-model producer measurement on a live 15,360 MiB NVIDIA Tesla T4 (`compute_75`)."""

    run_key: str
    v1_models_id: str
    gguf_file: str
    gguf_size_bytes: int
    gguf_sha256: str
    gguf_source: str
    backend: str
    n_cpu_moe: int | None
    max_context: int
    idle_vram_mib: int
    peak_vram_mib: int
    prompt_tokens: int
    completion_tokens: int
    sse_chunks: int
    ttft_ms: float
    measured_prefill_tok_s: float
    measured_stream_decode_tok_s: float
    measured_non_stream_e2e_tok_s: float
    auth_checks_passed: bool = True
    commit: str = TENSORSHARP_VALIDATED_COMMIT
    release: str = TENSORSHARP_VALIDATED_RELEASE
    hardware: str = T4_MEASURED_HARDWARE


TENSORSHARP_T4_MEASURED_VALIDATION: dict[str, TensorSharpT4MeasuredRun] = {
    "qwen3.5-9b-q4_k_m-resident": TensorSharpT4MeasuredRun(
        run_key="qwen3.5-9b-q4_k_m-resident",
        v1_models_id="Qwen3.5-9B-Q4_K_M",
        gguf_file="Qwen3.5-9B-Q4_K_M.gguf",
        gguf_size_bytes=5_680_522_464,
        gguf_sha256="03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        gguf_source="unsloth/Qwen3.5-9B-GGUF (Q4_K_M, 5.29 GiB)",
        backend="ggml_cuda",
        n_cpu_moe=None,
        max_context=4096,
        idle_vram_mib=5_420,
        peak_vram_mib=6_062,
        prompt_tokens=3_289,
        completion_tokens=96,
        sse_chunks=97,
        ttft_ms=3451.8,
        measured_prefill_tok_s=952.84,
        measured_stream_decode_tok_s=32.75,
        measured_non_stream_e2e_tok_s=14.55,
    ),
    "gpt-oss-20b-mxfp4-ncpumoe18": TensorSharpT4MeasuredRun(
        run_key="gpt-oss-20b-mxfp4-ncpumoe18",
        v1_models_id="gpt-oss-20b-MXFP4",
        gguf_file="gpt-oss-20b-MXFP4.gguf",
        gguf_size_bytes=13_793_422_816,
        gguf_sha256="8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716",
        gguf_source="dhiltgen/gpt-oss:20b-llamacpp (MXFP4, 24 MoE layers, 12.85 GiB)",
        backend="ggml_cuda",
        n_cpu_moe=18,
        max_context=4096,
        idle_vram_mib=6_104,
        peak_vram_mib=7_042,
        prompt_tokens=2_677,
        completion_tokens=96,
        sse_chunks=88,
        ttft_ms=4283.7,
        measured_prefill_tok_s=624.92,
        measured_stream_decode_tok_s=13.04,
        measured_non_stream_e2e_tok_s=8.11,
    ),
    "gpt-oss-20b-mxfp4-ncpumoe24": TensorSharpT4MeasuredRun(
        run_key="gpt-oss-20b-mxfp4-ncpumoe24",
        v1_models_id="gpt-oss-20b-MXFP4",
        gguf_file="gpt-oss-20b-MXFP4.gguf",
        gguf_size_bytes=13_793_422_816,
        gguf_sha256="8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716",
        gguf_source="dhiltgen/gpt-oss:20b-llamacpp (MXFP4, 24 MoE layers, 12.85 GiB)",
        backend="ggml_cuda",
        n_cpu_moe=24,
        max_context=4096,
        idle_vram_mib=3_662,
        peak_vram_mib=4_610,
        prompt_tokens=2_677,
        completion_tokens=96,
        sse_chunks=88,
        ttft_ms=4730.0,
        measured_prefill_tok_s=565.96,
        measured_stream_decode_tok_s=10.76,
        measured_non_stream_e2e_tok_s=6.88,
    ),
}


@dataclass(frozen=True)
class TensorSharpMoEOffloadPlan:
    """Estimated `--n-cpu-moe` placement plan interpolated from RTX PRO 6000 Blackwell reference endpoints.

    All `estimated_peak_vram_mib`, `estimated_pp8192_tok_s`, and `estimated_tg128_tok_s` values are
    explicitly labeled as **estimates** (`is_interpolated_estimate=True`), not direct hardware measurements.
    For measured single-model Tesla T4 numbers, see `TENSORSHARP_T4_MEASURED_VALIDATION`.
    """

    model_id: str
    v1_models_id: str
    gpu_name: str
    compute_capability: str
    total_vram_mib: int
    reserved_free_vram_mib: int
    usable_vram_mib: int
    total_moe_layers: int
    n_cpu_moe_layers: int
    estimated_peak_vram_mib: int
    estimated_pp8192_tok_s: float
    estimated_tg128_tok_s: float
    fits_vram_budget: bool
    cli_flags: list[str]
    is_interpolated_estimate: bool = True
    reference_hardware: str = BLACKWELL_REFERENCE_HARDWARE
    reference_quantization: str = "UD-IQ4_XS"
    estimation_note: str = (
        "ESTIMATE ONLY: Linearly interpolated from RTX PRO 6000 Blackwell reference endpoints "
        "(docs/moe_cpu_offload_benchmark.md). Actual throughput and VRAM depend on host CPU/RAM bandwidth, "
        "PCIe generation, quantization, and MAX_CONTEXT."
    )

    @property
    def expected_pp8192_tok_s(self) -> float:
        """Backward-compatible alias for `estimated_pp8192_tok_s`."""
        return self.estimated_pp8192_tok_s

    @property
    def expected_tg128_tok_s(self) -> float:
        """Backward-compatible alias for `estimated_tg128_tok_s`."""
        return self.estimated_tg128_tok_s


def plan_tensorsharp_moe_offload(
    model_id: str,
    *,
    total_vram_mib: int = T4_TOTAL_VRAM_MIB,
    reserved_free_vram_mib: int = T4_REQUIRED_FREE_VRAM_MIB,
    gpu_name: str = "NVIDIA Tesla T4",
    compute_capability: str = "compute_75",
) -> TensorSharpMoEOffloadPlan:
    """Compute an estimated `--n-cpu-moe` offload depth for a target GPU and VRAM headroom policy."""
    if model_id not in TENSORSHARP_BENCHMARK_CATALOG:
        raise KeyError(f"Unknown TensorSharp model_id: {model_id!r}")
    prof = TENSORSHARP_BENCHMARK_CATALOG[model_id]
    usable_vram_mib = max(0, total_vram_mib - max(0, reserved_free_vram_mib))

    if usable_vram_mib >= prof.resident_vram_mib:
        n_cpu_moe = 0
        est_vram = prof.resident_vram_mib
        pp_speed = (
            prof.tensorsharp_pp8192_tok_s * 1.25
            if prof.n_cpu_moe_recommended > 0
            else prof.tensorsharp_pp8192_tok_s
        )
        tg_speed = (
            prof.tensorsharp_tg128_tok_s * 1.15
            if prof.n_cpu_moe_recommended > 0
            else prof.tensorsharp_tg128_tok_s
        )
        fits = True
    elif usable_vram_mib < prof.min_offload_vram_mib:
        n_cpu_moe = prof.total_moe_layers
        est_vram = prof.min_offload_vram_mib
        pp_speed = prof.tensorsharp_pp8192_tok_s * 0.72
        tg_speed = prof.tensorsharp_tg128_tok_s * 0.65
        fits = False
    else:
        span = max(1, prof.resident_vram_mib - prof.min_offload_vram_mib)
        vram_need_to_shed = prof.resident_vram_mib - usable_vram_mib
        frac = min(1.0, max(0.0, vram_need_to_shed / span))

        n_cpu_moe = min(prof.total_moe_layers, max(1, int(math.ceil(frac * prof.total_moe_layers))))
        actual_frac = n_cpu_moe / max(1, prof.total_moe_layers)
        est_vram = int(round(prof.resident_vram_mib - actual_frac * span))
        pp_speed = round(prof.tensorsharp_pp8192_tok_s * (1.0 - 0.25 * (actual_frac - 0.5)), 1)
        tg_speed = round(prof.tensorsharp_tg128_tok_s * (1.0 - 0.22 * (actual_frac - 0.5)), 1)
        fits = est_vram <= usable_vram_mib

    cli_flags = [
        "--backend",
        "ggml_cuda",
        "--n-cpu-moe",
        str(n_cpu_moe),
        "--host",
        "127.0.0.1",
    ]
    return TensorSharpMoEOffloadPlan(
        model_id=model_id,
        v1_models_id=prof.v1_models_id,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        total_vram_mib=total_vram_mib,
        reserved_free_vram_mib=reserved_free_vram_mib,
        usable_vram_mib=usable_vram_mib,
        total_moe_layers=prof.total_moe_layers,
        n_cpu_moe_layers=n_cpu_moe,
        estimated_peak_vram_mib=est_vram,
        estimated_pp8192_tok_s=pp_speed,
        estimated_tg128_tok_s=tg_speed,
        fits_vram_budget=fits,
        cli_flags=cli_flags,
        is_interpolated_estimate=True,
        reference_hardware=prof.reference_hardware,
        reference_quantization=prof.quantization,
    )


@dataclass(frozen=True)
class TensorSharpSingleModelProducerSpec:
    """Secure single-model `TensorSharp.Server.Host` producer configuration with shield & agent commands.

    Enforces:
    1. `--host 127.0.0.1` loopback binding on `TensorSharp.Server.Host` (`--no-webui`).
    2. `--host 127.0.0.1 --port <shield_port>` on `hivemind_producer_shield.py` with `HIVEMIND_ENDPOINT_TOKEN`.
    3. Exact `/v1/models` ID (`v1_models_id`) across `--allow-model`, `--model <tier>=<v1_models_id>`, and chat payloads.
    4. Encrypted `https://` public URL exposing only the shield.
    """

    model_path: str
    v1_models_id: str
    public_tier_id: str
    public_https_shield_url: str
    endpoint_token: str
    hivemind_api_key: str
    backend: str = "ggml_cuda"
    n_cpu_moe: int | None = None
    max_context: int = 4_096
    server_host: str = "127.0.0.1"
    server_port: int = 11_436
    shield_host: str = "127.0.0.1"
    shield_port: int = 8_787
    max_concurrent: int = 2
    execution_class: str = "t4"
    gpu_vram_mib: int = T4_TOTAL_VRAM_MIB
    router_url: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app"

    def validate(self) -> None:
        if self.server_host != "127.0.0.1" and self.server_host != "localhost":
            raise ValueError(
                "TensorSharp.Server.Host must bind to loopback (--host 127.0.0.1) so unauthenticated "
                "listeners are never exposed directly."
            )
        if len(self.endpoint_token.strip()) < 16:
            raise ValueError("HIVEMIND_ENDPOINT_TOKEN must be at least 16 characters.")
        if not self.hivemind_api_key.strip():
            raise ValueError("HIVEMIND_API_KEY is required for producer heartbeat registration.")
        if not self.public_https_shield_url.startswith("https://"):
            raise ValueError(
                "Only the shield may be exposed publicly, and it must use an encrypted https:// URL."
            )
        if not self.v1_models_id.strip() or ":" in self.v1_models_id:
            raise ValueError(
                "v1_models_id must be the exact ID returned by GET /v1/models (GGUF stem), "
                "not an unmapped Ollama-style alias."
            )

    def build_credentials_env(self) -> dict[str, str]:
        self.validate()
        return {
            "MAX_CONTEXT": str(self.max_context),
            "HIVEMIND_ENDPOINT_TOKEN": self.endpoint_token,
            "HIVEMIND_API_KEY": self.hivemind_api_key,
        }

    def build_server_command(self) -> list[str]:
        self.validate()
        cmd = [
            "dotnet",
            "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll",
            "--model",
            self.model_path,
            "--backend",
            self.backend,
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "--no-webui",
            "--no-prefix-cache",
        ]
        if self.n_cpu_moe is not None and self.n_cpu_moe > 0:
            cmd.extend(["--n-cpu-moe", str(self.n_cpu_moe)])
        return cmd

    def build_shield_command(self) -> list[str]:
        self.validate()
        return [
            "python3",
            "scripts/hivemind_producer_shield.py",
            "--upstream",
            f"http://{self.server_host}:{self.server_port}",
            "--host",
            self.shield_host,
            "--port",
            str(self.shield_port),
            "--allow-model",
            self.v1_models_id,
            "--max-concurrent",
            str(self.max_concurrent),
        ]

    def build_agent_command(self) -> list[str]:
        self.validate()
        return [
            "python3",
            "scripts/hivemind_producer_agent.py",
            "--router-url",
            self.router_url,
            "--public-url",
            self.public_https_shield_url,
            "--local-url",
            f"http://{self.server_host}:{self.server_port}",
            "--model",
            f"{self.public_tier_id}={self.v1_models_id}",
            "--execution-class",
            self.execution_class,
            "--gpu-vram-mib",
            str(self.gpu_vram_mib),
        ]


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
    http_host: str = "127.0.0.1"
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
        if self.server_host_mode and self.http_host not in ("127.0.0.1", "localhost"):
            raise ValueError("TensorSharp.Server.Host must bind to loopback (--host 127.0.0.1)")

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
            cmd.extend(["--host", self.http_host, "--port", str(self.http_port), "--no-webui"])
        return cmd


@dataclass
class TensorSharpBeeLoRARouter:
    """Routes per-request text-LoRA adapters on `TensorSharp.Server.Host` using `--loras-dir` and `"lora"`.

    Contract (`TensorSharp.Server.Host/API_EXAMPLES.md`):
    - Adapters are registered at startup under `--loras-dir <dir>` (or `TS_LORAS_DIR`) by filename stem,
      or at runtime via `POST /v1/loras` (`{"name": "<stem>", "path": "<file>", "scale": 1.0}`, max 32).
    - Requests select an adapter via `"lora": "<adapter_name>"` (NOT `"lora_adapter"` with a file path)
      and must specify the exact `v1_models_id` returned by `GET /v1/models`.
    - Current text-LoRA support in `TensorSharp.Server.Host` covers single-device `nemotron_h`,
      `mistral3`, and `hunyuan-dense` on `cpu`, `ggml_cpu`, `ggml_cuda`, or `cuda`. Multi-GPU modes
      and other families (such as `qwen3.5` or `gpt-oss`) return HTTP 400 (`lora_not_applicable`).
    """

    base_model_id: str = "mistral3-instruct"
    v1_models_id: str = "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M"
    architecture: str = "mistral3"
    backend: str = "ggml_cuda"
    loras_dir: str = "/opt/hivemind/loras"
    multi_gpu: bool = False
    server_url: str = "http://127.0.0.1:8787"
    registered_bee_adapters: dict[str, str] = field(
        default_factory=lambda: {
            "hivemind/bee-stars": "bee-stars",
            "hivemind/bee-code": "bee-code",
            "hivemind/bee-math": "bee-math",
            "hivemind/bee-science": "bee-science",
            "hivemind/bee-cyber": "bee-cyber",
        }
    )

    def __post_init__(self) -> None:
        self.validate_lora_contract()

    def validate_lora_contract(self) -> None:
        if self.multi_gpu:
            raise ValueError(
                "TensorSharp.Server.Host text-LoRA is only supported on a single device; "
                "multi-GPU (--tp / --layer-split) returns HTTP 400 (lora_not_applicable)."
            )
        if self.architecture not in SUPPORTED_TENSORSHARP_TEXT_LORA_ARCHITECTURES:
            raise ValueError(
                f"Architecture {self.architecture!r} is not supported by TensorSharp text-LoRA. "
                f"Supported single-device families: {SUPPORTED_TENSORSHARP_TEXT_LORA_ARCHITECTURES}."
            )
        if self.backend not in SUPPORTED_TENSORSHARP_TEXT_LORA_BACKENDS:
            raise ValueError(
                f"Backend {self.backend!r} is not supported by TensorSharp text-LoRA. "
                f"Supported backends: {SUPPORTED_TENSORSHARP_TEXT_LORA_BACKENDS}."
            )
        if len(self.registered_bee_adapters) > 32:
            raise ValueError("TensorSharp.Server.Host supports at most 32 loaded LoRA adapters.")

    def build_server_lora_flags(self) -> list[str]:
        self.validate_lora_contract()
        return ["--loras-dir", self.loras_dir]

    def build_openai_chat_request(
        self,
        *,
        public_model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible `/v1/chat/completions` payload using `"lora": "<adapter_name>"`."""
        self.validate_lora_contract()
        adapter_name = self.registered_bee_adapters.get(public_model_id)
        payload: dict[str, Any] = {
            "model": self.v1_models_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if adapter_name:
            payload["lora"] = adapter_name
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
        """Build the `POST /v1/hivemind/producer/heartbeat` payload using the exact `/v1/models` ID."""
        self.validate_lora_contract()
        if not public_endpoint_url.startswith("https://"):
            raise ValueError("public_endpoint_url must be an encrypted https:// shield endpoint.")
        models = [
            {
                "id": f"hivemind/{self.base_model_id}",
                "local_name": self.v1_models_id,
                "loaded": True,
                "vram_mib": gpu_vram_mib,
            }
        ]
        for bee_id, adapter_name in sorted(self.registered_bee_adapters.items()):
            models.append(
                {
                    "id": bee_id,
                    "local_name": self.v1_models_id,
                    "lora": adapter_name,
                    "loaded": True,
                    "vram_mib": 128,
                }
            )
        return {
            "endpoint_url": public_endpoint_url,
            "execution_class": execution_class,
            "gpu_vram_mib": gpu_vram_mib,
            "cost_per_1k": cost_per_1k,
            "capacity_1k": capacity_1k,
            "engine": f"TensorSharp.Server.Host ({self.backend}, --host 127.0.0.1, --loras-dir)",
            "models": models,
        }


def build_zhongkaifu_collab_summary() -> dict[str, Any]:
    """Return a structured summary separating RTX PRO 6000 Blackwell reference benchmarks, estimates, and Tesla T4 measurements."""
    t4_gemma_estimate = plan_tensorsharp_moe_offload(
        "gemma4-26b-a4b",
        total_vram_mib=T4_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=T4_REQUIRED_FREE_VRAM_MIB,
        gpu_name="NVIDIA Tesla T4 (interpolated estimate from Blackwell reference)",
        compute_capability="compute_75",
    )
    rtx2070_gpt_oss_estimate = plan_tensorsharp_moe_offload(
        "gpt-oss-20b",
        total_vram_mib=RTX2070_TOTAL_VRAM_MIB,
        reserved_free_vram_mib=1_024,
        gpu_name="NVIDIA GeForce RTX 2070 (interpolated estimate from Blackwell reference)",
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
        "reference_benchmarks": {
            "muse_spark_816b_10x_a40": {
                "active_params_b": 42.0,
                "total_params_b": 816.0,
                "quantization": "UD-IQ3_XXS (~316 GiB GGUF)",
                "hardware": A40_CLUSTER_REFERENCE_HARDWARE,
                "layer_split_16k_prefill_tok_s": 1346.8,
                "llamacpp_16k_prefill_tok_s": 826.1,
                "tp10_single_decode_tok_s": 22.63,
                "llamacpp_single_decode_tok_s": 16.19,
                "tp10_4way_concurrent_decode_tok_s": 45.73,
            },
            "rtx_pro_6000_blackwell_moe_cpu_offload": {
                k: {
                    "reference_hardware": v.reference_hardware,
                    "quantization": v.quantization,
                    "gguf_artifact": v.gguf_artifact,
                    "v1_models_id": v.v1_models_id,
                    "pp8192_speedup_vs_llamacpp": v.prefill_speedup_vs_llamacpp,
                    "tg128_speedup_vs_llamacpp": v.decode_speedup_vs_llamacpp,
                    "tensorsharp_pp8192_tok_s": v.tensorsharp_pp8192_tok_s,
                    "tensorsharp_tg128_tok_s": v.tensorsharp_tg128_tok_s,
                }
                for k, v in TENSORSHARP_BENCHMARK_CATALOG.items()
            },
        },
        "tesla_t4_measured_single_model_validation": {
            "commit": TENSORSHARP_VALIDATED_COMMIT,
            "release": TENSORSHARP_VALIDATED_RELEASE,
            "hardware": T4_MEASURED_HARDWARE,
            "runs": {k: v for k, v in TENSORSHARP_T4_MEASURED_VALIDATION.items()},
        },
        "interpolated_hardware_estimates": {
            "is_interpolated_estimate": True,
            "tesla_t4_with_4300mib_headroom_guard": t4_gemma_estimate,
            "rtx_2070_8gb_prosumer": rtx2070_gpt_oss_estimate,
        },
        "collab_pillars": [
            "Pillar 1: Secure single-model TensorSharp.Server.Host producer on Tesla T4 (--host 127.0.0.1 + shield + exact /v1/models IDs)",
            "Pillar 2: Turing sm_75 (Tesla T4 15 GiB) Measured Single-Model Validation vs. RTX PRO 6000 Blackwell Reference Benchmarks",
            "Pillar 3: Multi-Node Layer-Split & Tensor Parallelism Bridge (TcpCommunicator v4 + HiveMind LayerStoRm PCIe/QUIC Nectar Cache)",
            "Pillar 4: Single-Device Text-LoRA Routing (--loras-dir + 'lora' field for nemotron_h, mistral3, hunyuan-dense)",
            "Pillar 5: Native C#/.NET Integration between TensorSharp & USS HiveMind Exoplanet Hunter Steam Fat Client (AppID 3133700)",
        ],
    }
