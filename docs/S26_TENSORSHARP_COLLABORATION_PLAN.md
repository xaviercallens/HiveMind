# Season 26 Co-Lab Plan: Project HiveMind $\times$ Zhongkai Fu (`@zhongkaifu` / `TensorSharp`)

- **Reddit Post Analyzed:** [`r/LovingOpenSourceAI/comments/1xcuesc/pushing_opensource_ai_further_running_an/`](https://www.reddit.com/r/LovingOpenSourceAI/comments/1xcuesc/pushing_opensource_ai_further_running_an/)
- **Contributor:** **Zhongkai Fu** (`@zhongkaifu`, `fuzhongkai@gmail.com`, [tensorsharp.ai](https://tensorsharp.ai))
- **GitHub Repository:** [`https://github.com/zhongkaifu/TensorSharp`](https://github.com/zhongkaifu/TensorSharp) (C# / .NET + Native CUDA 13 PTX `compute_75+` + GGML/Vulkan/MLX)
- **Companion Guide PR:** [`zhongkaifu/TensorSharp#308`](https://github.com/zhongkaifu/TensorSharp/pull/308)
- **HiveMind Integration Module:** [`src/hivemind/swarm/tensorsharp_collab_bridge.py`](../src/hivemind/swarm/tensorsharp_collab_bridge.py)
- **Live Tesla T4 Validation Telemetry:** [`artifacts/tensorsharp_t4_producer_validation.json`](../artifacts/tensorsharp_t4_producer_validation.json)

---

## 1. Measured Single-Model Producer Validation on NVIDIA Tesla T4 (`v2026.10.23`)

Per `@zhongkaifu`'s review on [`zhongkaifu/TensorSharp#308`](https://github.com/zhongkaifu/TensorSharp/pull/308#issuecomment-6526489885), we validated a reproducible single-model `TensorSharp.Server.Host` producer on our live **15,360 MiB NVIDIA Tesla T4 (`compute_75`)** (`socreateai-agora-hermes-node1`) with:

- **TensorSharp Commit / Release:** `a37448127f2fcd70079889c7050bdafb6b4bbfc7` (`v2026.10.23`, `tensorsharp-server-2026.10.23-linux-x64-cuda.tar.gz`)
- **Hardware / Driver / Toolkit:** 1x NVIDIA Tesla T4 (`15,360 MiB` GDDR6, Turing `compute_75`), Driver `580.178.04`, CUDA Driver `13.0` (host toolkit `CUDA 12.9` + bundled CUDA runtime), 8x vCPU Intel Xeon @ 2.20 GHz, `29 GiB` RAM
- **Security Binding:** `TensorSharp.Server.Host` bound to loopback (`--host 127.0.0.1 --port 11436 --no-webui --no-prefix-cache`, `MAX_CONTEXT=4096`) behind `hivemind_producer_shield.py` (`--upstream http://127.0.0.1:11436 --host 127.0.0.1 --port 8787 --allow-model <exact_v1_models_id>`) with `HIVEMIND_ENDPOINT_TOKEN`.

| `/v1/models` Exact ID | GGUF Artifact & SHA-256 | `--n-cpu-moe` | Idle VRAM | **Peak VRAM** | Shield Auth (`No Token` / `Bad Token` / `Alias` / `Valid`) | Streaming (`TTFT` / SSE Chunks) | Prefill (`tok/s`) | Streamed Decode (`tok/s`) | Non-Stream E2E (`tok/s`) |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `Qwen3.5-9B-Q4_K_M` | `Qwen3.5-9B-Q4_K_M.gguf` (`5.29 GiB`, `03b74727...52b7e8`) | None (GPU-resident) | `5,420 MiB` | **`6,062 MiB`** | `401` / `401` / `403` / `200` | `3,451.8 ms` / `97` chunks | **`952.8 tok/s`** | **`32.8 tok/s`** | `14.6 tok/s` |
| `gpt-oss-20b-MXFP4` | `gpt-oss-20b-MXFP4.gguf` (`12.85 GiB`, `8ecb5d4a...de716`) | `18` (of 24 layers) | `6,104 MiB` | **`7,042 MiB`** | `401` / `401` / `403` / `200` | `4,283.7 ms` / `88` chunks | **`624.9 tok/s`** | **`13.0 tok/s`** | `8.1 tok/s` |
| `gpt-oss-20b-MXFP4` | `gpt-oss-20b-MXFP4.gguf` (`12.85 GiB`, `8ecb5d4a...de716`) | `24` (of 24 layers) | `3,662 MiB` | **`4,610 MiB`** | `401` / `401` / `403` / `200` | `4,730.0 ms` / `88` chunks | **`566.0 tok/s`** | **`10.8 tok/s`** | `6.9 tok/s` |

---

## 2. Reference Benchmarks (`docs/moe_cpu_offload_benchmark.md` & `r/LovingOpenSourceAI`) vs. Interpolated Estimates

### A. 10x NVIDIA A40 Reference Benchmark (`Muse Spark 816B UD-IQ3_XXS`)

In [`r/LovingOpenSourceAI/comments/1xcuesc`](https://www.reddit.com/r/LovingOpenSourceAI/comments/1xcuesc/pushing_opensource_ai_further_running_an/), **Zhongkai Fu** benchmarked **Muse Spark 816B** (42B active, `UD-IQ3_XXS`, ~316 GiB) across **10x NVIDIA A40 GPUs** (46 GB each, PCIe 4.0 **without P2P** enabled):

| Parallelism Mode (`10x NVIDIA A40`, PCIe 4.0, No P2P) | Workload | `TensorSharp` | `llama.cpp` | Speedup |
| --- | --- | ---: | ---: | ---: |
| **Layer Splitting (`--layer-split 10`)** | 16K Prompt Prefill (`pp16384`) | **1,346.8 tok/s** | 826.1 tok/s | **1.63x** |
| **Layer Splitting (`--layer-split 10`)** | Single-Stream Decode | **16.7 tok/s** | ~16.7 tok/s | **1.00x** |
| **Tensor Parallelism (`--tp 10`)** | Single-Request Decode | **22.63 tok/s** | 16.19 tok/s | **1.40x** |
| **Tensor Parallelism (`--tp 10`)** | 4-Way Concurrent Decode | **45.73 tok/s** | — | **2.82x vs 1-req llama** |

### B. 2x NVIDIA RTX PRO 6000 Blackwell Reference Benchmarks (`docs/moe_cpu_offload_benchmark.md`)

The reference `--n-cpu-moe` results in [`docs/moe_cpu_offload_benchmark.md`](https://github.com/zhongkaifu/TensorSharp/blob/main/docs/moe_cpu_offload_benchmark.md) were recorded on **2x NVIDIA RTX PRO 6000 Blackwell Server Edition (`97,887 MiB` each, PCIe 5.0 x16, 2x Intel Xeon 6952P, `1,511 GiB` RAM, CUDA 12.8)**:

| Model | Quantization & GGUF (`docs/moe_cpu_offload_benchmark.md`) | `--n-cpu-moe` Offload | Reference VRAM (`Blackwell`) | `TensorSharp` `pp8192` | `llama.cpp` `pp8192` | Prefill Speedup | `TensorSharp` `tg128` | `llama.cpp` `tg128` | Decode Speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS` (`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`) | `16 of 30` | `~13.7 GiB` | **4,888 tok/s** | 854 tok/s | **5.72x** | **54.5 tok/s** | 21.9 tok/s | **2.49x** |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS` (`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`) | `30 of 30` | `10.8 GiB` | **3,072 tok/s** | 495 tok/s | **6.21x** | **39.7 tok/s** | 12.9 tok/s | **3.08x** |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS` (`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`) | `24 of 48` | `~17.3 GiB` | **5,259 tok/s** | 484 tok/s | **10.87x** | **52.3 tok/s** | 15.8 tok/s | **3.31x** |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS` (`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`) | `48 of 48` | `~12.9 GiB` | **3,709 tok/s** | 457 tok/s | **8.12x** | **38.6 tok/s** | 10.2 tok/s | **3.78x** |
| **GPT-OSS 20B** | `Q8_0 / MXFP4` (`gpt-oss-20b-MXFP4.gguf`) | `12 of 24` | `~6.8 GiB` | **6,394 tok/s** | 1,188 tok/s | **5.38x** | **51.7 tok/s** | 18.3 tok/s | **2.83x** |
| **GPT-OSS 20B** | `Q8_0 / MXFP4` (`gpt-oss-20b-MXFP4.gguf`) | `24 of 24` | `~2.2 GiB` | **3,798 tok/s** | 548 tok/s | **6.93x** | **27.7 tok/s** | 9.4 tok/s | **2.95x** |

> **Estimates vs. Measurements:** `plan_tensorsharp_moe_offload()` in [`src/hivemind/swarm/tensorsharp_collab_bridge.py`](../src/hivemind/swarm/tensorsharp_collab_bridge.py) marks all interpolated VRAM and throughput figures with `is_interpolated_estimate=True` and `reference_hardware=BLACKWELL_REFERENCE_HARDWARE`, keeping them strictly separate from the live Tesla T4 measurements in `TENSORSHARP_T4_MEASURED_VALIDATION`.

---

## 3. Text-LoRA API Contract (`--loras-dir` & `"lora"`)

Per `TensorSharp.Server.Host/API_EXAMPLES.md`:

- Adapters are registered at startup via `--loras-dir <dir>` (or `TS_LORAS_DIR`) by filename stem, or at runtime via `POST /v1/loras` (`{"name": "<stem>", "path": "<file>", "scale": 1.0}`, up to 32 adapters).
- Requests select a registered adapter with `"lora": "<adapter_name>"` (not `"lora_adapter"` with a file path) alongside the exact `/v1/models` `id`.
- Current text-LoRA support covers single-device `nemotron_h`, `mistral3`, and `hunyuan-dense` on `cpu`, `ggml_cpu`, `ggml_cuda`, or `cuda`. `TensorSharpBeeLoRARouter` enforces this contract and rejects unsupported model families (such as `qwen3.5`) or multi-GPU configurations.
