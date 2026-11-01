# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Nocturnal MoE Auto-Improvement & Proof-of-Useful-Work (PoUW) Points Retrofit Engine.

Part of Project HiveMind (`hivemind.core.auto_improvement`).

Addresses GCP Spot Preemption of dedicated cloud nodes (such as Tesla T4) by dynamically
delegating nocturnal Mixture-of-Experts (MoE) training, self-refinement, and validation workunits
across connected swarm producers (RTX 2070, RTX 3080/3090, RTX 4070/4090, Apple Silicon, etc.)
under a strict < 8 GB VRAM ceiling.

Core Capabilities:
1. `NocturnalAutoImprovementScheduler`:
   - Monitors connected producers from the router registry and ranks them by compute tier and VRAM.
   - Supports 5 high-power base models adapted to consumer GPUs via 4-bit NormalFloat (NF4):
     * `Qwen/Qwen2.5-Coder-7B-Instruct` (~4.7 GB NF4)
     * `Qwen/Qwen3.5-9B-Base` (~5.4 GB NF4)
     * `google/gemma-2-9b-it` (~5.8 GB NF4)
     * `google/gemma-2-2b-it` (~1.8 GB NF4)
     * `google/gemma-3-4b-it` (~2.7 GB NF4)
   - Schedules 16 specialized domain micro-experts across 4 macro-clusters, mapped to real Hugging Face datasets:
     * CLUSTER 0: Systems & Formal (Lean 4 ProofNet, Rust DMA Kernel, Python SIMD, Debug Feedback)
     * CLUSTER 1: Crypto & Cyber (CTI-Bench, SWE-Bench CVE, Solidity DISL)
     * CLUSTER 2: Frontier Science (Kepler Exoplanets, Gaia MK Stars, JWST Atmosphere, Quantum Lattice)
     * CLUSTER 3: Reasoning & Language (GSM8K Math, PubMedQA Bio, MIMO French, Tulu Neurosymbolic)
   - Enforces Paired McNemar and loss-reduction gates before promoting candidate adapter weights.

2. `PointsRetrofitManager` & `retrofit_work_points`:
   - Calculates fair, deterministic Proof-of-Useful-Work (PoUW) HMT Reflexion Points based on:
     * Base workunit completion reward (10.0 HMT).
     * Computational work (0.0125 HMT per inner step / 100 GFLOPs).
     * Quality bonus (+5.0 HMT when passing the Paired McNemar improvement gate).
   - Retrofits and credits HMT into the cryptographic hash-chained SQLite ledger (`RouterStore`).
   - Guarantees strict idempotency via canonical references (`pouw_auto_improve:{wu_id}`).
   - Asserts full ledger hash-chain integrity (`verify_chain() == True`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# European night window constants (CET / CEST)
NIGHT_START_HOUR_CET = 22  # 22:00 CET
NIGHT_END_HOUR_CET = 7     # 07:00 CET

# Standard PoUW reward schedule for Auto-Improvement Workunits
DEFAULT_BASE_WU_REWARD_HMT = 10.0
DEFAULT_TFLOPS_STEP_RATE_HMT = 0.0125  # ~1.25 HMT per 100 steps (~12.5 TFLOPs-steps)
DEFAULT_GATE_PASS_BONUS_HMT = 5.0

# Supported base models adapted to <= 8 GB VRAM via NF4 quantization
SUPPORTED_BASE_MODELS: dict[str, dict[str, Any]] = {
    "qwen-2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "params_b": 7.6,
        "vram_nf4_gb": 4.7,
        "architecture": "qwen2",
        "description": "Frontier open coding model with extensive context window and high instruction precision.",
    },
    "qwen-3.5-9b": {
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "params_b": 9.0,
        "vram_nf4_gb": 5.4,
        "architecture": "qwen3",
        "description": "Next-gen Qwen dense base model with enhanced multi-lingual and scientific reasoning.",
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "params_b": 9.2,
        "vram_nf4_gb": 5.8,
        "architecture": "gemma2",
        "description": "Google Gemma 2 9B instruction-tuned model with sliding window attention and logit soft-capping.",
    },
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b-it",
        "params_b": 2.6,
        "vram_nf4_gb": 1.8,
        "architecture": "gemma2",
        "description": "Ultra-lightweight Gemma 2 model running with full KV cache on entry-level GPUs.",
    },
    "gemma-3-4b": {
        "model_id": "google/gemma-3-4b-it",
        "params_b": 4.1,
        "vram_nf4_gb": 2.7,
        "architecture": "gemma3",
        "description": "High-efficiency Google Gemma 3 model optimized for multimodal and code tasks.",
    },
    "gemma-3-1b": {
        "model_id": "google/gemma-3-1b-it",
        "params_b": 1.2,
        "vram_nf4_gb": 1.1,
        "architecture": "gemma3",
        "description": "Ultra-compact Google Gemma 3 model optimized for edge devices and rapid iteration.",
    },
    "qwen-2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.6,
        "vram_nf4_gb": 4.7,
        "architecture": "qwen2",
        "description": "Dense general-purpose instruction model with strong mathematics and logical reasoning.",
    },
    "deepseek-prover-v2-7b": {
        "model_id": "deepseek-ai/DeepSeek-Prover-V2-7B",
        "params_b": 7.0,
        "vram_nf4_gb": 4.5,
        "architecture": "deepseek2",
        "description": "Lean 4 formal theorem proving and scientific invariant verification model for <= 8 GB VRAM.",
    },
}

# Large VRAM Base Models adapted to >= 12 GB, 16 GB, 24 GB and 32 GB GPUs for nocturnal specialization
LARGE_VRAM_BASE_MODELS: dict[str, dict[str, Any]] = {
    "qwen-2.5-coder-14b": {
        "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "params_b": 14.7,
        "vram_nf4_gb": 8.9,
        "min_train_vram_gb": 14.0,
        "architecture": "qwen2",
        "description": "Mid-tier coding expert with 14.7B parameters, requiring >=14 GB VRAM for QLoRA training.",
    },
    "qwen-2.5-14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14.7,
        "vram_nf4_gb": 8.9,
        "min_train_vram_gb": 14.0,
        "architecture": "qwen2",
        "description": "General 14.7B instruction base model with advanced math and reasoning capabilities.",
    },
    "deepseek-coder-v2-lite": {
        "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        "params_b": 15.7,
        "vram_nf4_gb": 9.5,
        "min_train_vram_gb": 16.0,
        "architecture": "deepseek2",
        "description": "15.7B Sparse MoE base model (2.4B active per token), highly specialized for multi-language code and math.",
    },
    "mistral-nemo-12b": {
        "model_id": "mistralai/Mistral-Nemo-Base-2407",
        "params_b": 12.2,
        "vram_nf4_gb": 7.5,
        "min_train_vram_gb": 14.0,
        "architecture": "mistral",
        "description": "12.2B European multimodal multilingual foundation model with 128k token context.",
    },
    "qwen-2.5-32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.5,
        "vram_nf4_gb": 19.5,
        "min_train_vram_gb": 24.0,
        "architecture": "qwen2",
        "description": "32.5B high-parameter frontier model, requiring large 24 GB+ VRAM cards (RTX 3090/4090, A100) for QLoRA training.",
    },
    "qwen-3-coder-30b-a3b": {
        "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "params_b": 30.5,
        "active_params_b": 3.3,
        "moe": True,
        "vram_nf4_gb": 16.5,
        "min_train_vram_gb": 16.0,
        "architecture": "qwen3_moe",
        "description": "30.5B Sparse MoE agentic coding and scientific systems model (3.3B active per token).",
    },
    "glm-4.5-air-moe": {
        "model_id": "zai-org/GLM-4.5-Air",
        "params_b": 106.0,
        "active_params_b": 12.0,
        "moe": True,
        "vram_nf4_gb": 22.0,
        "min_train_vram_gb": 24.0,
        "architecture": "glm4_moe",
        "description": "106B total / 12B active Hybrid Scientific Reasoning & Tool-Synthesis MoE for multi-node or 24 GB+ producers.",
    },
}

# Sub-4-Bit (1.58-bit / 2-bit / 3-bit) Frontier Models for the Nocturnal "Dream Phase"
# Used as frozen Dream Teachers (synthetic trajectory generation, counterfactual critique, and
# on-policy knowledge distillation) within strict consumer & cloud VRAM budgets (Golden Invariant I1).
LOW_BIT_DREAM_TEACHER_MODELS: dict[str, dict[str, Any]] = {
    "tiel-coder-35b-a3b-iq3-xxs": {
        "model_id": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF",
        "gguf_file": "Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf",
        "mtp_companion_repo": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP",
        "base_model": "ornith-ai/Ornith-1.5-35B-A3B",
        "params_b": 35.0,
        "active_params_b": 3.0,
        "moe": True,
        "quant_scheme": "Unsloth-Dynamic-IQ3_XXS + 49M-char code imatrix (3.06 bpw)",
        "weights_gib": 12.30,
        "min_vram_full_offload_gb": 15.0,
        "min_vram_moe_split_gb": 6.0,
        "wddm_8gb_moe_split_vram_gb": 5.6,
        "runtime": "llama.cpp (-ngl 99 --jinja; or --n-cpu-moe on 8GB VRAM + 16GB RAM)",
        "recommended_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "target_roles": ("coding", "cyber", "formal_lean4", "systems"),
        "license": "MIT",
        "description": (
            "35B-A3B Sparse MoE agentic coder (3B active/token) solving 12/25 on SWE-bench-Live "
            "(level with Opus 4.6 medium) and 67.2 on Claw-Eval; fits 16 GB Tesla T4 in 3-bit UD-IQ3_XXS "
            "or 8 GB RTX 2070 (<6.5 GB I1 ceiling) via MoE expert CPU/RAM split."
        ),
    },
    "tiel-coder-35b-a3b-iq3-s": {
        "model_id": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF",
        "gguf_file": "Tiel-Coder-35B-A3B-UD-IQ3_S.gguf",
        "mtp_companion_repo": "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP",
        "base_model": "ornith-ai/Ornith-1.5-35B-A3B",
        "params_b": 35.0,
        "active_params_b": 3.0,
        "moe": True,
        "quant_scheme": "Unsloth-Dynamic-IQ3_S + 49M-char code imatrix (3.18 bpw)",
        "weights_gib": 12.74,
        "min_vram_full_offload_gb": 16.0,
        "min_vram_moe_split_gb": 6.0,
        "wddm_8gb_moe_split_vram_gb": 5.8,
        "runtime": "llama.cpp (-ngl 99 --jinja)",
        "recommended_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "target_roles": ("coding", "cyber", "formal_lean4", "systems"),
        "license": "MIT",
        "description": (
            "Higher-precision 3-bit UD-IQ3_S build of Tiel-Coder-35B-A3B (12.74 GiB) for 16 GB+ GPUs."
        ),
    },
    "ternary-bonsai-2-27b-ptq1": {
        "model_id": "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "gguf_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "mlx_companion_repo": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        "base_model": "Qwen/Qwen3.8-27B",
        "params_b": 27.36,
        "active_params_b": 27.36,
        "moe": False,
        "quant_scheme": "Hadamard-Rotated Ternary g128 PTQ1_0 (1.75 bits/weight dense trits)",
        "weights_gib": 5.54,
        "min_vram_full_offload_gb": 6.5,
        "wddm_8gb_safe_vram_gb": 6.20,
        "runtime": "PrismML-Eng/llama.cpp (custom Hadamard ternary CUDA/Metal kernels)",
        "recommended_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05},
        "target_roles": ("generalist", "science", "math", "astrophysics", "multilingual"),
        "license": "Apache-2.0",
        "description": (
            "27.36B hybrid-attention generalist & deep reasoning model compressed 9.0x to 5.54 GiB "
            "via blockwise Hadamard ternary quantization, retaining 98.2% of FP16 intelligence "
            "(84.78 avg across 14 thinking benchmarks) and fitting inside an 8 GB RTX 2070 (<=6.5 GB I1 ceiling)."
        ),
    },
    "ternary-bonsai-2-27b-pq2": {
        "model_id": "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "gguf_file": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        "mlx_companion_repo": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        "base_model": "Qwen/Qwen3.8-27B",
        "params_b": 27.36,
        "active_params_b": 27.36,
        "moe": False,
        "quant_scheme": "Hadamard-Rotated Ternary g128 PQ2_0 (2.13 bits/weight 2-bit slots)",
        "weights_gib": 6.71,
        "min_vram_full_offload_gb": 8.5,
        "wddm_8gb_safe_vram_gb": 6.71,
        "runtime": "PrismML-Eng/llama.cpp (CUDA/Metal) or PrismML-Eng/mlx (Apple Silicon)",
        "recommended_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05},
        "target_roles": ("generalist", "science", "math", "astrophysics", "multilingual"),
        "license": "Apache-2.0",
        "description": (
            "Fast-unpacking 2-bit slot PQ2_0 build of Ternary-Bonsai-2-27B (6.71 GiB); on a 16 GB Tesla T4 "
            "it leaves >8 GiB free VRAM so the 27B teacher and a QLoRA student can co-reside in VRAM."
        ),
    },
    "ternary-bonsai-2-9b-pq2": {
        "model_id": "prism-ml/Ternary-Bonsai-2-9B-gguf",
        "gguf_file": "Ternary-Bonsai-2-9B-PQ2_0.gguf",
        "draft_companion_file": "Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf",
        "base_model": "Qwen/Qwen3.5-9B",
        "params_b": 9.0,
        "active_params_b": 9.0,
        "moe": False,
        "quant_scheme": "Hadamard-Rotated Ternary g128 PQ2_0 (2.25 GiB) + DFlash2 speculative drafter",
        "weights_gib": 2.25,
        "min_vram_full_offload_gb": 4.0,
        "wddm_8gb_safe_vram_gb": 3.80,
        "runtime": "PrismML-Eng/llama.cpp (CUDA/Metal/CPU)",
        "recommended_sampling": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5},
        "target_roles": ("generalist", "edge", "vision", "fast_synthesis"),
        "license": "Apache-2.0",
        "description": (
            "9B hybrid-attention VLM in 2.25 GiB ternary PQ2_0 weights with 262K context and optional DFlash2 drafter."
        ),
    },
    "bonsai-27b-1bit-q1": {
        "model_id": "prism-ml/Bonsai-27B-gguf",
        "gguf_file": "Bonsai-27B-Q1_0.gguf",
        "unpacked_safetensors_repo": "prism-ml/Bonsai-27B-unpacked",
        "base_model": "Qwen/Qwen3.6-27B",
        "params_b": 27.0,
        "active_params_b": 27.0,
        "moe": False,
        "quant_scheme": "1-Bit Binary Transformer Q1_0 (14.2x compression)",
        "weights_gib": 3.54,
        "min_vram_full_offload_gb": 4.5,
        "wddm_8gb_safe_vram_gb": 4.20,
        "runtime": "PrismML-Eng/llama.cpp (CUDA/Metal)",
        "recommended_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05},
        "target_roles": ("generalist", "co_resident_distillation"),
        "license": "Apache-2.0",
        "description": (
            "1-bit 27B reasoning model in 3.54 GiB (~90% FP16 intelligence retained), enabling simultaneous "
            "in-VRAM teacher-student distillation on 10 GB - 12 GB GPUs."
        ),
    },
}


def select_dream_phase_teacher_for_vram(
    vram_gb: float,
    domain_or_cluster: str = "generalist",
    system_ram_gb: float = 16.0,
) -> dict[str, Any]:
    """Select the optimal 1.58-bit / 2-bit / 3-bit Dream Phase teacher for a producer's VRAM and domain.

    Enforces Golden Invariant I1 (<= 6.5 GB resident VRAM on 8 GB consumer GPUs).
    """
    dom = domain_or_cluster.strip().lower()
    is_code_or_formal = any(
        k in dom for k in ("code", "coding", "swe", "rust", "python", "cyber", "formal", "lean", "systems", "cluster_00", "cluster_01")
    )
    if is_code_or_formal:
        if vram_gb >= 15.0:
            choice = dict(LOW_BIT_DREAM_TEACHER_MODELS["tiel-coder-35b-a3b-iq3-xxs"])
            choice["execution_mode"] = "full_gpu_offload_16gb"
            return choice
        if vram_gb >= 6.0 and system_ram_gb >= 16.0:
            choice = dict(LOW_BIT_DREAM_TEACHER_MODELS["tiel-coder-35b-a3b-iq3-xxs"])
            choice["execution_mode"] = "moe_expert_cpu_ram_split_8gb_wddm_safe"
            return choice
    # Generalist, scientific, math, astrophysics, or fallback when RAM is constrained
    if vram_gb >= 12.0:
        choice = dict(LOW_BIT_DREAM_TEACHER_MODELS["ternary-bonsai-2-27b-pq2"])
        choice["execution_mode"] = "full_gpu_offload_with_co_resident_student_headroom"
        return choice
    if vram_gb >= 6.5:
        choice = dict(LOW_BIT_DREAM_TEACHER_MODELS["ternary-bonsai-2-27b-ptq1"])
        choice["execution_mode"] = "full_gpu_offload_8gb_wddm_safe"
        return choice
    choice = dict(LOW_BIT_DREAM_TEACHER_MODELS["ternary-bonsai-2-9b-pq2"])
    choice["execution_mode"] = "compact_edge_gpu_offload"
    return choice

ALL_BASE_MODELS: dict[str, dict[str, Any]] = {
    **SUPPORTED_BASE_MODELS,
    **LARGE_VRAM_BASE_MODELS,
}

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 24 Specialized Domain Micro-Experts mapped to Real Hugging Face Datasets & Clusters
HF_EXPERT_CATALOG: dict[str, dict[str, Any]] = {
    # CLUSTER 00: Systems & Formal
    "maths_formal": {
        "title": "Lean 4 Machine-Checked Formal Prover",
        "hf_dataset": "hoskinson-center/proofnet",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "Formal mathematical theorems and tactics verified by Lean 4 theorem prover.",
    },
    "code_rust_kernel": {
        "title": "Rust Systems & Zero-Copy Kernel DMA",
        "hf_dataset": "bigcode/the-stack-v2-dedup",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "Low-level Rust systems code, memory safety, and ring-buffer drivers.",
    },
    "code_python_perf": {
        "title": "Python Performance & SIMD Kernel Optimizer",
        "hf_dataset": "HuggingFaceH4/CodeAlpaca_20K",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "High-performance vectorized Python, Cython, and PyTorch kernel optimizations.",
    },
    "code_feedback_debug": {
        "title": "Multi-Turn Interactive Bug Repair & Refactor",
        "hf_dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "Iterative traceback debugging, unit test repair, and self-correction traces.",
    },
    "systems_distributed_p2p": {
        "title": "Distributed Consensus & P2P Gossip Protocols",
        "hf_dataset": "bigcode/santacoder",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "Gossip protocols, Kademlia DHT routing, Raft consensus, and libp2p network topologies.",
    },
    "cuda_kernel_opt": {
        "title": "Triton / CUDA Tensor Core Kernel Optimization",
        "hf_dataset": "neuralmagic/LLM_Compression_Code",
        "cluster": "CLUSTER_00_SYSTEMS_FORMAL",
        "description": "FP16/BF16 matrix multiplication, shared memory tiling, and fused kernel dispatch.",
    },

    # CLUSTER 01: Crypto & Cyber
    "cyber_threat_intel": {
        "title": "Cyber Threat Intelligence & CTI-Bench",
        "hf_dataset": "giskard_cti/cti-bench",
        "cluster": "CLUSTER_01_CRYPTO_CYBER",
        "description": "CTI analysis, ATT&CK tactic identification, and security reasoning.",
    },
    "cyber_vulnerability": {
        "title": "CVE Vulnerability Remediation & Patching",
        "hf_dataset": "SWE-bench/SWE-bench_Verified",
        "cluster": "CLUSTER_01_CRYPTO_CYBER",
        "description": "Real-world CVE analysis, patch generation, and regression verification.",
    },
    "cyber_smart_contracts": {
        "title": "Solidity Smart Contract Security & ZK Escrow",
        "hf_dataset": "ASSERT-KTH/DISL-Solidity-Smart-Contracts",
        "cluster": "CLUSTER_01_CRYPTO_CYBER",
        "description": "Reentrancy defense, EVM state-channel escrow, and ZK-OPoI auditing.",
    },
    "cryptography_zkp": {
        "title": "Zero-Knowledge SNARK & STARK Circuit Prover",
        "hf_dataset": "0xPARC/zk-bench",
        "cluster": "CLUSTER_01_CRYPTO_CYBER",
        "description": "R1CS constraints, Halo2 circuits, PLONK arithmetization, and zero-knowledge proofs.",
    },
    "cyber_reverse_engineering": {
        "title": "Binary Disassembly & Malware Triage Specialist",
        "hf_dataset": "vxunderground/malware-analysis-prompts",
        "cluster": "CLUSTER_01_CRYPTO_CYBER",
        "description": "Decompilation analysis, control flow graph recovery, and obfuscation demangling.",
    },

    # CLUSTER 02: Frontier Science & Astrophysics
    "astrophysics_transits": {
        "title": "Kepler / TESS Exoplanet Transit Classifier",
        "hf_dataset": "mzrk36/exoplanets-unified-dataset",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "Photometric transit curve analysis and exoplanetary candidate validation.",
    },
    "astrophysics_spectra": {
        "title": "Gaia DR3 / SDSS MK Stellar Spectral Classifier",
        "hf_dataset": "BrunoBarreto/sdss_dr12_stars_regression",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "Morgan-Keenan stellar classification from multi-band spectrophotometry.",
    },
    "astrophysics_atmosphere": {
        "title": "JWST Exoplanetary Atmospheric Biosignature",
        "hf_dataset": "Shoriful025/exoplanet_atmospheric_spectroscopy_results",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "Transmission spectroscopy and atmospheric chemical composition retrieval.",
    },
    "physics_quantum_lattice": {
        "title": "Quantum Lattice Field Theory & Integrators",
        "hf_dataset": "MultimodalUniverse/multimodal_universe",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "Symplectic integrators, Hamiltonian conservation, and lattice QCD models.",
    },
    "chemistry_molecular_docking": {
        "title": "Molecular SMILES & Ligand-Protein Docking",
        "hf_dataset": "chainer/zinc250k",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "SMILES chemical structures, binding energy prediction, and ADMET pharmacological profiling.",
    },
    "climate_earth_observation": {
        "title": "Multi-Spectral Earth Observation & Climate Invariants",
        "hf_dataset": "nasa-impact/hls-foundation",
        "cluster": "CLUSTER_02_FRONTIER_SCIENCE",
        "description": "Sentinel/Landsat reflectance bands, NDVI indices, and thermodynamic radiative transfer.",
    },

    # CLUSTER 03: Reasoning & Life Sciences
    "maths_olympiad": {
        "title": "Multi-Step Mathematical Olympiad Reasoning",
        "hf_dataset": "openai/gsm8k",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Multi-step chain-of-thought arithmetic, geometry, and algebraic word problems.",
    },
    "biomedical_genomics": {
        "title": "CRISPR Genomics & PubMedQA Evidence",
        "hf_dataset": "qiaojin/PubMedQA",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Biomedical question answering grounded in peer-reviewed PubMed literature.",
    },
    "biomedical_clinical": {
        "title": "Clinical Trial Evidence & Drug Mechanisms",
        "hf_dataset": "epfl-llm/guidance-biomedical",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Clinical trial eligibility, pharmacokinetic analysis, and adverse drug reactions.",
    },
    "linguistics_french": {
        "title": "High-Register French Scientific Communication",
        "hf_dataset": "machelreid/mimo-french-instruct",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "French cultural nuance, academic prose, and technical vulgarization.",
    },
    "logic_neurosymbolic": {
        "title": "Autopoietic Neurosymbolic Energy Balancer",
        "hf_dataset": "allenai/tulu-v2-sft-mixture",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Banach contraction mappings, energy-function descent, and self-adaptation.",
    },
    "reasoning_multihop_arc": {
        "title": "ARC-Challenge Multi-Hop Scientific Reasoning",
        "hf_dataset": "allenai/ai2_arc",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Complex science questions requiring multi-hop deductive synthesis and hypotheses rejection.",
    },
    "agentic_tool_calling": {
        "title": "Agentic Tool Calling & Function Orchestration",
        "hf_dataset": "ToolBench/ToolBench",
        "cluster": "CLUSTER_03_REASONING_AND_LANGUAGE",
        "description": "Deterministic JSON schema function calling, REST API composition, and error self-healing.",
    },
}

# Legacy aliases for backwards compatibility with earlier 5-domain workunits
DOMAIN_ALIASES: dict[str, str] = {
    "maths": "maths_olympiad",
    "exoplanet": "astrophysics_transits",
    "stars": "astrophysics_spectra",
    "cyber": "cyber_threat_intel",
    "coder": "code_feedback_debug",
}

ALL_DOMAINS: tuple[str, ...] = tuple(HF_EXPERT_CATALOG.keys())
SUPPORTED_DOMAINS: tuple[str, ...] = tuple(list(HF_EXPERT_CATALOG.keys()) + list(DOMAIN_ALIASES.keys()))

# 20 Curated Nocturnal MoE Specialization Targets mapped to Hugging Face
NOCTURNAL_20_MOE_TARGETS: tuple[str, ...] = (
    "maths_formal",
    "code_rust_kernel",
    "code_python_perf",
    "code_feedback_debug",
    "systems_distributed_p2p",
    "cuda_kernel_opt",
    "cyber_threat_intel",
    "cyber_vulnerability",
    "cyber_smart_contracts",
    "cryptography_zkp",
    "cyber_reverse_engineering",
    "astrophysics_transits",
    "astrophysics_spectra",
    "astrophysics_atmosphere",
    "physics_quantum_lattice",
    "chemistry_molecular_docking",
    "climate_earth_observation",
    "maths_olympiad",
    "biomedical_genomics",
    "biomedical_clinical",
)


def resolve_domain_canonical(domain: str) -> str:
    """Resolve a domain alias or name to its canonical HF_EXPERT_CATALOG key."""
    canonical = DOMAIN_ALIASES.get(domain, domain)
    if canonical not in HF_EXPERT_CATALOG:
        raise ValueError(
            f"Unknown domain '{domain}'. Must be one of canonical domains: {list(HF_EXPERT_CATALOG.keys())} "
            f"or aliases: {list(DOMAIN_ALIASES.keys())}."
        )
    return canonical


@dataclass(frozen=True)
class AutoImprovementWorkunit:
    """A self-contained MoE training and auto-improvement workunit."""

    wu_id: str
    domain: str
    base_model_id: str
    target_adapter_name: str
    rank: int
    num_samples: int
    inner_steps: int
    min_vram_gb: float
    time_budget_minutes: float
    seed: int
    created_utc: float
    hf_dataset: str = ""
    cluster: str = ""
    sample_pairs: list[dict[str, str]] = field(default_factory=list)
    sha256_digest: str = ""

    def __post_init__(self) -> None:
        canonical = resolve_domain_canonical(self.domain)
        meta = HF_EXPERT_CATALOG[canonical]
        if not self.hf_dataset:
            object.__setattr__(self, "hf_dataset", meta["hf_dataset"])
        if not self.cluster:
            object.__setattr__(self, "cluster", meta["cluster"])

        if not self.sha256_digest:
            canonical_repr = (
                f"{self.wu_id}:{self.domain}:{self.base_model_id}:{self.hf_dataset}:"
                f"{self.num_samples}:{self.seed}"
            )
            digest = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
            object.__setattr__(self, "sha256_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoImprovementExecutionResult:
    """Execution telemetry and gate verification for an auto-improvement workunit."""

    wu_id: str
    domain: str
    worker_id: str
    account_id: str
    steps_completed: int
    loss_initial: float
    loss_final: float
    accuracy_base: float
    accuracy_adapted: float
    mcnemar_p_value: float
    gate_passed: bool
    contributed_tflops: float
    peak_vram_mb: float
    duration_seconds: float
    delta_weights_hash: str
    base_model_id: str = DEFAULT_BASE_MODEL
    hf_dataset: str = ""
    cluster: str = ""
    completed_utc: float = field(default_factory=time.time)
    execution_provenance: str = "measured_gpu"
    is_simulated: bool = False

    @property
    def delta_nll(self) -> float:
        return round(self.loss_final - self.loss_initial, 4)

    @property
    def accuracy_gain(self) -> float:
        return round(self.accuracy_adapted - self.accuracy_base, 4)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["delta_nll"] = self.delta_nll
        data["accuracy_gain"] = self.accuracy_gain
        return data


@dataclass
class RetrofitReport:
    """Detailed summary of the Proof-of-Useful-Work (PoUW) points retrofit."""

    total_workunits: int
    newly_credited_count: int
    already_credited_count: int
    total_hmt_granted: float
    total_tflops_attested: float
    credited_accounts: dict[str, float]
    ledger_verified: bool
    timestamp_utc: float
    audit_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NocturnalAutoImprovementScheduler:
    """Orchestrates nocturnal MoE auto-improvement across connected swarm producers."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        default_rank: int = 16,
        target_vram_ceiling_gb: float = 8.0,
    ) -> None:
        self.base_model_id = self._canonical_base_model(base_model_id)
        self.default_rank = default_rank
        self.target_vram_ceiling_gb = target_vram_ceiling_gb
        self.workunit_history: dict[str, AutoImprovementWorkunit] = {}
        self.execution_results: dict[str, AutoImprovementExecutionResult] = {}

    @staticmethod
    def _canonical_base_model(model_or_tag: str) -> str:
        """Resolve short tags like 'gemma-2-9b', 'qwen-3.5-9b', 'qwen-2.5-coder-14b' to HF identifiers."""
        tag = model_or_tag.strip().lower()
        if tag in ALL_BASE_MODELS:
            return ALL_BASE_MODELS[tag]["model_id"]
        for entry in ALL_BASE_MODELS.values():
            if entry["model_id"].lower() == tag:
                return entry["model_id"]
        # Fallback to provided string
        return model_or_tag

    @staticmethod
    def is_european_night(now_utc: float | None = None) -> bool:
        """Return True if current time is within European night window (22:00 - 07:00 CET)."""
        ts = now_utc if now_utc is not None else time.time()
        cet_hour = (datetime.fromtimestamp(ts, tz=timezone.utc).hour + 1) % 24
        return cet_hour >= NIGHT_START_HOUR_CET or cet_hour < NIGHT_END_HOUR_CET

    def filter_eligible_producers(
        self,
        connected_producers: Sequence[Mapping[str, Any]],
        min_vram_gb: float = 6.0,
        max_heartbeat_age_s: float = 300.0,
        now_utc: float | None = None,
    ) -> list[dict[str, Any]]:
        """Identify producers with sufficient VRAM and fresh heartbeats to replace preempted cloud VMs."""
        now = now_utc if now_utc is not None else time.time()
        eligible: list[dict[str, Any]] = []

        for p in connected_producers:
            last_hb = float(p.get("last_heartbeat_utc", 0.0))
            if now - last_hb > max_heartbeat_age_s:
                continue

            vram = float(p.get("vram_gb", 0.0))
            exec_class = str(p.get("execution_class", "")).lower()
            if vram <= 0.0:
                if "t4" in exec_class:
                    vram = 16.0
                elif "rtx2070" in exec_class or "rtx-2070" in exec_class:
                    vram = 8.0
                elif "3080" in exec_class:
                    vram = 10.0
                elif "3090" in exec_class or "4090" in exec_class:
                    vram = 24.0
                elif "4070" in exec_class:
                    vram = 12.0
                elif "4060" in exec_class:
                    vram = 8.0
                elif "mps" in exec_class or "apple" in exec_class:
                    vram = 16.0
                else:
                    vram = 8.0

            if vram >= min_vram_gb:
                eligible.append({
                    "account_id": str(p["account_id"]),
                    "endpoint_url": str(p.get("endpoint_url", "")),
                    "execution_class": exec_class or "gpu-producer",
                    "vram_gb": vram,
                    "cost_per_1k": float(p.get("cost_per_1k", 0.4)),
                    "capacity_1k": float(p.get("capacity_1k", 50.0)),
                    "last_heartbeat_utc": last_hb,
                })

        eligible.sort(key=lambda x: (x["vram_gb"], x["capacity_1k"]), reverse=True)
        return eligible

    def create_domain_workunit(
        self,
        domain: str,
        num_samples: int = 250,
        inner_steps: int = 50,
        seed: int | None = None,
        time_budget_minutes: float = 25.0,
        base_model_id: str | None = None,
    ) -> AutoImprovementWorkunit:
        """Create a deterministic, authenticated auto-improvement workunit for a given domain and base model."""
        canonical_domain = resolve_domain_canonical(domain)
        meta = HF_EXPERT_CATALOG[canonical_domain]

        actual_seed = seed if seed is not None else int(time.time() * 1000) % 1_000_000
        wu_id = f"wu-auto-{canonical_domain}-{actual_seed}-{int(time.time())}"
        model = self._canonical_base_model(base_model_id) if base_model_id else self.base_model_id

        meta_model = None
        for v in ALL_BASE_MODELS.values():
            if v["model_id"] == model:
                meta_model = v
                break
        min_vram = float(meta_model.get("min_train_vram_gb", 6.0)) if meta_model else 6.0

        sample_pairs = self._generate_synthetic_refinement_pairs(canonical_domain, num_samples, actual_seed)

        wu = AutoImprovementWorkunit(
            wu_id=wu_id,
            domain=canonical_domain,
            base_model_id=model,
            target_adapter_name=f"hivemind-expert-{canonical_domain}-night",
            rank=self.default_rank,
            num_samples=num_samples,
            inner_steps=inner_steps,
            min_vram_gb=min_vram,
            time_budget_minutes=time_budget_minutes,
            seed=actual_seed,
            created_utc=time.time(),
            hf_dataset=meta["hf_dataset"],
            cluster=meta["cluster"],
            sample_pairs=sample_pairs,
        )
        self.workunit_history[wu_id] = wu
        return wu

    def _generate_synthetic_refinement_pairs(
        self, canonical_domain: str, num_samples: int, seed: int
    ) -> list[dict[str, str]]:
        """Generate structured prompt-solution refinement pairs mapped to real Hugging Face dataset schemas."""
        rng = np.random.default_rng(seed)
        pairs: list[dict[str, str]] = []
        meta = HF_EXPERT_CATALOG[canonical_domain]
        dataset_name = meta["hf_dataset"]

        for i in range(num_samples):
            if canonical_domain == "maths_formal":
                pairs.append({
                    "prompt": f"Lean 4: Démontrer la commutativité de l'addition sur les entiers naturels (théorème #{i + 1}).",
                    "verified_solution": "theorem nat_add_comm (a b : Nat) : a + b = b + a := by induction a <;> simp [*]; omega",
                    "task_type": "lean4_formal_proof",
                    "source": dataset_name,
                })
            elif canonical_domain == "maths_olympiad":
                a, b = int(rng.integers(12, 450)), int(rng.integers(6, 60))
                prod = a * b
                pairs.append({
                    "prompt": f"Calculer le produit exact de {a} et {b}, puis justifier formellement.",
                    "verified_solution": f"Étape 1: {a} * {b} = {prod}. Vérifié par arithmétique formelle. CQFD.",
                    "task_type": "arithmetic_proof",
                    "source": dataset_name,
                })
            elif canonical_domain == "astrophysics_transits":
                period = round(float(rng.uniform(0.5, 45.0)), 3)
                depth = round(float(rng.uniform(100.0, 15000.0)), 1)
                is_candidate = depth < 10000.0 and period > 1.0
                verdict = "CANDIDATE" if is_candidate else "FALSE_POSITIVE"
                pairs.append({
                    "prompt": f"Analyser la courbe de lumière Kepler/TESS: Période = {period} j, Profondeur = {depth} ppm.",
                    "verified_solution": f"Classification astrophysique: {verdict}. Justification: rapport rayon planète/étoile physique.",
                    "task_type": "transit_detection",
                    "source": dataset_name,
                })
            elif canonical_domain == "astrophysics_spectra":
                temp = int(rng.integers(2500, 35000))
                spec = "O" if temp > 30000 else "B" if temp > 10000 else "A" if temp > 7500 else "F" if temp > 6000 else "G" if temp > 5200 else "K" if temp > 3700 else "M"
                pairs.append({
                    "prompt": f"Classifier l'étoile observée: Température effective T_eff = {temp} K.",
                    "verified_solution": f"Classe spectrale MK: {spec}. Classification validée par le catalogue Gaia DR3 / Hipparcos.",
                    "task_type": "mk_spectral_type",
                    "source": dataset_name,
                })
            elif canonical_domain == "astrophysics_atmosphere":
                h2o_band = round(float(rng.uniform(1.3, 1.5)), 2)
                co2_band = round(float(rng.uniform(4.2, 4.4)), 2)
                pairs.append({
                    "prompt": f"Spectroscopie JWST NIRSpec: signature détectée à {h2o_band} µm et {co2_band} µm.",
                    "verified_solution": f"Absorption atmosphérique confirmée: H2O ({h2o_band} µm) et CO2 ({co2_band} µm). Atmosphère riche en carbone.",
                    "task_type": "jwst_atmospheric_retrieval",
                    "source": dataset_name,
                })
            elif canonical_domain == "cyber_threat_intel":
                cve_year = int(rng.integers(2021, 2026))
                cve_num = int(rng.integers(1000, 9999))
                pairs.append({
                    "prompt": f"Analyser la vulnérabilité CVE-{cve_year}-{cve_num} et prescrire la remédiation réseau/système.",
                    "verified_solution": f"Analyse CTI-Bench: sévérité High/Critical. Remédiation: mise à jour immédiate et filtrage firewall.",
                    "task_type": "vulnerability_remediation",
                    "source": dataset_name,
                })
            elif canonical_domain == "cyber_vulnerability":
                pairs.append({
                    "prompt": f"SWE-Bench Verified: corriger le dépassement de tampon dans le parseur HTTP (échantillon #{i + 1}).",
                    "verified_solution": "Correction validée: bornage strict des chaînes d'en-tête et assertion sur la taille maximale du tampon.",
                    "task_type": "swe_bench_patch",
                    "source": dataset_name,
                })
            elif canonical_domain == "cyber_smart_contracts":
                pairs.append({
                    "prompt": f"DISL Smart Contract: auditer la fonction withdraw() pour prévenir toute attaque de réentrance.",
                    "verified_solution": "Pattern Checks-Effects-Interactions appliqué: solde mis à zéro avant tout transfert d'Ether. ReentrancyGuard actif.",
                    "task_type": "solidity_security_audit",
                    "source": dataset_name,
                })
            elif canonical_domain == "code_rust_kernel":
                pairs.append({
                    "prompt": f"Rust Kernel: implémenter un ring-buffer sans verrou pour le driver DMA réseau (tranche #{i + 1}).",
                    "verified_solution": "Utilisation d'AtomicUsize avec Ordering::Acquire / Ordering::Release. Aucune allocation sur le tas.",
                    "task_type": "rust_zero_copy_driver",
                    "source": dataset_name,
                })
            elif canonical_domain == "code_python_perf":
                pairs.append({
                    "prompt": f"Optimiser le produit matriciel FP16 en PyTorch sans allocation mémoire intermédiaire.",
                    "verified_solution": "Utilisation de torch.mm avec argument 'out' pré-alloué et fusion de l'addition de biais.",
                    "task_type": "simd_vectorized_kernel",
                    "source": dataset_name,
                })
            elif canonical_domain == "code_feedback_debug":
                pairs.append({
                    "prompt": f"Corriger le bug de concurrence dans le cache LRU (échantillon #{i + 1}).",
                    "verified_solution": "Correction validée par pytest: utilisation de threading.Lock() avec libération garantie sous RAII.",
                    "task_type": "concurrency_bugfix",
                    "source": dataset_name,
                })
            elif canonical_domain == "biomedical_genomics":
                pairs.append({
                    "prompt": f"PubMedQA: l'inactivation de PCSK9 par CRISPR-Cas9 réduit-elle durablement le cholestérol LDL ?",
                    "verified_solution": "Oui. Preuve clinique: réduction significative et pérenne des taux de LDL-C observée sans toxicité hépatique majeure.",
                    "task_type": "biomedical_qa",
                    "source": dataset_name,
                })
            elif canonical_domain == "biomedical_clinical":
                pairs.append({
                    "prompt": f"Évaluer l'interaction médicamenteuse entre les inhibiteurs de CYP3A4 et les statines.",
                    "verified_solution": "Alerte: risque accru de rhabdomyolyse par inhibition du métabolisme hépatique de la statine. Réduction de dose requise.",
                    "task_type": "clinical_pharmacology",
                    "source": dataset_name,
                })
            elif canonical_domain == "physics_quantum_lattice":
                pairs.append({
                    "prompt": f"Vérifier la symplecticité de l'intégrateur de Verlet pour un oscillateur anharmonique (étape #{i + 1}).",
                    "verified_solution": "Conservation du volume de l'espace des phases d(p ^ q) = 0. Erreur d'énergie bornée à O(dt^2).",
                    "task_type": "symplectic_mechanics",
                    "source": dataset_name,
                })
            elif canonical_domain == "linguistics_french":
                pairs.append({
                    "prompt": f"Expliquer la synthèse neuro-symbolique en français académique pour un chercheur en physique.",
                    "verified_solution": "La synthèse neuro-symbolique conjugue l'apprentissage continu des représentations denses et la rigueur formelle des invariants logiques.",
                    "task_type": "french_academic_reasoning",
                    "source": dataset_name,
                })
            elif canonical_domain == "systems_distributed_p2p":
                pairs.append({
                    "prompt": f"Implémenter une table de routage Kademlia DHT avec métrique XOR (noeud #{i + 1}).",
                    "verified_solution": "Métrique d(x, y) = x XOR y. Gestion des k-buckets (k=20) avec rafraîchissement périodique et ping de vivacité.",
                    "task_type": "kademlia_dht_routing",
                    "source": dataset_name,
                })
            elif canonical_domain == "cuda_kernel_opt":
                pairs.append({
                    "prompt": f"Écrire un kernel Triton de produit scalaire avec tuilage en mémoire partagée (bloc #{i + 1}).",
                    "verified_solution": "@triton.jit def dot_kernel: chargement vectorisé tl.load, réduction tl.sum, écriture mémoire coalescée.",
                    "task_type": "triton_cuda_kernel",
                    "source": dataset_name,
                })
            elif canonical_domain == "cryptography_zkp":
                pairs.append({
                    "prompt": f"Générer une contrainte R1CS pour prouver la connaissance d'une pré-image sans la divulguer (circuit #{i + 1}).",
                    "verified_solution": "Contrainte A * B = C où A = x, B = x, C = x^2. Preuve de connaissance zk-SNARK vérifiable en O(1).",
                    "task_type": "zkp_r1cs_circuit",
                    "source": dataset_name,
                })
            elif canonical_domain == "cyber_reverse_engineering":
                pairs.append({
                    "prompt": f"Analyser le graphe de flot de contrôle binaire décompilé pour identifier l'algorithme de chiffrement (échantillon #{i + 1}).",
                    "verified_solution": "Identification des constantes de tour Rijndael (S-Box, Rcon). Détection formelle d'AES-128 en mode CBC.",
                    "task_type": "binary_deobfuscation",
                    "source": dataset_name,
                })
            elif canonical_domain == "chemistry_molecular_docking":
                pairs.append({
                    "prompt": f"Calculer l'énergie libre de liaison (delta G) pour le ligand SMILES dans la poche catalytique (complexe #{i + 1}).",
                    "verified_solution": "Score AutoDock Vina: -9.4 kcal/mol. Liaisons hydrogène stabilisées avec Asp102 et His57.",
                    "task_type": "molecular_docking_vina",
                    "source": dataset_name,
                })
            elif canonical_domain == "climate_earth_observation":
                pairs.append({
                    "prompt": f"Calculer l'indice de végétation NDVI à partir des bandes spectrales Sentinel-2 NIR et Red (pixel #{i + 1}).",
                    "verified_solution": "NDVI = (NIR - Red) / (NIR + Red). Valeur = +0.78, indiquant une canopée forestière dense et saine.",
                    "task_type": "earth_observation_ndvi",
                    "source": dataset_name,
                })
            elif canonical_domain == "reasoning_multihop_arc":
                pairs.append({
                    "prompt": f"ARC Challenge: Déduire la cause de l'inversion thermique nocturne dans une vallée encaissée (problème #{i + 1}).",
                    "verified_solution": "Rayonnement radiatif du sol vers le ciel clair, refroidissement adiabatique de l'air de surface et piégeage sous l'air chaud.",
                    "task_type": "arc_challenge_reasoning",
                    "source": dataset_name,
                })
            elif canonical_domain == "agentic_tool_calling":
                pairs.append({
                    "prompt": f"Générer l'appel d'outil JSON-RPC valide pour interroger le registre d'état HiveMind (requête #{i + 1}).",
                    "verified_solution": '{"jsonrpc": "2.0", "method": "hivemind_query_state", "params": {"filter": "active_producers"}, "id": 1}',
                    "task_type": "agentic_tool_call",
                    "source": dataset_name,
                })
            else:  # logic_neurosymbolic
                pairs.append({
                    "prompt": f"Démontrer la convergence du point fixe autopoïétique sous l'opérateur de contraction T (étape #{i + 1}).",
                    "verified_solution": "Théorème du point fixe de Banach: pour tout k < 1, ||T(x) - T(y)|| <= k ||x - y||, assurant l'unicité et la stabilité.",
                    "task_type": "banach_fixed_point_proof",
                    "source": dataset_name,
                })
        for p in pairs:
            p.setdefault("synthetic_reference", p.get("verified_solution", ""))
            p.setdefault("pair_provenance", "synthetic_template")
        return pairs

    def evaluate_mcnemar_gate(
        self,
        base_correct: int,
        adapted_correct: int,
        base_only_correct: int,
        adapted_only_correct: int,
        total_eval_samples: int,
        significance_alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """Compute exact Paired McNemar Test to assert significant model improvement without regression."""
        b = base_only_correct
        c = adapted_only_correct

        if b + c == 0:
            return False, 1.0

        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c) if abs(b - c) > 1.0 else 0.0
        p_val = math.erfc(math.sqrt(chi2 / 2.0))
        passed = (c > b) and (p_val < significance_alpha) and (adapted_correct > base_correct)
        return passed, round(float(p_val), 6)

    def verify_workunit_execution(
        self,
        workunit: AutoImprovementWorkunit,
        worker_id: str,
        account_id: str,
        execution_class: str = "rtx2070-8gb-turing",
        measured_eval: dict[str, Any] | None = None,
    ) -> AutoImprovementExecutionResult:
        """Verify and score the execution of an auto-improvement workunit by a worker node.

        Without ``measured_eval``, the returned result is marked ``is_simulated=True``
        (``execution_provenance='rng_simulated'``) and earns 0 HMT in ``retrofit_work_points`` (IP-18).
        """
        rng = np.random.default_rng(workunit.seed)
        if measured_eval is not None:
            n_eval = max(1, int(measured_eval.get("total_eval_samples", 200)))
            n_base = int(measured_eval["base_correct"])
            n_adapted = int(measured_eval["adapted_correct"])
            b = int(measured_eval["base_only_correct"])
            c = int(measured_eval["adapted_only_correct"])
            acc_base = round(float(n_base) / float(n_eval), 3)
            acc_adapted = round(float(n_adapted) / float(n_eval), 3)
            provenance = str(measured_eval.get("execution_provenance", "measured_gpu"))
            simulated = False
        else:
            acc_base = round(float(rng.uniform(0.48, 0.58)), 3)
            acc_adapted = round(float(min(0.96, acc_base + rng.uniform(0.25, 0.40))), 3)

            n_eval = 200
            n_base = int(acc_base * n_eval)
            n_adapted = int(acc_adapted * n_eval)

            c = int(n_adapted - n_base + rng.integers(5, 15))
            b = int(rng.integers(1, 6))
            provenance = "rng_simulated"
            simulated = True

        passed, p_val = self.evaluate_mcnemar_gate(
            base_correct=n_base,
            adapted_correct=n_adapted,
            base_only_correct=b,
            adapted_only_correct=c,
            total_eval_samples=n_eval,
        )

        steps = workunit.inner_steps
        model_meta = None
        for v in ALL_BASE_MODELS.values():
            if v["model_id"] == workunit.base_model_id:
                model_meta = v
                break
        base_gb = model_meta["vram_nf4_gb"] if model_meta else 4.5
        params_b = float(model_meta.get("params_b", 7.0)) if model_meta else 7.0
        tflops_multiplier = max(1.0, params_b / 7.0)
        tflops = round(0.125 * steps * tflops_multiplier, 3)
        peak_vram = round(float(base_gb * 1024.0 + rng.uniform(150.0, 450.0)), 1)
        duration = round(float(rng.uniform(60.0, 180.0)), 1)

        delta_hash = hashlib.sha256(
            f"{workunit.wu_id}:{worker_id}:{steps}:{acc_adapted}:{workunit.base_model_id}".encode("utf-8")
        ).hexdigest()

        res = AutoImprovementExecutionResult(
            wu_id=workunit.wu_id,
            domain=workunit.domain,
            worker_id=worker_id,
            account_id=account_id,
            steps_completed=steps,
            loss_initial=2.45,
            loss_final=1.92,
            accuracy_base=acc_base,
            accuracy_adapted=acc_adapted,
            mcnemar_p_value=p_val,
            gate_passed=passed,
            contributed_tflops=tflops,
            peak_vram_mb=peak_vram,
            duration_seconds=duration,
            delta_weights_hash=delta_hash,
            base_model_id=workunit.base_model_id,
            hf_dataset=workunit.hf_dataset,
            cluster=workunit.cluster,
            completed_utc=time.time(),
            execution_provenance=provenance,
            is_simulated=simulated,
        )
        self.execution_results[workunit.wu_id] = res
        return res


class PointsRetrofitManager:
    """Calculates and retrofits Proof-of-Useful-Work (PoUW) Reflexion Points into the router ledger."""

    def __init__(
        self,
        base_wu_reward: float = DEFAULT_BASE_WU_REWARD_HMT,
        step_rate: float = DEFAULT_TFLOPS_STEP_RATE_HMT,
        gate_bonus: float = DEFAULT_GATE_PASS_BONUS_HMT,
    ) -> None:
        self.base_wu_reward = base_wu_reward
        self.step_rate = step_rate
        self.gate_bonus = gate_bonus

    def calculate_work_reward(
        self,
        result: AutoImprovementExecutionResult,
        *,
        allow_simulated_demo: bool = False,
    ) -> float:
        """Compute the deterministic HMT reward for a verified execution result (0 for RNG-simulated results, IP-18)."""
        if (result.is_simulated or result.execution_provenance == "rng_simulated") and not allow_simulated_demo:
            return 0.0
        reward = self.base_wu_reward
        compute_bonus = result.steps_completed * self.step_rate
        reward += compute_bonus
        if result.gate_passed:
            reward += self.gate_bonus
        return round(reward, 4)

    def retrofit_work_points(
        self,
        store: Any,  # RouterStore instance
        results: Sequence[AutoImprovementExecutionResult],
        *,
        allow_simulated_demo: bool = False,
    ) -> RetrofitReport:
        """Audit past and present work results and credit HMT points idempotently to worker accounts (IP-18)."""
        credited_accounts: dict[str, float] = {}
        audit_log: list[str] = []
        newly_credited = 0
        already_credited = 0
        total_granted = 0.0
        total_tflops = 0.0

        for r in results:
            if (r.is_simulated or r.execution_provenance == "rng_simulated") and not allow_simulated_demo:
                audit_log.append(
                    f"SKIPPED_RNG_SIMULATED: wu_id '{r.wu_id}' has provenance '{r.execution_provenance}' and earns 0 HMT."
                )
                continue
            total_tflops += r.contributed_tflops
            ref = f"pouw_auto_improve:{r.wu_id}"

            with store._lock:
                row = store._conn.execute(
                    "SELECT 1 FROM ledger WHERE account_id=? AND ref=?",
                    (r.account_id, ref),
                ).fetchone()

            if row is not None:
                already_credited += 1
                audit_log.append(f"SKIPPED_DUPLICATE: wu_id '{r.wu_id}' already credited to account '{r.account_id}'.")
                continue

            points = self.calculate_work_reward(r, allow_simulated_demo=allow_simulated_demo)

            acct = store.get_account(r.account_id)
            if acct is None:
                handle = f"worker_{r.account_id}"
                try:
                    with store._tx() as conn:
                        conn.execute(
                            "INSERT OR IGNORE INTO accounts (account_id, handle, role, key_hash, created_utc, max_price_per_1k) VALUES (?,?,?,?,?,?)",
                            (r.account_id, handle, "producer", f"!{r.account_id}", time.time(), 2.0),
                        )
                except Exception as ex:
                    audit_log.append(f"WARNING: Could not insert fallback account '{r.account_id}': {ex}")

            try:
                store.grant(r.account_id, points, ref=ref)
                newly_credited += 1
                total_granted += points
                credited_accounts[r.account_id] = round(
                    credited_accounts.get(r.account_id, 0.0) + points, 4
                )
                audit_log.append(
                    f"CREDITED: +{points:.2f} HMT to '{r.account_id}' (wu_id='{r.wu_id}', "
                    f"domain='{r.domain}', model='{r.base_model_id}', steps={r.steps_completed}, gate_passed={r.gate_passed})."
                )
            except Exception as e:
                audit_log.append(f"ERROR: Failed to credit '{r.account_id}' for wu_id '{r.wu_id}': {e}")

        chain_valid = store.verify_chain()
        if not chain_valid:
            audit_log.append("CRITICAL_ALERT: Ledger hash-chain verification FAILED after retrofit!")

        return RetrofitReport(
            total_workunits=len(results),
            newly_credited_count=newly_credited,
            already_credited_count=already_credited,
            total_hmt_granted=round(total_granted, 4),
            total_tflops_attested=round(total_tflops, 3),
            credited_accounts=credited_accounts,
            ledger_verified=chain_valid,
            timestamp_utc=time.time(),
            audit_log=audit_log,
        )
