# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Capability tiers for the HiveMind OpenRouter: which model a GPU of a given size should serve, under which public id.

Why tiers and not one id per model tag: the market clears per public model id, so every producer that serves the same id
competes in (and adds liquidity to) the same market. Consumers pick a tier by how much quality they need; producers pick
the biggest tier their card can hold. A producer advertises ``hivemind/<tier id>`` and maps it to whatever its own server
calls the model (``local_name``), so the tags below are recommendations, not requirements.

The sizes are the download sizes listed in the Ollama library (weights only, 4-bit quantisation for the dense models);
the VRAM figures add room for the context cache and are deliberately conservative guidance, not measurements. The gateway
does not yet verify that a producer really serves the model behind a tier id (``HM-S13-05``), so a tier id is a claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Same shape the gateway accepts for a model id; kept here so a catalog entry can never be unroutable.
_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_.")


@dataclass(frozen=True)
class ModelTier:
    """One public model id a producer can serve and a consumer can ask for."""

    tier_id: str  # advertised as ``hivemind/<tier_id>``
    title: str
    ollama_tag: str  # recommended local tag (verify with ``ollama pull``; the library may rename tags)
    weights_gb: float  # download size of that tag
    min_vram_gb: float  # smallest card we recommend for it, including context cache
    moe: bool
    active_params_b: float | None  # parameters active per token for MoE models, else None
    suggested_cost_per_1k: float  # HMT per 1,000 tokens a producer might ask; HMT have no cash value
    suggested_capacity_1k: float  # 1,000-token units per epoch to advertise
    good_for: str

    def __post_init__(self) -> None:
        if not self.tier_id or set(self.tier_id) - _ID_CHARS:
            raise ValueError(f"tier_id {self.tier_id!r} must be lowercase letters, digits, '-', '_' or '.'")
        if self.min_vram_gb <= 0 or self.weights_gb <= 0:
            raise ValueError("sizes must be positive")
        if self.moe != (self.active_params_b is not None):
            raise ValueError("active_params_b must be set exactly for MoE tiers")
        if self.suggested_cost_per_1k < 0 or self.suggested_capacity_1k <= 0:
            raise ValueError("suggested cost must be >= 0 and capacity > 0")

    @property
    def public_id(self) -> str:
        return f"hivemind/{self.tier_id}"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["public_id"] = self.public_id
        return out


# Ordered by min_vram_gb ascending; ``recommend_tier`` relies on that order (checked by __post_init__ of the tuple below).
TIERS: tuple[ModelTier, ...] = (
    ModelTier("crowd-v1", "CrowdGPT Crowd-v1 (1B Generalist Trunk)", "hf.co/Vxtzq/Crowd-v1", 1.1, 4.0, False, None, 0.15, 50.0,
              "1B citizen generalist trunk trained on 7B FineWeb tokens (CrowdGPT x La Ruche joint federation)"),
    ModelTier("coder-lite", "Coder Lite (3B)", "qwen2.5-coder:3b", 1.9, 4.0, False, None, 0.2, 40.0,
              "quick edits, explanations and small scripts on entry-level cards"),
    ModelTier("gemma-core", "Google Gemma 3 Core (4B Multimodal & Multilingual)", "gemma3:4b", 3.3, 4.5, False, None, 0.25, 55.0,
              "multilingual French/English synthesis, concise explanations, structured JSON and edge reasoning"),
    ModelTier("bee-stars", "Bee-Stars Astrophysics & Symplectic QLoRA Expert (7B)", "hivemind-bee-stars", 4.4, 5.5, False, None, 0.35, 50.0,
              "exoplanet transit timing (TTV), symplectic N-body mechanics, Kerr/Schwarzschild relativity and stellar physics"),
    ModelTier("mistral-core", "Mistral 7B Instruct v0.3 Core (European Bilingual)", "mistral:7b-instruct", 4.4, 6.0, False, None, 0.35, 55.0,
              "native French/English European bilingual assistant, technical documentation, summarization and tool calling"),
    ModelTier("qwen35-core", "Qwen 3.5 / Qwen 3 Hybrid Reasoning & Coding Core (8B)", "qwen3:8b", 5.2, 6.0, False, None, 0.4, 60.0,
              "hybrid thinking/non-thinking polyglot core for Python, Rust, TypeScript, math and multilingual tasks"),
    ModelTier("bee-rusty-sundials", "Rusty-SUNDIALS Scientific & Numeric Rescue Bee (7B)", "qwen2.5-coder:7b", 4.7, 6.0, False, None, 0.35, 50.0,
              "stiff ODE/DAE/PDE numeric rescue & conservation analysis across astronomy, fluid mechanics, quantum fluids & genetics"),
    ModelTier("coder", "Coder (7B)", "qwen2.5-coder:7b", 4.7, 6.0, False, None, 0.4, 60.0,
              "everyday Python and simple game code: functions, classes, pygame loops, tracebacks"),
    ModelTier("bee-lean-prover", "Goedel-Prover-V2 & DeepSeek-Prover-V2 Formal Lean 4 Bee (8B)", "hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q6_K", 6.7, 8.5, False, None, 0.55, 60.0,
              "formal Lean 4 theorem proving, sorry elimination, Mathlib4 tactic synthesis and invariant verification"),
    ModelTier("bee-ruche-moe", "La Ruche 8-Expert Sparse MoE Community Bee (2.4B active)", "gpt-oss:20b", 13.0, 10.0, True, 2.4, 0.65, 75.0,
              "8-expert top-2 sparse Mixture-of-Experts community bee combining code, math, astrophysics, fluids and bilingual synthesis"),
    ModelTier("coder-plus", "Coder Plus (14B)", "qwen2.5-coder:14b", 9.0, 12.0, False, None, 0.7, 80.0,
              "larger refactors, multi-file reasoning, tests"),
    ModelTier("moe-qwen35-bee", "Qwen 3.5 Polyglot-Singularity MoE Bee (3.0B active)", "gpt-oss:20b", 13.0, 15.0, True, 3.0, 0.85, 85.0,
              "high-diversity MoE specialist bee routing across coding, formal proofs, scientific PDEs and multilingual reasoning on large GPU producers"),
    # Measured 2026-10-24 on a Tesla T4 (15,360 MiB): 12,331 MiB resident at 4,096 context with an 8-bit KV cache, flash attention
    # and the card otherwise empty; it does not fit beside a 7B and a 3B beside it is partly on CPU. It thinks before answering.
    ModelTier("moe-general", "General MoE (gpt-oss 20B, 3.6B active)", "gpt-oss:20b", 14.0, 15.0, True, 3.6, 0.9, 80.0,
              "reasoning-heavy questions and design; a mixture-of-experts model that fits a 16 GB card"),
    ModelTier("moe-coder", "Coder MoE (Qwen3-Coder 30B, 3.3B active)", "qwen3-coder:30b", 19.0, 24.0, True, 3.3, 1.2, 100.0,
              "the strongest coding tier: agentic edits across files with a long context"),
)


@dataclass(frozen=True)
class PrewiredPartnerModel:
    """Metadata for a federated community partner model pre-wired into the HiveMind OpenRouter and La Ruche."""

    tier_id: str
    public_id: str
    partner_id: str
    hf_repo: str
    github_repo: str
    website: str
    title: str
    parameters_b: float
    tokens_trained_b: float
    tokenizer: str
    shard_format: str
    license: str
    status: str
    aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["aliases"] = list(self.aliases)
        return out


PREWIRED_PARTNER_MODELS: tuple[PrewiredPartnerModel, ...] = (
    PrewiredPartnerModel(
        tier_id="crowd-v1",
        public_id="hivemind/crowd-v1",
        partner_id="crowdgpt/crowd-v1",
        hf_repo="Vxtzq/Crowd-v1",
        github_repo="Vxtzq/CrowdGPT",
        website="https://crowdgpt.net",
        title="CrowdGPT Crowd-v1 (1B Citizen Generalist Trunk)",
        parameters_b=1.0,
        tokens_trained_b=7.0,
        tokenizer="Qwen3",
        shard_format="uint16_1mb_bin",
        license="MIT",
        status="PREWIRED_READY",
        aliases=("hivemind/crowd-v1", "crowdgpt/crowd-v1", "Vxtzq/Crowd-v1", "vxtzq/crowd-v1", "crowd-v1"),
    ),
    PrewiredPartnerModel(
        tier_id="bee-rusty-sundials",
        public_id="hivemind/bee-rusty-sundials",
        partner_id="rusty-sundials/numeric-7b",
        hf_repo="xaviercallens/rusty-SUNDIALS",
        github_repo="xaviercallens/rusty-SUNDIALS",
        website="https://github.com/xaviercallens/rusty-SUNDIALS",
        title="Rusty-SUNDIALS Scientific & Numeric Rescue Bee (Astronomy, Fluid Mechanics, Quantum Fluids & Genetics)",
        parameters_b=7.0,
        tokens_trained_b=18.0,
        tokenizer="Qwen2.5/Qwen3.5",
        shard_format="safetensors_lora_r16",
        license="BSD-3-Clause / AGPL-3.0",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/bee-rusty-sundials",
            "rusty-sundials/numeric-7b",
            "hivemind/bee-scientific-sundials",
            "hivemind/sundials",
            "bee-rusty-sundials",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="gemma-core",
        public_id="hivemind/gemma-core",
        partner_id="google/gemma-3-4b",
        hf_repo="google/gemma-3-4b-it",
        github_repo="google-deepmind/gemma",
        website="https://ai.google.dev/gemma",
        title="Google Gemma 3 4B Multimodal & Multilingual Core",
        parameters_b=4.0,
        tokens_trained_b=4000.0,
        tokenizer="Gemma3",
        shard_format="gguf_q4_k_m",
        license="Gemma Terms of Use",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/gemma-core",
            "hivemind/gemma",
            "google/gemma-3-4b",
            "gemma3:4b",
            "gemma-core",
            "gemma",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="qwen35-core",
        public_id="hivemind/qwen35-core",
        partner_id="qwen/qwen3.5-8b",
        hf_repo="Qwen/Qwen3-8B",
        github_repo="QwenLM/Qwen3",
        website="https://qwenlm.github.io",
        title="Qwen 3.5 / Qwen 3 8B Hybrid Reasoning & Coding Core",
        parameters_b=8.2,
        tokens_trained_b=36000.0,
        tokenizer="Qwen3",
        shard_format="gguf_q4_k_m",
        license="Apache-2.0",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/qwen35-core",
            "hivemind/qwen3.5",
            "hivemind/qwen3",
            "qwen/qwen3.5-8b",
            "qwen3:8b",
            "qwen35-core",
            "qwen3.5",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="mistral-core",
        public_id="hivemind/mistral-core",
        partner_id="mistralai/mistral-7b-instruct",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        github_repo="mistralai/mistral-inference",
        website="https://mistral.ai",
        title="Mistral 7B Instruct v0.3 European Bilingual Core",
        parameters_b=7.3,
        tokens_trained_b=8000.0,
        tokenizer="Mistral-v3-Tekken",
        shard_format="gguf_q4_k_m",
        license="Apache-2.0",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/mistral-core",
            "hivemind/mistral",
            "mistralai/mistral-7b-instruct",
            "mistral:7b-instruct",
            "mistral-core",
            "mistral",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="bee-lean-prover",
        public_id="hivemind/bee-lean-prover",
        partner_id="goedel-lm/goedel-prover-v2-8b",
        hf_repo="Goedel-LM/Goedel-Prover-V2-8B",
        github_repo="Goedel-LM/Goedel-Prover-V2",
        website="https://goedel-lm.github.io",
        title="Goedel-Prover-V2 8B & DeepSeek-Prover-V2 7B Formal Lean 4 Prover Bee",
        parameters_b=8.0,
        tokens_trained_b=200.0,
        tokenizer="Qwen3/DeepSeek",
        shard_format="gguf_q6_k",
        license="MIT / Apache-2.0",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/bee-lean-prover",
            "hivemind/bee-lean",
            "goedel-lm/goedel-prover-v2-8b",
            "deepseek-ai/deepseek-prover-v2-7b",
            "bee-lean-prover",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="bee-ruche-moe",
        public_id="hivemind/bee-ruche-moe",
        partner_id="la-ruche/community-moe-8x",
        hf_repo="xaviercallens/hivemind-la-ruche-moe",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
        title="La Ruche 8-Expert Sparse MoE Community Bee (2.4B Active)",
        parameters_b=20.0,
        tokens_trained_b=150.0,
        tokenizer="Qwen3/GPT-OSS",
        shard_format="safetensors_moe_top2",
        license="AGPL-3.0",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/bee-ruche-moe",
            "hivemind/la-ruche-moe",
            "la-ruche/community-moe-8x",
            "bee-ruche-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-qwen35-bee",
        public_id="hivemind/moe-qwen35-bee",
        partner_id="qwen/qwen3.5-moe-bee",
        hf_repo="Qwen/Qwen3-30B-A3B",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        title="Qwen 3.5 Polyglot-Singularity MoE Bee (3.0B Active)",
        parameters_b=20.0,
        tokens_trained_b=36000.0,
        tokenizer="Qwen3.5/GPT-OSS",
        shard_format="gguf_mxfp4_moe",
        license="Apache-2.0 / AGPL-3.0",
        status="LIVE_ON_BIG_GPU",
        aliases=(
            "hivemind/moe-qwen35-bee",
            "hivemind/moe-bee",
            "qwen/qwen3.5-moe-bee",
            "moe-qwen35-bee",
        ),
    ),
    # =========================================================================
    # 20 Nocturnal Sectoral MoE Adapters (Hugging Face Specialization Targets)
    # =========================================================================
    # CLUSTER 00: Systems & Formal Verification
    PrewiredPartnerModel(
        tier_id="bee-lean-prover",
        public_id="hivemind/moe-maths-formal",
        partner_id="huggingface/hoskinson-center-proofnet",
        hf_repo="hoskinson-center/proofnet",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/hoskinson-center/proofnet",
        title="Lean 4 Machine-Checked Formal Prover MoE Adapter",
        parameters_b=14.7,
        tokens_trained_b=25.0,
        tokenizer="Qwen2.5/DeepSeek",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-maths-formal",
            "moe-maths-formal",
            "maths_formal",
            "proofnet",
            "lean4-prover-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="coder",
        public_id="hivemind/moe-rust-kernel",
        partner_id="huggingface/bigcode-the-stack-v2",
        hf_repo="bigcode/the-stack-v2-dedup",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/bigcode/the-stack-v2-dedup",
        title="Rust Systems & Zero-Copy Kernel DMA MoE Adapter",
        parameters_b=15.7,
        tokens_trained_b=35.0,
        tokenizer="DeepSeek2/Qwen",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-rust-kernel",
            "moe-rust-kernel",
            "code_rust_kernel",
            "the-stack-rust",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="coder",
        public_id="hivemind/moe-python-perf",
        partner_id="huggingface/codealpaca-20k",
        hf_repo="HuggingFaceH4/CodeAlpaca_20K",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K",
        title="Python Performance & SIMD Kernel Optimizer MoE Adapter",
        parameters_b=12.2,
        tokens_trained_b=20.0,
        tokenizer="Mistral-v3/Tekken",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-python-perf",
            "moe-python-perf",
            "code_python_perf",
            "python-perf-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="coder",
        public_id="hivemind/moe-feedback-debug",
        partner_id="huggingface/codefeedback-filtered",
        hf_repo="m-a-p/CodeFeedback-Filtered-Instruction",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction",
        title="Multi-Turn Interactive Bug Repair & Refactor MoE Adapter",
        parameters_b=32.5,
        tokens_trained_b=50.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-feedback-debug",
            "moe-feedback-debug",
            "code_feedback_debug",
            "debug-repair-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-dist-p2p",
        partner_id="huggingface/bigcode-santacoder",
        hf_repo="bigcode/santacoder",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/bigcode/santacoder",
        title="Distributed Consensus & P2P Gossip Protocols MoE Adapter",
        parameters_b=9.0,
        tokens_trained_b=15.0,
        tokenizer="Qwen3",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-dist-p2p",
            "moe-dist-p2p",
            "systems_distributed_p2p",
            "p2p-gossip-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-cuda-opt",
        partner_id="huggingface/neuralmagic-llm-compression",
        hf_repo="neuralmagic/LLM_Compression_Code",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/neuralmagic/LLM_Compression_Code",
        title="Triton / CUDA Tensor Core Kernel Optimization MoE Adapter",
        parameters_b=9.2,
        tokens_trained_b=18.0,
        tokenizer="Gemma2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-cuda-opt",
            "moe-cuda-opt",
            "cuda_kernel_opt",
            "triton-cuda-moe",
        ),
    ),
    # CLUSTER 01: Crypto & Cybersecurity
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-cyber-cti",
        partner_id="huggingface/giskard-cti-bench",
        hf_repo="giskard_cti/cti-bench",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/giskard_cti/cti-bench",
        title="Cyber Threat Intelligence & CTI-Bench MoE Adapter",
        parameters_b=14.7,
        tokens_trained_b=22.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-cyber-cti",
            "moe-cyber-cti",
            "cyber_threat_intel",
            "cti-bench-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-cyber-cve",
        partner_id="huggingface/swe-bench-verified",
        hf_repo="SWE-bench/SWE-bench_Verified",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified",
        title="CVE Vulnerability Remediation & SWE-Bench MoE Adapter",
        parameters_b=15.7,
        tokens_trained_b=30.0,
        tokenizer="DeepSeek2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-cyber-cve",
            "moe-cyber-cve",
            "cyber_vulnerability",
            "swe-bench-cve-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-smart-contracts",
        partner_id="huggingface/assert-kth-disl-solidity",
        hf_repo="ASSERT-KTH/DISL-Solidity-Smart-Contracts",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/ASSERT-KTH/DISL-Solidity-Smart-Contracts",
        title="Solidity Smart Contract Security & ZK Escrow MoE Adapter",
        parameters_b=12.2,
        tokens_trained_b=20.0,
        tokenizer="Mistral-v3",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-smart-contracts",
            "moe-smart-contracts",
            "cyber_smart_contracts",
            "solidity-zk-escrow-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-crypto-zkp",
        partner_id="huggingface/0xparc-zk-bench",
        hf_repo="0xPARC/zk-bench",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/0xPARC/zk-bench",
        title="Zero-Knowledge SNARK & STARK Circuit Prover MoE Adapter",
        parameters_b=32.5,
        tokens_trained_b=40.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-crypto-zkp",
            "moe-crypto-zkp",
            "cryptography_zkp",
            "zk-circuit-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-reverse-eng",
        partner_id="huggingface/vxunderground-malware-analysis",
        hf_repo="vxunderground/malware-analysis-prompts",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/vxunderground/malware-analysis-prompts",
        title="Binary Disassembly & Malware Triage Specialist MoE Adapter",
        parameters_b=9.0,
        tokens_trained_b=16.0,
        tokenizer="Qwen3",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-reverse-eng",
            "moe-reverse-eng",
            "cyber_reverse_engineering",
            "reverse-engineering-moe",
        ),
    ),
    # CLUSTER 02: Frontier Science & Astrophysics
    PrewiredPartnerModel(
        tier_id="bee-stars",
        public_id="hivemind/moe-astro-transits",
        partner_id="huggingface/mzrk36-exoplanets-unified",
        hf_repo="mzrk36/exoplanets-unified-dataset",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/mzrk36/exoplanets-unified-dataset",
        title="Kepler / TESS Exoplanet Transit Classifier MoE Adapter",
        parameters_b=9.2,
        tokens_trained_b=20.0,
        tokenizer="Gemma2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-astro-transits",
            "moe-astro-transits",
            "astrophysics_transits",
            "exoplanet-transits-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="bee-stars",
        public_id="hivemind/moe-astro-spectra",
        partner_id="huggingface/brunobarreto-sdss-dr12-stars",
        hf_repo="BrunoBarreto/sdss_dr12_stars_regression",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/BrunoBarreto/sdss_dr12_stars_regression",
        title="Gaia DR3 / SDSS MK Stellar Spectral Classifier MoE Adapter",
        parameters_b=14.7,
        tokens_trained_b=25.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-astro-spectra",
            "moe-astro-spectra",
            "astrophysics_spectra",
            "stellar-spectra-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="bee-stars",
        public_id="hivemind/moe-astro-atmosphere",
        partner_id="huggingface/shoriful025-exoplanet-atmosphere",
        hf_repo="Shoriful025/exoplanet_atmospheric_spectroscopy_results",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/Shoriful025/exoplanet_atmospheric_spectroscopy_results",
        title="JWST Exoplanetary Atmospheric Biosignature MoE Adapter",
        parameters_b=15.7,
        tokens_trained_b=28.0,
        tokenizer="DeepSeek2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-astro-atmosphere",
            "moe-astro-atmosphere",
            "astrophysics_atmosphere",
            "jwst-atmosphere-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="bee-rusty-sundials",
        public_id="hivemind/moe-quantum-lattice",
        partner_id="huggingface/multimodaluniverse",
        hf_repo="MultimodalUniverse/multimodal_universe",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/MultimodalUniverse/multimodal_universe",
        title="Quantum Lattice Field Theory & Integrators MoE Adapter",
        parameters_b=12.2,
        tokens_trained_b=24.0,
        tokenizer="Mistral-v3",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-quantum-lattice",
            "moe-quantum-lattice",
            "physics_quantum_lattice",
            "quantum-lattice-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-qwen35-bee",
        public_id="hivemind/moe-chem-docking",
        partner_id="huggingface/chainer-zinc250k",
        hf_repo="chainer/zinc250k",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/chainer/zinc250k",
        title="Molecular SMILES & Ligand-Protein Docking MoE Adapter",
        parameters_b=32.5,
        tokens_trained_b=45.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-chem-docking",
            "moe-chem-docking",
            "chemistry_molecular_docking",
            "molecular-docking-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-qwen35-bee",
        public_id="hivemind/moe-climate-earth",
        partner_id="huggingface/nasa-impact-hls",
        hf_repo="nasa-impact/hls-foundation",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/nasa-impact/hls-foundation",
        title="Multi-Spectral Earth Observation & Climate Invariants MoE Adapter",
        parameters_b=9.0,
        tokens_trained_b=18.0,
        tokenizer="Qwen3",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-climate-earth",
            "moe-climate-earth",
            "climate_earth_observation",
            "earth-observation-moe",
        ),
    ),
    # CLUSTER 03: Reasoning, Life Sciences & Language
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-maths-olympiad",
        partner_id="huggingface/openai-gsm8k",
        hf_repo="openai/gsm8k",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/openai/gsm8k",
        title="Multi-Step Mathematical Olympiad Reasoning MoE Adapter",
        parameters_b=9.2,
        tokens_trained_b=22.0,
        tokenizer="Gemma2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-maths-olympiad",
            "moe-maths-olympiad",
            "maths_olympiad",
            "gsm8k-olympiad-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-bio-genomics",
        partner_id="huggingface/qiaojin-pubmedqa",
        hf_repo="qiaojin/PubMedQA",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/qiaojin/PubMedQA",
        title="CRISPR Genomics & PubMedQA Evidence MoE Adapter",
        parameters_b=14.7,
        tokens_trained_b=26.0,
        tokenizer="Qwen2.5",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-bio-genomics",
            "moe-bio-genomics",
            "biomedical_genomics",
            "pubmedqa-genomics-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-bio-clinical",
        partner_id="huggingface/epfl-guidance-biomedical",
        hf_repo="epfl-llm/guidance-biomedical",
        github_repo="xaviercallens/HiveMind",
        website="https://huggingface.co/datasets/epfl-llm/guidance-biomedical",
        title="Clinical Trial Evidence & Drug Mechanisms MoE Adapter",
        parameters_b=15.7,
        tokens_trained_b=32.0,
        tokenizer="DeepSeek2",
        shard_format="safetensors_qlora_moe_top2",
        license="Apache-2.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-bio-clinical",
            "moe-bio-clinical",
            "biomedical_clinical",
            "clinical-biomed-moe",
        ),
    ),
    # =========================================================================
    # 4 Macro-Cluster Super-MoE Ensembles
    # =========================================================================
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-systems-formal",
        partner_id="hivemind/cluster-00-systems-formal",
        hf_repo="xaviercallens/hivemind-moe-systems-formal",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        title="Systems, Kernels & Lean 4 Formal Super-MoE (Cluster 00)",
        parameters_b=20.0,
        tokens_trained_b=120.0,
        tokenizer="Qwen2.5/DeepSeek",
        shard_format="safetensors_moe_top2",
        license="AGPL-3.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-systems-formal",
            "moe-systems-formal",
            "systems-formal-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-coder",
        public_id="hivemind/moe-cyber-crypto",
        partner_id="hivemind/cluster-01-cyber-crypto",
        hf_repo="xaviercallens/hivemind-moe-cyber-crypto",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        title="Cybersecurity, ZK Circuits & Smart Contracts Super-MoE (Cluster 01)",
        parameters_b=20.0,
        tokens_trained_b=110.0,
        tokenizer="Qwen2.5/Mistral",
        shard_format="safetensors_moe_top2",
        license="AGPL-3.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-cyber-crypto",
            "moe-cyber-crypto",
            "cyber-crypto-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-qwen35-bee",
        public_id="hivemind/moe-frontier-science",
        partner_id="hivemind/cluster-02-frontier-science",
        hf_repo="xaviercallens/hivemind-moe-frontier-science",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        title="Frontier Astrophysics, Quantum Lattice & Molecular Super-MoE (Cluster 02)",
        parameters_b=20.0,
        tokens_trained_b=135.0,
        tokenizer="Gemma2/DeepSeek/Mistral",
        shard_format="safetensors_moe_top2",
        license="AGPL-3.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-frontier-science",
            "moe-frontier-science",
            "frontier-science-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/moe-reasoning-bio",
        partner_id="hivemind/cluster-03-reasoning-bio",
        hf_repo="xaviercallens/hivemind-moe-reasoning-bio",
        github_repo="xaviercallens/HiveMind",
        website="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        title="Olympiad Math, Clinical Genomics & Neurosymbolic Super-MoE (Cluster 03)",
        parameters_b=20.0,
        tokens_trained_b=140.0,
        tokenizer="Qwen2.5/Gemma2",
        shard_format="safetensors_moe_top2",
        license="AGPL-3.0",
        status="LIVE_NOCTURNAL_MOE",
        aliases=(
            "hivemind/moe-reasoning-bio",
            "moe-reasoning-bio",
            "reasoning-bio-moe",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="coder-lite",
        public_id="hivemind/qwen35-4b",
        partner_id="qwen/qwen3.5-4b-instruct",
        hf_repo="Qwen/Qwen3.5-4B-Instruct",
        github_repo="QwenLM/Qwen",
        website="https://promptquorum.com/local-llms/small-local-llm-models",
        title="Qwen 3.5 4B (Best Overall Small Model for 8 GB RAM / Modest CPU)",
        parameters_b=4.0,
        tokens_trained_b=18000.0,
        tokenizer="Qwen3.5",
        shard_format="gguf_q4_k_m",
        license="Apache-2.0",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/qwen35-4b",
            "qwen/qwen3.5-4b-instruct",
            "qwen3.5:4b",
            "qwen-4b",
            "qwen35-4b",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="coder-lite",
        public_id="hivemind/phi4-mini",
        partner_id="microsoft/phi-4-mini-instruct",
        hf_repo="microsoft/Phi-4-mini-instruct",
        github_repo="microsoft/Phi-4",
        website="https://promptquorum.com/local-llms/small-local-llm-models",
        title="Microsoft Phi-4 Mini 3.8B (Best Ultra-Lightweight Footprint ~2.5 GB RAM)",
        parameters_b=3.8,
        tokens_trained_b=5000.0,
        tokenizer="Phi4",
        shard_format="gguf_q4_k_m",
        license="MIT",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/phi4-mini",
            "microsoft/phi-4-mini-instruct",
            "phi-4-mini",
            "phi4-mini",
            "phi-4:3.8b",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="gemma-core",
        public_id="hivemind/gemma4-e4b",
        partner_id="google/gemma-4-e4b-it",
        hf_repo="google/gemma-4-e4b-it",
        github_repo="google-deepmind/gemma",
        website="https://ai.google.dev/gemma",
        title="Google Gemma 4 E4B (Best Multimodal Edge Text/Image/Audio for CPU Devices)",
        parameters_b=4.0,
        tokens_trained_b=6000.0,
        tokenizer="Gemma4",
        shard_format="gguf_q4_k_m",
        license="Gemma Terms of Use",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/gemma4-e4b",
            "google/gemma-4-e4b-it",
            "gemma-4-e4b",
            "gemma4-e4b",
            "gemma4:4b",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="crowd-v1",
        public_id="hivemind/qwen35-sub2b",
        partner_id="qwen/qwen3.5-2b-instruct",
        hf_repo="Qwen/Qwen3.5-2B-Instruct",
        github_repo="QwenLM/Qwen",
        website="https://promptquorum.com/local-llms/small-local-llm-models",
        title="Qwen 3.5 2B / 0.8B (Best Sub-2B Ultra-Tight RAM Limit < 3 GB Memory)",
        parameters_b=2.0,
        tokens_trained_b=18000.0,
        tokenizer="Qwen3.5",
        shard_format="gguf_q4_k_m",
        license="Apache-2.0",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/qwen35-sub2b",
            "qwen/qwen3.5-2b-instruct",
            "qwen3.5:2b",
            "qwen3.5:0.8b",
            "qwen-2b",
            "qwen-0.8b",
        ),
    ),
    PrewiredPartnerModel(
        tier_id="moe-general",
        public_id="hivemind/layerstorm-frontier-moe",
        partner_id="layerstorm/qwen3-30b-a3b-pcie",
        hf_repo="Qwen/Qwen3-30B-A3B-Instruct-2507",
        github_repo="Paradox-Lab-504/LayerStoRm",
        website="https://arxiv.org/abs/2604.04192",
        title="LayerStoRm x HiveMind Waggle-Warm Two-Zone PCIe MoE (Qwen3-30B-A3B on 8 GB VRAM + Host RAM)",
        parameters_b=30.0,
        tokens_trained_b=36000.0,
        tokenizer="Qwen3",
        shard_format="safetensors_pcie_two_zone_q4",
        license="Apache-2.0",
        status="PREWIRED_READY",
        aliases=(
            "hivemind/layerstorm-frontier-moe",
            "layerstorm-frontier-moe",
            "layerstorm-moe",
            "hivemind/moe-layerstorm",
            "layerstorm/qwen3-30b-a3b-pcie",
        ),
    ),
)

_ALIAS_TO_TIER_ID: dict[str, str] = {
    alias.lower(): partner.tier_id
    for partner in PREWIRED_PARTNER_MODELS
    for alias in partner.aliases
}


def normalize_model_alias(model_id: str) -> str:
    """Normalize a public or partner alias (e.g. ``crowdgpt/crowd-v1`` or ``Vxtzq/Crowd-v1``) to its bare tier id."""
    cleaned = model_id.strip()
    mapped = _ALIAS_TO_TIER_ID.get(cleaned.lower())
    if mapped is not None:
        return mapped
    return cleaned.removeprefix("hivemind/")


def tier_by_id(tier_id: str) -> ModelTier | None:
    """Look a tier up by its bare id (``coder``), its public id (``hivemind/coder``), or a partner alias."""
    bare = normalize_model_alias(tier_id)
    return next((tier for tier in TIERS if tier.tier_id == bare), None)


def tiers_for_vram(vram_gb: float) -> tuple[ModelTier, ...]:
    """Every tier a card with ``vram_gb`` of memory can hold, smallest first."""
    if vram_gb != vram_gb or vram_gb < 0:  # NaN or negative
        raise ValueError("vram_gb must be a non-negative number")
    return tuple(tier for tier in TIERS if tier.min_vram_gb <= vram_gb)


def recommend_tier(vram_gb: float) -> ModelTier | None:
    """The biggest tier a card with ``vram_gb`` can hold, or ``None`` when even the smallest does not fit."""
    fitting = tiers_for_vram(vram_gb)
    return fitting[-1] if fitting else None


def catalog_payload() -> dict[str, Any]:
    """The public JSON the gateway serves at ``/v1/hivemind/catalog``."""
    return {
        "object": "hivemind.catalog",
        "note": "Tags are recommendations; verify them with your server. HMT are points with no cash value.",
        "season": "HM-S19-Q35",
        "swarm_plan_endpoint": "/v1/hivemind/swarm-plan",
        "tiers": [tier.to_dict() for tier in TIERS],
        "prewired_partner_models": [partner.to_dict() for partner in PREWIRED_PARTNER_MODELS],
    }


def _validate_catalog(tiers: tuple[ModelTier, ...]) -> None:
    ids = [tier.tier_id for tier in tiers]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate tier ids")
    vram = [tier.min_vram_gb for tier in tiers]
    if vram != sorted(vram):
        raise ValueError("TIERS must be ordered by min_vram_gb")


_validate_catalog(TIERS)
