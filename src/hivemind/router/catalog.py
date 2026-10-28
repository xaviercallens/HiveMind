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
