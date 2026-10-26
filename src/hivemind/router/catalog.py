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
    ModelTier("coder", "Coder (7B)", "qwen2.5-coder:7b", 4.7, 6.0, False, None, 0.4, 60.0,
              "everyday Python and simple game code: functions, classes, pygame loops, tracebacks"),
    ModelTier("coder-plus", "Coder Plus (14B)", "qwen2.5-coder:14b", 9.0, 12.0, False, None, 0.7, 80.0,
              "larger refactors, multi-file reasoning, tests"),
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
