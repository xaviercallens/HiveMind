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
    ModelTier("coder-lite", "Coder Lite (3B)", "qwen2.5-coder:3b", 1.9, 4.0, False, None, 0.2, 40.0,
              "quick edits, explanations and small scripts on entry-level cards"),
    ModelTier("coder", "Coder (7B)", "qwen2.5-coder:7b", 4.7, 6.0, False, None, 0.4, 60.0,
              "everyday Python and simple game code: functions, classes, pygame loops, tracebacks"),
    ModelTier("coder-plus", "Coder Plus (14B)", "qwen2.5-coder:14b", 9.0, 12.0, False, None, 0.7, 80.0,
              "larger refactors, multi-file reasoning, tests"),
    ModelTier("moe-general", "General MoE (gpt-oss 20B, 3.6B active)", "gpt-oss:20b", 14.0, 16.0, True, 3.6, 0.9, 80.0,
              "reasoning-heavy questions and design; a mixture-of-experts model that fits a 16 GB card"),
    ModelTier("moe-coder", "Coder MoE (Qwen3-Coder 30B, 3.3B active)", "qwen3-coder:30b", 19.0, 24.0, True, 3.3, 1.2, 100.0,
              "the strongest coding tier: agentic edits across files with a long context"),
)


def tier_by_id(tier_id: str) -> ModelTier | None:
    """Look a tier up by its bare id (``coder``) or its public id (``hivemind/coder``)."""
    bare = tier_id.removeprefix("hivemind/")
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
    }


def _validate_catalog(tiers: tuple[ModelTier, ...]) -> None:
    ids = [tier.tier_id for tier in tiers]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate tier ids")
    vram = [tier.min_vram_gb for tier in tiers]
    if vram != sorted(vram):
        raise ValueError("TIERS must be ordered by min_vram_gb")


_validate_catalog(TIERS)
