# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Serving-layer experts: verified generators that sit beside the Season 3 LoRA domain experts.

A LoRA domain expert (``SEASON_3_DOMAIN_EXPERT_SPECS``) is a trained adapter that is hot-plugged into the sparse MoE. A
serving expert is different: it answers a request by calling a model server and checking the answer in a sandbox, and it has
no adapter, no dataset and no training step. They are kept in a separate table so that registering one can never add a
domain to ``train_and_publish_season3_domain_experts`` or change the number of experts in a live MoE.

``Python_Code_GWAYA`` is the first entry: GWAYA-Lite (a Q4 Qwen2.5-Coder tier behind GWAYA's AST stub audit and sandboxed
tests). A "VERIFIED" answer means it passed the visible tests, never that it is correct. The ``gwaya`` package is optional and
is imported only by :func:`load_gwaya_code_expert`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from hivemind.core.btm_moe import SEASON_3_DOMAIN_EXPERT_SPECS, keyword_route

PYTHON_CODE_GWAYA = "Python_Code_GWAYA"

# Keep in step with ``gwaya.hivemind_expert.KEYWORDS`` (checked by a test whenever gwaya is installed).
SERVING_EXPERT_SPECS: Dict[str, Dict[str, Any]] = {
    PYTHON_CODE_GWAYA: {
        "kind": "serving_verified",
        "keywords": [
            "def ", "import ", "python", "pytest", "unittest", "traceback", "assert ", "lambda", "numpy",
            "pandas", "list comprehension", "decorator", "generator", "dataclass", "pip ", ".py", "typing",
        ],
        "description": "Python function-level generation and repair gated by AST stub audit and sandboxed tests (GWAYA-Lite).",
    },
}


def route_with_serving_experts(
    prompt: str, top_k: int = 2, min_weight: float = 0.0
) -> Tuple[List[str], Dict[str, float]]:
    """Top-k routing over the LoRA domain experts and the serving experts, on the scale of ``route_domain_query``.

    Each domain scores ``1.0 + 2.5 * whole-word keyword hits``; the Top-k scores are normalised to weights that sum to 1.
    A serving expert is therefore chosen only when its keywords outscore the other domains for this prompt. A prompt with
    no keyword hit returns ``([], {})`` (use the base model).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    if isinstance(min_weight, bool) or not isinstance(min_weight, (int, float)) or not (0.0 <= float(min_weight) <= 1.0):
        raise ValueError("min_weight must be a number in [0.0, 1.0]")
    return keyword_route(prompt, (SEASON_3_DOMAIN_EXPERT_SPECS, SERVING_EXPERT_SPECS), top_k=top_k, min_weight=min_weight)


def load_gwaya_code_expert(large_url: str, large_model: str = "qwen2.5-coder:7b", *, small_url: str | None = None,
                           small_model: str = "qwen2.5-coder:3b", small_first: bool = False, max_rounds: int = 3) -> Any:
    """Build a ``gwaya.hivemind_expert.GwayaCodeExpert`` talking to llama-server (or any OpenAI-format server) at ``large_url``.

    The 7B tier alone is the default: GWAYA's T4 measurement (MBPP-sanitized, n=257) found a 3B-first cascade no better
    (83.3% against 82.9%). Raises ``ImportError`` with an install hint when a GWAYA release with the Code expert is absent.
    """
    if not isinstance(large_url, str) or not large_url.strip():
        raise ValueError("large_url must be a non-empty string")
    if not isinstance(large_model, str) or not large_model.strip():
        raise ValueError("large_model must be a non-empty string")
    if small_url is not None and (not isinstance(small_url, str) or not small_url.strip()):
        raise ValueError("small_url must be a non-empty string when provided")
    if not isinstance(small_model, str) or not small_model.strip():
        raise ValueError("small_model must be a non-empty string")
    if not isinstance(small_first, bool):
        raise ValueError("small_first must be a bool")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be an integer >= 1")
    try:
        from gwaya.hivemind_expert import GwayaCodeExpert
        from gwaya.llama_tier import LlamaServerClient
    except ImportError as exc:  # gwaya missing, or a release older than the Code expert
        raise ImportError(
            "the GWAYA Code expert needs the gwaya package from the GWAYA repo (main, PR #21 or later); "
            "install it with: pip install -e /path/to/GWAYA-GwenLayaOpenModel"
        ) from exc
    small = LlamaServerClient(small_model, small_url) if small_url else None
    return GwayaCodeExpert(small, LlamaServerClient(large_model, large_url), small_first=small_first, max_rounds=max_rounds)
