# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Privacy-preserving usage capture, prompt-intent & language classification, and client SDK attribution.

Every chat completion settled by the HiveMind OpenRouter gateway is classified in memory without ever storing raw
prompt text or raw client IP addresses. The resulting structured telemetry powers:

1. **Smart Intent Routing (`hivemind/auto`)** — automatically steers stiff ODE/PDE/fluid/quantum/genetics calculations
   to ``hivemind/bee-rusty-sundials``, astrophysics/orbital prompts to ``hivemind/bee-stars``, and software engineering
   prompts to ``hivemind/coder`` whenever those specialist tiers are live on the swarm.
2. **Real-Time French vs. English & IDE Adoption Telemetry (`GET /v1/hivemind/usage-capture`)** — tracks task domains,
   language distribution (``fr``, ``en``, ``code_heavy``), client SDKs (Cursor, Continue, Cline, OpenCode, Aider,
   Playground, OpenAI SDK), latency percentiles (``p50``, ``p95``), and user satisfaction feedback.
3. **Nocturnal QLoRA / BOINC Workunit Allocation (`nocturnal_moe_workunit_weights`)** — converts observed daytime
   domain demand and user thumbs-up/down ratings into workunit weights for the night training cycle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TASK_DOMAINS: tuple[str, ...] = (
    "science_rusty_sundials",
    "science_astronomy",
    "coding_rust",
    "coding_python",
    "coding_web_ts",
    "coding_general",
    "general_chat",
)

LANGUAGE_CODES: tuple[str, ...] = ("fr", "en", "code_heavy")

CLIENT_SDKS: tuple[str, ...] = (
    "playground_web",
    "cursor",
    "continue_dev",
    "cline_roo",
    "opencode",
    "aider",
    "zed",
    "antigravity_sdk",
    "openai_python",
    "openai_js",
    "curl",
    "custom_api",
)

_RUSTY_SUNDIALS_KEYWORDS: tuple[str, ...] = (
    "sundials",
    "cvode",
    "arkode",
    "ida",
    "kinsol",
    "stiff ode",
    "stiff pde",
    "navier-stokes",
    "navier stokes",
    "gross-pitaevskii",
    "gross pitaevskii",
    "bose-einstein",
    "superfluid",
    "wright-fisher",
    "wright fisher",
    "fisher-kpp",
    "lotka-volterra",
    "runge-kutta",
    "crank-nicolson",
    "bdf",
    "gmres",
    "jacobian",
    "cfl condition",
    "reynolds number",
    "vorticity",
)

_ASTRONOMY_KEYWORDS: tuple[str, ...] = (
    "exoplanet",
    "kepler",
    "transit timing",
    "ttv",
    "orbital",
    "n-body",
    "n body",
    "symplectic",
    "velocity-verlet",
    "astrophysics",
    "stellar",
    "supernova",
    "black hole",
    "schwarzschild",
    "kerr metric",
    "gravitational wave",
    "cosmology",
    "redshift",
    "light curve",
    "celestial",
)

_RUST_KEYWORDS: tuple[str, ...] = (
    "rust",
    "cargo",
    "tokio",
    "#[derive",
    "impl ",
    "fn main",
    "pub fn ",
    "borrow checker",
    "lifetime",
    "anyhow",
    "serde",
    "rayon",
)

_PYTHON_KEYWORDS: tuple[str, ...] = (
    "python",
    "pytest",
    "pytorch",
    "torch.",
    "fastapi",
    "pydantic",
    "numpy",
    "scipy",
    "pandas",
    "asyncio",
    "def ",
    "import ",
    "decorator",
    "qlora",
)

_WEB_TS_KEYWORDS: tuple[str, ...] = (
    "typescript",
    "javascript",
    "react",
    "next.js",
    "nextjs",
    "tailwind",
    "node.js",
    "interface ",
    "const ",
    "async function",
    "html",
    "css",
)

_CODING_GENERAL_KEYWORDS: tuple[str, ...] = (
    "sql",
    "SELECT ",
    "docker",
    "kubernetes",
    "bash",
    "shell",
    "git ",
    "algorithm",
    "refactor",
    "unit test",
    "regex",
    "leetcode",
    "compile",
    "binary search",
)

_FRENCH_WORD_RE = re.compile(
    r"\b(bonjour|salut|comment|pourquoi|explique|expliquer|ecris|écris|ecrire|écrire|"
    r"programme|fonction|avec|dans|pour|une|des|les|est|sont|nous|vous|ruche|"
    r"citoyen|calcul|modele|modèle|reseau|réseau|donnees|données|merci)\b",
    re.IGNORECASE,
)
_FRENCH_ACCENT_RE = re.compile(r"[éèêëàâùûçœæÉÈÊËÀÂÙÛÇŒÆ]")
_CODE_PUNCT_RE = re.compile(r"[{}\[\]();=<>/\\|_#$%^&*~`]")
_SAFE_TITLE_RE = re.compile(r"[^A-Za-z0-9 ._:+-]")


@dataclass(frozen=True)
class PromptIntent:
    """Privacy-preserving classification of a chat completion prompt."""

    task_domain: str
    language: str
    preferred_tier: str | None

    def __post_init__(self) -> None:
        for field_name in ("task_domain", "language"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip() or any(c in val for c in ("\r", "\n", "\x00")):
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.task_domain not in TASK_DOMAINS:
            raise ValueError(f"task_domain must be one of {TASK_DOMAINS}")
        if self.language not in LANGUAGE_CODES:
            raise ValueError(f"language must be one of {LANGUAGE_CODES}")
        if self.preferred_tier is not None and (
            not isinstance(self.preferred_tier, str)
            or not self.preferred_tier.strip()
            or any(c in self.preferred_tier for c in ("\r", "\n", "\x00"))
        ):
            raise ValueError("preferred_tier must be a non-empty string or None without control characters")


@dataclass(frozen=True)
class ClientAttribution:
    """Client SDK / IDE source parsed from standard HTTP & OpenRouter headers."""

    client_sdk: str
    app_title: str

    def __post_init__(self) -> None:
        for field_name in ("client_sdk", "app_title"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip() or any(c in val for c in ("\r", "\n", "\x00")):
                raise ValueError(f"{field_name} must be a non-empty string without control characters")
        if self.client_sdk not in CLIENT_SDKS:
            raise ValueError(f"client_sdk must be one of {CLIENT_SDKS}")


def _extract_messages_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(str(item["text"]))
    return "\n".join(parts)[:8192]


def _count_keyword_hits(lower_text: str, keywords: Sequence[str]) -> int:
    hits = 0
    for kw in keywords:
        if kw.lower() in lower_text:
            hits += 1
    return hits


def classify_prompt_intent_and_language(messages: Sequence[Mapping[str, Any]]) -> PromptIntent:
    """Classify the task domain, natural language (`fr`/`en`/`code_heavy`), and specialist tier without storing the prompt."""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a sequence of message mappings")
    text = _extract_messages_text(messages).strip()
    if not text:
        return PromptIntent(task_domain="general_chat", language="en", preferred_tier=None)

    lower = text.lower()

    # 1. Language detection (French vs English vs Code-Heavy)
    fr_words = len(_FRENCH_WORD_RE.findall(lower))
    fr_accents = len(_FRENCH_ACCENT_RE.findall(text))
    punct_count = len(_CODE_PUNCT_RE.findall(text))
    punct_ratio = punct_count / max(1, len(text))

    if fr_words >= 2 or (fr_words >= 1 and fr_accents >= 1) or fr_accents >= 3:
        language = "fr"
    elif punct_ratio >= 0.14 and len(text) >= 40:
        language = "code_heavy"
    else:
        language = "en"

    # 2. Domain & specialist tier classification
    scores: dict[str, int] = {
        "science_rusty_sundials": _count_keyword_hits(lower, _RUSTY_SUNDIALS_KEYWORDS) * 3,
        "science_astronomy": _count_keyword_hits(lower, _ASTRONOMY_KEYWORDS) * 3,
        "coding_rust": _count_keyword_hits(lower, _RUST_KEYWORDS) * 2,
        "coding_python": _count_keyword_hits(lower, _PYTHON_KEYWORDS) * 2,
        "coding_web_ts": _count_keyword_hits(lower, _WEB_TS_KEYWORDS) * 2,
        "coding_general": _count_keyword_hits(lower, _CODING_GENERAL_KEYWORDS) * 2,
    }

    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0:
        return PromptIntent(task_domain="general_chat", language=language, preferred_tier=None)

    if best_domain == "science_rusty_sundials":
        preferred_tier: str | None = "bee-rusty-sundials"
    elif best_domain == "science_astronomy":
        preferred_tier = "bee-stars"
    else:
        preferred_tier = "coder"

    return PromptIntent(task_domain=best_domain, language=language, preferred_tier=preferred_tier)


def detect_client_sdk(headers: Mapping[str, str]) -> ClientAttribution:
    """Identify the calling IDE, agent framework, or SDK from `User-Agent`, `X-Title`, and `HTTP-Referer`."""
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a mapping")
    ua = str(headers.get("user-agent", "") or "").lower()
    raw_title = str(headers.get("x-title", "") or headers.get("x-app-title", "") or "").strip()
    referer = str(headers.get("http-referer", "") or headers.get("referer", "") or "").lower()
    combined = f"{ua} {raw_title.lower()} {referer}"

    safe_title = _SAFE_TITLE_RE.sub("", raw_title)[:64].strip()

    if "hivemind-playground" in combined or referer.endswith("/playground"):
        sdk = "playground_web"
        safe_title = safe_title or "HiveMind-Playground"
    elif "cursor" in combined:
        sdk = "cursor"
    elif "continue" in combined:
        sdk = "continue_dev"
    elif "cline" in combined or "roo-code" in combined or "roocode" in combined:
        sdk = "cline_roo"
    elif "opencode" in combined:
        sdk = "opencode"
    elif "aider" in combined:
        sdk = "aider"
    elif "zed" in combined:
        sdk = "zed"
    elif "antigravity" in combined:
        sdk = "antigravity_sdk"
    elif "openai-python" in ua or "openai/python" in ua or "python-httpx" in ua or "python-requests" in ua:
        sdk = "openai_python"
    elif "openai-node" in ua or "openai/js" in ua or "openai/node" in ua or "node-fetch" in ua or "undici" in ua:
        sdk = "openai_js"
    elif "curl/" in ua:
        sdk = "curl"
    else:
        sdk = "custom_api"

    return ClientAttribution(client_sdk=sdk, app_title=safe_title or sdk)


def compute_nocturnal_moe_workunit_weights(domain_token_totals: Mapping[str, int]) -> dict[str, float]:
    """Translate observed daytime domain token demand into nocturnal BOINC / QLoRA workunit weights."""
    if not isinstance(domain_token_totals, Mapping):
        raise ValueError("domain_token_totals must be a mapping")
    for k, v in domain_token_totals.items():
        if not isinstance(k, str) or not k.strip() or any(c in k for c in ("\r", "\n", "\x00")):
            raise ValueError("domain_token_totals keys must be non-empty strings without control characters")
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"domain token count for {k!r} must be a non-negative integer")
    coder_tokens = sum(
        int(domain_token_totals.get(d, 0))
        for d in ("coding_python", "coding_rust", "coding_web_ts", "coding_general")
    )
    astro_tokens = int(domain_token_totals.get("science_astronomy", 0))
    sundials_tokens = int(domain_token_totals.get("science_rusty_sundials", 0))
    general_tokens = int(domain_token_totals.get("general_chat", 0))

    total = coder_tokens + astro_tokens + sundials_tokens + general_tokens
    if total <= 0:
        return {
            "hivemind/coder": 0.50,
            "hivemind/bee-rusty-sundials": 0.25,
            "hivemind/bee-stars": 0.15,
            "hivemind/moe-general": 0.10,
        }

    # Blend 70% observed demand with 30% minimum exploration floor so every active bee receives night refinement
    raw = {
        "hivemind/coder": 0.70 * (coder_tokens / total) + 0.30 * 0.45,
        "hivemind/bee-rusty-sundials": 0.70 * (sundials_tokens / total) + 0.30 * 0.25,
        "hivemind/bee-stars": 0.70 * (astro_tokens / total) + 0.30 * 0.20,
        "hivemind/moe-general": 0.70 * (general_tokens / total) + 0.30 * 0.10,
    }
    norm = sum(raw.values())
    return {k: round(v / norm, 4) for k, v in raw.items()}
