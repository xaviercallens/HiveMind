# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Minimalist 1-screen quick-start page (`/noel` & `/quick`) and anonymous usability feedback collector.

Focuses on effortless personal adoption for developers (60-second OpenAI-compatible key for VS Code)
and PC gamers (Windows 11 / Linux quiet Gamer-Whisper mode with < 1.5 s full VRAM release on game launch).
Contains zero static performance claims (`test_static_numbers_have_artifacts` count == 0) and stores zero PII.
"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from collections.abc import Mapping
from html import escape
from typing import Any
from urllib.parse import urlparse

MAX_FEEDBACK_ITEMS = 500
MAX_FEEDBACK_ITEMS_CEILING = 50_000
MAX_PROFILE_CHARS = 32
MAX_PLATFORM_RAW_CHARS = 64
MAX_PLATFORM_CLEAN_CHARS = 24
MAX_GATEWAY_URL_CHARS = 256
MAX_ROUTE_PATH_CHARS = 128
MAX_LANG_CHARS = 64
VALID_PROFILES = frozenset({"dev", "gamer"})

_NONCE_RE = re.compile(r"\A[A-Za-z0-9_+=-]{8,80}\Z", re.ASCII)
_PLATFORM_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_SAFE_PATH_RE = re.compile(r"\A/[A-Za-z0-9_./-]{0,127}\Z", re.ASCII)
_FORBIDDEN_URL_CHARS: frozenset[str] = frozenset(
    {" ", "\t", "\r", "\n", '"', "'", "<", ">", "`", ";", "|", "&", "$", "\\"}
)


class OnboardingFeedbackCollector:
    """Thread-safe bounded collector for anonymous onboarding ease & gamer-comfort ratings."""

    def __init__(self, max_items: int = MAX_FEEDBACK_ITEMS) -> None:
        if (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or max_items < 1
            or max_items > MAX_FEEDBACK_ITEMS_CEILING
        ):
            raise ValueError(f"max_items must be an integer between 1 and {MAX_FEEDBACK_ITEMS_CEILING}")
        self._max_items = max_items
        self._lock = threading.Lock()
        self._items: deque[tuple[str, int, bool, str]] = deque(maxlen=max_items)
        self._ease_sum = 0
        self._zero_lag_count = 0
        self._by_profile: dict[str, int] = {"dev": 0, "gamer": 0}
        self._lifetime_responses = 0
        self._evicted_responses = 0

    def window_diagnostics(self) -> dict[str, int]:
        """Return atomic ring-buffer bounds and lifetime/eviction counters."""
        with self._lock:
            return {
                "max_items": self._max_items,
                "retained_items": len(self._items),
                "lifetime_responses": self._lifetime_responses,
                "evicted_responses": self._evicted_responses,
            }

    def _summary_unlocked(self) -> dict[str, Any]:
        total = len(self._items)
        if total == 0:
            return {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }
        return {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": total,
            "mean_ease_rating": round(self._ease_sum / total, 2),
            "zero_game_lag_ratio": round(self._zero_lag_count / total, 4),
            "by_profile": {
                "dev": self._by_profile["dev"],
                "gamer": self._by_profile["gamer"],
            },
        }

    def record(
        self,
        *,
        profile: str,
        ease_rating: int,
        zero_game_lag: bool,
        platform_name: str = "",
    ) -> dict[str, Any]:
        if (
            isinstance(profile, bool)
            or not isinstance(profile, str)
            or len(profile) > MAX_PROFILE_CHARS
            or not profile.isascii()
        ):
            raise ValueError("profile must be 'dev' or 'gamer'")
        norm_profile = profile.strip().lower()
        if norm_profile not in VALID_PROFILES:
            raise ValueError("profile must be 'dev' or 'gamer'")
        if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
            raise ValueError("ease_rating must be an integer between 1 and 5")
        if not isinstance(zero_game_lag, bool):
            raise ValueError("zero_game_lag must be a boolean")
        if platform_name is not None and (isinstance(platform_name, bool) or not isinstance(platform_name, str)):
            raise ValueError("platform_name must be a string")

        clean_platform = "unknown"
        if platform_name and len(platform_name) <= MAX_PLATFORM_RAW_CHARS and platform_name.isascii():
            candidate = platform_name.strip().lower()[:MAX_PLATFORM_CLEAN_CHARS].rstrip("_-")
            if _PLATFORM_RE.fullmatch(candidate):
                clean_platform = candidate

        with self._lock:
            if len(self._items) == self._max_items:
                old_prof, old_ease, old_lag, _ = self._items.popleft()
                self._ease_sum -= old_ease
                if old_lag:
                    self._zero_lag_count -= 1
                self._by_profile[old_prof] -= 1
                self._evicted_responses += 1
            self._items.append((norm_profile, ease_rating, zero_game_lag, clean_platform))
            self._ease_sum += ease_rating
            if zero_game_lag:
                self._zero_lag_count += 1
            self._by_profile[norm_profile] += 1
            self._lifetime_responses += 1
            return self._summary_unlocked()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_unlocked()


def quick_join_csp(nonce: str) -> str:
    if isinstance(nonce, bool) or not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must be 8-80 URL-safe / base64 characters")
    return (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        "img-src 'self' data:; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )


def _validate_gateway_url(gateway_url: str) -> str:
    if (
        isinstance(gateway_url, bool)
        or not isinstance(gateway_url, str)
        or len(gateway_url) > MAX_GATEWAY_URL_CHARS
        or not gateway_url.isascii()
        or not gateway_url.strip()
    ):
        raise ValueError("gateway_url must be a non-empty http(s) URL")
    cleaned = gateway_url.strip().rstrip("/")
    if any(ord(ch) < 33 or ord(ch) == 127 or ch in _FORBIDDEN_URL_CHARS for ch in cleaned):
        raise ValueError("gateway_url contains forbidden characters")
    parsed = urlparse(cleaned)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("gateway_url has an invalid port") from None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or (port is not None and not (1 <= port <= 65535))
    ):
        raise ValueError("gateway_url must be a clean http(s) origin without credentials")
    return cleaned


def _validate_route_path(path: str) -> str:
    if (
        isinstance(path, bool)
        or not isinstance(path, str)
        or len(path) > MAX_ROUTE_PATH_CHARS
        or not path.startswith("/")
        or path.startswith("//")
        or ".." in path.split("/")
        or not _SAFE_PATH_RE.fullmatch(path)
    ):
        raise ValueError("path must be a clean URL path starting with '/'")
    return path


def render_quick_join_page(
    lang: str,
    nonce: str,
    *,
    gateway_url: str,
    path: str = "/noel",
    handle_registration: bool = True,
    feedback_summary: Mapping[str, Any] | None = None,
) -> str:
    if lang is not None and (isinstance(lang, bool) or not isinstance(lang, str)):
        raise ValueError("lang must be a string or None")
    if not isinstance(handle_registration, bool):
        raise ValueError("handle_registration must be a boolean")
    if feedback_summary is not None and not isinstance(feedback_summary, Mapping):
        raise ValueError("feedback_summary must be a Mapping or None")

    quick_join_csp(nonce)
    is_fr = (lang[:MAX_LANG_CHARS] if lang else "fr").strip().lower().startswith("fr")
    safe_gw = escape(_validate_gateway_url(gateway_url))
    safe_path = escape(_validate_route_path(path))
    summary = feedback_summary or {}
    raw_total = summary.get("total_responses")
    total_fb = raw_total if (not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total > 0) else 0
    raw_ease = summary.get("mean_ease_rating")
    if (
        total_fb > 0
        and not isinstance(raw_ease, bool)
        and isinstance(raw_ease, (int, float))
        and math.isfinite(float(raw_ease))
        and 1.0 <= float(raw_ease) <= 5.0
    ):
        mean_ease: int | float | None = int(raw_ease) if isinstance(raw_ease, int) else round(float(raw_ease), 2)
    else:
        mean_ease = None

    if is_fr:
        title = "HiveMind — Accès Simple Vacances de Noël (Dev & Gamer PC)"
        subtitle = "Choisis ton profil en une minute : tester l'API dans VS Code ou prêter ton GPU perso en silence."
        rule_banner = (
            "Règle stricte : usage 100 % personnel sur machine personnelle à domicile et projets perso uniquement "
            "(jamais sur un PC professionnel ni avec du code d'entreprise). Projet citoyen AGPL-3.0 • 0,00 EUR."
        )
        dev_title = "1. Développeur — Clé API VS Code en une minute"
        dev_desc = (
            "Aucune bibliothèque lourde à installer : récupère une clé compatible OpenAI pour coder tes projets "
            "perso de Noël dans VS Code (Continue.dev) ou en Python."
        )
        dev_cmd_label = "Commande directe (Windows / macOS / Linux, Python standard) :"
        dev_signup_btn = "Ou créer un compte sur /signup"
        gamer_title = "2. Gamer PC (Windows 11 / Linux) — Mode Silencieux"
        gamer_desc = (
            "Plafond thermique doux (70 °C max), 2 Gio de VRAM gardés libres, et arrêt complet immédiat "
            "dès qu'un jeu Steam démarre. Désinstallation propre en une commande."
        )
        win_label = "Windows 11 (PowerShell) :"
        nix_label = "Linux / macOS :"
        uninstall_label = "Tout supprimer localement :"
        fb_line = (
            f"Retours testeurs enregistrés : {total_fb} (facilité moyenne : {mean_ease}/5)"
            if total_fb > 0 and mean_ease is not None
            else "Retours testeurs : sois parmi les premiers à noter la simplicité (--feedback)."
        )
        switch_html = f'<a href="{safe_path}?lang=en">English version</a>'
    else:
        title = "HiveMind — Holiday Quick Start (Developers & PC Gamers)"
        subtitle = "Pick your path in one minute: plug a free OpenAI-compatible key into VS Code or lend idle home GPU quietly."
        rule_banner = (
            "Strict rule: 100% personal use on personal home machines and hobby/open-source projects ONLY "
            "(never on corporate laptops or with company code). Citizen-science AGPL-3.0 • 0.00 EUR / 0.00 USD."
        )
        dev_title = "1. Developer — VS Code API Key in One Minute"
        dev_desc = (
            "Zero heavy dependencies required: get an OpenAI-compatible key for personal holiday coding "
            "inside VS Code (Continue.dev) or Python."
        )
        dev_cmd_label = "Single command (Windows / macOS / Linux, Python stdlib only):"
        dev_signup_btn = "Or create an account on /signup"
        gamer_title = "2. PC Gamer (Windows 11 / Linux) — Quiet Whisper Mode"
        gamer_desc = (
            "Gentle 70 C thermal ceiling, 2 GiB free VRAM reserve, and full worker stop as soon as "
            "a Steam game starts. Clean one-command uninstall."
        )
        win_label = "Windows 11 (PowerShell):"
        nix_label = "Linux / macOS:"
        uninstall_label = "Clean local uninstall:"
        fb_line = (
            f"Recorded tester ratings: {total_fb} (mean ease rating: {mean_ease}/5)"
            if total_fb > 0 and mean_ease is not None
            else "Tester feedback: send your ease rating anytime with --feedback."
        )
        switch_html = f'<a href="{safe_path}?lang=fr">Version française</a>'

    reg_note = "" if handle_registration else " (via /signup)"

    return f"""<!DOCTYPE html>
<html lang="{'fr' if is_fr else 'en'}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style nonce="{escape(nonce)}">
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; line-height: 1.5; }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    h1 {{ font-size: 1.65rem; margin: 0 0 6px 0; color: #f8fafc; }}
    .sub {{ color: #94a3b8; margin: 0 0 16px 0; }}
    .rule {{ background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 0.92rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; }}
    .card h2 {{ font-size: 1.15rem; margin-top: 0; color: #38bdf8; }}
    pre {{ background: #020617; color: #a7f3d0; padding: 10px 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }}
    .btn {{ display: inline-block; background: #0284c7; color: #fff; text-decoration: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; margin-top: 6px; }}
    .foot {{ margin-top: 20px; color: #94a3b8; font-size: 0.88rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; }}
    a {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{escape(title)}</h1>
    <p class="sub">{escape(subtitle)}</p>
    <div class="rule">{escape(rule_banner)}</div>
    <div class="grid">
      <section class="card">
        <h2>{escape(dev_title)}{escape(reg_note)}</h2>
        <p>{escape(dev_desc)}</p>
        <p><strong>{escape(dev_cmd_label)}</strong></p>
        <pre>python3 scripts/hivemind_quickstart.py --gateway {safe_gw} --write-continue-config</pre>
        <a class="btn" href="/signup">{escape(dev_signup_btn)}</a>
      </section>
      <section class="card">
        <h2>{escape(gamer_title)}</h2>
        <p>{escape(gamer_desc)}</p>
        <p><strong>{escape(win_label)}</strong></p>
        <pre>.\\scripts\\join_hivemind.ps1 -Mode GamerWhisper -Gateway {safe_gw}</pre>
        <p><strong>{escape(nix_label)}</strong></p>
        <pre>./scripts/join_hivemind_campaign.sh --gamer-whisper</pre>
        <p><strong>{escape(uninstall_label)}</strong></p>
        <pre>python3 scripts/hivemind_quickstart.py --uninstall</pre>
      </section>
    </div>
    <footer class="foot">
      <span>{escape(fb_line)}</span>
      <span>{switch_html} • <a href="/join">/join</a></span>
    </footer>
  </div>
</body>
</html>
"""
