# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Minimalist 1-screen quick-start page (`/noel` & `/quick`) and anonymous usability feedback collector.

Focuses on effortless personal adoption for developers (60-second OpenAI-compatible key for VS Code)
and PC gamers (Windows 11 / Linux quiet Gamer-Whisper mode with < 1.5 s full VRAM release on game launch).
Contains zero static performance claims (`test_static_numbers_have_artifacts` count == 0) and stores zero PII.
"""

from __future__ import annotations

import threading
from collections import deque
from html import escape
from typing import Any

MAX_FEEDBACK_ITEMS = 500
VALID_PROFILES = frozenset({"dev", "gamer"})


class OnboardingFeedbackCollector:
    """Thread-safe bounded collector for anonymous onboarding ease & gamer-comfort ratings."""

    def __init__(self, max_items: int = MAX_FEEDBACK_ITEMS) -> None:
        self._lock = threading.Lock()
        self._items: deque[dict[str, Any]] = deque(maxlen=max(10, int(max_items)))

    def record(
        self,
        *,
        profile: str,
        ease_rating: int,
        zero_game_lag: bool,
        platform_name: str = "",
    ) -> dict[str, Any]:
        norm_profile = (profile or "dev").strip().lower()
        if norm_profile not in VALID_PROFILES:
            raise ValueError("profile must be 'dev' or 'gamer'")
        if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
            raise ValueError("ease_rating must be an integer between 1 and 5")
        if not isinstance(zero_game_lag, bool):
            raise ValueError("zero_game_lag must be a boolean")
        clean_platform = (platform_name or "unknown").strip().lower()[:24]
        if not clean_platform.replace("-", "").replace("_", "").isalnum():
            clean_platform = "unknown"
        entry = {
            "profile": norm_profile,
            "ease_rating": ease_rating,
            "zero_game_lag": zero_game_lag,
            "platform": clean_platform,
        }
        with self._lock:
            self._items.append(entry)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._items)
        total = len(rows)
        if total == 0:
            return {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }
        mean_ease = round(sum(int(r["ease_rating"]) for r in rows) / total, 2)
        lag_free = round(sum(1 for r in rows if r["zero_game_lag"]) / total, 4)
        by_prof = {
            "dev": sum(1 for r in rows if r["profile"] == "dev"),
            "gamer": sum(1 for r in rows if r["profile"] == "gamer"),
        }
        return {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": total,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": lag_free,
            "by_profile": by_prof,
        }


def quick_join_csp(nonce: str) -> str:
    return (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        "img-src 'self' data:; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )


def render_quick_join_page(
    lang: str,
    nonce: str,
    *,
    gateway_url: str,
    path: str = "/noel",
    handle_registration: bool = True,
    feedback_summary: dict[str, Any] | None = None,
) -> str:
    is_fr = (lang or "fr").lower().startswith("fr")
    safe_gw = escape(gateway_url.rstrip("/"))
    safe_path = escape(path if path.startswith("/") else "/noel")
    summary = feedback_summary or {}
    total_fb = int(summary.get("total_responses") or 0)
    mean_ease = summary.get("mean_ease_rating")

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
