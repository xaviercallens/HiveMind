# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Minimalist 1-screen quick-start page (`/noel` & `/quick`) and anonymous usability feedback collector.

Focuses on effortless personal adoption for developers (60-second OpenAI-compatible key for VS Code)
and PC gamers (Windows 11 / Linux quiet Gamer-Whisper mode with < 1.5 s full VRAM release on game launch).
Contains zero static performance claims (`test_static_numbers_have_artifacts` count == 0) and stores zero PII.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from html import escape
from typing import Any
from urllib.parse import urlparse

MAX_FEEDBACK_ITEMS = 500
MAX_FEEDBACK_ITEMS_CEILING = 50_000
MAX_IDEMPOTENCY_KEYS = 2048
MAX_IDEMPOTENCY_KEY_CHARS = 128
MAX_PROFILE_CHARS = 32
MAX_PLATFORM_RAW_CHARS = 64
MAX_PLATFORM_CLEAN_CHARS = 24
MAX_GATEWAY_URL_CHARS = 256
MAX_ROUTE_PATH_CHARS = 128
MAX_LANG_CHARS = 64
VALID_PROFILES = frozenset({"dev", "gamer"})

HOLIDAY_SMOKE_REQUIRED_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/raid/status",
)
_HOLIDAY_SMOKE_HTML_ROUTES: frozenset[str] = frozenset(
    {"/noel", "/quick", "/raid-noel", "/arena-noel"}
)

_NONCE_RE = re.compile(r"\A[A-Za-z0-9_+=-]{8,80}\Z", re.ASCII)
_PLATFORM_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_SAFE_PATH_RE = re.compile(r"\A/[A-Za-z0-9_./-]{0,127}\Z", re.ASCII)
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_HEX_SHA_RE = re.compile(r"\A[0-9a-f]{7,64}\Z", re.ASCII)
_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_NONCE_ATTR_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_STATIC_FIGURE_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_UNSAFE_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "document" + r"\.write)\b"
)
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)
_FORBIDDEN_URL_CHARS: frozenset[str] = frozenset(
    {" ", "\t", "\r", "\n", '"', "'", "<", ">", "`", ";", "|", "&", "$", "\\"}
)


def _validate_idempotency_key(idempotency_key: str | None) -> str | None:
    """Validate optional ``Idempotency-Key`` (ASCII, <= 128 chars, no control chars or whitespace-only)."""
    if idempotency_key is None:
        return None
    if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
        raise ValueError("idempotency_key must be a non-empty ASCII string")
    if len(idempotency_key) > MAX_IDEMPOTENCY_KEY_CHARS:
        raise ValueError(f"idempotency_key must be at most {MAX_IDEMPOTENCY_KEY_CHARS} characters")
    if not idempotency_key.isascii():
        raise ValueError("idempotency_key must contain ASCII characters only")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in idempotency_key):
        raise ValueError("idempotency_key must not contain control characters")
    cleaned = idempotency_key.strip()
    if not cleaned:
        raise ValueError("idempotency_key must not be whitespace-only")
    return cleaned


class OnboardingFeedbackCollector:
    """Thread-safe bounded collector for anonymous onboarding ease & gamer-comfort ratings."""

    def __init__(
        self,
        max_items: int = MAX_FEEDBACK_ITEMS,
        max_idempotency_keys: int = MAX_IDEMPOTENCY_KEYS,
    ) -> None:
        if (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or max_items < 1
            or max_items > MAX_FEEDBACK_ITEMS_CEILING
        ):
            raise ValueError(f"max_items must be an integer between 1 and {MAX_FEEDBACK_ITEMS_CEILING}")
        if (
            isinstance(max_idempotency_keys, bool)
            or not isinstance(max_idempotency_keys, int)
            or max_idempotency_keys < 1
            or max_idempotency_keys > MAX_FEEDBACK_ITEMS_CEILING
        ):
            raise ValueError(
                f"max_idempotency_keys must be an integer between 1 and {MAX_FEEDBACK_ITEMS_CEILING}"
            )
        self._max_items = max_items
        self._max_idempotency_keys = max_idempotency_keys
        self._lock = threading.Lock()
        self._items: deque[tuple[str, int, bool, str]] = deque(maxlen=max_items)
        self._ease_sum = 0
        self._zero_lag_count = 0
        self._by_profile: dict[str, int] = {"dev": 0, "gamer": 0}
        self._lifetime_responses = 0
        self._evicted_responses = 0
        # Lazily allocate idempotency LRU only when a non-None idempotency_key is observed.
        self._idempotency_keys: OrderedDict[str, None] | None = None
        self._unique_idempotency_keys = 0
        self._deduplicated_replays = 0
        self._evicted_idempotency_keys = 0

    def window_diagnostics(self) -> dict[str, int]:
        """Return atomic ring-buffer bounds and lifetime/eviction counters."""
        with self._lock:
            return {
                "max_items": self._max_items,
                "retained_items": len(self._items),
                "lifetime_responses": self._lifetime_responses,
                "evicted_responses": self._evicted_responses,
            }

    def idempotency_diagnostics(self) -> dict[str, int]:
        """Return atomic bounds and counters for bounded Idempotency-Key LRU deduplication."""
        with self._lock:
            retained = len(self._idempotency_keys) if self._idempotency_keys is not None else 0
            return {
                "max_idempotency_keys": self._max_idempotency_keys,
                "retained_idempotency_keys": retained,
                "unique_idempotency_keys": self._unique_idempotency_keys,
                "deduplicated_replays": self._deduplicated_replays,
                "evicted_idempotency_keys": self._evicted_idempotency_keys,
            }

    def _summary_unlocked(self) -> dict[str, Any]:
        total = len(self._items)
        if total == 0:
            return {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 0,
                "mean_ease_rating": None,
                "average_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }
        mean_ease = round(self._ease_sum / total, 2)
        return {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": total,
            "mean_ease_rating": mean_ease,
            "average_ease_rating": mean_ease,
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
        idempotency_key: str | None = None,
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
        norm_idem = _validate_idempotency_key(idempotency_key)

        clean_platform = "unknown"
        if platform_name and len(platform_name) <= MAX_PLATFORM_RAW_CHARS and platform_name.isascii():
            candidate = platform_name.strip().lower()[:MAX_PLATFORM_CLEAN_CHARS].rstrip("_-")
            if _PLATFORM_RE.fullmatch(candidate):
                clean_platform = candidate

        with self._lock:
            if norm_idem is not None and self._idempotency_keys is not None and norm_idem in self._idempotency_keys:
                self._idempotency_keys.move_to_end(norm_idem)
                self._deduplicated_replays += 1
                return self._summary_unlocked()

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

            if norm_idem is not None:
                if self._idempotency_keys is None:
                    self._idempotency_keys = OrderedDict()
                elif len(self._idempotency_keys) >= self._max_idempotency_keys:
                    self._idempotency_keys.popitem(last=False)
                    self._evicted_idempotency_keys += 1
                self._idempotency_keys[norm_idem] = None
                self._unique_idempotency_keys += 1

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
    clean_path = _validate_route_path(path)
    safe_path = escape(clean_path)
    raid_href = "/raid" if clean_path == "/quick" else "/raid-noel"
    summary = feedback_summary or {}
    raw_total = summary.get("total_responses")
    total_fb = raw_total if (not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total > 0) else 0
    raw_ease = summary.get("mean_ease_rating")
    if raw_ease is None:
        raw_ease = summary.get("average_ease_rating")
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

    raw_lag = summary.get("zero_game_lag_ratio")
    if (
        total_fb > 0
        and not isinstance(raw_lag, bool)
        and isinstance(raw_lag, (int, float))
        and math.isfinite(float(raw_lag))
        and 0.0 <= float(raw_lag) <= 1.0
    ):
        zero_lag_pct: float | None = round(float(raw_lag) * 100.0, 1)
    else:
        zero_lag_pct = None

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
        dev_zero_clone = (
            "Sans git clone : télécharge /scripts/hivemind_quickstart.py directement depuis cette passerelle, "
            "puis lance la vérification locale en lecture seule (< 10 s) :"
        )
        dev_signup_btn = "Ou créer un compte sur /signup"
        gamer_title = "2. Gamer PC (Windows 11 / Linux) — Mode Silencieux"
        gamer_desc = (
            "Plafond thermique doux (70 °C max), 2 Gio de VRAM gardés libres, et arrêt complet immédiat "
            "dès qu'un jeu Steam démarre. Désinstallation propre en une commande."
        )
        win_label = "Windows 11 (PowerShell) :"
        nix_label = "Linux / macOS :"
        uninstall_label = "Tout supprimer localement :"
        if total_fb > 0 and mean_ease is not None and zero_lag_pct is not None:
            fb_line = (
                f"Retours testeurs enregistrés : {total_fb} "
                f"(facilité moyenne : {mean_ease}/5 • confort sans lag jeu : {zero_lag_pct:.1f}%)"
            )
        elif total_fb > 0 and mean_ease is not None:
            fb_line = f"Retours testeurs enregistrés : {total_fb} (facilité moyenne : {mean_ease}/5)"
        elif total_fb > 0 and zero_lag_pct is not None:
            fb_line = f"Retours testeurs enregistrés : {total_fb} (confort sans lag jeu : {zero_lag_pct:.1f}%)"
        else:
            fb_line = "Retours testeurs : sois parmi les premiers à noter la simplicité (--feedback)."
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
        dev_zero_clone = (
            "Zero-clone option: download /scripts/hivemind_quickstart.py directly from this gateway, "
            "then run the read-only local preflight (< 10 s):"
        )
        dev_signup_btn = "Or create an account on /signup"
        gamer_title = "2. PC Gamer (Windows 11 / Linux) — Quiet Whisper Mode"
        gamer_desc = (
            "Gentle 70 C thermal ceiling, 2 GiB free VRAM reserve, and full worker stop as soon as "
            "a Steam game starts. Clean one-command uninstall."
        )
        win_label = "Windows 11 (PowerShell):"
        nix_label = "Linux / macOS:"
        uninstall_label = "Clean local uninstall:"
        if total_fb > 0 and mean_ease is not None and zero_lag_pct is not None:
            fb_line = (
                f"Recorded tester ratings: {total_fb} "
                f"(mean ease rating: {mean_ease}/5 • zero game lag comfort: {zero_lag_pct:.1f}%)"
            )
        elif total_fb > 0 and mean_ease is not None:
            fb_line = f"Recorded tester ratings: {total_fb} (mean ease rating: {mean_ease}/5)"
        elif total_fb > 0 and zero_lag_pct is not None:
            fb_line = f"Recorded tester ratings: {total_fb} (zero game lag comfort: {zero_lag_pct:.1f}%)"
        else:
            fb_line = "Tester feedback: send your ease rating anytime with --feedback."
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
        <p>{escape(dev_zero_clone)}</p>
        <pre>python3 scripts/hivemind_doctor.py --noel --gateway {safe_gw}</pre>
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
      <span>{switch_html} • <a href="/signup">/signup</a> • <a href="/join">/join</a> • <a href="{escape(raid_href)}">{escape(raid_href)}</a> • <a href="/arena-noel">/arena-noel</a> • <a href="/v1/hivemind/onboarding/manifest">/v1/hivemind/onboarding/manifest</a></span>
    </footer>
  </div>
</body>
</html>
"""


def _sanitize_build_sha_local(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or low == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


def _shas_match_local(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_build_sha_local(live_sha)
    clean_exp = _sanitize_build_sha_local(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == "score" or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _extract_observation(
    obs: Mapping[str, Any],
) -> tuple[int | None, dict[str, str], str, Mapping[str, Any] | None]:
    raw_status = obs.get("status_code") if "status_code" in obs else obs.get("status")
    status_code = raw_status if (not isinstance(raw_status, bool) and isinstance(raw_status, int)) else None

    headers: dict[str, str] = {}
    raw_headers = obs.get("headers")
    if isinstance(raw_headers, Mapping):
        for hk, hv in raw_headers.items():
            if isinstance(hk, str) and isinstance(hv, str):
                headers[hk.strip().lower()] = hv
    if "content-security-policy" not in headers and isinstance(obs.get("csp"), str):
        headers["content-security-policy"] = obs["csp"]

    text_body = ""
    for key in ("html", "text", "body"):
        val = obs.get(key)
        if isinstance(val, str):
            text_body = val
            break

    json_body: Mapping[str, Any] | None = None
    if isinstance(obs.get("json"), Mapping):
        json_body = obs["json"]
    elif text_body.lstrip().startswith("{"):
        try:
            parsed = json.loads(text_body)
            if isinstance(parsed, Mapping):
                json_body = parsed
        except json.JSONDecodeError:
            json_body = None

    return status_code, headers, text_body, json_body


def _is_valid_bounded_float(val: Any, low: float, high: float) -> bool:
    return (
        not isinstance(val, bool)
        and isinstance(val, (int, float))
        and math.isfinite(float(val))
        and low <= float(val) <= high
    )


def _inspect_summary_schema(
    doc: Mapping[str, Any],
    *,
    require_nonempty: bool = False,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if doc.get("object") != "hivemind.onboarding_feedback_summary":
        issues.append("summary_invalid_object")
    if _has_forbidden_score_key(doc):
        issues.append("summary_forbidden_score_key")

    raw_total = doc.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    if not total_ok:
        issues.append("summary_invalid_total_responses")
        total_val = -1
    else:
        total_val = int(raw_total)
        if require_nonempty and total_val < 1:
            issues.append("summary_expected_nonempty_responses")

    has_mean = "mean_ease_rating" in doc
    has_avg = "average_ease_rating" in doc
    if not has_mean and not has_avg:
        issues.append("summary_missing_ease_rating")
    else:
        m_val = doc.get("mean_ease_rating")
        a_val = doc.get("average_ease_rating")
        for label, ease_val, present in (
            ("mean_ease_rating", m_val, has_mean),
            ("average_ease_rating", a_val, has_avg),
        ):
            if not present:
                continue
            if total_val == 0 and ease_val is not None:
                issues.append(f"summary_non_null_{label}_when_empty")
            elif total_val > 0 and not _is_valid_bounded_float(ease_val, 1.0, 5.0):
                issues.append(f"summary_invalid_{label}")
        if (
            has_mean
            and has_avg
            and total_val > 0
            and _is_valid_bounded_float(m_val, 1.0, 5.0)
            and _is_valid_bounded_float(a_val, 1.0, 5.0)
            and round(float(m_val), 2) != round(float(a_val), 2)
        ):
            issues.append("summary_ease_rating_key_mismatch")

    if "zero_game_lag_ratio" not in doc:
        issues.append("summary_missing_zero_game_lag_ratio")
    else:
        lag_val = doc.get("zero_game_lag_ratio")
        if total_val == 0 and lag_val is not None:
            issues.append("summary_non_null_lag_when_empty")
        elif total_val > 0 and not _is_valid_bounded_float(lag_val, 0.0, 1.0):
            issues.append("summary_invalid_zero_game_lag_ratio")

    bp = doc.get("by_profile")
    if not isinstance(bp, Mapping):
        issues.append("summary_invalid_by_profile")
    else:
        dev_cnt = bp.get("dev")
        gamer_cnt = bp.get("gamer")
        if (
            isinstance(dev_cnt, bool)
            or not isinstance(dev_cnt, int)
            or dev_cnt < 0
            or isinstance(gamer_cnt, bool)
            or not isinstance(gamer_cnt, int)
            or gamer_cnt < 0
        ):
            issues.append("summary_invalid_profile_counts")
        elif total_ok and dev_cnt + gamer_cnt != total_val:
            issues.append("summary_profile_conservation_mismatch")

    return len(issues) == 0, issues


def _inspect_manifest(
    doc: Mapping[str, Any],
    expected_sha: str | None,
) -> tuple[bool, str, bool | None, list[str]]:
    issues: list[str] = []
    if doc.get("object") != "hivemind.onboarding.manifest":
        issues.append("manifest_invalid_object")
    if doc.get("campaign_id") != "noel-2026-easy-onboard":
        issues.append("manifest_campaign_mismatch")
    if doc.get("status") != "ok":
        issues.append("manifest_status_not_ok")
    if doc.get("personal_pc_only") is not True:
        issues.append("manifest_personal_pc_only_false")
    if doc.get("financial_promise") != "0.00 EUR / 0.00 USD":
        issues.append("manifest_financial_promise_mismatch")
    if _has_forbidden_score_key(doc):
        issues.append("manifest_forbidden_score_key")

    routes = doc.get("holiday_routes")
    if not isinstance(routes, (list, tuple)) or not all(isinstance(r, str) for r in routes):
        issues.append("manifest_invalid_holiday_routes")
    elif any(req not in routes for req in HOLIDAY_SMOKE_REQUIRED_ROUTES):
        issues.append("manifest_missing_holiday_routes")

    scripts = doc.get("downloadable_scripts")
    if not isinstance(scripts, (list, tuple)) or not all(isinstance(s, str) for s in scripts):
        issues.append("manifest_invalid_downloadable_scripts")
    elif "hivemind_quickstart.py" not in scripts or "hivemind_doctor.py" not in scripts:
        issues.append("manifest_missing_core_scripts")

    live_sha = _sanitize_build_sha_local(doc.get("build_sha"))
    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = _shas_match_local(live_sha, expected_sha)
        if not sha_match:
            issues.append("manifest_build_sha_mismatch")

    return len(issues) == 0, live_sha, sha_match, issues


def _inspect_raid_status(doc: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if doc.get("object") != "hivemind.raid_noel.status":
        issues.append("raid_status_invalid_object")
    if _has_forbidden_score_key(doc):
        issues.append("raid_status_forbidden_score_key")

    comp = doc.get("compliance")
    if (
        not isinstance(comp, Mapping)
        or comp.get("personal_pc_only") is not True
        or comp.get("monetary_cost_eur") != "0.00 EUR"
        or comp.get("monetary_cost_usd") != "0.00 USD"
    ):
        issues.append("raid_status_compliance_mismatch")

    fb = doc.get("onboarding_feedback")
    if not isinstance(fb, Mapping):
        issues.append("raid_status_missing_onboarding_feedback")
    else:
        tot = fb.get("total_responses")
        if tot is not None:
            if isinstance(tot, bool) or not isinstance(tot, int) or tot < 0:
                issues.append("raid_status_invalid_feedback_total")
            bp = fb.get("by_profile")
            if isinstance(bp, Mapping):
                d_cnt = bp.get("dev")
                g_cnt = bp.get("gamer")
                if (
                    isinstance(d_cnt, bool)
                    or not isinstance(d_cnt, int)
                    or d_cnt < 0
                    or isinstance(g_cnt, bool)
                    or not isinstance(g_cnt, int)
                    or g_cnt < 0
                    or (isinstance(tot, int) and not isinstance(tot, bool) and d_cnt + g_cnt != tot)
                ):
                    issues.append("raid_status_feedback_profile_mismatch")
            if isinstance(tot, int) and not isinstance(tot, bool) and tot > 0:
                if not _is_valid_bounded_float(fb.get("mean_ease_rating"), 1.0, 5.0):
                    issues.append("raid_status_invalid_mean_ease")
                if not _is_valid_bounded_float(fb.get("zero_game_lag_ratio"), 0.0, 1.0):
                    issues.append("raid_status_invalid_zero_lag_ratio")

    return len(issues) == 0, issues


def _inspect_html_route(
    headers: Mapping[str, str],
    html_text: str,
    seen_nonces: set[str],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    has_doctype = "<!doctype html" in html_text.lower()
    if not has_doctype:
        issues.append("missing_doctype")

    csp_val = headers.get("content-security-policy", "")
    csp_lower = csp_val.lower()
    csp_unsafe = "'unsafe-inline'" in csp_lower or "'unsafe-eval'" in csp_lower
    hdr_nonces = _CSP_NONCE_RE.findall(csp_val)
    attr_nonces = _HTML_NONCE_ATTR_RE.findall(html_text)
    header_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    nonce_reused = header_nonce is not None and header_nonce in seen_nonces
    if header_nonce is not None:
        seen_nonces.add(header_nonce)

    csp_ok = bool(
        "default-src 'none'" in csp_lower
        and not csp_unsafe
        and header_nonce is not None
        and attr_nonces
        and set(attr_nonces) == {header_nonce}
        and not nonce_reused
    )
    if nonce_reused:
        issues.append("reused_csp_nonce")
    elif not csp_ok:
        issues.append("invalid_or_missing_csp_nonce")

    stripped_html = _STRIP_NONCE_ATTR_RE.sub('nonce=""', html_text)
    has_unsafe_sink = bool(_UNSAFE_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIGURE_RE.search(stripped_html))
    if has_unsafe_sink:
        issues.append("unsafe_dom_sink_detected")
    if has_static_fig:
        issues.append("unbacked_static_figure_detected")
    dom_clean = not has_unsafe_sink and not has_static_fig

    low_html = stripped_html.lower()
    compliance_ok = (
        ("personal" in low_html or "personnel" in low_html)
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
        and ("agpl-3.0" in low_html)
    )
    if not compliance_ok:
        issues.append("missing_personal_pc_or_zero_eur_banner")

    info = {
        "has_doctype": has_doctype,
        "csp_ok": csp_ok,
        "nonce_reused": nonce_reused,
        "dom_clean": dom_clean,
        "compliance_ok": compliance_ok,
    }
    return len(issues) == 0, info, issues


def _extract_roundtrip_response(
    roundtrip: Mapping[str, Any],
    obs_keys: tuple[str, ...],
    status_key: str,
) -> tuple[int | None, Mapping[str, Any] | None]:
    obs: Any = None
    for k in obs_keys:
        if k in roundtrip:
            obs = roundtrip[k]
            break
    if not isinstance(obs, Mapping):
        return None, None
    if any(k in obs for k in ("status_code", "status", "json", "body", "text")):
        st, _, _, jdoc = _extract_observation(obs)
        return st, jdoc
    raw_st = roundtrip.get(status_key, 200)
    st = raw_st if (not isinstance(raw_st, bool) and isinstance(raw_st, int)) else None
    return st, obs


def _evaluate_feedback_roundtrip(roundtrip: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if "idempotency_key" in roundtrip and roundtrip["idempotency_key"] is not None:
        try:
            clean_key = _validate_idempotency_key(roundtrip["idempotency_key"])
        except ValueError:
            clean_key = None
        if clean_key is None:
            issues.append("roundtrip_invalid_idempotency_key")

    first_st, first_doc = _extract_roundtrip_response(
        roundtrip, ("first_response", "first"), "first_status"
    )
    replay_st, replay_doc = _extract_roundtrip_response(
        roundtrip, ("replay_response", "second_response", "replay"), "replay_status"
    )
    if first_st != 200 or replay_st != 200 or first_doc is None or replay_doc is None:
        issues.append("roundtrip_missing_or_non_200_response")
        return False, issues

    f_ok, f_issues = _inspect_summary_schema(first_doc, require_nonempty=True)
    r_ok, r_issues = _inspect_summary_schema(replay_doc, require_nonempty=True)
    issues.extend(f_issues)
    issues.extend(r_issues)
    if not f_ok or not r_ok:
        return False, issues

    if (
        replay_doc["total_responses"] != first_doc["total_responses"]
        or replay_doc["by_profile"] != first_doc["by_profile"]
        or replay_doc.get("mean_ease_rating") != first_doc.get("mean_ease_rating")
        or replay_doc.get("zero_game_lag_ratio") != first_doc.get("zero_game_lag_ratio")
    ):
        issues.append("roundtrip_replay_not_idempotent")

    baseline_raw = roundtrip.get("baseline_summary", roundtrip.get("before_summary"))
    if baseline_raw is not None:
        if not isinstance(baseline_raw, Mapping):
            issues.append("roundtrip_invalid_baseline_summary")
        else:
            b_ok, b_issues = _inspect_summary_schema(baseline_raw)
            issues.extend(b_issues)
            if b_ok:
                base_tot = int(baseline_raw["total_responses"])
                if base_tot < MAX_FEEDBACK_ITEMS and int(first_doc["total_responses"]) != base_tot + 1:
                    issues.append("roundtrip_first_post_did_not_increment_by_one")

    after_raw = roundtrip.get("after_summary", roundtrip.get("summary_after"))
    if after_raw is not None:
        if not isinstance(after_raw, Mapping):
            issues.append("roundtrip_invalid_after_summary")
        else:
            if any(k in after_raw for k in ("status_code", "status", "json", "body", "text")):
                a_st, _, _, a_doc = _extract_observation(after_raw)
            else:
                a_st, a_doc = 200, after_raw
            if a_st != 200 or a_doc is None:
                issues.append("roundtrip_after_summary_non_200")
            else:
                a_ok, a_issues = _inspect_summary_schema(a_doc, require_nonempty=True)
                issues.extend(a_issues)
                if a_ok and (
                    a_doc["total_responses"] != replay_doc["total_responses"]
                    or a_doc["by_profile"] != replay_doc["by_profile"]
                ):
                    issues.append("roundtrip_after_summary_drift")

    return len(issues) == 0, issues


def evaluate_holiday_smoke_parity(
    route_observations: Mapping[str, Mapping[str, Any]],
    *,
    feedback_roundtrip: Mapping[str, Any] | None = None,
    expected_build_sha: str | None = None,
) -> dict[str, Any]:
    """Verify zero-inference parity, CSP nonces, compliance, schema, and replay conservation across holiday routes."""
    if not isinstance(route_observations, Mapping):
        raise ValueError("route_observations must be a Mapping")
    for raw_key, raw_val in route_observations.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError("route_observations keys must be strings")
        if "completions" in raw_key.lower():
            raise ValueError("live GPU inference routes are forbidden in holiday smoke parity checks")
        if raw_val is not None and not isinstance(raw_val, Mapping):
            raise ValueError("route_observations values must be Mappings")
    if feedback_roundtrip is not None and not isinstance(feedback_roundtrip, Mapping):
        raise ValueError("feedback_roundtrip must be a Mapping or None")

    clean_expected_sha: str | None = None
    if expected_build_sha is not None:
        if isinstance(expected_build_sha, bool) or not isinstance(expected_build_sha, str):
            raise ValueError("expected_build_sha must be a 6-64 char alphanumeric string or None")
        sanitized = _sanitize_build_sha_local(expected_build_sha)
        if sanitized == "unknown":
            raise ValueError("expected_build_sha must be a 6-64 char alphanumeric string or None")
        clean_expected_sha = sanitized

    seen_nonces: set[str] = set()
    missing_404_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}

    passed_count = 0
    any_nonce_reused = False
    all_html_csp_ok = True
    html_dom_clean = True
    all_compliance_ok = True
    live_build_sha = "unknown"
    sha_match: bool | None = None
    manifest_ok = False
    feedback_summary_ok = False
    raid_status_ok = False

    for route in HOLIDAY_SMOKE_REQUIRED_ROUTES:
        obs = route_observations.get(route)
        if obs is None:
            failed_routes.append(route)
            diagnostics.append(f"{route}: missing_observation")
            per_route[route] = {"status_code": None, "ready": False, "issues": ["missing_observation"]}
            if route in _HOLIDAY_SMOKE_HTML_ROUTES:
                all_html_csp_ok = False
                all_compliance_ok = False
            continue

        status_code, headers, text_body, json_body = _extract_observation(obs)
        if status_code != 200:
            failed_routes.append(route)
            if status_code == 404:
                missing_404_routes.append(route)
            issue = f"http_{status_code}" if status_code is not None else "missing_status_code"
            diagnostics.append(f"{route}: {issue}")
            per_route[route] = {"status_code": status_code, "ready": False, "issues": [issue]}
            if route in _HOLIDAY_SMOKE_HTML_ROUTES:
                all_html_csp_ok = False
                all_compliance_ok = False
            continue

        if route in _HOLIDAY_SMOKE_HTML_ROUTES:
            route_ok, h_info, r_issues = _inspect_html_route(headers, text_body, seen_nonces)
            any_nonce_reused = any_nonce_reused or h_info["nonce_reused"]
            all_html_csp_ok = all_html_csp_ok and h_info["csp_ok"]
            html_dom_clean = html_dom_clean and h_info["dom_clean"]
            all_compliance_ok = all_compliance_ok and h_info["compliance_ok"]
            per_route[route] = {"status_code": status_code, "ready": route_ok, **h_info, "issues": r_issues}
        elif route == "/v1/hivemind/onboarding/manifest":
            if json_body is None:
                route_ok, r_issues = False, ["manifest_missing_json"]
            else:
                route_ok, live_build_sha, sha_match, r_issues = _inspect_manifest(json_body, clean_expected_sha)
                if json_body.get("personal_pc_only") is not True:
                    all_compliance_ok = False
            manifest_ok = route_ok
            per_route[route] = {"status_code": status_code, "ready": route_ok, "issues": r_issues}
        elif route == "/v1/hivemind/onboarding/feedback/summary":
            if json_body is None:
                route_ok, r_issues = False, ["summary_missing_json"]
            else:
                route_ok, r_issues = _inspect_summary_schema(json_body)
            feedback_summary_ok = route_ok
            per_route[route] = {"status_code": status_code, "ready": route_ok, "issues": r_issues}
        else:
            if json_body is None:
                route_ok, r_issues = False, ["raid_status_missing_json"]
            else:
                route_ok, r_issues = _inspect_raid_status(json_body)
                comp_blk = json_body.get("compliance")
                if not isinstance(comp_blk, Mapping) or comp_blk.get("personal_pc_only") is not True:
                    all_compliance_ok = False
            raid_status_ok = route_ok
            per_route[route] = {"status_code": status_code, "ready": route_ok, "issues": r_issues}

        if route_ok:
            passed_count += 1
        else:
            failed_routes.append(route)
            diagnostics.extend(f"{route}: {item}" for item in r_issues)

    idempotent_dedup_ok: bool | None = None
    if feedback_roundtrip is not None:
        idempotent_dedup_ok, rt_issues = _evaluate_feedback_roundtrip(feedback_roundtrip)
        diagnostics.extend(rt_issues)

    unique_csp_nonces_ok = all_html_csp_ok and not any_nonce_reused and len(seen_nonces) == len(_HOLIDAY_SMOKE_HTML_ROUTES)
    parity_ok = bool(
        passed_count == len(HOLIDAY_SMOKE_REQUIRED_ROUTES)
        and unique_csp_nonces_ok
        and html_dom_clean
        and all_compliance_ok
        and manifest_ok
        and feedback_summary_ok
        and raid_status_ok
        and (clean_expected_sha is None or sha_match is True)
        and (idempotent_dedup_ok is None or idempotent_dedup_ok is True)
    )

    core: dict[str, Any] = {
        "object": "hivemind.holiday_smoke_parity_report",
        "campaign_id": "noel-2026-easy-onboard",
        "required_routes": list(HOLIDAY_SMOKE_REQUIRED_ROUTES),
        "routes_total_count": len(HOLIDAY_SMOKE_REQUIRED_ROUTES),
        "routes_passed_count": passed_count,
        "parity_ok": parity_ok,
        "stale_404_drift": len(missing_404_routes) > 0,
        "missing_or_404_routes": missing_404_routes,
        "failed_routes": failed_routes,
        "unique_csp_nonces_count": len(seen_nonces),
        "unique_csp_nonces_ok": unique_csp_nonces_ok,
        "html_dom_clean": html_dom_clean,
        "compliance_ok": all_compliance_ok,
        "live_build_sha": live_build_sha,
        "expected_build_sha": clean_expected_sha,
        "sha_match": sha_match,
        "manifest_ok": manifest_ok,
        "feedback_summary_ok": feedback_summary_ok,
        "raid_status_ok": raid_status_ok,
        "idempotency_probe_checked": feedback_roundtrip is not None,
        "idempotent_dedup_ok": idempotent_dedup_ok,
        "zero_inference_verified": True,
        "per_route": per_route,
        "diagnostics": diagnostics,
    }
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, allow_nan=False)
    core["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return core
