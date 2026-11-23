# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Winter GPU Raid & Holiday Guild scoreboard (`/raid-noel`, `/raid`) and honest Wh electricity estimator.

Provides a bilingual French (`/raid-noel` default) and English (`/raid` default) page plus the JSON status builder
for `GET /v1/hivemind/raid/status`:
* Live swarm trainer, producer, registered account, and accepted OpenTrainer round readings (or `'n/a'` when absent).
* Symbolic `0.00 EUR / 0.00 USD` friendly holiday guild check-in tags (`Team Python`, `Team Rust/C++`,
  `Team Steam RPG`, `Team Night Owl`) with zero monetary stakes.
* Honest interactive client-side electricity transparency calculator (`GPU watts * hours * kWh tariff -> Wh & EUR`).
* Live anonymous onboarding ease rating and zero-game-lag comfort summary.
* Strict Amadeus / Personal-PC compliance rule (100% personal home PCs & personal open-source/hobby projects only).
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from hivemind.router.telemetry_figures import NOT_AVAILABLE, shown_figure

DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
SUPPORTED_RAID_LANGS: tuple[str, ...] = ("fr", "en")
VALID_CES_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
DEFAULT_CES_PLATFORM = "web-raid-noel"
PENDING_CES_STORAGE_KEY = "hivemind_pending_ces_ballot"
HOLIDAY_READINESS_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/raid/status",
    "/v1/hivemind/onboarding/feedback/summary",
)
_HTML_READINESS_ROUTES: frozenset[str] = frozenset({"/noel", "/quick", "/raid-noel"})
_NONCE_RE = re.compile(r"^[A-Za-z0-9_+=-]{8,80}$")
_CES_PLATFORM_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_FORBIDDEN_PLATFORM_TOKENS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "amadeus",
    "corporate",
    "email",
)

CES_COMPLIANCE_REMINDER_FR = (
    "Rappel conformité PC personnel uniquement : 100 % machine personnelle à domicile et projets perso / open-source "
    "(jamais de poste ni code d'entreprise) • 0,00 EUR / 0,00 USD (0.00 EUR / 0.00 USD) • Zéro PII (aucun champ texte libre)."
)
CES_COMPLIANCE_REMINDER_EN = (
    "Personal-PC-only compliance reminder: 100% personal home PCs and hobby/open-source projects ONLY "
    "(never corporate laptops or employer code/data) • 0.00 EUR / 0.00 USD • Zero PII (no free-text field)."
)

PERSONAL_PC_RULE_FR = (
    "Règle stricte d'intégrité personnelle : usage 100 % sur PC personnel à domicile et projets personnels / open-source "
    "uniquement (jamais sur un poste professionnel, un réseau d'entreprise, ni avec du code ou des données d'employeur). "
    "Projet citoyen AGPL-3.0 • 0,00 EUR / 0,00 USD."
)
PERSONAL_PC_RULE_EN = (
    "Strict Personal-PC Compliance Rule: 100% personal home PCs and personal hobby/open-source projects ONLY "
    "(never on corporate laptops, employer networks, or with company code/data). "
    "Citizen-science AGPL-3.0 • 0.00 EUR / 0.00 USD."
)

FRIENDLY_GUILDS: tuple[dict[str, str], ...] = (
    {
        "guild_id": "team-python",
        "label": "Team Python",
        "symbolic_fee": "0.00 EUR",
        "monetary_prize": "0.00 EUR",
        "motto_fr": "Scripts perso, notebooks scientifiques et extensions VS Code pendant les vacances.",
        "motto_en": "Personal scripts, science notebooks, and VS Code coding over the winter break.",
    },
    {
        "guild_id": "team-rust-cpp",
        "label": "Team Rust/C++",
        "symbolic_fee": "0.00 EUR",
        "monetary_prize": "0.00 EUR",
        "motto_fr": "Kernels systèmes, moteurs compilés et code bas niveau sur machine personnelle.",
        "motto_en": "Systems kernels, native engines, and low-level personal projects at home.",
    },
    {
        "guild_id": "team-steam-rpg",
        "label": "Team Steam RPG",
        "symbolic_fee": "0.00 EUR",
        "monetary_prize": "0.00 EUR",
        "motto_fr": "Mode silencieux Gamer-Whisper : libération complète de la VRAM (< 1.5 s) dès qu'un jeu démarre.",
        "motto_en": "Quiet Gamer-Whisper mode: instant (< 1.5 s) full VRAM release as soon as any game launches.",
    },
    {
        "guild_id": "team-night-owl",
        "label": "Team Night Owl",
        "symbolic_fee": "0.00 EUR",
        "monetary_prize": "0.00 EUR",
        "motto_fr": "Rondes OpenTrainer nocturnes douces pendant le sommeil (plafond thermique 70 °C).",
        "motto_en": "Gentle overnight OpenTrainer rounds while asleep (70 C thermal ceiling).",
    },
)

ELECTRICITY_PRESETS_WATTS: tuple[dict[str, Any], ...] = (
    {
        "id": "eco",
        "watts": 45,
        "label_fr": "Laptop / Eco (45 W)",
        "label_en": "Laptop / Eco (45 W)",
    },
    {
        "id": "whisper",
        "watts": 120,
        "label_fr": "Gamer-Whisper 70 °C (120 W)",
        "label_en": "Gamer-Whisper 70 C (120 W)",
    },
    {
        "id": "desktop",
        "watts": 220,
        "label_fr": "PC Fixe Plein Régime (220 W)",
        "label_en": "Desktop Full Load (220 W)",
    },
)

DEFAULT_GPU_WATTS = 120.0
DEFAULT_HOURS = 4.0
DEFAULT_KWH_TARIFF_EUR = 0.25


def raid_noel_csp(nonce: str) -> str:
    """Return a strict per-request nonce-pinned Content-Security-Policy for `/raid-noel` and `/raid`."""
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must be 8-80 URL-safe / base64 characters")
    return (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        f"script-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )


def pick_raid_language(
    query_lang: str | None,
    accept_language: str | None,
    default: str = "fr",
) -> str:
    """Resolve `'fr'` or `'en'` from `?lang=`, `Accept-Language`, or `default`."""
    if query_lang is not None and (isinstance(query_lang, bool) or not isinstance(query_lang, str)):
        raise ValueError("query_lang must be a string or None")
    if accept_language is not None and (isinstance(accept_language, bool) or not isinstance(accept_language, str)):
        raise ValueError("accept_language must be a string or None")
    if isinstance(default, bool) or not isinstance(default, str) or default not in SUPPORTED_RAID_LANGS:
        raise ValueError("default must be 'fr' or 'en'")

    if query_lang:
        q_primary = query_lang.strip().lower().split("-")[0].split("_")[0]
        if q_primary in SUPPORTED_RAID_LANGS:
            return q_primary

    best: tuple[float, int, str] | None = None
    for position, part in enumerate((accept_language or "").split(",")):
        tag, _, params = part.strip().partition(";")
        primary = tag.strip().lower().split("-")[0]
        if primary not in SUPPORTED_RAID_LANGS:
            continue
        quality = 1.0
        if "q" in params.lower():
            match = re.fullmatch(r"\s*q\s*=\s*([01](?:\.[0-9]{0,3})?)\s*", params, re.I)
            quality = float(match.group(1)) if match else 0.0
        if quality > 0 and (best is None or (quality, -position) > (best[0], -best[1])):
            best = (quality, position, primary)
    return best[2] if best else default


def estimate_gpu_electricity_cost(
    gpu_watts: float,
    hours: float,
    kwh_tariff_eur: float = DEFAULT_KWH_TARIFF_EUR,
) -> dict[str, float]:
    """Compute honest home-meter Wh, kWh, and EUR electricity estimate (`gpu_watts * hours * kwh_tariff_eur`)."""
    for name, val, upper in (
        ("gpu_watts", gpu_watts, 2000.0),
        ("hours", hours, 8760.0),
        ("kwh_tariff_eur", kwh_tariff_eur, 10.0),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite number")
        if float(val) < 0.0 or float(val) > upper:
            raise ValueError(f"{name} must be between 0 and {upper:g}")

    w = float(gpu_watts)
    h = float(hours)
    tariff = float(kwh_tariff_eur)
    energy_wh = w * h
    energy_kwh = energy_wh / 1000.0
    estimated_eur = energy_kwh * tariff
    return {
        "gpu_watts": round(w, 2),
        "hours": round(h, 2),
        "kwh_tariff_eur": round(tariff, 4),
        "energy_wh": round(energy_wh, 2),
        "energy_kwh": round(energy_kwh, 4),
        "estimated_eur": round(estimated_eur, 4),
    }


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) == int(value):
        return int(value)
    return None


def _opt_float(value: Any, decimals: int = 4) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), decimals)
    return None


def build_raid_noel_status(
    *,
    live_telemetry: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON payload for `GET /v1/hivemind/raid/status` from live router telemetry and feedback."""
    if live_telemetry is not None and not isinstance(live_telemetry, Mapping):
        raise ValueError("live_telemetry must be a Mapping or None")
    if feedback_summary is not None and not isinstance(feedback_summary, Mapping):
        raise ValueError("feedback_summary must be a Mapping or None")

    lt = live_telemetry or {}
    accounts = lt.get("accounts") if isinstance(lt.get("accounts"), Mapping) else {}
    producers = lt.get("producers") if isinstance(lt.get("producers"), Mapping) else {}
    opentrainer = lt.get("opentrainer") if isinstance(lt.get("opentrainer"), Mapping) else {}
    models = lt.get("models") if isinstance(lt.get("models"), Mapping) else {}

    fb = feedback_summary if feedback_summary is not None else {}
    raw_by_prof = fb.get("by_profile") if isinstance(fb.get("by_profile"), Mapping) else {}
    by_profile: dict[str, int | None] = {
        "dev": _opt_int(raw_by_prof.get("dev")) if feedback_summary is not None else None,
        "gamer": _opt_int(raw_by_prof.get("gamer")) if feedback_summary is not None else None,
    }

    default_estimate = estimate_gpu_electricity_cost(
        DEFAULT_GPU_WATTS,
        DEFAULT_HOURS,
        DEFAULT_KWH_TARIFF_EUR,
    )

    return {
        "object": "hivemind.raid_noel.status",
        "campaign": "winter-gpu-raid-holiday-guild",
        "supported_languages": list(SUPPORTED_RAID_LANGS),
        "compliance": {
            "personal_pc_only": True,
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
            "rule_fr": PERSONAL_PC_RULE_FR,
            "rule_en": PERSONAL_PC_RULE_EN,
        },
        "swarm": {
            "telemetry_status": str(lt.get("status")) if lt.get("status") else "UNAVAILABLE",
            "active_trainers": _opt_int(opentrainer.get("active_trainer_nodes")),
            "registered_trainer_nodes": _opt_int(opentrainer.get("total_registered_nodes")),
            "registered_trainer_accounts": _opt_int(opentrainer.get("registered_trainer_accounts_count")),
            "live_producers": _opt_int(producers.get("live_hiveminders")),
            "registered_producers": _opt_int(producers.get("total_hiveminders")),
            "registered_accounts": _opt_int(accounts.get("total_registered_accounts")),
            "active_accounts_24h": _opt_int(accounts.get("active_accounts_24h")),
            "accepted_opentrainer_rounds": _opt_int(opentrainer.get("accepted_rounds_count")),
            "live_models": _opt_int(models.get("live_models_count")),
        },
        "friendly_guilds": [dict(g) for g in FRIENDLY_GUILDS],
        "electricity_estimator": {
            "formula_wh": "gpu_watts * hours",
            "formula_kwh": "(gpu_watts * hours) / 1000",
            "formula_eur": "((gpu_watts * hours) / 1000) * kwh_tariff_eur",
            "default_estimate": default_estimate,
            "presets_watts": [dict(p) for p in ELECTRICITY_PRESETS_WATTS],
        },
        "onboarding_feedback": {
            "total_responses": _opt_int(fb.get("total_responses")) if feedback_summary is not None else None,
            "mean_ease_rating": _opt_float(fb.get("mean_ease_rating"), decimals=2),
            "zero_game_lag_ratio": _opt_float(fb.get("zero_game_lag_ratio"), decimals=4),
            "by_profile": by_profile,
        },
    }


def _validate_gateway_url(gateway_url: str) -> str:
    if not isinstance(gateway_url, str) or not gateway_url.strip():
        raise ValueError("gateway_url must be a non-empty http(s) URL")
    cleaned = gateway_url.strip().rstrip("/")
    if any(ch in cleaned for ch in (" ", "\t", "\r", "\n", '"', "'", "<", ">")):
        raise ValueError("gateway_url contains forbidden characters")
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("gateway_url must be a clean http(s) origin without credentials")
    return cleaned


def _format_zero_lag(ratio: float | None) -> str:
    if ratio is None:
        return NOT_AVAILABLE
    return f"{ratio * 100.0:.1f}%"


def _format_ease(ease: float | None) -> str:
    if ease is None:
        return NOT_AVAILABLE
    return f"{ease:.2f} / 5"


def build_one_click_ces_ballot(
    *,
    profile: str = "gamer",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_name: str = DEFAULT_CES_PLATFORM,
    note: str = "",
) -> dict[str, Any]:
    """Validate and normalize a strict zero-PII `OnboardingFeedbackRequest`-compatible 1-click ballot."""
    if isinstance(profile, bool) or not isinstance(profile, str) or len(profile) > 32 or not profile.isascii():
        raise ValueError("profile must be 'dev' or 'gamer'")
    norm_profile = profile.strip().lower()
    if norm_profile not in VALID_CES_PROFILES:
        raise ValueError("profile must be 'dev' or 'gamer'")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ValueError("ease_rating must be an integer between 1 and 5")
    if not isinstance(zero_game_lag, bool):
        raise ValueError("zero_game_lag must be a boolean")
    if (
        isinstance(platform_name, bool)
        or not isinstance(platform_name, str)
        or not platform_name.isascii()
        or len(platform_name) > 32
    ):
        raise ValueError("platform_name must be a 1-24 character ASCII platform slug")
    clean_platform = platform_name.strip().lower()
    if not _CES_PLATFORM_RE.fullmatch(clean_platform):
        raise ValueError("platform_name must match ^[a-z0-9][a-z0-9_-]{0,23}$")
    if clean_platform.startswith(("hm-", "hm_live_")) or any(
        tok in clean_platform for tok in _FORBIDDEN_PLATFORM_TOKENS
    ):
        raise ValueError("platform_name contains forbidden credential, PII, or corporate markers")
    if isinstance(note, bool) or not isinstance(note, str) or note != "":
        raise ValueError("note must be empty ('') to enforce strict zero-PII 1-click feedback")
    return {
        "profile": norm_profile,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": clean_platform,
        "note": "",
    }


def evaluate_holiday_route_readiness(
    route_probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Deterministically verify HTTP 200, CSP, and personal-PC 0.00 EUR/USD readiness across holiday routes."""
    if not isinstance(route_probes, Mapping):
        raise ValueError("route_probes must be a Mapping of route paths to probe dictionaries")
    for raw_key, raw_val in route_probes.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError("route_probes keys must be route path strings")
        if "completions" in raw_key.lower():
            raise ValueError("live GPU inference probes are forbidden in holiday route readiness checks")
        if raw_val is not None and not isinstance(raw_val, Mapping):
            raise ValueError("each entry in route_probes must be a Mapping")

    missing_or_stale: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}
    passed_count = 0

    for route in HOLIDAY_READINESS_ROUTES:
        probe = route_probes.get(route)
        if probe is None:
            missing_or_stale.append(route)
            per_route[route] = {
                "status_code": None,
                "http_200": False,
                "csp_present": False,
                "compliance_verified": False,
                "ready": False,
                "reason": "missing_probe",
            }
            continue

        raw_status = probe.get("status_code") if "status_code" in probe else probe.get("status")
        status_code = raw_status if (not isinstance(raw_status, bool) and isinstance(raw_status, int)) else None
        http_200 = status_code == 200

        raw_headers = probe.get("headers")
        norm_headers: dict[str, str] = {}
        if isinstance(raw_headers, Mapping):
            for hk, hv in raw_headers.items():
                if isinstance(hk, str) and isinstance(hv, str):
                    norm_headers[hk.strip().lower()] = hv
        csp_val = norm_headers.get("content-security-policy", "")
        if not csp_val and isinstance(probe.get("csp"), str):
            csp_val = probe["csp"]

        text_body = ""
        for bkey in ("body", "text", "html"):
            bval = probe.get(bkey)
            if isinstance(bval, str):
                text_body = bval
                break

        json_body: Mapping[str, Any] | None = None
        if isinstance(probe.get("json"), Mapping):
            json_body = probe["json"]
        elif text_body.lstrip().startswith("{"):
            try:
                parsed = json.loads(text_body)
                if isinstance(parsed, Mapping):
                    json_body = parsed
            except json.JSONDecodeError:
                json_body = None

        csp_unsafe = "'unsafe-inline'" in csp_val or "'unsafe-eval'" in csp_val
        if route in _HTML_READINESS_ROUTES:
            csp_ok = bool("default-src 'none'" in csp_val and "'nonce-" in csp_val and not csp_unsafe)
            lower_text = text_body.lower()
            has_personal = "personal" in lower_text or "personnel" in lower_text
            has_zero_cost = any(m in text_body for m in ("0.00 EUR", "0,00 EUR", "0.00 USD", "0,00 USD"))
            has_agpl = "agpl-3.0" in lower_text
            compliance_ok = bool(has_personal and has_zero_cost and has_agpl)
        elif route == "/v1/hivemind/raid/status":
            csp_ok = not csp_unsafe
            comp_block = json_body.get("compliance") if isinstance(json_body, Mapping) else None
            compliance_ok = bool(
                isinstance(json_body, Mapping)
                and json_body.get("object") == "hivemind.raid_noel.status"
                and isinstance(comp_block, Mapping)
                and comp_block.get("personal_pc_only") is True
                and comp_block.get("monetary_cost_eur") == "0.00 EUR"
                and comp_block.get("monetary_cost_usd") == "0.00 USD"
            )
        else:
            csp_ok = not csp_unsafe
            by_prof = json_body.get("by_profile") if isinstance(json_body, Mapping) else None
            tot_resp = json_body.get("total_responses") if isinstance(json_body, Mapping) else None
            dev_cnt = by_prof.get("dev") if isinstance(by_prof, Mapping) else None
            gamer_cnt = by_prof.get("gamer") if isinstance(by_prof, Mapping) else None
            compliance_ok = bool(
                isinstance(json_body, Mapping)
                and json_body.get("object", "hivemind.onboarding_feedback_summary")
                == "hivemind.onboarding_feedback_summary"
                and not isinstance(tot_resp, bool)
                and isinstance(tot_resp, int)
                and tot_resp >= 0
                and not isinstance(dev_cnt, bool)
                and isinstance(dev_cnt, int)
                and dev_cnt >= 0
                and not isinstance(gamer_cnt, bool)
                and isinstance(gamer_cnt, int)
                and gamer_cnt >= 0
            )

        csp_present = bool(csp_val and not csp_unsafe)
        route_ready = bool(http_200 and csp_ok and compliance_ok)
        if route_ready:
            passed_count += 1
            reason = "ok"
        elif not http_200:
            missing_or_stale.append(route)
            reason = f"http_{status_code}" if status_code is not None else "network_or_missing_status"
        elif not csp_ok:
            missing_or_stale.append(route)
            reason = "missing_or_weak_csp"
        else:
            missing_or_stale.append(route)
            reason = "missing_compliance_or_contract"

        per_route[route] = {
            "status_code": status_code,
            "http_200": http_200,
            "csp_present": csp_present,
            "compliance_verified": compliance_ok,
            "ready": route_ready,
            "reason": reason,
        }

    feedback_deployed = bool(per_route["/v1/hivemind/onboarding/feedback/summary"]["ready"])
    return {
        "object": "hivemind.raid_noel.holiday_route_readiness",
        "required_routes": list(HOLIDAY_READINESS_ROUTES),
        "routes_total_count": len(HOLIDAY_READINESS_ROUTES),
        "routes_passed_count": passed_count,
        "holiday_routes_ready": passed_count == len(HOLIDAY_READINESS_ROUTES) and not missing_or_stale,
        "onboarding_feedback_deployed": feedback_deployed,
        "missing_or_stale_routes": missing_or_stale,
        "zero_gpu_inference": True,
        "compliance": {
            "personal_pc_only": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
        },
        "per_route": per_route,
    }


# Client-side controller: strictly uses DOM textContent / value / aria-pressed (zero HTML string sinks).
_RAID_NOEL_JS = r"""(function () {
  var root = document.getElementById('raidNoelApp');
  var isFr = root && root.getAttribute('data-lang') === 'fr';
  var PENDING_CES_KEY = 'hivemind_pending_ces_ballot';
  var cesProfile = 'gamer';
  var cesEase = 5;
  var cesZeroLag = true;

  function byId(id) {
    return document.getElementById(id);
  }

  function fmtReading(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return String(v);
  }

  function fmtEase(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return n.toFixed(2) + ' / 5';
  }

  function fmtLagRatio(v) {
    if (v === null || v === undefined || typeof v === 'boolean') return 'n/a';
    var n = Number(v);
    if (!Number.isFinite(n)) return 'n/a';
    return (n * 100).toFixed(1) + '%';
  }

  function setText(id, val) {
    var el = byId(id);
    if (el) el.textContent = String(val);
  }

  function updateCalculator() {
    var wEl = byId('calcWatts');
    var hEl = byId('calcHours');
    var tEl = byId('calcTariff');
    if (!wEl || !hEl || !tEl) return;

    var wRaw = String(wEl.value || '').trim();
    var hRaw = String(hEl.value || '').trim();
    var tRaw = String(tEl.value || '').trim();
    var watts = wRaw === '' ? NaN : Number(wRaw);
    var hours = hRaw === '' ? NaN : Number(hRaw);
    var tariff = tRaw === '' ? NaN : Number(tRaw);

    if (
      !Number.isFinite(watts) || watts < 0 || watts > 2000 ||
      !Number.isFinite(hours) || hours < 0 || hours > 8760 ||
      !Number.isFinite(tariff) || tariff < 0 || tariff > 10
    ) {
      setText('calcResultWh', 'n/a');
      setText('calcResultEur', 'n/a');
      setText(
        'calcResultNote',
        isFr
          ? 'Valeurs attendues : 0–2000 W, 0–8760 h, tarif 0–10 EUR/kWh.'
          : 'Expected range: 0–2000 W, 0–8760 h, tariff 0–10 EUR/kWh.'
      );
      return;
    }

    var wh = watts * hours;
    var kwh = wh / 1000.0;
    var eur = kwh * tariff;
    setText('calcResultWh', wh.toFixed(1) + ' Wh (' + kwh.toFixed(3) + ' kWh)');
    setText('calcResultEur', eur.toFixed(2) + ' EUR');
    setText(
      'calcResultNote',
      isFr
        ? watts + ' W × ' + hours + ' h × ' + tariff.toFixed(2) + ' EUR/kWh = ' + wh.toFixed(1) + ' Wh (' + eur.toFixed(2) + ' EUR sur ton compteur électrique perso • 0,00 EUR versé à/par HiveMind).'
        : watts + ' W × ' + hours + ' h × ' + tariff.toFixed(2) + ' EUR/kWh = ' + wh.toFixed(1) + ' Wh (' + eur.toFixed(2) + ' EUR on your home meter • 0.00 EUR paid to/by HiveMind).'
    );
  }

  var guildMeta = {
    'team-python': 'Team Python',
    'team-rust-cpp': 'Team Rust/C++',
    'team-steam-rpg': 'Team Steam RPG',
    'team-night-owl': 'Team Night Owl'
  };

  function selectGuild(guildId, persist) {
    var label = guildMeta[guildId];
    if (!label) return;
    Object.keys(guildMeta).forEach(function (id) {
      var btn = byId('guildBtn-' + id);
      if (btn) {
        var active = id === guildId;
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.className = active ? 'guild-btn active' : 'guild-btn';
      }
    });
    if (persist) {
      try {
        if (window.localStorage) window.localStorage.setItem('hivemind_holiday_guild', guildId);
      } catch (_e) {}
    }
    setText(
      'guildCheckinStatus',
      isFr
        ? 'Check-in amical activé : ' + label + ' (tag symbolique 0,00 EUR / 0,00 USD — PC personnel uniquement).'
        : 'Friendly check-in active: ' + label + ' (symbolic 0.00 EUR / 0.00 USD tag — personal home PC only).'
    );
  }

  function applyFeedbackSummary(fb) {
    if (!fb || typeof fb !== 'object') return;
    setText('raidFeedbackResponses', fmtReading(fb.total_responses));
    setText('raidFeedbackEase', fmtEase(fb.mean_ease_rating));
    setText('raidFeedbackZeroLag', fmtLagRatio(fb.zero_game_lag_ratio));
    var prof = fb.by_profile || {};
    setText(
      'raidFeedbackProfiles',
      'dev: ' + fmtReading(prof.dev) + ' | gamer: ' + fmtReading(prof.gamer)
    );
  }

  function applyRaidStatusPayload(d) {
    if (!d || typeof d !== 'object') return;
    var sw = d.swarm || {};
    setText('raidLiveTrainers', fmtReading(sw.active_trainers));
    setText('raidRegisteredTrainers', fmtReading(sw.registered_trainer_accounts));
    setText('raidLiveProducers', fmtReading(sw.live_producers));
    setText('raidRegisteredProducers', fmtReading(sw.registered_producers));
    setText('raidRegisteredAccounts', fmtReading(sw.registered_accounts));
    setText('raidActiveAccounts24h', fmtReading(sw.active_accounts_24h));
    setText('raidAcceptedRounds', fmtReading(sw.accepted_opentrainer_rounds));
    setText('raidLiveModels', fmtReading(sw.live_models));
    applyFeedbackSummary(d.onboarding_feedback || {});
  }

  function refreshRaidStatus() {
    if (typeof fetch !== 'function') return Promise.resolve();
    setText(
      'raidLiveRefreshNote',
      isFr ? 'Actualisation en direct…' : 'Refreshing live readings…'
    );
    return fetch('/v1/hivemind/raid/status', { cache: 'no-store' })
      .then(function (r) {
        if (!r || !r.ok) throw new Error('status_' + (r ? r.status : 'err'));
        return r.json();
      })
      .then(function (d) {
        applyRaidStatusPayload(d);
        setText(
          'raidLiveRefreshNote',
          isFr
            ? 'Compteurs synchronisés en direct depuis /v1/hivemind/raid/status.'
            : 'Live counters synced from /v1/hivemind/raid/status.'
        );
      })
      .catch(function () {
        setText(
          'raidLiveRefreshNote',
          isFr
            ? 'Lecture hors-ligne : derniers compteurs affichés (aucun chiffre inventé).'
            : 'Offline view: showing last received readings (no figures invented).'
        );
      });
  }

  function fnv1aHex(str, seed) {
    var h = seed >>> 0;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return ('00000000' + h.toString(16)).slice(-8);
  }

  function buildCesIdempotencyKey(ballot) {
    var raw = ballot.profile + '|' + String(ballot.ease_rating) + '|' + (ballot.zero_game_lag ? '1' : '0') + '|' + ballot.platform;
    var hex = fnv1aHex(raw, 2166136261) + fnv1aHex(raw + '#raid-noel', 2166136261);
    return 'ces-' + ballot.profile + '-' + String(ballot.ease_rating) + '-' + (ballot.zero_game_lag ? 'lag0' : 'lag1') + '-' + hex;
  }

  function validateCesBallot(b) {
    if (!b || typeof b !== 'object') return null;
    var prof = b.profile === 'dev' ? 'dev' : (b.profile === 'gamer' ? 'gamer' : null);
    var ease = typeof b.ease_rating === 'number' && Number.isInteger(b.ease_rating) && b.ease_rating >= 1 && b.ease_rating <= 5 ? b.ease_rating : null;
    var lag = typeof b.zero_game_lag === 'boolean' ? b.zero_game_lag : null;
    if (!prof || ease === null || lag === null) return null;
    if (b.note !== undefined && b.note !== '') return null;
    return {
      profile: prof,
      ease_rating: ease,
      zero_game_lag: lag,
      platform: 'web-raid-noel',
      note: ''
    };
  }

  function buildCurrentCesBallot() {
    return {
      profile: cesProfile === 'dev' ? 'dev' : 'gamer',
      ease_rating: cesEase,
      zero_game_lag: Boolean(cesZeroLag),
      platform: 'web-raid-noel',
      note: ''
    };
  }

  function clearPendingCesBallot() {
    try {
      if (window.localStorage) {
        if (typeof window.localStorage.removeItem === 'function') {
          window.localStorage.removeItem(PENDING_CES_KEY);
        } else if (typeof window.localStorage.setItem === 'function') {
          window.localStorage.setItem(PENDING_CES_KEY, '');
        }
      }
    } catch (_e) {}
  }

  function savePendingCesBallot(ballot, idemKey, statusCode) {
    try {
      if (window.localStorage && typeof window.localStorage.setItem === 'function') {
        window.localStorage.setItem(
          PENDING_CES_KEY,
          JSON.stringify({
            ballot: ballot,
            idempotency_key: idemKey,
            last_status: statusCode
          })
        );
      }
    } catch (_e) {}
  }

  function loadPendingCesBallot() {
    try {
      if (!window.localStorage || typeof window.localStorage.getItem !== 'function') return null;
      var raw = window.localStorage.getItem(PENDING_CES_KEY);
      if (!raw || typeof raw !== 'string') return null;
      if (raw.length > 512) {
        clearPendingCesBallot();
        return null;
      }
      var parsed = JSON.parse(raw);
      var clean = validateCesBallot(parsed && parsed.ballot ? parsed.ballot : parsed);
      if (!clean) {
        clearPendingCesBallot();
        return null;
      }
      var key = (parsed && typeof parsed.idempotency_key === 'string' && /^[A-Za-z0-9_-]{8,80}$/.test(parsed.idempotency_key))
        ? parsed.idempotency_key
        : buildCesIdempotencyKey(clean);
      return { ballot: clean, idempotency_key: key };
    } catch (_e) {
      clearPendingCesBallot();
      return null;
    }
  }

  function cesSpoolMessage(statusCode) {
    if (statusCode === 404) {
      return isFr
        ? 'Route Cloud Run /v1/hivemind/onboarding/feedback non déployée (HTTP 404) : vote zéro-PII gardé localement. Clique sur « Réessayer l\'envoi en attente » après déploiement.'
        : 'Cloud Run route /v1/hivemind/onboarding/feedback not yet deployed (HTTP 404): zero-PII ballot saved locally. Click "Retry Pending Feedback" after deploy.';
    }
    if (statusCode === 429) {
      return isFr
        ? 'Limite de fréquence atteinte (HTTP 429) : vote zéro-PII gardé localement. Clique sur « Réessayer l\'envoi en attente » dans une minute.'
        : 'Rate limit reached (HTTP 429): zero-PII ballot saved locally. Click "Retry Pending Feedback" in a minute.';
    }
    if (statusCode >= 500) {
      return isFr
        ? 'Passerelle temporairement occupée (HTTP ' + statusCode + ') : vote zéro-PII gardé localement. Clique sur « Réessayer l\'envoi en attente ».'
        : 'Gateway temporarily busy (HTTP ' + statusCode + '): zero-PII ballot saved locally. Click "Retry Pending Feedback" to resend.';
    }
    return isFr
      ? 'Mode hors-ligne / erreur réseau : vote zéro-PII gardé localement. Clique sur « Réessayer l\'envoi en attente » dès le retour réseau.'
      : 'Offline / network error: zero-PII ballot saved locally. Click "Retry Pending Feedback" when back online.';
  }

  function sendCesBallot(ballot, idemKey, isRetry) {
    if (typeof fetch !== 'function') {
      savePendingCesBallot(ballot, idemKey, 0);
      setText('cesSubmitStatus', cesSpoolMessage(0));
      return Promise.resolve();
    }
    setText(
      'cesSubmitStatus',
      isFr ? 'Envoi du retour 1-clic zéro-PII…' : 'Sending 1-click zero-PII feedback…'
    );
    return fetch('/v1/hivemind/onboarding/feedback', {
      method: 'POST',
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': idemKey
      },
      body: JSON.stringify(ballot)
    })
      .then(function (r) {
        var code = r && typeof r.status === 'number' ? r.status : 0;
        if (r && r.ok) {
          return r.json().then(function (d) {
            clearPendingCesBallot();
            var fb = d && d.onboarding_feedback && typeof d.onboarding_feedback === 'object' ? d.onboarding_feedback : d;
            applyFeedbackSummary(fb);
            setText(
              'cesSubmitStatus',
              isRetry
                ? (isFr
                    ? 'Retour en attente synchronisé (HTTP 200 • zéro PII • 0,00 EUR / 0,00 USD).'
                    : 'Pending zero-PII feedback flushed (HTTP 200 • 0.00 EUR / 0.00 USD).')
                : (isFr
                    ? 'Merci ! Retour 1-clic zéro-PII enregistré en direct (HTTP 200 • 0,00 EUR / 0,00 USD).'
                    : 'Thank you! 1-click zero-PII feedback recorded live (HTTP 200 • 0.00 EUR / 0.00 USD).')
            );
          });
        }
        if (code === 404 || code === 408 || code === 429 || code >= 500 || code === 0) {
          savePendingCesBallot(ballot, idemKey, code);
          setText('cesSubmitStatus', cesSpoolMessage(code));
          return;
        }
        clearPendingCesBallot();
        setText(
          'cesSubmitStatus',
          isFr
            ? 'Envoi refusé (HTTP ' + code + ') : vote réinitialisé.'
            : 'Submission rejected (HTTP ' + code + '): ballot cleared.'
        );
      })
      .catch(function () {
        savePendingCesBallot(ballot, idemKey, 0);
        setText('cesSubmitStatus', cesSpoolMessage(0));
      });
  }

  function syncCesButtons() {
    var devBtn = byId('cesProfileDevBtn');
    var gamerBtn = byId('cesProfileGamerBtn');
    if (devBtn) {
      var isDev = cesProfile === 'dev';
      devBtn.setAttribute('aria-pressed', isDev ? 'true' : 'false');
      devBtn.className = isDev ? 'chip-btn active' : 'chip-btn';
    }
    if (gamerBtn) {
      var isGamer = cesProfile === 'gamer';
      gamerBtn.setAttribute('aria-pressed', isGamer ? 'true' : 'false');
      gamerBtn.className = isGamer ? 'chip-btn active' : 'chip-btn';
    }
    for (var rating = 1; rating <= 5; rating++) {
      var eBtn = byId('cesEaseBtn-' + rating);
      if (eBtn) {
        var activeEase = cesEase === rating;
        eBtn.setAttribute('aria-pressed', activeEase ? 'true' : 'false');
        eBtn.className = activeEase ? 'chip-btn active' : 'chip-btn';
      }
    }
    var lagYes = byId('cesLagYesBtn');
    var lagNo = byId('cesLagNoBtn');
    if (lagYes) {
      lagYes.setAttribute('aria-pressed', cesZeroLag ? 'true' : 'false');
      lagYes.className = cesZeroLag ? 'chip-btn active' : 'chip-btn';
    }
    if (lagNo) {
      lagNo.setAttribute('aria-pressed', !cesZeroLag ? 'true' : 'false');
      lagNo.className = !cesZeroLag ? 'chip-btn active' : 'chip-btn';
    }
  }

  ['calcWatts', 'calcHours', 'calcTariff'].forEach(function (id) {
    var el = byId(id);
    if (el) {
      el.addEventListener('input', updateCalculator);
      el.addEventListener('change', updateCalculator);
    }
  });

  var calcBtn = byId('calcComputeBtn');
  if (calcBtn) calcBtn.addEventListener('click', updateCalculator);

  [
    ['calcPresetEco', '45'],
    ['calcPresetWhisper', '120'],
    ['calcPresetDesktop', '220']
  ].forEach(function (pair) {
    var btn = byId(pair[0]);
    if (btn) {
      btn.addEventListener('click', function () {
        var wEl = byId('calcWatts');
        if (wEl) wEl.value = pair[1];
        updateCalculator();
      });
    }
  });

  Object.keys(guildMeta).forEach(function (guildId) {
    var btn = byId('guildBtn-' + guildId);
    if (btn) {
      btn.addEventListener('click', function () {
        selectGuild(guildId, true);
      });
    }
  });

  var refreshBtn = byId('refreshRaidBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', function () {
      return refreshRaidStatus();
    });
  }

  var cesDevEl = byId('cesProfileDevBtn');
  if (cesDevEl) {
    cesDevEl.addEventListener('click', function () {
      cesProfile = 'dev';
      syncCesButtons();
    });
  }
  var cesGamerEl = byId('cesProfileGamerBtn');
  if (cesGamerEl) {
    cesGamerEl.addEventListener('click', function () {
      cesProfile = 'gamer';
      syncCesButtons();
    });
  }
  [1, 2, 3, 4, 5].forEach(function (rating) {
    var eBtn = byId('cesEaseBtn-' + rating);
    if (eBtn) {
      eBtn.addEventListener('click', function () {
        cesEase = rating;
        syncCesButtons();
      });
    }
  });
  var cesLagYesEl = byId('cesLagYesBtn');
  if (cesLagYesEl) {
    cesLagYesEl.addEventListener('click', function () {
      cesZeroLag = true;
      syncCesButtons();
    });
  }
  var cesLagNoEl = byId('cesLagNoBtn');
  if (cesLagNoEl) {
    cesLagNoEl.addEventListener('click', function () {
      cesZeroLag = false;
      syncCesButtons();
    });
  }

  var submitCesEl = byId('submitCesBtn');
  if (submitCesEl) {
    submitCesEl.addEventListener('click', function () {
      var ballot = buildCurrentCesBallot();
      var idemKey = buildCesIdempotencyKey(ballot);
      return sendCesBallot(ballot, idemKey, false);
    });
    syncCesButtons();
    var existingPending = loadPendingCesBallot();
    if (existingPending) {
      setText(
        'cesSubmitStatus',
        isFr
          ? 'Un retour zéro-PII est en attente localement : clique sur « Réessayer l\'envoi en attente ».'
          : 'A zero-PII ballot is pending locally: click "Retry Pending Feedback" to send it.'
      );
    }
  }

  var retryCesEl = byId('retryPendingCesBtn');
  if (retryCesEl) {
    retryCesEl.addEventListener('click', function () {
      var pending = loadPendingCesBallot();
      if (!pending) {
        setText(
          'cesSubmitStatus',
          isFr
            ? 'Aucun retour zéro-PII en attente dans le stockage local.'
            : 'No pending zero-PII feedback ballot in local storage.'
        );
        return Promise.resolve();
      }
      return sendCesBallot(pending.ballot, pending.idempotency_key, true);
    });
  }

  try {
    var savedGuild = window.localStorage && window.localStorage.getItem('hivemind_holiday_guild');
    if (savedGuild && guildMeta[savedGuild]) selectGuild(savedGuild, false);
  } catch (_e) {}

  updateCalculator();
})();"""


def render_raid_noel_page(
    lang: str = "fr",
    nonce: str = "RaidNoelDefaultNonce2026",
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    path: str = "/raid-noel",
    live_telemetry: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
) -> str:
    """Render the strict-CSP bilingual (`/raid-noel` FR, `/raid` EN) Winter GPU Raid & Holiday Guild page."""
    if isinstance(lang, bool) or not isinstance(lang, str):
        raise ValueError("lang must be a string")
    raid_noel_csp(nonce)
    safe_gw = html.escape(_validate_gateway_url(gateway_url))
    if isinstance(path, bool) or not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must be a URL path starting with '/'")
    safe_path = html.escape(path)

    status_doc = build_raid_noel_status(
        live_telemetry=live_telemetry,
        feedback_summary=feedback_summary,
    )
    sw = status_doc["swarm"]
    fb = status_doc["onboarding_feedback"]
    default_est = status_doc["electricity_estimator"]["default_estimate"]

    is_fr = lang.strip().lower().startswith("fr")
    lang_code = "fr" if is_fr else "en"

    trainers_txt = html.escape(shown_figure(sw["active_trainers"]))
    reg_trainers_txt = html.escape(shown_figure(sw["registered_trainer_accounts"]))
    producers_txt = html.escape(shown_figure(sw["live_producers"]))
    reg_producers_txt = html.escape(shown_figure(sw["registered_producers"]))
    accounts_txt = html.escape(shown_figure(sw["registered_accounts"]))
    active_24h_txt = html.escape(shown_figure(sw["active_accounts_24h"]))
    rounds_txt = html.escape(shown_figure(sw["accepted_opentrainer_rounds"]))
    models_txt = html.escape(shown_figure(sw["live_models"]))

    fb_total_txt = html.escape(shown_figure(fb["total_responses"]))
    fb_ease_txt = html.escape(_format_ease(fb["mean_ease_rating"]))
    fb_lag_txt = html.escape(_format_zero_lag(fb["zero_game_lag_ratio"]))
    fb_profiles_txt = html.escape(
        f"dev: {shown_figure(fb['by_profile']['dev'])} | gamer: {shown_figure(fb['by_profile']['gamer'])}"
    )

    init_wh_txt = html.escape(
        f"{default_est['energy_wh']:.1f} Wh ({default_est['energy_kwh']:.3f} kWh)"
    )
    init_eur_txt = html.escape(f"{default_est['estimated_eur']:.2f} EUR")

    if is_fr:
        title = "HiveMind — Raid GPU d'Hiver & Guilde Amicale de Noël"
        pioneer_salute = "Merci aux Premiers Pionniers HiveMind ! Bienvenue dans le Raid GPU d'Hiver citoyen et transparent."
        subtitle = (
            "Tableau de bord honnête en direct, guildes amicales symboliques (0,00 EUR) et calculateur transparent "
            "de consommation électrique (Wh & EUR) avant de prêter ton GPU perso."
        )
        rule_text = PERSONAL_PC_RULE_FR
        board_h2 = "1. Compteurs de l'Essaim en Direct (Zéro Chiffre Inventé)"
        board_desc = (
            "Chaque valeur ci-dessous provient directement de la passerelle en direct (/v1/hivemind/raid/status) "
            "ou affiche 'n/a' lorsqu'aucune mesure n'est disponible."
        )
        lbl_trainers = "Entraîneurs OpenTrainer actifs"
        lbl_reg_trainers = "Comptes entraîneurs inscrits"
        lbl_producers = "Producteurs GPU en ligne"
        lbl_reg_producers = "Flotte producteurs inscrite"
        lbl_accounts = "Comptes inscrits"
        lbl_active_24h = "Comptes actifs (24 h)"
        lbl_rounds = "Rondes OpenTrainer acceptées"
        lbl_models = "Modèles servis en direct"
        btn_refresh = "Actualiser les compteurs en direct"
        refresh_note = "Source : /v1/hivemind/raid/status (lecture directe sans cache)."

        guild_h2 = "2. Guildes Amicales de Vacances (Check-in Symbolique • 0,00 EUR)"
        guild_desc = (
            "Choisis une étiquette d'équipe conviviale pour le plaisir communautaire : 0,00 EUR / 0,00 USD de frais, "
            "aucune récompense monétaire, juste de l'entraide open-source sur PC personnel."
        )
        guild_default_status = "Aucune guilde sélectionnée pour l'instant (participation libre et 100 % symbolique • 0,00 EUR)."

        calc_h2 = "3. Calculateur Transparent d'Électricité (Watts × Heures × Tarif kWh → Wh & EUR)"
        calc_desc = (
            "Transparence totale : HiveMind est un projet bénévole (0,00 EUR versé ou facturé par HiveMind), mais "
            "ton PC consomme des Wh sur ton compteur domestique. Estime le coût réel avant de lancer une session."
        )
        lbl_watts = "Puissance GPU dédiée (Watts)"
        lbl_hours = "Durée de session (Heures)"
        lbl_tariff = "Tarif électricité (EUR / kWh)"
        btn_calc = "Calculer Wh & EUR"
        lbl_res_wh = "Énergie estimée"
        lbl_res_eur = "Coût compteur estimé"
        init_calc_note = (
            f"{int(DEFAULT_GPU_WATTS)} W × {int(DEFAULT_HOURS)} h × {DEFAULT_KWH_TARIFF_EUR:.2f} EUR/kWh = "
            f"{default_est['energy_wh']:.1f} Wh ({default_est['estimated_eur']:.2f} EUR sur ton compteur perso • 0,00 EUR versé à/par HiveMind)."
        )

        fb_h2 = "4. Retours d'Expérience : Simplicité d'Accès & Zéro Lag en Jeu"
        fb_desc = "Résumé anonyme en direct des retours envoyés par les développeurs et joueurs PC (--feedback)."
        lbl_fb_total = "Avis enregistrés"
        lbl_fb_ease = "Facilité moyenne"
        lbl_fb_lag = "Sessions sans lag jeu"
        lbl_fb_prof = "Répartition profils"
        ces_card_h3 = "Micro-sondage 1-clic Zéro-PII : Facilité (CES) & Confort Steam (< 1.5 s)"
        ces_card_desc = (
            "Envoie ton avis anonyme en un clic directement depuis le navigateur (aucun identifiant, "
            "aucun texte libre, zéro PII • 0,00 EUR / 0,00 USD)."
        )
        ces_lbl_profile = "Profil :"
        ces_btn_dev = "Développeur (VS Code / API)"
        ces_btn_gamer = "Gamer PC (Steam / GPU)"
        ces_lbl_ease = "Facilité d'accès (1 à 5) :"
        ces_lbl_lag = "Confort Steam (< 1.5 s libération VRAM) :"
        ces_btn_lag_yes = "Zéro lag constaté (Oui)"
        ces_btn_lag_no = "Ralentissement constaté (Non)"
        ces_btn_submit = "Envoyer mon avis 1-clic (Zéro PII)"
        ces_btn_retry = "Réessayer l'envoi en attente"
        ces_init_status = "Prêt : sélectionne ton profil, ta note (défaut 5/5) et ton confort Steam, puis clique en 1 clic."
        ces_compliance_note = CES_COMPLIANCE_REMINDER_FR
        cmd_hint = f"python3 scripts/hivemind_doctor.py --gateway {safe_gw}"
        cmd_label = "Vérifier son PC en lecture seule (zéro dépendance) :"
        lang_switch = f'<a href="{safe_path}?lang=en">English version (/raid)</a>'
    else:
        title = "HiveMind — Winter GPU Raid & Friendly Holiday Guild"
        pioneer_salute = "Thank You HiveMind Pioneers! Welcome to the honest citizen-science Winter GPU Raid & Holiday Guild."
        subtitle = (
            "Live unvarnished swarm scoreboard, symbolic 0.00 EUR friendly guild tags, and an upfront Wh & EUR "
            "home electricity transparency estimator."
        )
        rule_text = PERSONAL_PC_RULE_EN
        board_h2 = "1. Live Swarm Scoreboard (Zero Invented Telemetry)"
        board_desc = (
            "Every reading below comes straight from the live gateway (/v1/hivemind/raid/status) "
            "or renders as 'n/a' when not reported."
        )
        lbl_trainers = "Active OpenTrainer Nodes"
        lbl_reg_trainers = "Registered trainer accounts"
        lbl_producers = "Live GPU Producers"
        lbl_reg_producers = "Registered producer fleet"
        lbl_accounts = "Registered Accounts"
        lbl_active_24h = "Active accounts (24h)"
        lbl_rounds = "Accepted OpenTrainer Rounds"
        lbl_models = "Live served models"
        btn_refresh = "Refresh Live Scoreboard"
        refresh_note = "Source: /v1/hivemind/raid/status (live uncached reading)."

        guild_h2 = "2. Friendly Holiday Guild Check-In Tags (Symbolic • 0.00 EUR)"
        guild_desc = (
            "Pick a friendly holiday guild tag just for open-source camaraderie: 0.00 EUR / 0.00 USD entry, "
            "zero cash or crypto prizes, strictly personal home PCs."
        )
        guild_default_status = "No guild tag selected yet (optional symbolic check-in • 0.00 EUR / 0.00 USD)."

        calc_h2 = "3. Electricity Transparency Calculator (GPU Watts × Hours × kWh Tariff → Wh & EUR)"
        calc_desc = (
            "Full honesty upfront: HiveMind is non-commercial (0.00 EUR paid to or by HiveMind), while your home "
            "meter bills the electricity your GPU draws. Estimate exact Wh and EUR before lending idle hours."
        )
        lbl_watts = "GPU Power Draw (Watts)"
        lbl_hours = "Session Duration (Hours)"
        lbl_tariff = "Electricity Tariff (EUR / kWh)"
        btn_calc = "Calculate Wh & EUR"
        lbl_res_wh = "Estimated Energy"
        lbl_res_eur = "Estimated Meter Cost"
        init_calc_note = (
            f"{int(DEFAULT_GPU_WATTS)} W × {int(DEFAULT_HOURS)} h × {DEFAULT_KWH_TARIFF_EUR:.2f} EUR/kWh = "
            f"{default_est['energy_wh']:.1f} Wh ({default_est['estimated_eur']:.2f} EUR on your home meter • 0.00 EUR paid to/by HiveMind)."
        )

        fb_h2 = "4. Live Onboarding Ease & Zero-Lag Gamer Feedback"
        fb_desc = "Live anonymous usability summary submitted by holiday developers and PC gamers (--feedback)."
        lbl_fb_total = "Recorded Ratings"
        lbl_fb_ease = "Mean Ease Rating"
        lbl_fb_lag = "Zero-Game-Lag Ratio"
        lbl_fb_prof = "By Profile"
        ces_card_h3 = "1-Click Zero-PII CES & Steam Zero-Lag Micro-Survey (< 1.5 s)"
        ces_card_desc = (
            "Share your anonymous ease rating and Steam comfort check in one click right from your browser "
            "(zero credentials, zero free-text, zero PII • 0.00 EUR / 0.00 USD)."
        )
        ces_lbl_profile = "Profile:"
        ces_btn_dev = "Developer (VS Code / API)"
        ces_btn_gamer = "PC Gamer (Steam / GPU)"
        ces_lbl_ease = "Ease Rating (1 to 5):"
        ces_lbl_lag = "Steam Comfort (< 1.5 s VRAM release):"
        ces_btn_lag_yes = "Zero Game Lag (Yes)"
        ces_btn_lag_no = "Noticeable Lag (No)"
        ces_btn_submit = "Submit 1-Click Feedback (Zero PII)"
        ces_btn_retry = "Retry Pending Feedback"
        ces_init_status = "Ready: choose your profile, ease rating (default 5/5), and Steam comfort, then submit in 1 click."
        ces_compliance_note = CES_COMPLIANCE_REMINDER_EN
        cmd_hint = f"python3 scripts/hivemind_doctor.py --gateway {safe_gw}"
        cmd_label = "Read-only 5-second local preflight check (Python stdlib only):"
        lang_switch = f'<a href="{safe_path}?lang=fr">Version française (/raid-noel)</a>'

    guild_cards_html: list[str] = []
    for g in FRIENDLY_GUILDS:
        gid = html.escape(g["guild_id"])
        glabel = html.escape(g["label"])
        motto = html.escape(g["motto_fr"] if is_fr else g["motto_en"])
        fee = html.escape(g["symbolic_fee"])
        guild_cards_html.append(
            f'<div class="guild-card">'
            f'<button type="button" id="guildBtn-{gid}" class="guild-btn" data-guild-id="{gid}" aria-pressed="false">'
            f"{glabel} · {fee}"
            f"</button>"
            f'<p class="muted">{motto}</p>'
            f"</div>"
        )

    preset_btns_html: list[str] = []
    preset_btn_ids = {"eco": "calcPresetEco", "whisper": "calcPresetWhisper", "desktop": "calcPresetDesktop"}
    for p in ELECTRICITY_PRESETS_WATTS:
        btn_id = html.escape(preset_btn_ids[p["id"]])
        plabel = html.escape(p["label_fr"] if is_fr else p["label_en"])
        preset_btns_html.append(
            f'<button type="button" id="{btn_id}" class="chip-btn">{plabel}</button>'
        )

    ces_ease_btns_html: list[str] = []
    for rating in range(1, 6):
        is_default_ease = rating == 5
        btn_cls = "chip-btn active" if is_default_ease else "chip-btn"
        pressed_str = "true" if is_default_ease else "false"
        ces_ease_btns_html.append(
            f'<button type="button" id="cesEaseBtn-{rating}" class="{btn_cls}" '
            f'data-ease="{rating}" aria-pressed="{pressed_str}">{rating} / 5</button>'
        )

    safe_nonce = html.escape(nonce)
    return f"""<!DOCTYPE html>
<html lang="{lang_code}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style nonce="{safe_nonce}">
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0b1120; color: #e2e8f0; margin: 0; padding: 22px; line-height: 1.5; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    nav.topnav {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; font-size: 0.9rem; }}
    a {{ color: #7dd3fc; }}
    h1 {{ font-size: 1.65rem; margin: 0 0 6px 0; color: #f8fafc; }}
    .pioneer {{ color: #fde68a; font-weight: 600; margin: 0 0 6px 0; font-size: 0.95rem; }}
    .sub {{ color: #94a3b8; margin: 0 0 14px 0; }}
    .rule {{ background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin-bottom: 18px; font-size: 0.92rem; }}
    .panel {{ background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-bottom: 18px; }}
    .panel h2 {{ font-size: 1.15rem; margin: 0 0 8px 0; color: #38bdf8; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 12px; }}
    .stat-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
    .stat-val {{ font-size: 1.5rem; font-weight: 700; color: #f8fafc; display: block; }}
    .stat-lbl {{ font-size: 0.86rem; color: #cbd5e1; }}
    .stat-sub {{ font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }}
    .guild-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 12px; }}
    .guild-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
    .guild-btn {{ width: 100%; background: #0f172a; color: #e2e8f0; border: 1px solid #38bdf8; border-radius: 6px; padding: 8px 10px; font-weight: 600; cursor: pointer; }}
    .guild-btn.active {{ background: #0284c7; color: #ffffff; border-color: #7dd3fc; }}
    .calc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 12px; }}
    label {{ display: block; font-size: 0.86rem; color: #cbd5e1; margin-bottom: 4px; }}
    input[type="number"] {{ width: 100%; box-sizing: border-box; background: #020617; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; font-size: 0.95rem; }}
    .presets {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .chip-btn, .act-btn {{ background: #1e293b; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 7px 12px; font-size: 0.86rem; cursor: pointer; }}
    .chip-btn.active {{ background: #0284c7; color: #ffffff; border-color: #7dd3fc; font-weight: 600; }}
    .act-btn {{ background: #0284c7; color: #ffffff; border-color: #0284c7; font-weight: 600; margin-top: 10px; }}
    .ces-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-top: 14px; }}
    .ces-card h3 {{ font-size: 1rem; margin: 0 0 6px 0; color: #f8fafc; }}
    .ces-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}
    .ces-lbl {{ font-size: 0.86rem; color: #cbd5e1; font-weight: 600; margin-right: 4px; }}
    .ces-actions {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 12px; }}
    .ces-actions .act-btn {{ margin-top: 0; }}
    .calc-out {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 12px; display: flex; flex-wrap: wrap; gap: 20px; align-items: baseline; }}
    .muted {{ color: #94a3b8; font-size: 0.86rem; margin: 6px 0 0 0; }}
    pre {{ background: #020617; color: #a7f3d0; padding: 10px 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; margin: 6px 0 0 0; }}
    footer.foot {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; color: #94a3b8; font-size: 0.86rem; margin-top: 16px; }}
  </style>
</head>
<body>
  <main id="raidNoelApp" class="wrap" data-lang="{lang_code}">
    <nav class="topnav" aria-label="HiveMind navigation">
      <a href="/noel">/noel</a>
      <a href="/quick">/quick</a>
      <a href="/signup">/signup</a>
      <a href="/join">/join</a>
      <a href="/portal">/portal</a>
      <a href="/crowdgpt-raid">/crowdgpt-raid</a>
      <span>•</span>
      {lang_switch}
    </nav>

    <h1>{html.escape(title)}</h1>
    <p class="pioneer">{html.escape(pioneer_salute)}</p>
    <p class="sub">{html.escape(subtitle)}</p>
    <div id="personalPcComplianceBanner" class="rule" role="note">{html.escape(rule_text)}</div>

    <section class="panel" aria-labelledby="scoreboardHeading">
      <h2 id="scoreboardHeading">{html.escape(board_h2)}</h2>
      <p class="muted">{html.escape(board_desc)}</p>
      <div class="stats-grid">
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_trainers)}</span>
          <span id="raidLiveTrainers" class="stat-val">{trainers_txt}</span>
          <div class="stat-sub">{html.escape(lbl_reg_trainers)}: <span id="raidRegisteredTrainers">{reg_trainers_txt}</span></div>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_producers)}</span>
          <span id="raidLiveProducers" class="stat-val">{producers_txt}</span>
          <div class="stat-sub">{html.escape(lbl_reg_producers)}: <span id="raidRegisteredProducers">{reg_producers_txt}</span></div>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_accounts)}</span>
          <span id="raidRegisteredAccounts" class="stat-val">{accounts_txt}</span>
          <div class="stat-sub">{html.escape(lbl_active_24h)}: <span id="raidActiveAccounts24h">{active_24h_txt}</span></div>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_rounds)}</span>
          <span id="raidAcceptedRounds" class="stat-val">{rounds_txt}</span>
          <div class="stat-sub">{html.escape(lbl_models)}: <span id="raidLiveModels">{models_txt}</span></div>
        </div>
      </div>
      <button type="button" id="refreshRaidBtn" class="act-btn">{html.escape(btn_refresh)}</button>
      <p id="raidLiveRefreshNote" class="muted">{html.escape(refresh_note)}</p>
    </section>

    <section class="panel" aria-labelledby="guildHeading">
      <h2 id="guildHeading">{html.escape(guild_h2)}</h2>
      <p class="muted">{html.escape(guild_desc)}</p>
      <div class="guild-grid">
        {"".join(guild_cards_html)}
      </div>
      <p id="guildCheckinStatus" class="muted">{html.escape(guild_default_status)}</p>
    </section>

    <section class="panel" aria-labelledby="calcHeading">
      <h2 id="calcHeading">{html.escape(calc_h2)}</h2>
      <p class="muted">{html.escape(calc_desc)}</p>
      <div class="presets">
        {"".join(preset_btns_html)}
      </div>
      <div class="calc-grid">
        <div>
          <label for="calcWatts">{html.escape(lbl_watts)}</label>
          <input id="calcWatts" type="number" min="0" max="2000" step="5" value="120">
        </div>
        <div>
          <label for="calcHours">{html.escape(lbl_hours)}</label>
          <input id="calcHours" type="number" min="0" max="8760" step="0.5" value="4">
        </div>
        <div>
          <label for="calcTariff">{html.escape(lbl_tariff)}</label>
          <input id="calcTariff" type="number" min="0" max="10" step="0.01" value="0.25">
        </div>
      </div>
      <button type="button" id="calcComputeBtn" class="act-btn">{html.escape(btn_calc)}</button>
      <div class="calc-out">
        <div>
          <span class="stat-lbl">{html.escape(lbl_res_wh)}</span>
          <span id="calcResultWh" class="stat-val">{init_wh_txt}</span>
        </div>
        <div>
          <span class="stat-lbl">{html.escape(lbl_res_eur)}</span>
          <span id="calcResultEur" class="stat-val">{init_eur_txt}</span>
        </div>
      </div>
      <p id="calcResultNote" class="muted">{html.escape(init_calc_note)}</p>
    </section>

    <section class="panel" aria-labelledby="feedbackHeading">
      <h2 id="feedbackHeading">{html.escape(fb_h2)}</h2>
      <p class="muted">{html.escape(fb_desc)}</p>
      <div class="stats-grid">
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_fb_total)}</span>
          <span id="raidFeedbackResponses" class="stat-val">{fb_total_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_fb_ease)}</span>
          <span id="raidFeedbackEase" class="stat-val">{fb_ease_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_fb_lag)}</span>
          <span id="raidFeedbackZeroLag" class="stat-val">{fb_lag_txt}</span>
        </div>
        <div class="stat-box">
          <span class="stat-lbl">{html.escape(lbl_fb_prof)}</span>
          <span id="raidFeedbackProfiles" class="stat-val">{fb_profiles_txt}</span>
        </div>
      </div>
      <div id="cesOneClickCard" class="ces-card">
        <h3>{html.escape(ces_card_h3)}</h3>
        <p class="muted">{html.escape(ces_card_desc)}</p>
        <div class="ces-row" role="group" aria-label="{html.escape(ces_lbl_profile)}">
          <span class="ces-lbl">{html.escape(ces_lbl_profile)}</span>
          <button type="button" id="cesProfileDevBtn" class="chip-btn" data-profile="dev" aria-pressed="false">{html.escape(ces_btn_dev)}</button>
          <button type="button" id="cesProfileGamerBtn" class="chip-btn active" data-profile="gamer" aria-pressed="true">{html.escape(ces_btn_gamer)}</button>
        </div>
        <div class="ces-row" role="group" aria-label="{html.escape(ces_lbl_ease)}">
          <span class="ces-lbl">{html.escape(ces_lbl_ease)}</span>
          {"".join(ces_ease_btns_html)}
        </div>
        <div class="ces-row" role="group" aria-label="{html.escape(ces_lbl_lag)}">
          <span class="ces-lbl">{html.escape(ces_lbl_lag)}</span>
          <button type="button" id="cesLagYesBtn" class="chip-btn active" data-zero-lag="true" aria-pressed="true">{html.escape(ces_btn_lag_yes)}</button>
          <button type="button" id="cesLagNoBtn" class="chip-btn" data-zero-lag="false" aria-pressed="false">{html.escape(ces_btn_lag_no)}</button>
        </div>
        <div class="ces-actions">
          <button type="button" id="submitCesBtn" class="act-btn">{html.escape(ces_btn_submit)}</button>
          <button type="button" id="retryPendingCesBtn" class="chip-btn">{html.escape(ces_btn_retry)}</button>
        </div>
        <p id="cesSubmitStatus" class="muted" role="status" aria-live="polite">{html.escape(ces_init_status)}</p>
        <p id="cesComplianceReminder" class="muted">{html.escape(ces_compliance_note)}</p>
      </div>
      <p class="muted"><strong>{html.escape(cmd_label)}</strong></p>
      <pre>{cmd_hint}</pre>
    </section>

    <footer class="foot">
      <span>AGPL-3.0 • 0.00 EUR / 0.00 USD</span>
      <span>{lang_switch} • <a href="/v1/hivemind/raid/status">/v1/hivemind/raid/status</a></span>
    </footer>
  </main>
  <script nonce="{safe_nonce}">{_RAID_NOEL_JS}</script>
</body>
</html>
"""
