#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Concise 5-line FR/EN Discord/Signal holiday raid bulletin generator (`SLICE-07`).

Generates a ready-to-paste 5-line French, English, or bilingual (FR/EN) holiday update
solely from measured JSON payloads:
  - ``GET /v1/hivemind/t4/status``
  - ``GET /v1/hivemind/onboarding/feedback/summary``

Strict Honesty, Privacy & Compliance Invariants:
  - Standard library only; zero third-party runtime dependencies.
  - Every absent, null, or unreceived metric strictly renders as ``'n/a'``; zero numbers are ever invented.
  - When ``telemetry_status`` is ``NO_TELEMETRY_RECEIVED`` (or ``telemetry_source`` is
    ``DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED``), static T4 fallback snapshot numbers are treated as absent (``'n/a'``).
  - Outbound live fetches are restricted to allowlisted hosts (``*.run.app``, ``*.googleapis.com``,
    ``datasets-server.huggingface.co``, or loopback) with proxies disabled.
  - Zero PII, zero Discord mass-ping tokens (``@everyone`` / ``@here``), and strict Personal-PC / Amadeus
    compliance (100% personal home PCs & personal hobby/open-source projects only, 0.00 EUR / 0.00 USD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.daily_raid_bulletin/1"
RECEIPT_OBJECT = "hivemind.daily_raid_bulletin_receipt"
CAMPAIGN_ID = "noel-2026-holiday-raid"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
HTTP_TIMEOUT_S = 5.0
NOT_AVAILABLE = "n/a"
BULLETIN_LINE_COUNT = 5
DISCORD_MAX_CHARS = 2000

SUPPORTED_LANGS: tuple[str, ...] = ("bilingual", "fr", "en")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
UNRECEIVED_TELEMETRY_STATUSES: frozenset[str] = frozenset(
    {
        "NO_TELEMETRY_RECEIVED",
        "UNAVAILABLE",
        "NONE",
    }
)
UNRECEIVED_TELEMETRY_SOURCES: frozenset[str] = frozenset(
    {
        "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED",
    }
)

COMPLIANCE_RULE_FR = (
    "100 % PC personnels à domicile et projets perso/open-source uniquement "
    "(conformité Amadeus : jamais de PC ni de code pro ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_RULE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus compliance: never corporate hardware or work code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ISO_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T\s].*)?$")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm[-_][A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')

HttpGetFn = Callable[[str, float], tuple[int, dict[str, Any]]]


class BulletinInputError(ValueError):
    """Raised when bulletin JSON inputs, gateway URLs, or CLI arguments are invalid."""


@dataclass(frozen=True)
class BulletinMetrics:
    """Validated, PII-free metrics extracted strictly from measured inputs (`None` -> `'n/a'`)."""

    report_date: str | None
    telemetry_status: str | None
    operational_mode: str | None
    active_producers_count: int | None
    distinct_physical_gpus: int | None
    vram_free_mib: float | None
    vram_total_mib: float | None
    temp_c: float | None
    power_w: float | None
    qlora_rounds_assimilated: int | None
    ratchet_accepted_rounds: int | None
    eval_loss_before: float | None
    eval_loss_after: float | None
    feedback_total_responses: int | None
    feedback_dev_count: int | None
    feedback_gamer_count: int | None
    mean_ease_rating: float | None
    zero_game_lag_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_date": self.report_date,
            "telemetry_status": self.telemetry_status,
            "operational_mode": self.operational_mode,
            "active_producers_count": self.active_producers_count,
            "distinct_physical_gpus": self.distinct_physical_gpus,
            "vram_free_mib": self.vram_free_mib,
            "vram_total_mib": self.vram_total_mib,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "qlora_rounds_assimilated": self.qlora_rounds_assimilated,
            "ratchet_accepted_rounds": self.ratchet_accepted_rounds,
            "eval_loss_before": self.eval_loss_before,
            "eval_loss_after": self.eval_loss_after,
            "feedback_total_responses": self.feedback_total_responses,
            "feedback_dev_count": self.feedback_dev_count,
            "feedback_gamer_count": self.feedback_gamer_count,
            "mean_ease_rating": self.mean_ease_rating,
            "zero_game_lag_ratio": self.zero_game_lag_ratio,
        }


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to `'bilingual'`, `'fr'`, or `'en'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise BulletinInputError("Language must be 'bilingual', 'fr', or 'en'.")
    cleaned = (raw or "bilingual").strip().lower().replace("_", "-")
    if cleaned in ("bilingual", "both", "fr-en", "en-fr", "all"):
        return "bilingual"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    raise BulletinInputError(f"Unsupported language {raw!r}; expected 'bilingual', 'fr', or 'en'.")


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host matches the HiveMind lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if (len(h) > 8 and h.endswith(".run.app")) or h == "storage.googleapis.com":
        return True
    if len(h) > 15 and h.endswith(".googleapis.com"):
        return True
    return False


def validate_gateway_url(url: str) -> str:
    """Validate and normalize a bare gateway origin URL against scheme and host allowlists."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise BulletinInputError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise BulletinInputError("Gateway URL must not contain control characters or newlines.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise BulletinInputError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise BulletinInputError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise BulletinInputError("Gateway URL must be a bare origin without path, query, or fragment.")
    try:
        port = parts.port
    except ValueError as exc:
        raise BulletinInputError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise BulletinInputError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise BulletinInputError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise BulletinInputError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise BulletinInputError("Remote gateway URL must use https.")
    return cleaned


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a read-only, proxy-free HTTP GET request to an allowlisted gateway endpoint."""
    parts = urlsplit(url)
    validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hivemind-daily-raid-bulletin/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            err_doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            err_doc = {"error": raw[:200]}
        return int(exc.code), (err_doc if isinstance(err_doc, dict) else {"data": err_doc})
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BulletinInputError(f"Unable to reach gateway endpoint {parts.path}: {reason}") from exc

    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise BulletinInputError(f"Malformed JSON from gateway endpoint {parts.path}.") from exc
    if not isinstance(parsed, dict):
        raise BulletinInputError(f"Expected JSON object from gateway endpoint {parts.path}.")
    return status_code, parsed


def fetch_live_bulletin_inputs(
    gateway_url: str,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Fetch `/v1/hivemind/t4/status` and `/v1/hivemind/onboarding/feedback/summary` read-only."""
    origin = validate_gateway_url(gateway_url)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)):
        raise BulletinInputError("timeout_s must be a finite positive number.")
    if float(timeout_s) <= 0.0:
        raise BulletinInputError("timeout_s must be > 0.")
    clamped_timeout = max(0.5, min(15.0, float(timeout_s)))

    t4_code, t4_doc = http_get(f"{origin}/v1/hivemind/t4/status", clamped_timeout)
    fb_code, fb_doc = http_get(f"{origin}/v1/hivemind/onboarding/feedback/summary", clamped_timeout)
    t4_payload = t4_doc if t4_code == 200 and isinstance(t4_doc, dict) else {}
    fb_payload = fb_doc if fb_code == 200 and isinstance(fb_doc, dict) else {}
    return t4_payload, fb_payload, {"t4_status_http": int(t4_code), "feedback_summary_http": int(fb_code)}


def _parse_iso_date(raw: Any, *, strict: bool) -> str | None:
    """Extract and calendar-validate a `YYYY-MM-DD` date string, returning None when absent."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, str):
        if strict:
            raise BulletinInputError("Date override must be a 'YYYY-MM-DD' string.")
        return None
    cleaned = raw.strip()
    if not cleaned:
        if strict:
            raise BulletinInputError("Date override must not be empty.")
        return None
    match = _ISO_DATE_PREFIX_RE.match(cleaned)
    if not match:
        if strict:
            raise BulletinInputError(f"Invalid date {raw!r}; expected 'YYYY-MM-DD'.")
        return None
    day_str = match.group(1)
    try:
        date.fromisoformat(day_str)
    except ValueError as exc:
        if strict:
            raise BulletinInputError(f"Invalid calendar date {day_str!r}.") from exc
        return None
    return day_str


def _opt_safe_token(raw: Any) -> str | None:
    """Validate an identifier token (rejecting control chars, spaces, or `@everyone` mentions)."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned or not _SAFE_TOKEN_RE.fullmatch(cleaned):
        return None
    return cleaned


def _opt_nonneg_int(val: Any, field_name: str) -> int | None:
    """Validate an optional non-negative integer (`None` -> `None`, invalid -> `BulletinInputError`)."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise BulletinInputError(f"Field '{field_name}' must be a non-negative integer.")
    if isinstance(val, float) and (not math.isfinite(val) or not val.is_integer()):
        raise BulletinInputError(f"Field '{field_name}' must be an exact integer.")
    ival = int(val)
    if ival < 0:
        raise BulletinInputError(f"Field '{field_name}' must be >= 0 (got {ival}).")
    return ival


def _opt_bounded_float(
    val: Any,
    field_name: str,
    *,
    low: float,
    high: float,
    decimals: int = 4,
) -> float | None:
    """Validate an optional finite float in `[low, high]` (`None` -> `None`)."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise BulletinInputError(f"Field '{field_name}' must be a finite number.")
    fval = float(val)
    if not math.isfinite(fval):
        raise BulletinInputError(f"Field '{field_name}' must be finite.")
    if fval < low or fval > high:
        raise BulletinInputError(f"Field '{field_name}' must be between {low:g} and {high:g} (got {fval}).")
    return round(fval, decimals)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, str]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key], key
    return None, (keys[0] if keys else "value")


def extract_bulletin_metrics(
    t4_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    date_override: str | None = None,
) -> BulletinMetrics:
    """Extract validated metrics from `/v1/hivemind/t4/status` and `/v1/hivemind/onboarding/feedback/summary`."""
    if t4_status is not None and not isinstance(t4_status, Mapping):
        raise BulletinInputError("t4_status must be a JSON object (Mapping) or None.")
    if feedback_summary is not None and not isinstance(feedback_summary, Mapping):
        raise BulletinInputError("feedback_summary must be a JSON object (Mapping) or None.")

    t4 = t4_status or {}
    fb = feedback_summary or {}

    explicit_day = _parse_iso_date(date_override, strict=True) if date_override is not None else None
    sched = t4.get("european_schedule")
    if sched is not None and not isinstance(sched, Mapping):
        raise BulletinInputError("Field 'european_schedule' in t4_status must be a JSON object.")
    inferred_day = (
        explicit_day
        or _parse_iso_date((sched or {}).get("current_utc_iso"), strict=False)
        or _parse_iso_date(t4.get("captured_utc") or t4.get("date"), strict=False)
    )

    telemetry_status = _opt_safe_token(t4.get("telemetry_status"))
    telemetry_source = _opt_safe_token(t4.get("telemetry_source"))
    telemetry_unreceived = bool(
        (telemetry_status and telemetry_status.upper() in UNRECEIVED_TELEMETRY_STATUSES)
        or (telemetry_source and telemetry_source.upper() in UNRECEIVED_TELEMETRY_SOURCES)
    )

    active_producers = _opt_nonneg_int(t4.get("active_producers_count"), "active_producers_count")
    distinct_gpus = _opt_nonneg_int(t4.get("distinct_physical_gpus"), "distinct_physical_gpus")

    if telemetry_unreceived:
        operational_mode: str | None = None
        vram_free_mib: float | None = None
        vram_total_mib: float | None = None
        temp_c: float | None = None
        power_w: float | None = None
        qlora_rounds: int | None = None
        ratchet_rounds: int | None = None
        loss_before: float | None = None
        loss_after: float | None = None
    else:
        operational_mode = _opt_safe_token(t4.get("operational_mode"))
        hw_raw = t4.get("hardware")
        gpu_raw = t4.get("gpu")
        if hw_raw is not None and not isinstance(hw_raw, Mapping):
            raise BulletinInputError("Field 'hardware' in t4_status must be a JSON object.")
        if gpu_raw is not None and not isinstance(gpu_raw, Mapping):
            raise BulletinInputError("Field 'gpu' in t4_status must be a JSON object.")
        hw_map: Mapping[str, Any] = hw_raw if isinstance(hw_raw, Mapping) else (gpu_raw or {})

        raw_free, free_key = _first_present(hw_map, ("vram_free_mib", "free_mib"))
        raw_total, total_key = _first_present(hw_map, ("vram_total_mib", "total_mib"))
        vram_free_mib = _opt_bounded_float(raw_free, f"hardware.{free_key}", low=0.0, high=1_048_576.0, decimals=1)
        vram_total_mib = _opt_bounded_float(raw_total, f"hardware.{total_key}", low=1.0, high=1_048_576.0, decimals=1)
        temp_c = _opt_bounded_float(hw_map.get("temp_c"), "hardware.temp_c", low=-50.0, high=150.0, decimals=1)
        power_w = _opt_bounded_float(hw_map.get("power_w"), "hardware.power_w", low=0.0, high=2000.0, decimals=1)

        qp_raw = t4.get("qlora_progress")
        if qp_raw is not None and not isinstance(qp_raw, Mapping):
            raise BulletinInputError("Field 'qlora_progress' in t4_status must be a JSON object.")
        qp_map: Mapping[str, Any] = qp_raw or {}

        raw_rounds, rounds_key = _first_present(qp_map, ("qlora_rounds_assimilated", "assimilated_workunits"))
        qlora_rounds = _opt_nonneg_int(raw_rounds, f"qlora_progress.{rounds_key}")
        ratchet_rounds = _opt_nonneg_int(
            qp_map.get("ratchet_accepted_rounds"),
            "qlora_progress.ratchet_accepted_rounds",
        )
        raw_loss_before, lb_key = _first_present(qp_map, ("held_out_eval_loss_before", "eval_loss_before"))
        raw_loss_after, la_key = _first_present(
            qp_map,
            ("held_out_eval_loss_after", "latest_eval_loss_after", "eval_loss_after"),
        )
        loss_before = _opt_bounded_float(
            raw_loss_before,
            f"qlora_progress.{lb_key}",
            low=0.0,
            high=1000.0,
            decimals=4,
        )
        loss_after = _opt_bounded_float(
            raw_loss_after,
            f"qlora_progress.{la_key}",
            low=0.0,
            high=1000.0,
            decimals=4,
        )

    raw_fb_total, fb_total_key = _first_present(fb, ("total_responses", "feedback_responses"))
    fb_total = _opt_nonneg_int(raw_fb_total, fb_total_key)

    by_prof_raw = fb.get("by_profile")
    if by_prof_raw is not None and not isinstance(by_prof_raw, Mapping):
        raise BulletinInputError("Field 'by_profile' in feedback_summary must be a JSON object.")
    by_prof: Mapping[str, Any] = by_prof_raw or {}
    fb_dev = _opt_nonneg_int(by_prof.get("dev"), "by_profile.dev")
    fb_gamer = _opt_nonneg_int(by_prof.get("gamer"), "by_profile.gamer")
    if fb_total is not None and (fb_dev or 0) + (fb_gamer or 0) > fb_total:
        raise BulletinInputError("Sum of 'by_profile' counts cannot exceed 'total_responses'.")

    raw_ease, ease_key = _first_present(fb, ("mean_ease_rating", "average_ease_rating"))
    mean_ease = _opt_bounded_float(raw_ease, ease_key, low=1.0, high=5.0, decimals=2)
    zero_lag = _opt_bounded_float(
        fb.get("zero_game_lag_ratio"),
        "zero_game_lag_ratio",
        low=0.0,
        high=1.0,
        decimals=4,
    )
    if fb_total == 0 and (mean_ease is not None or zero_lag is not None):
        raise BulletinInputError("mean_ease_rating and zero_game_lag_ratio must be null when total_responses is 0.")

    return BulletinMetrics(
        report_date=inferred_day,
        telemetry_status=telemetry_status,
        operational_mode=operational_mode,
        active_producers_count=active_producers,
        distinct_physical_gpus=distinct_gpus,
        vram_free_mib=vram_free_mib,
        vram_total_mib=vram_total_mib,
        temp_c=temp_c,
        power_w=power_w,
        qlora_rounds_assimilated=qlora_rounds,
        ratchet_accepted_rounds=ratchet_rounds,
        eval_loss_before=loss_before,
        eval_loss_after=loss_after,
        feedback_total_responses=fb_total,
        feedback_dev_count=fb_dev,
        feedback_gamer_count=fb_gamer,
        mean_ease_rating=mean_ease,
        zero_game_lag_ratio=zero_lag,
    )


def _fmt_int(val: int | None) -> str:
    return NOT_AVAILABLE if val is None else str(val)


def _fmt_num_unit(val: float | None, unit: str) -> str:
    if val is None:
        return NOT_AVAILABLE
    num_str = str(int(val)) if float(val).is_integer() else f"{val:.1f}"
    return f"{num_str} {unit}"


def _fmt_loss(val: float | None) -> str:
    return NOT_AVAILABLE if val is None else f"{val:.4f}"


def _fmt_ease(val: float | None) -> str:
    return NOT_AVAILABLE if val is None else f"{val:.2f}/5"


def _fmt_pct(ratio: float | None) -> str:
    return NOT_AVAILABLE if ratio is None else f"{ratio * 100.0:.1f}%"


def scan_output_safety(text: str) -> list[str]:
    """Verify rendered bulletin or JSON contains zero PII, mass-pings, or forbidden keys."""
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("home_path_detected")
    if _DISCORD_PING_RE.search(text):
        issues.append("discord_mass_ping_detected")
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise BulletinInputError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def render_bulletin_from_metrics(metrics: BulletinMetrics, *, lang: str = "bilingual") -> str:
    """Render an exact 5-line Discord/Signal holiday update in `'fr'`, `'en'`, or `'bilingual'`."""
    if not isinstance(metrics, BulletinMetrics):
        raise BulletinInputError("metrics must be a BulletinMetrics instance.")
    norm_lang = normalize_lang(lang)

    day = metrics.report_date or NOT_AVAILABLE
    status = metrics.telemetry_status or NOT_AVAILABLE
    mode = metrics.operational_mode or NOT_AVAILABLE
    producers = _fmt_int(metrics.active_producers_count)
    gpus = _fmt_int(metrics.distinct_physical_gpus)
    vram_free = _fmt_num_unit(metrics.vram_free_mib, "MiB")
    vram_total = _fmt_num_unit(metrics.vram_total_mib, "MiB")
    temp = _fmt_num_unit(metrics.temp_c, "C")
    power = _fmt_num_unit(metrics.power_w, "W")
    rounds = _fmt_int(metrics.qlora_rounds_assimilated)
    ratchet = _fmt_int(metrics.ratchet_accepted_rounds)
    loss_before = _fmt_loss(metrics.eval_loss_before)
    loss_after = _fmt_loss(metrics.eval_loss_after)
    responses = _fmt_int(metrics.feedback_total_responses)
    dev_n = _fmt_int(metrics.feedback_dev_count)
    gamer_n = _fmt_int(metrics.feedback_gamer_count)
    ease = _fmt_ease(metrics.mean_ease_rating)
    zero_lag = _fmt_pct(metrics.zero_game_lag_ratio)

    if norm_lang == "fr":
        lines = [
            f"🎄 [Raid Noël HiveMind • {day}] Télémétrie : {status} | Producteurs actifs : {producers} | GPU physiques : {gpus}",
            f"🖥️ Nœud T4 ({mode}) : VRAM libre {vram_free} / {vram_total} | Temp {temp} | Puissance {power}",
            f"🧠 QLoRA : rondes assimilées {rounds} (retenues ratchet : {ratchet}) | Loss éval {loss_before} -> {loss_after}",
            f"🎮 Confort ({responses} avis, dev {dev_n} / gamer {gamer_n}) : facilité {ease} | zéro saccade Steam {zero_lag}",
            "🛡️ PC perso uniquement (règle Amadeus, 0,00 EUR / 0,00 USD, AGPL-3.0) • /noel • /raid-noel • --uninstall",
        ]
    elif norm_lang == "en":
        lines = [
            f"🎄 [HiveMind Holiday Raid • {day}] Telemetry: {status} | Active producers: {producers} | Physical GPUs: {gpus}",
            f"🖥️ T4 Node ({mode}): Free VRAM {vram_free} / {vram_total} | Temp {temp} | Power {power}",
            f"🧠 QLoRA: assimilated rounds {rounds} (ratchet-kept: {ratchet}) | Eval loss {loss_before} -> {loss_after}",
            f"🎮 Comfort ({responses} replies, dev {dev_n} / gamer {gamer_n}): ease {ease} | zero Steam lag {zero_lag}",
            "🛡️ Personal home PCs ONLY (Amadeus rule, 0.00 EUR / 0.00 USD, AGPL-3.0) • /quick • /raid • --uninstall",
        ]
    else:
        lines = [
            f"🎄 [Raid Noël / Holiday Raid • {day}] Télémétrie/Telemetry: {status} | Producers: {producers} | GPUs: {gpus}",
            f"🖥️ T4 ({mode}) : VRAM libre/free {vram_free} / {vram_total} | Temp {temp} | Power {power}",
            f"🧠 QLoRA : rondes/rounds {rounds} (ratchet: {ratchet}) | Loss {loss_before} -> {loss_after}",
            f"🎮 Confort/Feedback ({responses} avis/replies, dev {dev_n} / gamer {gamer_n}) : facilité/ease {ease} | zéro-lag {zero_lag}",
            "🛡️ PC perso uniquement / Personal home PCs ONLY (Amadeus rule, 0.00 EUR / 0.00 USD, AGPL-3.0) • /noel • /raid-noel • --uninstall",
        ]

    rendered = "\n".join(lines)
    if len(rendered.splitlines()) != BULLETIN_LINE_COUNT or len(rendered) > DISCORD_MAX_CHARS:
        raise BulletinInputError("Rendered bulletin violated the 5-line / Discord length invariant.")
    return _assert_safe_output(rendered)


def render_daily_raid_bulletin(
    t4_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    lang: str = "bilingual",
    date_override: str | None = None,
) -> str:
    """High-level helper rendering a 5-line FR/EN Discord/Signal bulletin directly from JSON dicts."""
    metrics = extract_bulletin_metrics(t4_status, feedback_summary, date_override=date_override)
    return render_bulletin_from_metrics(metrics, lang=lang)


def build_bulletin_receipt(
    t4_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    lang: str = "bilingual",
    date_override: str | None = None,
    source_kind: str = "snapshot",
    http_statuses: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, SHA-256-attested JSON receipt containing the 5-line bulletins and metrics."""
    norm_lang = normalize_lang(lang)
    metrics = extract_bulletin_metrics(t4_status, feedback_summary, date_override=date_override)
    bulletin_fr = render_bulletin_from_metrics(metrics, lang="fr")
    bulletin_en = render_bulletin_from_metrics(metrics, lang="en")
    bulletin_bi = render_bulletin_from_metrics(metrics, lang="bilingual")
    selected = {"fr": bulletin_fr, "en": bulletin_en, "bilingual": bulletin_bi}[norm_lang]

    core_payload: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "source_kind": source_kind if _opt_safe_token(source_kind) else "snapshot",
        "language": norm_lang,
        "bulletin_line_count": BULLETIN_LINE_COUNT,
        "compliance": {
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
            "rule_fr": COMPLIANCE_RULE_FR,
            "rule_en": COMPLIANCE_RULE_EN,
        },
        "metrics": metrics.to_dict(),
        "bulletin": selected,
        "bulletin_lines": selected.splitlines(),
        "bulletin_fr": bulletin_fr,
        "bulletin_en": bulletin_en,
        "bulletin_bilingual": bulletin_bi,
    }
    if http_statuses is not None:
        core_payload["http_statuses"] = {str(k): int(v) for k, v in http_statuses.items()}

    digest = hashlib.sha256(
        json.dumps(core_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    receipt = dict(core_payload)
    receipt["receipt_sha256"] = digest
    _assert_safe_output(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return receipt


def load_json_file(path_str: str) -> dict[str, Any]:
    """Load and validate a JSON object from a local file path or `'-'` (stdin)."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise BulletinInputError("JSON input path must be non-empty.")
    cleaned = path_str.strip()
    try:
        raw_text = sys.stdin.read() if cleaned == "-" else Path(cleaned).read_text(encoding="utf-8")
    except OSError as exc:
        raise BulletinInputError(f"Unable to read JSON input file ({exc.strerror or 'io_error'}).") from exc
    if not raw_text.strip():
        raise BulletinInputError("JSON input is empty.")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise BulletinInputError(f"Invalid JSON syntax at line {exc.lineno}, column {exc.colno}.") from exc
    if not isinstance(parsed, dict):
        raise BulletinInputError("Top-level JSON value must be an object.")
    return parsed


def _unpack_bundle_snapshot(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unpack a combined `{'t4_status': ..., 'feedback_summary': ...}` snapshot or single-endpoint JSON."""
    has_t4_sub = "t4_status" in bundle
    has_fb_sub = "feedback_summary" in bundle or "onboarding_feedback" in bundle
    if has_t4_sub or has_fb_sub:
        t4_part = bundle.get("t4_status")
        fb_part = bundle.get("feedback_summary")
        if fb_part is None and "onboarding_feedback" in bundle:
            fb_part = bundle.get("onboarding_feedback")
        if t4_part is not None and not isinstance(t4_part, dict):
            raise BulletinInputError("Field 't4_status' in snapshot bundle must be a JSON object.")
        if fb_part is not None and not isinstance(fb_part, dict):
            raise BulletinInputError("Field 'feedback_summary' in snapshot bundle must be a JSON object.")
        return dict(t4_part or {}), dict(fb_part or {})
    if bundle.get("object") == "hivemind.onboarding_feedback_summary" or "mean_ease_rating" in bundle:
        return {}, dict(bundle)
    return dict(bundle), {}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for `scripts/daily_raid_bulletin.py`."""
    parser = argparse.ArgumentParser(
        prog="daily_raid_bulletin.py",
        description=(
            "Generate a concise 5-line FR/EN Discord/Signal holiday raid update from measured "
            "/v1/hivemind/t4/status and /v1/hivemind/onboarding/feedback/summary JSON inputs."
        ),
    )
    parser.add_argument(
        "--t4-status",
        type=str,
        default=None,
        help="Path to measured /v1/hivemind/t4/status JSON file (or '-' for stdin).",
    )
    parser.add_argument(
        "--feedback-summary",
        type=str,
        default=None,
        help="Path to measured /v1/hivemind/onboarding/feedback/summary JSON file (or '-' for stdin).",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Optional combined JSON bundle containing 't4_status' and 'feedback_summary' (or '-' for stdin).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live read-only /v1/hivemind/t4/status and /v1/hivemind/onboarding/feedback/summary.",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        default=None,
        help=f"Allowlisted HiveMind gateway URL (default with --live: {DEFAULT_GATEWAY}).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="bilingual",
        help="Bulletin language: 'bilingual' (default, 5 lines FR/EN), 'fr' (5 lines), or 'en' (5 lines).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        dest="date_override",
        help="Optional explicit YYYY-MM-DD date label (defaults to measured UTC date or 'n/a').",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
        help="Stdout format: 'text' (default 5-line Discord/Signal bulletin) or 'json'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_flag",
        help="Shorthand for --format json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write the 5-line text bulletin.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional file path to write the SHA-256 JSON receipt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=HTTP_TIMEOUT_S,
        help="HTTP GET timeout in seconds for --live mode (default: 5.0).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_get: HttpGetFn = default_http_get_json,
) -> int:
    """CLI entrypoint returning `0` on success or `2` on validation / network error."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        lang = normalize_lang(args.lang)
        has_file_inputs = bool(args.t4_status or args.feedback_summary or args.snapshot)
        has_live_inputs = bool(args.live or args.gateway)
        if has_file_inputs and has_live_inputs:
            raise BulletinInputError("Provide either local JSON file flags or --live/--gateway, not both.")
        if not has_file_inputs and not has_live_inputs:
            raise BulletinInputError(
                "Specify --t4-status / --feedback-summary / --snapshot, or --live [--gateway <url>]."
            )

        stdin_uses = sum(1 for p in (args.t4_status, args.feedback_summary, args.snapshot) if p == "-")
        if stdin_uses > 1:
            raise BulletinInputError("Only one input flag may read from '-' (stdin) at a time.")

        http_statuses: Mapping[str, int] | None = None
        if has_live_inputs:
            gw = args.gateway if args.gateway is not None else DEFAULT_GATEWAY
            t4_doc, fb_doc, http_statuses = fetch_live_bulletin_inputs(
                gw,
                http_get=http_get,
                timeout_s=args.timeout,
            )
            source_kind = "live_gateway"
        else:
            t4_doc, fb_doc = {}, {}
            if args.snapshot:
                t4_doc, fb_doc = _unpack_bundle_snapshot(load_json_file(args.snapshot))
            if args.t4_status:
                t4_doc = load_json_file(args.t4_status)
            if args.feedback_summary:
                fb_doc = load_json_file(args.feedback_summary)
            source_kind = "snapshot"

        receipt = build_bulletin_receipt(
            t4_doc,
            fb_doc,
            lang=lang,
            date_override=args.date_override,
            source_kind=source_kind,
            http_statuses=http_statuses,
        )
        bulletin_text = str(receipt["bulletin"]) + "\n"
        json_text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(bulletin_text, encoding="utf-8")

        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json_text, encoding="utf-8")

        emit_json = args.json_flag or args.output_format == "json"
        sys.stdout.write(json_text if emit_json else bulletin_text)
        return 0
    except BulletinInputError as exc:
        err_msg = str(exc)
        if scan_output_safety(err_msg):
            err_msg = "Input validation failed (redacted unsafe token)."
        sys.stderr.write(f"error: {err_msg}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
