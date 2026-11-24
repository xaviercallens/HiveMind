#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Privacy-preserving usability funnel and satisfaction analyzer (Christmas Adoption Plan, Section 5).

Computes aggregate, zero-PII onboarding funnel and comfort metrics from local JSON snapshots
or live read-only HiveMind gateway summary endpoints:
  - Signup-to-first-request activation rate (`activation_ratio`, plus optional `signup_conversion_ratio`)
  - Preflight-to-first-round completion rate (`preflight_to_first_round_ratio` & `install_dropoff_ratio`)
  - Anonymous usability ease rating (`mean_ease_rating`, target >= 4.2 / 5)
  - Gamer-Whisper comfort ratio (`zero_game_lag_ratio`, target 100%)
  - UX regression alerts (`< 20%` signup conversion, `> 30%` install dropoff, `ease_rating < 4.2`,
    and `< 100%` zero-game-lag comfort)

Strict Compliance & Privacy Invariants:
  - Standard library only; zero third-party runtime dependencies.
  - Only allowlisted hosts (`*.run.app`, `*.googleapis.com`, `datasets-server.huggingface.co`, or loopback).
  - Zero PII: handles, emails, account IDs, IP addresses, API keys, hostnames, and free-text notes are
    never stored or emitted; outputs are verified prior to emission.
  - 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD, AGPL-3.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.onboarding_funnel_report/1"
RECEIPT_OBJECT = "hivemind.onboarding_funnel_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
HTTP_TIMEOUT_S = 5.0

SUPPORTED_LANGS: tuple[str, ...] = ("bilingual", "fr", "en")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

SIGNUP_CONVERSION_MIN = Fraction(20, 100)  # Alert when < 20%
INSTALL_DROPOFF_MAX = Fraction(30, 100)  # Alert when > 30% (completion < 70%)
PREFLIGHT_FIRST_ROUND_MIN = Fraction(70, 100)
EASE_RATING_MIN = Fraction(42, 10)  # Target >= 4.2 / 5; alert when < 4.2
ZERO_GAME_LAG_TARGET = Fraction(1, 1)  # Target 100% (1.0)

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

COMPLIANCE_RULE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(never corporate hardware or work code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_RULE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(jamais de matériel ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "account_id",
        "api_key",
        "authorization",
        "client_ip",
        "comment",
        "e_mail",
        "email",
        "free_text",
        "gpu_uuid",
        "handle",
        "handles",
        "host_name",
        "hostname",
        "ip",
        "ip_address",
        "live_handles",
        "mac_address",
        "mail",
        "newest_registered",
        "notes",
        "raw_prompt",
        "remote_addr",
        "serial",
        "token",
        "user_id",
        "user_name",
        "username",
    }
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm[-_][A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOME_PATH_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')

HttpGetFn = Callable[[str, float], tuple[int, dict[str, Any]]]


class FunnelInputError(ValueError):
    """Raised when onboarding funnel input data or CLI configuration is malformed."""


@dataclass(frozen=True)
class NormalizedFunnelData:
    """Validated, PII-free aggregate counts and exact satisfaction fractions."""

    source_kind: str
    landing_visits: int | None
    signups: int | None
    first_requests: int | None
    preflight_runs: int | None
    install_completed: int | None
    first_rounds: int | None
    feedback_responses: int | None
    mean_ease_fraction: Fraction | None
    zero_game_lag_fraction: Fraction | None
    feedback_by_profile: dict[str, int | None]
    arena_total_votes: int | None
    redacted_fields_dropped: int


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to 'bilingual', 'fr', or 'en'."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise FunnelInputError("Language must be 'bilingual', 'fr', or 'en'.")
    cleaned = (raw or "bilingual").strip().lower().replace("_", "-")
    if cleaned in ("bilingual", "both", "fr-en", "en-fr", "all"):
        return "bilingual"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    raise FunnelInputError(f"Unsupported language {raw!r}; expected 'bilingual', 'fr', or 'en'.")


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host matches the strict HiveMind lateral-movement allowlist."""
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
    """Validate and normalize a gateway origin URL against the host and scheme allowlist."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise FunnelInputError("Gateway URL must be a non-empty string.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise FunnelInputError("Gateway URL must not contain whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise FunnelInputError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise FunnelInputError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise FunnelInputError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise FunnelInputError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise FunnelInputError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise FunnelInputError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise FunnelInputError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise FunnelInputError("Remote gateway URL must use https.")
    return cleaned


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a read-only, proxy-free HTTP GET request to an allowlisted gateway endpoint."""
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hivemind-onboarding-funnel-report/1.0",
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
    except urllib.error.URLError as exc:
        raise FunnelInputError(f"Unable to reach gateway endpoint {parts.path}: {exc.reason}") from exc
    except OSError as exc:
        raise FunnelInputError(f"Network error while reading {parts.path}: {exc}") from exc

    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise FunnelInputError(f"Malformed JSON from gateway endpoint {parts.path}.") from exc
    if not isinstance(parsed, dict):
        raise FunnelInputError(f"Expected JSON object from gateway endpoint {parts.path}.")
    return status_code, parsed


def fetch_live_gateway_snapshot(
    gateway_url: str,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch read-only summary endpoints from an allowlisted HiveMind gateway."""
    origin = validate_gateway_url(gateway_url)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)):
        raise FunnelInputError("timeout_s must be a finite positive number.")
    clamped_timeout = max(0.5, min(15.0, float(timeout_s)))

    endpoints = {
        "feedback_summary": "/v1/hivemind/onboarding/feedback/summary",
        "raid_status": "/v1/hivemind/raid/status",
        "live_telemetry": "/v1/hivemind/live-telemetry",
        "arena_summary": "/v1/hivemind/arena/summary",
    }
    bundle: dict[str, Any] = {"_source_kind": "live_gateway"}
    primary_ok = 0

    for key, path in endpoints.items():
        status_code, payload = http_get(f"{origin}{path}", clamped_timeout)
        if status_code == 200 and isinstance(payload, Mapping):
            bundle[key] = dict(payload)
            if key != "arena_summary":
                primary_ok += 1
        elif key != "arena_summary" and status_code >= 500:
            raise FunnelInputError(f"Gateway endpoint {path} returned HTTP {status_code}.")

    if primary_ok == 0:
        raise FunnelInputError("No live onboarding summary endpoints returned HTTP 200.")
    return bundle


def count_sensitive_fields(obj: Any) -> int:
    """Count PII-bearing or identifier keys present in raw input so they can be audited as dropped."""
    dropped = 0
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key_norm = str(k).strip().lower()
            if key_norm in _SENSITIVE_KEYS:
                if v not in (None, "", [], {}):
                    dropped += 1
            else:
                dropped += count_sensitive_fields(v)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            dropped += count_sensitive_fields(item)
    return dropped


def scan_for_pii_leaks(text: str) -> list[str]:
    """Scan serialized report output for accidental PII, tokens, local user paths, or forbidden keys."""
    if not isinstance(text, str):
        raise FunnelInputError("scan_for_pii_leaks requires a string.")
    findings: list[str] = []
    if _EMAIL_RE.search(text):
        findings.append("email_address_detected")
    if _API_KEY_RE.search(text):
        findings.append("hivemind_api_key_detected")
    if _BEARER_RE.search(text):
        findings.append("bearer_token_detected")
    if _IPV4_RE.search(text):
        findings.append("ipv4_literal_detected")
    if _HOME_PATH_RE.search(text):
        findings.append("local_home_path_detected")
    if _FORBIDDEN_KEY_RE.search(text):
        findings.append("forbidden_key_detected")
    return findings


def _assert_clean_output(text: str) -> str:
    leaks = scan_for_pii_leaks(text)
    if leaks:
        raise FunnelInputError(f"Output safety check failed: {', '.join(leaks)}")
    return text


def _opt_nonneg_int(val: Any, field_name: str) -> int | None:
    """Validate an optional non-negative integer, rejecting booleans, strings, floats, and negatives."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise FunnelInputError(f"Field '{field_name}' must be a non-negative integer.")
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer():
            raise FunnelInputError(f"Field '{field_name}' must be an exact integer.")
    int_val = int(val)
    if int_val < 0:
        raise FunnelInputError(f"Field '{field_name}' must be >= 0 (got {int_val}).")
    return int_val


def _opt_bounded_fraction(val: Any, field_name: str, low: Fraction, high: Fraction) -> Fraction | None:
    """Validate an optional finite number in [low, high] and convert to exact Fraction."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise FunnelInputError(f"Field '{field_name}' must be a finite number.")
    fval = float(val)
    if not math.isfinite(fval):
        raise FunnelInputError(f"Field '{field_name}' must be finite.")
    exact = Fraction(val) if isinstance(val, int) else Fraction(str(fval))
    if exact < low or exact > high:
        raise FunnelInputError(
            f"Field '{field_name}' must be between {float(low)} and {float(high)} (got {fval})."
        )
    return exact


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, Any, str]:
    """Return (found, value, key_name) for the first matching key present in mapping."""
    for key in keys:
        if key in mapping:
            return True, mapping[key], key
    return False, None, ""


def _extract_feedback(
    raw: Mapping[str, Any],
    feedback_override: Mapping[str, Any] | None = None,
) -> tuple[int | None, Fraction | None, Fraction | None, dict[str, int | None]]:
    """Extract and validate aggregate satisfaction metrics from summary dicts or anonymous item lists."""
    if feedback_override is not None and not isinstance(feedback_override, Mapping):
        raise FunnelInputError("feedback_override must be a JSON object.")

    fb_block: Mapping[str, Any] | None = feedback_override
    if fb_block is None:
        for candidate_key in ("feedback_summary", "onboarding_feedback", "feedback"):
            if candidate_key in raw and raw[candidate_key] is not None:
                sub = raw[candidate_key]
                if not isinstance(sub, Mapping):
                    raise FunnelInputError(f"Field '{candidate_key}' must be a JSON object.")
                fb_block = sub
                break
        if fb_block is None and isinstance(raw.get("raid_status"), Mapping):
            raid_fb = raw["raid_status"].get("onboarding_feedback")
            if raid_fb is not None:
                if not isinstance(raid_fb, Mapping):
                    raise FunnelInputError("Field 'raid_status.onboarding_feedback' must be a JSON object.")
                fb_block = raid_fb

    items_raw: Any = None
    for items_key in ("feedback_items", "responses"):
        if items_key in raw and raw[items_key] is not None:
            items_raw = raw[items_key]
            break
        if fb_block is not None and items_key in fb_block and fb_block[items_key] is not None:
            items_raw = fb_block[items_key]
            break

    if items_raw is not None:
        if isinstance(items_raw, (str, bytes, bytearray, Mapping)) or not isinstance(items_raw, Sequence):
            raise FunnelInputError("Feedback items list must be a JSON array of objects.")
        if len(items_raw) == 0:
            return 0, None, None, {"dev": 0, "gamer": 0}
        ease_sum = Fraction(0, 1)
        lag_ok_count = 0
        by_prof = {"dev": 0, "gamer": 0}
        for idx, entry in enumerate(items_raw):
            if not isinstance(entry, Mapping):
                raise FunnelInputError(f"Feedback item at index {idx} must be a JSON object.")
            if "ease_rating" not in entry:
                raise FunnelInputError(f"Feedback item at index {idx} is missing 'ease_rating'.")
            rating = _opt_bounded_fraction(
                entry["ease_rating"],
                f"feedback_items[{idx}].ease_rating",
                Fraction(1, 1),
                Fraction(5, 1),
            )
            assert rating is not None
            ease_sum += rating

            lag_val = entry.get("zero_game_lag", True)
            if not isinstance(lag_val, bool):
                raise FunnelInputError(f"Field 'feedback_items[{idx}].zero_game_lag' must be a boolean.")
            if lag_val:
                lag_ok_count += 1

            if "profile" in entry and entry["profile"] is not None:
                if isinstance(entry["profile"], bool) or not isinstance(entry["profile"], str):
                    raise FunnelInputError(f"Field 'feedback_items[{idx}].profile' must be 'dev' or 'gamer'.")
                prof_norm = entry["profile"].strip().lower()
                if prof_norm not in VALID_PROFILES:
                    raise FunnelInputError(
                        f"Field 'feedback_items[{idx}].profile' must be 'dev' or 'gamer' (got {entry['profile']!r})."
                    )
                by_prof[prof_norm] += 1

        total = len(items_raw)
        return (
            total,
            ease_sum / Fraction(total, 1),
            Fraction(lag_ok_count, total),
            {"dev": by_prof["dev"], "gamer": by_prof["gamer"]},
        )

    source_map = fb_block if fb_block is not None else raw
    has_total, total_raw, total_key = _first_present(
        source_map,
        ("total_responses", "feedback_responses", "responses_count"),
    )
    total_responses = _opt_nonneg_int(total_raw, total_key or "total_responses") if has_total else None

    ease_frac = _opt_bounded_fraction(
        source_map.get("mean_ease_rating"),
        "mean_ease_rating",
        Fraction(1, 1),
        Fraction(5, 1),
    )
    lag_frac = _opt_bounded_fraction(
        source_map.get("zero_game_lag_ratio"),
        "zero_game_lag_ratio",
        Fraction(0, 1),
        Fraction(1, 1),
    )

    if total_responses == 0 and (ease_frac is not None or lag_frac is not None):
        raise FunnelInputError("mean_ease_rating and zero_game_lag_ratio must be null when total_responses is 0.")

    by_profile_out: dict[str, int | None] = {"dev": None, "gamer": None}
    if "by_profile" in source_map and source_map["by_profile"] is not None:
        bp = source_map["by_profile"]
        if not isinstance(bp, Mapping):
            raise FunnelInputError("Field 'by_profile' must be a JSON object.")
        for k in bp:
            if str(k) not in VALID_PROFILES:
                raise FunnelInputError(f"Unsupported feedback profile key {k!r} in 'by_profile'.")
        dev_n = _opt_nonneg_int(bp.get("dev"), "by_profile.dev")
        gamer_n = _opt_nonneg_int(bp.get("gamer"), "by_profile.gamer")
        by_profile_out = {"dev": dev_n, "gamer": gamer_n}
        known_sum = (dev_n or 0) + (gamer_n or 0)
        if total_responses is not None and known_sum > total_responses:
            raise FunnelInputError("Sum of 'by_profile' counts cannot exceed 'total_responses'.")
        if total_responses is None and (dev_n is not None or gamer_n is not None):
            total_responses = known_sum

    return total_responses, ease_frac, lag_frac, by_profile_out


def _extract_router_telemetry_counts(
    raw: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """Derive bounded aggregate funnel counts from read-only router telemetry or raid status payloads."""
    raid_doc = raw.get("raid_status") if isinstance(raw.get("raid_status"), Mapping) else None
    lt_doc = raw.get("live_telemetry") if isinstance(raw.get("live_telemetry"), Mapping) else None

    if raid_doc is None and (raw.get("object") == "hivemind.raid_noel.status" or "swarm" in raw):
        if "swarm" in raw and raw["swarm"] is not None and not isinstance(raw["swarm"], Mapping):
            raise FunnelInputError("Field 'swarm' must be a JSON object.")
        raid_doc = raw

    if lt_doc is None and (
        raw.get("object") == "hivemind.live_telemetry"
        or ("accounts" in raw and isinstance(raw.get("accounts"), Mapping))
        or ("opentrainer" in raw and isinstance(raw.get("opentrainer"), Mapping))
    ):
        lt_doc = raw

    signups: int | None = None
    first_requests: int | None = None
    preflight_runs: int | None = None
    install_completed: int | None = None
    first_rounds: int | None = None

    if raid_doc is not None:
        swarm = raid_doc.get("swarm")
        if swarm is not None and not isinstance(swarm, Mapping):
            raise FunnelInputError("Field 'raid_status.swarm' must be a JSON object.")
        if isinstance(swarm, Mapping):
            signups = _opt_nonneg_int(swarm.get("registered_accounts"), "swarm.registered_accounts")
            active_24h = _opt_nonneg_int(swarm.get("active_accounts_24h"), "swarm.active_accounts_24h")
            if signups is not None and active_24h is not None:
                first_requests = min(signups, active_24h)

            tr_acc = _opt_nonneg_int(swarm.get("registered_trainer_accounts"), "swarm.registered_trainer_accounts")
            tr_nodes = _opt_nonneg_int(swarm.get("registered_trainer_nodes"), "swarm.registered_trainer_nodes")
            tr_active = _opt_nonneg_int(swarm.get("active_trainers"), "swarm.active_trainers")
            tr_rounds = _opt_nonneg_int(
                swarm.get("accepted_opentrainer_rounds"),
                "swarm.accepted_opentrainer_rounds",
            )
            candidates = [c for c in (tr_acc, tr_nodes) if c is not None]
            if candidates:
                preflight_runs = max(candidates)
                if tr_nodes is not None:
                    install_completed = min(preflight_runs, tr_nodes)
                cap = install_completed if install_completed is not None else preflight_runs
                if tr_rounds is not None:
                    if tr_rounds == 0:
                        first_rounds = 0
                    else:
                        first_rounds = min(cap, max(tr_active or 0, min(tr_rounds, cap)))

    if lt_doc is not None:
        accounts = lt_doc.get("accounts")
        if accounts is not None and not isinstance(accounts, Mapping):
            raise FunnelInputError("Field 'accounts' must be a JSON object.")
        if isinstance(accounts, Mapping):
            lt_signups = _opt_nonneg_int(
                accounts.get("total_registered_accounts"),
                "accounts.total_registered_accounts",
            )
            lt_active = _opt_nonneg_int(accounts.get("active_accounts_24h"), "accounts.active_accounts_24h")
            if signups is None and lt_signups is not None:
                signups = lt_signups
            if first_requests is None and signups is not None and lt_active is not None:
                first_requests = min(signups, lt_active)

        ot = lt_doc.get("opentrainer")
        if ot is not None and not isinstance(ot, Mapping):
            raise FunnelInputError("Field 'opentrainer' must be a JSON object.")
        if isinstance(ot, Mapping) and preflight_runs is None:
            ot_acc = _opt_nonneg_int(
                ot.get("registered_trainer_accounts_count"),
                "opentrainer.registered_trainer_accounts_count",
            )
            ot_nodes = _opt_nonneg_int(ot.get("total_registered_nodes"), "opentrainer.total_registered_nodes")
            ot_active = _opt_nonneg_int(ot.get("active_trainer_nodes"), "opentrainer.active_trainer_nodes")
            ot_rounds = _opt_nonneg_int(ot.get("accepted_rounds_count"), "opentrainer.accepted_rounds_count")
            candidates = [c for c in (ot_acc, ot_nodes) if c is not None]
            if candidates:
                preflight_runs = max(candidates)
                if ot_nodes is not None:
                    install_completed = min(preflight_runs, ot_nodes)
                cap = install_completed if install_completed is not None else preflight_runs
                if ot_rounds is not None:
                    if ot_rounds == 0:
                        first_rounds = 0
                    else:
                        first_rounds = min(cap, max(ot_active or 0, min(ot_rounds, cap)))

    return signups, first_requests, preflight_runs, install_completed, first_rounds


def normalize_funnel_snapshot(
    raw: Mapping[str, Any],
    *,
    feedback_override: Mapping[str, Any] | None = None,
) -> NormalizedFunnelData:
    """Validate and normalize any supported funnel JSON snapshot or live endpoint bundle."""
    if not isinstance(raw, Mapping):
        raise FunnelInputError("Funnel snapshot root must be a JSON object.")

    redacted_dropped = count_sensitive_fields(raw)
    if feedback_override is not None:
        redacted_dropped += count_sensitive_fields(feedback_override)

    source_kind = "live_gateway" if raw.get("_source_kind") == "live_gateway" else "snapshot"

    funnel_sub: Mapping[str, Any] = raw
    for container_key in ("funnel", "counts", "stages"):
        if container_key in raw and raw[container_key] is not None:
            if not isinstance(raw[container_key], Mapping):
                raise FunnelInputError(f"Field '{container_key}' must be a JSON object.")
            funnel_sub = raw[container_key]
            break

    has_vis, vis_raw, vis_k = _first_present(
        funnel_sub,
        ("landing_visits", "visitors", "signup_page_views", "page_views"),
    )
    has_sig, sig_raw, sig_k = _first_present(
        funnel_sub,
        ("signups", "signup_count", "registered_accounts", "keys_issued"),
    )
    has_req, req_raw, req_k = _first_present(
        funnel_sub,
        ("first_requests", "first_request_count", "activated_accounts", "accounts_with_first_request"),
    )
    has_pre, pre_raw, pre_k = _first_present(
        funnel_sub,
        ("preflight_runs", "preflight_count", "preflight_checks", "preflight_passed", "install_attempts"),
    )
    has_ins, ins_raw, ins_k = _first_present(
        funnel_sub,
        ("install_completed", "installs_completed", "nodes_installed"),
    )
    has_rnd, rnd_raw, rnd_k = _first_present(
        funnel_sub,
        ("first_rounds", "first_round_count", "completed_first_round", "nodes_with_first_round"),
    )

    landing_visits = _opt_nonneg_int(vis_raw, vis_k) if has_vis else None
    exp_signups = _opt_nonneg_int(sig_raw, sig_k) if has_sig else None
    exp_first_requests = _opt_nonneg_int(req_raw, req_k) if has_req else None
    exp_preflight = _opt_nonneg_int(pre_raw, pre_k) if has_pre else None
    exp_install = _opt_nonneg_int(ins_raw, ins_k) if has_ins else None
    exp_first_rounds = _opt_nonneg_int(rnd_raw, rnd_k) if has_rnd else None

    # Validate explicit funnel monotonicity before falling back to router telemetry
    if landing_visits is not None and exp_signups is not None and exp_signups > landing_visits:
        raise FunnelInputError(
            f"Malformed funnel: signups ({exp_signups}) cannot exceed landing_visits ({landing_visits})."
        )
    if exp_signups is not None and exp_first_requests is not None and exp_first_requests > exp_signups:
        raise FunnelInputError(
            f"Malformed funnel: first_requests ({exp_first_requests}) cannot exceed signups ({exp_signups})."
        )
    if exp_signups is None and exp_first_requests is not None and exp_first_requests > 0:
        raise FunnelInputError("Malformed funnel: first_requests > 0 provided without signups denominator.")
    if exp_preflight is not None and exp_install is not None and exp_install > exp_preflight:
        raise FunnelInputError(
            f"Malformed funnel: install_completed ({exp_install}) cannot exceed preflight_runs ({exp_preflight})."
        )
    if exp_preflight is not None and exp_first_rounds is not None and exp_first_rounds > exp_preflight:
        raise FunnelInputError(
            f"Malformed funnel: first_rounds ({exp_first_rounds}) cannot exceed preflight_runs ({exp_preflight})."
        )
    if exp_install is not None and exp_first_rounds is not None and exp_first_rounds > exp_install:
        raise FunnelInputError(
            f"Malformed funnel: first_rounds ({exp_first_rounds}) cannot exceed install_completed ({exp_install})."
        )
    if exp_preflight is None and exp_install is None and exp_first_rounds is not None and exp_first_rounds > 0:
        raise FunnelInputError("Malformed funnel: first_rounds > 0 provided without preflight_runs denominator.")

    rt_signups, rt_requests, rt_preflight, rt_install, rt_rounds = _extract_router_telemetry_counts(raw)

    signups = exp_signups if exp_signups is not None else rt_signups
    first_requests = exp_first_requests if exp_first_requests is not None else rt_requests
    preflight_runs = exp_preflight if exp_preflight is not None else rt_preflight
    install_completed = exp_install if exp_install is not None else rt_install
    first_rounds = exp_first_rounds if exp_first_rounds is not None else rt_rounds

    if signups is not None and first_requests is not None and first_requests > signups:
        raise FunnelInputError("Malformed funnel: first_requests exceeds signups.")
    if preflight_runs is not None and first_rounds is not None and first_rounds > preflight_runs:
        raise FunnelInputError("Malformed funnel: first_rounds exceeds preflight_runs.")

    fb_total, ease_frac, lag_frac, fb_by_prof = _extract_feedback(raw, feedback_override=feedback_override)

    arena_votes: int | None = None
    arena_doc = raw.get("arena_summary")
    if arena_doc is not None:
        if not isinstance(arena_doc, Mapping):
            raise FunnelInputError("Field 'arena_summary' must be a JSON object.")
        arena_votes = _opt_nonneg_int(arena_doc.get("total_votes"), "arena_summary.total_votes")
    elif raw.get("object") == "hivemind.blind_ab_arena.summary" or "arena_total_votes" in funnel_sub:
        raw_v = raw.get("total_votes") if "total_votes" in raw else funnel_sub.get("arena_total_votes")
        arena_votes = _opt_nonneg_int(raw_v, "arena_total_votes")

    return NormalizedFunnelData(
        source_kind=source_kind,
        landing_visits=landing_visits,
        signups=signups,
        first_requests=first_requests,
        preflight_runs=preflight_runs,
        install_completed=install_completed,
        first_rounds=first_rounds,
        feedback_responses=fb_total,
        mean_ease_fraction=ease_frac,
        zero_game_lag_fraction=lag_frac,
        feedback_by_profile=fb_by_prof,
        arena_total_votes=arena_votes,
        redacted_fields_dropped=redacted_dropped,
    )


def _round_frac(frac: Fraction | None, decimals: int = 4) -> float | None:
    return round(float(frac), decimals) if frac is not None else None


def _fmt_pct_en(frac: Fraction | None) -> str:
    return f"{float(frac) * 100.0:.1f}%" if frac is not None else "n/a"


def _fmt_pct_fr(frac: Fraction | None) -> str:
    if frac is None:
        return "n/a"
    return f"{float(frac) * 100.0:.1f} %".replace(".", ",")


def analyze_onboarding_funnel(
    raw: Mapping[str, Any],
    *,
    feedback_override: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Compute exact onboarding funnel ratios, comfort indicators, UX alerts, and SHA-256 receipt."""
    norm = normalize_funnel_snapshot(raw, feedback_override=feedback_override)

    visitor_to_signup_frac: Fraction | None = None
    if norm.landing_visits is not None and norm.landing_visits > 0 and norm.signups is not None:
        visitor_to_signup_frac = Fraction(norm.signups, norm.landing_visits)

    activation_frac: Fraction | None = None
    if norm.signups is not None and norm.signups > 0 and norm.first_requests is not None:
        activation_frac = Fraction(norm.first_requests, norm.signups)

    signup_conversion_frac = (
        visitor_to_signup_frac if visitor_to_signup_frac is not None else activation_frac
    )

    preflight_to_first_round_frac: Fraction | None = None
    if norm.preflight_runs is not None and norm.preflight_runs > 0 and norm.first_rounds is not None:
        preflight_to_first_round_frac = Fraction(norm.first_rounds, norm.preflight_runs)

    preflight_to_install_frac: Fraction | None = None
    if norm.preflight_runs is not None and norm.preflight_runs > 0 and norm.install_completed is not None:
        preflight_to_install_frac = Fraction(norm.install_completed, norm.preflight_runs)

    install_dropoff_frac: Fraction | None = None
    if preflight_to_first_round_frac is not None:
        install_dropoff_frac = Fraction(1, 1) - preflight_to_first_round_frac
        if preflight_to_install_frac is not None:
            install_dropoff_frac = max(install_dropoff_frac, Fraction(1, 1) - preflight_to_install_frac)
    elif preflight_to_install_frac is not None:
        install_dropoff_frac = Fraction(1, 1) - preflight_to_install_frac

    # Evaluate Section 5 UX alert thresholds using exact rational arithmetic
    observed_signup_candidates = [
        c for c in (visitor_to_signup_frac, activation_frac) if c is not None
    ]
    worst_signup_conv = min(observed_signup_candidates) if observed_signup_candidates else None

    low_signup_conversion = bool(
        worst_signup_conv is not None and worst_signup_conv < SIGNUP_CONVERSION_MIN
    )
    high_install_dropoff = bool(
        install_dropoff_frac is not None and install_dropoff_frac > INSTALL_DROPOFF_MAX
    )
    low_ease_rating = bool(
        norm.mean_ease_fraction is not None and norm.mean_ease_fraction < EASE_RATING_MIN
    )
    game_lag_detected = bool(
        norm.zero_game_lag_fraction is not None and norm.zero_game_lag_fraction < ZERO_GAME_LAG_TARGET
    )

    alerts: list[dict[str, Any]] = []
    if low_signup_conversion and worst_signup_conv is not None:
        obs = _round_frac(worst_signup_conv, 4)
        alerts.append(
            {
                "code": "low_signup_conversion",
                "severity": "warning",
                "metric": "signup_conversion_ratio" if visitor_to_signup_frac == worst_signup_conv else "activation_ratio",
                "observed": obs,
                "threshold": "< 0.20",
                "message_en": (
                    f"Signup conversion / activation is {_fmt_pct_en(worst_signup_conv)} "
                    "(below the 20.0% minimum UX threshold)."
                ),
                "message_fr": (
                    f"La conversion d'inscription / activation est de {_fmt_pct_fr(worst_signup_conv)} "
                    "(sous le seuil minimal UX de 20,0 %)."
                ),
            }
        )
    if high_install_dropoff and install_dropoff_frac is not None:
        obs = _round_frac(install_dropoff_frac, 4)
        alerts.append(
            {
                "code": "high_install_dropoff",
                "severity": "warning",
                "metric": "install_dropoff_ratio",
                "observed": obs,
                "threshold": "> 0.30",
                "message_en": (
                    f"Preflight-to-first-round install dropoff is {_fmt_pct_en(install_dropoff_frac)} "
                    "(above the 30.0% maximum UX dropoff threshold)."
                ),
                "message_fr": (
                    f"L'abandon entre le pré-contrôle et le 1er round est de {_fmt_pct_fr(install_dropoff_frac)} "
                    "(au-dessus du plafond UX de 30,0 %)."
                ),
            }
        )
    if low_ease_rating and norm.mean_ease_fraction is not None:
        obs_ease = _round_frac(norm.mean_ease_fraction, 2)
        ease_fr = f"{obs_ease:.2f}".replace(".", ",") if obs_ease is not None else "n/a"
        alerts.append(
            {
                "code": "low_ease_rating",
                "severity": "warning",
                "metric": "mean_ease_rating",
                "observed": obs_ease,
                "threshold": "< 4.20",
                "message_en": (
                    f"Mean usability ease rating is {obs_ease:.2f} / 5 "
                    "(below the >= 4.20 / 5 target)."
                ),
                "message_fr": (
                    f"L'indice moyen de facilité est de {ease_fr} / 5 "
                    "(sous la cible >= 4,20 / 5)."
                ),
            }
        )
    if game_lag_detected and norm.zero_game_lag_fraction is not None:
        obs_lag = _round_frac(norm.zero_game_lag_fraction, 4)
        alerts.append(
            {
                "code": "game_lag_detected",
                "severity": "warning",
                "metric": "zero_game_lag_ratio",
                "observed": obs_lag,
                "threshold": "< 1.00",
                "message_en": (
                    f"Zero-game-lag comfort ratio is {_fmt_pct_en(norm.zero_game_lag_fraction)} "
                    "(below the 100.0% Gamer-Whisper target)."
                ),
                "message_fr": (
                    f"Le taux de confort zéro gêne en jeu est de {_fmt_pct_fr(norm.zero_game_lag_fraction)} "
                    "(sous la cible Gamer-Whisper de 100,0 %)."
                ),
            }
        )

    has_any_metric = any(
        m is not None
        for m in (
            worst_signup_conv,
            install_dropoff_frac,
            norm.mean_ease_fraction,
            norm.zero_game_lag_fraction,
        )
    )
    if not has_any_metric:
        status = "insufficient_data"
    elif alerts:
        status = "alert"
    else:
        status = "healthy"

    stamp = (
        generated_at_utc.strip()
        if isinstance(generated_at_utc, str) and generated_at_utc.strip()
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign": CAMPAIGN_ID,
        "generated_at_utc": stamp,
        "source_kind": norm.source_kind,
        "status": status,
        "activation_ratio": _round_frac(activation_frac, 4),
        "signup_conversion_ratio": _round_frac(signup_conversion_frac, 4),
        "visitor_to_signup_ratio": _round_frac(visitor_to_signup_frac, 4),
        "preflight_to_first_round_ratio": _round_frac(preflight_to_first_round_frac, 4),
        "install_dropoff_ratio": _round_frac(install_dropoff_frac, 4),
        "mean_ease_rating": _round_frac(norm.mean_ease_fraction, 2),
        "zero_game_lag_ratio": _round_frac(norm.zero_game_lag_fraction, 4),
        "counts": {
            "landing_visits": norm.landing_visits,
            "signups": norm.signups,
            "first_requests": norm.first_requests,
            "preflight_runs": norm.preflight_runs,
            "install_completed": norm.install_completed,
            "first_rounds": norm.first_rounds,
            "feedback_responses": norm.feedback_responses,
            "feedback_by_profile": dict(norm.feedback_by_profile),
            "arena_total_votes": norm.arena_total_votes,
        },
        "targets": {
            "signup_conversion_min_ratio": 0.2,
            "install_dropoff_max_ratio": 0.3,
            "preflight_to_first_round_min_ratio": 0.7,
            "mean_ease_rating_min": 4.2,
            "zero_game_lag_target_ratio": 1.0,
        },
        "target_checks": {
            "signup_conversion_ok": (not low_signup_conversion) if worst_signup_conv is not None else None,
            "install_dropoff_ok": (not high_install_dropoff) if install_dropoff_frac is not None else None,
            "ease_rating_ok": (not low_ease_rating) if norm.mean_ease_fraction is not None else None,
            "zero_game_lag_ok": (not game_lag_detected) if norm.zero_game_lag_fraction is not None else None,
        },
        "alert_flags": {
            "low_signup_conversion": low_signup_conversion,
            "high_install_dropoff": high_install_dropoff,
            "low_ease_rating": low_ease_rating,
            "game_lag_detected": game_lag_detected,
        },
        "alerts": alerts,
        "privacy": {
            "zero_pii": True,
            "aggregated_only": True,
            "redacted_fields_dropped": norm.redacted_fields_dropped,
        },
        "compliance": {
            "personal_pc_only": True,
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
            "rule_en": COMPLIANCE_RULE_EN,
            "rule_fr": COMPLIANCE_RULE_FR,
        },
    }

    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _assert_clean_output(json.dumps(receipt, ensure_ascii=False))
    return receipt


def serialize_receipt_json(receipt: Mapping[str, Any]) -> str:
    """Serialize a funnel receipt to formatted, PII-audited JSON."""
    rendered = json.dumps(dict(receipt), indent=2, ensure_ascii=False) + "\n"
    return _assert_clean_output(rendered)


def _fmt_ratio_cell(val: Any, *, fr: bool) -> str:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return "n/a"
    pct = f"{float(val) * 100.0:.1f}"
    return f"{pct.replace('.', ',')} %" if fr else f"{pct}%"


def _fmt_ease_cell(val: Any, *, fr: bool) -> str:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return "n/a"
    num = f"{float(val):.2f}"
    return f"{num.replace('.', ',')} / 5" if fr else f"{num} / 5"


def _fmt_int_cell(val: Any) -> str:
    if val is None or isinstance(val, bool) or not isinstance(val, int):
        return "n/a"
    return str(val)


def _render_fr_section(receipt: Mapping[str, Any]) -> str:
    counts = receipt.get("counts") if isinstance(receipt.get("counts"), Mapping) else {}
    by_prof = counts.get("feedback_by_profile") if isinstance(counts.get("feedback_by_profile"), Mapping) else {}
    alerts = receipt.get("alerts") if isinstance(receipt.get("alerts"), Sequence) else []
    status = str(receipt.get("status", "insufficient_data"))
    status_label = {
        "healthy": "SAIN (toutes les cibles UX respectées)",
        "alert": "ALERTE UX (action requise)",
        "insufficient_data": "DONNÉES INSUFFISANTES (aucune mesure active)",
    }.get(status, status)

    lines = [
        "## Bilan Entonnoir d'Adoption & Satisfaction — Noël 2026 (FR)",
        "",
        f"> **Conformité :** {COMPLIANCE_RULE_FR}",
        "> **Confidentialité :** Agrégation stricte `zero-PII` (aucun pseudo, e-mail, IP ni clé API).",
        "",
        f"- **État global :** `{status_label}`",
        f"- **Source :** `{receipt.get('source_kind', 'snapshot')}` | **Horodatage UTC :** `{receipt.get('generated_at_utc', 'n/a')}`",
        "",
        "| Indicateur (Section 5) | Valeur mesurée | Volumes agrégés | Cible / Seuil UX |",
        "| :--- | :--- | :--- | :--- |",
        (
            "| Activation Inscription -> 1re requête (`activation_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('activation_ratio'), fr=True)}` | "
            f"`{_fmt_int_cell(counts.get('first_requests'))} / {_fmt_int_cell(counts.get('signups'))}` | "
            "Conversion `>= 20,0 %` (alerte `< 20 %`) |"
        ),
        (
            "| Pré-contrôle -> 1er round (`preflight_to_first_round_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('preflight_to_first_round_ratio'), fr=True)}` | "
            f"`{_fmt_int_cell(counts.get('first_rounds'))} / {_fmt_int_cell(counts.get('preflight_runs'))}` | "
            "`>= 70,0 %` (abandon `<= 30,0 %`) |"
        ),
        (
            "| Abandon d'installation (`install_dropoff_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('install_dropoff_ratio'), fr=True)}` | "
            f"Installés : `{_fmt_int_cell(counts.get('install_completed'))}` | "
            "`<= 30,0 %` (alerte `> 30 %`) |"
        ),
        (
            "| Indice moyen de facilité (`mean_ease_rating`) | "
            f"`{_fmt_ease_cell(receipt.get('mean_ease_rating'), fr=True)}` | "
            f"Réponses : `{_fmt_int_cell(counts.get('feedback_responses'))}` "
            f"(dev `{_fmt_int_cell(by_prof.get('dev'))}`, gamer `{_fmt_int_cell(by_prof.get('gamer'))}`) | "
            "`>= 4,20 / 5` (alerte `< 4,2`) |"
        ),
        (
            "| Confort zéro gêne en jeu (`zero_game_lag_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('zero_game_lag_ratio'), fr=True)}` | "
            f"Votes arène A/B : `{_fmt_int_cell(counts.get('arena_total_votes'))}` | "
            "Cible `100,0 %` |"
        ),
        "",
        "### Alertes UX (FR)",
    ]
    if status == "insufficient_data":
        lines.append("- Aucune donnée d'entonnoir ou de satisfaction enregistrée pour le moment (`n/a`).")
    elif not alerts:
        lines.append("- Aucune alerte UX active : tous les indicateurs mesurés respectent les seuils de Noël.")
    else:
        for alert in alerts:
            if isinstance(alert, Mapping):
                lines.append(f"- **[{alert.get('code', 'alert')}]** {alert.get('message_fr', '')}")
    return "\n".join(lines)


def _render_en_section(receipt: Mapping[str, Any]) -> str:
    counts = receipt.get("counts") if isinstance(receipt.get("counts"), Mapping) else {}
    by_prof = counts.get("feedback_by_profile") if isinstance(counts.get("feedback_by_profile"), Mapping) else {}
    alerts = receipt.get("alerts") if isinstance(receipt.get("alerts"), Sequence) else []
    status = str(receipt.get("status", "insufficient_data"))
    status_label = {
        "healthy": "HEALTHY (all UX targets met)",
        "alert": "UX ALERT (attention needed)",
        "insufficient_data": "INSUFFICIENT DATA (no active measurements yet)",
    }.get(status, status)

    lines = [
        "## Onboarding Usability Funnel & Satisfaction Report — Holiday 2026 (EN)",
        "",
        f"> **Compliance:** {COMPLIANCE_RULE_EN}",
        "> **Privacy:** Strictly aggregated `zero-PII` telemetry (zero handles, emails, IPs, or API keys).",
        "",
        f"- **Overall Status:** `{status_label}`",
        f"- **Source:** `{receipt.get('source_kind', 'snapshot')}` | **UTC Timestamp:** `{receipt.get('generated_at_utc', 'n/a')}`",
        "",
        "| Metric (Section 5) | Measured Value | Aggregate Counts | Target / UX Threshold |",
        "| :--- | :--- | :--- | :--- |",
        (
            "| Signup-to-first-request (`activation_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('activation_ratio'), fr=False)}` | "
            f"`{_fmt_int_cell(counts.get('first_requests'))} / {_fmt_int_cell(counts.get('signups'))}` | "
            "Conversion `>= 20.0%` (alert `< 20%`) |"
        ),
        (
            "| Preflight-to-first-round (`preflight_to_first_round_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('preflight_to_first_round_ratio'), fr=False)}` | "
            f"`{_fmt_int_cell(counts.get('first_rounds'))} / {_fmt_int_cell(counts.get('preflight_runs'))}` | "
            "`>= 70.0%` (dropoff `<= 30.0%`) |"
        ),
        (
            "| Install dropoff (`install_dropoff_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('install_dropoff_ratio'), fr=False)}` | "
            f"Installed: `{_fmt_int_cell(counts.get('install_completed'))}` | "
            "`<= 30.0%` (alert `> 30%`) |"
        ),
        (
            "| Mean ease rating (`mean_ease_rating`) | "
            f"`{_fmt_ease_cell(receipt.get('mean_ease_rating'), fr=False)}` | "
            f"Responses: `{_fmt_int_cell(counts.get('feedback_responses'))}` "
            f"(dev `{_fmt_int_cell(by_prof.get('dev'))}`, gamer `{_fmt_int_cell(by_prof.get('gamer'))}`) | "
            "`>= 4.20 / 5` (alert `< 4.2`) |"
        ),
        (
            "| Zero-game-lag comfort (`zero_game_lag_ratio`) | "
            f"`{_fmt_ratio_cell(receipt.get('zero_game_lag_ratio'), fr=False)}` | "
            f"Blind A/B votes: `{_fmt_int_cell(counts.get('arena_total_votes'))}` | "
            "Target `100.0%` |"
        ),
        "",
        "### UX Alerts (EN)",
    ]
    if status == "insufficient_data":
        lines.append("- No onboarding funnel or satisfaction observations recorded yet (`n/a`).")
    elif not alerts:
        lines.append("- Zero active UX alerts: all measured onboarding metrics meet Holiday targets.")
    else:
        for alert in alerts:
            if isinstance(alert, Mapping):
                lines.append(f"- **[{alert.get('code', 'alert')}]** {alert.get('message_en', '')}")
    return "\n".join(lines)


def render_markdown_report(receipt: Mapping[str, Any], *, lang: str = "bilingual") -> str:
    """Render a clean, PII-free French, English, or bilingual Markdown funnel report."""
    if not isinstance(receipt, Mapping):
        raise FunnelInputError("receipt must be a mapping.")
    norm_lang = normalize_lang(lang)
    sha = str(receipt.get("receipt_sha256", "n/a"))

    sections: list[str] = ["# HiveMind — Noël / Holiday 2026 Onboarding Funnel Receipt", ""]
    if norm_lang in ("bilingual", "fr"):
        sections.append(_render_fr_section(receipt))
    if norm_lang == "bilingual":
        sections.extend(["", "---", ""])
    if norm_lang in ("bilingual", "en"):
        sections.append(_render_en_section(receipt))
    sections.extend(["", f"`receipt_sha256: {sha}`", ""])

    markdown = "\n".join(sections)
    return _assert_clean_output(markdown)


def load_json_file(path_str: str) -> dict[str, Any]:
    """Load and validate a JSON object from a local file path or '-' (stdin)."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise FunnelInputError("Snapshot path must be non-empty.")
    cleaned = path_str.strip()
    try:
        raw_text = sys.stdin.read() if cleaned == "-" else Path(cleaned).read_text(encoding="utf-8")
    except OSError as exc:
        raise FunnelInputError(f"Unable to read snapshot JSON file ({exc.strerror or 'io_error'}).") from exc

    if not raw_text.strip():
        raise FunnelInputError("Snapshot JSON input is empty.")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FunnelInputError(f"Invalid JSON syntax at line {exc.lineno}, column {exc.colno}.") from exc
    if not isinstance(parsed, dict):
        raise FunnelInputError("Snapshot JSON top-level value must be an object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for scripts/onboarding_funnel_report.py."""
    parser = argparse.ArgumentParser(
        prog="onboarding_funnel_report.py",
        description=(
            "Compute privacy-preserving HiveMind Holiday onboarding funnel & satisfaction metrics "
            "(Markdown FR/EN + SHA-256 JSON receipt)."
        ),
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to a local JSON funnel/telemetry snapshot file (or '-' for stdin).",
    )
    parser.add_argument(
        "--feedback-snapshot",
        type=str,
        default=None,
        help="Optional separate JSON file for /v1/hivemind/onboarding/feedback/summary.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Query live read-only summary endpoints on --gateway.",
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
        help="Markdown language: 'bilingual' (default), 'fr', or 'en'.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        dest="output_format",
        help="Stdout output format: 'markdown' (default) or 'json'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_flag",
        help="Shorthand for --format json.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional file path to write the zero-PII JSON receipt.",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional file path to write the Markdown report.",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=None,
        help="Optional fixed ISO-8601 UTC timestamp for deterministic receipts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=HTTP_TIMEOUT_S,
        help="HTTP GET timeout in seconds for --live mode (default: 5.0).",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 1 when one or more UX alerts are active.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_get: HttpGetFn = default_http_get_json,
) -> int:
    """CLI entrypoint returning 0 (ok), 1 (UX alert with --fail-on-alert), or 2 (input/network error)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        lang = normalize_lang(args.lang)
        if args.snapshot and (args.live or args.gateway):
            raise FunnelInputError("Provide either --snapshot or --live/--gateway, not both.")

        if args.snapshot:
            raw_data = load_json_file(args.snapshot)
        elif args.live or args.gateway:
            gw_url = args.gateway if args.gateway is not None else DEFAULT_GATEWAY
            raw_data = fetch_live_gateway_snapshot(gw_url, http_get=http_get, timeout_s=args.timeout)
        else:
            raise FunnelInputError("Specify --snapshot <path> or --live [--gateway <url>].")

        fb_override = load_json_file(args.feedback_snapshot) if args.feedback_snapshot else None
        receipt = analyze_onboarding_funnel(
            raw_data,
            feedback_override=fb_override,
            generated_at_utc=args.generated_at,
        )
        json_text = serialize_receipt_json(receipt)
        md_text = render_markdown_report(receipt, lang=lang)

        if args.json_out:
            out_json_path = Path(args.json_out)
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            out_json_path.write_text(json_text, encoding="utf-8")

        if args.markdown_out:
            out_md_path = Path(args.markdown_out)
            out_md_path.parent.mkdir(parents=True, exist_ok=True)
            out_md_path.write_text(md_text, encoding="utf-8")

        emit_json = args.json_flag or args.output_format == "json"
        sys.stdout.write(json_text if emit_json else md_text)

        if args.fail_on_alert and receipt["status"] == "alert":
            return 1
        return 0
    except FunnelInputError as exc:
        err_msg = str(exc)
        for leak in scan_for_pii_leaks(err_msg):
            err_msg = f"Input validation failed ({leak})."
        sys.stderr.write(f"error: {err_msg}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
