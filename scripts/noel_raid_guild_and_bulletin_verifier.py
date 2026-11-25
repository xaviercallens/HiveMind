#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference /raid-noel guild scoreboard, electricity Wh/EUR estimator, and 5-line bulletin verifier.

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``),
     workspace path, and ``.git/config`` remote URLs (including worktree ``gitdir:`` / ``commondir``
     indirection) BEFORE any network call or file write. Fails closed with exit code 2 (0 network
     calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Zero-Inference /raid-noel & /raid Guild, Electricity & Parity Gate
     (``verify_raid_noel_and_bulletin_readiness``):
     probes ``GET /raid-noel``, ``GET /raid``, ``GET /v1/hivemind/raid/status``, ``GET /noel``,
     ``GET /v1/hivemind/onboarding/manifest``, and ``GET /v1/hivemind/onboarding/feedback/summary``
     (verifying HTTP 200, unique per-request CSP nonces ``default-src 'none'``, 0 unsafe DOM sinks,
     0 static figures after ``strip_masked_keys_and_nonces`` strips ``nonce="..."`` attributes and
     ``hm-***XXXX`` masked-key suffixes, zero completions calls, and symbolic ``0.00 EUR / 0.00 USD``
     personal-PC compliance copy).
  3. Honest Electricity Wh/EUR Transparency & 5-Line FR/EN Daily Bulletin Audit
     (``audit_electricity_transparency_math``, ``render_raid_daily_bulletin``,
     ``audit_daily_bulletin_contract``):
     verifies exact ``gpu_watts * hours`` Wh / kWh / EUR home-meter calculations and strict ``'n/a'``
     5-line FR/EN daily bulletin fallback when live telemetry is absent, and emits signed SHA-256
     FR/EN JSON/Markdown receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_raid_guild_and_bulletin_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_raid_guild_and_bulletin_receipt"
ELECTRICITY_AUDIT_OBJECT = "hivemind.noel_electricity_transparency_audit"
GUILD_AUDIT_OBJECT = "hivemind.noel_raid_guild_scoreboard_audit"
BULLETIN_AUDIT_OBJECT = "hivemind.noel_daily_raid_bulletin_audit"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_KWH_TARIFF_EUR = 0.2516
DEFAULT_GPU_WATTS = 60.0
DEFAULT_HOURS = 2.0
HTTP_TIMEOUT_S = 5.0
NOT_AVAILABLE = "n/a"
BULLETIN_LINE_COUNT = 5
DISCORD_MAX_CHARS = 2000

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

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

RAID_PARITY_HTML_ROUTES: tuple[str, ...] = (
    "/raid-noel",
    "/raid",
    "/noel",
)

RAID_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/raid-noel", "html", "holiday"),
    ("/raid", "html", "holiday"),
    ("/v1/hivemind/raid/status", "json", "raid_status"),
    ("/noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
)

EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid",
    "/raid-noel",
    "/v1/hivemind/arena/summary",
    "/v1/hivemind/arena/vote",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)

EXPECTED_FEEDBACK_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "object",
        "total_responses",
        "mean_ease_rating",
        "zero_game_lag_ratio",
        "by_profile",
    }
)

EXPECTED_GUILD_IDS: tuple[str, ...] = (
    "team-python",
    "team-rust-cpp",
    "team-steam-rpg",
    "team-night-owl",
)

EXPECTED_GUILD_LABELS: tuple[str, ...] = (
    "Team Python",
    "Team Rust/C++",
    "Team Steam RPG",
    "Team Night Owl",
)

EXPECTED_ELECTRICITY_PRESET_IDS: tuple[str, ...] = (
    "eco",
    "whisper",
    "desktop",
)

EXPECTED_ELECTRICITY_DOM_IDS: tuple[str, ...] = (
    "calcWatts",
    "calcHours",
    "calcTariff",
    "calcResultWh",
    "calcResultEur",
    "calcResultNote",
    "calcComputeBtn",
    "calcPresetEco",
    "calcPresetWhisper",
    "calcPresetDesktop",
)

EXPECTED_RAID_DOM_IDS: tuple[str, ...] = (
    "raidLiveTrainers",
    "raidRegisteredAccounts",
    "raidAcceptedRounds",
    "raidFeedbackResponses",
    "guildBtn-team-python",
    "guildBtn-team-rust-cpp",
    "guildBtn-team-steam-rpg",
    "guildBtn-team-night-owl",
)

CORPORATE_MARKERS: tuple[str, ...] = (
    "amadeus",
    "corp",
    "enterprise",
    "intranet",
    "internal",
    "zscaler",
    "netskope",
    "bluecoat",
    "fortinet",
    "paloalto",
)

INSPECTED_ENV_KEYS: tuple[str, ...] = (
    "USERDOMAIN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_FR} | {COMPLIANCE_NOTICE_EN}"

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ISO_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T\s].*)?$")

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_MASKED_KEY_RE = re.compile(r"(hm(?:_live_|-))\*\*\*[A-Za-z0-9_-]{4,}")
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
)

_CORP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amadeus", re.compile(r"amadeus", re.IGNORECASE)),
    (
        "corp",
        re.compile(
            r"(?:^|[^a-z])corp(?:orate|s)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])",
            re.IGNORECASE,
        ),
    ),
    ("enterprise", re.compile(r"enterprise", re.IGNORECASE)),
    ("intranet", re.compile(r"intranet", re.IGNORECASE)),
    (
        "internal",
        re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])|[a-z0-9]internal(?:$|[^a-z])", re.IGNORECASE),
    ),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto|palo[-_.]?alto", re.IGNORECASE)),
)

_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\b(?:hm(?:_live_|-)|sk-(?:ant-|or-v1-|proj-)?|gh[pousr]_|hf_)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_TROJAN_BIDI_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)

HttpFn = Callable[..., Any]


class RaidGuildVerifierError(ValueError):
    """Raised when verifier inputs, host allowlists, math bounds, or safety guards fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise RaidGuildVerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise RaidGuildVerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_bulletin_lang(raw: str | None) -> str:
    """Normalize bulletin language selector to ``'bilingual'``, ``'fr'``, or ``'en'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise RaidGuildVerifierError("Bulletin language must be 'bilingual', 'fr', or 'en'.")
    cleaned = (raw or "bilingual").strip().lower().replace("_", "-")
    if cleaned in ("bilingual", "both", "fr-en", "en-fr", "all"):
        return "bilingual"
    return normalize_lang(cleaned)


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw credentials never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return "<invalid>"
    if cleaned.startswith("hm_live_"):
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-"):
        return f"hm-***{cleaned[-4:]}"
    return "<invalid>"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and ``hm-***XXXX`` masked suffixes."""
    if not isinstance(text, str):
        raise RaidGuildVerifierError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    cleaned = _MASKED_KEY_RE.sub(r"\1***XXXX", cleaned)
    return cleaned


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace OS user home prefixes with ``'~'`` and map repo files to repo-relative paths."""
    raw = str(target)
    if not re.match(r"^[A-Za-z]:[\\/]", raw) and Path(raw).is_absolute():
        try:
            rel = Path(raw).resolve().relative_to(REPO_ROOT)
            return str(rel).replace("\\", "/")
        except (OSError, ValueError):
            pass
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (
            resolved_home,
            resolved_home.replace("\\", "/"),
            resolved_home.replace("/", "\\"),
        ):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    redacted = "~" + suffix.replace("\\", "/")
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted)
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw keys, Bearer headers, emails, non-loopback IPv4 literals, and OS home paths."""
    scrubbed = text
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (
            resolved_home,
            resolved_home.replace("\\", "/"),
            resolved_home.replace("/", "\\"),
        ):
            if variant and variant in scrubbed:
                scrubbed = scrubbed.replace(variant, "~")
    scrubbed = redact_home_path(scrubbed, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def _collect_json_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_json_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_json_keys(item))
    return found


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON, CLI, bulletin, or Markdown text has 0 PII, raw keys, score digits, or static figures."""
    if not isinstance(text, str):
        raise RaidGuildVerifierError("scan_output_safety requires a string.")
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_SCORE_KEY_RE.search(text):
        issues.append("forbidden_score_key_detected")
    if _STATIC_FIG_UNIT_RE.search(strip_masked_keys_and_nonces(text)):
        issues.append("static_figure_detected")
    if _TROJAN_BIDI_RE.search(text):
        issues.append("trojan_bidi_detected")
    if _DISCORD_PING_RE.search(text):
        issues.append("discord_mass_ping_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise RaidGuildVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise RaidGuildVerifierError("Forbidden JSON key detected in verifier receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def sanitize_build_sha(raw: object) -> str:
    """Sanitize a ``build_sha`` value to lowercase alphanumeric/hex or ``'unknown'``."""
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or low == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


def validate_expected_sha(raw: str) -> str:
    """Validate an explicit ``expected_sha`` parameter."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise RaidGuildVerifierError(
            "Expected SHA must be a 6..64 character alphanumeric/hex string (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live ``build_sha`` against ``expected_sha`` (exact or >=7-char hex prefix)."""
    clean_live = sanitize_build_sha(live_sha)
    clean_exp = sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def sanitize_cloud_run_revision(raw: object) -> str | None:
    """Sanitize a Cloud Run revision name (``K_REVISION``) or return ``None``."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or not _REVISION_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return None
    return cleaned


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff ``host`` is in the strict HiveMind allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate gateway origin URL to ``https://*.run.app``, ``https://*.googleapis.com``, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RaidGuildVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise RaidGuildVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise RaidGuildVerifierError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise RaidGuildVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise RaidGuildVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise RaidGuildVerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise RaidGuildVerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise RaidGuildVerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise RaidGuildVerifierError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise RaidGuildVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise RaidGuildVerifierError("Remote gateway URL must use https.")
    return cleaned


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    allowed_post_paths: frozenset[str] = ALLOWED_POST_PATHS,
) -> str:
    """Enforce allowlisted host, zero completions/prompts, and read-only HTTP GET/HEAD."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RaidGuildVerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise RaidGuildVerifierError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise RaidGuildVerifierError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise RaidGuildVerifierError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise RaidGuildVerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise RaidGuildVerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise RaidGuildVerifierError(f"Zero-inference read-only guard blocked HTTP {verb!r} to {parts.path!r}.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects to untrusted hosts or inference completion endpoints."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_safe_http_request(new_req.get_method(), new_req.full_url)
        return new_req


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    text = raw_bytes.decode("utf-8", errors="replace")
    ctype = lower_headers.get("content-type", "").lower()
    if "application/json" in ctype or text.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            return int(status_code), dict(parsed)
        return int(status_code), {"data": parsed}
    return int(status_code), {
        "html": text,
        "text": text,
        "html_length": len(text),
        "text_length": len(text),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
    }


def default_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) preserving exact JSON schema keys."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload, allowed_post_paths=ALLOWED_POST_PATHS)
    if payload is not None:
        raise RaidGuildVerifierError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-raid-guild-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise RaidGuildVerifierError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str

    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    bounded_timeout = min(max(0.2, float(timeout_s)), 15.0)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw_hdrs = dict(resp.headers.items()) if resp.headers is not None else {}
            return _parse_http_raw_response(status, resp.read(524_288), raw_hdrs)
    except urllib.error.HTTPError as exc:
        err_hdrs = dict(exc.headers.items()) if exc.headers is not None else {}
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return _parse_http_raw_response(int(exc.code), err_bytes, err_hdrs)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}


def _looks_like_http_headers(candidate: Mapping[str, Any]) -> bool:
    if not candidate:
        return True
    keys = {str(k).strip().lower() for k in candidate}
    header_markers = {"content-type", "content-security-policy", "cache-control", "vary", "content-language"}
    body_markers = {"object", "status", "total_responses", "swarm", "friendly_guilds", "data", "detail", "error"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise RaidGuildVerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 2:
        status_code, body = raw_ret
        hdr_map: dict[str, str] = {}
    else:
        status_code, part1, part2 = raw_ret
        if isinstance(part1, str) and isinstance(part2, Mapping):
            body, raw_hdrs = part1, part2
        elif isinstance(part1, Mapping) and isinstance(part2, str):
            raw_hdrs, body = part1, part2
        elif isinstance(part1, Mapping) and isinstance(part2, Mapping):
            if _looks_like_http_headers(part2) and not _looks_like_http_headers(part1):
                body, raw_hdrs = part1, part2
            else:
                raw_hdrs, body = part1, part2
        else:
            raw_hdrs, body = (part1 if isinstance(part1, Mapping) else {}), part2
        hdr_map = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()}

    if isinstance(body, Mapping):
        if "html" in body or "text" in body:
            out = dict(body)
            if hdr_map:
                existing = out.get("headers")
                merged = (
                    {**hdr_map, **{str(k).strip().lower(): str(v) for k, v in existing.items()}}
                    if isinstance(existing, Mapping)
                    else hdr_map
                )
                out["headers"] = merged
                out.setdefault("csp", merged.get("content-security-policy", ""))
            return int(status_code), out
        if hdr_map:
            return int(status_code), {"_json": dict(body), "_headers": hdr_map}
        return int(status_code), dict(body)
    if isinstance(body, str):
        stripped = body.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    return int(status_code), (
                        {"_json": dict(parsed), "_headers": hdr_map} if hdr_map else dict(parsed)
                    )
                return int(status_code), {"data": parsed, "_headers": hdr_map} if hdr_map else {"data": parsed}
            except json.JSONDecodeError:
                pass
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "text_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body, "_headers": hdr_map} if hdr_map else {"data": body}


def _unwrap_json_body(normalized: Mapping[str, Any]) -> dict[str, Any]:
    inner = normalized.get("_json")
    if isinstance(inner, Mapping):
        return dict(inner)
    return {k: v for k, v in normalized.items() if k != "_headers"}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke an injectable HTTP callable."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=ALLOWED_POST_PATHS,
    )

    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    param_names = {p.name for p in params}
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    try:
        if has_varkw:
            raw = http_fn(
                verb,
                clean_url,
                payload=None,
                headers=dict(headers) if headers is not None else None,
                timeout_s=timeout_s,
            )
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = None
            if "headers" in param_names:
                kwargs["headers"] = dict(headers) if headers is not None else None
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, None)
            elif len(pos_params) == 4 and not has_varargs:
                raw = http_fn(
                    verb,
                    clean_url,
                    None,
                    dict(headers) if headers is not None else None,
                )
            else:
                raw = http_fn(
                    verb,
                    clean_url,
                    None,
                    dict(headers) if headers is not None else None,
                    timeout_s,
                )
    except RaidGuildVerifierError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    """Extract remote URLs and URL rewrite rules from ``.git/config`` text."""
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` from ``workspace`` (walking parents and following ``gitdir:``/``commondir``)."""
    candidates: list[Path] = []
    for root in (workspace, *workspace.parents):
        git_entry = root / ".git"
        if git_entry.is_dir():
            cfg = git_entry / "config"
            if cfg.is_file():
                candidates.append(cfg)
            break
        if git_entry.is_file():
            try:
                first_lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            if first_lines and first_lines[0].strip().lower().startswith("gitdir:"):
                raw_gitdir = first_lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gitdir) if Path(raw_gitdir).is_absolute() else (root / raw_gitdir).resolve()
                direct_cfg = gitdir / "config"
                if direct_cfg.is_file():
                    candidates.append(direct_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    try:
                        common_rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        common_rel = ""
                    if common_rel:
                        common_cfg = (gitdir / common_rel).resolve() / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
            break
    return candidates


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 writes, 0 leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = environ if environ is not None else (env if env is not None else os.environ)
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _record(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            matched_markers.add(marker)

    for raw_k, raw_v in env_map.items():
        key_up = str(raw_k).strip().upper()
        val_str = str(raw_v or "").strip()
        if not key_up or not val_str:
            continue
        if key_up.startswith("AMADEUS"):
            _record("env:AMADEUS*", "amadeus")
            for hit in _find_corporate_markers(val_str):
                matched_markers.add(hit)
        elif key_up == "ZSCALER_CERT_PATH":
            _record("env:ZSCALER_CERT_PATH", "zscaler")
        elif key_up == "USERDNSDOMAIN":
            dns_hits = _find_corporate_markers(val_str)
            _record("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "corp")
            for hit in dns_hits:
                matched_markers.add(hit)
        elif key_up in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(val_str):
                _record(f"env:{key_up}:{hit}", hit)

    ws_texts = [str(workspace)]
    try:
        resolved_ws = str(workspace.resolve())
        if resolved_ws not in ws_texts:
            ws_texts.append(resolved_ws)
    except OSError:
        pass
    for ws_txt in ws_texts:
        for hit in _find_corporate_markers(ws_txt):
            _record(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            try:
                cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_url in extract_git_config_remote_urls(cfg_text):
                for hit in _find_corporate_markers(remote_url):
                    _record(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"Local personal-PC compliance verified ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement d'entreprise détecté "
            f"(0 appel réseau, 0 écriture fichier). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE BLOCKED ({joined}): corporate environment indicator detected "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "compliance_gate",
            "lang": lang_code,
            "compliance_ok": compliance_ok,
            "corporate_detected": not compliance_ok,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "exit_code": 0 if compliance_ok else 2,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _compliance_blocked_receipt(
    mode: str,
    comp_doc: Mapping[str, Any],
    *,
    lang_code: str,
    gateway: str = DEFAULT_GATEWAY,
) -> dict[str, Any]:
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "ready": False,
            "parity_ok": False,
            "stale_404_drift": False,
            "redeploy_required": False,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": [str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL)],
            "exit_code": 2,
            "summary": str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL),
        }
    )


def _validate_bounded_num(val: object, name: str, *, upper: float) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        raise RaidGuildVerifierError(f"{name} must be a finite number.")
    fval = float(val)
    if fval < 0.0 or fval > upper:
        raise RaidGuildVerifierError(f"{name} must be between 0 and {upper:g}.")
    return fval


def _compute_wh_eur(gpu_watts: float, hours: float, kwh_tariff_eur: float) -> dict[str, float]:
    w = _validate_bounded_num(gpu_watts, "gpu_watts", upper=2000.0)
    h = _validate_bounded_num(hours, "hours", upper=8760.0)
    t = _validate_bounded_num(kwh_tariff_eur, "kwh_tariff_eur", upper=10.0)
    energy_wh = w * h
    energy_kwh = energy_wh / 1000.0
    estimated_eur = energy_kwh * t
    return {
        "gpu_watts": round(w, 2),
        "hours": round(h, 2),
        "kwh_tariff_eur": round(t, 4),
        "energy_wh": round(energy_wh, 2),
        "energy_kwh": round(energy_kwh, 4),
        "estimated_eur": round(estimated_eur, 4),
    }


def audit_electricity_transparency_math(
    gpu_watts: float = DEFAULT_GPU_WATTS,
    hours: float = DEFAULT_HOURS,
    kwh_tariff_eur: float = DEFAULT_KWH_TARIFF_EUR,
    *,
    raid_status: Mapping[str, Any] | None = None,
    raid_html: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify exact home-meter Wh / kWh / EUR transparency math, server formulas, presets, and HTML controls."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    requested = _compute_wh_eur(gpu_watts, hours, kwh_tariff_eur)

    issues: list[str] = []
    server_default_ok: bool | None = None
    server_default_est: dict[str, float] | None = None
    presets_audited: list[dict[str, Any]] = []
    html_calc_ok: bool | None = None

    if raid_status is not None:
        if not isinstance(raid_status, Mapping):
            raise RaidGuildVerifierError("raid_status must be a mapping or None.")
        est_blk = raid_status.get("electricity_estimator")
        if not isinstance(est_blk, Mapping):
            server_default_ok = False
            issues.append(
                "Bloc `electricity_estimator` absent dans `/v1/hivemind/raid/status`."
                if is_fr
                else "Missing `electricity_estimator` block in `/v1/hivemind/raid/status`."
            )
        else:
            f_wh = str(est_blk.get("formula_wh") or "")
            f_kwh = str(est_blk.get("formula_kwh") or "")
            f_eur = str(est_blk.get("formula_eur") or "")
            formulas_ok = (
                "gpu_watts" in f_wh
                and "hours" in f_wh
                and "1000" in f_kwh
                and "kwh_tariff_eur" in f_eur
            )
            if not formulas_ok:
                issues.append(
                    "Formules `electricity_estimator` invalides ou incomplètes."
                    if is_fr
                    else "Invalid or incomplete `electricity_estimator` transparency formulas."
                )

            def_raw = est_blk.get("default_estimate")
            if not isinstance(def_raw, Mapping):
                server_default_ok = False
                issues.append(
                    "`default_estimate` absent dans `electricity_estimator`."
                    if is_fr
                    else "Missing `default_estimate` inside `electricity_estimator`."
                )
            else:
                try:
                    dw = _validate_bounded_num(def_raw.get("gpu_watts"), "default_estimate.gpu_watts", upper=2000.0)
                    dh = _validate_bounded_num(def_raw.get("hours"), "default_estimate.hours", upper=8760.0)
                    dt = _validate_bounded_num(
                        def_raw.get("kwh_tariff_eur"), "default_estimate.kwh_tariff_eur", upper=10.0
                    )
                    rep_wh = _validate_bounded_num(def_raw.get("energy_wh"), "default_estimate.energy_wh", upper=2e7)
                    rep_kwh = _validate_bounded_num(
                        def_raw.get("energy_kwh"), "default_estimate.energy_kwh", upper=2e4
                    )
                    rep_eur = _validate_bounded_num(
                        def_raw.get("estimated_eur"), "default_estimate.estimated_eur", upper=2e5
                    )
                    expected_srv = _compute_wh_eur(dw, dh, dt)
                    exact_match = (
                        abs(rep_wh - expected_srv["energy_wh"]) <= 0.05
                        and abs(rep_kwh - expected_srv["energy_kwh"]) <= 0.0005
                        and abs(rep_eur - expected_srv["estimated_eur"]) <= 0.0005
                    )
                    server_default_ok = bool(formulas_ok and exact_match)
                    server_default_est = expected_srv
                    if not exact_match:
                        issues.append(
                            "Incohérence arithmétique dans `default_estimate` Wh/kWh/EUR."
                            if is_fr
                            else "Arithmetic mismatch in server `default_estimate` Wh/kWh/EUR."
                        )
                except RaidGuildVerifierError as exc:
                    server_default_ok = False
                    issues.append(str(exc))

            raw_presets = est_blk.get("presets_watts")
            if not isinstance(raw_presets, list):
                issues.append(
                    "`presets_watts` absent dans `electricity_estimator`."
                    if is_fr
                    else "Missing `presets_watts` list in `electricity_estimator`."
                )
                server_default_ok = False
            else:
                seen_pids: set[str] = set()
                for item in raw_presets:
                    if not isinstance(item, Mapping):
                        issues.append("Invalid preset entry in `presets_watts`.")
                        server_default_ok = False
                        continue
                    pid = str(item.get("id") or "")
                    seen_pids.add(pid)
                    pw = item.get("watts")
                    try:
                        calc_p = _compute_wh_eur(pw, requested["hours"], requested["kwh_tariff_eur"])  # type: ignore[arg-type]
                        presets_audited.append({"id": pid, **calc_p})
                    except RaidGuildVerifierError as exc:
                        issues.append(str(exc))
                        server_default_ok = False
                missing_pids = [p for p in EXPECTED_ELECTRICITY_PRESET_IDS if p not in seen_pids]
                if missing_pids:
                    issues.append(f"Missing electricity presets: {', '.join(missing_pids)}.")
                    server_default_ok = False

    if raid_html is not None:
        if not isinstance(raid_html, str):
            raise RaidGuildVerifierError("raid_html must be a string or None.")
        stripped = strip_masked_keys_and_nonces(raid_html)
        missing_dom = [eid for eid in EXPECTED_ELECTRICITY_DOM_IDS if eid not in stripped]
        has_unsafe = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped))
        low_html = stripped.lower()
        has_zero_fee = "0.00 eur" in low_html or "0,00 eur" in low_html
        html_calc_ok = bool(not missing_dom and not has_unsafe and has_zero_fee)
        if missing_dom:
            issues.append(f"Missing calculator DOM controls: {', '.join(missing_dom)}.")
        if has_unsafe:
            issues.append("Forbidden DOM HTML sink found in calculator script.")
        if not has_zero_fee:
            issues.append("Missing 0.00 EUR non-monetary electricity disclaimer in HTML.")

    math_ok = len(issues) == 0
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": ELECTRICITY_AUDIT_OBJECT,
            "lang": lang_code,
            "math_ok": math_ok,
            "requested_estimate": requested,
            "server_default_estimate_ok": server_default_ok,
            "server_default_estimate": server_default_est,
            "presets_audited": presets_audited,
            "html_calculator_ok": html_calc_ok,
            "hivemind_monetary_cost": FINANCIAL_PROMISE,
            "personal_pc_only": True,
            "issues": issues,
            "exit_code": 0 if math_ok else 2,
        }
    )


def _opt_nonneg_int(val: Any, name: str) -> tuple[bool, int | None]:
    if val is None:
        return True, None
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        return False, None
    return True, int(val)


def _opt_bounded_float(val: Any, *, low: float, high: float) -> tuple[bool, float | None]:
    if val is None:
        return True, None
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return False, None
    fval = float(val)
    if fval < low or fval > high:
        return False, None
    return True, fval


def audit_raid_guild_scoreboard(
    raid_status: Mapping[str, Any],
    *,
    raid_noel_html: str = "",
    raid_en_html: str = "",
    feedback_summary: Mapping[str, Any] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit ``/v1/hivemind/raid/status`` guild zero-fee contracts, swarm fields, and HTML scoreboard parity."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if not isinstance(raid_status, Mapping):
        raise RaidGuildVerifierError("raid_status must be a mapping.")

    issues: list[str] = []
    if ("sco" + "re") in _collect_json_keys(raid_status):
        issues.append("Forbidden JSON key detected in raid status payload.")
    if raid_status.get("object") != RAID_STATUS_OBJECT:
        issues.append(
            "Champ `object` != `hivemind.raid_noel.status`."
            if is_fr
            else "Invalid `object` in `/v1/hivemind/raid/status`."
        )

    comp = raid_status.get("compliance")
    comp_ok = bool(
        isinstance(comp, Mapping)
        and comp.get("personal_pc_only") is True
        and comp.get("corporate_hardware_or_code_forbidden") is True
        and comp.get("monetary_cost_eur") == "0.00 EUR"
        and comp.get("monetary_cost_usd") == "0.00 USD"
        and comp.get("license") == "AGPL-3.0"
    )
    if not comp_ok:
        issues.append(
            "Bloc `compliance` (PC perso / 0.00 EUR / 0.00 USD / AGPL-3.0) invalide."
            if is_fr
            else "Raid status `compliance` block missing personal-PC / 0.00 EUR-USD / AGPL-3.0 guarantees."
        )

    guilds_raw = raid_status.get("friendly_guilds")
    guild_ids: list[str] = []
    guilds_ok = isinstance(guilds_raw, list)
    if isinstance(guilds_raw, list):
        for g in guilds_raw:
            if not isinstance(g, Mapping):
                guilds_ok = False
                continue
            gid = str(g.get("guild_id") or "")
            guild_ids.append(gid)
            if g.get("symbolic_fee") != "0.00 EUR" or g.get("monetary_prize") != "0.00 EUR":
                guilds_ok = False
                issues.append(f"Guild {gid!r} violated symbolic 0.00 EUR non-monetary invariant.")
    missing_guilds = [gid for gid in EXPECTED_GUILD_IDS if gid not in guild_ids]
    if not guilds_ok or missing_guilds:
        guilds_ok = False
        issues.append(
            f"Guildes amicales incomplètes ou non symboliques : {', '.join(missing_guilds) or 'invalid'}."
            if is_fr
            else f"Friendly holiday guilds incomplete or non-symbolic: {', '.join(missing_guilds) or 'invalid'}."
        )

    swarm = raid_status.get("swarm")
    swarm_ok = isinstance(swarm, Mapping)
    if isinstance(swarm, Mapping):
        for field in (
            "active_trainers",
            "registered_trainer_nodes",
            "registered_trainer_accounts",
            "live_producers",
            "registered_producers",
            "registered_accounts",
            "active_accounts_24h",
            "accepted_opentrainer_rounds",
            "live_models",
        ):
            ok_field, _ = _opt_nonneg_int(swarm.get(field), field)
            if not ok_field:
                swarm_ok = False
                issues.append(f"Invalid swarm counter {field!r}.")
    else:
        issues.append("Missing `swarm` object in `/v1/hivemind/raid/status`.")

    fb_blk = raid_status.get("onboarding_feedback")
    fb_crosscheck_ok = True
    if not isinstance(fb_blk, Mapping):
        fb_crosscheck_ok = False
        issues.append("Missing `onboarding_feedback` object in `/v1/hivemind/raid/status`.")
    elif feedback_summary is not None:
        r_tot = fb_blk.get("total_responses")
        s_tot = feedback_summary.get("total_responses")
        if r_tot is not None and (
            r_tot != s_tot
            or fb_blk.get("by_profile") != feedback_summary.get("by_profile")
            or fb_blk.get("mean_ease_rating") != feedback_summary.get("mean_ease_rating")
            or fb_blk.get("zero_game_lag_ratio") != feedback_summary.get("zero_game_lag_ratio")
        ):
            fb_crosscheck_ok = False
            issues.append(
                "Divergence entre `/v1/hivemind/raid/status` et `/v1/hivemind/onboarding/feedback/summary`."
                if is_fr
                else "Feedback mismatch between `/v1/hivemind/raid/status` and `/v1/hivemind/onboarding/feedback/summary`."
            )

    html_parity_ok = True
    for label, raw_doc in (("raid-noel", raid_noel_html), ("raid", raid_en_html)):
        if not raw_doc:
            continue
        missing_dom = [
            item
            for item in (*EXPECTED_RAID_DOM_IDS, *EXPECTED_GUILD_LABELS)
            if item not in raw_doc
        ]
        if missing_dom:
            html_parity_ok = False
            issues.append(f"HTML /{label} missing scoreboard/guild elements: {', '.join(missing_dom)}.")

    scoreboard_ok = len(issues) == 0
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": GUILD_AUDIT_OBJECT,
            "lang": lang_code,
            "scoreboard_ok": scoreboard_ok,
            "compliance_ok": comp_ok,
            "guilds_ok": guilds_ok,
            "guild_ids": guild_ids,
            "swarm_ok": swarm_ok,
            "feedback_crosscheck_ok": fb_crosscheck_ok,
            "html_parity_ok": html_parity_ok,
            "issues": issues,
            "exit_code": 0 if scoreboard_ok else 2,
        }
    )


def _fmt_or_na(val: int | None) -> str:
    return NOT_AVAILABLE if val is None else str(val)


def _fmt_ease_or_na(val: float | None) -> str:
    return NOT_AVAILABLE if val is None else f"{val:.2f}/5"


def _fmt_pct_or_na(val: float | None) -> str:
    return NOT_AVAILABLE if val is None else f"{val * 100.0:.1f}%"


def _validate_iso_date_override(raw: str | None) -> str:
    if raw is None:
        return date.today().isoformat()
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.strip():
        raise RaidGuildVerifierError("date_override must be a valid 'YYYY-MM-DD' string.")
    match = _ISO_DATE_RE.match(raw.strip())
    if not match:
        raise RaidGuildVerifierError("date_override must match 'YYYY-MM-DD'.")
    day = match.group(1)
    try:
        date.fromisoformat(day)
    except ValueError as exc:
        raise RaidGuildVerifierError(f"Invalid calendar date {day!r}.") from exc
    return day


def render_raid_daily_bulletin(
    raid_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    gpu_watts: float = DEFAULT_GPU_WATTS,
    hours: float = DEFAULT_HOURS,
    kwh_tariff_eur: float = DEFAULT_KWH_TARIFF_EUR,
    lang: str = "bilingual",
    date_override: str | None = None,
) -> str:
    """Render a strict 5-line FR/EN/bilingual daily raid bulletin with ``'n/a'`` when telemetry is absent."""
    norm_lang = normalize_bulletin_lang(lang)
    day_str = _validate_iso_date_override(date_override)
    elec = _compute_wh_eur(gpu_watts, hours, kwh_tariff_eur)

    rs = raid_status if isinstance(raid_status, Mapping) else {}
    swarm = rs.get("swarm") if isinstance(rs.get("swarm"), Mapping) else {}
    raw_status = swarm.get("telemetry_status")
    status_tok = raw_status.strip() if isinstance(raw_status, str) and _SAFE_TOKEN_RE.fullmatch(raw_status.strip()) else None
    telemetry_absent = (
        raid_status is None
        or not swarm
        or status_tok is None
        or status_tok.upper() in UNRECEIVED_TELEMETRY_STATUSES
    )

    if telemetry_absent:
        telemetry_label = NOT_AVAILABLE if (status_tok is None or status_tok.upper() in ("UNAVAILABLE", "NONE")) else status_tok
        trainers_s = NOT_AVAILABLE
        producers_s = NOT_AVAILABLE
        accounts_s = NOT_AVAILABLE
        rounds_s = NOT_AVAILABLE
        models_s = NOT_AVAILABLE
    else:
        telemetry_label = status_tok
        _, active_t = _opt_nonneg_int(swarm.get("active_trainers"), "active_trainers")
        _, live_p = _opt_nonneg_int(swarm.get("live_producers"), "live_producers")
        _, reg_a = _opt_nonneg_int(swarm.get("registered_accounts"), "registered_accounts")
        _, acc_r = _opt_nonneg_int(swarm.get("accepted_opentrainer_rounds"), "accepted_opentrainer_rounds")
        _, live_m = _opt_nonneg_int(swarm.get("live_models"), "live_models")
        trainers_s = _fmt_or_na(active_t)
        producers_s = _fmt_or_na(live_p)
        accounts_s = _fmt_or_na(reg_a)
        rounds_s = _fmt_or_na(acc_r)
        models_s = _fmt_or_na(live_m)

    guilds_raw = rs.get("friendly_guilds")
    guilds_s = str(len(guilds_raw)) if isinstance(guilds_raw, list) and guilds_raw else NOT_AVAILABLE

    fb_src = (
        feedback_summary
        if isinstance(feedback_summary, Mapping)
        else (rs.get("onboarding_feedback") if isinstance(rs.get("onboarding_feedback"), Mapping) else None)
    )
    if fb_src is None:
        resp_s = NOT_AVAILABLE
        dev_s = NOT_AVAILABLE
        gamer_s = NOT_AVAILABLE
        ease_s = NOT_AVAILABLE
        lag_s = NOT_AVAILABLE
    else:
        _, tot_fb = _opt_nonneg_int(fb_src.get("total_responses"), "total_responses")
        by_p = fb_src.get("by_profile") if isinstance(fb_src.get("by_profile"), Mapping) else {}
        _, dev_cnt = _opt_nonneg_int(by_p.get("dev"), "by_profile.dev")
        _, gamer_cnt = _opt_nonneg_int(by_p.get("gamer"), "by_profile.gamer")
        _, mean_ease = _opt_bounded_float(fb_src.get("mean_ease_rating"), low=1.0, high=5.0)
        _, zero_lag = _opt_bounded_float(fb_src.get("zero_game_lag_ratio"), low=0.0, high=1.0)
        if tot_fb == 0:
            mean_ease = None
            zero_lag = None
        resp_s = _fmt_or_na(tot_fb)
        dev_s = _fmt_or_na(dev_cnt)
        gamer_s = _fmt_or_na(gamer_cnt)
        ease_s = _fmt_ease_or_na(mean_ease)
        lag_s = _fmt_pct_or_na(zero_lag)

    w_str = f"{int(elec['gpu_watts'])}" if float(elec["gpu_watts"]).is_integer() else f"{elec['gpu_watts']:.1f}"
    h_str = f"{int(elec['hours'])}" if float(elec["hours"]).is_integer() else f"{elec['hours']:.1f}"
    wh_str = f"{elec['energy_wh']:.1f}"
    kwh_str = f"{elec['energy_kwh']:.3f}"
    eur_str = f"{elec['estimated_eur']:.2f}"
    tariff_str = f"{elec['kwh_tariff_eur']:.4f}"

    if norm_lang == "fr":
        lines = [
            f"🎄 [Raid Noël HiveMind • {day_str}] Télémétrie : {telemetry_label} | Entraîneurs : {trainers_s} | Producteurs : {producers_s} | Comptes : {accounts_s}",
            f"🧩 Rondes OpenTrainer : {rounds_s} | Modèles : {models_s} | Guildes amicales : {guilds_s} (0,00 EUR)",
            f"⚡ Compteur perso estimé : {w_str} W × {h_str} h = {wh_str} Wh ({kwh_str} kWh ≈ {eur_str} EUR à {tariff_str} EUR/kWh)",
            f"🎮 Confort ({resp_s} avis, dev {dev_s} / gamer {gamer_s}) : facilité {ease_s} | zéro lag Steam {lag_s}",
            "🛡️ PC perso uniquement (règle Amadeus, 0,00 EUR / 0,00 USD, AGPL-3.0) • /raid-noel • /noel",
        ]
    elif norm_lang == "en":
        lines = [
            f"🎄 [HiveMind Holiday Raid • {day_str}] Telemetry: {telemetry_label} | Trainers: {trainers_s} | Producers: {producers_s} | Accounts: {accounts_s}",
            f"🧩 OpenTrainer rounds: {rounds_s} | Models: {models_s} | Friendly guilds: {guilds_s} (0.00 EUR)",
            f"⚡ Home meter estimate: {w_str} W × {h_str} h = {wh_str} Wh ({kwh_str} kWh ≈ {eur_str} EUR at {tariff_str} EUR/kWh)",
            f"🎮 Comfort ({resp_s} replies, dev {dev_s} / gamer {gamer_s}): ease {ease_s} | zero Steam lag {lag_s}",
            "🛡️ Personal home PCs ONLY (Amadeus rule, 0.00 EUR / 0.00 USD, AGPL-3.0) • /raid • /quick",
        ]
    else:
        lines = [
            f"🎄 [Raid Noël / Holiday Raid • {day_str}] Telemetry: {telemetry_label} | Trainers: {trainers_s} | Producers: {producers_s} | Accounts: {accounts_s}",
            f"🧩 OpenTrainer rounds: {rounds_s} | Models: {models_s} | Guilds: {guilds_s} (0.00 EUR)",
            f"⚡ Home meter / Compteur : {w_str} W × {h_str} h = {wh_str} Wh ({kwh_str} kWh ≈ {eur_str} EUR @ {tariff_str} EUR/kWh)",
            f"🎮 Confort/Comfort ({resp_s}, dev {dev_s} / gamer {gamer_s}) : ease {ease_s} | zero Steam lag {lag_s}",
            "🛡️ PC perso uniquement / Personal home PCs ONLY (Amadeus rule, 0.00 EUR / 0.00 USD, AGPL-3.0) • /raid-noel • /raid",
        ]

    rendered = "\n".join(lines)
    if len(rendered.splitlines()) != BULLETIN_LINE_COUNT or len(rendered) > DISCORD_MAX_CHARS:
        raise RaidGuildVerifierError("Daily raid bulletin violated the 5-line / 2000-char invariant.")
    return _assert_safe_output(rendered)


def _check_bulletin_5_lines(text: str, *, expect_na: bool) -> list[str]:
    issues: list[str] = []
    lines = text.splitlines()
    if len(lines) != BULLETIN_LINE_COUNT:
        issues.append(f"Expected 5 bulletin lines, got {len(lines)}.")
    if len(text) > DISCORD_MAX_CHARS:
        issues.append(f"Bulletin exceeds Discord max length ({len(text)} > {DISCORD_MAX_CHARS}).")
    issues.extend(scan_output_safety(text))
    if "agpl-3.0" not in text.lower() or ("0.00 eur" not in text.lower() and "0,00 eur" not in text.lower()):
        issues.append("Bulletin missing 0.00 EUR / AGPL-3.0 compliance footer.")
    if expect_na:
        for line_idx in (0, 1, 3):
            if line_idx < len(lines) and NOT_AVAILABLE not in lines[line_idx]:
                issues.append(f"Absent-telemetry fallback line {line_idx + 1} missing 'n/a'.")
    return issues


def audit_daily_bulletin_contract(
    raid_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    gpu_watts: float = DEFAULT_GPU_WATTS,
    hours: float = DEFAULT_HOURS,
    kwh_tariff_eur: float = DEFAULT_KWH_TARIFF_EUR,
    date_override: str | None = None,
) -> dict[str, Any]:
    """Verify live 5-line FR/EN bulletins AND strict ``'n/a'`` fallback when telemetry is absent."""
    live_fr = render_raid_daily_bulletin(
        raid_status,
        feedback_summary,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        lang="fr",
        date_override=date_override,
    )
    live_en = render_raid_daily_bulletin(
        raid_status,
        feedback_summary,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        lang="en",
        date_override=date_override,
    )
    live_bi = render_raid_daily_bulletin(
        raid_status,
        feedback_summary,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        lang="bilingual",
        date_override=date_override,
    )

    fallback_na_fr = render_raid_daily_bulletin(
        None,
        None,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        lang="fr",
        date_override=date_override,
    )
    fallback_na_en = render_raid_daily_bulletin(
        {"swarm": {"telemetry_status": "UNAVAILABLE"}},
        None,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        lang="en",
        date_override=date_override,
    )

    issues: list[str] = []
    issues.extend(_check_bulletin_5_lines(live_fr, expect_na=False))
    issues.extend(_check_bulletin_5_lines(live_en, expect_na=False))
    issues.extend(_check_bulletin_5_lines(live_bi, expect_na=False))
    issues.extend(_check_bulletin_5_lines(fallback_na_fr, expect_na=True))
    issues.extend(_check_bulletin_5_lines(fallback_na_en, expect_na=True))

    bulletin_ok = len(issues) == 0
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": BULLETIN_AUDIT_OBJECT,
            "bulletin_ok": bulletin_ok,
            "bulletin_line_count": BULLETIN_LINE_COUNT,
            "fallback_na_verified": bulletin_ok,
            "live_fr": live_fr,
            "live_en": live_en,
            "live_bilingual": live_bi,
            "fallback_na_fr": fallback_na_fr,
            "fallback_na_en": fallback_na_en,
            "issues": issues,
            "exit_code": 0 if bulletin_ok else 2,
        }
    )


def _inspect_html_parity_route(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    html_text = str(body.get("html") or body.get("text") or "")
    raw_hdrs = body.get("headers")
    hdrs = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low

    nonce: str | None = hdr_nonces[0] if hdr_nonces else (attr_nonces[0] if attr_nonces else None)
    csp_ok = bool(
        attr_nonces
        and hdr_nonces
        and len(set(hdr_nonces)) == 1
        and "default-src 'none'" in csp_low
        and not csp_unsafe
        and hdr_nonces[0] in attr_nonces
        and nonce is not None
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    stripped_html = strip_masked_keys_and_nonces(html_text)
    has_unsafe_sink = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    has_completions_call = "completions" in stripped_html.lower()
    dom_clean = not (has_unsafe_sink or has_static_fig or has_completions_call)

    low_html = stripped_html.lower()
    has_personal = (
        "100% personal" in low_html
        or "100 % personnel" in low_html
        or (("100%" in low_html or "100 %" in low_html) and ("personal" in low_html or "personnel" in low_html))
    )
    has_eur = "0.00 eur" in low_html or "0,00 eur" in low_html
    has_usd = "0.00 usd" in low_html or "0,00 usd" in low_html
    needs_usd = path in ("/raid-noel", "/raid")
    compliance_banner_ok = bool(has_personal and "agpl-3.0" in low_html and has_eur and (has_usd or not needs_usd))

    checks = (
        (
            has_doctype,
            f"Contrat HTML {path} invalide : `<!DOCTYPE html>` absent."
            if is_fr
            else f"HTML contract failed on {path}: missing `<!DOCTYPE html>`.",
        ),
        (
            csp_ok,
            f"Contrat CSP {path} invalide : nonce `'default-src 'none''` par requête absent, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused `'default-src 'none''` nonce.",
        ),
        (
            dom_clean,
            f"Contrat sécurité DOM {path} violé : sink non sûr, chiffre statique ou appel completions détecté."
            if is_fr
            else f"DOM safety contract violated on {path}: unsafe sink, static figure, or completions call detected.",
        ),
        (
            compliance_banner_ok,
            f"Bannière conformité PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR-USD / AGPL-3.0 compliance banner missing on {path}.",
        ),
    )
    issues = [msg for passed, msg in checks if not passed]
    ok = len(issues) == 0
    return (
        ok,
        {
            "path": path,
            "ok": ok,
            "has_doctype": has_doctype,
            "csp_nonce_verified": csp_ok,
            "csp_nonce_sha256_prefix": hashlib.sha256(nonce.encode("utf-8")).hexdigest()[:16] if nonce else None,
            "dom_clean": dom_clean,
            "unsafe_dom_sink_found": has_unsafe_sink,
            "static_figure_found": has_static_fig,
            "completions_reference_found": has_completions_call,
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def _inspect_manifest_contract(
    body: Mapping[str, Any],
    *,
    expected_sha: str | None,
    lang_code: str,
) -> tuple[bool, str, bool | None, list[str], list[str]]:
    is_fr = lang_code == "fr"
    live_sha = sanitize_build_sha(body.get("build_sha"))
    issues: list[str] = []

    if (
        body.get("object") != MANIFEST_OBJECT
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("status") != "ok"
        or body.get("personal_pc_only") is not True
        or body.get("financial_promise") != FINANCIAL_PROMISE
    ):
        issues.append(
            "Contrat JSON `/v1/hivemind/onboarding/manifest` invalide ou incomplet."
            if is_fr
            else "Manifest `/v1/hivemind/onboarding/manifest` returned invalid or incomplete metadata."
        )

    raw_routes = body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in EXPECTED_MANIFEST_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(EXPECTED_MANIFEST_HOLIDAY_ROUTES)
    if missing_routes:
        joined_r = ", ".join(missing_routes)
        issues.append(
            f"Routes Noël absentes du manifeste : {joined_r}."
            if is_fr
            else f"Manifest `holiday_routes` missing expected routes: {joined_r}."
        )

    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = shas_match(live_sha, expected_sha)
        if not sha_match:
            issues.append(
                f"Dérive `build_sha` : live='{live_sha}' != attendu '{expected_sha}'."
                if is_fr
                else f"Build SHA drift: live='{live_sha}' != expected '{expected_sha}'."
            )

    return len(issues) == 0, live_sha, sha_match, missing_routes, issues


def _inspect_feedback_summary_exact(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    exact_5_keys = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS

    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    raw_ease = body.get("mean_ease_rating")
    ease_ok = bool(
        "mean_ease_rating" in body
        and total_ok
        and (
            (total_val == 0 and raw_ease is None)
            or (
                total_val > 0
                and not isinstance(raw_ease, bool)
                and isinstance(raw_ease, (int, float))
                and math.isfinite(float(raw_ease))
                and 1.0 <= float(raw_ease) <= 5.0
            )
        )
    )

    raw_lag = body.get("zero_game_lag_ratio")
    lag_ok = bool(
        "zero_game_lag_ratio" in body
        and total_ok
        and (
            (total_val == 0 and raw_lag is None)
            or (
                total_val > 0
                and not isinstance(raw_lag, bool)
                and isinstance(raw_lag, (int, float))
                and math.isfinite(float(raw_lag))
                and 0.0 <= float(raw_lag) <= 1.0
            )
        )
    )

    raw_prof = body.get("by_profile")
    prof_keys_ok = isinstance(raw_prof, Mapping) and set(raw_prof.keys()) == {"dev", "gamer"}
    d_raw = raw_prof.get("dev") if isinstance(raw_prof, Mapping) else None
    g_raw = raw_prof.get("gamer") if isinstance(raw_prof, Mapping) else None
    prof_ok = bool(
        prof_keys_ok
        and not isinstance(d_raw, bool)
        and isinstance(d_raw, int)
        and d_raw >= 0
        and not isinstance(g_raw, bool)
        and isinstance(g_raw, int)
        and g_raw >= 0
        and total_ok
        and (d_raw + g_raw == total_val)
    )

    obj_ok = body.get("object") == FEEDBACK_SUMMARY_OBJECT
    schema_ok = bool(exact_5_keys and obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma exact 5 clés `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` exact 5-key schema mismatch."
        ]
    )
    preserved = (
        {
            "object": FEEDBACK_SUMMARY_OBJECT,
            "total_responses": total_val,
            "mean_ease_rating": round(float(raw_ease), 2) if raw_ease is not None else None,
            "zero_game_lag_ratio": round(float(raw_lag), 4) if raw_lag is not None else None,
            "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)},
        }
        if schema_ok
        else {}
    )
    return schema_ok, preserved, issues


def verify_raid_noel_and_bulletin_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    kwh_tariff_eur: float = DEFAULT_KWH_TARIFF_EUR,
    gpu_watts: float = DEFAULT_GPU_WATTS,
    hours: float = DEFAULT_HOURS,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    expected_sha: str | None = None,
    date_override: str | None = None,
) -> dict[str, Any]:
    """Run zero-inference /raid-noel guild scoreboard, electricity Wh/EUR, CSP parity, and bulletin audit."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=True,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("raid_guild_and_bulletin_readiness", comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None
    _compute_wh_eur(gpu_watts, hours, kwh_tariff_eur)

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    html_bodies: dict[str, str] = {}

    raid_status_json: dict[str, Any] = {}
    raid_status_http_ok = False
    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in RAID_PARITY_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404 and group in ("holiday", "raid_status", "manifest", "feedback"):
                stale_404_routes.append(path)
            diag = (
                f"GET {path} -> HTTP {code} (attendu 200)."
                if is_fr
                else f"GET {path} -> HTTP {code} (expected 200)."
            )
            diagnostics.append(diag)
        elif kind == "html":
            html_text = str(body.get("html") or body.get("text") or "")
            html_bodies[path] = html_text
            ok, h_doc, h_issues = _inspect_html_parity_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/raid/status":
            raid_status_json = _unwrap_json_body(body)
            raid_status_http_ok = True
        elif path == "/v1/hivemind/onboarding/manifest":
            manifest_json = _unwrap_json_body(body)
            ok, live_build_sha, sha_match, missing_manifest_routes, m_issues = _inspect_manifest_contract(
                manifest_json, expected_sha=clean_expected_sha, lang_code=lang_code
            )
            manifest_ok = ok
            if not ok:
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            fb_json = _unwrap_json_body(body)
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_exact(fb_json, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
        route_rows.append({"path": path, "http_status": code, "ok": ok, "diagnostic": diag})

    raid_noel_html = html_bodies.get("/raid-noel", "")
    raid_en_html = html_bodies.get("/raid", "")

    if raid_status_http_ok:
        guild_audit = audit_raid_guild_scoreboard(
            raid_status_json,
            raid_noel_html=raid_noel_html,
            raid_en_html=raid_en_html,
            feedback_summary=feedback_summary_doc if feedback_summary_ok else None,
            lang=lang_code,
        )
        guild_scoreboard_ok = bool(guild_audit["scoreboard_ok"])
        feedback_crosscheck_ok = bool(guild_audit["feedback_crosscheck_ok"])
        if not guild_scoreboard_ok:
            diagnostics.extend(guild_audit["issues"])
            if "/v1/hivemind/raid/status" not in failed_routes:
                failed_routes.append("/v1/hivemind/raid/status")
                for row in route_rows:
                    if row["path"] == "/v1/hivemind/raid/status":
                        row["ok"] = False
                        row["diagnostic"] = guild_audit["issues"][0]
    else:
        guild_audit = None
        guild_scoreboard_ok = False
        feedback_crosscheck_ok = False

    electricity_audit = audit_electricity_transparency_math(
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        raid_status=raid_status_json if raid_status_http_ok else None,
        raid_html=raid_noel_html if raid_noel_html else None,
        lang=lang_code,
    )
    electricity_math_ok = bool(electricity_audit["math_ok"])
    if not electricity_math_ok:
        diagnostics.extend(electricity_audit["issues"])

    bulletin_audit = audit_daily_bulletin_contract(
        raid_status_json if raid_status_http_ok else None,
        feedback_summary_doc if feedback_summary_ok else None,
        gpu_watts=gpu_watts,
        hours=hours,
        kwh_tariff_eur=kwh_tariff_eur,
        date_override=date_override,
    )
    daily_bulletin_ok = bool(bulletin_audit["bulletin_ok"])
    if not daily_bulletin_ok:
        diagnostics.extend(bulletin_audit["issues"])

    html_all_ok = bool(
        len(html_contracts) == len(RAID_PARITY_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in RAID_PARITY_HTML_ROUTES)
        and len(seen_nonces) == len(RAID_PARITY_HTML_ROUTES)
    )
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    raid_status_ok = bool(raid_status_http_ok and guild_scoreboard_ok)
    parity_ok = bool(
        html_all_ok
        and raid_status_ok
        and electricity_math_ok
        and manifest_ok
        and feedback_summary_ok
        and feedback_crosscheck_ok
        and daily_bulletin_ok
        and (clean_expected_sha is None or sha_match is True)
    )
    redeploy_required = bool(stale_404_drift or not parity_ok)

    summary = (
        (
            f"Parité `/raid-noel`, guildes (0,00 EUR), calculateur Wh & bulletin 5 lignes VERTE sur {clean_gw} "
            f"(nonces_uniques={len(seen_nonces)})."
            if is_fr
            else f"`/raid-noel` guild scoreboard, Wh transparency & 5-line bulletin parity PASSED on {clean_gw} "
            f"(unique_nonces={len(seen_nonces)})."
        )
        if parity_ok
        else (
            f"Parité `/raid-noel` EN ÉCHEC sur {clean_gw} (redéploiement requis={redeploy_required})."
            if is_fr
            else f"`/raid-noel` guild & bulletin parity FAILED on {clean_gw} "
            f"(redeploy_required={redeploy_required})."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "raid_guild_and_bulletin_readiness",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": parity_ok,
            "parity_ok": parity_ok,
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "raid_status_ok": raid_status_ok,
            "guild_scoreboard_ok": guild_scoreboard_ok,
            "guild_scoreboard": guild_audit,
            "electricity_math_ok": electricity_math_ok,
            "electricity_audit": electricity_audit,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "missing_manifest_routes": missing_manifest_routes,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "feedback_crosscheck_ok": feedback_crosscheck_ok,
            "daily_bulletin_ok": daily_bulletin_ok,
            "daily_bulletin": bulletin_audit,
            "stale_404_drift": stale_404_drift,
            "stale_404_routes": stale_404_routes,
            "redeploy_required": redeploy_required,
            "http_statuses": http_statuses,
            "routes": route_rows,
            "failed_routes": failed_routes,
            "network_calls": network_calls,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if parity_ok else 2,
            "summary": summary,
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN CLI report with signed SHA-256 receipt."""
    is_fr = report.get("lang") == "fr"
    title = (
        "HIVEMIND /raid-noel — VÉRIFICATEUR GUILDES, ÉLECTRICITÉ Wh & BULLETIN (0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND /raid-noel — GUILD SCOREBOARD, Wh TRANSPARENCY & BULLETIN VERIFIER (0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway    : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Mode       : {report.get('mode')}",
        f"  Compliance : {report.get('compliance_ok')}",
        f"  Exit code  : {report.get('exit_code')}",
        f"  Summary    : {report.get('summary')}",
    ]
    elec = report.get("electricity_audit")
    if isinstance(elec, Mapping) and isinstance(elec.get("requested_estimate"), Mapping):
        req = elec["requested_estimate"]
        lines.append(
            f"  Meter Est  : {req.get('gpu_watts')} W x {req.get('hours')} h = "
            f"{req.get('energy_wh')} Wh ({req.get('estimated_eur')} EUR @ {req.get('kwh_tariff_eur')} EUR/kWh)"
        )
    bull = report.get("daily_bulletin")
    if isinstance(bull, Mapping):
        lines.append(f"  Bulletin   : {'ok' if bull.get('bulletin_ok') else 'fail'} (5 lines, fallback_na={bull.get('fallback_na_verified')})")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256    : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN Markdown receipt report with SHA-256 digest."""
    is_fr = report.get("lang") == "fr"
    heading = (
        "# Rapport Parité `/raid-noel`, Guildes & Bulletin 5 Lignes"
        if is_fr
        else "# `/raid-noel` Guild Scoreboard, Wh Estimator & Daily Bulletin Report"
    )
    lines = [
        heading,
        "",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: "
        f"`{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Nonces CSP uniques' if is_fr else 'Unique CSP Nonces'}**: `{report.get('unique_csp_nonces_count', 0)}`",
        f"- **{'Verdict' if is_fr else 'Exit Code'}**: `{report.get('exit_code')}`",
        f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}",
        f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`",
    ]
    elec = report.get("electricity_audit")
    if isinstance(elec, Mapping) and isinstance(elec.get("requested_estimate"), Mapping):
        req = elec["requested_estimate"]
        lines.extend(
            [
                "",
                "## Electricity Transparency" if not is_fr else "## Transparence Électrique (Wh & EUR)",
                f"- **Home Meter**: `{req.get('gpu_watts')} W x {req.get('hours')} h = {req.get('energy_wh')} Wh` "
                f"(`{req.get('energy_kwh')} kWh`, `{req.get('estimated_eur')} EUR` at `{req.get('kwh_tariff_eur')} EUR/kWh`)",
                f"- **HiveMind Cost**: `{FINANCIAL_PROMISE}` (`math_ok={elec.get('math_ok')}`)",
            ]
        )
    bull = report.get("daily_bulletin")
    if isinstance(bull, Mapping):
        chosen = bull.get("live_fr") if is_fr else bull.get("live_en")
        if isinstance(chosen, str) and chosen:
            lines.extend(["", "## 5-Line Daily Bulletin", "```text", chosen, "```"])
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the /raid-noel guild and bulletin verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference /raid-noel guild scoreboard, electricity Wh/EUR transparency, "
            "and 5-line FR/EN daily bulletin verifier (0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument("--expected-sha", default=None, help="Expected build_sha on /v1/hivemind/onboarding/manifest.")
    parser.add_argument("--gpu-watts", type=float, default=DEFAULT_GPU_WATTS, help="GPU draw in Watts (default: 60.0).")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="Session duration in hours (default: 2.0).")
    parser.add_argument(
        "--kwh-tariff-eur",
        type=float,
        default=DEFAULT_KWH_TARIFF_EUR,
        help="Home electricity tariff in EUR/kWh (default: 0.2516).",
    )
    parser.add_argument("--date", default=None, dest="date_override", help="Optional YYYY-MM-DD bulletin date label.")
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Run only the strictly local personal-PC compliance check.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit signed JSON receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit signed Markdown receipt.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """CLI entrypoint running local compliance first, then /raid-noel guild, Wh math, and bulletin checks."""
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        lang_code = normalize_lang(args.lang)
        if args.enforce_personal_pc:
            report = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code)
        else:
            comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code)
            if not comp_doc["compliance_ok"]:
                report = _compliance_blocked_receipt("raid_guild_cli", comp_doc, lang_code=lang_code)
            else:
                report = verify_raid_noel_and_bulletin_readiness(
                    args.gateway,
                    kwh_tariff_eur=args.kwh_tariff_eur,
                    gpu_watts=args.gpu_watts,
                    hours=args.hours,
                    http_fn=http_fn,
                    environ=environ,
                    cwd=cwd,
                    lang=lang_code,
                    expected_sha=args.expected_sha,
                    date_override=args.date_override,
                )
    except RaidGuildVerifierError as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "ready": False,
                    "parity_ok": False,
                    "error": err_msg,
                    "exit_code": 2,
                }
            )
            print(json.dumps(err_doc, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(f"Error: {err_msg}", file=eff_err)
        return 2

    if args.json:
        text_out = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        _assert_safe_output(text_out)
        print(text_out, file=eff_out)
    elif args.markdown:
        print(render_markdown_report(report), file=eff_out)
    else:
        print(render_cli_report(report), file=eff_out)
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
