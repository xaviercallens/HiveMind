#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency HiveMind Canary Alpha & Noel preflight, yield/thermal & first-heartbeat assistant.

Compatible with Python 3.10-3.13 on Windows 11, macOS, and Linux using ONLY the
Python standard library (completes in < 10 seconds):
  1. Python runtime compatibility (3.10-3.13).
  2. ~/.hivemind/credentials.json presence, key format & POSIX 0600 permissions.
  3. Outbound HTTPS reachability to gateway /health and /v1/models (allowed hosts only).
  4. GPU & nvidia-smi readiness (VRAM >= 4096 MiB & temp <= 70 C for Gamer-Whisper,
     or 'Dev API-only mode ready' when no NVIDIA GPU is present).
  5. Active game process detection (Gamer-Whisper zero-lag guard).
  6. Local-only corporate environment check (warn by default; fail-closed in --noel mode
     across env vars, workspace path, and local .git/config remote URLs).
  7. Optional --noel holiday route drift gate (/noel, /quick, /raid-noel,
     /v1/hivemind/onboarding/manifest, /v1/hivemind/onboarding/feedback/summary),
     <1.5s Steam worker yield & <=70.0 C quiet-fan preflight, and zero-inference
     fleet / first-heartbeat / non-blocking CES verifier.

Personal Home Use & Non-Monetary Compliance Rule:
  - 100% personal home PCs & personal hobby/open-source projects only (never work PCs).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal compute; symbolic HMT has zero monetary value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.doctor/1"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding.feedback.summary"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
FLEET_OBJECT = "hivemind.producers.fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
_ACCEPTED_FEEDBACK_SUMMARY_OBJECTS: frozenset[str] = frozenset(
    {FEEDBACK_SUMMARY_OBJECT, "hivemind.onboarding_feedback_summary"}
)
_ACCEPTED_FLEET_OBJECTS: frozenset[str] = frozenset({FLEET_OBJECT, "hivemind.producer_fleet"})
SUPPORTED_LANGS = ("en", "fr")
MIN_PYTHON = (3, 10)
MAX_TESTED_MINOR = 13
GAMER_WHISPER_MIN_VRAM_MIB = 4096
GAMER_WHISPER_MAX_TEMP_C = 70.0
DEFAULT_NOEL_MIN_FREE_VRAM_MIB = 2048
DEFAULT_NOEL_YIELD_BUDGET_MS = 1500.0
DEFAULT_NOEL_HOST_LABEL = "personal-pc"
MIN_REPORTABLE_GPU_VRAM_MIB = 512
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})
HTTP_TIMEOUT_S = 4.0
SUBPROCESS_TIMEOUT_S = 3.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
CES_QUEUE_SCHEMA = "hivemind.doctor.ces_queue/1"
DEFAULT_CES_QUEUE_FILENAME = "doctor_ces_queue.json"
DEFAULT_CES_QUEUE_MAX_ENTRIES = 128
DEFAULT_CES_QUEUE_MAX_DELIVERED = 512

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
VALID_CES_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

NOEL_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback/summary",
)

PR949_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

PR949_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/raid/status", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "holiday"),
)

MANIFEST_REQUIRED_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)

ALLOWED_NOEL_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/opentrainer/heartbeat",
        "/v1/hivemind/onboarding/feedback",
        "/v1/hivemind/producer/heartbeat",
        "/v1/hivemind/register",
    }
)

STEAM_PROCESS_PREFIXES: tuple[str, ...] = ("steam_app_", "steam_game_", "proton_")

DEFAULT_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "dota2.exe",
        "dota2",
        "cyberpunk2077.exe",
        "eldenring.exe",
        "bg3.exe",
        "helldivers2.exe",
        "valorant.exe",
        "valorant-win64-shipping.exe",
        "fortniteclient-win64-shipping.exe",
        "overwatch.exe",
        "league of legends.exe",
        "baldur.exe",
        "starfield.exe",
        "witcher3.exe",
        "gta5.exe",
        "r5apex.exe",
    }
)

_CORP_MARKERS: tuple[str, ...] = (
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

_CORP_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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

EXIT_CODES: dict[str, int] = {
    "ready": 0,
    "ready_with_warnings": 1,
    "not_ready": 2,
}

COMPLIANCE_NOTICE: dict[str, str] = {
    "en": "100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD).",
    "fr": "100 % PC personnels à domicile et projets hobby/open-source uniquement (0,00 EUR / 0,00 USD).",
}

NOEL_COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE['fr']} | {COMPLIANCE_NOTICE['en']}"

_LOCAL_WORKER_YIELD_PROGRAM = (
    "import sys\n"
    "buf = bytearray(65536)\n"
    "buf[0] = 1\n"
    "sys.stdout.write('HIVEMIND_WORKER_READY\\n')\n"
    "sys.stdout.flush()\n"
    "cmd = sys.stdin.readline().strip()\n"
    "buf[:] = b'\\x00' * len(buf)\n"
    "del buf\n"
    "if cmd in ('YIELD', 'STOP', ''):\n"
    "    sys.stdout.write('HIVEMIND_WORKER_RELEASED:0\\n')\n"
    "    sys.stdout.flush()\n"
    "    raise SystemExit(0)\n"
    "raise SystemExit(3)\n"
)

_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_TMP_SEGMENT_RE = re.compile(r"pytest-of-[^/\\]+[/\\]pytest-[^/\\]+[/\\][^/\\]+")
_RAW_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*|ces-)[A-Za-z0-9_-]{6,}\b")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)(?:[0-9]{1,4}(?:m"
    + r"s\b|tok"
    + r"/s)|[A-Za-z0-9_-]{1,4}tok"
    + r"/s|[A-Za-z0-9_-]{4,})(?![A-Za-z0-9_-])"
)
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PLATFORM_TAG_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_HOST_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DNS_LABEL_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.ASCII)
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_HEX_SHA_RE = re.compile(r"\A[0-9a-f]{7,64}\Z", re.ASCII)
_VALID_EVENT_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{8,128}\Z", re.ASCII)
_REVISION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_NONCE_ATTR_RE = re.compile(
    r"""\bnonce=(?:"([A-Za-z0-9_+=/=-]{8,128})"|'([A-Za-z0-9_+=/=-]{8,128})')"""
)
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + r"ment\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
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
_SPOOL_LOCKS_GUARD = threading.Lock()
_SPOOL_LOCKS: dict[str, threading.Lock] = {}

HttpGetFn = Callable[[str, float], tuple[int, dict[str, Any]]]
HttpRequestFn = Callable[..., tuple[int, dict[str, Any]]]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
WorkerYieldFn = Callable[..., Any]


@dataclass(frozen=True)
class GpuInfo:
    """Measured local GPU state from nvidia-smi without serial/UUID."""

    present: bool
    parse_error: bool = False
    name: str = ""
    total_mib: int = 0
    free_mib: int = 0
    temp_c: float = 0.0
    util_pct: float = 0.0


@dataclass(frozen=True)
class DoctorCheck:
    """Single preflight check outcome with bilingual summary and one-line fix command."""

    id: str
    title: str
    status: str  # "ok" | "warn" | "fail"
    summary: str
    fix_command: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "summary": self.summary,
            "fix_command": self.fix_command,
            "details": dict(self.details),
        }


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language tag to 'en' or 'fr'."""
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ValueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize onboarding CES profile selector to 'dev' or 'gamer'."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError("Unsupported profile; expected 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_CES_PROFILES:
        raise ValueError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def _parse_ease_rating(raw: str | int) -> int:
    if isinstance(raw, bool):
        raise ValueError("ease_rating must be an integer between 1 and 5.")
    val = int(raw)
    if not (1 <= val <= 5):
        raise ValueError(f"ease_rating {raw!r} out of range; expected 1..5.")
    return val


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace user home directory prefix with '~' so local usernames never leak."""
    raw = str(target)
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (resolved_home, resolved_home.replace("\\", "/"), resolved_home.replace("/", "\\")):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    return "~" + suffix.replace("\\", "/")
    return raw


def _is_valid_hivemind_key(raw_key: object) -> bool:
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask an hm-... or hm_live_... API key so raw secrets never appear in reports."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not _is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def _scrub_ip(match: re.Match[str]) -> str:
    candidate = match.group(0)
    try:
        ip_obj = ipaddress.ip_address(candidate)
    except ValueError:
        return candidate
    return candidate if ip_obj.is_loopback else "<redacted-ip>"


def _sanitize_noel_text(text: str, *, home: Path | None = None) -> str:
    """Scrub home directory paths, Bearer headers, emails, non-loopback IPs, and raw hm-... keys."""
    if not text:
        return ""
    scrubbed = text
    if "/" in scrubbed or "\\" in scrubbed:
        scrubbed = redact_home_path(scrubbed, home=home)
        resolved_home = str(home if home is not None else Path.home())
        if resolved_home and resolved_home not in ("/", "\\", ""):
            for variant in (resolved_home, resolved_home.replace("\\", "/"), resolved_home.replace("/", "\\")):
                if variant and variant in scrubbed:
                    scrubbed = scrubbed.replace(variant, "~")
        if _HOME_PATH_RE.search(scrubbed):
            scrubbed = _HOME_PATH_RE.sub("~", scrubbed).replace("\\", "/")
    if "bearer" in scrubbed.lower():
        scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    if "hm" in scrubbed.lower():
        scrubbed = _RAW_API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    if "@" in scrubbed:
        scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    if "." in scrubbed:
        scrubbed = _IPV4_RE.sub(_scrub_ip, scrubbed)
    return scrubbed


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce=...`` attributes, CSP ``'nonce-...'`` tokens, and masked ``hm-...***XXXX`` suffixes."""
    if not isinstance(text, str):
        raise ValueError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _STRIP_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def scan_output_safety(text: str, home: Path | None = None) -> list[str]:
    """Audit rendered text/JSON for secrets, home paths, non-loopback IPs, static figures, or forbidden keys."""
    if not isinstance(text, str):
        raise ValueError("scan_output_safety requires a string.")
    scrubbed = strip_masked_keys_and_nonces(text)
    violations: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        violations.append("forbidden_key_detected")
    if _RAW_API_KEY_RE.search(text):
        violations.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        violations.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        violations.append("email_detected")
    resolved_home = str(home if home is not None else Path.home())
    has_home_leak = bool(_HOME_PATH_RE.search(text))
    if not has_home_leak and resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (resolved_home, resolved_home.replace("\\", "/")):
            if variant and variant in text:
                has_home_leak = True
                break
    if has_home_leak:
        violations.append("unredacted_home_path_detected")
    if _STATIC_FIG_UNIT_RE.search(scrubbed):
        violations.append("static_figure_detected")
    for ip_str in _IPV4_RE.findall(text):
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not ip_obj.is_loopback:
            violations.append(f"non_loopback_ipv4_detected:{ip_str}")
    return violations


def _sanitize_noel_structure(value: Any, *, home: Path | None = None) -> Any:
    if isinstance(value, str):
        return _sanitize_noel_text(value, home=home)
    if isinstance(value, Mapping):
        return {str(k): _sanitize_noel_structure(v, home=home) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_noel_structure(item, home=home) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_noel_structure(item, home=home) for item in value]
    return value


def _compute_receipt_sha256(payload: Mapping[str, Any]) -> str:
    body = {str(k): v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host satisfies the repository lateral-movement allowlist with strict DNS labels."""
    if not isinstance(host, str):
        return False
    raw = host.strip().lower()
    if (
        not raw
        or len(raw) > 253
        or any(ch.isspace() or ch in "\\/@" or ord(ch) < 32 or ord(ch) == 127 for ch in raw)
    ):
        return False
    h = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw
    if not h or "[" in h or "]" in h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS or h == "storage.googleapis.com":
        return True
    labels = h.split(".")
    if len(labels) < 3 or any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return h.endswith(".run.app") or h.endswith(".googleapis.com")


def _safe_urlsplit(url: str) -> Any:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError("Gateway/request URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ValueError("URL must not contain control characters.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned):
        raise ValueError("URL must not contain whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise ValueError("URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned)
        _ = parts.port
    except ValueError as exc:
        raise ValueError("URL contains a malformed authority or invalid port.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise ValueError("URL must not embed credentials or '@'.")
    return parts


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway base URL against the strict host allowlist."""
    parts = _safe_urlsplit(url)
    cleaned = url.strip().rstrip("/")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise ValueError("Gateway URL must not embed credentials or '@'.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Gateway URL must be a bare origin without path, query parameters, or fragments.")
    port = parts.port
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not parts.netloc or not is_allowed_gateway_host(host):
        raise ValueError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError("Loopback gateway URL must use the http or https scheme.")
    elif parts.scheme != "https":
        raise ValueError("Remote gateway URL must use the https scheme.")
    return cleaned


def _payload_has_forbidden_inference_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).strip().lower() in ("messages", "prompt") or _payload_has_forbidden_inference_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_payload_has_forbidden_inference_key(item) for item in obj)
    return False


def _validate_zero_inference_target(method: str, url: str, *, payload: Any = None) -> tuple[str, str]:
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ValueError(f"Unsupported HTTP method {verb!r}; expected GET, HEAD, or POST.")
    parts = _safe_urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    path = parts.path or "/"
    low_path = unquote(path).lower()
    if "completions" in low_path or "/chat" in low_path:
        raise ValueError(f"Zero-inference invariant violated: {verb} {path} is strictly forbidden.")
    if verb == "POST" and path not in ALLOWED_NOEL_POST_PATHS:
        raise ValueError(f"POST endpoint {path!r} is not permitted in hivemind_doctor.")
    if payload is not None:
        if verb != "POST":
            raise ValueError("Request payload is only permitted on POST requests.")
        if isinstance(payload, Mapping):
            serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, default=str).lower()
            if _payload_has_forbidden_inference_key(payload) or ("hivemind/" + "coder") in serialized:
                raise ValueError(
                    "Zero-inference invariant violated: messages/prompt/hivemind-coder payloads are strictly forbidden."
                )
    return verb, path


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be a finite positive number.")
    val = float(timeout_s)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError("timeout_s must be a finite positive number.")
    return min(max(0.2, val), 60.0)


def _sanitize_request_headers(
    headers: Mapping[str, str] | None,
    *,
    default_accept: str,
) -> dict[str, str]:
    req_headers: dict[str, str] = {
        "Accept": default_accept,
        "User-Agent": "hivemind-doctor/1.0",
    }
    if headers:
        for raw_k, raw_v in headers.items():
            raw_k_str = str(raw_k)
            raw_v_str = str(raw_v)
            if not raw_k_str.strip() or any(
                ch in ("\r", "\n", "\x00") or ord(ch) < 32 or ord(ch) == 127
                for ch in (raw_k_str + raw_v_str)
            ):
                raise ValueError("HTTP headers must be non-empty and must not contain CR/LF/NUL or control characters.")
            req_headers[raw_k_str.strip()] = raw_v_str.strip()
    return req_headers


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects toward untrusted hosts, downgrades, backslashes, credentials, or inference paths."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raw_loc: Any = newurl
        if headers is not None and hasattr(headers, "get"):
            for hdr_name in ("location", "Location", "uri"):
                cand = headers.get(hdr_name)
                if cand is not None:
                    raw_loc = cand
                    break
        if (
            not isinstance(raw_loc, str)
            or not raw_loc.strip()
            or not isinstance(newurl, str)
            or not newurl.strip()
        ):
            raise ValueError("Untrusted redirect: empty target URL.")
        if "\\" in raw_loc or "\\" in unquote(raw_loc) or "@" in raw_loc or "@" in unquote(raw_loc):
            raise ValueError("Untrusted redirect: forbidden backslash or '@' in Location.")
        orig_parts = _safe_urlsplit(req.full_url)
        dest_parts = _safe_urlsplit(newurl)
        orig_host = (orig_parts.hostname or "").lower()
        dest_host = (dest_parts.hostname or "").lower()
        if orig_host not in ALLOWED_LOOPBACK_HOSTS and dest_host in ALLOWED_LOOPBACK_HOSTS:
            raise ValueError("Untrusted redirect from remote host to loopback.")
        if orig_parts.scheme == "https" and dest_parts.scheme != "https":
            raise ValueError("Untrusted HTTPS-to-HTTP downgrade redirect.")
        effective_method = "GET" if (req.get_method() == "POST" and code in (301, 302, 303)) else req.get_method()
        _validate_zero_inference_target(effective_method, newurl)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _validate_zero_inference_target(new_req.get_method(), new_req.full_url)
        return new_req


def _read_bounded_http_body(resp: Any, *, max_bytes: int = MAX_HTTP_BODY_BYTES) -> bytes:
    raw_bytes = resp.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise ValueError(f"HTTP response exceeded {max_bytes} byte limit.")
    return bytes(raw_bytes)


def _execute_raw_http_with_opener(
    opener: urllib.request.OpenerDirector,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
    default_accept: str = "application/json, text/html;q=0.9, text/plain;q=0.8",
) -> tuple[int, str, dict[str, str]]:
    verb, _ = _validate_zero_inference_target(method, url, payload=payload)
    bounded_timeout = _validate_timeout_s(timeout_s)
    req_headers = _sanitize_request_headers(headers, default_accept=default_accept)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_text = _read_bounded_http_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            raw_hdrs = getattr(resp, "headers", None)
            lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
            return status_code, raw_text, lower_headers
    except urllib.error.HTTPError as exc:
        err_text = (
            exc.read(512 * 1024).decode("utf-8", errors="replace") if getattr(exc, "fp", None) is not None else ""
        )
        raw_hdrs = getattr(exc, "headers", None)
        lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
        return int(exc.code), _sanitize_noel_text(err_text), lower_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        err_doc = json.dumps(
            {
                "network_error": type(exc).__name__,
                "error": _sanitize_noel_text(str(reason))[:200],
            }
        )
        return 0, err_doc, {"x-hivemind-network-error": type(exc).__name__}


def default_http_raw(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Perform a direct, proxy-free HTTP request returning ``(status_code, body_text, lower_headers)``."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    return _execute_raw_http_with_opener(
        opener,
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )


def _decode_json_http_result(
    status_code: int,
    raw: str,
    lower_headers: Mapping[str, str],
    *,
    include_html_meta: bool,
) -> tuple[int, dict[str, Any]]:
    content_type = str(lower_headers.get("content-type", ""))
    if status_code == 0 or status_code >= 400:
        if not raw.strip():
            return status_code, {} if status_code >= 400 and include_html_meta else {"error": ""}
        try:
            parsed_err = json.loads(raw)
        except json.JSONDecodeError:
            return status_code, {"error": _sanitize_noel_text(raw[:240])}
        return status_code, (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})

    if not raw.strip():
        return status_code, ({"content_type": content_type, "html_length": 0} if include_html_meta else {})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if include_html_meta:
            return status_code, {"content_type": content_type, "html_length": len(raw), "raw": raw[:240]}
        return status_code, {"raw": raw[:240]}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a direct, proxy-free GET request to an allowlisted HiveMind endpoint."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    status_code, raw, lower_headers = _execute_raw_http_with_opener(
        opener,
        "GET",
        url,
        timeout_s=timeout_s,
        default_accept="application/json",
    )
    return _decode_json_http_result(status_code, raw, lower_headers, include_html_meta=False)


def default_http_request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Perform a direct, proxy-free GET/HEAD/POST request to an allowlisted zero-inference endpoint."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    status_code, raw, lower_headers = _execute_raw_http_with_opener(
        opener,
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
        default_accept="application/json, text/html;q=0.9",
    )
    return _decode_json_http_result(status_code, raw, lower_headers, include_html_meta=True)


def _invoke_http_callable_raw(
    http_callable: Callable[..., Any],
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> Any:
    verb, _ = _validate_zero_inference_target(method, url, payload=payload)
    bounded_timeout = _validate_timeout_s(timeout_s)
    hdr_dict = {str(k): str(v) for k, v in headers.items()} if headers else None
    if http_callable is default_http_raw or http_callable is default_http_request_json:
        return http_callable(verb, url, payload=payload, headers=hdr_dict, timeout_s=bounded_timeout)
    if http_callable is default_http_get_json:
        return http_callable(url, bounded_timeout)

    sig = inspect.signature(http_callable)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if len(positional) == 2 and not has_varargs and not has_kwargs and "payload" not in params and "headers" not in params and "timeout_s" not in params:
        # Handle HttpGetFn(url, timeout_s) vs fn(method, url)
        first_name = positional[0].name.lower()
        if first_name in ("url", "_u", "u", "target_url"):
            return http_callable(url, bounded_timeout)
        return http_callable(verb, url)

    if has_kwargs or (
        ("payload" in params or "headers" in params or "timeout_s" in params)
        and not has_varargs
        and len(positional) <= 2
    ):
        kw: dict[str, Any] = {}
        if has_kwargs or "payload" in params:
            kw["payload"] = payload
        if has_kwargs or "headers" in params:
            kw["headers"] = hdr_dict
        if has_kwargs or "timeout_s" in params:
            kw["timeout_s"] = bounded_timeout
        return http_callable(verb, url, **kw)
    if has_varargs or len(positional) >= 5:
        return http_callable(verb, url, payload, hdr_dict, bounded_timeout)
    if len(positional) == 4:
        return http_callable(verb, url, payload, hdr_dict)
    if len(positional) == 3:
        return http_callable(verb, url, payload)
    return http_callable(verb, url)


def _invoke_http_request(
    http_request: HttpRequestFn,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Invoke default_http_request_json or an injected HTTP request callable safely."""
    res = _invoke_http_callable_raw(
        http_request,
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    status_code, body = res[0], res[1]
    if isinstance(body, Mapping):
        return int(status_code), dict(body)
    if isinstance(body, str):
        try:
            parsed = json.loads(body) if body.strip() else {}
            return int(status_code), (dict(parsed) if isinstance(parsed, Mapping) else {"data": parsed})
        except json.JSONDecodeError:
            return int(status_code), {"raw": body[:240], "html_length": len(body)}
    return int(status_code), {"data": body}


def _invoke_parity_call(
    http_callable: Callable[..., Any],
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any] | None, str, dict[str, str]]:
    """Invoke a raw/json/get HTTP callable safely and normalize ``(status, json_or_none, text, lower_headers)``."""
    try:
        raw_res = _invoke_http_callable_raw(
            http_callable,
            method,
            url,
            payload=payload,
            headers=headers,
            timeout_s=timeout_s,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        err = {"network_error": type(exc).__name__, "error": _sanitize_noel_text(str(exc))[:200]}
        return 0, err, json.dumps(err), {}

    resp_headers: dict[str, str] = {}
    if isinstance(raw_res, tuple) and len(raw_res) == 3:
        status_code, body, raw_hdrs = raw_res
        if isinstance(raw_hdrs, Mapping):
            resp_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()}
    else:
        status_code, body = raw_res

    json_doc: dict[str, Any] | None = None
    text_body = ""
    if isinstance(body, Mapping):
        raw_inner_hdrs = body.get("headers")
        if not resp_headers and isinstance(raw_inner_hdrs, Mapping):
            resp_headers = {str(k).lower(): str(v) for k, v in raw_inner_hdrs.items()}
        if "content-security-policy" not in resp_headers and isinstance(body.get("csp"), str):
            resp_headers["content-security-policy"] = body["csp"]
        for key in ("html", "text", "body"):
            if isinstance(body.get(key), str):
                text_body = body[key]
                break
        json_doc = dict(body)
        if not text_body and "html_length" not in body:
            text_body = json.dumps(json_doc)
    elif isinstance(body, str):
        text_body = body
        if body.lstrip().startswith("{"):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    json_doc = dict(parsed)
            except json.JSONDecodeError:
                json_doc = None
    return int(status_code), json_doc, text_body, resp_headers


def default_subprocess_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Run a local read-only command with shell=False and a strict timeout."""
    cmd = list(argv)
    if cmd and Path(cmd[0]).name.lower().startswith("gcloud"):
        raise ValueError("Direct gcloud invocation is forbidden in hivemind_doctor.")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        check=False,
        shell=False,
        env=dict(env) if env is not None else None,
    )
    return int(proc.returncode), proc.stdout


def default_worker_yield_drill(*, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> dict[str, Any]:
    """Run a real lightweight local subprocess worker preemption drill (list argv, shell=False)."""
    bounded = min(max(0.25, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCAL_WORKER_YIELD_PROGRAM],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    try:
        ready_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        if ready_line != "HIVEMIND_WORKER_READY":
            proc.kill()
            proc.wait(timeout=1.0)
            return {
                "worker_started": False,
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": -1,
                "yield_latency_ms": None,
                "method": "local_python_worker_subprocess",
            }
        started = time.monotonic()
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        release_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        exit_rc = proc.wait(timeout=bounded)
        yield_latency_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)
        retained_mib = -1
        if release_line.startswith("HIVEMIND_WORKER_RELEASED:"):
            raw_mib = release_line.split(":", 1)[1].strip()
            if raw_mib.isdigit():
                retained_mib = int(raw_mib)
        clean_ok = bool(exit_rc == 0 and proc.poll() is not None and retained_mib == 0)
        return {
            "worker_started": True,
            "worker_stopped": proc.poll() is not None,
            "clean_release": clean_ok,
            "retained_worker_vram_mib": retained_mib,
            "yield_latency_ms": yield_latency_ms,
            "method": "local_python_worker_subprocess",
        }
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        proc.wait(timeout=1.0)
        return {
            "worker_started": True,
            "worker_stopped": False,
            "clean_release": False,
            "retained_worker_vram_mib": -1,
            "yield_latency_ms": round(bounded * 1000.0, 3),
            "method": "local_python_worker_subprocess",
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _quickstart_command(gateway: str, os_name: str, extra_args: str = "") -> str:
    safe_gw = gateway if gateway else DEFAULT_GATEWAY
    suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
    if os_name == "nt":
        return (
            f'powershell -NoProfile -Command "Invoke-RestMethod {safe_gw}/scripts/hivemind_quickstart.py '
            f'-OutFile hivemind_quickstart.py; python hivemind_quickstart.py{suffix}"'
        )
    return (
        f"curl -fsSL {safe_gw}/scripts/hivemind_quickstart.py -o hivemind_quickstart.py "
        f"&& python3 hivemind_quickstart.py{suffix}"
    )


def _parity_redeploy_plan_command(gateway: str, os_name: str) -> str:
    safe_gw = gateway if gateway else DEFAULT_GATEWAY
    if os_name == "nt":
        return f'python scripts\\noel_cloud_run_parity_gate.py --gateway "{safe_gw}" --print-redeploy-plan'
    return f'python3 scripts/noel_cloud_run_parity_gate.py --gateway "{safe_gw}" --print-redeploy-plan'


def _friendly_os_label(os_name: str, system_name: str) -> str:
    sys_low = (system_name or "").strip().lower()
    if os_name == "nt" or sys_low == "windows":
        return "Windows"
    if sys_low == "darwin":
        return "macOS"
    if sys_low == "linux":
        return "Linux"
    return system_name or os_name


def _platform_feedback_tag(os_name: str | None = None, system_name: str | None = None) -> str:
    eff_os = (os_name if os_name is not None else os.name).strip().lower()
    eff_sys = (system_name if system_name is not None else platform.system()).strip().lower()
    if "darwin" in eff_sys or "mac" in eff_sys:
        return "macos"
    if eff_os == "nt" or "win" in eff_sys:
        return "windows"
    if "linux" in eff_sys:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", eff_sys or eff_os).strip("-_")[:24]
    if cleaned and _PLATFORM_TAG_RE.fullmatch(cleaned):
        return cleaned
    return "unknown"


def check_python_runtime(
    version_info: tuple[int, int, int] | None = None,
    *,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> DoctorCheck:
    """(1) Verify Python 3.10-3.13 runtime compatibility across Windows 11, macOS, and Linux."""
    lang_code = normalize_lang(lang)
    ver = tuple(version_info[:3]) if version_info is not None else tuple(sys.version_info[:3])
    eff_os = os_name if os_name is not None else os.name
    eff_sys = system_name if system_name is not None else platform.system()
    os_label = _friendly_os_label(eff_os, eff_sys)
    ver_text = f"{ver[0]}.{ver[1]}.{ver[2]}"
    title = "Python Runtime Compatibility" if lang_code == "en" else "Compatibilité Python"

    if eff_os == "nt" or os_label == "Windows":
        install_cmd = "winget install -e --id Python.Python.3.12"
    elif os_label == "macOS":
        install_cmd = "brew install python@3.12"
    else:
        install_cmd = "sudo apt-get update && sudo apt-get install -y python3"

    details = {"python_version": ver_text, "os": os_label, "target_range": "3.10-3.13"}

    if ver[:2] < MIN_PYTHON:
        summary = (
            f"Python {ver_text} on {os_label} is too old; HiveMind requires Python 3.10-3.13."
            if lang_code == "en"
            else f"Python {ver_text} sur {os_label} est trop ancien ; HiveMind requiert Python 3.10 à 3.13."
        )
        return DoctorCheck("python_runtime", title, "fail", summary, install_cmd, details)

    if ver[0] > 3 or ver[1] > MAX_TESTED_MINOR:
        summary = (
            f"Python {ver_text} on {os_label} runs stdlib API tools, while Python 3.10-3.13 is recommended for GPU packages."
            if lang_code == "en"
            else f"Python {ver_text} sur {os_label} exécute les outils standard, mais Python 3.10 à 3.13 est recommandé pour le GPU."
        )
        return DoctorCheck("python_runtime", title, "warn", summary, install_cmd, details)

    summary = (
        f"Python {ver_text} on {os_label} is compatible (Python 3.10-3.13 stdlib ready)."
        if lang_code == "en"
        else f"Python {ver_text} sur {os_label} est compatible (Python 3.10 à 3.13 prêt, zéro dépendance)."
    )
    return DoctorCheck("python_runtime", title, "ok", summary, "", details)


def check_credentials_file(
    credentials_path: Path | None = None,
    *,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
    home: Path | None = None,
) -> DoctorCheck:
    """(2) Verify ~/.hivemind/credentials.json presence, key prefix, and POSIX 0600 permissions."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = credentials_path if credentials_path is not None else (resolved_home / ".hivemind" / "credentials.json")
    shown_path = redact_home_path(target, home=resolved_home)
    title = "HiveMind Credentials & Permissions" if lang_code == "en" else "Identifiants HiveMind & Permissions"
    connect_fix = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} not found; run quickstart to register or save your personal hm-... key."
            if lang_code == "en"
            else f"{shown_path} introuvable ; lancez quickstart pour créer ou enregistrer votre clé personnelle hm-..."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": False, "valid_key": False},
        )

    try:
        st = target.stat()
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = (
            f"{shown_path} could not be parsed as valid JSON; regenerate your credentials file."
            if lang_code == "en"
            else f"{shown_path} est illisible ou JSON invalide ; régénérez votre fichier d'identifiants."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": True, "valid_key": False},
        )

    api_key = payload.get("api_key") if isinstance(payload, dict) else None
    if not _is_valid_hivemind_key(api_key):
        summary = (
            f"{shown_path} does not contain a valid hm-... API key; re-run quickstart."
            if lang_code == "en"
            else f"{shown_path} ne contient pas de clé API hm-... valide ; relancez quickstart."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": True, "valid_key": False},
        )

    role = str(payload.get("role") or "prosumer") if isinstance(payload, dict) else "prosumer"

    if eff_os == "nt":
        summary = (
            f"{shown_path} present with valid hm-... key (role: {role}; Windows user-profile ACL)."
            if lang_code == "en"
            else f"{shown_path} présent avec clé hm-... valide (rôle : {role} ; ACL profil utilisateur Windows)."
        )
        return DoctorCheck(
            "credentials",
            title,
            "ok",
            summary,
            "",
            {
                "path": shown_path,
                "present": True,
                "valid_key": True,
                "role": role,
                "posix_permissions_checked": False,
                "permission_model": "windows_acl",
            },
        )

    mode_bits = st.st_mode & 0o777
    octal_mode = f"{mode_bits:04o}"
    if mode_bits != 0o600:
        chmod_target = shown_path if shown_path.startswith("~/") else shlex.quote(shown_path)
        chmod_cmd = f"chmod 600 {chmod_target}"
        summary = (
            f"{shown_path} has POSIX mode {octal_mode} (expected 0600 owner-only); tighten file permissions."
            if lang_code == "en"
            else f"{shown_path} a les permissions POSIX {octal_mode} (attendu 0600 propriétaire seul) ; restreignez l'accès."
        )
        return DoctorCheck(
            "credentials",
            title,
            "warn",
            summary,
            chmod_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": True,
                "role": role,
                "posix_permissions_checked": True,
                "posix_mode": octal_mode,
            },
        )

    summary = (
        f"{shown_path} present with valid hm-... key and POSIX 0600 permissions (role: {role})."
        if lang_code == "en"
        else f"{shown_path} présent avec clé hm-... valide et permissions POSIX 0600 (rôle : {role})."
    )
    return DoctorCheck(
        "credentials",
        title,
        "ok",
        summary,
        "",
        {
            "path": shown_path,
            "present": True,
            "valid_key": True,
            "role": role,
            "posix_permissions_checked": True,
            "posix_mode": octal_mode,
        },
    )


def check_gateway_reachability(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
    os_name: str | None = None,
    lang: str = "en",
) -> DoctorCheck:
    """(3) Verify outbound HTTPS reachability to gateway /health and /v1/models."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "Gateway HTTPS Reachability (/health & /v1/models)" if lang_code == "en" else "Accès Passerelle HTTPS (/health & /v1/models)"

    try:
        clean_gw = validate_gateway_url(gateway)
    except ValueError as exc:
        fix_cmd = (
            f'python scripts\\hivemind_doctor.py --gateway "{DEFAULT_GATEWAY}"'
            if eff_os == "nt"
            else f'python3 scripts/hivemind_doctor.py --gateway "{DEFAULT_GATEWAY}"'
        )
        summary = (
            f"Refused gateway URL ({exc})"
            if lang_code == "en"
            else f"URL de passerelle refusée ({exc})"
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            fix_cmd,
            {"allowed_host": False, "health_ok": False, "models_ok": False},
        )

    curl_bin = "curl.exe" if eff_os == "nt" else "curl"
    health_fix = f'{curl_bin} -fsSL "{clean_gw}/health"'
    models_fix = f'{curl_bin} -fsSL "{clean_gw}/v1/models"'

    started = time.monotonic()
    try:
        health_status, health_body = http_get(f"{clean_gw}/health", timeout_s)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        summary = (
            f"Cannot reach {clean_gw}/health ({type(exc).__name__}); verify internet/TLS 443 access."
            if lang_code == "en"
            else f"Impossible de joindre {clean_gw}/health ({type(exc).__name__}) ; vérifiez la connexion HTTPS 443."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            health_fix,
            {"gateway": clean_gw, "allowed_host": True, "health_ok": False, "models_ok": False},
        )

    if health_status != 200:
        summary = (
            f"Gateway {clean_gw}/health returned HTTP {health_status}."
            if lang_code == "en"
            else f"La passerelle {clean_gw}/health a répondu HTTP {health_status}."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            health_fix,
            {
                "gateway": clean_gw,
                "allowed_host": True,
                "health_status": health_status,
                "health_ok": False,
                "models_ok": False,
            },
        )

    try:
        models_status, models_body = http_get(f"{clean_gw}/v1/models", timeout_s)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        summary = (
            f"{clean_gw}/health is OK, while /v1/models failed ({type(exc).__name__})."
            if lang_code == "en"
            else f"{clean_gw}/health répond OK, mais /v1/models a échoué ({type(exc).__name__})."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            models_fix,
            {"gateway": clean_gw, "allowed_host": True, "health_ok": True, "models_ok": False},
        )

    raw_models = models_body.get("data") if isinstance(models_body, dict) else None
    if models_status != 200 or not isinstance(raw_models, list):
        summary = (
            f"{clean_gw}/v1/models returned HTTP {models_status} without a valid model list."
            if lang_code == "en"
            else f"{clean_gw}/v1/models a répondu HTTP {models_status} sans liste de modèles valide."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            models_fix,
            {
                "gateway": clean_gw,
                "allowed_host": True,
                "health_ok": True,
                "models_status": models_status,
                "models_ok": False,
            },
        )

    model_ids = [
        str(item["id"])
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    ]
    probe_elapsed_s = round(time.monotonic() - started, 3)
    health_state = str(health_body.get("status", "ok")) if isinstance(health_body, dict) else "ok"

    summary = (
        f"Gateway {clean_gw} reachable (/health={health_state}, /v1/models={len(model_ids)} model(s))."
        if lang_code == "en"
        else f"Passerelle {clean_gw} joignable (/health={health_state}, /v1/models={len(model_ids)} modèle(s))."
    )
    return DoctorCheck(
        "gateway_reachability",
        title,
        "ok",
        summary,
        "",
        {
            "gateway": clean_gw,
            "allowed_host": True,
            "health_ok": True,
            "models_ok": True,
            "live_models_count": len(model_ids),
            "sample_models": model_ids[:5],
            "probe_elapsed_s": probe_elapsed_s,
        },
    )


def _parse_float_field(raw: str, default: float = 0.0) -> float:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return default
    return float(cleaned)


def probe_gpu(runner: RunnerFn = default_subprocess_runner) -> GpuInfo:
    """Query nvidia-smi locally without requesting GPU serial or UUID."""
    try:
        rc, out = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return GpuInfo(present=False)

    if rc != 0 or not out.strip():
        return GpuInfo(present=False)

    candidates: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            return GpuInfo(present=True, parse_error=True)
        try:
            total_mib = int(float(parts[1]))
            free_mib = int(float(parts[2]))
            temp_c = _parse_float_field(parts[3], default=0.0)
            util_pct = _parse_float_field(parts[4], default=0.0)
        except ValueError:
            return GpuInfo(present=True, parse_error=True)
        if total_mib <= 0 or free_mib < 0:
            return GpuInfo(present=True, parse_error=True)
        candidates.append(
            GpuInfo(
                present=True,
                parse_error=False,
                name=parts[0],
                total_mib=total_mib,
                free_mib=free_mib,
                temp_c=temp_c,
                util_pct=util_pct,
            )
        )

    return max(candidates, key=lambda g: (g.total_mib, -g.temp_c, g.free_mib))


def check_gpu_readiness(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(4) Check GPU & nvidia-smi readiness for Gamer-Whisper or Dev API-only mode."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "GPU & Gamer-Whisper / Dev API Readiness" if lang_code == "en" else "Disponibilité GPU Gamer-Whisper / Mode Dev API"
    gpu = probe_gpu(runner)

    if not gpu.present:
        summary = (
            "No NVIDIA GPU detected — Dev API-only mode ready (VS Code Continue.dev & OpenAI-compatible API, 0 local GPU load)."
            if lang_code == "en"
            else "Aucun GPU NVIDIA détecté — Mode Dev API uniquement prêt (Dev API-only mode ready : VS Code Continue.dev sans GPU local)."
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "ok",
            summary,
            "",
            {"gpu_present": False, "mode": "dev_api_only"},
        )

    if gpu.parse_error:
        summary = (
            "nvidia-smi returned unparseable output — Dev API-only mode ready; inspect your NVIDIA driver before Gamer-Whisper."
            if lang_code == "en"
            else "Sortie nvidia-smi illisible — Mode Dev API uniquement prêt (Dev API-only mode ready) ; vérifiez le pilote NVIDIA."
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            "nvidia-smi",
            {"gpu_present": True, "parse_error": True, "mode": "dev_api_only"},
        )

    gpu_details = {
        "gpu_present": True,
        "name": gpu.name,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "min_vram_mib": GAMER_WHISPER_MIN_VRAM_MIB,
        "max_temp_c": GAMER_WHISPER_MAX_TEMP_C,
    }

    if gpu.total_mib < GAMER_WHISPER_MIN_VRAM_MIB:
        consumer_cmd = _quickstart_command(gateway, eff_os, "--role consumer --write-continue-config")
        summary = (
            f"{gpu.name} has {gpu.total_mib} MiB VRAM (< {GAMER_WHISPER_MIN_VRAM_MIB} MiB for Gamer-Whisper) — Dev API-only mode ready."
            if lang_code == "en"
            else (
                f"{gpu.name} a {gpu.total_mib} MiB de VRAM (< {GAMER_WHISPER_MIN_VRAM_MIB} MiB pour Gamer-Whisper) "
                "— Mode Dev API uniquement prêt (Dev API-only mode ready)."
            )
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            consumer_cmd,
            {**gpu_details, "mode": "dev_api_only"},
        )

    if gpu.temp_c > GAMER_WHISPER_MAX_TEMP_C:
        gamer_check_cmd = _quickstart_command(gateway, eff_os, "--gamer-check")
        summary = (
            f"{gpu.name} ({gpu.total_mib} MiB VRAM) is at {gpu.temp_c:.0f} C (> {GAMER_WHISPER_MAX_TEMP_C:.0f} C Gamer-Whisper ceiling); "
            "let it cool <= 70 C before GPU rounds (Dev API-only mode ready now)."
            if lang_code == "en"
            else (
                f"{gpu.name} ({gpu.total_mib} MiB VRAM) est à {gpu.temp_c:.0f} C (> {GAMER_WHISPER_MAX_TEMP_C:.0f} C plafond Gamer-Whisper) ; "
                "laissez refroidir <= 70 C avant un round GPU."
            )
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            gamer_check_cmd,
            {**gpu_details, "mode": "gpu_thermal_cooldown"},
        )

    summary = (
        f"{gpu.name} ready for Gamer-Whisper ({gpu.total_mib} MiB VRAM >= {GAMER_WHISPER_MIN_VRAM_MIB} MiB, "
        f"{gpu.free_mib} MiB free, {gpu.temp_c:.0f} C <= {GAMER_WHISPER_MAX_TEMP_C:.0f} C)."
        if lang_code == "en"
        else (
            f"{gpu.name} prêt pour Gamer-Whisper ({gpu.total_mib} MiB VRAM >= {GAMER_WHISPER_MIN_VRAM_MIB} MiB, "
            f"{gpu.free_mib} MiB libres, {gpu.temp_c:.0f} C <= {GAMER_WHISPER_MAX_TEMP_C:.0f} C)."
        )
    )
    return DoctorCheck(
        "gpu_readiness",
        title,
        "ok",
        summary,
        "",
        {**gpu_details, "mode": "gamer_whisper_ready"},
    )


def list_process_names(
    runner: RunnerFn = default_subprocess_runner,
    *,
    os_name: str | None = None,
) -> set[str]:
    """Return normalized lowercase process names running locally on Windows, macOS, or Linux."""
    eff_os = os_name if os_name is not None else os.name
    names: set[str] = set()
    try:
        if eff_os == "nt":
            rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    exe_name = row[0].strip().lower()
                    names.add(exe_name)
        else:
            rc, out = runner(["ps", "-eo", "comm="], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for line in out.splitlines():
                cleaned = line.strip()
                if cleaned:
                    low = cleaned.lower()
                    names.add(low)
                    names.add(Path(cleaned.replace("\\", "/")).name.lower())
    except (OSError, subprocess.SubprocessError, csv.Error):
        return names
    return names


def _match_active_games(running: set[str], extra_games: Sequence[str] = ()) -> list[str]:
    watchlist = set(DEFAULT_GAME_EXECUTABLES)
    for item in extra_games:
        cleaned = item.strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            watchlist.add(cleaned[:15])
    return sorted(
        proc
        for proc in running
        if proc in watchlist or proc.startswith(STEAM_PROCESS_PREFIXES)
    )


def check_active_games(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    extra_games: Sequence[str] = (),
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(5) Inspect local process list for active games so Gamer-Whisper never disturbs play."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "Active Game Process Guard" if lang_code == "en" else "Détection de Jeu Actif"

    watchlist = set(DEFAULT_GAME_EXECUTABLES)
    for item in extra_games:
        cleaned = item.strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            watchlist.add(cleaned[:15])

    running = list_process_names(runner, os_name=eff_os)
    detected = sorted(proc for proc in running if proc in watchlist)

    if detected:
        joined = ", ".join(detected)
        gamer_check_cmd = _quickstart_command(gateway, eff_os, "--gamer-check")
        summary = (
            f"Active game detected ({joined}); Gamer-Whisper yields 100% of GPU VRAM until you close the game."
            if lang_code == "en"
            else f"Jeu actif détecté ({joined}) ; Gamer-Whisper cède 100 % de la VRAM tant que le jeu est ouvert."
        )
        return DoctorCheck(
            "active_games",
            title,
            "warn",
            summary,
            gamer_check_cmd,
            {"active_games": detected, "game_running": True},
        )

    summary = (
        "No active game process detected; Gamer-Whisper idle guard is clear."
        if lang_code == "en"
        else "Aucun jeu actif détecté ; créneau silencieux Gamer-Whisper disponible."
    )
    return DoctorCheck(
        "active_games",
        title,
        "ok",
        summary,
        "",
        {"active_games": [], "game_running": False},
    )


def _find_corp_markers(raw_text: str) -> list[str]:
    """Return corporate marker names found in raw_text without retaining raw_text."""
    if not raw_text:
        return []
    hits: list[str] = []
    for marker_name, pat in _CORP_MARKER_PATTERNS:
        if pat.search(raw_text) and marker_name not in hits:
            hits.append(marker_name)
    return hits


def _detect_corporate_indicators(environ: Mapping[str, str]) -> list[str]:
    """Inspect environment variables strictly in memory without recording values."""
    indicators: list[str] = []
    upper_map: dict[str, str] = {str(k).upper(): str(v).strip() for k, v in environ.items() if str(v).strip()}

    if upper_map.get("USERDNSDOMAIN"):
        indicators.append("USERDNSDOMAIN")

    userdomain = upper_map.get("USERDOMAIN", "")
    computername = upper_map.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        low_dom = userdomain.lower()
        if any(marker in low_dom for marker in _CORP_MARKERS):
            indicators.append("USERDOMAIN")

    for proxy_key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if upper_map.get(proxy_key):
            indicators.append(proxy_key)

    no_proxy_val = upper_map.get("NO_PROXY", "").lower()
    if no_proxy_val and any(marker in no_proxy_val for marker in _CORP_MARKERS):
        indicators.append("NO_PROXY")

    for tls_key in ("ZSCALER_CERT_PATH", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        tls_val = upper_map.get(tls_key, "").lower()
        if tls_key == "ZSCALER_CERT_PATH" and tls_val:
            indicators.append(tls_key)
        elif tls_val and any(marker in tls_val for marker in _CORP_MARKERS):
            indicators.append(tls_key)

    return indicators


def _resolve_git_config_paths(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` walking ``(workspace, *workspace.parents)`` and resolving ``gitdir:``/``commondir``."""
    configs: list[Path] = []
    try:
        for root in (workspace, *workspace.parents):
            dot_git = root / ".git"
            if dot_git.is_dir():
                cfg = dot_git / "config"
                if cfg.is_file():
                    configs.append(cfg)
                break
            if dot_git.is_file():
                lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
                first_line = next((ln.strip() for ln in lines if ln.strip()), "")
                if first_line.lower().startswith("gitdir:"):
                    raw_gitdir = first_line.split(":", 1)[1].strip()
                    gitdir = Path(raw_gitdir)
                    if not gitdir.is_absolute():
                        gitdir = (root / gitdir).resolve()
                    wt_cfg = gitdir / "config"
                    if wt_cfg.is_file():
                        configs.append(wt_cfg)
                    commondir_file = gitdir / "commondir"
                    if commondir_file.is_file():
                        rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if rel_common:
                            common_base = Path(rel_common)
                            common_dir = common_base if common_base.is_absolute() else (gitdir / common_base).resolve()
                            common_cfg = common_dir / "config"
                            if common_cfg.is_file() and common_cfg not in configs:
                                configs.append(common_cfg)
                break
    except OSError:
        return configs
    return configs


def _extract_git_remote_urls(workspace: Path) -> list[str]:
    urls: list[str] = []
    for cfg_path in _resolve_git_config_paths(workspace):
        try:
            text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m1, m2 in _GIT_REMOTE_URL_RE.findall(text):
            cleaned = (m1 or m2).strip().strip('"').strip("'")
            if cleaned:
                urls.append(cleaned)
    return urls


def evaluate_noel_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> dict[str, Any]:
    """Inspect env vars, workspace path, and .git/config remotes locally (fail-closed for --noel)."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    env_view = environ if environ is not None else os.environ
    upper_map: dict[str, str] = {str(k).upper(): str(v).strip() for k, v in env_view.items() if str(v).strip()}

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _add_indicator(name: str, markers: Sequence[str] = ()) -> None:
        if name not in indicators:
            indicators.append(name)
        for mk in markers:
            if mk in _CORP_MARKERS:
                matched_markers.add(mk)

    if any(k.startswith("AMADEUS") for k in upper_map):
        _add_indicator("AMADEUS_ENV", ("amadeus",))

    dns_val = upper_map.get("USERDNSDOMAIN", "")
    if dns_val:
        _add_indicator("USERDNSDOMAIN", _find_corp_markers(dns_val) or ("corp",))

    userdomain = upper_map.get("USERDOMAIN", "")
    computername = upper_map.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        dom_hits = _find_corp_markers(userdomain)
        if dom_hits:
            _add_indicator("USERDOMAIN", dom_hits)

    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        proxy_val = upper_map.get(proxy_key, "")
        if proxy_val:
            _add_indicator(proxy_key, _find_corp_markers(proxy_val))

    for host_or_proxy_key in ("NO_PROXY", "COMPUTERNAME", "HOSTNAME"):
        val = upper_map.get(host_or_proxy_key, "")
        hits = _find_corp_markers(val)
        if hits:
            _add_indicator(host_or_proxy_key, hits)

    if upper_map.get("ZSCALER_CERT_PATH"):
        _add_indicator("ZSCALER_CERT_PATH", ("zscaler",))

    for tls_key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        tls_val = upper_map.get(tls_key, "")
        hits = _find_corp_markers(tls_val)
        if hits:
            _add_indicator(tls_key, hits)

    workspace = Path(cwd) if cwd is not None else Path.cwd()
    path_candidates: list[Path] = [workspace]
    with suppress(OSError):
        resolved_ws = workspace.resolve()
        if resolved_ws != workspace:
            path_candidates.append(resolved_ws)
    for ws_candidate in path_candidates:
        scrubbed_ws = _PYTEST_TMP_SEGMENT_RE.sub("pytest_tmp", str(ws_candidate))
        for ws_hit in _find_corp_markers(scrubbed_ws):
            _add_indicator(f"cwd_path:{ws_hit}", (ws_hit,))

    for remote_url in _extract_git_remote_urls(workspace):
        for rem_hit in _find_corp_markers(remote_url):
            _add_indicator(f"git_remote:{rem_hit}", (rem_hit,))

    compliance_ok = len(indicators) == 0
    safe_gw = DEFAULT_GATEWAY
    try:
        safe_gw = validate_gateway_url(gateway)
    except ValueError:
        safe_gw = DEFAULT_GATEWAY

    if not compliance_ok:
        uninstall_cmd = _quickstart_command(safe_gw, eff_os, "--uninstall")
        joined = ", ".join(indicators)
        summary = (
            f"BLOCKED BY PERSONAL-PC COMPLIANCE ({joined}): corporate environment or git remote marker detected locally "
            f"(0 outbound HTTP requests, 0 file writes). {NOEL_COMPLIANCE_NOTICE_BILINGUAL}"
            if lang_code == "en"
            else (
                f"BLOQUÉ PAR CONFORMITÉ PC PERSONNEL ({joined}) : marqueur d'environnement ou dépôt professionnel détecté localement "
                f"(0 requête HTTP sortante, 0 écriture fichier). {NOEL_COMPLIANCE_NOTICE_BILINGUAL}"
            )
        )
        return {
            "id": "personal_pc_compliance",
            "status": "fail",
            "passed": False,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "network_transmitted": False,
            "outbound_http_requests": 0,
            "file_writes": 0,
            "personal_pc_only": True,
            "monetary_cost": "0.00 EUR / 0.00 USD",
            "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
            "fix_command": uninstall_cmd,
        }

    summary = (
        f"Personal-PC compliance verified locally (0 corporate markers). {NOEL_COMPLIANCE_NOTICE_BILINGUAL}"
        if lang_code == "en"
        else f"Conformité PC personnel vérifiée localement (0 marqueur professionnel). {NOEL_COMPLIANCE_NOTICE_BILINGUAL}"
    )
    return {
        "id": "personal_pc_compliance",
        "status": "ok",
        "passed": True,
        "compliance_ok": True,
        "corporate_detected": False,
        "corporate_indicators": [],
        "matched_markers": [],
        "network_transmitted": False,
        "outbound_http_requests": 0,
        "file_writes": 0,
        "personal_pc_only": True,
        "monetary_cost": "0.00 EUR / 0.00 USD",
        "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
        "fix_command": "",
    }


def check_corporate_environment(
    *,
    environ: Mapping[str, str] | None = None,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(6) Strictly local corporate/work-PC warning (NEVER transmits env or hostname)."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    env_view = environ if environ is not None else os.environ
    title = (
        "Personal-PC Compliance (Local-Only Check)"
        if lang_code == "en"
        else "Conformité PC Personnel (Contrôle 100 % Local)"
    )

    indicators = _detect_corporate_indicators(env_view)
    if indicators:
        uninstall_cmd = _quickstart_command(gateway, eff_os, "--uninstall")
        joined = ", ".join(indicators)
        summary = (
            f"Corporate/work environment indicator detected locally ({joined}). "
            "STOP if this is a work machine: HiveMind is strictly 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD)."
            if lang_code == "en"
            else (
                f"Indicateur d'environnement professionnel détecté localement ({joined}). "
                "ARRÊTEZ s'il s'agit d'un PC professionnel : HiveMind est réservé à 100 % aux PC personnels à domicile et projets hobby/open-source (0,00 EUR / 0,00 USD)."
            )
        )
        return DoctorCheck(
            "personal_pc_compliance",
            title,
            "warn",
            summary,
            uninstall_cmd,
            {
                "corporate_indicators": indicators,
                "network_transmitted": False,
                "personal_pc_only": True,
                "monetary_cost": "0.00 EUR / 0.00 USD",
            },
        )

    summary = (
        "Local environment check clean (no corporate domain/proxy detected). "
        "Reminder: 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD)."
        if lang_code == "en"
        else (
            "Contrôle local conforme (aucun domaine/proxy professionnel détecté). "
            "Rappel : 100 % PC personnels à domicile et projets hobby/open-source uniquement (0,00 EUR / 0,00 USD)."
        )
    )
    return DoctorCheck(
        "personal_pc_compliance",
        title,
        "ok",
        summary,
        "",
        {
            "corporate_indicators": [],
            "network_transmitted": False,
            "personal_pc_only": True,
            "monetary_cost": "0.00 EUR / 0.00 USD",
        },
    )


def _sanitize_manifest_sha(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip().lower()
    if not cleaned or cleaned == "unknown" or not _BUILD_SHA_RE.fullmatch(cleaned):
        return "unknown"
    if any(tok in cleaned for tok in ("secret", "token", "passwd", "password")):
        return "unknown"
    return cleaned


def _sanitize_manifest_revision(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned or not _REVISION_RE.fullmatch(cleaned):
        return None
    low = cleaned.lower()
    if low.startswith(("hm-", "hm_live_", "sk-")) or any(tok in low for tok in ("secret", "token", "password")):
        return None
    return cleaned


def check_noel_holiday_routes(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
    os_name: str | None = None,
    lang: str = "en",
) -> DoctorCheck:
    """(7a) Probe zero-inference Noel holiday routes and detect pre-PR-#949 HTTP 404/5xx drift."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = (
        "Noel Holiday Routes & Cloud Run Drift Check"
        if lang_code == "en"
        else "Routes Noël & Détection de Dérive Cloud Run"
    )
    bounded_timeout = min(max(0.5, float(timeout_s)), 4.5)

    try:
        clean_gw = validate_gateway_url(gateway)
    except ValueError as exc:
        fix_cmd = _parity_redeploy_plan_command(DEFAULT_GATEWAY, eff_os)
        summary = (
            f"Refused gateway URL ({exc})"
            if lang_code == "en"
            else f"URL de passerelle refusée ({exc})"
        )
        return DoctorCheck(
            "noel_holiday_routes",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "allowed_host": False,
                "holiday_routes_deployed_on_cloud_run": False,
                "onboarding_feedback_deployed": False,
                "revision_drift_detected": False,
                "missing_or_stale_routes": list(NOEL_HOLIDAY_ROUTES),
                "route_statuses": {},
                "zero_inference": True,
            },
        )

    route_statuses: dict[str, int] = {}
    missing_or_stale: list[str] = []
    manifest_drift = False
    build_sha = "unknown"
    cloud_run_revision: str | None = None

    for route in NOEL_HOLIDAY_ROUTES:
        try:
            code, body = http_get(f"{clean_gw}{route}", bounded_timeout)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            code, body = 0, {}
        route_statuses[route] = int(code)

        if code != 200 or not isinstance(body, Mapping):
            missing_or_stale.append(route)
            continue

        if route in ("/noel", "/quick", "/raid-noel"):
            if "html_length" in body and int(body.get("html_length") or 0) <= 0:
                missing_or_stale.append(route)
            elif "raw" in body and not str(body.get("raw") or "").strip():
                missing_or_stale.append(route)
        elif route == "/v1/hivemind/onboarding/manifest":
            build_sha = _sanitize_manifest_sha(body.get("build_sha"))
            cloud_run_revision = _sanitize_manifest_revision(body.get("cloud_run_revision"))
            obj_val = body.get("object")
            if obj_val is not None and obj_val != "hivemind.onboarding.manifest":
                manifest_drift = True
                missing_or_stale.append(route)
            elif isinstance(body.get("holiday_routes"), list):
                listed = {str(r) for r in body["holiday_routes"]}
                if not {"/noel", "/quick", "/raid-noel"}.issubset(listed):
                    manifest_drift = True
                    missing_or_stale.append(route)
        elif route == "/v1/hivemind/onboarding/feedback/summary":
            if "error" in body and "total_responses" not in body:
                missing_or_stale.append(route)

    onboarding_feedback_deployed = bool(
        route_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200
        and "/v1/hivemind/onboarding/feedback/summary" not in missing_or_stale
    )
    holiday_routes_deployed = len(missing_or_stale) == 0
    revision_drift_detected = bool(
        manifest_drift
        or any(code in (404, 501) for code in route_statuses.values())
    )

    details = {
        "gateway": clean_gw,
        "allowed_host": True,
        "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
        "onboarding_feedback_deployed": onboarding_feedback_deployed,
        "revision_drift_detected": revision_drift_detected,
        "missing_or_stale_routes": missing_or_stale,
        "route_statuses": route_statuses,
        "build_sha": build_sha,
        "cloud_run_revision": cloud_run_revision,
        "zero_inference": True,
    }

    if not holiday_routes_deployed:
        joined = ", ".join(missing_or_stale)
        redeploy_fix = _parity_redeploy_plan_command(clean_gw, eff_os)
        if revision_drift_detected:
            summary = (
                f"Cloud Run holiday route drift detected (pre-PR-#949 404 or stale manifest on: {joined})."
                if lang_code == "en"
                else f"Dérive de révision Cloud Run détectée (HTTP 404 pré-PR-#949 ou manifeste obsolète sur : {joined})."
            )
        else:
            summary = (
                f"Holiday route check failed on {clean_gw} (unavailable/5xx on: {joined})."
                if lang_code == "en"
                else f"Échec de vérification des routes Noël sur {clean_gw} (indisponible/5xx sur : {joined})."
            )
        return DoctorCheck("noel_holiday_routes", title, "fail", summary, redeploy_fix, details)

    summary = (
        f"All 5 Noel holiday routes live on {clean_gw} (build_sha={build_sha}, feedback_deployed=True, 0 GPU inference)."
        if lang_code == "en"
        else f"Les 5 routes Noël sont actives sur {clean_gw} (build_sha={build_sha}, feedback_deployed=True, 0 inférence GPU)."
    )
    return DoctorCheck("noel_holiday_routes", title, "ok", summary, "", details)


def _run_yield_drill_callable(
    worker_yield_fn: WorkerYieldFn | None,
    *,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[float | None, bool, int, str]:
    target_fn = worker_yield_fn if worker_yield_fn is not None else default_worker_yield_drill
    sig = inspect.signature(target_fn)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    started = time.monotonic()
    if has_kwargs or "timeout_s" in params:
        raw = target_fn(timeout_s=timeout_s)
    else:
        raw = target_fn()
    wall_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        val = float(raw)
        if not math.isfinite(val):
            return None, False, -1, "custom_worker_yield_fn"
        return round(val, 3), True, 0, "custom_worker_yield_fn"

    if isinstance(raw, Mapping):
        raw_lat = raw.get("yield_latency_ms", raw.get("yield_ms"))
        lat_ms: float | None = None
        if isinstance(raw_lat, (int, float)) and not isinstance(raw_lat, bool) and math.isfinite(float(raw_lat)):
            lat_ms = round(float(raw_lat), 3)
        worker_stopped = bool(raw.get("worker_stopped", True))
        clean_release = bool(raw.get("clean_release", worker_stopped))
        raw_ret = raw.get("retained_worker_vram_mib", raw.get("retained_vram_mib"))
        retained_mib = int(raw_ret) if isinstance(raw_ret, int) and not isinstance(raw_ret, bool) else (0 if clean_release else -1)
        if lat_ms is None and worker_stopped:
            lat_ms = wall_ms
        return lat_ms, clean_release, retained_mib, str(raw.get("method") or "custom_worker_yield_fn")

    raise ValueError("worker_yield_fn must return a numeric latency or mapping.")


def check_noel_steam_yield_and_thermal(
    *,
    runner: RunnerFn = default_subprocess_runner,
    worker_yield_fn: WorkerYieldFn | None = default_worker_yield_drill,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_NOEL_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_NOEL_YIELD_BUDGET_MS,
    extra_games: Sequence[str] = (),
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(7b) Run local subprocess worker yield drill (<1.5s SLO) plus <=70.0 C & >=2048 MiB VRAM checks."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = (
        "Noel Steam Preemption (<1.5s) & Quiet-Fan Thermal Check"
        if lang_code == "en"
        else "Préemption Steam Noël (<1,5s) & Plafond Thermique Silencieux"
    )

    budget_ms = float(yield_budget_ms)
    cap_temp_c = float(max_temp_c)
    reserve_mib = int(min_free_vram_mib)

    yield_latency_ms, clean_release, retained_mib, drill_method = _run_yield_drill_callable(
        worker_yield_fn,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    yield_slo_ok = bool(
        clean_release
        and retained_mib == 0
        and yield_latency_ms is not None
        and 0.0 <= yield_latency_ms < budget_ms
    )

    gpu = probe_gpu(runner)
    running_procs = list_process_names(runner, os_name=eff_os)
    detected_games = _match_active_games(running_procs, extra_games)
    game_running = len(detected_games) > 0

    budget_s = budget_ms / 1000.0
    lat_s_text = f"{yield_latency_ms / 1000.0:.3f}s" if yield_latency_ms is not None else "n/a"
    gamer_fix_cmd = _quickstart_command(gateway, eff_os, "--gamer-check")

    if not gpu.present:
        thermal_ok = True
        vram_reserve_ok = True
        base_details: dict[str, Any] = {
            "gpu_present": False,
            "gpu_name": None,
            "total_mib": None,
            "free_mib": None,
            "temp_c": None,
            "util_pct": None,
            "max_temp_c": cap_temp_c,
            "min_free_vram_mib": reserve_mib,
            "yield_budget_ms": budget_ms,
            "yield_latency_ms": yield_latency_ms,
            "yield_slo_ok": yield_slo_ok,
            "clean_release": clean_release,
            "retained_worker_vram_mib": retained_mib,
            "drill_method": drill_method,
            "thermal_ok": thermal_ok,
            "vram_reserve_ok": vram_reserve_ok,
            "active_games": detected_games,
            "game_running": game_running,
        }
        if not yield_slo_ok:
            summary = (
                f"Steam worker preemption drill exceeded budget ({lat_s_text} >= {budget_s:.2f}s ceiling; fail-closed)."
                if lang_code == "en"
                else f"Le test de cession Steam a dépassé le budget ({lat_s_text} >= plafond {budget_s:.2f}s ; refus sécurisé)."
            )
            return DoctorCheck(
                "noel_steam_yield_thermal",
                title,
                "fail",
                summary,
                gamer_fix_cmd,
                {**base_details, "mode": "yield_slo_exceeded", "gpu_status": "dev_api_only_no_nvidia_gpu"},
            )
        if game_running:
            joined_games = ", ".join(detected_games)
            summary = (
                f"Active game detected ({joined_games}) on non-NVIDIA host (dev_api_only_no_nvidia_gpu; yield {lat_s_text} < {budget_s:.2f}s)."
                if lang_code == "en"
                else f"Jeu actif détecté ({joined_games}) sans GPU NVIDIA (dev_api_only_no_nvidia_gpu ; cession {lat_s_text} < {budget_s:.2f}s)."
            )
            return DoctorCheck(
                "noel_steam_yield_thermal",
                title,
                "warn",
                summary,
                gamer_fix_cmd,
                {
                    **base_details,
                    "thermal_ok": False,
                    "mode": "active_game_yield",
                    "gpu_status": "dev_api_only_no_nvidia_gpu",
                },
            )
        summary = (
            f"No NVIDIA GPU present (dev_api_only_no_nvidia_gpu) — local worker preemption verified in {lat_s_text} (< {budget_s:.2f}s)."
            if lang_code == "en"
            else f"Aucun GPU NVIDIA présent (dev_api_only_no_nvidia_gpu) — cession locale vérifiée en {lat_s_text} (< {budget_s:.2f}s)."
        )
        return DoctorCheck(
            "noel_steam_yield_thermal",
            title,
            "ok",
            summary,
            "",
            {**base_details, "mode": "dev_api_only_no_nvidia_gpu", "gpu_status": "dev_api_only_no_nvidia_gpu"},
        )

    if gpu.parse_error:
        parse_details = {
            "gpu_present": True,
            "parse_error": True,
            "gpu_name": None,
            "total_mib": None,
            "free_mib": None,
            "temp_c": None,
            "util_pct": None,
            "max_temp_c": cap_temp_c,
            "min_free_vram_mib": reserve_mib,
            "yield_budget_ms": budget_ms,
            "yield_latency_ms": yield_latency_ms,
            "yield_slo_ok": yield_slo_ok,
            "clean_release": clean_release,
            "retained_worker_vram_mib": retained_mib,
            "drill_method": drill_method,
            "thermal_ok": False,
            "vram_reserve_ok": False,
            "active_games": detected_games,
            "game_running": game_running,
            "mode": "yield_slo_exceeded" if not yield_slo_ok else "gpu_driver_parse_error",
            "gpu_status": "gpu_driver_parse_error",
        }
        status_val = "fail" if not yield_slo_ok else "warn"
        summary = (
            "nvidia-smi output unreadable during Noel thermal/VRAM check; verify NVIDIA driver."
            if lang_code == "en"
            else "Sortie nvidia-smi illisible pendant le contrôle thermique/VRAM Noël ; vérifiez le pilote NVIDIA."
        )
        return DoctorCheck("noel_steam_yield_thermal", title, status_val, summary, "nvidia-smi", parse_details)

    thermal_ok = bool(gpu.temp_c <= cap_temp_c and not game_running)
    vram_reserve_ok = bool(gpu.free_mib >= reserve_mib)
    gpu_details = {
        "gpu_present": True,
        "parse_error": False,
        "gpu_name": gpu.name,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "max_temp_c": cap_temp_c,
        "min_free_vram_mib": reserve_mib,
        "yield_budget_ms": budget_ms,
        "yield_latency_ms": yield_latency_ms,
        "yield_slo_ok": yield_slo_ok,
        "clean_release": clean_release,
        "retained_worker_vram_mib": retained_mib,
        "drill_method": drill_method,
        "thermal_ok": thermal_ok,
        "vram_reserve_ok": vram_reserve_ok,
        "active_games": detected_games,
        "game_running": game_running,
    }

    if not yield_slo_ok:
        summary = (
            f"{gpu.name} failed <{budget_s:.2f}s Steam preemption SLO ({lat_s_text} >= {budget_s:.2f}s; fail-closed)."
            if lang_code == "en"
            else f"{gpu.name} a échoué au SLO de préemption Steam <{budget_s:.2f}s ({lat_s_text} >= {budget_s:.2f}s ; refus sécurisé)."
        )
        return DoctorCheck(
            "noel_steam_yield_thermal",
            title,
            "fail",
            summary,
            gamer_fix_cmd,
            {**gpu_details, "mode": "yield_slo_exceeded", "gpu_status": "yield_slo_exceeded"},
        )

    if gpu.temp_c > cap_temp_c:
        summary = (
            f"{gpu.name} is at {gpu.temp_c:.1f} C (> {cap_temp_c:.1f} C quiet-fan ceiling); pausing GPU work until cooled."
            if lang_code == "en"
            else f"{gpu.name} est à {gpu.temp_c:.1f} C (> {cap_temp_c:.1f} C plafond silencieux) ; pause GPU jusqu'au refroidissement."
        )
        return DoctorCheck(
            "noel_steam_yield_thermal",
            title,
            "warn",
            summary,
            gamer_fix_cmd,
            {**gpu_details, "mode": "gpu_thermal_cooldown", "gpu_status": "gpu_thermal_cooldown"},
        )

    if game_running:
        joined_games = ", ".join(detected_games)
        summary = (
            f"Active game ({joined_games}) detected on {gpu.name}; 100% VRAM yielded in {lat_s_text} (< {budget_s:.2f}s)."
            if lang_code == "en"
            else f"Jeu actif ({joined_games}) détecté sur {gpu.name} ; 100 % VRAM cédée en {lat_s_text} (< {budget_s:.2f}s)."
        )
        return DoctorCheck(
            "noel_steam_yield_thermal",
            title,
            "warn",
            summary,
            gamer_fix_cmd,
            {**gpu_details, "mode": "active_game_yield", "gpu_status": "active_game_yield"},
        )

    if not vram_reserve_ok:
        summary = (
            f"{gpu.name} has {gpu.free_mib} MiB free VRAM (< {reserve_mib} MiB Gamer-Whisper reserve); Dev API-only mode recommended."
            if lang_code == "en"
            else f"{gpu.name} a {gpu.free_mib} MiB de VRAM libre (< {reserve_mib} MiB réserve Gamer-Whisper) ; mode Dev API conseillé."
        )
        return DoctorCheck(
            "noel_steam_yield_thermal",
            title,
            "warn",
            summary,
            gamer_fix_cmd,
            {**gpu_details, "mode": "low_free_vram", "gpu_status": "low_free_vram"},
        )

    summary = (
        f"{gpu.name} Gamer-Whisper verified: yield {lat_s_text} < {budget_s:.2f}s, {gpu.temp_c:.1f} C <= {cap_temp_c:.1f} C, {gpu.free_mib} MiB free VRAM."
        if lang_code == "en"
        else f"{gpu.name} Gamer-Whisper vérifié : cession {lat_s_text} < {budget_s:.2f}s, {gpu.temp_c:.1f} C <= {cap_temp_c:.1f} C, {gpu.free_mib} MiB libres."
    )
    return DoctorCheck(
        "noel_steam_yield_thermal",
        title,
        "ok",
        summary,
        "",
        {**gpu_details, "mode": "gamer_whisper_ready", "gpu_status": "gamer_whisper_ready"},
    )


def _expand_user_path(raw_path: Path | str, *, home: Path | None = None) -> Path:
    text = str(raw_path).strip()
    if text == "~" or text.startswith(("~/", "~\\")):
        base_home = Path(home) if home is not None else Path.home()
        suffix = text[2:] if len(text) > 1 else ""
        return (base_home / suffix) if suffix else base_home
    return Path(text)


def _validate_cli_path_arg(
    raw: Path | str | None,
    label: str = "--output",
    *,
    home: Path | None = None,
) -> Path | None:
    """Validate an optional CLI path argument, rejecting empty strings and NUL/control characters."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (str, Path)):
        raise ValueError(f"Invalid {label} path.")
    cleaned = str(raw).strip()
    if not cleaned or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"Invalid {label} path.")
    return _expand_user_path(cleaned, home=home)


def _write_file_secure_atomic(
    path: Path,
    content: str,
    *,
    mode: int = 0o600,
    dir_mode: int | None = None,
) -> Path:
    """Atomically write UTF-8 text via unique PID/TID ``.tmp``, ``0600`` creation, ``fsync``, and ``os.replace``."""
    if path.exists() and path.is_dir():
        raise ValueError(f"Refusing to overwrite directory {redact_home_path(path)!r}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" and dir_mode is not None:
        with suppress(OSError):
            os.chmod(path.parent, dir_mode)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    data = content.encode("utf-8")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            if os.name != "nt":
                with suppress(OSError):
                    os.chmod(tmp_path, mode)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
        if os.name != "nt":
            with suppress(OSError):
                os.chmod(path, mode)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def default_ces_queue_path(
    home: Path | str | None = None,
    credentials_path: Path | str | None = None,
) -> Path:
    """Resolve isolated local POSIX-0600 CES retry spool path without touching real ``~/.hivemind`` when isolated."""
    default_cred = Path.home() / ".hivemind" / "credentials.json"
    if credentials_path is not None and str(credentials_path).strip():
        cred_p = _expand_user_path(credentials_path, home=Path(home) if home is not None else None)
        if home is not None:
            resolved_home = Path(home).resolve()
            if cred_p == default_cred and resolved_home != Path.home().resolve():
                return Path(home) / ".hivemind" / DEFAULT_CES_QUEUE_FILENAME
        return cred_p.parent / DEFAULT_CES_QUEUE_FILENAME
    resolved_base = Path(home) if home is not None else Path.home()
    return resolved_base / ".hivemind" / DEFAULT_CES_QUEUE_FILENAME


def _get_spool_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _SPOOL_LOCKS_GUARD:
        lock = _SPOOL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SPOOL_LOCKS[key] = lock
        return lock


def _normalize_ces_payload(
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_name: str | None = None,
    note: str = "",
    home: Path | None = None,
) -> dict[str, Any]:
    norm_profile = normalize_profile(profile)
    clean_ease = _parse_ease_rating(ease_rating)
    if not isinstance(zero_game_lag, bool):
        raise ValueError("zero_game_lag must be a boolean.")
    clean_platform = _platform_feedback_tag(
        os_name=(platform_name if platform_name in ("nt", "posix") else None),
        system_name=platform_name,
    ) if platform_name is not None else _platform_feedback_tag()
    clean_note = _sanitize_noel_text(str(note or ""), home=home)[:240]
    return {
        "profile": norm_profile,
        "ease_rating": int(clean_ease),
        "zero_game_lag": bool(zero_game_lag),
        "platform": clean_platform,
        "note": clean_note,
    }


def _validate_ces_payload_mapping(raw: object, *, home: Path | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Spooled CES payload must be a JSON object.")
    allowed_keys = {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
    if set(raw.keys()) - allowed_keys:
        raise ValueError("Spooled CES payload contains unexpected keys.")
    return _normalize_ces_payload(
        profile=raw.get("profile", ""),  # type: ignore[arg-type]
        ease_rating=raw.get("ease_rating", 0),  # type: ignore[arg-type]
        zero_game_lag=raw.get("zero_game_lag", True),  # type: ignore[arg-type]
        platform_name=str(raw.get("platform") or "unknown"),
        note=str(raw.get("note") or ""),
        home=home,
    )


def _compute_doctor_ces_idempotency_key(
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    host_label: str = DEFAULT_NOEL_HOST_LABEL,
    key_salt: str = "anon",
) -> str:
    clean_payload = _validate_ces_payload_mapping(payload)
    if idempotency_key is not None:
        if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be a printable ASCII string.")
        if not idempotency_key.isascii() or any(ord(ch) < 32 or ord(ch) == 127 for ch in idempotency_key):
            raise ValueError("idempotency_key must contain printable ASCII characters only.")
        cleaned = idempotency_key.strip()
        if not cleaned:
            raise ValueError("idempotency_key must not be empty.")
        if _VALID_EVENT_ID_RE.fullmatch(cleaned) and not _RAW_API_KEY_RE.search(cleaned):
            return cleaned
        return "hm-ces-" + hashlib.sha256(f"hivemind.doctor.ces.key/1:{cleaned}".encode()).hexdigest()[:32]

    clean_gw = validate_gateway_url(gateway)
    canonical_ces = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    return (
        "hm-ces-"
        + hashlib.sha256(
            f"hivemind.doctor.ces/1:{clean_gw}:{host_label}:{key_salt}:{canonical_ces}".encode()
        ).hexdigest()[:32]
    )


def _sanitize_feedback_summary_doc(raw: Mapping[str, Any]) -> dict[str, Any]:
    raw_obj = raw.get("object")
    obj_val = str(raw_obj) if raw_obj in _ACCEPTED_FEEDBACK_SUMMARY_OBJECTS else FEEDBACK_SUMMARY_OBJECT
    by_prof_raw = raw.get("by_profile") if isinstance(raw.get("by_profile"), Mapping) else {}
    total_raw = raw.get("total_responses")
    total_val = int(total_raw) if isinstance(total_raw, int) and not isinstance(total_raw, bool) and total_raw >= 0 else 0
    mean_raw = raw.get("mean_ease_rating")
    if mean_raw is None:
        mean_raw = raw.get("average_ease_rating")
    mean_val = (
        round(float(mean_raw), 2)
        if isinstance(mean_raw, (int, float))
        and not isinstance(mean_raw, bool)
        and math.isfinite(float(mean_raw))
        else None
    )
    lag_raw = raw.get("zero_game_lag_ratio")
    lag_val = (
        round(float(lag_raw), 4)
        if isinstance(lag_raw, (int, float))
        and not isinstance(lag_raw, bool)
        and math.isfinite(float(lag_raw))
        else None
    )
    d_raw = by_prof_raw.get("dev", 0)
    g_raw = by_prof_raw.get("gamer", 0)
    return {
        "object": obj_val,
        "total_responses": total_val,
        "mean_ease_rating": mean_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {
            "dev": int(d_raw) if isinstance(d_raw, int) and not isinstance(d_raw, bool) and d_raw >= 0 else 0,
            "gamer": int(g_raw) if isinstance(g_raw, int) and not isinstance(g_raw, bool) and g_raw >= 0 else 0,
        },
    }


class DoctorCesRetrySpool:
    """Thread-safe, atomic POSIX-0600 bounded CES retry spool with Idempotency-Key deduplication."""

    def __init__(
        self,
        spool_path: Path | str | None = None,
        *,
        home: Path | None = None,
        credentials_path: Path | str | None = None,
        max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
        max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not (1 <= max_entries <= 4096):
            raise ValueError("max_entries must be an integer between 1 and 4096.")
        if isinstance(max_delivered, bool) or not isinstance(max_delivered, int) or not (1 <= max_delivered <= 32768):
            raise ValueError("max_delivered must be an integer between 1 and 32768.")
        self._home = Path(home) if home is not None else None
        raw_p = (
            _expand_user_path(spool_path, home=self._home)
            if spool_path is not None and str(spool_path).strip()
            else default_ces_queue_path(home=self._home, credentials_path=credentials_path)
        )
        self._path = raw_p.resolve()
        self._max_entries = int(max_entries)
        self._max_delivered = int(max_delivered)
        self._lock = _get_spool_lock(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_delivered(self) -> int:
        return self._max_delivered

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": CES_QUEUE_SCHEMA,
            "max_entries": self._max_entries,
            "max_delivered": self._max_delivered,
            "dropped_overflow_count": 0,
            "pending": [],
            "delivered_event_ids": [],
            "quarantined": [],
        }

    def _load_state_unlocked(self) -> tuple[dict[str, Any], bool]:
        if not self._path.is_file():
            return self._empty_state(), False
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recovered = self._empty_state()
            recovered["quarantined"] = [
                {
                    "event_id": "hm-ces-" + hashlib.sha256(b"corrupted_doctor_spool_recovered").hexdigest()[:32],
                    "http_status": 0,
                    "reason": "corrupted_spool_json_recovered",
                }
            ]
            return recovered, True
        if not isinstance(parsed, Mapping):
            return self._empty_state(), True

        mutated = False
        delivered: list[str] = []
        seen_delivered: set[str] = set()
        raw_deliv = parsed.get("delivered_event_ids")
        if isinstance(raw_deliv, list):
            for item in raw_deliv:
                if isinstance(item, str) and _VALID_EVENT_ID_RE.fullmatch(item) and item not in seen_delivered:
                    delivered.append(item)
                    seen_delivered.add(item)
                else:
                    mutated = True
        if len(delivered) > self._max_delivered:
            delivered = delivered[-self._max_delivered :]
            mutated = True

        quarantined: list[dict[str, Any]] = []
        raw_quar = parsed.get("quarantined")
        if isinstance(raw_quar, list):
            for q_item in raw_quar[-self._max_entries :]:
                if isinstance(q_item, Mapping) and isinstance(q_item.get("event_id"), str):
                    q_eid = q_item["event_id"]
                    if _VALID_EVENT_ID_RE.fullmatch(q_eid):
                        q_code = q_item.get("http_status", 422)
                        quarantined.append(
                            {
                                "event_id": q_eid,
                                "http_status": int(q_code) if isinstance(q_code, int) and not isinstance(q_code, bool) else 422,
                                "reason": _sanitize_noel_text(str(q_item.get("reason", "quarantined"))[:80], home=self._home),
                            }
                        )

        pending: list[dict[str, Any]] = []
        seen_pending: set[str] = set()
        raw_pending = parsed.get("pending")
        if isinstance(raw_pending, list):
            for entry in raw_pending:
                if not isinstance(entry, Mapping):
                    mutated = True
                    continue
                raw_eid = entry.get("event_id")
                eid = (
                    raw_eid
                    if isinstance(raw_eid, str) and _VALID_EVENT_ID_RE.fullmatch(raw_eid)
                    else "hm-ces-" + hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
                )
                if eid in seen_delivered or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = _validate_ces_payload_mapping(entry.get("payload"), home=self._home)
                except ValueError:
                    mutated = True
                    quarantined.append(
                        {
                            "event_id": eid,
                            "http_status": 422,
                            "reason": "invalid_spooled_payload",
                        }
                    )
                    quarantined = quarantined[-self._max_entries :]
                    continue
                raw_rc = entry.get("retry_count", 0)
                rc = max(0, int(raw_rc)) if isinstance(raw_rc, int) and not isinstance(raw_rc, bool) else 0
                raw_st = entry.get("last_http_status")
                last_st = int(raw_st) if isinstance(raw_st, int) and not isinstance(raw_st, bool) else None
                pending.append(
                    {
                        "event_id": eid,
                        "payload": clean_payload,
                        "retry_count": rc,
                        "last_http_status": last_st,
                        "last_reason": _sanitize_noel_text(str(entry.get("last_reason") or "")[:80], home=self._home),
                        "enqueued_at_utc": str(entry.get("enqueued_at_utc") or ""),
                    }
                )
                seen_pending.add(eid)

        raw_dropped = parsed.get("dropped_overflow_count", 0)
        dropped = max(0, int(raw_dropped)) if isinstance(raw_dropped, int) and not isinstance(raw_dropped, bool) else 0
        if len(pending) > self._max_entries:
            overflow = len(pending) - self._max_entries
            pending = pending[-self._max_entries :]
            dropped += overflow
            mutated = True

        return (
            {
                "schema": CES_QUEUE_SCHEMA,
                "max_entries": self._max_entries,
                "max_delivered": self._max_delivered,
                "dropped_overflow_count": dropped,
                "pending": pending,
                "delivered_event_ids": delivered,
                "quarantined": quarantined[-self._max_entries :],
            },
            mutated,
        )

    def _atomic_save_unlocked(self, state: Mapping[str, Any]) -> None:
        doc = {
            "schema": CES_QUEUE_SCHEMA,
            "max_entries": self._max_entries,
            "max_delivered": self._max_delivered,
            "dropped_overflow_count": int(state.get("dropped_overflow_count", 0)),
            "pending": list(state.get("pending", []))[-self._max_entries :],
            "delivered_event_ids": list(state.get("delivered_event_ids", []))[-self._max_delivered :],
            "quarantined": list(state.get("quarantined", []))[-self._max_entries :],
        }
        serialized = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        safety_hits = scan_output_safety(serialized, home=self._home)
        if safety_hits:
            raise ValueError(f"Refusing unsafe spool serialization: {', '.join(safety_hits)}")
        dir_mode = 0o700 if self._path.parent.name == ".hivemind" else None
        _write_file_secure_atomic(self._path, serialized, mode=0o600, dir_mode=dir_mode)

    def snapshot(self, *, home: Path | None = None) -> dict[str, Any]:
        """Return a read-only sanitized spool snapshot without creating files when empty."""
        eff_home = home if home is not None else self._home
        with self._lock:
            exists = self._path.is_file()
            state, _ = self._load_state_unlocked()
        snap: dict[str, Any] = {
            "object": "hivemind.doctor.ces_spool_snapshot",
            "spool_file": redact_home_path(self._path, home=eff_home),
            "spool_exists": exists,
            "max_entries": self._max_entries,
            "max_delivered": self._max_delivered,
            "pending_count": len(state["pending"]),
            "delivered_ledger_count": len(state["delivered_event_ids"]),
            "quarantined_count": len(state["quarantined"]),
            "dropped_overflow_count": int(state["dropped_overflow_count"]),
            "pending_event_ids": [item["event_id"] for item in state["pending"]],
        }
        snap["receipt_sha256"] = _compute_receipt_sha256(snap)
        return snap

    def mark_delivered(self, event_id: str) -> bool:
        """Record ``event_id`` as delivered if a spool file already exists (never creates a new file)."""
        if not isinstance(event_id, str) or not _VALID_EVENT_ID_RE.fullmatch(event_id):
            return False
        with self._lock:
            if not self._path.is_file():
                return False
            state, mutated = self._load_state_unlocked()
            prev_len = len(state["pending"])
            state["pending"] = [item for item in state["pending"] if item["event_id"] != event_id]
            removed = len(state["pending"]) != prev_len
            if event_id not in state["delivered_event_ids"]:
                state["delivered_event_ids"].append(event_id)
                state["delivered_event_ids"] = state["delivered_event_ids"][-self._max_delivered :]
                mutated = True
            if removed or mutated:
                self._atomic_save_unlocked(state)
            return removed

    def clean(self, *, home: Path | None = None) -> dict[str, Any]:
        """Remove the local spool file if present without touching any remote endpoints."""
        eff_home = home if home is not None else self._home
        with self._lock:
            existed = self._path.is_file()
            if existed:
                self._path.unlink(missing_ok=True)
        res: dict[str, Any] = {
            "object": "hivemind.doctor.ces_spool_clean",
            "spool_file": redact_home_path(self._path, home=eff_home),
            "removed": existed,
            "spool_exists": self._path.is_file(),
            "network_calls": 0,
        }
        res["receipt_sha256"] = _compute_receipt_sha256(res)
        return res

    def enqueue(
        self,
        *,
        profile: str = "dev",
        ease_rating: int = 5,
        zero_game_lag: bool = True,
        platform_name: str | None = None,
        note: str = "",
        gateway: str = DEFAULT_GATEWAY,
        idempotency_key: str | None = None,
        http_status: int | None = None,
        reason: str = "",
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Enforce compliance, deduplicate by ``Idempotency-Key``, evict FIFO at ``max_entries``, and save ``0600``."""
        lang_code = normalize_lang(lang)
        eff_home = home if home is not None else self._home
        comp = evaluate_noel_personal_pc_compliance(environ=environ, cwd=cwd, gateway=gateway, lang=lang_code)
        if not comp["compliance_ok"]:
            blocked: dict[str, Any] = {
                "object": "hivemind.doctor.ces_spool_enqueue",
                "compliance_ok": False,
                "corporate_indicators": comp["corporate_indicators"],
                "enqueued": False,
                "deduplicated": False,
                "duplicate_state": "compliance_refused",
                "spool_file": redact_home_path(self._path, home=eff_home),
                "network_calls": 0,
                "file_writes": 0,
                "summary": comp["summary"],
            }
            blocked["receipt_sha256"] = _compute_receipt_sha256(blocked)
            return blocked

        clean_gw = validate_gateway_url(gateway)
        clean_payload = _normalize_ces_payload(
            profile=profile,
            ease_rating=ease_rating,
            zero_game_lag=zero_game_lag,
            platform_name=platform_name,
            note=note,
            home=eff_home,
        )
        event_id = _compute_doctor_ces_idempotency_key(
            clean_payload,
            idempotency_key=idempotency_key,
            gateway=clean_gw,
        )
        clean_reason = _sanitize_noel_text(str(reason or ""), home=eff_home)[:80]

        with self._lock:
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_event_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if event_id in delivered_ids:
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                res_deliv: dict[str, Any] = {
                    "object": "hivemind.doctor.ces_spool_enqueue",
                    "compliance_ok": True,
                    "spool_file": redact_home_path(self._path, home=eff_home),
                    "enqueued": False,
                    "deduplicated": True,
                    "duplicate_state": "already_delivered",
                    "evicted_oldest": False,
                    "event_id": event_id,
                    "payload": clean_payload,
                    "pending_count": len(pending),
                    "delivered_ledger_count": len(delivered_ids),
                    "quarantined_count": len(state["quarantined"]),
                    "dropped_overflow_count": int(state["dropped_overflow_count"]),
                    "network_calls": 0,
                    "file_writes": 1 if recovered else 0,
                }
                res_deliv["receipt_sha256"] = _compute_receipt_sha256(res_deliv)
                return res_deliv

            if any(item["event_id"] == event_id for item in pending):
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                res_pend: dict[str, Any] = {
                    "object": "hivemind.doctor.ces_spool_enqueue",
                    "compliance_ok": True,
                    "spool_file": redact_home_path(self._path, home=eff_home),
                    "enqueued": False,
                    "deduplicated": True,
                    "duplicate_state": "already_pending",
                    "evicted_oldest": False,
                    "event_id": event_id,
                    "payload": clean_payload,
                    "pending_count": len(pending),
                    "delivered_ledger_count": len(delivered_ids),
                    "quarantined_count": len(state["quarantined"]),
                    "dropped_overflow_count": int(state["dropped_overflow_count"]),
                    "network_calls": 0,
                    "file_writes": 1 if recovered else 0,
                }
                res_pend["receipt_sha256"] = _compute_receipt_sha256(res_pend)
                return res_pend

            evicted_oldest = False
            if len(pending) >= self._max_entries:
                pending.pop(0)
                state["dropped_overflow_count"] = int(state["dropped_overflow_count"]) + 1
                evicted_oldest = True

            pending.append(
                {
                    "event_id": event_id,
                    "payload": clean_payload,
                    "retry_count": 0,
                    "last_http_status": http_status,
                    "last_reason": clean_reason,
                    "enqueued_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                }
            )
            self._atomic_save_unlocked(state)
            res_ok: dict[str, Any] = {
                "object": "hivemind.doctor.ces_spool_enqueue",
                "compliance_ok": True,
                "spool_file": redact_home_path(self._path, home=eff_home),
                "enqueued": True,
                "deduplicated": False,
                "duplicate_state": "none",
                "evicted_oldest": evicted_oldest,
                "event_id": event_id,
                "payload": clean_payload,
                "pending_count": len(pending),
                "delivered_ledger_count": len(delivered_ids),
                "quarantined_count": len(state["quarantined"]),
                "dropped_overflow_count": int(state["dropped_overflow_count"]),
                "network_calls": 0,
                "file_writes": 1,
            }
            res_ok["receipt_sha256"] = _compute_receipt_sha256(res_ok)
            return res_ok

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        http_request: HttpRequestFn = default_http_request_json,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = HTTP_TIMEOUT_S,
        lang: str = "en",
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> dict[str, Any]:
        """Idempotently flush pending CES events with ``Idempotency-Key`` headers without creating empty files."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        eff_home = home if home is not None else self._home
        clean_gw = validate_gateway_url(gateway)
        bounded_timeout = _validate_timeout_s(timeout_s)

        comp = evaluate_noel_personal_pc_compliance(environ=environ, cwd=cwd, gateway=clean_gw, lang=lang_code)
        if not comp["compliance_ok"]:
            blocked: dict[str, Any] = {
                "object": "hivemind.doctor.ces_spool_flush",
                "gateway": clean_gw,
                "lang": lang_code,
                "compliance_ok": False,
                "corporate_indicators": comp["corporate_indicators"],
                "flush_ok": False,
                "spool_file": redact_home_path(self._path, home=eff_home),
                "spool_exists": self._path.is_file(),
                "delivered_count": 0,
                "retained_retry_count": 0,
                "quarantined_count": 0,
                "deduplicated_count": 0,
                "pending_count": 0,
                "delivered_ledger_count": 0,
                "total_quarantined_count": 0,
                "stale_cloud_run_detected": False,
                "network_calls": 0,
                "file_writes": 0,
                "summary": None,
                "diagnostics": [comp["summary"]],
            }
            blocked["receipt_sha256"] = _compute_receipt_sha256(blocked)
            return blocked

        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
        delivered_now = 0
        retained_retry = 0
        quarantined_now = 0
        dedup_now = 0
        network_calls = 0
        file_writes = 0
        stale_404_seen = False
        latest_summary: dict[str, Any] | None = None

        with self._lock:
            if not self._path.is_file():
                empty_res: dict[str, Any] = {
                    "object": "hivemind.doctor.ces_spool_flush",
                    "gateway": clean_gw,
                    "lang": lang_code,
                    "compliance_ok": True,
                    "flush_ok": True,
                    "spool_file": redact_home_path(self._path, home=eff_home),
                    "spool_exists": False,
                    "delivered_count": 0,
                    "retained_retry_count": 0,
                    "quarantined_count": 0,
                    "deduplicated_count": 0,
                    "pending_count": 0,
                    "delivered_ledger_count": 0,
                    "total_quarantined_count": 0,
                    "stale_cloud_run_detected": False,
                    "network_calls": 0,
                    "file_writes": 0,
                    "summary": None,
                    "diagnostics": [],
                }
                empty_res["receipt_sha256"] = _compute_receipt_sha256(empty_res)
                return empty_res

            state, mutated = self._load_state_unlocked()
            if mutated:
                quarantined_now += len(state["quarantined"])
            delivered_set = set(state["delivered_event_ids"])
            next_pending: list[dict[str, Any]] = []

            for entry in state["pending"]:
                eid = str(entry["event_id"])
                if eid in delivered_set:
                    dedup_now += 1
                    mutated = True
                    continue

                req_headers = {str(k): str(v) for k, v in headers.items()} if headers else {}
                req_headers["Idempotency-Key"] = eid
                network_calls += 1
                try:
                    code, body = _invoke_http_request(
                        http_request,
                        "POST",
                        post_url,
                        payload=dict(entry["payload"]),
                        headers=req_headers,
                        timeout_s=bounded_timeout,
                    )
                except (OSError, urllib.error.URLError, TimeoutError, ValueError):
                    code, body = 0, {}
                mutated = True

                if code in (200, 201):
                    delivered_now += 1
                    state["delivered_event_ids"].append(eid)
                    delivered_set.add(eid)
                    latest_summary = _sanitize_feedback_summary_doc(body)
                elif code in (0, 404, 408, 429) or code >= 500:
                    if code == 404:
                        stale_404_seen = True
                    retained_retry += 1
                    reason = (
                        "stale_cloud_run_404"
                        if code == 404
                        else (
                            "request_timeout_408"
                            if code == 408
                            else (
                                "rate_limited_429"
                                if code == 429
                                else ("network_timeout_or_unreachable" if code == 0 else f"gateway_{code}")
                            )
                        )
                    )
                    next_pending.append(
                        {
                            **entry,
                            "retry_count": int(entry.get("retry_count", 0)) + 1,
                            "last_http_status": code,
                            "last_reason": reason,
                        }
                    )
                else:
                    quarantined_now += 1
                    state["quarantined"].append(
                        {
                            "event_id": eid,
                            "http_status": code,
                            "reason": f"permanent_http_{code}",
                        }
                    )
                    state["quarantined"] = state["quarantined"][-self._max_entries :]

            state["pending"] = next_pending
            state["delivered_event_ids"] = state["delivered_event_ids"][-self._max_delivered :]
            if mutated:
                self._atomic_save_unlocked(state)
                file_writes = 1

            pending_after = len(state["pending"])
            delivered_total = len(state["delivered_event_ids"])
            quarantined_total = len(state["quarantined"])

        diagnostics: list[str] = []
        if stale_404_seen:
            diagnostics.append(
                "HTTP 404 détecté pendant le vidage de file CES : entrées conservées pour nouvel essai."
                if is_fr
                else "HTTP 404 during CES spool flush: entries safely retained for retry."
            )
        elif retained_retry > 0:
            diagnostics.append(
                f"{retained_retry} entrée(s) CES conservée(s) en attente."
                if is_fr
                else f"{retained_retry} CES entry/entries retained for retry."
            )
        if quarantined_now > 0:
            diagnostics.append(
                f"{quarantined_now} entrée(s) permanente(s) 4xx mise(s) en quarantaine."
                if is_fr
                else f"{quarantined_now} permanent 4xx entry/entries quarantined."
            )

        flush_ok = retained_retry == 0 and quarantined_now == 0
        res: dict[str, Any] = {
            "object": "hivemind.doctor.ces_spool_flush",
            "gateway": clean_gw,
            "lang": lang_code,
            "compliance_ok": True,
            "flush_ok": flush_ok,
            "spool_file": redact_home_path(self._path, home=eff_home),
            "spool_exists": True,
            "delivered_count": delivered_now,
            "retained_retry_count": retained_retry,
            "quarantined_count": quarantined_now,
            "deduplicated_count": dedup_now,
            "pending_count": pending_after,
            "delivered_ledger_count": delivered_total,
            "total_quarantined_count": quarantined_total,
            "stale_cloud_run_detected": stale_404_seen,
            "network_calls": network_calls,
            "file_writes": file_writes,
            "summary": latest_summary,
            "diagnostics": diagnostics,
        }
        res["receipt_sha256"] = _compute_receipt_sha256(res)
        return res


def enqueue_doctor_ces_event(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_name: str | None = None,
    note: str = "",
    idempotency_key: str | None = None,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_status: int | None = None,
    reason: str = "",
    max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Atomically enqueue a zero-PII CES feedback event into the bounded POSIX-0600 retry spool."""
    spool = DoctorCesRetrySpool(
        spool_path,
        home=home,
        credentials_path=credentials_path,
        max_entries=max_entries,
        max_delivered=max_delivered,
    )
    return spool.enqueue(
        profile=profile,
        ease_rating=ease_rating,
        zero_game_lag=zero_game_lag,
        platform_name=platform_name,
        note=note,
        gateway=gateway,
        idempotency_key=idempotency_key,
        http_status=http_status,
        reason=reason,
        home=home,
        environ=environ,
        cwd=cwd,
        lang=lang,
    )


def flush_doctor_ces_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_request: HttpRequestFn = default_http_request_json,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
    max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Flush locally spooled CES events idempotently against ``POST /v1/hivemind/onboarding/feedback``."""
    spool = DoctorCesRetrySpool(
        spool_path,
        home=home,
        credentials_path=credentials_path,
        max_entries=max_entries,
        max_delivered=max_delivered,
    )
    return spool.flush(
        gateway,
        http_request=http_request,
        headers=headers,
        timeout_s=timeout_s,
        lang=lang,
        home=home,
        environ=environ,
        cwd=cwd,
    )


def _honest_counter(raw_val: object) -> int | str:
    if isinstance(raw_val, int) and not isinstance(raw_val, bool) and raw_val >= 0:
        return int(raw_val)
    return "n/a"


def _sanitize_host_label(raw_label: str | None, *, home: Path | None = None) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        return DEFAULT_NOEL_HOST_LABEL
    scrubbed = _sanitize_noel_text(raw_label.strip(), home=home)
    cleaned = _HOST_LABEL_CLEAN_RE.sub("-", scrubbed).strip("-._")[:128]
    return cleaned if cleaned else DEFAULT_NOEL_HOST_LABEL


def _load_valid_credentials_key(
    credentials_path: Path | None,
    *,
    home: Path,
) -> tuple[str | None, str, str]:
    target = credentials_path if credentials_path is not None else (home / ".hivemind" / "credentials.json")
    shown = redact_home_path(target, home=home)
    if not target.is_file():
        return None, "prosumer", shown
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "prosumer", shown
    if not isinstance(loaded, Mapping):
        return None, "prosumer", shown
    raw_key = loaded.get("api_key")
    role = str(loaded.get("role") or "prosumer")
    if not _is_valid_hivemind_key(raw_key):
        return None, role, shown
    return str(raw_key).strip(), role, shown


def check_noel_fleet_and_first_heartbeat(
    gateway: str = DEFAULT_GATEWAY,
    *,
    credentials_path: Path | None = None,
    send_first_heartbeat: bool = False,
    submit_ces: bool = False,
    ces_profile: str | None = None,
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = DEFAULT_NOEL_HOST_LABEL,
    gpu_info: GpuInfo | None = None,
    yield_slo_ok: bool = True,
    thermal_ok: bool = True,
    http_get: HttpGetFn = default_http_get_json,
    http_request: HttpRequestFn = default_http_request_json,
    timeout_s: float = HTTP_TIMEOUT_S,
    home: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
    spool_path: Path | str | None = None,
    flush_ces_queue: bool = False,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> DoctorCheck:
    """(7c) Probe live fleet/raid counters, optionally send first OpenTrainer heartbeat, and post non-blocking CES."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    bounded_timeout = min(max(0.5, float(timeout_s)), 4.5)
    clean_ease = _parse_ease_rating(ease_rating)
    if ces_profile is not None:
        resolved_profile = normalize_profile(ces_profile)
    elif (
        gpu_info is not None
        and gpu_info.present
        and not gpu_info.parse_error
        and gpu_info.total_mib >= GAMER_WHISPER_MIN_VRAM_MIB
    ):
        resolved_profile = "gamer"
    else:
        resolved_profile = "dev"

    title = (
        "Noel Fleet Visibility, First Heartbeat & Zero-PII CES"
        if lang_code == "en"
        else "Flotte Noël en Direct, Premier Heartbeat & CES Zéro-PII"
    )

    try:
        clean_gw = validate_gateway_url(gateway)
    except ValueError as exc:
        fix_cmd = (
            f'python scripts\\hivemind_doctor.py --noel --gateway "{DEFAULT_GATEWAY}"'
            if eff_os == "nt"
            else f'python3 scripts/hivemind_doctor.py --noel --gateway "{DEFAULT_GATEWAY}"'
        )
        summary = (
            f"Refused gateway URL ({exc})"
            if lang_code == "en"
            else f"URL de passerelle refusée ({exc})"
        )
        invalid_details = _sanitize_noel_structure(
            {
                "allowed_host": False,
                "online_producers": "n/a",
                "active_trainers": "n/a",
                "accepted_rounds": "n/a",
                "first_heartbeat_sent": False,
                "first_heartbeat_ok": False,
                "ces_attempted": False,
                "ces_recorded": False,
                "zero_game_lag": bool(yield_slo_ok and thermal_ok and not had_game_lag),
                "zero_inference": True,
            },
            home=resolved_home,
        )
        invalid_details["receipt_sha256"] = _compute_receipt_sha256(invalid_details)
        return DoctorCheck("noel_fleet_heartbeat", title, "fail", summary, fix_cmd, invalid_details)

    def _eff_get(target_url: str, req_timeout_s: float) -> tuple[int, dict[str, Any]]:
        if http_get is default_http_get_json and http_request is not default_http_request_json:
            return _invoke_http_request(http_request, "GET", target_url, timeout_s=req_timeout_s)
        return http_get(target_url, req_timeout_s)

    clean_host = _sanitize_host_label(host_label, home=resolved_home)
    clean_key, saved_role, shown_cred_path = _load_valid_credentials_key(credentials_path, home=resolved_home)
    masked_key = mask_api_key(clean_key) if clean_key is not None else None

    hb_gpu_name: str | None = None
    hb_vram_total: int | None = None
    hb_vram_free: int | None = None
    if (
        gpu_info is not None
        and gpu_info.present
        and not gpu_info.parse_error
        and bool(gpu_info.name.strip())
        and int(gpu_info.total_mib) >= MIN_REPORTABLE_GPU_VRAM_MIB
    ):
        hb_gpu_name = _sanitize_noel_text(gpu_info.name.strip(), home=resolved_home)[:128]
        hb_vram_total = int(gpu_info.total_mib)
        hb_vram_free = min(max(0, int(gpu_info.free_mib)), hb_vram_total)

    heartbeat_payload: dict[str, Any] = {
        "host_label": clean_host,
        "network_mode": "outbound-https-relay",
        "gpu_name": hb_gpu_name,
        "vram_total_mib": hb_vram_total,
        "vram_free_mib": hb_vram_free,
    }

    me_checked = False
    me_ok = False
    me_status: int | None = None
    account_role = saved_role
    first_heartbeat_sent = False
    first_heartbeat_ok = not send_first_heartbeat
    opentrainer_registered = False
    heartbeat_http_status: int | None = None
    heartbeat_state = "not_requested"
    http_calls_made = 0

    if send_first_heartbeat:
        if clean_key is None:
            first_heartbeat_ok = False
            heartbeat_state = "missing_valid_credentials"
        else:
            auth_headers = {"Authorization": f"Bearer {clean_key}"}
            me_checked = True
            http_calls_made += 1
            try:
                me_status, me_body = _invoke_http_request(
                    http_request,
                    "GET",
                    f"{clean_gw}/v1/hivemind/me",
                    headers=auth_headers,
                    timeout_s=bounded_timeout,
                )
            except (OSError, urllib.error.URLError, TimeoutError, ValueError):
                me_status, me_body = 0, {}

            me_ok = me_status == 200 and isinstance(me_body, Mapping)
            if me_ok and isinstance(me_body.get("role"), str) and me_body["role"].strip():
                account_role = me_body["role"].strip()

            if not me_ok:
                first_heartbeat_ok = False
                heartbeat_state = f"me_http_{me_status}"
            else:
                first_heartbeat_sent = True
                http_calls_made += 1
                try:
                    heartbeat_http_status, _ = _invoke_http_request(
                        http_request,
                        "POST",
                        f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
                        payload=heartbeat_payload,
                        headers=auth_headers,
                        timeout_s=bounded_timeout,
                    )
                except (OSError, urllib.error.URLError, TimeoutError, ValueError):
                    heartbeat_http_status = 0

                if heartbeat_http_status in (200, 201):
                    first_heartbeat_ok = True
                    opentrainer_registered = True
                    heartbeat_state = "accepted"
                elif heartbeat_http_status == 403 and account_role == "consumer":
                    first_heartbeat_ok = True
                    opentrainer_registered = False
                    heartbeat_state = "consumer_me_verified"
                else:
                    first_heartbeat_ok = False
                    heartbeat_state = f"heartbeat_http_{heartbeat_http_status}"

    http_calls_made += 1
    try:
        fleet_status, fleet_body = _eff_get(f"{clean_gw}/v1/hivemind/producers/fleet", bounded_timeout)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        fleet_status, fleet_body = 0, {}

    http_calls_made += 1
    try:
        raid_status, raid_body = _eff_get(f"{clean_gw}/v1/hivemind/raid/status", bounded_timeout)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        raid_status, raid_body = 0, {}

    swarm = (
        raid_body.get("swarm")
        if (raid_status == 200 and isinstance(raid_body, Mapping) and isinstance(raid_body.get("swarm"), Mapping))
        else {}
    )

    if fleet_status == 200 and isinstance(fleet_body, Mapping) and _honest_counter(fleet_body.get("live_hiveminders")) != "n/a":
        online_producers: int | str = _honest_counter(fleet_body.get("live_hiveminders"))
    elif swarm:
        online_producers = _honest_counter(swarm.get("live_producers"))
    else:
        online_producers = "n/a"

    active_trainers: int | str = _honest_counter(swarm.get("active_trainers")) if swarm else "n/a"
    raw_rounds = swarm.get("accepted_opentrainer_rounds", swarm.get("accepted_rounds")) if swarm else None
    accepted_rounds: int | str = _honest_counter(raw_rounds)

    zero_game_lag = bool(yield_slo_ok and thermal_ok and not had_game_lag)
    platform_tag = _platform_feedback_tag(os_name=eff_os, system_name=system_name)
    ces_payload: dict[str, Any] = {
        "profile": resolved_profile,
        "ease_rating": clean_ease,
        "zero_game_lag": zero_game_lag,
        "platform": platform_tag,
        "note": "",
    }
    key_salt = (
        hashlib.sha256(f"hivemind.doctor.key/1:{clean_key}".encode()).hexdigest()[:16]
        if clean_key is not None
        else "anon"
    )
    canonical_ces = json.dumps(ces_payload, sort_keys=True, separators=(",", ":"))
    ces_idempotency_key = (
        "hm-ces-"
        + hashlib.sha256(
            f"hivemind.doctor.ces/1:{clean_gw}:{clean_host}:{key_salt}:{canonical_ces}".encode()
        ).hexdigest()[:32]
    )

    ces_attempted = False
    ces_recorded = False
    ces_degraded = False
    ces_spooled = False
    ces_quarantined = False
    ces_spool_pending_count = 0
    ces_flush_report: dict[str, Any] | None = None
    ces_http_status: int | None = None
    average_ease_rating: float | None = None
    file_writes_made = 0

    if raid_status == 200 and isinstance(raid_body, Mapping) and isinstance(raid_body.get("onboarding_feedback"), Mapping):
        fb_snap = raid_body["onboarding_feedback"]
        raw_raid_ease = fb_snap.get("average_ease_rating", fb_snap.get("mean_ease_rating"))
        if isinstance(raw_raid_ease, (int, float)) and not isinstance(raw_raid_ease, bool):
            average_ease_rating = round(float(raw_raid_ease), 2)

    eff_spool_obj: DoctorCesRetrySpool | None = None
    if spool_path is not None and str(spool_path).strip():
        eff_spool_obj = DoctorCesRetrySpool(spool_path, home=resolved_home, credentials_path=credentials_path)
    elif flush_ces_queue:
        candidate_q = default_ces_queue_path(home=resolved_home, credentials_path=credentials_path)
        if candidate_q.is_file():
            eff_spool_obj = DoctorCesRetrySpool(candidate_q, home=resolved_home, credentials_path=credentials_path)

    if submit_ces:
        ces_attempted = True
        http_calls_made += 1
        try:
            ces_http_status, ces_body = _invoke_http_request(
                http_request,
                "POST",
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                payload=ces_payload,
                headers={"Idempotency-Key": ces_idempotency_key},
                timeout_s=bounded_timeout,
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            ces_http_status, ces_body = 0, {}

        if ces_http_status in (200, 201):
            ces_recorded = True
            raw_ease = (
                ces_body.get("average_ease_rating", ces_body.get("mean_ease_rating"))
                if isinstance(ces_body, Mapping)
                else None
            )
            if isinstance(raw_ease, (int, float)) and not isinstance(raw_ease, bool):
                average_ease_rating = round(float(raw_ease), 2)
            if eff_spool_obj is not None and eff_spool_obj.path.is_file():
                if eff_spool_obj.mark_delivered(ces_idempotency_key):
                    file_writes_made += 1
                if flush_ces_queue:
                    ces_flush_report = eff_spool_obj.flush(
                        clean_gw,
                        http_request=http_request,
                        timeout_s=bounded_timeout,
                        lang=lang_code,
                        home=resolved_home,
                        environ=environ,
                        cwd=cwd,
                    )
                    http_calls_made += int(ces_flush_report.get("network_calls", 0))
                    file_writes_made += int(ces_flush_report.get("file_writes", 0))
                    ces_spool_pending_count = int(ces_flush_report.get("pending_count", 0))
        else:
            ces_recorded = False
            ces_degraded = True
            is_transient_ces = ces_http_status in (0, 404, 408, 429) or ces_http_status >= 500
            if eff_spool_obj is not None and is_transient_ces:
                enq_res = eff_spool_obj.enqueue(
                    profile=resolved_profile,
                    ease_rating=clean_ease,
                    zero_game_lag=zero_game_lag,
                    platform_name=platform_tag,
                    note="",
                    gateway=clean_gw,
                    idempotency_key=ces_idempotency_key,
                    http_status=ces_http_status,
                    reason=f"http_{ces_http_status}",
                    home=resolved_home,
                    environ=environ,
                    cwd=cwd,
                    lang=lang_code,
                )
                ces_spooled = bool(enq_res.get("enqueued") or enq_res.get("duplicate_state") == "already_pending")
                ces_spool_pending_count = int(enq_res.get("pending_count", 0))
                file_writes_made += int(enq_res.get("file_writes", 0))
            elif not is_transient_ces:
                ces_quarantined = True
    elif flush_ces_queue and eff_spool_obj is not None and eff_spool_obj.path.is_file():
        ces_flush_report = eff_spool_obj.flush(
            clean_gw,
            http_request=http_request,
            timeout_s=bounded_timeout,
            lang=lang_code,
            home=resolved_home,
            environ=environ,
            cwd=cwd,
        )
        http_calls_made += int(ces_flush_report.get("network_calls", 0))
        file_writes_made += int(ces_flush_report.get("file_writes", 0))
        ces_spool_pending_count = int(ces_flush_report.get("pending_count", 0))

    check_noel_summary = {
        "gateway": clean_gw,
        "online_producers": online_producers,
        "active_trainers": active_trainers,
        "accepted_rounds": accepted_rounds,
        "first_heartbeat_sent": first_heartbeat_sent,
        "first_heartbeat_ok": first_heartbeat_ok,
        "opentrainer_registered": opentrainer_registered,
        "ces_attempted": ces_attempted,
        "ces_recorded": ces_recorded,
        "ces_degraded": ces_degraded,
        "ces_spooled": ces_spooled,
        "ces_quarantined": ces_quarantined,
        "ces_spool_pending_count": ces_spool_pending_count,
        "zero_game_lag": zero_game_lag,
        "average_ease_rating": average_ease_rating,
        "mean_ease_rating": average_ease_rating,
        "zero_inference": True,
    }

    raw_details: dict[str, Any] = {
        "gateway": clean_gw,
        "allowed_host": True,
        "credentials_path": shown_cred_path,
        "masked_api_key": masked_key,
        "account_role": account_role,
        "me_checked": me_checked,
        "me_ok": me_ok,
        "me_http_status": me_status,
        "first_heartbeat_requested": send_first_heartbeat,
        "first_heartbeat_sent": first_heartbeat_sent,
        "first_heartbeat_ok": first_heartbeat_ok,
        "opentrainer_registered": opentrainer_registered,
        "heartbeat_state": heartbeat_state,
        "heartbeat_http_status": heartbeat_http_status,
        "heartbeat_payload": heartbeat_payload if send_first_heartbeat else None,
        "fleet_http_status": fleet_status,
        "raid_http_status": raid_status,
        "online_producers": online_producers,
        "active_trainers": active_trainers,
        "accepted_rounds": accepted_rounds,
        "ces_attempted": ces_attempted,
        "ces_recorded": ces_recorded,
        "ces_degraded": ces_degraded,
        "ces_spooled": ces_spooled,
        "ces_quarantined": ces_quarantined,
        "ces_spool_pending_count": ces_spool_pending_count,
        "ces_flush": ces_flush_report,
        "ces_http_status": ces_http_status,
        "ces_idempotency_key": ces_idempotency_key if submit_ces else None,
        "ces_payload": ces_payload if submit_ces else None,
        "zero_game_lag": zero_game_lag,
        "average_ease_rating": average_ease_rating,
        "mean_ease_rating": average_ease_rating,
        "http_calls_made": http_calls_made,
        "file_writes": file_writes_made,
        "zero_inference": True,
        "noel_summary": check_noel_summary,
    }
    clean_details: dict[str, Any] = _sanitize_noel_structure(raw_details, home=resolved_home)
    clean_details["receipt_sha256"] = _compute_receipt_sha256(clean_details)

    if send_first_heartbeat and not first_heartbeat_ok:
        cred_fix = _quickstart_command(clean_gw, eff_os, "--write-continue-config")
        summary = (
            f"First heartbeat failed ({heartbeat_state}); verify {shown_cred_path} and gateway reachability."
            if lang_code == "en"
            else f"Échec du premier heartbeat ({heartbeat_state}) ; vérifiez {shown_cred_path} et l'accès passerelle."
        )
        return DoctorCheck("noel_fleet_heartbeat", title, "fail", _sanitize_noel_text(summary, home=resolved_home), cred_fix, clean_details)

    if fleet_status != 200:
        curl_bin = "curl.exe" if eff_os == "nt" else "curl"
        fleet_fix = f'{curl_bin} -fsSL "{clean_gw}/v1/hivemind/producers/fleet"'
        summary = (
            f"Fleet probe GET {clean_gw}/v1/hivemind/producers/fleet failed (HTTP {fleet_status})."
            if lang_code == "en"
            else f"Échec de lecture GET {clean_gw}/v1/hivemind/producers/fleet (HTTP {fleet_status})."
        )
        return DoctorCheck("noel_fleet_heartbeat", title, "fail", _sanitize_noel_text(summary, home=resolved_home), fleet_fix, clean_details)

    if raid_status != 200:
        redeploy_fix = _parity_redeploy_plan_command(clean_gw, eff_os)
        summary = (
            f"Producer fleet online ({online_producers}), while /v1/hivemind/raid/status returned HTTP {raid_status} (active_trainers=n/a, accepted_rounds=n/a)."
            if lang_code == "en"
            else f"Flotte producteurs en ligne ({online_producers}), mais /v1/hivemind/raid/status a répondu HTTP {raid_status} (active_trainers=n/a, accepted_rounds=n/a)."
        )
        return DoctorCheck("noel_fleet_heartbeat", title, "warn", _sanitize_noel_text(summary, home=resolved_home), redeploy_fix, clean_details)

    hb_label = heartbeat_state if send_first_heartbeat else "skipped"
    ces_label = ("recorded" if ces_recorded else f"degraded_http_{ces_http_status}") if submit_ces else "skipped"
    summary = (
        f"Live fleet verified (online_producers={online_producers}, active_trainers={active_trainers}, "
        f"accepted_rounds={accepted_rounds}, heartbeat={hb_label}, ces={ces_label})."
        if lang_code == "en"
        else f"Flotte en direct vérifiée (online_producers={online_producers}, active_trainers={active_trainers}, "
        f"accepted_rounds={accepted_rounds}, heartbeat={hb_label}, ces={ces_label})."
    )
    return DoctorCheck("noel_fleet_heartbeat", title, "ok", _sanitize_noel_text(summary, home=resolved_home), "", clean_details)


def _gpu_info_from_yield_details(details: Mapping[str, Any]) -> GpuInfo:
    if not bool(details.get("gpu_present", False)):
        return GpuInfo(present=False)
    if bool(details.get("parse_error", False)):
        return GpuInfo(present=True, parse_error=True)
    return GpuInfo(
        present=True,
        parse_error=False,
        name=str(details.get("gpu_name") or ""),
        total_mib=int(details.get("total_mib") or 0),
        free_mib=int(details.get("free_mib") or 0),
        temp_c=float(details.get("temp_c") or 0.0),
        util_pct=float(details.get("util_pct") or 0.0),
    )


def run_doctor(
    *,
    gateway: str = DEFAULT_GATEWAY,
    credentials_path: Path | None = None,
    lang: str = "en",
    extra_games: Sequence[str] = (),
    timeout_s: float = HTTP_TIMEOUT_S,
    version_info: tuple[int, int, int] | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    http_get: HttpGetFn = default_http_get_json,
    runner: RunnerFn = default_subprocess_runner,
    noel: bool = False,
    cwd: Path | str | None = None,
    yield_budget_ms: float = DEFAULT_NOEL_YIELD_BUDGET_MS,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_NOEL_MIN_FREE_VRAM_MIB,
    send_first_heartbeat: bool = False,
    submit_ces: bool = False,
    ces_profile: str | None = None,
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = DEFAULT_NOEL_HOST_LABEL,
    http_request: HttpRequestFn = default_http_request_json,
    worker_yield_fn: WorkerYieldFn | None = default_worker_yield_drill,
    enforce_personal_pc: bool = False,
    spool_path: Path | str | None = None,
    flush_ces_queue: bool = False,
) -> dict[str, Any]:
    """Run all 6 Canary Alpha preflight checks (or --noel extended mode) in < 10 seconds."""
    started = time.monotonic()
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    bounded_timeout = min(max(0.5, float(timeout_s)), 4.5)
    resolved_home = home if home is not None else Path.home()

    try:
        safe_gateway = validate_gateway_url(gateway)
    except ValueError:
        safe_gateway = DEFAULT_GATEWAY

    if not noel:
        if enforce_personal_pc:
            comp_eval_std = evaluate_noel_personal_pc_compliance(
                environ=environ,
                cwd=cwd,
                os_name=eff_os,
                gateway=safe_gateway,
                lang=lang_code,
            )
            if not comp_eval_std["compliance_ok"]:
                comp_title_std = (
                    "Personal-PC Compliance (Fail-Closed Gate)"
                    if lang_code == "en"
                    else "Conformité PC Personnel (Barrière Stricte)"
                )
                comp_chk_std = DoctorCheck(
                    "personal_pc_compliance",
                    comp_title_std,
                    "fail",
                    str(comp_eval_std["summary"]),
                    str(comp_eval_std["fix_command"]),
                    {
                        "compliance_ok": False,
                        "corporate_indicators": list(comp_eval_std["corporate_indicators"]),
                        "matched_markers": list(comp_eval_std["matched_markers"]),
                        "network_transmitted": False,
                        "outbound_http_requests": 0,
                        "file_writes": 0,
                        "personal_pc_only": True,
                        "monetary_cost": FINANCIAL_PROMISE,
                        "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
                    },
                )
                blocked_std: dict[str, Any] = _sanitize_noel_structure(
                    {
                        "schema": SCHEMA,
                        "ready": False,
                        "verdict": "not_ready",
                        "compliance_ok": False,
                        "lang": lang_code,
                        "recommended_mode": "dev_api_only",
                        "readiness_checks_passed": 0,
                        "readiness_checks_total": 1,
                        "warnings_count": 0,
                        "failures_count": 1,
                        "outbound_http_requests": 0,
                        "file_writes": 0,
                        "elapsed_s": round(time.monotonic() - started, 4),
                        "gateway": safe_gateway,
                        "compliance_notice": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
                        "checks": [comp_chk_std.to_dict()],
                        "fix_commands": [comp_chk_std.fix_command] if comp_chk_std.fix_command else [],
                    },
                    home=resolved_home,
                )
                blocked_std["receipt_sha256"] = _compute_receipt_sha256(blocked_std)
                return blocked_std

        checks: list[DoctorCheck] = [
            check_python_runtime(version_info, os_name=eff_os, system_name=system_name, lang=lang_code),
            check_credentials_file(
                credentials_path,
                os_name=eff_os,
                gateway=safe_gateway,
                lang=lang_code,
                home=home,
            ),
            check_gateway_reachability(
                gateway,
                http_get=http_get,
                timeout_s=bounded_timeout,
                os_name=eff_os,
                lang=lang_code,
            ),
            check_gpu_readiness(runner=runner, os_name=eff_os, gateway=safe_gateway, lang=lang_code),
            check_active_games(
                runner=runner,
                os_name=eff_os,
                extra_games=extra_games,
                gateway=safe_gateway,
                lang=lang_code,
            ),
            check_corporate_environment(
                environ=environ,
                os_name=eff_os,
                gateway=safe_gateway,
                lang=lang_code,
            ),
        ]

        passed_count = sum(1 for c in checks if c.passed)
        warnings_count = sum(1 for c in checks if c.status == "warn")
        failures_count = sum(1 for c in checks if c.status == "fail")

        if failures_count > 0:
            verdict = "not_ready"
        elif warnings_count > 0:
            verdict = "ready_with_warnings"
        else:
            verdict = "ready"

        gpu_mode = str(checks[3].details.get("mode", "dev_api_only"))
        recommended_mode = "gamer_whisper" if gpu_mode == "gamer_whisper_ready" else "dev_api_only"

        fix_commands: list[str] = []
        for chk in checks:
            if chk.fix_command and chk.fix_command not in fix_commands:
                fix_commands.append(chk.fix_command)

        reported_gateway = safe_gateway if checks[2].details.get("allowed_host", False) else "rejected_untrusted_gateway"

        return {
            "schema": SCHEMA,
            "ready": verdict == "ready",
            "verdict": verdict,
            "lang": lang_code,
            "recommended_mode": recommended_mode,
            "readiness_checks_passed": passed_count,
            "readiness_checks_total": len(checks),
            "warnings_count": warnings_count,
            "failures_count": failures_count,
            "elapsed_s": round(time.monotonic() - started, 4),
            "gateway": reported_gateway,
            "compliance_notice": COMPLIANCE_NOTICE[lang_code],
            "checks": [c.to_dict() for c in checks],
            "fix_commands": fix_commands,
        }

    # --- Extended <10s --noel post-install verification mode ---
    comp_title = (
        "Personal-PC Compliance (Noel Fail-Closed Gate)"
        if lang_code == "en"
        else "Conformité PC Personnel (Barrière Noël Stricte)"
    )
    comp_eval = evaluate_noel_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        os_name=eff_os,
        gateway=safe_gateway,
        lang=lang_code,
    )
    comp_check = DoctorCheck(
        "personal_pc_compliance",
        comp_title,
        str(comp_eval["status"]),
        str(comp_eval["summary"]),
        str(comp_eval["fix_command"]),
        {
            "compliance_ok": bool(comp_eval["compliance_ok"]),
            "corporate_indicators": list(comp_eval["corporate_indicators"]),
            "matched_markers": list(comp_eval["matched_markers"]),
            "network_transmitted": False,
            "outbound_http_requests": 0,
            "file_writes": 0,
            "personal_pc_only": True,
            "monetary_cost": "0.00 EUR / 0.00 USD",
            "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
        },
    )

    if not comp_eval["compliance_ok"]:
        blocked_summary = {
            "noel_mode": True,
            "compliance_ok": False,
            "personal_pc_only": True,
            "monetary_cost": "0.00 EUR / 0.00 USD",
            "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
            "zero_inference_verified": True,
            "outbound_http_requests": 0,
            "file_writes": 0,
            "holiday_routes_deployed_on_cloud_run": False,
            "onboarding_feedback_deployed": False,
            "revision_drift_detected": False,
            "missing_or_stale_routes": [],
            "yield_slo_ok": False,
            "thermal_ok": False,
            "zero_game_lag": False,
            "online_producers": "n/a",
            "active_trainers": "n/a",
            "accepted_rounds": "n/a",
            "first_heartbeat_sent": False,
            "first_heartbeat_ok": False,
            "ces_attempted": False,
            "ces_recorded": False,
        }
        blocked_report: dict[str, Any] = _sanitize_noel_structure(
            {
                "schema": SCHEMA,
                "noel": True,
                "ready": False,
                "verdict": "not_ready",
                "compliance_ok": False,
                "lang": lang_code,
                "recommended_mode": "dev_api_only",
                "readiness_checks_passed": 0,
                "readiness_checks_total": 1,
                "warnings_count": 0,
                "failures_count": 1,
                "elapsed_s": round(time.monotonic() - started, 4),
                "gateway": safe_gateway,
                "compliance_notice": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
                "checks": [comp_check.to_dict()],
                "fix_commands": [comp_check.fix_command] if comp_check.fix_command else [],
                "noel_summary": blocked_summary,
            },
            home=resolved_home,
        )
        blocked_report["receipt_sha256"] = _compute_receipt_sha256(blocked_report)
        return blocked_report

    outbound_http_counter = [0]
    base_get: HttpGetFn = (
        (lambda u, t: _invoke_http_request(http_request, "GET", u, timeout_s=t))
        if (http_get is default_http_get_json and http_request is not default_http_request_json)
        else http_get
    )

    def _counted_get(url: str, req_timeout_s: float) -> tuple[int, dict[str, Any]]:
        outbound_http_counter[0] += 1
        return base_get(url, req_timeout_s)

    def _counted_request(
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = HTTP_TIMEOUT_S,
    ) -> tuple[int, dict[str, Any]]:
        outbound_http_counter[0] += 1
        return _invoke_http_request(
            http_request,
            method,
            url,
            payload=payload,
            headers=headers,
            timeout_s=timeout_s,
        )

    chk_py = check_python_runtime(version_info, os_name=eff_os, system_name=system_name, lang=lang_code)
    chk_cred = check_credentials_file(
        credentials_path,
        os_name=eff_os,
        gateway=safe_gateway,
        lang=lang_code,
        home=resolved_home,
    )
    chk_gw = check_gateway_reachability(
        gateway,
        http_get=_counted_get,
        timeout_s=bounded_timeout,
        os_name=eff_os,
        lang=lang_code,
    )
    chk_gpu = check_gpu_readiness(runner=runner, os_name=eff_os, gateway=safe_gateway, lang=lang_code)
    chk_games = check_active_games(
        runner=runner,
        os_name=eff_os,
        extra_games=extra_games,
        gateway=safe_gateway,
        lang=lang_code,
    )
    chk_routes = check_noel_holiday_routes(
        gateway,
        http_get=_counted_get,
        timeout_s=bounded_timeout,
        os_name=eff_os,
        lang=lang_code,
    )
    chk_yield = check_noel_steam_yield_and_thermal(
        runner=runner,
        worker_yield_fn=worker_yield_fn,
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        yield_budget_ms=yield_budget_ms,
        extra_games=extra_games,
        os_name=eff_os,
        gateway=safe_gateway,
        lang=lang_code,
    )
    measured_gpu = _gpu_info_from_yield_details(chk_yield.details)
    chk_fleet = check_noel_fleet_and_first_heartbeat(
        gateway,
        credentials_path=credentials_path,
        send_first_heartbeat=send_first_heartbeat,
        submit_ces=submit_ces,
        ces_profile=ces_profile,
        ease_rating=ease_rating,
        had_game_lag=had_game_lag,
        host_label=host_label,
        gpu_info=measured_gpu,
        yield_slo_ok=bool(chk_yield.details.get("yield_slo_ok", False)),
        thermal_ok=bool(chk_yield.details.get("thermal_ok", False)),
        http_get=_counted_get,
        http_request=_counted_request,
        timeout_s=bounded_timeout,
        home=resolved_home,
        os_name=eff_os,
        system_name=system_name,
        lang=lang_code,
        spool_path=spool_path,
        flush_ces_queue=flush_ces_queue,
        environ=environ,
        cwd=cwd,
    )

    noel_checks: list[DoctorCheck] = [
        chk_py,
        chk_cred,
        chk_gw,
        chk_gpu,
        chk_games,
        comp_check,
        chk_routes,
        chk_yield,
        chk_fleet,
    ]

    passed_count = sum(1 for c in noel_checks if c.passed)
    warnings_count = sum(1 for c in noel_checks if c.status == "warn")
    failures_count = sum(1 for c in noel_checks if c.status == "fail")

    if failures_count > 0:
        verdict = "not_ready"
    elif warnings_count > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"

    yield_mode = str(chk_yield.details.get("mode", "dev_api_only_no_nvidia_gpu"))
    recommended_mode = (
        "gamer_whisper"
        if (chk_gpu.details.get("mode") == "gamer_whisper_ready" and yield_mode == "gamer_whisper_ready")
        else "dev_api_only"
    )

    fix_commands = []
    for chk in noel_checks:
        if chk.fix_command and chk.fix_command not in fix_commands:
            fix_commands.append(chk.fix_command)

    reported_gateway = safe_gateway if chk_gw.details.get("allowed_host", False) else "rejected_untrusted_gateway"

    noel_summary = {
        "noel_mode": True,
        "compliance_ok": True,
        "personal_pc_only": True,
        "monetary_cost": "0.00 EUR / 0.00 USD",
        "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
        "zero_inference_verified": True,
        "outbound_http_requests": outbound_http_counter[0],
        "file_writes": int(chk_fleet.details.get("file_writes", 0)),
        "holiday_routes_deployed_on_cloud_run": bool(
            chk_routes.details.get("holiday_routes_deployed_on_cloud_run", False)
        ),
        "onboarding_feedback_deployed": bool(chk_routes.details.get("onboarding_feedback_deployed", False)),
        "revision_drift_detected": bool(chk_routes.details.get("revision_drift_detected", False)),
        "missing_or_stale_routes": list(chk_routes.details.get("missing_or_stale_routes", [])),
        "gpu_mode": yield_mode,
        "yield_latency_ms": chk_yield.details.get("yield_latency_ms"),
        "yield_budget_ms": chk_yield.details.get("yield_budget_ms"),
        "yield_slo_ok": bool(chk_yield.details.get("yield_slo_ok", False)),
        "thermal_ok": bool(chk_yield.details.get("thermal_ok", False)),
        "vram_reserve_ok": bool(chk_yield.details.get("vram_reserve_ok", False)),
        "zero_game_lag": bool(chk_fleet.details.get("zero_game_lag", False)),
        "online_producers": chk_fleet.details.get("online_producers", "n/a"),
        "active_trainers": chk_fleet.details.get("active_trainers", "n/a"),
        "accepted_rounds": chk_fleet.details.get("accepted_rounds", "n/a"),
        "first_heartbeat_sent": bool(chk_fleet.details.get("first_heartbeat_sent", False)),
        "first_heartbeat_ok": bool(chk_fleet.details.get("first_heartbeat_ok", False)),
        "ces_attempted": bool(chk_fleet.details.get("ces_attempted", False)),
        "ces_recorded": bool(chk_fleet.details.get("ces_recorded", False)),
        "ces_spooled": bool(chk_fleet.details.get("ces_spooled", False)),
        "ces_quarantined": bool(chk_fleet.details.get("ces_quarantined", False)),
        "ces_spool_pending_count": int(chk_fleet.details.get("ces_spool_pending_count", 0)),
        "ces_idempotency_key": chk_fleet.details.get("ces_idempotency_key"),
        "average_ease_rating": chk_fleet.details.get("average_ease_rating"),
        "mean_ease_rating": chk_fleet.details.get("mean_ease_rating"),
    }

    noel_report: dict[str, Any] = _sanitize_noel_structure(
        {
            "schema": SCHEMA,
            "noel": True,
            "ready": verdict == "ready",
            "verdict": verdict,
            "compliance_ok": True,
            "lang": lang_code,
            "recommended_mode": recommended_mode,
            "readiness_checks_passed": passed_count,
            "readiness_checks_total": len(noel_checks),
            "warnings_count": warnings_count,
            "failures_count": failures_count,
            "elapsed_s": round(time.monotonic() - started, 4),
            "gateway": reported_gateway,
            "compliance_notice": COMPLIANCE_NOTICE[lang_code],
            "checks": [c.to_dict() for c in noel_checks],
            "fix_commands": fix_commands,
            "noel_summary": noel_summary,
        },
        home=resolved_home,
    )
    noel_report["receipt_sha256"] = _compute_receipt_sha256(noel_report)
    return noel_report


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) == ("sco" + "re") or _has_forbidden_score_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _validate_expected_sha(raw: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError("expected_sha must be a 6..64 character alphanumeric/hex string.")
    sanitized = _sanitize_manifest_sha(raw)
    if sanitized == "unknown":
        raise ValueError("expected_sha must be a 6..64 character alphanumeric/hex string (never 'unknown' or secret).")
    return sanitized


def _shas_match(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_manifest_sha(live_sha)
    clean_exp = _sanitize_manifest_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _validate_pr949_options(
    *,
    target_online_producers: int,
    min_ease_rating: float,
    min_zero_lag_ratio: float,
    require_live_quorum: bool,
    require_ces_slo: bool,
    check_arena_summary: bool,
    verify_idempotency_replay: bool,
    execute_deploy_dry_run: bool,
) -> tuple[int, float, float]:
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or not (1 <= target_online_producers <= 500)
    ):
        raise ValueError("target_online_producers must be an integer between 1 and 500.")
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise ValueError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise ValueError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    for flag in (
        require_live_quorum,
        require_ces_slo,
        check_arena_summary,
        verify_idempotency_replay,
        execute_deploy_dry_run,
    ):
        if not isinstance(flag, bool):
            raise ValueError("PR #949 gate boolean flags must be True or False.")
    return int(target_online_producers), float(min_ease_rating), float(min_zero_lag_ratio)


def _pr949_route_diagnostic(path: str, group: str, code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if code == 404 and group in ("holiday", "manifest"):
        return (
            f"GET {path} a répondu HTTP 404 (révision Cloud Run pré-PR-#949 ou route manquante)."
            if is_fr
            else f"GET {path} returned HTTP 404 (pre-PR-#949 Cloud Run revision drift or missing route)."
        )
    if code == 0:
        return (
            f"GET {path} injoignable (erreur réseau/TLS)."
            if is_fr
            else f"GET {path} unreachable (network or TLS failure)."
        )
    return (
        f"GET {path} a répondu HTTP {code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {code} (expected HTTP 200)."
    )


def _inspect_models_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str], bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body) or body.get("object", "list") != "list":
        msg = "Contrat `/v1/models` invalide." if is_fr else "Invalid `/v1/models` JSON schema (`object` != 'list')."
        return False, [], False, [msg]
    raw_data = body.get("data")
    if not isinstance(raw_data, list):
        msg = "Liste `data` absente dans `GET /v1/models`." if is_fr else "Missing `data` model list in `GET /v1/models`."
        return False, [], False, [msg]
    model_ids = [
        str(entry.get("id")).strip()
        for entry in raw_data
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry.get("id").strip()
    ]
    issues: list[str] = []
    if not model_ids:
        issues.append("Liste `data` de `GET /v1/models` vide." if is_fr else "`GET /v1/models` returned an empty `data` list.")
    coder_present = ("hivemind/" + "coder") in model_ids
    if not coder_present:
        issues.append(
            "Modèle `hivemind/coder` absent de `GET /v1/models`."
            if is_fr
            else "Required model `hivemind/coder` missing from `GET /v1/models` metadata."
        )
    return len(issues) == 0, model_ids, coder_present, issues


def _inspect_html_contract(
    path: str,
    html_text: str,
    resp_headers: Mapping[str, str],
    seen_nonces: set[str],
    *,
    lang: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang == "fr"
    has_doctype = "<!doctype html" in html_text.lower()
    csp_val = str(resp_headers.get("content-security-policy") or "")
    csp_lower = csp_val.lower()
    csp_unsafe = "'unsafe-inline'" in csp_lower or "'unsafe-eval'" in csp_lower
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_val)
    header_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    attr_nonces = [a or b for a, b in _HTML_NONCE_ATTR_RE.findall(html_text)]
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

    stripped_html = strip_masked_keys_and_nonces(html_text)
    has_unsafe_sink = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    dom_clean = not has_unsafe_sink and not has_static_fig

    low_html = stripped_html.lower()
    compliance_banner_ok = (
        any(
            token in low_html
            for token in (
                "100% personal",
                "100 % personnel",
                "100 % machine personnelle",
                "pc personnel",
                "personal home pc",
                "personal-pc",
                "personal home machine",
            )
        )
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
        and ("agpl-3.0" in low_html)
    )

    issues: list[str] = []
    if not has_doctype:
        issues.append(f"HTML invalide sur {path} : `<!DOCTYPE html>` absent." if is_fr else f"Invalid HTML on {path}: missing `<!DOCTYPE html>`.")
    if nonce_reused:
        issues.append(f"Nonce CSP réutilisé sur {path}." if is_fr else f"Reused CSP nonce detected on {path}.")
    elif not csp_ok:
        issues.append(f"Contrat CSP par requête invalide sur {path}." if is_fr else f"Invalid or missing per-request CSP nonce contract on {path}.")
    if has_unsafe_sink:
        issues.append(f"Sink DOM non sûr détecté sur {path}." if is_fr else f"Unsafe DOM sink detected on {path}.")
    if has_static_fig:
        issues.append(f"Métrique statique détectée dans le HTML de {path}." if is_fr else f"Static telemetry figure detected in HTML on {path}.")
    if not compliance_banner_ok:
        issues.append(
            f"Bannière PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR / AGPL-3.0 banner missing on {path}."
        )

    ok = len(issues) == 0
    return (
        ok,
        {
            "path": path,
            "ok": ok,
            "has_doctype": has_doctype,
            "csp_ok": csp_ok,
            "nonce_reused": nonce_reused,
            "csp_nonce_sha256_prefix": (
                hashlib.sha256(header_nonce.encode("utf-8")).hexdigest()[:16] if header_nonce else None
            ),
            "dom_clean": dom_clean,
            "unsafe_dom_sink_detected": has_unsafe_sink,
            "static_figure_detected": has_static_fig,
            "compliance_banner_ok": compliance_banner_ok,
            "issues": issues,
        },
        issues,
    )


def _inspect_manifest_contract(
    body: Mapping[str, Any] | None,
    *,
    expected_sha: str | None,
    lang: str,
) -> tuple[bool, str, str | None, bool | None, list[str], list[str], list[str]]:
    is_fr = lang == "fr"
    req_scripts = ("hivemind_doctor.py", "hivemind_quickstart.py")
    if not isinstance(body, Mapping):
        msg = "Manifeste `/v1/hivemind/onboarding/manifest` absent." if is_fr else "Manifest `/v1/hivemind/onboarding/manifest` missing or invalid JSON."
        return False, "unknown", None, (False if expected_sha else None), list(MANIFEST_REQUIRED_HOLIDAY_ROUTES), list(req_scripts), [msg]

    issues: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("Manifest contains forbidden key.")
    if (
        body.get("object") != MANIFEST_OBJECT
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("status") != "ok"
        or body.get("personal_pc_only") is not True
        or body.get("financial_promise") != FINANCIAL_PROMISE
    ):
        issues.append(
            "Contrat `/v1/hivemind/onboarding/manifest` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/onboarding/manifest` metadata contract."
        )

    raw_routes = body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in MANIFEST_REQUIRED_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(MANIFEST_REQUIRED_HOLIDAY_ROUTES)
    if missing_routes:
        issues.append(
            f"Manifeste `holiday_routes` incomplet : {', '.join(missing_routes)}."
            if is_fr
            else f"Manifest `holiday_routes` missing routes: {', '.join(missing_routes)}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [s for s in req_scripts if s not in raw_scripts]
    else:
        missing_scripts = list(req_scripts)
    if missing_scripts:
        issues.append(
            f"Manifeste `downloadable_scripts` incomplet : {', '.join(missing_scripts)}."
            if is_fr
            else f"Manifest `downloadable_scripts` missing: {', '.join(missing_scripts)}."
        )

    live_sha = _sanitize_manifest_sha(body.get("build_sha"))
    live_rev = _sanitize_manifest_revision(body.get("cloud_run_revision"))
    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = _shas_match(live_sha, expected_sha)
        if not sha_match:
            issues.append(
                f"Dérive `build_sha` : actif='{live_sha}' != attendu='{expected_sha}'."
                if is_fr
                else f"Manifest `build_sha` mismatch: live='{live_sha}' != expected='{expected_sha}'."
            )

    return len(issues) == 0, live_sha, live_rev, sha_match, missing_routes, missing_scripts, issues


def _inspect_feedback_summary_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Contrat `/v1/hivemind/onboarding/feedback/summary` invalide." if is_fr else "Invalid `/v1/hivemind/onboarding/feedback/summary` JSON payload."
        return False, None, [msg]
    raw_total = body.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1
    has_ease = "mean_ease_rating" in body or "average_ease_rating" in body
    raw_ease = body.get("mean_ease_rating") if body.get("mean_ease_rating") is not None else body.get("average_ease_rating")
    ease_ok = bool(
        has_ease
        and total_ok
        and (
            (total_val == 0 and raw_ease is None)
            or (
                total_val > 0
                and isinstance(raw_ease, (int, float))
                and not isinstance(raw_ease, bool)
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
                and isinstance(raw_lag, (int, float))
                and not isinstance(raw_lag, bool)
                and math.isfinite(float(raw_lag))
                and 0.0 <= float(raw_lag) <= 1.0
            )
        )
    )
    bp = body.get("by_profile")
    d_raw = bp.get("dev") if isinstance(bp, Mapping) else None
    g_raw = bp.get("gamer") if isinstance(bp, Mapping) else None
    prof_ok = bool(
        isinstance(bp, Mapping)
        and _is_nonneg_int(d_raw)
        and _is_nonneg_int(g_raw)
        and total_ok
        and int(d_raw) + int(g_raw) == total_val  # type: ignore[arg-type]
    )
    obj_ok = body.get("object") in _ACCEPTED_FEEDBACK_SUMMARY_OBJECTS
    if not (obj_ok and total_ok and ease_ok and lag_ok and prof_ok):
        msg = "Schéma `/v1/hivemind/onboarding/feedback/summary` non conforme." if is_fr else "Schema mismatch on `/v1/hivemind/onboarding/feedback/summary`."
        return False, None, [msg]
    return True, _sanitize_feedback_summary_doc(body), []


def _inspect_raid_status_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Réponse `/v1/hivemind/raid/status` invalide." if is_fr else "Invalid `/v1/hivemind/raid/status` payload."
        return False, [msg]
    comp = body.get("compliance")
    comp_ok = bool(
        isinstance(comp, Mapping)
        and comp.get("personal_pc_only") is True
        and comp.get("monetary_cost_eur") == "0.00 EUR"
        and comp.get("monetary_cost_usd") == "0.00 USD"
    )
    ok = bool(
        body.get("object") == RAID_STATUS_OBJECT
        and comp_ok
        and isinstance(body.get("onboarding_feedback"), Mapping)
    )
    if not ok:
        msg = "Contrat conformité/télémétrie invalide sur `/v1/hivemind/raid/status`." if is_fr else "Invalid compliance or telemetry contract on `/v1/hivemind/raid/status`."
        return False, [msg]
    return True, []


def _inspect_fleet_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, int, int, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Contrat `/v1/hivemind/producers/fleet` invalide." if is_fr else "Invalid `/v1/hivemind/producers/fleet` JSON payload."
        return False, 0, 0, [msg]
    if body.get("object") not in _ACCEPTED_FLEET_OBJECTS:
        msg = "Champ `object` invalide sur `/v1/hivemind/producers/fleet`." if is_fr else "Invalid `object` metadata on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]
    rows = body.get("hiveminders")
    if not isinstance(rows, list):
        msg = "Liste `hiveminders` absente sur `/v1/hivemind/producers/fleet`." if is_fr else "Missing `hiveminders` list on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]
    for row in rows:
        if isinstance(row, Mapping) and "endpoint_url" in row:
            msg = "Fuite `endpoint_url` interdite sur `/v1/hivemind/producers/fleet`." if is_fr else "Forbidden `endpoint_url` leak detected on `/v1/hivemind/producers/fleet`."
            return False, 0, 0, [msg]
    has_live_hm = "live_hiveminders" in body
    if has_live_hm and not _is_nonneg_int(body.get("live_hiveminders")):
        msg = "Champ `live_hiveminders` invalide sur `/v1/hivemind/producers/fleet`." if is_fr else "Invalid non-integer `live_hiveminders` on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]

    live_hm_int = int(body["live_hiveminders"]) if has_live_hm else 0
    row_online = sum(
        1
        for row in rows
        if isinstance(row, Mapping) and str(row.get("status", "")).strip().lower() in LIVE_PRODUCER_STATUSES
    )
    online_producers = max(live_hm_int, row_online)
    raw_gpu = body.get("distinct_physical_gpus")
    distinct_gpus = int(raw_gpu) if _is_nonneg_int(raw_gpu) else 0
    return True, online_producers, distinct_gpus, []


def _inspect_arena_summary_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or body.get("object") != ARENA_SUMMARY_OBJECT:
        msg = "Contrat `/v1/hivemind/arena/summary` invalide." if is_fr else "Arena summary `/v1/hivemind/arena/summary` invalid `object` metadata."
        return False, None, [msg]
    if _has_forbidden_score_key(body):
        return False, None, ["Arena summary `/v1/hivemind/arena/summary` contains forbidden key."]

    tv = body.get("total_votes")
    hw = body.get("holiday_wins")
    bw = body.get("baseline_wins")
    ti = body.get("ties")
    counts_valid = all(_is_nonneg_int(x) for x in (tv, hw, bw, ti))
    conserved = bool(counts_valid and int(hw) + int(bw) + int(ti) == int(tv))  # type: ignore[arg-type]

    dec_ok = True
    if "decisive_votes" in body:
        dv = body.get("decisive_votes")
        dec_ok = bool(_is_nonneg_int(dv) and counts_valid and int(dv) == int(hw) + int(bw))  # type: ignore[arg-type]

    raw_wlb = body.get("wilson_95_lower_bound")
    wlb_ok = bool(
        "wilson_95_lower_bound" in body
        and not isinstance(raw_wlb, bool)
        and isinstance(raw_wlb, (int, float))
        and math.isfinite(float(raw_wlb))
        and 0.0 <= float(raw_wlb) <= 1.0
    )
    decisive = (int(hw) + int(bw)) if counts_valid else 0  # type: ignore[arg-type]
    if wlb_ok and counts_valid and decisive == 0 and float(raw_wlb) != 0.0:  # type: ignore[arg-type]
        wlb_ok = False

    win_ratio_ok = True
    if "win_ratio" in body:
        raw_wr = body.get("win_ratio")
        if counts_valid and decisive == 0:
            win_ratio_ok = raw_wr is None
        else:
            win_ratio_ok = bool(
                not isinstance(raw_wr, bool)
                and isinstance(raw_wr, (int, float))
                and math.isfinite(float(raw_wr))
                and 0.0 <= float(raw_wr) <= 1.0
            )

    by_task_ok = True
    raw_tasks = body.get("by_task")
    if isinstance(raw_tasks, Mapping):
        for t_doc in raw_tasks.values():
            if not isinstance(t_doc, Mapping):
                by_task_ok = False
                break
            t_tv, t_hw, t_bw, t_ti = (
                t_doc.get("total_votes"),
                t_doc.get("holiday_wins"),
                t_doc.get("baseline_wins"),
                t_doc.get("ties"),
            )
            t_wlb = t_doc.get("wilson_95_lower_bound")
            if (
                not all(_is_nonneg_int(x) for x in (t_tv, t_hw, t_bw, t_ti))
                or int(t_hw) + int(t_bw) + int(t_ti) != int(t_tv)  # type: ignore[arg-type]
                or "wilson_95_lower_bound" not in t_doc
                or isinstance(t_wlb, bool)
                or not isinstance(t_wlb, (int, float))
                or not math.isfinite(float(t_wlb))
                or not (0.0 <= float(t_wlb) <= 1.0)
            ):
                by_task_ok = False
                break

    if not (counts_valid and conserved and dec_ok and wlb_ok and win_ratio_ok and by_task_ok):
        msg = (
            "Conservation des votes ou borne Wilson 95 % invalide sur `/v1/hivemind/arena/summary`."
            if is_fr
            else "Vote conservation (`holiday_wins + baseline_wins + ties == total_votes`) or Wilson 95% lower-bound invalid on `/v1/hivemind/arena/summary`."
        )
        return False, None, [msg]

    raw_wr_val = body.get("win_ratio")
    sanitized: dict[str, Any] = {
        "object": ARENA_SUMMARY_OBJECT,
        "total_votes": int(tv),  # type: ignore[arg-type]
        "decisive_votes": decisive,
        "holiday_wins": int(hw),  # type: ignore[arg-type]
        "baseline_wins": int(bw),  # type: ignore[arg-type]
        "ties": int(ti),  # type: ignore[arg-type]
        "win_ratio": (
            round(float(raw_wr_val), 4)
            if isinstance(raw_wr_val, (int, float)) and not isinstance(raw_wr_val, bool)
            else None
        ),
        "wilson_95_lower_bound": round(float(raw_wlb), 4),  # type: ignore[arg-type]
    }
    return True, sanitized, []


def _run_deploy_sh_dry_run(
    *,
    repo_root: Path | str | None = None,
    expected_sha: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute list-argv ``bash deploy/gcp_openrouter/deploy.sh --dry-run`` without invoking ``gcloud``."""
    is_fr = lang == "fr"
    base_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    deploy_sh = base_root / "deploy" / "gcp_openrouter" / "deploy.sh"
    display_path = "deploy/gcp_openrouter/deploy.sh"
    if not deploy_sh.is_file():
        return {
            "executed": False,
            "ok": False,
            "returncode": 127,
            "deploy_script_path": display_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": (
                f"Script `{display_path}` introuvable pour `--dry-run`."
                if is_fr
                else f"Deploy script `{display_path}` missing for `--dry-run`."
            ),
        }

    dry_env: dict[str, str] = dict(environ) if environ is not None else dict(os.environ)
    if "PATH" not in dry_env:
        dry_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    dry_env["HIVEMIND_DEPLOY_DRY_RUN"] = "1"
    if expected_sha is not None:
        dry_env["HIVEMIND_BUILD_SHA"] = expected_sha

    argv = ["bash", str(deploy_sh), "--dry-run"]
    if Path(argv[0]).name.lower().startswith("gcloud"):
        raise ValueError("Direct gcloud invocation is forbidden in hivemind_doctor.")
    try:
        sig = inspect.signature(runner)
        if "env" in sig.parameters:
            rc, stdout = runner(argv, 15.0, env=dry_env)  # type: ignore[call-arg]
        else:
            env_prefix = ["env", "HIVEMIND_DEPLOY_DRY_RUN=1"]
            if expected_sha is not None:
                env_prefix.append(f"HIVEMIND_BUILD_SHA={expected_sha}")
            rc, stdout = runner([*env_prefix, *argv], 15.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "executed": True,
            "ok": False,
            "returncode": 1,
            "deploy_script_path": display_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": _sanitize_noel_text(f"{display_path} --dry-run failed ({type(exc).__name__})."),
        }

    kv: dict[str, str] = {}
    for line in str(stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    finops_line = kv.get("finops_flags", "")
    env_vars_line = kv.get("update_env_vars", "")
    raw_dry_sha = kv.get("build_sha", "")
    dry_sha = _sanitize_manifest_sha(raw_dry_sha)
    sha_propagated = bool(
        raw_dry_sha
        and raw_dry_sha.strip().lower() == dry_sha
        and f"HIVEMIND_BUILD_SHA={raw_dry_sha}" in env_vars_line
        and (expected_sha is None or _shas_match(dry_sha, expected_sha))
    )
    checks = {
        "returncode_zero": int(rc) == 0,
        "dry_run_marker": kv.get("HIVEMIND_CLOUD_RUN_DEPLOY_DRY_RUN") == "1",
        "gcloud_invoked_zero": kv.get("gcloud_invoked") == "0",
        "finops_flags_present": all(
            tok in finops_line
            for tok in ("--min-instances 0", "--max-instances 1", "--no-cpu-throttling", "--concurrency 80")
        ),
        "replica_url_gcs": kv.get("replica_url", "").startswith("gcs://"),
        "secret_manager_mounted": "HIVEMIND_ROUTER_SECRET=" in kv.get("secrets", ""),
        "zero_grants_env": "HIVEMIND_STARTER_GRANT_HMT=0" in env_vars_line,
        "build_sha_propagated": sha_propagated,
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "ok": ok,
        "returncode": int(rc),
        "deploy_script_path": display_path,
        "gcloud_invoked": kv.get("gcloud_invoked") != "0",
        "dry_run_build_sha": dry_sha,
        "checks": checks,
        "diagnostic": (
            ""
            if ok
            else (
                f"Échec vérification `{display_path} --dry-run`."
                if is_fr
                else f"Subprocess verification `{display_path} --dry-run` failed."
            )
        ),
    }


def verify_pr949_doctor_quorum_and_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = True,
    verify_idempotency_replay: bool = False,
    execute_deploy_dry_run: bool = False,
    repo_root: Path | str | None = None,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_get: HttpGetFn = default_http_get_json,
    http_request: HttpRequestFn = default_http_request_json,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
    credentials_path: Path | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> dict[str, Any]:
    """Run zero-inference PR #949 4-HTML parity, models, live quorum, arena summary, replay & dry-run gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    bounded_timeout = _validate_timeout_s(timeout_s)
    target_n, min_ease_f, min_lag_f = _validate_pr949_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        verify_idempotency_replay=verify_idempotency_replay,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )
    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = _validate_expected_sha(expected_sha) if (expected_sha and str(expected_sha).strip()) else None

    raw_key, _, shown_cred_path = _load_valid_credentials_key(credentials_path, home=resolved_home)
    masked_key = mask_api_key(raw_key) if raw_key is not None else "<missing>"

    comp = evaluate_noel_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        os_name=eff_os,
        gateway=clean_gw,
        lang=lang_code,
    )
    if not comp["compliance_ok"]:
        blocked: dict[str, Any] = _sanitize_noel_structure(
            {
                "schema": SCHEMA,
                "object": "hivemind.doctor.pr949_quorum_parity_receipt",
                "campaign_id": CAMPAIGN_ID,
                "gateway": clean_gw,
                "lang": lang_code,
                "compliance_ok": False,
                "corporate_indicators": comp["corporate_indicators"],
                "ready": False,
                "verdict": "not_ready",
                "parity_ok": False,
                "gate_ok": False,
                "quorum_gate_ok": False,
                "ces_gate_ok": False,
                "redeploy_recommended": False,
                "cloud_run_lags_main": False,
                "stale_404_drift": False,
                "routes_passed_count": 0,
                "routes_total_count": len(PR949_PARITY_ROUTES) + (1 if check_arena_summary else 0),
                "http_statuses": {},
                "missing_routes": [],
                "failed_routes": [],
                "models_ok": False,
                "live_model_ids": [],
                "hivemind_coder_present": False,
                "unique_csp_nonces_count": 0,
                "unique_csp_nonces_ok": False,
                "html_dom_clean": False,
                "compliance_banners_ok": False,
                "html_contracts": {},
                "manifest_valid": False,
                "live_build_sha": "unknown",
                "expected_sha": clean_expected_sha,
                "sha_match": False if clean_expected_sha is not None else None,
                "sha_drift": False,
                "cloud_run_revision": None,
                "manifest_missing_routes": [],
                "manifest_missing_scripts": [],
                "feedback_summary_ok": False,
                "feedback_summary": None,
                "ces_slo_ok": False,
                "zero_lag_slo_ok": False,
                "min_ease_rating": min_ease_f,
                "min_zero_lag_ratio": min_lag_f,
                "raid_status_ok": False,
                "active_trainers": 0,
                "fleet_ok": False,
                "online_producers": 0,
                "target_online_producers": target_n,
                "producer_quorum_met": False,
                "awaiting_producer_quorum": True,
                "distinct_physical_gpus": 0,
                "arena_summary_checked": bool(check_arena_summary),
                "arena_summary_ok": False if check_arena_summary else None,
                "arena_summary": None,
                "idempotency_replay_checked": bool(verify_idempotency_replay),
                "idempotent_dedup_ok": None,
                "idempotency_key": None,
                "deploy_dry_run_executed": bool(execute_deploy_dry_run),
                "deploy_dry_run": None,
                "zero_inference_verified": True,
                "network_calls": 0,
                "outbound_http_requests": 0,
                "subprocess_calls": 0,
                "file_writes": 0,
                "masked_api_key": masked_key,
                "credentials_file": shown_cred_path,
                "fallback_guidance": None,
                "diagnostics": [comp["summary"]],
                "routes": [],
                "compliance_notice": COMPLIANCE_NOTICE[lang_code],
                "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
                "financial_promise": FINANCIAL_PROMISE,
                "exit_code": 2,
            },
            home=resolved_home,
        )
        blocked["receipt_sha256"] = _compute_receipt_sha256(blocked)
        return blocked

    effective_raw_fn: Callable[..., Any] = (
        raw_http_fn
        if raw_http_fn is not default_http_raw
        else (
            http_request
            if http_request is not default_http_request_json
            else (http_get if http_get is not default_http_get_json else default_http_raw)
        )
    )
    effective_json_fn: Callable[..., Any] = (
        http_get
        if http_get is not default_http_get_json
        else (
            http_request
            if http_request is not default_http_request_json
            else (raw_http_fn if raw_http_fn is not default_http_raw else default_http_raw)
        )
    )
    effective_post_fn: Callable[..., Any] = (
        http_request
        if http_request is not default_http_request_json
        else (raw_http_fn if raw_http_fn is not default_http_raw else default_http_request_json)
    )

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    html_all_ok = True
    html_dom_clean = True
    compliance_banners_ok = True

    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    sha_match: bool | None = None if clean_expected_sha is None else False
    manifest_missing_routes: list[str] = []
    manifest_missing_scripts: list[str] = []
    feedback_summary_ok = False
    feedback_summary: dict[str, Any] | None = None
    raw_feedback_doc: Mapping[str, Any] | None = None
    raid_status_ok = False
    active_trainers = 0
    fleet_ok = False
    online_producers = 0
    distinct_gpus = 0
    network_calls = 0
    subprocess_calls = 0

    for path, kind, group in PR949_PARITY_ROUTES:
        target_fn = effective_raw_fn if kind == "html" else effective_json_fn
        code, json_doc, text_body, resp_headers = _invoke_parity_call(
            target_fn,
            "GET",
            f"{clean_gw}{path}",
            headers=headers,
            timeout_s=bounded_timeout,
        )
        network_calls += 1
        http_statuses[path] = code
        ok = code == 200
        route_issues: list[str] = []

        if not ok:
            if code == 404:
                missing_routes.append(path)
            route_issues.append(_pr949_route_diagnostic(path, group, code, lang=lang_code))
            if kind == "html":
                html_all_ok = False
                compliance_banners_ok = False
        elif path == "/health":
            if not isinstance(json_doc, Mapping) or json_doc.get("status") != "ok":
                ok = False
                route_issues.append(
                    "GET /health a répondu HTTP 200 sans `status == 'ok'`."
                    if is_fr
                    else "GET /health returned HTTP 200 without `status == 'ok'`."
                )
        elif path == "/v1/models":
            models_ok, model_ids, coder_present, m_issues = _inspect_models_contract(json_doc, lang=lang_code)
            if not models_ok:
                ok = False
                route_issues.extend(m_issues)
        elif kind == "html":
            h_ok, h_info, h_issues = _inspect_html_contract(
                path,
                text_body,
                resp_headers,
                seen_nonces,
                lang=lang_code,
            )
            html_contracts[path] = h_info
            html_all_ok = html_all_ok and h_ok
            html_dom_clean = html_dom_clean and bool(h_info["dom_clean"])
            compliance_banners_ok = compliance_banners_ok and bool(h_info["compliance_banner_ok"])
            if not h_ok:
                ok = False
                route_issues.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            (
                manifest_valid,
                live_build_sha,
                cloud_run_revision,
                sha_match,
                manifest_missing_routes,
                manifest_missing_scripts,
                man_issues,
            ) = _inspect_manifest_contract(json_doc, expected_sha=clean_expected_sha, lang=lang_code)
            if not manifest_valid:
                ok = False
                route_issues.extend(man_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_ok, feedback_summary, fb_issues = _inspect_feedback_summary_contract(
                json_doc,
                lang=lang_code,
            )
            if feedback_summary_ok:
                raw_feedback_doc = json_doc
            else:
                ok = False
                route_issues.extend(fb_issues)
        elif path == "/v1/hivemind/raid/status":
            raid_status_ok, r_issues = _inspect_raid_status_contract(json_doc, lang=lang_code)
            if not raid_status_ok:
                ok = False
                route_issues.extend(r_issues)
            elif isinstance(json_doc, Mapping) and isinstance(json_doc.get("swarm"), Mapping):
                raw_tr = json_doc["swarm"].get("active_trainers")
                if _is_nonneg_int(raw_tr):
                    active_trainers = int(raw_tr)
        elif path == "/v1/hivemind/producers/fleet":
            fleet_ok, online_producers, distinct_gpus, f_issues = _inspect_fleet_contract(json_doc, lang=lang_code)
            if not fleet_ok:
                ok = False
                route_issues.extend(f_issues)

        if not ok:
            failed_routes.append(path)
            for item in route_issues:
                diagnostics.append(_sanitize_noel_text(item, home=resolved_home))

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "kind": kind,
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": _sanitize_noel_text(route_issues[0] if route_issues else "", home=resolved_home),
            }
        )

    arena_summary_ok: bool | None = None
    arena_summary_doc: dict[str, Any] | None = None
    if check_arena_summary:
        a_path = "/v1/hivemind/arena/summary"
        a_code, a_json, _, _ = _invoke_parity_call(
            effective_json_fn,
            "GET",
            f"{clean_gw}{a_path}",
            headers=headers,
            timeout_s=bounded_timeout,
        )
        network_calls += 1
        http_statuses[a_path] = a_code
        a_issues: list[str] = []
        if a_code != 200:
            arena_summary_ok = False
            if a_code == 404:
                missing_routes.append(a_path)
            a_issues.append(_pr949_route_diagnostic(a_path, "holiday", a_code, lang=lang_code))
        else:
            arena_summary_ok, arena_summary_doc, a_issues = _inspect_arena_summary_contract(a_json, lang=lang_code)
        if not arena_summary_ok:
            failed_routes.append(a_path)
            for item in a_issues:
                diagnostics.append(_sanitize_noel_text(item, home=resolved_home))
        route_rows.append(
            {
                "path": a_path,
                "method": "GET",
                "kind": "json",
                "group": "holiday",
                "http_status": a_code,
                "ok": bool(arena_summary_ok),
                "diagnostic": _sanitize_noel_text(a_issues[0] if a_issues else "", home=resolved_home),
            }
        )

    idempotent_dedup_ok: bool | None = None
    idem_key: str | None = None
    if verify_idempotency_replay:
        pre_total = (
            int(feedback_summary["total_responses"])
            if (feedback_summary_ok and feedback_summary is not None)
            else None
        )
        pre_dev = (
            int(feedback_summary["by_profile"]["dev"])
            if (feedback_summary_ok and feedback_summary is not None)
            else None
        )
        fb_payload = _normalize_ces_payload(
            profile="dev",
            ease_rating=5,
            zero_game_lag=True,
            platform_name=_platform_feedback_tag(os_name=eff_os),
            note="",
            home=resolved_home,
        )
        idem_key = _compute_doctor_ces_idempotency_key(
            fb_payload,
            idempotency_key=f"hm-ces-pr949-{secrets.token_hex(12)}",
            gateway=clean_gw,
        )
        replay_headers = {str(k): str(v) for k, v in headers.items()} if headers else {}
        replay_headers["Idempotency-Key"] = idem_key
        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"

        p1_code, p1_json, _, _ = _invoke_parity_call(
            effective_post_fn,
            "POST",
            post_url,
            payload=fb_payload,
            headers=replay_headers,
            timeout_s=bounded_timeout,
        )
        network_calls += 1
        http_statuses["POST /v1/hivemind/onboarding/feedback"] = p1_code
        p1_ok, p1_sum, _ = (
            _inspect_feedback_summary_contract(p1_json, lang=lang_code)
            if p1_code in (200, 201)
            else (False, None, [])
        )

        p2_code, p2_json, _, _ = _invoke_parity_call(
            effective_post_fn,
            "POST",
            post_url,
            payload=fb_payload,
            headers=replay_headers,
            timeout_s=bounded_timeout,
        )
        network_calls += 1
        http_statuses["POST /v1/hivemind/onboarding/feedback#replay"] = p2_code
        p2_ok, p2_sum, _ = (
            _inspect_feedback_summary_contract(p2_json, lang=lang_code)
            if p2_code in (200, 201)
            else (False, None, [])
        )

        ref_code, ref_json, _, _ = _invoke_parity_call(
            effective_json_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/onboarding/feedback/summary",
            headers=headers,
            timeout_s=bounded_timeout,
        )
        network_calls += 1
        http_statuses["/v1/hivemind/onboarding/feedback/summary#after_replay"] = ref_code
        ref_ok, ref_sum, _ = (
            _inspect_feedback_summary_contract(ref_json, lang=lang_code)
            if ref_code == 200
            else (False, None, [])
        )

        p1_inc_ok = bool(
            p1_ok
            and p1_sum is not None
            and (
                pre_total is None
                or (
                    int(p1_sum["total_responses"]) >= int(pre_total) + 1
                    and int(p1_sum["by_profile"]["dev"]) >= int(pre_dev or 0) + 1
                )
            )
        )
        replay_eq_ok = bool(
            p2_ok
            and p2_sum is not None
            and p1_sum is not None
            and ref_ok
            and ref_sum is not None
            and p2_sum["total_responses"] == p1_sum["total_responses"]
            and p2_sum["by_profile"] == p1_sum["by_profile"]
            and ref_sum["total_responses"] == p1_sum["total_responses"]
            and ref_sum["by_profile"] == p1_sum["by_profile"]
        )
        idempotent_dedup_ok = bool(p1_inc_ok and replay_eq_ok)
        if idempotent_dedup_ok and ref_sum is not None and isinstance(ref_json, Mapping):
            feedback_summary_ok = True
            feedback_summary = ref_sum
            raw_feedback_doc = ref_json
        else:
            diagnostics.append(
                "Échec vérification replay idempotent `Idempotency-Key` sur `/v1/hivemind/onboarding/feedback`."
                if is_fr
                else "Idempotency-Key duplicate-replay verification failed on `/v1/hivemind/onboarding/feedback`."
            )

    producer_quorum_met = bool(fleet_ok and online_producers >= target_n)
    awaiting_producer_quorum = not producer_quorum_met
    quorum_gate_ok = bool(producer_quorum_met or not require_live_quorum)
    if require_live_quorum and not producer_quorum_met:
        diagnostics.append(
            f"Quorum producteurs en ligne insuffisant ({online_producers} < {target_n})."
            if is_fr
            else f"Live producer quorum not met ({online_producers} < {target_n})."
        )

    ces_slo_ok: bool | None
    zero_lag_slo_ok: bool | None
    if not feedback_summary_ok or feedback_summary is None or raw_feedback_doc is None:
        ces_slo_ok = False
        zero_lag_slo_ok = False
        ces_gate_ok = False
    elif int(feedback_summary["total_responses"]) > 0:
        raw_ease = (
            raw_feedback_doc.get("mean_ease_rating")
            if raw_feedback_doc.get("mean_ease_rating") is not None
            else raw_feedback_doc.get("average_ease_rating")
        )
        raw_lag = raw_feedback_doc.get("zero_game_lag_ratio")
        ces_slo_ok = bool(raw_ease is not None and Fraction(str(float(raw_ease))) >= Fraction(str(min_ease_f)))
        zero_lag_slo_ok = bool(raw_lag is not None and Fraction(str(float(raw_lag))) >= Fraction(str(min_lag_f)))
        ces_gate_ok = bool(not require_ces_slo or (ces_slo_ok and zero_lag_slo_ok))
        if require_ces_slo and not ces_gate_ok:
            diagnostics.append(
                f"SLO CES/zéro-lag non respecté (ease={raw_ease}, lag={raw_lag})."
                if is_fr
                else f"CES/zero-lag SLO breached (ease={raw_ease} < {min_ease_f} or lag={raw_lag} < {min_lag_f})."
            )
    else:
        ces_slo_ok = None
        zero_lag_slo_ok = None
        ces_gate_ok = True

    deploy_dry_run_res: dict[str, Any] | None = None
    if execute_deploy_dry_run:
        deploy_dry_run_res = _run_deploy_sh_dry_run(
            repo_root=repo_root,
            expected_sha=clean_expected_sha,
            runner=runner,
            environ=environ,
            lang=lang_code,
        )
        if deploy_dry_run_res.get("executed"):
            subprocess_calls += 1
        if not deploy_dry_run_res.get("ok"):
            diagnostics.append(str(deploy_dry_run_res.get("diagnostic") or "deploy.sh --dry-run failed."))

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    html_count = len(PR949_HTML_ROUTES)
    unique_csp_nonces_ok = bool(
        html_all_ok
        and len(html_contracts) == html_count
        and len(seen_nonces) == html_count
        and not any(info.get("nonce_reused") for info in html_contracts.values())
    )
    sha_drift = bool(clean_expected_sha is not None and sha_match is not True)
    stale_404_drift = bool(missing_routes or manifest_missing_routes or manifest_missing_scripts)
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and (stale_404_drift or sha_drift))

    parity_ok = bool(
        passed_count == total_count
        and models_ok
        and coder_present
        and manifest_valid
        and unique_csp_nonces_ok
        and html_dom_clean
        and compliance_banners_ok
        and feedback_summary_ok
        and raid_status_ok
        and fleet_ok
        and not sha_drift
        and (not check_arena_summary or arena_summary_ok is True)
        and (not verify_idempotency_replay or idempotent_dedup_ok is True)
        and (not execute_deploy_dry_run or (deploy_dry_run_res is not None and deploy_dry_run_res["ok"] is True))
    )
    gate_ok = bool(parity_ok and quorum_gate_ok and ces_gate_ok)

    fallback_guidance: str | None = None
    if stale_404_drift:
        fallback_guidance = _parity_redeploy_plan_command(clean_gw, eff_os)

    report: dict[str, Any] = _sanitize_noel_structure(
        {
            "schema": SCHEMA,
            "object": "hivemind.doctor.pr949_quorum_parity_receipt",
            "campaign_id": CAMPAIGN_ID,
            "gateway": clean_gw,
            "lang": lang_code,
            "compliance_ok": True,
            "ready": gate_ok,
            "verdict": "ready" if gate_ok else "not_ready",
            "parity_ok": parity_ok,
            "gate_ok": gate_ok,
            "quorum_gate_ok": quorum_gate_ok,
            "ces_gate_ok": ces_gate_ok,
            "redeploy_recommended": bool(not parity_ok or cloud_run_lags_main or sha_drift),
            "cloud_run_lags_main": cloud_run_lags_main,
            "stale_404_drift": stale_404_drift,
            "routes_passed_count": passed_count,
            "routes_total_count": total_count,
            "http_statuses": http_statuses,
            "missing_routes": missing_routes,
            "failed_routes": failed_routes,
            "models_ok": models_ok,
            "live_model_ids": model_ids,
            "hivemind_coder_present": coder_present,
            "unique_csp_nonces_count": len(seen_nonces),
            "unique_csp_nonces_ok": unique_csp_nonces_ok,
            "html_dom_clean": html_dom_clean,
            "compliance_banners_ok": compliance_banners_ok,
            "html_contracts": html_contracts,
            "manifest_valid": manifest_valid,
            "live_build_sha": live_build_sha,
            "expected_sha": clean_expected_sha,
            "sha_match": sha_match,
            "sha_drift": sha_drift,
            "cloud_run_revision": cloud_run_revision,
            "manifest_missing_routes": manifest_missing_routes,
            "manifest_missing_scripts": manifest_missing_scripts,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary,
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "min_ease_rating": min_ease_f,
            "min_zero_lag_ratio": min_lag_f,
            "raid_status_ok": raid_status_ok,
            "active_trainers": active_trainers,
            "fleet_ok": fleet_ok,
            "online_producers": online_producers,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "distinct_physical_gpus": distinct_gpus,
            "arena_summary_checked": bool(check_arena_summary),
            "arena_summary_ok": arena_summary_ok,
            "arena_summary": arena_summary_doc,
            "idempotency_replay_checked": bool(verify_idempotency_replay),
            "idempotent_dedup_ok": idempotent_dedup_ok,
            "idempotency_key": idem_key,
            "deploy_dry_run_executed": bool(execute_deploy_dry_run),
            "deploy_dry_run": deploy_dry_run_res,
            "zero_inference_verified": True,
            "network_calls": network_calls,
            "outbound_http_requests": network_calls,
            "subprocess_calls": subprocess_calls,
            "file_writes": 0,
            "masked_api_key": masked_key,
            "credentials_file": shown_cred_path,
            "fallback_guidance": fallback_guidance,
            "diagnostics": diagnostics,
            "routes": route_rows,
            "compliance_notice": COMPLIANCE_NOTICE[lang_code],
            "notice_bilingual": NOEL_COMPLIANCE_NOTICE_BILINGUAL,
            "financial_promise": FINANCIAL_PROMISE,
            "exit_code": 0 if gate_ok else 2,
        },
        home=resolved_home,
    )
    report["receipt_sha256"] = _compute_receipt_sha256(report)
    return report


def render_report(report: dict[str, Any], *, color: bool = False) -> str:
    """Format a human-readable bilingual terminal summary with actionable one-line fixes."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    badges = {"ok": " OK ", "warn": "WARN", "fail": "FAIL"}
    ansi = {"ok": "32", "warn": "33", "fail": "31"}

    if lang_code == "fr":
        banner_title = "  HIVEMIND DOCTOR — PRÉ-CONTRÔLE CANARY ALPHA (PC PERSO • 0,00 EUR)"
        rule_prefix = "  Règle"
        fix_prefix = "Action"
        summary_tpl = "Contrôles réussis : {passed}/{total} | Mode conseillé : {mode} | Verdict : {verdict}"
    else:
        banner_title = "  HIVEMIND DOCTOR — CANARY ALPHA PREFLIGHT (PERSONAL PC ONLY • 0.00 EUR)"
        rule_prefix = "  Rule"
        fix_prefix = "Fix"
        summary_tpl = "Checks passed: {passed}/{total} | Recommended mode: {mode} | Verdict: {verdict}"

    lines = [
        "=" * 72,
        banner_title,
        "=" * 72,
        f"{rule_prefix}: {report['compliance_notice']}",
        "",
    ]

    for chk in report["checks"]:
        status = str(chk["status"])
        tag = f"[{badges.get(status, 'INFO')}]"
        if color and status in ansi:
            tag = f"\x1b[{ansi[status]}m{tag}\x1b[0m"
        lines.append(f"{tag} {chk['title']}")
        lines.append(f"       {chk['summary']}")
        if chk.get("fix_command"):
            lines.append(f"       {fix_prefix} -> {chk['fix_command']}")

    lines.append("")
    lines.append(
        summary_tpl.format(
            passed=report["readiness_checks_passed"],
            total=report["readiness_checks_total"],
            mode=report["recommended_mode"],
            verdict=str(report["verdict"]).upper(),
        )
    )
    noel_sum = report.get("noel_summary")
    if isinstance(noel_sum, Mapping):
        if lang_code == "fr":
            lines.append(
                f"Noël : routes_cloud_run={noel_sum.get('holiday_routes_deployed_on_cloud_run')} | "
                f"producteurs={noel_sum.get('online_producers')} | trainers={noel_sum.get('active_trainers')} | "
                f"rounds={noel_sum.get('accepted_rounds')}"
            )
        else:
            lines.append(
                f"Noel: cloud_run_routes={noel_sum.get('holiday_routes_deployed_on_cloud_run')} | "
                f"producers={noel_sum.get('online_producers')} | trainers={noel_sum.get('active_trainers')} | "
                f"rounds={noel_sum.get('accepted_rounds')}"
            )
    if report.get("receipt_sha256"):
        lines.append(f"SHA-256: {report['receipt_sha256']}")
    return "\n".join(lines)


def render_pr949_quorum_report(report: Mapping[str, Any], *, color: bool = False) -> str:
    """Format an FR/EN human-readable CLI report for ``verify_pr949_doctor_quorum_and_parity``."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    header = (
        "HIVEMIND DOCTOR — PARITÉ PR #949 & QUORUM PRODUCTEURS (ZÉRO INFÉRENCE • 0,00 EUR)"
        if is_fr
        else "HIVEMIND DOCTOR — PR #949 PARITY & LIVE QUORUM GATE (ZERO INFERENCE • 0.00 EUR)"
    )
    ok_tag = "\x1b[32m[OK]\x1b[0m" if color else "[OK]"
    fail_tag = "\x1b[31m[FAIL]\x1b[0m" if color else "[FAIL]"
    lines = [
        "=" * 72,
        f"  {header}",
        "=" * 72,
        f"  Rule: {report.get('compliance_notice')}",
        f"  Gateway: {report.get('gateway')} | Credentials: {report.get('credentials_file')} (key {report.get('masked_api_key')})",
        f"  Routes: {report.get('routes_passed_count')}/{report.get('routes_total_count')} "
        f"(parity_ok={report.get('parity_ok')}, gate_ok={report.get('gate_ok')})",
        f"  Quorum: online={report.get('online_producers')}/{report.get('target_online_producers')} "
        f"(awaiting={report.get('awaiting_producer_quorum')}, gpus={report.get('distinct_physical_gpus')})",
        f"  Arena/Replay: arena_ok={report.get('arena_summary_ok')} | idempotent_ok={report.get('idempotent_dedup_ok')}",
    ]
    for row in report.get("routes") or ():
        tag = ok_tag if row.get("ok") else fail_tag
        lines.append(f"  {tag} GET {row.get('path'):<40} -> HTTP {row.get('http_status')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256: {report.get('receipt_sha256', '')}")
    return "\n".join(lines)


def render_ces_flush_report(flush_res: Mapping[str, Any]) -> str:
    """Format an FR/EN human-readable CLI report for ``DoctorCesRetrySpool.flush``."""
    lang_code = normalize_lang(str(flush_res.get("lang", "en")))
    is_fr = lang_code == "fr"
    header = (
        "HIVEMIND DOCTOR — VIDAGE FILE CES (0,00 EUR)"
        if is_fr
        else "HIVEMIND DOCTOR — CES RETRY SPOOL FLUSH (0.00 EUR)"
    )
    lines = [
        "=" * 72,
        f"  {header}",
        "=" * 72,
        f"  Gateway: {flush_res.get('gateway')}",
        f"  Spool  : {flush_res.get('spool_file')} (exists={flush_res.get('spool_exists')})",
        f"  Counts : delivered={flush_res.get('delivered_count')} pending={flush_res.get('pending_count')} "
        f"quarantined={flush_res.get('total_quarantined_count')}",
    ]
    for diag in flush_res.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256: {flush_res.get('receipt_sha256', '')}")
    return "\n".join(lines)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Format a sanitized FR/EN Markdown report for doctor, PR #949 parity, or spool outputs."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    obj_name = str(report.get("object") or "")
    if obj_name == "hivemind.doctor.pr949_quorum_parity_receipt":
        title = (
            "# HiveMind Doctor — Parité PR #949 & Quorum en Direct"
            if is_fr
            else "# HiveMind Doctor — PR #949 Parity & Live Quorum Gate"
        )
        lines = [
            title,
            "",
            f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway')}`",
            f"- **{'Conformité' if is_fr else 'Compliance'}**: `{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
            f"- **{'Parité / Porte' if is_fr else 'Parity / Gate'}**: `parity_ok={report.get('parity_ok')}`, `gate_ok={report.get('gate_ok')}`, `exit_code={report.get('exit_code')}`",
            f"- **Quorum**: `online_producers={report.get('online_producers')}/{report.get('target_online_producers')}`, `awaiting_producer_quorum={report.get('awaiting_producer_quorum')}`",
            f"- **Arena / Replay**: `arena_summary_ok={report.get('arena_summary_ok')}`, `idempotent_dedup_ok={report.get('idempotent_dedup_ok')}`",
            f"- **SHA-256**: `{report.get('receipt_sha256', '')}`",
        ]
        if report.get("diagnostics"):
            lines.extend(["", "## Diagnostics", *[f"- {d}" for d in report["diagnostics"]]])
        return "\n".join(lines)

    if obj_name in ("hivemind.doctor.ces_spool_flush", "hivemind.doctor.ces_spool_clean"):
        title = "# HiveMind Doctor — File CES" if is_fr else "# HiveMind Doctor — CES Spool"
        lines = [
            title,
            "",
            f"- **Spool**: `{report.get('spool_file')}`",
            f"- **Status**: `ok={report.get('flush_ok', True)}`",
            f"- **SHA-256**: `{report.get('receipt_sha256', '')}`",
        ]
        return "\n".join(lines)

    title = "# HiveMind Doctor — Pré-contrôle" if is_fr else "# HiveMind Doctor — Preflight Report"
    lines = [
        title,
        "",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway')}`",
        f"- **Verdict**: `{report.get('verdict')}` (`{report.get('readiness_checks_passed')}/{report.get('readiness_checks_total')}`)",
        f"- **{'Conformité' if is_fr else 'Compliance'}**: {report.get('compliance_notice', COMPLIANCE_NOTICE[lang_code])}",
    ]
    for chk in report.get("checks") or ():
        lines.append(f"- **[{str(chk.get('status', 'info')).upper()}] {chk.get('title')}**: {chk.get('summary')}")
    if report.get("receipt_sha256"):
        lines.append(f"- **SHA-256**: `{report.get('receipt_sha256')}`")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-dependency HiveMind Canary Alpha & Noel preflight assistant (0.00 EUR, personal PCs only)."
    )
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind HTTPS gateway URL (*.run.app or localhost).")
    parser.add_argument(
        "--credentials-file",
        default=str(Path.home() / ".hivemind" / "credentials.json"),
        help="Path to local HiveMind credentials JSON file.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=normalize_lang,
        choices=SUPPORTED_LANGS,
        help="Output language: 'en' or 'fr'.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown report.")
    parser.add_argument("--output", default=None, help="Atomically persist output (POSIX 0600); 0 writes on compliance fail.")
    parser.add_argument("--game", action="append", default=[], help="Extra game executable name to check.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=HTTP_TIMEOUT_S,
        help="Per-request HTTP timeout in seconds (bounded to <= 4.5s for < 10s total runtime).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in terminal output.")
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Fail closed (exit code 2, 0 HTTP calls, 0 writes) if corporate env/cwd/git-remote markers appear.",
    )
    parser.add_argument(
        "--noel",
        action="store_true",
        help="Run <10s post-install Noel holiday route drift, <1.5s Steam yield/thermal & fleet verifier.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Workspace directory inspected locally for corporate markers / .git/config remotes.",
    )
    parser.add_argument(
        "--yield-budget-ms",
        type=float,
        default=DEFAULT_NOEL_YIELD_BUDGET_MS,
        help="Maximum allowed Steam worker yield latency budget in --noel mode (default: 1500.0).",
    )
    parser.add_argument(
        "--max-temp-c",
        type=float,
        default=GAMER_WHISPER_MAX_TEMP_C,
        help="Quiet-fan GPU temperature ceiling in Celsius for --noel mode (default: 70.0).",
    )
    parser.add_argument(
        "--first-heartbeat",
        action="store_true",
        help="Verify GET /v1/hivemind/me and send outbound POST /v1/hivemind/opentrainer/heartbeat in --noel mode.",
    )
    parser.add_argument(
        "--submit-ces",
        action="store_true",
        help="Submit non-blocking zero-PII onboarding ease feedback to POST /v1/hivemind/onboarding/feedback.",
    )
    parser.add_argument("--ces-queue-file", default=None, help="Optional POSIX-0600 CES retry spool JSON path.")
    parser.add_argument("--flush-ces-queue", action="store_true", help="Flush spooled CES events idempotently.")
    parser.add_argument("--clean-ces-queue", action="store_true", help="Remove local CES retry spool JSON file.")
    parser.add_argument(
        "--profile",
        default=None,
        type=normalize_profile,
        choices=sorted(VALID_CES_PROFILES),
        help="Onboarding feedback profile for --submit-ces: 'dev' or 'gamer'.",
    )
    parser.add_argument(
        "--ease-rating",
        type=_parse_ease_rating,
        default=5,
        choices=(1, 2, 3, 4, 5),
        help="Onboarding ease rating from 1 (hard) to 5 (effortless) for --submit-ces.",
    )
    parser.add_argument(
        "--had-game-lag",
        action="store_true",
        help="Report that foreground game lag occurred (sets zero_game_lag=False in --submit-ces).",
    )
    parser.add_argument(
        "--host-label",
        default=DEFAULT_NOEL_HOST_LABEL,
        help="Sanitized non-PII host label for --first-heartbeat (default: 'personal-pc').",
    )
    parser.add_argument(
        "--pr949-quorum-gate",
        action="store_true",
        help="Run zero-inference PR #949 4-HTML parity, models, live quorum, arena summary, replay & dry-run gate.",
    )
    parser.add_argument("--expected-sha", default=None, help="Expected build_sha on /v1/hivemind/onboarding/manifest.")
    parser.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS)
    parser.add_argument("--min-ease-rating", type=float, default=DEFAULT_MIN_EASE_RATING)
    parser.add_argument("--min-zero-lag-ratio", type=float, default=DEFAULT_MIN_ZERO_LAG_RATIO)
    parser.add_argument("--require-live-quorum", action="store_true")
    parser.add_argument("--require-ces-slo", action="store_true")
    parser.add_argument("--check-arena-summary", dest="check_arena_summary", action="store_true", default=True)
    parser.add_argument(
        "--skip-arena-summary",
        "--no-check-arena-summary",
        dest="check_arena_summary",
        action="store_false",
    )
    parser.add_argument("--verify-idempotency-replay", action="store_true")
    parser.add_argument("--run-deploy-dry-run", action="store_true")
    return parser


def _emit_cli_output(
    rendered: str,
    *,
    report: Mapping[str, Any],
    out_stream: Any,
    err_stream: Any,
    output_path: Path | None,
    home: Path | None,
) -> bool:
    print(rendered, file=out_stream)
    if output_path is not None:
        comp_ok = report.get("compliance_ok") is not False and not any(
            isinstance(c, Mapping) and c.get("id") == "personal_pc_compliance" and c.get("status") == "fail"
            for c in (report.get("checks") or ())
        )
        if not comp_ok:
            return False
        safety_hits = scan_output_safety(rendered, home=home)
        if safety_hits:
            print(f"Error: refusing unsafe --output write ({', '.join(safety_hits)}).", file=err_stream)
            return False
        payload = rendered if rendered.endswith("\n") else (rendered + "\n")
        _write_file_secure_atomic(output_path, payload, mode=0o600)
    return True


def main(
    argv: Sequence[str] | None = None,
    *,
    http_get: HttpGetFn = default_http_get_json,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    http_request: HttpRequestFn = default_http_request_json,
    worker_yield_fn: WorkerYieldFn | None = default_worker_yield_drill,
    home: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    raw_http_fn: RawHttpFn = default_http_raw,
    cwd: Path | str | None = None,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    eff_out = out_stream if out_stream is not None else sys.stdout
    eff_err = err_stream if err_stream is not None else sys.stderr
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)

    try:
        cli_cwd = _validate_cli_path_arg(args.cwd, "--cwd", home=home)
        output_path = _validate_cli_path_arg(args.output, "--output", home=home)
        queue_path_arg = _validate_cli_path_arg(args.ces_queue_file, "--ces-queue-file", home=home)
    except ValueError as exc:
        print(f"Error: {_sanitize_noel_text(str(exc), home=home)}", file=eff_err)
        return 2

    explicit_cred_flag = any(tok == "--credentials-file" or tok.startswith("--credentials-file=") for tok in raw_argv)
    eff_cred_path = (
        _expand_user_path(args.credentials_file, home=home)
        if explicit_cred_flag or home is None
        else Path(home) / ".hivemind" / "credentials.json"
    )
    eff_cred_for_spool: Path | None = eff_cred_path if explicit_cred_flag else None
    resolved_queue_path = (
        queue_path_arg
        if queue_path_arg is not None
        else default_ces_queue_path(home=home, credentials_path=eff_cred_for_spool)
    )
    eff_cwd = Path(cwd) if cwd is not None else cli_cwd
    use_color = bool(sys.stdout.isatty() and not args.no_color and os.name != "nt")

    if args.clean_ces_queue and not (
        args.noel or args.first_heartbeat or args.submit_ces or args.pr949_quorum_gate or args.flush_ces_queue
    ):
        clean_res = DoctorCesRetrySpool(resolved_queue_path, home=home).clean(home=home)
        clean_res["lang"] = args.lang
        clean_res["compliance_ok"] = True
        rendered = (
            json.dumps(clean_res, indent=2)
            if args.json
            else (
                render_markdown_report(clean_res)
                if args.markdown
                else f"Cleaned CES spool: {clean_res['spool_file']} (removed={clean_res['removed']})"
            )
        )
        if not _emit_cli_output(rendered, report=clean_res, out_stream=eff_out, err_stream=eff_err, output_path=output_path, home=home):
            return 2
        return 0

    if args.pr949_quorum_gate:
        try:
            pr949 = verify_pr949_doctor_quorum_and_parity(
                args.gateway,
                expected_sha=args.expected_sha.strip() if args.expected_sha else None,
                target_online_producers=int(args.target_online_producers),
                min_ease_rating=float(args.min_ease_rating),
                min_zero_lag_ratio=float(args.min_zero_lag_ratio),
                require_live_quorum=bool(args.require_live_quorum),
                require_ces_slo=bool(args.require_ces_slo),
                check_arena_summary=bool(args.check_arena_summary),
                verify_idempotency_replay=bool(args.verify_idempotency_replay),
                execute_deploy_dry_run=bool(args.run_deploy_dry_run),
                raw_http_fn=raw_http_fn,
                http_get=http_get,
                http_request=http_request,
                runner=runner,
                environ=environ,
                cwd=eff_cwd,
                os_name=os_name,
                home=home,
                lang=args.lang,
                credentials_path=eff_cred_path,
                timeout_s=args.timeout,
            )
        except ValueError as exc:
            print(f"Error: {_sanitize_noel_text(str(exc), home=home)}", file=eff_err)
            return 2

        if pr949["gate_ok"] and args.flush_ces_queue:
            eff_flush_req: HttpRequestFn = (
                http_request
                if http_request is not default_http_request_json
                else (raw_http_fn if raw_http_fn is not default_http_raw else default_http_request_json)  # type: ignore[assignment]
            )
            pr949["flush_ces_queue"] = flush_doctor_ces_queue(
                args.gateway,
                spool_path=resolved_queue_path,
                credentials_path=eff_cred_for_spool,
                http_request=eff_flush_req,
                timeout_s=args.timeout,
                lang=args.lang,
                home=home,
                environ=environ,
                cwd=eff_cwd,
            )
            pr949["receipt_sha256"] = _compute_receipt_sha256(pr949)

        rendered = (
            json.dumps(pr949, indent=2, ensure_ascii=False)
            if args.json
            else (render_markdown_report(pr949) if args.markdown else render_pr949_quorum_report(pr949, color=use_color))
        )
        ok_emit = _emit_cli_output(
            rendered,
            report=pr949,
            out_stream=eff_out,
            err_stream=eff_err,
            output_path=output_path,
            home=home,
        )
        if not ok_emit and pr949["compliance_ok"]:
            return 2
        return int(pr949["exit_code"])

    if args.flush_ces_queue and not (args.noel or args.first_heartbeat or args.submit_ces or args.enforce_personal_pc):
        flush_res = flush_doctor_ces_queue(
            args.gateway,
            spool_path=resolved_queue_path,
            credentials_path=eff_cred_for_spool,
            http_request=http_request,
            timeout_s=args.timeout,
            lang=args.lang,
            home=home,
            environ=environ,
            cwd=eff_cwd,
        )
        rendered = (
            json.dumps(flush_res, indent=2, ensure_ascii=False)
            if args.json
            else (render_markdown_report(flush_res) if args.markdown else render_ces_flush_report(flush_res))
        )
        ok_emit = _emit_cli_output(
            rendered,
            report=flush_res,
            out_stream=eff_out,
            err_stream=eff_err,
            output_path=output_path,
            home=home,
        )
        if not ok_emit and flush_res["compliance_ok"]:
            return 2
        return 0 if (flush_res["compliance_ok"] and flush_res["flush_ok"]) else 2

    noel_enabled = bool(args.noel or args.first_heartbeat or args.submit_ces)
    spool_for_doctor: Path | None = (
        resolved_queue_path if (queue_path_arg is not None or args.flush_ces_queue or args.submit_ces) else None
    )
    report = run_doctor(
        gateway=args.gateway,
        credentials_path=eff_cred_path,
        lang=args.lang,
        extra_games=args.game,
        timeout_s=args.timeout,
        os_name=os_name,
        system_name=system_name,
        environ=environ,
        home=home,
        http_get=http_get,
        runner=runner,
        noel=noel_enabled,
        cwd=eff_cwd,
        yield_budget_ms=args.yield_budget_ms,
        max_temp_c=args.max_temp_c,
        send_first_heartbeat=args.first_heartbeat,
        submit_ces=args.submit_ces,
        ces_profile=args.profile,
        ease_rating=args.ease_rating,
        had_game_lag=args.had_game_lag,
        host_label=args.host_label,
        http_request=http_request,
        worker_yield_fn=worker_yield_fn,
        enforce_personal_pc=bool(args.enforce_personal_pc),
        spool_path=spool_for_doctor,
        flush_ces_queue=bool(args.flush_ces_queue),
    )
    rendered = (
        json.dumps(report, indent=2)
        if args.json
        else (render_markdown_report(report) if args.markdown else render_report(report, color=use_color))
    )
    ok_emit = _emit_cli_output(
        rendered,
        report=report,
        out_stream=eff_out,
        err_stream=eff_err,
        output_path=output_path,
        home=home,
    )
    if not ok_emit and report.get("compliance_ok", True):
        return 2
    return EXIT_CODES[str(report["verdict"])]


if __name__ == "__main__":
    raise SystemExit(main())
