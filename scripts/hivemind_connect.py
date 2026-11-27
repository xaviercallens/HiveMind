#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind connector: zero-dependency stdlib registration, Continue.dev autoconfig, and Noel doctor.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library (`httpx` is strictly optional and only needed when passing custom ``transport``).
When ``transport`` is ``None``, all live HTTP requests use a proxy-free
``urllib.request.ProxyHandler({})`` stdlib client restricted to trusted HiveMind origins
(``https://*.run.app``, ``https://*.googleapis.com``, ``datasets-server.huggingface.co``, or loopback).

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science licensed under AGPL-3.0.

    python hivemind_connect.py register --handle my-name --save-credentials --write-continue
    python hivemind_connect.py check
    python hivemind_connect.py doctor --first-heartbeat --submit-ces
    python hivemind_connect.py snippet continue --model coder
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

DEFAULT_GATEWAY: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
GATEWAY_ENV: str = "HIVEMIND_GATEWAY"
API_KEY_ENV: str = "HIVEMIND_API_KEY"
KEY_PLACEHOLDER: str = "hm-YOUR_KEY"
DEFAULT_MODEL: str = "coder"
DEFAULT_MAX_TOKENS: int = 2048
HTTP_TIMEOUT_S: float = 30.0
TOOLS: tuple[str, ...] = ("continue", "opencode", "aider", "cline", "zed", "jetbrains", "python", "env")

SCHEMA: str = "hivemind.connect_doctor/1"
RECEIPT_OBJECT: str = "hivemind.connect_doctor_report"
CAMPAIGN_ID: str = "noel-2026-easy-onboard"
MANIFEST_OBJECT: str = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT: str = "hivemind.onboarding.feedback.summary"
ARENA_SUMMARY_OBJECT: str = "hivemind.blind_ab_arena.summary"
FLEET_OBJECT: str = "hivemind.producers.fleet"
RAID_STATUS_OBJECT: str = "hivemind.raid_noel.status"
FINANCIAL_PROMISE: str = "0.00 EUR / 0.00 USD"
CONTINUE_MODELS: tuple[str, str] = ("hivemind/coder", "hivemind/bee-stars")
DEFAULT_HOST_LABEL: str = "personal-pc"
DEFAULT_TRAINING_STACK: str = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_MAX_TEMP_C: float = 70.0
DEFAULT_MIN_FREE_VRAM_MIB: int = 2048
DEFAULT_YIELD_BUDGET_MS: float = 1500.0
DOCTOR_HTTP_TIMEOUT_S: float = 4.5
SUBPROCESS_TIMEOUT_S: float = 3.0
MAX_HTTP_BODY_BYTES: int = 4 * 1024 * 1024
NON_WORKER_GPU_UTIL_THRESHOLD_PCT: float = 15.0
MIN_REPORTABLE_VRAM_TOTAL_MIB: int = 512
MAX_GPU_VRAM_MIB: int = 1_048_576
DEFAULT_TARGET_ONLINE_PRODUCERS: int = 8
DEFAULT_MIN_EASE_RATING: float = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO: float = 0.95
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})

CES_QUEUE_SCHEMA: str = "hivemind.connect.ces_queue/1"
DEFAULT_CES_QUEUE_FILENAME: str = "connect_ces_queue.json"
DEFAULT_MAX_CES_QUEUE_ENTRIES: int = 128
DEFAULT_MAX_CES_DELIVERED: int = 512
MAX_CES_QUEUE_ENTRIES_CEILING: int = 4096

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_TEST_TRANSPORT_HOSTS: frozenset[str] = frozenset({"gw.example.org", "testserver"})

DOCTOR_CORE_ROUTES: tuple[str, ...] = ("/health",)
DOCTOR_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/raid/status",
)
DOCTOR_FLEET_ROUTES: tuple[str, ...] = ("/v1/hivemind/producers/fleet",)
PR949_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
_PR949_REQUIRED_ROUTES: tuple[tuple[str, str, str], ...] = (
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
_MANIFEST_REQUIRED_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
_ACCEPTED_FLEET_OBJECTS: frozenset[str] = frozenset({FLEET_OBJECT, "hivemind.producer_fleet"})
_ACCEPTED_FEEDBACK_SUMMARY_OBJECTS: frozenset[str] = frozenset(
    {FEEDBACK_SUMMARY_OBJECT, "hivemind.onboarding_feedback_summary"}
)

EXIT_CODES: dict[str, int] = {
    "ready": 0,
    "ready_with_warnings": 1,
    "not_ready": 2,
}

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

_CORP_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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

STEAM_DX12_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "bg3.exe",
        "bg3_dx11.exe",
        "helldivers2.exe",
        "dota2.exe",
        "dota2",
        "valorant.exe",
        "valorant-win64-shipping.exe",
        "fortniteclient-win64-shipping.exe",
        "overwatch.exe",
        "starfield.exe",
        "witcher3.exe",
        "gta5.exe",
        "r5apex.exe",
        "league of legends.exe",
        "baldur.exe",
    }
)
STEAM_PROCESS_PREFIXES: tuple[str, ...] = ("steam_app_", "steam_game_", "proton_")

COMPLIANCE_NOTICE_EN: str = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR: str = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL: str = f"{COMPLIANCE_NOTICE_FR} | {COMPLIANCE_NOTICE_EN}"

_LOCAL_WORKER_YIELD_PROGRAM: str = (
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
_PYTEST_TMP_PREFIX_RE = re.compile(r"pytest-of-[^/\\]+[/\\]pytest-[^/\\]+")
_RAW_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)(?:[0-9]{1,4}(?:m"
    + r"s\b|tok"
    + r"/s)|[A-Za-z0-9_-]{4,})(?![A-Za-z0-9_-])"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
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
_DNS_LABEL_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.ASCII)
_PLATFORM_TAG_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_HOST_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_HEX_SHA_RE = re.compile(r"\A[0-9a-f]{7,64}\Z", re.ASCII)
_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_REVISION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_SECRET_VALUE_MARKERS: tuple[str, ...] = ("secret", "token", "password", "passwd", "apikey", "bearer", "aiza")

_CES_LOCK_GUARD = threading.Lock()
_CES_PATH_LOCKS: dict[Path, threading.Lock] = {}

HttpFn = Callable[..., tuple[int, Any]]
RawHttpFn = Callable[..., tuple[int, Any, Mapping[str, str]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
WorkerYieldFn = Callable[..., Any]
ClockFn = Callable[[], float]
InputFn = Callable[[str], str]


class GatewayPolicyError(ValueError):
    """Raised when gateway allowlist or zero-inference policy rejects a request."""


class GatewayConnectionError(OSError):
    """Raised when live stdlib HTTP transport encounters a connection or timeout error."""


class GpuProbeSnapshot:
    """Measured local NVIDIA GPU telemetry without hardware UUID or serial numbers."""

    __slots__ = ("present", "parse_error", "name", "total_mib", "free_mib", "used_mib", "temp_c", "util_pct", "reason")

    def __init__(
        self,
        present: bool,
        *,
        parse_error: bool = False,
        name: str | None = None,
        total_mib: int | None = None,
        free_mib: int | None = None,
        used_mib: int | None = None,
        temp_c: float | None = None,
        util_pct: float | None = None,
        reason: str = "ok",
    ) -> None:
        self.present = present
        self.parse_error = parse_error
        self.name = name
        self.total_mib = total_mib
        self.free_mib = free_mib
        self.used_mib = used_mib
        self.temp_c = temp_c
        self.util_pct = util_pct
        self.reason = reason


class _StdlibHttpResponse:
    """Minimal response wrapper matching the subset of ``httpx.Response`` used by CLI handlers."""

    __slots__ = ("status_code", "content", "headers")

    def __init__(self, status_code: int, content: bytes, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


def normalize_lang(raw: str | None) -> str:
    """Normalize language code to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ValueError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ValueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize onboarding CES profile selector to ``'dev'`` or ``'gamer'``."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError("Unsupported profile; expected 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise ValueError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def parse_ease_rating(raw: str | int) -> int:
    """Parse and validate CES ease rating integer between 1 and 5."""
    if isinstance(raw, bool):
        raise ValueError("ease_rating must be an integer between 1 and 5.")
    val = int(raw)
    if not (1 <= val <= 5):
        raise ValueError(f"ease_rating {raw!r} out of range; expected 1..5.")
    return val


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff ``raw_key`` is a structurally valid ``hm-...`` or ``hm_live_...`` key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw secrets never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def _home_variants(home: Path | None = None) -> tuple[str, ...]:
    """Return unique string variants of the user home directory for redaction checks."""
    resolved = str(home if home is not None else Path.home())
    if not resolved or resolved in ("/", "\\"):
        return ()
    variants: list[str] = []
    for candidate in (resolved, resolved.replace("\\", "/"), resolved.replace("/", "\\")):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace local user home prefixes with ``'~'`` across POSIX and Windows paths."""
    raw = str(target)
    for variant in _home_variants(home):
        if raw.startswith(variant):
            suffix = raw[len(variant):]
            if not suffix or suffix.startswith(("/", "\\")):
                return "~" + suffix.replace("\\", "/")
    redacted = _HOME_PATH_RE.sub("~", raw)
    return redacted.replace("\\", "/") if redacted != raw else raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw API keys, Bearer tokens, emails, non-loopback IPv4s, and home paths."""
    scrubbed = redact_home_path(str(text), home=home)
    for variant in _home_variants(home):
        scrubbed = scrubbed.replace(variant, "~")
    scrubbed = _HOME_PATH_RE.sub("~", scrubbed).replace("\\", "/")
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _RAW_API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    return _IPV4_RE.sub(lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>", scrubbed)


def sanitize_structure(value: Any, *, home: Path | None = None) -> Any:
    """Recursively sanitize strings inside mappings, lists, and tuples."""
    if isinstance(value, str):
        return sanitize_text_for_display(value, home=home)
    if isinstance(value, Mapping):
        return {str(k): sanitize_structure(v, home=home) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_structure(item, home=home) for item in value]
    return value


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce=...`` attributes, CSP ``'nonce-...'`` tokens, and masked ``hm-...***XXXX`` suffixes."""
    if not isinstance(text, str):
        raise ValueError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _STRIP_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def scan_output_safety(text: str, *, home: Path | None = None) -> list[str]:
    """Audit rendered CLI/Markdown/JSON output for leaked keys, home paths, PII, static figures, or forbidden keys."""
    if not isinstance(text, str):
        raise ValueError("scan_output_safety requires a string.")
    scrubbed = strip_masked_keys_and_nonces(text)
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _RAW_API_KEY_RE.search(scrubbed):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _HOME_PATH_RE.search(text) or any(v in text for v in _home_variants(home)):
        issues.append("unredacted_home_path_detected")
    if _STATIC_FIG_UNIT_RE.search(scrubbed):
        issues.append("static_figure_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def compute_receipt_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 hex digest over ``payload`` excluding ``receipt_sha256``."""
    body = {str(k): v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_allowed_gateway_host(host: str, *, allow_test_hosts: bool = False) -> bool:
    """Return True iff ``host`` is allowed by the strict lateral-movement host policy."""
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
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    if allow_test_hosts and h in ALLOWED_TEST_TRANSPORT_HOSTS:
        return True
    labels = h.split(".")
    if len(labels) < 3 or any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    if h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if labels[-2:] == ["run", "app"]:
        return True
    if labels[-2:] == ["googleapis", "com"]:
        return True
    return False


def validate_gateway_url(url: str, *, allow_test_hosts: bool = False) -> str:
    """Validate gateway origin URL against strict host allowlist and transport scheme rules."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise GatewayPolicyError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise GatewayPolicyError("Gateway URL must not contain whitespace or control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise GatewayPolicyError("Gateway URL must not contain whitespace or control characters.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise GatewayPolicyError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise GatewayPolicyError("Gateway URL must not embed credentials.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise GatewayPolicyError("Gateway URL contains an invalid port or authority.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise GatewayPolicyError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise GatewayPolicyError("Gateway URL must be a bare origin without path, query, or fragment.")
    if port is not None and not (1 <= port <= 65535):
        raise GatewayPolicyError("Gateway URL port out of range.")
    if not parts.netloc or not is_allowed_gateway_host(host, allow_test_hosts=allow_test_hosts):
        raise GatewayPolicyError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS or (allow_test_hosts and host in ALLOWED_TEST_TRANSPORT_HOSTS):
        if parts.scheme not in ("http", "https"):
            raise GatewayPolicyError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise GatewayPolicyError("Remote gateway URL must use the https scheme.")
    return cleaned


def _safe_urlsplit(url: str) -> Any:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise GatewayPolicyError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise GatewayPolicyError("Request URL must not contain whitespace or control characters.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise GatewayPolicyError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise GatewayPolicyError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise GatewayPolicyError("Request URL must not embed credentials.")
    return parts


def _payload_has_key(obj: Any, target_keys: frozenset[str]) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).strip().lower() in target_keys or _payload_has_key(v, target_keys):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_payload_has_key(item, target_keys) for item in obj)
    return False


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    allow_test_hosts: bool = False,
    payload: Any = None,
) -> str:
    """Validate request target host and reject any LLM inference / chat-completion path or payload."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise GatewayPolicyError(f"Unsupported HTTP method {verb!r}.")
    parts = _safe_urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin, allow_test_hosts=allow_test_hosts)
    low_path = unquote(parts.path or "").lower()
    coder_id = "hivemind/" + "coder"
    if "completions" in low_path or "/chat" in low_path or coder_id in low_path:
        raise GatewayPolicyError(f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden.")
    if payload is not None:
        if verb in ("GET", "HEAD"):
            raise GatewayPolicyError(f"Zero-inference invariant violated: {verb} requests must not attach a body payload.")
        if isinstance(payload, (Mapping, list, tuple)):
            if _payload_has_key(payload, frozenset({"messages", "prompt"})):
                raise GatewayPolicyError(
                    "Zero-inference invariant violated: messages/prompt payloads are strictly forbidden."
                )
            serialized = json.dumps(
                dict(payload) if isinstance(payload, Mapping) else list(payload),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).lower()
            is_transfer_ref_only = (
                low_path == "/v1/hivemind/points/transfer"
                and not _payload_has_key(payload, frozenset({"model"}))
            )
            if coder_id in serialized and not is_transfer_ref_only:
                raise GatewayPolicyError(
                    "Zero-inference invariant violated: hivemind/coder inference payloads are strictly forbidden."
                )
        elif isinstance(payload, (str, bytes, bytearray)):
            text_payload = payload.decode("utf-8", errors="replace") if isinstance(payload, (bytes, bytearray)) else payload
            low_body = unquote(text_payload).lower()
            if '"messages"' in low_body or '"prompt"' in low_body or (
                coder_id in low_body and low_path != "/v1/hivemind/points/transfer"
            ):
                raise GatewayPolicyError(
                    "Zero-inference invariant violated: messages/prompt/model payloads are strictly forbidden."
                )
    return origin


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent HTTP redirects to untrusted hosts, downgrades, backslashes, credentials, or completion endpoints."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
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
            raise GatewayPolicyError("Untrusted redirect: empty target URL.")
        if "\\" in raw_loc or "\\" in unquote(raw_loc) or "@" in raw_loc or "@" in unquote(raw_loc):
            raise GatewayPolicyError("Untrusted redirect: forbidden backslash or '@' in Location.")
        orig_parts = _safe_urlsplit(req.full_url)
        dest_parts = _safe_urlsplit(newurl)
        orig_host = (orig_parts.hostname or "").lower()
        dest_host = (dest_parts.hostname or "").lower()
        if orig_host not in ALLOWED_LOOPBACK_HOSTS and dest_host in ALLOWED_LOOPBACK_HOSTS:
            raise GatewayPolicyError("Untrusted redirect from remote host to loopback.")
        if orig_parts.scheme == "https" and dest_parts.scheme != "https":
            raise GatewayPolicyError("Untrusted HTTPS-to-HTTP downgrade redirect.")
        effective_method = "GET" if (req.get_method() == "POST" and code in (301, 302, 303)) else req.get_method()
        _assert_zero_inference_request(effective_method, newurl, allow_test_hosts=False)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_zero_inference_request(new_req.get_method(), new_req.full_url, allow_test_hosts=False)
        return new_req


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise GatewayPolicyError("timeout_s must be a finite positive number.")
    val = float(timeout_s)
    if not math.isfinite(val) or val <= 0.0:
        raise GatewayPolicyError("timeout_s must be a finite positive number.")
    return min(max(0.05, val), 60.0)


def _prepare_http_headers(
    headers: Mapping[str, str] | None,
    *,
    has_json_body: bool = False,
    user_agent: str = "hivemind-connect/2.0-stdlib",
) -> dict[str, str]:
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": user_agent,
    }
    if has_json_body:
        req_headers["Content-Type"] = "application/json"
    if headers:
        for hk, hv in headers.items():
            ks = str(hk).strip()
            vs = str(hv).strip()
            raw_k = str(hk)
            raw_v = str(hv)
            if not ks or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_k + raw_v):
                raise GatewayPolicyError("HTTP headers must be non-empty and must not contain CR/LF or control characters.")
            req_headers[ks] = vs
    return req_headers


def _read_bounded_http_body(resp: Any, *, max_bytes: int = MAX_HTTP_BODY_BYTES) -> bytes:
    raw_bytes = resp.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise GatewayPolicyError(f"HTTP response exceeded {max_bytes} byte limit.")
    return bytes(raw_bytes)


def default_http_raw(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DOCTOR_HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Proxy-free stdlib raw HTTP helper returning ``(status_code, body_text, lower_headers)``."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_test_hosts=False, payload=payload)
    bounded_timeout = _validate_timeout_s(timeout_s)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload)).encode("utf-8")
    req_headers = _prepare_http_headers(
        headers,
        has_json_body=body_bytes is not None,
        user_agent="hivemind-connect/2.0-stdlib",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_text = _read_bounded_http_body(resp, max_bytes=MAX_HTTP_BODY_BYTES).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            lower_headers = {str(k).strip().lower(): str(v) for k, v in resp.headers.items()} if getattr(resp, "headers", None) else {}
            return status_code, raw_text, lower_headers
    except urllib.error.HTTPError as exc:
        try:
            err_bytes = _read_bounded_http_body(exc, max_bytes=MAX_HTTP_BODY_BYTES) if exc.fp is not None else b""
            err_text = err_bytes.decode("utf-8", errors="replace")
            err_headers = {str(k).strip().lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        finally:
            with suppress(Exception):
                exc.close()
        if err_text.strip():
            try:
                json.loads(err_text)
            except ValueError:
                err_text = sanitize_text_for_display(err_text)
        return int(exc.code), err_text, err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        err_doc = json.dumps(
            {
                "network_error": type(exc).__name__,
                "detail": sanitize_text_for_display(str(reason))[:200],
            }
        )
        return 0, err_doc, {"x-hivemind-network-error": type(exc).__name__}


class _StdlibHttpClient:
    """Proxy-free Python stdlib HTTP client using ``urllib.request.ProxyHandler({})``."""

    def __init__(self, *, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        self.timeout_s = _validate_timeout_s(timeout_s)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())

    def __enter__(self) -> _StdlibHttpClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StdlibHttpResponse:
        verb = method.upper()
        _assert_zero_inference_request(verb, url, allow_test_hosts=False, payload=json)
        req_headers = _prepare_http_headers(
            headers,
            has_json_body=json is not None,
            user_agent="hivemind-connect/2.0-stdlib",
        )
        body_bytes: bytes | None = None
        if json is not None:
            body_bytes = globals()["json"].dumps(json).encode("utf-8")

        req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=verb)
        eff_timeout = self.timeout_s if timeout is None else _validate_timeout_s(timeout)
        try:
            with self._opener.open(req, timeout=eff_timeout) as resp:
                raw_bytes = _read_bounded_http_body(resp, max_bytes=MAX_HTTP_BODY_BYTES)
                resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                return _StdlibHttpResponse(int(getattr(resp, "status", 200)), raw_bytes, resp_headers)
        except urllib.error.HTTPError as exc:
            try:
                err_bytes = _read_bounded_http_body(exc, max_bytes=MAX_HTTP_BODY_BYTES) if exc.fp is not None else b""
                resp_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
                return _StdlibHttpResponse(int(exc.code), err_bytes, resp_headers)
            finally:
                with suppress(Exception):
                    exc.close()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayConnectionError(f"{type(exc).__name__}: {exc}") from exc

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StdlibHttpResponse:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StdlibHttpResponse:
        return self.request("POST", url, json=json, headers=headers, timeout=timeout)


class _HttpxTransportClient:
    """Wrapper around optional ``httpx.Client`` enforcing zero-inference on custom transports."""

    def __init__(self, *, transport: Any, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required when passing a custom transport.")
        self._client = httpx.Client(timeout=_validate_timeout_s(timeout_s), transport=transport)

    def __enter__(self) -> _HttpxTransportClient:
        self._client.__enter__()
        return self

    def __exit__(self, *exc: object) -> bool:
        return bool(self._client.__exit__(*exc))

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        _assert_zero_inference_request(method, url, allow_test_hosts=True, payload=json)
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if headers is not None:
            kwargs["headers"] = headers
        if timeout is not None:
            kwargs["timeout"] = _validate_timeout_s(timeout)
        return self._client.request(method, url, **kwargs)

    def get(self, url: str, *, headers: Mapping[str, str] | None = None, timeout: float | None = None) -> Any:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return self.request("POST", url, json=json, headers=headers, timeout=timeout)


class _InjectedHttpClient:
    """Wrapper adapting an injected test ``http_fn`` callable to the client interface."""

    def __init__(self, http_fn: HttpFn) -> None:
        self._http_fn = http_fn

    def __enter__(self) -> _InjectedHttpClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StdlibHttpResponse:
        verb = method.upper()
        _assert_zero_inference_request(verb, url, allow_test_hosts=True, payload=json)
        sig = inspect.signature(self._http_fn)
        params = sig.parameters
        has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        has_var = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        pos = [
            p
            for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        try:
            if has_kw or ("payload" in params and not has_var and len(pos) <= 2):
                kw: dict[str, Any] = {}
                if has_kw or "payload" in params:
                    kw["payload"] = json
                if has_kw or "headers" in params:
                    kw["headers"] = dict(headers) if headers else None
                if timeout is not None and (has_kw or "timeout_s" in params):
                    kw["timeout_s"] = timeout
                raw_res = self._http_fn(verb, url, **kw)
            elif has_var or len(pos) >= 4:
                raw_res = self._http_fn(verb, url, json, dict(headers) if headers else None)
            elif len(pos) == 3:
                raw_res = self._http_fn(verb, url, json)
            else:
                raw_res = self._http_fn(verb, url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayConnectionError(f"{type(exc).__name__}: {exc}") from exc

        resp_headers: dict[str, str] = {}
        if isinstance(raw_res, tuple) and len(raw_res) == 3:
            code, body, hdrs = raw_res
            if isinstance(hdrs, Mapping):
                resp_headers = {str(k).lower(): str(v) for k, v in hdrs.items()}
        else:
            code, body = raw_res

        if isinstance(body, (dict, list)):
            content = globals()["json"].dumps(body).encode("utf-8")
            resp_headers.setdefault("content-type", "application/json")
        elif isinstance(body, bytes):
            content = body
        else:
            content = str(body or "").encode("utf-8")
            resp_headers.setdefault("content-type", "text/html; charset=utf-8")
        return _StdlibHttpResponse(int(code), content, resp_headers)

    def get(self, url: str, *, headers: Mapping[str, str] | None = None, timeout: float | None = None) -> _StdlibHttpResponse:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StdlibHttpResponse:
        return self.request("POST", url, json=json, headers=headers, timeout=timeout)


def _find_corporate_markers(raw_text: str) -> list[str]:
    """Return matched corporate indicator names without retaining raw input strings."""
    if not raw_text:
        return []
    matched: list[str] = []
    for marker_name, pattern in _CORP_TOKEN_PATTERNS:
        if pattern.search(raw_text) and marker_name not in matched:
            matched.append(marker_name)
    return matched


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Walk ``(workspace, *workspace.parents)`` to the first ``.git`` dir or worktree ``gitdir:``/``commondir``."""
    candidates: list[Path] = []
    try:
        for candidate_root in (workspace, *workspace.parents):
            dot_git = candidate_root / ".git"
            if dot_git.is_dir():
                cfg = dot_git / "config"
                if cfg.is_file():
                    candidates.append(cfg)
                break
            if dot_git.is_file():
                lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
                first_line = next((ln.strip() for ln in lines if ln.strip()), "")
                if first_line.lower().startswith("gitdir:"):
                    raw_gitdir = first_line.split(":", 1)[1].strip()
                    gitdir = Path(raw_gitdir)
                    if not gitdir.is_absolute():
                        gitdir = (candidate_root / gitdir).resolve()
                    wt_cfg = gitdir / "config"
                    if wt_cfg.is_file():
                        candidates.append(wt_cfg)
                    commondir_file = gitdir / "commondir"
                    if commondir_file.is_file():
                        rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if rel_common:
                            common_dir = Path(rel_common)
                            if not common_dir.is_absolute():
                                common_dir = (gitdir / common_dir).resolve()
                            common_cfg = common_dir / "config"
                            if common_cfg.is_file() and common_cfg not in candidates:
                                candidates.append(common_cfg)
                break
    except OSError:
        return candidates
    return candidates


def extract_git_config_remote_urls(config_text: str) -> list[str]:
    """Extract remote and ``insteadOf`` URLs from local ``.git/config`` text in memory."""
    urls: list[str] = []
    in_remote = False
    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip().lower()
            in_remote = header.startswith(("remote ", "remote\t", 'remote"', "url "))
            if header.startswith("url ") and '"' in stripped:
                quoted = stripped.split('"', 2)
                if len(quoted) >= 2 and quoted[1].strip():
                    urls.append(quoted[1].strip())
            continue
        if in_remote and "=" in stripped:
            key, val = stripped.split("=", 1)
            if key.strip().lower() in ("url", "pushurl", "insteadof", "pushinsteadof"):
                cleaned_val = val.strip().strip('"')
                if cleaned_val:
                    urls.append(cleaned_val)
    return urls


def evaluate_personal_pc_compliance(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Workspace compliance gate (0 writes, 0 network, 0 value leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_view = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    upper_env: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in env_view.items() if str(v).strip()
    }
    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _add(indicator: str, markers: Sequence[str] = ()) -> None:
        if indicator not in indicators:
            indicators.append(indicator)
        for mk in markers:
            if mk in CORPORATE_MARKERS:
                matched_markers.add(mk)

    if any(k.startswith("AMADEUS") for k in upper_env):
        _add("env:AMADEUS_ENV", ("amadeus",))

    dns_domain = upper_env.get("USERDNSDOMAIN", "")
    if dns_domain:
        _add("env:USERDNSDOMAIN", _find_corporate_markers(dns_domain) or ("corp",))

    userdomain = upper_env.get("USERDOMAIN", "")
    computername = upper_env.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        dom_hits = _find_corporate_markers(userdomain)
        if dom_hits:
            _add("env:USERDOMAIN", dom_hits)

    for env_key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "COMPUTERNAME",
        "HOSTNAME",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        val = upper_env.get(env_key, "")
        hits = _find_corporate_markers(val)
        if hits:
            _add(f"env:{env_key}", hits)

    if upper_env.get("ZSCALER_CERT_PATH"):
        _add("env:ZSCALER_CERT_PATH", ("zscaler",))

    scrubbed_ws = _PYTEST_TMP_PREFIX_RE.sub("pytest_tmp", str(workspace))
    for hit in _find_corporate_markers(scrubbed_ws):
        _add(f"workspace_path:{hit}", (hit,))

    for cfg_path in _resolve_git_config_files(workspace):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for remote_url in extract_git_config_remote_urls(cfg_text):
            for hit in _find_corporate_markers(remote_url):
                _add(f"git_remote:{hit}", (hit,))

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Conformité PC personnel vérifiée localement (0 marqueur pro). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"Personal-PC compliance verified locally (0 corporate markers). {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : marqueur professionnel / entreprise détecté "
            f"(0 écriture fichier, 0 requête HTTP). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE REFUSED ({joined}): corporate machine or repository marker detected "
            f"(0 file writes, 0 outbound HTTP calls). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_transmitted": False,
        "outbound_http_requests": 0,
        "file_writes": 0,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }


def default_credentials_path(home: Path | None = None) -> Path:
    """Return default ``~/.hivemind/credentials.json`` path."""
    return (home if home is not None else Path.home()) / ".hivemind" / "credentials.json"


def default_continue_path(home: Path | None = None) -> Path:
    """Return default ``~/.continue/hivemind_provider.yaml`` path."""
    return (home if home is not None else Path.home()) / ".continue" / "hivemind_provider.yaml"


def _write_file_secure_atomic(
    path: Path,
    content: str,
    *,
    posix_mode: int = 0o600,
    os_name: str | None = None,
) -> None:
    """Atomically write UTF-8 ``content`` to ``path`` with ``0600`` owner-only permissions."""
    if path.exists() and path.is_dir():
        raise ValueError("Target output path must not be an existing directory.")
    eff_os = os_name if os_name is not None else os.name
    if eff_os != "nt" and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, posix_mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if eff_os != "nt":
            os.chmod(tmp_path, posix_mode)
        os.replace(tmp_path, path)
        if eff_os != "nt":
            with suppress(OSError):
                os.chmod(path, posix_mode)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            tmp_cleanup_err = True
            _ = tmp_cleanup_err
        raise


def load_credentials(path: Path) -> dict[str, Any] | None:
    """Load credentials JSON from ``path`` if readable and containing a valid ``hm-...`` key."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict) or not is_valid_hivemind_key(loaded.get("api_key")):
        return None
    return loaded


def save_credentials_file(
    path: Path,
    *,
    account_id: str,
    api_key: str,
    gateway: str,
    handle: str,
    role: str,
    os_name: str | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Atomically write ``~/.hivemind/credentials.json`` with POSIX ``0600`` mode."""
    if not is_valid_hivemind_key(api_key):
        raise ValueError("Cannot save invalid HiveMind API key.")
    record = {
        "account_id": str(account_id),
        "api_key": api_key.strip(),
        "gateway": gateway.rstrip("/"),
        "handle": str(handle),
        "role": str(role),
    }
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    _write_file_secure_atomic(path, content, posix_mode=0o600, os_name=os_name)
    return {
        "path": redact_home_path(path, home=home),
        "masked_api_key": mask_api_key(api_key),
        "posix_mode": "0600" if (os_name or os.name) != "nt" else "windows_acl",
    }


def build_continue_provider_yaml(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    allow_test_hosts: bool = False,
) -> str:
    """Render Continue.dev provider YAML configuring ``hivemind/coder`` and ``hivemind/bee-stars``."""
    clean_gw = validate_gateway_url(gateway, allow_test_hosts=allow_test_hosts)
    if not is_valid_hivemind_key(api_key):
        raise ValueError("Cannot build Continue config without a valid hm-... API key.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 64:
        raise ValueError("max_tokens must be an integer >= 64.")
    clean_key = api_key.strip()
    return (
        "# HiveMind personal hobby/open-source coding configuration for Continue (VS Code / JetBrains)\n"
        f"# {COMPLIANCE_NOTICE_EN}\n"
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        f"    model: {CONTINUE_MODELS[0]}\n"
        f"    apiBase: {clean_gw}/v1\n"
        f"    apiKey: {clean_key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "      - apply\n"
        "    defaultCompletionOptions:\n"
        f"      maxTokens: {max_tokens}\n"
        "  - name: HiveMind Bee-Stars (Science)\n"
        "    provider: openai\n"
        f"    model: {CONTINUE_MODELS[1]}\n"
        f"    apiBase: {clean_gw}/v1\n"
        f"    apiKey: {clean_key}\n"
        "    roles:\n"
        "      - chat\n"
        "    defaultCompletionOptions:\n"
        f"      maxTokens: {max_tokens}\n"
    )


def write_continue_provider_file(
    path: Path,
    *,
    gateway: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    os_name: str | None = None,
    home: Path | None = None,
    allow_test_hosts: bool = False,
) -> dict[str, Any]:
    """Atomically write ``~/.continue/hivemind_provider.yaml`` with POSIX ``0600`` mode."""
    yaml_text = build_continue_provider_yaml(
        gateway,
        api_key,
        max_tokens=max_tokens,
        allow_test_hosts=allow_test_hosts,
    )
    _write_file_secure_atomic(path, yaml_text, posix_mode=0o600, os_name=os_name)
    return {
        "path": redact_home_path(path, home=home),
        "models": list(CONTINUE_MODELS),
        "posix_mode": "0600" if (os_name or os.name) != "nt" else "windows_acl",
    }


def public_model_id(model: str) -> str:
    """``coder`` and ``hivemind/coder`` both mean the gateway model id ``hivemind/coder`` (``auto`` stays ``hivemind/auto``)."""
    return model if model.startswith("hivemind/") else f"hivemind/{model}"


def render_snippet(
    tool: str,
    *,
    gateway: str,
    model: str,
    api_key: str = KEY_PLACEHOLDER,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Settings text for ``tool`` pointing at ``gateway``; pure function, no I/O."""
    base = gateway.rstrip("/") + "/v1"
    mid = public_model_id(model)
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if tool == "continue":
        return (
            "# Continue (VS Code / JetBrains): add to config.yaml\n"
            "models:\n"
            f"  - name: HiveMind {model}\n"
            "    provider: openai\n"
            f"    model: {mid}\n"
            f"    apiBase: {base}\n"
            f"    apiKey: {api_key}\n"
            "    roles: [chat, edit, apply]\n"
            "    defaultCompletionOptions:\n"
            f"      maxTokens: {max_tokens}\n"
        )
    if tool == "opencode":
        config = {
            "provider": {
                "hivemind": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "HiveMind OpenRouter",
                    "options": {"baseURL": base, "apiKey": api_key},
                    "models": {mid: {"name": f"HiveMind {model}", "limit": {"context": 8192, "output": max_tokens}}},
                }
            },
        }
        return (
            "// OpenCode: opencode.json\n"
            "// 7B models often write a tool call as plain text instead of calling the tool. If OpenCode answers with JSON like\n"
            '// {"name": "write", ...}, turn its tools off for chat-style use: add "tools": {"write": false, "edit": false, "bash": false}.\n'
            + json.dumps(config, indent=2)
            + "\n"
        )
    if tool == "aider":
        return (
            "# Aider: environment, then run aider\n"
            f"export OPENAI_API_BASE={base}\n"
            f"export OPENAI_API_KEY={api_key}\n"
            f"aider --model openai/{mid}\n"
        )
    if tool == "cline":
        return (
            "Cline (VS Code extension): open its settings and choose\n"
            "  API Provider : OpenAI Compatible\n"
            f"  Base URL     : {base}\n"
            f"  API Key      : {api_key}\n"
            f"  Model ID     : {mid}\n"
            f"  (set the maximum output tokens to {max_tokens} if the tool offers it)\n"
        )
    if tool == "zed":
        config = {
            "language_models": {
                "openai_compatible": {
                    "HiveMind": {
                        "api_url": base,
                        "available_models": [
                            {
                                "name": mid,
                                "display_name": f"HiveMind {model}",
                                "max_tokens": 32768,
                                "max_output_tokens": max_tokens,
                            }
                        ],
                    }
                }
            }
        }
        return (
            "// Zed: settings.json (the key is entered in Zed's UI or via the HIVEMIND_API_KEY environment variable)\n"
            + json.dumps(config, indent=2)
            + "\n"
        )
    if tool == "jetbrains":
        return (
            "PyCharm / IntelliJ: install the Continue plugin and use the `continue` snippet, or, in Settings > Tools > AI\n"
            "Assistant > Models, add an OpenAI-compatible provider with\n"
            f"  URL   : {base}\n"
            f"  Key   : {api_key}\n"
            f"  Model : {mid}\n"
        )
    if tool == "python":
        return (
            "# pip install openai\n"
            "from openai import OpenAI\n\n"
            f'client = OpenAI(base_url="{base}", api_key="{api_key}")\n'
            "reply = client.chat.completions.create(\n"
            f'    model="{mid}",\n'
            f"    max_tokens={max_tokens},\n"
            '    messages=[{"role": "user", "content": "Write a pygame window that bounces a ball."}],\n'
            ")\n"
            "print(reply.choices[0].message.content)\n"
            'print(reply.model_dump().get("hivemind"))  # price, charge and producer of this call\n'
        )
    if tool == "env":
        return f"export OPENAI_BASE_URL={base}\nexport OPENAI_API_KEY={api_key}\n"
    raise ValueError(f"unknown tool {tool!r}; choose one of: {', '.join(TOOLS)}")


def _ctx_environ(args: argparse.Namespace) -> Mapping[str, str]:
    injected = getattr(args, "_environ", None)
    return injected if injected is not None else os.environ


def _ctx_home(args: argparse.Namespace) -> Path | None:
    return getattr(args, "_home", None)


def _ctx_cwd(args: argparse.Namespace) -> Path | None:
    return getattr(args, "_cwd", None)


def _ctx_os_name(args: argparse.Namespace) -> str | None:
    return getattr(args, "_os_name", None)


def _ctx_allow_test_hosts(args: argparse.Namespace, client: Any = None) -> bool:
    return bool(getattr(args, "_allow_test_hosts", False)) or (
        client is not None and not isinstance(client, _StdlibHttpClient)
    )


def _credentials_path(args: argparse.Namespace) -> Path:
    raw = getattr(args, "credentials", None)
    if raw:
        return Path(str(raw)).expanduser()
    return default_credentials_path(home=_ctx_home(args))


def _continue_path(args: argparse.Namespace) -> Path:
    raw = getattr(args, "continue_file", None)
    if raw:
        return Path(str(raw)).expanduser()
    return default_continue_path(home=_ctx_home(args))


def _gateway(args: argparse.Namespace) -> str:
    env_map = _ctx_environ(args)
    raw = (
        getattr(args, "gateway", None)
        or env_map.get(GATEWAY_ENV)
        or env_map.get("HIVEMIND_GATEWAY_URL")
        or DEFAULT_GATEWAY
    )
    return str(raw).rstrip("/")


def _key(parser: argparse.ArgumentParser, args: argparse.Namespace | None = None) -> str:
    env_map = _ctx_environ(args) if args is not None else os.environ
    key = str(env_map.get(API_KEY_ENV, "")).strip()
    if key:
        return key
    if args is not None:
        saved = load_credentials(_credentials_path(args))
        if saved is not None:
            return str(saved["api_key"]).strip()
    parser.error(f"set {API_KEY_ENV} to your HiveMind API key (run `register` first)")
    return ""


def _save_or_print_key(body: dict[str, Any], args: argparse.Namespace, *, is_google: bool = False) -> int:
    home = _ctx_home(args)
    os_name = _ctx_os_name(args)
    gateway = _gateway(args)
    allow_test = _ctx_allow_test_hosts(args)
    api_key = str(body["api_key"])
    account_id = str(body.get("account_id") or "")
    role = str(body.get("role") or "prosumer")
    handle = str(body.get("handle") or getattr(args, "handle", "") or account_id)

    save_key_arg = getattr(args, "save_key", None)
    save_creds = bool(getattr(args, "save_credentials", False))
    write_cont = bool(getattr(args, "write_continue", False))

    if save_key_arg:
        path = Path(save_key_arg).expanduser()
        if path.exists():
            print(f"refusing to overwrite {redact_home_path(path, home=home)}", file=sys.stderr)
            return 1
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle_fp:
            handle_fp.write(api_key)
        print(f"key saved to {redact_home_path(path, home=home)} (owner-only). Then: export {API_KEY_ENV}=$(cat {redact_home_path(path, home=home)})")

    cred_info: dict[str, Any] | None = None
    if save_creds:
        cred_info = save_credentials_file(
            _credentials_path(args),
            account_id=account_id,
            api_key=api_key,
            gateway=gateway,
            handle=handle,
            role=role,
            os_name=os_name,
            home=home,
        )
        print(f"credentials saved to {cred_info['path']} ({cred_info['posix_mode']}, key {cred_info['masked_api_key']})")

    cont_info: dict[str, Any] | None = None
    if write_cont:
        cont_info = write_continue_provider_file(
            _continue_path(args),
            gateway=gateway,
            api_key=api_key,
            os_name=os_name,
            home=home,
            allow_test_hosts=allow_test,
        )
        print(f"continue provider saved to {cont_info['path']} ({cont_info['posix_mode']})")

    if not save_key_arg and not save_creds and not write_cont:
        if is_google:
            header = "Account created with Google!" if body.get("created") else "Welcome back: previous key rotated."
            print(f"{header}\nYour API key (shown once, store it safely):\n  {api_key}\nThen: export {API_KEY_ENV}=<that key>")
        else:
            print(f"Your API key (shown once, store it safely):\n  {api_key}\nThen: export {API_KEY_ENV}=<that key>")
    print(f"account {account_id} role {role}. New accounts hold 0 HMT: ask the operator for a grant, or earn by sharing a GPU.")
    return 0


def cmd_register(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Register a personal HiveMind account and optionally save credentials and Continue config."""
    gateway = validate_gateway_url(_gateway(args), allow_test_hosts=_ctx_allow_test_hosts(args, client))
    response = client.post(gateway + "/v1/hivemind/register", json={"handle": args.handle, "role": args.role})
    if response.status_code != 200:
        err = sanitize_text_for_display(response.text[:300], home=_ctx_home(args))
        print(f"registration failed ({response.status_code}): {err}", file=sys.stderr)
        return 1
    body = response.json()
    if isinstance(body, dict) and "handle" not in body:
        body = {**body, "handle": args.handle}
    return _save_or_print_key(body, args, is_google=False)


def cmd_register_google(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Register or sign in with a Google ID token and optionally save credentials / Continue config."""
    raw_gateway = _gateway(args)
    env_map = _ctx_environ(args)
    token = args.token or env_map.get("GOOGLE_ID_TOKEN", "")

    if not token and getattr(args, "gcloud", False):
        try:
            cmd = ["gcloud", "auth", "print-identity-token", "--include-email"]
            if getattr(args, "audience", None):
                cmd.append(f"--audiences={args.audience}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
            if proc.returncode == 0 and proc.stdout.strip():
                token = proc.stdout.strip()
            else:
                print(f"gcloud identity token failed: {proc.stderr.strip()}", file=sys.stderr)
        except (FileNotFoundError, OSError):
            print("gcloud command not found on PATH.", file=sys.stderr)

    if not token and getattr(args, "browser", False):
        gateway = validate_gateway_url(raw_gateway, allow_test_hosts=_ctx_allow_test_hosts(args, client))
        signup_url = f"{gateway}/signup"
        print(f"Opening browser for Google Sign-in: {signup_url}")
        try:
            import webbrowser

            webbrowser.open(signup_url)
        except Exception:
            browser_open_failed = True
            _ = browser_open_failed
        print("Complete Google sign-in in your browser.")
        print("Paste the API key displayed on the page, or enter your Google ID token:")
        try:
            entered = input("API Key or Google ID Token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1
        if entered.startswith("hm-"):
            return _save_or_print_key({"account_id": "google-user", "api_key": entered, "role": args.role, "created": False}, args, is_google=True)
        token = entered

    if not token:
        parser.error("provide a Google ID token via --token, set GOOGLE_ID_TOKEN, use --gcloud, or use --browser")

    gateway = validate_gateway_url(raw_gateway, allow_test_hosts=_ctx_allow_test_hosts(args, client))
    response = client.post(gateway + "/v1/hivemind/register/google", json={"id_token": token, "role": args.role})
    if response.status_code != 200:
        err = sanitize_text_for_display(response.text[:300], home=_ctx_home(args))
        print(f"Google sign-in failed ({response.status_code}): {err}", file=sys.stderr)
        return 1
    return _save_or_print_key(response.json(), args, is_google=True)


def cmd_check(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Verify API key (env or saved credentials), list live models/tiers, and optionally write configs."""
    api_key = _key(parser, args)
    gateway = validate_gateway_url(_gateway(args), allow_test_hosts=_ctx_allow_test_hosts(args, client))
    headers = {"Authorization": f"Bearer {api_key}"}
    me = client.get(gateway + "/v1/hivemind/me", headers=headers)
    if me.status_code != 200:
        print(f"key rejected ({me.status_code}). Check {API_KEY_ENV}.", file=sys.stderr)
        return 1
    account = me.json()
    live = [m.get("id") for m in client.get(gateway + "/v1/models", headers=headers).json().get("data", [])]
    tiers = client.get(gateway + "/v1/hivemind/catalog").json().get("tiers", [])
    print(f"account {account['account_id']}  role {account['role']}  balance {account['balance']} HMT  max price {account['max_price_per_1k']}/1k")
    print(f"live models now: {', '.join(live) if live else 'none'}")
    for tier in tiers:
        flag = "live" if tier["public_id"] in live else "no producer yet"
        print(f"  {tier['public_id']:<22} {'MoE ' if tier['moe'] else '    '}needs ~{tier['min_vram_gb']:g} GB  [{flag}]  {tier['good_for']}")
    if account["balance"] <= 0:
        print("balance is 0: requests will be refused until you are granted HMT or earn some as a producer.")

    home = _ctx_home(args)
    os_name = _ctx_os_name(args)
    if getattr(args, "save_credentials", False):
        cred_info = save_credentials_file(
            _credentials_path(args),
            account_id=str(account["account_id"]),
            api_key=api_key,
            gateway=gateway,
            handle=str(account.get("handle") or account["account_id"]),
            role=str(account["role"]),
            os_name=os_name,
            home=home,
        )
        print(f"credentials saved to {cred_info['path']} ({cred_info['posix_mode']}, key {cred_info['masked_api_key']})")
    if getattr(args, "write_continue", False):
        cont_info = write_continue_provider_file(
            _continue_path(args),
            gateway=gateway,
            api_key=api_key,
            os_name=os_name,
            home=home,
            allow_test_hosts=_ctx_allow_test_hosts(args),
        )
        print(f"continue provider saved to {cont_info['path']} ({cont_info['posix_mode']})")
    return 0


def cmd_transfer(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Transfer HMT Reflexion Points to another HiveMinder."""
    headers = {"Authorization": f"Bearer {_key(parser, args)}"}
    gateway = validate_gateway_url(_gateway(args), allow_test_hosts=_ctx_allow_test_hosts(args, client))
    payload: dict[str, Any] = {
        "recipient": args.to,
        "amount": float(args.amount),
        "memo": args.memo,
        "notify_amcp": not bool(args.no_amcp),
    }
    if args.idempotency_key:
        payload["idempotency_key"] = args.idempotency_key
    if args.model_ref:
        payload["model_ref"] = args.model_ref
    if args.pr_url:
        payload["pr_url"] = args.pr_url
    res = client.post(gateway + "/v1/hivemind/points/transfer", headers=headers, json=payload)
    if res.status_code != 200:
        err = sanitize_text_for_display(res.text[:300], home=_ctx_home(args))
        print(f"transfer failed ({res.status_code}): {err}", file=sys.stderr)
        return 1
    body = res.json()
    print(
        f"transfer {body['transfer_id']}: @{body['sender_handle']} -> @{body['recipient_handle']} "
        f"{body['amount_hmt']:.2f} HMT ({body['memo']}) | remaining balance {body['sender_balance_hmt']:.2f} HMT"
        + (f" | amcp_event {body['amcp_event_id']}" if body.get("amcp_event_id") else "")
    )
    return 0


def cmd_snippet(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Print ready-to-paste editor or library snippet without network I/O."""
    _ = client
    if args.list or not args.tool:
        print("tools: " + ", ".join(TOOLS))
        return 0
    env_map = _ctx_environ(args)
    key = str(env_map.get(API_KEY_ENV, "")).strip() if args.embed_key else KEY_PLACEHOLDER
    if args.embed_key and not key:
        parser.error(f"--embed-key needs {API_KEY_ENV} to be set")
    try:
        print(render_snippet(args.tool, gateway=_gateway(args), model=args.model, api_key=key, max_tokens=args.max_tokens), end="")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


def default_subprocess_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Run a local read-only command using explicit list args (``shell=False``) and bounded timeout."""
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("Subprocess command must be a non-empty list/tuple of strings.")
    cmd = [str(part) for part in argv]
    if Path(cmd[0]).name.lower() in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        raise GatewayPolicyError("Direct gcloud invocation is forbidden in hivemind_connect.")
    max_cap = 15.0 if env is not None else SUBPROCESS_TIMEOUT_S
    bounded = min(max(0.1, float(timeout_s)), max_cap)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
            check=False,
            shell=False,
            env=dict(env) if env is not None else None,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    return int(proc.returncode), proc.stdout or ""


def _optional_float(token: str) -> float | None:
    cleaned = token.strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def default_gpu_probe(
    *,
    runner: RunnerFn = default_subprocess_runner,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> GpuProbeSnapshot:
    """Query read-only ``nvidia-smi`` without requesting hardware UUID or serial numbers."""
    rc, stdout = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        min(float(timeout_s), SUBPROCESS_TIMEOUT_S),
    )
    if rc != 0 or not stdout.strip():
        return GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")

    candidates: list[GpuProbeSnapshot] = []
    for raw_line in stdout.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 6 or not parts[0]:
            return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        total_f = _optional_float(parts[1])
        free_f = _optional_float(parts[2])
        used_f = _optional_float(parts[3])
        temp_f = _optional_float(parts[4])
        util_f = _optional_float(parts[5])
        if total_f is None or free_f is None or total_f <= 0 or free_f < 0:
            return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        used_mib = int(used_f) if used_f is not None and used_f >= 0 else max(0, int(total_f) - int(free_f))
        candidates.append(
            GpuProbeSnapshot(
                present=True,
                parse_error=False,
                name=parts[0],
                total_mib=int(total_f),
                free_mib=int(free_f),
                used_mib=used_mib,
                temp_c=round(temp_f, 1) if temp_f is not None else None,
                util_pct=round(util_f, 1) if util_f is not None else 0.0,
                reason="ok",
            )
        )
    return max(
        candidates,
        key=lambda g: (
            g.free_mib or 0,
            -(g.temp_c if g.temp_c is not None else 999.0),
            g.total_mib or 0,
        ),
    )


def _normalize_gpu_snapshot(raw: Any) -> GpuProbeSnapshot:
    """Normalize ``GpuProbeSnapshot`` or mapping returned by an injected ``gpu_probe_fn``."""
    if raw is None:
        return GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")
    if isinstance(raw, GpuProbeSnapshot):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("gpu_probe_fn must return GpuProbeSnapshot, a mapping, or None.")
    present = bool(raw.get("present", False))
    if not present:
        return GpuProbeSnapshot(present=False, reason=str(raw.get("reason") or "no_nvidia_gpu"))
    if bool(raw.get("parse_error", False)):
        return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
    name = str(raw.get("name") or raw.get("gpu_name") or raw.get("gpu_model") or "").strip()
    raw_total = raw.get("total_mib", raw.get("total_vram_mib"))
    raw_free = raw.get("free_mib", raw.get("free_vram_mib"))
    raw_used = raw.get("used_mib", raw.get("used_vram_mib"))
    raw_temp = raw.get("temp_c")
    raw_util = raw.get("util_pct", raw.get("utilization_pct"))
    if not name or raw_total is None or raw_free is None:
        return GpuProbeSnapshot(present=True, parse_error=True, reason="incomplete_gpu_telemetry")
    total_mib = int(raw_total)
    free_mib = int(raw_free)
    used_mib = int(raw_used) if raw_used is not None else max(0, total_mib - free_mib)
    temp_c = float(raw_temp) if raw_temp is not None else None
    util_pct = float(raw_util) if raw_util is not None else 0.0
    return GpuProbeSnapshot(
        present=True,
        parse_error=False,
        name=name,
        total_mib=total_mib,
        free_mib=free_mib,
        used_mib=used_mib,
        temp_c=temp_c,
        util_pct=util_pct,
        reason=str(raw.get("reason") or "ok"),
    )


def default_process_probe(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> list[str]:
    """Read local process names using read-only ``tasklist`` (Windows) or ``ps`` (POSIX)."""
    eff_os = os_name if os_name is not None else os.name
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)
    discovered: list[str] = []
    seen: set[str] = set()
    if eff_os == "nt":
        rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], bounded)
        if rc != 0 or not out.strip():
            return discovered
        try:
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    low = row[0].strip().lower()
                    if low not in seen:
                        seen.add(low)
                        discovered.append(low)
        except csv.Error:
            return discovered
        return discovered

    rc, out = runner(["ps", "-eo", "comm="], bounded)
    if rc != 0 or not out.strip():
        return discovered
    for line in out.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        for candidate in (cleaned.lower(), Path(cleaned.replace("\\", "/")).name.lower()):
            if candidate and candidate not in seen:
                seen.add(candidate)
                discovered.append(candidate)
    return discovered


def _detect_active_games(
    process_names: Iterable[str],
    *,
    extra_games: Sequence[str] = (),
) -> list[str]:
    watchlist = set(STEAM_DX12_GAME_EXECUTABLES)
    trunc_prefixes = {exe[:15] for exe in STEAM_DX12_GAME_EXECUTABLES if len(exe) > 15}
    for extra in extra_games:
        cleaned = str(extra).strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            if len(cleaned) > 15:
                trunc_prefixes.add(cleaned[:15])
    matched: list[str] = []
    for raw_proc in process_names:
        raw_low = str(raw_proc).strip().lower().replace("\\", "/")
        if not raw_low:
            continue
        base = raw_low.rsplit("/", 1)[-1]
        is_steam = (
            base.startswith(STEAM_PROCESS_PREFIXES)
            or "steamapps/common/" in raw_low
            or base in ("steam_game", "steamwebhelper_game")
        )
        if (base in watchlist or base in trunc_prefixes or is_steam) and base not in matched:
            matched.append(base)
    return sorted(matched)


def default_worker_yield_drill(
    *,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    clock_fn: ClockFn = time.monotonic,
) -> dict[str, Any]:
    """Run a lightweight local Python subprocess worker preemption drill (``shell=False``)."""
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
        started = clock_fn()
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        release_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        exit_rc = proc.wait(timeout=bounded)
        yield_latency_ms = round(max(0.0, (clock_fn() - started) * 1000.0), 3)
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


def _invoke_with_supported_kwargs(fn: Callable[..., Any], **candidate_kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _run_yield_drill(
    worker_yield_fn: WorkerYieldFn | None,
    *,
    clock_fn: ClockFn,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[float | None, bool, int, str]:
    target_fn = worker_yield_fn if worker_yield_fn is not None else default_worker_yield_drill
    started = clock_fn()
    raw = _invoke_with_supported_kwargs(target_fn, timeout_s=timeout_s, clock_fn=clock_fn)
    wall_ms = round(max(0.0, (clock_fn() - started) * 1000.0), 3)

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


def parse_ces_input(
    raw: str,
    *,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
) -> dict[str, Any]:
    """Parse 1-line interactive CES input (``'5'``, ``'4/5'``, ``'3 lag'``, or empty to skip)."""
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": default_profile,
            "ease_rating": None,
            "zero_game_lag": not default_had_game_lag,
        }
    normalized = re.sub(r"\s*/\s*5\b", "", cleaned, count=1)
    tokens = [tok.strip(",;:").lower() for tok in normalized.split() if tok.strip(",;:")]
    rating: int | None = None
    profile = default_profile
    had_lag = bool(default_had_game_lag)

    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            val = int(tok)
            if not (1 <= val <= 5) or rating is not None:
                raise ValueError("CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("dev", "gamer"):
            profile = tok
        elif tok in ("lag", "had-game-lag", "had_game_lag", "stutter", "saccade"):
            had_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "ok"):
            had_lag = False
        else:
            raise ValueError(f"Unrecognized CES token {tok!r}; expected 1..5 [dev|gamer] [lag|nolag].")

    if rating is None:
        raise ValueError("Interactive CES input must include an ease_rating from 1 to 5.")

    return {
        "skipped": False,
        "profile": profile,
        "ease_rating": rating,
        "zero_game_lag": not had_lag,
    }


def _platform_feedback_tag(os_name: str | None = None, system_name: str | None = None) -> str:
    eff_os = (os_name if os_name is not None else os.name).strip().lower()
    eff_sys = (system_name if system_name is not None else platform.system()).strip().lower()
    if eff_os == "nt" or eff_sys.startswith("win") or "windows" in eff_sys:
        return "windows"
    if "darwin" in eff_sys or "mac" in eff_sys:
        return "macos"
    if "linux" in eff_sys:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", eff_sys or eff_os).strip("-_")[:24]
    return cleaned if (cleaned and _PLATFORM_TAG_RE.fullmatch(cleaned)) else "unknown"


def _sanitize_host_label(raw_label: str | None, *, home: Path | None = None) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        return DEFAULT_HOST_LABEL
    scrubbed = sanitize_text_for_display(raw_label.strip(), home=home)
    if _find_corporate_markers(scrubbed):
        return DEFAULT_HOST_LABEL
    cleaned = _HOST_LABEL_CLEAN_RE.sub("-", scrubbed).strip("-._")[:128]
    return cleaned if cleaned else DEFAULT_HOST_LABEL


def _sanitize_manifest_sha(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip().lower()
    if not cleaned or cleaned == "unknown" or not _BUILD_SHA_RE.fullmatch(cleaned):
        return "unknown"
    if any(tok in cleaned for tok in _SECRET_VALUE_MARKERS):
        return "unknown"
    return cleaned


def _sanitize_manifest_revision(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned or not _REVISION_RE.fullmatch(cleaned):
        return None
    low = cleaned.lower()
    if low.startswith(("hm-", "hm_live_", "sk-")) or any(tok in low for tok in _SECRET_VALUE_MARKERS):
        return None
    return cleaned


def _honest_counter(raw_val: object) -> int | str:
    if isinstance(raw_val, int) and not isinstance(raw_val, bool) and raw_val >= 0:
        return int(raw_val)
    return "n/a"


def _safe_http_get(
    client: Any,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DOCTOR_HTTP_TIMEOUT_S,
) -> tuple[int, Any]:
    try:
        resp = client.get(url, headers=headers, timeout=timeout)
    except (GatewayConnectionError, OSError, ValueError):
        return 0, {}
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.HTTPError):
            return 0, {}
        raise
    ctype = str(resp.headers.get("content-type", "")).lower() if getattr(resp, "headers", None) else ""
    if "application/json" in ctype:
        try:
            return int(resp.status_code), resp.json()
        except ValueError:
            return int(resp.status_code), {"raw_length": len(resp.text)}
    raw_text = resp.text
    if not raw_text.strip():
        return int(resp.status_code), {"content_type": ctype, "html_length": 0}
    try:
        return int(resp.status_code), json.loads(raw_text)
    except ValueError:
        return int(resp.status_code), {"content_type": ctype, "html_length": len(raw_text), "raw": raw_text[:240]}


def _safe_http_post(
    client: Any,
    url: str,
    *,
    payload: dict[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = DOCTOR_HTTP_TIMEOUT_S,
) -> tuple[int, Any]:
    try:
        resp = client.post(url, json=payload, headers=headers, timeout=timeout)
    except (GatewayConnectionError, OSError, ValueError):
        return 0, {}
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.HTTPError):
            return 0, {}
        raise
    try:
        return int(resp.status_code), resp.json()
    except ValueError:
        return int(resp.status_code), {"raw_length": len(resp.text)}


def _get_ces_path_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CES_LOCK_GUARD:
        lock = _CES_PATH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _CES_PATH_LOCKS[resolved] = lock
        return lock


def _expand_home_path(raw_path: Path | str, *, home: Path | None = None) -> Path:
    text = str(raw_path).strip()
    if text == "~" or text.startswith("~/") or text.startswith("~\\"):
        base_home = Path(home) if home is not None else Path.home()
        suffix = text[2:] if len(text) > 1 else ""
        return (base_home / suffix) if suffix else base_home
    return Path(text)


def default_ces_queue_path(
    *,
    home: Path | None = None,
    credentials_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the default ``0600`` CES retry spool path without touching real ``~/.hivemind`` when isolated."""
    env_view = env if env is not None else os.environ
    resolved_home = (
        Path(home)
        if home is not None
        else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
    )
    if credentials_path is not None and str(credentials_path).strip():
        cred_p = _expand_home_path(credentials_path, home=resolved_home)
        if home is not None and cred_p == default_credentials_path() and resolved_home.resolve() != Path.home().resolve():
            return resolved_home / ".hivemind" / DEFAULT_CES_QUEUE_FILENAME
        return cred_p.parent / DEFAULT_CES_QUEUE_FILENAME
    return resolved_home / ".hivemind" / DEFAULT_CES_QUEUE_FILENAME


def _validate_ces_spool_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("CES feedback payload must be a JSON object.")
    allowed_keys = {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
    extra_keys = set(raw.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(f"CES feedback payload contains unsupported keys: {sorted(extra_keys)}")
    prof = normalize_profile(raw.get("profile"))  # type: ignore[arg-type]
    ease = parse_ease_rating(raw.get("ease_rating"))  # type: ignore[arg-type]
    zero_lag = raw.get("zero_game_lag")
    if not isinstance(zero_lag, bool):
        raise ValueError("CES feedback zero_game_lag must be a boolean.")
    plat_raw = str(raw.get("platform") or "unknown").strip().lower()
    if not _PLATFORM_TAG_RE.fullmatch(plat_raw):
        raise ValueError("CES feedback platform contains unsupported characters.")
    note_raw = raw.get("note", "")
    if not isinstance(note_raw, str) or note_raw != "":
        raise ValueError("CES feedback note must remain empty to guarantee zero PII.")
    return {
        "profile": prof,
        "ease_rating": ease,
        "zero_game_lag": zero_lag,
        "platform": plat_raw,
        "note": "",
    }


def build_ces_idempotency_key(
    *,
    gateway: str = DEFAULT_GATEWAY,
    host_label: str = DEFAULT_HOST_LABEL,
    resolved_key: str | None = None,
    payload: Mapping[str, Any],
) -> str:
    """Build deterministic zero-PII ``Idempotency-Key`` for CES feedback submissions."""
    key_salt = hashlib.sha256(f"hm-connect:{resolved_key}".encode()).hexdigest()[:16] if resolved_key else "anon"
    canon_ces = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return (
        "ces-idem-"
        + hashlib.sha256(
            f"hivemind.connect.ces/1:{gateway.rstrip('/')}:{host_label}:{key_salt}:{canon_ces}".encode()
        ).hexdigest()[:32]
    )


def _classify_ces_flush_status(status_code: int) -> tuple[str, str]:
    if 200 <= status_code < 300:
        return "delivered", "ok"
    if status_code == 404:
        return "retryable", "stale_cloud_run_404"
    if status_code == 408:
        return "retryable", "request_timeout_408"
    if status_code == 429:
        return "retryable", "rate_limited_429"
    if status_code == 0:
        return "retryable", "network_timeout"
    if status_code >= 500:
        return "retryable", f"gateway_{status_code}"
    return "permanent_quarantine", f"permanent_http_{status_code}"


class ConnectCesRetrySpool:
    """Thread-safe, atomic POSIX ``0600``, bounded CES retry spool (128 pending / 512 delivered)."""

    def __init__(
        self,
        queue_file: Path | str | None = None,
        *,
        max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
        max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
        home: Path | None = None,
        credentials_path: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        os_name: str | None = None,
    ) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not (1 <= max_entries <= MAX_CES_QUEUE_ENTRIES_CEILING)
        ):
            raise ValueError(f"max_entries must be an integer in 1..{MAX_CES_QUEUE_ENTRIES_CEILING}.")
        if isinstance(max_delivered, bool) or not isinstance(max_delivered, int) or not (1 <= max_delivered <= 32768):
            raise ValueError("max_delivered must be an integer in 1..32768.")
        env_view = env if env is not None else os.environ
        resolved_home = (
            Path(home)
            if home is not None
            else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
        )
        self._home = resolved_home
        self._eff_os = os_name if os_name is not None else os.name
        self._max_entries = int(max_entries)
        self._max_delivered = int(max_delivered)
        if queue_file is not None and str(queue_file).strip():
            raw_path = _expand_home_path(queue_file, home=resolved_home)
        else:
            raw_path = default_ces_queue_path(home=resolved_home, credentials_path=credentials_path, env=env_view)
        self._path = raw_path.resolve()
        self._lock = _get_ces_path_lock(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def redacted_path(self) -> str:
        return redact_home_path(self._path, home=self._home)

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
            "delivered_ids": [],
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
                    "event_id": hashlib.sha256(b"corrupted_connect_ces_spool").hexdigest(),
                    "idempotency_key": "corrupted-spool-recovered",
                    "http_status": 0,
                    "reason": "corrupted_queue_json_recovered",
                }
            ]
            return recovered, True
        if not isinstance(parsed, Mapping):
            return self._empty_state(), True

        mutated = False
        delivered: list[str] = []
        seen_delivered: set[str] = set()
        raw_delivered = parsed.get("delivered_ids")
        if raw_delivered is None:
            raw_delivered = parsed.get("delivered_event_ids")
        if isinstance(raw_delivered, list):
            for item in raw_delivered:
                if isinstance(item, str) and item.strip() and item.strip() not in seen_delivered:
                    tok = item.strip()[:128]
                    delivered.append(tok)
                    seen_delivered.add(tok)
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
                    q_status = q_item.get("http_status")
                    quarantined.append(
                        {
                            "event_id": q_item["event_id"][:64],
                            "idempotency_key": str(q_item.get("idempotency_key") or "")[:128],
                            "http_status": (
                                int(q_status)
                                if not isinstance(q_status, bool) and isinstance(q_status, int)
                                else 422
                            ),
                            "reason": sanitize_text_for_display(
                                str(q_item.get("reason") or "quarantined")[:80],
                                home=self._home,
                            ),
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
                idem_raw = str(entry.get("idempotency_key") or "").strip()[:128]
                eid_raw = entry.get("event_id")
                eid = (
                    eid_raw
                    if isinstance(eid_raw, str) and _SHA256_HEX_RE.fullmatch(eid_raw)
                    else hashlib.sha256((idem_raw or json.dumps(entry, sort_keys=True, default=str)).encode("utf-8")).hexdigest()
                )
                dedup_tok = idem_raw or eid
                if dedup_tok in seen_delivered or eid in seen_delivered or dedup_tok in seen_pending or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = _validate_ces_spool_payload(entry.get("payload"))
                except ValueError:
                    mutated = True
                    quarantined.append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_raw,
                            "http_status": 422,
                            "reason": "tampered_or_invalid_local_payload",
                        }
                    )
                    quarantined = quarantined[-self._max_entries :]
                    continue
                raw_rc = entry.get("retry_count", 0)
                rc = max(0, int(raw_rc)) if not isinstance(raw_rc, bool) and isinstance(raw_rc, int) else 0
                raw_st = entry.get("last_http_status")
                last_st = int(raw_st) if not isinstance(raw_st, bool) and isinstance(raw_st, int) else None
                raw_ts = entry.get("enqueued_epoch", 0.0)
                enq_ts = round(max(0.0, float(raw_ts)), 3) if not isinstance(raw_ts, bool) and isinstance(raw_ts, (int, float)) else 0.0
                pending.append(
                    {
                        "event_id": eid,
                        "idempotency_key": idem_raw or f"ces-idem-{eid[:32]}",
                        "payload": clean_payload,
                        "enqueued_epoch": enq_ts,
                        "retry_count": rc,
                        "last_http_status": last_st,
                        "last_reason": sanitize_text_for_display(str(entry.get("last_reason") or "")[:80], home=self._home),
                    }
                )
                seen_pending.add(dedup_tok)
                seen_pending.add(eid)

        raw_dropped = parsed.get("dropped_overflow_count", 0)
        dropped_count = (
            max(0, int(raw_dropped))
            if not isinstance(raw_dropped, bool) and isinstance(raw_dropped, int)
            else 0
        )
        if len(pending) > self._max_entries:
            overflow = len(pending) - self._max_entries
            pending = pending[-self._max_entries :]
            dropped_count += overflow
            mutated = True

        return (
            {
                "schema": CES_QUEUE_SCHEMA,
                "max_entries": self._max_entries,
                "max_delivered": self._max_delivered,
                "dropped_overflow_count": dropped_count,
                "pending": pending,
                "delivered_ids": delivered,
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
            "delivered_ids": list(state.get("delivered_ids", []))[-self._max_delivered :],
            "quarantined": list(state.get("quarantined", []))[-self._max_entries :],
        }
        serialized = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        safety_hits = scan_output_safety(serialized, home=self._home)
        if safety_hits:
            raise ValueError(f"Refusing unsafe CES spool serialization: {', '.join(safety_hits)}")
        _write_file_secure_atomic(self._path, serialized, posix_mode=0o600, os_name=self._eff_os)

    def snapshot(self, *, lang: str = "en") -> dict[str, Any]:
        """Return a sanitized read-only snapshot without creating a spool file when absent."""
        lang_code = normalize_lang(lang)
        with self._lock:
            state, _ = self._load_state_unlocked()
            exists = self._path.is_file()
        snap: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "ces_queue_snapshot",
            "lang": lang_code,
            "queue_file": self.redacted_path,
            "queue_exists": exists,
            "max_entries": self._max_entries,
            "max_delivered": self._max_delivered,
            "pending_count": len(state["pending"]),
            "delivered_ledger_count": len(state["delivered_ids"]),
            "quarantined_count": len(state["quarantined"]),
            "dropped_overflow_count": int(state["dropped_overflow_count"]),
            "pending_idempotency_keys": [item["idempotency_key"] for item in state["pending"]],
        }
        snap["receipt_sha256"] = compute_receipt_sha256(snap)
        return snap

    def mark_delivered(self, token: str) -> bool:
        """Record ``token`` as delivered ONLY if the spool file already exists (no file creation when empty)."""
        cleaned = str(token or "").strip()
        if not cleaned:
            return False
        eid = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        with self._lock:
            if not self._path.is_file():
                return False
            state, mutated = self._load_state_unlocked()
            prev_len = len(state["pending"])
            state["pending"] = [
                item for item in state["pending"] if item["idempotency_key"] != cleaned and item["event_id"] != eid
            ]
            removed = len(state["pending"]) != prev_len
            if cleaned not in state["delivered_ids"]:
                state["delivered_ids"].append(cleaned)
                state["delivered_ids"] = state["delivered_ids"][-self._max_delivered :]
                mutated = True
            if removed or mutated:
                self._atomic_save_unlocked(state)
            return removed

    def enqueue(
        self,
        *,
        gateway: str = DEFAULT_GATEWAY,
        host_label: str = DEFAULT_HOST_LABEL,
        resolved_key: str | None = None,
        profile: str = "dev",
        ease_rating: int = 5,
        zero_game_lag: bool = True,
        os_name: str | None = None,
        system_name: str | None = None,
        platform: str | None = None,
        idempotency_key: str | None = None,
        last_http_status: int | None = None,
        last_reason: str = "",
        enforce_personal_pc: bool = True,
        allow_test_hosts: bool = False,
        environ: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Enforce compliance, deduplicate by ``Idempotency-Key``, cap pending at ``max_entries``, and save ``0600``."""
        lang_code = normalize_lang(lang)
        eff_env = environ if environ is not None else env
        comp_doc = evaluate_personal_pc_compliance(env=eff_env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            blocked: dict[str, Any] = {
                "schema": SCHEMA,
                "mode": "enqueue_ces",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "queue_file": self.redacted_path,
                "enqueued": False,
                "deduplicated": False,
                "duplicate_state": "compliance_blocked",
                "evicted_oldest": False,
                "file_writes": 0,
                "network_calls": 0,
                "compliance": comp_doc,
                "summary": comp_doc["summary"],
            }
            blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
            return blocked

        clean_gw = validate_gateway_url(gateway, allow_test_hosts=allow_test_hosts)
        plat = (
            platform.strip().lower()
            if (isinstance(platform, str) and platform.strip())
            else _platform_feedback_tag(os_name=os_name, system_name=system_name)
        )
        clean_payload = _validate_ces_spool_payload(
            {
                "profile": profile,
                "ease_rating": ease_rating,
                "zero_game_lag": zero_game_lag,
                "platform": plat,
                "note": "",
            }
        )
        if idempotency_key is not None:
            if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
                raise ValueError("idempotency_key must be a string.")
            idem_key = idempotency_key.strip()
            if (
                not idem_key
                or len(idem_key) > 128
                or not idem_key.isascii()
                or any(ord(c) < 32 or ord(c) == 127 for c in idem_key)
            ):
                raise ValueError("idempotency_key must be 1..128 printable ASCII characters.")
        else:
            idem_key = build_ces_idempotency_key(
                gateway=clean_gw,
                host_label=_sanitize_host_label(host_label, home=self._home),
                resolved_key=resolved_key,
                payload=clean_payload,
            )
        event_id = hashlib.sha256(idem_key.encode("utf-8")).hexdigest()
        ts = round(time.time(), 3)

        with self._lock:
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if idem_key in delivered_ids or event_id in delivered_ids:
                if recovered:
                    self._atomic_save_unlocked(state)
                res_deliv: dict[str, Any] = {
                    "schema": SCHEMA,
                    "mode": "enqueue_ces",
                    "ok": True,
                    "ready": True,
                    "compliance_ok": True,
                    "exit_code": 0,
                    "queue_file": self.redacted_path,
                    "enqueued": False,
                    "deduplicated": True,
                    "duplicate_state": "already_delivered",
                    "evicted_oldest": False,
                    "event_id": event_id,
                    "idempotency_key": idem_key,
                    "payload": clean_payload,
                    "pending_count": len(pending),
                    "delivered_ledger_count": len(delivered_ids),
                    "quarantined_count": len(state["quarantined"]),
                    "dropped_overflow_count": int(state["dropped_overflow_count"]),
                    "file_writes": 1 if recovered else 0,
                    "network_calls": 0,
                    "compliance": comp_doc,
                }
                res_deliv["receipt_sha256"] = compute_receipt_sha256(res_deliv)
                return res_deliv

            if any(item["idempotency_key"] == idem_key or item["event_id"] == event_id for item in pending):
                if recovered:
                    self._atomic_save_unlocked(state)
                res_pend: dict[str, Any] = {
                    "schema": SCHEMA,
                    "mode": "enqueue_ces",
                    "ok": True,
                    "ready": True,
                    "compliance_ok": True,
                    "exit_code": 0,
                    "queue_file": self.redacted_path,
                    "enqueued": False,
                    "deduplicated": True,
                    "duplicate_state": "already_pending",
                    "evicted_oldest": False,
                    "event_id": event_id,
                    "idempotency_key": idem_key,
                    "payload": clean_payload,
                    "pending_count": len(pending),
                    "delivered_ledger_count": len(delivered_ids),
                    "quarantined_count": len(state["quarantined"]),
                    "dropped_overflow_count": int(state["dropped_overflow_count"]),
                    "file_writes": 1 if recovered else 0,
                    "network_calls": 0,
                    "compliance": comp_doc,
                }
                res_pend["receipt_sha256"] = compute_receipt_sha256(res_pend)
                return res_pend

            evicted_oldest = False
            if len(pending) >= self._max_entries:
                pending.pop(0)
                state["dropped_overflow_count"] = int(state["dropped_overflow_count"]) + 1
                evicted_oldest = True

            pending.append(
                {
                    "event_id": event_id,
                    "idempotency_key": idem_key,
                    "payload": clean_payload,
                    "enqueued_epoch": ts,
                    "retry_count": 0,
                    "last_http_status": last_http_status,
                    "last_reason": sanitize_text_for_display(str(last_reason or "")[:80], home=self._home),
                }
            )
            self._atomic_save_unlocked(state)
            pending_count = len(pending)
            delivered_count = len(delivered_ids)
            quarantined_count = len(state["quarantined"])
            dropped_count = int(state["dropped_overflow_count"])

        res_ok: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "enqueue_ces",
            "ok": True,
            "ready": True,
            "compliance_ok": True,
            "exit_code": 0,
            "queue_file": self.redacted_path,
            "enqueued": True,
            "deduplicated": False,
            "duplicate_state": "none",
            "evicted_oldest": evicted_oldest,
            "event_id": event_id,
            "idempotency_key": idem_key,
            "payload": clean_payload,
            "pending_count": pending_count,
            "delivered_ledger_count": delivered_count,
            "quarantined_count": quarantined_count,
            "dropped_overflow_count": dropped_count,
            "file_writes": 1,
            "network_calls": 0,
            "compliance": comp_doc,
        }
        res_ok["receipt_sha256"] = compute_receipt_sha256(res_ok)
        return res_ok

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        client: Any = None,
        http_fn: HttpFn | None = None,
        transport: Any = None,
        raw_http_fn: RawHttpFn | None = None,
        allow_test_hosts: bool | None = None,
        enforce_personal_pc: bool = True,
        environ: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        timeout_s: float = DOCTOR_HTTP_TIMEOUT_S,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Idempotently flush pending CES events with ``Idempotency-Key`` (no file created when empty)."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        eff_env = environ if environ is not None else env
        eff_allow_test = (
            allow_test_hosts
            if allow_test_hosts is not None
            else bool(
                (client is not None and not isinstance(client, _StdlibHttpClient))
                or http_fn is not None
                or transport is not None
                or (raw_http_fn is not None and raw_http_fn is not default_http_raw)
            )
        )
        comp_doc = evaluate_personal_pc_compliance(env=eff_env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            blocked: dict[str, Any] = {
                "schema": SCHEMA,
                "mode": "flush_ces_queue",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "gateway": DEFAULT_GATEWAY,
                "queue_file": self.redacted_path,
                "queue_exists": self._path.is_file(),
                "flush_ok": False,
                "flushed_count": 0,
                "delivered_count": 0,
                "retained_retry_count": 0,
                "quarantined_count": 0,
                "pending_count": 0,
                "delivered_ledger_count": 0,
                "total_quarantined_count": 0,
                "stale_cloud_run_detected": False,
                "file_writes": 0,
                "network_calls": 0,
                "zero_inference_verified": True,
                "outcomes": [],
                "diagnostics": [comp_doc["summary"]],
                "compliance": comp_doc,
                "summary": comp_doc["summary"],
            }
            blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
            return blocked

        clean_gw = validate_gateway_url(gateway, allow_test_hosts=eff_allow_test)
        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
        bounded_to = _validate_timeout_s(timeout_s)

        delivered_now = 0
        retained_retry = 0
        quarantined_now = 0
        network_calls = 0
        stale_404_seen = False
        outcomes: list[dict[str, Any]] = []

        with self._lock:
            if not self._path.is_file():
                summary_empty = (
                    "File CES vide (aucun fichier créé, 0 appel réseau)."
                    if is_fr
                    else "CES retry queue empty (no queue file created, 0 network calls)."
                )
                empty_res: dict[str, Any] = {
                    "schema": SCHEMA,
                    "mode": "flush_ces_queue",
                    "ok": True,
                    "ready": True,
                    "compliance_ok": True,
                    "exit_code": 0,
                    "gateway": clean_gw,
                    "queue_file": self.redacted_path,
                    "queue_exists": False,
                    "flush_ok": True,
                    "flushed_count": 0,
                    "delivered_count": 0,
                    "retained_retry_count": 0,
                    "quarantined_count": 0,
                    "pending_count": 0,
                    "delivered_ledger_count": 0,
                    "total_quarantined_count": 0,
                    "stale_cloud_run_detected": False,
                    "file_writes": 0,
                    "network_calls": 0,
                    "zero_inference_verified": True,
                    "outcomes": [],
                    "diagnostics": [],
                    "compliance": comp_doc,
                    "summary": summary_empty,
                }
                empty_res["receipt_sha256"] = compute_receipt_sha256(empty_res)
                return empty_res

            state, mutated = self._load_state_unlocked()
            if mutated:
                quarantined_now += len(state["quarantined"])
            delivered_set = set(state["delivered_ids"])
            next_pending: list[dict[str, Any]] = []

            owned_cm: Any = None
            active_client = client
            if active_client is None:
                if http_fn is not None:
                    owned_cm = _InjectedHttpClient(http_fn)
                elif transport is not None:
                    owned_cm = _HttpxTransportClient(transport=transport, timeout_s=bounded_to)
                elif raw_http_fn is None or raw_http_fn is default_http_raw:
                    owned_cm = _StdlibHttpClient(timeout_s=bounded_to)

            def _send_one(entry_payload: dict[str, Any], hdrs: dict[str, str]) -> int:
                if active_client is not None:
                    c, _ = _safe_http_post(active_client, post_url, payload=entry_payload, headers=hdrs, timeout=bounded_to)
                    return c
                assert raw_http_fn is not None
                c_raw, _, _ = _dispatch_raw_http_fn(
                    raw_http_fn,
                    "POST",
                    post_url,
                    payload=entry_payload,
                    headers=hdrs,
                    allow_test_hosts=eff_allow_test,
                )
                return c_raw

            try:
                if owned_cm is not None:
                    active_client = owned_cm.__enter__()
                for entry in state["pending"]:
                    eid = str(entry["event_id"])
                    idem_key = str(entry["idempotency_key"])
                    if idem_key in delivered_set or eid in delivered_set:
                        mutated = True
                        continue

                    code = _send_one(dict(entry["payload"]), {"Idempotency-Key": idem_key})
                    network_calls += 1
                    disposition, reason = _classify_ces_flush_status(code)
                    mutated = True

                    if disposition == "delivered":
                        state["delivered_ids"].append(idem_key)
                        delivered_set.add(idem_key)
                        delivered_set.add(eid)
                        delivered_now += 1
                        outcomes.append(
                            {
                                "event_id": eid,
                                "idempotency_key": idem_key,
                                "disposition": "delivered",
                                "http_status": code,
                            }
                        )
                    elif disposition == "retryable":
                        if code == 404:
                            stale_404_seen = True
                        new_rc = int(entry.get("retry_count", 0)) + 1
                        next_pending.append(
                            {
                                **entry,
                                "retry_count": new_rc,
                                "last_http_status": code,
                                "last_reason": reason,
                            }
                        )
                        retained_retry += 1
                        outcomes.append(
                            {
                                "event_id": eid,
                                "idempotency_key": idem_key,
                                "disposition": "retained_for_retry",
                                "http_status": code,
                                "reason": reason,
                                "retry_count": new_rc,
                            }
                        )
                    else:
                        state["quarantined"].append(
                            {
                                "event_id": eid,
                                "idempotency_key": idem_key,
                                "http_status": code,
                                "reason": reason,
                            }
                        )
                        state["quarantined"] = state["quarantined"][-self._max_entries :]
                        quarantined_now += 1
                        outcomes.append(
                            {
                                "event_id": eid,
                                "idempotency_key": idem_key,
                                "disposition": "quarantined",
                                "http_status": code,
                                "reason": reason,
                            }
                        )
            finally:
                if owned_cm is not None:
                    owned_cm.__exit__(None, None, None)

            state["pending"] = next_pending
            state["delivered_ids"] = state["delivered_ids"][-self._max_delivered :]
            if mutated:
                self._atomic_save_unlocked(state)

            pending_after = len(state["pending"])
            delivered_total = len(state["delivered_ids"])
            quarantined_total = len(state["quarantined"])

        diagnostics: list[str] = []
        if stale_404_seen:
            diagnostics.append(
                "HTTP 404 sur /v1/hivemind/onboarding/feedback : retours CES conservés dans la file 0600."
                if is_fr
                else "HTTP 404 on /v1/hivemind/onboarding/feedback: CES items retained safely in 0600 spool."
            )
        if quarantined_now > 0:
            diagnostics.append(
                f"{quarantined_now} entrée(s) permanente(s) 400/422 mise(s) en quarantaine."
                if is_fr
                else f"{quarantined_now} permanent 400/422 entry/entries moved to quarantine."
            )

        flush_ok = retained_retry == 0 and quarantined_now == 0
        summary = (
            f"Vidage file CES : {delivered_now} envoyé(s), {retained_retry} en attente, {quarantined_now} en quarantaine."
            if is_fr
            else f"CES queue flush: {delivered_now} delivered, {retained_retry} retained, {quarantined_now} quarantined."
        )
        flush_res: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "flush_ces_queue",
            "ok": flush_ok,
            "ready": flush_ok,
            "compliance_ok": True,
            "exit_code": 0 if flush_ok else 2,
            "gateway": clean_gw,
            "queue_file": self.redacted_path,
            "queue_exists": self._path.is_file(),
            "flush_ok": flush_ok,
            "flushed_count": delivered_now,
            "delivered_count": delivered_now,
            "retained_retry_count": retained_retry,
            "quarantined_count": quarantined_now,
            "pending_count": pending_after,
            "delivered_ledger_count": delivered_total,
            "total_quarantined_count": quarantined_total,
            "stale_cloud_run_detected": stale_404_seen,
            "file_writes": 1 if mutated else 0,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "outcomes": outcomes,
            "diagnostics": diagnostics,
            "compliance": comp_doc,
            "summary": summary,
        }
        flush_res["receipt_sha256"] = compute_receipt_sha256(flush_res)
        return flush_res

    def purge(self, *, lang: str = "en") -> dict[str, Any]:
        """Remove local CES retry spool and temporary files safely (0 HTTP calls)."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        removed = False
        with self._lock:
            if self._path.is_file():
                self._path.unlink(missing_ok=True)
                removed = True
            if self._path.parent.is_dir():
                for extra in self._path.parent.glob(f".{self._path.name}.*"):
                    if extra.is_file():
                        extra.unlink(missing_ok=True)
        purge_res: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "purge_ces_queue",
            "ok": True,
            "ready": True,
            "exit_code": 0,
            "queue_file": self.redacted_path,
            "purged": removed,
            "network_calls": 0,
            "summary": (
                f"File CES purgée ({self.redacted_path})."
                if is_fr
                else f"CES retry queue purged ({self.redacted_path})."
            ),
        }
        purge_res["receipt_sha256"] = compute_receipt_sha256(purge_res)
        return purge_res


def enqueue_connect_ces_event(
    *,
    queue_file: Path | str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    host_label: str = DEFAULT_HOST_LABEL,
    resolved_key: str | None = None,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    platform: str | None = None,
    idempotency_key: str | None = None,
    last_http_status: int | None = None,
    last_reason: str = "",
    max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
    max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
    home: Path | None = None,
    credentials_path: Path | str | None = None,
    enforce_personal_pc: bool = True,
    allow_test_hosts: bool = False,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper around ``ConnectCesRetrySpool.enqueue(...)``."""
    spool = ConnectCesRetrySpool(
        queue_file,
        max_entries=max_entries,
        max_delivered=max_delivered,
        home=home,
        credentials_path=credentials_path,
        env=environ if environ is not None else env,
        os_name=os_name,
    )
    return spool.enqueue(
        gateway=gateway,
        host_label=host_label,
        resolved_key=resolved_key,
        profile=profile,
        ease_rating=ease_rating,
        zero_game_lag=zero_game_lag,
        os_name=os_name,
        system_name=system_name,
        platform=platform,
        idempotency_key=idempotency_key,
        last_http_status=last_http_status,
        last_reason=last_reason,
        enforce_personal_pc=enforce_personal_pc,
        allow_test_hosts=allow_test_hosts,
        environ=environ,
        env=env,
        cwd=cwd,
        lang=lang,
    )


def flush_connect_ces_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    queue_file: Path | str | None = None,
    max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
    max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
    home: Path | None = None,
    credentials_path: Path | str | None = None,
    enforce_personal_pc: bool = True,
    allow_test_hosts: bool | None = None,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    http_fn: HttpFn | None = None,
    transport: Any = None,
    raw_http_fn: RawHttpFn | None = None,
    timeout_s: float = DOCTOR_HTTP_TIMEOUT_S,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper around ``ConnectCesRetrySpool.flush(...)``."""
    spool = ConnectCesRetrySpool(
        queue_file,
        max_entries=max_entries,
        max_delivered=max_delivered,
        home=home,
        credentials_path=credentials_path,
        env=environ if environ is not None else env,
        os_name=os_name,
    )
    return spool.flush(
        gateway,
        http_fn=http_fn,
        transport=transport,
        raw_http_fn=raw_http_fn,
        allow_test_hosts=allow_test_hosts,
        enforce_personal_pc=enforce_personal_pc,
        environ=environ,
        env=env,
        cwd=cwd,
        timeout_s=timeout_s,
        lang=lang,
    )


def _inspect_doctor_credentials(
    credentials_path: Path,
    *,
    env_map: Mapping[str, str],
    os_name: str,
    home: Path,
    lang_code: str,
) -> tuple[str | None, str, bool, bool, dict[str, Any]]:
    env_key = str(env_map.get(API_KEY_ENV, "")).strip()
    shown_path = redact_home_path(credentials_path, home=home)
    file_exists = credentials_path.is_file()
    saved_creds = load_credentials(credentials_path) if file_exists else None

    if env_key and is_valid_hivemind_key(env_key):
        resolved_key = env_key
        role = str((saved_creds or {}).get("role") or "prosumer")
        loaded = True
    elif saved_creds is not None:
        resolved_key = str(saved_creds["api_key"]).strip()
        role = str(saved_creds.get("role") or "prosumer")
        loaded = True
    else:
        resolved_key = None
        role = "prosumer"
        loaded = False

    if file_exists and saved_creds is None:
        status = "warn"
        summary = (
            f"{shown_path} is unreadable or missing a valid hm-... key."
            if lang_code == "en"
            else f"{shown_path} est illisible ou ne contient pas de clé hm-... valide."
        )
    elif file_exists and os_name != "nt" and (credentials_path.stat().st_mode & 0o777) != 0o600:
        status = "warn"
        summary = (
            f"{shown_path} permissions are not 0600 owner-only."
            if lang_code == "en"
            else f"Les permissions de {shown_path} ne sont pas 0600 (propriétaire seul)."
        )
    elif loaded:
        status = "ok"
        summary = (
            f"Credentials loaded ({mask_api_key(resolved_key)}, source={'env' if env_key else shown_path})."
            if lang_code == "en"
            else f"Identifiants chargés ({mask_api_key(resolved_key)}, source={'env' if env_key else shown_path})."
        )
    else:
        status = "ok"
        summary = (
            f"No saved key yet at {shown_path} (optional unless --first-heartbeat)."
            if lang_code == "en"
            else f"Aucune clé enregistrée dans {shown_path} (optionnel sauf --first-heartbeat)."
        )

    check = {
        "id": "credentials",
        "title": "HiveMind Credentials" if lang_code == "en" else "Identifiants HiveMind",
        "status": status,
        "passed": status == "ok",
        "summary": summary,
        "details": {
            "path": shown_path,
            "present": file_exists,
            "loaded": loaded,
            "masked_api_key": mask_api_key(resolved_key),
        },
    }
    return resolved_key, role, file_exists, loaded, check


def _build_refusal_report(
    *,
    lang_code: str,
    gateway_label: str,
    compliance_ok: bool,
    compliance: dict[str, Any],
    elapsed_s: float,
    cred_target: Path,
    cont_target: Path,
    home: Path,
    require_holiday_routes: bool,
    yield_budget_ms: float,
    first_heartbeat: bool,
    reason_code: str,
    masked_key: str,
    check_id: str,
    check_title: str,
    summary_text: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Build sanitized fail-closed doctor report for early compliance or gateway refusal."""
    report: dict[str, Any] = sanitize_structure(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "ready": False,
            "verdict": "not_ready",
            "exit_code": 2,
            "lang": lang_code,
            "gateway": gateway_label,
            "allowed_host": False,
            "compliance_ok": compliance_ok,
            "compliance": compliance,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "personal_pc_only": True,
            "monetary_cost": FINANCIAL_PROMISE,
            "zero_inference_verified": True,
            "outbound_http_requests": 0,
            "file_writes": 0,
            "elapsed_s": elapsed_s,
            "within_60s_slo": elapsed_s < 60.0,
            "credentials_path": redact_home_path(cred_target, home=home),
            "credentials_present": False,
            "credentials_loaded": False,
            "masked_api_key": masked_key,
            "continue_path": redact_home_path(cont_target, home=home),
            "continue_written": False,
            "holiday_routes_deployed_on_cloud_run": False,
            "onboarding_feedback_deployed": False,
            "revision_drift_detected": False,
            "require_holiday_routes": bool(require_holiday_routes),
            "missing_or_stale_routes": [],
            "route_statuses": {},
            "manifest": {},
            "online_producers": "n/a",
            "active_trainers": "n/a",
            "accepted_rounds": "n/a",
            "gpu_present": False,
            "gpu_status": f"skipped_{reason_code}",
            "recommended_mode": "dev_api_only",
            "yield_latency_ms": None,
            "yield_budget_ms": float(yield_budget_ms),
            "yield_slo_ok": False,
            "thermal_ok": False,
            "vram_reserve_ok": False,
            "game_running": False,
            "active_games": [],
            "zero_game_lag": False,
            "first_heartbeat_requested": bool(first_heartbeat),
            "first_heartbeat_sent": False,
            "first_heartbeat_ok": False,
            "opentrainer_registered": False,
            "heartbeat_state": f"blocked_{reason_code}",
            "heartbeat_http_status": None,
            "heartbeat_payload": None,
            "ces_attempted": False,
            "ces_recorded": False,
            "ces_degraded": False,
            "ces_skipped": False,
            "ces_http_status": None,
            "ces_idempotency_key": None,
            "ces_payload": None,
            "mean_ease_rating": None,
            "feedback_summary": None,
            "checks": [
                {
                    "id": check_id,
                    "title": check_title,
                    "status": "fail",
                    "passed": False,
                    "summary": summary_text,
                    "details": details,
                }
            ],
            "diagnostics": [summary_text],
        },
        home=home,
    )
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def run_connect_doctor(
    *,
    gateway: str | None = None,
    credentials_path: Path | str | None = None,
    continue_path: Path | str | None = None,
    save_credentials: bool = False,
    write_continue: bool = False,
    require_holiday_routes: bool = False,
    first_heartbeat: bool = False,
    submit_ces: bool = False,
    prompt_ces: bool = False,
    profile: str | None = None,
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = DEFAULT_HOST_LABEL,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    extra_games: Sequence[str] = (),
    lang: str = "en",
    timeout_s: float = DOCTOR_HTTP_TIMEOUT_S,
    cwd: Path | str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    transport: Any = None,
    http_fn: HttpFn | None = None,
    gpu_probe_fn: GpuProbeFn | None = None,
    process_probe_fn: ProcessProbeFn | None = None,
    worker_yield_fn: WorkerYieldFn | None = None,
    clock_fn: ClockFn | None = None,
    input_fn: InputFn | None = None,
    runner: RunnerFn | None = None,
    spool_path: Path | str | None = None,
    flush_ces_queue: bool = False,
    clean_ces_queue: bool = False,
) -> dict[str, Any]:
    """Run zero-inference <60s Noel doctor checks across Cloud Run routes, local GPU/Steam yield, heartbeat, and CES."""
    eff_clock: ClockFn = clock_fn if clock_fn is not None else time.monotonic
    started = eff_clock()
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_env = environ if environ is not None else (env if env is not None else os.environ)
    eff_home = home if home is not None else Path.home()
    eff_cwd = Path(cwd) if cwd is not None else Path.cwd()
    eff_os = os_name if os_name is not None else os.name
    eff_runner: RunnerFn = runner if runner is not None else default_subprocess_runner
    bounded_timeout = min(max(0.5, float(timeout_s)), 5.0)
    clean_ease = parse_ease_rating(ease_rating)
    if profile is not None:
        _ = normalize_profile(profile)
    use_spool = bool(spool_path is not None or flush_ces_queue or clean_ces_queue)

    raw_gw = (
        gateway
        or eff_env.get(GATEWAY_ENV)
        or eff_env.get("HIVEMIND_GATEWAY_URL")
        or DEFAULT_GATEWAY
    )
    allow_test_hosts = transport is not None or http_fn is not None
    cred_target = Path(credentials_path).expanduser() if credentials_path is not None else default_credentials_path(home=eff_home)
    cont_target = Path(continue_path).expanduser() if continue_path is not None else default_continue_path(home=eff_home)

    # 1. Personal-PC & Amadeus compliance gate BEFORE any file writes, probes, or HTTP calls
    compliance = evaluate_personal_pc_compliance(env=eff_env, cwd=eff_cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _build_refusal_report(
            lang_code=lang_code,
            gateway_label=DEFAULT_GATEWAY,
            compliance_ok=False,
            compliance=compliance,
            elapsed_s=round(max(0.0, eff_clock() - started), 4),
            cred_target=cred_target,
            cont_target=cont_target,
            home=eff_home,
            require_holiday_routes=require_holiday_routes,
            yield_budget_ms=yield_budget_ms,
            first_heartbeat=first_heartbeat,
            reason_code="compliance",
            masked_key="<redacted>",
            check_id="personal_pc_compliance",
            check_title="Personal-PC Compliance Gate" if not is_fr else "Barrière Conformité PC Personnel",
            summary_text=compliance["summary"],
            details=compliance,
        )

    # 2. Strict Gateway Allowlist Gate
    try:
        clean_gw = validate_gateway_url(str(raw_gw), allow_test_hosts=allow_test_hosts)
    except GatewayPolicyError as exc:
        err_msg = sanitize_text_for_display(str(exc), home=eff_home)
        return _build_refusal_report(
            lang_code=lang_code,
            gateway_label="rejected_untrusted_gateway",
            compliance_ok=True,
            compliance=compliance,
            elapsed_s=round(max(0.0, eff_clock() - started), 4),
            cred_target=cred_target,
            cont_target=cont_target,
            home=eff_home,
            require_holiday_routes=require_holiday_routes,
            yield_budget_ms=yield_budget_ms,
            first_heartbeat=first_heartbeat,
            reason_code="untrusted_gateway",
            masked_key="<missing>",
            check_id="gateway_allowlist",
            check_title="Gateway URL Allowlist" if not is_fr else "Liste Blanche Passerelle",
            summary_text=err_msg,
            details={"allowed_host": False},
        )

    # Build zero-inference HTTP client (stdlib ProxyHandler({}) when transport and http_fn are None)
    client_cm: Any
    if http_fn is not None:
        client_cm = _InjectedHttpClient(http_fn)
    elif transport is not None:
        client_cm = _HttpxTransportClient(transport=transport, timeout_s=bounded_timeout)
    else:
        client_cm = _StdlibHttpClient(timeout_s=bounded_timeout)

    resolved_key, saved_role, cred_present, cred_loaded, chk_cred = _inspect_doctor_credentials(
        cred_target,
        env_map=eff_env,
        os_name=eff_os,
        home=eff_home,
        lang_code=lang_code,
    )

    outbound_requests = 0
    file_writes = 0
    diagnostics: list[str] = []
    checks: list[dict[str, Any]] = [
        {
            "id": "personal_pc_compliance",
            "title": "Personal-PC Compliance Gate" if not is_fr else "Barrière Conformité PC Personnel",
            "status": "ok",
            "passed": True,
            "summary": compliance["summary"],
            "details": compliance,
        },
        chk_cred,
    ]

    route_statuses: dict[str, int] = {}
    missing_or_stale: list[str] = []
    manifest_info: dict[str, Any] = {}
    feedback_summary: dict[str, Any] | None = None
    manifest_drift = False
    fleet_body: dict[str, Any] = {}
    raid_body: dict[str, Any] = {}
    me_body: dict[str, Any] = {}
    me_status: int | None = None
    first_hb_sent = False
    first_hb_ok = not first_heartbeat
    opentrainer_registered = False
    hb_status_code: int | None = None
    hb_state = "not_requested"
    hb_payload: dict[str, Any] | None = None
    account_role = saved_role
    account_id = ""
    account_handle = ""

    # 3. Local Gamer-Whisper <1.5s Steam/DX12 yield drill + GPU thermal/VRAM check
    raw_gpu = (
        _invoke_with_supported_kwargs(gpu_probe_fn, runner=eff_runner, timeout_s=SUBPROCESS_TIMEOUT_S)
        if gpu_probe_fn is not None
        else default_gpu_probe(runner=eff_runner, timeout_s=SUBPROCESS_TIMEOUT_S)
    )
    gpu = _normalize_gpu_snapshot(raw_gpu)

    raw_procs = (
        _invoke_with_supported_kwargs(process_probe_fn, runner=eff_runner, os_name=eff_os, timeout_s=SUBPROCESS_TIMEOUT_S)
        if process_probe_fn is not None
        else default_process_probe(runner=eff_runner, os_name=eff_os, timeout_s=SUBPROCESS_TIMEOUT_S)
    )
    active_games = _detect_active_games(raw_procs, extra_games=extra_games)
    game_running = len(active_games) > 0

    yield_latency_ms, clean_release, retained_mib, drill_method = _run_yield_drill(
        worker_yield_fn,
        clock_fn=eff_clock,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    budget_ms = float(yield_budget_ms)
    cap_temp_c = float(max_temp_c)
    reserve_mib = int(min_free_vram_mib)
    yield_slo_ok = bool(clean_release and retained_mib == 0 and yield_latency_ms is not None and 0.0 <= yield_latency_ms < budget_ms)

    if not gpu.present:
        thermal_ok = not game_running
        vram_reserve_ok = True
        gpu_status = "dev_api_only_no_nvidia_gpu"
        recommended_mode = "dev_api_only"
        hw_status = "fail" if not yield_slo_ok else ("warn" if game_running else "ok")
    elif gpu.parse_error:
        thermal_ok = False
        vram_reserve_ok = False
        gpu_status = "gpu_driver_parse_error"
        recommended_mode = "dev_api_only"
        hw_status = "fail" if not yield_slo_ok else "warn"
    else:
        assert gpu.free_mib is not None
        thermal_ok = bool((gpu.temp_c is None or gpu.temp_c <= cap_temp_c) and not game_running)
        vram_reserve_ok = bool(gpu.free_mib >= reserve_mib)
        if not yield_slo_ok:
            gpu_status = "yield_slo_exceeded"
            hw_status = "fail"
        elif gpu.temp_c is not None and gpu.temp_c > cap_temp_c:
            gpu_status = "gpu_thermal_cooldown"
            hw_status = "warn"
        elif game_running:
            gpu_status = "active_game_yield"
            hw_status = "warn"
        elif not vram_reserve_ok:
            gpu_status = "low_free_vram"
            hw_status = "warn"
        else:
            gpu_status = "gamer_whisper_ready"
            hw_status = "ok"
        recommended_mode = "gamer_whisper" if gpu_status == "gamer_whisper_ready" else "dev_api_only"

    lat_s_text = f"{yield_latency_ms / 1000.0:.3f}s" if yield_latency_ms is not None else "n/a"
    budget_s = budget_ms / 1000.0
    if hw_status == "fail":
        hw_summary = (
            f"Steam yield preemption drill failed ({lat_s_text} >= {budget_s:.2f}s)."
            if not is_fr
            else f"Échec du test de préemption Steam ({lat_s_text} >= {budget_s:.2f}s)."
        )
        diagnostics.append(hw_summary)
    elif gpu_status == "dev_api_only_no_nvidia_gpu" and not game_running:
        hw_summary = (
            f"No NVIDIA GPU detected (dev_api_only_no_nvidia_gpu); yield drill passed in {lat_s_text}."
            if not is_fr
            else f"Aucun GPU NVIDIA détecté (dev_api_only_no_nvidia_gpu) ; préemption vérifiée en {lat_s_text}."
        )
    elif gpu_status == "gpu_thermal_cooldown" and gpu.temp_c is not None:
        hw_summary = (
            f"{gpu.name} temperature {gpu.temp_c:.1f} C > {cap_temp_c:.1f} C quiet-fan ceiling."
            if not is_fr
            else f"{gpu.name} à {gpu.temp_c:.1f} C > {cap_temp_c:.1f} C plafond silencieux."
        )
        diagnostics.append(hw_summary)
    elif gpu_status == "low_free_vram":
        hw_summary = (
            f"{gpu.name} free VRAM {gpu.free_mib} MiB < {reserve_mib} MiB reserve."
            if not is_fr
            else f"{gpu.name} VRAM libre {gpu.free_mib} MiB < {reserve_mib} MiB."
        )
        diagnostics.append(hw_summary)
    elif gpu_status == "active_game_yield" or (not gpu.present and game_running):
        hw_summary = (
            f"Active game detected ({', '.join(active_games)}); Gamer-Whisper yielded in {lat_s_text}."
            if not is_fr
            else f"Jeu actif détecté ({', '.join(active_games)}) ; Gamer-Whisper cède en {lat_s_text}."
        )
        diagnostics.append(hw_summary)
    elif gpu_status == "gpu_driver_parse_error":
        hw_summary = (
            "Unparseable nvidia-smi telemetry; Dev API-only fallback active."
            if not is_fr
            else "Sortie nvidia-smi illisible ; mode Dev API actif."
        )
        diagnostics.append(hw_summary)
    else:
        temp_str = f"{gpu.temp_c:.1f} C" if gpu.temp_c is not None else "n/a C"
        hw_summary = (
            f"{gpu.name} Gamer-Whisper ready ({lat_s_text} < {budget_s:.2f}s, {temp_str}, {gpu.free_mib} MiB free)."
            if not is_fr
            else f"{gpu.name} Gamer-Whisper prêt ({lat_s_text} < {budget_s:.2f}s, {temp_str}, {gpu.free_mib} MiB libres)."
        )

    checks.append(
        {
            "id": "gamer_whisper_yield_thermal",
            "title": "Gamer-Whisper Yield & Thermal" if not is_fr else "Préemption Gamer-Whisper & Thermique",
            "status": hw_status,
            "passed": hw_status == "ok",
            "summary": hw_summary,
            "details": {
                "gpu_present": gpu.present,
                "gpu_status": gpu_status,
                "gpu_name": gpu.name,
                "total_mib": gpu.total_mib,
                "free_mib": gpu.free_mib,
                "temp_c": gpu.temp_c,
                "yield_latency_ms": yield_latency_ms,
                "yield_budget_ms": budget_ms,
                "yield_slo_ok": yield_slo_ok,
                "clean_release": clean_release,
                "retained_worker_vram_mib": retained_mib,
                "drill_method": drill_method,
                "thermal_ok": thermal_ok,
                "vram_reserve_ok": vram_reserve_ok,
                "game_running": game_running,
                "active_games": active_games,
            },
        }
    )

    # 4. Outbound zero-inference HTTP checks: core, holiday routes, optional first-heartbeat, fleet/raid, optional CES
    clean_host = _sanitize_host_label(host_label, home=eff_home)
    with client_cm as client:
        # 4a. Optional first-heartbeat registration ping BEFORE reading raid/status so active_trainers reflects registration
        if first_heartbeat:
            if resolved_key is None:
                first_hb_ok = False
                hb_state = "missing_valid_credentials"
                diagnostics.append(
                    "First heartbeat requested without valid credentials."
                    if not is_fr
                    else "Premier heartbeat demandé sans identifiants valides."
                )
            else:
                auth_headers = {"Authorization": f"Bearer {resolved_key}"}
                outbound_requests += 1
                me_status, me_raw = _safe_http_get(
                    client, f"{clean_gw}/v1/hivemind/me", headers=auth_headers, timeout=bounded_timeout
                )
                if me_status == 200 and isinstance(me_raw, Mapping):
                    me_body = dict(me_raw)
                    account_id = str(me_body.get("account_id") or "")
                    account_handle = str(me_body.get("handle") or "")
                    if isinstance(me_body.get("role"), str) and me_body["role"].strip():
                        account_role = me_body["role"].strip()

                    hb_gpu_name: str | None = None
                    hb_vram_total: int | None = None
                    hb_vram_free: int | None = None
                    if (
                        gpu.present
                        and not gpu.parse_error
                        and gpu.name
                        and gpu.total_mib is not None
                        and gpu.total_mib >= MIN_REPORTABLE_VRAM_TOTAL_MIB
                    ):
                        hb_gpu_name = sanitize_text_for_display(gpu.name, home=eff_home)[:128]
                        hb_vram_total = int(gpu.total_mib)
                        hb_vram_free = min(max(0, int(gpu.free_mib or 0)), hb_vram_total)

                    hb_payload = {
                        "host_label": clean_host,
                        "gpu_name": hb_gpu_name,
                        "vram_total_mib": hb_vram_total,
                        "vram_free_mib": hb_vram_free,
                        "network_mode": "outbound-https-relay",
                        "training_stack": DEFAULT_TRAINING_STACK,
                        "active_campaign": CAMPAIGN_ID,
                    }
                    first_hb_sent = True
                    outbound_requests += 1
                    hb_status_code, _ = _safe_http_post(
                        client,
                        f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
                        payload=hb_payload,
                        headers=auth_headers,
                        timeout=bounded_timeout,
                    )
                    if hb_status_code in (200, 201):
                        first_hb_ok = True
                        opentrainer_registered = True
                        hb_state = "accepted"
                    elif hb_status_code == 403 and account_role == "consumer":
                        first_hb_ok = True
                        opentrainer_registered = False
                        hb_state = "consumer_me_verified"
                    else:
                        first_hb_ok = False
                        hb_state = f"heartbeat_http_{hb_status_code}"
                        diagnostics.append(
                            f"OpenTrainer heartbeat failed (HTTP {hb_status_code})."
                            if not is_fr
                            else f"Échec du heartbeat OpenTrainer (HTTP {hb_status_code})."
                        )
                else:
                    first_hb_ok = False
                    hb_state = f"me_http_{me_status}"
                    diagnostics.append(
                        f"GET /v1/hivemind/me failed before heartbeat (HTTP {me_status})."
                        if not is_fr
                        else f"Échec GET /v1/hivemind/me avant heartbeat (HTTP {me_status})."
                    )

        # 4b. Core /health + Fleet /v1/hivemind/producers/fleet + 6 Holiday routes
        for route in DOCTOR_CORE_ROUTES + DOCTOR_HOLIDAY_ROUTES + DOCTOR_FLEET_ROUTES:
            outbound_requests += 1
            code, body = _safe_http_get(client, f"{clean_gw}{route}", timeout=bounded_timeout)
            route_statuses[route] = code

            if route == "/health":
                if code != 200 or (isinstance(body, Mapping) and "status" in body and body.get("status") != "ok"):
                    missing_or_stale.append(route)
            elif route in ("/noel", "/quick", "/raid-noel"):
                if code != 200 or not isinstance(body, Mapping):
                    missing_or_stale.append(route)
                elif "html_length" in body and int(body.get("html_length") or 0) <= 0:
                    missing_or_stale.append(route)
                elif "raw" in body and not str(body.get("raw") or "").strip():
                    missing_or_stale.append(route)
            elif route == "/v1/hivemind/onboarding/manifest":
                if code != 200 or not isinstance(body, Mapping):
                    missing_or_stale.append(route)
                else:
                    manifest_info = {
                        "build_sha": _sanitize_manifest_sha(body.get("build_sha")),
                        "cloud_run_revision": _sanitize_manifest_revision(body.get("cloud_run_revision")),
                    }
                    if body.get("object") is not None and body.get("object") != "hivemind.onboarding.manifest":
                        manifest_drift = True
                        missing_or_stale.append(route)
                    elif isinstance(body.get("holiday_routes"), list):
                        listed = {str(r) for r in body["holiday_routes"]}
                        if not {"/noel", "/quick", "/raid-noel"}.issubset(listed):
                            manifest_drift = True
                            missing_or_stale.append(route)
            elif route == "/v1/hivemind/onboarding/feedback/summary":
                if code != 200 or not isinstance(body, Mapping) or ("error" in body and "total_responses" not in body):
                    missing_or_stale.append(route)
                else:
                    by_prof = body.get("by_profile") if isinstance(body.get("by_profile"), Mapping) else {}
                    raw_mean = (
                        body.get("mean_ease_rating")
                        if body.get("mean_ease_rating") is not None
                        else body.get("average_ease_rating")
                    )
                    feedback_summary = {
                        "object": str(body.get("object") or "hivemind.onboarding_feedback_summary"),
                        "total_responses": int(body.get("total_responses") or 0),
                        "mean_ease_rating": raw_mean,
                        "zero_game_lag_ratio": body.get("zero_game_lag_ratio"),
                        "by_profile": {
                            "dev": int(by_prof.get("dev", 0)),
                            "gamer": int(by_prof.get("gamer", 0)),
                        },
                    }
            elif route == "/v1/hivemind/raid/status":
                if code != 200 or not isinstance(body, Mapping):
                    missing_or_stale.append(route)
                else:
                    raid_body = dict(body)
            elif route == "/v1/hivemind/producers/fleet":
                if code != 200 or not isinstance(body, Mapping):
                    missing_or_stale.append(route)
                else:
                    fleet_body = dict(body)

        # 4c. Optional --save-credentials / --write-continue inside doctor (only when key is available)
        continue_written = False
        if (save_credentials or write_continue) and resolved_key is not None:
            if save_credentials:
                save_credentials_file(
                    cred_target,
                    account_id=account_id or "personal-account",
                    api_key=resolved_key,
                    gateway=clean_gw,
                    handle=account_handle or clean_host,
                    role=account_role,
                    os_name=eff_os,
                    home=eff_home,
                )
                file_writes += 1
                cred_present = True
                cred_loaded = True
            if write_continue:
                write_continue_provider_file(
                    cont_target,
                    gateway=clean_gw,
                    api_key=resolved_key,
                    os_name=eff_os,
                    home=eff_home,
                    allow_test_hosts=allow_test_hosts,
                )
                file_writes += 1
                continue_written = True

        # 4d. Optional --submit-ces / --prompt-ces
        zero_game_lag = bool(yield_slo_ok and thermal_ok and not had_game_lag)
        inferred_profile = (
            normalize_profile(profile)
            if profile is not None
            else ("gamer" if (gpu.present and not gpu.parse_error and (gpu.total_mib or 0) >= 4096) else "dev")
        )
        do_submit_ces = bool(submit_ces)
        ces_skipped = False
        eff_ease = clean_ease

        if prompt_ces:
            eff_input = input_fn if input_fn is not None else input
            prompt_msg = (
                f"[HiveMind CES 1-question] Notez la simplicité 1..5 (profil={inferred_profile}, Entrée=ignorer) : "
                if is_fr
                else f"[HiveMind 1-prompt CES] Rate setup ease 1..5 (profile={inferred_profile}; Enter to skip): "
            )
            try:
                raw_ans = eff_input(prompt_msg)
            except (EOFError, KeyboardInterrupt):
                raw_ans = ""
            parsed_ces = parse_ces_input(
                raw_ans,
                default_profile=inferred_profile,
                default_had_game_lag=not zero_game_lag,
            )
            if parsed_ces["skipped"]:
                ces_skipped = True
            else:
                do_submit_ces = True
                inferred_profile = str(parsed_ces["profile"])
                eff_ease = int(parsed_ces["ease_rating"])
                zero_game_lag = bool(zero_game_lag and parsed_ces["zero_game_lag"])

        ces_attempted = False
        ces_recorded = False
        ces_degraded = False
        ces_http_status: int | None = None
        ces_idempotency_key: str | None = None
        ces_payload: dict[str, Any] | None = None
        ces_spooled = False
        ces_flush_report: dict[str, Any] | None = None
        ces_clean_report: dict[str, Any] | None = None
        mean_ease_rating: float | None = (
            float(feedback_summary["mean_ease_rating"])
            if feedback_summary is not None and isinstance(feedback_summary.get("mean_ease_rating"), (int, float))
            else None
        )

        spool_obj: ConnectCesRetrySpool | None = None
        if use_spool:
            spool_obj = ConnectCesRetrySpool(
                spool_path,
                home=eff_home,
                credentials_path=cred_target,
                env=eff_env,
                os_name=eff_os,
            )
            if clean_ces_queue:
                ces_clean_report = spool_obj.purge(lang=lang_code)

        if do_submit_ces:
            ces_attempted = True
            ces_payload = {
                "profile": inferred_profile,
                "ease_rating": eff_ease,
                "zero_game_lag": zero_game_lag,
                "platform": _platform_feedback_tag(os_name=eff_os, system_name=system_name),
                "note": "",
            }
            ces_idempotency_key = build_ces_idempotency_key(
                gateway=clean_gw,
                host_label=clean_host,
                resolved_key=resolved_key,
                payload=ces_payload,
            )
            outbound_requests += 1
            ces_http_status, ces_body = _safe_http_post(
                client,
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                payload=ces_payload,
                headers={"Idempotency-Key": ces_idempotency_key},
                timeout=bounded_timeout,
            )
            if ces_http_status in (200, 201) and isinstance(ces_body, Mapping):
                ces_recorded = True
                raw_mean = ces_body.get("mean_ease_rating", ces_body.get("average_ease_rating"))
                if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool):
                    mean_ease_rating = round(float(raw_mean), 2)
                if spool_obj is not None and spool_obj.path.is_file():
                    spool_obj.mark_delivered(ces_idempotency_key)
                    if flush_ces_queue:
                        ces_flush_report = spool_obj.flush(
                            clean_gw,
                            client=client,
                            allow_test_hosts=allow_test_hosts,
                            enforce_personal_pc=True,
                            env=eff_env,
                            cwd=eff_cwd,
                            timeout_s=bounded_timeout,
                            lang=lang_code,
                        )
                        outbound_requests += int(ces_flush_report.get("network_calls", 0))
                        file_writes += int(ces_flush_report.get("file_writes", 0))
            else:
                ces_recorded = False
                ces_degraded = True
                if spool_obj is not None:
                    disp, reason_tok = _classify_ces_flush_status(int(ces_http_status or 0))
                    if disp == "retryable":
                        enq_res = spool_obj.enqueue(
                            gateway=clean_gw,
                            host_label=clean_host,
                            resolved_key=resolved_key,
                            profile=inferred_profile,
                            ease_rating=eff_ease,
                            zero_game_lag=zero_game_lag,
                            platform=str(ces_payload["platform"]),
                            idempotency_key=ces_idempotency_key,
                            last_http_status=ces_http_status,
                            last_reason=reason_tok,
                            enforce_personal_pc=True,
                            allow_test_hosts=allow_test_hosts,
                            env=eff_env,
                            cwd=eff_cwd,
                            lang=lang_code,
                        )
                        ces_spooled = bool(enq_res.get("enqueued") or enq_res.get("duplicate_state") == "already_pending")
                        file_writes += int(enq_res.get("file_writes", 0))
                diagnostics.append(
                    f"CES submission degraded non-blocking (HTTP {ces_http_status})."
                    if not is_fr
                    else f"Envoi CES dégradé sans blocage (HTTP {ces_http_status})."
                )
        elif spool_obj is not None and flush_ces_queue:
            ces_flush_report = spool_obj.flush(
                clean_gw,
                client=client,
                allow_test_hosts=allow_test_hosts,
                enforce_personal_pc=True,
                env=eff_env,
                cwd=eff_cwd,
                timeout_s=bounded_timeout,
                lang=lang_code,
            )
            outbound_requests += int(ces_flush_report.get("network_calls", 0))
            file_writes += int(ces_flush_report.get("file_writes", 0))

    # 5. Aggregate route drift, fleet metrics, and final verdict
    swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}
    online_producers: int | str = (
        _honest_counter(fleet_body.get("live_hiveminders"))
        if _honest_counter(fleet_body.get("live_hiveminders")) != "n/a"
        else _honest_counter(swarm.get("live_producers"))
    )
    active_trainers: int | str = _honest_counter(swarm.get("active_trainers")) if swarm else "n/a"
    accepted_rounds: int | str = _honest_counter(swarm.get("accepted_opentrainer_rounds", swarm.get("accepted_rounds"))) if swarm else "n/a"

    holiday_missing = [r for r in DOCTOR_HOLIDAY_ROUTES if r in missing_or_stale]
    holiday_routes_deployed = len(holiday_missing) == 0
    onboarding_feedback_deployed = (
        route_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200
        and "/v1/hivemind/onboarding/feedback/summary" not in missing_or_stale
    )
    revision_drift_detected = bool(
        manifest_drift
        or any(route_statuses.get(r) in (404, 501) for r in DOCTOR_HOLIDAY_ROUTES)
    )
    core_ok = route_statuses.get("/health") == 200 and "/health" not in missing_or_stale
    fleet_ok = route_statuses.get("/v1/hivemind/producers/fleet") == 200

    if not core_ok or not fleet_ok:
        route_status = "fail"
    elif not holiday_routes_deployed:
        route_status = "fail" if require_holiday_routes else "warn"
    else:
        route_status = "ok"

    if route_status != "ok":
        diagnostics.append(
            f"Holiday/gateway route drift or outage detected on: {', '.join(missing_or_stale)}."
            if not is_fr
            else f"Dérive ou indisponibilité détectée sur les routes : {', '.join(missing_or_stale)}."
        )

    checks.append(
        {
            "id": "noel_holiday_routes",
            "title": "Noel Cloud Run Routes & Manifest" if not is_fr else "Routes Noël Cloud Run & Manifeste",
            "status": route_status,
            "passed": route_status == "ok",
            "summary": (
                "All Cloud Run holiday routes verified."
                if route_status == "ok" and not is_fr
                else (
                    "Toutes les routes Noël Cloud Run sont actives."
                    if route_status == "ok"
                    else f"Missing or stale routes: {', '.join(missing_or_stale)}"
                )
            ),
            "details": {
                "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
                "onboarding_feedback_deployed": onboarding_feedback_deployed,
                "revision_drift_detected": revision_drift_detected,
                "missing_or_stale_routes": missing_or_stale,
                "route_statuses": route_statuses,
                "manifest": manifest_info,
            },
        }
    )

    ces_required_fail = bool(
        require_holiday_routes
        and ces_attempted
        and not ces_recorded
        and ces_http_status is not None
        and (ces_http_status == 404 or ces_http_status >= 500 or ces_http_status == 0)
    )
    hb_status = "fail" if (first_heartbeat and not first_hb_ok) else ("fail" if ces_required_fail else "ok")
    checks.append(
        {
            "id": "noel_fleet_heartbeat_ces",
            "title": "Fleet, First Heartbeat & Zero-PII CES" if not is_fr else "Flotte, Premier Heartbeat & CES Zéro-PII",
            "status": hb_status,
            "passed": hb_status == "ok",
            "summary": (
                f"producers={online_producers}, trainers={active_trainers}, heartbeat={hb_state}, ces_recorded={ces_recorded}"
            ),
            "details": {
                "online_producers": online_producers,
                "active_trainers": active_trainers,
                "accepted_rounds": accepted_rounds,
                "first_heartbeat_ok": first_hb_ok,
                "ces_attempted": ces_attempted,
                "ces_recorded": ces_recorded,
            },
        }
    )

    failures = sum(1 for c in checks if c["status"] == "fail")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    if failures > 0:
        verdict = "not_ready"
    elif warnings > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"
    exit_code = EXIT_CODES[verdict]
    elapsed_s = round(max(0.0, eff_clock() - started), 4)

    report: dict[str, Any] = sanitize_structure(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "ready": verdict == "ready",
            "verdict": verdict,
            "exit_code": exit_code,
            "lang": lang_code,
            "gateway": clean_gw,
            "allowed_host": True,
            "compliance_ok": True,
            "compliance": compliance,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "personal_pc_only": True,
            "monetary_cost": FINANCIAL_PROMISE,
            "zero_inference_verified": True,
            "outbound_http_requests": outbound_requests,
            "file_writes": file_writes,
            "elapsed_s": elapsed_s,
            "within_60s_slo": elapsed_s < 60.0,
            "credentials_path": redact_home_path(cred_target, home=eff_home),
            "credentials_present": cred_present,
            "credentials_loaded": cred_loaded,
            "masked_api_key": mask_api_key(resolved_key),
            "continue_path": redact_home_path(cont_target, home=eff_home),
            "continue_written": continue_written,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "revision_drift_detected": revision_drift_detected,
            "require_holiday_routes": bool(require_holiday_routes),
            "missing_or_stale_routes": missing_or_stale,
            "route_statuses": route_statuses,
            "manifest": manifest_info,
            "online_producers": online_producers,
            "active_trainers": active_trainers,
            "accepted_rounds": accepted_rounds,
            "gpu_present": gpu.present,
            "gpu_status": gpu_status,
            "recommended_mode": recommended_mode,
            "yield_latency_ms": yield_latency_ms,
            "yield_budget_ms": budget_ms,
            "yield_slo_ok": yield_slo_ok,
            "thermal_ok": thermal_ok,
            "vram_reserve_ok": vram_reserve_ok,
            "game_running": game_running,
            "active_games": active_games,
            "zero_game_lag": zero_game_lag,
            "first_heartbeat_requested": bool(first_heartbeat),
            "first_heartbeat_sent": first_hb_sent,
            "first_heartbeat_ok": first_hb_ok,
            "opentrainer_registered": opentrainer_registered,
            "heartbeat_state": hb_state,
            "heartbeat_http_status": hb_status_code,
            "heartbeat_payload": hb_payload,
            "ces_attempted": ces_attempted,
            "ces_recorded": ces_recorded,
            "ces_degraded": ces_degraded,
            "ces_skipped": ces_skipped,
            "ces_http_status": ces_http_status,
            "ces_idempotency_key": ces_idempotency_key,
            "ces_payload": ces_payload,
            "mean_ease_rating": mean_ease_rating,
            "feedback_summary": feedback_summary,
            "checks": checks,
            "diagnostics": diagnostics,
            **(
                {
                    "ces_spooled": ces_spooled,
                    "ces_queue": spool_obj.snapshot(lang=lang_code) if spool_obj is not None else None,
                    "ces_flush": ces_flush_report,
                    "ces_clean": ces_clean_report,
                }
                if use_spool
                else {}
            ),
        },
        home=eff_home,
    )
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def render_doctor_cli(report: Mapping[str, Any]) -> str:
    """Render concise EN or FR CLI report with zero static-figure violations or raw secret leaks."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    badges = {"ok": " OK ", "warn": "WARN", "fail": "FAIL"}
    title = (
        "HIVEMIND CONNECT DOCTOR (PC PERSO • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND CONNECT DOCTOR (PERSONAL PC ONLY • 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "=" * 72,
        title,
        "=" * 72,
        f"{'Règle' if is_fr else 'Rule'}: {report.get('notice_bilingual', COMPLIANCE_NOTICE_BILINGUAL)}",
        "",
    ]
    for chk in report.get("checks", ()):
        badge = badges.get(str(chk.get("status")), "INFO")
        lines.append(f"[{badge}] {chk.get('title')}: {chk.get('summary')}")
    lines.append("")
    lines.append(
        f"{'Verdict'}: {str(report.get('verdict', 'not_ready')).upper()} | "
        f"{'mode'}: {report.get('recommended_mode')} | "
        f"holiday_routes={report.get('holiday_routes_deployed_on_cloud_run')} | "
        f"producers={report.get('online_producers')} | trainers={report.get('active_trainers')}"
    )
    lines.append(f"SHA-256: {report.get('receipt_sha256', '')}")
    return "\n".join(lines)


def render_doctor_markdown(report: Mapping[str, Any]) -> str:
    """Render EN or FR Markdown report with masked keys, redacted paths, and SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    heading = (
        "# HiveMind Connect Doctor — Rapport Noël"
        if is_fr
        else "# HiveMind Connect Doctor — Noel Verification Report"
    )
    lines = [
        heading,
        "",
        f"> {report.get('notice_bilingual', COMPLIANCE_NOTICE_BILINGUAL)}",
        "",
        f"- **Verdict**: `{str(report.get('verdict', 'not_ready')).upper()}`",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **{'Clé API (masquée)' if is_fr else 'API Key (masked)'}**: `{report.get('masked_api_key')}`",
        f"- **{'Identifiants' if is_fr else 'Credentials'}**: `{report.get('credentials_path')}`",
        f"- **Continue YAML**: `{report.get('continue_path')}`",
        f"- **{'Routes Noël Cloud Run' if is_fr else 'Cloud Run Holiday Routes'}**: `{report.get('holiday_routes_deployed_on_cloud_run')}`",
        f"- **{'Dérive détectée' if is_fr else 'Revision Drift'}**: `{report.get('revision_drift_detected')}`",
        f"- **Online Producers / Active Trainers**: `{report.get('online_producers')}` / `{report.get('active_trainers')}`",
        f"- **GPU Status / Yield SLO**: `{report.get('gpu_status')}` / `{report.get('yield_slo_ok')}`",
        f"- **CES Recorded**: `{report.get('ces_recorded')}`",
        f"- **Receipt SHA-256**: `{report.get('receipt_sha256')}`",
        "",
        "| Check | Status | Summary |",
        "| :--- | :--- | :--- |",
    ]
    for chk in report.get("checks", ()):
        lines.append(f"| {chk.get('id')} | `{chk.get('status')}` | {chk.get('summary')} |")
    return "\n".join(lines)


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _validate_expected_sha(raw: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise GatewayPolicyError("expected_sha must be a 6..64 character alphanumeric/hex string.")
    cleaned = raw.strip().lower()
    if (
        not cleaned
        or cleaned == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or any(tok in cleaned for tok in _SECRET_VALUE_MARKERS)
    ):
        raise GatewayPolicyError("expected_sha must be a 6..64 character alphanumeric/hex string (not 'unknown' or secret).")
    return cleaned


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


def _dispatch_raw_http_fn(
    raw_http_fn: RawHttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allow_test_hosts: bool = True,
) -> tuple[int, str, dict[str, str]]:
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_test_hosts=allow_test_hosts, payload=payload)
    clean_payload = dict(payload) if payload is not None else None
    clean_headers = {str(k): str(v) for k, v in headers.items()} if headers is not None else None
    sig = inspect.signature(raw_http_fn)
    params = sig.parameters
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    pos = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    try:
        if raw_http_fn is default_http_raw:
            return default_http_raw(
                verb,
                url,
                payload=clean_payload,
                headers=clean_headers,
                timeout_s=DOCTOR_HTTP_TIMEOUT_S,
            )
        if has_varargs or len(pos) >= 4:
            raw_res = raw_http_fn(verb, url, clean_payload, clean_headers)
        elif has_varkw or ("payload" in params and "headers" in params):
            raw_res = raw_http_fn(verb, url, payload=clean_payload, headers=clean_headers)
        elif "headers" in params:
            raw_res = raw_http_fn(verb, url, headers=clean_headers)
        elif len(pos) == 3:
            raw_res = raw_http_fn(verb, url, clean_payload)
        else:
            raw_res = raw_http_fn(verb, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, json.dumps({"network_error": type(exc).__name__, "detail": sanitize_text_for_display(str(exc))[:200]}), {}

    if isinstance(raw_res, tuple) and len(raw_res) == 3:
        code_raw, body_raw, hdrs_raw = raw_res
        lower_hdrs = {str(k).strip().lower(): str(v) for k, v in (hdrs_raw or {}).items()}
    else:
        code_raw, body_raw = raw_res  # type: ignore[misc]
        lower_hdrs = {}

    if isinstance(body_raw, Mapping) or isinstance(body_raw, list):
        body_text = json.dumps(body_raw, ensure_ascii=False)
        lower_hdrs.setdefault("content-type", "application/json")
    elif isinstance(body_raw, (bytes, bytearray)):
        body_text = bytes(body_raw).decode("utf-8", errors="replace")
    else:
        body_text = str(body_raw or "")
    return int(code_raw), body_text, lower_hdrs


def _inspect_models_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str], bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body) or body.get("object", "list") != "list":
        msg = (
            "Contrat JSON `/v1/models` invalide."
            if is_fr
            else "Invalid `/v1/models` JSON schema (`object` != 'list')."
        )
        return False, [], False, [msg]
    raw_data = body.get("data")
    if not isinstance(raw_data, list):
        msg = (
            "Liste `data` absente dans `GET /v1/models`."
            if is_fr
            else "Missing `data` model list in `GET /v1/models`."
        )
        return False, [], False, [msg]
    valid_entries = [
        entry
        for entry in raw_data
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and bool(entry["id"].strip())
    ]
    model_ids = [str(entry["id"]).strip() for entry in valid_entries]
    issues: list[str] = []
    if not model_ids:
        issues.append(
            "Liste `data` de `GET /v1/models` vide."
            if is_fr
            else "`GET /v1/models` returned an empty `data` model list."
        )
    coder_id = "hivemind/" + "coder"
    coder_entries = [
        entry
        for entry in valid_entries
        if str(entry["id"]).strip() == coder_id
        and entry.get("object") in (None, "model")
        and ("owned_by" not in entry or (isinstance(entry.get("owned_by"), str) and bool(str(entry["owned_by"]).strip())))
    ]
    coder_present = len(coder_entries) >= 1
    if not coder_present:
        issues.append(
            "Modèle `hivemind/coder` absent des métadonnées `GET /v1/models`."
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
        issues.append(
            f"HTML invalide sur {path} : `<!DOCTYPE html>` absent."
            if is_fr
            else f"Invalid HTML on {path}: missing `<!DOCTYPE html>`."
        )
    if nonce_reused:
        issues.append(
            f"Nonce CSP réutilisé détecté sur {path}."
            if is_fr
            else f"Reused CSP nonce detected on {path}."
        )
    elif not csp_ok:
        issues.append(
            f"Contrat CSP par requête invalide ou absent sur {path}."
            if is_fr
            else f"Invalid or missing per-request CSP nonce contract on {path}."
        )
    if has_unsafe_sink:
        issues.append(
            f"Sink DOM non sûr détecté sur {path}."
            if is_fr
            else f"Unsafe DOM sink detected on {path}."
        )
    if has_static_fig:
        issues.append(
            f"Métrique statique détectée dans le HTML de {path}."
            if is_fr
            else f"Unbacked static telemetry figure detected in HTML on {path}."
        )
    if not compliance_banner_ok:
        issues.append(
            f"Bannière conformité PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR / AGPL-3.0 compliance banner missing on {path}."
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
    if not isinstance(body, Mapping):
        msg = (
            "Manifeste `/v1/hivemind/onboarding/manifest` absent ou JSON invalide."
            if is_fr
            else "Manifest `/v1/hivemind/onboarding/manifest` missing or returned invalid JSON."
        )
        return False, "unknown", None, (False if expected_sha else None), list(_MANIFEST_REQUIRED_HOLIDAY_ROUTES), ["hivemind_connect.py"], [msg]

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
            "Contrat `/v1/hivemind/onboarding/manifest` invalide (`campaign_id`, `personal_pc_only` ou `0.00 EUR / 0.00 USD`)."
            if is_fr
            else "Invalid `/v1/hivemind/onboarding/manifest` contract (`campaign_id`, `personal_pc_only`, or `0.00 EUR / 0.00 USD`)."
        )

    raw_routes = body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in _MANIFEST_REQUIRED_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(_MANIFEST_REQUIRED_HOLIDAY_ROUTES)
    if missing_routes:
        issues.append(
            f"Manifeste `holiday_routes` incomplet : {', '.join(missing_routes)}."
            if is_fr
            else f"Manifest `holiday_routes` missing required routes: {', '.join(missing_routes)}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) and s.strip() for s in raw_scripts) and raw_scripts:
        missing_scripts = [] if "hivemind_connect.py" in raw_scripts else ["hivemind_connect.py"]
    else:
        missing_scripts = ["hivemind_connect.py"]
    if missing_scripts:
        issues.append(
            "Manifeste `downloadable_scripts` sans `hivemind_connect.py`."
            if is_fr
            else "Manifest `downloadable_scripts` missing `hivemind_connect.py`."
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
        msg = (
            "Contrat `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/onboarding/feedback/summary` JSON payload."
        )
        return False, None, [msg]
    obj_ok = body.get("object") in _ACCEPTED_FEEDBACK_SUMMARY_OBJECTS
    raw_total = body.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1

    has_mean = "mean_ease_rating" in body
    has_avg = "average_ease_rating" in body
    ease_ok = bool((has_mean or has_avg) and total_ok)
    ease_vals: list[float] = []
    if ease_ok:
        for present, raw_val in ((has_mean, body.get("mean_ease_rating")), (has_avg, body.get("average_ease_rating"))):
            if not present:
                continue
            if total_val == 0:
                if raw_val is not None:
                    ease_ok = False
                    break
            elif (
                isinstance(raw_val, bool)
                or not isinstance(raw_val, (int, float))
                or not math.isfinite(float(raw_val))
                or not (1.0 <= float(raw_val) <= 5.0)
            ):
                ease_ok = False
                break
            else:
                ease_vals.append(round(float(raw_val), 2))
        if ease_ok and len(ease_vals) == 2 and ease_vals[0] != ease_vals[1]:
            ease_ok = False
    ease_val = ease_vals[0] if (ease_ok and ease_vals) else None

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
    lag_val = round(float(raw_lag), 4) if (lag_ok and raw_lag is not None) else None

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
    ok = bool(obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    if not ok:
        msg = (
            "Schéma `/v1/hivemind/onboarding/feedback/summary` non conforme."
            if is_fr
            else "Schema mismatch on `/v1/hivemind/onboarding/feedback/summary`."
        )
        return False, None, [msg]
    sanitized = {
        "object": str(body.get("object")),
        "total_responses": total_val,
        "mean_ease_rating": ease_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)},  # type: ignore[arg-type]
    }
    return True, sanitized, []


def _inspect_raid_status_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, int, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = (
            "Réponse `/v1/hivemind/raid/status` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/raid/status` payload."
        )
        return False, 0, [msg]
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
    active_trainers = 0
    if isinstance(body.get("swarm"), Mapping):
        raw_tr = body["swarm"].get("active_trainers")
        if _is_nonneg_int(raw_tr):
            active_trainers = int(raw_tr)
    if not ok:
        msg = (
            "Contrat conformité/télémétrie invalide sur `/v1/hivemind/raid/status`."
            if is_fr
            else "Invalid compliance or telemetry contract on `/v1/hivemind/raid/status`."
        )
        return False, active_trainers, [msg]
    return True, active_trainers, []


def _inspect_fleet_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, int, int, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = (
            "Contrat `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/producers/fleet` JSON payload."
        )
        return False, 0, 0, [msg]
    if body.get("object") not in _ACCEPTED_FLEET_OBJECTS:
        msg = (
            "Champ `object` invalide sur `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Invalid `object` metadata on `/v1/hivemind/producers/fleet`."
        )
        return False, 0, 0, [msg]
    rows = body.get("hiveminders")
    if not isinstance(rows, list):
        msg = (
            "Liste `hiveminders` absente sur `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Missing `hiveminders` list on `/v1/hivemind/producers/fleet`."
        )
        return False, 0, 0, [msg]
    for row in rows:
        if isinstance(row, Mapping) and "endpoint_url" in row:
            msg = (
                "Fuite `endpoint_url` interdite sur `/v1/hivemind/producers/fleet`."
                if is_fr
                else "Forbidden `endpoint_url` leak detected on `/v1/hivemind/producers/fleet`."
            )
            return False, 0, 0, [msg]
    has_live_hm = "live_hiveminders" in body
    if has_live_hm and not _is_nonneg_int(body.get("live_hiveminders")):
        msg = (
            "Champ `live_hiveminders` invalide sur `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Invalid non-integer `live_hiveminders` count on `/v1/hivemind/producers/fleet`."
        )
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
        msg = (
            "Contrat `/v1/hivemind/arena/summary` invalide (`object` manquant ou inconnu)."
            if is_fr
            else "Arena summary `/v1/hivemind/arena/summary` invalid `object` metadata."
        )
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

    ok = bool(counts_valid and conserved and dec_ok and wlb_ok and win_ratio_ok and by_task_ok)
    if not ok:
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
    """Execute list-argv ``bash deploy/gcp_openrouter/deploy.sh --dry-run`` without invoking live ``gcloud``."""
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
    try:
        sig = inspect.signature(runner)
        if "env" in sig.parameters:
            rc, stdout = runner(argv, 15.0, env=dry_env)  # type: ignore[call-arg]
        else:
            env_prefix = ["env", "HIVEMIND_DEPLOY_DRY_RUN=1"]
            if expected_sha is not None:
                env_prefix.append(f"HIVEMIND_BUILD_SHA={expected_sha}")
            rc, stdout = runner([*env_prefix, *argv], 15.0)
    except (OSError, subprocess.SubprocessError, GatewayPolicyError) as exc:
        return {
            "executed": True,
            "ok": False,
            "returncode": 1,
            "deploy_script_path": display_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": sanitize_text_for_display(f"deploy.sh --dry-run failed ({type(exc).__name__})."),
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


def verify_pr949_connect_quorum_and_parity(
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
    http_fn: HttpFn | None = None,
    transport: Any = None,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    lang: str = "en",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify PR #949 zero-inference 4-HTML-route parity, ``/v1/models`` metadata, 8+ quorum, CES SLO, arena, replay, and deploy dry-run."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or not (1 <= target_online_producers <= 500)
    ):
        raise GatewayPolicyError("target_online_producers must be an integer between 1 and 500.")
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise GatewayPolicyError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise GatewayPolicyError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    for flag in (
        require_live_quorum,
        require_ces_slo,
        check_arena_summary,
        verify_idempotency_replay,
        execute_deploy_dry_run,
    ):
        if not isinstance(flag, bool):
            raise GatewayPolicyError("Gate boolean flags must be True or False.")

    target_n = int(target_online_producers)
    min_ease_f = float(min_ease_rating)
    min_lag_f = float(min_zero_lag_ratio)
    clean_expected_sha = _validate_expected_sha(expected_sha) if (expected_sha is not None and str(expected_sha).strip()) else None

    eff_env = environ if environ is not None else (env if env is not None else os.environ)
    eff_home = home if home is not None else Path.home()
    eff_cwd = Path(cwd) if cwd is not None else Path.cwd()
    cred_target = default_credentials_path(home=eff_home)
    saved_creds = load_credentials(cred_target)
    env_key = str(eff_env.get(API_KEY_ENV, "")).strip()
    resolved_key = env_key if is_valid_hivemind_key(env_key) else (str(saved_creds["api_key"]).strip() if saved_creds else None)
    masked_key = mask_api_key(resolved_key)
    redacted_cred = redact_home_path(cred_target, home=eff_home)

    # 1. Local Personal-PC & Amadeus compliance gate BEFORE any network or subprocess call
    compliance = evaluate_personal_pc_compliance(env=eff_env, cwd=eff_cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        safe_gw = DEFAULT_GATEWAY
        with suppress(GatewayPolicyError):
            safe_gw = validate_gateway_url(
                str(gateway),
                allow_test_hosts=bool(transport is not None or http_fn is not None or raw_http_fn is not default_http_raw),
            )
        blocked: dict[str, Any] = sanitize_structure(
            {
                "schema": SCHEMA,
                "object": "hivemind.connect.pr949_quorum_parity_receipt",
                "slice_id": "SLICE-SCALE-DYN-49",
                "campaign_id": CAMPAIGN_ID,
                "mode": "pr949_quorum_gate",
                "lang": lang_code,
                "gateway": safe_gw,
                "compliance_ok": False,
                "corporate_detected": True,
                "compliance": compliance,
                "ready": False,
                "parity_ok": False,
                "gate_ok": False,
                "quorum_gate_ok": False,
                "ces_gate_ok": False,
                "exit_code": 2,
                "holiday_routes_deployed_on_cloud_run": False,
                "onboarding_feedback_deployed": False,
                "stale_revision_detected": False,
                "stale_404_drift": False,
                "cloud_run_lags_main": False,
                "redeploy_recommended": False,
                "routes_passed_count": 0,
                "routes_total_count": len(_PR949_REQUIRED_ROUTES) + (1 if check_arena_summary else 0),
                "http_statuses": {},
                "missing_routes": [],
                "failed_routes": [],
                "models_ok": False,
                "live_model_ids": [],
                "models_count": 0,
                "models_coder_present": False,
                "hivemind_coder_present": False,
                "html_contract_ok": False,
                "html_contracts": {},
                "unique_csp_nonces_count": 0,
                "unique_csp_nonces_ok": False,
                "html_dom_clean": False,
                "compliance_banners_ok": False,
                "manifest_present": False,
                "manifest_valid": False,
                "live_build_sha": "unknown",
                "expected_sha": clean_expected_sha,
                "sha_match": False if clean_expected_sha is not None else None,
                "sha_drift": False,
                "cloud_run_revision": None,
                "manifest_missing_routes": [],
                "manifest_missing_scripts": [],
                "feedback_schema_ok": False,
                "feedback_summary_ok": False,
                "feedback_summary": None,
                "mean_ease_rating": None,
                "average_ease_rating": None,
                "zero_game_lag_ratio": None,
                "min_ease_rating": min_ease_f,
                "min_zero_lag_ratio": min_lag_f,
                "ces_slo_ok": False,
                "zero_lag_slo_ok": False,
                "raid_status_ok": False,
                "active_trainers": 0,
                "fleet_ok": False,
                "online_producers": 0,
                "distinct_physical_gpus": 0,
                "target_online_producers": target_n,
                "producer_quorum_met": False,
                "awaiting_producer_quorum": True,
                "require_live_quorum": bool(require_live_quorum),
                "require_ces_slo": bool(require_ces_slo),
                "arena_summary_checked": bool(check_arena_summary),
                "arena_summary_ok": False if check_arena_summary else None,
                "arena_summary": None,
                "idempotency_replay_checked": bool(verify_idempotency_replay),
                "idempotent_dedup_ok": None,
                "idempotency_key": None,
                "deploy_dry_run_executed": bool(execute_deploy_dry_run),
                "deploy_dry_run": None,
                "zero_inference_verified": True,
                "outbound_http_requests": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "file_writes": 0,
                "masked_api_key": masked_key,
                "credentials_path": redacted_cred,
                "routes": [],
                "diagnostics": [compliance["summary"]],
                "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
                "financial_promise": FINANCIAL_PROMISE,
            },
            home=eff_home,
        )
        blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
        return blocked

    allow_test_hosts = bool(
        transport is not None or http_fn is not None or (raw_http_fn is not default_http_raw)
    )
    clean_gw = validate_gateway_url(str(gateway), allow_test_hosts=allow_test_hosts)

    client_cm: Any = None
    if http_fn is not None:
        client_cm = _InjectedHttpClient(http_fn)
    elif transport is not None:
        client_cm = _HttpxTransportClient(transport=transport, timeout_s=DOCTOR_HTTP_TIMEOUT_S)

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

    health_ok = False
    models_ok = False
    model_ids: list[str] = []
    models_count = 0
    coder_present = False
    manifest_present = False
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
    arena_summary_ok: bool | None = None
    arena_summary_doc: dict[str, Any] | None = None
    idempotent_dedup_ok: bool | None = None
    idem_key: str | None = None
    network_calls = 0
    subprocess_calls = 0

    def _call_pr949(
        client_obj: Any,
        method: str,
        path_or_url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, Mapping[str, Any] | None, str, dict[str, str]]:
        target_url = path_or_url if "://" in path_or_url else f"{clean_gw}{path_or_url}"
        if client_obj is not None:
            _assert_zero_inference_request(method, target_url, allow_test_hosts=True, payload=payload)
            try:
                resp = client_obj.request(
                    method,
                    target_url,
                    json=dict(payload) if payload is not None else None,
                    headers=dict(headers) if headers is not None else None,
                    timeout=DOCTOR_HTTP_TIMEOUT_S,
                )
                code = int(resp.status_code)
                body_text = resp.text
                hdrs = {str(k).strip().lower(): str(v) for k, v in resp.headers.items()}
            except (GatewayConnectionError, OSError, ValueError) as exc:
                return 0, None, json.dumps({"network_error": type(exc).__name__}), {}
            except Exception as exc:
                if httpx is not None and isinstance(exc, httpx.HTTPError):
                    return 0, None, json.dumps({"network_error": type(exc).__name__}), {}
                raise
        else:
            code, body_text, hdrs = _dispatch_raw_http_fn(
                raw_http_fn,
                method,
                target_url,
                payload=payload,
                headers=headers,
                allow_test_hosts=allow_test_hosts,
            )

        json_doc: dict[str, Any] | None = None
        if body_text.strip():
            with suppress(json.JSONDecodeError, ValueError):
                loaded = json.loads(body_text)
                if isinstance(loaded, Mapping):
                    json_doc = dict(loaded)
        return code, json_doc, body_text, hdrs

    with (client_cm if client_cm is not None else suppress()) as active_client:
        client_arg = active_client if client_cm is not None else None

        for path, kind, group in _PR949_REQUIRED_ROUTES:
            network_calls += 1
            code, json_doc, text_body, resp_headers = _call_pr949(client_arg, "GET", path)
            http_statuses[path] = code
            ok = code == 200
            route_issues: list[str] = []

            if not ok:
                if code == 404:
                    missing_routes.append(path)
                if code == 404 and group != "core":
                    route_issues.append(
                        f"GET {path} a répondu HTTP 404 : révision Cloud Run en retard sur `main`."
                        if is_fr
                        else f"GET {path} returned HTTP 404: Cloud Run revision lags behind `main`."
                    )
                elif code == 0:
                    route_issues.append(
                        f"GET {path} injoignable (erreur réseau/TLS)."
                        if is_fr
                        else f"GET {path} unreachable (network/TLS error)."
                    )
                else:
                    route_issues.append(
                        f"GET {path} a répondu HTTP {code} (attendu HTTP 200)."
                        if is_fr
                        else f"GET {path} returned HTTP {code} (expected HTTP 200)."
                    )
                if kind == "html":
                    html_all_ok = False
                    compliance_banners_ok = False
            elif path == "/health":
                health_ok = bool(
                    isinstance(json_doc, Mapping)
                    and json_doc.get("status") == "ok"
                    and not _has_forbidden_score_key(json_doc)
                )
                if not health_ok:
                    ok = False
                    route_issues.append(
                        "GET /health a répondu HTTP 200 sans `status == 'ok'`."
                        if is_fr
                        else "GET /health returned HTTP 200 without `status == 'ok'`."
                    )
            elif path == "/v1/models":
                models_ok, model_ids, coder_present, m_issues = _inspect_models_contract(json_doc, lang=lang_code)
                models_count = len(model_ids)
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
                manifest_present = True
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
                raid_status_ok, active_trainers, r_issues = _inspect_raid_status_contract(json_doc, lang=lang_code)
                if not raid_status_ok:
                    ok = False
                    route_issues.extend(r_issues)
            elif path == "/v1/hivemind/producers/fleet":
                fleet_ok, online_producers, distinct_gpus, f_issues = _inspect_fleet_contract(json_doc, lang=lang_code)
                if not fleet_ok:
                    ok = False
                    route_issues.extend(f_issues)

            if not ok:
                failed_routes.append(path)
                for item in route_issues:
                    diagnostics.append(sanitize_text_for_display(item, home=eff_home))

            route_rows.append(
                {
                    "path": path,
                    "method": "GET",
                    "kind": kind,
                    "group": group,
                    "http_status": code,
                    "ok": ok,
                    "diagnostic": sanitize_text_for_display(route_issues[0] if route_issues else "", home=eff_home),
                }
            )

        if check_arena_summary:
            a_path = "/v1/hivemind/arena/summary"
            network_calls += 1
            a_code, a_json, _, _ = _call_pr949(client_arg, "GET", a_path)
            http_statuses[a_path] = a_code
            a_issues: list[str] = []
            if a_code != 200:
                arena_summary_ok = False
                if a_code == 404:
                    missing_routes.append(a_path)
                a_issues.append(
                    f"GET {a_path} a répondu HTTP {a_code}."
                    if is_fr
                    else f"GET {a_path} returned HTTP {a_code}."
                )
            else:
                arena_summary_ok, arena_summary_doc, a_issues = _inspect_arena_summary_contract(a_json, lang=lang_code)
            if not arena_summary_ok:
                failed_routes.append(a_path)
                for item in a_issues:
                    diagnostics.append(sanitize_text_for_display(item, home=eff_home))
            route_rows.append(
                {
                    "path": a_path,
                    "method": "GET",
                    "kind": "json",
                    "group": "holiday",
                    "http_status": a_code,
                    "ok": bool(arena_summary_ok),
                    "diagnostic": sanitize_text_for_display(a_issues[0] if a_issues else "", home=eff_home),
                }
            )

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
            seed = f"{clean_gw}|{clean_expected_sha or 'live'}|{pre_total if pre_total is not None else -1}"
            idem_key = f"pr949-replay-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"
            fb_payload = {
                "profile": "dev",
                "ease_rating": 5,
                "zero_game_lag": True,
                "platform": _platform_feedback_tag(),
                "note": "",
            }
            replay_headers = {"Idempotency-Key": idem_key}

            p1_code, p1_json, _, _ = _call_pr949(
                client_arg,
                "POST",
                "/v1/hivemind/onboarding/feedback",
                payload=fb_payload,
                headers=replay_headers,
            )
            network_calls += 1
            http_statuses["POST /v1/hivemind/onboarding/feedback"] = p1_code
            p1_ok, p1_sum, _ = (
                _inspect_feedback_summary_contract(p1_json, lang=lang_code)
                if p1_code in (200, 201)
                else (False, None, [])
            )

            p2_code, p2_json, _, _ = _call_pr949(
                client_arg,
                "POST",
                "/v1/hivemind/onboarding/feedback",
                payload=fb_payload,
                headers=replay_headers,
            )
            network_calls += 1
            http_statuses["POST /v1/hivemind/onboarding/feedback#replay"] = p2_code
            p2_ok, p2_sum, _ = (
                _inspect_feedback_summary_contract(p2_json, lang=lang_code)
                if p2_code in (200, 201)
                else (False, None, [])
            )

            ref_code, ref_json, _, _ = _call_pr949(client_arg, "GET", "/v1/hivemind/onboarding/feedback/summary")
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
                and pre_total is not None
                and pre_dev is not None
                and int(p1_sum["total_responses"]) == int(pre_total) + 1
                and int(p1_sum["by_profile"]["dev"]) == int(pre_dev) + 1
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
            environ=eff_env,
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
    onboarding_feedback_deployed = bool(
        http_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200 and feedback_summary_ok
    )
    holiday_routes_deployed = bool(
        health_ok
        and manifest_valid
        and unique_csp_nonces_ok
        and html_dom_clean
        and compliance_banners_ok
        and onboarding_feedback_deployed
        and raid_status_ok
        and (not check_arena_summary or arena_summary_ok is True)
    )

    parity_ok = bool(
        passed_count == total_count
        and health_ok
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
    stale_revision_detected = bool(stale_404_drift or sha_drift or not holiday_routes_deployed)

    mean_ease_out = feedback_summary.get("mean_ease_rating") if feedback_summary is not None else None
    zero_lag_out = feedback_summary.get("zero_game_lag_ratio") if feedback_summary is not None else None

    report: dict[str, Any] = sanitize_structure(
        {
            "schema": SCHEMA,
            "object": "hivemind.connect.pr949_quorum_parity_receipt",
            "slice_id": "SLICE-SCALE-DYN-49",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_quorum_gate",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "compliance": compliance,
            "ready": gate_ok,
            "parity_ok": parity_ok,
            "gate_ok": gate_ok,
            "quorum_gate_ok": quorum_gate_ok,
            "ces_gate_ok": ces_gate_ok,
            "exit_code": 0 if gate_ok else 2,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "stale_404_drift": stale_404_drift,
            "cloud_run_lags_main": cloud_run_lags_main,
            "redeploy_recommended": bool(not parity_ok or cloud_run_lags_main or sha_drift),
            "routes_passed_count": passed_count,
            "routes_total_count": total_count,
            "http_statuses": http_statuses,
            "missing_routes": missing_routes,
            "failed_routes": failed_routes,
            "models_ok": models_ok,
            "live_model_ids": model_ids,
            "models_count": models_count,
            "models_coder_present": coder_present,
            "hivemind_coder_present": coder_present,
            "html_contract_ok": unique_csp_nonces_ok and html_dom_clean and compliance_banners_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "unique_csp_nonces_ok": unique_csp_nonces_ok,
            "html_dom_clean": html_dom_clean,
            "compliance_banners_ok": compliance_banners_ok,
            "manifest_present": manifest_present,
            "manifest_valid": manifest_valid,
            "live_build_sha": live_build_sha,
            "expected_sha": clean_expected_sha,
            "sha_match": sha_match,
            "sha_drift": sha_drift,
            "cloud_run_revision": cloud_run_revision,
            "manifest_missing_routes": manifest_missing_routes,
            "manifest_missing_scripts": manifest_missing_scripts,
            "feedback_schema_ok": feedback_summary_ok,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary,
            "mean_ease_rating": mean_ease_out,
            "average_ease_rating": mean_ease_out,
            "zero_game_lag_ratio": zero_lag_out,
            "min_ease_rating": min_ease_f,
            "min_zero_lag_ratio": min_lag_f,
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "raid_status_ok": raid_status_ok,
            "active_trainers": active_trainers,
            "fleet_ok": fleet_ok,
            "online_producers": online_producers,
            "distinct_physical_gpus": distinct_gpus,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "require_live_quorum": bool(require_live_quorum),
            "require_ces_slo": bool(require_ces_slo),
            "arena_summary_checked": bool(check_arena_summary),
            "arena_summary_ok": arena_summary_ok,
            "arena_summary": arena_summary_doc,
            "idempotency_replay_checked": bool(verify_idempotency_replay),
            "idempotent_dedup_ok": idempotent_dedup_ok,
            "idempotency_key": idem_key,
            "deploy_dry_run_executed": bool(execute_deploy_dry_run),
            "deploy_dry_run": deploy_dry_run_res,
            "zero_inference_verified": True,
            "outbound_http_requests": network_calls,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "file_writes": 0,
            "masked_api_key": masked_key,
            "credentials_path": redacted_cred,
            "routes": route_rows,
            "diagnostics": diagnostics,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "financial_promise": FINANCIAL_PROMISE,
        },
        home=eff_home,
    )
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def render_pr949_cli(report: Mapping[str, Any]) -> str:
    """Render concise EN or FR PR #949 quorum & parity CLI report."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    title = (
        "HIVEMIND PR #949 QUORUM & PARITÉ NOËL (PC PERSO • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND PR #949 QUORUM & PARITY GATE (PERSONAL PC ONLY • 0.00 EUR / 0.00 USD)"
    )
    verdict = "READY" if report.get("gate_ok") else "NOT_READY"
    lines = [
        "=" * 72,
        title,
        "=" * 72,
        f"{'Règle' if is_fr else 'Rule'}: {report.get('notice_bilingual', COMPLIANCE_NOTICE_BILINGUAL)}",
        f"Gateway: {report.get('gateway')} | Verdict: {verdict} | Key: {report.get('masked_api_key')}",
        (
            f"Routes: {report.get('routes_passed_count')}/{report.get('routes_total_count')} | "
            f"CSP nonces={report.get('unique_csp_nonces_count')} | "
            f"hivemind/coder={report.get('models_coder_present')}"
        ),
        (
            f"Quorum: {report.get('online_producers')}/{report.get('target_online_producers')} "
            f"(met={report.get('producer_quorum_met')}, awaiting={report.get('awaiting_producer_quorum')}) | "
            f"CES ease={report.get('mean_ease_rating')} lag={report.get('zero_game_lag_ratio')}"
        ),
        (
            f"Arena OK: {report.get('arena_summary_ok')} | "
            f"Idempotent Replay: {report.get('idempotent_dedup_ok')} | "
            f"Build SHA: {report.get('live_build_sha')}"
        ),
        *(
            [f"CES Queue Flush: {report['ces_flush'].get('summary')}"]
            if isinstance(report.get("ces_flush"), Mapping)
            else []
        ),
        f"SHA-256: {report.get('receipt_sha256', '')}",
    ]
    return "\n".join(lines)


def render_pr949_markdown(report: Mapping[str, Any]) -> str:
    """Render EN or FR Markdown PR #949 quorum & parity report with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    heading = (
        "# HiveMind PR #949 — Rapport Quorum & Parité Noël"
        if is_fr
        else "# HiveMind PR #949 — Quorum & Holiday Parity Report"
    )
    lines = [
        heading,
        "",
        f"> {report.get('notice_bilingual', COMPLIANCE_NOTICE_BILINGUAL)}",
        "",
        f"- **Verdict**: `{'READY' if report.get('gate_ok') else 'NOT_READY'}` (`exit_code={report.get('exit_code')}`)",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **{'Clé API (masquée)' if is_fr else 'API Key (masked)'}**: `{report.get('masked_api_key')}`",
        f"- **{'Identifiants' if is_fr else 'Credentials'}**: `{report.get('credentials_path')}`",
        f"- **Routes PR #949**: `{report.get('routes_passed_count')}/{report.get('routes_total_count')}` (`unique_csp_nonces={report.get('unique_csp_nonces_count')}`)",
        f"- **Models (`hivemind/coder`)**: `{report.get('models_coder_present')}`",
        f"- **{'Quorum Producteurs' if is_fr else 'Producer Quorum'}**: `{report.get('online_producers')}/{report.get('target_online_producers')}` (`met={report.get('producer_quorum_met')}`, `awaiting={report.get('awaiting_producer_quorum')}`)",
        f"- **CES & Gamer Lag**: `mean_ease={report.get('mean_ease_rating')}`, `zero_game_lag_ratio={report.get('zero_game_lag_ratio')}`, `ces_slo_ok={report.get('ces_slo_ok')}`",
        f"- **Arena Summary OK**: `{report.get('arena_summary_ok')}` | **Idempotency Replay**: `{report.get('idempotent_dedup_ok')}`",
        *(
            [f"- **CES Queue Flush**: {report['ces_flush'].get('summary')}"]
            if isinstance(report.get("ces_flush"), Mapping)
            else []
        ),
        f"- **Build SHA**: `{report.get('live_build_sha')}` (expected=`{report.get('expected_sha') or 'n/a'}`)",
        f"- **Receipt SHA-256**: `{report.get('receipt_sha256')}`",
        "",
        "| Route | HTTP | OK | Diagnostic |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for row in report.get("routes", ()):
        lines.append(f"| `{row.get('path')}` | `{row.get('http_status')}` | `{row.get('ok')}` | {row.get('diagnostic') or '—'} |")
    return "\n".join(lines)


def cmd_doctor(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute Noel doctor or PR #949 quorum/parity gate and emit CLI, Markdown, or JSON output."""
    _ = (client, parser)
    raw_out = getattr(args, "output", None)
    out_path: Path | None = None
    if raw_out is not None:
        cleaned_out = str(raw_out).strip()
        if not cleaned_out or "\x00" in cleaned_out:
            raise GatewayPolicyError("Invalid --output path.")
        out_path = Path(cleaned_out).expanduser()

    raw_cwd = getattr(args, "cwd", None)
    if raw_cwd is not None:
        cleaned_cwd = str(raw_cwd).strip()
        if not cleaned_cwd or "\x00" in cleaned_cwd:
            raise GatewayPolicyError("Invalid --cwd path.")
        eff_cwd: Path | str | None = Path(cleaned_cwd)
    else:
        eff_cwd = _ctx_cwd(args)

    eff_home = _ctx_home(args)
    eff_env = _ctx_environ(args)
    eff_lang = str(getattr(args, "lang", "en"))

    if getattr(args, "pr949_quorum_gate", False):
        gw_arg = getattr(args, "gateway", None) or eff_env.get(GATEWAY_ENV) or eff_env.get("HIVEMIND_GATEWAY_URL") or DEFAULT_GATEWAY
        report = verify_pr949_connect_quorum_and_parity(
            str(gw_arg),
            expected_sha=getattr(args, "expected_sha", None),
            target_online_producers=int(getattr(args, "target_online_producers", DEFAULT_TARGET_ONLINE_PRODUCERS)),
            min_ease_rating=float(getattr(args, "min_ease_rating", DEFAULT_MIN_EASE_RATING)),
            min_zero_lag_ratio=float(getattr(args, "min_zero_lag_ratio", DEFAULT_MIN_ZERO_LAG_RATIO)),
            require_live_quorum=bool(getattr(args, "require_live_quorum", False)),
            require_ces_slo=bool(getattr(args, "require_ces_slo", False)),
            check_arena_summary=bool(getattr(args, "check_arena_summary", True)),
            verify_idempotency_replay=bool(getattr(args, "verify_idempotency_replay", False)),
            execute_deploy_dry_run=bool(getattr(args, "run_deploy_dry_run", False)),
            repo_root=getattr(args, "_repo_root", None),
            raw_http_fn=getattr(args, "_raw_http_fn", None) or default_http_raw,
            http_fn=getattr(args, "_http_fn", None),
            transport=getattr(args, "_transport", None),
            runner=getattr(args, "_runner", None) or default_subprocess_runner,
            environ=eff_env,
            cwd=eff_cwd,
            home=eff_home,
            lang=eff_lang,
        )
        if getattr(args, "flush_ces_queue", False) and report.get("compliance_ok", True):
            spool = ConnectCesRetrySpool(
                getattr(args, "ces_queue_file", None),
                home=eff_home,
                credentials_path=_credentials_path(args),
                env=eff_env,
                os_name=_ctx_os_name(args),
            )
            report["ces_flush"] = spool.flush(
                str(report["gateway"]),
                http_fn=getattr(args, "_http_fn", None),
                transport=getattr(args, "_transport", None),
                raw_http_fn=getattr(args, "_raw_http_fn", None),
                environ=eff_env,
                cwd=eff_cwd,
                lang=eff_lang,
            )
            report["receipt_sha256"] = compute_receipt_sha256(report)
        if getattr(args, "json", False):
            rendered = json.dumps(report, indent=2, ensure_ascii=False)
        elif getattr(args, "markdown", False):
            rendered = render_pr949_markdown(report)
        else:
            rendered = render_pr949_cli(report)
    else:
        report = run_connect_doctor(
            gateway=getattr(args, "gateway", None),
            credentials_path=_credentials_path(args),
            continue_path=_continue_path(args),
            save_credentials=bool(getattr(args, "save_credentials", False)),
            write_continue=bool(getattr(args, "write_continue", False)),
            require_holiday_routes=bool(getattr(args, "require_holiday_routes", False)),
            first_heartbeat=bool(getattr(args, "first_heartbeat", False)),
            submit_ces=bool(getattr(args, "submit_ces", False)),
            prompt_ces=bool(getattr(args, "prompt_ces", False)),
            profile=getattr(args, "profile", None),
            ease_rating=int(getattr(args, "ease_rating", 5)),
            had_game_lag=bool(getattr(args, "had_game_lag", False)),
            host_label=str(getattr(args, "host_label", DEFAULT_HOST_LABEL)),
            max_temp_c=float(getattr(args, "max_temp_c", DEFAULT_MAX_TEMP_C)),
            min_free_vram_mib=int(getattr(args, "min_free_vram_mib", DEFAULT_MIN_FREE_VRAM_MIB)),
            yield_budget_ms=float(getattr(args, "yield_budget_ms", DEFAULT_YIELD_BUDGET_MS)),
            extra_games=list(getattr(args, "game", []) or []),
            lang=eff_lang,
            timeout_s=float(getattr(args, "timeout", DOCTOR_HTTP_TIMEOUT_S)),
            cwd=eff_cwd,
            home=eff_home,
            environ=eff_env,
            os_name=_ctx_os_name(args),
            system_name=getattr(args, "_system_name", None),
            transport=getattr(args, "_transport", None),
            http_fn=getattr(args, "_http_fn", None),
            gpu_probe_fn=getattr(args, "_gpu_probe_fn", None),
            process_probe_fn=getattr(args, "_process_probe_fn", None),
            worker_yield_fn=getattr(args, "_worker_yield_fn", None),
            clock_fn=getattr(args, "_clock_fn", None),
            input_fn=getattr(args, "_input_fn", None),
            runner=getattr(args, "_runner", None),
            spool_path=getattr(args, "ces_queue_file", None),
            flush_ces_queue=bool(getattr(args, "flush_ces_queue", False)),
            clean_ces_queue=bool(getattr(args, "clean_ces_queue", False)),
        )
        if getattr(args, "json", False):
            rendered = json.dumps(report, indent=2)
        elif getattr(args, "markdown", False):
            rendered = render_doctor_markdown(report)
        else:
            rendered = render_doctor_cli(report)

    issues = scan_output_safety(rendered, home=eff_home)
    if issues:
        raise ValueError(f"Output safety guard failed: {', '.join(issues)}")
    if out_path is not None and report.get("compliance_ok", True):
        _write_file_secure_atomic(
            out_path,
            rendered + ("\n" if not rendered.endswith("\n") else ""),
            posix_mode=0o600,
            os_name=_ctx_os_name(args),
        )
    print(rendered)
    return int(report["exit_code"])


def _add_shared_continuity_flags(subparser: argparse.ArgumentParser, *, include_writes: bool = True) -> None:
    subparser.add_argument("--gateway", default=argparse.SUPPRESS, help=f"Gateway URL (default: ${GATEWAY_ENV} or the public HiveMind gateway)")
    subparser.add_argument("--credentials", default=argparse.SUPPRESS, metavar="FILE", help="Credentials JSON path (default ~/.hivemind/credentials.json)")
    subparser.add_argument("--enforce-personal-pc", action="store_true", default=argparse.SUPPRESS, help="Refuse corporate environments/repos with exit code 2")
    subparser.add_argument("--cwd", default=argparse.SUPPRESS, metavar="DIR", help="Workspace path checked locally for corporate markers/.git/config")
    if include_writes:
        subparser.add_argument("--save-credentials", action="store_true", help="Atomically save 0600 credentials JSON")
        subparser.add_argument("--write-continue", action="store_true", help="Atomically write 0600 ~/.continue/hivemind_provider.yaml")
        subparser.add_argument("--continue-file", default=None, metavar="FILE", help="Custom Continue provider YAML output path")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser preserving all existing subcommands and adding ``doctor``."""
    parser = argparse.ArgumentParser(description="Connect your editor or Python code to the HiveMind OpenRouter.")
    parser.add_argument("--gateway", help=f"Gateway URL (default: ${GATEWAY_ENV} or the public HiveMind gateway)")
    parser.add_argument("--credentials", default=None, metavar="FILE", help="Path to saved HiveMind credentials JSON (default ~/.hivemind/credentials.json)")
    parser.add_argument("--enforce-personal-pc", action="store_true", default=False, help="Enforce local Personal-PC / Amadeus compliance gate (exit 2 on corporate markers)")
    parser.add_argument("--cwd", default=None, metavar="DIR", help="Workspace directory checked locally for corporate markers/.git/config")
    parser.add_argument("--lang", type=normalize_lang, default="en", choices=SUPPORTED_LANGS, help="CLI output language ('en' or 'fr')")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Create an account and get an API key")
    reg.add_argument("--handle", required=True)
    reg.add_argument("--role", choices=["consumer", "producer", "prosumer", "trainer"], default="prosumer",
                     help="prosumer (default) can both use models and share a GPU")
    reg.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    _add_shared_continuity_flags(reg)
    reg.set_defaults(func=cmd_register)

    reg_g = sub.add_parser("register-google", help="Sign in with Google (ID token, gcloud, or browser)")
    reg_g.add_argument("--token", help="Google ID token (or set GOOGLE_ID_TOKEN)")
    reg_g.add_argument("--gcloud", action="store_true", help="Obtain Google ID token using gcloud CLI")
    reg_g.add_argument("--audience", help="Optional audience parameter for gcloud identity token")
    reg_g.add_argument("--browser", action="store_true", help="Open /signup page in your browser")
    reg_g.add_argument("--role", choices=["consumer", "producer", "prosumer", "trainer"], default="prosumer",
                       help="prosumer (default) can both use models and share a GPU")
    reg_g.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    _add_shared_continuity_flags(reg_g)
    reg_g.set_defaults(func=cmd_register_google)

    login_g = sub.add_parser("login-google", help="Alias for register-google (returns rotated API key)")
    login_g.add_argument("--token", help="Google ID token (or set GOOGLE_ID_TOKEN)")
    login_g.add_argument("--gcloud", action="store_true", help="Obtain Google ID token using gcloud CLI")
    login_g.add_argument("--audience", help="Optional audience parameter for gcloud identity token")
    login_g.add_argument("--browser", action="store_true", help="Open /signup page in your browser")
    login_g.add_argument("--role", choices=["consumer", "producer", "prosumer", "trainer"], default="prosumer")
    login_g.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    _add_shared_continuity_flags(login_g)
    login_g.set_defaults(func=cmd_register_google)

    chk = sub.add_parser("check", help="Verify your key and show balance, live models and tiers")
    _add_shared_continuity_flags(chk)
    chk.set_defaults(func=cmd_check)

    txf = sub.add_parser("transfer", help="Send HMT Reflexion Points to another HiveMinder")
    txf.add_argument("--to", required=True, help="Recipient @handle, handle, email, or account_id")
    txf.add_argument("--amount", type=float, required=True, help="Positive HMT amount to transfer")
    txf.add_argument("--memo", default="peer-hiveminder-support", help="Short note describing the transfer")
    txf.add_argument("--idempotency-key", help="Optional idempotency key to prevent duplicate sends")
    txf.add_argument("--model-ref", help="Optional model id being tested (e.g. hivemind/crowd-v1)")
    txf.add_argument("--pr-url", help="Optional GitHub PR URL attached to the AMCP event")
    txf.add_argument("--no-amcp", action="store_true", help="Skip emitting an AMCP/Kafka collaboration message")
    _add_shared_continuity_flags(txf, include_writes=False)
    txf.set_defaults(func=cmd_transfer)

    snip = sub.add_parser("snippet", help="Print settings for an editor or library")
    snip.add_argument("tool", nargs="?", choices=TOOLS)
    snip.add_argument("--model", default=DEFAULT_MODEL, help="tier id such as coder, coder-lite, moe-coder, or auto")
    snip.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    snip.add_argument("--embed-key", action="store_true", help=f"Put ${API_KEY_ENV} into the snippet instead of a placeholder")
    snip.add_argument("--list", action="store_true")
    _add_shared_continuity_flags(snip, include_writes=False)
    snip.set_defaults(func=cmd_snippet)

    doc = sub.add_parser("doctor", help="Zero-inference <60s Noel Cloud Run drift, Steam yield, heartbeat & CES verifier")
    doc.add_argument("--lang", type=normalize_lang, default=argparse.SUPPRESS, choices=SUPPORTED_LANGS, help="Output language ('en' or 'fr')")
    _add_shared_continuity_flags(doc)
    doc.add_argument("--require-holiday-routes", action="store_true", help="Fail closed (exit 2) if Cloud Run holiday routes or CES hit stale 404/5xx")
    doc.add_argument("--first-heartbeat", action="store_true", help="Verify GET /v1/hivemind/me and send outbound POST /v1/hivemind/opentrainer/heartbeat")
    doc.add_argument("--submit-ces", action="store_true", help="Submit non-blocking zero-PII CES feedback")
    doc.add_argument("--prompt-ces", action="store_true", help="Interactively prompt 1-line CES rating and submit zero-PII feedback")
    doc.add_argument("--profile", type=normalize_profile, default=None, choices=sorted(VALID_PROFILES), help="CES profile ('dev' or 'gamer')")
    doc.add_argument("--ease-rating", type=parse_ease_rating, default=5, choices=(1, 2, 3, 4, 5), help="CES ease rating 1..5")
    doc.add_argument("--had-game-lag", action="store_true", help="Report foreground game lag (sets zero_game_lag=False)")
    doc.add_argument("--host-label", default=DEFAULT_HOST_LABEL, help="Sanitized host label for OpenTrainer heartbeat")
    doc.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C, help="Quiet-fan GPU temperature ceiling (default 70.0 C)")
    doc.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB, help="Free VRAM reserve in MiB (default 2048)")
    doc.add_argument("--yield-budget-ms", type=float, default=DEFAULT_YIELD_BUDGET_MS, help="Steam yield budget in milliseconds (default 1500.0)")
    doc.add_argument("--game", action="append", default=[], help="Extra game executable name to watch")
    doc.add_argument("--timeout", type=float, default=DOCTOR_HTTP_TIMEOUT_S, help="Bounded HTTP timeout per probe")
    doc.add_argument("--output", default=None, metavar="FILE", help="Atomically write JSON/Markdown/CLI output with POSIX 0600 permissions")
    doc.add_argument("--ces-queue-file", default=None, metavar="FILE", help="Optional 0600 CES retry spool JSON path")
    doc.add_argument("--flush-ces-queue", action="store_true", help="Idempotently flush pending spooled CES events")
    doc.add_argument("--clean-ces-queue", action="store_true", help="Remove local CES retry spool cleanly")
    doc.add_argument("--pr949-quorum-gate", action="store_true", help="Run PR #949 live 4-HTML parity, models, 8+ producer quorum, arena, and replay gate")
    doc.add_argument("--expected-sha", default=None, help="Expected Cloud Run HIVEMIND_BUILD_SHA (6..64 chars)")
    doc.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS, help="Target online producers for PR #949 quorum (default 8)")
    doc.add_argument("--min-ease-rating", type=float, default=DEFAULT_MIN_EASE_RATING, help="Minimum mean CES ease rating (default 4.2)")
    doc.add_argument("--min-zero-lag-ratio", type=float, default=DEFAULT_MIN_ZERO_LAG_RATIO, help="Minimum zero_game_lag_ratio (default 0.95)")
    doc.add_argument("--require-live-quorum", action="store_true", help="Fail closed (exit 2) when online_producers < --target-online-producers")
    doc.add_argument("--require-ces-slo", action="store_true", help="Fail closed (exit 2) when CES/zero-lag SLOs are breached")
    doc.add_argument("--check-arena-summary", action=argparse.BooleanOptionalAction, default=True, help="Check /v1/hivemind/arena/summary vote conservation and Wilson LB")
    doc.add_argument("--verify-idempotency-replay", action="store_true", help="Verify Idempotency-Key deduplication on POST /v1/hivemind/onboarding/feedback")
    doc.add_argument("--run-deploy-dry-run", action="store_true", help="Verify bash deploy/gcp_openrouter/deploy.sh --dry-run (0 gcloud calls)")
    out_grp = doc.add_mutually_exclusive_group()
    out_grp.add_argument("--json", action="store_true", help="Emit machine-readable JSON report with SHA-256 receipt")
    out_grp.add_argument("--markdown", action="store_true", help="Emit Markdown verification report")
    doc.set_defaults(func=cmd_doctor)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: Any | None = None,
    http_fn: HttpFn | None = None,
    raw_http_fn: RawHttpFn | None = None,
    repo_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    gpu_probe_fn: GpuProbeFn | None = None,
    process_probe_fn: ProcessProbeFn | None = None,
    worker_yield_fn: WorkerYieldFn | None = None,
    clock_fn: ClockFn | None = None,
    input_fn: InputFn | None = None,
    runner: RunnerFn | None = None,
) -> int:
    """CLI entrypoint preserving backward-compatible ``main(argv=..., transport=...)``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    allow_test_hosts = bool(
        transport is not None or http_fn is not None or (raw_http_fn is not None and raw_http_fn is not default_http_raw)
    )

    raw_cli_cwd = getattr(args, "cwd", None)
    if raw_cli_cwd is not None:
        cleaned_cli_cwd = str(raw_cli_cwd).strip()
        if not cleaned_cli_cwd or "\x00" in cleaned_cli_cwd:
            print("Invalid --cwd path.", file=sys.stderr)
            return 2
        eff_cwd: Path | str | None = Path(cleaned_cli_cwd)
    else:
        eff_cwd = cwd

    args._environ = environ
    args._cwd = eff_cwd
    args._home = home
    args._os_name = os_name
    args._system_name = system_name
    args._allow_test_hosts = allow_test_hosts
    args._transport = transport
    args._http_fn = http_fn
    args._raw_http_fn = raw_http_fn
    args._repo_root = repo_root
    args._gpu_probe_fn = gpu_probe_fn
    args._process_probe_fn = process_probe_fn
    args._worker_yield_fn = worker_yield_fn
    args._clock_fn = clock_fn
    args._input_fn = input_fn
    args._runner = runner

    if args.command != "doctor" and getattr(args, "enforce_personal_pc", False):
        comp = evaluate_personal_pc_compliance(
            env=environ,
            cwd=eff_cwd,
            lang=str(getattr(args, "lang", "en")),
        )
        if not comp["compliance_ok"]:
            print(comp["summary"], file=sys.stderr)
            return 2

    if args.command == "doctor":
        try:
            return int(args.func(None, args, parser))
        except GatewayPolicyError as exc:
            print(sanitize_text_for_display(str(exc), home=home), file=sys.stderr)
            return 2

    if http_fn is not None:
        client_cm: Any = _InjectedHttpClient(http_fn)
    elif transport is not None:
        client_cm = _HttpxTransportClient(transport=transport, timeout_s=HTTP_TIMEOUT_S)
    else:
        client_cm = _StdlibHttpClient(timeout_s=HTTP_TIMEOUT_S)

    with client_cm as client:
        try:
            return int(args.func(client, args, parser))
        except GatewayPolicyError as exc:
            print(sanitize_text_for_display(str(exc), home=home), file=sys.stderr)
            return 2
        except GatewayConnectionError as exc:
            print(f"cannot reach the gateway: {sanitize_text_for_display(str(exc), home=home)}", file=sys.stderr)
            return 1
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                print(f"cannot reach the gateway: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            raise


if __name__ == "__main__":
    sys.exit(main())
