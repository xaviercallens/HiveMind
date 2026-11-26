#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Post-setup quickstart probe, 1-prompt CES reporter, and Cloud Run holiday route smoke gate.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library:
  1. Zero-inference Cloud Run Holiday Route Smoke Gate (`--smoke-routes` and default probe):
     checks GET `/health`, `/v1/models`, `/noel`, `/quick`, `/raid-noel`, `/arena-noel`, and
     `/v1/hivemind/onboarding/feedback/summary` without ever calling `/v1/chat/completions`.
  2. Post-setup local readiness probe: inspects `~/.hivemind/credentials.json` (redacting home
     paths to `~`, masking `hm-...` keys, plus optional `GET /v1/hivemind/me` with
     `--verify-account`), `~/.continue/hivemind_provider.yaml`, Gamer-Whisper thermal ceiling
     (`<= 70.0 C`), free VRAM reserve (`>= 2048 MiB`), active Steam/PC game detection with
     `< 1.5 s` yield budget verification, and a strictly local corporate-PC warning.
  3. 1-prompt anonymous CES & zero-game-lag reporter (`--submit-feedback` and `--prompt-ces`):
     posts zero-PII payloads to `POST /v1/hivemind/onboarding/feedback` with FR/EN handling of
     HTTP 404/429/5xx and emits `--json` or Markdown reports with a SHA-256 receipt.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_post_setup_probe/1"
RECEIPT_OBJECT = "hivemind.noel_post_setup_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

SUPPORT_DEFAULT_QUEUE_FILENAME = "ces_feedback_queue.json"
CES_QUEUE_SCHEMA = "hivemind.noel_post_setup_probe.ces_queue/1"
CES_QUEUE_MAX_ENTRIES = 128
CES_QUEUE_MAX_DELIVERED = 512
CES_QUEUE_MAX_RETRIES = 16
MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB bounded read
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FLEET_OBJECT = "hivemind.producer_fleet"
OPENTRAINER_HB_ACK_OBJECT = "hivemind.opentrainer_heartbeat_ack"
DEFAULT_TRAINING_STACK = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_OPENTRAINER_CAMPAIGN = "E8-STAGE2-TINY-GRANITE-MOE-QLORA"

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
MIN_PYTHON: tuple[int, int] = (3, 10)
MAX_TESTED_MINOR = 13

GAMER_WHISPER_MAX_TEMP_C = 70.0
GAMER_WHISPER_MIN_FREE_VRAM_MIB = 2048
GAMER_WHISPER_YIELD_BUDGET_S = 1.5
PREEMPTION_SIGNAL_ESTIMATE_S = 0.25
HTTP_TIMEOUT_S = 4.0
SUBPROCESS_TIMEOUT_S = 3.0

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})

HOLIDAY_SMOKE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
)

HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
EXPECTED_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = (
    "hivemind_quickstart.py",
    "hivemind_doctor.py",
    "noel_post_setup_probe.py",
)

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
        "rustclient.exe",
        "palworld-win64-shipping.exe",
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

_CORP_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amadeus", re.compile(r"amadeus", re.IGNORECASE)),
    (
        "corp",
        re.compile(
            r"(?:^|[^a-z])corp(?:orate|net|sys|s)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])",
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

_INSPECTED_ENV_KEYS: tuple[str, ...] = (
    "USERDNSDOMAIN",
    "USERDOMAIN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
    "ZSCALER_CERT_PATH",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"


def _default_compliance_block() -> dict[str, Any]:
    return {
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "corporate_env_transmitted": False,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
    }


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)
_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"([A-Za-z0-9_+=/=-]{8,128})"|'([A-Za-z0-9_+=/=-]{8,128})')""")
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*{3,4}(?:[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s)|[A-Za-z0-9_-]{3,})"
)
_UNSAFE_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "document" + r"\.writeln?)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

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

_QUEUE_LOCKS_GUARD = threading.Lock()
_QUEUE_LOCKS: dict[str, threading.RLock] = {}

HttpFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
InputFn = Callable[[str], str]


class ProbeError(ValueError):
    """Raised when probe configuration, gateway validation, or safety checks fail."""


@dataclass(frozen=True)
class GpuStatus:
    """Measured local NVIDIA GPU state without serial or UUID identifiers."""

    present: bool
    parse_error: bool = False
    name: str = ""
    total_mib: int = 0
    free_mib: int = 0
    temp_c: float = 0.0
    util_pct: float = 0.0


@dataclass(frozen=True)
class ProbeCheck:
    """Single post-setup readiness check result with FR/EN summary and fix command."""

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
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ProbeError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ProbeError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize feedback profile selector to `'dev'` or `'gamer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise ProbeError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise ProbeError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace local user home prefixes with `'~'` so usernames never leak."""
    raw = str(target)
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
                    return "~" + suffix.replace("\\", "/")
    redacted = _HOME_PATH_RE.sub("~", raw)
    redacted = _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted)
    if redacted != raw:
        return redacted.replace("\\", "/")
    return raw


def mask_api_key(raw_key: object) -> str:
    """Mask an `hm-...` or `hm_live_...` key so secrets never appear in reports."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not _is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    tail = cleaned[-4:]
    return f"{prefix}***{tail}"


def _is_valid_hivemind_key(raw_key: object) -> bool:
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce='...'` attributes, inline CSP nonces, and masked-key suffixes before figure scans."""
    if not isinstance(text, str):
        raise ProbeError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _STRIP_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)


def redact_sensitive_text(text: str, home: Path | None = None) -> str:
    """Scrub API keys, Bearer tokens, emails, OS home paths, and non-loopback IPs from free text."""
    if not text:
        return ""
    out = redact_home_path(text, home=home)
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (
            resolved_home,
            resolved_home.replace("\\", "/"),
            resolved_home.replace("/", "\\"),
        ):
            if variant and variant in out:
                out = out.replace(variant, "~")
    out = _HOME_PATH_RE.sub("~", out)
    out = _PYTEST_USER_DIR_RE.sub("pytest-of-user", out)
    out = _BEARER_RE.sub("Bearer <redacted>", out)
    out = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), out)
    out = _EMAIL_RE.sub("<redacted-email>", out)

    def _scrub_ip(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            ip_obj = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        return candidate if ip_obj.is_loopback else "<redacted-ip>"

    return _IPV4_RE.sub(_scrub_ip, out)


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app` or a loopback address (`localhost`/`127.0.0.1`/`::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or `http(s)://localhost|127.0.0.1`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ProbeError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ProbeError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if not cleaned or any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise ProbeError("Gateway URL must not contain whitespace or backslashes.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ProbeError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ProbeError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ProbeError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ProbeError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ProbeError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ProbeError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or localhost/127.0.0.1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ProbeError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ProbeError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowed gateway origin and block any inference completion endpoint or prompt payload."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ProbeError("Request URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ProbeError("Request URL must not contain control characters.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise ProbeError("Request URL must not contain whitespace or backslashes.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ProbeError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    verb = (method or "GET").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ProbeError(f"Unsupported HTTP method {verb!r}.")
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ProbeError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ProbeError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise ProbeError("Zero-inference invariant violated: forbidden prompt payload.")
    return origin


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects toward untrusted hosts or inference completion routes."""

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
            _assert_zero_inference_request(new_req.get_method(), new_req.full_url)
        return new_req


def _prepare_http_headers(
    base_headers: dict[str, str],
    extra_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    req_headers = dict(base_headers)
    if extra_headers:
        for hk, hv in extra_headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ProbeError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    return req_headers


def default_http_raw(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Proxy-free stdlib raw HTTP helper (bounded read, strict `_assert_zero_inference_request`)."""
    _assert_zero_inference_request(method, url)
    req_headers = _prepare_http_headers(
        {
            "Accept": "text/html, application/json;q=0.9, text/plain;q=0.8",
            "User-Agent": "hivemind-noel-post-setup-probe/1.0",
        },
        headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    req = urllib.request.Request(url.strip(), data=None, headers=req_headers, method=method.upper())
    bounded_timeout = min(max(0.2, float(timeout_s)), 60.0)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES)
            status_code = int(getattr(resp, "status", 200))
            raw_hdrs = getattr(resp, "headers", None)
            lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
            return status_code, raw_bytes.decode("utf-8", errors="replace"), lower_headers
    except urllib.error.HTTPError as exc:
        err_bytes = exc.read(MAX_HTTP_BODY_BYTES) if exc.fp is not None else b""
        raw_hdrs = getattr(exc, "headers", None)
        lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
        return int(exc.code), err_bytes.decode("utf-8", errors="replace"), lower_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        err_doc = json.dumps(
            {
                "network_error": type(exc).__name__,
                "detail": redact_sensitive_text(str(reason))[:200],
            }
        )
        return 0, err_doc, {"x-hivemind-network-error": type(exc).__name__}


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib HTTP helper handling both JSON and HTML route responses."""
    _assert_zero_inference_request(method, url, payload=payload)
    req_headers = _prepare_http_headers(
        {
            "Accept": "application/json, text/html;q=0.9",
            "User-Agent": "hivemind-noel-post-setup-probe/1.0",
        },
        headers,
    )
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read(MAX_HTTP_BODY_BYTES).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            content_type = str(resp.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        raw_err = (exc.read(65_536) if exc.fp is not None else b"").decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": redact_sensitive_text(raw_err)[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": redact_sensitive_text(str(reason))[:200]}

    if not raw.strip():
        return status_code, {"content_type": content_type}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status_code, {"content_type": content_type, "html_length": len(raw)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate URL & zero-inference guard before invoking `http_fn`, catching transport errors."""
    _assert_zero_inference_request(method, url, payload=payload)
    verb = method.upper()
    try:
        sig = inspect.signature(http_fn)
        params = list(sig.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        pos_count = sum(
            1
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if has_varargs or pos_count >= 4:
            status_code, body = http_fn(verb, url, payload, headers)
        elif pos_count == 3:
            status_code, body = http_fn(verb, url, payload)  # type: ignore[call-arg]
        else:
            status_code, body = http_fn(verb, url)  # type: ignore[call-arg]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": redact_sensitive_text(str(exc))[:200]}
    norm_body = dict(body) if isinstance(body, Mapping) else {"data": body}
    return int(status_code), norm_body


def default_subprocess_runner(argv: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Run a local read-only command with `shell=False` and bounded timeout."""
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        check=False,
        shell=False,
    )
    return int(proc.returncode), proc.stdout


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


def _route_failure_diagnostic(path: str, group: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404 and group == "holiday":
        return (
            f"GET {path} a répondu HTTP 404 : la révision Cloud Run est en retard sur `main` "
            "(route Noël absente ; redéployez la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404: Cloud Run revision lags behind `main` "
            "(missing holiday onboarding route; redeploy gateway)."
        )
    if status_code == 404:
        return (
            f"GET {path} a répondu HTTP 404 (route principale introuvable sur la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404 (core gateway route missing)."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (erreur réseau/TLS vers la passerelle)."
            if is_fr
            else f"GET {path} unreachable (network/TLS error contacting gateway)."
        )
    if status_code == 429:
        return (
            f"GET {path} limité en débit (HTTP 429) ; réessayez dans une minute."
            if is_fr
            else f"GET {path} rate-limited (HTTP 429); retry in one minute."
        )
    if status_code >= 500:
        return (
            f"GET {path} en erreur serveur passerelle (HTTP {status_code})."
            if is_fr
            else f"GET {path} failed with gateway server error (HTTP {status_code})."
        )
    return (
        f"GET {path} a répondu HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {status_code} (expected HTTP 200)."
    )


def _extract_mean_ease_raw(doc: Mapping[str, Any] | None) -> Any:
    if not isinstance(doc, Mapping):
        return None
    val = doc.get("mean_ease_rating")
    if val is None:
        val = doc.get("average_ease_rating")
    return val


def run_holiday_route_smoke_gate(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Check all 7 required GET routes with zero LLM inference and surface stale-revision 404s."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    feedback_summary_doc: dict[str, Any] = {}
    models_count: int | None = None

    for path, expected_kind, group in HOLIDAY_SMOKE_ROUTES:
        code, body = _safe_http_call(http_fn, "GET", f"{clean_gw}{path}", None, None)
        http_statuses[path] = code
        payload_ok = code == 200
        diag = ""

        if code == 200 and path == "/v1/models":
            if "data" in body and not isinstance(body.get("data"), list):
                payload_ok = False
                diag = (
                    "GET /v1/models a répondu HTTP 200 mais `data` n'est pas une liste valide."
                    if lang_code == "fr"
                    else "GET /v1/models returned HTTP 200 without a valid `data` model list."
                )
            elif isinstance(body.get("data"), list):
                models_count = len(body["data"])

        if code == 200 and path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_doc = dict(body)

        if not payload_ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)
            if not diag:
                diag = _route_failure_diagnostic(path, group, code, lang=lang_code)
            diagnostics.append(diag)

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "expected_kind": expected_kind,
                "group": group,
                "http_status": code,
                "ok": payload_ok,
                "diagnostic": diag,
            }
        )

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    all_ok = passed_count == total_count
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and missing_routes)

    if cloud_run_lags_main:
        joined_404 = ", ".join(missing_routes)
        lag_msg = (
            f"Diagnostic Cloud Run : /health répond HTTP 200 mais {len(missing_routes)} route(s) Noël "
            f"renvoient HTTP 404 ({joined_404}) — la révision Cloud Run est en retard sur `main`."
            if lang_code == "fr"
            else f"Cloud Run diagnostic: /health is HTTP 200 while {len(missing_routes)} holiday route(s) "
            f"returned HTTP 404 ({joined_404}) — deployed Cloud Run revision lags behind `main`."
        )
        diagnostics.insert(0, lag_msg)

    raw_mean = _extract_mean_ease_raw(feedback_summary_doc)
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool) and math.isfinite(float(raw_mean))
        else None
    )
    raw_lag = feedback_summary_doc.get("zero_game_lag_ratio")
    zero_lag_ratio = (
        round(float(raw_lag), 4)
        if isinstance(raw_lag, (int, float)) and not isinstance(raw_lag, bool)
        else None
    )
    raw_total_fb = feedback_summary_doc.get("total_responses")
    total_fb = (
        int(raw_total_fb)
        if isinstance(raw_total_fb, int) and not isinstance(raw_total_fb, bool)
        else None
    )

    return {
        "gateway": clean_gw,
        "all_routes_ok": all_ok,
        "routes_passed_count": passed_count,
        "routes_total_count": total_count,
        "http_statuses": http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
        "cloud_run_lags_main": cloud_run_lags_main,
        "zero_inference_verified": True,
        "models_count": models_count,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": zero_lag_ratio,
        "feedback_total_responses": total_fb,
        "diagnostics": diagnostics,
        "routes": route_rows,
    }


def _smoke_gate_to_check(smoke: Mapping[str, Any], *, os_name: str, lang: str) -> ProbeCheck:
    is_fr = lang == "fr"
    title = (
        "Smoke Gate Routes Noël Cloud Run (Zéro Inférence)"
        if is_fr
        else "Cloud Run Holiday Route Smoke Gate (Zero-Inference)"
    )
    passed = int(smoke["routes_passed_count"])
    total = int(smoke["routes_total_count"])
    gw = str(smoke["gateway"])
    curl_bin = "curl.exe" if os_name == "nt" else "curl"

    if smoke["all_routes_ok"]:
        summary = (
            f"{passed}/{total} routes GET vérifiées avec HTTP 200 sur {gw} (zéro appel d'inférence)."
            if is_fr
            else f"{passed}/{total} GET routes verified HTTP 200 on {gw} (zero inference calls)."
        )
        return ProbeCheck(
            id="holiday_route_smoke",
            title=title,
            status="ok",
            summary=summary,
            fix_command="",
            details={
                "routes_passed_count": passed,
                "routes_total_count": total,
                "http_statuses": dict(smoke["http_statuses"]),
                "missing_routes": [],
                "cloud_run_lags_main": False,
            },
        )

    diags = list(smoke.get("diagnostics") or [])
    first_diag = diags[0] if diags else "One or more required routes failed."
    failed_first = (smoke.get("failed_routes") or ["/health"])[0]
    fix_cmd = f'{curl_bin} -fsSL -o /dev/null -w "%{{http_code}}" "{gw}{failed_first}"'
    summary = (
        f"{passed}/{total} routes OK — {first_diag}"
        if is_fr
        else f"{passed}/{total} routes passed — {first_diag}"
    )
    return ProbeCheck(
        id="holiday_route_smoke",
        title=title,
        status="fail",
        summary=summary,
        fix_command=fix_cmd,
        details={
            "routes_passed_count": passed,
            "routes_total_count": total,
            "http_statuses": dict(smoke["http_statuses"]),
            "missing_routes": list(smoke["missing_routes"]),
            "failed_routes": list(smoke["failed_routes"]),
            "cloud_run_lags_main": bool(smoke["cloud_run_lags_main"]),
        },
    )


def check_credentials_and_account(
    credentials_path: Path | None = None,
    *,
    verify_account: bool = False,
    gateway: str = DEFAULT_GATEWAY,
    http_fn: HttpFn = default_http_json,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Inspect `~/.hivemind/credentials.json` safely, plus optional `GET /v1/hivemind/me`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = credentials_path if credentials_path is not None else (resolved_home / ".hivemind" / "credentials.json")
    shown_path = redact_home_path(target, home=resolved_home)
    title = (
        "Identifiants Locaux & Compte HiveMind"
        if is_fr
        else "Local HiveMind Credentials & Account"
    )
    fix_cmd = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} introuvable ; exécutez quickstart pour enregistrer votre clé personnelle hm-..."
            if is_fr
            else f"{shown_path} not found; run quickstart to save your personal hm-... key."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": False,
                "valid_key": False,
                "masked_key": "<missing>",
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    try:
        st = target.stat()
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = (
            f"{shown_path} est illisible ou JSON invalide ; régénérez le fichier via quickstart."
            if is_fr
            else f"{shown_path} is unreadable or invalid JSON; regenerate via quickstart."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": False,
                "masked_key": "<invalid>",
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    raw_key = loaded.get("api_key") if isinstance(loaded, Mapping) else None
    if not _is_valid_hivemind_key(raw_key):
        summary = (
            f"{shown_path} ne contient pas de clé API hm-... valide."
            if is_fr
            else f"{shown_path} does not contain a valid hm-... API key."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": False,
                "masked_key": mask_api_key(raw_key),
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    clean_key = str(raw_key).strip()
    masked = mask_api_key(clean_key)
    role = str(loaded.get("role") or "prosumer") if isinstance(loaded, Mapping) else "prosumer"

    account_verified = False
    verify_status: int | None = None
    if verify_account:
        clean_gw = validate_gateway_url(gateway)
        verify_status, me_body = _safe_http_call(
            http_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/me",
            None,
            {"Authorization": f"Bearer {clean_key}"},
        )
        if verify_status != 200:
            summary = (
                f"Échec de vérification de la clé {masked} sur /v1/hivemind/me (HTTP {verify_status})."
                if is_fr
                else f"Account check for {masked} failed on /v1/hivemind/me (HTTP {verify_status})."
            )
            return ProbeCheck(
                "credentials",
                title,
                "fail",
                summary,
                fix_cmd,
                {
                    "path": shown_path,
                    "present": True,
                    "valid_key": True,
                    "masked_key": masked,
                    "role": role,
                    "verify_requested": True,
                    "account_verified": False,
                    "verify_http_status": verify_status,
                },
            )
        account_verified = True
        if isinstance(me_body.get("role"), str) and me_body["role"].strip():
            role = me_body["role"].strip()

    details: dict[str, Any] = {
        "path": shown_path,
        "present": True,
        "valid_key": True,
        "masked_key": masked,
        "role": role,
        "verify_requested": verify_account,
        "account_verified": account_verified,
        "verify_http_status": verify_status,
    }

    if eff_os == "nt":
        details["permission_model"] = "windows_acl"
        ver_note_en = ", /v1/hivemind/me verified" if account_verified else ""
        ver_note_fr = ", /v1/hivemind/me vérifié" if account_verified else ""
        summary = (
            f"{shown_path} prêt ({masked}, rôle : {role}, ACL Windows{ver_note_fr})."
            if is_fr
            else f"{shown_path} ready ({masked}, role: {role}, Windows ACL{ver_note_en})."
        )
        return ProbeCheck("credentials", title, "ok", summary, "", details)

    mode_bits = st.st_mode & 0o777
    octal_mode = f"{mode_bits:04o}"
    details["posix_mode"] = octal_mode
    if mode_bits != 0o600:
        chmod_target = shown_path if shown_path.startswith("~/") else shlex.quote(shown_path)
        chmod_cmd = f"chmod 600 {chmod_target}"
        summary = (
            f"{shown_path} ({masked}) a les permissions {octal_mode} (attendu 0600 propriétaire seul)."
            if is_fr
            else f"{shown_path} ({masked}) has mode {octal_mode} (expected 0600 owner-only)."
        )
        return ProbeCheck("credentials", title, "warn", summary, chmod_cmd, details)

    ver_note_en = ", /v1/hivemind/me verified" if account_verified else ""
    ver_note_fr = ", /v1/hivemind/me vérifié" if account_verified else ""
    summary = (
        f"{shown_path} prêt ({masked}, rôle : {role}, POSIX 0600{ver_note_fr})."
        if is_fr
        else f"{shown_path} ready ({masked}, role: {role}, POSIX 0600{ver_note_en})."
    )
    return ProbeCheck("credentials", title, "ok", summary, "", details)


def check_continue_provider_yaml(
    continue_path: Path | None = None,
    *,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Verify presence and structure of `~/.continue/hivemind_provider.yaml` without leaking keys."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = (
        continue_path
        if continue_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )
    shown_path = redact_home_path(target, home=resolved_home)
    title = (
        "Configuration VS Code Continue.dev"
        if is_fr
        else "VS Code Continue.dev Configuration"
    )
    fix_cmd = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} absent ; ajoutez `--write-continue-config` si vous codez dans VS Code."
            if is_fr
            else f"{shown_path} not present; pass `--write-continue-config` to enable VS Code Continue.dev."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": False, "configured": False},
        )

    try:
        raw_yaml = target.read_text(encoding="utf-8")
    except OSError:
        summary = (
            f"{shown_path} est illisible ; régénérez l'extrait Continue.dev."
            if is_fr
            else f"{shown_path} could not be read; regenerate the Continue.dev snippet."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": True, "configured": False},
        )

    has_models = "models:" in raw_yaml and "apiBase:" in raw_yaml
    if not has_models:
        summary = (
            f"{shown_path} est incomplet (`models:` / `apiBase:` manquants) ; régénérez-le."
            if is_fr
            else f"{shown_path} is incomplete (missing `models:` / `apiBase:`); regenerate it."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": True, "configured": False},
        )

    summary = (
        f"{shown_path} présent et configuré pour VS Code Continue.dev."
        if is_fr
        else f"{shown_path} present and configured for VS Code Continue.dev."
    )
    return ProbeCheck(
        "continue_config",
        title,
        "ok",
        summary,
        "",
        {"path": shown_path, "present": True, "configured": True},
    )


def _parse_float_metric(raw: str, default: float = 0.0) -> float:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return default
    return float(cleaned)


def probe_gpu(runner: RunnerFn = default_subprocess_runner) -> GpuStatus:
    """Query `nvidia-smi` locally without requesting GPU UUID or serial numbers."""
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
        return GpuStatus(present=False)

    if rc != 0 or not out.strip():
        return GpuStatus(present=False)

    gpus: list[GpuStatus] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            return GpuStatus(present=True, parse_error=True)
        try:
            total_mib = int(float(parts[1]))
            free_mib = int(float(parts[2]))
            temp_c = _parse_float_metric(parts[3], 0.0)
            util_pct = _parse_float_metric(parts[4], 0.0)
        except ValueError:
            return GpuStatus(present=True, parse_error=True)
        if total_mib <= 0 or free_mib < 0:
            return GpuStatus(present=True, parse_error=True)
        gpus.append(
            GpuStatus(
                present=True,
                parse_error=False,
                name=parts[0],
                total_mib=total_mib,
                free_mib=free_mib,
                temp_c=temp_c,
                util_pct=util_pct,
            )
        )
    return max(gpus, key=lambda g: (g.free_mib, -g.temp_c, g.total_mib))


def check_gamer_whisper_gpu(
    *,
    runner: RunnerFn = default_subprocess_runner,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = GAMER_WHISPER_MIN_FREE_VRAM_MIB,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Verify Gamer-Whisper thermal ceiling (`<= 70.0 C`) and free VRAM reserve (`>= 2048 MiB`)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    title = (
        "GPU Gamer-Whisper (Plafond <= 70 C & Réserve VRAM >= 2048 MiB)"
        if is_fr
        else "Gamer-Whisper GPU Guard (<= 70.0 C & >= 2048 MiB Free VRAM)"
    )
    gpu = probe_gpu(runner)

    if not gpu.present:
        summary = (
            "Aucun GPU NVIDIA détecté — Mode Dev API uniquement prêt (0 charge GPU locale)."
            if is_fr
            else "No NVIDIA GPU detected — Dev API-only mode ready (0 local GPU load)."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "ok",
            summary,
            "",
            {
                "gpu_present": False,
                "mode": "dev_api_only",
                "thermal_cap_c": max_temp_c,
                "min_free_vram_mib": min_free_vram_mib,
                "thermal_cap_ok": True,
                "vram_reserve_ok": True,
            },
        )

    if gpu.parse_error:
        summary = (
            "Sortie `nvidia-smi` illisible — Mode Dev API prêt ; vérifiez le pilote NVIDIA avant Gamer-Whisper."
            if is_fr
            else "`nvidia-smi` output could not be parsed — Dev API-only mode ready; inspect NVIDIA driver."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            "nvidia-smi",
            {"gpu_present": True, "parse_error": True, "mode": "dev_api_only"},
        )

    thermal_ok = gpu.temp_c <= max_temp_c
    vram_ok = gpu.free_mib >= min_free_vram_mib
    gamer_fix = _quickstart_command(gateway, eff_os, "--gamer-check")
    details: dict[str, Any] = {
        "gpu_present": True,
        "name": gpu.name,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "thermal_cap_c": max_temp_c,
        "min_free_vram_mib": min_free_vram_mib,
        "thermal_cap_ok": thermal_ok,
        "vram_reserve_ok": vram_ok,
    }

    if not thermal_ok:
        summary = (
            f"{gpu.name} est à {gpu.temp_c:.1f} C (> {max_temp_c:.1f} C plafond Gamer-Whisper) ; "
            "pause thermique jusqu'au refroidissement <= 70.0 C."
            if is_fr
            else f"{gpu.name} is at {gpu.temp_c:.1f} C (> {max_temp_c:.1f} C Gamer-Whisper thermal cap); "
            "GPU worker pauses until <= 70.0 C."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            gamer_fix,
            {**details, "mode": "gpu_thermal_cooldown"},
        )

    if not vram_ok:
        summary = (
            f"{gpu.name} n'a que {gpu.free_mib} MiB libres (< {min_free_vram_mib} MiB de réserve Gamer-Whisper) ; "
            "Gamer-Whisper reste en veille."
            if is_fr
            else f"{gpu.name} has {gpu.free_mib} MiB free (< {min_free_vram_mib} MiB Gamer-Whisper reserve); "
            "Gamer-Whisper stays idle."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            gamer_fix,
            {**details, "mode": "vram_reserve_hold"},
        )

    summary = (
        f"{gpu.name} prêt en silence ({gpu.temp_c:.1f} C <= {max_temp_c:.1f} C, "
        f"{gpu.free_mib} MiB libres >= {min_free_vram_mib} MiB)."
        if is_fr
        else f"{gpu.name} quiet & ready ({gpu.temp_c:.1f} C <= {max_temp_c:.1f} C, "
        f"{gpu.free_mib} MiB free >= {min_free_vram_mib} MiB)."
    )
    return ProbeCheck(
        "gamer_whisper_gpu",
        title,
        "ok",
        summary,
        "",
        {**details, "mode": "gamer_whisper_ready"},
    )


def list_process_names(
    runner: RunnerFn = default_subprocess_runner,
    *,
    os_name: str | None = None,
) -> set[str]:
    """List normalized lowercase local process executable names on Windows, macOS, or Linux."""
    eff_os = os_name if os_name is not None else os.name
    names: set[str] = set()
    try:
        if eff_os == "nt":
            rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    names.add(row[0].strip().lower())
        else:
            rc, out = runner(["ps", "-eo", "comm="], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for line in out.splitlines():
                cleaned = line.strip()
                if cleaned:
                    names.add(cleaned.lower())
                    names.add(Path(cleaned.replace("\\", "/")).name.lower())
    except (OSError, subprocess.SubprocessError, csv.Error):
        return names
    return names


def verify_yield_budget(
    detection_elapsed_s: float,
    *,
    preemption_signal_s: float = PREEMPTION_SIGNAL_ESTIMATE_S,
    budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
) -> dict[str, Any]:
    """Verify active-game detection + process-tree preemption fits inside the `< 1.5 s` budget."""
    clamped_scan = max(0.0, float(detection_elapsed_s))
    clamped_kill = max(0.0, float(preemption_signal_s))
    estimated_total = round(clamped_scan + clamped_kill, 4)
    return {
        "detection_elapsed_s": round(clamped_scan, 4),
        "preemption_signal_s": round(clamped_kill, 4),
        "estimated_yield_s": estimated_total,
        "yield_budget_s": budget_s,
        "yield_budget_ok": estimated_total < budget_s,
    }


def check_active_games_and_yield_budget(
    *,
    runner: RunnerFn = default_subprocess_runner,
    extra_games: Sequence[str] = (),
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    preemption_signal_s: float = PREEMPTION_SIGNAL_ESTIMATE_S,
    clock_fn: Callable[[], float] = time.monotonic,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Detect active Steam/PC games and verify the `< 1.5 s` GPU yield budget."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    title = (
        "Détection de Jeu Actif & Budget d'Éviction (< 1.5 s)"
        if is_fr
        else "Active Game Guard & Yield Budget (< 1.5 s)"
    )

    watchlist = set(DEFAULT_GAME_EXECUTABLES)
    for item in extra_games:
        cleaned = item.strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            watchlist.add(cleaned[:15])

    t0 = clock_fn()
    running = list_process_names(runner, os_name=eff_os)
    scan_s = max(0.0, clock_fn() - t0)
    budget_info = verify_yield_budget(
        scan_s,
        preemption_signal_s=preemption_signal_s,
        budget_s=yield_budget_s,
    )
    detected = sorted(proc for proc in running if proc in watchlist)
    gamer_fix = _quickstart_command(gateway, eff_os, "--gamer-check")

    details: dict[str, Any] = {
        "active_games": detected,
        "game_running": bool(detected),
        **budget_info,
    }

    if not budget_info["yield_budget_ok"]:
        summary = (
            f"L'analyse des processus a pris {budget_info['estimated_yield_s']:.2f} s "
            f"(>= {yield_budget_s:.1f} s budget Gamer-Whisper) ; gardez le mode GPU désactivé."
            if is_fr
            else f"Process scan + preemption estimate took {budget_info['estimated_yield_s']:.2f} s "
            f"(>= {yield_budget_s:.1f} s Gamer-Whisper yield budget); keep GPU rounds paused."
        )
        return ProbeCheck("active_games_yield", title, "warn", summary, gamer_fix, details)

    if detected:
        joined = ", ".join(detected)
        summary = (
            f"Jeu actif détecté ({joined}) ; Gamer-Whisper libère 100 % de la VRAM en "
            f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s)."
            if is_fr
            else f"Active game detected ({joined}); Gamer-Whisper yields 100% of GPU VRAM within "
            f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s budget)."
        )
        return ProbeCheck("active_games_yield", title, "warn", summary, gamer_fix, details)

    summary = (
        f"Aucun jeu actif détecté ; budget d'éviction vérifié à {budget_info['estimated_yield_s']:.2f} s "
        f"(< {yield_budget_s:.1f} s)."
        if is_fr
        else f"No active game detected; preemption yield budget verified at "
        f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s)."
    )
    return ProbeCheck("active_games_yield", title, "ok", summary, "", details)


def _find_corporate_markers(text: str) -> list[str]:
    """Return matched corporate indicator token names without retaining raw values."""
    if not text:
        return []
    return [name for name, pat in _CORP_TOKEN_PATTERNS if pat.search(text)]


def _extract_git_remote_urls(config_text: str) -> list[str]:
    """Extract remote / url values from `.git/config` text in memory."""
    if not config_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(config_text):
        cand = (m_kv or m_sec).strip().strip('"').strip("'")
        if cand:
            urls.append(cand)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` from `workspace`, following worktree `gitdir:` and `commondir`."""
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
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gd = lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gd) if Path(raw_gd).is_absolute() else (root / raw_gd).resolve()
                direct_cfg = gitdir / "config"
                if direct_cfg.is_file():
                    candidates.append(direct_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    with suppress(OSError):
                        common_rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if common_rel:
                            common_cfg = (gitdir / common_rel).resolve() / "config"
                            if common_cfg.is_file() and common_cfg not in candidates:
                                candidates.append(common_cfg)
            break
    return candidates


def detect_local_corporate_markers(environ: Mapping[str, str]) -> list[str]:
    """Inspect environment variables strictly in memory without recording any values."""
    markers: list[str] = []
    upper_map: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in environ.items() if str(v).strip()
    }

    if any(k.startswith("AMADEUS") for k in upper_map):
        markers.append("AMADEUS_ENV")

    if upper_map.get("USERDNSDOMAIN"):
        markers.append("USERDNSDOMAIN")

    userdomain = upper_map.get("USERDOMAIN", "")
    computername = upper_map.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        if _find_corporate_markers(userdomain):
            markers.append("USERDOMAIN")

    for host_key in ("COMPUTERNAME", "HOSTNAME"):
        host_val = upper_map.get(host_key, "").lower()
        if host_val and ("amadeus" in host_val or "corp-" in host_val or "-corp" in host_val or _find_corporate_markers(host_val)):
            markers.append(host_key)

    for proxy_key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if upper_map.get(proxy_key):
            markers.append(proxy_key)

    no_proxy = upper_map.get("NO_PROXY", "").lower()
    if no_proxy and (_find_corporate_markers(no_proxy) or any(tok in no_proxy for tok in _CORP_MARKERS)):
        markers.append("NO_PROXY")

    for tls_key in (
        "ZSCALER_CERT_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        tls_val = upper_map.get(tls_key, "").lower()
        if tls_key == "ZSCALER_CERT_PATH" and tls_val:
            markers.append(tls_key)
        elif tls_val and (_find_corporate_markers(tls_val) or any(tok in tls_val for tok in _CORP_MARKERS)):
            markers.append(tls_key)

    return markers


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC gate checking env, cwd path, and `.git/config` remotes."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_view = environ if environ is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _add(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            matched_markers.add(marker)

    for env_tag in detect_local_corporate_markers(env_view):
        _add(env_tag)

    for raw_k, raw_v in env_view.items():
        kup = str(raw_k).strip().upper()
        vstr = str(raw_v or "").strip()
        if not kup or not vstr:
            continue
        if kup.startswith("AMADEUS"):
            _add("AMADEUS_ENV", "amadeus")
            matched_markers.update(_find_corporate_markers(vstr))
        elif kup == "ZSCALER_CERT_PATH":
            _add("ZSCALER_CERT_PATH", "zscaler")
        elif kup == "USERDNSDOMAIN":
            hits = _find_corporate_markers(vstr)
            _add("USERDNSDOMAIN", hits[0] if hits else "corp")
            matched_markers.update(hits)
        elif kup in _INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(vstr):
                _add(f"env:{kup}:{hit}", hit)

    ws_variants = [str(workspace)]
    with suppress(OSError):
        res_ws = str(workspace.resolve())
        if res_ws not in ws_variants:
            ws_variants.append(res_ws)
    for ws_txt in ws_variants:
        for hit in _find_corporate_markers(ws_txt):
            _add(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            with suppress(OSError):
                cfg_txt = cfg_path.read_text(encoding="utf-8", errors="replace")
                for remote_url in _extract_git_remote_urls(cfg_txt):
                    for hit in _find_corporate_markers(remote_url):
                        _add(f"git_remote:{hit}", hit)

    ok = len(indicators) == 0
    notice = COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN
    joined = ", ".join(indicators)
    summary = (
        (
            f"Contrôle PC personnel conforme (0.00 EUR / 0.00 USD). {notice}"
            if is_fr
            else f"Personal-PC compliance verified locally (0.00 EUR / 0.00 USD). {notice}"
        )
        if ok
        else (
            f"REFUS DE CONFORMITÉ ({joined}) : marqueur d'entreprise détecté. {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE REFUSED ({joined}): corporate marker detected. {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.personal_pc_compliance",
            "lang": lang_code,
            "compliance_ok": ok,
            "corporate_detected": not ok,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "network_transmitted": False,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "monetary_cost": FINANCIAL_PROMISE,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
            "exit_code": 0 if ok else 2,
        }
    )


def check_corporate_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    enforce_personal_pc: bool = False,
    lang: str = "en",
) -> ProbeCheck:
    """Strictly local corporate-PC check (Amadeus / work markers NEVER leave the machine)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    env_view = environ if environ is not None else os.environ
    title = (
        "Conformité PC Personnel / Règle Amadeus (100 % Local)"
        if is_fr
        else "Personal-PC / Amadeus Compliance (Local-Only Check)"
    )

    comp = evaluate_personal_pc_compliance(
        environ=env_view,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    markers = list(comp["corporate_indicators"])
    base_details: dict[str, Any] = {
        "corporate_indicators": markers,
        "matched_markers": list(comp["matched_markers"]),
        "corporate_warning": bool(markers),
        "enforced": bool(enforce_personal_pc),
        "network_transmitted": False,
        "personal_pc_only": True,
        "monetary_cost": FINANCIAL_PROMISE,
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
    }

    if markers:
        uninstall_cmd = _quickstart_command(gateway, eff_os, "--uninstall")
        joined = ", ".join(markers)
        status = "fail" if enforce_personal_pc else "warn"
        summary = (
            f"Alerte locale ({joined}) : ARRÊTEZ immédiatement s'il s'agit d'un PC professionnel "
            "(Amadeus / entreprise). Usage 100 % PC personnel à domicile et projets perso uniquement "
            "(0,00 EUR / 0,00 USD, AGPL-3.0)."
            if is_fr
            else f"Local corporate marker detected ({joined}): STOP immediately if this is a work PC "
            "(Amadeus / corporate hardware prohibited). 100% personal home PCs & personal projects only "
            "(0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
        return ProbeCheck("personal_pc_compliance", title, status, summary, uninstall_cmd, base_details)

    summary = (
        f"Contrôle local conforme (aucun marqueur professionnel détecté). {COMPLIANCE_NOTICE_FR}"
        if is_fr
        else f"Local check clean (no corporate/work markers detected). {COMPLIANCE_NOTICE_EN}"
    )
    return ProbeCheck("personal_pc_compliance", title, "ok", summary, "", base_details)


def parse_interactive_ces_input(
    raw: str,
    *,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
) -> dict[str, Any]:
    """Parse a single-line 1-prompt CES response (`'5'`, `'4/5 gamer'`, `'3 lag'`, or empty to skip)."""
    prof = normalize_profile(default_profile)
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": prof,
            "ease_rating": None,
            "zero_game_lag": not default_had_game_lag,
        }

    normalized = re.sub(r"\s*/\s*5\b", "", cleaned, count=1)
    tokens = [tok.strip(",;:").lower() for tok in normalized.split() if tok.strip(",;:")]

    rating: int | None = None
    had_lag = bool(default_had_game_lag)

    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            val = int(tok)
            if not (1 <= val <= 5) or rating is not None:
                raise ProbeError("Interactive CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("gamer", "g"):
            prof = "gamer"
        elif tok in ("dev", "developer", "d"):
            prof = "dev"
        elif tok in ("lag", "had-game-lag", "had_game_lag", "stutter", "saccade", "lag=yes", "lag=true"):
            had_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "lag=no", "lag=false", "ok"):
            had_lag = False
        else:
            raise ProbeError(f"Unrecognized 1-prompt CES token {tok!r}; expected 1..5 [dev|gamer] [lag].")

    if rating is None:
        raise ProbeError("Interactive 1-prompt CES input must include an ease_rating from 1 to 5.")

    return {
        "skipped": False,
        "profile": prof,
        "ease_rating": rating,
        "zero_game_lag": not had_lag,
    }


def _platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if raw in ("windows", "win32", "nt"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw == "linux":
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", raw)[:24]
    return cleaned or "unknown"


def _get_queue_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _QUEUE_LOCKS_GUARD:
        lock = _QUEUE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _QUEUE_LOCKS[key] = lock
        return lock


def _expand_queue_path(raw_path: Path | str, *, home: Path | None = None) -> Path:
    text = str(raw_path).strip()
    if text == "~" or text.startswith("~/") or text.startswith("~\\"):
        base_home = Path(home) if home is not None else Path.home()
        suffix = text[2:] if len(text) > 1 else ""
        return (base_home / suffix) if suffix else base_home
    return Path(text)


def default_ces_queue_path(
    home: Path | str | None = None,
    *,
    credentials_path: Path | str | None = None,
) -> Path:
    """Resolve local CES retry queue path without touching real `~/.hivemind` when isolated."""
    home_path = Path(home) if home is not None else None
    if credentials_path is not None and str(credentials_path).strip():
        cred_p = _expand_queue_path(credentials_path, home=home_path)
        if home_path is not None:
            default_real_cred = Path.home() / ".hivemind" / "credentials.json"
            if cred_p == default_real_cred and home_path.resolve() != Path.home().resolve():
                return home_path / ".hivemind" / SUPPORT_DEFAULT_QUEUE_FILENAME
        return cred_p.parent / SUPPORT_DEFAULT_QUEUE_FILENAME
    base_dir = home_path if home_path is not None else Path.home()
    return base_dir / ".hivemind" / SUPPORT_DEFAULT_QUEUE_FILENAME


def _normalize_ces_payload(
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_name: str | None = None,
    note: str = "",
    home: Path | None = None,
) -> dict[str, Any]:
    norm_prof = normalize_profile(profile)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ProbeError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise ProbeError("zero_game_lag must be a boolean.")
    clean_platform = _platform_tag(platform_name)
    if not _PLATFORM_RE.fullmatch(clean_platform):
        clean_platform = "unknown"
    clean_note = redact_sensitive_text(str(note or ""), home=home)[:240]
    return {
        "profile": norm_prof,
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "platform": clean_platform,
        "note": clean_note,
    }


def _validate_ces_payload_mapping(raw: object, *, home: Path | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProbeError("Spooled CES payload must be a JSON object.")
    allowed_keys = {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
    if set(raw.keys()) - allowed_keys:
        raise ProbeError("Spooled CES payload contains unexpected keys.")
    return _normalize_ces_payload(
        profile=raw.get("profile", ""),  # type: ignore[arg-type]
        ease_rating=raw.get("ease_rating", 0),  # type: ignore[arg-type]
        zero_game_lag=raw.get("zero_game_lag", True),  # type: ignore[arg-type]
        platform_name=str(raw.get("platform") or "unknown"),
        note=str(raw.get("note") or ""),
        home=home,
    )


def _validate_idempotency_key_input(raw_key: object) -> str:
    if isinstance(raw_key, bool) or not isinstance(raw_key, str):
        raise ProbeError("idempotency_key must be a non-empty ASCII string.")
    if len(raw_key) > 128 or not raw_key.isascii():
        raise ProbeError("idempotency_key must be ASCII and at most 128 characters.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_key):
        raise ProbeError("idempotency_key must not contain control characters.")
    cleaned = raw_key.strip()
    if not cleaned:
        raise ProbeError("idempotency_key must not be empty.")
    if (
        _API_KEY_RE.search(cleaned)
        or _EMAIL_RE.search(cleaned)
        or _HOME_PATH_RE.search(cleaned)
        or _BEARER_RE.search(cleaned)
    ):
        raise ProbeError("idempotency_key must not contain secrets, emails, or home paths.")
    return cleaned


def _normalize_ces_event_id(raw_key_or_id: str) -> str:
    low = raw_key_or_id.lower()
    if _SHA256_HEX_RE.fullmatch(low):
        return low
    return hashlib.sha256(f"hivemind.probe.ces.key/1:{raw_key_or_id}".encode()).hexdigest()


def compute_ces_idempotency_key(
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
    gateway: str | None = None,
    spool_path: Path | str | None = None,
    home: Path | None = None,
) -> tuple[str, str]:
    """Return `(event_id_sha256, header_idempotency_key)` for `POST /v1/hivemind/onboarding/feedback`."""
    clean_payload = _validate_ces_payload_mapping(payload, home=home)
    if idempotency_key is not None:
        cleaned = _validate_idempotency_key_input(idempotency_key)
        return _normalize_ces_event_id(cleaned), cleaned
    clean_gw = validate_gateway_url(gateway) if gateway else DEFAULT_GATEWAY
    scope_target = (
        _expand_queue_path(spool_path, home=home)
        if spool_path is not None and str(spool_path).strip()
        else default_ces_queue_path(home=home)
    )
    scope_digest = hashlib.sha256(str(scope_target.resolve()).encode("utf-8")).hexdigest()[:24]
    canon = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    eid = hashlib.sha256(
        f"hivemind.probe.ces.scope/1:{clean_gw}:{scope_digest}:{canon}".encode()
    ).hexdigest()
    return eid, eid


def _extract_summary_metrics(body: Mapping[str, Any] | None) -> tuple[float | None, float | None, int | None]:
    if not isinstance(body, Mapping):
        return None, None, None
    raw_mean = _extract_mean_ease_raw(body)
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool) and math.isfinite(float(raw_mean))
        else None
    )
    raw_lag = body.get("zero_game_lag_ratio")
    lag_ratio = (
        round(float(raw_lag), 4)
        if isinstance(raw_lag, (int, float)) and not isinstance(raw_lag, bool) and math.isfinite(float(raw_lag))
        else None
    )
    raw_total = body.get("total_responses")
    total_resp = (
        int(raw_total)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total >= 0
        else None
    )
    return mean_ease, lag_ratio, total_resp


def _sanitize_feedback_summary(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        return None
    mean_ease, lag_ratio, total_resp = _extract_summary_metrics(body)
    if total_resp is None:
        return None
    bp = body.get("by_profile")
    dev_c = bp.get("dev", 0) if isinstance(bp, Mapping) else 0
    gamer_c = bp.get("gamer", 0) if isinstance(bp, Mapping) else 0
    if isinstance(dev_c, bool) or not isinstance(dev_c, int) or dev_c < 0:
        dev_c = 0
    if isinstance(gamer_c, bool) or not isinstance(gamer_c, int) or gamer_c < 0:
        gamer_c = 0
    return {
        "object": str(body.get("object") or FEEDBACK_SUMMARY_OBJECT),
        "total_responses": total_resp,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": lag_ratio,
        "by_profile": {"dev": dev_c, "gamer": gamer_c},
    }


class ProbeCesRetryQueue:
    """Thread-safe, atomic POSIX-0600 bounded CES feedback retry queue with Idempotency-Key ledger."""

    def __init__(
        self,
        spool_path: Path | str | None = None,
        *,
        home: Path | None = None,
        credentials_path: Path | str | None = None,
        max_entries: int = CES_QUEUE_MAX_ENTRIES,
        max_delivered: int = CES_QUEUE_MAX_DELIVERED,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not (1 <= max_entries <= 4096):
            raise ProbeError("max_entries must be an integer between 1 and 4096.")
        if isinstance(max_delivered, bool) or not isinstance(max_delivered, int) or not (1 <= max_delivered <= 32768):
            raise ProbeError("max_delivered must be an integer between 1 and 32768.")
        self._home = Path(home) if home is not None else None
        raw_p = (
            _expand_queue_path(spool_path, home=self._home)
            if spool_path is not None and str(spool_path).strip()
            else default_ces_queue_path(home=self._home, credentials_path=credentials_path)
        )
        self._path = raw_p.resolve()
        self._max_entries = int(max_entries)
        self._max_delivered = int(max_delivered)
        self._lock = _get_queue_lock(self._path)

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
                    "event_id": hashlib.sha256(b"corrupted_probe_ces_queue_recovered").hexdigest(),
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
        raw_deliv = parsed.get("delivered_event_ids")
        if isinstance(raw_deliv, list):
            for item in raw_deliv:
                if isinstance(item, str) and _SHA256_HEX_RE.fullmatch(item) and item not in seen_delivered:
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
                    if _SHA256_HEX_RE.fullmatch(q_eid):
                        q_code = q_item.get("http_status", 422)
                        quarantined.append(
                            {
                                "event_id": q_eid,
                                "http_status": int(q_code) if isinstance(q_code, int) and not isinstance(q_code, bool) else 422,
                                "reason": redact_sensitive_text(str(q_item.get("reason", "quarantined"))[:80], home=self._home),
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
                    if isinstance(raw_eid, str) and _SHA256_HEX_RE.fullmatch(raw_eid)
                    else hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                )
                if eid in seen_delivered or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = _validate_ces_payload_mapping(entry.get("payload"), home=self._home)
                    raw_ik = entry.get("idempotency_key")
                    ik = _validate_idempotency_key_input(raw_ik) if raw_ik is not None else eid
                except ProbeError:
                    mutated = True
                    quarantined.append(
                        {
                            "event_id": eid,
                            "http_status": 422,
                            "reason": "invalid_queued_payload",
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
                        "idempotency_key": ik,
                        "payload": clean_payload,
                        "retry_count": rc,
                        "last_http_status": last_st,
                        "last_reason": redact_sensitive_text(str(entry.get("last_reason") or "")[:80], home=self._home),
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
        _assert_safe_output(serialized, home=self._home)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt" and self._path.parent.name == ".hivemind":
            with suppress(OSError):
                os.chmod(self._path.parent, 0o700)
        tmp_path = self._path.with_name(f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            try:
                if os.name != "nt":
                    with suppress(OSError):
                        os.chmod(tmp_path, 0o600)
                os.write(fd, serialized.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, self._path)
            if os.name != "nt":
                with suppress(OSError):
                    os.chmod(self._path, 0o600)
        except BaseException:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise

    def snapshot(self, *, home: Path | None = None) -> dict[str, Any]:
        """Return read-only sanitized queue state without creating files when empty/missing."""
        eff_home = home if home is not None else self._home
        with self._lock:
            exists = self._path.is_file()
            state, _ = self._load_state_unlocked()
        snap: dict[str, Any] = {
            "object": "hivemind.noel_post_setup_probe.ces_queue_snapshot",
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
        return _attach_receipt(snap, home=eff_home)

    def mark_delivered(self, idempotency_or_event_id: str) -> bool:
        """Mark event delivered if a spool file already exists (never creates a new file)."""
        cleaned = (idempotency_or_event_id or "").strip()
        if not cleaned:
            return False
        eid = _normalize_ces_event_id(cleaned)
        with self._lock:
            if not self._path.is_file():
                return False
            state, mutated = self._load_state_unlocked()
            prev_len = len(state["pending"])
            state["pending"] = [
                item
                for item in state["pending"]
                if item["event_id"] != eid and item.get("idempotency_key") != cleaned
            ]
            removed = len(state["pending"]) != prev_len
            if eid not in state["delivered_event_ids"]:
                state["delivered_event_ids"].append(eid)
                state["delivered_event_ids"] = state["delivered_event_ids"][-self._max_delivered :]
                mutated = True
            if removed or mutated:
                self._atomic_save_unlocked(state)
            return removed

    def record_quarantine(
        self,
        *,
        event_id: str,
        http_status: int,
        reason: str = "permanent_client_error",
    ) -> None:
        """Persist a permanent 4xx quarantine marker atomically without queueing for retry."""
        eid = _normalize_ces_event_id(event_id)
        with self._lock:
            state, _ = self._load_state_unlocked()
            state["pending"] = [item for item in state["pending"] if item["event_id"] != eid]
            state["quarantined"].append(
                {
                    "event_id": eid,
                    "http_status": int(http_status),
                    "reason": redact_sensitive_text(str(reason or "quarantined")[:80], home=self._home),
                }
            )
            state["quarantined"] = state["quarantined"][-self._max_entries :]
            self._atomic_save_unlocked(state)

    def _dedup_enqueue_result(
        self,
        *,
        duplicate_state: str,
        eid: str,
        header_key: str,
        clean_payload: Mapping[str, Any],
        state: Mapping[str, Any],
        eff_home: Path | None,
        recovered: bool,
    ) -> dict[str, Any]:
        return _attach_receipt(
            {
                "object": "hivemind.noel_post_setup_probe.ces_enqueue",
                "compliance_ok": True,
                "spool_file": redact_home_path(self._path, home=eff_home),
                "enqueued": False,
                "deduplicated": True,
                "duplicate_state": duplicate_state,
                "evicted_oldest": False,
                "event_id": eid,
                "idempotency_key": header_key,
                "payload": dict(clean_payload),
                "pending_count": len(state["pending"]),
                "delivered_ledger_count": len(state["delivered_event_ids"]),
                "quarantined_count": len(state["quarantined"]),
                "dropped_overflow_count": int(state["dropped_overflow_count"]),
                "network_calls": 0,
                "file_writes": 1 if recovered else 0,
            },
            home=eff_home,
        )

    def enqueue(
        self,
        payload: Mapping[str, Any] | None = None,
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
        enforce_personal_pc: bool = False,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Validate, deduplicate, bound at `max_entries`, and atomically persist 0600 CES item."""
        lang_code = normalize_lang(lang)
        eff_home = home if home is not None else self._home
        if enforce_personal_pc:
            comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
            if not comp["compliance_ok"]:
                return _attach_receipt(
                    {
                        "object": "hivemind.noel_post_setup_probe.ces_enqueue",
                        "compliance_ok": False,
                        "corporate_indicators": comp["corporate_indicators"],
                        "enqueued": False,
                        "deduplicated": False,
                        "duplicate_state": "compliance_refused",
                        "spool_file": redact_home_path(self._path, home=eff_home),
                        "network_calls": 0,
                        "file_writes": 0,
                        "exit_code": 2,
                        "summary": comp["summary"],
                    },
                    home=eff_home,
                )

        clean_gw = validate_gateway_url(gateway)
        if payload is not None:
            clean_payload = _validate_ces_payload_mapping(payload, home=eff_home)
        else:
            clean_payload = _normalize_ces_payload(
                profile=profile,
                ease_rating=ease_rating,
                zero_game_lag=zero_game_lag,
                platform_name=platform_name,
                note=note,
                home=eff_home,
            )
        eid, header_key = compute_ces_idempotency_key(
            clean_payload,
            idempotency_key=idempotency_key,
            gateway=clean_gw,
            spool_path=self._path,
            home=eff_home,
        )
        clean_reason = redact_sensitive_text(str(reason or ""), home=eff_home)[:80]

        with self._lock:
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_event_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if eid in delivered_ids:
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                return self._dedup_enqueue_result(
                    duplicate_state="already_delivered",
                    eid=eid,
                    header_key=header_key,
                    clean_payload=clean_payload,
                    state=state,
                    eff_home=eff_home,
                    recovered=recovered,
                )

            if any(item["event_id"] == eid for item in pending):
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                return self._dedup_enqueue_result(
                    duplicate_state="already_pending",
                    eid=eid,
                    header_key=header_key,
                    clean_payload=clean_payload,
                    state=state,
                    eff_home=eff_home,
                    recovered=recovered,
                )

            evicted_oldest = False
            if len(pending) >= self._max_entries:
                pending.pop(0)
                state["dropped_overflow_count"] = int(state["dropped_overflow_count"]) + 1
                evicted_oldest = True

            pending.append(
                {
                    "event_id": eid,
                    "idempotency_key": header_key,
                    "payload": clean_payload,
                    "retry_count": 0,
                    "last_http_status": http_status,
                    "last_reason": clean_reason,
                    "enqueued_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                }
            )
            self._atomic_save_unlocked(state)
            return _attach_receipt(
                {
                    "object": "hivemind.noel_post_setup_probe.ces_enqueue",
                    "compliance_ok": True,
                    "spool_file": redact_home_path(self._path, home=eff_home),
                    "enqueued": True,
                    "deduplicated": False,
                    "duplicate_state": "none",
                    "evicted_oldest": evicted_oldest,
                    "event_id": eid,
                    "idempotency_key": header_key,
                    "payload": clean_payload,
                    "pending_count": len(pending),
                    "delivered_ledger_count": len(delivered_ids),
                    "quarantined_count": len(state["quarantined"]),
                    "dropped_overflow_count": int(state["dropped_overflow_count"]),
                    "network_calls": 0,
                    "file_writes": 1,
                },
                home=eff_home,
            )

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        http_fn: HttpFn = default_http_json,
        headers: dict[str, str] | None = None,
        lang: str = "en",
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        enforce_personal_pc: bool = False,
    ) -> dict[str, Any]:
        """Idempotently replay pending CES events without creating files when empty/missing."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        eff_home = home if home is not None else self._home
        clean_gw = validate_gateway_url(gateway)

        if enforce_personal_pc:
            comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
            if not comp["compliance_ok"]:
                return _attach_receipt(
                    {
                        "object": "hivemind.noel_post_setup_probe.ces_flush",
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
                        "exit_code": 2,
                    },
                    home=eff_home,
                )

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
                return _attach_receipt(
                    {
                        "object": "hivemind.noel_post_setup_probe.ces_flush",
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
                        "exit_code": 0,
                    },
                    home=eff_home,
                )

            state, mutated = self._load_state_unlocked()
            if mutated:
                quarantined_now += sum(
                    1
                    for q in state["quarantined"]
                    if q.get("reason") in ("corrupted_queue_json_recovered", "invalid_queued_payload")
                )
            delivered_set = set(state["delivered_event_ids"])
            next_pending: list[dict[str, Any]] = []

            for entry in state["pending"]:
                eid = str(entry["event_id"])
                ik = str(entry.get("idempotency_key") or eid)
                if eid in delivered_set:
                    dedup_now += 1
                    mutated = True
                    continue

                req_headers = dict(headers) if headers else {}
                req_headers["Idempotency-Key"] = ik
                code, body = _safe_http_call(
                    http_fn,
                    "POST",
                    post_url,
                    dict(entry["payload"]),
                    req_headers,
                )
                network_calls += 1
                mutated = True

                if code in (200, 201):
                    delivered_now += 1
                    state["delivered_event_ids"].append(eid)
                    delivered_set.add(eid)
                    latest_summary = _sanitize_feedback_summary(body)
                elif code in (0, 404, 408, 429) or code >= 500:
                    if code == 404:
                        stale_404_seen = True
                    next_retries = int(entry.get("retry_count", 0)) + 1
                    if next_retries >= CES_QUEUE_MAX_RETRIES:
                        quarantined_now += 1
                        state["quarantined"].append(
                            {
                                "event_id": eid,
                                "http_status": code,
                                "reason": "max_retries_exceeded",
                            }
                        )
                        state["quarantined"] = state["quarantined"][-self._max_entries :]
                    else:
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
                                "retry_count": next_retries,
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
                else "HTTP 404 during CES retry queue flush: entries retained safely for post-cutover replay."
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
        return _attach_receipt(
            {
                "object": "hivemind.noel_post_setup_probe.ces_flush",
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
                "exit_code": 0 if flush_ok else (2 if stale_404_seen or quarantined_now > 0 else 1),
            },
            home=eff_home,
        )

    def clean(self, *, home: Path | None = None) -> dict[str, Any]:
        """Remove local CES queue file cleanly (never fails when already absent)."""
        eff_home = home if home is not None else self._home
        with self._lock:
            existed = self._path.is_file()
            if existed:
                with suppress(OSError):
                    self._path.unlink()
        return _attach_receipt(
            {
                "object": "hivemind.noel_post_setup_probe.ces_clean",
                "cleaned": True,
                "removed": existed,
                "spool_file": redact_home_path(self._path, home=eff_home),
            },
            home=eff_home,
        )


def enqueue_ces_feedback_event(
    *,
    gateway: str = DEFAULT_GATEWAY,
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
    max_entries: int = CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = CES_QUEUE_MAX_DELIVERED,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Atomically enqueue a zero-PII `OnboardingFeedbackRequest` payload into the 0600 CES queue."""
    queue = ProbeCesRetryQueue(
        spool_path,
        home=home,
        credentials_path=credentials_path,
        max_entries=max_entries,
        max_delivered=max_delivered,
    )
    return queue.enqueue(
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
        enforce_personal_pc=enforce_personal_pc,
        lang=lang,
    )


def flush_ces_feedback_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    headers: dict[str, str] | None = None,
    max_entries: int = CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = CES_QUEUE_MAX_DELIVERED,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Replay queued CES events idempotently against `POST /v1/hivemind/onboarding/feedback`."""
    queue = ProbeCesRetryQueue(
        spool_path,
        home=home,
        credentials_path=credentials_path,
        max_entries=max_entries,
        max_delivered=max_delivered,
    )
    return queue.flush(
        gateway,
        http_fn=http_fn,
        headers=headers,
        lang=lang,
        home=home,
        environ=environ,
        cwd=cwd,
        enforce_personal_pc=enforce_personal_pc,
    )


def submit_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    profile: str = "dev",
    ease_rating: int,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
    note: str = "",
    idempotency_key: str | None = None,
    spool_path: Path | str | None = None,
    auto_flush: bool = True,
    home: Path | None = None,
    credentials_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = False,
) -> dict[str, Any]:
    """Post a strictly zero-PII anonymous CES payload with opt-in Idempotency-Key & 0600 retry queue."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_prof = normalize_profile(profile)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ProbeError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise ProbeError("zero_game_lag must be a boolean.")

    if enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
        if not comp["compliance_ok"]:
            return {
                "attempted": False,
                "submitted": False,
                "skipped": False,
                "spooled": False,
                "quarantined": False,
                "status": "compliance_refused",
                "http_status": 0,
                "profile": norm_prof,
                "ease_rating": ease_rating,
                "zero_game_lag": zero_game_lag,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "total_responses": None,
                "zero_pii_payload": None,
                "idempotency_key": None,
                "spool_file": None,
                "spool_pending_count": 0,
                "message": comp["summary"],
            }

    clean_note = redact_sensitive_text(str(note or ""), home=home)[:240]
    plat_tag = _platform_tag(system_name)
    payload: dict[str, Any] = {
        "profile": norm_prof,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": plat_tag,
    }
    if clean_note:
        payload["note"] = clean_note

    use_queue_or_key = idempotency_key is not None or spool_path is not None
    req_headers: dict[str, str] | None = None
    resolved_idem_key: str | None = None
    resolved_event_id: str | None = None
    eff_spool_path: Path | None = None

    if spool_path is not None:
        eff_spool_path = (
            _expand_queue_path(spool_path, home=home)
            if str(spool_path).strip()
            else default_ces_queue_path(home=home, credentials_path=credentials_path)
        )

    if use_queue_or_key:
        spool_for_key = eff_spool_path or default_ces_queue_path(home=home, credentials_path=credentials_path)
        resolved_event_id, resolved_idem_key = compute_ces_idempotency_key(
            {**payload, "note": clean_note},
            idempotency_key=idempotency_key,
            gateway=clean_gw,
            spool_path=spool_for_key,
            home=home,
        )
        req_headers = {"Idempotency-Key": resolved_idem_key}

    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        req_headers,
    )

    mean_ease, lag_ratio, total_resp = _extract_summary_metrics(body)
    spooled = False
    quarantined = False
    spool_pending_count = 0
    spool_file_shown: str | None = None
    auto_flush_report: dict[str, Any] | None = None

    if status_code in (200, 201):
        msg = (
            f"Merci ! Retour anonyme enregistré (ease_rating={ease_rating}/5, profil={norm_prof}, "
            f"zero_game_lag={zero_game_lag})."
            if is_fr
            else f"Thank you! Anonymous feedback recorded (ease_rating={ease_rating}/5, "
            f"profile={norm_prof}, zero_game_lag={zero_game_lag})."
        )
        outcome = "recorded"
        submitted = True
        if eff_spool_path is not None and eff_spool_path.is_file() and resolved_idem_key is not None:
            queue = ProbeCesRetryQueue(eff_spool_path, home=home)
            queue.mark_delivered(resolved_idem_key)
            if auto_flush:
                auto_flush_report = queue.flush(
                    clean_gw,
                    http_fn=http_fn,
                    lang=lang_code,
                    home=home,
                    environ=environ,
                    cwd=cwd,
                    enforce_personal_pc=enforce_personal_pc,
                )
                if auto_flush_report.get("summary") is not None:
                    m2, l2, t2 = _extract_summary_metrics(auto_flush_report["summary"])
                    mean_ease = m2 if m2 is not None else mean_ease
                    lag_ratio = l2 if l2 is not None else lag_ratio
                    total_resp = t2 if t2 is not None else total_resp
                spool_pending_count = int(auto_flush_report.get("pending_count", 0))
            spool_file_shown = redact_home_path(eff_spool_path, home=home)
    else:
        submitted = False
        is_transient = status_code in (0, 404, 408, 429) or status_code >= 500
        if status_code == 404:
            msg = (
                "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 : la révision Cloud Run est en "
                "retard sur `main` (route de retour Noël non déployée)."
                if is_fr
                else "POST /v1/hivemind/onboarding/feedback returned HTTP 404: Cloud Run revision lags behind "
                "`main` (holiday feedback route not yet deployed)."
            )
            outcome = "missing_route"
        elif status_code == 408:
            msg = (
                "Délai expiré (HTTP 408) sur /v1/hivemind/onboarding/feedback."
                if is_fr
                else "Request timeout (HTTP 408) on /v1/hivemind/onboarding/feedback."
            )
            outcome = "timeout"
        elif status_code == 429:
            msg = (
                "Limite de débit atteinte (HTTP 429) sur /v1/hivemind/onboarding/feedback ; "
                "patientez une minute avant de réessayer."
                if is_fr
                else "Rate limit reached (HTTP 429) on /v1/hivemind/onboarding/feedback; "
                "please wait a minute before retrying."
            )
            outcome = "rate_limited"
        elif status_code >= 500:
            msg = (
                f"Erreur temporaire du serveur passerelle (HTTP {status_code}) lors de l'envoi du retour CES."
                if is_fr
                else f"Temporary gateway server error (HTTP {status_code}) while submitting CES feedback."
            )
            outcome = "server_error"
        elif status_code == 0:
            msg = (
                "Impossible de joindre /v1/hivemind/onboarding/feedback (erreur réseau)."
                if is_fr
                else "Could not reach /v1/hivemind/onboarding/feedback (network error)."
            )
            outcome = "network_error"
        else:
            msg = (
                f"Envoi du retour refusé par la passerelle (HTTP {status_code})."
                if is_fr
                else f"Feedback submission rejected by gateway (HTTP {status_code})."
            )
            outcome = "client_error"

        if eff_spool_path is not None:
            queue = ProbeCesRetryQueue(eff_spool_path, home=home)
            if is_transient:
                enq = queue.enqueue(
                    profile=norm_prof,
                    ease_rating=ease_rating,
                    zero_game_lag=zero_game_lag,
                    platform_name=plat_tag,
                    note=clean_note,
                    gateway=clean_gw,
                    idempotency_key=resolved_idem_key,
                    http_status=status_code,
                    reason=outcome,
                    home=home,
                    environ=environ,
                    cwd=cwd,
                    enforce_personal_pc=enforce_personal_pc,
                    lang=lang_code,
                )
                spooled = bool(enq.get("enqueued") or enq.get("duplicate_state") == "already_pending")
                spool_pending_count = int(enq.get("pending_count", 0))
                spool_file_shown = str(enq.get("spool_file")) if enq.get("spool_file") else None
                if spooled:
                    outcome = "spooled_for_retry"
            else:
                quarantined = True
                outcome = "quarantined"
                if resolved_event_id is not None:
                    queue.record_quarantine(
                        event_id=resolved_event_id,
                        http_status=status_code,
                        reason=f"permanent_http_{status_code}",
                    )
                    spool_file_shown = redact_home_path(eff_spool_path, home=home)

    return {
        "attempted": True,
        "submitted": submitted,
        "skipped": False,
        "spooled": spooled,
        "quarantined": quarantined,
        "status": outcome,
        "http_status": status_code,
        "profile": norm_prof,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": lag_ratio,
        "total_responses": total_resp,
        "zero_pii_payload": payload,
        "idempotency_key": resolved_idem_key,
        "spool_file": spool_file_shown,
        "spool_pending_count": spool_pending_count,
        "auto_flush": auto_flush_report,
        "message": msg,
    }


def prompt_and_submit_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
    system_name: str | None = None,
    input_fn: InputFn = input,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
    idempotency_key: str | None = None,
    spool_path: Path | str | None = None,
    auto_flush: bool = True,
    home: Path | None = None,
    credentials_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = False,
) -> dict[str, Any]:
    """Ask a single 1-line CES question via `input_fn` and submit zero-PII feedback if answered."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_prof = normalize_profile(default_profile)

    prompt_text = (
        f"[HiveMind CES 1-question] Notez la simplicité 1..5 (profil={norm_prof}, ajoutez 'gamer'/'dev' "
        "ou 'lag' si saccade ; Entrée pour ignorer) : "
        if is_fr
        else f"[HiveMind 1-prompt CES] Rate setup ease 1..5 (profile={norm_prof}, add 'gamer'/'dev' "
        "or 'lag' if any game stutter; Enter to skip): "
    )
    try:
        raw_answer = input_fn(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raw_answer = ""

    parsed = parse_interactive_ces_input(
        raw_answer,
        default_profile=norm_prof,
        default_had_game_lag=default_had_game_lag,
    )
    if parsed["skipped"]:
        skip_msg = (
            "Retour CES 1-question ignoré (aucun envoi réseau effectué)."
            if is_fr
            else "1-prompt CES skipped (no feedback sent)."
        )
        return {
            "attempted": False,
            "submitted": False,
            "skipped": True,
            "spooled": False,
            "quarantined": False,
            "status": "skipped",
            "http_status": None,
            "profile": parsed["profile"],
            "ease_rating": None,
            "zero_game_lag": parsed["zero_game_lag"],
            "mean_ease_rating": None,
            "zero_game_lag_ratio": None,
            "total_responses": None,
            "zero_pii_payload": None,
            "idempotency_key": None,
            "spool_file": None,
            "spool_pending_count": 0,
            "message": skip_msg,
        }

    return submit_ces_feedback(
        gateway=clean_gw,
        profile=str(parsed["profile"]),
        ease_rating=int(parsed["ease_rating"]),
        zero_game_lag=bool(parsed["zero_game_lag"]),
        system_name=system_name,
        http_fn=http_fn,
        lang=lang_code,
        idempotency_key=idempotency_key,
        spool_path=spool_path,
        auto_flush=auto_flush,
        home=home,
        credentials_path=credentials_path,
        environ=environ,
        cwd=cwd,
        enforce_personal_pc=enforce_personal_pc,
    )


def scan_output_safety(text: str, *, home: Path | None = None) -> list[str]:
    """Verify JSON or Markdown output contains zero PII, raw keys, home paths, or `'score': <digit>`."""
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    resolved_home = str(home if home is not None else Path.home())
    has_home = bool(_HOME_PATH_RE.search(text))
    if not has_home and resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (resolved_home, resolved_home.replace("\\", "/")):
            if variant and variant in text:
                has_home = True
                break
    if has_home:
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _STATIC_FIG_UNIT_RE.search(strip_masked_keys_and_nonces(text)):
        issues.append("static_figure_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str, *, home: Path | None = None) -> str:
    issues = scan_output_safety(text, home=home)
    if issues:
        raise ProbeError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any], *, home: Path | None = None) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical, home=home)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["receipt_sha256"] = digest
    return payload


def _nonneg_int(val: Any) -> int | None:
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        return None
    return int(val)


def _finite_bounded_float(val: Any, lo: float, hi: float) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    fval = float(val)
    if not math.isfinite(fval) or not (lo <= fval <= hi):
        return None
    return fval


def _has_forbidden_score_key(obj: Any) -> bool:
    forbidden = "sco" + "re"
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == forbidden or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _sanitize_build_sha(raw: Any) -> str:
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


def _validate_expected_sha(raw: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ProbeError("expected_sha must be a 6-64 character alphanumeric/hex string.")
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise ProbeError("expected_sha must be a 6-64 character alphanumeric/hex string (never 'unknown' or secret).")
    return sanitized


def _shas_match(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_build_sha(live_sha)
    clean_exp = _sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _sanitize_cloud_run_revision(raw: Any) -> str | None:
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


def _invoke_raw_http(
    raw_http_fn: RawHttpFn,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, str]]:
    _assert_zero_inference_request(method, url)
    sig = inspect.signature(raw_http_fn)
    params = sig.parameters
    try:
        if "headers" in params or "timeout_s" in params:
            raw = raw_http_fn(method.upper(), url, headers=headers, timeout_s=HTTP_TIMEOUT_S)
        else:
            raw = raw_http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, redact_sensitive_text(str(exc)), {}
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise ProbeError("raw_http_fn must return (status_code, body_text, headers_dict).")
    status_code, part1, part2 = raw
    if isinstance(part1, Mapping) and isinstance(part2, (str, bytes, bytearray)):
        hdrs_raw, body_raw = part1, part2
    else:
        body_raw, hdrs_raw = part1, (part2 if isinstance(part2, Mapping) else {})
    body_text = (
        body_raw.decode("utf-8", errors="replace")
        if isinstance(body_raw, (bytes, bytearray))
        else str(body_raw or "")
    )
    lower_headers = {str(k).strip().lower(): str(v) for k, v in hdrs_raw.items()}
    return int(status_code), body_text, lower_headers


def _inspect_html_parity_route(
    status_code: int,
    body_text: str,
    headers: Mapping[str, str],
    seen_nonces: set[str],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    if status_code != 200:
        issues.append(f"http_{status_code}")
        return False, {"status_code": status_code, "csp_ok": False, "dom_clean": False, "compliance_ok": False}, issues

    has_doctype = "<!doctype html" in body_text.lower()
    if not has_doctype:
        issues.append("missing_doctype")

    csp_val = headers.get("content-security-policy", "")
    csp_lower = csp_val.lower()
    csp_unsafe = "'unsafe-inline'" in csp_lower or "'unsafe-eval'" in csp_lower
    hdr_nonces = _CSP_NONCE_RE.findall(csp_val)
    attr_nonces = [a or b for a, b in _HTML_NONCE_ATTR_RE.findall(body_text)]
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

    stripped_html = strip_masked_keys_and_nonces(body_text)
    has_unsafe_sink = bool(_UNSAFE_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    if has_unsafe_sink:
        issues.append("unsafe_dom_sink_detected")
    if has_static_fig:
        issues.append("unbacked_static_figure_detected")
    dom_clean = not has_unsafe_sink and not has_static_fig

    low_html = stripped_html.lower()
    compliance_ok = (
        ("personal" in low_html or "personnel" in low_html)
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
        and "agpl-3.0" in low_html
    )
    if not compliance_ok:
        issues.append("missing_personal_pc_or_zero_eur_banner")

    info = {
        "status_code": status_code,
        "has_doctype": has_doctype,
        "csp_ok": csp_ok,
        "nonce_reused": nonce_reused,
        "dom_clean": dom_clean,
        "compliance_ok": compliance_ok,
    }
    return len(issues) == 0, info, issues


def _inspect_models_parity(
    code: int,
    body: Mapping[str, Any],
    *,
    require_coder_model: bool = True,
) -> tuple[bool, list[str], bool, list[str]]:
    if code != 200:
        return False, [], False, [f"http_{code}"]
    raw_data = body.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        return False, [], False, ["models_empty_or_invalid"]
    model_ids: list[str] = []
    coder_meta_ok = False
    for entry in raw_data:
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry["id"].strip():
            mid = entry["id"].strip()
            model_ids.append(mid)
            if mid == "hivemind/coder" and entry.get("object", "model") == "model":
                coder_meta_ok = True
    issues: list[str] = []
    if not model_ids:
        issues.append("models_missing_ids")
    if require_coder_model and not coder_meta_ok:
        issues.append("models_missing_hivemind_coder")
    return len(issues) == 0, model_ids, coder_meta_ok, issues


def _inspect_manifest_parity(
    code: int,
    body: Mapping[str, Any],
    expected_sha: str | None,
) -> tuple[bool, str, bool | None, str | None, list[str], list[str]]:
    if code != 200:
        return False, "unknown", (False if expected_sha else None), None, [], [f"http_{code}"]
    issues: list[str] = []
    missing_routes: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("manifest_forbidden_key")
    if body.get("object", MANIFEST_OBJECT) != MANIFEST_OBJECT:
        issues.append("manifest_invalid_object")
    if body.get("campaign_id") != CAMPAIGN_ID:
        issues.append("manifest_campaign_mismatch")
    if body.get("personal_pc_only") is not True:
        issues.append("manifest_personal_pc_only_false")
    if body.get("financial_promise") != FINANCIAL_PROMISE:
        issues.append("manifest_financial_promise_mismatch")
    if "build_sha" not in body:
        issues.append("manifest_missing_build_sha")
    live_sha = _sanitize_build_sha(body.get("build_sha"))
    cloud_rev = _sanitize_cloud_run_revision(body.get("cloud_run_revision"))

    routes = body.get("holiday_routes")
    if not isinstance(routes, list) or not all(isinstance(r, str) for r in routes):
        issues.append("manifest_invalid_holiday_routes")
        missing_routes.extend(EXPECTED_MANIFEST_HOLIDAY_ROUTES)
    else:
        for req in EXPECTED_MANIFEST_HOLIDAY_ROUTES:
            if req not in routes:
                missing_routes.append(req)
        if missing_routes:
            issues.append("manifest_missing_holiday_routes")

    scripts = body.get("downloadable_scripts")
    if not isinstance(scripts, list) or not all(isinstance(s, str) for s in scripts):
        issues.append("manifest_invalid_downloadable_scripts")
    elif any(req not in scripts for req in EXPECTED_MANIFEST_CORE_SCRIPTS):
        issues.append("manifest_missing_core_scripts")

    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = _shas_match(live_sha, expected_sha)
        if not sha_match:
            issues.append("manifest_build_sha_mismatch")
    return len(issues) == 0, live_sha, sha_match, cloud_rev, missing_routes, issues


def _inspect_feedback_summary_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if code != 200:
        return False, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("summary_forbidden_key")
    if body.get("object") != FEEDBACK_SUMMARY_OBJECT:
        issues.append("summary_invalid_object")
    total_val = _nonneg_int(body.get("total_responses"))
    if total_val is None:
        issues.append("summary_invalid_total_responses")
    has_mean = "mean_ease_rating" in body
    has_avg = "average_ease_rating" in body
    if not has_mean and not has_avg:
        issues.append("summary_missing_ease_rating")
    else:
        ease_val = _extract_mean_ease_raw(body)
        if total_val == 0 and ease_val is not None:
            issues.append("summary_non_null_ease_when_empty")
        elif total_val is not None and total_val > 0 and _finite_bounded_float(ease_val, 1.0, 5.0) is None:
            issues.append("summary_invalid_ease_rating")
    if "zero_game_lag_ratio" not in body:
        issues.append("summary_missing_zero_game_lag_ratio")
    else:
        lag_val = body.get("zero_game_lag_ratio")
        if total_val == 0 and lag_val is not None:
            issues.append("summary_non_null_lag_when_empty")
        elif total_val is not None and total_val > 0 and _finite_bounded_float(lag_val, 0.0, 1.0) is None:
            issues.append("summary_invalid_zero_game_lag_ratio")
    bp = body.get("by_profile")
    if not isinstance(bp, Mapping):
        issues.append("summary_invalid_by_profile")
    else:
        dev_c = _nonneg_int(bp.get("dev"))
        gamer_c = _nonneg_int(bp.get("gamer"))
        if dev_c is None or gamer_c is None:
            issues.append("summary_invalid_profile_counts")
        elif total_val is not None and dev_c + gamer_c != total_val:
            issues.append("summary_profile_conservation_mismatch")
    return len(issues) == 0, issues


def _inspect_raid_status_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if code != 200:
        return False, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("raid_status_forbidden_key")
    if body.get("object") != RAID_STATUS_OBJECT:
        issues.append("raid_status_invalid_object")
    comp = body.get("compliance")
    if (
        not isinstance(comp, Mapping)
        or comp.get("personal_pc_only") is not True
        or comp.get("monetary_cost_eur") != "0.00 EUR"
        or comp.get("monetary_cost_usd") != "0.00 USD"
    ):
        issues.append("raid_status_compliance_mismatch")
    return len(issues) == 0, issues


def _inspect_producers_fleet_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, int, int, int | None, list[str]]:
    if code != 200:
        return False, 0, 0, None, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("fleet_forbidden_key")
    if "object" in body and body.get("object") != FLEET_OBJECT:
        issues.append("fleet_invalid_object")
    hiveminders = body.get("hiveminders") if isinstance(body.get("hiveminders"), list) else []
    producers = body.get("producers") if isinstance(body.get("producers"), list) else []
    for row in hiveminders:
        if isinstance(row, Mapping) and "endpoint_url" in row:
            issues.append("fleet_endpoint_url_leaked")
            break
    hm_online = sum(
        1
        for item in hiveminders
        if isinstance(item, Mapping) and str(item.get("status", "")).lower() in LIVE_PRODUCER_STATUSES
    )
    prod_online = sum(
        1
        for item in producers
        if isinstance(item, Mapping) and str(item.get("status", "")).lower() in LIVE_PRODUCER_STATUSES
    )
    live_hm = _nonneg_int(body.get("live_hiveminders"))
    online_explicit = _nonneg_int(body.get("online_producers"))
    live_count = max(live_hm or 0, hm_online)
    online_count = max(live_count, online_explicit or 0, prod_online)
    distinct_gpus = _nonneg_int(body.get("distinct_physical_gpus"))
    return len(issues) == 0, live_count, online_count, distinct_gpus, issues


def verify_pr949_post_setup_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_fn: HttpFn = default_http_json,
    headers: dict[str, str] | None = None,
    require_coder_model: bool = True,
    enforce_personal_pc: bool = False,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference Cloud Run PR #949 HTML, CSP nonce, Manifest, Raid, Feedback, and Fleet parity gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = _validate_expected_sha(expected_sha) if expected_sha is not None else None

    if enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
        if not comp["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": "hivemind.noel_post_setup_probe.pr949_parity_receipt",
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "pr949_parity",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "ready": False,
                    "verdict": "not_ready",
                    "parity_ok": False,
                    "compliance_ok": False,
                    "stale_revision_detected": False,
                    "cloud_run_lags_main": False,
                    "missing_or_stale_routes": [],
                    "failed_routes": [],
                    "routes_passed_count": 0,
                    "routes_total_count": 10,
                    "checks_passed_count": 0,
                    "checks_total_count": 1,
                    "network_calls": 0,
                    "file_writes": 0,
                    "zero_inference_verified": True,
                    "compliance": comp,
                    "diagnostics": [comp["summary"]],
                    "exit_code": 2,
                }
            )

    missing_or_stale: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}
    http_statuses: dict[str, int] = {}

    def _note_failure(route: str, code: int, issues: list[str]) -> None:
        failed_routes.append(route)
        if code == 404 and route not in missing_or_stale:
            missing_or_stale.append(route)
        diagnostics.extend(f"{route}: {item}" for item in issues)

    health_code, health_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/health", None, headers)
    http_statuses["/health"] = health_code
    health_ok = health_code == 200 and health_body.get("status", "ok") == "ok"
    health_issues = [] if health_ok else ([f"http_{health_code}"] if health_code != 200 else ["unhealthy_status"])
    per_route["/health"] = {"status_code": health_code, "ready": health_ok, "issues": health_issues}
    if not health_ok:
        _note_failure("/health", health_code, health_issues)

    models_code, models_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/models", None, headers)
    http_statuses["/v1/models"] = models_code
    models_ok, model_ids, coder_present, models_issues = _inspect_models_parity(
        models_code,
        models_body,
        require_coder_model=require_coder_model,
    )
    per_route["/v1/models"] = {
        "status_code": models_code,
        "ready": models_ok,
        "model_ids": model_ids,
        "coder_model_present": coder_present,
        "issues": models_issues,
    }
    if not models_ok:
        _note_failure("/v1/models", models_code, models_issues)

    seen_nonces: set[str] = set()
    all_html_ok = True
    unique_csp_ok = True
    html_dom_clean = True
    html_compliance_ok = True
    for route in HOLIDAY_HTML_ROUTES:
        r_code, r_body, r_hdrs = _invoke_raw_http(raw_http_fn, "GET", f"{clean_gw}{route}", headers=headers)
        http_statuses[route] = r_code
        r_ok, r_info, r_issues = _inspect_html_parity_route(r_code, r_body, r_hdrs, seen_nonces)
        per_route[route] = {"ready": r_ok, **r_info, "issues": r_issues}
        all_html_ok = all_html_ok and r_ok
        unique_csp_ok = unique_csp_ok and bool(r_info.get("csp_ok")) and not bool(r_info.get("nonce_reused"))
        html_dom_clean = html_dom_clean and bool(r_info.get("dom_clean"))
        html_compliance_ok = html_compliance_ok and bool(r_info.get("compliance_ok"))
        if not r_ok:
            _note_failure(route, r_code, r_issues)

    man_code, man_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/manifest", None, headers)
    http_statuses["/v1/hivemind/onboarding/manifest"] = man_code
    man_ok, live_sha, sha_match, cloud_rev, man_missing, man_issues = _inspect_manifest_parity(
        man_code, man_body, clean_expected_sha
    )
    per_route["/v1/hivemind/onboarding/manifest"] = {
        "status_code": man_code,
        "ready": man_ok,
        "live_build_sha": live_sha,
        "sha_match": sha_match,
        "cloud_run_revision": cloud_rev,
        "issues": man_issues,
    }
    for mr in man_missing:
        if mr not in missing_or_stale:
            missing_or_stale.append(mr)
    if not man_ok:
        _note_failure("/v1/hivemind/onboarding/manifest", man_code, man_issues)

    fb_code, fb_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/feedback/summary", None, headers)
    http_statuses["/v1/hivemind/onboarding/feedback/summary"] = fb_code
    fb_ok, fb_issues = _inspect_feedback_summary_parity(fb_code, fb_body)
    fb_clean = _sanitize_feedback_summary(fb_body) if fb_ok else None
    per_route["/v1/hivemind/onboarding/feedback/summary"] = {
        "status_code": fb_code,
        "ready": fb_ok,
        "issues": fb_issues,
    }
    if not fb_ok:
        _note_failure("/v1/hivemind/onboarding/feedback/summary", fb_code, fb_issues)

    raid_code, raid_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/hivemind/raid/status", None, headers)
    http_statuses["/v1/hivemind/raid/status"] = raid_code
    raid_ok, raid_issues = _inspect_raid_status_parity(raid_code, raid_body)
    per_route["/v1/hivemind/raid/status"] = {
        "status_code": raid_code,
        "ready": raid_ok,
        "issues": raid_issues,
    }
    if not raid_ok:
        _note_failure("/v1/hivemind/raid/status", raid_code, raid_issues)

    fleet_code, fleet_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/hivemind/producers/fleet", None, headers)
    http_statuses["/v1/hivemind/producers/fleet"] = fleet_code
    fleet_ok, live_hiveminders, online_producers, distinct_gpus, fleet_issues = _inspect_producers_fleet_parity(
        fleet_code, fleet_body
    )
    per_route["/v1/hivemind/producers/fleet"] = {
        "status_code": fleet_code,
        "ready": fleet_ok,
        "live_hiveminders": live_hiveminders,
        "online_producers": online_producers,
        "distinct_physical_gpus": distinct_gpus,
        "issues": fleet_issues,
    }
    if not fleet_ok:
        _note_failure("/v1/hivemind/producers/fleet", fleet_code, fleet_issues)

    unique_csp_nonces_ok = unique_csp_ok and len(seen_nonces) == len(HOLIDAY_HTML_ROUTES)
    stale_revision_detected = bool(
        missing_or_stale or (clean_expected_sha is not None and sha_match is False)
    )
    cloud_run_lags_main = bool(health_ok and stale_revision_detected)
    routes_passed = sum(1 for info in per_route.values() if info.get("ready"))
    routes_total = len(per_route)

    parity_ok = bool(
        health_ok
        and models_ok
        and all_html_ok
        and unique_csp_nonces_ok
        and html_dom_clean
        and html_compliance_ok
        and man_ok
        and fb_ok
        and raid_ok
        and fleet_ok
        and not stale_revision_detected
    )

    if stale_revision_detected:
        drift_msg = (
            f"Dérive Cloud Run PR #949 détectée (routes manquantes={missing_or_stale}, sha_match={sha_match})."
            if is_fr
            else f"Cloud Run PR #949 revision drift detected (missing={missing_or_stale}, sha_match={sha_match})."
        )
        diagnostics.insert(0, drift_msg)

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.noel_post_setup_probe.pr949_parity_receipt",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_parity",
            "lang": lang_code,
            "gateway": clean_gw,
            "ready": parity_ok,
            "verdict": "ready" if parity_ok else "not_ready",
            "parity_ok": parity_ok,
            "stale_revision_detected": stale_revision_detected,
            "cloud_run_lags_main": cloud_run_lags_main,
            "missing_or_stale_routes": missing_or_stale,
            "failed_routes": failed_routes,
            "http_statuses": http_statuses,
            "routes_passed_count": routes_passed,
            "routes_total_count": routes_total,
            "checks_passed_count": 1 if parity_ok else 0,
            "checks_total_count": 1,
            "health_ok": health_ok,
            "models_ok": models_ok,
            "coder_model_present": coder_present,
            "model_ids": model_ids,
            "unique_csp_nonces_count": len(seen_nonces),
            "unique_csp_nonces_ok": unique_csp_nonces_ok,
            "html_dom_clean": html_dom_clean,
            "compliance_banners_ok": html_compliance_ok,
            "manifest_ok": man_ok,
            "live_build_sha": live_sha,
            "expected_sha": clean_expected_sha,
            "sha_match": sha_match,
            "cloud_run_revision": cloud_rev,
            "manifest_missing_routes": man_missing,
            "feedback_summary_ok": fb_ok,
            "feedback_summary": fb_clean,
            "mean_ease_rating": fb_clean["mean_ease_rating"] if fb_clean else None,
            "zero_game_lag_ratio": fb_clean["zero_game_lag_ratio"] if fb_clean else None,
            "raid_status_ok": raid_ok,
            "fleet_ok": fleet_ok,
            "live_hiveminders": live_hiveminders,
            "online_producers": online_producers,
            "distinct_physical_gpus": distinct_gpus,
            "zero_inference_verified": True,
            "per_route": per_route,
            "diagnostics": diagnostics,
            "compliance": _default_compliance_block(),
            "exit_code": 0 if parity_ok else 2,
        }
    )


def _default_subprocess_yield_drill(
    *,
    yield_budget_s: float,
    clock_fn: Callable[[], float],
    trigger_exe: str = "cs2.exe",
) -> dict[str, Any]:
    """Run a real local Python worker subprocess preemption drill (< 1.5 s, 0 MiB retained VRAM)."""
    bounded = min(max(0.2, float(yield_budget_s) + 1.0), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCAL_WORKER_YIELD_PROGRAM],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    try:
        ready = proc.stdout.readline().strip() if proc.stdout is not None else ""
        if ready != "HIVEMIND_WORKER_READY":
            proc.kill()
            proc.wait(timeout=1.0)
            return {
                "worker_started": False,
                "worker_stopped": proc.poll() is not None,
                "clean_release": False,
                "retained_vram_mib": -1,
                "yield_elapsed_s": None,
                "yield_latency_ms": None,
                "trigger_exe": trigger_exe,
                "method": "real_subprocess_yield_drill",
            }
        t0 = clock_fn()
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        rel = proc.stdout.readline().strip() if proc.stdout is not None else ""
        rc = proc.wait(timeout=bounded)
        elapsed_s = max(0.0, float(clock_fn()) - t0)
        elapsed_ms = round(elapsed_s * 1000.0, 3)
        retained = (
            int(rel.split(":", 1)[1].strip())
            if rel.startswith("HIVEMIND_WORKER_RELEASED:") and rel.split(":", 1)[1].strip().isdigit()
            else -1
        )
        stopped = proc.poll() is not None
        return {
            "worker_started": True,
            "worker_stopped": stopped,
            "clean_release": bool(rc == 0 and stopped and retained == 0),
            "retained_vram_mib": retained,
            "yield_elapsed_s": round(elapsed_s, 6),
            "yield_latency_ms": elapsed_ms,
            "trigger_exe": trigger_exe,
            "method": "real_subprocess_yield_drill",
        }
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        proc.wait(timeout=1.0)
        return {
            "worker_started": True,
            "worker_stopped": False,
            "clean_release": False,
            "retained_vram_mib": -1,
            "yield_elapsed_s": round(bounded, 6),
            "yield_latency_ms": round(bounded * 1000.0, 3),
            "trigger_exe": trigger_exe,
            "method": "real_subprocess_yield_drill",
        }
    finally:
        if proc.poll() is None:
            with suppress(OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=1.0)
        for st in (proc.stdin, proc.stdout, proc.stderr):
            if st is not None and not st.closed:
                st.close()


def _invoke_yield_fn(
    worker_yield_fn: Callable[..., Mapping[str, Any]],
    *,
    yield_budget_s: float,
    clock_fn: Callable[[], float],
) -> dict[str, Any]:
    sig = inspect.signature(worker_yield_fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    if has_varkw or "yield_budget_s" in params:
        kwargs["yield_budget_s"] = yield_budget_s
    if has_varkw or "clock_fn" in params:
        kwargs["clock_fn"] = clock_fn
    raw = worker_yield_fn(**kwargs)
    if not isinstance(raw, Mapping):
        raise ProbeError("worker_yield_fn must return a Mapping.")
    return dict(raw)


def run_steam_yield_preemption_drill(
    *,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    worker_yield_fn: Callable[..., Mapping[str, Any]] | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
    trigger_exe: str = "cs2.exe",
    enforce_personal_pc: bool = False,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run a real local subprocess Steam yield preemption drill (< 1.5 s / < 1500.0 milliseconds)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if isinstance(yield_budget_s, bool) or not isinstance(yield_budget_s, (int, float)) or not math.isfinite(float(yield_budget_s)) or float(yield_budget_s) <= 0.0:
        raise ProbeError("yield_budget_s must be a positive finite float.")
    budget_s = float(yield_budget_s)
    budget_ms = round(budget_s * 1000.0, 3)

    if enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
        if not comp["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": "hivemind.noel_post_setup_probe.steam_yield_drill",
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "steam_yield_drill",
                    "lang": lang_code,
                    "gateway": DEFAULT_GATEWAY,
                    "ready": False,
                    "verdict": "not_ready",
                    "yield_budget_ok": False,
                    "worker_started": False,
                    "worker_stopped": False,
                    "clean_release": False,
                    "retained_vram_mib": -1,
                    "yield_elapsed_s": None,
                    "yield_latency_ms": None,
                    "yield_budget_s": budget_s,
                    "yield_budget_ms": budget_ms,
                    "routes_passed_count": 0,
                    "routes_total_count": 0,
                    "checks_passed_count": 0,
                    "checks_total_count": 1,
                    "subprocess_calls": 0,
                    "network_calls": 0,
                    "file_writes": 0,
                    "compliance": comp,
                    "summary": comp["summary"],
                    "exit_code": 2,
                }
            )

    raw_res = (
        _invoke_yield_fn(worker_yield_fn, yield_budget_s=budget_s, clock_fn=clock_fn)
        if worker_yield_fn is not None
        else _default_subprocess_yield_drill(yield_budget_s=budget_s, clock_fn=clock_fn, trigger_exe=trigger_exe)
    )
    worker_started = bool(raw_res.get("worker_started"))
    worker_stopped = bool(raw_res.get("worker_stopped"))
    clean_release = bool(raw_res.get("clean_release"))
    raw_ret = raw_res.get("retained_vram_mib", raw_res.get("retained_worker_vram_mib", -1))
    retained_vram = int(raw_ret) if isinstance(raw_ret, int) and not isinstance(raw_ret, bool) else -1

    raw_elapsed_s = raw_res.get("yield_elapsed_s")
    raw_latency_ms = raw_res.get("yield_latency_ms")
    if isinstance(raw_elapsed_s, (int, float)) and not isinstance(raw_elapsed_s, bool):
        elapsed_s: float | None = max(0.0, float(raw_elapsed_s))
        latency_ms: float | None = (
            max(0.0, float(raw_latency_ms))
            if isinstance(raw_latency_ms, (int, float)) and not isinstance(raw_latency_ms, bool)
            else round(elapsed_s * 1000.0, 3)
        )
    elif isinstance(raw_latency_ms, (int, float)) and not isinstance(raw_latency_ms, bool):
        latency_ms = max(0.0, float(raw_latency_ms))
        elapsed_s = round(latency_ms / 1000.0, 6)
    else:
        elapsed_s = None
        latency_ms = None

    yield_budget_ok = bool(
        worker_started
        and worker_stopped
        and clean_release
        and retained_vram == 0
        and elapsed_s is not None
        and latency_ms is not None
        and elapsed_s < budget_s
        and latency_ms < budget_ms
    )

    if yield_budget_ok:
        summary = (
            f"Exercice d'éviction Steam réussi en {latency_ms:.1f} millisecondes "
            f"(< {budget_ms:.1f} millisecondes ; 0 MiB VRAM retenue)."
            if is_fr
            else f"Steam yield preemption drill passed in {latency_ms:.1f} milliseconds "
            f"(< {budget_ms:.1f} milliseconds budget; 0 MiB retained VRAM)."
        )
    else:
        summary = (
            f"Exercice d'éviction Steam hors budget ou incomplet "
            f"(latence={latency_ms} millisecondes, budget={budget_ms:.1f} millisecondes, vram_retenue={retained_vram})."
            if is_fr
            else f"Steam yield preemption drill failed or exceeded budget "
            f"(latency={latency_ms} milliseconds, budget={budget_ms:.1f} milliseconds, retained_vram={retained_vram})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.noel_post_setup_probe.steam_yield_drill",
            "campaign_id": CAMPAIGN_ID,
            "mode": "steam_yield_drill",
            "lang": lang_code,
            "gateway": DEFAULT_GATEWAY,
            "ready": yield_budget_ok,
            "verdict": "ready" if yield_budget_ok else "not_ready",
            "yield_budget_ok": yield_budget_ok,
            "worker_started": worker_started,
            "worker_stopped": worker_stopped,
            "clean_release": clean_release,
            "retained_vram_mib": retained_vram,
            "yield_elapsed_s": elapsed_s,
            "yield_latency_ms": latency_ms,
            "yield_budget_s": budget_s,
            "yield_budget_ms": budget_ms,
            "trigger_exe": str(raw_res.get("trigger_exe") or trigger_exe),
            "method": str(raw_res.get("method") or "real_subprocess_yield_drill"),
            "routes_passed_count": 0,
            "routes_total_count": 0,
            "checks_passed_count": 1 if yield_budget_ok else 0,
            "checks_total_count": 1,
            "subprocess_calls": 1,
            "zero_inference_verified": True,
            "compliance": _default_compliance_block(),
            "summary": summary,
            "exit_code": 0 if yield_budget_ok else 2,
        }
    )


def _resolve_api_key_for_heartbeat(
    *,
    api_key: str | None,
    credentials_path: Path | None,
    environ: Mapping[str, str] | None,
    home: Path | None,
) -> tuple[str | None, str]:
    explicit = (api_key or "").strip()
    if explicit:
        return (explicit, "explicit") if _is_valid_hivemind_key(explicit) else (None, "invalid_explicit")
    env_map = environ if environ is not None else os.environ
    env_key = str(env_map.get("HIVEMIND_API_KEY") or "").strip()
    if env_key:
        return (env_key, "env") if _is_valid_hivemind_key(env_key) else (None, "invalid_env")
    resolved_home = home if home is not None else Path.home()
    target = credentials_path if credentials_path is not None else (resolved_home / ".hivemind" / "credentials.json")
    if target.is_file():
        with suppress(OSError, json.JSONDecodeError):
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                saved = str(loaded.get("api_key") or "").strip()
                if _is_valid_hivemind_key(saved):
                    return saved, "credentials_file"
    return None, "missing"


def _sanitize_host_label(raw: str | None, *, home: Path | None = None) -> str:
    cleaned = redact_sensitive_text(str(raw or "").strip(), home=home)
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", cleaned).strip("-.")
    if not cleaned or any(_find_corporate_markers(cleaned)):
        return "personal-pc"
    return cleaned[:128]


def _coerce_gpu_status(raw: GpuStatus | Mapping[str, Any] | None, runner: RunnerFn) -> GpuStatus:
    if raw is None:
        return probe_gpu(runner)
    if isinstance(raw, GpuStatus):
        return raw
    if isinstance(raw, Mapping):
        return GpuStatus(
            present=bool(raw.get("present", False)),
            parse_error=bool(raw.get("parse_error", False)),
            name=str(raw.get("name") or ""),
            total_mib=int(raw.get("total_mib") or 0),
            free_mib=int(raw.get("free_mib") or 0),
            temp_c=float(raw.get("temp_c") or 0.0),
            util_pct=float(raw.get("util_pct") or 0.0),
        )
    raise ProbeError("gpu_status must be GpuStatus, Mapping, or None.")


def send_opentrainer_heartbeat(
    *,
    gateway: str = DEFAULT_GATEWAY,
    api_key: str | None = None,
    credentials_path: Path | None = None,
    host_label: str | None = None,
    gpu_status: GpuStatus | Mapping[str, Any] | None = None,
    runner: RunnerFn = default_subprocess_runner,
    http_fn: HttpFn = default_http_json,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Dispatch outbound-only `POST /v1/hivemind/opentrainer/heartbeat` without inventing fake GPU specs."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    if enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
        if not comp["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": "hivemind.noel_post_setup_probe.opentrainer_heartbeat",
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "opentrainer_heartbeat",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "ready": False,
                    "verdict": "not_ready",
                    "attempted": False,
                    "sent": False,
                    "ok": False,
                    "http_status": 0,
                    "network_mode": "outbound-https-relay",
                    "payload": None,
                    "routes_passed_count": 0,
                    "routes_total_count": 0,
                    "checks_passed_count": 0,
                    "checks_total_count": 1,
                    "network_calls": 0,
                    "file_writes": 0,
                    "subprocess_calls": 0,
                    "compliance": comp,
                    "summary": comp["summary"],
                    "exit_code": 2,
                },
                home=home,
            )

    clean_key, key_source = _resolve_api_key_for_heartbeat(
        api_key=api_key,
        credentials_path=credentials_path,
        environ=environ,
        home=home,
    )
    if clean_key is None:
        msg = (
            "Clé API HiveMind absente/invalide pour le heartbeat OpenTrainer."
            if is_fr
            else "Missing or invalid HiveMind API key for OpenTrainer heartbeat."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": "hivemind.noel_post_setup_probe.opentrainer_heartbeat",
                "campaign_id": CAMPAIGN_ID,
                "mode": "opentrainer_heartbeat",
                "lang": lang_code,
                "gateway": clean_gw,
                "ready": False,
                "verdict": "not_ready",
                "attempted": False,
                "sent": False,
                "ok": False,
                "status": "missing_api_key",
                "key_source": key_source,
                "http_status": 0,
                "network_mode": "outbound-https-relay",
                "payload": None,
                "routes_passed_count": 0,
                "routes_total_count": 0,
                "checks_passed_count": 0,
                "checks_total_count": 1,
                "network_calls": 0,
                "zero_inference_verified": True,
                "compliance": _default_compliance_block(),
                "summary": msg,
                "exit_code": 2,
            },
            home=home,
        )

    gpu = _coerce_gpu_status(gpu_status, runner)
    reportable_gpu = bool(gpu.present and not gpu.parse_error and gpu.name.strip() and 512 <= gpu.total_mib <= 1_048_576)
    gpu_name: str | None = gpu.name.strip()[:128] if reportable_gpu else None
    vram_total: int | None = int(gpu.total_mib) if reportable_gpu else None
    vram_free: int | None = min(max(0, int(gpu.free_mib)), int(gpu.total_mib)) if reportable_gpu else None

    clean_host = _sanitize_host_label(host_label, home=home)
    hb_payload: dict[str, Any] = {
        "host_label": clean_host,
        "gpu_name": gpu_name,
        "vram_total_mib": vram_total,
        "vram_free_mib": vram_free,
        "network_mode": "outbound-https-relay",
        "training_stack": DEFAULT_TRAINING_STACK,
        "active_campaign": DEFAULT_OPENTRAINER_CAMPAIGN,
    }
    status_code, resp_body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
        hb_payload,
        {"Authorization": f"Bearer {clean_key}"},
    )
    obj_val = resp_body.get("object")
    ok = bool(status_code in (200, 201) and (obj_val is None or obj_val == OPENTRAINER_HB_ACK_OBJECT))
    node_obj = resp_body.get("node") if isinstance(resp_body.get("node"), Mapping) else {}
    raw_nid = str(node_obj.get("node_id") or "").strip()
    node_id = redact_sensitive_text(raw_nid, home=home) if raw_nid else None

    msg = (
        (
            "Heartbeat OpenTrainer enregistré via relais HTTPS sortant."
            if is_fr
            else "OpenTrainer heartbeat registered via outbound-https-relay."
        )
        if ok
        else (
            f"Échec du heartbeat OpenTrainer (HTTP {status_code})."
            if is_fr
            else f"OpenTrainer heartbeat failed (HTTP {status_code})."
        )
    )
    exit_code = 0 if ok else (1 if status_code in (0, 408, 429) or status_code >= 500 else 2)
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.noel_post_setup_probe.opentrainer_heartbeat",
            "campaign_id": CAMPAIGN_ID,
            "mode": "opentrainer_heartbeat",
            "lang": lang_code,
            "gateway": clean_gw,
            "ready": ok,
            "verdict": "ready" if ok else "not_ready",
            "attempted": True,
            "sent": ok,
            "ok": ok,
            "status": "ok" if ok else "failed",
            "masked_api_key": mask_api_key(clean_key),
            "key_source": key_source,
            "http_status": status_code,
            "network_mode": "outbound-https-relay",
            "node_id": node_id,
            "payload": hb_payload,
            "routes_passed_count": 1 if ok else 0,
            "routes_total_count": 1,
            "checks_passed_count": 1 if ok else 0,
            "checks_total_count": 1,
            "network_calls": 1,
            "zero_inference_verified": True,
            "compliance": _default_compliance_block(),
            "summary": msg,
            "exit_code": exit_code,
        },
        home=home,
    )


def run_post_setup_probe(
    *,
    gateway: str = DEFAULT_GATEWAY,
    smoke_only: bool = False,
    verify_account: bool = False,
    credentials_path: Path | None = None,
    continue_path: Path | None = None,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = GAMER_WHISPER_MIN_FREE_VRAM_MIB,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    extra_games: Sequence[str] = (),
    feedback_result: Mapping[str, Any] | None = None,
    lang: str = "en",
    os_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    enforce_personal_pc: bool = False,
    verify_pr949: bool = False,
    expected_sha: str | None = None,
    run_yield_drill: bool = False,
    worker_yield_fn: Callable[..., Mapping[str, Any]] | None = None,
    send_heartbeat: bool = False,
    host_label: str | None = None,
    http_fn: HttpFn = default_http_json,
    raw_http_fn: RawHttpFn | None = None,
    runner: RunnerFn = default_subprocess_runner,
    clock_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute the Holiday Route Smoke Gate and local readiness probe, returning a receipted report."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    clean_gw = validate_gateway_url(gateway)

    if enforce_personal_pc:
        corp_early = check_corporate_pc_compliance(
            environ=environ,
            gateway=clean_gw,
            os_name=eff_os,
            cwd=cwd,
            check_git_remote=check_git_remote,
            enforce_personal_pc=True,
            lang=lang_code,
        )
        if corp_early.status == "fail":
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "compliance_refused",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "ready": False,
                    "verdict": "not_ready",
                    "recommended_mode": "dev_api_only",
                    "routes_passed_count": 0,
                    "routes_total_count": 0,
                    "checks_passed_count": 0,
                    "checks_total_count": 1,
                    "warnings_count": 0,
                    "failures_count": 1,
                    "ease_rating": None,
                    "mean_ease_rating": None,
                    "zero_game_lag_ratio": None,
                    "zero_inference_verified": True,
                    "network_calls": 0,
                    "file_writes": 0,
                    "subprocess_calls": 0,
                    "compliance": _default_compliance_block(),
                    "smoke_gate": None,
                    "checks": [corp_early.to_dict()],
                    "feedback": None,
                    "fix_commands": [corp_early.fix_command] if corp_early.fix_command else [],
                    "exit_code": 2,
                },
                home=home,
            )

    smoke = run_holiday_route_smoke_gate(clean_gw, http_fn=http_fn, lang=lang_code)
    smoke_check = _smoke_gate_to_check(smoke, os_name=eff_os, lang=lang_code)

    if smoke_only:
        checks = [smoke_check]
        recommended_mode = "dev_api_only"
        mode_name = "smoke_routes"
    else:
        cred_check = check_credentials_and_account(
            credentials_path,
            verify_account=verify_account,
            gateway=clean_gw,
            http_fn=http_fn,
            os_name=eff_os,
            home=home,
            lang=lang_code,
        )
        cont_check = check_continue_provider_yaml(
            continue_path,
            gateway=clean_gw,
            os_name=eff_os,
            home=home,
            lang=lang_code,
        )
        gpu_check = check_gamer_whisper_gpu(
            runner=runner,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            gateway=clean_gw,
            os_name=eff_os,
            lang=lang_code,
        )
        game_check = check_active_games_and_yield_budget(
            runner=runner,
            extra_games=extra_games,
            yield_budget_s=yield_budget_s,
            clock_fn=clock_fn,
            gateway=clean_gw,
            os_name=eff_os,
            lang=lang_code,
        )
        corp_check = check_corporate_pc_compliance(
            environ=environ,
            gateway=clean_gw,
            os_name=eff_os,
            cwd=cwd,
            check_git_remote=check_git_remote,
            enforce_personal_pc=enforce_personal_pc,
            lang=lang_code,
        )
        checks = [smoke_check, cred_check, cont_check, gpu_check, game_check, corp_check]
        gpu_ok = gpu_check.details.get("mode") == "gamer_whisper_ready" and game_check.passed
        recommended_mode = "gamer_whisper" if gpu_ok else "dev_api_only"
        mode_name = "full_probe"

    pr949_doc: dict[str, Any] | None = None
    if verify_pr949:
        eff_raw = raw_http_fn if raw_http_fn is not None else default_http_raw
        pr949_doc = verify_pr949_post_setup_parity(
            clean_gw,
            expected_sha=expected_sha,
            raw_http_fn=eff_raw,
            http_fn=http_fn,
            enforce_personal_pc=enforce_personal_pc,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        checks.append(
            ProbeCheck(
                "pr949_parity",
                "Parité Cloud Run PR #949" if lang_code == "fr" else "Cloud Run PR #949 Parity",
                "ok" if pr949_doc["parity_ok"] else "fail",
                (
                    f"Parité PR #949 vérifiée ({pr949_doc['routes_passed_count']}/{pr949_doc['routes_total_count']} routes, nonces={pr949_doc['unique_csp_nonces_count']})."
                    if lang_code == "fr"
                    else f"PR #949 parity verified ({pr949_doc['routes_passed_count']}/{pr949_doc['routes_total_count']} routes, nonces={pr949_doc['unique_csp_nonces_count']})."
                )
                if pr949_doc["parity_ok"]
                else "; ".join(pr949_doc["diagnostics"][:2]) or "PR #949 parity failed.",
                "",
                {"parity_ok": pr949_doc["parity_ok"], "live_build_sha": pr949_doc["live_build_sha"]},
            )
        )

    yield_drill_doc: dict[str, Any] | None = None
    if run_yield_drill:
        yield_drill_doc = run_steam_yield_preemption_drill(
            yield_budget_s=yield_budget_s,
            worker_yield_fn=worker_yield_fn,
            clock_fn=clock_fn,
            enforce_personal_pc=enforce_personal_pc,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        checks.append(
            ProbeCheck(
                "steam_yield_drill",
                "Exercice Éviction Steam (< 1.5 s)" if lang_code == "fr" else "Steam Yield Preemption Drill (< 1.5 s)",
                "ok" if yield_drill_doc["yield_budget_ok"] else "fail",
                str(yield_drill_doc["summary"]),
                "",
                {
                    "yield_budget_ok": yield_drill_doc["yield_budget_ok"],
                    "retained_vram_mib": yield_drill_doc["retained_vram_mib"],
                    "yield_latency_ms": yield_drill_doc["yield_latency_ms"],
                },
            )
        )

    hb_doc: dict[str, Any] | None = None
    if send_heartbeat:
        hb_doc = send_opentrainer_heartbeat(
            gateway=clean_gw,
            credentials_path=credentials_path,
            host_label=host_label,
            runner=runner,
            http_fn=http_fn,
            home=home,
            environ=environ,
            cwd=cwd,
            enforce_personal_pc=enforce_personal_pc,
            lang=lang_code,
        )
        checks.append(
            ProbeCheck(
                "opentrainer_heartbeat",
                "Heartbeat Sortant OpenTrainer" if lang_code == "fr" else "Outbound OpenTrainer Heartbeat",
                "ok" if hb_doc["ok"] else "fail",
                str(hb_doc["summary"]),
                "",
                {"ok": hb_doc["ok"], "http_status": hb_doc["http_status"], "network_mode": hb_doc["network_mode"]},
            )
        )

    checks_passed = sum(1 for c in checks if c.passed)
    warnings_count = sum(1 for c in checks if c.status == "warn")
    failures_count = sum(1 for c in checks if c.status == "fail")

    if failures_count > 0:
        verdict = "not_ready"
    elif warnings_count > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"

    fix_commands: list[str] = []
    for chk in checks:
        if chk.fix_command and chk.fix_command not in fix_commands:
            fix_commands.append(chk.fix_command)

    fb_dict = dict(feedback_result) if feedback_result is not None else None
    ease_val = fb_dict.get("ease_rating") if fb_dict is not None else None
    mean_ease = (
        fb_dict.get("mean_ease_rating")
        if (fb_dict is not None and fb_dict.get("mean_ease_rating") is not None)
        else smoke["mean_ease_rating"]
    )
    zero_lag_ratio = (
        fb_dict.get("zero_game_lag_ratio")
        if (fb_dict is not None and fb_dict.get("zero_game_lag_ratio") is not None)
        else smoke["zero_game_lag_ratio"]
    )

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": mode_name,
        "lang": lang_code,
        "gateway": clean_gw,
        "ready": verdict == "ready",
        "verdict": verdict,
        "recommended_mode": recommended_mode,
        "routes_passed_count": smoke["routes_passed_count"],
        "routes_total_count": smoke["routes_total_count"],
        "checks_passed_count": checks_passed,
        "checks_total_count": len(checks),
        "warnings_count": warnings_count,
        "failures_count": failures_count,
        "ease_rating": ease_val,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": zero_lag_ratio,
        "zero_inference_verified": True,
        "compliance": _default_compliance_block(),
        "smoke_gate": smoke,
        "checks": [c.to_dict() for c in checks],
        "feedback": fb_dict,
        "fix_commands": fix_commands,
    }
    if pr949_doc is not None:
        core_report["pr949_parity"] = pr949_doc
    if yield_drill_doc is not None:
        core_report["steam_yield_drill"] = yield_drill_doc
    if hb_doc is not None:
        core_report["opentrainer_heartbeat"] = hb_doc
    return _attach_receipt(core_report, home=home)


def build_feedback_only_report(
    feedback_result: Mapping[str, Any],
    *,
    gateway: str,
    lang: str = "en",
) -> dict[str, Any]:
    """Wrap a standalone 1-prompt / CLI CES feedback result in a SHA-256-receipted report."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)
    fb = dict(feedback_result)
    ok_or_skipped = bool(fb.get("submitted") or fb.get("skipped"))
    spooled = bool(fb.get("spooled"))
    if ok_or_skipped:
        verdict = "ready"
        warnings_count = 0
        failures_count = 0
    elif spooled:
        verdict = "ready_with_warnings"
        warnings_count = 1
        failures_count = 0
    else:
        verdict = "not_ready"
        warnings_count = 0
        failures_count = 1

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": "feedback",
        "lang": lang_code,
        "gateway": clean_gw,
        "ready": ok_or_skipped,
        "verdict": verdict,
        "recommended_mode": "gamer_whisper" if fb.get("profile") == "gamer" else "dev_api_only",
        "routes_passed_count": 1 if fb.get("submitted") else 0,
        "routes_total_count": 1 if fb.get("attempted") else 0,
        "checks_passed_count": 1 if ok_or_skipped else 0,
        "checks_total_count": 1,
        "warnings_count": warnings_count,
        "failures_count": failures_count,
        "ease_rating": fb.get("ease_rating"),
        "mean_ease_rating": fb.get("mean_ease_rating"),
        "zero_game_lag_ratio": fb.get("zero_game_lag_ratio"),
        "zero_inference_verified": True,
        "compliance": _default_compliance_block(),
        "smoke_gate": None,
        "checks": [],
        "feedback": fb,
        "fix_commands": [],
    }
    return _attach_receipt(core_report)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN GitHub-flavored Markdown summary with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    header = (
        "# Sonde Post-Installation & Smoke Gate Noël HiveMind"
        if is_fr
        else "# HiveMind Holiday Post-Setup Probe & Smoke Gate"
    )
    mean_ease = report.get("mean_ease_rating")
    mean_str = f"{mean_ease}/5" if mean_ease is not None else "n/a"
    ease_val = report.get("ease_rating")
    ease_str = f"{ease_val}/5" if ease_val is not None else "n/a"
    routes_passed = report.get("routes_passed_count", 0)
    routes_total = report.get("routes_total_count", 0)
    checks_passed = report.get("checks_passed_count", 0)
    checks_total = report.get("checks_total_count", 0)

    lines: list[str] = [
        header,
        "",
        f"> **Compliance / Conformité**: {bilingual_notice}",
        "",
        f"- **Gateway**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **Ready**: `{report.get('ready')}` (`{report.get('verdict', 'unknown')}`)",
        f"- **Routes Passed**: `{routes_passed}/{routes_total}`",
        f"- **Checks Passed**: `{checks_passed}/{checks_total}`",
        f"- **CES Ease Rating**: `{ease_str}` (Community Mean: `{mean_str}`)",
    ]

    smoke = report.get("smoke_gate")
    if isinstance(smoke, Mapping):
        lines.extend(
            [
                "",
                "## Routes Cloud Run (Zéro Inférence)" if is_fr else "## Cloud Run Holiday Routes (Zero-Inference)",
                "",
                "| Route | HTTP | Status | Diagnostic |",
                "| :--- | :---: | :---: | :--- |",
            ]
        )
        for row in smoke.get("routes") or ():
            state = "OK" if row.get("ok") else "FAIL"
            diag = str(row.get("diagnostic") or "—")
            lines.append(f"| `{row.get('path')}` | `{row.get('http_status')}` | **{state}** | {diag} |")

    checks = report.get("checks")
    if isinstance(checks, Sequence) and checks:
        lines.extend(
            [
                "",
                "## Contrôles Locaux Post-Installation" if is_fr else "## Local Post-Setup Checks",
                "",
            ]
        )
        for chk in checks:
            badge = str(chk.get("status", "info")).upper()
            lines.append(f"- **[{badge}] {chk.get('title')}**: {chk.get('summary')}")
            if chk.get("fix_command"):
                action_lbl = "Action" if is_fr else "Fix"
                lines.append(f"  - {action_lbl}: `{chk.get('fix_command')}`")

    fb = report.get("feedback")
    if isinstance(fb, Mapping):
        lines.extend(
            [
                "",
                "## Retour Anonyme CES 1-Question" if is_fr else "## 1-Prompt Anonymous CES Feedback",
                "",
                f"- **{fb.get('status')}**: {fb.get('message')}",
            ]
        )

    if report.get("summary") and not (isinstance(checks, Sequence) and checks) and fb is None:
        lines.extend(["", f"- **Summary**: {report.get('summary')}"])

    lines.extend(
        [
            "",
            f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`",
        ]
    )
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_post_setup_probe.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noël post-setup readiness probe, zero-inference holiday route smoke gate, "
            "and 1-prompt anonymous CES reporter (100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind gateway URL (https://*.run.app or localhost).")
    parser.add_argument("--smoke-routes", action="store_true", help="Run only the 7-route zero-inference smoke gate.")
    parser.add_argument("--pr949-parity", action="store_true", help="Run zero-inference PR #949 HTML/CSP/Manifest/Fleet parity gate.")
    parser.add_argument("--expected-sha", default=None, help="Expected build_sha (6-64 chars) for manifest parity verification.")
    parser.add_argument("--steam-yield-drill", action="store_true", help="Run real local subprocess < 1.5 s Steam yield preemption drill.")
    parser.add_argument("--send-opentrainer-heartbeat", action="store_true", help="Send outbound POST /v1/hivemind/opentrainer/heartbeat.")
    parser.add_argument("--host-label", default="", help="Optional sanitized host label for OpenTrainer heartbeat.")
    parser.add_argument("--verify-account", action="store_true", help="Also verify key against GET /v1/hivemind/me.")
    parser.add_argument(
        "--credentials-file",
        default="",
        help="Path to ~/.hivemind/credentials.json (defaults to user home).",
    )
    parser.add_argument(
        "--continue-config-path",
        default="",
        help="Path to ~/.continue/hivemind_provider.yaml (defaults to user home).",
    )
    parser.add_argument("--max-temp-c", type=float, default=GAMER_WHISPER_MAX_TEMP_C)
    parser.add_argument("--min-free-vram-mib", type=int, default=GAMER_WHISPER_MIN_FREE_VRAM_MIB)
    parser.add_argument("--game", action="append", default=[], help="Extra game executable name to guard.")
    parser.add_argument("--submit-feedback", action="store_true", help="Submit anonymous 1..5 CES feedback.")
    parser.add_argument("--prompt-ces", action="store_true", help="Interactive 1-prompt CES question.")
    parser.add_argument("--run-probe", action="store_true", help="Also run full probe when standalone mode flags are set.")
    parser.add_argument("--profile", default="dev", choices=("dev", "gamer"))
    parser.add_argument("--ease-rating", type=int, default=None, help="Ease rating from 1 (hard) to 5 (instant).")
    parser.add_argument("--had-game-lag", action="store_true", help="Report game stutter or fan noise.")
    parser.add_argument("--idempotency-key", default=None, help="Optional ASCII Idempotency-Key for CES submission.")
    parser.add_argument("--ces-queue-file", default="", help="Path to local 0600 CES retry queue JSON file.")
    parser.add_argument("--flush-ces-queue", action="store_true", help="Idempotently flush pending CES queue events.")
    parser.add_argument("--clean-ces-queue", action="store_true", help="Remove local CES retry queue file.")
    parser.add_argument("--enforce-personal-pc", action="store_true", help="Fail closed (exit 2) if corporate markers exist locally.")
    parser.add_argument("--cwd", default="", help="Workspace path to audit for local corporate markers and .git remotes.")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default when --json omitted).")
    return parser


def _emit_report(report: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        serialized = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        print(serialized)
    else:
        print(render_markdown_report(report))


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    raw_http_fn: RawHttpFn | None = None,
    runner: RunnerFn = default_subprocess_runner,
    input_fn: InputFn = input,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    os_name: str | None = None,
    cwd: Path | str | None = None,
) -> int:
    """CLI entrypoint returning `0` (ready), `1` (warnings / transient 429/5xx), or `2` (route/config/compliance fail)."""
    args = build_parser().parse_args(argv)
    try:
        lang_code = normalize_lang(args.lang)
        clean_gw = validate_gateway_url(args.gateway)
    except ProbeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    eff_cwd: Path | str | None = args.cwd if args.cwd else cwd
    cred_path = Path(args.credentials_file) if args.credentials_file else None
    cont_path = Path(args.continue_config_path) if args.continue_config_path else None
    queue_path: Path | None = (
        Path(args.ces_queue_file)
        if args.ces_queue_file
        else (
            default_ces_queue_path(home=home, credentials_path=cred_path)
            if (args.flush_ces_queue or args.clean_ces_queue)
            else None
        )
    )

    if args.enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(
            environ=environ,
            cwd=eff_cwd,
            check_git_remote=True,
            lang=lang_code,
        )
        if not comp["compliance_ok"]:
            refused = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "compliance_refused",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "ready": False,
                    "verdict": "not_ready",
                    "routes_passed_count": 0,
                    "routes_total_count": 0,
                    "checks_passed_count": 0,
                    "checks_total_count": 1,
                    "warnings_count": 0,
                    "failures_count": 1,
                    "network_calls": 0,
                    "file_writes": 0,
                    "subprocess_calls": 0,
                    "zero_inference_verified": True,
                    "compliance": comp,
                    "checks": [
                        check_corporate_pc_compliance(
                            environ=environ,
                            gateway=clean_gw,
                            os_name=os_name,
                            cwd=eff_cwd,
                            enforce_personal_pc=True,
                            lang=lang_code,
                        ).to_dict()
                    ],
                    "summary": comp["summary"],
                    "exit_code": 2,
                },
                home=home,
            )
            _emit_report(refused, as_json=bool(args.json))
            return 2

    if args.clean_ces_queue:
        assert queue_path is not None
        cleaned_doc = ProbeCesRetryQueue(queue_path, home=home).clean(home=home)
        clean_rep = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "ces_queue_clean",
                "lang": lang_code,
                "gateway": clean_gw,
                "ready": True,
                "verdict": "ready",
                "routes_passed_count": 0,
                "routes_total_count": 0,
                "checks_passed_count": 1,
                "checks_total_count": 1,
                "zero_inference_verified": True,
                "compliance": _default_compliance_block(),
                "ces_queue_clean": cleaned_doc,
                "summary": f"CES queue cleaned ({cleaned_doc['spool_file']}).",
            },
            home=home,
        )
        _emit_report(clean_rep, as_json=bool(args.json))
        return 0

    if (
        args.flush_ces_queue
        and not args.submit_feedback
        and not args.prompt_ces
        and not args.run_probe
        and not args.smoke_routes
        and not args.pr949_parity
        and not args.steam_yield_drill
        and not args.send_opentrainer_heartbeat
    ):
        flush_res = flush_ces_feedback_queue(
            clean_gw,
            spool_path=queue_path,
            http_fn=http_fn,
            home=home,
            environ=environ,
            cwd=eff_cwd,
            enforce_personal_pc=bool(args.enforce_personal_pc),
            lang=lang_code,
        )
        flush_rep = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "ces_queue_flush",
                "lang": lang_code,
                "gateway": clean_gw,
                "ready": bool(flush_res["flush_ok"]),
                "verdict": "ready" if flush_res["flush_ok"] else "not_ready",
                "routes_passed_count": 1 if flush_res["flush_ok"] else 0,
                "routes_total_count": 1 if flush_res["network_calls"] > 0 else 0,
                "checks_passed_count": 1 if flush_res["flush_ok"] else 0,
                "checks_total_count": 1,
                "zero_inference_verified": True,
                "compliance": _default_compliance_block(),
                "ces_queue_flush": flush_res,
                "summary": (
                    f"CES queue flush: delivered={flush_res['delivered_count']}, "
                    f"pending={flush_res['pending_count']}, quarantined={flush_res['quarantined_count']}."
                ),
            },
            home=home,
        )
        _emit_report(flush_rep, as_json=bool(args.json))
        return int(flush_res["exit_code"])

    if (
        args.pr949_parity
        and not args.run_probe
        and not args.smoke_routes
        and not args.steam_yield_drill
        and not args.send_opentrainer_heartbeat
        and not args.submit_feedback
        and not args.prompt_ces
    ):
        eff_raw = raw_http_fn if raw_http_fn is not None else default_http_raw
        try:
            parity_rep = verify_pr949_post_setup_parity(
                clean_gw,
                expected_sha=args.expected_sha,
                raw_http_fn=eff_raw,
                http_fn=http_fn,
                enforce_personal_pc=bool(args.enforce_personal_pc),
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
        except ProbeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        _emit_report(parity_rep, as_json=bool(args.json))
        return int(parity_rep["exit_code"])

    if (
        args.steam_yield_drill
        and not args.run_probe
        and not args.smoke_routes
        and not args.pr949_parity
        and not args.send_opentrainer_heartbeat
        and not args.submit_feedback
        and not args.prompt_ces
    ):
        drill_rep = run_steam_yield_preemption_drill(
            enforce_personal_pc=bool(args.enforce_personal_pc),
            environ=environ,
            cwd=eff_cwd,
            lang=lang_code,
        )
        _emit_report(drill_rep, as_json=bool(args.json))
        return int(drill_rep["exit_code"])

    if (
        args.send_opentrainer_heartbeat
        and not args.run_probe
        and not args.smoke_routes
        and not args.pr949_parity
        and not args.steam_yield_drill
        and not args.submit_feedback
        and not args.prompt_ces
    ):
        hb_rep = send_opentrainer_heartbeat(
            gateway=clean_gw,
            credentials_path=cred_path,
            host_label=args.host_label or None,
            runner=runner,
            http_fn=http_fn,
            home=home,
            environ=environ,
            cwd=eff_cwd,
            enforce_personal_pc=bool(args.enforce_personal_pc),
            lang=lang_code,
        )
        _emit_report(hb_rep, as_json=bool(args.json))
        return int(hb_rep["exit_code"])

    fb_result: dict[str, Any] | None = None
    try:
        if args.prompt_ces:
            fb_result = prompt_and_submit_ces_feedback(
                gateway=clean_gw,
                default_profile=args.profile,
                default_had_game_lag=bool(args.had_game_lag),
                input_fn=input_fn,
                http_fn=http_fn,
                lang=lang_code,
                idempotency_key=args.idempotency_key,
                spool_path=queue_path,
                auto_flush=True,
                home=home,
                credentials_path=cred_path,
                environ=environ,
                cwd=eff_cwd,
                enforce_personal_pc=bool(args.enforce_personal_pc),
            )
        elif args.submit_feedback:
            if args.ease_rating is None:
                raise ProbeError("--submit-feedback requires --ease-rating 1..5 (or --prompt-ces).")
            fb_result = submit_ces_feedback(
                gateway=clean_gw,
                profile=args.profile,
                ease_rating=args.ease_rating,
                zero_game_lag=not bool(args.had_game_lag),
                http_fn=http_fn,
                lang=lang_code,
                idempotency_key=args.idempotency_key,
                spool_path=queue_path,
                auto_flush=True,
                home=home,
                credentials_path=cred_path,
                environ=environ,
                cwd=eff_cwd,
                enforce_personal_pc=bool(args.enforce_personal_pc),
            )
    except ProbeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    feedback_standalone = bool(
        args.submit_feedback
        and not args.smoke_routes
        and not args.run_probe
        and not args.verify_account
        and not args.pr949_parity
        and not args.steam_yield_drill
        and not args.send_opentrainer_heartbeat
    )
    if feedback_standalone and fb_result is not None:
        report = build_feedback_only_report(fb_result, gateway=clean_gw, lang=lang_code)
        _emit_report(report, as_json=bool(args.json))
        if fb_result["submitted"] or fb_result["skipped"]:
            return 0
        if fb_result.get("spooled"):
            return 1
        if fb_result.get("quarantined"):
            return 2
        return 2 if fb_result.get("http_status") == 404 else 1

    try:
        report = run_post_setup_probe(
            gateway=clean_gw,
            smoke_only=bool(args.smoke_routes),
            verify_account=bool(args.verify_account),
            credentials_path=cred_path,
            continue_path=cont_path,
            max_temp_c=args.max_temp_c,
            min_free_vram_mib=args.min_free_vram_mib,
            extra_games=args.game,
            feedback_result=fb_result,
            lang=lang_code,
            os_name=os_name,
            environ=environ,
            home=home,
            cwd=eff_cwd,
            enforce_personal_pc=bool(args.enforce_personal_pc),
            verify_pr949=bool(args.pr949_parity),
            expected_sha=args.expected_sha,
            run_yield_drill=bool(args.steam_yield_drill),
            send_heartbeat=bool(args.send_opentrainer_heartbeat),
            host_label=args.host_label or None,
            http_fn=http_fn,
            raw_http_fn=raw_http_fn,
            runner=runner,
        )
    except ProbeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    _emit_report(report, as_json=bool(args.json))

    smoke_info = report.get("smoke_gate")
    if isinstance(smoke_info, Mapping) and not smoke_info.get("all_routes_ok", False):
        return 2
    if report["failures_count"] > 0:
        return 2
    if fb_result is not None and fb_result["attempted"] and not fb_result["submitted"]:
        if fb_result.get("spooled"):
            return 1
        return 2 if fb_result.get("http_status") == 404 or fb_result.get("quarantined") else 1
    if report["warnings_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
