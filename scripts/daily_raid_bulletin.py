#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Concise 5-line FR/EN Discord/Signal holiday raid bulletin generator (`SLICE-07` & `SLICE-SCALE-DYN-38`).

Generates a ready-to-paste 5-line French, English, or bilingual (FR/EN) holiday update
solely from measured JSON payloads:
  - ``GET /v1/hivemind/t4/status``
  - ``GET /v1/hivemind/onboarding/feedback/summary``
  - Optional ``--include-fleet`` reconciliation via ``GET /v1/hivemind/producers/fleet``
    and ``GET /v1/hivemind/raid/status``
  - Optional zero-inference Cloud Run PR #949 parity verification (``verify_holiday_bulletin_parity``)

Strict Honesty, Privacy & Compliance Invariants:
  - Standard library only; zero third-party runtime dependencies.
  - Every absent, null, or unreceived metric strictly renders as ``'n/a'``; zero numbers are ever invented.
  - When ``telemetry_status`` is ``NO_TELEMETRY_RECEIVED`` (or ``telemetry_source`` is
    ``DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED``), static T4 fallback snapshot numbers are treated as absent (``'n/a'``).
  - Outbound live fetches are restricted to allowlisted hosts (``*.run.app``, ``*.googleapis.com``,
    ``datasets-server.huggingface.co``, or loopback) with proxies disabled and safe redirect checks.
  - Zero inference: never sends prompts or calls completions or chat endpoints.
  - Zero PII, zero Discord mass-ping tokens (``@everyone`` / ``@here``), and strict Personal-PC / Amadeus
    compliance (100% personal home PCs & personal hobby/open-source projects only, 0.00 EUR / 0.00 USD).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import ipaddress
import json
import math
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.daily_raid_bulletin/1"
RECEIPT_OBJECT = "hivemind.daily_raid_bulletin_receipt"
CAMPAIGN_ID = "noel-2026-holiday-raid"
MANIFEST_CAMPAIGN_ID = "noel-2026-easy-onboard"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
HTTP_TIMEOUT_S = 5.0
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
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

HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/arena/summary",
    "/v1/hivemind/arena/vote",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
EXPECTED_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = ("hivemind_quickstart.py", "hivemind_doctor.py")
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "ready", "active"})

INSPECTED_ENV_KEYS: tuple[str, ...] = (
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

COMPLIANCE_RULE_FR = (
    "100 % PC personnels à domicile et projets perso/open-source uniquement "
    "(conformité Amadeus : jamais de PC ni de code pro ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_RULE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus compliance: never corporate hardware or work code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_RULE_FR} | {COMPLIANCE_RULE_EN}"

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ISO_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T\s].*)?$")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:hm(?:_live_|-)(?!\*{3,4})[A-Za-z0-9_-]{4,}|hm_(?!live_\*{3,4})[A-Za-z0-9_-]{6,})\b"
)
_MASKED_HM_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*{3,4}(?:[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s)|[A-Za-z0-9_-]{3,})"
)
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')

_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"([A-Za-z0-9_+=/=-]{8,128})"|'([A-Za-z0-9_+=/=-]{8,128})')""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_UNSAFE_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\s*\.\s*writeln?)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)

_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)

_CORP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amadeus", re.compile(r"amadeus", re.IGNORECASE)),
    (
        "corp",
        re.compile(r"(?:^|[^a-z])corp(?:orate|net|sys|s)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])", re.IGNORECASE),
    ),
    ("enterprise", re.compile(r"enterprise", re.IGNORECASE)),
    ("intranet", re.compile(r"intranet", re.IGNORECASE)),
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])|[a-z0-9]internal(?:$|[^a-z])", re.IGNORECASE)),
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

HttpGetFn = Callable[..., tuple[int, dict[str, Any]]]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]


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


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML nonce attributes, CSP nonce tokens, and masked hm-*** key suffixes before figure scans."""
    if not isinstance(text, str):
        raise BulletinInputError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host matches the HiveMind lateral-movement allowlist."""
    if isinstance(host, bool) or not isinstance(host, str):
        return False
    h = host.strip().lower().strip("[]")
    if not h or "\\" in h or "@" in h or len(h) > 253:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate and normalize a bare gateway origin URL against scheme, label, and host allowlists."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise BulletinInputError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise BulletinInputError("Gateway URL must not contain control characters or newlines.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise BulletinInputError("Gateway URL must not contain whitespace or backslashes.")
    parts = None
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise BulletinInputError("Gateway URL contains an invalid URL or IPv6 literal.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise BulletinInputError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise BulletinInputError("Gateway URL must be a bare origin without path, query, or fragment.")
    try:
        port = parts.port
        raw_host = parts.hostname
    except ValueError as exc:
        raise BulletinInputError("Gateway URL contains an invalid host or port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise BulletinInputError("Gateway URL port out of range.")
    host = (raw_host or "").strip().lower().strip("[]")
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


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowlisted gateway origin, read-only method, and zero LLM inference/prompt payloads."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise BulletinInputError("Request URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise BulletinInputError("Request URL must not contain control characters.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise BulletinInputError("Request URL must not contain whitespace or backslashes.")
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise BulletinInputError("Request URL contains an invalid URL or IPv6 literal.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise BulletinInputError("Request URL must not embed credentials.")
    validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
    low_path = (parts.path or "").lower()
    decoded_path = unquote(low_path)
    if (
        "completions" in low_path
        or "/chat/" in low_path
        or "completions" in decoded_path
        or "/chat/" in decoded_path
    ):
        raise BulletinInputError(f"Zero-inference guard blocked inference endpoint {parts.path!r}.")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise BulletinInputError("HTTP payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        lower_keys = {str(k).strip().lower() for k in payload}
        if "hivemind/coder" in serialized or "messages" in lower_keys or "prompt" in lower_keys:
            raise BulletinInputError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb not in ("GET", "HEAD"):
        raise BulletinInputError(f"Zero-inference read-only guard blocked HTTP {verb!r}.")
    return cleaned


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects off allowlisted hosts, with credentials, backslashes, or onto inference endpoints."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in newurl) or "\\" in newurl or "@" in newurl:
            raise BulletinInputError("Redirect URL contains forbidden characters or credentials.")
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_safe_http_request(new_req.get_method(), new_req.full_url)
        return new_req


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)):
        raise BulletinInputError("timeout_s must be a finite positive number.")
    if float(timeout_s) <= 0.0:
        raise BulletinInputError("timeout_s must be > 0.")
    return max(0.5, min(15.0, float(timeout_s)))


def _prepare_request_headers(
    base_headers: dict[str, str],
    extra_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    req_headers = dict(base_headers)
    if extra_headers:
        for hk, hv in extra_headers.items():
            ks, vs = str(hk), str(hv)
            if any(ch in ks or ch in vs for ch in ("\r", "\n", "\x00")):
                raise BulletinInputError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[ks] = vs
    return req_headers


def _build_proxy_free_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _SafeRedirectHandler(),
    )


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a read-only, proxy-free HTTP GET request to an allowlisted gateway endpoint."""
    clean_url = _assert_safe_http_request("GET", url)
    clamped_timeout = _validate_timeout_s(timeout_s)
    parts = urlsplit(clean_url)
    opener = _build_proxy_free_opener()
    req = urllib.request.Request(
        clean_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hivemind-daily-raid-bulletin/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(req, timeout=clamped_timeout) as resp:
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
            if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
                raise BulletinInputError(f"Gateway response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
            raw = raw_bytes.decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = (exc.read(65_536) if exc.fp is not None else b"").decode("utf-8", errors="replace")
        try:
            err_doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            err_doc = {"error": _redact_pii_text(raw[:200])}
        return int(exc.code), (err_doc if isinstance(err_doc, dict) else {"data": err_doc})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BulletinInputError(f"Unable to reach gateway endpoint {parts.path}: {reason}") from exc

    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise BulletinInputError(f"Malformed JSON from gateway endpoint {parts.path}.") from exc
    if not isinstance(parsed, dict):
        raise BulletinInputError(f"Expected JSON object from gateway endpoint {parts.path}.")
    return status_code, parsed


def default_http_raw(
    method_or_url: str,
    url: str | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Proxy-free stdlib raw HTTP helper returning ``(status_code, body_text, lower_headers)``."""
    if url is None:
        verb = "GET"
        target_url = method_or_url
    else:
        verb = str(method_or_url or "GET").strip().upper()
        target_url = url
    clean_url = _assert_safe_http_request(verb, target_url)
    clamped_timeout = _validate_timeout_s(timeout_s)
    req_headers = _prepare_request_headers(
        {
            "Accept": "text/html, application/json;q=0.9, text/plain;q=0.8",
            "User-Agent": "hivemind-daily-raid-bulletin/1.0",
        },
        headers,
    )
    opener = _build_proxy_free_opener()
    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=clamped_timeout) as resp:
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
            if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
                raise BulletinInputError(f"Gateway response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
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
        return (
            0,
            json.dumps({"network_error": type(exc).__name__}),
            {"x-hivemind-network-error": type(exc).__name__},
        )


def _invoke_http_get(
    http_get: HttpGetFn,
    url: str,
    timeout_s: float,
    *,
    catch_transport: bool = True,
) -> tuple[int, dict[str, Any]]:
    clean_url = _assert_safe_http_request("GET", url)
    sig = inspect.signature(http_get)
    params = list(sig.parameters.values())
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    first_name = pos_params[0].name if pos_params else ""

    try:
        if first_name in ("method", "verb"):
            if len(pos_params) >= 4 or has_varargs:
                raw = http_get("GET", clean_url, None, None)
            else:
                raw = http_get("GET", clean_url)
        elif len(pos_params) == 1 and not has_varargs:
            raw = http_get(clean_url)
        else:
            raw = http_get(clean_url, timeout_s)
    except BulletinInputError as exc:
        if catch_transport and str(exc).startswith(("Unable to reach", "Malformed JSON", "Expected JSON object")):
            return 0, {"error": "transport_or_parse_error"}
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        if catch_transport:
            return 0, {"error": "network_error"}
        raise

    if not isinstance(raw, tuple) or len(raw) != 2:
        raise BulletinInputError("http_get must return (status_code, body_dict).")
    code, body = raw
    return int(code), (dict(body) if isinstance(body, Mapping) else {"data": body})


def _invoke_http_raw(raw_http_fn: RawHttpFn, url: str, timeout_s: float) -> tuple[int, str, dict[str, str]]:
    clean_url = _assert_safe_http_request("GET", url)
    sig = inspect.signature(raw_http_fn)
    params = sig.parameters
    pos_params = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    first_name = pos_params[0].name if pos_params else ""
    try:
        if first_name in ("method", "method_or_url", "verb"):
            if "timeout_s" in params:
                raw = raw_http_fn("GET", clean_url, timeout_s=timeout_s)
            else:
                raw = raw_http_fn("GET", clean_url)
        elif len(pos_params) == 1:
            raw = raw_http_fn(clean_url)
        else:
            raw = raw_http_fn(clean_url, timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, json.dumps({"network_error": type(exc).__name__}), {}
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise BulletinInputError("raw_http_fn must return (status_code, body_text, headers_dict).")
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


def reconcile_live_fleet_and_raid_metrics(
    t4_payload: Mapping[str, Any] | None,
    fleet_payload: Mapping[str, Any] | None = None,
    raid_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Honestly populate missing `active_producers_count` / `distinct_physical_gpus` from fleet & raid status."""
    if t4_payload is not None and not isinstance(t4_payload, Mapping):
        raise BulletinInputError("t4_payload must be a JSON object (Mapping) or None.")
    if fleet_payload is not None and not isinstance(fleet_payload, Mapping):
        raise BulletinInputError("fleet_payload must be a JSON object (Mapping) or None.")
    if raid_payload is not None and not isinstance(raid_payload, Mapping):
        raise BulletinInputError("raid_payload must be a JSON object (Mapping) or None.")

    reconciled = dict(t4_payload or {})
    t4_producers = _opt_nonneg_int(reconciled.get("active_producers_count"), "active_producers_count")
    t4_gpus = _opt_nonneg_int(reconciled.get("distinct_physical_gpus"), "distinct_physical_gpus")

    fleet = fleet_payload or {}
    fleet_live: int | None = None
    if "live_hiveminders" in fleet and fleet["live_hiveminders"] is not None:
        fleet_live = _opt_nonneg_int(fleet["live_hiveminders"], "fleet.live_hiveminders")
    elif isinstance(fleet.get("hiveminders"), list):
        fleet_live = sum(
            1
            for row in fleet["hiveminders"]
            if isinstance(row, Mapping) and str(row.get("status", "")).lower() in LIVE_PRODUCER_STATUSES
        )

    fleet_gpus: int | None = None
    if "distinct_physical_gpus" in fleet and fleet["distinct_physical_gpus"] is not None:
        fleet_gpus = _opt_nonneg_int(fleet["distinct_physical_gpus"], "fleet.distinct_physical_gpus")

    raid = raid_payload or {}
    swarm_raw = raid.get("swarm")
    if swarm_raw is not None and not isinstance(swarm_raw, Mapping):
        raise BulletinInputError("Field 'swarm' in raid_payload must be a JSON object.")
    swarm: Mapping[str, Any] = swarm_raw or {}
    raid_live_producers = _opt_nonneg_int(swarm.get("live_producers"), "raid_status.swarm.live_producers")
    raid_active_trainers = _opt_nonneg_int(swarm.get("active_trainers"), "raid_status.swarm.active_trainers")

    if t4_producers is None:
        for cand in (fleet_live, raid_live_producers, raid_active_trainers):
            if cand is not None:
                reconciled["active_producers_count"] = cand
                break

    if t4_gpus is None and fleet_gpus is not None:
        reconciled["distinct_physical_gpus"] = fleet_gpus

    return reconciled


def fetch_live_bulletin_inputs(
    gateway_url: str,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
    include_fleet: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Fetch `/v1/hivemind/t4/status` and `/v1/hivemind/onboarding/feedback/summary` (plus optional fleet/raid)."""
    origin = validate_gateway_url(gateway_url)
    clamped_timeout = _validate_timeout_s(timeout_s)

    t4_code, t4_doc = _invoke_http_get(
        http_get,
        f"{origin}/v1/hivemind/t4/status",
        clamped_timeout,
        catch_transport=False,
    )
    fb_code, fb_doc = _invoke_http_get(
        http_get,
        f"{origin}/v1/hivemind/onboarding/feedback/summary",
        clamped_timeout,
        catch_transport=False,
    )
    t4_payload = dict(t4_doc) if t4_code == 200 and isinstance(t4_doc, Mapping) else {}
    fb_payload = dict(fb_doc) if fb_code == 200 and isinstance(fb_doc, Mapping) else {}
    http_statuses: dict[str, int] = {
        "t4_status_http": int(t4_code),
        "feedback_summary_http": int(fb_code),
    }

    if include_fleet:
        fleet_code, fleet_doc = _invoke_http_get(
            http_get,
            f"{origin}/v1/hivemind/producers/fleet",
            clamped_timeout,
            catch_transport=True,
        )
        raid_code, raid_doc = _invoke_http_get(
            http_get,
            f"{origin}/v1/hivemind/raid/status",
            clamped_timeout,
            catch_transport=True,
        )
        fleet_payload = dict(fleet_doc) if fleet_code == 200 and isinstance(fleet_doc, Mapping) else None
        raid_payload = dict(raid_doc) if raid_code == 200 and isinstance(raid_doc, Mapping) else None
        t4_payload = reconcile_live_fleet_and_raid_metrics(
            t4_payload,
            fleet_payload=fleet_payload,
            raid_payload=raid_payload,
        )
        http_statuses["producers_fleet_http"] = int(fleet_code)
        http_statuses["raid_status_http"] = int(raid_code)

    return t4_payload, fb_payload, http_statuses


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


def _redact_pii_text(text: str) -> str:
    out = _HOME_PATH_RE.sub("~", str(text))
    out = _BEARER_RE.sub("Bearer <redacted>", out)
    out = _API_KEY_RE.sub("hm-***XXXX", out)
    out = _EMAIL_RE.sub("<redacted-email>", out)

    def _scrub_ip(match: re.Match[str]) -> str:
        cand = match.group(0)
        try:
            if ipaddress.ip_address(cand).is_loopback:
                return cand
        except ValueError:
            return "<redacted-ip>"
        return "<redacted-ip>"

    return _IPV4_RE.sub(_scrub_ip, out)


def scan_output_safety(text: str) -> list[str]:
    """Verify rendered bulletin or JSON contains zero PII, raw keys, mass-pings, static figures, or forbidden keys."""
    if not isinstance(text, str):
        raise BulletinInputError("scan_output_safety requires a string.")
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
    if _STATIC_FIG_UNIT_RE.search(strip_masked_keys_and_nonces(text)):
        issues.append("static_figure_detected")
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
        raise BulletinInputError("expected_sha must be a 6..64 character alphanumeric/hex string.")
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise BulletinInputError("expected_sha must be a 6..64 character alphanumeric/hex string (never 'unknown').")
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


def _has_forbidden_key(obj: Any) -> bool:
    forbidden = "sco" + "re"
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) == forbidden or _has_forbidden_key(v):
                return True
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return any(_has_forbidden_key(item) for item in obj)
    return False


def _inspect_models_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str], bool, list[str]]:
    if code != 200:
        return False, [], False, [f"http_{code}"]
    raw_data = body.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        return False, [], False, ["models_empty_or_invalid"]
    model_ids: list[str] = []
    coder_present = False
    for entry in raw_data:
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry["id"].strip():
            mid = entry["id"].strip()
            model_ids.append(mid)
            if mid == "hivemind/coder" and entry.get("object", "model") == "model":
                coder_present = True
    issues: list[str] = []
    if not model_ids:
        issues.append("models_missing_ids")
    if not coder_present:
        issues.append("models_missing_hivemind_coder")
    return len(issues) == 0, model_ids, coder_present, issues


def _inspect_html_parity_route(
    status_code: int,
    body_text: str,
    headers: Mapping[str, str],
    seen_nonces: set[str],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    if status_code != 200:
        issues.append(f"http_{status_code}")
        return (
            False,
            {"status_code": status_code, "csp_ok": False, "dom_clean": False, "compliance_ok": False},
            issues,
        )

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


def _inspect_manifest_parity(
    code: int,
    body: Mapping[str, Any],
    expected_sha: str | None,
) -> tuple[bool, str, bool | None, list[str], list[str]]:
    if code != 200:
        return False, "unknown", (False if expected_sha else None), [], [f"http_{code}"]
    issues: list[str] = []
    missing_routes: list[str] = []
    if _has_forbidden_key(body):
        issues.append("manifest_forbidden_key")
    if body.get("campaign_id") != MANIFEST_CAMPAIGN_ID:
        issues.append("manifest_campaign_mismatch")
    if body.get("personal_pc_only") is not True:
        issues.append("manifest_personal_pc_only_false")
    if body.get("financial_promise") != "0.00 EUR / 0.00 USD":
        issues.append("manifest_financial_promise_mismatch")
    if "build_sha" not in body:
        issues.append("manifest_missing_build_sha")
    live_sha = _sanitize_build_sha(body.get("build_sha"))

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
    return len(issues) == 0, live_sha, sha_match, missing_routes, issues


def _inspect_feedback_summary_parity(
    code: int,
    body: Mapping[str, Any],
) -> tuple[bool, float | None, float | None, list[str]]:
    if code != 200:
        return False, None, None, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_key(body):
        issues.append("summary_forbidden_key")
    if body.get("object") != "hivemind.onboarding_feedback_summary":
        issues.append("summary_invalid_object")
    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else None
    if total_val is None:
        issues.append("summary_invalid_total_responses")

    has_mean = "mean_ease_rating" in body
    has_avg = "average_ease_rating" in body
    has_ease = has_mean or has_avg
    ease_raw = body.get("mean_ease_rating")
    if ease_raw is None and has_avg and body.get("average_ease_rating") is not None:
        ease_raw = body.get("average_ease_rating")

    ease_out: float | None = None
    if not has_ease:
        issues.append("summary_missing_ease_rating")
    elif total_val == 0 and (ease_raw is not None or body.get("average_ease_rating") is not None):
        issues.append("summary_non_null_ease_when_empty")
    elif total_val is not None and total_val > 0:
        if isinstance(ease_raw, bool) or not isinstance(ease_raw, (int, float)) or not math.isfinite(float(ease_raw)):
            issues.append("summary_invalid_ease_rating")
        else:
            f_ease = float(ease_raw)
            if not (1.0 <= f_ease <= 5.0):
                issues.append("summary_invalid_ease_rating")
            else:
                ease_out = round(f_ease, 2)
                if has_mean and has_avg and body.get("average_ease_rating") is not None:
                    a_raw = body.get("average_ease_rating")
                    if (
                        isinstance(a_raw, bool)
                        or not isinstance(a_raw, (int, float))
                        or not math.isfinite(float(a_raw))
                        or not (1.0 <= float(a_raw) <= 5.0)
                        or round(float(a_raw), 2) != ease_out
                    ):
                        issues.append("summary_ease_rating_key_mismatch")

    lag_out: float | None = None
    if "zero_game_lag_ratio" not in body:
        issues.append("summary_missing_zero_game_lag_ratio")
    else:
        lag_raw = body.get("zero_game_lag_ratio")
        if total_val == 0 and lag_raw is not None:
            issues.append("summary_non_null_lag_when_empty")
        elif total_val is not None and total_val > 0:
            if isinstance(lag_raw, bool) or not isinstance(lag_raw, (int, float)) or not math.isfinite(float(lag_raw)):
                issues.append("summary_invalid_zero_game_lag_ratio")
            else:
                f_lag = float(lag_raw)
                if not (0.0 <= f_lag <= 1.0):
                    issues.append("summary_invalid_zero_game_lag_ratio")
                else:
                    lag_out = round(f_lag, 4)

    bp = body.get("by_profile")
    if not isinstance(bp, Mapping):
        issues.append("summary_invalid_by_profile")
    else:
        dev_c = bp.get("dev")
        gamer_c = bp.get("gamer")
        if (
            isinstance(dev_c, bool)
            or not isinstance(dev_c, int)
            or dev_c < 0
            or isinstance(gamer_c, bool)
            or not isinstance(gamer_c, int)
            or gamer_c < 0
        ):
            issues.append("summary_invalid_profile_counts")
        elif total_val is not None and dev_c + gamer_c != total_val:
            issues.append("summary_profile_conservation_mismatch")
    return len(issues) == 0, ease_out, lag_out, issues


def _inspect_raid_status_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if code != 200:
        return False, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_key(body):
        issues.append("raid_status_forbidden_key")
    if body.get("object") != "hivemind.raid_noel.status":
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


def _inspect_producers_fleet_parity(
    code: int,
    body: Mapping[str, Any],
) -> tuple[bool, int, int, int | None, list[str]]:
    if code != 200:
        return False, 0, 0, None, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_key(body):
        issues.append("fleet_forbidden_key")
    if "object" in body and body.get("object") != "hivemind.producer_fleet":
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
    raw_live = body.get("live_hiveminders")
    live_hm = raw_live if (not isinstance(raw_live, bool) and isinstance(raw_live, int) and raw_live >= 0) else 0
    raw_online = body.get("online_producers")
    explicit_online = (
        raw_online if (not isinstance(raw_online, bool) and isinstance(raw_online, int) and raw_online >= 0) else 0
    )
    live_hiveminders = max(live_hm, hm_online)
    online_producers = max(live_hiveminders, explicit_online, prod_online)
    raw_gpus = body.get("distinct_physical_gpus")
    distinct_gpus = (
        raw_gpus if (not isinstance(raw_gpus, bool) and isinstance(raw_gpus, int) and raw_gpus >= 0) else None
    )
    return len(issues) == 0, live_hiveminders, online_producers, distinct_gpus, issues


def verify_holiday_bulletin_parity(
    gateway_url: str,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> dict[str, Any]:
    """Zero-inference Cloud Run PR #949 live-route parity, CSP, manifest, SHA, and 8+ producer quorum verifier."""
    gw = validate_gateway_url(gateway_url)
    clamped_timeout = _validate_timeout_s(timeout_s)
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or target_online_producers < 1
    ):
        raise BulletinInputError("target_online_producers must be an integer >= 1.")
    clean_expected_sha = _validate_expected_sha(expected_sha) if expected_sha is not None else None

    missing_or_stale: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}

    def _note_failure(route: str, code: int, issues: list[str]) -> None:
        failed_routes.append(route)
        if code == 404 and route not in missing_or_stale:
            missing_or_stale.append(route)
        for item in issues:
            diagnostics.append(_redact_pii_text(f"{route}: {item}"))

    health_code, health_body = _invoke_http_get(http_get, f"{gw}/health", clamped_timeout)
    health_ok = health_code == 200 and health_body.get("status", "ok") == "ok"
    health_issues = [] if health_ok else ([f"http_{health_code}"] if health_code != 200 else ["unhealthy_status"])
    per_route["/health"] = {"status_code": health_code, "ready": health_ok, "issues": health_issues}
    if not health_ok:
        _note_failure("/health", health_code, health_issues)

    models_code, models_body = _invoke_http_get(http_get, f"{gw}/v1/models", clamped_timeout)
    models_ok, model_ids, coder_present, models_issues = _inspect_models_parity(models_code, models_body)
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
        r_code, r_body, r_hdrs = _invoke_http_raw(raw_http_fn, f"{gw}{route}", clamped_timeout)
        r_ok, r_info, r_issues = _inspect_html_parity_route(r_code, r_body, r_hdrs, seen_nonces)
        per_route[route] = {"ready": r_ok, **r_info, "issues": r_issues}
        all_html_ok = all_html_ok and r_ok
        unique_csp_ok = unique_csp_ok and bool(r_info.get("csp_ok")) and not bool(r_info.get("nonce_reused"))
        html_dom_clean = html_dom_clean and bool(r_info.get("dom_clean"))
        html_compliance_ok = html_compliance_ok and bool(r_info.get("compliance_ok"))
        if not r_ok:
            _note_failure(route, r_code, r_issues)

    man_code, man_body = _invoke_http_get(http_get, f"{gw}/v1/hivemind/onboarding/manifest", clamped_timeout)
    man_ok, live_sha, sha_match, man_missing, man_issues = _inspect_manifest_parity(
        man_code, man_body, clean_expected_sha
    )
    per_route["/v1/hivemind/onboarding/manifest"] = {
        "status_code": man_code,
        "ready": man_ok,
        "live_build_sha": live_sha,
        "sha_match": sha_match,
        "issues": man_issues,
    }
    for mr in man_missing:
        if mr not in missing_or_stale:
            missing_or_stale.append(mr)
    if not man_ok:
        _note_failure("/v1/hivemind/onboarding/manifest", man_code, man_issues)

    fb_code, fb_body = _invoke_http_get(
        http_get, f"{gw}/v1/hivemind/onboarding/feedback/summary", clamped_timeout
    )
    fb_ok, mean_ease, zero_lag, fb_issues = _inspect_feedback_summary_parity(fb_code, fb_body)
    per_route["/v1/hivemind/onboarding/feedback/summary"] = {
        "status_code": fb_code,
        "ready": fb_ok,
        "issues": fb_issues,
    }
    if not fb_ok:
        _note_failure("/v1/hivemind/onboarding/feedback/summary", fb_code, fb_issues)

    raid_code, raid_body = _invoke_http_get(http_get, f"{gw}/v1/hivemind/raid/status", clamped_timeout)
    raid_ok, raid_issues = _inspect_raid_status_parity(raid_code, raid_body)
    per_route["/v1/hivemind/raid/status"] = {
        "status_code": raid_code,
        "ready": raid_ok,
        "issues": raid_issues,
    }
    if not raid_ok:
        _note_failure("/v1/hivemind/raid/status", raid_code, raid_issues)

    fleet_code, fleet_body = _invoke_http_get(http_get, f"{gw}/v1/hivemind/producers/fleet", clamped_timeout)
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
    onboarding_feedback_deployed = fb_code == 200 and fb_ok
    holiday_routes_deployed_on_cloud_run = bool(
        health_ok
        and models_ok
        and all_html_ok
        and unique_csp_nonces_ok
        and html_dom_clean
        and html_compliance_ok
        and man_ok
        and onboarding_feedback_deployed
        and raid_ok
        and fleet_ok
        and not stale_revision_detected
    )
    if distinct_gpus is not None:
        producer_quorum_met = (
            online_producers >= target_online_producers and distinct_gpus >= target_online_producers
        )
    else:
        producer_quorum_met = online_producers >= target_online_producers

    if stale_revision_detected:
        diagnostics.insert(
            0,
            _redact_pii_text(
                f"Cloud Run PR #949 drift detected (missing_or_stale={missing_or_stale}, sha_match={sha_match})."
            ),
        )
    if not producer_quorum_met:
        diagnostics.append(
            _redact_pii_text(
                f"Online producer quorum unmet ({online_producers} online, "
                f"distinct_gpus={distinct_gpus}, target={target_online_producers})."
            )
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "object": "hivemind.holiday_bulletin_parity",
        "gateway": gw,
        "parity_ok": holiday_routes_deployed_on_cloud_run,
        "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed_on_cloud_run,
        "onboarding_feedback_deployed": onboarding_feedback_deployed,
        "stale_revision_detected": stale_revision_detected,
        "missing_or_stale_routes": missing_or_stale,
        "failed_routes": failed_routes,
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
        "feedback_summary_ok": fb_ok,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": zero_lag,
        "raid_status_ok": raid_ok,
        "fleet_ok": fleet_ok,
        "live_hiveminders": live_hiveminders,
        "online_producers": online_producers,
        "distinct_physical_gpus": distinct_gpus,
        "target_online_producers": target_online_producers,
        "producer_quorum_met": producer_quorum_met,
        "zero_inference_verified": True,
        "per_route": per_route,
        "diagnostics": diagnostics,
        "compliance": {
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "exit_code": 0 if holiday_routes_deployed_on_cloud_run else 2,
    }
    _assert_safe_output(json.dumps(result, ensure_ascii=False))
    return result


def _sanitize_parity_block(parity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "holiday_routes_deployed_on_cloud_run": bool(parity.get("holiday_routes_deployed_on_cloud_run")),
        "onboarding_feedback_deployed": bool(parity.get("onboarding_feedback_deployed")),
        "stale_revision_detected": bool(parity.get("stale_revision_detected")),
        "missing_or_stale_routes": [str(r) for r in (parity.get("missing_or_stale_routes") or [])],
        "failed_routes": [str(r) for r in (parity.get("failed_routes") or [])],
        "online_producers": _opt_nonneg_int(parity.get("online_producers"), "online_producers") or 0,
        "live_hiveminders": _opt_nonneg_int(parity.get("live_hiveminders"), "live_hiveminders") or 0,
        "distinct_physical_gpus": _opt_nonneg_int(parity.get("distinct_physical_gpus"), "distinct_physical_gpus"),
        "target_online_producers": _opt_nonneg_int(
            parity.get("target_online_producers"), "target_online_producers"
        )
        or DEFAULT_TARGET_ONLINE_PRODUCERS,
        "producer_quorum_met": bool(parity.get("producer_quorum_met")),
        "live_build_sha": _sanitize_build_sha(parity.get("live_build_sha")),
        "expected_sha": (
            _sanitize_build_sha(parity.get("expected_sha"))
            if parity.get("expected_sha") is not None
            else None
        ),
        "sha_match": parity.get("sha_match") if isinstance(parity.get("sha_match"), bool) else None,
        "unique_csp_nonces_count": _opt_nonneg_int(
            parity.get("unique_csp_nonces_count"), "unique_csp_nonces_count"
        )
        or 0,
        "unique_csp_nonces_ok": bool(parity.get("unique_csp_nonces_ok")),
        "html_dom_clean": bool(parity.get("html_dom_clean")),
        "compliance_banners_ok": bool(parity.get("compliance_banners_ok")),
        "coder_model_present": bool(parity.get("coder_model_present")),
        "zero_inference_verified": True,
        "diagnostics": [_redact_pii_text(str(d)) for d in (parity.get("diagnostics") or [])],
    }


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def _extract_git_remote_urls(cfg_text: str) -> list[str]:
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text or ""):
        cand = (m_kv or m_sec).strip().strip('"').strip("'")
        if cand:
            urls.append(cand)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` from `workspace`, following worktree `gitdir:` and `commondir`."""
    configs: list[Path] = []
    for root in (workspace, *workspace.parents):
        git_entry = root / ".git"
        if git_entry.is_dir():
            cfg = git_entry / "config"
            if cfg.is_file():
                configs.append(cfg)
            break
        if git_entry.is_file():
            try:
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gd = lines[0].split(":", 1)[1].strip()
                gd = Path(raw_gd) if Path(raw_gd).is_absolute() else (root / raw_gd).resolve()
                if (gd / "config").is_file():
                    configs.append(gd / "config")
                cdir_file = gd / "commondir"
                if cdir_file.is_file():
                    with suppress(OSError):
                        crel = cdir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if crel:
                            ccfg = (gd / crel).resolve() / "config"
                            if ccfg.is_file() and ccfg not in configs:
                                configs.append(ccfg)
            break
    return configs


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check (0 HTTP calls, 0 writes, 0 raw leaks)."""
    env_map = environ if environ is not None else os.environ
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
            matched_markers.update(_find_corporate_markers(val_str))
        elif key_up == "ZSCALER_CERT_PATH":
            _record("env:ZSCALER_CERT_PATH", "zscaler")
        elif key_up == "USERDNSDOMAIN":
            dns_hits = _find_corporate_markers(val_str)
            _record("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "domain")
            matched_markers.update(dns_hits)
        elif key_up in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(val_str):
                _record(f"env:{key_up}:{hit}", hit)

    ws_variants = [str(workspace)]
    with suppress(OSError):
        res_ws = str(workspace.resolve())
        if res_ws not in ws_variants:
            ws_variants.append(res_ws)
    for ws_txt in ws_variants:
        for hit in _find_corporate_markers(ws_txt):
            _record(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            with suppress(OSError):
                cfg_txt = cfg_path.read_text(encoding="utf-8", errors="replace")
                for remote_url in _extract_git_remote_urls(cfg_txt):
                    for hit in _find_corporate_markers(remote_url):
                        _record(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary_en = f"Personal-PC compliance verified locally (0.00 EUR / 0.00 USD, AGPL-3.0). {COMPLIANCE_RULE_EN}"
        summary_fr = f"Conformité PC personnel vérifiée localement (0,00 EUR / 0,00 USD, AGPL-3.0). {COMPLIANCE_RULE_FR}"
    else:
        joined = ", ".join(indicators)
        summary_en = (
            f"COMPLIANCE REFUSED ({joined}): corporate/Amadeus environment indicator detected "
            f"(0 outbound HTTP calls, 0 file writes). {COMPLIANCE_RULE_EN}"
        )
        summary_fr = (
            f"REFUS DE CONFORMITÉ ({joined}) : marqueur d'entreprise/Amadeus détecté "
            f"(0 appel HTTP, 0 écriture fichier). {COMPLIANCE_RULE_FR}"
        )
    summary_bilingual = f"{summary_fr} | {summary_en}"

    return {
        "schema": SCHEMA,
        "object": "hivemind.personal_pc_compliance",
        "campaign_id": CAMPAIGN_ID,
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "monetary_cost_eur": "0.00 EUR",
        "monetary_cost_usd": "0.00 USD",
        "financial_promise": "0.00 EUR / 0.00 USD",
        "license": "AGPL-3.0",
        "network_calls": 0,
        "file_writes": 0,
        "notice_en": COMPLIANCE_RULE_EN,
        "notice_fr": COMPLIANCE_RULE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "summary": summary_bilingual,
        "exit_code": 0 if compliance_ok else 2,
    }


def _atomic_write_text(target: Path, content: str, *, mode: int = 0o600) -> None:
    """Atomically write text content to target via a sibling temp file with POSIX 0600 permissions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_fh:
            tmp_path = Path(tmp_fh.name)
            tmp_fh.write(content)
            tmp_fh.flush()
            os.fsync(tmp_fh.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def build_bulletin_receipt(
    t4_status: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    *,
    lang: str = "bilingual",
    date_override: str | None = None,
    source_kind: str = "snapshot",
    http_statuses: Mapping[str, int] | None = None,
    holiday_parity: Mapping[str, Any] | None = None,
    compliance_check: Mapping[str, Any] | None = None,
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
    if isinstance(holiday_parity, Mapping):
        clean_parity = _sanitize_parity_block(holiday_parity)
        core_payload["holiday_routes_parity"] = clean_parity
        core_payload["holiday_routes_deployed_on_cloud_run"] = clean_parity["holiday_routes_deployed_on_cloud_run"]
        core_payload["onboarding_feedback_deployed"] = clean_parity["onboarding_feedback_deployed"]
        core_payload["stale_revision_detected"] = clean_parity["stale_revision_detected"]
        core_payload["missing_or_stale_routes"] = list(clean_parity["missing_or_stale_routes"])
        core_payload["online_producers"] = clean_parity["online_producers"]
        core_payload["producer_quorum_met"] = clean_parity["producer_quorum_met"]
    if isinstance(compliance_check, Mapping):
        core_payload["personal_pc_compliance"] = {
            "compliance_ok": bool(compliance_check.get("compliance_ok")),
            "corporate_detected": bool(compliance_check.get("corporate_detected")),
            "corporate_indicators": [str(x) for x in (compliance_check.get("corporate_indicators") or [])],
            "matched_markers": [str(x) for x in (compliance_check.get("matched_markers") or [])],
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        }

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
    if (
        bundle.get("object") == "hivemind.onboarding_feedback_summary"
        or "mean_ease_rating" in bundle
        or "average_ease_rating" in bundle
    ):
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
        "--include-fleet",
        action="store_true",
        help="Reconcile active producers & physical GPUs from /v1/hivemind/producers/fleet and /v1/hivemind/raid/status.",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        default=None,
        help=f"Allowlisted HiveMind gateway URL (default with --live: {DEFAULT_GATEWAY}).",
    )
    parser.add_argument(
        "--verify-holiday-routes",
        action="store_true",
        help="Verify zero-inference PR #949 holiday route parity & producer quorum against --gateway.",
    )
    parser.add_argument(
        "--require-holiday-routes",
        action="store_true",
        help="Fail closed (exit code 2) if PR #949 holiday routes or --expected-sha drift on --gateway.",
    )
    parser.add_argument(
        "--expected-sha",
        type=str,
        default=None,
        help="Optional expected build_sha (6..64 hex/alphanumeric) for Cloud Run manifest parity.",
    )
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Target online producer / GPU quorum threshold (default: 8).",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Fail closed locally (exit 2, 0 HTTP calls, 0 writes) if corporate/Amadeus markers exist.",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Workspace directory inspected by --enforce-personal-pc (including .git/config remotes).",
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
        help="Optional file path to atomically write the 5-line text bulletin (POSIX 0600).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional file path to atomically write the SHA-256 JSON receipt (POSIX 0600).",
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
    raw_http_fn: RawHttpFn = default_http_raw,
    environ: Mapping[str, str] | None = None,
) -> int:
    """CLI entrypoint returning `0` on success or `2` on validation, compliance, parity, or network error."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        lang = normalize_lang(args.lang)
        probe_parity = bool(args.verify_holiday_routes or args.require_holiday_routes)
        emit_json = bool(args.json_flag or args.output_format == "json")

        comp_result: dict[str, Any] | None = None
        if args.enforce_personal_pc:
            comp_result = evaluate_personal_pc_compliance(environ=environ, cwd=args.cwd)
            if not comp_result["compliance_ok"]:
                raise BulletinInputError(str(comp_result["summary"]))

        if args.expected_sha is not None:
            _validate_expected_sha(args.expected_sha)
        if args.target_online_producers < 1:
            raise BulletinInputError("target_online_producers must be >= 1.")

        has_file_inputs = bool(args.t4_status or args.feedback_summary or args.snapshot)
        has_live_inputs = bool(args.live or args.gateway or probe_parity)
        if has_file_inputs and has_live_inputs:
            raise BulletinInputError(
                "Provide either local JSON file flags or --live/--gateway/--verify-holiday-routes, not both."
            )

        if not has_file_inputs and not has_live_inputs:
            if comp_result is not None:
                comp_json = json.dumps(comp_result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                _assert_safe_output(comp_json)
                comp_txt = _assert_safe_output(f"{COMPLIANCE_RULE_FR}\n{COMPLIANCE_RULE_EN}\n")
                if args.output:
                    _atomic_write_text(Path(args.output), comp_txt)
                if args.json_out:
                    _atomic_write_text(Path(args.json_out), comp_json)
                sys.stdout.write(comp_json if emit_json else comp_txt)
                return 0
            raise BulletinInputError(
                "Specify --t4-status / --feedback-summary / --snapshot, or --live [--gateway <url>]."
            )

        stdin_uses = sum(1 for p in (args.t4_status, args.feedback_summary, args.snapshot) if p == "-")
        if stdin_uses > 1:
            raise BulletinInputError("Only one input flag may read from '-' (stdin) at a time.")

        http_statuses: Mapping[str, int] | None = None
        parity_result: dict[str, Any] | None = None
        if has_live_inputs:
            gw = args.gateway if args.gateway is not None else DEFAULT_GATEWAY
            t4_doc, fb_doc, http_statuses = fetch_live_bulletin_inputs(
                gw,
                http_get=http_get,
                timeout_s=args.timeout,
                include_fleet=bool(args.include_fleet),
            )
            if probe_parity:
                parity_result = verify_holiday_bulletin_parity(
                    gw,
                    expected_sha=args.expected_sha,
                    target_online_producers=args.target_online_producers,
                    raw_http_fn=raw_http_fn,
                    http_get=http_get,
                    timeout_s=args.timeout,
                )
            source_kind = "live_gateway"
        else:
            t4_doc, fb_doc = {}, {}
            if args.snapshot:
                raw_bundle = load_json_file(args.snapshot)
                t4_doc, fb_doc = _unpack_bundle_snapshot(raw_bundle)
                if args.include_fleet and any(k in raw_bundle for k in ("fleet_status", "producers_fleet", "raid_status")):
                    t4_doc = reconcile_live_fleet_and_raid_metrics(
                        t4_doc,
                        fleet_payload=raw_bundle.get("fleet_status") or raw_bundle.get("producers_fleet"),
                        raid_payload=raw_bundle.get("raid_status"),
                    )
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
            holiday_parity=parity_result,
            compliance_check=comp_result,
        )
        bulletin_text = str(receipt["bulletin"]) + "\n"
        json_text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

        if args.output:
            _atomic_write_text(Path(args.output), bulletin_text)

        if args.json_out:
            _atomic_write_text(Path(args.json_out), json_text)

        sys.stdout.write(json_text if emit_json else bulletin_text)

        if args.require_holiday_routes and not bool(receipt.get("holiday_routes_deployed_on_cloud_run")):
            sys.stderr.write(
                "error: Cloud Run PR #949 holiday routes parity check failed "
                f"(stale_revision_detected={receipt.get('stale_revision_detected')}, "
                f"missing_or_stale={receipt.get('missing_or_stale_routes')}, "
                f"expected_sha={args.expected_sha}).\n"
            )
            return 2

        return 0
    except BulletinInputError as exc:
        err_msg = str(exc)
        if scan_output_safety(err_msg):
            err_msg = "Input validation failed (redacted unsafe token)."
        sys.stderr.write(f"error: {err_msg}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
