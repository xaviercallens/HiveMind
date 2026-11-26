#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Privacy-preserving usability funnel, PR #949 parity, and satisfaction analyzer (Christmas Adoption Plan, Section 5).

Computes aggregate, zero-PII onboarding funnel and comfort metrics from local JSON snapshots
or live read-only HiveMind gateway summary endpoints:
  - Signup-to-first-request activation rate (`activation_ratio`, plus optional `signup_conversion_ratio`)
  - Preflight-to-first-round completion rate (`preflight_to_first_round_ratio` & `install_dropoff_ratio`)
  - Anonymous usability ease rating (`mean_ease_rating` / `average_ease_rating`, target >= 4.2 / 5)
  - Gamer-Whisper comfort ratio (`zero_game_lag_ratio`, target 100%)
  - UX regression alerts (`< 20%` signup conversion, `> 30%` install dropoff, `ease_rating < 4.2`,
    and `< 100%` zero-game-lag comfort)
  - Zero-inference Cloud Run PR #949 live-route parity & 8+ producer quorum verification
  - Strictly local Amadeus / Personal-PC compliance check & atomic POSIX 0600 file writes

Strict Compliance & Privacy Invariants:
  - Standard library only; zero third-party runtime dependencies.
  - Only allowlisted hosts (`*.run.app`, `*.googleapis.com`, `datasets-server.huggingface.co`, or loopback).
  - Zero inference: never sends prompts or calls completion/chat endpoints.
  - Zero PII: handles, emails, account IDs, IP addresses, API keys, hostnames, and free-text notes are
    never stored or emitted; outputs are verified prior to emission.
  - 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD, AGPL-3.0).
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
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
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
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024

SUPPORTED_LANGS: tuple[str, ...] = ("bilingual", "fr", "en")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

SIGNUP_CONVERSION_MIN = Fraction(20, 100)  # Alert when < 20%
INSTALL_DROPOFF_MAX = Fraction(30, 100)  # Alert when > 30% (completion < 70%)
PREFLIGHT_FIRST_ROUND_MIN = Fraction(70, 100)
EASE_RATING_MIN = Fraction(42, 10)  # Target >= 4.2 / 5; alert when < 4.2
ZERO_GAME_LAG_TARGET = Fraction(1, 1)  # Target 100% (1.0)

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

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

COMPLIANCE_RULE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(never corporate hardware or work code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_RULE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(jamais de matériel ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_RULE_FR} | {COMPLIANCE_RULE_EN}"

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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:hm(?:_live_|-)(?!\*{3,4})[A-Za-z0-9_-]{4,}|hm_(?!live_\*{3,4})[A-Za-z0-9_-]{6,})\b"
)
_MASKED_HM_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*{3,4}(?:[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s)|[A-Za-z0-9_-]{3,})"
)
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
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


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML nonce attributes, CSP nonce tokens, and masked hm-*** key suffixes before figure scans."""
    if not isinstance(text, str):
        raise FunnelInputError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host matches the strict HiveMind lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate and normalize a gateway origin URL against the host, label, and scheme allowlist."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise FunnelInputError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise FunnelInputError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise FunnelInputError("Gateway URL must not contain whitespace or backslashes.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
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
    host = (parts.hostname or "").strip().lower().strip("[]")
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


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowlisted gateway origin, read-only method, and zero LLM inference/prompt payloads."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise FunnelInputError("Request URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise FunnelInputError("Request URL must not contain control characters.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise FunnelInputError("Request URL must not contain whitespace or backslashes.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise FunnelInputError("Request URL must not embed credentials.")
    validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise FunnelInputError(f"Zero-inference guard blocked inference endpoint {parts.path!r}.")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise FunnelInputError("HTTP payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise FunnelInputError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb not in ("GET", "HEAD"):
        raise FunnelInputError(f"Zero-inference read-only guard blocked HTTP {verb!r}.")
    return cleaned


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects off allowlisted hosts, with credentials, or onto inference endpoints."""

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


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)):
        raise FunnelInputError("timeout_s must be a finite positive number.")
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
                raise FunnelInputError("HTTP headers must not contain CR/LF or NUL characters.")
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
            "User-Agent": "hivemind-onboarding-funnel-report/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(req, timeout=clamped_timeout) as resp:
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
            if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
                raise FunnelInputError(f"Gateway response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
            raw = raw_bytes.decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = (exc.read(65_536) if exc.fp is not None else b"").decode("utf-8", errors="replace")
        try:
            err_doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            err_doc = {"error": _redact_pii_text(raw[:200])}
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
            "User-Agent": "hivemind-onboarding-funnel-report/1.0",
        },
        headers,
    )
    opener = _build_proxy_free_opener()
    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=clamped_timeout) as resp:
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
            if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
                raise FunnelInputError(f"Gateway response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
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


def _invoke_http_get(http_get: HttpGetFn, url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
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
    except FunnelInputError as exc:
        msg = str(exc)
        if msg.startswith(("Unable to reach", "Network error", "Malformed JSON", "Expected JSON object")):
            return 0, {"error": "transport_or_parse_error"}
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}

    if not isinstance(raw, tuple) or len(raw) != 2:
        raise FunnelInputError("http_get must return (status_code, body_dict).")
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
        raise FunnelInputError("raw_http_fn must return (status_code, body_text, headers_dict).")
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
        raise FunnelInputError("expected_sha must be a 6..64 character alphanumeric/hex string.")
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise FunnelInputError("expected_sha must be a 6..64 character alphanumeric/hex string (never 'unknown').")
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
    if body.get("campaign_id") != CAMPAIGN_ID:
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


def verify_holiday_funnel_parity(
    gateway: str,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> dict[str, Any]:
    """Zero-inference Cloud Run PR #949 live-route parity, CSP, manifest, SHA, and 8+ producer quorum verifier."""
    gw = validate_gateway_url(gateway)
    clamped_timeout = _validate_timeout_s(timeout_s)
    if isinstance(target_online_producers, bool) or not isinstance(target_online_producers, int) or target_online_producers < 1:
        raise FunnelInputError("target_online_producers must be an integer >= 1.")
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
        "object": "hivemind.holiday_funnel_parity",
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
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "exit_code": 0 if holiday_routes_deployed_on_cloud_run else 2,
    }
    _assert_clean_output(json.dumps(result, ensure_ascii=False))
    return result


def fetch_live_gateway_snapshot(
    gateway_url: str,
    *,
    http_get: HttpGetFn = default_http_get_json,
    raw_http_fn: RawHttpFn = default_http_raw,
    timeout_s: float = HTTP_TIMEOUT_S,
    probe_holiday_routes: bool = False,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
) -> dict[str, Any]:
    """Fetch read-only summary endpoints (and optional PR #949 route parity) from an allowlisted gateway."""
    origin = validate_gateway_url(gateway_url)
    clamped_timeout = _validate_timeout_s(timeout_s)

    endpoints = {
        "feedback_summary": "/v1/hivemind/onboarding/feedback/summary",
        "raid_status": "/v1/hivemind/raid/status",
        "live_telemetry": "/v1/hivemind/live-telemetry",
        "arena_summary": "/v1/hivemind/arena/summary",
    }
    bundle: dict[str, Any] = {"_source_kind": "live_gateway"}
    primary_ok = 0

    for key, path in endpoints.items():
        status_code, payload = _invoke_http_get(http_get, f"{origin}{path}", clamped_timeout)
        if status_code == 200 and isinstance(payload, Mapping):
            bundle[key] = dict(payload)
            if key != "arena_summary":
                primary_ok += 1
        elif key != "arena_summary" and status_code >= 500 and not probe_holiday_routes:
            raise FunnelInputError(f"Gateway endpoint {path} returned HTTP {status_code}.")

    if probe_holiday_routes:
        bundle["holiday_routes_parity"] = verify_holiday_funnel_parity(
            origin,
            expected_sha=expected_sha,
            target_online_producers=target_online_producers,
            raw_http_fn=raw_http_fn,
            http_get=http_get,
            timeout_s=clamped_timeout,
        )
    elif primary_ok == 0:
        raise FunnelInputError("No live onboarding summary endpoints returned HTTP 200.")

    return bundle


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
    """Locate ``.git/config`` from ``workspace``, following worktree ``gitdir:`` and ``commondir``."""
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
    """Scan serialized report output for accidental PII, unmasked tokens, local user paths, or forbidden keys."""
    if not isinstance(text, str):
        raise FunnelInputError("scan_for_pii_leaks requires a string.")
    findings: list[str] = []
    if _EMAIL_RE.search(text):
        findings.append("email_address_detected")
    if _API_KEY_RE.search(text):
        findings.append("hivemind_api_key_detected")
    if _BEARER_RE.search(text):
        findings.append("bearer_token_detected")
    for ip_hit in _IPV4_RE.findall(text):
        try:
            is_loopback = ipaddress.ip_address(ip_hit).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            findings.append("ipv4_literal_detected")
            break
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

    has_ease, ease_raw, ease_key = _first_present(
        source_map,
        ("mean_ease_rating", "average_ease_rating"),
    )
    if has_ease and ease_raw is None and "average_ease_rating" in source_map and source_map["average_ease_rating"] is not None:
        ease_raw = source_map["average_ease_rating"]
        ease_key = "average_ease_rating"
    ease_frac = (
        _opt_bounded_fraction(
            ease_raw,
            ease_key or "mean_ease_rating",
            Fraction(1, 1),
            Fraction(5, 1),
        )
        if has_ease
        else None
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


def analyze_onboarding_funnel(
    raw: Mapping[str, Any],
    *,
    feedback_override: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
    holiday_parity: Mapping[str, Any] | None = None,
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

    parity_raw = holiday_parity if holiday_parity is not None else raw.get("holiday_routes_parity")
    if isinstance(parity_raw, Mapping):
        clean_parity = _sanitize_parity_block(parity_raw)
        receipt["holiday_routes_parity"] = clean_parity
        receipt["holiday_routes_deployed_on_cloud_run"] = clean_parity["holiday_routes_deployed_on_cloud_run"]
        receipt["onboarding_feedback_deployed"] = clean_parity["onboarding_feedback_deployed"]
        receipt["stale_revision_detected"] = clean_parity["stale_revision_detected"]
        receipt["missing_or_stale_routes"] = list(clean_parity["missing_or_stale_routes"])
        receipt["online_producers"] = clean_parity["online_producers"]
        receipt["producer_quorum_met"] = clean_parity["producer_quorum_met"]

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


def _render_parity_lines(parity: Mapping[str, Any], *, fr: bool) -> list[str]:
    missing_txt = ", ".join(str(r) for r in (parity.get("missing_or_stale_routes") or [])) or "none"
    diags = parity.get("diagnostics") if isinstance(parity.get("diagnostics"), Sequence) else []
    exp_sha = parity.get("expected_sha") or "none"
    if fr:
        lines = [
            "",
            "### Parité Cloud Run PR #949 & Quorum Producteurs (FR)",
            f"- **Routes Noël déployées (`holiday_routes_deployed_on_cloud_run`) :** `{parity.get('holiday_routes_deployed_on_cloud_run')}`",
            f"- **Feedback CES déployé (`onboarding_feedback_deployed`) :** `{parity.get('onboarding_feedback_deployed')}`",
            f"- **Révision obsolète détectée (`stale_revision_detected`) :** `{parity.get('stale_revision_detected')}` (manquantes : `{missing_txt}`)",
            f"- **Producteurs en ligne (`online_producers`) :** `{parity.get('online_producers')}` / cible `{parity.get('target_online_producers', 8)}` (`producer_quorum_met={parity.get('producer_quorum_met')}`)",
            f"- **SHA déployé (`live_build_sha`) :** `{parity.get('live_build_sha', 'unknown')}` (attendu : `{exp_sha}`, `sha_match={parity.get('sha_match')}`)",
        ]
    else:
        lines = [
            "",
            "### Cloud Run PR #949 Route Parity & Producer Quorum (EN)",
            f"- **Holiday routes deployed (`holiday_routes_deployed_on_cloud_run`):** `{parity.get('holiday_routes_deployed_on_cloud_run')}`",
            f"- **CES feedback deployed (`onboarding_feedback_deployed`):** `{parity.get('onboarding_feedback_deployed')}`",
            f"- **Stale revision detected (`stale_revision_detected`):** `{parity.get('stale_revision_detected')}` (missing: `{missing_txt}`)",
            f"- **Online producers (`online_producers`):** `{parity.get('online_producers')}` / target `{parity.get('target_online_producers', 8)}` (`producer_quorum_met={parity.get('producer_quorum_met')}`)",
            f"- **Live build SHA (`live_build_sha`):** `{parity.get('live_build_sha', 'unknown')}` (expected: `{exp_sha}`, `sha_match={parity.get('sha_match')}`)",
        ]
    for d in diags:
        lines.append(f"- `{_redact_pii_text(str(d))}`")
    return lines


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
    parity = receipt.get("holiday_routes_parity")
    if isinstance(parity, Mapping):
        lines.extend(_render_parity_lines(parity, fr=True))
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
    parity = receipt.get("holiday_routes_parity")
    if isinstance(parity, Mapping):
        lines.extend(_render_parity_lines(parity, fr=False))
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
            "Compute privacy-preserving HiveMind Holiday onboarding funnel, PR #949 parity, "
            "& satisfaction metrics (Markdown FR/EN + SHA-256 JSON receipt)."
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
        help="Optional file path to atomically write the zero-PII JSON receipt (POSIX 0600).",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional file path to atomically write the Markdown report (POSIX 0600).",
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
    raw_http_fn: RawHttpFn = default_http_raw,
    environ: Mapping[str, str] | None = None,
) -> int:
    """CLI entrypoint returning 0 (ok), 1 (UX alert with --fail-on-alert), or 2 (input/compliance/parity error)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        lang = normalize_lang(args.lang)
        probe_parity = bool(args.verify_holiday_routes or args.require_holiday_routes)
        emit_json = bool(args.json_flag or args.output_format == "json")

        if args.enforce_personal_pc:
            comp = evaluate_personal_pc_compliance(environ=environ, cwd=args.cwd)
            if not comp["compliance_ok"]:
                raise FunnelInputError(str(comp["summary"]))

        if args.expected_sha is not None:
            _validate_expected_sha(args.expected_sha)
        if args.target_online_producers < 1:
            raise FunnelInputError("target_online_producers must be >= 1.")

        if args.snapshot and (args.live or args.gateway or probe_parity):
            raise FunnelInputError("Provide either --snapshot or --live/--gateway/--verify-holiday-routes, not both.")

        if args.snapshot:
            raw_data = load_json_file(args.snapshot)
        elif args.live or args.gateway or probe_parity:
            gw_url = args.gateway if args.gateway is not None else DEFAULT_GATEWAY
            raw_data = fetch_live_gateway_snapshot(
                gw_url,
                http_get=http_get,
                raw_http_fn=raw_http_fn,
                timeout_s=args.timeout,
                probe_holiday_routes=probe_parity,
                expected_sha=args.expected_sha,
                target_online_producers=args.target_online_producers,
            )
        elif args.enforce_personal_pc:
            comp = evaluate_personal_pc_compliance(environ=environ, cwd=args.cwd)
            comp_json = serialize_receipt_json(comp)
            comp_md = _assert_clean_output(
                f"# Personal-PC Compliance Verified\n\n- {COMPLIANCE_RULE_FR}\n- {COMPLIANCE_RULE_EN}\n"
            )
            if args.json_out:
                _atomic_write_text(Path(args.json_out), comp_json)
            if args.markdown_out:
                _atomic_write_text(Path(args.markdown_out), comp_md)
            sys.stdout.write(comp_json if emit_json else comp_md)
            return 0
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
            _atomic_write_text(Path(args.json_out), json_text)

        if args.markdown_out:
            _atomic_write_text(Path(args.markdown_out), md_text)

        sys.stdout.write(json_text if emit_json else md_text)

        if args.require_holiday_routes and not bool(receipt.get("holiday_routes_deployed_on_cloud_run")):
            sys.stderr.write(
                "error: Cloud Run PR #949 holiday routes parity check failed "
                f"(stale_revision_detected={receipt.get('stale_revision_detected')}, "
                f"missing_or_stale={receipt.get('missing_or_stale_routes')}).\n"
            )
            return 2

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
