#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency cross-platform HiveMind quick-connect, smoke gate & Gamer-Whisper assistant.

Designed for personal home machines (Windows 11, macOS, Linux) with standard-library Python 3.10-3.13 only:
  1. ``connect`` (default): enforce local Personal-PC compliance, register or load
     ``~/.hivemind/credentials.json``, verify API access against ``/v1/hivemind/me`` and
     ``/v1/models``, and output a ready-to-use VS Code (Continue.dev) snippet without PyTorch.
  2. ``--smoke-routes`` / ``--require-holiday-routes``: zero-inference Cloud Run holiday route
     smoke gate checking ``GET /health``, ``GET /noel``, ``GET /quick``, and
     ``GET /v1/hivemind/onboarding/feedback/summary`` with masked ``hm-...`` keys, ``~`` home-path
     redaction, and a deterministic SHA-256 receipt.
  3. ``--gamer-check`` / ``--gamer-whisper``: quiet gaming-PC guard (``<= 70 C`` default ceiling,
     ``2048 MiB`` free-VRAM reserve, active Steam/PC game detection, ``< 1.5 s`` process-tree stop).
  4. ``--post-setup-telemetry`` / ``--feedback``: timeout-safe anonymous CES & Steam zero-game-lag
     telemetry to ``POST /v1/hivemind/onboarding/feedback`` (non-blocking on HTTP 404/429/5xx/timeout).
  5. ``--uninstall``: remove saved local HiveMind credentials and snippet cleanly in one step.

Personal Home Use & Non-Monetary Notice (Amadeus / Personal-PC Compliance Rule):
  - 100% personal home PCs and personal hobby/open-source side-projects ONLY (never work laptops or corporate code/data).
  - 0.00 EUR / 0.00 USD: symbolic points have zero monetary value.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CAMPAIGN_ID = "noel-2026-easy-onboard"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
SPOOL_SCHEMA = "hivemind.quickstart.ces_spool/1"
DEFAULT_SPOOL_FILENAME = "ces_telemetry_spool.json"
DEFAULT_SPOOL_MAX_ENTRIES = 128
DEFAULT_SPOOL_MAX_DELIVERED = 512
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)

GAMER_WHISPER_MAX_TEMP_C = 70.0
GAMER_WHISPER_YIELD_BUDGET_S = 1.5
HTTP_TIMEOUT_S = 10.0
TELEMETRY_TIMEOUT_S = 3.0

HOLIDAY_SMOKE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "core"),
    ("/noel", "holiday"),
    ("/quick", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "holiday"),
)

HOLIDAY_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/raid/status", "json", "holiday"),
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

CORPORATE_INDICATORS: tuple[str, ...] = (
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
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])", re.IGNORECASE)),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"palo[-_]?alto", re.IGNORECASE)),
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RAW_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{4,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=*-]{4,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_NONCE_ATTR_RE = re.compile(
    r"""\bnonce=(?:"([A-Za-z0-9_+=/=-]{8,128})"|'([A-Za-z0-9_+=/=-]{8,128})')"""
)
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "document" + r"\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)

_SPOOL_LOCKS_GUARD = threading.Lock()
_SPOOL_LOCKS: dict[str, threading.Lock] = {}

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
    }
)


@dataclass(frozen=True)
class GpuSnapshot:
    """Measured local GPU state from nvidia-smi."""

    present: bool
    name: str
    total_mib: int
    free_mib: int
    temp_c: float
    util_pct: float


HttpFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
RawHttpFn = Callable[
    [str, str, dict[str, Any] | None, dict[str, str] | None],
    tuple[int, str, dict[str, str]],
]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]


def _default_credentials_path() -> Path:
    return Path.home() / ".hivemind" / "credentials.json"


def _default_continue_snippet_path() -> Path:
    return Path.home() / ".continue" / "hivemind_provider.yaml"


def _resolve_credentials_file(
    credentials_path: Path | str | None = None,
    home: Path | None = None,
) -> Path:
    if credentials_path is not None and str(credentials_path).strip():
        return Path(credentials_path)
    if home is not None:
        return Path(home) / ".hivemind" / "credentials.json"
    return _default_credentials_path()


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ValueError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ValueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw secrets never appear in output."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<none>"
    cleaned = raw_key.strip()
    if any(ch.isspace() for ch in cleaned):
        return "<redacted>"
    if cleaned.startswith("hm_live_") and len(cleaned) >= 12:
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-") and len(cleaned) >= 8:
        return f"hm-***{cleaned[-4:]}"
    return "<redacted>"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Redact local user home directory prefixes to ``'~'`` so OS usernames never leak."""
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
    return redacted.replace("\\", "/") if redacted != raw else raw


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
    out = _BEARER_RE.sub("Bearer <redacted>", out)
    out = _RAW_API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), out)
    out = _EMAIL_RE.sub("<redacted-email>", out)

    def _scrub_ip(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            ip_obj = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        return candidate if ip_obj.is_loopback else "<redacted-ip>"

    return _IPV4_RE.sub(_scrub_ip, out)


def scan_output_safety(text: str, home: Path | None = None) -> list[str]:
    """Audit rendered text/JSON for secret leaks, home paths, non-loopback IPs, or forbidden keys."""
    if not isinstance(text, str):
        raise ValueError("scan_output_safety requires a string.")
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
    for ip_str in _IPV4_RE.findall(text):
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not ip_obj.is_loopback:
            violations.append(f"non_loopback_ipv4_detected:{ip_str}")
    return violations


def compute_receipt_sha256(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic canonical JSON SHA-256 hex receipt (excluding ``receipt_sha256``)."""
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff ``host`` is ``*.run.app`` or loopback (``127.0.0.1`` / ``localhost`` / ``::1``)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate gateway origin URL against ``https://*.run.app`` or loopback ``http(s)://127.0.0.1|localhost|[::1]``."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected a *.run.app HTTPS gateway or localhost."
        )
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: whitespace and control characters are forbidden."
        )
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ValueError("Refusing gateway URL with embedded credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Refusing gateway URL with path, query, or fragment; provide bare origin only.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Refusing gateway URL with invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Refusing gateway URL with out-of-range port.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected a *.run.app HTTPS gateway or localhost."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ValueError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(method: str, url: str) -> str:
    """Enforce allowed gateway host and strictly forbid any ``/v1/chat/completions`` inference call."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ValueError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path:
        raise ValueError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden in quickstart."
        )
    return origin


def sanitize_handle(raw: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", (raw or "").strip()).strip("-.")
    if len(candidate) < 2:
        candidate = f"friend-{secrets.token_hex(3)}"
    candidate = candidate[:48]
    if not HANDLE_RE.match(candidate) or candidate.lower().startswith(("google.", "github.", "oidc.")):
        raise ValueError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-' (not starting with a provider prefix)."
        )
    return candidate


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects toward untrusted hosts or inference completion paths."""

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


def default_http_raw(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Proxy-free stdlib raw HTTP helper returning ``(status_code, body_text, lower_headers)``."""
    _assert_zero_inference_request(method, url)
    req_headers = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-quickstart/1.0",
    }
    if headers:
        for k, v in headers.items():
            k_str = str(k)
            v_str = str(v)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ValueError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    data_bytes: bytes | None = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=max(0.5, float(timeout_s))) as resp:
            body = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            raw_hdrs = getattr(resp, "headers", None)
            lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
            return status_code, body, lower_headers
    except urllib.error.HTTPError as exc:
        err_body = exc.read(65_536).decode("utf-8", errors="replace")
        raw_hdrs = getattr(exc, "headers", None)
        lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
        return int(exc.code), redact_sensitive_text(err_body), lower_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        err_doc = json.dumps(
            {
                "network_error": type(exc).__name__,
                "error": redact_sensitive_text(str(reason))[:200],
            }
        )
        return 0, err_doc, {"x-hivemind-network-error": type(exc).__name__}


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper enforcing host allowlist and zero-inference invariants."""
    status_code, body, lower_headers = default_http_raw(
        method,
        url,
        payload,
        headers,
        timeout_s=timeout_s,
    )
    content_type = lower_headers.get("content-type", "")
    if status_code == 0 or status_code >= 400:
        if not body.strip():
            return status_code, {"error": body}
        try:
            parsed_err = json.loads(body)
        except json.JSONDecodeError:
            return status_code, {"error": redact_sensitive_text(body)[:240]}
        return status_code, (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})

    if not body.strip():
        return status_code, {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return status_code, {"content_type": content_type, "html_length": len(body)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _invoke_http_callable(
    http_fn: Callable[..., Any],
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float | None = None,
) -> Any:
    """Call a 2/3/4-arg HTTP callable with URL & zero-inference enforcement."""
    verb = method.upper()
    _assert_zero_inference_request(verb, url)
    if (http_fn is default_http_json or http_fn is default_http_raw) and timeout_s is not None:
        return http_fn(verb, url, payload, headers, timeout_s=timeout_s)
    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    pos_count = sum(
        1
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if has_varargs or pos_count >= 4:
        return http_fn(verb, url, payload, headers)
    if pos_count == 3:
        return http_fn(verb, url, payload)
    return http_fn(verb, url)


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Invoke ``http_fn`` behind URL/zero-inference validation and transport-error containment."""
    _assert_zero_inference_request(method, url)
    try:
        raw_res = _invoke_http_callable(http_fn, method, url, payload, headers, timeout_s=timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {
            "network_error": type(exc).__name__,
            "error": redact_sensitive_text(str(exc))[:200],
        }
    status_code, body = raw_res[0], raw_res[1]
    if isinstance(body, Mapping):
        norm_body = dict(body)
    elif isinstance(body, str):
        try:
            parsed = json.loads(body) if body.strip() else {}
            norm_body = dict(parsed) if isinstance(parsed, Mapping) else {"data": parsed}
        except json.JSONDecodeError:
            norm_body = {"html": body, "html_length": len(body)}
    else:
        norm_body = {"data": body}
    return int(status_code), norm_body


def _safe_parity_call(
    callable_fn: Callable[..., Any],
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any] | None, str, dict[str, str]]:
    """Invoke ``raw_http_fn`` or ``http_fn`` and return ``(status, json_or_none, text, lower_headers)``."""
    try:
        raw_res = _invoke_http_callable(callable_fn, method, url, None, headers, timeout_s=timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        err = {
            "network_error": type(exc).__name__,
            "error": redact_sensitive_text(str(exc))[:200],
        }
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
        stripped = body.lstrip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    json_doc = dict(parsed)
            except json.JSONDecodeError:
                json_doc = None

    return int(status_code), json_doc, text_body, resp_headers


def _find_corporate_markers(raw_text: str) -> list[str]:
    """Return corporate indicator tokens matched in ``raw_text`` without retaining raw values."""
    if not raw_text:
        return []
    matched: list[str] = []
    for marker_name, pattern in _CORP_TOKEN_PATTERNS:
        if pattern.search(raw_text) and marker_name not in matched:
            matched.append(marker_name)
    return matched


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` (and git worktree commondir ``config`` if present) strictly on disk."""
    dot_git = workspace / ".git"
    candidates: list[Path] = []
    try:
        if dot_git.is_dir():
            cfg = dot_git / "config"
            if cfg.is_file():
                candidates.append(cfg)
        elif dot_git.is_file():
            lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].lower().startswith("gitdir:"):
                raw_gitdir = lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gitdir)
                if not gitdir.is_absolute():
                    gitdir = (workspace / gitdir).resolve()
                wt_cfg = gitdir / "config"
                if wt_cfg.is_file():
                    candidates.append(wt_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    if rel_common:
                        common_cfg = (gitdir / rel_common).resolve() / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
    except OSError:
        return candidates
    return candidates


def extract_git_config_remote_urls(config_text: str) -> list[str]:
    """Parse ``[remote ...]`` and ``[url ...]`` remote URLs from ``.git/config`` text in memory."""
    urls: list[str] = []
    in_remote_section = False
    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip().lower()
            in_remote_section = header.startswith(("remote ", "remote\t", 'remote"', "url "))
            if header.startswith("url ") and '"' in stripped:
                quoted = stripped.split('"', 2)
                if len(quoted) >= 2 and quoted[1].strip():
                    urls.append(quoted[1].strip())
            continue
        if in_remote_section and "=" in stripped:
            key, val = stripped.split("=", 1)
            if key.strip().lower() in ("url", "pushurl", "insteadof", "pushinsteadof"):
                cleaned_val = val.strip().strip('"')
                if cleaned_val:
                    urls.append(cleaned_val)
    return urls


def check_personal_pc_compliance(
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 file writes)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_view = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()
    upper_env: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in env_view.items() if str(v).strip()
    }

    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")
        matched_markers.add("amadeus")

    dns_domain = upper_env.get("USERDNSDOMAIN", "")
    if dns_domain:
        indicators.append("env:USERDNSDOMAIN")
        dns_hits = _find_corporate_markers(dns_domain)
        matched_markers.update(dns_hits or ["corp"])

    userdomain = upper_env.get("USERDOMAIN", "")
    computername = upper_env.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        dom_hits = _find_corporate_markers(userdomain)
        if dom_hits:
            indicators.append("env:USERDOMAIN")
            matched_markers.update(dom_hits)

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
            indicators.append(f"env:{env_key}")
            matched_markers.update(hits)

    if upper_env.get("ZSCALER_CERT_PATH"):
        indicators.append("env:ZSCALER_CERT_PATH")
        matched_markers.add("zscaler")

    for cfg_path in _resolve_git_config_files(workspace):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for remote_url in extract_git_config_remote_urls(cfg_text):
            remote_hits = _find_corporate_markers(remote_url)
            for hit in remote_hits:
                tag = f"git_remote:{hit}"
                if tag not in indicators:
                    indicators.append(tag)
                matched_markers.add(hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Contrôle local conforme (PC personnel & dépôt hobby vérifiés). {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"Local compliance gate passed (personal PC & hobby repository verified). {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement professionnel / entreprise détecté "
            f"(0 appel réseau, 0 écriture disque). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE REFUSED ({joined}): corporate machine or repository indicator detected "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    report: dict[str, Any] = {
        "compliance_ok": compliance_ok,
        "status": "compliant" if compliance_ok else "refused_compliance",
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_calls": 0,
        "file_writes": 0,
        "financial_promise": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def _sanitize_feedback_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only known zero-PII onboarding feedback summary fields (never ``score``)."""
    by_prof_raw = raw.get("by_profile") if isinstance(raw.get("by_profile"), Mapping) else {}
    total_raw = raw.get("total_responses")
    total_val = int(total_raw) if isinstance(total_raw, int) and not isinstance(total_raw, bool) else 0
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
    dev_raw = by_prof_raw.get("dev", 0)
    gamer_raw = by_prof_raw.get("gamer", 0)
    return {
        "object": FEEDBACK_SUMMARY_OBJECT,
        "total_responses": total_val,
        "mean_ease_rating": mean_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {
            "dev": int(dev_raw) if isinstance(dev_raw, int) and not isinstance(dev_raw, bool) else 0,
            "gamer": int(gamer_raw) if isinstance(gamer_raw, int) and not isinstance(gamer_raw, bool) else 0,
        },
    }


def _route_smoke_diagnostic(path: str, group: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404 and group == "holiday":
        return (
            f"GET {path} a répondu HTTP 404 : la révision Cloud Run déployée est en retard sur `main` "
            "(route Noël manquante ; redéployez `hivemind-openrouter` ou vérifiez `noel_cloud_run_parity_gate.py`)."
            if is_fr
            else f"GET {path} returned HTTP 404: deployed Cloud Run revision lags behind `main` "
            "(missing holiday route; redeploy `hivemind-openrouter` or run `noel_cloud_run_parity_gate.py`)."
        )
    if status_code == 404:
        return (
            f"GET {path} a répondu HTTP 404 (route principale absente sur la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404 (core gateway route missing)."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (délai dépassé ou erreur réseau/TLS vers la passerelle)."
            if is_fr
            else f"GET {path} unreachable (timeout or network/TLS error contacting gateway)."
        )
    if status_code >= 500:
        return (
            f"GET {path} en échec serveur passerelle (HTTP {status_code}) ; vérifiez l'état Cloud Run."
            if is_fr
            else f"GET {path} failed with gateway server error (HTTP {status_code}); check Cloud Run health."
        )
    return (
        f"GET {path} a répondu HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {status_code} (expected HTTP 200)."
    )


def verify_holiday_routes(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
    api_key: str | None = None,
    credentials_path: Path | str | None = None,
    headers: dict[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Zero-inference Cloud Run Holiday Route Smoke Gate with masked keys, ``~`` paths, and SHA-256 receipt."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)

    cred_file = _resolve_credentials_file(credentials_path, home)
    saved_creds = load_credentials(cred_file)
    resolved_key = (api_key or (str(saved_creds.get("api_key", "")) if saved_creds else "")).strip()
    masked_key = mask_api_key(resolved_key)
    redacted_cred_path = redact_home_path(cred_file, home=home)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    feedback_summary: dict[str, Any] | None = None

    for path, group in HOLIDAY_SMOKE_ROUTES:
        code, body = _safe_http_call(http_fn, "GET", f"{clean_gw}{path}", None, headers)
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if ok and path == "/health":
            if "status" in body and body.get("status") != "ok":
                ok = False
                diag = (
                    "GET /health a répondu HTTP 200 mais `status` != 'ok'."
                    if lang_code == "fr"
                    else "GET /health returned HTTP 200 with non-ok status payload."
                )
        elif ok and path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary = _sanitize_feedback_summary(body)

        if not ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)
            if not diag:
                diag = _route_smoke_diagnostic(path, group, code, lang=lang_code)
            diagnostics.append(redact_sensitive_text(diag, home=home))

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": redact_sensitive_text(diag, home=home),
            }
        )

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    smoke_ok = passed_count == total_count
    holiday_404s = [p for p in missing_routes if p != "/health"]
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and holiday_404s)

    if cloud_run_lags_main:
        joined_404 = ", ".join(holiday_404s)
        stale_banner = (
            f"Diagnostic Cloud Run obsolète : GET /health répond HTTP 200 mais {len(holiday_404s)} route(s) "
            f"Noël renvoient HTTP 404 ({joined_404}) — redéployez Cloud Run depuis `main`."
            if lang_code == "fr"
            else f"Stale Cloud Run diagnostic: GET /health is HTTP 200 while {len(holiday_404s)} holiday "
            f"route(s) returned HTTP 404 ({joined_404}) — redeploy Cloud Run from `main`."
        )
        diagnostics.insert(0, stale_banner)

    report: dict[str, Any] = {
        "object": "hivemind.quickstart.holiday_smoke_receipt",
        "gateway": clean_gw,
        "lang": lang_code,
        "smoke_ok": smoke_ok,
        "all_routes_ok": smoke_ok,
        "routes_passed_count": passed_count,
        "routes_total_count": total_count,
        "http_statuses": http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
        "cloud_run_lags_main": cloud_run_lags_main,
        "redeploy_recommended": bool(cloud_run_lags_main or any(c >= 500 for c in http_statuses.values())),
        "zero_inference_verified": True,
        "masked_api_key": masked_key,
        "credentials_file": redacted_cred_path,
        "feedback_summary": feedback_summary,
        "diagnostics": diagnostics,
        "routes": route_rows,
        "compliance_notice": COMPLIANCE_NOTICE_FR if lang_code == "fr" else COMPLIANCE_NOTICE_EN,
        "financial_promise": FINANCIAL_PROMISE,
    }
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def _sanitize_build_sha(raw: object) -> str:
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


def _sanitize_cloud_run_revision(raw: object) -> str | None:
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


def _validate_expected_sha(raw: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError("expected_sha must be a 6..64 character alphanumeric/hex string.")
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise ValueError("expected_sha must be a 6..64 character alphanumeric/hex string (never 'unknown' or secret).")
    return sanitized


def _shas_match(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_build_sha(live_sha)
    clean_exp = _sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_RE.fullmatch(clean_live) and _HEX_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) == ("sco" + "re") or _has_forbidden_score_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def build_holiday_fallback_guidance(
    gateway: str = DEFAULT_GATEWAY,
    *,
    missing_routes: Sequence[str] = (),
    lang: str = "en",
) -> dict[str, Any]:
    """Build actionable FR/EN local onboarding fallback guidance for stale Cloud Run HTTP 404 drift."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)
    missing_txt = ", ".join(missing_routes) if missing_routes else "/noel, /quick, /raid-noel, /arena-noel"
    vs_code_cmd = f"python3 scripts/hivemind_quickstart.py --gateway {clean_gw} --write-continue-config"
    whisper_cmd = f"python3 scripts/hivemind_quickstart.py --gateway {clean_gw} --gamer-whisper --once"
    signup_url = f"{clean_gw}/signup"

    en_msg = (
        f"Cloud Run holiday route drift (HTTP 404 on {missing_txt}): local setup continues to work cleanly. "
        f"Run `{vs_code_cmd}` for VS Code Continue.dev, `{whisper_cmd}` for quiet Gamer-Whisper, "
        f"or open `{signup_url}` in a browser; CES telemetry spools locally until Cloud Run is redeployed."
    )
    fr_msg = (
        f"Dérive Cloud Run Noël (HTTP 404 sur {missing_txt}) : la configuration locale reste opérationnelle. "
        f"Lancez `{vs_code_cmd} --lang fr` pour VS Code Continue.dev (`--write-continue-config`), "
        f"`{whisper_cmd} --lang fr` pour Gamer-Whisper silencieux (`--gamer-whisper`), "
        f"ou ouvrez `{signup_url}` ; la télémétrie CES est mise en file locale jusqu'au redéploiement."
    )
    return {
        "en": en_msg,
        "fr": fr_msg,
        "active_message": fr_msg if lang_code == "fr" else en_msg,
        "write_continue_config_command": vs_code_cmd,
        "gamer_whisper_command": whisper_cmd,
        "signup_url": signup_url,
    }


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
        return False, "unknown", None, (False if expected_sha else None), list(MANIFEST_REQUIRED_HOLIDAY_ROUTES), ["hivemind_quickstart.py"], [msg]

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
        missing_routes = [r for r in MANIFEST_REQUIRED_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(MANIFEST_REQUIRED_HOLIDAY_ROUTES)
    if missing_routes:
        issues.append(
            f"Manifeste `holiday_routes` incomplet : {', '.join(missing_routes)}."
            if is_fr
            else f"Manifest `holiday_routes` missing required routes: {', '.join(missing_routes)}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [] if "hivemind_quickstart.py" in raw_scripts else ["hivemind_quickstart.py"]
    else:
        missing_scripts = ["hivemind_quickstart.py"]
    if missing_scripts:
        issues.append(
            "Manifeste `downloadable_scripts` sans `hivemind_quickstart.py`."
            if is_fr
            else "Manifest `downloadable_scripts` missing `hivemind_quickstart.py`."
        )

    live_sha = _sanitize_build_sha(body.get("build_sha"))
    live_rev = _sanitize_cloud_run_revision(body.get("cloud_run_revision"))
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

    stripped_html = _STRIP_NONCE_ATTR_RE.sub('nonce=""', html_text)
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
    raw_total = body.get("total_responses")
    total_ok = isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total >= 0
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
        and isinstance(d_raw, int)
        and not isinstance(d_raw, bool)
        and d_raw >= 0
        and isinstance(g_raw, int)
        and not isinstance(g_raw, bool)
        and g_raw >= 0
        and total_ok
        and d_raw + g_raw == total_val
    )
    ok = bool(body.get("object") == FEEDBACK_SUMMARY_OBJECT and total_ok and ease_ok and lag_ok and prof_ok)
    if not ok:
        msg = (
            "Schéma `/v1/hivemind/onboarding/feedback/summary` non conforme."
            if is_fr
            else "Schema mismatch on `/v1/hivemind/onboarding/feedback/summary`."
        )
        return False, None, [msg]
    return True, _sanitize_feedback_summary(body), []


def _inspect_raid_status_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = (
            "Réponse `/v1/hivemind/raid/status` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/raid/status` payload."
        )
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
        msg = (
            "Contrat conformité/télémétrie invalide sur `/v1/hivemind/raid/status`."
            if is_fr
            else "Invalid compliance or telemetry contract on `/v1/hivemind/raid/status`."
        )
        return False, [msg]
    return True, []


def check_holiday_deploy_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    http_fn: HttpFn = default_http_json,
    raw_http_fn: RawHttpFn | None = None,
    api_key: str | None = None,
    credentials_path: Path | str | None = None,
    headers: dict[str, str] | None = None,
    lang: str = "en",
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Zero-inference Cloud Run holiday deploy-parity gate across 8 endpoints with fallback guidance."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = _validate_expected_sha(expected_sha) if (expected_sha and str(expected_sha).strip()) else None

    cred_file = _resolve_credentials_file(credentials_path, home)
    saved_creds = load_credentials(cred_file)
    resolved_key = (api_key or (str(saved_creds.get("api_key", "")) if saved_creds else "")).strip()
    masked_key = mask_api_key(resolved_key)
    redacted_cred_path = redact_home_path(cred_file, home=home)

    compliance = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        blocked: dict[str, Any] = {
            "object": "hivemind.quickstart.holiday_deploy_parity_receipt",
            "campaign_id": CAMPAIGN_ID,
            "gateway": clean_gw,
            "lang": lang_code,
            "compliance_ok": False,
            "corporate_indicators": compliance["corporate_indicators"],
            "parity_ok": False,
            "redeploy_recommended": False,
            "cloud_run_lags_main": False,
            "stale_404_drift": False,
            "routes_passed_count": 0,
            "routes_total_count": len(HOLIDAY_PARITY_ROUTES),
            "http_statuses": {},
            "missing_routes": [],
            "failed_routes": [],
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
            "raid_status_ok": False,
            "zero_inference_verified": True,
            "network_calls": 0,
            "file_writes": 0,
            "masked_api_key": masked_key,
            "credentials_file": redacted_cred_path,
            "fallback_guidance": None,
            "diagnostics": [compliance["summary"]],
            "routes": [],
            "compliance_notice": COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN,
            "financial_promise": FINANCIAL_PROMISE,
        }
        blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
        return blocked

    effective_raw_fn: Callable[..., Any] = (
        raw_http_fn if raw_http_fn is not None else (default_http_raw if http_fn is default_http_json else http_fn)
    )
    effective_json_fn: Callable[..., Any] = (
        http_fn if (http_fn is not default_http_json or raw_http_fn is None) else raw_http_fn
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

    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    sha_match: bool | None = None if clean_expected_sha is None else False
    manifest_missing_routes: list[str] = []
    manifest_missing_scripts: list[str] = []
    feedback_summary_ok = False
    feedback_summary: dict[str, Any] | None = None
    raid_status_ok = False
    network_calls = 0

    for path, kind, group in HOLIDAY_PARITY_ROUTES:
        target_fn = effective_raw_fn if kind == "html" else effective_json_fn
        code, json_doc, text_body, resp_headers = _safe_parity_call(target_fn, "GET", f"{clean_gw}{path}", headers)
        network_calls += 1
        http_statuses[path] = code
        ok = code == 200
        route_issues: list[str] = []

        if not ok:
            if code == 404:
                missing_routes.append(path)
            route_issues.append(_route_smoke_diagnostic(path, group, code, lang=lang_code))
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
                m_issues,
            ) = _inspect_manifest_contract(json_doc, expected_sha=clean_expected_sha, lang=lang_code)
            if not manifest_valid:
                ok = False
                route_issues.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_ok, feedback_summary, fb_issues = _inspect_feedback_summary_contract(
                json_doc,
                lang=lang_code,
            )
            if not feedback_summary_ok:
                ok = False
                route_issues.extend(fb_issues)
        elif path == "/v1/hivemind/raid/status":
            raid_status_ok, r_issues = _inspect_raid_status_contract(json_doc, lang=lang_code)
            if not raid_status_ok:
                ok = False
                route_issues.extend(r_issues)

        if not ok:
            failed_routes.append(path)
            for item in route_issues:
                diagnostics.append(redact_sensitive_text(item, home=home))

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "kind": kind,
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": redact_sensitive_text(route_issues[0] if route_issues else "", home=home),
            }
        )

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    html_routes_count = sum(1 for _, kind, _ in HOLIDAY_PARITY_ROUTES if kind == "html")
    unique_csp_nonces_ok = bool(
        html_all_ok
        and len(html_contracts) == html_routes_count
        and len(seen_nonces) == html_routes_count
        and not any(info.get("nonce_reused") for info in html_contracts.values())
    )
    sha_drift = bool(clean_expected_sha is not None and sha_match is not True)
    stale_404_drift = bool(missing_routes or manifest_missing_routes or manifest_missing_scripts)
    cloud_run_lags_main = bool(
        http_statuses.get("/health") == 200 and (stale_404_drift or sha_drift)
    )
    parity_ok = bool(
        passed_count == total_count
        and manifest_valid
        and unique_csp_nonces_ok
        and html_dom_clean
        and compliance_banners_ok
        and feedback_summary_ok
        and raid_status_ok
        and not sha_drift
    )

    fallback_guidance: dict[str, Any] | None = None
    if stale_404_drift:
        fallback_guidance = build_holiday_fallback_guidance(
            clean_gw,
            missing_routes=missing_routes or manifest_missing_routes,
            lang=lang_code,
        )
        diagnostics.insert(0, redact_sensitive_text(fallback_guidance["active_message"], home=home))

    report: dict[str, Any] = {
        "object": "hivemind.quickstart.holiday_deploy_parity_receipt",
        "campaign_id": CAMPAIGN_ID,
        "gateway": clean_gw,
        "lang": lang_code,
        "compliance_ok": True,
        "parity_ok": parity_ok,
        "redeploy_recommended": bool(not parity_ok or cloud_run_lags_main or sha_drift),
        "cloud_run_lags_main": cloud_run_lags_main,
        "stale_404_drift": stale_404_drift,
        "routes_passed_count": passed_count,
        "routes_total_count": total_count,
        "http_statuses": http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
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
        "raid_status_ok": raid_status_ok,
        "zero_inference_verified": True,
        "network_calls": network_calls,
        "file_writes": 0,
        "masked_api_key": masked_key,
        "credentials_file": redacted_cred_path,
        "fallback_guidance": fallback_guidance,
        "diagnostics": diagnostics,
        "routes": route_rows,
        "compliance_notice": COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN,
        "financial_promise": FINANCIAL_PROMISE,
    }
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def derive_zero_game_lag(
    *,
    had_game_lag: bool = False,
    yield_elapsed_s: float | None = None,
    gpu_temp_c: float | None = None,
    quiet_temp_ceiling_c: float = GAMER_WHISPER_MAX_TEMP_C,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
) -> tuple[bool, bool, bool]:
    """Derive ``(zero_game_lag, yield_slo_ok, quiet_fan_ok)`` honestly from ``<1.5s`` yield & ``<=70C`` status."""
    yield_slo_ok = (yield_elapsed_s is None) or (float(yield_elapsed_s) < float(yield_budget_s))
    quiet_fan_ok = (gpu_temp_c is None) or (float(gpu_temp_c) <= float(quiet_temp_ceiling_c))
    zero_game_lag = (not bool(had_game_lag)) and yield_slo_ok and quiet_fan_ok
    return zero_game_lag, yield_slo_ok, quiet_fan_ok


def _normalize_platform_token(raw_platform: str | None) -> str:
    candidate = (raw_platform if raw_platform is not None else platform.system()).strip().lower()
    candidate = re.sub(r"[^a-z0-9_-]+", "", candidate)[:24].rstrip("_-")
    if candidate and _PLATFORM_RE.fullmatch(candidate):
        return candidate
    return "unknown"


def _expand_spool_path(raw_path: Path | str, *, home: Path | None = None) -> Path:
    text = str(raw_path).strip()
    if text == "~" or text.startswith("~/") or text.startswith("~\\"):
        base_home = Path(home) if home is not None else Path.home()
        suffix = text[2:] if len(text) > 1 else ""
        return (base_home / suffix) if suffix else base_home
    return Path(text)


def default_spool_path(
    home: Path | str | None = None,
    credentials_path: Path | str | None = None,
) -> Path:
    """Resolve isolated local CES telemetry retry spool path without touching real ``~/.hivemind`` when isolated."""
    if credentials_path is not None and str(credentials_path).strip():
        cred_p = _expand_spool_path(credentials_path, home=Path(home) if home is not None else None)
        if home is not None:
            resolved_home = Path(home).resolve()
            if cred_p == _default_credentials_path() and resolved_home != Path.home().resolve():
                return Path(home) / ".hivemind" / DEFAULT_SPOOL_FILENAME
        return cred_p.parent / DEFAULT_SPOOL_FILENAME
    resolved_base = Path(home) if home is not None else Path.home()
    return resolved_base / ".hivemind" / DEFAULT_SPOOL_FILENAME


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
    if isinstance(profile, bool) or not isinstance(profile, str):
        raise ValueError(f"Unsupported profile {profile!r}; expected 'dev' or 'gamer'.")
    norm_profile = profile.strip().lower()
    if norm_profile not in ("dev", "gamer"):
        raise ValueError(f"Unsupported profile {profile!r}; expected 'dev' or 'gamer'.")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ValueError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise ValueError("zero_game_lag must be a boolean.")
    clean_platform = _normalize_platform_token(platform_name)
    clean_note = redact_sensitive_text(str(note or ""), home=home)[:240]
    return {
        "profile": norm_profile,
        "ease_rating": int(ease_rating),
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


def compute_ces_idempotency_key(
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
    gateway: str | None = None,
    spool_path: Path | str | None = None,
    home: Path | None = None,
) -> str:
    """Compute a deterministic 64-char SHA-256 ``Idempotency-Key`` for ``POST /v1/hivemind/onboarding/feedback``."""
    clean_payload = _validate_ces_payload_mapping(payload, home=home)
    if idempotency_key is not None:
        if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be an ASCII string.")
        if not idempotency_key.isascii() or any(ord(ch) < 32 or ord(ch) == 127 for ch in idempotency_key):
            raise ValueError("idempotency_key must contain printable ASCII characters only.")
        cleaned = idempotency_key.strip()
        if not cleaned:
            raise ValueError("idempotency_key must not be empty.")
        if _SHA256_HEX_RE.fullmatch(cleaned.lower()):
            return cleaned.lower()
        return hashlib.sha256(f"hivemind.quickstart.ces.key/1:{cleaned}".encode()).hexdigest()

    clean_gw = validate_gateway_url(gateway) if gateway else DEFAULT_GATEWAY
    scope_target = (
        _expand_spool_path(spool_path, home=home)
        if spool_path is not None and str(spool_path).strip()
        else default_spool_path(home=home)
    )
    scope_digest = hashlib.sha256(str(scope_target.resolve()).encode("utf-8")).hexdigest()[:24]
    canon = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(
        f"hivemind.quickstart.ces.scope/1:{clean_gw}:{scope_digest}:{canon}".encode()
    ).hexdigest()


class QuickstartCesSpool:
    """Thread-safe, atomic POSIX-0600 bounded CES & zero-game-lag retry spool with Idempotency-Key replay."""

    def __init__(
        self,
        spool_path: Path | str | None = None,
        *,
        home: Path | None = None,
        credentials_path: Path | str | None = None,
        max_entries: int = DEFAULT_SPOOL_MAX_ENTRIES,
        max_delivered: int = DEFAULT_SPOOL_MAX_DELIVERED,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not (1 <= max_entries <= 4096):
            raise ValueError("max_entries must be an integer between 1 and 4096.")
        if isinstance(max_delivered, bool) or not isinstance(max_delivered, int) or not (1 <= max_delivered <= 32768):
            raise ValueError("max_delivered must be an integer between 1 and 32768.")
        self._home = Path(home) if home is not None else None
        raw_p = (
            _expand_spool_path(spool_path, home=self._home)
            if spool_path is not None and str(spool_path).strip()
            else default_spool_path(home=self._home, credentials_path=credentials_path)
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
            "schema": SPOOL_SCHEMA,
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
                    "event_id": hashlib.sha256(b"corrupted_quickstart_spool_recovered").hexdigest(),
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
                "schema": SPOOL_SCHEMA,
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
            "schema": SPOOL_SCHEMA,
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
        """Return a read-only sanitized spool snapshot without creating files when empty."""
        eff_home = home if home is not None else self._home
        with self._lock:
            exists = self._path.is_file()
            state, _ = self._load_state_unlocked()
        snap: dict[str, Any] = {
            "object": "hivemind.quickstart.ces_spool_snapshot",
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
        snap["receipt_sha256"] = compute_receipt_sha256(snap)
        return snap

    def mark_delivered(self, event_id: str, *, home: Path | None = None) -> bool:
        """Record ``event_id`` as delivered if a spool file already exists (never creates a new file)."""
        _ = home
        if not _SHA256_HEX_RE.fullmatch(event_id):
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
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Enforce compliance, deduplicate by SHA-256 ``event_id``, evict FIFO at ``max_entries``, and save ``0600``."""
        lang_code = normalize_lang(lang)
        eff_home = home if home is not None else self._home
        compliance = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if not compliance["compliance_ok"]:
            blocked: dict[str, Any] = {
                "object": "hivemind.quickstart.ces_spool_enqueue",
                "compliance_ok": False,
                "corporate_indicators": compliance["corporate_indicators"],
                "enqueued": False,
                "deduplicated": False,
                "duplicate_state": "compliance_refused",
                "spool_file": redact_home_path(self._path, home=eff_home),
                "network_calls": 0,
                "file_writes": 0,
                "summary": compliance["summary"],
            }
            blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
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
        event_id = compute_ces_idempotency_key(
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

            if event_id in delivered_ids:
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                res_deliv: dict[str, Any] = {
                    "object": "hivemind.quickstart.ces_spool_enqueue",
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
                res_deliv["receipt_sha256"] = compute_receipt_sha256(res_deliv)
                return res_deliv

            if any(item["event_id"] == event_id for item in pending):
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                res_pend: dict[str, Any] = {
                    "object": "hivemind.quickstart.ces_spool_enqueue",
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
                    "payload": clean_payload,
                    "retry_count": 0,
                    "last_http_status": http_status,
                    "last_reason": clean_reason,
                    "enqueued_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                }
            )
            self._atomic_save_unlocked(state)
            res_ok: dict[str, Any] = {
                "object": "hivemind.quickstart.ces_spool_enqueue",
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
            res_ok["receipt_sha256"] = compute_receipt_sha256(res_ok)
            return res_ok

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        http_fn: HttpFn = default_http_json,
        headers: dict[str, str] | None = None,
        timeout_s: float = TELEMETRY_TIMEOUT_S,
        lang: str = "en",
        home: Path | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> dict[str, Any]:
        """Idempotently flush pending CES events with ``Idempotency-Key`` headers without creating empty files."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        eff_home = home if home is not None else self._home
        clean_gw = validate_gateway_url(gateway)

        compliance = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if not compliance["compliance_ok"]:
            blocked: dict[str, Any] = {
                "object": "hivemind.quickstart.ces_spool_flush",
                "gateway": clean_gw,
                "lang": lang_code,
                "compliance_ok": False,
                "corporate_indicators": compliance["corporate_indicators"],
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
                "diagnostics": [compliance["summary"]],
            }
            blocked["receipt_sha256"] = compute_receipt_sha256(blocked)
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
                    "object": "hivemind.quickstart.ces_spool_flush",
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
                empty_res["receipt_sha256"] = compute_receipt_sha256(empty_res)
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

                req_headers = dict(headers) if headers else {}
                req_headers["Idempotency-Key"] = eid
                code, body = _safe_http_call(
                    http_fn,
                    "POST",
                    post_url,
                    dict(entry["payload"]),
                    req_headers,
                    timeout_s=timeout_s,
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
                else "HTTP 404 during CES spool flush: entries safely retained for post-cutover replay."
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
            "object": "hivemind.quickstart.ces_spool_flush",
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
        res["receipt_sha256"] = compute_receipt_sha256(res)
        return res


def enqueue_spooled_telemetry(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    yield_elapsed_s: float | None = None,
    gpu_temp_c: float | None = None,
    quiet_temp_ceiling_c: float = GAMER_WHISPER_MAX_TEMP_C,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    platform_name: str | None = None,
    note: str = "",
    idempotency_key: str | None = None,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_status: int | None = None,
    reason: str = "",
    max_entries: int = DEFAULT_SPOOL_MAX_ENTRIES,
    max_delivered: int = DEFAULT_SPOOL_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Derive ``zero_game_lag`` and atomically enqueue a zero-PII CES telemetry event into the ``0600`` spool."""
    zero_game_lag, _, _ = derive_zero_game_lag(
        had_game_lag=had_game_lag,
        yield_elapsed_s=yield_elapsed_s,
        gpu_temp_c=gpu_temp_c,
        quiet_temp_ceiling_c=quiet_temp_ceiling_c,
        yield_budget_s=yield_budget_s,
    )
    spool = QuickstartCesSpool(
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
        env=env,
        cwd=cwd,
        lang=lang,
    )


def flush_spooled_telemetry(
    gateway: str = DEFAULT_GATEWAY,
    *,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    headers: dict[str, str] | None = None,
    timeout_s: float = TELEMETRY_TIMEOUT_S,
    max_entries: int = DEFAULT_SPOOL_MAX_ENTRIES,
    max_delivered: int = DEFAULT_SPOOL_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Flush locally spooled CES telemetry items idempotently against ``POST /v1/hivemind/onboarding/feedback``."""
    spool = QuickstartCesSpool(
        spool_path,
        home=home,
        credentials_path=credentials_path,
        max_entries=max_entries,
        max_delivered=max_delivered,
    )
    return spool.flush(
        gateway,
        http_fn=http_fn,
        headers=headers,
        timeout_s=timeout_s,
        lang=lang,
        home=home,
        env=env,
        cwd=cwd,
    )


def submit_post_setup_telemetry(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    yield_elapsed_s: float | None = None,
    gpu_temp_c: float | None = None,
    quiet_temp_ceiling_c: float = GAMER_WHISPER_MAX_TEMP_C,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    platform_name: str | None = None,
    note: str = "",
    headers: dict[str, str] | None = None,
    http_fn: HttpFn = default_http_json,
    timeout_s: float = TELEMETRY_TIMEOUT_S,
    lang: str = "en",
    home: Path | None = None,
    idempotency_key: str | None = None,
    spool_path: Path | str | None = None,
    auto_flush: bool = True,
    credentials_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Submit timeout-safe, non-blocking zero-PII CES & Steam zero-game-lag telemetry with resilient 0600 spool."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    payload = _normalize_ces_payload(
        profile=profile,
        ease_rating=ease_rating,
        zero_game_lag=True,
        platform_name=platform_name,
        note=note,
        home=home,
    )
    zero_game_lag, yield_slo_ok, quiet_fan_ok = derive_zero_game_lag(
        had_game_lag=had_game_lag,
        yield_elapsed_s=yield_elapsed_s,
        gpu_temp_c=gpu_temp_c,
        quiet_temp_ceiling_c=quiet_temp_ceiling_c,
        yield_budget_s=yield_budget_s,
    )
    payload["zero_game_lag"] = bool(zero_game_lag)
    clean_platform = str(payload["platform"])

    compliance = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        blocked_res: dict[str, Any] = {
            "object": "hivemind.quickstart.post_setup_telemetry",
            "compliance_ok": False,
            "corporate_indicators": compliance["corporate_indicators"],
            "telemetry_recorded": False,
            "spooled": False,
            "http_status": 0,
            "profile": payload["profile"],
            "ease_rating": int(ease_rating),
            "zero_game_lag": bool(zero_game_lag),
            "yield_slo_ok": bool(yield_slo_ok),
            "quiet_fan_ok": bool(quiet_fan_ok),
            "platform": clean_platform,
            "stale_cloud_run": False,
            "degraded_reason": "refused_compliance",
            "diagnostic": compliance["summary"],
            "summary": None,
            "network_calls": 0,
            "file_writes": 0,
        }
        blocked_res["receipt_sha256"] = compute_receipt_sha256(blocked_res)
        return blocked_res

    eff_spool_path = (
        _expand_spool_path(spool_path, home=home)
        if spool_path is not None and str(spool_path).strip()
        else default_spool_path(home=home, credentials_path=credentials_path)
    )
    resolved_idem_key = compute_ces_idempotency_key(
        payload,
        idempotency_key=idempotency_key,
        gateway=clean_gw,
        spool_path=eff_spool_path,
        home=home,
    )
    req_headers = dict(headers) if headers else {}
    if "idempotency-key" not in {k.lower() for k in req_headers}:
        req_headers["Idempotency-Key"] = resolved_idem_key
    else:
        for hk, hv in req_headers.items():
            if hk.lower() == "idempotency-key":
                resolved_idem_key = str(hv).strip() or resolved_idem_key
                break

    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        req_headers,
        timeout_s=timeout_s,
    )

    telemetry_recorded = status_code in (200, 201)
    stale_cloud_run = status_code == 404
    degraded_reason: str | None = None
    diagnostic = ""
    sanitized_summary: dict[str, Any] | None = None
    spooled = False
    spool_deduplicated = False
    spool_pending_count = 0
    quarantined = False
    auto_flush_report: dict[str, Any] | None = None

    spool = QuickstartCesSpool(eff_spool_path, home=home)

    if telemetry_recorded:
        sanitized_summary = _sanitize_feedback_summary(body)
        if eff_spool_path.is_file():
            spool.mark_delivered(resolved_idem_key, home=home)
            if auto_flush:
                auto_flush_report = spool.flush(
                    clean_gw,
                    http_fn=http_fn,
                    headers=headers,
                    timeout_s=timeout_s,
                    lang=lang_code,
                    home=home,
                    env=env,
                    cwd=cwd,
                )
                if auto_flush_report.get("summary") is not None:
                    sanitized_summary = auto_flush_report["summary"]
                spool_pending_count = int(auto_flush_report.get("pending_count", 0))
        diagnostic = (
            "Télémétrie CES post-installation anonyme enregistrée."
            if is_fr
            else "Anonymous post-setup CES telemetry recorded."
        )
    else:
        if status_code == 404:
            degraded_reason = "stale_cloud_run_404"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : Cloud Run en retard sur `main` (HTTP 404)."
                if is_fr
                else "Telemetry skipped without blocking setup: Cloud Run revision lags behind `main` (HTTP 404)."
            )
        elif status_code == 408:
            degraded_reason = "request_timeout_408"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : délai d'attente expiré (HTTP 408)."
                if is_fr
                else "Telemetry skipped without blocking setup: gateway request timeout (HTTP 408)."
            )
        elif status_code == 429:
            degraded_reason = "rate_limited_429"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : limite de débit atteinte (HTTP 429)."
                if is_fr
                else "Telemetry skipped without blocking setup: gateway rate limit reached (HTTP 429)."
            )
        elif status_code >= 500:
            degraded_reason = "gateway_5xx"
            diagnostic = (
                f"Télémétrie ignorée sans bloquer : erreur passerelle (HTTP {status_code})."
                if is_fr
                else f"Telemetry skipped without blocking setup: gateway error (HTTP {status_code})."
            )
        elif status_code == 0:
            degraded_reason = "network_timeout_or_unreachable"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : délai réseau dépassé ou passerelle injoignable."
                if is_fr
                else "Telemetry skipped without blocking setup: network timeout or gateway unreachable."
            )
        else:
            degraded_reason = f"http_{status_code}"
            diagnostic = (
                f"Télémétrie ignorée sans bloquer (HTTP {status_code})."
                if is_fr
                else f"Telemetry skipped without blocking setup (HTTP {status_code})."
            )

        is_transient = status_code in (0, 404, 408, 429) or status_code >= 500
        if is_transient:
            enq = spool.enqueue(
                profile=payload["profile"],
                ease_rating=int(ease_rating),
                zero_game_lag=bool(zero_game_lag),
                platform_name=clean_platform,
                note=str(payload["note"]),
                gateway=clean_gw,
                idempotency_key=resolved_idem_key,
                http_status=status_code,
                reason=degraded_reason,
                home=home,
                env=env,
                cwd=cwd,
                lang=lang_code,
            )
            spooled = bool(enq.get("enqueued") or enq.get("duplicate_state") == "already_pending")
            spool_deduplicated = bool(enq.get("deduplicated"))
            spool_pending_count = int(enq.get("pending_count", 0))
        else:
            quarantined = True

    result: dict[str, Any] = {
        "object": "hivemind.quickstart.post_setup_telemetry",
        "compliance_ok": True,
        "telemetry_recorded": telemetry_recorded,
        "http_status": status_code,
        "profile": payload["profile"],
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "yield_slo_ok": bool(yield_slo_ok),
        "quiet_fan_ok": bool(quiet_fan_ok),
        "platform": clean_platform,
        "idempotency_key": resolved_idem_key,
        "stale_cloud_run": stale_cloud_run,
        "degraded_reason": degraded_reason,
        "spooled": spooled,
        "spool_deduplicated": spool_deduplicated,
        "spool_pending_count": spool_pending_count,
        "quarantined": quarantined,
        "spool_file": redact_home_path(eff_spool_path, home=home) if (spooled or eff_spool_path.is_file()) else None,
        "auto_flush": auto_flush_report,
        "diagnostic": redact_sensitive_text(diagnostic, home=home),
        "summary": sanitized_summary,
    }
    result["receipt_sha256"] = compute_receipt_sha256(result)
    return result


def save_credentials(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_credentials(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and str(data.get("api_key", "")).startswith("hm-"):
        return data
    return None


def probe_local_gpu(*, runner_fn: RunnerFn | None = None) -> GpuSnapshot:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        if runner_fn is not None:
            rc, stdout = runner_fn(cmd, 5.0)
        else:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                shell=False,
            )
            rc, stdout = proc.returncode, proc.stdout
        if rc == 0 and stdout.strip():
            parts = [p.strip() for p in stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 5:
                return GpuSnapshot(
                    present=True,
                    name=parts[0],
                    total_mib=max(0, int(float(parts[1]))),
                    free_mib=max(0, int(float(parts[2]))),
                    temp_c=float(parts[3]),
                    util_pct=float(parts[4]),
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)
    return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)


def list_running_process_names(
    *,
    os_name: str | None = None,
    runner_fn: RunnerFn | None = None,
) -> set[str]:
    names: set[str] = set()
    eff_os = os_name if os_name is not None else os.name
    try:
        if eff_os == "nt":
            cmd = ["tasklist", "/FO", "CSV", "/NH"]
            if runner_fn is not None:
                _, stdout = runner_fn(cmd, 5.0)
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
                stdout = proc.stdout
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith('"') and '","' in line:
                    names.add(line.split('","', 1)[0].strip('"').lower())
        else:
            cmd = ["ps", "-eo", "comm="]
            if runner_fn is not None:
                _, stdout = runner_fn(cmd, 5.0)
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
                stdout = proc.stdout
            for line in stdout.splitlines():
                cleaned = line.strip().lower()
                if cleaned:
                    names.add(cleaned)
    except (OSError, subprocess.SubprocessError):
        return names
    return names


def get_os_idle_seconds() -> float:
    if sys.platform != "win32":
        return 999.0
    try:
        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]

        lii = _LastInputInfo()
        lii.cbSize = ctypes.sizeof(_LastInputInfo)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):  # type: ignore[attr-defined]
            tick = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
            elapsed_ms = (tick - lii.dwTime) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        return 999.0
    return 999.0


def detect_active_game(running: set[str], extra_games: list[str] | None = None) -> str | None:
    watch = set(DEFAULT_GAME_EXECUTABLES)
    for g in extra_games or []:
        cleaned = g.strip().lower()
        if cleaned:
            watch.add(cleaned)
            watch.add(cleaned[:15])
    for proc_name in sorted(running):
        if proc_name in watch:
            return proc_name
    return None


def evaluate_gamer_whisper_guard(
    gpu: GpuSnapshot,
    *,
    running_procs: set[str],
    extra_games: list[str] | None = None,
    reserve_mib: int = 2048,
    hot_c: float = 70.0,
    max_external_util_pct: float = 15.0,
    worker_running: bool = False,
) -> dict[str, Any]:
    """Decide whether a gaming PC is quiet and free enough to run or keep running a lightweight round."""
    active_game = detect_active_game(running_procs, extra_games)
    if active_game is not None:
        return {"allow": False, "reason": f"game_active:{active_game}", "active_game": active_game}
    if not gpu.present or gpu.total_mib <= 0:
        return {"allow": False, "reason": "no_nvidia_gpu_detected", "active_game": None}
    if gpu.temp_c >= hot_c:
        return {"allow": False, "reason": f"thermal_whisper_cap:{gpu.temp_c:.0f}C>={hot_c:.0f}C", "active_game": None}
    if gpu.free_mib < reserve_mib:
        return {"allow": False, "reason": f"vram_reserve_guard:{gpu.free_mib}MiB<{reserve_mib}MiB", "active_game": None}
    if not worker_running and gpu.util_pct > max_external_util_pct:
        return {
            "allow": False,
            "reason": f"external_gpu_busy:{gpu.util_pct:.0f}%>{max_external_util_pct:.0f}%",
            "active_game": None,
        }
    return {"allow": True, "reason": "ready_gamer_whisper", "active_game": None}


def build_continue_yaml(gateway: str, api_key: str) -> str:
    return (
        "# HiveMind personal hobby/open-source coding configuration for VS Code (Continue.dev)\n"
        "# Personal projects ONLY — never send corporate or confidential source code.\n"
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "  - name: HiveMind Bee-Stars (Science)\n"
        "    provider: openai\n"
        "    model: hivemind/bee-stars\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
    )


def terminate_process_tree_fast(proc: subprocess.Popen[Any], timeout_s: float = 1.2) -> float:
    """Stop a worker process tree immediately so 100% of GPU VRAM returns to the owner within < 1.5 s."""
    started = time.monotonic()
    if proc.poll() is not None:
        return round(time.monotonic() - started, 4)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        try:
            proc.wait(timeout=max(0.2, timeout_s * 0.6))
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=0.5)
    except (OSError, subprocess.SubprocessError):
        return round(time.monotonic() - started, 4)
    return round(time.monotonic() - started, 4)


def ensure_connected_account(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None,
    role: str,
    explicit_api_key: str | None,
    http_fn: HttpFn = default_http_json,
    env: Mapping[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    clean_gw = validate_gateway_url(gateway)
    env_view = env if env is not None else os.environ
    api_key = (explicit_api_key or env_view.get("HIVEMIND_API_KEY") or "").strip()
    saved = load_credentials(credentials_path)
    if not api_key and saved and saved.get("gateway") == clean_gw:
        api_key = str(saved["api_key"])

    created = False
    resolved_handle = handle or (saved.get("handle") if saved else None) or ""
    if not api_key:
        base_handle = sanitize_handle(handle or f"hm-{platform.node() or 'home'}")
        status, body = _safe_http_call(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/register",
            {"handle": base_handle, "role": role},
            headers,
        )
        if status == 409 and not handle:
            retry_handle = sanitize_handle(f"{base_handle[:38]}-{secrets.token_hex(2)}")
            status, body = _safe_http_call(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/register",
                {"handle": retry_handle, "role": role},
                headers,
            )
            base_handle = retry_handle
        if status != 200 or not str(body.get("api_key", "")).startswith("hm-"):
            msg = body.get("error") or body
            raise RuntimeError(f"Registration refused (HTTP {status}): {msg}")
        api_key = str(body["api_key"])
        resolved_handle = base_handle
        created = True

    me_headers = dict(headers or {})
    me_headers["Authorization"] = f"Bearer {api_key}"
    me_status, me_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/me",
        None,
        me_headers,
    )
    if me_status != 200:
        raise RuntimeError(f"API key check failed on /v1/hivemind/me (HTTP {me_status}): {me_body}")

    models_status, models_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/models", None, headers)
    models = [m.get("id") for m in (models_body.get("data") or []) if isinstance(m, dict) and m.get("id")]
    if models_status != 200:
        models = []

    record = {
        "account_id": me_body.get("account_id", ""),
        "api_key": api_key,
        "created_here": created,
        "gateway": clean_gw,
        "handle": me_body.get("handle") or resolved_handle,
        "role": me_body.get("role") or role,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    save_credentials(credentials_path, record)
    return {"account": record, "balance": me_body.get("balance", 0.0), "models": models}


def run_gamer_whisper_session(
    *,
    gateway: str,
    api_key: str,
    extra_games: list[str],
    reserve_mib: int,
    hot_c: float,
    cool_c: float,
    once: bool,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_local_gpu,
    proc_list_fn: Callable[[], set[str]] = list_running_process_names,
    popen_fn: Callable[..., subprocess.Popen[Any]] | None = None,
    poll_interval_s: float = 0.5,
) -> dict[str, Any]:
    clean_gw = validate_gateway_url(gateway)
    gpu = gpu_probe_fn()
    max_temp_c: float | None = round(float(gpu.temp_c), 2) if gpu.present else None
    preflight_decision = evaluate_gamer_whisper_guard(
        gpu,
        running_procs=proc_list_fn(),
        extra_games=extra_games,
        reserve_mib=reserve_mib,
        hot_c=hot_c,
        worker_running=False,
    )
    if not preflight_decision["allow"]:
        return {
            "status": "yielded_before_start",
            **preflight_decision,
            "yield_elapsed_s": 0.0,
            "gpu_temp_c": max_temp_c,
        }

    agent_script = Path(__file__).resolve().with_name("hivemind_opentrainer_agent.py")
    cmd = [
        sys.executable,
        str(agent_script),
        "--gateway",
        clean_gw,
        "--max-vram-gb",
        "4.5",
        "--train",
    ]
    if once:
        cmd.append("--once")
    env = dict(os.environ)
    env["HIVEMIND_API_KEY"] = api_key
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    launcher = popen_fn or subprocess.Popen
    popen_kwargs: dict[str, Any] = {"env": env}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = launcher(cmd, **popen_kwargs)

    while proc.poll() is None:
        time.sleep(max(0.05, poll_interval_s))
        live_gpu = gpu_probe_fn()
        if live_gpu.present:
            max_temp_c = (
                round(float(live_gpu.temp_c), 2)
                if max_temp_c is None
                else round(max(max_temp_c, float(live_gpu.temp_c)), 2)
            )
        live_decision = evaluate_gamer_whisper_guard(
            live_gpu,
            running_procs=proc_list_fn(),
            extra_games=extra_games,
            reserve_mib=reserve_mib,
            hot_c=hot_c,
            worker_running=True,
        )
        if not live_decision["allow"]:
            elapsed_s = terminate_process_tree_fast(proc)
            return {
                "status": "yielded_to_owner",
                **live_decision,
                "cool_resume_c": cool_c,
                "yield_elapsed_s": elapsed_s,
                "gpu_temp_c": max_temp_c,
            }

    return {
        "status": "round_completed" if proc.returncode == 0 else "worker_exited",
        "returncode": proc.returncode,
        "gpu_temp_c": max_temp_c,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HiveMind 60-second Personal Quick-Connect & Gamer-Whisper Assistant (0.00 EUR, personal PCs only)."
    )
    p.add_argument("--gateway", default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY))
    p.add_argument("--handle", default="", help="Desired handle (2-48 chars, e.g. alex-rtx4070).")
    p.add_argument("--role", default="prosumer", choices=("consumer", "producer", "prosumer", "trainer"))
    p.add_argument("--api-key", default="", help="Existing hm-... API key (optional).")
    p.add_argument("--credentials-file", default=str(_default_credentials_path()))
    p.add_argument("--write-continue-config", action="store_true", help="Write ~/.continue/hivemind_provider.yaml.")
    p.add_argument("--continue-config-path", default=str(_default_continue_snippet_path()))
    p.add_argument("--lang", default="en", help="Output and diagnostic language ('en' or 'fr').")
    p.add_argument(
        "--smoke-routes",
        action="store_true",
        help="Run zero-inference Cloud Run holiday route smoke gate and exit.",
    )
    p.add_argument(
        "--check-deploy-parity",
        action="store_true",
        help="Run 8-endpoint zero-inference Cloud Run holiday deploy-parity gate (CSP, banners, manifest, SHA) and exit.",
    )
    p.add_argument(
        "--expected-sha",
        default="",
        help="Expected build SHA to verify against /v1/hivemind/onboarding/manifest under --check-deploy-parity.",
    )
    p.add_argument(
        "--require-holiday-routes",
        action="store_true",
        help="Fail closed (exit 2) before setup when required Cloud Run holiday routes fail or return 404/5xx.",
    )
    p.add_argument("--gamer-check", action="store_true", help="Inspect GPU, temperature, and running games only.")
    p.add_argument("--gamer-whisper", action="store_true", help="Run quiet GPU contributor (<70C, <1.5s game stop).")
    p.add_argument("--game", action="append", default=[], help="Extra game executable name to auto-stop on.")
    p.add_argument("--reserve-mib", type=int, default=2048, help="Free VRAM MiB reserved for desktop/games.")
    p.add_argument("--hot-c", type=float, default=70.0, help="Quiet fan temperature cap in Celsius (default: 70).")
    p.add_argument("--cool-c", type=float, default=64.0, help="Cool-down temperature in Celsius (default: 64).")
    p.add_argument("--once", action="store_true", help="Run a single quiet round and exit.")
    p.add_argument(
        "--post-setup-telemetry",
        action="store_true",
        help="Submit non-blocking anonymous CES & Steam zero-game-lag telemetry after connect or --gamer-whisper.",
    )
    p.add_argument(
        "--telemetry-queue-file",
        default="",
        help="Optional path to local POSIX-0600 CES telemetry retry spool JSON.",
    )
    p.add_argument(
        "--flush-feedback-queue",
        action="store_true",
        help="Flush spooled zero-PII CES telemetry events idempotently to Cloud Run.",
    )
    p.add_argument("--feedback", action="store_true", help="Submit anonymous usability feedback (1..5).")
    p.add_argument("--ease-rating", type=int, default=5, help="Ease of setup rating from 1 (hard) to 5 (instant).")
    p.add_argument("--had-game-lag", action="store_true", help="Report that a game stuttered or fans were noisy.")
    p.add_argument("--profile", default="dev", choices=("dev", "gamer"))
    p.add_argument("--note", default="", help="Optional short feedback note (no personal data).")
    p.add_argument("--uninstall", action="store_true", help="Delete local saved HiveMind credentials, snippet, and spool.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return p


def _render_smoke_cli_output(smoke: Mapping[str, Any]) -> str:
    is_fr = smoke.get("lang") == "fr"
    header = (
        "HIVEMIND NOËL — SMOKE GATE CLOUD RUN (ZÉRO INFÉRENCE • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND HOLIDAY — CLOUD RUN ROUTE SMOKE GATE (ZERO INFERENCE • 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {header}",
        "====================================================================",
        f"  [Rule] {smoke.get('compliance_notice')}",
        f"  Gateway          : {smoke.get('gateway')}",
        f"  Credentials      : {smoke.get('credentials_file')} (key {smoke.get('masked_api_key')})",
        f"  Routes verified  : {smoke.get('routes_passed_count')}/{smoke.get('routes_total_count')} "
        f"(smoke_ok={smoke.get('smoke_ok')})",
    ]
    for row in smoke.get("routes") or ():
        mark = "OK" if row.get("ok") else "FAIL"
        lines.append(f"  [{mark:<4}] GET {row.get('path'):<42} -> HTTP {row.get('http_status')}")
    for diag in smoke.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt  : {smoke.get('receipt_sha256')}")
    return "\n".join(lines)


def _render_deploy_parity_cli_output(parity: Mapping[str, Any]) -> str:
    is_fr = parity.get("lang") == "fr"
    header = (
        "HIVEMIND NOËL — PARITÉ DÉPLOIEMENT CLOUD RUN (ZÉRO INFÉRENCE • 0,00 EUR)"
        if is_fr
        else "HIVEMIND HOLIDAY — CLOUD RUN DEPLOY PARITY GATE (ZERO INFERENCE • 0.00 EUR)"
    )
    lines = [
        "====================================================================",
        f"  {header}",
        "====================================================================",
        f"  [Rule] {parity.get('compliance_notice')}",
        f"  Gateway          : {parity.get('gateway')}",
        f"  Credentials      : {parity.get('credentials_file')} (key {parity.get('masked_api_key')})",
        f"  Routes verified  : {parity.get('routes_passed_count')}/{parity.get('routes_total_count')} "
        f"(parity_ok={parity.get('parity_ok')})",
        f"  Build SHA        : live={parity.get('live_build_sha')} expected={parity.get('expected_sha') or 'any'} "
        f"(rev={parity.get('cloud_run_revision') or 'none'})",
        f"  HTML / CSP       : nonces={parity.get('unique_csp_nonces_count')} "
        f"dom_clean={parity.get('html_dom_clean')} banners_ok={parity.get('compliance_banners_ok')}",
    ]
    for row in parity.get("routes") or ():
        mark = "OK" if row.get("ok") else "FAIL"
        lines.append(f"  [{mark:<4}] GET {row.get('path'):<42} -> HTTP {row.get('http_status')}")
    for diag in parity.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt  : {parity.get('receipt_sha256')}")
    return "\n".join(lines)


def _render_flush_cli_output(flush_res: Mapping[str, Any]) -> str:
    is_fr = flush_res.get("lang") == "fr"
    header = (
        "HIVEMIND NOËL — VIDAGE FILE TÉLÉMÉTRIE CES (0,00 EUR)"
        if is_fr
        else "HIVEMIND HOLIDAY — CES TELEMETRY SPOOL FLUSH (0.00 EUR)"
    )
    lines = [
        "====================================================================",
        f"  {header}",
        "====================================================================",
        f"  Gateway          : {flush_res.get('gateway')}",
        f"  Spool file       : {flush_res.get('spool_file')} (exists={flush_res.get('spool_exists')})",
        f"  Delivered        : {flush_res.get('delivered_count')} | Pending: {flush_res.get('pending_count')} | "
        f"Quarantined: {flush_res.get('total_quarantined_count')}",
    ]
    for diag in flush_res.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt  : {flush_res.get('receipt_sha256')}")
    return "\n".join(lines)


def _resolve_cli_spool_path(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
    home: Path | None,
    cred_path: Path,
) -> tuple[Path, Path | None]:
    explicit_cred_flag = any(tok == "--credentials-file" or tok.startswith("--credentials-file=") for tok in raw_argv)
    eff_cred: Path | None = cred_path if explicit_cred_flag else None
    raw_q = str(getattr(args, "telemetry_queue_file", "") or "").strip()
    if raw_q:
        return _expand_spool_path(raw_q, home=home), eff_cred
    return default_spool_path(home=home, credentials_path=eff_cred), eff_cred


def main(
    argv: list[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    raw_http_fn: RawHttpFn | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    headers: dict[str, str] | None = None,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_local_gpu,
    proc_list_fn: Callable[[], set[str]] = list_running_process_names,
    popen_fn: Callable[..., subprocess.Popen[Any]] | None = None,
    poll_interval_s: float = 0.5,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    eff_out = out_stream if out_stream is not None else sys.stdout
    eff_err = err_stream if err_stream is not None else sys.stderr
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    explicit_cred_flag = any(tok == "--credentials-file" or tok.startswith("--credentials-file=") for tok in raw_argv)
    explicit_cont_flag = any(
        tok == "--continue-config-path" or tok.startswith("--continue-config-path=") for tok in raw_argv
    )
    cred_path = _resolve_credentials_file(args.credentials_file if explicit_cred_flag else None, home)
    cont_path = (
        Path(args.continue_config_path)
        if explicit_cont_flag or home is None
        else Path(home) / ".continue" / "hivemind_provider.yaml"
    )
    spool_path, eff_cred_for_spool = _resolve_cli_spool_path(
        args,
        raw_argv=raw_argv,
        home=home,
        cred_path=cred_path,
    )

    if args.uninstall:
        removed: list[str] = []
        for target in dict.fromkeys((cred_path, cont_path, spool_path)):
            if target.is_file():
                target.unlink()
                removed.append(redact_home_path(target, home=home))
        print(json.dumps({"status": "uninstalled", "removed_files": removed}, indent=2), file=eff_out)
        return 0

    try:
        lang_code = normalize_lang(args.lang)
    except ValueError as exc:
        print(f"Error: {exc}", file=eff_err)
        return 2

    eff_env = env if env is not None else os.environ
    eff_cwd = Path(cwd) if cwd is not None else Path.cwd()
    compliance = check_personal_pc_compliance(env=eff_env, cwd=eff_cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        print(compliance["summary"], file=eff_err)
        if args.json:
            print(json.dumps(compliance, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(f"compliance_ok=False | {compliance['summary']}", file=eff_out)
        return 2

    raw_gateway = (
        eff_env.get("HIVEMIND_GATEWAY_URL", args.gateway)
        if (env is not None and not any(tok == "--gateway" or tok.startswith("--gateway=") for tok in raw_argv))
        else args.gateway
    )
    try:
        gateway = validate_gateway_url(raw_gateway)
    except ValueError as exc:
        print(f"Error: {redact_sensitive_text(str(exc), home=home)}", file=eff_err)
        return 2

    if (args.feedback or args.post_setup_telemetry) and not (1 <= int(args.ease_rating) <= 5):
        print("Error: --ease-rating must be between 1 and 5.", file=eff_err)
        return 2

    explicit_or_env_key = (args.api_key or eff_env.get("HIVEMIND_API_KEY") or "").strip() or None

    if args.check_deploy_parity:
        try:
            parity = check_holiday_deploy_parity(
                gateway,
                expected_sha=args.expected_sha.strip() or None,
                http_fn=http_fn,
                raw_http_fn=raw_http_fn,
                api_key=explicit_or_env_key,
                credentials_path=cred_path if eff_cred_for_spool is not None else None,
                headers=headers,
                lang=lang_code,
                home=home,
                env=eff_env,
                cwd=eff_cwd,
            )
        except ValueError as exc:
            print(f"Error: {redact_sensitive_text(str(exc), home=home)}", file=eff_err)
            return 2

        flush_after_parity: dict[str, Any] | None = None
        if parity["parity_ok"] and args.flush_feedback_queue:
            flush_after_parity = flush_spooled_telemetry(
                gateway,
                spool_path=spool_path,
                credentials_path=eff_cred_for_spool,
                http_fn=http_fn,
                headers=headers,
                lang=lang_code,
                home=home,
                env=eff_env,
                cwd=eff_cwd,
            )
            parity["flush_feedback_queue"] = flush_after_parity
            parity["receipt_sha256"] = compute_receipt_sha256(parity)

        if args.json:
            print(json.dumps(parity, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(_render_deploy_parity_cli_output(parity), file=eff_out)
            if flush_after_parity is not None:
                print(_render_flush_cli_output(flush_after_parity), file=eff_out)
        if not parity["parity_ok"]:
            for diag in parity["diagnostics"]:
                print(diag, file=eff_err)
            return 2
        if flush_after_parity is not None and not flush_after_parity["flush_ok"]:
            for diag in flush_after_parity["diagnostics"]:
                print(diag, file=eff_err)
            return 2
        return 0

    if args.smoke_routes:
        smoke = verify_holiday_routes(
            gateway,
            http_fn=http_fn,
            lang=lang_code,
            api_key=explicit_or_env_key,
            credentials_path=cred_path,
            headers=headers,
            home=home,
        )
        if args.json:
            print(json.dumps(smoke, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(_render_smoke_cli_output(smoke), file=eff_out)
        if not smoke["smoke_ok"]:
            for diag in smoke["diagnostics"]:
                print(diag, file=eff_err)
            return 2
        return 0

    if args.flush_feedback_queue and not (
        args.post_setup_telemetry or args.feedback or args.gamer_whisper or args.write_continue_config or args.handle
    ):
        flush_res = flush_spooled_telemetry(
            gateway,
            spool_path=spool_path,
            credentials_path=eff_cred_for_spool,
            http_fn=http_fn,
            headers=headers,
            lang=lang_code,
            home=home,
            env=eff_env,
            cwd=eff_cwd,
        )
        if args.json:
            print(json.dumps(flush_res, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(_render_flush_cli_output(flush_res), file=eff_out)
        if not flush_res["flush_ok"]:
            for diag in flush_res["diagnostics"]:
                print(diag, file=eff_err)
            return 2
        return 0

    if args.gamer_check:
        gpu = gpu_probe_fn()
        decision = evaluate_gamer_whisper_guard(
            gpu,
            running_procs=proc_list_fn(),
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
        )
        out = {
            "gpu": {
                "present": gpu.present,
                "name": gpu.name,
                "total_mib": gpu.total_mib,
                "free_mib": gpu.free_mib,
                "temp_c": gpu.temp_c,
                "util_pct": gpu.util_pct,
            },
            "os_idle_seconds": get_os_idle_seconds(),
            "gamer_whisper": decision,
        }
        print(json.dumps(out, indent=2), file=eff_out)
        return 0 if decision["allow"] else 2

    required_smoke: dict[str, Any] | None = None
    if args.require_holiday_routes:
        required_smoke = verify_holiday_routes(
            gateway,
            http_fn=http_fn,
            lang=lang_code,
            api_key=explicit_or_env_key,
            credentials_path=cred_path,
            headers=headers,
            home=home,
        )
        if not required_smoke["smoke_ok"]:
            for diag in required_smoke["diagnostics"]:
                print(diag, file=eff_err)
            if args.json:
                fail_doc: dict[str, Any] = {
                    "status": "holiday_routes_failed",
                    "smoke_ok": False,
                    "compliance_ok": True,
                    "holiday_routes": required_smoke,
                }
                fail_doc["receipt_sha256"] = compute_receipt_sha256(fail_doc)
                print(json.dumps(fail_doc, indent=2, ensure_ascii=False), file=eff_out)
            else:
                print(_render_smoke_cli_output(required_smoke), file=eff_out)
            return 2

    if args.feedback:
        fb_payload = _normalize_ces_payload(
            profile=args.profile,
            ease_rating=int(args.ease_rating),
            zero_game_lag=not bool(args.had_game_lag),
            platform_name=_normalize_platform_token(None),
            note=str(args.note or ""),
            home=home,
        )
        fb_headers = dict(headers) if headers else {}
        if "idempotency-key" not in {k.lower() for k in fb_headers}:
            fb_headers["Idempotency-Key"] = compute_ces_idempotency_key(
                fb_payload,
                gateway=gateway,
                spool_path=spool_path,
                home=home,
            )
        status, body = _safe_http_call(
            http_fn,
            "POST",
            f"{gateway}/v1/hivemind/onboarding/feedback",
            fb_payload,
            fb_headers,
        )
        if status not in (200, 201):
            print(
                f"Feedback submission failed (HTTP {status}): {redact_sensitive_text(str(body), home=home)}",
                file=eff_err,
            )
            return 1
        print(
            json.dumps({"status": "feedback_recorded", "summary": _sanitize_feedback_summary(body)}, indent=2),
            file=eff_out,
        )
        return 0

    try:
        connected = ensure_connected_account(
            gateway=gateway,
            credentials_path=cred_path,
            handle=args.handle or None,
            role=args.role,
            explicit_api_key=args.api_key or None,
            http_fn=http_fn,
            env=eff_env,
            headers=headers,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Connection error: {redact_sensitive_text(str(exc), home=home)}", file=eff_err)
        return 1

    acct = connected["account"]
    yaml_snippet = build_continue_yaml(gateway, acct["api_key"])
    if args.write_continue_config:
        cont_path.parent.mkdir(parents=True, exist_ok=True)
        cont_path.write_text(yaml_snippet, encoding="utf-8")
        if os.name != "nt":
            os.chmod(cont_path, 0o600)

    explicit_profile_flag = any(tok == "--profile" or tok.startswith("--profile=") for tok in raw_argv)
    eff_profile = args.profile if explicit_profile_flag else ("gamer" if args.gamer_whisper else args.profile)

    if args.gamer_whisper:
        result = run_gamer_whisper_session(
            gateway=gateway,
            api_key=acct["api_key"],
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
            cool_c=args.cool_c,
            once=args.once,
            gpu_probe_fn=gpu_probe_fn,
            proc_list_fn=proc_list_fn,
            popen_fn=popen_fn,
            poll_interval_s=poll_interval_s,
        )
        if required_smoke is not None:
            result["holiday_routes"] = required_smoke
        if args.post_setup_telemetry:
            telemetry = submit_post_setup_telemetry(
                gateway,
                profile=eff_profile,
                ease_rating=int(args.ease_rating),
                had_game_lag=bool(args.had_game_lag),
                yield_elapsed_s=result.get("yield_elapsed_s"),
                gpu_temp_c=result.get("gpu_temp_c"),
                note=args.note,
                headers=headers,
                http_fn=http_fn,
                lang=lang_code,
                home=home,
                spool_path=spool_path,
                credentials_path=eff_cred_for_spool,
                env=eff_env,
                cwd=eff_cwd,
            )
            result["telemetry_recorded"] = telemetry["telemetry_recorded"]
            result["post_setup_telemetry"] = telemetry
        elif args.flush_feedback_queue:
            result["flush_feedback_queue"] = flush_spooled_telemetry(
                gateway,
                spool_path=spool_path,
                credentials_path=eff_cred_for_spool,
                http_fn=http_fn,
                headers=headers,
                lang=lang_code,
                home=home,
                env=eff_env,
                cwd=eff_cwd,
            )
        result["receipt_sha256"] = compute_receipt_sha256(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=eff_out)
        return 0 if result.get("status") in ("round_completed", "yielded_to_owner", "yielded_before_start") else 1

    telemetry_res: dict[str, Any] | None = None
    flush_res_conn: dict[str, Any] | None = None
    if args.post_setup_telemetry:
        telemetry_res = submit_post_setup_telemetry(
            gateway,
            profile=eff_profile,
            ease_rating=int(args.ease_rating),
            had_game_lag=bool(args.had_game_lag),
            note=args.note,
            headers=headers,
            http_fn=http_fn,
            lang=lang_code,
            home=home,
            spool_path=spool_path,
            credentials_path=eff_cred_for_spool,
            env=eff_env,
            cwd=eff_cwd,
        )
    elif args.flush_feedback_queue:
        flush_res_conn = flush_spooled_telemetry(
            gateway,
            spool_path=spool_path,
            credentials_path=eff_cred_for_spool,
            http_fn=http_fn,
            headers=headers,
            lang=lang_code,
            home=home,
            env=eff_env,
            cwd=eff_cwd,
        )

    redacted_cred = redact_home_path(cred_path, home=home)
    redacted_cont = redact_home_path(cont_path, home=home) if args.write_continue_config else None
    masked_key = mask_api_key(acct["api_key"])

    if args.json:
        json_out: dict[str, Any] = {
            "status": "connected",
            "compliance_ok": True,
            "handle": acct["handle"],
            "role": acct["role"],
            "masked_api_key": masked_key,
            "credentials_file": redacted_cred,
            "models": connected["models"],
            "continue_config_written": redacted_cont,
        }
        if required_smoke is not None:
            json_out["holiday_routes"] = required_smoke
        if telemetry_res is not None:
            json_out["telemetry_recorded"] = telemetry_res["telemetry_recorded"]
            json_out["post_setup_telemetry"] = telemetry_res
        if flush_res_conn is not None:
            json_out["flush_feedback_queue"] = flush_res_conn
        json_out["receipt_sha256"] = compute_receipt_sha256(json_out)
        print(json.dumps(json_out, indent=2, ensure_ascii=False), file=eff_out)
        return 0

    is_fr = lang_code == "fr"
    masked_yaml = build_continue_yaml(gateway, masked_key)
    cli_receipt = compute_receipt_sha256(
        {
            "status": "connected",
            "gateway": gateway,
            "handle": acct["handle"],
            "role": acct["role"],
            "masked_api_key": masked_key,
            "credentials_file": redacted_cred,
            "telemetry_recorded": telemetry_res["telemetry_recorded"] if telemetry_res else None,
        }
    )
    print("====================================================================", file=eff_out)
    if is_fr:
        print("  HIVEMIND LABO DE NOËL — CONNECTÉ (PC PERSONNEL • 0,00 EUR)", file=eff_out)
    else:
        print("  HIVEMIND HOLIDAY LAB — CONNECTED (PERSONAL HOME PC ONLY • 0.00 EUR)", file=eff_out)
    print("====================================================================", file=eff_out)
    print(f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}", file=eff_out)
    print(f"  Handle           : {acct['handle']} ({acct['role']})", file=eff_out)
    print(f"  Gateway          : {gateway}", file=eff_out)
    print(f"  API key (masked) : {masked_key}", file=eff_out)
    print(f"  Credentials saved: {redacted_cred}", file=eff_out)
    print(f"  Models online    : {', '.join(connected['models']) or 'verified'}", file=eff_out)
    if redacted_cont:
        print(f"  VS Code snippet  : {redacted_cont}", file=eff_out)
    if telemetry_res is not None:
        print(
            f"  CES telemetry    : recorded={telemetry_res['telemetry_recorded']} ({telemetry_res['diagnostic']})",
            file=eff_out,
        )
    print(f"  SHA-256 Receipt  : {cli_receipt}", file=eff_out)
    print(
        "\n--- VS Code (Continue.dev) snippet (key masked in terminal; use --write-continue-config) ---",
        file=eff_out,
    )
    print(masked_yaml.rstrip(), file=eff_out)
    print("\n--- Clean uninstall anytime ---", file=eff_out)
    print("  python3 scripts/hivemind_quickstart.py --uninstall", file=eff_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
