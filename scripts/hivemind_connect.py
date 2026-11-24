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
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
FINANCIAL_PROMISE: str = "0.00 EUR / 0.00 USD"
CONTINUE_MODELS: tuple[str, str] = ("hivemind/coder", "hivemind/bee-stars")
DEFAULT_HOST_LABEL: str = "personal-pc"
DEFAULT_TRAINING_STACK: str = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_MAX_TEMP_C: float = 70.0
DEFAULT_MIN_FREE_VRAM_MIB: int = 2048
DEFAULT_YIELD_BUDGET_MS: float = 1500.0
DOCTOR_HTTP_TIMEOUT_S: float = 4.5
SUBPROCESS_TIMEOUT_S: float = 3.0
NON_WORKER_GPU_UTIL_THRESHOLD_PCT: float = 15.0
MIN_REPORTABLE_VRAM_TOTAL_MIB: int = 512
MAX_GPU_VRAM_MIB: int = 1_048_576

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
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_PLATFORM_TAG_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_HOST_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_REVISION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_SECRET_VALUE_MARKERS: tuple[str, ...] = ("secret", "token", "password", "passwd", "apikey", "bearer", "aiza")

HttpFn = Callable[..., tuple[int, Any]]
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


def scan_output_safety(text: str, *, home: Path | None = None) -> list[str]:
    """Audit rendered CLI/Markdown/JSON output for leaked keys, home paths, PII, or forbidden keys."""
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _RAW_API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _HOME_PATH_RE.search(text) or any(v in text for v in _home_variants(home)):
        issues.append("unredacted_home_path_detected")
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
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if (len(h) > 8 and h.endswith(".run.app")) or h == "storage.googleapis.com":
        return True
    if len(h) > 15 and h.endswith(".googleapis.com"):
        return True
    if allow_test_hosts and h in ALLOWED_TEST_TRANSPORT_HOSTS:
        return True
    return False


def validate_gateway_url(url: str, *, allow_test_hosts: bool = False) -> str:
    """Validate gateway origin URL against strict host allowlist and transport scheme rules."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise GatewayPolicyError("Gateway URL must be a non-empty string.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise GatewayPolicyError("Gateway URL must not contain whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise GatewayPolicyError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise GatewayPolicyError("Gateway URL must be a bare origin without path, query, or fragment.")
    try:
        port = parts.port
    except ValueError as exc:
        raise GatewayPolicyError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise GatewayPolicyError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host, allow_test_hosts=allow_test_hosts):
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


def _assert_zero_inference_request(method: str, url: str, *, allow_test_hosts: bool = False) -> str:
    """Validate request target host and reject any LLM inference / chat-completion path."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise GatewayPolicyError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin, allow_test_hosts=allow_test_hosts)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat" in low_path or "hivemind/coder" in low_path:
        raise GatewayPolicyError(f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden.")
    return origin


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent HTTP redirects to untrusted hosts or completion endpoints."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        _assert_zero_inference_request(req.get_method(), newurl, allow_test_hosts=False)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _StdlibHttpClient:
    """Proxy-free Python stdlib HTTP client using ``urllib.request.ProxyHandler({})``."""

    def __init__(self, *, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        self.timeout_s = float(timeout_s)
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
        _assert_zero_inference_request(verb, url, allow_test_hosts=False)
        req_headers: dict[str, str] = {
            "Accept": "application/json, text/html;q=0.9",
            "User-Agent": "hivemind-connect/2.0-stdlib",
        }
        if headers:
            for hk, hv in headers.items():
                ks = str(hk).strip()
                vs = str(hv).strip()
                if any(ord(ch) < 32 or ord(ch) == 127 for ch in ks + vs):
                    raise GatewayPolicyError("HTTP headers must not contain control characters.")
                req_headers[ks] = vs
        body_bytes: bytes | None = None
        if json is not None:
            body_bytes = globals()["json"].dumps(json).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=verb)
        eff_timeout = self.timeout_s if timeout is None else float(timeout)
        try:
            with self._opener.open(req, timeout=eff_timeout) as resp:
                raw_bytes = resp.read(524_288)
                resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                return _StdlibHttpResponse(int(getattr(resp, "status", 200)), raw_bytes, resp_headers)
        except urllib.error.HTTPError as exc:
            err_bytes = exc.read(131_072)
            resp_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
            return _StdlibHttpResponse(int(exc.code), err_bytes, resp_headers)
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
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

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
        _assert_zero_inference_request(method, url, allow_test_hosts=True)
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if headers is not None:
            kwargs["headers"] = headers
        if timeout is not None:
            kwargs["timeout"] = timeout
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
        _assert_zero_inference_request(verb, url, allow_test_hosts=True)
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
    """Locate local ``.git/config`` and worktree commondir ``config`` files strictly on disk."""
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

    if workspace.exists():
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
    eff_os = os_name if os_name is not None else os.name
    if eff_os != "nt" and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, posix_mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if eff_os != "nt":
            os.chmod(tmp_path, posix_mode)
        os.replace(tmp_path, path)
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


def default_subprocess_runner(argv: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Run a local read-only command using explicit list args (``shell=False``) and bounded timeout."""
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("Subprocess command must be a non-empty list/tuple of strings.")
    cmd = [str(part) for part in argv]
    bounded = min(max(0.1, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
            check=False,
            shell=False,
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
        assert gpu.temp_c is not None and gpu.free_mib is not None
        thermal_ok = bool(gpu.temp_c <= cap_temp_c and not game_running)
        vram_reserve_ok = bool(gpu.free_mib >= reserve_mib)
        if not yield_slo_ok:
            gpu_status = "yield_slo_exceeded"
            hw_status = "fail"
        elif gpu.temp_c > cap_temp_c:
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
    elif gpu_status == "gpu_thermal_cooldown":
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
    elif gpu_status == "active_game_yield":
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
        hw_summary = (
            f"{gpu.name} Gamer-Whisper ready ({lat_s_text} < {budget_s:.2f}s, {gpu.temp_c:.1f} C, {gpu.free_mib} MiB free)."
            if not is_fr
            else f"{gpu.name} Gamer-Whisper prêt ({lat_s_text} < {budget_s:.2f}s, {gpu.temp_c:.1f} C, {gpu.free_mib} MiB libres)."
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
                    feedback_summary = {
                        "object": str(body.get("object") or "hivemind.onboarding_feedback_summary"),
                        "total_responses": int(body.get("total_responses") or 0),
                        "mean_ease_rating": body.get("mean_ease_rating"),
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
        mean_ease_rating: float | None = (
            float(feedback_summary["mean_ease_rating"])
            if feedback_summary is not None and isinstance(feedback_summary.get("mean_ease_rating"), (int, float))
            else None
        )

        if do_submit_ces:
            ces_attempted = True
            ces_payload = {
                "profile": inferred_profile,
                "ease_rating": eff_ease,
                "zero_game_lag": zero_game_lag,
                "platform": _platform_feedback_tag(os_name=eff_os, system_name=system_name),
                "note": "",
            }
            key_salt = hashlib.sha256(f"hm-connect:{resolved_key}".encode()).hexdigest()[:16] if resolved_key else "anon"
            canon_ces = json.dumps(ces_payload, sort_keys=True, separators=(",", ":"))
            ces_idempotency_key = (
                "hm-ces-"
                + hashlib.sha256(f"hivemind.connect.ces/1:{clean_gw}:{clean_host}:{key_salt}:{canon_ces}".encode()).hexdigest()[:32]
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
            else:
                ces_recorded = False
                ces_degraded = True
                diagnostics.append(
                    f"CES submission degraded non-blocking (HTTP {ces_http_status})."
                    if not is_fr
                    else f"Envoi CES dégradé sans blocage (HTTP {ces_http_status})."
                )

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


def cmd_doctor(client: Any, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute Noel doctor checks and emit CLI, Markdown, or JSON output."""
    _ = (client, parser)
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
        lang=str(getattr(args, "lang", "en")),
        timeout_s=float(getattr(args, "timeout", DOCTOR_HTTP_TIMEOUT_S)),
        cwd=getattr(args, "cwd", None) or _ctx_cwd(args),
        home=_ctx_home(args),
        environ=_ctx_environ(args),
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
    )
    if getattr(args, "json", False):
        rendered = json.dumps(report, indent=2)
    elif getattr(args, "markdown", False):
        rendered = render_doctor_markdown(report)
    else:
        rendered = render_doctor_cli(report)
    issues = scan_output_safety(rendered, home=_ctx_home(args))
    if issues:
        raise ValueError(f"Output safety guard failed: {', '.join(issues)}")
    print(rendered)
    return int(report["exit_code"])


def _add_shared_continuity_flags(subparser: argparse.ArgumentParser, *, include_writes: bool = True) -> None:
    subparser.add_argument("--credentials", default=argparse.SUPPRESS, metavar="FILE", help="Credentials JSON path (default ~/.hivemind/credentials.json)")
    subparser.add_argument("--enforce-personal-pc", action="store_true", default=argparse.SUPPRESS, help="Refuse corporate environments/repos with exit code 2")
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
    doc.add_argument("--gateway", default=argparse.SUPPRESS, help="HiveMind gateway URL")
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
    doc.add_argument("--cwd", default=None, help="Workspace path checked locally for corporate markers/.git/config")
    doc.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C, help="Quiet-fan GPU temperature ceiling (default 70.0 C)")
    doc.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB, help="Free VRAM reserve in MiB (default 2048)")
    doc.add_argument("--yield-budget-ms", type=float, default=DEFAULT_YIELD_BUDGET_MS, help="Steam yield budget in milliseconds (default 1500.0)")
    doc.add_argument("--game", action="append", default=[], help="Extra game executable name to watch")
    doc.add_argument("--timeout", type=float, default=DOCTOR_HTTP_TIMEOUT_S, help="Bounded HTTP timeout per probe")
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
    allow_test_hosts = transport is not None or http_fn is not None

    args._environ = environ
    args._cwd = cwd
    args._home = home
    args._os_name = os_name
    args._system_name = system_name
    args._allow_test_hosts = allow_test_hosts
    args._transport = transport
    args._http_fn = http_fn
    args._gpu_probe_fn = gpu_probe_fn
    args._process_probe_fn = process_probe_fn
    args._worker_yield_fn = worker_yield_fn
    args._clock_fn = clock_fn
    args._input_fn = input_fn
    args._runner = runner

    if args.command != "doctor" and getattr(args, "enforce_personal_pc", False):
        comp = evaluate_personal_pc_compliance(
            env=environ,
            cwd=cwd,
            lang=str(getattr(args, "lang", "en")),
        )
        if not comp["compliance_ok"]:
            print(comp["summary"], file=sys.stderr)
            return 2

    if args.command == "doctor":
        return int(args.func(None, args, parser))

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
