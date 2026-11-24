#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Concurrent 60-user Noël holiday router burst harness (`SLICE-SCALE-01`).

Simulates a realistic Noël-at-scale multi-friend holiday burst (default: 30 developers +
30 PC gamers = 60 concurrent personal-PC participants) using ONLY the Python standard
library (`concurrent.futures.ThreadPoolExecutor`, `time`, `statistics`, `hashlib`, `json`,
`urllib`, etc.) against allowed HiveMind gateway origins (`https://*.run.app` or loopback
`127.0.0.1` / `localhost` / `::1`) without ever invoking `POST /v1/chat/completions` or
prompting `hivemind/coder`:

  (a) Dev cohort (`--devs`, default 30):
      - `GET /noel` (HTML + per-request CSP nonce)
      - `GET /signup` (HTML + per-request CSP nonce)
      - `GET /v1/hivemind/onboarding/manifest`
      - `POST /v1/hivemind/register` (in local TestClient / loopback harness mode)
      - `GET /v1/models`
      - `POST /v1/hivemind/onboarding/feedback` (`profile='dev'`)

  (b) Gamer cohort (`--gamers`, default 30):
      - `GET /raid-noel` (HTML + per-request CSP nonce)
      - `GET /v1/hivemind/raid/status`
      - `GET /arena-noel` (HTML + per-request CSP nonce + HMAC-masked blind ballot)
      - `POST /v1/hivemind/arena/vote`
      - `GET /v1/hivemind/arena/summary`
      - `POST /v1/hivemind/onboarding/feedback` (`profile='gamer'`)

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import ipaddress
import json
import math
import os
import platform
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_scale_load_harness/1"
RECEIPT_OBJECT = "hivemind.noel_scale_load_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_DEVS = 30
DEFAULT_GAMERS = 30
DEFAULT_WORKERS = 60
MAX_PARTICIPANTS_PER_COHORT = 2000
MAX_WORKERS = 256

DEFAULT_P95_SLO_MS = 2500.0
DEFAULT_P99_SLO_MS = 5000.0
DEFAULT_MAX_ERROR_RATIO = 0.0
HTTP_TIMEOUT_S = 5.0
DEFAULT_RETRY_BACKOFF_S = 0.005
MAX_429_RETRIES_LIMIT = 3

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

CURATED_ARENA_TASK_IDS: tuple[str, ...] = (
    "python-sliding-median",
    "rust-frame-decoder",
    "lean4-energy-descent",
    "exoplanet-transit-bls",
)
DEV_EASE_RATINGS: tuple[int, ...] = (5, 5, 5, 5, 4)
GAMER_EASE_RATINGS: tuple[int, ...] = (5, 5, 5, 5, 5, 4)
GAMER_CHOSEN_SIDES: tuple[str, ...] = ("A", "B", "A", "tie")

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only — "
    "Amadeus / Personal-PC rule: never work PCs or corporate code (0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement — "
    "règle Amadeus / PC perso : jamais de PC ni de code d'entreprise (0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_ARENA_CARD_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"\s+data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_BURST_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,23}$")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

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
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto", re.IGNORECASE)),
)

HttpFn = Callable[..., Any]
ClockFn = Callable[[], float]
SleepFn = Callable[[float], None]


class ScaleHarnessError(ValueError):
    """Raised when scale harness parameters, URL guards, or output safety checks fail."""


@dataclass(frozen=True)
class HttpResponse:
    """Normalized zero-inference HTTP response returned by stdlib or injected callables."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    text: str = ""


@dataclass(frozen=True)
class StepObservation:
    """Single HTTP step observation recorded during a concurrent participant flow."""

    cohort: str
    participant_id: str
    method: str
    path: str
    is_html: bool
    status_code: int
    ok: bool
    latency_ms: float
    csp_nonce: str | None
    rate_limited: bool
    retried_429: int
    server_error: bool
    client_error: bool
    sqlite_locked: bool
    server_counter: int | None = None
    failure_code: str = ""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language code to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ScaleHarnessError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ScaleHarnessError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def _normalize_platform_tag(raw: str | None = None) -> str:
    source = (raw if raw is not None else platform.system()).strip().lower()
    if "darwin" in source or "mac" in source:
        return "macos"
    if "win" in source or source == "nt":
        return "windows"
    if "linux" in source:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", source)[:24]
    return cleaned if cleaned else "unknown"


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app` or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or `http(s)://127.0.0.1|localhost|[::1]`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ScaleHarnessError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ScaleHarnessError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ScaleHarnessError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ScaleHarnessError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ScaleHarnessError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ScaleHarnessError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ScaleHarnessError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ScaleHarnessError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ScaleHarnessError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or 127.0.0.1/localhost/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ScaleHarnessError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ScaleHarnessError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def is_loopback_gateway(url: str) -> bool:
    """Return True iff validated `url` targets a loopback host (`127.0.0.1`, `localhost`, `::1`)."""
    cleaned = validate_gateway_url(url)
    host = (urlsplit(cleaned).hostname or "").strip().lower().strip("[]")
    return host in ALLOWED_LOOPBACK_HOSTS


def simulated_participant_ip(participant_index: int) -> str:
    """Deterministically map `participant_index` (`>= 0`) into loopback `127.0.0.0/8` for per-PC buckets."""
    if isinstance(participant_index, bool) or not isinstance(participant_index, int) or participant_index < 0:
        raise ScaleHarnessError("participant_index must be a non-negative integer.")
    offset = 256 + (participant_index % 16_000_000)
    return str(ipaddress.IPv4Address((127 << 24) + offset))


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check (zero env values ever transmitted)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = environ if environ is not None else os.environ
    upper_env = {str(k).upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()}

    indicators: list[str] = []
    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")

    dns_domain = upper_env.get("USERDNSDOMAIN", "")
    if dns_domain:
        indicators.append("env:USERDNSDOMAIN")

    for env_key in ("USERDOMAIN", "COMPUTERNAME", "HOSTNAME", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        val = upper_env.get(env_key, "")
        if not val:
            continue
        if env_key == "USERDOMAIN" and val.upper() == upper_env.get("COMPUTERNAME", "").upper():
            continue
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(val):
                indicators.append(f"env:{env_key}:{marker_name}")
                break

    compliance_ok = len(indicators) == 0
    return {
        "compliance_ok": compliance_ok,
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "zero_llm_inference": True,
        "corporate_indicators": sorted(set(indicators)),
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "active_notice": COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN,
    }


def _assert_zero_inference_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowed gateway origin and block any inference completion or model prompt call."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ScaleHarnessError(f"Unsupported HTTP method {verb!r}; expected GET, HEAD, or POST.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ScaleHarnessError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ScaleHarnessError("HTTP JSON payload must be a mapping when provided.")
        if "messages" in payload or "prompt" in payload:
            raise ScaleHarnessError(
                "Zero-inference invariant violated: prompt/messages payloads are strictly forbidden."
            )
        serialized = json.dumps(dict(payload), sort_keys=True).lower()
        if "hivemind/coder" in serialized:
            raise ScaleHarnessError(
                "Zero-inference invariant violated: prompting hivemind/coder is strictly forbidden."
            )
    return origin


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON, Markdown, or CLI output leaks zero PII, raw keys, home paths, IPs, or `'score': <digit>`."""
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise ScaleHarnessError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def default_http_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Direct, proxy-free stdlib HTTP helper returning `(status_code, body_dict, lower_headers)`."""
    _assert_zero_inference_request(method, url, payload)
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9",
        "User-Agent": "hivemind-noel-scale-load-harness/1.0",
    }
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload)).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        err_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw_err[:240]}
        norm_err = parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err}
        return int(exc.code), norm_err, err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": str(reason)[:200]}, {}

    content_type = resp_headers.get("content-type", "")
    if not raw.strip():
        return status_code, {"content_type": content_type, "html_length": 0}, resp_headers
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (
            status_code,
            {"content_type": content_type, "html_length": len(raw), "html": raw},
            resp_headers,
        )
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed}), resp_headers


def _normalize_http_result(raw_result: Any) -> HttpResponse:
    """Normalize `HttpResponse`, 3-tuple `(status, body, headers)`, or 2-tuple `(status, body)`."""
    if isinstance(raw_result, HttpResponse):
        return raw_result
    if not isinstance(raw_result, tuple) or len(raw_result) not in (2, 3):
        raise ScaleHarnessError(
            "Injectable http_fn must return HttpResponse, (status, body), or (status, body, headers)."
        )
    status_code = int(raw_result[0])
    raw_body = raw_result[1]
    raw_headers = raw_result[2] if len(raw_result) == 3 else None

    headers: dict[str, str] = {}
    if isinstance(raw_headers, Mapping):
        headers.update({str(k).lower(): str(v) for k, v in raw_headers.items()})

    text = ""
    if isinstance(raw_body, Mapping):
        body = dict(raw_body)
        embedded_headers = body.get("headers")
        if isinstance(embedded_headers, Mapping):
            for hk, hv in embedded_headers.items():
                headers.setdefault(str(hk).lower(), str(hv))
        if isinstance(body.get("csp"), str):
            headers.setdefault("content-security-policy", body["csp"])
        if isinstance(body.get("html"), str):
            text = body["html"]
    elif isinstance(raw_body, str):
        text = raw_body
        try:
            parsed = json.loads(raw_body)
            body = parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            body = {"html_length": len(raw_body), "html": raw_body}
    else:
        body = {"data": raw_body}

    return HttpResponse(status_code=status_code, body=body, headers=headers, text=text)


def _contains_sqlite_lock_signal(resp: HttpResponse) -> bool:
    """Return True if any error body or text signals a SQLite lock or OperationalError."""
    chunks: list[str] = []
    if resp.text and resp.status_code >= 400:
        chunks.append(resp.text[:512])
    if resp.status_code == 0 or resp.status_code >= 400:
        chunks.append(json.dumps(resp.body, sort_keys=True, default=str)[:512])
    return any(_SQLITE_LOCK_RE.search(chunk) for chunk in chunks if chunk)


def _invoke_http_callable(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    *,
    clock_fn: ClockFn = time.perf_counter,
) -> tuple[HttpResponse, float, bool]:
    """Invoke `http_fn` with zero-inference verification, measuring latency in milliseconds."""
    _assert_zero_inference_request(method, url, payload)
    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

    dict_payload = dict(payload) if payload is not None else None
    dict_headers = {str(k): str(v) for k, v in headers.items()} if headers is not None else None

    t0 = float(clock_fn())
    try:
        if has_varargs or len(positional) >= 4:
            raw = http_fn(method.upper(), url, dict_payload, dict_headers)
        elif len(positional) == 3:
            raw = http_fn(method.upper(), url, dict_payload)
        else:
            raw = http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
        detail = str(exc)[:240]
        resp = HttpResponse(
            status_code=0,
            body={"network_error": type(exc).__name__, "detail": detail},
            headers={},
            text=detail,
        )
        return resp, elapsed_ms, bool(_SQLITE_LOCK_RE.search(detail))
    except Exception as exc:  # noqa: BLE001 - surface in-process sqlite3 OperationalError cleanly as a lock/500 finding
        if isinstance(exc, ScaleHarnessError):
            raise
        elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
        detail = f"{type(exc).__name__}: {exc}"[:240]
        is_lock = bool(_SQLITE_LOCK_RE.search(detail))
        resp = HttpResponse(
            status_code=500,
            body={"server_exception": type(exc).__name__, "sqlite_locked": is_lock},
            headers={},
            text="",
        )
        return resp, elapsed_ms, is_lock

    elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
    resp = _normalize_http_result(raw)
    return resp, elapsed_ms, _contains_sqlite_lock_signal(resp)


def extract_csp_nonce(headers: Mapping[str, str], html_text: str = "") -> str | None:
    """Extract and cross-verify the per-request CSP nonce from headers and HTML `<style>`/`<script>` tags."""
    csp_header = ""
    for k, v in headers.items():
        if str(k).lower() == "content-security-policy":
            csp_header = str(v)
            break

    header_nonces = set(_CSP_HEADER_NONCE_RE.findall(csp_header)) if csp_header else set()
    html_nonces = set(_HTML_ATTR_NONCE_RE.findall(html_text)) if html_text else set()

    if len(header_nonces) > 1 or len(html_nonces) > 1:
        return None
    header_nonce = next(iter(header_nonces), None)
    html_nonce = next(iter(html_nonces), None)

    if header_nonce is not None and html_nonce is not None:
        return header_nonce if header_nonce == html_nonce else None
    if header_nonce is not None:
        if html_text and ("<style" in html_text or "<script" in html_text):
            return None
        return header_nonce
    return html_nonce


def extract_arena_ballots(html_text: str) -> dict[str, str]:
    """Extract `{task_id: ballot_token}` pairs from rendered `/arena-noel` HTML task cards."""
    if not html_text:
        return {}
    return {task_id: ballot for task_id, ballot in _ARENA_CARD_BALLOT_RE.findall(html_text)}


def compute_percentile_ms(samples_ms: Sequence[float], percentile: float) -> float:
    """Compute exact deterministic percentile over finite non-negative latency samples (ms)."""
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not math.isfinite(float(percentile))
        or not (0.0 <= float(percentile) <= 100.0)
    ):
        raise ScaleHarnessError("percentile must be a finite number between 0.0 and 100.0.")

    cleaned: list[float] = []
    for item in samples_ms:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ScaleHarnessError("All latency samples must be numeric (not bool).")
        fval = float(item)
        if not math.isfinite(fval) or fval < 0.0:
            raise ScaleHarnessError("All latency samples must be finite and >= 0.0.")
        cleaned.append(fval)

    if not cleaned:
        return 0.0
    ordered = sorted(cleaned)
    n = len(ordered)
    if n == 1:
        return round(ordered[0], 3)
    pct = float(percentile)
    if pct == 50.0:
        return round(float(statistics.median(ordered)), 3)
    pos = (pct / 100.0) * float(n - 1)
    lo_idx = int(math.floor(pos))
    hi_idx = int(math.ceil(pos))
    if lo_idx == hi_idx:
        return round(ordered[lo_idx], 3)
    weight = pos - float(lo_idx)
    interpolated = ordered[lo_idx] * (1.0 - weight) + ordered[hi_idx] * weight
    return round(interpolated, 3)


def _execute_step(
    *,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    cohort: str,
    participant_id: str,
    method: str,
    gateway: str,
    path: str,
    is_html: bool,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    validator: Callable[[HttpResponse], tuple[bool, int | None, str]],
    max_429_retries: int,
    retry_backoff_s: float,
) -> tuple[StepObservation, HttpResponse]:
    """Execute one participant HTTP request with optional bounded HTTP 429 retry and payload validation."""
    url = f"{gateway}{path}"
    retried = 0
    total_latency_ms = 0.0
    sqlite_locked_any = False

    while True:
        resp, elapsed_ms, is_locked = _invoke_http_callable(
            http_fn,
            method,
            url,
            payload,
            headers,
            clock_fn=clock_fn,
        )
        total_latency_ms += elapsed_ms
        sqlite_locked_any = sqlite_locked_any or is_locked
        if resp.status_code == 429 and retried < max_429_retries:
            retried += 1
            retry_hdr = resp.headers.get("retry-after", "")
            wait_s = retry_backoff_s
            try:
                parsed_wait = float(retry_hdr)
                if math.isfinite(parsed_wait) and parsed_wait >= 0.0:
                    wait_s = min(retry_backoff_s, parsed_wait)
            except ValueError:
                wait_s = retry_backoff_s
            if wait_s > 0.0:
                sleep_fn(wait_s)
            continue
        break

    code = resp.status_code
    rate_limited = code == 429
    server_error = code == 0 or code >= 500
    client_error = 400 <= code < 500 and not rate_limited
    csp_nonce: str | None = None
    server_counter: int | None = None
    failure_code = ""

    if 200 <= code < 300:
        valid_body, server_counter, failure_code = validator(resp)
        if is_html:
            csp_nonce = extract_csp_nonce(resp.headers, resp.text)
            if csp_nonce is None and valid_body:
                valid_body = False
                failure_code = "missing_or_mismatched_csp_nonce"
        ok = valid_body and not sqlite_locked_any
    else:
        ok = False
        failure_code = f"http_{code}"

    obs = StepObservation(
        cohort=cohort,
        participant_id=participant_id,
        method=method,
        path=path,
        is_html=is_html,
        status_code=code,
        ok=ok,
        latency_ms=round(total_latency_ms, 3),
        csp_nonce=csp_nonce,
        rate_limited=rate_limited,
        retried_429=retried,
        server_error=server_error,
        client_error=client_error,
        sqlite_locked=sqlite_locked_any,
        server_counter=server_counter,
        failure_code=failure_code,
    )
    return obs, resp


def _validate_html_page(resp: HttpResponse) -> tuple[bool, int | None, str]:
    html_len = len(resp.text) if resp.text else int(resp.body.get("html_length") or 0)
    if html_len <= 0:
        return False, None, "empty_html_body"
    return True, None, ""


def _validate_manifest_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    if (
        body.get("object") != "hivemind.onboarding.manifest"
        or body.get("status") != "ok"
        or body.get("personal_pc_only") is not True
    ):
        return False, None, "invalid_manifest_contract"
    return True, None, ""


def _validate_register_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    acct_id = body.get("account_id")
    api_key = body.get("api_key")
    if (
        not isinstance(acct_id, str)
        or not acct_id.strip()
        or not isinstance(api_key, str)
        or not api_key.startswith(("hm-", "hm_live_"))
    ):
        return False, None, "invalid_register_response"
    return True, 1, ""


def _validate_models_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    data = resp.body.get("data")
    if not isinstance(data, list) or len(data) == 0:
        return False, None, "invalid_models_list"
    return True, len(data), ""


def _validate_raid_status_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    if resp.body.get("object") != "hivemind.raid_noel.status":
        return False, None, "invalid_raid_status_contract"
    return True, None, ""


def _validate_arena_vote_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_votes = body.get("total_votes")
    if (
        body.get("recorded") is not True
        or isinstance(total_votes, bool)
        or not isinstance(total_votes, int)
        or total_votes < 1
    ):
        return False, None, "invalid_arena_vote_response"
    return True, total_votes, ""


def _validate_arena_summary_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_votes = body.get("total_votes")
    if (
        body.get("object") != "hivemind.blind_ab_arena.summary"
        or isinstance(total_votes, bool)
        or not isinstance(total_votes, int)
        or total_votes < 0
    ):
        return False, None, "invalid_arena_summary_response"
    return True, total_votes, ""


def _validate_feedback_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_resp = body.get("total_responses")
    if (
        body.get("object") != "hivemind.onboarding_feedback_summary"
        or isinstance(total_resp, bool)
        or not isinstance(total_resp, int)
        or total_resp < 1
    ):
        return False, None, "invalid_feedback_response"
    return True, total_resp, ""


def _run_dev_participant(
    *,
    dev_idx: int,
    participant_ip_idx: int,
    gateway: str,
    include_register: bool,
    burst_tag: str,
    platform_tag: str,
    lang_code: str,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    max_429_retries: int,
    retry_backoff_s: float,
    start_gate: threading.Event,
) -> list[StepObservation]:
    """Execute the 5-step (remote) or 6-step (local harness) Dev cohort flow."""
    start_gate.wait(timeout=2.0)
    participant_id = f"dev-{dev_idx:04d}"
    headers = {
        "X-Forwarded-For": simulated_participant_ip(participant_ip_idx),
        "Accept-Language": "fr-FR,fr;q=0.9" if lang_code == "fr" else "en-US,en;q=0.9",
    }
    common = {
        "http_fn": http_fn,
        "clock_fn": clock_fn,
        "sleep_fn": sleep_fn,
        "cohort": "dev",
        "participant_id": participant_id,
        "gateway": gateway,
        "headers": headers,
        "max_429_retries": max_429_retries,
        "retry_backoff_s": retry_backoff_s,
    }
    observations: list[StepObservation] = []

    obs_noel, _ = _execute_step(
        **common,
        method="GET",
        path="/noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_noel)

    obs_signup, _ = _execute_step(
        **common,
        method="GET",
        path="/signup",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_signup)

    obs_manifest, _ = _execute_step(
        **common,
        method="GET",
        path="/v1/hivemind/onboarding/manifest",
        is_html=False,
        payload=None,
        validator=_validate_manifest_resp,
    )
    observations.append(obs_manifest)

    if include_register:
        reg_payload = {
            "handle": f"noel-dev-{burst_tag}-{dev_idx:04d}",
            "role": "prosumer",
        }
        obs_reg, _ = _execute_step(
            **common,
            method="POST",
            path="/v1/hivemind/register",
            is_html=False,
            payload=reg_payload,
            validator=_validate_register_resp,
        )
        observations.append(obs_reg)

    obs_models, _ = _execute_step(
        **common,
        method="GET",
        path="/v1/models",
        is_html=False,
        payload=None,
        validator=_validate_models_resp,
    )
    observations.append(obs_models)

    fb_payload = {
        "profile": "dev",
        "ease_rating": DEV_EASE_RATINGS[dev_idx % len(DEV_EASE_RATINGS)],
        "zero_game_lag": True,
        "platform": platform_tag,
    }
    obs_fb, _ = _execute_step(
        **common,
        method="POST",
        path="/v1/hivemind/onboarding/feedback",
        is_html=False,
        payload=fb_payload,
        validator=_validate_feedback_resp,
    )
    observations.append(obs_fb)
    return observations


def _run_gamer_participant(
    *,
    gamer_idx: int,
    participant_ip_idx: int,
    gateway: str,
    platform_tag: str,
    lang_code: str,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    max_429_retries: int,
    retry_backoff_s: float,
    start_gate: threading.Event,
) -> list[StepObservation]:
    """Execute the 6-step Gamer cohort flow (zero inference, blind A/B vote + CES feedback)."""
    start_gate.wait(timeout=2.0)
    participant_id = f"gamer-{gamer_idx:04d}"
    headers = {
        "X-Forwarded-For": simulated_participant_ip(participant_ip_idx),
        "Accept-Language": "fr-FR,fr;q=0.9" if lang_code == "fr" else "en-US,en;q=0.9",
    }
    common = {
        "http_fn": http_fn,
        "clock_fn": clock_fn,
        "sleep_fn": sleep_fn,
        "cohort": "gamer",
        "participant_id": participant_id,
        "gateway": gateway,
        "headers": headers,
        "max_429_retries": max_429_retries,
        "retry_backoff_s": retry_backoff_s,
    }
    observations: list[StepObservation] = []

    obs_raid_html, _ = _execute_step(
        **common,
        method="GET",
        path="/raid-noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_raid_html)

    obs_raid_status, _ = _execute_step(
        **common,
        method="GET",
        path="/v1/hivemind/raid/status",
        is_html=False,
        payload=None,
        validator=_validate_raid_status_resp,
    )
    observations.append(obs_raid_status)

    obs_arena_html, resp_arena_html = _execute_step(
        **common,
        method="GET",
        path="/arena-noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_arena_html)

    ballots = extract_arena_ballots(resp_arena_html.text)
    task_id = CURATED_ARENA_TASK_IDS[gamer_idx % len(CURATED_ARENA_TASK_IDS)]
    chosen_side = GAMER_CHOSEN_SIDES[gamer_idx % len(GAMER_CHOSEN_SIDES)]
    vote_payload: dict[str, Any] = {
        "task_id": task_id,
        "chosen_side": chosen_side,
        "profile": "gamer",
    }
    if task_id in ballots:
        vote_payload["ballot_token"] = ballots[task_id]

    obs_vote, _ = _execute_step(
        **common,
        method="POST",
        path="/v1/hivemind/arena/vote",
        is_html=False,
        payload=vote_payload,
        validator=_validate_arena_vote_resp,
    )
    observations.append(obs_vote)

    obs_summary, _ = _execute_step(
        **common,
        method="GET",
        path="/v1/hivemind/arena/summary",
        is_html=False,
        payload=None,
        validator=_validate_arena_summary_resp,
    )
    observations.append(obs_summary)

    fb_payload = {
        "profile": "gamer",
        "ease_rating": GAMER_EASE_RATINGS[gamer_idx % len(GAMER_EASE_RATINGS)],
        "zero_game_lag": True,
        "platform": platform_tag,
    }
    obs_fb, _ = _execute_step(
        **common,
        method="POST",
        path="/v1/hivemind/onboarding/feedback",
        is_html=False,
        payload=fb_payload,
        validator=_validate_feedback_resp,
    )
    observations.append(obs_fb)
    return observations


def _validate_int_range(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ScaleHarnessError(f"{name} must be an integer between {lo} and {hi}.")
    return int(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScaleHarnessError(f"{name} must be a positive finite number.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ScaleHarnessError(f"{name} must be a positive finite number.")
    return fval


def _summarize_observations(rows: Sequence[StepObservation]) -> dict[str, Any]:
    total = len(rows)
    ok_count = sum(1 for r in rows if r.ok)
    err_count = total - ok_count
    rl_count = sum(1 for r in rows if r.rate_limited)
    srv_err = sum(1 for r in rows if r.server_error)
    latencies = [r.latency_ms for r in rows]
    return {
        "requests": total,
        "ok_count": ok_count,
        "error_count": err_count,
        "rate_limited_count": rl_count,
        "server_error_count": srv_err,
        "p50_latency_ms": compute_percentile_ms(latencies, 50.0),
        "p95_latency_ms": compute_percentile_ms(latencies, 95.0),
        "p99_latency_ms": compute_percentile_ms(latencies, 99.0),
    }


def run_noel_scale_burst(
    gateway: str = DEFAULT_GATEWAY,
    *,
    devs: int = DEFAULT_DEVS,
    gamers: int = DEFAULT_GAMERS,
    workers: int = DEFAULT_WORKERS,
    local_harness: bool = False,
    include_register: bool | None = None,
    p95_slo_ms: float = DEFAULT_P95_SLO_MS,
    p99_slo_ms: float = DEFAULT_P99_SLO_MS,
    max_error_ratio: float = DEFAULT_MAX_ERROR_RATIO,
    tolerate_rate_limits: bool = False,
    max_429_retries: int = 0,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    burst_tag: str = "noel2026",
    platform_name: str | None = None,
    http_fn: HttpFn = default_http_request,
    clock_fn: ClockFn = time.perf_counter,
    sleep_fn: SleepFn = time.sleep,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute the concurrent Noël-at-scale router burst and compute exact deterministic SLO metrics."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    devs_n = _validate_int_range(devs, "devs", 0, MAX_PARTICIPANTS_PER_COHORT)
    gamers_n = _validate_int_range(gamers, "gamers", 0, MAX_PARTICIPANTS_PER_COHORT)
    if devs_n + gamers_n < 1:
        raise ScaleHarnessError("At least 1 participant is required (`devs + gamers >= 1`).")
    workers_n = _validate_int_range(workers, "workers", 1, MAX_WORKERS)
    retries_n = _validate_int_range(max_429_retries, "max_429_retries", 0, MAX_429_RETRIES_LIMIT)

    clean_p95_slo = _validate_positive_float(p95_slo_ms, "p95_slo_ms")
    clean_p99_slo = _validate_positive_float(p99_slo_ms, "p99_slo_ms")
    if clean_p99_slo < clean_p95_slo:
        raise ScaleHarnessError("p99_slo_ms must be >= p95_slo_ms.")

    if (
        isinstance(max_error_ratio, bool)
        or not isinstance(max_error_ratio, (int, float))
        or not math.isfinite(float(max_error_ratio))
        or not (0.0 <= float(max_error_ratio) <= 1.0)
    ):
        raise ScaleHarnessError("max_error_ratio must be a finite number between 0.0 and 1.0.")
    clean_max_err_ratio = float(max_error_ratio)

    if (
        isinstance(retry_backoff_s, bool)
        or not isinstance(retry_backoff_s, (int, float))
        or not math.isfinite(float(retry_backoff_s))
        or float(retry_backoff_s) < 0.0
    ):
        raise ScaleHarnessError("retry_backoff_s must be a finite non-negative number.")
    clean_backoff = float(retry_backoff_s)

    if isinstance(local_harness, bool) is False or isinstance(tolerate_rate_limits, bool) is False:
        raise ScaleHarnessError("local_harness and tolerate_rate_limits must be booleans.")
    if include_register is not None and not isinstance(include_register, bool):
        raise ScaleHarnessError("include_register must be a boolean or None.")

    clean_tag = (burst_tag or "").strip().lower()
    if not _BURST_TAG_RE.fullmatch(clean_tag):
        raise ScaleHarnessError("burst_tag must be 2..24 lowercase alphanumeric/hyphen/underscore chars.")

    platform_tag = _normalize_platform_tag(platform_name)
    loopback_mode = is_loopback_gateway(clean_gw)
    effective_register = (
        bool(local_harness or loopback_mode) if include_register is None else bool(include_register)
    )
    if effective_register and not loopback_mode and not local_harness:
        raise ScaleHarnessError(
            "POST /v1/hivemind/register is restricted to loopback or explicit local_harness mode."
        )

    compliance = evaluate_personal_pc_compliance(environ=environ, lang=lang_code)
    participants_total = devs_n + gamers_n
    expected_dev_steps = 6 if effective_register else 5
    expected_requests_total = devs_n * expected_dev_steps + gamers_n * 6

    if not compliance["compliance_ok"]:
        corp_msg = (
            "Exécution refusée : environnement d'entreprise détecté localement (règle Amadeus / PC perso uniquement)."
            if is_fr
            else "Execution refused: corporate environment indicator detected locally (Amadeus / Personal-PC rule)."
        )
        refused_report: dict[str, Any] = {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "lang": lang_code,
            "gateway": clean_gw,
            "local_harness_mode": bool(local_harness or loopback_mode),
            "include_register": effective_register,
            "participants_total": participants_total,
            "devs_count": devs_n,
            "gamers_count": gamers_n,
            "workers_count": workers_n,
            "requests_total": 0,
            "expected_requests_total": expected_requests_total,
            "ok_count": 0,
            "error_count": 0,
            "error_ratio": 1.0,
            "rate_limited_count": 0,
            "retried_429_count": 0,
            "server_error_count": 0,
            "client_error_count": 0,
            "sqlite_lock_error_count": 0,
            "min_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "p95_slo_ms": clean_p95_slo,
            "p99_slo_ms": clean_p99_slo,
            "max_error_ratio": clean_max_err_ratio,
            "tolerate_rate_limits": tolerate_rate_limits,
            "html_renders_total": 0,
            "unique_csp_nonces_count": 0,
            "unique_csp_nonces_ratio": 0.0,
            "registrations_recorded": 0,
            "expected_registrations": devs_n if effective_register else 0,
            "registrations_conserved": False,
            "arena_votes_recorded": 0,
            "expected_arena_votes": gamers_n,
            "arena_votes_conserved": False,
            "server_arena_total_votes": 0,
            "ces_responses_recorded": 0,
            "expected_ces_responses": participants_total,
            "ces_responses_conserved": False,
            "server_ces_total_responses": 0,
            "zero_inference_verified": True,
            "scale_slo_ok": False,
            "http_status_counts": {},
            "cohorts": {"dev": _summarize_observations(()), "gamer": _summarize_observations(())},
            "routes": [],
            "diagnostics": [corp_msg],
            "compliance": compliance,
        }
        return _attach_receipt(refused_report)

    effective_workers = min(workers_n, participants_total)
    start_gate = threading.Event()
    all_observations: list[StepObservation] = []

    task_specs: list[tuple[str, int, int]] = []
    max_len = max(devs_n, gamers_n)
    for idx in range(max_len):
        if idx < devs_n:
            task_specs.append(("dev", idx, idx))
        if idx < gamers_n:
            task_specs.append(("gamer", idx, devs_n + idx))

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures: list[concurrent.futures.Future[list[StepObservation]]] = []
        for cohort_kind, cohort_idx, ip_idx in task_specs:
            if cohort_kind == "dev":
                futures.append(
                    executor.submit(
                        _run_dev_participant,
                        dev_idx=cohort_idx,
                        participant_ip_idx=ip_idx,
                        gateway=clean_gw,
                        include_register=effective_register,
                        burst_tag=clean_tag,
                        platform_tag=platform_tag,
                        lang_code=lang_code,
                        http_fn=http_fn,
                        clock_fn=clock_fn,
                        sleep_fn=sleep_fn,
                        max_429_retries=retries_n,
                        retry_backoff_s=clean_backoff,
                        start_gate=start_gate,
                    )
                )
            else:
                futures.append(
                    executor.submit(
                        _run_gamer_participant,
                        gamer_idx=cohort_idx,
                        participant_ip_idx=ip_idx,
                        gateway=clean_gw,
                        platform_tag=platform_tag,
                        lang_code=lang_code,
                        http_fn=http_fn,
                        clock_fn=clock_fn,
                        sleep_fn=sleep_fn,
                        max_429_retries=retries_n,
                        retry_backoff_s=clean_backoff,
                        start_gate=start_gate,
                    )
                )
        start_gate.set()
        for fut in futures:
            all_observations.extend(fut.result())

    requests_total = len(all_observations)
    ok_count = sum(1 for r in all_observations if r.ok)
    error_count = requests_total - ok_count
    rate_limited_count = sum(1 for r in all_observations if r.rate_limited)
    retried_429_count = sum(r.retried_429 for r in all_observations)
    server_error_count = sum(1 for r in all_observations if r.server_error)
    client_error_count = sum(1 for r in all_observations if r.client_error)
    sqlite_lock_error_count = sum(1 for r in all_observations if r.sqlite_locked)

    error_ratio = round(error_count / requests_total, 6) if requests_total > 0 else 1.0
    non_rl_errors = max(0, error_count - rate_limited_count)
    effective_err_ratio = (
        round(non_rl_errors / requests_total, 6)
        if (tolerate_rate_limits and requests_total > 0)
        else error_ratio
    )

    latencies_ms = [r.latency_ms for r in all_observations]
    min_latency_ms = round(min(latencies_ms), 3) if latencies_ms else 0.0
    mean_latency_ms = round(float(statistics.fmean(latencies_ms)), 3) if latencies_ms else 0.0
    max_latency_ms = round(max(latencies_ms), 3) if latencies_ms else 0.0
    p50_latency_ms = compute_percentile_ms(latencies_ms, 50.0)
    p95_latency_ms = compute_percentile_ms(latencies_ms, 95.0)
    p99_latency_ms = compute_percentile_ms(latencies_ms, 99.0)

    html_rows = [r for r in all_observations if r.is_html]
    html_renders_total = len(html_rows)
    valid_nonces = [r.csp_nonce for r in html_rows if r.csp_nonce is not None]
    unique_csp_nonces_count = len(set(valid_nonces))
    unique_csp_nonces_ratio = (
        round(unique_csp_nonces_count / html_renders_total, 6) if html_renders_total > 0 else 1.0
    )

    reg_rows = [r for r in all_observations if r.path == "/v1/hivemind/register" and r.ok]
    registrations_recorded = len(reg_rows)
    expected_registrations = devs_n if effective_register else 0
    reg_rl = sum(1 for r in all_observations if r.path == "/v1/hivemind/register" and r.rate_limited)
    registrations_conserved = (
        registrations_recorded == (expected_registrations - reg_rl)
        if tolerate_rate_limits
        else registrations_recorded == expected_registrations
    )

    vote_ok_rows = [r for r in all_observations if r.path == "/v1/hivemind/arena/vote" and r.ok]
    arena_summary_counters = [
        r.server_counter
        for r in all_observations
        if r.path in ("/v1/hivemind/arena/vote", "/v1/hivemind/arena/summary")
        and r.server_counter is not None
    ]
    server_arena_total_votes = max(arena_summary_counters) if arena_summary_counters else 0
    arena_votes_recorded = (
        min(len(vote_ok_rows), server_arena_total_votes) if vote_ok_rows else 0
    )
    vote_rl = sum(1 for r in all_observations if r.path == "/v1/hivemind/arena/vote" and r.rate_limited)
    expected_arena_votes = gamers_n
    arena_votes_conserved = (
        arena_votes_recorded == (expected_arena_votes - vote_rl)
        if tolerate_rate_limits
        else arena_votes_recorded == expected_arena_votes
    )

    ces_ok_rows = [r for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.ok]
    ces_counters = [r.server_counter for r in ces_ok_rows if r.server_counter is not None]
    server_ces_total_responses = max(ces_counters) if ces_counters else 0
    ces_responses_recorded = (
        min(len(ces_ok_rows), server_ces_total_responses) if ces_ok_rows else 0
    )
    ces_rl = sum(
        1 for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.rate_limited
    )
    expected_ces_responses = participants_total
    ces_responses_conserved = (
        ces_responses_recorded == (expected_ces_responses - ces_rl)
        if tolerate_rate_limits
        else ces_responses_recorded == expected_ces_responses
    )

    status_counts_raw: dict[int, int] = {}
    for r in all_observations:
        status_counts_raw[r.status_code] = status_counts_raw.get(r.status_code, 0) + 1
    http_status_counts = {str(code): status_counts_raw[code] for code in sorted(status_counts_raw)}

    canonical_route_order: tuple[tuple[str, str, str], ...] = (
        ("dev", "GET", "/noel"),
        ("dev", "GET", "/signup"),
        ("dev", "GET", "/v1/hivemind/onboarding/manifest"),
        *(
            (("dev", "POST", "/v1/hivemind/register"),)
            if effective_register
            else ()
        ),
        ("dev", "GET", "/v1/models"),
        ("dev", "POST", "/v1/hivemind/onboarding/feedback"),
        ("gamer", "GET", "/raid-noel"),
        ("gamer", "GET", "/v1/hivemind/raid/status"),
        ("gamer", "GET", "/arena-noel"),
        ("gamer", "POST", "/v1/hivemind/arena/vote"),
        ("gamer", "GET", "/v1/hivemind/arena/summary"),
        ("gamer", "POST", "/v1/hivemind/onboarding/feedback"),
    )
    route_summaries: list[dict[str, Any]] = []
    for cohort_name, method_name, route_path in canonical_route_order:
        matching = [
            r
            for r in all_observations
            if r.cohort == cohort_name and r.method == method_name and r.path == route_path
        ]
        if not matching:
            continue
        route_summaries.append(
            {
                "cohort": cohort_name,
                "method": method_name,
                "path": route_path,
                **_summarize_observations(matching),
            }
        )

    cohorts_summary = {
        "dev": _summarize_observations([r for r in all_observations if r.cohort == "dev"]),
        "gamer": _summarize_observations([r for r in all_observations if r.cohort == "gamer"]),
    }

    diagnostics: list[str] = []
    if sqlite_lock_error_count > 0:
        diagnostics.append(
            f"Conflit de verrouillage SQLite détecté ({sqlite_lock_error_count} erreur(s)) : vérifiez le mode WAL et le verrou d'écriture `RouterStore`."
            if is_fr
            else f"SQLite lock contention detected ({sqlite_lock_error_count} error(s)): verify WAL mode and `RouterStore` write serialization."
        )
    if server_error_count > 0:
        diagnostics.append(
            f"Défaillance passerelle détectée ({server_error_count} réponse(s) HTTP 5xx ou erreur réseau) — échec fermé du SLO."
            if is_fr
            else f"Gateway degradation detected ({server_error_count} HTTP 5xx or network error response(s)) — failing SLO closed."
        )
    if rate_limited_count > 0:
        diagnostics.append(
            f"Limitation de débit HTTP 429 observée ({rate_limited_count} réponse(s)) : vérifiez `trusted_proxy_hops=1` ou activez `--tolerate-rate-limits` sur IP partagée."
            if is_fr
            else f"HTTP 429 rate limiting observed ({rate_limited_count} response(s)): verify `trusted_proxy_hops=1` behind proxy or pass `--tolerate-rate-limits` on shared NAT."
        )
    if effective_err_ratio > clean_max_err_ratio:
        diagnostics.append(
            f"Taux d'erreur mesuré ({effective_err_ratio:.4f}) supérieur au plafond SLO ({clean_max_err_ratio:.4f})."
            if is_fr
            else f"Measured error ratio ({effective_err_ratio:.4f}) exceeds SLO ceiling ({clean_max_err_ratio:.4f})."
        )
    if p95_latency_ms > clean_p95_slo or p99_latency_ms > clean_p99_slo:
        diagnostics.append(
            f"Budget de latence SLO dépassé (p95={p95_latency_ms:.2f} ms / plafond {clean_p95_slo:.2f} ms, p99={p99_latency_ms:.2f} ms / plafond {clean_p99_slo:.2f} ms)."
            if is_fr
            else f"Latency SLO budget exceeded (p95={p95_latency_ms:.2f} ms vs {clean_p95_slo:.2f} ms cap, p99={p99_latency_ms:.2f} ms vs {clean_p99_slo:.2f} ms cap)."
        )
    if unique_csp_nonces_ratio < 1.0:
        diagnostics.append(
            f"Violation CSP : ratio de nonces uniques = {unique_csp_nonces_ratio:.6f} ({unique_csp_nonces_count}/{html_renders_total} pages HTML)."
            if is_fr
            else f"CSP nonce uniqueness violation: unique_csp_nonces_ratio = {unique_csp_nonces_ratio:.6f} ({unique_csp_nonces_count}/{html_renders_total} HTML renders)."
        )
    if not arena_votes_conserved:
        diagnostics.append(
            f"Non-conservation des votes Arène A/B : {arena_votes_recorded} enregistré(s) sur {expected_arena_votes} attendu(s)."
            if is_fr
            else f"Blind A/B Arena vote conservation mismatch: {arena_votes_recorded} recorded vs {expected_arena_votes} expected."
        )
    if not ces_responses_conserved:
        diagnostics.append(
            f"Non-conservation des retours CES : {ces_responses_recorded} enregistré(s) sur {expected_ces_responses} attendu(s)."
            if is_fr
            else f"CES onboarding feedback conservation mismatch: {ces_responses_recorded} recorded vs {expected_ces_responses} expected."
        )

    scale_slo_ok = bool(
        requests_total == expected_requests_total
        and requests_total > 0
        and sqlite_lock_error_count == 0
        and server_error_count == 0
        and effective_err_ratio <= clean_max_err_ratio
        and p95_latency_ms <= clean_p95_slo
        and p99_latency_ms <= clean_p99_slo
        and unique_csp_nonces_ratio == 1.0
        and registrations_conserved
        and arena_votes_conserved
        and ces_responses_conserved
    )

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "lang": lang_code,
        "gateway": clean_gw,
        "local_harness_mode": bool(local_harness or loopback_mode),
        "include_register": effective_register,
        "participants_total": participants_total,
        "devs_count": devs_n,
        "gamers_count": gamers_n,
        "workers_count": effective_workers,
        "requests_total": requests_total,
        "expected_requests_total": expected_requests_total,
        "ok_count": ok_count,
        "error_count": error_count,
        "error_ratio": error_ratio,
        "rate_limited_count": rate_limited_count,
        "retried_429_count": retried_429_count,
        "server_error_count": server_error_count,
        "client_error_count": client_error_count,
        "sqlite_lock_error_count": sqlite_lock_error_count,
        "min_latency_ms": min_latency_ms,
        "mean_latency_ms": mean_latency_ms,
        "max_latency_ms": max_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "p95_slo_ms": clean_p95_slo,
        "p99_slo_ms": clean_p99_slo,
        "max_error_ratio": clean_max_err_ratio,
        "tolerate_rate_limits": tolerate_rate_limits,
        "html_renders_total": html_renders_total,
        "unique_csp_nonces_count": unique_csp_nonces_count,
        "unique_csp_nonces_ratio": unique_csp_nonces_ratio,
        "registrations_recorded": registrations_recorded,
        "expected_registrations": expected_registrations,
        "registrations_conserved": registrations_conserved,
        "arena_votes_recorded": arena_votes_recorded,
        "expected_arena_votes": expected_arena_votes,
        "arena_votes_conserved": arena_votes_conserved,
        "server_arena_total_votes": server_arena_total_votes,
        "ces_responses_recorded": ces_responses_recorded,
        "expected_ces_responses": expected_ces_responses,
        "ces_responses_conserved": ces_responses_conserved,
        "server_ces_total_responses": server_ces_total_responses,
        "zero_inference_verified": True,
        "scale_slo_ok": scale_slo_ok,
        "http_status_counts": http_status_counts,
        "cohorts": cohorts_summary,
        "routes": route_summaries,
        "diagnostics": diagnostics,
        "compliance": compliance,
    }
    return _attach_receipt(core_report)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render bilingual FR/EN Markdown scale burst report with compliance notice and SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    is_fr = lang_code == "fr"
    slo_ok = bool(report.get("scale_slo_ok"))
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    notice = str(comp.get("notice_fr") if is_fr else comp.get("notice_en") or COMPLIANCE_NOTICE_EN)

    if is_fr:
        title = "# HiveMind Noël — Rapport de Charge Concurrente Routeur (`SLICE-SCALE-01`)"
        verdict = "CONFORME SLO (`scale_slo_ok=True`)" if slo_ok else "ÉCHEC SLO (`scale_slo_ok=False`)"
        sec_metrics = "## 1. Métriques Déterministes de Charge"
        sec_routes = "## 2. Répartition par Cohorte et Route Zéro-Inférence"
        sec_diag = "## 3. Diagnostics Actionnables"
        hdr_route = "| Cohorte | Méthode | Route | Requêtes | OK | Erreurs | 429 | p50 (ms) | p95 (ms) | p99 (ms) |"
    else:
        title = "# HiveMind Noël — Concurrent Router Burst Scale Report (`SLICE-SCALE-01`)"
        verdict = "PASS (`scale_slo_ok=True`)" if slo_ok else "FAIL (`scale_slo_ok=False`)"
        sec_metrics = "## 1. Deterministic Scale & Conservation Metrics"
        sec_routes = "## 2. Zero-Inference Cohort & Route Breakdown"
        sec_diag = "## 3. Actionable Diagnostics"
        hdr_route = "| Cohort | Method | Route | Requests | OK | Errors | 429 | p50 (ms) | p95 (ms) | p99 (ms) |"

    lines: list[str] = [
        title,
        "",
        f"> **Compliance ({FINANCIAL_PROMISE})**: {notice}",
        "",
        f"- **Verdict**: **{verdict}**",
        f"- **Gateway**: `{report.get('gateway')}` (`local_harness_mode={report.get('local_harness_mode')}`)",
        f"- **Participants**: `{report.get('participants_total')}` (`devs={report.get('devs_count')}`, `gamers={report.get('gamers_count')}`, `workers={report.get('workers_count')}`)",
        "",
        sec_metrics,
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| `requests_total` | `{report.get('requests_total')}` |",
        f"| `error_count` | `{report.get('error_count')}` |",
        f"| `error_ratio` | `{report.get('error_ratio')}` |",
        f"| `rate_limited_count` | `{report.get('rate_limited_count')}` |",
        f"| `sqlite_lock_error_count` | `{report.get('sqlite_lock_error_count')}` |",
        f"| `p50_latency_ms` | `{report.get('p50_latency_ms')}` |",
        f"| `p95_latency_ms` | `{report.get('p95_latency_ms')}` |",
        f"| `p99_latency_ms` | `{report.get('p99_latency_ms')}` |",
        f"| `unique_csp_nonces_ratio` | `{report.get('unique_csp_nonces_ratio')}` (`{report.get('unique_csp_nonces_count')}/{report.get('html_renders_total')}`) |",
        f"| `arena_votes_recorded` | `{report.get('arena_votes_recorded')}` / `{report.get('expected_arena_votes')}` |",
        f"| `ces_responses_recorded` | `{report.get('ces_responses_recorded')}` / `{report.get('expected_ces_responses')}` |",
        f"| `zero_inference_verified` | `{report.get('zero_inference_verified')}` |",
        f"| `scale_slo_ok` | `{report.get('scale_slo_ok')}` |",
    ]

    routes = report.get("routes")
    if isinstance(routes, Sequence) and routes:
        lines.extend(
            [
                "",
                sec_routes,
                "",
                hdr_route,
                "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for row in routes:
            lines.append(
                f"| `{row.get('cohort')}` | `{row.get('method')}` | `{row.get('path')}` "
                f"| `{row.get('requests')}` | `{row.get('ok_count')}` | `{row.get('error_count')}` "
                f"| `{row.get('rate_limited_count')}` | `{row.get('p50_latency_ms')}` "
                f"| `{row.get('p95_latency_ms')}` | `{row.get('p99_latency_ms')}` |"
            )

    diags = report.get("diagnostics")
    if isinstance(diags, Sequence) and diags:
        lines.extend(["", sec_diag, ""])
        for item in diags:
            lines.append(f"- {item}")

    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact bilingual FR/EN terminal summary with `[OK]`/`[ERROR]` status badges."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    is_fr = lang_code == "fr"
    slo_ok = bool(report.get("scale_slo_ok"))
    badge = "[OK]" if slo_ok else "[ERROR]"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    notice = str(comp.get("notice_fr") if is_fr else comp.get("notice_en") or COMPLIANCE_NOTICE_EN)

    if is_fr:
        headline = (
            f"{badge} Rafale Noël Routeur ({report.get('participants_total')} participants : "
            f"{report.get('devs_count')} devs + {report.get('gamers_count')} gamers, "
            f"{report.get('requests_total')} requêtes, scale_slo_ok={slo_ok})"
        )
    else:
        headline = (
            f"{badge} Noël Router Scale Burst ({report.get('participants_total')} participants: "
            f"{report.get('devs_count')} devs + {report.get('gamers_count')} gamers, "
            f"{report.get('requests_total')} requests, scale_slo_ok={slo_ok})"
        )

    lines = [
        headline,
        f"  gateway={report.get('gateway')} errors={report.get('error_count')} ({report.get('error_ratio')}) "
        f"rl_429={report.get('rate_limited_count')} sqlite_locks={report.get('sqlite_lock_error_count')}",
        f"  latency_ms: p50={report.get('p50_latency_ms')} p95={report.get('p95_latency_ms')} p99={report.get('p99_latency_ms')}",
        f"  csp_nonces_ratio={report.get('unique_csp_nonces_ratio')} "
        f"arena_votes={report.get('arena_votes_recorded')} ces_responses={report.get('ces_responses_recorded')}",
        f"  compliance: {notice}",
    ]
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  receipt_sha256={report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_scale_load_harness.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noël concurrent 60-user holiday router burst harness "
            "(zero LLM inference, 100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="HiveMind gateway origin URL (https://*.run.app or loopback 127.0.0.1/localhost/::1).",
    )
    parser.add_argument(
        "--devs",
        type=int,
        default=DEFAULT_DEVS,
        help="Concurrent Dev cohort participants (default: 30).",
    )
    parser.add_argument(
        "--gamers",
        type=int,
        default=DEFAULT_GAMERS,
        help="Concurrent Gamer cohort participants (default: 30).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="ThreadPoolExecutor max_workers concurrency (default: 60).",
    )
    parser.add_argument(
        "--local-harness",
        action="store_true",
        help="Explicitly enable local TestClient harness mode (includes POST /v1/hivemind/register).",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip POST /v1/hivemind/register even when targeting a loopback gateway.",
    )
    parser.add_argument(
        "--p95-slo-ms",
        type=float,
        default=DEFAULT_P95_SLO_MS,
        help="Maximum allowed p95 latency in milliseconds (default: 2500.0).",
    )
    parser.add_argument(
        "--p99-slo-ms",
        type=float,
        default=DEFAULT_P99_SLO_MS,
        help="Maximum allowed p99 latency in milliseconds (default: 5000.0).",
    )
    parser.add_argument(
        "--max-error-ratio",
        type=float,
        default=DEFAULT_MAX_ERROR_RATIO,
        help="Maximum allowed HTTP error ratio in [0.0, 1.0] (default: 0.0).",
    )
    parser.add_argument(
        "--tolerate-rate-limits",
        action="store_true",
        help="Exclude HTTP 429 rate-limit responses from SLO failure ratio on shared-NAT runs.",
    )
    parser.add_argument(
        "--retry-429",
        type=int,
        default=0,
        help="Bounded retry attempts (0..3) when an endpoint answers HTTP 429.",
    )
    parser.add_argument(
        "--burst-tag",
        default="noel2026",
        help="Deterministic lowercase alphanumeric handle tag for local harness registrations.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--cli", action="store_true", help="Emit compact terminal summary output.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default).")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    clock_fn: ClockFn = time.perf_counter,
    sleep_fn: SleepFn = time.sleep,
    environ: Mapping[str, str] | None = None,
) -> int:
    """CLI entrypoint returning `0` when `scale_slo_ok` holds or `2` (fail-closed) otherwise."""
    args = build_parser().parse_args(argv)
    try:
        lang_code = normalize_lang(args.lang)
        inc_reg: bool | None = False if args.skip_register else (True if args.local_harness else None)
        report = run_noel_scale_burst(
            args.gateway,
            devs=args.devs,
            gamers=args.gamers,
            workers=args.workers,
            local_harness=bool(args.local_harness),
            include_register=inc_reg,
            p95_slo_ms=args.p95_slo_ms,
            p99_slo_ms=args.p99_slo_ms,
            max_error_ratio=args.max_error_ratio,
            tolerate_rate_limits=bool(args.tolerate_rate_limits),
            max_429_retries=args.retry_429,
            burst_tag=args.burst_tag,
            http_fn=http_fn,
            clock_fn=clock_fn,
            sleep_fn=sleep_fn,
            environ=environ,
            lang=lang_code,
        )
        if args.json:
            print(_assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False)))
        elif args.cli:
            print(render_cli_report(report))
        else:
            print(render_markdown_report(report))
        return 0 if report["scale_slo_ok"] else 2
    except ScaleHarnessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
