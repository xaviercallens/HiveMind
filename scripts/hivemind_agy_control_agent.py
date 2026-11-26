#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind Autonomous Control Agent powered by the Google Antigravity SDK.

Connects the operator's HiveMind account (``~/.hivemind/credentials.json`` or ``HIVEMIND_API_KEY``)
and GitHub account with a stateful Google Antigravity SDK ``Agent`` to:
  1. Track real external vs. internal adoption (registered accounts by role, active trainers/producers,
     14-day GitHub unique visitors & clones, onboarding ease ratings, and zero-game-lag comfort).
  2. Monitor active campaigns (``Winter GPU Raid Noel 2026``, ``CROWDGPT-MOE-AUTORESEARCH``,
     ``CROWDGPT-MOE-8X-UPCYCLE``, ``OpenTrainer`` accepted rounds, and T4 VRAM headroom) without
     ever sending inference traffic to ``hivemind/coder``.
  3. Continuously self-improve onboarding & platform quality by diagnosing UX friction, queuing
     small-perimeter test-gated slices (``>= 95%`` coverage), and persisting session memory via
     ``LocalAgentConfig(save_dir=..., conversation_id=...)``.

Compliance & Safety Rules:
  - Personal home PCs and personal hobby/open-source projects ONLY (never corporate laptops/data).
  - Symbolic compute points have 0.00 EUR / 0.00 USD monetary value.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_OPERATOR_HANDLE = "xavier-agy-control"
DEFAULT_GITHUB_REPO = "xaviercallens/HiveMind"
NOEL_LAUNCH_DATE = date(2026, 12, 19)
NOEL_END_DATE = date(2027, 1, 3)
HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")

MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024  # 4 MiB bounded HTTP read
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
LIVE_PRODUCER_STATUSES: tuple[str, ...] = ("live", "online", "healthy", "ready")
HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
EXPECTED_MANIFEST_CORE_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/raid/status",
)
EXPECTED_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = ("hivemind_quickstart.py", "hivemind_doctor.py")
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
    "USERDNSDOMAIN",
    "USERDOMAIN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
    "ZSCALER_CERT_PATH",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = ("secret", "token", "password", "passwd", "apikey", "bearer", "aiza")

_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*{3,4}(?:[A-Za-z0-9_-]{4,}|[0-9]{1,3}m" + r"s\b)"
)
_UNSAFE_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "document" + r"\.write)\b"
)
_STATIC_FIGURE_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_RAW_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])hm(?:_live_|-)(?!\*{3,4})[A-Za-z0-9_-]{4,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
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

HttpJsonFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
HttpRawFn = Callable[..., tuple[int, str, dict[str, str]]]
GhJsonFn = Callable[[list[str]], dict[str, Any] | list[Any] | None]


class SelfImprovementAction(BaseModel):
    """Concrete, small-perimeter improvement proposed or queued by the Control Agent."""

    slice_id: str = Field(description="Short identifier such as SLICE-03 or SLICE-DYN-01")
    title: str = Field(description="Concise PR-ready title")
    priority: str = Field(description="P0, P1, or P2")
    target_metric: str = Field(description="Measured adoption or comfort KPI improved by this slice")
    hypothesis: str = Field(description="Why this change reduces user drop-off or improves reliability")
    test_gate: str = Field(description="Automated pytest and >=95% coverage verification criteria")


class CampaignStatusSummary(BaseModel):
    """Live summary of an active training or adoption campaign."""

    campaign_id: str
    phase: str
    headline_metric: str
    healthy: bool
    blocker: str = ""


class ControlCycleReport(BaseModel):
    """Structured executive output returned by an Antigravity SDK control cycle."""

    generated_utc: str
    operator_handle: str
    adoption_stage: str = Field(description="pre_launch, canary_alpha, holiday_raid, or post_raid")
    external_adoption_verdict: str
    days_until_noel_launch: int
    key_metrics: dict[str, str]
    campaigns: list[CampaignStatusSummary]
    top_friction_bottleneck: str
    next_actions: list[SelfImprovementAction]


def default_state_dir() -> Path:
    return Path.home() / ".hivemind" / "agy_control_agent"


def default_credentials_path() -> Path:
    return Path.home() / ".hivemind" / "credentials.json"


def mask_api_key(token: str) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` API key so secrets are never written to stdout or logs."""
    cleaned = (token or "").strip()
    if not cleaned:
        return "<unset>"
    if cleaned.startswith("hm_live_"):
        if len(cleaned) <= 12:
            return "hm_live_****"
        return f"hm_live_****{cleaned[-4:]}"
    if len(cleaned) <= 8:
        return "hm-****"
    prefix = cleaned[:3] if cleaned.startswith("hm-") else "key-"
    return f"{prefix}****{cleaned[-4:]}"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked-key suffixes."""
    cleaned = _STRIP_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)


def redact_home_path(target: Path | str) -> str:
    """Replace local OS home directory prefixes with ``'~'`` to prevent username/path leaks."""
    raw = str(target)
    home_str = str(Path.home())
    scrubbed = raw
    if home_str and home_str not in ("/", "\\"):
        for variant in (home_str, home_str.replace("\\", "/"), home_str.replace("/", "\\")):
            if variant and scrubbed.startswith(variant):
                suffix = scrubbed[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    scrubbed = "~" + suffix.replace("\\", "/")
                    break
    scrubbed = _HOME_PATH_RE.sub("~", scrubbed)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(text: str) -> str:
    """Mask ``hm-...`` / ``hm_live_...`` keys, Bearer tokens, and home paths in output text."""
    scrubbed = text
    home_str = str(Path.home())
    if home_str and home_str not in ("/", "\\"):
        for variant in (home_str, home_str.replace("\\", "/"), home_str.replace("/", "\\")):
            if variant and variant in scrubbed:
                scrubbed = scrubbed.replace(variant, "~")
    scrubbed = _HOME_PATH_RE.sub("~", scrubbed)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _RAW_HM_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    return scrubbed


def _atomic_write_text(target: Path, content: str, *, mode: int = 0o600) -> None:
    """Atomically write ``content`` to ``target`` via a sibling temp file with POSIX 0600 permissions."""
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


def _append_jsonl_0600(target: Path, record: dict[str, Any]) -> None:
    """Append a sanitized JSON line to ``target`` while enforcing POSIX 0600 permissions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    line = sanitize_text_for_display(json.dumps(record, sort_keys=True)) + "\n"
    if os.name == "nt":
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(line)
    os.chmod(target, 0o600)


def validate_gateway_url(url: str) -> str:
    """Validate gateway origin URL to strict loopback, ``storage.googleapis.com``, ``*.run.app``, or ``*.googleapis.com``."""
    if isinstance(url, bool) or not isinstance(url, str):
        raise ValueError("Refusing untrusted gateway URL: expected a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ValueError("Refusing untrusted gateway URL: control characters are forbidden.")
    cleaned = url.strip().rstrip("/")
    if not cleaned or any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: empty or contains invalid whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: embedded credentials are forbidden.")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: path, query, and fragment must be empty.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: port out of range.")
    host = (parts.hostname or "").strip().lower().strip("[]")
    if not host:
        raise ValueError(f"Refusing untrusted gateway URL {url!r}: missing host.")
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"Refusing untrusted gateway URL {url!r}: invalid loopback scheme.")
    elif host == "storage.googleapis.com" or _RUN_APP_HOST_RE.fullmatch(host) or _GOOGLEAPIS_HOST_RE.fullmatch(host):
        if parts.scheme != "https":
            raise ValueError(f"Refusing untrusted gateway URL {url!r}: HTTPS is required for remote gateways.")
    else:
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected an HTTPS *.run.app, *.googleapis.com, or loopback gateway."
        )
    return cleaned


def sanitize_handle(raw: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", (raw or "").strip()).strip("-.")
    if len(candidate) < 2:
        candidate = f"agy-{secrets.token_hex(3)}"
    candidate = candidate[:48]
    if not HANDLE_RE.match(candidate) or candidate.lower().startswith(("google.", "github.", "oidc.")):
        raise ValueError("Invalid HiveMind handle.")
    return candidate


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> str:
    """Enforce allowlisted host/origin, valid port, no credentials, and zero LLM inference."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError("Refusing untrusted gateway URL: request URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ValueError("Refusing untrusted gateway URL: control characters are forbidden.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned) or "\\" in cleaned:
        raise ValueError("Refusing untrusted gateway URL: whitespace and backslashes are forbidden.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("Refusing untrusted gateway URL: embedded credentials are forbidden.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ValueError(f"Zero-inference guard blocked forbidden endpoint {parts.path!r}.")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise ValueError("Zero-inference guard blocked forbidden prompt payload.")
    _ = (method or "GET").strip().upper()
    return cleaned


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
            _assert_safe_http_request(new_req.get_method(), new_req.full_url)
        return new_req


_HTTP_OPENER: urllib.request.OpenerDirector | None = None


def _get_http_opener() -> urllib.request.OpenerDirector:
    """Return a module-cached proxy-free ``OpenerDirector`` (``ProxyHandler({})``)."""
    global _HTTP_OPENER
    if _HTTP_OPENER is None:
        _HTTP_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _SafeRedirectHandler(),
        )
    return _HTTP_OPENER


def _prepare_request_headers(
    base_headers: dict[str, str],
    extra_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    req_headers = dict(base_headers)
    if extra_headers:
        for hk, hv in extra_headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ValueError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    return req_headers


def default_http_raw(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 15.0,
) -> tuple[int, str, dict[str, str]]:
    """Proxy-free raw HTTP helper (``ProxyHandler({})``) with bounded 4 MiB body read."""
    verb = (method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=None)
    req_headers = _prepare_request_headers(
        {
            "Accept": "text/html, application/json;q=0.9, text/plain;q=0.8",
            "User-Agent": "hivemind-agy-control-agent/1.0",
        },
        headers,
    )
    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    opener = _get_http_opener()
    bounded_timeout = min(max(0.2, float(timeout_s)), 60.0)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:  # noqa: S310
            raw_bytes = resp.read(MAX_HTTP_BODY_BYTES)
            resp_headers = {str(k).strip().lower(): str(v) for k, v in resp.headers.items()} if resp.headers else {}
            return int(getattr(resp, "status", 200)), raw_bytes.decode("utf-8", errors="replace"), resp_headers
    except urllib.error.HTTPError as exc:
        err_bytes = exc.read(MAX_HTTP_BODY_BYTES) if exc.fp is not None else b""
        err_headers = {str(k).strip().lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), err_bytes.decode("utf-8", errors="replace"), err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, sanitize_text_for_display(str(exc)), {}


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib JSON HTTP helper (``ProxyHandler({})``) with bounded 4 MiB body read."""
    verb = (method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    req_headers = _prepare_request_headers(
        {"Accept": "application/json", "User-Agent": "hivemind-agy-control-agent/1.0"},
        headers,
    )
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(clean_url, data=body_bytes, headers=req_headers, method=verb)
    opener = _get_http_opener()
    try:
        with opener.open(req, timeout=15.0) as resp:  # noqa: S310
            raw = resp.read(MAX_HTTP_BODY_BYTES).decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            return int(resp.status), parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as exc:
        raw = (exc.read(MAX_HTTP_BODY_BYTES) if exc.fp is not None else b"").decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": sanitize_text_for_display(raw[:300])}
        return int(exc.code), parsed if isinstance(parsed, dict) else {"data": parsed}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": sanitize_text_for_display(str(exc))}


def default_gh_json(args: list[str]) -> dict[str, Any] | list[Any] | None:
    """Run read-only ``gh`` CLI commands with ``GITHUB_TOKEN`` and ``GH_TOKEN`` unset."""
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
    try:
        proc = subprocess.run(
            ["gh", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def ensure_operator_account(
    *,
    gateway: str = DEFAULT_GATEWAY,
    preferred_handle: str = DEFAULT_OPERATOR_HANDLE,
    credentials_path: Path | None = None,
    http_fn: HttpJsonFn = default_http_json,
) -> dict[str, Any]:
    """Load, verify, or provision the operator's HiveMind account credentials safely."""
    gw = validate_gateway_url(gateway)
    cred_file = credentials_path or default_credentials_path()
    api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    saved: dict[str, Any] = {}

    if not api_key and cred_file.exists():
        try:
            loaded = json.loads(cred_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                saved = loaded
                api_key = str(loaded.get("api_key", "")).strip()
        except (OSError, json.JSONDecodeError):
            saved = {}

    created_new = False
    handle = str(saved.get("handle") or preferred_handle)

    if api_key:
        code, me_data = http_fn(
            "GET",
            f"{gw}/v1/hivemind/me",
            None,
            {"Authorization": f"Bearer {api_key}"},
        )
        if code == 200:
            handle = str(me_data.get("handle") or handle)
            return {
                "authenticated": True,
                "created": False,
                "gateway": gw,
                "handle": handle,
                "account_id": str(me_data.get("account_id", saved.get("account_id", ""))),
                "role": str(me_data.get("role", saved.get("role", "prosumer"))),
                "masked_key": mask_api_key(api_key),
                "credentials_path": redact_home_path(cred_file),
            }

    clean_handle = sanitize_handle(preferred_handle)
    reg_code, reg_data = http_fn(
        "POST",
        f"{gw}/v1/hivemind/register",
        {"handle": clean_handle, "role": "prosumer"},
        None,
    )
    if reg_code in (200, 201) and reg_data.get("api_key"):
        api_key = str(reg_data["api_key"])
        handle = str(reg_data.get("handle") or clean_handle)
        created_new = True
    else:
        fallback_handle = sanitize_handle(f"{clean_handle[:36]}-{secrets.token_hex(2)}")
        reg_code, reg_data = http_fn(
            "POST",
            f"{gw}/v1/hivemind/register",
            {"handle": fallback_handle, "role": "prosumer"},
            None,
        )
        if reg_code not in (200, 201) or not reg_data.get("api_key"):
            return {
                "authenticated": False,
                "created": False,
                "gateway": gw,
                "handle": clean_handle,
                "masked_key": "<unset>",
                "error": f"Registration returned HTTP {reg_code}",
            }
        api_key = str(reg_data["api_key"])
        handle = str(reg_data.get("handle") or fallback_handle)
        created_new = True

    record = {
        "gateway": gw,
        "handle": handle,
        "account_id": str(reg_data.get("account_id", "")),
        "role": str(reg_data.get("role", "prosumer")),
        "api_key": api_key,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_text(cred_file, json.dumps(record, indent=2) + "\n", mode=0o600)

    return {
        "authenticated": True,
        "created": created_new,
        "gateway": gw,
        "handle": handle,
        "account_id": record["account_id"],
        "role": record["role"],
        "masked_key": mask_api_key(api_key),
        "credentials_path": redact_home_path(cred_file),
    }


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


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def _extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` from ``workspace``, following worktree ``gitdir:`` and ``commondir``."""
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
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC Compliance Gate (0 HTTP calls, 0 writes, 0 raw value leaks)."""
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
            _record("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "corp")
            matched_markers.update(dns_hits)
        elif key_up in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(val_str):
                _record(f"env:{key_up}:{hit}", hit)

    ws_texts = [str(workspace)]
    try:
        resolved_ws = str(workspace.resolve())
        if resolved_ws not in ws_texts:
            ws_texts.append(resolved_ws)
    except OSError:
        resolved_ws = ""
        _ = resolved_ws
    for p_txt in ws_texts:
        for hit in _find_corporate_markers(p_txt):
            _record(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            try:
                cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_url in _extract_git_config_remote_urls(cfg_text):
                for hit in _find_corporate_markers(remote_url):
                    _record(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
        "license": "AGPL-3.0",
        "network_calls": 0,
        "file_writes": 0,
        "exit_code": 0 if compliance_ok else 2,
    }


def _invoke_http_json(http_fn: Any, method: str, url: str) -> tuple[int, dict[str, Any]]:
    _assert_safe_http_request(method, url, payload=None)
    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    try:
        if len(pos_params) <= 2 and not has_varargs:
            res = http_fn(method, url)
        else:
            res = http_fn(method, url, None, None)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": sanitize_text_for_display(str(exc))}
    if not isinstance(res, tuple) or len(res) != 2:
        raise ValueError("http_fn must return (status_code, body_dict).")
    status_code, body = res
    return int(status_code), dict(body) if isinstance(body, Mapping) else {"data": body}


def _invoke_http_raw(raw_http_fn: Any, method: str, url: str) -> tuple[int, str, dict[str, str]]:
    _assert_safe_http_request(method, url, payload=None)
    sig = inspect.signature(raw_http_fn)
    params = sig.parameters
    try:
        if "headers" in params or "timeout_s" in params:
            raw = raw_http_fn(method, url, headers=None, timeout_s=15.0)
        else:
            raw = raw_http_fn(method, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, sanitize_text_for_display(str(exc)), {}
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise ValueError("raw_http_fn must return (status_code, body_text, headers_dict).")
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
    attr_nonces = [val for _, val in _HTML_NONCE_ATTR_RE.findall(body_text)]
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
    has_static_fig = bool(_STATIC_FIGURE_RE.search(stripped_html))
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


def _inspect_models_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str], bool, list[str]]:
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
    if not coder_meta_ok:
        issues.append("models_missing_hivemind_coder")
    return len(issues) == 0, model_ids, coder_meta_ok, issues


def _inspect_manifest_parity(
    code: int,
    body: Mapping[str, Any],
    expected_sha: str | None,
) -> tuple[bool, str, bool | None, list[str], list[str]]:
    if code != 200:
        return False, "unknown", (False if expected_sha else None), [], [f"http_{code}"]
    issues: list[str] = []
    missing_routes: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("manifest_forbidden_key")
    if body.get("campaign_id") != "noel-2026-easy-onboard":
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
        missing_routes.extend(EXPECTED_MANIFEST_CORE_ROUTES)
    else:
        for req in EXPECTED_MANIFEST_CORE_ROUTES:
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


def _inspect_feedback_summary_parity(code: int, body: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if code != 200:
        return False, [f"http_{code}"]
    issues: list[str] = []
    if _has_forbidden_score_key(body):
        issues.append("summary_forbidden_key")
    if body.get("object") != "hivemind.onboarding_feedback_summary":
        issues.append("summary_invalid_object")
    total_val = _nonneg_int(body.get("total_responses"))
    if total_val is None:
        issues.append("summary_invalid_total_responses")
    has_mean = "mean_ease_rating" in body
    has_avg = "average_ease_rating" in body
    if not has_mean and not has_avg:
        issues.append("summary_missing_ease_rating")
    else:
        ease_val = body.get("mean_ease_rating", body.get("average_ease_rating"))
        if ease_val is None and has_avg:
            ease_val = body.get("average_ease_rating")
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


def verify_holiday_routes_parity(
    gateway: str,
    *,
    expected_sha: str | None = None,
    raw_http_fn: HttpRawFn = default_http_raw,
    http_fn: HttpJsonFn = default_http_json,
) -> dict[str, Any]:
    """Zero-inference Cloud Run PR #949 holiday route, CSP, manifest, and SHA parity smoke verifier."""
    gw = validate_gateway_url(gateway)
    clean_expected_sha: str | None = None
    if expected_sha is not None:
        if isinstance(expected_sha, bool) or not isinstance(expected_sha, str):
            raise ValueError("expected_sha must be a 6-64 character alphanumeric/hex string or None.")
        clean_expected_sha = _sanitize_build_sha(expected_sha)
        if clean_expected_sha == "unknown":
            raise ValueError("expected_sha must be a 6-64 character alphanumeric/hex string or None.")

    missing_or_stale: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}

    def _note_route_failure(route: str, code: int, issues: list[str]) -> None:
        failed_routes.append(route)
        if code == 404 and route not in missing_or_stale:
            missing_or_stale.append(route)
        diagnostics.extend(f"{route}: {item}" for item in issues)

    health_code, health_body = _invoke_http_json(http_fn, "GET", f"{gw}/health")
    health_ok = health_code == 200 and health_body.get("status", "ok") == "ok"
    health_issues = [] if health_ok else ([f"http_{health_code}"] if health_code != 200 else ["unhealthy_status"])
    per_route["/health"] = {"status_code": health_code, "ready": health_ok, "issues": health_issues}
    if not health_ok:
        _note_route_failure("/health", health_code, health_issues)

    models_code, models_body = _invoke_http_json(http_fn, "GET", f"{gw}/v1/models")
    models_ok, model_ids, coder_present, models_issues = _inspect_models_parity(models_code, models_body)
    per_route["/v1/models"] = {
        "status_code": models_code,
        "ready": models_ok,
        "model_ids": model_ids,
        "issues": models_issues,
    }
    if not models_ok:
        _note_route_failure("/v1/models", models_code, models_issues)

    seen_nonces: set[str] = set()
    all_html_ok = True
    unique_csp_ok = True
    html_dom_clean = True
    html_compliance_ok = True
    for route in HOLIDAY_HTML_ROUTES:
        r_code, r_body, r_hdrs = _invoke_http_raw(raw_http_fn, "GET", f"{gw}{route}")
        r_ok, r_info, r_issues = _inspect_html_parity_route(r_code, r_body, r_hdrs, seen_nonces)
        per_route[route] = {"ready": r_ok, **r_info, "issues": r_issues}
        all_html_ok = all_html_ok and r_ok
        unique_csp_ok = unique_csp_ok and bool(r_info.get("csp_ok")) and not bool(r_info.get("nonce_reused"))
        html_dom_clean = html_dom_clean and bool(r_info.get("dom_clean"))
        html_compliance_ok = html_compliance_ok and bool(r_info.get("compliance_ok"))
        if not r_ok:
            _note_route_failure(route, r_code, r_issues)

    man_code, man_body = _invoke_http_json(http_fn, "GET", f"{gw}/v1/hivemind/onboarding/manifest")
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
        _note_route_failure("/v1/hivemind/onboarding/manifest", man_code, man_issues)

    fb_code, fb_body = _invoke_http_json(http_fn, "GET", f"{gw}/v1/hivemind/onboarding/feedback/summary")
    fb_ok, fb_issues = _inspect_feedback_summary_parity(fb_code, fb_body)
    per_route["/v1/hivemind/onboarding/feedback/summary"] = {
        "status_code": fb_code,
        "ready": fb_ok,
        "issues": fb_issues,
    }
    if not fb_ok:
        _note_route_failure("/v1/hivemind/onboarding/feedback/summary", fb_code, fb_issues)

    raid_code, raid_body = _invoke_http_json(http_fn, "GET", f"{gw}/v1/hivemind/raid/status")
    raid_ok, raid_issues = _inspect_raid_status_parity(raid_code, raid_body)
    per_route["/v1/hivemind/raid/status"] = {
        "status_code": raid_code,
        "ready": raid_ok,
        "issues": raid_issues,
    }
    if not raid_ok:
        _note_route_failure("/v1/hivemind/raid/status", raid_code, raid_issues)

    unique_csp_nonces_ok = unique_csp_ok and len(seen_nonces) == len(HOLIDAY_HTML_ROUTES)
    stale_revision_detected = bool(
        missing_or_stale or (clean_expected_sha is not None and sha_match is False)
    )
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
        and not stale_revision_detected
    )

    return {
        "gateway": gw,
        "parity_ok": parity_ok,
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
        "compliance_ok": html_compliance_ok and (man_body.get("personal_pc_only") is True if man_code == 200 else False),
        "manifest_ok": man_ok,
        "live_build_sha": live_sha,
        "expected_sha": clean_expected_sha,
        "sha_match": sha_match,
        "feedback_summary_ok": fb_ok,
        "raid_status_ok": raid_ok,
        "zero_inference_verified": True,
        "per_route": per_route,
        "diagnostics": diagnostics,
    }


def _extract_online_producers_and_gpus(fleet_body: Mapping[str, Any]) -> tuple[int, int | None]:
    """Extract live producer count and distinct GPUs across real and legacy fleet payloads."""
    hiveminders = fleet_body.get("hiveminders") if isinstance(fleet_body.get("hiveminders"), list) else []
    producers = fleet_body.get("producers") if isinstance(fleet_body.get("producers"), list) else []
    hm_online = sum(
        1
        for item in hiveminders
        if isinstance(item, Mapping) and str(item.get("status", "")).lower() in LIVE_PRODUCER_STATUSES
    )
    legacy_online = sum(
        1
        for item in producers
        if isinstance(item, Mapping) and str(item.get("status", "")).lower() in LIVE_PRODUCER_STATUSES
    )
    fleet_live_int = _nonneg_int(fleet_body.get("live_hiveminders")) or 0
    distinct_gpus = _nonneg_int(fleet_body.get("distinct_physical_gpus"))
    return max(fleet_live_int, hm_online, legacy_online), distinct_gpus


def collect_platform_snapshot(
    *,
    gateway: str = DEFAULT_GATEWAY,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    today: date | None = None,
    http_fn: HttpJsonFn = default_http_json,
    gh_fn: GhJsonFn = default_gh_json,
    probe_holiday_routes: bool = False,
    raw_http_fn: HttpRawFn | None = None,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Gather read-only adoption, campaign, and satisfaction telemetry and compute deltas."""
    gw = validate_gateway_url(gateway)
    sdir = state_dir or default_state_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    history_path = sdir / "telemetry_history.jsonl"

    now_dt = datetime.now(UTC)
    ref_day = today or now_dt.date()
    days_to_launch = (NOEL_LAUNCH_DATE - ref_day).days
    if ref_day < date(2026, 12, 5):
        stage = "pre_launch"
    elif ref_day < NOEL_LAUNCH_DATE:
        stage = "canary_alpha"
    elif ref_day <= NOEL_END_DATE:
        stage = "holiday_raid"
    else:
        stage = "post_raid"

    health_code, _ = http_fn("GET", f"{gw}/health", None, None)
    models_code, models_body = http_fn("GET", f"{gw}/v1/models", None, None)
    live_code, live_body = http_fn("GET", f"{gw}/v1/hivemind/live-telemetry", None, None)
    fleet_code, fleet_body = http_fn("GET", f"{gw}/v1/hivemind/producers/fleet", None, None)
    t4_code, t4_body = http_fn("GET", f"{gw}/v1/hivemind/t4/status", None, None)
    fb_code, fb_body = http_fn("GET", f"{gw}/v1/hivemind/onboarding/feedback/summary", None, None)

    accounts_info = live_body.get("accounts") if isinstance(live_body.get("accounts"), dict) else {}
    by_role = accounts_info.get("by_role") if isinstance(accounts_info.get("by_role"), dict) else {}
    opentrainer_info = live_body.get("opentrainer") if isinstance(live_body.get("opentrainer"), dict) else {}
    models_list = models_body.get("data") if isinstance(models_body.get("data"), list) else []

    online_producers, distinct_physical_gpus = _extract_online_producers_and_gpus(
        fleet_body if isinstance(fleet_body, Mapping) else {}
    )

    repo_meta = gh_fn(["api", f"repos/{github_repo}"])
    views_meta = gh_fn(["api", f"repos/{github_repo}/traffic/views"])
    clones_meta = gh_fn(["api", f"repos/{github_repo}/traffic/clones"])
    open_prs_meta = gh_fn(["pr", "list", "-R", github_repo, "--state", "open", "--json", "number,isDraft", "--limit", "50"])

    stars = int(repo_meta.get("stargazers_count", 0)) if isinstance(repo_meta, dict) else None
    forks = int(repo_meta.get("forks_count", 0)) if isinstance(repo_meta, dict) else None
    views_uniques_14d = int(views_meta.get("uniques", 0)) if isinstance(views_meta, dict) else None
    clones_uniques_14d = int(clones_meta.get("uniques", 0)) if isinstance(clones_meta, dict) else None
    open_prs_count = len(open_prs_meta) if isinstance(open_prs_meta, list) else None

    total_accounts = accounts_info.get("total_registered_accounts")
    accepted_rounds = opentrainer_info.get("accepted_rounds_count")
    trainer_nodes = opentrainer_info.get("total_registered_nodes")
    active_trainer_nodes = _nonneg_int(opentrainer_info.get("active_trainer_nodes"))

    raw_ease = fb_body.get("mean_ease_rating", fb_body.get("average_ease_rating")) if fb_code == 200 else None
    if raw_ease is None and fb_code == 200:
        raw_ease = fb_body.get("average_ease_rating")
    ease_val = (
        raw_ease
        if (
            fb_code == 200
            and not isinstance(raw_ease, bool)
            and isinstance(raw_ease, (int, float))
            and math.isfinite(float(raw_ease))
        )
        else None
    )
    fb_by_profile = (
        dict(fb_body["by_profile"])
        if (fb_code == 200 and isinstance(fb_body.get("by_profile"), Mapping))
        else None
    )

    holiday_parity: dict[str, Any] | None = None
    stale_revision_detected = fb_code == 404
    missing_or_stale_routes: list[str] = (
        ["/v1/hivemind/onboarding/feedback/summary"] if fb_code == 404 else []
    )
    if probe_holiday_routes:
        holiday_parity = verify_holiday_routes_parity(
            gw,
            expected_sha=expected_sha,
            raw_http_fn=raw_http_fn or default_http_raw,
            http_fn=http_fn,
        )
        stale_revision_detected = bool(holiday_parity["stale_revision_detected"])
        missing_or_stale_routes = list(holiday_parity["missing_or_stale_routes"])

    prev_snapshot: dict[str, Any] | None = None
    if history_path.exists():
        lines = [ln for ln in history_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                prev_snapshot = json.loads(lines[-1])
            except json.JSONDecodeError:
                prev_snapshot = None

    def _delta(curr: int | None, prev_key: str) -> int | None:
        if curr is None or not isinstance(prev_snapshot, dict):
            return None
        prev_val = prev_snapshot.get(prev_key)
        return (int(curr) - int(prev_val)) if isinstance(prev_val, int) else None

    snapshot: dict[str, Any] = {
        "captured_utc": now_dt.isoformat(),
        "adoption_stage": stage,
        "days_until_noel_launch": days_to_launch,
        "gateway_healthy": health_code == 200 and models_code == 200,
        "live_models_count": len(models_list),
        "total_registered_accounts": int(total_accounts) if isinstance(total_accounts, int) else None,
        "accounts_by_role": by_role,
        "online_producers": online_producers,
        "distinct_physical_gpus": distinct_physical_gpus,
        "trainer_nodes": int(trainer_nodes) if isinstance(trainer_nodes, int) else None,
        "active_trainer_nodes": active_trainer_nodes,
        "accepted_opentrainer_rounds": int(accepted_rounds) if isinstance(accepted_rounds, int) else None,
        "t4_status_http": t4_code,
        "t4_vram_free_mib": (t4_body.get("gpu") or {}).get("vram_free_mib")
        if isinstance(t4_body.get("gpu"), dict)
        else t4_body.get("vram_free_mib"),
        "onboarding_feedback_deployed": fb_code == 200,
        "onboarding_feedback_total": int(fb_body.get("total_responses", 0)) if fb_code == 200 else 0,
        "onboarding_mean_ease_rating": ease_val,
        "onboarding_average_ease_rating": ease_val,
        "onboarding_zero_game_lag_ratio": fb_body.get("zero_game_lag_ratio") if fb_code == 200 else None,
        "onboarding_feedback_by_profile": fb_by_profile,
        "holiday_routes_parity": holiday_parity,
        "stale_revision_detected": stale_revision_detected,
        "missing_or_stale_routes": missing_or_stale_routes,
        "github_stars": stars,
        "github_forks": forks,
        "github_views_uniques_14d": views_uniques_14d,
        "github_clones_uniques_14d": clones_uniques_14d,
        "github_open_prs": open_prs_count,
        "deltas_since_previous": {
            "accounts": _delta(int(total_accounts) if isinstance(total_accounts, int) else None, "total_registered_accounts"),
            "accepted_rounds": _delta(int(accepted_rounds) if isinstance(accepted_rounds, int) else None, "accepted_opentrainer_rounds"),
            "github_stars": _delta(stars, "github_stars"),
            "github_views_uniques_14d": _delta(views_uniques_14d, "github_views_uniques_14d"),
        },
        "live_telemetry_http": live_code,
        "fleet_http": fleet_code,
    }

    _append_jsonl_0600(history_path, snapshot)

    return snapshot


def evaluate_readiness_gates(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate objective adoption, UX satisfaction, and campaign readiness gates."""
    avg_ease = snapshot.get("onboarding_mean_ease_rating")
    if avg_ease is None:
        avg_ease = snapshot.get("onboarding_average_ease_rating")
    zero_lag = snapshot.get("onboarding_zero_game_lag_ratio")
    fb_deployed = bool(snapshot.get("onboarding_feedback_deployed"))
    parity_blk = snapshot.get("holiday_routes_parity")
    holiday_routes_ok = (
        fb_deployed and bool(parity_blk.get("parity_ok"))
        if isinstance(parity_blk, Mapping)
        else fb_deployed
    )
    gw_ok = bool(snapshot.get("gateway_healthy"))
    by_role = snapshot.get("accounts_by_role") if isinstance(snapshot.get("accounts_by_role"), dict) else {}
    role_trainers = _nonneg_int(by_role.get("trainer")) or 0
    active_trainers = _nonneg_int(snapshot.get("active_trainer_nodes")) or 0
    online_prods = _nonneg_int(snapshot.get("online_producers")) or 0
    active_volunteer_gpus = max(role_trainers, active_trainers, online_prods)

    ease_ok = (
        not isinstance(avg_ease, bool)
        and isinstance(avg_ease, (int, float))
        and math.isfinite(float(avg_ease))
        and float(avg_ease) >= 4.2
    )
    lag_ok = (
        not isinstance(zero_lag, bool)
        and isinstance(zero_lag, (int, float))
        and math.isfinite(float(zero_lag))
        and float(zero_lag) >= 0.95
    )

    gates = {
        "gateway_and_models_online": gw_ok,
        "holiday_routes_deployed_on_cloud_run": holiday_routes_ok,
        "satisfaction_ces_target_met": ease_ok,
        "gamer_zero_lag_target_met": lag_ok,
        "external_volunteer_gpu_target_noel": active_volunteer_gpus >= 8,
    }
    blockers: list[str] = []
    if not gw_ok:
        blockers.append("Cloud Run gateway /health or /v1/models is unreachable.")
    if not holiday_routes_ok:
        if isinstance(parity_blk, Mapping) and not parity_blk.get("parity_ok"):
            missing_list = parity_blk.get("missing_or_stale_routes") or parity_blk.get("failed_routes") or []
            blockers.append(
                "Holiday route parity drift detected on Cloud Run: "
                + (", ".join(str(r) for r in missing_list) or "PR #949 parity check failed.")
            )
        else:
            blockers.append(
                "Cloud Run revision has not yet been redeployed with PR #949 (/noel, /quick, /v1/hivemind/onboarding/feedback)."
            )
    if avg_ease is None:
        blockers.append("Zero onboarding satisfaction responses recorded yet (target CES >= 4.2/5).")
    elif not ease_ok:
        blockers.append(f"Average ease_rating {avg_ease} is below 4.2/5 target.")
    if zero_lag is not None and not lag_ok:
        blockers.append(f"Zero-game-lag comfort ratio {zero_lag} is below 0.95 target.")
    if active_volunteer_gpus < 8:
        blockers.append(
            f"Active external trainer/online producer GPUs ({active_volunteer_gpus}) below Christmas target (>= 8 external friend GPUs)."
        )

    return {
        "gates": gates,
        "passed_count": sum(1 for v in gates.values() if v),
        "total_gates": len(gates),
        "blockers": blockers,
    }


def is_unsafe_shell_command(args: dict[str, Any]) -> bool:
    """Safety policy predicate blocking destructive commands and live coder inference."""
    cmd = str(args.get("CommandLine") or args.get("command") or "").strip()
    if not cmd:
        return False
    lower = cmd.lower()
    tokens = set(re.split(r"\s+", lower))
    if "rm" in tokens and ("-rf" in tokens or "-fr" in tokens):
        return True
    if "git" in tokens and "push" in tokens and ("--force" in tokens or "-f" in tokens):
        return True
    if any(t.startswith("--no-") and t.endswith("verify") for t in tokens):
        return True
    if ("curl" in tokens or "wget" in tokens) and "|" in lower and ("sh" in tokens or "bash" in tokens):
        return True
    if "/v1/chat/completions" in lower and "hivemind/coder" in lower:
        return True
    return False


def load_or_read_backlog(backlog_path: Path | None = None) -> dict[str, Any]:
    if backlog_path and backlog_path.exists():
        try:
            data = json.loads(backlog_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"slices": []}


def append_improvement_proposal(
    *,
    slice_id: str,
    title: str,
    priority: str,
    target_metric: str,
    hypothesis: str,
    test_plan: str,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist a concrete, test-gated improvement action proposed by the Control Agent."""
    if priority not in ("P0", "P1", "P2"):
        raise ValueError("priority must be P0, P1, or P2")
    if not slice_id.strip() or not title.strip():
        raise ValueError("slice_id and title must be non-empty")
    sdir = state_dir or default_state_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "slice_id": slice_id.strip(),
        "title": title.strip(),
        "priority": priority,
        "target_metric": target_metric.strip(),
        "hypothesis": hypothesis.strip(),
        "test_plan": test_plan.strip(),
    }
    out_file = sdir / "improvement_proposals.jsonl"
    _append_jsonl_0600(out_file, record)
    return record


def build_control_tools(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    credentials_path: Path | None = None,
    http_fn: HttpJsonFn = default_http_json,
    gh_fn: GhJsonFn = default_gh_json,
    probe_holiday_routes: bool = False,
    raw_http_fn: HttpRawFn | None = None,
    expected_sha: str | None = None,
) -> list[Callable[..., str]]:
    """Create the custom tool functions registered with ``LocalAgentConfig(tools=...)``."""
    sdir = state_dir or default_state_dir()

    def check_operator_account() -> str:
        """Verify and return the operator's authenticated HiveMind account status (masked key, handle, role)."""
        res = ensure_operator_account(
            gateway=gateway,
            preferred_handle=operator_handle,
            credentials_path=credentials_path,
            http_fn=http_fn,
        )
        return json.dumps(res, indent=2, sort_keys=True)

    def collect_adoption_and_campaign_snapshot() -> str:
        """Collect live HiveMind gateway, T4 training, onboarding feedback, and GitHub adoption metrics."""
        snap = collect_platform_snapshot(
            gateway=gateway,
            github_repo=github_repo,
            state_dir=sdir,
            http_fn=http_fn,
            gh_fn=gh_fn,
            probe_holiday_routes=probe_holiday_routes,
            raw_http_fn=raw_http_fn,
            expected_sha=expected_sha,
        )
        return json.dumps(snap, indent=2, sort_keys=True)

    def evaluate_campaign_readiness_gates() -> str:
        """Evaluate objective readiness gates and blockers for the Christmas Holiday adoption campaign."""
        snap = collect_platform_snapshot(
            gateway=gateway,
            github_repo=github_repo,
            state_dir=sdir,
            http_fn=http_fn,
            gh_fn=gh_fn,
            probe_holiday_routes=probe_holiday_routes,
            raw_http_fn=raw_http_fn,
            expected_sha=expected_sha,
        )
        gates = evaluate_readiness_gates(snap)
        return json.dumps(gates, indent=2, sort_keys=True)

    def inspect_backlog_and_pr_velocity() -> str:
        """Inspect the Christmas onboarding improvement backlog and open PRs on GitHub."""
        backlog = load_or_read_backlog(backlog_path)
        slices = backlog.get("slices") if isinstance(backlog.get("slices"), list) else []
        counts: dict[str, int] = {"merged": 0, "in_pr": 0, "pending": 0}
        next_pending: list[dict[str, Any]] = []
        for item in slices:
            if not isinstance(item, dict):
                continue
            st = str(item.get("status", "pending"))
            counts[st] = counts.get(st, 0) + 1
            if st == "pending" and len(next_pending) < 3:
                next_pending.append({"id": item.get("id"), "title": item.get("title"), "week": item.get("week")})
        prs = gh_fn(["pr", "list", "-R", github_repo, "--state", "open", "--json", "number,title,isDraft", "--limit", "15"])
        return json.dumps(
            {
                "slice_counts": counts,
                "next_pending_slices": next_pending,
                "open_prs_sample": prs if isinstance(prs, list) else [],
            },
            indent=2,
            sort_keys=True,
        )

    def record_self_improvement_action(
        slice_id: str,
        title: str,
        priority: str,
        target_metric: str,
        hypothesis: str,
        test_plan: str,
    ) -> str:
        """Record a concrete, test-gated UX or reliability self-improvement proposal.

        Args:
            slice_id: Identifier such as SLICE-03 or SLICE-DYN-01.
            title: Concise PR-ready title.
            priority: One of P0, P1, or P2.
            target_metric: Exact adoption or comfort metric targeted.
            hypothesis: Why this slice reduces user friction or improves model/platform quality.
            test_plan: Automated pytest and >=95% line coverage plan.
        """
        saved = append_improvement_proposal(
            slice_id=slice_id,
            title=title,
            priority=priority,
            target_metric=target_metric,
            hypothesis=hypothesis,
            test_plan=test_plan,
            state_dir=sdir,
        )
        return json.dumps(saved, indent=2, sort_keys=True)

    return [
        check_operator_account,
        collect_adoption_and_campaign_snapshot,
        evaluate_campaign_readiness_gates,
        inspect_backlog_and_pr_velocity,
        record_self_improvement_action,
    ]


SYSTEM_INSTRUCTIONS = """You are the HiveMind Autonomous Control Agent for Xavier Callens (@xaviercallens).
Your mission is to:
1. Track honest, measured adoption and satisfaction (never invent numbers; report 'n/a' when unavailable).
2. Monitor active training & holiday campaigns (Winter GPU Raid Noel 2026 starting 2026-12-19, MoE Autoresearch, OpenTrainer) WITHOUT ever sending inference prompts to hivemind/coder while GPU training is active.
3. Drive continuous, small-perimeter self-improvement focused on extreme ease of installation and connection (<60s Dev VS Code setup, <5min Windows 11 Gamer-Whisper setup, <1.5s Steam GPU yield, <=70 C quiet fans, >=95% pytest coverage).
4. Strictly enforce the Amadeus / Corporate Compliance Rule: 100% personal home PCs and personal hobby/open-source projects only (never corporate laptops or enterprise code/data), and 0.00 EUR / 0.00 USD symbolic points only.

Always call your tools (`check_operator_account`, `collect_adoption_and_campaign_snapshot`, `evaluate_campaign_readiness_gates`, `inspect_backlog_and_pr_velocity`) before producing conclusions, and call `record_self_improvement_action` for your highest-priority concrete next improvement."""


def load_saved_conversation_id(state_dir: Path) -> str | None:
    session_file = state_dir / "session_state.json"
    if not session_file.exists():
        return None
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        cid = str(data.get("conversation_id", "")).strip()
        return cid or None
    except (OSError, json.JSONDecodeError):
        return None


def save_conversation_id(state_dir: Path, conversation_id: str | None) -> None:
    if not conversation_id:
        return
    session_file = state_dir / "session_state.json"
    content = (
        json.dumps(
            {
                "conversation_id": conversation_id,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    _atomic_write_text(session_file, sanitize_text_for_display(content), mode=0o600)


def build_antigravity_agent_config(
    *,
    tools: list[Callable[..., Any]],
    state_dir: Path,
    workspace_dir: Path,
    response_schema: type[BaseModel] | None = ControlCycleReport,
    resume_session: bool = True,
    periodic_interval_s: float | None = None,
    vertex: bool = False,
    project: str | None = None,
    location: str = "us-central1",
    model: str | None = None,
) -> Any:
    """Build a verified ``google.antigravity.LocalAgentConfig`` with subagents, policies, and persistence."""
    from google.antigravity import CapabilitiesConfig, CompactionConfig, LocalAgentConfig, types
    from google.antigravity.hooks import policy
    from google.antigravity.triggers import TriggerContext, every

    save_dir = (state_dir / "conversations").resolve()
    app_data_dir = (state_dir / "app_data").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    app_data_dir.mkdir(parents=True, exist_ok=True)

    subagents = [
        types.SubagentConfig(
            name="adoption_analyst",
            description="Analyzes external vs internal adoption funnel, GitHub unique visitors, and CES satisfaction.",
            system_instructions="Analyze measured adoption & UX satisfaction data objectively. Never invent metrics.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE, types.BuiltinTools.LIST_DIR],
            ),
        ),
        types.SubagentConfig(
            name="campaign_tracker",
            description="Monitors MoE Autoresearch, OpenTrainer rounds, and Winter GPU Raid readiness.",
            system_instructions="Monitor training & holiday campaign readiness without querying hivemind/coder.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE],
            ),
        ),
        types.SubagentConfig(
            name="ux_self_improver",
            description="Designs small-perimeter onboarding & reliability slices with >=95% pytest coverage.",
            system_instructions="Propose minimal, high-impact onboarding improvements for personal home PCs only.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE, types.BuiltinTools.SEARCH_DIR],
            ),
        ),
    ]

    safety_policies = [
        policy.deny(
            "run_command",
            when=is_unsafe_shell_command,
            name="block_destructive_or_live_coder_commands",
            reason="Destructive git/shell operations and live hivemind/coder inference are forbidden.",
        ),
        policy.confirm_run_command(),
    ]

    triggers_list: list[Any] = []
    if periodic_interval_s is not None and periodic_interval_s > 0:

        async def _periodic_control_tick(ctx: TriggerContext) -> None:
            await ctx.send(
                "Run a scheduled HiveMind control check: inspect operator account, collect adoption & campaign "
                "telemetry, evaluate readiness gates, inspect the backlog, and record the next self-improvement action."
            )

        triggers_list.append(every(periodic_interval_s, _periodic_control_tick))

    conv_id = load_saved_conversation_id(state_dir) if resume_session else None

    cfg_kwargs: dict[str, Any] = {
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "tools": tools,
        "subagents": subagents,
        "policies": safety_policies,
        "triggers": triggers_list or None,
        "workspaces": [str(workspace_dir.resolve())],
        "save_dir": str(save_dir),
        "app_data_dir": str(app_data_dir),
        "conversation_id": conv_id,
        "response_schema": response_schema,
        "capabilities": CapabilitiesConfig(
            enable_subagents=True,
            agent_behavior=types.AgentBehavior.AUTONOMOUS,
            max_subagent_depth=2,
            allowed_subagents=["adoption_analyst", "campaign_tracker", "ux_self_improver"],
        ),
        "budget_config": types.BudgetConfig(
            max_model_calls=25,
            max_tool_calls=40,
            max_total_tokens=250_000,
        ),
        "compaction_config": CompactionConfig(token_threshold=80_000),
    }
    if vertex:
        cfg_kwargs["vertex"] = True
        if project:
            cfg_kwargs["project"] = project
        cfg_kwargs["location"] = location
    if model:
        cfg_kwargs["model"] = model

    return LocalAgentConfig(**cfg_kwargs)


def format_report_markdown(report: dict[str, Any]) -> str:
    """Render a ``ControlCycleReport`` dictionary into concise GitHub-flavored Markdown."""
    lines = [
        f"# HiveMind Antigravity Control Report ({report.get('generated_utc', 'n/a')})",
        "",
        f"- **Operator Account**: `{report.get('operator_handle', 'n/a')}`",
        f"- **Campaign Stage**: `{report.get('adoption_stage', 'n/a')}` (`J-{report.get('days_until_noel_launch', 'n/a')}` before 19 Dec 2026 launch)",
        f"- **Adoption Verdict**: {report.get('external_adoption_verdict', 'n/a')}",
        f"- **Top Friction Bottleneck**: {report.get('top_friction_bottleneck', 'n/a')}",
        "",
        "## Key Measured Metrics",
    ]
    for k, v in (report.get("key_metrics") or {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Active Campaigns")
    for camp in report.get("campaigns") or []:
        if isinstance(camp, dict):
            icon = "OK" if camp.get("healthy") else "WARN"
            blk = f" — Blocker: {camp.get('blocker')}" if camp.get("blocker") else ""
            lines.append(
                f"- **[{icon}] `{camp.get('campaign_id')}`** ({camp.get('phase')}): {camp.get('headline_metric')}{blk}"
            )
    lines.append("")
    lines.append("## Prioritized Self-Improvement Actions")
    for act in report.get("next_actions") or []:
        if isinstance(act, dict):
            lines.append(
                f"- **[{act.get('priority')}] `{act.get('slice_id')}` — {act.get('title')}** "
                f"(Target: *{act.get('target_metric')}*, Gate: *{act.get('test_gate')}*)"
            )
    return sanitize_text_for_display("\n".join(lines) + "\n")


async def run_sdk_audit_cycle(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    resume_session: bool = True,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Execute one full Google Antigravity SDK control & self-improvement audit turn."""
    from google.antigravity import Agent

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=ControlCycleReport,
        resume_session=resume_session,
        vertex=vertex,
        project=project,
        model=model,
    )

    prompt = (
        "Run a complete HiveMind Control & Adoption Audit cycle now:\n"
        "1. Call `check_operator_account` to confirm Xavier's HiveMind operator account.\n"
        "2. Call `collect_adoption_and_campaign_snapshot` and `evaluate_campaign_readiness_gates`.\n"
        "3. Call `inspect_backlog_and_pr_velocity`.\n"
        "4. Call `record_self_improvement_action` for the single highest-leverage next UX/adoption improvement.\n"
        "5. Return the structured ControlCycleReport grounded strictly in the measured tool outputs."
    )

    async with Agent(config=config) as agent:
        response = await agent.chat(prompt)
        structured = await response.structured_output()
        if structured is None:
            raw_text = await response.text()
            structured = json.loads(raw_text)
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))

    sanitized_json = sanitize_text_for_display(json.dumps(structured, indent=2) + "\n")
    _atomic_write_text(sdir / "latest_report.json", sanitized_json, mode=0o600)
    _atomic_write_text(sdir / "latest_report.md", format_report_markdown(structured), mode=0o600)
    return json.loads(sanitized_json)


async def run_sdk_daemon(
    *,
    interval_seconds: float = 10800.0,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> None:  # pragma: no cover - long-running daemon entrypoint
    """Run the persistent Antigravity SDK Control Agent with periodic ``every(...)`` triggers."""
    from google.antigravity import Agent

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=None,
        periodic_interval_s=interval_seconds,
        vertex=vertex,
        project=project,
        model=model,
    )
    async with Agent(config=config) as agent:
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))
        await asyncio.Event().wait()


async def run_sdk_interactive_chat(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> None:  # pragma: no cover - interactive TTY loop
    """Launch an interactive Antigravity SDK terminal session with all HiveMind control tools."""
    from google.antigravity import Agent
    from google.antigravity.utils.interactive import run_interactive_loop

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=None,
        vertex=vertex,
        project=project,
        model=model,
    )
    async with Agent(config=config) as agent:
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))
        await run_interactive_loop(agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiveMind Autonomous Control Agent (Google Antigravity SDK)."
    )
    parser.add_argument(
        "--mode",
        choices=("status", "audit", "daemon", "chat"),
        default="status",
        help="Execution mode: deterministic status check, SDK audit cycle, periodic SDK daemon, or interactive chat.",
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="HiveMind Cloud Run gateway URL.")
    parser.add_argument("--handle", default=DEFAULT_OPERATOR_HANDLE, help="Operator HiveMind handle.")
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO, help="GitHub owner/repo to track.")
    parser.add_argument("--state-dir", type=Path, default=None, help="Directory for persistent agent state.")
    parser.add_argument("--backlog", type=Path, default=None, help="Optional JSON improvement backlog file.")
    parser.add_argument("--interval-seconds", type=float, default=10800.0, help="Daemon trigger interval (default 3h).")
    parser.add_argument("--fresh-session", action="store_true", help="Start a new Antigravity conversation_id.")
    parser.add_argument("--vertex", action="store_true", help="Authenticate via Vertex AI ADC.")
    parser.add_argument("--project", default=None, help="GCP project ID when --vertex is enabled.")
    parser.add_argument("--model", default=None, help="Optional explicit model identifier override.")
    parser.add_argument(
        "--verify-holiday-routes",
        action="store_true",
        help="Run zero-inference Cloud Run PR #949 route, CSP, manifest, and SHA parity verification.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Optional expected Git build SHA (6-64 chars) verified against /v1/hivemind/onboarding/manifest.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Fail closed with exit code 2 (0 HTTP calls, 0 writes) if corporate indicators are detected locally.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compliance: dict[str, Any] | None = None
    if args.enforce_personal_pc:
        compliance = evaluate_personal_pc_compliance()
        if not compliance["compliance_ok"]:
            print(
                json.dumps(
                    {
                        "compliance": compliance,
                        "error": "Corporate indicator detected; personal home PCs & hobby/open-source projects only (0.00 EUR / 0.00 USD).",
                        "exit_code": 2,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    if args.verify_holiday_routes:
        try:
            parity = verify_holiday_routes_parity(args.gateway, expected_sha=args.expected_sha)
        except ValueError as exc:
            print(sanitize_text_for_display(f"Error: {exc}"), file=sys.stderr)
            return 2
        out_doc: dict[str, Any] = {
            "gateway": parity["gateway"],
            "holiday_routes_parity": parity,
            "stale_revision_detected": parity["stale_revision_detected"],
            "missing_or_stale_routes": parity["missing_or_stale_routes"],
            "parity_ok": parity["parity_ok"],
            "exit_code": 0 if parity["parity_ok"] else 2,
        }
        if compliance is not None:
            out_doc["compliance"] = compliance
        print(sanitize_text_for_display(json.dumps(out_doc, indent=2, sort_keys=True)))
        return 0 if parity["parity_ok"] else 2

    sdir = (args.state_dir or default_state_dir()).resolve()

    if args.mode == "status":
        acct = ensure_operator_account(gateway=args.gateway, preferred_handle=args.handle)
        snap = collect_platform_snapshot(gateway=args.gateway, github_repo=args.github_repo, state_dir=sdir)
        gates = evaluate_readiness_gates(snap)
        status_doc: dict[str, Any] = {"account": acct, "snapshot": snap, "readiness": gates}
        if compliance is not None:
            status_doc["compliance"] = compliance
        print(sanitize_text_for_display(json.dumps(status_doc, indent=2, sort_keys=True)))
        return 0 if acct.get("authenticated") and snap.get("gateway_healthy") else 1

    try:
        import google.antigravity  # noqa: F401
    except ImportError:
        print(
            "Error: google-antigravity SDK is not installed in this Python environment. "
            "Install it with: uv pip install google-antigravity",
            file=sys.stderr,
        )
        return 2

    if args.mode == "audit":
        report = asyncio.run(
            run_sdk_audit_cycle(
                gateway=args.gateway,
                operator_handle=args.handle,
                github_repo=args.github_repo,
                state_dir=sdir,
                backlog_path=args.backlog,
                resume_session=not args.fresh_session,
                vertex=args.vertex,
                project=args.project,
                model=args.model,
            )
        )
        print(format_report_markdown(report))
        return 0

    if args.mode == "daemon":  # pragma: no cover
        asyncio.run(
            run_sdk_daemon(
                interval_seconds=args.interval_seconds,
                gateway=args.gateway,
                operator_handle=args.handle,
                github_repo=args.github_repo,
                state_dir=sdir,
                backlog_path=args.backlog,
                vertex=args.vertex,
                project=args.project,
                model=args.model,
            )
        )
        return 0

    asyncio.run(  # pragma: no cover
        run_sdk_interactive_chat(
            gateway=args.gateway,
            operator_handle=args.handle,
            github_repo=args.github_repo,
            state_dir=sdir,
            backlog_path=args.backlog,
            vertex=args.vertex,
            project=args.project,
            model=args.model,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
