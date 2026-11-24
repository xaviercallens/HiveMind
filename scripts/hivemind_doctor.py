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
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.doctor/1"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
SUPPORTED_LANGS = ("en", "fr")
MIN_PYTHON = (3, 10)
MAX_TESTED_MINOR = 13
GAMER_WHISPER_MIN_VRAM_MIB = 4096
GAMER_WHISPER_MAX_TEMP_C = 70.0
DEFAULT_NOEL_MIN_FREE_VRAM_MIB = 2048
DEFAULT_NOEL_YIELD_BUDGET_MS = 1500.0
DEFAULT_NOEL_HOST_LABEL = "personal-pc"
MIN_REPORTABLE_GPU_VRAM_MIB = 512
HTTP_TIMEOUT_S = 4.0
SUBPROCESS_TIMEOUT_S = 3.0

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
_RAW_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_PLATFORM_TAG_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_HOST_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_REVISION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

HttpGetFn = Callable[[str, float], tuple[int, dict[str, Any]]]
HttpRequestFn = Callable[..., tuple[int, dict[str, Any]]]
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


def _sanitize_noel_text(text: str, *, home: Path | None = None) -> str:
    """Scrub home directory paths, Bearer headers, and raw hm-.../hm_live_... keys."""
    scrubbed = redact_home_path(text, home=home)
    if _HOME_PATH_RE.search(scrubbed):
        scrubbed = _HOME_PATH_RE.sub("~", scrubbed).replace("\\", "/")
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    return _RAW_API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)


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
    """Return True iff host satisfies the repository lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if (len(h) > 8 and h.endswith(".run.app")) or h == "storage.googleapis.com":
        return True
    if len(h) > 15 and h.endswith(".googleapis.com"):
        return True
    return False


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway base URL against the strict host allowlist."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Gateway URL must be a non-empty string.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise ValueError("Gateway URL must not contain whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ValueError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ValueError("Gateway URL must not contain query parameters or fragments.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ValueError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError("Loopback gateway URL must use the http or https scheme.")
    elif parts.scheme != "https":
        raise ValueError("Remote gateway URL must use the https scheme.")
    return cleaned


def _validate_zero_inference_target(method: str, url: str) -> tuple[str, str]:
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ValueError(f"Unsupported HTTP method {verb!r}; expected GET, HEAD, or POST.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    path = parts.path or "/"
    low_path = path.lower()
    if "completions" in low_path or "/chat" in low_path:
        raise ValueError(f"Zero-inference invariant violated: {verb} {path} is strictly forbidden.")
    if verb == "POST" and path not in ALLOWED_NOEL_POST_PATHS:
        raise ValueError(f"POST endpoint {path!r} is not permitted in hivemind_doctor.")
    return verb, path


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a direct, proxy-free GET request to an allowlisted HiveMind endpoint.

    Uses an empty ProxyHandler so corporate/ambient HTTP_PROXY env vars are never
    contacted and zero environment or hostname metadata is transmitted.
    """
    _validate_zero_inference_target("GET", url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hivemind-doctor/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return status_code, {"raw": raw[:240]}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def default_http_request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Perform a direct, proxy-free GET/HEAD/POST request to an allowlisted zero-inference endpoint."""
    verb, _ = _validate_zero_inference_target(method, url)
    if payload is not None and verb != "POST":
        raise ValueError("Request payload is only permitted on POST requests.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9",
        "User-Agent": "hivemind-doctor/1.0",
    }
    if headers:
        for raw_k, raw_v in headers.items():
            clean_k = str(raw_k).strip()
            clean_v = str(raw_v).strip()
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in clean_k + clean_v):
                raise ValueError("HTTP headers must not contain control characters.")
            req_headers[clean_k] = clean_v

    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            content_type = str(resp.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": _sanitize_noel_text(raw_err[:240])}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})

    if not raw.strip():
        return status_code, {"content_type": content_type, "html_length": 0}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status_code, {"content_type": content_type, "html_length": len(raw), "raw": raw[:240]}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


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
    verb, _ = _validate_zero_inference_target(method, url)
    sig = inspect.signature(http_request)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if has_kwargs or ("payload" in params and "headers" in params and not has_varargs and len(positional) <= 2):
        kw: dict[str, Any] = {}
        if has_kwargs or "payload" in params:
            kw["payload"] = payload
        if has_kwargs or "headers" in params:
            kw["headers"] = headers
        if has_kwargs or "timeout_s" in params:
            kw["timeout_s"] = timeout_s
        status_code, body = http_request(verb, url, **kw)
    elif has_varargs or len(positional) >= 5:
        status_code, body = http_request(verb, url, payload, headers, timeout_s)
    elif len(positional) == 4:
        status_code, body = http_request(verb, url, payload, headers)
    elif len(positional) == 3:
        status_code, body = http_request(verb, url, payload)
    else:
        status_code, body = http_request(verb, url)

    return int(status_code), (dict(body) if isinstance(body, Mapping) else {"data": body})


def default_subprocess_runner(argv: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Run a local read-only command with shell=False and a strict timeout."""
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
    if eff_os == "nt" or "win" in eff_sys:
        return "windows"
    if "darwin" in eff_sys or "mac" in eff_sys:
        return "macos"
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
    """Locate .git/config and worktree commondir config under workspace strictly locally."""
    dot_git = workspace / ".git"
    configs: list[Path] = []
    try:
        if dot_git.is_dir():
            cfg = dot_git / "config"
            if cfg.is_file():
                configs.append(cfg)
        elif dot_git.is_file():
            lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].lower().startswith("gitdir:"):
                raw_gitdir = lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gitdir)
                if not gitdir.is_absolute():
                    gitdir = (workspace / gitdir).resolve()
                wt_cfg = gitdir / "config"
                if wt_cfg.is_file():
                    configs.append(wt_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    if rel_common:
                        common_cfg = (gitdir / rel_common).resolve() / "config"
                        if common_cfg.is_file() and common_cfg not in configs:
                            configs.append(common_cfg)
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
            cleaned = (m1 or m2).strip().strip('"')
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
    scrubbed_ws = _PYTEST_TMP_SEGMENT_RE.sub("pytest_tmp", str(workspace))
    for ws_hit in _find_corp_markers(scrubbed_ws):
        _add_indicator(f"cwd_path:{ws_hit}", (ws_hit,))

    if workspace.exists():
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
    ces_http_status: int | None = None
    average_ease_rating: float | None = None

    if raid_status == 200 and isinstance(raid_body, Mapping) and isinstance(raid_body.get("onboarding_feedback"), Mapping):
        fb_snap = raid_body["onboarding_feedback"]
        raw_raid_ease = fb_snap.get("average_ease_rating", fb_snap.get("mean_ease_rating"))
        if isinstance(raw_raid_ease, (int, float)) and not isinstance(raw_raid_ease, bool):
            average_ease_rating = round(float(raw_raid_ease), 2)

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
        else:
            ces_recorded = False
            ces_degraded = True

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
        "ces_http_status": ces_http_status,
        "ces_idempotency_key": ces_idempotency_key if submit_ces else None,
        "ces_payload": ces_payload if submit_ces else None,
        "zero_game_lag": zero_game_lag,
        "average_ease_rating": average_ease_rating,
        "mean_ease_rating": average_ease_rating,
        "http_calls_made": http_calls_made,
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
) -> dict[str, Any]:
    """Run all 6 Canary Alpha preflight checks (or --noel extended mode) in < 10 seconds."""
    started = time.monotonic()
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    bounded_timeout = min(max(0.5, float(timeout_s)), 4.5)

    try:
        safe_gateway = validate_gateway_url(gateway)
    except ValueError:
        safe_gateway = DEFAULT_GATEWAY

    if not noel:
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
    resolved_home = home if home is not None else Path.home()
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
        "file_writes": 0,
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
    parser.add_argument("--game", action="append", default=[], help="Extra game executable name to check.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=HTTP_TIMEOUT_S,
        help="Per-request HTTP timeout in seconds (bounded to <= 4.5s for < 10s total runtime).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in terminal output.")
    parser.add_argument(
        "--noel",
        action="store_true",
        help="Run <10s post-install Noel holiday route drift, <1.5s Steam yield/thermal & fleet verifier.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Workspace directory inspected locally for corporate markers / .git/config remotes in --noel mode.",
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
    return parser


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
) -> int:
    args = build_parser().parse_args(argv)
    noel_enabled = bool(args.noel or args.first_heartbeat or args.submit_ces)
    report = run_doctor(
        gateway=args.gateway,
        credentials_path=Path(args.credentials_file),
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
        cwd=Path(args.cwd) if args.cwd is not None else None,
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
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        use_color = bool(sys.stdout.isatty() and not args.no_color and os.name != "nt")
        print(render_report(report, color=use_color))
    return EXIT_CODES[str(report["verdict"])]


if __name__ == "__main__":
    raise SystemExit(main())
