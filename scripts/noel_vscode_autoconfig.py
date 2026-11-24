#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""60-second 1-command Developer VS Code autoconfig with personal-PC compliance and post-setup probe hook.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library:
  1. Strictly Local Personal-PC & Hobby-Workspace Compliance Gate (Amadeus Rule):
     inspects environment variables (`USERDNSDOMAIN`, `HTTP_PROXY`, `HTTPS_PROXY`, `COMPUTERNAME`,
     `HOSTNAME`, etc.), the target workspace path, and `.git/config` remote URLs (if present) for
     corporate indicators (`amadeus`, `corp`, `enterprise`, `intranet`, `internal`, `zscaler`,
     `netskope`, `bluecoat`, `fortinet`, `paloalto`) BEFORE writing any files or contacting the
     gateway. Refuses configuration (`exit code 2`, `compliance_ok=False`) when corporate markers
     are detected and NEVER transmits local env, paths, or git remotes over the network.
  2. < 60s Zero-Inference Developer Autoconfig Flow:
     loads `~/.hivemind/credentials.json` (or registers/verifies via `POST /v1/hivemind/register`,
     `GET /v1/hivemind/me`, and `GET /v1/models` — NEVER calling `/v1/chat/completions` or prompting
     `hivemind/coder`), then atomically and idempotently writes (`0600` on POSIX, `.bak` backup when
     existing user content differs):
       (a) `~/.continue/hivemind_provider.yaml` (`hivemind/coder` and `hivemind/bee-stars`),
       (b) optional workspace `.vscode/settings.json` non-secret OpenAI-compatible endpoint/model
           defaults (preserving existing user settings, keeping raw `hm-...` keys strictly out of
           workspace files, and surfacing a `.gitignore` safety warning when unignored), and
       (c) clean `--uninstall` removal of HiveMind-managed config artifacts.
  3. Chained Post-Setup Probe & Anonymous CES Hook:
     chains automatically by default (disableable via `--skip-probe`) into
     `scripts/noel_post_setup_probe.py` in zero-inference mode, with optional `--ease-rating 1..5`
     or `--prompt-ces` passthrough to `POST /v1/hivemind/onboarding/feedback` (`profile='dev'`),
     emitting redacted CLI, `--lang fr|en` Markdown, or `--json` reports with a SHA-256 receipt.
  4. Streamlined `<60s` Quick-Connect, Live Producer Heartbeat & `<1.5s` Steam-Yield Preflight:
     `--quick-connect`, `--check-producer-heartbeat`, `--send-opentrainer-heartbeat`,
     `--gamer-preflight`, `--max-temp-c`, and timeout-safe non-blocking `--post-setup-telemetry`.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_vscode_autoconfig/1"
RECEIPT_OBJECT = "hivemind.noel_vscode_autoconfig_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_MODELS: tuple[str, str] = ("hivemind/coder", "hivemind/bee-stars")
DEFAULT_MAX_TOKENS = 2048
SLO_BUDGET_SECONDS = 60.0
HTTP_TIMEOUT_S = 5.0
TELEMETRY_TIMEOUT_S = 2.5
SUBPROCESS_TIMEOUT_S = 3.0
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_YIELD_BUDGET_MS = 1500.0
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

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
        "rustclient.exe",
        "palworld-win64-shipping.exe",
        "re4.exe",
        "alanwake2.exe",
        "blackmythwukong.exe",
        "b1-win64-shipping.exe",
    }
)
STEAM_PROCESS_PREFIXES: tuple[str, ...] = ("steam_app_", "steam_game_", "proton_")
PREFLIGHT_GAME_SIGNATURES: tuple[str, ...] = (
    "cs2.exe",
    "eldenring.exe",
    "cyberpunk2077.exe",
    "bg3.exe",
    "helldivers2.exe",
    "steam_app_730",
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

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

HIVEMIND_VSCODE_PREFIX = "hivemind."

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

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
    ("paloalto", re.compile(r"paloalto", re.IGNORECASE)),
)

HttpFn = Callable[..., tuple[int, Any]]
ProbeFn = Callable[..., Mapping[str, Any]]
InputFn = Callable[[str], str]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
WorkerYieldFn = Callable[..., Any]
ClockFn = Callable[[], float]


@dataclass(frozen=True)
class GpuProbeSnapshot:
    """Truthful local NVIDIA GPU telemetry without UUID/serial identifiers or invented specs."""

    present: bool
    parse_error: bool = False
    name: str | None = None
    total_vram_mib: int | None = None
    free_vram_mib: int | None = None
    used_vram_mib: int | None = None
    temp_c: float | None = None
    util_pct: float | None = None
    reason: str = "ok"


class AutoconfigError(ValueError):
    """Raised when configuration inputs, gateway validation, or safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise AutoconfigError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise AutoconfigError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_role(raw: str | None) -> str:
    """Normalize account role to one of the supported HiveMind roles."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise AutoconfigError("Role must be one of consumer, producer, prosumer, or trainer.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_ROLES:
        raise AutoconfigError(
            f"Unsupported role {raw!r}; expected consumer, producer, prosumer, or trainer."
        )
    return cleaned


def sanitize_handle(raw: str | None) -> str:
    """Validate an explicit handle or generate a zero-PII personal handle."""
    if raw is None or not str(raw).strip():
        return f"dev-noel-{secrets.token_hex(3)}"
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(raw).strip()).strip("-.")
    candidate = candidate[:48]
    if (
        len(candidate) < 2
        or not _HANDLE_RE.fullmatch(candidate)
        or candidate.lower().startswith(("google.", "github.", "oidc."))
    ):
        raise AutoconfigError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-' "
            "(never starting with a sign-in provider prefix)."
        )
    return candidate


def sanitize_host_label(raw: str | None) -> str:
    """Produce a clean 1..64 ASCII personal host label free of PII, secrets, IPs, and home paths."""
    if raw is None or isinstance(raw, bool) or not str(raw).strip():
        return "personal-pc"
    scrubbed = sanitize_text_for_display(str(raw).strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", scrubbed).strip("-._")[:64]
    return cleaned if cleaned else "personal-pc"


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff `raw_key` is a structurally valid `hm-...` or `hm_live_...` API key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask an `hm-...` or `hm_live_...` API key so raw secrets never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace local user home prefixes with `'~'` across POSIX and Windows paths."""
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
    if redacted != raw:
        return redacted.replace("\\", "/")
    return raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw keys, Bearer tokens, emails, non-loopback IPv4s, and home paths from text."""
    scrubbed = redact_home_path(text, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


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
        raise AutoconfigError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise AutoconfigError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise AutoconfigError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise AutoconfigError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise AutoconfigError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise AutoconfigError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise AutoconfigError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise AutoconfigError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise AutoconfigError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or localhost/127.0.0.1/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise AutoconfigError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise AutoconfigError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(method: str, url: str) -> str:
    """Enforce allowed gateway origin and block any LLM completion endpoint."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise AutoconfigError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path:
        raise AutoconfigError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    return origin


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib HTTP helper handling JSON and HTML gateway responses."""
    _assert_zero_inference_request(method, url)
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9",
        "User-Agent": "hivemind-noel-vscode-autoconfig/1.0",
    }
    if headers:
        req_headers.update(headers)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    bounded_timeout = min(max(0.2, float(timeout_s)), 15.0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            content_type = str(resp.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw_err[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": str(reason)[:200]}

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
    *,
    timeout_s: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate URL and zero-inference guard before invoking `http_fn`, catching transport errors."""
    _assert_zero_inference_request(method, url)
    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    extra_kw: dict[str, Any] = {}
    if timeout_s is not None and "timeout_s" in sig.parameters:
        extra_kw["timeout_s"] = timeout_s
    try:
        if has_varargs or len(positional) >= 4:
            status_code, body = http_fn(method.upper(), url, payload, headers, **extra_kw)
        else:
            status_code, body = http_fn(method.upper(), url, **extra_kw)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": str(exc)[:200]}
    norm_body = dict(body) if isinstance(body, Mapping) else {"data": body}
    return int(status_code), norm_body


def _find_corporate_markers(raw_text: str) -> list[str]:
    """Return sorted corporate indicator tokens matched in `raw_text` without retaining `raw_text`."""
    if not raw_text:
        return []
    matched: list[str] = []
    for marker_name, pattern in _CORP_TOKEN_PATTERNS:
        if pattern.search(raw_text) and marker_name not in matched:
            matched.append(marker_name)
    return matched


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` (and worktree commondir `config` if present) strictly on local disk."""
    dot_git = workspace / ".git"
    candidates: list[Path] = []
    try:
        if dot_git.is_dir():
            cfg = dot_git / "config"
            if cfg.is_file():
                candidates.append(cfg)
        elif dot_git.is_file():
            first_line = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
            if first_line and first_line[0].lower().startswith("gitdir:"):
                raw_gitdir = first_line[0].split(":", 1)[1].strip()
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
                        common_path = (gitdir / rel_common).resolve()
                        common_cfg = common_path / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
    except OSError:
        return candidates
    return candidates


def extract_git_config_remote_urls(config_text: str) -> list[str]:
    """Parse `[remote ...]` and `[url ...]` remote URLs from `.git/config` text in memory."""
    urls: list[str] = []
    in_remote_or_url_section = False
    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip().lower()
            in_remote_or_url_section = header.startswith(("remote ", "remote\t", 'remote"', "url "))
            if header.startswith("url ") and '"' in stripped:
                quoted = stripped.split('"', 2)
                if len(quoted) >= 2 and quoted[1].strip():
                    urls.append(quoted[1].strip())
            continue
        if in_remote_or_url_section and "=" in stripped:
            key, val = stripped.split("=", 1)
            if key.strip().lower() in ("url", "pushurl", "insteadof", "pushinsteadof"):
                cleaned_val = val.strip().strip('"')
                if cleaned_val:
                    urls.append(cleaned_val)
    return urls


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    workspace: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Workspace Compliance Gate (zero values ever leave memory)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_view = environ if environ is not None else os.environ

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
        "COMPUTERNAME",
        "HOSTNAME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
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

    if workspace is not None:
        ws_hits = _find_corporate_markers(str(workspace))
        if ws_hits:
            for hit in ws_hits:
                tag = f"workspace_path:{hit}"
                if tag not in indicators:
                    indicators.append(tag)
            matched_markers.update(ws_hits)

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
            else f"Local compliance gate passed (personal PC & hobby workspace verified). {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFIGURATION ({joined}) : marqueur professionnel / entreprise détecté. "
            f"Aucun fichier écrit et aucune requête envoyée. {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"CONFIGURATION REFUSED ({joined}): corporate machine or repository marker detected. "
            f"Zero files written and zero network requests sent. {COMPLIANCE_NOTICE_EN}"
        )

    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_transmitted": False,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }


def _write_file_secure_atomic(path: Path, content: str, *, posix_mode: int, eff_os: str) -> None:
    """Atomically write UTF-8 `content` to `path` with `posix_mode` (`0600`) from file creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp_path, flags, posix_mode)
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
            tmp_cleanup_failed = True
            _ = tmp_cleanup_failed
        raise


def atomic_write_config_file(
    path: Path,
    content: str,
    *,
    posix_mode: int = 0o600,
    create_backup: bool = True,
    os_name: str | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Idempotently and atomically write `path` (`0600` on POSIX) with `.bak` backup on diff."""
    eff_os = os_name if os_name is not None else os.name
    shown_path = redact_home_path(path, home=home)
    bak_path = Path(f"{path}.bak")
    shown_bak = redact_home_path(bak_path, home=home)
    perm_meta = (
        {"permission_model": "windows_acl"}
        if eff_os == "nt"
        else {"posix_mode": f"{posix_mode:04o}"}
    )

    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            mode_tightened = False
            if eff_os != "nt":
                current_mode = path.stat().st_mode & 0o777
                if current_mode != posix_mode:
                    os.chmod(path, posix_mode)
                    mode_tightened = True
            return {
                "path": shown_path,
                "action": "unchanged",
                "written": False,
                "created": False,
                "updated": False,
                "backup_created": False,
                "backup_path": None,
                "permissions_tightened": mode_tightened,
                **perm_meta,
            }

        backup_written = False
        if create_backup:
            _write_file_secure_atomic(bak_path, existing, posix_mode=posix_mode, eff_os=eff_os)
            backup_written = True

        _write_file_secure_atomic(path, content, posix_mode=posix_mode, eff_os=eff_os)
        return {
            "path": shown_path,
            "action": "updated",
            "written": True,
            "created": False,
            "updated": True,
            "backup_created": backup_written,
            "backup_path": shown_bak if backup_written else None,
            "permissions_tightened": False,
            **perm_meta,
        }

    if eff_os != "nt" and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_file_secure_atomic(path, content, posix_mode=posix_mode, eff_os=eff_os)
    return {
        "path": shown_path,
        "action": "created",
        "written": True,
        "created": True,
        "updated": False,
        "backup_created": False,
        "backup_path": None,
        "permissions_tightened": False,
        **perm_meta,
    }


def build_continue_provider_yaml(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Render deterministic Continue OpenAI-compatible YAML for `hivemind/coder` and `hivemind/bee-stars`."""
    clean_gw = validate_gateway_url(gateway)
    if not is_valid_hivemind_key(api_key):
        raise AutoconfigError("Cannot render Continue provider YAML without a valid hm-... API key.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 64:
        raise AutoconfigError("max_tokens must be an integer >= 64.")
    clean_key = api_key.strip()
    return (
        "# HiveMind personal hobby/open-source coding configuration for VS Code (Continue)\n"
        "# 100% personal home PCs & personal hobby/open-source projects ONLY (0.00 EUR / 0.00 USD, AGPL-3.0).\n"
        "# Never use on corporate hardware or send corporate/confidential source code.\n"
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        f"    model: {DEFAULT_MODELS[0]}\n"
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
        f"    model: {DEFAULT_MODELS[1]}\n"
        f"    apiBase: {clean_gw}/v1\n"
        f"    apiKey: {clean_key}\n"
        "    roles:\n"
        "      - chat\n"
        "    defaultCompletionOptions:\n"
        f"      maxTokens: {max_tokens}\n"
    )


def strip_jsonc_comments_and_trailing_commas(raw: str) -> str:
    """Convert VS Code JSONC (`//`, `/* */`, trailing commas) into strict JSON without touching strings."""
    chars: list[str] = []
    in_string = False
    escaped = False
    n = len(raw)
    idx = 0

    while idx < n:
        ch = raw[idx]
        if in_string:
            chars.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue

        if ch == '"':
            in_string = True
            chars.append(ch)
            idx += 1
            continue

        if ch == "/" and idx + 1 < n and raw[idx + 1] == "/":
            idx += 2
            while idx < n and raw[idx] != "\n":
                idx += 1
            continue

        if ch == "/" and idx + 1 < n and raw[idx + 1] == "*":
            idx += 2
            while idx + 1 < n and not (raw[idx] == "*" and raw[idx + 1] == "/"):
                idx += 1
            idx = idx + 2 if idx + 1 < n else n
            continue

        chars.append(ch)
        idx += 1

    no_comments = "".join(chars)
    cleaned: list[str] = []
    in_str2 = False
    esc2 = False
    m = len(no_comments)
    pos = 0
    while pos < m:
        ch = no_comments[pos]
        if in_str2:
            cleaned.append(ch)
            if esc2:
                esc2 = False
            elif ch == "\\":
                esc2 = True
            elif ch == '"':
                in_str2 = False
            pos += 1
            continue
        if ch == '"':
            in_str2 = True
            cleaned.append(ch)
            pos += 1
            continue
        if ch == ",":
            look = pos + 1
            while look < m and no_comments[look].isspace():
                look += 1
            if look < m and no_comments[look] in ("}", "]"):
                pos += 1
                continue
        cleaned.append(ch)
        pos += 1

    return "".join(cleaned)


def _assert_workspace_content_secret_free(content: str) -> str:
    """Guarantee workspace files NEVER contain `hm-...` keys, Bearer tokens, or OS home paths."""
    if _API_KEY_RE.search(content):
        raise AutoconfigError("Workspace secret guard blocked raw hm-... API key in workspace file.")
    if _BEARER_RE.search(content):
        raise AutoconfigError("Workspace secret guard blocked Bearer token in workspace file.")
    if _HOME_PATH_RE.search(content):
        raise AutoconfigError("Workspace secret guard blocked unredacted OS home path in workspace file.")
    return content


def build_merged_vscode_settings_json(existing_text: str | None, *, gateway: str) -> str:
    """Merge non-secret HiveMind endpoint/model defaults into VS Code `settings.json`."""
    clean_gw = validate_gateway_url(gateway)
    parsed: dict[str, Any] = {}
    if existing_text is not None and existing_text.strip():
        strict_json = strip_jsonc_comments_and_trailing_commas(existing_text)
        if strict_json.strip():
            try:
                loaded = json.loads(strict_json)
            except json.JSONDecodeError as exc:
                raise AutoconfigError(
                    "Existing .vscode/settings.json contains invalid JSON; refusing to overwrite."
                ) from exc
            if not isinstance(loaded, dict):
                raise AutoconfigError(
                    "Existing .vscode/settings.json root must be a JSON object."
                )
            parsed = dict(loaded)

    # Remove any stale HiveMind keys (including any accidentally pasted secret key) before setting clean defaults
    cleaned_user_keys: dict[str, Any] = {
        k: v for k, v in parsed.items() if not str(k).startswith(HIVEMIND_VSCODE_PREFIX)
    }
    _assert_workspace_content_secret_free(json.dumps(cleaned_user_keys, ensure_ascii=False))

    hivemind_defaults: dict[str, Any] = {
        "hivemind.openaiCompatible.apiBase": f"{clean_gw}/v1",
        "hivemind.openaiCompatible.defaultModel": DEFAULT_MODELS[0],
        "hivemind.openaiCompatible.models": list(DEFAULT_MODELS),
        "hivemind.continue.providerConfigPath": "~/.continue/hivemind_provider.yaml",
        "hivemind.credentials.secretStorage": "~/.hivemind/credentials.json",
        "hivemind.compliance.personalPcOnly": True,
        "hivemind.compliance.financialPromise": FINANCIAL_PROMISE,
    }
    merged = {**cleaned_user_keys, **hivemind_defaults}
    rendered = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    return _assert_workspace_content_secret_free(rendered)


def check_workspace_gitignore_safety(workspace: Path, *, lang: str = "en") -> dict[str, Any]:
    """Verify `.gitignore` in `workspace` ignores `.vscode/settings.json` (and `.bak` backups)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    gitignore_path = workspace / ".gitignore"

    if not gitignore_path.is_file():
        warning = (
            "Avertissement sécurité `.gitignore` : aucun fichier `.gitignore` dans l'espace de travail ; "
            "ajoutez `.vscode/settings.json` et `*.bak` (aucune clé secrète `hm-...` n'est stockée dans `.vscode/`)."
            if is_fr
            else "Workspace `.gitignore` safety warning: `.gitignore` is missing; add `.vscode/settings.json` "
            "and `*.bak` so editor settings and backups stay untracked (raw `hm-...` keys are never stored in `.vscode/`)."
        )
        return {
            "safe": False,
            "gitignore_present": False,
            "vscode_ignored": False,
            "warning": warning,
        }

    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    accepted_patterns = {
        ".vscode",
        ".vscode/",
        "/.vscode",
        "/.vscode/",
        ".vscode/*",
        "/.vscode/*",
        ".vscode/**",
        "/.vscode/**",
        ".vscode/settings.json",
        "/.vscode/settings.json",
        "settings.json",
    }
    vscode_ignored = False
    for raw_line in lines:
        rule = raw_line.strip()
        if not rule or rule.startswith(("#", "!")):
            continue
        if rule in accepted_patterns:
            vscode_ignored = True
            break

    if not vscode_ignored:
        warning = (
            "Avertissement sécurité `.gitignore` : `.vscode/settings.json` n'est pas listé dans `.gitignore` ; "
            "ajoutez `.vscode/` ou `.vscode/settings.json` et `*.bak`."
            if is_fr
            else "Workspace `.gitignore` safety warning: `.vscode/settings.json` is not covered by `.gitignore`; "
            "add `.vscode/` or `.vscode/settings.json` and `*.bak`."
        )
        return {
            "safe": False,
            "gitignore_present": True,
            "vscode_ignored": False,
            "warning": warning,
        }

    return {
        "safe": True,
        "gitignore_present": True,
        "vscode_ignored": True,
        "warning": "",
    }


def load_saved_credentials(path: Path, *, expected_gateway: str | None = None) -> dict[str, Any] | None:
    """Load `~/.hivemind/credentials.json` if readable and holding a valid `hm-...` key."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    raw_key = loaded.get("api_key")
    if not is_valid_hivemind_key(raw_key):
        return None
    saved_gw = str(loaded.get("gateway") or "").strip().rstrip("/")
    if expected_gateway and saved_gw and saved_gw != expected_gateway.rstrip("/"):
        return None
    return loaded


def connect_or_register_dev_account(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None = None,
    role: str = "prosumer",
    explicit_api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
    http_fn: HttpFn = default_http_json,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Load or register personal HiveMind credentials and verify `/v1/hivemind/me` + `/v1/models`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_role = normalize_role(role)
    env_view = environ if environ is not None else os.environ

    env_key = str(env_view.get("HIVEMIND_API_KEY") or "").strip()
    candidate_key = (explicit_api_key or "").strip() or env_key
    if candidate_key and not is_valid_hivemind_key(candidate_key):
        raise AutoconfigError("Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key.")

    saved = load_saved_credentials(credentials_path, expected_gateway=clean_gw)
    if not candidate_key and saved is not None:
        candidate_key = str(saved["api_key"]).strip()

    created_account = False
    explicit_handle = bool(handle and str(handle).strip())
    resolved_handle = (
        sanitize_handle(handle)
        if explicit_handle
        else str((saved or {}).get("handle") or "").strip() or sanitize_handle(None)
    )
    account_id = str((saved or {}).get("account_id") or "")

    if not candidate_key:
        reg_status, reg_body = _safe_http_call(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/register",
            {"handle": resolved_handle, "role": norm_role},
            None,
        )
        if reg_status == 409 and not explicit_handle:
            resolved_handle = sanitize_handle(f"{resolved_handle[:38]}-{secrets.token_hex(2)}")
            reg_status, reg_body = _safe_http_call(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/register",
                {"handle": resolved_handle, "role": norm_role},
                None,
            )
        issued_key = reg_body.get("api_key")
        if reg_status != 200 or not is_valid_hivemind_key(issued_key):
            err_msg = (
                f"Échec d'inscription sur POST /v1/hivemind/register (HTTP {reg_status})."
                if is_fr
                else f"Account registration failed on POST /v1/hivemind/register (HTTP {reg_status})."
            )
            return {
                "ok": False,
                "http_status": reg_status,
                "stage": "register",
                "error": err_msg,
            }
        candidate_key = str(issued_key).strip()
        account_id = str(reg_body.get("account_id") or account_id)
        created_account = True

    masked = mask_api_key(candidate_key)
    me_status, me_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/me",
        None,
        {"Authorization": f"Bearer {candidate_key}"},
    )
    if me_status != 200:
        err_msg = (
            f"Échec de vérification de la clé {masked} sur GET /v1/hivemind/me (HTTP {me_status})."
            if is_fr
            else f"API key verification for {masked} failed on GET /v1/hivemind/me (HTTP {me_status})."
        )
        return {
            "ok": False,
            "http_status": me_status,
            "stage": "me",
            "error": err_msg,
        }

    models_status, models_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/models",
        None,
        None,
    )
    raw_models = models_body.get("data")
    if models_status != 200 or not isinstance(raw_models, list):
        err_msg = (
            f"Échec de vérification du catalogue GET /v1/models (HTTP {models_status})."
            if is_fr
            else f"Model catalog check failed on GET /v1/models (HTTP {models_status})."
        )
        return {
            "ok": False,
            "http_status": models_status,
            "stage": "models",
            "error": err_msg,
        }

    model_ids = [
        str(item["id"])
        for item in raw_models
        if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"].strip()
    ]
    verified_handle = str(me_body.get("handle") or resolved_handle)
    verified_role = str(me_body.get("role") or norm_role)
    verified_account_id = str(me_body.get("account_id") or account_id)

    cred_record = {
        "account_id": verified_account_id,
        "api_key": candidate_key,
        "gateway": clean_gw,
        "handle": verified_handle,
        "role": verified_role,
    }
    cred_content = json.dumps(cred_record, indent=2, sort_keys=True) + "\n"
    cred_write = atomic_write_config_file(
        credentials_path,
        cred_content,
        posix_mode=0o600,
        create_backup=True,
        os_name=os_name,
        home=home,
    )

    return {
        "ok": True,
        "api_key": candidate_key,
        "masked_key": masked,
        "account_id": verified_account_id,
        "handle": verified_handle,
        "role": verified_role,
        "created_account": created_account,
        "models": model_ids,
        "models_count": len(model_ids),
        "credentials_write": cred_write,
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


def parse_dev_ces_input(raw: str, *, default_had_game_lag: bool = False) -> dict[str, Any]:
    """Parse a 1-prompt developer CES response (`'5'`, `'4/5'`, `'3 lag'`, or empty to skip)."""
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": "dev",
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
                raise AutoconfigError("CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("dev", "developer", "d"):
            continue
        elif tok in ("lag", "had-game-lag", "had_game_lag", "stutter", "saccade"):
            had_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "ok"):
            had_lag = False
        else:
            raise AutoconfigError(f"Unrecognized CES token {tok!r}; expected 1..5 [lag|nolag].")

    if rating is None:
        raise AutoconfigError("Interactive CES input must include an ease_rating from 1 to 5.")

    return {
        "skipped": False,
        "profile": "dev",
        "ease_rating": rating,
        "zero_game_lag": not had_lag,
    }


def submit_dev_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    ease_rating: int,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Submit anonymous developer CES feedback (`profile='dev'`) to `/v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise AutoconfigError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise AutoconfigError("zero_game_lag must be a boolean.")

    payload: dict[str, Any] = {
        "profile": "dev",
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": _platform_tag(system_name),
    }
    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        None,
    )

    raw_mean = body.get("mean_ease_rating")
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )
    raw_total = body.get("total_responses")
    total_resp = (
        int(raw_total)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )

    if status_code in (200, 201):
        outcome = "recorded"
        submitted = True
        msg = (
            f"Merci ! Retour CES enregistré (ease_rating={ease_rating}/5, profil=dev)."
            if is_fr
            else f"Thank you! Anonymous CES feedback recorded (ease_rating={ease_rating}/5, profile=dev)."
        )
    elif status_code == 404:
        outcome = "missing_route"
        submitted = False
        msg = (
            "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 : la révision Cloud Run est en "
            "retard sur `main` (route Noël absente)."
            if is_fr
            else "POST /v1/hivemind/onboarding/feedback returned HTTP 404: Cloud Run revision lags behind "
            "`main` (holiday feedback route missing)."
        )
    elif status_code == 429:
        outcome = "rate_limited"
        submitted = False
        msg = (
            "Limite de débit atteinte (HTTP 429) sur /v1/hivemind/onboarding/feedback ; réessayez dans 1 minute."
            if is_fr
            else "Rate limit reached (HTTP 429) on /v1/hivemind/onboarding/feedback; retry in one minute."
        )
    elif status_code >= 500:
        outcome = "server_error"
        submitted = False
        msg = (
            f"Erreur temporaire du serveur passerelle (HTTP {status_code}) lors de l'envoi du retour CES."
            if is_fr
            else f"Temporary gateway server error (HTTP {status_code}) while submitting CES feedback."
        )
    elif status_code == 0:
        outcome = "network_error"
        submitted = False
        msg = (
            "Impossible de joindre /v1/hivemind/onboarding/feedback (erreur réseau)."
            if is_fr
            else "Could not reach /v1/hivemind/onboarding/feedback (network error)."
        )
    else:
        outcome = "client_error"
        submitted = False
        msg = (
            f"Envoi du retour CES refusé par la passerelle (HTTP {status_code})."
            if is_fr
            else f"CES feedback rejected by gateway (HTTP {status_code})."
        )

    return {
        "attempted": True,
        "submitted": submitted,
        "skipped": False,
        "status": outcome,
        "http_status": status_code,
        "profile": "dev",
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "mean_ease_rating": mean_ease,
        "total_responses": total_resp,
        "zero_pii_payload": payload,
        "message": msg,
    }


def prompt_and_submit_dev_ces(
    *,
    gateway: str = DEFAULT_GATEWAY,
    default_had_game_lag: bool = False,
    system_name: str | None = None,
    input_fn: InputFn = input,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Ask a single 1-line developer CES question and submit to `/v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    prompt_text = (
        "[HiveMind CES Dev 1-question] Notez la simplicité d'autoconfig VS Code 1..5 "
        "(profil=dev ; Entrée pour ignorer) : "
        if is_fr
        else "[HiveMind 1-prompt Dev CES] Rate VS Code autoconfig ease 1..5 "
        "(profile=dev; Enter to skip): "
    )
    try:
        raw_answer = input_fn(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raw_answer = ""

    parsed = parse_dev_ces_input(raw_answer, default_had_game_lag=default_had_game_lag)
    if parsed["skipped"]:
        skip_msg = (
            "Retour CES 1-question ignoré (aucun envoi réseau)."
            if is_fr
            else "1-prompt CES skipped (no feedback sent)."
        )
        return {
            "attempted": False,
            "submitted": False,
            "skipped": True,
            "status": "skipped",
            "http_status": None,
            "profile": "dev",
            "ease_rating": None,
            "zero_game_lag": parsed["zero_game_lag"],
            "mean_ease_rating": None,
            "total_responses": None,
            "zero_pii_payload": None,
            "message": skip_msg,
        }

    return submit_dev_ces_feedback(
        gateway=clean_gw,
        ease_rating=int(parsed["ease_rating"]),
        zero_game_lag=bool(parsed["zero_game_lag"]),
        system_name=system_name,
        http_fn=http_fn,
        lang=lang_code,
    )


def _load_sibling_probe_module() -> Any:
    """Load `scripts/noel_post_setup_probe.py` from the sibling script path via stdlib `importlib`."""
    cached = sys.modules.get("noel_post_setup_probe")
    if cached is not None and hasattr(cached, "run_post_setup_probe"):
        return cached
    probe_path = Path(__file__).resolve().with_name("noel_post_setup_probe.py")
    if not probe_path.is_file():
        raise AutoconfigError("Sibling scripts/noel_post_setup_probe.py not found; pass --skip-probe.")
    spec = importlib.util.spec_from_file_location("noel_post_setup_probe", probe_path)
    if spec is None or spec.loader is None:
        raise AutoconfigError("Unable to create module spec for scripts/noel_post_setup_probe.py.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["noel_post_setup_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def default_probe_fn(
    *,
    gateway: str,
    credentials_path: Path,
    continue_path: Path,
    http_fn: HttpFn = default_http_json,
    feedback_result: Mapping[str, Any] | None = None,
    lang: str = "en",
    os_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Chain into `scripts/noel_post_setup_probe.py` in zero-inference mode."""
    probe_mod = _load_sibling_probe_module()
    result = probe_mod.run_post_setup_probe(
        gateway=gateway,
        smoke_only=False,
        verify_account=False,
        credentials_path=credentials_path,
        continue_path=continue_path,
        feedback_result=feedback_result,
        lang=lang,
        os_name=os_name,
        environ=environ,
        home=home,
        http_fn=http_fn,
    )
    return dict(result)


def _invoke_probe_callable(
    probe_fn: ProbeFn,
    *,
    gateway: str,
    credentials_path: Path,
    continue_path: Path,
    http_fn: HttpFn,
    feedback_result: Mapping[str, Any] | None,
    lang: str,
    os_name: str | None,
    environ: Mapping[str, str] | None,
    home: Path | None,
) -> dict[str, Any]:
    """Invoke `probe_fn` passing all supported keyword parameters cleanly."""
    candidate_kwargs: dict[str, Any] = {
        "gateway": gateway,
        "credentials_path": credentials_path,
        "continue_path": continue_path,
        "http_fn": http_fn,
        "feedback_result": feedback_result,
        "lang": lang,
        "os_name": os_name,
        "environ": environ,
        "home": home,
    }
    sig = inspect.signature(probe_fn)
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_varkw:
        raw_res = probe_fn(**candidate_kwargs)
    else:
        filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
        raw_res = probe_fn(**filtered)
    if not isinstance(raw_res, Mapping):
        raise AutoconfigError("probe_fn must return a mapping report.")
    if raw_res.get("zero_inference_verified") is False:
        raise AutoconfigError("Zero-inference invariant violated by probe_fn.")
    return dict(raw_res)


def uninstall_hivemind_autoconfig(
    *,
    credentials_path: Path,
    continue_path: Path,
    vscode_settings_path: Path | None = None,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Remove HiveMind-managed config artifacts while preserving non-HiveMind VS Code user settings."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    removed_files: list[str] = []
    preserved_settings_files: list[str] = []

    for managed in (
        credentials_path,
        Path(f"{credentials_path}.bak"),
        continue_path,
        Path(f"{continue_path}.bak"),
    ):
        if managed.is_file():
            managed.unlink()
            removed_files.append(redact_home_path(managed, home=home))

    if vscode_settings_path is not None:
        vs_bak = Path(f"{vscode_settings_path}.bak")
        if vs_bak.is_file():
            vs_bak.unlink()
            removed_files.append(redact_home_path(vs_bak, home=home))

        if vscode_settings_path.is_file():
            raw_vs = vscode_settings_path.read_text(encoding="utf-8")
            strict_vs = strip_jsonc_comments_and_trailing_commas(raw_vs)
            try:
                loaded_vs = json.loads(strict_vs) if strict_vs.strip() else {}
            except json.JSONDecodeError:
                loaded_vs = None

            if isinstance(loaded_vs, dict):
                had_hm = any(str(k).startswith(HIVEMIND_VSCODE_PREFIX) for k in loaded_vs)
                remaining = {
                    k: v for k, v in loaded_vs.items() if not str(k).startswith(HIVEMIND_VSCODE_PREFIX)
                }
                if not remaining and had_hm:
                    vscode_settings_path.unlink()
                    removed_files.append(redact_home_path(vscode_settings_path, home=home))
                    try:
                        if vscode_settings_path.parent.is_dir() and not any(vscode_settings_path.parent.iterdir()):
                            vscode_settings_path.parent.rmdir()
                    except OSError:
                        dir_not_empty = True
                        _ = dir_not_empty
                elif had_hm:
                    rendered_rem = _assert_workspace_content_secret_free(
                        json.dumps(remaining, indent=2, ensure_ascii=False) + "\n"
                    )
                    _write_file_secure_atomic(
                        vscode_settings_path,
                        rendered_rem,
                        posix_mode=0o600,
                        eff_os=eff_os,
                    )
                    preserved_settings_files.append(redact_home_path(vscode_settings_path, home=home))

    return {
        "removed_files": removed_files,
        "removed_count": len(removed_files),
        "preserved_settings_files": preserved_settings_files,
        "lang": lang_code,
    }


def scan_output_safety(text: str) -> list[str]:
    """Verify CLI, Markdown, or JSON output contains zero PII, raw keys, home paths, or `'score': <digit>`."""
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise AutoconfigError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _summarize_probe_report(probe_report: Mapping[str, Any], *, home: Path | None = None) -> dict[str, Any]:
    """Extract a sanitized, receipt-verified summary of the chained `noel_post_setup_probe` report."""
    smoke = probe_report.get("smoke_gate")
    all_routes_ok = (
        bool(smoke.get("all_routes_ok", True))
        if isinstance(smoke, Mapping)
        else bool(probe_report.get("ready", True))
    )
    raw_diags = smoke.get("diagnostics") if isinstance(smoke, Mapping) else ()
    clean_diags = [
        sanitize_text_for_display(str(item), home=home)
        for item in (raw_diags or ())
        if str(item).strip()
    ]
    return {
        "ready": bool(probe_report.get("ready", False)),
        "verdict": str(probe_report.get("verdict") or ("ready" if probe_report.get("ready") else "not_ready")),
        "all_routes_ok": all_routes_ok,
        "routes_passed_count": int(probe_report.get("routes_passed_count") or 0),
        "routes_total_count": int(probe_report.get("routes_total_count") or 0),
        "checks_passed_count": int(probe_report.get("checks_passed_count") or 0),
        "checks_total_count": int(probe_report.get("checks_total_count") or 0),
        "warnings_count": int(probe_report.get("warnings_count") or 0),
        "failures_count": int(probe_report.get("failures_count") or 0),
        "zero_inference_verified": bool(probe_report.get("zero_inference_verified", True)),
        "mean_ease_rating": probe_report.get("mean_ease_rating"),
        "diagnostics": clean_diags,
        "probe_receipt_sha256": str(probe_report.get("receipt_sha256") or ""),
    }


def _invoke_with_supported_kwargs(fn: Callable[..., Any], **candidate_kwargs: Any) -> Any:
    """Invoke `fn` passing only keyword arguments accepted by its signature."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def default_subprocess_runner(
    cmd: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Execute a safe read-only command via list argv (`shell=False`) with `<= 3.0s` timeout."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise AutoconfigError("Subprocess command must be a non-empty sequence of string arguments.")
    argv = [str(part) for part in cmd]
    bounded = min(max(0.1, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    return int(completed.returncode), (completed.stdout or "")


def _optional_finite_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    cleaned = str(raw).strip()
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
    """Query local `nvidia-smi` metrics (`<= 3s` timeout) without UUID/serial or invented specs."""
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

    discovered: list[GpuProbeSnapshot] = []
    for raw_line in stdout.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 6 or not parts[0]:
            return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        total_f = _optional_finite_float(parts[1])
        free_f = _optional_finite_float(parts[2])
        used_f = _optional_finite_float(parts[3])
        temp_f = _optional_finite_float(parts[4])
        util_f = _optional_finite_float(parts[5])
        if total_f is None or free_f is None or total_f <= 0 or free_f < 0:
            return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        used_mib = int(used_f) if used_f is not None and used_f >= 0 else max(0, int(total_f) - int(free_f))
        discovered.append(
            GpuProbeSnapshot(
                present=True,
                parse_error=False,
                name=parts[0][:128],
                total_vram_mib=int(total_f),
                free_vram_mib=int(free_f),
                used_vram_mib=used_mib,
                temp_c=round(temp_f, 1) if temp_f is not None else None,
                util_pct=round(util_f, 1) if util_f is not None else 0.0,
                reason="ok",
            )
        )
    return max(
        discovered,
        key=lambda g: (
            g.free_vram_mib or 0,
            -(g.temp_c if g.temp_c is not None else 999.0),
            g.total_vram_mib or 0,
        ),
    )


def _normalize_gpu_snapshot(raw: Any) -> GpuProbeSnapshot:
    """Normalize `None`, `GpuProbeSnapshot`, mapping, or GPU telemetry object without inventing values."""
    if raw is None:
        return GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")
    if isinstance(raw, GpuProbeSnapshot):
        return raw
    if isinstance(raw, Mapping):
        view: Mapping[str, Any] = raw
    elif hasattr(raw, "present"):
        view = {
            "present": getattr(raw, "present", False),
            "parse_error": getattr(raw, "parse_error", False),
            "name": getattr(raw, "name", None) or getattr(raw, "gpu_name", None),
            "total_vram_mib": (
                getattr(raw, "total_vram_mib", None)
                if getattr(raw, "total_vram_mib", None) is not None
                else getattr(raw, "total_mib", None)
            ),
            "free_vram_mib": (
                getattr(raw, "free_vram_mib", None)
                if getattr(raw, "free_vram_mib", None) is not None
                else getattr(raw, "free_mib", None)
            ),
            "used_vram_mib": (
                getattr(raw, "used_vram_mib", None)
                if getattr(raw, "used_vram_mib", None) is not None
                else getattr(raw, "used_mib", None)
            ),
            "temp_c": getattr(raw, "temp_c", None),
            "util_pct": getattr(raw, "util_pct", None),
            "reason": getattr(raw, "reason", "ok"),
        }
    else:
        raise AutoconfigError("gpu_probe_fn / gpu_snapshot must be GpuProbeSnapshot, a mapping, or None.")

    present = bool(view.get("present", view.get("gpu_present", False)))
    if not present:
        return GpuProbeSnapshot(present=False, reason=str(view.get("reason") or "no_nvidia_gpu"))
    if bool(view.get("parse_error", False)):
        return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")

    name_raw = view.get("name") or view.get("gpu_name") or view.get("gpu_model")
    clean_name = str(name_raw).strip()[:128] if name_raw is not None and str(name_raw).strip() else None
    raw_total = view.get("total_vram_mib", view.get("vram_total_mib", view.get("total_mib")))
    raw_free = view.get("free_vram_mib", view.get("vram_free_mib", view.get("free_mib")))
    raw_used = view.get("used_vram_mib", view.get("vram_used_mib", view.get("used_mib")))
    temp_f = _optional_finite_float(view.get("temp_c"))
    util_f = _optional_finite_float(view.get("util_pct", view.get("utilization_pct")))

    if not clean_name or raw_total is None or raw_free is None:
        return GpuProbeSnapshot(present=True, parse_error=True, reason="incomplete_gpu_telemetry")

    total_mib = int(raw_total)
    free_mib = int(raw_free)
    used_mib = int(raw_used) if raw_used is not None else max(0, total_mib - free_mib)
    return GpuProbeSnapshot(
        present=True,
        parse_error=False,
        name=clean_name,
        total_vram_mib=total_mib,
        free_vram_mib=free_mib,
        used_vram_mib=used_mib,
        temp_c=round(temp_f, 1) if temp_f is not None else None,
        util_pct=round(util_f, 1) if util_f is not None else 0.0,
        reason=str(view.get("reason") or "ok"),
    )


def default_process_probe(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> list[str]:
    """Inspect active local process names via `tasklist` (Windows) or `ps` (POSIX) in `<= 3s`."""
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


def detect_steam_dx12_activity(
    process_names: Iterable[str],
    *,
    non_worker_gpu_util_pct: float | None = None,
    gpu_util_threshold_pct: float = NON_WORKER_GPU_UTIL_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Detect active Steam / DX12 game processes or non-worker GPU utilization `> 15%`."""
    trunc_prefixes = {exe[:15] for exe in STEAM_DX12_GAME_EXECUTABLES if len(exe) > 15}
    matched_games: list[str] = []
    for raw_proc in process_names:
        raw_low = str(raw_proc).strip().lower().replace("\\", "/")
        if not raw_low:
            continue
        base = raw_low.rsplit("/", 1)[-1]
        is_steam_game = (
            base.startswith(STEAM_PROCESS_PREFIXES)
            or "steamapps/common/" in raw_low
            or base in ("steam_game", "steam_game_process", "steamwebhelper_game")
        )
        if base in STEAM_DX12_GAME_EXECUTABLES or base in trunc_prefixes or is_steam_game:
            if base not in matched_games:
                matched_games.append(base)

    gpu_util_triggered = bool(
        non_worker_gpu_util_pct is not None
        and math.isfinite(non_worker_gpu_util_pct)
        and non_worker_gpu_util_pct > gpu_util_threshold_pct
    )
    return {
        "game_detected": bool(matched_games or gpu_util_triggered),
        "matched_games": matched_games,
        "gpu_util_triggered": gpu_util_triggered,
        "non_worker_gpu_util_pct": non_worker_gpu_util_pct,
        "gpu_util_threshold_pct": gpu_util_threshold_pct,
    }


def default_worker_yield_drill(
    *,
    clock_fn: ClockFn = time.monotonic,
    trigger_exe: str = "cs2.exe",
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a lightweight local subprocess worker preemption drill (`shell=False`, `<= 3s` timeout)."""
    bounded = min(max(0.2, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
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
                "trigger_exe": trigger_exe,
                "method": "local_python_worker_subprocess",
            }

        t_trigger = clock_fn()
        if not detect_steam_dx12_activity([trigger_exe])["game_detected"]:
            proc.kill()
            proc.wait(timeout=1.0)
            return {
                "worker_started": True,
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": -1,
                "yield_latency_ms": None,
                "trigger_exe": trigger_exe,
                "method": "local_python_worker_subprocess",
            }

        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()

        release_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        exit_rc = proc.wait(timeout=bounded)
        t_done = clock_fn()
        yield_latency_ms = round(max(0.0, (t_done - t_trigger) * 1000.0), 3)

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
            "trigger_exe": trigger_exe,
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
            "trigger_exe": trigger_exe,
            "method": "local_python_worker_subprocess",
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def run_steam_thermal_preflight(
    *,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    had_game_lag: bool = False,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: ClockFn = time.monotonic,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run the `<1.5s` Steam GPU yield & `<=70.0 C` quiet-fan thermal preflight check."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name

    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise AutoconfigError("max_temp_c must be a finite temperature between 25.0 and 95.0 C.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise AutoconfigError("min_free_vram_mib must be an integer >= 256 MiB.")
    if isinstance(yield_budget_ms, bool) or not math.isfinite(float(yield_budget_ms)) or float(yield_budget_ms) <= 0.0:
        raise AutoconfigError("yield_budget_ms must be a positive finite budget.")

    cap_c = round(float(max_temp_c), 1)
    reserve_mib = int(min_free_vram_mib)
    budget_ms = round(float(yield_budget_ms), 1)

    raw_gpu = _invoke_with_supported_kwargs(gpu_probe_fn, timeout_s=SUBPROCESS_TIMEOUT_S)
    gpu = _normalize_gpu_snapshot(raw_gpu)

    live_procs = list(
        _invoke_with_supported_kwargs(
            process_probe_fn,
            os_name=eff_os,
            timeout_s=SUBPROCESS_TIMEOUT_S,
        )
    )
    live_detection = detect_steam_dx12_activity(
        live_procs,
        non_worker_gpu_util_pct=gpu.util_pct if (gpu.present and not gpu.parse_error) else None,
    )

    signatures_verified = all(
        detect_steam_dx12_activity([sig])["game_detected"] for sig in PREFLIGHT_GAME_SIGNATURES
    ) and detect_steam_dx12_activity([], non_worker_gpu_util_pct=20.0)["game_detected"]

    trigger_exe = live_detection["matched_games"][0] if live_detection["matched_games"] else "cs2.exe"
    t_drill0 = clock_fn()
    raw_drill = _invoke_with_supported_kwargs(
        worker_yield_fn,
        clock_fn=clock_fn,
        trigger_exe=trigger_exe,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    measured_ms = round(max(0.0, (clock_fn() - t_drill0) * 1000.0), 3)

    if isinstance(raw_drill, (int, float)) and not isinstance(raw_drill, bool):
        yield_latency_ms: float | None = round(float(raw_drill), 3)
        worker_started = True
        worker_stopped = True
        clean_release = True
        retained_mib = 0
        drill_method = "custom_worker_yield_fn"
    elif isinstance(raw_drill, Mapping):
        rep_ms = _optional_finite_float(
            raw_drill.get("yield_latency_ms", raw_drill.get("yield_ms"))
        )
        if rep_ms is None:
            rep_s = _optional_finite_float(raw_drill.get("yield_elapsed_s"))
            rep_ms = (rep_s * 1000.0) if rep_s is not None else None
        worker_started = bool(raw_drill.get("worker_started", True))
        worker_stopped = bool(raw_drill.get("worker_stopped", True))
        clean_release = bool(raw_drill.get("clean_release", worker_stopped))
        raw_ret = raw_drill.get("retained_worker_vram_mib", raw_drill.get("retained_vram_mib"))
        if raw_ret is None:
            retained_mib = 0 if clean_release else -1
        else:
            retained_mib = int(raw_ret)
        yield_latency_ms = round(rep_ms, 3) if rep_ms is not None else (measured_ms if worker_stopped else None)
        drill_method = str(raw_drill.get("method") or "custom_worker_yield_fn")
    else:
        raise AutoconfigError("worker_yield_fn must return a mapping or numeric latency.")

    yield_slo_ok = bool(
        signatures_verified
        and worker_started
        and worker_stopped
        and clean_release
        and retained_mib == 0
        and yield_latency_ms is not None
        and 0.0 <= yield_latency_ms < budget_ms
    )

    budget_s = budget_ms / 1000.0
    lat_s_str = f"{yield_latency_ms / 1000.0:.3f}s" if yield_latency_ms is not None else "n/a"

    if not gpu.present:
        thermal_ok = True
        vram_reserve_ok = True
        zero_game_lag = bool((not had_game_lag) and yield_slo_ok and thermal_ok)
        if not yield_slo_ok:
            status = "yield_slo_exceeded"
            check_status = "fail"
            preflight_ok = False
            summary = (
                f"Échec du test de cession Steam ({lat_s_str} >= budget {budget_s:.2f}s)."
                if is_fr
                else f"Steam worker yield SLO exceeded ({lat_s_str} >= {budget_s:.2f}s budget)."
            )
        else:
            status = "dev_api_only_no_nvidia_gpu"
            check_status = "ok"
            preflight_ok = True
            summary = (
                f"Aucun GPU NVIDIA détecté — statut 'dev_api_only_no_nvidia_gpu' "
                f"(API Dev prête, cession worker vérifiée en {lat_s_str} < {budget_s:.2f}s)."
                if is_fr
                else f"No NVIDIA GPU detected — honest 'dev_api_only_no_nvidia_gpu' status "
                f"(Dev API ready, worker yield verified in {lat_s_str} < {budget_s:.2f}s)."
            )
        gpu_ready = False
    elif gpu.parse_error or gpu.temp_c is None or gpu.free_vram_mib is None:
        status = "gpu_driver_parse_error"
        check_status = "fail"
        preflight_ok = False
        thermal_ok = False
        vram_reserve_ok = False
        zero_game_lag = False
        gpu_ready = False
        summary = (
            "Télémetrie `nvidia-smi` incomplète ou illisible — worker GPU maintenu en pause."
            if is_fr
            else "Unreadable or incomplete `nvidia-smi` telemetry — GPU worker held paused."
        )
    else:
        thermal_ok = bool(gpu.temp_c <= cap_c)
        vram_reserve_ok = bool(gpu.free_vram_mib >= reserve_mib)
        zero_game_lag = bool((not had_game_lag) and yield_slo_ok and thermal_ok)
        if not yield_slo_ok:
            status = "yield_slo_exceeded"
            check_status = "fail"
            preflight_ok = False
            gpu_ready = False
            summary = (
                f"Échec critique de préemption Steam : {lat_s_str} >= budget {budget_s:.2f}s."
                if is_fr
                else f"Steam GPU preemption SLO failed closed: {lat_s_str} >= {budget_s:.2f}s budget."
            )
        elif not thermal_ok and not vram_reserve_ok:
            status = "thermal_and_vram_clamped_paused"
            check_status = "warn"
            preflight_ok = True
            gpu_ready = False
            summary = (
                f"{gpu.name} à {gpu.temp_c:.1f} C (> {cap_c:.1f} C) et {gpu.free_vram_mib} MiB libres "
                f"(< {reserve_mib} MiB) : gouverneur en pause pour garder les ventilateurs silencieux."
                if is_fr
                else f"{gpu.name} at {gpu.temp_c:.1f} C (> {cap_c:.1f} C) and {gpu.free_vram_mib} MiB free "
                f"(< {reserve_mib} MiB): governor paused to keep fans quiet and protect VRAM."
            )
        elif not thermal_ok:
            status = "thermal_clamped_paused"
            check_status = "warn"
            preflight_ok = True
            gpu_ready = False
            summary = (
                f"{gpu.name} à {gpu.temp_c:.1f} C (> {cap_c:.1f} C plafond silencieux) : "
                f"worker GPU mis en pause jusqu'au retour <= {cap_c:.1f} C."
                if is_fr
                else f"{gpu.name} at {gpu.temp_c:.1f} C (> {cap_c:.1f} C quiet-fan ceiling): "
                f"GPU worker paused until <= {cap_c:.1f} C."
            )
        elif not vram_reserve_ok:
            status = "vram_reserve_hold"
            check_status = "warn"
            preflight_ok = True
            gpu_ready = False
            summary = (
                f"{gpu.name} a {gpu.free_vram_mib} MiB de VRAM libre (< {reserve_mib} MiB requis) : "
                "worker GPU maintenu en veille."
                if is_fr
                else f"{gpu.name} has {gpu.free_vram_mib} MiB free VRAM (< {reserve_mib} MiB reserve): "
                "GPU worker held paused."
            )
        elif live_detection["game_detected"]:
            status = "steam_game_active_yielded"
            check_status = "ok"
            preflight_ok = True
            gpu_ready = False
            games_note = ", ".join(live_detection["matched_games"]) or "gpu_util>15%"
            summary = (
                f"Jeu Steam/DX12 actif détecté ({games_note}) : cession complète vérifiée en "
                f"{lat_s_str} (< {budget_s:.2f}s), {gpu.temp_c:.1f} C <= {cap_c:.1f} C."
                if is_fr
                else f"Active Steam/DX12 game detected ({games_note}): full worker yield verified in "
                f"{lat_s_str} (< {budget_s:.2f}s), {gpu.temp_c:.1f} C <= {cap_c:.1f} C."
            )
        else:
            status = "whisper_ready"
            check_status = "ok"
            preflight_ok = True
            gpu_ready = True
            summary = (
                f"{gpu.name} prêt & silencieux ({gpu.temp_c:.1f} C <= {cap_c:.1f} C, "
                f"{gpu.free_vram_mib} MiB libres >= {reserve_mib} MiB, cession {lat_s_str} < {budget_s:.2f}s)."
                if is_fr
                else f"{gpu.name} quiet & ready ({gpu.temp_c:.1f} C <= {cap_c:.1f} C, "
                f"{gpu.free_vram_mib} MiB free >= {reserve_mib} MiB, yield {lat_s_str} < {budget_s:.2f}s)."
            )

    return _attach_receipt(
        {
            "object": "hivemind.noel_vscode_autoconfig.steam_thermal_preflight",
            "status": status,
            "check_status": check_status,
            "preflight_ok": preflight_ok,
            "gpu_contributor_ready": gpu_ready,
            "gpu_present": gpu.present,
            "gpu_name": gpu.name if (gpu.present and not gpu.parse_error) else None,
            "vram_total_mib": gpu.total_vram_mib if (gpu.present and not gpu.parse_error) else None,
            "vram_free_mib": gpu.free_vram_mib if (gpu.present and not gpu.parse_error) else None,
            "vram_used_mib": gpu.used_vram_mib if (gpu.present and not gpu.parse_error) else None,
            "temp_c": gpu.temp_c if (gpu.present and not gpu.parse_error) else None,
            "util_pct": gpu.util_pct if (gpu.present and not gpu.parse_error) else None,
            "max_temp_c": cap_c,
            "min_free_vram_mib": reserve_mib,
            "yield_budget_ms": budget_ms,
            "yield_latency_ms": yield_latency_ms,
            "yield_slo_ok": yield_slo_ok,
            "thermal_ok": thermal_ok,
            "vram_reserve_ok": vram_reserve_ok,
            "zero_game_lag": zero_game_lag,
            "had_game_lag": bool(had_game_lag),
            "game_detected": bool(live_detection["game_detected"]),
            "matched_games": list(live_detection["matched_games"]),
            "gpu_util_triggered": bool(live_detection["gpu_util_triggered"]),
            "signatures_verified": signatures_verified,
            "worker_drill": {
                "worker_started": worker_started,
                "worker_stopped": worker_stopped,
                "clean_release": clean_release,
                "retained_worker_vram_mib": retained_mib,
                "yield_latency_ms": yield_latency_ms,
                "trigger_exe": trigger_exe,
                "method": drill_method,
            },
            "summary": sanitize_text_for_display(summary, home=home),
        }
    )


def verify_producer_heartbeat_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    api_key: str | None = None,
    role: str = "prosumer",
    send_opentrainer_heartbeat: bool = False,
    host_label: str = "personal-pc",
    gpu_snapshot: Any = None,
    http_fn: HttpFn = default_http_json,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference Live Producer & OpenTrainer Heartbeat Readiness Check."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_role = normalize_role(role)
    clean_host_label = sanitize_host_label(host_label)

    clean_key = (api_key or "").strip()
    if clean_key and not is_valid_hivemind_key(clean_key):
        raise AutoconfigError("Invalid hm-... API key passed to verify_producer_heartbeat_readiness.")
    has_key = bool(clean_key)
    auth_headers = {"Authorization": f"Bearer {clean_key}"} if has_key else None

    gpu = _normalize_gpu_snapshot(gpu_snapshot)
    honest_gpu_name: str | None = None
    honest_vram_total: int | None = None
    honest_vram_free: int | None = None
    if (
        gpu.present
        and not gpu.parse_error
        and gpu.name
        and gpu.total_vram_mib is not None
        and gpu.total_vram_mib >= MIN_REPORTABLE_VRAM_TOTAL_MIB
        and gpu.free_vram_mib is not None
        and gpu.free_vram_mib >= 0
    ):
        honest_gpu_name = gpu.name[:128]
        honest_vram_total = int(gpu.total_vram_mib)
        honest_vram_free = min(int(gpu.free_vram_mib), honest_vram_total)

    diagnostics: list[str] = []

    # Optional outbound-only POST /v1/hivemind/opentrainer/heartbeat (extra='forbid')
    ot_sent = False
    ot_accepted = False
    ot_http_status: int | None = None
    ot_node_id: str | None = None
    ot_payload: dict[str, Any] | None = None
    ot_ok = True

    if not send_opentrainer_heartbeat:
        ot_status_label = "not_requested"
    elif not has_key:
        ot_status_label = "missing_api_key"
        ot_ok = False
        diagnostics.append(
            "Clé `hm-...` requise pour envoyer le heartbeat OpenTrainer sortant."
            if is_fr
            else "Valid `hm-...` API key required to send outbound OpenTrainer heartbeat."
        )
    elif norm_role == "consumer":
        ot_status_label = "skipped_consumer_role"
    else:
        ot_payload = {
            "host_label": clean_host_label,
            "gpu_name": honest_gpu_name,
            "vram_total_mib": honest_vram_total,
            "vram_free_mib": honest_vram_free,
            "network_mode": "outbound-https-relay",
        }
        ot_sent = True
        ot_http_status, ot_body = _safe_http_call(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
            ot_payload,
            auth_headers,
        )
        ot_accepted = ot_http_status in (200, 201)
        ot_ok = ot_accepted
        node_doc = ot_body.get("node") if isinstance(ot_body.get("node"), Mapping) else {}
        if isinstance(node_doc.get("node_id"), str) and node_doc["node_id"].strip():
            ot_node_id = sanitize_text_for_display(node_doc["node_id"].strip(), home=home)
        if ot_accepted:
            ot_status_label = "accepted"
        else:
            ot_status_label = "rejected"
            diagnostics.append(
                f"Échec sur POST /v1/hivemind/opentrainer/heartbeat (HTTP {ot_http_status})."
                if is_fr
                else f"Outbound heartbeat failed on POST /v1/hivemind/opentrainer/heartbeat (HTTP {ot_http_status})."
            )

    # Inspect GET /v1/hivemind/producers/fleet, GET /v1/hivemind/raid/status, and GET /v1/hivemind/remote/config
    fleet_code, fleet_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/producers/fleet",
        None,
        None,
    )
    fleet_ok = fleet_code == 200
    if not fleet_ok:
        diagnostics.append(
            f"Échec de lecture GET /v1/hivemind/producers/fleet (HTTP {fleet_code})."
            if is_fr
            else f"Fleet check failed on GET /v1/hivemind/producers/fleet (HTTP {fleet_code})."
        )

    raid_code, raid_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/raid/status",
        None,
        None,
    )
    raid_ok = raid_code == 200
    stale_raid_route_404 = raid_code == 404
    if not raid_ok:
        diagnostics.append(
            f"Échec de lecture GET /v1/hivemind/raid/status (HTTP {raid_code})."
            if is_fr
            else f"Raid status check failed on GET /v1/hivemind/raid/status (HTTP {raid_code})."
        )

    remote_checked = has_key
    remote_code: int | None = None
    remote_ok = True
    node_state: str | None = None
    node_paused: bool | None = None
    vram_cap_mib: int | None = None
    if has_key:
        remote_code, remote_body = _safe_http_call(
            http_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/remote/config",
            None,
            auth_headers,
        )
        remote_ok = remote_code == 200
        rcfg = remote_body.get("remote_config") if isinstance(remote_body.get("remote_config"), Mapping) else {}
        if remote_ok:
            node_state = str(rcfg.get("node_state") or "active")
            node_paused = bool(rcfg.get("paused", False))
            raw_cap = rcfg.get("vram_cap_mib")
            if isinstance(raw_cap, int) and not isinstance(raw_cap, bool):
                vram_cap_mib = raw_cap
        else:
            diagnostics.append(
                f"Échec de lecture authentifiée GET /v1/hivemind/remote/config (HTTP {remote_code})."
                if is_fr
                else f"Authenticated remote config check failed on GET /v1/hivemind/remote/config (HTTP {remote_code})."
            )

    swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}

    def _safe_nonneg_int(val: Any, default: int = 0) -> int:
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            return val
        return default

    online_producers = (
        _safe_nonneg_int(fleet_body.get("live_hiveminders"))
        if "live_hiveminders" in fleet_body
        else _safe_nonneg_int(swarm.get("live_producers"))
    )
    registered_producers = _safe_nonneg_int(
        fleet_body.get("total_hiveminders"),
        _safe_nonneg_int(swarm.get("registered_producers")),
    )
    distinct_gpus = _safe_nonneg_int(fleet_body.get("distinct_physical_gpus"))
    live_endpoints = _safe_nonneg_int(fleet_body.get("live_endpoints"))
    raw_live_models = fleet_body.get("live_models")
    live_models = (
        [str(m) for m in raw_live_models if isinstance(m, str) and m.strip()]
        if isinstance(raw_live_models, list)
        else []
    )
    active_trainers = _safe_nonneg_int(swarm.get("active_trainers"))
    registered_trainer_nodes = _safe_nonneg_int(swarm.get("registered_trainer_nodes"))

    activation_needed = online_producers == 0
    activation_command = (
        f"python3 scripts/hivemind_quickstart.py --gateway {clean_gw} --gamer-whisper"
        if activation_needed
        else None
    )
    if activation_needed:
        activation_guidance = (
            f"Aucun producteur d'inférence en ligne (`online_producers=0`) — activez votre PC perso en "
            f"1 commande silencieuse (<= 70.0 C, cession < 1.5s) : `{activation_command}`"
            if is_fr
            else f"Zero live inference producers online (`online_producers=0`) — activate your personal PC in "
            f"1 quiet Gamer-Whisper command (<= 70.0 C, < 1.5s game yield): `{activation_command}`"
        )
    else:
        activation_guidance = ""

    core_routes_ok = bool(fleet_ok and remote_ok and ot_ok)
    all_ok = bool(core_routes_ok and raid_ok)
    swarm_visible = bool(
        online_producers > 0
        or active_trainers > 0
        or registered_producers > 0
        or registered_trainer_nodes > 0
    )

    return _attach_receipt(
        {
            "object": "hivemind.noel_vscode_autoconfig.producer_heartbeat_readiness",
            "ok": all_ok,
            "readiness_ok": all_ok,
            "core_routes_ok": core_routes_ok,
            "stale_raid_route_404": stale_raid_route_404,
            "zero_inference_verified": True,
            "online_producers": online_producers,
            "registered_producers": registered_producers,
            "distinct_physical_gpus": distinct_gpus,
            "live_endpoints": live_endpoints,
            "live_models": live_models,
            "active_trainers": active_trainers,
            "registered_trainer_nodes": registered_trainer_nodes,
            "swarm_visible": swarm_visible,
            "remote_config": {
                "checked": remote_checked,
                "ok": remote_ok,
                "http_status": remote_code,
                "node_state": node_state,
                "paused": node_paused,
                "vram_cap_mib": vram_cap_mib,
            },
            "opentrainer_heartbeat": {
                "requested": bool(send_opentrainer_heartbeat),
                "sent": ot_sent,
                "accepted": ot_accepted,
                "http_status": ot_http_status,
                "status": ot_status_label,
                "node_id": ot_node_id,
                "host_label": clean_host_label,
                "network_mode": "outbound-https-relay",
                "gpu_name": honest_gpu_name,
                "vram_total_mib": honest_vram_total,
                "vram_free_mib": honest_vram_free,
                "payload": ot_payload,
            },
            "endpoints": {
                "producers_fleet": {"ok": fleet_ok, "http_status": fleet_code},
                "raid_status": {"ok": raid_ok, "http_status": raid_code},
                "remote_config": {
                    "checked": remote_checked,
                    "ok": remote_ok,
                    "http_status": remote_code,
                },
            },
            "activation_needed": activation_needed,
            "activation_command": activation_command,
            "activation_guidance": sanitize_text_for_display(activation_guidance, home=home),
            "diagnostics": [sanitize_text_for_display(d, home=home) for d in diagnostics],
        }
    )


def submit_post_setup_telemetry(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool | None = None,
    had_game_lag: bool = False,
    yield_slo_ok: bool = True,
    thermal_ok: bool = True,
    system_name: str | None = None,
    note: str = "",
    http_fn: HttpFn = default_http_json,
    timeout_s: float = TELEMETRY_TIMEOUT_S,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Timeout-safe, non-blocking zero-PII CES & `zero_game_lag` post-setup telemetry ping."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    if isinstance(profile, bool) or not isinstance(profile, str) or profile.strip().lower() not in VALID_PROFILES:
        raise AutoconfigError(f"Unsupported telemetry profile {profile!r}; expected 'dev' or 'gamer'.")
    norm_profile = profile.strip().lower()
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise AutoconfigError("ease_rating must be an integer between 1 and 5.")

    effective_zero_lag = (
        (not bool(had_game_lag))
        and bool(yield_slo_ok)
        and bool(thermal_ok)
        and (True if zero_game_lag is None else bool(zero_game_lag))
    )
    clean_platform = _platform_tag(system_name)
    clean_note = sanitize_text_for_display(str(note or ""), home=home).strip()[:240]

    payload: dict[str, Any] = {
        "profile": norm_profile,
        "ease_rating": int(ease_rating),
        "zero_game_lag": effective_zero_lag,
        "platform": clean_platform,
    }
    if clean_note:
        payload["note"] = clean_note

    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        None,
        timeout_s=timeout_s,
    )

    telemetry_recorded = status_code in (200, 201)
    stale_cloud_run = status_code == 404
    raw_mean = body.get("mean_ease_rating")
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )
    raw_total = body.get("total_responses")
    total_resp = (
        int(raw_total)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )
    raw_lag_ratio = body.get("zero_game_lag_ratio")
    lag_ratio = (
        round(float(raw_lag_ratio), 4)
        if isinstance(raw_lag_ratio, (int, float)) and not isinstance(raw_lag_ratio, bool)
        else None
    )

    if telemetry_recorded:
        outcome = "recorded"
        degraded_reason: str | None = None
        msg = (
            f"Télémétrie CES post-installation enregistrée (profil={norm_profile}, ease_rating={ease_rating}/5)."
            if is_fr
            else f"Anonymous post-setup CES telemetry recorded (profile={norm_profile}, ease_rating={ease_rating}/5)."
        )
    elif status_code == 404:
        outcome = "missing_route"
        degraded_reason = "stale_cloud_run_404"
        msg = (
            "Télémétrie post-installation ignorée sans bloquer : révision Cloud Run en retard sur `main` (HTTP 404)."
            if is_fr
            else "Post-setup telemetry skipped without blocking setup: stale Cloud Run revision returned HTTP 404."
        )
    elif status_code == 429:
        outcome = "rate_limited"
        degraded_reason = "rate_limited_429"
        msg = (
            "Télémétrie post-installation ignorée sans bloquer : limite de débit atteinte (HTTP 429)."
            if is_fr
            else "Post-setup telemetry skipped without blocking setup: rate limit reached (HTTP 429)."
        )
    elif status_code >= 500:
        outcome = "server_error"
        degraded_reason = "gateway_5xx"
        msg = (
            f"Télémétrie post-installation ignorée sans bloquer : erreur passerelle (HTTP {status_code})."
            if is_fr
            else f"Post-setup telemetry skipped without blocking setup: gateway server error (HTTP {status_code})."
        )
    elif status_code == 0:
        outcome = "network_error"
        degraded_reason = "network_timeout_or_unreachable"
        msg = (
            "Télémétrie post-installation ignorée sans bloquer : délai dépassé ou passerelle injoignable."
            if is_fr
            else "Post-setup telemetry skipped without blocking setup: network timeout or gateway unreachable."
        )
    else:
        outcome = "client_error"
        degraded_reason = f"http_{status_code}"
        msg = (
            f"Télémétrie post-installation ignorée sans bloquer (HTTP {status_code})."
            if is_fr
            else f"Post-setup telemetry skipped without blocking setup (HTTP {status_code})."
        )

    return _attach_receipt(
        {
            "object": "hivemind.noel_vscode_autoconfig.post_setup_telemetry",
            "attempted": True,
            "telemetry_recorded": telemetry_recorded,
            "submitted": telemetry_recorded,
            "non_blocking": True,
            "degraded": not telemetry_recorded,
            "degraded_reason": degraded_reason,
            "stale_cloud_run": stale_cloud_run,
            "status": outcome,
            "http_status": status_code,
            "profile": norm_profile,
            "ease_rating": int(ease_rating),
            "zero_game_lag": effective_zero_lag,
            "yield_slo_ok": bool(yield_slo_ok),
            "thermal_ok": bool(thermal_ok),
            "platform": clean_platform,
            "mean_ease_rating": mean_ease,
            "total_responses": total_resp,
            "zero_game_lag_ratio": lag_ratio,
            "zero_pii_payload": payload,
            "message": sanitize_text_for_display(msg, home=home),
        }
    )


def run_vscode_autoconfig(
    *,
    gateway: str = DEFAULT_GATEWAY,
    handle: str | None = None,
    role: str = "prosumer",
    explicit_api_key: str | None = None,
    credentials_path: Path | None = None,
    continue_path: Path | None = None,
    workspace: Path | None = None,
    vscode_settings_path: Path | None = None,
    write_vscode_settings: bool | None = None,
    uninstall: bool = False,
    skip_probe: bool = False,
    ease_rating: int | None = None,
    prompt_ces: bool = False,
    had_game_lag: bool = False,
    quick_connect: bool = False,
    check_producer_heartbeat: bool = False,
    send_opentrainer_heartbeat: bool = False,
    host_label: str = "personal-pc",
    gamer_preflight: bool = False,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    post_setup_telemetry: bool = False,
    profile: str | None = None,
    gpu_snapshot: Any = None,
    system_name: str | None = None,
    lang: str = "en",
    os_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    http_fn: HttpFn = default_http_json,
    probe_fn: ProbeFn = default_probe_fn,
    input_fn: InputFn = input,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: Callable[[], float] = time.monotonic,
) -> tuple[int, dict[str, Any]]:
    """Run the 1-command VS Code developer autoconfig or Quick-Connect flow and return `(exit_code, receipted_report)`."""
    t0 = clock_fn()
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    resolved_cwd = cwd if cwd is not None else Path.cwd()
    run_mode = "quick_connect" if quick_connect else "autoconfig"

    cred_target = (
        credentials_path
        if credentials_path is not None
        else (resolved_home / ".hivemind" / "credentials.json")
    )
    cont_target = (
        continue_path
        if continue_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )

    eff_workspace = workspace if workspace is not None else resolved_cwd
    should_write_vscode = (
        bool(write_vscode_settings)
        if write_vscode_settings is not None
        else (workspace is not None or vscode_settings_path is not None)
    )
    vs_target: Path | None
    if vscode_settings_path is not None:
        vs_target = vscode_settings_path
    elif should_write_vscode or uninstall:
        vs_target = eff_workspace / ".vscode" / "settings.json"
    else:
        vs_target = None

    if uninstall:
        un_res = uninstall_hivemind_autoconfig(
            credentials_path=cred_target,
            continue_path=cont_target,
            vscode_settings_path=vs_target,
            os_name=eff_os,
            home=resolved_home,
            lang=lang_code,
        )
        elapsed_un = round(max(0.0, clock_fn() - t0), 3)
        uninstall_report = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "uninstall",
                "lang": lang_code,
                "gateway": clean_gw,
                "compliance_ok": True,
                "ready": True,
                "verdict": "uninstalled",
                "configured_targets_count": 0,
                "checks_passed_count": 1,
                "checks_total_count": 1,
                "warnings_count": 0,
                "failures_count": 0,
                "ease_rating": None,
                "mean_ease_rating": None,
                "zero_game_lag": None,
                "telemetry_recorded": False,
                "zero_inference_verified": True,
                "within_60s_slo": elapsed_un < SLO_BUDGET_SECONDS,
                "elapsed_s": elapsed_un,
                "compliance": {
                    "compliance_ok": True,
                    "corporate_detected": False,
                    "corporate_indicators": [],
                    "matched_markers": [],
                    "personal_pc_only": True,
                    "amadeus_corporate_prohibition": True,
                    "network_transmitted": False,
                    "financial_promise": FINANCIAL_PROMISE,
                    "monetary_cost": FINANCIAL_PROMISE,
                    "license": "AGPL-3.0",
                    "notice_en": COMPLIANCE_NOTICE_EN,
                    "notice_fr": COMPLIANCE_NOTICE_FR,
                    "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
                },
                "account": None,
                "targets": {},
                "uninstall": un_res,
                "gitignore_safety": None,
                "gamer_preflight": None,
                "producer_heartbeat": None,
                "post_setup_telemetry": None,
                "probe_skipped": True,
                "probe": None,
                "feedback": None,
                "warnings": [],
                "errors": [],
            }
        )
        return 0, uninstall_report

    if ease_rating is not None and (
        isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5)
    ):
        raise AutoconfigError("ease_rating must be an integer between 1 and 5.")
    if profile is not None and (
        isinstance(profile, bool) or not isinstance(profile, str) or profile.strip().lower() not in VALID_PROFILES
    ):
        raise AutoconfigError(f"Unsupported profile {profile!r}; expected 'dev' or 'gamer'.")
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise AutoconfigError("max_temp_c must be a finite temperature between 25.0 and 95.0 C.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise AutoconfigError("min_free_vram_mib must be an integer >= 256 MiB.")
    if isinstance(yield_budget_ms, bool) or not math.isfinite(float(yield_budget_ms)) or float(yield_budget_ms) <= 0.0:
        raise AutoconfigError("yield_budget_ms must be a positive finite budget.")

    # Step 1: Strictly local Personal-PC & Hobby-Workspace Compliance Gate BEFORE any write or network call
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        workspace=eff_workspace,
        lang=lang_code,
    )
    if not compliance["compliance_ok"]:
        elapsed_refused = round(max(0.0, clock_fn() - t0), 3)
        refused_report = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": run_mode,
                "lang": lang_code,
                "gateway": clean_gw,
                "compliance_ok": False,
                "ready": False,
                "verdict": "refused_compliance",
                "configured_targets_count": 0,
                "checks_passed_count": 0,
                "checks_total_count": 1,
                "warnings_count": 0,
                "failures_count": 1,
                "ease_rating": None,
                "mean_ease_rating": None,
                "zero_game_lag": False,
                "telemetry_recorded": False,
                "zero_inference_verified": True,
                "within_60s_slo": elapsed_refused < SLO_BUDGET_SECONDS,
                "elapsed_s": elapsed_refused,
                "compliance": compliance,
                "account": None,
                "targets": {},
                "uninstall": None,
                "gitignore_safety": None,
                "gamer_preflight": None,
                "producer_heartbeat": None,
                "post_setup_telemetry": None,
                "probe_skipped": True,
                "probe": None,
                "feedback": None,
                "warnings": [],
                "errors": [compliance["summary"]],
            }
        )
        return 2, refused_report

    # Pre-validate existing .vscode/settings.json BEFORE network registration or writing any files
    existing_vs_text: str | None = None
    if should_write_vscode and vs_target is not None and vs_target.is_file():
        existing_vs_text = vs_target.read_text(encoding="utf-8")
        _ = build_merged_vscode_settings_json(existing_vs_text, gateway=clean_gw)

    # Step 2: Load or register personal account & verify /v1/hivemind/me + /v1/models (zero inference)
    connected = connect_or_register_dev_account(
        gateway=clean_gw,
        credentials_path=cred_target,
        handle=handle,
        role=role,
        explicit_api_key=explicit_api_key,
        environ=environ,
        http_fn=http_fn,
        os_name=eff_os,
        home=resolved_home,
        lang=lang_code,
    )
    if not connected["ok"]:
        elapsed_err = round(max(0.0, clock_fn() - t0), 3)
        err_text = sanitize_text_for_display(str(connected["error"]), home=resolved_home)
        http_code = int(connected.get("http_status") or 0)
        fail_rc = 2 if http_code == 404 else 1
        conn_fail_report = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": run_mode,
                "lang": lang_code,
                "gateway": clean_gw,
                "compliance_ok": True,
                "ready": False,
                "verdict": "not_ready",
                "configured_targets_count": 0,
                "checks_passed_count": 1,
                "checks_total_count": 2,
                "warnings_count": 0,
                "failures_count": 1,
                "ease_rating": None,
                "mean_ease_rating": None,
                "zero_game_lag": None,
                "telemetry_recorded": False,
                "zero_inference_verified": True,
                "within_60s_slo": elapsed_err < SLO_BUDGET_SECONDS,
                "elapsed_s": elapsed_err,
                "compliance": compliance,
                "account": None,
                "targets": {},
                "uninstall": None,
                "gitignore_safety": None,
                "gamer_preflight": None,
                "producer_heartbeat": None,
                "post_setup_telemetry": None,
                "probe_skipped": True,
                "probe": None,
                "feedback": None,
                "warnings": [],
                "errors": [err_text],
            }
        )
        return fail_rc, conn_fail_report

    raw_api_key = str(connected["api_key"])
    verified_role = str(connected["role"])
    warnings: list[str] = []
    errors: list[str] = []

    # Step 3(a): Atomically + idempotently write ~/.continue/hivemind_provider.yaml (0600 + .bak)
    continue_yaml = build_continue_provider_yaml(clean_gw, raw_api_key)
    continue_write = atomic_write_config_file(
        cont_target,
        continue_yaml,
        posix_mode=0o600,
        create_backup=True,
        os_name=eff_os,
        home=resolved_home,
    )

    targets: dict[str, Any] = {
        "credentials": connected["credentials_write"],
        "continue_provider": continue_write,
    }
    configured_targets_count = 2
    gitignore_safety: dict[str, Any] | None = None

    # Step 3(b): Optional workspace .vscode/settings.json (non-secret defaults + .gitignore check)
    if should_write_vscode and vs_target is not None:
        merged_vs_json = build_merged_vscode_settings_json(existing_vs_text, gateway=clean_gw)
        vs_write = atomic_write_config_file(
            vs_target,
            merged_vs_json,
            posix_mode=0o600,
            create_backup=True,
            os_name=eff_os,
            home=resolved_home,
        )
        targets["vscode_settings"] = vs_write
        configured_targets_count += 1
        gitignore_safety = check_workspace_gitignore_safety(eff_workspace, lang=lang_code)
        if not gitignore_safety["safe"] and gitignore_safety["warning"]:
            warnings.append(gitignore_safety["warning"])
    else:
        targets["vscode_settings"] = {
            "path": redact_home_path(eff_workspace / ".vscode" / "settings.json", home=resolved_home),
            "action": "skipped",
            "written": False,
            "created": False,
            "updated": False,
            "backup_created": False,
            "backup_path": None,
        }

    # Step 4: Optional <1.5s Steam GPU Yield & <=70.0 C Quiet-Fan Thermal Preflight
    run_preflight = bool(quick_connect or gamer_preflight)
    preflight_result: dict[str, Any] | None = None
    preflight_failed = False
    if run_preflight:
        effective_gpu_probe: GpuProbeFn = (
            (lambda: gpu_snapshot) if gpu_snapshot is not None else gpu_probe_fn
        )
        preflight_result = run_steam_thermal_preflight(
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_budget_ms=yield_budget_ms,
            had_game_lag=had_game_lag,
            gpu_probe_fn=effective_gpu_probe,
            process_probe_fn=process_probe_fn,
            worker_yield_fn=worker_yield_fn,
            clock_fn=clock_fn,
            os_name=eff_os,
            home=resolved_home,
            lang=lang_code,
        )
        if preflight_result["check_status"] == "fail":
            preflight_failed = True
            errors.append(preflight_result["summary"])
        elif preflight_result["check_status"] == "warn":
            warnings.append(preflight_result["summary"])

    # Step 5: Optional Live Producer & OpenTrainer Heartbeat Readiness Check
    run_hb_check = bool(quick_connect or check_producer_heartbeat or send_opentrainer_heartbeat)
    producer_hb_result: dict[str, Any] | None = None
    hb_failed = False
    hb_failure_http: int | None = None
    if run_hb_check:
        hb_gpu_source: Any = gpu_snapshot
        if hb_gpu_source is None:
            if preflight_result is not None:
                hb_gpu_source = preflight_result
            elif send_opentrainer_heartbeat:
                hb_gpu_source = _invoke_with_supported_kwargs(gpu_probe_fn, timeout_s=SUBPROCESS_TIMEOUT_S)
        producer_hb_result = verify_producer_heartbeat_readiness(
            clean_gw,
            api_key=raw_api_key,
            role=verified_role,
            send_opentrainer_heartbeat=bool(send_opentrainer_heartbeat),
            host_label=host_label,
            gpu_snapshot=hb_gpu_source,
            http_fn=http_fn,
            home=resolved_home,
            lang=lang_code,
        )
        if not producer_hb_result["ok"]:
            if (
                quick_connect
                and not check_producer_heartbeat
                and producer_hb_result["core_routes_ok"]
                and producer_hb_result.get("stale_raid_route_404")
            ):
                for diag in producer_hb_result["diagnostics"]:
                    if diag not in warnings:
                        warnings.append(diag)
            else:
                hb_failed = True
                for diag in producer_hb_result["diagnostics"]:
                    if diag not in errors:
                        errors.append(diag)
                ep_map = producer_hb_result["endpoints"]
                ot_code = producer_hb_result["opentrainer_heartbeat"].get("http_status")
                candidate_codes = [
                    ep_map["producers_fleet"]["http_status"],
                    ep_map["raid_status"]["http_status"],
                    ep_map["remote_config"]["http_status"],
                    ot_code,
                ]
                hb_failure_http = 404 if 404 in candidate_codes else 503

    # Step 6: CES feedback or non-blocking --post-setup-telemetry ping
    eff_yield_ok = bool(preflight_result["yield_slo_ok"]) if preflight_result is not None else True
    eff_thermal_ok = bool(preflight_result["thermal_ok"]) if preflight_result is not None else True
    derived_zero_lag = (not bool(had_game_lag)) and eff_yield_ok and eff_thermal_ok

    fb_result: dict[str, Any] | None = None
    post_telemetry_result: dict[str, Any] | None = None
    if post_setup_telemetry:
        if profile is not None:
            resolved_profile = profile.strip().lower()
        elif (
            send_opentrainer_heartbeat and bool(preflight_result and preflight_result["gpu_present"])
        ) or verified_role in ("producer", "trainer"):
            resolved_profile = "gamer"
        else:
            resolved_profile = "dev"

        eff_ease = ease_rating if ease_rating is not None else 5
        if prompt_ces:
            prompt_ans = ""
            prompt_label = (
                "[HiveMind CES 1-question] Notez la simplicité Quick-Connect 1..5 (Entrée=5) : "
                if lang_code == "fr"
                else "[HiveMind 1-prompt CES] Rate Quick-Connect ease 1..5 (Enter=5): "
            )
            try:
                prompt_ans = input_fn(prompt_label)
            except (EOFError, KeyboardInterrupt):
                prompt_ans = ""
            parsed_ces = parse_dev_ces_input(prompt_ans, default_had_game_lag=not derived_zero_lag)
            if not parsed_ces["skipped"] and parsed_ces["ease_rating"] is not None:
                eff_ease = int(parsed_ces["ease_rating"])
                derived_zero_lag = bool(derived_zero_lag and parsed_ces["zero_game_lag"])

        post_telemetry_result = submit_post_setup_telemetry(
            clean_gw,
            profile=resolved_profile,
            ease_rating=eff_ease,
            zero_game_lag=derived_zero_lag,
            had_game_lag=had_game_lag,
            yield_slo_ok=eff_yield_ok,
            thermal_ok=eff_thermal_ok,
            system_name=system_name,
            http_fn=http_fn,
            home=resolved_home,
            lang=lang_code,
        )
        if not post_telemetry_result["telemetry_recorded"]:
            warnings.append(post_telemetry_result["message"])
    elif prompt_ces:
        fb_result = prompt_and_submit_dev_ces(
            gateway=clean_gw,
            default_had_game_lag=not derived_zero_lag,
            system_name=system_name,
            input_fn=input_fn,
            http_fn=http_fn,
            lang=lang_code,
        )
    elif ease_rating is not None:
        fb_result = submit_dev_ces_feedback(
            gateway=clean_gw,
            ease_rating=ease_rating,
            zero_game_lag=derived_zero_lag,
            system_name=system_name,
            http_fn=http_fn,
            lang=lang_code,
        )

    if fb_result is not None and fb_result["attempted"] and not fb_result["submitted"]:
        errors.append(sanitize_text_for_display(str(fb_result["message"]), home=resolved_home))

    # Step 7: Chained sibling probe (skipped automatically in streamlined --quick-connect mode)
    eff_skip_probe = bool(skip_probe or quick_connect)
    probe_summary: dict[str, Any] | None = None
    if not eff_skip_probe:
        raw_probe = _invoke_probe_callable(
            probe_fn,
            gateway=clean_gw,
            credentials_path=cred_target,
            continue_path=cont_target,
            http_fn=http_fn,
            feedback_result=post_telemetry_result or fb_result,
            lang=lang_code,
            os_name=eff_os,
            environ=environ,
            home=resolved_home,
        )
        probe_summary = _summarize_probe_report(raw_probe, home=resolved_home)
        if not probe_summary["all_routes_ok"] or probe_summary["failures_count"] > 0:
            for diag in probe_summary["diagnostics"][:3]:
                if diag not in errors:
                    errors.append(diag)
            if not errors:
                errors.append(
                    "Échec de la sonde post-installation (routes ou contrôles locaux)."
                    if lang_code == "fr"
                    else "Chained post-setup probe reported readiness failure."
                )

    # Compute aggregated check counts without ever emitting forbidden keys
    local_checks_total = 2 + (1 if should_write_vscode else 0)  # compliance + account/continue [+ vscode]
    local_checks_passed = local_checks_total
    preflight_total = 1 if preflight_result is not None else 0
    preflight_passed = 1 if (preflight_result is not None and not preflight_failed) else 0
    hb_total = 1 if producer_hb_result is not None else 0
    hb_passed = 1 if (producer_hb_result is not None and not hb_failed) else 0
    probe_passed = probe_summary["checks_passed_count"] if probe_summary is not None else 0
    probe_total = probe_summary["checks_total_count"] if probe_summary is not None else 0
    fb_total = 1 if (fb_result is not None and fb_result["attempted"]) else 0
    fb_passed = 1 if (fb_result is not None and fb_result["submitted"]) else 0

    checks_passed_count = local_checks_passed + preflight_passed + hb_passed + probe_passed + fb_passed
    checks_total_count = local_checks_total + preflight_total + hb_total + probe_total + fb_total

    probe_failed = bool(
        probe_summary is not None
        and (not probe_summary["all_routes_ok"] or probe_summary["failures_count"] > 0)
    )
    fb_failed = bool(fb_result is not None and fb_result["attempted"] and not fb_result["submitted"])
    failures_count = (
        (1 if preflight_failed else 0)
        + (1 if hb_failed else 0)
        + (1 if probe_failed else 0)
        + (1 if fb_failed else 0)
    )
    warnings_count = len(warnings) + (probe_summary["warnings_count"] if probe_summary is not None else 0)

    ready = failures_count == 0
    if not ready:
        verdict = "not_ready"
    elif warnings_count > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"

    active_fb = post_telemetry_result if post_telemetry_result is not None else fb_result
    mean_ease = (
        active_fb.get("mean_ease_rating")
        if (active_fb is not None and active_fb.get("mean_ease_rating") is not None)
        else (probe_summary.get("mean_ease_rating") if probe_summary is not None else None)
    )
    ease_val = active_fb.get("ease_rating") if active_fb is not None else None
    telemetry_recorded = bool(
        (post_telemetry_result and post_telemetry_result["telemetry_recorded"])
        or (fb_result and fb_result.get("submitted"))
    )
    elapsed_s = round(max(0.0, clock_fn() - t0), 3)

    account_public = {
        "account_id": connected["account_id"],
        "handle": connected["handle"],
        "role": connected["role"],
        "masked_key": connected["masked_key"],
        "created_account": connected["created_account"],
        "models": connected["models"],
        "models_count": connected["models_count"],
    }

    report = _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": run_mode,
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "ready": ready,
            "verdict": verdict,
            "configured_targets_count": configured_targets_count,
            "checks_passed_count": checks_passed_count,
            "checks_total_count": checks_total_count,
            "warnings_count": warnings_count,
            "failures_count": failures_count,
            "ease_rating": ease_val,
            "mean_ease_rating": mean_ease,
            "zero_game_lag": derived_zero_lag,
            "telemetry_recorded": telemetry_recorded,
            "zero_inference_verified": True,
            "within_60s_slo": elapsed_s < SLO_BUDGET_SECONDS,
            "elapsed_s": elapsed_s,
            "compliance": compliance,
            "account": account_public,
            "targets": targets,
            "uninstall": None,
            "gitignore_safety": gitignore_safety,
            "gamer_preflight": preflight_result,
            "producer_heartbeat": producer_hb_result,
            "post_setup_telemetry": post_telemetry_result,
            "probe_skipped": eff_skip_probe,
            "probe": probe_summary,
            "feedback": fb_result if fb_result is not None else post_telemetry_result,
            "warnings": warnings,
            "errors": errors,
        }
    )

    if preflight_failed or probe_failed:
        return 2, report
    if hb_failed:
        return (2 if hb_failure_http == 404 else 1), report
    if fb_failed:
        return (2 if fb_result and fb_result.get("http_status") == 404 else 1), report
    return 0, report


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render FR/EN GitHub-flavored Markdown autoconfig summary with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    title = (
        "# Autoconfiguration VS Code Développeur HiveMind Noël"
        if is_fr
        else "# HiveMind Holiday Developer VS Code Autoconfig"
    )
    ease_val = report.get("ease_rating")
    ease_str = f"{ease_val}/5" if ease_val is not None else "n/a"
    mean_val = report.get("mean_ease_rating")
    mean_str = f"{mean_val}/5" if mean_val is not None else "n/a"

    lines: list[str] = [
        title,
        "",
        f"> **Compliance / Conformité**: {bilingual_notice}",
        "",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **Compliance OK**: `{report.get('compliance_ok')}`",
        f"- **Ready**: `{report.get('ready')}` (`{report.get('verdict')}`)",
        f"- **Configured Targets**: `{report.get('configured_targets_count')}`",
        f"- **Checks Passed**: `{report.get('checks_passed_count')}/{report.get('checks_total_count')}`",
        f"- **CES Ease Rating**: `{ease_str}` (Community Mean: `{mean_str}`)",
    ]

    acct = report.get("account")
    if isinstance(acct, Mapping):
        lines.extend(
            [
                "",
                "## Compte Personnel & Modèles" if is_fr else "## Personal Account & Models",
                "",
                f"- **Handle**: `{acct.get('handle')}` (`{acct.get('role')}`)",
                f"- **API Key (Masked)**: `{acct.get('masked_key')}`",
                f"- **Models Verified**: `{', '.join(acct.get('models') or DEFAULT_MODELS)}`",
            ]
        )

    targets = report.get("targets")
    if isinstance(targets, Mapping) and targets:
        lines.extend(
            [
                "",
                "## Fichiers Configurés (Atomiques & 0600)" if is_fr else "## Configured Files (Atomic & 0600)",
                "",
            ]
        )
        for name, info in targets.items():
            if isinstance(info, Mapping):
                bak_note = f" (backup: `{info['backup_path']}`)" if info.get("backup_path") else ""
                lines.append(f"- **{name}**: `{info.get('path')}` — `{info.get('action')}`{bak_note}")

    un_info = report.get("uninstall")
    if isinstance(un_info, Mapping):
        removed = ", ".join(f"`{p}`" for p in (un_info.get("removed_files") or ())) or "none"
        lines.extend(
            [
                "",
                "## Désinstallation Propre" if is_fr else "## Clean Uninstall",
                "",
                f"- **Removed**: {removed}",
            ]
        )

    preflight_info = report.get("gamer_preflight")
    if isinstance(preflight_info, Mapping):
        lines.extend(
            [
                "",
                "## Pré-Vol Steam (<1.5s) & Ventilateurs Silencieux (<=70 C)"
                if is_fr
                else "## Steam Yield (<1.5s) & Quiet-Fan Thermal Preflight (<=70 C)",
                "",
                f"- **Status**: `{preflight_info.get('status')}` "
                f"(`yield_slo_ok={preflight_info.get('yield_slo_ok')}`, "
                f"`thermal_ok={preflight_info.get('thermal_ok')}`, "
                f"`vram_reserve_ok={preflight_info.get('vram_reserve_ok')}`, "
                f"`zero_game_lag={preflight_info.get('zero_game_lag')}`)",
                f"- **Summary**: {preflight_info.get('summary')}",
            ]
        )

    hb_info = report.get("producer_heartbeat")
    if isinstance(hb_info, Mapping):
        ot_sub = hb_info.get("opentrainer_heartbeat") if isinstance(hb_info.get("opentrainer_heartbeat"), Mapping) else {}
        lines.extend(
            [
                "",
                "## État Essaim Producteurs & Heartbeat OpenTrainer"
                if is_fr
                else "## Live Producer Fleet & OpenTrainer Heartbeat",
                "",
                f"- **Swarm Visibility**: `online_producers={hb_info.get('online_producers')}`, "
                f"`active_trainers={hb_info.get('active_trainers')}`, "
                f"`opentrainer_heartbeat={ot_sub.get('status')}`",
            ]
        )
        if hb_info.get("activation_guidance"):
            lines.append(f"- **Gamer-Whisper 1-Command**: {hb_info.get('activation_guidance')}")

    probe_info = report.get("probe")
    if isinstance(probe_info, Mapping):
        lines.extend(
            [
                "",
                "## Sonde Post-Installation (Zéro Inférence)"
                if is_fr
                else "## Chained Post-Setup Probe (Zero-Inference)",
                "",
                f"- **Verdict**: `{probe_info.get('verdict')}` "
                f"(routes `{probe_info.get('routes_passed_count')}/{probe_info.get('routes_total_count')}`, "
                f"checks `{probe_info.get('checks_passed_count')}/{probe_info.get('checks_total_count')}`)",
            ]
        )

    fb = report.get("feedback")
    if isinstance(fb, Mapping):
        lines.extend(
            [
                "",
                "## Retour Anonyme CES Développeur" if is_fr else "## Anonymous Developer CES Feedback",
                "",
                f"- **{fb.get('status')}**: {fb.get('message')}",
            ]
        )

    for warn_msg in report.get("warnings") or ():
        lines.append(f"\n> **Warning**: {warn_msg}")

    for err_msg in report.get("errors") or ():
        lines.append(f"\n> **Error**: {err_msg}")

    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render concise human-readable CLI output with masked keys, `~` paths, and SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    banner = (
        "HIVEMIND NOËL — AUTOCONFIG VS CODE (100 % PC PERSO • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND HOLIDAY — DEV VS CODE AUTOCONFIG (100% PERSONAL PC • 0.00 EUR / 0.00 USD)"
    )
    lines: list[str] = [
        "==========================================================================",
        f"  {banner}",
        "==========================================================================",
        f"  [Compliance] {bilingual_notice}",
        f"  Gateway              : {report.get('gateway')}",
        f"  Compliance OK        : {report.get('compliance_ok')}",
        f"  Ready                : {report.get('ready')} ({report.get('verdict')})",
        f"  Configured targets   : {report.get('configured_targets_count')}",
        f"  Checks passed        : {report.get('checks_passed_count')}/{report.get('checks_total_count')}",
    ]

    acct = report.get("account")
    if isinstance(acct, Mapping):
        lines.append(f"  Account              : {acct.get('handle')} ({acct.get('role')}, key {acct.get('masked_key')})")

    targets = report.get("targets")
    if isinstance(targets, Mapping):
        for name, info in targets.items():
            if isinstance(info, Mapping) and info.get("action") != "skipped":
                lines.append(f"  Target [{name:<17}]: {info.get('path')} ({info.get('action')})")

    preflight_info = report.get("gamer_preflight")
    if isinstance(preflight_info, Mapping):
        lines.append(
            f"  Gamer preflight      : {preflight_info.get('status')} "
            f"(yield_slo_ok={preflight_info.get('yield_slo_ok')}, "
            f"thermal_ok={preflight_info.get('thermal_ok')}, "
            f"zero_game_lag={preflight_info.get('zero_game_lag')})"
        )

    hb_info = report.get("producer_heartbeat")
    if isinstance(hb_info, Mapping):
        ot_sub = hb_info.get("opentrainer_heartbeat") if isinstance(hb_info.get("opentrainer_heartbeat"), Mapping) else {}
        lines.append(
            f"  Swarm heartbeat      : online_producers={hb_info.get('online_producers')}, "
            f"active_trainers={hb_info.get('active_trainers')}, "
            f"opentrainer={ot_sub.get('status')}"
        )
        if hb_info.get("activation_guidance"):
            lines.append(f"  [TIP] {hb_info.get('activation_guidance')}")

    un_info = report.get("uninstall")
    if isinstance(un_info, Mapping):
        lines.append(f"  Removed files        : {un_info.get('removed_count', 0)}")

    for warn_msg in report.get("warnings") or ():
        lines.append(f"  [WARN] {warn_msg}")

    for err_msg in report.get("errors") or ():
        lines.append(f"  [ERROR] {err_msg}")

    lines.append("  Clean uninstall      : python3 scripts/noel_vscode_autoconfig.py --uninstall")
    lines.append(f"  SHA-256 Receipt      : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_vscode_autoconfig.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind <60s 1-command Developer VS Code autoconfig with Personal-PC compliance "
            "and zero-inference post-setup probe hook (100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind gateway URL (https://*.run.app or localhost).")
    parser.add_argument("--handle", default="", help="Optional personal handle (2-48 ASCII chars).")
    parser.add_argument(
        "--role",
        default="prosumer",
        choices=("consumer", "producer", "prosumer", "trainer"),
        help="HiveMind account role (default: prosumer).",
    )
    parser.add_argument("--api-key", default="", help="Optional existing hm-... personal API key.")
    parser.add_argument("--credentials-file", default="", help="Path to ~/.hivemind/credentials.json.")
    parser.add_argument("--continue-config-path", default="", help="Path to ~/.continue/hivemind_provider.yaml.")
    parser.add_argument("--workspace", default="", help="Target hobby project workspace directory.")
    parser.add_argument("--vscode-settings-path", default="", help="Explicit path to .vscode/settings.json.")
    vs_group = parser.add_mutually_exclusive_group()
    vs_group.add_argument(
        "--write-vscode-settings",
        dest="write_vscode_settings",
        action="store_true",
        default=None,
        help="Write non-secret HiveMind defaults into workspace .vscode/settings.json.",
    )
    vs_group.add_argument(
        "--no-vscode-settings",
        "--skip-vscode-settings",
        dest="write_vscode_settings",
        action="store_false",
        help="Skip writing workspace .vscode/settings.json even when --workspace is given.",
    )
    parser.add_argument(
        "--quick-connect",
        action="store_true",
        help="Run streamlined <60s 1-command setup, live producer heartbeat check, and <1.5s Steam-yield preflight.",
    )
    parser.add_argument(
        "--check-producer-heartbeat",
        action="store_true",
        help="Inspect /v1/hivemind/producers/fleet, /v1/hivemind/raid/status, and /v1/hivemind/remote/config.",
    )
    parser.add_argument(
        "--send-opentrainer-heartbeat",
        action="store_true",
        help="Send outbound-only POST /v1/hivemind/opentrainer/heartbeat with honest GPU telemetry.",
    )
    parser.add_argument("--host-label", default="personal-pc", help="Personal PC host label for OpenTrainer heartbeat.")
    parser.add_argument(
        "--gamer-preflight",
        action="store_true",
        help="Run <1.5s Steam GPU yield and <=70.0 C quiet-fan thermal preflight check.",
    )
    parser.add_argument(
        "--max-temp-c",
        type=float,
        default=DEFAULT_MAX_TEMP_C,
        help="Maximum quiet-fan GPU temperature ceiling in Celsius (default: 70.0).",
    )
    parser.add_argument(
        "--min-free-vram-mib",
        type=int,
        default=DEFAULT_MIN_FREE_VRAM_MIB,
        help="Minimum free VRAM reserve in MiB (default: 2048).",
    )
    parser.add_argument(
        "--yield-budget-ms",
        type=float,
        default=DEFAULT_YIELD_BUDGET_MS,
        help="Maximum Steam worker yield budget before WDDM TDR margin (default: 1500.0).",
    )
    parser.add_argument(
        "--post-setup-telemetry",
        action="store_true",
        help="Submit timeout-safe non-blocking anonymous CES and zero_game_lag telemetry after setup.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=("dev", "gamer"),
        help="Optional onboarding telemetry profile override ('dev' or 'gamer').",
    )
    parser.add_argument("--skip-probe", action="store_true", help="Skip chaining into scripts/noel_post_setup_probe.py.")
    parser.add_argument("--ease-rating", type=int, default=None, help="Optional 1..5 developer CES ease rating.")
    parser.add_argument("--prompt-ces", action="store_true", help="Ask a single 1-line developer CES prompt.")
    parser.add_argument("--had-game-lag", action="store_true", help="Report any desktop/game lag in CES feedback.")
    parser.add_argument("--uninstall", action="store_true", help="Cleanly remove HiveMind-managed config artifacts.")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit FR/EN Markdown output with SHA-256 receipt.")
    return parser


def _emit_report(
    report: Mapping[str, Any],
    *,
    as_json: bool,
    as_markdown: bool,
    out_stream: Any = None,
) -> None:
    target_stream = out_stream if out_stream is not None else sys.stdout
    if as_json:
        print(_assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False)), file=target_stream)
    elif as_markdown:
        print(render_markdown_report(report), file=target_stream)
    else:
        print(render_cli_report(report), file=target_stream)


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    probe_fn: ProbeFn = default_probe_fn,
    input_fn: InputFn = input,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: ClockFn = time.monotonic,
    gpu_snapshot: Any = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    """CLI entrypoint returning `0` (ready/uninstalled), `1` (transient error), or `2` (refused/failed)."""
    args = build_parser().parse_args(argv)
    resolved_home = home if home is not None else Path.home()
    target_err = err_stream if err_stream is not None else sys.stderr
    try:
        exit_code, report = run_vscode_autoconfig(
            gateway=args.gateway,
            handle=args.handle or None,
            role=args.role,
            explicit_api_key=args.api_key or None,
            credentials_path=Path(args.credentials_file) if args.credentials_file else None,
            continue_path=Path(args.continue_config_path) if args.continue_config_path else None,
            workspace=Path(args.workspace) if args.workspace else None,
            vscode_settings_path=Path(args.vscode_settings_path) if args.vscode_settings_path else None,
            write_vscode_settings=args.write_vscode_settings,
            uninstall=bool(args.uninstall),
            skip_probe=bool(args.skip_probe),
            ease_rating=args.ease_rating,
            prompt_ces=bool(args.prompt_ces),
            had_game_lag=bool(args.had_game_lag),
            quick_connect=bool(args.quick_connect),
            check_producer_heartbeat=bool(args.check_producer_heartbeat),
            send_opentrainer_heartbeat=bool(args.send_opentrainer_heartbeat),
            host_label=args.host_label,
            gamer_preflight=bool(args.gamer_preflight),
            max_temp_c=args.max_temp_c,
            min_free_vram_mib=args.min_free_vram_mib,
            yield_budget_ms=args.yield_budget_ms,
            post_setup_telemetry=bool(args.post_setup_telemetry),
            profile=args.profile,
            gpu_snapshot=gpu_snapshot,
            system_name=system_name,
            lang=args.lang,
            os_name=os_name,
            environ=environ,
            home=resolved_home,
            cwd=cwd,
            http_fn=http_fn,
            probe_fn=probe_fn,
            input_fn=input_fn,
            gpu_probe_fn=gpu_probe_fn,
            process_probe_fn=process_probe_fn,
            worker_yield_fn=worker_yield_fn,
            clock_fn=clock_fn,
        )
    except AutoconfigError as exc:
        safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
        print(f"Error: {safe_err}", file=target_err)
        return 2

    _emit_report(
        report,
        as_json=bool(args.json),
        as_markdown=bool(args.markdown),
        out_stream=out_stream,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
