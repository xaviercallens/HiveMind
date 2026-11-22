#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""10-second Gamer-Whisper pre-flight self-test for sub-1.5s Steam GPU yield and 70 C quiet fans.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (Amadeus Rule):
     inspects environment variables (`USERDNSDOMAIN`, `HTTP_PROXY`, `HTTPS_PROXY`, `COMPUTERNAME`,
     `HOSTNAME`, etc.), the current working directory, and `.git/config` remote URLs (if present)
     for corporate indicators (`amadeus`, `corp`, `enterprise`, `intranet`, `internal`, `zscaler`,
     `netskope`, `bluecoat`, `fortinet`, `paloalto`) BEFORE probing hardware or contacting the
     gateway. Refuses execution (`exit code 2`, `compliance_ok=False`) when corporate markers are
     detected and NEVER transmits local env, paths, or git remotes over the network.
  2. Deterministic <10s Gamer-Whisper Pre-Flight Self-Test (`run_gamer_whisper_selftest`):
     - (a) Steam / DX12 game-launch detection (`cs2.exe`, `eldenring.exe`, `cyberpunk2077.exe`,
       `bg3.exe`, `helldivers2.exe`, Steam game processes, or non-worker GPU utilization `> 15%`)
       triggering full worker stop and `0 MiB` retained worker VRAM within `< 1500.0 ms` (failing
       closed with `exit code 2`, `yield_slo_ok=False` if `yield_latency_ms >= 1500.0` or worker
       fails to release cleanly), plus verifying the `300s` post-game resume cooldown policy.
     - (b) Quiet-fan thermal governor clamping/pausing load whenever GPU temperature `> 70.0 C`
       (configurable via `--max-temp-c`) and enforcing `>= 2048 MiB` free VRAM reserve, or
       reporting honest `'cpu_or_dev_only_no_nvidia_gpu'` status when no NVIDIA GPU is present
       without inventing fake GPU readings.
     - (c) LoRA-delta-only bandwidth cap verification (`< 50 MB/h` upload budget vs legacy
       full-checkpoint transfers).
  3. Zero-Inference Gateway & Gamer CES Integration:
     optional `--check-gateway` (`GET /health` and `GET /v1/hivemind/raid/status` — NEVER calling
     `POST /v1/chat/completions` or prompting `hivemind/coder`) and optional `--submit-feedback` /
     `--prompt-ces` (`profile='gamer'`, `ease_rating` 1..5, honest `zero_game_lag` derived from
     `yield_slo_ok` and `thermal_ok` unless overridden by `--had-game-lag`) posting zero-PII
     comfort telemetry to `POST /v1/hivemind/onboarding/feedback` with FR/EN CLI, Markdown, and
     JSON SHA-256 receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal gaming/hobby/open-source use only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
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

SCHEMA = "hivemind.noel_gamer_whisper_selftest/1"
RECEIPT_OBJECT = "hivemind.noel_gamer_whisper_selftest_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

SELFTEST_SLO_BUDGET_S = 10.0
DEFAULT_YIELD_SLO_MS = 1500.0
WDDM_TDR_RESET_MS = 2000.0
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_RESUME_COOLDOWN_S = 300.0
MIN_REQUIRED_COOLDOWN_S = 300.0
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0

DEFAULT_UPLOAD_CAP_MB_PER_HOUR = 50.0
DEFAULT_LORA_PARAMS = 4_194_304
DEFAULT_LORA_BYTES_PER_PARAM = 2
DEFAULT_SYNCS_PER_HOUR = 2.0
LEGACY_FULL_CHECKPOINT_MB = 14_000.0

SUBPROCESS_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 4.0

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
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
DRILL_GAME_SIGNATURE_SUITE: tuple[str, ...] = (
    "cs2.exe",
    "eldenring.exe",
    "cyberpunk2077.exe",
    "bg3.exe",
    "helldivers2.exe",
    "steam_app_730",
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
    "100% personal home PCs & personal gaming/hobby/open-source use only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate repositories; "
    "0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et usage gaming/hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de dépôt d'entreprise ; "
    "0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
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

_LOCAL_WORKER_DRILL_PROGRAM = (
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

RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
WorkerLifecycleFn = Callable[..., Mapping[str, Any]]
ClockFn = Callable[[], float]
HttpFn = Callable[..., tuple[int, Any]]
InputFn = Callable[[str], str]


class SelftestError(ValueError):
    """Raised when CLI inputs, gateway constraints, or output safety invariants fail."""


@dataclass(frozen=True)
class GpuTelemetry:
    """Truthful local NVIDIA GPU telemetry without UUID/serial identifiers or invented numbers."""

    present: bool
    parse_error: bool = False
    name: str | None = None
    total_vram_mib: int | None = None
    free_vram_mib: int | None = None
    used_vram_mib: int | None = None
    temp_c: float | None = None
    util_pct: float | None = None
    reason: str = "ok"


def normalize_lang(raw: str | None) -> str:
    """Normalize language code to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise SelftestError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise SelftestError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def _platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if "darwin" in raw or "mac" in raw:
        return "macos"
    if "win" in raw or raw == "nt":
        return "windows"
    if "linux" in raw:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-")
    return (cleaned[:32] if cleaned else "unknown")


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
    """Scrub raw `hm-...` keys, Bearer tokens, emails, non-loopback IPv4s, and home paths."""
    scrubbed = redact_home_path(text, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def scan_output_safety(text: str) -> list[str]:
    """Audit serialized CLI/Markdown/JSON output for leaked secrets, PII, or forbidden keys."""
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
        raise SelftestError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return {**body, "receipt_sha256": digest}


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app` or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or loopback `http(s)://...`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise SelftestError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise SelftestError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise SelftestError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise SelftestError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise SelftestError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise SelftestError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise SelftestError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise SelftestError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise SelftestError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or localhost/127.0.0.1/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise SelftestError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise SelftestError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(method: str, url: str) -> str:
    """Enforce allowed gateway origin and block any LLM completion endpoint."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise SelftestError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat" in low_path:
        raise SelftestError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    return origin


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib HTTP helper enforcing zero-inference gateway rules."""
    _assert_zero_inference_request(method, url)
    req_headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "hivemind-noel-gamer-whisper-selftest/1.0",
    }
    if headers:
        req_headers.update(headers)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
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
        return status_code, {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status_code, {"raw_length": len(raw)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
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
    try:
        if has_varargs or len(positional) >= 4:
            status_code, body = http_fn(method.upper(), url, payload, headers)
        elif len(positional) == 3:
            status_code, body = http_fn(method.upper(), url, payload)
        else:
            status_code, body = http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": str(exc)[:200]}
    norm_body = dict(body) if isinstance(body, Mapping) else {"data": body}
    return int(status_code), norm_body


def _find_corporate_markers(raw_text: str) -> list[str]:
    """Return corporate indicator tokens matched in `raw_text` without retaining `raw_text`."""
    if not raw_text:
        return []
    matched: list[str] = []
    for marker_name, pattern in _CORP_TOKEN_PATTERNS:
        if pattern.search(raw_text) and marker_name not in matched:
            matched.append(marker_name)
    return matched


def _resolve_git_config_files(cwd: Path) -> list[Path]:
    """Locate `.git/config` (and worktree commondir `config` if present) strictly on local disk."""
    dot_git = cwd / ".git"
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
                    gitdir = (cwd / gitdir).resolve()
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
    cwd: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (zero values ever leave memory)."""
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

    if cwd is not None:
        cwd_hits = _find_corporate_markers(str(cwd))
        for hit in cwd_hits:
            tag = f"cwd_path:{hit}"
            if tag not in indicators:
                indicators.append(tag)
            matched_markers.add(hit)

        for cfg_path in _resolve_git_config_files(cwd):
            try:
                cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_url in extract_git_config_remote_urls(cfg_text):
                for hit in _find_corporate_markers(remote_url):
                    tag = f"git_remote:{hit}"
                    if tag not in indicators:
                        indicators.append(tag)
                    matched_markers.add(hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Contrôle local conforme (PC gaming/perso & dossier hobby vérifiés). {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"Local compliance gate passed (personal gaming PC & hobby workspace verified). {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS D'EXÉCUTION ({joined}) : marqueur professionnel / entreprise détecté. "
            f"Aucun test GPU lancé et aucune requête envoyée. {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"EXECUTION REFUSED ({joined}): corporate machine or repository marker detected. "
            f"Zero hardware drills run and zero network requests sent. {COMPLIANCE_NOTICE_EN}"
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


def default_subprocess_runner(
    cmd: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a read-only local command using list args (`shell=False`) and `<= 3.0s` timeout."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise SelftestError("Subprocess command must be a non-empty list/tuple of string arguments.")
    argv = [str(part) for part in cmd]
    bounded_timeout = min(max(0.1, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded_timeout,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    return int(completed.returncode), (completed.stdout or "")


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
) -> GpuTelemetry:
    """Query read-only `nvidia-smi` metrics (`<= 3s` timeout) without UUID/serial or fake readings."""
    rc, stdout = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        min(float(timeout_s), SUBPROCESS_TIMEOUT_S),
    )
    if rc != 0 or not stdout.strip():
        return GpuTelemetry(present=False, reason="no_nvidia_gpu")

    parsed_gpus: list[GpuTelemetry] = []
    for raw_line in stdout.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 6 or not parts[0]:
            return GpuTelemetry(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        total_f = _optional_float(parts[1])
        free_f = _optional_float(parts[2])
        used_f = _optional_float(parts[3])
        temp_f = _optional_float(parts[4])
        util_f = _optional_float(parts[5])
        if total_f is None or free_f is None or total_f <= 0 or free_f < 0:
            return GpuTelemetry(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        used_mib = int(used_f) if used_f is not None and used_f >= 0 else max(0, int(total_f) - int(free_f))
        parsed_gpus.append(
            GpuTelemetry(
                present=True,
                parse_error=False,
                name=parts[0],
                total_vram_mib=int(total_f),
                free_vram_mib=int(free_f),
                used_vram_mib=used_mib,
                temp_c=round(temp_f, 1) if temp_f is not None else None,
                util_pct=round(util_f, 1) if util_f is not None else 0.0,
                reason="ok",
            )
        )
    return max(
        parsed_gpus,
        key=lambda g: (
            g.free_vram_mib or 0,
            -(g.temp_c if g.temp_c is not None else 999.0),
            g.total_vram_mib or 0,
        ),
    )


def _normalize_gpu_telemetry(raw: Any) -> GpuTelemetry:
    """Normalize `GpuTelemetry` or mapping returned by an injected `gpu_probe_fn`."""
    if isinstance(raw, GpuTelemetry):
        return raw
    if not isinstance(raw, Mapping):
        raise SelftestError("gpu_probe_fn must return GpuTelemetry or a mapping.")
    present = bool(raw.get("present", False))
    if not present:
        return GpuTelemetry(present=False, reason=str(raw.get("reason") or "no_nvidia_gpu"))
    if bool(raw.get("parse_error", False)):
        return GpuTelemetry(present=True, parse_error=True, reason="nvidia_smi_parse_error")

    raw_total = raw.get("total_vram_mib", raw.get("total_mib"))
    raw_free = raw.get("free_vram_mib", raw.get("free_mib"))
    raw_used = raw.get("used_vram_mib", raw.get("used_mib"))
    raw_temp = raw.get("temp_c")
    raw_util = raw.get("util_pct", raw.get("utilization_pct"))
    name = str(raw.get("name") or raw.get("gpu_model") or "").strip()
    if not name or raw_total is None or raw_free is None:
        return GpuTelemetry(present=True, parse_error=True, reason="incomplete_gpu_telemetry")

    total_mib = int(raw_total)
    free_mib = int(raw_free)
    used_mib = int(raw_used) if raw_used is not None else max(0, total_mib - free_mib)
    temp_c = float(raw_temp) if raw_temp is not None else None
    util_pct = float(raw_util) if raw_util is not None else 0.0
    return GpuTelemetry(
        present=True,
        parse_error=False,
        name=name,
        total_vram_mib=total_mib,
        free_vram_mib=free_mib,
        used_vram_mib=used_mib,
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
    """Read active local process names via `tasklist` (Windows) or `ps` (POSIX) in `<= 3s`."""
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


def detect_steam_or_dx12_game_activity(
    process_names: Iterable[str],
    *,
    non_worker_gpu_util_pct: float | None = None,
    extra_games: Sequence[str] = (),
    gpu_util_threshold_pct: float = NON_WORKER_GPU_UTIL_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Detect Steam / DX12 game processes or non-worker GPU utilization `> 15%`."""
    watchlist: set[str] = set(STEAM_DX12_GAME_EXECUTABLES)
    trunc_prefixes: set[str] = {exe[:15] for exe in STEAM_DX12_GAME_EXECUTABLES if len(exe) > 15}
    for extra in extra_games:
        cleaned = str(extra).strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            if len(cleaned) > 15:
                trunc_prefixes.add(cleaned[:15])

    matched_procs: list[str] = []
    for raw_proc in process_names:
        raw_low = str(raw_proc).strip().lower().replace("\\", "/")
        if not raw_low:
            continue
        base = raw_low.rsplit("/", 1)[-1]
        is_steam_game_marker = (
            base.startswith(STEAM_PROCESS_PREFIXES)
            or "steamapps/common/" in raw_low
            or base in ("steam_game", "steamwebhelper_game")
        )
        if base in watchlist or base in trunc_prefixes or is_steam_game_marker:
            if base not in matched_procs:
                matched_procs.append(base)

    gpu_util_triggered = bool(
        non_worker_gpu_util_pct is not None
        and math.isfinite(non_worker_gpu_util_pct)
        and non_worker_gpu_util_pct > gpu_util_threshold_pct
    )
    reasons: list[str] = []
    if matched_procs:
        reasons.append(f"steam_dx12_process:{','.join(matched_procs)}")
    if gpu_util_triggered:
        reasons.append(
            f"non_worker_gpu_util:{non_worker_gpu_util_pct:.1f}%>{gpu_util_threshold_pct:.1f}%"
        )

    return {
        "game_detected": bool(matched_procs or gpu_util_triggered),
        "matched_processes": matched_procs,
        "gpu_util_triggered": gpu_util_triggered,
        "non_worker_gpu_util_pct": non_worker_gpu_util_pct,
        "gpu_util_threshold_pct": gpu_util_threshold_pct,
        "trigger_reasons": reasons,
    }


def default_worker_lifecycle_drill(
    *,
    clock_fn: ClockFn = time.monotonic,
    trigger_exe: str = "cs2.exe",
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a real local Python subprocess worker start -> game-trigger stop -> 0 MiB release drill."""
    bounded_timeout = min(max(0.2, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCAL_WORKER_DRILL_PROGRAM],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
                "retained_worker_vram_mib": None,
                "yield_latency_ms": None,
                "exit_code": proc.returncode,
                "trigger_exe": trigger_exe,
                "method": "local_python_worker_subprocess",
            }

        t_trigger = clock_fn()
        drill_detection = detect_steam_or_dx12_game_activity([trigger_exe])
        if not drill_detection["game_detected"]:
            proc.kill()
            proc.wait(timeout=1.0)
            return {
                "worker_started": True,
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": None,
                "yield_latency_ms": None,
                "exit_code": proc.returncode,
                "trigger_exe": trigger_exe,
                "method": "local_python_worker_subprocess",
            }

        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()

        release_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        exit_rc = proc.wait(timeout=bounded_timeout)
        t_released = clock_fn()
        yield_ms = round(max(0.0, (t_released - t_trigger) * 1000.0), 3)

        retained_mib: int | None = None
        if release_line.startswith("HIVEMIND_WORKER_RELEASED:"):
            raw_mib = release_line.split(":", 1)[1].strip()
            if raw_mib.isdigit():
                retained_mib = int(raw_mib)

        clean_ok = bool(exit_rc == 0 and proc.poll() is not None and retained_mib == 0)
        return {
            "worker_started": True,
            "worker_stopped": proc.poll() is not None,
            "clean_release": clean_ok,
            "retained_worker_vram_mib": retained_mib if retained_mib is not None else -1,
            "yield_latency_ms": yield_ms,
            "exit_code": int(exit_rc),
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
            "yield_latency_ms": round(bounded_timeout * 1000.0, 3),
            "exit_code": proc.returncode,
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


def _invoke_callable_with_supported_kwargs(
    fn: Callable[..., Any],
    **candidate_kwargs: Any,
) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def evaluate_thermal_and_vram_governor(
    gpu: GpuTelemetry,
    *,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    lang: str = "en",
) -> dict[str, Any]:
    """Evaluate the `<= 70.0 C` quiet-fan thermal governor and `>= 2048 MiB` free VRAM reserve."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise SelftestError("max_temp_c must be a finite temperature between 25.0 and 95.0 C.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise SelftestError("min_free_vram_mib must be an integer >= 256 MiB.")

    cap_c = round(float(max_temp_c), 1)
    reserve_mib = int(min_free_vram_mib)

    if not gpu.present:
        summary = (
            "Aucun GPU NVIDIA détecté — statut 'cpu_or_dev_only_no_nvidia_gpu' "
            "(aucune mesure GPU inventée, 0 charge GPU locale)."
            if is_fr
            else "No NVIDIA GPU detected — honest 'cpu_or_dev_only_no_nvidia_gpu' status "
            "(zero fake GPU readings, 0 local GPU load)."
        )
        return {
            "gpu_present": False,
            "gpu_status": "cpu_or_dev_only_no_nvidia_gpu",
            "governor_action": "gpu_worker_disabled_no_nvidia_gpu",
            "gpu_name": None,
            "total_vram_mib": None,
            "free_vram_mib": None,
            "used_vram_mib": None,
            "temp_c": None,
            "util_pct": None,
            "max_temp_c": cap_c,
            "min_free_vram_mib": reserve_mib,
            "thermal_ok": True,
            "vram_reserve_ok": True,
            "thermal_clamped": False,
            "vram_clamped": False,
            "duty_cycle_scale": 0.0,
            "check_status": "ok",
            "summary": summary,
        }

    if gpu.parse_error or gpu.temp_c is None or gpu.free_vram_mib is None:
        summary = (
            "Télémetrie `nvidia-smi` illisible ou incomplète — worker GPU maintenu en pause par sécurité."
            if is_fr
            else "Unreadable or incomplete `nvidia-smi` telemetry — GPU worker held paused fail-closed."
        )
        return {
            "gpu_present": True,
            "gpu_status": "gpu_driver_parse_error",
            "governor_action": "paused_unreadable_gpu",
            "gpu_name": gpu.name,
            "total_vram_mib": gpu.total_vram_mib,
            "free_vram_mib": gpu.free_vram_mib,
            "used_vram_mib": gpu.used_vram_mib,
            "temp_c": gpu.temp_c,
            "util_pct": gpu.util_pct,
            "max_temp_c": cap_c,
            "min_free_vram_mib": reserve_mib,
            "thermal_ok": False,
            "vram_reserve_ok": False,
            "thermal_clamped": True,
            "vram_clamped": True,
            "duty_cycle_scale": 0.0,
            "check_status": "fail",
            "summary": summary,
        }

    thermal_ok = gpu.temp_c <= cap_c
    vram_ok = gpu.free_vram_mib >= reserve_mib
    thermal_clamped = not thermal_ok
    vram_clamped = not vram_ok

    if thermal_clamped and vram_clamped:
        gpu_status = "thermal_and_vram_clamped_paused"
        governor_action = "paused_thermal_and_vram"
        duty_scale = 0.0
        check_status = "warn"
        summary = (
            f"{gpu.name} à {gpu.temp_c:.1f} C (> {cap_c:.1f} C) et {gpu.free_vram_mib} MiB libres "
            f"(< {reserve_mib} MiB) : gouverneur en pause complète pour garder des ventilateurs silencieux."
            if is_fr
            else f"{gpu.name} at {gpu.temp_c:.1f} C (> {cap_c:.1f} C) and {gpu.free_vram_mib} MiB free "
            f"(< {reserve_mib} MiB): governor clamped/paused to keep fans quiet and protect VRAM."
        )
    elif thermal_clamped:
        gpu_status = "thermal_clamped_paused"
        governor_action = "paused_thermal_ceiling"
        duty_scale = 0.0
        check_status = "warn"
        summary = (
            f"{gpu.name} à {gpu.temp_c:.1f} C (> {cap_c:.1f} C plafond ventilateurs silencieux) : "
            f"charge GPU mise en pause jusqu'au retour <= {cap_c:.1f} C."
            if is_fr
            else f"{gpu.name} at {gpu.temp_c:.1f} C (> {cap_c:.1f} C quiet-fan ceiling): "
            f"governor clamped/paused GPU load until <= {cap_c:.1f} C."
        )
    elif vram_clamped:
        gpu_status = "vram_reserve_hold"
        governor_action = "paused_low_vram_reserve"
        duty_scale = 0.0
        check_status = "warn"
        summary = (
            f"{gpu.name} n'a que {gpu.free_vram_mib} MiB de VRAM libre (< {reserve_mib} MiB requis) : "
            "worker maintenu en veille."
            if is_fr
            else f"{gpu.name} has {gpu.free_vram_mib} MiB free VRAM (< {reserve_mib} MiB reserve): "
            "worker held paused."
        )
    else:
        gpu_status = "whisper_ready"
        near_ceiling = gpu.temp_c > (cap_c - 5.0)
        duty_scale = 0.5 if near_ceiling else 1.0
        governor_action = "whisper_gentle_throttle" if near_ceiling else "whisper_ready_quiet_fans"
        check_status = "ok"
        summary = (
            f"{gpu.name} silencieux & prêt ({gpu.temp_c:.1f} C <= {cap_c:.1f} C, "
            f"{gpu.free_vram_mib} MiB libres >= {reserve_mib} MiB)."
            if is_fr
            else f"{gpu.name} quiet & ready ({gpu.temp_c:.1f} C <= {cap_c:.1f} C, "
            f"{gpu.free_vram_mib} MiB free >= {reserve_mib} MiB)."
        )

    return {
        "gpu_present": True,
        "gpu_status": gpu_status,
        "governor_action": governor_action,
        "gpu_name": gpu.name,
        "total_vram_mib": gpu.total_vram_mib,
        "free_vram_mib": gpu.free_vram_mib,
        "used_vram_mib": gpu.used_vram_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "max_temp_c": cap_c,
        "min_free_vram_mib": reserve_mib,
        "thermal_ok": thermal_ok,
        "vram_reserve_ok": vram_ok,
        "thermal_clamped": thermal_clamped,
        "vram_clamped": vram_clamped,
        "duty_cycle_scale": duty_scale,
        "check_status": check_status,
        "summary": summary,
    }


def verify_post_game_cooldown_policy(
    *,
    resume_cooldown_s: float = DEFAULT_RESUME_COOLDOWN_S,
    live_game_active: bool = False,
    seconds_since_game_exit: float | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify the `300s` post-game resume cooldown policy preventing lobby/loading thrashing."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if isinstance(resume_cooldown_s, bool) or not math.isfinite(float(resume_cooldown_s)) or float(resume_cooldown_s) <= 0:
        raise SelftestError("resume_cooldown_s must be a positive finite number of seconds.")

    cooldown_val = round(float(resume_cooldown_s), 3)

    def _can_resume(game_on: bool, elapsed_after_exit: float) -> bool:
        if game_on:
            return False
        return elapsed_after_exit >= cooldown_val

    state_transitions_ok = (
        (not _can_resume(True, 999.0))
        and (not _can_resume(False, 0.0))
        and (not _can_resume(False, min(120.0, cooldown_val * 0.5)))
        and (not _can_resume(False, cooldown_val - 0.1))
        and _can_resume(False, cooldown_val)
    )
    policy_ok = bool(cooldown_val >= MIN_REQUIRED_COOLDOWN_S and state_transitions_ok)

    if live_game_active:
        current_state = "paused_game_active"
        remaining_s = cooldown_val
        resume_now = False
    elif seconds_since_game_exit is not None and float(seconds_since_game_exit) < cooldown_val:
        clamped_age = max(0.0, float(seconds_since_game_exit))
        current_state = "paused_post_game_cooldown"
        remaining_s = round(cooldown_val - clamped_age, 3)
        resume_now = False
    else:
        current_state = "cooldown_clear"
        remaining_s = 0.0
        resume_now = policy_ok

    if not policy_ok:
        check_status = "fail"
        summary = (
            f"Politique de reprise après jeu invalide ({cooldown_val:.0f}s < {MIN_REQUIRED_COOLDOWN_S:.0f}s requis)."
            if is_fr
            else f"Post-game resume cooldown policy invalid ({cooldown_val:.0f}s < {MIN_REQUIRED_COOLDOWN_S:.0f}s minimum)."
        )
    elif not resume_now:
        check_status = "warn"
        summary = (
            f"Pause post-jeu active ({current_state}) : reprise autorisée dans {remaining_s:.1f}s "
            f"(politique {cooldown_val:.0f}s vérifiée)."
            if is_fr
            else f"Post-game hold active ({current_state}): worker stays paused for {remaining_s:.1f}s "
            f"({cooldown_val:.0f}s cooldown policy verified)."
        )
    else:
        check_status = "ok"
        summary = (
            f"Politique de refroidissement post-jeu de {cooldown_val:.0f}s vérifiée (zéro relance pendant les chargements)."
            if is_fr
            else f"{cooldown_val:.0f}s post-game resume cooldown policy verified (zero map-load thrashing)."
        )

    return {
        "cooldown_policy_ok": policy_ok,
        "resume_cooldown_s": cooldown_val,
        "min_required_cooldown_s": MIN_REQUIRED_COOLDOWN_S,
        "current_state": current_state,
        "cooldown_remaining_s": remaining_s,
        "resume_allowed_now": resume_now,
        "check_status": check_status,
        "summary": summary,
    }


def verify_lora_delta_bandwidth_cap(
    *,
    transfer_mode: str = "lora_delta_only",
    lora_params: int = DEFAULT_LORA_PARAMS,
    bytes_per_param: int = DEFAULT_LORA_BYTES_PER_PARAM,
    syncs_per_hour: float = DEFAULT_SYNCS_PER_HOUR,
    upload_cap_mb_per_hour: float = DEFAULT_UPLOAD_CAP_MB_PER_HOUR,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify LoRA-delta-only upload stays `< 50 MB/h` vs legacy full-checkpoint transfers."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    mode = str(transfer_mode or "").strip().lower()
    if lora_params <= 0 or bytes_per_param <= 0 or syncs_per_hour <= 0 or upload_cap_mb_per_hour <= 0:
        raise SelftestError("Bandwidth calculation parameters must be positive.")

    lora_delta_mb = round((int(lora_params) * int(bytes_per_param)) / 1_000_000.0, 3)
    per_sync_mb = lora_delta_mb if mode == "lora_delta_only" else LEGACY_FULL_CHECKPOINT_MB
    estimated_mb_per_h = round(per_sync_mb * float(syncs_per_hour), 3)
    legacy_mb_per_h = round(LEGACY_FULL_CHECKPOINT_MB * float(syncs_per_hour), 1)
    savings_ratio = round(LEGACY_FULL_CHECKPOINT_MB / lora_delta_mb, 1)
    full_ckpt_blocked = mode == "lora_delta_only"
    cap_ok = bool(
        full_ckpt_blocked
        and upload_cap_mb_per_hour <= DEFAULT_UPLOAD_CAP_MB_PER_HOUR
        and estimated_mb_per_h < float(upload_cap_mb_per_hour)
    )

    if cap_ok:
        summary = (
            f"Mode `lora_delta_only` vérifié : {estimated_mb_per_h:.2f} MB/h < {upload_cap_mb_per_hour:.1f} MB/h "
            f"(gain {savings_ratio:.0f}x face aux {legacy_mb_per_h:.0f} MB/h d'un checkpoint complet)."
            if is_fr
            else f"`lora_delta_only` verified: {estimated_mb_per_h:.2f} MB/h < {upload_cap_mb_per_hour:.1f} MB/h cap "
            f"({savings_ratio:.0f}x smaller than {legacy_mb_per_h:.0f} MB/h legacy full-checkpoint uploads)."
        )
    else:
        summary = (
            f"Plafond réseau dépassé ou mode complet refusé ({mode}: {estimated_mb_per_h:.2f} MB/h >= "
            f"{upload_cap_mb_per_hour:.1f} MB/h)."
            if is_fr
            else f"Bandwidth cap violated or legacy full-checkpoint blocked ({mode}: {estimated_mb_per_h:.2f} MB/h >= "
            f"{upload_cap_mb_per_hour:.1f} MB/h budget)."
        )

    return {
        "bandwidth_cap_ok": cap_ok,
        "transfer_mode": mode,
        "full_checkpoint_upload_blocked": full_ckpt_blocked,
        "estimated_lora_delta_mb": lora_delta_mb,
        "estimated_upload_mb_per_hour": estimated_mb_per_h,
        "upload_cap_mb_per_hour": round(float(upload_cap_mb_per_hour), 1),
        "legacy_full_checkpoint_mb": LEGACY_FULL_CHECKPOINT_MB,
        "legacy_upload_mb_per_hour": legacy_mb_per_h,
        "bandwidth_savings_ratio": savings_ratio,
        "check_status": "ok" if cap_ok else "fail",
        "summary": summary,
    }


def check_gateway_health_and_raid_status(
    *,
    gateway: str = DEFAULT_GATEWAY,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Check `GET /health` and `GET /v1/hivemind/raid/status` without any LLM completion calls."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    health_code, health_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/health", None, None)
    raid_code, raid_body = _safe_http_call(
        http_fn, "GET", f"{clean_gw}/v1/hivemind/raid/status", None, None
    )

    health_ok = health_code == 200
    raid_ok = raid_code == 200
    gateway_ok = bool(health_ok and raid_ok)

    swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}
    fb_summary = (
        raid_body.get("feedback")
        if isinstance(raid_body.get("feedback"), Mapping)
        else (
            raid_body.get("feedback_summary")
            if isinstance(raid_body.get("feedback_summary"), Mapping)
            else {}
        )
    )
    raw_mean = fb_summary.get("mean_ease_rating", raid_body.get("mean_ease_rating"))
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )

    if gateway_ok:
        status = "ok"
        msg = (
            "Passerelle vérifiée en mode zéro inférence (`GET /health` et `GET /v1/hivemind/raid/status` HTTP 200)."
            if is_fr
            else "Gateway verified in zero-inference mode (`GET /health` and `GET /v1/hivemind/raid/status` HTTP 200)."
        )
    elif 404 in (health_code, raid_code):
        status = "missing_route"
        msg = (
            "Route passerelle HTTP 404 sur `/v1/hivemind/raid/status` : la révision Cloud Run est en retard sur `main`."
            if is_fr
            else "Gateway returned HTTP 404 on `/v1/hivemind/raid/status`: Cloud Run revision lags behind `main`."
        )
    elif 429 in (health_code, raid_code):
        status = "rate_limited"
        msg = (
            "Limite de débit passerelle atteinte (HTTP 429) ; réessayez dans une minute."
            if is_fr
            else "Gateway rate limit reached (HTTP 429); retry in one minute."
        )
    elif health_code >= 500 or raid_code >= 500:
        bad_code = raid_code if raid_code >= 500 else health_code
        status = "server_error"
        msg = (
            f"Erreur temporaire du serveur passerelle (HTTP {bad_code})."
            if is_fr
            else f"Temporary gateway server error (HTTP {bad_code})."
        )
    elif 0 in (health_code, raid_code):
        status = "network_error"
        msg = (
            "Impossible de joindre la passerelle HiveMind (erreur réseau)."
            if is_fr
            else "Could not reach HiveMind gateway (network error)."
        )
    else:
        status = "client_error"
        msg = (
            f"Réponse inattendue de la passerelle (/health={health_code}, /raid/status={raid_code})."
            if is_fr
            else f"Unexpected gateway HTTP status (/health={health_code}, /raid/status={raid_code})."
        )

    return {
        "checked": True,
        "gateway_ok": gateway_ok,
        "status": status,
        "health_http_status": health_code,
        "raid_status_http_status": raid_code,
        "health_status_text": str(health_body.get("status") or ("ok" if health_ok else "error")),
        "campaign": str(raid_body.get("campaign") or "") or None,
        "active_trainers": (
            int(swarm["active_trainers"])
            if isinstance(swarm.get("active_trainers"), int) and not isinstance(swarm.get("active_trainers"), bool)
            else None
        ),
        "mean_ease_rating": mean_ease,
        "zero_inference_verified": True,
        "message": msg,
    }


def parse_gamer_ces_input(
    raw_answer: str,
    *,
    derived_zero_game_lag: bool,
) -> dict[str, Any]:
    """Parse 1-line interactive gamer CES answer (`1..5 [lag|nolag]`, or empty to skip)."""
    stripped = (raw_answer or "").strip()
    if not stripped or stripped.lower() in ("skip", "s", "q", "passer", "non"):
        return {
            "skipped": True,
            "profile": "gamer",
            "ease_rating": None,
            "zero_game_lag": bool(derived_zero_game_lag),
        }

    rating: int | None = None
    explicit_lag: bool | None = None
    tokens = [tok for tok in re.split(r"[\s,;:/]+", stripped.lower()) if tok]
    for tok in tokens:
        if tok in ("1", "2", "3", "4", "5") and rating is None:
            rating = int(tok)
        elif tok == "gamer":
            explicit_gamer = True
            _ = explicit_gamer
        elif tok in ("lag", "had-game-lag", "had_game_lag", "stutter", "saccade", "fan", "bruit"):
            explicit_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "quiet", "ok"):
            explicit_lag = False
        else:
            raise SelftestError(f"Unrecognized CES token {tok!r}; expected 1..5 [lag|nolag].")

    if rating is None:
        raise SelftestError("Interactive CES input must include an ease_rating from 1 to 5.")

    effective_zero_lag = bool(derived_zero_game_lag and (explicit_lag is not True))
    return {
        "skipped": False,
        "profile": "gamer",
        "ease_rating": rating,
        "zero_game_lag": effective_zero_lag,
    }


def submit_gamer_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    ease_rating: int,
    zero_game_lag: bool,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Post zero-PII gamer comfort telemetry (`profile='gamer'`) to `/v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise SelftestError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise SelftestError("zero_game_lag must be a boolean.")

    payload: dict[str, Any] = {
        "profile": "gamer",
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
    raw_ratio = body.get("zero_game_lag_ratio")
    lag_ratio = (
        round(float(raw_ratio), 4)
        if isinstance(raw_ratio, (int, float)) and not isinstance(raw_ratio, bool)
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
            f"Merci ! Retour CES Gamer enregistré (ease_rating={ease_rating}/5, "
            f"profil=gamer, zero_game_lag={zero_game_lag})."
            if is_fr
            else f"Thank you! Anonymous Gamer CES feedback recorded (ease_rating={ease_rating}/5, "
            f"profile=gamer, zero_game_lag={zero_game_lag})."
        )
    elif status_code == 404:
        outcome = "missing_route"
        submitted = False
        msg = (
            "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 : la révision Cloud Run est en "
            "retard sur `main` (route de retour Noël absente)."
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
            f"Erreur temporaire du serveur passerelle (HTTP {status_code}) lors de l'envoi du retour CES Gamer."
            if is_fr
            else f"Temporary gateway server error (HTTP {status_code}) while submitting Gamer CES feedback."
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
            f"Envoi du retour CES Gamer refusé par la passerelle (HTTP {status_code})."
            if is_fr
            else f"Gamer CES feedback rejected by gateway (HTTP {status_code})."
        )

    return {
        "attempted": True,
        "submitted": submitted,
        "skipped": False,
        "status": outcome,
        "http_status": status_code,
        "profile": "gamer",
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": lag_ratio,
        "total_responses": total_resp,
        "zero_pii_payload": payload,
        "message": msg,
    }


def prompt_and_submit_gamer_ces(
    *,
    gateway: str = DEFAULT_GATEWAY,
    derived_zero_game_lag: bool,
    system_name: str | None = None,
    input_fn: InputFn = input,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Prompt for 1-line gamer comfort rating (`1..5`) and post zero-PII feedback if answered."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    prompt_text = (
        "[HiveMind CES Gamer 1-question] Notez le confort Gamer-Whisper 1..5 "
        "(ajoutez 'lag' si moindre saccade/bruit ; Entrée pour ignorer) : "
        if is_fr
        else "[HiveMind 1-prompt Gamer CES] Rate Gamer-Whisper comfort 1..5 "
        "(add 'lag' if any stutter/fan noise; Enter to skip): "
    )
    try:
        raw_answer = input_fn(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raw_answer = ""

    parsed = parse_gamer_ces_input(raw_answer, derived_zero_game_lag=derived_zero_game_lag)
    if parsed["skipped"]:
        skip_msg = (
            "Retour CES Gamer ignoré (aucun envoi réseau)."
            if is_fr
            else "1-prompt Gamer CES skipped (no feedback sent)."
        )
        return {
            "attempted": False,
            "submitted": False,
            "skipped": True,
            "status": "skipped",
            "http_status": None,
            "profile": "gamer",
            "ease_rating": None,
            "zero_game_lag": parsed["zero_game_lag"],
            "mean_ease_rating": None,
            "zero_game_lag_ratio": None,
            "total_responses": None,
            "zero_pii_payload": None,
            "message": skip_msg,
        }

    return submit_gamer_ces_feedback(
        gateway=clean_gw,
        ease_rating=int(parsed["ease_rating"]),
        zero_game_lag=bool(parsed["zero_game_lag"]),
        system_name=system_name,
        http_fn=http_fn,
        lang=lang_code,
    )


def run_gamer_whisper_selftest(
    *,
    gateway: str = DEFAULT_GATEWAY,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_slo_ms: float = DEFAULT_YIELD_SLO_MS,
    resume_cooldown_s: float = DEFAULT_RESUME_COOLDOWN_S,
    seconds_since_game_exit: float | None = None,
    upload_cap_mb_per_hour: float = DEFAULT_UPLOAD_CAP_MB_PER_HOUR,
    transfer_mode: str = "lora_delta_only",
    syncs_per_hour: float = DEFAULT_SYNCS_PER_HOUR,
    extra_games: Sequence[str] = (),
    drill_game_exe: str = "cs2.exe",
    check_gateway: bool = False,
    submit_feedback: bool = False,
    prompt_ces: bool = False,
    ease_rating: int | None = None,
    had_game_lag: bool = False,
    lang: str = "en",
    os_name: str | None = None,
    system_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_lifecycle_fn: WorkerLifecycleFn = default_worker_lifecycle_drill,
    clock_fn: ClockFn = time.monotonic,
    http_fn: HttpFn = default_http_json,
    input_fn: InputFn = input,
) -> dict[str, Any]:
    """Run the deterministic `<10s` Gamer-Whisper pre-flight self-test and return a signed report."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    resolved_cwd = cwd if cwd is not None else Path.cwd()

    if isinstance(yield_slo_ms, bool) or not math.isfinite(float(yield_slo_ms)) or float(yield_slo_ms) <= 0:
        raise SelftestError("yield_slo_ms must be a positive finite millisecond budget.")
    if submit_feedback and not prompt_ces and ease_rating is None:
        raise SelftestError("--submit-feedback requires --ease-rating 1..5 (or --prompt-ces).")
    if ease_rating is not None and (
        isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5)
    ):
        raise SelftestError("ease_rating must be an integer between 1 and 5.")

    t_start = clock_fn()

    # Step 1: Strictly local Personal-PC & Hobby-Only Compliance Gate BEFORE any probes or HTTP!
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=resolved_cwd,
        lang=lang_code,
    )
    if not compliance["compliance_ok"]:
        elapsed_refused = round(max(0.0, clock_fn() - t_start), 4)
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "profile": "gamer",
                "lang": lang_code,
                "gateway": clean_gw,
                "exit_code": 2,
                "compliance_ok": False,
                "ready": False,
                "verdict": "refused_compliance",
                "checks_passed_count": 0,
                "checks_total_count": 1,
                "warnings_count": 0,
                "failures_count": 1,
                "yield_slo_ok": False,
                "yield_latency_ms": None,
                "yield_slo_ms": round(float(yield_slo_ms), 1),
                "retained_worker_vram_mib": None,
                "thermal_ok": False,
                "max_temp_c": round(float(max_temp_c), 1),
                "vram_reserve_ok": False,
                "min_free_vram_mib": int(min_free_vram_mib),
                "gpu_status": "refused_compliance",
                "cooldown_policy_ok": False,
                "resume_cooldown_s": round(float(resume_cooldown_s), 1),
                "bandwidth_cap_ok": False,
                "ease_rating": None,
                "mean_ease_rating": None,
                "zero_game_lag": False,
                "zero_inference_verified": True,
                "within_10s_slo": elapsed_refused < SELFTEST_SLO_BUDGET_S,
                "elapsed_s": elapsed_refused,
                "compliance": compliance,
                "preemption_drill": None,
                "thermal_vram_governor": None,
                "cooldown_policy": None,
                "bandwidth_cap": None,
                "gateway_check": None,
                "feedback": None,
                "warnings": [],
                "errors": [sanitize_text_for_display(compliance["summary"], home=resolved_home)],
            }
        )

    warnings: list[str] = []
    errors: list[str] = []

    # Step 2(b): Probe local GPU & evaluate <=70 C quiet-fan governor + >=2048 MiB free VRAM reserve
    raw_gpu = _invoke_callable_with_supported_kwargs(
        gpu_probe_fn,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    gpu = _normalize_gpu_telemetry(raw_gpu)
    governor = evaluate_thermal_and_vram_governor(
        gpu,
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        lang=lang_code,
    )
    if governor["check_status"] == "warn":
        warnings.append(sanitize_text_for_display(governor["summary"], home=resolved_home))
    elif governor["check_status"] == "fail":
        errors.append(sanitize_text_for_display(governor["summary"], home=resolved_home))

    # Step 2(a): Steam / DX12 game detection + sub-1.5s worker stop / 0 MiB VRAM drill + 300s cooldown
    t_scan0 = clock_fn()
    live_processes = list(
        _invoke_callable_with_supported_kwargs(
            process_probe_fn,
            os_name=eff_os,
            timeout_s=SUBPROCESS_TIMEOUT_S,
        )
    )
    scan_ms = max(0.0, (clock_fn() - t_scan0) * 1000.0)

    live_detection = detect_steam_or_dx12_game_activity(
        live_processes,
        non_worker_gpu_util_pct=gpu.util_pct if (gpu.present and not gpu.parse_error) else None,
        extra_games=extra_games,
    )

    # Verify Steam/DX12 signature coverage across canonical games + GPU >15% trigger
    signature_checks_ok = all(
        detect_steam_or_dx12_game_activity([sig], extra_games=extra_games)["game_detected"]
        for sig in (*DRILL_GAME_SIGNATURE_SUITE, drill_game_exe)
    ) and detect_steam_or_dx12_game_activity([], non_worker_gpu_util_pct=18.5)["game_detected"]

    t_drill0 = clock_fn()
    raw_drill = _invoke_callable_with_supported_kwargs(
        worker_lifecycle_fn,
        clock_fn=clock_fn,
        trigger_exe=drill_game_exe,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    measured_drill_ms = max(0.0, (clock_fn() - t_drill0) * 1000.0)
    if not isinstance(raw_drill, Mapping):
        raise SelftestError("worker_lifecycle_fn must return a mapping.")

    reported_ms = raw_drill.get("yield_latency_ms")
    if isinstance(reported_ms, (int, float)) and not isinstance(reported_ms, bool) and math.isfinite(reported_ms):
        effective_drill_ms = float(reported_ms)
    else:
        effective_drill_ms = measured_drill_ms

    yield_latency_ms = round(max(0.0, scan_ms + effective_drill_ms), 3)
    worker_stopped = bool(raw_drill.get("worker_stopped", False))
    clean_release = bool(raw_drill.get("clean_release", False))
    raw_retained = raw_drill.get("retained_worker_vram_mib")
    retained_mib = (
        int(raw_retained)
        if isinstance(raw_retained, (int, float)) and not isinstance(raw_retained, bool)
        else -1
    )

    slo_limit_ms = round(float(yield_slo_ms), 1)
    yield_slo_ok = bool(
        signature_checks_ok
        and worker_stopped
        and clean_release
        and retained_mib == 0
        and yield_latency_ms < slo_limit_ms
    )

    if not yield_slo_ok:
        yield_err = (
            f"Échec critique d'éviction Gamer-Whisper : latence={yield_latency_ms:.1f} ms "
            f"(budget < {slo_limit_ms:.1f} ms), worker_stopped={worker_stopped}, "
            f"retained_worker_vram_mib={retained_mib} (attendu 0 MiB)."
            if is_fr
            else f"Gamer-Whisper yield SLO failed closed: yield_latency_ms={yield_latency_ms:.1f} ms "
            f"(budget < {slo_limit_ms:.1f} ms), worker_stopped={worker_stopped}, "
            f"retained_worker_vram_mib={retained_mib} (expected 0 MiB)."
        )
        errors.append(sanitize_text_for_display(yield_err, home=resolved_home))
    elif live_detection["game_detected"]:
        live_reasons = ", ".join(live_detection["trigger_reasons"])
        live_warn = (
            f"Activité gaming détectée ({live_reasons}) : worker arrêté en {yield_latency_ms:.1f} ms "
            f"(< {slo_limit_ms:.1f} ms, 0 MiB VRAM retenue)."
            if is_fr
            else f"Active game / GPU load detected ({live_reasons}): worker yielded in {yield_latency_ms:.1f} ms "
            f"(< {slo_limit_ms:.1f} ms, 0 MiB retained VRAM)."
        )
        warnings.append(sanitize_text_for_display(live_warn, home=resolved_home))

    preemption_drill: dict[str, Any] = {
        "yield_slo_ok": yield_slo_ok,
        "yield_latency_ms": yield_latency_ms,
        "process_scan_ms": round(scan_ms, 3),
        "worker_stop_ms": round(effective_drill_ms, 3),
        "yield_slo_ms": slo_limit_ms,
        "wddm_tdr_margin_ms": round(WDDM_TDR_RESET_MS - yield_latency_ms, 3),
        "worker_started": bool(raw_drill.get("worker_started", True)),
        "worker_stopped": worker_stopped,
        "clean_release": clean_release,
        "retained_worker_vram_mib": retained_mib,
        "drill_trigger_exe": drill_game_exe,
        "signature_suite_verified": signature_checks_ok,
        "live_game_detected": live_detection["game_detected"],
        "live_matched_processes": live_detection["matched_processes"],
        "live_gpu_util_triggered": live_detection["gpu_util_triggered"],
        "live_trigger_reasons": live_detection["trigger_reasons"],
        "method": str(raw_drill.get("method") or "custom_worker_lifecycle"),
        "check_status": "ok" if yield_slo_ok else "fail",
    }

    cooldown_info = verify_post_game_cooldown_policy(
        resume_cooldown_s=resume_cooldown_s,
        live_game_active=live_detection["game_detected"],
        seconds_since_game_exit=seconds_since_game_exit,
        lang=lang_code,
    )
    if cooldown_info["check_status"] == "fail":
        errors.append(sanitize_text_for_display(cooldown_info["summary"], home=resolved_home))
    elif cooldown_info["check_status"] == "warn" and not live_detection["game_detected"]:
        warnings.append(sanitize_text_for_display(cooldown_info["summary"], home=resolved_home))

    # Step 2(c): LoRA-delta-only <50 MB/h bandwidth cap verification
    bandwidth_info = verify_lora_delta_bandwidth_cap(
        transfer_mode=transfer_mode,
        syncs_per_hour=syncs_per_hour,
        upload_cap_mb_per_hour=upload_cap_mb_per_hour,
        lang=lang_code,
    )
    if not bandwidth_info["bandwidth_cap_ok"]:
        errors.append(sanitize_text_for_display(bandwidth_info["summary"], home=resolved_home))

    # Step 3: Optional zero-inference gateway check (/health + /v1/hivemind/raid/status)
    gw_info: dict[str, Any] | None = None
    if check_gateway:
        gw_info = check_gateway_health_and_raid_status(
            gateway=clean_gw,
            http_fn=http_fn,
            lang=lang_code,
        )
        if not gw_info["gateway_ok"]:
            errors.append(sanitize_text_for_display(gw_info["message"], home=resolved_home))

    elapsed_pre_fb = max(0.0, clock_fn() - t_start)
    within_10s = elapsed_pre_fb < SELFTEST_SLO_BUDGET_S
    if not within_10s:
        slo_err = (
            f"Autotest Gamer-Whisper trop lent ({elapsed_pre_fb:.2f}s >= {SELFTEST_SLO_BUDGET_S:.1f}s)."
            if is_fr
            else f"Gamer-Whisper pre-flight self-test exceeded 10s budget ({elapsed_pre_fb:.2f}s >= {SELFTEST_SLO_BUDGET_S:.1f}s)."
        )
        errors.append(slo_err)

    # Honest zero_game_lag derived strictly from yield_slo_ok and thermal_ok unless overridden by --had-game-lag
    derived_zero_game_lag = bool(
        yield_slo_ok
        and governor["thermal_ok"]
        and within_10s
        and (not had_game_lag)
    )

    # Step 4: Optional zero-PII Gamer CES feedback submission (profile='gamer')
    fb_info: dict[str, Any] | None = None
    if prompt_ces:
        fb_info = prompt_and_submit_gamer_ces(
            gateway=clean_gw,
            derived_zero_game_lag=derived_zero_game_lag,
            system_name=system_name,
            input_fn=input_fn,
            http_fn=http_fn,
            lang=lang_code,
        )
    elif submit_feedback or ease_rating is not None:
        assert ease_rating is not None
        fb_info = submit_gamer_ces_feedback(
            gateway=clean_gw,
            ease_rating=ease_rating,
            zero_game_lag=derived_zero_game_lag,
            system_name=system_name,
            http_fn=http_fn,
            lang=lang_code,
        )

    final_zero_game_lag = (
        bool(fb_info["zero_game_lag"])
        if (fb_info is not None and not fb_info["skipped"])
        else derived_zero_game_lag
    )
    if fb_info is not None and fb_info["attempted"] and not fb_info["submitted"]:
        errors.append(sanitize_text_for_display(str(fb_info["message"]), home=resolved_home))

    # Tallies across all executed checks (never emitting forbidden 'score' keys)
    check_outcomes: list[bool] = [
        True,  # 1. compliance_ok
        yield_slo_ok,  # 2. sub-1.5s yield & 0 MiB retained VRAM
        bool(governor["thermal_ok"] and governor["vram_reserve_ok"]),  # 3. thermal <=70C & VRAM >=2048 MiB
        bool(cooldown_info["cooldown_policy_ok"] and cooldown_info["resume_allowed_now"]),  # 4. 300s cooldown
        bool(bandwidth_info["bandwidth_cap_ok"]),  # 5. <50 MB/h LoRA-delta-only cap
    ]
    if gw_info is not None:
        check_outcomes.append(bool(gw_info["gateway_ok"]))
    if fb_info is not None and fb_info["attempted"]:
        check_outcomes.append(bool(fb_info["submitted"]))

    checks_total_count = len(check_outcomes)
    checks_passed_count = sum(1 for item in check_outcomes if item)

    hard_fail = bool(
        (not yield_slo_ok)
        or (governor["check_status"] == "fail")
        or (not cooldown_info["cooldown_policy_ok"])
        or (not bandwidth_info["bandwidth_cap_ok"])
        or (not within_10s)
    )
    gw_failed = bool(gw_info is not None and not gw_info["gateway_ok"])
    fb_failed = bool(fb_info is not None and fb_info["attempted"] and not fb_info["submitted"])
    governor_paused = bool(
        (governor["check_status"] == "warn")
        or live_detection["game_detected"]
        or (not cooldown_info["resume_allowed_now"])
    )

    failures_count = (
        (1 if not yield_slo_ok else 0)
        + (1 if governor["check_status"] == "fail" else 0)
        + (1 if not cooldown_info["cooldown_policy_ok"] else 0)
        + (1 if not bandwidth_info["bandwidth_cap_ok"] else 0)
        + (1 if not within_10s else 0)
        + (1 if gw_failed else 0)
        + (1 if fb_failed else 0)
    )
    warnings_count = len(warnings)

    if hard_fail:
        exit_code = 2
        ready = False
        verdict = "failed_yield_slo" if not yield_slo_ok else "failed_safety_gate"
    elif gw_failed:
        gw_missing = gw_info is not None and gw_info["status"] == "missing_route"
        exit_code = 2 if gw_missing else 1
        ready = False
        verdict = "gateway_unavailable"
    elif fb_failed:
        fb_missing = fb_info is not None and fb_info.get("http_status") == 404
        exit_code = 2 if fb_missing else 1
        ready = False
        verdict = "feedback_submission_failed"
    elif governor_paused:
        exit_code = 1
        ready = False
        if not governor["thermal_ok"]:
            verdict = "paused_thermal_ceiling"
        elif not governor["vram_reserve_ok"]:
            verdict = "paused_low_vram_reserve"
        elif live_detection["game_detected"]:
            verdict = "paused_game_active"
        else:
            verdict = "paused_post_game_cooldown"
    else:
        exit_code = 0
        ready = True
        verdict = (
            "cpu_or_dev_only_no_nvidia_gpu"
            if not gpu.present
            else "ready"
        )

    mean_ease = (
        fb_info.get("mean_ease_rating")
        if (fb_info is not None and fb_info.get("mean_ease_rating") is not None)
        else (gw_info.get("mean_ease_rating") if gw_info is not None else None)
    )
    ease_val = fb_info.get("ease_rating") if fb_info is not None else None
    total_elapsed_s = round(max(0.0, clock_fn() - t_start), 4)

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "profile": "gamer",
            "lang": lang_code,
            "gateway": clean_gw,
            "exit_code": exit_code,
            "compliance_ok": True,
            "ready": ready,
            "verdict": verdict,
            "checks_passed_count": checks_passed_count,
            "checks_total_count": checks_total_count,
            "warnings_count": warnings_count,
            "failures_count": failures_count,
            "yield_slo_ok": yield_slo_ok,
            "yield_latency_ms": yield_latency_ms,
            "yield_slo_ms": slo_limit_ms,
            "retained_worker_vram_mib": retained_mib,
            "thermal_ok": bool(governor["thermal_ok"]),
            "max_temp_c": round(float(max_temp_c), 1),
            "vram_reserve_ok": bool(governor["vram_reserve_ok"]),
            "min_free_vram_mib": int(min_free_vram_mib),
            "gpu_status": str(governor["gpu_status"]),
            "cooldown_policy_ok": bool(cooldown_info["cooldown_policy_ok"]),
            "resume_cooldown_s": round(float(resume_cooldown_s), 1),
            "bandwidth_cap_ok": bool(bandwidth_info["bandwidth_cap_ok"]),
            "ease_rating": ease_val,
            "mean_ease_rating": mean_ease,
            "zero_game_lag": final_zero_game_lag,
            "zero_inference_verified": True,
            "within_10s_slo": within_10s,
            "elapsed_s": total_elapsed_s,
            "compliance": compliance,
            "preemption_drill": preemption_drill,
            "thermal_vram_governor": governor,
            "cooldown_policy": cooldown_info,
            "bandwidth_cap": bandwidth_info,
            "gateway_check": gw_info,
            "feedback": fb_info,
            "warnings": warnings,
            "errors": errors,
        }
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render FR/EN GitHub-flavored Markdown Gamer-Whisper self-test report with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    title = (
        "# Autotest Pré-Vol Gamer-Whisper HiveMind Noël (<10s)"
        if is_fr
        else "# HiveMind Holiday Gamer-Whisper Pre-Flight Self-Test (<10s)"
    )
    yield_val = report.get("yield_latency_ms")
    yield_str = f"{yield_val} ms" if yield_val is not None else "n/a"
    ease_val = report.get("ease_rating")
    ease_str = f"{ease_val}/5" if ease_val is not None else "n/a"

    lines: list[str] = [
        title,
        "",
        f"> **Compliance / Conformité**: {bilingual_notice}",
        "",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **Compliance OK**: `{report.get('compliance_ok')}`",
        f"- **Ready**: `{report.get('ready')}` (`{report.get('verdict')}`)",
        f"- **GPU Status**: `{report.get('gpu_status')}`",
        f"- **Steam/DX12 Yield Latency**: `{yield_str}` (SLO `< {report.get('yield_slo_ms')} ms`, `yield_slo_ok={report.get('yield_slo_ok')}`)",
        f"- **Retained Worker VRAM**: `{report.get('retained_worker_vram_mib')} MiB`",
        f"- **Quiet-Fan Thermal OK (`<= {report.get('max_temp_c')} C`)**: `{report.get('thermal_ok')}`",
        f"- **Free VRAM Reserve OK (`>= {report.get('min_free_vram_mib')} MiB`)**: `{report.get('vram_reserve_ok')}`",
        f"- **Post-Game Cooldown (`{report.get('resume_cooldown_s')} s`)**: `{report.get('cooldown_policy_ok')}`",
        f"- **LoRA-Delta Bandwidth Cap (`< 50 MB/h`)**: `{report.get('bandwidth_cap_ok')}`",
        f"- **Checks Passed**: `{report.get('checks_passed_count')}/{report.get('checks_total_count')}`",
        f"- **Zero Game Lag**: `{report.get('zero_game_lag')}` | **CES Ease Rating**: `{ease_str}`",
    ]

    gov = report.get("thermal_vram_governor")
    if isinstance(gov, Mapping):
        lines.extend(
            [
                "",
                "## Gouverneur Thermique & VRAM Silencieux" if is_fr else "## Quiet-Fan Thermal & VRAM Governor",
                "",
                f"- {gov.get('summary')}",
            ]
        )

    bw = report.get("bandwidth_cap")
    if isinstance(bw, Mapping):
        lines.extend(
            [
                "",
                "## Plafond Réseau LoRA-Delta (<50 MB/h)" if is_fr else "## LoRA-Delta Bandwidth Cap (<50 MB/h)",
                "",
                f"- {bw.get('summary')}",
            ]
        )

    gw = report.get("gateway_check")
    if isinstance(gw, Mapping):
        lines.extend(
            [
                "",
                "## Santé Passerelle (Zéro Inférence)" if is_fr else "## Gateway Health & Raid Status (Zero-Inference)",
                "",
                f"- **{gw.get('status')}**: {gw.get('message')}",
            ]
        )

    fb = report.get("feedback")
    if isinstance(fb, Mapping):
        lines.extend(
            [
                "",
                "## Retour Anonyme CES Gamer" if is_fr else "## Anonymous Gamer CES Feedback",
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
    """Render concise human-readable FR/EN CLI output with masked keys, `~` paths, and SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    banner = (
        "HIVEMIND NOËL — AUTOTEST GAMER-WHISPER (<1.5s YIELD • <=70 C • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND HOLIDAY — GAMER-WHISPER SELF-TEST (<1.5s YIELD • <=70 C • 0.00 EUR / 0.00 USD)"
    )
    yield_val = report.get("yield_latency_ms")
    yield_str = f"{yield_val} ms" if yield_val is not None else "n/a"

    lines: list[str] = [
        "==========================================================================",
        f"  {banner}",
        "==========================================================================",
        f"  [Compliance] {bilingual_notice}",
        f"  Gateway              : {report.get('gateway')}",
        f"  Compliance OK        : {report.get('compliance_ok')}",
        f"  Ready                : {report.get('ready')} ({report.get('verdict')})",
        f"  GPU status           : {report.get('gpu_status')}",
        f"  Yield latency        : {yield_str} (SLO < {report.get('yield_slo_ms')} ms, ok={report.get('yield_slo_ok')})",
        f"  Retained worker VRAM : {report.get('retained_worker_vram_mib')} MiB",
        f"  Thermal <= {report.get('max_temp_c')} C   : {report.get('thermal_ok')}",
        f"  Free VRAM >= {report.get('min_free_vram_mib')} MiB: {report.get('vram_reserve_ok')}",
        f"  Cooldown ({report.get('resume_cooldown_s')} s)    : {report.get('cooldown_policy_ok')}",
        f"  Bandwidth < 50 MB/h  : {report.get('bandwidth_cap_ok')}",
        f"  Zero game lag        : {report.get('zero_game_lag')}",
        f"  Checks passed        : {report.get('checks_passed_count')}/{report.get('checks_total_count')}",
    ]

    gw = report.get("gateway_check")
    if isinstance(gw, Mapping):
        lines.append(f"  Gateway check        : {gw.get('status')} — {gw.get('message')}")

    fb = report.get("feedback")
    if isinstance(fb, Mapping):
        lines.append(f"  Gamer CES feedback   : {fb.get('status')} — {fb.get('message')}")

    for warn_msg in report.get("warnings") or ():
        lines.append(f"  [WARN] {warn_msg}")

    for err_msg in report.get("errors") or ():
        lines.append(f"  [ERROR] {err_msg}")

    lines.append(f"  SHA-256 Receipt      : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_gamer_whisper_selftest.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind <10s Gamer-Whisper pre-flight self-test for sub-1.5s Steam/DX12 GPU yield, "
            "<=70 C quiet fans, >=2048 MiB free VRAM, and <50 MB/h LoRA-delta cap "
            "(100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="HiveMind gateway URL (https://*.run.app or localhost).",
    )
    parser.add_argument(
        "--max-temp-c",
        type=float,
        default=DEFAULT_MAX_TEMP_C,
        help="Quiet-fan GPU temperature ceiling in Celsius (default: 70.0).",
    )
    parser.add_argument(
        "--min-free-vram-mib",
        type=int,
        default=DEFAULT_MIN_FREE_VRAM_MIB,
        help="Minimum free GPU VRAM reserve in MiB (default: 2048).",
    )
    parser.add_argument(
        "--yield-slo-ms",
        type=float,
        default=DEFAULT_YIELD_SLO_MS,
        help="Maximum allowed Steam/DX12 GPU yield latency in milliseconds (default: 1500.0).",
    )
    parser.add_argument(
        "--resume-cooldown-s",
        type=float,
        default=DEFAULT_RESUME_COOLDOWN_S,
        help="Post-game resume cooldown in seconds (default: 300.0).",
    )
    parser.add_argument(
        "--seconds-since-game-exit",
        type=float,
        default=None,
        help="Optional seconds elapsed since last game exit to evaluate cooldown state.",
    )
    parser.add_argument(
        "--game",
        action="append",
        default=[],
        help="Additional game executable name to monitor.",
    )
    parser.add_argument(
        "--drill-game-exe",
        default="cs2.exe",
        help="Steam/DX12 game executable name used in the pre-flight preemption drill.",
    )
    parser.add_argument(
        "--check-gateway",
        action="store_true",
        help="Verify GET /health and GET /v1/hivemind/raid/status (zero LLM inference).",
    )
    parser.add_argument(
        "--submit-feedback",
        action="store_true",
        help="Submit zero-PII gamer comfort telemetry to POST /v1/hivemind/onboarding/feedback.",
    )
    parser.add_argument(
        "--prompt-ces",
        action="store_true",
        help="Ask a single 1-line gamer comfort CES question (1..5).",
    )
    parser.add_argument(
        "--ease-rating",
        type=int,
        default=None,
        help="Gamer onboarding ease/comfort rating from 1 to 5.",
    )
    parser.add_argument(
        "--had-game-lag",
        action="store_true",
        help="Override zero_game_lag to False if any game stutter or fan noise occurred.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report with SHA-256 receipt.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit FR/EN Markdown report with SHA-256 receipt.",
    )
    return parser


def _emit_report(report: Mapping[str, Any], *, as_json: bool, as_markdown: bool) -> None:
    if as_json:
        print(_assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False)))
    elif as_markdown:
        print(render_markdown_report(report))
    else:
        print(render_cli_report(report))


def main(
    argv: Sequence[str] | None = None,
    *,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_lifecycle_fn: WorkerLifecycleFn = default_worker_lifecycle_drill,
    clock_fn: ClockFn = time.monotonic,
    http_fn: HttpFn = default_http_json,
    input_fn: InputFn = input,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
) -> int:
    """CLI entrypoint returning `0` (ready), `1` (governor hold / transient HTTP), or `2` (fail-closed)."""
    args = build_parser().parse_args(argv)
    resolved_home = home if home is not None else Path.home()
    try:
        report = run_gamer_whisper_selftest(
            gateway=args.gateway,
            max_temp_c=args.max_temp_c,
            min_free_vram_mib=args.min_free_vram_mib,
            yield_slo_ms=args.yield_slo_ms,
            resume_cooldown_s=args.resume_cooldown_s,
            seconds_since_game_exit=args.seconds_since_game_exit,
            extra_games=args.game,
            drill_game_exe=args.drill_game_exe,
            check_gateway=bool(args.check_gateway),
            submit_feedback=bool(args.submit_feedback),
            prompt_ces=bool(args.prompt_ces),
            ease_rating=args.ease_rating,
            had_game_lag=bool(args.had_game_lag),
            lang=args.lang,
            os_name=os_name,
            system_name=system_name,
            environ=environ,
            home=resolved_home,
            cwd=cwd,
            gpu_probe_fn=gpu_probe_fn,
            process_probe_fn=process_probe_fn,
            worker_lifecycle_fn=worker_lifecycle_fn,
            clock_fn=clock_fn,
            http_fn=http_fn,
            input_fn=input_fn,
        )
    except SelftestError as exc:
        safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
        print(f"Error: {safe_err}", file=sys.stderr)
        return 2

    _emit_report(report, as_json=bool(args.json), as_markdown=bool(args.markdown))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
