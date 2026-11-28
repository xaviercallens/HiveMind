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
       triggering full worker stop and `0 MiB` retained worker VRAM within `< 1500.0 milliseconds` (failing
       closed with `exit code 2`, `yield_slo_ok=False` if `yield_latency_ms >= 1500.0` or worker
       fails to release cleanly), plus verifying the `300s` post-game resume cooldown policy.
     - (b) Quiet-fan thermal governor clamping/pausing load whenever GPU temperature `> 70.0 C`
       (configurable via `--max-temp-c`) and enforcing `>= 2048 MiB` free VRAM reserve, or
       reporting honest `'cpu_or_dev_only_no_nvidia_gpu'` status when no NVIDIA GPU is present
       without inventing fake GPU readings.
     - (c) LoRA-delta-only bandwidth cap verification (`< 50 MB/h` upload budget vs legacy
       full-checkpoint transfers).
  3. Zero-Inference Gateway, PR #949 Live Quorum/Parity Gate & Gamer CES Integration:
     optional `--check-gateway` (`GET /health` and `GET /v1/hivemind/raid/status`), `--pr949-quorum-gate`
     (4-HTML parity, `/v1/models` `hivemind/coder`, live 8+ producer quorum, `/v1/hivemind/arena/summary`
     Wilson 95% LB & vote conservation, `Idempotency-Key` replay, and `deploy.sh --dry-run` — NEVER calling
     chat-completion endpoints or prompting `hivemind/coder`) and optional `--submit-feedback` /
     `--prompt-ces` (`profile='gamer'`, `ease_rating` 1..5, honest `zero_game_lag` derived from
     `yield_slo_ok` and `thermal_ok` unless overridden by `--had-game-lag`) with bounded POSIX-0600 retry
     spooling to `POST /v1/hivemind/onboarding/feedback` with FR/EN CLI, Markdown, and JSON SHA-256 receipts.

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
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_gamer_whisper_selftest/1"
RECEIPT_OBJECT = "hivemind.noel_gamer_whisper_selftest_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding.feedback.summary"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
_ACCEPTED_FEEDBACK_SUMMARY_OBJECTS: frozenset[str] = frozenset(
    {FEEDBACK_SUMMARY_OBJECT, "hivemind.onboarding_feedback_summary"}
)
_ACCEPTED_FLEET_OBJECTS: frozenset[str] = frozenset({FLEET_OBJECT, "hivemind.producers.fleet"})
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})
DEFAULT_HOST_LABEL = "personal-pc"
DEFAULT_TRAINING_STACK = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_ACTIVE_CAMPAIGN = CAMPAIGN_ID
DEFAULT_NETWORK_MODE = "outbound-https-relay"
DEFAULT_CREDENTIALS_REL = Path(".hivemind") / "credentials.json"
DEFAULT_CES_QUEUE_FILENAME = "gamer_whisper_ces_queue.json"
DEFAULT_CES_QUEUE_REL = Path(".hivemind") / DEFAULT_CES_QUEUE_FILENAME
CES_QUEUE_SCHEMA = "hivemind.gamer_whisper_ces_queue/1"
DEFAULT_CES_QUEUE_MAX_ENTRIES = 128
DEFAULT_CES_QUEUE_MAX_DELIVERED = 512
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
MAX_GPU_VRAM_MIB = 1_048_576
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024

EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
PR949_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
PR949_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
PR949_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/raid/status", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "holiday"),
)
VALID_FLEET_BADGES: frozenset[str] = frozenset(
    {"ONLINE", "BOOTSTRAPPING", "STALE_CLOUD_RUN", "OFFLINE"}
)
_ALLOWED_OPENTRAINER_HB_KEYS: frozenset[str] = frozenset(
    {
        "host_label",
        "gpu_name",
        "vram_total_mib",
        "vram_free_mib",
        "network_mode",
        "training_stack",
        "active_campaign",
    }
)
_NON_BLOCKING_FEEDBACK_CODES: frozenset[int] = frozenset({0, 404, 408, 429})
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]+$")
_CLOUD_RUN_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)

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

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_NONCE_ATTR_RE = re.compile(
    r"""\bnonce=(?:"([A-Za-z0-9_+=/=-]{8,128})"|'([A-Za-z0-9_+=/=-]{8,128})')"""
)
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)"
    r"(?:[A-Za-z0-9_-]{1,4}to" + r"k/s|[0-9]{1,4}m" + r"s\b|[A-Za-z0-9_-]{4,})(?![A-Za-z0-9_-])"
)
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + r"ment\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|to" + r"k/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_VALID_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{5,127}$")
_PYTEST_TMP_SEGMENT_RE = re.compile(r"pytest-of-[^/\\]+")
_ALLOWED_FEEDBACK_POST_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
)
_SPOOL_LOCKS_GUARD = threading.Lock()
_SPOOL_LOCKS: dict[str, threading.Lock] = {}

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
RawHttpFn = Callable[..., tuple[int, Any, dict[str, str]]]
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
    scrubbed = str(text)
    effective_home = home if home is not None else Path.home()
    home_str = str(effective_home)
    if home_str and home_str not in ("/", "\\", ""):
        for variant in (
            home_str,
            home_str.replace("\\", "/"),
            home_str.replace("/", "\\"),
        ):
            if variant:
                scrubbed = re.sub(
                    re.escape(variant) + r"(?=[/\\\s'\",;:]|$)",
                    "~",
                    scrubbed,
                )
    scrubbed = redact_home_path(scrubbed, home=home)
    scrubbed = _HOME_PATH_RE.sub("~", scrubbed)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def _sanitize_structure(value: Any, *, home: Path | None = None) -> Any:
    if isinstance(value, str):
        return sanitize_text_for_display(value, home=home)
    if isinstance(value, Mapping):
        return {str(k): _sanitize_structure(v, home=home) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_structure(item, home=home) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_structure(item, home=home) for item in value]
    return value


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce` attributes, CSP `'nonce-...'` tokens, and masked `hm-***...` key suffixes."""
    if not isinstance(text, str):
        raise SelftestError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def scan_output_safety(text: str, *, home: Path | None = None) -> list[str]:
    """Audit serialized CLI/Markdown/JSON output for leaked secrets, PII, or forbidden keys."""
    scrubbed = strip_masked_keys_and_nonces(text)
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(scrubbed):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    has_home = bool(_HOME_PATH_RE.search(text))
    if not has_home and home is not None:
        home_str = str(home)
        if home_str and home_str not in ("/", "\\", "") and (
            home_str in text or home_str.replace("\\", "/") in text
        ):
            has_home = True
    if has_home:
        issues.append("unredacted_home_path_detected")
    for ip_hit in _IPV4_RE.findall(text):
        try:
            ip_obj = ipaddress.ip_address(ip_hit)
        except ValueError:
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
            continue
        if not ip_obj.is_loopback:
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str, *, home: Path | None = None) -> str:
    issues = scan_output_safety(text, home=home)
    if issues:
        raise SelftestError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return {**body, "receipt_sha256": digest}


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is a strict DNS `*.run.app` or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    if not isinstance(host, str):
        return False
    raw = host.strip().lower()
    if (
        not raw
        or len(raw) > 253
        or any(ch.isspace() or ch in "\\/@" or ord(ch) < 32 or ord(ch) == 127 for ch in raw)
    ):
        return False
    h = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw
    if not h or "[" in h or "]" in h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    labels = h.split(".")
    if len(labels) < 3 or any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def _safe_urlsplit(url: str) -> Any:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise SelftestError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise SelftestError("Gateway URL must not contain control characters.")
    cleaned = url.strip()
    if any(ch.isspace() for ch in cleaned):
        raise SelftestError("Gateway URL must not contain whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise SelftestError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise SelftestError("Gateway URL must not embed credentials.")
    try:
        parts = urlsplit(cleaned)
        _ = parts.port
    except ValueError as exc:
        raise SelftestError("Gateway URL contains an invalid port.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise SelftestError("Gateway URL must not embed credentials.")
    return parts


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or loopback `http(s)://...`."""
    parts = _safe_urlsplit(url)
    cleaned = url.strip().rstrip("/")
    if parts.query or parts.fragment:
        raise SelftestError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise SelftestError("Gateway URL must be a bare origin without a path suffix.")
    port = parts.port
    if port is not None and not (1 <= port <= 65535):
        raise SelftestError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not parts.netloc or not is_allowed_gateway_host(host):
        raise SelftestError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or localhost/127.0.0.1/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise SelftestError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise SelftestError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _payload_has_forbidden_inference_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            if str(key).strip().lower() in ("messages", "prompt") or _payload_has_forbidden_inference_key(val):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_payload_has_forbidden_inference_key(item) for item in obj)
    return False


def _assert_zero_inference_request(method: str, url: str, payload: Any = None) -> str:
    """Enforce allowed gateway origin and block any LLM completion endpoint or inference payload."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise SelftestError(f"Unsupported HTTP method {verb!r}.")
    parts = _safe_urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    coder_id = "hivemind/" + "coder"
    if "completions" in low_path or "/chat" in low_path or coder_id in low_path:
        raise SelftestError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None:
        if verb != "POST":
            raise SelftestError("Request payload is only permitted on POST requests.")
        if isinstance(payload, Mapping):
            serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, default=str).lower()
            if _payload_has_forbidden_inference_key(payload) or coder_id in serialized:
                raise SelftestError(
                    "Zero-inference invariant violated: messages/prompt/hivemind-coder payloads are strictly forbidden."
                )
        elif isinstance(payload, (str, list, tuple)):
            serialized = json.dumps(payload, ensure_ascii=False, default=str).lower()
            if _payload_has_forbidden_inference_key(payload) or coder_id in serialized:
                raise SelftestError(
                    "Zero-inference invariant violated: messages/prompt/hivemind-coder payloads are strictly forbidden."
                )
    return origin


def _validate_http_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise SelftestError("timeout_s must be a positive finite number.")
    val = float(timeout_s)
    if not math.isfinite(val) or val <= 0.0:
        raise SelftestError("timeout_s must be a positive finite number.")
    return min(val, 60.0)


def _sanitize_http_headers(
    headers: Mapping[str, str] | None,
    *,
    default_accept: str = "application/json, text/html;q=0.9, text/plain;q=0.8",
) -> dict[str, str]:
    req_headers: dict[str, str] = {
        "Accept": default_accept,
        "User-Agent": "hivemind-noel-gamer-whisper-selftest/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            raw_k = str(hk)
            raw_v = str(hv)
            if not raw_k.strip() or any(
                ch in ("\r", "\n", "\x00") or ord(ch) < 32 or ord(ch) == 127
                for ch in (raw_k + raw_v)
            ):
                raise SelftestError("HTTP headers must be non-empty and must not contain control characters.")
            req_headers[raw_k.strip()] = raw_v.strip()
    return req_headers


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect guard blocking untrusted hosts, downgrades, backslashes, credentials, and inference."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raw_loc: Any = newurl
        if headers is not None and hasattr(headers, "get"):
            for hdr_name in ("location", "Location", "uri"):
                cand = headers.get(hdr_name)
                if cand is not None:
                    raw_loc = cand
                    break
        if (
            not isinstance(raw_loc, str)
            or not raw_loc.strip()
            or not isinstance(newurl, str)
            or not newurl.strip()
        ):
            raise SelftestError("Untrusted redirect: empty target URL.")
        if "\\" in raw_loc or "\\" in unquote(raw_loc) or "@" in raw_loc or "@" in unquote(raw_loc):
            raise SelftestError("Untrusted redirect: forbidden backslash or '@' in Location.")
        orig_parts = _safe_urlsplit(req.full_url)
        dest_parts = _safe_urlsplit(newurl)
        orig_host = (orig_parts.hostname or "").lower()
        dest_host = (dest_parts.hostname or "").lower()
        if orig_host not in ALLOWED_LOOPBACK_HOSTS and dest_host in ALLOWED_LOOPBACK_HOSTS:
            raise SelftestError("Untrusted redirect from remote host to loopback.")
        if orig_parts.scheme == "https" and dest_parts.scheme != "https":
            raise SelftestError("Untrusted HTTPS-to-HTTP downgrade redirect.")
        effective_method = "GET" if (req.get_method() == "POST" and code in (301, 302, 303)) else req.get_method()
        _assert_zero_inference_request(effective_method, newurl)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_zero_inference_request(new_req.get_method(), new_req.full_url)
        return new_req


def _read_bounded_http_body(resp: Any, *, max_bytes: int = MAX_HTTP_BODY_BYTES) -> bytes:
    raw_bytes = resp.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise SelftestError(f"HTTP response exceeded {max_bytes} byte limit.")
    return bytes(raw_bytes)


def default_http_raw(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Direct, proxy-free stdlib HTTP helper returning `(status_code, body_text, lower_headers)`."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, payload=payload)
    bounded_timeout = _validate_http_timeout_s(timeout_s)
    req_headers = _sanitize_http_headers(headers)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_text = _read_bounded_http_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            raw_hdrs = getattr(resp, "headers", None)
            lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
            return status_code, raw_text, lower_headers
    except urllib.error.HTTPError as exc:
        raw_err = (
            exc.read(512 * 1024).decode("utf-8", errors="replace") if getattr(exc, "fp", None) is not None else ""
        )
        raw_hdrs = getattr(exc, "headers", None)
        lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if raw_hdrs else {}
        return int(exc.code), sanitize_text_for_display(raw_err), lower_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        err_doc = json.dumps(
            {
                "network_error": type(exc).__name__,
                "detail": sanitize_text_for_display(str(reason))[:200],
            }
        )
        return 0, err_doc, {"x-hivemind-network-error": type(exc).__name__}


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib JSON HTTP helper enforcing zero-inference gateway rules."""
    status_code, raw_text, _ = default_http_raw(
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    if status_code == 0 or status_code >= 400:
        if not raw_text.strip():
            return status_code, {}
        try:
            parsed_err = json.loads(raw_text)
        except json.JSONDecodeError:
            return status_code, {"error": sanitize_text_for_display(raw_text[:240])}
        return status_code, (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})

    if not raw_text.strip():
        return status_code, {}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return status_code, {"raw_length": len(raw_text)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL and zero-inference guard before invoking `http_fn`, catching transport errors."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, payload=payload)
    bounded_timeout = _validate_http_timeout_s(timeout_s)
    sig = inspect.signature(http_fn)
    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    try:
        if http_fn is default_http_json:
            return default_http_json(verb, url, payload=payload, headers=headers, timeout_s=bounded_timeout)
        if has_kwargs or ("payload" in params and "headers" in params and not has_varargs and len(positional) <= 2):
            kw: dict[str, Any] = {"payload": payload, "headers": headers}
            if has_kwargs or "timeout_s" in params:
                kw["timeout_s"] = bounded_timeout
            res = http_fn(verb, url, **kw)
        elif has_varargs or len(positional) >= 4:
            res = http_fn(verb, url, payload, headers)
        elif len(positional) == 3:
            res = http_fn(verb, url, payload)
        else:
            res = http_fn(verb, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": sanitize_text_for_display(str(exc))[:200]}
    status_code, body = res[0], res[1]
    if isinstance(body, Mapping):
        norm_body = dict(body)
    elif isinstance(body, str):
        try:
            decoded = json.loads(body) if body.strip() else {}
            norm_body = dict(decoded) if isinstance(decoded, Mapping) else {"data": decoded}
        except json.JSONDecodeError:
            norm_body = {"raw_length": len(body)}
    else:
        norm_body = {"data": body}
    return int(status_code), norm_body


def _invoke_parity_call(
    http_callable: Callable[..., Any],
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any] | None, str, dict[str, str]]:
    """Invoke a raw/json HTTP callable safely and return `(status_code, json_or_none, body_text, lower_headers)`."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, payload=payload)
    bounded_timeout = _validate_http_timeout_s(timeout_s)
    hdr_dict = {str(k): str(v) for k, v in headers.items()} if headers else None
    sig = inspect.signature(http_callable)
    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    try:
        if http_callable is default_http_raw:
            raw_res: Any = default_http_raw(verb, url, payload=payload, headers=hdr_dict, timeout_s=bounded_timeout)
        elif http_callable is default_http_json:
            raw_res = default_http_json(verb, url, payload=payload, headers=hdr_dict, timeout_s=bounded_timeout)
        elif has_kwargs or (("payload" in params or "headers" in params or "timeout_s" in params) and not has_varargs and len(positional) <= 2):
            kw: dict[str, Any] = {}
            if has_kwargs or "payload" in params:
                kw["payload"] = payload
            if has_kwargs or "headers" in params:
                kw["headers"] = hdr_dict
            if has_kwargs or "timeout_s" in params:
                kw["timeout_s"] = bounded_timeout
            raw_res = http_callable(verb, url, **kw)
        elif has_varargs or len(positional) >= 5:
            raw_res = http_callable(verb, url, payload, hdr_dict, bounded_timeout)
        elif len(positional) == 4:
            raw_res = http_callable(verb, url, payload, hdr_dict)
        elif len(positional) == 3:
            raw_res = http_callable(verb, url, payload)
        elif len(positional) == 2 and positional[0].name.lower() in ("url", "_u", "u", "target_url"):
            raw_res = http_callable(url, bounded_timeout)
        else:
            raw_res = http_callable(verb, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, None, "", {"x-hivemind-network-error": type(exc).__name__}

    code = int(raw_res[0])
    body = raw_res[1]
    raw_hdrs = raw_res[2] if len(raw_res) >= 3 and isinstance(raw_res[2], Mapping) else {}
    lower_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()}
    if isinstance(body, Mapping):
        body_dict = dict(body)
        return code, body_dict, json.dumps(body_dict, ensure_ascii=False), lower_headers
    if isinstance(body, str):
        parsed_json: dict[str, Any] | None = None
        if body.strip():
            with suppress(json.JSONDecodeError):
                loaded = json.loads(body)
                if isinstance(loaded, dict):
                    parsed_json = loaded
        return code, parsed_json, body, lower_headers
    return code, None, str(body or ""), lower_headers


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
    """Locate `.git/config` walking `(cwd, *cwd.parents)` and resolving worktree `gitdir:`/`commondir`."""
    candidates: list[Path] = []
    seen_bases: set[Path] = set()
    roots: list[Path] = []
    for base in (cwd, *cwd.parents):
        roots.append(base)
    with suppress(OSError):
        resolved_cwd = cwd.resolve()
        if resolved_cwd != cwd:
            for base in (resolved_cwd, *resolved_cwd.parents):
                if base not in roots:
                    roots.append(base)

    try:
        for root in roots:
            if root in seen_bases:
                continue
            seen_bases.add(root)
            dot_git = root / ".git"
            if dot_git.is_dir():
                cfg = dot_git / "config"
                if cfg.is_file() and cfg not in candidates:
                    candidates.append(cfg)
                break
            if dot_git.is_file():
                lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
                first_line = next((ln.strip() for ln in lines if ln.strip()), "")
                if first_line.lower().startswith("gitdir:"):
                    raw_gitdir = first_line.split(":", 1)[1].strip()
                    gitdir = Path(raw_gitdir)
                    if not gitdir.is_absolute():
                        gitdir = (root / gitdir).resolve()
                    wt_cfg = gitdir / "config"
                    if wt_cfg.is_file() and wt_cfg not in candidates:
                        candidates.append(wt_cfg)
                    commondir_file = gitdir / "commondir"
                    if commondir_file.is_file():
                        rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if rel_common:
                            common_base = Path(rel_common)
                            common_dir = (
                                common_base if common_base.is_absolute() else (gitdir / common_base).resolve()
                            )
                            common_cfg = common_dir / "config"
                            if common_cfg.is_file() and common_cfg not in candidates:
                                candidates.append(common_cfg)
                break
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
            if header.startswith("url "):
                for quote_ch in ('"', "'"):
                    if quote_ch in stripped:
                        quoted = stripped.split(quote_ch, 2)
                        if len(quoted) >= 2 and quoted[1].strip():
                            urls.append(quoted[1].strip())
                        break
            continue
        if in_remote_or_url_section and "=" in stripped:
            key, val = stripped.split("=", 1)
            if key.strip().lower() in ("url", "pushurl", "insteadof", "pushinsteadof"):
                cleaned_val = val.strip().strip('"').strip("'")
                if cleaned_val:
                    urls.append(cleaned_val)
    return urls


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
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
        cwd_path_obj = Path(cwd)
        if check_cwd_path:
            path_candidates: list[Path] = [cwd_path_obj]
            with suppress(OSError):
                resolved_ws = cwd_path_obj.resolve()
                if resolved_ws != cwd_path_obj:
                    path_candidates.append(resolved_ws)
            for ws_candidate in path_candidates:
                scrubbed_ws = _PYTEST_TMP_SEGMENT_RE.sub("pytest_tmp", str(ws_candidate))
                for hit in _find_corporate_markers(scrubbed_ws):
                    tag = f"cwd_path:{hit}"
                    if tag not in indicators:
                        indicators.append(tag)
                    matched_markers.add(hit)

        if check_git_remote:
            for cfg_path in _resolve_git_config_files(cwd_path_obj):
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
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Run a local command using list args (`shell=False`) and bounded timeout."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise SelftestError("Subprocess command must be a non-empty list/tuple of string arguments.")
    argv = [str(part) for part in cmd]
    max_cap = 15.0 if env is not None else SUBPROCESS_TIMEOUT_S
    bounded_timeout = min(max(0.1, float(timeout_s)), max_cap)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded_timeout,
            env=dict(env) if env is not None else None,
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
        raid_body.get("onboarding_feedback")
        if isinstance(raid_body.get("onboarding_feedback"), Mapping)
        else (
            raid_body.get("feedback")
            if isinstance(raid_body.get("feedback"), Mapping)
            else (
                raid_body.get("feedback_summary")
                if isinstance(raid_body.get("feedback_summary"), Mapping)
                else {}
            )
        )
    )
    raw_mean = fb_summary.get("mean_ease_rating")
    if raw_mean is None:
        raw_mean = fb_summary.get("average_ease_rating")
    if raw_mean is None:
        raw_mean = raid_body.get("mean_ease_rating", raid_body.get("average_ease_rating"))
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float))
        and not isinstance(raw_mean, bool)
        and math.isfinite(float(raw_mean))
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


def _safe_nonneg_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return default


def _is_nonneg_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(val):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _sanitize_feedback_summary_doc(raw: Mapping[str, Any]) -> dict[str, Any]:
    raw_obj = raw.get("object")
    obj_val = str(raw_obj) if raw_obj in _ACCEPTED_FEEDBACK_SUMMARY_OBJECTS else FEEDBACK_SUMMARY_OBJECT
    total_val = _safe_nonneg_int(raw.get("total_responses"))
    raw_mean = raw.get("mean_ease_rating")
    if raw_mean is None:
        raw_mean = raw.get("average_ease_rating")
    mean_val = _safe_opt_float(raw_mean, decimals=2)
    lag_val = _safe_opt_float(raw.get("zero_game_lag_ratio"), decimals=4)
    bp = raw.get("by_profile") if isinstance(raw.get("by_profile"), Mapping) else {}
    return {
        "object": obj_val,
        "total_responses": total_val,
        "mean_ease_rating": mean_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {
            "dev": _safe_nonneg_int(bp.get("dev")),
            "gamer": _safe_nonneg_int(bp.get("gamer")),
        },
    }


def _safe_opt_nonneg_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _safe_opt_float(value: Any, decimals: int = 2) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return round(float(value), decimals)
    return None


def sanitize_host_label(raw_label: object, *, home: Path | None = None) -> str:
    """Sanitize a personal PC host label without leaking corporate markers, secrets, or paths."""
    if raw_label is None or isinstance(raw_label, bool) or not isinstance(raw_label, str):
        return DEFAULT_HOST_LABEL
    cleaned = sanitize_text_for_display(raw_label.strip(), home=home)
    if not cleaned or _find_corporate_markers(cleaned):
        return DEFAULT_HOST_LABEL
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", cleaned).strip(".-:")[:128]
    if not normalized or _find_corporate_markers(normalized):
        return DEFAULT_HOST_LABEL
    return normalized


def sanitize_build_sha(raw: object) -> str:
    """Sanitize Cloud Run `build_sha` to alphanumeric/hex or `'unknown'` without leaking secrets."""
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


def validate_expected_sha(raw: object) -> str:
    """Validate an explicit `--expected-sha` argument (6..64 alphanumeric/hex characters)."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise SelftestError(
            "expected_sha must be a 6..64 character alphanumeric/hex SHA (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Return True iff `live_sha` matches `expected_sha` exactly or by hex prefix."""
    clean_live = sanitize_build_sha(live_sha)
    clean_exp = sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def sanitize_cloud_run_revision(raw: object) -> str | None:
    """Sanitize Cloud Run `K_REVISION` identifier or return `None`."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or not _CLOUD_RUN_REVISION_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return None
    return cleaned


def load_saved_credentials(path: Path) -> dict[str, Any] | None:
    """Load `~/.hivemind/credentials.json` safely if readable and holding a valid `hm-...` key."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    if not is_valid_hivemind_key(loaded.get("api_key")):
        return None
    return loaded


def build_ces_idempotency_key(
    *,
    gateway: str,
    ease_rating: int,
    zero_game_lag: bool,
    platform_tag: str,
    client_ref: str = "anon",
) -> str:
    """Compute a deterministic zero-PII `Idempotency-Key` for `POST /v1/hivemind/onboarding/feedback`."""
    seed = (
        f"noel-gamer-whisper-ces/v1|{gateway.rstrip('/')}|{client_ref}|"
        f"gamer|{ease_rating}|{1 if zero_game_lag else 0}|{platform_tag}"
    )
    return "ces-gamer-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def validate_opentrainer_heartbeat_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate payload strictly matching `OpenTrainerHeartbeatRequest` (`extra='forbid'`)."""
    if not isinstance(payload, Mapping):
        raise SelftestError("OpenTrainer heartbeat payload must be a JSON object.")
    extra_keys = set(payload.keys()) - _ALLOWED_OPENTRAINER_HB_KEYS
    if extra_keys:
        joined = ", ".join(sorted(str(k) for k in extra_keys))
        raise SelftestError(
            f"Forbidden extra keys in OpenTrainer heartbeat payload (`extra='forbid'`): {joined}."
        )
    clean_host = sanitize_host_label(payload.get("host_label"))
    net_mode = payload.get("network_mode", DEFAULT_NETWORK_MODE)
    if net_mode != DEFAULT_NETWORK_MODE:
        raise SelftestError(
            f"OpenTrainer heartbeat `network_mode` must be {DEFAULT_NETWORK_MODE!r}."
        )
    raw_stack = payload.get("training_stack", DEFAULT_TRAINING_STACK)
    if isinstance(raw_stack, bool) or not isinstance(raw_stack, str) or not raw_stack.strip():
        raise SelftestError("OpenTrainer heartbeat `training_stack` must be a non-empty string.")
    clean_stack = raw_stack.strip()[:128]
    raw_campaign = payload.get("active_campaign", DEFAULT_ACTIVE_CAMPAIGN)
    if isinstance(raw_campaign, bool) or not isinstance(raw_campaign, str) or not raw_campaign.strip():
        raise SelftestError("OpenTrainer heartbeat `active_campaign` must be a non-empty string.")
    clean_campaign = raw_campaign.strip()[:128]

    raw_gpu = payload.get("gpu_name")
    clean_gpu: str | None = None
    if raw_gpu is not None:
        if isinstance(raw_gpu, bool) or not isinstance(raw_gpu, str) or not raw_gpu.strip():
            raise SelftestError("gpu_name must be a non-empty string or None.")
        clean_gpu = sanitize_text_for_display(raw_gpu.strip())[:128]

    raw_tot = payload.get("vram_total_mib")
    clean_tot: int | None = None
    if raw_tot is not None:
        if (
            isinstance(raw_tot, bool)
            or not isinstance(raw_tot, int)
            or raw_tot < MIN_REPORTABLE_VRAM_TOTAL_MIB
            or raw_tot > MAX_GPU_VRAM_MIB
        ):
            raise SelftestError(
                f"vram_total_mib must be None or int in [{MIN_REPORTABLE_VRAM_TOTAL_MIB}, {MAX_GPU_VRAM_MIB}]."
            )
        clean_tot = int(raw_tot)

    raw_free = payload.get("vram_free_mib")
    clean_free: int | None = None
    if raw_free is not None:
        if (
            isinstance(raw_free, bool)
            or not isinstance(raw_free, int)
            or raw_free < 0
            or raw_free > MAX_GPU_VRAM_MIB
        ):
            raise SelftestError(f"vram_free_mib must be None or int in [0, {MAX_GPU_VRAM_MIB}].")
        clean_free = min(int(raw_free), clean_tot) if clean_tot is not None else int(raw_free)

    if clean_tot is None or clean_gpu is None:
        clean_gpu = None
        clean_tot = None
        clean_free = None

    validated: dict[str, Any] = {
        "host_label": clean_host,
        "gpu_name": clean_gpu,
        "vram_total_mib": clean_tot,
        "vram_free_mib": clean_free,
        "network_mode": DEFAULT_NETWORK_MODE,
        "training_stack": clean_stack,
        "active_campaign": clean_campaign,
    }
    _assert_safe_output(json.dumps(validated, sort_keys=True))
    return validated


def build_opentrainer_heartbeat_payload(
    *,
    gpu: GpuTelemetry,
    host_label: str = DEFAULT_HOST_LABEL,
    training_stack: str = DEFAULT_TRAINING_STACK,
    active_campaign: str = DEFAULT_ACTIVE_CAMPAIGN,
    home: Path | None = None,
) -> dict[str, Any]:
    """Build truthful `OpenTrainerHeartbeatRequest` payload from real `GpuTelemetry` (zero invented GPU specs)."""
    reportable_gpu = bool(
        gpu.present
        and not gpu.parse_error
        and gpu.name
        and gpu.name.strip()
        and gpu.total_vram_mib is not None
        and gpu.total_vram_mib >= MIN_REPORTABLE_VRAM_TOTAL_MIB
    )
    if reportable_gpu:
        assert gpu.name is not None
        assert gpu.total_vram_mib is not None
        gpu_name: str | None = sanitize_text_for_display(gpu.name.strip(), home=home)[:128]
        vram_total: int | None = int(gpu.total_vram_mib)
        vram_free: int | None = (
            min(max(0, int(gpu.free_vram_mib)), vram_total)
            if gpu.free_vram_mib is not None
            else None
        )
    else:
        gpu_name = None
        vram_total = None
        vram_free = None

    return validate_opentrainer_heartbeat_payload(
        {
            "host_label": sanitize_host_label(host_label, home=home),
            "gpu_name": gpu_name,
            "vram_total_mib": vram_total,
            "vram_free_mib": vram_free,
            "network_mode": DEFAULT_NETWORK_MODE,
            "training_stack": training_stack,
            "active_campaign": active_campaign,
        }
    )


def _resolve_heartbeat_credentials(
    *,
    api_key: str | None,
    credentials_path: Path | str | None,
    environ: Mapping[str, str] | None,
    home: Path,
) -> tuple[str | None, str, str | None, str]:
    """Resolve API key from explicit arg -> `HIVEMIND_API_KEY` -> `~/.hivemind/credentials.json`."""
    env_view = environ if environ is not None else os.environ
    cred_target = (
        Path(credentials_path)
        if (credentials_path is not None and str(credentials_path).strip())
        else (home / DEFAULT_CREDENTIALS_REL)
    )
    shown_cred = redact_home_path(cred_target, home=home)

    if api_key is not None and str(api_key).strip():
        cleaned = str(api_key).strip()
        if not is_valid_hivemind_key(cleaned):
            raise SelftestError(
                "Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key."
            )
        return cleaned, "explicit_arg", None, shown_cred

    env_key = str(env_view.get("HIVEMIND_API_KEY") or "").strip()
    if env_key:
        if not is_valid_hivemind_key(env_key):
            return None, "invalid_env_key", None, shown_cred
        return env_key, "env:HIVEMIND_API_KEY", None, shown_cred

    loaded = load_saved_credentials(cred_target)
    if loaded is not None:
        saved_role = str(loaded.get("role") or "").strip().lower() or None
        return str(loaded["api_key"]).strip(), "credentials_file", saved_role, shown_cred

    return None, "missing", None, shown_cred


def _send_opentrainer_heartbeat(
    *,
    gateway: str,
    gpu: GpuTelemetry,
    api_key: str | None,
    credentials_path: Path | str | None,
    host_label: str,
    http_fn: HttpFn,
    environ: Mapping[str, str] | None,
    home: Path,
    lang: str,
) -> dict[str, Any]:
    """Check `GET /v1/hivemind/me` and send outbound `POST /v1/hivemind/opentrainer/heartbeat`."""
    is_fr = lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_host = sanitize_host_label(host_label, home=home)
    resolved_key, key_source, saved_role, shown_cred = _resolve_heartbeat_credentials(
        api_key=api_key,
        credentials_path=credentials_path,
        environ=environ,
        home=home,
    )

    if key_source == "invalid_env_key":
        msg = (
            "Variable HIVEMIND_API_KEY invalide (préfixe 'hm-...' ou 'hm_live_...' requis)."
            if is_fr
            else "Invalid HIVEMIND_API_KEY environment variable (expected 'hm-...' or 'hm_live_...')."
        )
        return {
            "requested": True,
            "heartbeat_ok": False,
            "me_checked": False,
            "me_ok": False,
            "me_http_status": None,
            "sent": False,
            "accepted": False,
            "opentrainer_registered": False,
            "http_status": None,
            "status": "invalid_api_key",
            "key_source": key_source,
            "masked_api_key": "<invalid>",
            "credentials_path": shown_cred,
            "account_role": saved_role,
            "host_label": clean_host,
            "network_mode": DEFAULT_NETWORK_MODE,
            "training_stack": DEFAULT_TRAINING_STACK,
            "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
            "payload": None,
            "message": msg,
        }

    if resolved_key is None:
        msg = (
            f"Clé API HiveMind introuvable (--api-key, HIVEMIND_API_KEY ou {shown_cred}) pour le heartbeat OpenTrainer."
            if is_fr
            else f"HiveMind API key not found (--api-key, HIVEMIND_API_KEY, or {shown_cred}) for OpenTrainer heartbeat."
        )
        return {
            "requested": True,
            "heartbeat_ok": False,
            "me_checked": False,
            "me_ok": False,
            "me_http_status": None,
            "sent": False,
            "accepted": False,
            "opentrainer_registered": False,
            "http_status": None,
            "status": "missing_api_key",
            "key_source": key_source,
            "masked_api_key": None,
            "credentials_path": shown_cred,
            "account_role": saved_role,
            "host_label": clean_host,
            "network_mode": DEFAULT_NETWORK_MODE,
            "training_stack": DEFAULT_TRAINING_STACK,
            "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
            "payload": None,
            "message": msg,
        }

    masked_key = mask_api_key(resolved_key)
    auth_headers = {"Authorization": f"Bearer {resolved_key}"}
    me_code, me_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/me",
        None,
        auth_headers,
    )
    me_ok = me_code == 200
    account_role = (
        str(me_body.get("role") or "").strip().lower()
        if me_ok and isinstance(me_body.get("role"), str) and me_body["role"].strip()
        else saved_role
    )

    if not me_ok:
        if me_code in (401, 403):
            st = "auth_failed"
        elif me_code == 404:
            st = "missing_route"
        elif me_code == 429:
            st = "rate_limited"
        elif me_code >= 500:
            st = "server_error"
        elif me_code == 0:
            st = "network_error"
        else:
            st = "client_error"
        msg = (
            f"Vérification GET /v1/hivemind/me en échec (HTTP {me_code}, clé={masked_key})."
            if is_fr
            else f"GET /v1/hivemind/me verification failed (HTTP {me_code}, key={masked_key})."
        )
        return {
            "requested": True,
            "heartbeat_ok": False,
            "me_checked": True,
            "me_ok": False,
            "me_http_status": me_code,
            "sent": False,
            "accepted": False,
            "opentrainer_registered": False,
            "http_status": None,
            "status": st,
            "key_source": key_source,
            "masked_api_key": masked_key,
            "credentials_path": shown_cred,
            "account_role": account_role,
            "host_label": clean_host,
            "network_mode": DEFAULT_NETWORK_MODE,
            "training_stack": DEFAULT_TRAINING_STACK,
            "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
            "payload": None,
            "message": msg,
        }

    hb_payload = build_opentrainer_heartbeat_payload(
        gpu=gpu,
        host_label=clean_host,
        home=home,
    )
    hb_code, hb_body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
        hb_payload,
        auth_headers,
    )
    node_doc = hb_body.get("node") if isinstance(hb_body.get("node"), Mapping) else {}
    node_role = str(node_doc.get("role") or account_role or "").strip().lower() or None

    if hb_code in (200, 201):
        heartbeat_ok = True
        accepted = True
        registered = True
        st = "accepted"
        msg = (
            f"Heartbeat OpenTrainer accepté via relais HTTPS sortant (clé={masked_key}, campagne={DEFAULT_ACTIVE_CAMPAIGN})."
            if is_fr
            else f"Outbound OpenTrainer heartbeat accepted over HTTPS relay (key={masked_key}, campaign={DEFAULT_ACTIVE_CAMPAIGN})."
        )
    elif hb_code == 403 and account_role == "consumer":
        heartbeat_ok = True
        accepted = False
        registered = False
        st = "consumer_me_verified"
        msg = (
            f"Compte consumer vérifié via /v1/hivemind/me (clé={masked_key}) ; enregistrement GPU OpenTrainer ignoré."
            if is_fr
            else f"Consumer account verified via /v1/hivemind/me (key={masked_key}); OpenTrainer GPU registration skipped."
        )
    else:
        heartbeat_ok = False
        accepted = False
        registered = False
        if hb_code == 404:
            st = "stale_route_404"
        elif hb_code in (401, 403):
            st = "auth_failed"
        elif hb_code == 429:
            st = "rate_limited"
        elif hb_code >= 500:
            st = "server_error"
        elif hb_code == 0:
            st = "network_error"
        else:
            st = "client_error"
        msg = (
            f"Échec POST /v1/hivemind/opentrainer/heartbeat (HTTP {hb_code}, clé={masked_key})."
            if is_fr
            else f"Outbound POST /v1/hivemind/opentrainer/heartbeat failed (HTTP {hb_code}, key={masked_key})."
        )

    return {
        "requested": True,
        "heartbeat_ok": heartbeat_ok,
        "me_checked": True,
        "me_ok": True,
        "me_http_status": me_code,
        "sent": True,
        "accepted": accepted,
        "opentrainer_registered": registered,
        "http_status": hb_code,
        "status": st,
        "key_source": key_source,
        "masked_api_key": masked_key,
        "credentials_path": shown_cred,
        "account_role": node_role,
        "host_label": clean_host,
        "network_mode": DEFAULT_NETWORK_MODE,
        "training_stack": DEFAULT_TRAINING_STACK,
        "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
        "payload": hb_payload,
        "message": sanitize_text_for_display(msg, home=home),
    }


def _inspect_manifest_response(
    manifest_code: int,
    manifest_body: Mapping[str, Any],
    *,
    clean_expected_sha: str | None,
) -> dict[str, Any]:
    live_build_sha = sanitize_build_sha(manifest_body.get("build_sha"))
    cloud_run_revision = sanitize_cloud_run_revision(manifest_body.get("cloud_run_revision"))
    raw_routes = manifest_body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in EXPECTED_MANIFEST_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(EXPECTED_MANIFEST_HOLIDAY_ROUTES)

    meta_ok = bool(
        manifest_code == 200
        and manifest_body.get("object") == MANIFEST_OBJECT
        and manifest_body.get("campaign_id") in (CAMPAIGN_ID, None)
        and manifest_body.get("status") in ("ok", None)
        and manifest_body.get("personal_pc_only") is True
        and manifest_body.get("financial_promise") == FINANCIAL_PROMISE
    )
    manifest_valid = bool(meta_ok and len(missing_routes) == 0)

    sha_matches: bool | None = None
    sha_drift_detected = False
    if clean_expected_sha is not None and manifest_code == 200:
        sha_matches = shas_match(live_build_sha, clean_expected_sha)
        sha_drift_detected = not sha_matches

    return {
        "manifest_valid": manifest_valid,
        "live_build_sha": live_build_sha,
        "cloud_run_revision": cloud_run_revision,
        "missing_holiday_routes": missing_routes,
        "sha_matches": sha_matches,
        "sha_drift_detected": sha_drift_detected,
    }


def verify_live_fleet_and_badge(
    *,
    gateway: str = DEFAULT_GATEWAY,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    http_fn: HttpFn = default_http_json,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    enforce_personal_pc: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Probe zero-inference health/manifest/raid/fleet routes and compute deterministic `fleet_badge`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or target_online_producers < 1
    ):
        raise SelftestError("target_online_producers must be an integer >= 1.")

    clean_expected_sha = (
        validate_expected_sha(expected_sha)
        if (expected_sha is not None and str(expected_sha).strip())
        else None
    )

    if enforce_personal_pc:
        compliance = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
        if not compliance["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "checked": False,
                    "compliance_ok": False,
                    "exit_code": 2,
                    "gateway": clean_gw,
                    "fleet_ok": False,
                    "fleet_badge": "OFFLINE",
                    "status": "refused_compliance",
                    "holiday_routes_deployed_on_cloud_run": False,
                    "stale_cloud_run_detected": False,
                    "online_producers": 0,
                    "live_hiveminders": 0,
                    "total_hiveminders": 0,
                    "distinct_physical_gpus": 0,
                    "live_endpoints": 0,
                    "live_models": [],
                    "active_trainers": 0,
                    "registered_trainer_nodes": 0,
                    "target_online_producers": target_online_producers,
                    "target_producers_met": False,
                    "producers_remaining_to_target": target_online_producers,
                    "build_sha": "unknown",
                    "expected_sha": clean_expected_sha,
                    "sha_matches": None,
                    "sha_drift_detected": False,
                    "cloud_run_revision": None,
                    "missing_holiday_routes": [],
                    "health_http_status": None,
                    "manifest_http_status": None,
                    "raid_status_http_status": None,
                    "fleet_http_status": None,
                    "mean_ease_rating": None,
                    "zero_game_lag_ratio": None,
                    "total_responses": None,
                    "network_calls": 0,
                    "zero_inference_verified": True,
                    "compliance": compliance,
                    "message": sanitize_text_for_display(str(compliance["summary"])),
                }
            )

    health_code, health_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/health", None, None)
    manifest_code, manifest_body = _safe_http_call(
        http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/manifest", None, None
    )
    raid_code, raid_body = _safe_http_call(
        http_fn, "GET", f"{clean_gw}/v1/hivemind/raid/status", None, None
    )
    fleet_code, fleet_body = _safe_http_call(
        http_fn, "GET", f"{clean_gw}/v1/hivemind/producers/fleet", None, None
    )

    raw_health_status = health_body.get("status")
    health_ok = bool(
        health_code == 200
        and (raw_health_status is None or str(raw_health_status).strip().lower() == "ok")
    )

    manifest_info = _inspect_manifest_response(
        manifest_code,
        manifest_body,
        clean_expected_sha=clean_expected_sha,
    )
    manifest_valid = bool(manifest_info["manifest_valid"])
    sha_drift_detected = bool(manifest_info["sha_drift_detected"])
    live_build_sha = str(manifest_info["live_build_sha"])
    cloud_run_revision = manifest_info["cloud_run_revision"]
    missing_routes = list(manifest_info["missing_holiday_routes"])
    sha_matches = manifest_info["sha_matches"]

    raid_obj = raid_body.get("object")
    raid_ok = bool(
        raid_code == 200
        and raid_obj in (RAID_STATUS_OBJECT, None)
    )
    swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}
    fb_summary = (
        raid_body.get("onboarding_feedback")
        if isinstance(raid_body.get("onboarding_feedback"), Mapping)
        else (
            raid_body.get("feedback")
            if isinstance(raid_body.get("feedback"), Mapping)
            else (
                raid_body.get("feedback_summary")
                if isinstance(raid_body.get("feedback_summary"), Mapping)
                else {}
            )
        )
    )
    raw_mean = fb_summary.get("mean_ease_rating")
    if raw_mean is None:
        raw_mean = fb_summary.get("average_ease_rating")
    if raw_mean is None:
        raw_mean = raid_body.get("mean_ease_rating", raid_body.get("average_ease_rating"))
    mean_ease = _safe_opt_float(raw_mean, decimals=2)
    lag_ratio = _safe_opt_float(fb_summary.get("zero_game_lag_ratio"), decimals=4)
    total_responses = _safe_opt_nonneg_int(fb_summary.get("total_responses"))

    fleet_obj = fleet_body.get("object")
    raw_hm = fleet_body.get("hiveminders")
    endpoint_leak_detected = isinstance(raw_hm, list) and any(
        isinstance(row, Mapping) and "endpoint_url" in row for row in raw_hm
    )
    fleet_route_ok = bool(
        fleet_code == 200
        and (fleet_obj in _ACCEPTED_FLEET_OBJECTS or fleet_obj is None)
        and not endpoint_leak_detected
    )

    row_online = (
        sum(
            1
            for row in raw_hm
            if isinstance(row, Mapping)
            and str(row.get("status", "")).strip().lower() in LIVE_PRODUCER_STATUSES
        )
        if isinstance(raw_hm, list)
        else 0
    )
    if endpoint_leak_detected:
        online_producers = 0
    elif "live_hiveminders" in fleet_body:
        online_producers = max(_safe_nonneg_int(fleet_body.get("live_hiveminders")), row_online)
    elif isinstance(raw_hm, list):
        online_producers = row_online
    else:
        online_producers = _safe_nonneg_int(swarm.get("live_producers"))
    total_hiveminders = _safe_nonneg_int(
        fleet_body.get("total_hiveminders"),
        _safe_nonneg_int(swarm.get("registered_producers")),
    )
    distinct_physical_gpus = _safe_nonneg_int(fleet_body.get("distinct_physical_gpus"))
    live_endpoints = _safe_nonneg_int(fleet_body.get("live_endpoints"))
    raw_models = fleet_body.get("live_models")
    live_models = (
        [str(m) for m in raw_models if isinstance(m, str) and m.strip()]
        if isinstance(raw_models, list)
        else []
    )
    active_trainers = _safe_nonneg_int(swarm.get("active_trainers"))
    registered_trainer_nodes = _safe_nonneg_int(swarm.get("registered_trainer_nodes"))

    stale_404 = bool(manifest_code == 404 or raid_code == 404)
    stale_cloud_run_detected = bool(
        stale_404
        or sha_drift_detected
        or (manifest_code == 200 and not manifest_valid)
    )
    holiday_routes_deployed_on_cloud_run = bool(
        manifest_valid and raid_ok and not stale_cloud_run_detected
    )
    gateway_reachable = bool(
        health_ok and manifest_valid and raid_ok and fleet_route_ok and not sha_drift_detected
    )

    if stale_cloud_run_detected:
        fleet_badge = "STALE_CLOUD_RUN"
        status_slug = "stale_cloud_run"
        fleet_ok = False
        exit_code = 2
        msg = (
            f"Révision Cloud Run obsolète (badge={fleet_badge}, manifest={manifest_code}, "
            f"raid={raid_code}, sha={live_build_sha}, drift_sha={sha_drift_detected})."
            if is_fr
            else f"Stale Cloud Run revision detected (badge={fleet_badge}, manifest={manifest_code}, "
            f"raid={raid_code}, sha={live_build_sha}, sha_drift={sha_drift_detected})."
        )
    elif not gateway_reachable:
        fleet_badge = "OFFLINE"
        status_slug = "offline"
        fleet_ok = False
        exit_code = 1
        msg = (
            f"Passerelle ou flotte hors ligne (badge={fleet_badge}, /health={health_code}, "
            f"/manifest={manifest_code}, /raid={raid_code}, /fleet={fleet_code})."
            if is_fr
            else f"Gateway or producer fleet offline (badge={fleet_badge}, /health={health_code}, "
            f"/manifest={manifest_code}, /raid={raid_code}, /fleet={fleet_code})."
        )
    elif online_producers >= target_online_producers:
        fleet_badge = "ONLINE"
        status_slug = "online"
        fleet_ok = True
        exit_code = 0
        msg = (
            f"Flotte ONLINE ({online_producers}/{target_online_producers} producteurs en ligne, "
            f"{distinct_physical_gpus} GPUs distincts, {active_trainers} trainers actifs)."
            if is_fr
            else f"Fleet ONLINE ({online_producers}/{target_online_producers} online_producers, "
            f"{distinct_physical_gpus} distinct GPUs, {active_trainers} active trainers)."
        )
    else:
        fleet_badge = "BOOTSTRAPPING"
        status_slug = "bootstrapping"
        fleet_ok = True
        exit_code = 0
        msg = (
            f"Flotte BOOTSTRAPPING ({online_producers}/{target_online_producers} producteurs en ligne, "
            f"{distinct_physical_gpus} GPUs distincts, {active_trainers} trainers actifs)."
            if is_fr
            else f"Fleet BOOTSTRAPPING ({online_producers}/{target_online_producers} online_producers, "
            f"{distinct_physical_gpus} distinct GPUs, {active_trainers} active trainers)."
        )

    target_met = bool(online_producers >= target_online_producers)
    remaining = max(0, target_online_producers - online_producers)

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "checked": True,
            "compliance_ok": True,
            "exit_code": exit_code,
            "gateway": clean_gw,
            "fleet_ok": fleet_ok,
            "fleet_badge": fleet_badge,
            "status": status_slug,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed_on_cloud_run,
            "stale_cloud_run_detected": stale_cloud_run_detected,
            "online_producers": online_producers,
            "live_hiveminders": online_producers,
            "total_hiveminders": total_hiveminders,
            "distinct_physical_gpus": distinct_physical_gpus,
            "live_endpoints": live_endpoints,
            "live_models": live_models,
            "active_trainers": active_trainers,
            "registered_trainer_nodes": registered_trainer_nodes,
            "target_online_producers": target_online_producers,
            "target_producers_met": target_met,
            "producers_remaining_to_target": remaining,
            "build_sha": live_build_sha,
            "expected_sha": clean_expected_sha,
            "sha_matches": sha_matches,
            "sha_drift_detected": sha_drift_detected,
            "cloud_run_revision": cloud_run_revision,
            "missing_holiday_routes": missing_routes,
            "health_http_status": health_code,
            "health_status_text": str(health_body.get("status") or ("ok" if health_ok else "error")),
            "manifest_http_status": manifest_code,
            "raid_status_http_status": raid_code,
            "fleet_http_status": fleet_code,
            "campaign": str(raid_body.get("campaign") or "") or None,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": lag_ratio,
            "total_responses": total_responses,
            "network_calls": 4,
            "zero_inference_verified": True,
            "message": sanitize_text_for_display(msg),
        }
    )


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


def _expand_user_path(raw: Path | str, *, home: Path | None = None) -> Path:
    text = str(raw).strip()
    if text == "~":
        return home if home is not None else Path.home()
    if text.startswith(("~/", "~\\")):
        base = home if home is not None else Path.home()
        return base / text[2:]
    return Path(text)


def _validate_cli_path_arg(
    raw: str | None,
    label: str,
    *,
    home: Path | None = None,
) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise SelftestError(f"Invalid {label} path.")
    cleaned = raw.strip()
    if not cleaned or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise SelftestError(f"Invalid {label} path.")
    return _expand_user_path(cleaned, home=home)


def _write_file_secure_atomic(
    path: Path,
    content: str,
    *,
    mode: int = 0o600,
    dir_mode: int | None = None,
) -> Path:
    """Atomically persist UTF-8 text via unique `.tmp`, POSIX `0600` creation, `fsync`, and `os.replace`."""
    if path.exists() and path.is_dir():
        raise SelftestError(f"Refusing to overwrite directory {redact_home_path(path)!r}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" and dir_mode is not None:
        with suppress(OSError):
            os.chmod(path.parent, dir_mode)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"
    )
    data = content.encode("utf-8")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            if os.name != "nt":
                with suppress(OSError):
                    os.chmod(tmp_path, mode)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
        if os.name != "nt":
            with suppress(OSError):
                os.chmod(path, mode)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def default_ces_queue_path(
    home: Path | str | None = None,
    credentials_path: Path | str | None = None,
) -> Path:
    """Resolve isolated local POSIX-0600 CES retry spool path without touching real `~/.hivemind`."""
    default_cred = Path.home() / DEFAULT_CREDENTIALS_REL
    if credentials_path is not None and str(credentials_path).strip():
        home_p = Path(home) if home is not None else None
        cred_p = _expand_user_path(credentials_path, home=home_p)
        if home_p is not None and cred_p == default_cred:
            with suppress(OSError):
                if home_p.resolve() != Path.home().resolve():
                    return home_p / DEFAULT_CES_QUEUE_REL
        return cred_p.parent / DEFAULT_CES_QUEUE_FILENAME
    resolved_base = Path(home) if home is not None else Path.home()
    return resolved_base / DEFAULT_CES_QUEUE_REL


def _get_spool_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _SPOOL_LOCKS_GUARD:
        lock = _SPOOL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SPOOL_LOCKS[key] = lock
        return lock


def _normalize_ces_event_id(
    *,
    gateway: str,
    ease_rating: int,
    zero_game_lag: bool,
    platform_tag: str,
    profile: str,
    idempotency_key: str | None = None,
    client_ref: str = "anon",
) -> str:
    if idempotency_key is not None:
        if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
            raise SelftestError("idempotency_key must be a printable ASCII string.")
        if not idempotency_key.isascii() or any(ord(ch) < 32 or ord(ch) == 127 for ch in idempotency_key):
            raise SelftestError("idempotency_key must contain printable ASCII characters only.")
        cleaned = idempotency_key.strip()
        if not cleaned:
            raise SelftestError("idempotency_key must not be empty.")
        if _VALID_EVENT_ID_RE.fullmatch(cleaned) and not _API_KEY_RE.search(cleaned):
            return cleaned
        return "ces-gamer-" + hashlib.sha256(f"gamer-ces-key/v1|{cleaned}".encode()).hexdigest()[:32]
    if profile == "gamer":
        return build_ces_idempotency_key(
            gateway=gateway,
            ease_rating=ease_rating,
            zero_game_lag=zero_game_lag,
            platform_tag=platform_tag,
            client_ref=client_ref,
        )
    seed = f"noel-ces/{profile}|{gateway.rstrip('/')}|{client_ref}|{ease_rating}|{int(zero_game_lag)}|{platform_tag}"
    return "ces-gamer-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _normalize_ces_spool_payload(
    *,
    profile: str = "gamer",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_tag: str | None = None,
    note: str = "",
    home: Path | None = None,
) -> dict[str, Any]:
    norm_prof = str(profile or "").strip().lower()
    if norm_prof not in ("gamer", "dev"):
        raise SelftestError("CES profile must be 'gamer' or 'dev'.")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise SelftestError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise SelftestError("zero_game_lag must be a boolean.")
    clean_plat = str(platform_tag or _platform_tag()).strip()[:32] or "unknown"
    payload: dict[str, Any] = {
        "profile": norm_prof,
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "platform": clean_plat,
    }
    clean_note = sanitize_text_for_display(str(note or ""), home=home).strip()[:240]
    if clean_note:
        payload["note"] = clean_note
    return payload


def _validate_spooled_ces_payload(raw: object, *, home: Path | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SelftestError("Spooled CES payload must be a JSON object.")
    if set(raw.keys()) - _ALLOWED_FEEDBACK_POST_KEYS:
        raise SelftestError("Spooled CES payload contains unexpected keys.")
    return _normalize_ces_spool_payload(
        profile=raw.get("profile", ""),  # type: ignore[arg-type]
        ease_rating=raw.get("ease_rating", 0),  # type: ignore[arg-type]
        zero_game_lag=raw.get("zero_game_lag", True),  # type: ignore[arg-type]
        platform_tag=str(raw.get("platform") or _platform_tag()),
        note=str(raw.get("note") or ""),
        home=home,
    )


class GamerCesRetrySpool:
    """Thread-safe, atomic POSIX-0600 bounded CES retry spool with Idempotency-Key deduplication."""

    def __init__(
        self,
        spool_path: Path | str | None = None,
        *,
        home: Path | None = None,
        credentials_path: Path | str | None = None,
        max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
        max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not (1 <= max_entries <= 4096):
            raise SelftestError("max_entries must be an integer between 1 and 4096.")
        if (
            isinstance(max_delivered, bool)
            or not isinstance(max_delivered, int)
            or not (1 <= max_delivered <= 32768)
        ):
            raise SelftestError("max_delivered must be an integer between 1 and 32768.")
        self._home = Path(home) if home is not None else None
        raw_p = (
            _expand_user_path(spool_path, home=self._home)
            if spool_path is not None and str(spool_path).strip()
            else default_ces_queue_path(home=self._home, credentials_path=credentials_path)
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
                    "event_id": "ces-gamer-" + hashlib.sha256(b"corrupted_spool_recovered").hexdigest()[:32],
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
                if isinstance(item, str) and _VALID_EVENT_ID_RE.fullmatch(item) and item not in seen_delivered:
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
                    if _VALID_EVENT_ID_RE.fullmatch(q_eid):
                        q_code = q_item.get("http_status", 422)
                        quarantined.append(
                            {
                                "event_id": q_eid,
                                "http_status": int(q_code)
                                if isinstance(q_code, int) and not isinstance(q_code, bool)
                                else 422,
                                "reason": sanitize_text_for_display(
                                    str(q_item.get("reason", "quarantined"))[:80], home=self._home
                                ),
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
                    if isinstance(raw_eid, str) and _VALID_EVENT_ID_RE.fullmatch(raw_eid)
                    else "ces-gamer-"
                    + hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
                )
                if eid in seen_delivered or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = _validate_spooled_ces_payload(entry.get("payload"), home=self._home)
                except SelftestError:
                    mutated = True
                    quarantined.append({"event_id": eid, "http_status": 422, "reason": "invalid_spooled_payload"})
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
                        "last_reason": sanitize_text_for_display(
                            str(entry.get("last_reason") or "")[:80], home=self._home
                        ),
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
        serialized = _assert_safe_output(
            json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            home=self._home,
        )
        dir_mode = 0o700 if self._path.parent.name == ".hivemind" else None
        _write_file_secure_atomic(self._path, serialized, mode=0o600, dir_mode=dir_mode)

    def snapshot(self, *, home: Path | None = None) -> dict[str, Any]:
        """Return a read-only sanitized spool snapshot without creating files when empty."""
        eff_home = home if home is not None else self._home
        with self._lock:
            exists = self._path.is_file()
            state, _ = self._load_state_unlocked()
        return _attach_receipt(
            {
                "object": "hivemind.gamer_whisper.ces_spool_snapshot",
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
        )

    def mark_delivered(self, event_id: str) -> bool:
        """Record `event_id` as delivered only if a spool file already exists (never creates empty files)."""
        if not isinstance(event_id, str) or not _VALID_EVENT_ID_RE.fullmatch(event_id):
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

    def clean(self, *, home: Path | None = None) -> dict[str, Any]:
        """Remove the local spool file if present without contacting any remote host."""
        eff_home = home if home is not None else self._home
        with self._lock:
            existed = self._path.is_file()
            if existed:
                self._path.unlink(missing_ok=True)
        return _attach_receipt(
            {
                "object": "hivemind.gamer_whisper.ces_spool_clean",
                "spool_file": redact_home_path(self._path, home=eff_home),
                "removed": existed,
                "spool_exists": self._path.is_file(),
                "network_calls": 0,
            }
        )

    def enqueue(
        self,
        *,
        profile: str = "gamer",
        ease_rating: int = 5,
        zero_game_lag: bool = True,
        platform_tag: str | None = None,
        note: str = "",
        gateway: str = DEFAULT_GATEWAY,
        idempotency_key: str | None = None,
        client_ref: str = "anon",
        http_status: int | None = None,
        reason: str = "",
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        check_git_remote: bool = True,
        check_cwd_path: bool = True,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Enforce compliance, deduplicate by `Idempotency-Key`, evict FIFO at `max_entries`, and save `0600`."""
        lang_code = normalize_lang(lang)
        eff_home = home if home is not None else self._home
        clean_gw = validate_gateway_url(gateway)
        comp = evaluate_personal_pc_compliance(
            environ=environ,
            cwd=cwd,
            check_git_remote=check_git_remote,
            check_cwd_path=check_cwd_path,
            lang=lang_code,
        )
        if not comp["compliance_ok"]:
            return _attach_receipt(
                {
                    "object": "hivemind.gamer_whisper.ces_spool_enqueue",
                    "compliance_ok": False,
                    "corporate_indicators": comp["corporate_indicators"],
                    "enqueued": False,
                    "deduplicated": False,
                    "duplicate_state": "compliance_refused",
                    "evicted_oldest": False,
                    "spool_file": redact_home_path(self._path, home=eff_home),
                    "network_calls": 0,
                    "file_writes": 0,
                    "summary": sanitize_text_for_display(comp["summary"], home=eff_home),
                }
            )

        clean_payload = _normalize_ces_spool_payload(
            profile=profile,
            ease_rating=ease_rating,
            zero_game_lag=zero_game_lag,
            platform_tag=platform_tag,
            note=note,
            home=eff_home,
        )
        event_id = _normalize_ces_event_id(
            gateway=clean_gw,
            ease_rating=int(clean_payload["ease_rating"]),
            zero_game_lag=bool(clean_payload["zero_game_lag"]),
            platform_tag=str(clean_payload["platform"]),
            profile=str(clean_payload["profile"]),
            idempotency_key=idempotency_key,
            client_ref=client_ref,
        )
        clean_reason = sanitize_text_for_display(str(reason or ""), home=eff_home)[:80]

        with self._lock:
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_event_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if event_id in delivered_ids or any(item["event_id"] == event_id for item in pending):
                dup_state = "already_delivered" if event_id in delivered_ids else "already_pending"
                if recovered:
                    self._atomic_save_unlocked(state)
                return _attach_receipt(
                    {
                        "object": "hivemind.gamer_whisper.ces_spool_enqueue",
                        "compliance_ok": True,
                        "spool_file": redact_home_path(self._path, home=eff_home),
                        "enqueued": False,
                        "deduplicated": True,
                        "duplicate_state": dup_state,
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
                )

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
            return _attach_receipt(
                {
                    "object": "hivemind.gamer_whisper.ces_spool_enqueue",
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
            )

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        http_fn: HttpFn = default_http_json,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = HTTP_TIMEOUT_S,
        lang: str = "en",
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        check_git_remote: bool = True,
        check_cwd_path: bool = True,
    ) -> dict[str, Any]:
        """Idempotently flush pending CES events with `Idempotency-Key` without creating empty spool files."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        eff_home = home if home is not None else self._home
        clean_gw = validate_gateway_url(gateway)
        bounded_timeout = _validate_http_timeout_s(timeout_s)

        comp = evaluate_personal_pc_compliance(
            environ=environ,
            cwd=cwd,
            check_git_remote=check_git_remote,
            check_cwd_path=check_cwd_path,
            lang=lang_code,
        )
        if not comp["compliance_ok"]:
            return _attach_receipt(
                {
                    "object": "hivemind.gamer_whisper.ces_spool_flush",
                    "gateway": clean_gw,
                    "lang": lang_code,
                    "compliance_ok": False,
                    "corporate_indicators": comp["corporate_indicators"],
                    "flush_ok": False,
                    "exit_code": 2,
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
                    "diagnostics": [sanitize_text_for_display(comp["summary"], home=eff_home)],
                }
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
                        "object": "hivemind.gamer_whisper.ces_spool_flush",
                        "gateway": clean_gw,
                        "lang": lang_code,
                        "compliance_ok": True,
                        "flush_ok": True,
                        "exit_code": 0,
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
                )

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

                req_headers = {str(k): str(v) for k, v in headers.items()} if headers else {}
                req_headers["Idempotency-Key"] = eid
                network_calls += 1
                code, body = _safe_http_call(
                    http_fn,
                    "POST",
                    post_url,
                    dict(entry["payload"]),
                    req_headers,
                    timeout_s=bounded_timeout,
                )
                mutated = True

                if code in (200, 201):
                    delivered_now += 1
                    state["delivered_event_ids"].append(eid)
                    delivered_set.add(eid)
                    latest_summary = _sanitize_feedback_summary_doc(body)
                elif code in _NON_BLOCKING_FEEDBACK_CODES or code >= 500:
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
                else "HTTP 404 during CES spool flush: entries safely retained for retry."
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
        exit_code = 0 if flush_ok else (2 if (stale_404_seen or quarantined_now > 0) else 1)
        return _attach_receipt(
            {
                "object": "hivemind.gamer_whisper.ces_spool_flush",
                "gateway": clean_gw,
                "lang": lang_code,
                "compliance_ok": True,
                "flush_ok": flush_ok,
                "exit_code": exit_code,
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
        )


def enqueue_gamer_ces_event(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "gamer",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    platform_tag: str | None = None,
    note: str = "",
    idempotency_key: str | None = None,
    client_ref: str = "anon",
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_status: int | None = None,
    reason: str = "",
    max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
) -> dict[str, Any]:
    """Atomically enqueue a zero-PII Gamer CES event into the bounded POSIX-0600 retry spool."""
    eff_plat = platform_tag if platform_tag is not None else _platform_tag(system_name)
    spool = GamerCesRetrySpool(
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
        platform_tag=eff_plat,
        note=note,
        gateway=gateway,
        idempotency_key=idempotency_key,
        client_ref=client_ref,
        http_status=http_status,
        reason=reason,
        home=home,
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        check_cwd_path=check_cwd_path,
        lang=lang,
    )


def flush_gamer_ces_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
    max_entries: int = DEFAULT_CES_QUEUE_MAX_ENTRIES,
    max_delivered: int = DEFAULT_CES_QUEUE_MAX_DELIVERED,
    lang: str = "en",
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
) -> dict[str, Any]:
    """Idempotently flush pending Gamer CES events with `Idempotency-Key` headers."""
    spool = GamerCesRetrySpool(
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
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        check_cwd_path=check_cwd_path,
    )


def submit_gamer_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    ease_rating: int,
    zero_game_lag: bool,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
    non_blocking_feedback: bool = False,
    idempotency_key: str | None = None,
    client_ref: str = "anon",
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
) -> dict[str, Any]:
    """Post zero-PII gamer comfort telemetry (`profile='gamer'`) to `/v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise SelftestError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise SelftestError("zero_game_lag must be a boolean.")

    platform_tag = _platform_tag(system_name)
    idem_key = (
        idempotency_key.strip()
        if (isinstance(idempotency_key, str) and idempotency_key.strip())
        else build_ces_idempotency_key(
            gateway=clean_gw,
            ease_rating=ease_rating,
            zero_game_lag=zero_game_lag,
            platform_tag=platform_tag,
            client_ref=client_ref,
        )
    )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in idem_key):
        raise SelftestError("idempotency_key must not contain control characters.")

    payload: dict[str, Any] = {
        "profile": "gamer",
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": platform_tag,
    }
    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        {"Idempotency-Key": idem_key},
    )

    raw_mean = body.get("mean_ease_rating")
    if raw_mean is None:
        raw_mean = body.get("average_ease_rating")
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float))
        and not isinstance(raw_mean, bool)
        and math.isfinite(float(raw_mean))
        else None
    )
    raw_ratio = body.get("zero_game_lag_ratio")
    lag_ratio = (
        round(float(raw_ratio), 4)
        if isinstance(raw_ratio, (int, float))
        and not isinstance(raw_ratio, bool)
        and math.isfinite(float(raw_ratio))
        else None
    )
    raw_total = body.get("total_responses")
    total_resp = (
        int(raw_total)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )

    degradable_error = status_code in _NON_BLOCKING_FEEDBACK_CODES or status_code >= 500
    degraded = bool(non_blocking_feedback and degradable_error)
    spooled = False
    spool_file_display: str | None = None

    if status_code in (200, 201):
        outcome = "recorded"
        submitted = True
        if spool_path is not None:
            GamerCesRetrySpool(
                spool_path,
                home=home,
                credentials_path=credentials_path,
            ).mark_delivered(idem_key)
        msg = (
            f"Merci ! Retour CES Gamer enregistré (ease_rating={ease_rating}/5, "
            f"profil=gamer, zero_game_lag={zero_game_lag})."
            if is_fr
            else f"Thank you! Anonymous Gamer CES feedback recorded (ease_rating={ease_rating}/5, "
            f"profile=gamer, zero_game_lag={zero_game_lag})."
        )
    else:
        submitted = False
        if status_code == 404:
            outcome = "missing_route"
            base_msg = (
                "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 (révision Cloud Run antérieure "
                "à PR #949) ; repli non bloquant activé (ces_recorded=False). Relancez après cutover Cloud Run."
                if (degraded and is_fr)
                else (
                    "POST /v1/hivemind/onboarding/feedback returned HTTP 404 (stale pre-PR-#949 Cloud Run); "
                    "degraded gracefully without blocking (ces_recorded=False). Retry after Cloud Run cutover."
                    if degraded
                    else (
                        "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 : la révision Cloud Run est en "
                        "retard sur `main` (route de retour Noël absente)."
                        if is_fr
                        else "POST /v1/hivemind/onboarding/feedback returned HTTP 404: Cloud Run revision lags behind "
                        "`main` (holiday feedback route missing)."
                    )
                )
            )
        elif status_code == 429:
            outcome = "rate_limited"
            base_msg = (
                "Limite de débit atteinte (HTTP 429) sur /v1/hivemind/onboarding/feedback ; "
                "repli non bloquant activé (ces_recorded=False), réessayez dans 1 minute."
                if (degraded and is_fr)
                else (
                    "Rate limit reached (HTTP 429) on /v1/hivemind/onboarding/feedback; "
                    "degraded gracefully (ces_recorded=False), retry in one minute."
                    if degraded
                    else (
                        "Limite de débit atteinte (HTTP 429) sur /v1/hivemind/onboarding/feedback ; réessayez dans 1 minute."
                        if is_fr
                        else "Rate limit reached (HTTP 429) on /v1/hivemind/onboarding/feedback; retry in one minute."
                    )
                )
            )
        elif status_code >= 500:
            outcome = "server_error"
            base_msg = (
                f"Erreur serveur passerelle (HTTP {status_code}) sur /v1/hivemind/onboarding/feedback ; "
                "repli non bloquant activé (ces_recorded=False)."
                if (degraded and is_fr)
                else (
                    f"Temporary gateway server error (HTTP {status_code}) on /v1/hivemind/onboarding/feedback; "
                    "degraded gracefully (ces_recorded=False)."
                    if degraded
                    else (
                        f"Erreur temporaire du serveur passerelle (HTTP {status_code}) lors de l'envoi du retour CES Gamer."
                        if is_fr
                        else f"Temporary gateway server error (HTTP {status_code}) while submitting Gamer CES feedback."
                    )
                )
            )
        elif status_code in (0, 408):
            outcome = "network_error"
            base_msg = (
                "Délai réseau dépassé sur /v1/hivemind/onboarding/feedback ; "
                "repli non bloquant activé (ces_recorded=False)."
                if (degraded and is_fr)
                else (
                    "Network error/timeout reaching /v1/hivemind/onboarding/feedback; "
                    "degraded gracefully (ces_recorded=False)."
                    if degraded
                    else (
                        "Impossible de joindre /v1/hivemind/onboarding/feedback (erreur réseau)."
                        if is_fr
                        else "Could not reach /v1/hivemind/onboarding/feedback (network error)."
                    )
                )
            )
        else:
            outcome = "client_error"
            base_msg = (
                f"Envoi du retour CES Gamer refusé par la passerelle (HTTP {status_code})."
                if is_fr
                else f"Gamer CES feedback rejected by gateway (HTTP {status_code})."
            )

        msg = base_msg
        if degradable_error and spool_path is not None:
            enq = enqueue_gamer_ces_event(
                clean_gw,
                profile="gamer",
                ease_rating=ease_rating,
                zero_game_lag=zero_game_lag,
                platform_tag=platform_tag,
                idempotency_key=idem_key,
                client_ref=client_ref,
                spool_path=spool_path,
                credentials_path=credentials_path,
                http_status=status_code,
                reason=outcome,
                lang=lang_code,
                home=home,
                environ=environ,
                cwd=cwd,
                check_git_remote=check_git_remote,
                check_cwd_path=check_cwd_path,
            )
            if enq.get("compliance_ok") and (enq.get("enqueued") or enq.get("deduplicated")):
                spooled = True
                spool_file_display = str(enq["spool_file"])
                msg = (
                    f"{base_msg} File locale 0600 : {spool_file_display}."
                    if is_fr
                    else f"{base_msg} Spooled locally at {spool_file_display}."
                )

    return {
        "attempted": True,
        "submitted": submitted,
        "ces_recorded": submitted,
        "non_blocking": bool(non_blocking_feedback),
        "degraded": degraded,
        "spooled": spooled,
        "spool_file": spool_file_display,
        "skipped": False,
        "status": outcome,
        "http_status": status_code,
        "idempotency_key": idem_key,
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
    non_blocking_feedback: bool = False,
    idempotency_key: str | None = None,
    client_ref: str = "anon",
    spool_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
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
            "ces_recorded": False,
            "non_blocking": bool(non_blocking_feedback),
            "degraded": False,
            "spooled": False,
            "spool_file": None,
            "skipped": True,
            "status": "skipped",
            "http_status": None,
            "idempotency_key": None,
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
        non_blocking_feedback=non_blocking_feedback,
        idempotency_key=idempotency_key,
        client_ref=client_ref,
        spool_path=spool_path,
        credentials_path=credentials_path,
        home=home,
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        check_cwd_path=check_cwd_path,
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
    noel_verify: bool = False,
    verify_fleet: bool = False,
    send_opentrainer_heartbeat: bool = False,
    api_key: str | None = None,
    credentials_path: Path | str | None = None,
    host_label: str = DEFAULT_HOST_LABEL,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    non_blocking_feedback: bool = False,
    expected_sha: str | None = None,
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
    spool_path: Path | str | None = None,
    flush_ces_queue: bool = False,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
) -> dict[str, Any]:
    """Run the deterministic `<10s` Gamer-Whisper pre-flight self-test and return a signed report."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    resolved_cwd = cwd if cwd is not None else Path.cwd()
    clean_host_label = sanitize_host_label(host_label, home=resolved_home)

    if isinstance(yield_slo_ms, bool) or not math.isfinite(float(yield_slo_ms)) or float(yield_slo_ms) <= 0:
        raise SelftestError("yield_slo_ms must be a positive finite millisecond budget.")
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or target_online_producers < 1
    ):
        raise SelftestError("target_online_producers must be an integer >= 1.")
    if expected_sha is not None and str(expected_sha).strip():
        validate_expected_sha(expected_sha)
    if api_key is not None and str(api_key).strip() and not is_valid_hivemind_key(str(api_key).strip()):
        raise SelftestError("Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key.")
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
        check_git_remote=check_git_remote,
        check_cwd_path=check_cwd_path,
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
                "holiday_routes_deployed_on_cloud_run": None,
                "stale_cloud_run_detected": False,
                "online_producers": None,
                "distinct_physical_gpus": None,
                "active_trainers": None,
                "target_online_producers": target_online_producers,
                "fleet_badge": None,
                "ces_recorded": False,
                "masked_api_key": None,
                "credentials_path": None,
                "compliance": compliance,
                "preemption_drill": None,
                "thermal_vram_governor": None,
                "cooldown_policy": None,
                "bandwidth_cap": None,
                "gateway_check": None,
                "fleet_check": None,
                "opentrainer_heartbeat": None,
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

    local_hard_fail = bool(
        (not yield_slo_ok)
        or (governor["check_status"] == "fail")
        or (not cooldown_info["cooldown_policy_ok"])
        or (not bandwidth_info["bandwidth_cap_ok"])
    )
    governor_paused = bool(
        (governor["check_status"] == "warn")
        or live_detection["game_detected"]
        or (not cooldown_info["resume_allowed_now"])
    )

    # Step 3(a): Optional outbound OpenTrainer heartbeat (after local safety checks pass)
    hb_info: dict[str, Any] | None = None
    if send_opentrainer_heartbeat:
        if local_hard_fail:
            hb_status = "blocked_yield_slo" if not yield_slo_ok else "blocked_safety_gate"
            hb_msg = (
                "Heartbeat OpenTrainer bloqué : échec du contrôle local de sécurité/cession <1.5s."
                if is_fr
                else "OpenTrainer heartbeat held back: local safety or <1.5s yield SLO check failed."
            )
            hb_info = {
                "requested": True,
                "heartbeat_ok": False,
                "me_checked": False,
                "me_ok": False,
                "me_http_status": None,
                "sent": False,
                "accepted": False,
                "opentrainer_registered": False,
                "http_status": None,
                "status": hb_status,
                "key_source": None,
                "masked_api_key": mask_api_key(api_key) if api_key else None,
                "credentials_path": redact_home_path(
                    credentials_path or (resolved_home / DEFAULT_CREDENTIALS_REL),
                    home=resolved_home,
                ),
                "account_role": None,
                "host_label": clean_host_label,
                "network_mode": DEFAULT_NETWORK_MODE,
                "training_stack": DEFAULT_TRAINING_STACK,
                "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
                "payload": None,
                "message": hb_msg,
            }
        elif governor_paused:
            hb_msg = (
                "Heartbeat OpenTrainer suspendu par le gouverneur local (jeu actif, >70 C, VRAM ou refroidissement)."
                if is_fr
                else "OpenTrainer heartbeat held paused by local governor (active game, >70 C, VRAM reserve, or cooldown)."
            )
            hb_info = {
                "requested": True,
                "heartbeat_ok": True,
                "me_checked": False,
                "me_ok": False,
                "me_http_status": None,
                "sent": False,
                "accepted": False,
                "opentrainer_registered": False,
                "http_status": None,
                "status": "paused_governor_hold",
                "key_source": None,
                "masked_api_key": mask_api_key(api_key) if api_key else None,
                "credentials_path": redact_home_path(
                    credentials_path or (resolved_home / DEFAULT_CREDENTIALS_REL),
                    home=resolved_home,
                ),
                "account_role": None,
                "host_label": clean_host_label,
                "network_mode": DEFAULT_NETWORK_MODE,
                "training_stack": DEFAULT_TRAINING_STACK,
                "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
                "payload": None,
                "message": hb_msg,
            }
            warnings.append(sanitize_text_for_display(hb_msg, home=resolved_home))
        else:
            hb_info = _send_opentrainer_heartbeat(
                gateway=clean_gw,
                gpu=gpu,
                api_key=api_key,
                credentials_path=credentials_path,
                host_label=clean_host_label,
                http_fn=http_fn,
                environ=environ,
                home=resolved_home,
                lang=lang_code,
            )
            if not hb_info["heartbeat_ok"]:
                errors.append(sanitize_text_for_display(hb_info["message"], home=resolved_home))

    # Step 3(b): Optional zero-inference fleet / holiday-route verifier & legacy gateway check
    fleet_info: dict[str, Any] | None = None
    gw_info: dict[str, Any] | None = None
    if noel_verify or verify_fleet:
        fleet_info = verify_live_fleet_and_badge(
            gateway=clean_gw,
            expected_sha=expected_sha,
            target_online_producers=target_online_producers,
            http_fn=http_fn,
            environ=environ,
            cwd=resolved_cwd,
            enforce_personal_pc=False,
            lang=lang_code,
        )
        if not fleet_info["fleet_ok"]:
            errors.append(sanitize_text_for_display(fleet_info["message"], home=resolved_home))
        if check_gateway:
            gw_info = {
                "checked": True,
                "gateway_ok": bool(
                    fleet_info["health_http_status"] == 200
                    and fleet_info["raid_status_http_status"] == 200
                ),
                "status": fleet_info["status"],
                "health_http_status": fleet_info["health_http_status"],
                "raid_status_http_status": fleet_info["raid_status_http_status"],
                "health_status_text": fleet_info["health_status_text"],
                "campaign": fleet_info["campaign"],
                "active_trainers": fleet_info["active_trainers"],
                "mean_ease_rating": fleet_info["mean_ease_rating"],
                "zero_inference_verified": True,
                "message": fleet_info["message"],
            }
    elif check_gateway:
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

    ces_key_ref = (
        str(hb_info.get("masked_api_key") or "anon")
        if hb_info is not None
        else (mask_api_key(api_key) if api_key else "anon")
    )
    ces_client_ref = f"{clean_host_label}:{ces_key_ref}"

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
            non_blocking_feedback=non_blocking_feedback,
            client_ref=ces_client_ref,
            spool_path=spool_path,
            credentials_path=credentials_path,
            home=resolved_home,
            environ=environ,
            cwd=resolved_cwd,
            check_git_remote=check_git_remote,
            check_cwd_path=check_cwd_path,
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
            non_blocking_feedback=non_blocking_feedback,
            client_ref=ces_client_ref,
            spool_path=spool_path,
            credentials_path=credentials_path,
            home=resolved_home,
            environ=environ,
            cwd=resolved_cwd,
            check_git_remote=check_git_remote,
            check_cwd_path=check_cwd_path,
        )

    ces_flush_info: dict[str, Any] | None = None
    if flush_ces_queue:
        eff_spool = spool_path if spool_path is not None else default_ces_queue_path(
            home=resolved_home,
            credentials_path=credentials_path,
        )
        ces_flush_info = flush_gamer_ces_queue(
            clean_gw,
            spool_path=eff_spool,
            credentials_path=credentials_path,
            http_fn=http_fn,
            lang=lang_code,
            home=resolved_home,
            environ=environ,
            cwd=resolved_cwd,
            check_git_remote=check_git_remote,
            check_cwd_path=check_cwd_path,
        )
        for flush_diag in ces_flush_info.get("diagnostics") or ():
            warnings.append(sanitize_text_for_display(str(flush_diag), home=resolved_home))

    final_zero_game_lag = (
        bool(fb_info["zero_game_lag"])
        if (fb_info is not None and not fb_info["skipped"])
        else derived_zero_game_lag
    )
    fb_degraded_non_blocking = bool(
        fb_info is not None and non_blocking_feedback and fb_info.get("degraded")
    )
    if fb_info is not None and fb_info["attempted"] and not fb_info["submitted"]:
        fb_msg = sanitize_text_for_display(str(fb_info["message"]), home=resolved_home)
        if fb_degraded_non_blocking:
            warnings.append(fb_msg)
        else:
            errors.append(fb_msg)

    # Tallies across all executed checks (never emitting forbidden 'score' keys)
    check_outcomes: list[bool] = [
        True,  # 1. compliance_ok
        yield_slo_ok,  # 2. sub-1.5s yield & 0 MiB retained VRAM
        bool(governor["thermal_ok"] and governor["vram_reserve_ok"]),  # 3. thermal <=70C & VRAM >=2048 MiB
        bool(cooldown_info["cooldown_policy_ok"] and cooldown_info["resume_allowed_now"]),  # 4. 300s cooldown
        bool(bandwidth_info["bandwidth_cap_ok"]),  # 5. <50 MB/h LoRA-delta-only cap
    ]
    if fleet_info is not None:
        check_outcomes.append(bool(fleet_info["fleet_ok"]))
    elif gw_info is not None:
        check_outcomes.append(bool(gw_info["gateway_ok"]))
    if hb_info is not None:
        check_outcomes.append(bool(hb_info["heartbeat_ok"]))
    if fb_info is not None and fb_info["attempted"] and not fb_degraded_non_blocking:
        check_outcomes.append(bool(fb_info["submitted"]))

    checks_total_count = len(check_outcomes)
    checks_passed_count = sum(1 for item in check_outcomes if item)

    hard_fail = bool(local_hard_fail or (not within_10s))
    fleet_stale = bool(fleet_info is not None and fleet_info["stale_cloud_run_detected"])
    fleet_offline = bool(fleet_info is not None and not fleet_info["fleet_ok"] and not fleet_stale)
    gw_failed = bool(gw_info is not None and not gw_info["gateway_ok"] and fleet_info is None)
    hb_failed = bool(hb_info is not None and not hb_info["heartbeat_ok"])
    fb_failed = bool(
        fb_info is not None
        and fb_info["attempted"]
        and not fb_info["submitted"]
        and not fb_degraded_non_blocking
    )

    failures_count = (
        (1 if not yield_slo_ok else 0)
        + (1 if governor["check_status"] == "fail" else 0)
        + (1 if not cooldown_info["cooldown_policy_ok"] else 0)
        + (1 if not bandwidth_info["bandwidth_cap_ok"] else 0)
        + (1 if not within_10s else 0)
        + (1 if (fleet_stale or fleet_offline or gw_failed) else 0)
        + (1 if hb_failed else 0)
        + (1 if fb_failed else 0)
    )
    warnings_count = len(warnings)

    if hard_fail:
        exit_code = 2
        ready = False
        verdict = "failed_yield_slo" if not yield_slo_ok else "failed_safety_gate"
    elif fleet_stale:
        exit_code = 2
        ready = False
        verdict = "stale_cloud_run"
    elif fleet_offline:
        exit_code = 1
        ready = False
        verdict = "fleet_offline"
    elif gw_failed:
        gw_missing = gw_info is not None and gw_info["status"] == "missing_route"
        exit_code = 2 if gw_missing else 1
        ready = False
        verdict = "gateway_unavailable"
    elif hb_failed:
        hb_st = hb_info.get("status") if hb_info is not None else ""
        exit_code = 2 if hb_st in ("missing_api_key", "invalid_api_key", "auth_failed", "missing_route", "stale_route_404") else 1
        ready = False
        verdict = "opentrainer_heartbeat_failed"
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
        else (
            fleet_info.get("mean_ease_rating")
            if (fleet_info is not None and fleet_info.get("mean_ease_rating") is not None)
            else (gw_info.get("mean_ease_rating") if gw_info is not None else None)
        )
    )
    ease_val = fb_info.get("ease_rating") if fb_info is not None else None
    total_elapsed_s = round(max(0.0, clock_fn() - t_start), 4)

    receipt_body: dict[str, Any] = {
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
        "holiday_routes_deployed_on_cloud_run": (
            fleet_info["holiday_routes_deployed_on_cloud_run"] if fleet_info is not None else None
        ),
        "stale_cloud_run_detected": (
            fleet_info["stale_cloud_run_detected"] if fleet_info is not None else False
        ),
        "online_producers": fleet_info["online_producers"] if fleet_info is not None else None,
        "distinct_physical_gpus": (
            fleet_info["distinct_physical_gpus"] if fleet_info is not None else None
        ),
        "active_trainers": (
            fleet_info["active_trainers"]
            if fleet_info is not None
            else (gw_info.get("active_trainers") if gw_info is not None else None)
        ),
        "target_online_producers": target_online_producers,
        "fleet_badge": fleet_info["fleet_badge"] if fleet_info is not None else None,
        "ces_recorded": bool(fb_info["submitted"]) if fb_info is not None else False,
        "masked_api_key": (
            hb_info.get("masked_api_key")
            if hb_info is not None
            else (mask_api_key(api_key) if api_key else None)
        ),
        "credentials_path": hb_info.get("credentials_path") if hb_info is not None else None,
        "compliance": compliance,
        "preemption_drill": preemption_drill,
        "thermal_vram_governor": governor,
        "cooldown_policy": cooldown_info,
        "bandwidth_cap": bandwidth_info,
        "gateway_check": gw_info,
        "fleet_check": fleet_info,
        "opentrainer_heartbeat": hb_info,
        "feedback": fb_info,
        "warnings": warnings,
        "errors": errors,
    }
    if ces_flush_info is not None:
        receipt_body["ces_spool_flush"] = ces_flush_info
    return _attach_receipt(receipt_body)


def _pr949_route_diagnostic(path: str, group: str, code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if code == 404 and group in ("holiday", "manifest"):
        return (
            f"GET {path} a répondu HTTP 404 (révision Cloud Run pré-PR-#949 ou route manquante)."
            if is_fr
            else f"GET {path} returned HTTP 404 (pre-PR-#949 Cloud Run revision drift or missing route)."
        )
    if code == 0:
        return (
            f"GET {path} injoignable (erreur réseau/TLS)."
            if is_fr
            else f"GET {path} unreachable (network or TLS failure)."
        )
    return (
        f"GET {path} a répondu HTTP {code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {code} (expected HTTP 200)."
    )


def _inspect_models_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str], bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body) or body.get("object", "list") != "list":
        msg = "Contrat `/v1/models` invalide." if is_fr else "Invalid `/v1/models` JSON schema (`object` != 'list')."
        return False, [], False, [msg]
    raw_data = body.get("data")
    if not isinstance(raw_data, list):
        msg = "Liste `data` absente dans `GET /v1/models`." if is_fr else "Missing `data` model list in `GET /v1/models`."
        return False, [], False, [msg]
    model_ids = [
        str(entry.get("id")).strip()
        for entry in raw_data
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry.get("id").strip()
    ]
    issues: list[str] = []
    if not model_ids:
        issues.append("Liste `data` de `GET /v1/models` vide." if is_fr else "`GET /v1/models` returned an empty `data` list.")
    coder_present = ("hivemind/" + "coder") in model_ids
    if not coder_present:
        issues.append(
            "Modèle `hivemind/coder` absent de `GET /v1/models`."
            if is_fr
            else "Required model `hivemind/coder` missing from `GET /v1/models` metadata."
        )
    return len(issues) == 0, model_ids, coder_present, issues


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

    stripped_html = strip_masked_keys_and_nonces(html_text)
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
                "100 % machine personnelle",
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
        issues.append(f"HTML invalide sur {path} : `<!DOCTYPE html>` absent." if is_fr else f"Invalid HTML on {path}: missing `<!DOCTYPE html>`.")
    if nonce_reused:
        issues.append(f"Nonce CSP réutilisé sur {path}." if is_fr else f"Reused CSP nonce detected on {path}.")
    elif not csp_ok:
        issues.append(f"Contrat CSP par requête invalide sur {path}." if is_fr else f"Invalid or missing per-request CSP nonce contract on {path}.")
    if has_unsafe_sink:
        issues.append(f"Sink DOM non sûr détecté sur {path}." if is_fr else f"Unsafe DOM sink detected on {path}.")
    if has_static_fig:
        issues.append(f"Métrique statique détectée dans le HTML de {path}." if is_fr else f"Static telemetry figure detected in HTML on {path}.")
    if not compliance_banner_ok:
        issues.append(
            f"Bannière PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR / AGPL-3.0 banner missing on {path}."
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


def _inspect_pr949_manifest_contract(
    body: Mapping[str, Any] | None,
    *,
    expected_sha: str | None,
    lang: str,
) -> tuple[bool, str, str | None, bool | None, list[str], list[str], list[str]]:
    is_fr = lang == "fr"
    req_scripts = ("noel_gamer_whisper_selftest.py",)
    if not isinstance(body, Mapping):
        msg = "Manifeste `/v1/hivemind/onboarding/manifest` absent." if is_fr else "Manifest `/v1/hivemind/onboarding/manifest` missing or invalid JSON."
        return False, "unknown", None, (False if expected_sha else None), list(PR949_MANIFEST_HOLIDAY_ROUTES), list(req_scripts), [msg]

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
            "Contrat `/v1/hivemind/onboarding/manifest` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/onboarding/manifest` metadata contract."
        )

    raw_routes = body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in PR949_MANIFEST_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(PR949_MANIFEST_HOLIDAY_ROUTES)
    if missing_routes:
        issues.append(
            f"Manifeste `holiday_routes` incomplet : {', '.join(missing_routes)}."
            if is_fr
            else f"Manifest `holiday_routes` missing routes: {', '.join(missing_routes)}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and raw_scripts and all(isinstance(s, str) and s.strip() for s in raw_scripts):
        missing_scripts = [s for s in req_scripts if s not in raw_scripts]
    else:
        missing_scripts = list(req_scripts)
    if missing_scripts:
        issues.append(
            f"Manifeste `downloadable_scripts` incomplet : {', '.join(missing_scripts)}."
            if is_fr
            else f"Manifest `downloadable_scripts` missing: {', '.join(missing_scripts)}."
        )

    live_sha = sanitize_build_sha(body.get("build_sha"))
    live_rev = sanitize_cloud_run_revision(body.get("cloud_run_revision"))
    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = shas_match(live_sha, expected_sha)
        if not sha_match:
            issues.append(
                f"Dérive `build_sha` : actif='{live_sha}' != attendu='{expected_sha}'."
                if is_fr
                else f"Manifest `build_sha` mismatch: live='{live_sha}' != expected='{expected_sha}'."
            )

    return len(issues) == 0, live_sha, live_rev, sha_match, missing_routes, missing_scripts, issues


def _inspect_feedback_summary_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Contrat `/v1/hivemind/onboarding/feedback/summary` invalide." if is_fr else "Invalid `/v1/hivemind/onboarding/feedback/summary` JSON payload."
        return False, None, [msg]
    raw_total = body.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1
    has_ease = "mean_ease_rating" in body or "average_ease_rating" in body
    raw_ease = body.get("mean_ease_rating")
    if raw_ease is None:
        raw_ease = body.get("average_ease_rating")
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
        and _is_nonneg_int(d_raw)
        and _is_nonneg_int(g_raw)
        and total_ok
        and int(d_raw) + int(g_raw) == total_val  # type: ignore[arg-type]
    )
    obj_ok = body.get("object") in _ACCEPTED_FEEDBACK_SUMMARY_OBJECTS
    if not (obj_ok and total_ok and ease_ok and lag_ok and prof_ok):
        msg = "Schéma `/v1/hivemind/onboarding/feedback/summary` non conforme." if is_fr else "Schema mismatch on `/v1/hivemind/onboarding/feedback/summary`."
        return False, None, [msg]
    return True, _sanitize_feedback_summary_doc(body), []


def _inspect_raid_status_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Réponse `/v1/hivemind/raid/status` invalide." if is_fr else "Invalid `/v1/hivemind/raid/status` payload."
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
        msg = "Contrat conformité/télémétrie invalide sur `/v1/hivemind/raid/status`." if is_fr else "Invalid compliance or telemetry contract on `/v1/hivemind/raid/status`."
        return False, [msg]
    return True, []


def _inspect_fleet_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, int, int, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        msg = "Contrat `/v1/hivemind/producers/fleet` invalide." if is_fr else "Invalid `/v1/hivemind/producers/fleet` JSON payload."
        return False, 0, 0, [msg]
    if body.get("object") not in _ACCEPTED_FLEET_OBJECTS:
        msg = "Champ `object` invalide sur `/v1/hivemind/producers/fleet`." if is_fr else "Invalid `object` metadata on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]
    rows = body.get("hiveminders")
    if not isinstance(rows, list):
        msg = "Liste `hiveminders` absente sur `/v1/hivemind/producers/fleet`." if is_fr else "Missing `hiveminders` list on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]
    for row in rows:
        if isinstance(row, Mapping) and "endpoint_url" in row:
            msg = "Fuite `endpoint_url` interdite sur `/v1/hivemind/producers/fleet`." if is_fr else "Forbidden `endpoint_url` leak detected on `/v1/hivemind/producers/fleet`."
            return False, 0, 0, [msg]
    has_live_hm = "live_hiveminders" in body
    if has_live_hm and not _is_nonneg_int(body.get("live_hiveminders")):
        msg = "Champ `live_hiveminders` invalide sur `/v1/hivemind/producers/fleet`." if is_fr else "Invalid non-integer `live_hiveminders` on `/v1/hivemind/producers/fleet`."
        return False, 0, 0, [msg]

    live_hm_int = int(body["live_hiveminders"]) if has_live_hm else 0
    row_online = sum(
        1
        for row in rows
        if isinstance(row, Mapping) and str(row.get("status", "")).strip().lower() in LIVE_PRODUCER_STATUSES
    )
    online_producers = max(live_hm_int, row_online)
    raw_gpu = body.get("distinct_physical_gpus")
    distinct_gpus = int(raw_gpu) if _is_nonneg_int(raw_gpu) else 0
    return True, online_producers, distinct_gpus, []


def _inspect_arena_summary_contract(
    body: Mapping[str, Any] | None,
    *,
    lang: str,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    is_fr = lang == "fr"
    if not isinstance(body, Mapping) or body.get("object") != ARENA_SUMMARY_OBJECT:
        msg = "Contrat `/v1/hivemind/arena/summary` invalide." if is_fr else "Arena summary `/v1/hivemind/arena/summary` invalid `object` metadata."
        return False, None, [msg]
    if _has_forbidden_score_key(body):
        return False, None, ["Arena summary `/v1/hivemind/arena/summary` contains forbidden key."]

    tv = body.get("total_votes")
    hw = body.get("holiday_wins")
    bw = body.get("baseline_wins")
    ti = body.get("ties")
    counts_valid = all(_is_nonneg_int(x) for x in (tv, hw, bw, ti))
    conserved = bool(counts_valid and int(hw) + int(bw) + int(ti) == int(tv))  # type: ignore[arg-type]

    dec_ok = True
    if "decisive_votes" in body:
        dv = body.get("decisive_votes")
        dec_ok = bool(_is_nonneg_int(dv) and counts_valid and int(dv) == int(hw) + int(bw))  # type: ignore[arg-type]

    raw_wlb = body.get("wilson_95_lower_bound")
    wlb_ok = bool(
        "wilson_95_lower_bound" in body
        and not isinstance(raw_wlb, bool)
        and isinstance(raw_wlb, (int, float))
        and math.isfinite(float(raw_wlb))
        and 0.0 <= float(raw_wlb) <= 1.0
    )
    decisive = (int(hw) + int(bw)) if counts_valid else 0  # type: ignore[arg-type]
    if wlb_ok and counts_valid and decisive == 0 and float(raw_wlb) != 0.0:  # type: ignore[arg-type]
        wlb_ok = False

    win_ratio_ok = True
    if "win_ratio" in body:
        raw_wr = body.get("win_ratio")
        if counts_valid and decisive == 0:
            win_ratio_ok = raw_wr is None
        else:
            win_ratio_ok = bool(
                not isinstance(raw_wr, bool)
                and isinstance(raw_wr, (int, float))
                and math.isfinite(float(raw_wr))
                and 0.0 <= float(raw_wr) <= 1.0
            )

    by_task_ok = True
    raw_tasks = body.get("by_task")
    if isinstance(raw_tasks, Mapping):
        for t_doc in raw_tasks.values():
            if not isinstance(t_doc, Mapping):
                by_task_ok = False
                break
            t_tv, t_hw, t_bw, t_ti = (
                t_doc.get("total_votes"),
                t_doc.get("holiday_wins"),
                t_doc.get("baseline_wins"),
                t_doc.get("ties"),
            )
            t_wlb = t_doc.get("wilson_95_lower_bound")
            if (
                not all(_is_nonneg_int(x) for x in (t_tv, t_hw, t_bw, t_ti))
                or int(t_hw) + int(t_bw) + int(t_ti) != int(t_tv)  # type: ignore[arg-type]
                or "wilson_95_lower_bound" not in t_doc
                or isinstance(t_wlb, bool)
                or not isinstance(t_wlb, (int, float))
                or not math.isfinite(float(t_wlb))
                or not (0.0 <= float(t_wlb) <= 1.0)
            ):
                by_task_ok = False
                break

    if not (counts_valid and conserved and dec_ok and wlb_ok and win_ratio_ok and by_task_ok):
        msg = (
            "Conservation des votes ou borne Wilson 95 % invalide sur `/v1/hivemind/arena/summary`."
            if is_fr
            else "Vote conservation (`holiday_wins + baseline_wins + ties == total_votes`) or Wilson 95% lower-bound invalid on `/v1/hivemind/arena/summary`."
        )
        return False, None, [msg]

    raw_wr_val = body.get("win_ratio")
    sanitized: dict[str, Any] = {
        "object": ARENA_SUMMARY_OBJECT,
        "total_votes": int(tv),  # type: ignore[arg-type]
        "decisive_votes": decisive,
        "holiday_wins": int(hw),  # type: ignore[arg-type]
        "baseline_wins": int(bw),  # type: ignore[arg-type]
        "ties": int(ti),  # type: ignore[arg-type]
        "win_ratio": (
            round(float(raw_wr_val), 4)
            if isinstance(raw_wr_val, (int, float)) and not isinstance(raw_wr_val, bool)
            else None
        ),
        "wilson_95_lower_bound": round(float(raw_wlb), 4),  # type: ignore[arg-type]
    }
    return True, sanitized, []


def _run_deploy_sh_dry_run(
    *,
    repo_root: Path | str | None = None,
    expected_sha: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute list-argv `bash deploy/gcp_openrouter/deploy.sh --dry-run` without invoking `gcloud`."""
    is_fr = lang == "fr"
    base_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    deploy_sh = base_root / "deploy" / "gcp_openrouter" / "deploy.sh"
    display_path = "deploy/gcp_openrouter/deploy.sh"
    if not deploy_sh.is_file():
        return {
            "executed": False,
            "ok": False,
            "returncode": 127,
            "deploy_script_path": display_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": (
                f"Script `{display_path}` introuvable pour `--dry-run`."
                if is_fr
                else f"Deploy script `{display_path}` missing for `--dry-run`."
            ),
        }

    dry_env: dict[str, str] = dict(environ) if environ is not None else dict(os.environ)
    if "PATH" not in dry_env:
        dry_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    dry_env["HIVEMIND_DEPLOY_DRY_RUN"] = "1"
    if expected_sha is not None:
        dry_env["HIVEMIND_BUILD_SHA"] = expected_sha

    argv = ["bash", str(deploy_sh), "--dry-run"]
    if Path(argv[0]).name.lower().startswith("gcloud"):
        raise SelftestError("Direct gcloud invocation is forbidden in noel_gamer_whisper_selftest.")
    try:
        sig = inspect.signature(runner)
        if "env" in sig.parameters:
            rc, stdout = runner(argv, 15.0, env=dry_env)  # type: ignore[call-arg]
        else:
            env_prefix = ["env", "HIVEMIND_DEPLOY_DRY_RUN=1"]
            if expected_sha is not None:
                env_prefix.append(f"HIVEMIND_BUILD_SHA={expected_sha}")
            rc, stdout = runner([*env_prefix, *argv], 15.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "executed": True,
            "ok": False,
            "returncode": 1,
            "deploy_script_path": display_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": sanitize_text_for_display(f"{display_path} --dry-run failed ({type(exc).__name__})."),
        }

    kv: dict[str, str] = {}
    for line in str(stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    finops_line = kv.get("finops_flags", "")
    env_vars_line = kv.get("update_env_vars", "")
    raw_dry_sha = kv.get("build_sha", "")
    dry_sha = sanitize_build_sha(raw_dry_sha)
    sha_propagated = bool(
        raw_dry_sha
        and raw_dry_sha.strip().lower() == dry_sha
        and f"HIVEMIND_BUILD_SHA={raw_dry_sha}" in env_vars_line
        and (expected_sha is None or shas_match(dry_sha, expected_sha))
    )
    checks = {
        "returncode_zero": int(rc) == 0,
        "dry_run_marker": kv.get("HIVEMIND_CLOUD_RUN_DEPLOY_DRY_RUN") == "1",
        "gcloud_invoked_zero": kv.get("gcloud_invoked") == "0",
        "finops_flags_present": all(
            tok in finops_line
            for tok in ("--min-instances 0", "--max-instances 1", "--no-cpu-throttling", "--concurrency 80")
        ),
        "replica_url_gcs": kv.get("replica_url", "").startswith("gcs://"),
        "secret_manager_mounted": "HIVEMIND_ROUTER_SECRET=" in kv.get("secrets", ""),
        "zero_grants_env": "HIVEMIND_STARTER_GRANT_HMT=0" in env_vars_line,
        "build_sha_propagated": sha_propagated,
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "ok": ok,
        "returncode": int(rc),
        "deploy_script_path": display_path,
        "gcloud_invoked": kv.get("gcloud_invoked") != "0",
        "dry_run_build_sha": dry_sha,
        "checks": checks,
        "diagnostic": (
            ""
            if ok
            else (
                f"Échec vérification `{display_path} --dry-run`."
                if is_fr
                else f"Subprocess verification `{display_path} --dry-run` failed."
            )
        ),
    }


def _validate_pr949_options(
    *,
    target_online_producers: Any,
    min_ease_rating: Any,
    min_zero_lag_ratio: Any,
) -> tuple[int, float, float]:
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or not (1 <= target_online_producers <= 1024)
    ):
        raise SelftestError("target_online_producers must be an integer between 1 and 1024.")
    if isinstance(min_ease_rating, bool) or not isinstance(min_ease_rating, (int, float)):
        raise SelftestError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    ease_f = float(min_ease_rating)
    if not math.isfinite(ease_f) or not (1.0 <= ease_f <= 5.0):
        raise SelftestError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    if isinstance(min_zero_lag_ratio, bool) or not isinstance(min_zero_lag_ratio, (int, float)):
        raise SelftestError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    lag_f = float(min_zero_lag_ratio)
    if not math.isfinite(lag_f) or not (0.0 <= lag_f <= 1.0):
        raise SelftestError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    return int(target_online_producers), ease_f, lag_f


def verify_pr949_gamer_quorum_and_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = True,
    verify_idempotency_replay: bool = False,
    execute_deploy_dry_run: bool = False,
    repo_root: Path | str | None = None,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run zero-inference PR #949 4-HTML parity, `/v1/models`, live quorum, arena summary, replay & dry-run gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    resolved_home = home if home is not None else Path.home()
    target_n, min_ease_f, min_lag_f = _validate_pr949_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
    )
    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = (
        validate_expected_sha(expected_sha)
        if (expected_sha is not None and str(expected_sha).strip())
        else None
    )

    comp = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        check_cwd_path=check_cwd_path,
        lang=lang_code,
    )
    if not comp["compliance_ok"]:
        return _attach_receipt(
            _sanitize_structure(
                {
                    "schema": SCHEMA,
                    "object": "hivemind.noel_gamer_whisper.pr949_quorum_parity_receipt",
                    "campaign_id": CAMPAIGN_ID,
                    "gateway": clean_gw,
                    "lang": lang_code,
                    "compliance_ok": False,
                    "corporate_indicators": comp["corporate_indicators"],
                    "ready": False,
                    "verdict": "refused_compliance",
                    "parity_ok": False,
                    "gate_ok": False,
                    "quorum_gate_ok": False,
                    "ces_gate_ok": False,
                    "redeploy_recommended": False,
                    "cloud_run_lags_main": False,
                    "stale_404_drift": False,
                    "routes_passed_count": 0,
                    "routes_total_count": len(PR949_PARITY_ROUTES) + (1 if check_arena_summary else 0),
                    "http_statuses": {},
                    "missing_routes": [],
                    "failed_routes": [],
                    "models_ok": False,
                    "live_model_ids": [],
                    "hivemind_coder_present": False,
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
                    "ces_slo_ok": False,
                    "zero_lag_slo_ok": False,
                    "min_ease_rating": min_ease_f,
                    "min_zero_lag_ratio": min_lag_f,
                    "raid_status_ok": False,
                    "active_trainers": 0,
                    "fleet_ok": False,
                    "online_producers": 0,
                    "target_online_producers": target_n,
                    "producer_quorum_met": False,
                    "awaiting_producer_quorum": True,
                    "distinct_physical_gpus": 0,
                    "arena_summary_checked": bool(check_arena_summary),
                    "arena_summary_ok": False if check_arena_summary else None,
                    "arena_summary": None,
                    "idempotency_replay_checked": bool(verify_idempotency_replay),
                    "idempotent_dedup_ok": None,
                    "idempotency_key": None,
                    "deploy_dry_run_executed": bool(execute_deploy_dry_run),
                    "deploy_dry_run": None,
                    "zero_inference_verified": True,
                    "network_calls": 0,
                    "subprocess_calls": 0,
                    "file_writes": 0,
                    "diagnostics": [comp["summary"]],
                    "routes": [],
                    "compliance": comp,
                    "financial_promise": FINANCIAL_PROMISE,
                    "exit_code": 2,
                },
                home=resolved_home,
            )
        )

    eff_raw_fn: Callable[..., Any] = raw_http_fn if raw_http_fn is not default_http_raw else (
        http_fn if http_fn is not default_http_json else default_http_raw
    )
    eff_json_fn: Callable[..., Any] = http_fn if http_fn is not default_http_json else (
        raw_http_fn if raw_http_fn is not default_http_raw else default_http_raw
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

    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    sha_match: bool | None = None if clean_expected_sha is None else False
    manifest_missing_routes: list[str] = []
    manifest_missing_scripts: list[str] = []
    feedback_summary_ok = False
    feedback_summary: dict[str, Any] | None = None
    raw_feedback_doc: Mapping[str, Any] | None = None
    raid_status_ok = False
    active_trainers = 0
    fleet_ok = False
    online_producers = 0
    distinct_gpus = 0
    network_calls = 0
    subprocess_calls = 0

    for path, kind, group in PR949_PARITY_ROUTES:
        call_fn = eff_raw_fn if kind == "html" else eff_json_fn
        code, json_doc, text_body, resp_headers = _invoke_parity_call(call_fn, "GET", f"{clean_gw}{path}")
        network_calls += 1
        http_statuses[path] = code
        ok = code == 200
        route_issues: list[str] = []

        if not ok:
            if code == 404:
                missing_routes.append(path)
            route_issues.append(_pr949_route_diagnostic(path, group, code, lang=lang_code))
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
        elif path == "/v1/models":
            models_ok, model_ids, coder_present, m_issues = _inspect_models_contract(json_doc, lang=lang_code)
            if not models_ok:
                ok = False
                route_issues.extend(m_issues)
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
                man_issues,
            ) = _inspect_pr949_manifest_contract(json_doc, expected_sha=clean_expected_sha, lang=lang_code)
            if not manifest_valid:
                ok = False
                route_issues.extend(man_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_ok, feedback_summary, fb_issues = _inspect_feedback_summary_contract(
                json_doc,
                lang=lang_code,
            )
            if feedback_summary_ok:
                raw_feedback_doc = json_doc
            else:
                ok = False
                route_issues.extend(fb_issues)
        elif path == "/v1/hivemind/raid/status":
            raid_status_ok, r_issues = _inspect_raid_status_contract(json_doc, lang=lang_code)
            if not raid_status_ok:
                ok = False
                route_issues.extend(r_issues)
            elif isinstance(json_doc, Mapping) and isinstance(json_doc.get("swarm"), Mapping):
                raw_tr = json_doc["swarm"].get("active_trainers")
                if _is_nonneg_int(raw_tr):
                    active_trainers = int(raw_tr)
        elif path == "/v1/hivemind/producers/fleet":
            fleet_ok, online_producers, distinct_gpus, f_issues = _inspect_fleet_contract(json_doc, lang=lang_code)
            if not fleet_ok:
                ok = False
                route_issues.extend(f_issues)

        if not ok:
            failed_routes.append(path)
            for item in route_issues:
                diagnostics.append(sanitize_text_for_display(item, home=resolved_home))

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "kind": kind,
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": sanitize_text_for_display(
                    route_issues[0] if route_issues else "", home=resolved_home
                ),
            }
        )

    arena_summary_ok: bool | None = None
    arena_summary_doc: dict[str, Any] | None = None
    if check_arena_summary:
        a_path = "/v1/hivemind/arena/summary"
        a_code, a_json, _, _ = _invoke_parity_call(eff_json_fn, "GET", f"{clean_gw}{a_path}")
        network_calls += 1
        http_statuses[a_path] = a_code
        a_issues: list[str] = []
        if a_code != 200:
            arena_summary_ok = False
            if a_code == 404:
                missing_routes.append(a_path)
            a_issues.append(_pr949_route_diagnostic(a_path, "holiday", a_code, lang=lang_code))
        else:
            arena_summary_ok, arena_summary_doc, a_issues = _inspect_arena_summary_contract(a_json, lang=lang_code)
        if not arena_summary_ok:
            failed_routes.append(a_path)
            for item in a_issues:
                diagnostics.append(sanitize_text_for_display(item, home=resolved_home))
        route_rows.append(
            {
                "path": a_path,
                "method": "GET",
                "kind": "json",
                "group": "holiday",
                "http_status": a_code,
                "ok": bool(arena_summary_ok),
                "diagnostic": sanitize_text_for_display(a_issues[0] if a_issues else "", home=resolved_home),
            }
        )

    idempotent_dedup_ok: bool | None = None
    idem_key: str | None = None
    if verify_idempotency_replay:
        pre_total = (
            int(feedback_summary["total_responses"])
            if (feedback_summary_ok and feedback_summary is not None)
            else None
        )
        pre_gamer = (
            int(feedback_summary["by_profile"]["gamer"])
            if (feedback_summary_ok and feedback_summary is not None)
            else None
        )
        fb_payload = _normalize_ces_spool_payload(
            profile="gamer",
            ease_rating=5,
            zero_game_lag=True,
            platform_tag=_platform_tag(),
            home=resolved_home,
        )
        idem_key = build_ces_idempotency_key(
            gateway=clean_gw,
            ease_rating=5,
            zero_game_lag=True,
            platform_tag=str(fb_payload["platform"]),
            client_ref=f"pr949-{secrets.token_hex(8)}",
        )
        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
        replay_headers = {"Idempotency-Key": idem_key}

        p1_code, p1_json, _, _ = _invoke_parity_call(eff_json_fn, "POST", post_url, payload=fb_payload, headers=replay_headers)
        network_calls += 1
        http_statuses["POST /v1/hivemind/onboarding/feedback"] = p1_code
        p1_ok, p1_sum, _ = (
            _inspect_feedback_summary_contract(p1_json, lang=lang_code)
            if p1_code in (200, 201)
            else (False, None, [])
        )

        p2_code, p2_json, _, _ = _invoke_parity_call(eff_json_fn, "POST", post_url, payload=fb_payload, headers=replay_headers)
        network_calls += 1
        http_statuses["POST /v1/hivemind/onboarding/feedback#replay"] = p2_code
        p2_ok, p2_sum, _ = (
            _inspect_feedback_summary_contract(p2_json, lang=lang_code)
            if p2_code in (200, 201)
            else (False, None, [])
        )

        ref_code, ref_json, _, _ = _invoke_parity_call(
            eff_json_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/onboarding/feedback/summary",
        )
        network_calls += 1
        http_statuses["/v1/hivemind/onboarding/feedback/summary#after_replay"] = ref_code
        ref_ok, ref_sum, _ = (
            _inspect_feedback_summary_contract(ref_json, lang=lang_code)
            if ref_code == 200
            else (False, None, [])
        )

        p1_inc_ok = bool(
            p1_ok
            and p1_sum is not None
            and (
                pre_total is None
                or (
                    int(p1_sum["total_responses"]) >= int(pre_total) + 1
                    and int(p1_sum["by_profile"]["gamer"]) >= int(pre_gamer or 0) + 1
                )
            )
        )
        replay_eq_ok = bool(
            p2_ok
            and p2_sum is not None
            and p1_sum is not None
            and ref_ok
            and ref_sum is not None
            and p2_sum["total_responses"] == p1_sum["total_responses"]
            and p2_sum["by_profile"] == p1_sum["by_profile"]
            and ref_sum["total_responses"] == p1_sum["total_responses"]
            and ref_sum["by_profile"] == p1_sum["by_profile"]
        )
        idempotent_dedup_ok = bool(p1_inc_ok and replay_eq_ok)
        if idempotent_dedup_ok and ref_sum is not None and isinstance(ref_json, Mapping):
            feedback_summary_ok = True
            feedback_summary = ref_sum
            raw_feedback_doc = ref_json
        else:
            diagnostics.append(
                "Échec vérification replay idempotent `Idempotency-Key` sur `/v1/hivemind/onboarding/feedback`."
                if is_fr
                else "Idempotency-Key duplicate-replay verification failed on `/v1/hivemind/onboarding/feedback`."
            )

    producer_quorum_met = bool(fleet_ok and online_producers >= target_n)
    awaiting_producer_quorum = not producer_quorum_met
    quorum_gate_ok = bool(producer_quorum_met or not require_live_quorum)
    if require_live_quorum and not producer_quorum_met:
        diagnostics.append(
            f"Quorum producteurs en ligne insuffisant ({online_producers} < {target_n})."
            if is_fr
            else f"Live producer quorum not met ({online_producers} < {target_n})."
        )

    ces_slo_ok: bool | None
    zero_lag_slo_ok: bool | None
    if not feedback_summary_ok or feedback_summary is None or raw_feedback_doc is None:
        ces_slo_ok = False
        zero_lag_slo_ok = False
        ces_gate_ok = False
    elif int(feedback_summary["total_responses"]) > 0:
        raw_ease = raw_feedback_doc.get("mean_ease_rating")
        if raw_ease is None:
            raw_ease = raw_feedback_doc.get("average_ease_rating")
        raw_lag = raw_feedback_doc.get("zero_game_lag_ratio")
        ces_slo_ok = bool(
            raw_ease is not None and Fraction(str(float(raw_ease))) >= Fraction(str(min_ease_f))
        )
        zero_lag_slo_ok = bool(
            raw_lag is not None and Fraction(str(float(raw_lag))) >= Fraction(str(min_lag_f))
        )
        ces_gate_ok = bool(not require_ces_slo or (ces_slo_ok and zero_lag_slo_ok))
        if require_ces_slo and not ces_gate_ok:
            diagnostics.append(
                f"SLO CES/zéro-lag non respecté (ease={raw_ease}, lag={raw_lag})."
                if is_fr
                else f"CES/zero-lag SLO breached (ease={raw_ease} < {min_ease_f} or lag={raw_lag} < {min_lag_f})."
            )
    else:
        ces_slo_ok = None
        zero_lag_slo_ok = None
        ces_gate_ok = True

    deploy_dry_run_res: dict[str, Any] | None = None
    if execute_deploy_dry_run:
        deploy_dry_run_res = _run_deploy_sh_dry_run(
            repo_root=repo_root,
            expected_sha=clean_expected_sha,
            runner=runner,
            environ=environ,
            lang=lang_code,
        )
        if deploy_dry_run_res.get("executed"):
            subprocess_calls += 1
        if not deploy_dry_run_res.get("ok"):
            diagnostics.append(str(deploy_dry_run_res.get("diagnostic") or "deploy.sh --dry-run failed."))

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    html_count = len(PR949_HTML_ROUTES)
    unique_csp_nonces_ok = bool(
        html_all_ok
        and len(html_contracts) == html_count
        and len(seen_nonces) == html_count
        and not any(info.get("nonce_reused") for info in html_contracts.values())
    )
    sha_drift = bool(clean_expected_sha is not None and sha_match is not True)
    stale_404_drift = bool(missing_routes or manifest_missing_routes or manifest_missing_scripts)
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and (stale_404_drift or sha_drift))

    parity_ok = bool(
        passed_count == total_count
        and models_ok
        and coder_present
        and manifest_valid
        and unique_csp_nonces_ok
        and html_dom_clean
        and compliance_banners_ok
        and feedback_summary_ok
        and raid_status_ok
        and fleet_ok
        and not sha_drift
        and (not check_arena_summary or arena_summary_ok is True)
        and (not verify_idempotency_replay or idempotent_dedup_ok is True)
        and (not execute_deploy_dry_run or (deploy_dry_run_res is not None and deploy_dry_run_res["ok"] is True))
    )
    gate_ok = bool(parity_ok and quorum_gate_ok and ces_gate_ok)

    report = _sanitize_structure(
        {
            "schema": SCHEMA,
            "object": "hivemind.noel_gamer_whisper.pr949_quorum_parity_receipt",
            "campaign_id": CAMPAIGN_ID,
            "gateway": clean_gw,
            "lang": lang_code,
            "compliance_ok": True,
            "ready": gate_ok,
            "verdict": "ready" if gate_ok else "not_ready",
            "parity_ok": parity_ok,
            "gate_ok": gate_ok,
            "quorum_gate_ok": quorum_gate_ok,
            "ces_gate_ok": ces_gate_ok,
            "redeploy_recommended": bool(not parity_ok or cloud_run_lags_main or sha_drift),
            "cloud_run_lags_main": cloud_run_lags_main,
            "stale_404_drift": stale_404_drift,
            "routes_passed_count": passed_count,
            "routes_total_count": total_count,
            "http_statuses": http_statuses,
            "missing_routes": missing_routes,
            "failed_routes": failed_routes,
            "models_ok": models_ok,
            "live_model_ids": model_ids,
            "hivemind_coder_present": coder_present,
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
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "min_ease_rating": min_ease_f,
            "min_zero_lag_ratio": min_lag_f,
            "raid_status_ok": raid_status_ok,
            "active_trainers": active_trainers,
            "fleet_ok": fleet_ok,
            "online_producers": online_producers,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "distinct_physical_gpus": distinct_gpus,
            "arena_summary_checked": bool(check_arena_summary),
            "arena_summary_ok": arena_summary_ok,
            "arena_summary": arena_summary_doc,
            "idempotency_replay_checked": bool(verify_idempotency_replay),
            "idempotent_dedup_ok": idempotent_dedup_ok,
            "idempotency_key": idem_key,
            "deploy_dry_run_executed": bool(execute_deploy_dry_run),
            "deploy_dry_run": deploy_dry_run_res,
            "zero_inference_verified": True,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "file_writes": 0,
            "diagnostics": diagnostics,
            "routes": route_rows,
            "compliance": comp,
            "financial_promise": FINANCIAL_PROMISE,
            "exit_code": 0 if gate_ok else 2,
        },
        home=resolved_home,
    )
    return _attach_receipt(report)


def render_pr949_quorum_report(report: Mapping[str, Any]) -> str:
    """Format an FR/EN human-readable CLI report for `verify_pr949_gamer_quorum_and_parity`."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)
    header = (
        "HIVEMIND NOËL — PARITÉ PR #949 & QUORUM GAMER-WHISPER (ZÉRO INFÉRENCE • 0,00 EUR)"
        if is_fr
        else "HIVEMIND HOLIDAY — PR #949 PARITY & GAMER-WHISPER QUORUM GATE (ZERO INFERENCE • 0.00 EUR)"
    )
    lines = [
        "==========================================================================",
        f"  {header}",
        "==========================================================================",
        f"  [Compliance] {bilingual_notice}",
        f"  Gateway      : {report.get('gateway')}",
        f"  Compliance OK: {report.get('compliance_ok')}",
        f"  Routes       : {report.get('routes_passed_count')}/{report.get('routes_total_count')} "
        f"(parity_ok={report.get('parity_ok')}, gate_ok={report.get('gate_ok')})",
        f"  Quorum       : online={report.get('online_producers')}/{report.get('target_online_producers')} "
        f"(awaiting={report.get('awaiting_producer_quorum')}, gpus={report.get('distinct_physical_gpus')})",
        f"  Arena/Replay : arena_ok={report.get('arena_summary_ok')} | idempotent_ok={report.get('idempotent_dedup_ok')}",
    ]
    for row in report.get("routes") or ():
        tag = "[OK]" if row.get("ok") else "[FAIL]"
        lines.append(f"  {tag} GET {row.get('path'):<40} -> HTTP {row.get('http_status')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt      : {report.get('receipt_sha256', '')}")
    return _assert_safe_output("\n".join(lines))


def render_ces_flush_report(flush_res: Mapping[str, Any]) -> str:
    """Format FR/EN human-readable CLI report for `GamerCesRetrySpool.flush` / `.clean`."""
    lang_code = normalize_lang(str(flush_res.get("lang", "en")))
    is_fr = lang_code == "fr"
    header = (
        "HIVEMIND GAMER-WHISPER — FILE CES (0,00 EUR)"
        if is_fr
        else "HIVEMIND GAMER-WHISPER — CES RETRY SPOOL (0.00 EUR)"
    )
    lines = [
        "==========================================================================",
        f"  {header}",
        "==========================================================================",
        f"  Gateway      : {flush_res.get('gateway', 'local-only')}",
        f"  Spool        : {flush_res.get('spool_file')} (exists={flush_res.get('spool_exists')})",
        f"  Compliance OK: {flush_res.get('compliance_ok', True)}",
    ]
    if "delivered_count" in flush_res:
        lines.append(
            f"  Counts       : delivered={flush_res.get('delivered_count')} "
            f"pending={flush_res.get('pending_count')} quarantined={flush_res.get('total_quarantined_count')}"
        )
    else:
        lines.append(f"  Removed      : {flush_res.get('removed')}")
    for diag in flush_res.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt      : {flush_res.get('receipt_sha256', '')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render FR/EN GitHub-flavored Markdown Gamer-Whisper self-test, PR #949, or spool report."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)
    obj_name = str(report.get("object") or "")

    if obj_name == "hivemind.noel_gamer_whisper.pr949_quorum_parity_receipt":
        title = (
            "# Autotest Gamer-Whisper — Parité PR #949 & Quorum en Direct"
            if is_fr
            else "# Gamer-Whisper Self-Test — PR #949 Parity & Live Quorum Gate"
        )
        lines = [
            title,
            "",
            f"> **Compliance / Conformité**: {bilingual_notice}",
            "",
            f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway')}`",
            f"- **Compliance OK**: `{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
            f"- **{'Parité / Porte' if is_fr else 'Parity / Gate'}**: `parity_ok={report.get('parity_ok')}`, `gate_ok={report.get('gate_ok')}`, `exit_code={report.get('exit_code')}`",
            f"- **Quorum**: `online_producers={report.get('online_producers')}/{report.get('target_online_producers')}`, `awaiting_producer_quorum={report.get('awaiting_producer_quorum')}`",
            f"- **Arena / Replay**: `arena_summary_ok={report.get('arena_summary_ok')}`, `idempotent_dedup_ok={report.get('idempotent_dedup_ok')}`",
            f"- **SHA-256 Receipt**: `{report.get('receipt_sha256', '')}`",
        ]
        if report.get("diagnostics"):
            lines.extend(["", "## Diagnostics", *[f"- {d}" for d in report["diagnostics"]]])
        return _assert_safe_output("\n".join(lines))

    if obj_name in ("hivemind.gamer_whisper.ces_spool_flush", "hivemind.gamer_whisper.ces_spool_clean"):
        title = (
            "# Gamer-Whisper — File de Réessai CES"
            if is_fr
            else "# Gamer-Whisper — CES Retry Spool"
        )
        lines = [
            title,
            "",
            f"> **Compliance / Conformité**: {bilingual_notice}",
            "",
            f"- **Spool**: `{report.get('spool_file')}`",
            f"- **Status**: `ok={report.get('flush_ok', True)}`, `compliance_ok={report.get('compliance_ok', True)}`",
            f"- **SHA-256 Receipt**: `{report.get('receipt_sha256', '')}`",
        ]
        if report.get("diagnostics"):
            lines.extend(["", "## Diagnostics", *[f"- {d}" for d in report["diagnostics"]]])
        return _assert_safe_output("\n".join(lines))

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

    fl = report.get("fleet_check")
    if isinstance(fl, Mapping):
        lines.extend(
            [
                "",
                "## Badge Flotte en Direct & Progression Producteurs"
                if is_fr
                else "## Live Fleet Badge & Producer Progress",
                "",
                f"- **Fleet Badge**: `{fl.get('fleet_badge')}` | **Online Producers**: `{fl.get('online_producers')}/{fl.get('target_online_producers')}` | **Distinct GPUs**: `{fl.get('distinct_physical_gpus')}` | **Active Trainers**: `{fl.get('active_trainers')}`",
                f"- **Holiday Routes Deployed**: `{fl.get('holiday_routes_deployed_on_cloud_run')}` | **Stale Cloud Run**: `{fl.get('stale_cloud_run_detected')}`",
                f"- {fl.get('message')}",
            ]
        )

    hb = report.get("opentrainer_heartbeat")
    if isinstance(hb, Mapping):
        lines.extend(
            [
                "",
                "## Heartbeat OpenTrainer Sortant (Relais HTTPS)"
                if is_fr
                else "## Outbound OpenTrainer Heartbeat (HTTPS Relay)",
                "",
                f"- **{hb.get('status')}** (`network_mode={hb.get('network_mode')}`, `campaign={hb.get('active_campaign')}`, `key={hb.get('masked_api_key')}`): {hb.get('message')}",
            ]
        )

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
    obj_name = str(report.get("object") or "")
    if obj_name == "hivemind.noel_gamer_whisper.pr949_quorum_parity_receipt":
        return render_pr949_quorum_report(report)
    if obj_name in ("hivemind.gamer_whisper.ces_spool_flush", "hivemind.gamer_whisper.ces_spool_clean"):
        return render_ces_flush_report(report)
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

    fl = report.get("fleet_check")
    if isinstance(fl, Mapping):
        fleet_lbl = "Badge flotte         " if is_fr else "Fleet badge          "
        routes_lbl = "Routes Noël Cloud Run" if is_fr else "Holiday routes live  "
        lines.append(
            f"  {fleet_lbl}: [{fl.get('fleet_badge')}] online_producers={fl.get('online_producers')}/{fl.get('target_online_producers')} "
            f"(gpus={fl.get('distinct_physical_gpus')}, trainers={fl.get('active_trainers')})"
        )
        lines.append(
            f"  {routes_lbl}: {fl.get('holiday_routes_deployed_on_cloud_run')} "
            f"(stale_cloud_run={fl.get('stale_cloud_run_detected')})"
        )

    hb = report.get("opentrainer_heartbeat")
    if isinstance(hb, Mapping):
        lines.append(
            f"  OpenTrainer heartbeat: {hb.get('status')} (mode={hb.get('network_mode')}, key={hb.get('masked_api_key')})"
        )

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
        "--noel-verify",
        action="store_true",
        help="Verify holiday manifest, raid status, and live producer fleet ONLINE badge (zero inference).",
    )
    parser.add_argument(
        "--verify-fleet",
        action="store_true",
        help="Alias for --noel-verify probing live fleet registration and ONLINE badge.",
    )
    parser.add_argument(
        "--send-opentrainer-heartbeat",
        action="store_true",
        help="Verify GET /v1/hivemind/me and send outbound POST /v1/hivemind/opentrainer/heartbeat.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional HiveMind API key (hm-... or hm_live_...); never logged raw.",
    )
    parser.add_argument(
        "--credentials",
        "--credentials-file",
        dest="credentials",
        default=None,
        help="Optional path to ~/.hivemind/credentials.json.",
    )
    parser.add_argument(
        "--host-label",
        default=DEFAULT_HOST_LABEL,
        help="Personal PC host label sent in OpenTrainer heartbeat (default: personal-pc).",
    )
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Target live online producers for ONLINE fleet badge (default: 8).",
    )
    parser.add_argument(
        "--non-blocking-feedback",
        action="store_true",
        help="Degrade gracefully (ces_recorded=False warning) when CES feedback hits 404/429/5xx/timeout.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Optional expected Cloud Run build_sha checked against /v1/hivemind/onboarding/manifest.",
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
    parser.add_argument(
        "--cwd",
        default=None,
        help="Workspace directory inspected locally for corporate markers and git remotes.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Atomically persist rendered output (POSIX 0600); writes 0 files if compliance fails.",
    )
    parser.add_argument(
        "--ces-queue-file",
        default=None,
        help="Optional path to local POSIX-0600 CES retry spool JSON file.",
    )
    parser.add_argument(
        "--flush-ces-queue",
        action="store_true",
        help="Flush spooled zero-PII CES events idempotently with Idempotency-Key headers.",
    )
    parser.add_argument(
        "--clean-ces-queue",
        action="store_true",
        help="Remove local CES retry spool JSON file without network requests.",
    )
    parser.add_argument(
        "--pr949-quorum-gate",
        action="store_true",
        help="Verify zero-inference PR #949 4-HTML parity, /v1/models, live quorum, arena summary, and optional replay/dry-run.",
    )
    parser.add_argument(
        "--min-ease-rating",
        type=float,
        default=DEFAULT_MIN_EASE_RATING,
        help="Minimum required mean CES ease rating (default: 4.2).",
    )
    parser.add_argument(
        "--min-zero-lag-ratio",
        type=float,
        default=DEFAULT_MIN_ZERO_LAG_RATIO,
        help="Minimum required zero-game-lag ratio (default: 0.95).",
    )
    parser.add_argument(
        "--require-live-quorum",
        action="store_true",
        help="Fail closed (exit code 2) in --pr949-quorum-gate when online_producers < target.",
    )
    parser.add_argument(
        "--require-ces-slo",
        action="store_true",
        help="Fail closed (exit code 2) in --pr949-quorum-gate if CES or zero-lag SLO is breached.",
    )
    parser.add_argument(
        "--check-arena-summary",
        dest="check_arena_summary",
        action="store_true",
        default=True,
        help="Inspect GET /v1/hivemind/arena/summary Wilson 95%% LB & vote conservation in --pr949-quorum-gate.",
    )
    parser.add_argument(
        "--skip-arena-summary",
        "--no-check-arena-summary",
        dest="check_arena_summary",
        action="store_false",
        help="Skip /v1/hivemind/arena/summary verification in --pr949-quorum-gate.",
    )
    parser.add_argument(
        "--verify-idempotency-replay",
        action="store_true",
        help="Replay an Idempotency-Key CES POST twice in --pr949-quorum-gate to verify 1-to-1 dedup.",
    )
    parser.add_argument(
        "--run-deploy-dry-run",
        action="store_true",
        help="Run bash deploy/gcp_openrouter/deploy.sh --dry-run (no gcloud) in --pr949-quorum-gate.",
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


def _emit_report(
    report: Mapping[str, Any],
    *,
    as_json: bool,
    as_markdown: bool,
    output_path: Path | None = None,
    home: Path | None = None,
) -> bool:
    if as_json:
        rendered = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False), home=home)
    elif as_markdown:
        rendered = render_markdown_report(report)
    else:
        rendered = render_cli_report(report)
    print(rendered)
    if output_path is not None:
        if report.get("compliance_ok") is False:
            return False
        payload = rendered if rendered.endswith("\n") else (rendered + "\n")
        _write_file_secure_atomic(output_path, payload, mode=0o600)
    return True


def main(
    argv: Sequence[str] | None = None,
    *,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_lifecycle_fn: WorkerLifecycleFn = default_worker_lifecycle_drill,
    clock_fn: ClockFn = time.monotonic,
    http_fn: HttpFn = default_http_json,
    input_fn: InputFn = input,
    raw_http_fn: RawHttpFn = default_http_raw,
    runner: RunnerFn = default_subprocess_runner,
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
        cli_cwd = _validate_cli_path_arg(args.cwd, "--cwd", home=resolved_home)
        output_path = _validate_cli_path_arg(args.output, "--output", home=resolved_home)
        queue_path_arg = _validate_cli_path_arg(args.ces_queue_file, "--ces-queue-file", home=resolved_home)
    except SelftestError as exc:
        safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
        print(f"Error: {safe_err}", file=sys.stderr)
        return 2

    eff_cwd = cwd if cwd is not None else cli_cwd
    resolved_queue_path = (
        queue_path_arg
        if queue_path_arg is not None
        else default_ces_queue_path(home=resolved_home, credentials_path=args.credentials)
    )

    if args.clean_ces_queue and not (
        args.check_gateway
        or args.noel_verify
        or args.verify_fleet
        or args.send_opentrainer_heartbeat
        or args.submit_feedback
        or args.prompt_ces
        or args.pr949_quorum_gate
        or args.flush_ces_queue
    ):
        try:
            comp = evaluate_personal_pc_compliance(
                environ=environ,
                cwd=eff_cwd if eff_cwd is not None else Path.cwd(),
                lang=args.lang,
            )
            if not comp["compliance_ok"]:
                blocked_clean = _attach_receipt(
                    {
                        "object": "hivemind.gamer_whisper.ces_spool_clean",
                        "lang": normalize_lang(args.lang),
                        "compliance_ok": False,
                        "corporate_indicators": comp["corporate_indicators"],
                        "spool_file": redact_home_path(resolved_queue_path, home=resolved_home),
                        "removed": False,
                        "spool_exists": resolved_queue_path.is_file(),
                        "network_calls": 0,
                        "compliance": comp,
                        "diagnostics": [sanitize_text_for_display(comp["summary"], home=resolved_home)],
                        "exit_code": 2,
                    }
                )
                _emit_report(
                    blocked_clean,
                    as_json=bool(args.json),
                    as_markdown=bool(args.markdown),
                    output_path=output_path,
                    home=resolved_home,
                )
                return 2
            clean_res = GamerCesRetrySpool(resolved_queue_path, home=resolved_home).clean(home=resolved_home)
            clean_res["lang"] = normalize_lang(args.lang)
            clean_res["compliance_ok"] = True
            clean_res["compliance"] = comp
            clean_res = _attach_receipt(clean_res)
            _emit_report(
                clean_res,
                as_json=bool(args.json),
                as_markdown=bool(args.markdown),
                output_path=output_path,
                home=resolved_home,
            )
            return 0
        except SelftestError as exc:
            safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
            print(f"Error: {safe_err}", file=sys.stderr)
            return 2

    if args.pr949_quorum_gate:
        try:
            pr949 = verify_pr949_gamer_quorum_and_parity(
                args.gateway,
                expected_sha=args.expected_sha,
                target_online_producers=args.target_online_producers,
                min_ease_rating=args.min_ease_rating,
                min_zero_lag_ratio=args.min_zero_lag_ratio,
                require_live_quorum=bool(args.require_live_quorum),
                require_ces_slo=bool(args.require_ces_slo),
                check_arena_summary=bool(args.check_arena_summary),
                verify_idempotency_replay=bool(args.verify_idempotency_replay),
                execute_deploy_dry_run=bool(args.run_deploy_dry_run),
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                runner=runner,
                environ=environ,
                cwd=eff_cwd if eff_cwd is not None else Path.cwd(),
                home=resolved_home,
                lang=args.lang,
            )
        except SelftestError as exc:
            safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
            print(f"Error: {safe_err}", file=sys.stderr)
            return 2

        if pr949["gate_ok"] and args.flush_ces_queue:
            eff_flush_http: HttpFn = (
                http_fn
                if http_fn is not default_http_json
                else (raw_http_fn if raw_http_fn is not default_http_raw else default_http_json)  # type: ignore[assignment]
            )
            pr949["flush_ces_queue"] = flush_gamer_ces_queue(
                args.gateway,
                spool_path=resolved_queue_path,
                credentials_path=args.credentials,
                http_fn=eff_flush_http,
                lang=args.lang,
                home=resolved_home,
                environ=environ,
                cwd=eff_cwd if eff_cwd is not None else Path.cwd(),
            )
            pr949 = _attach_receipt(pr949)

        _emit_report(
            pr949,
            as_json=bool(args.json),
            as_markdown=bool(args.markdown),
            output_path=output_path,
            home=resolved_home,
        )
        return int(pr949["exit_code"])

    if args.flush_ces_queue and not (
        args.check_gateway
        or args.noel_verify
        or args.verify_fleet
        or args.send_opentrainer_heartbeat
        or args.submit_feedback
        or args.prompt_ces
    ):
        try:
            eff_flush_http_standalone: HttpFn = (
                http_fn
                if http_fn is not default_http_json
                else (raw_http_fn if raw_http_fn is not default_http_raw else default_http_json)  # type: ignore[assignment]
            )
            flush_res = flush_gamer_ces_queue(
                args.gateway,
                spool_path=resolved_queue_path,
                credentials_path=args.credentials,
                http_fn=eff_flush_http_standalone,
                lang=args.lang,
                home=resolved_home,
                environ=environ,
                cwd=eff_cwd if eff_cwd is not None else Path.cwd(),
            )
        except SelftestError as exc:
            safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
            print(f"Error: {safe_err}", file=sys.stderr)
            return 2
        _emit_report(
            flush_res,
            as_json=bool(args.json),
            as_markdown=bool(args.markdown),
            output_path=output_path,
            home=resolved_home,
        )
        return int(flush_res["exit_code"])

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
            noel_verify=bool(args.noel_verify),
            verify_fleet=bool(args.verify_fleet),
            send_opentrainer_heartbeat=bool(args.send_opentrainer_heartbeat),
            api_key=args.api_key,
            credentials_path=args.credentials,
            host_label=args.host_label,
            target_online_producers=args.target_online_producers,
            non_blocking_feedback=bool(args.non_blocking_feedback),
            expected_sha=args.expected_sha,
            submit_feedback=bool(args.submit_feedback),
            prompt_ces=bool(args.prompt_ces),
            ease_rating=args.ease_rating,
            had_game_lag=bool(args.had_game_lag),
            lang=args.lang,
            os_name=os_name,
            system_name=system_name,
            environ=environ,
            home=resolved_home,
            cwd=eff_cwd,
            gpu_probe_fn=gpu_probe_fn,
            process_probe_fn=process_probe_fn,
            worker_lifecycle_fn=worker_lifecycle_fn,
            clock_fn=clock_fn,
            http_fn=http_fn,
            input_fn=input_fn,
            spool_path=queue_path_arg,
            flush_ces_queue=bool(args.flush_ces_queue),
        )
    except SelftestError as exc:
        safe_err = sanitize_text_for_display(str(exc), home=resolved_home)
        print(f"Error: {safe_err}", file=sys.stderr)
        return 2

    _emit_report(
        report,
        as_json=bool(args.json),
        as_markdown=bool(args.markdown),
        output_path=output_path,
        home=resolved_home,
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
