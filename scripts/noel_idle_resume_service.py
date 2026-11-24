#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Opt-in user-space idle-resume GPU service, < 1.5 s Steam yield / <= 70 C state machine & CES spool.

Compatible with Python 3.10-3.13 across Windows 11, Linux, and macOS using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict outbound allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (Amadeus / Personal-PC Rule):
     inspects environment variables, workspace path, and ``.git/config`` remotes locally BEFORE
     any file write, scheduler install, subprocess probe, heartbeat, or CES call.
  2. Opt-In Cross-Platform User-Space Service Manifests, Atomic 0600/0700 Install & Uninstall:
     - Windows Task Scheduler XML (strictly XML-escaped, LogonTrigger, IdleSettings, StopOnIdleEnd,
       RestartOnIdle, RunOnlyIfIdle, DisallowStartIfOnBatteries) + PowerShell wrappers.
     - Linux ``systemd --user`` unit (Nice=15, NoNewPrivileges=yes, PrivateTmp=yes, Restart=on-failure),
       timer, and 0700 bash runner.
     - macOS LaunchAgent plist (RunAtLoad, StartInterval, LowPriorityIO, ProcessType=Background) +
       0700 bash runner.
  3. Deterministic Idle-Resume State Machine (``IDLE_READY``, ``RUNNING_LORA_ONLY``,
     ``PAUSED_GAME_YIELD``, ``PAUSED_THERMAL_CAP``, ``PAUSED_LOW_VRAM``, ``PAUSED_USER_ACTIVE``,
     ``DEV_API_ONLY_NO_GPU``) with < 1.5 s (1500.0 milliseconds) Steam yield, >= 300.0 s cooldown,
     <= 70.0 C quiet-fan governor, >= 2048 MiB VRAM reserve, and < 50.0 MB/h LoRA delta cap.
  4. Outbound-Only OpenTrainer Heartbeat, Fleet Badge Check, and Atomic 0600 ``IdleServiceCesQueue``.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal gaming/hobby/open-source use only (never corporate PCs).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import inspect
import io
import json
import math
import os
import platform
import plistlib
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.sax.saxutils import escape as xml_escape

SCHEMA = "hivemind.noel_idle_resume_service/1"
STATE_SCHEMA = "hivemind.noel_idle_resume_state/1"
CES_QUEUE_SCHEMA = "hivemind.idle_service_ces_queue/1"
RECEIPT_OBJECT = "hivemind.noel_idle_resume_service_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_HANDLE = "noel-gamer"
DEFAULT_ROLE = "prosumer"
DEFAULT_HOST_LABEL = "personal-pc"
DEFAULT_TRAINING_STACK = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_ACTIVE_CAMPAIGN = CAMPAIGN_ID
DEFAULT_NETWORK_MODE = "outbound-https-relay"
DEFAULT_TRANSFER_MODE = "lora_delta_only"
DEFAULT_CREDENTIALS_REL = Path(".hivemind") / "credentials.json"
DEFAULT_STATE_REL = Path(".hivemind") / "idle_resume_state.json"
DEFAULT_CES_QUEUE_REL = Path(".hivemind") / "idle_service_ces_queue.json"

DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_IDLE_THRESHOLD_S = 180.0
DEFAULT_COOLDOWN_S = 300.0
MIN_REQUIRED_COOLDOWN_S = 300.0
DEFAULT_INTERVAL_S = 60
DEFAULT_YIELD_BUDGET_MS = 1500.0
WDDM_TDR_RESET_MS = 2000.0
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0

DEFAULT_UPLOAD_CAP_MB_PER_HOUR = 50.0
DEFAULT_LORA_PARAMS = 4_194_304
DEFAULT_LORA_BYTES_PER_PARAM = 2
DEFAULT_SYNCS_PER_HOUR = 2.0
LEGACY_FULL_CHECKPOINT_MB = 14_000.0

DEFAULT_MAX_CES_QUEUE_ENTRIES = 128
DEFAULT_MAX_CES_DELIVERED = 512
MAX_CES_QUEUE_ENTRIES_CEILING = 4096
MAX_STATE_HISTORY = 32
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
MAX_GPU_VRAM_MIB = 1_048_576

SUBPROCESS_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 5.0
TELEMETRY_TIMEOUT_S = 2.5

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_PLATFORMS: tuple[str, ...] = ("windows", "linux", "macos")
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
VALID_FLEET_BADGES: frozenset[str] = frozenset({"ONLINE", "BOOTSTRAPPING", "STALE_CLOUD_RUN", "OFFLINE"})

STATE_IDLE_READY = "IDLE_READY"
STATE_RUNNING_LORA_ONLY = "RUNNING_LORA_ONLY"
STATE_PAUSED_GAME_YIELD = "PAUSED_GAME_YIELD"
STATE_PAUSED_THERMAL_CAP = "PAUSED_THERMAL_CAP"
STATE_PAUSED_LOW_VRAM = "PAUSED_LOW_VRAM"
STATE_PAUSED_USER_ACTIVE = "PAUSED_USER_ACTIVE"
STATE_DEV_API_ONLY_NO_GPU = "DEV_API_ONLY_NO_GPU"

VALID_STATES: frozenset[str] = frozenset(
    {
        STATE_IDLE_READY,
        STATE_RUNNING_LORA_ONLY,
        STATE_PAUSED_GAME_YIELD,
        STATE_PAUSED_THERMAL_CAP,
        STATE_PAUSED_LOW_VRAM,
        STATE_PAUSED_USER_ACTIVE,
        STATE_DEV_API_ONLY_NO_GPU,
    }
)
ACTIVE_HEARTBEAT_STATES: frozenset[str] = frozenset(
    {STATE_IDLE_READY, STATE_RUNNING_LORA_ONLY, STATE_DEV_API_ONLY_NO_GPU}
)

WINDOWS_TASK_NAME = "HiveMindNoelIdleProducer"
WINDOWS_TASK_XML_NAME = "hivemind-noel-idle-resume-task.xml"
WINDOWS_INSTALL_PS1_NAME = "install-noel-idle-service.ps1"
WINDOWS_UNINSTALL_PS1_NAME = "uninstall-noel-idle-service.ps1"
LINUX_SERVICE_NAME = "hivemind-noel-idle-producer.service"
LINUX_TIMER_NAME = "hivemind-noel-idle-producer.timer"
POSIX_RUNNER_NAME = "hivemind-noel-idle-runner.sh"
MACOS_PLIST_LABEL = "org.hivemind.noel.idle-producer"
MACOS_PLIST_NAME = "org.hivemind.noel.idle-producer.plist"

WINDOWS_TASK_XML_NS = "http:" + "//schemas.microsoft.com/windows/2004/02/mit/task"

EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
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

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

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
    "NO_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
    "ZSCALER_CERT_PATH",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

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

_CES_LOCK_GUARD = threading.Lock()
_CES_PATH_LOCKS: dict[Path, threading.Lock] = {}

HttpFn = Callable[..., Any]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
IdleProbeFn = Callable[..., Any]
WorkerYieldFn = Callable[..., Any]
ClockFn = Callable[[], float]
InputFn = Callable[[str], str]


class IdleServiceError(ValueError):
    """Raised when idle-resume service inputs, URL allowlists, or safety guards fail."""


@dataclass(frozen=True)
class GpuProbeSnapshot:
    """Truthful local NVIDIA GPU telemetry without UUID/serial identifiers or invented defaults."""

    present: bool
    parse_error: bool = False
    name: str | None = None
    total_vram_mib: int | None = None
    free_vram_mib: int | None = None
    used_vram_mib: int | None = None
    temp_c: float | None = None
    util_pct: float | None = None
    non_worker_util_pct: float | None = None
    reason: str = "ok"


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise IdleServiceError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise IdleServiceError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_platform(raw: str | None, *, system_name: str | None = None) -> str:
    """Normalize target OS platform (`'auto'`, `'windows'`, `'linux'`, `'macos'`)."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise IdleServiceError("Platform must be 'auto', 'windows', 'linux', or 'macos'.")
    cleaned = (raw or "auto").strip().lower()
    if cleaned == "auto":
        detected = (system_name if system_name is not None else platform.system()).strip().lower()
        if "darwin" in detected or "mac" in detected:
            return "macos"
        if detected.startswith("win") or detected == "nt":
            return "windows"
        if "linux" in detected:
            return "linux"
        raise IdleServiceError(f"Unsupported host OS {detected!r} for idle-resume service.")
    if cleaned in ("win", "win32", "nt", "windows"):
        return "windows"
    if cleaned in ("darwin", "mac", "macos"):
        return "macos"
    if cleaned == "linux":
        return "linux"
    raise IdleServiceError(f"Unsupported platform {raw!r}; expected auto, windows, linux, or macos.")


def _platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if "darwin" in raw or "mac" in raw:
        return "macos"
    if raw in ("windows", "win32", "nt") or raw.startswith("win"):
        return "windows"
    if "linux" in raw:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", raw)[:24].strip("_-")
    if cleaned and _PLATFORM_RE.fullmatch(cleaned):
        return cleaned
    return "unknown"


def normalize_profile(raw: str | None) -> str:
    """Normalize onboarding profile selector to `'dev'` or `'gamer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise IdleServiceError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise IdleServiceError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def normalize_role(raw: str | None) -> str:
    """Normalize account role to `'consumer'`, `'producer'`, `'prosumer'`, or `'trainer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise IdleServiceError("Role must be consumer, producer, prosumer, or trainer.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_ROLES:
        raise IdleServiceError(f"Unsupported role {raw!r}; expected consumer, producer, prosumer, or trainer.")
    return cleaned


def parse_ease_rating(raw: int | str) -> int:
    """Validate an integer ease rating in `1..5`."""
    if isinstance(raw, bool):
        raise IdleServiceError("ease_rating must be an integer between 1 and 5.")
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise IdleServiceError("ease_rating must be an integer between 1 and 5.") from exc
    if not (1 <= val <= 5):
        raise IdleServiceError(f"ease_rating {raw!r} out of range; expected 1..5.")
    return val


def sanitize_handle(raw: str | None) -> str:
    """Validate an explicit handle or generate a zero-PII personal handle."""
    if raw is None or not str(raw).strip():
        return f"noel-gamer-{secrets.token_hex(3)}"
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(raw).strip()).strip("-.")[:48]
    if (
        len(candidate) < 2
        or not _HANDLE_RE.fullmatch(candidate)
        or candidate.lower().startswith(("google.", "github.", "oidc."))
    ):
        raise IdleServiceError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-'."
        )
    return candidate


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff `raw_key` is a structurally valid `hm-...` or `hm_live_...` key."""
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


def _resolve_home_dir(*, home: Path | str | None = None, env: Mapping[str, str] | None = None) -> Path:
    if home is not None:
        return Path(home)
    env_view = env if env is not None else os.environ
    env_home = str(env_view.get("HOME") or env_view.get("USERPROFILE") or "").strip()
    return Path(env_home) if env_home else Path.home()


def _resolve_path_with_home(raw: Path | str, *, home_dir: Path) -> Path:
    raw_str = str(raw).strip()
    if raw_str == "~" or raw_str.startswith(("~/", "~\\")):
        suffix = raw_str[2:] if len(raw_str) > 2 else ""
        return (home_dir / suffix) if suffix else home_dir
    return Path(raw_str)


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace user home directory prefixes with `'~'` and scrub temporary user segments."""
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
                    redacted_home = "~" + suffix.replace("\\", "/")
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted_home)
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw `hm-...` keys, Bearer tokens, emails, non-loopback IPv4s, and OS home paths."""
    scrubbed = redact_home_path(text, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def sanitize_host_label(raw: str | None, *, home: Path | None = None) -> str:
    """Produce a clean 1..64 ASCII personal host label free of PII, secrets, IPs, and paths."""
    if raw is None or isinstance(raw, bool) or not str(raw).strip():
        return DEFAULT_HOST_LABEL
    scrubbed = sanitize_text_for_display(str(raw).strip(), home=home)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", scrubbed).strip("-._")[:64]
    if not cleaned or any(pat.search(cleaned) for _, pat in _CORP_PATTERNS):
        return DEFAULT_HOST_LABEL
    return cleaned


def sanitize_build_sha(raw: object) -> str:
    """Sanitize Cloud Run `build_sha` to lowercase alphanumeric/hex or `'unknown'`."""
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


def validate_expected_sha(raw: str) -> str:
    """Validate an explicit `--expected-sha` value."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise IdleServiceError(
            "expected_sha must be a 6..64 character alphanumeric/hex SHA (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live Cloud Run `build_sha` against `expected_sha` (exact or hex prefix)."""
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
        or not _REVISION_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return None
    return cleaned


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff `host` matches the repository lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL against allowed hosts and schemes."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise IdleServiceError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise IdleServiceError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise IdleServiceError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise IdleServiceError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise IdleServiceError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise IdleServiceError("Gateway URL must be an origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise IdleServiceError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise IdleServiceError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise IdleServiceError(
            f"Gateway host {host!r} is blocked by lateral-movement allowlist "
            "(expected https://*.run.app, *.googleapis.com, or loopback)."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise IdleServiceError("Loopback gateway URL must use http or https scheme.")
    elif parts.scheme != "https":
        raise IdleServiceError("Remote gateway URL must use https scheme.")
    return cleaned


def _validate_outbound_request_url(url: str) -> str:
    """Verify full request URL is allowlisted and never hits forbidden LLM inference routes."""
    if not isinstance(url, str) or not url.strip():
        raise IdleServiceError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise IdleServiceError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise IdleServiceError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat" in low_path or "hivemind/coder" in low_path:
        raise IdleServiceError(f"Zero-inference guard blocked forbidden endpoint {parts.path!r}.")
    return cleaned


def scan_output_safety(text: str) -> list[str]:
    """Verify serialized output contains zero PII, raw keys, home paths, or `'score': <digit>`."""
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_SCORE_KEY_RE.search(text):
        issues.append("forbidden_score_key_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise IdleServiceError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    text = raw_bytes.decode("utf-8", errors="replace")
    ctype = lower_headers.get("content-type", "").lower()
    if "application/json" in ctype or text.lstrip().startswith("{"):
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            result = dict(parsed)
            result.setdefault("_headers", lower_headers)
            return int(status_code), result
        return int(status_code), {"data": parsed, "_headers": lower_headers}
    return int(status_code), {"raw_length": len(text), "headers": lower_headers}


def default_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Perform a proxy-free HTTP request to an allowlisted endpoint (`ProxyHandler({})`)."""
    clean_url = _validate_outbound_request_url(url)
    verb = str(method or "GET").strip().upper()
    req_headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "hivemind-noel-idle-resume-service/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            req_headers[str(hk)] = str(hv)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(clean_url, data=body_bytes, headers=req_headers, method=verb)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    bounded_timeout = min(max(0.2, float(timeout_s)), HTTP_TIMEOUT_S)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw_hdrs = dict(resp.headers.items()) if resp.headers is not None else {}
            return _parse_http_raw_response(status, resp.read(524_288), raw_hdrs)
    except urllib.error.HTTPError as exc:
        err_hdrs = dict(exc.headers.items()) if exc.headers is not None else {}
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return _parse_http_raw_response(int(exc.code), err_bytes, err_hdrs)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise IdleServiceError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 3:
        status_code, raw_hdrs, body = raw_ret
        hdr_map = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    else:
        status_code, body = raw_ret
        hdr_map = {}

    if isinstance(body, Mapping):
        out = dict(body)
        if hdr_map:
            out.setdefault("_headers", hdr_map)
        return int(status_code), out
    return int(status_code), {"data": body, "_headers": hdr_map}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL, invoke injectable `http_fn` across arities, and map transport failures to 0."""
    clean_url = _validate_outbound_request_url(url)
    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    param_names = {p.name for p in params}
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    try:
        if has_varkw:
            raw = http_fn(
                method,
                clean_url,
                payload=dict(payload) if payload is not None else None,
                headers=dict(headers) if headers is not None else None,
                timeout_s=timeout_s,
            )
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = dict(payload) if payload is not None else None
            if "headers" in param_names:
                kwargs["headers"] = dict(headers) if headers is not None else None
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(method, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(method, clean_url, dict(payload) if payload is not None else None)
            elif len(pos_params) == 4 and not has_varargs:
                raw = http_fn(
                    method,
                    clean_url,
                    dict(payload) if payload is not None else None,
                    dict(headers) if headers is not None else None,
                )
            else:
                raw = http_fn(
                    method,
                    clean_url,
                    dict(payload) if payload is not None else None,
                    dict(headers) if headers is not None else None,
                    timeout_s,
                )
    except IdleServiceError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def _invoke_callable_flexible(fn: Callable[..., Any], **candidate_kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    """Extract remote and URL rewrite targets from `.git/config` text in memory."""
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` inside `workspace` (supporting git worktree `gitdir:` indirection)."""
    git_entry = workspace / ".git"
    candidates: list[Path] = []
    if git_entry.is_dir():
        cfg = git_entry / "config"
        if cfg.is_file():
            candidates.append(cfg)
    elif git_entry.is_file():
        try:
            first_lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            first_lines = []
        if first_lines and first_lines[0].strip().lower().startswith("gitdir:"):
            raw_gitdir = first_lines[0].split(":", 1)[1].strip()
            gitdir = Path(raw_gitdir) if Path(raw_gitdir).is_absolute() else (workspace / raw_gitdir).resolve()
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
    return candidates


def evaluate_personal_pc_compliance(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 writes, 0 probes, 0 HTTP calls)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _add_indicator(tag: str, marker: str | None = None) -> None:
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
            _add_indicator("env:AMADEUS*", "amadeus")
            for hit in _find_corporate_markers(val_str):
                matched_markers.add(hit)
            continue
        if key_up == "ZSCALER_CERT_PATH":
            _add_indicator("env:ZSCALER_CERT_PATH", "zscaler")
            continue
        if key_up == "USERDNSDOMAIN":
            dns_hits = _find_corporate_markers(val_str)
            _add_indicator("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "corp")
            for hit in dns_hits:
                matched_markers.add(hit)
            continue
        if key_up in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(val_str):
                _add_indicator(f"env:{key_up}:{hit}", hit)

    for hit in _find_corporate_markers(str(workspace)):
        _add_indicator(f"workspace_path:{hit}", hit)

    for cfg_path in _resolve_git_config_files(workspace):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cfg_text = ""
        for remote_url in extract_git_config_remote_urls(cfg_text):
            for hit in _find_corporate_markers(remote_url):
                _add_indicator(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"Local personal-PC compliance gate passed ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement d'entreprise détecté "
            f"(0 fichier écrit, 0 appel réseau). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE GATE BLOCKED ({joined}): corporate machine, proxy, or repository detected "
            f"(0 file writes, 0 outbound HTTP calls). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "compliance_ok": compliance_ok,
            "corporate_detected": not compliance_ok,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "file_writes": 0,
            "subprocess_calls": 0,
            "network_calls": 0,
            "exit_code": 0 if compliance_ok else 2,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _write_file_secure_atomic(path: Path, content: str, *, posix_mode: int, eff_os: str) -> None:
    """Atomically write UTF-8 `content` to `path` with `posix_mode` (`0600`/`0700`) via `.tmp` + `os.replace`."""
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
        except OSError as cleanup_err:
            _ = cleanup_err
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
    """Idempotently and atomically write `path` with `0600`/`0700` mode and `.bak` backup on diff."""
    eff_os = os_name if os_name is not None else os.name
    shown_path = redact_home_path(path, home=home)
    bak_path = Path(f"{path}.bak")
    shown_bak = redact_home_path(bak_path, home=home)
    perm_meta = {"permission_model": "windows_acl"} if eff_os == "nt" else {"posix_mode": f"{posix_mode:04o}"}

    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            tightened = False
            if eff_os != "nt":
                current_mode = path.stat().st_mode & 0o777
                if current_mode != posix_mode:
                    os.chmod(path, posix_mode)
                    tightened = True
            return {
                "path": shown_path,
                "action": "unchanged",
                "written": False,
                "created": False,
                "updated": False,
                "backup_created": False,
                "backup_path": None,
                "permissions_tightened": tightened,
                **perm_meta,
            }

        backup_written = False
        if create_backup:
            _write_file_secure_atomic(bak_path, existing, posix_mode=0o600, eff_os=eff_os)
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


def _validate_manifest_path_arg(raw: Path | str, field_name: str) -> str:
    if isinstance(raw, bool):
        raise IdleServiceError(f"{field_name} must be a path string, not bool.")
    text = str(raw).strip()
    if not text:
        raise IdleServiceError(f"{field_name} must be a non-empty path string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise IdleServiceError(f"{field_name} must not contain control characters or newlines.")
    return text


def _validate_service_thresholds(
    *,
    max_temp_c: float,
    min_free_vram_mib: int,
    idle_threshold_s: float,
    cooldown_s: float,
    interval_s: int,
) -> tuple[float, int, float, float, int]:
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise IdleServiceError("max_temp_c must be finite in [25.0, 95.0] C.")
    if isinstance(min_free_vram_mib, bool) or not isinstance(min_free_vram_mib, int) or not (256 <= min_free_vram_mib <= MAX_GPU_VRAM_MIB):
        raise IdleServiceError(f"min_free_vram_mib must be an integer in [256, {MAX_GPU_VRAM_MIB}].")
    if isinstance(idle_threshold_s, bool) or not math.isfinite(float(idle_threshold_s)) or not (30.0 <= float(idle_threshold_s) <= 86400.0):
        raise IdleServiceError("idle_threshold_s must be finite in [30.0, 86400.0] s.")
    if isinstance(cooldown_s, bool) or not math.isfinite(float(cooldown_s)) or float(cooldown_s) < MIN_REQUIRED_COOLDOWN_S:
        raise IdleServiceError(f"cooldown_s must be finite and >= {MIN_REQUIRED_COOLDOWN_S:.1f} s.")
    if isinstance(interval_s, bool) or not isinstance(interval_s, int) or not (10 <= interval_s <= 86400):
        raise IdleServiceError("interval_s must be an integer in [10, 86400] s.")
    return (
        round(float(max_temp_c), 1),
        int(min_free_vram_mib),
        round(float(idle_threshold_s), 1),
        round(float(cooldown_s), 1),
        int(interval_s),
    )


def _xml_escape(text: str) -> str:
    return xml_escape(str(text), {'"': "&quot;", "'": "&apos;"})


def _ps_single_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _build_tick_cli_flags(
    *,
    gateway: str,
    credentials_path: str,
    max_temp_c: float,
    min_free_vram_mib: int,
    idle_threshold_s: float,
    cooldown_s: float,
    state_path: str | None = None,
    ces_queue_path: str | None = None,
) -> list[str]:
    flags = [
        "--tick",
        "--send-heartbeat",
        "--gateway",
        gateway,
        "--credentials-path",
        credentials_path,
        "--max-temp-c",
        f"{max_temp_c:.1f}",
        "--min-free-vram-mib",
        str(min_free_vram_mib),
        "--idle-threshold-s",
        f"{idle_threshold_s:.1f}",
        "--cooldown-s",
        f"{cooldown_s:.1f}",
        "--transfer-mode",
        DEFAULT_TRANSFER_MODE,
    ]
    if state_path:
        flags.extend(["--state-file", state_path])
    if ces_queue_path:
        flags.extend(["--ces-queue-file", ces_queue_path])
    return flags


def _render_posix_runner_script(
    *,
    python_exe: str,
    script_path: str,
    tick_flags: Sequence[str],
) -> str:
    exec_tokens = " ".join(shlex.quote(tok) for tok in (python_exe, script_path, *tick_flags))
    return (
        "#!/usr/bin/env bash\n"
        "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.\n"
        "# Opt-in user-space idle-resume LoRA delta worker (< 50 MB/h, outbound HTTPS 443, 0 inbound ports).\n"
        "# 100% personal home PCs & personal gaming/hobby use only (0.00 EUR / 0.00 USD, AGPL-3.0).\n"
        "set -euo pipefail\n"
        "export PYTHONUNBUFFERED=1\n"
        "export HIVEMIND_TRANSFER_MODE=lora_delta_only\n"
        f"exec {exec_tokens}\n"
    )


def _render_windows_manifests(
    *,
    gateway: str,
    python_exe: str,
    script_path: str,
    tick_flags: Sequence[str],
    idle_threshold_s: float,
    interval_s: int,
    max_temp_c: float,
    min_free_vram_mib: int,
    cooldown_s: float,
) -> tuple[dict[str, str], dict[str, int]]:
    win_args = " ".join(
        f'"{tok}"' if (" " in tok or any(ch in tok for ch in "<>&'")) else tok
        for tok in (script_path, *tick_flags)
    )
    idle_dur_s = int(round(idle_threshold_s))
    desc = (
        f"HiveMind Noel user-space idle-resume LoRA producer (outbound HTTPS 443 only, "
        f"zero inbound ports, < 50 MB/h LoRA delta, < 1.5 s Steam yield, <= {max_temp_c:.1f} C, "
        f"free VRAM >= {min_free_vram_mib} MiB, {cooldown_s:.0f} s cooldown; "
        "100% personal home PCs only, 0.00 EUR / 0.00 USD, AGPL-3.0)."
    )
    xml_text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Task version="1.2" xmlns="{_xml_escape(WINDOWS_TASK_XML_NS)}">\n'
        "  <RegistrationInfo>\n"
        "    <Author>Project HiveMind (Personal Home PC Only)</Author>\n"
        f"    <Description>{_xml_escape(desc)}</Description>\n"
        "  </RegistrationInfo>\n"
        "  <Triggers>\n"
        "    <LogonTrigger>\n"
        "      <Enabled>true</Enabled>\n"
        "      <Repetition>\n"
        f"        <Interval>PT{interval_s}S</Interval>\n"
        "        <StopAtDurationEnd>false</StopAtDurationEnd>\n"
        "      </Repetition>\n"
        "    </LogonTrigger>\n"
        "  </Triggers>\n"
        "  <Principals>\n"
        '    <Principal id="Author">\n'
        "      <LogonType>InteractiveToken</LogonType>\n"
        "      <RunLevel>LeastPrivilege</RunLevel>\n"
        "    </Principal>\n"
        "  </Principals>\n"
        "  <Settings>\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n"
        "    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>\n"
        "    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>\n"
        "    <AllowHardTerminate>true</AllowHardTerminate>\n"
        "    <StartWhenAvailable>true</StartWhenAvailable>\n"
        "    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>\n"
        "    <IdleSettings>\n"
        f"      <Duration>PT{idle_dur_s}S</Duration>\n"
        "      <WaitTimeout>PT1H</WaitTimeout>\n"
        "      <StopOnIdleEnd>true</StopOnIdleEnd>\n"
        "      <RestartOnIdle>true</RestartOnIdle>\n"
        "    </IdleSettings>\n"
        "    <AllowStartOnDemand>true</AllowStartOnDemand>\n"
        "    <Enabled>true</Enabled>\n"
        "    <Hidden>false</Hidden>\n"
        "    <RunOnlyIfIdle>true</RunOnlyIfIdle>\n"
        "    <WakeToRun>false</WakeToRun>\n"
        "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n"
        "    <Priority>7</Priority>\n"
        "  </Settings>\n"
        '  <Actions Context="Author">\n'
        "    <Exec>\n"
        f"      <Command>{_xml_escape(python_exe)}</Command>\n"
        f"      <Arguments>{_xml_escape(win_args)}</Arguments>\n"
        "    </Exec>\n"
        "  </Actions>\n"
        "</Task>\n"
    )
    parsed = ET.fromstring(xml_text)
    ns = {"t": WINDOWS_TASK_XML_NS}
    exec_nodes = parsed.findall(".//t:Exec", ns)
    if len(exec_nodes) != 1:
        raise IdleServiceError("XML validation failed: unexpected Exec node count.")

    install_ps1 = (
        "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.\n"
        "# Opt-in user-space Windows Task Scheduler installer (100% personal home PCs only, 0.00 EUR / 0.00 USD).\n"
        "# Outbound HTTPS 443 relay only, < 50 MB/h LoRA delta mode, zero inbound ports.\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$XmlPath = Join-Path $PSScriptRoot " + _ps_single_quote(WINDOWS_TASK_XML_NAME) + "\n"
        "$Gateway = " + _ps_single_quote(gateway) + "\n"
        f"& schtasks.exe /Create /TN {_ps_single_quote(WINDOWS_TASK_NAME)} /XML $XmlPath /F\n"
        f"Write-Host ('Installed user-space idle task {WINDOWS_TASK_NAME} -> ' + $Gateway)\n"
    )
    uninstall_ps1 = (
        "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.\n"
        "# Clean user-space uninstaller removing ONLY HiveMind Noel idle-resume artifacts.\n"
        "$ErrorActionPreference = 'SilentlyContinue'\n"
        f"& schtasks.exe /Delete /TN {_ps_single_quote(WINDOWS_TASK_NAME)} /F | Out-Null\n"
        "foreach ($Rel in @('.hivemind\\idle_resume_state.json', '.hivemind\\idle_service_ces_queue.json')) {\n"
        "    $Target = Join-Path $HOME $Rel\n"
        "    if (Test-Path $Target) { Remove-Item -Force $Target }\n"
        "}\n"
        f"Write-Host 'Removed user-space task {WINDOWS_TASK_NAME}.'\n"
    )
    files = {
        WINDOWS_TASK_XML_NAME: xml_text,
        WINDOWS_INSTALL_PS1_NAME: install_ps1,
        WINDOWS_UNINSTALL_PS1_NAME: uninstall_ps1,
    }
    modes = {name: 0o600 for name in files}
    return files, modes


def _render_linux_manifests(
    *,
    runner_exec_path: str,
    runner_content: str,
    interval_s: int,
    max_temp_c: float,
) -> tuple[dict[str, str], dict[str, int]]:
    service_unit = (
        "[Unit]\n"
        f"Description=Project HiveMind Noel Idle-Resume LoRA Producer (< 50 MB/h, <= {max_temp_c:.1f} C, outbound HTTPS 443)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "Nice=15\n"
        "IOSchedulingClass=idle\n"
        "NoNewPrivileges=yes\n"
        "PrivateTmp=yes\n"
        "Restart=on-failure\n"
        "RestartSec=15\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "Environment=HIVEMIND_TRANSFER_MODE=lora_delta_only\n"
        f"ExecStart={runner_exec_path}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    timer_unit = (
        "[Unit]\n"
        "Description=Project HiveMind Noel Idle-Resume LoRA Producer Timer (100% personal home PC only)\n"
        "\n"
        "[Timer]\n"
        "OnBootSec=60s\n"
        f"OnUnitActiveSec={interval_s}s\n"
        "AccuracySec=5s\n"
        f"Unit={LINUX_SERVICE_NAME}\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    files = {
        LINUX_SERVICE_NAME: service_unit,
        LINUX_TIMER_NAME: timer_unit,
        POSIX_RUNNER_NAME: runner_content,
    }
    modes = {
        LINUX_SERVICE_NAME: 0o600,
        LINUX_TIMER_NAME: 0o600,
        POSIX_RUNNER_NAME: 0o700,
    }
    return files, modes


def _render_macos_manifests(
    *,
    runner_exec_path: str,
    runner_content: str,
    interval_s: int,
) -> tuple[dict[str, str], dict[str, int]]:
    plist_obj = {
        "Label": MACOS_PLIST_LABEL,
        "ProgramArguments": ["/bin/bash", runner_exec_path],
        "RunAtLoad": True,
        "StartInterval": int(interval_s),
        "LowPriorityIO": True,
        "ProcessType": "Background",
        "Nice": 15,
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "HIVEMIND_TRANSFER_MODE": DEFAULT_TRANSFER_MODE,
        },
    }
    plist_bytes = plistlib.dumps(plist_obj, fmt=plistlib.FMT_XML, sort_keys=False)
    plist_text = plist_bytes.decode("utf-8")
    loaded = plistlib.loads(plist_bytes)
    if loaded.get("Label") != MACOS_PLIST_LABEL:
        raise IdleServiceError("macOS plist validation failed.")
    files = {
        MACOS_PLIST_NAME: plist_text,
        POSIX_RUNNER_NAME: runner_content,
    }
    modes = {
        MACOS_PLIST_NAME: 0o600,
        POSIX_RUNNER_NAME: 0o700,
    }
    return files, modes


def default_service_install_paths(
    platform_tag: str,
    *,
    service_dir: Path | str | None = None,
    home: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Resolve user-space service file destination paths without writing anything."""
    plat = normalize_platform(platform_tag)
    resolved_home = _resolve_home_dir(home=home, env=env)
    custom_dir = _resolve_path_with_home(service_dir, home_dir=resolved_home) if service_dir is not None else None

    if plat == "windows":
        base = custom_dir if custom_dir is not None else (resolved_home / ".hivemind" / "service")
        return {
            WINDOWS_TASK_XML_NAME: base / WINDOWS_TASK_XML_NAME,
            WINDOWS_INSTALL_PS1_NAME: base / WINDOWS_INSTALL_PS1_NAME,
            WINDOWS_UNINSTALL_PS1_NAME: base / WINDOWS_UNINSTALL_PS1_NAME,
        }
    if plat == "linux":
        unit_dir = custom_dir if custom_dir is not None else (resolved_home / ".config" / "systemd" / "user")
        share_dir = custom_dir if custom_dir is not None else (resolved_home / ".local" / "share" / "hivemind")
        return {
            LINUX_SERVICE_NAME: unit_dir / LINUX_SERVICE_NAME,
            LINUX_TIMER_NAME: unit_dir / LINUX_TIMER_NAME,
            POSIX_RUNNER_NAME: share_dir / POSIX_RUNNER_NAME,
        }
    plist_dir = custom_dir if custom_dir is not None else (resolved_home / "Library" / "LaunchAgents")
    share_dir = custom_dir if custom_dir is not None else (resolved_home / ".local" / "share" / "hivemind")
    return {
        MACOS_PLIST_NAME: plist_dir / MACOS_PLIST_NAME,
        POSIX_RUNNER_NAME: share_dir / POSIX_RUNNER_NAME,
    }


def _scheduler_activation_commands(platform_tag: str, target_paths: Mapping[str, Path]) -> list[list[str]]:
    if platform_tag == "windows":
        return [["schtasks", "/Create", "/TN", WINDOWS_TASK_NAME, "/XML", str(target_paths[WINDOWS_TASK_XML_NAME]), "/F"]]
    if platform_tag == "linux":
        return [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", LINUX_TIMER_NAME],
        ]
    return [["launchctl", "load", "-w", str(target_paths[MACOS_PLIST_NAME])]]


def _scheduler_deactivation_commands(platform_tag: str, target_paths: Mapping[str, Path]) -> list[list[str]]:
    if platform_tag == "windows":
        return [["schtasks", "/Delete", "/TN", WINDOWS_TASK_NAME, "/F"]]
    if platform_tag == "linux":
        return [
            ["systemctl", "--user", "disable", "--now", LINUX_TIMER_NAME],
            ["systemctl", "--user", "daemon-reload"],
        ]
    return [["launchctl", "unload", "-w", str(target_paths[MACOS_PLIST_NAME])]]


def build_idle_service_manifests(
    platform: str = "auto",
    *,
    gateway: str = DEFAULT_GATEWAY,
    python_exe: str | None = None,
    script_path: str | None = None,
    credentials_path: str | None = None,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    interval_s: int = DEFAULT_INTERVAL_S,
    runner_path: Path | str | None = None,
    state_path: Path | str | None = None,
    ces_queue_path: Path | str | None = None,
    home: Path | str | None = None,
    system_name: str | None = None,
) -> dict[str, Any]:
    """Generate cross-platform user-space idle-resume service manifests with strict escaping."""
    plat = normalize_platform(platform, system_name=system_name)
    clean_gw = validate_gateway_url(gateway)
    cap_c, min_vram, idle_s, cool_s, int_s = _validate_service_thresholds(
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        idle_threshold_s=idle_threshold_s,
        cooldown_s=cooldown_s,
        interval_s=interval_s,
    )
    resolved_home = _resolve_home_dir(home=home)
    clean_py = _validate_manifest_path_arg(
        sys.executable if python_exe is None else python_exe,
        "python_exe",
    )
    clean_script = _validate_manifest_path_arg(
        str(Path(__file__).resolve()) if script_path is None else script_path,
        "script_path",
    )
    clean_creds = _validate_manifest_path_arg(
        str(resolved_home / DEFAULT_CREDENTIALS_REL) if credentials_path is None else credentials_path,
        "credentials_path",
    )
    clean_state = (
        _validate_manifest_path_arg(state_path, "state_path") if state_path is not None else None
    )
    clean_queue = (
        _validate_manifest_path_arg(ces_queue_path, "ces_queue_path")
        if ces_queue_path is not None
        else None
    )

    tick_flags = _build_tick_cli_flags(
        gateway=clean_gw,
        credentials_path=clean_creds,
        max_temp_c=cap_c,
        min_free_vram_mib=min_vram,
        idle_threshold_s=idle_s,
        cooldown_s=cool_s,
        state_path=clean_state,
        ces_queue_path=clean_queue,
    )
    runner_exec = (
        _validate_manifest_path_arg(runner_path, "runner_path")
        if runner_path is not None
        else str(default_service_install_paths(plat, home=resolved_home).get(POSIX_RUNNER_NAME, ""))
    )

    if plat == "windows":
        files, modes = _render_windows_manifests(
            gateway=clean_gw,
            python_exe=clean_py,
            script_path=clean_script,
            tick_flags=tick_flags,
            idle_threshold_s=idle_s,
            interval_s=int_s,
            max_temp_c=cap_c,
            min_free_vram_mib=min_vram,
            cooldown_s=cool_s,
        )
    else:
        runner_sh = _render_posix_runner_script(
            python_exe=clean_py,
            script_path=clean_script,
            tick_flags=tick_flags,
        )
        if plat == "linux":
            files, modes = _render_linux_manifests(
                runner_exec_path=runner_exec,
                runner_content=runner_sh,
                interval_s=int_s,
                max_temp_c=cap_c,
            )
        else:
            files, modes = _render_macos_manifests(
                runner_exec_path=runner_exec,
                runner_content=runner_sh,
                interval_s=int_s,
            )

    default_targets = default_service_install_paths(plat, home=resolved_home)
    return {
        "schema": SCHEMA,
        "platform": plat,
        "gateway": clean_gw,
        "network_mode": DEFAULT_NETWORK_MODE,
        "transfer_mode": DEFAULT_TRANSFER_MODE,
        "outbound_only_https_443": True,
        "inbound_ports": 0,
        "upload_cap_mb_per_hour": DEFAULT_UPLOAD_CAP_MB_PER_HOUR,
        "max_temp_c": cap_c,
        "min_free_vram_mib": min_vram,
        "idle_threshold_s": idle_s,
        "cooldown_s": cool_s,
        "interval_s": int_s,
        "files": files,
        "file_modes": modes,
        "scheduler_commands": _scheduler_activation_commands(plat, default_targets),
    }


def install_user_space_service(
    platform: str = "auto",
    *,
    gateway: str = DEFAULT_GATEWAY,
    service_dir: Path | str | None = None,
    python_exe: str | None = None,
    script_path: str | None = None,
    credentials_path: Path | str | None = None,
    state_path: Path | str | None = None,
    ces_queue_path: Path | str | None = None,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    interval_s: int = DEFAULT_INTERVAL_S,
    activate_scheduler: bool = False,
    service_runner: RunnerFn | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Install user-space idle-resume manifests atomically (`0600`/`0700` + `.bak`), dry-run by default."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    plat = normalize_platform(platform, system_name=system_name)
    clean_gw = validate_gateway_url(gateway)
    env_view = env if env is not None else os.environ
    eff_os = os_name if os_name is not None else ("nt" if plat == "windows" else os.name)
    resolved_home = _resolve_home_dir(home=home, env=env_view)
    ws_path = Path(cwd) if cwd is not None else Path.cwd()

    compliance = evaluate_personal_pc_compliance(env=env_view, cwd=ws_path, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "install_service",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "verdict": "blocked_corporate_environment",
                "platform": plat,
                "gateway": clean_gw,
                "files": {},
                "file_writes": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "scheduler_dry_run": True,
                "scheduler_activated": False,
                "scheduler_commands": [],
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    cred_target = (
        _resolve_path_with_home(credentials_path, home_dir=resolved_home)
        if credentials_path is not None
        else (resolved_home / DEFAULT_CREDENTIALS_REL)
    )
    state_target = (
        _resolve_path_with_home(state_path, home_dir=resolved_home)
        if state_path is not None
        else (resolved_home / DEFAULT_STATE_REL)
    )
    queue_target = (
        _resolve_path_with_home(ces_queue_path, home_dir=resolved_home)
        if ces_queue_path is not None
        else (resolved_home / DEFAULT_CES_QUEUE_REL)
    )
    target_paths = default_service_install_paths(
        plat,
        service_dir=service_dir,
        home=resolved_home,
        env=env_view,
    )
    runner_exec = target_paths.get(POSIX_RUNNER_NAME)

    bundle = build_idle_service_manifests(
        plat,
        gateway=clean_gw,
        python_exe=python_exe or sys.executable,
        script_path=script_path or str(Path(__file__).resolve()),
        credentials_path=str(cred_target),
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        idle_threshold_s=idle_threshold_s,
        cooldown_s=cooldown_s,
        interval_s=interval_s,
        runner_path=runner_exec,
        state_path=state_target,
        ces_queue_path=queue_target,
        home=resolved_home,
        system_name=system_name,
    )

    files_report: dict[str, dict[str, Any]] = {}
    for fname, content in bundle["files"].items():
        dest = target_paths[fname]
        mode = int(bundle["file_modes"][fname])
        files_report[fname] = atomic_write_config_file(
            dest,
            content,
            posix_mode=mode,
            create_backup=True,
            os_name=eff_os,
            home=resolved_home,
        )

    sched_cmds = _scheduler_activation_commands(plat, target_paths)
    redacted_cmds = [
        [redact_home_path(tok, home=resolved_home) for tok in cmd] for cmd in sched_cmds
    ]

    subprocess_calls = 0
    scheduler_activated = False
    scheduler_results: list[dict[str, Any]] = []
    if activate_scheduler:
        eff_runner = service_runner if service_runner is not None else default_subprocess_runner
        all_ok = True
        for cmd, shown_cmd in zip(sched_cmds, redacted_cmds):
            rc, out = eff_runner(cmd, SUBPROCESS_TIMEOUT_S)
            subprocess_calls += 1
            cmd_ok = int(rc) == 0
            all_ok = all_ok and cmd_ok
            scheduler_results.append(
                {
                    "cmd": shown_cmd,
                    "rc": int(rc),
                    "ok": cmd_ok,
                    "output": sanitize_text_for_display(out[:180], home=resolved_home),
                }
            )
        scheduler_activated = all_ok
    else:
        scheduler_results = [{"cmd": shown_cmd, "dry_run": True} for shown_cmd in redacted_cmds]

    file_writes = sum(1 for item in files_report.values() if bool(item.get("written")))
    overall_ok = True if not activate_scheduler else scheduler_activated
    exit_code = 0 if overall_ok else 2
    if overall_ok:
        summary = (
            f"Service utilisateur idle-resume ({plat}) installé (writes={file_writes}, dry_run={not activate_scheduler})."
            if is_fr
            else f"User-space idle-resume service ({plat}) installed (writes={file_writes}, dry_run={not activate_scheduler})."
        )
    else:
        summary = (
            f"Échec d'activation du planificateur utilisateur ({plat})."
            if is_fr
            else f"User-space scheduler activation failed ({plat})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "install_service",
            "ok": overall_ok,
            "ready": overall_ok,
            "compliance_ok": True,
            "exit_code": exit_code,
            "verdict": "installed" if overall_ok else "scheduler_activation_failed",
            "platform": plat,
            "gateway": clean_gw,
            "network_mode": DEFAULT_NETWORK_MODE,
            "transfer_mode": DEFAULT_TRANSFER_MODE,
            "inbound_ports": 0,
            "upload_cap_mb_per_hour": DEFAULT_UPLOAD_CAP_MB_PER_HOUR,
            "files": files_report,
            "file_writes": file_writes,
            "network_calls": 0,
            "subprocess_calls": subprocess_calls,
            "scheduler_dry_run": not activate_scheduler,
            "scheduler_activated": scheduler_activated,
            "scheduler_commands": redacted_cmds,
            "scheduler_results": scheduler_results,
            "compliance": compliance,
            "summary": summary,
        }
    )


def uninstall_user_space_service(
    platform: str = "auto",
    *,
    service_dir: Path | str | None = None,
    state_path: Path | str | None = None,
    ces_queue_path: Path | str | None = None,
    activate_scheduler: bool = False,
    service_runner: RunnerFn | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Cleanly remove ONLY HiveMind-managed service, state, and spool artifacts (always permitted)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    plat = normalize_platform(platform, system_name=system_name)
    env_view = env if env is not None else os.environ
    resolved_home = _resolve_home_dir(home=home, env=env_view)

    state_target = (
        _resolve_path_with_home(state_path, home_dir=resolved_home)
        if state_path is not None
        else (resolved_home / DEFAULT_STATE_REL)
    )
    queue_target = (
        _resolve_path_with_home(ces_queue_path, home_dir=resolved_home)
        if ces_queue_path is not None
        else (resolved_home / DEFAULT_CES_QUEUE_REL)
    )

    all_platform_targets: list[Path] = []
    for p_tag in VALID_PLATFORMS:
        tmap = default_service_install_paths(
            p_tag,
            service_dir=service_dir,
            home=resolved_home,
            env=env_view,
        )
        all_platform_targets.extend(tmap.values())

    managed_files: list[Path] = []
    for candidate in (*all_platform_targets, state_target, queue_target):
        if candidate not in managed_files:
            managed_files.append(candidate)
        bak = Path(f"{candidate}.bak")
        if bak not in managed_files:
            managed_files.append(bak)

    subprocess_calls = 0
    if activate_scheduler:
        eff_runner = service_runner if service_runner is not None else default_subprocess_runner
        plat_targets = default_service_install_paths(
            plat,
            service_dir=service_dir,
            home=resolved_home,
            env=env_view,
        )
        for cmd in _scheduler_deactivation_commands(plat, plat_targets):
            eff_runner(cmd, SUBPROCESS_TIMEOUT_S)
            subprocess_calls += 1

    removed_files: list[str] = []
    for path_item in managed_files:
        if path_item.is_file():
            path_item.unlink(missing_ok=True)
            removed_files.append(redact_home_path(path_item, home=resolved_home))
        if path_item.parent.is_dir():
            for stale_tmp in path_item.parent.glob(f".{path_item.name}.*.tmp"):
                if stale_tmp.is_file():
                    stale_tmp.unlink(missing_ok=True)

    summary = (
        f"Désinstallation propre du service idle-resume terminée ({len(removed_files)} artefact(s) supprimé(s))."
        if is_fr
        else f"Clean user-space idle-resume uninstall complete ({len(removed_files)} artifact(s) removed)."
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "uninstall_service",
            "ok": True,
            "ready": True,
            "exit_code": 0,
            "platform": plat,
            "removed_files": removed_files,
            "removed_count": len(removed_files),
            "network_calls": 0,
            "subprocess_calls": subprocess_calls,
            "compliance_notice": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def default_subprocess_runner(
    cmd: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Execute a read-only local argv command (`shell=False`) with `<= 3.0 s` timeout."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise IdleServiceError("Subprocess command must be a non-empty sequence of arguments.")
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
    """Query local `nvidia-smi` metrics without UUID/serial identifiers or invented specs."""
    rc, stdout = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        min(float(timeout_s), SUBPROCESS_TIMEOUT_S),
    )
    if rc != 0 or not stdout.strip():
        return GpuProbeSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")

    snapshots: list[GpuProbeSnapshot] = []
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
        snapshots.append(
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
        snapshots,
        key=lambda g: (
            g.free_vram_mib or 0,
            -(g.temp_c if g.temp_c is not None else 999.0),
            g.total_vram_mib or 0,
        ),
    )


def _normalize_gpu_snapshot(raw: Any) -> GpuProbeSnapshot:
    if raw is None:
        return GpuProbeSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")
    if isinstance(raw, GpuProbeSnapshot):
        return raw
    if isinstance(raw, Mapping):
        view: Mapping[str, Any] = raw
    elif hasattr(raw, "present"):
        view = {
            "present": getattr(raw, "present", False),
            "parse_error": getattr(raw, "parse_error", False),
            "name": getattr(raw, "name", None) or getattr(raw, "gpu_name", None),
            "total_vram_mib": getattr(raw, "total_vram_mib", getattr(raw, "total_mib", None)),
            "free_vram_mib": getattr(raw, "free_vram_mib", getattr(raw, "free_mib", None)),
            "used_vram_mib": getattr(raw, "used_vram_mib", getattr(raw, "used_mib", None)),
            "temp_c": getattr(raw, "temp_c", None),
            "util_pct": getattr(raw, "util_pct", None),
            "non_worker_util_pct": getattr(raw, "non_worker_util_pct", None),
            "reason": getattr(raw, "reason", "ok"),
        }
    else:
        raise IdleServiceError("gpu_probe_fn must return GpuProbeSnapshot, a mapping, or None.")

    present = bool(view.get("present", view.get("gpu_present", False)))
    if not present:
        return GpuProbeSnapshot(present=False, reason=str(view.get("reason") or "dev_api_only_no_nvidia_gpu"))
    if bool(view.get("parse_error", False)):
        return GpuProbeSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")

    name_raw = view.get("name") or view.get("gpu_name") or view.get("gpu_model")
    clean_name = str(name_raw).strip()[:128] if name_raw is not None and str(name_raw).strip() else None
    raw_total = view.get("total_vram_mib", view.get("vram_total_mib", view.get("total_mib")))
    raw_free = view.get("free_vram_mib", view.get("vram_free_mib", view.get("free_mib")))
    raw_used = view.get("used_vram_mib", view.get("vram_used_mib", view.get("used_mib")))
    temp_f = _optional_finite_float(view.get("temp_c"))
    util_f = _optional_finite_float(view.get("util_pct", view.get("utilization_pct")))
    non_worker_f = _optional_finite_float(view.get("non_worker_util_pct"))
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
        non_worker_util_pct=round(non_worker_f, 1) if non_worker_f is not None else None,
        reason=str(view.get("reason") or "ok"),
    )


def default_process_probe(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> list[str]:
    """List active local process names via `tasklist` (Windows) or `ps` (POSIX) in `<= 3.0 s`."""
    eff_os = os_name if os_name is not None else os.name
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)
    names: list[str] = []
    seen: set[str] = set()
    if eff_os == "nt":
        rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], bounded)
        if rc == 0 and out.strip():
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    low = row[0].strip().lower()
                    if low not in seen:
                        seen.add(low)
                        names.append(low)
        return names

    rc, out = runner(["ps", "-eo", "comm="], bounded)
    if rc == 0 and out.strip():
        for ln in out.splitlines():
            cleaned = ln.strip()
            if not cleaned:
                continue
            item = Path(cleaned.replace("\\", "/")).name.lower()
            if item and item not in seen:
                seen.add(item)
                names.append(item)
    return names


def default_idle_probe(
    *,
    runner: RunnerFn = default_subprocess_runner,
    sys_platform: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> float:
    """Query OS user idle seconds safely across Windows, macOS, and Linux."""
    eff_plat = (sys_platform if sys_platform is not None else sys.platform).lower()
    env_view = env if env is not None else os.environ
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)

    if eff_plat == "win32":
        windll = getattr(ctypes, "windll", None)
        if windll is not None:
            class _LastInputInfo(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

            lii = _LastInputInfo()
            lii.cbSize = ctypes.sizeof(_LastInputInfo)
            if windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                tick_count = int(windll.kernel32.GetTickCount())
                elapsed_ms = (tick_count - int(lii.dwTime)) & 0xFFFFFFFF
                return round(elapsed_ms / 1000.0, 3)
        return 300.0

    if eff_plat == "darwin":
        rc, out = runner(["ioreg", "-c", "IOHIDSystem"], bounded)
        if rc == 0 and out:
            match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
            if match:
                return round(int(match.group(1)) / 1_000_000_000.0, 3)
        return 300.0

    if env_view.get("DISPLAY") or env_view.get("WAYLAND_DISPLAY"):
        rc, out = runner(["xprintidle"], bounded)
        if rc == 0 and out.strip().isdigit():
            return round(int(out.strip()) / 1000.0, 3)
    return 300.0


def detect_steam_or_dx12_game_activity(
    process_names: Iterable[str],
    *,
    non_worker_gpu_util_pct: float | None = None,
    extra_games: Sequence[str] = (),
    gpu_util_threshold_pct: float = NON_WORKER_GPU_UTIL_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Detect Steam / DX12 game processes or non-worker GPU utilization `> 15.0%`."""
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
        is_steam = (
            base.startswith(STEAM_PROCESS_PREFIXES)
            or "steamapps/common/" in raw_low
            or base in ("steam_game", "steamwebhelper_game")
        )
        if base in watchlist or base in trunc_prefixes or is_steam:
            if base not in matched_procs:
                matched_procs.append(base)

    gpu_util_triggered = bool(
        non_worker_gpu_util_pct is not None
        and math.isfinite(float(non_worker_gpu_util_pct))
        and float(non_worker_gpu_util_pct) > float(gpu_util_threshold_pct)
    )
    reasons: list[str] = []
    if matched_procs:
        reasons.append(f"steam_dx12_process:{','.join(matched_procs)}")
    if gpu_util_triggered and non_worker_gpu_util_pct is not None:
        reasons.append(f"non_worker_gpu_util:{float(non_worker_gpu_util_pct):.1f}%>{gpu_util_threshold_pct:.1f}%")

    return {
        "game_detected": bool(matched_procs or gpu_util_triggered),
        "matched_processes": matched_procs,
        "gpu_util_triggered": gpu_util_triggered,
        "non_worker_gpu_util_pct": non_worker_gpu_util_pct,
        "trigger_reasons": reasons,
    }


def default_worker_yield_drill(
    *,
    clock_fn: ClockFn = time.perf_counter,
    trigger_exe: str = "cs2.exe",
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a real lightweight local Python subprocess worker preemption drill (`shell=False`)."""
    bounded = min(max(0.2, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCAL_WORKER_DRILL_PROGRAM],
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
            proc.wait(timeout=0.5)
            return {
                "worker_started": False,
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": -1,
                "yield_latency_ms": None,
                "trigger_exe": trigger_exe,
                "method": "local_python_subprocess",
            }
        t0 = clock_fn()
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        released_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        rc = proc.wait(timeout=bounded)
        elapsed_ms = round(max(0.0, (clock_fn() - t0) * 1000.0), 3)
        retained_mib = -1
        if released_line.startswith("HIVEMIND_WORKER_RELEASED:"):
            raw_val = released_line.split(":", 1)[1].strip()
            if raw_val.isdigit():
                retained_mib = int(raw_val)
        clean_ok = bool(rc == 0 and proc.poll() is not None and retained_mib == 0)
        return {
            "worker_started": True,
            "worker_stopped": proc.poll() is not None,
            "clean_release": clean_ok,
            "retained_worker_vram_mib": retained_mib,
            "yield_latency_ms": elapsed_ms,
            "trigger_exe": trigger_exe,
            "method": "local_python_subprocess",
        }
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        try:
            proc.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, OSError) as wait_err:
            _ = wait_err
        return {
            "worker_started": True,
            "worker_stopped": False,
            "clean_release": False,
            "retained_worker_vram_mib": -1,
            "yield_latency_ms": round(bounded * 1000.0, 3),
            "trigger_exe": trigger_exe,
            "method": "local_python_subprocess",
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=0.5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _normalize_worker_yield_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bool):
        raise IdleServiceError("worker_yield_fn must return milliseconds float or mapping, not bool.")
    if isinstance(raw, (int, float)):
        val = float(raw)
        if not math.isfinite(val) or val < 0.0:
            return {
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": -1,
                "yield_latency_ms": None,
                "method": "invalid_yield_value",
            }
        return {
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 0,
            "yield_latency_ms": round(val, 3),
            "method": "custom_scalar_yield",
        }
    if isinstance(raw, Mapping):
        ms = _optional_finite_float(raw.get("yield_latency_ms", raw.get("elapsed_ms")))
        stopped = bool(raw.get("worker_stopped", True))
        clean = bool(raw.get("clean_release", True))
        raw_ret = raw.get("retained_worker_vram_mib", 0 if (stopped and clean) else -1)
        retained = int(raw_ret) if isinstance(raw_ret, (int, float)) and not isinstance(raw_ret, bool) else -1
        if ms is None or ms < 0.0:
            return {
                "worker_stopped": stopped,
                "clean_release": False,
                "retained_worker_vram_mib": retained,
                "yield_latency_ms": None,
                "method": str(raw.get("method") or "custom_mapping_yield"),
            }
        return {
            "worker_stopped": stopped,
            "clean_release": clean,
            "retained_worker_vram_mib": retained,
            "yield_latency_ms": round(ms, 3),
            "method": str(raw.get("method") or "custom_mapping_yield"),
        }
    raise IdleServiceError("worker_yield_fn must return a finite millisecond number or mapping.")


def verify_lora_delta_bandwidth_cap(
    *,
    transfer_mode: str = DEFAULT_TRANSFER_MODE,
    lora_params: int = DEFAULT_LORA_PARAMS,
    bytes_per_param: int = DEFAULT_LORA_BYTES_PER_PARAM,
    syncs_per_hour: float = DEFAULT_SYNCS_PER_HOUR,
    upload_cap_mb_per_hour: float = DEFAULT_UPLOAD_CAP_MB_PER_HOUR,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify LoRA-delta-only upload stays `< 50.0 MB/h` vs legacy full-checkpoint transfers."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    mode = str(transfer_mode or "").strip().lower()
    if lora_params <= 0 or bytes_per_param <= 0 or syncs_per_hour <= 0 or upload_cap_mb_per_hour <= 0:
        raise IdleServiceError("Bandwidth calculation parameters must be positive.")

    lora_delta_mb = round((int(lora_params) * int(bytes_per_param)) / 1_000_000.0, 3)
    per_sync_mb = lora_delta_mb if mode == DEFAULT_TRANSFER_MODE else LEGACY_FULL_CHECKPOINT_MB
    estimated_mb_per_h = round(per_sync_mb * float(syncs_per_hour), 3)
    legacy_mb_per_h = round(LEGACY_FULL_CHECKPOINT_MB * float(syncs_per_hour), 1)
    savings_ratio = round(LEGACY_FULL_CHECKPOINT_MB / lora_delta_mb, 1)
    full_ckpt_blocked = mode == DEFAULT_TRANSFER_MODE
    cap_ok = bool(
        full_ckpt_blocked
        and upload_cap_mb_per_hour <= DEFAULT_UPLOAD_CAP_MB_PER_HOUR
        and estimated_mb_per_h < float(upload_cap_mb_per_hour)
    )
    summary = (
        (
            f"Mode `{DEFAULT_TRANSFER_MODE}` vérifié : {estimated_mb_per_h:.2f} MB/h < {upload_cap_mb_per_hour:.1f} MB/h."
            if is_fr
            else f"`{DEFAULT_TRANSFER_MODE}` verified: {estimated_mb_per_h:.2f} MB/h < {upload_cap_mb_per_hour:.1f} MB/h cap."
        )
        if cap_ok
        else (
            f"Plafond réseau dépassé ({mode}: {estimated_mb_per_h:.2f} MB/h >= {upload_cap_mb_per_hour:.1f} MB/h)."
            if is_fr
            else f"Bandwidth cap violated ({mode}: {estimated_mb_per_h:.2f} MB/h >= {upload_cap_mb_per_hour:.1f} MB/h)."
        )
    )
    return {
        "bandwidth_cap_ok": cap_ok,
        "transfer_mode": mode,
        "full_checkpoint_upload_blocked": full_ckpt_blocked,
        "estimated_lora_delta_mb": lora_delta_mb,
        "estimated_upload_mb_per_hour": estimated_mb_per_h,
        "upload_cap_mb_per_hour": round(float(upload_cap_mb_per_hour), 1),
        "legacy_upload_mb_per_hour": legacy_mb_per_h,
        "bandwidth_savings_ratio": savings_ratio,
        "summary": summary,
    }


def _initial_idle_resume_state() -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "state": STATE_IDLE_READY,
        "worker_active": False,
        "tick_count": 0,
        "last_transition_epoch": None,
        "last_game_detected_epoch": None,
        "cooldown_until_epoch": None,
        "last_yield_latency_ms": None,
        "last_retained_vram_mib": 0,
        "had_game_lag": False,
        "last_gpu_status": None,
        "history": [],
    }


def _load_idle_resume_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return _initial_idle_resume_state()
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _initial_idle_resume_state()
    if not isinstance(raw, Mapping):
        return _initial_idle_resume_state()

    st = str(raw.get("state") or STATE_IDLE_READY)
    if st not in VALID_STATES:
        st = STATE_IDLE_READY
    raw_hist = raw.get("history")
    history: list[dict[str, Any]] = []
    if isinstance(raw_hist, list):
        for item in raw_hist[-MAX_STATE_HISTORY:]:
            if isinstance(item, Mapping):
                history.append(
                    {
                        "tick": int(item.get("tick") or 0),
                        "from_state": str(item.get("from_state") or STATE_IDLE_READY)[:32],
                        "to_state": str(item.get("to_state") or st)[:32],
                        "reason": str(item.get("reason") or "")[:64],
                        "epoch": _optional_finite_float(item.get("epoch")),
                    }
                )
    raw_tick = raw.get("tick_count", 0)
    tick_count = max(0, int(raw_tick)) if not isinstance(raw_tick, bool) and isinstance(raw_tick, int) else 0
    raw_ret = raw.get("last_retained_vram_mib", 0)
    retained = int(raw_ret) if not isinstance(raw_ret, bool) and isinstance(raw_ret, int) else 0
    return {
        "schema": STATE_SCHEMA,
        "state": st,
        "worker_active": bool(raw.get("worker_active", False)),
        "tick_count": tick_count,
        "last_transition_epoch": _optional_finite_float(raw.get("last_transition_epoch")),
        "last_game_detected_epoch": _optional_finite_float(raw.get("last_game_detected_epoch")),
        "cooldown_until_epoch": _optional_finite_float(raw.get("cooldown_until_epoch")),
        "last_yield_latency_ms": _optional_finite_float(raw.get("last_yield_latency_ms")),
        "last_retained_vram_mib": retained,
        "had_game_lag": bool(raw.get("had_game_lag", False)),
        "last_gpu_status": str(raw.get("last_gpu_status"))[:64] if raw.get("last_gpu_status") else None,
        "history": history,
    }


def _save_idle_resume_state(
    state_path: Path,
    state_doc: Mapping[str, Any],
    *,
    os_name: str | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    serialized = _assert_safe_output(json.dumps(dict(state_doc), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return atomic_write_config_file(
        state_path,
        serialized,
        posix_mode=0o600,
        create_backup=False,
        os_name=os_name,
        home=home,
    )


def evaluate_idle_resume_cycle(
    *,
    previous_state: Mapping[str, Any] | None = None,
    idle_probe_fn: IdleProbeFn = default_idle_probe,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: ClockFn = time.monotonic,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    transfer_mode: str = DEFAULT_TRANSFER_MODE,
    allow_lora_worker: bool = True,
    run_yield_drill: bool = False,
    had_game_lag: bool = False,
    extra_games: Sequence[str] = (),
    os_name: str | None = None,
    sys_platform: str | None = None,
    env: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Evaluate one deterministic idle-resume state transition cycle."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise IdleServiceError("max_temp_c must be finite in [25.0, 95.0] C.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise IdleServiceError("min_free_vram_mib must be >= 256 MiB.")
    if isinstance(idle_threshold_s, bool) or not math.isfinite(float(idle_threshold_s)) or float(idle_threshold_s) <= 0.0:
        raise IdleServiceError("idle_threshold_s must be positive and finite.")
    if isinstance(cooldown_s, bool) or not math.isfinite(float(cooldown_s)) or float(cooldown_s) < MIN_REQUIRED_COOLDOWN_S:
        raise IdleServiceError(f"cooldown_s must be finite and >= {MIN_REQUIRED_COOLDOWN_S:.1f} s.")
    if isinstance(yield_budget_ms, bool) or not math.isfinite(float(yield_budget_ms)) or float(yield_budget_ms) <= 0.0:
        raise IdleServiceError("yield_budget_ms must be positive and finite.")

    prev = dict(_initial_idle_resume_state())
    if previous_state is not None:
        prev.update(dict(previous_state))
    prev_state_name = str(prev.get("state") or STATE_IDLE_READY)
    prev_worker_active = bool(prev.get("worker_active", False))
    prev_cooldown_until = _optional_finite_float(prev.get("cooldown_until_epoch"))
    prev_last_game_epoch = _optional_finite_float(prev.get("last_game_detected_epoch"))
    now = float(clock_fn())

    raw_idle = _invoke_callable_flexible(
        idle_probe_fn,
        sys_platform=sys_platform,
        env=env,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    idle_f = _optional_finite_float(raw_idle)
    idle_seconds = round(idle_f, 3) if (idle_f is not None and idle_f >= 0.0) else 0.0

    gpu = _normalize_gpu_snapshot(
        _invoke_callable_flexible(gpu_probe_fn, timeout_s=SUBPROCESS_TIMEOUT_S)
    )
    processes = list(
        _invoke_callable_flexible(
            process_probe_fn,
            os_name=os_name,
            timeout_s=SUBPROCESS_TIMEOUT_S,
        )
    )

    non_worker_util = (
        gpu.non_worker_util_pct
        if gpu.non_worker_util_pct is not None
        else (gpu.util_pct if (gpu.present and not gpu.parse_error) else None)
    )
    game_detect = detect_steam_or_dx12_game_activity(
        processes,
        non_worker_gpu_util_pct=non_worker_util,
        extra_games=extra_games,
    )
    bw_cap = verify_lora_delta_bandwidth_cap(transfer_mode=transfer_mode, lang=lang_code)

    cap_c = round(float(max_temp_c), 1)
    reserve_mib = int(min_free_vram_mib)
    slo_ms = round(float(yield_budget_ms), 1)
    cool_val_s = round(float(cooldown_s), 1)

    cooldown_until = prev_cooldown_until
    last_game_epoch = prev_last_game_epoch
    if game_detect["game_detected"]:
        cooldown_until = round(now + cool_val_s, 3)
        last_game_epoch = round(now, 3)
    elif cooldown_until is not None and now >= cooldown_until:
        cooldown_until = None

    cooldown_active = cooldown_until is not None and now < cooldown_until
    cooldown_remaining_s = (
        round(min(cool_val_s, max(0.0, cooldown_until - now)), 3)
        if cooldown_active and cooldown_until is not None
        else 0.0
    )

    thermal_ok = True
    vram_reserve_ok = True
    gpu_status = "dev_api_only_no_nvidia_gpu"
    if gpu.present:
        if gpu.parse_error or gpu.temp_c is None or gpu.free_vram_mib is None:
            thermal_ok = False
            vram_reserve_ok = False
            gpu_status = "gpu_telemetry_parse_error"
        else:
            thermal_ok = float(gpu.temp_c) <= cap_c
            vram_reserve_ok = int(gpu.free_vram_mib) >= reserve_mib
            if not thermal_ok and not vram_reserve_ok:
                gpu_status = "thermal_and_vram_clamped_paused"
            elif not thermal_ok:
                gpu_status = "thermal_clamped_paused"
            elif not vram_reserve_ok:
                gpu_status = "vram_reserve_hold"
            else:
                gpu_status = "whisper_ready"

    need_preempt = bool(
        prev_worker_active
        and (
            game_detect["game_detected"]
            or cooldown_active
            or not gpu.present
            or not thermal_ok
            or not vram_reserve_ok
            or idle_seconds < float(idle_threshold_s)
        )
    )
    yield_info: dict[str, Any] = {
        "executed": False,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 0.0,
        "method": "idle_no_active_worker",
    }
    if need_preempt or run_yield_drill:
        trig = (
            game_detect["matched_processes"][0]
            if game_detect["matched_processes"]
            else "cs2.exe"
        )
        raw_yield = _invoke_callable_flexible(
            worker_yield_fn,
            clock_fn=clock_fn,
            trigger_exe=trig,
            timeout_s=SUBPROCESS_TIMEOUT_S,
        )
        norm_y = _normalize_worker_yield_result(raw_yield)
        yield_info = {"executed": True, **norm_y}

    y_ms = yield_info["yield_latency_ms"]
    yield_slo_ok = bool(
        yield_info["worker_stopped"]
        and yield_info["clean_release"]
        and int(yield_info["retained_worker_vram_mib"]) == 0
        and y_ms is not None
        and float(y_ms) < slo_ms
    )

    if game_detect["game_detected"] or cooldown_active:
        next_state = STATE_PAUSED_GAME_YIELD
        worker_active = False
        reason = (
            "active_game_yield"
            if game_detect["game_detected"]
            else "post_game_cooldown_hold"
        )
    elif not gpu.present:
        next_state = STATE_DEV_API_ONLY_NO_GPU
        worker_active = False
        reason = "dev_api_only_no_nvidia_gpu"
    elif not thermal_ok:
        next_state = STATE_PAUSED_THERMAL_CAP
        worker_active = False
        reason = "gpu_telemetry_parse_error" if gpu.parse_error else "thermal_ceiling_exceeded"
    elif not vram_reserve_ok:
        next_state = STATE_PAUSED_LOW_VRAM
        worker_active = False
        reason = "free_vram_below_reserve"
    elif idle_seconds < float(idle_threshold_s):
        next_state = STATE_PAUSED_USER_ACTIVE
        worker_active = False
        reason = "user_input_active"
    elif not bw_cap["bandwidth_cap_ok"]:
        next_state = STATE_PAUSED_USER_ACTIVE
        worker_active = False
        reason = "lora_bandwidth_cap_violation"
    elif allow_lora_worker:
        next_state = STATE_RUNNING_LORA_ONLY
        worker_active = True
        reason = "idle_whisper_lora_active"
    else:
        next_state = STATE_IDLE_READY
        worker_active = False
        reason = "idle_ready"

    zero_game_lag = bool(yield_slo_ok and thermal_ok and (not had_game_lag))
    ok = bool(yield_slo_ok and bw_cap["bandwidth_cap_ok"])
    exit_code = 0 if ok else 2

    tick_count = int(prev.get("tick_count", 0)) + 1
    history = list(prev.get("history") or [])
    if next_state != prev_state_name or not history:
        history.append(
            {
                "tick": tick_count,
                "from_state": prev_state_name,
                "to_state": next_state,
                "reason": reason,
                "epoch": round(now, 3),
            }
        )
    history = history[-MAX_STATE_HISTORY:]

    persisted_state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "state": next_state,
        "worker_active": worker_active,
        "tick_count": tick_count,
        "last_transition_epoch": round(now, 3),
        "last_game_detected_epoch": last_game_epoch,
        "cooldown_until_epoch": cooldown_until,
        "last_yield_latency_ms": y_ms,
        "last_retained_vram_mib": int(yield_info["retained_worker_vram_mib"]),
        "had_game_lag": not zero_game_lag,
        "last_gpu_status": gpu_status,
        "history": history,
    }

    if not yield_slo_ok:
        summary = (
            f"Échec critique Steam yield ({y_ms} milliseconds, retenu={yield_info['retained_worker_vram_mib']} MiB)."
            if is_fr
            else f"Steam yield SLO failed closed ({y_ms} milliseconds, retained={yield_info['retained_worker_vram_mib']} MiB)."
        )
    elif not bw_cap["bandwidth_cap_ok"]:
        summary = bw_cap["summary"]
    else:
        summary = (
            f"Cycle idle-resume : {prev_state_name} -> {next_state} ({reason})."
            if is_fr
            else f"Idle-resume cycle: {prev_state_name} -> {next_state} ({reason})."
        )

    return {
        "ok": ok,
        "exit_code": exit_code,
        "previous_state": prev_state_name,
        "state": next_state,
        "reason": reason,
        "worker_active": worker_active,
        "idle_seconds": idle_seconds,
        "idle_threshold_s": float(idle_threshold_s),
        "user_idle": idle_seconds >= float(idle_threshold_s),
        "game_detected": bool(game_detect["game_detected"]),
        "matched_games": list(game_detect["matched_processes"]),
        "cooldown_active": cooldown_active,
        "cooldown_remaining_s": cooldown_remaining_s,
        "cooldown_s": cool_val_s,
        "yield_slo_ok": yield_slo_ok,
        "yield_latency_ms": y_ms,
        "yield_budget_ms": slo_ms,
        "retained_worker_vram_mib": int(yield_info["retained_worker_vram_mib"]),
        "yield_drill": yield_info,
        "gpu_present": gpu.present,
        "gpu_parse_error": gpu.parse_error,
        "gpu_status": gpu_status,
        "gpu_name": gpu.name if (gpu.present and not gpu.parse_error) else None,
        "vram_total_mib": gpu.total_vram_mib if (gpu.present and not gpu.parse_error) else None,
        "vram_free_mib": gpu.free_vram_mib if (gpu.present and not gpu.parse_error) else None,
        "temp_c": gpu.temp_c if (gpu.present and not gpu.parse_error) else None,
        "max_temp_c": cap_c,
        "min_free_vram_mib": reserve_mib,
        "thermal_ok": thermal_ok,
        "vram_reserve_ok": vram_reserve_ok,
        "zero_game_lag": zero_game_lag,
        "bandwidth_cap": bw_cap,
        "bandwidth_cap_ok": bool(bw_cap["bandwidth_cap_ok"]),
        "persisted_state": persisted_state,
        "summary": summary,
    }


def load_saved_credentials(path: Path, *, expected_gateway: str | None = None) -> dict[str, Any] | None:
    """Load `~/.hivemind/credentials.json` safely if holding a valid `hm-...` / `hm_live_...` key."""
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
    saved_gw = str(loaded.get("gateway") or "").strip().rstrip("/")
    if expected_gateway and saved_gw and saved_gw != expected_gateway.rstrip("/"):
        return None
    return loaded


def ensure_gateway_credentials(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None = DEFAULT_HANDLE,
    role: str = DEFAULT_ROLE,
    explicit_api_key: str | None = None,
    register_if_missing: bool = False,
    http_fn: HttpFn = default_http_request,
    env: Mapping[str, str] | None = None,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> tuple[dict[str, Any], int]:
    """Load or register credentials and verify `GET /v1/hivemind/me`."""
    is_fr = lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_role = normalize_role(role)
    env_view = env if env is not None else os.environ
    net_calls = 0

    env_key = str(env_view.get("HIVEMIND_API_KEY") or "").strip()
    candidate_key = (explicit_api_key or "").strip() or env_key
    if candidate_key and not is_valid_hivemind_key(candidate_key):
        raise IdleServiceError("Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key.")

    saved = load_saved_credentials(credentials_path, expected_gateway=clean_gw)
    if not candidate_key and saved is not None:
        candidate_key = str(saved["api_key"]).strip()

    resolved_handle = (
        sanitize_handle(handle)
        if (handle and str(handle).strip())
        else str((saved or {}).get("handle") or "").strip() or sanitize_handle(None)
    )
    account_id = str((saved or {}).get("account_id") or "")
    created_account = False

    if not candidate_key:
        if not register_if_missing:
            return (
                {
                    "ok": False,
                    "stage": "credentials_missing",
                    "http_status": None,
                    "error": (
                        f"Identifiants introuvables ({redact_home_path(credentials_path, home=home)}) ; utilisez --register-if-missing."
                        if is_fr
                        else f"Credentials not found ({redact_home_path(credentials_path, home=home)}); pass --register-if-missing."
                    ),
                },
                0,
            )
        reg_status, reg_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/register",
            payload={"handle": resolved_handle, "role": norm_role},
        )
        net_calls += 1
        if reg_status == 409:
            resolved_handle = sanitize_handle(f"{resolved_handle[:38]}-{secrets.token_hex(3)}")
            reg_status, reg_body = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/register",
                payload={"handle": resolved_handle, "role": norm_role},
            )
            net_calls += 1
        issued_key = reg_body.get("api_key")
        if reg_status != 200 or not is_valid_hivemind_key(issued_key):
            return (
                {
                    "ok": False,
                    "stage": "register",
                    "http_status": reg_status,
                    "error": (
                        f"Échec POST /v1/hivemind/register (HTTP {reg_status})."
                        if is_fr
                        else f"POST /v1/hivemind/register failed (HTTP {reg_status})."
                    ),
                },
                net_calls,
            )
        candidate_key = str(issued_key).strip()
        account_id = str(reg_body.get("account_id") or account_id)
        created_account = True

    masked = mask_api_key(candidate_key)
    me_status, me_body = _invoke_http_fn(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/me",
        headers={"Authorization": f"Bearer {candidate_key}"},
    )
    net_calls += 1
    if me_status != 200:
        return (
            {
                "ok": False,
                "stage": "me",
                "http_status": me_status,
                "masked_key": masked,
                "error": (
                    f"Échec GET /v1/hivemind/me pour {masked} (HTTP {me_status})."
                    if is_fr
                    else f"GET /v1/hivemind/me failed for {masked} (HTTP {me_status})."
                ),
            },
            net_calls,
        )

    verified_handle = str(me_body.get("handle") or resolved_handle)
    verified_role = str(me_body.get("role") or norm_role).strip().lower()
    verified_account_id = str(me_body.get("account_id") or account_id)

    cred_record = {
        "account_id": verified_account_id,
        "api_key": candidate_key,
        "gateway": clean_gw,
        "handle": verified_handle,
        "role": verified_role,
    }
    cred_write = atomic_write_config_file(
        credentials_path,
        json.dumps(cred_record, indent=2, sort_keys=True) + "\n",
        posix_mode=0o600,
        create_backup=True,
        os_name=os_name,
        home=home,
    )
    return (
        {
            "ok": True,
            "api_key": candidate_key,
            "masked_key": masked,
            "account_id": verified_account_id,
            "handle": verified_handle,
            "role": verified_role,
            "created_account": created_account,
            "credentials_write": cred_write,
        },
        net_calls,
    )


def _inspect_manifest_response(
    manifest_code: int,
    manifest_body: Mapping[str, Any],
    *,
    clean_expected_sha: str | None,
) -> dict[str, Any]:
    live_build_sha = sanitize_build_sha(manifest_body.get("build_sha"))
    cloud_run_revision = sanitize_cloud_run_revision(manifest_body.get("cloud_run_revision"))
    raw_routes = manifest_body.get("holiday_routes")
    missing_routes = (
        [r for r in EXPECTED_MANIFEST_HOLIDAY_ROUTES if r not in raw_routes]
        if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes)
        else list(EXPECTED_MANIFEST_HOLIDAY_ROUTES)
    )
    meta_ok = bool(
        manifest_code == 200
        and manifest_body.get("object") == MANIFEST_OBJECT
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


def _safe_nonneg_int(val: Any, default: int = 0) -> int:
    if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
        return int(val)
    return default


def verify_live_fleet_and_badge(
    *,
    gateway: str = DEFAULT_GATEWAY,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    http_fn: HttpFn = default_http_request,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    enforce_personal_pc: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Probe `/health`, `/onboarding/manifest`, `/producers/fleet`, `/raid/status` and compute `fleet_badge`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    if isinstance(target_online_producers, bool) or not isinstance(target_online_producers, int) or target_online_producers < 1:
        raise IdleServiceError("target_online_producers must be an integer >= 1.")
    clean_expected = (
        validate_expected_sha(expected_sha)
        if (expected_sha is not None and str(expected_sha).strip())
        else None
    )

    if enforce_personal_pc:
        compliance = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if not compliance["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "fleet_check",
                    "ok": False,
                    "compliance_ok": False,
                    "exit_code": 2,
                    "gateway": clean_gw,
                    "fleet_badge": "OFFLINE",
                    "holiday_routes_deployed_on_cloud_run": False,
                    "stale_cloud_run_detected": False,
                    "online_producers": 0,
                    "active_trainers": 0,
                    "build_sha": "unknown",
                    "network_calls": 0,
                    "compliance": compliance,
                    "summary": compliance["summary"],
                }
            )

    h_code, h_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/health")
    m_code, m_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/manifest")
    f_code, f_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/producers/fleet")
    r_code, r_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/raid/status")

    health_ok = h_code == 200 and h_body.get("status") in ("ok", None)
    m_info = _inspect_manifest_response(m_code, m_body, clean_expected_sha=clean_expected)
    manifest_valid = bool(m_info["manifest_valid"])
    sha_drift = bool(m_info["sha_drift_detected"])
    raid_ok = r_code == 200 and r_body.get("object") in (RAID_STATUS_OBJECT, None)
    fleet_ok_route = f_code == 200 and f_body.get("object") in (FLEET_OBJECT, None)

    swarm = r_body.get("swarm") if isinstance(r_body.get("swarm"), Mapping) else {}
    online_producers = _safe_nonneg_int(
        f_body.get("live_hiveminders", swarm.get("live_producers", 0))
    )
    total_producers = _safe_nonneg_int(
        f_body.get("total_hiveminders", swarm.get("registered_producers", 0))
    )
    active_trainers = _safe_nonneg_int(swarm.get("active_trainers", 0))
    distinct_gpus = _safe_nonneg_int(f_body.get("distinct_physical_gpus", 0))

    stale_detected = bool(
        m_code == 404 or r_code == 404 or sha_drift or (m_code == 200 and not manifest_valid)
    )
    holiday_deployed = bool(manifest_valid and raid_ok and not stale_detected)
    reachable = bool(health_ok and manifest_valid and raid_ok and fleet_ok_route and not sha_drift)

    if stale_detected:
        badge = "STALE_CLOUD_RUN"
        fleet_ok = False
        exit_code = 2
    elif not reachable:
        badge = "OFFLINE"
        fleet_ok = False
        exit_code = 2
    elif online_producers >= target_online_producers:
        badge = "ONLINE"
        fleet_ok = True
        exit_code = 0
    else:
        badge = "BOOTSTRAPPING"
        fleet_ok = True
        exit_code = 0

    summary = (
        f"Flotte {badge} (producteurs={online_producers}/{target_online_producers}, trainers={active_trainers}, sha={m_info['live_build_sha']})."
        if is_fr
        else f"Fleet {badge} (online_producers={online_producers}/{target_online_producers}, active_trainers={active_trainers}, sha={m_info['live_build_sha']})."
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "fleet_check",
            "ok": fleet_ok,
            "compliance_ok": True,
            "exit_code": exit_code,
            "gateway": clean_gw,
            "fleet_badge": badge,
            "holiday_routes_deployed_on_cloud_run": holiday_deployed,
            "stale_cloud_run_detected": stale_detected,
            "online_producers": online_producers,
            "total_producers": total_producers,
            "distinct_physical_gpus": distinct_gpus,
            "active_trainers": active_trainers,
            "target_online_producers": target_online_producers,
            "build_sha": m_info["live_build_sha"],
            "expected_sha": clean_expected,
            "sha_matches": m_info["sha_matches"],
            "sha_drift_detected": sha_drift,
            "cloud_run_revision": m_info["cloud_run_revision"],
            "missing_holiday_routes": m_info["missing_holiday_routes"],
            "network_calls": 4,
            "zero_inference_verified": True,
            "summary": summary,
        }
    )


def build_opentrainer_heartbeat_payload(
    *,
    host_label: str | None,
    gpu_present: bool,
    gpu_parse_error: bool,
    gpu_name: str | None,
    vram_total_mib: int | None,
    vram_free_mib: int | None,
    training_stack: str = DEFAULT_TRAINING_STACK,
    active_campaign: str = DEFAULT_ACTIVE_CAMPAIGN,
    home: Path | None = None,
) -> dict[str, Any]:
    """Build payload strictly matching `OpenTrainerHeartbeatRequest` (`extra='forbid'`)."""
    clean_host = sanitize_host_label(host_label, home=home)
    reportable = bool(
        gpu_present
        and not gpu_parse_error
        and gpu_name
        and str(gpu_name).strip()
        and vram_total_mib is not None
        and int(vram_total_mib) >= MIN_REPORTABLE_VRAM_TOTAL_MIB
    )
    if reportable and vram_total_mib is not None and gpu_name is not None:
        eff_name: str | None = sanitize_text_for_display(gpu_name.strip(), home=home)[:128]
        eff_total: int | None = min(int(vram_total_mib), MAX_GPU_VRAM_MIB)
        eff_free: int | None = (
            min(max(0, int(vram_free_mib)), eff_total) if vram_free_mib is not None else None
        )
    else:
        eff_name = None
        eff_total = None
        eff_free = None

    payload = {
        "host_label": clean_host,
        "gpu_name": eff_name,
        "vram_total_mib": eff_total,
        "vram_free_mib": eff_free,
        "network_mode": DEFAULT_NETWORK_MODE,
        "training_stack": training_stack.strip()[:128] or DEFAULT_TRAINING_STACK,
        "active_campaign": active_campaign.strip()[:128] or DEFAULT_ACTIVE_CAMPAIGN,
    }
    if set(payload.keys()) != _ALLOWED_OPENTRAINER_HB_KEYS:
        raise IdleServiceError("OpenTrainer heartbeat payload keys drifted from schema.")
    return payload


def send_opentrainer_heartbeat(
    *,
    gateway: str,
    api_key: str,
    role: str,
    host_label: str | None,
    cycle_report: Mapping[str, Any],
    http_fn: HttpFn = default_http_request,
    home: Path | None = None,
    lang: str = "en",
) -> tuple[dict[str, Any], int]:
    """Dispatch outbound `POST /v1/hivemind/opentrainer/heartbeat` on active/ready ticks only."""
    is_fr = lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    state_name = str(cycle_report.get("state") or "")
    if state_name not in ACTIVE_HEARTBEAT_STATES:
        return (
            {
                "attempted": False,
                "sent": False,
                "ok": True,
                "status": "paused_state_hold",
                "state": state_name,
                "http_status": None,
                "network_mode": DEFAULT_NETWORK_MODE,
                "payload": None,
                "message": (
                    f"Heartbeat OpenTrainer suspendu dans l'état {state_name}."
                    if is_fr
                    else f"OpenTrainer heartbeat held in paused state {state_name}."
                ),
            },
            0,
        )

    if role == "consumer":
        return (
            {
                "attempted": False,
                "sent": False,
                "ok": True,
                "skipped_consumer_role": True,
                "status": "consumer_me_verified",
                "state": state_name,
                "http_status": None,
                "network_mode": DEFAULT_NETWORK_MODE,
                "payload": None,
                "message": (
                    "Rôle consumer vérifié (/me) ; heartbeat GPU OpenTrainer ignoré."
                    if is_fr
                    else "Consumer role verified (/me); OpenTrainer GPU heartbeat skipped."
                ),
            },
            0,
        )

    hb_payload = build_opentrainer_heartbeat_payload(
        host_label=host_label,
        gpu_present=bool(cycle_report.get("gpu_present")),
        gpu_parse_error=bool(cycle_report.get("gpu_parse_error")),
        gpu_name=cycle_report.get("gpu_name"),
        vram_total_mib=cycle_report.get("vram_total_mib"),
        vram_free_mib=cycle_report.get("vram_free_mib"),
        home=home,
    )
    status_code, resp_body = _invoke_http_fn(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
        payload=hb_payload,
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout_s=HTTP_TIMEOUT_S,
    )
    ok = status_code in (200, 201)
    node_obj = resp_body.get("node") if isinstance(resp_body.get("node"), Mapping) else {}
    return (
        {
            "attempted": True,
            "sent": True,
            "accepted": ok,
            "ok": ok,
            "skipped_consumer_role": False,
            "status": "accepted" if ok else f"http_{status_code}",
            "state": state_name,
            "http_status": status_code,
            "network_mode": DEFAULT_NETWORK_MODE,
            "node_id": sanitize_text_for_display(str(node_obj.get("node_id") or ""), home=home) or None,
            "payload": hb_payload,
            "message": (
                "Heartbeat OpenTrainer accepté via relais HTTPS sortant."
                if (ok and is_fr)
                else (
                    "Outbound OpenTrainer heartbeat accepted over HTTPS relay."
                    if ok
                    else (
                        f"Échec heartbeat OpenTrainer (HTTP {status_code})."
                        if is_fr
                        else f"Outbound OpenTrainer heartbeat failed (HTTP {status_code})."
                    )
                )
            ),
        },
        1,
    )


def build_ces_idempotency_key(
    *,
    gateway: str,
    account_ref: str,
    profile: str,
    ease_rating: int,
    zero_game_lag: bool,
    platform_tag: str,
) -> str:
    """Compute a deterministic zero-PII `Idempotency-Key` for `POST /v1/hivemind/onboarding/feedback`."""
    seed = (
        f"noel-idle-service-ces/v1|{gateway.rstrip('/')}|{account_ref}|"
        f"{profile}|{ease_rating}|{1 if zero_game_lag else 0}|{platform_tag}"
    )
    return "ces-idle-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _validate_ces_spool_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise IdleServiceError("CES feedback payload must be a JSON object.")
    allowed_keys = {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
    extra_keys = set(raw.keys()) - allowed_keys
    if extra_keys:
        raise IdleServiceError(f"CES feedback payload contains unsupported keys: {sorted(extra_keys)}")
    prof = normalize_profile(raw.get("profile"))
    ease = parse_ease_rating(raw.get("ease_rating"))  # type: ignore[arg-type]
    zero_lag = raw.get("zero_game_lag")
    if not isinstance(zero_lag, bool):
        raise IdleServiceError("CES feedback zero_game_lag must be a boolean.")
    plat_raw = str(raw.get("platform") or "unknown").strip().lower()
    if not _PLATFORM_RE.fullmatch(plat_raw):
        raise IdleServiceError("CES feedback platform contains unsupported characters.")
    note_raw = raw.get("note", "")
    if not isinstance(note_raw, str) or note_raw != "":
        raise IdleServiceError("CES feedback note must remain empty to guarantee zero PII.")
    return {
        "profile": prof,
        "ease_rating": ease,
        "zero_game_lag": zero_lag,
        "platform": plat_raw,
        "note": "",
    }


def _classify_ces_flush_status(status_code: int) -> tuple[str, str]:
    if 200 <= status_code < 300:
        return "delivered", "ok"
    if status_code == 404:
        return "retryable", "stale_cloud_run_404"
    if status_code == 408:
        return "retryable", "request_timeout_408"
    if status_code == 429:
        return "retryable", "rate_limited_429"
    if status_code == 0:
        return "retryable", "network_timeout"
    if status_code >= 500:
        return "retryable", f"gateway_{status_code}"
    return "permanent_quarantine", f"permanent_http_{status_code}"


def _get_ces_path_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CES_LOCK_GUARD:
        lock = _CES_PATH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _CES_PATH_LOCKS[resolved] = lock
        return lock


def submit_ces_feedback(
    *,
    gateway: str,
    account_ref: str = "anon",
    profile: str = "gamer",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_request,
    lang: str = "en",
) -> tuple[dict[str, Any], int]:
    """Send non-blocking zero-PII `POST /v1/hivemind/onboarding/feedback` (`extra='forbid'`)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_profile = normalize_profile(profile)
    clean_ease = parse_ease_rating(ease_rating)
    if not isinstance(zero_game_lag, bool):
        raise IdleServiceError("zero_game_lag must be a boolean.")
    plat = _platform_tag(system_name)
    idem_key = build_ces_idempotency_key(
        gateway=clean_gw,
        account_ref=account_ref,
        profile=norm_profile,
        ease_rating=clean_ease,
        zero_game_lag=zero_game_lag,
        platform_tag=plat,
    )
    ces_payload: dict[str, Any] = {
        "profile": norm_profile,
        "ease_rating": clean_ease,
        "zero_game_lag": zero_game_lag,
        "platform": plat,
        "note": "",
    }
    status_code, body = _invoke_http_fn(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload=ces_payload,
        headers={"Idempotency-Key": idem_key},
        timeout_s=TELEMETRY_TIMEOUT_S,
    )
    ces_recorded = status_code in (200, 201)
    _, reason = _classify_ces_flush_status(status_code)
    msg = (
        f"Retour CES enregistré (profil={norm_profile}, facilité={clean_ease}/5)."
        if (ces_recorded and is_fr)
        else (
            f"Zero-PII CES feedback recorded (profile={norm_profile}, ease_rating={clean_ease}/5)."
            if ces_recorded
            else (
                f"Retour CES différé/non enregistré (HTTP {status_code}: {reason})."
                if is_fr
                else f"CES feedback deferred/not recorded (HTTP {status_code}: {reason})."
            )
        )
    )
    raw_total = body.get("total_responses")
    raw_mean = body.get("mean_ease_rating")
    return (
        {
            "attempted": True,
            "ces_recorded": ces_recorded,
            "status": "recorded" if ces_recorded else reason,
            "http_status": status_code,
            "idempotency_key": idem_key,
            "profile": norm_profile,
            "ease_rating": clean_ease,
            "zero_game_lag": zero_game_lag,
            "total_responses": int(raw_total) if (not isinstance(raw_total, bool) and isinstance(raw_total, int)) else None,
            "mean_ease_rating": round(float(raw_mean), 2) if (not isinstance(raw_mean, bool) and isinstance(raw_mean, (int, float))) else None,
            "payload": ces_payload,
            "message": msg,
        },
        1,
    )


def parse_ces_ease_input(
    raw: str,
    *,
    default_profile: str = "gamer",
    default_zero_game_lag: bool = True,
) -> dict[str, Any]:
    """Parse 1-line zero-PII CES input (`'5'`, `'4 gamer nolag'`, or Enter/skip)."""
    norm_default_profile = normalize_profile(default_profile)
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": norm_default_profile,
            "ease_rating": None,
            "zero_game_lag": bool(default_zero_game_lag),
        }

    normalized = re.sub(r"\s*/\s*5\b", "", cleaned, count=1)
    tokens = [tok.strip(",;:").lower() for tok in normalized.split() if tok.strip(",;:")]
    rating: int | None = None
    resolved_profile = norm_default_profile
    zero_lag = bool(default_zero_game_lag)

    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            val = int(tok)
            if not (1 <= val <= 5) or rating is not None:
                raise IdleServiceError("CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("dev", "developer", "d"):
            resolved_profile = "dev"
        elif tok in ("gamer", "gaming", "g"):
            resolved_profile = "gamer"
        elif tok in ("lag", "had-lag", "had_game_lag", "stutter", "saccade"):
            zero_lag = False
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "smooth", "ok", "fluide"):
            zero_lag = bool(default_zero_game_lag)
        else:
            raise IdleServiceError(f"Unrecognized CES token {tok!r}; expected 1..5 [dev|gamer] [lag|nolag].")

    if rating is None:
        raise IdleServiceError("Interactive CES input must include an integer ease_rating from 1 to 5.")
    return {
        "skipped": False,
        "profile": resolved_profile,
        "ease_rating": rating,
        "zero_game_lag": zero_lag,
    }


class IdleServiceCesQueue:
    """Thread-safe, atomic `0600`, bounded local CES retry spool (`max_entries=128`, `max_delivered=512`)."""

    def __init__(
        self,
        queue_file: Path | str | None = None,
        *,
        max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
        max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
        home: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        os_name: str | None = None,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not (1 <= max_entries <= MAX_CES_QUEUE_ENTRIES_CEILING):
            raise IdleServiceError(f"max_entries must be an integer in 1..{MAX_CES_QUEUE_ENTRIES_CEILING}.")
        if isinstance(max_delivered, bool) or not isinstance(max_delivered, int) or not (1 <= max_delivered <= 16384):
            raise IdleServiceError("max_delivered must be an integer in 1..16384.")
        env_view = env if env is not None else os.environ
        resolved_home = _resolve_home_dir(home=home, env=env_view)
        self._home = resolved_home
        self._eff_os = os_name if os_name is not None else os.name
        self._max_entries = int(max_entries)
        self._max_delivered = int(max_delivered)

        if queue_file is not None and str(queue_file).strip():
            raw_path = _resolve_path_with_home(queue_file, home_dir=resolved_home)
        else:
            raw_path = resolved_home / DEFAULT_CES_QUEUE_REL
        self._path = raw_path.resolve()
        self._lock = _get_ces_path_lock(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def redacted_path(self) -> str:
        return redact_home_path(self._path, home=self._home)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": CES_QUEUE_SCHEMA,
            "max_entries": self._max_entries,
            "max_delivered": self._max_delivered,
            "dropped_overflow_count": 0,
            "pending": [],
            "delivered_ids": [],
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
                    "event_id": hashlib.sha256(b"corrupted_idle_service_ces_spool").hexdigest(),
                    "idempotency_key": "corrupted-spool-recovered",
                    "http_status": 0,
                    "reason": "corrupted_queue_json_recovered",
                }
            ]
            return recovered, True
        if not isinstance(parsed, Mapping):
            return self._empty_state(), True

        mutated = False
        delivered: list[str] = []
        seen_delivered: set[str] = set()
        raw_delivered = parsed.get("delivered_ids")
        if isinstance(raw_delivered, list):
            for item in raw_delivered:
                if isinstance(item, str) and item.strip() and item.strip() not in seen_delivered:
                    tok = item.strip()[:128]
                    delivered.append(tok)
                    seen_delivered.add(tok)
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
                    q_status = q_item.get("http_status")
                    quarantined.append(
                        {
                            "event_id": q_item["event_id"][:64],
                            "idempotency_key": str(q_item.get("idempotency_key") or "")[:128],
                            "http_status": int(q_status) if not isinstance(q_status, bool) and isinstance(q_status, int) else 422,
                            "reason": sanitize_text_for_display(str(q_item.get("reason") or "quarantined")[:80], home=self._home),
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
                idem_raw = str(entry.get("idempotency_key") or "").strip()[:128]
                eid_raw = entry.get("event_id")
                eid = (
                    eid_raw
                    if isinstance(eid_raw, str) and _SHA256_HEX_RE.fullmatch(eid_raw)
                    else hashlib.sha256((idem_raw or json.dumps(entry, sort_keys=True, default=str)).encode("utf-8")).hexdigest()
                )
                dedup_tok = idem_raw or eid
                if dedup_tok in seen_delivered or eid in seen_delivered or dedup_tok in seen_pending or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = _validate_ces_spool_payload(entry.get("payload"))
                except IdleServiceError:
                    mutated = True
                    quarantined.append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_raw,
                            "http_status": 422,
                            "reason": "tampered_or_invalid_local_payload",
                        }
                    )
                    quarantined = quarantined[-self._max_entries :]
                    continue
                raw_rc = entry.get("retry_count", 0)
                rc = max(0, int(raw_rc)) if not isinstance(raw_rc, bool) and isinstance(raw_rc, int) else 0
                raw_st = entry.get("last_http_status")
                last_st = int(raw_st) if not isinstance(raw_st, bool) and isinstance(raw_st, int) else None
                raw_ts = entry.get("enqueued_epoch", 0.0)
                enq_ts = round(max(0.0, float(raw_ts)), 3) if not isinstance(raw_ts, bool) and isinstance(raw_ts, (int, float)) else 0.0
                pending.append(
                    {
                        "event_id": eid,
                        "idempotency_key": idem_raw or f"ces-idle-{eid[:32]}",
                        "payload": clean_payload,
                        "enqueued_epoch": enq_ts,
                        "retry_count": rc,
                        "last_http_status": last_st,
                        "last_reason": sanitize_text_for_display(str(entry.get("last_reason") or "")[:80], home=self._home),
                    }
                )
                seen_pending.add(dedup_tok)
                seen_pending.add(eid)

        raw_dropped = parsed.get("dropped_overflow_count", 0)
        dropped_count = max(0, int(raw_dropped)) if not isinstance(raw_dropped, bool) and isinstance(raw_dropped, int) else 0
        if len(pending) > self._max_entries:
            overflow = len(pending) - self._max_entries
            pending = pending[-self._max_entries :]
            dropped_count += overflow
            mutated = True

        return (
            {
                "schema": CES_QUEUE_SCHEMA,
                "max_entries": self._max_entries,
                "max_delivered": self._max_delivered,
                "dropped_overflow_count": dropped_count,
                "pending": pending,
                "delivered_ids": delivered,
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
            "pending": list(state.get("pending", [])),
            "delivered_ids": list(state.get("delivered_ids", []))[-self._max_delivered :],
            "quarantined": list(state.get("quarantined", []))[-self._max_entries :],
        }
        serialized = _assert_safe_output(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        if self._eff_os != "nt" and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self._path.parent, 0o700)
        _write_file_secure_atomic(self._path, serialized, posix_mode=0o600, eff_os=self._eff_os)

    def snapshot(self, *, lang: str = "en") -> dict[str, Any]:
        """Return a sanitized read-only snapshot of the queue without creating any file."""
        lang_code = normalize_lang(lang)
        with self._lock:
            state, _ = self._load_state_unlocked()
            exists = self._path.is_file()
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "ces_queue_snapshot",
                "lang": lang_code,
                "queue_file": self.redacted_path,
                "queue_exists": exists,
                "max_entries": self._max_entries,
                "max_delivered": self._max_delivered,
                "pending_count": len(state["pending"]),
                "delivered_ledger_count": len(state["delivered_ids"]),
                "quarantined_count": len(state["quarantined"]),
                "dropped_overflow_count": int(state["dropped_overflow_count"]),
                "pending_idempotency_keys": [item["idempotency_key"] for item in state["pending"]],
            }
        )

    def enqueue(
        self,
        *,
        gateway: str = DEFAULT_GATEWAY,
        account_ref: str = "anon",
        profile: str = "gamer",
        ease_rating: int = 5,
        zero_game_lag: bool = True,
        system_name: str | None = None,
        platform: str | None = None,
        idempotency_key: str | None = None,
        last_http_status: int | None = None,
        last_reason: str = "",
        enforce_personal_pc: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Validate compliance & zero-PII schema, deduplicate, cap at `max_entries`, and persist `0600`."""
        lang_code = normalize_lang(lang)
        clean_gw = validate_gateway_url(gateway)
        comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "enqueue_ces",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": False,
                    "exit_code": 2,
                    "queue_file": self.redacted_path,
                    "enqueued": False,
                    "deduplicated": False,
                    "duplicate_state": "compliance_blocked",
                    "evicted_oldest": False,
                    "file_writes": 0,
                    "network_calls": 0,
                    "compliance": comp_doc,
                    "summary": comp_doc["summary"],
                }
            )

        plat = (
            platform.strip().lower()
            if (isinstance(platform, str) and platform.strip())
            else _platform_tag(system_name)
        )
        clean_payload = _validate_ces_spool_payload(
            {
                "profile": profile,
                "ease_rating": ease_rating,
                "zero_game_lag": zero_game_lag,
                "platform": plat,
                "note": "",
            }
        )
        if idempotency_key is not None:
            if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
                raise IdleServiceError("idempotency_key must be a string.")
            idem_key = idempotency_key.strip()
            if not idem_key or len(idem_key) > 128 or not idem_key.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in idem_key):
                raise IdleServiceError("idempotency_key must be 1..128 printable ASCII characters.")
        else:
            idem_key = build_ces_idempotency_key(
                gateway=clean_gw,
                account_ref=account_ref,
                profile=clean_payload["profile"],
                ease_rating=clean_payload["ease_rating"],
                zero_game_lag=clean_payload["zero_game_lag"],
                platform_tag=clean_payload["platform"],
            )
        event_id = hashlib.sha256(idem_key.encode("utf-8")).hexdigest()
        ts = round(time.time(), 3)

        with self._lock:
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if idem_key in delivered_ids or event_id in delivered_ids:
                if recovered:
                    self._atomic_save_unlocked(state)
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "mode": "enqueue_ces",
                        "ok": True,
                        "ready": True,
                        "compliance_ok": True,
                        "exit_code": 0,
                        "queue_file": self.redacted_path,
                        "enqueued": False,
                        "deduplicated": True,
                        "duplicate_state": "already_delivered",
                        "evicted_oldest": False,
                        "event_id": event_id,
                        "idempotency_key": idem_key,
                        "payload": clean_payload,
                        "pending_count": len(pending),
                        "delivered_ledger_count": len(delivered_ids),
                        "quarantined_count": len(state["quarantined"]),
                        "dropped_overflow_count": int(state["dropped_overflow_count"]),
                        "network_calls": 0,
                        "compliance": comp_doc,
                    }
                )

            if any(item["idempotency_key"] == idem_key or item["event_id"] == event_id for item in pending):
                if recovered:
                    self._atomic_save_unlocked(state)
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "mode": "enqueue_ces",
                        "ok": True,
                        "ready": True,
                        "compliance_ok": True,
                        "exit_code": 0,
                        "queue_file": self.redacted_path,
                        "enqueued": False,
                        "deduplicated": True,
                        "duplicate_state": "already_pending",
                        "evicted_oldest": False,
                        "event_id": event_id,
                        "idempotency_key": idem_key,
                        "payload": clean_payload,
                        "pending_count": len(pending),
                        "delivered_ledger_count": len(delivered_ids),
                        "quarantined_count": len(state["quarantined"]),
                        "dropped_overflow_count": int(state["dropped_overflow_count"]),
                        "network_calls": 0,
                        "compliance": comp_doc,
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
                    "idempotency_key": idem_key,
                    "payload": clean_payload,
                    "enqueued_epoch": ts,
                    "retry_count": 0,
                    "last_http_status": last_http_status,
                    "last_reason": sanitize_text_for_display(str(last_reason or "")[:80], home=self._home),
                }
            )
            self._atomic_save_unlocked(state)
            pending_count = len(pending)
            delivered_count = len(delivered_ids)
            quarantined_count = len(state["quarantined"])
            dropped_count = int(state["dropped_overflow_count"])

        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "enqueue_ces",
                "ok": True,
                "ready": True,
                "compliance_ok": True,
                "exit_code": 0,
                "queue_file": self.redacted_path,
                "enqueued": True,
                "deduplicated": False,
                "duplicate_state": "none",
                "evicted_oldest": evicted_oldest,
                "event_id": event_id,
                "idempotency_key": idem_key,
                "payload": clean_payload,
                "pending_count": pending_count,
                "delivered_ledger_count": delivered_count,
                "quarantined_count": quarantined_count,
                "dropped_overflow_count": dropped_count,
                "network_calls": 0,
                "compliance": comp_doc,
            }
        )

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        http_fn: HttpFn = default_http_request,
        enforce_personal_pc: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Idempotently flush pending CES events with `Idempotency-Key` headers (no file if empty)."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        clean_gw = validate_gateway_url(gateway)
        comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "flush_ces_queue",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": False,
                    "exit_code": 2,
                    "gateway": clean_gw,
                    "queue_file": self.redacted_path,
                    "queue_exists": self._path.is_file(),
                    "flush_ok": False,
                    "delivered_count": 0,
                    "retained_retry_count": 0,
                    "quarantined_count": 0,
                    "pending_count": 0,
                    "delivered_ledger_count": 0,
                    "total_quarantined_count": 0,
                    "stale_cloud_run_detected": False,
                    "file_writes": 0,
                    "network_calls": 0,
                    "zero_inference_verified": True,
                    "outcomes": [],
                    "compliance": comp_doc,
                    "summary": comp_doc["summary"],
                }
            )

        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
        delivered_now = 0
        retained_retry = 0
        quarantined_now = 0
        network_calls = 0
        stale_404_seen = False
        outcomes: list[dict[str, Any]] = []

        with self._lock:
            if not self._path.is_file():
                summary_empty = (
                    "File CES vide (aucun fichier créé, 0 appel réseau)."
                    if is_fr
                    else "CES retry queue empty (no queue file created, 0 network calls)."
                )
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "mode": "flush_ces_queue",
                        "ok": True,
                        "ready": True,
                        "compliance_ok": True,
                        "exit_code": 0,
                        "gateway": clean_gw,
                        "queue_file": self.redacted_path,
                        "queue_exists": False,
                        "flush_ok": True,
                        "delivered_count": 0,
                        "retained_retry_count": 0,
                        "quarantined_count": 0,
                        "pending_count": 0,
                        "delivered_ledger_count": 0,
                        "total_quarantined_count": 0,
                        "stale_cloud_run_detected": False,
                        "file_writes": 0,
                        "network_calls": 0,
                        "zero_inference_verified": True,
                        "outcomes": [],
                        "compliance": comp_doc,
                        "summary": summary_empty,
                    }
                )

            state, mutated = self._load_state_unlocked()
            if mutated:
                quarantined_now += len(state["quarantined"])
            delivered_set = set(state["delivered_ids"])
            next_pending: list[dict[str, Any]] = []

            for entry in state["pending"]:
                eid = str(entry["event_id"])
                idem_key = str(entry["idempotency_key"])
                if idem_key in delivered_set or eid in delivered_set:
                    mutated = True
                    continue

                code, _ = _invoke_http_fn(
                    http_fn,
                    "POST",
                    post_url,
                    payload=dict(entry["payload"]),
                    headers={"Idempotency-Key": idem_key},
                    timeout_s=TELEMETRY_TIMEOUT_S,
                )
                network_calls += 1
                disposition, reason = _classify_ces_flush_status(code)
                mutated = True

                if disposition == "delivered":
                    state["delivered_ids"].append(idem_key)
                    delivered_set.add(idem_key)
                    delivered_set.add(eid)
                    delivered_now += 1
                    outcomes.append(
                        {"event_id": eid, "idempotency_key": idem_key, "disposition": "delivered", "http_status": code}
                    )
                elif disposition == "retryable":
                    if code == 404:
                        stale_404_seen = True
                    new_rc = int(entry.get("retry_count", 0)) + 1
                    next_pending.append(
                        {**entry, "retry_count": new_rc, "last_http_status": code, "last_reason": reason}
                    )
                    retained_retry += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_key,
                            "disposition": "retained_for_retry",
                            "http_status": code,
                            "reason": reason,
                        }
                    )
                else:
                    state["quarantined"].append(
                        {"event_id": eid, "idempotency_key": idem_key, "http_status": code, "reason": reason}
                    )
                    state["quarantined"] = state["quarantined"][-self._max_entries :]
                    quarantined_now += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_key,
                            "disposition": "quarantined",
                            "http_status": code,
                            "reason": reason,
                        }
                    )

            state["pending"] = next_pending
            state["delivered_ids"] = state["delivered_ids"][-self._max_delivered :]
            if mutated:
                self._atomic_save_unlocked(state)

            pending_after = len(state["pending"])
            delivered_total = len(state["delivered_ids"])
            quarantined_total = len(state["quarantined"])

        flush_ok = retained_retry == 0 and quarantined_now == 0
        summary = (
            f"Vidage file CES : {delivered_now} envoyé(s), {retained_retry} en attente, {quarantined_now} en quarantaine."
            if is_fr
            else f"CES queue flush: {delivered_now} delivered, {retained_retry} retained, {quarantined_now} quarantined."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "flush_ces_queue",
                "ok": flush_ok,
                "ready": flush_ok,
                "compliance_ok": True,
                "exit_code": 0 if flush_ok else 2,
                "gateway": clean_gw,
                "queue_file": self.redacted_path,
                "queue_exists": self._path.is_file(),
                "flush_ok": flush_ok,
                "delivered_count": delivered_now,
                "retained_retry_count": retained_retry,
                "quarantined_count": quarantined_now,
                "pending_count": pending_after,
                "delivered_ledger_count": delivered_total,
                "total_quarantined_count": quarantined_total,
                "stale_cloud_run_detected": stale_404_seen,
                "file_writes": 1 if mutated else 0,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "outcomes": outcomes,
                "compliance": comp_doc,
                "summary": summary,
            }
        )

    def purge(self, *, lang: str = "en") -> dict[str, Any]:
        """Remove local CES retry spool and temporary files (0 HTTP calls)."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        removed = False
        with self._lock:
            if self._path.is_file():
                self._path.unlink(missing_ok=True)
                removed = True
            if self._path.parent.is_dir():
                for extra in self._path.parent.glob(f".{self._path.name}.*"):
                    if extra.is_file():
                        extra.unlink(missing_ok=True)
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "purge_ces_queue",
                "ok": True,
                "ready": True,
                "exit_code": 0,
                "queue_file": self.redacted_path,
                "purged": removed,
                "network_calls": 0,
                "summary": (
                    f"File CES purgée ({self.redacted_path})."
                    if is_fr
                    else f"CES retry queue purged ({self.redacted_path})."
                ),
            }
        )


def run_idle_service_tick(
    *,
    gateway: str = DEFAULT_GATEWAY,
    credentials_path: Path | str | None = None,
    state_path: Path | str | None = None,
    ces_queue_file: Path | str | None = None,
    handle: str | None = DEFAULT_HANDLE,
    role: str = DEFAULT_ROLE,
    explicit_api_key: str | None = None,
    register_if_missing: bool = False,
    host_label: str | None = None,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    transfer_mode: str = DEFAULT_TRANSFER_MODE,
    allow_lora_worker: bool = True,
    run_yield_drill: bool = False,
    send_heartbeat: bool = False,
    check_fleet: bool = False,
    require_holiday_routes: bool = False,
    submit_ces: bool = False,
    prompt_ces: bool = False,
    flush_ces_queue: bool = False,
    profile: str = "gamer",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    extra_games: Sequence[str] = (),
    idle_probe_fn: IdleProbeFn = default_idle_probe,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: ClockFn = time.monotonic,
    http_fn: HttpFn = default_http_request,
    input_fn: InputFn | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    sys_platform: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run one deterministic idle-resume tick with bounded `0600` state, heartbeat, fleet badge, and CES."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_profile = normalize_profile(profile)
    norm_role = normalize_role(role)
    clean_ease = parse_ease_rating(ease_rating)

    env_view = env if env is not None else os.environ
    eff_os = os_name if os_name is not None else os.name
    resolved_home = _resolve_home_dir(home=home, env=env_view)
    ws_path = Path(cwd) if cwd is not None else Path.cwd()
    cred_target = (
        _resolve_path_with_home(credentials_path, home_dir=resolved_home)
        if credentials_path is not None
        else (resolved_home / DEFAULT_CREDENTIALS_REL)
    )
    state_target = (
        _resolve_path_with_home(state_path, home_dir=resolved_home)
        if state_path is not None
        else (resolved_home / DEFAULT_STATE_REL)
    )
    queue_target = (
        _resolve_path_with_home(ces_queue_file, home_dir=resolved_home)
        if ces_queue_file is not None
        else (resolved_home / DEFAULT_CES_QUEUE_REL)
    )

    compliance = evaluate_personal_pc_compliance(env=env_view, cwd=ws_path, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "mode": "idle_service_tick",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "verdict": "blocked_corporate_environment",
                "gateway": clean_gw,
                "state": STATE_PAUSED_USER_ACTIVE,
                "previous_state": STATE_IDLE_READY,
                "yield_slo_ok": False,
                "zero_game_lag": False,
                "thermal_ok": False,
                "vram_reserve_ok": False,
                "bandwidth_cap_ok": False,
                "file_writes": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "fleet_badge": "OFFLINE",
                "heartbeat": {"attempted": False, "sent": False, "ok": False},
                "ces_recorded": False,
                "ces_queued_for_retry": False,
                "ces_queue_exists": queue_target.is_file(),
                "zero_inference_verified": True,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    prev_state = _load_idle_resume_state(state_target)
    cycle = evaluate_idle_resume_cycle(
        previous_state=prev_state,
        idle_probe_fn=idle_probe_fn,
        gpu_probe_fn=gpu_probe_fn,
        process_probe_fn=process_probe_fn,
        worker_yield_fn=worker_yield_fn,
        clock_fn=clock_fn,
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        idle_threshold_s=idle_threshold_s,
        cooldown_s=cooldown_s,
        yield_budget_ms=yield_budget_ms,
        transfer_mode=transfer_mode,
        allow_lora_worker=allow_lora_worker,
        run_yield_drill=run_yield_drill,
        had_game_lag=had_game_lag,
        extra_games=extra_games,
        os_name=eff_os,
        sys_platform=sys_platform,
        env=env_view,
        lang=lang_code,
    )
    state_write = _save_idle_resume_state(
        state_target,
        cycle["persisted_state"],
        os_name=eff_os,
        home=resolved_home,
    )
    file_writes = 1 if state_write.get("written") else 0
    network_calls = 0

    acct_info: dict[str, Any] | None = None
    if send_heartbeat or register_if_missing:
        acct_res, acct_calls = ensure_gateway_credentials(
            gateway=clean_gw,
            credentials_path=cred_target,
            handle=handle,
            role=norm_role,
            explicit_api_key=explicit_api_key,
            register_if_missing=register_if_missing,
            http_fn=http_fn,
            env=env_view,
            os_name=eff_os,
            home=resolved_home,
            lang=lang_code,
        )
        network_calls += acct_calls
        if not acct_res["ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "mode": "idle_service_tick",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": True,
                    "exit_code": 2,
                    "verdict": str(acct_res["stage"]),
                    "gateway": clean_gw,
                    "state": cycle["state"],
                    "previous_state": cycle["previous_state"],
                    "yield_slo_ok": cycle["yield_slo_ok"],
                    "zero_game_lag": cycle["zero_game_lag"],
                    "thermal_ok": cycle["thermal_ok"],
                    "vram_reserve_ok": cycle["vram_reserve_ok"],
                    "bandwidth_cap_ok": cycle["bandwidth_cap_ok"],
                    "file_writes": file_writes,
                    "network_calls": network_calls,
                    "fleet_badge": None,
                    "heartbeat": {"attempted": False, "sent": False, "ok": False, "status": acct_res["stage"]},
                    "ces_recorded": False,
                    "ces_queued_for_retry": False,
                    "ces_queue_exists": queue_target.is_file(),
                    "zero_inference_verified": True,
                    "compliance": compliance,
                    "summary": str(acct_res["error"]),
                }
            )
        if acct_res["credentials_write"].get("written"):
            file_writes += 1
        acct_info = acct_res

    fleet_report: dict[str, Any] | None = None
    fleet_badge: str | None = None
    stale_revision = False
    if check_fleet or require_holiday_routes or expected_sha is not None:
        fleet_report = verify_live_fleet_and_badge(
            gateway=clean_gw,
            expected_sha=expected_sha,
            target_online_producers=target_online_producers,
            http_fn=http_fn,
            env=env_view,
            cwd=ws_path,
            enforce_personal_pc=False,
            lang=lang_code,
        )
        network_calls += int(fleet_report["network_calls"])
        fleet_badge = str(fleet_report["fleet_badge"])
        stale_revision = bool(fleet_report["stale_cloud_run_detected"])

    hb_report: dict[str, Any] = {"attempted": False, "sent": False, "ok": True, "status": "not_requested"}
    if send_heartbeat and acct_info is not None:
        eff_host = host_label or str(env_view.get("HOSTNAME") or DEFAULT_HOST_LABEL)
        hb_report, hb_calls = send_opentrainer_heartbeat(
            gateway=clean_gw,
            api_key=str(acct_info["api_key"]),
            role=str(acct_info["role"]),
            host_label=eff_host,
            cycle_report=cycle,
            http_fn=http_fn,
            home=resolved_home,
            lang=lang_code,
        )
        network_calls += hb_calls

    ces_recorded = False
    ces_queued = False
    ces_report: dict[str, Any] = {"attempted": False, "ces_recorded": False, "status": "not_requested"}
    account_ref = (
        str(acct_info["account_id"] or acct_info["masked_key"])
        if acct_info is not None
        else sanitize_host_label(host_label, home=resolved_home)
    )
    eff_zero_lag = bool(cycle["zero_game_lag"])

    ces_choice: dict[str, Any] | None = None
    if prompt_ces:
        eff_in = input_fn if input_fn is not None else input
        prompt_text = (
            f"[HiveMind CES | profil={norm_profile}] Notez 1..5 ('lag' ou 'nolag'; Entrée pour ignorer) : "
            if is_fr
            else f"[HiveMind CES | profile={norm_profile}] Rate 1..5 ('lag' or 'nolag'; Enter to skip): "
        )
        try:
            raw_ans = eff_in(prompt_text)
        except (EOFError, KeyboardInterrupt):
            raw_ans = ""
        parsed = parse_ces_ease_input(
            raw_ans,
            default_profile=norm_profile,
            default_zero_game_lag=eff_zero_lag,
        )
        if parsed["skipped"]:
            ces_report = {"attempted": False, "skipped": True, "ces_recorded": False, "status": "skipped"}
        else:
            ces_choice = parsed
    elif submit_ces:
        ces_choice = {
            "skipped": False,
            "profile": norm_profile,
            "ease_rating": clean_ease,
            "zero_game_lag": eff_zero_lag,
        }

    if ces_choice is not None:
        ces_report, ces_calls = submit_ces_feedback(
            gateway=clean_gw,
            account_ref=account_ref,
            profile=str(ces_choice["profile"]),
            ease_rating=int(ces_choice["ease_rating"]),
            zero_game_lag=bool(ces_choice["zero_game_lag"]),
            system_name=system_name,
            http_fn=http_fn,
            lang=lang_code,
        )
        network_calls += ces_calls
        ces_recorded = bool(ces_report["ces_recorded"])
        http_st = int(ces_report["http_status"])
        if not ces_recorded and (http_st in (0, 404, 408, 429) or http_st >= 500):
            spool = IdleServiceCesQueue(
                queue_target, home=resolved_home, env=env_view, os_name=eff_os
            )
            enq = spool.enqueue(
                gateway=clean_gw,
                account_ref=account_ref,
                profile=str(ces_choice["profile"]),
                ease_rating=int(ces_choice["ease_rating"]),
                zero_game_lag=bool(ces_choice["zero_game_lag"]),
                system_name=system_name,
                idempotency_key=str(ces_report["idempotency_key"]),
                last_http_status=http_st,
                last_reason=str(ces_report["status"]),
                enforce_personal_pc=False,
                env=env_view,
                cwd=ws_path,
                lang=lang_code,
            )
            ces_queued = bool(enq.get("enqueued") or enq.get("deduplicated"))
            if enq.get("enqueued"):
                file_writes += 1
            if http_st == 404:
                stale_revision = True

    flush_report: dict[str, Any] | None = None
    if flush_ces_queue and not ces_queued:
        spool_fl = IdleServiceCesQueue(queue_target, home=resolved_home, env=env_view, os_name=eff_os)
        flush_report = spool_fl.flush(
            clean_gw,
            http_fn=http_fn,
            enforce_personal_pc=False,
            env=env_view,
            cwd=ws_path,
            lang=lang_code,
        )
        network_calls += int(flush_report["network_calls"])
        file_writes += int(flush_report["file_writes"])
        if flush_report["stale_cloud_run_detected"]:
            stale_revision = True

    fleet_gate_fail = bool(
        require_holiday_routes
        and (
            (fleet_report is not None and not fleet_report["holiday_routes_deployed_on_cloud_run"])
            or (ces_report.get("attempted") and not ces_recorded)
        )
    )
    hb_gate_fail = bool(send_heartbeat and not hb_report.get("ok", True))
    overall_ok = bool(cycle["ok"] and not fleet_gate_fail and not hb_gate_fail)
    exit_code = 0 if overall_ok else 2

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "mode": "idle_service_tick",
            "ok": overall_ok,
            "ready": overall_ok,
            "compliance_ok": True,
            "exit_code": exit_code,
            "verdict": "ready" if overall_ok else ("stale_cloud_run_revision" if fleet_gate_fail else "failed"),
            "gateway": clean_gw,
            "state": cycle["state"],
            "previous_state": cycle["previous_state"],
            "reason": cycle["reason"],
            "worker_active": cycle["worker_active"],
            "idle_seconds": cycle["idle_seconds"],
            "idle_threshold_s": cycle["idle_threshold_s"],
            "game_detected": cycle["game_detected"],
            "matched_games": cycle["matched_games"],
            "cooldown_active": cycle["cooldown_active"],
            "cooldown_remaining_s": cycle["cooldown_remaining_s"],
            "yield_slo_ok": cycle["yield_slo_ok"],
            "yield_latency_ms": cycle["yield_latency_ms"],
            "retained_worker_vram_mib": cycle["retained_worker_vram_mib"],
            "gpu_present": cycle["gpu_present"],
            "gpu_status": cycle["gpu_status"],
            "gpu_name": cycle["gpu_name"],
            "vram_total_mib": cycle["vram_total_mib"],
            "vram_free_mib": cycle["vram_free_mib"],
            "temp_c": cycle["temp_c"],
            "thermal_ok": cycle["thermal_ok"],
            "vram_reserve_ok": cycle["vram_reserve_ok"],
            "zero_game_lag": cycle["zero_game_lag"],
            "bandwidth_cap_ok": cycle["bandwidth_cap_ok"],
            "bandwidth_cap": cycle["bandwidth_cap"],
            "state_file": redact_home_path(state_target, home=resolved_home),
            "account": (
                {
                    "account_id": acct_info["account_id"],
                    "masked_key": acct_info["masked_key"],
                    "role": acct_info["role"],
                    "created_account": acct_info["created_account"],
                }
                if acct_info is not None
                else None
            ),
            "fleet_badge": fleet_badge,
            "fleet": fleet_report,
            "stale_revision_detected": stale_revision,
            "heartbeat": hb_report,
            "ces_recorded": ces_recorded,
            "ces_queued_for_retry": ces_queued,
            "ces_feedback": ces_report,
            "ces_flush": flush_report,
            "ces_queue_file": redact_home_path(queue_target, home=resolved_home),
            "ces_queue_exists": queue_target.is_file(),
            "file_writes": file_writes,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "compliance": compliance,
            "summary": cycle["summary"],
        }
    )


def render_cli_report(report: Mapping[str, Any], *, lang: str = "en") -> str:
    """Render a zero-secret, `~`-redacted CLI summary in English or French."""
    lang_code = normalize_lang(lang)
    status_tag = "OK" if report.get("ready", report.get("ok")) else "BLOCKED"
    lines = [
        f"[HiveMind Noel Idle-Resume Service | {lang_code.upper()} | {status_tag}]",
        f"Mode: {report.get('mode', '')}",
        f"Summary: {report.get('summary', '')}",
        f"Compliance: {COMPLIANCE_NOTICE_BILINGUAL}",
        f"Receipt SHA-256: {report.get('receipt_sha256', '')}",
    ]
    if report.get("state"):
        lines.append(
            f"State: {report['state']} | GPU: {report.get('gpu_status', 'unknown')} | "
            f"Yield: {report.get('yield_latency_ms')} milliseconds"
        )
    if report.get("fleet_badge"):
        lines.append(f"Fleet Badge: {report['fleet_badge']}")
    return _assert_safe_output("\n".join(lines) + "\n")


def render_markdown_report(report: Mapping[str, Any], *, lang: str = "en") -> str:
    """Render a zero-secret, `~`-redacted FR/EN Markdown report."""
    lang_code = normalize_lang(lang)
    heading = (
        "### Service Idle-Resume HiveMind Noël"
        if lang_code == "fr"
        else "### HiveMind Noel Idle-Resume Service"
    )
    lines = [
        heading,
        f"- **Mode:** `{report.get('mode', '')}`",
        f"- **Status:** `{'OK' if report.get('ready', report.get('ok')) else 'BLOCKED'}`",
        f"- **State:** `{report.get('state', 'n/a')}`",
        f"- **Fleet Badge:** `{report.get('fleet_badge', 'n/a')}`",
        f"- **Summary:** {report.get('summary', '')}",
        f"- **Compliance:** {COMPLIANCE_NOTICE_BILINGUAL}",
        f"- **Receipt SHA-256:** `{report.get('receipt_sha256', '')}`",
    ]
    return _assert_safe_output("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `scripts/noel_idle_resume_service.py`."""
    parser = argparse.ArgumentParser(
        prog="noel_idle_resume_service.py",
        description=(
            "Opt-in user-space idle-resume GPU service, < 1.5 s Steam yield / <= 70 C state machine, "
            "and 0600 CES retry spool (100% personal home PCs only, 0.00 EUR / 0.00 USD, AGPL-3.0)."
        ),
    )
    parser.add_argument("--platform", default="auto", choices=["auto", *VALID_PLATFORMS], help="Target OS platform.")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="HiveMind gateway URL.")
    parser.add_argument("--service-dir", default=None, help="Override service manifest target directory.")
    parser.add_argument("--credentials-path", default=None, help="Override ~/.hivemind/credentials.json.")
    parser.add_argument("--state-file", default=None, help="Override ~/.hivemind/idle_resume_state.json.")
    parser.add_argument("--ces-queue-file", default=None, help="Override ~/.hivemind/idle_service_ces_queue.json.")
    parser.add_argument("--workspace", default=None, help="Target personal workspace directory.")
    parser.add_argument("--handle", default=DEFAULT_HANDLE, help="Personal handle for registration.")
    parser.add_argument("--role", default=DEFAULT_ROLE, choices=sorted(VALID_ROLES), help="Account role.")
    parser.add_argument("--host-label", default=None, help="Sanitized personal host label.")
    parser.add_argument("--expected-sha", default=None, help="Expected Cloud Run manifest build_sha.")
    parser.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS, help="ONLINE badge target.")
    parser.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C, help="Quiet-fan ceiling in C.")
    parser.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB, help="Minimum free VRAM MiB.")
    parser.add_argument("--idle-threshold-s", type=float, default=DEFAULT_IDLE_THRESHOLD_S, help="Idle threshold seconds.")
    parser.add_argument("--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S, help="Post-game cooldown seconds (>= 300).")
    parser.add_argument("--interval-s", type=int, default=DEFAULT_INTERVAL_S, help="Scheduler polling interval seconds.")
    parser.add_argument("--yield-budget-ms", type=float, default=DEFAULT_YIELD_BUDGET_MS, help="Steam yield budget (1500.0 milliseconds).")
    parser.add_argument("--transfer-mode", default=DEFAULT_TRANSFER_MODE, choices=["lora_delta_only", "full_checkpoint"], help="Transfer mode.")
    parser.add_argument("--install", action="store_true", help="Install opt-in user-space idle-resume manifests.")
    parser.add_argument("--activate-scheduler", action="store_true", help="Register/enable installed scheduler units.")
    parser.add_argument("--uninstall", action="store_true", help="Remove HiveMind service/state/spool artifacts cleanly.")
    parser.add_argument("--tick", action="store_true", help="Run one idle-resume state machine tick.")
    parser.add_argument("--yield-drill", action="store_true", help="Run local Python subprocess < 1.5 s preemption drill.")
    parser.add_argument("--send-heartbeat", action="store_true", help="Send outbound OpenTrainer heartbeat on active ticks.")
    parser.add_argument("--check-fleet", action="store_true", help="Probe /health, /manifest, /fleet, /raid/status.")
    parser.add_argument("--register-if-missing", action="store_true", help="Register account if credentials are missing.")
    parser.add_argument("--require-holiday-routes", action="store_true", help="Fail closed if holiday routes are stale.")
    parser.add_argument("--submit-ces", action="store_true", help="Submit zero-PII CES feedback (spools on transient error).")
    parser.add_argument("--prompt-ces", action="store_true", help="Ask a 1-line zero-PII CES question interactively.")
    parser.add_argument("--flush-ces-queue", action="store_true", help="Flush pending 0600 CES retry spool entries.")
    parser.add_argument("--profile", default="gamer", choices=sorted(VALID_PROFILES), help="CES profile (dev|gamer).")
    parser.add_argument("--ease-rating", type=int, default=5, help="CES ease rating (1..5).")
    parser.add_argument("--had-game-lag", action="store_true", help="Report whether any game lag occurred.")
    parser.add_argument("--lang", default="en", choices=SUPPORTED_LANGS, help="Output language (en|fr).")
    parser.add_argument("--markdown", action="store_true", help="Render Markdown summary.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON receipt.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    sys_platform: str | None = None,
    http_fn: HttpFn = default_http_request,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    idle_probe_fn: IdleProbeFn = default_idle_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    service_runner: RunnerFn | None = None,
    clock_fn: ClockFn = time.monotonic,
    input_fn: InputFn | None = None,
    stdout: Any = None,
) -> int:
    """CLI entrypoint returning `0` on success or `2` on fail-closed compliance/SLO errors."""
    out_stream = stdout if stdout is not None else sys.stdout
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    eff_cwd = Path(args.workspace) if args.workspace else (Path(cwd) if cwd is not None else Path.cwd())

    try:
        lang_code = normalize_lang(args.lang)
        if args.uninstall:
            report = uninstall_user_space_service(
                args.platform,
                service_dir=args.service_dir,
                state_path=args.state_file,
                ces_queue_path=args.ces_queue_file,
                activate_scheduler=bool(args.activate_scheduler),
                service_runner=service_runner,
                env=env,
                home=home,
                system_name=system_name,
                lang=lang_code,
            )
        elif args.install and not (args.tick or args.yield_drill or args.send_heartbeat or args.submit_ces or args.prompt_ces or args.flush_ces_queue or args.check_fleet):
            report = install_user_space_service(
                args.platform,
                gateway=args.gateway,
                service_dir=args.service_dir,
                credentials_path=args.credentials_path,
                state_path=args.state_file,
                ces_queue_path=args.ces_queue_file,
                max_temp_c=float(args.max_temp_c),
                min_free_vram_mib=int(args.min_free_vram_mib),
                idle_threshold_s=float(args.idle_threshold_s),
                cooldown_s=float(args.cooldown_s),
                interval_s=int(args.interval_s),
                activate_scheduler=bool(args.activate_scheduler),
                service_runner=service_runner,
                env=env,
                cwd=eff_cwd,
                home=home,
                os_name=os_name,
                system_name=system_name,
                lang=lang_code,
            )
        elif args.flush_ces_queue and not (args.install or args.tick or args.yield_drill or args.send_heartbeat or args.check_fleet or args.submit_ces or args.prompt_ces):
            env_view = env if env is not None else os.environ
            resolved_home = _resolve_home_dir(home=home, env=env_view)
            queue_target = (
                _resolve_path_with_home(args.ces_queue_file, home_dir=resolved_home)
                if args.ces_queue_file is not None
                else (resolved_home / DEFAULT_CES_QUEUE_REL)
            )
            spool = IdleServiceCesQueue(queue_target, home=resolved_home, env=env_view, os_name=os_name)
            report = spool.flush(
                args.gateway,
                http_fn=http_fn,
                env=env_view,
                cwd=eff_cwd,
                lang=lang_code,
            )
        elif args.check_fleet and not (args.install or args.tick or args.yield_drill or args.send_heartbeat or args.submit_ces or args.prompt_ces or args.flush_ces_queue):
            report = verify_live_fleet_and_badge(
                gateway=args.gateway,
                expected_sha=args.expected_sha,
                target_online_producers=int(args.target_online_producers),
                http_fn=http_fn,
                env=env,
                cwd=eff_cwd,
                lang=lang_code,
            )
        else:
            install_sub: dict[str, Any] | None = None
            if args.install:
                install_sub = install_user_space_service(
                    args.platform,
                    gateway=args.gateway,
                    service_dir=args.service_dir,
                    credentials_path=args.credentials_path,
                    state_path=args.state_file,
                    ces_queue_path=args.ces_queue_file,
                    max_temp_c=float(args.max_temp_c),
                    min_free_vram_mib=int(args.min_free_vram_mib),
                    idle_threshold_s=float(args.idle_threshold_s),
                    cooldown_s=float(args.cooldown_s),
                    interval_s=int(args.interval_s),
                    activate_scheduler=bool(args.activate_scheduler),
                    service_runner=service_runner,
                    env=env,
                    cwd=eff_cwd,
                    home=home,
                    os_name=os_name,
                    system_name=system_name,
                    lang=lang_code,
                )
                if install_sub["exit_code"] != 0:
                    report = install_sub
                    install_sub = None
            if not args.install or install_sub is not None:
                report = run_idle_service_tick(
                    gateway=args.gateway,
                    credentials_path=args.credentials_path,
                    state_path=args.state_file,
                    ces_queue_file=args.ces_queue_file,
                    handle=args.handle,
                    role=args.role,
                    register_if_missing=bool(args.register_if_missing),
                    host_label=args.host_label,
                    expected_sha=args.expected_sha,
                    target_online_producers=int(args.target_online_producers),
                    max_temp_c=float(args.max_temp_c),
                    min_free_vram_mib=int(args.min_free_vram_mib),
                    idle_threshold_s=float(args.idle_threshold_s),
                    cooldown_s=float(args.cooldown_s),
                    yield_budget_ms=float(args.yield_budget_ms),
                    transfer_mode=args.transfer_mode,
                    run_yield_drill=bool(args.yield_drill),
                    send_heartbeat=bool(args.send_heartbeat),
                    check_fleet=bool(args.check_fleet),
                    require_holiday_routes=bool(args.require_holiday_routes),
                    submit_ces=bool(args.submit_ces),
                    prompt_ces=bool(args.prompt_ces),
                    flush_ces_queue=bool(args.flush_ces_queue),
                    profile=args.profile,
                    ease_rating=int(args.ease_rating),
                    had_game_lag=bool(args.had_game_lag),
                    idle_probe_fn=idle_probe_fn,
                    gpu_probe_fn=gpu_probe_fn,
                    process_probe_fn=process_probe_fn,
                    worker_yield_fn=worker_yield_fn,
                    clock_fn=clock_fn,
                    http_fn=http_fn,
                    input_fn=input_fn,
                    env=env,
                    cwd=eff_cwd,
                    home=home,
                    os_name=os_name,
                    system_name=system_name,
                    sys_platform=sys_platform,
                    lang=lang_code,
                )
                if install_sub is not None:
                    report = _attach_receipt(
                        {
                            **{k: v for k, v in report.items() if k != "receipt_sha256"},
                            "service_install": install_sub,
                            "file_writes": int(report.get("file_writes", 0)) + int(install_sub.get("file_writes", 0)),
                        }
                    )
    except IdleServiceError as exc:
        safe_err = sanitize_text_for_display(str(exc), home=Path(home) if home is not None else None)
        report = _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "error",
                "ok": False,
                "ready": False,
                "exit_code": 2,
                "error": safe_err,
                "compliance_notice": COMPLIANCE_NOTICE_BILINGUAL,
                "summary": safe_err,
            }
        )

    if args.json_output:
        rendered = _assert_safe_output(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    elif args.markdown:
        rendered = render_markdown_report(report, lang=args.lang if args.lang in SUPPORTED_LANGS else "en")
    else:
        rendered = render_cli_report(report, lang=args.lang if args.lang in SUPPORTED_LANGS else "en")
    out_stream.write(rendered)
    return int(report.get("exit_code", 2))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
